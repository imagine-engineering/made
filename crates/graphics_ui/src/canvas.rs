use std::{collections::VecDeque, time::Duration};

use gpui::{
    Bounds, Modifiers, PathBuilder, PathStyle, Pixels, Point, Rgba, ScrollWheelEvent, Window,
    point, px,
};
use graphics_core::formats::gdd::{GddArtboardCanvasOverlay, GddArtboardGeometry};
use graphite_editor::{
    consts::{
        COLOR_OVERLAY_BLUE, COLOR_OVERLAY_GRAY, VIEWPORT_SCROLL_RATE,
        VIEWPORT_ZOOM_MIN_FRACTION_COVER, VIEWPORT_ZOOM_WHEEL_RATE,
    },
    messages::{
        input_mapper::utility_types::{keyboard::Key, pointer::ScrollDelta},
        portfolio::document::{DocumentMessageHandler, utility_types::misc::PTZ},
        prelude::{
            InputPreprocessorMessageHandler, Message, MessageHandler, NavigationMessageHandler,
            ViewportMessage, ViewportMessageHandler,
        },
        tool::common_functionality::auto_panning::AutoPanning,
        tool::common_functionality::resize::Resize,
        tool::common_functionality::transformation_cage::{
            BoundingBoxManager, SelectedEdges, axis_align_drag,
        },
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

    pub fn paint(
        &self,
        origin: Point<Pixels>,
        show_name: bool,
        outline_offset: Option<DVec2>,
        window: &mut Window,
    ) {
        if show_name && let Ok(color) = Rgba::try_from(COLOR_OVERLAY_GRAY) {
            for path in &self.name_paths {
                if let Some(builder) = name_path_builder(path, origin) {
                    paint_overlay_path(builder, color, window);
                }
            }
        }
        if let Some(offset) = outline_offset {
            paint_artboard_outline(self.outline.map(|corner| corner + offset), origin, window);
        }
    }
}

pub(crate) fn paint_artboard_outline(
    corners: [DVec2; 4],
    origin: Point<Pixels>,
    window: &mut Window,
) {
    if let Ok(color) = Rgba::try_from(COLOR_OVERLAY_BLUE)
        && let [Some(first), Some(second), Some(third), Some(fourth)] =
            corners.map(|corner| pane_point(corner, origin))
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

pub(crate) fn paint_artboard_snap_point(
    position: DVec2,
    origin: Point<Pixels>,
    window: &mut Window,
) {
    if let (Some(center), Ok(color)) = (
        pane_point(position, origin),
        Rgba::try_from(COLOR_OVERLAY_BLUE),
    ) {
        let bounds = Bounds::new(center - point(px(2.5), px(2.5)), gpui::size(px(5.), px(5.)));
        window.paint_quad(gpui::fill(bounds, color));
    }
}

pub(crate) struct ArtboardDraw {
    resize: Resize,
    start_pointer: DVec2,
    viewport: ViewportMessageHandler,
}

impl ArtboardDraw {
    pub fn new(position: DVec2, document: &DocumentMessageHandler, viewport_size: DVec2) -> Self {
        let viewport = artboard_viewport(viewport_size, &mut VecDeque::new());
        let input = artboard_pointer(position, Modifiers::default());
        let mut resize = Resize::default();
        resize.start(document, &input, &viewport);
        let start_pointer = document
            .metadata()
            .document_to_viewport
            .inverse()
            .transform_point2(position);
        Self {
            resize,
            start_pointer,
            viewport,
        }
    }

    pub fn placement(
        &mut self,
        position: DVec2,
        modifiers: Modifiers,
        document: &DocumentMessageHandler,
    ) -> Option<GddArtboardGeometry> {
        let to_document = document.metadata().document_to_viewport.inverse();
        if !position.is_finite()
            || to_document
                .transform_point2(position)
                .abs_diff_eq(self.start_pointer, 1e-9)
        {
            self.resize.snap_manager.clear_indicator();
            return None;
        }
        let input = artboard_pointer(position, modifiers);
        let [start, end] = self
            .resize
            .calculate_points_ignore_layer(
                document,
                &input,
                &self.viewport,
                Key::Alt,
                Key::Shift,
                true,
            )
            .map(|point| to_document.transform_point2(point));
        let location = start.min(end).round();
        let dimensions = (start.round() - end.round()).abs();
        (location.is_finite() && dimensions.is_finite() && dimensions != DVec2::ZERO).then_some(
            GddArtboardGeometry {
                location,
                dimensions,
            },
        )
    }

    pub fn snap_point(&self) -> Option<DVec2> {
        self.resize.snap_manager.indicator_pos()
    }
}

pub(crate) struct ArtboardMove {
    start: DVec2,
    geometry: GddArtboardGeometry,
    document_to_pane: DAffine2,
}

pub(crate) struct ArtboardResize {
    edges: SelectedEdges,
    start: Point<Pixels>,
    pane_origin: Point<Pixels>,
    document_to_pane: DAffine2,
}

impl ArtboardResize {
    pub fn at_edge(
        bounds: [DVec2; 2],
        start: Point<Pixels>,
        pane_origin: Point<Pixels>,
        document_to_pane: DAffine2,
    ) -> Option<Self> {
        let cage = BoundingBoxManager {
            bounds,
            transform: document_to_pane,
            ..Default::default()
        };
        let (top, bottom, left, right) =
            cage.check_selected_edges(logical_vector(start - pane_origin))?;
        Some(Self {
            edges: SelectedEdges::new(top, bottom, left, right, bounds),
            start,
            pane_origin,
            document_to_pane,
        })
    }

    pub fn placement(
        &self,
        position: Point<Pixels>,
        constrain: bool,
        centered: bool,
    ) -> Option<GddArtboardGeometry> {
        if position == self.start {
            return None;
        }
        let center = centered.then(|| {
            let [first, last] = self.edges.bounds;
            self.document_to_pane.transform_point2((first + last) / 2.)
        });
        let (first, size) = self.edges.new_size(
            logical_vector(position - self.pane_origin),
            self.document_to_pane,
            center,
            constrain,
            None,
        );
        let last = first + size;
        let location = first.min(last).round();
        let dimensions = size.abs().round();
        (location.is_finite() && dimensions.is_finite()).then_some(GddArtboardGeometry {
            location,
            dimensions,
        })
    }

    pub fn outline(&self, geometry: GddArtboardGeometry) -> [DVec2; 4] {
        artboard_outline(geometry, self.document_to_pane)
    }
}

impl ArtboardMove {
    pub fn new(
        start: Point<Pixels>,
        geometry: GddArtboardGeometry,
        document_to_pane: DAffine2,
    ) -> Self {
        Self {
            start: logical_vector(start),
            geometry,
            document_to_pane,
        }
    }

    pub fn placement(
        &self,
        position: Point<Pixels>,
        constrain: bool,
    ) -> Option<(GddArtboardGeometry, DVec2)> {
        let position =
            axis_align_drag(constrain, DVec2::ZERO, logical_vector(position), self.start);
        let delta = self
            .document_to_pane
            .inverse()
            .transform_vector2(position - self.start);
        let geometry = self.geometry.translated(delta)?;
        let offset = self
            .document_to_pane
            .transform_vector2(geometry.location - self.geometry.location);
        Some((geometry, offset))
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
pub(crate) struct DocumentNavigation {
    pub camera: PTZ,
    pub document_size: DVec2,
}

impl DocumentNavigation {
    pub fn document_to_pane(&self, viewport_size: DVec2) -> DAffine2 {
        NavigationMessageHandler::default()
            .calculate_offset_transform(viewport_size / 2., &self.camera)
    }

    pub fn pan(&self, viewport_size: DVec2, delta: Point<Pixels>) -> Option<PTZ> {
        self.pan_by(viewport_size, logical_vector(delta))
    }

    pub fn pan_by(&self, viewport_size: DVec2, delta: DVec2) -> Option<PTZ> {
        let mut camera = self.camera;
        camera.pan += self.document_delta(viewport_size, delta);
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
                let coverage_per_zoom = (self.document_size / viewport_size).max_element();
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

pub(crate) fn artboard_outline(
    geometry: GddArtboardGeometry,
    document_to_pane: DAffine2,
) -> [DVec2; 4] {
    let first = geometry.location;
    let last = first + geometry.dimensions.max(DVec2::ONE);
    [
        first,
        DVec2::new(last.x, first.y),
        last,
        DVec2::new(first.x, last.y),
    ]
    .map(|corner| document_to_pane.transform_point2(corner))
}

pub(crate) fn artboard_auto_pan_delta(
    viewport_size: DVec2,
    cursor: Point<Pixels>,
    frame_duration: Duration,
) -> Option<DVec2> {
    let mut responses = VecDeque::new();
    let viewport = artboard_viewport(viewport_size, &mut responses);
    let mut input = artboard_pointer(logical_vector(cursor), Modifiers::default());
    input.frame_time.advance_timestamp(frame_duration);
    let mut auto_pan = AutoPanning::default();
    auto_pan.setup_by_mouse_position(&input, &viewport, &[], &mut responses);
    auto_pan
        .shift_viewport(&input, &viewport, &mut responses)
        .filter(|delta| delta.is_finite() && *delta != DVec2::ZERO)
}

pub(crate) fn set_artboard_snap_camera(
    document: &mut DocumentMessageHandler,
    navigation: DocumentNavigation,
    viewport_size: DVec2,
) {
    let transform = navigation.document_to_pane(viewport_size);
    let metadata = document.metadata();
    if metadata.document_to_viewport != transform {
        // Graphite caches renderer footprints in viewport space.
        let change = transform * metadata.document_to_viewport.inverse();
        let mut footprints = metadata.upstream_footprints.clone();
        for footprint in footprints.values_mut() {
            footprint.transform = change * footprint.transform;
        }
        let local_transforms = metadata.local_transforms.clone();
        document
            .network_interface
            .update_transforms(footprints, local_transforms);
        document
            .network_interface
            .set_document_to_viewport_transform(transform);
    }
    document.document_ptz = navigation.camera;
}

fn artboard_pointer(position: DVec2, modifiers: Modifiers) -> InputPreprocessorMessageHandler {
    let mut input = InputPreprocessorMessageHandler::default();
    input.mouse.position = position;
    if modifiers.alt {
        input.keyboard.set(Key::Alt as usize);
    }
    if modifiers.shift {
        input.keyboard.set(Key::Shift as usize);
    }
    input
}

fn artboard_viewport(
    viewport_size: DVec2,
    responses: &mut VecDeque<Message>,
) -> ViewportMessageHandler {
    let mut viewport = ViewportMessageHandler::default();
    viewport.process_message(
        ViewportMessage::Update {
            x: 0.,
            y: 0.,
            width: viewport_size.x,
            height: viewport_size.y,
            scale: 1.,
        },
        responses,
        (),
    );
    viewport
}

pub(crate) fn logical_vector(point: Point<Pixels>) -> DVec2 {
    DVec2::new(f64::from(f32::from(point.x)), f64::from(f32::from(point.y)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_gra_tool_008_1_draw_respects_source_grid_snapping() {
        use graphite_editor::messages::portfolio::document::utility_types::misc::GridType;

        let mut document = DocumentMessageHandler::default();
        document.snapping_state.grid_snapping = true;
        document.snapping_state.grid.grid_type = GridType::Rectangular {
            spacing: DVec2::splat(20.),
        };
        let size = DVec2::splat(200.);
        let navigation = DocumentNavigation {
            camera: PTZ::default(),
            document_size: DVec2::ZERO,
        };
        set_artboard_snap_camera(&mut document, navigation, size);
        let start = DVec2::splat(121.);
        let end = DVec2::splat(161.);
        for (enabled, location) in [(true, 20.), (false, 21.)] {
            document.snapping_state.snapping_enabled = enabled;
            let mut draw = ArtboardDraw::new(start, &document, size);
            assert_eq!(
                draw.placement(end, Modifiers::default(), &document),
                Some(GddArtboardGeometry {
                    location: DVec2::splat(location),
                    dimensions: DVec2::splat(40.)
                }),
            );
            assert_eq!(draw.snap_point(), enabled.then_some(DVec2::splat(60.)));
        }
    }
    use graphite_std::vector_types::kurbo::Shape;

    #[test]
    fn artboard_resize_applies_source_edge_flip_aspect_and_center_rules() {
        let bounds = [DVec2::ZERO, DVec2::splat(40.)];
        let transform = DAffine2::from_scale(DVec2::splat(2.));
        let pane = point(px(50.), px(30.));
        let at = |x: f32, y: f32| pane + point(px(2. * x), px(2. * y));
        assert!(ArtboardResize::at_edge(bounds, at(20., 20.), pane, transform).is_none());
        let corner = at(40., 40.);
        let resize =
            ArtboardResize::at_edge(bounds, corner, pane, transform).expect("source corner");
        assert_eq!(resize.placement(corner, false, false), None);
        assert_eq!(
            resize.placement(at(50., 43.), true, true),
            Some(GddArtboardGeometry {
                location: DVec2::splat(-10.),
                dimensions: DVec2::splat(60.),
            }),
        );
        let resize = ArtboardResize::at_edge(bounds, at(40., 20.), pane, transform)
            .expect("source right edge");
        assert_eq!(
            resize.placement(at(-8., 20.), false, false),
            Some(GddArtboardGeometry {
                location: DVec2::new(-8., 0.),
                dimensions: DVec2::new(8., 40.),
            }),
        );
    }

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
