use gpui::{Bounds, Pixels, Point, ScrollWheelEvent};
use graphite_editor::{
    consts::{VIEWPORT_SCROLL_RATE, VIEWPORT_ZOOM_MIN_FRACTION_COVER, VIEWPORT_ZOOM_WHEEL_RATE},
    messages::{
        input_mapper::utility_types::pointer::ScrollDelta,
        portfolio::document::utility_types::misc::PTZ, prelude::NavigationMessageHandler,
    },
};
use graphite_graph_craft::document::value::{DAffine2, DVec2};

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
        camera.pan += self
            .document_to_pane(viewport_size)
            .inverse()
            .transform_vector2(pane_delta);
        (camera != self.camera && camera.pan.is_finite()).then_some(camera)
    }
}

fn logical_vector(point: Point<Pixels>) -> DVec2 {
    DVec2::new(f64::from(f32::from(point.x)), f64::from(f32::from(point.y)))
}
