use std::sync::Arc;

use anyhow::{Context as _, Result, ensure};
use gpui::{RenderImage, swap_rgba_pa_to_bgra};
use graphite_editor::messages::portfolio::document::{
    DocumentMessageHandler,
    overlays::grid_overlays::grid_overlay,
    utility_types::misc::{GridSnapping, GridType, PTZ},
};
use graphite_graph_craft::document::value::{DVec2, UVec2};
use image::{Frame, RgbaImage};

use crate::canvas_overlay::{CanvasOverlay, OverlayBudget, source_overlay_context};

const GRID_BUDGET: OverlayBudget = OverlayBudget {
    paths: 350_000,
    segments: 1_800_000,
};

#[derive(Clone, PartialEq)]
pub(crate) struct CanvasGrid {
    pub resolution: UVec2,
    pub scale: f64,
    pub camera: PTZ,
    pub options: GridSnapping,
}

impl CanvasGrid {
    pub fn render(&self, pixel_budget: u32) -> Result<Option<Arc<RenderImage>>> {
        let size = self.resolution;
        ensure!(
            self.scale.is_finite()
                && self.scale > 0.
                && u64::from(size.x) * u64::from(size.y) <= u64::from(pixel_budget),
            "Graphite grid exceeds its viewport pixel budget",
        );
        let viewport = size.as_dvec2() / self.scale;
        let mut document = DocumentMessageHandler::default();
        document.document_ptz = self.camera;
        document.snapping_state.grid = self.options.clone();
        if !self.check_geometry(&document, viewport)? {
            return Ok(None);
        }
        let mut context =
            source_overlay_context(viewport, self.scale, document.overlays_visibility_settings)?;
        grid_overlay(&document, &mut context);
        let scene = context.take_scene();
        let overlay = CanvasOverlay::decode(scene.encoding(), 1., &GRID_BUDGET)?;
        if overlay.is_empty() {
            return Ok(None);
        }
        let size =
            tiny_skia::IntSize::from_wh(size.x, size.y).context("Invalid Graphite grid target")?;
        let byte_count = (size.width() as usize)
            .checked_mul(size.height() as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .context("Graphite grid target size overflow")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_count)
            .context("Unable to allocate Graphite grid")?;
        bytes.resize(byte_count, 0);
        let mut target =
            tiny_skia::Pixmap::from_vec(bytes, size).context("Invalid Graphite grid buffer")?;
        overlay.rasterize(&mut target)?;
        let mut bytes = target.take();
        for pixel in bytes.as_chunks_mut::<4>().0 {
            swap_rgba_pa_to_bgra(pixel);
        }
        let image = RgbaImage::from_raw(size.width(), size.height(), bytes)
            .context("Invalid Graphite grid pixel layout")?;
        Ok(Some(Arc::new(RenderImage::new(vec![Frame::new(image)]))))
    }

    fn check_geometry(&self, document: &DocumentMessageHandler, viewport: DVec2) -> Result<bool> {
        let options = &self.options;
        let transform = document
            .navigation_handler
            .calculate_offset_transform(viewport / 2., &self.camera)
            .inverse();
        ensure!(
            transform.is_finite() && options.origin.is_finite() && options.color.is_ascii(),
            "Invalid Graphite grid geometry or color",
        );
        let corners = [
            DVec2::ZERO,
            DVec2::new(viewport.x, 0.),
            DVec2::new(0., viewport.y),
            viewport,
        ]
        .map(|point| transform.transform_point2(point));
        ensure!(
            corners.iter().all(|point| point.is_finite()),
            "Invalid Graphite grid bounds"
        );
        let minimum = corners
            .into_iter()
            .fold(DVec2::splat(f64::INFINITY), DVec2::min);
        let maximum = corners
            .into_iter()
            .fold(DVec2::splat(f64::NEG_INFINITY), DVec2::max);
        let span = maximum - minimum;
        let lines = |distance: f64, spacing: f64| (distance / spacing.abs()).ceil() + 3.;
        // Estimate the largest source loop before Graphite expands the grid into vector paths.
        let (paths, segments) = match options.grid_type {
            GridType::Rectangular { spacing } => {
                let Some(spacing) = GridSnapping::compute_rectangle_spacing(spacing, &self.camera)
                else {
                    return Ok(false);
                };
                let columns = lines(span.x, spacing.x);
                let rows = lines(span.y, spacing.y);
                if options.dot_display {
                    // The source divides by the number of column intervals in this case.
                    if (maximum.x / spacing.x).floor() == (minimum.x / spacing.x).floor() {
                        return Ok(false);
                    }
                    let dots = columns * rows;
                    (dots, dots * 5.)
                } else {
                    (columns + rows, (columns + rows) * 3.)
                }
            }
            GridType::Isometric {
                y_axis_spacing,
                angle_a,
                angle_b,
            } => {
                let tangent_a = angle_a.to_radians().tan();
                let tangent_b = angle_b.to_radians().tan();
                let slope = tangent_a + tangent_b;
                ensure!(
                    slope.is_finite() && slope != 0.,
                    "Invalid Graphite isometric grid angles"
                );
                let Some(multiplier) =
                    GridSnapping::compute_isometric_multiplier(y_axis_spacing, slope, &self.camera)
                else {
                    return Ok(false);
                };
                let spacing = DVec2::new(y_axis_spacing / slope, y_axis_spacing) * multiplier;
                let columns = lines(span.x, spacing.x);
                let rows_a = lines(span.y + span.x * tangent_a.abs(), spacing.y);
                if options.dot_display {
                    let cosine_a = angle_a.to_radians().cos();
                    if cosine_a.abs() <= 0.00001 {
                        return Ok(false);
                    }
                    let dash_period = (spacing.x / cosine_a) / transform.matrix2.x_axis.length();
                    ensure!(
                        dash_period.is_finite() && dash_period > 1.,
                        "Invalid Graphite dotted grid spacing"
                    );
                    (rows_a, rows_a * columns * 4.)
                } else {
                    let count =
                        columns + rows_a + lines(span.y + span.x * tangent_b.abs(), spacing.y);
                    (count, count * 3.)
                }
            }
        };
        ensure!(
            paths <= GRID_BUDGET.paths as f64 && segments <= GRID_BUDGET.segments as f64,
            "Graphite grid exceeds its viewport geometry budget",
        );
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_grid() -> CanvasGrid {
        CanvasGrid {
            resolution: UVec2::new(100, 90),
            scale: 1.,
            camera: PTZ::default(),
            options: GridSnapping {
                grid_type: GridType::Rectangular {
                    spacing: DVec2::new(16., 20.),
                },
                origin: DVec2::new(3., 5.),
                color: "#C40A2880".to_owned(),
                ..Default::default()
            },
        }
    }

    fn sample(grid: &RenderImage, x: usize, y: usize) -> &[u8] {
        let offset = (y * grid.size(0).width.0 as usize + x) * 4;
        grid.as_bytes(0)
            .and_then(|bytes| bytes.get(offset..offset + 4))
            .expect("grid pixel in the measured viewport")
    }

    #[test]
    fn at_gra_view_006_1_renders_source_lines_and_dots_at_the_pinned_origin() {
        let mut request = source_grid();
        let lines = request
            .render(10_000)
            .expect("admitted grid")
            .expect("line image");
        assert_eq!(sample(&lines, 52, 55), [39, 9, 195, 128]);
        assert_eq!(sample(&lines, 54, 55), [0; 4]);
        request.options.dot_display = true;
        let dots = request
            .render(10_000)
            .expect("admitted grid")
            .expect("dot image");
        assert_eq!(sample(&dots, 52, 49), sample(&lines, 52, 55));
        assert_eq!(sample(&dots, 52, 55), [0; 4]);
    }

    #[test]
    fn at_gra_view_006_1_preserves_isometric_modes_and_rejects_invalid_frames() {
        let mut request = source_grid();
        request.options.grid_type = GridType::Isometric {
            y_axis_spacing: 13.,
            angle_a: 30.,
            angle_b: 45.,
        };
        for scale in [1., 1.5, 2.] {
            request.scale = scale;
            for dot_display in [false, true] {
                request.options.dot_display = dot_display;
                let image = request
                    .render(10_000)
                    .expect("admitted grid")
                    .expect("isometric image");
                let alpha = image
                    .as_bytes(0)
                    .expect("grid pixels")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|pixel| pixel[3] > 0)
                    .count();
                assert!(alpha > 0 && alpha < 9_000);
            }
        }
        assert!(request.render(1_000).is_err());
        request.options.color = "#aéabc".to_owned();
        assert!(request.render(10_000).is_err());
    }
}
