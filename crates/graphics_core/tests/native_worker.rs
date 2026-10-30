use std::{error::Error, time::Duration};

use graphics_core::evaluate::{
    CpuSvgArtboardDocument, CpuSvgArtboardParameters, CpuSvgEvaluationBudget,
};
use graphite_glam::{DAffine2, DVec2, UVec2};
use graphite_graph_craft::document::{NodeId, value::RenderOutputType};
use graphite_std::{
    application_io::{ExportFormat, RenderConfig, TimingInformation},
    transform::{Footprint, RenderQuality},
    vector_types::vector::style::RenderMode,
};

#[test]
fn cold_artboard_evaluation_fits_a_native_worker_stack() -> Result<(), Box<dyn Error>> {
    // A separate executable keeps the pinned registry cold before this worker starts.
    let worker = std::thread::Builder::new()
        .name("native-artboard-cold-start".into())
        .stack_size(256 * 1024)
        .spawn(|| -> Result<(), String> {
            let configuration = RenderConfig {
                viewport: Footprint {
                    transform: DAffine2::IDENTITY,
                    resolution: UVec2::new(64, 32),
                    quality: RenderQuality::Full,
                },
                scale: 1.,
                time: TimingInformation {
                    time: 0.,
                    animation_time: Duration::ZERO,
                },
                pointer: DVec2::ZERO,
                render_mode: RenderMode::Normal,
                export_format: ExportFormat::Svg,
                for_export: true,
                for_eyedropper: false,
            };
            let parameters = CpuSvgArtboardParameters {
                dimensions: DVec2::new(64., 32.),
                ..Default::default()
            };
            let mut document = CpuSvgArtboardDocument::new(
                1,
                NodeId(7001),
                parameters,
                configuration,
                CpuSvgEvaluationBudget {
                    max_viewport_pixels: 64 * 32,
                    max_output_bytes: 64 * 1024,
                },
            )
            .map_err(|error| error.to_string())?;
            let completion = futures::executor::block_on(document.submit().evaluate());
            let accepted = document
                .accept(completion)
                .map_err(|error| error.to_string())?;
            let output = accepted.output();
            let RenderOutputType::Svg { svg, image_data } = &output.data else {
                return Err("cold worker evaluation did not return SVG".into());
            };
            assert!(svg.contains("<svg"));
            assert!(image_data.is_empty());
            assert_eq!(output.metadata.backgrounds.len(), 1);
            assert_eq!(
                output
                    .metadata
                    .backgrounds
                    .first()
                    .ok_or("missing Artboard background")?
                    .dimensions,
                parameters.dimensions,
            );
            assert_eq!(accepted.parameters(), parameters);
            Ok(())
        })?;
    worker
        .join()
        .map_err(|_| "native Artboard worker panicked")??;
    Ok(())
}
