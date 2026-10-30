use graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION;
use serde_json::json;
use sim_runtime::{OutputCommitError, OutputProposal};
use std::fmt;

use crate::{GraphiteArtboardSession, PreparedArtboardPngAsset};

#[derive(Debug)]
pub enum ArtboardProposalError {
    SourceNotCurrent,
    PayloadMismatch,
    ExistingMetadata,
    Metadata(serde_json::Error),
    Host(OutputCommitError),
}

impl fmt::Display for ArtboardProposalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceNotCurrent => {
                formatter.write_str("The baked PNG source is no longer current.")
            }
            Self::PayloadMismatch => {
                formatter.write_str("The output proposal does not match the baked PNG.")
            }
            Self::ExistingMetadata => {
                formatter.write_str("The output proposal already has provenance metadata.")
            }
            Self::Metadata(error) => {
                write!(formatter, "Cannot encode Graphite provenance: {error}")
            }
            Self::Host(error) => write!(
                formatter,
                "Cannot prepare the host output proposal: {error}"
            ),
        }
    }
}

impl std::error::Error for ArtboardProposalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Metadata(error) => Some(error),
            Self::Host(error) => Some(error),
            Self::SourceNotCurrent | Self::PayloadMismatch | Self::ExistingMetadata => None,
        }
    }
}

#[derive(Clone)]
pub struct PreparedArtboardPngProposal {
    proposal: OutputProposal,
    prepared: PreparedArtboardPngAsset,
}

impl PreparedArtboardPngProposal {
    pub fn new(
        session: &GraphiteArtboardSession,
        prepared: PreparedArtboardPngAsset,
        proposal: OutputProposal,
    ) -> Result<Self, ArtboardProposalError> {
        if !session.png_is_current(&prepared) {
            return Err(ArtboardProposalError::SourceNotCurrent);
        }
        let png = prepared.png();
        if proposal.namespace() != sim_runtime::assets::AssetNamespace::Output
            || proposal.extension() != "png"
            || proposal.width() != png.width()
            || proposal.height() != png.height()
            || proposal.content() != png.encoded_bytes()
        {
            return Err(ArtboardProposalError::PayloadMismatch);
        }
        if !proposal.projection_metadata().is_empty() {
            return Err(ArtboardProposalError::ExistingMetadata);
        }
        let source = prepared.source_output();
        let parameters = source.parameters();
        let configuration = source.configuration();
        let quality = match configuration.viewport.quality {
            graphite_core_types::transform::RenderQuality::Preview => json!({"variant": "Preview"}),
            graphite_core_types::transform::RenderQuality::Full => json!({"variant": "Full"}),
            graphite_core_types::transform::RenderQuality::Scale(value) => {
                json!({"variant": "Scale", "f32_bits": format!("{:08x}", value.to_bits())})
            }
            graphite_core_types::transform::RenderQuality::Probability(value) => {
                json!({"variant": "Probability", "f32_bits": format!("{:08x}", value.to_bits())})
            }
        };
        let input = source.encoded_png_input().map(|input| {
            let source = input.source();
            json!({
                "asset_label": source.identity().asset_id.as_ref(),
                "version_label": source.identity().version_id.as_ref(),
                "encoded_sha256": hex(source.identity().content_sha256),
                "width": source.width(), "height": source.height(),
                "placement_f64_bits": bits(input.placement().to_cols_array()),
            })
        });
        let metadata = json!({
            "schema": "sim.graphite.artboard-png-proposal.v1",
            "qualification": "Captured evaluation provenance; labels and document encoding are not durable project versions or editable document serialization. Profile is an API contract, not embedded PNG metadata. Proposal is not a publication lease.",
            "source_revision": GRAPHITE_IMAGE_SOURCE_REVISION,
            "document_id_encoding": source.document_id_encoding().to_string(),
            "node_id": source.node_id().0.to_string(),
            "revision": source.revision().to_string(),
            "input": input,
            "artboard": {
                "location_f64_bits": bits(parameters.location.to_array()),
                "dimensions_f64_bits": bits(parameters.dimensions.to_array()),
                "background_linear_rgba_f32_bits": ([parameters.background.r(), parameters.background.g(), parameters.background.b(), parameters.background.a()].map(|value| format!("{:08x}", value.to_bits()))),
                "clip": parameters.clip,
            },
            "render": {
                "viewport_transform_f64_bits": bits(configuration.viewport.transform.to_cols_array()),
                "viewport_resolution": configuration.viewport.resolution.to_array(),
                "quality": quality,
                "scale_f64_bits": bits([configuration.scale]),
                "time_f64_bits": bits([configuration.time.time]),
                "animation_seconds": configuration.time.animation_time.as_secs().to_string(),
                "animation_nanoseconds": configuration.time.animation_time.subsec_nanos(),
                "pointer_f64_bits": bits(configuration.pointer.to_array()),
                "render_mode": serde_json::to_value(configuration.render_mode).map_err(ArtboardProposalError::Metadata)?,
                "export_format": serde_json::to_value(configuration.export_format).map_err(ArtboardProposalError::Metadata)?,
                "for_export": configuration.for_export,
                "for_eyedropper": configuration.for_eyedropper,
            },
            "output": {
                "content_type": png.content_type(), "width": png.width(), "height": png.height(),
                "encoded_bytes": png.encoded_bytes().len().to_string(),
                "encoded_sha256": hex(png.content_sha256()),
                "pixels_sha256": hex(prepared.pixels_sha256()),
                "sample_format": "rgba8", "color_space": "srgb", "alpha": "straight",
                "ownership": "immutable-cpu-bytes",
            },
        });
        let metadata = serde_json::to_vec(&metadata).map_err(ArtboardProposalError::Metadata)?;
        let proposal = proposal
            .with_projection_metadata(metadata)
            .map_err(ArtboardProposalError::Host)?;
        Ok(Self { proposal, prepared })
    }

    pub fn proposal(&self) -> &OutputProposal {
        &self.proposal
    }

    pub fn prepared(&self) -> &PreparedArtboardPngAsset {
        &self.prepared
    }

    pub fn is_current(&self, session: &GraphiteArtboardSession) -> bool {
        session.png_is_current(&self.prepared)
    }
}

fn bits<const LENGTH: usize>(values: [f64; LENGTH]) -> [String; LENGTH] {
    values.map(|value| format!("{:016x}", value.to_bits()))
}

fn hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
