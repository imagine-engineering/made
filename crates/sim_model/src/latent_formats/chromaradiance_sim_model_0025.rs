use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "ChromaRadiance";

const PREVIEW_FACTORS: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0025",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 3,
    dimensions: 2,
    spatial_downscale_ratio: 1,
    temporal_downscale_ratio: 1,
    scale_factor: 1.0,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Identity,
};
