use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SDXL_Playground_2_5";

const CHANNEL_MEANS: [f32; 4] = [-1.6574, 1.886, -1.383, 2.5155];
const CHANNEL_STDS: [f32; 4] = [8.4927, 5.9022, 6.5498, 5.2299];
const PREVIEW_FACTORS: [[f32; 3]; 4] = [
    [0.3920, 0.4054, 0.4549],
    [-0.2634, -0.0196, 0.0653],
    [0.0568, 0.1687, -0.0755],
    [-0.3112, -0.2359, -0.2076],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0048",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 4,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 0.5,
    shift_factor: 0.0,
    channel_means: &CHANNEL_MEANS,
    channel_stds: &CHANNEL_STDS,
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("taesdxl_decoder"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::PerChannelAffine,
};
