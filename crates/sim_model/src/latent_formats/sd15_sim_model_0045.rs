use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SD15";

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0045",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 4,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 0.18215,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[
        [0.3512, 0.2297, 0.3227],
        [0.3250, 0.4974, 0.2350],
        [-0.2829, 0.1762, 0.2721],
        [-0.2120, -0.2616, -0.7177],
    ],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("taesd_decoder"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
