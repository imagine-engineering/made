use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "HunyuanImage21Refiner";

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0036",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 64,
    dimensions: 3,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 1.03682,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::HunyuanImage21Refiner,
};
