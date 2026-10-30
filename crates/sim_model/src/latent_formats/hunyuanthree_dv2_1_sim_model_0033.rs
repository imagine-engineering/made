use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "Hunyuan3Dv2_1";

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0033",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 64,
    dimensions: 1,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 1.0039506158752403,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
