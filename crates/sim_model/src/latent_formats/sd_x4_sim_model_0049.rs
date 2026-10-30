use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SD_X4";

const PREVIEW_FACTORS: [[f32; 3]; 4] = [
    [-0.2340, -0.3863, -0.3257],
    [0.0994, 0.0885, -0.0908],
    [-0.2833, -0.2349, -0.3741],
    [0.2523, -0.0055, -0.1651],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0049",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 4,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 0.08333,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
