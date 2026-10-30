use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SC_B";

const PREVIEW_FACTORS: [[f32; 3]; 4] = [
    [0.1121, 0.2006, 0.1023],
    [-0.2093, -0.0222, -0.0195],
    [-0.3087, -0.1535, 0.0366],
    [0.0290, -0.1574, -0.4078],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0043",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 4,
    dimensions: 2,
    spatial_downscale_ratio: 4,
    temporal_downscale_ratio: 1,
    scale_factor: 1.0_f32 / 0.43_f32,
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
