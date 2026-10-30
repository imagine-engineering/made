use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SDXL";

const PREVIEW_FACTORS: [[f32; 3]; 4] = [
    [0.3651, 0.4232, 0.4341],
    [-0.2533, -0.0042, 0.1068],
    [0.1076, 0.1111, -0.0362],
    [-0.3165, -0.2492, -0.2188],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0047",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 4,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 0.13025,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: Some([0.1084, -0.0175, -0.0011]),
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("taesdxl_decoder"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
