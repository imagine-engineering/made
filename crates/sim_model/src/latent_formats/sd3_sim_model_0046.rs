use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SD3";

const PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [-0.0922, -0.0175, 0.0749],
    [0.0311, 0.0633, 0.0954],
    [0.1994, 0.0927, 0.0458],
    [0.0856, 0.0339, 0.0902],
    [0.0587, 0.0272, -0.0496],
    [-0.0006, 0.1104, 0.0309],
    [0.0978, 0.0306, 0.0427],
    [-0.0042, 0.1038, 0.1358],
    [-0.0194, 0.0020, 0.0669],
    [-0.0488, 0.0130, -0.0268],
    [0.0922, 0.0988, 0.0951],
    [-0.0278, 0.0524, -0.0542],
    [0.0332, 0.0456, 0.0895],
    [-0.0069, -0.0030, -0.0810],
    [-0.0596, -0.0465, -0.0293],
    [-0.1448, -0.1463, -0.1189],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0046",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 1.5305,
    shift_factor: 0.0609,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: Some([0.2394, 0.2135, 0.1925]),
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("taesd3_decoder"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
