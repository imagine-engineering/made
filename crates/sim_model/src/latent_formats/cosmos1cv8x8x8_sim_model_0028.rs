use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "Cosmos1CV8x8x8";

const PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [0.1817, 0.2284, 0.2423],
    [-0.0586, -0.0862, -0.3108],
    [-0.4703, -0.4255, -0.3995],
    [0.0803, 0.1963, 0.1001],
    [-0.0820, -0.1050, 0.0400],
    [0.2511, 0.3098, 0.2787],
    [-0.1830, -0.2117, -0.0040],
    [-0.0621, -0.2187, -0.0939],
    [0.3619, 0.1082, 0.1455],
    [0.3164, 0.3922, 0.2575],
    [0.1152, 0.0231, -0.0462],
    [-0.1434, -0.3609, -0.3665],
    [0.0635, 0.1471, 0.1680],
    [-0.3635, -0.1963, -0.3248],
    [-0.1865, 0.0365, 0.2346],
    [0.0447, 0.0994, 0.0881],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0028",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    dimensions: 3,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 8,
    scale_factor: 1.0,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: Some([-0.1223, -0.1889, -0.1976]),
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
