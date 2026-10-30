use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "SC_Prior";

const PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [-0.0326, -0.0204, -0.0127],
    [-0.1592, -0.0427, 0.0216],
    [0.0873, 0.0638, -0.0020],
    [-0.0602, 0.0442, 0.1304],
    [0.0800, -0.0313, -0.1796],
    [-0.0810, -0.0638, -0.1581],
    [0.1791, 0.1180, 0.0967],
    [0.0740, 0.1416, 0.0432],
    [-0.1745, -0.1888, -0.1373],
    [0.2412, 0.1577, 0.0928],
    [0.1908, 0.0998, 0.0682],
    [0.0209, 0.0365, -0.0092],
    [0.0448, -0.0650, -0.1728],
    [-0.1658, -0.1045, -0.1308],
    [0.0542, 0.1545, 0.1325],
    [-0.0352, -0.1672, -0.2541],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0044",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    dimensions: 2,
    spatial_downscale_ratio: 42,
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
    transform: LatentTransform::Affine,
};
