use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "Wan21";

const CHANNEL_MEANS: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const CHANNEL_STDS: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

const PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [-0.1299, -0.1692, 0.2932],
    [0.0671, 0.0406, 0.0442],
    [0.3568, 0.2548, 0.1747],
    [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328],
    [0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364],
    [-0.1293, 0.0740, 0.1636],
    [0.0680, 0.3019, 0.1128],
    [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699],
    [0.0060, -0.0633, 0.0005],
    [0.3477, 0.2275, 0.2950],
    [0.1984, 0.0913, 0.1861],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0053",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    dimensions: 3,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 4,
    scale_factor: 1.0,
    shift_factor: 0.0,
    channel_means: &CHANNEL_MEANS,
    channel_stds: &CHANNEL_STDS,
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: Some([-0.1835, -0.0868, -0.3360]),
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("lighttaew2_1"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::PerChannelAffine,
};
