use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "Mochi";

const CHANNEL_MEANS: [f32; 12] = [
    -0.067_308_96,
    -0.038_011_383,
    -0.074_778_21,
    -0.055_652_644,
    0.012_767_231,
    -0.047_035_426,
    0.043_896_97,
    -0.093_463_056,
    -0.099_183_15,
    -0.008_729_793,
    -0.011_931_556,
    -0.032_199_338,
];

const CHANNEL_STDS: [f32; 12] = [
    0.926_379_5,
    0.924_889_45,
    0.939_305_96,
    0.959_253_7,
    0.824_456_04,
    0.917_26,
    0.929_415_46,
    1.372_094_3,
    0.881_393_7,
    0.916_831_55,
    0.918_524_9,
    0.927_475_75,
];

const PREVIEW_FACTORS: [[f32; 3]; 12] = [
    [-0.0069, -0.0045, 0.0018],
    [0.0154, -0.0692, -0.0274],
    [0.0333, 0.0019, 0.0206],
    [-0.1390, 0.0628, 0.1678],
    [-0.0725, 0.0134, -0.1898],
    [0.0074, -0.0270, -0.0209],
    [-0.0176, -0.0277, -0.0221],
    [0.5294, 0.5204, 0.3852],
    [-0.0326, -0.0446, -0.0143],
    [-0.0659, 0.0153, -0.0153],
    [0.0185, -0.0217, 0.0014],
    [-0.0396, -0.0495, -0.0281],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0041",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 12,
    dimensions: 3,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 6,
    scale_factor: 1.0,
    shift_factor: 0.0,
    channel_means: &CHANNEL_MEANS,
    channel_stds: &CHANNEL_STDS,
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: Some([-0.0940, -0.1418, -0.1453]),
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::PerChannelAffine,
};
