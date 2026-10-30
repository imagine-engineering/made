use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "Flux";

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0029",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 0.3611,
    shift_factor: 0.1159,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[
        [-0.0346, 0.0244, 0.0681],
        [0.0034, 0.0210, 0.0687],
        [0.0275, -0.0668, -0.0433],
        [-0.0174, 0.0160, 0.0617],
        [0.0859, 0.0721, 0.0329],
        [0.0004, 0.0383, 0.0115],
        [0.0405, 0.0861, 0.0915],
        [-0.0236, -0.0185, -0.0259],
        [-0.0245, 0.0250, 0.1180],
        [0.1008, 0.0755, -0.0421],
        [-0.0515, 0.0201, 0.0011],
        [0.0428, -0.0012, -0.0036],
        [0.0817, 0.0765, 0.0749],
        [-0.1264, -0.0522, -0.1103],
        [-0.0280, -0.0881, -0.0499],
        [-0.1262, -0.0982, -0.0778],
    ],
    preview_bias: Some([-0.0329, -0.0718, -0.0851]),
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("taef1_decoder"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
