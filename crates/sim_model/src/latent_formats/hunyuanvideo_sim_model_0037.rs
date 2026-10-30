use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "HunyuanVideo";

const PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [-0.0395, -0.0331, 0.0445],
    [0.0696, 0.0795, 0.0518],
    [0.0135, -0.0945, -0.0282],
    [0.0108, -0.0250, -0.0765],
    [-0.0209, 0.0032, 0.0224],
    [-0.0804, -0.0254, -0.0639],
    [-0.0991, 0.0271, -0.0669],
    [-0.0646, -0.0422, -0.0400],
    [-0.0696, -0.0595, -0.0894],
    [-0.0799, -0.0208, -0.0375],
    [0.1166, 0.1627, 0.0962],
    [0.1165, 0.0432, 0.0407],
    [-0.2315, -0.1920, -0.1355],
    [-0.0270, 0.0401, -0.0821],
    [-0.0616, -0.0997, -0.0727],
    [0.0249, -0.0469, -0.1703],
];

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0037",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    dimensions: 3,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 4,
    scale_factor: 0.476_986,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW_FACTORS,
    preview_bias: Some([0.0259, -0.0192, -0.0761]),
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("taehv"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};
