use crate::{LatentFormatDefinition, LatentTensorLayout, LatentTransform, PreviewReshape};

pub const LATENT_FORMAT_IDENTIFIER: &str = "TripoSplat";

pub const LATENT_FORMAT: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-0052",
    identifier: LATENT_FORMAT_IDENTIFIER,
    channels: 16,
    // Comfy bypasses its inherited image-dimension hint for nested latents; the sampled
    // primary value is observably constructed and consumed as (batch, sequence, channels).
    dimensions: 1,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 1.0,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::SequenceChannelsLast,
    transform: LatentTransform::Identity,
};
