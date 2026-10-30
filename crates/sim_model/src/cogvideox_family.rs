use crate::{ModelFamilyError, ModelLayoutSignature, ModelProbe, ModelStateLayout};

pub const COGVIDEOX_DETECTION_MARKER_KEYS: &[&str] = &[
    "model.diffusion_model.blocks.0.norm1.linear.weight",
    "blocks.0.norm1.linear.weight",
];
pub const COGVIDEOX_PATCH_PROJECTION_KEYS: &[&str] = &[
    "model.diffusion_model.patch_embed.proj.weight",
    "patch_embed.proj.weight",
];

pub const COGVIDEOX_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.blocks.0.norm1.linear.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["blocks.0.norm1.linear.weight"],
        required_prefixes: &[],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CogVideoXLayout {
    Native,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CogVideoXLatentVariant {
    CogVideoX,
    CogVideoX1_5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CogVideoXConfiguration {
    pub layout: CogVideoXLayout,
    pub in_channels: u64,
    pub number_of_attention_heads: u64,
    pub time_embedding_dimension: u64,
    pub number_of_layers: usize,
    pub patch_size: u64,
    pub temporal_patch_size: Option<u64>,
    pub sample_height: u64,
    pub sample_width: u64,
    pub sample_frames: u64,
    pub text_embedding_dimension: Option<u64>,
    pub ofs_embedding_dimension: Option<u64>,
    pub learned_positional_embeddings: bool,
    pub latent_variant: CogVideoXLatentVariant,
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
    expected_in_channels: u64,
    family_identifier: &str,
) -> Result<CogVideoXConfiguration, ModelFamilyError> {
    let invalid_configuration = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "{family_identifier} configuration {message}"
        ))
    };
    let (layout, prefix) = match probe.select_layout(COGVIDEOX_LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (CogVideoXLayout::Native, "model.diffusion_model."),
        ModelStateLayout::Diffusers => (CogVideoXLayout::Diffusers, ""),
        ModelStateLayout::StandaloneNative => {
            return Err(invalid_configuration(
                "standalone-native layout is unsupported".to_owned(),
            ));
        }
    };
    let normalization_key = format!("{prefix}blocks.0.norm1.linear.weight");
    let normalization_shape = required_shape(probe, &normalization_key, &invalid_configuration)?;
    if normalization_shape.len() != 2 || normalization_shape[0] % 6 != 0 {
        return Err(invalid_configuration(
            "blocks.0.norm1.linear.weight shape".to_string(),
        ));
    }
    let hidden_dimension = normalization_shape[0] / 6;
    if hidden_dimension == 0 || hidden_dimension % 64 != 0 {
        return Err(invalid_configuration(
            "attention hidden dimension".to_string(),
        ));
    }
    let number_of_attention_heads = hidden_dimension / 64;
    let time_embedding_dimension = normalization_shape[1];
    if time_embedding_dimension == 0 {
        return Err(invalid_configuration(
            "time embedding dimension".to_string(),
        ));
    }
    let number_of_layers = probe.consecutive_block_count(&format!("{prefix}blocks.{{}}."))?;
    if number_of_layers == 0 {
        return Err(invalid_configuration("transformer layer count".to_string()));
    }

    let patch_key = format!("{prefix}patch_embed.proj.weight");
    let patch_shape = required_shape(probe, &patch_key, &invalid_configuration)?;
    let (in_channels, patch_size, temporal_patch_size, sample_height, sample_width, sample_frames) =
        match patch_shape {
            [_, in_channels, height, width] if height == width && *height > 0 => {
                (*in_channels, *height, None, 60, 90, 49)
            }
            [_, flattened] if *flattened > 0 && *flattened % 8 == 0 => {
                (*flattened / 8, 2, Some(2), 96, 170, 81)
            }
            _ => return Err(invalid_configuration("patch projection shape".to_string())),
        };
    if in_channels != expected_in_channels {
        return Err(invalid_configuration(format!(
            "in_channels {in_channels}; requires {expected_in_channels}"
        )));
    }

    let text_embedding_dimension = optional_dimension(
        probe,
        &format!("{prefix}patch_embed.text_proj.weight"),
        1,
        &invalid_configuration,
    )?;
    let ofs_embedding_dimension = optional_dimension(
        probe,
        &format!("{prefix}ofs_embedding_linear_1.weight"),
        1,
        &invalid_configuration,
    )?;
    Ok(CogVideoXConfiguration {
        layout,
        in_channels,
        number_of_attention_heads,
        time_embedding_dimension,
        number_of_layers,
        patch_size,
        temporal_patch_size,
        sample_height,
        sample_width,
        sample_frames,
        text_embedding_dimension,
        ofs_embedding_dimension,
        learned_positional_embeddings: probe
            .tensor_shapes
            .contains_key(&format!("{prefix}patch_embed.pos_embedding")),
        latent_variant: if number_of_attention_heads >= 48 {
            CogVideoXLatentVariant::CogVideoX1_5
        } else {
            CogVideoXLatentVariant::CogVideoX
        },
    })
}

fn required_shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}

fn optional_dimension(
    probe: &ModelProbe,
    key: &str,
    dimension: usize,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<Option<u64>, ModelFamilyError> {
    let Some(shape) = probe.tensor_shapes.get(key) else {
        return Ok(None);
    };
    shape
        .get(dimension)
        .copied()
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or_else(|| invalid_configuration(format!("missing dimension {dimension} for {key}")))
}
