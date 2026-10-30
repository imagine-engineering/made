use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyError, ModelForwardOperation, ModelForwardStep,
    ModelProbe, ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const LTX_DEFAULT_CROSS_ATTENTION_DIMENSION: u64 = 2_048;
pub const LTX_DEFAULT_ATTENTION_HEAD_COUNT: u64 = 32;
pub const LTXV_DEFAULT_LAYER_COUNT: usize = 28;
pub const LTXV_SAMPLING_SHIFT: f64 = 2.37;
pub const LTXV_BASE_MEMORY_USAGE_FACTOR: f64 = 5.5;
pub const LTXAV_MEMORY_USAGE_FACTOR: f64 = 0.077;
pub const LTX_MAX_TRANSFORMER_CONFIG_BYTES: usize = 16 * 1024;
pub const LTX_MAX_LAYER_COUNT: usize = 256;
pub const LTX_MAX_HEAD_DIMENSION: u64 = 512;
pub const LTX_MAX_CROSS_ATTENTION_DIMENSION: u64 = 65_536;
pub const LTX_MAX_CHANNELS: u64 = 4_096;

pub const LTXV_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_ltxv_sim_model_0040::LATENT_FORMAT;
pub const LTXAV_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_ltxav_sim_model_0039::LATENT_FORMAT;

pub const LTX_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];
pub const LTX_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.lt.LTXVT5Tokenizer",
        clip_model: "sim.text_encoders.lt.ltxv_te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: LTX_CLIP_CONFIGURATION,
        },
    }];
pub static LTX_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: LTX_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LtxConditioningFact {
    OptionalAttentionMask,
    OptionalCrossAttention,
    FrameRateDefault25,
    OptionalVideoDenoiseMask,
    OptionalKeyframeIndices,
    OptionalGuideAttentionEntries,
    VideoMaskedTimestepPatchification,
    ProcessAudioVideoTextEmbeddings,
    OptionalAudioDenoiseMask,
    OptionalLatentShapes,
    OptionalReferenceAudio,
    AudioMaskedTimestepPatchification,
}

pub const LTXV_CONDITIONING: &[LtxConditioningFact] = &[
    LtxConditioningFact::OptionalAttentionMask,
    LtxConditioningFact::OptionalCrossAttention,
    LtxConditioningFact::FrameRateDefault25,
    LtxConditioningFact::OptionalVideoDenoiseMask,
    LtxConditioningFact::OptionalKeyframeIndices,
    LtxConditioningFact::OptionalGuideAttentionEntries,
    LtxConditioningFact::VideoMaskedTimestepPatchification,
];
pub const LTXAV_CONDITIONING: &[LtxConditioningFact] = &[
    LtxConditioningFact::OptionalAttentionMask,
    LtxConditioningFact::OptionalCrossAttention,
    LtxConditioningFact::FrameRateDefault25,
    LtxConditioningFact::OptionalVideoDenoiseMask,
    LtxConditioningFact::OptionalKeyframeIndices,
    LtxConditioningFact::OptionalGuideAttentionEntries,
    LtxConditioningFact::VideoMaskedTimestepPatchification,
    LtxConditioningFact::ProcessAudioVideoTextEmbeddings,
    LtxConditioningFact::OptionalAudioDenoiseMask,
    LtxConditioningFact::OptionalLatentShapes,
    LtxConditioningFact::OptionalReferenceAudio,
    LtxConditioningFact::AudioMaskedTimestepPatchification,
];

pub const LTX_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Lightricks native video or audio-video transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "LTXV or LTXAV latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "LTX T5 conditioning encoder",
        required: false,
    },
];

pub const LTX_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.adaln_single.emb.timestep_embedder.linear_1.bias",
    "native.patchify_proj.weight",
    "native.transformer_blocks.0.attn2.to_k.weight",
    "native.proj_out.weight",
];
pub const LTX_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.patchify_proj.bias",
    "native.caption_projection.linear_1.weight",
    "native.caption_projection.linear_2.weight",
    "native.proj_out.bias",
    "native.audio_adaln_single.linear.weight",
    "native.audio_patchify_proj.weight",
    "native.audio_patchify_proj.bias",
    "native.audio_proj_out.weight",
    "native.audio_proj_out.bias",
];
pub const LTX_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: LTX_MODEL_REQUIRED_KEYS,
        optional_keys: LTX_MODEL_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "vae",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const LTX_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const LTX_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const LTX_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "input.video_patch_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.patchify_proj.weight",
            bias: Some("native.patchify_proj.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning.timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "transformer.block_0_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "output.video_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.proj_out.weight",
            bias: Some("native.proj_out.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const LTX_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };
pub const LTX_SAVED_MODEL_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };
pub const LTX_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Any":[{"Prefix":"adaln_single."},{"Prefix":"audio_adaln_single."},{"Prefix":"patchify_proj."},{"Prefix":"audio_patchify_proj."},{"Prefix":"caption_projection."},{"Prefix":"audio_caption_projection."},{"Prefix":"transformer_blocks."},{"Prefix":"scale_shift_table"},{"Prefix":"audio_scale_shift_table"},{"Prefix":"norm_out."},{"Prefix":"audio_norm_out."},{"Prefix":"proj_out."},{"Prefix":"audio_proj_out."}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"OrderedOptional":[{"Prefix":{"from":"adaln_single.","to":"native.adaln_single."}},{"Prefix":{"from":"audio_adaln_single.","to":"native.audio_adaln_single."}},{"Prefix":{"from":"patchify_proj.","to":"native.patchify_proj."}},{"Prefix":{"from":"audio_patchify_proj.","to":"native.audio_patchify_proj."}},{"Prefix":{"from":"caption_projection.","to":"native.caption_projection."}},{"Prefix":{"from":"audio_caption_projection.","to":"native.audio_caption_projection."}},{"Prefix":{"from":"transformer_blocks.","to":"native.transformer_blocks."}},{"Prefix":{"from":"scale_shift_table","to":"native.scale_shift_table"}},{"Prefix":{"from":"audio_scale_shift_table","to":"native.audio_scale_shift_table"}},{"Prefix":{"from":"norm_out.","to":"native.norm_out."}},{"Prefix":{"from":"audio_norm_out.","to":"native.audio_norm_out."}},{"Prefix":{"from":"proj_out.","to":"native.proj_out."}},{"Prefix":{"from":"audio_proj_out.","to":"native.audio_proj_out."}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const LTX_TIMESTEP_MARKER: &str = "adaln_single.emb.timestep_embedder.linear_1.bias";
pub const LTX_AUDIO_MARKER: &str = "audio_adaln_single.linear.weight";
pub const LTX_PIXART_COLLISION_MARKER: &str = "pos_embed.proj.bias";
pub const LTX_DIFFUSERS_MARKER: &str = "transformer_blocks.0.attn1.to_q.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LtxVariant {
    Video,
    AudioVideo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LtxLayout {
    PrefixedNative,
    SavedModel,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug)]
pub struct LtxConfiguration {
    pub variant: LtxVariant,
    pub layout: LtxLayout,
    pub input_channels: u64,
    pub inner_dimension: u64,
    pub number_of_layers: usize,
    pub attention_head_dimension: u64,
    pub number_of_attention_heads: u64,
    pub cross_attention_dimension: u64,
    pub audio_input_channels: Option<u64>,
    pub audio_inner_dimension: Option<u64>,
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub conditioning: &'static [LtxConditioningFact],
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
}

#[derive(Clone, Copy, Debug)]
pub struct LtxCommonMapping {
    pub components: &'static [ModelFamilyComponent],
    pub component_state_schemas: &'static [ModelFamilyComponentStateSchema],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub forward_program: &'static [ModelForwardStep],
}

pub static LTX_COMMON_MAPPING: LtxCommonMapping = LtxCommonMapping {
    components: LTX_COMPONENTS,
    component_state_schemas: LTX_COMPONENT_STATE_SCHEMAS,
    supported_dtypes: LTX_SUPPORTED_DTYPES,
    supported_devices: LTX_SUPPORTED_DEVICES,
    forward_program: LTX_FORWARD_PROGRAM,
};

pub fn common_mapping() -> &'static LtxCommonMapping {
    &LTX_COMMON_MAPPING
}

pub fn state_plan_for_layout(layout: LtxLayout) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        LtxLayout::PrefixedNative => &LTX_PREFIXED_STATE_PLAN,
        LtxLayout::SavedModel => &LTX_SAVED_MODEL_STATE_PLAN,
        LtxLayout::StandaloneNative => &LTX_STANDALONE_STATE_PLAN,
    }
}

pub fn ltxv_memory_usage_factor(cross_attention_dimension: u64) -> Result<f64, ModelFamilyError> {
    if cross_attention_dimension == 0
        || cross_attention_dimension > LTX_MAX_CROSS_ATTENTION_DIMENSION
    {
        return Err(invalid(format!(
            "cross-attention dimension {cross_attention_dimension} is outside 1..={LTX_MAX_CROSS_ATTENTION_DIMENSION}"
        )));
    }
    Ok(
        (cross_attention_dimension as f64 / LTX_DEFAULT_CROSS_ATTENTION_DIMENSION as f64)
            * LTXV_BASE_MEMORY_USAGE_FACTOR,
    )
}

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<LtxConfiguration, ModelFamilyError> {
    let (layout, prefix) = select_layout(probe)?;
    if probe
        .tensor_shapes
        .contains_key(&format!("{prefix}{LTX_PIXART_COLLISION_MARKER}"))
    {
        return Err(invalid(
            "PixArt collision marker is present; the layout is not LTX".to_owned(),
        ));
    }
    if probe
        .tensor_shapes
        .contains_key(&format!("{prefix}{LTX_DIFFUSERS_MARKER}"))
        && !probe
            .tensor_shapes
            .contains_key(&format!("{prefix}{LTX_TIMESTEP_MARKER}"))
    {
        return Err(invalid(
            "Diffusers layout is unsupported; only source-native layouts are accepted".to_owned(),
        ));
    }

    let variant = if probe
        .tensor_shapes
        .contains_key(&format!("{prefix}{LTX_AUDIO_MARKER}"))
    {
        LtxVariant::AudioVideo
    } else {
        LtxVariant::Video
    };
    let attention_shape = required_matrix(
        probe,
        &format!("{prefix}transformer_blocks.0.attn2.to_k.weight"),
    )?;
    if attention_shape[0] % LTX_DEFAULT_ATTENTION_HEAD_COUNT != 0 {
        return Err(invalid(format!(
            "attention projection width {} is not divisible by {} heads",
            attention_shape[0], LTX_DEFAULT_ATTENTION_HEAD_COUNT
        )));
    }
    let detected_head_dimension = attention_shape[0] / LTX_DEFAULT_ATTENTION_HEAD_COUNT;
    let detected_context_dimension = attention_shape[1];
    let patch_shape = required_matrix(probe, &format!("{prefix}patchify_proj.weight"))?;
    let detected_inner_dimension = patch_shape[0];
    let detected_input_channels = patch_shape[1];
    let detected_layers = checked_layer_count(probe, prefix)?;
    let overrides = transformer_overrides(probe)?;

    let number_of_layers = checked_override_usize(
        &overrides,
        "num_layers",
        detected_layers,
        1,
        LTX_MAX_LAYER_COUNT,
    )?;
    let attention_head_dimension = checked_override_u64(
        &overrides,
        "attention_head_dim",
        detected_head_dimension,
        1,
        LTX_MAX_HEAD_DIMENSION,
    )?;
    let number_of_attention_heads = checked_override_u64(
        &overrides,
        "num_attention_heads",
        LTX_DEFAULT_ATTENTION_HEAD_COUNT,
        1,
        256,
    )?;
    let cross_attention_dimension = checked_override_u64(
        &overrides,
        "cross_attention_dim",
        detected_context_dimension,
        1,
        LTX_MAX_CROSS_ATTENTION_DIMENSION,
    )?;
    let input_channels = checked_override_u64(
        &overrides,
        "in_channels",
        detected_input_channels,
        1,
        LTX_MAX_CHANNELS,
    )?;
    if number_of_attention_heads
        .checked_mul(attention_head_dimension)
        .ok_or(ModelFamilyError::MemoryOverflow)?
        != detected_inner_dimension
    {
        return Err(invalid(format!(
            "{} heads x head dimension {} does not match inner dimension {}",
            number_of_attention_heads, attention_head_dimension, detected_inner_dimension
        )));
    }

    let (audio_input_channels, audio_inner_dimension) = if variant == LtxVariant::AudioVideo {
        let audio_patch = required_matrix(probe, &format!("{prefix}audio_patchify_proj.weight"))?;
        let audio_input = checked_override_u64(
            &overrides,
            "audio_in_channels",
            audio_patch[1],
            1,
            LTX_MAX_CHANNELS,
        )?;
        let audio_inner = audio_patch[0];
        (Some(audio_input), Some(audio_inner))
    } else {
        if overrides.contains_key("audio_in_channels") {
            return Err(invalid(
                "video-only LTXV configuration cannot declare audio channels".to_owned(),
            ));
        }
        (None, None)
    };

    let (memory_usage_factor, conditioning, latent_format) = match variant {
        LtxVariant::Video => (
            ltxv_memory_usage_factor(cross_attention_dimension)?,
            LTXV_CONDITIONING,
            LTXV_LATENT_FORMAT,
        ),
        LtxVariant::AudioVideo => (
            LTXAV_MEMORY_USAGE_FACTOR,
            LTXAV_CONDITIONING,
            LTXAV_LATENT_FORMAT,
        ),
    };

    Ok(LtxConfiguration {
        variant,
        layout,
        input_channels,
        inner_dimension: detected_inner_dimension,
        number_of_layers,
        attention_head_dimension,
        number_of_attention_heads,
        cross_attention_dimension,
        audio_input_channels,
        audio_inner_dimension,
        sampling_shift: LTXV_SAMPLING_SHIFT,
        memory_usage_factor,
        conditioning,
        latent_format,
        clip_target: &LTX_CLIP_TARGET,
    })
}

fn select_layout(probe: &ModelProbe) -> Result<(LtxLayout, &'static str), ModelFamilyError> {
    let candidates = [
        (LtxLayout::PrefixedNative, "model.diffusion_model."),
        (LtxLayout::SavedModel, "model."),
        (LtxLayout::StandaloneNative, ""),
    ];
    let required = [
        LTX_TIMESTEP_MARKER,
        "patchify_proj.weight",
        "transformer_blocks.0.attn2.to_k.weight",
        "proj_out.weight",
    ];
    let matches = candidates
        .into_iter()
        .filter(|(_, prefix)| {
            required
                .iter()
                .all(|key| probe.tensor_shapes.contains_key(&format!("{prefix}{key}")))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [selected] => Ok(*selected),
        [] => {
            let diffusers = candidates.into_iter().any(|(_, prefix)| {
                probe
                    .tensor_shapes
                    .contains_key(&format!("{prefix}{LTX_DIFFUSERS_MARKER}"))
            });
            if diffusers {
                Err(invalid(
                    "Diffusers layout is unsupported; only source-native layouts are accepted"
                        .to_owned(),
                ))
            } else {
                Err(ModelFamilyError::ModelLayoutSelection(
                    "no supported LTX source-native layout matched the required keys".to_owned(),
                ))
            }
        }
        _ => Err(ModelFamilyError::ModelLayoutSelection(
            "LTX probe ambiguously matches multiple source-native layouts".to_owned(),
        )),
    }
}

fn required_matrix<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(invalid(format!(
            "{key} shape {shape:?} is not a non-empty matrix"
        )));
    }
    Ok(shape)
}

fn checked_layer_count(probe: &ModelProbe, prefix: &str) -> Result<usize, ModelFamilyError> {
    let pattern = format!("{prefix}transformer_blocks.{{}}.");
    let count = probe.consecutive_block_count(&pattern)?;
    if count == 0 || count > LTX_MAX_LAYER_COUNT {
        return Err(invalid(format!(
            "transformer layer count {count} is outside 1..={LTX_MAX_LAYER_COUNT}"
        )));
    }
    let stem = format!("{prefix}transformer_blocks.");
    let has_gap_or_later = probe.tensor_shapes.keys().any(|key| {
        key.strip_prefix(&stem)
            .and_then(|suffix| suffix.split('.').next())
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= count)
    });
    if has_gap_or_later {
        return Err(invalid(
            "transformer blocks are not a consecutive bounded sequence".to_owned(),
        ));
    }
    Ok(count)
}

fn transformer_overrides(
    probe: &ModelProbe,
) -> Result<serde_json::Map<String, serde_json::Value>, ModelFamilyError> {
    let Some(raw) = probe.metadata.get("config") else {
        return Ok(serde_json::Map::new());
    };
    if raw.len() > LTX_MAX_TRANSFORMER_CONFIG_BYTES {
        return Err(invalid(format!(
            "transformer config has {} bytes; maximum is {LTX_MAX_TRANSFORMER_CONFIG_BYTES}",
            raw.len()
        )));
    }
    let root: serde_json::Value = serde_json::from_str(raw)
        .map_err(|error| invalid(format!("transformer config is invalid JSON: {error}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| invalid("transformer config root must be an object".to_owned()))?;
    let Some(transformer) = root.get("transformer") else {
        return Ok(serde_json::Map::new());
    };
    transformer
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("transformer config must be an object".to_owned()))
}

fn checked_override_u64(
    overrides: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    detected: u64,
    minimum: u64,
    maximum: u64,
) -> Result<u64, ModelFamilyError> {
    let Some(value) = overrides.get(field) else {
        return Ok(detected);
    };
    let value = value
        .as_u64()
        .ok_or_else(|| invalid(format!("transformer.{field} must be an unsigned integer")))?;
    if value < minimum || value > maximum {
        return Err(invalid(format!(
            "transformer.{field} {value} is outside {minimum}..={maximum}"
        )));
    }
    if value != detected {
        return Err(invalid(format!(
            "transformer.{field} override {value} contradicts detected value {detected}"
        )));
    }
    Ok(value)
}

fn checked_override_usize(
    overrides: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    detected: usize,
    minimum: usize,
    maximum: usize,
) -> Result<usize, ModelFamilyError> {
    let value = checked_override_u64(
        overrides,
        field,
        detected as u64,
        minimum as u64,
        maximum as u64,
    )?;
    usize::try_from(value).map_err(|_| ModelFamilyError::MemoryOverflow)
}

fn invalid(message: String) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!("LTX configuration is invalid: {message}"))
}
