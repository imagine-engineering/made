use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyError, ModelFamilyStatePlanCase,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const KANDINSKY5_MEMORY_USAGE_FACTOR: f64 = 1.25;
pub const KANDINSKY5_VIDEO_SAMPLING_SHIFT: f64 = 10.0;
pub const KANDINSKY5_IMAGE_SAMPLING_SHIFT: f64 = 3.0;
pub const KANDINSKY5_TIME_DIMENSION: u64 = 512;
pub const KANDINSKY5_TEXT_INPUT_DIMENSION: u64 = 3_584;
pub const KANDINSKY5_POOLED_TEXT_INPUT_DIMENSION: u64 = 768;
pub const KANDINSKY5_OUTPUT_CHANNELS: u64 = 16;
pub const KANDINSKY5_PATCH_SIZE: [u64; 3] = [1, 2, 2];
pub const KANDINSKY5_TEXT_BLOCK_COUNT: usize = 2;
pub const KANDINSKY5_VISUAL_BLOCK_COUNT: usize = 32;
pub const KANDINSKY5_ROPE_THETA: f64 = 10_000.0;

pub const KANDINSKY5_VIDEO_LITE_MODEL_DIMENSION: u64 = 1_792;
pub const KANDINSKY5_VIDEO_PRO_MODEL_DIMENSION: u64 = 4_096;
pub const KANDINSKY5_IMAGE_LITE_MODEL_DIMENSION: u64 = 2_560;
pub const KANDINSKY5_VIDEO_VISUAL_EMBED_DIMENSION: u64 = 132;
pub const KANDINSKY5_IMAGE_VISUAL_EMBED_DIMENSION: u64 = 64;

pub const KANDINSKY5_VIDEO_LITE_AXES_DIMENSIONS: [u64; 3] = [16, 24, 24];
pub const KANDINSKY5_WIDE_AXES_DIMENSIONS: [u64; 3] = [32, 48, 48];
pub const KANDINSKY5_VIDEO_ROPE_SCALE_FACTOR: [f64; 3] = [1.0, 2.0, 2.0];
pub const KANDINSKY5_IMAGE_ROPE_SCALE_FACTOR: [f64; 3] = [1.0, 1.0, 1.0];

pub const KANDINSKY5_VIDEO_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanvideo_sim_model_0037::LATENT_FORMAT;
pub const KANDINSKY5_IMAGE_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_flux_sim_model_0029::LATENT_FORMAT;

pub const KANDINSKY5_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const KANDINSKY5_VIDEO_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.kandinsky5.Kandinsky5Tokenizer",
        clip_model: "sim.text_encoders.kandinsky5.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: KANDINSKY5_CLIP_CONFIGURATION,
        },
    }];
pub const KANDINSKY5_IMAGE_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.kandinsky5.Kandinsky5TokenizerImage",
        clip_model: "sim.text_encoders.kandinsky5.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: KANDINSKY5_CLIP_CONFIGURATION,
        },
    }];
pub static KANDINSKY5_VIDEO_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: KANDINSKY5_VIDEO_CLIP_CANDIDATES,
    dynamic_selection: false,
};
pub static KANDINSKY5_IMAGE_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: KANDINSKY5_IMAGE_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kandinsky5ConditioningFact {
    PooledOutputAdm,
    OptionalAttentionMask,
    OptionalCrossAttention,
    OptionalProcessedLatentTimeDimensionReplacement,
    ZeroImageAndInverseMaskVideoConcat,
}

pub const KANDINSKY5_VIDEO_CONDITIONING: &[Kandinsky5ConditioningFact] = &[
    Kandinsky5ConditioningFact::PooledOutputAdm,
    Kandinsky5ConditioningFact::OptionalAttentionMask,
    Kandinsky5ConditioningFact::OptionalCrossAttention,
    Kandinsky5ConditioningFact::OptionalProcessedLatentTimeDimensionReplacement,
    Kandinsky5ConditioningFact::ZeroImageAndInverseMaskVideoConcat,
];
pub const KANDINSKY5_IMAGE_CONDITIONING: &[Kandinsky5ConditioningFact] = &[
    Kandinsky5ConditioningFact::PooledOutputAdm,
    Kandinsky5ConditioningFact::OptionalAttentionMask,
    Kandinsky5ConditioningFact::OptionalCrossAttention,
    Kandinsky5ConditioningFact::OptionalProcessedLatentTimeDimensionReplacement,
];

pub const KANDINSKY5_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Kandinsky5 native diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Kandinsky5 Flux or HunyuanVideo latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Kandinsky5 Qwen 2.5 7B and CLIP-L conditioning encoders",
        required: false,
    },
];

pub const KANDINSKY5_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.visual_embeddings.in_layer.weight",
    "native.time_embeddings.in_layer.weight",
    "native.text_embeddings.in_layer.weight",
    "native.visual_transformer_blocks.0.cross_attention.key_norm.weight",
    "native.out_layer.out_layer.weight",
];
pub const KANDINSKY5_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.visual_embeddings.in_layer.bias",
    "native.time_embeddings.in_layer.bias",
    "native.time_embeddings.out_layer.weight",
    "native.text_embeddings.in_layer.bias",
    "native.pooled_text_embeddings.in_layer.weight",
    "native.text_transformer_blocks.0.self_attention.to_query.weight",
    "native.visual_transformer_blocks.0.feed_forward.in_layer.weight",
    "native.out_layer.out_layer.bias",
];
pub const KANDINSKY5_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: KANDINSKY5_MODEL_REQUIRED_KEYS,
        optional_keys: KANDINSKY5_MODEL_OPTIONAL_KEYS,
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

pub const KANDINSKY5_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const KANDINSKY5_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const KANDINSKY5_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "conditioning.text_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.text_embeddings.in_layer.weight",
            bias: Some("native.text_embeddings.in_layer.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning.timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embeddings.in_layer.weight",
            bias: Some("native.time_embeddings.in_layer.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning.timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "transformer.visual_block_0_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "output.visual_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.out_layer.out_layer.weight",
            bias: Some("native.out_layer.out_layer.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const KANDINSKY5_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
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

pub const KANDINSKY5_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"time_embeddings."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_embeddings.","to":"native.time_embeddings."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_embeddings."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_embeddings.","to":"native.text_embeddings."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"pooled_text_embeddings."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"pooled_text_embeddings.","to":"native.pooled_text_embeddings."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"visual_embeddings."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"visual_embeddings.","to":"native.visual_embeddings."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_transformer_blocks."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_transformer_blocks.","to":"native.text_transformer_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"visual_transformer_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"visual_transformer_blocks.","to":"native.visual_transformer_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"out_layer."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"out_layer.","to":"native.out_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const KANDINSKY5_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.visual_transformer_blocks.0.cross_attention.key_norm.weight",
            "model.diffusion_model.visual_embeddings.in_layer.bias",
            "model.diffusion_model.visual_embeddings.in_layer.weight",
            "model.diffusion_model.time_embeddings.in_layer.bias",
            "model.diffusion_model.visual_transformer_blocks.0.feed_forward.in_layer.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "visual_transformer_blocks.0.cross_attention.key_norm.weight",
            "visual_embeddings.in_layer.bias",
            "visual_embeddings.in_layer.weight",
            "time_embeddings.in_layer.bias",
            "visual_transformer_blocks.0.feed_forward.in_layer.weight",
        ],
        required_prefixes: &[],
    },
];

pub const KANDINSKY5_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &KANDINSKY5_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &KANDINSKY5_STANDALONE_STATE_PLAN,
    },
];

pub const KANDINSKY5_DIFFUSERS_MARKER: &str =
    "transformer_blocks.0.cross_attention.key_norm.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kandinsky5Variant {
    VideoLite,
    VideoPro,
    ImageLite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kandinsky5Layout {
    PrefixedNative,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug)]
pub struct Kandinsky5Configuration {
    pub variant: Kandinsky5Variant,
    pub layout: Kandinsky5Layout,
    pub input_visual_channels: u64,
    pub output_visual_channels: u64,
    pub model_dimension: u64,
    pub time_dimension: u64,
    pub feed_forward_dimension: u64,
    pub visual_embed_dimension: u64,
    pub patch_size: [u64; 3],
    pub text_block_count: usize,
    pub visual_block_count: usize,
    pub axes_dimensions: [u64; 3],
    pub attention_head_dimension: u64,
    pub attention_head_count: u64,
    pub rope_scale_factor: [f64; 3],
    pub rope_theta: f64,
    pub concat_conditioning: bool,
    pub conditioning: &'static [Kandinsky5ConditioningFact],
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
}

#[derive(Clone, Copy, Debug)]
pub struct Kandinsky5CommonMapping {
    pub components: &'static [ModelFamilyComponent],
    pub component_state_schemas: &'static [ModelFamilyComponentStateSchema],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub forward_program: &'static [ModelForwardStep],
}

pub static KANDINSKY5_COMMON_MAPPING: Kandinsky5CommonMapping = Kandinsky5CommonMapping {
    components: KANDINSKY5_COMPONENTS,
    component_state_schemas: KANDINSKY5_COMPONENT_STATE_SCHEMAS,
    supported_dtypes: KANDINSKY5_SUPPORTED_DTYPES,
    supported_devices: KANDINSKY5_SUPPORTED_DEVICES,
    forward_program: KANDINSKY5_FORWARD_PROGRAM,
};

pub fn common_mapping() -> &'static Kandinsky5CommonMapping {
    &KANDINSKY5_COMMON_MAPPING
}

pub fn state_plan_for_layout(
    layout: Kandinsky5Layout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        Kandinsky5Layout::PrefixedNative => &KANDINSKY5_PREFIXED_STATE_PLAN,
        Kandinsky5Layout::StandaloneNative => &KANDINSKY5_STANDALONE_STATE_PLAN,
    }
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Kandinsky5Configuration, ModelFamilyError> {
    let invalid = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "Kandinsky5 configuration is invalid: {message}"
        ))
    };
    if probe
        .tensor_shapes
        .contains_key(KANDINSKY5_DIFFUSERS_MARKER)
    {
        return Err(invalid(
            "Diffusers layout is unsupported; only source-native layouts are accepted".to_owned(),
        ));
    }
    let state_layout = probe.select_layout(KANDINSKY5_LAYOUT_SIGNATURES)?;
    let (layout, prefix) = match state_layout {
        ModelStateLayout::PrefixedNative => {
            (Kandinsky5Layout::PrefixedNative, "model.diffusion_model.")
        }
        ModelStateLayout::StandaloneNative => (Kandinsky5Layout::StandaloneNative, ""),
        ModelStateLayout::Diffusers => unreachable!("Kandinsky5 signatures exclude Diffusers"),
    };

    let model_dimension = exact_vector_dimension(
        probe,
        &format!("{prefix}visual_embeddings.in_layer.bias"),
        &invalid,
    )?;
    let time_dimension = exact_vector_dimension(
        probe,
        &format!("{prefix}time_embeddings.in_layer.bias"),
        &invalid,
    )?;
    if time_dimension != KANDINSKY5_TIME_DIMENSION {
        return Err(invalid(format!(
            "time dimension {time_dimension}; expected {KANDINSKY5_TIME_DIMENSION}"
        )));
    }
    let visual_embed_shape = required_shape(
        probe,
        &format!("{prefix}visual_embeddings.in_layer.weight"),
        &invalid,
    )?;
    if visual_embed_shape.len() != 2
        || visual_embed_shape[0] != model_dimension
        || visual_embed_shape[1] == 0
    {
        return Err(invalid(format!(
            "visual embedding shape {visual_embed_shape:?} does not match model dimension {model_dimension}"
        )));
    }
    let visual_embed_dimension = visual_embed_shape[1];
    let feed_forward_shape = required_shape(
        probe,
        &format!("{prefix}visual_transformer_blocks.0.feed_forward.in_layer.weight"),
        &invalid,
    )?;
    if feed_forward_shape.len() != 2
        || feed_forward_shape[1] != model_dimension
        || feed_forward_shape[0] != model_dimension.saturating_mul(4)
    {
        return Err(invalid(format!(
            "feed-forward shape {feed_forward_shape:?} does not encode 4x model width {model_dimension}"
        )));
    }
    let feed_forward_dimension = feed_forward_shape[0];
    let text_block_count = checked_exact_block_count(
        probe,
        &format!("{prefix}text_transformer_blocks.{{}}."),
        KANDINSKY5_TEXT_BLOCK_COUNT,
        "text",
        &invalid,
    )?;
    let visual_block_count = checked_exact_block_count(
        probe,
        &format!("{prefix}visual_transformer_blocks.{{}}."),
        KANDINSKY5_VISUAL_BLOCK_COUNT,
        "visual",
        &invalid,
    )?;

    let (
        variant,
        expected_visual_embed,
        axes_dimensions,
        rope_scale_factor,
        concat_conditioning,
        conditioning,
        sampling_shift,
        latent_format,
        clip_target,
    ) = if model_dimension == KANDINSKY5_IMAGE_LITE_MODEL_DIMENSION
        && visual_embed_dimension == KANDINSKY5_IMAGE_VISUAL_EMBED_DIMENSION
    {
        (
            Kandinsky5Variant::ImageLite,
            KANDINSKY5_IMAGE_VISUAL_EMBED_DIMENSION,
            KANDINSKY5_WIDE_AXES_DIMENSIONS,
            KANDINSKY5_IMAGE_ROPE_SCALE_FACTOR,
            false,
            KANDINSKY5_IMAGE_CONDITIONING,
            KANDINSKY5_IMAGE_SAMPLING_SHIFT,
            KANDINSKY5_IMAGE_LATENT_FORMAT,
            &KANDINSKY5_IMAGE_CLIP_TARGET,
        )
    } else {
        match model_dimension {
            KANDINSKY5_VIDEO_LITE_MODEL_DIMENSION => (
                Kandinsky5Variant::VideoLite,
                KANDINSKY5_VIDEO_VISUAL_EMBED_DIMENSION,
                KANDINSKY5_VIDEO_LITE_AXES_DIMENSIONS,
                KANDINSKY5_VIDEO_ROPE_SCALE_FACTOR,
                true,
                KANDINSKY5_VIDEO_CONDITIONING,
                KANDINSKY5_VIDEO_SAMPLING_SHIFT,
                KANDINSKY5_VIDEO_LATENT_FORMAT,
                &KANDINSKY5_VIDEO_CLIP_TARGET,
            ),
            KANDINSKY5_VIDEO_PRO_MODEL_DIMENSION => (
                Kandinsky5Variant::VideoPro,
                KANDINSKY5_VIDEO_VISUAL_EMBED_DIMENSION,
                KANDINSKY5_WIDE_AXES_DIMENSIONS,
                KANDINSKY5_VIDEO_ROPE_SCALE_FACTOR,
                true,
                KANDINSKY5_VIDEO_CONDITIONING,
                KANDINSKY5_VIDEO_SAMPLING_SHIFT,
                KANDINSKY5_VIDEO_LATENT_FORMAT,
                &KANDINSKY5_VIDEO_CLIP_TARGET,
            ),
            _ => {
                return Err(invalid(format!(
                    "unsupported model/visual dimensions {model_dimension}/{visual_embed_dimension}"
                )));
            }
        }
    };
    if visual_embed_dimension != expected_visual_embed {
        return Err(invalid(format!(
            "unsupported model/visual dimensions {model_dimension}/{visual_embed_dimension}; expected visual dimension {expected_visual_embed}"
        )));
    }

    let attention_head_dimension = axes_dimensions.iter().sum::<u64>();
    if model_dimension % attention_head_dimension != 0 {
        return Err(invalid(format!(
            "model dimension {model_dimension} is not divisible by head dimension {attention_head_dimension}"
        )));
    }
    let marker_shape = required_shape(
        probe,
        &format!("{prefix}visual_transformer_blocks.0.cross_attention.key_norm.weight"),
        &invalid,
    )?;
    if marker_shape != [attention_head_dimension] {
        return Err(invalid(format!(
            "cross-attention key norm shape {marker_shape:?}; expected [{attention_head_dimension}]"
        )));
    }
    let patch_volume = KANDINSKY5_PATCH_SIZE.iter().product::<u64>();
    if visual_embed_dimension % patch_volume != 0 {
        return Err(invalid(format!(
            "visual embedding dimension {visual_embed_dimension} is not divisible by patch volume {patch_volume}"
        )));
    }

    Ok(Kandinsky5Configuration {
        variant,
        layout,
        input_visual_channels: visual_embed_dimension / patch_volume,
        output_visual_channels: KANDINSKY5_OUTPUT_CHANNELS,
        model_dimension,
        time_dimension,
        feed_forward_dimension,
        visual_embed_dimension,
        patch_size: KANDINSKY5_PATCH_SIZE,
        text_block_count,
        visual_block_count,
        axes_dimensions,
        attention_head_dimension,
        attention_head_count: model_dimension / attention_head_dimension,
        rope_scale_factor,
        rope_theta: KANDINSKY5_ROPE_THETA,
        concat_conditioning,
        conditioning,
        sampling_shift,
        memory_usage_factor: KANDINSKY5_MEMORY_USAGE_FACTOR,
        latent_format,
        clip_target,
    })
}

fn required_shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing {key}")))
}

fn exact_vector_dimension(
    probe: &ModelProbe,
    key: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<u64, ModelFamilyError> {
    let shape = required_shape(probe, key, invalid)?;
    if shape.len() != 1 || shape[0] == 0 {
        return Err(invalid(format!("{key} shape {shape:?} is not a vector")));
    }
    Ok(shape[0])
}

fn checked_exact_block_count(
    probe: &ModelProbe,
    pattern: &str,
    expected: usize,
    kind: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    let next_prefix = pattern.replacen("{}", &count.to_string(), 1);
    let has_later_block = probe.tensor_shapes.keys().any(|key| {
        key.strip_prefix(pattern.split_once("{}").map_or("", |parts| parts.0))
            .and_then(|suffix| suffix.split('.').next())
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= count)
    });
    if count != expected
        || has_later_block
        || probe
            .tensor_shapes
            .keys()
            .any(|key| key.starts_with(&next_prefix))
    {
        return Err(invalid(format!(
            "{kind} transformer blocks are not exactly {expected} consecutive entries"
        )));
    }
    Ok(count)
}
