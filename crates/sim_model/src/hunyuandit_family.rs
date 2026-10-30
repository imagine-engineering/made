use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelFamilyComponent, ModelFamilyError, ModelFamilyStatePlanCase,
    ModelForwardOperation, ModelForwardStep, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const HUNYUANDIT_MEMORY_USAGE_FACTOR: f64 = 1.3;
pub const HUNYUANDIT_NUMBER_OF_HEADS: u64 = 16;
pub const HUNYUANDIT_DEFAULT_MLP_RATIO: f64 = 4.0;
pub const HUNYUANDIT_G_MLP_RATIO: f64 = 4.3637;
pub const HUNYUANDIT_G_HIDDEN_SIZE: u64 = 1_408;
pub const HUNYUANDIT_G_DEPTH: usize = 40;
pub const HUNYUANDIT_PATCH_SIZE: u64 = 2;
pub const HUNYUANDIT_INPUT_CHANNELS: u64 = 4;
pub const HUNYUANDIT_CLIP_TEXT_DIMENSION: u64 = 1_024;
pub const HUNYUANDIT_T5_TEXT_DIMENSION: u64 = 2_048;
pub const HUNYUANDIT_CLIP_TEXT_LENGTH: u64 = 77;
pub const HUNYUANDIT_T5_TEXT_LENGTH: u64 = 256;
pub const HUNYUANDIT_BASE_EXTRA_INPUT: u64 = 1_024;
pub const HUNYUANDIT_DIT1_EXTRA_INPUT: u64 = 3_968;
pub const HUNYUANDIT_IMAGE_META_DIMENSION: u64 = 6;
pub const HUNYUANDIT_IMAGE_META_EMBEDDING_DIMENSION: u64 = 256;
pub const HUNYUANDIT_LINEAR_START: f64 = 0.00085;
pub const HUNYUANDIT_LINEAR_END: f64 = 0.018;
pub const HUNYUANDIT1_LINEAR_END: f64 = 0.03;

pub const HUNYUANDIT_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_sdxl_sim_model_0047::LATENT_FORMAT;

pub const HUNYUANDIT_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.hydit.HyditTokenizer",
        clip_model: "sim.text_encoders.hydit.HyditModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

pub static HUNYUANDIT_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: HUNYUANDIT_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const HUNYUANDIT_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "HunyuanDiT native diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "SDXL latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "HunyuanDiT CLIP and mT5 conditioning encoders",
        required: false,
    },
];

pub const HUNYUANDIT_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
pub const HUNYUANDIT_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const HUNYUANDIT_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "conditioning.t5_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.mlp_t5.0.weight",
            bias: Some("native.mlp_t5.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning.t5_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "conditioning.extra_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.extra_embedder.0.weight",
            bias: Some("native.extra_embedder.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "transformer.block_0_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "output.final_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanDiTVariant {
    DiT,
    DiT1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanDiTLayout {
    PrefixedNative,
    SavedModel,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanDiTAttentionPrecision {
    Float32,
    Inherited,
}

#[derive(Clone, Copy, Debug)]
pub struct HunyuanDiTConfiguration {
    pub variant: HunyuanDiTVariant,
    pub layout: HunyuanDiTLayout,
    pub in_channels: u64,
    pub patch_size: u64,
    pub hidden_size: u64,
    pub depth: usize,
    pub number_of_heads: u64,
    pub mlp_ratio: f64,
    pub extra_input_dimension: u64,
    pub size_conditioning: bool,
    pub style_conditioning: bool,
    pub qk_normalization: bool,
    pub learn_sigma: bool,
    pub attention_precision: HunyuanDiTAttentionPrecision,
    pub sampling_linear_start: f64,
    pub sampling_linear_end: f64,
    pub memory_usage_factor: f64,
    pub latent_format: &'static LatentFormatDefinition,
}

#[derive(Clone, Copy, Debug)]
pub struct HunyuanDiTCommonMapping {
    pub clip_target: &'static ModelClipTargetDefinition,
    pub components: &'static [ModelFamilyComponent],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub latent_format: &'static LatentFormatDefinition,
    pub forward_program: &'static [ModelForwardStep],
    pub memory_usage_factor: f64,
}

pub static HUNYUANDIT_COMMON_MAPPING: HunyuanDiTCommonMapping = HunyuanDiTCommonMapping {
    clip_target: &HUNYUANDIT_CLIP_TARGET,
    components: HUNYUANDIT_COMPONENTS,
    supported_dtypes: HUNYUANDIT_SUPPORTED_DTYPES,
    supported_devices: HUNYUANDIT_SUPPORTED_DEVICES,
    latent_format: HUNYUANDIT_LATENT_FORMAT,
    forward_program: HUNYUANDIT_FORWARD_PROGRAM,
    memory_usage_factor: HUNYUANDIT_MEMORY_USAGE_FACTOR,
};

pub fn common_mapping() -> &'static HunyuanDiTCommonMapping {
    &HUNYUANDIT_COMMON_MAPPING
}

pub const HUNYUANDIT_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
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

pub const HUNYUANDIT_SAVED_MODEL_STATE_PLAN: ModelStateTransformPlanDefinition =
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

pub const HUNYUANDIT_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"mlp_t5."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"mlp_t5.","to":"native.mlp_t5."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Exact":"text_embedding_padding"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.text_embedding_padding"},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"pooler."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"pooler.","to":"native.pooler."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"style_embedder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"style_embedder.","to":"native.style_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"x_embedder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t_embedder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"t_embedder.","to":"native.t_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"extra_embedder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"extra_embedder.","to":"native.extra_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"blocks.","to":"native.blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_layer."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const HUNYUANDIT_STANDARD_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &HUNYUANDIT_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &HUNYUANDIT_STANDALONE_STATE_PLAN,
    },
];

pub fn state_plan_for_layout(
    layout: HunyuanDiTLayout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        HunyuanDiTLayout::PrefixedNative => &HUNYUANDIT_PREFIXED_STATE_PLAN,
        HunyuanDiTLayout::SavedModel => &HUNYUANDIT_SAVED_MODEL_STATE_PLAN,
        HunyuanDiTLayout::StandaloneNative => &HUNYUANDIT_STANDALONE_STATE_PLAN,
    }
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<HunyuanDiTConfiguration, ModelFamilyError> {
    let invalid = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "HunyuanDiT configuration is invalid: {message}"
        ))
    };
    let domains = [
        (HunyuanDiTLayout::PrefixedNative, "model.diffusion_model."),
        (HunyuanDiTLayout::SavedModel, "model."),
        (HunyuanDiTLayout::StandaloneNative, ""),
    ];
    let mut matches = Vec::new();
    let mut partial = false;
    for (layout, prefix) in domains {
        let markers = [
            format!("{prefix}mlp_t5.0.weight"),
            format!("{prefix}x_embedder.proj.weight"),
            format!("{prefix}extra_embedder.0.weight"),
        ];
        let marker_count = markers
            .iter()
            .filter(|key| probe.tensor_shapes.contains_key(*key))
            .count();
        partial |= marker_count > 0 && marker_count < markers.len();
        if marker_count == markers.len() {
            matches.push((layout, prefix));
        }
    }
    let (layout, prefix) = match matches.as_slice() {
        [entry] => *entry,
        [] if partial => return Err(invalid("partial marker set".to_owned())),
        [] => {
            return Err(ModelFamilyError::ModelLayoutSelection(
                "parsed tensor keys match no HunyuanDiT source layout".to_owned(),
            ));
        }
        _ => {
            return Err(ModelFamilyError::ModelLayoutSelection(
                "parsed tensor keys ambiguously match multiple HunyuanDiT source layouts"
                    .to_owned(),
            ));
        }
    };

    if probe
        .tensor_shapes
        .contains_key(&format!("{prefix}y_embedder.y_embedding"))
        || probe.tensor_shapes.contains_key(&format!(
            "{prefix}adaln_single.emb.timestep_embedder.linear_1.weight"
        ))
    {
        return Err(invalid(
            "PixArt cross-family markers collide with HunyuanDiT".to_owned(),
        ));
    }

    let x_projection = required_shape(probe, &format!("{prefix}x_embedder.proj.weight"), &invalid)?;
    if x_projection.len() != 4
        || x_projection[0] == 0
        || x_projection[1] != HUNYUANDIT_INPUT_CHANNELS
        || x_projection[2] != HUNYUANDIT_PATCH_SIZE
        || x_projection[3] != HUNYUANDIT_PATCH_SIZE
    {
        return Err(invalid("x_embedder.proj.weight shape".to_owned()));
    }
    let hidden_size = x_projection[0];
    if !hidden_size.is_multiple_of(HUNYUANDIT_NUMBER_OF_HEADS) {
        return Err(invalid(format!(
            "hidden size {hidden_size} is not divisible by {} heads",
            HUNYUANDIT_NUMBER_OF_HEADS
        )));
    }

    require_matrix(
        probe,
        &format!("{prefix}mlp_t5.0.weight"),
        [
            HUNYUANDIT_T5_TEXT_DIMENSION * 4,
            HUNYUANDIT_T5_TEXT_DIMENSION,
        ],
        &invalid,
    )?;
    require_matrix(
        probe,
        &format!("{prefix}mlp_t5.2.weight"),
        [
            HUNYUANDIT_CLIP_TEXT_DIMENSION,
            HUNYUANDIT_T5_TEXT_DIMENSION * 4,
        ],
        &invalid,
    )?;
    let extra = required_shape(probe, &format!("{prefix}extra_embedder.0.weight"), &invalid)?;
    let expanded_hidden_size = hidden_size
        .checked_mul(4)
        .ok_or_else(|| invalid("hidden-size expansion overflow".to_owned()))?;
    if extra.len() != 2 || extra[0] != expanded_hidden_size || extra[1] == 0 {
        return Err(invalid("extra_embedder.0.weight shape".to_owned()));
    }

    let dit1_expected_input = hidden_size
        .checked_add(HUNYUANDIT_BASE_EXTRA_INPUT)
        .and_then(|value| {
            HUNYUANDIT_IMAGE_META_DIMENSION
                .checked_mul(HUNYUANDIT_IMAGE_META_EMBEDDING_DIMENSION)
                .and_then(|image_meta| value.checked_add(image_meta))
        })
        .ok_or_else(|| invalid("DiT1 conditioning dimension overflow".to_owned()))?;
    let (variant, size_conditioning, style_conditioning) = match extra[1] {
        HUNYUANDIT_BASE_EXTRA_INPUT => (HunyuanDiTVariant::DiT, false, false),
        HUNYUANDIT_DIT1_EXTRA_INPUT if dit1_expected_input == HUNYUANDIT_DIT1_EXTRA_INPUT => {
            (HunyuanDiTVariant::DiT1, true, true)
        }
        value => {
            return Err(invalid(format!(
                "extra embedder input width {value} is neither base DiT nor a consistent DiT1 width"
            )));
        }
    };

    let depth = checked_depth(probe, &format!("{prefix}blocks."), &invalid)?;
    let mlp_ratio = if hidden_size == HUNYUANDIT_G_HIDDEN_SIZE && depth == HUNYUANDIT_G_DEPTH {
        HUNYUANDIT_G_MLP_RATIO
    } else {
        HUNYUANDIT_DEFAULT_MLP_RATIO
    };
    let (attention_precision, sampling_linear_end) = match variant {
        HunyuanDiTVariant::DiT => (HunyuanDiTAttentionPrecision::Float32, HUNYUANDIT_LINEAR_END),
        HunyuanDiTVariant::DiT1 => (
            HunyuanDiTAttentionPrecision::Inherited,
            HUNYUANDIT1_LINEAR_END,
        ),
    };
    Ok(HunyuanDiTConfiguration {
        variant,
        layout,
        in_channels: HUNYUANDIT_INPUT_CHANNELS,
        patch_size: HUNYUANDIT_PATCH_SIZE,
        hidden_size,
        depth,
        number_of_heads: HUNYUANDIT_NUMBER_OF_HEADS,
        mlp_ratio,
        extra_input_dimension: extra[1],
        size_conditioning,
        style_conditioning,
        qk_normalization: true,
        learn_sigma: true,
        attention_precision,
        sampling_linear_start: HUNYUANDIT_LINEAR_START,
        sampling_linear_end,
        memory_usage_factor: HUNYUANDIT_MEMORY_USAGE_FACTOR,
        latent_format: HUNYUANDIT_LATENT_FORMAT,
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
        .ok_or_else(|| invalid(format!("missing required tensor {key}")))
}

fn require_matrix(
    probe: &ModelProbe,
    key: &str,
    expected: [u64; 2],
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<(), ModelFamilyError> {
    let shape = required_shape(probe, key, invalid)?;
    if shape != expected {
        return Err(invalid(format!(
            "tensor {key} has shape {shape:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn checked_depth(
    probe: &ModelProbe,
    prefix: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<usize, ModelFamilyError> {
    let depth = probe.consecutive_block_count(&format!("{prefix}{{}}."))?;
    if depth == 0 {
        return Err(invalid(format!("{prefix} has no consecutive block zero")));
    }
    for key in probe
        .tensor_shapes
        .keys()
        .filter(|key| key.starts_with(prefix))
    {
        let ordinal = key[prefix.len()..]
            .split('.')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| invalid(format!("malformed block key {key}")))?;
        if ordinal >= depth {
            return Err(invalid(format!(
                "{prefix} block ordinals are not consecutive before {ordinal}"
            )));
        }
    }
    Ok(depth)
}
