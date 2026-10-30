use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyError, ModelForwardOperation, ModelForwardStep, ModelProbe,
    ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const PIXELDIT_INPUT_CHANNELS: u64 = 3;
pub const PIXELDIT_GROUP_COUNT: u64 = 24;
pub const PIXELDIT_HIDDEN_SIZE: u64 = 1_536;
pub const PIXELDIT_PIXEL_HIDDEN_SIZE: u64 = 16;
pub const PIXELDIT_PIXEL_ATTENTION_HIDDEN_SIZE: u64 = 1_152;
pub const PIXELDIT_PIXEL_GROUP_COUNT: u64 = 16;
pub const PIXELDIT_PATCH_DEPTH: usize = 14;
pub const PIXELDIT_PIXEL_DEPTH: usize = 2;
pub const PIXELDIT_PATCH_SIZE: u64 = 16;
pub const PIXELDIT_TEXT_FEATURE_DIMENSION: u64 = 2_304;
pub const PIXELDIT_TEXT_MAX_LENGTH: u64 = 300;
pub const PIXELDIT_TEXT_ROPE_THETA: f64 = 10_000.0;
pub const PIXELDIT_SAMPLING_SHIFT: f64 = 4.0;
pub const PID_SAMPLING_SHIFT: f64 = 1.5;
pub const PIXELDIT_PID_MEMORY_USAGE_FACTOR: f64 = 0.04;
pub const PID_SR_SCALE: u64 = 4;
pub const PIXELDIT_PID_MAX_BLOCK_COUNT: usize = 64;

pub const PIXELDIT_PID_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_pixelditpixel_sim_model_0042::LATENT_FORMAT;

pub const PIXELDIT_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.pixeldit.PixelDiTGemma2Tokenizer",
        clip_model: "sim.text_encoders.pixeldit.PixelDiTGemma2TE",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
pub static PIXELDIT_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: PIXELDIT_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelDitPidVariant {
    PixelDitT2I,
    PiD,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelDitPidLayout {
    CoreNative,
    NetNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelDitPidConditioningKey {
    AttentionMask,
    LqLatent,
    DegradeSigma,
}

impl PixelDitPidConditioningKey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AttentionMask => "attention_mask",
            Self::LqLatent => "lq_latent",
            Self::DegradeSigma => "degrade_sigma",
        }
    }
}

pub const PIXELDIT_CONDITIONING_KEYS: &[PixelDitPidConditioningKey] =
    &[PixelDitPidConditioningKey::AttentionMask];
pub const PID_CONDITIONING_KEYS: &[PixelDitPidConditioningKey] = &[
    PixelDitPidConditioningKey::AttentionMask,
    PixelDitPidConditioningKey::LqLatent,
    PixelDitPidConditioningKey::DegradeSigma,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PiDConfiguration {
    pub lq_latent_channels: u64,
    pub lq_hidden_dimension: u64,
    pub latent_spatial_down_factor: u64,
    pub lq_gate_count: usize,
    pub lq_interval: usize,
    pub sr_scale: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct PixelDitPidConfiguration {
    pub variant: PixelDitPidVariant,
    pub layout: PixelDitPidLayout,
    pub input_channels: u64,
    pub hidden_size: u64,
    pub group_count: u64,
    pub pixel_hidden_size: u64,
    pub pixel_attention_hidden_size: u64,
    pub pixel_group_count: u64,
    pub patch_depth: usize,
    pub pixel_depth: usize,
    pub patch_size: u64,
    pub text_feature_dimension: u64,
    pub text_max_length: u64,
    pub text_rope_theta: f64,
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub pid: Option<PiDConfiguration>,
    pub conditioning_keys: &'static [PixelDitPidConditioningKey],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
}

pub const PIXELDIT_PID_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "PixelDiT image transformer or PiD pixel diffusion decoder",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "optional source latent codec for PiD low-quality conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "optional Gemma2 PixelDiT text encoder",
        required: false,
    },
];

pub const PIXELDIT_PID_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.pixel_embedder.proj.weight",
    "native.s_embedder.proj.weight",
    "native.y_embedder.proj.weight",
    "native.pixel_blocks.0.adaLN_modulation_msa.weight",
    "native.pixel_blocks.0.adaLN_modulation_mlp.weight",
    "native.final_layer.linear.weight",
];
pub const PIXELDIT_PID_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.lq_proj.latent_proj.0.weight",
    "native.lq_proj.gate_modules.0.content_proj.weight",
];
pub const PIXELDIT_PID_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: PIXELDIT_PID_MODEL_REQUIRED_KEYS,
        optional_keys: PIXELDIT_PID_MODEL_OPTIONAL_KEYS,
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

pub const PIXELDIT_PID_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const PIXELDIT_PID_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const PIXELDIT_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "pixel.embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.pixel_embedder.proj.weight",
            bias: Some("native.pixel_embedder.proj.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "patch.embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.s_embedder.proj.weight",
            bias: Some("native.s_embedder.proj.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "patch.activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "pixel.output",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const PID_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "lq.projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.lq_proj.output_heads.0.weight",
            bias: Some("native.lq_proj.output_heads.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "lq.gate",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "pixel.output",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const PIXELDIT_CORE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"Any":[{"Prefix":"_repa_projector"},{"Prefix":"net_ema."}]},"minimum_matches":0,"maximum_matches":16384}}},
            {"TransformBranchesEach":{"selector":{"predicate":{"All":[{"Prefix":"core."},{"Contains":"pixel_blocks."},{"Contains":".adaLN_modulation.0."},{"Suffix":"weight"}]},"minimum_matches":1,"maximum_matches":64},"pre_transform":{"Reshape":{"shape":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Multiply":[{"Literal":6},{"SourceDimension":{"key":"core.pixel_embedder.proj.weight","dimension":0}}]}]},{"Literal":6},{"SourceDimension":{"key":"core.pixel_embedder.proj.weight","dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]}},"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"core.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_msa."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":0,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]},{"CurrentTensorDimension":{"dimension":3}}]}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"core.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_mlp."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":3,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]},{"CurrentTensorDimension":{"dimension":3}}]}}]}}]}},
	            {"TransformBranchesEach":{"selector":{"predicate":{"All":[{"Prefix":"core."},{"Contains":"pixel_blocks."},{"Contains":".adaLN_modulation.0."},{"Suffix":"bias"}]},"minimum_matches":0,"maximum_matches":64},"pre_transform":{"Reshape":{"shape":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Multiply":[{"Literal":6},{"SourceDimension":{"key":"core.pixel_embedder.proj.weight","dimension":0}}]}]},{"Literal":6},{"SourceDimension":{"key":"core.pixel_embedder.proj.weight","dimension":0}}]}},"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"core.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_msa."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":0,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]}]}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"core.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_mlp."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":3,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]}]}}]}}]}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"core."},{"Not":{"Contains":".adaLN_modulation.0."}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"core.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const PIXELDIT_NET_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"Any":[{"Prefix":"_repa_projector"},{"Prefix":"net_ema."}]},"minimum_matches":0,"maximum_matches":16384}}},
            {"TransformBranchesEach":{"selector":{"predicate":{"All":[{"Prefix":"net."},{"Contains":"pixel_blocks."},{"Contains":".adaLN_modulation.0."},{"Suffix":"weight"}]},"minimum_matches":1,"maximum_matches":64},"pre_transform":{"Reshape":{"shape":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Multiply":[{"Literal":6},{"SourceDimension":{"key":"net.pixel_embedder.proj.weight","dimension":0}}]}]},{"Literal":6},{"SourceDimension":{"key":"net.pixel_embedder.proj.weight","dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]}},"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"net.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_msa."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":0,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]},{"CurrentTensorDimension":{"dimension":3}}]}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"net.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_mlp."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":3,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]},{"CurrentTensorDimension":{"dimension":3}}]}}]}}]}},
	            {"TransformBranchesEach":{"selector":{"predicate":{"All":[{"Prefix":"net."},{"Contains":"pixel_blocks."},{"Contains":".adaLN_modulation.0."},{"Suffix":"bias"}]},"minimum_matches":0,"maximum_matches":64},"pre_transform":{"Reshape":{"shape":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Multiply":[{"Literal":6},{"SourceDimension":{"key":"net.pixel_embedder.proj.weight","dimension":0}}]}]},{"Literal":6},{"SourceDimension":{"key":"net.pixel_embedder.proj.weight","dimension":0}}]}},"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"net.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_msa."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":0,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]}]}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"net.","to":"native."}},{"Contains":{"from":".adaLN_modulation.0.","to":".adaLN_modulation_mlp."}}]},"transform":{"Sequence":[{"Narrow":{"dimension":1,"start":3,"length":3}},{"Reshape":{"shape":[{"Multiply":[{"Multiply":[{"CurrentTensorDimension":{"dimension":0}},{"CurrentTensorDimension":{"dimension":1}}]},{"CurrentTensorDimension":{"dimension":2}}]}]}}]}}]}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"net."},{"Not":{"Contains":".adaLN_modulation.0."}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"net.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub fn state_plan_for_layout(
    layout: PixelDitPidLayout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        PixelDitPidLayout::CoreNative => &PIXELDIT_CORE_STATE_PLAN,
        PixelDitPidLayout::NetNative => &PIXELDIT_NET_STATE_PLAN,
    }
}

pub fn forward_program_for_variant(variant: PixelDitPidVariant) -> &'static [ModelForwardStep] {
    match variant {
        PixelDitPidVariant::PixelDitT2I => PIXELDIT_FORWARD_PROGRAM,
        PixelDitPidVariant::PiD => PID_FORWARD_PROGRAM,
    }
}

pub fn conditioning_keys_for_variant(
    variant: PixelDitPidVariant,
) -> &'static [PixelDitPidConditioningKey] {
    match variant {
        PixelDitPidVariant::PixelDitT2I => PIXELDIT_CONDITIONING_KEYS,
        PixelDitPidVariant::PiD => PID_CONDITIONING_KEYS,
    }
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<PixelDitPidConfiguration, ModelFamilyError> {
    let (layout, prefix) = select_layout(probe)?;
    let pixel = required_matrix(probe, &format!("{prefix}pixel_embedder.proj.weight"))?;
    if pixel != [PIXELDIT_PIXEL_HIDDEN_SIZE, PIXELDIT_INPUT_CHANNELS] {
        return Err(invalid(format!(
            "pixel embedder shape {pixel:?} must be [{PIXELDIT_PIXEL_HIDDEN_SIZE}, {PIXELDIT_INPUT_CHANNELS}]"
        )));
    }
    let patch_input = PIXELDIT_INPUT_CHANNELS
        .checked_mul(PIXELDIT_PATCH_SIZE)
        .and_then(|value| value.checked_mul(PIXELDIT_PATCH_SIZE))
        .ok_or(ModelFamilyError::MemoryOverflow)?;
    require_shape(
        probe,
        &format!("{prefix}s_embedder.proj.weight"),
        &[PIXELDIT_HIDDEN_SIZE, patch_input],
    )?;
    require_shape(
        probe,
        &format!("{prefix}y_embedder.proj.weight"),
        &[PIXELDIT_HIDDEN_SIZE, PIXELDIT_TEXT_FEATURE_DIMENSION],
    )?;
    require_shape(
        probe,
        &format!("{prefix}final_layer.linear.weight"),
        &[PIXELDIT_INPUT_CHANNELS, PIXELDIT_PIXEL_HIDDEN_SIZE],
    )?;
    let patch_depth = checked_block_count(
        probe,
        &format!("{prefix}patch_blocks.{{}}.attn.qkv_x.weight"),
        "patch",
    )?;
    let pixel_depth = checked_block_count(
        probe,
        &format!("{prefix}pixel_blocks.{{}}.adaLN_modulation.0.weight"),
        "pixel",
    )?;
    if patch_depth != PIXELDIT_PATCH_DEPTH || pixel_depth != PIXELDIT_PIXEL_DEPTH {
        return Err(invalid(format!(
            "patch/pixel depths must be {PIXELDIT_PATCH_DEPTH}/{PIXELDIT_PIXEL_DEPTH}, found {patch_depth}/{pixel_depth}"
        )));
    }
    let modulation_width = 6_u64
        .checked_mul(PIXELDIT_PIXEL_HIDDEN_SIZE)
        .and_then(|value| value.checked_mul(PIXELDIT_PATCH_SIZE))
        .and_then(|value| value.checked_mul(PIXELDIT_PATCH_SIZE))
        .ok_or(ModelFamilyError::MemoryOverflow)?;
    for index in 0..pixel_depth {
        require_shape(
            probe,
            &format!("{prefix}pixel_blocks.{index}.adaLN_modulation.0.weight"),
            &[modulation_width, PIXELDIT_HIDDEN_SIZE],
        )?;
    }

    let lq_key = format!("{prefix}lq_proj.latent_proj.0.weight");
    let (variant, pid, sampling_shift, conditioning_keys) = if let Some(lq_shape) =
        probe.tensor_shapes.get(&lq_key)
    {
        if lq_shape.len() != 4 || lq_shape.contains(&0) || lq_shape[2] != 3 || lq_shape[3] != 3 {
            return Err(invalid(format!(
                "{lq_key} shape {lq_shape:?} must be [hidden, channels, 3, 3]"
            )));
        }
        let gate_count = checked_block_count(
            probe,
            &format!("{prefix}lq_proj.gate_modules.{{}}.content_proj.weight"),
            "PiD gate",
        )?;
        let interval = PIXELDIT_PATCH_DEPTH
            .checked_add(gate_count - 1)
            .ok_or(ModelFamilyError::MemoryOverflow)?
            / gate_count;
        let channels = lq_shape[1];
        (
            PixelDitPidVariant::PiD,
            Some(PiDConfiguration {
                lq_latent_channels: channels,
                lq_hidden_dimension: lq_shape[0],
                latent_spatial_down_factor: if channels >= 64 { 16 } else { 8 },
                lq_gate_count: gate_count,
                lq_interval: interval,
                sr_scale: PID_SR_SCALE,
            }),
            PID_SAMPLING_SHIFT,
            PID_CONDITIONING_KEYS,
        )
    } else {
        (
            PixelDitPidVariant::PixelDitT2I,
            None,
            PIXELDIT_SAMPLING_SHIFT,
            PIXELDIT_CONDITIONING_KEYS,
        )
    };

    Ok(PixelDitPidConfiguration {
        variant,
        layout,
        input_channels: PIXELDIT_INPUT_CHANNELS,
        hidden_size: PIXELDIT_HIDDEN_SIZE,
        group_count: PIXELDIT_GROUP_COUNT,
        pixel_hidden_size: PIXELDIT_PIXEL_HIDDEN_SIZE,
        pixel_attention_hidden_size: PIXELDIT_PIXEL_ATTENTION_HIDDEN_SIZE,
        pixel_group_count: PIXELDIT_PIXEL_GROUP_COUNT,
        patch_depth,
        pixel_depth,
        patch_size: PIXELDIT_PATCH_SIZE,
        text_feature_dimension: PIXELDIT_TEXT_FEATURE_DIMENSION,
        text_max_length: PIXELDIT_TEXT_MAX_LENGTH,
        text_rope_theta: PIXELDIT_TEXT_ROPE_THETA,
        sampling_shift,
        memory_usage_factor: PIXELDIT_PID_MEMORY_USAGE_FACTOR,
        pid,
        conditioning_keys,
        supported_dtypes: PIXELDIT_PID_SUPPORTED_DTYPES,
        supported_devices: PIXELDIT_PID_SUPPORTED_DEVICES,
        latent_format: PIXELDIT_PID_LATENT_FORMAT,
        clip_target: &PIXELDIT_CLIP_TARGET,
    })
}

fn select_layout(
    probe: &ModelProbe,
) -> Result<(PixelDitPidLayout, &'static str), ModelFamilyError> {
    let candidates = [
        (PixelDitPidLayout::CoreNative, "core."),
        (PixelDitPidLayout::NetNative, "net."),
    ];
    let matches = candidates
        .into_iter()
        .filter(|(_, prefix)| {
            [
                "pixel_embedder.proj.weight",
                "s_embedder.proj.weight",
                "y_embedder.proj.weight",
                "patch_blocks.0.attn.qkv_x.weight",
                "pixel_blocks.0.adaLN_modulation.0.weight",
                "final_layer.linear.weight",
            ]
            .iter()
            .all(|key| probe.tensor_shapes.contains_key(&format!("{prefix}{key}")))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [selected] => Ok(*selected),
        [] => Err(ModelFamilyError::ModelLayoutSelection(
            "no exact PixelDiT/PiD core or net native layout matched".to_owned(),
        )),
        _ => Err(ModelFamilyError::ModelLayoutSelection(
            "PixelDiT/PiD probe ambiguously matches core and net native layouts".to_owned(),
        )),
    }
}

fn checked_block_count(
    probe: &ModelProbe,
    pattern: &str,
    label: &str,
) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    if count == 0 || count > PIXELDIT_PID_MAX_BLOCK_COUNT {
        return Err(invalid(format!(
            "{label} block count {count} is outside 1..={PIXELDIT_PID_MAX_BLOCK_COUNT}"
        )));
    }
    let (stem, suffix) = pattern
        .split_once("{}")
        .ok_or_else(|| invalid(format!("{label} block pattern has no placeholder")))?;
    let has_gap_or_later = probe.tensor_shapes.keys().any(|key| {
        key.strip_prefix(stem)
            .and_then(|tail| tail.strip_suffix(suffix))
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= count)
    });
    if has_gap_or_later {
        return Err(invalid(format!(
            "{label} blocks are not a consecutive bounded sequence"
        )));
    }
    Ok(count)
}

fn required_matrix<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != 2 || shape.contains(&0) {
        return Err(invalid(format!(
            "{key} shape {shape:?} is not a non-empty matrix"
        )));
    }
    Ok(shape)
}

fn require_shape(probe: &ModelProbe, key: &str, expected: &[u64]) -> Result<(), ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape != expected {
        return Err(invalid(format!(
            "{key} shape {shape:?} must be {expected:?}"
        )));
    }
    Ok(())
}

fn invalid(message: String) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "PixelDiT/PiD configuration is invalid: {message}"
    ))
}
