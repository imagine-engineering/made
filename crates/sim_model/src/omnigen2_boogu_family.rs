use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelDetectionRule,
    ModelFamilyComponent, ModelFamilyComponentStateSchema, ModelFamilyError, ModelForwardOperation,
    ModelForwardStep, ModelProbe, ModelStateTransformPlanDefinition,
};
use sim_tensor::{BackendCapabilityMatrix, DType};
use sim_types::DeviceKind;

pub const OMNIGEN2_HIDDEN_SIZE: u64 = 2_520;
pub const OMNIGEN2_LAYER_COUNT: usize = 32;
pub const OMNIGEN2_REFINER_LAYER_COUNT: usize = 2;
pub const OMNIGEN2_HEAD_COUNT: u64 = 21;
pub const OMNIGEN2_KV_HEAD_COUNT: u64 = 7;
pub const OMNIGEN2_TEXT_FEATURE_DIMENSION: u64 = 2_048;
pub const OMNIGEN2_PATCH_SIZE: u64 = 2;
pub const OMNIGEN2_INPUT_CHANNELS: u64 = 16;
pub const OMNIGEN2_MULTIPLE_OF: u64 = 256;
pub const OMNIGEN2_TIMESTEP_SCALE: f64 = 1_000.0;
pub const OMNIGEN2_SAMPLING_SHIFT: f64 = 2.6;
pub const OMNIGEN2_MEMORY_USAGE_FACTOR: f64 = 1.95;
pub const OMNIGEN2_AXES_DIMENSIONS: &[u64] = &[40, 40, 40];
pub const OMNIGEN2_AXES_LENGTHS: &[u64] = &[1_024, 1_664, 1_664];

pub const BOOGU_HEAD_COUNT: u64 = 28;
pub const BOOGU_KV_HEAD_COUNT: u64 = 7;
pub const BOOGU_PATCH_SIZE: u64 = 2;
pub const BOOGU_INPUT_CHANNELS: u64 = 16;
pub const BOOGU_MULTIPLE_OF: u64 = 256;
pub const BOOGU_TIMESTEP_SCALE: f64 = 1_000.0;
pub const BOOGU_SAMPLING_SHIFT: f64 = 3.16;
pub const BOOGU_MEMORY_USAGE_FACTOR: f64 = 2.15;
pub const BOOGU_AXES_DIMENSIONS: &[u64] = &[40, 40, 40];
pub const BOOGU_AXES_LENGTHS: &[u64] = &[2_048, 1_664, 1_664];
pub const OMNIGEN2_BOOGU_MAX_LAYER_COUNT: usize = 256;

pub const OMNIGEN2_BOOGU_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_flux_sim_model_0029::LATENT_FORMAT;

pub const OMNIGEN2_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const OMNIGEN2_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.omnigen2.Omnigen2Tokenizer",
        clip_model: "sim.text_encoders.omnigen2.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: OMNIGEN2_CLIP_CONFIGURATION,
        },
    }];
pub static OMNIGEN2_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: OMNIGEN2_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const BOOGU_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const BOOGU_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.boogu.BooguTokenizer",
        clip_model: "sim.text_encoders.boogu.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: BOOGU_CLIP_CONFIGURATION,
        },
    }];
pub static BOOGU_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: BOOGU_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Omnigen2BooguVariant {
    Omnigen2,
    Boogu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Omnigen2BooguLayout {
    PrefixedNative,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Omnigen2BooguConditioningFact {
    OptionalAttentionMask,
    OmitAllOnesAttentionMask,
    NumTokensFromMaskAtLeastOne,
    CrossAttention,
    OptionalReferenceLatents,
    ProcessReferenceLatentsWithCanonicalFlux,
    ReferenceLatentsAffectMemoryEstimate,
}

pub const OMNIGEN2_BOOGU_CONDITIONING: &[Omnigen2BooguConditioningFact] = &[
    Omnigen2BooguConditioningFact::OptionalAttentionMask,
    Omnigen2BooguConditioningFact::OmitAllOnesAttentionMask,
    Omnigen2BooguConditioningFact::NumTokensFromMaskAtLeastOne,
    Omnigen2BooguConditioningFact::CrossAttention,
    Omnigen2BooguConditioningFact::OptionalReferenceLatents,
    Omnigen2BooguConditioningFact::ProcessReferenceLatentsWithCanonicalFlux,
    Omnigen2BooguConditioningFact::ReferenceLatentsAffectMemoryEstimate,
];

#[derive(Clone, Copy, Debug)]
pub struct Omnigen2BooguConfiguration {
    pub variant: Omnigen2BooguVariant,
    pub layout: Omnigen2BooguLayout,
    pub hidden_size: u64,
    pub number_of_layers: usize,
    pub number_of_double_stream_layers: usize,
    pub number_of_refiner_layers: usize,
    pub number_of_attention_heads: u64,
    pub number_of_kv_heads: u64,
    pub instruction_feature_dimension: u64,
    pub patch_size: u64,
    pub input_channels: u64,
    pub output_channels: u64,
    pub multiple_of: u64,
    pub axes_dimensions: &'static [u64],
    pub axes_lengths: &'static [u64],
    pub timestep_scale: f64,
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub base_supported_dtypes: &'static [DType],
    pub conditioning: &'static [Omnigen2BooguConditioningFact],
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub memory_estimator: MemoryEstimatorDescriptor,
}

pub const OMNIGEN2_BOOGU_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "OmniGen2 or Boogu native image diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "generated bounded reference-latent conditioning metadata",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "optional canonical Flux latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Qwen text or multimodal conditioning encoder",
        required: false,
    },
];

pub const OMNIGEN2_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.x_embedder.weight",
    "native.x_embedder.bias",
    "native.time_caption_embed.timestep_embedder.linear_1.bias",
    "native.layers.0.attn.to_q.weight",
    "native.norm_out.linear_2.weight",
    "native.norm_out.linear_2.bias",
];
pub const OMNIGEN2_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.ref_image_patch_embedder.weight",
    "native.noise_refiner.0.attn.to_q.weight",
    "native.ref_image_refiner.0.attn.to_q.weight",
    "native.context_refiner.0.attn.to_q.weight",
    "native.image_index_embedding",
];
pub const BOOGU_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.x_embedder.weight",
    "native.x_embedder.bias",
    "native.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
    "native.norm_out.linear_2.weight",
    "native.norm_out.linear_2.bias",
];
pub const BOOGU_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.time_caption_embed.caption_embedder.0.weight",
    "native.single_stream_layers.0.attn.to_q.weight",
    "native.noise_refiner.0.attn.to_q.weight",
    "native.ref_image_refiner.0.attn.to_q.weight",
    "native.context_refiner.0.attn.to_q.weight",
    "native.image_index_embedding",
];

pub const BOOGU_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: BOOGU_MODEL_REQUIRED_KEYS,
        optional_keys: BOOGU_MODEL_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &["reference_latent_count"],
        optional_keys: &[],
        allow_unexpected: false,
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

pub const OMNIGEN2_BASE_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const OMNIGEN2_EXTENDED_SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const BOOGU_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const OMNIGEN2_BOOGU_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const OMNIGEN2_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 4,
    activation_bytes_per_element: 8,
};
pub const BOOGU_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 4,
    activation_bytes_per_element: 9,
};

pub const OMNIGEN2_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "patch_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.x_embedder.weight",
            bias: Some("native.x_embedder.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "single_stream_query",
        operation: ModelForwardOperation::Linear {
            weight: "native.layers.0.attn.to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "single_stream_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.norm_out.linear_2.weight",
            bias: Some("native.norm_out.linear_2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "negated_output",
        operation: ModelForwardOperation::MultiplyScalar(-1.0),
    },
];

pub const BOOGU_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "patch_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.x_embedder.weight",
            bias: Some("native.x_embedder.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "double_stream_query",
        operation: ModelForwardOperation::Linear {
            weight: "native.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "double_stream_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.norm_out.linear_2.weight",
            bias: Some("native.norm_out.linear_2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "negated_output",
        operation: ModelForwardOperation::MultiplyScalar(-1.0),
    },
];

pub const OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"unsigned":0},"dtype":"i64","output":{"component":"runtime_conditioning","key":"reference_latent_count"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Any":[{"Prefix":"x_embedder."},{"Prefix":"ref_image_patch_embedder."},{"Prefix":"time_caption_embed."},{"Prefix":"noise_refiner."},{"Prefix":"ref_image_refiner."},{"Prefix":"context_refiner."},{"Prefix":"layers."},{"Prefix":"double_stream_layers."},{"Prefix":"single_stream_layers."},{"Prefix":"norm_out."},{"Prefix":"image_index_embedding"}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"OrderedOptional":[{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},{"Prefix":{"from":"ref_image_patch_embedder.","to":"native.ref_image_patch_embedder."}},{"Prefix":{"from":"time_caption_embed.","to":"native.time_caption_embed."}},{"Prefix":{"from":"noise_refiner.","to":"native.noise_refiner."}},{"Prefix":{"from":"ref_image_refiner.","to":"native.ref_image_refiner."}},{"Prefix":{"from":"context_refiner.","to":"native.context_refiner."}},{"Prefix":{"from":"layers.","to":"native.layers."}},{"Prefix":{"from":"double_stream_layers.","to":"native.double_stream_layers."}},{"Prefix":{"from":"single_stream_layers.","to":"native.single_stream_layers."}},{"Prefix":{"from":"norm_out.","to":"native.norm_out."}},{"Prefix":{"from":"image_index_embedding","to":"native.image_index_embedding"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"unsigned":0},"dtype":"i64","output":{"component":"runtime_conditioning","key":"reference_latent_count"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const BOOGU_DETECTION_MARKER_KEYS: &[&str] = &[
    "model.diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
    "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
];
pub const OMNIGEN2_DETECTION_MARKER_KEYS: &[&str] = &[
    "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
    "time_caption_embed.timestep_embedder.linear_1.bias",
];
pub const BOOGU_DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::AnyKeyPresent {
    keys: BOOGU_DETECTION_MARKER_KEYS,
    score: 1_000,
}];

pub fn state_plan_for_layout(
    layout: Omnigen2BooguLayout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        Omnigen2BooguLayout::PrefixedNative => &OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN,
        Omnigen2BooguLayout::StandaloneNative => &OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN,
    }
}

pub fn supported_dtypes_for_capabilities(
    variant: Omnigen2BooguVariant,
    capabilities: &BackendCapabilityMatrix,
) -> &'static [DType] {
    match variant {
        Omnigen2BooguVariant::Omnigen2 if capabilities.supports_dtype(DType::F16) => {
            OMNIGEN2_EXTENDED_SUPPORTED_DTYPES
        }
        Omnigen2BooguVariant::Omnigen2 => OMNIGEN2_BASE_SUPPORTED_DTYPES,
        Omnigen2BooguVariant::Boogu => BOOGU_SUPPORTED_DTYPES,
    }
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Omnigen2BooguConfiguration, ModelFamilyError> {
    let (layout, prefix) = select_layout(probe)?;
    let boogu_marker =
        format!("{prefix}double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight");
    let variant = if probe.tensor_shapes.contains_key(&boogu_marker) {
        Omnigen2BooguVariant::Boogu
    } else {
        Omnigen2BooguVariant::Omnigen2
    };
    let x_key = format!("{prefix}x_embedder.weight");
    let x = required_matrix(probe, &x_key)?;
    let hidden_size = x[0];
    let expected_input_width = OMNIGEN2_PATCH_SIZE
        .checked_mul(OMNIGEN2_PATCH_SIZE)
        .and_then(|value| value.checked_mul(OMNIGEN2_INPUT_CHANNELS))
        .ok_or(ModelFamilyError::MemoryOverflow)?;
    if x[1] != expected_input_width {
        return Err(invalid(format!(
            "{x_key} input width {} must equal {expected_input_width}",
            x[1]
        )));
    }
    let norm_key = format!("{prefix}norm_out.linear_2.weight");
    let norm = required_matrix(probe, &norm_key)?;
    if norm != [expected_input_width, hidden_size] {
        return Err(invalid(format!(
            "{norm_key} shape {norm:?} must be [{expected_input_width}, {hidden_size}]"
        )));
    }

    let (
        number_of_layers,
        number_of_double_stream_layers,
        number_of_refiner_layers,
        number_of_attention_heads,
        number_of_kv_heads,
        instruction_feature_dimension,
        axes_dimensions,
        axes_lengths,
        sampling_shift,
        memory_usage_factor,
        base_supported_dtypes,
        clip_target,
        memory_estimator,
    ) = match variant {
        Omnigen2BooguVariant::Boogu => {
            let layers = checked_block_count(
                probe,
                &format!("{prefix}single_stream_layers.{{}}.attn.to_q.weight"),
                "Boogu single-stream",
            )?;
            let double_layers = checked_block_count(
                probe,
                &format!(
                    "{prefix}double_stream_layers.{{}}.img_instruct_attn.processor.img_to_q.weight"
                ),
                "Boogu double-stream",
            )?;
            let refiners = checked_block_count(
                probe,
                &format!("{prefix}noise_refiner.{{}}.attn.to_q.weight"),
                "Boogu refiner",
            )?;
            let instruction_key = format!("{prefix}time_caption_embed.caption_embedder.0.weight");
            let instruction = required_matrix(probe, &instruction_key)?;
            (
                layers,
                double_layers,
                refiners,
                BOOGU_HEAD_COUNT,
                BOOGU_KV_HEAD_COUNT,
                instruction[0],
                BOOGU_AXES_DIMENSIONS,
                BOOGU_AXES_LENGTHS,
                BOOGU_SAMPLING_SHIFT,
                BOOGU_MEMORY_USAGE_FACTOR,
                BOOGU_SUPPORTED_DTYPES,
                &BOOGU_CLIP_TARGET,
                BOOGU_MEMORY_ESTIMATOR,
            )
        }
        Omnigen2BooguVariant::Omnigen2 => {
            let layers = checked_block_count(
                probe,
                &format!("{prefix}layers.{{}}.attn.to_q.weight"),
                "OmniGen2 transformer",
            )?;
            let refiners = checked_block_count(
                probe,
                &format!("{prefix}noise_refiner.{{}}.attn.to_q.weight"),
                "OmniGen2 refiner",
            )?;
            if hidden_size != OMNIGEN2_HIDDEN_SIZE
                || layers != OMNIGEN2_LAYER_COUNT
                || refiners != OMNIGEN2_REFINER_LAYER_COUNT
            {
                return Err(invalid(format!(
                    "OmniGen2 requires hidden/layer/refiner values {OMNIGEN2_HIDDEN_SIZE}/{OMNIGEN2_LAYER_COUNT}/{OMNIGEN2_REFINER_LAYER_COUNT}, found {hidden_size}/{layers}/{refiners}"
                )));
            }
            (
                layers,
                0,
                refiners,
                OMNIGEN2_HEAD_COUNT,
                OMNIGEN2_KV_HEAD_COUNT,
                OMNIGEN2_TEXT_FEATURE_DIMENSION,
                OMNIGEN2_AXES_DIMENSIONS,
                OMNIGEN2_AXES_LENGTHS,
                OMNIGEN2_SAMPLING_SHIFT,
                OMNIGEN2_MEMORY_USAGE_FACTOR,
                OMNIGEN2_BASE_SUPPORTED_DTYPES,
                &OMNIGEN2_CLIP_TARGET,
                OMNIGEN2_MEMORY_ESTIMATOR,
            )
        }
    };
    if !hidden_size.is_multiple_of(number_of_attention_heads) {
        return Err(invalid(format!(
            "hidden size {hidden_size} is not divisible by {number_of_attention_heads} heads"
        )));
    }
    if number_of_kv_heads == 0 || !number_of_attention_heads.is_multiple_of(number_of_kv_heads) {
        return Err(invalid(format!(
            "{number_of_attention_heads} attention heads are not grouped by {number_of_kv_heads} KV heads"
        )));
    }

    Ok(Omnigen2BooguConfiguration {
        variant,
        layout,
        hidden_size,
        number_of_layers,
        number_of_double_stream_layers,
        number_of_refiner_layers,
        number_of_attention_heads,
        number_of_kv_heads,
        instruction_feature_dimension,
        patch_size: OMNIGEN2_PATCH_SIZE,
        input_channels: OMNIGEN2_INPUT_CHANNELS,
        output_channels: OMNIGEN2_INPUT_CHANNELS,
        multiple_of: OMNIGEN2_MULTIPLE_OF,
        axes_dimensions,
        axes_lengths,
        timestep_scale: OMNIGEN2_TIMESTEP_SCALE,
        sampling_shift,
        memory_usage_factor,
        base_supported_dtypes,
        conditioning: OMNIGEN2_BOOGU_CONDITIONING,
        latent_format: OMNIGEN2_BOOGU_LATENT_FORMAT,
        clip_target,
        memory_estimator,
    })
}

fn select_layout(
    probe: &ModelProbe,
) -> Result<(Omnigen2BooguLayout, &'static str), ModelFamilyError> {
    let candidates = [
        (
            Omnigen2BooguLayout::PrefixedNative,
            "model.diffusion_model.",
        ),
        (Omnigen2BooguLayout::StandaloneNative, ""),
    ];
    let matches = candidates
        .into_iter()
        .filter(|(_, prefix)| {
            let common = ["x_embedder.weight", "norm_out.linear_2.weight"]
                .iter()
                .all(|key| probe.tensor_shapes.contains_key(&format!("{prefix}{key}")));
            let boogu = probe.tensor_shapes.contains_key(&format!(
                "{prefix}double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight"
            ));
            let omnigen = probe.tensor_shapes.contains_key(&format!(
                "{prefix}time_caption_embed.timestep_embedder.linear_1.bias"
            ));
            common && (boogu || omnigen)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [selected] => Ok(*selected),
        [] => Err(ModelFamilyError::ModelLayoutSelection(
            "no exact OmniGen2/Boogu source-native layout matched".to_owned(),
        )),
        _ => Err(ModelFamilyError::ModelLayoutSelection(
            "OmniGen2/Boogu probe ambiguously matches prefixed and standalone native layouts"
                .to_owned(),
        )),
    }
}

fn checked_block_count(
    probe: &ModelProbe,
    pattern: &str,
    label: &str,
) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    if count == 0 || count > OMNIGEN2_BOOGU_MAX_LAYER_COUNT {
        return Err(invalid(format!(
            "{label} block count {count} is outside 1..={OMNIGEN2_BOOGU_MAX_LAYER_COUNT}"
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

fn invalid(message: String) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "OmniGen2/Boogu configuration is invalid: {message}"
    ))
}
