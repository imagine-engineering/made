use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelStateLayout, ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "SD15";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0117";
pub const MODEL_FAMILY_FIXTURE: &str = "sd15-comfy-model-0117";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 3;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "f12de62cf3cb37ecf5d9f353a6a847e1181f43a63511d0dda11e2802fb78898b";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.0;
pub const SOURCE_MODEL_CHANNELS: u64 = 320;
pub const SOURCE_CONTEXT_DIMENSION: u64 = 768;
pub const SOURCE_INPUT_CHANNELS: u64 = 4;
pub const SOURCE_ATTENTION_HEADS: u64 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sd15Configuration {
    pub layout: ModelStateLayout,
    pub in_channels: u64,
    pub model_channels: u64,
    pub context_dimension: u64,
    pub attention_heads: u64,
    pub uses_linear_transformer_projection: bool,
    pub uses_temporal_attention: bool,
    pub adm_in_channels: Option<u64>,
}

pub const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "SD 1.x latent diffusion U-Net",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SD 1 CLIP-L conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "SD 1 latent codec",
        required: false,
    },
];

pub const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sd1_clip.SD1Tokenizer",
        clip_model: "sd1_clip.SD1ClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
pub const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const INPUT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.0.0.weight",
    "conv_in.weight",
];
const CONTEXT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
];
const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: INPUT_KEYS,
        score: 350,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 0,
        values: &[SOURCE_MODEL_CHANNELS],
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 1,
        values: &[SOURCE_INPUT_CHANNELS],
        score: 400,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: CONTEXT_KEYS,
        dimension: 1,
        values: &[SOURCE_CONTEXT_DIMENSION],
        score: 200,
    },
];

pub const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "model.diffusion_model.",
        target_prefix: "native.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "cond_stage_model.",
        target_prefix: "clip_l.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "first_stage_model.",
        target_prefix: "native.",
        required: false,
    },
];

pub const REQUIRED_KEYS: &[&str] = &[
    "native.input_blocks.0.0.weight",
    "native.time_embed.0.weight",
    "native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
    "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
    "native.out.2.weight",
];
pub const OPTIONAL_KEYS: &[&str] = &[
    "native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "native.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight",
    "native.time_embed.0.bias",
    "native.out.2.bias",
];
pub const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "unet_input_attention",
        operation: ModelForwardOperation::Linear {
            weight: "native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "cross_attention_conditioning",
        operation: ModelForwardOperation::Linear {
            weight: "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "unet_residual_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "epsilon_prediction",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sd15-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: SUPPORTED_DTYPES,
    supported_devices: SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: FORWARD_PROGRAM,
};

pub const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"TransformEach":{"selector":{"predicate":{"Exact":"cond_stage_model.transformer.embeddings.position_ids"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_l.transformer.text_model.embeddings.position_ids"},"component":"text_encoder","transform":{"ConditionalRound":{"decimals":0,"condition":{"DType":"f32"}}}}},
            {"TransformEach":{"selector":{"predicate":{"Exact":"cond_stage_model.transformer.text_model.embeddings.position_ids"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_l.transformer.text_model.embeddings.position_ids"},"component":"text_encoder","transform":{"ConditionalRound":{"decimals":0,"condition":{"DType":"f32"}}}}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"cond_stage_model.transformer.text_model."},{"Not":{"Exact":"cond_stage_model.transformer.text_model.embeddings.position_ids"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.transformer.text_model.","to":"clip_l.transformer.text_model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"cond_stage_model.transformer."},{"Not":{"Prefix":"cond_stage_model.transformer.text_model."}},{"Not":{"Exact":"cond_stage_model.transformer.embeddings.position_ids"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.transformer.","to":"clip_l.transformer.text_model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"cond_stage_model."},{"Not":{"Prefix":"cond_stage_model.transformer."}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"clip_l."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Exact":"conv_in.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.0.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_embedding.linear_1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.time_embed.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.middle_block.1.transformer_blocks.0.attn2.to_q.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.out.2.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoder.","to":"clip_l."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &DIFFUSERS_STATE_PLAN,
    },
];

pub const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.out.2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
            "conv_out.weight",
        ],
        required_prefixes: &[],
    },
];

pub const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "vae",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 3,
    source_architecture: "model_base.BaseModel",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Sd15Configuration, ModelFamilyError> {
    configuration_for_probe_kind(
        probe,
        SOURCE_INPUT_CHANNELS,
        MODEL_FAMILY_IDENTIFIER,
    )
}

pub fn configuration_for_probe_kind(
    probe: &ModelProbe,
    expected_input_channels: u64,
    family: &str,
) -> Result<Sd15Configuration, ModelFamilyError> {
    let layout = probe.select_layout(LAYOUT_SIGNATURES)?;
    let (input_key, context_key) = match layout {
        ModelStateLayout::PrefixedNative => (
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
        ),
        ModelStateLayout::Diffusers => (
            "conv_in.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        ),
        ModelStateLayout::StandaloneNative => {
            return Err(invalid(family, "standalone-native layout is unsupported"));
        }
    };
    let input = shape(probe, input_key, 4, family)?;
    if input[0] != SOURCE_MODEL_CHANNELS || input[1] != expected_input_channels {
        return Err(invalid(
            family,
            format!(
                "{input_key} begins [{}, {}], expected [{SOURCE_MODEL_CHANNELS}, {expected_input_channels}]",
                input[0], input[1]
            ),
        ));
    }
    let context = shape(probe, context_key, 2, family)?;
    if context[1] != SOURCE_CONTEXT_DIMENSION {
        return Err(invalid(
            family,
            format!(
                "{context_key} context width is {}, expected {SOURCE_CONTEXT_DIMENSION}",
                context[1]
            ),
        ));
    }
    Ok(Sd15Configuration {
        layout,
        in_channels: expected_input_channels,
        model_channels: SOURCE_MODEL_CHANNELS,
        context_dimension: SOURCE_CONTEXT_DIMENSION,
        attention_heads: SOURCE_ATTENTION_HEADS,
        uses_linear_transformer_projection: false,
        uses_temporal_attention: false,
        adm_in_channels: None,
    })
}

fn shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    rank: usize,
    family: &str,
) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| invalid(family, format!("missing {key}")))?;
    if shape.len() != rank || shape.contains(&0) {
        return Err(invalid(
            family,
            format!("{key} must have non-zero rank {rank}"),
        ));
    }
    Ok(shape)
}

fn invalid(family: &str, message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "{family} source configuration mismatch: {}",
        message.into()
    ))
}
