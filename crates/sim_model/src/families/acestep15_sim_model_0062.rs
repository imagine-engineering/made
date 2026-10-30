use crate::{
    MemoryEstimatorDescriptor, ModelClipConfigurationFactDefinition,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "ACEStep15";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0062";
pub const MODEL_FAMILY_FIXTURE: &str = "acestep15-comfy-model-0062";
pub const SOURCE_ARCHITECTURE: &str = "model_base.ACEStep15";
pub const SOURCE_MEMORY_USAGE_FACTOR: f64 = 4.7;
pub const SOURCE_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const SOURCE_SAMPLING_SHIFT: f64 = 3.0;

const COMPONENTS: [ModelFamilyComponent; 3] = [
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "diffusion",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "latent_codec",
        required: false,
    },
];

const DETECTION_RULES: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &[
        "model.diffusion_model.encoder.lyric_encoder.layers.0.input_layernorm.weight",
        "encoder.lyric_encoder.layers.0.input_layernorm.weight",
    ],
    score: 1_000,
}];

const WEIGHT_RULES: [ModelWeightRule; 3] = [
    ModelWeightRule {
        source_prefix: "decoder.",
        target_prefix: "decoder.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "text_encoders.",
        target_prefix: "model.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "vae.",
        target_prefix: "model.",
        required: false,
    },
];

const REQUIRED_KEYS: [&str; 2] = [
    "decoder.condition_embedder.weight",
    "decoder.norm_out.weight",
];

const OPTIONAL_KEYS: [&str; 6] = [
    "decoder.layers.0.self_attn_norm.weight",
    "decoder.layers.0.self_attn.q_proj.weight",
    "decoder.layers.0.mlp.gate_proj.weight",
    "decoder.layers.1.self_attn_norm.weight",
    "encoder.lyric_encoder.layers.0.input_layernorm.weight",
    "encoder.lyric_encoder.layers.0.self_attn.q_proj.weight",
];

const SUPPORTED_DTYPES: [DType; 2] = [DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: [DeviceKind; 1] = [DeviceKind::Cpu];

const CLIP_FACTORY_CONFIGURATION_2B: [ModelClipConfigurationFactDefinition; 1] =
    [ModelClipConfigurationFactDefinition::Expand {
        source: "detect_qwen3_2b",
    }];
const CLIP_FACTORY_CONFIGURATION_4B: [ModelClipConfigurationFactDefinition; 1] =
    [ModelClipConfigurationFactDefinition::Expand {
        source: "detect_qwen3_4b",
    }];
const CLIP_CANDIDATE_2B: [ModelClipTargetCandidateDefinition; 1] =
    [ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.ace15.ACE15Tokenizer",
        clip_model: "sim.text_encoders.ace15.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: &CLIP_FACTORY_CONFIGURATION_2B,
        },
    }];
const CLIP_CANDIDATE_4B: [ModelClipTargetCandidateDefinition; 1] =
    [ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.ace15.ACE15Tokenizer",
        clip_model: "sim.text_encoders.ace15.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: &CLIP_FACTORY_CONFIGURATION_4B,
        },
    }];
const CLIP_CANDIDATES_DYNAMIC: [ModelClipTargetCandidateDefinition; 2] = [
    CLIP_CANDIDATE_2B[0],
    CLIP_CANDIDATE_4B[0],
];
static CLIP_TARGET_2B: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &CLIP_CANDIDATE_2B,
    dynamic_selection: false,
};
static CLIP_TARGET_4B: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &CLIP_CANDIDATE_4B,
    dynamic_selection: false,
};
static CLIP_TARGET_DYNAMIC: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &CLIP_CANDIDATES_DYNAMIC,
    dynamic_selection: true,
};

const FORWARD_PROGRAM: [ModelForwardStep; 4] = [
    ModelForwardStep {
        checkpoint: "decoder-input-projection",
        operation: ModelForwardOperation::Linear {
            weight: "decoder.condition_embedder.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "decoder-output-normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: Some("decoder.norm_out.weight"),
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "decoder-gated-activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "decoder-self-attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "ace-step-1.5-v1",
    latent_feature_id: "COMFY-MODEL-0024",
    latent_identifier: "ACEAudio15",
    clip_target: &CLIP_TARGET_DYNAMIC,
    components: &COMPONENTS,
    detection_rules: &DETECTION_RULES,
    weight_rules: &WEIGHT_RULES,
    required_keys: &REQUIRED_KEYS,
    optional_keys: &OPTIONAL_KEYS,
    supported_dtypes: &SUPPORTED_DTYPES,
    supported_devices: &SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 5,
        activation_bytes_per_element: 5,
    },
    forward_program: &FORWARD_PROGRAM,
};

static PREFIXED_NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: 1,
        encoded_plan: r#"{
        "operations":[
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":""}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

static UNPREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: 1,
    encoded_plan: r#"{
        "operations":[
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}}
        ],
        "unmatched":{"Route":{"component":"denoiser","rewrite":"Identity"}}
    }"#,
};

const STATE_PLAN_CASES: [ModelFamilyStatePlanCase; 2] = [
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &PREFIXED_NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &UNPREFIXED_STATE_PLAN,
    },
];

const LAYOUT_SIGNATURES: [ModelLayoutSignature; 2] = [
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.encoder.lyric_encoder.layers.0.input_layernorm.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["encoder.lyric_encoder.layers.0.input_layernorm.weight"],
        required_prefixes: &[],
    },
];

const SOURCE_CONFIGURATION: [ModelSourceConfigurationRule; 0] = [];

const REQUIRED_STATE_KEYS: [&str; 5] = [
    "decoder.condition_embedder.weight",
    "decoder.norm_out.weight",
    "decoder.layers.0.self_attn_norm.weight",
    "decoder.layers.0.self_attn.q_proj.weight",
    "encoder.lyric_encoder.layers.0.input_layernorm.weight",
];

const COMPONENT_SCHEMAS: [ModelFamilyComponentStateSchema; 3] = [
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: &REQUIRED_STATE_KEYS,
        optional_keys: &[],
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
    source_ordinal: 74,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: &SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: crate::ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: &LAYOUT_SIGNATURES,
        cases: &STATE_PLAN_CASES,
    },
    component_state_schemas: &COMPONENT_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    validate_source_configuration(probe)?;
    let qwen_2b = has_llama_dtype_marker(probe, "text_encoders.qwen3_2b.transformer.");
    let qwen_4b = has_llama_dtype_marker(probe, "text_encoders.qwen3_4b.transformer.");
    let clip_target = match (qwen_2b, qwen_4b) {
        (true, false) => &CLIP_TARGET_2B,
        (false, true) => &CLIP_TARGET_4B,
        (false, false) => &CLIP_TARGET_DYNAMIC,
        (true, true) => {
            return Err(ModelFamilyError::InvalidSelectorOutput(
                "ACE-Step 1.5 checkpoint contains both qwen3_2b and qwen3_4b dtype markers"
                    .to_owned(),
            ));
        }
    };
    Ok(ModelFamilyProfile {
        latent_feature_id: MODEL_FAMILY.latent_feature_id,
        latent_identifier: MODEL_FAMILY.latent_identifier,
        clip_target,
        supported_dtypes: MODEL_FAMILY.supported_dtypes,
        supported_devices: MODEL_FAMILY.supported_devices,
        memory_estimator: MODEL_FAMILY.memory_estimator,
        forward_program: MODEL_FAMILY.forward_program,
    })
}

fn has_llama_dtype_marker(probe: &ModelProbe, prefix: &str) -> bool {
    ["model.norm.weight", "model.layers.0.input_layernorm.weight"]
        .iter()
        .any(|suffix| probe.tensor_shapes().contains_key(&format!("{prefix}{suffix}")))
}

fn validate_source_configuration(probe: &ModelProbe) -> Result<(), ModelFamilyError> {
    let prefix = match probe.select_layout(&LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => "model.diffusion_model.",
        ModelStateLayout::Diffusers => "",
        layout => {
            return Err(ModelFamilyError::InvalidSelectorOutput(format!(
                "ACE-Step 1.5 does not support the selected {layout:?} layout"
            )));
        }
    };
    let hidden_size = dimension(
        probe,
        &format!("{prefix}decoder.layers.0.self_attn_norm.weight"),
        0,
    )?;
    if hidden_size == 0 {
        return Err(invalid_configuration("decoder hidden size"));
    }
    let intermediate_size = dimensions(
        probe,
        &format!("{prefix}decoder.layers.0.mlp.gate_proj.weight"),
    )?;
    let expected_intermediate = hidden_size
        .checked_mul(3)
        .ok_or_else(|| invalid_configuration("decoder hidden size overflow"))?;
    if intermediate_size != [expected_intermediate, hidden_size] {
        return Err(invalid_configuration("decoder intermediate dimensions"));
    }
    let query = dimensions(
        probe,
        &format!("{prefix}decoder.layers.0.self_attn.q_proj.weight"),
    )?;
    if query.len() != 2
        || query.get(1).copied() != Some(hidden_size)
        || query
            .first()
            .copied()
            .is_none_or(|value| value == 0 || value % 128 != 0)
    {
        return Err(invalid_configuration("decoder attention dimensions"));
    }
    let encoder_hidden = dimension(
        probe,
        &format!("{prefix}encoder.lyric_encoder.layers.0.input_layernorm.weight"),
        0,
    )?;
    if encoder_hidden == 0 {
        return Err(invalid_configuration("encoder hidden size"));
    }
    let encoder_query = dimensions(
        probe,
        &format!("{prefix}encoder.lyric_encoder.layers.0.self_attn.q_proj.weight"),
    )?;
    if encoder_query.len() != 2
        || encoder_query.get(1).copied() != Some(encoder_hidden)
        || encoder_query
            .first()
            .copied()
            .is_none_or(|value| value == 0 || value % 128 != 0)
    {
        return Err(invalid_configuration("encoder attention dimensions"));
    }
    let encoder_intermediate = dimensions(
        probe,
        &format!("{prefix}encoder.lyric_encoder.layers.0.mlp.gate_proj.weight"),
    )?;
    let expected_encoder_intermediate = encoder_hidden
        .checked_mul(3)
        .ok_or_else(|| invalid_configuration("encoder hidden size overflow"))?;
    if encoder_intermediate != [expected_encoder_intermediate, encoder_hidden] {
        return Err(invalid_configuration("encoder intermediate dimensions"));
    }
    let layer_count = probe.consecutive_block_count(&format!("{prefix}decoder.layers.{{}}."))?;
    if layer_count == 0 {
        return Err(invalid_configuration("decoder layer count"));
    }
    Ok(())
}

fn dimension(probe: &ModelProbe, key: &str, dimension: usize) -> Result<u64, ModelFamilyError> {
    dimensions(probe, key)?
        .get(dimension)
        .copied()
        .ok_or_else(|| invalid_configuration(key))
}

fn dimensions<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes()
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(key))
}

fn invalid_configuration(detail: &str) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "ACE-Step 1.5 source configuration mismatch: {detail}"
    ))
}
