use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelSourceConfigurationRule,
    ModelStateLayout, ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "Anima";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0063";
pub const MODEL_FAMILY_FIXTURE: &str = "anima-comfy-model-0063";
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "32e94c1d1213ec70bd710c79498c6c8317ab8d882747ea51b10248a13fa21976";

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.anima.AnimaTokenizer",
        clip_model: "sim.text_encoders.anima.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "diffusion",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "latent_decoder",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.blocks.0.mlp.layer1.weight",
            "blocks.0.mlp.layer1.weight",
        ],
        score: 500,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "llm_adapter.blocks.0.cross_attn.q_proj.weight",
        ],
        score: 500,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &["native.llm_adapter.in_proj.weight"];
const OPTIONAL_KEYS: &[&str] = &[
    "native.llm_adapter.in_proj.bias",
    "native.blocks.0.mlp.layer1.weight",
    "native.llm_adapter.blocks.0.cross_attn.q_proj.weight",
    "native.x_embedder.proj.1.weight",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "llm_adapter.in_proj",
        operation: ModelForwardOperation::Linear {
            weight: "native.llm_adapter.in_proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "llm_adapter.activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "llm_adapter.normalized",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "anima-source-1099-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
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
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "model.diffusion_model." },
                        "minimum_matches": 1,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "model.diffusion_model.", "to": "native." }
                    },
                    "component": "model"
                }
            },
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "first_stage_model." },
                        "minimum_matches": 0,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "first_stage_model.", "to": "vae." }
                    },
                    "component": "vae"
                }
            },
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "cond_stage_model." },
                        "minimum_matches": 0,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "cond_stage_model.", "to": "text_encoder." }
                    },
                    "component": "text_encoder"
                }
            }
        ],
        "unmatched": "Reject"
    }"#,
};

const DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "llm_adapter." },
                        "minimum_matches": 1,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "llm_adapter.", "to": "native.llm_adapter." }
                    },
                    "component": "model"
                }
            },
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "blocks." },
                        "minimum_matches": 1,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "blocks.", "to": "native.blocks." }
                    },
                    "component": "model"
                }
            },
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "x_embedder." },
                        "minimum_matches": 1,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "x_embedder.", "to": "native.x_embedder." }
                    },
                    "component": "model"
                }
            },
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "vae." },
                        "minimum_matches": 0,
                        "maximum_matches": 16384
                    },
                    "rewrite": "Identity",
                    "component": "vae"
                }
            },
            {
                "Route": {
                    "selector": {
                        "predicate": { "Prefix": "text_encoders." },
                        "minimum_matches": 0,
                        "maximum_matches": 16384
                    },
                    "rewrite": {
                        "Prefix": { "from": "text_encoders.", "to": "text_encoder." }
                    },
                    "component": "text_encoder"
                }
            }
        ],
        "unmatched": "Reject"
    }"#,
};

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &DIFFUSERS_STATE_PLAN,
    },
];

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.llm_adapter.in_proj.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["llm_adapter.in_proj.weight"],
        required_prefixes: &[],
    },
];

const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
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

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 84,
    source_architecture: "model_base.Anima",
    source_configuration: SOURCE_CONFIGURATION,
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
    let prefix = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => "model.diffusion_model.",
        ModelStateLayout::Diffusers => "",
        layout => {
            return Err(ModelFamilyError::InvalidSelectorOutput(format!(
                "Anima does not support the selected {layout:?} layout"
            )));
        }
    };
    let x_embedder_key = format!("{prefix}x_embedder.proj.1.weight");
    let x_embedder_shape = probe
        .tensor_shapes()
        .get(&x_embedder_key)
        .ok_or_else(|| invalid_configuration("missing x_embedder projection"))?;
    let [model_channels, packed_channels, ..] = x_embedder_shape.as_slice() else {
        return Err(invalid_configuration("x_embedder projection rank"));
    };
    if !matches!(*model_channels, 2_048 | 5_120) {
        return Err(invalid_configuration("model channels"));
    }
    let _input_channels = packed_channels
        .checked_div(4)
        .and_then(|channels| channels.checked_sub(1))
        .filter(|channels| matches!(*channels, 16 | 17))
        .ok_or_else(|| invalid_configuration("packed input channels"))?;
    if packed_channels % 4 != 0 {
        return Err(invalid_configuration("packed input channel divisibility"));
    }
    let mut profile = ModelFamilyProfile::from_definition(&MODEL_FAMILY);
    let source_memory_factor = (*model_channels as f64 / 2_048.0) * 0.95 * 1.4;
    let conservative_bytes = source_memory_factor.ceil() as u32;
    profile.memory_estimator = MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: conservative_bytes,
        activation_bytes_per_element: conservative_bytes,
    };
    Ok(profile)
}

fn invalid_configuration(detail: &str) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "Anima source configuration mismatch: {detail}"
    ))
}
