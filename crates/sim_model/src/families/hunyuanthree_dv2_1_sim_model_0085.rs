use crate::{
    HUNYUAN3D_COMPONENTS, HUNYUAN3D_MEMORY_USAGE_FACTOR, HUNYUAN3D_PREFIXED_STATE_PLAN,
    HUNYUAN3D_SAVED_MODEL_STATE_PLAN, HUNYUAN3D_STANDALONE_STATE_PLAN,
    HUNYUAN3D_SUPPORTED_DEVICES, HUNYUAN3D_SUPPORTED_DTYPES, Hunyuan3DVariant,
    MemoryEstimatorDescriptor, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep, ModelLayoutSignature,
    ModelProbe, ModelStateLayout, ModelWeightRule, hunyuan3d_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Hunyuan3Dv2_1";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0085";
pub const MODEL_FAMILY_FIXTURE: &str = "hunyuanthree-dv2-1-comfy-model-0085";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 67;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "e2286db9df5b5fd25f12d53d196869f4fd23fae61525d1903550585c5de4217b";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = HUNYUAN3D_MEMORY_USAGE_FACTOR;
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 1.0;

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.x_embedder.weight",
            "model.x_embedder.weight",
            "x_embedder.weight",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.t_embedder.mlp.2.weight",
            "model.t_embedder.mlp.2.weight",
            "t_embedder.mlp.2.weight",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.blocks.0.attn1.k_norm.weight",
            "model.blocks.0.attn1.k_norm.weight",
            "blocks.0.attn1.k_norm.weight",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.blocks.0.attn1.q_proj.weight",
            "model.blocks.0.attn1.q_proj.weight",
            "blocks.0.attn1.q_proj.weight",
        ],
        score: 250,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.x_embedder.weight",
    "native.t_embedder.mlp.0.weight",
    "native.t_embedder.mlp.0.bias",
    "native.t_embedder.mlp.2.weight",
    "native.t_embedder.mlp.2.bias",
    "native.t_embedder.cond_proj.weight",
    "native.t_embedder.cond_proj.bias",
    "native.blocks.0.attn1.k_norm.weight",
    "native.blocks.0.attn1.q_proj.weight",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.bias",
    "native.blocks.0.attn1.q_norm.weight",
    "native.blocks.0.attn1.k_proj.weight",
    "native.blocks.0.attn1.v_proj.weight",
    "native.blocks.0.attn1.out_proj.weight",
    "native.final_layer.norm_final.weight",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "time_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.mlp.0.weight",
            bias: Some("native.t_embedder.mlp.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "time_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "time_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.cond_proj.weight",
            bias: Some("native.t_embedder.cond_proj.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "latent_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.cond_proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "self_attention_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.attn1.q_proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "transformer_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "latent_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "hunyuan3d-v2-1-flow-dit-v1",
    latent_feature_id: "COMFY-MODEL-0033",
    latent_identifier: "Hunyuan3Dv2_1",
    clip_target: &CLIP_TARGET,
    components: HUNYUAN3D_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: HUNYUAN3D_SUPPORTED_DTYPES,
    supported_devices: HUNYUAN3D_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: FORWARD_PROGRAM,
};

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.x_embedder.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["model.x_embedder.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &["x_embedder.weight"],
        required_prefixes: &[],
    },
];

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &HUNYUAN3D_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &HUNYUAN3D_SAVED_MODEL_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &HUNYUAN3D_STANDALONE_STATE_PLAN,
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
        component: "clip_vision",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 67,
    source_architecture: "model_base.Hunyuan3Dv2_1",
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
    let configuration = hunyuan3d_configuration_for_probe(probe)?;
    if configuration.variant != Hunyuan3DVariant::V2_1 {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Hunyuan3Dv2_1 row cannot admit {:?}",
            configuration.variant
        )));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
