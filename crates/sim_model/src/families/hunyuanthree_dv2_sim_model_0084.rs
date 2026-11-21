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

pub const MODEL_FAMILY_IDENTIFIER: &str = "Hunyuan3Dv2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0084";
pub const MODEL_FAMILY_FIXTURE: &str = "hunyuanthree-dv2-comfy-model-0084";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 66;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "12f28c9d82aea3a819befbcd17328e77e9d8dcd3953c711e5c1863f497864437";
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
            "model.diffusion_model.latent_in.weight",
            "model.latent_in.weight",
            "latent_in.weight",
        ],
        score: 200,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.cond_in.weight",
            "model.cond_in.weight",
            "cond_in.weight",
        ],
        score: 200,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.double_blocks.8.img_attn.proj.weight",
            "model.double_blocks.8.img_attn.proj.weight",
            "double_blocks.8.img_attn.proj.weight",
        ],
        score: 300,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.single_blocks.0.linear1.weight",
            "model.single_blocks.0.linear1.weight",
            "single_blocks.0.linear1.weight",
        ],
        score: 300,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.latent_in.weight",
    "native.cond_in.weight",
    "native.double_blocks.0.img_attn.proj.weight",
    "native.single_blocks.0.linear1.weight",
    "native.time_in.in_layer.weight",
    "native.time_in.in_layer.bias",
    "native.time_in.out_layer.weight",
    "native.time_in.out_layer.bias",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.latent_in.bias",
    "native.cond_in.bias",
    "native.guidance_in.in_layer.weight",
    "native.guidance_in.in_layer.bias",
    "native.guidance_in.out_layer.weight",
    "native.guidance_in.out_layer.bias",
    "native.final_layer.adaLN_modulation.1.weight",
    "native.final_layer.adaLN_modulation.1.bias",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "time_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_in.in_layer.weight",
            bias: Some("native.time_in.in_layer.bias"),
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
            weight: "native.time_in.out_layer.weight",
            bias: Some("native.time_in.out_layer.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_in.out_layer.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "double_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.double_blocks.0.img_attn.proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "single_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.single_blocks.0.linear1.weight",
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
        checkpoint: "latent_projection",
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
    architecture_version: "hunyuan3d-v2-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0032",
    latent_identifier: "Hunyuan3Dv2",
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
        required_keys: &["model.diffusion_model.latent_in.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["model.latent_in.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &["latent_in.weight"],
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
    source_ordinal: 66,
    source_architecture: "model_base.Hunyuan3Dv2",
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
    if configuration.variant != Hunyuan3DVariant::V2 {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Hunyuan3Dv2 row cannot admit {:?}",
            configuration.variant
        )));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
