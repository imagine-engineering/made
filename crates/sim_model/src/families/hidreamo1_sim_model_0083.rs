use crate::{
    HIDREAM_O1_ARCHITECTURE_VERSION, HIDREAM_O1_CLIP_TARGET,
    HIDREAM_O1_COMPONENT_STATE_SCHEMAS, HIDREAM_O1_COMPONENTS, HIDREAM_O1_LATENT_FEATURE_ID,
    HIDREAM_O1_LATENT_IDENTIFIER, HIDREAM_O1_LAYOUT_SIGNATURES, HIDREAM_O1_MEMORY_USAGE_FACTOR,
    HIDREAM_O1_STATE_PLAN_CASES, HIDREAM_O1_SUPPORTED_DEVICES, HIDREAM_O1_SUPPORTED_DTYPES,
    HIDREAM_O1_WEIGHT_RULES, MemoryEstimatorDescriptor, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelProbe, hidream_o1_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "HiDreamO1";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0083";
pub const MODEL_FAMILY_FIXTURE: &str = "hidreamo1-comfy-model-0083";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 70;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "fab656778560af76a8afb46e817ba49f6a3c8e99afc342aca41e5f306b743ea1";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = HIDREAM_O1_MEMORY_USAGE_FACTOR;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 3.0;
pub const MODEL_FAMILY_NOISE_SCALE: f64 = 8.0;

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.t_embedder1.mlp.0.weight",
            "t_embedder1.mlp.0.weight",
        ],
        dimension: 0,
        values: &[4_096],
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.t_embedder1.mlp.0.weight",
            "t_embedder1.mlp.0.weight",
        ],
        dimension: 1,
        values: &[256],
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.x_embedder.proj1.weight",
            "x_embedder.proj1.weight",
        ],
        dimension: 0,
        values: &[1_024],
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.x_embedder.proj1.weight",
            "x_embedder.proj1.weight",
        ],
        dimension: 1,
        values: &[3_072],
        score: 250,
    },
];

const REQUIRED_KEYS: &[&str] = &[
    "native.t_embedder1.mlp.0.weight",
    "native.t_embedder1.mlp.2.weight",
    "native.t_embedder1.mlp.2.bias",
    "native.x_embedder.proj1.weight",
    "native.x_embedder.proj2.weight",
    "native.x_embedder.proj2.bias",
    "native.visual.patch_embed.proj.weight",
    "native.language_model.layers.0.self_attn.q_proj.weight",
    "native.final_layer2.linear.weight",
    "native.final_layer2.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.visual.patch_embed.proj.bias",
    "native.language_model.embed_tokens.weight",
    "native.language_model.layers.0.self_attn.k_proj.weight",
    "native.language_model.layers.0.self_attn.v_proj.weight",
    "native.language_model.layers.0.self_attn.o_proj.weight",
    "native.final_layer2.norm_final.weight",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder1.mlp.2.weight",
            bias: Some("native.t_embedder1.mlp.2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.x_embedder.proj2.weight",
            bias: Some("native.x_embedder.proj2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "pixel_patch_bottleneck",
        operation: ModelForwardOperation::Linear {
            weight: "native.x_embedder.proj2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "reference_vision_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.visual.patch_embed.proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "language_model_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.language_model.layers.0.self_attn.q_proj.weight",
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
        checkpoint: "pixel_patch_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer2.linear.weight",
            bias: Some("native.final_layer2.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "pixel_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: HIDREAM_O1_ARCHITECTURE_VERSION,
    latent_feature_id: HIDREAM_O1_LATENT_FEATURE_ID,
    latent_identifier: HIDREAM_O1_LATENT_IDENTIFIER,
    clip_target: &HIDREAM_O1_CLIP_TARGET,
    components: HIDREAM_O1_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: HIDREAM_O1_WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: HIDREAM_O1_SUPPORTED_DTYPES,
    supported_devices: HIDREAM_O1_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 70,
    source_architecture: "model_base.HiDreamO1",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&HIDREAM_O1_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: HIDREAM_O1_LAYOUT_SIGNATURES,
        cases: HIDREAM_O1_STATE_PLAN_CASES,
    },
    component_state_schemas: HIDREAM_O1_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    hidream_o1_configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
