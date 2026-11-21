use crate::{
    COSMOS_CLIP_TARGET, COSMOS_LAYOUT_SIGNATURES, COSMOS_PATCH_PROJECTION_KEYS,
    COSMOS_PREDICT2_DETECTION_MARKER_KEYS, COSMOS_PREDICT2_STATE_PLAN_CASES,
    COSMOS_SUPPORTED_DEVICES, COSMOS_SUPPORTED_DTYPES, COSMOS_WEIGHT_RULES, CosmosArchitecture,
    CosmosConfiguration, CosmosModelSize, MemoryEstimatorDescriptor, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep, ModelProbe,
    cosmos_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "CosmosT2IPredict2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0073";
pub const MODEL_FAMILY_FIXTURE: &str = "cosmost2ipredict2-comfy-model-0073";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 43;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "b6be3927f0ac495a02caf635df8b77d4cfa3d2f1df990dc21149c7034deca1a1";
pub const MODEL_FAMILY_DECLARED_MEMORY_USAGE_FACTOR: f64 = 1.0;
pub const MODEL_FAMILY_BASE_MEMORY_USAGE_FACTOR: f64 = 0.95;
pub const MODEL_FAMILY_SIGMA_DATA: f64 = 1.0;
pub const MODEL_FAMILY_SIGMA_MAX: f64 = 80.0;
pub const MODEL_FAMILY_SIGMA_MIN: f64 = 0.002;

pub type CosmosT2IPredict2ModelSize = CosmosModelSize;
pub type CosmosT2IPredict2Configuration = CosmosConfiguration;

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Cosmos Predict2 text-to-image diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Wan21 latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Cosmos T5 conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: COSMOS_PREDICT2_DETECTION_MARKER_KEYS,
        score: 700,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: COSMOS_PATCH_PROJECTION_KEYS,
        dimension: 1,
        values: &[68],
        score: 300,
    },
];

const REQUIRED_KEYS: &[&str] = &[
    "native.t_embedder.1.linear_1.weight",
    "native.blocks.0.mlp.layer1.weight",
    "native.final_layer.linear.weight",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.proj.1.weight",
    "native.t_embedder.1.linear_2.weight",
    "native.blocks.0.self_attn.q_proj.weight",
    "native.t_embedding_norm.weight",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.1.linear_1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "predict2_mlp_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.mlp.layer1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "predict2_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "image_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "cosmos-predict2-t2i-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
    clip_target: &COSMOS_CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: COSMOS_WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: COSMOS_SUPPORTED_DTYPES,
    supported_devices: COSMOS_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: FORWARD_PROGRAM,
};

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
    source_ordinal: 43,
    source_architecture: "model_base.CosmosPredict2",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&COSMOS_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: COSMOS_LAYOUT_SIGNATURES,
        cases: COSMOS_PREDICT2_STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<CosmosT2IPredict2Configuration, ModelFamilyError> {
    cosmos_configuration_for_probe(
        probe,
        CosmosArchitecture::Predict2,
        false,
        MODEL_FAMILY_IDENTIFIER,
    )
}
