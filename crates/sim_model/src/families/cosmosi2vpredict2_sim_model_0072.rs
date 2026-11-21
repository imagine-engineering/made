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

pub const MODEL_FAMILY_IDENTIFIER: &str = "CosmosI2VPredict2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0072";
pub const MODEL_FAMILY_FIXTURE: &str = "cosmosi2vpredict2-comfy-model-0072";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 44;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "a15ee62529b7e580b268ded30430bf30d5a81d9e0ae2207894eb866ef99f193d";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.0;
pub const MODEL_FAMILY_TWO_B_MEMORY_USAGE_FACTOR: f64 = 0.95;
pub const MODEL_FAMILY_FOURTEEN_B_MEMORY_USAGE_FACTOR: f64 = 2.375;
pub const MODEL_FAMILY_SIGMA_DATA: f64 = 1.0;
pub const MODEL_FAMILY_SIGMA_MAX: f64 = 80.0;
pub const MODEL_FAMILY_SIGMA_MIN: f64 = 0.002;

pub type CosmosI2VPredict2Configuration = CosmosConfiguration;

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Cosmos Predict2 image-to-video diffusion transformer",
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
        values: &[72],
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
        checkpoint: "predict2_mlp_normalized",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "video_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

const TWO_B_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 1,
    activation_bytes_per_element: 1,
};

const FOURTEEN_B_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 3,
    activation_bytes_per_element: 3,
};

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "cosmos-predict2-i2v-v1",
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
    memory_estimator: TWO_B_MEMORY_ESTIMATOR,
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
    source_ordinal: 44,
    source_architecture: "model_base.CosmosPredict2(image_to_video=True)",
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
    let configuration = configuration_for_probe(probe)?;
    let memory_estimator = match configuration.model_size {
        CosmosModelSize::Predict2TwoB => TWO_B_MEMORY_ESTIMATOR,
        CosmosModelSize::FourteenB => FOURTEEN_B_MEMORY_ESTIMATOR,
        CosmosModelSize::SevenB => {
            return Err(ModelFamilyError::InvalidSelectorOutput(
                "CosmosI2VPredict2 cannot select the GeneralDIT seven-billion profile".to_owned(),
            ));
        }
    };
    Ok(ModelFamilyProfile {
        latent_feature_id: MODEL_FAMILY.latent_feature_id,
        latent_identifier: MODEL_FAMILY.latent_identifier,
        clip_target: &COSMOS_CLIP_TARGET,
        supported_dtypes: COSMOS_SUPPORTED_DTYPES,
        supported_devices: COSMOS_SUPPORTED_DEVICES,
        memory_estimator,
        forward_program: FORWARD_PROGRAM,
    })
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<CosmosI2VPredict2Configuration, ModelFamilyError> {
    cosmos_configuration_for_probe(
        probe,
        CosmosArchitecture::Predict2,
        true,
        MODEL_FAMILY_IDENTIFIER,
    )
}
