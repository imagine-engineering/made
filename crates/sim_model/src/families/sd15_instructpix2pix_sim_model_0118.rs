use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector, ModelProbe,
    generated_sd15_sim_model_0117 as sd15,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "SD15_instructpix2pix";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0118";
pub const MODEL_FAMILY_FIXTURE: &str = "sd15-instructpix2pix-comfy-model-0118";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 2;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str = sd15::MODEL_FAMILY_SOURCE_SHA256;
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "262e9286c7741d041d2e40d593628b82590b5eda05a46d0b3007ca5dac34dd11";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = sd15::MODEL_FAMILY_MEMORY_USAGE_FACTOR;
pub const SOURCE_INPUT_CHANNELS: u64 = 8;

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
        values: &[sd15::SOURCE_MODEL_CHANNELS],
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
        values: &[sd15::SOURCE_CONTEXT_DIMENSION],
        score: 200,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sd15-instruct-pix2pix-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &sd15::CLIP_TARGET,
    components: sd15::COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: sd15::WEIGHT_RULES,
    required_keys: sd15::REQUIRED_KEYS,
    optional_keys: sd15::OPTIONAL_KEYS,
    supported_dtypes: sd15::SUPPORTED_DTYPES,
    supported_devices: sd15::SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: sd15::FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 2,
    source_architecture: "model_base.SD15_instructpix2pix",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&sd15::CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: sd15::LAYOUT_SIGNATURES,
        cases: sd15::STATE_PLAN_CASES,
    },
    component_state_schemas: sd15::COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<sd15::Sd15Configuration, ModelFamilyError> {
    sd15::configuration_for_probe_kind(probe, SOURCE_INPUT_CHANNELS, MODEL_FAMILY_IDENTIFIER)
}
