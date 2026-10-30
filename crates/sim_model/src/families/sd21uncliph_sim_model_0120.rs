use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector, ModelProbe,
    generated_sd20_sim_model_0119 as sd20,
    model_family::{ModelWeightStatisticObservation, ModelWeightStatisticRequest},
    sd2_family::{
        SD2_CLIP_TARGET, SD2_COMPONENT_STATE_SCHEMAS, SD2_COMPONENTS,
        SD2_FORWARD_PROGRAM, SD2_LAYOUT_SIGNATURES, SD2_MEMORY_USAGE_FACTOR,
        SD2_MODEL_OPTIONAL_KEYS, SD2_MODEL_REQUIRED_KEYS, SD2_SUPPORTED_DEVICES,
        SD2_SUPPORTED_DTYPES, Sd2Configuration, Sd2Variant,
        weight_statistic_request_for_probe as sd2_weight_statistic_request_for_probe,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "SD21UnclipH";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0120";
pub const MODEL_FAMILY_FIXTURE: &str = "sd21uncliph-comfy-model-0120";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 6;
pub const MODEL_FAMILY_SOURCE_PATH: &str = sd20::MODEL_FAMILY_SOURCE_PATH;
pub const MODEL_FAMILY_SOURCE_SHA256: &str = sd20::MODEL_FAMILY_SOURCE_SHA256;
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "25417841fc5119d0fe1b963842e694fbc32f6f0758ce3832f8fda6e056adc7b6";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = SD2_MEMORY_USAGE_FACTOR;
pub const SOURCE_ADM_IN_CHANNELS: u64 = 2_048;
pub const SOURCE_TIMESTEP_DIMENSION: u64 = 1_024;

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: sd20::INPUT_KEYS,
        score: 300,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: sd20::INPUT_KEYS,
        dimension: 0,
        values: &[crate::SD2_MODEL_CHANNELS],
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: sd20::INPUT_KEYS,
        dimension: 1,
        values: sd20::SOURCE_INPUT_CHANNELS,
        score: 300,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: sd20::CONTEXT_KEYS,
        dimension: 1,
        values: &[crate::SD2_CONTEXT_DIMENSION],
        score: 350,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: sd20::ADM_KEYS,
        dimension: 1,
        values: &[SOURCE_ADM_IN_CHANNELS],
        score: 400,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sd21-unclip-h-native-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &SD2_CLIP_TARGET,
    components: SD2_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: sd20::WEIGHT_RULES,
    required_keys: SD2_MODEL_REQUIRED_KEYS,
    optional_keys: SD2_MODEL_OPTIONAL_KEYS,
    supported_dtypes: SD2_SUPPORTED_DTYPES,
    supported_devices: SD2_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: SD2_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 6,
    source_architecture: "model_base.SD21UNCLIP",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&SD2_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: SD2_LAYOUT_SIGNATURES,
        cases: sd20::STATE_PLAN_CASES,
    },
    component_state_schemas: SD2_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    sd20::validate_variant_discriminator(
        probe,
        Some(SOURCE_ADM_IN_CHANNELS),
        MODEL_FAMILY_IDENTIFIER,
    )?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
    statistic: Option<&ModelWeightStatisticObservation>,
) -> Result<Sd2Configuration, ModelFamilyError> {
    sd20::configuration_for_probe_kind(
        probe,
        statistic,
        Sd2Variant::Sd21UnclipH,
        MODEL_FAMILY_IDENTIFIER,
    )
}

pub fn weight_statistic_request_for_probe(
    probe: &ModelProbe,
) -> Result<Option<ModelWeightStatisticRequest>, ModelFamilyError> {
    sd20::validate_variant_discriminator(
        probe,
        Some(SOURCE_ADM_IN_CHANNELS),
        MODEL_FAMILY_IDENTIFIER,
    )?;
    sd2_weight_statistic_request_for_probe(probe)
}
