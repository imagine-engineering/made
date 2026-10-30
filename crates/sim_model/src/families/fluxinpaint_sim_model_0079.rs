use crate::{
    FluxChromaConfiguration, FluxChromaVariant, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector, ModelProbe, flux_chroma_configuration_for_probe,
    flux_chroma_family::{
        FLUX_ARCHITECTURE_VERSION, FLUX_CLIP_TARGET, FLUX_COMPONENT_STATE_SCHEMAS, FLUX_COMPONENTS,
        FLUX_FORWARD_PROGRAM, FLUX_GUIDANCE_PROJECTION_KEYS, FLUX_INPUT_PROJECTION_KEYS,
        FLUX_LATENT_FEATURE_ID, FLUX_LATENT_IDENTIFIER, FLUX_LAYOUT_SIGNATURES,
        FLUX_MEMORY_ESTIMATOR, FLUX_MEMORY_USAGE_FACTOR, FLUX_MODEL_OPTIONAL_KEYS,
        FLUX_MODEL_REQUIRED_KEYS, FLUX_STATE_PLAN_CASES, FLUX_SUPPORTED_DEVICES, FLUX_WEIGHT_RULES,
    },
};
use sim_tensor::DType;

pub const MODEL_FAMILY_IDENTIFIER: &str = "FluxInpaint";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0079";
pub const MODEL_FAMILY_FIXTURE: &str = "fluxinpaint-comfy-model-0079";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 27;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "ef6ad66bca46237e050020b6302005360b24ee393bdc084212a7da9744f76bcb";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = FLUX_MEMORY_USAGE_FACTOR;
pub const MODEL_FAMILY_IN_CHANNELS: u64 = 96;

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: FLUX_INPUT_PROJECTION_KEYS,
        dimension: 1,
        values: &[384],
        score: 800,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: FLUX_GUIDANCE_PROJECTION_KEYS,
        score: 400,
    },
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: FLUX_ARCHITECTURE_VERSION,
    latent_feature_id: FLUX_LATENT_FEATURE_ID,
    latent_identifier: FLUX_LATENT_IDENTIFIER,
    clip_target: &FLUX_CLIP_TARGET,
    components: FLUX_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: FLUX_WEIGHT_RULES,
    required_keys: FLUX_MODEL_REQUIRED_KEYS,
    optional_keys: FLUX_MODEL_OPTIONAL_KEYS,
    supported_dtypes: SUPPORTED_DTYPES,
    supported_devices: FLUX_SUPPORTED_DEVICES,
    memory_estimator: FLUX_MEMORY_ESTIMATOR,
    forward_program: FLUX_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 27,
    source_architecture: "model_base.Flux",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&FLUX_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: FLUX_LAYOUT_SIGNATURES,
        cases: FLUX_STATE_PLAN_CASES,
    },
    component_state_schemas: FLUX_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<FluxChromaConfiguration, ModelFamilyError> {
    let configuration = flux_chroma_configuration_for_probe(
        probe,
        FluxChromaVariant::Flux,
        MODEL_FAMILY_IDENTIFIER,
    )?;
    if configuration.in_channels != MODEL_FAMILY_IN_CHANNELS {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "FluxInpaint configuration is invalid: in_channels {}; expected {}",
            configuration.in_channels, MODEL_FAMILY_IN_CHANNELS
        )));
    }
    if !configuration.guidance_embedding {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "FluxInpaint configuration is invalid: guidance embedding is missing".to_owned(),
        ));
    }
    Ok(configuration)
}
