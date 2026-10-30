use crate::{
    LTX_AUDIO_MARKER, LTX_CLIP_TARGET, LTX_COMPONENT_STATE_SCHEMAS, LTX_COMPONENTS,
    LTX_FORWARD_PROGRAM, LTX_MODEL_OPTIONAL_KEYS, LTX_MODEL_REQUIRED_KEYS, LTX_SUPPORTED_DEVICES,
    LTX_SUPPORTED_DTYPES, LTXAV_LATENT_FORMAT, LTXAV_MEMORY_USAGE_FACTOR,
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanSelector,
    ModelProbe, LtxVariant, ltx_configuration_for_probe,
};
use crate::generated_ltxv_sim_model_0103::{
    ATTENTION_KEYS, LAYOUT_SIGNATURES, OUTPUT_KEYS, PATCH_KEYS, STATE_PLAN_CASES, TIMESTEP_KEYS,
    WEIGHT_RULES,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "LTXAV";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0102";
pub const MODEL_FAMILY_FIXTURE: &str = "ltxav-comfy-model-0102";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 33;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "8ad8b009ca551da18b106ab345d432d9bc2190b6d85ff24ea0b05c394d633592";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = LTXAV_MEMORY_USAGE_FACTOR;

const AUDIO_KEYS: &[&str] = &[
    "model.diffusion_model.audio_adaln_single.linear.weight",
    "model.audio_adaln_single.linear.weight",
    LTX_AUDIO_MARKER,
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: TIMESTEP_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: PATCH_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: ATTENTION_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: OUTPUT_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: AUDIO_KEYS,
        score: 400,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "ltxav-audio-video-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0039",
    latent_identifier: "LTXAV",
    clip_target: &LTX_CLIP_TARGET,
    components: LTX_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: LTX_MODEL_REQUIRED_KEYS,
    optional_keys: LTX_MODEL_OPTIONAL_KEYS,
    supported_dtypes: LTX_SUPPORTED_DTYPES,
    supported_devices: LTX_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: LTX_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 33,
    source_architecture: "model_base.LTXAV",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&LTX_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: LTX_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = ltx_configuration_for_probe(probe)?;
    if configuration.variant != LtxVariant::AudioVideo {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "LTXAV row requires the source audio discriminator".to_owned(),
        ));
    }
    if configuration.latent_format.feature_id != LTXAV_LATENT_FORMAT.feature_id
        || configuration.latent_format.identifier != LTXAV_LATENT_FORMAT.identifier
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "LTXAV latent selection drifted".to_owned(),
        ));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
