use crate::{
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanSelector, ModelProbe,
    flux_chroma_family::{
        FLUX_COMPONENT_STATE_SCHEMAS, FLUX_FORWARD_PROGRAM, FLUX_INPUT_PROJECTION_KEYS,
        FLUX_LATENT_FEATURE_ID, FLUX_LATENT_IDENTIFIER, FLUX_LAYOUT_SIGNATURES,
        FLUX_MEMORY_ESTIMATOR, FLUX_MODEL_OPTIONAL_KEYS, FLUX_MODEL_REQUIRED_KEYS,
        FLUX_STATE_PLAN_CASES, FLUX_SUPPORTED_DEVICES, FLUX_SUPPORTED_DTYPES,
        FLUX_TEXT_PROJECTION_KEYS, FLUX_WEIGHT_RULES, FluxChromaConfiguration,
        FluxChromaVariant, configuration_for_probe as flux_chroma_configuration_for_probe,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "LongCatImage";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0105";
pub const MODEL_FAMILY_FIXTURE: &str = "longcatimage-comfy-model-0105";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 29;
pub const MODEL_FAMILY_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "d9125d4e44fd86b00156b09a4c90f99c4e513c6e4de00032b1795d38afc35fec";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.5;
pub const SOURCE_CONTEXT_INPUT_DIMENSION: u64 = 3_584;
pub const SOURCE_INPUT_CHANNELS: u64 = 16;
pub const SOURCE_PATCH_SIZE: u64 = 2;
pub const SOURCE_DOUBLE_BLOCK_COUNT: usize = 19;
pub const SOURCE_SINGLE_BLOCK_COUNT: usize = 38;

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "comfy.text_encoders.hunyuan_video.llama_detect.qwen25_7b",
    }];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "comfy.text_encoders.longcat_image.LongCatImageTokenizer",
        clip_model: "comfy.text_encoders.longcat_image.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];

pub const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "LongCat Image flow-matching transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Flux latent decoder",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "LongCat Qwen 2.5 7B conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: FLUX_TEXT_PROJECTION_KEYS,
        dimension: 1,
        values: &[SOURCE_CONTEXT_INPUT_DIMENSION],
        score: 900,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: FLUX_INPUT_PROJECTION_KEYS,
        dimension: 1,
        values: &[SOURCE_INPUT_CHANNELS * SOURCE_PATCH_SIZE * SOURCE_PATCH_SIZE],
        score: 300,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "longcat-image-flux-v1",
    latent_feature_id: FLUX_LATENT_FEATURE_ID,
    latent_identifier: FLUX_LATENT_IDENTIFIER,
    clip_target: &CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: FLUX_WEIGHT_RULES,
    required_keys: FLUX_MODEL_REQUIRED_KEYS,
    optional_keys: FLUX_MODEL_OPTIONAL_KEYS,
    supported_dtypes: FLUX_SUPPORTED_DTYPES,
    supported_devices: FLUX_SUPPORTED_DEVICES,
    memory_estimator: FLUX_MEMORY_ESTIMATOR,
    forward_program: FLUX_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 29,
    source_architecture: "model_base.LongCatImage",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
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
    flux_chroma_configuration_for_probe(
        probe,
        FluxChromaVariant::LongCatImage,
        MODEL_FAMILY_IDENTIFIER,
    )
}
