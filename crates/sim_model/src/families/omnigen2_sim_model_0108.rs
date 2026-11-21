use crate::{
    ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelWeightRule,
    omnigen2_boogu_configuration_for_probe,
    omnigen2_boogu_family::{
        OMNIGEN2_BASE_SUPPORTED_DTYPES, OMNIGEN2_BOOGU_COMPONENTS,
        OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN, OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN,
        OMNIGEN2_BOOGU_SUPPORTED_DEVICES, OMNIGEN2_CLIP_TARGET, OMNIGEN2_FORWARD_PROGRAM,
        OMNIGEN2_MEMORY_ESTIMATOR, OMNIGEN2_MODEL_OPTIONAL_KEYS, OMNIGEN2_MODEL_REQUIRED_KEYS,
        Omnigen2BooguConfiguration, Omnigen2BooguVariant,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Omnigen2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0108";
pub const MODEL_FAMILY_FIXTURE: &str = "omnigen2-comfy-model-0108";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 75;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "d7c697805b6e0d6244cdb4171cef04600f121354f02907a0ec22ae7791e63314";
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 2.6;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.95;

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::AnyKeyPresent {
    keys: &[
        "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
        "time_caption_embed.timestep_embedder.linear_1.bias",
    ],
    score: 1_000,
}];
const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: OMNIGEN2_MODEL_REQUIRED_KEYS,
        optional_keys: OMNIGEN2_MODEL_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &["reference_latent_count"],
        optional_keys: &[],
        allow_unexpected: false,
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

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "omnigen2-transformer-2d-v1",
    latent_feature_id: "COMFY-MODEL-0029",
    latent_identifier: "Flux",
    clip_target: &OMNIGEN2_CLIP_TARGET,
    components: OMNIGEN2_BOOGU_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: OMNIGEN2_MODEL_REQUIRED_KEYS,
    optional_keys: OMNIGEN2_MODEL_OPTIONAL_KEYS,
    supported_dtypes: OMNIGEN2_BASE_SUPPORTED_DTYPES,
    supported_devices: OMNIGEN2_BOOGU_SUPPORTED_DEVICES,
    memory_estimator: OMNIGEN2_MEMORY_ESTIMATOR,
    forward_program: OMNIGEN2_FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN,
    },
];
const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.x_embedder.weight",
            "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
            "model.diffusion_model.norm_out.linear_2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "x_embedder.weight",
            "time_caption_embed.timestep_embedder.linear_1.bias",
            "norm_out.linear_2.weight",
        ],
        required_prefixes: &[],
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 75,
    source_architecture: "model_base.Omnigen2",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&OMNIGEN2_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Omnigen2BooguConfiguration, ModelFamilyError> {
    let configuration = omnigen2_boogu_configuration_for_probe(probe)?;
    if configuration.variant != Omnigen2BooguVariant::Omnigen2 {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Omnigen2 row cannot admit {:?}",
            configuration.variant
        )));
    }
    Ok(configuration)
}
