use crate::{
    ModelClipTargetSelector, ModelFamilyDefinition, ModelFamilyRegistration,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelLayoutSignature, ModelProbe, ModelSourceConfigurationRule, ModelStateLayout,
    ModelWeightRule, Omnigen2BooguConfiguration, Omnigen2BooguVariant,
    omnigen2_boogu_family::{
        BOOGU_CLIP_TARGET, BOOGU_COMPONENT_STATE_SCHEMAS, BOOGU_DETECTION_RULES,
        BOOGU_FORWARD_PROGRAM, BOOGU_MEMORY_ESTIMATOR, BOOGU_MODEL_OPTIONAL_KEYS,
        BOOGU_MODEL_REQUIRED_KEYS, BOOGU_SUPPORTED_DTYPES, OMNIGEN2_BOOGU_COMPONENTS,
        OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN, OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN,
        OMNIGEN2_BOOGU_SUPPORTED_DEVICES,
    },
    omnigen2_boogu_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Boogu";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0065";
pub const MODEL_FAMILY_FIXTURE: &str = "boogu-comfy-model-0065";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 76;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "3fa974328bf109ea87a9e4506ccb16b18b84a0f657636b04313a62f089a1242d";
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 3.16;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.15;

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "boogu-transformer-2d-v1",
    latent_feature_id: "COMFY-MODEL-0029",
    latent_identifier: "Flux",
    clip_target: &BOOGU_CLIP_TARGET,
    components: OMNIGEN2_BOOGU_COMPONENTS,
    detection_rules: BOOGU_DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: BOOGU_MODEL_REQUIRED_KEYS,
    optional_keys: BOOGU_MODEL_OPTIONAL_KEYS,
    supported_dtypes: BOOGU_SUPPORTED_DTYPES,
    supported_devices: OMNIGEN2_BOOGU_SUPPORTED_DEVICES,
    memory_estimator: BOOGU_MEMORY_ESTIMATOR,
    forward_program: BOOGU_FORWARD_PROGRAM,
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
            "model.diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
            "model.diffusion_model.norm_out.linear_2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "x_embedder.weight",
            "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
            "norm_out.linear_2.weight",
        ],
        required_prefixes: &[],
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 76,
    source_architecture: "model_base.Boogu",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&BOOGU_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: BOOGU_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Omnigen2BooguConfiguration, ModelFamilyError> {
    let configuration = omnigen2_boogu_configuration_for_probe(probe)?;
    if configuration.variant != Omnigen2BooguVariant::Boogu {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Boogu row cannot admit {:?}",
            configuration.variant
        )));
    }
    if configuration.latent_format.feature_id != MODEL_FAMILY.latent_feature_id
        || configuration.latent_format.identifier != MODEL_FAMILY.latent_identifier
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "Boogu latent selection drifted from canonical Flux".to_owned(),
        ));
    }
    Ok(configuration)
}
