use crate::{
    ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanSelector, ModelProbe,
    ModelSourceConfigurationRule, ModelStateTransformPlan, ModelWeightRule,
    pixart_configuration_for_probe, pixart_diffusers_state_plan,
    pixart_family::{
        PIXART_ALPHA_LATENT_FORMAT, PIXART_CLIP_TARGET, PIXART_COMPONENTS,
        PIXART_COMPONENT_STATE_SCHEMAS, PIXART_FORWARD_PROGRAM, PIXART_MEMORY_ESTIMATOR,
        PIXART_MODEL_OPTIONAL_KEYS, PIXART_MODEL_REQUIRED_KEYS, PIXART_PREFIXED_NATIVE_STATE_PLAN,
        PIXART_STANDALONE_NATIVE_STATE_PLAN, PIXART_SUPPORTED_DEVICES, PIXART_SUPPORTED_DTYPES,
        PixArtConfiguration, PixArtLayout, PixArtVariant,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "PixArtAlpha";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0110";
pub const MODEL_FAMILY_FIXTURE: &str = "pixartalpha-comfy-model-0110";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 23;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "07df55364e47885262ced5e10826c49404371b80273ee5f5d87046c8f2816f86";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 0.5;

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.csize_embedder.mlp.0.weight",
            "csize_embedder.mlp.0.weight",
            "adaln_single.emb.resolution_embedder.linear_1.weight",
        ],
        score: 600,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.blocks.0.attn.qkv.weight",
            "blocks.0.attn.qkv.weight",
            "transformer_blocks.0.attn1.to_q.weight",
        ],
        score: 400,
    },
];
const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "pixart-alpha-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &PIXART_CLIP_TARGET,
    components: PIXART_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: PIXART_MODEL_REQUIRED_KEYS,
    optional_keys: PIXART_MODEL_OPTIONAL_KEYS,
    supported_dtypes: PIXART_SUPPORTED_DTYPES,
    supported_devices: PIXART_SUPPORTED_DEVICES,
    memory_estimator: PIXART_MEMORY_ESTIMATOR,
    forward_program: PIXART_FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 23,
    source_architecture: "model_base.PixArt",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&PIXART_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Probe(state_plan_for_probe),
    component_state_schemas: PIXART_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<PixArtConfiguration, ModelFamilyError> {
    let configuration = pixart_configuration_for_probe(probe)?;
    if configuration.variant != PixArtVariant::Alpha {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "PixArtAlpha row cannot admit {:?}",
            configuration.variant
        )));
    }
    if configuration.latent_format.feature_id != PIXART_ALPHA_LATENT_FORMAT.feature_id {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "PixArtAlpha latent selection drifted from canonical SD15".to_owned(),
        ));
    }
    Ok(configuration)
}

fn state_plan_for_probe(
    probe: &ModelProbe,
) -> Result<ModelStateTransformPlan, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    match configuration.layout {
        PixArtLayout::PrefixedNative => PIXART_PREFIXED_NATIVE_STATE_PLAN.compile(),
        PixArtLayout::StandaloneNative => PIXART_STANDALONE_NATIVE_STATE_PLAN.compile(),
        PixArtLayout::Diffusers => {
            pixart_diffusers_state_plan(configuration.depth, configuration.variant)
        }
    }
}
