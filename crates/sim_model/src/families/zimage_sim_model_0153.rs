use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelWeightRule,
    lumina_zimage_family::{
        LUMINA_ZIMAGE_COMPONENT_STATE_SCHEMAS, LUMINA_ZIMAGE_COMPONENTS,
        LUMINA_ZIMAGE_FORWARD_PROGRAM, LUMINA_ZIMAGE_MODEL_REQUIRED_KEYS,
        LUMINA_ZIMAGE_PREFIXED_STATE_PLAN, LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
        LUMINA_ZIMAGE_SUPPORTED_DEVICES, LUMINA_ZIMAGE_SUPPORTED_DTYPES,
        LuminaZImageConfiguration, LuminaZImageLayout, LuminaZImageVariant, ZIMAGE_CLIP_TARGET,
        ZIMAGE_DIFFUSERS_STATE_PLAN, ZIMAGE_MEMORY_USAGE_FACTOR,
        configuration_for_probe as lumina_zimage_configuration_for_probe,
    },
};

// ModelStateTransformPlanDefinition and ModelForwardOperation remain owned by
// model_family; this row only selects the consolidation owner's immutable plans
// and forward program.

pub const MODEL_FAMILY_IDENTIFIER: &str = "ZImage";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0153";
pub const MODEL_FAMILY_FIXTURE: &str = "zimage-comfy-model-0153";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 46;
pub const MODEL_FAMILY_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "3cbd0174c3da63727b52aa2f40e85edf6173da0323041672d978bce26a8d78db";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = ZIMAGE_MEMORY_USAGE_FACTOR;
pub const SOURCE_ARCHITECTURE: &str = "model_base.Lumina2";

pub const PREFIXED_CAPTION_WEIGHT: &str =
    "model.diffusion_model.cap_embedder.1.weight";
pub const STANDALONE_CAPTION_WEIGHT: &str = "cap_embedder.1.weight";
pub const DIFFUSERS_CAPTION_WEIGHT: &str = "cap_embedder.1.weight";
pub const PREFIXED_QK_NORM_WEIGHT: &str =
    "model.diffusion_model.noise_refiner.0.attention.k_norm.weight";
pub const STANDALONE_QK_NORM_WEIGHT: &str =
    "noise_refiner.0.attention.k_norm.weight";
pub const DIFFUSERS_QK_NORM_WEIGHT: &str =
    "noise_refiner.0.attention.norm_k.weight";
pub const PREFIXED_X_WEIGHT: &str = "model.diffusion_model.x_embedder.weight";
pub const STANDALONE_X_WEIGHT: &str = "x_embedder.weight";
pub const DIFFUSERS_X_WEIGHT: &str = "all_x_embedder.2-1.weight";
pub const PREFIXED_FINAL_WEIGHT: &str =
    "model.diffusion_model.final_layer.linear.weight";
pub const STANDALONE_FINAL_WEIGHT: &str = "final_layer.linear.weight";
pub const DIFFUSERS_FINAL_WEIGHT: &str = "all_final_layer.2-1.linear.weight";

const CAPTION_WEIGHT_KEYS: &[&str] = &[
    PREFIXED_CAPTION_WEIGHT,
    STANDALONE_CAPTION_WEIGHT,
];
const QK_NORM_WEIGHT_KEYS: &[&str] = &[
    PREFIXED_QK_NORM_WEIGHT,
    STANDALONE_QK_NORM_WEIGHT,
    DIFFUSERS_QK_NORM_WEIGHT,
];
const X_WEIGHT_KEYS: &[&str] = &[
    PREFIXED_X_WEIGHT,
    STANDALONE_X_WEIGHT,
    DIFFUSERS_X_WEIGHT,
];
const FINAL_WEIGHT_KEYS: &[&str] = &[
    PREFIXED_FINAL_WEIGHT,
    STANDALONE_FINAL_WEIGHT,
    DIFFUSERS_FINAL_WEIGHT,
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: CAPTION_WEIGHT_KEYS,
        dimension: 0,
        values: &[3_840],
        score: 350,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: QK_NORM_WEIGHT_KEYS,
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: X_WEIGHT_KEYS,
        dimension: 1,
        values: &[64],
        score: 175,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: FINAL_WEIGHT_KEYS,
        dimension: 0,
        values: &[64],
        score: 175,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.bias",
    "native.cap_embedder.1.bias",
    "native.final_layer.linear.bias",
    "native.cap_pad_token",
    "native.clip_text_pooled_proj.0.weight",
    "native.siglip_embedder.0.weight",
    "native.dec_net.cond_embed.weight",
    "native.__x0__",
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "zimage-nextdit-v1",
    latent_feature_id: "COMFY-MODEL-0029",
    latent_identifier: "Flux",
    clip_target: &ZIMAGE_CLIP_TARGET,
    components: LUMINA_ZIMAGE_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: LUMINA_ZIMAGE_MODEL_REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: LUMINA_ZIMAGE_SUPPORTED_DTYPES,
    supported_devices: LUMINA_ZIMAGE_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 3,
        activation_bytes_per_element: 3,
    },
    forward_program: LUMINA_ZIMAGE_FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];

const PREFIXED_REQUIRED: &[&str] = &[
    PREFIXED_CAPTION_WEIGHT,
    PREFIXED_QK_NORM_WEIGHT,
    PREFIXED_X_WEIGHT,
    PREFIXED_FINAL_WEIGHT,
    "model.diffusion_model.layers.0.attention.qkv.weight",
];
const STANDALONE_REQUIRED: &[&str] = &[
    STANDALONE_CAPTION_WEIGHT,
    STANDALONE_QK_NORM_WEIGHT,
    STANDALONE_X_WEIGHT,
    STANDALONE_FINAL_WEIGHT,
    "layers.0.attention.qkv.weight",
];
const DIFFUSERS_REQUIRED: &[&str] = &[
    DIFFUSERS_CAPTION_WEIGHT,
    DIFFUSERS_QK_NORM_WEIGHT,
    "noise_refiner.0.attention.to_q.weight",
    DIFFUSERS_X_WEIGHT,
    DIFFUSERS_FINAL_WEIGHT,
    "layers.0.attention.to_q.weight",
];

pub const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: PREFIXED_REQUIRED,
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: STANDALONE_REQUIRED,
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: DIFFUSERS_REQUIRED,
        required_prefixes: &[],
    },
];

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &LUMINA_ZIMAGE_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &ZIMAGE_DIFFUSERS_STATE_PLAN,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 46,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&ZIMAGE_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: LUMINA_ZIMAGE_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<LuminaZImageConfiguration, ModelFamilyError> {
    let configuration = lumina_zimage_configuration_for_probe(probe)?;
    if configuration.variant != LuminaZImageVariant::ZImage {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "ZImage registration cannot claim {:?}",
            configuration.variant
        )));
    }
    if !matches!(
        configuration.layout,
        LuminaZImageLayout::PrefixedNative
            | LuminaZImageLayout::StandaloneNative
            | LuminaZImageLayout::Diffusers
    ) {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "ZImage admits only prefixed-native, standalone-native, or pinned Diffusers layouts"
                .to_owned(),
        ));
    }
    Ok(configuration)
}
