use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe, ModelSourceConfigurationRule,
    ModelStateLayout, ModelWeightRule,
    lumina_zimage_family::{
        LUMINA_CLIP_TARGET, LUMINA_ZIMAGE_COMPONENTS, LUMINA_ZIMAGE_COMPONENT_STATE_SCHEMAS,
        LUMINA_ZIMAGE_FORWARD_PROGRAM, LUMINA_ZIMAGE_PREFIXED_STATE_PLAN,
        LUMINA_ZIMAGE_STANDALONE_STATE_PLAN, LUMINA_ZIMAGE_SUPPORTED_DEVICES,
        LUMINA_ZIMAGE_SUPPORTED_DTYPES, LuminaZImageConfiguration, LuminaZImageVariant,
    },
    lumina_zimage_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Lumina2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0107";
pub const MODEL_FAMILY_FIXTURE: &str = "lumina2-comfy-model-0107";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 49;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "75c847ca25aedb69a04f166021b8e45a1ef8195822cad5423521cbd200e66d14";
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 6.0;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.4;

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.cap_embedder.1.weight",
            "model.cap_embedder.1.weight",
            "cap_embedder.1.weight",
        ],
        dimension: 0,
        values: &[2_304],
        score: 700,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.noise_refiner.0.attention.qkv.weight",
            "model.noise_refiner.0.attention.qkv.weight",
            "noise_refiner.0.attention.qkv.weight",
        ],
        score: 300,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.cap_embedder.1.weight",
    "native.noise_refiner.0.attention.k_norm.weight",
    "native.x_embedder.weight",
    "native.final_layer.linear.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.bias",
    "native.cap_embedder.1.bias",
    "native.final_layer.linear.bias",
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "lumina2-nextdit-v1",
    latent_feature_id: "COMFY-MODEL-0029",
    latent_identifier: "Flux",
    clip_target: &LUMINA_CLIP_TARGET,
    components: LUMINA_ZIMAGE_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: LUMINA_ZIMAGE_SUPPORTED_DTYPES,
    supported_devices: LUMINA_ZIMAGE_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 4,
        activation_bytes_per_element: 8,
    },
    forward_program: LUMINA_ZIMAGE_FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &LUMINA_ZIMAGE_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
    },
];
const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.cap_embedder.1.weight",
            "model.diffusion_model.noise_refiner.0.attention.k_norm.weight",
            "model.diffusion_model.x_embedder.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "cap_embedder.1.weight",
            "noise_refiner.0.attention.k_norm.weight",
            "x_embedder.weight",
            "final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 49,
    source_architecture: "model_base.Lumina2",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&LUMINA_CLIP_TARGET),
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
    if configuration.variant != LuminaZImageVariant::Lumina2 {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Lumina2 row cannot admit {:?}",
            configuration.variant
        )));
    }
    Ok(configuration)
}
