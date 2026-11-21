use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe, ModelSourceConfigurationRule,
    ModelStateLayout, ModelWeightRule, pixeldit_pid_configuration_for_probe,
    pixeldit_pid_family::{
        PID_FORWARD_PROGRAM, PIXELDIT_CLIP_TARGET, PIXELDIT_CORE_STATE_PLAN,
        PIXELDIT_NET_STATE_PLAN, PIXELDIT_PID_COMPONENTS, PIXELDIT_PID_COMPONENT_STATE_SCHEMAS,
        PIXELDIT_PID_SUPPORTED_DEVICES, PIXELDIT_PID_SUPPORTED_DTYPES, PixelDitPidConfiguration,
        PixelDitPidVariant,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "PiD";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0109";
pub const MODEL_FAMILY_FIXTURE: &str = "pid-comfy-model-0109";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 47;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "0b768e17c73c88781e73a843407abb6c424737734c601a43bd262309a62fa86d";
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 1.5;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 0.04;

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::AnyKeyPresent {
    keys: &[
        "core.lq_proj.latent_proj.0.weight",
        "net.lq_proj.latent_proj.0.weight",
    ],
    score: 1_000,
}];
const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];
const REQUIRED_KEYS: &[&str] = &[
    "native.pixel_embedder.proj.weight",
    "native.s_embedder.proj.weight",
    "native.y_embedder.proj.weight",
    "native.pixel_blocks.0.adaLN_modulation_msa.weight",
    "native.pixel_blocks.0.adaLN_modulation_mlp.weight",
    "native.lq_proj.latent_proj.0.weight",
    "native.lq_proj.gate_modules.0.content_proj.weight",
    "native.lq_proj.output_heads.0.weight",
    "native.final_layer.linear.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.lq_proj.output_heads.0.bias",
    "native.final_layer.linear.bias",
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "pixeldit-pid-v1",
    latent_feature_id: "COMFY-MODEL-0042",
    latent_identifier: "PixelDiTPixel",
    clip_target: &PIXELDIT_CLIP_TARGET,
    components: PIXELDIT_PID_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: PIXELDIT_PID_SUPPORTED_DTYPES,
    supported_devices: PIXELDIT_PID_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 4,
        activation_bytes_per_element: 8,
    },
    forward_program: PID_FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &PIXELDIT_CORE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &PIXELDIT_NET_STATE_PLAN,
    },
];
const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "core.pixel_embedder.proj.weight",
            "core.lq_proj.latent_proj.0.weight",
            "core.final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "net.pixel_embedder.proj.weight",
            "net.lq_proj.latent_proj.0.weight",
            "net.final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 47,
    source_architecture: "model_base.PiD",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&PIXELDIT_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: PIXELDIT_PID_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<PixelDitPidConfiguration, ModelFamilyError> {
    let configuration = pixeldit_pid_configuration_for_probe(probe)?;
    if configuration.variant != PixelDitPidVariant::PiD {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "PiD row cannot admit {:?}",
            configuration.variant
        )));
    }
    Ok(configuration)
}
