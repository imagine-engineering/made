use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    generated_sam3_sim_model_0115 as sam3,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "SAM31";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0116";
pub const MODEL_FAMILY_FIXTURE: &str = "sam31-comfy-model-0116";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 88;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "9895bb001105397b5697fa456bac1136eb224398e0faf89a6116c06d45abc031";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;

pub type Sam31Configuration = sam3::Sam3Configuration;

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
        score: 450,
    },
    ModelDetectionRule::KeyPresent {
        key: "detector.transformer.decoder.query_embed.weight",
        score: 350,
    },
    ModelDetectionRule::KeyPresent {
        key: sam3::SAM31_PROPAGATION_KEY,
        score: 400,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sam3.1-detector-tracker-native-v1",
    latent_feature_id: MODEL_FAMILY_FEATURE_ID,
    latent_identifier: "LatentFormat",
    clip_target: &sam3::CLIP_TARGET,
    components: sam3::COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: sam3::WEIGHT_RULES,
    required_keys: sam3::REQUIRED_KEYS,
    optional_keys: sam3::OPTIONAL_KEYS,
    supported_dtypes: sam3::SUPPORTED_DTYPES,
    supported_devices: sam3::SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: sam3::FORWARD_PROGRAM,
};

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &sam3::NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &sam3::SOURCE_CHECKPOINT_STATE_PLAN,
    },
];

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
            "detector.transformer.decoder.query_embed.weight",
            "detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight",
            "tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
            "detector.transformer.decoder.query_embed.weight",
            "detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight",
            "tracker.model.sam_decoder.transformer.layers.0.self_attn.in_proj_weight",
        ],
        required_prefixes: &[],
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 88,
    source_architecture: "model_base.SAM3",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&sam3::CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: sam3::COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Sam31Configuration, ModelFamilyError> {
    // Select against the SAM31 signatures first so incomplete propagation checkpoints fail closed.
    probe.select_layout(LAYOUT_SIGNATURES)?;
    let propagation = probe
        .tensor_shapes
        .get(sam3::SAM31_PROPAGATION_KEY)
        .map(Vec::as_slice)
        .ok_or_else(|| {
            ModelFamilyError::InvalidSelectorOutput(
                "SAM31 configuration is invalid: missing propagation convolution".to_owned(),
            )
        })?;
    let [output_channels, input_channels, 1, 1] = propagation else {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "SAM31 configuration is invalid: propagation convolution shape".to_owned(),
        ));
    };
    if *output_channels == 0 || *input_channels == 0 {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "SAM31 configuration is invalid: propagation channels must be nonzero".to_owned(),
        ));
    }
    sam3::configuration_for_probe_kind(probe, true)
}
