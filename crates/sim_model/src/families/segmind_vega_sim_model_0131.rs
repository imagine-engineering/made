use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector, ModelProbe, ModelWeightRule, SdxlConfiguration, SdxlVariant,
    sdxl_family,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Segmind_Vega";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0131";
pub const MODEL_FAMILY_FIXTURE: &str = "segmind-vega-comfy-model-0131";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 13;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "79765457e6a52ba75c6c0228e0eca3ee2f4a9c250045c5649c403deaa1551ea8";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = sdxl_family::SDXL_MEMORY_USAGE_FACTOR;
pub const SOURCE_ARCHITECTURE: &str = "model_base.SDXL";

const INPUT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.0.0.weight",
    "input_blocks.0.0.weight",
    "conv_in.weight",
];
const ADM_KEYS: &[&str] = &[
    "model.diffusion_model.label_emb.0.0.weight",
    "label_emb.0.0.weight",
    "add_embedding.linear_1.weight",
];
const DEEPEST_CONTEXT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.1.attn2.to_k.weight",
    "input_blocks.7.1.transformer_blocks.1.attn2.to_k.weight",
    "down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_k.weight",
];
const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 0,
        values: &[sdxl_family::SDXL_MODEL_CHANNELS],
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 1,
        values: &[4],
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: ADM_KEYS,
        dimension: 1,
        values: &[sdxl_family::SDXL_ADM_INPUT_DIMENSION],
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: DEEPEST_CONTEXT_KEYS,
        dimension: 1,
        values: &[sdxl_family::SDXL_CONTEXT_DIMENSION],
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
    architecture_version: "sdxl-segmind-vega-v1",
    latent_feature_id: "COMFY-MODEL-0047",
    latent_identifier: "SDXL",
    clip_target: &sdxl_family::SDXL_CLIP_TARGET,
    components: sdxl_family::SDXL_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: sdxl_family::SDXL_MODEL_REQUIRED_KEYS,
    optional_keys: sdxl_family::SDXL_MODEL_OPTIONAL_KEYS,
    supported_dtypes: sdxl_family::SDXL_SUPPORTED_DTYPES,
    supported_devices: sdxl_family::SDXL_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: sdxl_family::SDXL_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 13,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&sdxl_family::SDXL_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: sdxl_family::SDXL_LAYOUT_SIGNATURES,
        cases: sdxl_family::SDXL_STATE_PLAN_CASES,
    },
    component_state_schemas: sdxl_family::SDXL_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<SdxlConfiguration, ModelFamilyError> {
    let configuration = sdxl_family::configuration_for_probe(probe)?;
    if configuration.variant != SdxlVariant::SegmindVega {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Segmind_Vega requires the source-exact 4-channel/depth-2 profile, found {:?}",
            configuration.variant
        )));
    }
    Ok(configuration)
}
