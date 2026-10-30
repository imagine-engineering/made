use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "RT_DETR_v4";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0114";
pub const MODEL_FAMILY_FIXTURE: &str = "rt-detr-v4-comfy-model-0114";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 85;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "35fdb721ddf496b31aea09e4a9662ea03846e9b2ebb54b404ff2618da495ad86";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RtDetrV4Layout {
    PrefixedNative,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RtDetrV4Configuration {
    pub layout: RtDetrV4Layout,
    pub encoder_hidden_size: u64,
    pub input_channels: u64,
    pub class_count: u64,
    pub decoder_hidden_size: u64,
    pub query_position_dimensions: u64,
    pub decoder_layer_count: usize,
    pub query_count: u64,
    pub feature_strides: [u64; 3],
}

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "RT-DETR v4 backbone, hybrid encoder, and transformer decoder",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "inherited optional first-stage state",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "inherited optional conditioning state without a CLIP target",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.encoder.pan_blocks.1.cv4.conv.weight",
            "encoder.pan_blocks.1.cv4.conv.weight",
        ],
        score: 600,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.decoder.enc_score_head.weight",
            "decoder.enc_score_head.weight",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.decoder.query_pos_head.layers.0.weight",
            "decoder.query_pos_head.layers.0.weight",
        ],
        score: 150,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "model.diffusion_model.",
        target_prefix: "native.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "encoder.",
        target_prefix: "native.encoder.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "decoder.",
        target_prefix: "native.decoder.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "backbone.",
        target_prefix: "native.backbone.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "first_stage_model.",
        target_prefix: "vae.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "cond_stage_model.",
        target_prefix: "text_encoder.",
        required: false,
    },
];

const REQUIRED_KEYS: &[&str] = &[
    "native.encoder.pan_blocks.1.cv4.conv.weight",
    "native.decoder.enc_score_head.weight",
    "native.decoder.enc_score_head.bias",
    "native.decoder.query_pos_head.layers.0.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.backbone.conv1.weight",
    "native.decoder.decoder.layers.0.self_attn.q_proj.weight",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "decoder_class_logits",
        operation: ModelForwardOperation::Linear {
            weight: "native.decoder.enc_score_head.weight",
            bias: Some("native.decoder.enc_score_head.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "decoder_probabilities",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "rt-detr-v4-native-v1",
    latent_feature_id: MODEL_FAMILY_FEATURE_ID,
    latent_identifier: "LatentFormat",
    clip_target: &CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: SUPPORTED_DTYPES,
    supported_devices: SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: FORWARD_PROGRAM,
};

const PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"encoder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"encoder.","to":"native.encoder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"decoder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"decoder.","to":"native.decoder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"backbone."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"backbone.","to":"native.backbone."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &STANDALONE_STATE_PLAN,
    },
];

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.encoder.pan_blocks.1.cv4.conv.weight",
            "model.diffusion_model.decoder.enc_score_head.weight",
            "model.diffusion_model.decoder.query_pos_head.layers.0.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "encoder.pan_blocks.1.cv4.conv.weight",
            "decoder.enc_score_head.weight",
            "decoder.query_pos_head.layers.0.weight",
        ],
        required_prefixes: &[],
    },
];

const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
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

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 85,
    source_architecture: "model_base.RT_DETR_v4",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
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
) -> Result<RtDetrV4Configuration, ModelFamilyError> {
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => {
            (RtDetrV4Layout::PrefixedNative, "model.diffusion_model.")
        }
        ModelStateLayout::StandaloneNative => (RtDetrV4Layout::StandaloneNative, ""),
        ModelStateLayout::Diffusers => {
            return Err(invalid_configuration(
                "Diffusers layout is not supported by the pinned RT-DETR source",
            ));
        }
    };
    let encoder = shape(
        probe,
        &format!("{prefix}encoder.pan_blocks.1.cv4.conv.weight"),
    )?;
    let [encoder_hidden_size, input_channels, 1, 1] = encoder else {
        return Err(invalid_configuration("encoder detection convolution shape"));
    };
    if *encoder_hidden_size == 0 || *input_channels == 0 {
        return Err(invalid_configuration("encoder dimensions must be nonzero"));
    }
    let score = shape(probe, &format!("{prefix}decoder.enc_score_head.weight"))?;
    let [class_count, decoder_hidden_size] = score else {
        return Err(invalid_configuration("decoder score head shape"));
    };
    if *class_count == 0 || *decoder_hidden_size != *encoder_hidden_size {
        return Err(invalid_configuration("decoder score head dimensions"));
    }
    let query = shape(
        probe,
        &format!("{prefix}decoder.query_pos_head.layers.0.weight"),
    )?;
    let [query_hidden_size, query_position_dimensions] = query else {
        return Err(invalid_configuration("query position head shape"));
    };
    if *query_hidden_size != *decoder_hidden_size || *query_position_dimensions != 4 {
        return Err(invalid_configuration("query position head dimensions"));
    }
    let decoder_layer_count = probe
        .consecutive_block_count(&format!("{prefix}decoder.decoder.layers.{{}}."))?;
    Ok(RtDetrV4Configuration {
        layout,
        encoder_hidden_size: *encoder_hidden_size,
        input_channels: *input_channels,
        class_count: *class_count,
        decoder_hidden_size: *decoder_hidden_size,
        query_position_dimensions: *query_position_dimensions,
        decoder_layer_count,
        query_count: 300,
        feature_strides: [8, 16, 32],
    })
}

fn shape<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}

fn invalid_configuration(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "RT_DETR_v4 configuration is invalid: {}",
        message.into()
    ))
}
