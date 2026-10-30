use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "SAM3";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0115";
pub const MODEL_FAMILY_FIXTURE: &str = "sam3-comfy-model-0115";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 87;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "1ebf4c25f92a29306bbd880a6efdb85c865462afadc8f74c4bb5d5009451c2d4";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const SAM31_PROPAGATION_KEY: &str =
    "detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sam3Layout {
    SourceNative,
    SourceCheckpoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sam3Configuration {
    pub layout: Sam3Layout,
    pub hidden_size: u64,
    pub query_count: u64,
    pub tracker_layer_count: usize,
    pub propagation_convolutions: bool,
}

pub const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.sam3_clip.SAM3TokenizerWrapper",
        clip_model: "sim.text_encoders.sam3_clip.SAM3ClipModelWrapper",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
pub const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "SAM 3 detector and video tracker",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SAM 3 CLIP language backbone",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
        score: 550,
    },
    ModelDetectionRule::KeyPresent {
        key: "detector.transformer.decoder.query_embed.weight",
        score: 450,
    },
];

pub const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "detector.",
        target_prefix: "native.detector.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "tracker.",
        target_prefix: "native.tracker.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "sam3_clip.",
        target_prefix: "text_encoder.sam3_clip.",
        required: false,
    },
];

pub const REQUIRED_KEYS: &[&str] = &[
    "native.detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
    "native.detector.transformer.decoder.query_embed.weight",
    "native.tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight",
    "native.tracker.sam_decoder.transformer.layers.0.self_attn.k_proj.weight",
    "native.tracker.sam_decoder.transformer.layers.0.self_attn.v_proj.weight",
    "native.tracker.sam_decoder.transformer.layers.0.mlp.0.weight",
    "native.tracker.sam_decoder.transformer.layers.0.norm_final.weight",
];
pub const OPTIONAL_KEYS: &[&str] = &[
    "native.tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.bias",
    "native.tracker.sam_decoder.transformer.layers.0.self_attn.k_proj.bias",
    "native.tracker.sam_decoder.transformer.layers.0.self_attn.v_proj.bias",
    "native.tracker.sam_decoder.transformer.layers.0.mlp.2.weight",
    "native.detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight",
];
pub const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "detector_queries",
        operation: ModelForwardOperation::Linear {
            weight: "native.detector.transformer.decoder.query_embed.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "tracker_query_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "segmentation_embedding",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sam3-detector-tracker-native-v1",
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

pub const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"Suffix":".attn.freqs_cis"},"minimum_matches":0,"maximum_matches":16384}}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"detector."},{"Not":{"Suffix":".attn.freqs_cis"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"detector.","to":"native.detector."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"tracker."},{"Not":{"Suffix":".attn.freqs_cis"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"tracker.","to":"native.tracker."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"sam3_clip."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"sam3_clip.","to":"text_encoder.sam3_clip."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const SOURCE_CHECKPOINT_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"All":[{"Prefix":"detector.backbone.language_backbone."},{"Contains":"resizer."}]},"minimum_matches":0,"maximum_matches":16384}}},
            {"Drop":{"selector":{"predicate":{"All":[{"Prefix":"backbone.language_backbone."},{"Contains":"resizer."}]},"minimum_matches":0,"maximum_matches":16384}}},
            {"Move":{"selector":{"predicate":{"Prefix":"detector.backbone.language_backbone.encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"detector.backbone.language_backbone.encoder.","to":"text_encoder.sam3_clip.transformer."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"backbone.language_backbone.encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"backbone.language_backbone.encoder.","to":"text_encoder.sam3_clip.transformer."}},"component":"text_encoder"}},
            {"Drop":{"selector":{"predicate":{"Suffix":".attn.freqs_cis"},"minimum_matches":0,"maximum_matches":16384}}},
            {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"detector."},{"Not":{"Prefix":"detector.backbone.language_backbone."}},{"Suffix":".in_proj_weight"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"detector.","to":"native.detector."}},{"Suffix":{"from":"in_proj_weight","to":"q_proj.weight"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"detector.","to":"native.detector."}},{"Suffix":{"from":"in_proj_weight","to":"k_proj.weight"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"detector.","to":"native.detector."}},{"Suffix":{"from":"in_proj_weight","to":"v_proj.weight"}}]}}]}},
            {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"detector."},{"Not":{"Prefix":"detector.backbone.language_backbone."}},{"Suffix":".in_proj_bias"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"detector.","to":"native.detector."}},{"Suffix":{"from":"in_proj_bias","to":"q_proj.bias"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"detector.","to":"native.detector."}},{"Suffix":{"from":"in_proj_bias","to":"k_proj.bias"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"detector.","to":"native.detector."}},{"Suffix":{"from":"in_proj_bias","to":"v_proj.bias"}}]}}]}},
            {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"tracker.model."},{"Suffix":".in_proj_weight"}]},"minimum_matches":1,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Suffix":{"from":"in_proj_weight","to":"q_proj.weight"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Suffix":{"from":"in_proj_weight","to":"k_proj.weight"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Suffix":{"from":"in_proj_weight","to":"v_proj.weight"}}]}}]}},
            {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"tracker.model."},{"Suffix":".in_proj_bias"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Suffix":{"from":"in_proj_bias","to":"q_proj.bias"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Suffix":{"from":"in_proj_bias","to":"k_proj.bias"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Suffix":{"from":"in_proj_bias","to":"v_proj.bias"}}]}}]}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"tracker.model."},{"Contains":".mlp.lin1."}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Contains":{"from":".mlp.lin1.","to":".mlp.0."}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"tracker.model."},{"Contains":".mlp.lin2."}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Contains":{"from":".mlp.lin2.","to":".mlp.2."}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"tracker.model."},{"Contains":".norm_final_attn."}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},{"Contains":{"from":".norm_final_attn.","to":".norm_final."}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"tracker.model."},{"Not":{"Contains":".in_proj_"}},{"Not":{"Contains":".mlp.lin1."}},{"Not":{"Contains":".mlp.lin2."}},{"Not":{"Contains":".norm_final_attn."}},{"Not":{"Suffix":".attn.freqs_cis"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"tracker.model.","to":"native.tracker."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"detector."},{"Not":{"Prefix":"detector.backbone.language_backbone."}},{"Not":{"Contains":".in_proj_"}},{"Not":{"Suffix":".attn.freqs_cis"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"detector.","to":"native.detector."}},"component":"model"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &SOURCE_CHECKPOINT_STATE_PLAN,
    },
];

pub const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
            "detector.transformer.decoder.query_embed.weight",
            "tracker.sam_decoder.transformer.layers.0.self_attn.q_proj.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
            "detector.transformer.decoder.query_embed.weight",
            "tracker.model.sam_decoder.transformer.layers.0.self_attn.in_proj_weight",
        ],
        required_prefixes: &[],
    },
];

pub const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
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
    source_ordinal: 87,
    source_architecture: "model_base.SAM3",
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

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Sam3Configuration, ModelFamilyError> {
    if probe.tensor_shapes.contains_key(SAM31_PROPAGATION_KEY) {
        return Err(invalid_configuration(
            "SAM 3.1 propagation weights require the SAM31 family",
        ));
    }
    configuration_for_probe_kind(probe, false)
}

pub fn configuration_for_probe_kind(
    probe: &ModelProbe,
    propagation_convolutions: bool,
) -> Result<Sam3Configuration, ModelFamilyError> {
    if propagation_convolutions && !probe.tensor_shapes.contains_key(SAM31_PROPAGATION_KEY) {
        return Err(invalid_configuration("missing SAM 3.1 propagation convolution"));
    }
    let layout = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::StandaloneNative => Sam3Layout::SourceNative,
        ModelStateLayout::Diffusers => Sam3Layout::SourceCheckpoint,
        ModelStateLayout::PrefixedNative => {
            return Err(invalid_configuration("prefixed-native layout is unsupported"));
        }
    };
    let qkv = shape(
        probe,
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight",
    )?;
    let [projected_hidden_size, hidden_size] = qkv else {
        return Err(invalid_configuration("vision QKV projection rank"));
    };
    if *hidden_size == 0 || *projected_hidden_size != hidden_size.saturating_mul(3) {
        return Err(invalid_configuration("vision QKV projection shape"));
    }
    let query = shape(probe, "detector.transformer.decoder.query_embed.weight")?;
    let [query_count, query_hidden_size] = query else {
        return Err(invalid_configuration("query embedding rank"));
    };
    if *query_count == 0 || *query_hidden_size != *hidden_size {
        return Err(invalid_configuration("query embedding shape"));
    }
    let tracker_prefix = match layout {
        Sam3Layout::SourceNative => "tracker.sam_decoder.transformer.layers.",
        Sam3Layout::SourceCheckpoint => "tracker.model.sam_decoder.transformer.layers.",
    };
    let tracker_layer_count = probe.consecutive_block_count(&format!("{tracker_prefix}{{}}."))?;
    if tracker_layer_count == 0 {
        return Err(invalid_configuration("tracker decoder has no layers"));
    }
    Ok(Sam3Configuration {
        layout,
        hidden_size: *hidden_size,
        query_count: *query_count,
        tracker_layer_count,
        propagation_convolutions,
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
        "SAM3 configuration is invalid: {}",
        message.into()
    ))
}
