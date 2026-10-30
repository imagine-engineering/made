use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelSourceConfigurationRule,
    ModelStateLayout, ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "Krea2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0101";
pub const MODEL_FAMILY_FIXTURE: &str = "krea2-comfy-model-0101";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 79;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "28002cd96a0f1a625fed794d7965c2e61831bb35e57aa153d231682a23b3d2e7";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.2;
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 1.15;
pub const MODEL_FAMILY_PATCH_SIZE: u64 = 2;
pub const MODEL_FAMILY_ATTENTION_HEAD_DIMENSION: u64 = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Krea2Layout {
    PrefixedNative,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Krea2Configuration {
    pub layout: Krea2Layout,
    pub feature_dimension: u64,
    pub channels: u64,
    pub patch_size: u64,
    pub layer_count: usize,
    pub attention_heads: u64,
    pub key_value_heads: u64,
    pub text_layer_count: u64,
    pub text_feature_dimension: u64,
    pub supports_temporal_batches: bool,
}

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.qwen3vl_4b",
    },
];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.krea2.Krea2Tokenizer",
        clip_model: "sim.text_encoders.krea2.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Krea 2 single-stream diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "source sampling constants",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Wan 2.1 latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Qwen3-VL-4B twelve-layer conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::Metadata {
    key: "image_model",
    value: "krea2",
    score: 1_000,
}];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.first.weight",
    "native.txtfusion.projector.weight",
    "native.txtfusion.layerwise_blocks.0.prenorm.scale",
    "native.blocks.0.attn.wq.weight",
    "native.blocks.0.attn.wk.weight",
    "native.tmlp.0.weight",
    "native.tmlp.0.bias",
    "native.blocks.0.mlp.down.weight",
    "native.last.linear.weight",
    "native.last.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.first.bias",
    "native.blocks.0.attn.wv.weight",
    "native.blocks.0.attn.wo.weight",
    "native.blocks.0.attn.gate.weight",
    "native.blocks.0.mlp.gate.weight",
    "native.blocks.0.mlp.up.weight",
    "native.txtmlp.1.weight",
    "native.txtmlp.3.weight",
    "native.tproj.1.weight",
    "native.last.norm.scale",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.tmlp.0.weight",
            bias: Some("native.tmlp.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "shared_stream_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "shared_stream_mlp",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.mlp.down.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.last.linear.weight",
            bias: Some("native.last.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "krea2-single-stream-dit-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
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
        bytes_per_parameter: 3,
        activation_bytes_per_element: 3,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] =
    &[ModelSourceConfigurationRule::Metadata {
        key: "image_model",
        value: "krea2",
    }];

const PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_multiplier"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.15},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"first."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"first.","to":"native.first."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"blocks.","to":"native.blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"tmlp."},"minimum_matches":1,"maximum_matches":128},"rewrite":{"Prefix":{"from":"tmlp.","to":"native.tmlp."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"txtfusion."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"txtfusion.","to":"native.txtfusion."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"txtmlp."},"minimum_matches":0,"maximum_matches":128},"rewrite":{"Prefix":{"from":"txtmlp.","to":"native.txtmlp."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"tproj."},"minimum_matches":0,"maximum_matches":128},"rewrite":{"Prefix":{"from":"tproj.","to":"native.tproj."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"last."},"minimum_matches":1,"maximum_matches":128},"rewrite":{"Prefix":{"from":"last.","to":"native.last."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_multiplier"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.15},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.txtfusion.projector.weight",
            "model.diffusion_model.first.weight",
            "model.diffusion_model.blocks.0.attn.wq.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "txtfusion.projector.weight",
            "first.weight",
            "blocks.0.attn.wq.weight",
        ],
        required_prefixes: &[],
    },
];

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

const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &["sampling_multiplier", "sampling_shift"],
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

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 79,
    source_architecture: "model_base.Krea2",
    source_configuration: SOURCE_CONFIGURATION,
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

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Krea2Configuration, ModelFamilyError> {
    reject_diffusers(probe)?;
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => {
            (Krea2Layout::PrefixedNative, "model.diffusion_model.")
        }
        ModelStateLayout::StandaloneNative => (Krea2Layout::StandaloneNative, ""),
        ModelStateLayout::Diffusers => {
            return Err(invalid_configuration("Diffusers layout is unsupported"));
        }
    };
    let first = shape(probe, &format!("{prefix}first.weight"))?;
    let [feature_dimension, patchified_channels] = first else {
        return Err(invalid_configuration("first.weight rank"));
    };
    let patch_area = MODEL_FAMILY_PATCH_SIZE * MODEL_FAMILY_PATCH_SIZE;
    if *feature_dimension == 0 || *patchified_channels == 0 || patchified_channels % patch_area != 0 {
        return Err(invalid_configuration("first.weight shape"));
    }
    let query = shape(probe, &format!("{prefix}blocks.0.attn.wq.weight"))?;
    if query.len() != 2 || query[1] != *feature_dimension || query[0] % MODEL_FAMILY_ATTENTION_HEAD_DIMENSION != 0 {
        return Err(invalid_configuration("attention query shape"));
    }
    let key = shape(probe, &format!("{prefix}blocks.0.attn.wk.weight"))?;
    if key.len() != 2 || key[1] != *feature_dimension || key[0] % MODEL_FAMILY_ATTENTION_HEAD_DIMENSION != 0 {
        return Err(invalid_configuration("attention key shape"));
    }
    let text_projector = shape(probe, &format!("{prefix}txtfusion.projector.weight"))?;
    if text_projector.len() != 2 || text_projector[0] != 1 || text_projector[1] == 0 {
        return Err(invalid_configuration("txtfusion.projector.weight shape"));
    }
    let text_norm = shape(
        probe,
        &format!("{prefix}txtfusion.layerwise_blocks.0.prenorm.scale"),
    )?;
    let [text_feature_dimension] = text_norm else {
        return Err(invalid_configuration("text fusion norm shape"));
    };
    if *text_feature_dimension == 0 {
        return Err(invalid_configuration("text feature dimension"));
    }
    let layer_count = probe.consecutive_block_count(&format!("{prefix}blocks.{{}}."))?;
    if layer_count == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }
    Ok(Krea2Configuration {
        layout,
        feature_dimension: *feature_dimension,
        channels: patchified_channels / patch_area,
        patch_size: MODEL_FAMILY_PATCH_SIZE,
        layer_count,
        attention_heads: query[0] / MODEL_FAMILY_ATTENTION_HEAD_DIMENSION,
        key_value_heads: key[0] / MODEL_FAMILY_ATTENTION_HEAD_DIMENSION,
        text_layer_count: text_projector[1],
        text_feature_dimension: *text_feature_dimension,
        supports_temporal_batches: true,
    })
}

fn reject_diffusers(probe: &ModelProbe) -> Result<(), ModelFamilyError> {
    let diffusers_format = probe
        .format_identities()
        .iter()
        .any(|identity| identity.eq_ignore_ascii_case("diffusers"));
    let diffusers_layout = probe
        .metadata()
        .get("model_layout")
        .is_some_and(|layout| layout.eq_ignore_ascii_case("diffusers"));
    if diffusers_format || diffusers_layout {
        return Err(invalid_configuration("Diffusers layout is unsupported"));
    }
    Ok(())
}

fn shape<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes()
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}

fn invalid_configuration(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "Krea2 configuration is invalid: {}",
        message.into()
    ))
}
