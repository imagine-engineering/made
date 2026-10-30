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

pub const MODEL_FAMILY_IDENTIFIER: &str = "Ideogram4";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0096";
pub const MODEL_FAMILY_FIXTURE: &str = "ideogram4-comfy-model-0096";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 78;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "44dcd4fd8577fdbda47bce13d86c770e483487579a2059c1651a4a6be55c7170";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 11.6;
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 1.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ideogram4Layout {
    PrefixedNative,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ideogram4Configuration {
    pub layout: Ideogram4Layout,
    pub hidden_size: u64,
    pub in_channels: u64,
    pub layer_count: usize,
    pub attention_heads: u64,
    pub attention_head_dimension: u64,
    pub intermediate_size: u64,
    pub adaln_dimension: u64,
    pub llm_feature_dimension: u64,
    pub patch_size: u64,
    pub autoencoder_channels: u64,
    pub rope_theta: u64,
    pub mrope_sections: [u64; 3],
}

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.qwen3vl_8b",
    },
];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.ideogram4.Ideogram4Tokenizer",
        clip_model: "sim.text_encoders.ideogram4.te",
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
        role: "Ideogram 4 single-stream diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "source sampling constants",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Flux2 latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Qwen3-VL-8B thirteen-layer conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::Metadata {
    key: "image_model",
    value: "ideogram4",
    score: 1_000,
}];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.input_proj.weight",
    "native.input_proj.bias",
    "native.llm_cond_proj.weight",
    "native.layers.0.attention.qkv.weight",
    "native.layers.0.attention.norm_q.weight",
    "native.layers.0.feed_forward.w2.weight",
    "native.adaln_proj.weight",
    "native.embed_image_indicator.weight",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.llm_cond_norm.weight",
    "native.t_embedding.mlp_in.weight",
    "native.t_embedding.mlp_out.weight",
    "native.layers.0.attention.norm_k.weight",
    "native.layers.0.attention.o.weight",
    "native.layers.0.feed_forward.w1.weight",
    "native.layers.0.feed_forward.w3.weight",
    "native.final_layer.norm_final.weight",
    "native.final_layer.adaln_modulation.weight",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "conditioning_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.llm_cond_proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "transformer_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "feed_forward_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.layers.0.feed_forward.w2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
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
    architecture_version: "ideogram4-nextdit-v1",
    latent_feature_id: "COMFY-MODEL-0030",
    latent_identifier: "Flux2",
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
        bytes_per_parameter: 12,
        activation_bytes_per_element: 12,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] =
    &[ModelSourceConfigurationRule::Metadata {
        key: "image_model",
        value: "ideogram4",
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
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"input_proj."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"input_proj.","to":"native.input_proj."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"llm_cond_"},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"llm_cond_","to":"native.llm_cond_"}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"layers."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"layers.","to":"native.layers."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"adaln_proj."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"adaln_proj.","to":"native.adaln_proj."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"embed_image_indicator."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"embed_image_indicator.","to":"native.embed_image_indicator."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_layer."},"minimum_matches":1,"maximum_matches":128},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t_embedding."},"minimum_matches":0,"maximum_matches":128},"rewrite":{"Prefix":{"from":"t_embedding.","to":"native.t_embedding."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_multiplier"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.embed_image_indicator.weight",
            "model.diffusion_model.input_proj.weight",
            "model.diffusion_model.layers.0.attention.qkv.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "embed_image_indicator.weight",
            "input_proj.weight",
            "layers.0.attention.qkv.weight",
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
    source_ordinal: 78,
    source_architecture: "model_base.Ideogram4",
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

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Ideogram4Configuration, ModelFamilyError> {
    reject_diffusers(probe)?;
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => {
            (Ideogram4Layout::PrefixedNative, "model.diffusion_model.")
        }
        ModelStateLayout::StandaloneNative => (Ideogram4Layout::StandaloneNative, ""),
        ModelStateLayout::Diffusers => {
            return Err(invalid_configuration("Diffusers layout is unsupported"));
        }
    };
    let input_projection = shape(probe, &format!("{prefix}input_proj.weight"))?;
    let [hidden_size, in_channels] = input_projection else {
        return Err(invalid_configuration("input_proj.weight rank"));
    };
    if *hidden_size == 0 || *in_channels == 0 || *in_channels % 4 != 0 {
        return Err(invalid_configuration("input_proj.weight shape"));
    }
    let query_key_value = shape(probe, &format!("{prefix}layers.0.attention.qkv.weight"))?;
    if query_key_value != [hidden_size * 3, *hidden_size] {
        return Err(invalid_configuration("attention.qkv.weight shape"));
    }
    let query_norm = shape(probe, &format!("{prefix}layers.0.attention.norm_q.weight"))?;
    let [attention_head_dimension] = query_norm else {
        return Err(invalid_configuration("attention.norm_q.weight rank"));
    };
    if *attention_head_dimension == 0 || hidden_size % attention_head_dimension != 0 {
        return Err(invalid_configuration("attention head dimension"));
    }
    let feed_forward = shape(probe, &format!("{prefix}layers.0.feed_forward.w2.weight"))?;
    if feed_forward.len() != 2 || feed_forward[0] != *hidden_size || feed_forward[1] == 0 {
        return Err(invalid_configuration("feed_forward.w2.weight shape"));
    }
    let adaln = shape(probe, &format!("{prefix}adaln_proj.weight"))?;
    if adaln.len() != 2 || adaln[1] != *hidden_size || adaln[0] == 0 {
        return Err(invalid_configuration("adaln_proj.weight shape"));
    }
    let llm_projection = shape(probe, &format!("{prefix}llm_cond_proj.weight"))?;
    if llm_projection.len() != 2 || llm_projection[0] != *hidden_size || llm_projection[1] == 0 {
        return Err(invalid_configuration("llm_cond_proj.weight shape"));
    }
    let layer_count = probe.consecutive_block_count(&format!("{prefix}layers.{{}}."))?;
    if layer_count == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }
    Ok(Ideogram4Configuration {
        layout,
        hidden_size: *hidden_size,
        in_channels: *in_channels,
        layer_count,
        attention_heads: hidden_size / attention_head_dimension,
        attention_head_dimension: *attention_head_dimension,
        intermediate_size: feed_forward[1],
        adaln_dimension: adaln[0],
        llm_feature_dimension: llm_projection[1],
        patch_size: 2,
        autoencoder_channels: in_channels / 4,
        rope_theta: 5_000_000,
        mrope_sections: [24, 20, 20],
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
        "Ideogram4 configuration is invalid: {}",
        message.into()
    ))
}
