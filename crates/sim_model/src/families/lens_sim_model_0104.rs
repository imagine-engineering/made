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

pub const MODEL_FAMILY_IDENTIFIER: &str = "Lens";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0104";
pub const MODEL_FAMILY_FIXTURE: &str = "lens-comfy-model-0104";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 81;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "b9b980a35d18a6db0e11098c73aee45b8d1b974f3e75d2631259772b818dc8ff";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 4.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 1.829;
pub const MODEL_FAMILY_PATCH_SIZE: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LensLayout {
    PrefixedNative,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LensConfiguration {
    pub layout: LensLayout,
    pub hidden_size: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub patch_size: u64,
    pub layer_count: usize,
    pub attention_heads: u64,
    pub attention_head_dimension: u64,
    pub text_feature_dimension: u64,
    pub selected_text_layer_count: usize,
    pub multi_layer_text_features: bool,
    pub rope_axes_dimensions: [u64; 3],
}

const DETECTED_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.gpt_oss",
    },
];
const DEFAULT_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[];

const DETECTED_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.gpt_oss.LensTokenizer",
        clip_model: "sim.text_encoders.gpt_oss.lens_te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: DETECTED_CLIP_CONFIGURATION,
        },
    }];
const DEFAULT_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.gpt_oss.LensTokenizer",
        clip_model: "sim.text_encoders.gpt_oss.lens_te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: DEFAULT_CLIP_CONFIGURATION,
        },
    }];

const DETECTED_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: DETECTED_CLIP_CANDIDATES,
    dynamic_selection: false,
};
const DEFAULT_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: DEFAULT_CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Lens dual-stream multimodal diffusion transformer",
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
        role: "GPT-OSS-20B multi-layer conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::Metadata {
    key: "image_model",
    value: "lens",
    score: 1_000,
}];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.img_in.weight",
    "native.proj_out.weight",
    "native.transformer_blocks.0.attn.norm_added_q.weight",
    "native.transformer_blocks.0.img_mlp.w1.weight",
    "native.time_text_embed.timestep_embedder.linear_1.weight",
    "native.time_text_embed.timestep_embedder.linear_1.bias",
    "native.transformer_blocks.0.img_mlp.w2.weight",
    "native.transformer_blocks.0.attn.to_out.0.weight",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.img_in.bias",
    "native.proj_out.bias",
    "native.txt_norm.weight",
    "native.txt_norm.0.weight",
    "native.txt_norm.1.weight",
    "native.txt_norm.2.weight",
    "native.txt_norm.3.weight",
    "native.txt_in.weight",
    "native.transformer_blocks.0.attn.img_qkv.weight",
    "native.transformer_blocks.0.attn.txt_qkv.weight",
    "native.transformer_blocks.0.img_mlp.w3.weight",
    "native.norm_out.linear.weight",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_text_embed.timestep_embedder.linear_1.weight",
            bias: Some("native.time_text_embed.timestep_embedder.linear_1.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "joint_stream_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "image_mlp_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.transformer_blocks.0.img_mlp.w2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.transformer_blocks.0.attn.to_out.0.weight",
            bias: None,
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
    architecture_version: "lens-dual-stream-mmdit-v1",
    latent_feature_id: "COMFY-MODEL-0030",
    latent_identifier: "Flux2",
    clip_target: &DEFAULT_CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: SUPPORTED_DTYPES,
    supported_devices: SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] =
    &[ModelSourceConfigurationRule::Metadata {
        key: "image_model",
        value: "lens",
    }];

const PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.829},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"img_in."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"img_in.","to":"native.img_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"proj_out."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"proj_out.","to":"native.proj_out."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_text_embed."},"minimum_matches":1,"maximum_matches":128},"rewrite":{"Prefix":{"from":"time_text_embed.","to":"native.time_text_embed."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"transformer_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"transformer_blocks.","to":"native.transformer_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"txt_norm"},"minimum_matches":1,"maximum_matches":128},"rewrite":{"Prefix":{"from":"txt_norm","to":"native.txt_norm"}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"txt_in."},"minimum_matches":0,"maximum_matches":64},"rewrite":{"Prefix":{"from":"txt_in.","to":"native.txt_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"norm_out."},"minimum_matches":0,"maximum_matches":64},"rewrite":{"Prefix":{"from":"norm_out.","to":"native.norm_out."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.829},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight",
            "model.diffusion_model.transformer_blocks.0.img_mlp.w1.weight",
            "model.diffusion_model.img_in.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.w1.weight",
            "img_in.weight",
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
        required_keys: &["sampling_shift"],
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
    source_ordinal: 81,
    source_architecture: "model_base.Lens",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    let detected = [
        "text_encoders.gpt_oss.transformer.layers.0.self_attn.sinks",
        "text_encoders.layers.0.self_attn.sinks",
    ]
    .iter()
    .any(|key| probe.tensor_shapes().contains_key(*key));
    Ok(ModelFamilyProfile {
        latent_feature_id: MODEL_FAMILY.latent_feature_id,
        latent_identifier: MODEL_FAMILY.latent_identifier,
        clip_target: if detected {
            &DETECTED_CLIP_TARGET
        } else {
            &DEFAULT_CLIP_TARGET
        },
        supported_dtypes: MODEL_FAMILY.supported_dtypes,
        supported_devices: MODEL_FAMILY.supported_devices,
        memory_estimator: MODEL_FAMILY.memory_estimator,
        forward_program: MODEL_FAMILY.forward_program,
    })
}

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<LensConfiguration, ModelFamilyError> {
    reject_diffusers(probe)?;
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (LensLayout::PrefixedNative, "model.diffusion_model."),
        ModelStateLayout::StandaloneNative => (LensLayout::StandaloneNative, ""),
        ModelStateLayout::Diffusers => {
            return Err(invalid_configuration("Diffusers layout is unsupported"));
        }
    };
    let image_input = shape(probe, &format!("{prefix}img_in.weight"))?;
    let [hidden_size, in_channels] = image_input else {
        return Err(invalid_configuration("img_in.weight rank"));
    };
    if *hidden_size == 0 || *in_channels == 0 {
        return Err(invalid_configuration("img_in.weight shape"));
    }
    let query_norm = shape(
        probe,
        &format!("{prefix}transformer_blocks.0.attn.norm_added_q.weight"),
    )?;
    let [attention_head_dimension] = query_norm else {
        return Err(invalid_configuration("attention norm shape"));
    };
    if *attention_head_dimension == 0 || hidden_size % attention_head_dimension != 0 {
        return Err(invalid_configuration("attention head dimension"));
    }
    let image_mlp = shape(
        probe,
        &format!("{prefix}transformer_blocks.0.img_mlp.w1.weight"),
    )?;
    if image_mlp.len() != 2 || image_mlp[1] != *hidden_size || image_mlp[0] == 0 {
        return Err(invalid_configuration("image MLP shape"));
    }
    let output = shape(probe, &format!("{prefix}proj_out.weight"))?;
    let patch_area = MODEL_FAMILY_PATCH_SIZE * MODEL_FAMILY_PATCH_SIZE;
    if output.len() != 2 || output[1] != *hidden_size || output[0] % patch_area != 0 {
        return Err(invalid_configuration("proj_out.weight shape"));
    }
    let multi_layer_text_features = probe
        .tensor_shapes()
        .contains_key(&format!("{prefix}txt_norm.0.weight"));
    let (text_feature_dimension, selected_text_layer_count) = if multi_layer_text_features {
        let norm = shape(probe, &format!("{prefix}txt_norm.0.weight"))?;
        let [dimension] = norm else {
            return Err(invalid_configuration("txt_norm.0.weight rank"));
        };
        let count = probe.consecutive_block_count(&format!("{prefix}txt_norm.{{}}."))?;
        (*dimension, count)
    } else {
        let norm = shape(probe, &format!("{prefix}txt_norm.weight"))?;
        let [dimension] = norm else {
            return Err(invalid_configuration("txt_norm.weight rank"));
        };
        (*dimension, 1)
    };
    if text_feature_dimension == 0 || selected_text_layer_count == 0 {
        return Err(invalid_configuration("text feature configuration"));
    }
    let layer_count =
        probe.consecutive_block_count(&format!("{prefix}transformer_blocks.{{}}."))?;
    if layer_count == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }
    Ok(LensConfiguration {
        layout,
        hidden_size: *hidden_size,
        in_channels: *in_channels,
        out_channels: output[0] / patch_area,
        patch_size: MODEL_FAMILY_PATCH_SIZE,
        layer_count,
        attention_heads: hidden_size / attention_head_dimension,
        attention_head_dimension: *attention_head_dimension,
        text_feature_dimension,
        selected_text_layer_count,
        multi_layer_text_features,
        rope_axes_dimensions: [8, 28, 28],
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
        "Lens configuration is invalid: {}",
        message.into()
    ))
}
