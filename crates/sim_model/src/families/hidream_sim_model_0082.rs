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

pub const MODEL_FAMILY_IDENTIFIER: &str = "HiDream";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0082";
pub const MODEL_FAMILY_FIXTURE: &str = "hidream-comfy-model-0082";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 69;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "ea914e4a82a8979ffb5b98b98c0f842d2e8820ebed9337f12c00d8cf919561ab";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HiDreamLayout {
    Native,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HiDreamConfiguration {
    pub layout: HiDreamLayout,
    pub patch_size: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub number_of_layers: usize,
    pub number_of_single_layers: usize,
    pub attention_head_dimension: u64,
    pub number_of_attention_heads: u64,
    pub inner_dimension: u64,
    pub caption_channels: [u64; 2],
    pub text_embedding_dimension: u64,
    pub number_of_routed_experts: u64,
    pub number_of_activated_experts: u64,
    pub rope_axes_dimensions: [u64; 3],
    pub maximum_resolution: [u64; 2],
    pub llama_layer_count: usize,
}

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "HiDream image mixture-of-experts diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Flux latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "optional external HiDream conditioning state",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.caption_projection.0.linear.weight",
            "caption_projection.0.linear.weight",
        ],
        score: 400,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.double_stream_blocks.0.block.attn1.to_out.weight",
            "double_stream_blocks.0.block.attn1.to_out.weight",
        ],
        score: 300,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.final_layer.linear.weight",
            "final_layer.linear.weight",
        ],
        score: 300,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "model.diffusion_model.",
        target_prefix: "native.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "vae.",
        target_prefix: "vae.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "text_encoders.",
        target_prefix: "text_encoder.",
        required: false,
    },
];

const REQUIRED_KEYS: &[&str] = &[
    "native.x_embedder.proj.weight",
    "native.t_embedder.timestep_embedder.linear_1.weight",
    "native.t_embedder.timestep_embedder.linear_1.bias",
    "native.t_embedder.timestep_embedder.linear_2.weight",
    "native.t_embedder.timestep_embedder.linear_2.bias",
    "native.caption_projection.0.linear.weight",
    "native.double_stream_blocks.0.block.attn1.to_out.weight",
    "native.single_stream_blocks.0.block.attn1.to_out.weight",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.proj.bias",
    "native.p_embedder.pooled_embedder.linear_1.weight",
    "native.p_embedder.pooled_embedder.linear_1.bias",
    "native.p_embedder.pooled_embedder.linear_2.weight",
    "native.p_embedder.pooled_embedder.linear_2.bias",
    "native.double_stream_blocks.0.block.attn1.to_q.weight",
    "native.double_stream_blocks.0.block.attn1.to_k.weight",
    "native.double_stream_blocks.0.block.attn1.to_v.weight",
    "native.single_stream_blocks.0.block.attn1.to_q.weight",
    "native.single_stream_blocks.0.block.attn1.to_k.weight",
    "native.single_stream_blocks.0.block.attn1.to_v.weight",
    "native.final_layer.adaLN_modulation.1.weight",
    "native.final_layer.adaLN_modulation.1.bias",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.timestep_embedder.linear_1.weight",
            bias: Some("native.t_embedder.timestep_embedder.linear_1.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.timestep_embedder.linear_2.weight",
            bias: Some("native.t_embedder.timestep_embedder.linear_2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "caption_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.caption_projection.0.linear.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "double_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.double_stream_blocks.0.block.attn1.to_out.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "single_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.single_stream_blocks.0.block.attn1.to_out.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
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
    architecture_version: "hidream-image-moe-dit-v1",
    latent_feature_id: "COMFY-MODEL-0029",
    latent_identifier: "Flux",
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

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"x_embedder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t_embedder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"t_embedder.","to":"native.t_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"p_embedder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"p_embedder.","to":"native.p_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"caption_projection."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"caption_projection.","to":"native.caption_projection."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"double_stream_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_stream_blocks.","to":"native.double_stream_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"single_stream_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"single_stream_blocks.","to":"native.single_stream_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_layer."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &DIFFUSERS_STATE_PLAN,
    },
];

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.caption_projection.0.linear.weight",
            "model.diffusion_model.x_embedder.proj.weight",
            "model.diffusion_model.double_stream_blocks.0.block.attn1.to_out.weight",
            "model.diffusion_model.single_stream_blocks.0.block.attn1.to_out.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "caption_projection.0.linear.weight",
            "x_embedder.proj.weight",
            "double_stream_blocks.0.block.attn1.to_out.weight",
            "single_stream_blocks.0.block.attn1.to_out.weight",
            "final_layer.linear.weight",
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
    source_ordinal: 69,
    source_architecture: "model_base.HiDream",
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
) -> Result<HiDreamConfiguration, ModelFamilyError> {
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (HiDreamLayout::Native, "model.diffusion_model."),
        ModelStateLayout::Diffusers => (HiDreamLayout::Diffusers, ""),
        ModelStateLayout::StandaloneNative => {
            return Err(invalid_configuration(
                "standalone-native layout is unsupported",
            ));
        }
    };

    let caption_projection = shape(probe, &format!("{prefix}caption_projection.0.linear.weight"))?;
    let [detected_inner_dimension, caption_dimension] = caption_projection else {
        return Err(invalid_configuration("caption projection rank"));
    };
    if *detected_inner_dimension == 0 || *caption_dimension == 0 {
        return Err(invalid_configuration("caption projection shape"));
    }

    let patch_projection = shape(probe, &format!("{prefix}x_embedder.proj.weight"))?;
    if patch_projection.len() != 2
        || patch_projection[0] != *detected_inner_dimension
        || patch_projection[1] == 0
    {
        return Err(invalid_configuration("patch projection shape"));
    }

    for key in [
        "double_stream_blocks.0.block.attn1.to_out.weight",
        "single_stream_blocks.0.block.attn1.to_out.weight",
    ] {
        let projection = shape(probe, &format!("{prefix}{key}"))?;
        if projection != [*detected_inner_dimension, *detected_inner_dimension] {
            return Err(invalid_configuration(format!("{key} shape")));
        }
    }

    let final_projection = shape(probe, &format!("{prefix}final_layer.linear.weight"))?;
    if final_projection.len() != 2
        || final_projection[0] == 0
        || final_projection[1] != *detected_inner_dimension
    {
        return Err(invalid_configuration("final projection shape"));
    }

    let number_of_layers =
        probe.consecutive_block_count(&format!("{prefix}double_stream_blocks.{{}}."))?;
    let number_of_single_layers =
        probe.consecutive_block_count(&format!("{prefix}single_stream_blocks.{{}}."))?;
    if number_of_layers != 16 || number_of_single_layers != 32 {
        return Err(invalid_configuration(format!(
            "expected 16 double-stream and 32 single-stream layers, found {number_of_layers} and {number_of_single_layers}"
        )));
    }

    Ok(HiDreamConfiguration {
        layout,
        patch_size: 2,
        in_channels: 16,
        out_channels: 16,
        number_of_layers,
        number_of_single_layers,
        attention_head_dimension: 128,
        number_of_attention_heads: 20,
        inner_dimension: 2_560,
        caption_channels: [4_096, 4_096],
        text_embedding_dimension: 2_048,
        number_of_routed_experts: 4,
        number_of_activated_experts: 2,
        rope_axes_dimensions: [64, 32, 32],
        maximum_resolution: [128, 128],
        llama_layer_count: 48,
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
        "HiDream configuration is invalid: {}",
        message.into()
    ))
}
