use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "ErnieImage";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0076";
pub const MODEL_FAMILY_FIXTURE: &str = "ernieimage-comfy-model-0076";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 86;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "00510ab5856b2429a3d1fa2ba36f6457f6a92ed8622d42a9e6d05332f02e371e";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 10.0;
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1_000.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 3.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErnieImageLayout {
    Native,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ErnieImageConfiguration {
    pub layout: ErnieImageLayout,
    pub hidden_size: u64,
    pub number_of_attention_heads: u64,
    pub number_of_layers: usize,
    pub feed_forward_hidden_size: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub patch_size: u64,
    pub text_input_dimension: u64,
    pub rope_theta: u64,
    pub rope_axes_dimensions: [u64; 3],
    pub qk_layer_normalization: bool,
}

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.ernie.ErnieTokenizer",
        clip_model: "sim.text_encoders.ernie.te",
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
        role: "Ernie image diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Flux2 latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Ernie Ministral conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.layers.0.mlp.linear_fc2.weight",
            "layers.0.mlp.linear_fc2.weight",
        ],
        score: 400,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.text_proj.weight",
            "text_proj.weight",
        ],
        score: 300,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.final_linear.weight",
            "final_linear.weight",
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
    "native.x_embedder.proj.weight",
    "native.text_proj.weight",
    "native.time_embedding.linear_1.weight",
    "native.time_embedding.linear_1.bias",
    "native.time_embedding.linear_2.weight",
    "native.time_embedding.linear_2.bias",
    "native.layers.0.self_attention.to_q.weight",
    "native.layers.0.self_attention.norm_q.weight",
    "native.layers.0.mlp.linear_fc2.weight",
    "native.final_linear.weight",
    "native.final_linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.proj.bias",
    "native.layers.0.self_attention.to_k.weight",
    "native.layers.0.self_attention.to_v.weight",
    "native.layers.0.self_attention.norm_k.weight",
    "native.layers.0.self_attention.to_out.0.weight",
    "native.layers.0.mlp.gate_proj.weight",
    "native.layers.0.mlp.up_proj.weight",
    "native.adaLN_modulation.1.weight",
    "native.adaLN_modulation.1.bias",
    "native.final_norm.linear.weight",
    "native.final_norm.linear.bias",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding.linear_1.weight",
            bias: Some("native.time_embedding.linear_1.bias"),
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
            weight: "native.time_embedding.linear_2.weight",
            bias: Some("native.time_embedding.linear_2.bias"),
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
        checkpoint: "transformer_mlp_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.layers.0.mlp.linear_fc2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_linear.weight",
            bias: Some("native.final_linear.bias"),
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
    architecture_version: "ernie-image-transformer-v1",
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
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
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
            {"Move":{"selector":{"predicate":{"Prefix":"text_proj."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_proj.","to":"native.text_proj."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_embedding."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_embedding.","to":"native.time_embedding."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"adaLN_modulation."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"adaLN_modulation.","to":"native.adaLN_modulation."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"layers."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"layers.","to":"native.layers."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_norm."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_norm.","to":"native.final_norm."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_linear."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_linear.","to":"native.final_linear."}},"component":"model"}},
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
            "model.diffusion_model.x_embedder.proj.weight",
            "model.diffusion_model.text_proj.weight",
            "model.diffusion_model.layers.0.self_attention.to_q.weight",
            "model.diffusion_model.layers.0.self_attention.norm_q.weight",
            "model.diffusion_model.layers.0.mlp.linear_fc2.weight",
            "model.diffusion_model.final_linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "x_embedder.proj.weight",
            "text_proj.weight",
            "layers.0.self_attention.to_q.weight",
            "layers.0.self_attention.norm_q.weight",
            "layers.0.mlp.linear_fc2.weight",
            "final_linear.weight",
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
    source_ordinal: 86,
    source_architecture: "model_base.ErnieImage",
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
) -> Result<ErnieImageConfiguration, ModelFamilyError> {
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (ErnieImageLayout::Native, "model.diffusion_model."),
        ModelStateLayout::Diffusers => (ErnieImageLayout::Diffusers, ""),
        ModelStateLayout::StandaloneNative => {
            return Err(invalid_configuration(
                "standalone-native layout is unsupported",
            ));
        }
    };
    let layer_marker = shape(probe, &format!("{prefix}layers.0.mlp.linear_fc2.weight"))?;
    if layer_marker.len() != 2 || layer_marker[0] == 0 || layer_marker[1] == 0 {
        return Err(invalid_configuration(
            "layers.0.mlp.linear_fc2.weight shape",
        ));
    }
    let hidden_size = layer_marker[0];
    let feed_forward_hidden_size = layer_marker[1];

    let patch = shape(probe, &format!("{prefix}x_embedder.proj.weight"))?;
    let [patch_hidden, in_channels, patch_height, patch_width] = patch else {
        return Err(invalid_configuration("x_embedder.proj.weight rank"));
    };
    if patch_hidden != &hidden_size || patch_height != patch_width || *patch_height == 0 {
        return Err(invalid_configuration("x_embedder.proj.weight shape"));
    }
    let patch_size = *patch_height;

    let query = shape(
        probe,
        &format!("{prefix}layers.0.self_attention.to_q.weight"),
    )?;
    if query != [hidden_size, hidden_size] {
        return Err(invalid_configuration("self-attention query shape"));
    }
    let query_norm = shape(
        probe,
        &format!("{prefix}layers.0.self_attention.norm_q.weight"),
    )?;
    let [head_dimension] = query_norm else {
        return Err(invalid_configuration("self-attention norm shape"));
    };
    if *head_dimension == 0 || hidden_size % head_dimension != 0 {
        return Err(invalid_configuration("attention head dimension"));
    }
    let number_of_attention_heads = hidden_size / head_dimension;

    let text_projection = shape(probe, &format!("{prefix}text_proj.weight"))?;
    if text_projection.len() != 2 || text_projection[0] != hidden_size {
        return Err(invalid_configuration("text_proj.weight shape"));
    }
    let text_input_dimension = text_projection[1];

    let final_projection = shape(probe, &format!("{prefix}final_linear.weight"))?;
    if final_projection.len() != 2
        || final_projection[1] != hidden_size
        || final_projection[0] % (patch_size * patch_size) != 0
    {
        return Err(invalid_configuration("final_linear.weight shape"));
    }
    let out_channels = final_projection[0] / (patch_size * patch_size);
    let number_of_layers = probe.consecutive_block_count(&format!("{prefix}layers.{{}}."))?;
    if number_of_layers == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }

    Ok(ErnieImageConfiguration {
        layout,
        hidden_size,
        number_of_attention_heads,
        number_of_layers,
        feed_forward_hidden_size,
        in_channels: *in_channels,
        out_channels,
        patch_size,
        text_input_dimension,
        rope_theta: 256,
        rope_axes_dimensions: [32, 48, 48],
        qk_layer_normalization: true,
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
        "ErnieImage configuration is invalid: {}",
        message.into()
    ))
}
