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

pub const MODEL_FAMILY_IDENTIFIER: &str = "GenmoMochi";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0081";
pub const MODEL_FAMILY_FIXTURE: &str = "genmomochi-comfy-model-0081";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 31;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "e89d34b2cd18e9128e0d47ddb06f430f998cee5c7cd88922629498afc407fafc";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 6.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenmoMochiLayout {
    Native,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenmoMochiConfiguration {
    pub layout: GenmoMochiLayout,
    pub depth: u64,
    pub patch_size: u64,
    pub number_of_attention_heads: u64,
    pub hidden_size_x: u64,
    pub hidden_size_y: u64,
    pub text_feature_dimension: u64,
    pub in_channels: u64,
    pub out_channels: u64,
    pub mlp_ratio_x: u64,
    pub mlp_ratio_y: u64,
    pub qk_normalization: bool,
    pub qkv_bias: bool,
    pub output_bias: bool,
    pub patch_embedding_bias: bool,
    pub positional_encoding_preserves_area: bool,
    pub timestep_mlp_bias: bool,
    pub attends_to_padding: bool,
}

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.genmo.MochiT5Tokenizer",
        clip_model: "sim.text_encoders.genmo.mochi_te",
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
        role: "Genmo Mochi asymmetric video diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "generated Mochi token-count conditioning",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Mochi latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Mochi T5 conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.t5_yproj.weight",
            "t5_yproj.weight",
        ],
        score: 300,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.blocks.0.attn.proj_x.weight",
            "blocks.0.attn.proj_x.weight",
        ],
        score: 400,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.final_layer.linear.weight",
            "final_layer.linear.weight",
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
    "native.t_embedder.mlp.0.weight",
    "native.t_embedder.mlp.0.bias",
    "native.t_embedder.mlp.2.weight",
    "native.t_embedder.mlp.2.bias",
    "native.t5_yproj.weight",
    "native.blocks.0.attn.proj_x.weight",
    "native.blocks.0.attn.proj_x.bias",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.proj.bias",
    "native.t5_yproj.bias",
    "native.t5_y_embedder.to_kv.weight",
    "native.pos_frequencies",
    "native.blocks.0.attn.qkv_x.weight",
    "native.blocks.0.attn.qkv_y.weight",
    "native.blocks.0.attn.q_norm_x.weight",
    "native.blocks.0.attn.k_norm_x.weight",
    "native.blocks.0.mlp_x.w1.weight",
    "native.blocks.0.mlp_x.w2.weight",
    "native.final_layer.mod.weight",
    "native.final_layer.mod.bias",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.mlp.0.weight",
            bias: Some("native.t_embedder.mlp.0.bias"),
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
            weight: "native.t_embedder.mlp.2.weight",
            bias: Some("native.t_embedder.mlp.2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "joint_block_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.attn.proj_x.weight",
            bias: Some("native.blocks.0.attn.proj_x.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "joint_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "video_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "genmo-mochi-asymmetric-joint-dit-v1",
    latent_feature_id: "COMFY-MODEL-0041",
    latent_identifier: "Mochi",
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
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":256.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"num_tokens_default"}}}
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
            {"Move":{"selector":{"predicate":{"Prefix":"t5_yproj."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"t5_yproj.","to":"native.t5_yproj."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t5_y_embedder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"t5_y_embedder.","to":"native.t5_y_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Exact":"pos_frequencies"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Prefix":{"from":"pos_frequencies","to":"native.pos_frequencies"}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"blocks.","to":"native.blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_layer."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":256.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"num_tokens_default"}}}
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
            "model.diffusion_model.t5_yproj.weight",
            "model.diffusion_model.x_embedder.proj.weight",
            "model.diffusion_model.blocks.0.attn.proj_x.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "t5_yproj.weight",
            "x_embedder.proj.weight",
            "blocks.0.attn.proj_x.weight",
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
        component: "runtime_conditioning",
        required_keys: &["num_tokens_default"],
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
    source_ordinal: 31,
    source_architecture: "model_base.GenmoMochi",
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
) -> Result<GenmoMochiConfiguration, ModelFamilyError> {
    let (layout, prefix) = match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (GenmoMochiLayout::Native, "model.diffusion_model."),
        ModelStateLayout::Diffusers => (GenmoMochiLayout::Diffusers, ""),
        ModelStateLayout::StandaloneNative => {
            return Err(invalid_configuration(
                "standalone-native layout is unsupported",
            ));
        }
    };
    let text_projection = shape(probe, &format!("{prefix}t5_yproj.weight"))?;
    if text_projection.len() != 2 || text_projection[0] == 0 || text_projection[1] == 0 {
        return Err(invalid_configuration("t5_yproj.weight shape"));
    }
    let patch_projection = shape(probe, &format!("{prefix}x_embedder.proj.weight"))?;
    let [hidden_size_x, in_channels, height, width] = patch_projection else {
        return Err(invalid_configuration("x_embedder.proj.weight rank"));
    };
    if *in_channels != 12 || *height != 2 || *width != 2 || *hidden_size_x == 0 {
        return Err(invalid_configuration(
            "x_embedder.proj.weight requires [hidden, 12, 2, 2]",
        ));
    }
    let hidden_size_y = text_projection[0];
    let text_feature_dimension = text_projection[1];
    let block_projection = shape(probe, &format!("{prefix}blocks.0.attn.proj_x.weight"))?;
    if block_projection != [*hidden_size_x, *hidden_size_x] {
        return Err(invalid_configuration("blocks.0.attn.proj_x.weight shape"));
    }
    let final_projection = shape(probe, &format!("{prefix}final_layer.linear.weight"))?;
    if final_projection.len() != 2
        || final_projection[0] == 0
        || final_projection[1] != *hidden_size_x
    {
        return Err(invalid_configuration("final_layer.linear.weight shape"));
    }

    Ok(GenmoMochiConfiguration {
        layout,
        depth: 48,
        patch_size: 2,
        number_of_attention_heads: 24,
        hidden_size_x: *hidden_size_x,
        hidden_size_y,
        text_feature_dimension,
        in_channels: *in_channels,
        out_channels: 12,
        mlp_ratio_x: 4,
        mlp_ratio_y: 4,
        qk_normalization: true,
        qkv_bias: false,
        output_bias: true,
        patch_embedding_bias: true,
        positional_encoding_preserves_area: true,
        timestep_mlp_bias: true,
        attends_to_padding: false,
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
    ModelFamilyError::InvalidSelectorOutput(format!("GenmoMochi configuration {}", message.into()))
}
