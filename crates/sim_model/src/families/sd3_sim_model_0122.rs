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

pub const MODEL_FAMILY_IDENTIFIER: &str = "SD3";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0122";
pub const MODEL_FAMILY_FIXTURE: &str = "sd3-comfy-model-0122";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 19;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "3a756043942115b4b97dca9d4640e18701f8cc94199fb2d575c9862fd5491598";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.6;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 3.0;
pub const SOURCE_INPUT_CHANNELS: u64 = 16;
pub const SOURCE_PATCH_SIZE: u64 = 2;
pub const SOURCE_HEAD_DIMENSION: u64 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sd3Configuration {
    pub layout: ModelStateLayout,
    pub in_channels: u64,
    pub patch_size: u64,
    pub hidden_size: u64,
    pub attention_head_count: u64,
    pub block_count: usize,
    pub clip_l: bool,
    pub clip_g: bool,
    pub t5xxl: bool,
}

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Bind {
        parameter: "clip_l",
        source: "text_encoders.clip_l.transformer.text_model.final_layer_norm.weight",
    },
    ModelClipConfigurationFactDefinition::Bind {
        parameter: "clip_g",
        source: "text_encoders.clip_g.transformer.text_model.final_layer_norm.weight",
    },
    ModelClipConfigurationFactDefinition::Bind {
        parameter: "t5",
        source: "text_encoders.t5xxl.transformer.detected",
    },
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    },
];
const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.sd3_clip.SD3Tokenizer",
        clip_model: "sim.text_encoders.sd3_clip.sd3_clip",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];
const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: true,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "SD3 multimodal diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SD3 CLIP-L, CLIP-G, and T5XXL conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "SD3 sixteen-channel latent codec",
        required: false,
    },
];

const INPUT_KEYS: &[&str] = &[
    "model.diffusion_model.x_embedder.proj.weight",
    "pos_embed.proj.weight",
];
const JOINT_ATTENTION_KEYS: &[&str] = &[
    "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight",
    "transformer_blocks.0.attn.add_q_proj.weight",
];
const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: INPUT_KEYS,
        score: 350,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 1,
        values: &[SOURCE_INPUT_CHANNELS],
        score: 300,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: JOINT_ATTENTION_KEYS,
        score: 350,
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
    "native.joint_blocks.0.x_block.attn.qkv.weight",
    "native.joint_blocks.0.context_block.attn.qkv.weight",
    "native.joint_blocks.0.x_block.attn.proj.weight",
    "native.final_layer.linear.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.pos_embed",
    "native.context_embedder.weight",
    "native.joint_blocks.0.context_block.attn.proj.weight",
    "native.final_layer.linear.bias",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];
const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "sd3_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.mlp.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "sd3_image_attention",
        operation: ModelForwardOperation::Linear {
            weight: "native.joint_blocks.0.x_block.attn.proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "sd3_joint_conditioning",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "sd3_final_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "sd3_flow_prediction",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sd3-mmdit-v1",
    latent_feature_id: "COMFY-MODEL-0046",
    latent_identifier: "SD3",
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
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":""}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Assemble":{"sources":[{"Source":"transformer_blocks.0.attn.to_q.weight"},{"Source":"transformer_blocks.0.attn.to_k.weight"},{"Source":"transformer_blocks.0.attn.to_v.weight"}],"dimension":0,"output":{"component":"denoiser","key":"native.joint_blocks.0.x_block.attn.qkv.weight"}}},
            {"Assemble":{"sources":[{"Source":"transformer_blocks.0.attn.add_q_proj.weight"},{"Source":"transformer_blocks.0.attn.add_k_proj.weight"},{"Source":"transformer_blocks.0.attn.add_v_proj.weight"}],"dimension":0,"output":{"component":"denoiser","key":"native.joint_blocks.0.context_block.attn.qkv.weight"}}},
            {"Move":{"selector":{"predicate":{"Exact":"pos_embed.proj.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.x_embedder.proj.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_text_embed.timestep_embedder.linear_1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.t_embedder.mlp.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.to_out.0.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.joint_blocks.0.x_block.attn.proj.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"proj_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.final_layer.linear.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":""}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"native."}},"component":"vae"}}
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
            "model.diffusion_model.t_embedder.mlp.0.weight",
            "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight",
            "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight",
            "model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "pos_embed.proj.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.attn.add_k_proj.weight",
            "transformer_blocks.0.attn.add_v_proj.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "proj_out.weight",
        ],
        required_prefixes: &[],
    },
];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
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
    ModelFamilyComponentStateSchema {
        component: "vae",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 19,
    source_architecture: "model_base.SD3",
    source_configuration: &[],
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
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Sd3Configuration, ModelFamilyError> {
    let layout = probe.select_layout(LAYOUT_SIGNATURES)?;
    let (input_key, block_pattern) = match layout {
        ModelStateLayout::PrefixedNative => (
            "model.diffusion_model.x_embedder.proj.weight",
            "model.diffusion_model.joint_blocks.{}.",
        ),
        ModelStateLayout::Diffusers => ("pos_embed.proj.weight", "transformer_blocks.{}."),
        ModelStateLayout::StandaloneNative => {
            return Err(invalid("standalone-native layout is unsupported"));
        }
    };
    let input = shape(probe, input_key, 4)?;
    if input[1] != SOURCE_INPUT_CHANNELS
        || input[2] != SOURCE_PATCH_SIZE
        || input[3] != SOURCE_PATCH_SIZE
        || input[0] % SOURCE_HEAD_DIMENSION != 0
    {
        return Err(invalid(format!(
            "{input_key} must be [heads*{SOURCE_HEAD_DIMENSION}, {SOURCE_INPUT_CHANNELS}, {SOURCE_PATCH_SIZE}, {SOURCE_PATCH_SIZE}]"
        )));
    }
    let block_count = probe.consecutive_block_count(block_pattern)?;
    if block_count == 0 {
        return Err(invalid("MMDiT has no consecutive joint transformer blocks"));
    }
    Ok(Sd3Configuration {
        layout,
        in_channels: SOURCE_INPUT_CHANNELS,
        patch_size: SOURCE_PATCH_SIZE,
        hidden_size: input[0],
        attention_head_count: input[0] / SOURCE_HEAD_DIMENSION,
        block_count,
        clip_l: probe.tensor_shapes().contains_key(
            "text_encoders.clip_l.transformer.text_model.final_layer_norm.weight",
        ),
        clip_g: probe.tensor_shapes().contains_key(
            "text_encoders.clip_g.transformer.text_model.final_layer_norm.weight",
        ),
        t5xxl: probe
            .tensor_shapes()
            .keys()
            .any(|key| key.starts_with("text_encoders.t5xxl.transformer.")),
    })
}

fn shape<'a>(probe: &'a ModelProbe, key: &str, rank: usize) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != rank || shape.contains(&0) {
        return Err(invalid(format!("{key} must have non-zero rank {rank}")));
    }
    Ok(shape)
}

fn invalid(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "SD3 source configuration mismatch: {}",
        message.into()
    ))
}
