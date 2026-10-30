use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
};
use crate::model_family::{
    MODEL_FAMILY_EXECUTION_PROJECTION_SCHEMA_VERSION,
    ModelFamilyExecutionProjectionDefinition, ModelFamilyExecutionProjectionTensorDefinition,
    ModelFamilyExecutionProjectionValue,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "AuraFlow";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0064";
pub const MODEL_FAMILY_FIXTURE: &str = "auraflow-comfy-model-0064";
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "af15a94161e82214f1b1d8288c6c4fa810d1cfdc73691d684533d5abd1a009a5";
pub const SOURCE_ARCHITECTURE: &str = "model_base.AuraFlow";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 22;
pub const SOURCE_CONDITIONING_DIMENSION: u64 = 2_048;
pub const SOURCE_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const SOURCE_SAMPLING_SHIFT: f64 = 1.73;
pub const SOURCE_MEMORY_USAGE_FACTOR: f64 = 2.0;

pub const DENOISER_INVOCATION_REQUIRED_KEYS: &[&str] = &[
    "native.init_x_linear.weight",
    "native.init_x_linear.bias",
    "native.positional_encoding",
    "native.register_tokens",
    "native.cond_seq_linear.weight",
    "native.t_embedder.mlp.0.weight",
    "native.t_embedder.mlp.0.bias",
    "native.t_embedder.mlp.2.weight",
    "native.t_embedder.mlp.2.bias",
    "native.double_layers.0.modC.1.weight",
    "native.double_layers.0.modX.1.weight",
    "native.double_layers.0.attn.w1q.weight",
    "native.double_layers.0.attn.w1k.weight",
    "native.double_layers.0.attn.w1v.weight",
    "native.double_layers.0.attn.w1o.weight",
    "native.double_layers.0.attn.w2q.weight",
    "native.double_layers.0.attn.w2k.weight",
    "native.double_layers.0.attn.w2v.weight",
    "native.double_layers.0.attn.w2o.weight",
    "native.double_layers.0.mlpC.c_fc1.weight",
    "native.double_layers.0.mlpC.c_fc2.weight",
    "native.double_layers.0.mlpC.c_proj.weight",
    "native.double_layers.0.mlpX.c_fc1.weight",
    "native.double_layers.0.mlpX.c_fc2.weight",
    "native.double_layers.0.mlpX.c_proj.weight",
    "native.single_layers.0.modCX.1.weight",
    "native.single_layers.0.attn.w1q.weight",
    "native.single_layers.0.attn.w1k.weight",
    "native.single_layers.0.attn.w1v.weight",
    "native.single_layers.0.attn.w1o.weight",
    "native.single_layers.0.mlp.c_fc1.weight",
    "native.single_layers.0.mlp.c_fc2.weight",
    "native.single_layers.0.mlp.c_proj.weight",
    "native.modF.1.weight",
    "native.final_linear.weight",
];
pub const DENOISER_INVOCATION_LATENT_RANK: usize = 4;
pub const DENOISER_INVOCATION_CHANNELS: usize = 4;
pub const DENOISER_INVOCATION_WIDTH: usize = 2;
pub const DENOISER_INVOCATION_CONTEXT_WIDTH: usize = SOURCE_CONDITIONING_DIMENSION as usize;
pub const DENOISER_INVOCATION_PATCH_SIZE: usize = 2;
pub const DENOISER_INVOCATION_REGISTER_TOKENS: usize = 8;
pub const DENOISER_INVOCATION_MLP_WIDTH: usize = 256;

macro_rules! execution_tensor {
    ($key:literal, $shape:expr) => {
        ModelFamilyExecutionProjectionTensorDefinition {
            key: $key,
            shape: $shape,
            value: ModelFamilyExecutionProjectionValue::Finite,
        }
    };
}

const EXECUTION_PROJECTION_TENSORS: &[ModelFamilyExecutionProjectionTensorDefinition] = &[
    execution_tensor!("native.init_x_linear.weight", &[2, 16]),
    execution_tensor!("native.init_x_linear.bias", &[2]),
    execution_tensor!("native.positional_encoding", &[1, 16, 2]),
    execution_tensor!("native.register_tokens", &[1, 8, 2]),
    execution_tensor!("native.cond_seq_linear.weight", &[2, 2_048]),
    execution_tensor!("native.t_embedder.mlp.0.weight", &[2, 256]),
    execution_tensor!("native.t_embedder.mlp.0.bias", &[2]),
    execution_tensor!("native.t_embedder.mlp.2.weight", &[2, 2]),
    execution_tensor!("native.t_embedder.mlp.2.bias", &[2]),
    execution_tensor!("native.double_layers.0.modC.1.weight", &[12, 2]),
    execution_tensor!("native.double_layers.0.modX.1.weight", &[12, 2]),
    execution_tensor!("native.double_layers.0.attn.w1q.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w1k.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w1v.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w1o.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w2q.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w2k.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w2v.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.attn.w2o.weight", &[2, 2]),
    execution_tensor!("native.double_layers.0.mlpC.c_fc1.weight", &[256, 2]),
    execution_tensor!("native.double_layers.0.mlpC.c_fc2.weight", &[256, 2]),
    execution_tensor!("native.double_layers.0.mlpC.c_proj.weight", &[2, 256]),
    execution_tensor!("native.double_layers.0.mlpX.c_fc1.weight", &[256, 2]),
    execution_tensor!("native.double_layers.0.mlpX.c_fc2.weight", &[256, 2]),
    execution_tensor!("native.double_layers.0.mlpX.c_proj.weight", &[2, 256]),
    execution_tensor!("native.single_layers.0.modCX.1.weight", &[12, 2]),
    execution_tensor!("native.single_layers.0.attn.w1q.weight", &[2, 2]),
    execution_tensor!("native.single_layers.0.attn.w1k.weight", &[2, 2]),
    execution_tensor!("native.single_layers.0.attn.w1v.weight", &[2, 2]),
    execution_tensor!("native.single_layers.0.attn.w1o.weight", &[2, 2]),
    execution_tensor!("native.single_layers.0.mlp.c_fc1.weight", &[256, 2]),
    execution_tensor!("native.single_layers.0.mlp.c_fc2.weight", &[256, 2]),
    execution_tensor!("native.single_layers.0.mlp.c_proj.weight", &[2, 256]),
    execution_tensor!("native.modF.1.weight", &[4, 2]),
    execution_tensor!("native.final_linear.weight", &[16, 2]),
];

pub(crate) const DENOISER_EXECUTION_PROJECTION: ModelFamilyExecutionProjectionDefinition =
    ModelFamilyExecutionProjectionDefinition {
        schema_version: MODEL_FAMILY_EXECUTION_PROJECTION_SCHEMA_VERSION,
        identifier: "auraflow-reduced-denoiser-v1",
        family_feature_id: MODEL_FAMILY_FEATURE_ID,
        family_identifier: MODEL_FAMILY_IDENTIFIER,
        source_ordinal: MODEL_FAMILY_SOURCE_ORDINAL,
        source_architecture: SOURCE_ARCHITECTURE,
        tensors: EXECUTION_PROJECTION_TENSORS,
    };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuraFlowConfiguration {
    pub layout: ModelStateLayout,
    pub maximum_sequence_length: u64,
    pub conditioning_dimension: u64,
    pub double_layer_count: usize,
    pub layer_count: usize,
}

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "conditioning",
        required: false,
    },
];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.aura_t5.AuraT5Tokenizer",
        clip_model: "sim.text_encoders.aura_t5.AuraT5Model",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::AnyKeyPresent {
    keys: &[
        "model.diffusion_model.double_layers.0.attn.w1q.weight",
        "joint_transformer_blocks.0.attn.add_k_proj.weight",
    ],
    score: 1_000,
}];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.double_layers.0.attn.w1q.weight",
    "native.single_layers.0.attn.w1q.weight",
    "native.final_linear.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.cond_seq_linear.weight",
    "native.double_layers.0.attn.w1k.weight",
    "native.positional_encoding",
    "native.register_tokens",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "double_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.double_layers.0.attn.w1q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "double_stream_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "single_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.single_layers.0.attn.w1q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "single_stream_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "final_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_linear.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "flow_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "auraflow-mmdit-v1",
    latent_feature_id: "COMFY-MODEL-0047",
    latent_identifier: "SDXL",
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

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Route":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}},
            {"Route":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Route":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched": "Reject"
    }"#,
};

const DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Exact":"context_embedder.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.cond_seq_linear.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"joint_transformer_blocks.0.attn.add_q_proj.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.double_layers.0.attn.w1q.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"joint_transformer_blocks.0.attn.add_k_proj.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.double_layers.0.attn.w1k.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"proj_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.final_linear.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"pos_embed.pos_embed"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.positional_encoding"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"register_tokens"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.register_tokens"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.attn.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.single_layers.0.attn.w1q.weight"},"component":"denoiser"}},
            {"Route":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Route":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched": "Reject"
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
            "model.diffusion_model.cond_seq_linear.weight",
            "model.diffusion_model.positional_encoding",
            "model.diffusion_model.double_layers.0.attn.w1q.weight",
            "model.diffusion_model.single_layers.0.attn.w1q.weight",
            "model.diffusion_model.final_linear.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "context_embedder.weight",
            "pos_embed.pos_embed",
            "joint_transformer_blocks.0.attn.add_q_proj.weight",
            "joint_transformer_blocks.0.attn.add_k_proj.weight",
            "single_transformer_blocks.0.attn.to_q.weight",
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
    source_ordinal: 22,
    source_architecture: SOURCE_ARCHITECTURE,
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
) -> Result<AuraFlowConfiguration, ModelFamilyError> {
    match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => native_configuration(probe),
        ModelStateLayout::Diffusers => diffusers_configuration(probe),
        ModelStateLayout::StandaloneNative => Err(ModelFamilyError::InvalidSelectorOutput(
            "AuraFlow standalone-native layout is unsupported".to_owned(),
        )),
    }
}

fn native_configuration(probe: &ModelProbe) -> Result<AuraFlowConfiguration, ModelFamilyError> {
    let conditioning_dimension =
        dimension(probe, "model.diffusion_model.cond_seq_linear.weight", 1)?;
    require_expected_dimension(
        "model.diffusion_model.cond_seq_linear.weight",
        1,
        conditioning_dimension,
        SOURCE_CONDITIONING_DIMENSION,
    )?;
    let maximum_sequence_length = dimension(probe, "model.diffusion_model.positional_encoding", 1)?;
    let double_layer_count = consecutive_blocks(probe, "model.diffusion_model.double_layers.{}.")?;
    let single_layer_count = consecutive_blocks(probe, "model.diffusion_model.single_layers.{}.")?;
    Ok(AuraFlowConfiguration {
        layout: ModelStateLayout::PrefixedNative,
        maximum_sequence_length,
        conditioning_dimension,
        double_layer_count,
        layer_count: double_layer_count
            .checked_add(single_layer_count)
            .ok_or(ModelFamilyError::ProbeDimensionOverflow)?,
    })
}

fn diffusers_configuration(probe: &ModelProbe) -> Result<AuraFlowConfiguration, ModelFamilyError> {
    let conditioning_dimension = dimension(probe, "context_embedder.weight", 1)?;
    require_expected_dimension(
        "context_embedder.weight",
        1,
        conditioning_dimension,
        SOURCE_CONDITIONING_DIMENSION,
    )?;
    let maximum_sequence_length = dimension(probe, "pos_embed.pos_embed", 1)?;
    let double_layer_count = consecutive_blocks(probe, "joint_transformer_blocks.{}.")?;
    let single_layer_count = consecutive_blocks(probe, "single_transformer_blocks.{}.")?;
    Ok(AuraFlowConfiguration {
        layout: ModelStateLayout::Diffusers,
        maximum_sequence_length,
        conditioning_dimension,
        double_layer_count,
        layer_count: double_layer_count
            .checked_add(single_layer_count)
            .ok_or(ModelFamilyError::ProbeDimensionOverflow)?,
    })
}

fn dimension(probe: &ModelProbe, key: &str, dimension: usize) -> Result<u64, ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .and_then(|shape| shape.get(dimension))
        .copied()
        .ok_or_else(|| {
            ModelFamilyError::InvalidSelectorOutput(format!(
                "AuraFlow probe is missing dimension {dimension} for {key}"
            ))
        })
}

fn require_expected_dimension(
    key: &str,
    dimension: usize,
    actual: u64,
    expected: u64,
) -> Result<(), ModelFamilyError> {
    if actual != expected {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "AuraFlow {key} dimension {dimension} is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn consecutive_blocks(probe: &ModelProbe, pattern: &str) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    if count == 0 {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "AuraFlow probe has no consecutive blocks for {pattern}"
        )));
    }
    Ok(count)
}
