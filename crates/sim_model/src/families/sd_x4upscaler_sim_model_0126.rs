use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelConfigurationKind,
    ModelConfigurationValue, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "SD_X4Upscaler";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0126";
pub const MODEL_FAMILY_FIXTURE: &str = "sd-x4upscaler-comfy-model-0126";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 14;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "c32bbf1e1ce2a8442b4007bac0dc7c24d4472502bcaa2a757ab1b5fdadb75874";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.0;
pub const MODEL_FAMILY_LINEAR_START: f64 = 0.0001;
pub const MODEL_FAMILY_LINEAR_END: f64 = 0.02;

const INPUT_WEIGHT: &str = "model.diffusion_model.input_blocks.0.0.weight";
const NATIVE_INPUT_SHAPE: &[u64] = &[256, 7, 3, 3];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdX4UpscalerConfiguration {
    pub model_channels: u64,
    pub input_channels: u64,
    pub context_dimension: u64,
    pub linear_transformer_projection: bool,
    pub temporal_attention: bool,
}

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.sd2_clip.SD2Tokenizer",
        clip_model: "sim.text_encoders.sd2_clip.SD2ClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "x4 latent upscaler denoiser",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "image latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SD2 text conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::ExactShape {
    key: INPUT_WEIGHT,
    shape: NATIVE_INPUT_SHAPE,
    score: 1_000,
}];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.input_blocks.1.0.in_layers.2.weight",
    "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
    "native.output_blocks.0.0.in_layers.2.weight",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.input_blocks.0.0.weight",
    "native.input_blocks.1.0.in_layers.0.weight",
    "native.input_blocks.1.0.out_layers.3.weight",
    "native.input_blocks.1.1.proj_in.weight",
    "native.out.2.weight",
    "native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "native.output_blocks.0.0.in_layers.0.weight",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "noise_level_conditioning",
        operation: ModelForwardOperation::Linear {
            weight: "native.input_blocks.1.0.in_layers.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "upscaler_input_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "middle_transformer_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "middle_transformer_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "upscaler_output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.output_blocks.0.0.in_layers.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "upscaled_latent",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sd-x4-upscaler-unet-v1",
    latent_feature_id: "COMFY-MODEL-0049",
    latent_identifier: "SD_X4",
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

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] =
    &[ModelSourceConfigurationRule::ExactTensorShape {
        key: INPUT_WEIGHT,
        shape: NATIVE_INPUT_SHAPE,
    }];

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
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

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[ModelFamilyStatePlanCase {
    layout: ModelStateLayout::PrefixedNative,
    plan: &NATIVE_STATE_PLAN,
}];

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[ModelLayoutSignature {
    layout: ModelStateLayout::PrefixedNative,
    required_keys: &[
        INPUT_WEIGHT,
        "model.diffusion_model.input_blocks.1.0.in_layers.2.weight",
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
        "model.diffusion_model.output_blocks.0.0.in_layers.2.weight",
    ],
    required_prefixes: &[],
}];

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
    source_ordinal: 14,
    source_architecture: "model_base.SD_X4Upscaler",
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
) -> Result<SdX4UpscalerConfiguration, ModelFamilyError> {
    let configuration = probe.normalized_configuration()?;
    if configuration.kind() != ModelConfigurationKind::Native {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "SD_X4Upscaler is not present in the pinned source Diffusers detector table".to_owned(),
        ));
    }
    let model_channels = unsigned_fact(&configuration, "model_channels")?;
    let input_channels = unsigned_fact(&configuration, "in_channels")?;
    let context_dimension = unsigned_fact(&configuration, "context_dim")?;
    let linear_transformer_projection = boolean_fact(&configuration, "use_linear_in_transformer")?;
    let temporal_attention = boolean_fact(&configuration, "use_temporal_attention")?;
    if model_channels != 256
        || input_channels != 7
        || context_dimension != 1_024
        || !linear_transformer_projection
        || temporal_attention
        || !matches!(configuration.fact("adm_in_channels"), Some(ModelConfigurationValue::None))
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "SD_X4Upscaler normalized configuration is incompatible: {:?}",
            configuration.facts()
        )));
    }
    Ok(SdX4UpscalerConfiguration {
        model_channels,
        input_channels,
        context_dimension,
        linear_transformer_projection,
        temporal_attention,
    })
}

fn unsigned_fact(
    configuration: &crate::ModelNormalizedConfiguration,
    name: &str,
) -> Result<u64, ModelFamilyError> {
    match configuration.fact(name) {
        Some(ModelConfigurationValue::Unsigned(value)) => Ok(*value),
        value => Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "SD_X4Upscaler requires unsigned {name}, found {value:?}"
        ))),
    }
}

fn boolean_fact(
    configuration: &crate::ModelNormalizedConfiguration,
    name: &str,
) -> Result<bool, ModelFamilyError> {
    match configuration.fact(name) {
        Some(ModelConfigurationValue::Boolean(value)) => Ok(*value),
        value => Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "SD_X4Upscaler requires boolean {name}, found {value:?}"
        ))),
    }
}
