use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor, ModelClipTargetDefinition,
    ModelClipTargetSelector, ModelConfigurationKind, ModelConfigurationValue, ModelDetectionRule,
    ModelFamilyComponent, ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep, ModelLayoutSignature,
    ModelProbe, ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "SV3D_p";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0128";
pub const MODEL_FAMILY_FIXTURE: &str = "svthree-d-p-comfy-model-0128";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 18;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "b280af9213163e875b698aa6cb726ba6d88b76ccefdffa8147b78771c4e29554";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const MODEL_FAMILY_SIGMA_MAX: f64 = 700.0;
pub const MODEL_FAMILY_SIGMA_MIN: f64 = 0.002;
pub const MODEL_FAMILY_ADM_IN_CHANNELS: u64 = 1_280;

const INPUT_WEIGHT: &str = "model.diffusion_model.input_blocks.0.0.weight";
const LABEL_WEIGHT: &str = "model.diffusion_model.label_emb.0.0.weight";
const NATIVE_INPUT_SHAPE: &[u64] = &[320, 8, 3, 3];
const LABEL_SHAPE: &[u64] = &[2, MODEL_FAMILY_ADM_IN_CHANNELS];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sv3dPConfiguration {
    pub model_channels: u64,
    pub input_channels: u64,
    pub context_dimension: u64,
    pub adm_input_channels: u64,
    pub temporal_attention: bool,
    pub temporal_residual_blocks: bool,
}

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "pose-conditioned stable-view denoiser",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "stable-view latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vision_encoder",
        role: "OpenCLIP visual conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::ExactShape {
        key: INPUT_WEIGHT,
        shape: NATIVE_INPUT_SHAPE,
        score: 400,
    },
    ModelDetectionRule::ExactShape {
        key: LABEL_WEIGHT,
        shape: LABEL_SHAPE,
        score: 600,
    },
];

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
    "native.input_blocks.1.1.time_stack.0.attn1.to_q.weight",
    "native.input_blocks.1.1.time_stack.0.attn2.to_q.weight",
    "native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "native.label_emb.0.0.weight",
    "native.out.2.weight",
    "native.output_blocks.0.0.in_layers.0.weight",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "pose_conditioning_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.input_blocks.1.0.in_layers.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "spatial_transformer_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "temporal_transformer_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "temporal_attention_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "view_output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.output_blocks.0.0.in_layers.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "view_latent_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "sv3d-pose-conditioned-unet-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
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
        activation_bytes_per_element: 8,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[
    ModelSourceConfigurationRule::ExactTensorShape {
        key: INPUT_WEIGHT,
        shape: NATIVE_INPUT_SHAPE,
    },
    ModelSourceConfigurationRule::ExactTensorShape {
        key: LABEL_WEIGHT,
        shape: LABEL_SHAPE,
    },
];

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.embedders.1.encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.embedders.1.encoder.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.embedders.0.open_clip.model.visual."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.embedders.0.open_clip.model.visual.","to":"vision_encoder."}},"component":"vision_encoder"}}
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
        LABEL_WEIGHT,
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
        component: "vision_encoder",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 18,
    source_architecture: "model_base.SV3D_p",
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

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Sv3dPConfiguration, ModelFamilyError> {
    let configuration = probe.normalized_configuration()?;
    if configuration.kind() != ModelConfigurationKind::Native {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "SV3D_p is not present in the pinned source Diffusers detector table".to_owned(),
        ));
    }
    let model_channels = unsigned_fact(&configuration, "model_channels")?;
    let input_channels = unsigned_fact(&configuration, "in_channels")?;
    let context_dimension = unsigned_fact(&configuration, "context_dim")?;
    let adm_input_channels = unsigned_fact(&configuration, "adm_in_channels")?;
    let temporal_attention = boolean_fact(&configuration, "use_temporal_attention")?;
    let temporal_residual_blocks = boolean_fact(&configuration, "use_temporal_resblock")?;
    if model_channels != 320
        || input_channels != 8
        || context_dimension != 1_024
        || adm_input_channels != MODEL_FAMILY_ADM_IN_CHANNELS
        || !boolean_fact(&configuration, "use_linear_in_transformer")?
        || !temporal_attention
        || !temporal_residual_blocks
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "SV3D_p normalized configuration is incompatible: {:?}",
            configuration.facts()
        )));
    }
    Ok(Sv3dPConfiguration {
        model_channels,
        input_channels,
        context_dimension,
        adm_input_channels,
        temporal_attention,
        temporal_residual_blocks,
    })
}

fn unsigned_fact(
    configuration: &crate::ModelNormalizedConfiguration,
    name: &str,
) -> Result<u64, ModelFamilyError> {
    match configuration.fact(name) {
        Some(ModelConfigurationValue::Unsigned(value)) => Ok(*value),
        value => Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "SV3D_p requires unsigned {name}, found {value:?}"
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
            "SV3D_p requires boolean {name}, found {value:?}"
        ))),
    }
}
