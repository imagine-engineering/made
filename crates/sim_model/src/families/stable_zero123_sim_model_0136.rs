use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyComponent, ModelFamilyComponentStateSchema, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep, ModelLayoutSignature,
    ModelProbe, ModelSourceConfigurationRule, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "Stable_Zero123";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0136";
pub const MODEL_FAMILY_FIXTURE: &str = "stable-zero123-comfy-model-0136";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 1;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "ae0f2b12c2f555835e62f432e33d0c59a100a3f4a2da507d5403d06dad8fbebe";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.Stable_Zero123";

const DENOISER_PREFIX: &str = "model.diffusion_model.";
const INPUT_WEIGHT: &str = "model.diffusion_model.input_blocks.0.0.weight";
const CONTEXT_WEIGHT: &str =
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight";
const PROJECTION_WEIGHT: &str = "cc_projection.weight";
const PROJECTION_BIAS: &str = "cc_projection.bias";
const INPUT_SHAPE: &[u64] = &[320, 8, 3, 3];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StableZero123Configuration {
    pub model_channels: u64,
    pub input_channels: u64,
    pub context_dimension: u64,
    pub linear_transformer_projection: bool,
    pub adm_input_channels: Option<u64>,
    pub temporal_attention: bool,
    pub attention_heads: u64,
    pub projection_input_dimension: u64,
    pub projection_output_dimension: u64,
}

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Stable Zero123 eight-channel view-conditioned denoiser",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "cc_projection",
        role: "CLIP vision cross-attention projection",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vision_encoder",
        role: "OpenCLIP visual conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "SD15 latent codec",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::ExactShape {
        key: INPUT_WEIGHT,
        shape: INPUT_SHAPE,
        score: 300,
    },
    ModelDetectionRule::KeyPresent {
        key: PROJECTION_WEIGHT,
        score: 350,
    },
    ModelDetectionRule::KeyPresent {
        key: PROJECTION_BIAS,
        score: 350,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: DENOISER_PREFIX,
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.input_blocks.1.0.in_layers.2.weight",
    "native.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
    "native.output_blocks.0.0.in_layers.2.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.input_blocks.0.0.weight",
    "native.middle_block.1.transformer_blocks.0.attn2.to_k.weight",
    "native.out.2.weight",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "view_cross_attention_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.input_blocks.1.0.in_layers.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "view_conditioning_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "zero123_middle_attention",
        operation: ModelForwardOperation::Linear {
            weight: "native.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "zero123_attention_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "zero123_output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.output_blocks.0.0.in_layers.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "zero123_latent_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "stable-zero123-unet-v1",
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
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] =
    &[ModelSourceConfigurationRule::ExactTensorShape {
        key: INPUT_WEIGHT,
        shape: INPUT_SHAPE,
    }];

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Exact":"cc_projection.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"weight"},"component":"cc_projection"}},
            {"Move":{"selector":{"predicate":{"Exact":"cc_projection.bias"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"bias"},"component":"cc_projection"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model.model.visual."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.model.visual.","to":"model.visual."}},"component":"vision_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"model."}},"component":"vae"}}
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
    required_keys: &[INPUT_WEIGHT, PROJECTION_WEIGHT, PROJECTION_BIAS],
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
        component: "cc_projection",
        required_keys: &["weight", "bias"],
        optional_keys: &[],
        allow_unexpected: false,
    },
    ModelFamilyComponentStateSchema {
        component: "vision_encoder",
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
    source_ordinal: 1,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[PROJECTION_WEIGHT, PROJECTION_BIAS],
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
) -> Result<StableZero123Configuration, ModelFamilyError> {
    reject_diffusers(probe)?;
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(invalid_configuration("unsupported state layout"));
    }
    let input = shape(probe, INPUT_WEIGHT)?;
    if input != INPUT_SHAPE {
        return Err(invalid_configuration("input convolution must be [320,8,3,3]"));
    }
    let context = shape(probe, CONTEXT_WEIGHT)?;
    if context.len() != 2 || context[1] != 768 || context[0] == 0 {
        return Err(invalid_configuration(
            "cross-attention key projection must consume 768 features",
        ));
    }
    let projection = shape(probe, PROJECTION_WEIGHT)?;
    let bias = shape(probe, PROJECTION_BIAS)?;
    let [projection_output_dimension, projection_input_dimension] = projection else {
        return Err(invalid_configuration("cc_projection.weight must be rank two"));
    };
    if *projection_output_dimension != 768
        || *projection_input_dimension == 0
        || bias != [*projection_output_dimension]
    {
        return Err(invalid_configuration(
            "cc_projection must project a non-empty vision embedding to 768 features",
        ));
    }
    Ok(StableZero123Configuration {
        model_channels: 320,
        input_channels: 8,
        context_dimension: 768,
        linear_transformer_projection: false,
        adm_input_channels: None,
        temporal_attention: false,
        attention_heads: 8,
        projection_input_dimension: *projection_input_dimension,
        projection_output_dimension: *projection_output_dimension,
    })
}

fn reject_diffusers(probe: &ModelProbe) -> Result<(), ModelFamilyError> {
    if probe
        .format_identities()
        .iter()
        .any(|identity| identity.eq_ignore_ascii_case("diffusers"))
        || probe
            .metadata()
            .get("model_layout")
            .is_some_and(|layout| layout.eq_ignore_ascii_case("diffusers"))
    {
        return Err(invalid_configuration(
            "the pinned Diffusers detector table has no Stable_Zero123 row",
        ));
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
        "Stable_Zero123 configuration is invalid: {}",
        message.into()
    ))
}
