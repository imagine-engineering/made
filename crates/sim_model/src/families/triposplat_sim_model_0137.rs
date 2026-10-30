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

pub const MODEL_FAMILY_IDENTIFIER: &str = "TripoSplat";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0137";
pub const MODEL_FAMILY_FIXTURE: &str = "triposplat-comfy-model-0137";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 68;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "bb4a0f41d007b1a05f31a9e3d1c19d4f3aa2ca125f18eee29fab2d41ba029304";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 0.6;
pub const MODEL_FAMILY_SHIFT: f64 = 3.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.TripoSplat";

const MODEL_PREFIX: &str = "model.diffusion_model.";
const CAMERA_OUTPUT: &str = "model.diffusion_model.cam_out_layer.weight";
const REPO_FINAL_MAP: &str = "model.diffusion_model.repo_layers.0.final_map.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TripoSplatConfiguration {
    pub query_token_length: u64,
    pub latent_channels: u64,
    pub model_channels: u64,
    pub conditioning_channels: u64,
    pub secondary_conditioning_channels: u64,
    pub camera_channels: u64,
    pub output_channels: u64,
    pub block_count: u64,
    pub refiner_block_count: u64,
    pub attention_heads: u64,
    pub attention_head_channels: u64,
    pub shared_modulation: bool,
    pub qk_rms_norm: bool,
}

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "TripoSplat latent-sequence multimodal flow denoiser",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "flow sampling shift",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "TripoSplat octree Gaussian decoder",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: CAMERA_OUTPUT,
        score: 500,
    },
    ModelDetectionRule::KeyPresent {
        key: REPO_FINAL_MAP,
        score: 500,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: MODEL_PREFIX,
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.t_embedder.mlp.0.weight",
    "native.blocks.0.mlp.mlp.2.weight",
    "native.out_layer.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.cam_out_layer.weight",
    "native.repo_layers.0.final_map.weight",
    "native.input_layer.weight",
    "native.cond_embedder.weight",
    "native.cond_embedder2.weight",
    "native.blocks.0.attn.qkv.weight",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "triposplat_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.mlp.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "triposplat_conditioning_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "triposplat_unified_transformer",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.mlp.mlp.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "triposplat_multimodal_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "triposplat_latent_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.out_layer.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "triposplat_latent_and_camera_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "triposplat-latent-sequence-flow-v1",
    latent_feature_id: "COMFY-MODEL-0052",
    latent_identifier: "TripoSplat",
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
        bytes_per_parameter: 1,
        activation_bytes_per_element: 1,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"model."}},"component":"vae"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":3.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}}
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
    required_keys: &[CAMERA_OUTPUT, REPO_FINAL_MAP],
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
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 68,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[CAMERA_OUTPUT, REPO_FINAL_MAP],
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
) -> Result<TripoSplatConfiguration, ModelFamilyError> {
    reject_diffusers(probe)?;
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(invalid_configuration("unsupported state layout"));
    }
    let camera = shape(probe, CAMERA_OUTPUT)?;
    let repo = shape(probe, REPO_FINAL_MAP)?;
    if camera != [5, 1_024] {
        return Err(invalid_configuration(
            "cam_out_layer.weight must be [5,1024]",
        ));
    }
    if repo != [48, 128] {
        return Err(invalid_configuration(
            "repo_layers.0.final_map.weight must be [48,128]",
        ));
    }
    Ok(TripoSplatConfiguration {
        query_token_length: 8_192,
        latent_channels: 16,
        model_channels: 1_024,
        conditioning_channels: 1_280,
        secondary_conditioning_channels: 128,
        camera_channels: 5,
        output_channels: 16,
        block_count: 24,
        refiner_block_count: 2,
        attention_heads: 16,
        attention_head_channels: 64,
        shared_modulation: true,
        qk_rms_norm: true,
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
            "the pinned Diffusers detector table has no TripoSplat row",
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
        "TripoSplat configuration is invalid: {}",
        message.into()
    ))
}
