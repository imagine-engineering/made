use crate::{
    COGVIDEOX_DETECTION_MARKER_KEYS, COGVIDEOX_LAYOUT_SIGNATURES,
    COGVIDEOX_PATCH_PROJECTION_KEYS, CogVideoXConfiguration, CogVideoXLatentVariant,
    CogVideoXLayout, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule, cogvideox_configuration_for_probe,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "CogVideoX_T2V";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0070";
pub const MODEL_FAMILY_FIXTURE: &str = "cogvideox-t2v-comfy-model-0070";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 91;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "90206a62b3e1b6e25105cdf371428ae6585c732187d1f4083192614d63c65213";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;

pub type CogVideoXT2VLayout = CogVideoXLayout;
pub type CogVideoXT2VLatentVariant = CogVideoXLatentVariant;
pub type CogVideoXT2VConfiguration = CogVideoXConfiguration;

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.cogvideo.CogVideoXT5Tokenizer",
        clip_model: "sim.text_encoders.sd3_clip.T5XXLModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "text-to-video diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "CogVideoX latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "T5 conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: COGVIDEOX_DETECTION_MARKER_KEYS,
        score: 700,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: COGVIDEOX_PATCH_PROJECTION_KEYS,
        dimension: 1,
        values: &[16, 128],
        score: 300,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.patch_embed.proj.weight",
    "native.blocks.0.norm1.linear.weight",
    "native.time_embedding_linear_1.weight",
    "native.time_embedding_linear_1.bias",
    "native.time_embedding_linear_2.weight",
    "native.time_embedding_linear_2.bias",
    "native.proj_out.weight",
    "native.proj_out.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.patch_embed.proj.bias",
    "native.patch_embed.text_proj.weight",
    "native.patch_embed.pos_embedding",
    "native.ofs_embedding_linear_1.weight",
    "native.ofs_embedding_linear_1.bias",
    "native.ofs_embedding_linear_2.weight",
    "native.ofs_embedding_linear_2.bias",
    "native.norm_final.weight",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "time_embedding_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding_linear_1.weight",
            bias: Some("native.time_embedding_linear_1.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "time_embedding_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "time_embedding_output",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding_linear_2.weight",
            bias: Some("native.time_embedding_linear_2.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "transformer_output_normalization",
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
            weight: "native.proj_out.weight",
            bias: Some("native.proj_out.bias"),
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
    architecture_version: "cogvideox-t2v-transformer-3d-v1",
    latent_feature_id: "COMFY-MODEL-0026",
    latent_identifier: "CogVideoX",
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
            {"Move":{"selector":{"predicate":{"Prefix":"patch_embed."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"patch_embed.","to":"native.patch_embed."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"blocks.","to":"native.blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_embedding_"},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_embedding_","to":"native.time_embedding_"}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"ofs_embedding_"},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"ofs_embedding_","to":"native.ofs_embedding_"}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"norm_"},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"norm_","to":"native.norm_"}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"proj_out."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"proj_out.","to":"native.proj_out."}},"component":"model"}},
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
    source_ordinal: 91,
    source_architecture: "model_base.CogVideoX(image_to_video=False)",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: COGVIDEOX_LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    let (latent_feature_id, latent_identifier) = match configuration.latent_variant {
        CogVideoXT2VLatentVariant::CogVideoX => ("COMFY-MODEL-0026", "CogVideoX"),
        CogVideoXT2VLatentVariant::CogVideoX1_5 => ("COMFY-MODEL-0027", "CogVideoX1_5"),
    };
    Ok(ModelFamilyProfile {
        latent_feature_id,
        latent_identifier,
        clip_target: &CLIP_TARGET,
        supported_dtypes: SUPPORTED_DTYPES,
        supported_devices: SUPPORTED_DEVICES,
        memory_estimator: MODEL_FAMILY.memory_estimator,
        forward_program: FORWARD_PROGRAM,
    })
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<CogVideoXT2VConfiguration, ModelFamilyError> {
    cogvideox_configuration_for_probe(probe, 16, MODEL_FAMILY_IDENTIFIER)
}
