use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    generated_wan21_causalar_t2v_sim_model_0139::{
        WAN21_CLIP_TARGET, WAN21_SUPPORTED_DEVICES, WAN21_SUPPORTED_DTYPES,
    },
    generated_wan21_t2v_sim_model_0146::{
        FFN_WEIGHT, HEAD_MODULATION, HEAD_WEIGHT, MASK_PATCH_WEIGHT, PATCH_WEIGHT,
        POSE_PATCH_WEIGHT, WAN21_OPTIONAL_KEYS, WAN21_REQUIRED_KEYS, WAN21_WEIGHT_RULES,
        Wan21BatchConfiguration, Wan21BatchVariant, batch_configuration_for_probe,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "WAN21_SCAIL2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0145";
pub const MODEL_FAMILY_FIXTURE: &str = "wan21-scail2-comfy-model-0145";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 63;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "18e2cb8a4950b25738427893437714a0ceb6084ab69df6a8a7e4aafd9667cceb";
pub const MODEL_FAMILY_SHIFT: f64 = 8.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.WAN21_SCAIL2";

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Wan 2.1 SCAIL2 pose/reference/mask-conditioned video transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "SCAIL2 multi-reference mask stream and source sampling shift",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Wan UMT5-XXL text conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Wan 2.1 video latent codec",
        required: false,
    },
];
const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: HEAD_MODULATION,
        score: 300,
    },
    ModelDetectionRule::KeyPresent {
        key: PATCH_WEIGHT,
        score: 200,
    },
    ModelDetectionRule::KeyPresent {
        key: POSE_PATCH_WEIGHT,
        score: 200,
    },
    ModelDetectionRule::KeyPresent {
        key: MASK_PATCH_WEIGHT,
        score: 500,
    },
];
const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "scail2_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "scail2_mask_conditioning",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "scail2_transformer_block",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.ffn.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "scail2_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "scail2_head_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_projection.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "scail2_video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "wan21-scail2-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
    clip_target: &WAN21_CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WAN21_WEIGHT_RULES,
    required_keys: WAN21_REQUIRED_KEYS,
    optional_keys: WAN21_OPTIONAL_KEYS,
    supported_dtypes: WAN21_SUPPORTED_DTYPES,
    supported_devices: WAN21_SUPPORTED_DEVICES,
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
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":8.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"reference_conditioning"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"pose_conditioning"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"mask_conditioning"}}}
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
        HEAD_MODULATION,
        HEAD_WEIGHT,
        PATCH_WEIGHT,
        FFN_WEIGHT,
        POSE_PATCH_WEIGHT,
        MASK_PATCH_WEIGHT,
    ],
    required_prefixes: &[],
}];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: WAN21_REQUIRED_KEYS,
        optional_keys: WAN21_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &[
            "sampling_shift",
            "reference_conditioning",
            "pose_conditioning",
            "mask_conditioning",
        ],
        optional_keys: &[],
        allow_unexpected: false,
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
    source_ordinal: 63,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[POSE_PATCH_WEIGHT, MASK_PATCH_WEIGHT],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&WAN21_CLIP_TARGET),
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
) -> Result<Wan21BatchConfiguration, ModelFamilyError> {
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "WAN21_SCAIL2 requires the source-native prefixed layout".to_owned(),
        ));
    }
    batch_configuration_for_probe(probe, Wan21BatchVariant::Scail2)
}
