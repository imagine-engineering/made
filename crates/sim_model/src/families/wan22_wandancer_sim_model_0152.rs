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
        FFN_WEIGHT, GLOBAL_PATCH_WEIGHT, HEAD_MODULATION, HEAD_WEIGHT, PATCH_WEIGHT,
        WAN21_REQUIRED_KEYS, WAN21_WEIGHT_RULES,
    },
    generated_wan22_animate_sim_model_0148::{
        WAN22_OPTIONAL_KEYS, Wan22BatchConfiguration, Wan22BatchVariant,
        batch_configuration_for_probe,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "WAN22_WanDancer";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0152";
pub const MODEL_FAMILY_FIXTURE: &str = "wan22-wandancer-comfy-model-0152";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 64;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "e624d4dd0f61f71fed07463cda14fd979db7488eed9dcd571a62895c063206f5";
pub const MODEL_FAMILY_SHIFT: f64 = 8.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.WAN22_WanDancer";

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent { identifier: "model", role: "Wan 2.2 WanDancer music/image-conditioned video transformer", required: true },
    ModelFamilyComponent { identifier: "runtime_conditioning", role: "music, reference image, fps, audio scale, image-to-video, and sampling-shift contract", required: true },
    ModelFamilyComponent { identifier: "text_encoder", role: "Wan UMT5-XXL text conditioning", required: false },
    ModelFamilyComponent { identifier: "vae", role: "Wan 2.1 video latent codec", required: false },
];
const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent { key: HEAD_MODULATION, score: 200 },
    ModelDetectionRule::KeyPresent { key: PATCH_WEIGHT, score: 200 },
    ModelDetectionRule::KeyPresent { key: GLOBAL_PATCH_WEIGHT, score: 800 },
];
const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep { checkpoint: "wandancer_timestep_embedding", operation: ModelForwardOperation::Linear { weight: "native.time_embedding.0.weight", bias: None, input_features: 2, output_features: 2 } },
    ModelForwardStep { checkpoint: "wandancer_music_image_conditioning", operation: ModelForwardOperation::Silu },
    ModelForwardStep { checkpoint: "wandancer_transformer_block", operation: ModelForwardOperation::Linear { weight: "native.blocks.0.ffn.2.weight", bias: None, input_features: 2, output_features: 2 } },
    ModelForwardStep { checkpoint: "wandancer_block_normalization", operation: ModelForwardOperation::LayerNorm { normalized_shape: &[2], weight: None, bias: None, epsilon: 1.0e-6 } },
    ModelForwardStep { checkpoint: "wandancer_head_projection", operation: ModelForwardOperation::Linear { weight: "native.time_projection.1.weight", bias: None, input_features: 2, output_features: 2 } },
    ModelForwardStep { checkpoint: "wandancer_video_output", operation: ModelForwardOperation::Tanh },
];
pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID, identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "wan22-wandancer-v1", latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21", clip_target: &WAN21_CLIP_TARGET, components: COMPONENTS,
    detection_rules: DETECTION_RULES, weight_rules: WAN21_WEIGHT_RULES,
    required_keys: WAN21_REQUIRED_KEYS, optional_keys: WAN22_OPTIONAL_KEYS,
    supported_dtypes: WAN21_SUPPORTED_DTYPES, supported_devices: WAN21_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor { fixed_bytes: 0, bytes_per_parameter: 1, activation_bytes_per_element: 1 },
    forward_program: FORWARD_PROGRAM,
};
const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Contains":"music_encoder"},{"Suffix":"self_attn.in_proj_weight"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"self_attn.in_proj_weight","to":"self_attn.q_proj.weight"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"self_attn.in_proj_weight","to":"self_attn.k_proj.weight"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"self_attn.in_proj_weight","to":"self_attn.v_proj.weight"}}]}}]}},
            {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Contains":"music_encoder"},{"Suffix":"self_attn.in_proj_bias"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"self_attn.in_proj_bias","to":"self_attn.q_proj.bias"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"self_attn.in_proj_bias","to":"self_attn.k_proj.bias"}}]}},{"component":"model","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"self_attn.in_proj_bias","to":"self_attn.v_proj.bias"}}]}}]}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Suffix":"self_attn.in_proj_weight"}},{"Not":{"Suffix":"self_attn.in_proj_bias"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":8.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"image_to_video"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"audio_embed"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"clip_reference"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":30.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"fps"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"audio_inject_scale"}}}
        ], "unmatched":"Reject"
    }"#,
};
const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[ModelFamilyStatePlanCase { layout: ModelStateLayout::PrefixedNative, plan: &NATIVE_STATE_PLAN }];
const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[ModelLayoutSignature { layout: ModelStateLayout::PrefixedNative, required_keys: &[HEAD_MODULATION, HEAD_WEIGHT, PATCH_WEIGHT, FFN_WEIGHT, GLOBAL_PATCH_WEIGHT], required_prefixes: &[] }];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema { component: "model", required_keys: WAN21_REQUIRED_KEYS, optional_keys: WAN22_OPTIONAL_KEYS, allow_unexpected: true },
    ModelFamilyComponentStateSchema { component: "runtime_conditioning", required_keys: &["sampling_shift", "image_to_video", "audio_embed", "clip_reference", "fps", "audio_inject_scale"], optional_keys: &[], allow_unexpected: false },
    ModelFamilyComponentStateSchema { component: "text_encoder", required_keys: &[], optional_keys: &[], allow_unexpected: true },
    ModelFamilyComponentStateSchema { component: "vae", required_keys: &[], optional_keys: &[], allow_unexpected: true },
];
pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 64,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[GLOBAL_PATCH_WEIGHT],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&WAN21_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout { signatures: LAYOUT_SIGNATURES, cases: STATE_PLAN_CASES },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};
fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> { configuration_for_probe(probe)?; Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY)) }
pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Wan22BatchConfiguration, ModelFamilyError> {
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative { return Err(ModelFamilyError::InvalidSelectorOutput("WAN22_WanDancer requires the source-native prefixed layout".to_owned())); }
    batch_configuration_for_probe(probe, Wan22BatchVariant::WanDancer)
}
