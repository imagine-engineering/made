use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
    generated_stable_cascade_b_sim_model_0134::{invalid, reject_diffusers, shape},
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "Stable_Cascade_C";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0135";
pub const MODEL_FAMILY_FIXTURE: &str = "stable-cascade-c-comfy-model-0135";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 15;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "2b8619420b562703d2314c1abbc78afa0a29a1f22317aa2126b163d8e6c228a9";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const MODEL_FAMILY_SHIFT: f64 = 2.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.StableCascade_C";

const DENOISER_PREFIX: &str = "model.diffusion_model.";
const CLASSIFIER_WEIGHT: &str = "model.diffusion_model.clf.1.weight";
const TEXT_MAPPER_WEIGHT: &str = "model.diffusion_model.clip_txt_mapper.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableCascadeCVariant {
    Full,
    Lite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StableCascadeCConfiguration {
    pub variant: StableCascadeCVariant,
    pub conditioning_dimension: u64,
    pub hidden_dimensions: [u64; 2],
    pub attention_heads: [u64; 2],
    pub down_blocks: [u64; 2],
    pub up_blocks: [u64; 2],
}

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sdxl_clip.StableCascadeTokenizer",
        clip_model: "sdxl_clip.StableCascadeClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};
const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "Stable Cascade stage-C semantic prior",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Stable Cascade text conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vision_encoder",
        role: "CLIP-L visual conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Stable Cascade prior latent codec",
        required: false,
    },
];
const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: CLASSIFIER_WEIGHT,
        score: 400,
    },
    ModelDetectionRule::KeyPresent {
        key: TEXT_MAPPER_WEIGHT,
        score: 600,
    },
];
const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: DENOISER_PREFIX,
    target_prefix: "native.",
    required: true,
}];
const REQUIRED_KEYS: &[&str] = &[
    "native.embedding.1.weight",
    "native.down_blocks.0.0.channelwise.0.weight",
    "native.clf.1.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.clip_txt_mapper.weight",
    "native.clip_txt_pooled_mapper.weight",
    "native.clip_img_mapper.weight",
    "native.down_blocks.0.0.attention.attn.to_q.weight",
    "native.down_blocks.0.0.attention.attn.to_k.weight",
    "native.down_blocks.0.0.attention.attn.to_v.weight",
    "native.down_blocks.0.0.attention.attn.to_q.bias",
    "native.down_blocks.0.0.attention.attn.to_k.bias",
    "native.down_blocks.0.0.attention.attn.to_v.bias",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];
const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "stage_c_text_image_fusion",
        operation: ModelForwardOperation::Linear {
            weight: "native.embedding.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "stage_c_semantic_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "stage_c_prior_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.down_blocks.0.0.channelwise.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "stage_c_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "stage_c_classifier_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.clf.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "stable_cascade_c_prior_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "stable-cascade-stage-c-v1",
    latent_feature_id: "COMFY-MODEL-0044",
    latent_identifier: "SC_Prior",
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

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
      "operations":[
        {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Suffix":"in_proj_weight"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"denoiser","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"in_proj_weight","to":"to_q.weight"}}]}},{"component":"denoiser","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"in_proj_weight","to":"to_k.weight"}}]}},{"component":"denoiser","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"in_proj_weight","to":"to_v.weight"}}]}}]}},
        {"SplitEach":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Suffix":"in_proj_bias"}]},"minimum_matches":0,"maximum_matches":16384},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"denoiser","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"in_proj_bias","to":"to_q.bias"}}]}},{"component":"denoiser","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"in_proj_bias","to":"to_k.bias"}}]}},{"component":"denoiser","rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":"in_proj_bias","to":"to_v.bias"}}]}}]}},
        {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Any":[{"Suffix":"in_proj_weight"},{"Suffix":"in_proj_bias"}]}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}},
        {"TransformEach":{"selector":{"predicate":{"Exact":"text_encoder.clip_g.text_projection"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_g.transformer.text_projection.weight"},"component":"text_encoder","transform":{"Transpose":{"first_dimension":0,"second_dimension":1}}}},
        {"Move":{"selector":{"predicate":{"All":[{"Prefix":"text_encoder."},{"Not":{"Exact":"text_encoder.clip_g.text_projection"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoder.","to":""}},"component":"text_encoder"}},
        {"Move":{"selector":{"predicate":{"Prefix":"clip_l_vision."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"clip_l_vision.","to":""}},"component":"vision_encoder"}},
        {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":""}},"component":"vae"}}
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
        CLASSIFIER_WEIGHT,
        TEXT_MAPPER_WEIGHT,
        "model.diffusion_model.embedding.1.weight",
        "model.diffusion_model.down_blocks.0.0.channelwise.0.weight",
    ],
    required_prefixes: &[],
}];
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
    source_ordinal: 15,
    source_architecture: "model_base.StableCascade_C",
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
) -> Result<StableCascadeCConfiguration, ModelFamilyError> {
    reject_diffusers(probe, MODEL_FAMILY_IDENTIFIER)?;
    probe.select_layout(LAYOUT_SIGNATURES)?;
    let mapper = shape(probe, TEXT_MAPPER_WEIGHT, 2, MODEL_FAMILY_IDENTIFIER)?;
    match mapper[0] {
        2_048 => Ok(StableCascadeCConfiguration {
            variant: StableCascadeCVariant::Full,
            conditioning_dimension: 2_048,
            hidden_dimensions: [2_048, 2_048],
            attention_heads: [32, 32],
            down_blocks: [8, 24],
            up_blocks: [24, 8],
        }),
        1_536 => Ok(StableCascadeCConfiguration {
            variant: StableCascadeCVariant::Lite,
            conditioning_dimension: 1_536,
            hidden_dimensions: [1_536, 1_536],
            attention_heads: [24, 24],
            down_blocks: [4, 12],
            up_blocks: [12, 4],
        }),
        channels => Err(invalid(
            MODEL_FAMILY_IDENTIFIER,
            format!("unsupported stage-C text mapper width {channels}"),
        )),
    }
}
