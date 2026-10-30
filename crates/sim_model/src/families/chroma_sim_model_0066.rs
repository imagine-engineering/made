use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelProbe, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
    flux_chroma_family::{
        CHROMA_LAYOUT_SIGNATURES, FluxChromaVariant,
        configuration_for_probe as flux_chroma_configuration_for_probe,
    },
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "Chroma";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0066";
pub const MODEL_FAMILY_FIXTURE: &str = "chroma-comfy-model-0066";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 71;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "e0257f75cb9152b18570cb7ae3f0dd171032541df9ef46a82d085a09621f6510";
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 3.2;

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.pixart_t5.PixArtTokenizer",
        clip_model: "sim.text_encoders.pixart_t5.pixart_te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "flow-matching image transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "generated guidance conditioning",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Flux latent decoder",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "PixArt T5 conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight",
            "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
            "double_blocks.0.img_attn.norm.key_norm.weight",
            "double_blocks.0.img_attn.norm.key_norm.scale",
        ],
        score: 300,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.distilled_guidance_layer.0.norms.0.weight",
            "model.diffusion_model.distilled_guidance_layer.0.norms.0.scale",
            "model.diffusion_model.distilled_guidance_layer.norms.0.weight",
            "model.diffusion_model.distilled_guidance_layer.norms.0.scale",
            "distilled_guidance_layer.0.norms.0.weight",
            "distilled_guidance_layer.0.norms.0.scale",
            "distilled_guidance_layer.norms.0.weight",
            "distilled_guidance_layer.norms.0.scale",
        ],
        score: 400,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.final_layer.linear.weight",
            "final_layer.linear.weight",
        ],
        score: 300,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.double_blocks.0.img_attn.proj.weight",
    "native.single_blocks.0.linear2.weight",
    "native.final_layer.linear.weight",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.double_blocks.0.img_attn.norm.key_norm.weight",
    "native.distilled_guidance_layer.norms.0.weight",
    "native.distilled_guidance_layer.in_proj.weight",
    "native.img_in.weight",
    "native.txt_in.weight",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "double_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.double_blocks.0.img_attn.proj.weight",
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
            weight: "native.single_blocks.0.linear2.weight",
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
            weight: "native.final_layer.linear.weight",
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
    architecture_version: "chroma-flux-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0029",
    latent_identifier: "Flux",
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

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"guidance_default"}}}
        ],
        "unmatched":"Reject"
    }"#,
};

const UNPREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"final_layer."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"final_layer."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"distilled_guidance_layer."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"distilled_guidance_layer.","to":"native.distilled_guidance_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"distilled_guidance_layer."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"distilled_guidance_layer.","to":"native.distilled_guidance_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"img_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"img_in.","to":"native.img_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"txt_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"txt_in.","to":"native.txt_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"guidance_default"}}}
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
        layout: ModelStateLayout::StandaloneNative,
        plan: &UNPREFIXED_STATE_PLAN,
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
        component: "runtime_conditioning",
        required_keys: &["guidance_default"],
        optional_keys: &[],
        allow_unexpected: false,
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
    source_ordinal: 71,
    source_architecture: "model_base.Chroma",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: CHROMA_LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    flux_chroma_configuration_for_probe(probe, FluxChromaVariant::Chroma, MODEL_FAMILY_IDENTIFIER)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
