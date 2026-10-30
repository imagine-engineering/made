use crate::{
    MemoryEstimatorDescriptor, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep, ModelLayoutSignature,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "ACEStep";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0061";
pub const MODEL_FAMILY_FIXTURE: &str = "acestep-comfy-model-0061";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 73;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 3.0;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 0.5;

const COMPONENTS: [ModelFamilyComponent; 4] = [
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "audio diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "conditioning",
        role: "generated runtime conditioning",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "audio latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "lyrics and prompt encoder",
        required: false,
    },
];

const DETECTION_RULES: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &[
        "model.diffusion_model.genre_embedder.weight",
        "genre_embedder.weight",
    ],
    score: 1_000,
}];

const WEIGHT_RULES: [ModelWeightRule; 3] = [
    ModelWeightRule {
        source_prefix: "model.diffusion_model.genre_embedder.",
        target_prefix: "genre_embedder.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "model.diffusion_model.final_layer.linear.",
        target_prefix: "final_layer.linear.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "model.diffusion_model.transformer_blocks.",
        target_prefix: "transformer_blocks.",
        required: false,
    },
];

const REQUIRED_KEYS: [&str; 4] = [
    "genre_embedder.weight",
    "genre_embedder.bias",
    "final_layer.linear.weight",
    "final_layer.linear.bias",
];

const OPTIONAL_KEYS: [&str; 1] = ["transformer_blocks.0.attn.to_q.weight"];
const SUPPORTED_DTYPES: [DType; 2] = [DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: [DeviceKind; 1] = [DeviceKind::Cpu];

const CLIP_CANDIDATES: [ModelClipTargetCandidateDefinition; 1] =
    [ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.ace.AceT5Tokenizer",
        clip_model: "sim.text_encoders.ace.AceT5Model",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

static CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &CLIP_CANDIDATES,
    dynamic_selection: false,
};

const FORWARD_PROGRAM: [ModelForwardStep; 5] = [
    ModelForwardStep {
        checkpoint: "genre_projection",
        operation: ModelForwardOperation::Linear {
            weight: "genre_embedder.weight",
            bias: Some("genre_embedder.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "genre_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "transformer_residual",
        operation: ModelForwardOperation::AddScalar(0.0),
    },
    ModelForwardStep {
        checkpoint: "final_projection",
        operation: ModelForwardOperation::Linear {
            weight: "final_layer.linear.weight",
            bias: Some("final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "audio_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "ace-step-transformer-2d-v1",
    latent_feature_id: "COMFY-MODEL-0023",
    latent_identifier: "ACEAudio",
    clip_target: &CLIP_TARGET,
    components: &COMPONENTS,
    detection_rules: &DETECTION_RULES,
    weight_rules: &WEIGHT_RULES,
    required_keys: &REQUIRED_KEYS,
    optional_keys: &OPTIONAL_KEYS,
    supported_dtypes: &SUPPORTED_DTYPES,
    supported_devices: &SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: &FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: [ModelSourceConfigurationRule; 0] = [];

static NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: 1,
    encoded_plan: r#"{
            "operations":[
                {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":""}},"component":"denoiser"}},
                {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":""}},"component":"vae"}},
                {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":""}},"component":"text_encoder"}},
                {"Generate":{"shape":[{"Literal":1},{"Literal":2}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"conditioning","key":"speaker_embeds"}}}
            ],
            "unmatched":"Reject"
        }"#,
};

static DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: 1,
        encoded_plan: r#"{
            "operations":[
                {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":""}},"component":"vae"}},
                {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":""}},"component":"text_encoder"}},
                {"Generate":{"shape":[{"Literal":1},{"Literal":2}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"conditioning","key":"speaker_embeds"}}}
            ],
            "unmatched":{"Route":{"component":"denoiser","rewrite":"Identity"}}
        }"#,
    };

const STATE_PLAN_CASES: [ModelFamilyStatePlanCase; 2] = [
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &DIFFUSERS_STATE_PLAN,
    },
];

const LAYOUT_SIGNATURES: [ModelLayoutSignature; 2] = [
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.genre_embedder.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["genre_embedder.weight"],
        required_prefixes: &[],
    },
];

const COMPONENT_STATE_SCHEMAS: [ModelFamilyComponentStateSchema; 4] = [
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: &REQUIRED_KEYS,
        optional_keys: &OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "conditioning",
        required_keys: &["speaker_embeds"],
        optional_keys: &[],
        allow_unexpected: false,
    },
    ModelFamilyComponentStateSchema {
        component: "vae",
        required_keys: &["decoder.weight"],
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: &["embedding.weight"],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 73,
    source_architecture: "ACEStep",
    source_configuration: &SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: None,
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: &LAYOUT_SIGNATURES,
        cases: &STATE_PLAN_CASES,
    },
    component_state_schemas: &COMPONENT_STATE_SCHEMAS,
};
