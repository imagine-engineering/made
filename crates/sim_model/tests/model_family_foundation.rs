#[path = "../build.rs"]
#[allow(dead_code)]
mod build_script;

use sha2::{Digest, Sha256};
use sim_model::model_family::{ModelPerTensorTransform, ModelSplitOutputRule};
use sim_model::{
    GENERATED_MODEL_FAMILIES, GENERATED_MODEL_FAMILY_FIXTURES, GENERATED_MODULES,
    MODEL_CLIP_TARGET_SCHEMA_VERSION, MODEL_FAMILY_SCHEMA_VERSION,
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MappedModelWeights, MemoryEstimatorDescriptor,
    ModelClipConfigurationFact, ModelClipModelInvocation, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetCandidateDescriptor, ModelClipTargetCase,
    ModelClipTargetDefinition, ModelClipTargetDescriptor, ModelClipTargetSelector,
    ModelDetectionRule, ModelDimensionExpression, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyIdentity,
    ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyRegistry, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep, ModelKeyPredicate,
    ModelKeyRewrite, ModelKeySelector, ModelLayoutSignature, ModelNativeTargetIdentifier,
    ModelProbe, ModelRoundCondition, ModelSourceConfigurationRule, ModelStateLayout,
    ModelStateTarget, ModelStateTensorReference, ModelStateTransaction,
    ModelStateTransformOperation, ModelStateTransformPlan, ModelStateTransformPlanDefinition,
    ModelTransformBranchOutputRule, ModelUnmatchedKeyDisposition, ModelWeightRule,
    NativeFamilyBuildOptions, PatchApplication, PatchGraph, PatchGraphError, PatchKind,
    PatchOperation, PatchTarget, build_model_family, build_model_family_for_probe,
    describe_model_family, map_model_weights,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, Scalar, StreamId, Tensor,
    TensorBackend, TensorDescriptor, TensorError,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
    generated_tensor_creation_01::TensorCreationPartOneError,
};
use sim_types::{CancellationToken, DeviceKind};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const OTHER_DIGEST: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
const MAX_TEST_DIMENSION_EXPRESSION_DEPTH: usize = 32;
const COMPONENTS: [ModelFamilyComponent; 1] = [ModelFamilyComponent {
    identifier: "denoiser",
    role: "diffusion",
    required: true,
}];
const DETECTORS: [ModelDetectionRule; 2] = [
    ModelDetectionRule::ExactShape {
        key: "source.weight",
        shape: &[2],
        score: 100,
    },
    ModelDetectionRule::Metadata {
        key: "family",
        value: "foundation",
        score: 20,
    },
];
const WEIGHT_RULES: [ModelWeightRule; 1] = [ModelWeightRule {
    source_prefix: "source.",
    target_prefix: "model.",
    required: true,
}];
const REQUIRED: [&str; 1] = ["model.weight"];
const OPTIONAL: [&str; 1] = ["model.bias"];
const DTYPES: [DType; 2] = [DType::F32, DType::F16];
const DEVICES: [DeviceKind; 2] = [DeviceKind::Cpu, DeviceKind::Metal];
const FOUNDATION_CLIP_CANDIDATES: [ModelClipTargetCandidateDefinition; 1] =
    [ModelClipTargetCandidateDefinition {
        tokenizer: "sd1_clip.SD1Tokenizer",
        clip_model: "sd1_clip.SD1ClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
static FOUNDATION_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &FOUNDATION_CLIP_CANDIDATES,
    dynamic_selection: false,
};
const ALTERNATE_CLIP_CANDIDATES: [ModelClipTargetCandidateDefinition; 1] =
    [ModelClipTargetCandidateDefinition {
        tokenizer: "sdxl_clip.SDXLTokenizer",
        clip_model: "sdxl_clip.SDXLClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
static ALTERNATE_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &ALTERNATE_CLIP_CANDIDATES,
    dynamic_selection: false,
};
const CLIP_TARGET_CASES: [ModelClipTargetCase; 1] = [ModelClipTargetCase {
    metadata_value: "alternate",
    target: &ALTERNATE_CLIP_TARGET,
}];
const PROGRAM: [ModelForwardStep; 3] = [
    ModelForwardStep {
        checkpoint: "weight",
        operation: ModelForwardOperation::MultiplyWeight("model.weight"),
    },
    ModelForwardStep {
        checkpoint: "bias",
        operation: ModelForwardOperation::AddScalar(1.0),
    },
    ModelForwardStep {
        checkpoint: "activation",
        operation: ModelForwardOperation::Silu,
    },
];
static FOUNDATION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9006",
    identifier: "FoundationFamily",
    architecture_version: "foundation-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &FOUNDATION_CLIP_TARGET,
    components: &COMPONENTS,
    detection_rules: &DETECTORS,
    weight_rules: &WEIGHT_RULES,
    required_keys: &REQUIRED,
    optional_keys: &OPTIONAL,
    supported_dtypes: &DTYPES,
    supported_devices: &DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 8,
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
    },
    forward_program: &PROGRAM,
};
const TIE_DETECTORS: [ModelDetectionRule; 2] = DETECTORS;
static TIE: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9007",
    identifier: "FoundationTie",
    architecture_version: "foundation-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &FOUNDATION_CLIP_TARGET,
    components: &COMPONENTS,
    detection_rules: &TIE_DETECTORS,
    weight_rules: &WEIGHT_RULES,
    required_keys: &REQUIRED,
    optional_keys: &OPTIONAL,
    supported_dtypes: &DTYPES,
    supported_devices: &DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 8,
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
    },
    forward_program: &PROGRAM,
};
static ONE_FAMILY: [ModelFamilyDefinition; 1] = [FOUNDATION];
static TIED_FAMILIES: [ModelFamilyDefinition; 2] = [FOUNDATION, TIE];
const ANY_KEY_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &["source.weight", "alternate.weight"],
    score: 90,
}];
static ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9010",
    identifier: "FoundationAnyKeyFamily",
    detection_rules: &ANY_KEY_DETECTORS,
    ..FOUNDATION
};
static ANY_KEY_FAMILIES: [ModelFamilyDefinition; 1] = [ANY_KEY_FAMILY];
const EMPTY_ANY_KEY_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &[],
    score: 90,
}];
static EMPTY_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9011",
    identifier: "FoundationEmptyAnyKeyFamily",
    detection_rules: &EMPTY_ANY_KEY_DETECTORS,
    ..FOUNDATION
};
const DUPLICATE_ANY_KEY_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &["source.weight", "source.weight"],
    score: 90,
}];
static DUPLICATE_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9012",
    identifier: "FoundationDuplicateAnyKeyFamily",
    detection_rules: &DUPLICATE_ANY_KEY_DETECTORS,
    ..FOUNDATION
};
const MALFORMED_ANY_KEY_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &["source.weight", "alternate\0weight"],
    score: 90,
}];
static MALFORMED_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9013",
    identifier: "FoundationMalformedAnyKeyFamily",
    detection_rules: &MALFORMED_ANY_KEY_DETECTORS,
    ..FOUNDATION
};
const OVER_LIMIT_ANY_KEY_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &[
        "alternative.00",
        "alternative.01",
        "alternative.02",
        "alternative.03",
        "alternative.04",
        "alternative.05",
        "alternative.06",
        "alternative.07",
        "alternative.08",
        "alternative.09",
        "alternative.10",
        "alternative.11",
        "alternative.12",
        "alternative.13",
        "alternative.14",
        "alternative.15",
        "alternative.16",
    ],
    score: 90,
}];
static OVER_LIMIT_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9014",
    identifier: "FoundationOverLimitAnyKeyFamily",
    detection_rules: &OVER_LIMIT_ANY_KEY_DETECTORS,
    ..FOUNDATION
};
const ZERO_SCORE_ANY_KEY_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::AnyKeyPresent {
    keys: &["source.weight", "alternate.weight"],
    score: 0,
}];
static ZERO_SCORE_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9015",
    identifier: "FoundationZeroScoreAnyKeyFamily",
    detection_rules: &ZERO_SCORE_ANY_KEY_DETECTORS,
    ..FOUNDATION
};
const OVERFLOW_ANY_KEY_DETECTORS: [ModelDetectionRule; 2] = [
    ModelDetectionRule::AnyKeyPresent {
        keys: &["source.weight"],
        score: u32::MAX,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &["alternate.weight"],
        score: 1,
    },
];
static OVERFLOW_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9016",
    identifier: "FoundationOverflowAnyKeyFamily",
    detection_rules: &OVERFLOW_ANY_KEY_DETECTORS,
    ..FOUNDATION
};
static OVERFLOW_ANY_KEY_FAMILIES: [ModelFamilyDefinition; 1] = [OVERFLOW_ANY_KEY_FAMILY];
static TIE_ANY_KEY_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9017",
    identifier: "FoundationTieAnyKeyFamily",
    detection_rules: &ANY_KEY_DETECTORS,
    ..FOUNDATION
};
static TIED_ANY_KEY_FAMILIES: [ModelFamilyDefinition; 2] = [ANY_KEY_FAMILY, TIE_ANY_KEY_FAMILY];
static FOUNDATION_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: 1,
        encoded_plan: r#"{
            "operations":[{"Move":{
                "selector":{"predicate":{"Exact":"source.weight"},"minimum_matches":1,"maximum_matches":1},
                "rewrite":{"Prefix":{"from":"source.","to":"model."}},
                "component":"denoiser"
            }}],
            "unmatched":"Reject"
        }"#,
    };
static FOUNDATION_BAD_OUTPUT_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: 1,
        encoded_plan: r#"{
            "operations":[{"Move":{
                "selector":{"predicate":{"Exact":"source.weight"},"minimum_matches":1,"maximum_matches":1},
                "rewrite":{"Exact":"model.other"},
                "component":"denoiser"
            }}],
            "unmatched":"Reject"
        }"#,
    };
static FOUNDATION_UNDECLARED_COMPONENT_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: 1,
        encoded_plan: r#"{
            "operations":[{"Move":{
                "selector":{"predicate":{"Exact":"source.weight"},"minimum_matches":1,"maximum_matches":1},
                "rewrite":{"Exact":"model.weight"},
                "component":"vae"
            }}],
            "unmatched":"Reject"
        }"#,
    };
static CONST_BRANCH_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: 1,
    encoded_plan: r#"{
        "operations":[{"TransformBranchesEach":{
            "selector":{"predicate":{"Exact":"pixeldit.branch"},"minimum_matches":1,"maximum_matches":1},
            "pre_transform":{"Reshape":{"shape":[{"CurrentTensorDimension":{"dimension":0}},{"Multiply":[{"CurrentTensorDimension":{"dimension":1}},{"CurrentTensorDimension":{"dimension":2}}]}]}},
            "outputs":[{"component":"denoiser","rewrite":{"Exact":"pixeldit.left"},"transform":{"Sequence":[{"Narrow":{"dimension":0,"start":0,"length":2}},{"Reshape":{"shape":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":2},{"Literal":2}]}}]}}]
        }}],
        "unmatched":"Drop"
    }"#,
};
const FOUNDATION_PLAN_CASES: [ModelFamilyStatePlanCase; 1] = [ModelFamilyStatePlanCase {
    layout: ModelStateLayout::StandaloneNative,
    plan: &FOUNDATION_STATE_PLAN,
}];
const FOUNDATION_BAD_OUTPUT_CASES: [ModelFamilyStatePlanCase; 1] = [ModelFamilyStatePlanCase {
    layout: ModelStateLayout::StandaloneNative,
    plan: &FOUNDATION_BAD_OUTPUT_PLAN,
}];
const FOUNDATION_LAYOUT_SIGNATURES: [ModelLayoutSignature; 1] = [ModelLayoutSignature {
    layout: ModelStateLayout::StandaloneNative,
    required_keys: &["source.weight"],
    required_prefixes: &[],
}];
const FOUNDATION_COMPONENT_SCHEMAS: [ModelFamilyComponentStateSchema; 1] =
    [ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: &["model.weight"],
        optional_keys: &["model.bias"],
        allow_unexpected: false,
    }];
static FOUNDATION_TRANSACTION_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        definition: &FOUNDATION,
        source_ordinal: 42,
        source_architecture: "foundation",
        source_configuration: &[],
        required_state_keys: &["source.weight"],
        profile_selector: None,
        clip_target_selector: ModelClipTargetSelector::Profile,
        state_plan_selector: ModelFamilyStatePlanSelector::Layout {
            signatures: &FOUNDATION_LAYOUT_SIGNATURES,
            cases: &FOUNDATION_PLAN_CASES,
        },
        component_state_schemas: &FOUNDATION_COMPONENT_SCHEMAS,
    }];
fn foundation_probe_state_plan(
    probe: &ModelProbe,
) -> Result<ModelStateTransformPlan, ModelFamilyError> {
    probe.select_layout(&FOUNDATION_LAYOUT_SIGNATURES)?;
    FOUNDATION_STATE_PLAN.compile()
}
static FOUNDATION_PROBE_TRANSACTION_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        state_plan_selector: ModelFamilyStatePlanSelector::Probe(foundation_probe_state_plan),
        ..FOUNDATION_TRANSACTION_REGISTRATION[0]
    }];
fn foundation_probe_undeclared_component_plan(
    _: &ModelProbe,
) -> Result<ModelStateTransformPlan, ModelFamilyError> {
    FOUNDATION_UNDECLARED_COMPONENT_PLAN.compile()
}
static FOUNDATION_BAD_PROBE_TRANSACTION_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        state_plan_selector: ModelFamilyStatePlanSelector::Probe(
            foundation_probe_undeclared_component_plan,
        ),
        ..FOUNDATION_TRANSACTION_REGISTRATION[0]
    }];
static FOUNDATION_BAD_OUTPUT_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        state_plan_selector: ModelFamilyStatePlanSelector::Layout {
            signatures: &FOUNDATION_LAYOUT_SIGNATURES,
            cases: &FOUNDATION_BAD_OUTPUT_CASES,
        },
        ..FOUNDATION_TRANSACTION_REGISTRATION[0]
    }];
static FOUNDATION_UNDECLARED_COMPONENT_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        state_plan_selector: ModelFamilyStatePlanSelector::Static(
            &FOUNDATION_UNDECLARED_COMPONENT_PLAN,
        ),
        ..FOUNDATION_TRANSACTION_REGISTRATION[0]
    }];
static FOUNDATION_CLIP_SELECTOR_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        clip_target_selector: ModelClipTargetSelector::Metadata {
            key: "clip_target",
            cases: &CLIP_TARGET_CASES,
        },
        ..FOUNDATION_TRANSACTION_REGISTRATION[0]
    }];
const SCHEMA_COMPONENTS: [ModelFamilyComponent; 2] = [
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "diffusion",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "conditioning",
        required: false,
    },
];
static SCHEMA_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9010",
    identifier: "ComponentSchemaFamily",
    architecture_version: "component-schema-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &FOUNDATION_CLIP_TARGET,
    components: &SCHEMA_COMPONENTS,
    detection_rules: &DETECTORS,
    weight_rules: &WEIGHT_RULES,
    required_keys: &REQUIRED,
    optional_keys: &OPTIONAL,
    supported_dtypes: &DTYPES,
    supported_devices: &DEVICES,
    memory_estimator: FOUNDATION.memory_estimator,
    forward_program: &PROGRAM,
};
const SCHEMA_RULES: [ModelFamilyComponentStateSchema; 2] = [
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: &["model.weight"],
        optional_keys: &[],
        allow_unexpected: false,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: &["model.clip_weight"],
        optional_keys: &[],
        allow_unexpected: false,
    },
];
static SCHEMA_ABSENT_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: 1,
    encoded_plan: r#"{
        "operations":[{"Move":{
            "selector":{"predicate":{"Exact":"source.weight"},"minimum_matches":1,"maximum_matches":1},
            "rewrite":{"Exact":"model.weight"},"component":"denoiser"
        }}],"unmatched":"Drop"
    }"#,
};
static SCHEMA_PRESENT_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: 1,
    encoded_plan: r#"{
        "operations":[
            {"Move":{"selector":{"predicate":{"Exact":"source.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"model.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"source.clip"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"model.clip_weight"},"component":"text_encoder"}}
        ],"unmatched":"Reject"
    }"#,
};
static SCHEMA_EXTRA_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: 1,
    encoded_plan: r#"{
        "operations":[
            {"Move":{"selector":{"predicate":{"Exact":"source.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"model.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"source.clip"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"model.clip_weight"},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Exact":"source.extra"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"model.extra"},"component":"text_encoder"}}
        ],"unmatched":"Reject"
    }"#,
};
static SCHEMA_ABSENT_REGISTRATION: [ModelFamilyRegistration; 1] = [ModelFamilyRegistration {
    definition: &SCHEMA_FAMILY,
    source_ordinal: 50,
    source_architecture: "component_schema",
    source_configuration: &[],
    required_state_keys: &["source.weight"],
    profile_selector: None,
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Static(&SCHEMA_ABSENT_PLAN),
    component_state_schemas: &SCHEMA_RULES,
}];
static SCHEMA_PRESENT_REGISTRATION: [ModelFamilyRegistration; 1] = [ModelFamilyRegistration {
    state_plan_selector: ModelFamilyStatePlanSelector::Static(&SCHEMA_PRESENT_PLAN),
    ..SCHEMA_ABSENT_REGISTRATION[0]
}];
static SCHEMA_EXTRA_REGISTRATION: [ModelFamilyRegistration; 1] = [ModelFamilyRegistration {
    state_plan_selector: ModelFamilyStatePlanSelector::Static(&SCHEMA_EXTRA_PLAN),
    ..SCHEMA_ABSENT_REGISTRATION[0]
}];
static SCHEMA_MISSING_REQUIRED_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        component_state_schemas: &[SCHEMA_RULES[1]],
        ..SCHEMA_ABSENT_REGISTRATION[0]
    }];
static SCHEMA_PRESENT_UNSCHEMATIZED_OPTIONAL_REGISTRATION: [ModelFamilyRegistration; 1] =
    [ModelFamilyRegistration {
        state_plan_selector: ModelFamilyStatePlanSelector::Static(&SCHEMA_PRESENT_PLAN),
        component_state_schemas: &[SCHEMA_RULES[0]],
        ..SCHEMA_ABSENT_REGISTRATION[0]
    }];

const OP_DETECTORS: [ModelDetectionRule; 1] = [ModelDetectionRule::Metadata {
    key: "family",
    value: "foundation-operators",
    score: 100,
}];
const OP_REQUIRED: [&str; 7] = [
    "model.linear_weight",
    "model.linear_bias",
    "model.conv1d_weight",
    "model.conv2d_weight",
    "model.conv3d_weight",
    "model.norm_weight",
    "model.norm_bias",
];
const CPU_ONLY: [DeviceKind; 1] = [DeviceKind::Cpu];
const F32_ONLY: [DType; 1] = [DType::F32];
const LINEAR_PROGRAM: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "linear",
    operation: ModelForwardOperation::Linear {
        weight: "model.linear_weight",
        bias: Some("model.linear_bias"),
        input_features: 2,
        output_features: 2,
    },
}];
const CONV1D_PROGRAM: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "convolution-1d",
    operation: ModelForwardOperation::Convolution1d {
        weight: "model.conv1d_weight",
        bias: None,
        input_channels: 1,
        output_channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    },
}];
const CONV2D_PROGRAM: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "convolution-2d",
    operation: ModelForwardOperation::Convolution2d {
        weight: "model.conv2d_weight",
        bias: None,
        input_channels: 1,
        output_channels: 1,
        kernel_size: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    },
}];
const CONV3D_PROGRAM: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "convolution-3d",
    operation: ModelForwardOperation::Convolution3d {
        weight: "model.conv3d_weight",
        bias: None,
        input_channels: 1,
        output_channels: 1,
        kernel_size: [1, 1, 1],
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        dilation: [1, 1, 1],
        groups: 1,
    },
}];
const NORM_PROGRAM: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "normalization",
    operation: ModelForwardOperation::LayerNorm {
        normalized_shape: &[2],
        weight: Some("model.norm_weight"),
        bias: Some("model.norm_bias"),
        epsilon: 1.0e-5,
    },
}];
const ATTENTION_PROGRAM: [ModelForwardStep; 1] = [ModelForwardStep {
    checkpoint: "attention",
    operation: ModelForwardOperation::SelfAttention { heads: 1 },
}];
const ACTIVATION_PROGRAM: [ModelForwardStep; 2] = [
    ModelForwardStep {
        checkpoint: "silu",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "tanh",
        operation: ModelForwardOperation::Tanh,
    },
];
static OPERATOR_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9008",
    identifier: "FoundationOperatorFamily",
    architecture_version: "foundation-operators-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &FOUNDATION_CLIP_TARGET,
    components: &COMPONENTS,
    detection_rules: &OP_DETECTORS,
    weight_rules: &WEIGHT_RULES,
    required_keys: &OP_REQUIRED,
    optional_keys: &[],
    supported_dtypes: &F32_ONLY,
    supported_devices: &CPU_ONLY,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 40,
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
    },
    forward_program: &LINEAR_PROGRAM,
};
const OP_SOURCE_CONFIGURATION: [ModelSourceConfigurationRule; 2] = [
    ModelSourceConfigurationRule::Metadata {
        key: "source_architecture",
        value: "foundation_ops",
    },
    ModelSourceConfigurationRule::ExactTensorShape {
        key: "source.linear_weight",
        shape: &[2, 2],
    },
];
const OP_STATE_KEYS: [&str; 1] = ["source.linear_weight"];
const OP_COMPONENT_SCHEMAS: [ModelFamilyComponentStateSchema; 1] =
    [ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: &OP_REQUIRED,
        optional_keys: &[],
        allow_unexpected: false,
    }];
static OPERATOR_REGISTRATION: [ModelFamilyRegistration; 1] = [ModelFamilyRegistration {
    definition: &OPERATOR_FAMILY,
    source_ordinal: 17,
    source_architecture: "foundation_ops",
    source_configuration: &OP_SOURCE_CONFIGURATION,
    required_state_keys: &OP_STATE_KEYS,
    profile_selector: Some(select_operator_profile),
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: sim_model::ModelFamilyStatePlanSelector::LegacyDefinitionRules,
    component_state_schemas: &OP_COMPONENT_SCHEMAS,
}];
static ORDERED_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    ModelFamilyRegistration {
        definition: &FOUNDATION,
        source_ordinal: 30,
        source_architecture: "foundation",
        source_configuration: &[],
        required_state_keys: &[],
        profile_selector: None,
        clip_target_selector: ModelClipTargetSelector::Profile,
        state_plan_selector: sim_model::ModelFamilyStatePlanSelector::LegacyDefinitionRules,
        component_state_schemas: &FOUNDATION_COMPONENT_SCHEMAS,
    },
    OPERATOR_REGISTRATION[0],
];
static DUPLICATE_ORDINAL_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    ModelFamilyRegistration {
        definition: &FOUNDATION,
        source_ordinal: 17,
        source_architecture: "foundation",
        source_configuration: &[],
        required_state_keys: &[],
        profile_selector: None,
        clip_target_selector: ModelClipTargetSelector::Profile,
        state_plan_selector: sim_model::ModelFamilyStatePlanSelector::LegacyDefinitionRules,
        component_state_schemas: &FOUNDATION_COMPONENT_SCHEMAS,
    },
    OPERATOR_REGISTRATION[0],
];

fn select_operator_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let program = match probe.metadata.get("program").map(String::as_str) {
        Some("linear") => &LINEAR_PROGRAM[..],
        Some("conv1d") => &CONV1D_PROGRAM[..],
        Some("conv2d") => &CONV2D_PROGRAM[..],
        Some("conv3d") => &CONV3D_PROGRAM[..],
        Some("norm") => &NORM_PROGRAM[..],
        Some("attention") => &ATTENTION_PROGRAM[..],
        Some("activation") => &ACTIVATION_PROGRAM[..],
        Some("invalid") => {
            return Ok(ModelFamilyProfile {
                latent_feature_id: "COMFY-MODEL-0045",
                latent_identifier: "SD15",
                clip_target: &FOUNDATION_CLIP_TARGET,
                supported_dtypes: &[DType::F64],
                supported_devices: &CPU_ONLY,
                memory_estimator: OPERATOR_FAMILY.memory_estimator,
                forward_program: &LINEAR_PROGRAM,
            });
        }
        _ => {
            return Err(ModelFamilyError::InvalidSelectorOutput(
                "unknown operator profile".to_owned(),
            ));
        }
    };
    let attention = program == &ATTENTION_PROGRAM;
    Ok(ModelFamilyProfile {
        latent_feature_id: if attention {
            "COMFY-MODEL-0046"
        } else {
            "COMFY-MODEL-0045"
        },
        latent_identifier: if attention { "SDXL" } else { "SD15" },
        clip_target: &FOUNDATION_CLIP_TARGET,
        supported_dtypes: &F32_ONLY,
        supported_devices: &CPU_ONLY,
        memory_estimator: MemoryEstimatorDescriptor {
            fixed_bytes: if attention { 80 } else { 40 },
            bytes_per_parameter: 4,
            activation_bytes_per_element: 4,
        },
        forward_program: program,
    })
}

#[test]
fn identity_registry_detection_and_descriptor_are_checked_and_stable()
-> Result<(), Box<dyn std::error::Error>> {
    let identity = ModelFamilyIdentity::new(
        FOUNDATION.feature_id,
        FOUNDATION.identifier,
        FOUNDATION.architecture_version,
    )?;
    let encoded = serde_json::to_vec(&identity)?;
    assert!(
        String::from_utf8(encoded.clone())?
            .contains(&format!("\"schema_version\":{MODEL_FAMILY_SCHEMA_VERSION}"))
    );
    assert_eq!(
        serde_json::from_slice::<ModelFamilyIdentity>(&encoded)?,
        identity
    );
    assert!(serde_json::from_str::<ModelFamilyIdentity>(
        r#"{"schema_version":2,"feature_id":"COMFY-MODEL-9006","identifier":"FoundationFamily","architecture_version":"foundation-v1"}"#,
    ).is_err());

    let registry = ModelFamilyRegistry::checked(&ONE_FAMILY)?;
    let probe = foundation_probe();
    let detection = registry.detect(&probe)?;
    assert_eq!(detection.identity, identity);
    assert_eq!(detection.score, 120);
    assert_eq!(detection.evidence.len(), 2);
    assert!(matches!(
        ModelFamilyRegistry::checked(&TIED_FAMILIES)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 120, .. })
    ));
    assert!(matches!(
        registry.detect(&ModelProbe::default()),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let descriptor = describe_model_family(&FOUNDATION)?;
    assert_eq!(descriptor.identifier, FOUNDATION.identifier);
    assert_eq!(descriptor.latent_format, "SD15");
    assert_eq!(descriptor.supported_dtypes, ["float32", "float16"]);
    Ok(())
}

#[test]
fn any_key_detection_is_disjunctive_and_checked() -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked(&ANY_KEY_FAMILIES)?;
    for key in ["source.weight", "alternate.weight"] {
        let probe = ModelProbe {
            tensor_shapes: BTreeMap::from([(key.to_owned(), vec![2])]),
            metadata: BTreeMap::new(),
        };
        let detection = registry.detect(&probe)?;
        assert_eq!(detection.identity.feature_id(), ANY_KEY_FAMILY.feature_id);
        assert_eq!(detection.score, 90);
        assert_eq!(detection.evidence.len(), 1);
    }
    let both_alternatives_probe = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("source.weight".to_owned(), vec![2]),
            ("alternate.weight".to_owned(), vec![2]),
        ]),
        metadata: BTreeMap::new(),
    };
    let detection = registry.detect(&both_alternatives_probe)?;
    assert_eq!(detection.score, 90);
    assert_eq!(detection.evidence.len(), 1);
    assert!(matches!(
        registry.detect(&ModelProbe::default()),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    assert!(matches!(
        describe_model_family(&EMPTY_ANY_KEY_FAMILY),
        Err(ModelFamilyError::InvalidDefinition(message))
            if message == "any-key detector has 0 alternatives; expected 1..=16"
    ));
    assert!(matches!(
        describe_model_family(&DUPLICATE_ANY_KEY_FAMILY),
        Err(ModelFamilyError::DuplicateDefinitionValue(value))
            if value == "source.weight"
    ));
    assert!(matches!(
        describe_model_family(&MALFORMED_ANY_KEY_FAMILY),
        Err(ModelFamilyError::InvalidDefinition(message))
            if message == "invalid model key"
    ));
    assert!(matches!(
        describe_model_family(&OVER_LIMIT_ANY_KEY_FAMILY),
        Err(ModelFamilyError::InvalidDefinition(message))
            if message == "any-key detector has 17 alternatives; expected 1..=16"
    ));
    assert!(matches!(
        describe_model_family(&ZERO_SCORE_ANY_KEY_FAMILY),
        Err(ModelFamilyError::InvalidDefinition(message))
            if message == "zero detection score"
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked(&OVERFLOW_ANY_KEY_FAMILIES)?.detect(&both_alternatives_probe),
        Err(ModelFamilyError::DetectionScoreOverflow)
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked(&TIED_ANY_KEY_FAMILIES)?.detect(&both_alternatives_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 90, .. })
    ));
    Ok(())
}

fn project_catalog_clip_model_expression(
    expression: &str,
) -> Result<(ModelNativeTargetIdentifier, ModelClipModelInvocation), Box<dyn std::error::Error>> {
    let Some(open_parenthesis) = expression.find('(') else {
        return Ok((
            ModelNativeTargetIdentifier::checked(expression)?,
            ModelClipModelInvocation::Reference,
        ));
    };
    if !expression.ends_with(')')
        || expression[open_parenthesis + 1..expression.len() - 1]
            .chars()
            .any(|character| matches!(character, '(' | ')'))
    {
        return Err(format!("malformed catalog CLIP model expression {expression:?}").into());
    }
    let target = ModelNativeTargetIdentifier::checked(&expression[..open_parenthesis])?;
    let arguments = &expression[open_parenthesis + 1..expression.len() - 1];
    let mut configuration = Vec::new();
    if !arguments.is_empty() {
        for argument in arguments.split(',') {
            let argument = argument.trim();
            if let Some(source) = argument.strip_prefix("**") {
                configuration.push(ModelClipConfigurationFact::expand(source)?);
                continue;
            }
            let (parameter, source) = argument
                .split_once('=')
                .ok_or_else(|| format!("unsupported catalog CLIP argument {argument:?}"))?;
            if source.contains('=') {
                return Err(format!("malformed catalog CLIP binding {argument:?}").into());
            }
            configuration.push(ModelClipConfigurationFact::bind(
                parameter.trim(),
                source.trim(),
            )?);
        }
    }
    Ok((target, ModelClipModelInvocation::Factory { configuration }))
}

#[test]
fn tokenizer_and_clip_target_descriptors_cover_catalog_and_data_only_selection()
-> Result<(), Box<dyn std::error::Error>> {
    let catalog_bytes =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"))?;
    let catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    let rows = catalog["models"]
        .as_array()
        .ok_or("catalog models missing")?;
    assert_eq!(rows.len(), 94);
    let mut covered_rows = 0_usize;
    let mut projected_factory_calls = 0_usize;
    for row in rows {
        covered_rows += 1;
        let clip_target = &row["clip_target"];
        let candidates = clip_target["calls"]
            .as_array()
            .ok_or("catalog CLIP calls missing")?
            .iter()
            .map(|call| {
                let expression = call["clip_model"].as_str().ok_or("CLIP model missing")?;
                let (target, invocation) = project_catalog_clip_model_expression(expression)?;
                if matches!(&invocation, ModelClipModelInvocation::Factory { .. }) {
                    projected_factory_calls += 1;
                }
                ModelClipTargetCandidateDescriptor::checked_with_invocation(
                    call["tokenizer"].as_str().ok_or("tokenizer missing")?,
                    target.as_str(),
                    invocation,
                )
                .map_err(Into::into)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        let descriptor = ModelClipTargetDescriptor::checked(
            candidates,
            clip_target["has_dynamic_control_flow"]
                .as_bool()
                .ok_or("dynamic CLIP flag missing")?,
        )?;
        let encoded = serde_json::to_vec(&descriptor)?;
        assert_eq!(
            serde_json::from_slice::<ModelClipTargetDescriptor>(&encoded)?,
            descriptor
        );
    }
    assert_eq!(covered_rows, 94);
    assert!(projected_factory_calls > 0);
    assert!(
        serde_json::from_str::<ModelClipTargetDescriptor>(
            r#"{"schema_version":3,"candidates":[],"dynamic_selection":false}"#,
        )
        .is_err()
    );
    assert!(ModelClipTargetDescriptor::checked(Vec::new(), true).is_err());
    let serialized_raw_expression = serde_json::json!({
        "schema_version": MODEL_CLIP_TARGET_SCHEMA_VERSION,
        "candidates": [{
            "tokenizer": {
                "schema_version": MODEL_CLIP_TARGET_SCHEMA_VERSION,
                "identifier": "valid.Tokenizer",
            },
            "clip_model": {
                "schema_version": MODEL_CLIP_TARGET_SCHEMA_VERSION,
                "target": "valid.Factory(**detect)",
                "invocation": "Reference",
            },
        }],
        "dynamic_selection": false,
    });
    assert!(
        serde_json::from_value::<ModelClipTargetDescriptor>(serialized_raw_expression).is_err()
    );
    let serialized_unknown_fact = serde_json::json!({
        "schema_version": MODEL_CLIP_TARGET_SCHEMA_VERSION,
        "candidates": [{
            "tokenizer": {
                "schema_version": MODEL_CLIP_TARGET_SCHEMA_VERSION,
                "identifier": "valid.Tokenizer",
            },
            "clip_model": {
                "schema_version": MODEL_CLIP_TARGET_SCHEMA_VERSION,
                "target": "valid.Factory",
                "invocation": {"Factory": {"configuration": [
                    {"Expand": {"source": "detect", "unknown": true}}
                ]}},
            },
        }],
        "dynamic_selection": false,
    });
    assert!(serde_json::from_value::<ModelClipTargetDescriptor>(serialized_unknown_fact).is_err());
    for malformed in [
        "sim.text_encoders.ace15.te(**detect)",
        "sim..text_encoder",
        "sim.text-encoder.Model",
        ".sim.text_encoder.Model",
    ] {
        assert!(ModelClipTargetCandidateDescriptor::checked("valid.Tokenizer", malformed).is_err());
    }
    for malformed_expression in [
        "valid.Factory(**detect",
        "valid.Factory(positional)",
        "valid.Factory(name='python-value')",
        "valid.Factory(nested=call())",
    ] {
        assert!(project_catalog_clip_model_expression(malformed_expression).is_err());
    }
    let (factory_target, factory_invocation) = project_catalog_clip_model_expression(
        "sim.text_encoders.sd3_clip.sd3_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, **t5_detect)",
    )?;
    assert_eq!(
        factory_target.as_str(),
        "sim.text_encoders.sd3_clip.sd3_clip"
    );
    let ModelClipModelInvocation::Factory { configuration } = factory_invocation else {
        return Err("catalog factory expression projected as a reference".into());
    };
    assert_eq!(configuration.len(), 4);
    assert!(
        ModelClipTargetCandidateDescriptor::checked_with_invocation(
            "valid.Tokenizer",
            "valid.Factory",
            ModelClipModelInvocation::Factory {
                configuration: vec![ModelClipConfigurationFact::expand("detect")?; 65],
            },
        )
        .is_err()
    );
    assert!(
        ModelClipTargetCandidateDescriptor::checked_with_invocation(
            "valid.Tokenizer",
            "valid.Factory",
            ModelClipModelInvocation::Factory {
                configuration: vec![
                    ModelClipConfigurationFact::bind("clip_l", "clip_l")?,
                    ModelClipConfigurationFact::bind("clip_l", "other")?,
                ],
            },
        )
        .is_err()
    );

    let registry =
        ModelFamilyRegistry::checked_registrations(&FOUNDATION_CLIP_SELECTOR_REGISTRATION)?;
    let mut probe = foundation_probe();
    probe
        .metadata
        .insert("clip_target".to_owned(), "alternate".to_owned());
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(
        resolved.clip_target().candidates()[0]
            .tokenizer()
            .identifier(),
        "sdxl_clip.SDXLTokenizer"
    );
    assert_eq!(
        resolved.clip_target().candidates()[0]
            .clip_model()
            .target()
            .as_str(),
        "sdxl_clip.SDXLClipModel"
    );
    Ok(())
}

#[test]
fn registrations_preserve_source_order_and_reject_duplicate_ordinals()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&ORDERED_REGISTRATIONS)?;
    assert_eq!(
        registry
            .definitions_in_source_order()
            .into_iter()
            .map(|definition| definition.identifier)
            .collect::<Vec<_>>(),
        ["FoundationOperatorFamily", "FoundationFamily"]
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&DUPLICATE_ORDINAL_REGISTRATIONS),
        Err(ModelFamilyError::DuplicateSourceOrdinal(17))
    ));
    Ok(())
}

fn compile_encoded_plan(encoded_plan: String) -> Result<ModelStateTransformPlan, ModelFamilyError> {
    let encoded_plan = Box::leak(encoded_plan.into_boxed_str());
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan,
    }
    .compile()
}

#[test]
fn encoded_state_plans_revalidate_selectors_expressions_and_evaluation_context()
-> Result<(), Box<dyn std::error::Error>> {
    for (minimum_matches, maximum_matches) in [(1, 16_385), (2, 1)] {
        let encoded = serde_json::json!({
            "operations": [{
                "Drop": {
                    "selector": {
                        "predicate": {"Exact": "source.weight"},
                        "minimum_matches": minimum_matches,
                        "maximum_matches": maximum_matches,
                    }
                }
            }],
            "unmatched": "Drop",
        });
        assert!(matches!(
            compile_encoded_plan(serde_json::to_string(&encoded)?),
            Err(ModelFamilyError::InvalidKeySelectorBounds { .. })
        ));
    }

    let split_plan = |expression: serde_json::Value| {
        serde_json::json!({
            "operations": [{
                "Split": {
                    "source": {"Source": "source.weight"},
                    "dimension": 0,
                    "sizes": [expression],
                    "outputs": [{"component": "denoiser", "key": "model.weight"}],
                }
            }],
            "unmatched": "Drop",
        })
    };
    assert!(matches!(
        compile_encoded_plan(serde_json::to_string(&split_plan(serde_json::json!({
            "SourceDimension": {"key": "invalid..key", "dimension": 0}
        })))?),
        Err(ModelFamilyError::InvalidStateKey(key)) if key == "invalid..key"
    ));
    assert!(matches!(
        compile_encoded_plan(serde_json::to_string(&split_plan(serde_json::json!({
            "CurrentTensorDimension": {"dimension": 0}
        })))?),
        Err(ModelFamilyError::CurrentTensorDimensionUnavailable)
    ));

    let mut expression = serde_json::json!({"Literal": 1});
    for _ in 0..MAX_TEST_DIMENSION_EXPRESSION_DEPTH {
        expression = serde_json::json!({
            "Add": [expression, {"Literal": 1}]
        });
    }
    assert!(matches!(
        compile_encoded_plan(serde_json::to_string(&split_plan(expression))?),
        Err(ModelFamilyError::DimensionExpressionTooDeep)
    ));
    Ok(())
}

#[test]
fn probe_resolution_checks_source_contract_and_revalidates_selector_output()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&OPERATOR_REGISTRATION)?;
    let attention = registry.resolve(&operator_probe("attention"))?;
    assert_eq!(attention.source_ordinal(), 17);
    assert_eq!(attention.source_architecture(), "foundation_ops");
    assert_eq!(attention.profile().latent_identifier, "SDXL");
    assert_eq!(attention.profile().memory_estimator.fixed_bytes, 80);
    assert!(matches!(
        registry.resolve(&operator_probe("invalid")),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));

    let mut wrong_configuration = operator_probe("linear");
    wrong_configuration.metadata.insert(
        "source_architecture".to_owned(),
        "not-foundation-ops".to_owned(),
    );
    assert!(matches!(
        registry.resolve(&wrong_configuration),
        Err(ModelFamilyError::SourceConfigurationMismatch { .. })
    ));
    let mut missing_state = operator_probe("linear");
    missing_state.tensor_shapes.clear();
    assert!(matches!(
        registry.resolve(&missing_state),
        Err(ModelFamilyError::MissingRequiredStateKey(key)) if key == "source.linear_weight"
    ));
    Ok(())
}

#[test]
fn data_plan_resolution_maps_component_schema_and_binds_probe_family_and_profile()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let registry =
        ModelFamilyRegistry::checked_registrations(&FOUNDATION_TRANSACTION_REGISTRATION)?;
    let probe = foundation_probe();
    let resolved = registry.resolve(&probe)?;
    assert!(resolved.state_plan().is_some());
    let probe_registry =
        ModelFamilyRegistry::checked_registrations(&FOUNDATION_PROBE_TRANSACTION_REGISTRATION)?;
    let probe_resolved = probe_registry.resolve(&probe)?;
    assert_eq!(
        probe_resolved
            .state_plan()
            .ok_or("probe selector omitted its state plan")?
            .identity(),
        resolved
            .state_plan()
            .ok_or("layout selector omitted its state plan")?
            .identity()
    );
    let bad_probe_registry =
        ModelFamilyRegistry::checked_registrations(&FOUNDATION_BAD_PROBE_TRANSACTION_REGISTRATION)?;
    assert!(matches!(
        bad_probe_registry.resolve(&probe),
        Err(ModelFamilyError::UndeclaredComponent(component)) if component == "vae"
    ));
    let source = BTreeMap::from([(
        "source.weight".to_owned(),
        tensor(&backend, &[2], &[2.0, 3.0], DType::F32, &context)?,
    )]);
    let transaction = ModelStateTransaction::new(&backend, &context);
    let mapped_components = resolved.map_state_dictionary(&transaction, DIGEST, &source)?;
    assert_eq!(
        mapped_components
            .binding()
            .ok_or("missing component mapping binding")?
            .family(),
        &resolved.detection().identity
    );
    let weights = resolved.map_primary_weights(&transaction, DIGEST, &source)?;
    let binding = weights.binding().ok_or("missing mapped-weight binding")?;
    assert_eq!(binding.family(), &resolved.detection().identity);
    assert_eq!(
        weights.tensors().keys().collect::<Vec<_>>(),
        ["model.weight"]
    );
    build_model_family_for_probe(
        &registry,
        &probe,
        weights.clone(),
        NativeFamilyBuildOptions {
            dtype: DType::F32,
            device: DeviceKind::Cpu,
            activation_elements: 2,
            memory_budget_bytes: 1024,
            allow_unexpected_weights: false,
        },
    )?;

    let mut spoofed_plan_probe = foundation_probe();
    spoofed_plan_probe
        .metadata
        .insert("state_plan".to_owned(), "unsupported".to_owned());
    assert!(
        registry
            .resolve(&spoofed_plan_probe)?
            .state_plan()
            .is_some()
    );

    let mut changed_probe = probe.clone();
    changed_probe
        .metadata
        .insert("unrelated".to_owned(), "changed".to_owned());
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &changed_probe,
            weights.clone(),
            NativeFamilyBuildOptions {
                dtype: DType::F32,
                device: DeviceKind::Cpu,
                activation_elements: 2,
                memory_budget_bytes: 1024,
                allow_unexpected_weights: false,
            },
        ),
        Err(ModelFamilyError::WeightBindingMismatch(_))
    ));

    let operator_registry = ModelFamilyRegistry::checked_registrations(&OPERATOR_REGISTRATION)?;
    assert!(matches!(
        build_model_family_for_probe(
            &operator_registry,
            &operator_probe("linear"),
            weights,
            operator_options(2, 1024),
        ),
        Err(ModelFamilyError::WeightBindingMismatch(_))
    ));

    let mut drifted_source = source.clone();
    drifted_source.insert(
        "source.extra".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        resolved.map_primary_weights(&transaction, DIGEST, &drifted_source),
        Err(ModelFamilyError::ResolvedProbeDrift(_))
    ));

    let bad_output_registry =
        ModelFamilyRegistry::checked_registrations(&FOUNDATION_BAD_OUTPUT_REGISTRATION)?;
    let bad_output_resolved = bad_output_registry.resolve(&probe)?;
    assert!(matches!(
        bad_output_resolved.map_primary_weights(&transaction, DIGEST, &source),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "denoiser" && key == "model.weight"
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(
            &FOUNDATION_UNDECLARED_COMPONENT_REGISTRATION
        ),
        Err(ModelFamilyError::UndeclaredComponent(component)) if component == "vae"
    ));
    Ok(())
}

#[test]
fn component_schemas_require_required_coverage_and_close_optional_components_when_present()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&SCHEMA_MISSING_REQUIRED_REGISTRATION),
        Err(ModelFamilyError::MissingComponentSchema(component)) if component == "denoiser"
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(
            &SCHEMA_PRESENT_UNSCHEMATIZED_OPTIONAL_REGISTRATION
        ),
        Err(ModelFamilyError::MissingComponentSchema(component)) if component == "text_encoder"
    ));
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let transaction = ModelStateTransaction::new(&backend, &context);
    let absent_registry = ModelFamilyRegistry::checked_registrations(&SCHEMA_ABSENT_REGISTRATION)?;
    let absent_probe = foundation_probe();
    let absent_source = BTreeMap::from([(
        "source.weight".to_owned(),
        tensor(&backend, &[2], &[1.0, 2.0], DType::F32, &context)?,
    )]);
    let absent = absent_registry
        .resolve(&absent_probe)?
        .map_state_dictionary(&transaction, DIGEST, &absent_source)?;
    assert!(absent.component("denoiser").is_some());
    assert!(absent.component("text_encoder").is_none());

    let present_registry =
        ModelFamilyRegistry::checked_registrations(&SCHEMA_PRESENT_REGISTRATION)?;
    let mut present_probe = foundation_probe();
    present_probe
        .tensor_shapes
        .insert("source.clip".to_owned(), vec![2]);
    let mut present_source = absent_source;
    present_source.insert(
        "source.clip".to_owned(),
        tensor(&backend, &[2], &[3.0, 4.0], DType::F32, &context)?,
    );
    let present = present_registry
        .resolve(&present_probe)?
        .map_state_dictionary(&transaction, DIGEST, &present_source)?;
    assert_eq!(
        present
            .component("text_encoder")
            .ok_or("optional component missing")?
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["model.clip_weight"]
    );

    let extra_registry = ModelFamilyRegistry::checked_registrations(&SCHEMA_EXTRA_REGISTRATION)?;
    let mut extra_probe = present_probe;
    extra_probe
        .tensor_shapes
        .insert("source.extra".to_owned(), vec![1]);
    let mut extra_source = present_source;
    extra_source.insert(
        "source.extra".to_owned(),
        tensor(&backend, &[1], &[5.0], DType::F32, &context)?,
    );
    assert!(matches!(
        extra_registry.resolve(&extra_probe)?.map_state_dictionary(
            &transaction,
            DIGEST,
            &extra_source,
        ),
        Err(ModelFamilyError::UnexpectedComponentKey { component, key })
            if component == "text_encoder" && key == "model.extra"
    ));
    Ok(())
}

#[test]
fn shape_reduced_programs_delegate_every_foundational_step_class()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let registry = ModelFamilyRegistry::checked_registrations(&OPERATOR_REGISTRATION)?;

    let cases: [(&str, &[u64], &[f32], &[f32]); 4] = [
        ("linear", &[1, 2], &[2.0, 3.0], &[2.5, 2.5]),
        ("conv1d", &[1, 1, 2], &[2.0, 3.0], &[4.0, 6.0]),
        ("conv2d", &[1, 1, 1, 2], &[2.0, 3.0], &[4.0, 6.0]),
        ("conv3d", &[1, 1, 1, 1, 2], &[2.0, 3.0], &[4.0, 6.0]),
    ];
    for (program, shape, input_values, expected) in cases {
        let weights = operator_weights(&backend, &context, &registry, program)?;
        let model = build_model_family_for_probe(
            &registry,
            &operator_probe(program),
            weights,
            operator_options(4, 1024),
        )?;
        let input = tensor(&backend, shape, input_values, DType::F32, &context)?;
        let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(
            values(&backend, &checkpoints[0].tensor, &context)?,
            expected
        );
    }

    let normalization_weights = operator_weights(&backend, &context, &registry, "norm")?;
    let normalization = build_model_family_for_probe(
        &registry,
        &operator_probe("norm"),
        normalization_weights,
        operator_options(2, 1024),
    )?;
    let normalization_input = tensor(&backend, &[1, 2], &[1.0, 3.0], DType::F32, &context)?;
    let normalized = normalization.forward_checkpoints(&backend, &normalization_input, &context)?;
    let normalized_values = values(&backend, &normalized[0].tensor, &context)?;
    assert!((normalized_values[0] + 0.999_995).abs() < 1.0e-5);
    assert!((normalized_values[1] - 0.999_995).abs() < 1.0e-5);

    let attention_weights = operator_weights(&backend, &context, &registry, "attention")?;
    let attention = build_model_family_for_probe(
        &registry,
        &operator_probe("attention"),
        attention_weights,
        operator_options(4, 1024),
    )?;
    assert_eq!(attention.source_architecture(), Some("foundation_ops"));
    let attention_input = tensor(
        &backend,
        &[1, 2, 2],
        &[1.0, 0.0, 0.0, 1.0],
        DType::F32,
        &context,
    )?;
    let attended = attention.forward_checkpoints(&backend, &attention_input, &context)?;
    assert_eq!(attended[0].tensor.descriptor().shape(), [1, 2, 2]);
    assert!(
        values(&backend, &attended[0].tensor, &context)?
            .into_iter()
            .all(f32::is_finite)
    );

    let activation_weights = operator_weights(&backend, &context, &registry, "activation")?;
    let activation = build_model_family_for_probe(
        &registry,
        &operator_probe("activation"),
        activation_weights,
        operator_options(2, 1024),
    )?;
    let activation_input = tensor(&backend, &[2], &[-1.0, 2.0], DType::F32, &context)?;
    let activated = activation.forward_checkpoints(&backend, &activation_input, &context)?;
    assert_eq!(
        activated
            .iter()
            .map(|checkpoint| checkpoint.name.as_str())
            .collect::<Vec<_>>(),
        ["silu", "tanh"]
    );
    Ok(())
}

#[test]
fn probe_aware_build_is_typed_failure_atomic_for_oom_dtype_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let registry = ModelFamilyRegistry::checked_registrations(&OPERATOR_REGISTRATION)?;
    let attention_weights = operator_weights(&backend, &context, &registry, "attention")?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &operator_probe("attention"),
            attention_weights,
            operator_options(4, 147),
        ),
        Err(ModelFamilyError::OutOfMemory {
            required: 148,
            budget: 147,
        })
    ));
    let linear_weights = operator_weights(&backend, &context, &registry, "linear")?;
    let mut unsupported = operator_options(4, 1024);
    unsupported.dtype = DType::F64;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &operator_probe("linear"),
            linear_weights.clone(),
            unsupported,
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));

    let model = build_model_family_for_probe(
        &registry,
        &operator_probe("linear"),
        linear_weights,
        operator_options(2, 1024),
    )?;
    let input = tensor(&backend, &[1, 2], &[2.0, 3.0], DType::F32, &context)?;
    cancellation.cancel();
    assert!(matches!(
        model.forward_checkpoints(&backend, &input, &context),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

#[test]
fn build_manifest_owns_sorted_family_registry_and_fixture_closure()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(GENERATED_MODULES.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        GENERATED_MODULES
            .iter()
            .filter(|name| name.starts_with("families/"))
            .count(),
        GENERATED_MODEL_FAMILIES.len(),
    );
    assert_eq!(
        GENERATED_MODEL_FAMILIES.len(),
        GENERATED_MODEL_FAMILY_FIXTURES.len()
    );
    ModelFamilyRegistry::checked(GENERATED_MODEL_FAMILIES)?;

    let source = concat!(
        "pub const MODEL_FAMILY_IDENTIFIER: &str = \"Family\";\n",
        "pub const MODEL_FAMILY_FEATURE_ID: &str = \"COMFY-MODEL-9006\";\n",
        "pub const MODEL_FAMILY_FIXTURE: &str = \"family-comfy-model-9006\";\n",
    );
    let path = std::path::Path::new("family.rs");
    assert_eq!(
        build_script::source_constant(source, path, "MODEL_FAMILY_IDENTIFIER")?,
        "Family"
    );
    let mut families = Vec::new();
    build_script::register_model_family(
        &mut families,
        "family",
        "Family",
        "COMFY-MODEL-9006",
        "family-comfy-model-9006",
    )?;
    assert!(
        build_script::register_model_family(
            &mut families,
            "duplicate",
            "Family",
            "COMFY-MODEL-9007",
            "duplicate-comfy-model-9007",
        )
        .is_err()
    );

    let directory = tempfile::tempdir()?;
    assert!(build_script::model_family_fixture_names_in(&families, directory.path()).is_err());
    let fixture = directory.path().join("family-comfy-model-9006");
    std::fs::create_dir(&fixture)?;
    std::fs::write(fixture.join("family.json"), b"{}")?;
    assert_eq!(
        build_script::model_family_fixture_names_in(&families, directory.path())?,
        ["family-comfy-model-9006"],
    );
    let orphan = directory.path().join("orphan");
    std::fs::create_dir(&orphan)?;
    std::fs::write(orphan.join("family.json"), b"{}")?;
    assert!(build_script::model_family_fixture_names_in(&families, directory.path()).is_err());
    Ok(())
}

#[test]
fn foundation_mapping_build_memory_forward_dtype_device_oom_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(64 * 1024)?,
        &cancellation,
    );
    let weights = mapped_weights(&backend, &context, true)?;
    assert_eq!(weights.unexpected_keys(), ["extra.unmapped"]);
    let options = NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes: 32,
        allow_unexpected_weights: true,
    };
    let model = build_model_family(&FOUNDATION, weights.clone(), options)?;
    assert_eq!(model.memory_estimate().parameter_elements, 2);
    assert_eq!(model.memory_estimate().total_bytes, 24);
    let input = tensor(&backend, &[2], &[2.0, -1.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_eq!(
        checkpoints
            .iter()
            .map(|checkpoint| checkpoint.name.as_str())
            .collect::<Vec<_>>(),
        ["weight", "bias", "activation"]
    );
    assert_eq!(
        values(&backend, &checkpoints[0].tensor, &context)?,
        [4.0, 3.0]
    );
    let final_values = values(&backend, &checkpoints[2].tensor, &context)?;
    assert!((final_values[0] - 4.966_536).abs() < 1.0e-5);
    assert!((final_values[1] - 3.928_055).abs() < 1.0e-5);

    let mut strict = options;
    strict.allow_unexpected_weights = false;
    assert!(matches!(
        build_model_family(&FOUNDATION, weights.clone(), strict),
        Err(ModelFamilyError::UnexpectedKeys(_))
    ));
    let mut oom = options;
    oom.memory_budget_bytes = 23;
    assert!(matches!(
        build_model_family(&FOUNDATION, weights.clone(), oom),
        Err(ModelFamilyError::OutOfMemory {
            required: 24,
            budget: 23
        })
    ));
    let mut unsupported = options;
    unsupported.dtype = DType::F64;
    assert!(matches!(
        build_model_family(&FOUNDATION, weights.clone(), unsupported),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
    let mut unavailable = options;
    unavailable.device = DeviceKind::Metal;
    assert!(build_model_family(&FOUNDATION, weights, unavailable).is_err());

    cancellation.cancel();
    assert!(
        model
            .forward_checkpoints(&backend, &input, &context)
            .is_err()
    );
    Ok(())
}

#[test]
fn state_transaction_key_primitives_are_bounded_alias_preserving_and_multi_component()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(256 * 1024)?,
        &cancellation,
    );
    let source = BTreeMap::from([
        (
            "exact.weight".to_owned(),
            tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
        ),
        (
            "prefix.a".to_owned(),
            tensor(&backend, &[1], &[2.0], DType::F32, &context)?,
        ),
        (
            "prefix.b".to_owned(),
            tensor(&backend, &[1], &[3.0], DType::F32, &context)?,
        ),
        (
            "tail.suffix".to_owned(),
            tensor(&backend, &[1], &[4.0], DType::F32, &context)?,
        ),
        (
            "left.middle.right".to_owned(),
            tensor(&backend, &[1], &[5.0], DType::F32, &context)?,
        ),
        (
            "drop.me".to_owned(),
            tensor(&backend, &[1], &[6.0], DType::F32, &context)?,
        ),
        (
            "route.me".to_owned(),
            tensor(&backend, &[1], &[7.0], DType::F32, &context)?,
        ),
        (
            "unmatched.weight".to_owned(),
            tensor(&backend, &[1], &[8.0], DType::F32, &context)?,
        ),
    ]);
    let plan = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::Move {
                selector: ModelKeySelector::exact("exact.weight")?,
                rewrite: ModelKeyRewrite::exact("renamed.weight")?,
                component: "denoiser".to_owned(),
            },
            ModelStateTransformOperation::Copy {
                selector: ModelKeySelector::bounded(ModelKeyPredicate::prefix("prefix.")?, 2, 2)?,
                rewrite: ModelKeyRewrite::prefix("prefix.", "copy.")?,
                component: "encoder".to_owned(),
            },
            ModelStateTransformOperation::Copy {
                selector: ModelKeySelector::exact("prefix.a")?,
                rewrite: ModelKeyRewrite::exact("copy.again")?,
                component: "encoder".to_owned(),
            },
            ModelStateTransformOperation::Move {
                selector: ModelKeySelector::bounded(ModelKeyPredicate::suffix(".suffix")?, 1, 1)?,
                rewrite: ModelKeyRewrite::suffix(".suffix", ".bias")?,
                component: "denoiser".to_owned(),
            },
            ModelStateTransformOperation::Route {
                selector: ModelKeySelector::bounded(
                    ModelKeyPredicate::contains(".middle.")?,
                    1,
                    1,
                )?,
                rewrite: ModelKeyRewrite::contains(".middle.", ".center.")?,
                component: "router".to_owned(),
            },
            ModelStateTransformOperation::Drop {
                selector: ModelKeySelector::exact("drop.me")?,
            },
            ModelStateTransformOperation::Route {
                selector: ModelKeySelector::exact("route.me")?,
                rewrite: ModelKeyRewrite::Identity,
                component: "router".to_owned(),
            },
        ],
        ModelUnmatchedKeyDisposition::Route {
            component: "extras".to_owned(),
            rewrite: ModelKeyRewrite::prefix("unmatched.", "kept.")?,
        },
    )?;
    let mapped = ModelStateTransaction::new(&backend, &context).execute(&plan, DIGEST, &source)?;
    assert_eq!(mapped.base_artifact_digest(), DIGEST);
    assert_eq!(mapped.components().len(), 4);
    let denoiser = mapped.component("denoiser").ok_or("missing denoiser")?;
    assert_eq!(
        denoiser["renamed.weight"].storage_id(),
        source["exact.weight"].storage_id()
    );
    assert_eq!(
        denoiser["tail.bias"].storage_id(),
        source["tail.suffix"].storage_id()
    );
    let encoder = mapped.component("encoder").ok_or("missing encoder")?;
    assert_eq!(
        encoder["copy.a"].storage_id(),
        source["prefix.a"].storage_id()
    );
    assert_eq!(
        encoder["copy.again"].storage_id(),
        source["prefix.a"].storage_id()
    );
    assert_eq!(
        encoder["copy.b"].storage_id(),
        source["prefix.b"].storage_id()
    );
    assert_eq!(
        mapped.component("extras").ok_or("missing extras")?["kept.weight"].storage_id(),
        source["unmatched.weight"].storage_id()
    );
    assert!(
        !mapped
            .components()
            .values()
            .any(|component| component.contains_key("drop.me"))
    );

    assert!(matches!(
        ModelKeySelector::bounded(ModelKeyPredicate::prefix("x")?, 2, 1),
        Err(ModelFamilyError::InvalidKeySelectorBounds { .. })
    ));
    assert!(ModelKeyPredicate::contains("").is_err());
    assert!(ModelKeyRewrite::exact("invalid..key").is_err());

    let legacy_tensor = tensor(&backend, &[2], &[11.0, 12.0], DType::F32, &context)?;
    let legacy_storage = legacy_tensor.storage_id();
    let legacy = map_model_weights(
        &FOUNDATION,
        DIGEST,
        BTreeMap::from([("source.weight".to_owned(), legacy_tensor)]),
    )?;
    assert_eq!(
        legacy.tensors()["model.weight"].storage_id(),
        legacy_storage
    );
    Ok(())
}

#[test]
fn state_transaction_delegates_split_assembly_views_round_and_generation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = BTreeMap::from([
        (
            "split".to_owned(),
            tensor(&backend, &[4], &[1.0, 2.0, 3.0, 4.0], DType::F32, &context)?,
        ),
        (
            "assemble.a".to_owned(),
            tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?,
        ),
        (
            "assemble.b".to_owned(),
            tensor(&backend, &[1, 1], &[3.0], DType::F32, &context)?,
        ),
        (
            "transpose".to_owned(),
            tensor(
                &backend,
                &[2, 2],
                &[1.0, 2.0, 3.0, 4.0],
                DType::F32,
                &context,
            )?,
        ),
        (
            "reshape".to_owned(),
            tensor(
                &backend,
                &[2, 2],
                &[5.0, 6.0, 7.0, 8.0],
                DType::F32,
                &context,
            )?,
        ),
        (
            "round".to_owned(),
            tensor(&backend, &[2], &[1.25, 1.75], DType::F32, &context)?,
        ),
        (
            "round.skip".to_owned(),
            tensor(&backend, &[2], &[9.0, 10.0], DType::F32, &context)?,
        ),
    ]);
    let plan = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::Split {
                source: source_ref("split")?,
                dimension: 0,
                sizes: vec![
                    ModelDimensionExpression::Literal(2),
                    ModelDimensionExpression::Literal(2),
                ],
                outputs: vec![target("parts", "left")?, target("parts", "right")?],
            },
            ModelStateTransformOperation::Assemble {
                sources: vec![source_ref("assemble.a")?, source_ref("assemble.b")?],
                dimension: 1,
                output: target("assembled", "weight")?,
            },
            ModelStateTransformOperation::Transpose {
                source: source_ref("transpose")?,
                first_dimension: 0,
                second_dimension: 1,
                output: target("views", "transposed")?,
            },
            ModelStateTransformOperation::Reshape {
                source: source_ref("reshape")?,
                shape: vec![ModelDimensionExpression::Literal(4)],
                output: target("views", "reshaped")?,
            },
            ModelStateTransformOperation::ConditionalRound {
                source: source_ref("round")?,
                decimals: 0,
                condition: ModelRoundCondition::DType(DType::F32),
                output: target("numeric", "rounded")?,
            },
            ModelStateTransformOperation::ConditionalRound {
                source: source_ref("round.skip")?,
                decimals: 0,
                condition: ModelRoundCondition::Rank(2),
                output: target("numeric", "unchanged")?,
            },
            ModelStateTransformOperation::Generate {
                shape: vec![
                    ModelDimensionExpression::Add(
                        Box::new(ModelDimensionExpression::Literal(1)),
                        Box::new(ModelDimensionExpression::Literal(1)),
                    ),
                    ModelDimensionExpression::Multiply(
                        Box::new(ModelDimensionExpression::Literal(1)),
                        Box::new(ModelDimensionExpression::DivideExact(
                            Box::new(ModelDimensionExpression::source_dimension("assemble.a", 1)?),
                            Box::new(ModelDimensionExpression::Literal(1)),
                        )),
                    ),
                ],
                fill: Scalar::Float(3.0),
                dtype: DType::F32,
                output: target("generated", "constant")?,
            },
            ModelStateTransformOperation::GenerateArange {
                start: Scalar::Signed(0),
                end: Scalar::Signed(4),
                step: Scalar::Signed(1),
                dtype: DType::F32,
                shape: vec![
                    ModelDimensionExpression::Literal(1),
                    ModelDimensionExpression::Literal(4),
                ],
                output: target("generated", "range")?,
            },
            ModelStateTransformOperation::Narrow {
                source: ModelStateTensorReference::staged(target("generated", "range")?),
                dimension: 1,
                start: 1,
                length: 2,
                output: target("generated", "narrowed")?,
            },
            ModelStateTransformOperation::Permute {
                source: ModelStateTensorReference::staged(target("generated", "narrowed")?),
                dimensions: vec![1, 0],
                output: target("generated", "permuted")?,
            },
            ModelStateTransformOperation::Expand {
                source: ModelStateTensorReference::staged(target("generated", "permuted")?),
                shape: vec![
                    ModelDimensionExpression::Literal(2),
                    ModelDimensionExpression::Literal(3),
                ],
                output: target("generated", "expanded")?,
            },
        ],
        ModelUnmatchedKeyDisposition::Reject,
    )?;
    let mapped = ModelStateTransaction::new(&backend, &context).execute(&plan, DIGEST, &source)?;
    assert_eq!(
        values(
            &backend,
            &mapped.component("parts").ok_or("parts")?["left"],
            &context
        )?,
        [1.0, 2.0]
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("parts").ok_or("parts")?["right"],
            &context
        )?,
        [3.0, 4.0]
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("assembled").ok_or("assembled")?["weight"],
            &context
        )?,
        [1.0, 2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("views").ok_or("views")?["transposed"],
            &context
        )?,
        [1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(
        mapped.component("views").ok_or("views")?["transposed"].storage_id(),
        source["transpose"].storage_id()
    );
    assert_eq!(
        mapped.component("views").ok_or("views")?["reshaped"].storage_id(),
        source["reshape"].storage_id()
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("numeric").ok_or("numeric")?["rounded"],
            &context
        )?,
        [1.0, 2.0]
    );
    assert_eq!(
        mapped.component("numeric").ok_or("numeric")?["unchanged"].storage_id(),
        source["round.skip"].storage_id()
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("generated").ok_or("generated")?["constant"],
            &context
        )?,
        [3.0, 3.0, 3.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("generated").ok_or("generated")?["range"],
            &context
        )?,
        [0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &mapped.component("generated").ok_or("generated")?["expanded"],
            &context
        )?,
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
    Ok(())
}

#[test]
fn selector_driven_transforms_cover_dynamic_source_patterns_without_callbacks()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = BTreeMap::from([
        (
            "blocks.0.narrow".to_owned(),
            tensor(&backend, &[4], &[1.0, 2.0, 3.0, 4.0], DType::F32, &context)?,
        ),
        (
            "blocks.1.transpose".to_owned(),
            tensor(
                &backend,
                &[2, 2],
                &[1.0, 2.0, 3.0, 4.0],
                DType::F32,
                &context,
            )?,
        ),
        (
            "blocks.2.permute".to_owned(),
            tensor(&backend, &[1, 2], &[5.0, 6.0], DType::F32, &context)?,
        ),
        (
            "blocks.3.reshape".to_owned(),
            tensor(
                &backend,
                &[2, 2],
                &[7.0, 8.0, 9.0, 10.0],
                DType::F32,
                &context,
            )?,
        ),
        (
            "blocks.4.expand".to_owned(),
            tensor(&backend, &[2, 1], &[11.0, 12.0], DType::F32, &context)?,
        ),
        (
            "blocks.5.round".to_owned(),
            tensor(&backend, &[1], &[1.75], DType::F32, &context)?,
        ),
        (
            "blocks.6.split".to_owned(),
            tensor(
                &backend,
                &[4],
                &[13.0, 14.0, 15.0, 16.0],
                DType::F32,
                &context,
            )?,
        ),
    ]);
    let exact_suffix_selector = |suffix: &str| {
        ModelKeySelector::bounded(
            ModelKeyPredicate::all(vec![
                ModelKeyPredicate::prefix("blocks.")?,
                ModelKeyPredicate::suffix(suffix)?,
                ModelKeyPredicate::negate(ModelKeyPredicate::contains("skip")?)?,
            ])?,
            1,
            1,
        )
    };
    let output_rewrite = |suffix: &str| {
        ModelKeyRewrite::pipeline(vec![
            ModelKeyRewrite::prefix("blocks.", "model.blocks.")?,
            ModelKeyRewrite::suffix(suffix, ".weight")?,
        ])
    };
    let plan = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::TransformEach {
                selector: exact_suffix_selector(".narrow")?,
                rewrite: output_rewrite(".narrow")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::Narrow {
                    dimension: 0,
                    start: 1,
                    length: 2,
                },
            },
            ModelStateTransformOperation::TransformEach {
                selector: ModelKeySelector::bounded(
                    ModelKeyPredicate::any(vec![
                        ModelKeyPredicate::exact("blocks.1.transpose")?,
                        ModelKeyPredicate::exact("never")?,
                    ])?,
                    1,
                    1,
                )?,
                rewrite: output_rewrite(".transpose")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::Sequence(vec![
                    ModelPerTensorTransform::Transpose {
                        first_dimension: 0,
                        second_dimension: 1,
                    },
                    ModelPerTensorTransform::Contiguous,
                ]),
            },
            ModelStateTransformOperation::TransformEach {
                selector: exact_suffix_selector(".permute")?,
                rewrite: output_rewrite(".permute")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::Permute {
                    dimensions: vec![1, 0],
                },
            },
            ModelStateTransformOperation::TransformEach {
                selector: exact_suffix_selector(".reshape")?,
                rewrite: output_rewrite(".reshape")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::Reshape {
                    shape: vec![ModelDimensionExpression::Literal(4)],
                },
            },
            ModelStateTransformOperation::TransformEach {
                selector: exact_suffix_selector(".expand")?,
                rewrite: output_rewrite(".expand")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::Expand {
                    shape: vec![
                        ModelDimensionExpression::Literal(2),
                        ModelDimensionExpression::Literal(3),
                    ],
                },
            },
            ModelStateTransformOperation::TransformEach {
                selector: exact_suffix_selector(".round")?,
                rewrite: output_rewrite(".round")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::ConditionalRound {
                    decimals: 0,
                    condition: ModelRoundCondition::Always,
                },
            },
            ModelStateTransformOperation::SplitEach {
                selector: exact_suffix_selector(".split")?,
                dimension: 0,
                sizes: vec![
                    ModelDimensionExpression::Literal(2),
                    ModelDimensionExpression::Literal(2),
                ],
                outputs: vec![
                    ModelSplitOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::suffix(".split", ".left")?,
                    },
                    ModelSplitOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::suffix(".split", ".right")?,
                    },
                ],
            },
        ],
        ModelUnmatchedKeyDisposition::Reject,
    )?;
    let mapped = ModelStateTransaction::new(&backend, &context).execute(&plan, DIGEST, &source)?;
    let denoiser = mapped.component("denoiser").ok_or("denoiser")?;
    assert_eq!(
        values(&backend, &denoiser["model.blocks.0.weight"], &context)?,
        [2.0, 3.0]
    );
    assert_eq!(
        values(&backend, &denoiser["model.blocks.1.weight"], &context)?,
        [1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(
        values(&backend, &denoiser["model.blocks.4.weight"], &context)?,
        [11.0, 11.0, 11.0, 12.0, 12.0, 12.0]
    );
    assert_eq!(
        values(&backend, &denoiser["blocks.6.left"], &context)?,
        [13.0, 14.0]
    );
    assert_eq!(
        values(&backend, &denoiser["blocks.6.right"], &context)?,
        [15.0, 16.0]
    );
    Ok(())
}

#[test]
fn selected_tensor_dimensions_and_branch_transforms_cover_pinned_family_shapes()
-> Result<(), Box<dyn std::error::Error>> {
    let const_plan = CONST_BRANCH_PLAN.compile()?;
    assert!(matches!(
        const_plan.operations(),
        [ModelStateTransformOperation::TransformBranchesEach { .. }]
    ));
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = BTreeMap::from([
        (
            "stable_cascade.projection".to_owned(),
            tensor(
                &backend,
                &[9, 2],
                &(0..18).map(|value| value as f32).collect::<Vec<_>>(),
                DType::F32,
                &context,
            )?,
        ),
        (
            "sam3.qkv".to_owned(),
            tensor(
                &backend,
                &[12],
                &(0..12).map(|value| value as f32).collect::<Vec<_>>(),
                DType::F32,
                &context,
            )?,
        ),
        (
            "pixeldit.branch".to_owned(),
            tensor(
                &backend,
                &[4, 2, 2],
                &(0..16).map(|value| value as f32).collect::<Vec<_>>(),
                DType::F32,
                &context,
            )?,
        ),
    ]);
    let third_of_current = ModelDimensionExpression::DivideExact(
        Box::new(ModelDimensionExpression::current_tensor_dimension(0)),
        Box::new(ModelDimensionExpression::Literal(3)),
    );
    let pixel_pre_shape = vec![
        ModelDimensionExpression::current_tensor_dimension(0),
        ModelDimensionExpression::Multiply(
            Box::new(ModelDimensionExpression::current_tensor_dimension(1)),
            Box::new(ModelDimensionExpression::current_tensor_dimension(2)),
        ),
    ];
    let pixel_branch_shape = vec![
        ModelDimensionExpression::current_tensor_dimension(0),
        ModelDimensionExpression::Literal(2),
        ModelDimensionExpression::Literal(2),
    ];
    let branch = |start| {
        ModelPerTensorTransform::Sequence(vec![
            ModelPerTensorTransform::Narrow {
                dimension: 0,
                start,
                length: 2,
            },
            ModelPerTensorTransform::Reshape {
                shape: pixel_branch_shape.clone(),
            },
        ])
    };
    let plan = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::TransformEach {
                selector: ModelKeySelector::exact("stable_cascade.projection")?,
                rewrite: ModelKeyRewrite::exact("stable.projection")?,
                component: "denoiser".to_owned(),
                transform: ModelPerTensorTransform::Reshape {
                    shape: vec![
                        third_of_current.clone(),
                        ModelDimensionExpression::Literal(3),
                        ModelDimensionExpression::current_tensor_dimension(1),
                    ],
                },
            },
            ModelStateTransformOperation::SplitEach {
                selector: ModelKeySelector::exact("sam3.qkv")?,
                dimension: 0,
                sizes: vec![
                    third_of_current.clone(),
                    third_of_current.clone(),
                    third_of_current,
                ],
                outputs: vec![
                    ModelSplitOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::exact("sam3.q")?,
                    },
                    ModelSplitOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::exact("sam3.k")?,
                    },
                    ModelSplitOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::exact("sam3.v")?,
                    },
                ],
            },
            ModelStateTransformOperation::TransformBranchesEach {
                selector: ModelKeySelector::exact("pixeldit.branch")?,
                pre_transform: ModelPerTensorTransform::Reshape {
                    shape: pixel_pre_shape,
                },
                outputs: vec![
                    ModelTransformBranchOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::exact("pixeldit.left")?,
                        transform: branch(0),
                    },
                    ModelTransformBranchOutputRule {
                        component: "denoiser".to_owned(),
                        rewrite: ModelKeyRewrite::exact("pixeldit.right")?,
                        transform: branch(2),
                    },
                ],
            },
        ],
        ModelUnmatchedKeyDisposition::Reject,
    )?;
    let mapped = ModelStateTransaction::new(&backend, &context).execute(&plan, DIGEST, &source)?;
    let denoiser = mapped.component("denoiser").ok_or("denoiser missing")?;
    assert_eq!(
        denoiser["stable.projection"].descriptor().shape(),
        [3, 3, 2]
    );
    assert_eq!(denoiser["sam3.q"].descriptor().shape(), [4]);
    assert_eq!(
        values(&backend, &denoiser["sam3.q"], &context)?,
        [0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        values(&backend, &denoiser["sam3.v"], &context)?,
        [8.0, 9.0, 10.0, 11.0]
    );
    assert_eq!(denoiser["pixeldit.left"].descriptor().shape(), [2, 2, 2]);
    assert_eq!(denoiser["pixeldit.right"].descriptor().shape(), [2, 2, 2]);
    assert_eq!(
        values(&backend, &denoiser["pixeldit.right"], &context)?,
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );

    let invalid_source = BTreeMap::from([(
        "stable_cascade.projection".to_owned(),
        tensor(&backend, &[10, 2], &[0.0; 20], DType::F32, &context)?,
    )]);
    let invalid_plan = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::TransformEach {
            selector: ModelKeySelector::exact("stable_cascade.projection")?,
            rewrite: ModelKeyRewrite::Identity,
            component: "denoiser".to_owned(),
            transform: ModelPerTensorTransform::Reshape {
                shape: vec![
                    ModelDimensionExpression::DivideExact(
                        Box::new(ModelDimensionExpression::current_tensor_dimension(0)),
                        Box::new(ModelDimensionExpression::Literal(3)),
                    ),
                    ModelDimensionExpression::Literal(3),
                    ModelDimensionExpression::current_tensor_dimension(1),
                ],
            },
        }],
        ModelUnmatchedKeyDisposition::Reject,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &invalid_plan,
            DIGEST,
            &invalid_source,
        ),
        Err(ModelFamilyError::DimensionDivisionRemainder {
            numerator: 10,
            denominator: 3,
        })
    ));
    Ok(())
}

#[test]
fn state_transaction_prevalidates_errors_and_rolls_back_on_failure_or_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = BTreeMap::from([
        (
            "one".to_owned(),
            tensor(&backend, &[2], &[1.0, 2.0], DType::F32, &context)?,
        ),
        (
            "matrix".to_owned(),
            tensor(&backend, &[1, 2], &[3.0, 4.0], DType::F32, &context)?,
        ),
        (
            "mismatch".to_owned(),
            tensor(&backend, &[2, 1], &[5.0, 6.0], DType::F32, &context)?,
        ),
        (
            "half".to_owned(),
            tensor(&backend, &[2], &[1.25, 1.75], DType::F16, &context)?,
        ),
    ]);
    let original_storage = source
        .iter()
        .map(|(key, tensor)| (key.clone(), tensor.storage_id()))
        .collect::<BTreeMap<_, _>>();

    let missing_match = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Move {
            selector: ModelKeySelector::exact("missing")?,
            rewrite: ModelKeyRewrite::Identity,
            component: "model".to_owned(),
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&missing_match, DIGEST, &source),
        Err(ModelFamilyError::KeySelectorCardinality { actual: 0, .. })
    ));

    let rewrite_mismatch = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Move {
            selector: ModelKeySelector::exact("one")?,
            rewrite: ModelKeyRewrite::prefix("wrong.", "model.")?,
            component: "model".to_owned(),
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&rewrite_mismatch, DIGEST, &source),
        Err(ModelFamilyError::KeyRewriteMismatch { .. })
    ));

    let reject_unmatched = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Drop {
            selector: ModelKeySelector::exact("one")?,
        }],
        ModelUnmatchedKeyDisposition::Reject,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&reject_unmatched, DIGEST, &source),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys.len() == 3
    ));

    let collision = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::Copy {
                selector: ModelKeySelector::exact("one")?,
                rewrite: ModelKeyRewrite::exact("same")?,
                component: "model".to_owned(),
            },
            ModelStateTransformOperation::Generate {
                shape: vec![ModelDimensionExpression::Literal(1)],
                fill: Scalar::Float(0.0),
                dtype: DType::F32,
                output: target("model", "same")?,
            },
        ],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&collision, DIGEST, &source),
        Err(ModelFamilyError::DuplicateComponentKey { .. })
    ));

    let overlap = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::Move {
                selector: ModelKeySelector::exact("one")?,
                rewrite: ModelKeyRewrite::exact("first")?,
                component: "model".to_owned(),
            },
            ModelStateTransformOperation::Drop {
                selector: ModelKeySelector::exact("one")?,
            },
        ],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&overlap, DIGEST, &source),
        Err(ModelFamilyError::OverlappingStateSelection(key)) if key == "one"
    ));

    let incomplete = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Split {
            source: source_ref("one")?,
            dimension: 0,
            sizes: vec![ModelDimensionExpression::Literal(1)],
            outputs: vec![target("model", "part")?],
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&incomplete, DIGEST, &source),
        Err(ModelFamilyError::IncompleteAssembly { .. })
    ));

    let assembly_mismatch = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Assemble {
            sources: vec![source_ref("matrix")?, source_ref("mismatch")?],
            dimension: 1,
            output: target("model", "assembled")?,
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&assembly_mismatch, DIGEST, &source),
        Err(ModelFamilyError::AssemblyShapeMismatch(_))
    ));

    let reshape_mismatch = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Reshape {
            source: source_ref("one")?,
            shape: vec![ModelDimensionExpression::Literal(3)],
            output: target("model", "reshaped")?,
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&reshape_mismatch, DIGEST, &source),
        Err(ModelFamilyError::ReshapeElementCount { .. })
    ));

    let overflow = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Generate {
            shape: vec![ModelDimensionExpression::Add(
                Box::new(ModelDimensionExpression::Literal(u64::MAX)),
                Box::new(ModelDimensionExpression::Literal(1)),
            )],
            fill: Scalar::Float(0.0),
            dtype: DType::F32,
            output: target("model", "overflow")?,
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&overflow, DIGEST, &source),
        Err(ModelFamilyError::DimensionExpressionOverflow)
    ));

    for (expression, division_by_zero) in [
        (
            ModelDimensionExpression::DivideExact(
                Box::new(ModelDimensionExpression::Literal(1)),
                Box::new(ModelDimensionExpression::Literal(0)),
            ),
            true,
        ),
        (
            ModelDimensionExpression::DivideExact(
                Box::new(ModelDimensionExpression::Literal(3)),
                Box::new(ModelDimensionExpression::Literal(2)),
            ),
            false,
        ),
    ] {
        let plan = ModelStateTransformPlan::checked(
            vec![ModelStateTransformOperation::Generate {
                shape: vec![expression],
                fill: Scalar::Float(0.0),
                dtype: DType::F32,
                output: target(
                    "model",
                    if division_by_zero {
                        "division-zero"
                    } else {
                        "division-remainder"
                    },
                )?,
            }],
            ModelUnmatchedKeyDisposition::Drop,
        )?;
        let error = ModelStateTransaction::new(&backend, &context)
            .execute(&plan, DIGEST, &source)
            .expect_err("non-exact division must fail");
        assert!(if division_by_zero {
            matches!(error, ModelFamilyError::DimensionDivisionByZero)
        } else {
            matches!(error, ModelFamilyError::DimensionDivisionRemainder { .. })
        });
    }

    let mut too_deep = ModelDimensionExpression::Literal(1);
    for _ in 0..32 {
        too_deep = ModelDimensionExpression::Add(
            Box::new(too_deep),
            Box::new(ModelDimensionExpression::Literal(0)),
        );
    }
    assert!(matches!(
        ModelStateTransformPlan::checked(
            vec![ModelStateTransformOperation::Generate {
                shape: vec![too_deep],
                fill: Scalar::Float(0.0),
                dtype: DType::F32,
                output: target("model", "too-deep")?,
            }],
            ModelUnmatchedKeyDisposition::Drop,
        ),
        Err(ModelFamilyError::DimensionExpressionTooDeep)
    ));

    let missing_dimension = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Generate {
            shape: vec![ModelDimensionExpression::source_dimension("one", 2)?],
            fill: Scalar::Float(0.0),
            dtype: DType::F32,
            output: target("model", "missing-dimension")?,
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&missing_dimension, DIGEST, &source),
        Err(ModelFamilyError::DimensionOutOfBounds { .. })
    ));

    let rollback = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::Generate {
                shape: vec![ModelDimensionExpression::Literal(128)],
                fill: Scalar::Float(42.0),
                dtype: DType::F32,
                output: target("staged", "not-published")?,
            },
            ModelStateTransformOperation::ConditionalRound {
                source: source_ref("half")?,
                decimals: 0,
                condition: ModelRoundCondition::Always,
                output: target("model", "unsupported")?,
            },
        ],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(
        ModelStateTransaction::new(&backend, &context)
            .execute(&rollback, DIGEST, &source)
            .is_err()
    );
    assert_eq!(
        source
            .iter()
            .map(|(key, tensor)| (key.clone(), tensor.storage_id()))
            .collect::<BTreeMap<_, _>>(),
        original_storage
    );

    let unavailable_staged = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Transpose {
            source: ModelStateTensorReference::staged(target("future", "tensor")?),
            first_dimension: 0,
            second_dimension: 0,
            output: target("model", "never-published")?,
        }],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &unavailable_staged,
            DIGEST,
            &source,
        ),
        Err(ModelFamilyError::StagedOutputUnavailable { component, key })
            if component == "future" && key == "tensor"
    ));
    assert!(matches!(
        ModelStateTransformPlan::checked(
            vec![ModelStateTransformOperation::Copy {
                selector: ModelKeySelector::exact("one")?,
                rewrite: ModelKeyRewrite::Pipeline(Vec::new()),
                component: "model".to_owned(),
            }],
            ModelUnmatchedKeyDisposition::Drop,
        ),
        Err(ModelFamilyError::InvalidStateTransform(_))
    ));

    let (oom_backend, oom_authority) = CpuWorkspaceAuthority::create_backend(1024)?;
    let oom_cancellation = CancellationToken::default();
    let oom_context = oom_backend.execution_context(
        StreamId::DEFAULT,
        oom_authority.authorize_workspace(1024)?,
        &oom_cancellation,
    );
    let oom_plan = ModelStateTransformPlan::checked(
        vec![
            ModelStateTransformOperation::Generate {
                shape: vec![ModelDimensionExpression::Literal(128)],
                fill: Scalar::Float(1.0),
                dtype: DType::F32,
                output: target("staged", "first")?,
            },
            ModelStateTransformOperation::Generate {
                shape: vec![ModelDimensionExpression::Literal(256)],
                fill: Scalar::Float(2.0),
                dtype: DType::F32,
                output: target("staged", "oom")?,
            },
        ],
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    let oom_source = BTreeMap::new();
    let oom_baseline = oom_backend.memory_snapshot().current_bytes;
    let oom_error = ModelStateTransaction::new(&oom_backend, &oom_context)
        .execute(&oom_plan, DIGEST, &oom_source)
        .expect_err("second staged allocation must exceed the canonical backend limit");
    assert!(matches!(
        oom_error,
        ModelFamilyError::TensorCreationOperation(TensorCreationPartOneError::Tensor {
            source: TensorError::AllocationFailed {
                requested: 1_024,
                ..
            },
            ..
        })
    ));
    assert_eq!(oom_backend.memory_snapshot().current_bytes, oom_baseline);
    assert!(oom_source.is_empty());

    let (cancellable_backend, cancellable_authority) =
        CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellable_backend = Arc::new(cancellable_backend);
    let staged_cancellation = CancellationToken::default();
    let cancellable_context = cancellable_backend.execution_context(
        StreamId::DEFAULT,
        cancellable_authority.authorize_workspace(1024 * 1024)?,
        &staged_cancellation,
    );
    let staged_cancel_plan = ModelStateTransformPlan::checked(
        (0..4_096)
            .map(|index| ModelStateTransformOperation::Generate {
                shape: vec![ModelDimensionExpression::Literal(256)],
                fill: Scalar::Float(index as f64),
                dtype: DType::F32,
                output: ModelStateTarget {
                    component: "staged".to_owned(),
                    key: format!("tensor.{index}"),
                },
            })
            .collect(),
        ModelUnmatchedKeyDisposition::Drop,
    )?;
    let cancellation_watcher = thread::spawn({
        let cancellable_backend = Arc::clone(&cancellable_backend);
        let staged_cancellation = staged_cancellation.clone();
        move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if cancellable_backend.memory_snapshot().current_bytes > 0 {
                    staged_cancellation.cancel();
                    return true;
                }
                thread::yield_now();
            }
            false
        }
    });
    let staged_cancel_error =
        ModelStateTransaction::new(&cancellable_backend, &cancellable_context)
            .execute(&staged_cancel_plan, DIGEST, &BTreeMap::new())
            .expect_err("post-staging cancellation must roll back the transaction");
    assert!(
        matches!(staged_cancel_error, ModelFamilyError::Cancelled(_)),
        "unexpected staged cancellation error: {staged_cancel_error:?}"
    );
    let observed_staging = cancellation_watcher
        .join()
        .map_err(|_| "staged cancellation watcher panicked")?;
    assert!(
        observed_staging,
        "watcher did not observe a staged allocation"
    );
    assert_eq!(cancellable_backend.memory_snapshot().current_bytes, 0);

    let oversized_operation = ModelStateTransformOperation::Drop {
        selector: ModelKeySelector::exact("one")?,
    };
    assert!(matches!(
        ModelStateTransformPlan::checked(
            vec![oversized_operation; 4_097],
            ModelUnmatchedKeyDisposition::Drop,
        ),
        Err(ModelFamilyError::StatePlanTooLarge(4_097))
    ));

    let source_template = source["one"].clone();
    let oversized_source = (0..16_385)
        .map(|index| (format!("tensor.{index}"), source_template.clone()))
        .collect::<BTreeMap<_, _>>();
    let empty_plan =
        ModelStateTransformPlan::checked(Vec::new(), ModelUnmatchedKeyDisposition::Drop)?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &empty_plan,
            DIGEST,
            &oversized_source,
        ),
        Err(ModelFamilyError::StateSourceTooLarge(16_385))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(64 * 1024)?,
        &cancelled,
    );
    let invalid_after_cancel = ModelStateTransformPlan::checked(
        vec![ModelStateTransformOperation::Move {
            selector: ModelKeySelector::exact("missing")?,
            rewrite: ModelKeyRewrite::Identity,
            component: "model".to_owned(),
        }],
        ModelUnmatchedKeyDisposition::Reject,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &cancelled_context).execute(
            &invalid_after_cancel,
            "not-a-digest",
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

const MODEL_FAMILY_CATALOG_INPUT_ALLOWLIST: [&str; 3] = [
    "projects/comfy/ComfyUI/sim/supported_models.py",
    "projects/comfy/ComfyUI/sim/supported_models_base.py",
    "crates/sim_test_support/parity/catalogs/backend-models.csv",
];

fn normalize_catalog_source_bytes(bytes: Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Ok(String::from_utf8(bytes)?
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .into_bytes())
}

fn verify_catalog_input_provenance(
    catalog: &serde_json::Value,
    repository_root: &Path,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let repository_root = repository_root.canonicalize()?;
    let inputs = catalog["inputs"]
        .as_array()
        .ok_or("catalog inputs missing")?;
    if inputs.len() != MODEL_FAMILY_CATALOG_INPUT_ALLOWLIST.len() {
        return Err(format!(
            "catalog has {} source inputs, expected {}",
            inputs.len(),
            MODEL_FAMILY_CATALOG_INPUT_ALLOWLIST.len()
        )
        .into());
    }
    let mut by_path = BTreeMap::new();
    for input in inputs {
        let path = input["path"].as_str().ok_or("catalog input path missing")?;
        if !MODEL_FAMILY_CATALOG_INPUT_ALLOWLIST.contains(&path) {
            return Err(format!("catalog input {path:?} is not allowlisted").into());
        }
        if by_path.insert(path, input).is_some() {
            return Err(format!("catalog input {path:?} is duplicated").into());
        }
    }

    let mut verified = Vec::new();
    for relative_path in MODEL_FAMILY_CATALOG_INPUT_ALLOWLIST {
        let input = by_path
            .get(relative_path)
            .copied()
            .ok_or_else(|| format!("catalog input {relative_path:?} is missing"))?;
        let resolved_path = repository_root.join(relative_path).canonicalize()?;
        if !resolved_path.starts_with(&repository_root) {
            return Err(format!("catalog input {relative_path:?} escapes the repository").into());
        }
        let normalized = normalize_catalog_source_bytes(fs::read(&resolved_path)?)?;
        let computed_digest = format!("{:x}", Sha256::digest(&normalized));
        let pinned_digest = input["normalized_sha256"]
            .as_str()
            .ok_or("catalog input digest missing")?;
        if computed_digest != pinned_digest {
            return Err(format!(
                "catalog input {relative_path:?} digest mismatch: pinned {pinned_digest}, computed {computed_digest}"
            )
            .into());
        }
        let pinned_bytes = input["normalized_bytes"]
            .as_u64()
            .ok_or("catalog normalized byte count missing")?;
        let normalized_bytes = u64::try_from(normalized.len())?;
        if normalized_bytes != pinned_bytes {
            return Err(format!(
                "catalog input {relative_path:?} byte-count mismatch: pinned {pinned_bytes}, computed {normalized_bytes}"
            )
            .into());
        }
        verified.push(serde_json::json!({
            "path": relative_path,
            "normalized_bytes": normalized_bytes,
            "normalized_sha256": computed_digest,
        }));
    }
    Ok(verified)
}

#[test]
fn catalog_input_provenance_recomputes_allowlisted_sources_and_rejects_mismatch()
-> Result<(), Box<dyn std::error::Error>> {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let catalog_bytes =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"))?;
    let catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    let verified = verify_catalog_input_provenance(&catalog, &repository_root)?;
    assert_eq!(verified.len(), MODEL_FAMILY_CATALOG_INPUT_ALLOWLIST.len());

    let mut stale = catalog;
    stale["inputs"][0]["normalized_sha256"] = serde_json::Value::String("0".repeat(64));
    assert!(verify_catalog_input_provenance(&stale, &repository_root).is_err());
    Ok(())
}

#[test]
fn val_model_family_foundation_001_writes_nonaggregate_transaction_artifact()
-> Result<(), Box<dyn std::error::Error>> {
    identity_registry_detection_and_descriptor_are_checked_and_stable()?;
    tokenizer_and_clip_target_descriptors_cover_catalog_and_data_only_selection()?;
    registrations_preserve_source_order_and_reject_duplicate_ordinals()?;
    encoded_state_plans_revalidate_selectors_expressions_and_evaluation_context()?;
    probe_resolution_checks_source_contract_and_revalidates_selector_output()?;
    data_plan_resolution_maps_component_schema_and_binds_probe_family_and_profile()?;
    component_schemas_require_required_coverage_and_close_optional_components_when_present()?;
    shape_reduced_programs_delegate_every_foundational_step_class()?;
    probe_aware_build_is_typed_failure_atomic_for_oom_dtype_and_cancellation()?;
    build_manifest_owns_sorted_family_registry_and_fixture_closure()?;
    foundation_mapping_build_memory_forward_dtype_device_oom_and_cancellation()?;
    state_transaction_key_primitives_are_bounded_alias_preserving_and_multi_component()?;
    state_transaction_delegates_split_assembly_views_round_and_generation()?;
    selector_driven_transforms_cover_dynamic_source_patterns_without_callbacks()?;
    selected_tensor_dimensions_and_branch_transforms_cover_pinned_family_shapes()?;
    state_transaction_prevalidates_errors_and_rolls_back_on_failure_or_cancellation()?;
    patch_graph_is_ordered_copy_on_write_transactional_and_typed()?;
    patch_graph_delegates_tensor_arithmetic_to_sim_tensor()?;
    catalog_input_provenance_recomputes_allowlisted_sources_and_rejects_mismatch()?;

    let catalog_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json");
    let catalog_bytes = fs::read(&catalog_path)?;
    let catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    let models = catalog["models"]
        .as_array()
        .ok_or("catalog models missing")?;
    assert_eq!(catalog["model_count"].as_u64(), Some(94));
    assert_eq!(models.len(), 94);
    let source_ordinals = models
        .iter()
        .map(|model| {
            model["source_ordinal"]
                .as_u64()
                .ok_or("source ordinal missing")
        })
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    assert_eq!(source_ordinals.len(), 94);
    assert_eq!(source_ordinals.first().copied(), Some(0));
    assert_eq!(source_ordinals.last().copied(), Some(93));
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pinned_catalog_inputs = verify_catalog_input_provenance(&catalog, &repository_root)?;
    let catalog_digest = format!("{:x}", Sha256::digest(&catalog_bytes));
    let test_source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/model_family_foundation.rs");
    let generator_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/generate_model_family_catalog.py");
    let test_source_digest = format!("{:x}", Sha256::digest(fs::read(test_source_path)?));
    let generator_digest = format!("{:x}", Sha256::digest(fs::read(generator_path)?));
    let artifact = serde_json::json!({
        "schema_version": 1,
        "validation_id": "VAL-MODEL-FAMILY-FOUNDATION-001",
        "scope": "synthetic-foundation-only",
        "aggregate_model_family_breadth_claimed": false,
        "backend": "sim_tensor::CpuBackend",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        },
        "fixture_digests": {
            "model_family_catalog": catalog_digest,
            "foundation_test_source": test_source_digest,
            "catalog_generator": generator_digest,
            "pinned_catalog_inputs": pinned_catalog_inputs,
        },
        "cases": [
            {"id": "identity-registration-profile", "status": "passed"},
            {"id": "catalog-tokenizer-clip-target-descriptors", "status": "passed"},
            {"id": "ordered-registration-duplicate-rejection", "status": "passed"},
            {"id": "encoded-plan-structural-revalidation", "status": "passed"},
            {"id": "probe-resolution-contract", "status": "passed"},
            {"id": "data-plan-component-schema-binding", "status": "passed"},
            {"id": "required-and-optional-component-schema-closure", "status": "passed"},
            {"id": "shape-reduced-forward-programs", "status": "passed"},
            {"id": "probe-aware-build-failures", "status": "passed"},
            {"id": "generated-manifest-closure", "status": "passed"},
            {"id": "mapping-build-memory-forward", "status": "passed"},
            {"id": "bounded-key-selection-rewrite", "status": "passed"},
            {"id": "dimension-expression-split-assembly", "status": "passed"},
            {"id": "selector-driven-dynamic-transforms", "status": "passed"},
            {"id": "selected-dimensions-and-branch-transforms", "status": "passed"},
            {"id": "prevalidation-collision-overlap-coverage-shape-overflow", "status": "passed"},
            {"id": "ordered-copy-on-write-patch-graph", "status": "passed"},
            {"id": "canonical-tensor-arithmetic-delegation", "status": "passed"},
            {"id": "computed-catalog-input-provenance", "status": "passed"},
        ],
        "passed": 19,
        "failed": 0,
        "skipped": 0,
    });
    let output = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/sim-parity/val-model-family-foundation-001.json");
    let parent = output.parent().ok_or("artifact has no parent")?;
    fs::create_dir_all(parent)?;
    fs::write(&output, serde_json::to_vec_pretty(&artifact)?)?;
    let decoded: serde_json::Value = serde_json::from_slice(&fs::read(&output)?)?;
    assert_eq!(decoded, artifact);
    Ok(())
}

#[test]
fn patch_graph_is_ordered_copy_on_write_transactional_and_typed()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(64 * 1024)?,
        &cancellation,
    );
    let source = mapped_weights(&backend, &context, false)?;
    let original_weight = source
        .tensors()
        .get("model.weight")
        .ok_or("missing test weight")?;
    let operations = all_patch_kinds();
    let graph = PatchGraph::checked(DIGEST, operations.clone())?;
    let reordered = PatchGraph::checked(DIGEST, operations.into_iter().rev().collect())?;
    assert_ne!(
        graph.identity().ordered_digest,
        reordered.identity().ordered_digest
    );
    assert_eq!(source.base_artifact_digest(), DIGEST);
    assert_ne!(source.cache_identity(), DIGEST);
    let patched = graph.apply(&backend, &source, &context)?;
    let patched_again = graph.apply(&backend, &source, &context)?;
    let reordered_patch = reordered.apply(&backend, &source, &context)?;
    let applied_identity = |ordered_digest: &str, compute_dtype: &str, storage_dtype: &str| {
        let mut digest = Sha256::new();
        digest.update(b"zed-sim-patch-applied-compute-v1\0");
        digest.update(u64::try_from(ordered_digest.len())?.to_le_bytes());
        digest.update(ordered_digest.as_bytes());
        digest.update(u64::try_from("configured".len())?.to_le_bytes());
        digest.update(b"configured");
        digest.update(1_u64.to_le_bytes());
        for value in ["model.weight", compute_dtype, storage_dtype] {
            digest.update(u64::try_from(value.len())?.to_le_bytes());
            digest.update(value.as_bytes());
        }
        Ok::<String, std::num::TryFromIntError>(format!("{:x}", digest.finalize()))
    };
    let graph_applied_identity = applied_identity(&graph.identity().ordered_digest, "f32", "f32")?;
    let mut expected_cache_identity = Sha256::new();
    expected_cache_identity.update(b"zed.sim.model-weights-cache-identity.v1\0");
    expected_cache_identity.update(source.cache_identity().as_bytes());
    expected_cache_identity.update([0]);
    expected_cache_identity.update(graph_applied_identity.as_bytes());
    assert_ne!(patched.cache_identity(), source.cache_identity());
    assert_eq!(
        patched.cache_identity(),
        format!("{:x}", expected_cache_identity.finalize())
    );
    assert_eq!(patched.cache_identity(), patched_again.cache_identity());
    assert_ne!(patched.cache_identity(), reordered_patch.cache_identity());
    let composed_patch = reordered.apply(&backend, &patched, &context)?;
    let reordered_applied_identity =
        applied_identity(&reordered.identity().ordered_digest, "f32", "f32")?;
    let mut expected_composed_identity = Sha256::new();
    expected_composed_identity.update(b"zed.sim.model-weights-cache-identity.v1\0");
    expected_composed_identity.update(patched.cache_identity().as_bytes());
    expected_composed_identity.update([0]);
    expected_composed_identity.update(reordered_applied_identity.as_bytes());
    assert_eq!(
        composed_patch.cache_identity(),
        format!("{:x}", expected_composed_identity.finalize())
    );
    assert_ne!(
        composed_patch.cache_identity(),
        reordered_patch.cache_identity()
    );
    assert_eq!(source.base_artifact_digest(), DIGEST);
    assert_eq!(values(&backend, original_weight, &context)?, [2.0, -3.0]);
    assert_eq!(
        values(
            &backend,
            patched
                .tensors()
                .get("model.weight")
                .ok_or("missing patched weight")?,
            &context
        )?,
        [6.0, 6.0]
    );
    assert_ne!(
        original_weight.storage_id(),
        patched
            .tensors()
            .get("model.weight")
            .ok_or("missing patched weight")?
            .storage_id()
    );

    let invalid = PatchGraph::checked(
        DIGEST,
        vec![
            PatchOperation {
                identifier: "staged-before-failure".to_owned(),
                kind: PatchKind::Lora,
                scale: 1.0,
                targets: vec![PatchTarget {
                    key: "model.weight".to_owned(),
                    expected_shape: vec![2],
                    values: vec![100.0, 100.0],
                    application: PatchApplication::Add,
                }],
            },
            PatchOperation {
                identifier: "missing-target".to_owned(),
                kind: PatchKind::Lora,
                scale: 1.0,
                targets: vec![PatchTarget {
                    key: "model.missing".to_owned(),
                    expected_shape: vec![2],
                    values: vec![1.0, 1.0],
                    application: PatchApplication::Add,
                }],
            },
        ],
    )?;
    assert!(matches!(
        invalid.apply(&backend, &source, &context),
        Err(PatchGraphError::MissingTarget(_))
    ));
    assert_eq!(values(&backend, original_weight, &context)?, [2.0, -3.0]);
    assert_eq!(source.base_artifact_digest(), DIGEST);
    assert!(matches!(
        PatchGraph::checked(OTHER_DIGEST, all_patch_kinds())?.apply(&backend, &source, &context),
        Err(PatchGraphError::BaseDigestMismatch { .. })
    ));

    let half_source = map_model_weights(
        &FOUNDATION,
        DIGEST,
        BTreeMap::from([(
            "source.weight".to_owned(),
            tensor(&backend, &[2], &[2.0, -3.0], DType::F16, &context)?,
        )]),
    )?;
    let half_patched = graph.apply(&backend, &half_source, &context)?;
    let half_weight = half_patched
        .tensors()
        .get("model.weight")
        .ok_or("missing half patched weight")?;
    assert_eq!(half_weight.descriptor().dtype(), DType::F16);
    assert_eq!(
        half_weight.descriptor().device(),
        half_source
            .tensors()
            .get("model.weight")
            .ok_or("missing half source weight")?
            .descriptor()
            .device()
    );
    assert_eq!(values(&backend, half_weight, &context)?, [6.0, 6.0]);

    cancellation.cancel();
    assert!(graph.apply(&backend, &source, &context).is_err());
    assert_eq!(source.base_artifact_digest(), DIGEST);
    Ok(())
}

#[test]
fn patch_graph_delegates_tensor_arithmetic_to_sim_tensor() -> Result<(), Box<dyn std::error::Error>>
{
    let source = include_str!("../src/patch_graph.rs");
    let boundary_start = source
        .find("pub fn apply_with_compute_boundary(")
        .ok_or("missing checked compute boundary")?;
    let implementation_start = source
        .find("fn apply_semantic_operation(")
        .ok_or("missing canonical patch application helper")?;
    let boundary = &source[boundary_start..implementation_start];
    assert!(
        boundary.contains("backend_cast_tensor("),
        "the checked compute boundary must own source, intermediate, and final storage casts"
    );
    let implementation_end = source[implementation_start..]
        .find("\nfn apply_payload_with_original(")
        .map(|offset| implementation_start + offset)
        .ok_or("missing canonical patch application helper boundary")?;
    let implementation = &source[implementation_start..implementation_end];

    for required_call in [
        "apply_payload_with_original(",
        "scatter_slices(",
        "validate_finite_result(",
    ] {
        assert!(
            implementation.contains(required_call),
            "patch application must delegate through {required_call}"
        );
    }
    for forbidden_local_mechanic in [
        "tensor_to_f32_with_context_exact_native(",
        ".iter(",
        ".iter_mut(",
        ".zip(",
        ".enumerate(",
        "for ",
        "+=",
        "*=",
    ] {
        assert!(
            !implementation.contains(forbidden_local_mechanic),
            "patch application must not own local arithmetic: {forbidden_local_mechanic}"
        );
    }
    Ok(())
}

fn foundation_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([("source.weight".to_owned(), vec![2])]),
        metadata: BTreeMap::from([("family".to_owned(), "foundation".to_owned())]),
    }
}

fn operator_probe(program: &str) -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("source.linear_weight".to_owned(), vec![2, 2]),
            ("source.linear_bias".to_owned(), vec![2]),
            ("source.conv1d_weight".to_owned(), vec![1, 1, 1]),
            ("source.conv2d_weight".to_owned(), vec![1, 1, 1, 1]),
            ("source.conv3d_weight".to_owned(), vec![1, 1, 1, 1, 1]),
            ("source.norm_weight".to_owned(), vec![2]),
            ("source.norm_bias".to_owned(), vec![2]),
        ]),
        metadata: BTreeMap::from([
            ("family".to_owned(), "foundation-operators".to_owned()),
            (
                "source_architecture".to_owned(),
                "foundation_ops".to_owned(),
            ),
            ("program".to_owned(), program.to_owned()),
        ]),
    }
}

fn operator_options(
    activation_elements: u64,
    memory_budget_bytes: u64,
) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements,
        memory_budget_bytes,
        allow_unexpected_weights: false,
    }
}

fn operator_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    registry: &ModelFamilyRegistry,
    program: &str,
) -> Result<MappedModelWeights, Box<dyn std::error::Error>> {
    let source = BTreeMap::from([
        (
            "source.linear_weight".to_owned(),
            tensor(backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], DType::F32, context)?,
        ),
        (
            "source.linear_bias".to_owned(),
            tensor(backend, &[2], &[0.5, -0.5], DType::F32, context)?,
        ),
        (
            "source.conv1d_weight".to_owned(),
            tensor(backend, &[1, 1, 1], &[2.0], DType::F32, context)?,
        ),
        (
            "source.conv2d_weight".to_owned(),
            tensor(backend, &[1, 1, 1, 1], &[2.0], DType::F32, context)?,
        ),
        (
            "source.conv3d_weight".to_owned(),
            tensor(backend, &[1, 1, 1, 1, 1], &[2.0], DType::F32, context)?,
        ),
        (
            "source.norm_weight".to_owned(),
            tensor(backend, &[2], &[1.0, 1.0], DType::F32, context)?,
        ),
        (
            "source.norm_bias".to_owned(),
            tensor(backend, &[2], &[0.0, 0.0], DType::F32, context)?,
        ),
    ]);
    let resolved = registry.resolve(&operator_probe(program))?;
    Ok(resolved.map_primary_weights(
        &ModelStateTransaction::new(backend, context),
        DIGEST,
        &source,
    )?)
}

fn mapped_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    extra: bool,
) -> Result<MappedModelWeights, Box<dyn std::error::Error>> {
    let mut source = BTreeMap::from([(
        "source.weight".to_owned(),
        tensor(backend, &[2], &[2.0, -3.0], DType::F32, context)?,
    )]);
    if extra {
        source.insert(
            "extra.unmapped".to_owned(),
            tensor(backend, &[1], &[7.0], DType::F32, context)?,
        );
    }
    Ok(map_model_weights(&FOUNDATION, DIGEST, source)?)
}

fn all_patch_kinds() -> Vec<PatchOperation> {
    let kinds = [
        PatchKind::Lora,
        PatchKind::Loha,
        PatchKind::Lokr,
        PatchKind::Glora,
        PatchKind::ControlNet,
        PatchKind::Adapter,
        PatchKind::Replacement,
    ];
    kinds
        .into_iter()
        .enumerate()
        .map(|(index, kind)| {
            let replacement = kind == PatchKind::Replacement;
            PatchOperation {
                identifier: format!("patch-{index}"),
                kind,
                scale: 1.0,
                targets: vec![PatchTarget {
                    key: "model.weight".to_owned(),
                    expected_shape: vec![2],
                    values: if replacement {
                        vec![f32::from(index as u16), f32::from(index as u16)]
                    } else {
                        vec![1.0, 1.0]
                    },
                    application: if replacement {
                        PatchApplication::Replace
                    } else {
                        PatchApplication::Add
                    },
                }],
            }
        })
        .collect()
}

fn target(component: &str, key: &str) -> Result<ModelStateTarget, Box<dyn std::error::Error>> {
    Ok(ModelStateTarget::checked(component, key)?)
}

fn source_ref(key: &str) -> Result<ModelStateTensorReference, Box<dyn std::error::Error>> {
    Ok(ModelStateTensorReference::source(key)?)
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
    if dtype == DType::F32 {
        Ok(tensor)
    } else {
        Ok(
            sim_tensor::generated_sim_operator_indirection_01::cast_to_with_context_exact_native(
                backend,
                &tensor,
                dtype,
                backend.device(),
                false,
                false,
                context,
            )?,
        )
    }
}

fn values(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend, tensor, context,
    )?)
}
