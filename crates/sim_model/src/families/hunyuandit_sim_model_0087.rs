use crate::{
    HUNYUANDIT_BASE_EXTRA_INPUT, HUNYUANDIT_CLIP_TARGET, HUNYUANDIT_COMPONENTS,
    HUNYUANDIT_FORWARD_PROGRAM, HUNYUANDIT_MEMORY_USAGE_FACTOR, HUNYUANDIT_PREFIXED_STATE_PLAN,
    HUNYUANDIT_SAVED_MODEL_STATE_PLAN, HUNYUANDIT_STANDALONE_STATE_PLAN,
    HUNYUANDIT_SUPPORTED_DEVICES, HUNYUANDIT_SUPPORTED_DTYPES, HunyuanDiTVariant,
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelLayoutSignature, ModelProbe, ModelStateLayout, ModelTensorFactPredicate,
    ModelTensorFactRelation, ModelTensorFactSubject, ModelWeightRule,
    hunyuandit_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "HunyuanDiT";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0087";
pub const MODEL_FAMILY_FIXTURE: &str = "hunyuandit-comfy-model-0087";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 25;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "67ec87dadfe0744beb031c051e27d8809447f3940d774812840991ed2ea26d95";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = HUNYUANDIT_MEMORY_USAGE_FACTOR;
pub const MODEL_FAMILY_SAMPLING_LINEAR_START: f64 = 0.00085;
pub const MODEL_FAMILY_SAMPLING_LINEAR_END: f64 = 0.018;

const T5_PROJECTION_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(0),
        relation: ModelTensorFactRelation::Equal,
        value: 8_192,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::Equal,
        value: 2_048,
    },
];

const PATCH_EMBEDDING_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 4,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::Equal,
        value: 4,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(2),
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(3),
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
];

const BASE_EXTRA_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::Equal,
        value: HUNYUANDIT_BASE_EXTRA_INPUT,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.mlp_t5.0.weight",
            "model.mlp_t5.0.weight",
            "mlp_t5.0.weight",
        ],
        predicates: T5_PROJECTION_FACTS,
        score: 250,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.x_embedder.proj.weight",
            "model.x_embedder.proj.weight",
            "x_embedder.proj.weight",
        ],
        predicates: PATCH_EMBEDDING_FACTS,
        score: 250,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.extra_embedder.0.weight",
            "model.extra_embedder.0.weight",
            "extra_embedder.0.weight",
        ],
        predicates: BASE_EXTRA_FACTS,
        score: 500,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.mlp_t5.0.weight",
    "native.mlp_t5.0.bias",
    "native.mlp_t5.2.weight",
    "native.text_embedding_padding",
    "native.pooler.q_proj.weight",
    "native.x_embedder.proj.weight",
    "native.t_embedder.mlp.0.weight",
    "native.extra_embedder.0.weight",
    "native.extra_embedder.0.bias",
    "native.blocks.0.attn1.qkv.weight",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.mlp_t5.2.bias",
    "native.pooler.k_proj.weight",
    "native.pooler.v_proj.weight",
    "native.pooler.c_proj.weight",
    "native.t_embedder.mlp.0.bias",
    "native.style_embedder.weight",
    "native.final_layer.adaLN_modulation.1.weight",
    "native.final_layer.adaLN_modulation.1.bias",
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "hunyuandit-v-prediction-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0047",
    latent_identifier: "SDXL",
    clip_target: &HUNYUANDIT_CLIP_TARGET,
    components: HUNYUANDIT_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: HUNYUANDIT_SUPPORTED_DTYPES,
    supported_devices: HUNYUANDIT_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: HUNYUANDIT_FORWARD_PROGRAM,
};

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.mlp_t5.0.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["model.mlp_t5.0.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &["mlp_t5.0.weight"],
        required_prefixes: &[],
    },
];

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &HUNYUANDIT_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &HUNYUANDIT_SAVED_MODEL_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &HUNYUANDIT_STANDALONE_STATE_PLAN,
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
    source_ordinal: 25,
    source_architecture: "model_base.HunyuanDiT",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&HUNYUANDIT_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = hunyuandit_configuration_for_probe(probe)?;
    if configuration.variant != HunyuanDiTVariant::DiT {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "HunyuanDiT row cannot admit {:?}",
            configuration.variant
        )));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
