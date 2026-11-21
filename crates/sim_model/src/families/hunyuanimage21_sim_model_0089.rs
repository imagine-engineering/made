use crate::{
    HUNYUAN_IMAGE_CLIP_TARGET, HUNYUAN_IMAGE21_LATENT_FORMAT,
    HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS, HUNYUAN_VIDEO_COMPONENTS,
    HUNYUAN_VIDEO_FORWARD_PROGRAM, HUNYUAN_VIDEO_PREFIXED_STATE_PLAN,
    HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN, HUNYUAN_VIDEO_STANDALONE_STATE_PLAN,
    HUNYUAN_VIDEO_SUPPORTED_DEVICES, HUNYUAN_VIDEO_SUPPORTED_DTYPES, HunyuanVideoVariant,
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe,
    ModelStateLayout, ModelTensorFactPredicate, ModelTensorFactRelation, ModelTensorFactSubject,
    ModelWeightRule, hunyuan_video_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "HunyuanImage21";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0089";
pub const MODEL_FAMILY_FIXTURE: &str = "hunyuanimage21-comfy-model-0089";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 37;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "2bb071c561221be7a662c86760d3f91c449bc033b849fed58ad2ec60714ef07b";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 8.7;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 5.0;

const MARKER_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 1,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(0),
        relation: ModelTensorFactRelation::GreaterThan,
        value: 0,
    },
];

const PATCH_PROJECTION_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 4,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::Equal,
        value: 64,
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

const FINAL_PROJECTION_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(0),
        relation: ModelTensorFactRelation::Equal,
        value: 256,
    },
];

const CONTEXT_PROJECTION_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::GreaterThan,
        value: 0,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.txt_in.individual_token_refiner.blocks.0.norm1.weight",
            "model.txt_in.individual_token_refiner.blocks.0.norm1.weight",
            "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        ],
        predicates: MARKER_FACTS,
        score: 100,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.img_in.proj.weight",
            "model.img_in.proj.weight",
            "img_in.proj.weight",
        ],
        predicates: PATCH_PROJECTION_FACTS,
        score: 450,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.final_layer.linear.weight",
            "model.final_layer.linear.weight",
            "final_layer.linear.weight",
        ],
        predicates: FINAL_PROJECTION_FACTS,
        score: 250,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.txt_in.input_embedder.weight",
            "model.txt_in.input_embedder.weight",
            "txt_in.input_embedder.weight",
        ],
        predicates: CONTEXT_PROJECTION_FACTS,
        score: 200,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &[
    "native.img_in.proj.weight",
    "native.final_layer.linear.weight",
    "native.final_layer.linear.bias",
    "native.txt_in.input_embedder.weight",
    "native.txt_in.input_embedder.bias",
    "native.txt_in.individual_token_refiner.blocks.0.norm1.weight",
];

const OPTIONAL_KEYS: &[&str] = &["native.txt_in.t_embedder.in_layer.weight"];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "hunyuan-image-2.1-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0035",
    latent_identifier: "HunyuanImage21",
    clip_target: &HUNYUAN_IMAGE_CLIP_TARGET,
    components: HUNYUAN_VIDEO_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: HUNYUAN_VIDEO_SUPPORTED_DTYPES,
    supported_devices: HUNYUAN_VIDEO_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: HUNYUAN_VIDEO_FORWARD_PROGRAM,
};

const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.txt_in.individual_token_refiner.blocks.0.norm1.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &["model.txt_in.individual_token_refiner.blocks.0.norm1.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &["txt_in.individual_token_refiner.blocks.0.norm1.weight"],
        required_prefixes: &[],
    },
];

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &HUNYUAN_VIDEO_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &HUNYUAN_VIDEO_STANDALONE_STATE_PLAN,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 37,
    source_architecture: "model_base.HunyuanImage21",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&HUNYUAN_IMAGE_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = hunyuan_video_configuration_for_probe(probe)?;
    if configuration.variant != HunyuanVideoVariant::Image21 {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "HunyuanImage21 row cannot admit {:?}",
            configuration.variant
        )));
    }
    if configuration.latent_format.feature_id != HUNYUAN_IMAGE21_LATENT_FORMAT.feature_id
        || configuration.latent_format.identifier != HUNYUAN_IMAGE21_LATENT_FORMAT.identifier
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "HunyuanImage21 latent selection drifted".to_owned(),
        ));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
