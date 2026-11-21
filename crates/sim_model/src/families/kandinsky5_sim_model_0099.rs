use crate::{
    KANDINSKY5_COMPONENT_STATE_SCHEMAS, KANDINSKY5_COMPONENTS,
    KANDINSKY5_FORWARD_PROGRAM, KANDINSKY5_LAYOUT_SIGNATURES, KANDINSKY5_MEMORY_USAGE_FACTOR,
    KANDINSKY5_MODEL_REQUIRED_KEYS, KANDINSKY5_STATE_PLAN_CASES, KANDINSKY5_SUPPORTED_DEVICES,
    KANDINSKY5_SUPPORTED_DTYPES,
    KANDINSKY5_VIDEO_CLIP_TARGET, KANDINSKY5_VIDEO_LATENT_FORMAT, Kandinsky5Variant,
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanSelector,
    ModelProbe, ModelTensorFactPredicate, ModelTensorFactRelation, ModelTensorFactSubject,
    ModelWeightRule, kandinsky5_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Kandinsky5";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0099";
pub const MODEL_FAMILY_FIXTURE: &str = "kandinsky5-comfy-model-0099";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 83;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "3ebe95468ad1e7e8429178ed043dbddabd25e291242ca328988a1022ddac5114";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = KANDINSKY5_MEMORY_USAGE_FACTOR;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 10.0;

const VECTOR_FACTS: &[ModelTensorFactPredicate] = &[ModelTensorFactPredicate {
    subject: ModelTensorFactSubject::Rank,
    relation: ModelTensorFactRelation::Equal,
    value: 1,
}];

const EMBEDDING_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::Equal,
        value: 132,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.visual_embeddings.in_layer.bias",
            "visual_embeddings.in_layer.bias",
        ],
        dimension: 0,
        values: &[1_792, 4_096],
        score: 300,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.time_embeddings.in_layer.bias",
            "time_embeddings.in_layer.bias",
        ],
        predicates: VECTOR_FACTS,
        score: 150,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.time_embeddings.in_layer.bias",
            "time_embeddings.in_layer.bias",
        ],
        dimension: 0,
        values: &[512],
        score: 150,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.visual_embeddings.in_layer.weight",
            "visual_embeddings.in_layer.weight",
        ],
        predicates: EMBEDDING_FACTS,
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: &[
            "model.diffusion_model.visual_transformer_blocks.0.cross_attention.key_norm.weight",
            "visual_transformer_blocks.0.cross_attention.key_norm.weight",
        ],
        dimension: 0,
        values: &[64, 128],
        score: 150,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const OPTIONAL_KEYS: &[&str] = &[
    "native.visual_embeddings.in_layer.bias",
    "native.time_embeddings.in_layer.bias",
    "native.time_embeddings.out_layer.weight",
    "native.pooled_text_embeddings.in_layer.weight",
    "native.text_embeddings.in_layer.bias",
    "native.text_transformer_blocks.0.self_attention.to_query.weight",
    "native.visual_transformer_blocks.0.feed_forward.in_layer.weight",
    "native.out_layer.out_layer.bias",
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "kandinsky5-video-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0037",
    latent_identifier: "HunyuanVideo",
    clip_target: &KANDINSKY5_VIDEO_CLIP_TARGET,
    components: KANDINSKY5_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: KANDINSKY5_MODEL_REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: KANDINSKY5_SUPPORTED_DTYPES,
    supported_devices: KANDINSKY5_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: KANDINSKY5_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 83,
    source_architecture: "model_base.Kandinsky5",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&KANDINSKY5_VIDEO_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: KANDINSKY5_LAYOUT_SIGNATURES,
        cases: KANDINSKY5_STATE_PLAN_CASES,
    },
    component_state_schemas: KANDINSKY5_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = kandinsky5_configuration_for_probe(probe)?;
    if !matches!(
        configuration.variant,
        Kandinsky5Variant::VideoLite | Kandinsky5Variant::VideoPro
    ) {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "Kandinsky5 row cannot admit {:?}",
            configuration.variant
        )));
    }
    if configuration.latent_format.feature_id != KANDINSKY5_VIDEO_LATENT_FORMAT.feature_id
        || configuration.latent_format.identifier != KANDINSKY5_VIDEO_LATENT_FORMAT.identifier
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "Kandinsky5 video latent selection drifted".to_owned(),
        ));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
