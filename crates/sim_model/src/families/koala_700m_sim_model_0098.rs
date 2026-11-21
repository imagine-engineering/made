use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector, ModelProbe, ModelTensorFactPredicate,
    ModelTensorFactRelation, ModelTensorFactSubject, ModelWeightRule, SDXL_CLIP_TARGET,
    SDXL_COMPONENT_STATE_SCHEMAS, SDXL_COMPONENTS, SDXL_FORWARD_PROGRAM,
    SDXL_KOALA_700M_TRANSFORMER_DEPTH, SDXL_LAYOUT_SIGNATURES, SDXL_LATENT_FORMAT,
    SDXL_MEMORY_USAGE_FACTOR, SDXL_STATE_PLAN_CASES, SDXL_SUPPORTED_DEVICES,
    SDXL_SUPPORTED_DTYPES, SdxlVariant, sdxl_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "KOALA_700M";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0098";
pub const MODEL_FAMILY_FIXTURE: &str = "koala-700m-comfy-model-0098";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 11;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "9b4cc83a375dcbdb48add0debaee0b603e76a371cb3e7d53da6ffb2b920db0e2";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = SDXL_MEMORY_USAGE_FACTOR;

const INPUT_FACTS: &[ModelTensorFactPredicate] = &[
    fact(ModelTensorFactSubject::Rank, 4),
    fact(ModelTensorFactSubject::Dimension(0), 320),
    fact(ModelTensorFactSubject::Dimension(1), 4),
    fact(ModelTensorFactSubject::Dimension(2), 3),
    fact(ModelTensorFactSubject::Dimension(3), 3),
];
const TIME_FACTS: &[ModelTensorFactPredicate] = &[
    fact(ModelTensorFactSubject::Rank, 2),
    fact(ModelTensorFactSubject::Dimension(0), 1_280),
    fact(ModelTensorFactSubject::Dimension(1), 320),
];
const ADM_FACTS: &[ModelTensorFactPredicate] = &[
    fact(ModelTensorFactSubject::Rank, 2),
    fact(ModelTensorFactSubject::Dimension(0), 1_280),
    fact(ModelTensorFactSubject::Dimension(1), 2_816),
];
const CONTEXT_FACTS: &[ModelTensorFactPredicate] = &[
    fact(ModelTensorFactSubject::Rank, 2),
    fact(ModelTensorFactSubject::Dimension(0), 320),
    fact(ModelTensorFactSubject::Dimension(1), 2_048),
];

const fn fact(subject: ModelTensorFactSubject, value: u64) -> ModelTensorFactPredicate {
    ModelTensorFactPredicate {
        subject,
        relation: ModelTensorFactRelation::Equal,
        value,
    }
}

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "input_blocks.0.0.weight",
            "conv_in.weight",
        ],
        predicates: INPUT_FACTS,
        score: 200,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.time_embed.0.weight",
            "time_embed.0.weight",
            "time_embedding.linear_1.weight",
        ],
        predicates: TIME_FACTS,
        score: 150,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.label_emb.0.0.weight",
            "label_emb.0.0.weight",
            "add_embedding.linear_1.weight",
        ],
        predicates: ADM_FACTS,
        score: 200,
    },
    ModelDetectionRule::AnyTensorFact {
        keys: &[
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.4.attn2.to_k.weight",
            "input_blocks.5.1.transformer_blocks.4.attn2.to_k.weight",
            "down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_k.weight",
        ],
        predicates: CONTEXT_FACTS,
        score: 450,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "koala-700m-sdxl-unet-v1",
    latent_feature_id: "COMFY-MODEL-0047",
    latent_identifier: "SDXL",
    clip_target: &SDXL_CLIP_TARGET,
    components: SDXL_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: crate::SDXL_MODEL_REQUIRED_KEYS,
    optional_keys: crate::SDXL_MODEL_OPTIONAL_KEYS,
    supported_dtypes: SDXL_SUPPORTED_DTYPES,
    supported_devices: SDXL_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 1,
        activation_bytes_per_element: 1,
    },
    forward_program: SDXL_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 11,
    source_architecture: "model_base.SDXL",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&SDXL_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: SDXL_LAYOUT_SIGNATURES,
        cases: SDXL_STATE_PLAN_CASES,
    },
    component_state_schemas: SDXL_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = sdxl_configuration_for_probe(probe)?;
    if configuration.variant != SdxlVariant::Koala700M
        || configuration.transformer_depth != SDXL_KOALA_700M_TRANSFORMER_DEPTH
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(format!(
            "KOALA_700M row cannot admit {:?}",
            configuration.variant
        )));
    }
    if configuration.latent_format.feature_id != SDXL_LATENT_FORMAT.feature_id
        || configuration.latent_format.identifier != SDXL_LATENT_FORMAT.identifier
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "KOALA_700M latent selection drifted".to_owned(),
        ));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
