use crate::{
    LOTUS_CONDITIONING, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule, SD2_CLIP_TARGET,
    SD2_COMPONENT_STATE_SCHEMAS, SD2_COMPONENTS, SD2_CONTEXT_DIMENSION,
    SD2_DIFFUSERS_STATE_PLAN, SD2_FORWARD_PROGRAM, SD2_MEMORY_USAGE_FACTOR, SD2_MODEL_CHANNELS,
    SD2_MODEL_OPTIONAL_KEYS, SD2_MODEL_REQUIRED_KEYS, SD2_PREFIXED_STATE_PLAN,
    SD2_SUPPORTED_DEVICES, SD2_SUPPORTED_DTYPES, Sd2Configuration, Sd2ModelType, Sd2Variant,
    sd2_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "LotusD";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0106";
pub const MODEL_FAMILY_FIXTURE: &str = "lotusd-comfy-model-0106";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 0;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "337fb69915e02edd144ca36e4fa6ee72d34742d1a49eee59d30e1383b2d63b03";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = SD2_MEMORY_USAGE_FACTOR;

const INPUT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.0.0.weight",
    "input_blocks.0.0.weight",
    "conv_in.weight",
];
const CONTEXT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
];
const ADM_KEYS: &[&str] = &[
    "model.diffusion_model.label_emb.0.0.weight",
    "label_emb.0.0.weight",
    "class_embedding.linear_1.weight",
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: INPUT_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 0,
        values: &[SD2_MODEL_CHANNELS],
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 1,
        values: &[4],
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: CONTEXT_KEYS,
        dimension: 1,
        values: &[SD2_CONTEXT_DIMENSION],
        score: 200,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: ADM_KEYS,
        dimension: 1,
        values: &[4],
        score: 550,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "model.diffusion_model.",
        target_prefix: "native.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "cond_stage_model.model.",
        target_prefix: "clip_h.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "first_stage_model.",
        target_prefix: "native.",
        required: false,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "lotus-depth-unet-v1",
    latent_feature_id: "COMFY-MODEL-0045",
    latent_identifier: "SD15",
    clip_target: &SD2_CLIP_TARGET,
    components: SD2_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: SD2_MODEL_REQUIRED_KEYS,
    optional_keys: SD2_MODEL_OPTIONAL_KEYS,
    supported_dtypes: SD2_SUPPORTED_DTYPES,
    supported_devices: SD2_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: SD2_FORWARD_PROGRAM,
};

const LOTUS_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"input_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"input_blocks.","to":"native.input_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_embed."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_embed.","to":"native.time_embed."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"label_emb."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"label_emb.","to":"native.label_emb."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"middle_block."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"middle_block.","to":"native.middle_block."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"output_blocks."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"output_blocks.","to":"native.output_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"out."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"out.","to":"native.out."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"clip_h."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const LOTUS_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.out.2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "input_blocks.0.0.weight",
            "time_embed.0.weight",
            "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            "input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            "out.2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
            "conv_out.weight",
        ],
        required_prefixes: &[],
    },
];

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &SD2_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &SD2_DIFFUSERS_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &LOTUS_STANDALONE_STATE_PLAN,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 0,
    source_architecture: "model_base.Lotus",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&SD2_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LOTUS_LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: SD2_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    if configuration.variant != Sd2Variant::LotusD
        || configuration.model_type != Sd2ModelType::ImgToImg
        || configuration.conditioning != LOTUS_CONDITIONING
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "LotusD row requires the exact four-channel ADM depth profile".to_owned(),
        ));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<Sd2Configuration, ModelFamilyError> {
    if probe.select_layout(LOTUS_LAYOUT_SIGNATURES)? != ModelStateLayout::StandaloneNative {
        return sd2_configuration_for_probe(probe, None);
    }
    let prefixed = ModelProbe {
        tensor_shapes: probe
            .tensor_shapes
            .iter()
            .map(|(key, shape)| (format!("model.diffusion_model.{key}"), shape.clone()))
            .collect(),
        metadata: probe.metadata.clone(),
    };
    sd2_configuration_for_probe(&prefixed, None)
}
