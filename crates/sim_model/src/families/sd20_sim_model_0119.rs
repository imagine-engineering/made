use crate::{
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelProbe, ModelStateLayout,
    ModelWeightRule,
    model_family::{ModelWeightStatisticObservation, ModelWeightStatisticRequest},
    sd2_family::{
        SD2_CLIP_TARGET, SD2_COMPONENT_STATE_SCHEMAS, SD2_COMPONENTS, SD2_CONTEXT_DIMENSION,
        SD2_DIFFUSERS_STATE_PLAN, SD2_FORWARD_PROGRAM, SD2_LAYOUT_SIGNATURES,
        SD2_MEMORY_USAGE_FACTOR, SD2_MODEL_CHANNELS, SD2_MODEL_OPTIONAL_KEYS,
        SD2_MODEL_REQUIRED_KEYS, SD2_PREFIXED_STATE_PLAN, SD2_SUPPORTED_DEVICES,
        SD2_SUPPORTED_DTYPES, Sd2Configuration, Sd2Layout, Sd2Variant,
        configuration_for_probe as sd2_configuration_for_probe,
        weight_statistic_request_for_probe as sd2_weight_statistic_request_for_probe,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "SD20";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0119";
pub const MODEL_FAMILY_FIXTURE: &str = "sd20-comfy-model-0119";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 4;
pub const MODEL_FAMILY_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "176e2c1811195b314300ae92add8bba762c777d0fc1c3c8a4e13d04fd550118c";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = SD2_MEMORY_USAGE_FACTOR;
pub const SOURCE_INPUT_CHANNELS: &[u64] = &[4, 9];

pub const INPUT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.0.0.weight",
    "conv_in.weight",
];
pub const CONTEXT_KEYS: &[&str] = &[
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
];
pub const ADM_KEYS: &[&str] = &[
    "model.diffusion_model.label_emb.0.0.weight",
    "class_embedding.linear_1.weight",
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: INPUT_KEYS,
        score: 300,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 0,
        values: &[SD2_MODEL_CHANNELS],
        score: 250,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: INPUT_KEYS,
        dimension: 1,
        values: SOURCE_INPUT_CHANNELS,
        score: 300,
    },
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: CONTEXT_KEYS,
        dimension: 1,
        values: &[SD2_CONTEXT_DIMENSION],
        score: 350,
    },
];

pub const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "model.diffusion_model.",
        target_prefix: "native.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "conditioner.embedders.0.model.",
        target_prefix: "clip_h.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "cond_stage_model.model.",
        target_prefix: "clip_h.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "embedder.model.visual.",
        target_prefix: "clip_vision.visual.",
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
    architecture_version: "sd20-native-v1",
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

pub const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &SD2_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &SD2_DIFFUSERS_STATE_PLAN,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 4,
    source_architecture: "model_base.BaseModel",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&SD2_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: SD2_LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: SD2_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    validate_variant_discriminator(probe, None, MODEL_FAMILY_IDENTIFIER)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
    statistic: Option<&ModelWeightStatisticObservation>,
) -> Result<Sd2Configuration, ModelFamilyError> {
    configuration_for_probe_kind(probe, statistic, Sd2Variant::Sd20, MODEL_FAMILY_IDENTIFIER)
}

pub fn weight_statistic_request_for_probe(
    probe: &ModelProbe,
) -> Result<Option<ModelWeightStatisticRequest>, ModelFamilyError> {
    validate_variant_discriminator(probe, None, MODEL_FAMILY_IDENTIFIER)?;
    sd2_weight_statistic_request_for_probe(probe)
}

pub fn configuration_for_probe_kind(
    probe: &ModelProbe,
    statistic: Option<&ModelWeightStatisticObservation>,
    expected_variant: Sd2Variant,
    family: &str,
) -> Result<Sd2Configuration, ModelFamilyError> {
    let configuration = sd2_configuration_for_probe(probe, statistic)?;
    if configuration.variant != expected_variant {
        return Err(invalid(
            family,
            format!(
                "resolved {:?}, expected {expected_variant:?}",
                configuration.variant
            ),
        ));
    }
    Ok(configuration)
}

pub fn validate_variant_discriminator(
    probe: &ModelProbe,
    expected_adm: Option<u64>,
    family: &str,
) -> Result<Sd2Layout, ModelFamilyError> {
    let layout = match probe.select_layout(SD2_LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => Sd2Layout::PrefixedNative,
        ModelStateLayout::Diffusers => Sd2Layout::Diffusers,
        ModelStateLayout::StandaloneNative => {
            return Err(invalid(family, "standalone-native state is unsupported"));
        }
    };
    let (input_key, context_key, adm_key) = match layout {
        Sd2Layout::PrefixedNative => (INPUT_KEYS[0], CONTEXT_KEYS[0], ADM_KEYS[0]),
        Sd2Layout::Diffusers => (INPUT_KEYS[1], CONTEXT_KEYS[1], ADM_KEYS[1]),
    };
    let input = required_shape(probe, input_key, 4, family)?;
    if input[0] != SD2_MODEL_CHANNELS
        || !SOURCE_INPUT_CHANNELS.contains(&input[1])
        || input[2..] != [3, 3]
    {
        return Err(invalid(
            family,
            format!("unsupported input tensor {input_key}={input:?}"),
        ));
    }
    if layout == Sd2Layout::Diffusers && input[1] != 4 {
        return Err(invalid(
            family,
            "the pinned Diffusers table admits four input channels",
        ));
    }
    let context = required_shape(probe, context_key, 2, family)?;
    if context[1] != SD2_CONTEXT_DIMENSION {
        return Err(invalid(
            family,
            format!("unsupported context tensor {context_key}={context:?}"),
        ));
    }
    match (expected_adm, probe.tensor_shapes.get(adm_key)) {
        (None, None) => {}
        (None, Some(shape)) => {
            return Err(invalid(
                family,
                format!("unexpected ADM tensor {adm_key}={shape:?}"),
            ));
        }
        (Some(expected), Some(shape))
            if shape.len() == 2 && !shape.contains(&0) && shape[1] == expected => {}
        (Some(expected), Some(shape)) => {
            return Err(invalid(
                family,
                format!("ADM tensor {adm_key}={shape:?}, expected dimension {expected}"),
            ));
        }
        (Some(expected), None) => {
            return Err(invalid(
                family,
                format!("missing ADM tensor {adm_key} with dimension {expected}"),
            ));
        }
    }
    Ok(layout)
}

fn required_shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    rank: usize,
    family: &str,
) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .ok_or_else(|| invalid(family, format!("missing {key}")))?;
    if shape.len() != rank || shape.contains(&0) {
        return Err(invalid(
            family,
            format!("{key} must have non-zero rank {rank}, got {shape:?}"),
        ));
    }
    Ok(shape)
}

fn invalid(family: &str, message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "{family} source configuration mismatch: {}",
        message.into()
    ))
}
