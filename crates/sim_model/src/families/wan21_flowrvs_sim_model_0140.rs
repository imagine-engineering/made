use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
    generated_wan21_causalar_t2v_sim_model_0139::{
        WAN21_CLIP_TARGET, WAN21_SUPPORTED_DEVICES, WAN21_SUPPORTED_DTYPES, memory_usage_factor,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "WAN21_FlowRVS";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0140";
pub const MODEL_FAMILY_FIXTURE: &str = "wan21-flowrvs-comfy-model-0140";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 61;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "7820349df2b07dfb93d47ba470e4b3ae3b8ebb6bb0994b39df3659da91e33a59";
pub const MODEL_FAMILY_SHIFT: f64 = 8.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.WAN21_FlowRVS";
pub const FLOW_RVS_CONFIG_METADATA: &str =
    r#"{"transformer":{"model_type":"flow_rvs"}}"#;

pub const MODEL_PREFIX: &str = "model.diffusion_model.";
pub const HEAD_MODULATION: &str = "model.diffusion_model.head.modulation";
pub const HEAD_WEIGHT: &str = "model.diffusion_model.head.head.weight";
pub const PATCH_WEIGHT: &str = "model.diffusion_model.patch_embedding.weight";
pub const FFN_WEIGHT: &str = "model.diffusion_model.blocks.0.ffn.0.weight";
pub const IMAGE_BIAS: &str = "model.diffusion_model.img_emb.proj.0.bias";
pub const AUDIO_BIAS: &str =
    "model.diffusion_model.audio_proj.audio_proj_glob_1.layer.bias";
pub const CONTROL_WEIGHT: &str = "model.diffusion_model.control_adapter.conv.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wan21ExtendedVariant {
    FlowRvs,
    FunControl2V,
    HuMo,
    I2V,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wan21ExtendedConfiguration {
    pub variant: Wan21ExtendedVariant,
    pub image_model: &'static str,
    pub model_type: &'static str,
    pub architecture_model_type: &'static str,
    pub image_to_video: bool,
    pub audio_conditioning: bool,
    pub dimension: u64,
    pub input_channels: u64,
    pub output_channels: u64,
    pub attention_heads: u64,
    pub feed_forward_dimension: u64,
    pub layer_count: usize,
    pub patch_size: [u64; 3],
    pub frequency_dimension: u64,
    pub qk_norm: bool,
    pub cross_attention_norm: bool,
    pub epsilon_millionths: u64,
    pub memory_usage_factor: f64,
}

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Wan 2.1 reverse-flow video transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "image-to-video reverse-flow mode and sampling shift",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Wan UMT5-XXL text conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Wan 2.1 video latent codec",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: HEAD_MODULATION,
        score: 300,
    },
    ModelDetectionRule::KeyPresent {
        key: PATCH_WEIGHT,
        score: 200,
    },
    ModelDetectionRule::Metadata {
        key: "config",
        value: FLOW_RVS_CONFIG_METADATA,
        score: 500,
    },
];
pub const WAN21_WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: MODEL_PREFIX,
    target_prefix: "native.",
    required: true,
}];
pub const WAN21_REQUIRED_KEYS: &[&str] = &[
    "native.time_embedding.0.weight",
    "native.blocks.0.ffn.2.weight",
    "native.time_projection.1.weight",
];
pub const WAN21_OPTIONAL_KEYS: &[&str] = &[
    "native.head.head.weight",
    "native.head.modulation",
    "native.patch_embedding.weight",
    "native.blocks.0.ffn.0.weight",
    "native.img_emb.proj.0.bias",
    "native.audio_proj.audio_proj_glob_1.layer.bias",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "flow_rvs_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "flow_rvs_conditioning_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "flow_rvs_transformer_block",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.ffn.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "flow_rvs_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "flow_rvs_head_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_projection.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "flow_rvs_video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "wan21-flow-rvs-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
    clip_target: &WAN21_CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WAN21_WEIGHT_RULES,
    required_keys: WAN21_REQUIRED_KEYS,
    optional_keys: WAN21_OPTIONAL_KEYS,
    supported_dtypes: WAN21_SUPPORTED_DTYPES,
    supported_devices: WAN21_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 1,
        activation_bytes_per_element: 1,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] =
    &[ModelSourceConfigurationRule::Metadata {
        key: "config",
        value: FLOW_RVS_CONFIG_METADATA,
    }];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":8.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"image_to_video"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"reverse_flow"}}}
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
    required_keys: &[HEAD_MODULATION, HEAD_WEIGHT, PATCH_WEIGHT, FFN_WEIGHT],
    required_prefixes: &[],
}];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: WAN21_REQUIRED_KEYS,
        optional_keys: WAN21_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &["sampling_shift", "image_to_video", "reverse_flow"],
        optional_keys: &[],
        allow_unexpected: false,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
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
    source_ordinal: 61,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&WAN21_CLIP_TARGET),
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
) -> Result<Wan21ExtendedConfiguration, ModelFamilyError> {
    if probe.metadata().get("config").map(String::as_str) != Some(FLOW_RVS_CONFIG_METADATA) {
        return Err(invalid_configuration(
            "source config metadata must explicitly declare transformer.model_type=flow_rvs",
        ));
    }
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(invalid_configuration("unsupported state layout"));
    }
    extended_configuration_for_probe(probe, Wan21ExtendedVariant::FlowRvs)
}

pub fn extended_configuration_for_probe(
    probe: &ModelProbe,
    variant: Wan21ExtendedVariant,
) -> Result<Wan21ExtendedConfiguration, ModelFamilyError> {
    reject_diffusers(probe)?;
    let modulation = shape(probe, HEAD_MODULATION)?;
    let [one, two, dimension] = modulation else {
        return Err(invalid_configuration("head.modulation must be rank three"));
    };
    if *one != 1 || *two != 2 || *dimension == 0 || dimension % 128 != 0 {
        return Err(invalid_configuration(
            "head.modulation must be [1,2,dim] with dim divisible by 128",
        ));
    }
    let head = shape(probe, HEAD_WEIGHT)?;
    if head.len() != 2 || head[1] != *dimension || head[0] == 0 || head[0] % 4 != 0 {
        return Err(invalid_configuration("head.head.weight shape"));
    }
    let ffn = shape(probe, FFN_WEIGHT)?;
    if ffn.len() != 2 || ffn[1] != *dimension || ffn[0] == 0 {
        return Err(invalid_configuration("blocks.0.ffn.0.weight shape"));
    }
    let patch = shape(probe, PATCH_WEIGHT)?;
    if patch.len() != 5
        || patch[0] != *dimension
        || patch[1] == 0
        || patch[2..] != [1, 2, 2]
    {
        return Err(invalid_configuration("patch_embedding.weight shape"));
    }
    let layer_count = probe.consecutive_block_count("model.diffusion_model.blocks.{}.")?;
    if layer_count == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }

    let has_image_bias = probe.tensor_shapes().contains_key(IMAGE_BIAS);
    let has_audio_bias = probe.tensor_shapes().contains_key(AUDIO_BIAS);
    let has_control = probe.tensor_shapes().contains_key(CONTROL_WEIGHT);
    let (model_type, architecture_model_type, image_to_video, audio_conditioning) = match variant {
        Wan21ExtendedVariant::FlowRvs => {
            if probe.metadata().get("config").map(String::as_str)
                != Some(FLOW_RVS_CONFIG_METADATA)
            {
                return Err(invalid_configuration("flow_rvs checkpoint configuration"));
            }
            ("flow_rvs", "t2v", true, false)
        }
        Wan21ExtendedVariant::FunControl2V => {
            if patch[1] != 48 || !has_image_bias || has_audio_bias || has_control {
                return Err(invalid_configuration(
                    "FunControl2V requires in_dim=48, the image embedder, and no specialized adapter",
                ));
            }
            ("i2v", "i2v", false, false)
        }
        Wan21ExtendedVariant::HuMo => {
            if !has_audio_bias || has_control {
                return Err(invalid_configuration(
                    "HuMo requires the audio projection and no camera adapter",
                ));
            }
            if shape(probe, AUDIO_BIAS)? != [*dimension] {
                return Err(invalid_configuration(
                    "HuMo audio projection bias must match the transformer dimension",
                ));
            }
            ("humo", "humo", false, true)
        }
        Wan21ExtendedVariant::I2V => {
            if patch[1] != 36 || !has_image_bias || has_audio_bias || has_control {
                return Err(invalid_configuration(
                    "I2V requires in_dim=36, the image embedder, and no specialized adapter",
                ));
            }
            ("i2v", "i2v", true, false)
        }
    };
    validate_transformer_metadata(
        probe,
        model_type,
        *dimension,
        head[0] / 4,
        ffn[0],
        layer_count,
        patch[1],
    )?;

    Ok(Wan21ExtendedConfiguration {
        variant,
        image_model: "wan2.1",
        model_type,
        architecture_model_type,
        image_to_video,
        audio_conditioning,
        dimension: *dimension,
        input_channels: patch[1],
        output_channels: head[0] / 4,
        attention_heads: dimension / 128,
        feed_forward_dimension: ffn[0],
        layer_count,
        patch_size: [1, 2, 2],
        frequency_dimension: 256,
        qk_norm: true,
        cross_attention_norm: true,
        epsilon_millionths: 1,
        memory_usage_factor: memory_usage_factor(*dimension)?,
    })
}

fn validate_transformer_metadata(
    probe: &ModelProbe,
    model_type: &str,
    dimension: u64,
    output_channels: u64,
    feed_forward_dimension: u64,
    layer_count: usize,
    input_channels: u64,
) -> Result<(), ModelFamilyError> {
    let Some(raw) = probe.metadata().get("config") else {
        return Ok(());
    };
    let document: serde_json::Value = serde_json::from_str(raw)
        .map_err(|_| invalid_configuration("config metadata must be valid JSON"))?;
    let root = document
        .as_object()
        .ok_or_else(|| invalid_configuration("config metadata must be a JSON object"))?;
    let Some(transformer) = root.get("transformer") else {
        return Ok(());
    };
    let transformer = transformer
        .as_object()
        .ok_or_else(|| invalid_configuration("config.transformer must be a JSON object"))?;
    check_string_fact(transformer, "image_model", "wan2.1")?;
    check_string_fact(transformer, "model_type", model_type)?;
    check_u64_fact(transformer, "dim", dimension)?;
    check_u64_fact(transformer, "out_dim", output_channels)?;
    check_u64_fact(transformer, "num_heads", dimension / 128)?;
    check_u64_fact(transformer, "ffn_dim", feed_forward_dimension)?;
    check_u64_fact(
        transformer,
        "num_layers",
        u64::try_from(layer_count)
            .map_err(|_| invalid_configuration("transformer layer count overflow"))?,
    )?;
    check_u64_fact(transformer, "in_dim", input_channels)
}

fn check_string_fact(
    transformer: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    expected: &str,
) -> Result<(), ModelFamilyError> {
    let Some(value) = transformer.get(key) else {
        return Ok(());
    };
    if value.as_str() != Some(expected) {
        return Err(invalid_configuration(format!(
            "config.transformer.{key} conflicts with key-derived value {expected}"
        )));
    }
    Ok(())
}

fn check_u64_fact(
    transformer: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    expected: u64,
) -> Result<(), ModelFamilyError> {
    let Some(value) = transformer.get(key) else {
        return Ok(());
    };
    if value.as_u64() != Some(expected) {
        return Err(invalid_configuration(format!(
            "config.transformer.{key} conflicts with key-derived value {expected}"
        )));
    }
    Ok(())
}

fn reject_diffusers(probe: &ModelProbe) -> Result<(), ModelFamilyError> {
    if probe
        .format_identities()
        .iter()
        .any(|identity| identity.eq_ignore_ascii_case("diffusers"))
        || probe
            .metadata()
            .get("model_layout")
            .is_some_and(|layout| layout.eq_ignore_ascii_case("diffusers"))
    {
        return Err(invalid_configuration(
            "the pinned Diffusers detector table has no Wan 2.1 family row",
        ));
    }
    Ok(())
}

fn shape<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes()
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}

fn invalid_configuration(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "Wan 2.1 extended configuration is invalid: {}",
        message.into()
    ))
}
