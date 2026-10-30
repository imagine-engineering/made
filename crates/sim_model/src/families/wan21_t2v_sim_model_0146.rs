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

pub const MODEL_FAMILY_IDENTIFIER: &str = "WAN21_T2V";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0146";
pub const MODEL_FAMILY_FIXTURE: &str = "wan21-t2v-comfy-model-0146";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 52;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "4204d7e004c37c0fc799dcd6cf6ae077e02d40bad2b776f2372ae62a3d384802";
pub const MODEL_FAMILY_SHIFT: f64 = 8.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.WAN21";

pub const MODEL_PREFIX: &str = "model.diffusion_model.";
pub const HEAD_MODULATION: &str = "model.diffusion_model.head.modulation";
pub const HEAD_WEIGHT: &str = "model.diffusion_model.head.head.weight";
pub const PATCH_WEIGHT: &str = "model.diffusion_model.patch_embedding.weight";
pub const FFN_WEIGHT: &str = "model.diffusion_model.blocks.0.ffn.0.weight";
pub const VACE_PATCH_WEIGHT: &str = "model.diffusion_model.vace_patch_embedding.weight";
pub const POSE_PATCH_WEIGHT: &str = "model.diffusion_model.patch_embedding_pose.weight";
pub const MASK_PATCH_WEIGHT: &str = "model.diffusion_model.patch_embedding_mask.weight";
pub const IMAGE_BIAS: &str = "model.diffusion_model.img_emb.proj.0.bias";
pub const CONTROL_WEIGHT: &str = "model.diffusion_model.control_adapter.conv.weight";
pub const CAUSAL_AUDIO_WEIGHT: &str =
    "model.diffusion_model.casual_audio_encoder.encoder.final_linear.weight";
pub const AUDIO_BIAS: &str =
    "model.diffusion_model.audio_proj.audio_proj_glob_1.layer.bias";
pub const FACE_ADAPTER_WEIGHT: &str =
    "model.diffusion_model.face_adapter.fuser_blocks.0.k_norm.weight";
pub const GLOBAL_PATCH_WEIGHT: &str = "model.diffusion_model.patch_embedding_global.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wan21BatchVariant {
    Scail,
    Scail2,
    T2V,
    Vace,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wan21BatchConfiguration {
    pub variant: Wan21BatchVariant,
    pub image_model: &'static str,
    pub model_type: &'static str,
    pub architecture_model_type: &'static str,
    pub image_to_video: bool,
    pub reference_conditioning: bool,
    pub pose_conditioning: bool,
    pub mask_conditioning: bool,
    pub vace_conditioning: bool,
    pub dimension: u64,
    pub input_channels: u64,
    pub output_channels: u64,
    pub attention_heads: u64,
    pub feed_forward_dimension: u64,
    pub layer_count: usize,
    pub auxiliary_input_channels: Option<u64>,
    pub auxiliary_layer_count: Option<usize>,
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
        role: "Wan 2.1 text-to-video transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "text-to-video mode and source sampling shift",
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
    "native.vace_patch_embedding.weight",
    "native.vace_blocks.0.ffn.0.weight",
    "native.patch_embedding_pose.weight",
    "native.patch_embedding_mask.weight",
    "native.img_emb.proj.0.bias",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "t2v_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "t2v_text_conditioning",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "t2v_transformer_block",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.ffn.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "t2v_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "t2v_head_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_projection.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "t2v_video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "wan21-t2v-v1",
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

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":8.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"text_to_video"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"image_to_video"}}}
        ],
        "unmatched":"Reject"
    }"#,
};
const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[ModelFamilyStatePlanCase {
    layout: ModelStateLayout::PrefixedNative,
    plan: &NATIVE_STATE_PLAN,
}];
pub const BASE_LAYOUT_KEYS: &[&str] = &[HEAD_MODULATION, HEAD_WEIGHT, PATCH_WEIGHT, FFN_WEIGHT];
const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[ModelLayoutSignature {
    layout: ModelStateLayout::PrefixedNative,
    required_keys: BASE_LAYOUT_KEYS,
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
        required_keys: &["sampling_shift", "text_to_video", "image_to_video"],
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
    source_ordinal: 52,
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
) -> Result<Wan21BatchConfiguration, ModelFamilyError> {
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(invalid_configuration("unsupported state layout"));
    }
    batch_configuration_for_probe(probe, Wan21BatchVariant::T2V)
}

pub fn batch_configuration_for_probe(
    probe: &ModelProbe,
    variant: Wan21BatchVariant,
) -> Result<Wan21BatchConfiguration, ModelFamilyError> {
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
    validate_patch_shape(patch, *dimension, None, "patch_embedding.weight")?;
    let layer_count = probe.consecutive_block_count("model.diffusion_model.blocks.{}.")?;
    if layer_count == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }

    let has_vace = has(probe, VACE_PATCH_WEIGHT);
    let has_pose = has(probe, POSE_PATCH_WEIGHT);
    let has_mask = has(probe, MASK_PATCH_WEIGHT);
    let has_image = has(probe, IMAGE_BIAS);
    let has_other_adapter = [
        CONTROL_WEIGHT,
        CAUSAL_AUDIO_WEIGHT,
        AUDIO_BIAS,
        FACE_ADAPTER_WEIGHT,
        GLOBAL_PATCH_WEIGHT,
    ]
    .iter()
    .any(|key| has(probe, key));

    let (
        model_type,
        architecture_model_type,
        reference_conditioning,
        pose_conditioning,
        mask_conditioning,
        vace_conditioning,
        auxiliary_input_channels,
        auxiliary_layer_count,
        memory_multiplier,
    ) = match variant {
        Wan21BatchVariant::T2V => {
            if has_vace || has_pose || has_mask || has_image || has_other_adapter {
                return Err(invalid_configuration(
                    "T2V requires the absence of every source-specialized Wan adapter key",
                ));
            }
            ("t2v", "t2v", false, false, false, false, None, None, 1.0)
        }
        Wan21BatchVariant::Scail => {
            if !has_pose || has_mask || has_vace || has_other_adapter {
                return Err(invalid_configuration(
                    "SCAIL requires only the pose patch embedding specialization",
                ));
            }
            validate_patch_shape(
                shape(probe, POSE_PATCH_WEIGHT)?,
                *dimension,
                Some(patch[1]),
                "patch_embedding_pose.weight",
            )?;
            ("scail", "i2v", true, true, false, false, None, None, 1.0)
        }
        Wan21BatchVariant::Scail2 => {
            if !has_pose || !has_mask || has_vace || has_other_adapter {
                return Err(invalid_configuration(
                    "SCAIL2 requires pose and mask patch embeddings without another adapter",
                ));
            }
            validate_patch_shape(
                shape(probe, POSE_PATCH_WEIGHT)?,
                *dimension,
                Some(patch[1]),
                "patch_embedding_pose.weight",
            )?;
            validate_patch_shape(
                shape(probe, MASK_PATCH_WEIGHT)?,
                *dimension,
                Some(28),
                "patch_embedding_mask.weight",
            )?;
            ("scail2", "i2v", true, true, true, false, Some(28), None, 1.0)
        }
        Wan21BatchVariant::Vace => {
            if !has_vace || has_pose || has_mask || has_image || has_other_adapter {
                return Err(invalid_configuration(
                    "Vace requires only the Vace patch/block specialization",
                ));
            }
            let vace_patch = shape(probe, VACE_PATCH_WEIGHT)?;
            validate_patch_shape(vace_patch, *dimension, None, "vace_patch_embedding.weight")?;
            let vace_layers = probe.consecutive_block_count("model.diffusion_model.vace_blocks.{}.")?;
            if vace_layers == 0 || vace_layers > layer_count || layer_count % vace_layers != 0 {
                return Err(invalid_configuration(
                    "Vace layer count must be nonzero and divide the base transformer layer count",
                ));
            }
            (
                "vace",
                "t2v",
                false,
                false,
                false,
                true,
                Some(vace_patch[1]),
                Some(vace_layers),
                1.2,
            )
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
        auxiliary_input_channels,
        auxiliary_layer_count,
    )?;
    let memory_usage_factor = memory_usage_factor(*dimension)? * memory_multiplier;
    if !memory_usage_factor.is_finite() {
        return Err(invalid_configuration("memory usage factor must be finite"));
    }

    Ok(Wan21BatchConfiguration {
        variant,
        image_model: "wan2.1",
        model_type,
        architecture_model_type,
        image_to_video: false,
        reference_conditioning,
        pose_conditioning,
        mask_conditioning,
        vace_conditioning,
        dimension: *dimension,
        input_channels: patch[1],
        output_channels: head[0] / 4,
        attention_heads: dimension / 128,
        feed_forward_dimension: ffn[0],
        layer_count,
        auxiliary_input_channels,
        auxiliary_layer_count,
        patch_size: [1, 2, 2],
        frequency_dimension: 256,
        qk_norm: true,
        cross_attention_norm: true,
        epsilon_millionths: 1,
        memory_usage_factor,
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
    auxiliary_input_channels: Option<u64>,
    auxiliary_layer_count: Option<usize>,
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
    check_u64_fact(transformer, "num_layers", checked_usize(layer_count)?)?;
    check_u64_fact(transformer, "in_dim", input_channels)?;
    if let Some(value) = auxiliary_input_channels {
        let key = if model_type == "vace" { "vace_in_dim" } else { "mask_in_dim" };
        check_u64_fact(transformer, key, value)?;
    }
    if let Some(value) = auxiliary_layer_count {
        check_u64_fact(transformer, "vace_layers", checked_usize(value)?)?;
    }
    Ok(())
}

fn validate_patch_shape(
    value: &[u64],
    dimension: u64,
    expected_input_channels: Option<u64>,
    label: &str,
) -> Result<(), ModelFamilyError> {
    if value.len() != 5
        || value[0] != dimension
        || value[1] == 0
        || expected_input_channels.is_some_and(|expected| value[1] != expected)
        || value[2..] != [1, 2, 2]
    {
        return Err(invalid_configuration(format!("{label} shape")));
    }
    Ok(())
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

fn checked_usize(value: usize) -> Result<u64, ModelFamilyError> {
    u64::try_from(value).map_err(|_| invalid_configuration("layer count overflow"))
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

fn has(probe: &ModelProbe, key: &str) -> bool {
    probe.tensor_shapes().contains_key(key)
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
        "Wan 2.1 batch configuration is invalid: {}",
        message.into()
    ))
}
