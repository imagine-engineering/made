use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    generated_wan21_causalar_t2v_sim_model_0139::{
        WAN21_CLIP_TARGET, WAN21_SUPPORTED_DEVICES, WAN21_SUPPORTED_DTYPES, memory_usage_factor,
    },
    generated_wan21_t2v_sim_model_0146::{
        AUDIO_BIAS, CAUSAL_AUDIO_WEIGHT, CONTROL_WEIGHT, FACE_ADAPTER_WEIGHT,
        FFN_WEIGHT, GLOBAL_PATCH_WEIGHT, HEAD_MODULATION, HEAD_WEIGHT, IMAGE_BIAS,
        MASK_PATCH_WEIGHT, PATCH_WEIGHT, POSE_PATCH_WEIGHT, VACE_PATCH_WEIGHT,
        WAN21_OPTIONAL_KEYS, WAN21_REQUIRED_KEYS, WAN21_WEIGHT_RULES,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "WAN22_Animate";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0148";
pub const MODEL_FAMILY_FIXTURE: &str = "wan22-animate-comfy-model-0148";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 60;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "3e3d3b5586c44c06cdb1e51e184e0b8858afccc5239127f607f8850360211678";
pub const MODEL_FAMILY_SHIFT: f64 = 8.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.WAN22_Animate";
pub const ANIMATE_POSE_WEIGHT: &str = "model.diffusion_model.pose_patch_embedding.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wan22BatchVariant {
    Animate,
    Camera,
    S2V,
    T2V,
    WanDancer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wan22BatchConfiguration {
    pub variant: Wan22BatchVariant,
    pub image_model: &'static str,
    pub model_type: &'static str,
    pub architecture_model_type: &'static str,
    pub image_to_video: bool,
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
    pub camera_condition_channels: Option<u64>,
    pub face_conditioning: bool,
    pub pose_conditioning: bool,
    pub audio_conditioning: bool,
    pub reference_conditioning: bool,
    pub motion_conditioning: bool,
    pub control_conditioning: bool,
    pub music_conditioning: bool,
    pub memory_usage_factor: f64,
}

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Wan 2.2 Animate face/pose-conditioned video transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "face video, pose latent, and source sampling-shift contract",
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
        score: 200,
    },
    ModelDetectionRule::KeyPresent {
        key: PATCH_WEIGHT,
        score: 200,
    },
    ModelDetectionRule::KeyPresent {
        key: FACE_ADAPTER_WEIGHT,
        score: 800,
    },
];
pub const WAN22_OPTIONAL_KEYS: &[&str] = &[
    WAN21_OPTIONAL_KEYS[0],
    WAN21_OPTIONAL_KEYS[1],
    WAN21_OPTIONAL_KEYS[2],
    WAN21_OPTIONAL_KEYS[3],
    WAN21_OPTIONAL_KEYS[4],
    WAN21_OPTIONAL_KEYS[5],
    WAN21_OPTIONAL_KEYS[6],
    WAN21_OPTIONAL_KEYS[7],
    WAN21_OPTIONAL_KEYS[8],
    "native.control_adapter.conv.weight",
    "native.casual_audio_encoder.encoder.final_linear.weight",
    "native.audio_proj.audio_proj_glob_1.layer.bias",
    "native.face_adapter.fuser_blocks.0.k_norm.weight",
    "native.pose_patch_embedding.weight",
    "native.patch_embedding_global.weight",
    "native.music_encoder.0.self_attn.q_proj.weight",
    "native.music_encoder.0.self_attn.k_proj.weight",
    "native.music_encoder.0.self_attn.v_proj.weight",
    "native.music_encoder.0.self_attn.q_proj.bias",
    "native.music_encoder.0.self_attn.k_proj.bias",
    "native.music_encoder.0.self_attn.v_proj.bias",
];
const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "animate_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "animate_face_pose_conditioning",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "animate_transformer_block",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.ffn.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "animate_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "animate_head_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_projection.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "animate_video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "wan22-animate-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
    clip_target: &WAN21_CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WAN21_WEIGHT_RULES,
    required_keys: WAN21_REQUIRED_KEYS,
    optional_keys: WAN22_OPTIONAL_KEYS,
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
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"face_video"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"pose_latents"}}}
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
    required_keys: &[
        HEAD_MODULATION,
        HEAD_WEIGHT,
        PATCH_WEIGHT,
        FFN_WEIGHT,
        FACE_ADAPTER_WEIGHT,
    ],
    required_prefixes: &[],
}];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: WAN21_REQUIRED_KEYS,
        optional_keys: WAN22_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &["sampling_shift", "face_video", "pose_latents"],
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
    source_ordinal: 60,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[FACE_ADAPTER_WEIGHT],
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
) -> Result<Wan22BatchConfiguration, ModelFamilyError> {
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(invalid_configuration("unsupported state layout"));
    }
    batch_configuration_for_probe(probe, Wan22BatchVariant::Animate)
}

pub fn batch_configuration_for_probe(
    probe: &ModelProbe,
    variant: Wan22BatchVariant,
) -> Result<Wan22BatchConfiguration, ModelFamilyError> {
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

    let present = |key: &str| probe.tensor_shapes().contains_key(key);
    let specialized = [
        VACE_PATCH_WEIGHT,
        CONTROL_WEIGHT,
        CAUSAL_AUDIO_WEIGHT,
        AUDIO_BIAS,
        FACE_ADAPTER_WEIGHT,
        MASK_PATCH_WEIGHT,
        POSE_PATCH_WEIGHT,
        GLOBAL_PATCH_WEIGHT,
    ];
    let output_channels = head[0] / 4;
    let (
        model_type,
        architecture_model_type,
        image_to_video,
        expected_input,
        expected_output,
        camera_condition_channels,
        face_conditioning,
        pose_conditioning,
        audio_conditioning,
        reference_conditioning,
        motion_conditioning,
        control_conditioning,
        music_conditioning,
        fixed_memory,
    ) = match variant {
        Wan22BatchVariant::Animate => {
            require_only(probe, FACE_ADAPTER_WEIGHT, &specialized)?;
            if let Some(pose) = probe.tensor_shapes().get(ANIMATE_POSE_WEIGHT) {
                validate_patch_shape(pose, *dimension, Some(16), "pose_patch_embedding.weight")?;
            }
            (
                "animate", "i2v", false, None, Some(16), None, true, true, false, false,
                false, false, false, None,
            )
        }
        Wan22BatchVariant::Camera => {
            require_only(probe, CONTROL_WEIGHT, &specialized)?;
            if present(IMAGE_BIAS) {
                return Err(invalid_configuration(
                    "Camera 2.2 requires the source camera_2.2 discriminator without img_emb bias",
                ));
            }
            let control = shape(probe, CONTROL_WEIGHT)?;
            if control.len() != 4
                || control[0] != *dimension
                || control[1] != 24
                || control[2..] != [2, 2]
            {
                return Err(invalid_configuration("control_adapter.conv.weight shape"));
            }
            (
                "camera_2.2", "t2v", false, Some(36), Some(16), Some(24), false, false,
                false, false, false, true, false, None,
            )
        }
        Wan22BatchVariant::S2V => {
            require_only(probe, CAUSAL_AUDIO_WEIGHT, &specialized)?;
            let audio = shape(probe, CAUSAL_AUDIO_WEIGHT)?;
            if audio.len() != 2 || audio[0] != *dimension || audio[1] == 0 {
                return Err(invalid_configuration(
                    "casual_audio_encoder.encoder.final_linear.weight shape",
                ));
            }
            (
                "s2v", "t2v", false, None, Some(16), None, false, false, true, true, true,
                true, false, None,
            )
        }
        Wan22BatchVariant::T2V => {
            if specialized.iter().any(|key| present(key)) || present(IMAGE_BIAS) {
                return Err(invalid_configuration(
                    "Wan 2.2 T2V requires the absence of every specialized adapter key",
                ));
            }
            (
                "t2v", "t2v", true, Some(16), Some(48), None, false, false, false, false,
                false, false, false, None,
            )
        }
        Wan22BatchVariant::WanDancer => {
            require_only(probe, GLOBAL_PATCH_WEIGHT, &specialized)?;
            validate_patch_shape(
                shape(probe, GLOBAL_PATCH_WEIGHT)?,
                *dimension,
                Some(36),
                "patch_embedding_global.weight",
            )?;
            (
                "wandancer", "i2v", true, Some(36), Some(16), None, false, false, true,
                true, false, false, true, Some(1.8),
            )
        }
    };
    if expected_input.is_some_and(|expected| patch[1] != expected) {
        return Err(invalid_configuration("variant input-channel geometry"));
    }
    if expected_output.is_some_and(|expected| output_channels != expected) {
        return Err(invalid_configuration("variant output-channel geometry"));
    }
    validate_transformer_metadata(
        probe,
        model_type,
        *dimension,
        output_channels,
        ffn[0],
        layer_count,
        patch[1],
    )?;
    let memory_usage_factor = fixed_memory.unwrap_or(memory_usage_factor(*dimension)?);
    if !memory_usage_factor.is_finite() {
        return Err(invalid_configuration("memory usage factor must be finite"));
    }
    Ok(Wan22BatchConfiguration {
        variant,
        image_model: "wan2.1",
        model_type,
        architecture_model_type,
        image_to_video,
        dimension: *dimension,
        input_channels: patch[1],
        output_channels,
        attention_heads: dimension / 128,
        feed_forward_dimension: ffn[0],
        layer_count,
        patch_size: [1, 2, 2],
        frequency_dimension: 256,
        qk_norm: true,
        cross_attention_norm: true,
        epsilon_millionths: 1,
        camera_condition_channels,
        face_conditioning,
        pose_conditioning,
        audio_conditioning,
        reference_conditioning,
        motion_conditioning,
        control_conditioning,
        music_conditioning,
        memory_usage_factor,
    })
}

fn require_only(
    probe: &ModelProbe,
    required: &str,
    specialized: &[&str],
) -> Result<(), ModelFamilyError> {
    if !probe.tensor_shapes().contains_key(required)
        || specialized
            .iter()
            .any(|key| *key != required && probe.tensor_shapes().contains_key(*key))
    {
        return Err(invalid_configuration(format!(
            "variant requires only its {required} specialization"
        )));
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
    for (key, expected) in [
        ("dim", dimension),
        ("out_dim", output_channels),
        ("num_heads", dimension / 128),
        ("ffn_dim", feed_forward_dimension),
        ("num_layers", u64::try_from(layer_count).map_err(|_| invalid_configuration("layer count overflow"))?),
        ("in_dim", input_channels),
    ] {
        check_u64_fact(transformer, key, expected)?;
    }
    Ok(())
}

fn check_string_fact(
    transformer: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    expected: &str,
) -> Result<(), ModelFamilyError> {
    if transformer
        .get(key)
        .is_some_and(|value| value.as_str() != Some(expected))
    {
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
    if transformer
        .get(key)
        .is_some_and(|value| value.as_u64() != Some(expected))
    {
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
            "the pinned Diffusers detector table has no Wan 2.2 family row",
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
        "Wan 2.2 batch configuration is invalid: {}",
        message.into()
    ))
}
