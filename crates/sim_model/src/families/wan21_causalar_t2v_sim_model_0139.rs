use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelSourceConfigurationRule,
    ModelStateLayout, ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "WAN21_CausalAR_T2V";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0139";
pub const MODEL_FAMILY_FIXTURE: &str = "wan21-causalar-t2v-comfy-model-0139";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 51;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "fd7a6e48e53e67c38b606c445dba881acece5f07678c3e40f00f18a59b662a3b";
pub const MODEL_FAMILY_SHIFT: f64 = 5.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.WAN21_CausalAR";
pub const CAUSAL_CONFIG_METADATA: &str = r#"{"transformer":{"causal_ar":true}}"#;

const MODEL_PREFIX: &str = "model.diffusion_model.";
const HEAD_MODULATION: &str = "model.diffusion_model.head.modulation";
const HEAD_WEIGHT: &str = "model.diffusion_model.head.head.weight";
const PATCH_WEIGHT: &str = "model.diffusion_model.patch_embedding.weight";
const FFN_WEIGHT: &str = "model.diffusion_model.blocks.0.ffn.0.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wan21Variant {
    Camera,
    CausalArT2V,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wan21Configuration {
    pub variant: Wan21Variant,
    pub image_model: &'static str,
    pub model_type: &'static str,
    pub causal_ar: bool,
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
    pub memory_usage_factor: f64,
}

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];
const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.wan.WanT5Tokenizer",
        clip_model: "sim.text_encoders.wan.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];
pub const WAN21_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Wan 2.1 causal autoregressive text-to-video transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "runtime_conditioning",
        role: "causal autoregressive mode and sampling shift",
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
        value: CAUSAL_CONFIG_METADATA,
        score: 500,
    },
];
const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: MODEL_PREFIX,
    target_prefix: "native.",
    required: true,
}];
const REQUIRED_KEYS: &[&str] = &[
    "native.time_embedding.0.weight",
    "native.blocks.0.ffn.2.weight",
    "native.time_projection.1.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.head.head.weight",
    "native.head.modulation",
    "native.patch_embedding.weight",
    "native.blocks.0.ffn.0.weight",
    "native.blocks.0.self_attn.q.weight",
    "native.blocks.0.self_attn.k.weight",
    "native.blocks.0.self_attn.v.weight",
];
pub const WAN21_SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const WAN21_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "causal_wan_timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embedding.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "causal_wan_timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "causal_kv_cached_transformer_block",
        operation: ModelForwardOperation::Linear {
            weight: "native.blocks.0.ffn.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "causal_wan_block_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "causal_wan_head_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_projection.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "causal_wan_video_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "wan21-causal-ar-t2v-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
    clip_target: &WAN21_CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
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
        value: CAUSAL_CONFIG_METADATA,
    }];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"model."}},"component":"vae"}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":5.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"sampling_shift"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"runtime_conditioning","key":"causal_ar"}}}
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
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "runtime_conditioning",
        required_keys: &["sampling_shift", "causal_ar"],
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
    source_ordinal: 51,
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

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<Wan21Configuration, ModelFamilyError> {
    if probe.metadata().get("config").map(String::as_str) != Some(CAUSAL_CONFIG_METADATA) {
        return Err(invalid_configuration(
            "source config metadata must explicitly declare transformer.causal_ar=true",
        ));
    }
    if probe.select_layout(LAYOUT_SIGNATURES)? != ModelStateLayout::PrefixedNative {
        return Err(invalid_configuration("unsupported state layout"));
    }
    wan21_configuration_for_probe(probe, Wan21Variant::CausalArT2V)
}

pub fn wan21_configuration_for_probe(
    probe: &ModelProbe,
    variant: Wan21Variant,
) -> Result<Wan21Configuration, ModelFamilyError> {
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
        || patch[2..] != [1, 2, 2]
        || patch[1] == 0
    {
        return Err(invalid_configuration("patch_embedding.weight shape"));
    }
    let layer_count = probe.consecutive_block_count("model.diffusion_model.blocks.{}.")?;
    if layer_count == 0 {
        return Err(invalid_configuration("transformer layer count"));
    }
    let (model_type, causal_ar, expected_input, camera_condition_channels) = match variant {
        Wan21Variant::Camera => {
            let control = shape(probe, "model.diffusion_model.control_adapter.conv.weight")?;
            if control.len() != 4
                || control[0] != *dimension
                || control[1] == 0
                || control[1] % 64 != 0
                || control[2..] != [2, 2]
            {
                return Err(invalid_configuration("camera control adapter shape"));
            }
            let image_bias = shape(probe, "model.diffusion_model.img_emb.proj.0.bias")?;
            if image_bias != [*dimension] {
                return Err(invalid_configuration("camera image embedder shape"));
            }
            ("camera", false, 32, Some(control[1] / 64))
        }
        Wan21Variant::CausalArT2V => {
            if probe
                .tensor_shapes()
                .contains_key("model.diffusion_model.control_adapter.conv.weight")
            {
                return Err(invalid_configuration(
                    "causal T2V must not contain the camera control adapter",
                ));
            }
            ("t2v", true, 16, None)
        }
    };
    if patch[1] != expected_input {
        return Err(invalid_configuration(format!(
            "{model_type} input channels must be {expected_input}"
        )));
    }
    Ok(Wan21Configuration {
        variant,
        image_model: "wan2.1",
        model_type,
        causal_ar,
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
        camera_condition_channels,
        memory_usage_factor: memory_usage_factor(*dimension)?,
    })
}

pub fn memory_usage_factor(dimension: u64) -> Result<f64, ModelFamilyError> {
    if dimension == 0 {
        return Err(invalid_configuration("zero model dimension"));
    }
    let factor = dimension as f64 / 2_222.0;
    if !factor.is_finite() || factor <= 0.0 {
        return Err(invalid_configuration("invalid dynamic memory factor"));
    }
    Ok(factor)
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
        "Wan 2.1 configuration is invalid: {}",
        message.into()
    ))
}
