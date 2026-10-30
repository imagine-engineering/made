use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe,
    ModelSourceConfigurationRule, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "StableAudio3";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0133";
pub const MODEL_FAMILY_FIXTURE: &str = "stableaudio3-comfy-model-0133";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 20;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "a5d654d17f204bb571145cdd2156765d53611b8f2bede7a8e7a6686ad3c1c273";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 7.0;
pub const MODEL_FAMILY_SHIFT: f64 = 2.0;
pub const MODEL_FAMILY_MULTIPLIER: f64 = 1.0;
pub const SOURCE_ARCHITECTURE: &str = "model_base.StableAudio3";

const DENOISER_PREFIX: &str = "model.model.";
const ROTARY_FREQUENCY: &str = "model.model.transformer.rotary_pos_emb.inv_freq";
const GLOBAL_PROJECTION: &str = "model.model.to_global_embed.0.weight";
const TOKEN_PROJECTION: &str = "model.model.to_cond_embed.0.weight";
const TIMESTEP_PROJECTION: &str = "model.model.to_timestep_embed.0.weight";
const POSTPROCESS_CONVOLUTION: &str = "model.model.postprocess_conv.weight";
const TRANSFORMER_INPUT: &str = "model.model.transformer.project_in.weight";
const SHARED_GLOBAL_PROJECTION: &str =
    "model.model.transformer.global_cond_embedder.0.weight";
const SECONDS_TOTAL_WEIGHT: &str =
    "conditioner.conditioners.seconds_total.embedder.embedding.0.weights";
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableAudio3AttentionConfiguration {
    Plain,
    LayerNormalized,
    RmsNormalized { differential: bool, heads: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StableAudio3Configuration {
    pub global_condition_dimension: u64,
    pub project_condition_tokens: bool,
    pub embedding_dimension: u64,
    pub memory_tokens: Option<u64>,
    pub attention: StableAudio3AttentionConfiguration,
    pub learned_timestep_features: bool,
    pub io_channels: u64,
    pub input_concat_dimension: u64,
    pub local_add_condition_dimension: Option<u64>,
    pub depth: usize,
    pub shared_global_embedding: bool,
    pub max_text_tokens: u64,
}

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.sa3.SAT5GemmaTokenizer",
        clip_model: "sim.text_encoders.sa3.SAT5GemmaModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "shared-global continuous audio diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "seconds_total_conditioner",
        role: "bounded duration number conditioner",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "prompt_conditioner",
        role: "optional cross-attention padding embedding",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SAT5 Gemma text conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "StableAudio pretransform codec",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: ROTARY_FREQUENCY,
        score: 300,
    },
    ModelDetectionRule::KeyPresent {
        key: SHARED_GLOBAL_PROJECTION,
        score: 400,
    },
    ModelDetectionRule::KeyPresent {
        key: SECONDS_TOTAL_WEIGHT,
        score: 300,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: DENOISER_PREFIX,
    target_prefix: "native.",
    required: true,
}];
const REQUIRED_KEYS: &[&str] = &[
    "native.to_global_embed.0.weight",
    "native.to_cond_embed.0.weight",
    "native.to_timestep_embed.0.weight",
    "native.transformer.global_cond_embedder.0.weight",
];
const OPTIONAL_KEYS: &[&str] = &[
    "native.transformer.rotary_pos_emb.inv_freq",
    "native.timestep_features.weight",
    "native.transformer.layers.0.self_attn.q_norm.weight",
    "native.transformer.layers.0.self_attn.q_norm.gamma",
    "native.transformer.layers.0.self_attn.to_qkv.weight",
    "native.transformer.memory_tokens",
    "native.transformer.project_in.weight",
    "native.transformer.layers.0.to_local_embed.0.weight",
    "native.postprocess_conv.weight",
];
const SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "duration_shared_global_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.to_global_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "padded_cross_attention_conditioning",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "audio_token_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.to_cond_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "shared_global_audio_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "audio_timestep_output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.to_timestep_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "stableaudio3_latent_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "stable-audio-3-dit-1.0-v1",
    latent_feature_id: "COMFY-MODEL-0051",
    latent_identifier: "StableAudio3",
    clip_target: &CLIP_TARGET,
    components: COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: SUPPORTED_DTYPES,
    supported_devices: SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 7,
        activation_bytes_per_element: 7,
    },
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];
const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"All":[{"Prefix":"model.model."},{"Any":[{"Suffix":".cross_attend_norm.beta"},{"Suffix":".ff_norm.beta"},{"Suffix":".pre_norm.beta"}]}]},"minimum_matches":0,"maximum_matches":16384}}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.model."},{"Not":{"Any":[{"Suffix":".cross_attend_norm.beta"},{"Suffix":".ff_norm.beta"},{"Suffix":".pre_norm.beta"}]}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.model.","to":"native."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.conditioners.seconds_total."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.conditioners.seconds_total.","to":""}},"component":"seconds_total_conditioner"}},
            {"Move":{"selector":{"predicate":{"Exact":"conditioner.conditioners.prompt.padding_embedding"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"padding_embedding"},"component":"prompt_conditioner"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"pretransform.model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"pretransform.model.","to":"model."}},"component":"vae"}}
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
        ROTARY_FREQUENCY,
        GLOBAL_PROJECTION,
        TOKEN_PROJECTION,
        TIMESTEP_PROJECTION,
        POSTPROCESS_CONVOLUTION,
        TRANSFORMER_INPUT,
        SHARED_GLOBAL_PROJECTION,
        SECONDS_TOTAL_WEIGHT,
    ],
    required_prefixes: &[],
}];
const CONDITIONER_REQUIRED_KEYS: &[&str] = &["embedder.embedding.0.weights"];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "seconds_total_conditioner",
        required_keys: CONDITIONER_REQUIRED_KEYS,
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "prompt_conditioner",
        required_keys: &[],
        optional_keys: &["padding_embedding"],
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
    source_ordinal: 20,
    source_architecture: "model_base.StableAudio3",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
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
) -> Result<StableAudio3Configuration, ModelFamilyError> {
    if probe
        .metadata()
        .get("model_layout")
        .is_some_and(|layout| layout == "diffusers")
        || probe.tensor_shapes().contains_key("conv_in.weight")
    {
        return Err(invalid("the pinned source Diffusers detector table has no StableAudio entry"));
    }
    probe.select_layout(LAYOUT_SIGNATURES)?;
    if probe.tensor_shapes().keys().any(|key| {
        key.starts_with("conditioner.conditioners.seconds_start.")
    }) {
        return Err(invalid("StableAudio1 seconds_start conditioning is not valid for StableAudio3"));
    }

    let global = shape(probe, GLOBAL_PROJECTION, 2, "global projection")?;
    let token = shape(probe, TOKEN_PROJECTION, 2, "token projection")?;
    let timestep = shape(probe, TIMESTEP_PROJECTION, 2, "timestep projection")?;
    let shared = shape(probe, SHARED_GLOBAL_PROJECTION, 2, "shared global projection")?;
    if shared[0] == 0 || shared[1] != global[0] {
        return Err(invalid("shared global projection is incompatible with global embedding"));
    }
    let embedding_dimension = timestep[0];
    let qkv = shape(
        probe,
        "model.model.transformer.layers.0.self_attn.to_qkv.weight",
        2,
        "query/key/value projection",
    )?;
    let differential = qkv[0] == qkv[1].checked_mul(5).ok_or_else(|| invalid("attention overflow"))?;
    let layer_norm = probe.tensor_shapes().contains_key(
        "model.model.transformer.layers.0.self_attn.q_norm.weight",
    );
    let rms_key = "model.model.transformer.layers.0.self_attn.q_norm.gamma";
    let rms_norm = probe.tensor_shapes().contains_key(rms_key);
    let attention = match (layer_norm, rms_norm) {
        (true, false) => StableAudio3AttentionConfiguration::LayerNormalized,
        (false, true) => {
            let rms = shape(probe, rms_key, 1, "RMS normalization")?;
            if embedding_dimension % rms[0] != 0 {
                return Err(invalid("embedding dimension is not divisible by RMS head dimension"));
            }
            StableAudio3AttentionConfiguration::RmsNormalized {
                differential,
                heads: embedding_dimension / rms[0],
            }
        }
        (false, false) => StableAudio3AttentionConfiguration::Plain,
        (true, true) => return Err(invalid("both layer and RMS attention normalization are present")),
    };
    let post = shape(probe, POSTPROCESS_CONVOLUTION, 3, "postprocess convolution")?;
    let input = shape(probe, TRANSFORMER_INPUT, 2, "transformer input projection")?;
    let input_concat_dimension = input[1]
        .checked_sub(post[0])
        .ok_or_else(|| invalid("transformer input is narrower than audio I/O"))?;
    let depth = probe.consecutive_block_count("model.model.transformer.layers.{}.")?;
    if depth == 0 {
        return Err(invalid("transformer depth is zero"));
    }
    Ok(StableAudio3Configuration {
        global_condition_dimension: global[1],
        project_condition_tokens: token[0] != token[1],
        embedding_dimension,
        memory_tokens: optional_dimension(probe, "model.model.transformer.memory_tokens", 0)?,
        attention,
        learned_timestep_features: probe
            .tensor_shapes()
            .contains_key("model.model.timestep_features.weight"),
        io_channels: post[0],
        input_concat_dimension,
        local_add_condition_dimension: optional_dimension(
            probe,
            "model.model.transformer.layers.0.to_local_embed.0.weight",
            1,
        )?,
        depth,
        shared_global_embedding: true,
        max_text_tokens: 256,
    })
}

fn shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    rank: usize,
    name: &str,
) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != rank || shape.contains(&0) {
        return Err(invalid(format!("{name} must have non-zero rank {rank}")));
    }
    Ok(shape)
}

fn optional_dimension(
    probe: &ModelProbe,
    key: &str,
    dimension: usize,
) -> Result<Option<u64>, ModelFamilyError> {
    probe
        .tensor_shapes()
        .get(key)
        .map(|shape| {
            shape
                .get(dimension)
                .copied()
                .filter(|value| *value != 0)
                .ok_or_else(|| invalid(format!("invalid {key} dimension")))
        })
        .transpose()
}

fn invalid(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "StableAudio3 source configuration mismatch: {}",
        message.into()
    ))
}
