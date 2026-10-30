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

pub const MODEL_FAMILY_IDENTIFIER: &str = "StableAudio";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0132";
pub const MODEL_FAMILY_FIXTURE: &str = "stableaudio-comfy-model-0132";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 21;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "952c054410c664838498a30c78efa44e129c85aaf75dc7c0360ce9253bce89c4";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const MODEL_FAMILY_SIGMA_MAX: f64 = 500.0;
pub const MODEL_FAMILY_SIGMA_MIN: f64 = 0.03;
pub const SOURCE_ARCHITECTURE: &str = "model_base.StableAudio1";

const DENOISER_PREFIX: &str = "model.model.";
const ROTARY_FREQUENCY: &str = "model.model.transformer.rotary_pos_emb.inv_freq";
const GLOBAL_PROJECTION: &str = "model.model.to_global_embed.0.weight";
const TOKEN_PROJECTION: &str = "model.model.to_cond_embed.0.weight";
const TIMESTEP_PROJECTION: &str = "model.model.to_timestep_embed.0.weight";
const POSTPROCESS_CONVOLUTION: &str = "model.model.postprocess_conv.weight";
const TRANSFORMER_INPUT: &str = "model.model.transformer.project_in.weight";
const SECONDS_START_WEIGHT: &str =
    "conditioner.conditioners.seconds_start.embedder.embedding.0.weights";
const SECONDS_TOTAL_WEIGHT: &str =
    "conditioner.conditioners.seconds_total.embedder.embedding.0.weights";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableAudioAttentionConfiguration {
    Plain,
    LayerNormalized { feature_scale: bool },
    RmsNormalized { differential: bool, heads: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableAudioTimestepFeatures {
    Learned,
    Exponential,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StableAudioConfiguration {
    pub global_condition_dimension: u64,
    pub project_condition_tokens: bool,
    pub embedding_dimension: u64,
    pub memory_tokens: Option<u64>,
    pub attention: StableAudioAttentionConfiguration,
    pub timestep_features: StableAudioTimestepFeatures,
    pub io_channels: u64,
    pub input_concat_dimension: u64,
    pub local_add_condition_dimension: Option<u64>,
    pub depth: usize,
}

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.sa_t5.SAT5Tokenizer",
        clip_model: "sim.text_encoders.sa_t5.SAT5Model",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "continuous audio diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "seconds_start_conditioner",
        role: "bounded start-time number conditioner",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "seconds_total_conditioner",
        role: "bounded duration number conditioner",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SAT5 text conditioning",
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
        score: 500,
    },
    ModelDetectionRule::KeyPresent {
        key: SECONDS_START_WEIGHT,
        score: 250,
    },
    ModelDetectionRule::KeyPresent {
        key: SECONDS_TOTAL_WEIGHT,
        score: 250,
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
        checkpoint: "seconds_conditioning_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.to_global_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "cross_attention_conditioning",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "timestep_conditioning_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.to_cond_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "audio_transformer_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "audio_output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.to_timestep_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "audio_latent_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "stable-audio-dit-1.0-v1",
    latent_feature_id: "COMFY-MODEL-0050",
    latent_identifier: "StableAudio1",
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
        bytes_per_parameter: 4,
        activation_bytes_per_element: 4,
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
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.conditioners.seconds_start."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.conditioners.seconds_start.","to":""}},"component":"seconds_start_conditioner"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.conditioners.seconds_total."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.conditioners.seconds_total.","to":""}},"component":"seconds_total_conditioner"}},
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
        SECONDS_START_WEIGHT,
        SECONDS_TOTAL_WEIGHT,
    ],
    required_prefixes: &[],
}];

const REQUIRED_STATE_KEYS: &[&str] = &[
    ROTARY_FREQUENCY,
    GLOBAL_PROJECTION,
    TOKEN_PROJECTION,
    TIMESTEP_PROJECTION,
    POSTPROCESS_CONVOLUTION,
    TRANSFORMER_INPUT,
    SECONDS_START_WEIGHT,
    SECONDS_TOTAL_WEIGHT,
];

const CONDITIONER_REQUIRED_KEYS: &[&str] = &["embedder.embedding.0.weights"];
const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "seconds_start_conditioner",
        required_keys: CONDITIONER_REQUIRED_KEYS,
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "seconds_total_conditioner",
        required_keys: CONDITIONER_REQUIRED_KEYS,
        optional_keys: &[],
        allow_unexpected: true,
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
    source_ordinal: 21,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: REQUIRED_STATE_KEYS,
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
) -> Result<StableAudioConfiguration, ModelFamilyError> {
    if probe
        .metadata()
        .get("model_layout")
        .is_some_and(|layout| layout == "diffusers")
        || probe.tensor_shapes().contains_key("conv_in.weight")
    {
        return Err(invalid_configuration(
            "the pinned source Diffusers detector table has no StableAudio entry",
        ));
    }
    match probe.select_layout(LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => {}
        layout => {
            return Err(invalid_configuration(format!(
                "unsupported {layout:?} state-dictionary layout"
            )));
        }
    }
    if probe
        .tensor_shapes()
        .contains_key("model.model.transformer.global_cond_embedder.0.weight")
        || probe
            .tensor_shapes()
            .contains_key("conditioner.conditioners.prompt.padding_embedding")
    {
        return Err(invalid_configuration(
            "StableAudio3 shared global conditioning cannot be claimed as StableAudio",
        ));
    }

    let global_projection = dimensions(probe, GLOBAL_PROJECTION)?;
    require_rank(global_projection, 2, "global conditioning projection")?;
    let global_condition_dimension = global_projection[1];

    let token_projection = dimensions(probe, TOKEN_PROJECTION)?;
    require_rank(token_projection, 2, "token conditioning projection")?;
    let project_condition_tokens = token_projection[0] != token_projection[1];

    let timestep_projection = dimensions(probe, TIMESTEP_PROJECTION)?;
    require_rank(timestep_projection, 2, "timestep projection")?;
    let embedding_dimension = timestep_projection[0];

    let memory_tokens = optional_dimension(
        probe,
        "model.model.transformer.memory_tokens",
        0,
    )?;
    let query_key = "model.model.transformer.layers.0.self_attn.to_qkv.weight";
    let query_shape = dimensions(probe, query_key)?;
    require_rank(query_shape, 2, "first attention query/key/value projection")?;
    let differential = query_shape[0]
        == query_shape[1]
            .checked_mul(5)
            .ok_or_else(|| invalid_configuration("attention dimension overflow"))?;

    let layer_norm = probe
        .tensor_shapes()
        .contains_key("model.model.transformer.layers.0.self_attn.q_norm.weight");
    let rms_key = "model.model.transformer.layers.0.self_attn.q_norm.gamma";
    let rms_norm = probe.tensor_shapes().contains_key(rms_key);
    let attention = match (layer_norm, rms_norm) {
        (true, false) => StableAudioAttentionConfiguration::LayerNormalized {
            feature_scale: true,
        },
        (false, true) => {
            let rms_shape = dimensions(probe, rms_key)?;
            require_rank(rms_shape, 1, "RMS query normalization")?;
            let head_dimension = rms_shape[0];
            if head_dimension == 0 || embedding_dimension % head_dimension != 0 {
                return Err(invalid_configuration(
                    "embedding dimension is not divisible by RMS head dimension",
                ));
            }
            StableAudioAttentionConfiguration::RmsNormalized {
                differential,
                heads: embedding_dimension / head_dimension,
            }
        }
        (false, false) => StableAudioAttentionConfiguration::Plain,
        (true, true) => {
            return Err(invalid_configuration(
                "both layer-normalized and RMS-normalized attention markers are present",
            ));
        }
    };

    let timestep_features = if probe
        .tensor_shapes()
        .contains_key("model.model.timestep_features.weight")
    {
        StableAudioTimestepFeatures::Learned
    } else {
        StableAudioTimestepFeatures::Exponential
    };

    let postprocess = dimensions(probe, POSTPROCESS_CONVOLUTION)?;
    require_rank(postprocess, 3, "postprocess convolution")?;
    let io_channels = postprocess[0];
    let transformer_input = dimensions(probe, TRANSFORMER_INPUT)?;
    require_rank(transformer_input, 2, "transformer input projection")?;
    let input_concat_dimension = transformer_input[1]
        .checked_sub(io_channels)
        .ok_or_else(|| invalid_configuration("transformer input is narrower than audio I/O"))?;
    let local_add_condition_dimension = optional_dimension(
        probe,
        "model.model.transformer.layers.0.to_local_embed.0.weight",
        1,
    )?;
    let depth = probe.consecutive_block_count("model.model.transformer.layers.{}.")?;
    if global_condition_dimension == 0
        || embedding_dimension == 0
        || io_channels == 0
        || depth == 0
    {
        return Err(invalid_configuration(
            "zero-sized global, embedding, I/O, or transformer depth fact",
        ));
    }

    Ok(StableAudioConfiguration {
        global_condition_dimension,
        project_condition_tokens,
        embedding_dimension,
        memory_tokens,
        attention,
        timestep_features,
        io_channels,
        input_concat_dimension,
        local_add_condition_dimension,
        depth,
    })
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
                .ok_or_else(|| invalid_configuration(format!("invalid {key} dimension")))
        })
        .transpose()
}

fn dimensions<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes()
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}

fn require_rank(shape: &[u64], rank: usize, name: &str) -> Result<(), ModelFamilyError> {
    if shape.len() != rank || shape.contains(&0) {
        return Err(invalid_configuration(format!(
            "{name} must have non-zero rank {rank}"
        )));
    }
    Ok(())
}

fn invalid_configuration(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "StableAudio source configuration mismatch: {}",
        message.into()
    ))
}
