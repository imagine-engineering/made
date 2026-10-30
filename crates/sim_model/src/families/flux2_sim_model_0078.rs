use crate::{
    MemoryEstimatorDescriptor, ModelClipConfigurationFactDefinition,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanSelector,
    ModelProbe,
    flux_chroma_family::{
        FLUX_COMPONENT_STATE_SCHEMAS, FLUX_COMPONENTS, FLUX_FORWARD_PROGRAM,
        FLUX_INPUT_PROJECTION_KEYS, FLUX_LAYOUT_SIGNATURES, FLUX_MEMORY_USAGE_FACTOR,
        FLUX_MODEL_OPTIONAL_KEYS, FLUX_MODEL_REQUIRED_KEYS, FLUX_STATE_PLAN_CASES,
        FLUX_SUPPORTED_DEVICES, FLUX_SUPPORTED_DTYPES, FLUX_WEIGHT_RULES,
        FLUX2_DISCRIMINATOR_KEYS, FluxChromaConfiguration, FluxChromaVariant,
        configuration_for_probe as flux_chroma_configuration_for_probe,
    },
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "Flux2";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0078";
pub const MODEL_FAMILY_FIXTURE: &str = "flux2-comfy-model-0078";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 80;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "965c2ebf8580a67768c115a5b451dc844cbe917bce4f57adfc3fc8318200c144";
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 2.02;
pub const MODEL_FAMILY_INHERITED_MEMORY_USAGE_FACTOR: f64 = FLUX_MEMORY_USAGE_FACTOR;
pub const MODEL_FAMILY_MEMORY_DIMENSION_DIVISOR: f64 = 2_604.0;

const QWEN_4B_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.qwen3_4b",
    },
    ModelClipConfigurationFactDefinition::Bind {
        parameter: "model_type",
        source: "qwen3_4b",
    },
];
const QWEN_8B_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.qwen3_8b",
    },
    ModelClipConfigurationFactDefinition::Bind {
        parameter: "model_type",
        source: "qwen3_8b",
    },
];
const MISTRAL_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.mistral3_24b",
    }];
const MISTRAL_PRUNED_CONFIGURATION: &[ModelClipConfigurationFactDefinition] = &[
    ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect.mistral3_24b",
    },
    ModelClipConfigurationFactDefinition::Bind {
        parameter: "pruned",
        source: "true",
    },
];

const QWEN_4B_CANDIDATE: ModelClipTargetCandidateDefinition = ModelClipTargetCandidateDefinition {
    tokenizer: "sim.text_encoders.flux.KleinTokenizer",
    clip_model: "sim.text_encoders.flux.klein_te",
    invocation: ModelClipModelInvocationDefinition::Factory {
        configuration: QWEN_4B_CONFIGURATION,
    },
};
const QWEN_8B_CANDIDATE: ModelClipTargetCandidateDefinition = ModelClipTargetCandidateDefinition {
    tokenizer: "sim.text_encoders.flux.KleinTokenizer8B",
    clip_model: "sim.text_encoders.flux.klein_te",
    invocation: ModelClipModelInvocationDefinition::Factory {
        configuration: QWEN_8B_CONFIGURATION,
    },
};
const MISTRAL_CANDIDATE: ModelClipTargetCandidateDefinition = ModelClipTargetCandidateDefinition {
    tokenizer: "sim.text_encoders.flux.Flux2Tokenizer",
    clip_model: "sim.text_encoders.flux.flux2_te",
    invocation: ModelClipModelInvocationDefinition::Factory {
        configuration: MISTRAL_CONFIGURATION,
    },
};
const MISTRAL_PRUNED_CANDIDATE: ModelClipTargetCandidateDefinition =
    ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.flux.Flux2Tokenizer",
        clip_model: "sim.text_encoders.flux.flux2_te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: MISTRAL_PRUNED_CONFIGURATION,
        },
    };

const DYNAMIC_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[QWEN_4B_CANDIDATE, QWEN_8B_CANDIDATE, MISTRAL_CANDIDATE];
const QWEN_4B_CANDIDATES: &[ModelClipTargetCandidateDefinition] = &[QWEN_4B_CANDIDATE];
const QWEN_8B_CANDIDATES: &[ModelClipTargetCandidateDefinition] = &[QWEN_8B_CANDIDATE];
const MISTRAL_CANDIDATES: &[ModelClipTargetCandidateDefinition] = &[MISTRAL_CANDIDATE];
const MISTRAL_PRUNED_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[MISTRAL_PRUNED_CANDIDATE];
const NO_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] = &[];

const DYNAMIC_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: DYNAMIC_CANDIDATES,
    dynamic_selection: true,
};
const QWEN_4B_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: QWEN_4B_CANDIDATES,
    dynamic_selection: false,
};
const QWEN_8B_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: QWEN_8B_CANDIDATES,
    dynamic_selection: false,
};
const MISTRAL_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: MISTRAL_CANDIDATES,
    dynamic_selection: false,
};
const MISTRAL_PRUNED_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: MISTRAL_PRUNED_CANDIDATES,
    dynamic_selection: false,
};
const NO_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: NO_CLIP_CANDIDATES,
    dynamic_selection: false,
};

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyTensorDimensionValue {
        keys: FLUX_INPUT_PROJECTION_KEYS,
        dimension: 1,
        values: &[16],
        score: 600,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: FLUX2_DISCRIMINATOR_KEYS,
        score: 400,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "flux2-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0030",
    latent_identifier: "Flux2",
    clip_target: &DYNAMIC_CLIP_TARGET,
    components: FLUX_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: FLUX_WEIGHT_RULES,
    required_keys: FLUX_MODEL_REQUIRED_KEYS,
    optional_keys: FLUX_MODEL_OPTIONAL_KEYS,
    supported_dtypes: FLUX_SUPPORTED_DTYPES,
    supported_devices: FLUX_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 15,
        activation_bytes_per_element: 15,
    },
    forward_program: FLUX_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 80,
    source_architecture: "model_base.Flux2",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: FLUX_LAYOUT_SIGNATURES,
        cases: FLUX_STATE_PLAN_CASES,
    },
    component_state_schemas: FLUX_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    let clip_target = select_clip_target(probe);
    let bytes_per_element = memory_bytes_per_element(configuration.hidden_size)?;
    Ok(ModelFamilyProfile {
        latent_feature_id: MODEL_FAMILY.latent_feature_id,
        latent_identifier: MODEL_FAMILY.latent_identifier,
        clip_target,
        supported_dtypes: MODEL_FAMILY.supported_dtypes,
        supported_devices: MODEL_FAMILY.supported_devices,
        memory_estimator: MemoryEstimatorDescriptor {
            fixed_bytes: 0,
            bytes_per_parameter: bytes_per_element,
            activation_bytes_per_element: bytes_per_element,
        },
        forward_program: MODEL_FAMILY.forward_program,
    })
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<FluxChromaConfiguration, ModelFamilyError> {
    flux_chroma_configuration_for_probe(probe, FluxChromaVariant::Flux2, MODEL_FAMILY_IDENTIFIER)
}

pub fn memory_usage_factor(hidden_size: u64) -> f64 {
    MODEL_FAMILY_INHERITED_MEMORY_USAGE_FACTOR
        * (2.0 * 2.0)
        * (hidden_size as f64 / MODEL_FAMILY_MEMORY_DIMENSION_DIVISOR)
}

fn memory_bytes_per_element(hidden_size: u64) -> Result<u32, ModelFamilyError> {
    let factor = memory_usage_factor(hidden_size).ceil();
    if !factor.is_finite() || factor <= 0.0 || factor > f64::from(u32::MAX) {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "Flux2 memory usage factor is outside the supported range".to_owned(),
        ));
    }
    Ok(factor as u32)
}

fn select_clip_target(probe: &ModelProbe) -> &'static ModelClipTargetDefinition {
    if has_llama_detection(probe, "text_encoders.qwen3_4b.transformer.") {
        &QWEN_4B_CLIP_TARGET
    } else if has_llama_detection(probe, "text_encoders.qwen3_8b.transformer.") {
        &QWEN_8B_CLIP_TARGET
    } else if has_llama_detection(probe, "text_encoders.mistral3_24b.transformer.") {
        if probe.tensor_shapes().contains_key(
            "text_encoders.mistral3_24b.transformer.model.layers.39.post_attention_layernorm.weight",
        ) {
            &MISTRAL_CLIP_TARGET
        } else {
            &MISTRAL_PRUNED_CLIP_TARGET
        }
    } else {
        &NO_CLIP_TARGET
    }
}

fn has_llama_detection(probe: &ModelProbe, prefix: &str) -> bool {
    ["model.norm.weight", "model.layers.0.input_layernorm.weight"]
        .iter()
        .any(|suffix| {
            probe
                .tensor_shapes()
                .contains_key(&format!("{prefix}{suffix}"))
        })
        || probe
            .tensor_shapes()
            .keys()
            .any(|key| key.starts_with(prefix) && key.ends_with(".sim_quant"))
}
