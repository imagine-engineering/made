use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, ModelClipConfigurationFactDefinition,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelFamilyError, ModelFamilyStatePlanCase, ModelLayoutSignature,
    ModelProbe, ModelStateLayout, ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const COSMOS_GENERAL_DETECTION_MARKER_KEYS: &[&str] =
    &["net.blocks.block0.blocks.0.block.attn.to_q.0.weight"];
pub const COSMOS_PREDICT2_DETECTION_MARKER_KEYS: &[&str] = &["net.blocks.0.mlp.layer1.weight"];
pub const COSMOS_ANIMA_DETECTION_MARKER_KEYS: &[&str] =
    &["net.llm_adapter.blocks.0.cross_attn.q_proj.weight"];
pub const COSMOS_PATCH_PROJECTION_KEYS: &[&str] = &["net.x_embedder.proj.1.weight"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CosmosArchitecture {
    GeneralDit,
    Predict2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CosmosModelSize {
    Predict2TwoB,
    SevenB,
    FourteenB,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CosmosRatio {
    pub numerator: u64,
    pub denominator: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosmosConfiguration {
    pub architecture: CosmosArchitecture,
    pub in_channels: u64,
    pub out_channels: u64,
    pub model_channels: u64,
    pub number_of_blocks: usize,
    pub number_of_heads: u64,
    pub maximum_image_height: u64,
    pub maximum_image_width: u64,
    pub maximum_frames: u64,
    pub spatial_patch_size: u64,
    pub temporal_patch_size: u64,
    pub concatenate_padding_mask: bool,
    pub image_to_video: bool,
    pub positional_embeddings_learnable: bool,
    pub rope_extrapolation: [CosmosRatio; 3],
    pub extra_extrapolation: Option<[CosmosRatio; 3]>,
    pub extra_per_block_absolute_position: Option<bool>,
    pub cross_attention_embedding_channels: Option<u64>,
    pub minimum_frames_per_second: Option<u64>,
    pub maximum_frames_per_second: Option<u64>,
    pub adaln_lora_dimension: u64,
    pub memory_usage_factor: f64,
    pub model_size: CosmosModelSize,
}

pub const COSMOS_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];

pub const COSMOS_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.cosmos.CosmosT5Tokenizer",
        clip_model: "sim.text_encoders.cosmos.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: COSMOS_CLIP_CONFIGURATION,
        },
    }];

pub const COSMOS_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: COSMOS_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const COSMOS_WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "net.",
    target_prefix: "native.",
    required: true,
}];

pub const COSMOS_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
pub const COSMOS_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const COSMOS_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[ModelLayoutSignature {
    layout: ModelStateLayout::StandaloneNative,
    required_keys: &["net.x_embedder.proj.1.weight"],
    required_prefixes: &[],
}];

pub const COSMOS_GENERAL_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"net."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"net.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const COSMOS_PREDICT2_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"net."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"net.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const COSMOS_GENERAL_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] =
    &[ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &COSMOS_GENERAL_STATE_PLAN,
    }];

pub const COSMOS_PREDICT2_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] =
    &[ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &COSMOS_PREDICT2_STATE_PLAN,
    }];

pub fn configuration_for_probe(
    probe: &ModelProbe,
    architecture: CosmosArchitecture,
    image_to_video: bool,
    family_identifier: &str,
) -> Result<CosmosConfiguration, ModelFamilyError> {
    let invalid_configuration = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "{family_identifier} configuration is invalid: {message}"
        ))
    };
    if probe.select_layout(COSMOS_LAYOUT_SIGNATURES)? != ModelStateLayout::StandaloneNative {
        return Err(invalid_configuration(
            "only standalone-native layout is supported".to_owned(),
        ));
    }

    let general_marker = COSMOS_GENERAL_DETECTION_MARKER_KEYS
        .iter()
        .any(|key| probe.tensor_shapes.contains_key(*key));
    let predict2_marker = COSMOS_PREDICT2_DETECTION_MARKER_KEYS
        .iter()
        .any(|key| probe.tensor_shapes.contains_key(*key));
    let anima_marker = COSMOS_ANIMA_DETECTION_MARKER_KEYS
        .iter()
        .any(|key| probe.tensor_shapes.contains_key(*key));
    match architecture {
        CosmosArchitecture::GeneralDit if predict2_marker => {
            return Err(invalid_configuration("Cosmos Predict2 marker".to_string()));
        }
        CosmosArchitecture::GeneralDit if !general_marker => {
            return Err(invalid_configuration(
                "missing Cosmos GeneralDIT marker".to_string(),
            ));
        }
        CosmosArchitecture::Predict2 if anima_marker => {
            return Err(invalid_configuration("Anima marker".to_string()));
        }
        CosmosArchitecture::Predict2 if !predict2_marker => {
            return Err(invalid_configuration(
                "missing Cosmos Predict2 marker".to_string(),
            ));
        }
        _ => {}
    }

    let projection = required_shape(
        probe,
        "net.x_embedder.proj.1.weight",
        &invalid_configuration,
    )?;
    if projection.len() != 2 || projection[1] % 4 != 0 {
        return Err(invalid_configuration(
            "x_embedder.proj.1.weight shape".to_string(),
        ));
    }
    let in_channels = (projection[1] / 4)
        .checked_sub(1)
        .ok_or_else(|| invalid_configuration("x_embedder channel underflow".to_string()))?;
    let expected_in_channels = if image_to_video { 17 } else { 16 };
    if in_channels != expected_in_channels {
        return Err(invalid_configuration(format!(
            "in_channels {in_channels}; requires {expected_in_channels}"
        )));
    }

    let model_channels = match architecture {
        CosmosArchitecture::GeneralDit => {
            let attention = required_shape(
                probe,
                "net.blocks.block0.blocks.0.block.attn.to_q.0.weight",
                &invalid_configuration,
            )?;
            if attention.len() != 2 || attention[1] == 0 {
                return Err(invalid_configuration(
                    "Cosmos attention projection shape".to_string(),
                ));
            }
            attention[0]
        }
        CosmosArchitecture::Predict2 => projection[0],
    };

    let (model_size, number_of_blocks, number_of_heads) = match (architecture, model_channels) {
        (CosmosArchitecture::GeneralDit, 4_096) => (CosmosModelSize::SevenB, 28, 32),
        (CosmosArchitecture::GeneralDit, 5_120) => (CosmosModelSize::FourteenB, 36, 40),
        (CosmosArchitecture::Predict2, 2_048) => (CosmosModelSize::Predict2TwoB, 28, 16),
        (CosmosArchitecture::Predict2, 5_120) => (CosmosModelSize::FourteenB, 36, 40),
        (CosmosArchitecture::GeneralDit, value) => {
            return Err(invalid_configuration(format!(
                "model_channels {value}; expected 4096 or 5120"
            )));
        }
        (CosmosArchitecture::Predict2, value) => {
            return Err(invalid_configuration(format!(
                "model_channels {value}; expected 2048 or 5120"
            )));
        }
    };

    let one = CosmosRatio {
        numerator: 1,
        denominator: 1,
    };
    let two = CosmosRatio {
        numerator: 2,
        denominator: 1,
    };
    let (rope_extrapolation, extra_extrapolation, extra_per_block_absolute_position) =
        match (architecture, image_to_video, model_channels) {
            (CosmosArchitecture::GeneralDit, _, 4_096) => ([one, one, two], None, Some(true)),
            (CosmosArchitecture::GeneralDit, _, 5_120) => {
                ([two, two, two], Some([two, two, two]), Some(true))
            }
            (CosmosArchitecture::Predict2, false, _) => (
                [
                    CosmosRatio {
                        numerator: 4,
                        denominator: 1,
                    },
                    CosmosRatio {
                        numerator: 4,
                        denominator: 1,
                    },
                    one,
                ],
                Some([one, one, one]),
                Some(false),
            ),
            (CosmosArchitecture::Predict2, true, 2_048) => (
                [
                    CosmosRatio {
                        numerator: 3,
                        denominator: 1,
                    },
                    CosmosRatio {
                        numerator: 3,
                        denominator: 1,
                    },
                    one,
                ],
                Some([one, one, one]),
                Some(false),
            ),
            (CosmosArchitecture::Predict2, true, 5_120) => (
                [
                    two,
                    two,
                    CosmosRatio {
                        numerator: 5,
                        denominator: 6,
                    },
                ],
                Some([one, one, one]),
                None,
            ),
            _ => {
                return Err(invalid_configuration(
                    "unreachable Cosmos profile".to_string(),
                ));
            }
        };

    Ok(CosmosConfiguration {
        architecture,
        in_channels,
        out_channels: 16,
        model_channels,
        number_of_blocks,
        number_of_heads,
        maximum_image_height: 240,
        maximum_image_width: 240,
        maximum_frames: 128,
        spatial_patch_size: 2,
        temporal_patch_size: 1,
        concatenate_padding_mask: true,
        image_to_video,
        positional_embeddings_learnable: architecture == CosmosArchitecture::Predict2,
        rope_extrapolation,
        extra_extrapolation,
        extra_per_block_absolute_position,
        cross_attention_embedding_channels: (architecture == CosmosArchitecture::Predict2)
            .then_some(1_024),
        minimum_frames_per_second: (architecture == CosmosArchitecture::Predict2).then_some(1),
        maximum_frames_per_second: (architecture == CosmosArchitecture::Predict2).then_some(30),
        adaln_lora_dimension: 256,
        memory_usage_factor: match architecture {
            CosmosArchitecture::GeneralDit => 1.6,
            CosmosArchitecture::Predict2 => (model_channels as f64 / 2_048.0) * 0.95,
        },
        model_size,
    })
}

fn required_shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}
