use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipTargetDefinition, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile,
    ModelFamilyRegistration, ModelFamilyStatePlanSelector, ModelForwardOperation, ModelForwardStep,
    ModelProbe, ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "DepthAnything3";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0075";
pub const MODEL_FAMILY_FIXTURE: &str = "depthanything3-comfy-model-0075";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 93;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "8388072289c3c98d37af62d1ba2de8528785e60e477208c83001b8cc4219627e";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 2.0;
pub const MODEL_FAMILY_PATCH_SIZE: u64 = 14;
pub const MODEL_FAMILY_IMAGE_SIZE: u64 = 518;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthAnything3Backbone {
    VitSmall,
    VitBase,
    VitLarge,
    VitGiant,
}

impl DepthAnything3Backbone {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::VitSmall => "vits",
            Self::VitBase => "vitb",
            Self::VitLarge => "vitl",
            Self::VitGiant => "vitg",
        }
    }

    pub const fn hidden_size(self) -> u64 {
        match self {
            Self::VitSmall => 384,
            Self::VitBase => 768,
            Self::VitLarge => 1_024,
            Self::VitGiant => 1_536,
        }
    }

    pub const fn layer_count(self) -> usize {
        match self {
            Self::VitSmall | Self::VitBase => 12,
            Self::VitLarge => 24,
            Self::VitGiant => 40,
        }
    }

    pub const fn attention_heads(self) -> u64 {
        match self {
            Self::VitSmall => 6,
            Self::VitBase => 12,
            Self::VitLarge => 16,
            Self::VitGiant => 24,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthAnything3Head {
    Dpt,
    DualDpt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DepthAnything3Configuration {
    pub backbone: DepthAnything3Backbone,
    pub hidden_size: u64,
    pub layer_count: usize,
    pub attention_heads: u64,
    pub patch_size: u64,
    pub image_size: u64,
    pub qknorm_start: Option<usize>,
    pub alternate_attention_start: Option<usize>,
    pub rope_start: Option<usize>,
    pub concatenate_camera_token: bool,
    pub head: DepthAnything3Head,
    pub head_dimension: u64,
    pub head_features: u64,
    pub head_out_channels: [u64; 4],
    pub head_output_dimension: u64,
    pub output_layers: [usize; 4],
    pub use_sky_head: bool,
    pub has_camera_encoder: bool,
    pub camera_encoder_dimension: Option<u64>,
    pub has_camera_decoder: bool,
    pub camera_decoder_dimension: Option<u64>,
}

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: &[],
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Depth Anything 3 DINOv2 backbone and depth head",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "inherited optional first-stage state",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "inherited optional conditioning state without a CLIP target",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::KeyPresent {
        key: "model.diffusion_model.backbone.embeddings.patch_embeddings.projection.weight",
        score: 700,
    },
    ModelDetectionRule::KeyPresent {
        key: "model.diffusion_model.head.projects.0.weight",
        score: 150,
    },
    ModelDetectionRule::KeyPresent {
        key: "model.diffusion_model.head.scratch.refinenet1.out_conv.weight",
        score: 150,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[
    ModelWeightRule {
        source_prefix: "model.diffusion_model.",
        target_prefix: "native.",
        required: true,
    },
    ModelWeightRule {
        source_prefix: "first_stage_model.",
        target_prefix: "vae.",
        required: false,
    },
    ModelWeightRule {
        source_prefix: "cond_stage_model.",
        target_prefix: "text_encoder.",
        required: false,
    },
];

const REQUIRED_KEYS: &[&str] = &[
    "native.head.projects.0.weight",
    "native.head.scratch.refinenet1.out_conv.weight",
    "native.head.scratch.output_conv2.2.weight",
];

const OPTIONAL_KEYS: &[&str] = &[
    "native.backbone.embeddings.patch_embeddings.projection.weight",
    "native.backbone.embeddings.camera_token",
    "native.head.scratch.refinenet1_aux.out_conv.weight",
    "native.head.scratch.sky_output_conv2.0.weight",
    "native.cam_enc.token_norm.weight",
    "native.cam_enc.pose_branch.fc2.weight",
    "native.cam_dec.fc_t.weight",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "backbone_feature_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.head.projects.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "backbone_feature_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "depth_refinement",
        operation: ModelForwardOperation::Linear {
            weight: "native.head.scratch.refinenet1.out_conv.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "depth_refinement_normalized",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "depth_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.head.scratch.output_conv2.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "depth_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "depth-anything-3-dinov2-v1",
    latent_feature_id: MODEL_FAMILY_FEATURE_ID,
    latent_identifier: "LatentFormat",
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

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: REQUIRED_KEYS,
        optional_keys: OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "vae",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 93,
    source_architecture: "model_base.DepthAnything3",
    source_configuration: &[],
    required_state_keys: &[
        "model.diffusion_model.backbone.embeddings.patch_embeddings.projection.weight",
        "model.diffusion_model.head.projects.0.weight",
        "model.diffusion_model.head.scratch.refinenet1.out_conv.weight",
    ],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Static(&NATIVE_STATE_PLAN),
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<DepthAnything3Configuration, ModelFamilyError> {
    let prefix = "model.diffusion_model.";
    if probe.unet_prefix_selection()?.prefix() != prefix {
        return Err(invalid_configuration(
            "the pinned source supports only the native model.diffusion_model. layout",
        ));
    }

    let projection = shape(
        probe,
        "model.diffusion_model.backbone.embeddings.patch_embeddings.projection.weight",
    )?;
    if projection.len() != 4
        || projection[1] != 3
        || projection[2] != MODEL_FAMILY_PATCH_SIZE
        || projection[3] != MODEL_FAMILY_PATCH_SIZE
    {
        return Err(invalid_configuration("DINOv2 patch projection shape"));
    }
    let backbone = match projection[0] {
        384 => DepthAnything3Backbone::VitSmall,
        768 => DepthAnything3Backbone::VitBase,
        1_024 => DepthAnything3Backbone::VitLarge,
        1_536 => DepthAnything3Backbone::VitGiant,
        value => {
            return Err(invalid_configuration(format!(
                "unsupported backbone hidden size {value}"
            )));
        }
    };

    let layer_pattern = "model.diffusion_model.backbone.encoder.layer.{}.";
    let layer_count = probe.consecutive_block_count(layer_pattern)?;
    if layer_count != backbone.layer_count() {
        return Err(invalid_configuration(format!(
            "{} requires {} consecutive layers, found {layer_count}",
            backbone.identifier(),
            backbone.layer_count()
        )));
    }

    let qknorm_indices = (0..layer_count)
        .filter(|index| {
            probe.tensor_shapes.contains_key(&format!(
                "{prefix}backbone.encoder.layer.{index}.attention.q_norm.weight"
            ))
        })
        .collect::<Vec<_>>();
    let qknorm_start = qknorm_indices.first().copied();
    if let Some(start) = qknorm_start
        && qknorm_indices != (start..layer_count).collect::<Vec<_>>()
    {
        return Err(invalid_configuration(
            "qk-normalization layers must form a contiguous suffix",
        ));
    }

    let concatenate_camera_token = probe
        .tensor_shapes
        .contains_key(&format!("{prefix}backbone.embeddings.camera_token"));
    let (alternate_attention_start, rope_start, qknorm_start) = if concatenate_camera_token {
        (qknorm_start, qknorm_start, qknorm_start)
    } else {
        (None, None, None)
    };

    let head_dimension = second_dimension(probe, &format!("{prefix}head.projects.0.weight"))?;
    let head_features = first_dimension(
        probe,
        &format!("{prefix}head.scratch.refinenet1.out_conv.weight"),
    )?;
    let mut head_out_channels = [0_u64; 4];
    for (index, channel) in head_out_channels.iter_mut().enumerate() {
        *channel = first_dimension(probe, &format!("{prefix}head.projects.{index}.weight"))?;
    }

    let has_aux = probe.tensor_shapes.contains_key(&format!(
        "{prefix}head.scratch.refinenet1_aux.out_conv.weight"
    ));
    let (head, head_output_dimension, use_sky_head) = if has_aux {
        (DepthAnything3Head::DualDpt, 2, false)
    } else {
        (
            DepthAnything3Head::Dpt,
            first_dimension(
                probe,
                &format!("{prefix}head.scratch.output_conv2.2.weight"),
            )?,
            probe
                .tensor_shapes
                .contains_key(&format!("{prefix}head.scratch.sky_output_conv2.0.weight")),
        )
    };
    let output_layers = if layer_count >= 24 {
        if has_aux {
            [11, 15, 19, 23]
        } else {
            [4, 11, 17, 23]
        }
    } else {
        [5, 7, 9, 11]
    };

    let has_camera_encoder = probe
        .tensor_shapes
        .contains_key(&format!("{prefix}cam_enc.token_norm.weight"));
    let camera_encoder_dimension = has_camera_encoder
        .then(|| {
            optional_first_dimension(probe, &format!("{prefix}cam_enc.pose_branch.fc2.weight"))
        })
        .transpose()?
        .flatten();
    let has_camera_decoder = probe
        .tensor_shapes
        .contains_key(&format!("{prefix}cam_dec.fc_t.weight"));
    let camera_decoder_dimension = has_camera_decoder
        .then(|| optional_second_dimension(probe, &format!("{prefix}cam_dec.fc_t.weight")))
        .transpose()?
        .flatten();

    Ok(DepthAnything3Configuration {
        backbone,
        hidden_size: backbone.hidden_size(),
        layer_count,
        attention_heads: backbone.attention_heads(),
        patch_size: MODEL_FAMILY_PATCH_SIZE,
        image_size: MODEL_FAMILY_IMAGE_SIZE,
        qknorm_start,
        alternate_attention_start,
        rope_start,
        concatenate_camera_token,
        head,
        head_dimension,
        head_features,
        head_out_channels,
        head_output_dimension,
        output_layers,
        use_sky_head,
        has_camera_encoder,
        camera_encoder_dimension,
        has_camera_decoder,
        camera_decoder_dimension,
    })
}

fn shape<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing {key}")))
}

fn first_dimension(probe: &ModelProbe, key: &str) -> Result<u64, ModelFamilyError> {
    shape(probe, key)?
        .first()
        .copied()
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| invalid_configuration(format!("invalid first dimension for {key}")))
}

fn second_dimension(probe: &ModelProbe, key: &str) -> Result<u64, ModelFamilyError> {
    shape(probe, key)?
        .get(1)
        .copied()
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| invalid_configuration(format!("invalid second dimension for {key}")))
}

fn optional_first_dimension(
    probe: &ModelProbe,
    key: &str,
) -> Result<Option<u64>, ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(|_| first_dimension(probe, key))
        .transpose()
}

fn optional_second_dimension(
    probe: &ModelProbe,
    key: &str,
) -> Result<Option<u64>, ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(|_| second_dimension(probe, key))
        .transpose()
}

fn invalid_configuration(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "DepthAnything3 configuration is invalid: {}",
        message.into()
    ))
}
