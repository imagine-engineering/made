use crate::{
    LatentExtent, LatentFormatDefinition, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyError, ModelLayoutSignature, ModelProbe,
    ModelStateLayout, PatchGraph, PatchGraphError, PatchOperation,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const QWEN_IMAGE_PATCH_SIZE: u64 = 2;
pub const QWEN_IMAGE_ATTENTION_HEAD_DIMENSION: u64 = 128;
pub const QWEN_IMAGE_NUMBER_OF_ATTENTION_HEADS: u64 = 24;
pub const QWEN_IMAGE_INNER_DIMENSION: u64 = 3_072;
pub const QWEN_IMAGE_JOINT_ATTENTION_DIMENSION: u64 = 3_584;
pub const QWEN_IMAGE_POOLED_PROJECTION_DIMENSION: u64 = 768;
pub const QWEN_IMAGE_AXES_DIMENSIONS: [u64; 3] = [16, 56, 56];
pub const QWEN_IMAGE_SAMPLING_SHIFT: f64 = 1.15;
pub const QWEN_IMAGE_MEMORY_USAGE_FACTOR: f64 = 1.8;
pub const QWEN_IMAGE_MAXIMUM_DEPTH: usize = 128;
pub const QWEN_IMAGE_MAXIMUM_LAYERED_SLICES: u64 = 4_096;

pub const QWEN_IMAGE_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_wan21_sim_model_0053::LATENT_FORMAT;

pub const QWEN_IMAGE_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const QWEN_IMAGE_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];
pub const QWEN_IMAGE_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 4,
    activation_bytes_per_element: 4,
};

pub const QWEN_IMAGE_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const QWEN_IMAGE_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.qwen_image.QwenImageTokenizer",
        clip_model: "sim.text_encoders.qwen_image.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: QWEN_IMAGE_CLIP_CONFIGURATION,
        },
    }];
pub static QWEN_IMAGE_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: QWEN_IMAGE_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const QWEN_IMAGE_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.txt_norm.weight",
            "model.diffusion_model.img_in.weight",
            "model.diffusion_model.transformer_blocks.0.img_mod.1.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "txt_norm.weight",
            "img_in.weight",
            "transformer_blocks.0.img_mod.1.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "transformer.txt_norm.weight",
            "transformer.img_in.weight",
            "transformer.transformer_blocks.0.img_mod.1.weight",
        ],
        required_prefixes: &[],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenImageReferenceMethod {
    Index,
    IndexTimestepZero,
    NegativeIndex,
}

impl QwenImageReferenceMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::IndexTimestepZero => "index_timestep_zero",
            Self::NegativeIndex => "negative_index",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenImageConditioningKey {
    AttentionMask,
    CrossAttention,
    ReferenceLatents,
    ReferenceLatentsMethod,
    AdditionalTimestepCondition,
}

impl QwenImageConditioningKey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AttentionMask => "attention_mask",
            Self::CrossAttention => "c_crossattn",
            Self::ReferenceLatents => "ref_latents",
            Self::ReferenceLatentsMethod => "ref_latents_method",
            Self::AdditionalTimestepCondition => "additional_t_cond",
        }
    }
}

pub const QWEN_IMAGE_BASE_CONDITIONING_KEYS: &[QwenImageConditioningKey] = &[
    QwenImageConditioningKey::AttentionMask,
    QwenImageConditioningKey::CrossAttention,
    QwenImageConditioningKey::ReferenceLatents,
    QwenImageConditioningKey::ReferenceLatentsMethod,
];
pub const QWEN_IMAGE_LAYERED_CONDITIONING_KEYS: &[QwenImageConditioningKey] = &[
    QwenImageConditioningKey::AttentionMask,
    QwenImageConditioningKey::CrossAttention,
    QwenImageConditioningKey::ReferenceLatents,
    QwenImageConditioningKey::ReferenceLatentsMethod,
    QwenImageConditioningKey::AdditionalTimestepCondition,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenImageBlockPrefix {
    pub role: &'static str,
    pub source_prefix: &'static str,
    pub mapped_prefix: &'static str,
}

pub const QWEN_IMAGE_BLOCK_PREFIXES: &[QwenImageBlockPrefix] = &[
    QwenImageBlockPrefix {
        role: "base_transformer",
        source_prefix: "transformer_blocks.",
        mapped_prefix: "native.transformer_blocks.",
    },
    QwenImageBlockPrefix {
        role: "blockwise_control",
        source_prefix: "controlnet_blocks.",
        mapped_prefix: "native.controlnet_blocks.",
    },
    QwenImageBlockPrefix {
        role: "fun_control",
        source_prefix: "control_blocks.",
        mapped_prefix: "native.control_blocks.",
    },
];

pub const QWEN_IMAGE_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Qwen Image diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "optional canonical Wan21 latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "optional Qwen 2.5 7B image text encoder",
        required: false,
    },
];

pub const QWEN_IMAGE_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.txt_norm.weight",
    "native.img_in.weight",
    "native.txt_in.weight",
    "native.transformer_blocks.0.img_mod.1.weight",
    "native.transformer_blocks.0.txt_mod.1.weight",
    "native.transformer_blocks.0.attn.to_q.weight",
    "native.proj_out.weight",
];
pub const QWEN_IMAGE_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.__index_timestep_zero__",
    "native.time_text_embed.addition_t_embedding.weight",
];
pub const QWEN_IMAGE_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: QWEN_IMAGE_MODEL_REQUIRED_KEYS,
        optional_keys: QWEN_IMAGE_MODEL_OPTIONAL_KEYS,
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

#[derive(Clone, Copy, Debug)]
pub struct QwenImageConfiguration {
    pub layout: ModelStateLayout,
    pub input_channels: u64,
    pub output_channels: u64,
    pub number_of_layers: usize,
    pub inner_dimension: u64,
    pub number_of_attention_heads: u64,
    pub attention_head_dimension: u64,
    pub joint_attention_dimension: u64,
    pub pooled_projection_dimension: u64,
    pub patch_size: u64,
    pub axes_dimensions: [u64; 3],
    pub txt_norm: bool,
    pub timestep_zero_marker: bool,
    pub use_additional_timestep_condition: bool,
    pub supports_reference_images: bool,
    pub reference_method: QwenImageReferenceMethod,
    pub conditioning_keys: &'static [QwenImageConditioningKey],
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub memory_estimator: MemoryEstimatorDescriptor,
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<QwenImageConfiguration, ModelFamilyError> {
    let layout = probe.select_layout(QWEN_IMAGE_LAYOUT_SIGNATURES)?;
    let prefix = match layout {
        ModelStateLayout::PrefixedNative => "model.diffusion_model.",
        ModelStateLayout::StandaloneNative => "",
        ModelStateLayout::Diffusers => {
            return Err(invalid(
                "the pinned Diffusers layout is not admitted; a source-native state dictionary is required",
            ));
        }
    };

    let txt_norm_key = format!("{prefix}txt_norm.weight");
    let txt_norm = required_vector(probe, &txt_norm_key)?;
    let joint_attention_dimension = txt_norm[0];
    if joint_attention_dimension != QWEN_IMAGE_JOINT_ATTENTION_DIMENSION {
        return Err(invalid(format!(
            "{txt_norm_key} dimension {joint_attention_dimension} must be {QWEN_IMAGE_JOINT_ATTENTION_DIMENSION}"
        )));
    }

    let img_in_key = format!("{prefix}img_in.weight");
    let img_in = required_matrix(probe, &img_in_key)?;
    let inner_dimension = img_in[0];
    let input_channels = img_in[1];
    if inner_dimension != QWEN_IMAGE_INNER_DIMENSION
        || input_channels == 0
        || input_channels % 4 != 0
    {
        return Err(invalid(format!(
            "{img_in_key} shape {img_in:?} must have {QWEN_IMAGE_INNER_DIMENSION} outputs and a positive input width divisible by four"
        )));
    }

    let txt_in_key = format!("{prefix}txt_in.weight");
    require_matrix_shape(
        probe,
        &txt_in_key,
        [inner_dimension, joint_attention_dimension],
    )?;
    let number_of_layers = consecutive_block_count(
        probe,
        &format!("{prefix}transformer_blocks.{{}}."),
        "Qwen Image transformer",
    )?;
    if number_of_layers == 0 || number_of_layers > QWEN_IMAGE_MAXIMUM_DEPTH {
        return Err(invalid(format!(
            "transformer depth {number_of_layers} is outside 1..={QWEN_IMAGE_MAXIMUM_DEPTH}"
        )));
    }
    for ordinal in 0..number_of_layers {
        let block = format!("{prefix}transformer_blocks.{ordinal}");
        require_matrix_shape(
            probe,
            &format!("{block}.img_mod.1.weight"),
            [inner_dimension * 6, inner_dimension],
        )?;
        require_matrix_shape(
            probe,
            &format!("{block}.txt_mod.1.weight"),
            [inner_dimension * 6, inner_dimension],
        )?;
        require_matrix_shape(
            probe,
            &format!("{block}.attn.to_q.weight"),
            [inner_dimension, inner_dimension],
        )?;
        require_vector_shape(
            probe,
            &format!("{block}.attn.norm_q.weight"),
            QWEN_IMAGE_ATTENTION_HEAD_DIMENSION,
        )?;
    }

    let proj_out_key = format!("{prefix}proj_out.weight");
    let proj_out = required_matrix(probe, &proj_out_key)?;
    if proj_out[1] != inner_dimension || !proj_out[0].is_multiple_of(QWEN_IMAGE_PATCH_SIZE.pow(2)) {
        return Err(invalid(format!(
            "{proj_out_key} shape {proj_out:?} contradicts the checked inner dimension or patch size"
        )));
    }
    let output_channels = proj_out[0] / QWEN_IMAGE_PATCH_SIZE.pow(2);
    if output_channels == 0 {
        return Err(invalid("Qwen Image output channel count is zero"));
    }

    let timestep_zero_key = format!("{prefix}__index_timestep_zero__");
    let timestep_zero_marker = probe.tensor_shapes.contains_key(&timestep_zero_key);
    if timestep_zero_marker && probe.tensor_shapes[&timestep_zero_key] != Vec::<u64>::new() {
        return Err(invalid(format!(
            "{timestep_zero_key} must be a scalar marker"
        )));
    }
    let addition_key = format!("{prefix}time_text_embed.addition_t_embedding.weight");
    let use_additional_timestep_condition = probe.tensor_shapes.contains_key(&addition_key);
    if use_additional_timestep_condition {
        require_matrix_shape(probe, &addition_key, [2, inner_dimension])?;
    }
    let reference_method = if use_additional_timestep_condition {
        QwenImageReferenceMethod::NegativeIndex
    } else if timestep_zero_marker {
        QwenImageReferenceMethod::IndexTimestepZero
    } else {
        QwenImageReferenceMethod::Index
    };

    Ok(QwenImageConfiguration {
        layout,
        input_channels,
        output_channels,
        number_of_layers,
        inner_dimension,
        number_of_attention_heads: QWEN_IMAGE_NUMBER_OF_ATTENTION_HEADS,
        attention_head_dimension: QWEN_IMAGE_ATTENTION_HEAD_DIMENSION,
        joint_attention_dimension,
        pooled_projection_dimension: QWEN_IMAGE_POOLED_PROJECTION_DIMENSION,
        patch_size: QWEN_IMAGE_PATCH_SIZE,
        axes_dimensions: QWEN_IMAGE_AXES_DIMENSIONS,
        txt_norm: true,
        timestep_zero_marker,
        use_additional_timestep_condition,
        supports_reference_images: true,
        reference_method,
        conditioning_keys: if use_additional_timestep_condition {
            QWEN_IMAGE_LAYERED_CONDITIONING_KEYS
        } else {
            QWEN_IMAGE_BASE_CONDITIONING_KEYS
        },
        sampling_shift: QWEN_IMAGE_SAMPLING_SHIFT,
        memory_usage_factor: QWEN_IMAGE_MEMORY_USAGE_FACTOR,
        latent_format: QWEN_IMAGE_LATENT_FORMAT,
        clip_target: &QWEN_IMAGE_CLIP_TARGET,
        supported_dtypes: QWEN_IMAGE_SUPPORTED_DTYPES,
        supported_devices: QWEN_IMAGE_SUPPORTED_DEVICES,
        memory_estimator: QWEN_IMAGE_MEMORY_ESTIMATOR,
    })
}

pub fn layered_latent_extent(
    width: u64,
    height: u64,
    layers: u64,
    batch: u64,
) -> Result<LatentExtent, ModelFamilyError> {
    if width < 16 || height < 16 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(invalid(
            "layered latent width and height must be positive multiples of 16",
        ));
    }
    if batch == 0 || batch > 4_096 {
        return Err(invalid("layered latent batch must be in 1..=4096"));
    }
    let latent_slices = layers
        .checked_add(1)
        .ok_or(ModelFamilyError::ProbeDimensionOverflow)?;
    if latent_slices > QWEN_IMAGE_MAXIMUM_LAYERED_SLICES {
        return Err(invalid(format!(
            "layered latent slice count {latent_slices} exceeds {QWEN_IMAGE_MAXIMUM_LAYERED_SLICES}"
        )));
    }
    let frames = latent_slices
        .checked_mul(QWEN_IMAGE_LATENT_FORMAT.temporal_downscale_ratio)
        .ok_or(ModelFamilyError::ProbeDimensionOverflow)?;
    Ok(LatentExtent::ThreeDimensional {
        batch,
        frames,
        width,
        height,
    })
}

pub fn checked_patch_graph(
    base_artifact_digest: impl Into<String>,
    operations: Vec<PatchOperation>,
) -> Result<PatchGraph, PatchGraphError> {
    for operation in &operations {
        for target in &operation.targets {
            if !QWEN_IMAGE_BLOCK_PREFIXES
                .iter()
                .any(|prefix| target.key.starts_with(prefix.mapped_prefix))
            {
                return Err(PatchGraphError::InvalidPayload(format!(
                    "Qwen Image patch target {:?} is outside the canonical block-prefix catalog",
                    target.key
                )));
            }
        }
    }
    PatchGraph::checked(base_artifact_digest, operations)
}

fn required_matrix<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != 2 || shape.contains(&0) {
        return Err(invalid(format!(
            "{key} shape {shape:?} is not a non-empty matrix"
        )));
    }
    Ok(shape)
}

fn required_vector<'a>(probe: &'a ModelProbe, key: &str) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != 1 || shape[0] == 0 {
        return Err(invalid(format!(
            "{key} shape {shape:?} is not a non-empty vector"
        )));
    }
    Ok(shape)
}

fn require_matrix_shape(
    probe: &ModelProbe,
    key: &str,
    expected: [u64; 2],
) -> Result<(), ModelFamilyError> {
    let actual = required_matrix(probe, key)?;
    if actual != expected {
        return Err(invalid(format!(
            "{key} shape {actual:?} must be {expected:?}"
        )));
    }
    Ok(())
}

fn require_vector_shape(
    probe: &ModelProbe,
    key: &str,
    expected: u64,
) -> Result<(), ModelFamilyError> {
    let actual = required_vector(probe, key)?;
    if actual != [expected] {
        return Err(invalid(format!(
            "{key} shape {actual:?} must be [{expected}]"
        )));
    }
    Ok(())
}

fn consecutive_block_count(
    probe: &ModelProbe,
    pattern: &str,
    label: &str,
) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    let (stem, suffix) = pattern
        .split_once("{}")
        .ok_or_else(|| invalid(format!("{label} block pattern has no placeholder")))?;
    let has_gap_or_later = probe.tensor_shapes.keys().any(|key| {
        key.strip_prefix(stem)
            .and_then(|tail| tail.split_once(suffix))
            .and_then(|(index, _)| index.parse::<usize>().ok())
            .is_some_and(|index| index >= count)
    });
    if has_gap_or_later {
        return Err(invalid(format!(
            "{label} blocks are not a consecutive bounded sequence"
        )));
    }
    Ok(count)
}

fn invalid(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "Qwen Image configuration is invalid: {}",
        message.into()
    ))
}
