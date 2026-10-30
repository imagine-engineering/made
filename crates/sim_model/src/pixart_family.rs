use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyError, ModelForwardOperation, ModelForwardStep, ModelKeyPredicate, ModelKeyRewrite,
    ModelKeySelector, ModelProbe, ModelStateTarget, ModelStateTensorReference,
    ModelStateTransformOperation, ModelStateTransformPlan, ModelStateTransformPlanDefinition,
    ModelUnmatchedKeyDisposition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const PIXART_HIDDEN_SIZE: u64 = 1_152;
pub const PIXART_HEAD_COUNT: u64 = 16;
pub const PIXART_INPUT_CHANNELS: u64 = 4;
pub const PIXART_PATCH_SIZE: u64 = 2;
pub const PIXART_CAPTION_CHANNELS: u64 = 4_096;
pub const PIXART_MLP_RATIO: f64 = 4.0;
pub const PIXART_MEMORY_USAGE_FACTOR: f64 = 0.5;
pub const PIXART_MAX_DEPTH: usize = 64;
pub const PIXART_MAX_MODEL_LENGTH: u64 = 4_096;
pub const PIXART_TIMESTEPS: u64 = 1_000;
pub const PIXART_LINEAR_START: f64 = 0.0001;
pub const PIXART_LINEAR_END: f64 = 0.02;
pub const PIXART_BETA_SCHEDULE: &str = "sqrt_linear";

pub const PIXART_ALPHA_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_sd15_sim_model_0045::LATENT_FORMAT;
pub const PIXART_SIGMA_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_sdxl_sim_model_0047::LATENT_FORMAT;

pub const PIXART_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.pixart_t5.PixArtTokenizer",
        clip_model: "sim.text_encoders.pixart_t5.PixArtT5XXL",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
pub static PIXART_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: PIXART_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixArtVariant {
    Alpha,
    Sigma,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixArtLayout {
    PrefixedNative,
    StandaloneNative,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixArtConditioningKey {
    CrossAttention,
    Size,
    AspectRatio,
}

impl PixArtConditioningKey {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CrossAttention => "c_crossattn",
            Self::Size => "c_size",
            Self::AspectRatio => "c_ar",
        }
    }
}

pub const PIXART_ALPHA_CONDITIONING_KEYS: &[PixArtConditioningKey] = &[
    PixArtConditioningKey::CrossAttention,
    PixArtConditioningKey::Size,
    PixArtConditioningKey::AspectRatio,
];
pub const PIXART_SIGMA_CONDITIONING_KEYS: &[PixArtConditioningKey] =
    &[PixArtConditioningKey::CrossAttention];

#[derive(Clone, Copy, Debug)]
pub struct PixArtConfiguration {
    pub variant: PixArtVariant,
    pub layout: PixArtLayout,
    pub hidden_size: u64,
    pub number_of_heads: u64,
    pub depth: usize,
    pub input_channels: u64,
    pub patch_size: u64,
    pub caption_channels: u64,
    pub model_max_length: u64,
    pub input_size: Option<u64>,
    pub positional_interpolation: Option<u64>,
    pub micro_conditioning: bool,
    pub conditioning_keys: &'static [PixArtConditioningKey],
    pub memory_usage_factor: f64,
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub memory_estimator: MemoryEstimatorDescriptor,
}

pub const PIXART_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "PixArt Alpha or Sigma image diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "optional canonical SD15 or SDXL latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "optional PixArt T5XXL text encoder",
        required: false,
    },
];

pub const PIXART_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.x_embedder.proj.weight",
    "native.t_embedder.mlp.0.weight",
    "native.y_embedder.y_proj.fc1.weight",
    "native.blocks.0.attn.qkv.weight",
    "native.blocks.0.cross_attn.q_linear.weight",
    "native.blocks.0.cross_attn.kv_linear.weight",
    "native.final_layer.linear.weight",
];
pub const PIXART_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.x_embedder.proj.bias",
    "native.t_embedder.mlp.0.bias",
    "native.y_embedder.y_proj.fc1.bias",
    "native.final_layer.linear.bias",
    "native.csize_embedder.mlp.0.weight",
    "native.ar_embedder.mlp.0.weight",
    "native.pos_embed",
];
pub const PIXART_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: PIXART_MODEL_REQUIRED_KEYS,
        optional_keys: PIXART_MODEL_OPTIONAL_KEYS,
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

pub const PIXART_SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const PIXART_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];
pub const PIXART_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 4,
    activation_bytes_per_element: 4,
};

pub const PIXART_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "patch_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.x_embedder.proj.weight",
            bias: Some("native.x_embedder.proj.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.t_embedder.mlp.0.weight",
            bias: Some("native.t_embedder.mlp.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "timestep_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "caption_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.y_embedder.y_proj.fc1.weight",
            bias: Some("native.y_embedder.y_proj.fc1.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_output",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const PIXART_PREFIXED_NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations":[
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const PIXART_STANDALONE_NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations":[
            {"Move":{"selector":{"predicate":{"Prefix":"t_block."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"t_block.","to":"native.t_block."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t_embedder."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"t_embedder.","to":"native.t_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"x_embedder."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"y_embedder."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"y_embedder.","to":"native.y_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"csize_embedder."},"minimum_matches":0,"maximum_matches":64},"rewrite":{"Prefix":{"from":"csize_embedder.","to":"native.csize_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"ar_embedder."},"minimum_matches":0,"maximum_matches":64},"rewrite":{"Prefix":{"from":"ar_embedder.","to":"native.ar_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"blocks.","to":"native.blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_layer."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Exact":"pos_embed"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.pos_embed"},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub fn native_state_plan_for_layout(
    layout: PixArtLayout,
) -> Result<&'static ModelStateTransformPlanDefinition, ModelFamilyError> {
    match layout {
        PixArtLayout::PrefixedNative => Ok(&PIXART_PREFIXED_NATIVE_STATE_PLAN),
        PixArtLayout::StandaloneNative => Ok(&PIXART_STANDALONE_NATIVE_STATE_PLAN),
        PixArtLayout::Diffusers => Err(invalid(
            "Diffusers conversion requires the checked depth/variant plan builder".to_owned(),
        )),
    }
}

pub fn conditioning_keys_for_variant(variant: PixArtVariant) -> &'static [PixArtConditioningKey] {
    match variant {
        PixArtVariant::Alpha => PIXART_ALPHA_CONDITIONING_KEYS,
        PixArtVariant::Sigma => PIXART_SIGMA_CONDITIONING_KEYS,
    }
}

pub fn diffusers_state_plan(
    depth: usize,
    variant: PixArtVariant,
) -> Result<ModelStateTransformPlan, ModelFamilyError> {
    if depth == 0 || depth > PIXART_MAX_DEPTH {
        return Err(invalid(format!(
            "Diffusers depth {depth} is outside 1..={PIXART_MAX_DEPTH}"
        )));
    }
    let mut operations = Vec::new();
    let micro = [
        (
            "adaln_single.emb.resolution_embedder.linear_1.weight",
            "native.csize_embedder.mlp.0.weight",
        ),
        (
            "adaln_single.emb.resolution_embedder.linear_1.bias",
            "native.csize_embedder.mlp.0.bias",
        ),
        (
            "adaln_single.emb.resolution_embedder.linear_2.weight",
            "native.csize_embedder.mlp.2.weight",
        ),
        (
            "adaln_single.emb.resolution_embedder.linear_2.bias",
            "native.csize_embedder.mlp.2.bias",
        ),
        (
            "adaln_single.emb.aspect_ratio_embedder.linear_1.weight",
            "native.ar_embedder.mlp.0.weight",
        ),
        (
            "adaln_single.emb.aspect_ratio_embedder.linear_1.bias",
            "native.ar_embedder.mlp.0.bias",
        ),
        (
            "adaln_single.emb.aspect_ratio_embedder.linear_2.weight",
            "native.ar_embedder.mlp.2.weight",
        ),
        (
            "adaln_single.emb.aspect_ratio_embedder.linear_2.bias",
            "native.ar_embedder.mlp.2.bias",
        ),
    ];
    if variant == PixArtVariant::Alpha {
        for (source, target) in micro {
            push_move_exact(&mut operations, source, target, true)?;
        }
    }
    for (source, target) in [
        ("pos_embed.proj.weight", "native.x_embedder.proj.weight"),
        ("pos_embed.proj.bias", "native.x_embedder.proj.bias"),
        (
            "caption_projection.y_embedding",
            "native.y_embedder.y_embedding",
        ),
        (
            "caption_projection.linear_1.weight",
            "native.y_embedder.y_proj.fc1.weight",
        ),
        (
            "caption_projection.linear_1.bias",
            "native.y_embedder.y_proj.fc1.bias",
        ),
        (
            "caption_projection.linear_2.weight",
            "native.y_embedder.y_proj.fc2.weight",
        ),
        (
            "caption_projection.linear_2.bias",
            "native.y_embedder.y_proj.fc2.bias",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_1.weight",
            "native.t_embedder.mlp.0.weight",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_1.bias",
            "native.t_embedder.mlp.0.bias",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "native.t_embedder.mlp.2.weight",
        ),
        (
            "adaln_single.emb.timestep_embedder.linear_2.bias",
            "native.t_embedder.mlp.2.bias",
        ),
        ("adaln_single.linear.weight", "native.t_block.1.weight"),
        ("adaln_single.linear.bias", "native.t_block.1.bias"),
        ("proj_out.weight", "native.final_layer.linear.weight"),
        ("proj_out.bias", "native.final_layer.linear.bias"),
        ("scale_shift_table", "native.final_layer.scale_shift_table"),
    ] {
        push_move_exact(&mut operations, source, target, true)?;
    }

    for index in 0..depth {
        for suffix in ["weight", "bias"] {
            push_assemble(
                &mut operations,
                &[
                    format!("transformer_blocks.{index}.attn1.to_q.{suffix}"),
                    format!("transformer_blocks.{index}.attn1.to_k.{suffix}"),
                    format!("transformer_blocks.{index}.attn1.to_v.{suffix}"),
                ],
                &format!("native.blocks.{index}.attn.qkv.{suffix}"),
            )?;
            push_move_exact(
                &mut operations,
                &format!("transformer_blocks.{index}.attn2.to_q.{suffix}"),
                &format!("native.blocks.{index}.cross_attn.q_linear.{suffix}"),
                true,
            )?;
            push_assemble(
                &mut operations,
                &[
                    format!("transformer_blocks.{index}.attn2.to_k.{suffix}"),
                    format!("transformer_blocks.{index}.attn2.to_v.{suffix}"),
                ],
                &format!("native.blocks.{index}.cross_attn.kv_linear.{suffix}"),
            )?;
        }
        for (source, target) in [
            ("scale_shift_table", "scale_shift_table"),
            ("attn1.to_out.0.weight", "attn.proj.weight"),
            ("attn1.to_out.0.bias", "attn.proj.bias"),
            ("ff.net.0.proj.weight", "mlp.fc1.weight"),
            ("ff.net.0.proj.bias", "mlp.fc1.bias"),
            ("ff.net.2.weight", "mlp.fc2.weight"),
            ("ff.net.2.bias", "mlp.fc2.bias"),
            ("attn2.to_out.0.weight", "cross_attn.proj.weight"),
            ("attn2.to_out.0.bias", "cross_attn.proj.bias"),
        ] {
            push_move_exact(
                &mut operations,
                &format!("transformer_blocks.{index}.{source}"),
                &format!("native.blocks.{index}.{target}"),
                true,
            )?;
        }
    }
    push_optional_route(&mut operations, "vae.", "vae", "vae.")?;
    push_optional_route(
        &mut operations,
        "text_encoders.",
        "text_encoder",
        "text_encoder.",
    )?;
    ModelStateTransformPlan::checked(operations, ModelUnmatchedKeyDisposition::Reject)
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<PixArtConfiguration, ModelFamilyError> {
    let (layout, prefix) = select_layout(probe)?;
    let (variant, depth, model_max_length, input_size, positional_interpolation) = match layout {
        PixArtLayout::PrefixedNative | PixArtLayout::StandaloneNative => {
            validate_native(probe, prefix)?
        }
        PixArtLayout::Diffusers => validate_diffusers(probe)?,
    };
    let micro_conditioning = variant == PixArtVariant::Alpha;
    Ok(PixArtConfiguration {
        variant,
        layout,
        hidden_size: PIXART_HIDDEN_SIZE,
        number_of_heads: PIXART_HEAD_COUNT,
        depth,
        input_channels: PIXART_INPUT_CHANNELS,
        patch_size: PIXART_PATCH_SIZE,
        caption_channels: PIXART_CAPTION_CHANNELS,
        model_max_length,
        input_size,
        positional_interpolation,
        micro_conditioning,
        conditioning_keys: conditioning_keys_for_variant(variant),
        memory_usage_factor: PIXART_MEMORY_USAGE_FACTOR,
        latent_format: match variant {
            PixArtVariant::Alpha => PIXART_ALPHA_LATENT_FORMAT,
            PixArtVariant::Sigma => PIXART_SIGMA_LATENT_FORMAT,
        },
        clip_target: &PIXART_CLIP_TARGET,
        supported_dtypes: PIXART_SUPPORTED_DTYPES,
        supported_devices: PIXART_SUPPORTED_DEVICES,
        memory_estimator: PIXART_MEMORY_ESTIMATOR,
    })
}

fn select_layout(probe: &ModelProbe) -> Result<(PixArtLayout, &'static str), ModelFamilyError> {
    let candidates = [
        (
            PixArtLayout::PrefixedNative,
            "model.diffusion_model.",
            [
                "model.diffusion_model.t_block.1.weight",
                "model.diffusion_model.x_embedder.proj.weight",
                "model.diffusion_model.blocks.0.attn.qkv.weight",
                "model.diffusion_model.final_layer.linear.weight",
            ],
        ),
        (
            PixArtLayout::StandaloneNative,
            "",
            [
                "t_block.1.weight",
                "x_embedder.proj.weight",
                "blocks.0.attn.qkv.weight",
                "final_layer.linear.weight",
            ],
        ),
        (
            PixArtLayout::Diffusers,
            "",
            [
                "adaln_single.emb.timestep_embedder.linear_1.bias",
                "pos_embed.proj.bias",
                "transformer_blocks.0.attn1.to_q.weight",
                "proj_out.weight",
            ],
        ),
    ];
    let matches = candidates
        .into_iter()
        .filter(|(_, _, markers)| {
            markers
                .iter()
                .all(|key| probe.tensor_shapes.contains_key(*key))
        })
        .map(|(layout, prefix, _)| (layout, prefix))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [selected] => Ok(*selected),
        [] => Err(ModelFamilyError::ModelLayoutSelection(
            "no exact PixArt native or Diffusers layout matched".to_owned(),
        )),
        _ => Err(ModelFamilyError::ModelLayoutSelection(
            "PixArt probe ambiguously matches multiple layouts".to_owned(),
        )),
    }
}

fn validate_native(
    probe: &ModelProbe,
    prefix: &str,
) -> Result<(PixArtVariant, usize, u64, Option<u64>, Option<u64>), ModelFamilyError> {
    require_shape(
        probe,
        &format!("{prefix}x_embedder.proj.weight"),
        &[
            PIXART_HIDDEN_SIZE,
            PIXART_INPUT_CHANNELS,
            PIXART_PATCH_SIZE,
            PIXART_PATCH_SIZE,
        ],
    )?;
    require_shape(
        probe,
        &format!("{prefix}t_block.1.weight"),
        &[6 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
    )?;
    require_shape(
        probe,
        &format!("{prefix}blocks.0.attn.qkv.weight"),
        &[3 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
    )?;
    require_shape(
        probe,
        &format!("{prefix}final_layer.linear.weight"),
        &[
            2 * PIXART_INPUT_CHANNELS * PIXART_PATCH_SIZE * PIXART_PATCH_SIZE,
            PIXART_HIDDEN_SIZE,
        ],
    )?;
    let depth = checked_depth(probe, &format!("{prefix}blocks.{{}}.attn.qkv.weight"))?;
    for index in 0..depth {
        require_shape(
            probe,
            &format!("{prefix}blocks.{index}.attn.qkv.weight"),
            &[3 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
        )?;
    }
    let y = required_matrix(probe, &format!("{prefix}y_embedder.y_embedding"))?;
    if y[1] != PIXART_CAPTION_CHANNELS || y[0] == 0 || y[0] > PIXART_MAX_MODEL_LENGTH {
        return Err(invalid(format!(
            "caption embedding shape {y:?} is unsupported"
        )));
    }
    let variant = native_variant(probe, prefix)?;
    let (input_size, interpolation) =
        if let Some(shape) = probe.tensor_shapes.get(&format!("{prefix}pos_embed")) {
            if shape.len() != 3 || shape[0] != 1 || shape[2] != PIXART_HIDDEN_SIZE {
                return Err(invalid(format!(
                    "native position embedding shape {shape:?} is invalid"
                )));
            }
            let side = exact_square_side(shape[1])?;
            let input = side
                .checked_mul(PIXART_PATCH_SIZE)
                .ok_or(ModelFamilyError::MemoryOverflow)?;
            (Some(input), Some(input / 64))
        } else {
            (None, None)
        };
    Ok((variant, depth, y[0], input_size, interpolation))
}

fn validate_diffusers(
    probe: &ModelProbe,
) -> Result<(PixArtVariant, usize, u64, Option<u64>, Option<u64>), ModelFamilyError> {
    require_shape(
        probe,
        "pos_embed.proj.weight",
        &[
            PIXART_HIDDEN_SIZE,
            PIXART_INPUT_CHANNELS,
            PIXART_PATCH_SIZE,
            PIXART_PATCH_SIZE,
        ],
    )?;
    require_shape(
        probe,
        "adaln_single.linear.weight",
        &[6 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
    )?;
    for projection in ["to_q", "to_k", "to_v"] {
        require_shape(
            probe,
            &format!("transformer_blocks.0.attn1.{projection}.weight"),
            &[PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
        )?;
    }
    require_shape(
        probe,
        "proj_out.weight",
        &[
            2 * PIXART_INPUT_CHANNELS * PIXART_PATCH_SIZE * PIXART_PATCH_SIZE,
            PIXART_HIDDEN_SIZE,
        ],
    )?;
    let depth = checked_depth(probe, "transformer_blocks.{}.attn1.to_q.weight")?;
    for index in 0..depth {
        for projection in ["to_q", "to_k", "to_v"] {
            require_shape(
                probe,
                &format!("transformer_blocks.{index}.attn1.{projection}.weight"),
                &[PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
            )?;
        }
    }
    let y = required_matrix(probe, "caption_projection.y_embedding")?;
    if y[1] != PIXART_CAPTION_CHANNELS || y[0] == 0 || y[0] > PIXART_MAX_MODEL_LENGTH {
        return Err(invalid(format!(
            "caption embedding shape {y:?} is unsupported"
        )));
    }
    let variant = diffusers_variant(probe)?;
    Ok((variant, depth, y[0], None, None))
}

fn native_variant(probe: &ModelProbe, prefix: &str) -> Result<PixArtVariant, ModelFamilyError> {
    let ar = format!("{prefix}ar_embedder.mlp.0.weight");
    let csize = format!("{prefix}csize_embedder.mlp.0.weight");
    match (
        probe.tensor_shapes.contains_key(&ar),
        probe.tensor_shapes.contains_key(&csize),
    ) {
        (true, true) => {
            require_shape(probe, &ar, &[PIXART_HIDDEN_SIZE / 3, 256])?;
            require_shape(probe, &csize, &[PIXART_HIDDEN_SIZE / 3, 256])?;
            Ok(PixArtVariant::Alpha)
        }
        (false, false) => Ok(PixArtVariant::Sigma),
        _ => Err(invalid(
            "Alpha micro-conditioning size/aspect embedders are incomplete".to_owned(),
        )),
    }
}

fn diffusers_variant(probe: &ModelProbe) -> Result<PixArtVariant, ModelFamilyError> {
    let micro = [
        "adaln_single.emb.resolution_embedder.linear_1.weight",
        "adaln_single.emb.aspect_ratio_embedder.linear_1.weight",
    ];
    match micro.map(|key| probe.tensor_shapes.contains_key(key)) {
        [true, true] => {
            for key in micro {
                require_shape(probe, key, &[PIXART_HIDDEN_SIZE / 3, 256])?;
            }
            Ok(PixArtVariant::Alpha)
        }
        [false, false] => Ok(PixArtVariant::Sigma),
        _ => Err(invalid(
            "Diffusers Alpha micro-conditioning projection set is incomplete".to_owned(),
        )),
    }
}

fn checked_depth(probe: &ModelProbe, pattern: &str) -> Result<usize, ModelFamilyError> {
    let depth = probe.consecutive_block_count(pattern)?;
    if depth == 0 || depth > PIXART_MAX_DEPTH {
        return Err(invalid(format!(
            "block depth {depth} is outside 1..={PIXART_MAX_DEPTH}"
        )));
    }
    let (stem, suffix) = pattern
        .split_once("{}")
        .ok_or_else(|| invalid("block pattern has no placeholder".to_owned()))?;
    if probe.tensor_shapes.keys().any(|key| {
        key.strip_prefix(stem)
            .and_then(|tail| tail.strip_suffix(suffix))
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= depth)
    }) {
        return Err(invalid(
            "PixArt blocks are not a consecutive bounded sequence".to_owned(),
        ));
    }
    Ok(depth)
}

fn exact_square_side(value: u64) -> Result<u64, ModelFamilyError> {
    if value == 0 || value > 16 * 1024 * 1024 {
        return Err(invalid(format!(
            "position token count {value} is unsupported"
        )));
    }
    let side = (value as f64).sqrt() as u64;
    if side.checked_mul(side) != Some(value) {
        return Err(invalid(format!(
            "position token count {value} is not square"
        )));
    }
    Ok(side)
}

fn push_move_exact(
    operations: &mut Vec<ModelStateTransformOperation>,
    source: &str,
    target: &str,
    required: bool,
) -> Result<(), ModelFamilyError> {
    let predicate = ModelKeyPredicate::exact(source)?;
    let selector = if required {
        ModelKeySelector::exact(source)?
    } else {
        ModelKeySelector::bounded(predicate, 0, 1)?
    };
    operations.push(ModelStateTransformOperation::Move {
        selector,
        rewrite: ModelKeyRewrite::exact(target)?,
        component: "model".to_owned(),
    });
    Ok(())
}

fn push_assemble(
    operations: &mut Vec<ModelStateTransformOperation>,
    sources: &[String],
    target: &str,
) -> Result<(), ModelFamilyError> {
    operations.push(ModelStateTransformOperation::Assemble {
        sources: sources
            .iter()
            .map(|source| ModelStateTensorReference::source(source))
            .collect::<Result<Vec<_>, _>>()?,
        dimension: 0,
        output: ModelStateTarget::checked("model", target)?,
    });
    Ok(())
}

fn push_optional_route(
    operations: &mut Vec<ModelStateTransformOperation>,
    source_prefix: &str,
    component: &str,
    target_prefix: &str,
) -> Result<(), ModelFamilyError> {
    operations.push(ModelStateTransformOperation::Move {
        selector: ModelKeySelector::bounded(ModelKeyPredicate::prefix(source_prefix)?, 0, 16_384)?,
        rewrite: ModelKeyRewrite::prefix(source_prefix, target_prefix)?,
        component: component.to_owned(),
    });
    Ok(())
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

fn require_shape(probe: &ModelProbe, key: &str, expected: &[u64]) -> Result<(), ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape != expected {
        return Err(invalid(format!(
            "{key} shape {shape:?} must be {expected:?}"
        )));
    }
    Ok(())
}

fn invalid(message: String) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!("PixArt configuration is invalid: {message}"))
}
