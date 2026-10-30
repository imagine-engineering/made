use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyError, ModelForwardOperation, ModelForwardStep,
    ModelProbe, ModelStateTransformPlanDefinition,
};
use sim_tensor::{
    BinaryOperation, CpuBackend, DType, ExecutionContext, ResizeMode, RngTransaction, Scalar,
    ScalarSide, Tensor, TensorBackend, TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_16::add_method_with_context_exact_native,
    generated_external_tensor_kernel_01::resize_with_context_exact_native,
    generated_random_number_generation_01::randn_like_with_context_exact_native,
};
use sim_types::DeviceKind;

pub const HUNYUAN_VIDEO_SAVE_PREFIX: &str = "model.model.";
pub const HUNYUAN_VIDEO_THETA: u64 = 256;
pub const HUNYUAN_VIDEO_HEAD_DIMENSION: u64 = 128;
pub const HUNYUAN_VIDEO_MLP_RATIO: f64 = 4.0;
pub const HUNYUAN_VIDEO_VECTOR_INPUT_DIMENSION: u64 = 768;
pub const HUNYUAN_VIDEO_BYT5_INPUT_DIMENSION: u64 = 1_472;
pub const HUNYUAN_VIDEO_BYT5_INTERMEDIATE_DIMENSION: u64 = 2_048;
pub const HUNYUAN_VIDEO15_VISION_INPUT_DIMENSION: u64 = 1_152;
pub const HUNYUAN_REFINER_IMAGE_SCALE: f32 = 0.75;
pub const HUNYUAN_REFINER_SEED_OFFSET: i64 = -10;

pub const HUNYUAN_IMAGE21_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanimage21_sim_model_0035::LATENT_FORMAT;
pub const HUNYUAN_IMAGE21_REFINER_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanimage21refiner_sim_model_0036::LATENT_FORMAT;
pub const HUNYUAN_VIDEO_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanvideo_sim_model_0037::LATENT_FORMAT;
pub const HUNYUAN_VIDEO15_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanvideo15_sim_model_0038::LATENT_FORMAT;

pub const HUNYUAN_VIDEO_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const HUNYUAN_VIDEO_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideoTokenizer",
        clip_model: "sim.text_encoders.hunyuan_video.hunyuan_video_clip",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: HUNYUAN_VIDEO_CLIP_CONFIGURATION,
        },
    }];
pub static HUNYUAN_VIDEO_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: HUNYUAN_VIDEO_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const HUNYUAN_IMAGE_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const HUNYUAN_IMAGE_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.hunyuan_image.HunyuanImageTokenizer",
        clip_model: "sim.text_encoders.hunyuan_image.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: HUNYUAN_IMAGE_CLIP_CONFIGURATION,
        },
    }];
pub static HUNYUAN_IMAGE_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: HUNYUAN_IMAGE_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const HUNYUAN_VIDEO15_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer",
        clip_model: "sim.text_encoders.hunyuan_image.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: HUNYUAN_IMAGE_CLIP_CONFIGURATION,
        },
    }];
pub static HUNYUAN_VIDEO15_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: HUNYUAN_VIDEO15_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const HUNYUAN_VIDEO_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Hunyuan native image/video diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Hunyuan family latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Hunyuan Llama/Qwen and ByT5 conditioning encoders",
        required: false,
    },
];

pub const HUNYUAN_VIDEO_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.img_in.proj.weight",
    "native.final_layer.linear.weight",
    "native.txt_in.input_embedder.weight",
    "native.txt_in.individual_token_refiner.blocks.0.norm1.weight",
];
pub const HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: HUNYUAN_VIDEO_MODEL_REQUIRED_KEYS,
        optional_keys: &[],
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

pub const HUNYUAN_VIDEO_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const HUNYUAN_VIDEO15_SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const HUNYUAN_VIDEO_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const HUNYUAN_VIDEO_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "conditioning.context_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.txt_in.input_embedder.weight",
            bias: Some("native.txt_in.input_embedder.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "conditioning.context_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "transformer.double_block_0_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "output.final_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const HUNYUAN_VIDEO_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: concat!(
            "{\"operations\":[{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"model.diffusion_model.\"},\"minimum_matches\":1,\"maximum_matches\":16384},\"rewrite\":",
            r#"{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"OrderedOptional":[{"Contains":{"from":"txt_in.t_embedder.mlp.0.","to":"txt_in.t_embedder.in_layer."}},{"Contains":{"from":"txt_in.t_embedder.mlp.2.","to":"txt_in.t_embedder.out_layer."}},{"Contains":{"from":"txt_in.c_embedder.linear_1.","to":"txt_in.c_embedder.in_layer."}},{"Contains":{"from":"txt_in.c_embedder.linear_2.","to":"txt_in.c_embedder.out_layer."}},{"Contains":{"from":"_mod.linear.","to":"_mod.lin."}},{"Contains":{"from":"_attn_qkv.","to":"_attn.qkv."}},{"Contains":{"from":"mlp.fc1.","to":"mlp.0."}},{"Contains":{"from":"mlp.fc2.","to":"mlp.2."}},{"Contains":{"from":"_attn_q_norm.weight","to":"_attn.norm.query_norm.weight"}},{"Contains":{"from":"_attn_k_norm.weight","to":"_attn.norm.key_norm.weight"}},{"Contains":{"from":".q_norm.weight","to":".norm.query_norm.weight"}},{"Contains":{"from":".k_norm.weight","to":".norm.key_norm.weight"}},{"Contains":{"from":"_attn_proj.","to":"_attn.proj."}},{"Contains":{"from":".modulation.linear.","to":".modulation.lin."}},{"Contains":{"from":"_in.mlp.2.","to":"_in.out_layer."}},{"Contains":{"from":"_in.mlp.0.","to":"_in.in_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]}]}"#,
            ",\"component\":\"model\"}},{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"vae.\"},\"minimum_matches\":0,\"maximum_matches\":16384},\"rewrite\":\"Identity\",\"component\":\"vae\"}},{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"text_encoders.\"},\"minimum_matches\":0,\"maximum_matches\":16384},\"rewrite\":{\"Prefix\":{\"from\":\"text_encoders.\",\"to\":\"text_encoder.\"}},\"component\":\"text_encoder\"}}],\"unmatched\":\"Reject\"}"
        ),
    };
pub const HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: concat!(
            "{\"operations\":[{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"model.\"},\"minimum_matches\":1,\"maximum_matches\":16384},\"rewrite\":",
            r#"{"Pipeline":[{"Prefix":{"from":"model.","to":"native."}},{"OrderedOptional":[{"Contains":{"from":"txt_in.t_embedder.mlp.0.","to":"txt_in.t_embedder.in_layer."}},{"Contains":{"from":"txt_in.t_embedder.mlp.2.","to":"txt_in.t_embedder.out_layer."}},{"Contains":{"from":"txt_in.c_embedder.linear_1.","to":"txt_in.c_embedder.in_layer."}},{"Contains":{"from":"txt_in.c_embedder.linear_2.","to":"txt_in.c_embedder.out_layer."}},{"Contains":{"from":"_mod.linear.","to":"_mod.lin."}},{"Contains":{"from":"_attn_qkv.","to":"_attn.qkv."}},{"Contains":{"from":"mlp.fc1.","to":"mlp.0."}},{"Contains":{"from":"mlp.fc2.","to":"mlp.2."}},{"Contains":{"from":"_attn_q_norm.weight","to":"_attn.norm.query_norm.weight"}},{"Contains":{"from":"_attn_k_norm.weight","to":"_attn.norm.key_norm.weight"}},{"Contains":{"from":".q_norm.weight","to":".norm.query_norm.weight"}},{"Contains":{"from":".k_norm.weight","to":".norm.key_norm.weight"}},{"Contains":{"from":"_attn_proj.","to":"_attn.proj."}},{"Contains":{"from":".modulation.linear.","to":".modulation.lin."}},{"Contains":{"from":"_in.mlp.2.","to":"_in.out_layer."}},{"Contains":{"from":"_in.mlp.0.","to":"_in.in_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]}]}"#,
            ",\"component\":\"model\"}},{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"vae.\"},\"minimum_matches\":0,\"maximum_matches\":16384},\"rewrite\":\"Identity\",\"component\":\"vae\"}},{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"text_encoders.\"},\"minimum_matches\":0,\"maximum_matches\":16384},\"rewrite\":{\"Prefix\":{\"from\":\"text_encoders.\",\"to\":\"text_encoder.\"}},\"component\":\"text_encoder\"}}],\"unmatched\":\"Reject\"}"
        ),
    };
pub const HUNYUAN_VIDEO_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: concat!(
            "{\"operations\":[{\"Move\":{\"selector\":{\"predicate\":{\"Any\":[{\"Prefix\":\"img_in.\"},{\"Prefix\":\"final_layer.\"},{\"Prefix\":\"txt_in.\"},{\"Prefix\":\"vector_in.\"},{\"Prefix\":\"time_r_in.\"},{\"Prefix\":\"double_blocks.\"},{\"Prefix\":\"single_blocks.\"},{\"Prefix\":\"byt5_in.\"},{\"Prefix\":\"guidance_in.\"},{\"Prefix\":\"cond_type_embedding.\"},{\"Prefix\":\"vision_in.\"},{\"Prefix\":\"time_in.\"}]},\"minimum_matches\":1,\"maximum_matches\":16384},\"rewrite\":",
            r#"{"OrderedOptional":[{"Contains":{"from":"txt_in.t_embedder.mlp.0.","to":"txt_in.t_embedder.in_layer."}},{"Contains":{"from":"txt_in.t_embedder.mlp.2.","to":"txt_in.t_embedder.out_layer."}},{"Contains":{"from":"txt_in.c_embedder.linear_1.","to":"txt_in.c_embedder.in_layer."}},{"Contains":{"from":"txt_in.c_embedder.linear_2.","to":"txt_in.c_embedder.out_layer."}},{"Contains":{"from":"_mod.linear.","to":"_mod.lin."}},{"Contains":{"from":"_attn_qkv.","to":"_attn.qkv."}},{"Contains":{"from":"mlp.fc1.","to":"mlp.0."}},{"Contains":{"from":"mlp.fc2.","to":"mlp.2."}},{"Contains":{"from":"_attn_q_norm.weight","to":"_attn.norm.query_norm.weight"}},{"Contains":{"from":"_attn_k_norm.weight","to":"_attn.norm.key_norm.weight"}},{"Contains":{"from":".q_norm.weight","to":".norm.query_norm.weight"}},{"Contains":{"from":".k_norm.weight","to":".norm.key_norm.weight"}},{"Contains":{"from":"_attn_proj.","to":"_attn.proj."}},{"Contains":{"from":".modulation.linear.","to":".modulation.lin."}},{"Contains":{"from":"_in.mlp.2.","to":"_in.out_layer."}},{"Contains":{"from":"_in.mlp.0.","to":"_in.in_layer."}},{"Suffix":{"from":".scale","to":".weight"}},{"Prefix":{"from":"img_in.","to":"native.img_in."}},{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},{"Prefix":{"from":"txt_in.","to":"native.txt_in."}},{"Prefix":{"from":"vector_in.","to":"native.vector_in."}},{"Prefix":{"from":"time_r_in.","to":"native.time_r_in."}},{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},{"Prefix":{"from":"byt5_in.","to":"native.byt5_in."}},{"Prefix":{"from":"guidance_in.","to":"native.guidance_in."}},{"Prefix":{"from":"cond_type_embedding.","to":"native.cond_type_embedding."}},{"Prefix":{"from":"vision_in.","to":"native.vision_in."}},{"Prefix":{"from":"time_in.","to":"native.time_in."}}]}"#,
            ",\"component\":\"model\"}},{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"vae.\"},\"minimum_matches\":0,\"maximum_matches\":16384},\"rewrite\":\"Identity\",\"component\":\"vae\"}},{\"Move\":{\"selector\":{\"predicate\":{\"Prefix\":\"text_encoders.\"},\"minimum_matches\":0,\"maximum_matches\":16384},\"rewrite\":{\"Prefix\":{\"from\":\"text_encoders.\",\"to\":\"text_encoder.\"}},\"component\":\"text_encoder\"}}],\"unmatched\":\"Reject\"}"
        ),
    };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanVideoVariant {
    Image21,
    Image21Refiner,
    Video,
    VideoI2V,
    VideoSkyreelsI2V,
    Video15,
    Video15SrDistilled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HunyuanVideoLayout {
    PrefixedNative,
    SavedModel,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug)]
pub struct HunyuanVideoConfiguration {
    pub variant: HunyuanVideoVariant,
    pub layout: HunyuanVideoLayout,
    pub in_channels: u64,
    pub out_channels: u64,
    pub patch_size: [u64; 3],
    pub patch_rank: usize,
    pub context_input_dimension: u64,
    pub hidden_size: u64,
    pub number_of_heads: u64,
    pub double_block_depth: usize,
    pub single_block_depth: usize,
    pub vector_input_dimension: Option<u64>,
    pub axes_dimensions: [u64; 3],
    pub guidance_embedding: bool,
    pub byt5_conditioning: bool,
    pub mean_flow: bool,
    pub condition_type_embedding: bool,
    pub vision_input_dimension: Option<u64>,
    pub mean_flow_sum: bool,
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub supported_dtypes: &'static [DType],
}

pub fn state_plan_for_layout(
    layout: HunyuanVideoLayout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        HunyuanVideoLayout::PrefixedNative => &HUNYUAN_VIDEO_PREFIXED_STATE_PLAN,
        HunyuanVideoLayout::SavedModel => &HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN,
        HunyuanVideoLayout::StandaloneNative => &HUNYUAN_VIDEO_STANDALONE_STATE_PLAN,
    }
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<HunyuanVideoConfiguration, ModelFamilyError> {
    let invalid = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "Hunyuan image/video configuration is invalid: {message}"
        ))
    };
    let domains = [
        (HunyuanVideoLayout::PrefixedNative, "model.diffusion_model."),
        (HunyuanVideoLayout::SavedModel, "model."),
        (HunyuanVideoLayout::StandaloneNative, ""),
    ];
    let mut matches = Vec::new();
    let mut partial = false;
    for (layout, prefix) in domains {
        let markers = [
            format!("{prefix}txt_in.individual_token_refiner.blocks.0.norm1.weight"),
            format!("{prefix}img_in.proj.weight"),
            format!("{prefix}final_layer.linear.weight"),
            format!("{prefix}txt_in.input_embedder.weight"),
        ];
        let present = markers
            .iter()
            .filter(|key| probe.tensor_shapes.contains_key(*key))
            .count();
        partial |= present > 0 && present < markers.len();
        if present == markers.len() {
            matches.push((layout, prefix));
        }
    }
    let (layout, prefix) = match matches.as_slice() {
        [entry] => *entry,
        [] if partial => return Err(invalid("partial marker set".to_owned())),
        [] => {
            return Err(ModelFamilyError::ModelLayoutSelection(
                "parsed tensor keys match no Hunyuan image/video source layout".to_owned(),
            ));
        }
        _ => {
            return Err(ModelFamilyError::ModelLayoutSelection(
                "parsed tensor keys ambiguously match multiple Hunyuan image/video layouts"
                    .to_owned(),
            ));
        }
    };

    for collision in ["t_block.1.weight", "y_embedder.y_embedding"] {
        if probe
            .tensor_shapes
            .contains_key(&format!("{prefix}{collision}"))
        {
            return Err(invalid("PixArt cross-family marker".to_owned()));
        }
    }

    let patch_projection = required_shape(probe, &format!("{prefix}img_in.proj.weight"), &invalid)?;
    if !(patch_projection.len() == 4 || patch_projection.len() == 5)
        || patch_projection.contains(&0)
    {
        return Err(invalid("img_in.proj.weight shape".to_owned()));
    }
    let hidden_size = patch_projection[0];
    let in_channels = patch_projection[1];
    if !hidden_size.is_multiple_of(HUNYUAN_VIDEO_HEAD_DIMENSION) {
        return Err(invalid(format!(
            "hidden size {hidden_size} is not divisible by head dimension {HUNYUAN_VIDEO_HEAD_DIMENSION}"
        )));
    }
    let number_of_heads = hidden_size / HUNYUAN_VIDEO_HEAD_DIMENSION;
    let patch_rank = patch_projection.len() - 2;
    let mut patch_size = [1, 1, 1];
    patch_size[..patch_rank].copy_from_slice(&patch_projection[2..]);
    let patch_volume = patch_projection[2..]
        .iter()
        .try_fold(1_u64, |value, dimension| value.checked_mul(*dimension))
        .ok_or_else(|| invalid("patch volume overflow".to_owned()))?;
    let output = required_shape(
        probe,
        &format!("{prefix}final_layer.linear.weight"),
        &invalid,
    )?;
    if output.len() != 2 || output[1] != hidden_size || !output[0].is_multiple_of(patch_volume) {
        return Err(invalid("final_layer.linear.weight shape".to_owned()));
    }
    let out_channels = output[0] / patch_volume;

    let context = required_shape(
        probe,
        &format!("{prefix}txt_in.input_embedder.weight"),
        &invalid,
    )?;
    if context.len() != 2 || context[0] != hidden_size || context[1] == 0 {
        return Err(invalid("txt_in.input_embedder.weight shape".to_owned()));
    }
    let double_block_depth = checked_depth(probe, &format!("{prefix}double_blocks."), &invalid)?;
    let single_block_depth = checked_depth(probe, &format!("{prefix}single_blocks."), &invalid)?;
    let has_prefix = |candidate: &str| {
        probe
            .tensor_shapes
            .keys()
            .any(|key| key.starts_with(&format!("{prefix}{candidate}")))
    };
    let vector_input_dimension =
        has_prefix("vector_in.").then_some(HUNYUAN_VIDEO_VECTOR_INPUT_DIMENSION);
    let guidance_embedding = has_prefix("guidance_in.");
    let byt5_conditioning = probe
        .tensor_shapes
        .contains_key(&format!("{prefix}byt5_in.fc1.weight"));
    let mean_flow = has_prefix("time_r_in.");
    let condition_type_embedding = probe
        .tensor_shapes
        .contains_key(&format!("{prefix}cond_type_embedding.weight"));
    let vision_input_dimension = probe
        .tensor_shapes
        .get(&format!("{prefix}vision_in.proj.0.weight"))
        .map(|shape| match shape.as_slice() {
            [dimension] if *dimension > 0 => Ok(*dimension),
            _ => Err(invalid("vision_in.proj.0.weight shape".to_owned())),
        })
        .transpose()?;
    let mean_flow_sum = vision_input_dimension.is_some();

    let (
        variant,
        sampling_shift,
        memory_usage_factor,
        latent_format,
        clip_target,
        supported_dtypes,
    ) = if vision_input_dimension.is_some() && in_channels == 98 {
        (
            HunyuanVideoVariant::Video15SrDistilled,
            2.0,
            4.0,
            HUNYUAN_VIDEO15_LATENT_FORMAT,
            &HUNYUAN_VIDEO15_CLIP_TARGET,
            HUNYUAN_VIDEO15_SUPPORTED_DTYPES,
        )
    } else if vision_input_dimension.is_some() {
        (
            HunyuanVideoVariant::Video15,
            7.0,
            4.0,
            HUNYUAN_VIDEO15_LATENT_FORMAT,
            &HUNYUAN_VIDEO15_CLIP_TARGET,
            HUNYUAN_VIDEO15_SUPPORTED_DTYPES,
        )
    } else if patch_rank == 2 && vector_input_dimension.is_none() {
        (
            HunyuanVideoVariant::Image21,
            5.0,
            8.7,
            HUNYUAN_IMAGE21_LATENT_FORMAT,
            &HUNYUAN_IMAGE_CLIP_TARGET,
            HUNYUAN_VIDEO_SUPPORTED_DTYPES,
        )
    } else if patch_size == [1, 1, 1] && vector_input_dimension.is_none() {
        (
            HunyuanVideoVariant::Image21Refiner,
            4.0,
            1.8,
            HUNYUAN_IMAGE21_REFINER_LATENT_FORMAT,
            &HUNYUAN_VIDEO_CLIP_TARGET,
            HUNYUAN_VIDEO_SUPPORTED_DTYPES,
        )
    } else if in_channels == 33 {
        (
            HunyuanVideoVariant::VideoI2V,
            7.0,
            1.8,
            HUNYUAN_VIDEO_LATENT_FORMAT,
            &HUNYUAN_VIDEO_CLIP_TARGET,
            HUNYUAN_VIDEO_SUPPORTED_DTYPES,
        )
    } else if in_channels == 32 {
        (
            HunyuanVideoVariant::VideoSkyreelsI2V,
            7.0,
            1.8,
            HUNYUAN_VIDEO_LATENT_FORMAT,
            &HUNYUAN_VIDEO_CLIP_TARGET,
            HUNYUAN_VIDEO_SUPPORTED_DTYPES,
        )
    } else {
        (
            HunyuanVideoVariant::Video,
            7.0,
            1.8,
            HUNYUAN_VIDEO_LATENT_FORMAT,
            &HUNYUAN_VIDEO_CLIP_TARGET,
            HUNYUAN_VIDEO_SUPPORTED_DTYPES,
        )
    };
    if matches!(
        variant,
        HunyuanVideoVariant::Video15 | HunyuanVideoVariant::Video15SrDistilled
    ) && vision_input_dimension != Some(HUNYUAN_VIDEO15_VISION_INPUT_DIMENSION)
    {
        return Err(invalid(format!(
            "vision input dimension {:?}; expected {HUNYUAN_VIDEO15_VISION_INPUT_DIMENSION}",
            vision_input_dimension
        )));
    }
    let axes_dimensions = if patch_rank == 2 {
        [0, 64, 64]
    } else {
        [16, 56, 56]
    };
    if axes_dimensions.iter().sum::<u64>() != hidden_size / number_of_heads {
        return Err(invalid(
            "rank-dependent axes do not match head width".to_owned(),
        ));
    }

    Ok(HunyuanVideoConfiguration {
        variant,
        layout,
        in_channels,
        out_channels,
        patch_size,
        patch_rank,
        context_input_dimension: context[1],
        hidden_size,
        number_of_heads,
        double_block_depth,
        single_block_depth,
        vector_input_dimension,
        axes_dimensions,
        guidance_embedding,
        byt5_conditioning,
        mean_flow,
        condition_type_embedding,
        vision_input_dimension,
        mean_flow_sum,
        sampling_shift,
        memory_usage_factor,
        latent_format,
        clip_target,
        supported_dtypes,
    })
}

pub fn augment_refiner_conditioning(
    backend: &CpuBackend,
    model_latent: &Tensor,
    output_height: u64,
    output_width: u64,
    noise_augmentation: f32,
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ModelFamilyError> {
    context.cancellation.check()?;
    if !noise_augmentation.is_finite() || !(0.0..=1.0).contains(&noise_augmentation) {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "Hunyuan refiner noise augmentation must be finite and in 0..=1".to_owned(),
        ));
    }
    let resized = resize_with_context_exact_native(
        backend,
        model_latent,
        output_height,
        output_width,
        ResizeMode::Bilinear,
        false,
        context,
    )?;
    let image_scale = (1.0 - noise_augmentation).min(HUNYUAN_REFINER_IMAGE_SCALE);
    let scaled_image = scale_tensor(backend, &resized, image_scale, context)?;
    if noise_augmentation == 0.0 {
        return Ok(scaled_image);
    }
    let random =
        randn_like_with_context_exact_native(backend, &resized, transaction.clone(), context)?;
    let scaled_noise = scale_tensor(backend, &random.tensor, noise_augmentation, context)?;
    let output = add_method_with_context_exact_native(
        backend,
        &scaled_image,
        ElementwiseOperand::Tensor(&scaled_noise),
        1.0,
        context,
    )?;
    context.cancellation.check()?;
    *transaction = random.transaction;
    Ok(output)
}

fn scale_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ModelFamilyError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    Ok(backend
        .binary_scalar(
            BinaryOperation::Multiply,
            input,
            Scalar::Float(f64::from(scale)),
            ScalarSide::Right,
            descriptor,
            context,
        )?
        .0)
}

fn required_shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing {key}")))
}

fn checked_depth(
    probe: &ModelProbe,
    prefix: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<usize, ModelFamilyError> {
    let mut indices = probe
        .tensor_shapes
        .keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)
                .and_then(|suffix| suffix.split('.').next())
                .and_then(|index| index.parse::<usize>().ok())
        })
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if indices.is_empty() || indices.len() > 256 {
        return Err(invalid(format!("{prefix} block count")));
    }
    if indices.iter().copied().ne(0..indices.len()) {
        return Err(invalid(format!(
            "{prefix} block indices are not consecutive"
        )));
    }
    Ok(indices.len())
}
