use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyError, ModelForwardOperation, ModelForwardStep,
    ModelProbe, ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const LUMINA_DIMENSION: u64 = 2_304;
pub const ZIMAGE_DIMENSION: u64 = 3_840;
pub const LUMINA_HEAD_COUNT: u64 = 24;
pub const LUMINA_KV_HEAD_COUNT: u64 = 8;
pub const ZIMAGE_HEAD_COUNT: u64 = 30;
pub const ZIMAGE_KV_HEAD_COUNT: u64 = 30;
pub const LUMINA_PATCH_SIZE: u64 = 2;
pub const LUMINA_INPUT_CHANNELS: u64 = 16;
pub const LUMINA_SAMPLING_SHIFT: f64 = 6.0;
pub const ZIMAGE_SAMPLING_SHIFT: f64 = 3.0;
pub const LUMINA_MEMORY_USAGE_FACTOR: f64 = 1.4;
pub const ZIMAGE_MEMORY_USAGE_FACTOR: f64 = 2.8;
pub const ZIMAGE_PIXEL_MEMORY_USAGE_FACTOR: f64 = 0.03;
pub const ZIMAGE_PAD_TOKENS_MULTIPLE: u64 = 32;
pub const ZIMAGE_TIME_SCALE: f64 = 1_000.0;
pub const LUMINA_ROPE_THETA: f64 = 10_000.0;
pub const ZIMAGE_ROPE_THETA: f64 = 256.0;
pub const LUMINA_AXES_DIMENSIONS: &[u64] = &[32, 32, 32];
pub const LUMINA_AXES_LENGTHS: &[u64] = &[300, 512, 512];
pub const ZIMAGE_AXES_DIMENSIONS: &[u64] = &[32, 48, 48];
pub const ZIMAGE_AXES_LENGTHS: &[u64] = &[1_536, 512, 512];
pub const LUMINA_MAX_LAYER_COUNT: usize = 256;
pub const ZIMAGE_MAX_PIXEL_PATCH_SIZE: u64 = 64;
pub const ZIMAGE_MAX_DECODER_FREQUENCIES: u64 = 4_096;

pub const LUMINA_ZIMAGE_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_flux_sim_model_0029::LATENT_FORMAT;
pub const ZIMAGE_PIXEL_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_zimagepixelspace_sim_model_0055::LATENT_FORMAT;

pub const LUMINA_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const LUMINA_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.lumina2.LuminaTokenizer",
        clip_model: "sim.text_encoders.lumina2.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: LUMINA_CLIP_CONFIGURATION,
        },
    }];
pub static LUMINA_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: LUMINA_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const ZIMAGE_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.hunyuan_video.llama_detect",
    }];
pub const ZIMAGE_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.z_image.ZImageTokenizer",
        clip_model: "sim.text_encoders.z_image.te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: ZIMAGE_CLIP_CONFIGURATION,
        },
    }];
pub static ZIMAGE_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: ZIMAGE_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LuminaZImageVariant {
    Lumina2,
    ZImage,
    ZImagePixelSpace,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LuminaZImageLayout {
    PrefixedNative,
    SavedModel,
    StandaloneNative,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LuminaZImageConditioningFact {
    OptionalAttentionMask,
    OmitAllOnesAttentionMask,
    NumTokensFromMaskAtLeastOne,
    CrossAttention,
    NumTokensFromCrossAttention,
    OptionalPooledText,
    OptionalSiglipVisionFeatures,
    OptionalReferenceLatents,
    ProcessReferenceLatentsWithCanonicalFormat,
    OptionalReferenceContexts,
    ReferenceLatentsAffectMemoryEstimate,
}

pub const LUMINA_ZIMAGE_CONDITIONING: &[LuminaZImageConditioningFact] = &[
    LuminaZImageConditioningFact::OptionalAttentionMask,
    LuminaZImageConditioningFact::OmitAllOnesAttentionMask,
    LuminaZImageConditioningFact::NumTokensFromMaskAtLeastOne,
    LuminaZImageConditioningFact::CrossAttention,
    LuminaZImageConditioningFact::NumTokensFromCrossAttention,
    LuminaZImageConditioningFact::OptionalPooledText,
    LuminaZImageConditioningFact::OptionalSiglipVisionFeatures,
    LuminaZImageConditioningFact::OptionalReferenceLatents,
    LuminaZImageConditioningFact::ProcessReferenceLatentsWithCanonicalFormat,
    LuminaZImageConditioningFact::OptionalReferenceContexts,
    LuminaZImageConditioningFact::ReferenceLatentsAffectMemoryEstimate,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZImagePixelDecoderConfiguration {
    pub input_channels: u64,
    pub hidden_size: u64,
    pub number_of_residual_blocks: usize,
    pub maximum_frequencies: u64,
    pub uses_x0: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct LuminaZImageConfiguration {
    pub variant: LuminaZImageVariant,
    pub layout: LuminaZImageLayout,
    pub dimension: u64,
    pub caption_feature_dimension: u64,
    pub number_of_layers: usize,
    pub number_of_heads: u64,
    pub number_of_kv_heads: u64,
    pub axes_dimensions: &'static [u64],
    pub axes_lengths: &'static [u64],
    pub rope_theta: f64,
    pub feed_forward_multiplier: f64,
    pub patch_size: u64,
    pub input_channels: u64,
    pub output_channels: u64,
    pub qk_norm: bool,
    pub zimage_modulation: bool,
    pub time_scale: Option<f64>,
    pub pad_tokens_multiple: Option<u64>,
    pub clip_text_dimension: Option<u64>,
    pub siglip_feature_dimension: Option<u64>,
    pub pixel_decoder: Option<ZImagePixelDecoderConfiguration>,
    pub sampling_shift: f64,
    pub memory_usage_factor: f64,
    pub supported_dtypes: &'static [DType],
    pub conditioning: &'static [LuminaZImageConditioningFact],
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
}

pub const LUMINA_ZIMAGE_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Lumina2, ZImage, or ZImage pixel-space diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Gemma2 or Qwen3 text conditioning encoder",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "optional Flux latent codec; omitted by pixel-space ZImage",
        required: false,
    },
];

pub const LUMINA_ZIMAGE_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.cap_embedder.1.weight",
    "native.noise_refiner.0.attention.k_norm.weight",
    "native.x_embedder.weight",
    "native.final_layer.linear.weight",
];
pub const LUMINA_ZIMAGE_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.cap_pad_token",
    "native.clip_text_pooled_proj.0.weight",
    "native.siglip_embedder.0.weight",
    "native.dec_net.cond_embed.weight",
    "native.__x0__",
];
pub const LUMINA_ZIMAGE_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: LUMINA_ZIMAGE_MODEL_REQUIRED_KEYS,
        optional_keys: LUMINA_ZIMAGE_MODEL_OPTIONAL_KEYS,
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

pub const LUMINA_ZIMAGE_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const LUMINA_ZIMAGE_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const LUMINA_ZIMAGE_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "input.patch_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.x_embedder.weight",
            bias: Some("native.x_embedder.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "caption.projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.cap_embedder.1.weight",
            bias: Some("native.cap_embedder.1.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "transformer.activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "output.patch_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: Some("native.final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const LUMINA_ZIMAGE_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const LUMINA_ZIMAGE_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Any":[{"Prefix":"cap_embedder."},{"Prefix":"noise_refiner."},{"Prefix":"context_refiner."},{"Prefix":"layers."},{"Prefix":"x_embedder."},{"Prefix":"t_embedder."},{"Prefix":"time_text_embed."},{"Prefix":"clip_text_pooled_proj."},{"Prefix":"siglip_embedder."},{"Prefix":"siglip_refiner."},{"Prefix":"final_layer."},{"Prefix":"dec_net."},{"Prefix":"norm_final."},{"Prefix":"cap_pad_token"},{"Prefix":"siglip_pad_token"},{"Prefix":"x_pad_token"},{"Prefix":"__x0__"}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"OrderedOptional":[{"Prefix":{"from":"cap_embedder.","to":"native.cap_embedder."}},{"Prefix":{"from":"noise_refiner.","to":"native.noise_refiner."}},{"Prefix":{"from":"context_refiner.","to":"native.context_refiner."}},{"Prefix":{"from":"layers.","to":"native.layers."}},{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},{"Prefix":{"from":"t_embedder.","to":"native.t_embedder."}},{"Prefix":{"from":"time_text_embed.","to":"native.time_text_embed."}},{"Prefix":{"from":"clip_text_pooled_proj.","to":"native.clip_text_pooled_proj."}},{"Prefix":{"from":"siglip_embedder.","to":"native.siglip_embedder."}},{"Prefix":{"from":"siglip_refiner.","to":"native.siglip_refiner."}},{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},{"Prefix":{"from":"dec_net.","to":"native.dec_net."}},{"Prefix":{"from":"norm_final.","to":"native.norm_final."}},{"Prefix":{"from":"cap_pad_token","to":"native.cap_pad_token"}},{"Prefix":{"from":"siglip_pad_token","to":"native.siglip_pad_token"}},{"Prefix":{"from":"x_pad_token","to":"native.x_pad_token"}},{"Prefix":{"from":"__x0__","to":"native.__x0__"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

// This is the pinned z_image_to_diffusers mapping boundary. Canonical entry and
// exit tensors are copied to native names used by the common program; every
// converted block tensor remains namespaced so no second loader owns it.
pub const ZIMAGE_DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Copy":{"selector":{"predicate":{"Exact":"all_x_embedder.2-1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.x_embedder.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"all_x_embedder.2-1.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.x_embedder.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"cap_embedder.1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.cap_embedder.1.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"cap_embedder.1.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.cap_embedder.1.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"noise_refiner.0.attention.norm_k.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.noise_refiner.0.attention.k_norm.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"all_final_layer.2-1.linear.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.final_layer.linear.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"all_final_layer.2-1.linear.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.final_layer.linear.bias"},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Any":[{"Prefix":"layers."},{"Prefix":"context_refiner."},{"Prefix":"noise_refiner."},{"Prefix":"all_x_embedder."},{"Prefix":"all_final_layer."},{"Prefix":"cap_embedder."},{"Prefix":"t_embedder."},{"Prefix":"x_pad_token"},{"Prefix":"cap_pad_token"}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"OrderedOptional":[{"Prefix":{"from":"layers.","to":"native.diffusers.layers."}},{"Prefix":{"from":"context_refiner.","to":"native.diffusers.context_refiner."}},{"Prefix":{"from":"noise_refiner.","to":"native.diffusers.noise_refiner."}},{"Prefix":{"from":"all_x_embedder.","to":"native.diffusers.all_x_embedder."}},{"Prefix":{"from":"all_final_layer.","to":"native.diffusers.all_final_layer."}},{"Prefix":{"from":"cap_embedder.","to":"native.diffusers.cap_embedder."}},{"Prefix":{"from":"t_embedder.","to":"native.diffusers.t_embedder."}},{"Prefix":{"from":"x_pad_token","to":"native.diffusers.x_pad_token"}},{"Prefix":{"from":"cap_pad_token","to":"native.diffusers.cap_pad_token"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

#[derive(Clone, Copy, Debug)]
pub struct LuminaZImageCommonMapping {
    pub components: &'static [ModelFamilyComponent],
    pub component_state_schemas: &'static [ModelFamilyComponentStateSchema],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub forward_program: &'static [ModelForwardStep],
}

pub static LUMINA_ZIMAGE_COMMON_MAPPING: LuminaZImageCommonMapping = LuminaZImageCommonMapping {
    components: LUMINA_ZIMAGE_COMPONENTS,
    component_state_schemas: LUMINA_ZIMAGE_COMPONENT_STATE_SCHEMAS,
    supported_dtypes: LUMINA_ZIMAGE_SUPPORTED_DTYPES,
    supported_devices: LUMINA_ZIMAGE_SUPPORTED_DEVICES,
    forward_program: LUMINA_ZIMAGE_FORWARD_PROGRAM,
};

pub fn common_mapping() -> &'static LuminaZImageCommonMapping {
    &LUMINA_ZIMAGE_COMMON_MAPPING
}

pub fn state_plan_for_layout(
    layout: LuminaZImageLayout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        LuminaZImageLayout::PrefixedNative => &LUMINA_ZIMAGE_PREFIXED_STATE_PLAN,
        LuminaZImageLayout::SavedModel => &LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN,
        LuminaZImageLayout::StandaloneNative => &LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
        LuminaZImageLayout::Diffusers => &ZIMAGE_DIFFUSERS_STATE_PLAN,
    }
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<LuminaZImageConfiguration, ModelFamilyError> {
    let (layout, prefix) = select_layout(probe)?;
    let diffusers = layout == LuminaZImageLayout::Diffusers;
    let cap_key = format!("{prefix}cap_embedder.1.weight");
    let x_key = if diffusers {
        "all_x_embedder.2-1.weight".to_owned()
    } else {
        format!("{prefix}x_embedder.weight")
    };
    let final_key = if diffusers {
        "all_final_layer.2-1.linear.weight".to_owned()
    } else {
        format!("{prefix}final_layer.linear.weight")
    };
    let cap = required_matrix(probe, &cap_key)?;
    let dimension = cap[0];
    let caption_feature_dimension = cap[1];
    let x = required_matrix(probe, &x_key)?;
    let final_projection = required_matrix(probe, &final_key)?;
    if x[0] != dimension || final_projection[1] != dimension {
        return Err(invalid(format!(
            "projection shapes {x:?} and {final_projection:?} contradict dimension {dimension}"
        )));
    }

    let block_key = if diffusers {
        format!("{prefix}layers.{{}}.attention.to_q.weight")
    } else {
        format!("{prefix}layers.{{}}.attention.qkv.weight")
    };
    let number_of_layers = checked_block_count(probe, &block_key, "transformer")?;

    let pixel_marker = format!("{prefix}dec_net.cond_embed.weight");
    let has_pixel_decoder = probe.tensor_shapes.contains_key(&pixel_marker);
    let variant = if has_pixel_decoder {
        if diffusers {
            return Err(invalid(
                "ZImagePixelSpace does not admit the pinned Diffusers conversion".to_owned(),
            ));
        }
        if dimension != ZIMAGE_DIMENSION {
            return Err(invalid(format!(
                "pixel decoder requires ZImage dimension {ZIMAGE_DIMENSION}, found {dimension}"
            )));
        }
        LuminaZImageVariant::ZImagePixelSpace
    } else if dimension == ZIMAGE_DIMENSION {
        LuminaZImageVariant::ZImage
    } else if dimension == LUMINA_DIMENSION {
        if diffusers {
            return Err(invalid(
                "Lumina2 does not admit the pinned ZImage Diffusers conversion".to_owned(),
            ));
        }
        LuminaZImageVariant::Lumina2
    } else {
        return Err(invalid(format!(
            "caption projection dimension {dimension} is neither Lumina2 {LUMINA_DIMENSION} nor ZImage {ZIMAGE_DIMENSION}"
        )));
    };

    let (
        number_of_heads,
        number_of_kv_heads,
        axes_dimensions,
        axes_lengths,
        rope_theta,
        feed_forward_multiplier,
        sampling_shift,
        memory_usage_factor,
        latent_format,
        clip_target,
    ) = match variant {
        LuminaZImageVariant::Lumina2 => (
            LUMINA_HEAD_COUNT,
            LUMINA_KV_HEAD_COUNT,
            LUMINA_AXES_DIMENSIONS,
            LUMINA_AXES_LENGTHS,
            LUMINA_ROPE_THETA,
            4.0,
            LUMINA_SAMPLING_SHIFT,
            LUMINA_MEMORY_USAGE_FACTOR,
            LUMINA_ZIMAGE_LATENT_FORMAT,
            &LUMINA_CLIP_TARGET,
        ),
        LuminaZImageVariant::ZImage => (
            ZIMAGE_HEAD_COUNT,
            ZIMAGE_KV_HEAD_COUNT,
            ZIMAGE_AXES_DIMENSIONS,
            ZIMAGE_AXES_LENGTHS,
            ZIMAGE_ROPE_THETA,
            8.0 / 3.0,
            ZIMAGE_SAMPLING_SHIFT,
            ZIMAGE_MEMORY_USAGE_FACTOR,
            LUMINA_ZIMAGE_LATENT_FORMAT,
            &ZIMAGE_CLIP_TARGET,
        ),
        LuminaZImageVariant::ZImagePixelSpace => (
            ZIMAGE_HEAD_COUNT,
            ZIMAGE_KV_HEAD_COUNT,
            ZIMAGE_AXES_DIMENSIONS,
            ZIMAGE_AXES_LENGTHS,
            ZIMAGE_ROPE_THETA,
            8.0 / 3.0,
            ZIMAGE_SAMPLING_SHIFT,
            ZIMAGE_PIXEL_MEMORY_USAGE_FACTOR,
            ZIMAGE_PIXEL_LATENT_FORMAT,
            &ZIMAGE_CLIP_TARGET,
        ),
    };

    let (patch_size, input_channels, output_channels, pixel_decoder) = match variant {
        LuminaZImageVariant::ZImagePixelSpace => {
            pixel_configuration(probe, prefix, x, final_projection)?
        }
        _ => {
            let expected_width = LUMINA_PATCH_SIZE
                .checked_mul(LUMINA_PATCH_SIZE)
                .and_then(|value| value.checked_mul(LUMINA_INPUT_CHANNELS))
                .ok_or(ModelFamilyError::MemoryOverflow)?;
            if x[1] != expected_width || final_projection[0] != expected_width {
                return Err(invalid(format!(
                    "latent projection widths {} and {} must both equal {expected_width}",
                    x[1], final_projection[0]
                )));
            }
            (
                LUMINA_PATCH_SIZE,
                LUMINA_INPUT_CHANNELS,
                LUMINA_INPUT_CHANNELS,
                None,
            )
        }
    };

    validate_attention_projection(
        probe,
        prefix,
        layout,
        dimension,
        number_of_heads,
        number_of_kv_heads,
    )?;

    let pad_tokens_multiple = probe
        .tensor_shapes
        .contains_key(&format!("{prefix}cap_pad_token"))
        .then_some(ZIMAGE_PAD_TOKENS_MULTIPLE);
    let clip_text_dimension =
        optional_matrix_first_dimension(probe, &format!("{prefix}clip_text_pooled_proj.0.weight"))?;
    let siglip_feature_dimension =
        optional_matrix_first_dimension(probe, &format!("{prefix}siglip_embedder.0.weight"))?;

    Ok(LuminaZImageConfiguration {
        variant,
        layout,
        dimension,
        caption_feature_dimension,
        number_of_layers,
        number_of_heads,
        number_of_kv_heads,
        axes_dimensions,
        axes_lengths,
        rope_theta,
        feed_forward_multiplier,
        patch_size,
        input_channels,
        output_channels,
        qk_norm: true,
        zimage_modulation: variant != LuminaZImageVariant::Lumina2,
        time_scale: (variant != LuminaZImageVariant::Lumina2).then_some(ZIMAGE_TIME_SCALE),
        pad_tokens_multiple,
        clip_text_dimension,
        siglip_feature_dimension,
        pixel_decoder,
        sampling_shift,
        memory_usage_factor,
        supported_dtypes: LUMINA_ZIMAGE_SUPPORTED_DTYPES,
        conditioning: LUMINA_ZIMAGE_CONDITIONING,
        latent_format,
        clip_target,
    })
}

fn select_layout(
    probe: &ModelProbe,
) -> Result<(LuminaZImageLayout, &'static str), ModelFamilyError> {
    let native_required = [
        "cap_embedder.1.weight",
        "noise_refiner.0.attention.k_norm.weight",
        "x_embedder.weight",
        "final_layer.linear.weight",
        "layers.0.attention.qkv.weight",
    ];
    let native_candidates = [
        (LuminaZImageLayout::PrefixedNative, "model.diffusion_model."),
        (LuminaZImageLayout::SavedModel, "model."),
        (LuminaZImageLayout::StandaloneNative, ""),
    ];
    let mut matches = native_candidates
        .into_iter()
        .filter(|(_, prefix)| {
            native_required
                .iter()
                .all(|key| probe.tensor_shapes.contains_key(&format!("{prefix}{key}")))
        })
        .collect::<Vec<_>>();
    let diffusers_required = [
        "cap_embedder.1.weight",
        "noise_refiner.0.attention.norm_k.weight",
        "noise_refiner.0.attention.to_q.weight",
        "all_x_embedder.2-1.weight",
        "all_final_layer.2-1.linear.weight",
        "layers.0.attention.to_q.weight",
    ];
    if diffusers_required
        .iter()
        .all(|key| probe.tensor_shapes.contains_key(*key))
    {
        matches.push((LuminaZImageLayout::Diffusers, ""));
    }
    match matches.as_slice() {
        [selected] => Ok(*selected),
        [] => Err(ModelFamilyError::ModelLayoutSelection(
            "no exact Lumina2/ZImage native or pinned ZImage Diffusers layout matched".to_owned(),
        )),
        _ => Err(ModelFamilyError::ModelLayoutSelection(
            "Lumina2/ZImage probe ambiguously matches multiple layouts".to_owned(),
        )),
    }
}

fn validate_attention_projection(
    probe: &ModelProbe,
    prefix: &str,
    layout: LuminaZImageLayout,
    dimension: u64,
    number_of_heads: u64,
    number_of_kv_heads: u64,
) -> Result<(), ModelFamilyError> {
    if !dimension.is_multiple_of(number_of_heads) {
        return Err(invalid(format!(
            "dimension {dimension} is not divisible by {number_of_heads} heads"
        )));
    }
    let head_dimension = dimension / number_of_heads;
    let kv_dimension = head_dimension
        .checked_mul(number_of_kv_heads)
        .ok_or(ModelFamilyError::MemoryOverflow)?;
    if layout == LuminaZImageLayout::Diffusers {
        for (name, output) in [
            ("to_q", dimension),
            ("to_k", kv_dimension),
            ("to_v", kv_dimension),
        ] {
            let key = format!("{prefix}noise_refiner.0.attention.{name}.weight");
            let shape = required_matrix(probe, &key)?;
            if shape != [output, dimension] {
                return Err(invalid(format!(
                    "{key} shape {shape:?} must be [{output}, {dimension}]"
                )));
            }
        }
    } else {
        let key = format!("{prefix}noise_refiner.0.attention.qkv.weight");
        let shape = required_matrix(probe, &key)?;
        let output = dimension
            .checked_add(
                kv_dimension
                    .checked_mul(2)
                    .ok_or(ModelFamilyError::MemoryOverflow)?,
            )
            .ok_or(ModelFamilyError::MemoryOverflow)?;
        if shape != [output, dimension] {
            return Err(invalid(format!(
                "{key} shape {shape:?} must be [{output}, {dimension}]"
            )));
        }
    }
    Ok(())
}

fn pixel_configuration(
    probe: &ModelProbe,
    prefix: &str,
    x: &[u64],
    final_projection: &[u64],
) -> Result<(u64, u64, u64, Option<ZImagePixelDecoderConfiguration>), ModelFamilyError> {
    if !x[1].is_multiple_of(3) {
        return Err(invalid(format!(
            "pixel x_embedder input width {} is not divisible by RGB channels",
            x[1]
        )));
    }
    let patch_area = x[1] / 3;
    let patch_size =
        exact_square_root(patch_area, ZIMAGE_MAX_PIXEL_PATCH_SIZE, "pixel patch area")?;
    if final_projection[0] != x[1] {
        return Err(invalid(format!(
            "pixel final projection width {} does not equal x_embedder width {}",
            final_projection[0], x[1]
        )));
    }
    let cond_key = format!("{prefix}dec_net.cond_embed.weight");
    let cond = required_matrix(probe, &cond_key)?;
    let final_key = format!("{prefix}dec_net.final_layer.linear.weight");
    let decoder_final = required_matrix(probe, &final_key)?;
    let input_key = format!("{prefix}dec_net.input_embedder.embedder.0.weight");
    let input_embedder = required_matrix(probe, &input_key)?;
    let decoder_input_channels = decoder_final[0];
    if decoder_input_channels != x[1] || decoder_final[1] != cond[0] {
        return Err(invalid(format!(
            "pixel decoder final shape {decoder_final:?} contradicts patch width {} and hidden size {}",
            x[1], cond[0]
        )));
    }
    let frequency_area = input_embedder[1]
        .checked_sub(decoder_input_channels)
        .ok_or_else(|| {
            invalid("pixel decoder embedding width is smaller than its input channels".to_owned())
        })?;
    let maximum_frequencies = exact_square_root(
        frequency_area,
        ZIMAGE_MAX_DECODER_FREQUENCIES,
        "pixel decoder frequency area",
    )?;
    let block_pattern = format!("{prefix}dec_net.res_blocks.{{}}.in_ln.weight");
    let number_of_residual_blocks = checked_block_count(probe, &block_pattern, "pixel decoder")?;
    let uses_x0 = probe.tensor_shapes.contains_key(&format!("{prefix}__x0__"));
    Ok((
        patch_size,
        3,
        3,
        Some(ZImagePixelDecoderConfiguration {
            input_channels: decoder_input_channels,
            hidden_size: cond[0],
            number_of_residual_blocks,
            maximum_frequencies,
            uses_x0,
        }),
    ))
}

fn exact_square_root(value: u64, maximum: u64, label: &str) -> Result<u64, ModelFamilyError> {
    if value == 0 {
        return Err(invalid(format!("{label} must be non-zero")));
    }
    let mut root = (value as f64).sqrt() as u64;
    while root.checked_mul(root).is_some_and(|square| square > value) {
        root -= 1;
    }
    while root
        .checked_add(1)
        .and_then(|next| next.checked_mul(next))
        .is_some_and(|square| square <= value)
    {
        root += 1;
    }
    if root > maximum || root.checked_mul(root) != Some(value) {
        return Err(invalid(format!(
            "{label} {value} is not a perfect square within 1..={maximum}"
        )));
    }
    Ok(root)
}

fn checked_block_count(
    probe: &ModelProbe,
    pattern: &str,
    label: &str,
) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    if count == 0 || count > LUMINA_MAX_LAYER_COUNT {
        return Err(invalid(format!(
            "{label} block count {count} is outside 1..={LUMINA_MAX_LAYER_COUNT}"
        )));
    }
    let (stem, suffix) = pattern
        .split_once("{}")
        .ok_or_else(|| invalid(format!("{label} block pattern has no placeholder")))?;
    let has_gap_or_later = probe.tensor_shapes.keys().any(|key| {
        key.strip_prefix(stem)
            .and_then(|tail| tail.strip_suffix(suffix))
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= count)
    });
    if has_gap_or_later {
        return Err(invalid(format!(
            "{label} blocks are not a consecutive bounded sequence"
        )));
    }
    Ok(count)
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

fn optional_matrix_first_dimension(
    probe: &ModelProbe,
    key: &str,
) -> Result<Option<u64>, ModelFamilyError> {
    probe
        .tensor_shapes
        .contains_key(key)
        .then(|| required_matrix(probe, key).map(|shape| shape[0]))
        .transpose()
}

fn invalid(message: String) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "Lumina2/ZImage configuration is invalid: {message}"
    ))
}
