use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyError, ModelFamilyStatePlanCase, ModelForwardOperation, ModelForwardStep,
    ModelLayoutSignature, ModelProbe, ModelStateLayout, ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const SDXL_MEMORY_USAGE_FACTOR: f64 = 0.8;
pub const SDXL_REFINER_MEMORY_USAGE_FACTOR: f64 = 1.0;
pub const SDXL_MODEL_CHANNELS: u64 = 320;
pub const SDXL_REFINER_MODEL_CHANNELS: u64 = 384;
pub const SDXL_CONTEXT_DIMENSION: u64 = 2_048;
pub const SDXL_REFINER_CONTEXT_DIMENSION: u64 = 1_280;
pub const SDXL_ADM_INPUT_DIMENSION: u64 = 2_816;
pub const SDXL_REFINER_ADM_INPUT_DIMENSION: u64 = 2_560;
pub const SDXL_ATTENTION_HEAD_CHANNELS: u64 = 64;

pub const SDXL_TRANSFORMER_DEPTH: &[usize] = &[0, 0, 2, 2, 10, 10];
pub const SDXL_TRANSFORMER_DEPTH_OUTPUT: &[usize] = &[0, 0, 0, 2, 2, 2, 10, 10, 10];
pub const SDXL_REFINER_TRANSFORMER_DEPTH: &[usize] = &[0, 0, 4, 4, 4, 4, 0, 0];
pub const SDXL_REFINER_TRANSFORMER_DEPTH_OUTPUT: &[usize] = &[0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0];
pub const SDXL_SSD1B_TRANSFORMER_DEPTH: &[usize] = &[0, 0, 2, 2, 4, 4];
pub const SDXL_SSD1B_TRANSFORMER_DEPTH_OUTPUT: &[usize] = &[0, 0, 0, 1, 1, 2, 10, 4, 4];
pub const SDXL_SEGMIND_TRANSFORMER_DEPTH: &[usize] = &[0, 0, 1, 1, 2, 2];
pub const SDXL_SEGMIND_TRANSFORMER_DEPTH_OUTPUT: &[usize] = &[0, 0, 0, 1, 1, 1, 2, 2, 2];
pub const SDXL_KOALA_700M_TRANSFORMER_DEPTH: &[usize] = &[0, 2, 5];
pub const SDXL_KOALA_700M_TRANSFORMER_DEPTH_OUTPUT: &[usize] = &[0, 0, 2, 2, 5, 5];
pub const SDXL_KOALA_1B_TRANSFORMER_DEPTH: &[usize] = &[0, 2, 6];
pub const SDXL_KOALA_1B_TRANSFORMER_DEPTH_OUTPUT: &[usize] = &[0, 0, 2, 2, 6, 6];

pub const SDXL_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_sdxl_sim_model_0047::LATENT_FORMAT;

pub const SDXL_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.sdxl_clip.SDXLTokenizer",
        clip_model: "sim.sdxl_clip.SDXLClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
pub const SDXL_REFINER_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.sdxl_clip.SDXLTokenizer",
        clip_model: "sim.sdxl_clip.SDXLRefinerClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

pub static SDXL_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: SDXL_CLIP_CANDIDATES,
    dynamic_selection: false,
};
pub static SDXL_REFINER_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: SDXL_REFINER_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const SDXL_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "SDXL latent diffusion U-Net",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SDXL CLIP-L/CLIP-G conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "SDXL latent codec",
        required: false,
    },
];

pub const SDXL_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.input_blocks.0.0.weight",
    "native.time_embed.0.weight",
    "native.label_emb.0.0.weight",
    "native.out.2.weight",
];
pub const SDXL_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.time_embed.0.bias",
    "native.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight",
    "native.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight",
    "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
    "native.out.2.bias",
];
pub const SDXL_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: SDXL_MODEL_REQUIRED_KEYS,
        optional_keys: SDXL_MODEL_OPTIONAL_KEYS,
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

pub const SDXL_SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const SDXL_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const SDXL_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "conditioning.timestep_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embed.0.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "unet.residual_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "output.noise_prediction",
        operation: ModelForwardOperation::Linear {
            weight: "native.out.2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
];

pub const SDXL_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.embedders.0.transformer.text_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.embedders.0.transformer.text_model.","to":"clip_l.transformer.text_model."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.embedders.0.model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_g."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.embedders.1.model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.embedders.1.model.","to":"clip_g."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const SDXL_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"input_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"input_blocks.","to":"native.input_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_embed."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_embed.","to":"native.time_embed."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"label_emb."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"label_emb.","to":"native.label_emb."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"middle_block."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"middle_block.","to":"native.middle_block."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"output_blocks."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"output_blocks.","to":"native.output_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"out."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"out.","to":"native.out."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"clip."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const SDXL_DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Exact":"conv_in.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.0.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_embedding.linear_1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.time_embed.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"add_embedding.linear_1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.label_emb.0.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"down_blocks."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"down_blocks.","to":"native.diffusers.down_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"mid_block."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"mid_block.","to":"native.diffusers.mid_block."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"up_blocks."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"up_blocks.","to":"native.diffusers.up_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.out.2.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoder.","to":"clip_l."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoder_2."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoder_2.","to":"clip_g."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const SDXL_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "model.diffusion_model.out.2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "input_blocks.0.0.weight",
            "time_embed.0.weight",
            "label_emb.0.0.weight",
            "out.2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "add_embedding.linear_1.weight",
            "conv_out.weight",
        ],
        required_prefixes: &[],
    },
];

pub const SDXL_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &SDXL_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &SDXL_STANDALONE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &SDXL_DIFFUSERS_STATE_PLAN,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdxlVariant {
    InstructPix2Pix,
    Refiner,
    Base,
    Ssd1B,
    Koala700M,
    Koala1B,
    SegmindVega,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdxlLayout {
    PrefixedNative,
    StandaloneNative,
    Diffusers,
}

#[derive(Clone, Copy, Debug)]
pub struct SdxlConfiguration {
    pub variant: SdxlVariant,
    pub layout: SdxlLayout,
    pub in_channels: u64,
    pub out_channels: u64,
    pub model_channels: u64,
    pub context_dimension: u64,
    pub adm_in_channels: u64,
    pub attention_head_channels: u64,
    pub num_res_blocks: &'static [usize],
    pub transformer_depth: &'static [usize],
    pub transformer_depth_output: &'static [usize],
    pub transformer_depth_middle: isize,
    pub uses_linear_transformer_projection: bool,
    pub uses_temporal_attention: bool,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub latent_format: &'static LatentFormatDefinition,
    pub memory_usage_factor: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SdxlCommonMapping {
    pub components: &'static [ModelFamilyComponent],
    pub component_state_schemas: &'static [ModelFamilyComponentStateSchema],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub latent_format: &'static LatentFormatDefinition,
    pub forward_program: &'static [ModelForwardStep],
}

pub static SDXL_COMMON_MAPPING: SdxlCommonMapping = SdxlCommonMapping {
    components: SDXL_COMPONENTS,
    component_state_schemas: SDXL_COMPONENT_STATE_SCHEMAS,
    supported_dtypes: SDXL_SUPPORTED_DTYPES,
    supported_devices: SDXL_SUPPORTED_DEVICES,
    latent_format: SDXL_LATENT_FORMAT,
    forward_program: SDXL_FORWARD_PROGRAM,
};

pub fn common_mapping() -> &'static SdxlCommonMapping {
    &SDXL_COMMON_MAPPING
}

pub fn state_plan_for_layout(layout: SdxlLayout) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        SdxlLayout::PrefixedNative => &SDXL_PREFIXED_STATE_PLAN,
        SdxlLayout::StandaloneNative => &SDXL_STANDALONE_STATE_PLAN,
        SdxlLayout::Diffusers => &SDXL_DIFFUSERS_STATE_PLAN,
    }
}

pub fn configuration_for_probe(probe: &ModelProbe) -> Result<SdxlConfiguration, ModelFamilyError> {
    let invalid = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!("SDXL configuration {message}"))
    };
    let state_layout = probe.select_layout(SDXL_LAYOUT_SIGNATURES)?;
    let (layout, prefix, input_key, time_key, adm_key, out_key, standard_marker, deep_prefix) =
        match state_layout {
            ModelStateLayout::PrefixedNative => (
                SdxlLayout::PrefixedNative,
                "model.diffusion_model.",
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.time_embed.0.weight",
                "model.diffusion_model.label_emb.0.0.weight",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.input_blocks.2.0.in_layers.0.weight",
                "model.diffusion_model.input_blocks.7.1.transformer_blocks.",
            ),
            ModelStateLayout::StandaloneNative => (
                SdxlLayout::StandaloneNative,
                "",
                "input_blocks.0.0.weight",
                "time_embed.0.weight",
                "label_emb.0.0.weight",
                "out.2.weight",
                "input_blocks.2.0.in_layers.0.weight",
                "input_blocks.7.1.transformer_blocks.",
            ),
            ModelStateLayout::Diffusers => (
                SdxlLayout::Diffusers,
                "",
                "conv_in.weight",
                "time_embedding.linear_1.weight",
                "add_embedding.linear_1.weight",
                "conv_out.weight",
                "down_blocks.0.resnets.1.conv1.weight",
                "down_blocks.2.attentions.0.transformer_blocks.",
            ),
        };

    let input = required_shape(probe, input_key, &invalid)?;
    if input.len() != 4 || input.contains(&0) || input[2..] != [3, 3] {
        return Err(invalid(format!("{input_key} shape {input:?}")));
    }
    let model_channels = input[0];
    let in_channels = input[1];
    let time = required_shape(probe, time_key, &invalid)?;
    if time != [model_channels * 4, model_channels] {
        return Err(invalid(format!("{time_key} shape {time:?}")));
    }
    let adm = required_shape(probe, adm_key, &invalid)?;
    if adm.len() != 2 || adm.contains(&0) {
        return Err(invalid(format!("{adm_key} shape {adm:?}")));
    }
    let out = required_shape(probe, out_key, &invalid)?;
    if out.len() != 4 || out[0] != 4 || out[1] != model_channels || out[2..] != [3, 3] {
        return Err(invalid(format!("{out_key} shape {out:?}")));
    }

    let standard_residual_layout = probe.tensor_shapes.contains_key(standard_marker);
    let (depth, context_key, first_depth) = if standard_residual_layout {
        let depth = checked_depth(probe, deep_prefix, &invalid)?;
        let context_key = format!("{deep_prefix}0.attn2.to_k.weight");
        (depth, context_key, None)
    } else {
        let (first_prefix, koala_deep_prefix) = match layout {
            SdxlLayout::PrefixedNative => (
                format!("{prefix}input_blocks.3.1.transformer_blocks."),
                format!("{prefix}input_blocks.5.1.transformer_blocks."),
            ),
            SdxlLayout::StandaloneNative => (
                "input_blocks.3.1.transformer_blocks.".to_owned(),
                "input_blocks.5.1.transformer_blocks.".to_owned(),
            ),
            SdxlLayout::Diffusers => (
                "down_blocks.1.attentions.0.transformer_blocks.".to_owned(),
                "down_blocks.2.attentions.0.transformer_blocks.".to_owned(),
            ),
        };
        let first_depth = checked_depth(probe, &first_prefix, &invalid)?;
        let depth = checked_depth(probe, &koala_deep_prefix, &invalid)?;
        let context_key = format!("{koala_deep_prefix}0.attn2.to_k.weight");
        (depth, context_key, Some(first_depth))
    };
    let context = required_shape(probe, &context_key, &invalid)?;
    if context.len() != 2 || context.contains(&0) {
        return Err(invalid(format!("{context_key} shape {context:?}")));
    }
    let context_dimension = context[1];

    let (
        variant,
        num_res_blocks,
        transformer_depth,
        transformer_depth_output,
        middle,
        clip,
        memory,
    ) = if model_channels == SDXL_REFINER_MODEL_CHANNELS
        && in_channels == 4
        && adm[1] == SDXL_REFINER_ADM_INPUT_DIMENSION
        && context_dimension == SDXL_REFINER_CONTEXT_DIMENSION
        && standard_residual_layout
        && depth == 4
    {
        (
            SdxlVariant::Refiner,
            &[2, 2, 2, 2][..],
            SDXL_REFINER_TRANSFORMER_DEPTH,
            SDXL_REFINER_TRANSFORMER_DEPTH_OUTPUT,
            4,
            &SDXL_REFINER_CLIP_TARGET,
            SDXL_REFINER_MEMORY_USAGE_FACTOR,
        )
    } else if model_channels == SDXL_MODEL_CHANNELS
        && adm[1] == SDXL_ADM_INPUT_DIMENSION
        && context_dimension == SDXL_CONTEXT_DIMENSION
    {
        if standard_residual_layout {
            let variant = match (in_channels, depth) {
                (8, 10) => SdxlVariant::InstructPix2Pix,
                (4, 10) => SdxlVariant::Base,
                (4, 4) => SdxlVariant::Ssd1B,
                (4, 2) => SdxlVariant::SegmindVega,
                _ => {
                    return Err(invalid(format!(
                        "unsupported standard input/depth pair ({in_channels}, {depth})"
                    )));
                }
            };
            let (transformer_depth, transformer_depth_output, middle) = match variant {
                SdxlVariant::InstructPix2Pix | SdxlVariant::Base => {
                    (SDXL_TRANSFORMER_DEPTH, SDXL_TRANSFORMER_DEPTH_OUTPUT, 10)
                }
                SdxlVariant::Ssd1B => (
                    SDXL_SSD1B_TRANSFORMER_DEPTH,
                    SDXL_SSD1B_TRANSFORMER_DEPTH_OUTPUT,
                    -1,
                ),
                SdxlVariant::SegmindVega => (
                    SDXL_SEGMIND_TRANSFORMER_DEPTH,
                    SDXL_SEGMIND_TRANSFORMER_DEPTH_OUTPUT,
                    -1,
                ),
                _ => unreachable!(),
            };
            (
                variant,
                &[2, 2, 2][..],
                transformer_depth,
                transformer_depth_output,
                middle,
                &SDXL_CLIP_TARGET,
                SDXL_MEMORY_USAGE_FACTOR,
            )
        } else {
            if in_channels != 4 || first_depth != Some(2) {
                return Err(invalid(format!(
                    "KOALA requires four input channels and first depth two, got {in_channels}/{first_depth:?}"
                )));
            }
            let (variant, transformer_depth, transformer_depth_output, middle) = match depth {
                5 => (
                    SdxlVariant::Koala700M,
                    SDXL_KOALA_700M_TRANSFORMER_DEPTH,
                    SDXL_KOALA_700M_TRANSFORMER_DEPTH_OUTPUT,
                    -2,
                ),
                6 => (
                    SdxlVariant::Koala1B,
                    SDXL_KOALA_1B_TRANSFORMER_DEPTH,
                    SDXL_KOALA_1B_TRANSFORMER_DEPTH_OUTPUT,
                    6,
                ),
                _ => return Err(invalid(format!("unsupported KOALA depth {depth}"))),
            };
            (
                variant,
                &[1, 1, 1][..],
                transformer_depth,
                transformer_depth_output,
                middle,
                &SDXL_CLIP_TARGET,
                SDXL_MEMORY_USAGE_FACTOR,
            )
        }
    } else {
        return Err(invalid(format!(
            "unsupported channels/context/ADM profile {model_channels}/{context_dimension}/{}",
            adm[1]
        )));
    };

    Ok(SdxlConfiguration {
        variant,
        layout,
        in_channels,
        out_channels: 4,
        model_channels,
        context_dimension,
        adm_in_channels: adm[1],
        attention_head_channels: SDXL_ATTENTION_HEAD_CHANNELS,
        num_res_blocks,
        transformer_depth,
        transformer_depth_output,
        transformer_depth_middle: middle,
        uses_linear_transformer_projection: true,
        uses_temporal_attention: false,
        clip_target: clip,
        latent_format: SDXL_LATENT_FORMAT,
        memory_usage_factor: memory,
    })
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
    if indices.is_empty() || indices.len() > 32 || indices.iter().copied().ne(0..indices.len()) {
        return Err(invalid(format!(
            "{prefix} transformer blocks are not bounded and consecutive"
        )));
    }
    Ok(indices.len())
}
