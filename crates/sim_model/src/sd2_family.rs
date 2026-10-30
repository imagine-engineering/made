use crate::model_family::{
    ModelWeightStatistic, ModelWeightStatisticObservation, ModelWeightStatisticRequest,
};
use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelConfigurationKind, ModelConfigurationValue,
    ModelFamilyComponent, ModelFamilyComponentStateSchema, ModelFamilyError, ModelForwardOperation,
    ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const SD2_MODEL_CHANNELS: u64 = 320;
pub const SD2_CONTEXT_DIMENSION: u64 = 1_024;
pub const SD2_ATTENTION_HEAD_CHANNELS: u64 = 64;
pub const SD2_MEMORY_USAGE_FACTOR: f64 = 1.0;
pub const SD2_V_PREDICTION_THRESHOLD: f64 = 0.09;
pub const SD2_UNCLIP_TIMESTEPS: u64 = 1_000;
pub const SD2_UNCLIP_BETA_SCHEDULE: &str = "squaredcos_cap_v2";
pub const SD2_UNCLIP_NOISE_AUGMENT_MERGE: f64 = 0.05;
pub const SD2_UNCLIP_SEED_OFFSET: i64 = -10;

pub const SD2_NUM_RES_BLOCKS: &[u64] = &[2, 2, 2, 2];
pub const SD2_CHANNEL_MULTIPLIERS: &[u64] = &[1, 2, 4, 4];
pub const SD2_TRANSFORMER_DEPTH: &[u64] = &[1, 1, 1, 1, 1, 1, 0, 0];
pub const SD2_TRANSFORMER_DEPTH_OUTPUT: &[u64] = &[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0];

pub const SD2_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_sd15_sim_model_0045::LATENT_FORMAT;

pub const SD2_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.sd2_clip.SD2Tokenizer",
        clip_model: "sim.text_encoders.sd2_clip.SD2ClipModel",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];
pub static SD2_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: SD2_CLIP_CANDIDATES,
    dynamic_selection: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sd2Variant {
    LotusD,
    Sd20,
    Sd21UnclipL,
    Sd21UnclipH,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sd2Layout {
    PrefixedNative,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sd2ModelType {
    Eps,
    VPrediction,
    ImgToImg,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Sd2ConditioningFact {
    CrossAttention,
    InpaintLatentAndMask,
    LotusDeterministicTaskEmbedding,
    UnclipVisionEmbedding,
    UnclipNoiseLevelEmbedding,
    UnclipZeroFallback,
}

pub const SD2_CONDITIONING: &[Sd2ConditioningFact] = &[
    Sd2ConditioningFact::CrossAttention,
    Sd2ConditioningFact::InpaintLatentAndMask,
];
pub const LOTUS_CONDITIONING: &[Sd2ConditioningFact] = &[
    Sd2ConditioningFact::CrossAttention,
    Sd2ConditioningFact::LotusDeterministicTaskEmbedding,
];
pub const UNCLIP_CONDITIONING: &[Sd2ConditioningFact] = &[
    Sd2ConditioningFact::CrossAttention,
    Sd2ConditioningFact::UnclipVisionEmbedding,
    Sd2ConditioningFact::UnclipNoiseLevelEmbedding,
    Sd2ConditioningFact::UnclipZeroFallback,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sd2UnclipConfiguration {
    pub timestep_dimension: u64,
    pub timesteps: u64,
    pub beta_schedule: &'static str,
    pub seed_offset: i64,
}

pub const SD2_UNCLIP_L_CONFIGURATION: Sd2UnclipConfiguration = Sd2UnclipConfiguration {
    timestep_dimension: 768,
    timesteps: SD2_UNCLIP_TIMESTEPS,
    beta_schedule: SD2_UNCLIP_BETA_SCHEDULE,
    seed_offset: SD2_UNCLIP_SEED_OFFSET,
};
pub const SD2_UNCLIP_H_CONFIGURATION: Sd2UnclipConfiguration = Sd2UnclipConfiguration {
    timestep_dimension: 1_024,
    timesteps: SD2_UNCLIP_TIMESTEPS,
    beta_schedule: SD2_UNCLIP_BETA_SCHEDULE,
    seed_offset: SD2_UNCLIP_SEED_OFFSET,
};

#[derive(Clone, Copy, Debug)]
pub struct Sd2Configuration {
    pub variant: Sd2Variant,
    pub layout: Sd2Layout,
    pub model_type: Sd2ModelType,
    pub input_channels: u64,
    pub output_channels: u64,
    pub model_channels: u64,
    pub context_dimension: u64,
    pub adm_in_channels: Option<u64>,
    pub attention_head_channels: u64,
    pub uses_linear_transformer_projection: bool,
    pub uses_temporal_attention: bool,
    pub conditioning: &'static [Sd2ConditioningFact],
    pub unclip: Option<Sd2UnclipConfiguration>,
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub memory_usage_factor: f64,
}

pub const SD2_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "SD2, Unclip, or Lotus latent diffusion U-Net",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "SD2 OpenCLIP-H text conditioning",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vision_encoder",
        role: "optional Unclip OpenCLIP vision encoder",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "canonical SD15-compatible latent codec",
        required: false,
    },
];

pub const SD2_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.input_blocks.0.0.weight",
    "native.time_embed.0.weight",
    "native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
    "native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
    "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
    "native.out.2.weight",
];
pub const SD2_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.label_emb.0.0.weight",
    "native.output_blocks.11.1.transformer_blocks.0.norm1.bias",
    "native.time_embed.0.bias",
    "native.out.2.bias",
];
pub const SD2_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: SD2_MODEL_REQUIRED_KEYS,
        optional_keys: SD2_MODEL_OPTIONAL_KEYS,
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: &[],
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "vision_encoder",
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

pub const SD2_SUPPORTED_DTYPES: &[DType] = &[DType::F16, DType::Bf16, DType::F32];
pub const SD2_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

pub const SD2_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "timestep_embedding",
        operation: ModelForwardOperation::Linear {
            weight: "native.time_embed.0.weight",
            bias: Some("native.time_embed.0.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "input_self_attention",
        operation: ModelForwardOperation::Linear {
            weight: "native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "middle_cross_attention",
        operation: ModelForwardOperation::Linear {
            weight: "native.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "unet_residual_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "latent_prediction",
        operation: ModelForwardOperation::Tanh,
    },
];

// The native plans implement the source's two accepted SD2 CLIP prefixes and
// its OpenCLIP-to-transformers conversion. In-projection tensors are split by
// the canonical state transaction rather than by a private loader.
pub const SD2_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"SplitEach":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Contains":"transformer.resblocks."},{"Suffix":"attn.in_proj_weight"}]},"minimum_matches":0,"maximum_matches":32},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"text_encoder","rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Suffix":{"from":"attn.in_proj_weight","to":"self_attn.q_proj.weight"}}]}},{"component":"text_encoder","rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Suffix":{"from":"attn.in_proj_weight","to":"self_attn.k_proj.weight"}}]}},{"component":"text_encoder","rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Suffix":{"from":"attn.in_proj_weight","to":"self_attn.v_proj.weight"}}]}}]}},
            {"SplitEach":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Contains":"transformer.resblocks."},{"Suffix":"attn.in_proj_bias"}]},"minimum_matches":0,"maximum_matches":32},"dimension":0,"sizes":[{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]},{"DivideExact":[{"CurrentTensorDimension":{"dimension":0}},{"Literal":3}]}],"outputs":[{"component":"text_encoder","rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Suffix":{"from":"attn.in_proj_bias","to":"self_attn.q_proj.bias"}}]}},{"component":"text_encoder","rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Suffix":{"from":"attn.in_proj_bias","to":"self_attn.k_proj.bias"}}]}},{"component":"text_encoder","rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Suffix":{"from":"attn.in_proj_bias","to":"self_attn.v_proj.bias"}}]}}]}},
            {"Move":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Suffix":"positional_embedding"}]},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_h.transformer.text_model.embeddings.position_embedding.weight"},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Suffix":"token_embedding.weight"}]},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_h.transformer.text_model.embeddings.token_embedding.weight"},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Suffix":"ln_final.weight"}]},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_h.transformer.text_model.final_layer_norm.weight"},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Suffix":"ln_final.bias"}]},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_h.transformer.text_model.final_layer_norm.bias"},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Suffix":"text_projection.weight"}]},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_h.transformer.text_projection.weight"},"component":"text_encoder"}},
            {"TransformEach":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Suffix":"text_projection"},{"Not":{"Suffix":"text_projection.weight"}}]},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"clip_h.transformer.text_projection.weight"},"component":"text_encoder","transform":{"Transpose":{"first_dimension":0,"second_dimension":1}}}},
            {"Move":{"selector":{"predicate":{"All":[{"Any":[{"Prefix":"conditioner.embedders.0.model."},{"Prefix":"cond_stage_model.model."}]},{"Not":{"Suffix":"attn.in_proj_weight"}},{"Not":{"Suffix":"attn.in_proj_bias"}},{"Not":{"Suffix":"positional_embedding"}},{"Not":{"Suffix":"token_embedding.weight"}},{"Not":{"Suffix":"ln_final.weight"}},{"Not":{"Suffix":"ln_final.bias"}},{"Not":{"Suffix":"text_projection.weight"}},{"Not":{"Suffix":"text_projection"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"OrderedOptional":[{"Prefix":{"from":"conditioner.embedders.0.model.","to":"clip_h."}},{"Prefix":{"from":"cond_stage_model.model.","to":"clip_h."}},{"Contains":{"from":"transformer.resblocks.","to":"transformer.text_model.encoder.layers."}},{"Contains":{"from":".ln_1.","to":".layer_norm1."}},{"Contains":{"from":".ln_2.","to":".layer_norm2."}},{"Contains":{"from":".attn.out_proj.","to":".self_attn.out_proj."}},{"Contains":{"from":".mlp.c_fc.","to":".mlp.fc1."}},{"Contains":{"from":".mlp.c_proj.","to":".mlp.fc2."}}]},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"model.diffusion_model."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"embedder.model.visual."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"embedder.model.visual.","to":"clip_vision.visual."}},"component":"vision_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const SD2_DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Exact":"conv_in.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.0.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_in.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.0.0.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_embedding.linear_1.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.time_embed.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_embedding.linear_1.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.time_embed.0.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_embedding.linear_2.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.time_embed.2.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"time_embedding.linear_2.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.time_embed.2.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"class_embedding.linear_1.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.label_emb.0.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"class_embedding.linear_1.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.label_emb.0.0.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"class_embedding.linear_2.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.label_emb.0.2.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"class_embedding.linear_2.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.label_emb.0.2.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.middle_block.1.transformer_blocks.0.attn2.to_q.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"up_blocks.3.attentions.2.transformer_blocks.0.norm1.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.output_blocks.11.1.transformer_blocks.0.norm1.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"down_blocks."},{"Not":{"Exact":"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"}},{"Not":{"Exact":"down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"down_blocks.","to":"native.diffusers.down_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"mid_block."},{"Not":{"Exact":"mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"mid_block.","to":"native.diffusers.mid_block."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"up_blocks."},{"Not":{"Exact":"up_blocks.3.attentions.2.transformer_blocks.0.norm1.bias"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"up_blocks.","to":"native.diffusers.up_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_norm_out.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.out.0.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_norm_out.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.out.0.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.out.2.weight"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"conv_out.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.out.2.bias"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoder.","to":"clip_h.transformer."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"image_encoder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"image_encoder.","to":"clip_vision."}},"component":"vision_encoder"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vae.","to":"native."}},"component":"vae"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const SD2_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight",
            "model.diffusion_model.out.2.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
            "conv_out.weight",
        ],
        required_prefixes: &[],
    },
];

#[derive(Clone, Copy, Debug)]
pub struct Sd2CommonMapping {
    pub components: &'static [ModelFamilyComponent],
    pub component_state_schemas: &'static [ModelFamilyComponentStateSchema],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub latent_format: &'static LatentFormatDefinition,
    pub clip_target: &'static ModelClipTargetDefinition,
    pub forward_program: &'static [ModelForwardStep],
}

pub static SD2_COMMON_MAPPING: Sd2CommonMapping = Sd2CommonMapping {
    components: SD2_COMPONENTS,
    component_state_schemas: SD2_COMPONENT_STATE_SCHEMAS,
    supported_dtypes: SD2_SUPPORTED_DTYPES,
    supported_devices: SD2_SUPPORTED_DEVICES,
    latent_format: SD2_LATENT_FORMAT,
    clip_target: &SD2_CLIP_TARGET,
    forward_program: SD2_FORWARD_PROGRAM,
};

pub fn common_mapping() -> &'static Sd2CommonMapping {
    &SD2_COMMON_MAPPING
}

pub fn state_plan_for_layout(layout: Sd2Layout) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        Sd2Layout::PrefixedNative => &SD2_PREFIXED_STATE_PLAN,
        Sd2Layout::Diffusers => &SD2_DIFFUSERS_STATE_PLAN,
    }
}

pub fn lotus_task_embedding() -> [f32; 4] {
    [1.0_f32.sin(), 0.0_f32.sin(), 1.0_f32.cos(), 0.0_f32.cos()]
}

pub fn weight_statistic_request_for_probe(
    probe: &ModelProbe,
) -> Result<Option<ModelWeightStatisticRequest>, ModelFamilyError> {
    let layout = layout_for_probe(probe)?;
    let (input_key, statistic_key) = layout_keys(layout);
    let input_channels = required_shape(probe, input_key, 4)?[1];
    let normalized = probe.normalized_configuration()?;
    let is_lotus = optional_unsigned(&normalized, "adm_in_channels")? == Some(4);
    if is_lotus || input_channels != 4 || !probe.tensor_shapes().contains_key(statistic_key) {
        return Ok(None);
    }
    Ok(Some(
        ModelWeightStatisticRequest::population_standard_deviation(statistic_key, DeviceKind::Cpu)?,
    ))
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
    statistic: Option<&ModelWeightStatisticObservation>,
) -> Result<Sd2Configuration, ModelFamilyError> {
    let layout = layout_for_probe(probe)?;
    let (input_key, statistic_key) = layout_keys(layout);
    let normalized = probe.normalized_configuration()?;
    let expected_kind = match layout {
        Sd2Layout::PrefixedNative => ModelConfigurationKind::Native,
        Sd2Layout::Diffusers => ModelConfigurationKind::Diffusers,
    };
    if normalized.kind() != expected_kind {
        return Err(invalid("layout and normalized configuration kind disagree"));
    }

    let input = required_shape(probe, input_key, 4)?;
    if input[0] != SD2_MODEL_CHANNELS || !matches!(input[1], 4 | 9) || input[2..] != [3, 3] {
        return Err(invalid(format!(
            "unsupported input shape {input_key}={input:?}"
        )));
    }
    let input_channels = input[1];
    expect_unsigned(&normalized, "model_channels", SD2_MODEL_CHANNELS)?;
    expect_unsigned(&normalized, "in_channels", input_channels)?;
    expect_unsigned(&normalized, "out_channels", 4)?;
    expect_unsigned(&normalized, "context_dim", SD2_CONTEXT_DIMENSION)?;
    expect_unsigned_list(&normalized, "num_res_blocks", SD2_NUM_RES_BLOCKS)?;
    expect_unsigned_list(&normalized, "transformer_depth", SD2_TRANSFORMER_DEPTH)?;
    if expected_kind == ModelConfigurationKind::Native {
        expect_unsigned_list(&normalized, "channel_mult", SD2_CHANNEL_MULTIPLIERS)?;
        expect_unsigned_list(
            &normalized,
            "transformer_depth_output",
            SD2_TRANSFORMER_DEPTH_OUTPUT,
        )?;
        expect_signed(&normalized, "transformer_depth_middle", 1)?;
        expect_boolean(&normalized, "use_linear_in_transformer", true)?;
        expect_boolean(&normalized, "use_temporal_attention", false)?;
    }

    let adm_in_channels = optional_unsigned(&normalized, "adm_in_channels")?;
    let variant = match adm_in_channels {
        Some(4) if input_channels == 4 => Sd2Variant::LotusD,
        None => Sd2Variant::Sd20,
        Some(1_536) => Sd2Variant::Sd21UnclipL,
        Some(2_048) => Sd2Variant::Sd21UnclipH,
        Some(value) => {
            return Err(invalid(format!(
                "unsupported ADM dimension {value}; Lotus/Unclip precedence is exact"
            )));
        }
    };
    if layout == Sd2Layout::Diffusers && input_channels != 4 {
        return Err(invalid(
            "the pinned Diffusers SD21 table admits exactly four input channels",
        ));
    }

    let model_type = if variant == Sd2Variant::LotusD {
        if statistic.is_some() {
            return Err(invalid("Lotus does not consume the SD2 weight statistic"));
        }
        Sd2ModelType::ImgToImg
    } else if input_channels != 4 || !probe.tensor_shapes().contains_key(statistic_key) {
        if statistic.is_some() {
            return Err(invalid(
                "a weight statistic was supplied without the canonical source tensor",
            ));
        }
        Sd2ModelType::Eps
    } else {
        let observation = statistic.ok_or_else(|| {
            invalid("the canonical loaded-weight statistic observation is required")
        })?;
        if observation.tensor_name() != statistic_key
            || observation.statistic() != ModelWeightStatistic::PopulationStandardDeviation
        {
            return Err(invalid(format!(
                "weight statistic observation is not bound to {statistic_key}"
            )));
        }
        if observation.exceeds_checked(SD2_V_PREDICTION_THRESHOLD)? {
            Sd2ModelType::VPrediction
        } else {
            Sd2ModelType::Eps
        }
    };

    let (conditioning, unclip) = match variant {
        Sd2Variant::LotusD => (LOTUS_CONDITIONING, None),
        Sd2Variant::Sd20 => (SD2_CONDITIONING, None),
        Sd2Variant::Sd21UnclipL => (UNCLIP_CONDITIONING, Some(SD2_UNCLIP_L_CONFIGURATION)),
        Sd2Variant::Sd21UnclipH => (UNCLIP_CONDITIONING, Some(SD2_UNCLIP_H_CONFIGURATION)),
    };
    Ok(Sd2Configuration {
        variant,
        layout,
        model_type,
        input_channels,
        output_channels: 4,
        model_channels: SD2_MODEL_CHANNELS,
        context_dimension: SD2_CONTEXT_DIMENSION,
        adm_in_channels,
        attention_head_channels: SD2_ATTENTION_HEAD_CHANNELS,
        uses_linear_transformer_projection: true,
        uses_temporal_attention: false,
        conditioning,
        unclip,
        latent_format: SD2_LATENT_FORMAT,
        clip_target: &SD2_CLIP_TARGET,
        memory_usage_factor: SD2_MEMORY_USAGE_FACTOR,
    })
}

fn layout_for_probe(probe: &ModelProbe) -> Result<Sd2Layout, ModelFamilyError> {
    match probe.select_layout(SD2_LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => Ok(Sd2Layout::PrefixedNative),
        ModelStateLayout::StandaloneNative => Err(invalid(
            "standalone-native layout is not a pinned SD2 source layout",
        )),
        ModelStateLayout::Diffusers => Ok(Sd2Layout::Diffusers),
    }
}

fn layout_keys(layout: Sd2Layout) -> (&'static str, &'static str) {
    match layout {
        Sd2Layout::PrefixedNative => (
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias",
        ),
        Sd2Layout::Diffusers => (
            "conv_in.weight",
            "up_blocks.3.attentions.2.transformer_blocks.0.norm1.bias",
        ),
    }
}

fn required_shape<'a>(
    probe: &'a ModelProbe,
    key: &str,
    rank: usize,
) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))?;
    if shape.len() != rank || shape.contains(&0) {
        return Err(invalid(format!(
            "{key} must have non-zero rank {rank}, got {shape:?}"
        )));
    }
    Ok(shape)
}

fn expect_unsigned(
    configuration: &crate::ModelNormalizedConfiguration,
    key: &str,
    expected: u64,
) -> Result<(), ModelFamilyError> {
    match configuration.fact(key) {
        Some(ModelConfigurationValue::Unsigned(actual)) if *actual == expected => Ok(()),
        actual => Err(invalid(format!(
            "normalized {key} is {actual:?}, expected {expected}"
        ))),
    }
}

fn expect_signed(
    configuration: &crate::ModelNormalizedConfiguration,
    key: &str,
    expected: i64,
) -> Result<(), ModelFamilyError> {
    match configuration.fact(key) {
        Some(ModelConfigurationValue::Signed(actual)) if *actual == expected => Ok(()),
        actual => Err(invalid(format!(
            "normalized {key} is {actual:?}, expected {expected}"
        ))),
    }
}

fn expect_boolean(
    configuration: &crate::ModelNormalizedConfiguration,
    key: &str,
    expected: bool,
) -> Result<(), ModelFamilyError> {
    match configuration.fact(key) {
        Some(ModelConfigurationValue::Boolean(actual)) if *actual == expected => Ok(()),
        actual => Err(invalid(format!(
            "normalized {key} is {actual:?}, expected {expected}"
        ))),
    }
}

fn expect_unsigned_list(
    configuration: &crate::ModelNormalizedConfiguration,
    key: &str,
    expected: &[u64],
) -> Result<(), ModelFamilyError> {
    match configuration.fact(key) {
        Some(ModelConfigurationValue::UnsignedList(actual)) if actual == expected => Ok(()),
        actual => Err(invalid(format!(
            "normalized {key} is {actual:?}, expected {expected:?}"
        ))),
    }
}

fn optional_unsigned(
    configuration: &crate::ModelNormalizedConfiguration,
    key: &str,
) -> Result<Option<u64>, ModelFamilyError> {
    match configuration.fact(key) {
        Some(ModelConfigurationValue::None) => Ok(None),
        Some(ModelConfigurationValue::Unsigned(value)) => Ok(Some(*value)),
        actual => Err(invalid(format!(
            "normalized {key} must be None or unsigned, got {actual:?}"
        ))),
    }
}

fn invalid(message: impl Into<String>) -> ModelFamilyError {
    ModelFamilyError::InvalidSelectorOutput(format!(
        "SD2/Lotus source configuration mismatch: {}",
        message.into()
    ))
}
