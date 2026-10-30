use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelFamilyComponent,
    ModelFamilyComponentStateSchema, ModelFamilyError, ModelFamilyStatePlanCase,
    ModelForwardOperation, ModelForwardStep, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const FLUX_ARCHITECTURE_VERSION: &str = "flux-transformer-v1";
pub const FLUX_LATENT_FEATURE_ID: &str = "COMFY-MODEL-0029";
pub const FLUX_LATENT_IDENTIFIER: &str = "Flux";
pub const FLUX_MEMORY_USAGE_FACTOR: f64 = 3.1;
pub const FLUX_MEMORY_ESTIMATOR: MemoryEstimatorDescriptor = MemoryEstimatorDescriptor {
    fixed_bytes: 0,
    bytes_per_parameter: 4,
    activation_bytes_per_element: 4,
};

pub const FLUX_CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];

pub const FLUX_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.flux.FluxTokenizer",
        clip_model: "sim.text_encoders.flux.flux_clip",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: FLUX_CLIP_CONFIGURATION,
        },
    }];

pub const FLUX_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: FLUX_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const FLUX_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "flow-matching image transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Flux latent decoder",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "Flux T5 conditioning",
        required: false,
    },
];

pub const FLUX_WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

pub const FLUX_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.double_blocks.0.img_attn.proj.weight",
    "native.single_blocks.0.linear2.weight",
    "native.final_layer.linear.weight",
];

pub const FLUX_MODEL_OPTIONAL_KEYS: &[&str] = &[
    "native.double_blocks.0.img_attn.norm.key_norm.weight",
    "native.img_in.weight",
    "native.txt_in.weight",
    "native.vector_in.in_layer.weight",
    "native.guidance_in.in_layer.weight",
    "native.double_stream_modulation_img.lin.weight",
    "native.double_stream_modulation_txt.lin.weight",
];

pub const FLUX_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
pub const FLUX_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];
pub const FLUX_NATIVE_KEY_NORM_KEYS: &[&str] = &[
    "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight",
    "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
    "double_blocks.0.img_attn.norm.key_norm.weight",
    "double_blocks.0.img_attn.norm.key_norm.scale",
];
pub const FLUX_DIFFUSERS_KEY_NORM_KEYS: &[&str] = &[
    "transformer_blocks.0.attn.norm_k.weight",
    "transformer_blocks.0.attn.norm_k.scale",
];
pub const FLUX_INPUT_PROJECTION_KEYS: &[&str] = &[
    "model.diffusion_model.img_in.weight",
    "img_in.weight",
    "x_embedder.weight",
];
pub const FLUX_TEXT_PROJECTION_KEYS: &[&str] = &[
    "model.diffusion_model.txt_in.weight",
    "txt_in.weight",
    "context_embedder.weight",
];
pub const FLUX_GUIDANCE_PROJECTION_KEYS: &[&str] = &[
    "model.diffusion_model.guidance_in.in_layer.weight",
    "guidance_in.in_layer.weight",
    "time_text_embed.guidance_embedder.linear_1.weight",
];
pub const FLUX2_DISCRIMINATOR_KEYS: &[&str] = &[
    "model.diffusion_model.double_stream_modulation_img.lin.weight",
    "double_stream_modulation_img.lin.weight",
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
];
pub const FLUX_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[],
        required_prefixes: &["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm."],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[],
        required_prefixes: &["double_blocks.0.img_attn.norm.key_norm."],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "x_embedder.weight",
            "x_embedder.bias",
            "context_embedder.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_v.weight",
            "transformer_blocks.0.attn.to_out.0.weight",
            "single_transformer_blocks.0.attn.to_q.weight",
            "single_transformer_blocks.0.attn.to_k.weight",
            "single_transformer_blocks.0.attn.to_v.weight",
            "single_transformer_blocks.0.proj_mlp.weight",
            "single_transformer_blocks.0.proj_out.weight",
            "proj_out.weight",
        ],
        required_prefixes: &[
            "transformer_blocks.0.attn.norm_k.",
            "single_transformer_blocks.0.attn.norm_k.",
        ],
    },
];
pub const CHROMA_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[],
        required_prefixes: &["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm."],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[],
        required_prefixes: &["double_blocks.0.img_attn.norm.key_norm."],
    },
];

pub const FLUX_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "double_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.double_blocks.0.img_attn.proj.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "double_stream_activation",
        operation: ModelForwardOperation::Silu,
    },
    ModelForwardStep {
        checkpoint: "single_stream_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.single_blocks.0.linear2.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "single_stream_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: None,
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "final_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.final_layer.linear.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "flow_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const FLUX_NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const FLUX_UNPREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"final_layer."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"final_layer."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"img_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"img_in.","to":"native.img_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"txt_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"txt_in.","to":"native.txt_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_in.","to":"native.time_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vector_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"vector_in.","to":"native.vector_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"guidance_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"guidance_in.","to":"native.guidance_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"double_stream_modulation_img."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_stream_modulation_img.","to":"native.double_stream_modulation_img."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"double_stream_modulation_txt."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_stream_modulation_txt.","to":"native.double_stream_modulation_txt."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const FLUX_DIFFUSERS_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Copy":{"selector":{"predicate":{"Exact":"x_embedder.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.img_in.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"x_embedder.bias"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.img_in.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"context_embedder.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.txt_in.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"context_embedder.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.txt_in.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"proj_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.final_layer.linear.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"proj_out.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.final_layer.linear.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.to_out.0.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.double_blocks.0.img_attn.proj.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.to_out.0.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.double_blocks.0.img_attn.proj.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.norm_k.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.double_blocks.0.img_attn.norm.key_norm.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.norm_k.scale"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.double_blocks.0.img_attn.norm.key_norm.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.proj_out.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"native.single_blocks.0.linear2.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.proj_out.bias"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.single_blocks.0.linear2.bias"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.double_blocks.0.img_attn.q.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.to_k.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.double_blocks.0.img_attn.k.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"transformer_blocks.0.attn.to_v.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.double_blocks.0.img_attn.v.weight"},"component":"model"}},
            {"Assemble":{"sources":[{"Staged":{"component":"model","key":"conversion.double_blocks.0.img_attn.q.weight"}},{"Staged":{"component":"model","key":"conversion.double_blocks.0.img_attn.k.weight"}},{"Staged":{"component":"model","key":"conversion.double_blocks.0.img_attn.v.weight"}}],"dimension":0,"output":{"component":"model","key":"native.double_blocks.0.img_attn.qkv.weight"}}},
            {"Copy":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.attn.to_q.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.single_blocks.0.q.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.attn.to_k.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.single_blocks.0.k.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.attn.to_v.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.single_blocks.0.v.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"single_transformer_blocks.0.proj_mlp.weight"},"minimum_matches":1,"maximum_matches":1},"rewrite":{"Exact":"conversion.single_blocks.0.mlp.weight"},"component":"model"}},
            {"Assemble":{"sources":[{"Staged":{"component":"model","key":"conversion.single_blocks.0.q.weight"}},{"Staged":{"component":"model","key":"conversion.single_blocks.0.k.weight"}},{"Staged":{"component":"model","key":"conversion.single_blocks.0.v.weight"}},{"Staged":{"component":"model","key":"conversion.single_blocks.0.mlp.weight"}}],"dimension":0,"output":{"component":"model","key":"native.single_blocks.0.linear1.weight"}}},
            {"Copy":{"selector":{"predicate":{"Exact":"time_text_embed.timestep_embedder.linear_1.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.time_in.in_layer.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"time_text_embed.text_embedder.linear_1.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.vector_in.in_layer.weight"},"component":"model"}},
            {"Copy":{"selector":{"predicate":{"Exact":"time_text_embed.guidance_embedder.linear_1.weight"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.guidance_in.in_layer.weight"},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"x_embedder."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"x_embedder.","to":"native.diffusers.x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"context_embedder."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"context_embedder.","to":"native.diffusers.context_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"transformer_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"transformer_blocks.","to":"native.diffusers.transformer_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"single_transformer_blocks."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"single_transformer_blocks.","to":"native.diffusers.single_transformer_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_text_embed."},"minimum_matches":0,"maximum_matches":256},"rewrite":{"Prefix":{"from":"time_text_embed.","to":"native.diffusers.time_text_embed."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"norm_out."},"minimum_matches":0,"maximum_matches":64},"rewrite":{"Prefix":{"from":"norm_out.","to":"native.diffusers.norm_out."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"proj_out."},"minimum_matches":1,"maximum_matches":64},"rewrite":{"Prefix":{"from":"proj_out.","to":"native.diffusers.proj_out."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"controlnet_x_embedder."},"minimum_matches":0,"maximum_matches":64},"rewrite":{"Prefix":{"from":"controlnet_x_embedder.","to":"native.diffusers.controlnet_x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"text_encoders.","to":"text_encoder."}},"component":"text_encoder"}}
        ],
        "unmatched":{"Route":{"component":"model","rewrite":"Identity"}}
    }"#,
    };

pub const FLUX_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &FLUX_NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &FLUX_UNPREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &FLUX_DIFFUSERS_STATE_PLAN,
    },
];

pub const FLUX_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: FLUX_MODEL_REQUIRED_KEYS,
        optional_keys: FLUX_MODEL_OPTIONAL_KEYS,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FluxChromaVariant {
    Flux,
    Flux2,
    Chroma,
    ChromaRadiance,
    LongCatImage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FluxChromaLayout {
    Native,
    Unprefixed,
    Diffusers,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FluxChromaFinalHead {
    Linear,
    Convolution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FluxChromaConfiguration {
    pub variant: FluxChromaVariant,
    pub layout: FluxChromaLayout,
    pub in_channels: u64,
    pub out_channels: u64,
    pub patch_size: u64,
    pub hidden_size: u64,
    pub context_input_dimension: u64,
    pub vector_input_dimension: Option<u64>,
    pub attention_heads: u64,
    pub double_block_count: usize,
    pub single_block_count: usize,
    pub guidance_embedding: bool,
    pub text_id_dimensions: &'static [u64],
    pub final_head: FluxChromaFinalHead,
    pub use_x0_prediction: bool,
    pub use_sequential_text_ids: bool,
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
    expected_variant: FluxChromaVariant,
    family_identifier: &str,
) -> Result<FluxChromaConfiguration, ModelFamilyError> {
    let invalid_configuration = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "{family_identifier} configuration is invalid: {message}"
        ))
    };
    let (layout, prefix) = match probe.select_layout(FLUX_LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (FluxChromaLayout::Native, "model.diffusion_model."),
        ModelStateLayout::StandaloneNative => (FluxChromaLayout::Unprefixed, ""),
        ModelStateLayout::Diffusers => (FluxChromaLayout::Diffusers, ""),
    };
    let diffusers = layout == FluxChromaLayout::Diffusers;
    let img_input_key = if diffusers {
        "x_embedder.weight".to_owned()
    } else {
        format!("{prefix}img_in.weight")
    };
    let text_input_key = if diffusers {
        "context_embedder.weight".to_owned()
    } else {
        format!("{prefix}txt_in.weight")
    };
    let vector_input_key = if diffusers {
        "time_text_embed.text_embedder.linear_1.weight".to_owned()
    } else {
        format!("{prefix}vector_in.in_layer.weight")
    };
    let guidance_input_key = if diffusers {
        "time_text_embed.guidance_embedder.linear_1.weight".to_owned()
    } else {
        format!("{prefix}guidance_in.in_layer.weight")
    };
    let (double_block_pattern, single_block_pattern) = if diffusers {
        ("transformer_blocks.{}.", "single_transformer_blocks.{}.")
    } else {
        ("double_blocks.{}.", "single_blocks.{}.")
    };

    if diffusers {
        require_weight_or_scale(
            probe,
            "",
            "transformer_blocks.0.attn.norm_k",
            &invalid_configuration,
        )?;
    } else {
        require_weight_or_scale(
            probe,
            prefix,
            "double_blocks.0.img_attn.norm.key_norm",
            &invalid_configuration,
        )?;
    }
    let has_img_input = probe.tensor_shapes.contains_key(&img_input_key);
    let has_distilled_guidance = !diffusers
        && (has_weight_or_scale(probe, prefix, "distilled_guidance_layer.0.norms.0")
            || has_weight_or_scale(probe, prefix, "distilled_guidance_layer.norms.0"));
    if !has_img_input && !has_distilled_guidance {
        return Err(invalid_configuration(
            "missing img_in or distilled-guidance branch marker".to_string(),
        ));
    }
    let has_radiance = !diffusers && has_weight_or_scale(probe, prefix, "nerf_blocks.0.norm");
    let has_flux2 = if diffusers {
        probe
            .tensor_shapes
            .contains_key("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight")
    } else {
        probe
            .tensor_shapes
            .contains_key(&format!("{prefix}double_stream_modulation_img.lin.weight"))
    };
    let mut variant = if has_radiance {
        FluxChromaVariant::ChromaRadiance
    } else if has_distilled_guidance {
        FluxChromaVariant::Chroma
    } else if has_flux2 {
        FluxChromaVariant::Flux2
    } else {
        FluxChromaVariant::Flux
    };
    let flux_or_longcat_expectation = matches!(
        expected_variant,
        FluxChromaVariant::Flux | FluxChromaVariant::LongCatImage
    );
    if (variant != FluxChromaVariant::Flux && variant != expected_variant)
        || (variant == FluxChromaVariant::Flux && !flux_or_longcat_expectation)
    {
        return Err(invalid_configuration(format!(
            "detector selected {variant:?}, expected {expected_variant:?}"
        )));
    }
    let (mut in_channels, mut out_channels, mut patch_size, mut hidden_size, axes_dimension) =
        match variant {
            FluxChromaVariant::Flux => (16, 16, 2, 3_072, 128),
            FluxChromaVariant::Flux2 => (16, 128, 1, 3_072, 128),
            FluxChromaVariant::Chroma => (64, 64, 2, 5_120, 128),
            FluxChromaVariant::ChromaRadiance => (3, 3, 0, 0, 128),
            FluxChromaVariant::LongCatImage => (16, 16, 2, 3_072, 128),
        };
    let mut context_input_dimension = 4_096;

    if variant == FluxChromaVariant::ChromaRadiance {
        let patch_shape = required_dimensions(
            probe,
            &format!("{prefix}img_in_patch.weight"),
            &invalid_configuration,
        )?;
        if patch_shape.len() != 4
            || patch_shape[1] != 3
            || patch_shape[2] == 0
            || patch_shape[2] != patch_shape[3]
        {
            return Err(invalid_configuration(
                "img_in_patch.weight shape".to_string(),
            ));
        }
        patch_size = patch_shape[2];
    } else if let Some(shape) = probe.tensor_shapes.get(&img_input_key) {
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(invalid_configuration("img_in.weight shape".to_string()));
        }
        let patch_area = patch_size
            .checked_mul(patch_size)
            .ok_or_else(|| invalid_configuration("patch area overflow".to_string()))?;
        if shape[1] % patch_area != 0 {
            return Err(invalid_configuration(
                "img_in.weight input dimension".to_string(),
            ));
        }
        in_channels = shape[1] / patch_area;
        hidden_size = shape[0];
    }

    if let Some(shape) = probe.tensor_shapes.get(&text_input_key) {
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(invalid_configuration("txt_in.weight shape".to_string()));
        }
        hidden_size = shape[0];
        context_input_dimension = shape[1];
    } else if variant == FluxChromaVariant::ChromaRadiance {
        return Err(invalid_configuration(
            "missing tensor txt_in.weight".to_string(),
        ));
    }

    if variant == FluxChromaVariant::Chroma {
        require_vector_dimension(
            probe,
            prefix,
            &["double_blocks.0.img_attn.norm.key_norm"],
            128,
            &invalid_configuration,
        )?;
        require_vector_dimension(
            probe,
            prefix,
            &[
                "distilled_guidance_layer.0.norms.0",
                "distilled_guidance_layer.norms.0",
            ],
            5_120,
            &invalid_configuration,
        )?;
        hidden_size = 5_120;
        out_channels = 64;
    }
    if hidden_size == 0 || hidden_size % axes_dimension != 0 {
        return Err(invalid_configuration(format!(
            "hidden size {hidden_size} is not divisible by {axes_dimension}"
        )));
    }
    let attention_heads = hidden_size / axes_dimension;
    let double_block_count = consecutive_blocks(
        probe,
        &format!("{prefix}{double_block_pattern}"),
        &invalid_configuration,
    )?;
    let single_block_count = consecutive_blocks(
        probe,
        &format!("{prefix}{single_block_pattern}"),
        &invalid_configuration,
    )?;
    let vector_input_dimension =
        optional_matrix_dimension(probe, &vector_input_key, 1, &invalid_configuration)?;
    let guidance_embedding = probe.tensor_shapes.contains_key(&guidance_input_key);

    if variant == FluxChromaVariant::Flux
        && context_input_dimension == 3_584
        && vector_input_dimension.is_none()
        && !guidance_embedding
    {
        variant = FluxChromaVariant::LongCatImage;
    }
    if variant != expected_variant {
        return Err(invalid_configuration(format!(
            "detector selected {variant:?}, expected {expected_variant:?}"
        )));
    }
    let yak_mlp = !diffusers
        && probe
            .tensor_shapes
            .contains_key(&format!("{prefix}double_blocks.0.img_mlp.gate_proj.weight"));
    let text_normalization = !diffusers && has_weight_or_scale(probe, prefix, "txt_norm");
    let text_id_dimensions = match variant {
        FluxChromaVariant::Flux2 => &[3][..],
        FluxChromaVariant::LongCatImage => &[1, 2][..],
        FluxChromaVariant::Flux if yak_mlp && text_normalization => &[1, 2][..],
        _ => &[][..],
    };

    let final_head = if variant == FluxChromaVariant::ChromaRadiance
        && has_weight_or_scale(probe, prefix, "nerf_final_layer_conv.norm")
    {
        require_key(
            probe,
            &format!("{prefix}nerf_final_layer_conv.conv.weight"),
            &invalid_configuration,
        )?;
        FluxChromaFinalHead::Convolution
    } else {
        if variant == FluxChromaVariant::ChromaRadiance {
            require_weight_or_scale(
                probe,
                prefix,
                "nerf_final_layer.norm",
                &invalid_configuration,
            )?;
            require_key(
                probe,
                &format!("{prefix}nerf_final_layer.linear.weight"),
                &invalid_configuration,
            )?;
        }
        FluxChromaFinalHead::Linear
    };
    if variant == FluxChromaVariant::ChromaRadiance {
        let nerf_depth = consecutive_blocks(
            probe,
            &format!("{prefix}nerf_blocks.{{}}."),
            &invalid_configuration,
        )?;
        if nerf_depth != 4 {
            return Err(invalid_configuration(format!(
                "NeRF depth is {nerf_depth}, expected 4"
            )));
        }
    }

    Ok(FluxChromaConfiguration {
        variant,
        layout,
        in_channels,
        out_channels,
        patch_size,
        hidden_size,
        context_input_dimension,
        vector_input_dimension,
        attention_heads,
        double_block_count,
        single_block_count,
        guidance_embedding,
        text_id_dimensions,
        final_head,
        use_x0_prediction: probe.tensor_shapes.contains_key(&format!("{prefix}__x0__")),
        use_sequential_text_ids: probe
            .tensor_shapes
            .contains_key(&format!("{prefix}__sequential__")),
    })
}

fn required_dimensions<'a>(
    probe: &'a ModelProbe,
    key: &str,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<&'a [u64], ModelFamilyError> {
    probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_configuration(format!("missing tensor {key}")))
}

fn require_key(
    probe: &ModelProbe,
    key: &str,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<(), ModelFamilyError> {
    if probe.tensor_shapes.contains_key(key) {
        Ok(())
    } else {
        Err(invalid_configuration(format!("missing tensor {key}")))
    }
}

fn has_weight_or_scale(probe: &ModelProbe, prefix: &str, stem: &str) -> bool {
    ["weight", "scale"].iter().any(|suffix| {
        probe
            .tensor_shapes
            .contains_key(&format!("{prefix}{stem}.{suffix}"))
    })
}

fn require_weight_or_scale(
    probe: &ModelProbe,
    prefix: &str,
    stem: &str,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<(), ModelFamilyError> {
    let matches = ["weight", "scale"]
        .iter()
        .filter(|suffix| {
            probe
                .tensor_shapes
                .contains_key(&format!("{prefix}{stem}.{suffix}"))
        })
        .count();
    if matches == 1 {
        Ok(())
    } else {
        Err(invalid_configuration(format!(
            "expected exactly one weight/scale tensor for {prefix}{stem}"
        )))
    }
}

fn require_vector_dimension(
    probe: &ModelProbe,
    prefix: &str,
    stems: &[&str],
    expected: u64,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<(), ModelFamilyError> {
    let matches = stems
        .iter()
        .flat_map(|stem| ["weight", "scale"].map(|suffix| format!("{prefix}{stem}.{suffix}")))
        .filter_map(|key| probe.tensor_shapes.get(&key).map(|shape| (key, shape)))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(invalid_configuration(
            "expected exactly one distilled-guidance weight/scale tensor".to_string(),
        ));
    }
    let (key, shape) = &matches[0];
    if shape.as_slice() != [expected] {
        return Err(invalid_configuration(format!(
            "{key} shape is {shape:?}, expected [{expected}]"
        )));
    }
    Ok(())
}

fn consecutive_blocks(
    probe: &ModelProbe,
    pattern: &str,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<usize, ModelFamilyError> {
    let count = probe.consecutive_block_count(pattern)?;
    if count == 0 {
        Err(invalid_configuration(format!(
            "no consecutive blocks for {pattern}"
        )))
    } else {
        Ok(count)
    }
}

fn optional_matrix_dimension(
    probe: &ModelProbe,
    key: &str,
    dimension: usize,
    invalid_configuration: &impl Fn(String) -> ModelFamilyError,
) -> Result<Option<u64>, ModelFamilyError> {
    let Some(shape) = probe.tensor_shapes.get(key) else {
        return Ok(None);
    };
    if shape.len() != 2 {
        return Err(invalid_configuration(format!("{key} is not a matrix")));
    }
    shape
        .get(dimension)
        .copied()
        .filter(|value| *value > 0)
        .map(Some)
        .ok_or_else(|| invalid_configuration(format!("invalid dimension {dimension} for {key}")))
}
