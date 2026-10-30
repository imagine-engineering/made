use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    ModelClipModelInvocationDefinition, ModelClipTargetCandidateDefinition,
    ModelClipTargetDefinition, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyError, ModelFamilyStatePlanCase, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelStateTransformPlanDefinition, ModelWeightRule,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const HIDREAM_O1_ARCHITECTURE_VERSION: &str = "hidream-o1-pixel-transformer-v1";
pub const HIDREAM_O1_LATENT_FEATURE_ID: &str = "COMFY-MODEL-0031";
pub const HIDREAM_O1_LATENT_IDENTIFIER: &str = "HiDreamO1Pixel";
pub const HIDREAM_O1_MEMORY_USAGE_FACTOR: f64 = 0.033;
pub const HIDREAM_O1_PATCH_SIZE: u64 = 32;
pub const HIDREAM_O1_IMAGE_TOKEN_ID: u32 = 151_655;
pub const HIDREAM_O1_TMS_TOKEN_ID: u32 = 151_673;

pub const HIDREAM_O1_PAD_TOKEN_ID: u32 = 151_643;
pub const HIDREAM_O1_IM_START_TOKEN_ID: u32 = 151_644;
pub const HIDREAM_O1_IM_END_TOKEN_ID: u32 = 151_645;
pub const HIDREAM_O1_ASSISTANT_TOKEN_ID: u32 = 77_091;
pub const HIDREAM_O1_USER_TOKEN_ID: u32 = 872;
pub const HIDREAM_O1_NEWLINE_TOKEN_ID: u32 = 198;
pub const HIDREAM_O1_VISION_START_TOKEN_ID: u32 = 151_652;
pub const HIDREAM_O1_VISION_END_TOKEN_ID: u32 = 151_653;
pub const HIDREAM_O1_VIDEO_TOKEN_ID: u32 = 151_656;
pub const HIDREAM_O1_BOI_TOKEN_ID: u32 = 151_669;
pub const HIDREAM_O1_BOR_TOKEN_ID: u32 = 151_670;
pub const HIDREAM_O1_EOR_TOKEN_ID: u32 = 151_671;
pub const HIDREAM_O1_BOT_TOKEN_ID: u32 = 151_672;

pub const HIDREAM_O1_VISION_PATCH_SIZE: u64 = 16;
pub const HIDREAM_O1_VISION_MERGE_SIZE: u64 = 2;
pub const HIDREAM_O1_VISION_IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const HIDREAM_O1_VISION_IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];
pub const HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT: &str = "visual.deepstack_merger_list";
pub const HIDREAM_O1_PIXEL_VAE_SENTINEL: &str = "pixel_space_vae";
pub const HIDREAM_O1_TEXT_ENCODER_SENTINEL: &str = "_hidream_o1_te_sentinel";

pub const HIDREAM_O1_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hidreamo1pixel_sim_model_0031::LATENT_FORMAT;

pub const HIDREAM_O1_CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.hidream_o1.HiDreamO1Tokenizer",
        clip_model: "sim.text_encoders.hidream_o1.HiDreamO1TE",
        invocation: ModelClipModelInvocationDefinition::Reference,
    }];

pub const HIDREAM_O1_CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: HIDREAM_O1_CLIP_CANDIDATES,
    dynamic_selection: false,
};

pub const HIDREAM_O1_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "HiDream O1 pixel-space transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "pixel-space VAE sentinel",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "tokenizer-only text encoder sentinel",
        required: true,
    },
];

pub const HIDREAM_O1_WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

pub const HIDREAM_O1_MODEL_REQUIRED_KEYS: &[&str] = &[
    "native.t_embedder1.mlp.0.weight",
    "native.x_embedder.proj1.weight",
];
pub const HIDREAM_O1_VAE_REQUIRED_KEYS: &[&str] = &[HIDREAM_O1_PIXEL_VAE_SENTINEL];
pub const HIDREAM_O1_TEXT_ENCODER_REQUIRED_KEYS: &[&str] = &[HIDREAM_O1_TEXT_ENCODER_SENTINEL];

pub const HIDREAM_O1_COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "model",
        required_keys: HIDREAM_O1_MODEL_REQUIRED_KEYS,
        optional_keys: &[],
        allow_unexpected: true,
    },
    ModelFamilyComponentStateSchema {
        component: "vae",
        required_keys: HIDREAM_O1_VAE_REQUIRED_KEYS,
        optional_keys: &[],
        allow_unexpected: false,
    },
    ModelFamilyComponentStateSchema {
        component: "text_encoder",
        required_keys: HIDREAM_O1_TEXT_ENCODER_REQUIRED_KEYS,
        optional_keys: &[],
        allow_unexpected: false,
    },
];

pub const HIDREAM_O1_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F32];
pub const HIDREAM_O1_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];
pub const HIDREAM_O1_LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &["model.diffusion_model.t_embedder1.mlp.0.weight"],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &["t_embedder1.mlp.0.weight"],
        required_prefixes: &[],
    },
];

pub const HIDREAM_O1_NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Contains":"visual.deepstack_merger_list"}]},"minimum_matches":0,"maximum_matches":16384}}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Contains":"visual.deepstack_merger_list"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Drop":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384}}},
            {"Drop":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"vae","key":"pixel_space_vae"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"text_encoder","key":"_hidream_o1_te_sentinel"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const HIDREAM_O1_UNPREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Drop":{"selector":{"predicate":{"Contains":"visual.deepstack_merger_list"},"minimum_matches":0,"maximum_matches":16384}}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"visual."},{"Not":{"Contains":"visual.deepstack_merger_list"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"visual.","to":"native.visual."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"language_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"language_model.","to":"native.language_model."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t_embedder1."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"t_embedder1.","to":"native.t_embedder1."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"x_embedder."},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"final_layer2."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer2.","to":"native.final_layer2."}},"component":"model"}},
            {"Drop":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384}}},
            {"Drop":{"selector":{"predicate":{"Prefix":"text_encoders."},"minimum_matches":0,"maximum_matches":16384}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":1.0},"dtype":"f32","output":{"component":"vae","key":"pixel_space_vae"}}},
            {"Generate":{"shape":[{"Literal":1}],"fill":{"float":0.0},"dtype":"f32","output":{"component":"text_encoder","key":"_hidream_o1_te_sentinel"}}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const HIDREAM_O1_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &HIDREAM_O1_NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &HIDREAM_O1_UNPREFIXED_STATE_PLAN,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HiDreamO1Layout {
    Native,
    Unprefixed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HiDreamO1Configuration {
    pub layout: HiDreamO1Layout,
    pub patch_size: u64,
    pub input_channels: u64,
    pub patch_dimension: u64,
    pub bottleneck_dimension: u64,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub hidden_layer_count: u64,
    pub attention_head_count: u64,
    pub key_value_head_count: u64,
    pub attention_head_dimension: u64,
    pub maximum_position_embeddings: u64,
    pub vision_hidden_size: u64,
    pub vision_intermediate_size: u64,
    pub vision_depth: u64,
    pub vision_head_count: u64,
    pub vision_position_embedding_count: u64,
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<HiDreamO1Configuration, ModelFamilyError> {
    let invalid_configuration = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "HiDreamO1 configuration is invalid: {message}"
        ))
    };
    let (layout, prefix) = match probe.select_layout(HIDREAM_O1_LAYOUT_SIGNATURES)? {
        ModelStateLayout::PrefixedNative => (HiDreamO1Layout::Native, "model.diffusion_model."),
        ModelStateLayout::StandaloneNative => (HiDreamO1Layout::Unprefixed, ""),
        ModelStateLayout::Diffusers => {
            return Err(invalid_configuration(
                "Diffusers layout is unsupported".to_owned(),
            ));
        }
    };

    let timestep_shape = required_shape(
        probe,
        &format!("{prefix}t_embedder1.mlp.0.weight"),
        &invalid_configuration,
    )?;
    if timestep_shape != [4_096, 256] {
        return Err(invalid_configuration(format!(
            "t_embedder1.mlp.0.weight shape is {timestep_shape:?}, expected [4096, 256]"
        )));
    }

    let patch_shape = required_shape(
        probe,
        &format!("{prefix}x_embedder.proj1.weight"),
        &invalid_configuration,
    )?;
    let patch_dimension = HIDREAM_O1_PATCH_SIZE
        .checked_mul(HIDREAM_O1_PATCH_SIZE)
        .and_then(|area| area.checked_mul(HIDREAM_O1_LATENT_FORMAT.channels))
        .ok_or_else(|| invalid_configuration("patch dimension overflowed".to_owned()))?;
    if patch_shape != [1_024, patch_dimension] {
        return Err(invalid_configuration(format!(
            "x_embedder.proj1.weight shape is {patch_shape:?}, expected [1024, {patch_dimension}]"
        )));
    }

    Ok(HiDreamO1Configuration {
        layout,
        patch_size: HIDREAM_O1_PATCH_SIZE,
        input_channels: HIDREAM_O1_LATENT_FORMAT.channels,
        patch_dimension,
        bottleneck_dimension: 1_024,
        hidden_size: 4_096,
        intermediate_size: 12_288,
        hidden_layer_count: 36,
        attention_head_count: 32,
        key_value_head_count: 8,
        attention_head_dimension: 128,
        maximum_position_embeddings: 128_000,
        vision_hidden_size: 1_152,
        vision_intermediate_size: 4_304,
        vision_depth: 27,
        vision_head_count: 16,
        vision_position_embedding_count: 2_304,
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
        .ok_or_else(|| invalid_configuration(format!("missing required detector marker {key}")))
}
