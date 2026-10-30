use crate::{
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelClipConfigurationFactDefinition, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetDefinition, ModelClipTargetSelector,
    ModelDetectionRule, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanSelector, ModelForwardOperation,
    ModelForwardStep, ModelProbe, ModelStateLayout, ModelStateTransformPlanDefinition,
    ModelWeightRule,
    flux_chroma_family::{
        CHROMA_LAYOUT_SIGNATURES, FluxChromaConfiguration, FluxChromaFinalHead, FluxChromaLayout,
        FluxChromaVariant, configuration_for_probe as flux_chroma_configuration_for_probe,
    },
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const MODEL_FAMILY_IDENTIFIER: &str = "ChromaRadiance";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0067";
pub const MODEL_FAMILY_FIXTURE: &str = "chromaradiance-comfy-model-0067";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 72;
pub const SOURCE_ARCHITECTURE: &str = "model_base.ChromaRadiance";
pub const SOURCE_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const SOURCE_MEMORY_USAGE_FACTOR: f64 = 0.044;
pub const SOURCE_NERF_HIDDEN_SIZE: u64 = 64;
pub const SOURCE_NERF_MLP_RATIO: u64 = 4;
pub const SOURCE_NERF_DEPTH: usize = 4;
pub const SOURCE_NERF_MAX_FREQUENCIES: u64 = 8;
pub const SOURCE_NERF_TILE_SIZE: u64 = 512;

pub type ChromaRadianceLayout = FluxChromaLayout;
pub type ChromaRadianceFinalHead = FluxChromaFinalHead;
pub type ChromaRadianceConfiguration = FluxChromaConfiguration;

const CLIP_CONFIGURATION: &[ModelClipConfigurationFactDefinition] =
    &[ModelClipConfigurationFactDefinition::Expand {
        source: "sim.text_encoders.sd3_clip.t5_xxl_detect",
    }];

const CLIP_CANDIDATES: &[ModelClipTargetCandidateDefinition] =
    &[ModelClipTargetCandidateDefinition {
        tokenizer: "sim.text_encoders.pixart_t5.PixArtTokenizer",
        clip_model: "sim.text_encoders.pixart_t5.pixart_te",
        invocation: ModelClipModelInvocationDefinition::Factory {
            configuration: CLIP_CONFIGURATION,
        },
    }];

const CLIP_TARGET: ModelClipTargetDefinition = ModelClipTargetDefinition {
    candidates: CLIP_CANDIDATES,
    dynamic_selection: false,
};

const COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "denoiser",
        role: "pixel-space diffusion transformer and NeRF head",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "pixel-space latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "text_encoder",
        role: "T5 conditioning",
        required: false,
    },
];

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight",
            "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
            "double_blocks.0.img_attn.norm.key_norm.weight",
            "double_blocks.0.img_attn.norm.key_norm.scale",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.img_in_patch.weight",
            "img_in_patch.weight",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.nerf_blocks.0.norm.weight",
            "model.diffusion_model.nerf_blocks.0.norm.scale",
            "nerf_blocks.0.norm.weight",
            "nerf_blocks.0.norm.scale",
        ],
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.nerf_final_layer.linear.weight",
            "model.diffusion_model.nerf_final_layer_conv.conv.weight",
            "nerf_final_layer.linear.weight",
            "nerf_final_layer_conv.conv.weight",
        ],
        score: 250,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const REQUIRED_KEYS: &[&str] = &["native.nerf_blocks.0.norm.weight"];

const OPTIONAL_KEYS: &[&str] = &[
    "native.img_in_patch.weight",
    "native.img_in_patch.bias",
    "native.txt_in.weight",
    "native.double_blocks.0.img_attn.norm.key_norm.weight",
    "native.single_blocks.0.linear1.weight",
    "native.distilled_guidance_layer.norms.0.weight",
    "native.nerf_blocks.0.param_generator.weight",
    "native.nerf_blocks.1.norm.weight",
    "native.nerf_blocks.2.norm.weight",
    "native.nerf_blocks.3.norm.weight",
    "native.nerf_final_layer.norm.weight",
    "native.nerf_final_layer.linear.weight",
    "native.nerf_final_layer.linear.bias",
    "native.nerf_final_layer_conv.norm.weight",
    "native.nerf_final_layer_conv.conv.weight",
    "native.nerf_final_layer_conv.conv.bias",
    "native.__x0__",
    "native.__sequential__",
];

const COMPONENT_OPTIONAL_KEYS: &[&str] = &[
    "native.single_blocks.0.linear1.weight",
    "native.nerf_blocks.0.param_generator.weight",
    "native.nerf_blocks.1.norm.weight",
    "native.nerf_blocks.2.norm.weight",
    "native.nerf_blocks.3.norm.weight",
    "native.nerf_final_layer.norm.weight",
    "native.nerf_final_layer.linear.weight",
    "native.nerf_final_layer.linear.bias",
    "native.nerf_final_layer_conv.norm.weight",
    "native.nerf_final_layer_conv.conv.weight",
    "native.nerf_final_layer_conv.conv.bias",
    "native.__x0__",
    "native.__sequential__",
];

const SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
const SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const LINEAR_FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "nerf_final_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: Some("native.nerf_final_layer.norm.weight"),
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "linear_radiance_output",
        operation: ModelForwardOperation::Linear {
            weight: "native.nerf_final_layer.linear.weight",
            bias: Some("native.nerf_final_layer.linear.bias"),
            input_features: 2,
            output_features: 2,
        },
    },
];

const CONVOLUTION_FORWARD_PROGRAM: &[ModelForwardStep] = &[ModelForwardStep {
    checkpoint: "convolution_radiance_output",
    operation: ModelForwardOperation::Convolution2d {
        weight: "native.nerf_final_layer_conv.conv.weight",
        bias: Some("native.nerf_final_layer_conv.conv.bias"),
        input_channels: 2,
        output_channels: 3,
        kernel_size: [1, 1],
        stride: [1, 1],
        padding: [0, 0],
        dilation: [1, 1],
        groups: 1,
    },
}];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "chroma-radiance-v1",
    latent_feature_id: "COMFY-MODEL-0025",
    latent_identifier: "ChromaRadiance",
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
        bytes_per_parameter: 1,
        activation_bytes_per_element: 1,
    },
    forward_program: LINEAR_FORWARD_PROGRAM,
};

const NATIVE_STATE_PLAN: ModelStateTransformPlanDefinition = ModelStateTransformPlanDefinition {
    schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
    encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"denoiser"}}
        ],
        "unmatched":"Reject"
    }"#,
};

const UNPREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"Prefix":"first_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"first_stage_model.","to":"vae."}},"component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"cond_stage_model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_stage_model.","to":"text_encoder."}},"component":"text_encoder"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"img_in_patch."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"img_in_patch.","to":"native.img_in_patch."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"img_in_patch."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"img_in_patch.","to":"native.img_in_patch."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"txt_in."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"txt_in.","to":"native.txt_in."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"txt_in."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"txt_in.","to":"native.txt_in."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"distilled_guidance_layer."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"distilled_guidance_layer.","to":"native.distilled_guidance_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"distilled_guidance_layer."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"distilled_guidance_layer.","to":"native.distilled_guidance_layer."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_image_embedder."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"nerf_image_embedder.","to":"native.nerf_image_embedder."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_image_embedder."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"nerf_image_embedder.","to":"native.nerf_image_embedder."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"nerf_blocks.","to":"native.nerf_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"nerf_blocks.","to":"native.nerf_blocks."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_final_layer."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"nerf_final_layer.","to":"native.nerf_final_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_final_layer."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"nerf_final_layer.","to":"native.nerf_final_layer."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_final_layer_conv."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"nerf_final_layer_conv.","to":"native.nerf_final_layer_conv."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"nerf_final_layer_conv."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"nerf_final_layer_conv.","to":"native.nerf_final_layer_conv."}},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"__x0__"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.__x0__"},"component":"denoiser"}},
            {"Move":{"selector":{"predicate":{"Exact":"__sequential__"},"minimum_matches":0,"maximum_matches":1},"rewrite":{"Exact":"native.__sequential__"},"component":"denoiser"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &NATIVE_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &UNPREFIXED_STATE_PLAN,
    },
];

const COMPONENT_STATE_SCHEMAS: &[ModelFamilyComponentStateSchema] = &[
    ModelFamilyComponentStateSchema {
        component: "denoiser",
        required_keys: &[
            "native.img_in_patch.weight",
            "native.img_in_patch.bias",
            "native.txt_in.weight",
            "native.double_blocks.0.img_attn.norm.key_norm.weight",
            "native.distilled_guidance_layer.norms.0.weight",
            "native.nerf_blocks.0.norm.weight",
        ],
        optional_keys: COMPONENT_OPTIONAL_KEYS,
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
    source_ordinal: 72,
    source_architecture: SOURCE_ARCHITECTURE,
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: CHROMA_LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    let mut profile = ModelFamilyProfile::from_definition(&MODEL_FAMILY);
    profile.forward_program = match configuration.final_head {
        ChromaRadianceFinalHead::Linear => LINEAR_FORWARD_PROGRAM,
        ChromaRadianceFinalHead::Convolution => CONVOLUTION_FORWARD_PROGRAM,
    };
    Ok(profile)
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<ChromaRadianceConfiguration, ModelFamilyError> {
    flux_chroma_configuration_for_probe(
        probe,
        FluxChromaVariant::ChromaRadiance,
        MODEL_FAMILY_IDENTIFIER,
    )
}
