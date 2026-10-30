use crate::{
    ModelClipTargetSelector, ModelDetectionRule, ModelDimensionExpression,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration,
    ModelFamilyStatePlanSelector,
    ModelForwardOperation, ModelForwardStep, ModelKeyPredicate,
    ModelKeyRewrite, ModelKeySelector, ModelProbe, ModelSourceConfigurationRule,
    ModelStateLayout, ModelStateTarget, ModelStateTransformOperation, ModelStateTransformPlan,
    ModelUnmatchedKeyDisposition, ModelWeightRule, qwen_image_configuration_for_probe,
    qwen_image_family::{
        QWEN_IMAGE_CLIP_TARGET, QWEN_IMAGE_COMPONENTS, QWEN_IMAGE_COMPONENT_STATE_SCHEMAS,
        QWEN_IMAGE_MEMORY_ESTIMATOR, QWEN_IMAGE_MODEL_REQUIRED_KEYS,
        QWEN_IMAGE_SUPPORTED_DEVICES, QWEN_IMAGE_SUPPORTED_DTYPES, QwenImageConfiguration,
        QwenImageReferenceMethod,
    },
};
use crate::model_family::{
    MODEL_FAMILY_EXECUTION_PROJECTION_SCHEMA_VERSION,
    ModelFamilyExecutionProjectionDefinition, ModelFamilyExecutionProjectionTensorDefinition,
    ModelFamilyExecutionProjectionValue,
};
use sim_tensor::{DType, Scalar};

pub const MODEL_FAMILY_IDENTIFIER: &str = "QwenImage";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0113";
pub const MODEL_FAMILY_FIXTURE: &str = "qwenimage-comfy-model-0113";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 77;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "27138e5691f93151df4199f0c0a03a3e5df5908168a82893466a07944d835753";
pub const MODEL_FAMILY_CATALOG_ROW_SHA256: &str =
    "89cf6c7d714b4dcbd548549de08144a0ea4c243d7541417ae0ebb74373d0024b";
pub const MODEL_FAMILY_SAMPLING_MULTIPLIER: f64 = 1.0;
pub const MODEL_FAMILY_SAMPLING_SHIFT: f64 = 1.15;
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = 1.8;

pub const DENOISER_INVOCATION_REQUIRED_KEYS: &[&str] = &[
    "native.img_in.weight",
    "native.img_in.bias",
    "native.txt_norm.weight",
    "native.txt_in.weight",
    "native.txt_in.bias",
    "native.time_text_embed.timestep_embedder.linear_1.weight",
    "native.time_text_embed.timestep_embedder.linear_1.bias",
    "native.time_text_embed.timestep_embedder.linear_2.weight",
    "native.time_text_embed.timestep_embedder.linear_2.bias",
    "native.transformer_blocks.0.img_mod.1.weight",
    "native.transformer_blocks.0.img_mod.1.bias",
    "native.transformer_blocks.0.txt_mod.1.weight",
    "native.transformer_blocks.0.txt_mod.1.bias",
    "native.transformer_blocks.0.attn.norm_q.weight",
    "native.transformer_blocks.0.attn.norm_k.weight",
    "native.transformer_blocks.0.attn.norm_added_q.weight",
    "native.transformer_blocks.0.attn.norm_added_k.weight",
    "native.transformer_blocks.0.attn.to_q.weight",
    "native.transformer_blocks.0.attn.to_q.bias",
    "native.transformer_blocks.0.attn.to_k.weight",
    "native.transformer_blocks.0.attn.to_k.bias",
    "native.transformer_blocks.0.attn.to_v.weight",
    "native.transformer_blocks.0.attn.to_v.bias",
    "native.transformer_blocks.0.attn.add_q_proj.weight",
    "native.transformer_blocks.0.attn.add_q_proj.bias",
    "native.transformer_blocks.0.attn.add_k_proj.weight",
    "native.transformer_blocks.0.attn.add_k_proj.bias",
    "native.transformer_blocks.0.attn.add_v_proj.weight",
    "native.transformer_blocks.0.attn.add_v_proj.bias",
    "native.transformer_blocks.0.attn.to_out.0.weight",
    "native.transformer_blocks.0.attn.to_out.0.bias",
    "native.transformer_blocks.0.attn.to_add_out.weight",
    "native.transformer_blocks.0.attn.to_add_out.bias",
    "native.transformer_blocks.0.img_mlp.net.0.proj.weight",
    "native.transformer_blocks.0.img_mlp.net.0.proj.bias",
    "native.transformer_blocks.0.img_mlp.net.2.weight",
    "native.transformer_blocks.0.img_mlp.net.2.bias",
    "native.transformer_blocks.0.txt_mlp.net.0.proj.weight",
    "native.transformer_blocks.0.txt_mlp.net.0.proj.bias",
    "native.transformer_blocks.0.txt_mlp.net.2.weight",
    "native.transformer_blocks.0.txt_mlp.net.2.bias",
    "native.norm_out.linear.weight",
    "native.norm_out.linear.bias",
    "native.proj_out.weight",
    "native.proj_out.bias",
];
pub const DENOISER_INVOCATION_LATENT_RANK: usize = 5;
pub const DENOISER_INVOCATION_CHANNELS: usize = 16;
pub const DENOISER_INVOCATION_CONTEXT_WIDTH: usize = 3_584;
pub const DENOISER_INVOCATION_HEAD_WIDTH: usize = 128;
pub const DENOISER_INVOCATION_MLP_WIDTH: usize = 512;
pub const DENOISER_INVOCATION_PATCH_SIZE: usize = 2;

macro_rules! execution_tensor {
    ($key:literal, $shape:expr) => {
        ModelFamilyExecutionProjectionTensorDefinition {
            key: $key,
            shape: $shape,
            value: ModelFamilyExecutionProjectionValue::Finite,
        }
    };
    ($key:literal, $shape:expr, $bits:expr) => {
        ModelFamilyExecutionProjectionTensorDefinition {
            key: $key,
            shape: $shape,
            value: ModelFamilyExecutionProjectionValue::ExactF32($bits),
        }
    };
}

const EXECUTION_PROJECTION_TENSORS: &[ModelFamilyExecutionProjectionTensorDefinition] = &[
    execution_tensor!("native.img_in.weight", &[128, 64]),
    execution_tensor!("native.img_in.bias", &[128]),
    execution_tensor!("native.txt_norm.weight", &[3_584]),
    execution_tensor!("native.txt_in.weight", &[128, 3_584]),
    execution_tensor!("native.txt_in.bias", &[128]),
    execution_tensor!("native.time_text_embed.timestep_embedder.linear_1.weight", &[128, 256]),
    execution_tensor!("native.time_text_embed.timestep_embedder.linear_1.bias", &[128]),
    execution_tensor!("native.time_text_embed.timestep_embedder.linear_2.weight", &[128, 128]),
    execution_tensor!("native.time_text_embed.timestep_embedder.linear_2.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.img_mod.1.weight", &[768, 128]),
    execution_tensor!("native.transformer_blocks.0.img_mod.1.bias", &[768]),
    execution_tensor!("native.transformer_blocks.0.txt_mod.1.weight", &[768, 128]),
    execution_tensor!("native.transformer_blocks.0.txt_mod.1.bias", &[768]),
    execution_tensor!("native.transformer_blocks.0.attn.norm_q.weight", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.norm_k.weight", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.norm_added_q.weight", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.norm_added_k.weight", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_q.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_q.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_k.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_k.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_v.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_v.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.add_q_proj.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.add_q_proj.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.add_k_proj.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.add_k_proj.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.add_v_proj.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.add_v_proj.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_out.0.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_out.0.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_add_out.weight", &[128, 128]),
    execution_tensor!("native.transformer_blocks.0.attn.to_add_out.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.img_mlp.net.0.proj.weight", &[512, 128]),
    execution_tensor!("native.transformer_blocks.0.img_mlp.net.0.proj.bias", &[512]),
    execution_tensor!("native.transformer_blocks.0.img_mlp.net.2.weight", &[128, 512]),
    execution_tensor!("native.transformer_blocks.0.img_mlp.net.2.bias", &[128]),
    execution_tensor!("native.transformer_blocks.0.txt_mlp.net.0.proj.weight", &[512, 128]),
    execution_tensor!("native.transformer_blocks.0.txt_mlp.net.0.proj.bias", &[512]),
    execution_tensor!("native.transformer_blocks.0.txt_mlp.net.2.weight", &[128, 512]),
    execution_tensor!("native.transformer_blocks.0.txt_mlp.net.2.bias", &[128]),
    execution_tensor!("native.norm_out.linear.weight", &[256, 128]),
    execution_tensor!("native.norm_out.linear.bias", &[256]),
    execution_tensor!("native.proj_out.weight", &[64, 128]),
    execution_tensor!("native.proj_out.bias", &[64]),
    execution_tensor!("native.__sampling_shift__", &[1], 1.15_f32.to_bits()),
    execution_tensor!("native.__reference_method__", &[1], 0.0_f32.to_bits()),
    execution_tensor!("native.__additional_timestep_condition__", &[1], 0.0_f32.to_bits()),
];

pub(crate) const DENOISER_EXECUTION_PROJECTION: ModelFamilyExecutionProjectionDefinition =
    ModelFamilyExecutionProjectionDefinition {
        schema_version: MODEL_FAMILY_EXECUTION_PROJECTION_SCHEMA_VERSION,
        identifier: "qwen-image-reduced-denoiser-v1",
        family_feature_id: MODEL_FAMILY_FEATURE_ID,
        family_identifier: MODEL_FAMILY_IDENTIFIER,
        source_ordinal: MODEL_FAMILY_SOURCE_ORDINAL,
        source_architecture: "model_base.QwenImage",
        tensors: EXECUTION_PROJECTION_TENSORS,
    };

const DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.transformer_blocks.0.txt_mod.1.weight",
            "transformer_blocks.0.txt_mod.1.weight",
            "transformer.transformer_blocks.0.txt_mod.1.weight",
        ],
        score: 1_600,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer.transformer_blocks.0.attn.norm_q.weight",
        ],
        score: 800,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: &[
            "model.diffusion_model.txt_norm.weight",
            "txt_norm.weight",
            "transformer.txt_norm.weight",
        ],
        score: 400,
    },
];

const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

const OPTIONAL_KEYS: &[&str] = &[
    "native.__index_timestep_zero__",
    "native.time_text_embed.addition_t_embedding.weight",
    "native.__sampling_shift__",
    "native.__reference_method__",
    "native.__additional_timestep_condition__",
];

const FORWARD_PROGRAM: &[ModelForwardStep] = &[
    ModelForwardStep {
        checkpoint: "image_patch_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.img_in.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "text_conditioning_normalization",
        operation: ModelForwardOperation::LayerNorm {
            normalized_shape: &[2],
            weight: Some("native.txt_norm.weight"),
            bias: None,
            epsilon: 1.0e-6,
        },
    },
    ModelForwardStep {
        checkpoint: "text_conditioning_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.txt_in.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "text_stream_modulation",
        operation: ModelForwardOperation::Linear {
            weight: "native.transformer_blocks.0.txt_mod.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "reference_image_modulation",
        operation: ModelForwardOperation::Linear {
            weight: "native.transformer_blocks.0.img_mod.1.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "joint_attention_query",
        operation: ModelForwardOperation::Linear {
            weight: "native.transformer_blocks.0.attn.to_q.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "joint_attention",
        operation: ModelForwardOperation::SelfAttention { heads: 1 },
    },
    ModelForwardStep {
        checkpoint: "image_output_projection",
        operation: ModelForwardOperation::Linear {
            weight: "native.proj_out.weight",
            bias: None,
            input_features: 2,
            output_features: 2,
        },
    },
    ModelForwardStep {
        checkpoint: "image_output",
        operation: ModelForwardOperation::Tanh,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "qwen-image-mmdit-v1",
    latent_feature_id: "COMFY-MODEL-0053",
    latent_identifier: "Wan21",
    clip_target: &QWEN_IMAGE_CLIP_TARGET,
    components: QWEN_IMAGE_COMPONENTS,
    detection_rules: DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: QWEN_IMAGE_MODEL_REQUIRED_KEYS,
    optional_keys: OPTIONAL_KEYS,
    supported_dtypes: QWEN_IMAGE_SUPPORTED_DTYPES,
    supported_devices: QWEN_IMAGE_SUPPORTED_DEVICES,
    memory_estimator: QWEN_IMAGE_MEMORY_ESTIMATOR,
    forward_program: FORWARD_PROGRAM,
};

const SOURCE_CONFIGURATION: &[ModelSourceConfigurationRule] = &[];

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 77,
    source_architecture: "model_base.QwenImage",
    source_configuration: SOURCE_CONFIGURATION,
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Profile,
    state_plan_selector: ModelFamilyStatePlanSelector::Probe(state_plan_for_probe),
    component_state_schemas: QWEN_IMAGE_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    Ok(ModelFamilyProfile {
        latent_feature_id: configuration.latent_format.feature_id,
        latent_identifier: configuration.latent_format.identifier,
        clip_target: configuration.clip_target,
        supported_dtypes: configuration.supported_dtypes,
        supported_devices: configuration.supported_devices,
        memory_estimator: configuration.memory_estimator,
        forward_program: FORWARD_PROGRAM,
    })
}

pub fn configuration_for_probe(
    probe: &ModelProbe,
) -> Result<QwenImageConfiguration, ModelFamilyError> {
    qwen_image_configuration_for_probe(probe)
}

fn state_plan_for_probe(probe: &ModelProbe) -> Result<ModelStateTransformPlan, ModelFamilyError> {
    let configuration = configuration_for_probe(probe)?;
    let mut operations = Vec::new();
    match configuration.layout {
        ModelStateLayout::PrefixedNative => {
            operations.push(move_prefix(
                "model.diffusion_model.",
                "native.",
                "model",
                1,
                16_384,
            )?);
        }
        ModelStateLayout::StandaloneNative => {
            for (from, to, minimum, maximum) in [
                ("txt_norm.", "native.txt_norm.", 1, 64),
                ("img_in.", "native.img_in.", 1, 64),
                ("txt_in.", "native.txt_in.", 1, 64),
                (
                    "transformer_blocks.",
                    "native.transformer_blocks.",
                    1,
                    16_384,
                ),
                ("norm_out.", "native.norm_out.", 0, 64),
                ("proj_out.", "native.proj_out.", 1, 64),
                (
                    "time_text_embed.",
                    "native.time_text_embed.",
                    0,
                    128,
                ),
            ] {
                operations.push(move_prefix(from, to, "model", minimum, maximum)?);
            }
            operations.push(move_optional_exact(
                "__index_timestep_zero__",
                "native.__index_timestep_zero__",
                "model",
            )?);
        }
        ModelStateLayout::Diffusers => {
            return Err(ModelFamilyError::InvalidSelectorOutput(
                "QwenImage Diffusers state dictionaries are not admitted".to_owned(),
            ));
        }
    }
    operations.push(move_prefix("vae.", "vae.", "vae", 0, 16_384)?);
    operations.push(move_prefix(
        "text_encoders.",
        "text_encoder.",
        "text_encoder",
        0,
        16_384,
    )?);
    operations.extend(conditioning_operations(&configuration)?);
    ModelStateTransformPlan::checked(operations, ModelUnmatchedKeyDisposition::Reject)
}

fn conditioning_operations(
    configuration: &QwenImageConfiguration,
) -> Result<Vec<ModelStateTransformOperation>, ModelFamilyError> {
    let reference_method = match configuration.reference_method {
        QwenImageReferenceMethod::Index => 0.0,
        QwenImageReferenceMethod::IndexTimestepZero => 1.0,
        QwenImageReferenceMethod::NegativeIndex => -1.0,
    };
    Ok(vec![
        generate_fact(
            "native.__sampling_shift__",
            configuration.sampling_shift,
        )?,
        generate_fact("native.__reference_method__", reference_method)?,
        generate_fact(
            "native.__additional_timestep_condition__",
            f64::from(u8::from(configuration.use_additional_timestep_condition)),
        )?,
    ])
}

fn move_prefix(
    from: &str,
    to: &str,
    component: &str,
    minimum_matches: usize,
    maximum_matches: usize,
) -> Result<ModelStateTransformOperation, ModelFamilyError> {
    Ok(ModelStateTransformOperation::Move {
        selector: ModelKeySelector::bounded(
            ModelKeyPredicate::prefix(from)?,
            minimum_matches,
            maximum_matches,
        )?,
        rewrite: ModelKeyRewrite::prefix(from, to)?,
        component: component.to_owned(),
    })
}

fn move_optional_exact(
    source: &str,
    target: &str,
    component: &str,
) -> Result<ModelStateTransformOperation, ModelFamilyError> {
    Ok(ModelStateTransformOperation::Move {
        selector: ModelKeySelector::bounded(ModelKeyPredicate::exact(source)?, 0, 1)?,
        rewrite: ModelKeyRewrite::exact(target)?,
        component: component.to_owned(),
    })
}

fn generate_fact(
    key: &str,
    value: f64,
) -> Result<ModelStateTransformOperation, ModelFamilyError> {
    Ok(ModelStateTransformOperation::Generate {
        shape: vec![ModelDimensionExpression::Literal(1)],
        fill: Scalar::Float(value),
        dtype: DType::F32,
        output: ModelStateTarget::checked("model", key)?,
    })
}
