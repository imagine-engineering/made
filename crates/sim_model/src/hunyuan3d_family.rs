use crate::{
    LatentFormatDefinition, MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, ModelFamilyComponent,
    ModelFamilyError, ModelFamilyStatePlanCase, ModelStateLayout,
    ModelStateTransformPlanDefinition,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

pub const HUNYUAN3D_MEMORY_USAGE_FACTOR: f64 = 3.5;
pub const HUNYUAN3D_NUMBER_OF_HEADS: u64 = 16;
pub const HUNYUAN3D_MLP_RATIO: f64 = 4.0;
pub const HUNYUAN3D_V21_CONTEXT_DIMENSION: u64 = 1_024;
pub const HUNYUAN3D_MINI_DEPTH: usize = 8;
pub const HUNYUAN3D_SCALE_SUFFIX: &str = ".scale";
pub const HUNYUAN3D_WEIGHT_SUFFIX: &str = ".weight";

pub const HUNYUAN3D_V2_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanthree_dv2_sim_model_0032::LATENT_FORMAT;
pub const HUNYUAN3D_V21_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanthree_dv2_1_sim_model_0033::LATENT_FORMAT;
pub const HUNYUAN3D_MINI_LATENT_FORMAT: &LatentFormatDefinition =
    &crate::generated_hunyuanthree_dv2mini_sim_model_0034::LATENT_FORMAT;

pub const HUNYUAN3D_COMPONENTS: &[ModelFamilyComponent] = &[
    ModelFamilyComponent {
        identifier: "model",
        role: "Hunyuan3D native diffusion transformer",
        required: true,
    },
    ModelFamilyComponent {
        identifier: "vae",
        role: "Hunyuan3D latent codec",
        required: false,
    },
    ModelFamilyComponent {
        identifier: "clip_vision",
        role: "Hunyuan3D main image encoder",
        required: false,
    },
];

pub const HUNYUAN3D_SUPPORTED_DTYPES: &[DType] = &[DType::Bf16, DType::F16, DType::F32];
pub const HUNYUAN3D_SUPPORTED_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Hunyuan3DVariant {
    V2,
    V2_1,
    V2Mini,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Hunyuan3DLayout {
    PrefixedNative,
    SavedModel,
    StandaloneNative,
}

#[derive(Clone, Copy, Debug)]
pub struct Hunyuan3DConfiguration {
    pub variant: Hunyuan3DVariant,
    pub layout: Hunyuan3DLayout,
    pub in_channels: u64,
    pub context_dimension: u64,
    pub hidden_size: u64,
    pub mlp_ratio: f64,
    pub number_of_heads: u64,
    pub depth: usize,
    pub single_block_depth: usize,
    pub qkv_bias: bool,
    pub guidance_embedding: bool,
    pub memory_usage_factor: f64,
    pub latent_format: &'static LatentFormatDefinition,
}

#[derive(Clone, Copy, Debug)]
pub struct Hunyuan3DCommonMapping {
    pub components: &'static [ModelFamilyComponent],
    pub supported_dtypes: &'static [DType],
    pub supported_devices: &'static [DeviceKind],
    pub scale_suffix: &'static str,
    pub weight_suffix: &'static str,
    pub memory_usage_factor: f64,
}

pub static HUNYUAN3D_COMMON_MAPPING: Hunyuan3DCommonMapping = Hunyuan3DCommonMapping {
    components: HUNYUAN3D_COMPONENTS,
    supported_dtypes: HUNYUAN3D_SUPPORTED_DTYPES,
    supported_devices: HUNYUAN3D_SUPPORTED_DEVICES,
    scale_suffix: HUNYUAN3D_SCALE_SUFFIX,
    weight_suffix: HUNYUAN3D_WEIGHT_SUFFIX,
    memory_usage_factor: HUNYUAN3D_MEMORY_USAGE_FACTOR,
};

pub fn common_mapping() -> &'static Hunyuan3DCommonMapping {
    &HUNYUAN3D_COMMON_MAPPING
}

pub const HUNYUAN3D_PREFIXED_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"model.diffusion_model.","to":"native."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model.diffusion_model."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.diffusion_model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.main_image_encoder.model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.main_image_encoder.model.","to":"clip_vision."}},"component":"clip_vision"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const HUNYUAN3D_SAVED_MODEL_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"model.","to":"native."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"model."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":1,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"model.","to":"native."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.main_image_encoder.model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.main_image_encoder.model.","to":"clip_vision."}},"component":"clip_vision"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const HUNYUAN3D_STANDALONE_STATE_PLAN: ModelStateTransformPlanDefinition =
    ModelStateTransformPlanDefinition {
        schema_version: MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION,
        encoded_plan: r#"{
        "operations": [
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"latent_in."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"latent_in.","to":"native.latent_in."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"latent_in."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"latent_in.","to":"native.latent_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"cond_in."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"cond_in.","to":"native.cond_in."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"cond_in."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"cond_in.","to":"native.cond_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"double_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"double_blocks.","to":"native.double_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"single_blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"single_blocks.","to":"native.single_blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"time_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"time_in.","to":"native.time_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"guidance_in."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"guidance_in.","to":"native.guidance_in."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"x_embedder."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"x_embedder."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"x_embedder.","to":"native.x_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"t_embedder."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"t_embedder.","to":"native.t_embedder."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"blocks."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"blocks.","to":"native.blocks."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"blocks."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"blocks.","to":"native.blocks."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"final_layer."},{"Suffix":".scale"}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Pipeline":[{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},{"Suffix":{"from":".scale","to":".weight"}}]},"component":"model"}},
            {"Move":{"selector":{"predicate":{"All":[{"Prefix":"final_layer."},{"Not":{"Suffix":".scale"}}]},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"final_layer.","to":"native.final_layer."}},"component":"model"}},
            {"Move":{"selector":{"predicate":{"Prefix":"vae."},"minimum_matches":0,"maximum_matches":16384},"rewrite":"Identity","component":"vae"}},
            {"Move":{"selector":{"predicate":{"Prefix":"conditioner.main_image_encoder.model."},"minimum_matches":0,"maximum_matches":16384},"rewrite":{"Prefix":{"from":"conditioner.main_image_encoder.model.","to":"clip_vision."}},"component":"clip_vision"}}
        ],
        "unmatched":"Reject"
    }"#,
    };

pub const HUNYUAN3D_STANDARD_STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &HUNYUAN3D_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &HUNYUAN3D_STANDALONE_STATE_PLAN,
    },
];

pub fn state_plan_for_layout(
    layout: Hunyuan3DLayout,
) -> &'static ModelStateTransformPlanDefinition {
    match layout {
        Hunyuan3DLayout::PrefixedNative => &HUNYUAN3D_PREFIXED_STATE_PLAN,
        Hunyuan3DLayout::SavedModel => &HUNYUAN3D_SAVED_MODEL_STATE_PLAN,
        Hunyuan3DLayout::StandaloneNative => &HUNYUAN3D_STANDALONE_STATE_PLAN,
    }
}

pub fn configuration_for_probe(
    probe: &crate::ModelProbe,
) -> Result<Hunyuan3DConfiguration, ModelFamilyError> {
    let invalid = |message: String| {
        ModelFamilyError::InvalidSelectorOutput(format!(
            "Hunyuan3D configuration is invalid: {message}"
        ))
    };
    let domains = [
        (Hunyuan3DLayout::PrefixedNative, "model.diffusion_model."),
        (Hunyuan3DLayout::SavedModel, "model."),
        (Hunyuan3DLayout::StandaloneNative, ""),
    ];
    let mut matches = Vec::new();
    let mut partial_v21 = false;
    for (layout, prefix) in domains {
        let classic = probe
            .tensor_shapes
            .contains_key(&format!("{prefix}latent_in.weight"));
        let v21_markers = [
            format!("{prefix}t_embedder.mlp.2.weight"),
            format!("{prefix}blocks.0.attn1.k_norm.weight"),
            format!("{prefix}x_embedder.weight"),
        ];
        let v21_count = v21_markers
            .iter()
            .filter(|key| probe.tensor_shapes.contains_key(*key))
            .count();
        if v21_count > 0 && v21_count < v21_markers.len() {
            partial_v21 = true;
        }
        if classic && v21_count == v21_markers.len() {
            return Err(invalid(format!(
                "layout {layout:?} mixes classic and 2.1 markers"
            )));
        }
        if classic || v21_count == v21_markers.len() {
            matches.push((layout, prefix, classic));
        }
    }
    let (layout, prefix, classic) = match matches.as_slice() {
        [entry] => *entry,
        [] if partial_v21 => {
            return Err(invalid("partial Hunyuan3D 2.1 marker set".to_owned()));
        }
        [] => {
            return Err(ModelFamilyError::ModelLayoutSelection(
                "parsed tensor keys match no Hunyuan3D layout".to_owned(),
            ));
        }
        _ => {
            return Err(ModelFamilyError::ModelLayoutSelection(
                "parsed tensor keys ambiguously match multiple Hunyuan3D layouts".to_owned(),
            ));
        }
    };

    let genmo_collision = [
        format!("{prefix}t5_yproj.weight"),
        format!("{prefix}blocks.0.attn.proj_x.weight"),
    ]
    .iter()
    .any(|key| probe.tensor_shapes.contains_key(key));
    if genmo_collision {
        return Err(invalid(
            "Genmo Mochi markers collide with Hunyuan3D".to_owned(),
        ));
    }

    if classic {
        classic_configuration(probe, layout, prefix, &invalid)
    } else {
        v21_configuration(probe, layout, prefix, &invalid)
    }
}

fn classic_configuration(
    probe: &crate::ModelProbe,
    layout: Hunyuan3DLayout,
    prefix: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<Hunyuan3DConfiguration, ModelFamilyError> {
    let latent = required_matrix(probe, &format!("{prefix}latent_in.weight"), invalid)?;
    let conditioning = required_matrix(probe, &format!("{prefix}cond_in.weight"), invalid)?;
    if latent[0] != conditioning[0] {
        return Err(invalid(format!(
            "latent/conditioning hidden sizes differ: {} versus {}",
            latent[0], conditioning[0]
        )));
    }
    checked_head_geometry(latent[0], invalid)?;
    let depth = checked_depth(probe, &format!("{prefix}double_blocks."), invalid)?;
    let single_block_depth = checked_depth(probe, &format!("{prefix}single_blocks."), invalid)?;
    let variant = if depth == HUNYUAN3D_MINI_DEPTH {
        Hunyuan3DVariant::V2Mini
    } else {
        Hunyuan3DVariant::V2
    };
    let latent_format = match variant {
        Hunyuan3DVariant::V2Mini => HUNYUAN3D_MINI_LATENT_FORMAT,
        Hunyuan3DVariant::V2 => HUNYUAN3D_V2_LATENT_FORMAT,
        Hunyuan3DVariant::V2_1 => unreachable!("classic detector cannot select 2.1"),
    };
    Ok(Hunyuan3DConfiguration {
        variant,
        layout,
        in_channels: latent[1],
        context_dimension: conditioning[1],
        hidden_size: latent[0],
        mlp_ratio: HUNYUAN3D_MLP_RATIO,
        number_of_heads: HUNYUAN3D_NUMBER_OF_HEADS,
        depth,
        single_block_depth,
        qkv_bias: true,
        guidance_embedding: probe
            .tensor_shapes
            .contains_key(&format!("{prefix}guidance_in.in_layer.weight")),
        memory_usage_factor: HUNYUAN3D_MEMORY_USAGE_FACTOR,
        latent_format,
    })
}

fn v21_configuration(
    probe: &crate::ModelProbe,
    layout: Hunyuan3DLayout,
    prefix: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<Hunyuan3DConfiguration, ModelFamilyError> {
    let embedding = required_matrix(probe, &format!("{prefix}x_embedder.weight"), invalid)?;
    checked_head_geometry(embedding[0], invalid)?;
    let depth = checked_depth(probe, &format!("{prefix}blocks."), invalid)?;
    Ok(Hunyuan3DConfiguration {
        variant: Hunyuan3DVariant::V2_1,
        layout,
        in_channels: embedding[1],
        context_dimension: HUNYUAN3D_V21_CONTEXT_DIMENSION,
        hidden_size: embedding[0],
        mlp_ratio: HUNYUAN3D_MLP_RATIO,
        number_of_heads: HUNYUAN3D_NUMBER_OF_HEADS,
        depth,
        single_block_depth: 0,
        qkv_bias: false,
        guidance_embedding: false,
        memory_usage_factor: HUNYUAN3D_MEMORY_USAGE_FACTOR,
        latent_format: HUNYUAN3D_V21_LATENT_FORMAT,
    })
}

fn required_matrix<'a>(
    probe: &'a crate::ModelProbe,
    key: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<&'a [u64], ModelFamilyError> {
    let shape = probe
        .tensor_shapes
        .get(key)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("missing required tensor {key}")))?;
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(invalid(format!("tensor {key} is not a non-empty matrix")));
    }
    Ok(shape)
}

fn checked_head_geometry(
    hidden_size: u64,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<(), ModelFamilyError> {
    if !hidden_size.is_multiple_of(HUNYUAN3D_NUMBER_OF_HEADS) {
        return Err(invalid(format!(
            "hidden size {hidden_size} is not divisible by {} heads",
            HUNYUAN3D_NUMBER_OF_HEADS
        )));
    }
    Ok(())
}

fn checked_depth(
    probe: &crate::ModelProbe,
    prefix: &str,
    invalid: &impl Fn(String) -> ModelFamilyError,
) -> Result<usize, ModelFamilyError> {
    let depth = probe.consecutive_block_count(&format!("{prefix}{{}}."))?;
    if depth == 0 {
        return Err(invalid(format!("{prefix} has no consecutive block zero")));
    }
    for key in probe
        .tensor_shapes
        .keys()
        .filter(|key| key.starts_with(prefix))
    {
        let ordinal = key[prefix.len()..]
            .split('.')
            .next()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| invalid(format!("malformed block key {key}")))?;
        if ordinal >= depth {
            return Err(invalid(format!(
                "{prefix} block ordinals are not consecutive before {ordinal}"
            )));
        }
    }
    Ok(depth)
}
