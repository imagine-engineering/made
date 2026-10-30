use crate::{
    ArtifactIndex, LatentFormatDefinition, LoadedModel, ModelStore, NativeModule, NativeOpsError,
    NativeStructuredVae, NativeVisionModelError, NativeVisionStateKind, NativeVisionStateSpec,
    VaeDescriptor, VaeError, VaeGaussianSplatBatch, VaeKernelProfile, VaeShapeField,
    VaeStructuredDecodeRequest, VaeStructuredResult,
    vae::VaeModelBinding,
    vae_image::{add_tensor, find_module, reshape_read_only, softmax_last_dimension},
    vae_video::{begin_vae_rng, contiguous_copy, narrow_contiguous, permute_read_only},
    vision_models::{
        canonical_vision_model_store_dtype, load_vision_state_from_model_store_with_context,
        load_vision_state_with_sibling_namespaces_from_model_store_with_context,
    },
};
use sim_tensor::generated_activation_normalization_functional_01::GeluApproximation;
use sim_tensor::generated_sim_operator_indirection_01::{
    tensor_from_f32_with_backend_exact_native, tensor_to_f32_with_backend_exact_native,
};
use sim_tensor::{
    BinaryOperation, CpuBackend, DType, ExecutionContext, LinearAlgebraOperation,
    ReductionOperation, ReductionSpec, RngTransaction, Scalar, ScalarSide, Tensor, TensorBackend,
    TensorDescriptor,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    f32::consts::PI,
    sync::Arc,
};
use thiserror::Error;

pub const HUNYUAN_SHAPE_ARCHITECTURE: &str = "sim.ldm.hunyuan3d.vae.ShapeVAE.v1";
pub const TRIPO_SPLAT_ARCHITECTURE: &str = "sim.ldm.triposplat.vae.OctreeGaussianDecoder.v1";
pub const TRIPO_GAUSSIANS_PER_TOKEN: u32 = 32;
pub const TRIPO_MAX_OCTREE_LEVEL: u8 = 8;
pub const TRIPO_GAUSSIAN_FEATURES_PER_TOKEN: usize = 480;

const HUNYUAN_EQUATIONS: &[&str] = &[
    "post_kl_channels_last_linear",
    "sixteen_residual_self_attention_blocks",
    "geo_final_layer_norm_eps_1e_5",
    "fourier_xyz_query_embedding",
    "chunked_cross_attention_occupancy_projection",
    "inclusive_resolution_plus_one_volume_grid",
];

const TRIPO_EQUATIONS: &[&str] = &[
    "systematic_octree_probability_resampling",
    "caller_addressed_coordinate_jitter",
    "log2_absolute_position_embedding",
    "four_cross_only_octree_blocks",
    "sixteen_self_cross_gaussian_blocks",
    "octree_and_gaussian_final_layer_norm_eps_1e_5",
    "hammersley_atanh_offset_perturbation",
    "softplus_scale_and_sigmoid_opacity_activation",
    "y_up_position_and_quaternion_transform",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredVaeStateCheckpoint {
    pub name: &'static str,
    pub shape: &'static [u64],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeStructuredVaeArchitecture {
    profile: VaeKernelProfile,
    architecture: &'static str,
    equations: &'static [&'static str],
    checkpoints: &'static [StructuredVaeStateCheckpoint],
}

impl NativeStructuredVaeArchitecture {
    pub fn profile(&self) -> &VaeKernelProfile {
        &self.profile
    }

    pub const fn architecture(&self) -> &'static str {
        self.architecture
    }

    pub const fn equation_checkpoints(&self) -> &'static [&'static str] {
        self.equations
    }

    pub const fn state_checkpoints(&self) -> &'static [StructuredVaeStateCheckpoint] {
        self.checkpoints
    }
}

const HUNYUAN_CHECKPOINTS: &[StructuredVaeStateCheckpoint] = &[
    StructuredVaeStateCheckpoint {
        name: "post_kl.weight",
        shape: &[1024, 64],
    },
    StructuredVaeStateCheckpoint {
        name: "transformer.resblocks.15.attn.c_qkv.weight",
        shape: &[3072, 1024],
    },
    StructuredVaeStateCheckpoint {
        name: "geo_decoder.query_proj.weight",
        shape: &[1024, 51],
    },
    StructuredVaeStateCheckpoint {
        name: "geo_decoder.cross_attn_decoder.ln_1.bias",
        shape: &[1024],
    },
    StructuredVaeStateCheckpoint {
        name: "geo_decoder.output_proj.weight",
        shape: &[1, 1024],
    },
];

const TRIPO_CHECKPOINTS: &[StructuredVaeStateCheckpoint] = &[
    StructuredVaeStateCheckpoint {
        name: "octree.out_proj.weight",
        shape: &[8, 1024],
    },
    StructuredVaeStateCheckpoint {
        name: "octree.blocks.3.cross_attn.to_kv.weight",
        shape: &[2048, 16],
    },
    StructuredVaeStateCheckpoint {
        name: "gs.blocks.15.cross_attn.to_kv.weight",
        shape: &[2048, 16],
    },
    StructuredVaeStateCheckpoint {
        name: "gs.out_proj.weight",
        shape: &[480, 1024],
    },
    StructuredVaeStateCheckpoint {
        name: "gs.points_offset_perturbation",
        shape: &[32, 3],
    },
    StructuredVaeStateCheckpoint {
        name: "gs.base_offset_scale",
        shape: &[],
    },
];

#[derive(Debug, Error)]
pub enum StructuredVaeError {
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(transparent)]
    Tensor(#[from] sim_tensor::TensorError),
    #[error("structured VAE profile {0:?} has no structured architecture adapter")]
    UnsupportedProfile(VaeKernelProfile),
    #[error("structured VAE tensor shape is invalid: {0}")]
    InvalidShape(String),
    #[error("structured VAE values contain a non-finite number")]
    NonFinite,
    #[error("structured VAE shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("structured VAE RNG failed: {0}")]
    Rng(String),
    #[error("structured VAE tensor operation failed: {0}")]
    TensorOperation(String),
    #[error(transparent)]
    NativeModule(#[from] NativeOpsError),
    #[error(transparent)]
    VisionState(#[from] NativeVisionModelError),
    #[error("structured VAE state is missing {0}")]
    MissingState(String),
    #[error("structured VAE state contains unexpected tensor {0}")]
    UnexpectedState(String),
    #[error("structured VAE state tensor {name} expected {expected:?}, got {actual:?}")]
    StateShape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("structured VAE state tensor {name} uses unsupported dtype {dtype}")]
    StateDType { name: String, dtype: String },
}

pub fn structured_vae_source_plan(
    profile: &VaeKernelProfile,
) -> Result<NativeStructuredVaeArchitecture, StructuredVaeError> {
    match profile {
        VaeKernelProfile::HunyuanShapeV1 => Ok(NativeStructuredVaeArchitecture {
            profile: profile.clone(),
            architecture: HUNYUAN_SHAPE_ARCHITECTURE,
            equations: HUNYUAN_EQUATIONS,
            checkpoints: HUNYUAN_CHECKPOINTS,
        }),
        VaeKernelProfile::TripoSplatV1 => Ok(NativeStructuredVaeArchitecture {
            profile: profile.clone(),
            architecture: TRIPO_SPLAT_ARCHITECTURE,
            equations: TRIPO_EQUATIONS,
            checkpoints: TRIPO_CHECKPOINTS,
        }),
        _ => Err(StructuredVaeError::UnsupportedProfile(profile.clone())),
    }
}

#[derive(Clone, Debug)]
struct StructuredStateShape {
    name: String,
    shape: Vec<u64>,
    kind: NativeVisionStateKind,
    linear: bool,
}

fn parameter(state: &mut Vec<StructuredStateShape>, name: impl Into<String>, shape: Vec<u64>) {
    state.push(StructuredStateShape {
        name: name.into(),
        shape,
        kind: NativeVisionStateKind::Parameter,
        linear: false,
    });
}

fn buffer(state: &mut Vec<StructuredStateShape>, name: impl Into<String>, shape: Vec<u64>) {
    state.push(StructuredStateShape {
        name: name.into(),
        shape,
        kind: NativeVisionStateKind::Buffer,
        linear: false,
    });
}

fn linear(
    state: &mut Vec<StructuredStateShape>,
    prefix: impl AsRef<str>,
    output: u64,
    input: u64,
    bias: bool,
) {
    let prefix = prefix.as_ref();
    state.push(StructuredStateShape {
        name: format!("{prefix}.weight"),
        shape: vec![output, input],
        kind: NativeVisionStateKind::Parameter,
        linear: true,
    });
    if bias {
        parameter(state, format!("{prefix}.bias"), vec![output]);
    }
}

fn affine_norm(state: &mut Vec<StructuredStateShape>, prefix: &str, width: u64) {
    parameter(state, format!("{prefix}.weight"), vec![width]);
    parameter(state, format!("{prefix}.bias"), vec![width]);
}

fn hunyuan_state_shapes() -> Vec<StructuredStateShape> {
    let mut state = Vec::new();
    linear(&mut state, "post_kl", 1024, 64, true);
    for block in 0..16 {
        let prefix = format!("transformer.resblocks.{block}");
        affine_norm(&mut state, &format!("{prefix}.ln_1"), 1024);
        linear(
            &mut state,
            format!("{prefix}.attn.c_qkv"),
            3072,
            1024,
            false,
        );
        affine_norm(&mut state, &format!("{prefix}.attn.attention.q_norm"), 64);
        affine_norm(&mut state, &format!("{prefix}.attn.attention.k_norm"), 64);
        linear(
            &mut state,
            format!("{prefix}.attn.c_proj"),
            1024,
            1024,
            true,
        );
        affine_norm(&mut state, &format!("{prefix}.ln_2"), 1024);
        linear(&mut state, format!("{prefix}.mlp.c_fc"), 4096, 1024, true);
        linear(&mut state, format!("{prefix}.mlp.c_proj"), 1024, 4096, true);
    }
    linear(&mut state, "geo_decoder.query_proj", 1024, 51, true);
    let prefix = "geo_decoder.cross_attn_decoder";
    affine_norm(&mut state, &format!("{prefix}.ln_1"), 1024);
    affine_norm(&mut state, &format!("{prefix}.ln_2"), 1024);
    affine_norm(&mut state, &format!("{prefix}.ln_3"), 1024);
    linear(&mut state, format!("{prefix}.attn.c_q"), 1024, 1024, false);
    linear(&mut state, format!("{prefix}.attn.c_kv"), 2048, 1024, false);
    affine_norm(&mut state, &format!("{prefix}.attn.attention.q_norm"), 64);
    affine_norm(&mut state, &format!("{prefix}.attn.attention.k_norm"), 64);
    linear(
        &mut state,
        format!("{prefix}.attn.c_proj"),
        1024,
        1024,
        true,
    );
    linear(&mut state, format!("{prefix}.mlp.c_fc"), 4096, 1024, true);
    linear(&mut state, format!("{prefix}.mlp.c_proj"), 1024, 4096, true);
    affine_norm(&mut state, "geo_decoder.ln_post", 1024);
    linear(&mut state, "geo_decoder.output_proj", 1, 1024, true);
    state
}

fn tripo_attention_state(
    state: &mut Vec<StructuredStateShape>,
    prefix: &str,
    context_width: Option<u64>,
) {
    if let Some(context_width) = context_width {
        linear(state, format!("{prefix}.to_q"), 1024, 1024, true);
        linear(state, format!("{prefix}.to_kv"), 2048, context_width, true);
    } else {
        linear(state, format!("{prefix}.to_qkv"), 3072, 1024, true);
    }
    parameter(state, format!("{prefix}.q_rms_norm.gamma"), vec![16, 64]);
    parameter(state, format!("{prefix}.k_rms_norm.gamma"), vec![16, 64]);
    linear(state, format!("{prefix}.to_out"), 1024, 1024, true);
}

fn tripo_mlp_state(state: &mut Vec<StructuredStateShape>, prefix: &str) {
    linear(state, format!("{prefix}.mlp.0"), 4096, 1024, true);
    linear(state, format!("{prefix}.mlp.2"), 1024, 4096, true);
}

fn tripo_state_shapes() -> Vec<StructuredStateShape> {
    let mut state = Vec::new();
    linear(&mut state, "octree.input_layer", 1024, 1024, true);
    linear(&mut state, "octree.l_embedder.mlp.0", 1024, 256, true);
    linear(&mut state, "octree.l_embedder.mlp.2", 1024, 1024, true);
    linear(&mut state, "octree.adaLN_modulation.1", 6144, 1024, true);
    linear(&mut state, "octree.out_proj", 8, 1024, true);
    linear(&mut state, "octree.in_proj", 1024, 3, true);
    for block in 0..4 {
        let prefix = format!("octree.blocks.{block}");
        tripo_attention_state(&mut state, &format!("{prefix}.cross_attn"), Some(16));
        tripo_mlp_state(&mut state, &format!("{prefix}.mlp"));
    }
    linear(&mut state, "gs.input_layer", 1024, 1024, true);
    linear(&mut state, "gs.in_proj", 1024, 3, true);
    linear(&mut state, "gs.out_proj", 480, 1024, true);
    buffer(&mut state, "gs.points_offset_perturbation", vec![32, 3]);
    buffer(&mut state, "gs.base_offset_scale", vec![]);
    for block in 0..16 {
        let prefix = format!("gs.blocks.{block}");
        affine_norm(&mut state, &format!("{prefix}.norm2"), 1024);
        tripo_attention_state(&mut state, &format!("{prefix}.self_attn"), None);
        tripo_attention_state(&mut state, &format!("{prefix}.cross_attn"), Some(16));
        tripo_mlp_state(&mut state, &format!("{prefix}.mlp"));
    }
    state
}

fn exact_state_shapes(
    profile: &VaeKernelProfile,
) -> Result<Vec<StructuredStateShape>, StructuredVaeError> {
    match profile {
        VaeKernelProfile::HunyuanShapeV1 => Ok(hunyuan_state_shapes()),
        VaeKernelProfile::TripoSplatV1 => Ok(tripo_state_shapes()),
        _ => Err(StructuredVaeError::UnsupportedProfile(profile.clone())),
    }
}

pub fn structured_vae_source_state_count(
    profile: &VaeKernelProfile,
) -> Result<usize, StructuredVaeError> {
    Ok(exact_state_shapes(profile)?.len())
}

pub fn structured_vae_source_state_schema(
    descriptor: &VaeDescriptor,
    model: &LoadedModel,
) -> Result<Vec<NativeVisionStateSpec>, StructuredVaeError> {
    let shapes = exact_state_shapes(descriptor.identity().profile())?;
    let mut names = BTreeSet::new();
    let mut schema = Vec::new();
    for expected in shapes {
        if !names.insert(expected.name.clone()) {
            return Err(StructuredVaeError::UnexpectedState(expected.name));
        }
        let metadata = model
            .tensors()
            .get(&expected.name)
            .ok_or_else(|| StructuredVaeError::MissingState(expected.name.clone()))?;
        if metadata.shape != expected.shape {
            return Err(StructuredVaeError::StateShape {
                name: expected.name,
                expected: expected.shape,
                actual: metadata.shape.clone(),
            });
        }
        let dtype = canonical_vision_model_store_dtype(&metadata.data_type).ok_or_else(|| {
            StructuredVaeError::StateDType {
                name: expected.name.clone(),
                dtype: metadata.data_type.clone(),
            }
        })?;
        if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
            return Err(StructuredVaeError::StateDType {
                name: expected.name,
                dtype: metadata.data_type.clone(),
            });
        }
        schema.push(NativeVisionStateSpec {
            name: expected.name,
            shape: expected.shape,
            dtype,
            kind: expected.kind,
        });
    }
    Ok(schema)
}

fn linear_prefixes(profile: &VaeKernelProfile) -> Result<BTreeSet<String>, StructuredVaeError> {
    Ok(exact_state_shapes(profile)?
        .into_iter()
        .filter(|state| state.linear)
        .filter_map(|state| state.name.strip_suffix(".weight").map(str::to_owned))
        .collect())
}

fn build_structured_module(
    profile: &VaeKernelProfile,
    mut state: BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    execution_dtype: DType,
    execution_device: sim_tensor::DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, StructuredVaeError> {
    let mut children = Vec::new();
    for prefix in linear_prefixes(profile)? {
        context.check()?;
        let weight_name = format!("{prefix}.weight");
        let weight = state
            .remove(&weight_name)
            .ok_or_else(|| StructuredVaeError::MissingState(weight_name.clone()))?;
        let shape = weight.descriptor().shape();
        let [output, input] = shape else {
            return Err(StructuredVaeError::StateShape {
                name: weight_name,
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        };
        let bias = state.remove(&format!("{prefix}.bias"));
        let mut module = NativeModule::linear(
            prefix,
            usize::try_from(*input).map_err(|_| StructuredVaeError::ShapeOverflow)?,
            usize::try_from(*output).map_err(|_| StructuredVaeError::ShapeOverflow)?,
            bias.is_some(),
            false,
        )?;
        module.load_dense_parameters(weight, bias)?;
        children.push(module);
    }
    for (name, tensor) in state {
        children.push(NativeModule::buffer(name, tensor)?);
    }
    let mut module = NativeModule::module_dict(format!("structured-vae:{profile:?}"), children)?;
    module.materialize_execution_state_with_context(
        backend,
        execution_dtype,
        execution_device,
        context,
    )?;
    Ok(module)
}

pub fn load_structured_vae_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: Arc<LoadedModel>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeStructuredVae, StructuredVaeError> {
    context.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let plan = structured_vae_source_plan(descriptor.identity().profile())?;
    if descriptor.identity().architecture().as_str() != plan.architecture() {
        return Err(VaeError::ModelArchitectureMismatch {
            expected: plan.architecture().to_owned(),
            actual: descriptor.identity().architecture().as_str().to_owned(),
        }
        .into());
    }
    let schema = structured_vae_source_state_schema(&descriptor, &model)?;
    let state = if descriptor.identity().profile() == &VaeKernelProfile::HunyuanShapeV1 {
        load_vision_state_with_sibling_namespaces_from_model_store_with_context(
            backend,
            store,
            index,
            &model,
            &schema,
            &["encoder.", "pre_kl."],
            context,
        )?
    } else {
        load_vision_state_from_model_store_with_context(
            backend, store, index, &model, &schema, context,
        )?
    };
    let module = build_structured_module(
        descriptor.identity().profile(),
        state,
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    let binding =
        VaeModelBinding::checked(&descriptor, store, model, module, context.cancellation)?;
    Ok(NativeStructuredVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        structured_decode_raw,
    )?)
}

pub(crate) fn load_structured_vae_from_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeStructuredVae, StructuredVaeError> {
    context.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let plan = structured_vae_source_plan(descriptor.identity().profile())?;
    if descriptor.identity().architecture().as_str() != plan.architecture() {
        return Err(VaeError::ModelArchitectureMismatch {
            expected: plan.architecture().to_owned(),
            actual: descriptor.identity().architecture().as_str().to_owned(),
        }
        .into());
    }
    let mut state = BTreeMap::new();
    for (name, tensor) in ordered_state {
        if state.insert(name.clone(), tensor).is_some() {
            return Err(StructuredVaeError::UnexpectedState(name));
        }
    }
    let expected = exact_state_shapes(descriptor.identity().profile())?;
    if state.len() != expected.len() {
        return Err(StructuredVaeError::UnexpectedState(
            "structured VAE state cardinality".to_owned(),
        ));
    }
    for specification in expected {
        let tensor = state
            .get(&specification.name)
            .ok_or_else(|| StructuredVaeError::MissingState(specification.name.clone()))?;
        if tensor.descriptor().shape() != specification.shape {
            return Err(StructuredVaeError::StateShape {
                name: specification.name,
                expected: specification.shape,
                actual: tensor.descriptor().shape().to_vec(),
            });
        }
        if !matches!(
            tensor.descriptor().dtype(),
            DType::F16 | DType::Bf16 | DType::F32
        ) {
            return Err(StructuredVaeError::StateDType {
                name: specification.name,
                dtype: tensor.descriptor().dtype().catalog_name().to_owned(),
            });
        }
    }
    let module = build_structured_module(
        descriptor.identity().profile(),
        state,
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
        context,
    )?;
    let binding = VaeModelBinding::checked_recipe(
        &descriptor,
        artifact_sha256,
        module,
        context.cancellation,
    )?;
    Ok(NativeStructuredVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        structured_decode_raw,
    )?)
}

#[cfg(feature = "test-support")]
pub(crate) fn load_structured_vae_from_reduced_recipe_state(
    backend: &CpuBackend,
    artifact_sha256: &str,
    ordered_state: Vec<(String, Tensor)>,
    descriptor: VaeDescriptor,
    latent_definition: &'static LatentFormatDefinition,
    context: &ExecutionContext<'_>,
) -> Result<NativeStructuredVae, StructuredVaeError> {
    context.check()?;
    crate::vae::validate_native_vae_backend_binding(
        backend,
        descriptor.identity().dtype(),
        descriptor.identity().device(),
    )?;
    let plan = structured_vae_source_plan(descriptor.identity().profile())?;
    if descriptor.identity().profile() != &VaeKernelProfile::HunyuanShapeV1 {
        return Err(StructuredVaeError::UnsupportedProfile(
            descriptor.identity().profile().clone(),
        ));
    }
    if descriptor.identity().architecture().as_str() != plan.architecture() {
        return Err(VaeError::ModelArchitectureMismatch {
            expected: plan.architecture().to_owned(),
            actual: descriptor.identity().architecture().as_str().to_owned(),
        }
        .into());
    }
    if !ordered_state
        .iter()
        .any(|(name, _)| name == "geo_decoder.cross_attn_decoder.ln_1.bias")
    {
        return Err(StructuredVaeError::MissingState(
            "geo_decoder.cross_attn_decoder.ln_1.bias".to_owned(),
        ));
    }
    let module = crate::vae::reduced_recipe_state_module(
        "structured-vae-reduced-recipe",
        ordered_state,
        context.cancellation,
    )?;
    let binding = VaeModelBinding::checked_recipe(
        &descriptor,
        artifact_sha256,
        module,
        context.cancellation,
    )?;
    Ok(NativeStructuredVae::checked_kernel(
        descriptor,
        latent_definition,
        binding,
        structured_decode_raw,
    )?)
}

fn structured_decode_raw(
    module: &NativeModule,
    backend: &CpuBackend,
    latent: &Tensor,
    request: &VaeStructuredDecodeRequest,
    context: &ExecutionContext<'_>,
) -> Result<VaeStructuredResult, VaeError> {
    let result = match request {
        VaeStructuredDecodeRequest::Shape {
            bounds,
            octree_resolution,
            chunk_size,
        } => hunyuan_decode(
            module,
            backend,
            latent,
            *bounds,
            *octree_resolution,
            *chunk_size,
            context,
        ),
        VaeStructuredDecodeRequest::GaussianSplats {
            num_gaussians,
            octree_level,
        } => tripo_decode(
            module,
            backend,
            latent,
            *num_gaussians,
            *octree_level,
            context,
        ),
    };
    result.map_err(|error| VaeError::NativeOps(NativeOpsError::InvalidOwned(error.to_string())))
}

fn run_linear(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    context.check()?;
    let mut linear = find_module(module, name)
        .cloned()
        .ok_or_else(|| StructuredVaeError::MissingState(format!("{name}.weight")))?;
    Ok(linear.forward_with_context(backend, input, context)?)
}

fn parameter_tensor<'a>(
    module: &'a NativeModule,
    name: &str,
) -> Result<&'a Tensor, StructuredVaeError> {
    find_module(module, name)
        .and_then(NativeModule::registered_buffer)
        .ok_or_else(|| StructuredVaeError::MissingState(name.to_owned()))
}

fn run_activation(
    backend: &CpuBackend,
    input: &Tensor,
    approximation: GeluApproximation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let mut gelu = NativeModule::gelu("structured.gelu", approximation)?;
    Ok(gelu.forward_with_context(backend, input, context)?)
}

fn run_silu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let mut silu = NativeModule::silu("structured.silu")?;
    Ok(silu.forward_with_context(backend, input, context)?)
}

fn run_layer_norm_last(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    affine: bool,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let width = input
        .descriptor()
        .shape()
        .last()
        .copied()
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let mut layer_norm = NativeModule::layer_norm(
        prefix,
        vec![usize::try_from(width).map_err(|_| StructuredVaeError::ShapeOverflow)?],
        epsilon,
        affine,
        affine,
        false,
    )?;
    if affine {
        layer_norm.load_dense_parameters(
            parameter_tensor(module, &format!("{prefix}.weight"))?.clone(),
            Some(parameter_tensor(module, &format!("{prefix}.bias"))?.clone()),
        )?;
        layer_norm.materialize_execution_state_with_context(
            backend,
            input.descriptor().dtype(),
            input.descriptor().device(),
            context,
        )?;
    }
    Ok(layer_norm.forward_with_context(backend, input, context)?)
}

fn scalar_operation(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: BinaryOperation,
    scalar: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let (output, event) = backend.binary_scalar(
        operation,
        input,
        Scalar::Float(scalar),
        ScalarSide::Right,
        input.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn binary_operation(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    operation: BinaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let (output, event) =
        backend.binary(operation, left, right, left.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn batch_matrix_multiply(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    shape: Vec<u64>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let descriptor = TensorDescriptor::contiguous(
        shape,
        left.descriptor().dtype(),
        left.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[left.clone(), right.clone()],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn rms_norm_heads(
    module: &NativeModule,
    backend: &dyn TensorBackend,
    input: &Tensor,
    gamma_name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[2] != 16 || shape[3] != 64 {
        return Err(StructuredVaeError::InvalidShape(
            "multi-head RMS normalization requires BxLx16x64".to_owned(),
        ));
    }
    let squared = binary_operation(backend, input, input, BinaryOperation::Multiply, context)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![shape[0], shape[1], shape[2], 1],
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (mean, event) = backend.reduction(
        &ReductionSpec {
            operation: ReductionOperation::Mean,
            dimensions: vec![3],
            keep_dimensions: true,
            accumulation_dtype: Some(input.descriptor().dtype()),
            correction: 0,
        },
        &squared,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    let mean = scalar_operation(backend, &mean, BinaryOperation::Add, 1.0e-6, context)?;
    let (inverse, event) = backend.unary(
        sim_tensor::UnaryOperation::ReciprocalSquareRoot,
        &mean,
        mean.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    let normalized =
        binary_operation(backend, input, &inverse, BinaryOperation::Multiply, context)?;
    let gamma = reshape_read_only(parameter_tensor(module, gamma_name)?, vec![1, 1, 16, 64])?;
    binary_operation(
        backend,
        &normalized,
        &gamma,
        BinaryOperation::Multiply,
        context,
    )
}

fn layer_norm_heads(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    weight_name: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 || shape[3] != 64 {
        return Err(StructuredVaeError::InvalidShape(
            "head layer normalization requires BxLxHx64".to_owned(),
        ));
    }
    let flattened = reshape_read_only(
        input,
        vec![
            shape[0]
                .checked_mul(shape[1])
                .and_then(|value| value.checked_mul(shape[2]))
                .ok_or(StructuredVaeError::ShapeOverflow)?,
            64,
        ],
    )?;
    let bias_name = weight_name
        .strip_suffix(".weight")
        .map(|prefix| format!("{prefix}.bias"))
        .ok_or_else(|| StructuredVaeError::MissingState(weight_name.to_owned()))?;
    let mut norm = NativeModule::layer_norm(
        "structured.head-layer-norm",
        vec![64],
        1.0e-6,
        true,
        true,
        false,
    )?;
    norm.load_dense_parameters(
        parameter_tensor(module, weight_name)?.clone(),
        Some(parameter_tensor(module, &bias_name)?.clone()),
    )?;
    norm.materialize_execution_state_with_context(
        backend,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?;
    let output = norm.forward_with_context(backend, &flattened, context)?;
    Ok(reshape_read_only(&output, shape.to_vec())?)
}

fn attention_from_projected(
    module: &NativeModule,
    backend: &CpuBackend,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_norm: &str,
    k_norm: &str,
    rms_norm: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let query_shape = query.descriptor().shape().to_vec();
    let key_shape = key.descriptor().shape().to_vec();
    if query_shape.len() != 3
        || key_shape.len() != 3
        || value.descriptor().shape() != key_shape
        || query_shape[0] != key_shape[0]
        || query_shape[2] != 1024
        || key_shape[2] != 1024
    {
        return Err(StructuredVaeError::InvalidShape(
            "attention projections must be BxLx1024".to_owned(),
        ));
    }
    let query = reshape_read_only(&query, vec![query_shape[0], query_shape[1], 16, 64])?;
    let key = reshape_read_only(&key, vec![key_shape[0], key_shape[1], 16, 64])?;
    let value = reshape_read_only(&value, vec![key_shape[0], key_shape[1], 16, 64])?;
    let query = if rms_norm {
        rms_norm_heads(module, backend, &query, q_norm, context)?
    } else {
        layer_norm_heads(module, backend, &query, q_norm, context)?
    };
    let key = if rms_norm {
        rms_norm_heads(module, backend, &key, k_norm, context)?
    } else {
        layer_norm_heads(module, backend, &key, k_norm, context)?
    };
    let prepare = |tensor: &Tensor, transpose: bool| -> Result<Tensor, StructuredVaeError> {
        let permutation: &[usize] = if transpose {
            &[0, 2, 3, 1]
        } else {
            &[0, 2, 1, 3]
        };
        let tensor = permute_read_only(tensor, permutation)?;
        let tensor = contiguous_copy(backend, &tensor, context)?;
        let shape = tensor.descriptor().shape();
        Ok(reshape_read_only(
            &tensor,
            vec![shape[0] * shape[1], shape[2], shape[3]],
        )?)
    };
    let query = prepare(&query, false)?;
    let key = prepare(&key, true)?;
    let value = prepare(&value, false)?;
    let batch_heads = query.descriptor().shape()[0];
    let query_tokens = query.descriptor().shape()[1];
    let key_tokens = value.descriptor().shape()[1];
    let scores = batch_matrix_multiply(
        backend,
        &query,
        &key,
        vec![batch_heads, query_tokens, key_tokens],
        context,
    )?;
    let scores = scalar_operation(
        backend,
        &scores,
        BinaryOperation::Multiply,
        64.0_f64.sqrt().recip(),
        context,
    )?;
    let scores = softmax_last_dimension(backend, &scores, context)?;
    let attended = batch_matrix_multiply(
        backend,
        &scores,
        &value,
        vec![batch_heads, query_tokens, 64],
        context,
    )?;
    let attended = reshape_read_only(&attended, vec![query_shape[0], 16, query_tokens, 64])?;
    let attended = permute_read_only(&attended, &[0, 2, 1, 3])?;
    let attended = contiguous_copy(backend, &attended, context)?;
    Ok(reshape_read_only(
        &attended,
        vec![query_shape[0], query_tokens, 1024],
    )?)
}

fn hunyuan_self_attention(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let qkv = run_linear(module, backend, input, &format!("{prefix}.c_qkv"), context)?;
    let query = narrow_contiguous(backend, &qkv, 2, 0, 1024, context)?;
    let key = narrow_contiguous(backend, &qkv, 2, 1024, 1024, context)?;
    let value = narrow_contiguous(backend, &qkv, 2, 2048, 1024, context)?;
    let attended = attention_from_projected(
        module,
        backend,
        query,
        key,
        value,
        &format!("{prefix}.attention.q_norm.weight"),
        &format!("{prefix}.attention.k_norm.weight"),
        false,
        context,
    )?;
    run_linear(
        module,
        backend,
        &attended,
        &format!("{prefix}.c_proj"),
        context,
    )
}

fn hunyuan_cross_attention(
    module: &NativeModule,
    backend: &CpuBackend,
    query: &Tensor,
    data: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let query = run_linear(module, backend, query, &format!("{prefix}.c_q"), context)?;
    let key_value = run_linear(module, backend, data, &format!("{prefix}.c_kv"), context)?;
    let key = narrow_contiguous(backend, &key_value, 2, 0, 1024, context)?;
    let value = narrow_contiguous(backend, &key_value, 2, 1024, 1024, context)?;
    let attended = attention_from_projected(
        module,
        backend,
        query,
        key,
        value,
        &format!("{prefix}.attention.q_norm.weight"),
        &format!("{prefix}.attention.k_norm.weight"),
        false,
        context,
    )?;
    run_linear(
        module,
        backend,
        &attended,
        &format!("{prefix}.c_proj"),
        context,
    )
}

fn hunyuan_mlp(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let hidden = run_linear(module, backend, input, &format!("{prefix}.c_fc"), context)?;
    let hidden = run_activation(backend, &hidden, GeluApproximation::None, context)?;
    run_linear(
        module,
        backend,
        &hidden,
        &format!("{prefix}.c_proj"),
        context,
    )
}

fn hunyuan_transformer(
    module: &NativeModule,
    backend: &CpuBackend,
    mut input: Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    for block in 0..16 {
        context.check()?;
        let prefix = format!("transformer.resblocks.{block}");
        let normalized = run_layer_norm_last(
            module,
            backend,
            &input,
            &format!("{prefix}.ln_1"),
            true,
            1.0e-6,
            context,
        )?;
        let attended = hunyuan_self_attention(
            module,
            backend,
            &normalized,
            &format!("{prefix}.attn"),
            context,
        )?;
        input = add_tensor(backend, &input, &attended, context)?;
        let normalized = run_layer_norm_last(
            module,
            backend,
            &input,
            &format!("{prefix}.ln_2"),
            true,
            1.0e-6,
            context,
        )?;
        let projected = hunyuan_mlp(
            module,
            backend,
            &normalized,
            &format!("{prefix}.mlp"),
            context,
        )?;
        input = add_tensor(backend, &input, &projected, context)?;
    }
    Ok(input)
}

fn hunyuan_fourier_queries(
    backend: &CpuBackend,
    coordinates: &[[f32; 3]],
    batch: u64,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let count = coordinates.len();
    let values_count = usize::try_from(batch)
        .map_err(|_| StructuredVaeError::ShapeOverflow)?
        .checked_mul(count)
        .and_then(|value| value.checked_mul(51))
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let mut values = backend.workspace_vec(context, values_count)?;
    for _ in 0..batch {
        for coordinate in coordinates {
            context.check()?;
            for value in coordinate {
                values.try_push(*value)?;
            }
            for value in coordinate {
                for frequency in 0..8 {
                    values.try_push((*value * (1_u32 << frequency) as f32).sin())?;
                }
            }
            for value in coordinate {
                for frequency in 0..8 {
                    values.try_push((*value * (1_u32 << frequency) as f32).cos())?;
                }
            }
        }
    }
    tensor_from_f32_with_backend_exact_native(
        backend,
        &[
            batch,
            u64::try_from(count).map_err(|_| StructuredVaeError::ShapeOverflow)?,
            51,
        ],
        &values,
        dtype,
        backend.device(),
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))
}

fn hunyuan_geo_chunk(
    module: &NativeModule,
    backend: &CpuBackend,
    coordinates: &[[f32; 3]],
    latents: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let queries = hunyuan_fourier_queries(
        backend,
        coordinates,
        latents.descriptor().shape()[0],
        latents.descriptor().dtype(),
        context,
    )?;
    let queries = run_linear(module, backend, &queries, "geo_decoder.query_proj", context)?;
    let prefix = "geo_decoder.cross_attn_decoder";
    let normalized_query = run_layer_norm_last(
        module,
        backend,
        &queries,
        &format!("{prefix}.ln_1"),
        true,
        1.0e-6,
        context,
    )?;
    let normalized_latents = run_layer_norm_last(
        module,
        backend,
        latents,
        &format!("{prefix}.ln_2"),
        true,
        1.0e-6,
        context,
    )?;
    let attended = hunyuan_cross_attention(
        module,
        backend,
        &normalized_query,
        &normalized_latents,
        &format!("{prefix}.attn"),
        context,
    )?;
    let hidden = add_tensor(backend, &queries, &attended, context)?;
    let normalized = run_layer_norm_last(
        module,
        backend,
        &hidden,
        &format!("{prefix}.ln_3"),
        true,
        1.0e-6,
        context,
    )?;
    let projected = hunyuan_mlp(
        module,
        backend,
        &normalized,
        &format!("{prefix}.mlp"),
        context,
    )?;
    let hidden = add_tensor(backend, &hidden, &projected, context)?;
    let hidden = run_layer_norm_last(
        module,
        backend,
        &hidden,
        "geo_decoder.ln_post",
        true,
        1.0e-5,
        context,
    )?;
    run_linear(module, backend, &hidden, "geo_decoder.output_proj", context)
}

fn hunyuan_decode(
    module: &NativeModule,
    backend: &CpuBackend,
    latent: &Tensor,
    bounds: [f32; 6],
    resolution: u16,
    chunk_size: u32,
    context: &ExecutionContext<'_>,
) -> Result<VaeStructuredResult, StructuredVaeError> {
    let shape = latent.descriptor().shape();
    if shape.len() != 3 || shape[1] != 64 || shape[2] == 0 {
        return Err(StructuredVaeError::InvalidShape(
            "Hunyuan ShapeVAE latent must be Bx64xL".to_owned(),
        ));
    }
    let latent = permute_read_only(latent, &[0, 2, 1])?;
    let latent = contiguous_copy(backend, &latent, context)?;
    let latent = run_linear(module, backend, &latent, "post_kl", context)?;
    let latent = hunyuan_transformer(module, backend, latent, context)?;
    let grid = shape_grid_coordinates(backend, bounds, resolution, context)?;
    let batch = usize::try_from(shape[0]).map_err(|_| StructuredVaeError::ShapeOverflow)?;
    let count = grid.len();
    let mut logits = backend.workspace_vec(
        context,
        batch
            .checked_mul(count)
            .ok_or(StructuredVaeError::ShapeOverflow)?,
    )?;
    for _ in 0..batch
        .checked_mul(count)
        .ok_or(StructuredVaeError::ShapeOverflow)?
    {
        logits.try_push(0.0)?;
    }
    let chunk_size = usize::try_from(chunk_size).map_err(|_| StructuredVaeError::ShapeOverflow)?;
    for start in (0..count).step_by(chunk_size) {
        context.check()?;
        let end = start.saturating_add(chunk_size).min(count);
        let output = hunyuan_geo_chunk(module, backend, &grid[start..end], &latent, context)?;
        let values = tensor_to_f32_with_backend_exact_native(backend, &output, context)
            .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(StructuredVaeError::NonFinite);
        }
        let chunk = end - start;
        if values.len()
            != batch
                .checked_mul(chunk)
                .ok_or(StructuredVaeError::ShapeOverflow)?
        {
            return Err(StructuredVaeError::InvalidShape(
                "Hunyuan occupancy chunk has an invalid output shape".to_owned(),
            ));
        }
        for batch_index in 0..batch {
            for local in 0..chunk {
                logits[batch_index * count + start + local] = values[batch_index * chunk + local];
            }
        }
    }
    let extent = u64::from(resolution)
        .checked_add(1)
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let logits = tensor_from_f32_with_backend_exact_native(
        backend,
        &[shape[0], extent, extent, extent],
        &logits,
        latent.descriptor().dtype(),
        latent.descriptor().device(),
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    let logits = permute_read_only(&logits, &[0, 1, 3, 2])?;
    shape_output_from_logits(logits, bounds, resolution)
}

fn tripo_position_embedding(
    backend: &CpuBackend,
    points: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let shape = points.descriptor().shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(StructuredVaeError::InvalidShape(
            "Tripo position embedding requires BxLx3".to_owned(),
        ));
    }
    let dtype = points.descriptor().dtype();
    let device = points.descriptor().device();
    let points = tensor_to_f32_with_backend_exact_native(backend, points, context)
        .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    let count = usize::try_from(shape[0])
        .map_err(|_| StructuredVaeError::ShapeOverflow)?
        .checked_mul(usize::try_from(shape[1]).map_err(|_| StructuredVaeError::ShapeOverflow)?)
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let mut values = backend.workspace_vec(
        context,
        count
            .checked_mul(1024)
            .ok_or(StructuredVaeError::ShapeOverflow)?,
    )?;
    for point in points.chunks_exact(3) {
        context.check()?;
        for coordinate in point {
            for frequency in 0..170 {
                let exponent = 10.0 * frequency as f32 / 169.0;
                values.try_push((coordinate * 2.0_f32.powf(exponent) * PI).sin())?;
            }
            for frequency in 0..170 {
                let exponent = 10.0 * frequency as f32 / 169.0;
                values.try_push((coordinate * 2.0_f32.powf(exponent) * PI).cos())?;
            }
        }
        for _ in 0..4 {
            values.try_push(0.0)?;
        }
    }
    tensor_from_f32_with_backend_exact_native(
        backend,
        &[shape[0], shape[1], 1024],
        &values,
        dtype,
        device,
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))
}

fn tripo_mlp(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let hidden = run_linear(module, backend, input, &format!("{prefix}.mlp.0"), context)?;
    let hidden = run_activation(backend, &hidden, GeluApproximation::Tanh, context)?;
    run_linear(
        module,
        backend,
        &hidden,
        &format!("{prefix}.mlp.2"),
        context,
    )
}

fn tripo_attention(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    context_input: Option<&Tensor>,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let (query, key, value) = if let Some(context_input) = context_input {
        let query = run_linear(module, backend, input, &format!("{prefix}.to_q"), context)?;
        let key_value = run_linear(
            module,
            backend,
            context_input,
            &format!("{prefix}.to_kv"),
            context,
        )?;
        (
            query,
            narrow_contiguous(backend, &key_value, 2, 0, 1024, context)?,
            narrow_contiguous(backend, &key_value, 2, 1024, 1024, context)?,
        )
    } else {
        let qkv = run_linear(module, backend, input, &format!("{prefix}.to_qkv"), context)?;
        (
            narrow_contiguous(backend, &qkv, 2, 0, 1024, context)?,
            narrow_contiguous(backend, &qkv, 2, 1024, 1024, context)?,
            narrow_contiguous(backend, &qkv, 2, 2048, 1024, context)?,
        )
    };
    let attended = attention_from_projected(
        module,
        backend,
        query,
        key,
        value,
        &format!("{prefix}.q_rms_norm.gamma"),
        &format!("{prefix}.k_rms_norm.gamma"),
        true,
        context,
    )?;
    run_linear(
        module,
        backend,
        &attended,
        &format!("{prefix}.to_out"),
        context,
    )
}

fn tripo_modulated(
    backend: &dyn TensorBackend,
    normalized: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let scale = scalar_operation(backend, scale, BinaryOperation::Add, 1.0, context)?;
    let scaled = binary_operation(
        backend,
        normalized,
        &scale,
        BinaryOperation::Multiply,
        context,
    )?;
    binary_operation(backend, &scaled, shift, BinaryOperation::Add, context)
}

fn tripo_gated_residual(
    backend: &dyn TensorBackend,
    input: &Tensor,
    update: &Tensor,
    gate: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let update = binary_operation(backend, update, gate, BinaryOperation::Multiply, context)?;
    Ok(add_tensor(backend, input, &update, context)?)
}

fn modulation_slice(
    backend: &CpuBackend,
    modulation: &Tensor,
    index: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let start = i64::try_from(
        index
            .checked_mul(1024)
            .ok_or(StructuredVaeError::ShapeOverflow)?,
    )
    .map_err(|_| StructuredVaeError::ShapeOverflow)?;
    let slice = narrow_contiguous(backend, modulation, 1, start, 1024, context)?;
    Ok(reshape_read_only(
        &slice,
        vec![modulation.descriptor().shape()[0], 1, 1024],
    )?)
}

fn tripo_octree_forward(
    module: &NativeModule,
    backend: &CpuBackend,
    points: &Tensor,
    level: u32,
    conditioning: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let embedded = tripo_position_embedding(backend, points, context)?;
    let projected = run_linear(module, backend, points, "octree.in_proj", context)?;
    let hidden = add_tensor(backend, &projected, &embedded, context)?;
    let mut hidden = run_linear(module, backend, &hidden, "octree.input_layer", context)?;
    let levels = vec![
        level;
        usize::try_from(points.descriptor().shape()[0])
            .map_err(|_| StructuredVaeError::ShapeOverflow)?
    ];
    let level_values = level_embedding(&levels, 256)?;
    let level_tensor = tensor_from_f32_with_backend_exact_native(
        backend,
        &[points.descriptor().shape()[0], 256],
        &level_values,
        points.descriptor().dtype(),
        points.descriptor().device(),
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    let level_tensor = run_linear(
        module,
        backend,
        &level_tensor,
        "octree.l_embedder.mlp.0",
        context,
    )?;
    let level_tensor = run_silu(backend, &level_tensor, context)?;
    let level_tensor = run_linear(
        module,
        backend,
        &level_tensor,
        "octree.l_embedder.mlp.2",
        context,
    )?;
    let level_tensor = run_silu(backend, &level_tensor, context)?;
    let modulation = run_linear(
        module,
        backend,
        &level_tensor,
        "octree.adaLN_modulation.1",
        context,
    )?;
    for block in 0..4 {
        context.check()?;
        let prefix = format!("octree.blocks.{block}");
        let normalized = run_layer_norm_last(
            module,
            backend,
            &hidden,
            &format!("{prefix}.norm1"),
            false,
            1.0e-6,
            context,
        )?;
        let query = tripo_modulated(
            backend,
            &normalized,
            &modulation_slice(backend, &modulation, 0, context)?,
            &modulation_slice(backend, &modulation, 1, context)?,
            context,
        )?;
        let attended = tripo_attention(
            module,
            backend,
            &query,
            Some(conditioning),
            &format!("{prefix}.cross_attn"),
            context,
        )?;
        hidden = tripo_gated_residual(
            backend,
            &hidden,
            &attended,
            &modulation_slice(backend, &modulation, 2, context)?,
            context,
        )?;
        let normalized = run_layer_norm_last(
            module,
            backend,
            &hidden,
            &format!("{prefix}.norm2"),
            false,
            1.0e-6,
            context,
        )?;
        let mlp_input = tripo_modulated(
            backend,
            &normalized,
            &modulation_slice(backend, &modulation, 3, context)?,
            &modulation_slice(backend, &modulation, 4, context)?,
            context,
        )?;
        let mlp = tripo_mlp(
            module,
            backend,
            &mlp_input,
            &format!("{prefix}.mlp"),
            context,
        )?;
        hidden = tripo_gated_residual(
            backend,
            &hidden,
            &mlp,
            &modulation_slice(backend, &modulation, 5, context)?,
            context,
        )?;
    }
    let hidden = run_layer_norm_last(
        module,
        backend,
        &hidden,
        "octree.output_norm",
        false,
        1.0e-5,
        context,
    )?;
    run_linear(module, backend, &hidden, "octree.out_proj", context)
}

fn tripo_gaussian_forward(
    module: &NativeModule,
    backend: &CpuBackend,
    points: &Tensor,
    conditioning: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let embedded = tripo_position_embedding(backend, points, context)?;
    let projected = run_linear(module, backend, points, "gs.in_proj", context)?;
    let hidden = add_tensor(backend, &projected, &embedded, context)?;
    let mut hidden = run_linear(module, backend, &hidden, "gs.input_layer", context)?;
    for block in 0..16 {
        context.check()?;
        let prefix = format!("gs.blocks.{block}");
        let normalized = run_layer_norm_last(
            module,
            backend,
            &hidden,
            &format!("{prefix}.norm1"),
            false,
            1.0e-6,
            context,
        )?;
        let attended = tripo_attention(
            module,
            backend,
            &normalized,
            None,
            &format!("{prefix}.self_attn"),
            context,
        )?;
        hidden = add_tensor(backend, &hidden, &attended, context)?;
        let normalized = run_layer_norm_last(
            module,
            backend,
            &hidden,
            &format!("{prefix}.norm2"),
            true,
            1.0e-6,
            context,
        )?;
        let attended = tripo_attention(
            module,
            backend,
            &normalized,
            Some(conditioning),
            &format!("{prefix}.cross_attn"),
            context,
        )?;
        hidden = add_tensor(backend, &hidden, &attended, context)?;
        let normalized = run_layer_norm_last(
            module,
            backend,
            &hidden,
            &format!("{prefix}.norm3"),
            false,
            1.0e-6,
            context,
        )?;
        let projected = tripo_mlp(
            module,
            backend,
            &normalized,
            &format!("{prefix}.mlp"),
            context,
        )?;
        hidden = add_tensor(backend, &hidden, &projected, context)?;
    }
    let hidden = run_layer_norm_last(
        module,
        backend,
        &hidden,
        "gs.output_norm",
        false,
        1.0e-5,
        context,
    )?;
    run_linear(module, backend, &hidden, "gs.out_proj", context)
}

#[derive(Clone, Debug)]
struct OctreeNode {
    coordinate: [u32; 3],
    count: u32,
    log_probability: f32,
}

fn child_coordinate(parent: [u32; 3], child: usize) -> Result<[u32; 3], StructuredVaeError> {
    let offsets = [
        u32::try_from(child & 1).map_err(|_| StructuredVaeError::ShapeOverflow)?,
        u32::try_from((child >> 1) & 1).map_err(|_| StructuredVaeError::ShapeOverflow)?,
        u32::try_from((child >> 2) & 1).map_err(|_| StructuredVaeError::ShapeOverflow)?,
    ];
    Ok([
        parent[0]
            .checked_mul(2)
            .and_then(|value| value.checked_add(offsets[0]))
            .ok_or(StructuredVaeError::ShapeOverflow)?,
        parent[1]
            .checked_mul(2)
            .and_then(|value| value.checked_add(offsets[1]))
            .ok_or(StructuredVaeError::ShapeOverflow)?,
        parent[2]
            .checked_mul(2)
            .and_then(|value| value.checked_add(offsets[2]))
            .ok_or(StructuredVaeError::ShapeOverflow)?,
    ])
}

fn log_softmax_rows(logits: &[f32]) -> Result<Vec<Vec<f32>>, StructuredVaeError> {
    if !logits.len().is_multiple_of(8) || logits.iter().any(|value| !value.is_finite()) {
        return Err(StructuredVaeError::NonFinite);
    }
    logits
        .chunks_exact(8)
        .map(|row| {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = row
                .iter()
                .map(|value| (*value - maximum).exp())
                .sum::<f32>();
            let normalization = maximum + sum.ln();
            Ok(row.iter().map(|value| *value - normalization).collect())
        })
        .collect()
}

fn tripo_sample_octree(
    module: &NativeModule,
    backend: &CpuBackend,
    conditioning: &Tensor,
    num_points: u32,
    level: u8,
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, StructuredVaeError> {
    let batch = usize::try_from(conditioning.descriptor().shape()[0])
        .map_err(|_| StructuredVaeError::ShapeOverflow)?;
    let mut nodes = vec![
        vec![OctreeNode {
            coordinate: [0, 0, 0],
            count: num_points,
            log_probability: 0.0,
        }];
        batch
    ];
    for current_level in 1..=level {
        context.check()?;
        let width = nodes.iter().map(Vec::len).max().unwrap_or(0);
        if width == 0 {
            return Err(StructuredVaeError::InvalidShape(
                "octree sampling lost every active node".to_owned(),
            ));
        }
        let parent_resolution = 1_u32
            .checked_shl(u32::from(current_level - 1))
            .ok_or(StructuredVaeError::ShapeOverflow)?;
        let resolution = 1_u32
            .checked_shl(u32::from(current_level))
            .ok_or(StructuredVaeError::ShapeOverflow)?;
        let mut point_values = backend.workspace_vec(
            context,
            batch
                .checked_mul(width)
                .and_then(|value| value.checked_mul(3))
                .ok_or(StructuredVaeError::ShapeOverflow)?,
        )?;
        let mut counts = Vec::with_capacity(batch * width);
        for batch_nodes in &nodes {
            for index in 0..width {
                let node = batch_nodes.get(index);
                for axis in 0..3 {
                    let coordinate = node.map_or(0, |node| node.coordinate[axis]);
                    point_values.try_push((coordinate as f32 + 0.5) / parent_resolution as f32)?;
                }
                counts.push(node.map_or(0, |node| node.count));
            }
        }
        let points = tensor_from_f32_with_backend_exact_native(
            backend,
            &[
                u64::try_from(batch).map_err(|_| StructuredVaeError::ShapeOverflow)?,
                u64::try_from(width).map_err(|_| StructuredVaeError::ShapeOverflow)?,
                3,
            ],
            &point_values,
            conditioning.descriptor().dtype(),
            conditioning.descriptor().device(),
            context,
        )
        .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
        let logits =
            tripo_octree_forward(module, backend, &points, resolution, conditioning, context)?;
        let logits = tensor_to_f32_with_backend_exact_native(backend, &logits, context)
            .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
        let log_probabilities = log_softmax_rows(&logits)?;
        let probabilities = log_probabilities
            .iter()
            .map(|row| row.iter().map(|value| value.exp()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let sampled = systematic_sample_counts(&probabilities, &counts, transaction, context)?;
        let mut next = Vec::with_capacity(batch);
        for batch_index in 0..batch {
            let mut next_batch = Vec::new();
            for parent_index in 0..width {
                let Some(parent) = nodes[batch_index].get(parent_index) else {
                    continue;
                };
                let row = batch_index * width + parent_index;
                for child in 0..8 {
                    let count = sampled[row][child];
                    if count == 0 {
                        continue;
                    }
                    next_batch.push(OctreeNode {
                        coordinate: child_coordinate(parent.coordinate, child)?,
                        count,
                        log_probability: parent.log_probability + log_probabilities[row][child],
                    });
                }
            }
            next.push(next_batch);
        }
        nodes = next;
    }
    let resolution = 1_u32
        .checked_shl(u32::from(level))
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let mut values = backend.workspace_vec(
        context,
        batch
            .checked_mul(
                usize::try_from(num_points).map_err(|_| StructuredVaeError::ShapeOverflow)?,
            )
            .and_then(|value| value.checked_mul(3))
            .ok_or(StructuredVaeError::ShapeOverflow)?,
    )?;
    for batch_nodes in nodes {
        let mut produced = 0_u32;
        for node in batch_nodes {
            if !node.log_probability.is_finite() {
                return Err(StructuredVaeError::NonFinite);
            }
            for _ in 0..node.count {
                context.check()?;
                for axis in 0..3 {
                    let jitter = transaction
                        .next_unit_f32(context.cancellation)
                        .map_err(|error| StructuredVaeError::Rng(error.to_string()))?;
                    values.try_push((node.coordinate[axis] as f32 + jitter) / resolution as f32)?;
                }
                produced = produced
                    .checked_add(1)
                    .ok_or(StructuredVaeError::ShapeOverflow)?;
            }
        }
        if produced != num_points {
            return Err(StructuredVaeError::InvalidShape(format!(
                "octree produced {produced} points instead of {num_points}"
            )));
        }
    }
    tensor_from_f32_with_backend_exact_native(
        backend,
        &[
            u64::try_from(batch).map_err(|_| StructuredVaeError::ShapeOverflow)?,
            u64::from(num_points),
            3,
        ],
        &values,
        conditioning.descriptor().dtype(),
        conditioning.descriptor().device(),
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))
}

fn tripo_decode(
    module: &NativeModule,
    backend: &CpuBackend,
    latent: &Tensor,
    num_gaussians: u32,
    level: u8,
    context: &ExecutionContext<'_>,
) -> Result<VaeStructuredResult, StructuredVaeError> {
    let shape = latent.descriptor().shape();
    if shape.len() != 3 || shape[2] != 16 || shape[1] == 0 {
        return Err(StructuredVaeError::InvalidShape(
            "TripoSplat latent must be BxLx16".to_owned(),
        ));
    }
    if num_gaussians == 0 || !(1..=TRIPO_MAX_OCTREE_LEVEL).contains(&level) {
        return Err(VaeError::InvalidStructuredRequest(
            "TripoSplat requires a positive Gaussian count and octree level in 1..=8".to_owned(),
        )
        .into());
    }
    let num_tokens = (num_gaussians / TRIPO_GAUSSIANS_PER_TOKEN).max(1);
    let mut transaction = begin_vae_rng(context)?;
    let points = tripo_sample_octree(
        module,
        backend,
        latent,
        num_tokens,
        level,
        &mut transaction,
        context,
    )?;
    let features = tripo_gaussian_forward(module, backend, &points, latent, context)?;
    let perturbation_values = tensor_to_f32_with_backend_exact_native(
        backend,
        parameter_tensor(module, "gs.points_offset_perturbation")?,
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    let perturbations = perturbation_values
        .chunks_exact(3)
        .map(|values| [values[0], values[1], values[2]])
        .collect::<Vec<_>>();
    if perturbations.len()
        != usize::try_from(TRIPO_GAUSSIANS_PER_TOKEN)
            .map_err(|_| StructuredVaeError::ShapeOverflow)?
        || perturbation_values.len() % 3 != 0
        || perturbations
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(StructuredVaeError::InvalidShape(
            "TripoSplat point-offset perturbation buffer must be finite 32x3".to_owned(),
        ));
    }
    let base_offset_scale_values = tensor_to_f32_with_backend_exact_native(
        backend,
        parameter_tensor(module, "gs.base_offset_scale")?,
        context,
    )
    .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    let [base_offset_scale] = base_offset_scale_values.as_slice() else {
        return Err(StructuredVaeError::InvalidShape(
            "TripoSplat base-offset scale buffer must be scalar".to_owned(),
        ));
    };
    if !base_offset_scale.is_finite() {
        return Err(StructuredVaeError::NonFinite);
    }
    tripo_gaussian_output_with_constants(
        backend,
        &points,
        &features,
        &perturbations,
        *base_offset_scale,
        context,
    )
}

pub fn shape_grid_coordinates(
    backend: &CpuBackend,
    bounds: [f32; 6],
    resolution: u16,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::CpuWorkspaceVec<[f32; 3]>, StructuredVaeError> {
    if bounds.iter().any(|value| !value.is_finite())
        || bounds[0] >= bounds[3]
        || bounds[1] >= bounds[4]
        || bounds[2] >= bounds[5]
        || resolution == 0
    {
        return Err(VaeError::InvalidStructuredRequest(
            "shape grid requires finite ascending bounds and a positive resolution".to_owned(),
        )
        .into());
    }
    let extent = usize::from(resolution)
        .checked_add(1)
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let count = extent
        .checked_mul(extent)
        .and_then(|value| value.checked_mul(extent))
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let mut coordinates = backend.workspace_vec(context, count)?;
    let denominator = f32::from(resolution);
    for x in 0..extent {
        context.check()?;
        let x = bounds[0] + (bounds[3] - bounds[0]) * x as f32 / denominator;
        for y in 0..extent {
            let y = bounds[1] + (bounds[4] - bounds[1]) * y as f32 / denominator;
            for z in 0..extent {
                let z = bounds[2] + (bounds[5] - bounds[2]) * z as f32 / denominator;
                coordinates.try_push([x, y, z])?;
            }
        }
    }
    Ok(coordinates)
}

pub fn shape_output_from_logits(
    logits: Tensor,
    bounds: [f32; 6],
    resolution: u16,
) -> Result<VaeStructuredResult, StructuredVaeError> {
    Ok(VaeStructuredResult::Shape(VaeShapeField::checked(
        logits, bounds, resolution,
    )?))
}

pub fn radical_inverse(base: u32, mut index: u32) -> f32 {
    let inverse_base = 1.0 / base as f32;
    let mut inverse_power = inverse_base;
    let mut value = 0.0;
    while index > 0 {
        value += (index % base) as f32 * inverse_power;
        index /= base;
        inverse_power *= inverse_base;
    }
    value
}

pub fn hammersley_3d(index: u32, count: u32) -> Result<[f32; 3], StructuredVaeError> {
    if count == 0 || index >= count {
        return Err(VaeError::InvalidStructuredRequest(
            "Hammersley index must be smaller than a positive sample count".to_owned(),
        )
        .into());
    }
    Ok([
        index as f32 / count as f32,
        radical_inverse(2, index),
        radical_inverse(3, index),
    ])
}

pub fn systematic_sample_counts(
    probabilities: &[Vec<f32>],
    counts: &[u32],
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Vec<u32>>, StructuredVaeError> {
    if probabilities.len() != counts.len() || probabilities.is_empty() {
        return Err(StructuredVaeError::InvalidShape(
            "probability rows and counts must be non-empty and aligned".to_owned(),
        ));
    }
    let bins = probabilities[0].len();
    if bins == 0 || probabilities.iter().any(|row| row.len() != bins) {
        return Err(StructuredVaeError::InvalidShape(
            "probability rows must have one common positive width".to_owned(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(probabilities.len())
        .map_err(|_| StructuredVaeError::ShapeOverflow)?;
    let draws_offsets = counts.iter().any(|count| *count > 0);
    for (row, count) in probabilities.iter().zip(counts) {
        context.check()?;
        if row.iter().any(|value| !value.is_finite()) {
            return Err(StructuredVaeError::NonFinite);
        }
        let mut normalized = row.iter().map(|value| value.max(0.0)).collect::<Vec<_>>();
        let sum = normalized.iter().sum::<f32>();
        if sum == 0.0 {
            normalized.fill(1.0 / bins as f32);
        } else {
            for value in &mut normalized {
                *value /= sum.max(1.0);
            }
        }
        let mut cumulative = Vec::new();
        cumulative
            .try_reserve_exact(bins)
            .map_err(|_| StructuredVaeError::ShapeOverflow)?;
        let mut running = 0.0_f32;
        for value in normalized {
            running = (running + value).min(1.0 - 1.0e-12);
            cumulative.push(running);
        }
        let mut sampled = vec![0_u32; bins];
        let offset = if draws_offsets {
            Some(
                transaction
                    .next_unit_f32(context.cancellation)
                    .map_err(|error| StructuredVaeError::Rng(error.to_string()))?,
            )
        } else {
            None
        };
        if let Some(offset) = offset.filter(|_| *count > 0) {
            for sample in 0..*count {
                context.check()?;
                let threshold = ((offset + sample as f32) / *count as f32).min(1.0 - 1.0e-12);
                let bin = cumulative.partition_point(|value| *value < threshold);
                let bin = bin.min(bins - 1);
                sampled[bin] = sampled[bin]
                    .checked_add(1)
                    .ok_or(StructuredVaeError::ShapeOverflow)?;
            }
        }
        output.push(sampled);
    }
    Ok(output)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn normalize_quaternion(mut quaternion: [f32; 4]) -> [f32; 4] {
    let norm = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    for value in &mut quaternion {
        *value /= norm;
    }
    quaternion
}

fn transform_position(position: [f32; 3]) -> [f32; 3] {
    [position[0], -position[2], position[1]]
}

fn transform_quaternion(quaternion: [f32; 4]) -> [f32; 4] {
    let [w, x, y, z] = normalize_quaternion(quaternion);
    let matrix = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let transformed = [
        matrix[0],
        [-matrix[2][0], -matrix[2][1], -matrix[2][2]],
        matrix[1],
    ];
    let trace = transformed[0][0] + transformed[1][1] + transformed[2][2];
    if trace > -1.0 + 1.0e-6 {
        let scale = (trace + 1.0).max(0.0).sqrt() * 2.0;
        if scale > f32::MIN_POSITIVE {
            return normalize_quaternion([
                0.25 * scale,
                (transformed[2][1] - transformed[1][2]) / scale,
                (transformed[0][2] - transformed[2][0]) / scale,
                (transformed[1][0] - transformed[0][1]) / scale,
            ]);
        }
    }
    let (axis, next, last) =
        if transformed[0][0] >= transformed[1][1] && transformed[0][0] >= transformed[2][2] {
            (0, 1, 2)
        } else if transformed[1][1] >= transformed[2][2] {
            (1, 2, 0)
        } else {
            (2, 0, 1)
        };
    let scale = (1.0 + transformed[axis][axis] - transformed[next][next] - transformed[last][last])
        .max(0.0)
        .sqrt()
        * 2.0;
    if scale <= f32::MIN_POSITIVE {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let mut result = [0.0; 4];
    result[0] = (transformed[last][next] - transformed[next][last]) / scale;
    result[axis + 1] = 0.25 * scale;
    result[next + 1] = (transformed[axis][next] + transformed[next][axis]) / scale;
    result[last + 1] = (transformed[axis][last] + transformed[last][axis]) / scale;
    normalize_quaternion(result)
}

pub fn tripo_gaussian_output_from_predictions(
    backend: &CpuBackend,
    points: &Tensor,
    features: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<VaeStructuredResult, StructuredVaeError> {
    let perturbations = (0..TRIPO_GAUSSIANS_PER_TOKEN)
        .map(|index| {
            hammersley_3d(index, TRIPO_GAUSSIANS_PER_TOKEN)
                .map(|values| values.map(|value| ((value * 2.0 - 1.0) / 1.5).atanh()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tripo_gaussian_output_with_constants(
        backend,
        points,
        features,
        &perturbations,
        (0.05_f32.exp() - 1.0).ln(),
        context,
    )
}

fn tripo_gaussian_output_with_constants(
    backend: &CpuBackend,
    points: &Tensor,
    features: &Tensor,
    perturbations: &[[f32; 3]],
    base_offset_scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<VaeStructuredResult, StructuredVaeError> {
    context.check()?;
    let point_shape = points.descriptor().shape();
    let feature_shape = features.descriptor().shape();
    if point_shape.len() != 3
        || point_shape[2] != 3
        || feature_shape != [point_shape[0], point_shape[1], 480]
        || point_shape[0] == 0
        || point_shape[1] == 0
        || points.descriptor().dtype() != features.descriptor().dtype()
        || points.descriptor().device() != features.descriptor().device()
        || points.descriptor().stream() != features.descriptor().stream()
    {
        return Err(StructuredVaeError::InvalidShape(
            "Tripo predictions must be aligned BxTx3 and BxTx480 tensors".to_owned(),
        ));
    }
    if perturbations.len()
        != usize::try_from(TRIPO_GAUSSIANS_PER_TOKEN)
            .map_err(|_| StructuredVaeError::ShapeOverflow)?
        || perturbations
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || !base_offset_scale.is_finite()
    {
        return Err(StructuredVaeError::InvalidShape(
            "Tripo Gaussian constants must be finite 32x3 perturbations and one scale".to_owned(),
        ));
    }
    let point_values = tensor_to_f32_with_backend_exact_native(backend, points, context)
        .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    let feature_values = tensor_to_f32_with_backend_exact_native(backend, features, context)
        .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))?;
    if point_values
        .iter()
        .chain(&feature_values)
        .any(|value| !value.is_finite())
    {
        return Err(StructuredVaeError::NonFinite);
    }
    let tokens = usize::try_from(point_shape[1]).map_err(|_| StructuredVaeError::ShapeOverflow)?;
    let gaussians = tokens
        .checked_mul(
            usize::try_from(TRIPO_GAUSSIANS_PER_TOKEN)
                .map_err(|_| StructuredVaeError::ShapeOverflow)?,
        )
        .ok_or(StructuredVaeError::ShapeOverflow)?;
    let batches = usize::try_from(point_shape[0]).map_err(|_| StructuredVaeError::ShapeOverflow)?;
    let scale_bias = 0.004_f32 + (-(-0.004_f32).exp_m1()).ln();
    let opacity_bias = (0.1_f32 / 0.9).ln();
    let mut results = Vec::new();
    results
        .try_reserve_exact(batches)
        .map_err(|_| StructuredVaeError::ShapeOverflow)?;
    for batch in 0..batches {
        context.check()?;
        let mut positions = backend.workspace_vec(
            context,
            gaussians
                .checked_mul(3)
                .ok_or(StructuredVaeError::ShapeOverflow)?,
        )?;
        let mut colors = backend.workspace_vec(
            context,
            gaussians
                .checked_mul(3)
                .ok_or(StructuredVaeError::ShapeOverflow)?,
        )?;
        let mut scales = backend.workspace_vec(
            context,
            gaussians
                .checked_mul(3)
                .ok_or(StructuredVaeError::ShapeOverflow)?,
        )?;
        let mut rotations = backend.workspace_vec(
            context,
            gaussians
                .checked_mul(4)
                .ok_or(StructuredVaeError::ShapeOverflow)?,
        )?;
        let mut opacities = backend.workspace_vec(context, gaussians)?;
        for token in 0..tokens {
            context.check()?;
            let point_offset = (batch * tokens + token) * 3;
            let center = [
                point_values[point_offset],
                point_values[point_offset + 1],
                point_values[point_offset + 2],
            ];
            let feature_offset = (batch * tokens + token) * TRIPO_GAUSSIAN_FEATURES_PER_TOKEN;
            for gaussian in 0..usize::try_from(TRIPO_GAUSSIANS_PER_TOKEN)
                .map_err(|_| StructuredVaeError::ShapeOverflow)?
            {
                let perturbation = perturbations
                    .get(gaussian)
                    .ok_or(StructuredVaeError::ShapeOverflow)?;
                let raw_offset_scale = feature_values[feature_offset + 448 + gaussian];
                let offset_scale = softplus(raw_offset_scale + base_offset_scale);
                let mut offset = [0.0; 3];
                for axis in 0..3 {
                    let raw = feature_values[feature_offset + gaussian * 3 + axis];
                    offset[axis] = ((raw + perturbation[axis]).tanh() * 0.75) * offset_scale;
                }
                let position = transform_position([
                    (center[0] + offset[0]) + -0.5,
                    (center[1] + offset[1]) + -0.5,
                    (center[2] + offset[2]) + -0.5,
                ]);
                for value in position {
                    positions.try_push(value)?;
                }
                let color = 96 + gaussian * 3;
                for value in &feature_values[feature_offset + color..feature_offset + color + 3] {
                    colors.try_push(*value)?;
                }
                let scaling = 192 + gaussian * 3;
                for axis in 0..3 {
                    let activated =
                        softplus(feature_values[feature_offset + scaling + axis] + scale_bias);
                    scales.try_push((activated * activated + 0.0009_f32.powi(2)).sqrt())?;
                }
                let rotation = 288 + gaussian * 4;
                for value in transform_quaternion([
                    feature_values[feature_offset + rotation] * 0.1 + 1.0,
                    feature_values[feature_offset + rotation + 1] * 0.1,
                    feature_values[feature_offset + rotation + 2] * 0.1,
                    feature_values[feature_offset + rotation + 3] * 0.1,
                ]) {
                    rotations.try_push(value)?;
                }
                opacities.try_push(sigmoid(
                    feature_values[feature_offset + 416 + gaussian] + opacity_bias,
                ))?;
            }
        }
        let upload = |shape: &[u64], values: &[f32]| {
            tensor_from_f32_with_backend_exact_native(
                backend,
                shape,
                values,
                points.descriptor().dtype(),
                points.descriptor().device(),
                context,
            )
            .map_err(|error| StructuredVaeError::TensorOperation(error.to_string()))
        };
        if positions
            .iter()
            .chain(colors.iter())
            .chain(scales.iter())
            .chain(rotations.iter())
            .chain(opacities.iter())
            .any(|value| !value.is_finite())
        {
            return Err(StructuredVaeError::NonFinite);
        }
        let count = u64::try_from(gaussians).map_err(|_| StructuredVaeError::ShapeOverflow)?;
        results.push(VaeGaussianSplatBatch::checked(
            upload(&[count, 3], &positions)?,
            upload(&[count, 1, 3], &colors)?,
            upload(&[count, 3], &scales)?,
            upload(&[count, 4], &rotations)?,
            upload(&[count, 1], &opacities)?,
        )?);
    }
    Ok(VaeStructuredResult::GaussianSplats(results))
}

pub fn level_embedding(levels: &[u32], dimension: usize) -> Result<Vec<f32>, StructuredVaeError> {
    if dimension == 0 {
        return Err(StructuredVaeError::InvalidShape(
            "level embedding dimension must be positive".to_owned(),
        ));
    }
    let half = dimension / 2;
    let mut output = Vec::new();
    output
        .try_reserve_exact(
            levels
                .len()
                .checked_mul(dimension)
                .ok_or(StructuredVaeError::ShapeOverflow)?,
        )
        .map_err(|_| StructuredVaeError::ShapeOverflow)?;
    for level in levels {
        for index in 0..half {
            let frequency = (-1024.0_f32.ln() * index as f32 / half.max(1) as f32).exp();
            output.push((*level as f32 * frequency * 2.0 * PI).cos());
        }
        for index in 0..half {
            let frequency = (-1024.0_f32.ln() * index as f32 / half.max(1) as f32).exp();
            output.push((*level as f32 * frequency * 2.0 * PI).sin());
        }
        if dimension % 2 == 1 {
            output.push(0.0);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tripo_checkpoint_constants_have_one_buffer_owner() -> Result<(), StructuredVaeError> {
        let state = tripo_state_shapes();
        let names = state
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), state.len());
        let buffers = state
            .iter()
            .filter(|entry| entry.kind == NativeVisionStateKind::Buffer)
            .map(|entry| (entry.name.as_str(), entry.shape.as_slice()))
            .collect::<Vec<_>>();
        assert_eq!(
            buffers,
            vec![
                ("gs.points_offset_perturbation", &[32, 3][..]),
                ("gs.base_offset_scale", &[][..]),
            ]
        );
        Ok(())
    }
}
