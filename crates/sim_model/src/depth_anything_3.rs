use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, ImageTensor, Layout,
    ResizeCrop, ResizeMode, RngAlgorithm, RngProfileVersion, StorageId, StreamId, Tensor,
    TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, GeluApproximation, gelu_with_context_exact_native,
        layer_norm_with_context_exact_native, relu_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, ElementwiseRuntimePartThreeError,
    },
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError, div_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_14::{
        ElementwiseRuntimePartFourteenError, argsort_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, add_method_with_context_exact_native,
        mul_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_17::{
        ElementwiseRuntimePartSeventeenError, clip_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_18::{
        ElementwiseRuntimePartEighteenError, sub_method_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_21::{
        ElementwiseRuntimePartTwentyOneError, exp_with_context_exact_native,
    },
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_linear_algebra_01::{
        LinearAlgebraPartOneError, QrMode, determinant_with_context_exact_native,
        qr_with_context_exact_native,
    },
    generated_linear_algebra_02::{
        LinearAlgebraPartTwoError, bmm_with_context_exact_native, svd_with_context_exact_native,
    },
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError, linear_with_context_exact_native,
    },
    generated_random_number_generation_01::{
        RandomNumberGenerationPartOneError, generator_exact_native,
        randperm_with_context_exact_native,
    },
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, torch_cat_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        ShapeLayoutTransformPartThreeError, tensor_permute_exact_native,
        tensor_squeeze_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, InterpolateConfiguration, InterpolateMode,
        SpatialFunctionalKernelError, conv_2d_tensor_with_context_exact_native,
        conv_transpose_2d_tensor_with_context_exact_native,
        interpolate_tensor_with_context_exact_native,
    },
    generated_tensor_creation_01::{
        TensorCreationPartOneError, linspace_with_context_exact_native,
    },
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::{
    ModelProbe,
    attention::{
        AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionRequest,
        scaled_dot_product_attention_with_context,
    },
    dino2::{
        NativeDino2Backbone, NativeDino2Configuration, NativeDino2Error,
        NativeDino2ReferenceStrategy,
    },
    generated_depthanything3_sim_model_0075::{
        DepthAnything3Backbone, DepthAnything3Configuration, DepthAnything3Head,
        configuration_for_probe,
    },
};

pub const NODES_DEPTH_ANYTHING_3_SOURCE_SHA256: &str =
    "adfce28637b6904a08596aa23e22502d20089bc28fff6bcdaabe0b3c35fb7f02";
pub const DEPTH_ANYTHING_3_MODEL_SOURCE_SHA256: &str =
    "6f05ba0c22a34304f6bd6cde7e6dd26ceef474a99ad51d6632940f8d2decf6b0";
pub const DEPTH_ANYTHING_3_PREPROCESS_SOURCE_SHA256: &str =
    "6bf00e9929451c39763a0661aa1430dbc78917bc028517a4c8dc290897601845";
pub const DEPTH_ANYTHING_3_DPT_SOURCE_SHA256: &str =
    "756fa18408e161cb2ddf8adde82902d9fe3aa555be8252b60d045cbc76513ee5";
pub const DEPTH_ANYTHING_3_CAMERA_SOURCE_SHA256: &str =
    "b9c1bc79862c8f2b59a6058da1bf47c1aaef84ca75ec5131805fbdf2f81dca9a";
pub const DEPTH_ANYTHING_3_RAY_POSE_SOURCE_SHA256: &str =
    "a5ed28c0acc2daaeea57754dec4020fe91af60fd2af6548dfa68134713b36694";
pub const DEPTH_ANYTHING_3_REFERENCE_VIEW_SOURCE_SHA256: &str =
    "24e9428a820b5287d622bc865d4fd6520486294c4337a28de71fca6ec62e0c29";
pub const DEPTH_ANYTHING_3_TRANSFORM_SOURCE_SHA256: &str =
    "30291a7f8d3d83cc6a911daf603342444ed30e7141abb3ba8ccc7f41273ac763";
pub const DEPTH_ANYTHING_3_DINO2_SOURCE_SHA256: &str =
    "1dec8c1d6104c268e593cea20302d925f637266edce2a6e4dfa142af8a00d579";
pub const DEPTH_ANYTHING_3_MODEL_DETECTION_SOURCE_SHA256: &str =
    "f13b11988fccf9fa4d878ef5f63313c23c5f1400ec8cde04a502584e157c5072";

const MAX_STATE_TENSORS: usize = 16_384;
const MAX_STATE_KEY_BYTES: usize = 1_024;

#[derive(Clone, Debug)]
pub struct NativeDepthAnything3Checkpoint {
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub memory_budget_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDepthAnything3ResizeMethod {
    UpperBound,
    LowerBound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDepthAnything3ReferenceStrategy {
    First,
    Middle,
    SaddleBalanced,
    SaddleSimRange,
}

#[derive(Clone, Debug)]
pub struct NativeDepthAnything3Invocation<'a> {
    pub image: &'a ImageTensor,
    pub views_per_sample: u64,
    pub process_resolution: u64,
    pub resize_method: NativeDepthAnything3ResizeMethod,
    pub reference_strategy: NativeDepthAnything3ReferenceStrategy,
    pub use_ray_pose: bool,
    pub ransac_seed: u64,
    pub extrinsics: Option<&'a Tensor>,
    pub intrinsics: Option<&'a Tensor>,
}

#[derive(Clone, Debug)]
pub struct NativeDepthAnything3Geometry {
    pub depth: Tensor,
    pub confidence: Option<Tensor>,
    pub sky: Option<Tensor>,
    pub extrinsics: Option<Tensor>,
    pub intrinsics: Option<Tensor>,
    pub original_height: u64,
    pub original_width: u64,
    pub views_per_sample: u64,
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct NativeDepthAnything3TestTrace {
    pub raw_ray: Tensor,
    pub raw_ray_confidence: Tensor,
    pub ransac_samples: Vec<Vec<u64>>,
    pub ransac_views: Vec<NativeDepthAnything3RansacViewTrace>,
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct NativeDepthAnything3RansacViewTrace {
    pub candidate_indices: Vec<u64>,
    pub best_iteration: u64,
    pub best_inliers: Vec<u64>,
    pub best_score_bits: u32,
    pub fallback: bool,
    pub normalized_homography: [f32; 9],
    pub signed_homography: [f32; 9],
    pub rotation: [f32; 9],
    pub lower: [f32; 9],
    pub c2w_pre_inverse: [f32; 12],
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthAnything3FixtureProfile {
    Dpt,
    DualDpt,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug)]
pub struct DepthAnything3FixtureMutation<'a> {
    pub state_key: &'a str,
    pub lane: usize,
    pub delta: f32,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthAnything3FixtureStateParity {
    pub key: String,
    pub shape: Vec<u64>,
    pub source_sha256: String,
    pub projected_f32_sha256: String,
}

#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DepthAnything3FixtureCheckpointParity {
    pub source_sha256: String,
    pub projected_f32_sha256: String,
    pub states: Vec<DepthAnything3FixtureStateParity>,
}

#[cfg(any(test, feature = "test-support"))]
pub fn deterministic_reduced_depth_anything_3_checkpoint(
    backend: &CpuBackend,
    profile: DepthAnything3FixtureProfile,
    source_dtype: DType,
    memory_budget_bytes: u64,
    context: &ExecutionContext<'_>,
) -> Result<NativeDepthAnything3Checkpoint, NativeDepthAnything3Error> {
    if !matches!(source_dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(NativeDepthAnything3Error::InvalidCheckpoint(
            "reduced fixture dtype must be F16, BF16, or F32".to_owned(),
        ));
    }
    let mut markers = BTreeMap::new();
    if profile == DepthAnything3FixtureProfile::DualDpt {
        let marker = tensor_from_f32_with_context_exact_native(
            backend,
            &[1],
            &[0.0],
            source_dtype,
            DeviceId::CPU,
            context,
        )?;
        markers.insert(
            "native.head.scratch.refinenet1_aux.out_conv.weight".to_owned(),
            marker.clone(),
        );
        markers.insert(
            "native.backbone.embeddings.camera_token".to_owned(),
            marker.clone(),
        );
        markers.insert(
            "native.cam_enc.pose_branch.fc1.weight".to_owned(),
            marker.clone(),
        );
        markers.insert("native.cam_dec.fc_t.weight".to_owned(), marker);
    }
    let execution_profile = DepthAnything3ExecutionProfile::reduced(&markers);
    let specifications = depth_anything_3_state_manifest(execution_profile)?;
    let mut ordered_state = Vec::new();
    ordered_state
        .try_reserve_exact(specifications.len())
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for (state_index, specification) in specifications.into_iter().enumerate() {
        context.check()?;
        let elements = specification
            .shape
            .iter()
            .try_fold(1_usize, |total, dimension| {
                total
                    .checked_mul(usize_from(*dimension)?)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)
            })?;
        let mut values = filled_f32(elements, 0.0)?;
        let is_norm_weight = specification.key.ends_with(".weight")
            && (specification.key.contains(".norm") || specification.key.contains("layernorm"));
        let is_layer_scale =
            specification.key.ends_with(".lambda1") || specification.key.ends_with(".gamma");
        for (value_index, value) in values.iter_mut().enumerate() {
            if value_index.is_multiple_of(16_384) {
                context.check()?;
            }
            *value = if is_norm_weight {
                1.0
            } else if is_layer_scale {
                0.125
            } else if specification.key.ends_with(".bias") {
                0.0
            } else {
                let lane = ((state_index * 17 + value_index * 13) % 29) as f32 - 14.0;
                lane * 0.0025
            };
        }
        if specification.key == "native.cam_dec.fc_qvec.bias" && values.len() == 4 {
            values[3] = 1.0;
        }
        if specification.key == "native.cam_dec.fc_fov.0.bias" && values.len() == 2 {
            values[0] = 0.75;
            values[1] = 1.0;
        }
        if specification.key == "native.head.scratch.output_conv2_aux.3.5.weight" {
            let value = values
                .get_mut(194)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
            *value += 1.0;
        }
        if specification.key == "native.head.scratch.output_conv2_aux.3.5.bias" {
            *values
                .get_mut(2)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)? = 1.0e-4;
        }
        ordered_state.push((
            specification.key,
            tensor_from_f32_with_context_exact_native(
                backend,
                &specification.shape,
                &values,
                source_dtype,
                DeviceId::CPU,
                context,
            )?,
        ));
    }
    Ok(NativeDepthAnything3Checkpoint {
        artifact_sha256: match profile {
            DepthAnything3FixtureProfile::Dpt => {
                "31fb08778cfcab7adce81435b4799fe4b70aceb134c008106b03924de804b34f"
            }
            DepthAnything3FixtureProfile::DualDpt => {
                "e23ba80bff17b5362d92f8a5f6078657bd2f61cd2df24b369e68b70f20934619"
            }
        }
        .to_owned(),
        ordered_state,
        memory_budget_bytes,
    })
}

#[cfg(any(test, feature = "test-support"))]
pub fn reduced_depth_anything_3_checkpoint_parity_for_fixture(
    resource: &NativeDepthAnything3Resource,
    cancellation: &CancellationToken,
) -> Result<DepthAnything3FixtureCheckpointParity, NativeDepthAnything3Error> {
    cancellation.check()?;
    let mut states = Vec::new();
    states
        .try_reserve_exact(resource.source_state.len())
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    let mut source_aggregate = Sha256::new();
    source_aggregate.update(b"zed.sim.depth-anything-3.checkpoint-source-aggregate.v1\0");
    let mut projected_aggregate = Sha256::new();
    projected_aggregate.update(b"zed.sim.depth-anything-3.checkpoint-projected-aggregate.v1\0");
    let state_count = u64::try_from(resource.source_state.len())
        .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?;
    source_aggregate.update(state_count.to_le_bytes());
    projected_aggregate.update(state_count.to_le_bytes());
    for (index, (key, source)) in resource.source_state.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        let projected = resource
            .execution_state
            .get(key)
            .ok_or_else(|| NativeDepthAnything3Error::MissingState(key.clone()))?;
        let source_sha256 = fixture_state_identity_sha256(
            b"zed.sim.depth-anything-3.checkpoint-source-state.v1\0",
            key,
            source,
            cancellation,
        )?;
        let projected_f32_sha256 = fixture_state_identity_sha256(
            b"zed.sim.depth-anything-3.checkpoint-projected-state.v1\0",
            key,
            projected,
            cancellation,
        )?;
        source_aggregate.update(source_sha256.as_bytes());
        projected_aggregate.update(projected_f32_sha256.as_bytes());
        states.push(DepthAnything3FixtureStateParity {
            key: key.clone(),
            shape: source.descriptor().shape().to_vec(),
            source_sha256,
            projected_f32_sha256,
        });
    }
    if resource.execution_state.len() != states.len() {
        return Err(NativeDepthAnything3Error::SemanticStateChanged);
    }
    cancellation.check()?;
    Ok(DepthAnything3FixtureCheckpointParity {
        source_sha256: format!("{:x}", source_aggregate.finalize()),
        projected_f32_sha256: format!("{:x}", projected_aggregate.finalize()),
        states,
    })
}

#[cfg(any(test, feature = "test-support"))]
fn fixture_state_identity_sha256(
    domain: &[u8],
    key: &str,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<String, NativeDepthAnything3Error> {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(
        u64::try_from(key.len())
            .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(key.as_bytes());
    let dtype = tensor.descriptor().dtype().catalog_name();
    hasher.update(
        u64::try_from(dtype.len())
            .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?
            .to_le_bytes(),
    );
    hasher.update(dtype.as_bytes());
    hasher.update(
        u64::try_from(tensor.descriptor().shape().len())
            .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?
            .to_le_bytes(),
    );
    for dimension in tensor.descriptor().shape() {
        hasher.update(dimension.to_le_bytes());
    }
    let bytes = tensor.contiguous_bytes()?;
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?
            .to_le_bytes(),
    );
    for chunk in bytes.chunks(64 * 1_024) {
        cancellation.check()?;
        hasher.update(chunk);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(any(test, feature = "test-support"))]
pub fn mutate_reduced_depth_anything_3_checkpoint(
    backend: &CpuBackend,
    checkpoint: &mut NativeDepthAnything3Checkpoint,
    mutation: DepthAnything3FixtureMutation<'_>,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeDepthAnything3Error> {
    context.check()?;
    let (_, tensor) = checkpoint
        .ordered_state
        .iter_mut()
        .find(|(key, _)| key == mutation.state_key)
        .ok_or_else(|| NativeDepthAnything3Error::MissingState(mutation.state_key.to_owned()))?;
    let descriptor = tensor.descriptor().clone();
    let mut values = tensor_to_f32_with_context_exact_native(backend, tensor, context)?;
    let value = values
        .get_mut(mutation.lane)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    *value += mutation.delta;
    context.check()?;
    *tensor = tensor_from_f32_with_context_exact_native(
        backend,
        descriptor.shape(),
        &values,
        descriptor.dtype(),
        descriptor.device(),
        context,
    )?;
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
pub fn select_reduced_depth_anything_3_reference_for_fixture(
    class_tokens: &[f32],
    views: usize,
    strategy: NativeDepthAnything3ReferenceStrategy,
    cancellation: &CancellationToken,
) -> Result<usize, NativeDepthAnything3Error> {
    if views == 0 || !class_tokens.len().is_multiple_of(views) {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    }
    let strategy = match strategy {
        NativeDepthAnything3ReferenceStrategy::First => NativeDino2ReferenceStrategy::First,
        NativeDepthAnything3ReferenceStrategy::Middle => NativeDino2ReferenceStrategy::Middle,
        NativeDepthAnything3ReferenceStrategy::SaddleBalanced => {
            NativeDino2ReferenceStrategy::SaddleBalanced
        }
        NativeDepthAnything3ReferenceStrategy::SaddleSimRange => {
            NativeDino2ReferenceStrategy::SaddleSimRange
        }
    };
    crate::dino2::select_reference_indices(
        class_tokens,
        1,
        views,
        1,
        class_tokens.len() / views,
        strategy,
        cancellation,
    )?
    .into_iter()
    .next()
    .ok_or(NativeDepthAnything3Error::ShapeOverflow)
}

#[cfg(any(test, feature = "test-support"))]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "test consumer is the reduced DINOv2 fixture suite"
    )
)]
pub(crate) fn execute_reduced_dino2_ordinary_for_fixture(
    backend: &CpuBackend,
    image: &ImageTensor,
    source_dtype: DType,
    mutation: Option<DepthAnything3FixtureMutation<'_>>,
    memory_budget_bytes: u64,
    context: &ExecutionContext<'_>,
) -> Result<
    (
        Vec<crate::dino2::NativeDino2Feature>,
        crate::dino2::NativeDino2Feature,
    ),
    NativeDepthAnything3Error,
> {
    const MASK_TOKEN_KEY: &str = "native.backbone.embeddings.mask_token";
    let mut checkpoint = deterministic_reduced_depth_anything_3_checkpoint(
        backend,
        DepthAnything3FixtureProfile::Dpt,
        source_dtype,
        memory_budget_bytes,
        context,
    )?;
    if let Some(mutation) = mutation
        .as_ref()
        .filter(|mutation| mutation.state_key != MASK_TOKEN_KEY)
    {
        mutate_reduced_depth_anything_3_checkpoint(
            backend,
            &mut checkpoint,
            mutation.clone(),
            context,
        )?;
    }
    let resource =
        NativeDepthAnything3Resource::from_reduced_fixture(backend, checkpoint, context)?;
    let image = preprocess_image(backend, image, 4, 4, context)?;
    let mut configuration = resource.configuration;
    configuration.qknorm_start = None;
    configuration.alternate_attention_start = None;
    configuration.rope_start = None;
    configuration.concatenate_camera_token = false;
    configuration.head_dimension = configuration.hidden_size;
    let backbone = native_dino2_backbone_with_mask_token(configuration, true)?;
    let mut execution_state = BTreeMap::new();
    let mut resident_bytes = resource.resident_bytes;
    for specification in backbone.state_manifest()? {
        let is_mask_token = specification.key == MASK_TOKEN_KEY;
        let tensor = if is_mask_token {
            let mut values = [-0.035_f32, -0.0025, 0.03, -0.01];
            if let Some(mutation) = mutation
                .as_ref()
                .filter(|mutation| mutation.state_key == MASK_TOKEN_KEY)
            {
                let value = values
                    .get_mut(mutation.lane)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
                *value += mutation.delta;
            }
            let source = tensor_from_f32_with_context_exact_native(
                backend,
                &specification.shape,
                &values,
                source_dtype,
                DeviceId::CPU,
                context,
            )?;
            backbone.project_state_tensor(backend, &specification.key, &source, context)?
        } else {
            resource
                .execution_state
                .get(&specification.key)
                .ok_or_else(|| NativeDepthAnything3Error::MissingState(specification.key.clone()))?
                .clone()
        };
        if is_mask_token {
            resident_bytes = resident_bytes
                .checked_add(tensor.storage_byte_len())
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        }
        execution_state.insert(specification.key, tensor);
    }
    let execution = backbone.bind(
        &execution_state,
        resource.memory_budget_bytes,
        resident_bytes,
    );
    let intermediates = execution.get_intermediate_layers(backend, &image, 1, context)?;
    let forward = execution.forward(backend, &image, 1, context)?;
    Ok((intermediates, forward))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StateSpecification {
    key: String,
    shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DepthAnything3ExecutionProfile {
    configuration: DepthAnything3Configuration,
    source_exact: bool,
}

impl DepthAnything3ExecutionProfile {
    fn production(
        configuration: DepthAnything3Configuration,
    ) -> Result<Self, NativeDepthAnything3Error> {
        if configuration.patch_size != 14
            || configuration.image_size != 518
            || configuration.hidden_size != configuration.backbone.hidden_size()
            || configuration.layer_count != configuration.backbone.layer_count()
            || configuration.attention_heads != configuration.backbone.attention_heads()
        {
            return Err(NativeDepthAnything3Error::UnsupportedArchitecture);
        }
        Ok(Self {
            configuration,
            source_exact: true,
        })
    }

    #[cfg(any(test, feature = "test-support"))]
    fn reduced(state: &BTreeMap<String, Tensor>) -> Self {
        let head = if state
            .keys()
            .any(|key| key.starts_with("native.head.scratch.refinenet1_aux."))
        {
            DepthAnything3Head::DualDpt
        } else {
            DepthAnything3Head::Dpt
        };
        let concatenate_camera_token =
            state.contains_key("native.backbone.embeddings.camera_token");
        let has_camera_encoder = state.contains_key("native.cam_enc.pose_branch.fc1.weight");
        let has_camera_decoder = state.contains_key("native.cam_dec.fc_t.weight");
        let hidden_size = 4;
        Self {
            configuration: DepthAnything3Configuration {
                backbone: DepthAnything3Backbone::VitSmall,
                hidden_size,
                layer_count: 4,
                attention_heads: 1,
                patch_size: 2,
                image_size: 4,
                qknorm_start: Some(1),
                alternate_attention_start: concatenate_camera_token.then_some(2),
                rope_start: Some(1),
                concatenate_camera_token,
                head,
                head_dimension: if concatenate_camera_token {
                    hidden_size * 2
                } else {
                    hidden_size
                },
                head_features: if head == DepthAnything3Head::DualDpt {
                    8
                } else {
                    4
                },
                head_out_channels: if head == DepthAnything3Head::DualDpt {
                    [4, 4, 4, 4]
                } else {
                    [2, 2, 2, 2]
                },
                head_output_dimension: if head == DepthAnything3Head::DualDpt {
                    2
                } else {
                    1
                },
                output_layers: [0, 1, 2, 3],
                use_sky_head: head == DepthAnything3Head::Dpt,
                has_camera_encoder,
                camera_encoder_dimension: has_camera_encoder.then_some(hidden_size),
                has_camera_decoder,
                camera_decoder_dimension: has_camera_decoder.then_some(
                    if concatenate_camera_token {
                        hidden_size * 2
                    } else {
                        hidden_size
                    },
                ),
            },
            source_exact: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeDepthAnything3Resource {
    configuration: DepthAnything3Configuration,
    artifact_sha256: String,
    source_state: BTreeMap<String, Tensor>,
    execution_state: BTreeMap<String, Tensor>,
    source_dtype: DType,
    stream: StreamId,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    semantic_digest_sha256: String,
    source_exact: bool,
}

impl NativeDepthAnything3Resource {
    pub fn from_checkpoint(
        backend: &CpuBackend,
        checkpoint: NativeDepthAnything3Checkpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDepthAnything3Error> {
        Self::checked(backend, checkpoint, true, context)
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn from_reduced_fixture(
        backend: &CpuBackend,
        checkpoint: NativeDepthAnything3Checkpoint,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDepthAnything3Error> {
        Self::checked(backend, checkpoint, false, context)
    }

    fn checked(
        backend: &CpuBackend,
        checkpoint: NativeDepthAnything3Checkpoint,
        source_exact: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, NativeDepthAnything3Error> {
        context.check()?;
        validate_sha256(&checkpoint.artifact_sha256)?;
        if checkpoint.ordered_state.is_empty()
            || checkpoint.ordered_state.len() > MAX_STATE_TENSORS
            || checkpoint.memory_budget_bytes == 0
        {
            return Err(NativeDepthAnything3Error::InvalidCheckpoint(
                "state cardinality or memory budget is invalid".to_owned(),
            ));
        }

        let mut source_state = BTreeMap::new();
        for (index, (key, tensor)) in checkpoint.ordered_state.into_iter().enumerate() {
            if index.is_multiple_of(32) {
                context.check()?;
            }
            validate_state_key(&key)?;
            if tensor.descriptor().stream() != context.stream {
                return Err(NativeDepthAnything3Error::InvalidCheckpoint(format!(
                    "state {key} is on a foreign execution stream"
                )));
            }
            if source_state.insert(key.clone(), tensor).is_some() {
                return Err(NativeDepthAnything3Error::DuplicateStateKey(key));
            }
        }

        let profile = if source_exact {
            DepthAnything3ExecutionProfile::production(configuration_from_native_state(
                &source_state,
            )?)?
        } else {
            #[cfg(any(test, feature = "test-support"))]
            {
                DepthAnything3ExecutionProfile::reduced(&source_state)
            }
            #[cfg(not(any(test, feature = "test-support")))]
            {
                return Err(NativeDepthAnything3Error::UnsupportedArchitecture);
            }
        };
        let configuration = profile.configuration;
        let specifications = depth_anything_3_state_manifest(profile)?;
        let source_dtype = validate_strict_source_state(
            &source_state,
            &specifications,
            context.stream,
            context.cancellation,
        )?;

        let preflight_resident_bytes = conservative_projected_resident_bytes(
            &checkpoint.artifact_sha256,
            &source_state,
            context.cancellation,
        )?;
        if preflight_resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeDepthAnything3Error::OutOfMemory {
                required: preflight_resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }

        let backbone = native_dino2_backbone(configuration)?;
        let mut execution_state = BTreeMap::new();
        for (index, (key, tensor)) in source_state.iter().enumerate() {
            if index.is_multiple_of(16) {
                context.check()?;
            }
            let projected = if backbone.owns_state_key(key) {
                backbone.project_state_tensor(backend, key, tensor, context)?
            } else {
                cast_to_with_context_exact_native(
                    backend,
                    tensor,
                    DType::F32,
                    DeviceId::CPU,
                    false,
                    true,
                    context,
                )?
            };
            execution_state.insert(key.clone(), projected);
        }
        validate_f32_execution_state(
            &execution_state,
            &specifications,
            context.stream,
            context.cancellation,
        )?;

        let semantic_digest_sha256 = semantic_digest(
            &configuration,
            &checkpoint.artifact_sha256,
            source_dtype,
            &source_state,
            &execution_state,
            context.cancellation,
        )?;
        let resident_bytes =
            resident_tensor_bytes([&source_state, &execution_state], context.cancellation)?
                .checked_add(resident_owned_bytes(
                    &checkpoint.artifact_sha256,
                    &semantic_digest_sha256,
                    &source_state,
                    &execution_state,
                )?)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        if resident_bytes > checkpoint.memory_budget_bytes {
            return Err(NativeDepthAnything3Error::OutOfMemory {
                required: resident_bytes,
                budget: checkpoint.memory_budget_bytes,
            });
        }
        context.check()?;
        Ok(Self {
            configuration,
            artifact_sha256: checkpoint.artifact_sha256,
            source_state,
            execution_state,
            source_dtype,
            stream: context.stream,
            memory_budget_bytes: checkpoint.memory_budget_bytes,
            resident_bytes,
            semantic_digest_sha256,
            source_exact,
        })
    }

    pub fn identifier(&self) -> &'static str {
        match (self.configuration.backbone, self.configuration.head) {
            (DepthAnything3Backbone::VitSmall, DepthAnything3Head::DualDpt) => "da3-small",
            (DepthAnything3Backbone::VitBase, DepthAnything3Head::DualDpt) => "da3-base",
            (DepthAnything3Backbone::VitLarge, DepthAnything3Head::Dpt) => "da3-large",
            (DepthAnything3Backbone::VitGiant, DepthAnything3Head::Dpt) => "da3-giant",
            _ => "da3-unsupported",
        }
    }

    pub fn configuration(&self) -> &DepthAnything3Configuration {
        &self.configuration
    }

    pub const fn source_dtype(&self) -> DType {
        self.source_dtype
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn is_source_exact_profile(&self) -> bool {
        self.source_exact
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, NativeDepthAnything3Error> {
        resident_owned_bytes(
            &self.artifact_sha256,
            &self.semantic_digest_sha256,
            &self.source_state,
            &self.execution_state,
        )
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(StorageId, u64)>, NativeDepthAnything3Error> {
        resident_tensor_allocations(
            [&self.source_state, &self.execution_state],
            &CancellationToken::default(),
        )
    }

    pub fn reconstruct_checkpoint(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeDepthAnything3Checkpoint, NativeDepthAnything3Error> {
        self.validate(cancellation)?;
        let mut ordered_state = Vec::new();
        ordered_state
            .try_reserve_exact(self.source_state.len())
            .map_err(|_| NativeDepthAnything3Error::Allocation)?;
        for (index, (key, tensor)) in self.source_state.iter().enumerate() {
            if index.is_multiple_of(32) {
                cancellation.check()?;
            }
            ordered_state.push((key.clone(), tensor.clone()));
        }
        Ok(NativeDepthAnything3Checkpoint {
            artifact_sha256: self.artifact_sha256.clone(),
            ordered_state,
            memory_budget_bytes: self.memory_budget_bytes,
        })
    }

    pub fn validate(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeDepthAnything3Error> {
        cancellation.check()?;
        let specifications = depth_anything_3_state_manifest(DepthAnything3ExecutionProfile {
            configuration: self.configuration,
            source_exact: self.source_exact,
        })?;
        let dtype = validate_strict_source_state(
            &self.source_state,
            &specifications,
            self.stream,
            cancellation,
        )?;
        if dtype != self.source_dtype {
            return Err(NativeDepthAnything3Error::SemanticStateChanged);
        }
        validate_f32_execution_state(
            &self.execution_state,
            &specifications,
            self.stream,
            cancellation,
        )?;
        let digest = semantic_digest(
            &self.configuration,
            &self.artifact_sha256,
            self.source_dtype,
            &self.source_state,
            &self.execution_state,
            cancellation,
        )?;
        let resident =
            resident_tensor_bytes([&self.source_state, &self.execution_state], cancellation)?
                .checked_add(self.resident_owned_bytes()?)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        if digest != self.semantic_digest_sha256 || resident != self.resident_bytes {
            return Err(NativeDepthAnything3Error::SemanticStateChanged);
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn execute(
        &self,
        backend: &CpuBackend,
        invocation: NativeDepthAnything3Invocation<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeDepthAnything3Geometry, NativeDepthAnything3Error> {
        self.execute_internal(backend, invocation, context, None)
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn execute_with_test_trace(
        &self,
        backend: &CpuBackend,
        invocation: NativeDepthAnything3Invocation<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<
        (NativeDepthAnything3Geometry, NativeDepthAnything3TestTrace),
        NativeDepthAnything3Error,
    > {
        let mut trace = DepthAnything3ExecutionTrace::default();
        let geometry = self.execute_internal(backend, invocation, context, Some(&mut trace))?;
        Ok((
            geometry,
            NativeDepthAnything3TestTrace {
                raw_ray: trace
                    .raw_ray
                    .ok_or(NativeDepthAnything3Error::SemanticStateChanged)?,
                raw_ray_confidence: trace
                    .raw_ray_confidence
                    .ok_or(NativeDepthAnything3Error::SemanticStateChanged)?,
                ransac_samples: trace
                    .ransac_samples
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|value| {
                                u64::try_from(value)
                                    .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
                            })
                            .collect()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                ransac_views: trace
                    .ransac_views
                    .into_iter()
                    .map(|view| {
                        Ok(NativeDepthAnything3RansacViewTrace {
                            candidate_indices: view
                                .candidate_indices
                                .into_iter()
                                .map(|value| {
                                    u64::try_from(value)
                                        .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                            best_iteration: u64::try_from(view.best_iteration)
                                .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
                            best_inliers: view
                                .best_inliers
                                .into_iter()
                                .map(|value| {
                                    u64::try_from(value)
                                        .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                            best_score_bits: view.best_score.to_bits(),
                            fallback: view.fallback,
                            normalized_homography: view.normalized_homography,
                            signed_homography: view.signed_homography,
                            rotation: view.rotation,
                            lower: view.lower,
                            c2w_pre_inverse: view.c2w_pre_inverse,
                        })
                    })
                    .collect::<Result<Vec<_>, NativeDepthAnything3Error>>()?,
            },
        ))
    }

    #[cfg(test)]
    fn execute_pre_geometry_head_with_test_trace(
        &self,
        backend: &CpuBackend,
        invocation: NativeDepthAnything3Invocation<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<
        (NativeDepthAnything3Geometry, DepthAnything3ExecutionTrace),
        NativeDepthAnything3Error,
    > {
        let mut trace = DepthAnything3ExecutionTrace {
            stop_before_ray_pose: true,
            ..Default::default()
        };
        let geometry = self.execute_internal(backend, invocation, context, Some(&mut trace))?;
        Ok((geometry, trace))
    }

    fn execute_internal(
        &self,
        backend: &CpuBackend,
        invocation: NativeDepthAnything3Invocation<'_>,
        context: &ExecutionContext<'_>,
        trace: Option<&mut DepthAnything3ExecutionTrace>,
    ) -> Result<NativeDepthAnything3Geometry, NativeDepthAnything3Error> {
        self.validate(context.cancellation)?;
        context.check()?;
        if context.stream != self.stream {
            return Err(NativeDepthAnything3Error::StaleExecutionStream {
                expected: self.stream,
                actual: context.stream,
            });
        }
        let (flat_batch, original_height, original_width, channels) =
            invocation.image.dimensions()?;
        if channels != 3
            || flat_batch == 0
            || invocation.views_per_sample == 0
            || !flat_batch.is_multiple_of(invocation.views_per_sample)
            || invocation.process_resolution == 0
        {
            return Err(NativeDepthAnything3Error::InvalidImage(
                "expected a nonempty BHWC RGB batch divisible by the view count".to_owned(),
            ));
        }
        if invocation.views_per_sample == 1
            && flat_batch > 1
            && invocation.extrinsics.is_none()
            && invocation.intrinsics.is_none()
        {
            return self.execute_mono_serial(backend, invocation, context);
        }
        validate_finite_image(invocation.image, context.cancellation)?;
        validate_camera_inputs(
            invocation.extrinsics,
            invocation.intrinsics,
            flat_batch / invocation.views_per_sample,
            invocation.views_per_sample,
            context,
        )?;
        if invocation.use_ray_pose && self.configuration.head != DepthAnything3Head::DualDpt {
            return Err(NativeDepthAnything3Error::UnsupportedInvocation(
                "ray-pose geometry requires a DualDPT resource".to_owned(),
            ));
        }
        let (target_height, target_width) = target_size(
            original_height,
            original_width,
            invocation.process_resolution,
            invocation.resize_method,
            self.configuration.patch_size,
        )?;
        if !target_height.is_multiple_of(self.configuration.patch_size)
            || !target_width.is_multiple_of(self.configuration.patch_size)
        {
            return Err(NativeDepthAnything3Error::InvalidImage(
                "source target size is not divisible by the patch size".to_owned(),
            ));
        }
        preflight_invocation_memory(
            self,
            flat_batch,
            invocation.views_per_sample,
            target_height,
            target_width,
            original_height,
            original_width,
            context,
        )?;
        let preprocessed = preprocess_image(
            backend,
            invocation.image,
            target_height,
            target_width,
            context,
        )?;
        let camera_token = match (invocation.extrinsics, invocation.intrinsics) {
            (Some(extrinsics), Some(intrinsics)) => Some(encode_camera_token(
                self,
                backend,
                extrinsics,
                intrinsics,
                usize_from(flat_batch / invocation.views_per_sample)?,
                usize_from(invocation.views_per_sample)?,
                usize_from(target_height)?,
                usize_from(target_width)?,
                context,
            )?),
            (None, None) => None,
            _ => {
                return Err(NativeDepthAnything3Error::UnsupportedInvocation(
                    "camera inputs changed after validation".to_owned(),
                ));
            }
        };
        let features = execute_backbone(
            self,
            backend,
            &preprocessed,
            usize_from(flat_batch / invocation.views_per_sample)?,
            usize_from(invocation.views_per_sample)?,
            invocation.reference_strategy,
            camera_token.as_deref(),
            context,
        )?;
        let head = execute_depth_head(
            self,
            backend,
            &features,
            usize_from(flat_batch / invocation.views_per_sample)?,
            usize_from(invocation.views_per_sample)?,
            usize_from(target_height)?,
            usize_from(target_width)?,
            invocation.use_ray_pose,
            invocation.ransac_seed,
            context,
            trace,
        )?;
        project_geometry(
            backend,
            head,
            flat_batch,
            original_height,
            original_width,
            invocation.views_per_sample,
            context,
        )
    }

    fn execution_tensor(&self, key: &str) -> Result<&Tensor, NativeDepthAnything3Error> {
        self.execution_state
            .get(key)
            .ok_or_else(|| NativeDepthAnything3Error::MissingState(key.to_owned()))
    }

    fn execute_mono_serial(
        &self,
        backend: &CpuBackend,
        invocation: NativeDepthAnything3Invocation<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeDepthAnything3Geometry, NativeDepthAnything3Error> {
        let (batch, height, width, _) = invocation.image.dimensions()?;
        preflight_serial_outputs(
            self,
            batch,
            height,
            width,
            invocation.process_resolution,
            invocation.resize_method,
            context,
        )?;
        let mut depths = Vec::new();
        let mut confidences = Vec::new();
        let mut skies = Vec::new();
        depths
            .try_reserve_exact(usize_from(batch)?)
            .map_err(|_| NativeDepthAnything3Error::Allocation)?;
        confidences
            .try_reserve_exact(usize_from(batch)?)
            .map_err(|_| NativeDepthAnything3Error::Allocation)?;
        skies
            .try_reserve_exact(usize_from(batch)?)
            .map_err(|_| NativeDepthAnything3Error::Allocation)?;
        for index in 0..batch {
            context.check()?;
            let single = ImageTensor::from_tensor(narrow_method_exact_native(
                invocation.image.tensor(),
                0,
                i64::try_from(index).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
                1,
                context.cancellation,
            )?)?;
            let geometry = self.execute(
                backend,
                NativeDepthAnything3Invocation {
                    image: &single,
                    views_per_sample: 1,
                    process_resolution: invocation.process_resolution,
                    resize_method: invocation.resize_method,
                    reference_strategy: invocation.reference_strategy,
                    use_ray_pose: invocation.use_ray_pose,
                    ransac_seed: invocation.ransac_seed,
                    extrinsics: None,
                    intrinsics: None,
                },
                context,
            )?;
            depths.push(geometry.depth);
            if let Some(confidence) = geometry.confidence {
                confidences.push(confidence);
            }
            if let Some(sky) = geometry.sky {
                skies.push(sky);
            }
        }
        let confidence = if confidences.is_empty() {
            None
        } else if confidences.len() == depths.len() {
            Some(torch_cat_with_context_exact_native(
                backend,
                &confidences,
                0,
                context,
            )?)
        } else {
            return Err(NativeDepthAnything3Error::SemanticStateChanged);
        };
        let sky = if skies.is_empty() {
            None
        } else if skies.len() == depths.len() {
            Some(torch_cat_with_context_exact_native(
                backend, &skies, 0, context,
            )?)
        } else {
            return Err(NativeDepthAnything3Error::SemanticStateChanged);
        };
        Ok(NativeDepthAnything3Geometry {
            depth: torch_cat_with_context_exact_native(backend, &depths, 0, context)?,
            confidence,
            sky,
            extrinsics: None,
            intrinsics: None,
            original_height: height,
            original_width: width,
            views_per_sample: 1,
        })
    }
}

fn preflight_serial_outputs(
    resource: &NativeDepthAnything3Resource,
    batch: u64,
    height: u64,
    width: u64,
    process_resolution: u64,
    resize_method: NativeDepthAnything3ResizeMethod,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeDepthAnything3Error> {
    context.check()?;
    let required = serial_invocation_memory_required(
        resource,
        batch,
        height,
        width,
        process_resolution,
        resize_method,
    )?;
    if required > resource.memory_budget_bytes {
        return Err(NativeDepthAnything3Error::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    context.check()?;
    Ok(())
}

fn serial_invocation_memory_required(
    resource: &NativeDepthAnything3Resource,
    batch: u64,
    height: u64,
    width: u64,
    process_resolution: u64,
    resize_method: NativeDepthAnything3ResizeMethod,
) -> Result<u64, NativeDepthAnything3Error> {
    let output_planes = 1_u64
        .checked_add(u64::from(resource.configuration.head_output_dimension > 1))
        .and_then(|value| value.checked_add(u64::from(resource.configuration.use_sky_head)))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let output_bytes = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(output_planes))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let single_output_bytes = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(output_planes))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let original_input_bytes = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let (target_height, target_width) = target_size(
        height,
        width,
        process_resolution,
        resize_method,
        resource.configuration.patch_size,
    )?;
    let single_invocation =
        invocation_memory_required(resource, 1, 1, target_height, target_width, height, width)?;
    let single_transient = single_invocation
        .checked_sub(resource.resident_bytes)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let prior_outputs = batch
        .checked_sub(1)
        .and_then(|value| value.checked_mul(single_output_bytes))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let final_iteration_peak = resource
        .resident_bytes
        .checked_add(original_input_bytes)
        .and_then(|value| value.checked_add(prior_outputs))
        .and_then(|value| value.checked_add(single_transient))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let concatenation_peak = resource
        .resident_bytes
        .checked_add(original_input_bytes)
        .and_then(|value| value.checked_add(output_bytes.checked_mul(2)?))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    Ok(final_iteration_peak.max(concatenation_peak))
}

#[derive(Debug, Error)]
pub enum NativeDepthAnything3Error {
    #[error("Depth Anything 3 execution was cancelled")]
    Cancelled,
    #[error("Depth Anything 3 checkpoint is invalid: {0}")]
    InvalidCheckpoint(String),
    #[error("Depth Anything 3 checkpoint has duplicate state key {0}")]
    DuplicateStateKey(String),
    #[error("Depth Anything 3 architecture is unsupported or ambiguous")]
    UnsupportedArchitecture,
    #[error("Depth Anything 3 checkpoint is missing state key {0}")]
    MissingState(String),
    #[error("Depth Anything 3 checkpoint has unexpected state key {0}")]
    UnexpectedState(String),
    #[error("Depth Anything 3 state {key} expected {expected:?}, got {actual:?} {actual_dtype:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        actual_dtype: DType,
    },
    #[error("Depth Anything 3 retained semantic state changed")]
    SemanticStateChanged,
    #[error("Depth Anything 3 execution stream is stale: expected {expected:?}, got {actual:?}")]
    StaleExecutionStream {
        expected: StreamId,
        actual: StreamId,
    },
    #[error("Depth Anything 3 image is invalid: {0}")]
    InvalidImage(String),
    #[error("Depth Anything 3 invocation is unsupported: {0}")]
    UnsupportedInvocation(String),
    #[error("Depth Anything 3 memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error("Depth Anything 3 shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("Depth Anything 3 allocation failed")]
    Allocation,
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error(transparent)]
    ElementwiseThree(#[from] ElementwiseRuntimePartThreeError),
    #[error(transparent)]
    ElementwiseFourteen(#[from] ElementwiseRuntimePartFourteenError),
    #[error(transparent)]
    ElementwiseFive(#[from] ElementwiseRuntimePartFiveError),
    #[error(transparent)]
    ElementwiseSixteen(#[from] ElementwiseRuntimePartSixteenError),
    #[error(transparent)]
    ElementwiseSeventeen(#[from] ElementwiseRuntimePartSeventeenError),
    #[error(transparent)]
    ElementwiseEighteen(#[from] ElementwiseRuntimePartEighteenError),
    #[error(transparent)]
    ElementwiseTwentyOne(#[from] ElementwiseRuntimePartTwentyOneError),
    #[error(transparent)]
    NeuralFunctional(#[from] NeuralNetworkFunctionalError),
    #[error(transparent)]
    ShapeLayoutThree(#[from] ShapeLayoutTransformPartThreeError),
    #[error(transparent)]
    Spatial(#[from] SpatialFunctionalKernelError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    LinearAlgebraOne(#[from] LinearAlgebraPartOneError),
    #[error(transparent)]
    LinearAlgebraTwo(#[from] LinearAlgebraPartTwoError),
    #[error(transparent)]
    Random(#[from] RandomNumberGenerationPartOneError),
    #[error(transparent)]
    TensorCreationOne(#[from] TensorCreationPartOneError),
    #[error(transparent)]
    IndexingOne(#[from] IndexingMaskingPartOneError),
    #[error(transparent)]
    ShapeLayoutTwo(#[from] ShapeLayoutTransformPartTwoError),
}

#[derive(Clone, Debug)]
struct DepthHeadOutput {
    depth: Tensor,
    confidence: Option<Tensor>,
    sky: Option<Tensor>,
    extrinsics: Option<Tensor>,
    intrinsics: Option<Tensor>,
}

#[derive(Clone, Debug, Default)]
struct DepthAnything3ExecutionTrace {
    stop_before_ray_pose: bool,
    auxiliary_resized: Vec<Tensor>,
    auxiliary_fusion: Vec<Tensor>,
    auxiliary_processed: Vec<Tensor>,
    auxiliary_selected_convolutions: Vec<Tensor>,
    auxiliary_position_lane: Option<DepthAnything3PositionLaneTrace>,
    auxiliary_positioned: Option<Tensor>,
    auxiliary_convolution: Option<Tensor>,
    auxiliary_normalized: Option<Tensor>,
    auxiliary_relu: Option<Tensor>,
    auxiliary_logits: Option<Tensor>,
    raw_ray: Option<Tensor>,
    raw_ray_confidence: Option<Tensor>,
    ransac_samples: Vec<Vec<usize>>,
    ransac_views: Vec<DepthAnything3RansacViewTrace>,
}

#[derive(Clone, Debug, Default)]
struct DepthAnything3PositionLaneTrace {
    x_coordinate_bits: Vec<u32>,
    y_coordinate_bits: Vec<u32>,
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    exponent_ratio_bits: u32,
    positive_power_bits: u32,
    reciprocal_bits: u32,
    negative_power_bits: u32,
    frequency_bits: u32,
    discriminator_ratio_bits: u32,
    discriminator_positive_power_bits: u32,
    discriminator_reciprocal_bits: u32,
    discriminator_negative_power_bits: u32,
    position_bits: u32,
    phase_bits: u32,
    sine_bits: u32,
    cosine_bits: u32,
    selected_trig: &'static str,
    selected_trig_bits: u32,
    ratio_bits: u32,
    embedding_bits: u32,
    input_bits: u32,
    output_bits: u32,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    not(any(test, feature = "test-support")),
    expect(
        dead_code,
        reason = "fields are projected by the exact Depth Anything 3 test trace"
    )
)]
struct DepthAnything3RansacViewTrace {
    candidate_indices: Vec<usize>,
    best_iteration: usize,
    best_inliers: Vec<usize>,
    best_score: f32,
    fallback: bool,
    normalized_homography: [f32; 9],
    signed_homography: [f32; 9],
    rotation: [f32; 9],
    lower: [f32; 9],
    c2w_pre_inverse: [f32; 12],
}

impl From<sim_types::CancellationError> for NativeDepthAnything3Error {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<crate::dino2::NativeDino2Error> for NativeDepthAnything3Error {
    fn from(error: crate::dino2::NativeDino2Error) -> Self {
        match error {
            NativeDino2Error::Cancelled => Self::Cancelled,
            NativeDino2Error::UnsupportedArchitecture => Self::UnsupportedArchitecture,
            NativeDino2Error::InvalidImage(message) => Self::InvalidImage(message),
            NativeDino2Error::MissingState(key) => Self::MissingState(key),
            NativeDino2Error::UnexpectedState(key) => Self::UnexpectedState(key),
            NativeDino2Error::StateShape {
                key,
                expected,
                actual,
                actual_dtype,
            } => Self::StateShape {
                key,
                expected,
                actual,
                actual_dtype,
            },
            NativeDino2Error::ShapeOverflow => Self::ShapeOverflow,
            NativeDino2Error::Allocation => Self::Allocation,
            NativeDino2Error::OutOfMemory { required, budget } => {
                Self::OutOfMemory { required, budget }
            }
            NativeDino2Error::SemanticStateChanged => Self::SemanticStateChanged,
            NativeDino2Error::Tensor(error) => Self::Tensor(error),
            NativeDino2Error::Operator(error) => Self::Operator(error),
            NativeDino2Error::Functional(error) => Self::Functional(error),
            NativeDino2Error::ElementwiseSixteen(error) => Self::ElementwiseSixteen(error),
            NativeDino2Error::NeuralFunctional(error) => Self::NeuralFunctional(error),
            NativeDino2Error::ShapeLayoutThree(error) => Self::ShapeLayoutThree(error),
            NativeDino2Error::Spatial(error) => Self::Spatial(error),
            NativeDino2Error::Attention(error) => Self::Attention(error),
        }
    }
}

fn validate_finite_image(
    image: &ImageTensor,
    cancellation: &CancellationToken,
) -> Result<(), NativeDepthAnything3Error> {
    for (index, values) in image.as_f32_slice()?.chunks(16_384).enumerate() {
        if index.is_multiple_of(4) {
            cancellation.check()?;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(NativeDepthAnything3Error::InvalidImage(
                "RGB pixels must be finite".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_camera_inputs(
    extrinsics: Option<&Tensor>,
    intrinsics: Option<&Tensor>,
    batch: u64,
    views: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeDepthAnything3Error> {
    if extrinsics.is_some() != intrinsics.is_some() {
        return Err(NativeDepthAnything3Error::UnsupportedInvocation(
            "extrinsics and intrinsics must be supplied together".to_owned(),
        ));
    }
    for (name, tensor, tail) in [
        ("extrinsics", extrinsics, [3_u64, 4_u64]),
        ("intrinsics", intrinsics, [3_u64, 3_u64]),
    ] {
        let Some(tensor) = tensor else { continue };
        if tensor.descriptor().shape() != [batch, views, tail[0], tail[1]]
            || tensor.descriptor().dtype() != DType::F32
            || tensor.descriptor().device() != DeviceId::CPU
            || tensor.descriptor().stream() != context.stream
            || !tensor.descriptor().is_contiguous()?
        {
            return Err(NativeDepthAnything3Error::UnsupportedInvocation(format!(
                "{name} has invalid shape, dtype, placement, or stream"
            )));
        }
        for bytes in tensor.contiguous_bytes()?.chunks(64 * 1_024) {
            context.check()?;
            if bytes.chunks_exact(4).any(|value| {
                !f32::from_le_bytes([value[0], value[1], value[2], value[3]]).is_finite()
            }) {
                return Err(NativeDepthAnything3Error::UnsupportedInvocation(format!(
                    "{name} contains a non-finite value"
                )));
            }
        }
    }
    Ok(())
}

fn target_size(
    height: u64,
    width: u64,
    process_resolution: u64,
    method: NativeDepthAnything3ResizeMethod,
    patch_size: u64,
) -> Result<(u64, u64), NativeDepthAnything3Error> {
    if height == 0 || width == 0 || patch_size == 0 {
        return Err(NativeDepthAnything3Error::InvalidImage(
            "image and patch dimensions must be nonzero".to_owned(),
        ));
    }
    let reference = match method {
        NativeDepthAnything3ResizeMethod::UpperBound => height.max(width),
        NativeDepthAnything3ResizeMethod::LowerBound => height.min(width),
    };
    let scale = process_resolution as f64 / reference as f64;
    let scaled_height = (height as f64 * scale).round_ties_even();
    let scaled_width = (width as f64 * scale).round_ties_even();
    if !scaled_height.is_finite() || !scaled_width.is_finite() {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    }
    Ok((
        round_to_patch(scaled_height.max(1.0) as u64, patch_size)?.max(1),
        round_to_patch(scaled_width.max(1.0) as u64, patch_size)?.max(1),
    ))
}

fn round_to_patch(value: u64, patch: u64) -> Result<u64, NativeDepthAnything3Error> {
    let down = value / patch * patch;
    let up = down
        .checked_add(patch)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    Ok(if up - value <= value - down { up } else { down })
}

fn preflight_invocation_memory(
    resource: &NativeDepthAnything3Resource,
    flat_batch: u64,
    views: u64,
    height: u64,
    width: u64,
    original_height: u64,
    original_width: u64,
    context: &ExecutionContext<'_>,
) -> Result<(), NativeDepthAnything3Error> {
    context.check()?;
    let required = invocation_memory_required(
        resource,
        flat_batch,
        views,
        height,
        width,
        original_height,
        original_width,
    )?;
    if required > resource.memory_budget_bytes {
        return Err(NativeDepthAnything3Error::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    context.check()?;
    Ok(())
}

fn invocation_memory_required(
    resource: &NativeDepthAnything3Resource,
    flat_batch: u64,
    views: u64,
    height: u64,
    width: u64,
    original_height: u64,
    original_width: u64,
) -> Result<u64, NativeDepthAnything3Error> {
    let hidden = resource.configuration.hidden_size;
    let patches = height
        .checked_div(resource.configuration.patch_size)
        .and_then(|height| {
            width
                .checked_div(resource.configuration.patch_size)
                .and_then(|width| height.checked_mul(width))
        })
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let input = flat_batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let token_phase = flat_batch
        .checked_mul(
            patches
                .checked_add(1)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
        )
        .and_then(|value| value.checked_mul(hidden))
        .and_then(|value| value.checked_mul(20))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let samples = flat_batch
        .checked_div(views)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let global_tokens = views
        .checked_mul(
            patches
                .checked_add(1)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
        )
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let attention_phase = samples
        .checked_mul(global_tokens)
        .and_then(|value| value.checked_mul(global_tokens))
        .and_then(|value| value.checked_mul(resource.configuration.attention_heads))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let features = resource.configuration.head_features;
    let patch_height = height / resource.configuration.patch_size;
    let patch_width = width / resource.configuration.patch_size;
    let stage_three_pixels = patch_height
        .checked_add(1)
        .and_then(|value| value.checked_div(2))
        .and_then(|stage_height| {
            patch_width
                .checked_add(1)
                .and_then(|value| value.checked_div(2))
                .and_then(|stage_width| stage_height.checked_mul(stage_width))
        })
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let resized_pixels = patches
        .checked_mul(21)
        .and_then(|value| value.checked_add(stage_three_pixels))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let resized_phase = flat_batch
        .checked_mul(features)
        .and_then(|value| value.checked_mul(resized_pixels))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let full_pixels = height
        .checked_mul(width)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let main_channels = features
        .checked_mul(3)
        .and_then(|value| value.checked_add(features / 2))
        .and_then(|value| value.checked_add(32))
        .and_then(|value| value.checked_add(resource.configuration.head_output_dimension))
        .and_then(|value| {
            value.checked_add(if resource.configuration.use_sky_head {
                33
            } else {
                0
            })
        })
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let main_head_phase = flat_batch
        .checked_mul(full_pixels)
        .and_then(|value| value.checked_mul(main_channels))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let dual_phase = if resource.configuration.head == DepthAnything3Head::DualDpt {
        let retained_aux = flat_batch
            .checked_mul(features)
            .and_then(|value| value.checked_mul(resized_pixels))
            .and_then(|value| value.checked_mul(4))
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        let aux_working_channels = features
            .checked_mul(3)
            .and_then(|value| value.checked_add(features / 2))
            .and_then(|value| value.checked_add(39))
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        let aux_working = flat_batch
            .checked_mul(full_pixels)
            .and_then(|value| value.checked_mul(aux_working_channels))
            .and_then(|value| value.checked_mul(4))
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        let ray_pose = flat_batch
            .checked_mul(full_pixels)
            .and_then(|value| value.checked_mul(19))
            .and_then(|value| value.checked_mul(4))
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        retained_aux
            .checked_add(aux_working)
            .and_then(|value| value.checked_add(ray_pose))
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?
    } else {
        0
    };
    let output_planes = 1_u64
        .checked_add(u64::from(resource.configuration.head_output_dimension > 1))
        .and_then(|value| value.checked_add(u64::from(resource.configuration.use_sky_head)))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let original_projection_phase = flat_batch
        .checked_mul(original_height)
        .and_then(|value| value.checked_mul(original_width))
        .and_then(|value| value.checked_mul(output_planes))
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let required = resource
        .resident_bytes
        .checked_add(input)
        .and_then(|value| value.checked_add(token_phase.checked_mul(4)?))
        .and_then(|value| value.checked_add(attention_phase))
        .and_then(|value| value.checked_add(resized_phase))
        .and_then(|value| value.checked_add(main_head_phase))
        .and_then(|value| value.checked_add(dual_phase))
        .and_then(|value| value.checked_add(original_projection_phase))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    Ok(required)
}

fn preprocess_image(
    backend: &CpuBackend,
    image: &ImageTensor,
    target_height: u64,
    target_width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    context.check()?;
    let (_, image_height, image_width, _) = image.dimensions()?;
    let resized = if (image_height, image_width) == (target_height, target_width) {
        image.tensor().clone()
    } else {
        image
            .resize(
                target_width,
                target_height,
                ResizeMode::Lanczos,
                ResizeCrop::Disabled,
                backend,
                context,
            )?
            .tensor()
            .clone()
    };
    let clipped = clip_with_context_exact_native(backend, &resized, Some(0.0), Some(1.0), context)?;
    let channels_first =
        tensor_permute_exact_native(&clipped, &[0, 3, 1, 2], context.cancellation)?;
    let mean = tensor_from_f32_with_context_exact_native(
        backend,
        &[1, 3, 1, 1],
        &[0.485, 0.456, 0.406],
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let standard_deviation = tensor_from_f32_with_context_exact_native(
        backend,
        &[1, 3, 1, 1],
        &[0.229, 0.224, 0.225],
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let centered = sub_method_with_context_exact_native(
        backend,
        &channels_first,
        ElementwiseOperand::Tensor(&mean),
        1.0,
        context,
    )?;
    Ok(div_with_context_exact_native(
        backend,
        &centered,
        ElementwiseOperand::Tensor(&standard_deviation),
        context,
    )?)
}

fn execute_backbone(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    image: &Tensor,
    batch: usize,
    views: usize,
    reference_strategy: NativeDepthAnything3ReferenceStrategy,
    camera_token: Option<&[f32]>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<crate::dino2::NativeDino2Feature>, NativeDepthAnything3Error> {
    let backbone = native_dino2_backbone(resource.configuration)?;
    let reference_strategy = match reference_strategy {
        NativeDepthAnything3ReferenceStrategy::First => NativeDino2ReferenceStrategy::First,
        NativeDepthAnything3ReferenceStrategy::Middle => NativeDino2ReferenceStrategy::Middle,
        NativeDepthAnything3ReferenceStrategy::SaddleBalanced => {
            NativeDino2ReferenceStrategy::SaddleBalanced
        }
        NativeDepthAnything3ReferenceStrategy::SaddleSimRange => {
            NativeDino2ReferenceStrategy::SaddleSimRange
        }
    };
    Ok(backbone
        .bind_parent_preflighted(
            &resource.execution_state,
            resource.memory_budget_bytes,
            resource.resident_bytes,
        )
        .get_intermediate_layers_da3(
            backend,
            image,
            batch,
            views,
            reference_strategy,
            camera_token,
            context,
        )?)
}

fn convolution(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    stride: usize,
    padding: usize,
    transposed: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let configuration = ConvolutionConfiguration {
        stride: vec![stride, stride],
        padding: vec![padding, padding],
        dilation: vec![1, 1],
        groups: 1,
        output_padding: vec![0, 0],
    };
    let weight = resource.execution_tensor(&format!("{prefix}.weight"))?;
    let bias = resource.execution_state.get(&format!("{prefix}.bias"));
    Ok(if transposed {
        conv_transpose_2d_tensor_with_context_exact_native(
            backend,
            input,
            weight,
            bias,
            &configuration,
            context,
        )?
    } else {
        conv_2d_tensor_with_context_exact_native(
            backend,
            input,
            weight,
            bias,
            &configuration,
            context,
        )?
    })
}

fn layer_norm_values(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    prefix: &str,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    let normalized = *shape
        .last()
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let weight = tensor_values(
        resource.execution_tensor(&format!("{prefix}.weight"))?,
        context.cancellation,
    )?;
    let bias = tensor_values(
        resource.execution_tensor(&format!("{prefix}.bias"))?,
        context.cancellation,
    )?;
    Ok(layer_norm_with_context_exact_native(
        backend,
        input,
        shape,
        &[normalized],
        Some(&weight),
        Some(&bias),
        epsilon,
        DeviceId::CPU,
        context,
    )?)
}

fn linear_values(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    let weight_tensor = resource.execution_tensor(&format!("{prefix}.weight"))?;
    let weight = tensor_values(weight_tensor, context.cancellation)?;
    let weight_shape = shape_usize(weight_tensor)?;
    let bias = resource
        .execution_state
        .get(&format!("{prefix}.bias"))
        .map(|tensor| tensor_values(tensor, context.cancellation))
        .transpose()?;
    Ok(linear_with_context_exact_native(
        backend,
        input,
        input_shape,
        &weight,
        &weight_shape,
        bias.as_deref(),
        DeviceId::CPU,
        context,
    )?
    .values)
}

fn multiply_by_state(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    key: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    let input_tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(shape)?,
        input,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let output = mul_method_with_context_exact_native(
        backend,
        &input_tensor,
        resource.execution_tensor(key)?,
        context,
    )?;
    Ok(tensor_to_f32_with_context_exact_native(
        backend, &output, context,
    )?)
}

fn add_value_slices(
    backend: &CpuBackend,
    left: &[f32],
    right: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    if left.len() != right.len() {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    }
    let shape = [u64::try_from(left.len()).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?];
    let left = tensor_from_f32_with_context_exact_native(
        backend,
        &shape,
        left,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let right = tensor_from_f32_with_context_exact_native(
        backend,
        &shape,
        right,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let output = add_method_with_context_exact_native(
        backend,
        &left,
        ElementwiseOperand::Tensor(&right),
        1.0,
        context,
    )?;
    Ok(tensor_to_f32_with_context_exact_native(
        backend, &output, context,
    )?)
}

fn execute_depth_head(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    features: &[crate::dino2::NativeDino2Feature],
    batch: usize,
    views: usize,
    height: usize,
    width: usize,
    use_ray_pose: bool,
    ransac_seed: u64,
    context: &ExecutionContext<'_>,
    mut trace: Option<&mut DepthAnything3ExecutionTrace>,
) -> Result<DepthHeadOutput, NativeDepthAnything3Error> {
    if features.len() != 4 {
        return Err(NativeDepthAnything3Error::UnsupportedArchitecture);
    }
    let patch = usize_from(resource.configuration.patch_size)?;
    let patch_height = height / patch;
    let patch_width = width / patch;
    let flat_batch = batch * views;
    let mut resized = Vec::new();
    resized
        .try_reserve_exact(4)
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for (stage, feature) in features.iter().enumerate() {
        context.check()?;
        if feature.patches != patch_height * patch_width {
            return Err(NativeDepthAnything3Error::ShapeOverflow);
        }
        let values = if resource.configuration.head == DepthAnything3Head::DualDpt {
            layer_norm_values(
                resource,
                backend,
                &feature.patch_values,
                &[flat_batch, feature.patches, feature.channels],
                "native.head.norm",
                1.0e-5,
                context,
            )?
        } else {
            feature.patch_values.clone()
        };
        let tensor = tokens_to_nchw(
            backend,
            &values,
            flat_batch,
            patch_height,
            patch_width,
            feature.channels,
            context,
        )?;
        let mut tensor = convolution(
            resource,
            backend,
            &tensor,
            &format!("native.head.projects.{stage}"),
            1,
            0,
            false,
            context,
        )?;
        if resource.configuration.head == DepthAnything3Head::DualDpt {
            tensor = add_stateless_position_embedding(backend, &tensor, width, height, context)?;
        }
        tensor = match stage {
            0 => convolution(
                resource,
                backend,
                &tensor,
                "native.head.resize_layers.0",
                4,
                0,
                true,
                context,
            )?,
            1 => convolution(
                resource,
                backend,
                &tensor,
                "native.head.resize_layers.1",
                2,
                0,
                true,
                context,
            )?,
            2 => tensor,
            3 => convolution(
                resource,
                backend,
                &tensor,
                "native.head.resize_layers.3",
                2,
                1,
                false,
                context,
            )?,
            _ => return Err(NativeDepthAnything3Error::ShapeOverflow),
        };
        resized.push(convolution(
            resource,
            backend,
            &tensor,
            &format!("native.head.scratch.layer{}_rn", stage + 1),
            1,
            1,
            false,
            context,
        )?);
    }
    if let Some(trace) = trace.as_deref_mut()
        && trace.stop_before_ray_pose
    {
        trace.auxiliary_resized = resized.clone();
    }
    let mut main = fusion_block(
        resource,
        backend,
        &resized[3],
        None,
        "native.head.scratch.refinenet4",
        Some(spatial_size(&resized[2])?),
        context,
    )?;
    let mut auxiliary = if use_ray_pose {
        let first = fusion_block(
            resource,
            backend,
            &resized[3],
            None,
            "native.head.scratch.refinenet4_aux",
            Some(spatial_size(&resized[2])?),
            context,
        )?;
        if let Some(trace) = trace.as_deref_mut()
            && trace.stop_before_ray_pose
        {
            trace.auxiliary_fusion.push(first.clone());
        }
        Some(vec![first])
    } else {
        None
    };
    for stage in (0..3).rev() {
        main = fusion_block(
            resource,
            backend,
            &main,
            Some(&resized[stage]),
            &format!("native.head.scratch.refinenet{}", stage + 1),
            if stage == 0 {
                None
            } else {
                Some(spatial_size(&resized[stage - 1])?)
            },
            context,
        )?;
        if let Some(auxiliary) = auxiliary.as_mut() {
            let next = fusion_block(
                resource,
                backend,
                auxiliary
                    .last()
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
                Some(&resized[stage]),
                &format!("native.head.scratch.refinenet{}_aux", stage + 1),
                if stage == 0 {
                    None
                } else {
                    Some(spatial_size(&resized[stage - 1])?)
                },
                context,
            )?;
            if let Some(trace) = trace.as_deref_mut()
                && trace.stop_before_ray_pose
            {
                trace.auxiliary_fusion.push(next.clone());
            }
            auxiliary.push(next);
        }
    }
    let main = convolution(
        resource,
        backend,
        &main,
        "native.head.scratch.output_conv1",
        1,
        1,
        false,
        context,
    )?;
    let mut main = resize_nchw(
        backend,
        &main,
        height,
        width,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    if resource.configuration.head == DepthAnything3Head::DualDpt {
        main = add_stateless_position_embedding(backend, &main, width, height, context)?;
    }
    let fused = main;
    let main = convolution(
        resource,
        backend,
        &fused,
        "native.head.scratch.output_conv2.0",
        1,
        1,
        false,
        context,
    )?;
    let main = relu_tensor(backend, &main, context)?;
    let main_logits = convolution(
        resource,
        backend,
        &main,
        "native.head.scratch.output_conv2.2",
        1,
        0,
        false,
        context,
    )?;
    let (depth, confidence) = depth_and_confidence(
        backend,
        &main_logits,
        resource.configuration.head_output_dimension,
        context,
    )?;
    let sky = if resource.configuration.use_sky_head {
        let sky = convolution(
            resource,
            backend,
            &fused,
            "native.head.scratch.sky_output_conv2.0",
            1,
            1,
            false,
            context,
        )?;
        let sky = relu_tensor(backend, &sky, context)?;
        Some(relu_tensor(
            backend,
            &convolution(
                resource,
                backend,
                &sky,
                "native.head.scratch.sky_output_conv2.2",
                1,
                0,
                false,
                context,
            )?,
            context,
        )?)
    } else {
        None
    };
    let (ray, ray_confidence) = if let Some(auxiliary) = auxiliary {
        let mut processed = Vec::new();
        processed
            .try_reserve_exact(auxiliary.len())
            .map_err(|_| NativeDepthAnything3Error::Allocation)?;
        for (level, tensor) in auxiliary.into_iter().enumerate() {
            let mut tensor = tensor;
            for convolution_index in 0..5 {
                tensor = convolution(
                    resource,
                    backend,
                    &tensor,
                    &format!("native.head.scratch.output_conv1_aux.{level}.{convolution_index}"),
                    1,
                    1,
                    false,
                    context,
                )?;
                if level == 3
                    && let Some(trace) = trace.as_deref_mut()
                    && trace.stop_before_ray_pose
                {
                    trace.auxiliary_selected_convolutions.push(tensor.clone());
                }
            }
            if let Some(trace) = trace.as_deref_mut()
                && trace.stop_before_ray_pose
            {
                trace.auxiliary_processed.push(tensor.clone());
            }
            processed.push(tensor);
        }
        let mut last = processed
            .pop()
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        let capture_position_lane = trace
            .as_deref()
            .is_some_and(|trace| trace.stop_before_ray_pose);
        let mut position_lane = DepthAnything3PositionLaneTrace::default();
        last = if capture_position_lane {
            add_stateless_position_embedding_internal(
                backend,
                &last,
                width,
                height,
                context,
                Some(&mut position_lane),
            )?
        } else {
            add_stateless_position_embedding(backend, &last, width, height, context)?
        };
        if let Some(trace) = trace.as_deref_mut()
            && trace.stop_before_ray_pose
        {
            trace.auxiliary_position_lane = Some(position_lane);
            trace.auxiliary_positioned = Some(last.clone());
        }
        let last = convolution(
            resource,
            backend,
            &last,
            "native.head.scratch.output_conv2_aux.3.0",
            1,
            1,
            false,
            context,
        )?;
        if let Some(trace) = trace.as_deref_mut()
            && trace.stop_before_ray_pose
        {
            trace.auxiliary_convolution = Some(last.clone());
        }
        let last = layer_norm_nchw(
            resource,
            backend,
            &last,
            "native.head.scratch.output_conv2_aux.3.2",
            context,
        )?;
        if let Some(trace) = trace.as_deref_mut()
            && trace.stop_before_ray_pose
        {
            trace.auxiliary_normalized = Some(last.clone());
        }
        let last = relu_tensor(backend, &last, context)?;
        if let Some(trace) = trace.as_deref_mut()
            && trace.stop_before_ray_pose
        {
            trace.auxiliary_relu = Some(last.clone());
        }
        let last = convolution(
            resource,
            backend,
            &last,
            "native.head.scratch.output_conv2_aux.3.5",
            1,
            0,
            false,
            context,
        )?;
        if let Some(trace) = trace.as_deref_mut()
            && trace.stop_before_ray_pose
        {
            trace.auxiliary_logits = Some(last.clone());
        }
        split_ray_output(backend, &last, batch, views, context)?
    } else {
        (None, None)
    };
    let (extrinsics, intrinsics) =
        if let (Some(ray), Some(ray_confidence)) = (ray.as_ref(), ray_confidence.as_ref()) {
            if let Some(trace) = trace.as_deref_mut() {
                trace.raw_ray = Some(ray.clone());
                trace.raw_ray_confidence = Some(ray_confidence.clone());
            }
            if trace
                .as_deref()
                .is_some_and(|trace| trace.stop_before_ray_pose)
            {
                (None, None)
            } else {
                ray_pose_geometry(
                    backend,
                    ray,
                    ray_confidence,
                    batch,
                    views,
                    ransac_seed,
                    context,
                    trace,
                )?
            }
        } else if resource.configuration.has_camera_decoder && views > 1 {
            decode_camera_geometry(
                resource,
                backend,
                &features[3].camera_values,
                batch,
                views,
                height,
                width,
                context,
            )?
        } else {
            (None, None)
        };
    Ok(DepthHeadOutput {
        depth,
        confidence,
        sky,
        extrinsics,
        intrinsics,
    })
}

fn tokens_to_nchw(
    backend: &CpuBackend,
    values: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[batch, height, width, channels])?,
        values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    Ok(tensor_permute_exact_native(
        &tensor,
        &[0, 3, 1, 2],
        context.cancellation,
    )?)
}

fn add_stateless_position_embedding(
    backend: &CpuBackend,
    input: &Tensor,
    source_width: usize,
    source_height: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    add_stateless_position_embedding_internal(
        backend,
        input,
        source_width,
        source_height,
        context,
        None,
    )
}

fn position_trigonometry(value: f32, sine: bool) -> f32 {
    unsafe extern "C" {
        fn sinf(value: f32) -> f32;
        fn cosf(value: f32) -> f32;
    }
    let function: unsafe extern "C" fn(f32) -> f32 = if sine { sinf } else { cosf };
    // LLVM's constant-folded intrinsic can differ from the pinned platform F32 primitive.
    // The volatile function-pointer read keeps the operation at the native F32 boundary.
    let function = unsafe { std::ptr::read_volatile(&function) };
    // Both C functions accept every F32 value and return a scalar without borrowed memory.
    unsafe { function(value) }
}

fn add_stateless_position_embedding_internal(
    backend: &CpuBackend,
    input: &Tensor,
    source_width: usize,
    source_height: usize,
    context: &ExecutionContext<'_>,
    mut lane_trace: Option<&mut DepthAnything3PositionLaneTrace>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let shape = shape_usize(input)?;
    let [batch, channels, height, width] = shape.as_slice() else {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    };
    if channels % 4 != 0 || source_height == 0 {
        return Err(NativeDepthAnything3Error::UnsupportedArchitecture);
    }
    let (x_coordinates, y_coordinates) = stateless_position_coordinates(
        backend,
        source_width,
        source_height,
        *width,
        *height,
        context,
    )?;
    let mut embedding = filled_f32(batch * channels * height * width, 0.0)?;
    for batch_index in 0..*batch {
        for y in 0..*height {
            context.check()?;
            let y_position = *y_coordinates
                .get(y)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
            for x in 0..*width {
                let x_position = *x_coordinates
                    .get(x)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
                for channel in 0..*channels {
                    let axis_channels = channels / 2;
                    let (position, local) = if channel < axis_channels {
                        (x_position, channel)
                    } else {
                        (y_position, channel - axis_channels)
                    };
                    let half = axis_channels / 2;
                    let exponent_ratio = (local % half) as f32 / half as f32;
                    let positive_power = 100.0_f32.powf(exponent_ratio);
                    let reciprocal = 1.0 / positive_power;
                    let negative_power = 100.0_f32.powf(-exponent_ratio);
                    let frequency = reciprocal;
                    let phase = position * frequency;
                    let (selected_trig, trig) = if local < half {
                        ("sin", position_trigonometry(phase, true))
                    } else {
                        ("cos", position_trigonometry(phase, false))
                    };
                    let value = trig * 0.1;
                    embedding[((batch_index * channels + channel) * height + y) * width + x] =
                        value;
                    if batch_index == 0
                        && channel == 0
                        && y == 0
                        && x == 0
                        && let Some(trace) = lane_trace.as_deref_mut()
                    {
                        trace.x_coordinate_bits =
                            x_coordinates.iter().map(|value| value.to_bits()).collect();
                        trace.y_coordinate_bits =
                            y_coordinates.iter().map(|value| value.to_bits()).collect();
                        trace.batch = batch_index;
                        trace.channel = channel;
                        trace.y = y;
                        trace.x = x;
                        trace.exponent_ratio_bits = exponent_ratio.to_bits();
                        trace.positive_power_bits = positive_power.to_bits();
                        trace.reciprocal_bits = reciprocal.to_bits();
                        trace.negative_power_bits = negative_power.to_bits();
                        trace.frequency_bits = frequency.to_bits();
                        let discriminator_ratio = 1.0_f32 / 3.0;
                        let discriminator_positive_power = 100.0_f32.powf(discriminator_ratio);
                        trace.discriminator_ratio_bits = discriminator_ratio.to_bits();
                        trace.discriminator_positive_power_bits =
                            discriminator_positive_power.to_bits();
                        trace.discriminator_reciprocal_bits =
                            (1.0_f32 / discriminator_positive_power).to_bits();
                        trace.discriminator_negative_power_bits =
                            100.0_f32.powf(-discriminator_ratio).to_bits();
                        trace.position_bits = position.to_bits();
                        trace.phase_bits = phase.to_bits();
                        trace.sine_bits = position_trigonometry(phase, true).to_bits();
                        trace.cosine_bits = position_trigonometry(phase, false).to_bits();
                        trace.selected_trig = selected_trig;
                        trace.selected_trig_bits = trig.to_bits();
                        trace.ratio_bits = 0.1_f32.to_bits();
                        trace.embedding_bits = value.to_bits();
                    }
                }
            }
        }
    }
    let embedding = tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &embedding,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let output = add_method_with_context_exact_native(
        backend,
        input,
        ElementwiseOperand::Tensor(&embedding),
        1.0,
        context,
    )?;
    if let Some(trace) = lane_trace {
        trace.input_bits = tensor_to_f32_with_context_exact_native(backend, input, context)?
            .first()
            .copied()
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?
            .to_bits();
        trace.output_bits = tensor_to_f32_with_context_exact_native(backend, &output, context)?
            .first()
            .copied()
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?
            .to_bits();
    }
    Ok(output)
}

fn stateless_position_coordinates(
    backend: &CpuBackend,
    source_width: usize,
    source_height: usize,
    width: usize,
    height: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<f32>, Vec<f32>), NativeDepthAnything3Error> {
    let aspect = source_width as f64 / source_height as f64;
    let diagonal = (aspect * aspect + 1.0).sqrt();
    let span_x = aspect / diagonal;
    let span_y = 1.0 / diagonal;
    let left = -span_x * (width - 1) as f64 / width as f64;
    let right = span_x * (width - 1) as f64 / width as f64;
    let top = -span_y * (height - 1) as f64 / height as f64;
    let bottom = span_y * (height - 1) as f64 / height as f64;
    let x_coordinates = tensor_values(
        &linspace_with_context_exact_native(
            backend,
            left,
            right,
            u64::try_from(width).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )?,
        context.cancellation,
    )?;
    let y_coordinates = tensor_values(
        &linspace_with_context_exact_native(
            backend,
            top,
            bottom,
            u64::try_from(height).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )?,
        context.cancellation,
    )?;
    Ok((x_coordinates, y_coordinates))
}

fn fusion_block(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &Tensor,
    residual: Option<&Tensor>,
    prefix: &str,
    target: Option<(usize, usize)>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let mut output = input.clone();
    if let Some(residual) = residual {
        let residual = residual_unit(
            resource,
            backend,
            residual,
            &format!("{prefix}.resConfUnit1"),
            context,
        )?;
        output = add_method_with_context_exact_native(
            backend,
            &output,
            ElementwiseOperand::Tensor(&residual),
            1.0,
            context,
        )?;
    }
    output = residual_unit(
        resource,
        backend,
        &output,
        &format!("{prefix}.resConfUnit2"),
        context,
    )?;
    let (height, width) = match target {
        Some(target) => target,
        None => {
            let (_, _, height, width) = shape4(&output)?;
            (
                height
                    .checked_mul(2)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
                width
                    .checked_mul(2)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
            )
        }
    };
    let output = resize_nchw(
        backend,
        &output,
        height,
        width,
        InterpolateMode::Bilinear,
        Some(true),
        false,
        context,
    )?;
    convolution(
        resource,
        backend,
        &output,
        &format!("{prefix}.out_conv"),
        1,
        0,
        false,
        context,
    )
}

fn residual_unit(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let output = relu_tensor(backend, input, context)?;
    let output = convolution(
        resource,
        backend,
        &output,
        &format!("{prefix}.conv1"),
        1,
        1,
        false,
        context,
    )?;
    let output = relu_tensor(backend, &output, context)?;
    let output = convolution(
        resource,
        backend,
        &output,
        &format!("{prefix}.conv2"),
        1,
        1,
        false,
        context,
    )?;
    Ok(add_method_with_context_exact_native(
        backend,
        &output,
        ElementwiseOperand::Tensor(input),
        1.0,
        context,
    )?)
}

fn resize_nchw(
    backend: &CpuBackend,
    input: &Tensor,
    height: usize,
    width: usize,
    mode: InterpolateMode,
    align_corners: Option<bool>,
    antialias: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    Ok(interpolate_tensor_with_context_exact_native(
        backend,
        input,
        &InterpolateConfiguration {
            output_size: Some(vec![height, width]),
            scale_factor: None,
            mode,
            align_corners,
            recompute_scale_factor: None,
            antialias,
        },
        context,
    )?)
}

fn relu_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let output = relu_with_context_exact_native(backend, &values, DeviceId::CPU, context)?;
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        input.descriptor().shape(),
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )?)
}

fn depth_and_confidence(
    backend: &CpuBackend,
    logits: &Tensor,
    channels: u64,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Option<Tensor>), NativeDepthAnything3Error> {
    let (batch, actual_channels, height, width) = shape4(logits)?;
    if actual_channels != usize_from(channels)? {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    }
    let values = tensor_to_f32_with_context_exact_native(backend, logits, context)?;
    let mut depth = filled_f32(batch * height * width, 0.0)?;
    let mut confidence = (actual_channels > 1)
        .then(|| filled_f32(batch * height * width, 0.0))
        .transpose()?;
    for batch_index in 0..batch {
        context.check()?;
        for y in 0..height {
            for x in 0..width {
                let output = (batch_index * height + y) * width + x;
                depth[output] = values[((batch_index * actual_channels) * height + y) * width + x];
                if let Some(confidence) = confidence.as_mut() {
                    confidence[output] =
                        values[(((batch_index * actual_channels + actual_channels - 1) * height
                            + y)
                            * width)
                            + x];
                }
            }
        }
    }
    let descriptor = shape_u64(&[batch, 1, height, width])?;
    let depth = tensor_from_f32_with_context_exact_native(
        backend,
        &descriptor,
        &depth,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let depth = exp_with_context_exact_native(backend, &depth, context)?;
    let confidence = confidence
        .map(|confidence| {
            let confidence = tensor_from_f32_with_context_exact_native(
                backend,
                &descriptor,
                &confidence,
                DType::F32,
                DeviceId::CPU,
                context,
            )?;
            let confidence = exp_with_context_exact_native(backend, &confidence, context)?;
            Ok::<_, NativeDepthAnything3Error>(add_method_with_context_exact_native(
                backend,
                &confidence,
                ElementwiseOperand::Scalar(sim_tensor::Scalar::Float(1.0)),
                1.0,
                context,
            )?)
        })
        .transpose()?;
    Ok((depth, confidence))
}

fn layer_norm_nchw(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let (batch, channels, height, width) = shape4(input)?;
    let channels_last = tensor_permute_exact_native(input, &[0, 2, 3, 1], context.cancellation)?;
    let values = tensor_to_f32_with_context_exact_native(backend, &channels_last, context)?;
    let values = layer_norm_values(
        resource,
        backend,
        &values,
        &[batch, height, width, channels],
        prefix,
        1.0e-5,
        context,
    )?;
    let tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[batch, height, width, channels])?,
        &values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    Ok(tensor_permute_exact_native(
        &tensor,
        &[0, 3, 1, 2],
        context.cancellation,
    )?)
}

fn split_ray_output(
    backend: &CpuBackend,
    input: &Tensor,
    batch: usize,
    views: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Option<Tensor>, Option<Tensor>), NativeDepthAnything3Error> {
    let (flat_batch, channels, height, width) = shape4(input)?;
    if flat_batch != batch * views || channels != 7 {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    }
    let values = tensor_to_f32_with_context_exact_native(backend, input, context)?;
    let mut ray = filled_f32(batch * views * height * width * 6, 0.0)?;
    let mut confidence = filled_f32(batch * views * height * width, 0.0)?;
    for flat in 0..flat_batch {
        context.check()?;
        for y in 0..height {
            for x in 0..width {
                let pixel = (flat * height + y) * width + x;
                for channel in 0..6 {
                    ray[pixel * 6 + channel] =
                        values[((flat * channels + channel) * height + y) * width + x];
                }
                confidence[pixel] = values[((flat * channels + 6) * height + y) * width + x];
            }
        }
    }
    let confidence = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[batch, views, height, width])?,
        &confidence,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let confidence = exp_with_context_exact_native(backend, &confidence, context)?;
    let confidence = add_method_with_context_exact_native(
        backend,
        &confidence,
        ElementwiseOperand::Scalar(sim_tensor::Scalar::Float(1.0)),
        1.0,
        context,
    )?;
    Ok((
        Some(tensor_from_f32_with_context_exact_native(
            backend,
            &shape_u64(&[batch, views, height, width, 6])?,
            &ray,
            DType::F32,
            DeviceId::CPU,
            context,
        )?),
        Some(confidence),
    ))
}

#[allow(clippy::too_many_arguments)]
fn encode_camera_token(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    extrinsics: &Tensor,
    intrinsics: &Tensor,
    batch: usize,
    views: usize,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    if !resource.configuration.has_camera_encoder {
        return Err(NativeDepthAnything3Error::UnsupportedInvocation(
            "camera inputs require a checkpoint camera encoder".to_owned(),
        ));
    }
    let inverse_extrinsics = affine_inverse_3x4(backend, extrinsics, context)?;
    let extrinsics = tensor_values(&inverse_extrinsics, context.cancellation)?;
    let intrinsics = tensor_values(intrinsics, context.cancellation)?;
    let count = batch * views;
    let mut pose = filled_f32(count * 9, 0.0)?;
    for index in 0..count {
        context.check()?;
        let matrix = &extrinsics[index * 12..index * 12 + 12];
        let rotation = [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ];
        let translation = [matrix[3], matrix[7], matrix[11]];
        let quaternion = rotation_matrix_to_quaternion(&rotation);
        pose[index * 9..index * 9 + 3].copy_from_slice(&translation);
        pose[index * 9 + 3..index * 9 + 7].copy_from_slice(&quaternion);
        let intrinsic = &intrinsics[index * 9..index * 9 + 9];
        pose[index * 9 + 7] = 2.0 * ((height as f32 / 2.0) / intrinsic[4]).atan();
        pose[index * 9 + 8] = 2.0 * ((width as f32 / 2.0) / intrinsic[0]).atan();
    }
    let hidden = usize_from(resource.configuration.hidden_size)?;
    let mut values = linear_values(
        resource,
        backend,
        &pose,
        &[count, 9],
        "native.cam_enc.pose_branch.fc1",
        context,
    )?;
    values = gelu_with_context_exact_native(
        backend,
        &values,
        GeluApproximation::None,
        DeviceId::CPU,
        context,
    )?;
    values = linear_values(
        resource,
        backend,
        &values,
        &[count, hidden / 2],
        "native.cam_enc.pose_branch.fc2",
        context,
    )?;
    values = layer_norm_values(
        resource,
        backend,
        &values,
        &[batch, views, hidden],
        "native.cam_enc.token_norm",
        1.0e-5,
        context,
    )?;
    for block in 0..4 {
        values = camera_transformer_block(
            resource, backend, &values, batch, views, hidden, block, context,
        )?;
    }
    layer_norm_values(
        resource,
        backend,
        &values,
        &[batch, views, hidden],
        "native.cam_enc.trunk_norm",
        1.0e-5,
        context,
    )
}

fn affine_inverse_3x4(
    backend: &CpuBackend,
    affine: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDepthAnything3Error> {
    let shape = shape_usize(affine)?;
    let [batch, views, 3, 4] = shape.as_slice() else {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    };
    let count = batch
        .checked_mul(*views)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let values = tensor_values(affine, context.cancellation)?;
    let mut rotations = filled_f32(count * 9, 0.0)?;
    let mut translations = filled_f32(count * 3, 0.0)?;
    for index in 0..count {
        context.check()?;
        let matrix = values
            .get(index * 12..index * 12 + 12)
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        rotations[index * 9..index * 9 + 9].copy_from_slice(&[
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ]);
        translations[index * 3..index * 3 + 3].copy_from_slice(&[matrix[3], matrix[7], matrix[11]]);
    }
    let rotations = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[count, 3, 3])?,
        &rotations,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let inverse_rotations =
        tensor_permute_exact_native(&rotations, &[0, 2, 1], context.cancellation)?;
    let translations = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[count, 3, 1])?,
        &translations,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let inverse_translations =
        bmm_with_context_exact_native(backend, &inverse_rotations, &translations, context)?;
    let negative_one = tensor_from_f32_with_context_exact_native(
        backend,
        &[1],
        &[-1.0],
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let inverse_translations = mul_method_with_context_exact_native(
        backend,
        &inverse_translations,
        &negative_one,
        context,
    )?;
    let inverse_translation_values = tensor_values(&inverse_translations, context.cancellation)?;
    let inverse_rotation_values =
        tensor_to_f32_with_context_exact_native(backend, &inverse_rotations, context)?;
    let mut output = filled_f32(count * 12, 0.0)?;
    for index in 0..count {
        context.check()?;
        let rotation = inverse_rotation_values
            .get(index * 9..index * 9 + 9)
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        let translation = inverse_translation_values
            .get(index * 3..index * 3 + 3)
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        for row in 0..3 {
            output[index * 12 + row * 4..index * 12 + row * 4 + 3]
                .copy_from_slice(&rotation[row * 3..row * 3 + 3]);
            output[index * 12 + row * 4 + 3] = translation[row];
        }
    }
    tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[*batch, *views, 3, 4])?,
        &output,
        DType::F32,
        DeviceId::CPU,
        context,
    )
    .map_err(Into::into)
}

fn camera_transformer_block(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    input: &[f32],
    batch: usize,
    views: usize,
    hidden: usize,
    block: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    let prefix = format!("native.cam_enc.trunk.{block}");
    let normalized = layer_norm_values(
        resource,
        backend,
        input,
        &[batch, views, hidden],
        &format!("{prefix}.norm1"),
        1.0e-5,
        context,
    )?;
    let qkv = linear_values(
        resource,
        backend,
        &normalized,
        &[batch, views, hidden],
        &format!("{prefix}.attn.qkv"),
        context,
    )?;
    let heads = (hidden / 64).max(1);
    let head_dimension = hidden / heads;
    let mut query = filled_f32(batch * views * hidden, 0.0)?;
    let mut key = filled_f32(query.len(), 0.0)?;
    let mut value = filled_f32(query.len(), 0.0)?;
    for row in 0..batch * views {
        context.check()?;
        query[row * hidden..(row + 1) * hidden]
            .copy_from_slice(&qkv[row * hidden * 3..row * hidden * 3 + hidden]);
        key[row * hidden..(row + 1) * hidden]
            .copy_from_slice(&qkv[row * hidden * 3 + hidden..row * hidden * 3 + hidden * 2]);
        value[row * hidden..(row + 1) * hidden]
            .copy_from_slice(&qkv[row * hidden * 3 + hidden * 2..(row + 1) * hidden * 3]);
    }
    let attended = scaled_dot_product_attention_with_context(
        backend,
        AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch,
            query_tokens: views,
            key_tokens: views,
            heads,
            head_dimension,
            value_dimension: head_dimension,
            scale: None,
            workspace_limit_bytes: resource
                .memory_budget_bytes
                .saturating_sub(resource.resident_bytes)
                .try_into()
                .unwrap_or(usize::MAX),
        },
        &query,
        &key,
        &value,
        None,
        context,
    )?
    .values;
    let attended = linear_values(
        resource,
        backend,
        &attended,
        &[batch, views, hidden],
        &format!("{prefix}.attn.proj"),
        context,
    )?;
    let attended = multiply_by_state(
        resource,
        backend,
        &attended,
        &[batch, views, hidden],
        &format!("{prefix}.ls1.gamma"),
        context,
    )?;
    let residual = add_value_slices(backend, input, &attended, context)?;
    let normalized = layer_norm_values(
        resource,
        backend,
        &residual,
        &[batch, views, hidden],
        &format!("{prefix}.norm2"),
        1.0e-5,
        context,
    )?;
    let projected = linear_values(
        resource,
        backend,
        &normalized,
        &[batch, views, hidden],
        &format!("{prefix}.mlp.fc1"),
        context,
    )?;
    let projected = gelu_with_context_exact_native(
        backend,
        &projected,
        GeluApproximation::None,
        DeviceId::CPU,
        context,
    )?;
    let projected = linear_values(
        resource,
        backend,
        &projected,
        &[batch, views, hidden * 4],
        &format!("{prefix}.mlp.fc2"),
        context,
    )?;
    let projected = multiply_by_state(
        resource,
        backend,
        &projected,
        &[batch, views, hidden],
        &format!("{prefix}.ls2.gamma"),
        context,
    )?;
    add_value_slices(backend, &residual, &projected, context)
}

fn rotation_matrix_to_quaternion(matrix: &[f32; 9]) -> [f32; 4] {
    let candidates = [
        (1.0 + matrix[0] + matrix[4] + matrix[8]).max(0.0).sqrt(),
        (1.0 + matrix[0] - matrix[4] - matrix[8]).max(0.0).sqrt(),
        (1.0 - matrix[0] + matrix[4] - matrix[8]).max(0.0).sqrt(),
        (1.0 - matrix[0] - matrix[4] + matrix[8]).max(0.0).sqrt(),
    ];
    let mut selected = 0;
    for candidate in 1..4 {
        if candidates[candidate] > candidates[selected] {
            selected = candidate;
        }
    }
    let denominator = 2.0 * candidates[selected].max(0.1);
    let rijk = match selected {
        0 => [
            candidates[0] * candidates[0],
            matrix[7] - matrix[5],
            matrix[2] - matrix[6],
            matrix[3] - matrix[1],
        ],
        1 => [
            matrix[7] - matrix[5],
            candidates[1] * candidates[1],
            matrix[3] + matrix[1],
            matrix[2] + matrix[6],
        ],
        2 => [
            matrix[2] - matrix[6],
            matrix[3] + matrix[1],
            candidates[2] * candidates[2],
            matrix[5] + matrix[7],
        ],
        _ => [
            matrix[3] - matrix[1],
            matrix[6] + matrix[2],
            matrix[7] + matrix[5],
            candidates[3] * candidates[3],
        ],
    };
    let mut quaternion = [
        rijk[1] / denominator,
        rijk[2] / denominator,
        rijk[3] / denominator,
        rijk[0] / denominator,
    ];
    if quaternion[3] < 0.0 {
        quaternion.iter_mut().for_each(|value| *value = -*value);
    }
    quaternion
}

#[allow(clippy::too_many_arguments)]
fn decode_camera_geometry(
    resource: &NativeDepthAnything3Resource,
    backend: &CpuBackend,
    camera_features: &[f32],
    batch: usize,
    views: usize,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Option<Tensor>, Option<Tensor>), NativeDepthAnything3Error> {
    let dimension = usize_from(
        resource
            .configuration
            .camera_decoder_dimension
            .ok_or(NativeDepthAnything3Error::UnsupportedArchitecture)?,
    )?;
    let mut hidden = linear_values(
        resource,
        backend,
        camera_features,
        &[batch * views, dimension],
        "native.cam_dec.backbone.0",
        context,
    )?;
    hidden = relu_with_context_exact_native(backend, &hidden, DeviceId::CPU, context)?;
    hidden = linear_values(
        resource,
        backend,
        &hidden,
        &[batch * views, dimension],
        "native.cam_dec.backbone.2",
        context,
    )?;
    hidden = relu_with_context_exact_native(backend, &hidden, DeviceId::CPU, context)?;
    let translation = linear_values(
        resource,
        backend,
        &hidden,
        &[batch * views, dimension],
        "native.cam_dec.fc_t",
        context,
    )?;
    let quaternion = linear_values(
        resource,
        backend,
        &hidden,
        &[batch * views, dimension],
        "native.cam_dec.fc_qvec",
        context,
    )?;
    let field_of_view = linear_values(
        resource,
        backend,
        &hidden,
        &[batch * views, dimension],
        "native.cam_dec.fc_fov.0",
        context,
    )?;
    let field_of_view =
        relu_with_context_exact_native(backend, &field_of_view, DeviceId::CPU, context)?;
    pose_encoding_geometry(
        backend,
        &translation,
        &quaternion,
        &field_of_view,
        batch,
        views,
        height,
        width,
        context,
    )
}

fn ray_pose_geometry(
    backend: &CpuBackend,
    ray: &Tensor,
    confidence: &Tensor,
    batch: usize,
    views: usize,
    ransac_seed: u64,
    context: &ExecutionContext<'_>,
    mut trace: Option<&mut DepthAnything3ExecutionTrace>,
) -> Result<(Option<Tensor>, Option<Tensor>), NativeDepthAnything3Error> {
    let ray_shape = shape_usize(ray)?;
    let [ray_batch, ray_views, height, width, channels] = ray_shape.as_slice() else {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    };
    if (*ray_batch, *ray_views, *channels) != (batch, views, 6)
        || confidence.descriptor().shape() != shape_u64(&[batch, views, *height, *width])?
    {
        return Err(NativeDepthAnything3Error::ShapeOverflow);
    }
    let rays = tensor_values(ray, context.cancellation)?;
    let confidence = tensor_values(confidence, context.cancellation)?;
    let points = height
        .checked_mul(*width)
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    if points < 8 {
        return Err(NativeDepthAnything3Error::UnsupportedInvocation(
            "ray-pose geometry requires at least eight pixels".to_owned(),
        ));
    }
    let candidate_count = 8_usize.max((points as f64 * 0.3_f64) as usize);
    let (random_samples, mut transaction) =
        ray_pose_random_samples(backend, candidate_count, ransac_seed, context)?;
    if let Some(trace) = trace.as_deref_mut() {
        trace.ransac_samples = random_samples.clone();
    }
    let mut c2w = filled_f32(batch * views * 12, 0.0)?;
    let mut focal = filled_f32(batch * views * 2, 0.0)?;
    let mut principal = filled_f32(batch * views * 2, 0.0)?;
    let (horizontal, vertical) = ray_identity_coordinates(backend, *width, *height, context)?;
    for flat in 0..batch * views {
        context.check()?;
        let mut source = filled_f32(points * 2, 0.0)?;
        let mut destination = filled_f32(points * 2, 0.0)?;
        let mut weights = filled_f32(points, 0.0)?;
        for y in 0..*height {
            for x in 0..*width {
                let point = y * width + x;
                let ray_offset = (flat * points + point) * 6;
                let source_x = *horizontal
                    .get(x)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
                let source_y = *vertical
                    .get(y)
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
                source[point * 2] = source_x;
                source[point * 2 + 1] = source_y;
                let target_z = rays[ray_offset + 2];
                if target_z.abs() > 1.0e-4 {
                    destination[point * 2] = rays[ray_offset] / target_z;
                    destination[point * 2 + 1] = rays[ray_offset + 1] / target_z;
                    weights[point] = confidence[flat * points + point];
                }
            }
        }
        let mut sorted = argsort_descending_indices(backend, &weights, context)?;
        sorted.truncate(candidate_count);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_iteration = 0;
        let mut best_inliers = Vec::new();
        for (iteration, sample) in random_samples.iter().enumerate() {
            context.check()?;
            let selected = sample
                .iter()
                .map(|index| sorted[*index])
                .collect::<Vec<_>>();
            let homography =
                weighted_homography(backend, &source, &destination, &weights, &selected, context)?;
            let mut score = 0.0_f32;
            let mut inliers = Vec::new();
            for point in 0..points {
                let x = source[point * 2];
                let y = source[point * 2 + 1];
                let denominator = x * homography[6] + y * homography[7] + homography[8];
                let projected_x =
                    (x * homography[0] + y * homography[1] + homography[2]) / denominator;
                let projected_y =
                    (x * homography[3] + y * homography[4] + homography[5]) / denominator;
                let error = ((projected_x - destination[point * 2]).powi(2)
                    + (projected_y - destination[point * 2 + 1]).powi(2))
                .sqrt();
                if error < 0.2 {
                    score += weights[point];
                    inliers.push(point);
                }
            }
            if score > best_score {
                best_score = score;
                best_iteration = iteration;
                best_inliers = inliers;
            }
        }
        let fallback = best_inliers.len() < 4;
        let homography = if fallback {
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        } else {
            let inlier_weights = best_inliers
                .iter()
                .map(|index| weights[*index])
                .collect::<Vec<_>>();
            let sorted_inliers = argsort_descending_indices(backend, &inlier_weights, context)?;
            best_inliers = sorted_inliers
                .into_iter()
                .map(|index| best_inliers[index])
                .collect();
            transaction =
                refit_inliers_if_needed(backend, &mut best_inliers, transaction, context)?;
            weighted_homography(
                backend,
                &source,
                &destination,
                &weights,
                &best_inliers,
                context,
            )?
        };
        let normalized_homography = homography;
        let homography_tensor = tensor_from_f32_with_context_exact_native(
            backend,
            &[1, 3, 3],
            &homography,
            DType::F32,
            DeviceId::CPU,
            context,
        )?;
        let determinant = tensor_values(
            &determinant_with_context_exact_native(backend, &homography_tensor, context)?,
            context.cancellation,
        )?[0];
        let homography = if determinant < 0.0 {
            homography.map(|value| -value)
        } else {
            homography
        };
        let signed_homography = homography;
        let (rotation, lower) = ql_decomposition(backend, &homography, context)?;
        let scale = lower[8];
        let output = flat * 12;
        c2w[output..output + 3].copy_from_slice(&rotation[0..3]);
        c2w[output + 4..output + 7].copy_from_slice(&rotation[3..6]);
        c2w[output + 8..output + 11].copy_from_slice(&rotation[6..9]);
        let raw_confidence = confidence
            .get(flat * points..(flat + 1) * points)
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        let total_weight = raw_confidence.iter().copied().sum::<f32>();
        for axis in 0..3 {
            c2w[output + axis * 4 + 3] = (0..points)
                .map(|point| rays[(flat * points + point) * 6 + 3 + axis] * raw_confidence[point])
                .sum::<f32>()
                / total_weight;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.ransac_views.push(DepthAnything3RansacViewTrace {
                candidate_indices: sorted.clone(),
                best_iteration,
                best_inliers: best_inliers.clone(),
                best_score,
                fallback,
                normalized_homography,
                signed_homography,
                rotation,
                lower,
                c2w_pre_inverse: c2w[output..output + 12]
                    .try_into()
                    .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            });
        }
        focal[flat * 2] = 1.0 / (lower[0] / scale);
        focal[flat * 2 + 1] = 1.0 / (lower[4] / scale);
        principal[flat * 2] = lower[6] / scale + 1.0;
        principal[flat * 2 + 1] = lower[7] / scale + 1.0;
    }
    let mut intrinsics = filled_f32(batch * views * 9, 0.0)?;
    for flat in 0..batch * views {
        intrinsics[flat * 9] = focal[flat * 2] / 2.0 * *width as f32;
        intrinsics[flat * 9 + 4] = focal[flat * 2 + 1] / 2.0 * *height as f32;
        intrinsics[flat * 9 + 2] = principal[flat * 2] * *width as f32 * 0.5;
        intrinsics[flat * 9 + 5] = principal[flat * 2 + 1] * *height as f32 * 0.5;
        intrinsics[flat * 9 + 8] = 1.0;
    }
    Ok((
        Some(affine_inverse_3x4(
            backend,
            &tensor_from_f32_with_context_exact_native(
                backend,
                &shape_u64(&[batch, views, 3, 4])?,
                &c2w,
                DType::F32,
                DeviceId::CPU,
                context,
            )?,
            context,
        )?),
        Some(tensor_from_f32_with_context_exact_native(
            backend,
            &shape_u64(&[batch, views, 3, 3])?,
            &intrinsics,
            DType::F32,
            DeviceId::CPU,
            context,
        )?),
    ))
}

fn ray_pose_random_samples(
    backend: &CpuBackend,
    candidate_count: usize,
    seed: u64,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<Vec<usize>>, sim_tensor::RngTransaction), NativeDepthAnything3Error> {
    let address = context.rng_phase.ok_or_else(|| {
        NativeDepthAnything3Error::UnsupportedInvocation(
            "ray-pose execution requires a versioned RANSAC RNG phase".to_owned(),
        )
    })?;
    let stream = generator_exact_native(
        RngProfileVersion::V2,
        RngAlgorithm::Mt19937,
        seed,
        address.clone(),
        context.cancellation,
    )?;
    let mut transaction = stream
        .begin(None)
        .map_err(RandomNumberGenerationPartOneError::from)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(100)
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for _ in 0..100 {
        context.check()?;
        let generated = randperm_with_context_exact_native(
            backend,
            u64::try_from(candidate_count).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            transaction,
            context,
        )?;
        transaction = generated.transaction;
        let permutation = tensor_i64_values(&generated.tensor, context.cancellation)?;
        samples.push(
            permutation
                .get(..8)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?
                .iter()
                .map(|index| {
                    usize::try_from(*index).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok((samples, transaction))
}

fn refit_inliers_if_needed(
    backend: &CpuBackend,
    best_inliers: &mut Vec<usize>,
    transaction: sim_tensor::RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::RngTransaction, NativeDepthAnything3Error> {
    if best_inliers.len() <= 8_000 {
        return Ok(transaction);
    }
    let keep = ((best_inliers.len() as f64 * 0.95_f64) as usize).max(8_000);
    best_inliers.truncate(keep);
    let generated = randperm_with_context_exact_native(
        backend,
        u64::try_from(keep).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
        transaction,
        context,
    )?;
    let permutation = tensor_i64_values(&generated.tensor, context.cancellation)?;
    let mut sampled = Vec::new();
    sampled
        .try_reserve_exact(8_000)
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for index in permutation.iter().take(8_000) {
        sampled.push(
            *best_inliers
                .get(
                    usize::try_from(*index)
                        .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
                )
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
        );
    }
    *best_inliers = sampled;
    Ok(generated.transaction)
}

fn ray_identity_coordinates(
    backend: &CpuBackend,
    width: usize,
    height: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Vec<f32>, Vec<f32>), NativeDepthAnything3Error> {
    let horizontal_delta = 1.0_f64 / width as f64;
    let vertical_delta = 1.0_f64 / height as f64;
    let horizontal = tensor_values(
        &linspace_with_context_exact_native(
            backend,
            -(1.0 - horizontal_delta),
            1.0 - horizontal_delta,
            u64::try_from(width).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )?,
        context.cancellation,
    )?;
    let vertical = tensor_values(
        &linspace_with_context_exact_native(
            backend,
            -(1.0 - vertical_delta),
            1.0 - vertical_delta,
            u64::try_from(height).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            false,
            None,
            context,
        )?,
        context.cancellation,
    )?;
    Ok((horizontal, vertical))
}

fn argsort_descending_indices(
    backend: &CpuBackend,
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<usize>, NativeDepthAnything3Error> {
    let tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[values.len()])?,
        values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    tensor_i64_values(
        &argsort_with_context_exact_native(backend, &tensor, -1, true, false, context)?,
        context.cancellation,
    )?
    .into_iter()
    .map(|index| usize::try_from(index).map_err(|_| NativeDepthAnything3Error::ShapeOverflow))
    .collect()
}

fn weighted_homography(
    backend: &CpuBackend,
    source: &[f32],
    destination: &[f32],
    weights: &[f32],
    indices: &[usize],
    context: &ExecutionContext<'_>,
) -> Result<[f32; 9], NativeDepthAnything3Error> {
    let mut matrix = filled_f32(indices.len() * 18, 0.0)?;
    for (row, index) in indices.iter().copied().enumerate() {
        context.check()?;
        let weight = weights[index].sqrt();
        let x = source[index * 2];
        let y = source[index * 2 + 1];
        let u = destination[index * 2];
        let v = destination[index * 2 + 1];
        matrix[row * 9..row * 9 + 9].copy_from_slice(&[
            -x * weight,
            -y * weight,
            -weight,
            0.0,
            0.0,
            0.0,
            x * u * weight,
            y * u * weight,
            u * weight,
        ]);
        let second = (indices.len() + row) * 9;
        matrix[second..second + 9].copy_from_slice(&[
            0.0,
            0.0,
            0.0,
            -x * weight,
            -y * weight,
            -weight,
            x * v * weight,
            y * v * weight,
            v * weight,
        ]);
    }
    let tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[indices.len() * 2, 9])?,
        &matrix,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let decomposition = svd_with_context_exact_native(backend, &tensor, true, context)?;
    let values = tensor_values(&decomposition.vh, context.cancellation)?;
    let mut homography = [0.0_f32; 9];
    homography.copy_from_slice(
        values
            .get(values.len().saturating_sub(9)..)
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
    );
    let divisor = homography[8];
    homography.iter_mut().for_each(|value| *value /= divisor);
    Ok(homography)
}

fn ql_decomposition(
    backend: &CpuBackend,
    matrix: &[f32; 9],
    context: &ExecutionContext<'_>,
) -> Result<([f32; 9], [f32; 9]), NativeDepthAnything3Error> {
    let permutation = [0.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let matrix = tensor_from_f32_with_context_exact_native(
        backend,
        &[1, 3, 3],
        matrix,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let permutation_tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &[1, 3, 3],
        &permutation,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let transformed =
        bmm_with_context_exact_native(backend, &matrix, &permutation_tensor, context)?;
    let decomposition =
        qr_with_context_exact_native(backend, &transformed, QrMode::Reduced, context)?;
    let q_tilde = decomposition
        .q
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    let q = bmm_with_context_exact_native(backend, &q_tilde, &permutation_tensor, context)?;
    let lower = bmm_with_context_exact_native(
        backend,
        &bmm_with_context_exact_native(backend, &permutation_tensor, &decomposition.r, context)?,
        &permutation_tensor,
        context,
    )?;
    let mut q_values: [f32; 9] = tensor_values(&q, context.cancellation)?
        .try_into()
        .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?;
    let mut lower_values: [f32; 9] = tensor_values(&lower, context.cancellation)?
        .try_into()
        .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?;
    for diagonal in 0..3 {
        let diagonal_value = lower_values[diagonal * 3 + diagonal];
        let sign = if diagonal_value > 0.0 {
            1.0
        } else if diagonal_value < 0.0 {
            -1.0
        } else {
            0.0
        };
        for row in 0..3 {
            q_values[row * 3 + diagonal] *= sign;
        }
        for column in 0..3 {
            lower_values[diagonal * 3 + column] *= sign;
        }
    }
    Ok((q_values, lower_values))
}

#[allow(clippy::too_many_arguments)]
fn pose_encoding_geometry(
    backend: &CpuBackend,
    translation: &[f32],
    quaternion: &[f32],
    field_of_view: &[f32],
    batch: usize,
    views: usize,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Option<Tensor>, Option<Tensor>), NativeDepthAnything3Error> {
    let count = batch * views;
    let mut extrinsics = filled_f32(count * 12, 0.0)?;
    let mut intrinsics = filled_f32(count * 9, 0.0)?;
    for index in 0..count {
        context.check()?;
        let quaternion = &quaternion[index * 4..index * 4 + 4];
        let norm = quaternion
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1.0e-6);
        let x = quaternion[0] / norm;
        let y = quaternion[1] / norm;
        let z = quaternion[2] / norm;
        let real = quaternion[3] / norm;
        let two = 2.0 / (x * x + y * y + z * z + real * real);
        let rotation = [
            1.0 - two * (y * y + z * z),
            two * (x * y - z * real),
            two * (x * z + y * real),
            two * (x * y + z * real),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * real),
            two * (x * z - y * real),
            two * (y * z + x * real),
            1.0 - two * (x * x + y * y),
        ];
        for row in 0..3 {
            for column in 0..3 {
                extrinsics[index * 12 + row * 4 + column] = rotation[column * 3 + row];
            }
            extrinsics[index * 12 + row * 4 + 3] = -(0..3)
                .map(|column| rotation[column * 3 + row] * translation[index * 3 + column])
                .sum::<f32>();
        }
        let fov_height = field_of_view[index * 2];
        let fov_width = field_of_view[index * 2 + 1];
        let tangent_width = (fov_width / 2.0).tan();
        let tangent_height = (fov_height / 2.0).tan();
        let tangent_width = if tangent_width.is_nan() {
            tangent_width
        } else {
            tangent_width.max(1.0e-6)
        };
        let tangent_height = if tangent_height.is_nan() {
            tangent_height
        } else {
            tangent_height.max(1.0e-6)
        };
        intrinsics[index * 9] = (width as f32 / 2.0) / tangent_width;
        intrinsics[index * 9 + 4] = (height as f32 / 2.0) / tangent_height;
        intrinsics[index * 9 + 2] = width as f32 / 2.0;
        intrinsics[index * 9 + 5] = height as f32 / 2.0;
        intrinsics[index * 9 + 8] = 1.0;
    }
    Ok((
        Some(tensor_from_f32_with_context_exact_native(
            backend,
            &shape_u64(&[batch, views, 3, 4])?,
            &extrinsics,
            DType::F32,
            DeviceId::CPU,
            context,
        )?),
        Some(tensor_from_f32_with_context_exact_native(
            backend,
            &shape_u64(&[batch, views, 3, 3])?,
            &intrinsics,
            DType::F32,
            DeviceId::CPU,
            context,
        )?),
    ))
}

fn project_geometry(
    backend: &CpuBackend,
    output: DepthHeadOutput,
    flat_batch: u64,
    original_height: u64,
    original_width: u64,
    views: u64,
    context: &ExecutionContext<'_>,
) -> Result<NativeDepthAnything3Geometry, NativeDepthAnything3Error> {
    let height = usize_from(original_height)?;
    let width = usize_from(original_width)?;
    let depth = resize_nchw(
        backend,
        &output.depth,
        height,
        width,
        InterpolateMode::Bilinear,
        Some(false),
        false,
        context,
    )?;
    let depth = tensor_squeeze_exact_native(&depth, Some(&[1]), context.cancellation)?;
    let confidence = output
        .confidence
        .map(|tensor| {
            resize_nchw(
                backend,
                &tensor,
                height,
                width,
                InterpolateMode::Bilinear,
                Some(false),
                false,
                context,
            )
        })
        .transpose()?
        .map(|tensor| tensor_squeeze_exact_native(&tensor, Some(&[1]), context.cancellation))
        .transpose()?;
    let sky = output
        .sky
        .map(|tensor| {
            resize_nchw(
                backend,
                &tensor,
                height,
                width,
                InterpolateMode::Bilinear,
                Some(false),
                false,
                context,
            )
        })
        .transpose()?
        .map(|tensor| tensor_squeeze_exact_native(&tensor, Some(&[1]), context.cancellation))
        .transpose()?;
    let expected = [flat_batch, original_height, original_width];
    validate_geometry_tensor("depth", &depth, &expected, context.cancellation)?;
    if let Some(confidence) = &confidence {
        validate_geometry_tensor("confidence", confidence, &expected, context.cancellation)?;
    }
    if let Some(sky) = &sky {
        validate_geometry_tensor("sky", sky, &expected, context.cancellation)?;
    }
    let samples = flat_batch / views;
    if let Some(extrinsics) = &output.extrinsics {
        validate_geometry_tensor(
            "extrinsics",
            extrinsics,
            &[samples, views, 3, 4],
            context.cancellation,
        )?;
    }
    if let Some(intrinsics) = &output.intrinsics {
        validate_geometry_tensor(
            "intrinsics",
            intrinsics,
            &[samples, views, 3, 3],
            context.cancellation,
        )?;
    }
    context.check()?;
    Ok(NativeDepthAnything3Geometry {
        depth,
        confidence,
        sky,
        extrinsics: output.extrinsics,
        intrinsics: output.intrinsics,
        original_height,
        original_width,
        views_per_sample: views,
    })
}

fn validate_geometry_tensor(
    name: &'static str,
    tensor: &Tensor,
    expected_shape: &[u64],
    cancellation: &CancellationToken,
) -> Result<(), NativeDepthAnything3Error> {
    if tensor.descriptor().shape() != expected_shape || tensor.descriptor().dtype() != DType::F32 {
        return Err(NativeDepthAnything3Error::UnsupportedInvocation(format!(
            "{name} output has an invalid shape or dtype"
        )));
    }
    for bytes in tensor.contiguous_bytes()?.chunks(64 * 1_024) {
        cancellation.check()?;
        for value in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(
                value
                    .try_into()
                    .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
            );
            if !value.is_finite() {
                return Err(NativeDepthAnything3Error::UnsupportedInvocation(format!(
                    "{name} output contains non-finite values"
                )));
            }
        }
    }
    Ok(())
}

fn tensor_values(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(NativeDepthAnything3Error::SemanticStateChanged);
    }
    let bytes = tensor.contiguous_bytes()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        values.push(f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
        ));
    }
    Ok(values)
}

fn tensor_i64_values(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<i64>, NativeDepthAnything3Error> {
    if tensor.descriptor().dtype() != DType::I64 {
        return Err(NativeDepthAnything3Error::SemanticStateChanged);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(tensor.contiguous_bytes()?.len() / 8)
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for (index, bytes) in tensor.contiguous_bytes()?.chunks_exact(8).enumerate() {
        if index.is_multiple_of(8_192) {
            cancellation.check()?;
        }
        values.push(i64::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?,
        ));
    }
    Ok(values)
}

fn shape_usize(tensor: &Tensor) -> Result<Vec<usize>, NativeDepthAnything3Error> {
    tensor
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| usize_from(*dimension))
        .collect()
}

fn shape_u64(shape: &[usize]) -> Result<Vec<u64>, NativeDepthAnything3Error> {
    shape
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
        })
        .collect()
}

fn shape4(tensor: &Tensor) -> Result<(usize, usize, usize, usize), NativeDepthAnything3Error> {
    let shape = shape_usize(tensor)?;
    match shape.as_slice() {
        [batch, channels, height, width] => Ok((*batch, *channels, *height, *width)),
        _ => Err(NativeDepthAnything3Error::ShapeOverflow),
    }
}

fn spatial_size(tensor: &Tensor) -> Result<(usize, usize), NativeDepthAnything3Error> {
    let (_, _, height, width) = shape4(tensor)?;
    Ok((height, width))
}

fn filled_f32(length: usize, value: f32) -> Result<Vec<f32>, NativeDepthAnything3Error> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    output.resize(length, value);
    Ok(output)
}

fn configuration_from_native_state(
    state: &BTreeMap<String, Tensor>,
) -> Result<DepthAnything3Configuration, NativeDepthAnything3Error> {
    let mut tensor_shapes = BTreeMap::new();
    for (key, tensor) in state {
        let Some(source_key) = key.strip_prefix("native.") else {
            return Err(NativeDepthAnything3Error::UnexpectedState(key.clone()));
        };
        tensor_shapes.insert(
            format!("model.diffusion_model.{source_key}"),
            tensor.descriptor().shape().to_vec(),
        );
    }
    configuration_for_probe(&ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    })
    .map_err(|_| NativeDepthAnything3Error::UnsupportedArchitecture)
}

fn depth_anything_3_state_manifest(
    profile: DepthAnything3ExecutionProfile,
) -> Result<Vec<StateSpecification>, NativeDepthAnything3Error> {
    if profile.source_exact {
        DepthAnything3ExecutionProfile::production(profile.configuration)?;
    }
    let configuration = &profile.configuration;
    let backbone = native_dino2_backbone(*configuration)?;
    let mut states = backbone
        .state_manifest()?
        .into_iter()
        .map(|state| StateSpecification {
            key: state.key,
            shape: state.shape,
        })
        .collect::<Vec<_>>();
    add_head_manifest(&mut states, configuration)?;
    if configuration.has_camera_encoder {
        add_camera_encoder_manifest(&mut states, configuration)?;
    }
    if configuration.has_camera_decoder {
        add_camera_decoder_manifest(&mut states, configuration)?;
    }
    Ok(states)
}

fn native_dino2_backbone(
    configuration: DepthAnything3Configuration,
) -> Result<NativeDino2Backbone, NativeDepthAnything3Error> {
    native_dino2_backbone_with_mask_token(configuration, false)
}

fn native_dino2_backbone_with_mask_token(
    configuration: DepthAnything3Configuration,
    use_mask_token: bool,
) -> Result<NativeDino2Backbone, NativeDepthAnything3Error> {
    let [output0, output1, output2, output3] = configuration.output_layers;
    Ok(NativeDino2Backbone::new(NativeDino2Configuration {
        prefix: "native.backbone",
        hidden: usize_from(configuration.hidden_size)?,
        layer_count: configuration.layer_count,
        attention_heads: usize_from(configuration.attention_heads)?,
        patch: usize_from(configuration.patch_size)?,
        image: usize_from(configuration.image_size)?,
        qknorm_start: configuration.qknorm_start,
        alternate_attention_start: configuration.alternate_attention_start,
        rope_start: configuration.rope_start,
        concatenate_camera_token: configuration.concatenate_camera_token,
        use_mask_token,
        swiglu: configuration.backbone == DepthAnything3Backbone::VitGiant,
        output_layers: [output0, output1, output2, output3],
    })?)
}

fn add_head_manifest(
    states: &mut Vec<StateSpecification>,
    configuration: &DepthAnything3Configuration,
) -> Result<(), NativeDepthAnything3Error> {
    let input = usize_from(configuration.head_dimension)?;
    let features = usize_from(configuration.head_features)?;
    let outputs = configuration
        .head_out_channels
        .map(usize_from)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    if outputs.len() != 4 || features == 0 {
        return Err(NativeDepthAnything3Error::UnsupportedArchitecture);
    }
    if configuration.head == DepthAnything3Head::DualDpt {
        add_affine(states, "native.head.norm", input)?;
    }
    for (index, channels) in outputs.iter().copied().enumerate() {
        add_conv(
            states,
            &format!("native.head.projects.{index}"),
            channels,
            input,
            1,
            true,
        )?;
    }
    add_conv_transpose(
        states,
        "native.head.resize_layers.0",
        outputs[0],
        outputs[0],
        4,
        true,
    )?;
    add_conv_transpose(
        states,
        "native.head.resize_layers.1",
        outputs[1],
        outputs[1],
        2,
        true,
    )?;
    add_conv(
        states,
        "native.head.resize_layers.3",
        outputs[3],
        outputs[3],
        3,
        true,
    )?;
    for (index, channels) in outputs.iter().copied().enumerate() {
        add_conv(
            states,
            &format!("native.head.scratch.layer{}_rn", index + 1),
            features,
            channels,
            3,
            false,
        )?;
    }
    for index in 1..=4 {
        add_refine_net(
            states,
            &format!("native.head.scratch.refinenet{index}"),
            features,
            index != 4,
        )?;
    }
    add_conv(
        states,
        "native.head.scratch.output_conv1",
        features / 2,
        features,
        3,
        true,
    )?;
    add_conv(
        states,
        "native.head.scratch.output_conv2.0",
        32,
        features / 2,
        3,
        true,
    )?;
    add_conv(
        states,
        "native.head.scratch.output_conv2.2",
        usize_from(configuration.head_output_dimension)?,
        32,
        1,
        true,
    )?;
    if configuration.use_sky_head {
        add_conv(
            states,
            "native.head.scratch.sky_output_conv2.0",
            32,
            features / 2,
            3,
            true,
        )?;
        add_conv(
            states,
            "native.head.scratch.sky_output_conv2.2",
            1,
            32,
            1,
            true,
        )?;
    }
    if configuration.head == DepthAnything3Head::DualDpt {
        for index in 1..=4 {
            add_refine_net(
                states,
                &format!("native.head.scratch.refinenet{index}_aux"),
                features,
                index != 4,
            )?;
        }
        for level in 0..4 {
            let prefix = format!("native.head.scratch.output_conv1_aux.{level}");
            for (index, (output, input)) in [
                (features / 2, features),
                (features, features / 2),
                (features / 2, features),
                (features, features / 2),
                (features / 2, features),
            ]
            .into_iter()
            .enumerate()
            {
                add_conv(states, &format!("{prefix}.{index}"), output, input, 3, true)?;
            }
            let prefix = format!("native.head.scratch.output_conv2_aux.{level}");
            add_conv(states, &format!("{prefix}.0"), 32, features / 2, 3, true)?;
            add_affine(states, &format!("{prefix}.2"), 32)?;
            add_conv(states, &format!("{prefix}.5"), 7, 32, 1, true)?;
        }
    }
    Ok(())
}

fn add_refine_net(
    states: &mut Vec<StateSpecification>,
    prefix: &str,
    features: usize,
    residual: bool,
) -> Result<(), NativeDepthAnything3Error> {
    if residual {
        for convolution in ["conv1", "conv2"] {
            add_conv(
                states,
                &format!("{prefix}.resConfUnit1.{convolution}"),
                features,
                features,
                3,
                true,
            )?;
        }
    }
    for convolution in ["conv1", "conv2"] {
        add_conv(
            states,
            &format!("{prefix}.resConfUnit2.{convolution}"),
            features,
            features,
            3,
            true,
        )?;
    }
    add_conv(
        states,
        &format!("{prefix}.out_conv"),
        features,
        features,
        1,
        true,
    )
}

fn add_camera_encoder_manifest(
    states: &mut Vec<StateSpecification>,
    configuration: &DepthAnything3Configuration,
) -> Result<(), NativeDepthAnything3Error> {
    let hidden = usize_from(configuration.hidden_size)?;
    add_affine(states, "native.cam_enc.token_norm", hidden)?;
    add_affine(states, "native.cam_enc.trunk_norm", hidden)?;
    add_linear(
        states,
        "native.cam_enc.pose_branch.fc1",
        hidden / 2,
        9,
        true,
    )?;
    add_linear(
        states,
        "native.cam_enc.pose_branch.fc2",
        hidden,
        hidden / 2,
        true,
    )?;
    for block in 0..4 {
        let prefix = format!("native.cam_enc.trunk.{block}");
        add_affine(states, &format!("{prefix}.norm1"), hidden)?;
        add_linear(
            states,
            &format!("{prefix}.attn.qkv"),
            hidden * 3,
            hidden,
            true,
        )?;
        add_linear(states, &format!("{prefix}.attn.proj"), hidden, hidden, true)?;
        add_state(states, &format!("{prefix}.ls1.gamma"), &[hidden])?;
        add_affine(states, &format!("{prefix}.norm2"), hidden)?;
        add_linear(
            states,
            &format!("{prefix}.mlp.fc1"),
            hidden * 4,
            hidden,
            true,
        )?;
        add_linear(
            states,
            &format!("{prefix}.mlp.fc2"),
            hidden,
            hidden * 4,
            true,
        )?;
        add_state(states, &format!("{prefix}.ls2.gamma"), &[hidden])?;
    }
    Ok(())
}

fn add_camera_decoder_manifest(
    states: &mut Vec<StateSpecification>,
    configuration: &DepthAnything3Configuration,
) -> Result<(), NativeDepthAnything3Error> {
    let dimension = usize_from(
        configuration
            .camera_decoder_dimension
            .ok_or(NativeDepthAnything3Error::UnsupportedArchitecture)?,
    )?;
    add_linear(
        states,
        "native.cam_dec.backbone.0",
        dimension,
        dimension,
        true,
    )?;
    add_linear(
        states,
        "native.cam_dec.backbone.2",
        dimension,
        dimension,
        true,
    )?;
    add_linear(states, "native.cam_dec.fc_t", 3, dimension, true)?;
    add_linear(states, "native.cam_dec.fc_qvec", 4, dimension, true)?;
    add_linear(states, "native.cam_dec.fc_fov.0", 2, dimension, true)
}

fn add_conv(
    states: &mut Vec<StateSpecification>,
    prefix: &str,
    output: usize,
    input: usize,
    kernel: usize,
    bias: bool,
) -> Result<(), NativeDepthAnything3Error> {
    add_state(
        states,
        &format!("{prefix}.weight"),
        &[output, input, kernel, kernel],
    )?;
    if bias {
        add_state(states, &format!("{prefix}.bias"), &[output])?;
    }
    Ok(())
}

fn add_conv_transpose(
    states: &mut Vec<StateSpecification>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    bias: bool,
) -> Result<(), NativeDepthAnything3Error> {
    add_state(
        states,
        &format!("{prefix}.weight"),
        &[input, output, kernel, kernel],
    )?;
    if bias {
        add_state(states, &format!("{prefix}.bias"), &[output])?;
    }
    Ok(())
}

fn add_linear(
    states: &mut Vec<StateSpecification>,
    prefix: &str,
    output: usize,
    input: usize,
    bias: bool,
) -> Result<(), NativeDepthAnything3Error> {
    add_state(states, &format!("{prefix}.weight"), &[output, input])?;
    if bias {
        add_state(states, &format!("{prefix}.bias"), &[output])?;
    }
    Ok(())
}

fn add_affine(
    states: &mut Vec<StateSpecification>,
    prefix: &str,
    features: usize,
) -> Result<(), NativeDepthAnything3Error> {
    add_state(states, &format!("{prefix}.weight"), &[features])?;
    add_state(states, &format!("{prefix}.bias"), &[features])
}

fn add_state(
    states: &mut Vec<StateSpecification>,
    key: &str,
    shape: &[usize],
) -> Result<(), NativeDepthAnything3Error> {
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(shape.len())
        .map_err(|_| NativeDepthAnything3Error::Allocation)?;
    for dimension in shape {
        converted
            .push(u64::try_from(*dimension).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)?);
    }
    states.push(StateSpecification {
        key: key.to_owned(),
        shape: converted,
    });
    Ok(())
}

fn validate_strict_source_state(
    state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<DType, NativeDepthAnything3Error> {
    if state.len() != specifications.len() {
        let expected = specifications
            .iter()
            .map(|specification| specification.key.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(key) = state.keys().find(|key| !expected.contains(key.as_str())) {
            return Err(NativeDepthAnything3Error::UnexpectedState(key.clone()));
        }
        if let Some(specification) = specifications
            .iter()
            .find(|specification| !state.contains_key(&specification.key))
        {
            return Err(NativeDepthAnything3Error::MissingState(
                specification.key.clone(),
            ));
        }
    }
    let mut dtype = None;
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        let tensor = state
            .get(&specification.key)
            .ok_or_else(|| NativeDepthAnything3Error::MissingState(specification.key.clone()))?;
        let descriptor = tensor.descriptor();
        if descriptor.device() != DeviceId::CPU
            || descriptor.stream() != stream
            || !matches!(descriptor.dtype(), DType::F16 | DType::Bf16 | DType::F32)
            || descriptor.shape() != specification.shape
        {
            return Err(NativeDepthAnything3Error::StateShape {
                key: specification.key.clone(),
                expected: specification.shape.clone(),
                actual: descriptor.shape().to_vec(),
                actual_dtype: descriptor.dtype(),
            });
        }
        match dtype {
            Some(expected) if expected != descriptor.dtype() => {
                return Err(NativeDepthAnything3Error::InvalidCheckpoint(
                    "mixed retained-state dtypes are ambiguous".to_owned(),
                ));
            }
            None => dtype = Some(descriptor.dtype()),
            _ => {}
        }
        if descriptor.dtype() == DType::F32 {
            for bytes in tensor.contiguous_bytes()?.chunks(64 * 1_024) {
                cancellation.check()?;
                if bytes.chunks_exact(4).any(|value| {
                    !f32::from_le_bytes([value[0], value[1], value[2], value[3]]).is_finite()
                }) {
                    return Err(NativeDepthAnything3Error::InvalidCheckpoint(format!(
                        "state {} contains a non-finite value",
                        specification.key
                    )));
                }
            }
        }
    }
    dtype.ok_or(NativeDepthAnything3Error::UnsupportedArchitecture)
}

fn validate_f32_execution_state(
    state: &BTreeMap<String, Tensor>,
    specifications: &[StateSpecification],
    stream: StreamId,
    cancellation: &CancellationToken,
) -> Result<(), NativeDepthAnything3Error> {
    for (index, specification) in specifications.iter().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        let tensor = state
            .get(&specification.key)
            .ok_or_else(|| NativeDepthAnything3Error::MissingState(specification.key.clone()))?;
        if tensor.descriptor().dtype() != DType::F32
            || tensor.descriptor().device() != DeviceId::CPU
            || tensor.descriptor().stream() != stream
            || tensor.descriptor().shape() != specification.shape
        {
            return Err(NativeDepthAnything3Error::SemanticStateChanged);
        }
        for bytes in tensor.contiguous_bytes()?.chunks(64 * 1_024) {
            cancellation.check()?;
            if bytes.chunks_exact(4).any(|value| {
                !f32::from_le_bytes([value[0], value[1], value[2], value[3]]).is_finite()
            }) {
                return Err(NativeDepthAnything3Error::SemanticStateChanged);
            }
        }
    }
    if state.len() != specifications.len() {
        return Err(NativeDepthAnything3Error::SemanticStateChanged);
    }
    Ok(())
}

fn semantic_digest(
    configuration: &DepthAnything3Configuration,
    artifact_sha256: &str,
    source_dtype: DType,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, NativeDepthAnything3Error> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-depth-anything-3-resource-v1\0");
    hasher.update(artifact_sha256.as_bytes());
    hasher.update(configuration.backbone.identifier().as_bytes());
    hasher.update(match configuration.head {
        DepthAnything3Head::Dpt => b"dpt".as_slice(),
        DepthAnything3Head::DualDpt => b"dualdpt".as_slice(),
    });
    hasher.update(source_dtype.catalog_name().as_bytes());
    for (index, (key, tensor)) in source_state.iter().enumerate() {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update(tensor.descriptor().dtype().catalog_name().as_bytes());
        for dimension in tensor.descriptor().shape() {
            hasher.update(dimension.to_le_bytes());
        }
        for bytes in tensor.contiguous_bytes()?.chunks(64 * 1_024) {
            cancellation.check()?;
            hasher.update(bytes);
        }
        let projected = execution_state
            .get(key)
            .ok_or_else(|| NativeDepthAnything3Error::MissingState(key.clone()))?;
        for bytes in projected.contiguous_bytes()?.chunks(64 * 1_024) {
            cancellation.check()?;
            hasher.update(bytes);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resident_tensor_bytes<'a>(
    states: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeDepthAnything3Error> {
    resident_tensor_allocations(states, cancellation)?
        .into_iter()
        .try_fold(0_u64, |total, (_, bytes)| total.checked_add(bytes))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)
}

fn conservative_projected_resident_bytes(
    artifact_sha256: &String,
    source_state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<u64, NativeDepthAnything3Error> {
    let source_bytes = resident_tensor_bytes([source_state], cancellation)?;
    let mut projected_bytes = 0_u64;
    for (index, tensor) in source_state.values().enumerate() {
        if index.is_multiple_of(32) {
            cancellation.check()?;
        }
        projected_bytes = projected_bytes
            .checked_add(
                tensor
                    .descriptor()
                    .element_count()?
                    .checked_mul(DType::F32.byte_width())
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
            )
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    }
    let empty = BTreeMap::new();
    let owned = resident_owned_bytes(
        artifact_sha256,
        &String::with_capacity(64),
        source_state,
        &empty,
    )?;
    let projected_map_bytes = source_state
        .len()
        .checked_mul(std::mem::size_of::<(String, Tensor)>())
        .and_then(|bytes| {
            source_state
                .keys()
                .try_fold(bytes, |total, key| total.checked_add(key.capacity()))
        })
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    source_bytes
        .checked_add(projected_bytes)
        .and_then(|bytes| bytes.checked_add(owned))
        .and_then(|bytes| {
            u64::try_from(projected_map_bytes)
                .ok()
                .and_then(|map| bytes.checked_add(map))
        })
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)
}

fn resident_tensor_allocations<'a>(
    states: impl IntoIterator<Item = &'a BTreeMap<String, Tensor>>,
    cancellation: &CancellationToken,
) -> Result<Vec<(StorageId, u64)>, NativeDepthAnything3Error> {
    let mut allocations = BTreeMap::new();
    let mut index = 0_usize;
    for state in states {
        for tensor in state.values() {
            if index.is_multiple_of(32) {
                cancellation.check()?;
            }
            index = index
                .checked_add(1)
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
            match allocations.entry(tensor.storage_id().get()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((tensor.storage_id(), tensor.storage_byte_len()));
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get().1 != tensor.storage_byte_len() =>
                {
                    return Err(NativeDepthAnything3Error::SemanticStateChanged);
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(allocations.into_values().collect())
}

fn resident_owned_bytes(
    artifact_sha256: &String,
    semantic_digest_sha256: &String,
    source_state: &BTreeMap<String, Tensor>,
    execution_state: &BTreeMap<String, Tensor>,
) -> Result<u64, NativeDepthAnything3Error> {
    let mut bytes = std::mem::size_of::<NativeDepthAnything3Resource>();
    bytes = bytes
        .checked_add(artifact_sha256.capacity())
        .and_then(|value| value.checked_add(semantic_digest_sha256.capacity()))
        .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
    for state in [source_state, execution_state] {
        bytes = bytes
            .checked_add(
                state
                    .len()
                    .checked_mul(std::mem::size_of::<(String, Tensor)>())
                    .ok_or(NativeDepthAnything3Error::ShapeOverflow)?,
            )
            .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        for key in state.keys() {
            bytes = bytes
                .checked_add(key.capacity())
                .ok_or(NativeDepthAnything3Error::ShapeOverflow)?;
        }
    }
    u64::try_from(bytes).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
}

fn validate_sha256(value: &str) -> Result<(), NativeDepthAnything3Error> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(NativeDepthAnything3Error::InvalidCheckpoint(
            "artifact SHA-256 is malformed".to_owned(),
        ))
    }
}

fn validate_state_key(value: &str) -> Result<(), NativeDepthAnything3Error> {
    if value.is_empty()
        || value.len() > MAX_STATE_KEY_BYTES
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(NativeDepthAnything3Error::InvalidCheckpoint(
            "state key is empty, oversized, or contains control bytes".to_owned(),
        ));
    }
    Ok(())
}

fn usize_from(value: u64) -> Result<usize, NativeDepthAnything3Error> {
    usize::try_from(value).map_err(|_| NativeDepthAnything3Error::ShapeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, RetryRngPolicy, RngStreamAddress};

    const REDUCED_ORACLE: &str = if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        include_str!(
            "../../sim_test_support/fixtures/models/depth-anything-3-resource-foundation/oracle-linux-gnu-x86_64.json"
        )
    } else {
        include_str!(
            "../../sim_test_support/fixtures/models/depth-anything-3-resource-foundation/oracle.json"
        )
    };

    fn fixture_index_rows_sha256(domain: &str, rows: &[Vec<usize>]) -> String {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain.as_bytes());
        digest.update((rows.len() as u64).to_le_bytes());
        for row in rows {
            digest.update((row.len() as u64).to_le_bytes());
            for value in row {
                digest.update((*value as u64).to_le_bytes());
            }
        }
        format!("{:x}", digest.finalize())
    }

    #[test]
    fn reduced_da3_ransac_uses_the_versioned_shared_sample_stream()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let base_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let address = RngStreamAddress::new(
            "task390",
            "fixture",
            "depth-anything-3-ray-pose",
            0,
            "reduced-ray-pose",
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let context = ExecutionContext {
            stream: base_context.stream,
            scratch: base_context.scratch.clone(),
            rng_phase: Some(&address),
            cancellation: base_context.cancellation,
        };
        let (samples, _) = ray_pose_random_samples(&backend, 76, 17, &context)?;
        assert_eq!(samples.first(), Some(&vec![60, 49, 11, 21, 55, 35, 5, 45]));
        assert_eq!(samples.last(), Some(&vec![3, 40, 68, 45, 54, 23, 58, 71]));
        assert_eq!(
            fixture_index_rows_sha256("zed.sim.depth-anything-3.ransac-samples.v1", &samples,),
            "3ec843b71311674d0c4a0398c695277266171086b191b43d3439e5d102e67207"
        );
        let (changed_seed, _) = ray_pose_random_samples(&backend, 76, 18, &context)?;
        assert_ne!(changed_seed, samples);
        assert_eq!(
            fixture_index_rows_sha256("zed.sim.depth-anything-3.ransac-samples.v1", &changed_seed,),
            "e06d07bd06353a26e748280c7037adffd2a9ce0bb66bfd507504aa7c3e985874"
        );
        let changed_address = RngStreamAddress::new(
            "task390",
            "fixture",
            "depth-anything-3-ray-pose",
            0,
            "reduced-ray-pose-mutated",
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let changed_context = ExecutionContext {
            stream: base_context.stream,
            scratch: base_context.scratch.clone(),
            rng_phase: Some(&changed_address),
            cancellation: base_context.cancellation,
        };
        let (changed_phase, _) = ray_pose_random_samples(&backend, 76, 17, &changed_context)?;
        assert_ne!(changed_phase, samples);
        assert_ne!(changed_phase, changed_seed);
        assert_eq!(
            fixture_index_rows_sha256("zed.sim.depth-anything-3.ransac-samples.v1", &changed_phase,),
            "8732477ad3af2972b8280df320a79a17327300da47f918bdad66b78f4bf3b425"
        );
        Ok(())
    }

    #[test]
    fn reduced_da3_auxiliary_head_trace_localizes_pre_geometry_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let oracle: serde_json::Value = serde_json::from_str(REDUCED_ORACLE)?;
        let workspace_bytes = 256 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let resource = NativeDepthAnything3Resource::from_reduced_fixture(
            &backend,
            deterministic_reduced_depth_anything_3_checkpoint(
                &backend,
                DepthAnything3FixtureProfile::DualDpt,
                DType::F32,
                workspace_bytes,
                &context,
            )?,
            &context,
        )?;
        for mutation in oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/fixture_state")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 ray fixture state trace is missing")?
        {
            let key = mutation
                .get("key")
                .and_then(serde_json::Value::as_str)
                .ok_or("DA3 ray fixture state key is missing")?;
            let index = mutation
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or("DA3 ray fixture state index is missing")?;
            let expected = mutation
                .get("bits")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("DA3 ray fixture state bits are missing")?;
            let values = tensor_to_f32_with_context_exact_native(
                &backend,
                resource.execution_tensor(key)?,
                &context,
            )?;
            assert_eq!(
                values
                    .get(index)
                    .copied()
                    .ok_or("DA3 ray fixture state lane is unavailable")?
                    .to_bits(),
                expected,
                "DA3 ray fixture state {key}[{index}] changed"
            );
        }
        let multiview_values = oracle
            .get("multiview_input_bits")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 multiview input oracle is missing")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .map(f32::from_bits)
                    .ok_or("DA3 multiview input bits are invalid")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let image = ImageTensor::from_f32(&backend, &context, 3, 4, 4, 3, &multiview_values)?;
        let (_, trace) = resource.execute_pre_geometry_head_with_test_trace(
            &backend,
            NativeDepthAnything3Invocation {
                image: &image,
                views_per_sample: 3,
                process_resolution: 4,
                resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
                reference_strategy: NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
                use_ray_pose: true,
                ransac_seed: 17,
                extrinsics: None,
                intrinsics: None,
            },
            &context,
        )?;
        let mut prefix_phases = Vec::new();
        for (stage, tensor) in trace.auxiliary_resized.into_iter().enumerate() {
            prefix_phases.push((format!("resized_{stage}"), tensor));
        }
        for (phase, tensor) in [
            "refinenet4_aux",
            "refinenet3_aux",
            "refinenet2_aux",
            "refinenet1_aux",
        ]
        .into_iter()
        .zip(trace.auxiliary_fusion)
        {
            prefix_phases.push((phase.to_owned(), tensor));
        }
        let mut processed = trace.auxiliary_processed.into_iter();
        for level in 0..3 {
            prefix_phases.push((
                format!("output_conv1_aux_{level}"),
                processed
                    .next()
                    .ok_or("DA3 auxiliary processed phase is unavailable")?,
            ));
        }
        for (convolution, tensor) in trace
            .auxiliary_selected_convolutions
            .into_iter()
            .enumerate()
        {
            prefix_phases.push((format!("output_conv1_aux_3_conv_{convolution}"), tensor));
        }
        prefix_phases.push((
            "output_conv1_aux_3".to_owned(),
            processed
                .next()
                .ok_or("DA3 selected auxiliary processed phase is unavailable")?,
        ));
        if processed.next().is_some() {
            return Err("DA3 auxiliary processed phase count changed".into());
        }
        let expected_prefix_phases = oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/auxiliary_head_prefix_phases")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 auxiliary head prefix phase oracle is missing")?;
        if prefix_phases.len() != expected_prefix_phases.len() {
            return Err(format!(
                "DA3 auxiliary head prefix phase count: actual {}, expected {}",
                prefix_phases.len(),
                expected_prefix_phases.len()
            )
            .into());
        }
        for ((phase, actual), expected) in prefix_phases.into_iter().zip(expected_prefix_phases) {
            if expected.get("phase").and_then(serde_json::Value::as_str) != Some(phase.as_str()) {
                return Err(format!("DA3 auxiliary head phase order changed at {phase}").into());
            }
            let expected_shape = expected
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 auxiliary head prefix shape is missing")?
                .iter()
                .map(|value| value.as_u64().ok_or("DA3 auxiliary head shape is invalid"))
                .collect::<Result<Vec<_>, _>>()?;
            if actual.descriptor().shape() != expected_shape {
                return Err(format!(
                    "DA3 auxiliary head {phase} shape: actual {:?}, expected {expected_shape:?}",
                    actual.descriptor().shape()
                )
                .into());
            }
            let values = tensor_to_f32_with_context_exact_native(&backend, &actual, &context)?;
            let bits = values.iter().copied().map(f32::to_bits).collect::<Vec<_>>();
            let expected_edge_bits = |field: &str| -> Result<Vec<u32>, Box<dyn std::error::Error>> {
                Ok(expected
                    .get(field)
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| format!("DA3 auxiliary head {phase} {field} is missing"))?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .ok_or_else(|| format!("DA3 auxiliary head {phase} {field} is invalid"))
                    })
                    .collect::<Result<Vec<_>, _>>()?)
            };
            let first = bits.iter().copied().take(8).collect::<Vec<_>>();
            let last = bits
                .iter()
                .copied()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>();
            let expected_first = expected_edge_bits("first_bits")?;
            let expected_last = expected_edge_bits("last_bits")?;
            for (edge, actual, expected) in [
                ("first", &first, &expected_first),
                ("last", &last, &expected_last),
            ] {
                if let Some((index, (actual, expected))) = actual
                    .iter()
                    .zip(expected)
                    .enumerate()
                    .find(|(_, (actual, expected))| actual != expected)
                {
                    return Err(format!(
                        "DA3 auxiliary head {phase} {edge}[{index}]: actual {actual:#010x}, expected {expected:#010x}"
                    )
                    .into());
                }
            }
            let domain = expected
                .get("raw_f32_sha256_domain")
                .and_then(serde_json::Value::as_str)
                .ok_or("DA3 auxiliary head SHA domain is missing")?;
            let mut digest = Sha256::new();
            digest.update((domain.len() as u64).to_le_bytes());
            digest.update(domain.as_bytes());
            digest.update((phase.len() as u64).to_le_bytes());
            digest.update(phase.as_bytes());
            digest.update((actual.descriptor().shape().len() as u64).to_le_bytes());
            for dimension in actual.descriptor().shape() {
                digest.update(dimension.to_le_bytes());
            }
            let raw_bytes = u64::try_from(bits.len())?
                .checked_mul(4)
                .ok_or("DA3 auxiliary head raw byte count overflowed")?;
            digest.update(raw_bytes.to_le_bytes());
            for value in bits {
                digest.update(value.to_le_bytes());
            }
            let actual_digest = format!("{:x}", digest.finalize());
            let expected_digest = expected
                .get("raw_f32_sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or("DA3 auxiliary head SHA is missing")?;
            if actual_digest != expected_digest {
                return Err(format!(
                    "DA3 auxiliary head {phase} raw F32 SHA: actual {actual_digest}, expected {expected_digest}"
                )
                .into());
            }
        }
        let position_lane = trace
            .auxiliary_position_lane
            .ok_or("DA3 auxiliary position lane trace is unavailable")?;
        let expected_position_lane = oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/auxiliary_position_lane")
            .ok_or("DA3 auxiliary position lane oracle is missing")?;
        for (field, actual) in [
            ("x_coordinate_bits", &position_lane.x_coordinate_bits),
            ("y_coordinate_bits", &position_lane.y_coordinate_bits),
        ] {
            let expected = expected_position_lane
                .get(field)
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 position coordinate oracle is missing")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or("DA3 position coordinate bit is invalid")
                })
                .collect::<Result<Vec<_>, _>>()?;
            if actual.len() != expected.len() {
                return Err(format!(
                    "DA3 auxiliary position {field} length: actual {}, expected {}",
                    actual.len(),
                    expected.len()
                )
                .into());
            }
            if let Some((index, (actual, expected))) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (actual, expected))| actual != expected)
            {
                return Err(format!(
                    "DA3 auxiliary position {field}[{index}]: actual {actual:#010x}, expected {expected:#010x}"
                )
                .into());
            }
        }
        for (field, actual) in [
            ("batch", position_lane.batch),
            ("channel", position_lane.channel),
            ("y", position_lane.y),
            ("x", position_lane.x),
        ] {
            let expected = expected_position_lane
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or("DA3 auxiliary position lane index is missing")?;
            if actual != expected {
                return Err(format!(
                    "DA3 auxiliary position {field}: actual {actual}, expected {expected}"
                )
                .into());
            }
        }
        for (field, actual) in [
            ("exponent_ratio_bits", position_lane.exponent_ratio_bits),
            ("positive_power_bits", position_lane.positive_power_bits),
            ("reciprocal_bits", position_lane.reciprocal_bits),
            ("negative_power_bits", position_lane.negative_power_bits),
            ("frequency_bits", position_lane.frequency_bits),
            (
                "discriminator_ratio_bits",
                position_lane.discriminator_ratio_bits,
            ),
            (
                "discriminator_positive_power_bits",
                position_lane.discriminator_positive_power_bits,
            ),
            (
                "discriminator_reciprocal_bits",
                position_lane.discriminator_reciprocal_bits,
            ),
            (
                "discriminator_negative_power_bits",
                position_lane.discriminator_negative_power_bits,
            ),
            ("position_bits", position_lane.position_bits),
            ("phase_bits", position_lane.phase_bits),
        ] {
            let expected = expected_position_lane
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("DA3 auxiliary position scalar oracle is missing")?;
            if actual != expected {
                return Err(format!(
                    "DA3 auxiliary position {field}: actual {actual:#010x}, expected {expected:#010x}"
                )
                .into());
            }
        }
        for (function, actual, source_field, libc_field) in [
            (
                "sin",
                position_lane.sine_bits,
                "python_double_sine_bits",
                "libc_sine_bits",
            ),
            (
                "cos",
                position_lane.cosine_bits,
                "python_double_cosine_bits",
                "libc_cosine_bits",
            ),
        ] {
            let source = expected_position_lane
                .get(source_field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("DA3 source position transcendental is missing")?;
            let libc = expected_position_lane
                .get(libc_field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("DA3 libc position transcendental is missing")?;
            if actual != libc {
                return Err(format!(
                    "DA3 auxiliary position {function}: actual {actual:#010x}, source-graph {source:#010x}, pinned-libc {libc:#010x}"
                )
                .into());
            }
            if function == "sin" && source == libc {
                return Err("DA3 auxiliary position sinf discriminator was washed out".into());
            }
        }
        if position_lane.discriminator_reciprocal_bits
            == position_lane.discriminator_negative_power_bits
        {
            return Err("DA3 reciprocal-positive-pow discriminator was washed out".into());
        }
        if expected_position_lane
            .get("selected_trig")
            .and_then(serde_json::Value::as_str)
            != Some(position_lane.selected_trig)
        {
            return Err("DA3 auxiliary position selected transcendental changed".into());
        }
        for (field, actual) in [
            ("selected_trig_bits", position_lane.selected_trig_bits),
            ("ratio_bits", position_lane.ratio_bits),
            ("embedding_bits", position_lane.embedding_bits),
            ("input_bits", position_lane.input_bits),
            ("output_bits", position_lane.output_bits),
        ] {
            let expected = expected_position_lane
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("DA3 auxiliary position result oracle is missing")?;
            if actual != expected {
                return Err(format!(
                    "DA3 auxiliary position {field}: actual {actual:#010x}, expected {expected:#010x}"
                )
                .into());
            }
        }
        let confidence_activation = oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/auxiliary_confidence_activation_lane")
            .ok_or("DA3 auxiliary confidence activation oracle is missing")?;
        let confidence_index = confidence_activation
            .get("confidence_index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("DA3 auxiliary confidence index is missing")?;
        let logit_index = confidence_activation
            .get("logit_index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("DA3 auxiliary confidence logit index is missing")?;
        let logits = tensor_to_f32_with_context_exact_native(
            &backend,
            trace
                .auxiliary_logits
                .as_ref()
                .ok_or("DA3 auxiliary confidence logits are unavailable")?,
            &context,
        )?;
        let logit = logits
            .get(logit_index)
            .copied()
            .ok_or("DA3 auxiliary confidence logit lane is unavailable")?;
        let expected_bits = |field: &str| -> Result<u32, Box<dyn std::error::Error>> {
            Ok(confidence_activation
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("DA3 auxiliary confidence {field} is missing"))?)
        };
        assert_eq!(logit.to_bits(), expected_bits("logit_bits")?);
        let python_double_exp = expected_bits("python_double_exp_bits")?;
        let libc_expf = expected_bits("libc_expf_bits")?;
        if !cfg!(all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )) {
            assert_ne!(python_double_exp, libc_expf);
        }
        assert_eq!(logit.exp().to_bits(), libc_expf);
        let confidence = tensor_to_f32_with_context_exact_native(
            &backend,
            trace
                .raw_ray_confidence
                .as_ref()
                .ok_or("DA3 auxiliary confidence tensor is unavailable")?,
            &context,
        )?;
        assert_eq!(
            confidence
                .get(confidence_index)
                .copied()
                .ok_or("DA3 auxiliary confidence lane is unavailable")?
                .to_bits(),
            expected_bits("libc_expp1_bits")?
        );
        if !cfg!(all(
            target_os = "linux",
            target_arch = "x86_64",
            target_env = "gnu"
        )) {
            assert_ne!(
                expected_bits("python_double_expp1_bits")?,
                expected_bits("libc_expp1_bits")?,
            );
        }
        let phases = [
            ("positioned", trace.auxiliary_positioned),
            ("convolution_3_0", trace.auxiliary_convolution),
            ("normalized", trace.auxiliary_normalized),
            ("relu", trace.auxiliary_relu),
            ("logits", trace.auxiliary_logits),
            ("ray", trace.raw_ray),
            ("confidence", trace.raw_ray_confidence),
        ];
        for (phase, actual) in phases {
            let actual = actual.ok_or("DA3 auxiliary head phase is unavailable")?;
            let expected = oracle
                .pointer(&format!(
                    "/reduced_dualdpt/ray_pose_trace/auxiliary_head_phases/{phase}"
                ))
                .ok_or("DA3 auxiliary head phase oracle is missing")?;
            let mut expected_shape = expected
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 auxiliary head phase shape is missing")?
                .iter()
                .map(|value| value.as_u64().ok_or("DA3 auxiliary head shape is invalid"))
                .collect::<Result<Vec<_>, _>>()?;
            if matches!(phase, "ray" | "confidence") {
                expected_shape.insert(0, 1);
            }
            if actual.descriptor().shape() != expected_shape {
                return Err(format!(
                    "DA3 auxiliary head {phase} shape: actual {:?}, expected {expected_shape:?}",
                    actual.descriptor().shape()
                )
                .into());
            }
            let actual_bits = tensor_to_f32_with_context_exact_native(&backend, &actual, &context)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let expected_bits = expected
                .get("bits")
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 auxiliary head phase bits are missing")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or("DA3 auxiliary head phase bit is invalid")
                })
                .collect::<Result<Vec<_>, _>>()?;
            if actual_bits.len() != expected_bits.len() {
                return Err(format!(
                    "DA3 auxiliary head {phase} length: actual {}, expected {}",
                    actual_bits.len(),
                    expected_bits.len()
                )
                .into());
            }
            if let Some((index, (actual, expected))) = actual_bits
                .iter()
                .zip(&expected_bits)
                .enumerate()
                .find(|(_, (actual, expected))| actual != expected)
            {
                return Err(format!(
                    "DA3 auxiliary head {phase}[{index}]: actual {actual:#010x}, expected {expected:#010x}"
                )
                .into());
            }
        }
        let mutation = oracle
            .pointer("/full_path_mutations/dual_auxiliary_ray")
            .ok_or("DA3 auxiliary ray mutation oracle is missing")?;
        let mut mutated_checkpoint = deterministic_reduced_depth_anything_3_checkpoint(
            &backend,
            DepthAnything3FixtureProfile::DualDpt,
            DType::F32,
            workspace_bytes,
            &context,
        )?;
        mutate_reduced_depth_anything_3_checkpoint(
            &backend,
            &mut mutated_checkpoint,
            DepthAnything3FixtureMutation {
                state_key: mutation
                    .get("state_key")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("DA3 auxiliary ray mutation key is missing")?,
                lane: mutation
                    .get("lane")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or("DA3 auxiliary ray mutation lane is missing")?,
                delta: mutation
                    .get("delta_bits")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .map(f32::from_bits)
                    .ok_or("DA3 auxiliary ray mutation delta is missing")?,
            },
            &context,
        )?;
        let mutated_resource = NativeDepthAnything3Resource::from_reduced_fixture(
            &backend,
            mutated_checkpoint,
            &context,
        )?;
        let (mutated_geometry, mutated_trace) = mutated_resource
            .execute_pre_geometry_head_with_test_trace(
                &backend,
                NativeDepthAnything3Invocation {
                    image: &image,
                    views_per_sample: 3,
                    process_resolution: 4,
                    resize_method: NativeDepthAnything3ResizeMethod::LowerBound,
                    reference_strategy: NativeDepthAnything3ReferenceStrategy::SaddleSimRange,
                    use_ray_pose: true,
                    ransac_seed: 17,
                    extrinsics: None,
                    intrinsics: None,
                },
                &context,
            )?;
        let mutated_confidence = mutated_geometry
            .confidence
            .as_ref()
            .ok_or("DA3 mutated main confidence is unavailable")?;
        let mutated_ray = mutated_trace
            .raw_ray
            .as_ref()
            .ok_or("DA3 mutated raw ray is unavailable")?;
        let mutated_ray_confidence = mutated_trace
            .raw_ray_confidence
            .as_ref()
            .ok_or("DA3 mutated raw ray confidence is unavailable")?;
        for (phase, actual) in [
            ("depth", &mutated_geometry.depth),
            ("confidence", mutated_confidence),
            ("raw_ray", mutated_ray),
            ("raw_ray_confidence", mutated_ray_confidence),
        ] {
            let actual = tensor_to_f32_with_context_exact_native(&backend, actual, &context)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let expected = mutation
                .get(phase)
                .and_then(|value| value.get("bits"))
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 auxiliary ray mutation phase is missing")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or("DA3 auxiliary ray mutation bit is invalid")
                })
                .collect::<Result<Vec<_>, _>>()?;
            if actual.len() != expected.len() {
                return Err(format!(
                    "DA3 auxiliary ray mutation {phase} length: actual {}, expected {}",
                    actual.len(),
                    expected.len()
                )
                .into());
            }
            if let Some((index, (actual, expected))) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .find(|(_, (actual, expected))| actual != expected)
            {
                return Err(format!(
                    "DA3 auxiliary ray mutation {phase}[{index}]: actual {actual:#010x}, expected {expected:#010x}"
                )
                .into());
            }
        }
        let baseline_ray = oracle
            .pointer("/reduced_dualdpt/ray_pose_trace/pre_geometry_ray/bits")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 baseline raw ray oracle is missing")?;
        let mutated_ray_bits = mutation
            .pointer("/raw_ray/bits")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 mutated raw ray oracle is missing")?;
        assert_ne!(mutated_ray_bits, baseline_ray);
        assert_eq!(
            mutation
                .get("source_raw_ray_changed")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn reduced_da3_ray_pose_trace_localizes_source_equation_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let oracle: serde_json::Value = serde_json::from_str(REDUCED_ORACLE)?;
        let fixture_tensor = |pointer: &str,
                              backend: &CpuBackend,
                              context: &ExecutionContext<'_>|
         -> Result<Tensor, Box<dyn std::error::Error>> {
            let document = oracle
                .pointer(pointer)
                .ok_or_else(|| format!("missing DA3 oracle {pointer}"))?;
            let mut shape = document
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("missing DA3 oracle shape {pointer}"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| format!("invalid DA3 oracle shape {pointer}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut values = document
                .get("bits")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("missing DA3 oracle bits {pointer}"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .map(f32::from_bits)
                        .ok_or_else(|| format!("invalid DA3 oracle bits {pointer}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let values_per_view = shape
                .iter()
                .skip(1)
                .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("invalid DA3 oracle view shape {pointer}"))?;
            *shape
                .first_mut()
                .ok_or_else(|| format!("empty DA3 oracle view shape {pointer}"))? = 1;
            shape.insert(0, 1);
            values.truncate(values_per_view);
            Ok(tensor_from_f32_with_context_exact_native(
                backend,
                &shape,
                &values,
                DType::F32,
                DeviceId::CPU,
                context,
            )?)
        };
        let workspace_bytes = 64 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let base_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let address = RngStreamAddress::new(
            "task390",
            "fixture",
            "depth-anything-3-ray-pose",
            0,
            "reduced-ray-pose",
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let context = ExecutionContext {
            stream: base_context.stream,
            scratch: base_context.scratch.clone(),
            rng_phase: Some(&address),
            cancellation: base_context.cancellation,
        };
        let ray = fixture_tensor(
            "/reduced_dualdpt/ray_pose_trace/pre_geometry_ray",
            &backend,
            &context,
        )?;
        let confidence = fixture_tensor(
            "/reduced_dualdpt/ray_pose_trace/pre_geometry_confidence",
            &backend,
            &context,
        )?;
        let mut trace = DepthAnything3ExecutionTrace::default();
        let (extrinsics, intrinsics) = ray_pose_geometry(
            &backend,
            &ray,
            &confidence,
            1,
            1,
            17,
            &context,
            Some(&mut trace),
        )?;
        let expected_trace = oracle
            .pointer("/reduced_dualdpt/ray_pose_trace")
            .ok_or("missing DA3 ray-pose trace")?;
        let expected_samples = expected_trace
            .get("samples")
            .and_then(serde_json::Value::as_array)
            .ok_or("missing DA3 ray-pose samples")?
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or("invalid DA3 ray-pose sample row")?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or("invalid DA3 ray-pose sample")
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            trace.ransac_samples, expected_samples,
            "RANSAC sample table"
        );
        let expected_views = expected_trace
            .get("views")
            .and_then(serde_json::Value::as_array)
            .ok_or("missing DA3 ray-pose view traces")?;
        assert_eq!(trace.ransac_views.len(), 1);
        for (view_index, (actual, expected)) in trace
            .ransac_views
            .iter()
            .zip(expected_views.iter().take(1))
            .enumerate()
        {
            let expected_candidates = expected
                .get("candidate_indices")
                .and_then(serde_json::Value::as_array)
                .ok_or("missing DA3 RANSAC candidates")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or("invalid DA3 RANSAC candidate")
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(
                actual.candidate_indices, expected_candidates,
                "view {view_index} candidates"
            );
            assert_eq!(
                actual.best_iteration as u64,
                expected
                    .get("best_iteration")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("missing DA3 RANSAC best iteration")?,
                "view {view_index} best iteration"
            );
            assert_eq!(
                actual.best_score.to_bits() as u64,
                expected
                    .get("best_score_bits")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("missing DA3 RANSAC best score")?,
                "view {view_index} best score"
            );
            assert_eq!(
                actual.best_inliers.len() as u64,
                expected
                    .get("best_inlier_count")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or("missing DA3 RANSAC inlier count")?,
                "view {view_index} inlier count"
            );
            let inlier_domain = expected
                .get("best_inliers_sha256_domain")
                .and_then(serde_json::Value::as_str)
                .ok_or("missing DA3 RANSAC inlier domain")?;
            assert_eq!(
                fixture_index_rows_sha256(
                    inlier_domain,
                    std::slice::from_ref(&actual.best_inliers)
                ),
                expected
                    .get("best_inliers_sha256")
                    .and_then(serde_json::Value::as_str)
                    .ok_or("missing DA3 RANSAC inlier SHA")?,
                "view {view_index} inlier identity"
            );
            for (phase, actual_values) in [
                (
                    "normalized_homography_bits",
                    actual.normalized_homography.as_slice(),
                ),
                (
                    "homography_post_sign_bits",
                    actual.signed_homography.as_slice(),
                ),
                ("rotation_bits", actual.rotation.as_slice()),
                ("lower_bits", actual.lower.as_slice()),
                ("c2w_pre_inverse_bits", actual.c2w_pre_inverse.as_slice()),
            ] {
                let expected_values = expected
                    .get(phase)
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| format!("missing DA3 RANSAC {phase}"))?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .map(f32::from_bits)
                            .ok_or_else(|| format!("invalid DA3 RANSAC {phase}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (lane, (actual_value, expected_value)) in actual_values
                    .iter()
                    .copied()
                    .zip(expected_values)
                    .enumerate()
                {
                    let tolerance = 2.5e-3_f32.max(expected_value.abs() * 2.5e-3);
                    assert!(
                        (actual_value - expected_value).abs() <= tolerance,
                        "view {view_index} {phase}[{lane}]: {actual_value} != {expected_value}"
                    );
                }
            }
        }
        for (actual, pointer) in [
            (
                extrinsics.as_ref().ok_or("missing DA3 ray extrinsics")?,
                "/reduced_dualdpt/ray_pose_trace/geometry/extrinsics/bits",
            ),
            (
                intrinsics.as_ref().ok_or("missing DA3 ray intrinsics")?,
                "/reduced_dualdpt/ray_pose_trace/geometry/intrinsics/bits",
            ),
        ] {
            let mut expected = oracle
                .pointer(pointer)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("missing DA3 ray geometry {pointer}"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .map(f32::from_bits)
                        .ok_or_else(|| format!("invalid DA3 ray geometry {pointer}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let actual = tensor_values(actual, &cancellation)?;
            expected.truncate(actual.len());
            assert_eq!(actual.len(), expected.len());
            for (lane, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                let tolerance = 2.5e-3_f32.max(expected.abs() * 2.5e-3);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{pointer}[{lane}]: {actual} != {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn affine_inverse_materializes_the_transposed_rotation_view()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let affine = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 1, 3, 4],
            &[
                1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 6.0, 11.0, 7.0, 8.0, 9.0, 12.0,
            ],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;

        let inverse = affine_inverse_3x4(&backend, &affine, &context)?;

        assert_eq!(inverse.descriptor().shape(), &[1, 1, 3, 4]);
        assert_eq!(
            tensor_to_f32_with_context_exact_native(&backend, &inverse, &context)?,
            [
                1.0, 4.0, 7.0, -138.0, 2.0, 5.0, 8.0, -171.0, 3.0, 6.0, 9.0, -204.0,
            ]
        );
        Ok(())
    }

    #[test]
    fn camera_field_of_view_uses_the_f32_atan_boundary() -> Result<(), Box<dyn std::error::Error>> {
        let oracle: serde_json::Value = serde_json::from_str(REDUCED_ORACLE)?;
        let discriminator = oracle
            .pointer("/camera_projection/atanf_discriminator")
            .ok_or("DA3 atanf discriminator is missing")?;
        let input = f32::from_bits(u32::try_from(
            discriminator
                .get("input_bits")
                .and_then(serde_json::Value::as_u64)
                .ok_or("DA3 atanf input bits are missing")?,
        )?);
        let expected = u32::try_from(
            discriminator
                .get("output_bits")
                .and_then(serde_json::Value::as_u64)
                .ok_or("DA3 atanf output bits are missing")?,
        )?;
        let double_cast = u32::try_from(
            discriminator
                .get("python_double_cast_bits")
                .and_then(serde_json::Value::as_u64)
                .ok_or("DA3 double atanf output bits are missing")?,
        )?;

        assert_eq!(input.atan().to_bits(), expected);
        if let Some(probes) = oracle.get("native_libm_discriminators") {
            let probes = probes.as_array().ok_or("DA3 libm probes are malformed")?;
            assert!(!probes.is_empty());
            for probe in probes {
                let bits = |name: &str| -> Result<u32, Box<dyn std::error::Error>> {
                    Ok(u32::try_from(
                        probe
                            .get(name)
                            .and_then(serde_json::Value::as_u64)
                            .ok_or("DA3 libm probe field is missing")?,
                    )?)
                };
                let value = f32::from_bits(bits("input_bits")?);
                let observed = match probe.get("operation").and_then(serde_json::Value::as_str) {
                    Some("atan") => value.atan(),
                    Some("exp") => value.exp(),
                    _ => return Err("DA3 libm probe operation is unknown".into()),
                };
                assert_eq!(observed.to_bits(), bits("output_bits")?);
                assert_ne!(bits("output_bits")?, bits("double_cast_bits")?);
            }
        } else {
            assert_ne!(expected, double_cast);
        }
        Ok(())
    }

    #[test]
    fn reduced_da3_storage_projection_and_dpt_execution_are_checked()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 128 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let oracle: serde_json::Value = serde_json::from_str(REDUCED_ORACLE)?;
        let storage_input = oracle
            .pointer("/storage_projection/input_bits")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 storage input oracle is missing")?
            .iter()
            .map(|value| {
                u32::try_from(value.as_u64().ok_or("DA3 storage input bit is invalid")?)
                    .map(f32::from_bits)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        for (dtype, storage_pointer, projected_pointer) in [
            (
                DType::F16,
                "/storage_projection/f16_storage_bits",
                "/storage_projection/f16_projected_bits",
            ),
            (
                DType::Bf16,
                "/storage_projection/bf16_storage_bits",
                "/storage_projection/bf16_projected_bits",
            ),
        ] {
            let mut checkpoint = deterministic_reduced_depth_anything_3_checkpoint(
                &backend,
                DepthAnything3FixtureProfile::Dpt,
                dtype,
                workspace_bytes,
                &context,
            )?;
            let (state_key, tensor) = checkpoint
                .ordered_state
                .iter_mut()
                .find(|(key, tensor)| {
                    key.ends_with("patch_embeddings.projection.weight")
                        && tensor
                            .descriptor()
                            .element_count()
                            .is_ok_and(|count| count >= storage_input.len() as u64)
                })
                .ok_or("DA3 storage projection state is missing")?;
            let state_key = state_key.clone();
            let shape = tensor.descriptor().shape().to_vec();
            let mut values = vec![0.0; usize::try_from(tensor.descriptor().element_count()?)?];
            values[..storage_input.len()].copy_from_slice(&storage_input);
            *tensor = tensor_from_f32_with_context_exact_native(
                &backend,
                &shape,
                &values,
                dtype,
                DeviceId::CPU,
                &context,
            )?;
            let resource =
                NativeDepthAnything3Resource::from_reduced_fixture(&backend, checkpoint, &context)?;
            let source = resource
                .source_state
                .get(&state_key)
                .ok_or("DA3 retained source state is missing")?;
            let actual_storage = (0..storage_input.len())
                .map(|index| {
                    let bytes = source.linear_element_bytes(u64::try_from(index)?)?;
                    Ok::<_, Box<dyn std::error::Error>>(u16::from_le_bytes(
                        bytes.try_into().map_err(|_| "DA3 storage width changed")?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let expected_storage = oracle
                .pointer(storage_pointer)
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 storage oracle is missing")?
                .iter()
                .map(|value| {
                    u16::try_from(value.as_u64().ok_or("DA3 storage bit is invalid")?)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            assert_eq!(actual_storage, expected_storage);
            let projected = tensor_values(
                resource
                    .execution_state
                    .get(&state_key)
                    .ok_or("DA3 projected execution state is missing")?,
                &cancellation,
            )?;
            let expected_projected = oracle
                .pointer(projected_pointer)
                .and_then(serde_json::Value::as_array)
                .ok_or("DA3 projected oracle is missing")?
                .iter()
                .map(|value| {
                    u32::try_from(value.as_u64().ok_or("DA3 projected bit is invalid")?)
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            assert_eq!(
                projected[..storage_input.len()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected_projected
            );
            let reconstructed = resource.reconstruct_checkpoint(&cancellation)?;
            let reconstructed = reconstructed
                .ordered_state
                .into_iter()
                .find(|(key, _)| key == &state_key)
                .ok_or("DA3 reconstructed state is missing")?
                .1;
            assert_eq!(reconstructed.storage_id(), source.storage_id());
        }
        let mut identities = BTreeSet::new();
        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let checkpoint = deterministic_reduced_depth_anything_3_checkpoint(
                &backend,
                DepthAnything3FixtureProfile::Dpt,
                dtype,
                workspace_bytes,
                &context,
            )?;
            let resource =
                NativeDepthAnything3Resource::from_reduced_fixture(&backend, checkpoint, &context)?;
            assert_eq!(resource.source_dtype(), dtype);
            resource.validate(&cancellation)?;
            assert_eq!(
                resource
                    .reconstruct_checkpoint(&cancellation)?
                    .ordered_state
                    .len(),
                resource.source_state.len()
            );
            identities.insert(resource.semantic_digest_sha256().to_owned());
        }
        assert_eq!(identities.len(), 3);

        let resource = NativeDepthAnything3Resource::from_reduced_fixture(
            &backend,
            deterministic_reduced_depth_anything_3_checkpoint(
                &backend,
                DepthAnything3FixtureProfile::Dpt,
                DType::F32,
                workspace_bytes,
                &context,
            )?,
            &context,
        )?;
        let pixels = (0..4 * 4 * 3)
            .map(|index| (index as f32 + 1.0) / 64.0)
            .collect::<Vec<_>>();
        let image = ImageTensor::from_f32(&backend, &context, 1, 4, 4, 3, &pixels)?;
        let geometry = resource.execute(
            &backend,
            NativeDepthAnything3Invocation {
                image: &image,
                views_per_sample: 1,
                process_resolution: 4,
                resize_method: NativeDepthAnything3ResizeMethod::UpperBound,
                reference_strategy: NativeDepthAnything3ReferenceStrategy::First,
                use_ray_pose: false,
                ransac_seed: 7,
                extrinsics: None,
                intrinsics: None,
            },
            &context,
        )?;
        assert_eq!(geometry.depth.descriptor().shape(), &[1, 4, 4]);
        assert!(geometry.confidence.is_none());
        assert_eq!(
            geometry
                .sky
                .as_ref()
                .map(|tensor| tensor.descriptor().shape()),
            Some([1, 4, 4].as_slice())
        );

        Ok(())
    }

    #[test]
    fn reduced_da3_admission_is_atomic_and_alias_aware() -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 128 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let checkpoint = || {
            deterministic_reduced_depth_anything_3_checkpoint(
                &backend,
                DepthAnything3FixtureProfile::Dpt,
                DType::F16,
                workspace_bytes,
                &context,
            )
        };
        let mut unexpected = checkpoint()?;
        unexpected.ordered_state.push((
            "native.backbone.embeddings.mask_token".to_owned(),
            tensor_from_f32_with_context_exact_native(
                &backend,
                &[1],
                &[0.0],
                DType::F16,
                DeviceId::CPU,
                &context,
            )?,
        ));
        assert!(matches!(
            NativeDepthAnything3Resource::from_reduced_fixture(&backend, unexpected, &context),
            Err(NativeDepthAnything3Error::UnexpectedState(_))
        ));

        let mut nonfinite = checkpoint()?;
        let nonfinite_key = {
            let (key, tensor) = nonfinite
                .ordered_state
                .iter_mut()
                .find(|(key, _)| key.ends_with("patch_embeddings.projection.weight"))
                .ok_or("patch weight is missing")?;
            let key = key.clone();
            let shape = tensor.descriptor().shape().to_vec();
            let count = tensor.descriptor().element_count()? as usize;
            let mut values = filled_f32(count, 0.0)?;
            values[0] = f32::NAN;
            *tensor = tensor_from_f32_with_context_exact_native(
                &backend,
                &shape,
                &values,
                DType::F16,
                DeviceId::CPU,
                &context,
            )?;
            key
        };
        assert!(
            matches!(
                NativeDepthAnything3Resource::from_reduced_fixture(&backend, nonfinite, &context),
                Err(NativeDepthAnything3Error::InvalidCheckpoint(_)
                    | NativeDepthAnything3Error::SemanticStateChanged)
            ),
            "nonfinite state {nonfinite_key} was admitted"
        );

        let mut oom = checkpoint()?;
        oom.memory_budget_bytes = 1;
        assert!(matches!(
            NativeDepthAnything3Resource::from_reduced_fixture(&backend, oom, &context),
            Err(NativeDepthAnything3Error::OutOfMemory { .. })
        ));

        let distinct = checkpoint()?;
        let distinct_resource = NativeDepthAnything3Resource::from_reduced_fixture(
            &backend,
            distinct.clone(),
            &context,
        )?;
        let mut aliased = distinct;
        let mut pair = None;
        for first in 0..aliased.ordered_state.len() {
            for second in first + 1..aliased.ordered_state.len() {
                if aliased.ordered_state[first].1.descriptor().shape()
                    == aliased.ordered_state[second].1.descriptor().shape()
                {
                    pair = Some((first, second));
                    break;
                }
            }
            if pair.is_some() {
                break;
            }
        }
        let (first, second) = pair.ok_or("no same-shaped state pair")?;
        let removed_bytes = aliased.ordered_state[second].1.storage_byte_len();
        aliased.ordered_state[second].1 = aliased.ordered_state[first].1.clone();
        let aliased_resource =
            NativeDepthAnything3Resource::from_reduced_fixture(&backend, aliased, &context)?;
        assert_eq!(
            distinct_resource.resident_bytes() - aliased_resource.resident_bytes(),
            removed_bytes
        );

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancelled,
        );
        assert!(matches!(
            NativeDepthAnything3Resource::from_reduced_fixture(
                &backend,
                checkpoint()?,
                &cancelled_context,
            ),
            Err(NativeDepthAnything3Error::Cancelled
                | NativeDepthAnything3Error::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn da3_source_helper_dispositions_are_exact() -> Result<(), Box<dyn std::error::Error>> {
        let oracle: serde_json::Value = serde_json::from_str(REDUCED_ORACLE)?;
        let bits = |pointer: &str| -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            oracle
                .pointer(pointer)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| format!("missing DA3 oracle {pointer}"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| "DA3 oracle bit is not an integer".into())
                        .and_then(|value| u32::try_from(value).map_err(Into::into))
                })
                .collect()
        };
        let tied = rotation_matrix_to_quaternion(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]);
        assert_eq!(
            tied.map(f32::to_bits).to_vec(),
            bits("/quaternion_tie/output_bits")?
        );
        assert_eq!(
            target_size(
                1,
                10_000,
                14,
                NativeDepthAnything3ResizeMethod::UpperBound,
                14,
            )?,
            (1, 14)
        );
        assert_eq!(
            {
                let diagonal_value = -0.0_f32;
                if diagonal_value > 0.0 {
                    1.0
                } else if diagonal_value < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
            0.0
        );
        let workspace_bytes = 1 << 20;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let nonsquare_input = bits("/nonsquare_resize_projection/input_bits")?
            .into_iter()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        let nonsquare = ImageTensor::from_f32(&backend, &context, 1, 2, 4, 3, &nonsquare_input)?;
        for (name, target) in [("upper_bound", (4, 6)), ("lower_bound", (6, 12))] {
            let preprocessed =
                preprocess_image(&backend, &nonsquare, target.0, target.1, &context)?;
            let actual = tensor_values(&preprocessed, &cancellation)?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                bits(&format!(
                    "/nonsquare_resize_projection/cases/{name}/preprocessed/bits"
                ))?,
                "{name} Lanczos preprocessing diverged"
            );
        }
        let rotated = crate::dino2::apply_da3_rotary(
            &[0.75, -0.25, 0.5, 1.25],
            1,
            1,
            1,
            4,
            &[2],
            &[1],
            &cancellation,
        )?;
        assert_eq!(
            rotated.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            bits("/asymmetric_rope/output_bits")?
        );
        let (x, y) = stateless_position_coordinates(&backend, 5, 3, 3, 2, &context)?;
        let expected_grid = oracle
            .pointer("/position_grid/xy_bits")
            .and_then(serde_json::Value::as_array)
            .ok_or("DA3 position grid oracle is missing")?;
        let actual_grid = (0..2)
            .flat_map(|row| {
                let x = &x;
                let y = &y;
                (0..3).map(move |column| vec![x[column].to_bits(), y[row].to_bits()])
            })
            .collect::<Vec<_>>();
        let expected_grid = expected_grid
            .iter()
            .map(|pair| {
                pair.as_array()
                    .ok_or("DA3 position-grid pair is missing")?
                    .iter()
                    .map(|value| {
                        u32::try_from(value.as_u64().ok_or("DA3 grid bit is not an integer")?)
                            .map_err(Into::into)
                    })
                    .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        assert_eq!(actual_grid, expected_grid);
        let (ray_x, ray_y) = ray_identity_coordinates(&backend, 5, 3, &context)?;
        assert_eq!(
            ray_x.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            bits("/ray_identity_grid/x_bits")?
        );
        assert_eq!(
            ray_y.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            bits("/ray_identity_grid/y_bits")?
        );
        let (extrinsics, intrinsics) = pose_encoding_geometry(
            &backend,
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0],
            &[0.75, 1.0],
            1,
            1,
            4,
            8,
            &context,
        )?;
        assert_eq!(
            extrinsics
                .as_ref()
                .map(|tensor| tensor.descriptor().shape()),
            Some([1, 1, 3, 4].as_slice())
        );
        let intrinsics = tensor_values(
            intrinsics.as_ref().ok_or("DA3 intrinsics missing")?,
            &cancellation,
        )?;
        assert_eq!(
            vec![
                intrinsics[0].to_bits(),
                intrinsics[4].to_bits(),
                intrinsics[8].to_bits()
            ],
            bits("/camera_projection/intrinsics_diagonal_bits")?
        );
        assert_eq!(
            oracle
                .pointer("/ransac/profile_version")
                .and_then(serde_json::Value::as_u64),
            Some(2)
        );
        assert_eq!(
            oracle
                .pointer("/ransac/iterations")
                .and_then(serde_json::Value::as_u64),
            Some(100)
        );
        assert_eq!(
            oracle
                .pointer("/ransac/maximum_refit_inliers")
                .and_then(serde_json::Value::as_u64),
            Some(8_000)
        );
        let address = RngStreamAddress::new(
            "task390",
            "fixture",
            "refit",
            0,
            "source-order",
            0,
            0,
            RetryRngPolicy::Replay,
        )?;
        let stream = generator_exact_native(
            RngProfileVersion::V2,
            RngAlgorithm::Mt19937,
            17,
            address.clone(),
            &cancellation,
        )?;
        let mut inliers = (0..10_000).collect::<Vec<_>>();
        let transaction =
            refit_inliers_if_needed(&backend, &mut inliers, stream.begin(None)?, &context)?;
        assert_eq!(inliers.len(), 8_000);
        assert!(inliers.iter().all(|index| *index < 9_500));
        let after_helper = randperm_with_context_exact_native(&backend, 16, transaction, &context)?;
        let replay = generator_exact_native(
            RngProfileVersion::V2,
            RngAlgorithm::Mt19937,
            17,
            address,
            &cancellation,
        )?;
        let first =
            randperm_with_context_exact_native(&backend, 9_500, replay.begin(None)?, &context)?;
        let expected_inliers = tensor_i64_values(&first.tensor, &cancellation)?
            .into_iter()
            .take(8_000)
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(inliers, expected_inliers);
        let after_manual =
            randperm_with_context_exact_native(&backend, 16, first.transaction, &context)?;
        assert_eq!(
            tensor_i64_values(&after_helper.tensor, &cancellation)?,
            tensor_i64_values(&after_manual.tensor, &cancellation)?,
            "95%-then-8000 refit must consume exactly one evolving randperm transaction"
        );
        Ok(())
    }

    #[test]
    fn da3_original_projection_and_serial_retention_are_preflighted()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 128 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let checkpoint = deterministic_reduced_depth_anything_3_checkpoint(
            &backend,
            DepthAnything3FixtureProfile::Dpt,
            DType::F32,
            workspace_bytes,
            &context,
        )?;
        let mut resource =
            NativeDepthAnything3Resource::from_reduced_fixture(&backend, checkpoint, &context)?;
        resource.memory_budget_bytes = 0;
        let small_required =
            match preflight_invocation_memory(&resource, 1, 1, 2, 2, 2, 2, &context) {
                Err(NativeDepthAnything3Error::OutOfMemory { required, .. }) => required,
                Ok(()) => return Err("expected small projection OOM".into()),
                Err(error) => return Err(error.into()),
            };
        resource.memory_budget_bytes = 0;
        let large_required =
            match preflight_invocation_memory(&resource, 1, 1, 2, 2, 1_024, 1_024, &context) {
                Err(NativeDepthAnything3Error::OutOfMemory { required, .. }) => required,
                result => {
                    return Err(format!("expected original projection OOM, got {result:?}").into());
                }
            };
        assert!(large_required > small_required);
        resource.memory_budget_bytes = large_required - 1;
        assert!(matches!(
            preflight_invocation_memory(&resource, 1, 1, 2, 2, 1_024, 1_024, &context),
            Err(NativeDepthAnything3Error::OutOfMemory { .. })
        ));

        let batch = 64;
        let height = 32;
        let width = 32;
        let process_resolution = 32;
        let single_required =
            invocation_memory_required(&resource, 1, 1, height, width, height, width)?;
        let output_planes = 1_u64
            + u64::from(resource.configuration.head_output_dimension > 1)
            + u64::from(resource.configuration.use_sky_head);
        let old_serial_required =
            resource.resident_bytes + batch * height * width * output_planes * 4 * 2;
        let combined_required = serial_invocation_memory_required(
            &resource,
            batch,
            height,
            width,
            process_resolution,
            NativeDepthAnything3ResizeMethod::UpperBound,
        )?;
        let old_separate_bound = single_required.max(old_serial_required);
        assert!(combined_required > old_separate_bound);
        resource.memory_budget_bytes = old_separate_bound;
        assert!(
            preflight_invocation_memory(&resource, 1, 1, height, width, height, width, &context,)
                .is_ok()
        );
        assert!(old_serial_required <= resource.memory_budget_bytes);
        assert!(matches!(
            preflight_serial_outputs(
                &resource,
                batch,
                height,
                width,
                process_resolution,
                NativeDepthAnything3ResizeMethod::UpperBound,
                &context,
            ),
            Err(NativeDepthAnything3Error::OutOfMemory { required, budget })
                if required == combined_required && budget == old_separate_bound
        ));
        Ok(())
    }
}
