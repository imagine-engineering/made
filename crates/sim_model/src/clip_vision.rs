use crate::ParserLimits;
use crate::native_node_payload::{
    NativeModelPayloadError, NativeModelResourceIdentity, NativeModelResourceRole,
    NativeStructuredProjection, require_structured_projection,
};
use crate::native_ops::{
    GeluApproximation, NativeExecutionRequirements, NativeModule, NativeOpsError,
};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, Layout,
    MemoryFormatReference, OperationSupport, ResizeCrop, ResizeMode, ResizeSpec, Scalar,
    ScalarSide, StorageId, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    UnaryOperation, ViewAccess,
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, normalize_with_context_exact_native,
        resize_with_context_exact_native,
    },
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_native_diffusion::{NativeDiffusionTensorError, quick_gelu as canonical_quick_gelu},
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, tensor_repeat_with_context_exact_native,
        torch_cat_with_context_exact_native, torch_reshape_with_context_exact_native,
        torch_stack_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        ShapeLayoutTransformPartThreeError, tensor_permute_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
    generated_storage_dtype_device_01::{
        StorageDTypeDeviceError, contiguous_with_context_exact_native,
    },
};
use std::{collections::BTreeMap, mem};
use thiserror::Error;

pub const CLIP_VISION_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/clip_model.py";
pub const CLIP_VISION_SOURCE_SHA256: &str =
    "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845";
pub const CLIP_VISION_SOURCE_TYPE_ID: &str = "CLIP_VISION";
pub const CLIP_VISION_RESOURCE_ROLE: &str = "clip_vision";
const CLIP_VISION_RESOURCE_FORMAT: &str = "zed-native-sim-clip-vision-v1";
pub const CLIP_VISION_CATALOG_SYMBOLS: [&str; 9] = [
    "clip_preprocess",
    "siglip2_flex_calc_resolution",
    "siglip2_preprocess",
    "siglip2_pos_embed",
    "Siglip2Embeddings",
    "CLIPVisionEmbeddings",
    "CLIPVision",
    "LlavaProjector",
    "CLIPVisionModelProjection",
];

const DEFAULT_CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const DEFAULT_CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
const DEFAULT_SIGLIP_MEAN: [f32; 3] = [0.5; 3];
const DEFAULT_SIGLIP_STD: [f32; 3] = [0.5; 3];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipVisionModelType {
    Clip,
    Siglip,
    Siglip2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipVisionActivation {
    QuickGelu,
    Gelu,
    GeluTanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipVisionIntermediate {
    None,
    Layer(isize),
    All,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipVisionConfiguration {
    pub model_type: ClipVisionModelType,
    pub dtype: DType,
    pub device: DeviceId,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention_heads: usize,
    pub layer_count: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub max_num_patches: usize,
    pub activation: ClipVisionActivation,
    pub projection_dimension: Option<usize>,
    pub llava_projection_dimension: Option<usize>,
}

impl ClipVisionConfiguration {
    pub fn validate(&self) -> Result<(), ClipVisionError> {
        if self.device != DeviceId::CPU || self.dtype != DType::F32 {
            return Err(ClipVisionError::UnsupportedTarget {
                dtype: self.dtype,
                device: self.device,
            });
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.attention_heads == 0
            || self.layer_count == 0
            || self.patch_size == 0
            || self.num_channels != 3
            || !self.hidden_size.is_multiple_of(self.attention_heads)
            || self.projection_dimension == Some(0)
            || self.llava_projection_dimension == Some(0)
        {
            return Err(ClipVisionError::InvalidConfiguration(
                "hidden/intermediate/layer dimensions, heads, channels, or projections are invalid",
            ));
        }
        if self.model_type != ClipVisionModelType::Siglip2 && self.image_size == 0 {
            return Err(ClipVisionError::InvalidConfiguration(
                "fixed-resolution CLIP vision requires a nonzero image size",
            ));
        }
        if self.image_size > 0 && !self.image_size.is_multiple_of(self.patch_size) {
            return Err(ClipVisionError::InvalidConfiguration(
                "fixed-resolution CLIP vision image size must be divisible by patch size",
            ));
        }
        if self.model_type == ClipVisionModelType::Siglip2 && self.max_num_patches == 0 {
            return Err(ClipVisionError::InvalidConfiguration(
                "SigLIP2 requires a nonzero maximum patch count",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ClipVisionLayerWeights {
    pub layer_norm_1_weight: Tensor,
    pub layer_norm_1_bias: Tensor,
    pub query_weight: Tensor,
    pub query_bias: Tensor,
    pub key_weight: Tensor,
    pub key_bias: Tensor,
    pub value_weight: Tensor,
    pub value_bias: Tensor,
    pub output_weight: Tensor,
    pub output_bias: Tensor,
    pub layer_norm_2_weight: Tensor,
    pub layer_norm_2_bias: Tensor,
    pub feed_forward_1_weight: Tensor,
    pub feed_forward_1_bias: Tensor,
    pub feed_forward_2_weight: Tensor,
    pub feed_forward_2_bias: Tensor,
}

#[derive(Clone, Debug)]
pub struct ClipVisionWeights {
    pub patch_embedding_weight: Tensor,
    pub patch_embedding_bias: Option<Tensor>,
    pub class_embedding: Option<Tensor>,
    pub position_embedding: Tensor,
    pub pre_layer_norm_weight: Option<Tensor>,
    pub pre_layer_norm_bias: Option<Tensor>,
    pub layers: Vec<ClipVisionLayerWeights>,
    pub post_layer_norm_weight: Tensor,
    pub post_layer_norm_bias: Tensor,
    pub visual_projection_weight: Option<Tensor>,
    pub llava_linear_1_weight: Option<Tensor>,
    pub llava_linear_1_bias: Option<Tensor>,
    pub llava_linear_2_weight: Option<Tensor>,
    pub llava_linear_2_bias: Option<Tensor>,
}

#[derive(Clone, Debug)]
struct ClipVisionLayer {
    layer_norm_1: NativeModule,
    attention: NativeModule,
    layer_norm_2: NativeModule,
    feed_forward_1: NativeModule,
    activation: NativeModule,
    feed_forward_2: NativeModule,
    quick_gelu: bool,
}

#[derive(Clone, Debug)]
pub struct NativeClipVision {
    configuration: ClipVisionConfiguration,
    stream: StreamId,
    patch_embedding: NativeModule,
    class_embedding: Option<Tensor>,
    position_embedding: Tensor,
    pre_layer_norm: Option<NativeModule>,
    layers: Vec<ClipVisionLayer>,
    post_layer_norm: NativeModule,
    visual_projection: Option<NativeModule>,
    llava_linear_1: Option<NativeModule>,
    llava_activation: NativeModule,
    llava_linear_2: Option<NativeModule>,
    canonical_state: Vec<(String, Tensor)>,
    recipe_binding_sha256: Option<String>,
    semantic_identity: NativeModelResourceIdentity,
    module_state_digest_sha256: String,
    training: bool,
}

pub struct NativeClipVisionRecipeConfiguration {
    pub artifact_sha256: String,
    pub bytes: Vec<u8>,
    pub parser_limits: ParserLimits,
}

#[derive(Clone, Debug)]
pub struct NativeClipVisionExecutionSession {
    model: NativeClipVision,
}

#[derive(Debug)]
pub(crate) struct NativeClipVisionCheckedForward {
    pub(crate) output: ClipVisionOutput,
    pub(crate) pooled_hidden: Tensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClipVisionConstructionPhase {
    Entry,
    CanonicalState,
    ParameterProjection,
    PatchEmbedding,
    PreLayerNorm,
    LayerNorm1(usize),
    AttentionQuery(usize),
    AttentionKey(usize),
    AttentionValue(usize),
    AttentionOutput(usize),
    LayerNorm2(usize),
    MlpInput(usize),
    MlpActivation(usize),
    MlpOutput(usize),
    PostLayerNorm,
    VisualProjection,
    LlavaProjection,
    SemanticDigest,
    ModuleDigest,
    Validation,
    Return,
}

fn construction_checkpoint(
    cancellation: &CancellationToken,
    phase: ClipVisionConstructionPhase,
    phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
) -> Result<(), ClipVisionError> {
    phase_hook(phase, cancellation);
    cancellation.check()?;
    Ok(())
}

struct ClipVisionCanonicalStateCursor<'a> {
    state: &'a [(String, Tensor)],
    index: usize,
}

impl<'a> ClipVisionCanonicalStateCursor<'a> {
    fn new(state: &'a [(String, Tensor)]) -> Self {
        Self { state, index: 0 }
    }

    fn take(
        &mut self,
        expected_name: &str,
        cancellation: &CancellationToken,
        phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
    ) -> Result<Tensor, ClipVisionError> {
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::CanonicalState,
            phase_hook,
        )?;
        let (name, tensor) =
            self.state
                .get(self.index)
                .ok_or(ClipVisionError::InvalidConfiguration(
                    "canonical state is missing a required tensor",
                ))?;
        if name != expected_name {
            return Err(ClipVisionError::InvalidConfiguration(
                "canonical state tensor order changed",
            ));
        }
        self.index = self
            .index
            .checked_add(1)
            .ok_or(ClipVisionError::Overflow("canonical state cursor"))?;
        Ok(tensor.clone())
    }

    fn take_optional(
        &mut self,
        expected_name: &str,
        present: bool,
        cancellation: &CancellationToken,
        phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
    ) -> Result<Option<Tensor>, ClipVisionError> {
        present
            .then(|| self.take(expected_name, cancellation, phase_hook))
            .transpose()
    }

    fn finish(
        self,
        cancellation: &CancellationToken,
        phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
    ) -> Result<(), ClipVisionError> {
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::CanonicalState,
            phase_hook,
        )?;
        if self.index != self.state.len() {
            return Err(ClipVisionError::InvalidConfiguration(
                "canonical state contains an unexpected tensor",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClipVisionTensorResidentAllocation {
    storage_id: StorageId,
    resident_bytes: u64,
}

impl ClipVisionTensorResidentAllocation {
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipVisionResidentParts {
    owned_bytes: u64,
    tensor_allocations: Vec<ClipVisionTensorResidentAllocation>,
}

impl ClipVisionResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocations(&self) -> &[ClipVisionTensorResidentAllocation] {
        &self.tensor_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, ClipVisionError> {
        self.tensor_allocations
            .iter()
            .try_fold(self.owned_bytes, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes)
                    .ok_or(ClipVisionError::Overflow("resident total bytes"))
            })
    }
}

#[derive(Clone, Debug)]
pub struct ClipVisionOutput {
    pub last_hidden_state: Tensor,
    pub intermediate: Option<Tensor>,
    pub image_embeds: Tensor,
    pub projected_intermediate: Option<Tensor>,
    image_sizes: Box<[[u64; 3]]>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl ClipVisionOutput {
    pub const SOURCE_TYPE_ID: &'static str = "CLIP_VISION_OUTPUT";

    pub fn checked(
        last_hidden_state: Tensor,
        intermediate: Option<Tensor>,
        image_embeds: Tensor,
        projected_intermediate: Option<Tensor>,
        image_sizes: Vec<[u64; 3]>,
    ) -> Result<Self, ClipVisionError> {
        let image_sizes = image_sizes.into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) = project_clip_vision_output(
            &last_hidden_state,
            intermediate.as_ref(),
            &image_embeds,
            projected_intermediate.as_ref(),
            &image_sizes,
        )?;
        Ok(Self {
            last_hidden_state,
            intermediate,
            image_embeds,
            projected_intermediate,
            image_sizes,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub fn image_sizes(&self) -> &[[u64; 3]] {
        &self.image_sizes
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<ClipVisionResidentParts, ClipVisionError> {
        clip_vision_output_resident_parts(
            &self.last_hidden_state,
            self.intermediate.as_ref(),
            &self.image_embeds,
            self.projected_intermediate.as_ref(),
            &self.image_sizes,
        )
    }

    pub fn validate(&self) -> Result<(), ClipVisionError> {
        let (semantic_digest_sha256, resident_bytes) = project_clip_vision_output(
            &self.last_hidden_state,
            self.intermediate.as_ref(),
            &self.image_embeds,
            self.projected_intermediate.as_ref(),
            &self.image_sizes,
        )?;
        require_structured_projection(
            self.semantic_digest_sha256,
            semantic_digest_sha256,
            self.resident_bytes,
            resident_bytes,
        )?;
        if self.resident_parts()?.resident_bytes()? != self.resident_bytes {
            return Err(ClipVisionError::StructuredPayload(
                NativeModelPayloadError::StructuredProjectionChanged,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ClipVisionError {
    #[error(transparent)]
    StructuredPayload(#[from] NativeModelPayloadError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    ImageOperation(#[from] ExternalTensorKernelPartOneError),
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error(transparent)]
    NativeDiffusion(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Indexing(#[from] IndexingMaskingPartOneError),
    #[error(transparent)]
    ShapeLayoutTwo(#[from] ShapeLayoutTransformPartTwoError),
    #[error(transparent)]
    ShapeLayoutThree(#[from] ShapeLayoutTransformPartThreeError),
    #[error(transparent)]
    StorageDTypeDevice(#[from] StorageDTypeDeviceError),
    #[error("CLIP vision configuration is invalid: {0}")]
    InvalidConfiguration(&'static str),
    #[error("CLIP vision input is invalid: {0}")]
    InvalidInput(&'static str),
    #[error("CLIP vision tensor {name} has shape {actual:?}; expected {expected:?}")]
    Shape {
        name: &'static str,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error(
        "CLIP vision tensor {name} target mismatch: expected {expected_dtype:?} on {expected_device:?} stream {expected_stream:?}, got {actual_dtype:?} on {actual_device:?} stream {actual_stream:?}"
    )]
    TensorTarget {
        name: &'static str,
        expected_dtype: DType,
        actual_dtype: DType,
        expected_device: DeviceId,
        actual_device: DeviceId,
        expected_stream: StreamId,
        actual_stream: StreamId,
    },
    #[error("CLIP vision execution target {dtype:?} on {device:?} is unsupported")]
    UnsupportedTarget { dtype: DType, device: DeviceId },
    #[error("CLIP vision intermediate layer {requested} is outside {available} layers")]
    IntermediateOutOfRange { requested: isize, available: usize },
    #[error("Llava projection requires an intermediate CLIP vision tensor")]
    MissingLlavaIntermediate,
    #[error("CLIP vision shape arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("CLIP vision semantic identity changed")]
    SemanticIdentityChanged,
    #[error("CLIP vision model is in training mode; call eval() before native execution")]
    EvaluationRequired,
    #[error("CLIP vision execution was cancelled")]
    Cancelled,
}

impl From<sim_types::CancellationError> for ClipVisionError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

fn project_clip_vision_output(
    last_hidden_state: &Tensor,
    intermediate: Option<&Tensor>,
    image_embeds: &Tensor,
    projected_intermediate: Option<&Tensor>,
    image_sizes: &[[u64; 3]],
) -> Result<([u8; 32], u64), ClipVisionError> {
    let last_descriptor = last_hidden_state.descriptor();
    let last_shape = last_descriptor.shape();
    if last_shape.len() != 3 || last_shape.contains(&0) {
        return Err(ClipVisionError::InvalidInput(
            "CLIP vision last hidden state must be nonempty [batch, tokens, hidden]",
        ));
    }
    let batch = usize::try_from(last_shape[0])
        .map_err(|_| ClipVisionError::Overflow("CLIP vision output batch"))?;
    let first_image_size = image_sizes
        .first()
        .copied()
        .ok_or(ClipVisionError::InvalidInput(
            "CLIP vision image sizes must contain one entry per batch item",
        ))?;
    if image_sizes.len() != batch
        || first_image_size[0] != 3
        || first_image_size[1] == 0
        || first_image_size[2] == 0
        || image_sizes.iter().any(|size| *size != first_image_size)
    {
        return Err(ClipVisionError::InvalidInput(
            "CLIP vision image sizes must repeat a nonempty [3, height, width] source shape",
        ));
    }

    let image_embed_shape = image_embeds.descriptor().shape();
    if !matches!(image_embed_shape.len(), 2 | 3)
        || image_embed_shape.first().copied() != Some(last_shape[0])
        || image_embed_shape.contains(&0)
    {
        return Err(ClipVisionError::InvalidInput(
            "CLIP vision image embeds must be nonempty [batch, projection] or [batch, tokens, projection]",
        ));
    }
    require_clip_output_target(last_hidden_state, image_embeds, "image embeds")?;

    if let Some(intermediate) = intermediate {
        let shape = intermediate.descriptor().shape();
        let shape_matches = match shape {
            [intermediate_batch, tokens, hidden] => {
                *intermediate_batch == last_shape[0]
                    && *tokens == last_shape[1]
                    && *hidden == last_shape[2]
            }
            [intermediate_batch, layers, tokens, hidden] => {
                *intermediate_batch == last_shape[0]
                    && *layers > 0
                    && *tokens == last_shape[1]
                    && *hidden == last_shape[2]
            }
            _ => false,
        };
        if !shape_matches {
            return Err(ClipVisionError::InvalidInput(
                "CLIP vision intermediate must preserve batch, token, and hidden dimensions",
            ));
        }
        require_clip_output_target(last_hidden_state, intermediate, "intermediate")?;
    }

    if let Some(projected_intermediate) = projected_intermediate {
        let intermediate = intermediate.ok_or(ClipVisionError::InvalidInput(
            "projected CLIP vision intermediate requires a selected intermediate layer",
        ))?;
        let intermediate_shape = intermediate.descriptor().shape();
        let projected_shape = projected_intermediate.descriptor().shape();
        if intermediate_shape.len() != 3
            || projected_shape.len() != 3
            || projected_shape[0] != last_shape[0]
            || intermediate_shape[1] <= 1
            || projected_shape[1] != intermediate_shape[1] - 1
            || projected_shape[2] == 0
        {
            return Err(ClipVisionError::InvalidInput(
                "projected CLIP vision intermediate must drop exactly the class token",
            ));
        }
        require_clip_output_target(
            last_hidden_state,
            projected_intermediate,
            "projected intermediate",
        )?;
    }

    let mut projection = NativeStructuredProjection::new::<ClipVisionOutput>(
        b"zed.sim.model.clip-vision-output.v1",
    )?;
    projection.hash_float_tensor(last_hidden_state)?;
    match intermediate {
        Some(intermediate) => {
            projection.hash_tag(1);
            projection.hash_float_tensor(intermediate)?;
        }
        None => projection.hash_tag(0),
    }
    projection.hash_float_tensor(image_embeds)?;
    match projected_intermediate {
        Some(projected_intermediate) => {
            projection.hash_tag(1);
            projection.hash_float_tensor(projected_intermediate)?;
        }
        None => projection.hash_tag(0),
    }
    projection.hash_len(image_sizes.len())?;
    projection.add_allocation::<[u64; 3]>(image_sizes.len())?;
    for image_size in image_sizes {
        for dimension in image_size {
            projection.hash_u64(*dimension);
        }
    }
    Ok(projection.finish())
}

fn clip_vision_resident_parts<'a>(
    owned_bytes: u64,
    tensors: impl IntoIterator<Item = &'a Tensor>,
) -> Result<ClipVisionResidentParts, ClipVisionError> {
    let mut storages = BTreeMap::new();
    for tensor in tensors {
        let storage_id = tensor.storage_id();
        let resident_bytes = tensor.storage_byte_len();
        if let Some(existing) = storages.insert(storage_id.get(), (storage_id, resident_bytes))
            && existing.1 != resident_bytes
        {
            return Err(ClipVisionError::Overflow(
                "resident tensor allocation changed",
            ));
        }
    }
    let parts = ClipVisionResidentParts {
        owned_bytes,
        tensor_allocations: storages
            .into_values()
            .map(
                |(storage_id, resident_bytes)| ClipVisionTensorResidentAllocation {
                    storage_id,
                    resident_bytes,
                },
            )
            .collect(),
    };
    parts.resident_bytes()?;
    Ok(parts)
}

fn clip_vision_output_resident_parts(
    last_hidden_state: &Tensor,
    intermediate: Option<&Tensor>,
    image_embeds: &Tensor,
    projected_intermediate: Option<&Tensor>,
    image_sizes: &[[u64; 3]],
) -> Result<ClipVisionResidentParts, ClipVisionError> {
    let owned_bytes = u64::try_from(mem::size_of::<ClipVisionOutput>())
        .map_err(|_| ClipVisionError::Overflow("resident output bytes"))?
        .checked_add(
            u64::try_from(
                mem::size_of::<[u64; 3]>()
                    .checked_mul(image_sizes.len())
                    .ok_or(ClipVisionError::Overflow("resident image sizes"))?,
            )
            .map_err(|_| ClipVisionError::Overflow("resident image sizes"))?,
        )
        .ok_or(ClipVisionError::Overflow("resident output bytes"))?;
    clip_vision_resident_parts(
        owned_bytes,
        std::iter::once(last_hidden_state)
            .chain(intermediate)
            .chain(std::iter::once(image_embeds))
            .chain(projected_intermediate),
    )
}

fn require_clip_output_target(
    reference: &Tensor,
    tensor: &Tensor,
    field: &'static str,
) -> Result<(), ClipVisionError> {
    let expected = reference.descriptor();
    let actual = tensor.descriptor();
    if expected.dtype() != actual.dtype()
        || expected.device() != actual.device()
        || expected.stream() != actual.stream()
    {
        return Err(ClipVisionError::InvalidInput(field));
    }
    Ok(())
}

struct ClipVisionSemanticHasher(Sha256);

impl ClipVisionSemanticHasher {
    fn new(domain: &[u8]) -> Result<Self, ClipVisionError> {
        let mut hasher = Self(Sha256::new());
        hasher.field(domain)?;
        Ok(hasher)
    }

    fn field(&mut self, value: &[u8]) -> Result<(), ClipVisionError> {
        self.0.update(
            u64::try_from(value.len())
                .map_err(|_| ClipVisionError::Overflow("semantic digest field"))?
                .to_le_bytes(),
        );
        self.0.update(value);
        Ok(())
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}

fn clip_vision_architecture_digest(
    configuration: &ClipVisionConfiguration,
) -> Result<String, ClipVisionError> {
    let mut digest = ClipVisionSemanticHasher::new(b"zed.sim.model.clip-vision-architecture.v1")?;
    digest.field(CLIP_VISION_SOURCE_SHA256.as_bytes())?;
    digest.field(CLIP_VISION_SOURCE_TYPE_ID.as_bytes())?;
    digest.field(CLIP_VISION_RESOURCE_ROLE.as_bytes())?;
    digest.field(&[match configuration.model_type {
        ClipVisionModelType::Clip => 1,
        ClipVisionModelType::Siglip => 2,
        ClipVisionModelType::Siglip2 => 3,
    }])?;
    digest.field(&[match configuration.activation {
        ClipVisionActivation::QuickGelu => 1,
        ClipVisionActivation::Gelu => 2,
        ClipVisionActivation::GeluTanh => 3,
    }])?;
    for value in [
        configuration.hidden_size,
        configuration.intermediate_size,
        configuration.attention_heads,
        configuration.layer_count,
        configuration.image_size,
        configuration.patch_size,
        configuration.num_channels,
        configuration.max_num_patches,
        configuration.projection_dimension.unwrap_or(0),
        configuration.llava_projection_dimension.unwrap_or(0),
    ] {
        digest.u64(
            u64::try_from(value)
                .map_err(|_| ClipVisionError::Overflow("architecture dimension"))?,
        );
    }
    Ok(digest.finish())
}

fn clip_vision_canonical_state(
    weights: &ClipVisionWeights,
    cancellation: &CancellationToken,
    phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
) -> Result<Vec<(String, Tensor)>, ClipVisionError> {
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::CanonicalState,
        phase_hook,
    )?;
    let mut state = Vec::new();
    state
        .try_reserve_exact(
            14_usize
                .checked_add(
                    weights
                        .layers
                        .len()
                        .checked_mul(16)
                        .ok_or(ClipVisionError::Overflow("canonical state entries"))?,
                )
                .ok_or(ClipVisionError::Overflow("canonical state entries"))?,
        )
        .map_err(|_| ClipVisionError::Overflow("canonical state entries"))?;
    let mut push = |name: &str, tensor: Option<&Tensor>| -> Result<(), ClipVisionError> {
        if let Some(tensor) = tensor {
            construction_checkpoint(
                cancellation,
                ClipVisionConstructionPhase::CanonicalState,
                phase_hook,
            )?;
            state.push((name.to_owned(), tensor.clone()));
        }
        Ok(())
    };
    push(
        "patch_embedding.weight",
        Some(&weights.patch_embedding_weight),
    )?;
    push(
        "patch_embedding.bias",
        weights.patch_embedding_bias.as_ref(),
    )?;
    push("class_embedding", weights.class_embedding.as_ref())?;
    push("position_embedding", Some(&weights.position_embedding))?;
    push(
        "pre_layer_norm.weight",
        weights.pre_layer_norm_weight.as_ref(),
    )?;
    push("pre_layer_norm.bias", weights.pre_layer_norm_bias.as_ref())?;
    for (index, layer) in weights.layers.iter().enumerate() {
        for (name, tensor) in layer_tensors(layer) {
            push(&format!("layers.{index}.{name}"), Some(tensor))?;
        }
    }
    push(
        "post_layer_norm.weight",
        Some(&weights.post_layer_norm_weight),
    )?;
    push("post_layer_norm.bias", Some(&weights.post_layer_norm_bias))?;
    push(
        "visual_projection.weight",
        weights.visual_projection_weight.as_ref(),
    )?;
    push(
        "llava_linear_1.weight",
        weights.llava_linear_1_weight.as_ref(),
    )?;
    push("llava_linear_1.bias", weights.llava_linear_1_bias.as_ref())?;
    push(
        "llava_linear_2.weight",
        weights.llava_linear_2_weight.as_ref(),
    )?;
    push("llava_linear_2.bias", weights.llava_linear_2_bias.as_ref())?;
    Ok(state)
}

fn clip_vision_state_digest(
    state: &[(String, Tensor)],
    cancellation: &CancellationToken,
) -> Result<String, ClipVisionError> {
    cancellation.check()?;
    let mut digest = ClipVisionSemanticHasher::new(b"zed.sim.model.clip-vision-state.v1")?;
    digest.u64(u64::try_from(state.len()).map_err(|_| ClipVisionError::Overflow("state entries"))?);
    for (name, tensor) in state {
        cancellation.check()?;
        digest.field(name.as_bytes())?;
        let descriptor = tensor.descriptor();
        digest.u64(
            u64::try_from(descriptor.shape().len())
                .map_err(|_| ClipVisionError::Overflow("state rank"))?,
        );
        for dimension in descriptor.shape() {
            digest.u64(*dimension);
        }
        for bytes in tensor.contiguous_bytes()?.chunks(64 * 1024) {
            cancellation.check()?;
            digest.field(bytes)?;
        }
    }
    Ok(digest.finish())
}

fn clip_vision_identity(
    configuration: &ClipVisionConfiguration,
    state: &[(String, Tensor)],
    recipe_binding_sha256: Option<&str>,
    cancellation: &CancellationToken,
) -> Result<NativeModelResourceIdentity, ClipVisionError> {
    let architecture_digest = clip_vision_architecture_digest(configuration)?;
    let architecture_digest = match recipe_binding_sha256 {
        Some(recipe_binding_sha256) => {
            let mut hasher = Sha256::new();
            hasher.update(b"zed.sim.clip-vision-recipe-identity.v1");
            hasher.update(architecture_digest.as_bytes());
            hasher.update(recipe_binding_sha256.as_bytes());
            format!("{:x}", hasher.finalize())
        }
        None => architecture_digest,
    };
    Ok(NativeModelResourceIdentity::checked(
        NativeModelResourceRole::ClipVision,
        "sim.clip_model.CLIPVision",
        CLIP_VISION_RESOURCE_FORMAT,
        clip_vision_state_digest(state, cancellation)?,
        architecture_digest,
    )?)
}

fn append_module_digest(
    digest: &mut ClipVisionSemanticHasher,
    name: &str,
    module: &NativeModule,
    cancellation: &CancellationToken,
) -> Result<(), ClipVisionError> {
    digest.field(name.as_bytes())?;
    digest.field(module.semantic_state_digest(cancellation)?.as_bytes())?;
    Ok(())
}

fn clip_vision_module_state_digest(
    model: &NativeClipVision,
    cancellation: &CancellationToken,
) -> Result<String, ClipVisionError> {
    cancellation.check()?;
    let mut digest = ClipVisionSemanticHasher::new(b"zed.sim.model.clip-vision-modules.v1")?;
    append_module_digest(
        &mut digest,
        "patch_embedding",
        &model.patch_embedding,
        cancellation,
    )?;
    if let Some(module) = &model.pre_layer_norm {
        append_module_digest(&mut digest, "pre_layer_norm", module, cancellation)?;
    }
    for (index, layer) in model.layers.iter().enumerate() {
        for (name, module) in [
            ("layer_norm_1", &layer.layer_norm_1),
            ("attention", &layer.attention),
            ("layer_norm_2", &layer.layer_norm_2),
            ("feed_forward_1", &layer.feed_forward_1),
            ("activation", &layer.activation),
            ("feed_forward_2", &layer.feed_forward_2),
        ] {
            append_module_digest(
                &mut digest,
                &format!("layers.{index}.{name}"),
                module,
                cancellation,
            )?;
        }
    }
    append_module_digest(
        &mut digest,
        "post_layer_norm",
        &model.post_layer_norm,
        cancellation,
    )?;
    for (name, module) in [
        ("visual_projection", model.visual_projection.as_ref()),
        ("llava_linear_1", model.llava_linear_1.as_ref()),
        ("llava_linear_2", model.llava_linear_2.as_ref()),
    ] {
        if let Some(module) = module {
            append_module_digest(&mut digest, name, module, cancellation)?;
        }
    }
    cancellation.check()?;
    Ok(digest.finish())
}

impl NativeClipVision {
    pub fn from_recipe_state(
        source_form: crate::model_family::NativeModelRecipeSourceForm,
        configurations: Vec<NativeClipVisionRecipeConfiguration>,
        ordered_state: Vec<(String, Tensor)>,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipVisionError> {
        cancellation.check()?;
        let mut selected = None;
        for candidate in configurations {
            candidate.parser_limits.validate().map_err(|_| {
                ClipVisionError::InvalidConfiguration("CLIP vision parser limits are invalid")
            })?;
            if candidate.artifact_sha256.len() != 64
                || !candidate
                    .artifact_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                || format!("{:x}", Sha256::digest(&candidate.bytes)) != candidate.artifact_sha256
            {
                return Err(ClipVisionError::InvalidConfiguration(
                    "CLIP vision configuration digest is invalid",
                ));
            }
            let document =
                crate::formats::parse_strict_json(&candidate.bytes, &candidate.parser_limits)
                    .map_err(|_| {
                        ClipVisionError::InvalidConfiguration(
                            "CLIP vision configuration JSON is invalid",
                        )
                    })?;
            if document.get("weight_map").is_some() {
                continue;
            }
            let vision = document.get("vision_config").unwrap_or(&document);
            let semantic = [
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_hidden_layers",
                "image_size",
                "patch_size",
            ]
            .iter()
            .all(|field| vision.get(*field).is_some());
            if semantic && selected.replace((document, candidate)).is_some() {
                return Err(ClipVisionError::InvalidConfiguration(
                    "CLIP vision source has multiple semantic configurations",
                ));
            }
        }
        let (document, selected_configuration) =
            selected.ok_or(ClipVisionError::InvalidConfiguration(
                "CLIP vision source omitted its semantic configuration",
            ))?;
        let configuration = document.get("vision_config").unwrap_or(&document);
        let usize_field = |name: &'static str| {
            configuration
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(ClipVisionError::InvalidConfiguration(name))
        };
        let model_type = match configuration
            .get("model_type")
            .and_then(serde_json::Value::as_str)
        {
            Some("siglip") => ClipVisionModelType::Siglip,
            Some("siglip2") => ClipVisionModelType::Siglip2,
            Some("clip_vision_model") | Some("clip") | None => ClipVisionModelType::Clip,
            Some(_) => {
                return Err(ClipVisionError::InvalidConfiguration(
                    "CLIP vision model type is unsupported",
                ));
            }
        };
        let activation = match configuration
            .get("hidden_act")
            .and_then(serde_json::Value::as_str)
        {
            Some("gelu") => ClipVisionActivation::Gelu,
            Some("gelu_pytorch_tanh") => ClipVisionActivation::GeluTanh,
            Some("quick_gelu") | None => ClipVisionActivation::QuickGelu,
            Some(_) => {
                return Err(ClipVisionError::InvalidConfiguration(
                    "CLIP vision activation is unsupported",
                ));
            }
        };
        let mut state = BTreeMap::new();
        for (name, tensor) in ordered_state {
            let normalized = match source_form {
                crate::model_family::NativeModelRecipeSourceForm::Checkpoint => name
                    .strip_prefix("cond_stage_model.transformer.")
                    .or_else(|| name.strip_prefix("model."))
                    .unwrap_or(&name)
                    .to_owned(),
                crate::model_family::NativeModelRecipeSourceForm::Standalone => name,
            };
            if state.insert(normalized.clone(), tensor).is_some() {
                return Err(ClipVisionError::InvalidConfiguration(
                    "CLIP vision state contains a duplicate normalized key",
                ));
            }
        }
        let take = |name: &str| {
            state
                .get(name)
                .cloned()
                .ok_or(ClipVisionError::InvalidConfiguration(
                    "CLIP vision state is missing a required tensor",
                ))
        };
        let layer_count = usize_field("num_hidden_layers")?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(layer_count)
            .map_err(|_| ClipVisionError::Overflow("recipe layer allocation"))?;
        for layer in 0..layer_count {
            cancellation.check()?;
            let prefix = format!("vision_model.encoder.layers.{layer}");
            layers.push(ClipVisionLayerWeights {
                layer_norm_1_weight: take(&format!("{prefix}.layer_norm1.weight"))?,
                layer_norm_1_bias: take(&format!("{prefix}.layer_norm1.bias"))?,
                query_weight: take(&format!("{prefix}.self_attn.q_proj.weight"))?,
                query_bias: take(&format!("{prefix}.self_attn.q_proj.bias"))?,
                key_weight: take(&format!("{prefix}.self_attn.k_proj.weight"))?,
                key_bias: take(&format!("{prefix}.self_attn.k_proj.bias"))?,
                value_weight: take(&format!("{prefix}.self_attn.v_proj.weight"))?,
                value_bias: take(&format!("{prefix}.self_attn.v_proj.bias"))?,
                output_weight: take(&format!("{prefix}.self_attn.out_proj.weight"))?,
                output_bias: take(&format!("{prefix}.self_attn.out_proj.bias"))?,
                layer_norm_2_weight: take(&format!("{prefix}.layer_norm2.weight"))?,
                layer_norm_2_bias: take(&format!("{prefix}.layer_norm2.bias"))?,
                feed_forward_1_weight: take(&format!("{prefix}.mlp.fc1.weight"))?,
                feed_forward_1_bias: take(&format!("{prefix}.mlp.fc1.bias"))?,
                feed_forward_2_weight: take(&format!("{prefix}.mlp.fc2.weight"))?,
                feed_forward_2_bias: take(&format!("{prefix}.mlp.fc2.bias"))?,
            });
        }
        let optional = |name: &str| state.get(name).cloned();
        let projection_dimension = document
            .get("projection_dim")
            .or_else(|| configuration.get("projection_dim"))
            .and_then(serde_json::Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| ClipVisionError::Overflow("recipe projection dimension"))?;
        let llava_projection_dimension = configuration
            .get("llava_projection_dimension")
            .and_then(serde_json::Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| ClipVisionError::Overflow("recipe Llava projection dimension"))?;
        let configuration = ClipVisionConfiguration {
            model_type,
            dtype: DType::F32,
            device: DeviceId::CPU,
            hidden_size: usize_field("hidden_size")?,
            intermediate_size: usize_field("intermediate_size")?,
            attention_heads: usize_field("num_attention_heads")?,
            layer_count,
            image_size: usize_field("image_size")?,
            patch_size: usize_field("patch_size")?,
            num_channels: configuration
                .get("num_channels")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(3),
            max_num_patches: configuration
                .get("max_num_patches")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0),
            activation,
            projection_dimension,
            llava_projection_dimension,
        };
        let weights = ClipVisionWeights {
            patch_embedding_weight: take("vision_model.embeddings.patch_embedding.weight")?,
            patch_embedding_bias: optional("vision_model.embeddings.patch_embedding.bias"),
            class_embedding: optional("vision_model.embeddings.class_embedding"),
            position_embedding: take("vision_model.embeddings.position_embedding.weight")?,
            pre_layer_norm_weight: optional("vision_model.pre_layrnorm.weight"),
            pre_layer_norm_bias: optional("vision_model.pre_layrnorm.bias"),
            layers,
            post_layer_norm_weight: take("vision_model.post_layernorm.weight")?,
            post_layer_norm_bias: take("vision_model.post_layernorm.bias")?,
            visual_projection_weight: optional("visual_projection.weight"),
            llava_linear_1_weight: optional("llava_projector.linear_1.weight"),
            llava_linear_1_bias: optional("llava_projector.linear_1.bias"),
            llava_linear_2_weight: optional("llava_projector.linear_2.weight"),
            llava_linear_2_bias: optional("llava_projector.linear_2.bias"),
        };
        let mut model = Self::new_with_cancellation(configuration, weights, cancellation)?;
        let mut binding = Sha256::new();
        binding.update(b"zed.sim.clip-vision-recipe-binding.v1");
        binding.update(selected_configuration.artifact_sha256.as_bytes());
        let limits = &selected_configuration.parser_limits;
        binding.update(limits.version.to_be_bytes());
        binding.update(limits.manifest_bytes.to_be_bytes());
        binding.update(limits.maximum_depth.to_be_bytes());
        binding.update(limits.maximum_tensors.to_be_bytes());
        binding.update(limits.maximum_tensor_bytes.to_be_bytes());
        binding.update(limits.maximum_aggregate_tensor_bytes.to_be_bytes());
        binding.update(limits.maximum_name_bytes.to_be_bytes());
        binding.update(limits.maximum_archive_entries.to_be_bytes());
        binding.update(limits.maximum_metadata_values.to_be_bytes());
        model.recipe_binding_sha256 = Some(format!("{:x}", binding.finalize()));
        model.semantic_identity = clip_vision_identity(
            &model.configuration,
            &model.canonical_state,
            model.recipe_binding_sha256.as_deref(),
            cancellation,
        )?;
        model.validate(cancellation)?;
        Ok(model)
    }

    pub fn new(
        configuration: ClipVisionConfiguration,
        weights: ClipVisionWeights,
    ) -> Result<Self, ClipVisionError> {
        Self::new_with_cancellation(configuration, weights, &CancellationToken::default())
    }

    pub fn new_with_cancellation(
        configuration: ClipVisionConfiguration,
        weights: ClipVisionWeights,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipVisionError> {
        Self::new_with_cancellation_and_phase_hook(
            configuration,
            weights,
            cancellation,
            &mut |_, _| {},
        )
    }

    fn new_with_cancellation_and_phase_hook(
        configuration: ClipVisionConfiguration,
        weights: ClipVisionWeights,
        cancellation: &CancellationToken,
        phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
    ) -> Result<Self, ClipVisionError> {
        construction_checkpoint(cancellation, ClipVisionConstructionPhase::Entry, phase_hook)?;
        configuration.validate()?;
        if weights.layers.len() != configuration.layer_count {
            return Err(ClipVisionError::InvalidConfiguration(
                "configured layer count does not match the weight graph",
            ));
        }
        let canonical_state = clip_vision_canonical_state(&weights, cancellation, phase_hook)?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::SemanticDigest,
            phase_hook,
        )?;
        let semantic_identity =
            clip_vision_identity(&configuration, &canonical_state, None, cancellation)?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::ParameterProjection,
            phase_hook,
        )?;
        let stream = weights.position_embedding.descriptor().stream();
        validate_parameter_target(
            "position embedding",
            &weights.position_embedding,
            configuration.dtype,
            configuration.device,
            stream,
        )?;
        for (name, tensor) in [
            (
                "patch embedding weight",
                Some(&weights.patch_embedding_weight),
            ),
            (
                "patch embedding bias",
                weights.patch_embedding_bias.as_ref(),
            ),
            ("class embedding", weights.class_embedding.as_ref()),
            (
                "pre-layer norm weight",
                weights.pre_layer_norm_weight.as_ref(),
            ),
            ("pre-layer norm bias", weights.pre_layer_norm_bias.as_ref()),
            (
                "post-layer norm weight",
                Some(&weights.post_layer_norm_weight),
            ),
            ("post-layer norm bias", Some(&weights.post_layer_norm_bias)),
            (
                "visual projection",
                weights.visual_projection_weight.as_ref(),
            ),
            (
                "Llava linear 1 weight",
                weights.llava_linear_1_weight.as_ref(),
            ),
            ("Llava linear 1 bias", weights.llava_linear_1_bias.as_ref()),
            (
                "Llava linear 2 weight",
                weights.llava_linear_2_weight.as_ref(),
            ),
            ("Llava linear 2 bias", weights.llava_linear_2_bias.as_ref()),
        ] {
            if let Some(tensor) = tensor {
                construction_checkpoint(
                    cancellation,
                    ClipVisionConstructionPhase::ParameterProjection,
                    phase_hook,
                )?;
                validate_parameter_target(
                    name,
                    tensor,
                    configuration.dtype,
                    configuration.device,
                    stream,
                )?;
            }
        }

        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::PatchEmbedding,
            phase_hook,
        )?;
        let patch_width = configuration
            .num_channels
            .checked_mul(configuration.patch_size)
            .and_then(|value| value.checked_mul(configuration.patch_size))
            .ok_or(ClipVisionError::Overflow("patch width"))?;
        let has_patch_bias = weights.patch_embedding_bias.is_some();
        let mut patch_embedding = match configuration.model_type {
            ClipVisionModelType::Siglip2 => NativeModule::linear(
                "vision_model.embeddings.patch_embedding",
                patch_width,
                configuration.hidden_size,
                true,
                false,
            )?,
            ClipVisionModelType::Clip | ClipVisionModelType::Siglip => NativeModule::conv_2d(
                "vision_model.embeddings.patch_embedding",
                configuration.num_channels,
                configuration.hidden_size,
                [configuration.patch_size; 2],
                [configuration.patch_size; 2],
                [0; 2],
                [1; 2],
                1,
                configuration.model_type == ClipVisionModelType::Siglip,
                false,
            )?,
        };
        patch_embedding
            .load_dense_parameters(weights.patch_embedding_weight, weights.patch_embedding_bias)?;

        let expected_positions = match configuration.model_type {
            ClipVisionModelType::Clip => fixed_patch_count(&configuration)?
                .checked_add(1)
                .ok_or(ClipVisionError::Overflow("CLIP positions"))?,
            ClipVisionModelType::Siglip => fixed_patch_count(&configuration)?,
            ClipVisionModelType::Siglip2 => configuration.max_num_patches,
        };
        require_shape(
            "position embedding",
            &weights.position_embedding,
            &[
                usize_to_u64(expected_positions, "position count")?,
                usize_to_u64(configuration.hidden_size, "hidden size")?,
            ],
        )?;
        match configuration.model_type {
            ClipVisionModelType::Clip => {
                let class_embedding = weights.class_embedding.as_ref().ok_or(
                    ClipVisionError::InvalidConfiguration("CLIP requires a class embedding"),
                )?;
                require_shape(
                    "class embedding",
                    class_embedding,
                    &[usize_to_u64(configuration.hidden_size, "hidden size")?],
                )?;
                if has_patch_bias {
                    return Err(ClipVisionError::InvalidConfiguration(
                        "CLIP patch convolution must not have a bias",
                    ));
                }
            }
            ClipVisionModelType::Siglip => {
                if weights.class_embedding.is_some() || !has_patch_bias {
                    return Err(ClipVisionError::InvalidConfiguration(
                        "SigLIP requires a biased patch convolution and no class embedding",
                    ));
                }
            }
            ClipVisionModelType::Siglip2 => {
                if weights.class_embedding.is_some() || !has_patch_bias {
                    return Err(ClipVisionError::InvalidConfiguration(
                        "SigLIP2 requires a biased patch projection and no class embedding",
                    ));
                }
                let side = integer_square_root(expected_positions);
                if side.checked_mul(side) != Some(expected_positions) {
                    return Err(ClipVisionError::InvalidConfiguration(
                        "SigLIP2 position count must be a square",
                    ));
                }
            }
        }

        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::PreLayerNorm,
            phase_hook,
        )?;
        let pre_layer_norm = match configuration.model_type {
            ClipVisionModelType::Clip => Some(layer_norm_from_weights(
                "vision_model.pre_layrnorm",
                configuration.hidden_size,
                weights
                    .pre_layer_norm_weight
                    .ok_or(ClipVisionError::InvalidConfiguration(
                        "CLIP requires pre-layer norm weight",
                    ))?,
                weights
                    .pre_layer_norm_bias
                    .ok_or(ClipVisionError::InvalidConfiguration(
                        "CLIP requires pre-layer norm bias",
                    ))?,
            )?),
            ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => {
                if weights.pre_layer_norm_weight.is_some() || weights.pre_layer_norm_bias.is_some()
                {
                    return Err(ClipVisionError::InvalidConfiguration(
                        "SigLIP vision must not configure a pre-layer norm",
                    ));
                }
                None
            }
        };

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(configuration.layer_count)
            .map_err(|_| ClipVisionError::Overflow("vision layers"))?;
        for (index, weights) in weights.layers.into_iter().enumerate() {
            layers.push(build_layer(
                index,
                &configuration,
                weights,
                stream,
                cancellation,
                phase_hook,
            )?);
        }
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::PostLayerNorm,
            phase_hook,
        )?;
        let post_layer_norm = layer_norm_from_weights(
            "vision_model.post_layernorm",
            configuration.hidden_size,
            weights.post_layer_norm_weight,
            weights.post_layer_norm_bias,
        )?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::VisualProjection,
            phase_hook,
        )?;
        let visual_projection = optional_linear(
            "visual_projection",
            configuration.hidden_size,
            configuration.projection_dimension,
            weights.visual_projection_weight,
            None,
        )?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::LlavaProjection,
            phase_hook,
        )?;
        let llava_linear_1 = optional_linear(
            "multi_modal_projector.linear_1",
            configuration.hidden_size,
            configuration.llava_projection_dimension,
            weights.llava_linear_1_weight,
            weights.llava_linear_1_bias,
        )?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::LlavaProjection,
            phase_hook,
        )?;
        let llava_linear_2 = optional_linear(
            "multi_modal_projector.linear_2",
            configuration.llava_projection_dimension.unwrap_or(1),
            configuration.llava_projection_dimension,
            weights.llava_linear_2_weight,
            weights.llava_linear_2_bias,
        )?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::LlavaProjection,
            phase_hook,
        )?;
        let llava_activation =
            NativeModule::gelu("multi_modal_projector.gelu", GeluApproximation::None)?;

        let mut model = Self {
            configuration,
            stream,
            patch_embedding,
            class_embedding: weights.class_embedding,
            position_embedding: weights.position_embedding,
            pre_layer_norm,
            layers,
            post_layer_norm,
            visual_projection,
            llava_linear_1,
            llava_activation,
            llava_linear_2,
            canonical_state,
            recipe_binding_sha256: None,
            semantic_identity,
            module_state_digest_sha256: String::new(),
            training: false,
        };
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::ModuleDigest,
            phase_hook,
        )?;
        model.module_state_digest_sha256 = clip_vision_module_state_digest(&model, cancellation)?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::Validation,
            phase_hook,
        )?;
        model.validate(cancellation)?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::Return,
            phase_hook,
        )?;
        Ok(model)
    }

    pub fn reconstruct(&self, cancellation: &CancellationToken) -> Result<Self, ClipVisionError> {
        self.reconstruct_with_phase_hook(cancellation, &mut |_, _| {})
    }

    fn reconstruct_with_phase_hook(
        &self,
        cancellation: &CancellationToken,
        phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
    ) -> Result<Self, ClipVisionError> {
        construction_checkpoint(cancellation, ClipVisionConstructionPhase::Entry, phase_hook)?;
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::Validation,
            phase_hook,
        )?;
        self.validate(cancellation)?;
        let mut cursor = ClipVisionCanonicalStateCursor::new(&self.canonical_state);
        let patch_embedding_weight =
            cursor.take("patch_embedding.weight", cancellation, phase_hook)?;
        let patch_embedding_bias = cursor.take_optional(
            "patch_embedding.bias",
            self.configuration.model_type != ClipVisionModelType::Clip,
            cancellation,
            phase_hook,
        )?;
        let class_embedding = cursor.take_optional(
            "class_embedding",
            self.configuration.model_type == ClipVisionModelType::Clip,
            cancellation,
            phase_hook,
        )?;
        let position_embedding = cursor.take("position_embedding", cancellation, phase_hook)?;
        let pre_layer_norm_weight = cursor.take_optional(
            "pre_layer_norm.weight",
            self.configuration.model_type == ClipVisionModelType::Clip,
            cancellation,
            phase_hook,
        )?;
        let pre_layer_norm_bias = cursor.take_optional(
            "pre_layer_norm.bias",
            self.configuration.model_type == ClipVisionModelType::Clip,
            cancellation,
            phase_hook,
        )?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.configuration.layer_count)
            .map_err(|_| ClipVisionError::Overflow("reconstructed vision layers"))?;
        for index in 0..self.configuration.layer_count {
            let mut take = |suffix: &str| {
                cursor.take(
                    &format!("layers.{index}.{suffix}"),
                    cancellation,
                    phase_hook,
                )
            };
            layers.push(ClipVisionLayerWeights {
                layer_norm_1_weight: take("layer norm 1 weight")?,
                layer_norm_1_bias: take("layer norm 1 bias")?,
                query_weight: take("query weight")?,
                query_bias: take("query bias")?,
                key_weight: take("key weight")?,
                key_bias: take("key bias")?,
                value_weight: take("value weight")?,
                value_bias: take("value bias")?,
                output_weight: take("output weight")?,
                output_bias: take("output bias")?,
                layer_norm_2_weight: take("layer norm 2 weight")?,
                layer_norm_2_bias: take("layer norm 2 bias")?,
                feed_forward_1_weight: take("feed forward 1 weight")?,
                feed_forward_1_bias: take("feed forward 1 bias")?,
                feed_forward_2_weight: take("feed forward 2 weight")?,
                feed_forward_2_bias: take("feed forward 2 bias")?,
            });
        }
        let post_layer_norm_weight =
            cursor.take("post_layer_norm.weight", cancellation, phase_hook)?;
        let post_layer_norm_bias = cursor.take("post_layer_norm.bias", cancellation, phase_hook)?;
        let visual_projection_weight = cursor.take_optional(
            "visual_projection.weight",
            self.configuration.projection_dimension.is_some(),
            cancellation,
            phase_hook,
        )?;
        let has_llava = self.configuration.llava_projection_dimension.is_some();
        let llava_linear_1_weight =
            cursor.take_optional("llava_linear_1.weight", has_llava, cancellation, phase_hook)?;
        let llava_linear_1_bias =
            cursor.take_optional("llava_linear_1.bias", has_llava, cancellation, phase_hook)?;
        let llava_linear_2_weight =
            cursor.take_optional("llava_linear_2.weight", has_llava, cancellation, phase_hook)?;
        let llava_linear_2_bias =
            cursor.take_optional("llava_linear_2.bias", has_llava, cancellation, phase_hook)?;
        cursor.finish(cancellation, phase_hook)?;
        let mut reconstructed = Self::new_with_cancellation_and_phase_hook(
            self.configuration.clone(),
            ClipVisionWeights {
                patch_embedding_weight,
                patch_embedding_bias,
                class_embedding,
                position_embedding,
                pre_layer_norm_weight,
                pre_layer_norm_bias,
                layers,
                post_layer_norm_weight,
                post_layer_norm_bias,
                visual_projection_weight,
                llava_linear_1_weight,
                llava_linear_1_bias,
                llava_linear_2_weight,
                llava_linear_2_bias,
            },
            cancellation,
            phase_hook,
        )?;
        reconstructed.recipe_binding_sha256 = self.recipe_binding_sha256.clone();
        reconstructed.semantic_identity = clip_vision_identity(
            &reconstructed.configuration,
            &reconstructed.canonical_state,
            reconstructed.recipe_binding_sha256.as_deref(),
            cancellation,
        )?;
        reconstructed.set_training(self.training);
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::Validation,
            phase_hook,
        )?;
        if reconstructed.configuration != self.configuration
            || reconstructed.semantic_identity != self.semantic_identity
            || reconstructed.module_state_digest_sha256 != self.module_state_digest_sha256
            || reconstructed.resident_parts()? != self.resident_parts()?
            || reconstructed.canonical_state.len() != self.canonical_state.len()
        {
            return Err(ClipVisionError::SemanticIdentityChanged);
        }
        for ((actual_name, actual), (expected_name, expected)) in reconstructed
            .canonical_state
            .iter()
            .zip(&self.canonical_state)
        {
            construction_checkpoint(
                cancellation,
                ClipVisionConstructionPhase::Validation,
                phase_hook,
            )?;
            if actual_name != expected_name
                || actual.descriptor() != expected.descriptor()
                || actual.storage_id() != expected.storage_id()
                || actual.contiguous_bytes()? != expected.contiguous_bytes()?
            {
                return Err(ClipVisionError::SemanticIdentityChanged);
            }
        }
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::Return,
            phase_hook,
        )?;
        Ok(reconstructed)
    }

    pub fn configuration(&self) -> &ClipVisionConfiguration {
        &self.configuration
    }

    pub fn dtype(&self) -> DType {
        self.configuration.dtype
    }

    pub fn device(&self) -> DeviceId {
        self.configuration.device
    }

    pub const fn is_training(&self) -> bool {
        self.training
    }

    pub fn train(&mut self) {
        self.set_training(true);
    }

    pub fn eval(&mut self) {
        self.set_training(false);
    }

    pub fn semantic_identity(&self) -> &NativeModelResourceIdentity {
        &self.semantic_identity
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        self.semantic_identity.digest_sha256()
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, ClipVisionError> {
        self.resident_parts()?
            .tensor_allocations()
            .iter()
            .try_fold(0_u64, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes())
                    .ok_or(ClipVisionError::Overflow("resident storage bytes"))
            })
    }

    fn resident_owned_bytes(&self) -> Result<u64, ClipVisionError> {
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| ClipVisionError::Overflow("resident object bytes"))?;
        for allocation in [
            self.layers
                .capacity()
                .checked_mul(mem::size_of::<ClipVisionLayer>()),
            self.canonical_state
                .capacity()
                .checked_mul(mem::size_of::<(String, Tensor)>()),
        ] {
            bytes = bytes
                .checked_add(
                    u64::try_from(
                        allocation.ok_or(ClipVisionError::Overflow("resident allocations"))?,
                    )
                    .map_err(|_| ClipVisionError::Overflow("resident allocations"))?,
                )
                .ok_or(ClipVisionError::Overflow("resident allocations"))?;
        }
        for (name, _) in &self.canonical_state {
            bytes = bytes
                .checked_add(
                    u64::try_from(name.capacity())
                        .map_err(|_| ClipVisionError::Overflow("resident state names"))?,
                )
                .ok_or(ClipVisionError::Overflow("resident state names"))?;
        }
        bytes = bytes
            .checked_add(
                self.recipe_binding_sha256
                    .as_ref()
                    .and_then(|binding| u64::try_from(binding.capacity()).ok())
                    .unwrap_or(0),
            )
            .ok_or(ClipVisionError::Overflow("resident recipe binding"))?;
        bytes = bytes
            .checked_add(
                self.semantic_identity
                    .resident_owned_bytes()
                    .map_err(|_| ClipVisionError::Overflow("resident identity"))?,
            )
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(self.module_state_digest_sha256.capacity()).ok()?)
            })
            .ok_or(ClipVisionError::Overflow("resident identity"))?;
        Ok(bytes)
    }

    pub fn resident_parts(&self) -> Result<ClipVisionResidentParts, ClipVisionError> {
        clip_vision_resident_parts(
            self.resident_owned_bytes()?,
            self.canonical_state.iter().map(|(_, tensor)| tensor),
        )
    }

    pub fn resident_bytes(&self) -> Result<u64, ClipVisionError> {
        self.resident_parts()?.resident_bytes()
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), ClipVisionError> {
        cancellation.check()?;
        self.configuration.validate()?;
        self.semantic_identity.validate()?;
        let expected_identity = clip_vision_identity(
            &self.configuration,
            &self.canonical_state,
            self.recipe_binding_sha256.as_deref(),
            cancellation,
        )?;
        if self.semantic_identity != expected_identity
            || self.module_state_digest_sha256
                != clip_vision_module_state_digest(self, cancellation)?
        {
            return Err(ClipVisionError::SemanticIdentityChanged);
        }
        self.resident_bytes()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn execution_session(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeClipVisionExecutionSession, ClipVisionError> {
        self.validate(cancellation)?;
        if self.training {
            return Err(ClipVisionError::EvaluationRequired);
        }
        Ok(NativeClipVisionExecutionSession {
            model: self.clone(),
        })
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in [
            Some(&mut self.patch_embedding),
            self.pre_layer_norm.as_mut(),
            Some(&mut self.post_layer_norm),
            self.visual_projection.as_mut(),
            self.llava_linear_1.as_mut(),
            Some(&mut self.llava_activation),
            self.llava_linear_2.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            module.set_training(training);
        }
        for layer in &mut self.layers {
            for module in [
                &mut layer.layer_norm_1,
                &mut layer.attention,
                &mut layer.layer_norm_2,
                &mut layer.feed_forward_1,
                &mut layer.activation,
                &mut layer.feed_forward_2,
            ] {
                module.set_training(training);
            }
        }
    }

    fn execution_requirements(&self) -> NativeExecutionRequirements {
        let mut requirements = clip_vision_tensor_requirements(false);
        for module in [
            Some(&self.patch_embedding),
            self.pre_layer_norm.as_ref(),
            Some(&self.post_layer_norm),
            self.visual_projection.as_ref(),
            self.llava_linear_1.as_ref(),
            Some(&self.llava_activation),
            self.llava_linear_2.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            requirements.extend(module.execution_requirements(DType::F32).iter());
        }
        for layer in &self.layers {
            for module in [
                &layer.layer_norm_1,
                &layer.attention,
                &layer.layer_norm_2,
                &layer.feed_forward_1,
                &layer.activation,
                &layer.feed_forward_2,
            ] {
                requirements.extend(module.execution_requirements(DType::F32).iter());
            }
            if layer.quick_gelu {
                requirements.extend([
                    OperationSupport::unary_input(
                        UnaryOperation::Exponential,
                        DType::F32,
                        Layout::Contiguous,
                    ),
                    OperationSupport::unary_output(
                        UnaryOperation::Exponential,
                        DType::F32,
                        Layout::Contiguous,
                    ),
                ]);
            }
        }
        requirements
    }

    pub fn preprocess(
        &self,
        backend: &CpuBackend,
        image: &Tensor,
        crop: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, ClipVisionError> {
        clip_vision_tensor_requirements(true).admit_backend_target(
            backend,
            self.configuration.device,
            self.configuration.dtype,
            Layout::Contiguous,
            self.stream,
            context,
        )?;
        let image = match self.configuration.model_type {
            ClipVisionModelType::Clip => clip_preprocess_with_context(
                backend,
                image,
                self.configuration.image_size,
                DEFAULT_CLIP_MEAN,
                DEFAULT_CLIP_STD,
                crop,
                context,
            ),
            ClipVisionModelType::Siglip => clip_preprocess_with_context(
                backend,
                image,
                self.configuration.image_size,
                DEFAULT_SIGLIP_MEAN,
                DEFAULT_SIGLIP_STD,
                crop,
                context,
            ),
            ClipVisionModelType::Siglip2 => siglip2_preprocess_with_context(
                backend,
                image,
                self.configuration.image_size,
                self.configuration.patch_size,
                self.configuration.max_num_patches,
                DEFAULT_SIGLIP_MEAN,
                DEFAULT_SIGLIP_STD,
                crop,
                context,
            ),
        }?;
        if image.descriptor().dtype() == self.configuration.dtype
            && image.descriptor().device() == self.configuration.device
        {
            Ok(image)
        } else {
            Ok(cast_to_with_context_exact_native(
                backend,
                &image,
                self.configuration.dtype,
                self.configuration.device,
                false,
                false,
                context,
            )?)
        }
    }

    pub fn forward(
        &mut self,
        backend: &CpuBackend,
        pixel_values: &Tensor,
        intermediate: ClipVisionIntermediate,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipVisionOutput, ClipVisionError> {
        Ok(self
            .forward_checked(backend, pixel_values, intermediate, context)?
            .output)
    }

    pub(crate) fn forward_checked(
        &mut self,
        backend: &CpuBackend,
        pixel_values: &Tensor,
        intermediate: ClipVisionIntermediate,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeClipVisionCheckedForward, ClipVisionError> {
        self.execution_requirements().admit_backend_target(
            backend,
            self.configuration.device,
            self.configuration.dtype,
            Layout::Contiguous,
            self.stream,
            context,
        )?;
        require_execution_tensor(
            "pixel values",
            pixel_values,
            self.configuration.dtype,
            self.configuration.device,
            self.stream,
            context,
        )?;
        let shape = pixel_values.descriptor().shape();
        if shape.len() != 4 || shape[0] == 0 || shape[1] != 3 || shape[2] == 0 || shape[3] == 0 {
            return Err(ClipVisionError::InvalidInput(
                "pixel values must have nonempty [batch, 3, height, width] shape",
            ));
        }
        let image_sizes = vec![
            [shape[1], shape[2], shape[3]];
            u64_to_usize(shape[0], "CLIP vision output batch")?
        ];
        let patch_size = usize_to_u64(self.configuration.patch_size, "patch size")?;
        if !shape[2].is_multiple_of(patch_size) || !shape[3].is_multiple_of(patch_size) {
            return Err(ClipVisionError::InvalidInput(
                "pixel dimensions must be divisible by patch size",
            ));
        }
        if self.configuration.model_type != ClipVisionModelType::Siglip2
            && (shape[2] != usize_to_u64(self.configuration.image_size, "image size")?
                || shape[3] != usize_to_u64(self.configuration.image_size, "image size")?)
        {
            return Err(ClipVisionError::InvalidInput(
                "fixed-resolution CLIP vision input does not match configured image size",
            ));
        }
        let patch_count = shape[2]
            .checked_div(patch_size)
            .and_then(|height| {
                shape[3]
                    .checked_div(patch_size)
                    .and_then(|width| height.checked_mul(width))
            })
            .ok_or(ClipVisionError::Overflow("input patch count"))?;
        if self.configuration.model_type == ClipVisionModelType::Siglip2
            && patch_count
                > usize_to_u64(self.configuration.max_num_patches, "maximum patch count")?
        {
            return Err(ClipVisionError::InvalidInput(
                "SigLIP2 input exceeds the configured maximum patch count",
            ));
        }
        let selected_layer = resolve_intermediate(intermediate, self.layers.len())?;
        let mut hidden = self.patch_embeddings(backend, pixel_values, context)?;
        if let Some(pre_layer_norm) = self.pre_layer_norm.as_mut() {
            hidden = pre_layer_norm.forward_with_context(backend, &hidden, context)?;
        }
        let mut captured = None;
        let mut all = match intermediate {
            ClipVisionIntermediate::All => {
                let mut layers = Vec::new();
                layers
                    .try_reserve_exact(self.layers.len())
                    .map_err(|_| ClipVisionError::Overflow("all-layer capture"))?;
                Some(layers)
            }
            _ => None,
        };
        for (index, layer) in self.layers.iter_mut().enumerate() {
            context.check()?;
            hidden = layer.forward(backend, &hidden, context)?;
            if selected_layer == Some(index) {
                captured = Some(hidden.clone());
            }
            if let Some(all) = all.as_mut() {
                all.push(hidden.clone());
            }
        }
        let intermediate = match all {
            Some(all) => Some(stack_layers(backend, &all, context)?),
            None => captured,
        };
        let (last_hidden_state, pooled) = match self.configuration.model_type {
            ClipVisionModelType::Clip => {
                let pooled = select_token(backend, &hidden, 0, context)?;
                let pooled = self
                    .post_layer_norm
                    .forward_with_context(backend, &pooled, context)?;
                (hidden, pooled)
            }
            ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => {
                let normalized = self
                    .post_layer_norm
                    .forward_with_context(backend, &hidden, context)?;
                (normalized.clone(), normalized)
            }
        };
        let pooled_hidden = pooled.clone();
        let image_embeds = match self.visual_projection.as_mut() {
            Some(projection) => projection.forward_with_context(backend, &pooled, context)?,
            None => pooled,
        };
        let projected_intermediate =
            match (self.llava_linear_1.as_mut(), self.llava_linear_2.as_mut()) {
                (Some(linear_1), Some(linear_2)) => {
                    let intermediate = intermediate
                        .as_ref()
                        .ok_or(ClipVisionError::MissingLlavaIntermediate)?;
                    if intermediate.descriptor().rank() != 3 {
                        return Err(ClipVisionError::InvalidInput(
                            "Llava projection requires one selected intermediate layer",
                        ));
                    }
                    let intermediate = drop_first_token(backend, intermediate, context)?;
                    let projected =
                        linear_1.forward_with_context(backend, &intermediate, context)?;
                    let projected = self
                        .llava_activation
                        .forward_with_context(backend, &projected, context)?;
                    Some(linear_2.forward_with_context(backend, &projected, context)?)
                }
                (None, None) => None,
                _ => {
                    return Err(ClipVisionError::InvalidConfiguration(
                        "Llava projection module graph is incomplete",
                    ));
                }
            };
        context.check()?;
        let output = ClipVisionOutput::checked(
            last_hidden_state,
            intermediate,
            image_embeds,
            projected_intermediate,
            image_sizes,
        )?;
        Ok(NativeClipVisionCheckedForward {
            output,
            pooled_hidden,
        })
    }

    fn patch_embeddings(
        &mut self,
        backend: &CpuBackend,
        pixel_values: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, ClipVisionError> {
        let shape = pixel_values.descriptor().shape();
        let batch = u64_to_usize(shape[0], "batch")?;
        let height = u64_to_usize(shape[2], "height")?;
        let width = u64_to_usize(shape[3], "width")?;
        let patch = self.configuration.patch_size;
        let grid_height = height / patch;
        let grid_width = width / patch;
        let embeds = match self.configuration.model_type {
            ClipVisionModelType::Siglip2 => {
                let patches =
                    patchify(backend, pixel_values, batch, height, width, patch, context)?;
                self.patch_embedding
                    .forward_with_context(backend, &patches, context)?
            }
            ClipVisionModelType::Clip | ClipVisionModelType::Siglip => {
                let convolution =
                    self.patch_embedding
                        .forward_with_context(backend, pixel_values, context)?;
                flatten_convolution_patches(
                    backend,
                    &convolution,
                    self.configuration.hidden_size,
                    context,
                )?
            }
        };
        match self.configuration.model_type {
            ClipVisionModelType::Siglip2 => siglip2_position_embedding(
                backend,
                &embeds,
                &self.position_embedding,
                grid_height,
                grid_width,
                context,
            ),
            ClipVisionModelType::Clip | ClipVisionModelType::Siglip => {
                let embeds = match &self.class_embedding {
                    Some(class_embedding) => {
                        prepend_class_embedding(backend, &embeds, class_embedding, context)?
                    }
                    None => embeds,
                };
                add_position_embedding(backend, &embeds, &self.position_embedding, context)
            }
        }
    }
}

impl NativeClipVisionExecutionSession {
    pub fn semantic_identity(&self) -> &NativeModelResourceIdentity {
        self.model.semantic_identity()
    }

    pub fn forward(
        &mut self,
        backend: &CpuBackend,
        pixel_values: &Tensor,
        intermediate: ClipVisionIntermediate,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipVisionOutput, ClipVisionError> {
        self.model
            .forward(backend, pixel_values, intermediate, context)
    }
}

impl ClipVisionLayer {
    fn forward(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, ClipVisionError> {
        let normalized = self
            .layer_norm_1
            .forward_with_context(backend, input, context)?;
        // The canonical nn.MultiheadAttention owner takes [tokens, batch, hidden], while CLIP stores [batch, tokens, hidden].
        let normalized =
            tensor_permute_exact_native(&normalized, &[1, 0, 2], context.cancellation)?;
        let normalized = contiguous_with_context_exact_native(
            backend,
            &normalized,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        )?;
        let attention = self.attention.forward_attention_with_context(
            backend,
            &normalized,
            &normalized,
            &normalized,
            context,
        )?;
        let attention = tensor_permute_exact_native(&attention, &[1, 0, 2], context.cancellation)?;
        let attention = contiguous_with_context_exact_native(
            backend,
            &attention,
            MemoryFormatReference::Layout(Layout::Contiguous),
            context,
        )?;
        let hidden = add_exact(backend, input, &attention, context)?;
        let normalized = self
            .layer_norm_2
            .forward_with_context(backend, &hidden, context)?;
        let feed_forward =
            self.feed_forward_1
                .forward_with_context(backend, &normalized, context)?;
        let feed_forward = if self.quick_gelu {
            quick_gelu(backend, &feed_forward, context)?
        } else {
            self.activation
                .forward_with_context(backend, &feed_forward, context)?
        };
        let feed_forward =
            self.feed_forward_2
                .forward_with_context(backend, &feed_forward, context)?;
        add_exact(backend, &hidden, &feed_forward, context)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn clip_preprocess_with_context(
    backend: &CpuBackend,
    image: &Tensor,
    size: usize,
    mean: [f32; 3],
    standard_deviation: [f32; 3],
    crop: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    clip_vision_tensor_requirements(true).admit_backend_target(
        backend,
        image.descriptor().device(),
        image.descriptor().dtype(),
        image.descriptor().layout(),
        image.descriptor().stream(),
        context,
    )?;
    if size == 0 {
        return Err(ClipVisionError::InvalidConfiguration(
            "CLIP preprocessing size must be nonzero",
        ));
    }
    validate_normalization(mean, standard_deviation)?;
    let (_, height, width, _) = bhwc_shape(image, context)?;
    let mut image = bhwc_rgb_to_nchw(backend, image, context)?;
    if height != size || width != size {
        let (resize_height, resize_width) = if crop {
            let scale = size as f64 / height.min(width) as f64;
            (
                (scale * height as f64).round_ties_even().max(1.0) as usize,
                (scale * width as f64).round_ties_even().max(1.0) as usize,
            )
        } else {
            (size, size)
        };
        let mut output_shape = image.descriptor().shape().to_vec();
        output_shape[2] = usize_to_u64(resize_height, "resize height")?;
        output_shape[3] = usize_to_u64(resize_width, "resize width")?;
        let output_descriptor =
            TensorDescriptor::contiguous(output_shape, DType::F32, DeviceId::CPU, context.stream)?;
        let (resized, event) = backend.resize(
            ResizeSpec {
                width: usize_to_u64(resize_width, "resize width")?,
                height: usize_to_u64(resize_height, "resize height")?,
                mode: ResizeMode::Bicubic,
                crop: ResizeCrop::Disabled,
                antialias: true,
                align_corners: false,
            },
            &image,
            output_descriptor,
            context,
        )?;
        backend.wait_event(event, context)?;
        image = if crop {
            let offset_y = (resize_height - size) / 2;
            let offset_x = (resize_width - size) / 2;
            let cropped = narrow_method_exact_native(
                &resized,
                2,
                usize_to_i64(offset_y, "crop y offset")?,
                usize_to_u64(size, "crop height")?,
                context.cancellation,
            )?;
            let cropped = narrow_method_exact_native(
                &cropped,
                3,
                usize_to_i64(offset_x, "crop x offset")?,
                usize_to_u64(size, "crop width")?,
                context.cancellation,
            )?;
            contiguous_with_context_exact_native(
                backend,
                &cropped,
                MemoryFormatReference::Layout(Layout::Contiguous),
                context,
            )?
        } else {
            resized
        };
    }
    quantize_and_normalize(backend, &image, mean, standard_deviation, context)
}

#[allow(clippy::too_many_arguments)]
pub fn siglip2_preprocess_with_context(
    backend: &CpuBackend,
    image: &Tensor,
    size: usize,
    patch_size: usize,
    max_num_patches: usize,
    mean: [f32; 3],
    standard_deviation: [f32; 3],
    crop: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    clip_vision_tensor_requirements(true).admit_backend_target(
        backend,
        image.descriptor().device(),
        image.descriptor().dtype(),
        image.descriptor().layout(),
        image.descriptor().stream(),
        context,
    )?;
    if size > 0 {
        return clip_preprocess_with_context(
            backend,
            image,
            size,
            mean,
            standard_deviation,
            crop,
            context,
        );
    }
    if patch_size == 0 || max_num_patches == 0 {
        return Err(ClipVisionError::InvalidConfiguration(
            "flexible SigLIP2 preprocessing requires patch size and patch count",
        ));
    }
    validate_normalization(mean, standard_deviation)?;
    let (_, height, width, _) = bhwc_shape(image, context)?;
    let (height, width) = siglip2_flex_resolution(height, width, patch_size, max_num_patches)?;
    let image = bhwc_rgb_to_nchw(backend, image, context)?;
    let image = resize_with_context_exact_native(
        backend,
        &image,
        usize_to_u64(height, "SigLIP2 height")?,
        usize_to_u64(width, "SigLIP2 width")?,
        ResizeMode::Bilinear,
        true,
        context,
    )?;
    quantize_and_normalize(backend, &image, mean, standard_deviation, context)
}

pub fn siglip2_flex_resolution(
    original_height: usize,
    original_width: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> Result<(usize, usize), ClipVisionError> {
    if original_height == 0 || original_width == 0 || patch_size == 0 || max_num_patches == 0 {
        return Err(ClipVisionError::InvalidInput(
            "flexible resolution dimensions and limits must be nonzero",
        ));
    }
    fn scale_dimension(size: usize, scale: f64, patch: usize) -> Result<usize, ClipVisionError> {
        let scaled = (size as f64 * scale / patch as f64).ceil() * patch as f64;
        if !scaled.is_finite() || scaled > usize::MAX as f64 {
            return Err(ClipVisionError::Overflow("flexible resolution"));
        }
        Ok((scaled as usize).max(patch))
    }
    let mut low = 1.0e-6_f64;
    let mut high = 100.0_f64;
    while high - low >= 1.0e-5 {
        let middle = (low + high) / 2.0;
        let height = scale_dimension(original_height, middle, patch_size)?;
        let width = scale_dimension(original_width, middle, patch_size)?;
        let patches = height
            .checked_div(patch_size)
            .and_then(|height| {
                width
                    .checked_div(patch_size)
                    .and_then(|width| height.checked_mul(width))
            })
            .ok_or(ClipVisionError::Overflow("flexible patch count"))?;
        if patches <= max_num_patches {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok((
        scale_dimension(original_height, low, patch_size)?,
        scale_dimension(original_width, low, patch_size)?,
    ))
}

fn build_layer(
    index: usize,
    configuration: &ClipVisionConfiguration,
    weights: ClipVisionLayerWeights,
    stream: StreamId,
    cancellation: &CancellationToken,
    phase_hook: &mut impl FnMut(ClipVisionConstructionPhase, &CancellationToken),
) -> Result<ClipVisionLayer, ClipVisionError> {
    for (name, tensor) in layer_tensors(&weights) {
        construction_checkpoint(
            cancellation,
            ClipVisionConstructionPhase::ParameterProjection,
            phase_hook,
        )?;
        validate_parameter_target(
            name,
            tensor,
            configuration.dtype,
            configuration.device,
            stream,
        )?;
    }
    let prefix = format!("vision_model.encoder.layers.{index}");
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::LayerNorm1(index),
        phase_hook,
    )?;
    let layer_norm_1 = layer_norm_from_weights(
        format!("{prefix}.layer_norm1"),
        configuration.hidden_size,
        weights.layer_norm_1_weight,
        weights.layer_norm_1_bias,
    )?;
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::AttentionQuery(index),
        phase_hook,
    )?;
    let mut attention = NativeModule::multihead_attention(
        format!("{prefix}.self_attn"),
        configuration.hidden_size,
        configuration.attention_heads,
        true,
        false,
    )?;
    for (phase, name, weight, bias) in [
        (
            ClipVisionConstructionPhase::AttentionQuery(index),
            "q_proj",
            weights.query_weight,
            weights.query_bias,
        ),
        (
            ClipVisionConstructionPhase::AttentionKey(index),
            "k_proj",
            weights.key_weight,
            weights.key_bias,
        ),
        (
            ClipVisionConstructionPhase::AttentionValue(index),
            "v_proj",
            weights.value_weight,
            weights.value_bias,
        ),
        (
            ClipVisionConstructionPhase::AttentionOutput(index),
            "out_proj",
            weights.output_weight,
            weights.output_bias,
        ),
    ] {
        construction_checkpoint(cancellation, phase, phase_hook)?;
        attention
            .child_mut(name)
            .ok_or(ClipVisionError::InvalidConfiguration(
                "canonical attention projection is missing",
            ))?
            .load_dense_parameters(weight, Some(bias))?;
    }
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::LayerNorm2(index),
        phase_hook,
    )?;
    let layer_norm_2 = layer_norm_from_weights(
        format!("{prefix}.layer_norm2"),
        configuration.hidden_size,
        weights.layer_norm_2_weight,
        weights.layer_norm_2_bias,
    )?;
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::MlpInput(index),
        phase_hook,
    )?;
    let mut feed_forward_1 = NativeModule::linear(
        format!("{prefix}.mlp.fc1"),
        configuration.hidden_size,
        configuration.intermediate_size,
        true,
        false,
    )?;
    feed_forward_1.load_dense_parameters(
        weights.feed_forward_1_weight,
        Some(weights.feed_forward_1_bias),
    )?;
    let approximation = match configuration.activation {
        ClipVisionActivation::Gelu => GeluApproximation::None,
        ClipVisionActivation::GeluTanh => GeluApproximation::Tanh,
        ClipVisionActivation::QuickGelu => GeluApproximation::None,
    };
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::MlpActivation(index),
        phase_hook,
    )?;
    let activation = NativeModule::gelu(format!("{prefix}.mlp.activation"), approximation)?;
    construction_checkpoint(
        cancellation,
        ClipVisionConstructionPhase::MlpOutput(index),
        phase_hook,
    )?;
    let mut feed_forward_2 = NativeModule::linear(
        format!("{prefix}.mlp.fc2"),
        configuration.intermediate_size,
        configuration.hidden_size,
        true,
        false,
    )?;
    feed_forward_2.load_dense_parameters(
        weights.feed_forward_2_weight,
        Some(weights.feed_forward_2_bias),
    )?;
    Ok(ClipVisionLayer {
        layer_norm_1,
        attention,
        layer_norm_2,
        feed_forward_1,
        activation,
        feed_forward_2,
        quick_gelu: configuration.activation == ClipVisionActivation::QuickGelu,
    })
}

fn layer_tensors(weights: &ClipVisionLayerWeights) -> [(&'static str, &Tensor); 16] {
    [
        ("layer norm 1 weight", &weights.layer_norm_1_weight),
        ("layer norm 1 bias", &weights.layer_norm_1_bias),
        ("query weight", &weights.query_weight),
        ("query bias", &weights.query_bias),
        ("key weight", &weights.key_weight),
        ("key bias", &weights.key_bias),
        ("value weight", &weights.value_weight),
        ("value bias", &weights.value_bias),
        ("output weight", &weights.output_weight),
        ("output bias", &weights.output_bias),
        ("layer norm 2 weight", &weights.layer_norm_2_weight),
        ("layer norm 2 bias", &weights.layer_norm_2_bias),
        ("feed forward 1 weight", &weights.feed_forward_1_weight),
        ("feed forward 1 bias", &weights.feed_forward_1_bias),
        ("feed forward 2 weight", &weights.feed_forward_2_weight),
        ("feed forward 2 bias", &weights.feed_forward_2_bias),
    ]
}

fn layer_norm_from_weights(
    name: impl Into<String>,
    width: usize,
    weight: Tensor,
    bias: Tensor,
) -> Result<NativeModule, ClipVisionError> {
    let mut module = NativeModule::layer_norm(name, vec![width], 1.0e-5, true, true, false)?;
    module.load_dense_parameters(weight, Some(bias))?;
    Ok(module)
}

fn optional_linear(
    name: &'static str,
    input: usize,
    output: Option<usize>,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
) -> Result<Option<NativeModule>, ClipVisionError> {
    match (output, weight, bias) {
        (None, None, None) => Ok(None),
        (Some(output), Some(weight), bias) => {
            let mut module = NativeModule::linear(name, input, output, bias.is_some(), false)?;
            module.load_dense_parameters(weight, bias)?;
            Ok(Some(module))
        }
        _ => Err(ClipVisionError::InvalidConfiguration(
            "projection configuration and weights do not match",
        )),
    }
}

fn fixed_patch_count(configuration: &ClipVisionConfiguration) -> Result<usize, ClipVisionError> {
    let side = configuration.image_size / configuration.patch_size;
    side.checked_mul(side)
        .ok_or(ClipVisionError::Overflow("fixed patch count"))
}

fn resolve_intermediate(
    requested: ClipVisionIntermediate,
    layers: usize,
) -> Result<Option<usize>, ClipVisionError> {
    let ClipVisionIntermediate::Layer(requested) = requested else {
        return Ok(None);
    };
    let resolved = if requested < 0 {
        isize::try_from(layers)
            .map_err(|_| ClipVisionError::Overflow("layer count"))?
            .checked_add(requested)
    } else {
        Some(requested)
    }
    .ok_or(ClipVisionError::IntermediateOutOfRange {
        requested,
        available: layers,
    })?;
    let resolved =
        usize::try_from(resolved).map_err(|_| ClipVisionError::IntermediateOutOfRange {
            requested,
            available: layers,
        })?;
    if resolved >= layers {
        return Err(ClipVisionError::IntermediateOutOfRange {
            requested,
            available: layers,
        });
    }
    Ok(Some(resolved))
}

fn bhwc_shape(
    image: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(usize, usize, usize, usize), ClipVisionError> {
    require_execution_tensor(
        "input image",
        image,
        DType::F32,
        DeviceId::CPU,
        context.stream,
        context,
    )?;
    let [batch, height, width, channels] = image.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "preprocessing expects a [batch, height, width, channels] tensor",
        ));
    };
    if *batch == 0 || *height == 0 || *width == 0 || *channels < 3 {
        return Err(ClipVisionError::InvalidInput(
            "preprocessing requires nonempty images with at least three channels",
        ));
    }
    Ok((
        u64_to_usize(*batch, "batch")?,
        u64_to_usize(*height, "height")?,
        u64_to_usize(*width, "width")?,
        u64_to_usize(*channels, "channels")?,
    ))
}

fn bhwc_rgb_to_nchw(
    backend: &CpuBackend,
    image: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    bhwc_shape(image, context)?;
    let rgb = narrow_method_exact_native(image, 3, 0, 3, context.cancellation)?;
    let nchw = tensor_permute_exact_native(&rgb, &[0, 3, 1, 2], context.cancellation)?;
    Ok(contiguous_with_context_exact_native(
        backend,
        &nchw,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?)
}

fn quantize_and_normalize(
    backend: &CpuBackend,
    image: &Tensor,
    mean: [f32; 3],
    standard_deviation: [f32; 3],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let shape = image.descriptor().shape();
    if shape.len() != 4 || shape[1] != 3 {
        return Err(ClipVisionError::InvalidInput(
            "normalization requires NCHW RGB input",
        ));
    }
    let scaled = binary_scalar_exact(backend, image, BinaryOperation::Multiply, 255.0, context)?;
    let clamped = binary_scalar_exact(backend, &scaled, BinaryOperation::Maximum, 0.0, context)?;
    let clamped = binary_scalar_exact(backend, &clamped, BinaryOperation::Minimum, 255.0, context)?;
    let rounded = unary_exact(backend, &clamped, UnaryOperation::Round, context)?;
    let unit = binary_scalar_exact(backend, &rounded, BinaryOperation::Divide, 255.0, context)?;
    Ok(normalize_with_context_exact_native(
        backend,
        &unit,
        &mean,
        &standard_deviation,
        context,
    )?)
}

fn validate_normalization(
    mean: [f32; 3],
    standard_deviation: [f32; 3],
) -> Result<(), ClipVisionError> {
    if mean.iter().any(|value| !value.is_finite())
        || standard_deviation
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ClipVisionError::InvalidConfiguration(
            "normalization statistics must be finite and standard deviations positive",
        ));
    }
    Ok(())
}

fn patchify(
    backend: &CpuBackend,
    image: &Tensor,
    batch: usize,
    height: usize,
    width: usize,
    patch: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let grid_height = height / patch;
    let grid_width = width / patch;
    let patch_width = 3_usize
        .checked_mul(patch)
        .and_then(|value| value.checked_mul(patch))
        .ok_or(ClipVisionError::Overflow("patch width"))?;
    let patch_count = grid_height
        .checked_mul(grid_width)
        .ok_or(ClipVisionError::Overflow("patch count"))?;
    let image = contiguous_with_context_exact_native(
        backend,
        image,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    let channel_stride = height
        .checked_mul(width)
        .ok_or(ClipVisionError::Overflow("patch channel stride"))?;
    let batch_stride = channel_stride
        .checked_mul(3)
        .ok_or(ClipVisionError::Overflow("patch batch stride"))?;
    let view = image.view(
        TensorDescriptor::new_strided(
            vec![
                usize_to_u64(batch, "batch")?,
                usize_to_u64(grid_height, "grid height")?,
                usize_to_u64(grid_width, "grid width")?,
                usize_to_u64(patch, "patch height")?,
                usize_to_u64(patch, "patch width")?,
                3,
            ],
            vec![
                usize_to_i64(batch_stride, "patch batch stride")?,
                usize_to_i64(
                    patch
                        .checked_mul(width)
                        .ok_or(ClipVisionError::Overflow("patch row stride"))?,
                    "patch row stride",
                )?,
                usize_to_i64(patch, "patch column stride")?,
                usize_to_i64(width, "patch offset row stride")?,
                1,
                usize_to_i64(channel_stride, "patch channel stride")?,
            ],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            context.stream,
        )?,
        ViewAccess::ReadOnly,
    )?;
    let patches = contiguous_with_context_exact_native(
        backend,
        &view,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    Ok(torch_reshape_with_context_exact_native(
        backend,
        &patches,
        &[
            usize_to_i64(batch, "batch")?,
            usize_to_i64(patch_count, "patch count")?,
            usize_to_i64(patch_width, "patch width")?,
        ],
        context,
    )?)
}

fn flatten_convolution_patches(
    backend: &CpuBackend,
    convolution: &Tensor,
    hidden: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let [batch, channels, height, width] = convolution.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "patch convolution did not produce NCHW output",
        ));
    };
    if *channels != usize_to_u64(hidden, "hidden size")? {
        return Err(ClipVisionError::InvalidInput(
            "patch convolution output width is invalid",
        ));
    }
    let patches = tensor_permute_exact_native(convolution, &[0, 2, 3, 1], context.cancellation)?;
    let patches = contiguous_with_context_exact_native(
        backend,
        &patches,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    Ok(torch_reshape_with_context_exact_native(
        backend,
        &patches,
        &[
            u64_to_i64(*batch, "batch")?,
            u64_to_i64(
                height
                    .checked_mul(*width)
                    .ok_or(ClipVisionError::Overflow("patch count"))?,
                "patch count",
            )?,
            usize_to_i64(hidden, "hidden size")?,
        ],
        context,
    )?)
}

fn prepend_class_embedding(
    backend: &CpuBackend,
    embeds: &Tensor,
    class_embedding: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let [batch, _tokens, hidden] = embeds.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "patch embeddings must have rank three",
        ));
    };
    require_shape("class embedding", class_embedding, &[*hidden])?;
    let class = torch_reshape_with_context_exact_native(
        backend,
        class_embedding,
        &[1, 1, u64_to_i64(*hidden, "hidden")?],
        context,
    )?;
    let class = tensor_repeat_with_context_exact_native(
        backend,
        &class,
        &[u64_to_i64(*batch, "batch")?, 1, 1],
        context,
    )?;
    Ok(torch_cat_with_context_exact_native(
        backend,
        &[class, embeds.clone()],
        1,
        context,
    )?)
}

fn add_position_embedding(
    backend: &CpuBackend,
    embeds: &Tensor,
    position_embedding: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let [batch, tokens, hidden] = embeds.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "vision embeddings must have rank three",
        ));
    };
    require_shape(
        "position embedding",
        position_embedding,
        &[*tokens, *hidden],
    )?;
    let positions = torch_reshape_with_context_exact_native(
        backend,
        position_embedding,
        &[
            1,
            u64_to_i64(*tokens, "tokens")?,
            u64_to_i64(*hidden, "hidden")?,
        ],
        context,
    )?;
    let positions = tensor_repeat_with_context_exact_native(
        backend,
        &positions,
        &[u64_to_i64(*batch, "batch")?, 1, 1],
        context,
    )?;
    add_exact(backend, embeds, &positions, context)
}

fn siglip2_position_embedding(
    backend: &CpuBackend,
    embeds: &Tensor,
    position_embedding: &Tensor,
    grid_height: usize,
    grid_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let [batch, tokens, hidden] = embeds.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "SigLIP2 embeddings must have rank three",
        ));
    };
    let grid_tokens = grid_height
        .checked_mul(grid_width)
        .ok_or(ClipVisionError::Overflow("grid tokens"))?;
    if *tokens != usize_to_u64(grid_tokens, "grid tokens")? {
        return Err(ClipVisionError::InvalidInput(
            "SigLIP2 embeddings do not match the patch grid",
        ));
    }
    let [position_count, position_hidden] = position_embedding.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "SigLIP2 position embedding must have rank two",
        ));
    };
    if position_hidden != hidden {
        return Err(ClipVisionError::Shape {
            name: "SigLIP2 position embedding",
            expected: vec![*position_count, *hidden],
            actual: position_embedding.descriptor().shape().to_vec(),
        });
    }
    let side = integer_square_root(u64_to_usize(*position_count, "position count")?);
    if side.checked_mul(side) != Some(u64_to_usize(*position_count, "position count")?) {
        return Err(ClipVisionError::InvalidConfiguration(
            "SigLIP2 position embedding count must be square",
        ));
    }
    let positions = torch_reshape_with_context_exact_native(
        backend,
        position_embedding,
        &[
            usize_to_i64(side, "position side")?,
            usize_to_i64(side, "position side")?,
            u64_to_i64(*hidden, "hidden")?,
        ],
        context,
    )?;
    let positions = tensor_permute_exact_native(&positions, &[2, 0, 1], context.cancellation)?;
    let positions = contiguous_with_context_exact_native(
        backend,
        &positions,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    let nchw = torch_reshape_with_context_exact_native(
        backend,
        &positions,
        &[
            1,
            u64_to_i64(*hidden, "hidden")?,
            usize_to_i64(side, "position side")?,
            usize_to_i64(side, "position side")?,
        ],
        context,
    )?;
    let resized = resize_with_context_exact_native(
        backend,
        &nchw,
        usize_to_u64(grid_height, "grid height")?,
        usize_to_u64(grid_width, "grid width")?,
        ResizeMode::Bilinear,
        true,
        context,
    )?;
    let positions = tensor_permute_exact_native(&resized, &[0, 2, 3, 1], context.cancellation)?;
    let positions = contiguous_with_context_exact_native(
        backend,
        &positions,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    let positions = torch_reshape_with_context_exact_native(
        backend,
        &positions,
        &[
            1,
            usize_to_i64(
                grid_height
                    .checked_mul(grid_width)
                    .ok_or(ClipVisionError::Overflow("grid positions"))?,
                "grid positions",
            )?,
            u64_to_i64(*hidden, "hidden")?,
        ],
        context,
    )?;
    let positions = tensor_repeat_with_context_exact_native(
        backend,
        &positions,
        &[u64_to_i64(*batch, "batch")?, 1, 1],
        context,
    )?;
    add_exact(backend, embeds, &positions, context)
}

fn select_token(
    backend: &CpuBackend,
    input: &Tensor,
    token: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let [batch, tokens, hidden] = input.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "token selection requires rank-three input",
        ));
    };
    if usize_to_u64(token, "selected token")? >= *tokens {
        return Err(ClipVisionError::InvalidInput(
            "selected token is out of range",
        ));
    }
    let selected = narrow_method_exact_native(
        input,
        1,
        usize_to_i64(token, "selected token")?,
        1,
        context.cancellation,
    )?;
    Ok(torch_reshape_with_context_exact_native(
        backend,
        &selected,
        &[u64_to_i64(*batch, "batch")?, u64_to_i64(*hidden, "hidden")?],
        context,
    )?)
}

fn drop_first_token(
    _backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let [_batch, tokens, _hidden] = input.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "token slicing requires rank-three input",
        ));
    };
    if *tokens <= 1 {
        return Err(ClipVisionError::InvalidInput(
            "Llava projection requires a class token and at least one patch token",
        ));
    }
    Ok(narrow_method_exact_native(
        input,
        1,
        1,
        tokens - 1,
        context.cancellation,
    )?)
}

fn stack_layers(
    backend: &CpuBackend,
    layers: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let first = layers.first().ok_or(ClipVisionError::InvalidConfiguration(
        "all-layer capture requires at least one layer",
    ))?;
    let [_batch, _tokens, _hidden] = first.descriptor().shape() else {
        return Err(ClipVisionError::InvalidInput(
            "captured layer must have rank three",
        ));
    };
    for layer in layers {
        if layer.descriptor().shape() != first.descriptor().shape() {
            return Err(ClipVisionError::InvalidInput(
                "captured layer shapes differ",
            ));
        }
    }
    Ok(torch_stack_with_context_exact_native(
        backend, layers, 1, context,
    )?)
}

fn quick_gelu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    Ok(canonical_quick_gelu(backend, input, context)?)
}

fn clip_vision_tensor_requirements(preprocess: bool) -> NativeExecutionRequirements {
    let mut requirements = NativeExecutionRequirements::new();
    requirements.extend([
        OperationSupport::allocation(DType::F32, Layout::Contiguous),
        OperationSupport::copy_input(DType::F32, Layout::Contiguous),
        OperationSupport::copy_output(DType::F32, Layout::Contiguous),
        OperationSupport::select_input(DType::F32, Layout::Contiguous),
        OperationSupport::select_output(DType::F32, Layout::Contiguous),
        OperationSupport::narrow_input(DType::F32, Layout::Contiguous),
        OperationSupport::narrow_output(DType::F32, Layout::Contiguous),
        OperationSupport::record_event(),
        OperationSupport::wait_event(),
    ]);
    for operation in [
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
        BinaryOperation::Minimum,
        BinaryOperation::Maximum,
    ] {
        requirements.extend([
            OperationSupport::binary_input(operation, DType::F32, Layout::Contiguous),
            OperationSupport::binary_output(operation, DType::F32, Layout::Contiguous),
            OperationSupport::binary_scalar_input(operation, DType::F32, Layout::Contiguous),
            OperationSupport::binary_scalar_output(operation, DType::F32, Layout::Contiguous),
        ]);
    }
    requirements.extend([
        OperationSupport::unary_input(UnaryOperation::Round, DType::F32, Layout::Contiguous),
        OperationSupport::unary_output(UnaryOperation::Round, DType::F32, Layout::Contiguous),
        OperationSupport::resize_input(ResizeMode::Bilinear, DType::F32, Layout::Contiguous),
        OperationSupport::resize_output(ResizeMode::Bilinear, DType::F32, Layout::Contiguous),
    ]);
    if preprocess {
        requirements.extend([
            OperationSupport::resize_input(ResizeMode::Bicubic, DType::F32, Layout::Contiguous),
            OperationSupport::resize_output(ResizeMode::Bicubic, DType::F32, Layout::Contiguous),
        ]);
    }
    requirements
}

fn add_exact(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    if left.descriptor().shape() != right.descriptor().shape()
        || left.descriptor().dtype() != right.descriptor().dtype()
        || left.descriptor().device() != right.descriptor().device()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return Err(ClipVisionError::InvalidInput(
            "added tensors must have identical shape, dtype, device, and stream",
        ));
    }
    let left = contiguous_with_context_exact_native(
        backend,
        left,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    let right = contiguous_with_context_exact_native(
        backend,
        right,
        MemoryFormatReference::Layout(Layout::Contiguous),
        context,
    )?;
    let (output, event) = backend.binary(
        BinaryOperation::Add,
        &left,
        &right,
        left.descriptor().clone(),
        context,
    )?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn binary_scalar_exact(
    backend: &CpuBackend,
    input: &Tensor,
    operation: BinaryOperation,
    scalar: f64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
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

fn unary_exact(
    backend: &CpuBackend,
    input: &Tensor,
    operation: UnaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipVisionError> {
    let (output, event) = backend.unary(operation, input, input.descriptor().clone(), context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn validate_parameter_target(
    name: &'static str,
    tensor: &Tensor,
    dtype: DType,
    device: DeviceId,
    stream: StreamId,
) -> Result<(), ClipVisionError> {
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != dtype || descriptor.device() != device || descriptor.stream() != stream
    {
        return Err(ClipVisionError::TensorTarget {
            name,
            expected_dtype: dtype,
            actual_dtype: descriptor.dtype(),
            expected_device: device,
            actual_device: descriptor.device(),
            expected_stream: stream,
            actual_stream: descriptor.stream(),
        });
    }
    Ok(())
}

fn require_execution_tensor(
    name: &'static str,
    tensor: &Tensor,
    dtype: DType,
    device: DeviceId,
    stream: StreamId,
    context: &ExecutionContext<'_>,
) -> Result<(), ClipVisionError> {
    context.check()?;
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != dtype
        || descriptor.device() != device
        || descriptor.stream() != stream
        || stream != context.stream
    {
        return Err(ClipVisionError::TensorTarget {
            name,
            expected_dtype: dtype,
            actual_dtype: descriptor.dtype(),
            expected_device: device,
            actual_device: descriptor.device(),
            expected_stream: stream,
            actual_stream: descriptor.stream(),
        });
    }
    Ok(())
}

fn require_shape(
    name: &'static str,
    tensor: &Tensor,
    expected: &[u64],
) -> Result<(), ClipVisionError> {
    if tensor.descriptor().shape() != expected {
        return Err(ClipVisionError::Shape {
            name,
            expected: expected.to_vec(),
            actual: tensor.descriptor().shape().to_vec(),
        });
    }
    Ok(())
}

fn integer_square_root(value: usize) -> usize {
    if value < 2 {
        return value;
    }
    let mut low = 1_usize;
    let mut high = value / 2 + 1;
    let mut root = 1_usize;
    while low <= high {
        let middle = low + (high - low) / 2;
        if middle <= value / middle {
            root = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    root
}

fn usize_to_u64(value: usize, operation: &'static str) -> Result<u64, ClipVisionError> {
    u64::try_from(value).map_err(|_| ClipVisionError::Overflow(operation))
}

fn usize_to_i64(value: usize, operation: &'static str) -> Result<i64, ClipVisionError> {
    i64::try_from(value).map_err(|_| ClipVisionError::Overflow(operation))
}

fn u64_to_i64(value: u64, operation: &'static str) -> Result<i64, ClipVisionError> {
    i64::try_from(value).map_err(|_| ClipVisionError::Overflow(operation))
}

fn u64_to_usize(value: u64, operation: &'static str) -> Result<usize, ClipVisionError> {
    usize::try_from(value).map_err(|_| ClipVisionError::Overflow(operation))
}

#[cfg(any(test, feature = "test-support"))]
fn f32_tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: Vec<u64>,
    values: &[f32],
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, context.stream)?;
    let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
    Ok(tensor)
}

#[cfg(any(test, feature = "test-support"))]
fn filled_tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: Vec<u64>,
    value: f32,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or(ClipVisionError::Overflow("test tensor elements"))?;
    f32_tensor(
        backend,
        context,
        shape,
        &vec![value; usize::try_from(count)?],
    )
}

#[cfg(any(test, feature = "test-support"))]
fn tiny_clip_configuration() -> ClipVisionConfiguration {
    ClipVisionConfiguration {
        model_type: ClipVisionModelType::Clip,
        dtype: DType::F32,
        device: DeviceId::CPU,
        hidden_size: 2,
        intermediate_size: 2,
        attention_heads: 1,
        layer_count: 1,
        image_size: 2,
        patch_size: 1,
        num_channels: 3,
        max_num_patches: 4,
        activation: ClipVisionActivation::QuickGelu,
        projection_dimension: None,
        llava_projection_dimension: None,
    }
}

#[cfg(any(test, feature = "test-support"))]
fn tiny_clip_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<ClipVisionWeights, Box<dyn std::error::Error>> {
    let zero_2 = filled_tensor(backend, context, vec![2], 0.0)?;
    let zero_2x2 = filled_tensor(backend, context, vec![2, 2], 0.0)?;
    let layer = ClipVisionLayerWeights {
        layer_norm_1_weight: filled_tensor(backend, context, vec![2], 1.0)?,
        layer_norm_1_bias: zero_2.clone(),
        query_weight: zero_2x2.clone(),
        query_bias: zero_2.clone(),
        key_weight: zero_2x2.clone(),
        key_bias: zero_2.clone(),
        value_weight: zero_2x2.clone(),
        value_bias: zero_2.clone(),
        output_weight: zero_2x2.clone(),
        output_bias: zero_2.clone(),
        layer_norm_2_weight: filled_tensor(backend, context, vec![2], 1.0)?,
        layer_norm_2_bias: zero_2.clone(),
        feed_forward_1_weight: zero_2x2.clone(),
        feed_forward_1_bias: zero_2.clone(),
        feed_forward_2_weight: zero_2x2,
        feed_forward_2_bias: zero_2.clone(),
    };
    Ok(ClipVisionWeights {
        patch_embedding_weight: filled_tensor(backend, context, vec![2, 3, 1, 1], 0.0)?,
        patch_embedding_bias: None,
        class_embedding: Some(zero_2.clone()),
        position_embedding: filled_tensor(backend, context, vec![5, 2], 0.0)?,
        pre_layer_norm_weight: Some(filled_tensor(backend, context, vec![2], 1.0)?),
        pre_layer_norm_bias: Some(zero_2.clone()),
        layers: vec![layer],
        post_layer_norm_weight: filled_tensor(backend, context, vec![2], 1.0)?,
        post_layer_norm_bias: zero_2,
        visual_projection_weight: None,
        llava_linear_1_weight: None,
        llava_linear_1_bias: None,
        llava_linear_2_weight: None,
        llava_linear_2_bias: None,
    })
}

#[cfg(test)]
fn tiny_clip_vision(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipVision, Box<dyn std::error::Error>> {
    Ok(NativeClipVision::new(
        tiny_clip_configuration(),
        tiny_clip_weights(backend, context)?,
    )?)
}

#[cfg(any(test, feature = "test-support"))]
pub fn deterministic_reduced_clip_vision_encode_fixture(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipVision, Box<dyn std::error::Error>> {
    context.check()?;
    let mut configuration = tiny_clip_configuration();
    configuration.layer_count = 2;
    let mut weights = tiny_clip_weights(backend, context)?;
    let mut layer = weights
        .layers
        .first()
        .ok_or(ClipVisionError::InvalidConfiguration(
            "reduced vision fixture layer missing",
        ))?
        .clone();
    layer.value_weight = f32_tensor(backend, context, vec![2, 2], &[1.0, 0.0, 0.0, 1.0])?;
    layer.output_weight = f32_tensor(backend, context, vec![2, 2], &[0.2, 0.0, 0.0, 0.2])?;
    weights.layers = vec![layer.clone(), layer];
    weights.patch_embedding_weight = f32_tensor(
        backend,
        context,
        vec![2, 3, 1, 1],
        &[0.7, -0.3, 0.2, -0.1, 0.6, 0.4],
    )?;
    weights.class_embedding = Some(f32_tensor(backend, context, vec![2], &[0.3, -0.2])?);
    weights.position_embedding = f32_tensor(
        backend,
        context,
        vec![5, 2],
        &[
            0.02, -0.01, 0.03, 0.05, -0.04, 0.01, 0.06, -0.03, -0.02, 0.04,
        ],
    )?;
    NativeClipVision::new_with_cancellation(configuration, weights, context.cancellation)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority};
    use std::{collections::BTreeSet, error::Error, mem, sync::Arc};

    #[test]
    fn reduced_encode_fixture_has_two_layers_and_nonzero_source_selectable_state()
    -> Result<(), Box<dyn Error>> {
        with_context(TEST_MEMORY_LIMIT_BYTES / 2, |backend, context| {
            let model = deterministic_reduced_clip_vision_encode_fixture(backend, context)?;
            assert_eq!(model.configuration().layer_count, 2);
            assert_eq!(model.configuration().model_type, ClipVisionModelType::Clip);
            assert!(!model.is_training());
            model.validate(context.cancellation)?;
            let image = f32_tensor(
                backend,
                context,
                vec![1, 3, 5, 3],
                &(0..45).map(|value| value as f32 / 44.0).collect::<Vec<_>>(),
            )?;
            let pixels = model.preprocess(backend, &image, true, context)?;
            let output = model.execution_session(context.cancellation)?.forward(
                backend,
                &pixels,
                ClipVisionIntermediate::Layer(-2),
                context,
            )?;
            assert_eq!(
                output
                    .intermediate
                    .as_ref()
                    .ok_or("Source intermediate missing")?
                    .descriptor()
                    .shape(),
                &[1, 5, 2]
            );
            assert!(
                sim_tensor::generated_native_diffusion::tensor_to_f32(
                    backend,
                    output
                        .intermediate
                        .as_ref()
                        .ok_or("Source intermediate missing")?,
                    context
                )?
                .iter()
                .any(|value| *value != 0.0)
            );
            Ok(())
        })
    }
    #[test]
    fn reduced_encode_fixture_precancel_allocates_nothing() -> Result<(), Box<dyn Error>> {
        with_context(0, |backend, context| {
            let before = backend.memory_snapshot().current_bytes;
            context.cancellation.cancel();
            assert!(deterministic_reduced_clip_vision_encode_fixture(backend, context).is_err());
            assert_eq!(backend.memory_snapshot().current_bytes, before);
            Ok(())
        })
    }

    const TEST_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024;

    fn with_context<ResultType>(
        workspace_bytes: u64,
        run: impl FnOnce(&CpuBackend, &ExecutionContext<'_>) -> Result<ResultType, Box<dyn Error>>,
    ) -> Result<ResultType, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        run(&backend, &context)
    }

    fn with_stream_contexts<ResultType>(
        run: impl FnOnce(
            &CpuBackend,
            &ExecutionContext<'_>,
            &ExecutionContext<'_>,
        ) -> Result<ResultType, Box<dyn Error>>,
    ) -> Result<ResultType, Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let default_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let alternate_context = backend.execution_context(
            StreamId::new(7),
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        run(&backend, &default_context, &alternate_context)
    }

    #[test]
    fn caller_cancellation_stops_every_named_clip_vision_construction_phase()
    -> Result<(), Box<dyn Error>> {
        let phases = [
            ClipVisionConstructionPhase::Entry,
            ClipVisionConstructionPhase::CanonicalState,
            ClipVisionConstructionPhase::ParameterProjection,
            ClipVisionConstructionPhase::PatchEmbedding,
            ClipVisionConstructionPhase::PreLayerNorm,
            ClipVisionConstructionPhase::LayerNorm1(0),
            ClipVisionConstructionPhase::AttentionQuery(0),
            ClipVisionConstructionPhase::AttentionKey(0),
            ClipVisionConstructionPhase::AttentionValue(0),
            ClipVisionConstructionPhase::AttentionOutput(0),
            ClipVisionConstructionPhase::LayerNorm2(0),
            ClipVisionConstructionPhase::MlpInput(0),
            ClipVisionConstructionPhase::MlpActivation(0),
            ClipVisionConstructionPhase::MlpOutput(0),
            ClipVisionConstructionPhase::PostLayerNorm,
            ClipVisionConstructionPhase::VisualProjection,
            ClipVisionConstructionPhase::LlavaProjection,
            ClipVisionConstructionPhase::SemanticDigest,
            ClipVisionConstructionPhase::ModuleDigest,
            ClipVisionConstructionPhase::Validation,
            ClipVisionConstructionPhase::Return,
        ];
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let upload_cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(2 * 1024 * 1024)?,
            &upload_cancellation,
        );
        let model = tiny_clip_vision(&backend, &context)?;
        for phase in phases {
            let cancellation = CancellationToken::default();
            let error = NativeClipVision::new_with_cancellation_and_phase_hook(
                tiny_clip_configuration(),
                tiny_clip_weights(&backend, &context)?,
                &cancellation,
                &mut |actual, token| {
                    if actual == phase {
                        token.cancel();
                    }
                },
            )
            .expect_err("the named construction boundary must not publish a model");
            assert!(
                matches!(error, ClipVisionError::Cancelled),
                "{phase:?}: {error}"
            );

            let reconstruction_cancellation = CancellationToken::default();
            let error = model
                .reconstruct_with_phase_hook(&reconstruction_cancellation, &mut |actual, token| {
                    if actual == phase {
                        token.cancel();
                    }
                })
                .expect_err("the named reconstruction boundary must not publish a model");
            assert!(
                matches!(error, ClipVisionError::Cancelled),
                "reconstruction {phase:?}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn reconstruction_reprojects_canonical_state_and_checked_forward_retains_pooled_hidden()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(2 * 1024 * 1024)?,
            &cancellation,
        );
        let mut configuration = tiny_clip_configuration();
        configuration.projection_dimension = Some(2);
        let mut weights = tiny_clip_weights(&backend, &context)?;
        weights.visual_projection_weight = Some(f32_tensor(
            &backend,
            &context,
            vec![2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        )?);
        let model = NativeClipVision::new_with_cancellation(configuration, weights, &cancellation)?;
        let reconstructed = model.reconstruct(&cancellation)?;
        assert_eq!(reconstructed.configuration, model.configuration);
        assert_eq!(reconstructed.semantic_identity, model.semantic_identity);
        assert_eq!(
            reconstructed.module_state_digest_sha256,
            model.module_state_digest_sha256
        );
        assert_eq!(reconstructed.resident_parts()?, model.resident_parts()?);
        for ((actual_name, actual), (expected_name, expected)) in reconstructed
            .canonical_state
            .iter()
            .zip(&model.canonical_state)
        {
            assert_eq!(actual_name, expected_name);
            assert_eq!(actual.storage_id(), expected.storage_id());
            assert_eq!(actual.descriptor(), expected.descriptor());
            assert_eq!(actual.contiguous_bytes()?, expected.contiguous_bytes()?);
        }

        let input = filled_tensor(&backend, &context, vec![1, 3, 2, 2], 0.25)?;
        let mut checked_model = reconstructed.clone();
        let checked = checked_model.forward_checked(
            &backend,
            &input,
            ClipVisionIntermediate::None,
            &context,
        )?;
        assert_eq!(
            checked.pooled_hidden.contiguous_bytes()?,
            checked.output.image_embeds.contiguous_bytes()?
        );
        let mut public_model = reconstructed;
        let public =
            public_model.forward(&backend, &input, ClipVisionIntermediate::None, &context)?;
        assert_eq!(
            public.semantic_digest_sha256(),
            checked.output.semantic_digest_sha256()
        );

        let mut misordered = model;
        misordered.canonical_state.swap(0, 1);
        assert!(matches!(
            misordered.reconstruct(&cancellation),
            Err(ClipVisionError::SemanticIdentityChanged)
                | Err(ClipVisionError::InvalidConfiguration(_))
        ));
        Ok(())
    }

    #[test]
    fn clip_vision_resource_identity_sessions_and_alias_accounting_are_owner_derived()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(2 * 1024 * 1024)?,
            &cancellation,
        );
        let model = tiny_clip_vision(&backend, &context)?;
        model.validate(&cancellation)?;
        assert!(!model.is_training());
        assert_eq!(
            model.semantic_identity().role(),
            NativeModelResourceRole::ClipVision
        );
        assert_eq!(
            model.semantic_identity().role().source_type_id(),
            CLIP_VISION_SOURCE_TYPE_ID
        );
        assert_eq!(model.semantic_digest_sha256().len(), 64);

        let unique_storage =
            model
                .canonical_state
                .iter()
                .fold(BTreeSet::new(), |mut storages, (_, tensor)| {
                    storages.insert(tensor.storage_id().get());
                    storages
                });
        assert!(unique_storage.len() < model.canonical_state.len());
        assert!(model.resident_bytes()? > model.resident_storage_bytes()?);

        let payload = crate::NativeModelPayload::clip_vision(Arc::new(model.clone()))?;
        payload.validate()?;
        assert_eq!(payload.identity(), model.semantic_identity());
        let retained =
            payload
                .clip_vision_resource()
                .ok_or(ClipVisionError::InvalidConfiguration(
                    "CLIP vision resource is missing",
                ))?;
        let backing = payload
            .resident_parts()?
            .backing_allocations()
            .first()
            .copied()
            .ok_or(ClipVisionError::InvalidConfiguration(
                "CLIP vision backing allocation is missing",
            ))?;
        assert_eq!(backing.kind(), crate::NativeModelBackingKind::ClipVision);
        assert_eq!(
            backing.resident_bytes(),
            retained.resident_parts()?.owned_bytes()
        );
        assert_eq!(
            payload.resident_parts()?.tensor_allocations().len(),
            retained.resident_parts()?.tensor_allocations().len()
        );

        let input = filled_tensor(&backend, &context, vec![1, 3, 2, 2], 0.0)?;
        let mut first = model.execution_session(&cancellation)?;
        let mut second = model.execution_session(&cancellation)?;
        let first_output =
            first.forward(&backend, &input, ClipVisionIntermediate::None, &context)?;
        let second_output =
            second.forward(&backend, &input, ClipVisionIntermediate::None, &context)?;
        assert_eq!(
            first_output.semantic_digest_sha256(),
            second_output.semantic_digest_sha256()
        );
        assert_eq!(
            first.semantic_identity().digest_sha256(),
            model.semantic_digest_sha256()
        );

        let mut training = model.clone();
        training.train();
        assert!(matches!(
            crate::NativeModelPayload::clip_vision(Arc::new(training.clone())),
            Err(NativeModelPayloadError::ResourceMismatch(
                "CLIP_VISION evaluation state"
            ))
        ));
        assert!(matches!(
            training.execution_session(&cancellation),
            Err(ClipVisionError::EvaluationRequired)
        ));
        training.eval();
        training.validate(&cancellation)?;

        let mut drifted = model.clone();
        drifted.module_state_digest_sha256 = "0".repeat(64);
        assert!(matches!(
            drifted.validate(&cancellation),
            Err(ClipVisionError::SemanticIdentityChanged)
        ));
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            model.execution_session(&cancelled),
            Err(ClipVisionError::Cancelled)
        ));
        model.validate(&cancellation)?;
        Ok(())
    }

    #[test]
    fn clip_vision_output_checks_source_shapes_and_deduplicates_aliased_storage()
    -> Result<(), Box<dyn Error>> {
        with_context(0, |backend, context| {
            let last_hidden_state = f32_tensor(backend, context, vec![1, 5, 4], &[0.25; 20])?;
            let image_embeds = f32_tensor(backend, context, vec![1, 3], &[0.5; 3])?;
            let projected_intermediate = f32_tensor(backend, context, vec![1, 4, 6], &[0.75; 24])?;
            let output = ClipVisionOutput::checked(
                last_hidden_state.clone(),
                Some(last_hidden_state),
                image_embeds,
                Some(projected_intermediate),
                vec![[3, 224, 224]],
            )?;
            output.validate()?;
            assert_eq!(output.image_sizes(), &[[3, 224, 224]]);
            let resident_parts = output.resident_parts()?;
            assert_eq!(resident_parts.tensor_allocations().len(), 3);
            assert_eq!(output.resident_bytes(), resident_parts.resident_bytes()?);
            assert_eq!(
                resident_parts.owned_bytes(),
                u64::try_from(mem::size_of::<ClipVisionOutput>())?
                    + u64::try_from(mem::size_of::<[u64; 3]>())?
            );
            Ok(())
        })
    }

    #[test]
    fn clip_vision_output_digest_excludes_storage_identity_and_rejects_nonfinite_content()
    -> Result<(), Box<dyn Error>> {
        with_stream_contexts(|backend, default_context, alternate_context| {
            let output = |backend: &CpuBackend,
                          context: &ExecutionContext<'_>|
             -> Result<ClipVisionOutput, Box<dyn Error>> {
                let hidden = f32_tensor(backend, context, vec![1, 2, 2], &[1.0; 4])?;
                let embeds = f32_tensor(backend, context, vec![1, 2], &[2.0; 2])?;
                Ok(ClipVisionOutput::checked(
                    hidden.clone(),
                    Some(hidden),
                    embeds,
                    None,
                    vec![[3, 32, 32]],
                )?)
            };
            let first = output(backend, default_context)?;
            let second = output(backend, alternate_context)?;
            assert_ne!(
                first.last_hidden_state.storage_id(),
                second.last_hidden_state.storage_id()
            );
            assert_ne!(
                first.last_hidden_state.descriptor().stream(),
                second.last_hidden_state.descriptor().stream()
            );
            assert_eq!(
                first.semantic_digest_sha256(),
                second.semantic_digest_sha256()
            );
            assert_eq!(first.resident_bytes(), second.resident_bytes());
            assert_ne!(
                first.resident_parts()?.tensor_allocations(),
                second.resident_parts()?.tensor_allocations()
            );

            let changed_content = ClipVisionOutput::checked(
                f32_tensor(backend, default_context, vec![1, 2, 2], &[1.5; 4])?,
                Some(f32_tensor(
                    backend,
                    default_context,
                    vec![1, 2, 2],
                    &[1.0; 4],
                )?),
                f32_tensor(backend, default_context, vec![1, 2], &[2.0; 2])?,
                None,
                vec![[3, 32, 32]],
            )?;
            assert_ne!(
                first.semantic_digest_sha256(),
                changed_content.semantic_digest_sha256()
            );

            let changed_shape = ClipVisionOutput::checked(
                f32_tensor(backend, default_context, vec![1, 3, 2], &[1.0; 6])?,
                None,
                f32_tensor(backend, default_context, vec![1, 2], &[2.0; 2])?,
                None,
                vec![[3, 32, 32]],
            )?;
            assert_ne!(
                first.semantic_digest_sha256(),
                changed_shape.semantic_digest_sha256()
            );

            let non_finite = f32_tensor(backend, default_context, vec![1, 1, 1], &[f32::INFINITY])?;
            let embeds = f32_tensor(backend, default_context, vec![1, 1], &[0.0])?;
            assert!(
                ClipVisionOutput::checked(non_finite, None, embeds, None, vec![[3, 16, 16]],)
                    .is_err()
            );
            Ok(())
        })
    }

    #[test]
    fn clip_vision_output_preserves_all_layer_cardinality_and_projection_rules()
    -> Result<(), Box<dyn Error>> {
        with_context(0, |backend, context| {
            let last_hidden_state = f32_tensor(backend, context, vec![1, 3, 2], &[0.0; 6])?;
            let all_layers = f32_tensor(backend, context, vec![1, 2, 3, 2], &[0.0; 12])?;
            let image_embeds = f32_tensor(backend, context, vec![1, 3, 4], &[0.0; 12])?;
            let output = ClipVisionOutput::checked(
                last_hidden_state.clone(),
                Some(all_layers.clone()),
                image_embeds.clone(),
                None,
                vec![[3, 32, 48]],
            )?;
            output.validate()?;

            let projected = f32_tensor(backend, context, vec![1, 2, 4], &[0.0; 8])?;
            assert!(
                ClipVisionOutput::checked(
                    last_hidden_state,
                    Some(all_layers),
                    image_embeds,
                    Some(projected),
                    vec![[3, 32, 48]],
                )
                .is_err()
            );
            Ok(())
        })
    }
}
