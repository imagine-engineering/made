use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext, Tensor, TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, GeluApproximation, gelu_with_context_exact_native,
        layer_norm_with_context_exact_native, silu_tensor_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_16::{
        ElementwiseRuntimePartSixteenError, add_method_with_context_exact_native,
        mul_method_with_context_exact_native,
    },
    generated_neural_network_functional_01::{
        NeuralNetworkFunctionalError, linear_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        ShapeLayoutTransformPartThreeError, tensor_permute_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        ConvolutionConfiguration, InterpolateConfiguration, InterpolateMode,
        SpatialFunctionalKernelError, conv_2d_tensor_with_context_exact_native,
        interpolate_tensor_with_context_exact_native,
    },
};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::attention::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionRequest,
    RotaryFrequencyLayout, RotaryPairLayout, RotaryPositionSequence, RotaryPositions,
    RotaryScaling, RotaryTableRequest, apply_rotary_table, precompute_rotary_table,
    scaled_dot_product_attention_with_context,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeDino2Configuration {
    pub(crate) prefix: &'static str,
    pub(crate) hidden: usize,
    pub(crate) layer_count: usize,
    pub(crate) attention_heads: usize,
    pub(crate) patch: usize,
    pub(crate) image: usize,
    pub(crate) qknorm_start: Option<usize>,
    pub(crate) alternate_attention_start: Option<usize>,
    pub(crate) rope_start: Option<usize>,
    pub(crate) concatenate_camera_token: bool,
    pub(crate) use_mask_token: bool,
    pub(crate) swiglu: bool,
    pub(crate) output_layers: [usize; 4],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NativeDino2StateSpecification {
    pub(crate) key: String,
    pub(crate) shape: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeDino2Backbone {
    configuration: NativeDino2Configuration,
}

pub(crate) struct NativeDino2Execution<'a> {
    backbone: NativeDino2Backbone,
    execution_state: &'a BTreeMap<String, Tensor>,
    memory_budget_bytes: u64,
    resident_bytes: u64,
    parent_preflighted: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct NativeDino2Feature {
    pub(crate) patch_values: Vec<f32>,
    pub(crate) camera_values: Vec<f32>,
    pub(crate) patches: usize,
    pub(crate) channels: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeDino2ReferenceStrategy {
    First,
    Middle,
    SaddleBalanced,
    SaddleSimRange,
}

impl NativeDino2Backbone {
    pub(crate) fn new(configuration: NativeDino2Configuration) -> Result<Self, NativeDino2Error> {
        if configuration.hidden == 0
            || configuration.layer_count == 0
            || configuration.attention_heads == 0
            || configuration.patch == 0
            || configuration.image == 0
            || !configuration
                .hidden
                .is_multiple_of(configuration.attention_heads)
            || !configuration.image.is_multiple_of(configuration.patch)
            || configuration
                .qknorm_start
                .is_some_and(|start| start >= configuration.layer_count)
            || configuration
                .rope_start
                .is_some_and(|start| start >= configuration.layer_count)
            || configuration
                .alternate_attention_start
                .is_some_and(|start| start >= configuration.layer_count)
            || configuration.output_layers[3] >= configuration.layer_count
            || !configuration
                .output_layers
                .windows(2)
                .all(|layers| layers[0] < layers[1])
        {
            return Err(NativeDino2Error::UnsupportedArchitecture);
        }
        Ok(Self { configuration })
    }

    pub(crate) fn state_manifest(
        self,
    ) -> Result<Vec<NativeDino2StateSpecification>, NativeDino2Error> {
        let configuration = self.configuration;
        let mut states = Vec::new();
        add_conv(
            &mut states,
            &format!(
                "{}.embeddings.patch_embeddings.projection",
                configuration.prefix
            ),
            configuration.hidden,
            3,
            configuration.patch,
            true,
        )?;
        let patch_side = configuration.image / configuration.patch;
        let position_tokens = patch_side
            .checked_mul(patch_side)
            .and_then(|patches| patches.checked_add(1))
            .ok_or(NativeDino2Error::ShapeOverflow)?;
        add_state(
            &mut states,
            &format!("{}.embeddings.position_embeddings", configuration.prefix),
            &[1, position_tokens, configuration.hidden],
        )?;
        add_state(
            &mut states,
            &format!("{}.embeddings.cls_token", configuration.prefix),
            &[1, 1, configuration.hidden],
        )?;
        if configuration.use_mask_token {
            add_state(
                &mut states,
                &format!("{}.embeddings.mask_token", configuration.prefix),
                &[1, configuration.hidden],
            )?;
        }
        if configuration.concatenate_camera_token {
            add_state(
                &mut states,
                &format!("{}.embeddings.camera_token", configuration.prefix),
                &[1, 2, configuration.hidden],
            )?;
        }
        for layer in 0..configuration.layer_count {
            let prefix = format!("{}.encoder.layer.{layer}", configuration.prefix);
            add_affine(
                &mut states,
                &format!("{prefix}.norm1"),
                configuration.hidden,
            )?;
            for name in ["query", "key", "value"] {
                add_linear(
                    &mut states,
                    &format!("{prefix}.attention.attention.{name}"),
                    configuration.hidden,
                    configuration.hidden,
                    true,
                )?;
            }
            add_linear(
                &mut states,
                &format!("{prefix}.attention.output.dense"),
                configuration.hidden,
                configuration.hidden,
                true,
            )?;
            if configuration
                .qknorm_start
                .is_some_and(|start| layer >= start)
            {
                let head = configuration.hidden / configuration.attention_heads;
                add_affine(&mut states, &format!("{prefix}.attention.q_norm"), head)?;
                add_affine(&mut states, &format!("{prefix}.attention.k_norm"), head)?;
            }
            add_state(
                &mut states,
                &format!("{prefix}.layer_scale1.lambda1"),
                &[configuration.hidden],
            )?;
            add_state(
                &mut states,
                &format!("{prefix}.layer_scale2.lambda1"),
                &[configuration.hidden],
            )?;
            add_affine(
                &mut states,
                &format!("{prefix}.norm2"),
                configuration.hidden,
            )?;
            if configuration.swiglu {
                let intermediate = configuration
                    .hidden
                    .checked_mul(4)
                    .and_then(|value| value.checked_mul(2))
                    .map(|value| value / 3)
                    .and_then(|value| value.checked_add(7))
                    .map(|value| value / 8)
                    .and_then(|value| value.checked_mul(8))
                    .ok_or(NativeDino2Error::ShapeOverflow)?;
                add_linear(
                    &mut states,
                    &format!("{prefix}.mlp.weights_in"),
                    intermediate
                        .checked_mul(2)
                        .ok_or(NativeDino2Error::ShapeOverflow)?,
                    configuration.hidden,
                    true,
                )?;
                add_linear(
                    &mut states,
                    &format!("{prefix}.mlp.weights_out"),
                    configuration.hidden,
                    intermediate,
                    true,
                )?;
            } else {
                let expanded = configuration
                    .hidden
                    .checked_mul(4)
                    .ok_or(NativeDino2Error::ShapeOverflow)?;
                add_linear(
                    &mut states,
                    &format!("{prefix}.mlp.fc1"),
                    expanded,
                    configuration.hidden,
                    true,
                )?;
                add_linear(
                    &mut states,
                    &format!("{prefix}.mlp.fc2"),
                    configuration.hidden,
                    expanded,
                    true,
                )?;
            }
        }
        add_affine(
            &mut states,
            &format!("{}.layernorm", configuration.prefix),
            configuration.hidden,
        )?;
        Ok(states)
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "production consumer is sim-parity-native-moge-resource-foundation"
        )
    )]
    pub(crate) fn bind<'a>(
        self,
        execution_state: &'a BTreeMap<String, Tensor>,
        memory_budget_bytes: u64,
        resident_bytes: u64,
    ) -> NativeDino2Execution<'a> {
        NativeDino2Execution {
            backbone: self,
            execution_state,
            memory_budget_bytes,
            resident_bytes,
            parent_preflighted: false,
        }
    }

    pub(crate) fn bind_parent_preflighted<'a>(
        self,
        execution_state: &'a BTreeMap<String, Tensor>,
        memory_budget_bytes: u64,
        resident_bytes: u64,
    ) -> NativeDino2Execution<'a> {
        NativeDino2Execution {
            backbone: self,
            execution_state,
            memory_budget_bytes,
            resident_bytes,
            parent_preflighted: true,
        }
    }

    pub(crate) fn owns_state_key(self, key: &str) -> bool {
        key.strip_prefix(self.configuration.prefix)
            .is_some_and(|suffix| {
                suffix == ".layernorm.weight"
                    || suffix == ".layernorm.bias"
                    || suffix.starts_with(".embeddings.")
                    || suffix.starts_with(".encoder.layer.")
            })
    }

    pub(crate) fn project_state_tensor(
        self,
        backend: &CpuBackend,
        key: &str,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeDino2Error> {
        if !self.owns_state_key(key) {
            return Err(NativeDino2Error::UnexpectedState(key.to_owned()));
        }
        let specification = self
            .state_manifest()?
            .into_iter()
            .find(|specification| specification.key == key)
            .ok_or_else(|| NativeDino2Error::UnexpectedState(key.to_owned()))?;
        let descriptor = tensor.descriptor();
        if descriptor.shape() != specification.shape
            || !matches!(descriptor.dtype(), DType::F16 | DType::Bf16 | DType::F32)
        {
            return Err(NativeDino2Error::StateShape {
                key: key.to_owned(),
                expected: specification.shape,
                actual: descriptor.shape().to_vec(),
                actual_dtype: descriptor.dtype(),
            });
        }
        Ok(cast_to_with_context_exact_native(
            backend,
            tensor,
            DType::F32,
            DeviceId::CPU,
            false,
            true,
            context,
        )?)
    }
}

#[derive(Debug, Error)]
pub(crate) enum NativeDino2Error {
    #[error("DINOv2 execution was cancelled")]
    Cancelled,
    #[error("DINOv2 architecture is unsupported")]
    UnsupportedArchitecture,
    #[error("DINOv2 image is invalid: {0}")]
    InvalidImage(String),
    #[error("DINOv2 state is missing key {0}")]
    MissingState(String),
    #[error("DINOv2 state is unexpected: {0}")]
    UnexpectedState(String),
    #[error("DINOv2 state {key} expected {expected:?}, got {actual:?} {actual_dtype:?}")]
    StateShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        actual_dtype: DType,
    },
    #[error("DINOv2 retained semantic state changed")]
    SemanticStateChanged,
    #[error("DINOv2 shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("DINOv2 allocation failed")]
    Allocation,
    #[error("DINOv2 memory requirement {required} exceeds budget {budget}")]
    OutOfMemory { required: u64, budget: u64 },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error(transparent)]
    ElementwiseSixteen(#[from] ElementwiseRuntimePartSixteenError),
    #[error(transparent)]
    NeuralFunctional(#[from] NeuralNetworkFunctionalError),
    #[error(transparent)]
    ShapeLayoutThree(#[from] ShapeLayoutTransformPartThreeError),
    #[error(transparent)]
    Spatial(#[from] SpatialFunctionalKernelError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
}

impl From<sim_types::CancellationError> for NativeDino2Error {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl NativeDino2Execution<'_> {
    fn execution_tensor(&self, key: &str) -> Result<&Tensor, NativeDino2Error> {
        self.execution_state
            .get(key)
            .ok_or_else(|| NativeDino2Error::MissingState(key.to_owned()))
    }

    fn validate_execution_state(&self) -> Result<(), NativeDino2Error> {
        let specifications = self.backbone.state_manifest()?;
        if !self.parent_preflighted && self.execution_state.len() != specifications.len() {
            if let Some(key) = self.execution_state.keys().find(|key| {
                !specifications
                    .iter()
                    .any(|specification| specification.key.as_str() == key.as_str())
            }) {
                return Err(NativeDino2Error::UnexpectedState(key.clone()));
            }
        }
        for specification in specifications {
            let tensor = self.execution_tensor(&specification.key)?;
            let descriptor = tensor.descriptor();
            if descriptor.shape() != specification.shape || descriptor.dtype() != DType::F32 {
                return Err(NativeDino2Error::StateShape {
                    key: specification.key,
                    expected: specification.shape,
                    actual: descriptor.shape().to_vec(),
                    actual_dtype: descriptor.dtype(),
                });
            }
        }
        Ok(())
    }
}

fn preflight_execution(
    resource: &NativeDino2Execution<'_>,
    image_shape: &[usize],
    flat_views: usize,
) -> Result<(), NativeDino2Error> {
    let [_, channels, height, width] = image_shape else {
        return Err(NativeDino2Error::ShapeOverflow);
    };
    let configuration = resource.backbone.configuration;
    if *channels != 3 || *height < configuration.patch || *width < configuration.patch {
        return Err(NativeDino2Error::InvalidImage(
            "preprocessed image is smaller than the patch embedding".to_owned(),
        ));
    }
    let patch_height = height
        .checked_sub(configuration.patch)
        .map(|value| value / configuration.patch + 1)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let patch_width = width
        .checked_sub(configuration.patch)
        .map(|value| value / configuration.patch + 1)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let patches = patch_height
        .checked_mul(patch_width)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let tokens = patches
        .checked_add(1)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let input_elements = image_shape
        .iter()
        .try_fold(1_usize, |product, dimension| {
            product.checked_mul(*dimension)
        })
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let patch_elements = flat_views
        .checked_mul(patches)
        .and_then(|value| value.checked_mul(configuration.hidden))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let token_elements = flat_views
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(configuration.hidden))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let attention_elements = flat_views
        .checked_mul(configuration.attention_heads)
        .and_then(|value| value.checked_mul(tokens))
        .and_then(|value| value.checked_mul(tokens))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let output_channels = if configuration.concatenate_camera_token {
        configuration
            .hidden
            .checked_mul(2)
            .ok_or(NativeDino2Error::ShapeOverflow)?
    } else {
        configuration.hidden
    };
    let retained_outputs = flat_views
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(output_channels))
        .and_then(|value| value.checked_mul(configuration.output_layers.len()))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let live_elements = input_elements
        .checked_add(patch_elements)
        .and_then(|value| value.checked_add(token_elements.checked_mul(32)?))
        .and_then(|value| value.checked_add(attention_elements.checked_mul(2)?))
        .and_then(|value| value.checked_add(retained_outputs))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let live_bytes = u64::try_from(live_elements)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let required = resource
        .resident_bytes
        .checked_add(live_bytes)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    if required > resource.memory_budget_bytes {
        return Err(NativeDino2Error::OutOfMemory {
            required,
            budget: resource.memory_budget_bytes,
        });
    }
    Ok(())
}

fn convolution(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    input: &Tensor,
    prefix: &str,
    stride: usize,
    padding: usize,
    transposed: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeDino2Error> {
    if transposed {
        return Err(NativeDino2Error::UnsupportedArchitecture);
    }
    let configuration = ConvolutionConfiguration {
        stride: vec![stride, stride],
        padding: vec![padding, padding],
        dilation: vec![1, 1],
        groups: 1,
        output_padding: vec![0, 0],
    };
    let weight = resource.execution_tensor(&format!("{prefix}.weight"))?;
    let bias = resource.execution_state.get(&format!("{prefix}.bias"));
    Ok(conv_2d_tensor_with_context_exact_native(
        backend,
        input,
        weight,
        bias,
        &configuration,
        context,
    )?)
}

fn tensor_values(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeDino2Error> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(NativeDino2Error::SemanticStateChanged);
    }
    let bytes = tensor.contiguous_bytes()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / 4)
        .map_err(|_| NativeDino2Error::Allocation)?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(16_384) {
            cancellation.check()?;
        }
        values.push(f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| NativeDino2Error::ShapeOverflow)?,
        ));
    }
    Ok(values)
}

fn shape_usize(tensor: &Tensor) -> Result<Vec<usize>, NativeDino2Error> {
    tensor
        .descriptor()
        .shape()
        .iter()
        .map(|dimension| usize_from(*dimension))
        .collect()
}

fn shape_u64(shape: &[usize]) -> Result<Vec<u64>, NativeDino2Error> {
    shape
        .iter()
        .map(|dimension| u64::try_from(*dimension).map_err(|_| NativeDino2Error::ShapeOverflow))
        .collect()
}

fn spatial_size(tensor: &Tensor) -> Result<(usize, usize), NativeDino2Error> {
    let shape = shape_usize(tensor)?;
    match shape.as_slice() {
        [_, _, height, width] => Ok((*height, *width)),
        _ => Err(NativeDino2Error::ShapeOverflow),
    }
}

fn filled_f32(length: usize, value: f32) -> Result<Vec<f32>, NativeDino2Error> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| NativeDino2Error::Allocation)?;
    output.resize(length, value);
    Ok(output)
}

fn integer_square_root(value: usize) -> usize {
    if value < 2 {
        return value;
    }
    let mut low = 1_usize;
    let mut high = value / 2 + 1;
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if middle <= value / middle {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn usize_from(value: u64) -> Result<usize, NativeDino2Error> {
    usize::try_from(value).map_err(|_| NativeDino2Error::ShapeOverflow)
}

impl NativeDino2Execution<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_intermediate_layers_da3(
        &self,
        backend: &CpuBackend,
        image: &Tensor,
        batch: usize,
        views: usize,
        reference_strategy: NativeDino2ReferenceStrategy,
        camera_token: Option<&[f32]>,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<NativeDino2Feature>, NativeDino2Error> {
        execute_da3_backbone(
            self,
            backend,
            image,
            batch,
            views,
            reference_strategy,
            camera_token,
            false,
            context,
        )
    }

    pub(crate) fn get_intermediate_layers(
        &self,
        backend: &CpuBackend,
        image: &Tensor,
        batch: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<NativeDino2Feature>, NativeDino2Error> {
        if self
            .backbone
            .configuration
            .alternate_attention_start
            .is_some()
            || self.backbone.configuration.qknorm_start.is_some()
            || self.backbone.configuration.rope_start.is_some()
            || self.backbone.configuration.concatenate_camera_token
        {
            return Err(NativeDino2Error::UnsupportedArchitecture);
        }
        execute_da3_backbone(
            self,
            backend,
            image,
            batch,
            1,
            NativeDino2ReferenceStrategy::First,
            None,
            true,
            context,
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn forward(
        &self,
        backend: &CpuBackend,
        image: &Tensor,
        batch: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeDino2Feature, NativeDino2Error> {
        self.get_intermediate_layers(backend, image, batch, context)?
            .pop()
            .ok_or(NativeDino2Error::UnsupportedArchitecture)
    }
}

fn execute_da3_backbone(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    image: &Tensor,
    batch: usize,
    views: usize,
    reference_strategy: NativeDino2ReferenceStrategy,
    camera_token: Option<&[f32]>,
    ordinary: bool,
    context: &ExecutionContext<'_>,
) -> Result<Vec<NativeDino2Feature>, NativeDino2Error> {
    context.check()?;
    let image_shape = shape_usize(image)?;
    let flat_views = batch
        .checked_mul(views)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    if image_shape.len() != 4 || image_shape[0] != flat_views || image_shape[1] != 3 {
        return Err(NativeDino2Error::InvalidImage(
            "preprocessed image geometry changed".to_owned(),
        ));
    }
    let patch = resource.backbone.configuration.patch;
    let hidden = resource.backbone.configuration.hidden;
    if !resource.parent_preflighted {
        preflight_execution(resource, &image_shape, flat_views)?;
    }
    resource.validate_execution_state()?;
    let patch_tensor = convolution(
        resource,
        backend,
        image,
        "native.backbone.embeddings.patch_embeddings.projection",
        patch,
        0,
        false,
        context,
    )?;
    let patch_shape = shape_usize(&patch_tensor)?;
    let patch_height = *patch_shape.get(2).ok_or(NativeDino2Error::ShapeOverflow)?;
    let patch_width = *patch_shape.get(3).ok_or(NativeDino2Error::ShapeOverflow)?;
    let patches = patch_height
        .checked_mul(patch_width)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let patch_values = tensor_to_f32_with_context_exact_native(
        backend,
        &tensor_permute_exact_native(&patch_tensor, &[0, 2, 3, 1], context.cancellation)?,
        context,
    )?;
    let cls = tensor_values(
        resource.execution_tensor("native.backbone.embeddings.cls_token")?,
        context.cancellation,
    )?;
    let positions =
        interpolated_position_embeddings(resource, backend, patch_height, patch_width, context)?;
    let tokens = patches
        .checked_add(1)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let mut values = filled_f32(
        batch
            .checked_mul(views)
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|value| value.checked_mul(hidden))
            .ok_or(NativeDino2Error::ShapeOverflow)?,
        0.0,
    )?;
    for flat_view in 0..flat_views {
        context.check()?;
        for channel in 0..hidden {
            values[(flat_view * tokens) * hidden + channel] = cls[channel] + positions[channel];
        }
        for patch_index in 0..patches {
            for channel in 0..hidden {
                let destination = ((flat_view * tokens + patch_index + 1) * hidden) + channel;
                values[destination] = patch_values
                    [(flat_view * patches + patch_index) * hidden + channel]
                    + positions[(patch_index + 1) * hidden + channel];
            }
        }
    }

    let mut local_values = values.clone();
    let mut reference_indices = None;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(resource.backbone.configuration.output_layers.len())
        .map_err(|_| NativeDino2Error::Allocation)?;
    for layer in 0..resource.backbone.configuration.layer_count {
        context.check()?;
        if resource
            .backbone
            .configuration
            .alternate_attention_start
            .is_some_and(|start| layer + 1 == start)
            && views >= 3
            && camera_token.is_none()
        {
            let indices = select_reference_indices(
                &values,
                batch,
                views,
                tokens,
                hidden,
                reference_strategy,
                context.cancellation,
            )?;
            reorder_views_in_place(
                &mut values,
                batch,
                views,
                tokens,
                hidden,
                &indices,
                false,
                context.cancellation,
            )?;
            reorder_views_in_place(
                &mut local_values,
                batch,
                views,
                tokens,
                hidden,
                &indices,
                false,
                context.cancellation,
            )?;
            reference_indices = Some(indices);
        }
        if resource
            .backbone
            .configuration
            .alternate_attention_start
            .is_some_and(|start| layer == start)
        {
            inject_learned_camera_tokens(
                resource,
                &mut values,
                batch,
                views,
                tokens,
                hidden,
                camera_token,
                context.cancellation,
            )?;
        }
        let global = resource
            .backbone
            .configuration
            .alternate_attention_start
            .is_some_and(|start| layer >= start && layer % 2 == 1);
        if global {
            values = transformer_block(
                resource,
                backend,
                &values,
                batch,
                views
                    .checked_mul(tokens)
                    .ok_or(NativeDino2Error::ShapeOverflow)?,
                hidden,
                layer,
                patch_height,
                patch_width,
                views,
                true,
                context,
            )?;
        } else {
            values = transformer_block(
                resource,
                backend,
                &values,
                flat_views,
                tokens,
                hidden,
                layer,
                patch_height,
                patch_width,
                1,
                false,
                context,
            )?;
            local_values.clone_from(&values);
        }
        if resource
            .backbone
            .configuration
            .output_layers
            .contains(&layer)
        {
            let output_rows = flat_views
                .checked_mul(tokens)
                .ok_or(NativeDino2Error::ShapeOverflow)?;
            let mut output_values = if resource.backbone.configuration.concatenate_camera_token {
                concatenate_last_dimension(
                    &local_values,
                    &values,
                    output_rows,
                    hidden,
                    context.cancellation,
                )?
            } else {
                values.clone()
            };
            let channels = if resource.backbone.configuration.concatenate_camera_token {
                hidden
                    .checked_mul(2)
                    .ok_or(NativeDino2Error::ShapeOverflow)?
            } else {
                hidden
            };
            if let Some(indices) = reference_indices.as_ref() {
                reorder_views_in_place(
                    &mut output_values,
                    batch,
                    views,
                    tokens,
                    channels,
                    indices,
                    true,
                    context.cancellation,
                )?;
            }
            let normalized = final_backbone_norm(
                resource,
                backend,
                &output_values,
                flat_views,
                tokens,
                channels,
                hidden,
                context,
            )?;
            let camera_values = collect_token(
                if ordinary {
                    &normalized
                } else {
                    &output_values
                },
                flat_views,
                tokens,
                channels,
                0,
                context.cancellation,
            )?;
            outputs.push(NativeDino2Feature {
                patch_values: drop_first_token(
                    &normalized,
                    flat_views,
                    tokens,
                    channels,
                    context.cancellation,
                )?,
                camera_values,
                patches,
                channels,
            });
        }
    }
    if outputs.len() != 4 {
        return Err(NativeDino2Error::UnsupportedArchitecture);
    }
    Ok(outputs)
}

fn interpolated_position_embeddings(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    patch_height: usize,
    patch_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
    let tensor = resource.execution_tensor("native.backbone.embeddings.position_embeddings")?;
    let shape = shape_usize(tensor)?;
    let hidden = resource.backbone.configuration.hidden;
    let source_patches = shape
        .get(1)
        .copied()
        .and_then(|tokens| tokens.checked_sub(1))
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let source_side = integer_square_root(source_patches);
    if source_side * source_side != source_patches {
        return Err(NativeDino2Error::UnsupportedArchitecture);
    }
    let values = tensor_values(tensor, context.cancellation)?;
    if source_side == patch_height && source_side == patch_width {
        return Ok(values);
    }
    let patch_values = values
        .get(hidden..)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let patch_tensor = tensor_from_f32_with_context_exact_native(
        backend,
        &[
            1,
            u64::try_from(source_side).map_err(|_| NativeDino2Error::ShapeOverflow)?,
            u64::try_from(source_side).map_err(|_| NativeDino2Error::ShapeOverflow)?,
            u64::try_from(hidden).map_err(|_| NativeDino2Error::ShapeOverflow)?,
        ],
        patch_values,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let patch_tensor =
        tensor_permute_exact_native(&patch_tensor, &[0, 3, 1, 2], context.cancellation)?;
    let resized = interpolate_tensor_with_context_exact_native(
        backend,
        &patch_tensor,
        &InterpolateConfiguration {
            output_size: None,
            scale_factor: Some(vec![
                (patch_height as f64 + 0.1) / source_side as f64,
                (patch_width as f64 + 0.1) / source_side as f64,
            ]),
            mode: InterpolateMode::Bicubic,
            align_corners: Some(false),
            recompute_scale_factor: None,
            antialias: false,
        },
        context,
    )?;
    if spatial_size(&resized)? != (patch_height, patch_width) {
        return Err(NativeDino2Error::ShapeOverflow);
    }
    let resized = tensor_permute_exact_native(&resized, &[0, 2, 3, 1], context.cancellation)?;
    let resized = tensor_to_f32_with_context_exact_native(backend, &resized, context)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(hidden + resized.len())
        .map_err(|_| NativeDino2Error::Allocation)?;
    output.extend_from_slice(
        values
            .get(..hidden)
            .ok_or(NativeDino2Error::ShapeOverflow)?,
    );
    output.extend_from_slice(&resized);
    Ok(output)
}

fn transformer_block(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    input: &[f32],
    batch: usize,
    tokens: usize,
    hidden: usize,
    layer: usize,
    patch_height: usize,
    patch_width: usize,
    view_groups: usize,
    global_positions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
    let prefix = format!("native.backbone.encoder.layer.{layer}");
    let normalized = layer_norm_values(
        resource,
        backend,
        input,
        &[batch, tokens, hidden],
        &format!("{prefix}.norm1"),
        1.0e-6,
        context,
    )?;
    let mut query = linear_values(
        resource,
        backend,
        &normalized,
        &[batch, tokens, hidden],
        &format!("{prefix}.attention.attention.query"),
        context,
    )?;
    let mut key = linear_values(
        resource,
        backend,
        &normalized,
        &[batch, tokens, hidden],
        &format!("{prefix}.attention.attention.key"),
        context,
    )?;
    let value = linear_values(
        resource,
        backend,
        &normalized,
        &[batch, tokens, hidden],
        &format!("{prefix}.attention.attention.value"),
        context,
    )?;
    let heads = resource.backbone.configuration.attention_heads;
    let head_dimension = hidden
        .checked_div(heads)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    if resource
        .backbone
        .configuration
        .qknorm_start
        .is_some_and(|start| layer >= start)
    {
        query = layer_norm_values(
            resource,
            backend,
            &query,
            &[batch, tokens, heads, head_dimension],
            &format!("{prefix}.attention.q_norm"),
            1.0e-6,
            context,
        )?;
        key = layer_norm_values(
            resource,
            backend,
            &key,
            &[batch, tokens, heads, head_dimension],
            &format!("{prefix}.attention.k_norm"),
            1.0e-6,
            context,
        )?;
    }
    if resource
        .backbone
        .configuration
        .rope_start
        .is_some_and(|start| layer >= start)
    {
        let (y_positions, x_positions) = rotary_positions(
            batch,
            tokens,
            patch_height,
            patch_width,
            view_groups,
            global_positions,
        )?;
        query = apply_da3_rotary(
            &query,
            batch,
            tokens,
            heads,
            head_dimension,
            &y_positions,
            &x_positions,
            context.cancellation,
        )?;
        key = apply_da3_rotary(
            &key,
            batch,
            tokens,
            heads,
            head_dimension,
            &y_positions,
            &x_positions,
            context.cancellation,
        )?;
    }
    let attended = scaled_dot_product_attention_with_context(
        backend,
        AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch,
            query_tokens: tokens,
            key_tokens: tokens,
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
    let projected = linear_values(
        resource,
        backend,
        &attended,
        &[batch, tokens, hidden],
        &format!("{prefix}.attention.output.dense"),
        context,
    )?;
    let scaled = multiply_by_state(
        resource,
        backend,
        &projected,
        &[batch, tokens, hidden],
        &format!("{prefix}.layer_scale1.lambda1"),
        context,
    )?;
    let residual = add_value_slices(backend, input, &scaled, context)?;
    let normalized = layer_norm_values(
        resource,
        backend,
        &residual,
        &[batch, tokens, hidden],
        &format!("{prefix}.norm2"),
        1.0e-6,
        context,
    )?;
    let feed_forward = if resource.backbone.configuration.swiglu {
        swiglu_values(
            resource,
            backend,
            &normalized,
            batch,
            tokens,
            hidden,
            &prefix,
            context,
        )?
    } else {
        let projected = linear_values(
            resource,
            backend,
            &normalized,
            &[batch, tokens, hidden],
            &format!("{prefix}.mlp.fc1"),
            context,
        )?;
        let activated = gelu_with_context_exact_native(
            backend,
            &projected,
            GeluApproximation::None,
            DeviceId::CPU,
            context,
        )?;
        linear_values(
            resource,
            backend,
            &activated,
            &[batch, tokens, hidden * 4],
            &format!("{prefix}.mlp.fc2"),
            context,
        )?
    };
    let scaled = multiply_by_state(
        resource,
        backend,
        &feed_forward,
        &[batch, tokens, hidden],
        &format!("{prefix}.layer_scale2.lambda1"),
        context,
    )?;
    add_value_slices(backend, &residual, &scaled, context)
}

fn rotary_positions(
    _batch: usize,
    tokens: usize,
    patch_height: usize,
    patch_width: usize,
    view_groups: usize,
    global: bool,
) -> Result<(Vec<usize>, Vec<usize>), NativeDino2Error> {
    let patches = patch_height
        .checked_mul(patch_width)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let per_view_tokens = patches
        .checked_add(1)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    if tokens != per_view_tokens * view_groups {
        return Err(NativeDino2Error::ShapeOverflow);
    }
    let mut y_positions = Vec::new();
    let mut x_positions = Vec::new();
    y_positions
        .try_reserve_exact(tokens)
        .map_err(|_| NativeDino2Error::Allocation)?;
    x_positions
        .try_reserve_exact(tokens)
        .map_err(|_| NativeDino2Error::Allocation)?;
    for _ in 0..view_groups {
        y_positions.push(0);
        x_positions.push(0);
        for patch in 0..patches {
            if global {
                y_positions.push(1);
                x_positions.push(1);
            } else {
                y_positions.push(patch / patch_width + 1);
                x_positions.push(patch % patch_width + 1);
            }
        }
    }
    Ok((y_positions, x_positions))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_da3_rotary(
    values: &[f32],
    batch: usize,
    tokens: usize,
    heads: usize,
    head_dimension: usize,
    y_positions: &[usize],
    x_positions: &[usize],
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeDino2Error> {
    let axis_dimension = head_dimension / 2;
    if !head_dimension.is_multiple_of(4)
        || y_positions.len() != tokens
        || x_positions.len() != tokens
        || values.len() != batch * tokens * heads * head_dimension
    {
        return Err(NativeDino2Error::UnsupportedArchitecture);
    }
    let axis_count = batch * tokens * heads * axis_dimension;
    let mut vertical = filled_f32(axis_count, 0.0)?;
    let mut horizontal = filled_f32(axis_count, 0.0)?;
    for row in 0..batch * tokens * heads {
        if row.is_multiple_of(4_096) {
            cancellation.check()?;
        }
        vertical[row * axis_dimension..(row + 1) * axis_dimension]
            .copy_from_slice(&values[row * head_dimension..row * head_dimension + axis_dimension]);
        horizontal[row * axis_dimension..(row + 1) * axis_dimension].copy_from_slice(
            &values[row * head_dimension + axis_dimension..(row + 1) * head_dimension],
        );
    }
    let vertical_table = precompute_rotary_table(
        RotaryTableRequest {
            positions: RotaryPositions::Scalar(RotaryPositionSequence::Unsigned(y_positions)),
            rotary_dimension: axis_dimension,
            axis_dimensions: &[],
            theta: 100.0,
            scaling: RotaryScaling::None,
            frequency_layout: RotaryFrequencyLayout::Global,
        },
        cancellation,
    )?;
    let horizontal_table = precompute_rotary_table(
        RotaryTableRequest {
            positions: RotaryPositions::Scalar(RotaryPositionSequence::Unsigned(x_positions)),
            rotary_dimension: axis_dimension,
            axis_dimensions: &[],
            theta: 100.0,
            scaling: RotaryScaling::None,
            frequency_layout: RotaryFrequencyLayout::Global,
        },
        cancellation,
    )?;
    let vertical = apply_rotary_table(
        &vertical,
        batch,
        tokens,
        heads,
        axis_dimension,
        &vertical_table,
        RotaryPairLayout::SplitHalf,
        cancellation,
    )?;
    let horizontal = apply_rotary_table(
        &horizontal,
        batch,
        tokens,
        heads,
        axis_dimension,
        &horizontal_table,
        RotaryPairLayout::SplitHalf,
        cancellation,
    )?;
    let mut output = filled_f32(values.len(), 0.0)?;
    for row in 0..batch * tokens * heads {
        if row.is_multiple_of(4_096) {
            cancellation.check()?;
        }
        output[row * head_dimension..row * head_dimension + axis_dimension]
            .copy_from_slice(&vertical[row * axis_dimension..(row + 1) * axis_dimension]);
        output[row * head_dimension + axis_dimension..(row + 1) * head_dimension]
            .copy_from_slice(&horizontal[row * axis_dimension..(row + 1) * axis_dimension]);
    }
    Ok(output)
}

fn layer_norm_values(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    prefix: &str,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
    let normalized = *shape.last().ok_or(NativeDino2Error::ShapeOverflow)?;
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
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    input: &[f32],
    input_shape: &[usize],
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
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
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    input: &[f32],
    shape: &[usize],
    key: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
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

fn swiglu_values(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    input: &[f32],
    batch: usize,
    tokens: usize,
    hidden: usize,
    prefix: &str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
    let projected = linear_values(
        resource,
        backend,
        input,
        &[batch, tokens, hidden],
        &format!("{prefix}.mlp.weights_in"),
        context,
    )?;
    let intermediate = projected
        .len()
        .checked_div(batch * tokens * 2)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let mut first = filled_f32(batch * tokens * intermediate, 0.0)?;
    let mut second = filled_f32(first.len(), 0.0)?;
    for row in 0..batch * tokens {
        context.check()?;
        let source = row * intermediate * 2;
        let destination = row * intermediate;
        first[destination..destination + intermediate]
            .copy_from_slice(&projected[source..source + intermediate]);
        second[destination..destination + intermediate]
            .copy_from_slice(&projected[source + intermediate..source + intermediate * 2]);
    }
    let first = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[batch, tokens, intermediate])?,
        &first,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let first = silu_tensor_with_context_exact_native(backend, &first, context)?;
    let second = tensor_from_f32_with_context_exact_native(
        backend,
        &shape_u64(&[batch, tokens, intermediate])?,
        &second,
        DType::F32,
        DeviceId::CPU,
        context,
    )?;
    let product = mul_method_with_context_exact_native(backend, &first, &second, context)?;
    let product = tensor_to_f32_with_context_exact_native(backend, &product, context)?;
    linear_values(
        resource,
        backend,
        &product,
        &[batch, tokens, intermediate],
        &format!("{prefix}.mlp.weights_out"),
        context,
    )
}

fn add_value_slices(
    backend: &CpuBackend,
    left: &[f32],
    right: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
    if left.len() != right.len() {
        return Err(NativeDino2Error::ShapeOverflow);
    }
    let shape = [u64::try_from(left.len()).map_err(|_| NativeDino2Error::ShapeOverflow)?];
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn select_reference_indices(
    values: &[f32],
    batch: usize,
    views: usize,
    tokens: usize,
    channels: usize,
    strategy: NativeDino2ReferenceStrategy,
    cancellation: &CancellationToken,
) -> Result<Vec<usize>, NativeDino2Error> {
    select_reference_indices_with_check(values, batch, views, tokens, channels, strategy, || {
        cancellation.check().map_err(NativeDino2Error::from)
    })
}

#[allow(clippy::too_many_arguments)]
fn select_reference_indices_with_check(
    values: &[f32],
    batch: usize,
    views: usize,
    tokens: usize,
    channels: usize,
    strategy: NativeDino2ReferenceStrategy,
    mut check_cancellation: impl FnMut() -> Result<(), NativeDino2Error>,
) -> Result<Vec<usize>, NativeDino2Error> {
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(batch)
        .map_err(|_| NativeDino2Error::Allocation)?;
    for batch_index in 0..batch {
        check_cancellation()?;
        match strategy {
            NativeDino2ReferenceStrategy::First => selected.push(0),
            NativeDino2ReferenceStrategy::Middle => selected.push(views / 2),
            NativeDino2ReferenceStrategy::SaddleBalanced
            | NativeDino2ReferenceStrategy::SaddleSimRange => {
                let mut normalized = filled_f32(views * channels, 0.0)?;
                let mut norms = filled_f32(views, 0.0)?;
                let mut variances = filled_f32(views, 0.0)?;
                for view in 0..views {
                    check_cancellation()?;
                    let offset = ((batch_index * views + view) * tokens) * channels;
                    let class = values
                        .get(offset..offset + channels)
                        .ok_or(NativeDino2Error::ShapeOverflow)?;
                    let norm = class.iter().map(|value| value * value).sum::<f32>().sqrt();
                    norms[view] = norm;
                    let divisor = norm;
                    for channel in 0..channels {
                        if channel.is_multiple_of(4_096) {
                            check_cancellation()?;
                        }
                        normalized[view * channels + channel] = class[channel] / divisor;
                    }
                    let mean = normalized[view * channels..(view + 1) * channels]
                        .iter()
                        .copied()
                        .sum::<f32>()
                        / channels as f32;
                    variances[view] = normalized[view * channels..(view + 1) * channels]
                        .iter()
                        .map(|value| (value - mean) * (value - mean))
                        .sum::<f32>()
                        / channels.saturating_sub(1).max(1) as f32;
                }
                let mut similarities = filled_f32(views, 0.0)?;
                let mut similarity_ranges = filled_f32(views, 0.0)?;
                for view in 0..views {
                    check_cancellation()?;
                    let mut minimum = f32::INFINITY;
                    let mut maximum = f32::NEG_INFINITY;
                    for other in 0..views {
                        check_cancellation()?;
                        let mut dot = 0.0_f32;
                        for channel in 0..channels {
                            if channel.is_multiple_of(4_096) {
                                check_cancellation()?;
                            }
                            dot += normalized[view * channels + channel]
                                * normalized[other * channels + channel];
                        }
                        similarities[view] += dot - if view == other { 1.0 } else { 0.0 };
                        let without_diagonal = dot - if view == other { 1.0 } else { 0.0 };
                        minimum = minimum.min(without_diagonal);
                        maximum = maximum.max(without_diagonal);
                    }
                    similarities[view] /= views.saturating_sub(1).max(1) as f32;
                    similarity_ranges[view] = maximum - minimum;
                }
                if strategy == NativeDino2ReferenceStrategy::SaddleSimRange {
                    let mut reference = 0;
                    for view in 1..views {
                        if similarity_ranges[view] > similarity_ranges[reference] {
                            reference = view;
                        }
                    }
                    selected.push(reference);
                    continue;
                }
                normalize_metric(&mut similarities);
                normalize_metric(&mut norms);
                normalize_metric(&mut variances);
                let reference = (0..views)
                    .min_by(|left, right| {
                        let left_score = (similarities[*left] - 0.5).abs()
                            + (norms[*left] - 0.5).abs()
                            + (variances[*left] - 0.5).abs();
                        let right_score = (similarities[*right] - 0.5).abs()
                            + (norms[*right] - 0.5).abs()
                            + (variances[*right] - 0.5).abs();
                        left_score.total_cmp(&right_score)
                    })
                    .ok_or(NativeDino2Error::ShapeOverflow)?;
                selected.push(reference);
            }
        }
    }
    Ok(selected)
}

fn normalize_metric(values: &mut [f32]) {
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for value in values {
        *value = (*value - minimum) / (maximum - minimum + 1.0e-8);
    }
}

#[allow(clippy::too_many_arguments)]
fn reorder_views_in_place(
    values: &mut Vec<f32>,
    batch: usize,
    views: usize,
    tokens: usize,
    channels: usize,
    references: &[usize],
    restore: bool,
    cancellation: &CancellationToken,
) -> Result<(), NativeDino2Error> {
    let mut output = filled_f32(values.len(), 0.0)?;
    let view_stride = tokens
        .checked_mul(channels)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    for batch_index in 0..batch {
        cancellation.check()?;
        let reference = *references
            .get(batch_index)
            .ok_or(NativeDino2Error::ShapeOverflow)?;
        if reference >= views {
            return Err(NativeDino2Error::ShapeOverflow);
        }
        for destination in 0..views {
            let source = if restore {
                if destination < reference {
                    destination + 1
                } else if destination == reference {
                    0
                } else {
                    destination
                }
            } else if destination == 0 {
                reference
            } else if destination <= reference {
                destination - 1
            } else {
                destination
            };
            let source_offset = (batch_index * views + source) * view_stride;
            let destination_offset = (batch_index * views + destination) * view_stride;
            output[destination_offset..destination_offset + view_stride]
                .copy_from_slice(&values[source_offset..source_offset + view_stride]);
        }
    }
    *values = output;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn inject_learned_camera_tokens(
    resource: &NativeDino2Execution<'_>,
    values: &mut [f32],
    batch: usize,
    views: usize,
    tokens: usize,
    channels: usize,
    computed: Option<&[f32]>,
    cancellation: &CancellationToken,
) -> Result<(), NativeDino2Error> {
    let learned = if computed.is_none() {
        Some(tensor_values(
            resource.execution_tensor("native.backbone.embeddings.camera_token")?,
            cancellation,
        )?)
    } else {
        None
    };
    for batch_index in 0..batch {
        cancellation.check()?;
        for view in 0..views {
            let destination = ((batch_index * views + view) * tokens) * channels;
            let source_values = if let Some(computed) = computed {
                let source = (batch_index * views + view) * channels;
                computed
                    .get(source..source + channels)
                    .ok_or(NativeDino2Error::ShapeOverflow)?
            } else {
                let learned = learned.as_ref().ok_or(NativeDino2Error::ShapeOverflow)?;
                let source = if view == 0 { 0 } else { channels };
                &learned[source..source + channels]
            };
            values[destination..destination + channels].copy_from_slice(source_values);
        }
    }
    Ok(())
}

fn concatenate_last_dimension(
    left: &[f32],
    right: &[f32],
    rows: usize,
    channels: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeDino2Error> {
    if left.len() != rows * channels || right.len() != left.len() {
        return Err(NativeDino2Error::ShapeOverflow);
    }
    let mut output = filled_f32(rows * channels * 2, 0.0)?;
    for row in 0..rows {
        if row.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        let destination = row * channels * 2;
        output[destination..destination + channels]
            .copy_from_slice(&left[row * channels..(row + 1) * channels]);
        output[destination + channels..destination + channels * 2]
            .copy_from_slice(&right[row * channels..(row + 1) * channels]);
    }
    Ok(output)
}

fn collect_token(
    values: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    token: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeDino2Error> {
    let mut output = filled_f32(batch * channels, 0.0)?;
    for row in 0..batch {
        if row.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        let source = (row * tokens + token) * channels;
        output[row * channels..(row + 1) * channels]
            .copy_from_slice(&values[source..source + channels]);
    }
    Ok(output)
}

fn final_backbone_norm(
    resource: &NativeDino2Execution<'_>,
    backend: &CpuBackend,
    values: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    hidden: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, NativeDino2Error> {
    if channels == hidden {
        return layer_norm_values(
            resource,
            backend,
            values,
            &[batch, tokens, hidden],
            "native.backbone.layernorm",
            1.0e-6,
            context,
        );
    }
    if channels != hidden * 2 {
        return Err(NativeDino2Error::ShapeOverflow);
    }
    let mut left = filled_f32(batch * tokens * hidden, 0.0)?;
    let mut right = filled_f32(left.len(), 0.0)?;
    for row in 0..batch * tokens {
        context.check()?;
        left[row * hidden..(row + 1) * hidden]
            .copy_from_slice(&values[row * channels..row * channels + hidden]);
        right[row * hidden..(row + 1) * hidden]
            .copy_from_slice(&values[row * channels + hidden..(row + 1) * channels]);
    }
    let right = layer_norm_values(
        resource,
        backend,
        &right,
        &[batch, tokens, hidden],
        "native.backbone.layernorm",
        1.0e-6,
        context,
    )?;
    concatenate_last_dimension(&left, &right, batch * tokens, hidden, context.cancellation)
}

fn drop_first_token(
    values: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, NativeDino2Error> {
    let patches = tokens
        .checked_sub(1)
        .ok_or(NativeDino2Error::ShapeOverflow)?;
    let mut output = filled_f32(batch * patches * channels, 0.0)?;
    for row in 0..batch {
        if row.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        let source = (row * tokens + 1) * channels;
        let destination = row * patches * channels;
        output[destination..destination + patches * channels]
            .copy_from_slice(&values[source..source + patches * channels]);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn add_state(
    states: &mut Vec<NativeDino2StateSpecification>,
    key: &str,
    shape: &[usize],
) -> Result<(), NativeDino2Error> {
    states
        .try_reserve(1)
        .map_err(|_| NativeDino2Error::Allocation)?;
    states.push(NativeDino2StateSpecification {
        key: key.to_owned(),
        shape: shape
            .iter()
            .map(|value| u64::try_from(*value).map_err(|_| NativeDino2Error::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?,
    });
    Ok(())
}

fn add_affine(
    states: &mut Vec<NativeDino2StateSpecification>,
    prefix: &str,
    channels: usize,
) -> Result<(), NativeDino2Error> {
    add_state(states, &format!("{prefix}.weight"), &[channels])?;
    add_state(states, &format!("{prefix}.bias"), &[channels])
}

fn add_linear(
    states: &mut Vec<NativeDino2StateSpecification>,
    prefix: &str,
    output: usize,
    input: usize,
    bias: bool,
) -> Result<(), NativeDino2Error> {
    add_state(states, &format!("{prefix}.weight"), &[output, input])?;
    if bias {
        add_state(states, &format!("{prefix}.bias"), &[output])?;
    }
    Ok(())
}

fn add_conv(
    states: &mut Vec<NativeDino2StateSpecification>,
    prefix: &str,
    output: usize,
    input: usize,
    kernel: usize,
    bias: bool,
) -> Result<(), NativeDino2Error> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use sim_tensor::{CpuWorkspaceAuthority, ImageTensor, StreamId};

    use crate::depth_anything_3::{
        DepthAnything3FixtureMutation, execute_reduced_dino2_ordinary_for_fixture,
    };

    const ORACLE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/dinov2-backbone-owner-foundation/oracle.json"
    ));

    fn value_bits(value: &Value) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(value
            .as_array()
            .ok_or("DINOv2 bit vector is missing")?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .map(f32::from_bits)
                    .ok_or("DINOv2 bit value is invalid")
            })
            .collect::<Result<Vec<_>, _>>()?)
    }

    fn fixture_image(
        backend: &CpuBackend,
        oracle: &Value,
        context: &ExecutionContext<'_>,
    ) -> Result<ImageTensor, Box<dyn std::error::Error>> {
        let shape = oracle["input_shape"]
            .as_array()
            .ok_or("DINOv2 input shape is missing")?
            .iter()
            .map(|value| value.as_u64().ok_or("DINOv2 input shape is invalid"))
            .collect::<Result<Vec<_>, _>>()?;
        let values = value_bits(&oracle["input_bits"])?;
        Ok(ImageTensor::from_tensor(
            tensor_from_f32_with_context_exact_native(
                backend,
                &shape,
                &values,
                DType::F32,
                DeviceId::CPU,
                context,
            )?,
        )?)
    }

    fn assert_features(
        actual: &[NativeDino2Feature],
        expected: &Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected = expected
            .as_array()
            .ok_or("DINOv2 ordinary output list is missing")?;
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.patches, 4);
            assert_eq!(actual.channels, 4);
            assert_eq!(actual.patch_values, value_bits(&expected["patch_bits"])?);
            assert_eq!(actual.camera_values, value_bits(&expected["class_bits"])?);
        }
        Ok(())
    }

    fn mask_token_identity(
        source_dtype: DType,
        source: &Tensor,
        projected: &Tensor,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let source_dtype = match source_dtype {
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::F32 => "f32",
            _ => return Err("invalid DINOv2 mask-token fixture dtype".into()),
        };
        let key = "native.backbone.embeddings.mask_token";
        let source_bytes = source.contiguous_bytes()?;
        let projected_bytes = projected.contiguous_bytes()?;
        let mut digest = Sha256::new();
        digest.update(b"dinov2-forward-unused-mask-token-v1");
        digest.update(u64::try_from(key.len())?.to_le_bytes());
        digest.update(key.as_bytes());
        digest.update(u64::try_from(source_dtype.len())?.to_le_bytes());
        digest.update(source_dtype.as_bytes());
        digest.update(2_u64.to_le_bytes());
        digest.update(1_u64.to_le_bytes());
        digest.update(4_u64.to_le_bytes());
        digest.update(u64::try_from(source_bytes.len())?.to_le_bytes());
        digest.update(source_bytes);
        digest.update(u64::try_from(projected_bytes.len())?.to_le_bytes());
        digest.update(projected_bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    #[test]
    fn ordinary_route_matches_independent_f32_f16_bf16_oracle()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 64 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let oracle: Value = serde_json::from_str(ORACLE)?;
        let image = fixture_image(&backend, &oracle, &context)?;
        for (name, dtype) in [
            ("f32", DType::F32),
            ("f16", DType::F16),
            ("bf16", DType::Bf16),
        ] {
            let (features, forward) = execute_reduced_dino2_ordinary_for_fixture(
                &backend,
                &image,
                dtype,
                None,
                workspace_bytes,
                &context,
            )?;
            assert_features(&features, &oracle["ordinary_routes"][name])?;
            assert_features(
                std::slice::from_ref(&forward),
                &Value::Array(vec![oracle["ordinary_routes"][name][3].clone()]),
            )?;
        }
        Ok(())
    }

    #[test]
    fn ordinary_route_mutations_match_independent_oracle() -> Result<(), Box<dyn std::error::Error>>
    {
        let workspace_bytes = 64 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let oracle: Value = serde_json::from_str(ORACLE)?;
        let image = fixture_image(&backend, &oracle, &context)?;
        for (name, mutation) in oracle["ordinary_mutations"]
            .as_object()
            .ok_or("DINOv2 mutations are missing")?
        {
            let state_key = mutation["state_key"]
                .as_str()
                .ok_or("DINOv2 mutation key is missing")?;
            let lane = mutation["lane"]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or("DINOv2 mutation lane is invalid")?;
            let delta = mutation["delta_bits"]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .map(f32::from_bits)
                .ok_or("DINOv2 mutation delta is invalid")?;
            let (features, forward) = execute_reduced_dino2_ordinary_for_fixture(
                &backend,
                &image,
                DType::F32,
                Some(DepthAnything3FixtureMutation {
                    state_key,
                    lane,
                    delta,
                }),
                workspace_bytes,
                &context,
            )?;
            assert_features(&features, &mutation["outputs"])
                .map_err(|error| format!("DINOv2 mutation {name}: {error}"))?;
            assert_features(
                std::slice::from_ref(&forward),
                &Value::Array(vec![mutation["outputs"][3].clone()]),
            )?;
            let changes_output = mutation["changes_output"]
                .as_bool()
                .ok_or("DINOv2 mutation disposition is missing")?;
            if changes_output {
                assert_ne!(
                    mutation["outputs"], oracle["ordinary_routes"]["f32"],
                    "DINOv2 mutation {name} must be discriminating"
                );
            } else {
                assert_eq!(name, "forward_unused_mask_token");
                assert_eq!(mutation["outputs"], oracle["ordinary_routes"]["f32"]);
                assert_ne!(
                    mutation["mask_token"]["identity_sha256"],
                    oracle["forward_unused_mask_token"]["f32"]["identity_sha256"]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn ordinary_mask_token_is_strict_projected_identity_but_forward_unused()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 4 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let oracle: Value = serde_json::from_str(ORACLE)?;
        let backbone = NativeDino2Backbone::new(NativeDino2Configuration {
            prefix: "native.backbone",
            hidden: 4,
            layer_count: 4,
            attention_heads: 1,
            patch: 2,
            image: 4,
            qknorm_start: None,
            alternate_attention_start: None,
            rope_start: None,
            concatenate_camera_token: false,
            use_mask_token: true,
            swiglu: false,
            output_layers: [0, 1, 2, 3],
        })?;
        assert!(backbone.state_manifest()?.iter().any(|specification| {
            specification.key == "native.backbone.embeddings.mask_token"
                && specification.shape == [1, 4]
        }));
        for (name, dtype) in [
            ("f32", DType::F32),
            ("f16", DType::F16),
            ("bf16", DType::Bf16),
        ] {
            let source = tensor_from_f32_with_context_exact_native(
                &backend,
                &[1, 4],
                &[-0.035, -0.0025, 0.03, -0.01],
                dtype,
                DeviceId::CPU,
                &context,
            )?;
            let projected = backbone.project_state_tensor(
                &backend,
                "native.backbone.embeddings.mask_token",
                &source,
                &context,
            )?;
            let expected = &oracle["forward_unused_mask_token"][name];
            assert_eq!(
                source.contiguous_bytes()?,
                expected["source_bytes"]
                    .as_array()
                    .ok_or("mask-token source bytes are missing")?
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| u8::try_from(value).ok())
                            .ok_or("mask-token source byte is invalid")
                    })
                    .collect::<Result<Vec<_>, _>>()?
            );
            assert_eq!(
                tensor_to_f32_with_context_exact_native(&backend, &projected, &context)?,
                value_bits(&expected["projected_bits"])?
            );
            assert_eq!(
                mask_token_identity(dtype, &source, &projected)?,
                expected["identity_sha256"]
                    .as_str()
                    .ok_or("mask-token identity is missing")?
            );
            assert_eq!(
                source.storage_byte_len(),
                expected["storage_bytes"]
                    .as_u64()
                    .ok_or("mask-token storage size is missing")?
            );
        }

        let da3 = NativeDino2Backbone::new(NativeDino2Configuration {
            use_mask_token: false,
            ..backbone.configuration
        })?;
        let source = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 4],
            &[0.0; 4],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        assert!(matches!(
            da3.project_state_tensor(
                &backend,
                "native.backbone.embeddings.mask_token",
                &source,
                &context,
            ),
            Err(NativeDino2Error::UnexpectedState(_))
        ));
        Ok(())
    }

    #[test]
    fn owner_manifest_is_fallible_and_distinguishes_mlp_variants()
    -> Result<(), Box<dyn std::error::Error>> {
        let ordinary = NativeDino2Backbone::new(NativeDino2Configuration {
            prefix: "native.backbone",
            hidden: 4,
            layer_count: 4,
            attention_heads: 1,
            patch: 2,
            image: 4,
            qknorm_start: None,
            alternate_attention_start: None,
            rope_start: None,
            concatenate_camera_token: false,
            use_mask_token: true,
            swiglu: false,
            output_layers: [0, 1, 2, 3],
        })?;
        let ordinary = ordinary.state_manifest()?;
        assert!(
            ordinary
                .iter()
                .any(|state| state.key.ends_with("mlp.fc1.weight"))
        );
        assert!(
            !ordinary
                .iter()
                .any(|state| state.key.ends_with("mlp.weights_in.weight"))
        );

        let swiglu = NativeDino2Backbone::new(NativeDino2Configuration {
            swiglu: true,
            ..NativeDino2Configuration {
                prefix: "native.backbone",
                hidden: 24,
                layer_count: 4,
                attention_heads: 1,
                patch: 2,
                image: 4,
                qknorm_start: None,
                alternate_attention_start: None,
                rope_start: None,
                concatenate_camera_token: false,
                use_mask_token: true,
                swiglu: false,
                output_layers: [0, 1, 2, 3],
            }
        })?
        .state_manifest()?;
        assert!(
            swiglu
                .iter()
                .any(|state| state.key.ends_with("mlp.weights_in.weight"))
        );
        assert!(
            !swiglu
                .iter()
                .any(|state| state.key.ends_with("mlp.fc1.weight"))
        );
        for output_layers in [[0, 0, 2, 3], [0, 1, 2, 4]] {
            let invalid = NativeDino2Backbone::new(NativeDino2Configuration {
                prefix: "native.backbone",
                hidden: 4,
                layer_count: 4,
                attention_heads: 1,
                patch: 2,
                image: 4,
                qknorm_start: None,
                alternate_attention_start: None,
                rope_start: None,
                concatenate_camera_token: false,
                use_mask_token: true,
                swiglu: false,
                output_layers,
            });
            assert!(matches!(
                invalid,
                Err(NativeDino2Error::UnsupportedArchitecture)
            ));
        }
        let overflow = NativeDino2Backbone::new(NativeDino2Configuration {
            prefix: "native.backbone",
            hidden: usize::MAX,
            layer_count: 4,
            attention_heads: 1,
            patch: 1,
            image: 1,
            qknorm_start: None,
            alternate_attention_start: None,
            rope_start: None,
            concatenate_camera_token: false,
            use_mask_token: true,
            swiglu: false,
            output_layers: [0, 1, 2, 3],
        })?
        .state_manifest();
        assert!(matches!(overflow, Err(NativeDino2Error::ShapeOverflow)));
        Ok(())
    }

    #[test]
    fn ordinary_memory_and_cancellation_fail_before_owner_mutation()
    -> Result<(), Box<dyn std::error::Error>> {
        let workspace_bytes = 4 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancellation,
        );
        let input = tensor_from_f32_with_context_exact_native(
            &backend,
            &[1, 3, 4, 4],
            &[0.0; 48],
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        let backbone = NativeDino2Backbone::new(NativeDino2Configuration {
            prefix: "native.backbone",
            hidden: 4,
            layer_count: 4,
            attention_heads: 1,
            patch: 2,
            image: 4,
            qknorm_start: None,
            alternate_attention_start: None,
            rope_start: None,
            concatenate_camera_token: false,
            use_mask_token: true,
            swiglu: false,
            output_layers: [0, 1, 2, 3],
        })?;
        let error = backbone
            .project_state_tensor(
                &backend,
                "native.backbone.encoder.layer.0.unknown.weight",
                &input,
                &context,
            )
            .expect_err("unknown DINOv2 state must fail");
        assert!(matches!(error, NativeDino2Error::UnexpectedState(_)));
        let error = backbone
            .project_state_tensor(
                &backend,
                "native.backbone.embeddings.cls_token",
                &input,
                &context,
            )
            .expect_err("wrong DINOv2 state shape must fail");
        assert!(matches!(error, NativeDino2Error::StateShape { .. }));
        let invalid_dtype = cast_to_with_context_exact_native(
            &backend,
            &tensor_from_f32_with_context_exact_native(
                &backend,
                &[1, 1, 4],
                &[0.0; 4],
                DType::F32,
                DeviceId::CPU,
                &context,
            )?,
            DType::I64,
            DeviceId::CPU,
            false,
            true,
            &context,
        )?;
        let error = backbone
            .project_state_tensor(
                &backend,
                "native.backbone.embeddings.cls_token",
                &invalid_dtype,
                &context,
            )
            .expect_err("unsupported DINOv2 state dtype must fail");
        assert!(matches!(error, NativeDino2Error::StateShape { .. }));
        let state = BTreeMap::new();
        let memory_before = backend.memory_snapshot().current_bytes;
        let error = backbone
            .bind(&state, 0, 0)
            .forward(&backend, &input, 1, &context)
            .expect_err("zero owner budget must fail");
        let NativeDino2Error::OutOfMemory {
            required,
            budget: 0,
        } = error
        else {
            return Err(format!("unexpected DINOv2 memory error: {error}").into());
        };
        assert!(required > 0);
        let error = backbone
            .bind(&state, required - 1, 0)
            .forward(&backend, &input, 1, &context)
            .expect_err("one byte below the owner bound must fail");
        assert!(matches!(error, NativeDino2Error::OutOfMemory { .. }));
        assert_eq!(backend.memory_snapshot().current_bytes, memory_before);
        let error = backbone
            .bind(&state, required, 0)
            .forward(&backend, &input, 1, &context)
            .expect_err("the exact bound must advance to state admission");
        assert!(matches!(error, NativeDino2Error::MissingState(_)));
        assert_eq!(backend.memory_snapshot().current_bytes, memory_before);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(workspace_bytes)?,
            &cancelled,
        );
        let error = backbone
            .bind(&state, required, 0)
            .forward(&backend, &input, 1, &cancelled_context)
            .expect_err("pre-cancelled owner execution must fail");
        assert!(matches!(
            error,
            NativeDino2Error::Tensor(TensorError::Cancelled)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, memory_before);

        let traversal_cancellation = CancellationToken::default();
        let class_tokens = vec![1.0_f32; 4 * 1_000_000];
        assert!(!traversal_cancellation.is_cancelled());
        let mut checks = 0;
        let error = select_reference_indices_with_check(
            &class_tokens,
            1,
            4,
            1,
            1_000_000,
            NativeDino2ReferenceStrategy::SaddleBalanced,
            || {
                checks += 1;
                if checks == 6 {
                    traversal_cancellation.cancel();
                }
                traversal_cancellation
                    .check()
                    .map_err(NativeDino2Error::from)
            },
        )
        .expect_err("mid-traversal cancellation must fail");
        assert_eq!(checks, 6);
        assert!(matches!(error, NativeDino2Error::Cancelled));
        Ok(())
    }
}
