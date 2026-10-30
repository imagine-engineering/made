use crate::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionMask, AttentionMaskShape,
    AttentionRequest, EmbeddingOptions, GeluApproximation, MappedModelWeights,
    NativeExecutionRequirements, NativeModule, NativeOpsError,
    scaled_dot_product_attention_with_context,
};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId,
    ExecutionContext, Layout, LinearAlgebraOperation, OperationSupport, ReductionOperation,
    StorageId, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    generated_native_diffusion::{
        NativeDiffusionTensorError, add, quick_gelu, tensor_from_f32, tensor_to_f32,
    },
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, torch_stack_with_context_exact_native,
    },
};
use std::{collections::BTreeSet, mem};
use thiserror::Error;

pub const CLIP_TEXT_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/clip_model.py";
pub const CLIP_TEXT_SOURCE_SHA256: &str =
    "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845";
pub const SD1_CLIP_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/sd1_clip.py";
pub const SD1_CLIP_SOURCE_SHA256: &str =
    "46a778884423fe070144d8a0cb8ce94189452b92b18e5bf4145e83bb61f1b4b2";
pub const CLIP_TEXT_CATALOG_SYMBOLS: [&str; 10] = [
    "CLIPAttention",
    "CLIPMLP",
    "CLIPLayer",
    "CLIPEncoder",
    "CLIPEmbeddings",
    "CLIPTextModel_",
    "CLIPTextModel",
    "SDClipModel",
    "SD1CheckpointClipModel",
    "SD1ClipModel",
];

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), ClipTextError> {
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| ClipTextError::Overflow("CLIP text digest field"))?
            .to_be_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClipTextActivation {
    QuickGelu,
    Gelu,
    GeluTanh,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClipTextIntermediate {
    None,
    Layer(isize),
    Layers(Vec<isize>),
    All,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipTextConfiguration {
    pub dtype: DType,
    pub device: DeviceId,
    pub vocabulary_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention_heads: usize,
    pub layer_count: usize,
    pub eos_token_id: u32,
    pub activation: ClipTextActivation,
    pub projection_dimension: Option<usize>,
}

impl ClipTextConfiguration {
    pub fn validate(&self) -> Result<(), ClipTextError> {
        if self.dtype != DType::F32 || self.device != DeviceId::CPU {
            return Err(ClipTextError::UnsupportedTarget {
                dtype: self.dtype,
                device: self.device,
            });
        }
        if self.vocabulary_size == 0
            || self.max_position_embeddings == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.attention_heads == 0
            || self.layer_count == 0
            || !self.hidden_size.is_multiple_of(self.attention_heads)
            || self.projection_dimension == Some(0)
            || usize::try_from(self.eos_token_id)
                .map_or_else(|_| true, |token| token >= self.vocabulary_size)
        {
            return Err(ClipTextError::InvalidConfiguration(
                "CLIP text dimensions, heads, projection, or EOS token are invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ClipTextLayerWeights {
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
pub struct ClipTextWeights {
    pub token_embedding: Tensor,
    pub position_embedding: Tensor,
    pub layers: Vec<ClipTextLayerWeights>,
    pub final_layer_norm_weight: Tensor,
    pub final_layer_norm_bias: Tensor,
}

#[derive(Clone, Copy, Debug)]
pub enum ClipTextInput<'a> {
    Tokens(&'a Tensor),
    Embeddings(&'a Tensor),
}

#[derive(Clone, Debug)]
pub struct ClipTextRequest<'a> {
    pub input: ClipTextInput<'a>,
    pub attention_mask: Option<&'a Tensor>,
    pub num_tokens: Option<&'a [usize]>,
    pub intermediate: ClipTextIntermediate,
    pub final_layer_norm_intermediate: bool,
    pub project_pooled: bool,
    pub zero_out_masked: bool,
}

#[derive(Clone, Debug)]
pub struct ClipTextOutput {
    last_hidden_state: Tensor,
    intermediate: Option<Tensor>,
    projected_pooled: Option<Tensor>,
    pooled: Tensor,
    attention_mask: Option<Tensor>,
}

impl ClipTextOutput {
    pub fn last_hidden_state(&self) -> &Tensor {
        &self.last_hidden_state
    }

    pub fn intermediate(&self) -> Option<&Tensor> {
        self.intermediate.as_ref()
    }

    pub fn projected_pooled(&self) -> Option<&Tensor> {
        self.projected_pooled.as_ref()
    }

    pub fn pooled(&self) -> &Tensor {
        &self.pooled
    }

    pub fn attention_mask(&self) -> Option<&Tensor> {
        self.attention_mask.as_ref()
    }
}

#[derive(Clone, Debug)]
struct ClipTextLayer {
    layer_norm_1: NativeModule,
    query: NativeModule,
    key: NativeModule,
    value: NativeModule,
    output: NativeModule,
    layer_norm_2: NativeModule,
    feed_forward_1: NativeModule,
    activation: Option<NativeModule>,
    quick_gelu: bool,
    feed_forward_2: NativeModule,
}

#[derive(Clone, Debug)]
pub struct NativeClipText {
    configuration: ClipTextConfiguration,
    token_embedding: NativeModule,
    position_embedding: NativeModule,
    layers: Vec<ClipTextLayer>,
    final_layer_norm: NativeModule,
    text_projection: Option<NativeModule>,
    stream: StreamId,
}

impl NativeClipText {
    pub fn new(
        configuration: ClipTextConfiguration,
        weights: ClipTextWeights,
        text_projection_weight: Option<Tensor>,
    ) -> Result<Self, ClipTextError> {
        configuration.validate()?;
        if weights.layers.len() != configuration.layer_count {
            return Err(ClipTextError::InvalidConfiguration(
                "CLIP text layer count does not match the configuration",
            ));
        }
        let stream = weights.token_embedding.descriptor().stream();
        for tensor in all_weight_tensors(&weights, text_projection_weight.as_ref())? {
            require_parameter_target(tensor, stream)?;
        }

        let mut token_embedding = NativeModule::embedding(
            "text_model.embeddings.token_embedding",
            configuration.vocabulary_size,
            configuration.hidden_size,
            EmbeddingOptions::default(),
            false,
        )?;
        token_embedding.load_dense_parameters(weights.token_embedding, None)?;
        let mut position_embedding = NativeModule::embedding(
            "text_model.embeddings.position_embedding",
            configuration.max_position_embeddings,
            configuration.hidden_size,
            EmbeddingOptions::default(),
            false,
        )?;
        position_embedding.load_dense_parameters(weights.position_embedding, None)?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(configuration.layer_count)
            .map_err(|_| ClipTextError::Allocation("CLIP text layers"))?;
        for (index, layer_weights) in weights.layers.into_iter().enumerate() {
            layers.push(build_layer(index, &configuration, layer_weights)?);
        }
        let final_layer_norm = layer_norm_from_weights(
            "text_model.final_layer_norm",
            configuration.hidden_size,
            weights.final_layer_norm_weight,
            weights.final_layer_norm_bias,
        )?;
        let text_projection = match (configuration.projection_dimension, text_projection_weight) {
            (None, None) => None,
            (Some(output), Some(weight)) => {
                let mut module = NativeModule::linear(
                    "text_projection",
                    configuration.hidden_size,
                    output,
                    false,
                    false,
                )?;
                module.load_dense_parameters(weight, None)?;
                Some(module)
            }
            _ => {
                return Err(ClipTextError::InvalidConfiguration(
                    "CLIP text projection configuration and weight must agree",
                ));
            }
        };
        Ok(Self {
            configuration,
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm,
            text_projection,
            stream,
        })
    }

    pub fn configuration(&self) -> &ClipTextConfiguration {
        &self.configuration
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipTextError> {
        cancellation.check().map_err(TensorError::from)?;
        if !mapped.unexpected_keys().is_empty() {
            return Err(ClipTextError::InvalidConfiguration(
                "CLIP text mapped weights contain unexpected parameters",
            ));
        }
        let mut consumed = BTreeSet::new();
        let tensor =
            |name: &str, consumed: &mut BTreeSet<String>| -> Result<Tensor, ClipTextError> {
                let tensor = mapped.tensors().get(name).cloned().ok_or(
                    ClipTextError::InvalidConfiguration("CLIP text mapped weight is missing"),
                )?;
                consumed.insert(name.to_owned());
                Ok(tensor)
            };
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.configuration.layer_count)
            .map_err(|_| ClipTextError::Allocation("CLIP text reconstructed layers"))?;
        for index in 0..self.configuration.layer_count {
            cancellation.check().map_err(TensorError::from)?;
            let prefix = format!("layers.{index}");
            layers.push(ClipTextLayerWeights {
                layer_norm_1_weight: tensor(
                    &format!("{prefix}.layer_norm_1.weight"),
                    &mut consumed,
                )?,
                layer_norm_1_bias: tensor(&format!("{prefix}.layer_norm_1.bias"), &mut consumed)?,
                query_weight: tensor(&format!("{prefix}.query.weight"), &mut consumed)?,
                query_bias: tensor(&format!("{prefix}.query.bias"), &mut consumed)?,
                key_weight: tensor(&format!("{prefix}.key.weight"), &mut consumed)?,
                key_bias: tensor(&format!("{prefix}.key.bias"), &mut consumed)?,
                value_weight: tensor(&format!("{prefix}.value.weight"), &mut consumed)?,
                value_bias: tensor(&format!("{prefix}.value.bias"), &mut consumed)?,
                output_weight: tensor(&format!("{prefix}.output.weight"), &mut consumed)?,
                output_bias: tensor(&format!("{prefix}.output.bias"), &mut consumed)?,
                layer_norm_2_weight: tensor(
                    &format!("{prefix}.layer_norm_2.weight"),
                    &mut consumed,
                )?,
                layer_norm_2_bias: tensor(&format!("{prefix}.layer_norm_2.bias"), &mut consumed)?,
                feed_forward_1_weight: tensor(
                    &format!("{prefix}.feed_forward_1.weight"),
                    &mut consumed,
                )?,
                feed_forward_1_bias: tensor(
                    &format!("{prefix}.feed_forward_1.bias"),
                    &mut consumed,
                )?,
                feed_forward_2_weight: tensor(
                    &format!("{prefix}.feed_forward_2.weight"),
                    &mut consumed,
                )?,
                feed_forward_2_bias: tensor(
                    &format!("{prefix}.feed_forward_2.bias"),
                    &mut consumed,
                )?,
            });
        }
        let weights = ClipTextWeights {
            token_embedding: tensor("token_embedding.weight", &mut consumed)?,
            position_embedding: tensor("position_embedding.weight", &mut consumed)?,
            layers,
            final_layer_norm_weight: tensor("final_layer_norm.weight", &mut consumed)?,
            final_layer_norm_bias: tensor("final_layer_norm.bias", &mut consumed)?,
        };
        let projection = if self.configuration.projection_dimension.is_some() {
            Some(tensor("text_projection.weight", &mut consumed)?)
        } else {
            None
        };
        if consumed.len() != mapped.tensors().len() {
            return Err(ClipTextError::InvalidConfiguration(
                "CLIP text mapped weights contain unconsumed parameters",
            ));
        }
        let reconstructed = Self::new(self.configuration.clone(), weights, projection)?;
        reconstructed.semantic_state_digest(cancellation)?;
        cancellation.check().map_err(TensorError::from)?;
        Ok(reconstructed)
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, ClipTextError> {
        cancellation.check().map_err(TensorError::from)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-clip-text.v2\0f32\0cpu");
        for value in [
            self.configuration.vocabulary_size,
            self.configuration.max_position_embeddings,
            self.configuration.hidden_size,
            self.configuration.intermediate_size,
            self.configuration.attention_heads,
            self.configuration.layer_count,
        ] {
            hasher.update(
                u64::try_from(value)
                    .map_err(|_| ClipTextError::Overflow("CLIP text digest"))?
                    .to_be_bytes(),
            );
        }
        hasher.update(self.configuration.eos_token_id.to_be_bytes());
        hasher.update([match self.configuration.activation {
            ClipTextActivation::QuickGelu => 0,
            ClipTextActivation::Gelu => 1,
            ClipTextActivation::GeluTanh => 2,
        }]);
        match self.configuration.projection_dimension {
            Some(value) => {
                hasher.update([1]);
                hasher.update(
                    u64::try_from(value)
                        .map_err(|_| ClipTextError::Overflow("CLIP projection digest"))?
                        .to_be_bytes(),
                );
            }
            None => hasher.update([0]),
        }
        for (name, module) in self.named_modules() {
            cancellation.check().map_err(TensorError::from)?;
            hash_bytes(&mut hasher, name.as_bytes())?;
            hash_bytes(
                &mut hasher,
                module.semantic_state_digest(cancellation)?.as_bytes(),
            )?;
        }
        cancellation.check().map_err(TensorError::from)?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_bytes(&self) -> Result<u64, ClipTextError> {
        self.resident_tensor_allocations().into_iter().try_fold(
            self.resident_owned_bytes()?,
            |bytes, (_, allocation)| {
                bytes
                    .checked_add(allocation)
                    .ok_or(ClipTextError::Overflow("CLIP text residency"))
            },
        )
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, ClipTextError> {
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| ClipTextError::Overflow("CLIP text residency"))?;
        bytes = bytes
            .checked_add(
                u64::try_from(
                    self.layers
                        .capacity()
                        .checked_mul(mem::size_of::<ClipTextLayer>())
                        .ok_or(ClipTextError::Overflow("CLIP text layers"))?,
                )
                .map_err(|_| ClipTextError::Overflow("CLIP text layers"))?,
            )
            .ok_or(ClipTextError::Overflow("CLIP text residency"))?;
        for (_, module) in self.named_modules() {
            let tensor_bytes = module.resident_tensor_allocations().into_iter().try_fold(
                0_u64,
                |total, (_, allocation)| {
                    total
                        .checked_add(allocation)
                        .ok_or(ClipTextError::Overflow("CLIP text tensor residency"))
                },
            )?;
            bytes = bytes
                .checked_add(
                    module
                        .resident_storage_bytes()?
                        .checked_sub(tensor_bytes)
                        .ok_or(ClipTextError::Overflow("CLIP text module residency"))?,
                )
                .ok_or(ClipTextError::Overflow("CLIP text module residency"))?;
        }
        Ok(bytes)
    }

    pub fn resident_tensor_allocations(&self) -> Vec<(StorageId, u64)> {
        let mut allocations = Vec::new();
        for (_, module) in self.named_modules() {
            for allocation in module.resident_tensor_allocations() {
                if !allocations
                    .iter()
                    .any(|(existing, _)| *existing == allocation.0)
                {
                    allocations.push(allocation);
                }
            }
        }
        allocations
    }

    fn named_modules(&self) -> Vec<(String, &NativeModule)> {
        let mut modules = vec![
            ("token_embedding".to_owned(), &self.token_embedding),
            ("position_embedding".to_owned(), &self.position_embedding),
            ("final_layer_norm".to_owned(), &self.final_layer_norm),
        ];
        if let Some(module) = &self.text_projection {
            modules.push(("text_projection".to_owned(), module));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            for (suffix, module) in [
                ("layer_norm_1", &layer.layer_norm_1),
                ("query", &layer.query),
                ("key", &layer.key),
                ("value", &layer.value),
                ("output", &layer.output),
                ("layer_norm_2", &layer.layer_norm_2),
                ("feed_forward_1", &layer.feed_forward_1),
                ("feed_forward_2", &layer.feed_forward_2),
            ] {
                modules.push((format!("{prefix}.{suffix}"), module));
            }
        }
        modules
    }

    pub fn admit_execution_target(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(), ClipTextError> {
        self.execution_requirements().admit_backend_target(
            backend,
            self.configuration.device,
            self.configuration.dtype,
            Layout::Contiguous,
            self.stream,
            context,
        )?;
        Ok(())
    }

    pub(crate) fn execution_requirements(&self) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        for module in [
            Some(&self.token_embedding),
            Some(&self.position_embedding),
            Some(&self.final_layer_norm),
            self.text_projection.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            requirements.extend(module.execution_requirements(DType::F32).iter());
        }
        for layer in &self.layers {
            for module in [
                &layer.layer_norm_1,
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.output,
                &layer.layer_norm_2,
                &layer.feed_forward_1,
                &layer.feed_forward_2,
            ] {
                requirements.extend(module.execution_requirements(DType::F32).iter());
            }
            if let Some(activation) = &layer.activation {
                requirements.extend(activation.execution_requirements(DType::F32).iter());
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
        requirements.extend([
            OperationSupport::allocation(DType::F32, Layout::Contiguous),
            OperationSupport::copy_input(DType::F32, Layout::Contiguous),
            OperationSupport::copy_output(DType::F32, Layout::Contiguous),
            OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous),
            OperationSupport::binary_output(BinaryOperation::Add, DType::F32, Layout::Contiguous),
            OperationSupport::linear_algebra_input(
                LinearAlgebraOperation::BatchMatrixMultiply,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::linear_algebra_output(
                LinearAlgebraOperation::BatchMatrixMultiply,
                DType::F32,
                Layout::Contiguous,
            ),
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
            OperationSupport::reduction_input(
                ReductionOperation::Sum,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::reduction_output(
                ReductionOperation::Sum,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::binary_input(BinaryOperation::Divide, DType::F32, Layout::Contiguous),
            OperationSupport::binary_output(
                BinaryOperation::Divide,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::record_event(),
            OperationSupport::wait_event(),
        ]);
        requirements
    }

    pub fn embed_tokens(
        &self,
        backend: &CpuBackend,
        tokens: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, ClipTextError> {
        self.admit_execution_target(backend, context)?;
        let request = ClipTextRequest {
            input: ClipTextInput::Tokens(tokens),
            attention_mask: None,
            num_tokens: None,
            intermediate: ClipTextIntermediate::None,
            final_layer_norm_intermediate: false,
            project_pooled: false,
            zero_out_masked: false,
        };
        validate_input(backend, &self.configuration, &request, context)?;
        let mut token_embedding = self.token_embedding.clone();
        Ok(token_embedding.forward_with_context(backend, tokens, context)?)
    }

    pub fn forward(
        &self,
        backend: &CpuBackend,
        request: ClipTextRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipTextOutput, ClipTextError> {
        self.admit_execution_target(backend, context)?;
        let (batch, tokens, token_values) =
            validate_input(backend, &self.configuration, &request, context)?;
        let hidden_shape = [
            usize_to_u64(batch, "batch")?,
            usize_to_u64(tokens, "token count")?,
            usize_to_u64(self.configuration.hidden_size, "hidden size")?,
        ];
        let mut token_embedding = self.token_embedding.clone();
        let mut position_embedding = self.position_embedding.clone();
        let mut hidden = match request.input {
            ClipTextInput::Tokens(input) => {
                let token_embeds = token_embedding.forward_with_context(backend, input, context)?;
                let positions = position_indices(backend, batch, tokens, context)?;
                let position_embeds =
                    position_embedding.forward_with_context(backend, &positions, context)?;
                add(backend, &token_embeds, &position_embeds, context)?
            }
            ClipTextInput::Embeddings(embeddings) => {
                let positions = position_indices(backend, batch, tokens, context)?;
                let position_embeds =
                    position_embedding.forward_with_context(backend, &positions, context)?;
                add(backend, embeddings, &position_embeds, context)?
            }
        };
        if hidden.descriptor().shape() != hidden_shape {
            return Err(ClipTextError::InvalidInput(
                "CLIP embeddings produced an unexpected shape",
            ));
        }

        let attention_mask =
            build_attention_mask(backend, request.attention_mask, batch, tokens, context)?;
        let capture_indices = resolve_capture_indices(&request.intermediate, self.layers.len())?;
        let mut captures = Vec::new();
        captures
            .try_reserve_exact(capture_indices.len())
            .map_err(|_| ClipTextError::Allocation("intermediate captures"))?;
        for _ in &capture_indices {
            captures.push(None);
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                backend,
                &hidden,
                attention_mask.mask(),
                batch,
                tokens,
                self.configuration.attention_heads,
                self.configuration.hidden_size,
                context,
            )?;
            for (capture_index, requested) in capture_indices.iter().enumerate() {
                if *requested == layer_index {
                    let destination = captures
                        .get_mut(capture_index)
                        .ok_or(ClipTextError::Overflow("intermediate capture"))?;
                    *destination = Some(hidden.clone());
                }
            }
        }

        let mut final_layer_norm = self.final_layer_norm.clone();
        let final_hidden = final_layer_norm.forward_with_context(backend, &hidden, context)?;
        let mut intermediate =
            assemble_intermediate(backend, &request.intermediate, captures, context)?;
        if request.final_layer_norm_intermediate
            && let Some(captured) = intermediate.take()
        {
            let mut normalization = self.final_layer_norm.clone();
            intermediate = Some(normalization.forward_with_context(backend, &captured, context)?);
        }

        let pooled = pool_final_hidden(
            backend,
            &final_hidden,
            token_values.as_deref(),
            request.num_tokens,
            batch,
            tokens,
            self.configuration.hidden_size,
            self.configuration.eos_token_id,
            context,
        )?;
        let projected_pooled = if request.project_pooled {
            let mut projection = self
                .text_projection
                .clone()
                .ok_or(ClipTextError::MissingProjection)?;
            Some(projection.forward_with_context(backend, &pooled, context)?)
        } else {
            None
        };

        let (last_hidden_state, intermediate) = if request.zero_out_masked {
            let mask = request.attention_mask.ok_or(ClipTextError::InvalidInput(
                "zeroing masked tokens requires an attention mask",
            ))?;
            (
                zero_masked(backend, &final_hidden, mask, batch, tokens, context)?,
                intermediate
                    .map(|value| zero_masked(backend, &value, mask, batch, tokens, context))
                    .transpose()?,
            )
        } else {
            (final_hidden, intermediate)
        };
        Ok(ClipTextOutput {
            last_hidden_state,
            intermediate,
            projected_pooled,
            pooled,
            attention_mask: request.attention_mask.cloned(),
        })
    }
}

impl ClipTextLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        mask: AttentionMask<'_>,
        batch: usize,
        tokens: usize,
        heads: usize,
        hidden_size: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, ClipTextError> {
        context.check()?;
        let mut layer_norm_1 = self.layer_norm_1.clone();
        let normalized = layer_norm_1.forward_with_context(backend, input, context)?;
        let mut query = self.query.clone();
        let mut key = self.key.clone();
        let mut value = self.value.clone();
        let query = query.forward_with_context(backend, &normalized, context)?;
        let key = key.forward_with_context(backend, &normalized, context)?;
        let value = value.forward_with_context(backend, &normalized, context)?;
        let query_values = tensor_to_f32(backend, &query, context)?;
        let key_values = tensor_to_f32(backend, &key, context)?;
        let value_values = tensor_to_f32(backend, &value, context)?;
        let workspace_limit_bytes = tokens
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(ClipTextError::Overflow("attention workspace"))?;
        let attention = scaled_dot_product_attention_with_context(
            backend,
            AttentionRequest {
                backend: AttentionBackend::PytorchSdp,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch,
                query_tokens: tokens,
                key_tokens: tokens,
                heads,
                head_dimension: hidden_size / heads,
                value_dimension: hidden_size / heads,
                scale: None,
                workspace_limit_bytes,
            },
            &query_values,
            &key_values,
            &value_values,
            Some(mask),
            context,
        )?;
        let attention = tensor_from_f32(
            backend,
            input.descriptor().shape(),
            &attention.values,
            context,
        )?;
        let mut output = self.output.clone();
        let attention = output.forward_with_context(backend, &attention, context)?;
        let hidden = add(backend, input, &attention, context)?;

        let mut layer_norm_2 = self.layer_norm_2.clone();
        let normalized = layer_norm_2.forward_with_context(backend, &hidden, context)?;
        let mut feed_forward_1 = self.feed_forward_1.clone();
        let feed_forward = feed_forward_1.forward_with_context(backend, &normalized, context)?;
        let feed_forward = if self.quick_gelu {
            quick_gelu(backend, &feed_forward, context)?
        } else {
            let mut activation =
                self.activation
                    .clone()
                    .ok_or(ClipTextError::InvalidConfiguration(
                        "CLIP text activation module is missing",
                    ))?;
            activation.forward_with_context(backend, &feed_forward, context)?
        };
        let mut feed_forward_2 = self.feed_forward_2.clone();
        let feed_forward = feed_forward_2.forward_with_context(backend, &feed_forward, context)?;
        Ok(add(backend, &hidden, &feed_forward, context)?)
    }
}

#[derive(Debug, Error)]
pub enum ClipTextError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    NativeDiffusion(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    ShapeLayout(#[from] ShapeLayoutTransformPartTwoError),
    #[error("CLIP text configuration is invalid: {0}")]
    InvalidConfiguration(&'static str),
    #[error("CLIP text input is invalid: {0}")]
    InvalidInput(&'static str),
    #[error("CLIP text execution target {dtype:?} on {device:?} is unsupported")]
    UnsupportedTarget { dtype: DType, device: DeviceId },
    #[error("CLIP text token ID {0} is outside the configured vocabulary")]
    TokenOutOfRange(i64),
    #[error("CLIP text intermediate layer {requested} is outside {available} layers")]
    IntermediateOutOfRange { requested: isize, available: usize },
    #[error("CLIP text intermediate layer list contains a duplicate index")]
    DuplicateIntermediate,
    #[error("CLIP text projected pooling was requested without a projection")]
    MissingProjection,
    #[error("CLIP text arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("CLIP text allocation failed for {0}")]
    Allocation(&'static str),
}

#[derive(Debug)]
struct PreparedAttentionMask {
    values: CpuWorkspaceVec<f32>,
    shape: AttentionMaskShape,
}

impl PreparedAttentionMask {
    fn mask(&self) -> AttentionMask<'_> {
        AttentionMask::Additive {
            values: &self.values,
            shape: self.shape,
        }
    }
}

fn build_layer(
    index: usize,
    configuration: &ClipTextConfiguration,
    weights: ClipTextLayerWeights,
) -> Result<ClipTextLayer, ClipTextError> {
    let prefix = format!("text_model.encoder.layers.{index}");
    let layer_norm_1 = layer_norm_from_weights(
        format!("{prefix}.layer_norm1"),
        configuration.hidden_size,
        weights.layer_norm_1_weight,
        weights.layer_norm_1_bias,
    )?;
    let query = linear_from_weights(
        format!("{prefix}.self_attn.q_proj"),
        configuration.hidden_size,
        configuration.hidden_size,
        weights.query_weight,
        Some(weights.query_bias),
    )?;
    let key = linear_from_weights(
        format!("{prefix}.self_attn.k_proj"),
        configuration.hidden_size,
        configuration.hidden_size,
        weights.key_weight,
        Some(weights.key_bias),
    )?;
    let value = linear_from_weights(
        format!("{prefix}.self_attn.v_proj"),
        configuration.hidden_size,
        configuration.hidden_size,
        weights.value_weight,
        Some(weights.value_bias),
    )?;
    let output = linear_from_weights(
        format!("{prefix}.self_attn.out_proj"),
        configuration.hidden_size,
        configuration.hidden_size,
        weights.output_weight,
        Some(weights.output_bias),
    )?;
    let layer_norm_2 = layer_norm_from_weights(
        format!("{prefix}.layer_norm2"),
        configuration.hidden_size,
        weights.layer_norm_2_weight,
        weights.layer_norm_2_bias,
    )?;
    let feed_forward_1 = linear_from_weights(
        format!("{prefix}.mlp.fc1"),
        configuration.hidden_size,
        configuration.intermediate_size,
        weights.feed_forward_1_weight,
        Some(weights.feed_forward_1_bias),
    )?;
    let feed_forward_2 = linear_from_weights(
        format!("{prefix}.mlp.fc2"),
        configuration.intermediate_size,
        configuration.hidden_size,
        weights.feed_forward_2_weight,
        Some(weights.feed_forward_2_bias),
    )?;
    let (activation, quick_gelu) = match configuration.activation {
        ClipTextActivation::QuickGelu => (None, true),
        ClipTextActivation::Gelu => (
            Some(NativeModule::gelu(
                format!("{prefix}.mlp.gelu"),
                GeluApproximation::None,
            )?),
            false,
        ),
        ClipTextActivation::GeluTanh => (
            Some(NativeModule::gelu(
                format!("{prefix}.mlp.gelu_tanh"),
                GeluApproximation::Tanh,
            )?),
            false,
        ),
    };
    Ok(ClipTextLayer {
        layer_norm_1,
        query,
        key,
        value,
        output,
        layer_norm_2,
        feed_forward_1,
        activation,
        quick_gelu,
        feed_forward_2,
    })
}

fn layer_norm_from_weights(
    name: impl Into<String>,
    hidden_size: usize,
    weight: Tensor,
    bias: Tensor,
) -> Result<NativeModule, ClipTextError> {
    let mut module = NativeModule::layer_norm(name, vec![hidden_size], 1.0e-5, true, true, false)?;
    module.load_dense_parameters(weight, Some(bias))?;
    Ok(module)
}

fn linear_from_weights(
    name: impl Into<String>,
    input: usize,
    output: usize,
    weight: Tensor,
    bias: Option<Tensor>,
) -> Result<NativeModule, ClipTextError> {
    let mut module = NativeModule::linear(name, input, output, bias.is_some(), false)?;
    module.load_dense_parameters(weight, bias)?;
    Ok(module)
}

fn all_weight_tensors<'a>(
    weights: &'a ClipTextWeights,
    projection: Option<&'a Tensor>,
) -> Result<Vec<&'a Tensor>, ClipTextError> {
    let capacity = weights
        .layers
        .len()
        .checked_mul(16)
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(usize::from(projection.is_some())))
        .ok_or(ClipTextError::Overflow("CLIP text parameter count"))?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(capacity)
        .map_err(|_| ClipTextError::Allocation("CLIP text parameter references"))?;
    tensors.extend([
        &weights.token_embedding,
        &weights.position_embedding,
        &weights.final_layer_norm_weight,
        &weights.final_layer_norm_bias,
    ]);
    for layer in &weights.layers {
        tensors.extend([
            &layer.layer_norm_1_weight,
            &layer.layer_norm_1_bias,
            &layer.query_weight,
            &layer.query_bias,
            &layer.key_weight,
            &layer.key_bias,
            &layer.value_weight,
            &layer.value_bias,
            &layer.output_weight,
            &layer.output_bias,
            &layer.layer_norm_2_weight,
            &layer.layer_norm_2_bias,
            &layer.feed_forward_1_weight,
            &layer.feed_forward_1_bias,
            &layer.feed_forward_2_weight,
            &layer.feed_forward_2_bias,
        ]);
    }
    tensors.extend(projection);
    Ok(tensors)
}

fn require_parameter_target(tensor: &Tensor, stream: StreamId) -> Result<(), ClipTextError> {
    if tensor.descriptor().dtype() != DType::F32
        || tensor.descriptor().device() != DeviceId::CPU
        || tensor.descriptor().stream() != stream
        || !tensor.descriptor().is_contiguous()?
    {
        return Err(ClipTextError::InvalidConfiguration(
            "CLIP text parameters must be contiguous CPU F32 tensors on one stream",
        ));
    }
    Ok(())
}

fn validate_input(
    backend: &CpuBackend,
    configuration: &ClipTextConfiguration,
    request: &ClipTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<(usize, usize, Option<CpuWorkspaceVec<i64>>), ClipTextError> {
    context.check()?;
    match request.input {
        ClipTextInput::Tokens(tokens) => {
            require_input_target(tokens, DType::I64, context)?;
            let [batch, token_count] = tokens.descriptor().shape() else {
                return Err(ClipTextError::InvalidInput(
                    "token input must have [batch, tokens] shape",
                ));
            };
            let batch = u64_to_usize(*batch, "batch")?;
            let token_count = u64_to_usize(*token_count, "token count")?;
            validate_batch_and_tokens(batch, token_count, configuration)?;
            let values = read_i64(backend, tokens, context)?;
            for value in values.iter() {
                let index =
                    usize::try_from(*value).map_err(|_| ClipTextError::TokenOutOfRange(*value))?;
                if index >= configuration.vocabulary_size {
                    return Err(ClipTextError::TokenOutOfRange(*value));
                }
            }
            Ok((batch, token_count, Some(values)))
        }
        ClipTextInput::Embeddings(embeddings) => {
            require_input_target(embeddings, DType::F32, context)?;
            let [batch, token_count, hidden] = embeddings.descriptor().shape() else {
                return Err(ClipTextError::InvalidInput(
                    "embedding input must have [batch, tokens, hidden] shape",
                ));
            };
            let batch = u64_to_usize(*batch, "batch")?;
            let token_count = u64_to_usize(*token_count, "token count")?;
            validate_batch_and_tokens(batch, token_count, configuration)?;
            if u64_to_usize(*hidden, "hidden size")? != configuration.hidden_size {
                return Err(ClipTextError::InvalidInput(
                    "embedding input hidden size does not match the model",
                ));
            }
            if request.num_tokens.is_none() {
                return Err(ClipTextError::InvalidInput(
                    "embedding input requires explicit token counts for pooling",
                ));
            }
            Ok((batch, token_count, None))
        }
    }
}

fn validate_batch_and_tokens(
    batch: usize,
    tokens: usize,
    configuration: &ClipTextConfiguration,
) -> Result<(), ClipTextError> {
    if batch == 0 || tokens == 0 || tokens > configuration.max_position_embeddings {
        return Err(ClipTextError::InvalidInput(
            "CLIP text batch and token count must be nonzero and position-bounded",
        ));
    }
    Ok(())
}

fn require_input_target(
    tensor: &Tensor,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<(), ClipTextError> {
    if tensor.descriptor().dtype() != dtype
        || tensor.descriptor().device() != DeviceId::CPU
        || tensor.descriptor().stream() != context.stream
        || !tensor.descriptor().is_contiguous()?
    {
        return Err(ClipTextError::InvalidInput(
            "CLIP text input target, stream, or layout is invalid",
        ));
    }
    Ok(())
}

fn position_indices(
    backend: &CpuBackend,
    batch: usize,
    tokens: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipTextError> {
    let count = batch
        .checked_mul(tokens)
        .ok_or(ClipTextError::Overflow("position indices"))?;
    let byte_count = count
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(ClipTextError::Overflow("position index bytes"))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for _ in 0..batch {
        for token in 0..tokens {
            context.check()?;
            let token =
                i64::try_from(token).map_err(|_| ClipTextError::Overflow("position index"))?;
            for byte in token.to_ne_bytes() {
                bytes.try_push(byte)?;
            }
        }
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![
            usize_to_u64(batch, "batch")?,
            usize_to_u64(tokens, "tokens")?,
        ],
        DType::I64,
        DeviceId::CPU,
        context.stream,
    )?;
    let (tensor, event) = backend.upload_bytes(descriptor, &bytes, context)?;
    backend.wait_event(event, context)?;
    Ok(tensor)
}

fn read_i64(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<i64>, ClipTextError> {
    context.check()?;
    let bytes = tensor.contiguous_bytes()?;
    let width = std::mem::size_of::<i64>();
    if !bytes.len().is_multiple_of(width) {
        return Err(ClipTextError::InvalidInput(
            "I64 tensor bytes are unaligned",
        ));
    }
    let mut values = backend.workspace_vec(context, bytes.len() / width)?;
    for (index, chunk) in bytes.chunks_exact(width).enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        let encoded: [u8; 8] = chunk
            .try_into()
            .map_err(|_| ClipTextError::InvalidInput("I64 tensor bytes are unaligned"))?;
        values.try_push(i64::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

fn read_mask_values(
    backend: &CpuBackend,
    mask: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ClipTextError> {
    match mask.descriptor().dtype() {
        DType::F32 => Ok(tensor_to_f32(backend, mask, context)?),
        DType::I64 => {
            let values = read_i64(backend, mask, context)?;
            let mut output = backend.workspace_vec(context, values.len())?;
            for value in values.iter() {
                output.try_push(*value as f32)?;
            }
            Ok(output)
        }
        _ => Err(ClipTextError::InvalidInput(
            "attention mask must use F32 or I64 values",
        )),
    }
}

fn build_attention_mask(
    backend: &CpuBackend,
    attention_mask: Option<&Tensor>,
    batch: usize,
    tokens: usize,
    context: &ExecutionContext<'_>,
) -> Result<PreparedAttentionMask, ClipTextError> {
    let batch_multiplier = if attention_mask.is_some() { batch } else { 1 };
    let count = batch_multiplier
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(tokens))
        .ok_or(ClipTextError::Overflow("attention mask"))?;
    let source = attention_mask
        .map(|mask| {
            if mask.descriptor().shape()
                != [usize_to_u64(batch, "batch")?, usize_to_u64(tokens, "tokens")?]
                || mask.descriptor().device() != DeviceId::CPU
                || mask.descriptor().stream() != context.stream
                || !mask.descriptor().is_contiguous()?
            {
                return Err(ClipTextError::InvalidInput(
                    "attention mask must have contiguous [batch, tokens] shape on the execution stream",
                ));
            }
            read_mask_values(backend, mask, context)
        })
        .transpose()?;
    if let Some(values) = &source
        && values.iter().any(|value| !matches!(*value, 0.0 | 1.0))
    {
        return Err(ClipTextError::InvalidInput(
            "attention mask values must be exactly zero or one",
        ));
    }
    let mut values = backend.workspace_vec(context, count)?;
    for batch_index in 0..batch_multiplier {
        for query in 0..tokens {
            for key in 0..tokens {
                context.check()?;
                let source_index = batch_index
                    .checked_mul(tokens)
                    .and_then(|value| value.checked_add(key))
                    .ok_or(ClipTextError::Overflow("attention mask index"))?;
                let padding_masked = source
                    .as_ref()
                    .and_then(|source| source.get(source_index))
                    .is_some_and(|value| *value == 0.0);
                values.try_push(if key > query || padding_masked {
                    -f32::MAX
                } else {
                    0.0
                })?;
            }
        }
    }
    Ok(PreparedAttentionMask {
        values,
        shape: if attention_mask.is_some() {
            AttentionMaskShape::BatchQueryByKey
        } else {
            AttentionMaskShape::QueryByKey
        },
    })
}

fn resolve_capture_indices(
    request: &ClipTextIntermediate,
    layers: usize,
) -> Result<Vec<usize>, ClipTextError> {
    match request {
        ClipTextIntermediate::None => Ok(Vec::new()),
        ClipTextIntermediate::Layer(index) => {
            let mut resolved = Vec::new();
            resolved
                .try_reserve_exact(1)
                .map_err(|_| ClipTextError::Allocation("intermediate layer index"))?;
            resolved.push(resolve_layer(*index, layers)?);
            Ok(resolved)
        }
        ClipTextIntermediate::All => {
            let mut resolved = Vec::new();
            resolved
                .try_reserve_exact(layers)
                .map_err(|_| ClipTextError::Allocation("intermediate layer indices"))?;
            for layer in 0..layers {
                resolved.push(layer);
            }
            Ok(resolved)
        }
        ClipTextIntermediate::Layers(indices) => {
            if indices.is_empty() {
                return Err(ClipTextError::InvalidInput(
                    "intermediate layer list must not be empty",
                ));
            }
            let mut resolved = Vec::new();
            resolved
                .try_reserve_exact(indices.len())
                .map_err(|_| ClipTextError::Allocation("intermediate layer indices"))?;
            for index in indices {
                let index = resolve_layer(*index, layers)?;
                if resolved.contains(&index) {
                    return Err(ClipTextError::DuplicateIntermediate);
                }
                resolved.push(index);
            }
            Ok(resolved)
        }
    }
}

fn resolve_layer(requested: isize, layers: usize) -> Result<usize, ClipTextError> {
    let layers_isize =
        isize::try_from(layers).map_err(|_| ClipTextError::Overflow("layer count"))?;
    let resolved = if requested < 0 {
        layers_isize
            .checked_add(requested)
            .ok_or(ClipTextError::IntermediateOutOfRange {
                requested,
                available: layers,
            })?
    } else {
        requested
    };
    let resolved =
        usize::try_from(resolved).map_err(|_| ClipTextError::IntermediateOutOfRange {
            requested,
            available: layers,
        })?;
    if resolved >= layers {
        return Err(ClipTextError::IntermediateOutOfRange {
            requested,
            available: layers,
        });
    }
    Ok(resolved)
}

fn assemble_intermediate(
    backend: &CpuBackend,
    request: &ClipTextIntermediate,
    captures: Vec<Option<Tensor>>,
    context: &ExecutionContext<'_>,
) -> Result<Option<Tensor>, ClipTextError> {
    if matches!(request, ClipTextIntermediate::None) {
        return Ok(None);
    }
    let mut resolved_captures = Vec::new();
    resolved_captures
        .try_reserve_exact(captures.len())
        .map_err(|_| ClipTextError::Allocation("intermediate tensors"))?;
    for capture in captures {
        resolved_captures.push(capture.ok_or(ClipTextError::InvalidConfiguration(
            "requested intermediate layer was not captured",
        ))?);
    }
    let captures = resolved_captures;
    if matches!(request, ClipTextIntermediate::Layer(_)) {
        return captures
            .into_iter()
            .next()
            .map(Some)
            .ok_or(ClipTextError::InvalidConfiguration(
                "single intermediate layer was not captured",
            ));
    }
    Ok(Some(torch_stack_with_context_exact_native(
        backend, &captures, 1, context,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn pool_final_hidden(
    backend: &CpuBackend,
    hidden: &Tensor,
    token_values: Option<&[i64]>,
    num_tokens: Option<&[usize]>,
    batch: usize,
    tokens: usize,
    hidden_size: usize,
    eos_token_id: u32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipTextError> {
    if let Some(counts) = num_tokens
        && (counts.len() != batch || counts.iter().any(|count| *count == 0 || *count > tokens))
    {
        return Err(ClipTextError::InvalidInput(
            "explicit token counts must contain one in-range value per batch row",
        ));
    }
    let values = tensor_to_f32(backend, hidden, context)?;
    let output_count = batch
        .checked_mul(hidden_size)
        .ok_or(ClipTextError::Overflow("pooled output"))?;
    let mut output = backend.workspace_vec(context, output_count)?;
    for batch_index in 0..batch {
        context.check()?;
        let token_index = if let Some(counts) = num_tokens {
            counts
                .get(batch_index)
                .copied()
                .ok_or(ClipTextError::InvalidInput(
                    "explicit token counts are incomplete",
                ))?
                .checked_sub(1)
                .ok_or(ClipTextError::InvalidInput(
                    "explicit token counts must be nonzero",
                ))?
        } else {
            let tokens_values = token_values.ok_or(ClipTextError::InvalidInput(
                "token IDs or explicit token counts are required for pooling",
            ))?;
            let row_start = batch_index
                .checked_mul(tokens)
                .ok_or(ClipTextError::Overflow("token row"))?;
            let row_end = row_start
                .checked_add(tokens)
                .ok_or(ClipTextError::Overflow("token row"))?;
            tokens_values
                .get(row_start..row_end)
                .and_then(|row| {
                    row.iter()
                        .position(|token| *token == i64::from(eos_token_id))
                })
                .unwrap_or(0)
        };
        let start = batch_index
            .checked_mul(tokens)
            .and_then(|value| value.checked_add(token_index))
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or(ClipTextError::Overflow("pooled offset"))?;
        let end = start
            .checked_add(hidden_size)
            .ok_or(ClipTextError::Overflow("pooled range"))?;
        for value in values.get(start..end).ok_or(ClipTextError::InvalidInput(
            "pooled range is outside hidden state",
        ))? {
            output.try_push(*value)?;
        }
    }
    tensor_from_f32(
        backend,
        &[
            usize_to_u64(batch, "batch")?,
            usize_to_u64(hidden_size, "hidden")?,
        ],
        &output,
        context,
    )
    .map_err(ClipTextError::from)
}

fn zero_masked(
    backend: &CpuBackend,
    input: &Tensor,
    attention_mask: &Tensor,
    batch: usize,
    tokens: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipTextError> {
    let shape = input.descriptor().shape();
    let hidden = *shape.last().ok_or(ClipTextError::InvalidInput(
        "masked hidden state has no channel axis",
    ))?;
    let hidden = u64_to_usize(hidden, "hidden size")?;
    let layer_multiplier = match shape {
        [actual_batch, actual_tokens, _]
            if *actual_batch == usize_to_u64(batch, "batch")?
                && *actual_tokens == usize_to_u64(tokens, "tokens")? =>
        {
            1
        }
        [actual_batch, layers, actual_tokens, _]
            if *actual_batch == usize_to_u64(batch, "batch")?
                && *actual_tokens == usize_to_u64(tokens, "tokens")? =>
        {
            u64_to_usize(*layers, "captured layers")?
        }
        _ => {
            return Err(ClipTextError::InvalidInput(
                "masked hidden state has an unexpected shape",
            ));
        }
    };
    let mask = read_mask_values(backend, attention_mask, context)?;
    let values = tensor_to_f32(backend, input, context)?;
    let mut output = backend.workspace_vec(context, values.len())?;
    for batch_index in 0..batch {
        for layer in 0..layer_multiplier {
            for token in 0..tokens {
                let mask_offset = batch_index
                    .checked_mul(tokens)
                    .and_then(|value| value.checked_add(token))
                    .ok_or(ClipTextError::Overflow("attention-mask offset"))?;
                let enabled = *mask
                    .get(mask_offset)
                    .ok_or(ClipTextError::InvalidInput("attention mask is incomplete"))?;
                for channel in 0..hidden {
                    let offset = batch_index
                        .checked_mul(layer_multiplier)
                        .and_then(|value| value.checked_add(layer))
                        .and_then(|value| value.checked_mul(tokens))
                        .and_then(|value| value.checked_add(token))
                        .and_then(|value| value.checked_mul(hidden))
                        .and_then(|value| value.checked_add(channel))
                        .ok_or(ClipTextError::Overflow("masked hidden-state offset"))?;
                    output.try_push(
                        values
                            .get(offset)
                            .copied()
                            .ok_or(ClipTextError::InvalidInput("hidden state is incomplete"))?
                            * enabled,
                    )?;
                }
            }
        }
    }
    Ok(tensor_from_f32(backend, shape, &output, context)?)
}

fn usize_to_u64(value: usize, name: &'static str) -> Result<u64, ClipTextError> {
    u64::try_from(value).map_err(|_| ClipTextError::Overflow(name))
}

fn u64_to_usize(value: u64, name: &'static str) -> Result<usize, ClipTextError> {
    usize::try_from(value).map_err(|_| ClipTextError::Overflow(name))
}
