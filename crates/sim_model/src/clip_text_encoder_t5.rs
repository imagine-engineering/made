use crate::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionMask, AttentionMaskShape,
    AttentionRequest, EmbeddingOptions, GeluApproximation, MappedModelWeights,
    NativeExecutionRequirements, NativeModule, NativeOpsError, NativePromptTokenizer,
    NativeTokenizedPrompt, scaled_dot_product_attention_with_context,
};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId,
    ExecutionContext, Layout, LinearAlgebraOperation, OperationSupport, ReductionOperation,
    StorageId, StreamId, Tensor, TensorError, UnaryOperation,
    generated_activation_normalization_functional_01::{
        FunctionalError, rms_norm_with_context_exact_native,
    },
    generated_native_diffusion::{NativeDiffusionTensorError, add, tensor_from_f32, tensor_to_f32},
};
use std::{collections::BTreeSet, mem};
use thiserror::Error;

pub const T5_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/t5.py";
pub const T5_SOURCE_SHA256: &str =
    "797f11b69256dffa8d9a6d236fe1e05d0c250c84d231a5993accde0ff44238bc";
pub const BERT_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/bert.py";
pub const BERT_SOURCE_SHA256: &str =
    "3f1f32353da95790285a10f452959a871aa949aab15a89b646a95abc6165955c";
pub const SPIECE_TOKENIZER_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/sim/text_encoders/spiece_tokenizer.py";
pub const SPIECE_TOKENIZER_SOURCE_SHA256: &str =
    "300de9904c109171b2270524a35c372e6b8c145b88b3f9d242613d45c2bb559f";
pub const T5_BIDIRECTIONAL_CATALOG_SYMBOLS: [&str; 19] = [
    "BertAttention",
    "BertOutput",
    "BertAttentionBlock",
    "BertIntermediate",
    "BertBlock",
    "BertEncoder",
    "BertEmbeddings",
    "BertModel_",
    "BertModel",
    "SPieceTokenizer",
    "T5LayerNorm",
    "T5DenseActDense",
    "T5DenseGatedActDense",
    "T5LayerFF",
    "T5Attention",
    "T5LayerSelfAttention",
    "T5Block",
    "T5Stack",
    "T5",
];

fn hash_bidirectional_bytes(
    hasher: &mut Sha256,
    bytes: &[u8],
) -> Result<(), BidirectionalTextError> {
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| BidirectionalTextError::Overflow("bidirectional digest field"))?
            .to_be_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BidirectionalTextArchitecture {
    T5,
    Bert,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BidirectionalFeedForwardActivation {
    Relu,
    Gelu,
    GeluTanh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BidirectionalPooling {
    None,
    FirstToken,
    MeanUnmasked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BidirectionalTextConfiguration {
    pub architecture: BidirectionalTextArchitecture,
    pub dtype: DType,
    pub device: DeviceId,
    pub vocabulary_size: usize,
    pub maximum_tokens: usize,
    pub type_vocabulary_size: usize,
    pub hidden_size: usize,
    pub attention_inner_size: usize,
    pub feed_forward_size: usize,
    pub attention_heads: usize,
    pub layer_count: usize,
    pub normalization_epsilon_bits: u32,
    pub gated_feed_forward: bool,
    pub activation: BidirectionalFeedForwardActivation,
    pub relative_attention: bool,
    pub relative_attention_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub projection_dimension: Option<usize>,
}

impl BidirectionalTextConfiguration {
    pub fn normalization_epsilon(&self) -> f32 {
        f32::from_bits(self.normalization_epsilon_bits)
    }

    pub fn validate(&self) -> Result<(), BidirectionalTextError> {
        let epsilon = self.normalization_epsilon();
        if self.dtype != DType::F32 || self.device != DeviceId::CPU {
            return Err(BidirectionalTextError::UnsupportedTarget {
                dtype: self.dtype,
                device: self.device,
            });
        }
        if self.vocabulary_size == 0
            || self.maximum_tokens == 0
            || self.hidden_size == 0
            || self.attention_inner_size == 0
            || self.feed_forward_size == 0
            || self.attention_heads == 0
            || self.layer_count == 0
            || !self
                .attention_inner_size
                .is_multiple_of(self.attention_heads)
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || self.projection_dimension == Some(0)
        {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "encoder dimensions, heads, epsilon, or projection are invalid",
            ));
        }
        match self.architecture {
            BidirectionalTextArchitecture::T5 => {
                if self.type_vocabulary_size != 0
                    || self.relative_attention_buckets < 4
                    || !self.relative_attention_buckets.is_multiple_of(2)
                    || self.relative_attention_max_distance <= self.relative_attention_buckets / 4
                {
                    return Err(BidirectionalTextError::InvalidConfiguration(
                        "T5 type vocabulary or relative-attention configuration is invalid",
                    ));
                }
            }
            BidirectionalTextArchitecture::Bert => {
                if self.type_vocabulary_size == 0
                    || self.attention_inner_size != self.hidden_size
                    || !self.hidden_size.is_multiple_of(self.attention_heads)
                    || self.gated_feed_forward
                    || self.relative_attention
                {
                    return Err(BidirectionalTextError::InvalidConfiguration(
                        "BERT type vocabulary or feed-forward configuration is invalid",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct BidirectionalLayerWeights {
    pub attention_norm_weight: Tensor,
    pub attention_norm_bias: Option<Tensor>,
    pub query_weight: Tensor,
    pub query_bias: Option<Tensor>,
    pub key_weight: Tensor,
    pub key_bias: Option<Tensor>,
    pub value_weight: Tensor,
    pub value_bias: Option<Tensor>,
    pub attention_output_weight: Tensor,
    pub attention_output_bias: Option<Tensor>,
    pub feed_forward_norm_weight: Tensor,
    pub feed_forward_norm_bias: Option<Tensor>,
    pub feed_forward_input_weight: Tensor,
    pub feed_forward_input_bias: Option<Tensor>,
    pub feed_forward_gate_weight: Option<Tensor>,
    pub feed_forward_gate_bias: Option<Tensor>,
    pub feed_forward_output_weight: Tensor,
    pub feed_forward_output_bias: Option<Tensor>,
    pub relative_attention_bias: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct BidirectionalTextWeights {
    pub token_embedding: Tensor,
    pub position_embedding: Option<Tensor>,
    pub token_type_embedding: Option<Tensor>,
    pub embedding_norm_weight: Option<Tensor>,
    pub embedding_norm_bias: Option<Tensor>,
    pub layers: Vec<BidirectionalLayerWeights>,
    pub final_norm_weight: Option<Tensor>,
    pub projection_weight: Option<Tensor>,
    pub projection_bias: Option<Tensor>,
}

#[derive(Clone, Copy, Debug)]
pub enum BidirectionalTextInput<'a> {
    Tokens(&'a Tensor),
    Embeddings(&'a Tensor),
}

#[derive(Clone, Debug)]
pub struct BidirectionalTextRequest<'a> {
    pub input: BidirectionalTextInput<'a>,
    pub attention_mask: Option<&'a Tensor>,
    pub token_type_ids: Option<&'a Tensor>,
    pub intermediate_layer: Option<isize>,
    pub final_norm_intermediate: bool,
    pub pooling: BidirectionalPooling,
    pub project_pooled: bool,
}

#[derive(Clone, Debug)]
pub struct BidirectionalTextOutput {
    last_hidden_state: Tensor,
    intermediate: Option<Tensor>,
    pooled: Option<Tensor>,
    projected_pooled: Option<Tensor>,
}

impl BidirectionalTextOutput {
    pub fn last_hidden_state(&self) -> &Tensor {
        &self.last_hidden_state
    }

    pub fn intermediate(&self) -> Option<&Tensor> {
        self.intermediate.as_ref()
    }

    pub fn pooled(&self) -> Option<&Tensor> {
        self.pooled.as_ref()
    }

    pub fn projected_pooled(&self) -> Option<&Tensor> {
        self.projected_pooled.as_ref()
    }
}

#[derive(Clone, Debug)]
struct NativeBidirectionalLayer {
    attention_norm: NativeNorm,
    query: NativeModule,
    key: NativeModule,
    value: NativeModule,
    attention_output: NativeModule,
    feed_forward_norm: NativeNorm,
    feed_forward_input: NativeModule,
    feed_forward_gate: Option<NativeModule>,
    activation: NativeModule,
    feed_forward_output: NativeModule,
    relative_attention_bias: Option<Tensor>,
}

#[derive(Clone, Debug)]
enum NativeNorm {
    Rms { weight: Tensor, epsilon: f32 },
    Layer(NativeModule),
}

#[derive(Clone, Debug)]
pub struct NativeT5TextEncoder {
    configuration: BidirectionalTextConfiguration,
    token_embedding: NativeModule,
    position_embedding: Option<NativeModule>,
    token_type_embedding: Option<NativeModule>,
    embedding_norm: Option<NativeNorm>,
    layers: Vec<NativeBidirectionalLayer>,
    final_norm: Option<NativeNorm>,
    projection: Option<NativeModule>,
    stream: StreamId,
}

#[derive(Debug, Error)]
pub enum BidirectionalTextError {
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorOperation(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    Tokenizer(#[from] crate::NativeTokenizerError),
    #[error("bidirectional encoder target {device:?}/{dtype:?} is unsupported")]
    UnsupportedTarget { dtype: DType, device: DeviceId },
    #[error("bidirectional encoder configuration is invalid: {0}")]
    InvalidConfiguration(&'static str),
    #[error("bidirectional encoder input is invalid: {0}")]
    InvalidInput(&'static str),
    #[error("token {0} is outside the encoder vocabulary")]
    TokenOutOfRange(i64),
    #[error("intermediate layer {requested} is outside {available} layers")]
    IntermediateOutOfRange { requested: isize, available: usize },
    #[error("bidirectional encoder arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("bidirectional encoder allocation failed for {0}")]
    Allocation(&'static str),
    #[error("pooled projection was requested without a configured projection")]
    MissingProjection,
}

impl NativeT5TextEncoder {
    pub fn new(
        configuration: BidirectionalTextConfiguration,
        weights: BidirectionalTextWeights,
    ) -> Result<Self, BidirectionalTextError> {
        configuration.validate()?;
        if weights.layers.len() != configuration.layer_count {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "weight layer count does not match configuration",
            ));
        }
        let stream = weights.token_embedding.descriptor().stream();
        require_parameter(&weights.token_embedding, stream)?;
        let mut token_embedding = NativeModule::embedding(
            "bidirectional.token_embedding",
            configuration.vocabulary_size,
            configuration.hidden_size,
            EmbeddingOptions::default(),
            false,
        )?;
        token_embedding.load_dense_parameters(weights.token_embedding, None)?;

        let position_embedding = match (configuration.architecture, weights.position_embedding) {
            (BidirectionalTextArchitecture::Bert, Some(weight)) => Some(embedding_module(
                "bert.position_embedding",
                configuration.maximum_tokens,
                configuration.hidden_size,
                weight,
                stream,
            )?),
            (BidirectionalTextArchitecture::T5, None) => None,
            _ => {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "position embeddings must exist only for BERT",
                ));
            }
        };
        let token_type_embedding = match (configuration.architecture, weights.token_type_embedding)
        {
            (BidirectionalTextArchitecture::Bert, Some(weight)) => Some(embedding_module(
                "bert.token_type_embedding",
                configuration.type_vocabulary_size,
                configuration.hidden_size,
                weight,
                stream,
            )?),
            (BidirectionalTextArchitecture::T5, None) => None,
            _ => {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "token-type embeddings must exist only for BERT",
                ));
            }
        };
        let embedding_norm = match (
            configuration.architecture,
            weights.embedding_norm_weight,
            weights.embedding_norm_bias,
        ) {
            (BidirectionalTextArchitecture::Bert, Some(weight), Some(bias)) => Some(build_norm(
                "bert.embedding_norm".to_owned(),
                &configuration,
                weight,
                Some(bias),
                stream,
            )?),
            (BidirectionalTextArchitecture::T5, None, None) => None,
            _ => {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "embedding normalization must exist only for BERT",
                ));
            }
        };

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(configuration.layer_count)
            .map_err(|_| BidirectionalTextError::Allocation("encoder layers"))?;
        for (index, layer) in weights.layers.into_iter().enumerate() {
            layers.push(build_layer(index, &configuration, layer, stream)?);
        }

        let final_norm = match (configuration.architecture, weights.final_norm_weight) {
            (BidirectionalTextArchitecture::T5, Some(weight)) => {
                require_vector_parameter(&weight, configuration.hidden_size, stream)?;
                Some(NativeNorm::Rms {
                    weight,
                    epsilon: configuration.normalization_epsilon(),
                })
            }
            (BidirectionalTextArchitecture::Bert, None) => None,
            _ => {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "final normalization must exist only for T5",
                ));
            }
        };
        let projection = match (
            configuration.projection_dimension,
            weights.projection_weight,
        ) {
            (Some(output), Some(weight)) => {
                let mut module = NativeModule::linear(
                    "bidirectional.text_projection",
                    configuration.hidden_size,
                    output,
                    weights.projection_bias.is_some(),
                    false,
                )?;
                require_parameter(&weight, stream)?;
                if let Some(bias) = weights.projection_bias.as_ref() {
                    require_parameter(bias, stream)?;
                }
                module.load_dense_parameters(weight, weights.projection_bias)?;
                Some(module)
            }
            (None, None) if weights.projection_bias.is_none() => None,
            _ => {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "projection configuration and parameters must agree",
                ));
            }
        };
        Ok(Self {
            configuration,
            token_embedding,
            position_embedding,
            token_type_embedding,
            embedding_norm,
            layers,
            final_norm,
            projection,
            stream,
        })
    }

    pub fn configuration(&self) -> &BidirectionalTextConfiguration {
        &self.configuration
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &CancellationToken,
    ) -> Result<Self, BidirectionalTextError> {
        cancellation.check().map_err(TensorError::from)?;
        if !mapped.unexpected_keys().is_empty() {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "bidirectional mapped weights contain unexpected parameters",
            ));
        }
        let mut consumed = BTreeSet::new();
        let required = |name: &str,
                        consumed: &mut BTreeSet<String>|
         -> Result<Tensor, BidirectionalTextError> {
            let tensor = mapped.tensors().get(name).cloned().ok_or(
                BidirectionalTextError::InvalidConfiguration(
                    "bidirectional mapped weight is missing",
                ),
            )?;
            consumed.insert(name.to_owned());
            Ok(tensor)
        };
        let optional = |name: &str,
                        present: bool,
                        consumed: &mut BTreeSet<String>|
         -> Result<Option<Tensor>, BidirectionalTextError> {
            if present {
                required(name, consumed).map(Some)
            } else {
                Ok(None)
            }
        };
        let mut layers = Vec::new();
        layers.try_reserve_exact(self.layers.len()).map_err(|_| {
            BidirectionalTextError::Allocation("reconstructed bidirectional layers")
        })?;
        for (index, layer) in self.layers.iter().enumerate() {
            cancellation.check().map_err(TensorError::from)?;
            let prefix = format!("layers.{index}");
            let attention_layer_norm = matches!(layer.attention_norm, NativeNorm::Layer(_));
            let feed_forward_layer_norm = matches!(layer.feed_forward_norm, NativeNorm::Layer(_));
            let (_, query_bias) = layer.query.dense_parameters()?;
            let (_, key_bias) = layer.key.dense_parameters()?;
            let (_, value_bias) = layer.value.dense_parameters()?;
            let (_, attention_output_bias) = layer.attention_output.dense_parameters()?;
            let (_, feed_forward_input_bias) = layer.feed_forward_input.dense_parameters()?;
            let (_, feed_forward_output_bias) = layer.feed_forward_output.dense_parameters()?;
            let gate_bias = layer
                .feed_forward_gate
                .as_ref()
                .map(NativeModule::dense_parameters)
                .transpose()?
                .and_then(|(_, bias)| bias);
            layers.push(BidirectionalLayerWeights {
                attention_norm_weight: required(
                    &format!("{prefix}.attention_norm.weight"),
                    &mut consumed,
                )?,
                attention_norm_bias: optional(
                    &format!("{prefix}.attention_norm.bias"),
                    attention_layer_norm,
                    &mut consumed,
                )?,
                query_weight: required(&format!("{prefix}.query.weight"), &mut consumed)?,
                query_bias: optional(
                    &format!("{prefix}.query.bias"),
                    query_bias.is_some(),
                    &mut consumed,
                )?,
                key_weight: required(&format!("{prefix}.key.weight"), &mut consumed)?,
                key_bias: optional(
                    &format!("{prefix}.key.bias"),
                    key_bias.is_some(),
                    &mut consumed,
                )?,
                value_weight: required(&format!("{prefix}.value.weight"), &mut consumed)?,
                value_bias: optional(
                    &format!("{prefix}.value.bias"),
                    value_bias.is_some(),
                    &mut consumed,
                )?,
                attention_output_weight: required(
                    &format!("{prefix}.attention_output.weight"),
                    &mut consumed,
                )?,
                attention_output_bias: optional(
                    &format!("{prefix}.attention_output.bias"),
                    attention_output_bias.is_some(),
                    &mut consumed,
                )?,
                feed_forward_norm_weight: required(
                    &format!("{prefix}.feed_forward_norm.weight"),
                    &mut consumed,
                )?,
                feed_forward_norm_bias: optional(
                    &format!("{prefix}.feed_forward_norm.bias"),
                    feed_forward_layer_norm,
                    &mut consumed,
                )?,
                feed_forward_input_weight: required(
                    &format!("{prefix}.feed_forward_input.weight"),
                    &mut consumed,
                )?,
                feed_forward_input_bias: optional(
                    &format!("{prefix}.feed_forward_input.bias"),
                    feed_forward_input_bias.is_some(),
                    &mut consumed,
                )?,
                feed_forward_gate_weight: optional(
                    &format!("{prefix}.feed_forward_gate.weight"),
                    layer.feed_forward_gate.is_some(),
                    &mut consumed,
                )?,
                feed_forward_gate_bias: optional(
                    &format!("{prefix}.feed_forward_gate.bias"),
                    gate_bias.is_some(),
                    &mut consumed,
                )?,
                feed_forward_output_weight: required(
                    &format!("{prefix}.feed_forward_output.weight"),
                    &mut consumed,
                )?,
                feed_forward_output_bias: optional(
                    &format!("{prefix}.feed_forward_output.bias"),
                    feed_forward_output_bias.is_some(),
                    &mut consumed,
                )?,
                relative_attention_bias: optional(
                    &format!("{prefix}.relative_attention_bias"),
                    layer.relative_attention_bias.is_some(),
                    &mut consumed,
                )?,
            });
        }
        let embedding_layer_norm = matches!(self.embedding_norm, Some(NativeNorm::Layer(_)));
        let projection_bias = self
            .projection
            .as_ref()
            .map(NativeModule::dense_parameters)
            .transpose()?
            .and_then(|(_, bias)| bias);
        let weights = BidirectionalTextWeights {
            token_embedding: required("token_embedding.weight", &mut consumed)?,
            position_embedding: optional(
                "position_embedding.weight",
                self.position_embedding.is_some(),
                &mut consumed,
            )?,
            token_type_embedding: optional(
                "token_type_embedding.weight",
                self.token_type_embedding.is_some(),
                &mut consumed,
            )?,
            embedding_norm_weight: optional(
                "embedding_norm.weight",
                self.embedding_norm.is_some(),
                &mut consumed,
            )?,
            embedding_norm_bias: optional(
                "embedding_norm.bias",
                embedding_layer_norm,
                &mut consumed,
            )?,
            layers,
            final_norm_weight: optional(
                "final_norm.weight",
                self.final_norm.is_some(),
                &mut consumed,
            )?,
            projection_weight: optional(
                "projection.weight",
                self.projection.is_some(),
                &mut consumed,
            )?,
            projection_bias: optional("projection.bias", projection_bias.is_some(), &mut consumed)?,
        };
        if consumed.len() != mapped.tensors().len() {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "bidirectional mapped weights contain unconsumed parameters",
            ));
        }
        let reconstructed = Self::new(self.configuration.clone(), weights)?;
        reconstructed.semantic_state_digest(cancellation)?;
        Ok(reconstructed)
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, BidirectionalTextError> {
        cancellation.check().map_err(TensorError::from)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-bidirectional-text.v2\0f32\0cpu");
        hasher.update([match self.configuration.architecture {
            BidirectionalTextArchitecture::T5 => 0,
            BidirectionalTextArchitecture::Bert => 1,
        }]);
        for value in [
            self.configuration.vocabulary_size,
            self.configuration.maximum_tokens,
            self.configuration.type_vocabulary_size,
            self.configuration.hidden_size,
            self.configuration.attention_inner_size,
            self.configuration.feed_forward_size,
            self.configuration.attention_heads,
            self.configuration.layer_count,
            self.configuration.relative_attention_buckets,
            self.configuration.relative_attention_max_distance,
        ] {
            hasher.update(
                u64::try_from(value)
                    .map_err(|_| BidirectionalTextError::Overflow("bidirectional digest"))?
                    .to_be_bytes(),
            );
        }
        hasher.update(self.configuration.normalization_epsilon_bits.to_be_bytes());
        hasher.update([
            u8::from(self.configuration.gated_feed_forward),
            match self.configuration.activation {
                BidirectionalFeedForwardActivation::Relu => 0,
                BidirectionalFeedForwardActivation::Gelu => 1,
                BidirectionalFeedForwardActivation::GeluTanh => 2,
            },
            u8::from(self.configuration.relative_attention),
        ]);
        match self.configuration.projection_dimension {
            Some(value) => {
                hasher.update([1]);
                hasher.update(
                    u64::try_from(value)
                        .map_err(|_| {
                            BidirectionalTextError::Overflow("bidirectional projection digest")
                        })?
                        .to_be_bytes(),
                );
            }
            None => hasher.update([0]),
        }
        for (name, module) in self.named_modules() {
            cancellation.check().map_err(TensorError::from)?;
            hash_bidirectional_bytes(&mut hasher, name.as_bytes())?;
            hash_bidirectional_bytes(
                &mut hasher,
                module.semantic_state_digest(cancellation)?.as_bytes(),
            )?;
        }
        for (name, tensor) in self.normalization_tensors() {
            cancellation.check().map_err(TensorError::from)?;
            hash_bidirectional_bytes(&mut hasher, name.as_bytes())?;
            hash_bidirectional_bytes(&mut hasher, tensor.contiguous_bytes()?)?;
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_bytes(&self) -> Result<u64, BidirectionalTextError> {
        self.resident_tensor_allocations().into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, allocation)| {
                total
                    .checked_add(allocation)
                    .ok_or(BidirectionalTextError::Overflow("bidirectional residency"))
            },
        )
    }
    pub fn resident_owned_bytes(&self) -> Result<u64, BidirectionalTextError> {
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| BidirectionalTextError::Overflow("bidirectional residency"))?;
        bytes = bytes
            .checked_add(
                u64::try_from(
                    self.layers
                        .capacity()
                        .checked_mul(mem::size_of::<NativeBidirectionalLayer>())
                        .ok_or(BidirectionalTextError::Overflow("bidirectional layers"))?,
                )
                .map_err(|_| BidirectionalTextError::Overflow("bidirectional layers"))?,
            )
            .ok_or(BidirectionalTextError::Overflow("bidirectional residency"))?;
        for (_, module) in self.named_modules() {
            let tensor_bytes = module.resident_tensor_allocations().into_iter().try_fold(
                0_u64,
                |total, (_, allocation)| {
                    total
                        .checked_add(allocation)
                        .ok_or(BidirectionalTextError::Overflow(
                            "bidirectional tensor residency",
                        ))
                },
            )?;
            bytes = bytes
                .checked_add(
                    module
                        .resident_storage_bytes()?
                        .checked_sub(tensor_bytes)
                        .ok_or(BidirectionalTextError::Overflow(
                            "bidirectional module residency",
                        ))?,
                )
                .ok_or(BidirectionalTextError::Overflow("bidirectional residency"))?;
        }
        Ok(bytes)
    }
    pub fn resident_tensor_allocations(&self) -> Vec<(StorageId, u64)> {
        let mut allocations = Vec::new();
        let mut insert = |allocation: (StorageId, u64)| {
            if !allocations
                .iter()
                .any(|(existing, _)| *existing == allocation.0)
            {
                allocations.push(allocation);
            }
        };
        for (_, module) in self.named_modules() {
            for allocation in module.resident_tensor_allocations() {
                insert(allocation);
            }
        }
        for (_, tensor) in self.normalization_tensors() {
            insert((tensor.storage_id(), tensor.storage_byte_len()));
        }
        allocations
    }

    fn named_modules(&self) -> Vec<(String, &NativeModule)> {
        let mut modules = vec![("token_embedding".to_owned(), &self.token_embedding)];
        for (name, module) in [
            ("position_embedding", self.position_embedding.as_ref()),
            ("token_type_embedding", self.token_type_embedding.as_ref()),
            ("projection", self.projection.as_ref()),
        ] {
            if let Some(module) = module {
                modules.push((name.to_owned(), module));
            }
        }
        if let Some(NativeNorm::Layer(module)) = &self.embedding_norm {
            modules.push(("embedding_norm".to_owned(), module));
        }
        if let Some(NativeNorm::Layer(module)) = &self.final_norm {
            modules.push(("final_norm".to_owned(), module));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            for (suffix, module) in [
                ("query", &layer.query),
                ("key", &layer.key),
                ("value", &layer.value),
                ("attention_output", &layer.attention_output),
                ("feed_forward_input", &layer.feed_forward_input),
                ("feed_forward_output", &layer.feed_forward_output),
            ] {
                modules.push((format!("{prefix}.{suffix}"), module));
            }
            if let Some(module) = &layer.feed_forward_gate {
                modules.push((format!("{prefix}.feed_forward_gate"), module));
            }
            if let NativeNorm::Layer(module) = &layer.attention_norm {
                modules.push((format!("{prefix}.attention_norm"), module));
            }
            if let NativeNorm::Layer(module) = &layer.feed_forward_norm {
                modules.push((format!("{prefix}.feed_forward_norm"), module));
            }
        }
        modules
    }
    fn normalization_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::new();
        if let Some(NativeNorm::Rms { weight, .. }) = &self.embedding_norm {
            tensors.push(("embedding_norm.weight".to_owned(), weight));
        }
        if let Some(NativeNorm::Rms { weight, .. }) = &self.final_norm {
            tensors.push(("final_norm.weight".to_owned(), weight));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if let NativeNorm::Rms { weight, .. } = &layer.attention_norm {
                tensors.push((format!("layers.{index}.attention_norm.weight"), weight));
            }
            if let NativeNorm::Rms { weight, .. } = &layer.feed_forward_norm {
                tensors.push((format!("layers.{index}.feed_forward_norm.weight"), weight));
            }
            if let Some(relative) = &layer.relative_attention_bias {
                tensors.push((format!("layers.{index}.relative_attention_bias"), relative));
            }
        }
        tensors
    }

    pub fn execution_requirements(&self) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        for module in [
            Some(&self.token_embedding),
            self.position_embedding.as_ref(),
            self.token_type_embedding.as_ref(),
            self.projection.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            requirements.extend(module.execution_requirements(DType::F32).iter());
        }
        if let Some(NativeNorm::Layer(module)) = &self.embedding_norm {
            requirements.extend(module.execution_requirements(DType::F32).iter());
        }
        for layer in &self.layers {
            for module in [
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.attention_output,
                &layer.feed_forward_input,
                &layer.activation,
                &layer.feed_forward_output,
            ] {
                requirements.extend(module.execution_requirements(DType::F32).iter());
            }
            if let Some(module) = &layer.feed_forward_gate {
                requirements.extend(module.execution_requirements(DType::F32).iter());
            }
            for norm in [&layer.attention_norm, &layer.feed_forward_norm] {
                if let NativeNorm::Layer(module) = norm {
                    requirements.extend(module.execution_requirements(DType::F32).iter());
                }
            }
        }
        requirements.extend([
            OperationSupport::allocation(DType::F32, Layout::Contiguous),
            OperationSupport::copy_input(DType::F32, Layout::Contiguous),
            OperationSupport::copy_output(DType::F32, Layout::Contiguous),
            OperationSupport::binary_input(BinaryOperation::Add, DType::F32, Layout::Contiguous),
            OperationSupport::binary_output(BinaryOperation::Add, DType::F32, Layout::Contiguous),
            OperationSupport::binary_input(
                BinaryOperation::Multiply,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::binary_output(
                BinaryOperation::Multiply,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::reduction_input(
                ReductionOperation::Mean,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::reduction_output(
                ReductionOperation::Mean,
                DType::F32,
                Layout::Contiguous,
            ),
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
                UnaryOperation::SquareRoot,
                DType::F32,
                Layout::Contiguous,
            ),
            OperationSupport::unary_output(
                UnaryOperation::SquareRoot,
                DType::F32,
                Layout::Contiguous,
            ),
        ]);
        requirements
    }

    pub fn admit_execution_target(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(), BidirectionalTextError> {
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

    pub fn embed_tokens(
        &self,
        backend: &CpuBackend,
        tokens: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, BidirectionalTextError> {
        self.admit_execution_target(backend, context)?;
        let request = BidirectionalTextRequest {
            input: BidirectionalTextInput::Tokens(tokens),
            attention_mask: None,
            token_type_ids: None,
            intermediate_layer: None,
            final_norm_intermediate: false,
            pooling: BidirectionalPooling::None,
            project_pooled: false,
        };
        validate_input(&self.configuration, &request, context)?;
        let mut token_embedding = self.token_embedding.clone();
        Ok(token_embedding.forward_with_context(backend, tokens, context)?)
    }

    pub fn forward(
        &self,
        backend: &CpuBackend,
        request: BidirectionalTextRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<BidirectionalTextOutput, BidirectionalTextError> {
        self.admit_execution_target(backend, context)?;
        let (batch, tokens) = validate_input(&self.configuration, &request, context)?;
        let mut token_embedding = self.token_embedding.clone();
        let mut hidden = match request.input {
            BidirectionalTextInput::Tokens(input) => {
                validate_token_values(backend, input, self.configuration.vocabulary_size, context)?;
                token_embedding.forward_with_context(backend, input, context)?
            }
            BidirectionalTextInput::Embeddings(input) => input.clone(),
        };
        if self.configuration.architecture == BidirectionalTextArchitecture::Bert {
            hidden = self.add_bert_embeddings(
                backend,
                &hidden,
                request.token_type_ids,
                batch,
                tokens,
                context,
            )?;
        }
        let capture = request
            .intermediate_layer
            .map(|index| resolve_layer(index, self.layers.len()))
            .transpose()?;
        let mut intermediate = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let inherited_relative_bias = self
                .layers
                .first()
                .and_then(|first| first.relative_attention_bias.as_ref())
                .filter(|_| self.configuration.relative_attention);
            let mask = prepare_attention_mask(
                backend,
                request.attention_mask,
                batch,
                tokens,
                self.configuration.attention_heads,
                layer
                    .relative_attention_bias
                    .as_ref()
                    .or(inherited_relative_bias),
                &self.configuration,
                context,
            )?;
            hidden = layer.forward(
                backend,
                &hidden,
                &mask,
                batch,
                tokens,
                &self.configuration,
                context,
            )?;
            if capture == Some(index) {
                intermediate = Some(hidden.clone());
            }
        }
        let last_hidden_state = match &self.final_norm {
            Some(norm) => apply_norm(
                backend,
                norm,
                &hidden,
                self.configuration.hidden_size,
                context,
            )?,
            None => hidden,
        };
        if request.final_norm_intermediate
            && let (Some(norm), Some(captured)) = (&self.final_norm, intermediate.take())
        {
            intermediate = Some(apply_norm(
                backend,
                norm,
                &captured,
                self.configuration.hidden_size,
                context,
            )?);
        }
        let pooled = pool_hidden(
            backend,
            &last_hidden_state,
            request.attention_mask,
            request.pooling,
            batch,
            tokens,
            self.configuration.hidden_size,
            context,
        )?;
        let projected_pooled = if request.project_pooled {
            let pooled = pooled.as_ref().ok_or(BidirectionalTextError::InvalidInput(
                "projection requires pooled output",
            ))?;
            let mut projection = self
                .projection
                .clone()
                .ok_or(BidirectionalTextError::MissingProjection)?;
            Some(projection.forward_with_context(backend, pooled, context)?)
        } else {
            None
        };
        Ok(BidirectionalTextOutput {
            last_hidden_state,
            intermediate,
            pooled,
            projected_pooled,
        })
    }

    fn add_bert_embeddings(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        token_type_ids: Option<&Tensor>,
        batch: usize,
        tokens: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, BidirectionalTextError> {
        let positions = repeated_indices(backend, batch, tokens, false, context)?;
        let mut position_embedding =
            self.position_embedding
                .clone()
                .ok_or(BidirectionalTextError::InvalidConfiguration(
                    "BERT position embedding is missing",
                ))?;
        let positions = position_embedding.forward_with_context(backend, &positions, context)?;
        let hidden = add(backend, input, &positions, context)?;
        let owned_types;
        let type_ids = match token_type_ids {
            Some(type_ids) => {
                validate_token_values(
                    backend,
                    type_ids,
                    self.configuration.type_vocabulary_size,
                    context,
                )?;
                type_ids
            }
            None => {
                owned_types = repeated_indices(backend, batch, tokens, true, context)?;
                &owned_types
            }
        };
        let mut token_type_embedding = self.token_type_embedding.clone().ok_or(
            BidirectionalTextError::InvalidConfiguration("BERT token-type embedding is missing"),
        )?;
        let types = token_type_embedding.forward_with_context(backend, type_ids, context)?;
        let hidden = add(backend, &hidden, &types, context)?;
        let norm =
            self.embedding_norm
                .as_ref()
                .ok_or(BidirectionalTextError::InvalidConfiguration(
                    "BERT embedding normalization is missing",
                ))?;
        apply_norm(
            backend,
            norm,
            &hidden,
            self.configuration.hidden_size,
            context,
        )
    }
}

impl NativeBidirectionalLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        mask: &PreparedAttentionMask,
        batch: usize,
        tokens: usize,
        configuration: &BidirectionalTextConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, BidirectionalTextError> {
        context.check()?;
        let attention_input = match configuration.architecture {
            BidirectionalTextArchitecture::T5 => apply_norm(
                backend,
                &self.attention_norm,
                input,
                configuration.hidden_size,
                context,
            )?,
            BidirectionalTextArchitecture::Bert => input.clone(),
        };
        let mut query = self.query.clone();
        let mut key = self.key.clone();
        let mut value = self.value.clone();
        let query = query.forward_with_context(backend, &attention_input, context)?;
        let key = key.forward_with_context(backend, &attention_input, context)?;
        let value = value.forward_with_context(backend, &attention_input, context)?;
        let query_values = tensor_to_f32(backend, &query, context)?;
        let key_values = tensor_to_f32(backend, &key, context)?;
        let value_values = tensor_to_f32(backend, &value, context)?;
        let attention = scaled_dot_product_attention_with_context(
            backend,
            AttentionRequest {
                backend: AttentionBackend::PytorchSdp,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch,
                query_tokens: tokens,
                key_tokens: tokens,
                heads: configuration.attention_heads,
                head_dimension: configuration.attention_inner_size / configuration.attention_heads,
                value_dimension: configuration.attention_inner_size / configuration.attention_heads,
                scale: (configuration.architecture == BidirectionalTextArchitecture::T5)
                    .then_some(1.0),
                workspace_limit_bytes: tokens
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(BidirectionalTextError::Overflow("attention workspace"))?,
            },
            &query_values,
            &key_values,
            &value_values,
            Some(mask.as_attention_mask()),
            context,
        )?;
        let attention = tensor_from_f32(
            backend,
            &[
                usize_to_u64(batch, "attention batch")?,
                usize_to_u64(tokens, "attention tokens")?,
                usize_to_u64(configuration.attention_inner_size, "attention inner size")?,
            ],
            &attention.values,
            context,
        )?;
        let mut attention_output = self.attention_output.clone();
        let attention = attention_output.forward_with_context(backend, &attention, context)?;
        let attention_residual = add(backend, input, &attention, context)?;
        let feed_forward_input = match configuration.architecture {
            BidirectionalTextArchitecture::T5 => apply_norm(
                backend,
                &self.feed_forward_norm,
                &attention_residual,
                configuration.hidden_size,
                context,
            )?,
            BidirectionalTextArchitecture::Bert => apply_norm(
                backend,
                &self.attention_norm,
                &attention_residual,
                configuration.hidden_size,
                context,
            )?,
        };
        let mut feed_forward_input_module = self.feed_forward_input.clone();
        let feed_forward = feed_forward_input_module.forward_with_context(
            backend,
            &feed_forward_input,
            context,
        )?;
        let mut activation = self.activation.clone();
        let activated = activation.forward_with_context(backend, &feed_forward, context)?;
        let activated = match &self.feed_forward_gate {
            Some(gate) => {
                let mut gate = gate.clone();
                let gate = gate.forward_with_context(backend, &feed_forward_input, context)?;
                multiply(backend, &activated, &gate, context)?
            }
            None => activated,
        };
        let mut feed_forward_output = self.feed_forward_output.clone();
        let feed_forward =
            feed_forward_output.forward_with_context(backend, &activated, context)?;
        let residual_base = match configuration.architecture {
            BidirectionalTextArchitecture::T5 => &attention_residual,
            BidirectionalTextArchitecture::Bert => &feed_forward_input,
        };
        let residual = add(backend, residual_base, &feed_forward, context)?;
        match configuration.architecture {
            BidirectionalTextArchitecture::T5 => Ok(residual),
            BidirectionalTextArchitecture::Bert => apply_norm(
                backend,
                &self.feed_forward_norm,
                &residual,
                configuration.hidden_size,
                context,
            ),
        }
    }
}

struct PreparedAttentionMask {
    values: CpuWorkspaceVec<f32>,
    shape: AttentionMaskShape,
}

impl PreparedAttentionMask {
    fn as_attention_mask(&self) -> AttentionMask<'_> {
        AttentionMask::Additive {
            values: &self.values,
            shape: self.shape,
        }
    }
}

pub fn relative_position_bucket(
    relative_position: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> Result<usize, BidirectionalTextError> {
    if num_buckets < 2 || max_distance == 0 || (bidirectional && !num_buckets.is_multiple_of(2)) {
        return Err(BidirectionalTextError::InvalidConfiguration(
            "relative-position bucket configuration is invalid",
        ));
    }
    let mut buckets = num_buckets;
    let mut result = 0_usize;
    let distance = if bidirectional {
        buckets /= 2;
        if relative_position > 0 {
            result = buckets;
        }
        usize::try_from(relative_position.unsigned_abs())
            .map_err(|_| BidirectionalTextError::Overflow("relative position"))?
    } else {
        usize::try_from(relative_position.saturating_neg().max(0))
            .map_err(|_| BidirectionalTextError::Overflow("relative position"))?
    };
    let maximum_exact = buckets / 2;
    if distance < maximum_exact {
        return result
            .checked_add(distance)
            .ok_or(BidirectionalTextError::Overflow("relative bucket"));
    }
    if maximum_exact == 0 || max_distance <= maximum_exact {
        return Err(BidirectionalTextError::InvalidConfiguration(
            "relative-position logarithmic range is invalid",
        ));
    }
    let logarithmic = ((distance as f64 / maximum_exact as f64).ln()
        / (max_distance as f64 / maximum_exact as f64).ln()
        * (buckets - maximum_exact) as f64) as usize;
    result
        .checked_add(maximum_exact)
        .and_then(|value| value.checked_add(logarithmic.min(buckets - maximum_exact - 1)))
        .ok_or(BidirectionalTextError::Overflow("relative bucket"))
}

pub fn tokenize_bidirectional_prompt(
    tokenizer: &NativePromptTokenizer,
    text: &str,
    cancellation: &sim_types::CancellationToken,
) -> Result<NativeTokenizedPrompt, BidirectionalTextError> {
    tokenizer
        .tokenize(text, cancellation)
        .map_err(BidirectionalTextError::from)
}

fn build_layer(
    index: usize,
    configuration: &BidirectionalTextConfiguration,
    weights: BidirectionalLayerWeights,
    stream: StreamId,
) -> Result<NativeBidirectionalLayer, BidirectionalTextError> {
    let prefix = format!("bidirectional.layers.{index}");
    let attention_norm = build_norm(
        format!("{prefix}.attention_norm"),
        configuration,
        weights.attention_norm_weight,
        weights.attention_norm_bias,
        stream,
    )?;
    let feed_forward_norm = build_norm(
        format!("{prefix}.feed_forward_norm"),
        configuration,
        weights.feed_forward_norm_weight,
        weights.feed_forward_norm_bias,
        stream,
    )?;
    let query = linear_module(
        format!("{prefix}.query"),
        configuration.hidden_size,
        configuration.attention_inner_size,
        weights.query_weight,
        weights.query_bias,
        stream,
    )?;
    let key = linear_module(
        format!("{prefix}.key"),
        configuration.hidden_size,
        configuration.attention_inner_size,
        weights.key_weight,
        weights.key_bias,
        stream,
    )?;
    let value = linear_module(
        format!("{prefix}.value"),
        configuration.hidden_size,
        configuration.attention_inner_size,
        weights.value_weight,
        weights.value_bias,
        stream,
    )?;
    let attention_output = linear_module(
        format!("{prefix}.attention_output"),
        configuration.attention_inner_size,
        configuration.hidden_size,
        weights.attention_output_weight,
        weights.attention_output_bias,
        stream,
    )?;
    let feed_forward_input = linear_module(
        format!("{prefix}.feed_forward_input"),
        configuration.hidden_size,
        configuration.feed_forward_size,
        weights.feed_forward_input_weight,
        weights.feed_forward_input_bias,
        stream,
    )?;
    let feed_forward_gate = match weights.feed_forward_gate_weight {
        Some(weight) if configuration.gated_feed_forward => Some(linear_module(
            format!("{prefix}.feed_forward_gate"),
            configuration.hidden_size,
            configuration.feed_forward_size,
            weight,
            weights.feed_forward_gate_bias,
            stream,
        )?),
        None if !configuration.gated_feed_forward && weights.feed_forward_gate_bias.is_none() => {
            None
        }
        _ => {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "feed-forward gate configuration and weights must agree",
            ));
        }
    };
    let feed_forward_output = linear_module(
        format!("{prefix}.feed_forward_output"),
        configuration.feed_forward_size,
        configuration.hidden_size,
        weights.feed_forward_output_weight,
        weights.feed_forward_output_bias,
        stream,
    )?;
    let activation = match configuration.activation {
        BidirectionalFeedForwardActivation::Relu => {
            NativeModule::relu(format!("{prefix}.activation"))?
        }
        BidirectionalFeedForwardActivation::Gelu => {
            NativeModule::gelu(format!("{prefix}.activation"), GeluApproximation::None)?
        }
        BidirectionalFeedForwardActivation::GeluTanh => {
            NativeModule::gelu(format!("{prefix}.activation"), GeluApproximation::Tanh)?
        }
    };
    match (
        &weights.relative_attention_bias,
        configuration.architecture,
        index,
    ) {
        (Some(bias), BidirectionalTextArchitecture::T5, 0) => {
            require_parameter(bias, stream)?;
            let expected = [
                usize_to_u64(configuration.relative_attention_buckets, "relative buckets")?,
                usize_to_u64(configuration.attention_heads, "attention heads")?,
            ];
            if bias.descriptor().shape() != expected {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "relative-attention bias shape is invalid",
                ));
            }
        }
        (Some(bias), BidirectionalTextArchitecture::T5, _) if !configuration.relative_attention => {
            require_parameter(bias, stream)?;
            let expected = [
                usize_to_u64(configuration.relative_attention_buckets, "relative buckets")?,
                usize_to_u64(configuration.attention_heads, "attention heads")?,
            ];
            if bias.descriptor().shape() != expected {
                return Err(BidirectionalTextError::InvalidConfiguration(
                    "relative-attention bias shape is invalid",
                ));
            }
        }
        (None, BidirectionalTextArchitecture::T5, 0) => {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "the configured T5 layer requires relative-attention bias",
            ));
        }
        (None, BidirectionalTextArchitecture::T5, _) if !configuration.relative_attention => {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "the configured T5 layer requires relative-attention bias",
            ));
        }
        (None, _, _) => {}
        _ => {
            return Err(BidirectionalTextError::InvalidConfiguration(
                "relative-attention bias is owned only by the configured T5 layers",
            ));
        }
    }
    Ok(NativeBidirectionalLayer {
        attention_norm,
        query,
        key,
        value,
        attention_output,
        feed_forward_norm,
        feed_forward_input,
        feed_forward_gate,
        activation,
        feed_forward_output,
        relative_attention_bias: weights.relative_attention_bias,
    })
}

fn build_norm(
    name: String,
    configuration: &BidirectionalTextConfiguration,
    weight: Tensor,
    bias: Option<Tensor>,
    stream: StreamId,
) -> Result<NativeNorm, BidirectionalTextError> {
    require_vector_parameter(&weight, configuration.hidden_size, stream)?;
    match configuration.architecture {
        BidirectionalTextArchitecture::T5 if bias.is_none() => Ok(NativeNorm::Rms {
            weight,
            epsilon: configuration.normalization_epsilon(),
        }),
        BidirectionalTextArchitecture::Bert => {
            let bias = bias.ok_or(BidirectionalTextError::InvalidConfiguration(
                "BERT layer normalization requires bias",
            ))?;
            require_vector_parameter(&bias, configuration.hidden_size, stream)?;
            let mut module = NativeModule::layer_norm(
                name,
                vec![configuration.hidden_size],
                configuration.normalization_epsilon(),
                true,
                true,
                false,
            )?;
            module.load_dense_parameters(weight, Some(bias))?;
            Ok(NativeNorm::Layer(module))
        }
        BidirectionalTextArchitecture::T5 => Err(BidirectionalTextError::InvalidConfiguration(
            "T5 RMS normalization does not accept bias",
        )),
    }
}

fn linear_module(
    name: String,
    input: usize,
    output: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    stream: StreamId,
) -> Result<NativeModule, BidirectionalTextError> {
    require_parameter(&weight, stream)?;
    if let Some(bias) = bias.as_ref() {
        require_parameter(bias, stream)?;
    }
    let mut module = NativeModule::linear(name, input, output, bias.is_some(), false)?;
    module.load_dense_parameters(weight, bias)?;
    Ok(module)
}

fn embedding_module(
    name: &'static str,
    count: usize,
    dimensions: usize,
    weight: Tensor,
    stream: StreamId,
) -> Result<NativeModule, BidirectionalTextError> {
    require_parameter(&weight, stream)?;
    let mut module =
        NativeModule::embedding(name, count, dimensions, EmbeddingOptions::default(), false)?;
    module.load_dense_parameters(weight, None)?;
    Ok(module)
}

fn require_parameter(tensor: &Tensor, stream: StreamId) -> Result<(), BidirectionalTextError> {
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != stream
        || !descriptor.is_contiguous()?
    {
        return Err(BidirectionalTextError::InvalidConfiguration(
            "parameters must be contiguous CPU F32 tensors on one stream",
        ));
    }
    Ok(())
}

fn require_vector_parameter(
    tensor: &Tensor,
    width: usize,
    stream: StreamId,
) -> Result<(), BidirectionalTextError> {
    require_parameter(tensor, stream)?;
    if tensor.descriptor().shape() != [usize_to_u64(width, "parameter width")?] {
        return Err(BidirectionalTextError::InvalidConfiguration(
            "normalization parameter width is invalid",
        ));
    }
    Ok(())
}

fn validate_input(
    configuration: &BidirectionalTextConfiguration,
    request: &BidirectionalTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<(usize, usize), BidirectionalTextError> {
    let tensor = match request.input {
        BidirectionalTextInput::Tokens(tensor) | BidirectionalTextInput::Embeddings(tensor) => {
            tensor
        }
    };
    let shape = tensor.descriptor().shape();
    let (batch, tokens) = match request.input {
        BidirectionalTextInput::Tokens(_) if shape.len() == 2 => (
            usize::try_from(shape[0]).map_err(|_| BidirectionalTextError::Overflow("batch"))?,
            usize::try_from(shape[1]).map_err(|_| BidirectionalTextError::Overflow("tokens"))?,
        ),
        BidirectionalTextInput::Embeddings(_) if shape.len() == 3 => {
            if shape[2] != usize_to_u64(configuration.hidden_size, "hidden size")? {
                return Err(BidirectionalTextError::InvalidInput(
                    "embedding width does not match configuration",
                ));
            }
            (
                usize::try_from(shape[0]).map_err(|_| BidirectionalTextError::Overflow("batch"))?,
                usize::try_from(shape[1])
                    .map_err(|_| BidirectionalTextError::Overflow("tokens"))?,
            )
        }
        _ => {
            return Err(BidirectionalTextError::InvalidInput(
                "tokens require [batch, tokens] and embeddings require [batch, tokens, hidden]",
            ));
        }
    };
    if batch == 0 || tokens == 0 || tokens > configuration.maximum_tokens {
        return Err(BidirectionalTextError::InvalidInput(
            "batch and token dimensions must be nonzero and bounded",
        ));
    }
    let descriptor = tensor.descriptor();
    let expected_dtype = match request.input {
        BidirectionalTextInput::Tokens(_) => DType::I64,
        BidirectionalTextInput::Embeddings(_) => DType::F32,
    };
    if descriptor.dtype() != expected_dtype
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != context.stream
        || !descriptor.is_contiguous()?
    {
        return Err(BidirectionalTextError::InvalidInput(
            "input target, dtype, stream, or layout is invalid",
        ));
    }
    if let Some(mask) = request.attention_mask
        && mask.descriptor().shape()
            != [
                usize_to_u64(batch, "batch")?,
                usize_to_u64(tokens, "tokens")?,
            ]
    {
        return Err(BidirectionalTextError::InvalidInput(
            "attention mask must have [batch, tokens] shape",
        ));
    }
    if let Some(types) = request.token_type_ids
        && (configuration.architecture != BidirectionalTextArchitecture::Bert
            || types.descriptor().shape()
                != [
                    usize_to_u64(batch, "batch")?,
                    usize_to_u64(tokens, "tokens")?,
                ])
    {
        return Err(BidirectionalTextError::InvalidInput(
            "token-type IDs are valid only for BERT with [batch, tokens] shape",
        ));
    }
    Ok((batch, tokens))
}

fn validate_token_values(
    backend: &CpuBackend,
    tensor: &Tensor,
    vocabulary_size: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), BidirectionalTextError> {
    if tensor.descriptor().dtype() != DType::I64 {
        return Err(BidirectionalTextError::InvalidInput(
            "token tensors must use I64",
        ));
    }
    for value in read_i64(backend, tensor, context)?.iter() {
        if *value < 0 || usize::try_from(*value).map_or(true, |value| value >= vocabulary_size) {
            return Err(BidirectionalTextError::TokenOutOfRange(*value));
        }
    }
    Ok(())
}

fn read_i64(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<i64>, BidirectionalTextError> {
    context.check()?;
    let bytes = tensor.contiguous_bytes()?;
    if !bytes.len().is_multiple_of(std::mem::size_of::<i64>()) {
        return Err(BidirectionalTextError::InvalidInput(
            "I64 tensor bytes are unaligned",
        ));
    }
    let mut values = backend.workspace_vec(context, bytes.len() / std::mem::size_of::<i64>())?;
    for (index, bytes) in bytes.chunks_exact(std::mem::size_of::<i64>()).enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        let encoded: [u8; 8] = bytes
            .try_into()
            .map_err(|_| BidirectionalTextError::InvalidInput("I64 tensor bytes are unaligned"))?;
        values.try_push(i64::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

fn read_mask(
    backend: &CpuBackend,
    mask: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, BidirectionalTextError> {
    if mask.descriptor().device() != DeviceId::CPU
        || mask.descriptor().stream() != context.stream
        || !mask.descriptor().is_contiguous()?
    {
        return Err(BidirectionalTextError::InvalidInput(
            "attention mask target, stream, or layout is invalid",
        ));
    }
    let values = match mask.descriptor().dtype() {
        DType::F32 => tensor_to_f32(backend, mask, context)?,
        DType::I64 => {
            let integers = read_i64(backend, mask, context)?;
            let mut values = backend.workspace_vec(context, integers.len())?;
            for value in integers.iter() {
                values.try_push(*value as f32)?;
            }
            values
        }
        _ => {
            return Err(BidirectionalTextError::InvalidInput(
                "attention mask must use F32 or I64",
            ));
        }
    };
    if values.iter().any(|value| !matches!(*value, 0.0 | 1.0)) {
        return Err(BidirectionalTextError::InvalidInput(
            "attention mask values must be exactly zero or one",
        ));
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn prepare_attention_mask(
    backend: &CpuBackend,
    attention_mask: Option<&Tensor>,
    batch: usize,
    tokens: usize,
    heads: usize,
    relative_attention_bias: Option<&Tensor>,
    configuration: &BidirectionalTextConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<PreparedAttentionMask, BidirectionalTextError> {
    let mask = attention_mask
        .map(|mask| read_mask(backend, mask, context))
        .transpose()?;
    let relative_bias = relative_attention_bias
        .map(|bias| tensor_to_f32(backend, bias, context))
        .transpose()?;
    let count = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(tokens))
        .and_then(|value| value.checked_mul(tokens))
        .ok_or(BidirectionalTextError::Overflow("attention mask"))?;
    let mut values = backend.workspace_vec(context, count)?;
    for batch_index in 0..batch {
        for head in 0..heads {
            for query in 0..tokens {
                for key in 0..tokens {
                    context.check()?;
                    let padding = mask
                        .as_ref()
                        .is_some_and(|mask| mask.get(batch_index * tokens + key) == Some(&0.0));
                    let value = if padding {
                        -f32::MAX
                    } else if let Some(relative_bias) = &relative_bias {
                        let relative = i64::try_from(key)
                            .and_then(|key| i64::try_from(query).map(|query| key - query))
                            .map_err(|_| BidirectionalTextError::Overflow("relative position"))?;
                        let bucket = relative_position_bucket(
                            relative,
                            true,
                            configuration.relative_attention_buckets,
                            configuration.relative_attention_max_distance,
                        )?;
                        *relative_bias.get(bucket * heads + head).ok_or(
                            BidirectionalTextError::InvalidConfiguration(
                                "relative-attention bias index is invalid",
                            ),
                        )?
                    } else {
                        0.0
                    };
                    values.try_push(value)?;
                }
            }
        }
    }
    Ok(PreparedAttentionMask {
        values,
        shape: AttentionMaskShape::BatchHeadQueryByKey,
    })
}

fn apply_norm(
    backend: &CpuBackend,
    norm: &NativeNorm,
    input: &Tensor,
    hidden_size: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, BidirectionalTextError> {
    match norm {
        NativeNorm::Layer(module) => {
            let mut module = module.clone();
            module
                .forward_with_context(backend, input, context)
                .map_err(Into::into)
        }
        NativeNorm::Rms { weight, epsilon } => {
            let input_values = tensor_to_f32(backend, input, context)?;
            let weight_values = tensor_to_f32(backend, weight, context)?;
            let shape = input
                .descriptor()
                .shape()
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .map_err(|_| BidirectionalTextError::Overflow("normalization shape"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let values = rms_norm_with_context_exact_native(
                backend,
                &input_values,
                &shape,
                &[hidden_size],
                Some(&weight_values),
                Some(*epsilon),
                DeviceId::CPU,
                context,
            )?;
            tensor_from_f32(backend, input.descriptor().shape(), &values, context)
                .map_err(Into::into)
        }
    }
}

fn multiply(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, BidirectionalTextError> {
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(BidirectionalTextError::InvalidInput(
            "gated feed-forward tensors must have equal shape",
        ));
    }
    let left_values = tensor_to_f32(backend, left, context)?;
    let right_values = tensor_to_f32(backend, right, context)?;
    let mut values = backend.workspace_vec(context, left_values.len())?;
    for (index, (left, right)) in left_values.iter().zip(right_values.iter()).enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        values.try_push(left * right)?;
    }
    tensor_from_f32(backend, left.descriptor().shape(), &values, context).map_err(Into::into)
}

fn repeated_indices(
    backend: &CpuBackend,
    batch: usize,
    tokens: usize,
    zeros: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, BidirectionalTextError> {
    let count = batch
        .checked_mul(tokens)
        .ok_or(BidirectionalTextError::Overflow("index tensor"))?;
    let byte_count = count
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(BidirectionalTextError::Overflow("index tensor bytes"))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for _batch_index in 0..batch {
        for token in 0..tokens {
            context.check()?;
            let value = if zeros {
                0_i64
            } else {
                i64::try_from(token)
                    .map_err(|_| BidirectionalTextError::Overflow("position index"))?
            };
            for byte in value.to_ne_bytes() {
                bytes.try_push(byte)?;
            }
        }
    }
    let descriptor = sim_tensor::TensorDescriptor::contiguous(
        vec![
            usize_to_u64(batch, "batch")?,
            usize_to_u64(tokens, "tokens")?,
        ],
        DType::I64,
        DeviceId::CPU,
        context.stream,
    )?;
    let (tensor, _) = backend.upload_bytes(descriptor, &bytes, context)?;
    Ok(tensor)
}

#[allow(clippy::too_many_arguments)]
fn pool_hidden(
    backend: &CpuBackend,
    hidden: &Tensor,
    attention_mask: Option<&Tensor>,
    pooling: BidirectionalPooling,
    batch: usize,
    tokens: usize,
    hidden_size: usize,
    context: &ExecutionContext<'_>,
) -> Result<Option<Tensor>, BidirectionalTextError> {
    if pooling == BidirectionalPooling::None {
        return Ok(None);
    }
    let hidden_values = tensor_to_f32(backend, hidden, context)?;
    let mask = attention_mask
        .map(|mask| read_mask(backend, mask, context))
        .transpose()?;
    let output_count = batch
        .checked_mul(hidden_size)
        .ok_or(BidirectionalTextError::Overflow("pooled output"))?;
    let mut output = backend.workspace_vec(context, output_count)?;
    for batch_index in 0..batch {
        match pooling {
            BidirectionalPooling::FirstToken => {
                let start = batch_index
                    .checked_mul(tokens)
                    .and_then(|value| value.checked_mul(hidden_size))
                    .ok_or(BidirectionalTextError::Overflow("first-token pool"))?;
                let end = start
                    .checked_add(hidden_size)
                    .ok_or(BidirectionalTextError::Overflow("first-token pool"))?;
                let values =
                    hidden_values
                        .get(start..end)
                        .ok_or(BidirectionalTextError::InvalidInput(
                            "hidden tensor is truncated",
                        ))?;
                for value in values {
                    output.try_push(*value)?;
                }
            }
            BidirectionalPooling::MeanUnmasked => {
                let included = (0..tokens)
                    .filter(|token| {
                        mask.as_ref().is_none_or(|mask| {
                            mask.get(batch_index * tokens + *token) == Some(&1.0)
                        })
                    })
                    .count();
                if included == 0 {
                    return Err(BidirectionalTextError::InvalidInput(
                        "mean pooling requires at least one unmasked token",
                    ));
                }
                for feature in 0..hidden_size {
                    let mut total = 0.0_f32;
                    for token in 0..tokens {
                        if mask.as_ref().is_some_and(|mask| {
                            mask.get(batch_index * tokens + token) == Some(&0.0)
                        }) {
                            continue;
                        }
                        let index = batch_index
                            .checked_mul(tokens)
                            .and_then(|value| value.checked_add(token))
                            .and_then(|value| value.checked_mul(hidden_size))
                            .and_then(|value| value.checked_add(feature))
                            .ok_or(BidirectionalTextError::Overflow("mean pool"))?;
                        total += hidden_values.get(index).copied().ok_or(
                            BidirectionalTextError::InvalidInput("hidden tensor is truncated"),
                        )?;
                    }
                    output.try_push(total / included as f32)?;
                }
            }
            BidirectionalPooling::None => unreachable!(),
        }
    }
    Ok(Some(tensor_from_f32(
        backend,
        &[
            usize_to_u64(batch, "batch")?,
            usize_to_u64(hidden_size, "hidden size")?,
        ],
        &output,
        context,
    )?))
}

fn resolve_layer(requested: isize, available: usize) -> Result<usize, BidirectionalTextError> {
    let available_isize =
        isize::try_from(available).map_err(|_| BidirectionalTextError::Overflow("layer count"))?;
    let resolved = if requested < 0 {
        available_isize.checked_add(requested).ok_or(
            BidirectionalTextError::IntermediateOutOfRange {
                requested,
                available,
            },
        )?
    } else {
        requested
    };
    let resolved =
        usize::try_from(resolved).map_err(|_| BidirectionalTextError::IntermediateOutOfRange {
            requested,
            available,
        })?;
    if resolved >= available {
        return Err(BidirectionalTextError::IntermediateOutOfRange {
            requested,
            available,
        });
    }
    Ok(resolved)
}

fn usize_to_u64(value: usize, name: &'static str) -> Result<u64, BidirectionalTextError> {
    u64::try_from(value).map_err(|_| BidirectionalTextError::Overflow(name))
}
