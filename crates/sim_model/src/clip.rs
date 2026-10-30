use crate::clip_text_encoder_composite::{
    compose_source_hidream, compose_source_sd3, compose_source_sdxl,
};
use crate::clip_text_encoder_multimodal::{
    GemmaConditioningRequest, GemmaPreparedAudio, GemmaPreparedVisual, NativeGemmaMultimodal,
    NativeMultimodalOwnedAllocationKind, NativeQwen25Multimodal, Qwen25ConditioningRequest,
    Qwen25PreparedImage,
};
use crate::clip_tokenizer::{
    ClipBpeTokenizer, Llama3BpeTokenizer, NativeTokenizerFamily, NativeTokenizerInvocationOptions,
    SentencePieceTokenizer, TokenizerConfiguration, apply_empty_baseline_token_weights,
};
use crate::native_ops::{
    NativeExecutionRequirements, NativeModelExecutionAdmission, NativeModelExecutionRole,
    admit_native_model_resource_execution,
};
use crate::{
    ArtifactIndex, AttentionError, BidirectionalFeedForwardActivation, BidirectionalLayerWeights,
    BidirectionalPooling, BidirectionalTextArchitecture, BidirectionalTextConfiguration,
    BidirectionalTextInput, BidirectionalTextRequest, BidirectionalTextWeights, ClipTextActivation,
    ClipTextConfiguration, ClipTextError, ClipTextInput, ClipTextIntermediate,
    ClipTextLayerWeights, ClipTextRequest, ClipTextWeights, DecoderActivation, DecoderArchitecture,
    DecoderAttentionWeights, DecoderLayerKind, DecoderLayerWeights, DecoderPreparedTextRequest,
    DecoderRopeConfiguration, DecoderRopePositions, DecoderTextConfiguration, DecoderTextWeights,
    EmbeddingOptions, HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT, LoadedModel, MappedModelWeights,
    ModelClipTargetCandidateDescriptor, ModelClipTargetDescriptor, ModelFamilyIdentity, ModelProbe,
    ModelStorageDType, ModelStore, ModelStoreError, ModelTokenizerDescriptor, NativeClipText,
    NativeDecoderTextEncoder, NativeModule, NativeOpsError, NativePromptTokenizer,
    NativeT5TextEncoder, NativeTokenValue, ParserLimits, PatchGraph, PatchGraphError,
    PatchGraphIdentity, PatchGraphIdentityError, ResolvedModelFamily, RopeScaling,
};
use regex::Regex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId,
    ExecutionContext, Layout, LinearAlgebraOperation, OperationSupport, ReductionOperation,
    ResizeMode, StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    ViewAccess,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32, tensor_to_f32},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    sync::Arc,
};
use thiserror::Error;

pub const SD1_CONTEXT_LENGTH: usize = 77;
pub const SD1_VOCABULARY_SIZE: usize = 49_408;
pub const SD1_MERGE_COUNT: usize = 48_894;
pub const SD1_START_TOKEN: u32 = 49_406;
pub const SD1_END_TOKEN: u32 = 49_407;
const MAX_TOKEN_BATCH: usize = 64;
const SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH: usize = 99_999_999;
pub const SD1_MAX_WEIGHTED_SEGMENTS: usize = 1_024;
pub const SD1_MAX_PROMPT_BYTES: usize = 1_048_576;
const MAX_CLIP_ARTIFACTS: usize = 4;
#[allow(
    dead_code,
    reason = "registered CLIP architecture leaves construct bounded manifests"
)]
const MAX_CLIP_PARAMETERS: usize = 262_144;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipType {
    StableDiffusion,
    StableCascade,
    Sd3,
    StableAudio,
    HunyuanDit,
    Flux,
    Mochi,
    Ltxv,
    HunyuanVideo,
    Pixart,
    Cosmos,
    Lumina2,
    Wan,
    Hidream,
    Chroma,
    Ace,
    Omnigen2,
    QwenImage,
    HunyuanImage,
    HunyuanVideo15,
    Ovis,
    Kandinsky5,
    Kandinsky5Image,
    Newbie,
    Flux2,
    LongcatImage,
    Cogvideox,
    Lens,
    Pixeldit,
    Ideogram4,
    Boogu,
    Krea2,
}

impl ClipType {
    pub const ALL: [Self; 32] = [
        Self::StableDiffusion,
        Self::StableCascade,
        Self::Sd3,
        Self::StableAudio,
        Self::HunyuanDit,
        Self::Flux,
        Self::Mochi,
        Self::Ltxv,
        Self::HunyuanVideo,
        Self::Pixart,
        Self::Cosmos,
        Self::Lumina2,
        Self::Wan,
        Self::Hidream,
        Self::Chroma,
        Self::Ace,
        Self::Omnigen2,
        Self::QwenImage,
        Self::HunyuanImage,
        Self::HunyuanVideo15,
        Self::Ovis,
        Self::Kandinsky5,
        Self::Kandinsky5Image,
        Self::Newbie,
        Self::Flux2,
        Self::LongcatImage,
        Self::Cogvideox,
        Self::Lens,
        Self::Pixeldit,
        Self::Ideogram4,
        Self::Boogu,
        Self::Krea2,
    ];

    pub const fn source_ordinal(self) -> u8 {
        self as u8 + 1
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextEncoderModel {
    ClipL,
    ClipH,
    ClipG,
    T5Xxl,
    T5Xl,
    T5Base,
    Llama3_8,
    T5XxlOld,
    Gemma2_2b,
    Qwen25_3b,
    Qwen25_7b,
    ByT5SmallGlyph,
    Gemma3_4b,
    Mistral3_24b,
    Mistral3_24bPrunedFlux2,
    Qwen3_4b,
    Qwen3_2b,
    Gemma3_12b,
    JinaClip2,
    Qwen3_8b,
    Qwen3_06b,
    Gemma3_4bVision,
    Qwen35_08b,
    Qwen35_2b,
    Qwen35_4b,
    Qwen35_9b,
    Qwen35_27b,
    Ministral3_3b,
    Gemma4E4b,
    Gemma4E2b,
    Gemma4_31b,
    T5Gemma,
    GptOss20b,
    Qwen3Vl4b,
    Qwen3Vl8b,
}

impl TextEncoderModel {
    pub const ALL: [Self; 35] = [
        Self::ClipL,
        Self::ClipH,
        Self::ClipG,
        Self::T5Xxl,
        Self::T5Xl,
        Self::T5Base,
        Self::Llama3_8,
        Self::T5XxlOld,
        Self::Gemma2_2b,
        Self::Qwen25_3b,
        Self::Qwen25_7b,
        Self::ByT5SmallGlyph,
        Self::Gemma3_4b,
        Self::Mistral3_24b,
        Self::Mistral3_24bPrunedFlux2,
        Self::Qwen3_4b,
        Self::Qwen3_2b,
        Self::Gemma3_12b,
        Self::JinaClip2,
        Self::Qwen3_8b,
        Self::Qwen3_06b,
        Self::Gemma3_4bVision,
        Self::Qwen35_08b,
        Self::Qwen35_2b,
        Self::Qwen35_4b,
        Self::Qwen35_9b,
        Self::Qwen35_27b,
        Self::Ministral3_3b,
        Self::Gemma4E4b,
        Self::Gemma4E2b,
        Self::Gemma4_31b,
        Self::T5Gemma,
        Self::GptOss20b,
        Self::Qwen3Vl4b,
        Self::Qwen3Vl8b,
    ];

    pub const fn source_ordinal(self) -> u8 {
        self as u8 + 1
    }
}

pub fn detect_text_encoder_model(
    tensor_shapes: &BTreeMap<String, Vec<u64>>,
) -> Result<Option<TextEncoderModel>, ClipError> {
    let has = |name: &str| tensor_shapes.contains_key(name);
    let first = |name: &str| -> Result<Option<u64>, ClipError> {
        tensor_shapes
            .get(name)
            .map(|shape| {
                shape
                    .first()
                    .copied()
                    .ok_or_else(|| ClipError::InvalidDetectorTensor(name.to_owned()))
            })
            .transpose()
    };
    if has("text_model.encoder.layers.30.mlp.fc1.weight") {
        return Ok(Some(TextEncoderModel::ClipG));
    }
    if has("text_model.encoder.layers.22.mlp.fc1.weight") {
        return Ok(Some(TextEncoderModel::ClipH));
    }
    if has("text_model.encoder.layers.0.mlp.fc1.weight") {
        return Ok(Some(TextEncoderModel::ClipL));
    }
    if has("model.encoder.layers.0.mixer.Wqkv.weight") {
        return Ok(Some(TextEncoderModel::JinaClip2));
    }
    match first("encoder.block.23.layer.1.DenseReluDense.wi_1.weight")? {
        Some(10_240) => return Ok(Some(TextEncoderModel::T5Xxl)),
        Some(5_120) => return Ok(Some(TextEncoderModel::T5Xl)),
        _ => {}
    }
    if has("encoder.block.23.layer.1.DenseReluDense.wi.weight") {
        return Ok(Some(TextEncoderModel::T5XxlOld));
    }
    if let Some(width) = first("encoder.block.0.layer.0.SelfAttention.k.weight")? {
        return Ok(Some(if width == 384 {
            TextEncoderModel::ByT5SmallGlyph
        } else {
            TextEncoderModel::T5Base
        }));
    }
    if has("model.encoder.layers.0.pre_self_attn_layernorm.weight") {
        return Ok(Some(TextEncoderModel::T5Gemma));
    }
    if has("model.layers.0.post_feedforward_layernorm.weight") {
        if has("model.layers.59.self_attn.q_norm.weight") {
            return Ok(Some(TextEncoderModel::Gemma4_31b));
        }
        if has("model.layers.41.self_attn.q_norm.weight")
            && !has("model.layers.47.self_attn.q_norm.weight")
        {
            return Ok(Some(TextEncoderModel::Gemma4E4b));
        }
        if has("model.layers.34.self_attn.q_norm.weight")
            && !has("model.layers.41.self_attn.q_norm.weight")
        {
            return Ok(Some(TextEncoderModel::Gemma4E2b));
        }
        if has("model.layers.47.self_attn.q_norm.weight") {
            return Ok(Some(TextEncoderModel::Gemma3_12b));
        }
        if has("model.layers.0.self_attn.q_norm.weight") {
            return Ok(Some(
                if has("vision_model.embeddings.patch_embedding.weight") {
                    TextEncoderModel::Gemma3_4bVision
                } else {
                    TextEncoderModel::Gemma3_4b
                },
            ));
        }
        return Ok(Some(TextEncoderModel::Gemma2_2b));
    }
    if has("layers.0.self_attn.sinks") && has("layers.0.mlp.experts.gate_up_proj.weight") {
        return Ok(Some(TextEncoderModel::GptOss20b));
    }
    match first("model.layers.0.self_attn.k_proj.bias")? {
        Some(256) => return Ok(Some(TextEncoderModel::Qwen25_3b)),
        Some(512) => return Ok(Some(TextEncoderModel::Qwen25_7b)),
        _ => {}
    }
    if has("model.language_model.layers.0.linear_attn.A_log")
        && has("model.language_model.layers.0.input_layernorm.weight")
    {
        return Ok(Some(
            match first("model.language_model.layers.0.input_layernorm.weight")? {
                Some(1_024) => TextEncoderModel::Qwen35_08b,
                Some(2_560) => TextEncoderModel::Qwen35_4b,
                Some(4_096) => TextEncoderModel::Qwen35_9b,
                Some(5_120) => TextEncoderModel::Qwen35_27b,
                _ => TextEncoderModel::Qwen35_2b,
            },
        ));
    }
    let deepstack_norm = format!("model.{HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT}.0.norm.weight");
    if has(&deepstack_norm) {
        let merger_width = first("model.visual.merger.linear_fc2.weight")?.ok_or_else(|| {
            ClipError::InvalidDetectorTensor("model.visual.merger.linear_fc2.weight".to_owned())
        })?;
        return Ok(Some(if merger_width == 2_560 {
            TextEncoderModel::Qwen3Vl4b
        } else {
            TextEncoderModel::Qwen3Vl8b
        }));
    }
    if let Some(width) = first("model.layers.0.post_attention_layernorm.weight")? {
        if has("model.layers.0.self_attn.q_norm.weight") {
            match width {
                2_560 => return Ok(Some(TextEncoderModel::Qwen3_4b)),
                2_048 => return Ok(Some(TextEncoderModel::Qwen3_2b)),
                4_096 => return Ok(Some(TextEncoderModel::Qwen3_8b)),
                1_024 => return Ok(Some(TextEncoderModel::Qwen3_06b)),
                _ => {}
            }
        }
        if width == 5_120 {
            return Ok(Some(
                if has("model.layers.39.post_attention_layernorm.weight") {
                    TextEncoderModel::Mistral3_24b
                } else {
                    TextEncoderModel::Mistral3_24bPrunedFlux2
                },
            ));
        }
        if width == 3_072 {
            return Ok(Some(TextEncoderModel::Ministral3_3b));
        }
        return Ok(Some(TextEncoderModel::Llama3_8));
    }
    Ok(None)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClipDetectedConfiguration {
    artifact_index: usize,
    weight_dtype: Option<String>,
    mixed_per_layer_quantization: bool,
}

impl ClipDetectedConfiguration {
    pub const fn artifact_index(&self) -> usize {
        self.artifact_index
    }

    pub fn weight_dtype(&self) -> Option<&str> {
        self.weight_dtype.as_deref()
    }

    pub const fn mixed_per_layer_quantization(&self) -> bool {
        self.mixed_per_layer_quantization
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClipArchitectureSelection {
    tokenizer: String,
    clip_model: String,
    text_encoder_models: Vec<Option<TextEncoderModel>>,
    t5xxl_configuration: Option<ClipDetectedConfiguration>,
    llama_configuration: Option<ClipDetectedConfiguration>,
    digest: String,
}

fn add_clip_resident_bytes(
    total: &mut u64,
    bytes: usize,
    field: &'static str,
) -> Result<(), ClipError> {
    *total = total
        .checked_add(u64::try_from(bytes).map_err(|_| ClipError::Overflow(field))?)
        .ok_or(ClipError::Overflow(field))?;
    Ok(())
}

impl ClipArchitectureSelection {
    pub fn tokenizer(&self) -> &str {
        &self.tokenizer
    }

    pub fn clip_model(&self) -> &str {
        &self.clip_model
    }

    pub fn text_encoder_models(&self) -> &[Option<TextEncoderModel>] {
        &self.text_encoder_models
    }

    pub fn t5xxl_configuration(&self) -> Option<&ClipDetectedConfiguration> {
        self.t5xxl_configuration.as_ref()
    }

    pub fn llama_configuration(&self) -> Option<&ClipDetectedConfiguration> {
        self.llama_configuration.as_ref()
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn resident_owned_bytes(&self) -> Result<u64, ClipError> {
        let mut bytes = 0_u64;
        for text in [&self.tokenizer, &self.clip_model, &self.digest] {
            add_clip_resident_bytes(&mut bytes, text.capacity(), "CLIP architecture strings")?;
        }
        add_clip_resident_bytes(
            &mut bytes,
            self.text_encoder_models
                .capacity()
                .checked_mul(std::mem::size_of::<Option<TextEncoderModel>>())
                .ok_or(ClipError::Overflow("CLIP architecture model list"))?,
            "CLIP architecture model list",
        )?;
        for configuration in [
            self.t5xxl_configuration.as_ref(),
            self.llama_configuration.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(dtype) = &configuration.weight_dtype {
                add_clip_resident_bytes(&mut bytes, dtype.capacity(), "CLIP architecture dtype")?;
            }
        }
        Ok(bytes)
    }
}

pub fn t5xxl_detect(probes: &[ModelProbe]) -> Result<Option<ClipDetectedConfiguration>, ClipError> {
    detect_encoder_configuration(
        probes,
        &[
            "encoder.block.23.layer.1.DenseReluDense.wi_1.weight",
            "encoder.block.23.layer.1.DenseReluDense.wi.weight",
        ],
        &["encoder.final_layer_norm.weight"],
    )
}

pub fn llama_detect(probes: &[ModelProbe]) -> Result<Option<ClipDetectedConfiguration>, ClipError> {
    detect_encoder_configuration(
        probes,
        &[
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.linear_attn.in_proj_a.weight",
        ],
        &["model.norm.weight", "model.layers.0.input_layernorm.weight"],
    )
}

fn detect_encoder_configuration(
    probes: &[ModelProbe],
    trigger_keys: &[&str],
    dtype_keys: &[&str],
) -> Result<Option<ClipDetectedConfiguration>, ClipError> {
    for (artifact_index, probe) in probes.iter().enumerate() {
        if !trigger_keys
            .iter()
            .any(|key| probe.tensor_shapes().contains_key(*key))
        {
            continue;
        }
        let weight_dtype = dtype_keys
            .iter()
            .find_map(|key| probe.storage_dtype(key))
            .map(ModelStorageDType::normalized_name);
        let mixed_per_layer_quantization = probe
            .tensor_shapes()
            .keys()
            .any(|key| key.ends_with(".sim_quant"));
        return Ok(Some(ClipDetectedConfiguration {
            artifact_index,
            weight_dtype,
            mixed_per_layer_quantization,
        }));
    }
    Ok(None)
}

pub fn select_clip_architecture(
    clip_type: ClipType,
    probes: &[ModelProbe],
) -> Result<ClipArchitectureSelection, ClipError> {
    if probes.is_empty() || probes.len() > MAX_CLIP_ARTIFACTS {
        return Err(ClipError::InvalidArtifactSet(probes.len()));
    }
    let text_encoder_models = probes
        .iter()
        .map(|probe| detect_text_encoder_model(probe.tensor_shapes()))
        .collect::<Result<Vec<_>, _>>()?;
    let (tokenizer, clip_model) = match probes.len() {
        1 => single_encoder_target(clip_type, text_encoder_models[0], &probes[0]),
        2 => dual_encoder_target(clip_type, &text_encoder_models),
        3 => (
            "sim.text_encoders.sd3_clip.SD3Tokenizer",
            "sim.text_encoders.sd3_clip.sd3_clip",
        ),
        4 => (
            "sim.text_encoders.hidream.HiDreamTokenizer",
            "sim.text_encoders.hidream.hidream_clip",
        ),
        count => return Err(ClipError::InvalidArtifactSet(count)),
    };
    let t5xxl_configuration = t5xxl_detect(probes)?;
    let llama_configuration = loader_llama_configuration(clip_type, &text_encoder_models, probes)?;
    let encoded = serde_json::to_vec(&(
        clip_type,
        &text_encoder_models,
        tokenizer,
        clip_model,
        &t5xxl_configuration,
        &llama_configuration,
    ))
    .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
    Ok(ClipArchitectureSelection {
        tokenizer: tokenizer.to_owned(),
        clip_model: clip_model.to_owned(),
        text_encoder_models,
        t5xxl_configuration,
        llama_configuration,
        digest: sha256(&encoded),
    })
}

fn loader_llama_configuration(
    clip_type: ClipType,
    text_encoder_models: &[Option<TextEncoderModel>],
    probes: &[ModelProbe],
) -> Result<Option<ClipDetectedConfiguration>, ClipError> {
    let rewrites_language_model_prefix = probes.len() == 1
        && matches!(
            text_encoder_models.first().copied().flatten(),
            Some(
                TextEncoderModel::Qwen35_08b
                    | TextEncoderModel::Qwen35_2b
                    | TextEncoderModel::Qwen35_4b
                    | TextEncoderModel::Qwen35_9b
                    | TextEncoderModel::Qwen35_27b
            )
        )
        || probes.len() == 1
            && !matches!(clip_type, ClipType::Flux | ClipType::Flux2)
            && matches!(
                text_encoder_models.first().copied().flatten(),
                Some(TextEncoderModel::Qwen3Vl4b | TextEncoderModel::Qwen3Vl8b)
            );
    if rewrites_language_model_prefix {
        detect_encoder_configuration(
            probes,
            &[
                "model.language_model.layers.0.self_attn.k_proj.weight",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            ],
            &[
                "model.language_model.norm.weight",
                "model.language_model.layers.0.input_layernorm.weight",
            ],
        )
    } else {
        llama_detect(probes)
    }
}

fn single_encoder_target(
    clip_type: ClipType,
    text_encoder_model: Option<TextEncoderModel>,
    probe: &ModelProbe,
) -> (&'static str, &'static str) {
    use ClipType as C;
    use TextEncoderModel as T;
    match (text_encoder_model, clip_type) {
        (Some(T::ClipG), C::StableCascade) => (
            "sim.sdxl_clip.StableCascadeTokenizer",
            "sim.sdxl_clip.StableCascadeClipModel",
        ),
        (Some(T::ClipG), C::Sd3) => (
            "sim.text_encoders.sd3_clip.SD3Tokenizer",
            "sim.text_encoders.sd3_clip.sd3_clip",
        ),
        (Some(T::ClipG), C::Hidream) => (
            "sim.text_encoders.hidream.HiDreamTokenizer",
            "sim.text_encoders.hidream.hidream_clip",
        ),
        (Some(T::ClipG), _) => (
            "sim.sdxl_clip.SDXLTokenizer",
            "sim.sdxl_clip.SDXLRefinerClipModel",
        ),
        (Some(T::ClipH), _) => (
            "sim.text_encoders.sd2_clip.SD2Tokenizer",
            "sim.text_encoders.sd2_clip.SD2ClipModel",
        ),
        (Some(T::T5Xxl), C::Sd3) => (
            "sim.text_encoders.sd3_clip.SD3Tokenizer",
            "sim.text_encoders.sd3_clip.sd3_clip",
        ),
        (Some(T::T5Xxl), C::Ltxv) => (
            "sim.text_encoders.lt.LTXVT5Tokenizer",
            "sim.text_encoders.lt.ltxv_te",
        ),
        (Some(T::T5Xxl), C::Pixart | C::Chroma) => (
            "sim.text_encoders.pixart_t5.PixArtTokenizer",
            "sim.text_encoders.pixart_t5.pixart_te",
        ),
        (Some(T::T5Xxl), C::Wan) => (
            "sim.text_encoders.wan.WanT5Tokenizer",
            "sim.text_encoders.wan.te",
        ),
        (Some(T::T5Xxl), C::Hidream) => (
            "sim.text_encoders.hidream.HiDreamTokenizer",
            "sim.text_encoders.hidream.hidream_clip",
        ),
        (Some(T::T5Xxl), C::Cogvideox) => (
            "sim.text_encoders.cogvideo.CogVideoXTokenizer",
            "sim.text_encoders.cogvideo.cogvideo_te",
        ),
        (Some(T::T5Xxl), _) => (
            "sim.text_encoders.genmo.MochiT5Tokenizer",
            "sim.text_encoders.genmo.mochi_te",
        ),
        (Some(T::T5XxlOld), _) => (
            "sim.text_encoders.cosmos.CosmosT5Tokenizer",
            "sim.text_encoders.cosmos.te",
        ),
        (Some(T::T5Xl), _) => (
            "sim.text_encoders.aura_t5.AuraT5Tokenizer",
            "sim.text_encoders.aura_t5.AuraT5Model",
        ),
        (Some(T::T5Base), C::Ace) => (
            "sim.text_encoders.ace.AceT5Tokenizer",
            "sim.text_encoders.ace.AceT5Model",
        ),
        (Some(T::T5Base), _) if probe.tensor_shapes().contains_key("spiece_model") => (
            "sim.text_encoders.ace.AceT5Tokenizer",
            "sim.text_encoders.ace.AceT5Model",
        ),
        (Some(T::T5Base), _) => (
            "sim.text_encoders.sa_t5.SAT5Tokenizer",
            "sim.text_encoders.sa_t5.SAT5Model",
        ),
        (Some(T::T5Gemma), _) => (
            "sim.text_encoders.sa3.SAT5GemmaTokenizer",
            "sim.text_encoders.sa3.SAT5GemmaModel",
        ),
        (Some(T::Gemma4E4b), _) => (
            "sim.text_encoders.gemma4.Gemma4_E4B.tokenizer",
            "sim.text_encoders.gemma4.gemma4_te",
        ),
        (Some(T::Gemma4E2b), _) => (
            "sim.text_encoders.gemma4.Gemma4_E2B.tokenizer",
            "sim.text_encoders.gemma4.gemma4_te",
        ),
        (Some(T::Gemma4_31b), _) => (
            "sim.text_encoders.gemma4.Gemma4_31B.tokenizer",
            "sim.text_encoders.gemma4.gemma4_te",
        ),
        (Some(T::Gemma2_2b), C::Pixeldit) => (
            "sim.text_encoders.pixeldit.PixelDiTGemma2Tokenizer",
            "sim.text_encoders.pixeldit.pixeldit_te",
        ),
        (Some(T::Gemma2_2b), _) => (
            "sim.text_encoders.lumina2.LuminaTokenizer",
            "sim.text_encoders.lumina2.te",
        ),
        (Some(T::Gemma3_4b | T::Gemma3_4bVision), _) => (
            "sim.text_encoders.lumina2.NTokenizer",
            "sim.text_encoders.lumina2.te",
        ),
        (Some(T::Gemma3_12b), _) => (
            "sim.text_encoders.lt.Gemma3_12BTokenizer",
            "sim.text_encoders.lt.gemma3_te",
        ),
        (Some(T::Llama3_8), _) => (
            "sim.text_encoders.hidream.HiDreamTokenizer",
            "sim.text_encoders.hidream.hidream_clip",
        ),
        (Some(T::Qwen25_3b), _) => (
            "sim.text_encoders.omnigen2.Omnigen2Tokenizer",
            "sim.text_encoders.omnigen2.te",
        ),
        (Some(T::Qwen25_7b), C::HunyuanImage) => (
            "sim.text_encoders.hunyuan_image.HunyuanImageTokenizer",
            "sim.text_encoders.hunyuan_image.te",
        ),
        (Some(T::Qwen25_7b), C::LongcatImage) => (
            "sim.text_encoders.longcat_image.LongCatImageTokenizer",
            "sim.text_encoders.longcat_image.te",
        ),
        (Some(T::Qwen25_7b), _) => (
            "sim.text_encoders.qwen_image.QwenImageTokenizer",
            "sim.text_encoders.qwen_image.te",
        ),
        (Some(T::Mistral3_24b | T::Mistral3_24bPrunedFlux2), _) => (
            "sim.text_encoders.flux.Flux2Tokenizer",
            "sim.text_encoders.flux.flux2_te",
        ),
        (Some(T::GptOss20b), _) => (
            "sim.text_encoders.gpt_oss.LensTokenizer",
            "sim.text_encoders.gpt_oss.lens_te",
        ),
        (Some(T::Qwen3_4b), C::Flux | C::Flux2) => (
            "sim.text_encoders.flux.KleinTokenizer",
            "sim.text_encoders.flux.klein_te",
        ),
        (Some(T::Qwen3_4b), _) => (
            "sim.text_encoders.z_image.ZImageTokenizer",
            "sim.text_encoders.z_image.te",
        ),
        (Some(T::Qwen3_2b), _) => (
            "sim.text_encoders.ovis.OvisTokenizer",
            "sim.text_encoders.ovis.te",
        ),
        (Some(T::Qwen3_8b), C::Ideogram4) => (
            "sim.text_encoders.ideogram4.Ideogram4Tokenizer",
            "sim.text_encoders.ideogram4.te",
        ),
        (Some(T::Qwen3_8b), _) => (
            "sim.text_encoders.flux.KleinTokenizer8B",
            "sim.text_encoders.flux.klein_te",
        ),
        (Some(T::JinaClip2), _) => (
            "sim.text_encoders.jina_clip_2.JinaClip2TokenizerWrapper",
            "sim.text_encoders.jina_clip_2.JinaClip2TextModelWrapper",
        ),
        (Some(T::Qwen35_08b | T::Qwen35_2b | T::Qwen35_4b | T::Qwen35_9b | T::Qwen35_27b), _) => (
            "sim.text_encoders.qwen35.tokenizer",
            "sim.text_encoders.qwen35.te",
        ),
        (Some(T::Qwen3Vl8b), C::Ideogram4) => (
            "sim.text_encoders.ideogram4.Ideogram4Qwen3VLTokenizer",
            "sim.text_encoders.ideogram4.te_qwen3vl",
        ),
        (Some(T::Qwen3Vl8b), C::Boogu) => (
            "sim.text_encoders.boogu.BooguTokenizer",
            "sim.text_encoders.boogu.te",
        ),
        (Some(T::Qwen3Vl4b), C::Krea2) => (
            "sim.text_encoders.krea2.Krea2Tokenizer",
            "sim.text_encoders.krea2.te",
        ),
        (Some(T::Qwen3Vl8b), C::Flux | C::Flux2) => (
            "sim.text_encoders.flux.KleinTokenizer8B",
            "sim.text_encoders.flux.klein_te",
        ),
        (Some(T::Qwen3Vl4b), C::Flux | C::Flux2) => (
            "sim.text_encoders.flux.KleinTokenizer",
            "sim.text_encoders.flux.klein_te",
        ),
        (Some(T::Qwen3Vl4b | T::Qwen3Vl8b), _) => (
            "sim.text_encoders.qwen3vl.tokenizer",
            "sim.text_encoders.qwen3vl.te",
        ),
        (Some(T::Qwen3_06b), _) => (
            "sim.text_encoders.anima.AnimaTokenizer",
            "sim.text_encoders.anima.te",
        ),
        (Some(T::Ministral3_3b), _) => (
            "sim.text_encoders.ernie.ErnieTokenizer",
            "sim.text_encoders.ernie.te",
        ),
        (_, C::Sd3) => (
            "sim.text_encoders.sd3_clip.SD3Tokenizer",
            "sim.text_encoders.sd3_clip.sd3_clip",
        ),
        (_, C::Hidream) => (
            "sim.text_encoders.hidream.HiDreamTokenizer",
            "sim.text_encoders.hidream.hidream_clip",
        ),
        _ => ("sim.sd1_clip.SD1Tokenizer", "sim.sd1_clip.SD1ClipModel"),
    }
}

fn dual_encoder_target(
    clip_type: ClipType,
    text_encoder_models: &[Option<TextEncoderModel>],
) -> (&'static str, &'static str) {
    use ClipType as C;
    match clip_type {
        C::Sd3 => (
            "sim.text_encoders.sd3_clip.SD3Tokenizer",
            "sim.text_encoders.sd3_clip.sd3_clip",
        ),
        C::HunyuanDit => (
            "sim.text_encoders.hydit.HyditTokenizer",
            "sim.text_encoders.hydit.HyditModel",
        ),
        C::Flux => (
            "sim.text_encoders.flux.FluxTokenizer",
            "sim.text_encoders.flux.flux_clip",
        ),
        C::HunyuanVideo => (
            "sim.text_encoders.hunyuan_video.HunyuanVideoTokenizer",
            "sim.text_encoders.hunyuan_video.hunyuan_video_clip",
        ),
        C::Hidream => (
            "sim.text_encoders.hidream.HiDreamTokenizer",
            "sim.text_encoders.hidream.hidream_clip",
        ),
        C::HunyuanImage => (
            "sim.text_encoders.hunyuan_image.HunyuanImageTokenizer",
            "sim.text_encoders.hunyuan_image.te",
        ),
        C::HunyuanVideo15 => (
            "sim.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer",
            "sim.text_encoders.hunyuan_image.te",
        ),
        C::Kandinsky5 => (
            "sim.text_encoders.kandinsky5.Kandinsky5Tokenizer",
            "sim.text_encoders.kandinsky5.te",
        ),
        C::Kandinsky5Image => (
            "sim.text_encoders.kandinsky5.Kandinsky5TokenizerImage",
            "sim.text_encoders.kandinsky5.te",
        ),
        C::Ltxv => (
            "sim.text_encoders.lt.LTXAVGemmaTokenizer",
            "sim.text_encoders.lt.ltxav_te",
        ),
        C::Newbie => (
            "sim.text_encoders.newbie.NewBieTokenizer",
            "sim.text_encoders.newbie.te",
        ),
        C::Ace => (
            "sim.text_encoders.ace15.ACE15Tokenizer",
            "sim.text_encoders.ace15.te",
        ),
        _ if text_encoder_models.len() == 2 => {
            ("sim.sdxl_clip.SDXLTokenizer", "sim.sdxl_clip.SDXLClipModel")
        }
        _ => unreachable!("dual target is called only for two artifacts"),
    }
}

pub fn clip_artifact_bundle_identity(
    store: &ModelStore,
    artifacts: &[Arc<LoadedModel>],
    cancellation: &CancellationToken,
) -> Result<String, ClipError> {
    cancellation.check().map_err(TensorError::from)?;
    if artifacts.is_empty() || artifacts.len() > MAX_CLIP_ARTIFACTS {
        return Err(ClipError::InvalidArtifactSet(artifacts.len()));
    }
    let mut digests = Vec::new();
    digests
        .try_reserve_exact(artifacts.len())
        .map_err(|_| ClipError::Allocation("CLIP artifact identities"))?;
    for artifact in artifacts {
        store.family_probe(artifact, cancellation)?;
        digests.push(artifact.identity().to_owned());
    }
    artifact_bundle_identity(&digests)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipNativeModuleKind {
    Linear {
        input_features: usize,
        output_features: usize,
    },
    Embedding {
        embeddings: usize,
        dimensions: usize,
    },
    LayerNorm {
        normalized_shape: Vec<usize>,
        epsilon_bits: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClipParameterSpec {
    artifact_index: usize,
    name: String,
    shape: Vec<u64>,
    storage_dtype: String,
}

impl ClipParameterSpec {
    #[allow(
        dead_code,
        reason = "registered CLIP architecture leaves are the only production constructors"
    )]
    pub(crate) fn checked(
        artifact_index: usize,
        name: impl Into<String>,
        shape: Vec<u64>,
        storage_dtype: impl Into<String>,
    ) -> Result<Self, ClipError> {
        let name = name.into();
        let storage_dtype = storage_dtype.into();
        if name.is_empty()
            || name.len() > 4_096
            || name.contains('\0')
            || shape.is_empty()
            || shape.contains(&0)
            || storage_dtype.is_empty()
            || storage_dtype.len() > 64
            || storage_dtype.contains('\0')
        {
            return Err(ClipError::InvalidParameterManifest(
                "parameter name, shape, or storage dtype is invalid".to_owned(),
            ));
        }
        Ok(Self {
            artifact_index,
            name,
            shape,
            storage_dtype,
        })
    }

    pub const fn artifact_index(&self) -> usize {
        self.artifact_index
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn storage_dtype(&self) -> &str {
        &self.storage_dtype
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClipNativeModuleSpec {
    name: String,
    artifact_index: usize,
    weight: String,
    bias: Option<String>,
    kind: ClipNativeModuleKind,
}

impl ClipNativeModuleSpec {
    #[allow(
        dead_code,
        reason = "registered CLIP architecture leaves are the only production constructors"
    )]
    pub(crate) fn checked(
        name: impl Into<String>,
        artifact_index: usize,
        weight: impl Into<String>,
        bias: Option<String>,
        kind: ClipNativeModuleKind,
    ) -> Result<Self, ClipError> {
        let name = name.into();
        let weight = weight.into();
        if name.is_empty()
            || name.len() > 4_096
            || name.contains('\0')
            || weight.is_empty()
            || weight.contains('\0')
            || bias
                .as_ref()
                .is_some_and(|value| value.is_empty() || value.contains('\0'))
        {
            return Err(ClipError::InvalidParameterManifest(
                "native module binding is invalid".to_owned(),
            ));
        }
        Ok(Self {
            name,
            artifact_index,
            weight,
            bias,
            kind,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipParameterManifest {
    architecture_digest: Option<String>,
    artifact_count: usize,
    parameters: Vec<ClipParameterSpec>,
    auxiliary: BTreeSet<(usize, String)>,
    modules: Vec<ClipNativeModuleSpec>,
    digest: String,
}

impl ClipParameterManifest {
    #[cfg(test)]
    fn checked(
        artifact_count: usize,
        parameters: Vec<ClipParameterSpec>,
        auxiliary: BTreeSet<(usize, String)>,
        modules: Vec<ClipNativeModuleSpec>,
    ) -> Result<Self, ClipError> {
        Self::checked_impl(None, artifact_count, parameters, auxiliary, modules)
    }

    #[allow(
        dead_code,
        reason = "registered CLIP architecture leaves bind their exact module graph here"
    )]
    pub(crate) fn from_model_store(
        architecture: &ClipArchitectureSelection,
        store: &ModelStore,
        artifacts: &[Arc<LoadedModel>],
        modules: Vec<ClipNativeModuleSpec>,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if artifacts.len() != architecture.text_encoder_models().len() {
            return Err(ClipError::ArtifactCount {
                expected: architecture.text_encoder_models().len(),
                actual: artifacts.len(),
            });
        }
        let mut probes = Vec::new();
        probes
            .try_reserve_exact(artifacts.len())
            .map_err(|_| ClipError::Allocation("CLIP manifest probes"))?;
        for artifact in artifacts {
            probes.push(store.family_probe(artifact, cancellation)?);
        }
        let detected = probes
            .iter()
            .map(|probe| detect_text_encoder_model(probe.tensor_shapes()))
            .collect::<Result<Vec<_>, _>>()?;
        if detected != architecture.text_encoder_models() {
            return Err(ClipError::ManifestDetectedArchitectureMismatch {
                expected: architecture.text_encoder_models().to_vec(),
                actual: detected,
            });
        }
        let mut parameter_keys = BTreeSet::new();
        for module in &modules {
            if module.artifact_index >= artifacts.len() {
                return Err(ClipError::InvalidParameterManifest(
                    "native module artifact index is outside the canonical bundle".to_owned(),
                ));
            }
            parameter_keys.insert((module.artifact_index, module.weight.clone()));
            if let Some(bias) = &module.bias {
                parameter_keys.insert((module.artifact_index, bias.clone()));
            }
        }
        if parameter_keys.is_empty() {
            return Err(ClipError::InvalidParameterManifest(
                "a canonical CLIP manifest must construct at least one NativeModule".to_owned(),
            ));
        }
        let mut parameters = Vec::new();
        parameters
            .try_reserve_exact(parameter_keys.len())
            .map_err(|_| ClipError::Allocation("CLIP manifest parameters"))?;
        for (artifact_index, name) in &parameter_keys {
            cancellation.check().map_err(TensorError::from)?;
            let probe = probes
                .get(*artifact_index)
                .ok_or(ClipError::Overflow("CLIP manifest artifact"))?;
            let shape = probe
                .tensor_shapes()
                .get(name)
                .cloned()
                .ok_or_else(|| ClipError::MissingManifestParameter(name.clone()))?;
            let storage_dtype = probe
                .storage_dtype(name)
                .map(ModelStorageDType::normalized_name)
                .ok_or_else(|| ClipError::MissingManifestParameter(name.clone()))?;
            parameters.push(ClipParameterSpec::checked(
                *artifact_index,
                name.clone(),
                shape,
                storage_dtype,
            )?);
        }
        let parameter_keys = &parameter_keys;
        let auxiliary = probes
            .iter()
            .enumerate()
            .flat_map(move |(artifact_index, probe)| {
                probe
                    .tensor_shapes()
                    .keys()
                    .filter(move |name| {
                        !parameter_keys.contains(&(artifact_index, (*name).clone()))
                    })
                    .map(move |name| (artifact_index, name.clone()))
            })
            .collect::<BTreeSet<_>>();
        Self::checked_impl(
            Some(architecture.digest().to_owned()),
            artifacts.len(),
            parameters,
            auxiliary,
            modules,
        )
    }

    #[allow(
        dead_code,
        reason = "registered CLIP architecture leaves reach this through ModelStore derivation"
    )]
    fn checked_impl(
        architecture_digest: Option<String>,
        artifact_count: usize,
        parameters: Vec<ClipParameterSpec>,
        auxiliary: BTreeSet<(usize, String)>,
        modules: Vec<ClipNativeModuleSpec>,
    ) -> Result<Self, ClipError> {
        if !(1..=MAX_CLIP_ARTIFACTS).contains(&artifact_count)
            || parameters.is_empty()
            || parameters.len() > MAX_CLIP_PARAMETERS
        {
            return Err(ClipError::InvalidParameterManifest(
                "artifact or parameter count is outside the checked bound".to_owned(),
            ));
        }
        let mut keys = BTreeSet::new();
        for parameter in &parameters {
            if parameter.artifact_index >= artifact_count
                || !keys.insert((parameter.artifact_index, parameter.name.clone()))
            {
                return Err(ClipError::InvalidParameterManifest(
                    "parameter is out of range or ambiguous".to_owned(),
                ));
            }
        }
        for (artifact_index, name) in &auxiliary {
            if *artifact_index >= artifact_count
                || name.is_empty()
                || name.contains('\0')
                || keys.contains(&(*artifact_index, name.clone()))
            {
                return Err(ClipError::InvalidParameterManifest(
                    "auxiliary tensor is invalid or overlaps a parameter".to_owned(),
                ));
            }
        }
        let by_key = parameters
            .iter()
            .map(|parameter| {
                (
                    (parameter.artifact_index, parameter.name.as_str()),
                    parameter,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut module_names = BTreeSet::new();
        for module in &modules {
            if module.artifact_index >= artifact_count || !module_names.insert(module.name.as_str())
            {
                return Err(ClipError::InvalidParameterManifest(
                    "native module name is duplicated or out of range".to_owned(),
                ));
            }
            let weight = by_key
                .get(&(module.artifact_index, module.weight.as_str()))
                .ok_or_else(|| ClipError::MissingManifestParameter(module.weight.clone()))?;
            let bias = module
                .bias
                .as_deref()
                .map(|name| {
                    by_key
                        .get(&(module.artifact_index, name))
                        .copied()
                        .ok_or_else(|| ClipError::MissingManifestParameter(name.to_owned()))
                })
                .transpose()?;
            validate_module_manifest(module, weight, bias)?;
        }
        let encoded = serde_json::to_vec(&(
            &architecture_digest,
            artifact_count,
            &parameters,
            &auxiliary,
            &modules,
        ))
        .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        Ok(Self {
            architecture_digest,
            artifact_count,
            parameters,
            auxiliary,
            modules,
            digest: sha256(&encoded),
        })
    }

    pub const fn artifact_count(&self) -> usize {
        self.artifact_count
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn architecture_digest(&self) -> Option<&str> {
        self.architecture_digest.as_deref()
    }

    pub fn validate_architecture(
        &self,
        architecture: &ClipArchitectureSelection,
    ) -> Result<(), ClipError> {
        if self.architecture_digest() != Some(architecture.digest()) {
            return Err(ClipError::ManifestArchitectureMismatch {
                expected: architecture.digest().to_owned(),
                actual: self.architecture_digest().map(str::to_owned),
            });
        }
        Ok(())
    }

    pub fn validate_probes(&self, probes: &[ModelProbe]) -> Result<(), ClipError> {
        self.validate_probes_cancellable(probes, &CancellationToken::default())
    }

    fn validate_probes_cancellable(
        &self,
        probes: &[ModelProbe],
        cancellation: &CancellationToken,
    ) -> Result<(), ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if probes.len() != self.artifact_count {
            return Err(ClipError::ArtifactCount {
                expected: self.artifact_count,
                actual: probes.len(),
            });
        }
        for (artifact_index, probe) in probes.iter().enumerate() {
            cancellation.check().map_err(TensorError::from)?;
            let expected = self
                .parameters
                .iter()
                .filter(|parameter| parameter.artifact_index == artifact_index)
                .map(|parameter| parameter.name.as_str())
                .chain(
                    self.auxiliary
                        .iter()
                        .filter(|(index, _)| *index == artifact_index)
                        .map(|(_, name)| name.as_str()),
                )
                .collect::<BTreeSet<_>>();
            let actual = probe
                .tensor_shapes()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if expected != actual {
                return Err(ClipError::ParameterSetMismatch {
                    artifact_index,
                    missing: expected
                        .difference(&actual)
                        .map(|value| (*value).to_owned())
                        .collect(),
                    unexpected: actual
                        .difference(&expected)
                        .map(|value| (*value).to_owned())
                        .collect(),
                });
            }
            for parameter in self
                .parameters
                .iter()
                .filter(|parameter| parameter.artifact_index == artifact_index)
            {
                cancellation.check().map_err(TensorError::from)?;
                let actual_shape = probe
                    .tensor_shapes()
                    .get(parameter.name())
                    .ok_or_else(|| ClipError::MissingManifestParameter(parameter.name.clone()))?;
                if actual_shape != parameter.shape() {
                    return Err(ClipError::ManifestParameterShape {
                        name: parameter.name.clone(),
                        expected: parameter.shape.clone(),
                        actual: actual_shape.clone(),
                    });
                }
                let actual_dtype = probe
                    .storage_dtype(parameter.name())
                    .map(|dtype| dtype.normalized_name())
                    .ok_or_else(|| ClipError::MissingManifestParameter(parameter.name.clone()))?;
                if actual_dtype != parameter.storage_dtype {
                    return Err(ClipError::ManifestParameterDType {
                        name: parameter.name.clone(),
                        expected: parameter.storage_dtype.clone(),
                        actual: actual_dtype,
                    });
                }
            }
        }
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "registered CLIP architecture leaves publish modules through the atomic loader"
    )]
    pub(crate) fn project_native_modules(
        &self,
        tensors: &[BTreeMap<String, Tensor>],
        dtype: DType,
        device: DeviceId,
    ) -> Result<BTreeMap<String, NativeModule>, ClipError> {
        if tensors.len() != self.artifact_count {
            return Err(ClipError::ArtifactCount {
                expected: self.artifact_count,
                actual: tensors.len(),
            });
        }
        for parameter in &self.parameters {
            let tensor = tensors
                .get(parameter.artifact_index)
                .and_then(|state| state.get(parameter.name()))
                .ok_or_else(|| ClipError::MissingManifestParameter(parameter.name.clone()))?;
            if tensor.descriptor().shape() != parameter.shape
                || tensor.descriptor().dtype() != dtype
                || tensor.descriptor().device() != device
            {
                return Err(ClipError::NativeProjectionMismatch(parameter.name.clone()));
            }
        }
        let expected = self
            .parameters
            .iter()
            .map(|parameter| (parameter.artifact_index, parameter.name.as_str()))
            .collect::<BTreeSet<_>>();
        let actual = tensors
            .iter()
            .enumerate()
            .flat_map(|(artifact_index, state)| {
                state
                    .keys()
                    .map(move |name| (artifact_index, name.as_str()))
            })
            .collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(ClipError::NativeProjectionSetMismatch);
        }
        let mut projected = BTreeMap::new();
        for module in &self.modules {
            let state = tensors
                .get(module.artifact_index)
                .ok_or(ClipError::Overflow("native module artifact"))?;
            let weight = state
                .get(&module.weight)
                .cloned()
                .ok_or_else(|| ClipError::MissingManifestParameter(module.weight.clone()))?;
            let bias = module
                .bias
                .as_ref()
                .map(|name| {
                    state
                        .get(name)
                        .cloned()
                        .ok_or_else(|| ClipError::MissingManifestParameter(name.clone()))
                })
                .transpose()?;
            let mut native = native_module_from_spec(module)?;
            native.load_dense_parameters(weight, bias)?;
            projected.insert(module.name.clone(), native);
        }
        Ok(projected)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipLoadDomain {
    family: ModelFamilyIdentity,
    target: ModelClipTargetDescriptor,
    selected: ModelClipTargetCandidateDescriptor,
    architecture: ClipArchitectureSelection,
    clip_type: ClipType,
    text_encoder_models: Vec<Option<TextEncoderModel>>,
    artifact_identity: ClipBindingIdentity,
    model_identity: ClipBindingIdentity,
    patch_identity: ClipBindingIdentity,
    manifest_identity: String,
    dtype: DType,
    device: DeviceId,
    digest: String,
}

impl ClipLoadDomain {
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_store(
        family: &ResolvedModelFamily,
        clip_type: ClipType,
        store: &ModelStore,
        artifacts: &[Arc<LoadedModel>],
        patch_graph: &PatchGraph,
        manifest: &ClipParameterManifest,
        dtype: DType,
        device: DeviceId,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if artifacts.len() != manifest.artifact_count() {
            return Err(ClipError::ArtifactCount {
                expected: manifest.artifact_count(),
                actual: artifacts.len(),
            });
        }
        let mut probes = Vec::new();
        probes
            .try_reserve_exact(artifacts.len())
            .map_err(|_| ClipError::Allocation("CLIP model probes"))?;
        for artifact in artifacts {
            probes.push(store.family_probe(artifact, cancellation)?);
        }
        manifest.validate_probes_cancellable(&probes, cancellation)?;
        let architecture = select_clip_architecture(clip_type, &probes)?;
        manifest.validate_architecture(&architecture)?;
        let artifact_digests = artifacts
            .iter()
            .map(|artifact| artifact.identity().to_owned())
            .collect::<Vec<_>>();
        Self::checked_facts(
            family.detection().identity.clone(),
            family.clip_target().clone(),
            clip_type,
            architecture,
            artifact_digests,
            patch_graph.identity(),
            manifest.digest(),
            dtype,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn checked_facts(
        family: ModelFamilyIdentity,
        target: ModelClipTargetDescriptor,
        clip_type: ClipType,
        architecture: ClipArchitectureSelection,
        artifact_digests: Vec<String>,
        patch: PatchGraphIdentity,
        manifest_identity: &str,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Self, ClipError> {
        if artifact_digests.is_empty()
            || artifact_digests.len() > MAX_CLIP_ARTIFACTS
            || artifact_digests.len() != architecture.text_encoder_models().len()
        {
            return Err(ClipError::InvalidArtifactSet(artifact_digests.len()));
        }
        validate_sha256("manifest", manifest_identity)?;
        let artifact_bundle = artifact_bundle_identity(&artifact_digests)?;
        match patch.validate_for_base(&artifact_bundle) {
            Ok(()) => {}
            Err(PatchGraphIdentityError::BaseDigestMismatch { expected, actual }) => {
                return Err(ClipError::PatchBaseMismatch { expected, actual });
            }
            Err(error) => return Err(ClipError::PatchIdentity(error)),
        }
        if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64) {
            return Err(ClipError::InvalidExecutionDType(dtype));
        }
        let selected = target
            .candidates()
            .iter()
            .find(|candidate| canonical_candidate_matches(candidate, &architecture))
            .cloned()
            .ok_or_else(|| ClipError::CanonicalTargetMismatch {
                tokenizer: architecture.tokenizer().to_owned(),
                clip_model: architecture.clip_model().to_owned(),
            })?;
        let family_json = serde_json::to_vec(&family)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let target_json = serde_json::to_vec(&target)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let selected_json = serde_json::to_vec(&selected)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let clip_type_json = serde_json::to_vec(&clip_type)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let architecture_json = serde_json::to_vec(&architecture)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let dtype_json = serde_json::to_vec(&dtype)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let device_json = serde_json::to_vec(&device)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let digest = digest_fields(&[
            &family_json,
            &target_json,
            &selected_json,
            &clip_type_json,
            &architecture_json,
            artifact_bundle.as_bytes(),
            patch.ordered_digest.as_bytes(),
            manifest_identity.as_bytes(),
            &dtype_json,
            &device_json,
        ]);
        Ok(Self {
            family,
            target,
            selected,
            text_encoder_models: architecture.text_encoder_models().to_vec(),
            architecture,
            clip_type,
            artifact_identity: ClipBindingIdentity::derived("artifact", artifact_bundle),
            model_identity: ClipBindingIdentity::derived("model", digest.clone()),
            patch_identity: ClipBindingIdentity::derived("patch", patch.ordered_digest),
            manifest_identity: manifest_identity.to_owned(),
            dtype,
            device,
            digest,
        })
    }

    pub fn family(&self) -> &ModelFamilyIdentity {
        &self.family
    }

    pub fn target(&self) -> &ModelClipTargetDescriptor {
        &self.target
    }

    pub fn selected(&self) -> &ModelClipTargetCandidateDescriptor {
        &self.selected
    }

    pub fn architecture(&self) -> &ClipArchitectureSelection {
        &self.architecture
    }

    pub const fn clip_type(&self) -> ClipType {
        self.clip_type
    }

    pub fn text_encoder_models(&self) -> &[Option<TextEncoderModel>] {
        &self.text_encoder_models
    }

    pub fn manifest_identity(&self) -> &str {
        &self.manifest_identity
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn device(&self) -> DeviceId {
        self.device
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

fn canonical_candidate_matches(
    candidate: &ModelClipTargetCandidateDescriptor,
    architecture: &ClipArchitectureSelection,
) -> bool {
    let tokenizer = candidate.tokenizer().identifier();
    let clip_model = candidate.clip_model().target().as_str();
    if tokenizer == architecture.tokenizer() && clip_model == architecture.clip_model() {
        return true;
    }
    if architecture.tokenizer() == "sim.sd1_clip.SD1Tokenizer"
        && architecture.clip_model() == "sim.sd1_clip.SD1ClipModel"
        && tokenizer == "sim.sd1.tokenizer"
        && clip_model == "sim.sd1.clip"
    {
        return true;
    }
    architecture.tokenizer() == "sim.text_encoders.cogvideo.CogVideoXTokenizer"
        && architecture.clip_model() == "sim.text_encoders.cogvideo.cogvideo_te"
        && tokenizer == "sim.text_encoders.cogvideo.CogVideoXT5Tokenizer"
        && clip_model == "sim.text_encoders.sd3_clip.T5XXLModel"
}

#[derive(Debug)]
pub struct LoadedClipArchitecture {
    domain: ClipLoadDomain,
    modules: BTreeMap<String, NativeModule>,
}

fn clip_architecture_execution_requirements(dtype: DType) -> NativeExecutionRequirements {
    let mut requirements = NativeExecutionRequirements::new();
    requirements.extend([
        OperationSupport::allocation(dtype, Layout::Contiguous),
        OperationSupport::copy_input(dtype, Layout::Contiguous),
        OperationSupport::copy_output(dtype, Layout::Contiguous),
        OperationSupport::resize_input(ResizeMode::Bilinear, dtype, Layout::Contiguous),
        OperationSupport::resize_output(ResizeMode::Bilinear, dtype, Layout::Contiguous),
        OperationSupport::resize_input(ResizeMode::Bicubic, dtype, Layout::Contiguous),
        OperationSupport::resize_output(ResizeMode::Bicubic, dtype, Layout::Contiguous),
        OperationSupport::convolution_input(dtype, Layout::Contiguous),
        OperationSupport::convolution_output(dtype, Layout::Contiguous),
        OperationSupport::binary_input(BinaryOperation::Add, dtype, Layout::Contiguous),
        OperationSupport::binary_output(BinaryOperation::Add, dtype, Layout::Contiguous),
        OperationSupport::linear_algebra_input(
            LinearAlgebraOperation::BatchMatrixMultiply,
            dtype,
            Layout::Contiguous,
        ),
        OperationSupport::linear_algebra_output(
            LinearAlgebraOperation::BatchMatrixMultiply,
            dtype,
            Layout::Contiguous,
        ),
    ]);
    for operation in [
        UnaryOperation::Exponential,
        UnaryOperation::Sigmoid,
        UnaryOperation::HyperbolicTangent,
        UnaryOperation::SquareRoot,
    ] {
        requirements.extend([
            OperationSupport::unary_input(operation, dtype, Layout::Contiguous),
            OperationSupport::unary_output(operation, dtype, Layout::Contiguous),
        ]);
    }
    for operation in [
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
    ] {
        requirements.extend([
            OperationSupport::binary_input(operation, dtype, Layout::Contiguous),
            OperationSupport::binary_output(operation, dtype, Layout::Contiguous),
        ]);
    }
    for operation in [ReductionOperation::Mean, ReductionOperation::Variance] {
        requirements.extend([
            OperationSupport::reduction_input(operation, dtype, Layout::Contiguous),
            OperationSupport::reduction_output(operation, dtype, Layout::Contiguous),
        ]);
    }
    requirements.extend([
        OperationSupport::record_event(),
        OperationSupport::wait_event(),
    ]);
    requirements
}

impl LoadedClipArchitecture {
    #[allow(clippy::too_many_arguments)]
    #[allow(
        dead_code,
        reason = "registered CLIP architecture leaves are the only production callers"
    )]
    pub(crate) fn from_model_store(
        family: &ResolvedModelFamily,
        clip_type: ClipType,
        store: &ModelStore,
        index: &ArtifactIndex,
        artifacts: &[Arc<LoadedModel>],
        patch_graph: &PatchGraph,
        manifest: &ClipParameterManifest,
        backend: &CpuBackend,
        dtype: DType,
        device: DeviceId,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        clip_architecture_execution_requirements(dtype).admit_backend_target(
            backend,
            device,
            dtype,
            Layout::Contiguous,
            context.stream,
            context,
        )?;
        let domain = ClipLoadDomain::from_model_store(
            family,
            clip_type,
            store,
            artifacts,
            patch_graph,
            manifest,
            dtype,
            device,
            context.cancellation,
        )?;
        let modules = load_manifest_modules(
            store, index, artifacts, manifest, backend, dtype, device, context,
        )?;
        Ok(Self { domain, modules })
    }

    pub fn domain(&self) -> &ClipLoadDomain {
        &self.domain
    }

    pub fn modules(&self) -> &BTreeMap<String, NativeModule> {
        &self.modules
    }

    pub fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = clip_architecture_execution_requirements(dtype);
        for module in self.modules.values() {
            requirements.extend(module.execution_requirements(dtype).iter());
        }
        requirements
    }
}

#[allow(
    dead_code,
    reason = "registered CLIP architecture leaves use this single failure-atomic load path"
)]
fn load_manifest_modules(
    store: &ModelStore,
    index: &ArtifactIndex,
    artifacts: &[Arc<LoadedModel>],
    manifest: &ClipParameterManifest,
    backend: &CpuBackend,
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, NativeModule>, ClipError> {
    clip_architecture_execution_requirements(dtype).admit_backend_target(
        backend,
        device,
        dtype,
        Layout::Contiguous,
        context.stream,
        context,
    )?;
    if artifacts.len() != manifest.artifact_count() {
        return Err(ClipError::ArtifactCount {
            expected: manifest.artifact_count(),
            actual: artifacts.len(),
        });
    }
    let mut states = Vec::new();
    states
        .try_reserve_exact(artifacts.len())
        .map_err(|_| ClipError::Allocation("CLIP native state dictionaries"))?;
    for (artifact_index, artifact) in artifacts.iter().enumerate() {
        context.check()?;
        let parameters = manifest
            .parameters
            .iter()
            .filter(|parameter| parameter.artifact_index == artifact_index)
            .collect::<Vec<_>>();
        let names = parameters
            .iter()
            .map(|parameter| parameter.name())
            .collect::<Vec<_>>();
        let source = store.read_tensors(index, artifact, names, context.cancellation)?;
        let mut state = BTreeMap::new();
        for parameter in parameters {
            context.check()?;
            let parameter_dtype = manifest_parameter_dtype(parameter)?;
            if parameter_dtype != dtype {
                return Err(ClipError::LoadDTypeMismatch {
                    name: parameter.name.clone(),
                    storage: parameter_dtype,
                    requested: dtype,
                });
            }
            let bytes = source
                .get(parameter.name())
                .ok_or_else(|| ClipError::MissingManifestParameter(parameter.name.clone()))?;
            let descriptor = TensorDescriptor::contiguous(
                parameter.shape.clone(),
                parameter_dtype,
                device,
                context.stream,
            )?;
            let (tensor, _) = backend.upload_bytes(descriptor, bytes, context)?;
            state.insert(parameter.name.clone(), tensor);
        }
        states.push(state);
    }
    context.check()?;
    let modules = manifest.project_native_modules(&states, dtype, device)?;
    context.check()?;
    Ok(modules)
}

#[allow(
    dead_code,
    reason = "registered CLIP architecture leaves use this through the atomic load path"
)]
fn manifest_parameter_dtype(parameter: &ClipParameterSpec) -> Result<DType, ClipError> {
    match parameter.storage_dtype() {
        "bool" => Ok(DType::Bool),
        "uint8" => Ok(DType::U8),
        "int8" => Ok(DType::I8),
        "uint16" => Ok(DType::U16),
        "int16" => Ok(DType::I16),
        "float16" => Ok(DType::F16),
        "bfloat16" => Ok(DType::Bf16),
        "uint32" => Ok(DType::U32),
        "int32" => Ok(DType::I32),
        "float32" => Ok(DType::F32),
        "float8_e4m3fn" => Ok(DType::Float8E4m3Fn),
        "float8_e5m2" => Ok(DType::Float8E5m2),
        "float8_e4m3fnuz" => Ok(DType::Float8E4m3Fnuz),
        "float8_e5m2fnuz" => Ok(DType::Float8E5m2Fnuz),
        "float8_e8m0fnu" => Ok(DType::Float8E8m0Fnu),
        "uint64" => Ok(DType::U64),
        "int64" => Ok(DType::I64),
        "float64" => Ok(DType::F64),
        "complex64" => Ok(DType::Complex64),
        "complex128" => Ok(DType::Complex128),
        _ => Err(ClipError::UnsupportedManifestStorage {
            name: parameter.name.clone(),
            storage_dtype: parameter.storage_dtype.clone(),
        }),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerIdentity {
    descriptor: ModelTokenizerDescriptor,
    vocabulary_sha256: String,
    merges_sha256: String,
    digest: String,
}

impl TokenizerIdentity {
    pub fn checked(
        descriptor: ModelTokenizerDescriptor,
        vocabulary: &[u8],
        merges: &[u8],
    ) -> Result<Self, ClipError> {
        if vocabulary.is_empty() || merges.is_empty() {
            return Err(ClipError::InvalidTokenizerIdentity(
                "tokenizer vocabulary and merges must both be non-empty".to_owned(),
            ));
        }
        let vocabulary_sha256 = sha256(vocabulary);
        let merges_sha256 = sha256(merges);
        let digest = digest_fields(&[
            descriptor.identifier().as_bytes(),
            vocabulary_sha256.as_bytes(),
            merges_sha256.as_bytes(),
        ]);
        Ok(Self {
            descriptor,
            vocabulary_sha256,
            merges_sha256,
            digest,
        })
    }

    pub fn descriptor(&self) -> &ModelTokenizerDescriptor {
        &self.descriptor
    }

    pub fn vocabulary_sha256(&self) -> &str {
        &self.vocabulary_sha256
    }

    pub fn merges_sha256(&self) -> &str {
        &self.merges_sha256
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn resident_owned_bytes(&self) -> Result<u64, ClipError> {
        let mut bytes = self
            .descriptor
            .owned_resident_bytes()
            .ok_or(ClipError::Overflow("tokenizer identity"))?;
        for text in [&self.vocabulary_sha256, &self.merges_sha256, &self.digest] {
            add_clip_resident_bytes(&mut bytes, text.capacity(), "tokenizer identity")?;
        }
        Ok(bytes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedText {
    text: String,
    weight: f32,
}

impl WeightedText {
    pub fn checked(text: impl Into<String>, weight: f32) -> Result<Self, ClipError> {
        let text = text.into();
        if !weight.is_finite() {
            return Err(ClipError::InvalidWeight(weight));
        }
        Ok(Self { text, weight })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn weight(&self) -> f32 {
        self.weight
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedToken {
    token: u32,
    weight: f32,
}

impl WeightedToken {
    pub fn checked(token: u32, weight: f32) -> Result<Self, ClipError> {
        if !weight.is_finite() {
            return Err(ClipError::InvalidWeight(weight));
        }
        Ok(Self { token, weight })
    }

    pub const fn token(self) -> u32 {
        self.token
    }

    pub const fn weight(self) -> f32 {
        self.weight
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenSequence {
    tokens: Vec<WeightedToken>,
    content_tokens: usize,
}

impl TokenSequence {
    pub fn checked(
        tokens: Vec<WeightedToken>,
        content_tokens: usize,
        context_length: usize,
        end_token: u32,
    ) -> Result<Self, ClipError> {
        if context_length < 2 || tokens.len() != context_length {
            return Err(ClipError::TokenShape {
                expected: context_length,
                actual: tokens.len(),
            });
        }
        if content_tokens > context_length.saturating_sub(2) {
            return Err(ClipError::InvalidContentLength(content_tokens));
        }
        let end_index = content_tokens
            .checked_add(1)
            .ok_or(ClipError::Overflow("end token index"))?;
        if tokens.get(end_index).map(|token| token.token()) != Some(end_token)
            || tokens
                .get(end_index..)
                .ok_or(ClipError::Overflow("end token suffix"))?
                .iter()
                .any(|token| token.token() != end_token)
        {
            return Err(ClipError::InvalidPadding);
        }
        Ok(Self {
            tokens,
            content_tokens,
        })
    }

    pub fn tokens(&self) -> &[WeightedToken] {
        &self.tokens
    }

    pub const fn content_tokens(&self) -> usize {
        self.content_tokens
    }

    pub const fn first_end_index(&self) -> usize {
        self.content_tokens + 1
    }

    pub fn sd1_token_ids(&self) -> Result<[u32; SD1_CONTEXT_LENGTH], ClipError> {
        let tokens: &[WeightedToken; SD1_CONTEXT_LENGTH] = self
            .tokens
            .as_slice()
            .try_into()
            .map_err(|_| ClipError::TokenShape {
                expected: SD1_CONTEXT_LENGTH,
                actual: self.tokens.len(),
            })?;
        Ok(tokens.map(WeightedToken::token))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenBatch {
    tokenizer_identity: TokenizerIdentity,
    rows: Vec<TokenSequence>,
    context_length: usize,
    end_token: u32,
    digest: String,
}

impl TokenBatch {
    pub fn checked(
        tokenizer_identity: TokenizerIdentity,
        rows: Vec<TokenSequence>,
        context_length: usize,
        end_token: u32,
    ) -> Result<Self, ClipError> {
        if rows.is_empty() || rows.len() > MAX_TOKEN_BATCH {
            return Err(ClipError::InvalidBatchSize(rows.len()));
        }
        for row in &rows {
            if row.tokens().len() != context_length
                || row
                    .tokens()
                    .get(row.first_end_index())
                    .map(|token| token.token())
                    != Some(end_token)
            {
                return Err(ClipError::InvalidPadding);
            }
        }
        let mut hasher = Sha256::new();
        update_digest_field(&mut hasher, tokenizer_identity.digest().as_bytes());
        update_digest_field(&mut hasher, &u64_bytes(context_length)?);
        update_digest_field(&mut hasher, &end_token.to_le_bytes());
        for row in &rows {
            update_digest_field(&mut hasher, &u64_bytes(row.content_tokens())?);
            for token in row.tokens() {
                update_digest_field(&mut hasher, &token.token().to_le_bytes());
                update_digest_field(&mut hasher, &token.weight().to_bits().to_le_bytes());
            }
        }
        Ok(Self {
            tokenizer_identity,
            rows,
            context_length,
            end_token,
            digest: format!("{:x}", hasher.finalize()),
        })
    }

    pub fn tokenizer_identity(&self) -> &TokenizerIdentity {
        &self.tokenizer_identity
    }

    pub fn rows(&self) -> &[TokenSequence] {
        &self.rows
    }

    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    pub const fn end_token(&self) -> u32 {
        self.end_token
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn causal_attention_mask(&self) -> Result<Vec<bool>, ClipError> {
        let count = self
            .context_length
            .checked_mul(self.context_length)
            .ok_or(ClipError::Overflow("causal attention mask"))?;
        let mut mask = Vec::new();
        mask.try_reserve_exact(count)
            .map_err(|_| ClipError::Allocation("causal attention mask"))?;
        for query in 0..self.context_length {
            for key in 0..self.context_length {
                mask.push(key <= query);
            }
        }
        Ok(mask)
    }
}

pub trait NativeTokenizer: Send + Sync {
    fn identity(&self) -> &TokenizerIdentity;

    fn tokenize_batch(
        &self,
        prompts: &[Vec<WeightedText>],
        cancellation: &CancellationToken,
    ) -> Result<TokenBatch, ClipError>;
}

#[derive(Clone, Debug)]
pub struct Sd1Tokenizer {
    identity: TokenizerIdentity,
    vocabulary: BTreeMap<String, u32>,
    inverse_vocabulary: BTreeMap<u32, String>,
    merge_ranks: BTreeMap<(String, String), usize>,
    byte_encoder: BTreeMap<u8, char>,
    byte_decoder: BTreeMap<char, u8>,
    token_pattern: Regex,
}

impl Sd1Tokenizer {
    pub fn from_json_and_merges(
        descriptor: ModelTokenizerDescriptor,
        vocabulary_json: &str,
        merges: &str,
    ) -> Result<Self, ClipError> {
        let vocabulary: BTreeMap<String, u32> = serde_json::from_str(vocabulary_json)
            .map_err(|error| ClipError::Tokenizer(error.to_string()))?;
        if vocabulary.len() != SD1_VOCABULARY_SIZE
            || vocabulary.get("<|startoftext|>") != Some(&SD1_START_TOKEN)
            || vocabulary.get("<|endoftext|>") != Some(&SD1_END_TOKEN)
        {
            return Err(ClipError::Tokenizer(
                "SD1 vocabulary identity is invalid".to_owned(),
            ));
        }
        let mut inverse_vocabulary = BTreeMap::new();
        for (piece, token) in &vocabulary {
            if inverse_vocabulary.insert(*token, piece.clone()).is_some() {
                return Err(ClipError::Tokenizer(
                    "SD1 vocabulary contains a duplicate token ID".to_owned(),
                ));
            }
        }
        if inverse_vocabulary
            .keys()
            .copied()
            .ne(0..u32::try_from(SD1_VOCABULARY_SIZE)
                .map_err(|_| ClipError::Overflow("SD1 vocabulary size"))?)
        {
            return Err(ClipError::Tokenizer(
                "SD1 vocabulary token IDs are not the contiguous canonical domain".to_owned(),
            ));
        }
        let mut merge_ranks = BTreeMap::new();
        let mut merge_lines = merges.lines();
        if merge_lines.next() != Some("#version: 0.2") {
            return Err(ClipError::Tokenizer(
                "SD1 merges are missing the canonical version header".to_owned(),
            ));
        }
        for line in merge_lines.filter(|line| !line.is_empty()) {
            let mut parts = line.split_whitespace();
            let left = parts
                .next()
                .ok_or_else(|| ClipError::Tokenizer("invalid merge".to_owned()))?;
            let right = parts
                .next()
                .ok_or_else(|| ClipError::Tokenizer("invalid merge".to_owned()))?;
            if parts.next().is_some() {
                return Err(ClipError::Tokenizer("invalid merge".to_owned()));
            }
            let rank = merge_ranks.len();
            if merge_ranks
                .insert((left.to_owned(), right.to_owned()), rank)
                .is_some()
            {
                return Err(ClipError::Tokenizer(
                    "SD1 merges contain a duplicate pair".to_owned(),
                ));
            }
        }
        if merge_ranks.len() != SD1_MERGE_COUNT {
            return Err(ClipError::Tokenizer(format!(
                "SD1 merge identity expected {SD1_MERGE_COUNT} pairs, got {}",
                merge_ranks.len()
            )));
        }
        let token_pattern = Regex::new(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )
        .map_err(|error| ClipError::Tokenizer(error.to_string()))?;
        let identity =
            TokenizerIdentity::checked(descriptor, vocabulary_json.as_bytes(), merges.as_bytes())?;
        let byte_encoder = byte_encoder()?;
        let byte_decoder = byte_encoder
            .iter()
            .map(|(byte, character)| (*character, *byte))
            .collect();
        Ok(Self {
            identity,
            vocabulary,
            inverse_vocabulary,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            token_pattern,
        })
    }

    pub fn resident_bytes(&self) -> Result<u64, ClipError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| ClipError::Overflow("tokenizer resident bytes"))?;
        for piece in self.vocabulary.keys() {
            add_clip_resident_bytes(
                &mut bytes,
                std::mem::size_of::<(String, u32)>(),
                "tokenizer vocabulary entry",
            )?;
            add_clip_resident_bytes(&mut bytes, piece.capacity(), "tokenizer vocabulary text")?;
        }
        for piece in self.inverse_vocabulary.values() {
            add_clip_resident_bytes(
                &mut bytes,
                std::mem::size_of::<(u32, String)>(),
                "inverse tokenizer vocabulary entry",
            )?;
            add_clip_resident_bytes(
                &mut bytes,
                piece.capacity(),
                "inverse tokenizer vocabulary text",
            )?;
        }
        for (left, right) in self.merge_ranks.keys() {
            add_clip_resident_bytes(
                &mut bytes,
                std::mem::size_of::<((String, String), usize)>(),
                "tokenizer merge entry",
            )?;
            add_clip_resident_bytes(&mut bytes, left.capacity(), "tokenizer merge text")?;
            add_clip_resident_bytes(&mut bytes, right.capacity(), "tokenizer merge text")?;
        }
        add_clip_resident_bytes(
            &mut bytes,
            self.byte_encoder
                .len()
                .checked_mul(std::mem::size_of::<(u8, char)>())
                .ok_or(ClipError::Overflow("tokenizer byte encoder"))?,
            "tokenizer byte encoder",
        )?;
        add_clip_resident_bytes(
            &mut bytes,
            self.byte_decoder
                .len()
                .checked_mul(std::mem::size_of::<(char, u8)>())
                .ok_or(ClipError::Overflow("tokenizer byte decoder"))?,
            "tokenizer byte decoder",
        )?;
        Ok(bytes)
    }

    pub fn encode(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<TokenSequence, ClipError> {
        self.encode_weighted(&[WeightedText::checked(text, 1.0)?], cancellation)
    }

    pub fn encode_fixed_token_ids(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<[u32; SD1_CONTEXT_LENGTH], ClipError> {
        self.encode(text, cancellation)?.sd1_token_ids()
    }

    pub fn encode_weighted(
        &self,
        segments: &[WeightedText],
        cancellation: &CancellationToken,
    ) -> Result<TokenSequence, ClipError> {
        let content = self.encode_weighted_content(segments, cancellation)?;
        let content_tokens = content.len().min(SD1_CONTEXT_LENGTH - 2);
        cancellation.check().map_err(TensorError::from)?;
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(SD1_CONTEXT_LENGTH)
            .map_err(|_| ClipError::Allocation("token sequence"))?;
        encoded.push(WeightedToken::checked(SD1_START_TOKEN, 1.0)?);
        encoded.extend(content.into_iter().take(content_tokens));
        while encoded.len() < SD1_CONTEXT_LENGTH {
            encoded.push(WeightedToken::checked(SD1_END_TOKEN, 1.0)?);
        }
        TokenSequence::checked(encoded, content_tokens, SD1_CONTEXT_LENGTH, SD1_END_TOKEN)
    }

    pub fn encode_content(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<Vec<WeightedToken>, ClipError> {
        self.encode_weighted_content(&[WeightedText::checked(text, 1.0)?], cancellation)
    }

    pub fn encode_weighted_content(
        &self,
        segments: &[WeightedText],
        cancellation: &CancellationToken,
    ) -> Result<Vec<WeightedToken>, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if segments.len() > SD1_MAX_WEIGHTED_SEGMENTS {
            return Err(ClipError::TooManyWeightedSegments(segments.len()));
        }
        let total_bytes = segments.iter().try_fold(0_usize, |total, segment| {
            total
                .checked_add(segment.text().len())
                .ok_or(ClipError::Overflow("prompt byte length"))
        })?;
        if total_bytes > SD1_MAX_PROMPT_BYTES {
            return Err(ClipError::PromptTooLarge(total_bytes));
        }
        let mut encoded = Vec::new();
        encoded
            .try_reserve(total_bytes.min(4_096))
            .map_err(|_| ClipError::Allocation("content token sequence"))?;
        for segment in segments {
            cancellation.check().map_err(TensorError::from)?;
            let normalized = normalize_sd1_text(segment.text())?;
            for capture in self.token_pattern.find_iter(&normalized) {
                cancellation.check().map_err(TensorError::from)?;
                let mut token = String::new();
                token
                    .try_reserve(
                        capture
                            .as_str()
                            .len()
                            .checked_mul(2)
                            .ok_or(ClipError::Overflow("byte-encoded token capacity"))?,
                    )
                    .map_err(|_| ClipError::Allocation("byte-encoded token"))?;
                for byte in capture.as_str().as_bytes() {
                    token.push(*self.byte_encoder.get(byte).ok_or_else(|| {
                        ClipError::Tokenizer("byte encoder is incomplete".to_owned())
                    })?);
                }
                for piece in self.bpe(&token, cancellation)? {
                    let token = *self.vocabulary.get(&piece).ok_or_else(|| {
                        ClipError::Tokenizer(format!("unknown BPE token {piece:?}"))
                    })?;
                    if encoded.len() == SD1_MAX_PROMPT_BYTES {
                        return Err(ClipError::PromptTooLarge(total_bytes));
                    }
                    encoded
                        .try_reserve(1)
                        .map_err(|_| ClipError::Allocation("content token sequence"))?;
                    encoded.push(WeightedToken::checked(token, segment.weight())?);
                }
            }
        }
        Ok(encoded)
    }

    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special: bool,
        cancellation: &CancellationToken,
    ) -> Result<String, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if tokens.len() > SD1_MAX_PROMPT_BYTES {
            return Err(ClipError::PromptTooLarge(tokens.len()));
        }
        let mut decoded = String::new();
        let mut word_bytes = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            if index.is_multiple_of(256) {
                cancellation.check().map_err(TensorError::from)?;
            }
            if skip_special && (*token == SD1_START_TOKEN || *token == SD1_END_TOKEN) {
                continue;
            }
            let piece = self
                .inverse_vocabulary
                .get(token)
                .ok_or_else(|| ClipError::Tokenizer(format!("unknown SD1 token ID {token}")))?;
            let (piece, ends_word) = piece
                .strip_suffix("</w>")
                .map(|piece| (piece, true))
                .unwrap_or((piece, false));
            let pending_length = word_bytes
                .len()
                .checked_add(piece.len())
                .ok_or(ClipError::Overflow("decoded SD1 word bytes"))?;
            if pending_length > SD1_MAX_PROMPT_BYTES {
                return Err(ClipError::PromptTooLarge(pending_length));
            }
            word_bytes
                .try_reserve(piece.len())
                .map_err(|_| ClipError::Allocation("decoded SD1 word"))?;
            for character in piece.chars() {
                word_bytes.push(*self.byte_decoder.get(&character).ok_or_else(|| {
                    ClipError::Tokenizer(format!(
                        "SD1 vocabulary contains invalid byte character {character:?}"
                    ))
                })?);
            }
            if ends_word {
                let word = std::str::from_utf8(&word_bytes).map_err(|_| {
                    ClipError::Tokenizer("decoded SD1 bytes are not UTF-8".to_owned())
                })?;
                let decoded_length = decoded
                    .len()
                    .checked_add(word.len())
                    .and_then(|length| length.checked_add(1))
                    .ok_or(ClipError::Overflow("decoded SD1 text"))?;
                if decoded_length > SD1_MAX_PROMPT_BYTES {
                    return Err(ClipError::PromptTooLarge(decoded_length));
                }
                decoded
                    .try_reserve(word.len().saturating_add(1))
                    .map_err(|_| ClipError::Allocation("decoded SD1 text"))?;
                decoded.push_str(word);
                decoded.push(' ');
                word_bytes.clear();
            }
        }
        if !word_bytes.is_empty() {
            let word = std::str::from_utf8(&word_bytes)
                .map_err(|_| ClipError::Tokenizer("decoded SD1 bytes are not UTF-8".to_owned()))?;
            let decoded_length = decoded
                .len()
                .checked_add(word.len())
                .ok_or(ClipError::Overflow("decoded SD1 text"))?;
            if decoded_length > SD1_MAX_PROMPT_BYTES {
                return Err(ClipError::PromptTooLarge(decoded_length));
            }
            decoded
                .try_reserve(word.len())
                .map_err(|_| ClipError::Allocation("decoded SD1 text"))?;
            decoded.push_str(word);
        } else if decoded.ends_with(' ') {
            decoded.pop();
        }
        Ok(decoded)
    }

    fn bpe(&self, token: &str, cancellation: &CancellationToken) -> Result<Vec<String>, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        let mut word = Vec::new();
        word.try_reserve_exact(token.chars().count())
            .map_err(|_| ClipError::Allocation("BPE word"))?;
        for value in token.chars() {
            let mut character = String::new();
            character
                .try_reserve_exact(value.len_utf8())
                .map_err(|_| ClipError::Allocation("BPE character"))?;
            character.push(value);
            word.push(character);
        }
        let last = word
            .last_mut()
            .ok_or_else(|| ClipError::Tokenizer("empty BPE token".to_owned()))?;
        last.push_str("</w>");
        loop {
            cancellation.check().map_err(TensorError::from)?;
            let candidate = word
                .windows(2)
                .filter_map(|pair| {
                    self.merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .copied()
                        .map(|rank| (rank, pair[0].clone(), pair[1].clone()))
                })
                .min_by_key(|candidate| candidate.0);
            let Some((_, left, right)) = candidate else {
                break;
            };
            let mut merged = Vec::new();
            merged
                .try_reserve_exact(word.len())
                .map_err(|_| ClipError::Allocation("BPE merge"))?;
            let mut index = 0;
            while index < word.len() {
                if index + 1 < word.len() && word[index] == left && word[index + 1] == right {
                    let capacity = left
                        .len()
                        .checked_add(right.len())
                        .ok_or(ClipError::Overflow("BPE merged token"))?;
                    let mut piece = String::new();
                    piece
                        .try_reserve_exact(capacity)
                        .map_err(|_| ClipError::Allocation("BPE merged token"))?;
                    piece.push_str(&left);
                    piece.push_str(&right);
                    merged.push(piece);
                    index += 2;
                } else {
                    merged.push(word[index].clone());
                    index += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        Ok(word)
    }
}

fn normalize_sd1_text(text: &str) -> Result<String, ClipError> {
    let mut normalized = String::new();
    normalized
        .try_reserve(text.len())
        .map_err(|_| ClipError::Allocation("normalized SD1 prompt"))?;
    for character in text.trim().chars() {
        for lowercase in character.to_lowercase() {
            normalized
                .try_reserve(lowercase.len_utf8())
                .map_err(|_| ClipError::Allocation("normalized SD1 prompt"))?;
            normalized.push(lowercase);
            if normalized.len() > SD1_MAX_PROMPT_BYTES {
                return Err(ClipError::PromptTooLarge(normalized.len()));
            }
        }
    }
    Ok(normalized)
}

impl NativeTokenizer for Sd1Tokenizer {
    fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    fn tokenize_batch(
        &self,
        prompts: &[Vec<WeightedText>],
        cancellation: &CancellationToken,
    ) -> Result<TokenBatch, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if prompts.is_empty() || prompts.len() > MAX_TOKEN_BATCH {
            return Err(ClipError::InvalidBatchSize(prompts.len()));
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(prompts.len())
            .map_err(|_| ClipError::Allocation("token batch"))?;
        for prompt in prompts {
            rows.push(self.encode_weighted(prompt, cancellation)?);
        }
        TokenBatch::checked(
            self.identity.clone(),
            rows,
            SD1_CONTEXT_LENGTH,
            SD1_END_TOKEN,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipBindingIdentity {
    kind: &'static str,
    digest: String,
}

impl ClipBindingIdentity {
    fn derived(kind: &'static str, digest: String) -> Self {
        Self { kind, digest }
    }

    pub const fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn as_str(&self) -> &str {
        &self.digest
    }

    fn resident_owned_bytes(&self) -> Result<u64, ClipError> {
        u64::try_from(self.digest.capacity())
            .map_err(|_| ClipError::Overflow("CLIP binding identity"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipLayerSelection {
    Final,
    Hidden(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipPooling {
    None,
    EndToken,
    Mean,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClipExecutionPlan {
    target: ModelClipTargetDescriptor,
    tokenizer_identity: TokenizerIdentity,
    clip_model_identifier: String,
    artifact_identity: ClipBindingIdentity,
    model_identity: ClipBindingIdentity,
    patch_identity: ClipBindingIdentity,
    layer: ClipLayerSelection,
    pooling: ClipPooling,
    dtype: DType,
    device: DeviceId,
    context_length: usize,
    hidden_width: usize,
    digest: String,
}

impl ClipExecutionPlan {
    pub fn checked_from_domain(
        domain: &ClipLoadDomain,
        tokenizer_identity: TokenizerIdentity,
        layer: ClipLayerSelection,
        pooling: ClipPooling,
        context_length: usize,
        hidden_width: usize,
    ) -> Result<Self, ClipError> {
        if domain.selected().tokenizer() != tokenizer_identity.descriptor() {
            return Err(ClipError::DescriptorMismatch {
                tokenizer: tokenizer_identity.descriptor().identifier().to_owned(),
                clip_model: domain.selected().clip_model().target().as_str().to_owned(),
            });
        }
        Self::checked_legacy(
            domain.target().clone(),
            tokenizer_identity,
            domain.selected().clip_model().target().as_str(),
            domain.artifact_identity.clone(),
            domain.model_identity.clone(),
            domain.patch_identity.clone(),
            layer,
            pooling,
            domain.dtype(),
            domain.device(),
            context_length,
            hidden_width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn checked_legacy(
        target: ModelClipTargetDescriptor,
        tokenizer_identity: TokenizerIdentity,
        clip_model_identifier: impl Into<String>,
        artifact_identity: ClipBindingIdentity,
        model_identity: ClipBindingIdentity,
        patch_identity: ClipBindingIdentity,
        layer: ClipLayerSelection,
        pooling: ClipPooling,
        dtype: DType,
        device: DeviceId,
        context_length: usize,
        hidden_width: usize,
    ) -> Result<Self, ClipError> {
        require_binding_kind(&artifact_identity, "artifact")?;
        require_binding_kind(&model_identity, "model")?;
        require_binding_kind(&patch_identity, "patch")?;
        let clip_model_identifier = clip_model_identifier.into();
        let descriptor_matches = target.candidates().iter().any(|candidate| {
            candidate.tokenizer() == tokenizer_identity.descriptor()
                && candidate.clip_model().target().as_str() == clip_model_identifier
        });
        if !descriptor_matches {
            return Err(ClipError::DescriptorMismatch {
                tokenizer: tokenizer_identity.descriptor().identifier().to_owned(),
                clip_model: clip_model_identifier,
            });
        }
        if context_length < 2 || hidden_width == 0 {
            return Err(ClipError::InvalidExecutionShape {
                context_length,
                hidden_width,
            });
        }
        if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64) {
            return Err(ClipError::InvalidExecutionDType(dtype));
        }
        let clip_model_identifier = target
            .candidates()
            .iter()
            .find(|candidate| {
                candidate.tokenizer() == tokenizer_identity.descriptor()
                    && candidate.clip_model().target().as_str() == clip_model_identifier
            })
            .map(|candidate| candidate.clip_model().target().as_str().to_owned())
            .ok_or_else(|| ClipError::DescriptorMismatch {
                tokenizer: tokenizer_identity.descriptor().identifier().to_owned(),
                clip_model: "selected CLIP candidate disappeared".to_owned(),
            })?;
        let layer_json = serde_json::to_vec(&layer)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let pooling_json = serde_json::to_vec(&pooling)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let dtype_json = serde_json::to_vec(&dtype)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let device_json = serde_json::to_vec(&device)
            .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        let context_bytes = u64_bytes(context_length)?;
        let width_bytes = u64_bytes(hidden_width)?;
        let digest = digest_fields(&[
            tokenizer_identity.digest().as_bytes(),
            clip_model_identifier.as_bytes(),
            artifact_identity.kind().as_bytes(),
            artifact_identity.as_str().as_bytes(),
            model_identity.kind().as_bytes(),
            model_identity.as_str().as_bytes(),
            patch_identity.kind().as_bytes(),
            patch_identity.as_str().as_bytes(),
            &layer_json,
            &pooling_json,
            &dtype_json,
            &device_json,
            &context_bytes,
            &width_bytes,
        ]);
        Ok(Self {
            target,
            tokenizer_identity,
            clip_model_identifier,
            artifact_identity,
            model_identity,
            patch_identity,
            layer,
            pooling,
            dtype,
            device,
            context_length,
            hidden_width,
            digest,
        })
    }

    fn with_request_layer(&self, layer: ClipLayerSelection) -> Result<Self, ClipError> {
        Self::checked_legacy(
            self.target.clone(),
            self.tokenizer_identity.clone(),
            self.clip_model_identifier.clone(),
            self.artifact_identity.clone(),
            self.model_identity.clone(),
            self.patch_identity.clone(),
            layer,
            self.pooling,
            self.dtype,
            self.device,
            self.context_length,
            self.hidden_width,
        )
    }

    pub fn target(&self) -> &ModelClipTargetDescriptor {
        &self.target
    }

    pub fn tokenizer_identity(&self) -> &TokenizerIdentity {
        &self.tokenizer_identity
    }

    pub fn clip_model_identifier(&self) -> &str {
        &self.clip_model_identifier
    }

    pub fn artifact_identity(&self) -> &ClipBindingIdentity {
        &self.artifact_identity
    }

    pub fn model_identity(&self) -> &ClipBindingIdentity {
        &self.model_identity
    }

    pub fn patch_identity(&self) -> &ClipBindingIdentity {
        &self.patch_identity
    }

    pub const fn layer(&self) -> ClipLayerSelection {
        self.layer
    }

    pub const fn pooling(&self) -> ClipPooling {
        self.pooling
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn device(&self) -> DeviceId {
        self.device
    }

    fn resident_owned_bytes(&self) -> Result<u64, ClipError> {
        let mut bytes = self.tokenizer_identity.resident_owned_bytes()?;
        bytes = bytes
            .checked_add(
                self.target
                    .owned_resident_bytes()
                    .ok_or(ClipError::Overflow("CLIP target candidates"))?,
            )
            .ok_or(ClipError::Overflow("CLIP target candidates"))?;
        for text in [&self.clip_model_identifier, &self.digest] {
            add_clip_resident_bytes(&mut bytes, text.capacity(), "CLIP execution plan")?;
        }
        for identity in [
            &self.artifact_identity,
            &self.model_identity,
            &self.patch_identity,
        ] {
            bytes = bytes
                .checked_add(identity.resident_owned_bytes()?)
                .ok_or(ClipError::Overflow("CLIP execution identity"))?;
        }
        Ok(bytes)
    }

    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    pub const fn hidden_width(&self) -> usize {
        self.hidden_width
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

#[derive(Clone, Debug)]
pub struct ClipEncoding {
    plan_identity: String,
    token_batch_identity: String,
    conditioning: Tensor,
    pooled: Option<Tensor>,
    digest: String,
}

impl ClipEncoding {
    fn checked(
        plan: &ClipExecutionPlan,
        batch: &TokenBatch,
        conditioning: Tensor,
        pooled: Option<Tensor>,
        values_digest: String,
    ) -> Result<Self, ClipError> {
        let batch_size =
            u64::try_from(batch.rows().len()).map_err(|_| ClipError::Overflow("batch size"))?;
        let context_length = u64::try_from(plan.context_length())
            .map_err(|_| ClipError::Overflow("context length"))?;
        let hidden_width =
            u64::try_from(plan.hidden_width()).map_err(|_| ClipError::Overflow("hidden width"))?;
        let expected = [batch_size, context_length, hidden_width];
        if conditioning.descriptor().shape() != expected
            || conditioning.descriptor().dtype() != plan.dtype()
            || conditioning.descriptor().device() != plan.device()
        {
            return Err(ClipError::OutputMismatch);
        }
        match (&pooled, plan.pooling()) {
            (None, ClipPooling::None) => {}
            (Some(pooled), ClipPooling::EndToken | ClipPooling::Mean)
                if pooled.descriptor().shape() == [batch_size, hidden_width]
                    && pooled.descriptor().dtype() == plan.dtype()
                    && pooled.descriptor().device() == plan.device() => {}
            _ => return Err(ClipError::OutputMismatch),
        }
        let digest = digest_fields(&[
            plan.digest().as_bytes(),
            batch.digest().as_bytes(),
            values_digest.as_bytes(),
        ]);
        Ok(Self {
            plan_identity: plan.digest().to_owned(),
            token_batch_identity: batch.digest().to_owned(),
            conditioning,
            pooled,
            digest,
        })
    }

    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }

    pub fn token_batch_identity(&self) -> &str {
        &self.token_batch_identity
    }

    pub fn conditioning(&self) -> &Tensor {
        &self.conditioning
    }

    pub fn pooled(&self) -> Option<&Tensor> {
        self.pooled.as_ref()
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

pub trait NativeTextEncoder: Send + Sync {
    fn execute(
        &self,
        plan: &ClipExecutionPlan,
        batch: &TokenBatch,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipEncoding, ClipError>;
}

pub type Sd1ClipLayerTensors = ClipTextLayerWeights;
pub type Sd1ClipTensors = ClipTextWeights;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Sd1ClipParameterLayout {
    name: String,
    shape: Vec<u64>,
}

impl Sd1ClipParameterLayout {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sd1ClipArtifactProfile {
    source_prefix: String,
    vocabulary_size: usize,
    context_length: usize,
    hidden_width: usize,
    intermediate_width: usize,
    layer_count: usize,
    attention_heads: usize,
    parameters: Vec<Sd1ClipParameterLayout>,
    digest: String,
}

impl Sd1ClipArtifactProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        source_prefix: impl Into<String>,
        vocabulary_size: usize,
        context_length: usize,
        hidden_width: usize,
        intermediate_width: usize,
        layer_count: usize,
        attention_heads: usize,
    ) -> Result<Self, ClipError> {
        let source_prefix = source_prefix.into();
        if source_prefix.is_empty()
            || !source_prefix.ends_with('.')
            || source_prefix.contains('\0')
            || vocabulary_size == 0
            || context_length < 2
            || hidden_width == 0
            || intermediate_width == 0
            || layer_count == 0
            || attention_heads == 0
            || !hidden_width.is_multiple_of(attention_heads)
        {
            return Err(ClipError::InvalidEncoderShape);
        }
        let mut parameters = Vec::new();
        let mut add = |suffix: String, shape: Vec<u64>| {
            parameters.push(Sd1ClipParameterLayout {
                name: format!("{source_prefix}{suffix}"),
                shape,
            });
        };
        add(
            "text_model.embeddings.token_embedding.weight".to_owned(),
            vec![
                u64_from_usize(vocabulary_size)?,
                u64_from_usize(hidden_width)?,
            ],
        );
        add(
            "text_model.embeddings.position_embedding.weight".to_owned(),
            vec![
                u64_from_usize(context_length)?,
                u64_from_usize(hidden_width)?,
            ],
        );
        for layer in 0..layer_count {
            let prefix = format!("text_model.encoder.layers.{layer}");
            for norm in ["layer_norm1", "layer_norm2"] {
                add(
                    format!("{prefix}.{norm}.weight"),
                    vec![u64_from_usize(hidden_width)?],
                );
                add(
                    format!("{prefix}.{norm}.bias"),
                    vec![u64_from_usize(hidden_width)?],
                );
            }
            for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                add(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    vec![u64_from_usize(hidden_width)?, u64_from_usize(hidden_width)?],
                );
                add(
                    format!("{prefix}.self_attn.{projection}.bias"),
                    vec![u64_from_usize(hidden_width)?],
                );
            }
            add(
                format!("{prefix}.mlp.fc1.weight"),
                vec![
                    u64_from_usize(intermediate_width)?,
                    u64_from_usize(hidden_width)?,
                ],
            );
            add(
                format!("{prefix}.mlp.fc1.bias"),
                vec![u64_from_usize(intermediate_width)?],
            );
            add(
                format!("{prefix}.mlp.fc2.weight"),
                vec![
                    u64_from_usize(hidden_width)?,
                    u64_from_usize(intermediate_width)?,
                ],
            );
            add(
                format!("{prefix}.mlp.fc2.bias"),
                vec![u64_from_usize(hidden_width)?],
            );
        }
        add(
            "text_model.final_layer_norm.weight".to_owned(),
            vec![u64_from_usize(hidden_width)?],
        );
        add(
            "text_model.final_layer_norm.bias".to_owned(),
            vec![u64_from_usize(hidden_width)?],
        );
        parameters.sort_by(|left, right| left.name.cmp(&right.name));
        if parameters
            .windows(2)
            .any(|pair| pair[0].name == pair[1].name)
        {
            return Err(ClipError::InvalidParameterManifest(
                "SD1 CLIP artifact layout contains duplicate parameters".to_owned(),
            ));
        }
        let encoded = serde_json::to_vec(&(
            &source_prefix,
            vocabulary_size,
            context_length,
            hidden_width,
            intermediate_width,
            layer_count,
            attention_heads,
            &parameters,
        ))
        .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        Ok(Self {
            source_prefix,
            vocabulary_size,
            context_length,
            hidden_width,
            intermediate_width,
            layer_count,
            attention_heads,
            parameters,
            digest: sha256(&encoded),
        })
    }

    pub fn source_prefix(&self) -> &str {
        &self.source_prefix
    }

    pub fn parameters(&self) -> &[Sd1ClipParameterLayout] {
        &self.parameters
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn bind_execution(
        &self,
        family: ModelFamilyIdentity,
        artifact_digest: &str,
        patch_graph: &PatchGraph,
        tokenizer_identity: TokenizerIdentity,
    ) -> Result<Sd1ClipExecutionBinding, ClipError> {
        validate_sha256("artifact", artifact_digest)?;
        let patch_identity = patch_graph.identity();
        let architecture = self.projected_architecture()?;
        let candidate =
            ModelClipTargetCandidateDescriptor::checked("sim.sd1.tokenizer", "sim.sd1.clip")
                .map_err(|error| ClipError::InvalidParameterManifest(error.to_string()))?;
        let target = ModelClipTargetDescriptor::checked(vec![candidate], false)
            .map_err(|error| ClipError::InvalidParameterManifest(error.to_string()))?;
        let domain = ClipLoadDomain::checked_facts(
            family,
            target,
            ClipType::StableDiffusion,
            architecture.clone(),
            vec![artifact_digest.to_owned()],
            patch_identity,
            self.digest(),
            DType::F32,
            DeviceId::CPU,
        )?;
        let plan = ClipExecutionPlan::checked_from_domain(
            &domain,
            tokenizer_identity,
            ClipLayerSelection::Final,
            ClipPooling::None,
            self.context_length,
            self.hidden_width,
        )?;
        Ok(Sd1ClipExecutionBinding {
            architecture,
            profile_identity: self.digest.clone(),
            plan,
        })
    }

    fn projected_architecture(&self) -> Result<ClipArchitectureSelection, ClipError> {
        let tensor_shapes = self
            .parameters
            .iter()
            .map(|parameter| {
                parameter
                    .name
                    .strip_prefix(&self.source_prefix)
                    .map(|name| (name.to_owned(), parameter.shape.clone()))
                    .ok_or_else(|| {
                        ClipError::InvalidParameterManifest(
                            "SD1 CLIP parameter escaped its source prefix".to_owned(),
                        )
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        select_clip_architecture(
            ClipType::StableDiffusion,
            &[ModelProbe {
                tensor_shapes,
                metadata: BTreeMap::new(),
            }],
        )
    }
}

#[derive(Clone, Debug)]
pub struct Sd1ClipExecutionBinding {
    architecture: ClipArchitectureSelection,
    profile_identity: String,
    plan: ClipExecutionPlan,
}

impl Sd1ClipExecutionBinding {
    pub fn architecture(&self) -> &ClipArchitectureSelection {
        &self.architecture
    }

    pub fn plan(&self) -> &ClipExecutionPlan {
        &self.plan
    }

    pub fn profile_identity(&self) -> &str {
        &self.profile_identity
    }
}

#[derive(Debug)]
pub struct LoadedSd1Clip {
    binding: Sd1ClipExecutionBinding,
    encoder: Sd1ClipTextEncoder,
}

impl LoadedSd1Clip {
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_store(
        profile: &Sd1ClipArtifactProfile,
        binding: Sd1ClipExecutionBinding,
        store: &ModelStore,
        index: &ArtifactIndex,
        loaded: &LoadedModel,
        backend: Arc<CpuBackend>,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        context.check()?;
        if loaded.identity() != binding.plan().artifact_identity().as_str()
            || profile.digest() != binding.profile_identity()
        {
            return Err(ClipError::BindingMismatch);
        }
        let expected = profile
            .parameters
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<BTreeSet<_>>();
        let actual = loaded
            .tensors()
            .keys()
            .filter(|name| name.starts_with(profile.source_prefix()))
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(ClipError::ParameterSetMismatch {
                artifact_index: 0,
                missing: expected
                    .difference(&actual)
                    .map(|name| (*name).to_owned())
                    .collect(),
                unexpected: actual
                    .difference(&expected)
                    .map(|name| (*name).to_owned())
                    .collect(),
            });
        }
        let mut bytes = store.read_tensors(
            index,
            loaded,
            profile
                .parameters
                .iter()
                .map(|parameter| parameter.name.as_str()),
            context.cancellation,
        )?;
        let mut tensors = BTreeMap::new();
        for parameter in &profile.parameters {
            context.check()?;
            let metadata = loaded
                .tensors()
                .get(&parameter.name)
                .ok_or_else(|| ClipError::MissingManifestParameter(parameter.name.clone()))?;
            if metadata.shape != parameter.shape {
                return Err(ClipError::ManifestParameterShape {
                    name: parameter.name.clone(),
                    expected: parameter.shape.clone(),
                    actual: metadata.shape.clone(),
                });
            }
            if metadata.data_type != "F32" {
                return Err(ClipError::ManifestParameterDType {
                    name: parameter.name.clone(),
                    expected: "F32".to_owned(),
                    actual: metadata.data_type.clone(),
                });
            }
            let encoded = bytes
                .remove(&metadata.name)
                .ok_or_else(|| ClipError::MissingManifestParameter(metadata.name.clone()))?;
            if !encoded.len().is_multiple_of(std::mem::size_of::<f32>()) {
                return Err(ClipError::InvalidParameterBytes(metadata.name.clone()));
            }
            let mut values = backend.workspace_vec(context, encoded.len() / 4)?;
            for (index, chunk) in encoded.chunks_exact(4).enumerate() {
                if index.is_multiple_of(64) {
                    context.check()?;
                }
                let value = <[u8; 4]>::try_from(chunk)
                    .map_err(|_| ClipError::InvalidParameterBytes(metadata.name.clone()))?;
                values.try_push(f32::from_le_bytes(value))?;
            }
            let tensor = tensor_from_f32(&backend, &parameter.shape, &values, context)?;
            tensors.insert(parameter.name.clone(), tensor);
        }
        if !bytes.is_empty() {
            return Err(ClipError::NativeProjectionSetMismatch);
        }
        let clip_tensors = profile.take_tensors(&mut tensors)?;
        if !tensors.is_empty() {
            return Err(ClipError::NativeProjectionSetMismatch);
        }
        let parameters = Sd1ClipParameters::checked_for_plan(
            binding.plan(),
            clip_tensors,
            profile.vocabulary_size,
            profile.attention_heads,
        )?;
        let encoder = Sd1ClipTextEncoder::new(backend, parameters)?;
        Ok(Self { binding, encoder })
    }

    pub fn architecture(&self) -> &ClipArchitectureSelection {
        self.binding.architecture()
    }

    pub fn plan(&self) -> &ClipExecutionPlan {
        self.binding.plan()
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, ClipError> {
        let mut storages = BTreeSet::new();
        parameter_tensors(&self.encoder.parameters.tensors)
            .into_iter()
            .try_fold(0_u64, |total, tensor| {
                if !storages.insert(tensor.storage_id().get()) {
                    return Ok(total);
                }
                total
                    .checked_add(tensor.storage_byte_len())
                    .ok_or(ClipError::Overflow("CLIP resident storage bytes"))
            })
    }

    pub fn resident_bytes(&self) -> Result<u64, ClipError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| ClipError::Overflow("CLIP resident object bytes"))?;
        add_clip_resident_bytes(
            &mut bytes,
            self.binding.profile_identity.capacity(),
            "CLIP profile identity",
        )?;
        bytes = bytes
            .checked_add(self.binding.architecture.resident_owned_bytes()?)
            .ok_or(ClipError::Overflow("CLIP architecture resident bytes"))?;
        bytes = bytes
            .checked_add(self.binding.plan.resident_owned_bytes()?)
            .ok_or(ClipError::Overflow("CLIP plan resident bytes"))?;
        for identity in [
            &self.encoder.parameters.artifact_identity,
            &self.encoder.parameters.model_identity,
            &self.encoder.parameters.patch_identity,
        ] {
            bytes = bytes
                .checked_add(identity.resident_owned_bytes()?)
                .ok_or(ClipError::Overflow("CLIP parameter identity bytes"))?;
        }
        add_clip_resident_bytes(
            &mut bytes,
            self.encoder
                .parameters
                .tensors
                .layers
                .capacity()
                .checked_mul(std::mem::size_of::<Sd1ClipLayerTensors>())
                .ok_or(ClipError::Overflow("CLIP layer tensor handles"))?,
            "CLIP layer tensor handles",
        )?;
        bytes
            .checked_add(self.resident_storage_bytes()?)
            .ok_or(ClipError::Overflow("CLIP resident total bytes"))
    }

    pub fn execute_with_source_layer(
        &self,
        batch: &TokenBatch,
        source_layer: Option<i64>,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipEncoding, ClipError> {
        context.check()?;
        let Some(source_layer) = source_layer else {
            return self.execute(batch, context);
        };
        let count = self.encoder.parameters.layer_count();
        let layer = match source_clip_layer_selection(
            &NativeClipLayerSelection::Last,
            count,
            source_layer,
        )? {
            NativeClipLayerSelection::Last => ClipLayerSelection::Final,
            NativeClipLayerSelection::Hidden(index) => {
                let index = if index < 0 {
                    count
                        .checked_add_signed(index)
                        .ok_or(ClipError::Overflow("source SD1 CLIP layer"))?
                } else {
                    usize::try_from(index)
                        .map_err(|_| ClipError::Overflow("source SD1 CLIP layer"))?
                };
                ClipLayerSelection::Hidden(
                    u32::try_from(index)
                        .map_err(|_| ClipError::Overflow("source SD1 CLIP layer"))?,
                )
            }
            NativeClipLayerSelection::All => {
                return Err(ClipError::InvalidNativeResource(
                    "SD1 source layer cannot select all states".to_owned(),
                ));
            }
        };
        let _request_metadata_lease = self
            .encoder
            .backend
            .reserve_workspace(context, self.plan().resident_owned_bytes()?)?;
        let plan = self.plan().with_request_layer(layer)?;
        let encoding = self.encoder.execute(&plan, batch, context)?;
        context.check()?;
        Ok(encoding)
    }

    pub fn resident_tensor_allocations(&self) -> Vec<(sim_tensor::StorageId, u64)> {
        self.encoder.transformer.resident_tensor_allocations()
    }

    pub fn execute(
        &self,
        batch: &TokenBatch,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipEncoding, ClipError> {
        self.encoder.execute(self.binding.plan(), batch, context)
    }
}

impl Sd1ClipArtifactProfile {
    fn take_tensors(
        &self,
        tensors: &mut BTreeMap<String, Tensor>,
    ) -> Result<Sd1ClipTensors, ClipError> {
        let take = |tensors: &mut BTreeMap<String, Tensor>, suffix: &str| {
            let name = format!("{}{suffix}", self.source_prefix);
            tensors
                .remove(&name)
                .ok_or(ClipError::MissingManifestParameter(name))
        };
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.layer_count)
            .map_err(|_| ClipError::Allocation("SD1 CLIP layers"))?;
        for layer in 0..self.layer_count {
            let prefix = format!("text_model.encoder.layers.{layer}");
            layers.push(Sd1ClipLayerTensors {
                layer_norm_1_weight: take(tensors, &format!("{prefix}.layer_norm1.weight"))?,
                layer_norm_1_bias: take(tensors, &format!("{prefix}.layer_norm1.bias"))?,
                query_weight: take(tensors, &format!("{prefix}.self_attn.q_proj.weight"))?,
                query_bias: take(tensors, &format!("{prefix}.self_attn.q_proj.bias"))?,
                key_weight: take(tensors, &format!("{prefix}.self_attn.k_proj.weight"))?,
                key_bias: take(tensors, &format!("{prefix}.self_attn.k_proj.bias"))?,
                value_weight: take(tensors, &format!("{prefix}.self_attn.v_proj.weight"))?,
                value_bias: take(tensors, &format!("{prefix}.self_attn.v_proj.bias"))?,
                output_weight: take(tensors, &format!("{prefix}.self_attn.out_proj.weight"))?,
                output_bias: take(tensors, &format!("{prefix}.self_attn.out_proj.bias"))?,
                layer_norm_2_weight: take(tensors, &format!("{prefix}.layer_norm2.weight"))?,
                layer_norm_2_bias: take(tensors, &format!("{prefix}.layer_norm2.bias"))?,
                feed_forward_1_weight: take(tensors, &format!("{prefix}.mlp.fc1.weight"))?,
                feed_forward_1_bias: take(tensors, &format!("{prefix}.mlp.fc1.bias"))?,
                feed_forward_2_weight: take(tensors, &format!("{prefix}.mlp.fc2.weight"))?,
                feed_forward_2_bias: take(tensors, &format!("{prefix}.mlp.fc2.bias"))?,
            });
        }
        Ok(Sd1ClipTensors {
            token_embedding: take(tensors, "text_model.embeddings.token_embedding.weight")?,
            position_embedding: take(tensors, "text_model.embeddings.position_embedding.weight")?,
            layers,
            final_layer_norm_weight: take(tensors, "text_model.final_layer_norm.weight")?,
            final_layer_norm_bias: take(tensors, "text_model.final_layer_norm.bias")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Sd1ClipParameters {
    artifact_identity: ClipBindingIdentity,
    model_identity: ClipBindingIdentity,
    patch_identity: ClipBindingIdentity,
    tensors: Sd1ClipTensors,
    vocabulary_size: usize,
    context_length: usize,
    hidden_width: usize,
    attention_heads: usize,
    stream: StreamId,
}

impl Sd1ClipParameters {
    pub fn checked_for_plan(
        plan: &ClipExecutionPlan,
        tensors: Sd1ClipTensors,
        vocabulary_size: usize,
        attention_heads: usize,
    ) -> Result<Self, ClipError> {
        Self::checked_legacy(
            plan.artifact_identity().clone(),
            plan.model_identity().clone(),
            plan.patch_identity().clone(),
            tensors,
            vocabulary_size,
            plan.context_length(),
            plan.hidden_width(),
            attention_heads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn checked_legacy(
        artifact_identity: ClipBindingIdentity,
        model_identity: ClipBindingIdentity,
        patch_identity: ClipBindingIdentity,
        tensors: Sd1ClipTensors,
        vocabulary_size: usize,
        context_length: usize,
        hidden_width: usize,
        attention_heads: usize,
    ) -> Result<Self, ClipError> {
        require_binding_kind(&artifact_identity, "artifact")?;
        require_binding_kind(&model_identity, "model")?;
        require_binding_kind(&patch_identity, "patch")?;
        if vocabulary_size == 0
            || context_length < 2
            || hidden_width == 0
            || attention_heads == 0
            || !hidden_width.is_multiple_of(attention_heads)
            || tensors.layers.is_empty()
        {
            return Err(ClipError::InvalidEncoderShape);
        }
        require_tensor(
            &tensors.token_embedding,
            &[vocabulary_size, hidden_width],
            "token embedding",
        )?;
        require_tensor(
            &tensors.position_embedding,
            &[context_length, hidden_width],
            "position embedding",
        )?;
        require_tensor(
            &tensors.final_layer_norm_weight,
            &[hidden_width],
            "final layer norm weight",
        )?;
        require_tensor(
            &tensors.final_layer_norm_bias,
            &[hidden_width],
            "final layer norm bias",
        )?;
        for layer in &tensors.layers {
            require_layer(layer, hidden_width)?;
        }
        let stream = tensors.token_embedding.descriptor().stream();
        if parameter_tensors(&tensors)
            .into_iter()
            .any(|tensor| tensor.descriptor().stream() != stream)
        {
            return Err(ClipError::ParameterStreamMismatch);
        }
        Ok(Self {
            artifact_identity,
            model_identity,
            patch_identity,
            tensors,
            vocabulary_size,
            context_length,
            hidden_width,
            attention_heads,
            stream,
        })
    }

    pub fn layer_count(&self) -> usize {
        self.tensors.layers.len()
    }
}

#[derive(Debug)]
pub struct Sd1ClipTextEncoder {
    backend: Arc<CpuBackend>,
    parameters: Sd1ClipParameters,
    transformer: NativeClipText,
}

impl Sd1ClipTextEncoder {
    pub fn new(backend: Arc<CpuBackend>, parameters: Sd1ClipParameters) -> Result<Self, ClipError> {
        let intermediate_size = parameters
            .tensors
            .layers
            .first()
            .and_then(|layer| layer.feed_forward_1_weight.descriptor().shape().first())
            .copied()
            .ok_or(ClipError::InvalidEncoderShape)
            .and_then(|value| {
                usize::try_from(value).map_err(|_| ClipError::Overflow("feed-forward width"))
            })?;
        let transformer = NativeClipText::new(
            ClipTextConfiguration {
                dtype: DType::F32,
                device: DeviceId::CPU,
                vocabulary_size: parameters.vocabulary_size,
                max_position_embeddings: parameters.context_length,
                hidden_size: parameters.hidden_width,
                intermediate_size,
                attention_heads: parameters.attention_heads,
                layer_count: parameters.layer_count(),
                eos_token_id: SD1_END_TOKEN,
                activation: ClipTextActivation::QuickGelu,
                projection_dimension: None,
            },
            parameters.tensors.clone(),
            None,
        )?;
        Ok(Self {
            backend,
            parameters,
            transformer,
        })
    }

    fn validate_bindings(
        &self,
        plan: &ClipExecutionPlan,
        batch: &TokenBatch,
        context: &ExecutionContext<'_>,
    ) -> Result<(), ClipError> {
        context.check()?;
        if plan.tokenizer_identity() != batch.tokenizer_identity() {
            return Err(ClipError::TokenizerMismatch);
        }
        if plan.artifact_identity() != &self.parameters.artifact_identity
            || plan.model_identity() != &self.parameters.model_identity
            || plan.patch_identity() != &self.parameters.patch_identity
        {
            return Err(ClipError::BindingMismatch);
        }
        if plan.context_length() != self.parameters.context_length
            || plan.hidden_width() != self.parameters.hidden_width
            || batch.context_length() != self.parameters.context_length
        {
            return Err(ClipError::InvalidEncoderShape);
        }
        match plan.layer() {
            ClipLayerSelection::Final => {}
            ClipLayerSelection::Hidden(index) => {
                let index =
                    usize::try_from(index).map_err(|_| ClipError::Overflow("CLIP layer"))?;
                if index >= self.parameters.layer_count() {
                    return Err(ClipError::LayerOutOfRange {
                        requested: index,
                        available: self.parameters.layer_count(),
                    });
                }
            }
        }
        if self.parameters.stream != context.stream {
            return Err(ClipError::ParameterStreamMismatch);
        }
        Ok(())
    }

    fn admit_execution_target(
        &self,
        plan: &ClipExecutionPlan,
        context: &ExecutionContext<'_>,
    ) -> Result<(), ClipError> {
        context.check()?;
        if plan.dtype() != DType::F32 || plan.device() != DeviceId::CPU {
            return Err(ClipError::UnsupportedConcreteTarget {
                dtype: plan.dtype(),
                device: plan.device(),
            });
        }
        Ok(())
    }
}

impl NativeTextEncoder for Sd1ClipTextEncoder {
    fn execute(
        &self,
        plan: &ClipExecutionPlan,
        batch: &TokenBatch,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipEncoding, ClipError> {
        self.admit_execution_target(plan, context)?;
        self.validate_bindings(plan, batch, context)?;
        self.transformer
            .admit_execution_target(&self.backend, context)?;
        let batch_size = batch.rows().len();
        let has_weights = batch
            .rows()
            .iter()
            .any(|row| row.tokens().iter().any(|token| token.weight() != 1.0));
        let input_tokens = sd1_token_input(&self.backend, batch, has_weights, context)?;
        let intermediate = match plan.layer() {
            ClipLayerSelection::Final => ClipTextIntermediate::None,
            ClipLayerSelection::Hidden(index) => ClipTextIntermediate::Layer(
                isize::try_from(index).map_err(|_| ClipError::Overflow("CLIP layer"))?,
            ),
        };
        let output = self.transformer.forward(
            &self.backend,
            ClipTextRequest {
                input: ClipTextInput::Tokens(&input_tokens),
                attention_mask: None,
                num_tokens: None,
                intermediate,
                final_layer_norm_intermediate: true,
                project_pooled: false,
                zero_out_masked: false,
            },
            context,
        )?;
        let final_values = tensor_to_f32(&self.backend, output.last_hidden_state(), context)?;
        let conditioning_tensor = output
            .intermediate()
            .unwrap_or_else(|| output.last_hidden_state());
        let conditioning_values = tensor_to_f32(&self.backend, conditioning_tensor, context)?;
        let pooled_values = pool_values(
            &self.backend,
            &final_values,
            batch,
            self.parameters.hidden_width,
            plan.pooling(),
            context,
        )?;
        let actual_shape = [
            u64::try_from(batch_size).map_err(|_| ClipError::Overflow("batch size"))?,
            u64::try_from(self.parameters.context_length)
                .map_err(|_| ClipError::Overflow("context length"))?,
            u64::try_from(self.parameters.hidden_width)
                .map_err(|_| ClipError::Overflow("hidden width"))?,
        ];
        let mut weighted_values = if has_weights {
            Some(apply_token_weights(
                &self.backend,
                &conditioning_values,
                batch,
                self.parameters.hidden_width,
                context,
            )?)
        } else {
            None
        };
        let digest_values = weighted_values.as_deref().unwrap_or(&conditioning_values);
        let values_digest = digest_f32(digest_values, pooled_values.as_deref(), context)?;
        let pooled = match pooled_values {
            None => None,
            Some(values) => Some(tensor_from_f32(
                &self.backend,
                &[
                    u64::try_from(batch_size).map_err(|_| ClipError::Overflow("batch size"))?,
                    u64::try_from(self.parameters.hidden_width)
                        .map_err(|_| ClipError::Overflow("hidden width"))?,
                ],
                &values,
                context,
            )?),
        };
        let conditioning = match weighted_values.take() {
            Some(values) => tensor_from_f32(&self.backend, &actual_shape, &values, context)?,
            None => conditioning_tensor.clone(),
        };
        drop(final_values);
        drop(conditioning_values);
        ClipEncoding::checked(plan, batch, conditioning, pooled, values_digest)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeClipProfile {
    Sd1,
    Sdxl,
    Sd3,
    PixArt,
    Lumina,
    HiDream,
    Qwen,
    Gemma,
}

impl NativeClipProfile {
    const fn execution_profile(self) -> &'static str {
        match self {
            Self::Sd1 => "clip.sd1",
            Self::Sdxl => "clip.sdxl",
            Self::Sd3 => "clip.sd3",
            Self::PixArt => "clip.pixart",
            Self::Lumina => "clip.lumina",
            Self::HiDream => "clip.hidream",
            Self::Qwen => "clip.qwen",
            Self::Gemma => "clip.gemma",
        }
    }
}

fn native_clip_profile_execution_requirements(
    profile: NativeClipProfile,
    dtype: DType,
) -> NativeExecutionRequirements {
    let mut requirements = NativeExecutionRequirements::new();
    requirements.append_tensor_io(dtype);
    requirements.append_linear_algebra(LinearAlgebraOperation::BatchMatrixMultiply, dtype);
    for operation in [
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
    ] {
        requirements.append_binary(operation, dtype);
    }
    for operation in [UnaryOperation::Exponential, UnaryOperation::SquareRoot] {
        requirements.append_unary(operation, dtype);
    }
    for operation in [
        ReductionOperation::Sum,
        ReductionOperation::Mean,
        ReductionOperation::Variance,
    ] {
        requirements.append_reduction(operation, dtype);
    }
    if matches!(profile, NativeClipProfile::Qwen | NativeClipProfile::Gemma) {
        requirements.extend([
            OperationSupport::resize_input(ResizeMode::Bilinear, dtype, Layout::Contiguous),
            OperationSupport::resize_output(ResizeMode::Bilinear, dtype, Layout::Contiguous),
            OperationSupport::resize_input(ResizeMode::Bicubic, dtype, Layout::Contiguous),
            OperationSupport::resize_output(ResizeMode::Bicubic, dtype, Layout::Contiguous),
            OperationSupport::convolution_input(dtype, Layout::Contiguous),
            OperationSupport::convolution_output(dtype, Layout::Contiguous),
        ]);
        requirements.append_unary(UnaryOperation::Sigmoid, dtype);
        requirements.append_unary(UnaryOperation::HyperbolicTangent, dtype);
    }
    requirements
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeClipComponentRole {
    ClipL,
    ClipG,
    T5,
    Llama,
    Qwen,
    Gemma,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeClipLayerSelection {
    Last,
    Hidden(isize),
    All,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeClipComponentOptions {
    tokenizer: NativeTokenizerInvocationOptions,
    layer: NativeClipLayerSelection,
    final_layer_norm_intermediate: bool,
    projected_pooled: bool,
    attention_mask: bool,
    zero_out_masked: bool,
}

impl NativeClipComponentOptions {
    pub fn checked(
        tokenizer: NativeTokenizerInvocationOptions,
        layer: NativeClipLayerSelection,
        projected_pooled: bool,
        attention_mask: bool,
    ) -> Result<Self, ClipError> {
        Self::checked_source(
            tokenizer,
            layer,
            false,
            projected_pooled,
            attention_mask,
            false,
        )
    }

    pub fn checked_source(
        tokenizer: NativeTokenizerInvocationOptions,
        layer: NativeClipLayerSelection,
        final_layer_norm_intermediate: bool,
        projected_pooled: bool,
        attention_mask: bool,
        zero_out_masked: bool,
    ) -> Result<Self, ClipError> {
        if zero_out_masked && !attention_mask {
            return Err(ClipError::InvalidNativeResource(
                "zeroing masked CLIP outputs requires attention-mask execution".to_owned(),
            ));
        }
        Ok(Self {
            tokenizer,
            layer,
            final_layer_norm_intermediate,
            projected_pooled,
            attention_mask,
            zero_out_masked,
        })
    }

    pub const fn tokenizer(&self) -> NativeTokenizerInvocationOptions {
        self.tokenizer
    }
}

#[derive(Clone)]
pub enum NativeClipEncoder {
    ClipText(Arc<NativeClipText>),
    Bidirectional(Arc<NativeT5TextEncoder>),
    Decoder(Arc<NativeDecoderTextEncoder>),
    Qwen25(Arc<NativeQwen25Multimodal>),
    Gemma4(Arc<NativeGemmaMultimodal>),
}

impl NativeClipEncoder {
    fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        match self {
            Self::ClipText(owner) => owner.execution_requirements(),
            Self::Bidirectional(owner) => owner.execution_requirements(),
            Self::Decoder(owner) => owner.execution_requirements(),
            Self::Qwen25(owner) => owner.execution_requirements(dtype),
            Self::Gemma4(owner) => owner.execution_requirements(dtype),
        }
    }
}

#[derive(Clone)]
pub struct NativeClipComponent {
    role: NativeClipComponentRole,
    artifact_sha256: String,
    base_weights: Arc<MappedModelWeights>,
    patch_graph: Arc<PatchGraph>,
    tokenizer: Arc<NativePromptTokenizer>,
    encoder: NativeClipEncoder,
    options: NativeClipComponentOptions,
    recipe_binding_sha256: Option<String>,
    semantic_digest_sha256: String,
}

impl NativeClipComponent {
    pub fn checked_clip_text(
        role: NativeClipComponentRole,
        base_weights: Arc<MappedModelWeights>,
        tokenizer: Arc<NativePromptTokenizer>,
        encoder_template: &NativeClipText,
        options: NativeClipComponentOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        if !matches!(
            role,
            NativeClipComponentRole::ClipL | NativeClipComponentRole::ClipG
        ) {
            return Err(ClipError::InvalidNativeResource(
                "CLIP text owner requires a CLIP-L or CLIP-G role".to_owned(),
            ));
        }
        validate_sha256(
            "native CLIP component artifact",
            base_weights.base_artifact_digest(),
        )?;
        let encoder =
            encoder_template.reconstruct_from_mapped_weights(&base_weights, cancellation)?;
        Self::from_sealed_owner(
            role,
            base_weights,
            tokenizer,
            NativeClipEncoder::ClipText(Arc::new(encoder)),
            options,
            cancellation,
        )
    }

    pub fn checked_bidirectional(
        role: NativeClipComponentRole,
        base_weights: Arc<MappedModelWeights>,
        tokenizer: Arc<NativePromptTokenizer>,
        encoder_template: &NativeT5TextEncoder,
        options: NativeClipComponentOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        if role != NativeClipComponentRole::T5 {
            return Err(ClipError::InvalidNativeResource(
                "bidirectional text owner requires the T5 role".to_owned(),
            ));
        }
        validate_sha256(
            "native CLIP component artifact",
            base_weights.base_artifact_digest(),
        )?;
        let encoder = encoder_template
            .reconstruct_from_mapped_weights(&base_weights, cancellation)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        Self::from_sealed_owner(
            role,
            base_weights,
            tokenizer,
            NativeClipEncoder::Bidirectional(Arc::new(encoder)),
            options,
            cancellation,
        )
    }

    pub fn checked_decoder(
        role: NativeClipComponentRole,
        base_weights: Arc<MappedModelWeights>,
        tokenizer: Arc<NativePromptTokenizer>,
        encoder_template: &NativeDecoderTextEncoder,
        options: NativeClipComponentOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        if !matches!(
            role,
            NativeClipComponentRole::Llama
                | NativeClipComponentRole::Qwen
                | NativeClipComponentRole::Gemma
        ) {
            return Err(ClipError::InvalidNativeResource(
                "decoder text owner requires a Llama, Qwen, or Gemma role".to_owned(),
            ));
        }
        validate_sha256(
            "native CLIP component artifact",
            base_weights.base_artifact_digest(),
        )?;
        let encoder = encoder_template
            .reconstruct_from_mapped_weights(&base_weights, cancellation)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        Self::from_sealed_owner(
            role,
            base_weights,
            tokenizer,
            NativeClipEncoder::Decoder(Arc::new(encoder)),
            options,
            cancellation,
        )
    }

    pub fn checked_qwen25(
        base_weights: Arc<MappedModelWeights>,
        tokenizer: Arc<NativePromptTokenizer>,
        encoder_template: &NativeQwen25Multimodal,
        options: NativeClipComponentOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        if !Arc::ptr_eq(encoder_template.tokenizer(), &tokenizer) {
            return Err(ClipError::InvalidNativeResource(
                "Qwen2.5 tokenizer is not its retained multimodal owner".to_owned(),
            ));
        }
        validate_sha256(
            "native CLIP component artifact",
            base_weights.base_artifact_digest(),
        )?;
        let owner = encoder_template
            .reconstruct_from_mapped_weights(&base_weights, cancellation)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        Self::from_sealed_owner(
            NativeClipComponentRole::Qwen,
            base_weights,
            tokenizer,
            NativeClipEncoder::Qwen25(Arc::new(owner)),
            options,
            cancellation,
        )
    }

    pub fn checked_gemma4(
        base_weights: Arc<MappedModelWeights>,
        tokenizer: Arc<NativePromptTokenizer>,
        encoder_template: &NativeGemmaMultimodal,
        options: NativeClipComponentOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        if !Arc::ptr_eq(encoder_template.tokenizer(), &tokenizer)
            || !encoder_template.is_source_exact_profile()
            || encoder_template.gemma4_vision().is_none()
        {
            return Err(ClipError::InvalidNativeResource(
                "Gemma4 tokenizer and retained multimodal owner are incompatible".to_owned(),
            ));
        }
        validate_sha256(
            "native CLIP component artifact",
            base_weights.base_artifact_digest(),
        )?;
        let owner = encoder_template
            .reconstruct_from_mapped_weights(&base_weights, cancellation)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        Self::from_sealed_owner(
            NativeClipComponentRole::Gemma,
            base_weights,
            tokenizer,
            NativeClipEncoder::Gemma4(Arc::new(owner)),
            options,
            cancellation,
        )
    }

    fn from_sealed_owner(
        role: NativeClipComponentRole,
        base_weights: Arc<MappedModelWeights>,
        tokenizer: Arc<NativePromptTokenizer>,
        encoder: NativeClipEncoder,
        options: NativeClipComponentOptions,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        match &encoder {
            NativeClipEncoder::ClipText(owner) => {
                if !matches!(
                    role,
                    NativeClipComponentRole::ClipL | NativeClipComponentRole::ClipG
                ) || (options.projected_pooled
                    && owner.configuration().projection_dimension.is_none())
                {
                    return Err(ClipError::InvalidNativeResource(
                        "CLIP text role or projected-pooling option is incompatible with its owner"
                            .to_owned(),
                    ));
                }
            }
            NativeClipEncoder::Bidirectional(owner) => {
                if role != NativeClipComponentRole::T5
                    || options.layer == NativeClipLayerSelection::All
                    || (options.projected_pooled
                        && owner.configuration().projection_dimension.is_none())
                {
                    return Err(ClipError::InvalidNativeResource(
                        "T5 role, layer, or projected-pooling option is incompatible with its owner".to_owned(),
                    ));
                }
            }
            NativeClipEncoder::Decoder(_) => {
                if options.projected_pooled
                    || options.zero_out_masked
                    || (options.layer == NativeClipLayerSelection::All
                        && !matches!(
                            role,
                            NativeClipComponentRole::Llama | NativeClipComponentRole::Gemma
                        ))
                {
                    return Err(ClipError::InvalidNativeResource(
                        "decoder layer or output options are incompatible with its CLIP role"
                            .to_owned(),
                    ));
                }
            }
            NativeClipEncoder::Qwen25(_) => {
                if role != NativeClipComponentRole::Qwen
                    || options.layer == NativeClipLayerSelection::All
                    || options.projected_pooled
                    || options.zero_out_masked
                    || !options.attention_mask
                    || options.tokenizer != NativeTokenizerInvocationOptions::default()
                {
                    return Err(ClipError::InvalidNativeResource(
                        "Qwen2.5 CLIP options are incompatible with its multimodal owner"
                            .to_owned(),
                    ));
                }
            }
            NativeClipEncoder::Gemma4(_) => {
                if role != NativeClipComponentRole::Gemma
                    || options.layer != NativeClipLayerSelection::All
                    || options.projected_pooled
                    || options.zero_out_masked
                    || !options.attention_mask
                    || options.tokenizer != NativeTokenizerInvocationOptions::default()
                {
                    return Err(ClipError::InvalidNativeResource(
                        "Gemma4 CLIP options are incompatible with its multimodal owner".to_owned(),
                    ));
                }
            }
        }
        let artifact_sha256 = base_weights.base_artifact_digest().to_owned();
        let mut component = Self {
            role,
            artifact_sha256: artifact_sha256.clone(),
            base_weights,
            patch_graph: Arc::new(PatchGraph::checked_semantic(artifact_sha256, Vec::new())?),
            tokenizer,
            encoder,
            options,
            recipe_binding_sha256: None,
            semantic_digest_sha256: String::new(),
        };
        component.semantic_digest_sha256 = component.project_semantic_digest(cancellation)?;
        Ok(component)
    }

    pub const fn role(&self) -> NativeClipComponentRole {
        self.role
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub fn keyed_state(&self) -> &BTreeMap<String, Tensor> {
        self.base_weights.tensors()
    }

    pub fn source_scale_matching_operations(
        &self,
        rules: &[(crate::patches::StateKeySelector, f32)],
    ) -> Result<Vec<crate::SemanticPatchOperation>, ClipError> {
        if !matches!(self.encoder, NativeClipEncoder::ClipText(_)) {
            return Err(ClipError::UnsupportedSourceStateTransform(self.role));
        }
        let mut shapes = BTreeMap::new();
        let mut targets = BTreeMap::new();
        for (key, tensor) in self.keyed_state() {
            let source_key = if let Some(layer_key) = key.strip_prefix("layers.") {
                let (layer, parameter) = layer_key.split_once('.').ok_or_else(|| {
                    ClipError::InvalidNativeResource("CLIP layer key is invalid".to_owned())
                })?;
                let parameter = [
                    ("query.", "self_attn.q_proj."),
                    ("key.", "self_attn.k_proj."),
                    ("value.", "self_attn.v_proj."),
                    ("output.", "self_attn.out_proj."),
                    ("layer_norm_1.", "layer_norm1."),
                    ("layer_norm_2.", "layer_norm2."),
                    ("feed_forward_1.", "mlp.fc1."),
                    ("feed_forward_2.", "mlp.fc2."),
                ]
                .into_iter()
                .find_map(|(canonical, source)| {
                    parameter
                        .strip_prefix(canonical)
                        .map(|suffix| format!("{source}{suffix}"))
                })
                .ok_or_else(|| {
                    ClipError::InvalidNativeResource(format!(
                        "CLIP layer key has no source projection: {key}"
                    ))
                })?;
                format!("text_model.encoder.layers.{layer}.{parameter}")
            } else {
                match key.as_str() {
                    "token_embedding.weight" => {
                        "text_model.embeddings.token_embedding.weight".to_owned()
                    }
                    "position_embedding.weight" => {
                        "text_model.embeddings.position_embedding.weight".to_owned()
                    }
                    "final_layer_norm.weight" | "final_layer_norm.bias" => {
                        format!("text_model.{key}")
                    }
                    "text_projection.weight" => key.clone(),
                    _ => {
                        return Err(ClipError::InvalidNativeResource(format!(
                            "CLIP key has no source projection: {key}"
                        )));
                    }
                }
            };
            shapes.insert(source_key.clone(), tensor.descriptor().shape().to_vec());
            targets.insert(source_key, key);
        }
        let mut operations = crate::patches::scale_matching_operations(rules, &shapes)
            .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
        for operation in &mut operations {
            operation.target_key = targets
                .get(&operation.target_key)
                .ok_or_else(|| {
                    ClipError::InvalidNativeResource(
                        "CLIP source key lost its canonical target".to_owned(),
                    )
                })?
                .to_string();
        }
        Ok(operations)
    }

    pub fn patch_graph(&self) -> &Arc<PatchGraph> {
        &self.patch_graph
    }

    fn bind_recipe_source(
        mut self,
        architecture: NativeClipRecipeArchitecture,
        configuration_sha256: Option<&str>,
        parser_limits: &ParserLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        parser_limits
            .validate()
            .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
        if let Some(configuration_sha256) = configuration_sha256 {
            validate_sha256("CLIP recipe configuration", configuration_sha256)?;
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-clip-recipe-binding.v1");
        hasher.update([native_clip_role_tag(architecture.role())]);
        match configuration_sha256 {
            Some(configuration_sha256) => {
                hasher.update([1]);
                hash_clip_bytes(&mut hasher, configuration_sha256.as_bytes())?;
            }
            None => hasher.update([0]),
        }
        hasher.update(parser_limits.version.to_be_bytes());
        hasher.update(parser_limits.manifest_bytes.to_be_bytes());
        hasher.update(parser_limits.maximum_depth.to_be_bytes());
        hasher.update(parser_limits.maximum_tensors.to_be_bytes());
        hasher.update(parser_limits.maximum_tensor_bytes.to_be_bytes());
        hasher.update(parser_limits.maximum_aggregate_tensor_bytes.to_be_bytes());
        hasher.update(parser_limits.maximum_name_bytes.to_be_bytes());
        hasher.update(parser_limits.maximum_archive_entries.to_be_bytes());
        hasher.update(parser_limits.maximum_metadata_values.to_be_bytes());
        self.recipe_binding_sha256 = Some(format!("{:x}", hasher.finalize()));
        self.semantic_digest_sha256 = self.project_semantic_digest(cancellation)?;
        Ok(self)
    }

    fn materialize(
        &self,
        patch: &PatchGraph,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        patch
            .identity()
            .validate_for_base(self.base_weights.base_artifact_digest())?;
        let patch = self
            .patch_graph
            .append_semantic(patch.semantic_operations().iter().cloned())?;
        let unpatched = self.base_weights.unpatched_for_reconstruction();
        let patched = Arc::new(patch.apply(backend, &unpatched, context)?);
        let encoder = match &self.encoder {
            NativeClipEncoder::ClipText(owner) => NativeClipEncoder::ClipText(Arc::new(
                owner.reconstruct_from_mapped_weights(&patched, context.cancellation)?,
            )),
            NativeClipEncoder::Bidirectional(owner) => NativeClipEncoder::Bidirectional(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&patched, context.cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
            NativeClipEncoder::Decoder(owner) => NativeClipEncoder::Decoder(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&patched, context.cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
            NativeClipEncoder::Qwen25(owner) => NativeClipEncoder::Qwen25(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&patched, context.cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
            NativeClipEncoder::Gemma4(owner) => NativeClipEncoder::Gemma4(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&patched, context.cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
        };
        let mut component = Self {
            role: self.role,
            artifact_sha256: self.artifact_sha256.clone(),
            base_weights: patched,
            patch_graph: Arc::new(patch),
            tokenizer: self.tokenizer.clone(),
            encoder,
            options: self.options.clone(),
            recipe_binding_sha256: self.recipe_binding_sha256.clone(),
            semantic_digest_sha256: String::new(),
        };
        component.semantic_digest_sha256 =
            component.project_semantic_digest(context.cancellation)?;
        Ok(component)
    }

    fn rebind_patch_graph(
        &self,
        patch: &PatchGraph,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        patch
            .identity()
            .validate_for_base(self.base_weights.base_artifact_digest())?;
        let unpatched = self.base_weights.unpatched_for_reconstruction();
        let patched = Arc::new(patch.apply(backend, &unpatched, context)?);
        let mut component = self.clone();
        component.base_weights = patched;
        component.patch_graph = Arc::new(patch.clone());
        component = component.restart(context.cancellation)?;
        component.semantic_digest_sha256 =
            component.project_semantic_digest(context.cancellation)?;
        Ok(component)
    }

    fn restart(&self, cancellation: &CancellationToken) -> Result<Self, ClipError> {
        let encoder = match &self.encoder {
            NativeClipEncoder::ClipText(owner) => NativeClipEncoder::ClipText(Arc::new(
                owner.reconstruct_from_mapped_weights(&self.base_weights, cancellation)?,
            )),
            NativeClipEncoder::Bidirectional(owner) => NativeClipEncoder::Bidirectional(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&self.base_weights, cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
            NativeClipEncoder::Decoder(owner) => NativeClipEncoder::Decoder(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&self.base_weights, cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
            NativeClipEncoder::Qwen25(owner) => NativeClipEncoder::Qwen25(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&self.base_weights, cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
            NativeClipEncoder::Gemma4(owner) => NativeClipEncoder::Gemma4(Arc::new(
                owner
                    .reconstruct_from_mapped_weights(&self.base_weights, cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )),
        };
        let mut component = Self::from_sealed_owner(
            self.role,
            self.base_weights.clone(),
            self.tokenizer.clone(),
            encoder,
            self.options.clone(),
            cancellation,
        )?;
        component.recipe_binding_sha256 = self.recipe_binding_sha256.clone();
        component.patch_graph = self.patch_graph.clone();
        component.semantic_digest_sha256 = component.project_semantic_digest(cancellation)?;
        Ok(component)
    }

    fn project_semantic_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.native-clip-component.v2");
        hasher.update([native_clip_role_tag(self.role)]);
        hash_clip_bytes(&mut hasher, self.artifact_sha256.as_bytes())?;
        hash_clip_bytes(&mut hasher, self.base_weights.cache_identity().as_bytes())?;
        hash_clip_bytes(
            &mut hasher,
            self.patch_graph.identity().ordered_digest.as_bytes(),
        )?;
        match &self.recipe_binding_sha256 {
            Some(binding) => {
                hasher.update([1]);
                hash_clip_bytes(&mut hasher, binding.as_bytes())?;
            }
            None => hasher.update([0]),
        }
        hash_clip_bytes(
            &mut hasher,
            self.tokenizer
                .semantic_digest(cancellation)
                .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                .as_bytes(),
        )?;
        hash_clip_component_options(&mut hasher, &self.options)?;
        match &self.encoder {
            NativeClipEncoder::ClipText(owner) => hash_clip_bytes(
                &mut hasher,
                owner.semantic_state_digest(cancellation)?.as_bytes(),
            )?,
            NativeClipEncoder::Bidirectional(owner) => hash_clip_bytes(
                &mut hasher,
                owner
                    .semantic_state_digest(cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                    .as_bytes(),
            )?,
            NativeClipEncoder::Decoder(owner) => hash_clip_bytes(
                &mut hasher,
                owner
                    .semantic_state_digest(cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                    .as_bytes(),
            )?,
            NativeClipEncoder::Qwen25(owner) => hash_clip_bytes(
                &mut hasher,
                owner
                    .semantic_state_digest(cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                    .as_bytes(),
            )?,
            NativeClipEncoder::Gemma4(owner) => hash_clip_bytes(
                &mut hasher,
                owner
                    .semantic_state_digest(cancellation)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                    .as_bytes(),
            )?,
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn resident_tensor_allocations(&self) -> Result<Vec<(sim_tensor::StorageId, u64)>, ClipError> {
        Ok(match &self.encoder {
            NativeClipEncoder::ClipText(owner) => owner.resident_tensor_allocations(),
            NativeClipEncoder::Bidirectional(owner) => owner.resident_tensor_allocations(),
            NativeClipEncoder::Decoder(owner) => owner.resident_tensor_allocations(),
            NativeClipEncoder::Qwen25(owner) => owner
                .resident_tensor_allocations()
                .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            NativeClipEncoder::Gemma4(owner) => owner
                .resident_tensor_allocations()
                .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
        })
    }

    fn structural_owned_bytes(&self) -> Result<u64, ClipError> {
        u64::try_from(mem::size_of::<Self>())
            .map_err(|_| ClipError::Overflow("native CLIP component residency"))?
            .checked_add(
                u64::try_from(self.artifact_sha256.capacity())
                    .map_err(|_| ClipError::Overflow("native CLIP artifact identity"))?,
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    self.recipe_binding_sha256
                        .as_ref()
                        .and_then(|binding| u64::try_from(binding.capacity()).ok())
                        .unwrap_or(0),
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(self.semantic_digest_sha256.capacity()).ok()?)
            })
            .ok_or(ClipError::Overflow("native CLIP component residency"))
    }

    fn encoder_owned_allocations(&self) -> Result<Vec<NativeClipResidentAllocation>, ClipError> {
        let allocation = |address, resident_bytes| NativeClipResidentAllocation {
            kind: NativeClipResidentOwnerKind::Encoder,
            address,
            resident_bytes,
        };
        match &self.encoder {
            NativeClipEncoder::ClipText(owner) => Ok(vec![allocation(
                Arc::as_ptr(owner) as *const () as usize,
                owner.resident_owned_bytes()?,
            )]),
            NativeClipEncoder::Bidirectional(owner) => Ok(vec![allocation(
                Arc::as_ptr(owner) as *const () as usize,
                owner
                    .resident_owned_bytes()
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )]),
            NativeClipEncoder::Decoder(owner) => Ok(vec![allocation(
                Arc::as_ptr(owner) as *const () as usize,
                owner
                    .resident_owned_bytes()
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
            )]),
            NativeClipEncoder::Qwen25(owner) => Ok(owner
                .resident_owned_allocations()
                .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                .into_iter()
                .map(native_clip_multimodal_owned_allocation)
                .collect()),
            NativeClipEncoder::Gemma4(owner) => Ok(owner
                .resident_owned_allocations()
                .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
                .into_iter()
                .map(native_clip_multimodal_owned_allocation)
                .collect()),
        }
    }
}

fn native_clip_multimodal_owned_allocation(
    owner: crate::clip_text_encoder_multimodal::NativeMultimodalOwnedAllocation,
) -> NativeClipResidentAllocation {
    NativeClipResidentAllocation {
        kind: if owner.kind() == NativeMultimodalOwnedAllocationKind::Tokenizer {
            NativeClipResidentOwnerKind::Tokenizer
        } else {
            NativeClipResidentOwnerKind::Encoder
        },
        address: owner.address(),
        resident_bytes: owner.resident_bytes(),
    }
}

fn mapped_weights_owned_allocations(
    mapped: &Arc<MappedModelWeights>,
) -> Result<Vec<NativeClipResidentAllocation>, ClipError> {
    Ok(mapped
        .resident_owned_allocations()
        .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?
        .into_iter()
        .map(|allocation| NativeClipResidentAllocation {
            kind: NativeClipResidentOwnerKind::MappedWeights,
            address: allocation.address(),
            resident_bytes: allocation.resident_bytes(),
        })
        .collect())
}

fn source_clip_layer_selection(
    original: &NativeClipLayerSelection,
    layer_count: usize,
    source_layer: i64,
) -> Result<NativeClipLayerSelection, ClipError> {
    if *original == NativeClipLayerSelection::All {
        return Ok(NativeClipLayerSelection::All);
    }
    let layer_count =
        u64::try_from(layer_count).map_err(|_| ClipError::Overflow("source CLIP layer count"))?;
    if source_layer.unsigned_abs() > layer_count {
        Ok(NativeClipLayerSelection::Last)
    } else {
        Ok(NativeClipLayerSelection::Hidden(
            isize::try_from(source_layer)
                .map_err(|_| ClipError::Overflow("source CLIP selected layer"))?,
        ))
    }
}

fn source_clip_component_layer_selection(
    component: &NativeClipComponent,
    source_layer_override: Option<i64>,
) -> Result<NativeClipLayerSelection, ClipError> {
    let Some(source_layer) = source_layer_override else {
        return Ok(component.options.layer.clone());
    };
    let layer_count = match &component.encoder {
        NativeClipEncoder::ClipText(owner) => owner.configuration().layer_count,
        NativeClipEncoder::Bidirectional(owner) => owner.configuration().layer_count,
        NativeClipEncoder::Decoder(owner) => owner.configuration().layer_kinds.len(),
        NativeClipEncoder::Qwen25(owner) => owner.decoder().configuration().layer_kinds.len(),
        NativeClipEncoder::Gemma4(owner) => owner.decoder().configuration().layer_kinds.len(),
    };
    source_clip_layer_selection(&component.options.layer, layer_count, source_layer)
}

#[derive(Clone)]
pub struct NativeClipScheduleSpec {
    start_percent_bits: u64,
    end_percent_bits: u64,
    hook_metadata_json: Vec<u8>,
    component_patches: Vec<Option<PatchGraph>>,
}

impl NativeClipScheduleSpec {
    pub fn checked(
        start_percent: f64,
        end_percent: f64,
        hook_metadata_json: Vec<u8>,
        component_patches: Vec<Option<PatchGraph>>,
    ) -> Result<Self, ClipError> {
        if !start_percent.is_finite()
            || !end_percent.is_finite()
            || !(0.0..=1.0).contains(&start_percent)
            || !(0.0..=1.0).contains(&end_percent)
            || start_percent >= end_percent
            || hook_metadata_json.len() > SD1_MAX_PROMPT_BYTES
            || serde_json::from_slice::<serde_json::Value>(&hook_metadata_json).is_err()
        {
            return Err(ClipError::InvalidNativeResource(
                "CLIP schedule interval or hook metadata is invalid".to_owned(),
            ));
        }
        Ok(Self {
            start_percent_bits: start_percent.to_bits(),
            end_percent_bits: end_percent.to_bits(),
            hook_metadata_json,
            component_patches,
        })
    }
}

#[derive(Clone)]
struct NativeClipScheduleWindow {
    start_percent_bits: u64,
    end_percent_bits: u64,
    hook_metadata_json: Vec<u8>,
    components: Vec<Arc<NativeClipComponent>>,
    component_patches: Vec<Option<PatchGraph>>,
    patch_identities: Vec<Option<PatchGraphIdentity>>,
}

impl NativeClipScheduleWindow {
    fn materialize(
        spec: NativeClipScheduleSpec,
        base: &[Arc<NativeClipComponent>],
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        if spec.component_patches.len() != base.len() {
            return Err(ClipError::InvalidNativeResource(
                "CLIP schedule patch count differs from component count".to_owned(),
            ));
        }
        let mut components = Vec::new();
        let mut patch_identities = Vec::new();
        components
            .try_reserve_exact(base.len())
            .map_err(|_| ClipError::Allocation("scheduled CLIP components"))?;
        patch_identities
            .try_reserve_exact(base.len())
            .map_err(|_| ClipError::Allocation("scheduled CLIP patch identities"))?;
        for (component, patch) in base.iter().zip(&spec.component_patches) {
            context.check()?;
            match patch {
                Some(patch) => {
                    components.push(Arc::new(component.materialize(patch, backend, context)?));
                    patch_identities.push(Some(patch.identity()));
                }
                None => {
                    components.push(component.clone());
                    patch_identities.push(None);
                }
            }
        }
        Ok(Self {
            start_percent_bits: spec.start_percent_bits,
            end_percent_bits: spec.end_percent_bits,
            hook_metadata_json: spec.hook_metadata_json,
            components,
            component_patches: spec.component_patches,
            patch_identities,
        })
    }

    fn start_percent(&self) -> f64 {
        f64::from_bits(self.start_percent_bits)
    }
    fn end_percent(&self) -> f64 {
        f64::from_bits(self.end_percent_bits)
    }

    fn specification(&self) -> NativeClipScheduleSpec {
        NativeClipScheduleSpec {
            start_percent_bits: self.start_percent_bits,
            end_percent_bits: self.end_percent_bits,
            hook_metadata_json: self.hook_metadata_json.clone(),
            component_patches: self.component_patches.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeClipOutput {
    conditioning: Tensor,
    pooled: Option<Tensor>,
    attention_mask: Option<Tensor>,
    conditioning_llama3: Option<Tensor>,
    clip_start_percent_bits: u64,
    clip_end_percent_bits: u64,
    hook_metadata_json: Vec<u8>,
}

impl NativeClipOutput {
    pub fn conditioning(&self) -> &Tensor {
        &self.conditioning
    }
    pub fn pooled(&self) -> Option<&Tensor> {
        self.pooled.as_ref()
    }
    pub fn attention_mask(&self) -> Option<&Tensor> {
        self.attention_mask.as_ref()
    }
    pub fn conditioning_llama3(&self) -> Option<&Tensor> {
        self.conditioning_llama3.as_ref()
    }
    pub fn clip_start_percent(&self) -> f64 {
        f64::from_bits(self.clip_start_percent_bits)
    }
    pub fn clip_end_percent(&self) -> f64 {
        f64::from_bits(self.clip_end_percent_bits)
    }
    pub fn hook_metadata_json(&self) -> &[u8] {
        &self.hook_metadata_json
    }
}

#[derive(Clone, Copy)]
pub enum NativeClipExecutionRequest<'a> {
    Text {
        prompt: &'a str,
    },
    Qwen25 {
        prompt: &'a str,
        prepared_images: &'a [Qwen25PreparedImage],
        custom_template: Option<&'a str>,
    },
    Gemma {
        prompt: &'a str,
        prepared_visuals: &'a [GemmaPreparedVisual],
        prepared_audio: Option<&'a GemmaPreparedAudio>,
        use_default_template: bool,
        thinking: bool,
    },
}

impl<'a> NativeClipExecutionRequest<'a> {
    fn prompt(self) -> &'a str {
        match self {
            Self::Text { prompt } | Self::Qwen25 { prompt, .. } | Self::Gemma { prompt, .. } => {
                prompt
            }
        }
    }
}

#[derive(Clone)]
struct NativeRetainedSd1Clip {
    tokenizer: Arc<Sd1Tokenizer>,
    clip: Arc<LoadedSd1Clip>,
}

#[derive(Clone)]
pub struct NativeClipResource {
    profile: NativeClipProfile,
    retained_sd1: Option<NativeRetainedSd1Clip>,
    source_layer_override: Option<i64>,
    schedule_enabled: bool,
    components: Vec<Arc<NativeClipComponent>>,
    schedule: Vec<NativeClipScheduleWindow>,
    semantic_digest_sha256: String,
}

#[derive(Clone)]
pub struct NativeClipRecipeSource {
    pub source_form: crate::model_family::NativeModelRecipeSourceForm,
    pub artifact_sha256: String,
    pub ordered_state: Vec<(String, Tensor)>,
    pub parser_limits: ParserLimits,
    pub auxiliary_artifacts: Vec<NativeClipRecipeAuxiliaryArtifact>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeClipRecipeAuxiliaryFormat {
    JsonConfiguration,
    JsonTokenizer,
    SentencePiece,
    Tiktoken,
}

#[derive(Clone)]
pub struct NativeClipRecipeAuxiliaryArtifact {
    pub format: NativeClipRecipeAuxiliaryFormat,
    pub target_ordinal: Option<u8>,
    pub artifact_sha256: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeClipRecipeArchitecture {
    ClipL,
    ClipG,
    T5,
    Llama,
}

impl NativeClipRecipeArchitecture {
    const fn role(self) -> NativeClipComponentRole {
        match self {
            Self::ClipL => NativeClipComponentRole::ClipL,
            Self::ClipG => NativeClipComponentRole::ClipG,
            Self::T5 => NativeClipComponentRole::T5,
            Self::Llama => NativeClipComponentRole::Llama,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum NativeClipResidentOwnerKind {
    Resource,
    RetainedSd1Tokenizer,
    RetainedSd1Clip,
    Component,
    Tokenizer,
    Encoder,
    MappedWeights,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeClipResidentAllocation {
    kind: NativeClipResidentOwnerKind,
    address: usize,
    resident_bytes: u64,
}

impl NativeClipResidentAllocation {
    pub const fn kind(&self) -> NativeClipResidentOwnerKind {
        self.kind
    }
    pub const fn address(&self) -> usize {
        self.address
    }
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl NativeClipResource {
    pub fn admit_execution_target(
        &self,
        backend: &dyn TensorBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeModelExecutionAdmission, ClipError> {
        context.check()?;
        if let Some(backing) = &self.retained_sd1 {
            self.validate(context.cancellation)?;
            let cpu = backend
                .cpu_backend()
                .ok_or(NativeOpsError::UnsupportedDevice {
                    device: backend.device(),
                })?;
            backing
                .clip
                .encoder
                .backend
                .validate_scratch_reservation(&context.scratch)?;
            cpu.validate_scratch_reservation(&context.scratch)?;
            let parameters = &backing.clip.encoder.parameters;
            return Ok(admit_native_model_resource_execution(
                NativeModelExecutionRole::Clip,
                self.profile.execution_profile(),
                &self.semantic_digest_sha256,
                &backing.clip.encoder.transformer.execution_requirements(),
                || {
                    let bytes =
                        self.resident_bytes()
                            .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?
                            .checked_sub(
                                backing.clip.resident_storage_bytes().map_err(|error| {
                                    NativeOpsError::InvalidOwned(error.to_string())
                                })?,
                            )
                            .ok_or(NativeOpsError::ResidentBytesOverflow)?;
                    let retained = parameter_tensors(&parameters.tensors);
                    let mut tensors = Vec::new();
                    tensors.try_reserve_exact(retained.len()).map_err(|_| {
                        NativeOpsError::Invalid(
                            "retained SD1 CLIP tensor accounting allocation failed",
                        )
                    })?;
                    for tensor in retained {
                        tensors.push(tensor.clone());
                    }
                    Ok((tensors, bytes))
                },
                backend,
                DeviceId::CPU,
                DType::F32,
                Layout::Contiguous,
                parameters.stream,
                context,
            )?);
        }
        let tensors = self
            .components
            .iter()
            .chain(
                self.schedule
                    .iter()
                    .flat_map(|window| window.components.iter()),
            )
            .flat_map(|component| component.base_weights.tensors().values());
        let (dtype, device, stream) = match tensors.clone().next() {
            Some(tensor) => (
                tensor.descriptor().dtype(),
                tensor.descriptor().device(),
                tensor.descriptor().stream(),
            ),
            // Optional encoders can all be absent; source composition then creates f32 zeros.
            None => (DType::F32, backend.device(), context.stream),
        };
        for tensor in tensors {
            if tensor.descriptor().dtype() != dtype
                || tensor.descriptor().device() != device
                || tensor.descriptor().stream() != stream
                || !tensor.descriptor().is_contiguous()?
            {
                return Err(ClipError::InvalidNativeResource(
                    "native CLIP retained tensors do not share one contiguous execution target"
                        .to_owned(),
                ));
            }
        }
        let mut requirements = native_clip_profile_execution_requirements(self.profile, dtype);
        for component in self.components.iter().chain(
            self.schedule
                .iter()
                .flat_map(|window| window.components.iter()),
        ) {
            requirements.extend(component.encoder.execution_requirements(dtype).iter());
        }
        Ok(admit_native_model_resource_execution(
            NativeModelExecutionRole::Clip,
            self.profile.execution_profile(),
            &self.semantic_digest_sha256,
            &requirements,
            || {
                let structural_resident_bytes = self
                    .resident_owned_allocations()
                    .map_err(|error| NativeOpsError::InvalidOwned(error.to_string()))?
                    .into_iter()
                    .try_fold(0_u64, |total, allocation| {
                        total
                            .checked_add(allocation.resident_bytes())
                            .ok_or(NativeOpsError::ResidentBytesOverflow)
                    })?;
                let components = self.components.iter().chain(
                    self.schedule
                        .iter()
                        .flat_map(|window| window.components.iter()),
                );
                let tensor_count = components.clone().try_fold(0_usize, |total, component| {
                    total
                        .checked_add(component.base_weights.unpatched_tensors().len())
                        .and_then(|value| value.checked_add(component.base_weights.tensors().len()))
                        .ok_or(NativeOpsError::ResidentBytesOverflow)
                })?;
                let mut tensors = Vec::new();
                tensors.try_reserve_exact(tensor_count).map_err(|_| {
                    NativeOpsError::Invalid("CLIP resident tensor accounting allocation failed")
                })?;
                for component in components {
                    tensors.extend(component.base_weights.unpatched_tensors().values().cloned());
                    tensors.extend(component.base_weights.tensors().values().cloned());
                }
                Ok((tensors, structural_resident_bytes))
            },
            backend,
            device,
            dtype,
            Layout::Contiguous,
            stream,
            context,
        )?)
    }

    pub fn from_recipe_sources(
        sources: Vec<NativeClipRecipeSource>,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        context.check()?;
        if sources.is_empty() || sources.len() > 4 {
            return Err(ClipError::InvalidArtifactSet(sources.len()));
        }
        let mut sources = expand_checkpoint_recipe_sources(sources)?;
        for source in &mut sources {
            canonicalize_open_clip_checkpoint_source(source, backend, context)?;
        }
        if sources.is_empty() || sources.len() > 4 {
            return Err(ClipError::InvalidArtifactSet(sources.len()));
        }
        let mut architectures = Vec::new();
        architectures
            .try_reserve_exact(sources.len())
            .map_err(|_| ClipError::Allocation("CLIP recipe architectures"))?;
        for source in &mut sources {
            architectures.push(detect_clip_recipe_architecture(
                source,
                context.cancellation,
            )?);
        }
        let profile = clip_recipe_profile(&architectures)?;
        let mut components = Vec::new();
        components
            .try_reserve_exact(sources.len())
            .map_err(|_| ClipError::Allocation("CLIP recipe components"))?;
        for (source, architecture) in sources.into_iter().zip(architectures) {
            context.check()?;
            let component = match architecture {
                NativeClipRecipeArchitecture::ClipL | NativeClipRecipeArchitecture::ClipG => {
                    clip_text_recipe_component(profile, architecture.role(), source, context)?
                }
                NativeClipRecipeArchitecture::T5 => t5_recipe_component(profile, source, context)?,
                NativeClipRecipeArchitecture::Llama => llama_recipe_component(source, context)?,
            };
            components.push(Arc::new(component));
        }
        Self::checked(profile, false, components, Vec::new(), backend, context)
    }

    pub fn checked(
        profile: NativeClipProfile,
        schedule_enabled: bool,
        components: Vec<Arc<NativeClipComponent>>,
        schedule_specs: Vec<NativeClipScheduleSpec>,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        validate_native_clip_component_order(profile, &components)?;
        let mut schedule = Vec::new();
        schedule
            .try_reserve_exact(schedule_specs.len())
            .map_err(|_| ClipError::Allocation("CLIP schedule"))?;
        for spec in schedule_specs {
            let window =
                NativeClipScheduleWindow::materialize(spec, &components, backend, context)?;
            if schedule
                .last()
                .is_some_and(|previous: &NativeClipScheduleWindow| {
                    previous.end_percent() > window.start_percent()
                })
            {
                return Err(ClipError::InvalidNativeResource(
                    "CLIP schedule windows overlap or are out of order".to_owned(),
                ));
            }
            schedule.push(window);
        }
        let mut resource = Self {
            profile,
            retained_sd1: None,
            source_layer_override: None,
            schedule_enabled,
            components,
            schedule,
            semantic_digest_sha256: String::new(),
        };
        resource.semantic_digest_sha256 = resource.project_semantic_digest(context.cancellation)?;
        resource.validate(context.cancellation)?;
        Ok(resource)
    }

    pub const fn profile(&self) -> NativeClipProfile {
        self.profile
    }
    pub const fn schedule_enabled(&self) -> bool {
        self.schedule_enabled
    }
    pub fn components(&self) -> &[Arc<NativeClipComponent>] {
        &self.components
    }
    pub fn semantic_digest_sha256(&self) -> &str {
        &self.semantic_digest_sha256
    }

    pub fn keyed_state(&self) -> Result<BTreeMap<String, Tensor>, ClipError> {
        if self.retained_sd1.is_some() {
            return Err(ClipError::UnsupportedSourceStateTransform(
                NativeClipComponentRole::ClipL,
            ));
        }
        let mut state = BTreeMap::new();
        for component in &self.components {
            for (key, tensor) in component.keyed_state() {
                if state.insert(key.clone(), tensor.clone()).is_some() {
                    return Err(ClipError::InvalidNativeResource(format!(
                        "CLIP components repeat keyed state {key}"
                    )));
                }
            }
        }
        Ok(state)
    }

    pub fn from_retained_sd1(
        tokenizer: Arc<Sd1Tokenizer>,
        clip: Arc<LoadedSd1Clip>,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        context.check()?;
        let identity = crate::native_node_payload::retained_sd1_clip_identity(&tokenizer, &clip)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        let _metadata_lease = backend.reserve_workspace(
            context,
            u64::try_from(mem::size_of::<Self>())
                .map_err(|_| ClipError::Overflow("retained SD1 CLIP wrapper"))?
                .checked_add(
                    u64::try_from(identity.execution_sha256().len())
                        .map_err(|_| ClipError::Overflow("retained SD1 CLIP identity"))?,
                )
                .ok_or(ClipError::Overflow("retained SD1 CLIP wrapper"))?,
        )?;
        Self {
            profile: NativeClipProfile::Sd1,
            retained_sd1: Some(NativeRetainedSd1Clip { tokenizer, clip }),
            source_layer_override: None,
            schedule_enabled: false,
            components: Vec::new(),
            schedule: Vec::new(),
            semantic_digest_sha256: String::new(),
        }
        .with_source_layer_override(None, context.cancellation)
    }

    pub fn retained_sd1(&self) -> Option<(&Arc<Sd1Tokenizer>, &Arc<LoadedSd1Clip>)> {
        self.retained_sd1
            .as_ref()
            .map(|backing| (&backing.tokenizer, &backing.clip))
    }

    pub fn execute_retained_sd1_batch(
        &self,
        batch: &TokenBatch,
        context: &ExecutionContext<'_>,
    ) -> Result<ClipEncoding, ClipError> {
        self.validate(context.cancellation)?;
        let backing = self.retained_sd1.as_ref().ok_or_else(|| {
            ClipError::InvalidNativeResource(
                "CLIP resource does not retain an SD1 batch owner".to_owned(),
            )
        })?;
        let encoding =
            backing
                .clip
                .execute_with_source_layer(batch, self.source_layer_override, context)?;
        context.check()?;
        Ok(encoding)
    }

    pub const fn source_layer_override(&self) -> Option<i64> {
        self.source_layer_override
    }

    pub fn clone_with_source_layer(
        &self,
        source_layer: i64,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let _metadata_lease = backend.reserve_workspace(context, self.structural_owned_bytes()?)?;
        let mut components = Vec::new();
        components
            .try_reserve_exact(self.components.len())
            .map_err(|_| ClipError::Allocation("source CLIP wrapper components"))?;
        for component in &self.components {
            context.check()?;
            components.push(component.clone());
        }
        let mut schedule = Vec::new();
        schedule
            .try_reserve_exact(self.schedule.len())
            .map_err(|_| ClipError::Allocation("source CLIP wrapper schedule"))?;
        for window in &self.schedule {
            context.check()?;
            schedule.push(window.clone());
        }
        Self {
            profile: self.profile,
            retained_sd1: self.retained_sd1.clone(),
            source_layer_override: None,
            schedule_enabled: self.schedule_enabled,
            components,
            schedule,
            semantic_digest_sha256: self.semantic_digest_sha256.clone(),
        }
        .with_source_layer_override(Some(source_layer), context.cancellation)
    }

    fn with_source_layer_override(
        mut self,
        source_layer_override: Option<i64>,
        cancellation: &CancellationToken,
    ) -> Result<Self, ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        self.source_layer_override = source_layer_override;
        self.semantic_digest_sha256 = self.project_semantic_digest(cancellation)?;
        self.validate(cancellation)?;
        Ok(self)
    }

    pub fn append_component_patches(
        &self,
        component_patches: Vec<PatchGraph>,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        context.check()?;
        if self.retained_sd1.is_some() {
            return Err(ClipError::UnsupportedSourceStateTransform(
                NativeClipComponentRole::ClipL,
            ));
        }

        if component_patches.len() != self.components.len() {
            return Err(ClipError::InvalidNativeResource(
                "CLIP transform patch count differs from component count".to_owned(),
            ));
        }
        let components = self
            .components
            .iter()
            .zip(&component_patches)
            .map(|(component, patch)| component.materialize(patch, backend, context).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let schedule_specs = self
            .schedule
            .iter()
            .map(NativeClipScheduleWindow::specification)
            .collect();
        Self::checked(
            self.profile,
            self.schedule_enabled,
            components,
            schedule_specs,
            backend,
            context,
        )?
        .with_source_layer_override(self.source_layer_override, context.cancellation)
    }

    pub fn rebind_component_patch_graphs(
        &self,
        component_patches: Vec<PatchGraph>,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        context.check()?;
        if self.retained_sd1.is_some() {
            return Err(ClipError::UnsupportedSourceStateTransform(
                NativeClipComponentRole::ClipL,
            ));
        }

        if component_patches.len() != self.components.len() {
            return Err(ClipError::InvalidNativeResource(
                "CLIP transform patch count differs from component count".to_owned(),
            ));
        }
        let components = self
            .components
            .iter()
            .zip(&component_patches)
            .map(|(component, patch)| {
                component
                    .rebind_patch_graph(patch, backend, context)
                    .map(Arc::new)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let schedule_specs = self
            .schedule
            .iter()
            .map(NativeClipScheduleWindow::specification)
            .collect();
        Self::checked(
            self.profile,
            self.schedule_enabled,
            components,
            schedule_specs,
            backend,
            context,
        )?
        .with_source_layer_override(self.source_layer_override, context.cancellation)
    }

    pub fn execute(
        &self,
        backend: &dyn TensorBackend,
        prompt: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<NativeClipOutput>, ClipError> {
        self.execute_request(
            backend,
            NativeClipExecutionRequest::Text { prompt },
            context,
        )
    }

    pub fn execute_request(
        &self,
        backend: &dyn TensorBackend,
        request: NativeClipExecutionRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<NativeClipOutput>, ClipError> {
        self.admit_execution_target(backend, context)?;
        let backend = backend
            .cpu_backend()
            .ok_or(NativeOpsError::UnsupportedDevice {
                device: backend.device(),
            })?;
        self.validate(context.cancellation)?;
        if let Some(backing) = &self.retained_sd1 {
            let NativeClipExecutionRequest::Text { prompt } = request else {
                return Err(ClipError::InvalidNativeResource(
                    "retained SD1 CLIP admits source text requests only".to_owned(),
                ));
            };
            let prompts = vec![vec![WeightedText::checked(prompt, 1.0)?]];
            let batch = backing
                .tokenizer
                .tokenize_batch(&prompts, context.cancellation)?;
            let encoding = self.execute_retained_sd1_batch(&batch, context)?;
            context.check()?;
            return Ok(vec![NativeClipOutput {
                conditioning: encoding.conditioning().clone(),
                pooled: encoding.pooled().cloned(),
                attention_mask: None,
                conditioning_llama3: None,
                clip_start_percent_bits: 0.0_f64.to_bits(),
                clip_end_percent_bits: 1.0_f64.to_bits(),
                hook_metadata_json: Vec::new(),
            }]);
        }
        if self.schedule_enabled && !self.schedule.is_empty() {
            let mut outputs = Vec::new();
            outputs
                .try_reserve_exact(self.schedule.len())
                .map_err(|_| ClipError::Allocation("scheduled CLIP outputs"))?;
            for window in &self.schedule {
                context.check()?;
                outputs.push(execute_native_clip_profile(
                    self.profile,
                    &window.components,
                    self.source_layer_override,
                    backend,
                    request,
                    window.start_percent(),
                    window.end_percent(),
                    &window.hook_metadata_json,
                    context,
                )?);
            }
            return Ok(outputs);
        }
        execute_native_clip_profile(
            self.profile,
            &self.components,
            self.source_layer_override,
            backend,
            request,
            0.0,
            1.0,
            &[],
            context,
        )
        .map(|output| vec![output])
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), ClipError> {
        cancellation.check().map_err(TensorError::from)?;
        if let Some(backing) = &self.retained_sd1 {
            if self.profile != NativeClipProfile::Sd1
                || !self.components.is_empty()
                || !self.schedule.is_empty()
                || self.schedule_enabled
            {
                return Err(ClipError::InvalidNativeResource(
                    "retained SD1 wrapper has foreign component or schedule state".to_owned(),
                ));
            }
            crate::native_node_payload::retained_sd1_clip_identity(
                &backing.tokenizer,
                &backing.clip,
            )
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
            if self.semantic_digest_sha256 != self.project_semantic_digest(cancellation)? {
                return Err(ClipError::InvalidNativeResource(
                    "retained SD1 wrapper identity drifted".to_owned(),
                ));
            }
            cancellation.check().map_err(TensorError::from)?;
            return Ok(());
        }
        validate_native_clip_component_order(self.profile, &self.components)?;
        if self.semantic_digest_sha256 != self.project_semantic_digest(cancellation)? {
            return Err(ClipError::InvalidNativeResource(
                "CLIP resource semantic identity drifted".to_owned(),
            ));
        }
        self.resident_tensor_allocations()?;
        Ok(())
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, ClipError> {
        if let Some(backing) = &self.retained_sd1 {
            return Ok(backing.clip.resident_tensor_allocations());
        }
        let mut allocations = Vec::new();
        for component in self.components.iter().chain(
            self.schedule
                .iter()
                .flat_map(|window| window.components.iter()),
        ) {
            for (storage_id, bytes) in component.resident_tensor_allocations()? {
                if let Some((_, existing)) = allocations
                    .iter()
                    .find(|(candidate, _)| *candidate == storage_id)
                {
                    if *existing != bytes {
                        return Err(ClipError::InvalidNativeResource(
                            "shared CLIP storage changed resident size".to_owned(),
                        ));
                    }
                } else {
                    allocations.push((storage_id, bytes));
                }
            }
        }
        Ok(allocations)
    }

    pub fn resident_bytes(&self) -> Result<u64, ClipError> {
        let bytes = self.resident_owned_allocations()?.into_iter().try_fold(
            0_u64,
            |total, allocation| {
                total
                    .checked_add(allocation.resident_bytes)
                    .ok_or(ClipError::Overflow("CLIP resource residency"))
            },
        )?;
        if self.retained_sd1.is_some() {
            return Ok(bytes);
        }
        self.resident_tensor_allocations()?
            .into_iter()
            .try_fold(bytes, |total, (_, allocation)| {
                total
                    .checked_add(allocation)
                    .ok_or(ClipError::Overflow("CLIP resource residency"))
            })
    }

    pub fn resident_owned_allocations(
        &self,
    ) -> Result<Vec<NativeClipResidentAllocation>, ClipError> {
        let mut allocations = Vec::new();
        let scheduled_component_count =
            self.schedule.iter().try_fold(0_usize, |total, window| {
                total
                    .checked_add(window.components.len())
                    .ok_or(ClipError::Overflow("CLIP resident owners"))
            })?;
        let owner_bound = self
            .components
            .len()
            .checked_add(scheduled_component_count)
            .and_then(|value| value.checked_mul(12))
            .and_then(|value| value.checked_add(if self.retained_sd1.is_some() { 3 } else { 1 }))
            .ok_or(ClipError::Overflow("CLIP resident owners"))?;
        allocations
            .try_reserve_exact(owner_bound)
            .map_err(|_| ClipError::Allocation("CLIP resident owners"))?;
        allocations.push(NativeClipResidentAllocation {
            kind: NativeClipResidentOwnerKind::Resource,
            address: self as *const Self as usize,
            resident_bytes: self.structural_owned_bytes()?,
        });
        if let Some(backing) = &self.retained_sd1 {
            allocations.push(NativeClipResidentAllocation {
                kind: NativeClipResidentOwnerKind::RetainedSd1Tokenizer,
                address: Arc::as_ptr(&backing.tokenizer) as usize,
                resident_bytes: backing.tokenizer.resident_bytes()?,
            });
            allocations.push(NativeClipResidentAllocation {
                kind: NativeClipResidentOwnerKind::RetainedSd1Clip,
                address: Arc::as_ptr(&backing.clip) as usize,
                resident_bytes: backing.clip.resident_bytes()?,
            });
            return Ok(allocations);
        }
        let mut component_owners = BTreeSet::new();
        let mut tokenizer_owners = BTreeSet::new();
        let mut encoder_owners = BTreeSet::new();
        let mut mapped_owners = BTreeSet::new();
        for component in self.components.iter().chain(
            self.schedule
                .iter()
                .flat_map(|window| window.components.iter()),
        ) {
            if component_owners.insert(Arc::as_ptr(component) as usize) {
                allocations.push(NativeClipResidentAllocation {
                    kind: NativeClipResidentOwnerKind::Component,
                    address: Arc::as_ptr(component) as usize,
                    resident_bytes: component.structural_owned_bytes()?,
                });
            }
            if tokenizer_owners.insert(Arc::as_ptr(&component.tokenizer) as usize) {
                allocations.push(NativeClipResidentAllocation {
                    kind: NativeClipResidentOwnerKind::Tokenizer,
                    address: Arc::as_ptr(&component.tokenizer) as usize,
                    resident_bytes: component
                        .tokenizer
                        .resident_bytes()
                        .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
                });
            }
            for allocation in component.encoder_owned_allocations()? {
                let owners = match allocation.kind {
                    NativeClipResidentOwnerKind::Tokenizer => &mut tokenizer_owners,
                    NativeClipResidentOwnerKind::Encoder => &mut encoder_owners,
                    _ => {
                        return Err(ClipError::InvalidNativeResource(
                            "CLIP encoder exposed an invalid retained-owner kind".to_owned(),
                        ));
                    }
                };
                if owners.insert(allocation.address) {
                    allocations.push(allocation);
                }
            }
            for allocation in mapped_weights_owned_allocations(&component.base_weights)? {
                if mapped_owners.insert(allocation.address) {
                    allocations.push(allocation);
                }
            }
        }
        Ok(allocations)
    }

    pub fn restart(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ClipError> {
        if let Some(backing) = &self.retained_sd1 {
            self.validate(context.cancellation)?;
            return Self::from_retained_sd1(
                backing.tokenizer.clone(),
                backing.clip.clone(),
                backend,
                context,
            )?
            .with_source_layer_override(self.source_layer_override, context.cancellation);
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(self.components.len())
            .map_err(|_| ClipError::Allocation("restarted CLIP components"))?;
        for component in &self.components {
            context.check()?;
            components.push(Arc::new(component.restart(context.cancellation)?));
        }
        let mut specs = Vec::new();
        specs
            .try_reserve_exact(self.schedule.len())
            .map_err(|_| ClipError::Allocation("restarted CLIP schedule"))?;
        for window in &self.schedule {
            specs.push(NativeClipScheduleSpec {
                start_percent_bits: window.start_percent_bits,
                end_percent_bits: window.end_percent_bits,
                hook_metadata_json: window.hook_metadata_json.clone(),
                component_patches: window.component_patches.clone(),
            });
        }
        Self::checked(
            self.profile,
            self.schedule_enabled,
            components,
            specs,
            backend,
            context,
        )?
        .with_source_layer_override(self.source_layer_override, context.cancellation)
    }

    fn structural_owned_bytes(&self) -> Result<u64, ClipError> {
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| ClipError::Overflow("CLIP resource residency"))?;
        for allocation in [
            self.semantic_digest_sha256.capacity(),
            self.components
                .capacity()
                .checked_mul(mem::size_of::<Arc<NativeClipComponent>>())
                .ok_or(ClipError::Overflow("CLIP component vector"))?,
            self.schedule
                .capacity()
                .checked_mul(mem::size_of::<NativeClipScheduleWindow>())
                .ok_or(ClipError::Overflow("CLIP schedule vector"))?,
        ] {
            bytes = bytes
                .checked_add(
                    u64::try_from(allocation)
                        .map_err(|_| ClipError::Overflow("CLIP resource residency"))?,
                )
                .ok_or(ClipError::Overflow("CLIP resource residency"))?;
        }
        for window in &self.schedule {
            for allocation in [
                window.hook_metadata_json.capacity(),
                window
                    .components
                    .capacity()
                    .checked_mul(mem::size_of::<Arc<NativeClipComponent>>())
                    .ok_or(ClipError::Overflow("CLIP scheduled components"))?,
                window
                    .component_patches
                    .capacity()
                    .checked_mul(mem::size_of::<Option<PatchGraph>>())
                    .ok_or(ClipError::Overflow("CLIP scheduled patches"))?,
                window
                    .patch_identities
                    .capacity()
                    .checked_mul(mem::size_of::<Option<PatchGraphIdentity>>())
                    .ok_or(ClipError::Overflow("CLIP patch identities"))?,
            ] {
                bytes = bytes
                    .checked_add(
                        u64::try_from(allocation)
                            .map_err(|_| ClipError::Overflow("CLIP schedule residency"))?,
                    )
                    .ok_or(ClipError::Overflow("CLIP schedule residency"))?;
            }
            for patch in window.component_patches.iter().flatten() {
                let patch_owned = patch
                    .resident_bytes()?
                    .checked_sub(
                        u64::try_from(mem::size_of::<PatchGraph>())
                            .map_err(|_| ClipError::Overflow("CLIP patch residency"))?,
                    )
                    .ok_or(ClipError::Overflow("CLIP patch residency"))?;
                bytes = bytes
                    .checked_add(patch_owned)
                    .ok_or(ClipError::Overflow("CLIP patch residency"))?;
            }
            for identity in window.patch_identities.iter().flatten() {
                bytes = bytes
                    .checked_add(
                        identity
                            .owned_resident_bytes()
                            .ok_or(ClipError::Overflow("CLIP patch identity residency"))?,
                    )
                    .ok_or(ClipError::Overflow("CLIP patch identity residency"))?;
            }
        }
        Ok(bytes)
    }

    fn project_semantic_digest(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<String, ClipError> {
        let mut hasher = Sha256::new();
        if let Some(backing) = &self.retained_sd1 {
            let identity = crate::native_node_payload::retained_sd1_clip_identity(
                &backing.tokenizer,
                &backing.clip,
            )
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
            if self.source_layer_override.is_none() {
                cancellation.check().map_err(TensorError::from)?;
                return Ok(identity.execution_sha256().to_owned());
            }
            hasher.update(b"zed.sim.native-clip-retained-sd1.v1");
            hash_clip_bytes(&mut hasher, identity.execution_sha256().as_bytes())?;
        } else {
            hasher.update(b"zed.sim.native-clip-resource.v2");
            hasher.update([
                native_clip_profile_tag(self.profile),
                u8::from(self.schedule_enabled),
            ]);
            for component in &self.components {
                hash_clip_bytes(&mut hasher, component.semantic_digest_sha256.as_bytes())?;
            }
            for window in &self.schedule {
                hasher.update(window.start_percent_bits.to_be_bytes());
                hasher.update(window.end_percent_bits.to_be_bytes());
                hash_clip_bytes(&mut hasher, &window.hook_metadata_json)?;
                for identity in &window.patch_identities {
                    match identity {
                        Some(identity) => {
                            hasher.update([1]);
                            hash_clip_bytes(&mut hasher, identity.ordered_digest.as_bytes())?;
                        }
                        None => hasher.update([0]),
                    }
                }
                for component in &window.components {
                    hash_clip_bytes(&mut hasher, component.semantic_digest_sha256.as_bytes())?;
                }
            }
        }
        if let Some(source_layer) = self.source_layer_override {
            hasher.update(b"zed.sim.native-clip-source-layer.v1");
            hasher.update(source_layer.to_be_bytes());
        }
        cancellation.check().map_err(TensorError::from)?;
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn expand_checkpoint_recipe_sources(
    sources: Vec<NativeClipRecipeSource>,
) -> Result<Vec<NativeClipRecipeSource>, ClipError> {
    let mut expanded = Vec::new();
    for source in sources {
        if source.source_form != crate::model_family::NativeModelRecipeSourceForm::Checkpoint {
            if source
                .auxiliary_artifacts
                .iter()
                .any(|artifact| artifact.target_ordinal.is_some_and(|ordinal| ordinal != 0))
            {
                return Err(ClipError::InvalidNativeResource(
                    "standalone CLIP source targets an auxiliary artifact at a foreign ordinal"
                        .to_owned(),
                ));
            }
            expanded.push(source);
            continue;
        }
        let has_cond_stage = source.ordered_state.iter().any(|(name, _)| {
            name.strip_prefix("cond_stage_model.transformer.")
                .is_some_and(clip_recipe_architecture_key)
        });
        let mut embedder_ordinals = BTreeSet::new();
        for (name, _) in &source.ordered_state {
            let Some(rest) = name.strip_prefix("conditioner.embedders.") else {
                continue;
            };
            let Some((ordinal, projected, open_clip)) = rest
                .split_once(".transformer.")
                .map(|(ordinal, projected)| (ordinal, projected, false))
                .or_else(|| {
                    rest.split_once(".model.")
                        .map(|(ordinal, projected)| (ordinal, projected, true))
                })
            else {
                continue;
            };
            if !ordinal.is_empty()
                && ordinal.bytes().all(|byte| byte.is_ascii_digit())
                && if open_clip {
                    open_clip_recipe_architecture_key(projected)
                } else {
                    clip_recipe_architecture_key(projected)
                }
            {
                embedder_ordinals.insert(ordinal.parse::<usize>().map_err(|_| {
                    ClipError::InvalidNativeResource(
                        "CLIP checkpoint embedder ordinal overflows".to_owned(),
                    )
                })?);
            }
        }
        if has_cond_stage && !embedder_ordinals.is_empty() {
            return Err(ClipError::InvalidNativeResource(
                "CLIP checkpoint mixes cond-stage and conditioner encoder layouts".to_owned(),
            ));
        }
        if !embedder_ordinals.is_empty()
            && embedder_ordinals
                .iter()
                .copied()
                .ne(0..embedder_ordinals.len())
        {
            return Err(ClipError::InvalidNativeResource(
                "CLIP checkpoint embedder ordinals are not contiguous".to_owned(),
            ));
        }
        if embedder_ordinals.len() <= 1 {
            if source
                .auxiliary_artifacts
                .iter()
                .any(|artifact| artifact.target_ordinal.is_some_and(|ordinal| ordinal != 0))
            {
                return Err(ClipError::InvalidNativeResource(
                    "CLIP checkpoint targets an auxiliary artifact at a missing encoder ordinal"
                        .to_owned(),
                ));
            }
            expanded.push(source);
            continue;
        }
        let encoder_count = embedder_ordinals.len();
        for ordinal in embedder_ordinals {
            let root = format!("conditioner.embedders.{ordinal}.");
            let mut projected = source.clone();
            projected
                .ordered_state
                .retain(|(name, _)| name.starts_with(&root));
            projected.auxiliary_artifacts.retain(|artifact| {
                artifact.target_ordinal.is_none()
                    || artifact.target_ordinal == u8::try_from(ordinal).ok()
            });
            expanded.push(projected);
        }
        if source.auxiliary_artifacts.iter().any(|artifact| {
            artifact
                .target_ordinal
                .is_some_and(|ordinal| usize::from(ordinal) >= encoder_count)
        }) {
            return Err(ClipError::InvalidNativeResource(
                "CLIP checkpoint targets an auxiliary artifact at a missing encoder ordinal"
                    .to_owned(),
            ));
        }
    }
    Ok(expanded)
}

fn clip_recipe_architecture_key(name: &str) -> bool {
    name.starts_with("text_model.")
        || name == "text_projection.weight"
        || name == "logit_scale"
        || name == "shared.weight"
        || name.starts_with("encoder.")
        || name.starts_with("model.")
        || name == "lm_head.weight"
}

fn open_clip_recipe_architecture_key(name: &str) -> bool {
    name == "token_embedding.weight"
        || name == "positional_embedding"
        || name.starts_with("transformer.resblocks.")
        || name.starts_with("ln_final.")
        || name == "text_projection"
        || name == "text_projection.weight"
        || name == "logit_scale"
}

fn canonicalize_open_clip_checkpoint_source(
    source: &mut NativeClipRecipeSource,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(), ClipError> {
    if source.source_form != crate::model_family::NativeModelRecipeSourceForm::Checkpoint {
        return Ok(());
    }
    let mut prefixes = BTreeSet::new();
    for (name, _) in &source.ordered_state {
        let Some(rest) = name.strip_prefix("conditioner.embedders.") else {
            continue;
        };
        let Some((ordinal, projected)) = rest.split_once(".model.") else {
            continue;
        };
        if !ordinal.is_empty()
            && ordinal.bytes().all(|byte| byte.is_ascii_digit())
            && open_clip_recipe_architecture_key(projected)
        {
            prefixes.insert(format!("conditioner.embedders.{ordinal}.model."));
        }
    }
    if prefixes.is_empty() {
        return Ok(());
    }
    if prefixes.len() != 1 {
        return Err(ClipError::InvalidNativeResource(
            "checkpoint contains multiple unsplit OpenCLIP encoders".to_owned(),
        ));
    }
    let prefix = prefixes
        .into_iter()
        .next()
        .ok_or_else(|| ClipError::InvalidNativeResource("OpenCLIP prefix is missing".to_owned()))?;
    let canonical_prefix = prefix
        .strip_suffix("model.")
        .map(|root| format!("{root}transformer."))
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("OpenCLIP prefix is malformed".to_owned())
        })?;
    let mut canonical = BTreeMap::new();
    for (name, tensor) in std::mem::take(&mut source.ordered_state) {
        if let Some(projected) = name.strip_prefix(&prefix) {
            project_open_clip_recipe_tensor(projected, tensor, &mut canonical, backend, context)?;
        } else if name.starts_with("cond_stage_model.")
            || name.starts_with("conditioner.embedders.")
        {
            return Err(ClipError::InvalidNativeResource(
                "OpenCLIP checkpoint contains a disallowed text-encoder sibling".to_owned(),
            ));
        }
    }
    source.ordered_state = canonical
        .into_iter()
        .map(|(name, tensor)| (format!("{canonical_prefix}{name}"), tensor))
        .collect();
    Ok(())
}

fn clip_recipe_profile(
    architectures: &[NativeClipRecipeArchitecture],
) -> Result<NativeClipProfile, ClipError> {
    let roles = architectures
        .iter()
        .map(|architecture| architecture.role())
        .collect::<Vec<_>>();
    let is_ordered_subset = |expected: &[NativeClipComponentRole]| {
        roles
            .iter()
            .try_fold(0_usize, |offset, role| {
                expected
                    .get(offset..)?
                    .iter()
                    .position(|candidate| candidate == role)
                    .map(|position| offset + position + 1)
            })
            .is_some()
    };
    if roles == [NativeClipComponentRole::ClipL] {
        return Ok(NativeClipProfile::Sd1);
    }
    if !roles.contains(&NativeClipComponentRole::T5)
        && !roles.contains(&NativeClipComponentRole::Llama)
        && is_ordered_subset(&[
            NativeClipComponentRole::ClipL,
            NativeClipComponentRole::ClipG,
        ])
    {
        return Ok(NativeClipProfile::Sdxl);
    }
    if roles == [NativeClipComponentRole::T5] {
        return Ok(NativeClipProfile::PixArt);
    }
    if !roles.contains(&NativeClipComponentRole::Llama)
        && is_ordered_subset(&[
            NativeClipComponentRole::ClipL,
            NativeClipComponentRole::ClipG,
            NativeClipComponentRole::T5,
        ])
    {
        return Ok(NativeClipProfile::Sd3);
    }
    if roles.contains(&NativeClipComponentRole::Llama)
        && is_ordered_subset(&[
            NativeClipComponentRole::ClipL,
            NativeClipComponentRole::ClipG,
            NativeClipComponentRole::T5,
            NativeClipComponentRole::Llama,
        ])
    {
        return Ok(NativeClipProfile::HiDream);
    }
    Err(ClipError::InvalidNativeResource(
        "ordered CLIP recipe architectures are unsupported or ambiguous".to_owned(),
    ))
}

fn detect_clip_recipe_architecture(
    source: &NativeClipRecipeSource,
    cancellation: &CancellationToken,
) -> Result<NativeClipRecipeArchitecture, ClipError> {
    cancellation.check().map_err(TensorError::from)?;
    validate_sha256("native CLIP recipe artifact", &source.artifact_sha256)?;
    validate_clip_recipe_parser_limits(&source.parser_limits, None, &source.auxiliary_artifacts)?;
    let state = checked_recipe_source_state(
        source.source_form,
        source.ordered_state.clone(),
        "CLIP architecture",
    )?;
    let has_clip = state.contains_key("text_model.embeddings.token_embedding.weight")
        || state.contains_key("text_model.encoder.layers.0.layer_norm1.weight");
    let has_t5 = state.contains_key("shared.weight")
        || state.contains_key("encoder.block.0.layer.0.SelfAttention.q.weight");
    let has_llama = state.contains_key("model.embed_tokens.weight")
        || state.contains_key("model.layers.0.self_attn.q_proj.weight");
    if usize::from(has_clip) + usize::from(has_t5) + usize::from(has_llama) != 1 {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe state has an ambiguous or incomplete architecture".to_owned(),
        ));
    }
    if has_t5 {
        recipe_configuration_artifact(source, NativeClipRecipeArchitecture::T5, true)?;
        return Ok(NativeClipRecipeArchitecture::T5);
    }
    if has_llama {
        recipe_configuration_artifact(source, NativeClipRecipeArchitecture::Llama, true)?;
        return Ok(NativeClipRecipeArchitecture::Llama);
    }
    let embedding = state
        .get("text_model.embeddings.token_embedding.weight")
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(
                "CLIP recipe state omits its token embedding".to_owned(),
            )
        })?;
    let hidden_size = embedding
        .descriptor()
        .shape()
        .get(1)
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(
                "CLIP recipe token embedding shape is invalid".to_owned(),
            )
        })?;
    let configuration =
        recipe_configuration_artifact(source, NativeClipRecipeArchitecture::ClipL, false)?;
    if let Some((document, _)) = configuration {
        if let Some(configured_hidden) = document
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            && configured_hidden != hidden_size
        {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe config hidden size differs from verified state".to_owned(),
            ));
        }
        return match document
            .get("projection_dim")
            .and_then(serde_json::Value::as_u64)
        {
            Some(768) => Ok(NativeClipRecipeArchitecture::ClipL),
            Some(1_280) => Ok(NativeClipRecipeArchitecture::ClipG),
            Some(_) => Err(ClipError::InvalidNativeResource(
                "CLIP recipe config projection dimension is unsupported".to_owned(),
            )),
            None if hidden_size == 768 => Ok(NativeClipRecipeArchitecture::ClipL),
            None if hidden_size == 1_280 => Ok(NativeClipRecipeArchitecture::ClipG),
            None => Err(ClipError::InvalidNativeResource(
                "CLIP recipe state does not identify CLIP-L or CLIP-G".to_owned(),
            )),
        };
    }
    match hidden_size {
        768 => Ok(NativeClipRecipeArchitecture::ClipL),
        1_280 => Ok(NativeClipRecipeArchitecture::ClipG),
        _ => Err(ClipError::InvalidNativeResource(
            "CLIP recipe state does not identify CLIP-L or CLIP-G".to_owned(),
        )),
    }
}

fn recipe_configuration_artifact(
    source: &NativeClipRecipeSource,
    architecture: NativeClipRecipeArchitecture,
    required: bool,
) -> Result<Option<(serde_json::Value, &NativeClipRecipeAuxiliaryArtifact)>, ClipError> {
    let mut selected = None;
    for artifact in source
        .auxiliary_artifacts
        .iter()
        .filter(|artifact| artifact.format == NativeClipRecipeAuxiliaryFormat::JsonConfiguration)
    {
        validate_sha256("CLIP recipe config artifact", &artifact.artifact_sha256)?;
        if format!("{:x}", Sha256::digest(&artifact.bytes)) != artifact.artifact_sha256 {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe config artifact digest changed".to_owned(),
            ));
        }
        let document = crate::formats::parse_strict_json(&artifact.bytes, &source.parser_limits)
            .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
        let matches = match architecture {
            NativeClipRecipeArchitecture::ClipL | NativeClipRecipeArchitecture::ClipG => {
                document.get("hidden_size").is_some()
                    && (document.get("projection_dim").is_some()
                        || document.get("model_type").is_some())
            }
            NativeClipRecipeArchitecture::T5 => {
                document.get("d_model").is_some()
                    && document.get("num_layers").is_some()
                    && document.get("d_ff").is_some()
            }
            NativeClipRecipeArchitecture::Llama => {
                document.get("hidden_size").is_some()
                    && document.get("num_hidden_layers").is_some()
                    && document.get("num_key_value_heads").is_some()
            }
        };
        if matches && selected.replace((document, artifact)).is_some() {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe contains multiple matching semantic configs".to_owned(),
            ));
        }
    }
    if required && selected.is_none() {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe omitted its checked semantic config".to_owned(),
        ));
    }
    Ok(selected)
}

fn clip_text_recipe_component(
    profile: NativeClipProfile,
    role: NativeClipComponentRole,
    source: NativeClipRecipeSource,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipComponent, ClipError> {
    let configuration_sha256 =
        recipe_configuration_artifact(&source, NativeClipRecipeArchitecture::ClipL, false)?
            .map(|(_, artifact)| artifact.artifact_sha256.clone());
    let NativeClipRecipeSource {
        source_form,
        artifact_sha256,
        ordered_state,
        parser_limits,
        mut auxiliary_artifacts,
    } = source;
    if let Some(configuration_sha256) = &configuration_sha256 {
        remove_clip_recipe_auxiliary_identity(&mut auxiliary_artifacts, configuration_sha256)?;
    }
    validate_clip_recipe_parser_limits(&parser_limits, None, &auxiliary_artifacts)?;
    validate_sha256("native CLIP recipe artifact", &artifact_sha256)?;
    let mut source_state = checked_recipe_source_state(source_form, ordered_state, "CLIP")?;
    let token_embedding = take_clip_recipe_tensor(
        &mut source_state,
        "text_model.embeddings.token_embedding.weight",
    )?;
    let position_embedding = take_clip_recipe_tensor(
        &mut source_state,
        "text_model.embeddings.position_embedding.weight",
    )?;
    let token_shape = token_embedding.descriptor().shape();
    let position_shape = position_embedding.descriptor().shape();
    if token_shape.len() != 2 || position_shape.len() != 2 || token_shape[1] != position_shape[1] {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe embedding shapes are invalid".to_owned(),
        ));
    }
    let vocabulary_size = usize::try_from(token_shape[0])
        .map_err(|_| ClipError::Overflow("CLIP recipe vocabulary"))?;
    let hidden_size = usize::try_from(token_shape[1])
        .map_err(|_| ClipError::Overflow("CLIP recipe hidden width"))?;
    let maximum_length = usize::try_from(position_shape[0])
        .map_err(|_| ClipError::Overflow("CLIP recipe context length"))?;
    let mut layers = Vec::new();
    let mut canonical_state = BTreeMap::new();
    insert_clip_recipe_tensor(
        &mut canonical_state,
        "token_embedding.weight",
        token_embedding.clone(),
    )?;
    insert_clip_recipe_tensor(
        &mut canonical_state,
        "position_embedding.weight",
        position_embedding.clone(),
    )?;
    let mut layer = 0_usize;
    while source_state.contains_key(&format!(
        "text_model.encoder.layers.{layer}.layer_norm1.weight"
    )) {
        context.check()?;
        let prefix = format!("text_model.encoder.layers.{layer}");
        let canonical = format!("layers.{layer}");
        let layer_weights = ClipTextLayerWeights {
            layer_norm_1_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.layer_norm1.weight"),
                &format!("{canonical}.layer_norm_1.weight"),
            )?,
            layer_norm_1_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.layer_norm1.bias"),
                &format!("{canonical}.layer_norm_1.bias"),
            )?,
            query_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &format!("{canonical}.query.weight"),
            )?,
            query_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.q_proj.bias"),
                &format!("{canonical}.query.bias"),
            )?,
            key_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &format!("{canonical}.key.weight"),
            )?,
            key_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.k_proj.bias"),
                &format!("{canonical}.key.bias"),
            )?,
            value_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &format!("{canonical}.value.weight"),
            )?,
            value_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.v_proj.bias"),
                &format!("{canonical}.value.bias"),
            )?,
            output_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.out_proj.weight"),
                &format!("{canonical}.output.weight"),
            )?,
            output_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.self_attn.out_proj.bias"),
                &format!("{canonical}.output.bias"),
            )?,
            layer_norm_2_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.layer_norm2.weight"),
                &format!("{canonical}.layer_norm_2.weight"),
            )?,
            layer_norm_2_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.layer_norm2.bias"),
                &format!("{canonical}.layer_norm_2.bias"),
            )?,
            feed_forward_1_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.mlp.fc1.weight"),
                &format!("{canonical}.feed_forward_1.weight"),
            )?,
            feed_forward_1_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.mlp.fc1.bias"),
                &format!("{canonical}.feed_forward_1.bias"),
            )?,
            feed_forward_2_weight: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.mlp.fc2.weight"),
                &format!("{canonical}.feed_forward_2.weight"),
            )?,
            feed_forward_2_bias: take_clip_recipe_mapped(
                &mut source_state,
                &mut canonical_state,
                &format!("{prefix}.mlp.fc2.bias"),
                &format!("{canonical}.feed_forward_2.bias"),
            )?,
        };
        layers.push(layer_weights);
        layer = layer
            .checked_add(1)
            .ok_or(ClipError::Overflow("CLIP recipe layers"))?;
    }
    if layers.is_empty() {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe has no transformer layers".to_owned(),
        ));
    }
    let final_layer_norm_weight = take_clip_recipe_mapped(
        &mut source_state,
        &mut canonical_state,
        "text_model.final_layer_norm.weight",
        "final_layer_norm.weight",
    )?;
    let final_layer_norm_bias = take_clip_recipe_mapped(
        &mut source_state,
        &mut canonical_state,
        "text_model.final_layer_norm.bias",
        "final_layer_norm.bias",
    )?;
    let text_projection = source_state.remove("text_projection.weight");
    if let Some(projection) = &text_projection {
        insert_clip_recipe_tensor(
            &mut canonical_state,
            "text_projection.weight",
            projection.clone(),
        )?;
    }
    source_state.remove("logit_scale");
    if let Some(unexpected) = source_state.keys().next() {
        return Err(ClipError::InvalidNativeResource(format!(
            "CLIP recipe contains unexpected tensor {unexpected}"
        )));
    }
    let intermediate_size = layers
        .first()
        .and_then(|layer| layer.feed_forward_1_weight.descriptor().shape().first())
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("CLIP recipe feed-forward shape is invalid".to_owned())
        })?;
    let attention_heads = match role {
        NativeClipComponentRole::ClipL if hidden_size == 768 => 12,
        NativeClipComponentRole::ClipG if hidden_size == 1_280 => 20,
        _ if hidden_size < 64 => 1,
        _ if hidden_size.is_multiple_of(64) => hidden_size / 64,
        _ => {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe hidden width has no checked attention-head projection".to_owned(),
            ));
        }
    };
    let configuration = ClipTextConfiguration {
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size,
        max_position_embeddings: maximum_length,
        hidden_size,
        intermediate_size,
        attention_heads,
        layer_count: layers.len(),
        eos_token_id: SD1_END_TOKEN,
        activation: ClipTextActivation::QuickGelu,
        projection_dimension: text_projection.as_ref().and_then(|tensor| {
            tensor
                .descriptor()
                .shape()
                .first()
                .copied()
                .and_then(|value| usize::try_from(value).ok())
        }),
    };
    let template = NativeClipText::new(
        configuration,
        ClipTextWeights {
            token_embedding,
            position_embedding,
            layers,
            final_layer_norm_weight,
            final_layer_norm_bias,
        },
        text_projection,
    )?;
    let tokenizer_artifact =
        take_clip_recipe_tokenizer_json(&mut auxiliary_artifacts, &parser_limits)?;
    validate_remaining_clip_recipe_auxiliary(&auxiliary_artifacts)?;
    let tokenizer = Arc::new(clip_recipe_tokenizer(
        &tokenizer_artifact.bytes,
        profile,
        role,
        hidden_size,
        &parser_limits,
    )?);
    let options = match (profile, role) {
        (NativeClipProfile::Sd1, NativeClipComponentRole::ClipL) => {
            NativeClipComponentOptions::checked_source(
                NativeTokenizerInvocationOptions::default(),
                NativeClipLayerSelection::Last,
                true,
                false,
                false,
                false,
            )?
        }
        (NativeClipProfile::Sdxl | NativeClipProfile::Sd3, NativeClipComponentRole::ClipL) => {
            NativeClipComponentOptions::checked_source(
                NativeTokenizerInvocationOptions::default(),
                NativeClipLayerSelection::Hidden(-2),
                false,
                false,
                false,
                false,
            )?
        }
        (
            NativeClipProfile::Sdxl | NativeClipProfile::Sd3 | NativeClipProfile::HiDream,
            NativeClipComponentRole::ClipG,
        ) => NativeClipComponentOptions::checked_source(
            NativeTokenizerInvocationOptions::default(),
            NativeClipLayerSelection::Hidden(-2),
            false,
            true,
            false,
            false,
        )?,
        (NativeClipProfile::HiDream, NativeClipComponentRole::ClipL) => {
            NativeClipComponentOptions::checked_source(
                NativeTokenizerInvocationOptions::default(),
                NativeClipLayerSelection::Last,
                true,
                true,
                false,
                false,
            )?
        }
        _ => {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe role is not owned by the CLIP text adapter".to_owned(),
            ));
        }
    };
    NativeClipComponent::checked_clip_text(
        role,
        Arc::new(MappedModelWeights::from_parts(
            artifact_sha256,
            canonical_state,
            Vec::new(),
        )),
        tokenizer,
        &template,
        options,
        context.cancellation,
    )?
    .bind_recipe_source(
        match role {
            NativeClipComponentRole::ClipL => NativeClipRecipeArchitecture::ClipL,
            NativeClipComponentRole::ClipG => NativeClipRecipeArchitecture::ClipG,
            _ => {
                return Err(ClipError::InvalidNativeResource(
                    "CLIP text recipe selected a non-CLIP architecture".to_owned(),
                ));
            }
        },
        configuration_sha256.as_deref(),
        &parser_limits,
        context.cancellation,
    )
}

fn take_clip_recipe_tensor(
    state: &mut BTreeMap<String, Tensor>,
    name: &str,
) -> Result<Tensor, ClipError> {
    state.remove(name).ok_or_else(|| {
        ClipError::InvalidNativeResource(format!("CLIP recipe is missing tensor {name}"))
    })
}

fn take_clip_recipe_auxiliary(
    artifacts: &mut Vec<NativeClipRecipeAuxiliaryArtifact>,
    format: NativeClipRecipeAuxiliaryFormat,
) -> Result<NativeClipRecipeAuxiliaryArtifact, ClipError> {
    let mut matching = artifacts
        .iter()
        .enumerate()
        .filter(|(_, artifact)| artifact.format == format);
    let (index, artifact) = matching.next().ok_or_else(|| {
        ClipError::InvalidNativeResource(format!(
            "CLIP recipe omitted its {format:?} auxiliary artifact"
        ))
    })?;
    if matching.next().is_some() {
        return Err(ClipError::InvalidNativeResource(format!(
            "CLIP recipe contains multiple {format:?} auxiliary artifacts"
        )));
    }
    validate_sha256("CLIP recipe auxiliary artifact", &artifact.artifact_sha256)?;
    if format!("{:x}", Sha256::digest(&artifact.bytes)) != artifact.artifact_sha256 {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe auxiliary artifact digest changed".to_owned(),
        ));
    }
    Ok(artifacts.remove(index))
}

fn remove_clip_recipe_auxiliary_identity(
    artifacts: &mut Vec<NativeClipRecipeAuxiliaryArtifact>,
    artifact_sha256: &str,
) -> Result<(), ClipError> {
    let mut matching = artifacts
        .iter()
        .enumerate()
        .filter(|(_, artifact)| artifact.artifact_sha256 == artifact_sha256);
    let index = matching.next().map(|(index, _)| index).ok_or_else(|| {
        ClipError::InvalidNativeResource(
            "CLIP recipe selected a missing auxiliary identity".to_owned(),
        )
    })?;
    if matching.next().is_some() {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe duplicated a selected auxiliary identity".to_owned(),
        ));
    }
    artifacts.remove(index);
    Ok(())
}

fn take_clip_recipe_tokenizer_json(
    artifacts: &mut Vec<NativeClipRecipeAuxiliaryArtifact>,
    parser_limits: &ParserLimits,
) -> Result<NativeClipRecipeAuxiliaryArtifact, ClipError> {
    let mut selected = None;
    for (index, artifact) in artifacts.iter().enumerate() {
        if artifact.format != NativeClipRecipeAuxiliaryFormat::JsonTokenizer {
            continue;
        }
        validate_sha256("CLIP recipe auxiliary artifact", &artifact.artifact_sha256)?;
        if format!("{:x}", Sha256::digest(&artifact.bytes)) != artifact.artifact_sha256 {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe auxiliary artifact digest changed".to_owned(),
            ));
        }
        let Ok(document) = crate::formats::parse_strict_json(&artifact.bytes, parser_limits) else {
            continue;
        };
        let model = document.get("model").unwrap_or(&document);
        if model.get("vocab").is_some()
            && model.get("merges").is_some_and(serde_json::Value::is_array)
        {
            if selected.replace(index).is_some() {
                return Err(ClipError::InvalidNativeResource(
                    "CLIP recipe contains multiple tokenizer model artifacts".to_owned(),
                ));
            }
        }
    }
    let index = selected.ok_or_else(|| {
        ClipError::InvalidNativeResource(
            "CLIP recipe omitted its checked tokenizer model artifact".to_owned(),
        )
    })?;
    Ok(artifacts.remove(index))
}

fn validate_remaining_clip_recipe_auxiliary(
    artifacts: &[NativeClipRecipeAuxiliaryArtifact],
) -> Result<(), ClipError> {
    let mut identities = BTreeSet::new();
    for artifact in artifacts {
        validate_sha256("CLIP recipe auxiliary artifact", &artifact.artifact_sha256)?;
        if format!("{:x}", Sha256::digest(&artifact.bytes)) != artifact.artifact_sha256 {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe auxiliary artifact digest changed".to_owned(),
            ));
        }
        if !identities.insert((artifact.format, artifact.artifact_sha256.as_str())) {
            return Err(ClipError::InvalidNativeResource(
                "CLIP recipe duplicates an auxiliary artifact".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_clip_recipe_parser_limits(
    limits: &ParserLimits,
    configuration_json: Option<&[u8]>,
    artifacts: &[NativeClipRecipeAuxiliaryArtifact],
) -> Result<(), ClipError> {
    limits
        .validate()
        .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
    let maximum = usize::try_from(limits.manifest_bytes)
        .map_err(|_| ClipError::Overflow("CLIP parser manifest limit"))?;
    if configuration_json.is_some_and(|bytes| bytes.len() > maximum)
        || artifacts
            .iter()
            .any(|artifact| artifact.bytes.len() > maximum)
    {
        return Err(ClipError::InvalidNativeResource(
            "CLIP recipe auxiliary artifact exceeds its checked parser limit".to_owned(),
        ));
    }
    Ok(())
}

fn recipe_config_usize(
    configuration: &serde_json::Value,
    name: &'static str,
) -> Result<usize, ClipError> {
    configuration
        .get(name)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(format!(
                "CLIP recipe config field {name} is missing or invalid"
            ))
        })
}

fn recipe_config_f32(
    configuration: &serde_json::Value,
    name: &'static str,
) -> Result<f32, ClipError> {
    configuration
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(format!(
                "CLIP recipe config field {name} is missing or invalid"
            ))
        })
}

fn checked_recipe_source_state(
    source_form: crate::model_family::NativeModelRecipeSourceForm,
    ordered_state: Vec<(String, Tensor)>,
    owner: &'static str,
) -> Result<BTreeMap<String, Tensor>, ClipError> {
    let checkpoint_prefix =
        if source_form == crate::model_family::NativeModelRecipeSourceForm::Checkpoint {
            let mut prefixes = BTreeSet::new();
            for (name, _) in &ordered_state {
                if let Some(projected) = name.strip_prefix("cond_stage_model.transformer.")
                    && clip_recipe_architecture_key(projected)
                {
                    prefixes.insert("cond_stage_model.transformer.".to_owned());
                }
                if let Some(rest) = name.strip_prefix("conditioner.embedders.") {
                    if let Some((ordinal, projected)) = rest.split_once(".transformer.")
                        && !ordinal.is_empty()
                        && ordinal.bytes().all(|byte| byte.is_ascii_digit())
                        && clip_recipe_architecture_key(projected)
                    {
                        prefixes.insert(format!("conditioner.embedders.{ordinal}.transformer."));
                    }
                }
            }
            if prefixes.len() != 1 {
                return Err(ClipError::InvalidNativeResource(format!(
                    "{owner} checkpoint has an ambiguous or incomplete text-encoder bundle"
                )));
            }
            prefixes.into_iter().next()
        } else {
            None
        };
    let mut state = BTreeMap::new();
    for (name, tensor) in ordered_state {
        let normalized = match (&checkpoint_prefix, source_form) {
            (Some(prefix), crate::model_family::NativeModelRecipeSourceForm::Checkpoint) => {
                let Some(projected) = name.strip_prefix(prefix) else {
                    if name.starts_with("cond_stage_model.")
                        || name.starts_with("conditioner.embedders.")
                    {
                        return Err(ClipError::InvalidNativeResource(format!(
                            "{owner} checkpoint contains a disallowed text-encoder sibling"
                        )));
                    }
                    continue;
                };
                projected.to_owned()
            }
            (None, crate::model_family::NativeModelRecipeSourceForm::Standalone) => {
                if name.starts_with("cond_stage_model.")
                    || name.starts_with("conditioner.embedders.")
                {
                    return Err(ClipError::InvalidNativeResource(format!(
                        "standalone {owner} source contains a checkpoint wrapper"
                    )));
                }
                name
            }
            _ => {
                return Err(ClipError::InvalidNativeResource(format!(
                    "{owner} checkpoint projection is unavailable"
                )));
            }
        };
        if state.insert(normalized.clone(), tensor).is_some() {
            return Err(ClipError::InvalidNativeResource(format!(
                "{owner} recipe contains duplicate tensor {normalized}"
            )));
        }
    }
    Ok(state)
}

fn project_open_clip_recipe_tensor(
    source_name: &str,
    tensor: Tensor,
    state: &mut BTreeMap<String, Tensor>,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(), ClipError> {
    let direct = match source_name {
        "token_embedding.weight" => Some("text_model.embeddings.token_embedding.weight"),
        "positional_embedding" => Some("text_model.embeddings.position_embedding.weight"),
        "ln_final.weight" => Some("text_model.final_layer_norm.weight"),
        "ln_final.bias" => Some("text_model.final_layer_norm.bias"),
        "text_projection.weight" | "logit_scale" => Some(source_name),
        _ => None,
    };
    if let Some(target) = direct {
        return insert_clip_recipe_tensor(state, target, tensor);
    }
    if source_name == "text_projection" {
        if tensor.descriptor().shape().len() != 2 {
            return Err(ClipError::InvalidNativeResource(
                "OpenCLIP text projection must have rank 2".to_owned(),
            ));
        }
        let descriptor = tensor.descriptor().permuted_view(&[1, 0])?;
        let view = tensor.view(descriptor, ViewAccess::ReadOnly)?;
        return insert_clip_recipe_tensor(
            state,
            "text_projection.weight",
            clip_recipe_contiguous_copy(backend, &view, context)?,
        );
    }
    let Some(rest) = source_name.strip_prefix("transformer.resblocks.") else {
        return Err(ClipError::InvalidNativeResource(format!(
            "OpenCLIP recipe contains unsupported tensor {source_name}"
        )));
    };
    let Some((layer, field)) = rest.split_once('.') else {
        return Err(ClipError::InvalidNativeResource(format!(
            "OpenCLIP recipe tensor has no layer field: {source_name}"
        )));
    };
    if layer.is_empty() || !layer.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ClipError::InvalidNativeResource(format!(
            "OpenCLIP recipe tensor has an invalid layer: {source_name}"
        )));
    }
    let mapped = match field {
        "ln_1.weight" => Some("layer_norm1.weight"),
        "ln_1.bias" => Some("layer_norm1.bias"),
        "ln_2.weight" => Some("layer_norm2.weight"),
        "ln_2.bias" => Some("layer_norm2.bias"),
        "mlp.c_fc.weight" => Some("mlp.fc1.weight"),
        "mlp.c_fc.bias" => Some("mlp.fc1.bias"),
        "mlp.c_proj.weight" => Some("mlp.fc2.weight"),
        "mlp.c_proj.bias" => Some("mlp.fc2.bias"),
        "attn.out_proj.weight" => Some("self_attn.out_proj.weight"),
        "attn.out_proj.bias" => Some("self_attn.out_proj.bias"),
        _ => None,
    };
    if let Some(mapped) = mapped {
        return insert_clip_recipe_tensor(
            state,
            &format!("text_model.encoder.layers.{layer}.{mapped}"),
            tensor,
        );
    }
    let projection_suffix = match field {
        "attn.in_proj_weight" => "weight",
        "attn.in_proj_bias" => "bias",
        _ => {
            return Err(ClipError::InvalidNativeResource(format!(
                "OpenCLIP recipe contains unsupported tensor {source_name}"
            )));
        }
    };
    let projection_elements = tensor
        .descriptor()
        .shape()
        .first()
        .copied()
        .filter(|elements| *elements > 0 && elements.is_multiple_of(3))
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(format!(
                "OpenCLIP combined attention projection has invalid shape: {source_name}"
            ))
        })?
        / 3;
    for (ordinal, projection) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
        let start = i64::try_from(ordinal)
            .ok()
            .and_then(|ordinal| ordinal.checked_mul(i64::try_from(projection_elements).ok()?))
            .ok_or_else(|| ClipError::Allocation("OpenCLIP attention projection offset"))?;
        let descriptor = tensor
            .descriptor()
            .narrowed_view(0, start, projection_elements)?;
        let view = tensor.view(descriptor, ViewAccess::ReadOnly)?;
        insert_clip_recipe_tensor(
            state,
            &format!(
                "text_model.encoder.layers.{layer}.self_attn.{projection}.{projection_suffix}"
            ),
            clip_recipe_contiguous_copy(backend, &view, context)?,
        )?;
    }
    Ok(())
}

fn clip_recipe_contiguous_copy(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipError> {
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        input.descriptor().dtype(),
        input.descriptor().device(),
        context.stream,
    )?;
    let (output, event) = backend.copy(input, descriptor, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn t5_recipe_component(
    profile: NativeClipProfile,
    source: NativeClipRecipeSource,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipComponent, ClipError> {
    let (configuration_document, configuration_artifact) =
        recipe_configuration_artifact(&source, NativeClipRecipeArchitecture::T5, true)?
            .ok_or_else(|| {
                ClipError::InvalidNativeResource("T5 recipe omitted its checked config".to_owned())
            })?;
    let configuration_sha256 = configuration_artifact.artifact_sha256.clone();
    let NativeClipRecipeSource {
        source_form,
        artifact_sha256,
        ordered_state,
        parser_limits,
        mut auxiliary_artifacts,
    } = source;
    validate_clip_recipe_parser_limits(&parser_limits, None, &auxiliary_artifacts)?;
    validate_sha256("native T5 recipe artifact", &artifact_sha256)?;
    remove_clip_recipe_auxiliary_identity(&mut auxiliary_artifacts, &configuration_sha256)?;
    let vocabulary_size = recipe_config_usize(&configuration_document, "vocab_size")?;
    let hidden_size = recipe_config_usize(&configuration_document, "d_model")?;
    let attention_heads = recipe_config_usize(&configuration_document, "num_heads")?;
    let layer_count = recipe_config_usize(&configuration_document, "num_layers")?;
    let feed_forward_size = recipe_config_usize(&configuration_document, "d_ff")?;
    let key_value_width = recipe_config_usize(&configuration_document, "d_kv")?;
    let attention_inner_size = attention_heads
        .checked_mul(key_value_width)
        .ok_or(ClipError::Overflow("T5 attention width"))?;
    let epsilon = recipe_config_f32(&configuration_document, "layer_norm_epsilon")?;
    let feed_forward_projection = configuration_document
        .get("feed_forward_proj")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("gated-gelu");
    let (gated_feed_forward, activation) = match feed_forward_projection {
        "gated-gelu" | "gated-gelu_new" => (true, BidirectionalFeedForwardActivation::GeluTanh),
        "gated-relu" => (true, BidirectionalFeedForwardActivation::Relu),
        "relu" => (false, BidirectionalFeedForwardActivation::Relu),
        _ => {
            return Err(ClipError::InvalidNativeResource(
                "T5 recipe feed-forward projection is unsupported".to_owned(),
            ));
        }
    };
    let mut state = checked_recipe_source_state(source_form, ordered_state, "T5")?;
    let token_embedding = state
        .remove("shared.weight")
        .or_else(|| state.remove("encoder.embed_tokens.weight"))
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(
                "T5 recipe is missing its shared token embedding".to_owned(),
            )
        })?;
    let mut canonical_state = BTreeMap::new();
    insert_clip_recipe_tensor(
        &mut canonical_state,
        "token_embedding.weight",
        token_embedding.clone(),
    )?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(layer_count)
        .map_err(|_| ClipError::Allocation("T5 recipe layers"))?;
    for index in 0..layer_count {
        context.check()?;
        let source_prefix = format!("encoder.block.{index}");
        let canonical = format!("layers.{index}");
        let attention_norm_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.0.layer_norm.weight"),
            &format!("{canonical}.attention_norm.weight"),
        )?;
        let query_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.0.SelfAttention.q.weight"),
            &format!("{canonical}.query.weight"),
        )?;
        let key_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.0.SelfAttention.k.weight"),
            &format!("{canonical}.key.weight"),
        )?;
        let value_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.0.SelfAttention.v.weight"),
            &format!("{canonical}.value.weight"),
        )?;
        let attention_output_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.0.SelfAttention.o.weight"),
            &format!("{canonical}.attention_output.weight"),
        )?;
        let feed_forward_norm_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.1.layer_norm.weight"),
            &format!("{canonical}.feed_forward_norm.weight"),
        )?;
        let feed_forward_input_source = if gated_feed_forward {
            format!("{source_prefix}.layer.1.DenseReluDense.wi_0.weight")
        } else {
            format!("{source_prefix}.layer.1.DenseReluDense.wi.weight")
        };
        let feed_forward_input_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &feed_forward_input_source,
            &format!("{canonical}.feed_forward_input.weight"),
        )?;
        let feed_forward_gate_weight = if gated_feed_forward {
            Some(take_clip_recipe_mapped(
                &mut state,
                &mut canonical_state,
                &format!("{source_prefix}.layer.1.DenseReluDense.wi_1.weight"),
                &format!("{canonical}.feed_forward_gate.weight"),
            )?)
        } else {
            None
        };
        let feed_forward_output_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.layer.1.DenseReluDense.wo.weight"),
            &format!("{canonical}.feed_forward_output.weight"),
        )?;
        let relative_attention_bias = if index == 0 {
            Some(take_clip_recipe_mapped(
                &mut state,
                &mut canonical_state,
                &format!("{source_prefix}.layer.0.SelfAttention.relative_attention_bias.weight"),
                &format!("{canonical}.relative_attention_bias"),
            )?)
        } else {
            None
        };
        layers.push(BidirectionalLayerWeights {
            attention_norm_weight,
            attention_norm_bias: None,
            query_weight,
            query_bias: None,
            key_weight,
            key_bias: None,
            value_weight,
            value_bias: None,
            attention_output_weight,
            attention_output_bias: None,
            feed_forward_norm_weight,
            feed_forward_norm_bias: None,
            feed_forward_input_weight,
            feed_forward_input_bias: None,
            feed_forward_gate_weight,
            feed_forward_gate_bias: None,
            feed_forward_output_weight,
            feed_forward_output_bias: None,
            relative_attention_bias,
        });
    }
    let final_norm_weight = take_clip_recipe_mapped(
        &mut state,
        &mut canonical_state,
        "encoder.final_layer_norm.weight",
        "final_norm.weight",
    )?;
    if let Some(unexpected) = state.keys().next() {
        return Err(ClipError::InvalidNativeResource(format!(
            "T5 recipe contains unexpected tensor {unexpected}"
        )));
    }
    let relative_attention_buckets = layers
        .first()
        .and_then(|layer| layer.relative_attention_bias.as_ref())
        .and_then(|tensor| tensor.descriptor().shape().first())
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            ClipError::InvalidNativeResource(
                "T5 recipe relative-attention bias shape is invalid".to_owned(),
            )
        })?;
    let configuration = BidirectionalTextConfiguration {
        architecture: BidirectionalTextArchitecture::T5,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size,
        maximum_tokens: SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH,
        type_vocabulary_size: 0,
        hidden_size,
        attention_inner_size,
        feed_forward_size,
        attention_heads,
        layer_count,
        normalization_epsilon_bits: epsilon.to_bits(),
        gated_feed_forward,
        activation,
        relative_attention: true,
        relative_attention_buckets,
        relative_attention_max_distance: configuration_document
            .get("relative_attention_max_distance")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(128),
        projection_dimension: None,
    };
    let template = NativeT5TextEncoder::new(
        configuration,
        BidirectionalTextWeights {
            token_embedding,
            position_embedding: None,
            token_type_embedding: None,
            embedding_norm_weight: None,
            embedding_norm_bias: None,
            layers,
            final_norm_weight: Some(final_norm_weight),
            projection_weight: None,
            projection_bias: None,
        },
    )
    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
    let sentence_piece = take_clip_recipe_auxiliary(
        &mut auxiliary_artifacts,
        NativeClipRecipeAuxiliaryFormat::SentencePiece,
    )?;
    validate_remaining_clip_recipe_auxiliary(&auxiliary_artifacts)?;
    let tokenizer = SentencePieceTokenizer::from_model_bytes(
        &sentence_piece.bytes,
        sentence_piece.artifact_sha256,
        &parser_limits,
        context.cancellation,
    )
    .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
    let (minimum_length, maximum_length, attention_mask) = match profile {
        NativeClipProfile::Sd3 => (77, SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH, true),
        NativeClipProfile::PixArt => (1, SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH, false),
        NativeClipProfile::HiDream => (128, 128, true),
        _ => {
            return Err(ClipError::InvalidNativeResource(
                "T5 recipe selected an incompatible CLIP profile".to_owned(),
            ));
        }
    };
    let tokenizer = Arc::new(
        NativePromptTokenizer::checked(
            NativeTokenizerFamily::SentencePiece(tokenizer),
            TokenizerConfiguration {
                maximum_length,
                minimum_length: Some(minimum_length),
                minimum_padding: None,
                pad_to_maximum_length: false,
                pad_left: false,
                start_token: None,
                end_token: Some(1),
                pad_token: 0,
                maximum_word_length: 8,
                disable_weights: false,
                embedding_width: Some(hidden_size),
            },
            BTreeMap::new(),
        )
        .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?,
    );
    let options = NativeClipComponentOptions::checked_source(
        NativeTokenizerInvocationOptions::default(),
        NativeClipLayerSelection::Last,
        false,
        false,
        attention_mask,
        false,
    )?;
    NativeClipComponent::checked_bidirectional(
        NativeClipComponentRole::T5,
        Arc::new(MappedModelWeights::from_parts(
            artifact_sha256,
            canonical_state,
            Vec::new(),
        )),
        tokenizer,
        &template,
        options,
        context.cancellation,
    )?
    .bind_recipe_source(
        NativeClipRecipeArchitecture::T5,
        Some(&configuration_sha256),
        &parser_limits,
        context.cancellation,
    )
}

fn llama_recipe_component(
    source: NativeClipRecipeSource,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipComponent, ClipError> {
    let (configuration_document, configuration_artifact) =
        recipe_configuration_artifact(&source, NativeClipRecipeArchitecture::Llama, true)?
            .ok_or_else(|| {
                ClipError::InvalidNativeResource(
                    "Llama recipe omitted its checked config".to_owned(),
                )
            })?;
    let configuration_sha256 = configuration_artifact.artifact_sha256.clone();
    let NativeClipRecipeSource {
        source_form,
        artifact_sha256,
        ordered_state,
        parser_limits,
        mut auxiliary_artifacts,
    } = source;
    validate_clip_recipe_parser_limits(&parser_limits, None, &auxiliary_artifacts)?;
    validate_sha256("native Llama recipe artifact", &artifact_sha256)?;
    remove_clip_recipe_auxiliary_identity(&mut auxiliary_artifacts, &configuration_sha256)?;
    let vocabulary_size = recipe_config_usize(&configuration_document, "vocab_size")?;
    let maximum_tokens = recipe_config_usize(&configuration_document, "max_position_embeddings")?;
    let hidden_size = recipe_config_usize(&configuration_document, "hidden_size")?;
    let feed_forward_size = recipe_config_usize(&configuration_document, "intermediate_size")?;
    let layer_count = recipe_config_usize(&configuration_document, "num_hidden_layers")?;
    let attention_heads = recipe_config_usize(&configuration_document, "num_attention_heads")?;
    let key_value_heads = recipe_config_usize(&configuration_document, "num_key_value_heads")?;
    let head_dimension = hidden_size
        .checked_div(attention_heads)
        .filter(|value| *value > 0 && hidden_size.is_multiple_of(attention_heads))
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("Llama recipe attention width is invalid".to_owned())
        })?;
    let epsilon = recipe_config_f32(&configuration_document, "rms_norm_eps")?;
    let rope_theta = recipe_config_f32(&configuration_document, "rope_theta")?;
    let mut state = checked_recipe_source_state(source_form, ordered_state, "Llama")?;
    let token_embedding = take_clip_recipe_tensor(&mut state, "model.embed_tokens.weight")?;
    let mut canonical_state = BTreeMap::new();
    insert_clip_recipe_tensor(
        &mut canonical_state,
        "token_embedding.weight",
        token_embedding.clone(),
    )?;
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(layer_count)
        .map_err(|_| ClipError::Allocation("Llama recipe layers"))?;
    for index in 0..layer_count {
        context.check()?;
        let source_prefix = format!("model.layers.{index}");
        let canonical = format!("layers.{index}");
        let attention_norm_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.input_layernorm.weight"),
            &format!("{canonical}.attention_norm_weight"),
        )?;
        let query_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.self_attn.q_proj.weight"),
            &format!("{canonical}.query.weight"),
        )?;
        let key_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.self_attn.k_proj.weight"),
            &format!("{canonical}.key.weight"),
        )?;
        let value_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.self_attn.v_proj.weight"),
            &format!("{canonical}.value.weight"),
        )?;
        let attention_output_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.self_attn.o_proj.weight"),
            &format!("{canonical}.attention_output.weight"),
        )?;
        let feed_forward_norm_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.post_attention_layernorm.weight"),
            &format!("{canonical}.feed_forward_norm_weight"),
        )?;
        let feed_forward_gate_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.mlp.gate_proj.weight"),
            &format!("{canonical}.feed_forward_gate.weight"),
        )?;
        let feed_forward_up_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.mlp.up_proj.weight"),
            &format!("{canonical}.feed_forward_up.weight"),
        )?;
        let feed_forward_down_weight = take_clip_recipe_mapped(
            &mut state,
            &mut canonical_state,
            &format!("{source_prefix}.mlp.down_proj.weight"),
            &format!("{canonical}.feed_forward_down.weight"),
        )?;
        layers.push(DecoderLayerWeights {
            attention_norm_weight,
            attention: DecoderAttentionWeights::DotProduct {
                query_weight,
                key_weight,
                value_weight,
                query_norm_weight: None,
                key_norm_weight: None,
                output_weight: attention_output_weight,
            },
            feed_forward_norm_weight,
            feed_forward_gate_weight,
            feed_forward_up_weight,
            feed_forward_down_weight,
            post_attention_norm_weight: None,
            post_feed_forward_norm_weight: None,
            attention_sink: None,
            gemma4_layer_input: None,
        });
    }
    let final_norm_weight = take_clip_recipe_mapped(
        &mut state,
        &mut canonical_state,
        "model.norm.weight",
        "final_norm_weight",
    )?;
    let output_head_weight = state.remove("lm_head.weight");
    if let Some(weight) = output_head_weight.as_ref() {
        insert_clip_recipe_tensor(&mut canonical_state, "output_head.weight", weight.clone())?;
    }
    if let Some(unexpected) = state.keys().next() {
        return Err(ClipError::InvalidNativeResource(format!(
            "Llama recipe contains unexpected tensor {unexpected}"
        )));
    }
    let tied_output_head = output_head_weight.is_none();
    let configuration = DecoderTextConfiguration {
        architecture: DecoderArchitecture::Llama,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size,
        maximum_tokens,
        hidden_size,
        feed_forward_size,
        layer_kinds: vec![DecoderLayerKind::FullAttention; layer_count],
        attention_heads,
        key_value_heads,
        head_dimension,
        query_key_norm: false,
        qwen35_linear: None,
        gemma3: None,
        gemma4: None,
        normalization_epsilon_bits: epsilon.to_bits(),
        rope: DecoderRopeConfiguration {
            theta: rope_theta,
            rotary_dimension: head_dimension,
            interleaved_sections: Vec::new(),
            scaling: RopeScaling::None,
        },
        sliding_window: None,
        activation: DecoderActivation::Silu,
        embedding_scale_bits: 1.0_f32.to_bits(),
        residual_scale_bits: 1.0_f32.to_bits(),
        norm_weight_offset_bits: 0.0_f32.to_bits(),
        logits_soft_cap_bits: None,
        tied_output_head,
        stop_tokens: vec![128_001, 128_009],
    };
    let template = NativeDecoderTextEncoder::new(
        configuration,
        DecoderTextWeights {
            token_embedding,
            layers,
            final_norm_weight,
            output_head_weight,
            gemma4_per_layer: None,
        },
    )
    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
    let tokenizer_artifact =
        take_clip_recipe_tokenizer_json(&mut auxiliary_artifacts, &parser_limits)?;
    validate_remaining_clip_recipe_auxiliary(&auxiliary_artifacts)?;
    let tokenizer = Llama3BpeTokenizer::from_tokenizer_json(
        &tokenizer_artifact.bytes,
        &tokenizer_artifact.artifact_sha256,
        &parser_limits,
        context.cancellation,
    )
    .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
    let tokenizer = Arc::new(
        NativePromptTokenizer::checked(
            NativeTokenizerFamily::Llama3ByteBpe(tokenizer),
            TokenizerConfiguration {
                maximum_length: SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH,
                minimum_length: Some(128),
                minimum_padding: None,
                pad_to_maximum_length: false,
                pad_left: false,
                start_token: Some(128_000),
                end_token: None,
                pad_token: 128_009,
                maximum_word_length: 8,
                disable_weights: false,
                embedding_width: Some(hidden_size),
            },
            BTreeMap::new(),
        )
        .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?,
    );
    let options = NativeClipComponentOptions::checked_source(
        NativeTokenizerInvocationOptions::default(),
        NativeClipLayerSelection::All,
        false,
        false,
        true,
        false,
    )?;
    NativeClipComponent::checked_decoder(
        NativeClipComponentRole::Llama,
        Arc::new(MappedModelWeights::from_parts(
            artifact_sha256,
            canonical_state,
            Vec::new(),
        )),
        tokenizer,
        &template,
        options,
        context.cancellation,
    )?
    .bind_recipe_source(
        NativeClipRecipeArchitecture::Llama,
        Some(&configuration_sha256),
        &parser_limits,
        context.cancellation,
    )
}

fn insert_clip_recipe_tensor(
    state: &mut BTreeMap<String, Tensor>,
    name: &str,
    tensor: Tensor,
) -> Result<(), ClipError> {
    if state.insert(name.to_owned(), tensor).is_some() {
        return Err(ClipError::InvalidNativeResource(format!(
            "CLIP recipe duplicates canonical tensor {name}"
        )));
    }
    Ok(())
}

fn take_clip_recipe_mapped(
    source: &mut BTreeMap<String, Tensor>,
    mapped: &mut BTreeMap<String, Tensor>,
    source_name: &str,
    mapped_name: &str,
) -> Result<Tensor, ClipError> {
    let tensor = take_clip_recipe_tensor(source, source_name)?;
    insert_clip_recipe_tensor(mapped, mapped_name, tensor.clone())?;
    Ok(tensor)
}

fn clip_recipe_tokenizer(
    tokenizer_json: &[u8],
    profile: NativeClipProfile,
    role: NativeClipComponentRole,
    hidden_size: usize,
    parser_limits: &ParserLimits,
) -> Result<NativePromptTokenizer, ClipError> {
    let document = crate::formats::parse_strict_json(tokenizer_json, parser_limits)
        .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
    let model = document.get("model").unwrap_or(&document);
    let vocabulary = model
        .get("vocab")
        .or_else(|| document.get("vocab"))
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("CLIP tokenizer JSON omits its vocabulary".to_owned())
        })?;
    let vocabulary = serde_json::to_string(vocabulary)
        .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
    let merges = model
        .get("merges")
        .or_else(|| document.get("merges"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("CLIP tokenizer JSON omits its merges".to_owned())
        })?;
    let mut merge_text = String::from("#version: 0.2\n");
    for merge in merges {
        let line = if let Some(line) = merge.as_str() {
            line.to_owned()
        } else if let Some(pair) = merge.as_array() {
            let first = pair
                .first()
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    ClipError::InvalidNativeResource("CLIP tokenizer merge is invalid".to_owned())
                })?;
            let second = pair
                .get(1)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    ClipError::InvalidNativeResource("CLIP tokenizer merge is invalid".to_owned())
                })?;
            format!("{first} {second}")
        } else {
            return Err(ClipError::InvalidNativeResource(
                "CLIP tokenizer merge is invalid".to_owned(),
            ));
        };
        merge_text.push_str(&line);
        merge_text.push('\n');
    }
    let pad_token = if role == NativeClipComponentRole::ClipG {
        0
    } else {
        SD1_END_TOKEN
    };
    NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(
            ClipBpeTokenizer::from_json_and_merges(
                ModelTokenizerDescriptor::checked(
                    format!("sim.native_model_resource.{profile:?}.{role:?}").to_ascii_lowercase(),
                )
                .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?,
                &vocabulary,
                &merge_text,
            )
            .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?,
        ),
        TokenizerConfiguration {
            maximum_length: SD1_CONTEXT_LENGTH,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: true,
            pad_left: false,
            start_token: Some(SD1_START_TOKEN),
            end_token: Some(SD1_END_TOKEN),
            pad_token,
            maximum_word_length: 8,
            disable_weights: false,
            embedding_width: Some(hidden_size),
        },
        BTreeMap::new(),
    )
    .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))
}

fn validate_native_clip_component_order(
    profile: NativeClipProfile,
    components: &[Arc<NativeClipComponent>],
) -> Result<(), ClipError> {
    let expected: &[NativeClipComponentRole] = match profile {
        NativeClipProfile::Sd1 => &[NativeClipComponentRole::ClipL],
        NativeClipProfile::Sdxl => &[
            NativeClipComponentRole::ClipL,
            NativeClipComponentRole::ClipG,
        ],
        NativeClipProfile::Sd3 => &[
            NativeClipComponentRole::ClipL,
            NativeClipComponentRole::ClipG,
            NativeClipComponentRole::T5,
        ],
        NativeClipProfile::PixArt => &[NativeClipComponentRole::T5],
        NativeClipProfile::Lumina | NativeClipProfile::Gemma => &[NativeClipComponentRole::Gemma],
        NativeClipProfile::HiDream => &[
            NativeClipComponentRole::ClipL,
            NativeClipComponentRole::ClipG,
            NativeClipComponentRole::T5,
            NativeClipComponentRole::Llama,
        ],
        NativeClipProfile::Qwen => &[NativeClipComponentRole::Qwen],
    };
    let roles = components
        .iter()
        .map(|component| component.role)
        .collect::<Vec<_>>();
    let valid = if matches!(
        profile,
        NativeClipProfile::Sdxl | NativeClipProfile::Sd3 | NativeClipProfile::HiDream
    ) {
        roles
            .iter()
            .try_fold(0_usize, |offset, role| {
                expected[offset..]
                    .iter()
                    .position(|candidate| candidate == role)
                    .map(|position| offset + position + 1)
            })
            .is_some()
    } else {
        roles == expected
    };
    if !valid {
        return Err(ClipError::InvalidNativeResource(
            "native CLIP component order is invalid".to_owned(),
        ));
    }
    for component in components {
        let owner_matches = match (profile, component.role, &component.encoder) {
            (
                NativeClipProfile::Sd1,
                NativeClipComponentRole::ClipL,
                NativeClipEncoder::ClipText(_),
            )
            | (
                NativeClipProfile::Sdxl | NativeClipProfile::Sd3 | NativeClipProfile::HiDream,
                NativeClipComponentRole::ClipL | NativeClipComponentRole::ClipG,
                NativeClipEncoder::ClipText(_),
            )
            | (
                NativeClipProfile::Sd3 | NativeClipProfile::PixArt | NativeClipProfile::HiDream,
                NativeClipComponentRole::T5,
                NativeClipEncoder::Bidirectional(_),
            )
            | (
                NativeClipProfile::Lumina,
                NativeClipComponentRole::Gemma,
                NativeClipEncoder::Decoder(_),
            )
            | (
                NativeClipProfile::HiDream,
                NativeClipComponentRole::Llama,
                NativeClipEncoder::Decoder(_),
            )
            | (
                NativeClipProfile::Qwen,
                NativeClipComponentRole::Qwen,
                NativeClipEncoder::Qwen25(_),
            )
            | (
                NativeClipProfile::Gemma,
                NativeClipComponentRole::Gemma,
                NativeClipEncoder::Gemma4(_),
            ) => true,
            _ => false,
        };
        if !owner_matches {
            return Err(ClipError::InvalidNativeResource(
                "native CLIP profile retained the wrong concrete component owner".to_owned(),
            ));
        }
        if !native_clip_component_matches_source_profile(profile, component)
            && !(cfg!(any(test, feature = "test-support"))
                && component.recipe_binding_sha256.is_some())
        {
            return Err(ClipError::InvalidNativeResource(
                "native CLIP component options, dimensions, or tokenizer differ from the source profile"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

fn native_clip_component_matches_source_profile(
    profile: NativeClipProfile,
    component: &NativeClipComponent,
) -> bool {
    let width = match &component.encoder {
        NativeClipEncoder::ClipText(owner) => owner.configuration().hidden_size,
        NativeClipEncoder::Bidirectional(owner) => owner.configuration().hidden_size,
        NativeClipEncoder::Decoder(owner) => owner.configuration().hidden_size,
        NativeClipEncoder::Qwen25(owner) => owner.decoder().configuration().hidden_size,
        NativeClipEncoder::Gemma4(owner) => owner.decoder().configuration().hidden_size,
    };
    native_clip_source_contract(
        profile,
        component.role,
        width,
        &component.options,
        component.tokenizer.configuration(),
    )
}

fn native_clip_source_contract(
    profile: NativeClipProfile,
    role: NativeClipComponentRole,
    width: usize,
    options: &NativeClipComponentOptions,
    tokenizer: &crate::clip_tokenizer::TokenizerConfiguration,
) -> bool {
    if tokenizer.embedding_width != Some(width) {
        return false;
    }
    let clip_fixed = |expected_width, layer, final_norm, projected, pad_token| {
        width == expected_width
            && options.layer == layer
            && options.final_layer_norm_intermediate == final_norm
            && options.projected_pooled == projected
            && !options.attention_mask
            && !options.zero_out_masked
            && tokenizer.maximum_length == 77
            && tokenizer.pad_to_maximum_length
            && !tokenizer.pad_left
            && tokenizer.start_token == Some(49_406)
            && tokenizer.end_token == Some(49_407)
            && tokenizer.pad_token == pad_token
            && !tokenizer.disable_weights
    };
    let t5 = |minimum_length, maximum_length, attention_mask| {
        width == 4_096
            && options.layer == NativeClipLayerSelection::Last
            && !options.final_layer_norm_intermediate
            && !options.projected_pooled
            && options.attention_mask == attention_mask
            && !options.zero_out_masked
            && tokenizer.minimum_length == Some(minimum_length)
            && tokenizer.minimum_padding.is_none()
            && tokenizer.maximum_length == maximum_length
            && !tokenizer.pad_to_maximum_length
            && !tokenizer.pad_left
            && tokenizer.start_token.is_none()
            && tokenizer.end_token == Some(1)
            && tokenizer.pad_token == 0
            && !tokenizer.disable_weights
    };
    match (profile, role) {
        (NativeClipProfile::Sd1, NativeClipComponentRole::ClipL) => {
            options.layer == NativeClipLayerSelection::Last
                && options.final_layer_norm_intermediate
                && !options.projected_pooled
                && !options.attention_mask
                && !options.zero_out_masked
                && tokenizer.maximum_length == 77
                && tokenizer.pad_to_maximum_length
                && !tokenizer.pad_left
                && tokenizer.start_token == Some(49_406)
                && tokenizer.end_token == Some(49_407)
                && tokenizer.pad_token == 49_407
                && !tokenizer.disable_weights
        }
        (NativeClipProfile::Sdxl, NativeClipComponentRole::ClipL)
        | (NativeClipProfile::Sd3, NativeClipComponentRole::ClipL) => clip_fixed(
            768,
            NativeClipLayerSelection::Hidden(-2),
            false,
            false,
            49_407,
        ),
        (NativeClipProfile::Sdxl, NativeClipComponentRole::ClipG)
        | (NativeClipProfile::Sd3, NativeClipComponentRole::ClipG) => {
            clip_fixed(1_280, NativeClipLayerSelection::Hidden(-2), false, true, 0)
        }
        (NativeClipProfile::Sd3, NativeClipComponentRole::T5) => t5(
            77,
            SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH,
            options.attention_mask,
        ),
        (NativeClipProfile::PixArt, NativeClipComponentRole::T5) => {
            t5(1, SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH, false)
        }
        (NativeClipProfile::Lumina, NativeClipComponentRole::Gemma) => {
            matches!(width, 2_304 | 2_560)
                && options.layer == NativeClipLayerSelection::Hidden(-2)
                && !options.final_layer_norm_intermediate
                && !options.projected_pooled
                && options.attention_mask
                && !options.zero_out_masked
                && tokenizer.minimum_length == Some(1)
                && tokenizer.maximum_length == SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH
                && !tokenizer.pad_to_maximum_length
                && !tokenizer.pad_left
                && tokenizer.start_token == Some(2)
                && tokenizer.end_token.is_none()
                && tokenizer.pad_token == 0
                && tokenizer.disable_weights == (width == 2_560)
        }
        (NativeClipProfile::HiDream, NativeClipComponentRole::ClipL) => {
            clip_fixed(768, NativeClipLayerSelection::Last, true, true, 49_407)
        }
        (NativeClipProfile::HiDream, NativeClipComponentRole::ClipG) => {
            clip_fixed(1_280, NativeClipLayerSelection::Hidden(-2), false, true, 0)
        }
        (NativeClipProfile::HiDream, NativeClipComponentRole::T5) => t5(128, 128, true),
        (NativeClipProfile::HiDream, NativeClipComponentRole::Llama) => {
            width == 4_096
                && options.layer == NativeClipLayerSelection::All
                && !options.final_layer_norm_intermediate
                && !options.projected_pooled
                && options.attention_mask
                && !options.zero_out_masked
                && tokenizer.minimum_length == Some(128)
                && tokenizer.minimum_padding.is_none()
                && tokenizer.maximum_length == SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH
                && !tokenizer.pad_to_maximum_length
                && !tokenizer.pad_left
                && tokenizer.start_token == Some(128_000)
                && tokenizer.end_token.is_none()
                && tokenizer.pad_token == 128_009
                && !tokenizer.disable_weights
        }
        (NativeClipProfile::Qwen, NativeClipComponentRole::Qwen) => {
            width == 3_584
                && options.layer == NativeClipLayerSelection::Last
                && !options.final_layer_norm_intermediate
                && !options.projected_pooled
                && options.attention_mask
                && !options.zero_out_masked
                && tokenizer.minimum_length == Some(1)
                && tokenizer.maximum_length == SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH
                && !tokenizer.pad_to_maximum_length
                && !tokenizer.pad_left
                && tokenizer.start_token.is_none()
                && tokenizer.end_token.is_none()
                && tokenizer.pad_token == 151_643
        }
        (NativeClipProfile::Gemma, NativeClipComponentRole::Gemma) => {
            options.layer == NativeClipLayerSelection::All
                && !options.final_layer_norm_intermediate
                && !options.projected_pooled
                && options.attention_mask
                && !options.zero_out_masked
                && tokenizer.minimum_length == Some(1)
                && tokenizer.maximum_length == SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH
                && !tokenizer.pad_to_maximum_length
                && tokenizer.pad_left
                && tokenizer.start_token == Some(2)
                && tokenizer.end_token.is_none()
                && tokenizer.pad_token == 0
                && tokenizer.disable_weights
        }
        _ => false,
    }
}

fn execute_native_clip_profile(
    profile: NativeClipProfile,
    components: &[Arc<NativeClipComponent>],
    source_layer_override: Option<i64>,
    backend: &CpuBackend,
    request: NativeClipExecutionRequest<'_>,
    start_percent: f64,
    end_percent: f64,
    hook_metadata_json: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<NativeClipOutput, ClipError> {
    validate_native_clip_component_order(profile, components)?;
    if profile == NativeClipProfile::Qwen {
        let component = components.first().ok_or_else(|| {
            ClipError::InvalidNativeResource("Qwen2.5 CLIP component is missing".to_owned())
        })?;
        let NativeClipEncoder::Qwen25(owner) = &component.encoder else {
            return Err(ClipError::InvalidNativeResource(
                "Qwen2.5 CLIP does not retain its multimodal owner".to_owned(),
            ));
        };
        let (prompt, prepared_images, custom_template) = match request {
            NativeClipExecutionRequest::Text { prompt } => (prompt, &[][..], None),
            NativeClipExecutionRequest::Qwen25 {
                prompt,
                prepared_images,
                custom_template,
            } => (prompt, prepared_images, custom_template),
            NativeClipExecutionRequest::Gemma { .. } => {
                return Err(ClipError::InvalidNativeResource(
                    "Gemma modalities cannot be routed to Qwen2.5 CLIP".to_owned(),
                ));
            }
        };
        let capture_layer =
            match source_clip_component_layer_selection(component, source_layer_override)? {
                NativeClipLayerSelection::Last => None,
                NativeClipLayerSelection::Hidden(layer) => Some(layer),
                NativeClipLayerSelection::All => {
                    return Err(ClipError::InvalidNativeResource(
                        "Qwen2.5 does not admit all-layer conditioning".to_owned(),
                    ));
                }
            };
        let output = owner
            .encode_conditioning(
                backend,
                Qwen25ConditioningRequest {
                    prompt,
                    prepared_images,
                    custom_template,
                    capture_layer,
                },
                context,
            )
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        return Ok(NativeClipOutput {
            conditioning: output.hidden().clone(),
            pooled: None,
            attention_mask: output.attention_mask().cloned(),
            conditioning_llama3: None,
            clip_start_percent_bits: start_percent.to_bits(),
            clip_end_percent_bits: end_percent.to_bits(),
            hook_metadata_json: hook_metadata_json.to_vec(),
        });
    }
    if profile == NativeClipProfile::Gemma {
        let component = components.first().ok_or_else(|| {
            ClipError::InvalidNativeResource("Gemma4 CLIP component is missing".to_owned())
        })?;
        let NativeClipEncoder::Gemma4(owner) = &component.encoder else {
            return Err(ClipError::InvalidNativeResource(
                "Gemma4 CLIP does not retain its multimodal owner".to_owned(),
            ));
        };
        let (prompt, prepared_visuals, prepared_audio, use_default_template, thinking) =
            match request {
                NativeClipExecutionRequest::Text { prompt } => {
                    (prompt, &[][..], None, false, false)
                }
                NativeClipExecutionRequest::Gemma {
                    prompt,
                    prepared_visuals,
                    prepared_audio,
                    use_default_template,
                    thinking,
                } => (
                    prompt,
                    prepared_visuals,
                    prepared_audio,
                    use_default_template,
                    thinking,
                ),
                NativeClipExecutionRequest::Qwen25 { .. } => {
                    return Err(ClipError::InvalidNativeResource(
                        "Qwen2.5 modalities cannot be routed to Gemma4 CLIP".to_owned(),
                    ));
                }
            };
        let output = owner
            .encode_conditioning(
                backend,
                GemmaConditioningRequest {
                    prompt,
                    prepared_visuals,
                    prepared_audio,
                    use_default_template,
                    thinking,
                },
                context,
            )
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        return Ok(NativeClipOutput {
            conditioning: output.hidden().clone(),
            pooled: None,
            attention_mask: Some(output.attention_mask().clone()),
            conditioning_llama3: None,
            clip_start_percent_bits: start_percent.to_bits(),
            clip_end_percent_bits: end_percent.to_bits(),
            hook_metadata_json: hook_metadata_json.to_vec(),
        });
    }
    if matches!(
        request,
        NativeClipExecutionRequest::Qwen25 { .. } | NativeClipExecutionRequest::Gemma { .. }
    ) {
        return Err(ClipError::InvalidNativeResource(
            "multimodal inputs cannot be routed to another CLIP profile".to_owned(),
        ));
    }
    let prompt = request.prompt();
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(components.len())
        .map_err(|_| ClipError::Allocation("native CLIP component outputs"))?;
    for component in components {
        let (hidden, pooled, attention_mask) = execute_clip_text_component(
            profile,
            component,
            source_layer_override,
            backend,
            prompt,
            context,
        )?;
        encoded.push(NativeClipComponentEncoding {
            role: component.role,
            hidden,
            pooled,
            attention_mask,
        });
    }
    let (conditioning, pooled, attention_mask, conditioning_llama3) =
        compose_native_clip_profile(profile, &encoded, backend, context)?;
    Ok(NativeClipOutput {
        conditioning,
        pooled,
        attention_mask,
        conditioning_llama3,
        clip_start_percent_bits: start_percent.to_bits(),
        clip_end_percent_bits: end_percent.to_bits(),
        hook_metadata_json: hook_metadata_json.to_vec(),
    })
}

struct NativeClipComponentEncoding {
    role: NativeClipComponentRole,
    hidden: Tensor,
    pooled: Option<Tensor>,
    attention_mask: Option<Tensor>,
}

fn compose_native_clip_profile(
    profile: NativeClipProfile,
    encoded: &[NativeClipComponentEncoding],
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>, Option<Tensor>), ClipError> {
    let find = |role| encoded.iter().find(|component| component.role == role);
    match profile {
        NativeClipProfile::Sd1
        | NativeClipProfile::PixArt
        | NativeClipProfile::Lumina
        | NativeClipProfile::Qwen
        | NativeClipProfile::Gemma => {
            let component = encoded.first().ok_or_else(|| {
                ClipError::InvalidNativeResource("native CLIP component is missing".to_owned())
            })?;
            Ok((
                component.hidden.clone(),
                component.pooled.clone(),
                component.attention_mask.clone(),
                None,
            ))
        }
        NativeClipProfile::Sdxl => {
            let clip_l = find(NativeClipComponentRole::ClipL);
            let clip_g = find(NativeClipComponentRole::ClipG);
            if let Some(component) = clip_l.or(clip_g)
                && clip_l.is_none_or(|clip_l| std::ptr::eq(clip_l, component))
                && clip_g.is_none_or(|clip_g| std::ptr::eq(clip_g, component))
            {
                return Ok((
                    component.hidden.clone(),
                    component.pooled.clone(),
                    component.attention_mask.clone(),
                    None,
                ));
            }
            let clip_l = clip_l.ok_or_else(|| {
                ClipError::InvalidNativeResource("SDXL CLIP-L output is missing".to_owned())
            })?;
            let clip_g = clip_g.ok_or_else(|| {
                ClipError::InvalidNativeResource("SDXL CLIP-G output is missing".to_owned())
            })?;
            let pooled = clip_g.pooled.clone().ok_or_else(|| {
                ClipError::InvalidNativeResource(
                    "SDXL projected CLIP-G pooled output is missing".to_owned(),
                )
            })?;
            let output =
                compose_source_sdxl(backend, &clip_l.hidden, &clip_g.hidden, &pooled, context)
                    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
            Ok((
                output.hidden,
                Some(output.pooled),
                None,
                output.conditioning_llama3,
            ))
        }
        NativeClipProfile::Sd3 => compose_sd3(encoded, backend, context),
        NativeClipProfile::HiDream => {
            let t5 = find(NativeClipComponentRole::T5);
            let output = compose_source_hidream(
                backend,
                find(NativeClipComponentRole::ClipL)
                    .and_then(|component| component.pooled.as_ref()),
                find(NativeClipComponentRole::ClipG)
                    .and_then(|component| component.pooled.as_ref()),
                t5.map(|component| &component.hidden),
                find(NativeClipComponentRole::Llama).map(|component| &component.hidden),
                context,
            )
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
            Ok((
                output.hidden,
                Some(output.pooled),
                t5.and_then(|component| component.attention_mask.clone()),
                output.conditioning_llama3,
            ))
        }
    }
}

fn compose_sd3(
    encoded: &[NativeClipComponentEncoding],
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>, Option<Tensor>), ClipError> {
    let clip_l = encoded
        .iter()
        .find(|component| component.role == NativeClipComponentRole::ClipL);
    let clip_g = encoded
        .iter()
        .find(|component| component.role == NativeClipComponentRole::ClipG);
    let t5 = encoded
        .iter()
        .find(|component| component.role == NativeClipComponentRole::T5);
    let output = compose_source_sd3(
        backend,
        clip_l.and_then(|component| {
            component
                .pooled
                .as_ref()
                .map(|pooled| (&component.hidden, pooled))
        }),
        clip_g.and_then(|component| {
            component
                .pooled
                .as_ref()
                .map(|pooled| (&component.hidden, pooled))
        }),
        t5.map(|component| &component.hidden),
        context,
    )
    .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
    Ok((
        output.hidden,
        Some(output.pooled),
        t5.and_then(|component| component.attention_mask.clone()),
        output.conditioning_llama3,
    ))
}

fn execute_clip_text_component(
    profile: NativeClipProfile,
    component: &NativeClipComponent,
    source_layer_override: Option<i64>,
    backend: &CpuBackend,
    prompt: &str,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Option<Tensor>, Option<Tensor>), ClipError> {
    let layer_selection = source_clip_component_layer_selection(component, source_layer_override)?;
    let tokenized = component
        .tokenizer
        .tokenize_with_options(prompt, component.options.tokenizer, context.cancellation)
        .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
    let sections = tokenized.sections();
    let tokens = sections
        .iter()
        .map(|section| section.tokens().len())
        .max()
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("CLIP tokenizer returned no sections".to_owned())
        })?;
    let hidden_width = component
        .tokenizer
        .configuration()
        .embedding_width
        .ok_or_else(|| {
            ClipError::InvalidNativeResource("CLIP tokenizer has no embedding width".to_owned())
        })?;
    let has_weights = sections
        .iter()
        .any(|section| section.tokens().iter().any(|token| token.weight() != 1.0));
    let rows = sections
        .len()
        .checked_add(usize::from(has_weights))
        .ok_or(ClipError::Overflow("CLIP token rows"))?;
    let mut token_values = Vec::new();
    let mut section_weights = Vec::new();
    let mut num_tokens = Vec::new();
    let mut overrides = Vec::<(usize, usize, Arc<[f32]>)>::new();
    for allocation in [
        token_values.try_reserve_exact(rows),
        section_weights.try_reserve_exact(sections.len()),
        num_tokens.try_reserve_exact(rows),
        overrides.try_reserve_exact(
            sections
                .len()
                .checked_mul(tokens)
                .ok_or(ClipError::Overflow("textual-inversion overrides"))?,
        ),
    ] {
        allocation.map_err(|_| ClipError::Allocation("CLIP token projection"))?;
    }
    for (row, section) in sections.iter().enumerate() {
        let mut values = Vec::new();
        let mut weights = Vec::new();
        values
            .try_reserve_exact(tokens)
            .map_err(|_| ClipError::Allocation("CLIP token row"))?;
        weights
            .try_reserve_exact(tokens)
            .map_err(|_| ClipError::Allocation("CLIP token weights"))?;
        for (position, token) in section.tokens().iter().enumerate() {
            match token.value() {
                NativeTokenValue::Token(value) => values.push(i64::from(*value)),
                NativeTokenValue::Embedding {
                    values: embedding, ..
                } => {
                    if embedding.len() != hidden_width {
                        return Err(ClipError::InvalidNativeResource(
                            "textual-inversion width differs from the encoder".to_owned(),
                        ));
                    }
                    values.push(i64::from(component.tokenizer.configuration().pad_token));
                    overrides.push((row, position, embedding.clone()));
                }
            }
            weights.push(token.weight());
        }
        let pooled_count = component
            .tokenizer
            .configuration()
            .end_token
            .and_then(|end| {
                section.tokens().iter().position(
                    |token| matches!(token.value(), NativeTokenValue::Token(value) if *value == end),
                )
            })
            .and_then(|position| position.checked_add(1))
            .unwrap_or(section.tokens().len());
        num_tokens.push(pooled_count);
        values
            .try_reserve_exact(tokens.saturating_sub(values.len()))
            .map_err(|_| ClipError::Allocation("CLIP token padding"))?;
        weights
            .try_reserve_exact(tokens.saturating_sub(weights.len()))
            .map_err(|_| ClipError::Allocation("CLIP weight padding"))?;
        values.resize(
            tokens,
            i64::from(component.tokenizer.configuration().pad_token),
        );
        weights.resize(tokens, 1.0);
        token_values.push(values);
        section_weights.push(weights);
    }
    if has_weights {
        let empty_end =
            native_clip_empty_baseline_end(profile, component.tokenizer.configuration().end_token);
        let empty_tokens = NativePromptTokenizer::empty_token_ids(
            component.tokenizer.configuration().start_token,
            empty_end,
            component.tokenizer.configuration().pad_token,
            tokens,
        )
        .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
        let mut projected_empty_tokens = Vec::new();
        projected_empty_tokens
            .try_reserve_exact(empty_tokens.len())
            .map_err(|_| ClipError::Allocation("CLIP empty token row"))?;
        projected_empty_tokens.extend(empty_tokens.into_iter().map(i64::from));
        token_values.push(projected_empty_tokens);
        let empty_token_count =
            usize::from(component.tokenizer.configuration().start_token.is_some())
                .checked_add(usize::from(empty_end.is_some()))
                .ok_or(ClipError::Overflow("CLIP empty token count"))?;
        num_tokens.push(if profile == NativeClipProfile::PixArt {
            empty_token_count
        } else {
            empty_token_count.max(1)
        });
    }
    let flattened_capacity = rows
        .checked_mul(tokens)
        .ok_or(ClipError::Overflow("CLIP flattened tokens"))?;
    let mut flattened = Vec::new();
    flattened
        .try_reserve_exact(flattened_capacity)
        .map_err(|_| ClipError::Allocation("CLIP flattened tokens"))?;
    flattened.extend(token_values.iter().flatten().copied());
    let token_tensor = tensor_from_i64_clip(
        backend,
        &[
            u64::try_from(rows).map_err(|_| ClipError::Overflow("CLIP rows"))?,
            u64::try_from(tokens).map_err(|_| ClipError::Overflow("CLIP tokens"))?,
        ],
        &flattened,
        context,
    )?;
    let embeddings = match &component.encoder {
        NativeClipEncoder::ClipText(owner) => {
            owner.embed_tokens(backend, &token_tensor, context)?
        }
        NativeClipEncoder::Bidirectional(owner) => owner
            .embed_tokens(backend, &token_tensor, context)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
        NativeClipEncoder::Decoder(owner) => owner
            .embed_tokens(backend, &token_tensor, context)
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?,
        NativeClipEncoder::Qwen25(_) | NativeClipEncoder::Gemma4(_) => {
            return Err(ClipError::InvalidNativeResource(
                "multimodal execution bypassed its checked request".to_owned(),
            ));
        }
    };
    let mut embedding_values = tensor_to_f32(backend, &embeddings, context)?;
    for (row, position, replacement) in overrides {
        let start = row
            .checked_mul(tokens)
            .and_then(|value| value.checked_add(position))
            .and_then(|value| value.checked_mul(hidden_width))
            .ok_or(ClipError::Overflow("textual-inversion offset"))?;
        let end = start
            .checked_add(hidden_width)
            .ok_or(ClipError::Overflow("textual-inversion end"))?;
        embedding_values
            .get_mut(start..end)
            .ok_or_else(|| {
                ClipError::InvalidNativeResource("textual-inversion offset is invalid".to_owned())
            })?
            .copy_from_slice(&replacement);
    }
    let embeddings = tensor_from_f32(
        backend,
        &[
            u64::try_from(rows).map_err(|_| ClipError::Overflow("CLIP rows"))?,
            u64::try_from(tokens).map_err(|_| ClipError::Overflow("CLIP tokens"))?,
            u64::try_from(hidden_width).map_err(|_| ClipError::Overflow("CLIP width"))?,
        ],
        &embedding_values,
        context,
    )?;
    let mut mask_values = Vec::new();
    mask_values
        .try_reserve_exact(
            rows.checked_mul(tokens)
                .ok_or(ClipError::Overflow("CLIP attention mask"))?,
        )
        .map_err(|_| ClipError::Allocation("CLIP attention mask"))?;
    for count in &num_tokens {
        mask_values.extend((0..tokens).map(|position| i64::from(position < *count)));
    }
    let attention_mask = tensor_from_i64_clip(
        backend,
        &[
            u64::try_from(rows).map_err(|_| ClipError::Overflow("CLIP mask rows"))?,
            u64::try_from(tokens).map_err(|_| ClipError::Overflow("CLIP mask tokens"))?,
        ],
        &mask_values,
        context,
    )?;
    let (hidden, pooled_tensor) = match &component.encoder {
        NativeClipEncoder::ClipText(owner) => {
            let intermediate = match layer_selection {
                NativeClipLayerSelection::Last => ClipTextIntermediate::None,
                NativeClipLayerSelection::Hidden(layer) => ClipTextIntermediate::Layer(layer),
                NativeClipLayerSelection::All => ClipTextIntermediate::All,
            };
            let output = owner.forward(
                backend,
                ClipTextRequest {
                    input: ClipTextInput::Embeddings(&embeddings),
                    attention_mask: component.options.attention_mask.then_some(&attention_mask),
                    num_tokens: Some(&num_tokens),
                    intermediate: intermediate.clone(),
                    final_layer_norm_intermediate: component.options.final_layer_norm_intermediate,
                    project_pooled: component.options.projected_pooled,
                    zero_out_masked: component.options.zero_out_masked,
                },
                context,
            )?;
            let hidden = if matches!(intermediate, ClipTextIntermediate::None) {
                output.last_hidden_state().clone()
            } else {
                output
                    .intermediate()
                    .ok_or_else(|| {
                        ClipError::InvalidNativeResource(
                            "CLIP intermediate output is missing".to_owned(),
                        )
                    })?
                    .clone()
            };
            let pooled = if component.options.projected_pooled {
                output.projected_pooled().cloned()
            } else {
                Some(output.pooled().clone())
            };
            (hidden, pooled)
        }
        NativeClipEncoder::Bidirectional(owner) => {
            let intermediate_layer = match layer_selection {
                NativeClipLayerSelection::Last => None,
                NativeClipLayerSelection::Hidden(layer) => Some(layer),
                NativeClipLayerSelection::All => {
                    return Err(ClipError::InvalidNativeResource(
                        "bidirectional text owner admits one selected layer".to_owned(),
                    ));
                }
            };
            let output = owner
                .forward(
                    backend,
                    BidirectionalTextRequest {
                        input: BidirectionalTextInput::Embeddings(&embeddings),
                        attention_mask: component.options.attention_mask.then_some(&attention_mask),
                        token_type_ids: None,
                        intermediate_layer,
                        final_norm_intermediate: component.options.final_layer_norm_intermediate,
                        pooling: BidirectionalPooling::FirstToken,
                        project_pooled: component.options.projected_pooled,
                    },
                    context,
                )
                .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
            let hidden = if intermediate_layer.is_some() {
                output
                    .intermediate()
                    .ok_or_else(|| {
                        ClipError::InvalidNativeResource(
                            "bidirectional intermediate output is missing".to_owned(),
                        )
                    })?
                    .clone()
            } else {
                output.last_hidden_state().clone()
            };
            let pooled = if component.options.projected_pooled {
                output.projected_pooled().cloned()
            } else {
                output.pooled().cloned()
            };
            (hidden, pooled)
        }
        NativeClipEncoder::Decoder(owner) => {
            let capture_layer = match layer_selection {
                NativeClipLayerSelection::Last | NativeClipLayerSelection::All => None,
                NativeClipLayerSelection::Hidden(layer) => Some(layer),
            };
            let mut positions = Vec::new();
            positions
                .try_reserve_exact(tokens)
                .map_err(|_| ClipError::Allocation("CLIP decoder positions"))?;
            positions.extend(0..tokens);
            let request = DecoderPreparedTextRequest {
                embeddings: &embeddings,
                attention_mask: component.options.attention_mask.then_some(&attention_mask),
                rope_positions: DecoderRopePositions::Scalar(&positions),
                causal_positions: &positions,
                cache: None,
                capture_layer,
                deepstack: None,
                initial_input_ids: Some(&flattened),
            };
            let output = if layer_selection == NativeClipLayerSelection::All {
                owner.forward_prepared_all_layers(backend, request, context)
            } else {
                owner.forward_prepared(backend, request, context)
            }
            .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
            let hidden = match layer_selection {
                NativeClipLayerSelection::All => output
                    .all_intermediate()
                    .ok_or_else(|| {
                        ClipError::InvalidNativeResource(
                            "decoder all-layer output is missing".to_owned(),
                        )
                    })?
                    .clone(),
                NativeClipLayerSelection::Hidden(_) => output
                    .intermediate()
                    .ok_or_else(|| {
                        ClipError::InvalidNativeResource(
                            "decoder intermediate output is missing".to_owned(),
                        )
                    })?
                    .clone(),
                NativeClipLayerSelection::Last => output.last_hidden_state().clone(),
            };
            (hidden, None)
        }
        NativeClipEncoder::Qwen25(_) | NativeClipEncoder::Gemma4(_) => {
            return Err(ClipError::InvalidNativeResource(
                "multimodal execution bypassed its checked request".to_owned(),
            ));
        }
    };
    if layer_selection == NativeClipLayerSelection::All {
        let hidden = collapse_all_decoder_sections(
            backend,
            &hidden,
            sections.len(),
            tokens,
            hidden_width,
            &section_weights,
            has_weights,
            context,
        )?;
        let attention_mask = component
            .options
            .attention_mask
            .then(|| {
                collapse_clip_attention_mask(
                    backend,
                    &attention_mask,
                    sections.len(),
                    tokens,
                    context,
                )
            })
            .transpose()?;
        return Ok((hidden, None, attention_mask));
    }
    let hidden_values = tensor_to_f32(backend, &hidden, context)?;
    let row_values = tokens
        .checked_mul(hidden_width)
        .ok_or(ClipError::Overflow("CLIP hidden row"))?;
    let mut encoded_sections = Vec::new();
    encoded_sections
        .try_reserve_exact(sections.len())
        .map_err(|_| ClipError::Allocation("CLIP encoded sections"))?;
    for values in hidden_values.chunks_exact(row_values).take(sections.len()) {
        let mut section = Vec::new();
        section
            .try_reserve_exact(values.len())
            .map_err(|_| ClipError::Allocation("CLIP encoded section"))?;
        section.extend_from_slice(values);
        encoded_sections.push(section);
    }
    if has_weights {
        let empty = hidden_values
            .get(
                sections
                    .len()
                    .checked_mul(row_values)
                    .ok_or(ClipError::Overflow("CLIP empty row"))?
                    ..rows
                        .checked_mul(row_values)
                        .ok_or(ClipError::Overflow("CLIP empty end"))?,
            )
            .ok_or_else(|| {
                ClipError::InvalidNativeResource("CLIP empty baseline is missing".to_owned())
            })?;
        encoded_sections = apply_empty_baseline_token_weights(
            &encoded_sections,
            empty,
            &section_weights,
            hidden_width,
        )
        .map_err(|error| ClipError::NativeResourceOwner(error.to_string()))?;
    }
    let conditioning_capacity = sections
        .len()
        .checked_mul(row_values)
        .ok_or(ClipError::Overflow("CLIP conditioning values"))?;
    let mut conditioning_values = Vec::new();
    conditioning_values
        .try_reserve_exact(conditioning_capacity)
        .map_err(|_| ClipError::Allocation("CLIP conditioning values"))?;
    conditioning_values.extend(encoded_sections.into_iter().flatten());
    let conditioning = tensor_from_f32(
        backend,
        &[
            1,
            u64::try_from(
                sections
                    .len()
                    .checked_mul(tokens)
                    .ok_or(ClipError::Overflow("CLIP output tokens"))?,
            )
            .map_err(|_| ClipError::Overflow("CLIP output tokens"))?,
            u64::try_from(hidden_width).map_err(|_| ClipError::Overflow("CLIP output width"))?,
        ],
        &conditioning_values,
        context,
    )?;
    let pooled = pooled_tensor
        .as_ref()
        .map(|tensor| first_tensor_row(backend, tensor, context))
        .transpose()?;
    let attention_mask = component
        .options
        .attention_mask
        .then(|| {
            collapse_clip_attention_mask(backend, &attention_mask, sections.len(), tokens, context)
        })
        .transpose()?;
    Ok((conditioning, pooled, attention_mask))
}

fn native_clip_empty_baseline_end(profile: NativeClipProfile, end: Option<u32>) -> Option<u32> {
    (profile != NativeClipProfile::PixArt)
        .then_some(end)
        .flatten()
}

fn collapse_all_decoder_sections(
    backend: &CpuBackend,
    hidden: &Tensor,
    sections: usize,
    tokens: usize,
    width: usize,
    section_weights: &[Vec<f32>],
    has_weights: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipError> {
    let shape = hidden.descriptor().shape();
    let rows = sections
        .checked_add(usize::from(has_weights))
        .ok_or(ClipError::Overflow("decoder rows"))?;
    if shape.len() != 4
        || shape[0] != u64::try_from(rows).map_err(|_| ClipError::Overflow("decoder sections"))?
        || shape[2] != u64::try_from(tokens).map_err(|_| ClipError::Overflow("decoder tokens"))?
        || shape[3] != u64::try_from(width).map_err(|_| ClipError::Overflow("decoder width"))?
    {
        return Err(ClipError::InvalidNativeResource(
            "decoder all-layer output shape changed".to_owned(),
        ));
    }
    let layers = usize::try_from(shape[1]).map_err(|_| ClipError::Overflow("decoder layers"))?;
    let mut source = tensor_to_f32(backend, hidden, context)?.to_vec();
    let row = tokens
        .checked_mul(width)
        .ok_or(ClipError::Overflow("decoder layer row"))?;
    let section_stride = layers
        .checked_mul(row)
        .ok_or(ClipError::Overflow("decoder section stride"))?;
    if has_weights {
        let empty_start = sections
            .checked_mul(section_stride)
            .ok_or(ClipError::Overflow("decoder empty baseline"))?;
        for (section, weights) in section_weights.iter().enumerate() {
            for layer in 0..layers {
                let weight = *weights.get(layer).ok_or_else(|| {
                    ClipError::InvalidNativeResource(
                        "all-layer token weights do not cover every captured layer".to_owned(),
                    )
                })?;
                let start = section
                    .checked_mul(section_stride)
                    .and_then(|value| value.checked_add(layer.checked_mul(row)?))
                    .ok_or(ClipError::Overflow("decoder weighted layer"))?;
                let empty = empty_start
                    .checked_add(
                        layer
                            .checked_mul(row)
                            .ok_or(ClipError::Overflow("decoder empty layer"))?,
                    )
                    .ok_or(ClipError::Overflow("decoder empty layer"))?;
                for offset in 0..row {
                    let baseline = *source.get(empty + offset).ok_or_else(|| {
                        ClipError::InvalidNativeResource(
                            "decoder empty baseline is missing".to_owned(),
                        )
                    })?;
                    let value = source.get_mut(start + offset).ok_or_else(|| {
                        ClipError::InvalidNativeResource(
                            "decoder weighted layer is missing".to_owned(),
                        )
                    })?;
                    *value = (*value - baseline) * weight + baseline;
                }
            }
        }
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            sections
                .checked_mul(section_stride)
                .ok_or(ClipError::Overflow("decoder all-layer output"))?,
        )
        .map_err(|_| ClipError::Allocation("decoder all-layer output"))?;
    for layer in 0..layers {
        for section in 0..sections {
            let start = section
                .checked_mul(section_stride)
                .and_then(|value| value.checked_add(layer.checked_mul(row)?))
                .ok_or(ClipError::Overflow("decoder all-layer offset"))?;
            values.extend_from_slice(source.get(start..start + row).ok_or_else(|| {
                ClipError::InvalidNativeResource("decoder all-layer row is missing".to_owned())
            })?);
        }
    }
    Ok(tensor_from_f32(
        backend,
        &[
            1,
            shape[1],
            u64::try_from(
                sections
                    .checked_mul(tokens)
                    .ok_or(ClipError::Overflow("decoder output tokens"))?,
            )
            .map_err(|_| ClipError::Overflow("decoder output tokens"))?,
            shape[3],
        ],
        &values,
        context,
    )?)
}

fn collapse_clip_attention_mask(
    backend: &CpuBackend,
    mask: &Tensor,
    sections: usize,
    tokens: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipError> {
    let values = tensor_to_f32(backend, mask, context)?;
    let retained = sections
        .checked_mul(tokens)
        .ok_or(ClipError::Overflow("CLIP attention mask"))?;
    let values = values.get(..retained).ok_or_else(|| {
        ClipError::InvalidNativeResource("CLIP attention mask omits source sections".to_owned())
    })?;
    Ok(tensor_from_f32(
        backend,
        &[
            1,
            u64::try_from(retained).map_err(|_| ClipError::Overflow("CLIP attention mask"))?,
        ],
        values,
        context,
    )?)
}

fn first_tensor_row(
    backend: &CpuBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 2 || shape[0] == 0 {
        return Err(ClipError::InvalidNativeResource(
            "CLIP pooled tensor shape is invalid".to_owned(),
        ));
    }
    let width = usize::try_from(shape[1]).map_err(|_| ClipError::Overflow("CLIP pooled width"))?;
    let values = tensor_to_f32(backend, tensor, context)?;
    let first = values
        .get(..width)
        .ok_or_else(|| ClipError::InvalidNativeResource("CLIP pooled row is missing".to_owned()))?;
    Ok(tensor_from_f32(backend, &[1, shape[1]], first, context)?)
}

fn tensor_from_i64_clip(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipError> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, context.stream)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            values
                .len()
                .checked_mul(mem::size_of::<i64>())
                .ok_or(ClipError::Overflow("CLIP token bytes"))?,
        )
        .map_err(|_| ClipError::Allocation("CLIP token bytes"))?;
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let (tensor, _) = backend.upload_bytes(descriptor, &bytes, context)?;
    Ok(tensor)
}

fn native_clip_profile_tag(profile: NativeClipProfile) -> u8 {
    match profile {
        NativeClipProfile::Sd1 => 0,
        NativeClipProfile::Sdxl => 1,
        NativeClipProfile::Sd3 => 2,
        NativeClipProfile::PixArt => 3,
        NativeClipProfile::Lumina => 4,
        NativeClipProfile::HiDream => 5,
        NativeClipProfile::Qwen => 6,
        NativeClipProfile::Gemma => 7,
    }
}
fn native_clip_role_tag(role: NativeClipComponentRole) -> u8 {
    match role {
        NativeClipComponentRole::ClipL => 0,
        NativeClipComponentRole::ClipG => 1,
        NativeClipComponentRole::T5 => 2,
        NativeClipComponentRole::Llama => 3,
        NativeClipComponentRole::Qwen => 4,
        NativeClipComponentRole::Gemma => 5,
    }
}

fn hash_clip_component_options(
    hasher: &mut Sha256,
    options: &NativeClipComponentOptions,
) -> Result<(), ClipError> {
    for value in [
        options.tokenizer.minimum_length,
        options.tokenizer.minimum_padding,
    ] {
        match value {
            Some(value) => {
                hasher.update([1]);
                hasher.update(
                    u64::try_from(value)
                        .map_err(|_| ClipError::Overflow("CLIP option digest"))?
                        .to_be_bytes(),
                );
            }
            None => hasher.update([0]),
        }
    }
    match options.tokenizer.disable_weights {
        Some(value) => hasher.update([1, u8::from(value)]),
        None => hasher.update([0]),
    }
    match &options.layer {
        NativeClipLayerSelection::Last => hasher.update([0]),
        NativeClipLayerSelection::Hidden(layer) => {
            hasher.update([1]);
            hasher.update(
                i64::try_from(*layer)
                    .map_err(|_| ClipError::Overflow("CLIP layer digest"))?
                    .to_be_bytes(),
            );
        }
        NativeClipLayerSelection::All => hasher.update([2]),
    }
    hasher.update([
        u8::from(options.final_layer_norm_intermediate),
        u8::from(options.projected_pooled),
        u8::from(options.attention_mask),
        u8::from(options.zero_out_masked),
    ]);
    Ok(())
}

fn hash_clip_bytes(hasher: &mut Sha256, bytes: &[u8]) -> Result<(), ClipError> {
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| ClipError::Overflow("CLIP digest field"))?
            .to_be_bytes(),
    );
    hasher.update(bytes);
    Ok(())
}

#[derive(Debug, Error)]
pub enum ClipError {
    #[error("source-key scaling is unavailable for CLIP component role {0:?}")]
    UnsupportedSourceStateTransform(NativeClipComponentRole),
    #[error("invalid native CLIP resource: {0}")]
    InvalidNativeResource(String),
    #[error("native CLIP owner failed: {0}")]
    NativeResourceOwner(String),
    #[error("invalid tokenizer identity: {0}")]
    InvalidTokenizerIdentity(String),
    #[error("native tokenizer error: {0}")]
    Tokenizer(String),
    #[error("token weight must be finite, got {0}")]
    InvalidWeight(f32),
    #[error("token sequence expected {expected} entries, got {actual}")]
    TokenShape { expected: usize, actual: usize },
    #[error("token sequence content length is invalid: {0}")]
    InvalidContentLength(usize),
    #[error("token sequence padding is not a suffix of end tokens")]
    InvalidPadding,
    #[error("token batch size must be in 1..={MAX_TOKEN_BATCH}, got {0}")]
    InvalidBatchSize(usize),
    #[error("prompt has too many weighted segments: {0}")]
    TooManyWeightedSegments(usize),
    #[error("prompt byte length exceeds the native tokenizer bound: {0}")]
    PromptTooLarge(usize),
    #[error("native CLIP allocation failed for {0}")]
    Allocation(&'static str),
    #[error("native CLIP arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("invalid {kind} SHA-256 identity {digest:?}")]
    InvalidBindingIdentity { kind: &'static str, digest: String },
    #[error("CLIP artifact set must contain 1..={MAX_CLIP_ARTIFACTS} entries, got {0}")]
    InvalidArtifactSet(usize),
    #[error("CLIP detector tensor {0} has no leading dimension")]
    InvalidDetectorTensor(String),
    #[error("CLIP artifact count expected {expected}, got {actual}")]
    ArtifactCount { expected: usize, actual: usize },
    #[error("CLIP patch base identity expected {expected}, got {actual}")]
    PatchBaseMismatch { expected: String, actual: String },
    #[error(transparent)]
    PatchIdentity(#[from] PatchGraphIdentityError),
    #[error(
        "canonical CLIP loader selected tokenizer {tokenizer:?} and model {clip_model:?}, but the resolved family does not expose that target"
    )]
    CanonicalTargetMismatch {
        tokenizer: String,
        clip_model: String,
    },
    #[error("CLIP manifest architecture expected {expected}, got {actual:?}")]
    ManifestArchitectureMismatch {
        expected: String,
        actual: Option<String>,
    },
    #[error("CLIP manifest detector binding expected {expected:?}, got {actual:?}")]
    ManifestDetectedArchitectureMismatch {
        expected: Vec<Option<TextEncoderModel>>,
        actual: Vec<Option<TextEncoderModel>>,
    },
    #[error("invalid CLIP parameter manifest: {0}")]
    InvalidParameterManifest(String),
    #[error("CLIP parameter manifest does not declare {0}")]
    MissingManifestParameter(String),
    #[error(
        "CLIP artifact {artifact_index} parameter set differs: missing {missing:?}, unexpected {unexpected:?}"
    )]
    ParameterSetMismatch {
        artifact_index: usize,
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[error("CLIP parameter {name} expected shape {expected:?}, got {actual:?}")]
    ManifestParameterShape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("CLIP parameter {name} expected storage dtype {expected}, got {actual}")]
    ManifestParameterDType {
        name: String,
        expected: String,
        actual: String,
    },
    #[error("CLIP parameter {0} has invalid f32 bytes")]
    InvalidParameterBytes(String),
    #[error("native CLIP projection does not match parameter {0}")]
    NativeProjectionMismatch(String),
    #[error("native CLIP projection contains a missing or unexpected parameter")]
    NativeProjectionSetMismatch,
    #[error("CLIP load requested {requested:?}, but the native loader owns {backend:?}")]
    LoadBackendMismatch {
        requested: DeviceId,
        backend: DeviceId,
    },
    #[error("CLIP parameter {name} has {storage:?} storage, but the load requested {requested:?}")]
    LoadDTypeMismatch {
        name: String,
        storage: DType,
        requested: DType,
    },
    #[error("CLIP parameter {name} uses unsupported native storage dtype {storage_dtype}")]
    UnsupportedManifestStorage { name: String, storage_dtype: String },
    #[error("CLIP binding identity expected kind {expected:?}, got {actual:?}")]
    InvalidBindingKind {
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "selected tokenizer {tokenizer:?} and CLIP model {clip_model:?} are not a target candidate"
    )]
    DescriptorMismatch {
        tokenizer: String,
        clip_model: String,
    },
    #[error("CLIP execution shape is invalid: context={context_length}, hidden={hidden_width}")]
    InvalidExecutionShape {
        context_length: usize,
        hidden_width: usize,
    },
    #[error("CLIP execution dtype must be floating point, got {0:?}")]
    InvalidExecutionDType(DType),
    #[error("CLIP identity serialization failed: {0}")]
    IdentitySerialization(String),
    #[error("CLIP encoding output does not match its execution plan")]
    OutputMismatch,
    #[error("CLIP parameter tensor {name} expected shape {expected:?}, got {actual:?}")]
    ParameterShape {
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<u64>,
    },
    #[error("CLIP parameter tensor {0} must be contiguous F32 on CPU")]
    ParameterStorage(&'static str),
    #[error("CLIP encoder dimensions or attention heads are invalid")]
    InvalidEncoderShape,
    #[error("token batch was produced by a different tokenizer identity")]
    TokenizerMismatch,
    #[error("CLIP artifact, model, or patch identity does not match loaded parameters")]
    BindingMismatch,
    #[error("SD1 CLIP executor supports CPU F32, got {dtype:?} on {device:?}")]
    UnsupportedConcreteTarget { dtype: DType, device: DeviceId },
    #[error("requested CLIP layer {requested}, but only {available} layers are loaded")]
    LayerOutOfRange { requested: usize, available: usize },
    #[error("CLIP parameters and caller execution context use different streams")]
    ParameterStreamMismatch,
    #[error("CLIP token {0} is outside the loaded vocabulary")]
    TokenOutOfRange(u32),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorOperation(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    ModelStore(#[from] ModelStoreError),
    #[error(transparent)]
    NativeModule(#[from] NativeOpsError),
    #[error(transparent)]
    TextTransformer(#[from] ClipTextError),
    #[error(transparent)]
    PatchGraph(#[from] PatchGraphError),
}

fn validate_sha256(kind: &'static str, digest: &str) -> Result<(), ClipError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ClipError::InvalidBindingIdentity {
            kind,
            digest: digest.to_owned(),
        });
    }
    Ok(())
}

fn artifact_bundle_identity(digests: &[String]) -> Result<String, ClipError> {
    if digests.is_empty() || digests.len() > MAX_CLIP_ARTIFACTS {
        return Err(ClipError::InvalidArtifactSet(digests.len()));
    }
    for digest in digests {
        validate_sha256("artifact", digest)?;
    }
    if digests.len() == 1 {
        return Ok(digests[0].clone());
    }
    let fields = digests
        .iter()
        .map(|digest| digest.as_bytes())
        .collect::<Vec<_>>();
    Ok(digest_fields(&fields))
}

#[allow(
    dead_code,
    reason = "registered CLIP architecture leaves validate module declarations here"
)]
fn validate_module_manifest(
    module: &ClipNativeModuleSpec,
    weight: &ClipParameterSpec,
    bias: Option<&ClipParameterSpec>,
) -> Result<(), ClipError> {
    let (expected_weight, expected_bias) = match &module.kind {
        ClipNativeModuleKind::Linear {
            input_features,
            output_features,
        } => (
            vec![
                u64::try_from(*output_features)
                    .map_err(|_| ClipError::Overflow("linear output features"))?,
                u64::try_from(*input_features)
                    .map_err(|_| ClipError::Overflow("linear input features"))?,
            ],
            Some(vec![
                u64::try_from(*output_features)
                    .map_err(|_| ClipError::Overflow("linear bias features"))?,
            ]),
        ),
        ClipNativeModuleKind::Embedding {
            embeddings,
            dimensions,
        } => (
            vec![
                u64::try_from(*embeddings).map_err(|_| ClipError::Overflow("embedding count"))?,
                u64::try_from(*dimensions)
                    .map_err(|_| ClipError::Overflow("embedding dimensions"))?,
            ],
            None,
        ),
        ClipNativeModuleKind::LayerNorm {
            normalized_shape,
            epsilon_bits,
        } => {
            let epsilon = f32::from_bits(*epsilon_bits);
            if normalized_shape.is_empty()
                || normalized_shape.contains(&0)
                || !epsilon.is_finite()
                || epsilon <= 0.0
            {
                return Err(ClipError::InvalidParameterManifest(
                    "layer-normalization projection is invalid".to_owned(),
                ));
            }
            (
                normalized_shape
                    .iter()
                    .map(|value| {
                        u64::try_from(*value)
                            .map_err(|_| ClipError::Overflow("layer-normalization shape"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                Some(
                    normalized_shape
                        .iter()
                        .map(|value| {
                            u64::try_from(*value)
                                .map_err(|_| ClipError::Overflow("layer-normalization bias shape"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            )
        }
    };
    if weight.shape != expected_weight {
        return Err(ClipError::ManifestParameterShape {
            name: weight.name.clone(),
            expected: expected_weight,
            actual: weight.shape.clone(),
        });
    }
    match (bias, expected_bias) {
        (None, None) | (None, Some(_)) => {}
        (Some(_), None) => {
            return Err(ClipError::InvalidParameterManifest(format!(
                "module {:?} does not accept a bias",
                module.name
            )));
        }
        (Some(bias), Some(expected)) if bias.shape == expected => {}
        (Some(bias), Some(expected)) => {
            return Err(ClipError::ManifestParameterShape {
                name: bias.name.clone(),
                expected,
                actual: bias.shape.clone(),
            });
        }
    }
    Ok(())
}

#[allow(
    dead_code,
    reason = "registered CLIP architecture leaves construct modules through this adapter"
)]
fn native_module_from_spec(module: &ClipNativeModuleSpec) -> Result<NativeModule, ClipError> {
    match &module.kind {
        ClipNativeModuleKind::Linear {
            input_features,
            output_features,
        } => Ok(NativeModule::linear(
            &module.name,
            *input_features,
            *output_features,
            module.bias.is_some(),
            false,
        )?),
        ClipNativeModuleKind::Embedding {
            embeddings,
            dimensions,
        } => Ok(NativeModule::embedding(
            &module.name,
            *embeddings,
            *dimensions,
            EmbeddingOptions::default(),
            false,
        )?),
        ClipNativeModuleKind::LayerNorm {
            normalized_shape,
            epsilon_bits,
        } => Ok(NativeModule::layer_norm(
            &module.name,
            normalized_shape.clone(),
            f32::from_bits(*epsilon_bits),
            true,
            module.bias.is_some(),
            false,
        )?),
    }
}

fn byte_encoder() -> Result<BTreeMap<u8, char>, ClipError> {
    let mut bytes = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect::<Vec<_>>();
    let mut codepoints = bytes
        .iter()
        .map(|value| u32::from(*value))
        .collect::<Vec<_>>();
    let mut extra = 0_u32;
    for byte in 0_u16..=255 {
        let byte = u8::try_from(byte).map_err(|_| ClipError::Overflow("byte encoder"))?;
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + extra);
            extra += 1;
        }
    }
    bytes
        .into_iter()
        .zip(codepoints)
        .map(|(byte, codepoint)| {
            char::from_u32(codepoint)
                .map(|character| (byte, character))
                .ok_or_else(|| {
                    ClipError::Tokenizer("byte encoder code point is invalid".to_owned())
                })
        })
        .collect()
}

fn require_binding_kind(
    identity: &ClipBindingIdentity,
    expected: &'static str,
) -> Result<(), ClipError> {
    if identity.kind() != expected {
        return Err(ClipError::InvalidBindingKind {
            expected,
            actual: identity.kind(),
        });
    }
    Ok(())
}

fn require_tensor(
    tensor: &Tensor,
    expected: &[usize],
    name: &'static str,
) -> Result<(), ClipError> {
    let expected_u64 = expected
        .iter()
        .copied()
        .map(|value| u64::try_from(value).map_err(|_| ClipError::Overflow("parameter shape")))
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.descriptor().shape() != expected_u64 {
        return Err(ClipError::ParameterShape {
            name,
            expected: expected.to_vec(),
            actual: tensor.descriptor().shape().to_vec(),
        });
    }
    if tensor.descriptor().dtype() != DType::F32
        || tensor.descriptor().device() != DeviceId::CPU
        || !tensor.descriptor().is_contiguous()?
    {
        return Err(ClipError::ParameterStorage(name));
    }
    Ok(())
}

fn require_layer(layer: &Sd1ClipLayerTensors, hidden_width: usize) -> Result<(), ClipError> {
    let intermediate = usize::try_from(
        *layer
            .feed_forward_1_weight
            .descriptor()
            .shape()
            .first()
            .ok_or(ClipError::InvalidEncoderShape)?,
    )
    .map_err(|_| ClipError::Overflow("feed-forward width"))?;
    if intermediate == 0 {
        return Err(ClipError::InvalidEncoderShape);
    }
    for (tensor, shape, name) in [
        (
            &layer.layer_norm_1_weight,
            vec![hidden_width],
            "layer norm 1 weight",
        ),
        (
            &layer.layer_norm_1_bias,
            vec![hidden_width],
            "layer norm 1 bias",
        ),
        (
            &layer.query_weight,
            vec![hidden_width, hidden_width],
            "query weight",
        ),
        (&layer.query_bias, vec![hidden_width], "query bias"),
        (
            &layer.key_weight,
            vec![hidden_width, hidden_width],
            "key weight",
        ),
        (&layer.key_bias, vec![hidden_width], "key bias"),
        (
            &layer.value_weight,
            vec![hidden_width, hidden_width],
            "value weight",
        ),
        (&layer.value_bias, vec![hidden_width], "value bias"),
        (
            &layer.output_weight,
            vec![hidden_width, hidden_width],
            "output weight",
        ),
        (&layer.output_bias, vec![hidden_width], "output bias"),
        (
            &layer.layer_norm_2_weight,
            vec![hidden_width],
            "layer norm 2 weight",
        ),
        (
            &layer.layer_norm_2_bias,
            vec![hidden_width],
            "layer norm 2 bias",
        ),
        (
            &layer.feed_forward_1_weight,
            vec![intermediate, hidden_width],
            "feed forward 1 weight",
        ),
        (
            &layer.feed_forward_1_bias,
            vec![intermediate],
            "feed forward 1 bias",
        ),
        (
            &layer.feed_forward_2_weight,
            vec![hidden_width, intermediate],
            "feed forward 2 weight",
        ),
        (
            &layer.feed_forward_2_bias,
            vec![hidden_width],
            "feed forward 2 bias",
        ),
    ] {
        require_tensor(tensor, &shape, name)?;
    }
    Ok(())
}

fn parameter_tensors(tensors: &Sd1ClipTensors) -> Vec<&Tensor> {
    let mut all = vec![
        &tensors.token_embedding,
        &tensors.position_embedding,
        &tensors.final_layer_norm_weight,
        &tensors.final_layer_norm_bias,
    ];
    for layer in &tensors.layers {
        all.extend([
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
    all
}

fn sd1_token_input(
    backend: &CpuBackend,
    batch: &TokenBatch,
    include_empty_baseline: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ClipError> {
    let batch_size = batch
        .rows()
        .len()
        .checked_add(usize::from(include_empty_baseline))
        .ok_or(ClipError::Overflow("weighted CLIP batch"))?;
    let value_count = batch_size
        .checked_mul(batch.context_length())
        .ok_or(ClipError::Overflow("CLIP token input"))?;
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or(ClipError::Overflow("CLIP token input bytes"))?;
    let mut bytes = backend.workspace_vec(context, byte_count)?;
    for row in batch.rows() {
        for token in row.tokens() {
            context.check()?;
            for byte in i64::from(token.token()).to_ne_bytes() {
                bytes.try_push(byte)?;
            }
        }
    }
    if include_empty_baseline {
        for position in 0..batch.context_length() {
            context.check()?;
            let token = if position == 0 {
                SD1_START_TOKEN
            } else {
                SD1_END_TOKEN
            };
            for byte in i64::from(token).to_ne_bytes() {
                bytes.try_push(byte)?;
            }
        }
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(2)
        .map_err(|_| ClipError::Allocation("CLIP token input shape"))?;
    shape.push(u64::try_from(batch_size).map_err(|_| ClipError::Overflow("batch size"))?);
    shape.push(
        u64::try_from(batch.context_length()).map_err(|_| ClipError::Overflow("context length"))?,
    );
    let descriptor =
        TensorDescriptor::contiguous(shape, DType::I64, DeviceId::CPU, context.stream)?;
    let (tokens, event) = backend.upload_bytes(descriptor, &bytes, context)?;
    backend.wait_event(event, context)?;
    Ok(tokens)
}

fn pool_values(
    backend: &CpuBackend,
    values: &[f32],
    batch: &TokenBatch,
    hidden_width: usize,
    pooling: ClipPooling,
    context: &ExecutionContext<'_>,
) -> Result<Option<CpuWorkspaceVec<f32>>, ClipError> {
    if pooling == ClipPooling::None {
        return Ok(None);
    }
    let output_count = batch
        .rows()
        .len()
        .checked_mul(hidden_width)
        .ok_or(ClipError::Overflow("pooled output"))?;
    let mut pooled = backend.workspace_vec(context, output_count)?;
    for (batch_index, row) in batch.rows().iter().enumerate() {
        context.check()?;
        match pooling {
            ClipPooling::None => unreachable!(),
            ClipPooling::EndToken => {
                let offset = batch_index
                    .checked_mul(batch.context_length())
                    .and_then(|value| value.checked_add(row.first_end_index()))
                    .and_then(|value| value.checked_mul(hidden_width))
                    .ok_or(ClipError::Overflow("end-token pool offset"))?;
                let end = offset
                    .checked_add(hidden_width)
                    .ok_or(ClipError::Overflow("end-token pool range"))?;
                for value in values.get(offset..end).ok_or(ClipError::OutputMismatch)? {
                    pooled.try_push(*value)?;
                }
            }
            ClipPooling::Mean => {
                let token_count = row
                    .first_end_index()
                    .checked_add(1)
                    .ok_or(ClipError::Overflow("mean token count"))?;
                for channel in 0..hidden_width {
                    let mut sum = 0.0_f32;
                    for token in 0..token_count {
                        let offset = batch_index
                            .checked_mul(batch.context_length())
                            .and_then(|value| value.checked_add(token))
                            .and_then(|value| value.checked_mul(hidden_width))
                            .and_then(|value| value.checked_add(channel))
                            .ok_or(ClipError::Overflow("mean pool offset"))?;
                        sum += *values.get(offset).ok_or(ClipError::OutputMismatch)?;
                    }
                    pooled.try_push(sum / token_count as f32)?;
                }
            }
        }
    }
    Ok(Some(pooled))
}

fn apply_token_weights(
    backend: &CpuBackend,
    values: &[f32],
    batch: &TokenBatch,
    hidden_width: usize,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, ClipError> {
    let row_width = batch
        .context_length()
        .checked_mul(hidden_width)
        .ok_or(ClipError::Overflow("CLIP row width"))?;
    let output_count = batch
        .rows()
        .len()
        .checked_mul(row_width)
        .ok_or(ClipError::Overflow("weighted CLIP output"))?;
    let baseline_offset = output_count;
    if values.len()
        != output_count
            .checked_add(row_width)
            .ok_or(ClipError::Overflow("weighted CLIP baseline"))?
    {
        return Err(ClipError::OutputMismatch);
    }
    let mut weighted = backend.workspace_vec(context, output_count)?;
    for (batch_index, row) in batch.rows().iter().enumerate() {
        for (position, token) in row.tokens().iter().enumerate() {
            context.check()?;
            for channel in 0..hidden_width {
                let position_offset = position
                    .checked_mul(hidden_width)
                    .and_then(|value| value.checked_add(channel))
                    .ok_or(ClipError::Overflow("weighted CLIP position offset"))?;
                let offset = batch_index
                    .checked_mul(row_width)
                    .and_then(|value| value.checked_add(position_offset))
                    .ok_or(ClipError::Overflow("weighted CLIP offset"))?;
                let baseline_index = baseline_offset
                    .checked_add(position_offset)
                    .ok_or(ClipError::Overflow("weighted CLIP baseline offset"))?;
                let baseline = *values
                    .get(baseline_index)
                    .ok_or(ClipError::OutputMismatch)?;
                let value = *values.get(offset).ok_or(ClipError::OutputMismatch)?;
                weighted.try_push((value - baseline).mul_add(token.weight(), baseline))?;
            }
        }
    }
    Ok(weighted)
}

fn digest_f32(
    values: &[f32],
    pooled: Option<&[f32]>,
    context: &ExecutionContext<'_>,
) -> Result<String, ClipError> {
    let mut hasher = Sha256::new();
    for (index, value) in values.iter().enumerate() {
        if index.is_multiple_of(1_024) {
            context.check()?;
        }
        hasher.update(value.to_bits().to_le_bytes());
    }
    if let Some(pooled) = pooled {
        for value in pooled {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_fields(fields: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for field in fields {
        update_digest_field(&mut hasher, field);
    }
    format!("{:x}", hasher.finalize())
}

fn update_digest_field(hasher: &mut Sha256, field: &[u8]) {
    hasher.update((field.len() as u64).to_le_bytes());
    hasher.update(field);
}

fn u64_bytes(value: usize) -> Result<[u8; 8], ClipError> {
    Ok(u64::try_from(value)
        .map_err(|_| ClipError::Overflow("identity field"))?
        .to_le_bytes())
}

fn u64_from_usize(value: usize) -> Result<u64, ClipError> {
    u64::try_from(value).map_err(|_| ClipError::Overflow("parameter dimension"))
}

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub fn native_clip_transform_fixture(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipResource, ClipError> {
    let tensor = |shape: &[u64], values: Vec<f32>| {
        tensor_from_f32(backend, shape, &values, context).map_err(ClipError::from)
    };
    let vector = |values| tensor(&[2], values);
    let matrix = |values| tensor(&[2, 2], values);
    let layer = ClipTextLayerWeights {
        layer_norm_1_weight: vector(vec![1.0, 1.0])?,
        layer_norm_1_bias: vector(vec![0.0, 0.0])?,
        query_weight: matrix(vec![0.0; 4])?,
        query_bias: vector(vec![0.0; 2])?,
        key_weight: matrix(vec![0.0; 4])?,
        key_bias: vector(vec![0.0; 2])?,
        value_weight: matrix(vec![0.0; 4])?,
        value_bias: vector(vec![0.0; 2])?,
        output_weight: matrix(vec![0.0; 4])?,
        output_bias: vector(vec![0.0; 2])?,
        layer_norm_2_weight: vector(vec![1.0, 1.0])?,
        layer_norm_2_bias: vector(vec![0.0, 0.0])?,
        feed_forward_1_weight: tensor(&[4, 2], vec![0.0; 8])?,
        feed_forward_1_bias: tensor(&[4], vec![0.0; 4])?,
        feed_forward_2_weight: tensor(&[2, 4], vec![0.0; 8])?,
        feed_forward_2_bias: vector(vec![0.0; 2])?,
    };
    let weights = ClipTextWeights {
        token_embedding: tensor(
            &[SD1_VOCABULARY_SIZE as u64, 2],
            vec![0.0; SD1_VOCABULARY_SIZE * 2],
        )?,
        position_embedding: tensor(
            &[SD1_CONTEXT_LENGTH as u64, 2],
            vec![0.0; SD1_CONTEXT_LENGTH * 2],
        )?,
        layers: vec![layer],
        final_layer_norm_weight: vector(vec![1.0, 1.0])?,
        final_layer_norm_bias: vector(vec![0.0, 0.0])?,
    };
    let configuration = ClipTextConfiguration {
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: SD1_VOCABULARY_SIZE,
        max_position_embeddings: SD1_CONTEXT_LENGTH,
        hidden_size: 2,
        intermediate_size: 4,
        attention_heads: 1,
        layer_count: 1,
        eos_token_id: SD1_END_TOKEN,
        activation: ClipTextActivation::QuickGelu,
        projection_dimension: None,
    };
    let template = NativeClipText::new(configuration, weights.clone(), None)?;
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(
            ClipBpeTokenizer::from_json_and_merges(
                ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")
                    .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?,
                include_str!("../../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"),
                include_str!("../../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"),
            )
            .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?,
        ),
        TokenizerConfiguration {
            maximum_length: SD1_CONTEXT_LENGTH,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: true,
            pad_left: false,
            start_token: Some(SD1_START_TOKEN),
            end_token: Some(SD1_END_TOKEN),
            pad_token: SD1_END_TOKEN,
            maximum_word_length: 8,
            disable_weights: false,
            embedding_width: Some(2),
        },
        BTreeMap::new(),
    )
    .map_err(|error| ClipError::InvalidNativeResource(error.to_string()))?;
    let artifact = "c".repeat(64);
    let mut state = BTreeMap::from([
        (
            "token_embedding.weight".to_owned(),
            weights.token_embedding.clone(),
        ),
        (
            "position_embedding.weight".to_owned(),
            weights.position_embedding.clone(),
        ),
        (
            "final_layer_norm.weight".to_owned(),
            weights.final_layer_norm_weight.clone(),
        ),
        (
            "final_layer_norm.bias".to_owned(),
            weights.final_layer_norm_bias.clone(),
        ),
    ]);
    let layer = weights.layers.first().ok_or_else(|| {
        ClipError::InvalidNativeResource("CLIP transform fixture layer is missing".to_owned())
    })?;
    for (name, tensor) in [
        ("layer_norm_1.weight", &layer.layer_norm_1_weight),
        ("layer_norm_1.bias", &layer.layer_norm_1_bias),
        ("query.weight", &layer.query_weight),
        ("query.bias", &layer.query_bias),
        ("key.weight", &layer.key_weight),
        ("key.bias", &layer.key_bias),
        ("value.weight", &layer.value_weight),
        ("value.bias", &layer.value_bias),
        ("output.weight", &layer.output_weight),
        ("output.bias", &layer.output_bias),
        ("layer_norm_2.weight", &layer.layer_norm_2_weight),
        ("layer_norm_2.bias", &layer.layer_norm_2_bias),
        ("feed_forward_1.weight", &layer.feed_forward_1_weight),
        ("feed_forward_1.bias", &layer.feed_forward_1_bias),
        ("feed_forward_2.weight", &layer.feed_forward_2_weight),
        ("feed_forward_2.bias", &layer.feed_forward_2_bias),
    ] {
        state.insert(format!("layers.0.{name}"), tensor.clone());
    }
    let component = NativeClipComponent::checked_clip_text(
        NativeClipComponentRole::ClipL,
        Arc::new(MappedModelWeights::from_parts(artifact, state, Vec::new())),
        Arc::new(tokenizer),
        &template,
        NativeClipComponentOptions::checked_source(
            NativeTokenizerInvocationOptions::default(),
            NativeClipLayerSelection::Last,
            true,
            false,
            false,
            false,
        )?,
        context.cancellation,
    )?;
    NativeClipResource::checked(
        NativeClipProfile::Sd1,
        false,
        vec![Arc::new(component)],
        Vec::new(),
        backend,
        context,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArtifactKey, ArtifactRoot, ClipBpeTokenizer, ModelClipTargetCandidateDescriptor,
        ModelParsedFacts, ModelParsedTensorFact, NativeModelBackingKind, NativeModelPayload,
        NativeTokenizerFamily, ParserLimits, PatchApplication, PatchKind, PatchOperation,
        PatchTarget, TokenizerConfiguration,
    };
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use sim_types::DeviceKind;
    use std::{fs, io::Write, path::Path};

    fn fixture_tokenizer() -> Result<Sd1Tokenizer, Box<dyn std::error::Error>> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let fixture = root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
        Ok(Sd1Tokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
            &fs::read_to_string(fixture.join("vocab.json"))?,
            &fs::read_to_string(fixture.join("merges.txt"))?,
        )?)
    }

    fn binding(byte: char) -> Result<ClipBindingIdentity, ClipError> {
        let kind = match byte {
            'a' => "artifact",
            'b' => "model",
            'c' => "patch",
            _ => "test",
        };
        let digest = byte.to_string().repeat(64);
        validate_sha256(kind, &digest)?;
        Ok(ClipBindingIdentity::derived(kind, digest))
    }

    fn target() -> Result<ModelClipTargetDescriptor, Box<dyn std::error::Error>> {
        Ok(ModelClipTargetDescriptor::checked(
            vec![ModelClipTargetCandidateDescriptor::checked(
                "sim.sd1.tokenizer",
                "sim.sd1.clip",
            )?],
            false,
        )?)
    }

    fn detector_fixture(model: TextEncoderModel) -> BTreeMap<String, Vec<u64>> {
        let mut shapes = BTreeMap::new();
        let mut insert = |name: &str, width: u64| {
            shapes.insert(name.to_owned(), vec![width, 2]);
        };
        match model {
            TextEncoderModel::ClipL => insert("text_model.encoder.layers.0.mlp.fc1.weight", 1),
            TextEncoderModel::ClipH => insert("text_model.encoder.layers.22.mlp.fc1.weight", 1),
            TextEncoderModel::ClipG => insert("text_model.encoder.layers.30.mlp.fc1.weight", 1),
            TextEncoderModel::JinaClip2 => insert("model.encoder.layers.0.mixer.Wqkv.weight", 1),
            TextEncoderModel::T5Xxl => insert(
                "encoder.block.23.layer.1.DenseReluDense.wi_1.weight",
                10_240,
            ),
            TextEncoderModel::T5Xl => {
                insert("encoder.block.23.layer.1.DenseReluDense.wi_1.weight", 5_120)
            }
            TextEncoderModel::T5XxlOld => {
                insert("encoder.block.23.layer.1.DenseReluDense.wi.weight", 1)
            }
            TextEncoderModel::ByT5SmallGlyph => {
                insert("encoder.block.0.layer.0.SelfAttention.k.weight", 384)
            }
            TextEncoderModel::T5Base => {
                insert("encoder.block.0.layer.0.SelfAttention.k.weight", 768)
            }
            TextEncoderModel::T5Gemma => {
                insert("model.encoder.layers.0.pre_self_attn_layernorm.weight", 1)
            }
            TextEncoderModel::Gemma4_31b => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1);
                insert("model.layers.59.self_attn.q_norm.weight", 1);
            }
            TextEncoderModel::Gemma4E4b => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1);
                insert("model.layers.41.self_attn.q_norm.weight", 1);
            }
            TextEncoderModel::Gemma4E2b => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1);
                insert("model.layers.34.self_attn.q_norm.weight", 1);
            }
            TextEncoderModel::Gemma3_12b => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1);
                insert("model.layers.47.self_attn.q_norm.weight", 1);
            }
            TextEncoderModel::Gemma3_4bVision => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1);
                insert("model.layers.0.self_attn.q_norm.weight", 1);
                insert("vision_model.embeddings.patch_embedding.weight", 1);
            }
            TextEncoderModel::Gemma3_4b => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1);
                insert("model.layers.0.self_attn.q_norm.weight", 1);
            }
            TextEncoderModel::Gemma2_2b => {
                insert("model.layers.0.post_feedforward_layernorm.weight", 1)
            }
            TextEncoderModel::GptOss20b => {
                insert("layers.0.self_attn.sinks", 1);
                insert("layers.0.mlp.experts.gate_up_proj.weight", 1);
            }
            TextEncoderModel::Qwen25_3b => insert("model.layers.0.self_attn.k_proj.bias", 256),
            TextEncoderModel::Qwen25_7b => insert("model.layers.0.self_attn.k_proj.bias", 512),
            TextEncoderModel::Qwen35_08b
            | TextEncoderModel::Qwen35_2b
            | TextEncoderModel::Qwen35_4b
            | TextEncoderModel::Qwen35_9b
            | TextEncoderModel::Qwen35_27b => {
                insert("model.language_model.layers.0.linear_attn.A_log", 1);
                let width = match model {
                    TextEncoderModel::Qwen35_08b => 1_024,
                    TextEncoderModel::Qwen35_4b => 2_560,
                    TextEncoderModel::Qwen35_9b => 4_096,
                    TextEncoderModel::Qwen35_27b => 5_120,
                    _ => 2_048,
                };
                insert(
                    "model.language_model.layers.0.input_layernorm.weight",
                    width,
                );
            }
            TextEncoderModel::Qwen3Vl4b | TextEncoderModel::Qwen3Vl8b => {
                insert(
                    &format!("model.{HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT}.0.norm.weight"),
                    1,
                );
                insert(
                    "model.visual.merger.linear_fc2.weight",
                    if model == TextEncoderModel::Qwen3Vl4b {
                        2_560
                    } else {
                        4_096
                    },
                );
            }
            TextEncoderModel::Qwen3_4b
            | TextEncoderModel::Qwen3_2b
            | TextEncoderModel::Qwen3_8b
            | TextEncoderModel::Qwen3_06b => {
                let width = match model {
                    TextEncoderModel::Qwen3_4b => 2_560,
                    TextEncoderModel::Qwen3_2b => 2_048,
                    TextEncoderModel::Qwen3_8b => 4_096,
                    _ => 1_024,
                };
                insert("model.layers.0.post_attention_layernorm.weight", width);
                insert("model.layers.0.self_attn.q_norm.weight", 1);
            }
            TextEncoderModel::Mistral3_24b | TextEncoderModel::Mistral3_24bPrunedFlux2 => {
                insert("model.layers.0.post_attention_layernorm.weight", 5_120);
                if model == TextEncoderModel::Mistral3_24b {
                    insert("model.layers.39.post_attention_layernorm.weight", 1);
                }
            }
            TextEncoderModel::Ministral3_3b => {
                insert("model.layers.0.post_attention_layernorm.weight", 3_072)
            }
            TextEncoderModel::Llama3_8 => {
                insert("model.layers.0.post_attention_layernorm.weight", 3_584)
            }
        }
        shapes
    }

    fn detector_probe(model: TextEncoderModel) -> Result<ModelProbe, ClipError> {
        ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: detector_fixture(model)
                .into_iter()
                .map(|(name, shape)| {
                    (
                        name,
                        ModelParsedTensorFact {
                            shape,
                            storage_dtype: "F32".to_owned(),
                        },
                    )
                })
                .collect(),
            formats: Vec::new(),
        })
        .map_err(|error| ClipError::Tokenizer(error.to_string()))
    }

    #[test]
    fn pinned_sd_clip_type_and_te_model_rows_are_complete_and_source_ordered()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(ClipType::ALL.len(), 32);
        for (index, clip_type) in ClipType::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(clip_type.source_ordinal()), index + 1);
        }
        assert_eq!(TextEncoderModel::ALL.len(), 35);
        for (index, model) in TextEncoderModel::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(model.source_ordinal()), index + 1);
            assert_eq!(
                detect_text_encoder_model(&detector_fixture(model))?,
                Some(model)
            );
        }
        assert_eq!(detect_text_encoder_model(&BTreeMap::new())?, None);

        let mut precedence = detector_fixture(TextEncoderModel::Qwen25_7b);
        precedence.extend(detector_fixture(TextEncoderModel::GptOss20b));
        assert_eq!(
            detect_text_encoder_model(&precedence)?,
            Some(TextEncoderModel::GptOss20b)
        );
        assert!(matches!(
            detect_text_encoder_model(&BTreeMap::from([(
                format!("model.{HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT}.0.norm.weight"),
                vec![1]
            )])),
            Err(ClipError::InvalidDetectorTensor(_))
        ));
        Ok(())
    }

    #[test]
    fn source_loader_table_projects_configuration_and_ordered_one_to_four_artifacts()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut t5_tensors = detector_fixture(TextEncoderModel::T5Xxl)
            .into_iter()
            .map(|(name, shape)| {
                (
                    name,
                    ModelParsedTensorFact {
                        shape,
                        storage_dtype: "F32".to_owned(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        t5_tensors.insert(
            "encoder.final_layer_norm.weight".to_owned(),
            ModelParsedTensorFact {
                shape: vec![4_096],
                storage_dtype: "F16".to_owned(),
            },
        );
        t5_tensors.insert(
            "encoder.block.0.layer.0.sim_quant".to_owned(),
            ModelParsedTensorFact {
                shape: vec![1],
                storage_dtype: "U8".to_owned(),
            },
        );
        let t5 = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: t5_tensors,
            formats: Vec::new(),
        })?;
        let t5_configuration = t5xxl_detect(std::slice::from_ref(&t5))?.ok_or("t5 config")?;
        assert_eq!(t5_configuration.artifact_index(), 0);
        assert_eq!(t5_configuration.weight_dtype(), Some("float16"));
        assert!(t5_configuration.mixed_per_layer_quantization());

        let mut llama_tensors = detector_fixture(TextEncoderModel::Llama3_8)
            .into_iter()
            .map(|(name, shape)| {
                (
                    name,
                    ModelParsedTensorFact {
                        shape,
                        storage_dtype: "F32".to_owned(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        llama_tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_owned(),
            ModelParsedTensorFact {
                shape: vec![2, 2],
                storage_dtype: "F32".to_owned(),
            },
        );
        llama_tensors.insert(
            "model.norm.weight".to_owned(),
            ModelParsedTensorFact {
                shape: vec![2],
                storage_dtype: "BF16".to_owned(),
            },
        );
        let llama = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: llama_tensors,
            formats: Vec::new(),
        })?;
        let llama_configuration = llama_detect(&[t5.clone(), llama.clone()])?.ok_or("llama")?;
        assert_eq!(llama_configuration.artifact_index(), 1);
        assert_eq!(llama_configuration.weight_dtype(), Some("bfloat16"));

        let mut language_model_tensors = detector_fixture(TextEncoderModel::Qwen35_2b)
            .into_iter()
            .map(|(name, shape)| {
                (
                    name,
                    ModelParsedTensorFact {
                        shape,
                        storage_dtype: "F32".to_owned(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        language_model_tensors.insert(
            "model.language_model.layers.0.self_attn.k_proj.weight".to_owned(),
            ModelParsedTensorFact {
                shape: vec![2, 2],
                storage_dtype: "F32".to_owned(),
            },
        );
        language_model_tensors.insert(
            "model.language_model.norm.weight".to_owned(),
            ModelParsedTensorFact {
                shape: vec![2],
                storage_dtype: "BF16".to_owned(),
            },
        );
        let language_model = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: language_model_tensors,
            formats: Vec::new(),
        })?;
        assert_eq!(llama_detect(std::slice::from_ref(&language_model))?, None);
        let normalized_qwen35 = select_clip_architecture(
            ClipType::StableDiffusion,
            std::slice::from_ref(&language_model),
        )?;
        assert_eq!(
            normalized_qwen35
                .llama_configuration()
                .and_then(ClipDetectedConfiguration::weight_dtype),
            Some("bfloat16")
        );

        let single = select_clip_architecture(ClipType::Wan, std::slice::from_ref(&t5))?;
        assert_eq!(single.tokenizer(), "sim.text_encoders.wan.WanT5Tokenizer");
        assert_eq!(single.clip_model(), "sim.text_encoders.wan.te");
        let cogvideo = select_clip_architecture(ClipType::Cogvideox, std::slice::from_ref(&t5))?;
        let canonical_cogvideo = ModelClipTargetCandidateDescriptor::checked(
            "sim.text_encoders.cogvideo.CogVideoXT5Tokenizer",
            "sim.text_encoders.sd3_clip.T5XXLModel",
        )?;
        assert!(canonical_candidate_matches(&canonical_cogvideo, &cogvideo));
        let substituted_cogvideo = ModelClipTargetCandidateDescriptor::checked(
            "sim.text_encoders.cogvideo.CogVideoXT5Tokenizer",
            "sim.text_encoders.sd3_clip.sd3_clip",
        )?;
        assert!(!canonical_candidate_matches(
            &substituted_cogvideo,
            &cogvideo
        ));
        let dual = select_clip_architecture(
            ClipType::Flux,
            &[detector_probe(TextEncoderModel::ClipL)?, t5.clone()],
        )?;
        assert_eq!(dual.tokenizer(), "sim.text_encoders.flux.FluxTokenizer");
        let reversed = select_clip_architecture(
            ClipType::Flux,
            &[t5.clone(), detector_probe(TextEncoderModel::ClipL)?],
        )?;
        assert_ne!(dual.digest(), reversed.digest());
        let triple = select_clip_architecture(
            ClipType::StableDiffusion,
            &[
                detector_probe(TextEncoderModel::ClipL)?,
                detector_probe(TextEncoderModel::ClipG)?,
                t5.clone(),
            ],
        )?;
        assert_eq!(triple.clip_model(), "sim.text_encoders.sd3_clip.sd3_clip");
        let quadruple = select_clip_architecture(
            ClipType::StableDiffusion,
            &[
                detector_probe(TextEncoderModel::ClipL)?,
                detector_probe(TextEncoderModel::ClipG)?,
                t5,
                llama,
            ],
        )?;
        assert_eq!(
            quadruple.clip_model(),
            "sim.text_encoders.hidream.hidream_clip"
        );
        Ok(())
    }

    #[test]
    fn every_source_loader_decision_row_selects_the_exact_native_target()
    -> Result<(), Box<dyn std::error::Error>> {
        use TextEncoderModel as T;
        let single_rows = [
            (T::ClipL, "sim.sd1_clip.SD1ClipModel"),
            (T::ClipH, "sim.text_encoders.sd2_clip.SD2ClipModel"),
            (T::ClipG, "sim.sdxl_clip.SDXLRefinerClipModel"),
            (T::T5Xxl, "sim.text_encoders.genmo.mochi_te"),
            (T::T5Xl, "sim.text_encoders.aura_t5.AuraT5Model"),
            (T::T5Base, "sim.text_encoders.sa_t5.SAT5Model"),
            (T::Llama3_8, "sim.text_encoders.hidream.hidream_clip"),
            (T::T5XxlOld, "sim.text_encoders.cosmos.te"),
            (T::Gemma2_2b, "sim.text_encoders.lumina2.te"),
            (T::Qwen25_3b, "sim.text_encoders.omnigen2.te"),
            (T::Qwen25_7b, "sim.text_encoders.qwen_image.te"),
            (T::ByT5SmallGlyph, "sim.sd1_clip.SD1ClipModel"),
            (T::Gemma3_4b, "sim.text_encoders.lumina2.te"),
            (T::Mistral3_24b, "sim.text_encoders.flux.flux2_te"),
            (
                T::Mistral3_24bPrunedFlux2,
                "sim.text_encoders.flux.flux2_te",
            ),
            (T::Qwen3_4b, "sim.text_encoders.z_image.te"),
            (T::Qwen3_2b, "sim.text_encoders.ovis.te"),
            (T::Gemma3_12b, "sim.text_encoders.lt.gemma3_te"),
            (
                T::JinaClip2,
                "sim.text_encoders.jina_clip_2.JinaClip2TextModelWrapper",
            ),
            (T::Qwen3_8b, "sim.text_encoders.flux.klein_te"),
            (T::Qwen3_06b, "sim.text_encoders.anima.te"),
            (T::Gemma3_4bVision, "sim.text_encoders.lumina2.te"),
            (T::Qwen35_08b, "sim.text_encoders.qwen35.te"),
            (T::Qwen35_2b, "sim.text_encoders.qwen35.te"),
            (T::Qwen35_4b, "sim.text_encoders.qwen35.te"),
            (T::Qwen35_9b, "sim.text_encoders.qwen35.te"),
            (T::Qwen35_27b, "sim.text_encoders.qwen35.te"),
            (T::Ministral3_3b, "sim.text_encoders.ernie.te"),
            (T::Gemma4E4b, "sim.text_encoders.gemma4.gemma4_te"),
            (T::Gemma4E2b, "sim.text_encoders.gemma4.gemma4_te"),
            (T::Gemma4_31b, "sim.text_encoders.gemma4.gemma4_te"),
            (T::T5Gemma, "sim.text_encoders.sa3.SAT5GemmaModel"),
            (T::GptOss20b, "sim.text_encoders.gpt_oss.lens_te"),
            (T::Qwen3Vl4b, "sim.text_encoders.qwen3vl.te"),
            (T::Qwen3Vl8b, "sim.text_encoders.qwen3vl.te"),
        ];
        for (model, expected) in single_rows {
            let selection =
                select_clip_architecture(ClipType::StableDiffusion, &[detector_probe(model)?])?;
            assert_eq!(selection.clip_model(), expected, "{model:?}");
        }

        let conditional_rows = [
            (
                T::ClipG,
                ClipType::StableCascade,
                "sim.sdxl_clip.StableCascadeClipModel",
            ),
            (
                T::ClipG,
                ClipType::Sd3,
                "sim.text_encoders.sd3_clip.sd3_clip",
            ),
            (
                T::ClipG,
                ClipType::Hidream,
                "sim.text_encoders.hidream.hidream_clip",
            ),
            (
                T::T5Xxl,
                ClipType::Sd3,
                "sim.text_encoders.sd3_clip.sd3_clip",
            ),
            (T::T5Xxl, ClipType::Ltxv, "sim.text_encoders.lt.ltxv_te"),
            (
                T::T5Xxl,
                ClipType::Pixart,
                "sim.text_encoders.pixart_t5.pixart_te",
            ),
            (
                T::T5Xxl,
                ClipType::Chroma,
                "sim.text_encoders.pixart_t5.pixart_te",
            ),
            (T::T5Xxl, ClipType::Wan, "sim.text_encoders.wan.te"),
            (
                T::T5Xxl,
                ClipType::Hidream,
                "sim.text_encoders.hidream.hidream_clip",
            ),
            (
                T::T5Xxl,
                ClipType::Cogvideox,
                "sim.text_encoders.cogvideo.cogvideo_te",
            ),
            (T::T5Base, ClipType::Ace, "sim.text_encoders.ace.AceT5Model"),
            (
                T::Gemma2_2b,
                ClipType::Pixeldit,
                "sim.text_encoders.pixeldit.pixeldit_te",
            ),
            (
                T::Qwen25_7b,
                ClipType::HunyuanImage,
                "sim.text_encoders.hunyuan_image.te",
            ),
            (
                T::Qwen25_7b,
                ClipType::LongcatImage,
                "sim.text_encoders.longcat_image.te",
            ),
            (
                T::Qwen3_4b,
                ClipType::Flux,
                "sim.text_encoders.flux.klein_te",
            ),
            (
                T::Qwen3_4b,
                ClipType::Flux2,
                "sim.text_encoders.flux.klein_te",
            ),
            (
                T::Qwen3_8b,
                ClipType::Ideogram4,
                "sim.text_encoders.ideogram4.te",
            ),
            (
                T::Qwen3Vl8b,
                ClipType::Ideogram4,
                "sim.text_encoders.ideogram4.te_qwen3vl",
            ),
            (T::Qwen3Vl8b, ClipType::Boogu, "sim.text_encoders.boogu.te"),
            (T::Qwen3Vl4b, ClipType::Krea2, "sim.text_encoders.krea2.te"),
            (
                T::Qwen3Vl4b,
                ClipType::Flux,
                "sim.text_encoders.flux.klein_te",
            ),
            (
                T::Qwen3Vl8b,
                ClipType::Flux2,
                "sim.text_encoders.flux.klein_te",
            ),
        ];
        for (model, clip_type, expected) in conditional_rows {
            let selection = select_clip_architecture(clip_type, &[detector_probe(model)?])?;
            assert_eq!(selection.clip_model(), expected, "{model:?} {clip_type:?}");
        }
        let mut t5_base_with_spiece = detector_fixture(T::T5Base)
            .into_iter()
            .map(|(name, shape)| {
                (
                    name,
                    ModelParsedTensorFact {
                        shape,
                        storage_dtype: "F32".to_owned(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        t5_base_with_spiece.insert(
            "spiece_model".to_owned(),
            ModelParsedTensorFact {
                shape: vec![1],
                storage_dtype: "U8".to_owned(),
            },
        );
        let t5_base_with_spiece = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: t5_base_with_spiece,
            formats: Vec::new(),
        })?;
        assert_eq!(
            select_clip_architecture(
                ClipType::StableDiffusion,
                std::slice::from_ref(&t5_base_with_spiece),
            )?
            .clip_model(),
            "sim.text_encoders.ace.AceT5Model"
        );

        let clip_l = detector_probe(T::ClipL)?;
        let t5 = detector_probe(T::T5Xxl)?;
        let dual_rows = [
            (ClipType::Sd3, "sim.text_encoders.sd3_clip.sd3_clip"),
            (ClipType::HunyuanDit, "sim.text_encoders.hydit.HyditModel"),
            (ClipType::Flux, "sim.text_encoders.flux.flux_clip"),
            (
                ClipType::HunyuanVideo,
                "sim.text_encoders.hunyuan_video.hunyuan_video_clip",
            ),
            (ClipType::Hidream, "sim.text_encoders.hidream.hidream_clip"),
            (ClipType::HunyuanImage, "sim.text_encoders.hunyuan_image.te"),
            (
                ClipType::HunyuanVideo15,
                "sim.text_encoders.hunyuan_image.te",
            ),
            (ClipType::Kandinsky5, "sim.text_encoders.kandinsky5.te"),
            (ClipType::Kandinsky5Image, "sim.text_encoders.kandinsky5.te"),
            (ClipType::Ltxv, "sim.text_encoders.lt.ltxav_te"),
            (ClipType::Newbie, "sim.text_encoders.newbie.te"),
            (ClipType::Ace, "sim.text_encoders.ace15.te"),
            (ClipType::StableDiffusion, "sim.sdxl_clip.SDXLClipModel"),
        ];
        for (clip_type, expected) in dual_rows {
            let selection = select_clip_architecture(clip_type, &[clip_l.clone(), t5.clone()])?;
            assert_eq!(selection.clip_model(), expected, "{clip_type:?}");
        }
        Ok(())
    }

    fn linear_manifest() -> Result<ClipParameterManifest, ClipError> {
        ClipParameterManifest::checked(
            1,
            vec![
                ClipParameterSpec::checked(0, "projection.weight", vec![2, 3], "float32")?,
                ClipParameterSpec::checked(0, "projection.bias", vec![2], "float32")?,
            ],
            BTreeSet::new(),
            vec![ClipNativeModuleSpec::checked(
                "projection",
                0,
                "projection.weight",
                Some("projection.bias".to_owned()),
                ClipNativeModuleKind::Linear {
                    input_features: 3,
                    output_features: 2,
                },
            )?],
        )
    }

    fn domain_target(
        dynamic: bool,
    ) -> Result<ModelClipTargetDescriptor, Box<dyn std::error::Error>> {
        Ok(ModelClipTargetDescriptor::checked(
            vec![
                ModelClipTargetCandidateDescriptor::checked("sim.sd1.tokenizer", "sim.sd1.clip")?,
                ModelClipTargetCandidateDescriptor::checked("sim.sd2.tokenizer", "sim.sd2.clip")?,
            ],
            dynamic,
        )?)
    }

    fn test_architecture(
        tokenizer: &str,
        clip_model: &str,
        clip_type: ClipType,
        text_encoder_models: Vec<Option<TextEncoderModel>>,
    ) -> Result<ClipArchitectureSelection, ClipError> {
        let encoded = serde_json::to_vec(&(
            clip_type,
            &text_encoder_models,
            tokenizer,
            clip_model,
            Option::<ClipDetectedConfiguration>::None,
            Option::<ClipDetectedConfiguration>::None,
        ))
        .map_err(|error| ClipError::IdentitySerialization(error.to_string()))?;
        Ok(ClipArchitectureSelection {
            tokenizer: tokenizer.to_owned(),
            clip_model: clip_model.to_owned(),
            text_encoder_models,
            t5xxl_configuration: None,
            llama_configuration: None,
            digest: sha256(&encoded),
        })
    }

    fn checked_domain(
        family_identifier: &str,
        target: ModelClipTargetDescriptor,
        select_second_test_target: bool,
        artifact: char,
        patch: char,
    ) -> Result<ClipLoadDomain, Box<dyn std::error::Error>> {
        Ok(ClipLoadDomain::checked_facts(
            ModelFamilyIdentity::new("COMFY-MODEL-0999", family_identifier, "v1")?,
            target,
            ClipType::StableDiffusion,
            test_architecture(
                if select_second_test_target {
                    "sim.sd2.tokenizer"
                } else {
                    "sim.sd1.tokenizer"
                },
                if select_second_test_target {
                    "sim.sd2.clip"
                } else {
                    "sim.sd1.clip"
                },
                ClipType::StableDiffusion,
                vec![Some(TextEncoderModel::ClipL)],
            )?,
            vec![artifact.to_string().repeat(64)],
            PatchGraphIdentity {
                schema_version: crate::PATCH_GRAPH_SCHEMA_VERSION,
                base_artifact_digest: artifact.to_string().repeat(64),
                ordered_digest: patch.to_string().repeat(64),
            },
            linear_manifest()?.digest(),
            DType::F32,
            DeviceId::CPU,
        )?)
    }

    fn contract_domain(
        clip_type: ClipType,
        text_encoder_model: Option<TextEncoderModel>,
        dtype: DType,
        device: DeviceId,
        manifest_identity: &str,
    ) -> Result<ClipLoadDomain, Box<dyn std::error::Error>> {
        let discriminator = text_encoder_model
            .map(|model| model.source_ordinal().to_string())
            .unwrap_or_else(|| "none".to_owned());
        let tokenizer = format!("sim.test.tokenizer_{discriminator}");
        let clip_model = format!("sim.test.clip_{discriminator}");
        Ok(ClipLoadDomain::checked_facts(
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "family_a", "v1")?,
            ModelClipTargetDescriptor::checked(
                vec![ModelClipTargetCandidateDescriptor::checked(
                    tokenizer.clone(),
                    clip_model.clone(),
                )?],
                true,
            )?,
            clip_type,
            test_architecture(&tokenizer, &clip_model, clip_type, vec![text_encoder_model])?,
            vec!["a".repeat(64)],
            PatchGraphIdentity {
                schema_version: crate::PATCH_GRAPH_SCHEMA_VERSION,
                base_artifact_digest: "a".repeat(64),
                ordered_digest: "b".repeat(64),
            },
            manifest_identity,
            dtype,
            device,
        )?)
    }

    #[test]
    fn loader_domain_binds_family_target_artifact_patch_manifest_dtype_and_device()
    -> Result<(), Box<dyn std::error::Error>> {
        let first = checked_domain("family_a", domain_target(true)?, false, 'a', 'b')?;
        let repeated = checked_domain("family_a", domain_target(true)?, false, 'a', 'b')?;
        assert_eq!(first.digest(), repeated.digest());
        assert_eq!(first.manifest_identity(), linear_manifest()?.digest());
        assert_eq!(
            first.selected().clip_model().target().as_str(),
            "sim.sd1.clip"
        );

        let different_family = checked_domain("family_b", domain_target(true)?, false, 'a', 'b')?;
        let different_target = checked_domain("family_a", domain_target(true)?, true, 'a', 'b')?;
        let different_artifact = checked_domain("family_a", domain_target(true)?, false, 'c', 'b')?;
        let different_patch = checked_domain("family_a", domain_target(true)?, false, 'a', 'd')?;
        assert_ne!(first.digest(), different_family.digest());
        assert_ne!(first.digest(), different_target.digest());
        assert_ne!(first.digest(), different_artifact.digest());
        assert_ne!(first.digest(), different_patch.digest());

        let manifest_identity = linear_manifest()?.digest().to_owned();
        let clip_type_identities = ClipType::ALL
            .into_iter()
            .map(|clip_type| {
                contract_domain(
                    clip_type,
                    Some(TextEncoderModel::ClipL),
                    DType::F32,
                    DeviceId::CPU,
                    &manifest_identity,
                )
                .map(|domain| domain.digest().to_owned())
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        assert_eq!(clip_type_identities.len(), ClipType::ALL.len());
        let te_model_identities = TextEncoderModel::ALL
            .into_iter()
            .map(|model| {
                contract_domain(
                    ClipType::StableDiffusion,
                    Some(model),
                    DType::F32,
                    DeviceId::CPU,
                    &manifest_identity,
                )
                .map(|domain| domain.digest().to_owned())
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        assert_eq!(te_model_identities.len(), TextEncoderModel::ALL.len());
        let different_manifest = contract_domain(
            ClipType::StableDiffusion,
            Some(TextEncoderModel::ClipL),
            DType::F32,
            DeviceId::CPU,
            &"e".repeat(64),
        )?;
        let different_dtype = contract_domain(
            ClipType::StableDiffusion,
            Some(TextEncoderModel::ClipL),
            DType::F16,
            DeviceId::CPU,
            &manifest_identity,
        )?;
        let different_device = contract_domain(
            ClipType::StableDiffusion,
            Some(TextEncoderModel::ClipL),
            DType::F32,
            DeviceId::from_source_device("metal")?,
            &manifest_identity,
        )?;
        assert_ne!(first.digest(), different_manifest.digest());
        assert_ne!(first.digest(), different_dtype.digest());
        assert_ne!(first.digest(), different_device.digest());
        assert_eq!(artifact_bundle_identity(&["a".repeat(64)])?, "a".repeat(64));
        let bundle = artifact_bundle_identity(&["a".repeat(64), "b".repeat(64)])?;
        let reversed_bundle = artifact_bundle_identity(&["b".repeat(64), "a".repeat(64)])?;
        assert_eq!(bundle.len(), 64);
        assert_ne!(bundle, reversed_bundle);
        assert!(matches!(
            artifact_bundle_identity(&Vec::new()),
            Err(ClipError::InvalidArtifactSet(0))
        ));
        assert!(matches!(
            contract_domain(
                ClipType::StableDiffusion,
                Some(TextEncoderModel::ClipL),
                DType::I64,
                DeviceId::CPU,
                &manifest_identity,
            ),
            Err(error) if matches!(
                error.downcast_ref::<ClipError>(),
                Some(ClipError::InvalidExecutionDType(DType::I64))
            )
        ));

        let tokenizer = fixture_tokenizer()?;
        let plan = ClipExecutionPlan::checked_from_domain(
            &first,
            tokenizer.identity().clone(),
            ClipLayerSelection::Final,
            ClipPooling::None,
            SD1_CONTEXT_LENGTH,
            2,
        )?;
        let substituted = ClipExecutionPlan::checked_from_domain(
            &different_family,
            tokenizer.identity().clone(),
            ClipLayerSelection::Final,
            ClipPooling::None,
            SD1_CONTEXT_LENGTH,
            2,
        )?;
        assert_ne!(plan.model_identity(), substituted.model_identity());
        assert_ne!(plan.digest(), substituted.digest());

        let static_composite_second =
            checked_domain("family_a", domain_target(false)?, true, 'a', 'b')?;
        assert_eq!(
            static_composite_second
                .selected()
                .clip_model()
                .target()
                .as_str(),
            "sim.sd2.clip"
        );
        assert!(matches!(
            ClipLoadDomain::checked_facts(
                ModelFamilyIdentity::new("COMFY-MODEL-0999", "family_a", "v1")?,
                domain_target(true)?,
                ClipType::StableDiffusion,
                test_architecture(
                    "sim.sd1.tokenizer",
                    "sim.sd1.clip",
                    ClipType::StableDiffusion,
                    vec![Some(TextEncoderModel::ClipL)],
                )?,
                vec!["a".repeat(64)],
                PatchGraphIdentity {
                    schema_version: crate::PATCH_GRAPH_SCHEMA_VERSION,
                    base_artifact_digest: "c".repeat(64),
                    ordered_digest: "b".repeat(64),
                },
                linear_manifest()?.digest(),
                DType::F32,
                DeviceId::CPU,
            ),
            Err(ClipError::PatchBaseMismatch { .. })
        ));
        Ok(())
    }

    fn plan(
        tokenizer: &Sd1Tokenizer,
        layer: ClipLayerSelection,
        pooling: ClipPooling,
        dtype: DType,
        device: DeviceId,
    ) -> Result<ClipExecutionPlan, Box<dyn std::error::Error>> {
        Ok(ClipExecutionPlan::checked_legacy(
            target()?,
            tokenizer.identity().clone(),
            "sim.sd1.clip",
            binding('a')?,
            binding('b')?,
            binding('c')?,
            layer,
            pooling,
            dtype,
            device,
            SD1_CONTEXT_LENGTH,
            2,
        )?)
    }

    #[test]
    fn sd1_tokenizer_binds_artifacts_bpe_weights_padding_and_batch_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = fixture_tokenizer()?;
        let cancellation = CancellationToken::default();
        let sequence = tokenizer.encode("a test", &cancellation)?;
        assert_eq!(
            sequence
                .tokens()
                .iter()
                .take(4)
                .map(|token| token.token())
                .collect::<Vec<_>>(),
            [SD1_START_TOKEN, 320, 1_628, SD1_END_TOKEN]
        );
        assert_eq!(sequence.content_tokens(), 2);
        assert!(
            sequence.tokens()[sequence.first_end_index()..]
                .iter()
                .all(|token| token.token() == SD1_END_TOKEN)
        );

        let weighted = tokenizer.encode_weighted(
            &[
                WeightedText::checked("a", 0.5)?,
                WeightedText::checked("test", 1.75)?,
            ],
            &cancellation,
        )?;
        assert_eq!(weighted.tokens()[1].weight(), 0.5);
        assert_eq!(weighted.tokens()[2].weight(), 1.75);
        assert_eq!(weighted.tokens()[0].weight(), 1.0);
        assert_eq!(weighted.tokens()[weighted.first_end_index()].weight(), 1.0);

        let long = "a ".repeat(200);
        let truncated = tokenizer.encode(&long, &cancellation)?;
        assert_eq!(truncated.tokens().len(), SD1_CONTEXT_LENGTH);
        assert_eq!(truncated.content_tokens(), SD1_CONTEXT_LENGTH - 2);
        assert_eq!(
            truncated.tokens()[SD1_CONTEXT_LENGTH - 1].token(),
            SD1_END_TOKEN
        );

        let prompts = vec![
            vec![WeightedText::checked("a test", 1.0)?],
            vec![WeightedText::checked("", 1.0)?],
        ];
        let first = tokenizer.tokenize_batch(&prompts, &cancellation)?;
        let second = tokenizer.tokenize_batch(&prompts, &cancellation)?;
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.rows().len(), 2);
        assert_eq!(first.rows()[1].content_tokens(), 0);
        assert_ne!(
            tokenizer.identity().vocabulary_sha256(),
            tokenizer.identity().merges_sha256()
        );
        assert_eq!(tokenizer.identity().digest().len(), 64);
        assert!(tokenizer.resident_bytes()? > u64::try_from(std::mem::size_of::<Sd1Tokenizer>())?);

        let mask = first.causal_attention_mask()?;
        assert!(!mask[1]);
        assert!(mask[SD1_CONTEXT_LENGTH]);
        assert!(mask[SD1_CONTEXT_LENGTH + 1]);
        assert!(!mask[SD1_CONTEXT_LENGTH + 2]);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            tokenizer.tokenize_batch(&prompts, &cancelled),
            Err(ClipError::Tensor(TensorError::Cancelled))
        ));
        assert!(matches!(
            WeightedText::checked("invalid", f32::NAN),
            Err(ClipError::InvalidWeight(_))
        ));
        Ok(())
    }

    #[test]
    fn execution_plan_rejects_descriptor_dtype_and_identity_mismatches()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = fixture_tokenizer()?;
        let first = plan(
            &tokenizer,
            ClipLayerSelection::Final,
            ClipPooling::None,
            DType::F32,
            DeviceId::CPU,
        )?;
        let second = plan(
            &tokenizer,
            ClipLayerSelection::Final,
            ClipPooling::None,
            DType::F32,
            DeviceId::CPU,
        )?;
        assert_eq!(first.digest(), second.digest());
        assert_eq!(
            first.resident_owned_bytes()?,
            second.resident_owned_bytes()?
        );
        assert!(matches!(
            ClipExecutionPlan::checked_legacy(
                target()?,
                tokenizer.identity().clone(),
                "sim.sd1.wrong",
                binding('a')?,
                binding('b')?,
                binding('c')?,
                ClipLayerSelection::Final,
                ClipPooling::None,
                DType::F32,
                DeviceId::CPU,
                SD1_CONTEXT_LENGTH,
                2,
            ),
            Err(ClipError::DescriptorMismatch { .. })
        ));
        assert!(matches!(
            ClipExecutionPlan::checked_legacy(
                target()?,
                tokenizer.identity().clone(),
                "sim.sd1.clip",
                binding('a')?,
                binding('b')?,
                binding('c')?,
                ClipLayerSelection::Final,
                ClipPooling::None,
                DType::I64,
                DeviceId::CPU,
                SD1_CONTEXT_LENGTH,
                2,
            ),
            Err(ClipError::InvalidExecutionDType(DType::I64))
        ));
        assert!(matches!(
            validate_sha256("model", "ABC"),
            Err(ClipError::InvalidBindingIdentity { .. })
        ));
        Ok(())
    }

    fn tensor(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        shape: &[u64],
        values: Vec<f32>,
        cancellation: &CancellationToken,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let bytes = u64::try_from(values.len())?
            .checked_mul(4)
            .ok_or("tensor bytes")?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(bytes)?,
            rng_phase: None,
            cancellation,
        };
        Ok(tensor_from_f32(backend, shape, &values, &context)?)
    }

    fn mapped_clip_text_weights(digest: &str, weights: &ClipTextWeights) -> MappedModelWeights {
        let mut tensors = BTreeMap::from([
            (
                "token_embedding.weight".to_owned(),
                weights.token_embedding.clone(),
            ),
            (
                "position_embedding.weight".to_owned(),
                weights.position_embedding.clone(),
            ),
            (
                "final_layer_norm.weight".to_owned(),
                weights.final_layer_norm_weight.clone(),
            ),
            (
                "final_layer_norm.bias".to_owned(),
                weights.final_layer_norm_bias.clone(),
            ),
        ]);
        for (index, layer) in weights.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            for (name, tensor) in [
                ("layer_norm_1.weight", &layer.layer_norm_1_weight),
                ("layer_norm_1.bias", &layer.layer_norm_1_bias),
                ("query.weight", &layer.query_weight),
                ("query.bias", &layer.query_bias),
                ("key.weight", &layer.key_weight),
                ("key.bias", &layer.key_bias),
                ("value.weight", &layer.value_weight),
                ("value.bias", &layer.value_bias),
                ("output.weight", &layer.output_weight),
                ("output.bias", &layer.output_bias),
                ("layer_norm_2.weight", &layer.layer_norm_2_weight),
                ("layer_norm_2.bias", &layer.layer_norm_2_bias),
                ("feed_forward_1.weight", &layer.feed_forward_1_weight),
                ("feed_forward_1.bias", &layer.feed_forward_1_bias),
                ("feed_forward_2.weight", &layer.feed_forward_2_weight),
                ("feed_forward_2.bias", &layer.feed_forward_2_bias),
            ] {
                tensors.insert(format!("{prefix}.{name}"), tensor.clone());
            }
        }
        MappedModelWeights::from_parts(digest.to_owned(), tensors, Vec::new())
    }

    fn native_sd1_resource_fixture(
        schedule_enabled: bool,
        schedule: Vec<NativeClipScheduleSpec>,
    ) -> Result<
        (
            NativeClipResource,
            CpuBackend,
            CpuWorkspaceAuthority,
            CancellationToken,
        ),
        Box<dyn std::error::Error>,
    > {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let mut embeddings = vec![0.0; SD1_VOCABULARY_SIZE * 2];
        for (index, values) in embeddings.chunks_exact_mut(2).enumerate() {
            let value = (index % 97) as f32 / 97.0;
            values.copy_from_slice(&[value, 1.0 - value]);
        }
        let weights = ClipTextWeights {
            token_embedding: tensor(
                &backend,
                &authority,
                &[SD1_VOCABULARY_SIZE as u64, 2],
                embeddings,
                &cancellation,
            )?,
            position_embedding: tensor(
                &backend,
                &authority,
                &[SD1_CONTEXT_LENGTH as u64, 2],
                vec![0.0; SD1_CONTEXT_LENGTH * 2],
                &cancellation,
            )?,
            layers: vec![layer(&backend, &authority, &cancellation)?],
            final_layer_norm_weight: tensor(
                &backend,
                &authority,
                &[2],
                vec![1.0, 1.0],
                &cancellation,
            )?,
            final_layer_norm_bias: tensor(
                &backend,
                &authority,
                &[2],
                vec![0.0, 0.0],
                &cancellation,
            )?,
        };
        let configuration = ClipTextConfiguration {
            dtype: DType::F32,
            device: DeviceId::CPU,
            vocabulary_size: SD1_VOCABULARY_SIZE,
            max_position_embeddings: SD1_CONTEXT_LENGTH,
            hidden_size: 2,
            intermediate_size: 4,
            attention_heads: 1,
            layer_count: 1,
            eos_token_id: SD1_END_TOKEN,
            activation: ClipTextActivation::QuickGelu,
            projection_dimension: None,
        };
        let template = NativeClipText::new(configuration, weights.clone(), None)?;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let fixture = root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
        let tokenizer = NativePromptTokenizer::checked(
            NativeTokenizerFamily::ClipBpe(ClipBpeTokenizer::from_json_and_merges(
                ModelTokenizerDescriptor::checked("sim.sd1.tokenizer")?,
                &fs::read_to_string(fixture.join("vocab.json"))?,
                &fs::read_to_string(fixture.join("merges.txt"))?,
            )?),
            TokenizerConfiguration {
                maximum_length: SD1_CONTEXT_LENGTH,
                minimum_length: None,
                minimum_padding: None,
                pad_to_maximum_length: true,
                pad_left: false,
                start_token: Some(SD1_START_TOKEN),
                end_token: Some(SD1_END_TOKEN),
                pad_token: SD1_END_TOKEN,
                maximum_word_length: 8,
                disable_weights: false,
                embedding_width: Some(2),
            },
            BTreeMap::new(),
        )?;
        let digest = "a".repeat(64);
        let component = NativeClipComponent::checked_clip_text(
            NativeClipComponentRole::ClipL,
            Arc::new(mapped_clip_text_weights(&digest, &weights)),
            Arc::new(tokenizer),
            &template,
            NativeClipComponentOptions::checked_source(
                NativeTokenizerInvocationOptions::default(),
                NativeClipLayerSelection::Last,
                true,
                false,
                false,
                false,
            )?,
            &cancellation,
        )?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(32 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let resource = NativeClipResource::checked(
            NativeClipProfile::Sd1,
            schedule_enabled,
            vec![Arc::new(component)],
            schedule,
            &backend,
            &context,
        )?;
        Ok((resource, backend, authority, cancellation))
    }

    fn recipe_clip_source(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
        digest_byte: char,
    ) -> Result<NativeClipRecipeSource, Box<dyn std::error::Error>> {
        let mut ordered_state = vec![
            (
                "text_model.embeddings.token_embedding.weight".to_owned(),
                tensor(
                    backend,
                    authority,
                    &[SD1_VOCABULARY_SIZE as u64, 2],
                    vec![0.0; SD1_VOCABULARY_SIZE * 2],
                    cancellation,
                )?,
            ),
            (
                "text_model.embeddings.position_embedding.weight".to_owned(),
                tensor(
                    backend,
                    authority,
                    &[SD1_CONTEXT_LENGTH as u64, 2],
                    vec![0.0; SD1_CONTEXT_LENGTH * 2],
                    cancellation,
                )?,
            ),
            (
                "text_model.final_layer_norm.weight".to_owned(),
                tensor(backend, authority, &[2], vec![1.0; 2], cancellation)?,
            ),
            (
                "text_model.final_layer_norm.bias".to_owned(),
                tensor(backend, authority, &[2], vec![0.0; 2], cancellation)?,
            ),
            (
                "text_projection.weight".to_owned(),
                tensor(backend, authority, &[2, 2], vec![0.0; 4], cancellation)?,
            ),
        ];
        for layer_index in 0..2 {
            let layer = layer(backend, authority, cancellation)?;
            for (name, value) in [
                ("layer_norm1.weight", layer.layer_norm_1_weight),
                ("layer_norm1.bias", layer.layer_norm_1_bias),
                ("self_attn.q_proj.weight", layer.query_weight),
                ("self_attn.q_proj.bias", layer.query_bias),
                ("self_attn.k_proj.weight", layer.key_weight),
                ("self_attn.k_proj.bias", layer.key_bias),
                ("self_attn.v_proj.weight", layer.value_weight),
                ("self_attn.v_proj.bias", layer.value_bias),
                ("self_attn.out_proj.weight", layer.output_weight),
                ("self_attn.out_proj.bias", layer.output_bias),
                ("layer_norm2.weight", layer.layer_norm_2_weight),
                ("layer_norm2.bias", layer.layer_norm_2_bias),
                ("mlp.fc1.weight", layer.feed_forward_1_weight),
                ("mlp.fc1.bias", layer.feed_forward_1_bias),
                ("mlp.fc2.weight", layer.feed_forward_2_weight),
                ("mlp.fc2.bias", layer.feed_forward_2_bias),
            ] {
                ordered_state.push((
                    format!("text_model.encoder.layers.{layer_index}.{name}"),
                    value,
                ));
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let fixture = root.join("crates/sim_test_support/fixtures/models/sd15-tiny-v1");
        let vocabulary: serde_json::Value =
            serde_json::from_slice(&fs::read(fixture.join("vocab.json"))?)?;
        let merges = fs::read_to_string(fixture.join("merges.txt"))?
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with("#version"))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let tokenizer_json = serde_json::to_vec(&serde_json::json!({
            "model": { "vocab": vocabulary, "merges": merges }
        }))?;
        let tokenizer_sha256 = format!("{:x}", Sha256::digest(&tokenizer_json));
        let configuration_json = serde_json::to_vec(&serde_json::json!({
            "model_type": "clip",
            "hidden_size": 2,
            "projection_dim": if digest_byte == 'b' { 1_280 } else { 768 }
        }))?;
        let configuration_sha256 = format!("{:x}", Sha256::digest(&configuration_json));
        Ok(NativeClipRecipeSource {
            source_form: crate::model_family::NativeModelRecipeSourceForm::Standalone,
            artifact_sha256: digest_byte.to_string().repeat(64),
            ordered_state,
            parser_limits: ParserLimits::default(),
            auxiliary_artifacts: vec![
                NativeClipRecipeAuxiliaryArtifact {
                    format: NativeClipRecipeAuxiliaryFormat::JsonConfiguration,
                    target_ordinal: None,
                    artifact_sha256: configuration_sha256,
                    bytes: configuration_json,
                },
                NativeClipRecipeAuxiliaryArtifact {
                    format: NativeClipRecipeAuxiliaryFormat::JsonTokenizer,
                    target_ordinal: None,
                    artifact_sha256: tokenizer_sha256,
                    bytes: tokenizer_json,
                },
            ],
        })
    }

    fn recipe_open_clip_state(
        source: NativeClipRecipeSource,
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
    ) -> Result<Vec<(String, Tensor)>, Box<dyn std::error::Error>> {
        let mut projected = Vec::new();
        for (name, tensor) in source.ordered_state {
            let mapped = match name.as_str() {
                "text_model.embeddings.token_embedding.weight" => "token_embedding.weight",
                "text_model.embeddings.position_embedding.weight" => "positional_embedding",
                "text_model.final_layer_norm.weight" => "ln_final.weight",
                "text_model.final_layer_norm.bias" => "ln_final.bias",
                "text_projection.weight" => "text_projection",
                _ => {
                    let Some(rest) = name.strip_prefix("text_model.encoder.layers.") else {
                        return Err(
                            format!("unsupported canonical CLIP fixture tensor {name}").into()
                        );
                    };
                    let Some((layer, field)) = rest.split_once('.') else {
                        return Err(format!("invalid canonical CLIP fixture tensor {name}").into());
                    };
                    if field.starts_with("self_attn.q_proj.")
                        || field.starts_with("self_attn.k_proj.")
                        || field.starts_with("self_attn.v_proj.")
                    {
                        continue;
                    }
                    let field = match field {
                        "layer_norm1.weight" => "ln_1.weight",
                        "layer_norm1.bias" => "ln_1.bias",
                        "self_attn.out_proj.weight" => "attn.out_proj.weight",
                        "self_attn.out_proj.bias" => "attn.out_proj.bias",
                        "layer_norm2.weight" => "ln_2.weight",
                        "layer_norm2.bias" => "ln_2.bias",
                        "mlp.fc1.weight" => "mlp.c_fc.weight",
                        "mlp.fc1.bias" => "mlp.c_fc.bias",
                        "mlp.fc2.weight" => "mlp.c_proj.weight",
                        "mlp.fc2.bias" => "mlp.c_proj.bias",
                        _ => {
                            return Err(
                                format!("unsupported canonical CLIP layer tensor {name}").into()
                            );
                        }
                    };
                    projected.push((format!("transformer.resblocks.{layer}.{field}"), tensor));
                    continue;
                }
            };
            projected.push((mapped.to_owned(), tensor));
        }
        for layer in 0..2 {
            projected.push((
                format!("transformer.resblocks.{layer}.attn.in_proj_weight"),
                tensor(backend, authority, &[6, 2], vec![0.0; 12], cancellation)?,
            ));
            projected.push((
                format!("transformer.resblocks.{layer}.attn.in_proj_bias"),
                tensor(backend, authority, &[6], vec![0.0; 6], cancellation)?,
            ));
        }
        Ok(projected)
    }

    fn protobuf_varint_fixture(mut value: u64, output: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            output.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    fn sentence_piece_fixture() -> Vec<u8> {
        let mut model = Vec::new();
        for (piece, piece_type) in [("<unk>", 2_u64), ("</s>", 1), ("▁", 1)] {
            let mut record = Vec::new();
            record.push(0x0a);
            protobuf_varint_fixture(piece.len() as u64, &mut record);
            record.extend_from_slice(piece.as_bytes());
            record.push(0x15);
            record.extend_from_slice(&0.0_f32.to_bits().to_le_bytes());
            record.push(0x18);
            protobuf_varint_fixture(piece_type, &mut record);
            model.push(0x0a);
            protobuf_varint_fixture(record.len() as u64, &mut model);
            model.extend(record);
        }
        model
    }

    fn recipe_t5_source(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
    ) -> Result<NativeClipRecipeSource, Box<dyn std::error::Error>> {
        let mut state = Vec::new();
        for (name, shape, values) in [
            ("shared.weight", vec![3, 4], 12),
            ("encoder.block.0.layer.0.layer_norm.weight", vec![4], 4),
            (
                "encoder.block.0.layer.0.SelfAttention.q.weight",
                vec![4, 4],
                16,
            ),
            (
                "encoder.block.0.layer.0.SelfAttention.k.weight",
                vec![4, 4],
                16,
            ),
            (
                "encoder.block.0.layer.0.SelfAttention.v.weight",
                vec![4, 4],
                16,
            ),
            (
                "encoder.block.0.layer.0.SelfAttention.o.weight",
                vec![4, 4],
                16,
            ),
            (
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                vec![32, 1],
                32,
            ),
            ("encoder.block.0.layer.1.layer_norm.weight", vec![4], 4),
            (
                "encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
                vec![8, 4],
                32,
            ),
            (
                "encoder.block.0.layer.1.DenseReluDense.wi_1.weight",
                vec![8, 4],
                32,
            ),
            (
                "encoder.block.0.layer.1.DenseReluDense.wo.weight",
                vec![4, 8],
                32,
            ),
            ("encoder.final_layer_norm.weight", vec![4], 4),
        ] {
            state.push((
                name.to_owned(),
                tensor(backend, authority, &shape, vec![0.0; values], cancellation)?,
            ));
        }
        let sentence_piece = sentence_piece_fixture();
        let sentence_piece_sha256 = format!("{:x}", Sha256::digest(&sentence_piece));
        let configuration_json = serde_json::to_vec(&serde_json::json!({
            "vocab_size": 3,
            "d_model": 4,
            "num_heads": 1,
            "num_layers": 1,
            "d_ff": 8,
            "d_kv": 4,
            "layer_norm_epsilon": 1.0e-6,
            "feed_forward_proj": "gated-gelu",
            "relative_attention_max_distance": 128
        }))?;
        let configuration_sha256 = format!("{:x}", Sha256::digest(&configuration_json));
        Ok(NativeClipRecipeSource {
            source_form: crate::model_family::NativeModelRecipeSourceForm::Standalone,
            artifact_sha256: "c".repeat(64),
            ordered_state: state,
            parser_limits: ParserLimits::default(),
            auxiliary_artifacts: vec![
                NativeClipRecipeAuxiliaryArtifact {
                    format: NativeClipRecipeAuxiliaryFormat::JsonConfiguration,
                    target_ordinal: None,
                    artifact_sha256: configuration_sha256,
                    bytes: configuration_json,
                },
                NativeClipRecipeAuxiliaryArtifact {
                    format: NativeClipRecipeAuxiliaryFormat::SentencePiece,
                    target_ordinal: None,
                    artifact_sha256: sentence_piece_sha256,
                    bytes: sentence_piece,
                },
            ],
        })
    }

    fn recipe_llama_source(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
    ) -> Result<NativeClipRecipeSource, Box<dyn std::error::Error>> {
        let mut state = Vec::new();
        for (name, shape, values) in [
            ("model.embed_tokens.weight", vec![128_256, 4], 513_024),
            ("model.layers.0.input_layernorm.weight", vec![4], 4),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 16),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 4], 16),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 4], 16),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4], 16),
            ("model.layers.0.post_attention_layernorm.weight", vec![4], 4),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4], 32),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4], 32),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8], 32),
            ("model.norm.weight", vec![4], 4),
        ] {
            state.push((
                name.to_owned(),
                tensor(backend, authority, &shape, vec![0.0; values], cancellation)?,
            ));
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root is unavailable")?;
        let tokenizer_json = fs::read(
            root.join("projects/comfy/ComfyUI/sim/text_encoders/llama_tokenizer/tokenizer.json"),
        )?;
        let tokenizer_sha256 = format!("{:x}", Sha256::digest(&tokenizer_json));
        let configuration_json = serde_json::to_vec(&serde_json::json!({
            "vocab_size": 128_256,
            "max_position_embeddings": 2048,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 10_000.0
        }))?;
        let configuration_sha256 = format!("{:x}", Sha256::digest(&configuration_json));
        Ok(NativeClipRecipeSource {
            source_form: crate::model_family::NativeModelRecipeSourceForm::Standalone,
            artifact_sha256: "d".repeat(64),
            ordered_state: state,
            parser_limits: ParserLimits::default(),
            auxiliary_artifacts: vec![
                NativeClipRecipeAuxiliaryArtifact {
                    format: NativeClipRecipeAuxiliaryFormat::JsonConfiguration,
                    target_ordinal: None,
                    artifact_sha256: configuration_sha256,
                    bytes: configuration_json,
                },
                NativeClipRecipeAuxiliaryArtifact {
                    format: NativeClipRecipeAuxiliaryFormat::JsonTokenizer,
                    target_ordinal: None,
                    artifact_sha256: tokenizer_sha256,
                    bytes: tokenizer_json,
                },
            ],
        })
    }

    #[test]
    fn native_model_recipe_constructs_real_one_to_four_component_clip_owners()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(128 * 1024 * 1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(32 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let resources = vec![
            NativeClipResource::from_recipe_sources(
                vec![recipe_clip_source(
                    &backend,
                    &authority,
                    &cancellation,
                    'a',
                )?],
                &backend,
                &context,
            )?,
            NativeClipResource::from_recipe_sources(
                vec![
                    recipe_clip_source(&backend, &authority, &cancellation, 'a')?,
                    recipe_clip_source(&backend, &authority, &cancellation, 'b')?,
                ],
                &backend,
                &context,
            )?,
            NativeClipResource::from_recipe_sources(
                vec![
                    recipe_clip_source(&backend, &authority, &cancellation, 'a')?,
                    recipe_clip_source(&backend, &authority, &cancellation, 'b')?,
                    recipe_t5_source(&backend, &authority, &cancellation)?,
                ],
                &backend,
                &context,
            )?,
            NativeClipResource::from_recipe_sources(
                vec![
                    recipe_clip_source(&backend, &authority, &cancellation, 'a')?,
                    recipe_clip_source(&backend, &authority, &cancellation, 'b')?,
                    recipe_t5_source(&backend, &authority, &cancellation)?,
                    recipe_llama_source(&backend, &authority, &cancellation)?,
                ],
                &backend,
                &context,
            )?,
        ];
        assert_eq!(
            resources
                .iter()
                .map(NativeClipResource::profile)
                .collect::<Vec<_>>(),
            vec![
                NativeClipProfile::Sd1,
                NativeClipProfile::Sdxl,
                NativeClipProfile::Sd3,
                NativeClipProfile::HiDream,
            ]
        );
        assert_eq!(
            resources
                .iter()
                .map(|resource| {
                    resource
                        .components()
                        .iter()
                        .map(|component| component.role())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            vec![
                vec![NativeClipComponentRole::ClipL],
                vec![
                    NativeClipComponentRole::ClipL,
                    NativeClipComponentRole::ClipG,
                ],
                vec![
                    NativeClipComponentRole::ClipL,
                    NativeClipComponentRole::ClipG,
                    NativeClipComponentRole::T5,
                ],
                vec![
                    NativeClipComponentRole::ClipL,
                    NativeClipComponentRole::ClipG,
                    NativeClipComponentRole::T5,
                    NativeClipComponentRole::Llama,
                ],
            ]
        );
        assert_eq!(
            resources
                .iter()
                .map(|resource| resource.semantic_digest_sha256())
                .collect::<BTreeSet<_>>()
                .len(),
            4
        );
        for (expected, resource) in (1..=4).zip(&resources) {
            assert_eq!(resource.components().len(), expected);
            assert!(!resource.semantic_digest_sha256().is_empty());
            assert!(!resource.resident_tensor_allocations()?.is_empty());
        }
        for (source, expected_profile, expected_role) in [
            (
                recipe_clip_source(&backend, &authority, &cancellation, 'b')?,
                NativeClipProfile::Sdxl,
                NativeClipComponentRole::ClipG,
            ),
            (
                recipe_t5_source(&backend, &authority, &cancellation)?,
                NativeClipProfile::PixArt,
                NativeClipComponentRole::T5,
            ),
            (
                recipe_llama_source(&backend, &authority, &cancellation)?,
                NativeClipProfile::HiDream,
                NativeClipComponentRole::Llama,
            ),
        ] {
            let resource =
                NativeClipResource::from_recipe_sources(vec![source], &backend, &context)?;
            assert_eq!(resource.profile(), expected_profile);
            assert_eq!(
                resource
                    .components()
                    .first()
                    .ok_or("single-source CLIP resource has no component")?
                    .role(),
                expected_role
            );
            assert!(!resource.resident_tensor_allocations()?.is_empty());
        }
        let reversed = NativeClipResource::from_recipe_sources(
            vec![
                recipe_clip_source(&backend, &authority, &cancellation, 'b')?,
                recipe_clip_source(&backend, &authority, &cancellation, 'a')?,
            ],
            &backend,
            &context,
        );
        assert!(matches!(reversed, Err(ClipError::InvalidNativeResource(_))));

        let mut checkpoint_clip_l = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        let mut checkpoint_clip_g = recipe_clip_source(&backend, &authority, &cancellation, 'b')?;
        checkpoint_clip_l.source_form =
            crate::model_family::NativeModelRecipeSourceForm::Checkpoint;
        for artifact in &mut checkpoint_clip_l.auxiliary_artifacts {
            artifact.target_ordinal = Some(0);
        }
        for artifact in &mut checkpoint_clip_g.auxiliary_artifacts {
            artifact.target_ordinal = Some(1);
        }
        checkpoint_clip_l
            .auxiliary_artifacts
            .append(&mut checkpoint_clip_g.auxiliary_artifacts);
        let checkpoint_clip_g_state = recipe_open_clip_state(
            checkpoint_clip_g.clone(),
            &backend,
            &authority,
            &cancellation,
        )?;
        checkpoint_clip_l.ordered_state =
            checkpoint_clip_l
                .ordered_state
                .into_iter()
                .map(|(name, tensor)| {
                    (
                        format!("conditioner.embedders.0.transformer.{name}"),
                        tensor,
                    )
                })
                .chain(checkpoint_clip_g_state.into_iter().map(|(name, tensor)| {
                    (format!("conditioner.embedders.1.model.{name}"), tensor)
                }))
                .collect();
        let checkpoint_resource =
            NativeClipResource::from_recipe_sources(vec![checkpoint_clip_l], &backend, &context)?;
        assert_eq!(checkpoint_resource.profile(), NativeClipProfile::Sdxl);
        assert_eq!(
            checkpoint_resource
                .components()
                .iter()
                .map(|component| component.role())
                .collect::<Vec<_>>(),
            vec![
                NativeClipComponentRole::ClipL,
                NativeClipComponentRole::ClipG,
            ]
        );

        let mut ambiguous_checkpoint =
            recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        ambiguous_checkpoint.source_form =
            crate::model_family::NativeModelRecipeSourceForm::Checkpoint;
        for (name, _) in &mut ambiguous_checkpoint.ordered_state {
            *name = format!("conditioner.embedders.0.transformer.{name}");
        }
        let sibling = ambiguous_checkpoint
            .ordered_state
            .first()
            .ok_or("CLIP checkpoint fixture has no state")?
            .1
            .clone();
        ambiguous_checkpoint.ordered_state.push((
            "conditioner.embedders.1.transformer.text_model.embeddings.token_embedding.weight"
                .to_owned(),
            sibling,
        ));
        assert!(matches!(
            NativeClipResource::from_recipe_sources(vec![ambiguous_checkpoint], &backend, &context,),
            Err(ClipError::InvalidNativeResource(_))
        ));

        let mut bounded = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        let index_json = serde_json::to_vec(&serde_json::json!({
            "metadata": { "total_size": 4 },
            "weight_map": { "weight": "model-00001-of-00001.safetensors" }
        }))?;
        bounded
            .auxiliary_artifacts
            .push(NativeClipRecipeAuxiliaryArtifact {
                format: NativeClipRecipeAuxiliaryFormat::JsonConfiguration,
                target_ordinal: None,
                artifact_sha256: format!("{:x}", Sha256::digest(&index_json)),
                bytes: index_json,
            });
        let base = NativeClipResource::from_recipe_sources(vec![bounded], &backend, &context)?;
        let mut changed_limits = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        changed_limits.parser_limits.maximum_depth -= 1;
        let changed_limits =
            NativeClipResource::from_recipe_sources(vec![changed_limits], &backend, &context)?;
        assert_ne!(
            base.semantic_digest_sha256(),
            changed_limits.semantic_digest_sha256()
        );
        let mut changed_config = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        let configuration = changed_config
            .auxiliary_artifacts
            .iter_mut()
            .find(|artifact| artifact.format == NativeClipRecipeAuxiliaryFormat::JsonConfiguration)
            .ok_or("CLIP fixture configuration is missing")?;
        let mut document: serde_json::Value = serde_json::from_slice(&configuration.bytes)?;
        document
            .as_object_mut()
            .ok_or("CLIP fixture configuration is not an object")?
            .insert("identity_nonce".to_owned(), serde_json::json!(1));
        configuration.bytes = serde_json::to_vec(&document)?;
        configuration.artifact_sha256 = format!("{:x}", Sha256::digest(&configuration.bytes));
        let changed_config =
            NativeClipResource::from_recipe_sources(vec![changed_config], &backend, &context)?;
        assert_ne!(
            base.semantic_digest_sha256(),
            changed_config.semantic_digest_sha256()
        );
        let mut bounded_config = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        bounded_config.parser_limits.maximum_metadata_values = 1;
        assert!(matches!(
            NativeClipResource::from_recipe_sources(vec![bounded_config], &backend, &context),
            Err(ClipError::InvalidNativeResource(_))
        ));
        let mut mismatched_config = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        mismatched_config
            .auxiliary_artifacts
            .iter_mut()
            .find(|artifact| artifact.format == NativeClipRecipeAuxiliaryFormat::JsonConfiguration)
            .ok_or("CLIP fixture configuration is missing")?
            .artifact_sha256 = "f".repeat(64);
        assert!(matches!(
            NativeClipResource::from_recipe_sources(vec![mismatched_config], &backend, &context),
            Err(ClipError::InvalidNativeResource(_))
        ));
        let mut duplicate_tokenizer = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        let tokenizer = duplicate_tokenizer
            .auxiliary_artifacts
            .iter_mut()
            .find(|artifact| artifact.format == NativeClipRecipeAuxiliaryFormat::JsonTokenizer)
            .ok_or("CLIP fixture tokenizer is missing")?;
        tokenizer.bytes = br#"{"model":{"vocab":{"a":0,"a":1},"merges":[]}}"#.to_vec();
        tokenizer.artifact_sha256 = format!("{:x}", Sha256::digest(&tokenizer.bytes));
        assert!(matches!(
            NativeClipResource::from_recipe_sources(vec![duplicate_tokenizer], &backend, &context),
            Err(ClipError::InvalidNativeResource(_))
        ));
        let mut shallow_tokenizer = recipe_clip_source(&backend, &authority, &cancellation, 'a')?;
        shallow_tokenizer.parser_limits.maximum_depth = 1;
        assert!(matches!(
            NativeClipResource::from_recipe_sources(vec![shallow_tokenizer], &backend, &context),
            Err(ClipError::InvalidNativeResource(_))
        ));
        Ok(())
    }

    #[test]
    fn source_layer_selection_matches_original_sdclip_integer_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        for index in [i64::MIN, -4, 4, i64::MAX] {
            assert_eq!(
                source_clip_layer_selection(&NativeClipLayerSelection::Last, 3, index)?,
                NativeClipLayerSelection::Last,
            );
        }
        for index in [-3, -2, -1, 0, 1, 2, 3] {
            assert_eq!(
                source_clip_layer_selection(&NativeClipLayerSelection::Hidden(-2), 3, index)?,
                NativeClipLayerSelection::Hidden(isize::try_from(index)?),
            );
        }
        for index in [i64::MIN, -3, -1, 0, 3, i64::MAX] {
            assert_eq!(
                source_clip_layer_selection(&NativeClipLayerSelection::All, 3, index)?,
                NativeClipLayerSelection::All,
            );
        }
        Ok(())
    }

    #[test]
    fn source_layer_clone_preserves_aliases_options_hooks_and_original_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let patch = PatchGraph::checked(
            "a".repeat(64),
            vec![PatchOperation {
                identifier: "source-layer-retained-patch".to_owned(),
                kind: PatchKind::DenseDiff,
                scale: 1.0,
                targets: vec![PatchTarget {
                    key: "final_layer_norm.bias".to_owned(),
                    expected_shape: vec![2],
                    values: vec![0.25, -0.25],
                    application: PatchApplication::Add,
                }],
            }],
        )?;
        let windows = vec![
            NativeClipScheduleSpec::checked(0.0, 0.5, br#"{"hook":"base"}"#.to_vec(), vec![None])?,
            NativeClipScheduleSpec::checked(
                0.5,
                1.0,
                br#"{"hook":"patched"}"#.to_vec(),
                vec![Some(patch)],
            )?,
        ];
        let (original, backend, authority, cancellation) =
            native_sd1_resource_fixture(true, windows)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(32 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let before = original.execute(&backend, "a test", &context)?;
        let original_digest = original.semantic_digest_sha256().to_owned();
        let cloned = original.clone_with_source_layer(-1, &backend, &context)?;
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(original.source_layer_override(), None);
        assert_eq!(original.semantic_digest_sha256(), original_digest);
        assert_eq!(cloned.source_layer_override(), Some(-1));
        assert_ne!(cloned.semantic_digest_sha256(), original_digest);
        assert_eq!(
            original.resident_tensor_allocations()?,
            cloned.resident_tensor_allocations()?
        );
        for (source, target) in original
            .components
            .iter()
            .chain(
                original
                    .schedule
                    .iter()
                    .flat_map(|window| &window.components),
            )
            .zip(
                cloned
                    .components
                    .iter()
                    .chain(cloned.schedule.iter().flat_map(|window| &window.components)),
            )
        {
            assert!(Arc::ptr_eq(source, target));
            assert!(Arc::ptr_eq(&source.base_weights, &target.base_weights));
            assert!(Arc::ptr_eq(&source.patch_graph, &target.patch_graph));
            assert!(Arc::ptr_eq(&source.tokenizer, &target.tokenizer));
            assert_eq!(source.options, target.options);
            assert_eq!(source.recipe_binding_sha256, target.recipe_binding_sha256);
            match (&source.encoder, &target.encoder) {
                (NativeClipEncoder::ClipText(source), NativeClipEncoder::ClipText(target)) => {
                    assert!(Arc::ptr_eq(source, target));
                }
                _ => return Err("Fixture did not retain the CLIP text owner".into()),
            }
        }
        assert!(Arc::ptr_eq(
            cloned.components.first().ok_or("Missing base component")?,
            cloned
                .schedule
                .first()
                .ok_or("Missing base window")?
                .components
                .first()
                .ok_or("Missing scheduled component")?,
        ));
        for (source, target) in original.schedule.iter().zip(&cloned.schedule) {
            assert_eq!(source.start_percent_bits, target.start_percent_bits);
            assert_eq!(source.end_percent_bits, target.end_percent_bits);
            assert_eq!(source.hook_metadata_json, target.hook_metadata_json);
            assert_eq!(source.patch_identities, target.patch_identities);
        }
        let after = original.execute(&backend, "a test", &context)?;
        assert_eq!(before.len(), after.len());
        for (before, after) in before.iter().zip(&after) {
            let before = tensor_to_f32(&backend, before.conditioning(), &context)?;
            let after = tensor_to_f32(&backend, after.conditioning(), &context)?;
            assert_eq!(&*before, &*after);
        }
        cloned.execute(&backend, "a test", &context)?;
        Ok(())
    }

    #[test]
    fn source_layer_restart_and_patch_rebind_retain_override_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let (original, backend, authority, cancellation) =
            native_sd1_resource_fixture(false, Vec::new())?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(32 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let cloned = original.clone_with_source_layer(-1, &backend, &context)?;
        let restarted = cloned.restart(&backend, &context)?;
        assert_eq!(restarted.source_layer_override(), Some(-1));
        assert_eq!(
            restarted.semantic_digest_sha256(),
            cloned.semantic_digest_sha256()
        );
        let graphs: Vec<PatchGraph> = cloned
            .components
            .iter()
            .map(|component| component.patch_graph.as_ref().clone())
            .collect();
        let expected = original
            .rebind_component_patch_graphs(graphs.clone(), &backend, &context)?
            .clone_with_source_layer(-1, &backend, &context)?;
        let rebound = cloned.rebind_component_patch_graphs(graphs, &backend, &context)?;
        assert_eq!(rebound.source_layer_override(), Some(-1));
        assert_eq!(
            rebound.semantic_digest_sha256(),
            expected.semantic_digest_sha256()
        );
        for (source, target) in cloned.components.iter().zip(&rebound.components) {
            assert_eq!(source.patch_graph.identity(), target.patch_graph.identity());
            assert_eq!(source.options, target.options);
            assert!(Arc::ptr_eq(&source.tokenizer, &target.tokenizer));
            assert_eq!(
                source.base_weights.tensors().len(),
                target.base_weights.tensors().len()
            );
            for (name, source) in source.base_weights.tensors() {
                let target = target
                    .base_weights
                    .tensors()
                    .get(name)
                    .ok_or("Rebound tensor missing")?;
                assert_eq!(source.contiguous_bytes()?, target.contiguous_bytes()?);
            }
        }
        let before = cloned.execute(&backend, "a test", &context)?;
        let after = rebound.execute(&backend, "a test", &context)?;
        assert_eq!(before.len(), after.len());
        for (before, after) in before.iter().zip(&after) {
            assert_eq!(
                before.conditioning().contiguous_bytes()?,
                after.conditioning().contiguous_bytes()?
            );
        }
        let again = cloned.clone_with_source_layer(-1, &backend, &context)?;
        assert_eq!(
            again.semantic_digest_sha256(),
            cloned.semantic_digest_sha256()
        );
        let overwritten = cloned.clone_with_source_layer(i64::MIN, &backend, &context)?;
        assert_eq!(overwritten.source_layer_override(), Some(i64::MIN));
        assert_eq!(cloned.source_layer_override(), Some(-1));
        overwritten.execute(&backend, "a test", &context)?;
        Ok(())
    }

    #[test]
    fn source_layer_cancel_and_actual_metadata_budget_failure_release_workspace()
    -> Result<(), Box<dyn std::error::Error>> {
        let (original, backend, authority, cancellation) =
            native_sd1_resource_fixture(false, Vec::new())?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            original.clone_with_source_layer(-1, &backend, &context),
            Err(ClipError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            )),
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(original.source_layer_override(), None);
        cancellation.cancel();
        assert!(matches!(
            original.clone_with_source_layer(i64::MIN, &backend, &context),
            Err(ClipError::Tensor(TensorError::Cancelled)),
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn source_layer_actual_backend_budget_oom_is_atomic_and_recovers()
    -> Result<(), Box<dyn std::error::Error>> {
        let (original, backend, authority, cancellation) =
            native_sd1_resource_fixture(false, Vec::new())?;
        let before = backend.memory_snapshot();
        let available = before
            .limit_bytes
            .checked_sub(before.current_bytes)
            .ok_or("Canonical backend budget underflow")?;
        let blocker_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(before.limit_bytes)?,
            &cancellation,
        );
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(before.limit_bytes)?,
            &cancellation,
        );
        let blocker = backend.reserve_workspace(&blocker_context, available)?;
        assert_eq!(backend.memory_snapshot().current_bytes, before.limit_bytes);
        assert!(matches!(
            original.clone_with_source_layer(-1, &backend, &context),
            Err(ClipError::Tensor(TensorError::AllocationFailed { .. })),
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, before.limit_bytes);
        assert_eq!(original.source_layer_override(), None);
        drop(blocker);
        assert_eq!(blocker_context.scratch.in_use_bytes(), 0);
        assert_eq!(
            backend.memory_snapshot().current_bytes,
            before.current_bytes
        );
        let cloned = original.clone_with_source_layer(-1, &backend, &context)?;
        assert_eq!(cloned.source_layer_override(), Some(-1));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(
            backend.memory_snapshot().current_bytes,
            before.current_bytes
        );
        Ok(())
    }

    #[test]
    fn source_layer_empty_optional_owner_retains_wrapper_identity_and_reconstruction()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let original = NativeClipResource::checked(
            NativeClipProfile::Sd3,
            false,
            Vec::new(),
            Vec::new(),
            &backend,
            &context,
        )?;
        let digest = original.semantic_digest_sha256().to_owned();
        let cloned = original.clone_with_source_layer(-1, &backend, &context)?;
        assert_eq!(cloned.source_layer_override(), Some(-1));
        assert_ne!(cloned.semantic_digest_sha256(), digest);
        assert_eq!(original.semantic_digest_sha256(), digest);
        assert_eq!(original.source_layer_override(), None);
        let restarted = cloned.restart(&backend, &context)?;
        assert_eq!(restarted.source_layer_override(), Some(-1));
        assert_eq!(
            restarted.semantic_digest_sha256(),
            cloned.semantic_digest_sha256()
        );
        let rebound = cloned.rebind_component_patch_graphs(Vec::new(), &backend, &context)?;
        assert_eq!(rebound.source_layer_override(), Some(-1));
        assert_eq!(
            rebound.semantic_digest_sha256(),
            cloned.semantic_digest_sha256()
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn native_sd1_schedule_applies_patch_before_each_owner_encode()
    -> Result<(), Box<dyn std::error::Error>> {
        let digest = "a".repeat(64);
        let patch = PatchGraph::checked(
            digest,
            vec![PatchOperation {
                identifier: "scheduled-final-norm".to_owned(),
                kind: PatchKind::DenseDiff,
                scale: 1.0,
                targets: vec![PatchTarget {
                    key: "final_layer_norm.bias".to_owned(),
                    expected_shape: vec![2],
                    values: vec![0.25, -0.25],
                    application: PatchApplication::Add,
                }],
            }],
        )?;
        let window = NativeClipScheduleSpec::checked(
            0.0,
            1.0,
            br#"{"hook":"fixture"}"#.to_vec(),
            vec![Some(patch)],
        )?;
        let (scheduled, backend, authority, cancellation) =
            native_sd1_resource_fixture(true, vec![window])?;
        let (base, _, _, _) = native_sd1_resource_fixture(false, Vec::new())?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(32 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let scheduled_outputs = scheduled.execute(&backend, "a test", &context)?;
        let base_outputs = base.execute(&backend, "a test", &context)?;
        assert_eq!(scheduled_outputs.len(), 1);
        assert_eq!(scheduled_outputs[0].clip_start_percent(), 0.0);
        assert_eq!(scheduled_outputs[0].clip_end_percent(), 1.0);
        assert_eq!(
            scheduled_outputs[0].hook_metadata_json(),
            br#"{"hook":"fixture"}"#
        );
        let scheduled_values =
            tensor_to_f32(&backend, scheduled_outputs[0].conditioning(), &context)?;
        let base_values = tensor_to_f32(&backend, base_outputs[0].conditioning(), &context)?;
        assert!(
            scheduled_values
                .iter()
                .zip(base_values.iter())
                .any(|(scheduled, base)| scheduled.to_bits() != base.to_bits())
        );
        scheduled.validate(&cancellation)?;
        assert!(scheduled.resident_bytes()? > base.resident_bytes()?);
        let restarted = scheduled.restart(&backend, &context)?;
        assert_eq!(
            restarted.semantic_digest_sha256(),
            scheduled.semantic_digest_sha256()
        );
        assert!(!Arc::ptr_eq(
            &restarted.components()[0],
            &scheduled.components()[0]
        ));
        let restarted_outputs = restarted.execute(&backend, "a test", &context)?;
        assert_eq!(restarted_outputs.len(), scheduled_outputs.len());
        assert_eq!(
            &*tensor_to_f32(&backend, restarted_outputs[0].conditioning(), &context)?,
            &*scheduled_values
        );
        let owned_allocations = scheduled.resident_owned_allocations()?;
        let unique_owned_allocations = owned_allocations
            .iter()
            .map(|allocation| (allocation.kind(), allocation.address()))
            .collect::<BTreeSet<_>>();
        assert_eq!(unique_owned_allocations.len(), owned_allocations.len());
        let resident_kinds = owned_allocations
            .iter()
            .map(|allocation| allocation.kind())
            .collect::<BTreeSet<_>>();
        assert_eq!(resident_kinds.len(), 5);
        assert!(
            owned_allocations
                .iter()
                .filter(|allocation| {
                    allocation.kind() == NativeClipResidentOwnerKind::MappedWeights
                })
                .count()
                > 2
        );
        let payload = NativeModelPayload::native_clip(Arc::new(scheduled))?;
        let payload_parts = payload.resident_parts()?;
        assert_eq!(payload_parts.resident_bytes()?, payload.resident_bytes());
        let backing_kinds = payload_parts
            .backing_allocations()
            .iter()
            .map(|allocation| allocation.kind())
            .collect::<BTreeSet<_>>();
        assert!(backing_kinds.contains(&NativeModelBackingKind::NativeClipResource));
        assert!(backing_kinds.contains(&NativeModelBackingKind::NativeClipComponent));
        assert!(backing_kinds.contains(&NativeModelBackingKind::NativeClipTokenizer));
        assert!(backing_kinds.contains(&NativeModelBackingKind::NativeClipEncoder));
        assert!(backing_kinds.contains(&NativeModelBackingKind::NativeClipMappedWeights));
        Ok(())
    }

    #[test]
    fn native_sd3_without_encoders_admits_source_zero_conditioning()
    -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024 * 1024)?,
            &cancellation,
        );
        let resource = NativeClipResource::checked(
            NativeClipProfile::Sd3,
            true,
            Vec::new(),
            Vec::new(),
            &backend,
            &context,
        )?;
        resource.admit_execution_target(&backend, &context)?;
        let outputs = resource.execute(&backend, "all source encoders absent", &context)?;
        let [output] = outputs.as_slice() else {
            return Err("SD3 source fallback must emit one output".into());
        };
        assert_eq!(output.conditioning().descriptor().shape(), &[1, 77, 4096]);
        let pooled = output.pooled().ok_or("SD3 source pooled fallback absent")?;
        assert_eq!(pooled.descriptor().shape(), &[1, 2048]);
        for tensor in [output.conditioning(), pooled] {
            assert!(tensor.contiguous_bytes()?.iter().all(|value| *value == 0));
        }
        cancellation.cancel();
        assert!(matches!(
            resource.admit_execution_target(&backend, &context),
            Err(ClipError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }

    #[test]
    fn native_clip_profile_contract_rejects_source_option_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let (resource, backend, authority, cancellation) =
            native_sd1_resource_fixture(false, Vec::new())?;
        let mut component = resource.components()[0].as_ref().clone();
        component.options.final_layer_norm_intermediate = false;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(32 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            NativeClipResource::checked(
                NativeClipProfile::Sd1,
                false,
                vec![Arc::new(component)],
                Vec::new(),
                &backend,
                &context,
            ),
            Err(ClipError::InvalidNativeResource(_))
        ));
        Ok(())
    }

    #[test]
    fn native_clip_source_contract_rejects_tokenizer_and_pooling_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let clip_options = NativeClipComponentOptions::checked_source(
            NativeTokenizerInvocationOptions::default(),
            NativeClipLayerSelection::Last,
            true,
            true,
            false,
            false,
        )?;
        let mut clip_tokenizer = crate::clip_tokenizer::TokenizerConfiguration {
            maximum_length: 77,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: true,
            pad_left: false,
            start_token: Some(49_406),
            end_token: Some(49_407),
            pad_token: 49_407,
            maximum_word_length: 8,
            disable_weights: false,
            embedding_width: Some(768),
        };
        assert!(native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::ClipL,
            768,
            &clip_options,
            &clip_tokenizer,
        ));
        let mut unprojected = clip_options.clone();
        unprojected.projected_pooled = false;
        assert!(!native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::ClipL,
            768,
            &unprojected,
            &clip_tokenizer,
        ));
        clip_tokenizer.pad_token = 0;
        assert!(!native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::ClipL,
            768,
            &clip_options,
            &clip_tokenizer,
        ));
        clip_tokenizer.pad_token = 49_407;
        clip_tokenizer.disable_weights = true;
        assert!(!native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::ClipL,
            768,
            &clip_options,
            &clip_tokenizer,
        ));

        let t5_options = NativeClipComponentOptions::checked_source(
            NativeTokenizerInvocationOptions::default(),
            NativeClipLayerSelection::Last,
            false,
            false,
            false,
            false,
        )?;
        let mut t5_tokenizer = crate::clip_tokenizer::TokenizerConfiguration {
            maximum_length: SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH,
            minimum_length: Some(77),
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: false,
            start_token: None,
            end_token: Some(1),
            pad_token: 0,
            maximum_word_length: 8,
            disable_weights: false,
            embedding_width: Some(4_096),
        };
        assert!(native_clip_source_contract(
            NativeClipProfile::Sd3,
            NativeClipComponentRole::T5,
            4_096,
            &t5_options,
            &t5_tokenizer,
        ));
        t5_tokenizer.maximum_length = 77;
        assert!(!native_clip_source_contract(
            NativeClipProfile::Sd3,
            NativeClipComponentRole::T5,
            4_096,
            &t5_options,
            &t5_tokenizer,
        ));
        t5_tokenizer.maximum_length = SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH;
        t5_tokenizer.minimum_padding = Some(1);
        assert!(!native_clip_source_contract(
            NativeClipProfile::Sd3,
            NativeClipComponentRole::T5,
            4_096,
            &t5_options,
            &t5_tokenizer,
        ));
        t5_tokenizer.minimum_padding = None;
        t5_tokenizer.disable_weights = true;
        assert!(!native_clip_source_contract(
            NativeClipProfile::Sd3,
            NativeClipComponentRole::T5,
            4_096,
            &t5_options,
            &t5_tokenizer,
        ));
        t5_tokenizer.disable_weights = false;
        t5_tokenizer.minimum_length = Some(1);
        assert!(native_clip_source_contract(
            NativeClipProfile::PixArt,
            NativeClipComponentRole::T5,
            4_096,
            &t5_options,
            &t5_tokenizer,
        ));
        t5_tokenizer.maximum_length = 1;
        assert!(!native_clip_source_contract(
            NativeClipProfile::PixArt,
            NativeClipComponentRole::T5,
            4_096,
            &t5_options,
            &t5_tokenizer,
        ));

        let decoder_options = NativeClipComponentOptions::checked_source(
            NativeTokenizerInvocationOptions::default(),
            NativeClipLayerSelection::Hidden(-2),
            false,
            false,
            true,
            false,
        )?;
        let mut decoder_tokenizer = crate::clip_tokenizer::TokenizerConfiguration {
            maximum_length: SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH,
            minimum_length: Some(1),
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: false,
            start_token: Some(2),
            end_token: None,
            pad_token: 0,
            maximum_word_length: 8,
            disable_weights: false,
            embedding_width: Some(2_304),
        };
        assert!(native_clip_source_contract(
            NativeClipProfile::Lumina,
            NativeClipComponentRole::Gemma,
            2_304,
            &decoder_options,
            &decoder_tokenizer,
        ));
        decoder_tokenizer.maximum_length = 77;
        assert!(!native_clip_source_contract(
            NativeClipProfile::Lumina,
            NativeClipComponentRole::Gemma,
            2_304,
            &decoder_options,
            &decoder_tokenizer,
        ));

        let llama_options = NativeClipComponentOptions::checked_source(
            NativeTokenizerInvocationOptions::default(),
            NativeClipLayerSelection::All,
            false,
            false,
            true,
            false,
        )?;
        decoder_tokenizer.maximum_length = SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH;
        decoder_tokenizer.minimum_length = Some(128);
        decoder_tokenizer.start_token = Some(128_000);
        decoder_tokenizer.pad_token = 128_009;
        decoder_tokenizer.embedding_width = Some(4_096);
        assert!(native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::Llama,
            4_096,
            &llama_options,
            &decoder_tokenizer,
        ));
        decoder_tokenizer.maximum_length = 128;
        assert!(!native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::Llama,
            4_096,
            &llama_options,
            &decoder_tokenizer,
        ));
        decoder_tokenizer.maximum_length = SOURCE_UNBOUNDED_TOKENIZER_MAXIMUM_LENGTH;
        decoder_tokenizer.minimum_padding = Some(1);
        assert!(!native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::Llama,
            4_096,
            &llama_options,
            &decoder_tokenizer,
        ));
        decoder_tokenizer.minimum_padding = None;
        decoder_tokenizer.disable_weights = true;
        assert!(!native_clip_source_contract(
            NativeClipProfile::HiDream,
            NativeClipComponentRole::Llama,
            4_096,
            &llama_options,
            &decoder_tokenizer,
        ));
        Ok(())
    }

    #[test]
    fn mapped_clip_owned_allocations_count_arc_backing_and_preserve_aliases()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let mapped = Arc::new(MappedModelWeights::from_parts(
            "b".repeat(64),
            BTreeMap::from([(
                String::from("weight.with.retained.capacity"),
                tensor(&backend, &authority, &[1], vec![1.0], &cancellation)?,
            )]),
            vec![String::from("unexpected.with.retained.capacity")],
        ));
        let allocations = mapped_weights_owned_allocations(&mapped)?;
        let alias = mapped.clone();
        assert_eq!(allocations.len(), 3);
        assert_eq!(allocations, mapped_weights_owned_allocations(&alias)?);
        assert_eq!(
            allocations.iter().try_fold(0_u64, |total, allocation| {
                total
                    .checked_add(allocation.resident_bytes())
                    .ok_or("mapped allocation total overflow")
            })?,
            mapped.resident_owned_bytes()?
        );
        assert_eq!(
            allocations
                .iter()
                .map(NativeClipResidentAllocation::address)
                .collect::<BTreeSet<_>>()
                .len(),
            allocations.len()
        );
        let map_allocation = allocations
            .iter()
            .find(|allocation| allocation.address() == mapped.tensors() as *const _ as usize)
            .ok_or("mapped tensor-map allocation is missing")?;
        assert!(
            map_allocation.resident_bytes()
                >= u64::try_from(
                    mem::size_of::<BTreeMap<String, Tensor>>()
                        + mem::size_of::<(String, Tensor)>()
                        + mem::size_of::<usize>() * 2
                )?
        );
        let unexpected_allocation = allocations
            .iter()
            .find(|allocation| allocation.address() == mapped.unexpected_keys().as_ptr() as usize)
            .ok_or("unexpected-key allocation is missing")?;
        assert!(
            unexpected_allocation.resident_bytes()
                >= u64::try_from(mem::size_of::<String>() + mem::size_of::<usize>() * 2)?
        );
        Ok(())
    }

    #[test]
    fn native_pixart_empty_baseline_is_pad_only_and_weighted_masks_exclude_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let pad_token = 9_u32;
        let empty_tokens = NativePromptTokenizer::empty_token_ids(
            None,
            native_clip_empty_baseline_end(NativeClipProfile::PixArt, Some(1)),
            pad_token,
            4,
        )?;
        assert_eq!(empty_tokens, vec![pad_token; 4]);

        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024)?;
        let mask = tensor(
            &backend,
            &authority,
            &[3, 4],
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0],
            &cancellation,
        )?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let collapsed = collapse_clip_attention_mask(&backend, &mask, 2, 4, &context)?;
        assert_eq!(collapsed.descriptor().shape(), [1, 8]);
        assert_eq!(
            &*tensor_to_f32(&backend, &collapsed, &context)?,
            &[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn parameter_manifest_rejects_partial_ambiguous_state_and_projects_native_modules()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = linear_manifest()?;
        let probe = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: BTreeMap::from([
                (
                    "projection.weight".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2, 3],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
                (
                    "projection.bias".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
            ]),
            formats: Vec::new(),
        })?;
        manifest.validate_probes(std::slice::from_ref(&probe))?;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            manifest.validate_probes_cancellable(std::slice::from_ref(&probe), &cancelled),
            Err(ClipError::Tensor(TensorError::Cancelled))
        ));

        let partial = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: BTreeMap::from([(
                "projection.weight".to_owned(),
                ModelParsedTensorFact {
                    shape: vec![2, 3],
                    storage_dtype: "F32".to_owned(),
                },
            )]),
            formats: Vec::new(),
        })?;
        assert!(matches!(
            manifest.validate_probes(&[partial]),
            Err(ClipError::ParameterSetMismatch { .. })
        ));
        let unexpected = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: BTreeMap::from([
                (
                    "projection.weight".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2, 3],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
                (
                    "projection.bias".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
                (
                    "unexpected".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![1],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
            ]),
            formats: Vec::new(),
        })?;
        assert!(matches!(
            manifest.validate_probes(&[unexpected]),
            Err(ClipError::ParameterSetMismatch { .. })
        ));
        let wrong_shape = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: BTreeMap::from([
                (
                    "projection.weight".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![3, 2],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
                (
                    "projection.bias".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
            ]),
            formats: Vec::new(),
        })?;
        assert!(matches!(
            manifest.validate_probes(&[wrong_shape]),
            Err(ClipError::ManifestParameterShape { .. })
        ));
        let wrong_dtype = ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors: BTreeMap::from([
                (
                    "projection.weight".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2, 3],
                        storage_dtype: "F16".to_owned(),
                    },
                ),
                (
                    "projection.bias".to_owned(),
                    ModelParsedTensorFact {
                        shape: vec![2],
                        storage_dtype: "F32".to_owned(),
                    },
                ),
            ]),
            formats: Vec::new(),
        })?;
        assert!(matches!(
            manifest.validate_probes(&[wrong_dtype]),
            Err(ClipError::ManifestParameterDType { .. })
        ));
        assert!(matches!(
            ClipParameterManifest::checked(
                1,
                vec![
                    ClipParameterSpec::checked(0, "duplicate", vec![1], "f32")?,
                    ClipParameterSpec::checked(0, "duplicate", vec![1], "f32")?,
                ],
                BTreeSet::new(),
                Vec::new(),
            ),
            Err(ClipError::InvalidParameterManifest(_))
        ));

        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1_024)?;
        let state = BTreeMap::from([
            (
                "projection.weight".to_owned(),
                tensor(&backend, &authority, &[2, 3], vec![1.0; 6], &cancellation)?,
            ),
            (
                "projection.bias".to_owned(),
                tensor(&backend, &authority, &[2], vec![0.0; 2], &cancellation)?,
            ),
        ]);
        let projected = manifest.project_native_modules(&[state], DType::F32, DeviceId::CPU)?;
        let module = projected.get("projection").ok_or("projection missing")?;
        assert_eq!(module.generation(), 1);
        assert_eq!(module.layer_name(), "projection");
        Ok(())
    }

    fn write_clip_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let header = concat!(
            r#"{"projection.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"#,
            r#""projection.bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]},"#,
            r#""text_model.encoder.layers.0.mlp.fc1.weight":{"dtype":"U8","shape":[1],"data_offsets":[32,33]}}"#
        );
        let mut file = fs::File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(header.as_bytes())?;
        for value in [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.25, -0.25] {
            file.write_all(&value.to_le_bytes())?;
        }
        file.write_all(&[1])?;
        Ok(())
    }

    fn loaded_clip_fixture() -> Result<
        (
            tempfile::TempDir,
            ArtifactIndex,
            ModelStore,
            Arc<LoadedModel>,
        ),
        Box<dyn std::error::Error>,
    > {
        let directory = tempfile::tempdir()?;
        write_clip_safetensors(&directory.path().join("clip.safetensors"))?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "models",
            "text_encoders",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("models", "clip.safetensors")?,
            &cancellation,
        )?;
        Ok((directory, index, store, loaded))
    }

    fn loaded_clip_manifest(
        store: &ModelStore,
        loaded: &Arc<LoadedModel>,
    ) -> Result<ClipParameterManifest, Box<dyn std::error::Error>> {
        let probe = store.family_probe(loaded, &CancellationToken::default())?;
        let architecture =
            select_clip_architecture(ClipType::StableDiffusion, std::slice::from_ref(&probe))?;
        Ok(ClipParameterManifest::from_model_store(
            &architecture,
            store,
            std::slice::from_ref(loaded),
            vec![ClipNativeModuleSpec::checked(
                "projection",
                0,
                "projection.weight",
                Some("projection.bias".to_owned()),
                ClipNativeModuleKind::Linear {
                    input_features: 3,
                    output_features: 2,
                },
            )?],
            &CancellationToken::default(),
        )?)
    }

    #[test]
    fn model_store_native_module_load_is_atomic_cancellable_and_workspace_convergent()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_directory, index, store, loaded) = loaded_clip_fixture()?;
        let manifest = loaded_clip_manifest(&store, &loaded)?;
        let probe = store.family_probe(&loaded, &CancellationToken::default())?;
        let cross_target = select_clip_architecture(ClipType::Sd3, &[probe])?;
        assert!(matches!(
            manifest.validate_architecture(&cross_target),
            Err(ClipError::ManifestArchitectureMismatch { .. })
        ));
        assert!(matches!(
            linear_manifest()?.validate_architecture(&cross_target),
            Err(ClipError::ManifestArchitectureMismatch { actual: None, .. })
        ));
        let artifacts = [loaded];

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(48)?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let modules = load_manifest_modules(
            &store,
            &index,
            &artifacts,
            &manifest,
            &backend,
            DType::F32,
            DeviceId::CPU,
            &context,
        )?;
        assert_eq!(
            modules.get("projection").map(NativeModule::generation),
            Some(1)
        );
        assert_eq!(backend.memory_snapshot().current_bytes, 48);
        drop(modules);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);

        let (small_backend, small_authority) = CpuWorkspaceAuthority::create_backend(47)?;
        let small_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: small_authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            load_manifest_modules(
                &store,
                &index,
                &artifacts,
                &manifest,
                &small_backend,
                DType::F32,
                DeviceId::CPU,
                &small_context,
            ),
            Err(ClipError::Tensor(TensorError::AllocationFailed { .. }))
        ));
        assert_eq!(small_backend.memory_snapshot().current_bytes, 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            load_manifest_modules(
                &store,
                &index,
                &artifacts,
                &manifest,
                &backend,
                DType::F32,
                DeviceId::CPU,
                &cancelled_context,
            ),
            Err(ClipError::NativeModule(NativeOpsError::Cancelled))
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    fn layer(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
    ) -> Result<Sd1ClipLayerTensors, Box<dyn std::error::Error>> {
        let vector = |values| tensor(backend, authority, &[2], values, cancellation);
        let matrix = |values| tensor(backend, authority, &[2, 2], values, cancellation);
        let feed_forward_1 = |values| tensor(backend, authority, &[4, 2], values, cancellation);
        let feed_forward_2 = |values| tensor(backend, authority, &[2, 4], values, cancellation);
        Ok(Sd1ClipLayerTensors {
            layer_norm_1_weight: vector(vec![1.0, 1.0])?,
            layer_norm_1_bias: vector(vec![0.0, 0.0])?,
            query_weight: matrix(vec![0.0; 4])?,
            query_bias: vector(vec![0.0; 2])?,
            key_weight: matrix(vec![0.0; 4])?,
            key_bias: vector(vec![0.0; 2])?,
            value_weight: matrix(vec![0.0; 4])?,
            value_bias: vector(vec![0.0; 2])?,
            output_weight: matrix(vec![0.0; 4])?,
            output_bias: vector(vec![0.0; 2])?,
            layer_norm_2_weight: vector(vec![1.0, 1.0])?,
            layer_norm_2_bias: vector(vec![0.0, 0.0])?,
            feed_forward_1_weight: feed_forward_1(vec![0.0; 8])?,
            feed_forward_1_bias: tensor(backend, authority, &[4], vec![0.0; 4], cancellation)?,
            feed_forward_2_weight: feed_forward_2(vec![0.0; 8])?,
            feed_forward_2_bias: vector(vec![0.0; 2])?,
        })
    }

    fn encoder_fixture() -> Result<
        (Sd1ClipTextEncoder, CpuWorkspaceAuthority, CancellationToken),
        Box<dyn std::error::Error>,
    > {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let mut embeddings = vec![0.0; SD1_VOCABULARY_SIZE * 2];
        for (index, values) in embeddings.chunks_exact_mut(2).enumerate() {
            let value = (index % 97) as f32 / 97.0;
            values[0] = value;
            values[1] = 1.0 - value;
        }
        let positions = (0..SD1_CONTEXT_LENGTH)
            .flat_map(|position| [position as f32 / 100.0, -(position as f32) / 100.0])
            .collect::<Vec<_>>();
        let tensors = Sd1ClipTensors {
            token_embedding: tensor(
                &backend,
                &authority,
                &[SD1_VOCABULARY_SIZE as u64, 2],
                embeddings,
                &cancellation,
            )?,
            position_embedding: tensor(
                &backend,
                &authority,
                &[SD1_CONTEXT_LENGTH as u64, 2],
                positions,
                &cancellation,
            )?,
            layers: vec![
                layer(&backend, &authority, &cancellation)?,
                layer(&backend, &authority, &cancellation)?,
            ],
            final_layer_norm_weight: tensor(
                &backend,
                &authority,
                &[2],
                vec![1.0, 1.0],
                &cancellation,
            )?,
            final_layer_norm_bias: tensor(
                &backend,
                &authority,
                &[2],
                vec![0.0, 0.0],
                &cancellation,
            )?,
        };
        let parameters = Sd1ClipParameters::checked_legacy(
            binding('a')?,
            binding('b')?,
            binding('c')?,
            tensors,
            SD1_VOCABULARY_SIZE,
            SD1_CONTEXT_LENGTH,
            2,
            1,
        )?;
        Ok((
            Sd1ClipTextEncoder::new(Arc::new(backend), parameters)?,
            authority,
            cancellation,
        ))
    }

    fn retained_sd1_source_layer_fixture() -> Result<
        (
            Arc<Sd1Tokenizer>,
            Arc<LoadedSd1Clip>,
            CpuWorkspaceAuthority,
            CancellationToken,
        ),
        Box<dyn std::error::Error>,
    > {
        let tokenizer = Arc::new(fixture_tokenizer()?);
        let (encoder, authority, cancellation) = encoder_fixture()?;
        let mut parameters = encoder.parameters.clone();
        parameters
            .tensors
            .layers
            .get_mut(1)
            .ok_or("Missing second canonical layer")?
            .feed_forward_2_bias = tensor(
            &encoder.backend,
            &authority,
            &[2],
            vec![4.0, -4.0],
            &cancellation,
        )?;
        let backend = encoder.backend;
        let encoder = Sd1ClipTextEncoder::new(backend, parameters)?;
        let execution_plan = plan(
            &tokenizer,
            ClipLayerSelection::Final,
            ClipPooling::EndToken,
            DType::F32,
            DeviceId::CPU,
        )?;
        let architecture = select_clip_architecture(
            ClipType::StableDiffusion,
            &[detector_probe(TextEncoderModel::ClipL)?],
        )?;
        let clip = Arc::new(LoadedSd1Clip {
            binding: Sd1ClipExecutionBinding {
                architecture,
                profile_identity: "d".repeat(64),
                plan: execution_plan,
            },
            encoder,
        });
        Ok((tokenizer, clip, authority, cancellation))
    }

    #[test]
    fn retained_sd1_bridge_preserves_actual_storage_baseline_and_alias_accounting()
    -> Result<(), Box<dyn std::error::Error>> {
        let (tokenizer, clip, authority, cancellation) = retained_sd1_source_layer_fixture()?;
        let backend = clip.encoder.backend.clone();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32 * 1024 * 1024)?,
            &cancellation,
        );
        let legacy = NativeModelPayload::sd1_clip(tokenizer.clone(), clip.clone())?;
        let before = backend.memory_snapshot().current_bytes;
        let wrapper = NativeClipResource::from_retained_sd1(
            tokenizer.clone(),
            clip.clone(),
            &backend,
            &context,
        )?;
        let default = NativeModelPayload::native_clip_with_cancellation(
            Arc::new(wrapper.clone()),
            &cancellation,
        )?;
        assert_eq!(legacy.identity(), default.identity());
        assert_eq!(
            wrapper.resident_tensor_allocations()?,
            clip.resident_tensor_allocations()
        );
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        let changed = legacy.with_source_clip_layer(-2, &backend, &context)?;
        assert_eq!(
            legacy.identity().artifact_sha256(),
            changed.identity().artifact_sha256()
        );
        assert_eq!(
            legacy.identity().identifier(),
            changed.identity().identifier()
        );
        assert_eq!(legacy.identity().format(), changed.identity().format());
        assert_ne!(
            legacy.identity().execution_sha256(),
            changed.identity().execution_sha256()
        );
        let retained = changed
            .native_clip_resource()
            .ok_or("Wrapped source CLIP missing")?;
        assert_eq!(retained.source_layer_override(), Some(-2));
        let replaced = changed.with_source_clip_layer(-1, &backend, &context)?;
        let independently_replaced = legacy.with_source_clip_layer(-1, &backend, &context)?;
        assert_eq!(replaced.identity(), independently_replaced.identity());
        assert_eq!(
            replaced
                .native_clip_resource()
                .ok_or("Replaced CLIP missing")?
                .source_layer_override(),
            Some(-1),
        );
        let (actual_tokenizer, actual_clip) = retained
            .retained_sd1()
            .ok_or("Retained SD1 backing missing")?;
        assert!(Arc::ptr_eq(&tokenizer, actual_tokenizer));
        assert!(Arc::ptr_eq(&clip, actual_clip));
        for payload in [&legacy, &changed] {
            assert!(matches!(
                payload.scale_matching_state(&[], &backend, &context),
                Err(crate::native_node_payload::NativeModelPayloadError::UnsupportedTransformBacking {
                    role: crate::native_node_payload::NativeModelResourceRole::Clip,
                    architecture,
                }) if architecture == "zed-native-sd1-clip-v1"
            ));
        }
        let old_parts = legacy.resident_parts()?;
        let new_parts = changed.resident_parts()?;
        assert!(new_parts.tensor_allocations().is_empty());
        for old in old_parts.backing_allocations() {
            assert_eq!(
                new_parts
                    .backing_allocations()
                    .iter()
                    .filter(|new| **new == *old)
                    .count(),
                1
            );
        }
        assert_eq!(
            new_parts.backing_allocations().len(),
            old_parts.backing_allocations().len() + 1
        );
        let restarted = retained.restart(&backend, &context)?;
        assert_eq!(
            restarted.semantic_digest_sha256(),
            retained.semantic_digest_sha256()
        );
        let (_, restarted_clip) = restarted
            .retained_sd1()
            .ok_or("Restarted backing missing")?;
        assert!(Arc::ptr_eq(&clip, restarted_clip));
        assert_eq!(
            restarted.resident_tensor_allocations()?,
            clip.resident_tensor_allocations()
        );
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn retained_sd1_source_request_uses_actual_selected_layer_and_keeps_original_plan()
    -> Result<(), Box<dyn std::error::Error>> {
        let (tokenizer, clip, authority, cancellation) = retained_sd1_source_layer_fixture()?;
        let backend = clip.encoder.backend.clone();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32 * 1024 * 1024)?,
            &cancellation,
        );
        let batch = tokenizer.tokenize_batch(
            &[vec![WeightedText::checked("a test", 1.0)?]],
            &cancellation,
        )?;
        let original_digest = clip.plan().digest().to_owned();
        let original = clip.execute(&batch, &context)?;
        let earlier = clip.execute_with_source_layer(&batch, Some(-2), &context)?;
        let direct_plan = clip
            .plan()
            .with_request_layer(ClipLayerSelection::Hidden(0))?;
        let direct = clip.encoder.execute(&direct_plan, &batch, &context)?;
        assert_eq!(earlier.digest(), direct.digest());
        assert_eq!(
            earlier.conditioning().contiguous_bytes()?,
            direct.conditioning().contiguous_bytes()?
        );
        assert_ne!(
            earlier.conditioning().contiguous_bytes()?,
            original.conditioning().contiguous_bytes()?
        );
        for index in [i64::MIN, -3, 3, i64::MAX] {
            assert_eq!(
                clip.execute_with_source_layer(&batch, Some(index), &context)?
                    .digest(),
                original.digest()
            );
        }
        assert!(matches!(
            clip.execute_with_source_layer(&batch, Some(2), &context),
            Err(ClipError::LayerOutOfRange {
                requested: 2,
                available: 2
            })
        ));
        let wrapper =
            NativeClipResource::from_retained_sd1(tokenizer, clip.clone(), &backend, &context)?;
        let changed = wrapper.clone_with_source_layer(-2, &backend, &context)?;
        assert_eq!(
            changed
                .execute_retained_sd1_batch(&batch, &context)?
                .digest(),
            earlier.digest()
        );
        assert_eq!(clip.plan().digest(), original_digest);
        assert_eq!(clip.execute(&batch, &context)?.digest(), original.digest());
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn retained_sd1_bridge_cancellation_budget_and_foreign_backend_are_atomic()
    -> Result<(), Box<dyn std::error::Error>> {
        let (tokenizer, clip, authority, cancellation) = retained_sd1_source_layer_fixture()?;
        let backend = clip.encoder.backend.clone();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32 * 1024 * 1024)?,
            &cancellation,
        );
        let wrapper = NativeClipResource::from_retained_sd1(tokenizer, clip, &backend, &context)?;
        let before = backend.memory_snapshot().current_bytes;
        let constrained = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1)?,
            &cancellation,
        );
        assert!(matches!(
            wrapper.clone_with_source_layer(-2, &backend, &constrained),
            Err(ClipError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(constrained.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        let (foreign, foreign_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let foreign_context = foreign.execution_context(
            StreamId::DEFAULT,
            foreign_authority.authorize_workspace(32 * 1024 * 1024)?,
            &cancellation,
        );
        assert!(matches!(
            wrapper.execute(&foreign, "a test", &foreign_context),
            Err(ClipError::Tensor(
                TensorError::WorkspaceAuthorizationMismatch { .. }
            ))
        ));
        assert_eq!(foreign.memory_snapshot().current_bytes, 0);
        cancellation.cancel();
        assert!(matches!(
            wrapper.clone_with_source_layer(-2, &backend, &context),
            Err(ClipError::Tensor(TensorError::Cancelled))
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        Ok(())
    }

    #[test]
    fn sd1_executor_is_deterministic_layered_pooled_cancellable_and_workspace_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = fixture_tokenizer()?;
        let prompts = vec![
            vec![WeightedText::checked("a test", 1.0)?],
            vec![WeightedText::checked("a", 0.5)?],
        ];
        let batch = tokenizer.tokenize_batch(&prompts, &CancellationToken::default())?;
        let (encoder, authority, cancellation) = encoder_fixture()?;
        let execution_plan = plan(
            &tokenizer,
            ClipLayerSelection::Hidden(0),
            ClipPooling::EndToken,
            DType::F32,
            DeviceId::CPU,
        )?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(16 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let first = encoder.execute(&execution_plan, &batch, &context)?;
        let second = encoder.execute(&execution_plan, &batch, &context)?;
        assert_eq!(first.digest(), second.digest());
        assert_eq!(
            first.conditioning().descriptor().shape(),
            [2, SD1_CONTEXT_LENGTH as u64, 2]
        );
        assert_eq!(
            first.pooled().ok_or("missing pooled")?.descriptor().shape(),
            [2, 2]
        );
        assert_eq!(first.plan_identity(), execution_plan.digest());
        assert_eq!(first.token_batch_identity(), batch.digest());

        let final_pooling_plan = plan(
            &tokenizer,
            ClipLayerSelection::Final,
            ClipPooling::EndToken,
            DType::F32,
            DeviceId::CPU,
        )?;
        let final_pooling = encoder.execute(&final_pooling_plan, &batch, &context)?;
        let hidden_pool_values = tensor_to_f32(
            &encoder.backend,
            first.pooled().ok_or("missing hidden-plan pool")?,
            &context,
        )?;
        let final_pool_values = tensor_to_f32(
            &encoder.backend,
            final_pooling.pooled().ok_or("missing final-plan pool")?,
            &context,
        )?;
        assert_eq!(
            &*hidden_pool_values, &*final_pool_values,
            "a hidden-layer conditioning selection must still pool the final transformer state"
        );
        drop(hidden_pool_values);
        drop(final_pool_values);

        let unweighted_batch = tokenizer.tokenize_batch(
            &[
                vec![WeightedText::checked("a test", 1.0)?],
                vec![WeightedText::checked("a", 1.0)?],
            ],
            &CancellationToken::default(),
        )?;
        let unweighted = encoder.execute(&execution_plan, &unweighted_batch, &context)?;
        assert_ne!(unweighted.digest(), first.digest());

        let mean_plan = plan(
            &tokenizer,
            ClipLayerSelection::Final,
            ClipPooling::Mean,
            DType::F32,
            DeviceId::CPU,
        )?;
        let mean = encoder.execute(&mean_plan, &batch, &context)?;
        assert!(mean.pooled().is_some());
        assert_ne!(mean.digest(), first.digest());

        let invalid_layer = plan(
            &tokenizer,
            ClipLayerSelection::Hidden(2),
            ClipPooling::None,
            DType::F32,
            DeviceId::CPU,
        )?;
        assert!(matches!(
            encoder.execute(&invalid_layer, &batch, &context),
            Err(ClipError::LayerOutOfRange { .. })
        ));

        for dtype in [DType::F16, DType::Bf16, DType::F64] {
            let unsupported_plan = plan(
                &tokenizer,
                ClipLayerSelection::Final,
                ClipPooling::None,
                dtype,
                DeviceId::CPU,
            )?;
            let memory_before = encoder.backend.memory_snapshot();
            let scratch_before = context.scratch.in_use_bytes();
            assert!(matches!(
                encoder.execute(&unsupported_plan, &batch, &context),
                Err(ClipError::UnsupportedConcreteTarget { .. })
            ));
            assert_eq!(encoder.backend.memory_snapshot(), memory_before);
            assert_eq!(context.scratch.in_use_bytes(), scratch_before);
        }
        for device_kind in DeviceKind::ALL
            .into_iter()
            .filter(|device_kind| *device_kind != DeviceKind::Cpu)
        {
            let unsupported_plan = plan(
                &tokenizer,
                ClipLayerSelection::Final,
                ClipPooling::None,
                DType::F32,
                DeviceId::new(device_kind, 0),
            )?;
            let memory_before = encoder.backend.memory_snapshot();
            let scratch_before = context.scratch.in_use_bytes();
            assert!(matches!(
                encoder.execute(&unsupported_plan, &batch, &context),
                Err(ClipError::UnsupportedConcreteTarget { .. })
            ));
            assert_eq!(encoder.backend.memory_snapshot(), memory_before);
            assert_eq!(context.scratch.in_use_bytes(), scratch_before);
        }

        let insufficient_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        assert!(matches!(
            encoder.execute(&execution_plan, &batch, &insufficient_context),
            Err(ClipError::Tensor(
                TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(insufficient_context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(1024)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            encoder.execute(&execution_plan, &batch, &cancelled_context),
            Err(ClipError::Tensor(TensorError::Cancelled))
        ));
        Ok(())
    }
}
