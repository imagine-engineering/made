use crate::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionMask, AttentionMaskShape,
    AttentionRequest, BidirectionalTextError, BidirectionalTextOutput, BidirectionalTextRequest,
    ClipTextError, ClipTextOutput, ClipTextRequest, ClipVisionActivation, ClipVisionConfiguration,
    ClipVisionError, ClipVisionIntermediate, ClipVisionModelType, ClipVisionOutput,
    DecoderArchitecture, DecoderLayerKind, DecoderPreparedDeepstack,
    DecoderPreparedGenerationPrompt, DecoderRopePositions, DecoderTextConfiguration,
    DecoderTextError, DecoderTextOutput, DecoderTextRequest, GeluApproximation,
    Gemma3DecoderProfile, Gemma4DecoderProfile, GemmaTokenizerProfile, LLAMA_SOURCE_SHA256,
    MappedModelWeights, NativeClipText, NativeClipVision, NativeDecoderTextEncoder,
    NativeExecutionRequirements, NativeModule, NativeOpsError, NativePromptTokenizer,
    NativeT5TextEncoder, NativeTextGenerationRequest, NativeTextGenerationResult,
    NativeTokenizerError, QWEN25_TOKENIZER_ARTIFACT_DIGEST, QWEN35_SOURCE_SHA256,
    QWEN35_TOKENIZER_ARTIFACT_DIGEST, Qwen2PretokenizerProfile, SD1_CLIP_SOURCE_SHA256,
    SPIECE_TOKENIZER_SOURCE_SHA256, decoder_profile_fact, gemma3_decoder_configuration,
    gemma4_decoder_configuration, scaled_dot_product_attention_with_context,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sim_tensor::{
    CpuBackend, CpuWorkspaceVec, DType, DeviceId, ExecutionContext, ImageTensor, ResizeCrop,
    ResizeMode, RngTransaction, StreamId, Tensor, TensorDescriptor, TensorError,
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_linear_algebra_01::{LinearAlgebraPartOneError, matmul_with_context_exact_native},
    generated_native_diffusion::{
        NativeDiffusionTensorError, add as native_tensor_add, tensor_from_f32, tensor_to_f32,
    },
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, torch_cat_with_context_exact_native,
        torch_reshape_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        ShapeLayoutTransformPartThreeError, tensor_permute_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
    generated_spatial_functional_kernel_01::{
        InterpolateConfiguration, InterpolateMode, SpatialFunctionalKernelError,
        interpolate_with_context_exact_native,
    },
    generated_spectral_transform_01::{SpectralTransformError, fftn_with_context_exact_native},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    mem,
    sync::Arc,
};
use thiserror::Error;

pub const IDEOGRAM4_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/ideogram4.py";
pub const IDEOGRAM4_SOURCE_SHA256: &str =
    "dfe31ad9c3204bf2f98c81cbc498284f661f4162f35486baed38a9a847c52343";
pub const JINA_CLIP2_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/jina_clip_2.py";
pub const JINA_CLIP2_SOURCE_SHA256: &str =
    "d5f8f32dc9ebcdc55956bc6249f115c78483555bdd895b82814376d06a6fb3d1";
pub const OVIS_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/ovis.py";
pub const OVIS_SOURCE_SHA256: &str =
    "80749123bfa24b2947fef548101a17cd73085e6743b07dd1cb3f30490f45b65f";
pub const QWEN3VL_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/qwen3vl.py";
pub const QWEN3VL_SOURCE_SHA256: &str =
    "b2dce382d1319926af148ed65939c3f731bd4c9a277fc39f8ef0ca3ce8af7d06";
pub const QWEN_VL_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/qwen_vl.py";
pub const QWEN_VL_SOURCE_SHA256: &str =
    "7edda1550592057ebf71deae3b5e7d5577502d68f26e578e41b367ca8824070f";
pub const SAM3_CLIP_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/sam3_clip.py";
pub const SAM3_CLIP_SOURCE_SHA256: &str =
    "86f1f28b88cc364adc585efdfc6b72947c09c16c1fe52c81ac543f9717fa60df";

pub const IDEOGRAM4_TAP_LAYERS: [usize; 13] = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36];
pub const QWEN3VL_4B_DEEPSTACK_LAYERS: [usize; 3] = [5, 11, 17];
pub const QWEN3VL_8B_DEEPSTACK_LAYERS: [usize; 3] = [8, 16, 24];
pub const QWEN2VL_FULL_ATTENTION_LAYERS: [usize; 4] = [7, 15, 23, 31];
pub const QWEN3VL_IMAGE_PAD_TOKEN: i64 = 151_655;
pub const QWEN35_IMAGE_PAD_TOKEN: i64 = 248_056;
pub const QWEN35_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const QWEN35_IMAGE_STANDARD_DEVIATION: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
pub const QWEN_MULTIMODAL_ROUTING_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/sd.py";
pub const QWEN_MULTIMODAL_ROUTING_SOURCE_SHA256: &str =
    "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42";
pub const QWEN3VL_IMAGE_MINIMUM_PIXELS: u64 = 3_136;
pub const QWEN3VL_IMAGE_MAXIMUM_PIXELS: u64 = 12_845_056;
pub const QWEN3VL_IMAGE_PATCH_SIZE: usize = 16;
pub const QWEN3VL_IMAGE_TEMPORAL_PATCH_SIZE: usize = 2;
pub const QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE: usize = 2;
pub const GEMMA3_IMAGE_AREA_PIXELS: u64 = 896 * 896;
pub const GEMMA3_MAXIMUM_PREPARED_PIXELS: u64 = GEMMA3_IMAGE_AREA_PIXELS * 4;
pub const GEMMA3_IMAGE_SOFT_TOKENS: usize = 256;
pub const GEMMA3_MULTIMODAL_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/text_encoders/lt.py";
pub const GEMMA3_MULTIMODAL_SOURCE_SHA256: &str =
    "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2";
pub const GEMMA3_FOUR_B_MULTIMODAL_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/sim/ldm/lumina/model/lumina2.py";
pub const GEMMA3_FOUR_B_MULTIMODAL_SOURCE_SHA256: &str =
    "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84";
pub const GEMMA4_MULTIMODAL_SOURCE_PATH: &str =
    "projects/comfy/ComfyUI/sim/text_encoders/gemma4.py";
pub const GEMMA4_MULTIMODAL_SOURCE_SHA256: &str =
    "c6ffbb2fbecd8f97e781a654a06ccf3910dc670867d38c0ce30542312f00cde6";
pub const GEMMA4_IMAGE_PATCH_SIZE: u64 = 16;
pub const GEMMA4_IMAGE_POOLING_SIZE: u64 = 3;
pub const GEMMA4_IMAGE_SOFT_TOKENS: usize = 280;
pub const GEMMA4_VIDEO_SOFT_TOKENS: usize = 70;
pub const GEMMA4_VIDEO_SOURCE_FPS: usize = 24;
pub const GEMMA4_AUDIO_SAMPLE_RATE: u32 = 16_000;
pub const GEMMA4_AUDIO_MINIMUM_SAMPLE_RATE: u32 = 8_000;
pub const GEMMA4_AUDIO_MAXIMUM_SAMPLE_RATE: u32 = 384_000;
pub const GEMMA4_AUDIO_FRAME_LENGTH: usize = 320;
pub const GEMMA4_AUDIO_FRAME_STEP: usize = 160;
pub const GEMMA4_AUDIO_FFT_LENGTH: usize = 512;
pub const GEMMA4_AUDIO_MEL_BINS: usize = 128;
pub const GEMMA4_AUDIO_MAXIMUM_TOKENS: usize = 750;

const GEMMA4_AUDIO_PADDING_MULTIPLE: usize = 128;
const GEMMA4_AUDIO_KAISER_BETA: f64 = 6.5;
const GEMMA4_AUDIO_FILTER_HALF_WIDTH: usize = 80;
const GEMMA4_AUDIO_MAXIMUM_FILTER_TAPS: usize = 2_000_001;
const GEMMA4_AUDIO_MAXIMUM_RESAMPLE_MULTIPLY_ADDS: usize = 250_000_000;

pub const MULTIMODAL_TEXT_ENCODER_CATALOG_SYMBOLS: [&str; 53] = [
    "Qwen3VLTokenizer",
    "Ideogram4Tokenizer",
    "Qwen3VL8BModel",
    "Ideogram4TEModel",
    "te",
    "Ideogram4Qwen3VLClipModel",
    "Ideogram4Qwen3VLTEModel",
    "Ideogram4Qwen3VLTokenizer",
    "te_qwen3vl",
    "JinaClip2Tokenizer",
    "JinaClip2TokenizerWrapper",
    "XLMRobertaConfig",
    "XLMRobertaEmbeddings",
    "RotaryEmbedding",
    "MHA",
    "MLP",
    "Block",
    "XLMRobertaEncoder",
    "XLMRobertaModel_",
    "XLMRobertaModel",
    "JinaClip2TextModel",
    "JinaClip2TextModelWrapper",
    "Qwen3Tokenizer",
    "OvisTokenizer",
    "Ovis25_2BModel",
    "OvisTEModel",
    "te",
    "Qwen3VLDeepstackMerger",
    "Qwen3VLVisionModel",
    "Qwen3VL",
    "_make_qwen3vl_model",
    "Qwen3VLClipModel",
    "Qwen3VLTEModel",
    "Qwen3VLSDTokenizer",
    "Qwen3VLTokenizer",
    "tokenizer",
    "te",
    "process_qwen2vl_images",
    "qwen2vl_mrope_position_ids",
    "VisionPatchEmbed",
    "rotate_half",
    "apply_rotary_pos_emb_vision",
    "VisionRotaryEmbedding",
    "PatchMerger",
    "VisionAttention",
    "VisionMLP",
    "VisionBlock",
    "Qwen2VLVisionTransformer",
    "SAM3ClipModel",
    "SAM3Tokenizer",
    "_parse_prompts",
    "SAM3TokenizerWrapper",
    "SAM3ClipModelWrapper",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalSymbolBehavior {
    Profile,
    ProfileFactory,
    TokenizerAdapter,
    BidirectionalTextDelegation,
    DecoderTextDelegation,
    ClipTextDelegation,
    VisionDelegation,
    ImagePreprocessDelegation,
    PositionConstruction,
    ModalityJoin,
    Projection,
    PromptPacking,
    ModelAdapter,
}

pub fn multimodal_symbol_behavior(
    source_path: &str,
    symbol: &str,
) -> Option<MultimodalSymbolBehavior> {
    use MultimodalSymbolBehavior as Behavior;
    Some(match (source_path, symbol) {
        (IDEOGRAM4_SOURCE_PATH, "Qwen3VLTokenizer")
        | (IDEOGRAM4_SOURCE_PATH, "Ideogram4Tokenizer")
        | (IDEOGRAM4_SOURCE_PATH, "Ideogram4Qwen3VLTokenizer")
        | (JINA_CLIP2_SOURCE_PATH, "JinaClip2Tokenizer")
        | (JINA_CLIP2_SOURCE_PATH, "JinaClip2TokenizerWrapper")
        | (OVIS_SOURCE_PATH, "Qwen3Tokenizer")
        | (OVIS_SOURCE_PATH, "OvisTokenizer")
        | (QWEN3VL_SOURCE_PATH, "Qwen3VLSDTokenizer")
        | (QWEN3VL_SOURCE_PATH, "Qwen3VLTokenizer")
        | (QWEN3VL_SOURCE_PATH, "tokenizer")
        | (SAM3_CLIP_SOURCE_PATH, "SAM3Tokenizer") => Behavior::TokenizerAdapter,
        (JINA_CLIP2_SOURCE_PATH, "XLMRobertaConfig") => Behavior::Profile,
        (QWEN3VL_SOURCE_PATH, "_make_qwen3vl_model") => Behavior::ProfileFactory,
        (JINA_CLIP2_SOURCE_PATH, "XLMRobertaEmbeddings")
        | (JINA_CLIP2_SOURCE_PATH, "MHA")
        | (JINA_CLIP2_SOURCE_PATH, "MLP")
        | (JINA_CLIP2_SOURCE_PATH, "Block")
        | (JINA_CLIP2_SOURCE_PATH, "XLMRobertaEncoder")
        | (JINA_CLIP2_SOURCE_PATH, "XLMRobertaModel_")
        | (JINA_CLIP2_SOURCE_PATH, "XLMRobertaModel") => Behavior::BidirectionalTextDelegation,
        (IDEOGRAM4_SOURCE_PATH, "Qwen3VL8BModel") | (OVIS_SOURCE_PATH, "Ovis25_2BModel") => {
            Behavior::DecoderTextDelegation
        }
        (SAM3_CLIP_SOURCE_PATH, "SAM3ClipModel") => Behavior::ClipTextDelegation,
        (QWEN3VL_SOURCE_PATH, "Qwen3VLVisionModel")
        | (QWEN_VL_SOURCE_PATH, "VisionPatchEmbed")
        | (QWEN_VL_SOURCE_PATH, "VisionAttention")
        | (QWEN_VL_SOURCE_PATH, "VisionMLP")
        | (QWEN_VL_SOURCE_PATH, "VisionBlock")
        | (QWEN_VL_SOURCE_PATH, "Qwen2VLVisionTransformer") => Behavior::VisionDelegation,
        (QWEN_VL_SOURCE_PATH, "process_qwen2vl_images") => Behavior::ImagePreprocessDelegation,
        (JINA_CLIP2_SOURCE_PATH, "RotaryEmbedding")
        | (QWEN_VL_SOURCE_PATH, "qwen2vl_mrope_position_ids")
        | (QWEN_VL_SOURCE_PATH, "rotate_half")
        | (QWEN_VL_SOURCE_PATH, "apply_rotary_pos_emb_vision")
        | (QWEN_VL_SOURCE_PATH, "VisionRotaryEmbedding") => Behavior::PositionConstruction,
        (QWEN3VL_SOURCE_PATH, "Qwen3VL") => Behavior::ModalityJoin,
        (IDEOGRAM4_SOURCE_PATH, "Ideogram4TEModel")
        | (IDEOGRAM4_SOURCE_PATH, "Ideogram4Qwen3VLTEModel")
        | (QWEN3VL_SOURCE_PATH, "Qwen3VLDeepstackMerger")
        | (QWEN_VL_SOURCE_PATH, "PatchMerger") => Behavior::Projection,
        (SAM3_CLIP_SOURCE_PATH, "_parse_prompts")
        | (SAM3_CLIP_SOURCE_PATH, "SAM3TokenizerWrapper")
        | (SAM3_CLIP_SOURCE_PATH, "SAM3ClipModelWrapper") => Behavior::PromptPacking,
        (IDEOGRAM4_SOURCE_PATH, "te")
        | (IDEOGRAM4_SOURCE_PATH, "Ideogram4Qwen3VLClipModel")
        | (IDEOGRAM4_SOURCE_PATH, "te_qwen3vl")
        | (JINA_CLIP2_SOURCE_PATH, "JinaClip2TextModel")
        | (JINA_CLIP2_SOURCE_PATH, "JinaClip2TextModelWrapper")
        | (OVIS_SOURCE_PATH, "OvisTEModel")
        | (OVIS_SOURCE_PATH, "te")
        | (QWEN3VL_SOURCE_PATH, "Qwen3VLClipModel")
        | (QWEN3VL_SOURCE_PATH, "Qwen3VLTEModel")
        | (QWEN3VL_SOURCE_PATH, "te") => Behavior::ModelAdapter,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalFamily {
    Ideogram4,
    JinaClip2,
    Ovis25,
    Qwen3Vl4B,
    Qwen3Vl8B,
    Qwen2Vl,
    Sam3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultimodalTextOwner {
    Bidirectional,
    Decoder,
    ClipText,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultimodalProfileFact {
    pub family: MultimodalFamily,
    pub text_owner: MultimodalTextOwner,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub maximum_tokens: usize,
    pub vocabulary_size: usize,
    pub pad_token: i64,
    pub projection_size: Option<usize>,
    pub patch_size: Option<usize>,
    pub temporal_patch_size: Option<usize>,
    pub spatial_merge_size: Option<usize>,
    pub visual_layer_count: Option<usize>,
    pub rope_theta_bits: Option<u32>,
    pub layer_taps: &'static [usize],
    pub deepstack_layers: &'static [usize],
}

impl MultimodalProfileFact {
    pub fn rope_theta(&self) -> Option<f32> {
        self.rope_theta_bits.map(f32::from_bits)
    }
}

pub const MULTIMODAL_PROFILE_FACTS: [MultimodalProfileFact; 7] = [
    MultimodalProfileFact {
        family: MultimodalFamily::Ideogram4,
        text_owner: MultimodalTextOwner::Decoder,
        hidden_size: 4096,
        intermediate_size: 12_288,
        layer_count: 36,
        attention_heads: 32,
        maximum_tokens: 99_999_999,
        vocabulary_size: 151_936,
        pad_token: 151_643,
        projection_size: Some(53_248),
        patch_size: Some(16),
        temporal_patch_size: Some(2),
        spatial_merge_size: Some(2),
        visual_layer_count: Some(27),
        rope_theta_bits: Some(5_000_000.0_f32.to_bits()),
        layer_taps: &IDEOGRAM4_TAP_LAYERS,
        deepstack_layers: &QWEN3VL_8B_DEEPSTACK_LAYERS,
    },
    MultimodalProfileFact {
        family: MultimodalFamily::JinaClip2,
        text_owner: MultimodalTextOwner::Bidirectional,
        hidden_size: 1024,
        intermediate_size: 4096,
        layer_count: 24,
        attention_heads: 16,
        maximum_tokens: 8192,
        vocabulary_size: 250_002,
        pad_token: 1,
        projection_size: None,
        patch_size: None,
        temporal_patch_size: None,
        spatial_merge_size: None,
        visual_layer_count: None,
        rope_theta_bits: Some(20_000.0_f32.to_bits()),
        layer_taps: &[],
        deepstack_layers: &[],
    },
    MultimodalProfileFact {
        family: MultimodalFamily::Ovis25,
        text_owner: MultimodalTextOwner::Decoder,
        hidden_size: 2048,
        intermediate_size: 6144,
        layer_count: 28,
        attention_heads: 16,
        maximum_tokens: 99_999_999,
        vocabulary_size: 151_936,
        pad_token: 151_643,
        projection_size: None,
        patch_size: None,
        temporal_patch_size: None,
        spatial_merge_size: None,
        visual_layer_count: None,
        rope_theta_bits: Some(1_000_000.0_f32.to_bits()),
        layer_taps: &[],
        deepstack_layers: &[],
    },
    MultimodalProfileFact {
        family: MultimodalFamily::Qwen3Vl4B,
        text_owner: MultimodalTextOwner::Decoder,
        hidden_size: 2560,
        intermediate_size: 9728,
        layer_count: 36,
        attention_heads: 32,
        maximum_tokens: 262_144,
        vocabulary_size: 151_936,
        pad_token: 151_643,
        projection_size: None,
        patch_size: Some(16),
        temporal_patch_size: Some(2),
        spatial_merge_size: Some(2),
        visual_layer_count: Some(24),
        rope_theta_bits: Some(5_000_000.0_f32.to_bits()),
        layer_taps: &[],
        deepstack_layers: &QWEN3VL_4B_DEEPSTACK_LAYERS,
    },
    MultimodalProfileFact {
        family: MultimodalFamily::Qwen3Vl8B,
        text_owner: MultimodalTextOwner::Decoder,
        hidden_size: 4096,
        intermediate_size: 12_288,
        layer_count: 36,
        attention_heads: 32,
        maximum_tokens: 262_144,
        vocabulary_size: 151_936,
        pad_token: 151_643,
        projection_size: None,
        patch_size: Some(16),
        temporal_patch_size: Some(2),
        spatial_merge_size: Some(2),
        visual_layer_count: Some(27),
        rope_theta_bits: Some(5_000_000.0_f32.to_bits()),
        layer_taps: &[],
        deepstack_layers: &QWEN3VL_8B_DEEPSTACK_LAYERS,
    },
    MultimodalProfileFact {
        family: MultimodalFamily::Qwen2Vl,
        text_owner: MultimodalTextOwner::Decoder,
        hidden_size: 3584,
        intermediate_size: 3420,
        layer_count: 32,
        attention_heads: 16,
        maximum_tokens: 128_000,
        vocabulary_size: 152_064,
        pad_token: 151_643,
        projection_size: Some(3584),
        patch_size: Some(14),
        temporal_patch_size: Some(2),
        spatial_merge_size: Some(2),
        visual_layer_count: Some(32),
        rope_theta_bits: Some(10_000.0_f32.to_bits()),
        layer_taps: &QWEN2VL_FULL_ATTENTION_LAYERS,
        deepstack_layers: &[],
    },
    MultimodalProfileFact {
        family: MultimodalFamily::Sam3,
        text_owner: MultimodalTextOwner::ClipText,
        hidden_size: 1024,
        intermediate_size: 4096,
        layer_count: 24,
        attention_heads: 16,
        maximum_tokens: 32,
        vocabulary_size: 49_408,
        pad_token: 0,
        projection_size: Some(512),
        patch_size: None,
        temporal_patch_size: None,
        spatial_merge_size: None,
        visual_layer_count: None,
        rope_theta_bits: None,
        layer_taps: &[],
        deepstack_layers: &[],
    },
];

pub fn multimodal_profile(family: MultimodalFamily) -> &'static MultimodalProfileFact {
    MULTIMODAL_PROFILE_FACTS
        .iter()
        .find(|profile| profile.family == family)
        .expect("every closed multimodal family has one immutable profile")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultimodalSpan {
    pub start: usize,
    pub size: usize,
    pub grid_thw: [usize; 3],
}

#[derive(Clone, Debug)]
pub struct MultimodalImageEmbedding<'a> {
    pub span: MultimodalSpan,
    pub embedding: &'a Tensor,
    pub deepstack: &'a [Tensor],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultimodalPositionIds {
    temporal: Vec<i64>,
    height: Vec<i64>,
    width: Vec<i64>,
}

impl MultimodalPositionIds {
    pub fn temporal(&self) -> &[i64] {
        &self.temporal
    }

    pub fn height(&self) -> &[i64] {
        &self.height
    }

    pub fn width(&self) -> &[i64] {
        &self.width
    }

    pub fn len(&self) -> usize {
        self.temporal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.temporal.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct MultimodalDeepstackJoin {
    pub visual_position_mask: Vec<bool>,
    pub layers: Vec<Tensor>,
}

#[derive(Clone, Debug)]
pub struct Qwen3VlPreparedImage {
    patches: Tensor,
    grid_thw: [usize; 3],
    merged_tokens: usize,
    family: QwenVisionFamily,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmaPreparedVisualKind {
    Gemma3Image,
    Gemma4Image,
    Gemma4VideoFrame,
}

#[derive(Clone, Debug)]
pub struct GemmaPreparedVisual {
    image: ImageTensor,
    kind: GemmaPreparedVisualKind,
    maximum_soft_tokens: usize,
    source_frame_index: usize,
    timestamp_seconds: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct GemmaPreparedAudio {
    log_mel: Tensor,
    frame_mask: Tensor,
    marker_tokens: usize,
    original_sample_rate: u32,
    original_samples: usize,
    resampled_samples: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Gemma3VisionProfile {
    FourBVision,
    TwelveB,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Gemma3VisionConfiguration {
    pub source_profile: Option<Gemma3VisionProfile>,
    pub vision_hidden_size: usize,
    pub output_hidden_size: usize,
    pub patches_per_side: usize,
    pub tokens_per_side: usize,
    pub normalization_epsilon_bits: u32,
    pub normalization_weight_offset_bits: u32,
}

impl Gemma3VisionConfiguration {
    pub fn source(profile: Gemma3VisionProfile) -> Self {
        Self {
            source_profile: Some(profile),
            vision_hidden_size: 1152,
            output_hidden_size: match profile {
                Gemma3VisionProfile::FourBVision => 2560,
                Gemma3VisionProfile::TwelveB => 3840,
            },
            patches_per_side: 64,
            tokens_per_side: 16,
            normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
            normalization_weight_offset_bits: 1.0_f32.to_bits(),
        }
    }

    pub fn reduced_fixture(
        vision_hidden_size: usize,
        output_hidden_size: usize,
        patches_per_side: usize,
        tokens_per_side: usize,
    ) -> Self {
        Self {
            source_profile: None,
            vision_hidden_size,
            output_hidden_size,
            patches_per_side,
            tokens_per_side,
            normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
            normalization_weight_offset_bits: 1.0_f32.to_bits(),
        }
    }

    fn validate(&self) -> Result<(), MultimodalTextError> {
        if self.vision_hidden_size == 0
            || self.output_hidden_size == 0
            || self.patches_per_side == 0
            || self.tokens_per_side == 0
            || !self.patches_per_side.is_multiple_of(self.tokens_per_side)
            || !f32::from_bits(self.normalization_epsilon_bits).is_finite()
            || f32::from_bits(self.normalization_epsilon_bits) <= 0.0
            || !f32::from_bits(self.normalization_weight_offset_bits).is_finite()
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma3 vision projection dimensions or normalization are invalid",
            ));
        }
        if let Some(profile) = self.source_profile
            && self != &Self::source(profile)
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma3 production vision configuration does not match its closed source profile",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NativeGemma3VisionProjector {
    configuration: Gemma3VisionConfiguration,
    vision: Arc<NativeClipVision>,
    normalization_weight: Tensor,
    input_projection_weight: Tensor,
    semantic_state_digest_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Gemma3VisionProjection {
    pub embedding: Tensor,
    pub tokens_per_image: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Gemma4VisionProfile {
    E2B,
    E4B,
    ThirtyOneB,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Gemma4VisionConfiguration {
    pub profile: Gemma4VisionProfile,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub head_dimension: usize,
    pub patch_size: usize,
    pub position_embeddings: usize,
    pub pooling_kernel_size: usize,
    pub output_hidden_size: usize,
    pub normalization_epsilon_bits: u32,
    pub rotary_theta_bits: u32,
    pub source_exact_profile: bool,
}

impl Gemma4VisionConfiguration {
    pub fn source(profile: Gemma4VisionProfile) -> Self {
        let (hidden_size, intermediate_size, layer_count, attention_heads, head_dimension) =
            match profile {
                Gemma4VisionProfile::E2B | Gemma4VisionProfile::E4B => (768, 3072, 16, 12, 64),
                Gemma4VisionProfile::ThirtyOneB => (1152, 4304, 27, 16, 72),
            };
        Self {
            profile,
            hidden_size,
            intermediate_size,
            layer_count,
            attention_heads,
            head_dimension,
            patch_size: 16,
            position_embeddings: 10_240,
            pooling_kernel_size: 3,
            output_hidden_size: match profile {
                Gemma4VisionProfile::E2B => 1536,
                Gemma4VisionProfile::E4B => 2560,
                Gemma4VisionProfile::ThirtyOneB => 5376,
            },
            normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
            rotary_theta_bits: 100.0_f32.to_bits(),
            source_exact_profile: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reduced_fixture(
        profile: Gemma4VisionProfile,
        hidden_size: usize,
        intermediate_size: usize,
        layer_count: usize,
        attention_heads: usize,
        head_dimension: usize,
        patch_size: usize,
        position_embeddings: usize,
        pooling_kernel_size: usize,
        output_hidden_size: usize,
    ) -> Self {
        Self {
            profile,
            hidden_size,
            intermediate_size,
            layer_count,
            attention_heads,
            head_dimension,
            patch_size,
            position_embeddings,
            pooling_kernel_size,
            output_hidden_size,
            normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
            rotary_theta_bits: 100.0_f32.to_bits(),
            source_exact_profile: false,
        }
    }

    fn validate(&self) -> Result<(), MultimodalTextError> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.layer_count == 0
            || self.attention_heads == 0
            || self.head_dimension == 0
            || self.patch_size == 0
            || self.position_embeddings == 0
            || self.pooling_kernel_size == 0
            || self.output_hidden_size == 0
            || !self.head_dimension.is_multiple_of(4)
            || self.hidden_size
                != self
                    .attention_heads
                    .checked_mul(self.head_dimension)
                    .ok_or(MultimodalTextError::Overflow(
                        "Gemma4 vision attention width",
                    ))?
            || !f32::from_bits(self.normalization_epsilon_bits).is_finite()
            || f32::from_bits(self.normalization_epsilon_bits) <= 0.0
            || !f32::from_bits(self.rotary_theta_bits).is_finite()
            || f32::from_bits(self.rotary_theta_bits) <= 0.0
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 vision configuration dimensions are invalid",
            ));
        }
        if self.source_exact_profile && self != &Self::source(self.profile) {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 production vision configuration does not match its closed source profile",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Gemma4ClippedLinearWeights {
    pub weight: Tensor,
    pub input_minimum: Tensor,
    pub input_maximum: Tensor,
    pub output_minimum: Tensor,
    pub output_maximum: Tensor,
}

#[derive(Clone, Debug)]
pub struct Gemma4VisionBlockWeights {
    pub input_normalization_weight: Tensor,
    pub query: Gemma4ClippedLinearWeights,
    pub key: Gemma4ClippedLinearWeights,
    pub value: Gemma4ClippedLinearWeights,
    pub query_normalization_weight: Tensor,
    pub key_normalization_weight: Tensor,
    pub attention_output: Gemma4ClippedLinearWeights,
    pub post_attention_normalization_weight: Tensor,
    pub pre_feed_forward_normalization_weight: Tensor,
    pub feed_forward_gate: Gemma4ClippedLinearWeights,
    pub feed_forward_up: Gemma4ClippedLinearWeights,
    pub feed_forward_down: Gemma4ClippedLinearWeights,
    pub post_feed_forward_normalization_weight: Tensor,
}

#[derive(Clone, Debug)]
pub struct Gemma4VisionWeights {
    pub patch_projection_weight: Tensor,
    pub position_embedding: Tensor,
    pub blocks: Vec<Gemma4VisionBlockWeights>,
    pub projector_weight: Tensor,
}

#[derive(Clone, Debug)]
struct NativeGemma4ClippedLinear {
    linear: NativeModule,
    input_minimum: Tensor,
    input_maximum: Tensor,
    output_minimum: Tensor,
    output_maximum: Tensor,
}

#[derive(Clone, Debug)]
struct NativeGemma4VisionBlock {
    input_normalization_weight: Tensor,
    query: NativeGemma4ClippedLinear,
    key: NativeGemma4ClippedLinear,
    value: NativeGemma4ClippedLinear,
    query_normalization_weight: Tensor,
    key_normalization_weight: Tensor,
    attention_output: NativeGemma4ClippedLinear,
    post_attention_normalization_weight: Tensor,
    pre_feed_forward_normalization_weight: Tensor,
    feed_forward_gate: NativeGemma4ClippedLinear,
    feed_forward_up: NativeGemma4ClippedLinear,
    feed_forward_down: NativeGemma4ClippedLinear,
    post_feed_forward_normalization_weight: Tensor,
}

#[derive(Clone, Debug)]
pub struct NativeGemma4VisionEncoder {
    configuration: Gemma4VisionConfiguration,
    patch_projection: NativeModule,
    position_embedding: Tensor,
    blocks: Vec<NativeGemma4VisionBlock>,
    projector: NativeModule,
    semantic_state_digest_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Gemma4VisionProjection {
    pub embedding: Tensor,
    pub tokens: usize,
    pub kind: GemmaPreparedVisualKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum Gemma4AudioProfile {
    E2B,
    E4B,
    ThirtyOneB,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Gemma4AudioConfiguration {
    pub profile: Gemma4AudioProfile,
    pub mel_bins: usize,
    pub first_convolution_channels: usize,
    pub second_convolution_channels: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub convolution_kernel_size: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    pub encoder_output_size: usize,
    pub output_hidden_size: usize,
    pub normalization_epsilon_bits: u32,
    pub residual_weight_bits: u32,
    pub attention_logit_cap_bits: u32,
    pub source_exact_profile: bool,
}

impl Gemma4AudioConfiguration {
    pub fn source(profile: Gemma4AudioProfile) -> Result<Self, MultimodalTextError> {
        if profile == Gemma4AudioProfile::ThirtyOneB {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 31B has no audio encoder",
            ));
        }
        Ok(Self {
            profile,
            mel_bins: 128,
            first_convolution_channels: 128,
            second_convolution_channels: 32,
            hidden_size: 1024,
            intermediate_size: 4096,
            layer_count: 12,
            attention_heads: 8,
            convolution_kernel_size: 5,
            attention_chunk_size: 12,
            attention_context_left: 13,
            attention_context_right: 0,
            encoder_output_size: 1536,
            output_hidden_size: if profile == Gemma4AudioProfile::E2B {
                1536
            } else {
                2560
            },
            normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
            residual_weight_bits: 0.5_f32.to_bits(),
            attention_logit_cap_bits: 50.0_f32.to_bits(),
            source_exact_profile: true,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reduced_fixture(
        profile: Gemma4AudioProfile,
        mel_bins: usize,
        first_convolution_channels: usize,
        second_convolution_channels: usize,
        hidden_size: usize,
        intermediate_size: usize,
        layer_count: usize,
        attention_heads: usize,
        convolution_kernel_size: usize,
        attention_chunk_size: usize,
        attention_context_left: usize,
        encoder_output_size: usize,
        output_hidden_size: usize,
    ) -> Result<Self, MultimodalTextError> {
        let configuration = Self {
            profile,
            mel_bins,
            first_convolution_channels,
            second_convolution_channels,
            hidden_size,
            intermediate_size,
            layer_count,
            attention_heads,
            convolution_kernel_size,
            attention_chunk_size,
            attention_context_left,
            attention_context_right: 0,
            encoder_output_size,
            output_hidden_size,
            normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
            residual_weight_bits: 0.5_f32.to_bits(),
            attention_logit_cap_bits: 50.0_f32.to_bits(),
            source_exact_profile: false,
        };
        configuration.validate()?;
        Ok(configuration)
    }

    fn validate(&self) -> Result<(), MultimodalTextError> {
        let after_first = self.mel_bins.div_ceil(2);
        let after_second = after_first.div_ceil(2);
        if self.profile == Gemma4AudioProfile::ThirtyOneB
            || self.mel_bins == 0
            || self.first_convolution_channels == 0
            || self.second_convolution_channels == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.layer_count == 0
            || self.attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.attention_heads)
            || self.convolution_kernel_size == 0
            || self.attention_chunk_size == 0
            || self.attention_context_left == 0
            || self.encoder_output_size == 0
            || self.output_hidden_size == 0
            || after_second
                .checked_mul(self.second_convolution_channels)
                .is_none()
            || !f32::from_bits(self.normalization_epsilon_bits).is_finite()
            || f32::from_bits(self.normalization_epsilon_bits) <= 0.0
            || !f32::from_bits(self.residual_weight_bits).is_finite()
            || !f32::from_bits(self.attention_logit_cap_bits).is_finite()
            || f32::from_bits(self.attention_logit_cap_bits) <= 0.0
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio configuration dimensions are invalid",
            ));
        }
        if self.source_exact_profile && self != &Self::source(self.profile)? {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 production audio configuration does not match its closed source profile",
            ));
        }
        Ok(())
    }

    fn subsampled_feature_width(&self) -> Result<usize, MultimodalTextError> {
        self.mel_bins
            .div_ceil(2)
            .div_ceil(2)
            .checked_mul(self.second_convolution_channels)
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 audio subsampled feature width",
            ))
    }
}

#[derive(Clone, Debug)]
pub struct Gemma4AudioFeedForwardWeights {
    pub pre_normalization_weight: Tensor,
    pub first: Gemma4ClippedLinearWeights,
    pub second: Gemma4ClippedLinearWeights,
    pub post_normalization_weight: Tensor,
}

#[derive(Clone, Debug)]
pub struct Gemma4AudioBlockWeights {
    pub feed_forward_one: Gemma4AudioFeedForwardWeights,
    pub query: Gemma4ClippedLinearWeights,
    pub key: Gemma4ClippedLinearWeights,
    pub value: Gemma4ClippedLinearWeights,
    pub attention_output: Gemma4ClippedLinearWeights,
    pub attention_scale: Tensor,
    pub relative_key_projection_weight: Tensor,
    pub pre_attention_normalization_weight: Tensor,
    pub post_attention_normalization_weight: Tensor,
    pub convolution_pre_normalization_weight: Tensor,
    pub convolution_start: Gemma4ClippedLinearWeights,
    pub depthwise_convolution_weight: Tensor,
    pub convolution_normalization_weight: Tensor,
    pub convolution_end: Gemma4ClippedLinearWeights,
    pub feed_forward_two: Gemma4AudioFeedForwardWeights,
    pub output_normalization_weight: Tensor,
}

#[derive(Clone, Debug)]
pub struct Gemma4AudioWeights {
    pub first_convolution_weight: Tensor,
    pub first_convolution_normalization_weight: Tensor,
    pub second_convolution_weight: Tensor,
    pub second_convolution_normalization_weight: Tensor,
    pub subsample_projection_weight: Tensor,
    pub blocks: Vec<Gemma4AudioBlockWeights>,
    pub encoder_output_weight: Tensor,
    pub encoder_output_bias: Tensor,
    pub projector_weight: Tensor,
}

#[derive(Clone, Debug)]
struct NativeGemma4AudioFeedForward {
    pre_normalization_weight: Tensor,
    first: NativeGemma4ClippedLinear,
    second: NativeGemma4ClippedLinear,
    post_normalization_weight: Tensor,
}

#[derive(Clone, Debug)]
struct NativeGemma4AudioBlock {
    feed_forward_one: NativeGemma4AudioFeedForward,
    query: NativeGemma4ClippedLinear,
    key: NativeGemma4ClippedLinear,
    value: NativeGemma4ClippedLinear,
    attention_output: NativeGemma4ClippedLinear,
    attention_scale: Tensor,
    relative_key_projection: NativeModule,
    pre_attention_normalization_weight: Tensor,
    post_attention_normalization_weight: Tensor,
    convolution_pre_normalization_weight: Tensor,
    convolution_start: NativeGemma4ClippedLinear,
    depthwise_convolution_weight: Tensor,
    convolution_normalization_weight: Tensor,
    convolution_end: NativeGemma4ClippedLinear,
    feed_forward_two: NativeGemma4AudioFeedForward,
    output_normalization_weight: Tensor,
}

#[derive(Clone, Debug)]
pub struct NativeGemma4AudioEncoder {
    configuration: Gemma4AudioConfiguration,
    first_convolution_weight: Tensor,
    first_convolution_normalization_weight: Tensor,
    second_convolution_weight: Tensor,
    second_convolution_normalization_weight: Tensor,
    subsample_projection: NativeModule,
    blocks: Vec<NativeGemma4AudioBlock>,
    encoder_output: NativeModule,
    projector: NativeModule,
    semantic_state_digest_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Gemma4AudioProjection {
    pub embedding: Tensor,
    pub tokens: usize,
}

impl GemmaPreparedAudio {
    pub fn log_mel(&self) -> &Tensor {
        &self.log_mel
    }

    pub fn frame_mask(&self) -> &Tensor {
        &self.frame_mask
    }

    pub const fn marker_tokens(&self) -> usize {
        self.marker_tokens
    }

    pub const fn original_sample_rate(&self) -> u32 {
        self.original_sample_rate
    }

    pub const fn original_samples(&self) -> usize {
        self.original_samples
    }

    pub const fn resampled_samples(&self) -> usize {
        self.resampled_samples
    }
}

impl GemmaPreparedVisual {
    pub fn image(&self) -> &ImageTensor {
        &self.image
    }

    pub const fn kind(&self) -> GemmaPreparedVisualKind {
        self.kind
    }

    pub const fn maximum_soft_tokens(&self) -> usize {
        self.maximum_soft_tokens
    }

    pub const fn source_frame_index(&self) -> usize {
        self.source_frame_index
    }

    pub const fn timestamp_seconds(&self) -> Option<usize> {
        self.timestamp_seconds
    }
}

impl Qwen3VlPreparedImage {
    pub fn patches(&self) -> &Tensor {
        &self.patches
    }

    pub const fn grid_thw(&self) -> [usize; 3] {
        self.grid_thw
    }

    pub const fn merged_tokens(&self) -> usize {
        self.merged_tokens
    }

    pub const fn family(&self) -> QwenVisionFamily {
        self.family
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QwenVisionFamily {
    Qwen3Vl4B,
    Qwen3Vl8B,
    Qwen35_08B,
    Qwen35_2B,
    Qwen35_4B,
    Qwen35_9B,
    Qwen35_27B,
}

impl QwenVisionFamily {
    pub const fn image_pad_token(self) -> i64 {
        match self {
            Self::Qwen3Vl4B | Self::Qwen3Vl8B => QWEN3VL_IMAGE_PAD_TOKEN,
            Self::Qwen35_08B
            | Self::Qwen35_2B
            | Self::Qwen35_4B
            | Self::Qwen35_9B
            | Self::Qwen35_27B => QWEN35_IMAGE_PAD_TOKEN,
        }
    }

    pub const fn normalization(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Self::Qwen3Vl4B | Self::Qwen3Vl8B => ([0.5; 3], [0.5; 3]),
            Self::Qwen35_08B
            | Self::Qwen35_2B
            | Self::Qwen35_4B
            | Self::Qwen35_9B
            | Self::Qwen35_27B => (QWEN35_IMAGE_MEAN, QWEN35_IMAGE_STANDARD_DEVIATION),
        }
    }

    pub const fn deepstack_layers(self) -> &'static [usize] {
        match self {
            Self::Qwen3Vl4B => &QWEN3VL_4B_DEEPSTACK_LAYERS,
            Self::Qwen3Vl8B => &QWEN3VL_8B_DEEPSTACK_LAYERS,
            _ => &[],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenVisionConfiguration {
    pub family: QwenVisionFamily,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_count: usize,
    pub attention_heads: usize,
    pub output_hidden_size: usize,
    pub position_embeddings: usize,
    pub spatial_merge_size: usize,
    pub source_exact_profile: bool,
}

impl QwenVisionConfiguration {
    pub fn source(family: QwenVisionFamily) -> Self {
        let (hidden_size, intermediate_size, layer_count, attention_heads, output_hidden_size) =
            match family {
                QwenVisionFamily::Qwen3Vl4B => (1024, 4096, 24, 16, 2560),
                QwenVisionFamily::Qwen3Vl8B => (1152, 4304, 27, 16, 4096),
                QwenVisionFamily::Qwen35_08B => (768, 3072, 12, 12, 1024),
                QwenVisionFamily::Qwen35_2B => (1024, 4096, 24, 16, 2048),
                QwenVisionFamily::Qwen35_4B => (1024, 4096, 24, 16, 2560),
                QwenVisionFamily::Qwen35_9B => (1152, 4304, 27, 16, 4096),
                QwenVisionFamily::Qwen35_27B => (1152, 4304, 27, 16, 5120),
            };
        Self {
            family,
            hidden_size,
            intermediate_size,
            layer_count,
            attention_heads,
            output_hidden_size,
            position_embeddings: 2304,
            spatial_merge_size: 2,
            source_exact_profile: true,
        }
    }

    pub fn reduced_fixture(
        family: QwenVisionFamily,
        hidden_size: usize,
        intermediate_size: usize,
        layer_count: usize,
        attention_heads: usize,
        output_hidden_size: usize,
    ) -> Self {
        Self {
            family,
            hidden_size,
            intermediate_size,
            layer_count,
            attention_heads,
            output_hidden_size,
            position_embeddings: 2304,
            spatial_merge_size: 2,
            source_exact_profile: false,
        }
    }

    fn validate(&self) -> Result<(), MultimodalTextError> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.layer_count == 0
            || self.attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.attention_heads)
            || self.output_hidden_size == 0
            || self.position_embeddings != 2304
            || self.spatial_merge_size != 2
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen vision configuration dimensions are invalid",
            ));
        }
        if self.source_exact_profile && self != &Self::source(self.family) {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen production vision configuration does not match its closed source profile",
            ));
        }
        let deepstack = self.family.deepstack_layers();
        if !deepstack.is_empty() && deepstack.iter().any(|layer| *layer >= self.layer_count) {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen deepstack capture layers exceed the vision depth",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct QwenVisionBlockWeights {
    pub normalization_one_weight: Tensor,
    pub normalization_one_bias: Tensor,
    pub query_key_value_weight: Tensor,
    pub query_key_value_bias: Tensor,
    pub attention_output_weight: Tensor,
    pub attention_output_bias: Tensor,
    pub normalization_two_weight: Tensor,
    pub normalization_two_bias: Tensor,
    pub feed_forward_up_weight: Tensor,
    pub feed_forward_up_bias: Tensor,
    pub feed_forward_down_weight: Tensor,
    pub feed_forward_down_bias: Tensor,
}

#[derive(Clone, Debug)]
pub struct QwenVisionMergerWeights {
    pub normalization_weight: Tensor,
    pub normalization_bias: Tensor,
    pub first_weight: Tensor,
    pub first_bias: Tensor,
    pub second_weight: Tensor,
    pub second_bias: Tensor,
}

#[derive(Clone, Debug)]
pub struct QwenVisionWeights {
    pub patch_weight: Tensor,
    pub patch_bias: Tensor,
    pub position_embedding: Tensor,
    pub blocks: Vec<QwenVisionBlockWeights>,
    pub merger: QwenVisionMergerWeights,
    pub deepstack_mergers: Vec<QwenVisionMergerWeights>,
}

#[derive(Clone, Debug)]
pub struct Qwen25VisionBlockWeights {
    pub normalization_one_weight: Tensor,
    pub query_key_value_weight: Tensor,
    pub query_key_value_bias: Tensor,
    pub attention_output_weight: Tensor,
    pub attention_output_bias: Tensor,
    pub normalization_two_weight: Tensor,
    pub feed_forward_gate_weight: Tensor,
    pub feed_forward_gate_bias: Tensor,
    pub feed_forward_up_weight: Tensor,
    pub feed_forward_up_bias: Tensor,
    pub feed_forward_down_weight: Tensor,
    pub feed_forward_down_bias: Tensor,
}

#[derive(Clone, Debug)]
pub struct Qwen25VisionWeights {
    pub patch_weight: Tensor,
    pub blocks: Vec<Qwen25VisionBlockWeights>,
    pub merger_normalization_weight: Tensor,
    pub merger_first_weight: Tensor,
    pub merger_first_bias: Tensor,
    pub merger_second_weight: Tensor,
    pub merger_second_bias: Tensor,
}

#[derive(Clone, Debug)]
struct NativeQwen25VisionBlock {
    normalization_one_weight: Tensor,
    query_key_value: NativeModule,
    attention_output: NativeModule,
    normalization_two_weight: Tensor,
    feed_forward_gate: NativeModule,
    feed_forward_up: NativeModule,
    feed_forward_down: NativeModule,
}

#[derive(Clone, Debug)]
pub struct NativeQwen25VisionEncoder {
    patch_projection: NativeModule,
    blocks: Vec<NativeQwen25VisionBlock>,
    merger_normalization_weight: Tensor,
    merger_first: NativeModule,
    merger_second: NativeModule,
    stream: StreamId,
}

#[derive(Clone, Debug)]
pub struct Qwen25PreparedImage {
    patches: Tensor,
    grid_thw: [usize; 3],
    merged_tokens: usize,
}

#[derive(Clone)]
pub struct NativeQwen25Multimodal {
    tokenizer: Arc<NativePromptTokenizer>,
    decoder: Arc<NativeDecoderTextEncoder>,
    vision: Arc<NativeQwen25VisionEncoder>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeMultimodalOwnedAllocationKind {
    Resource,
    Tokenizer,
    Decoder,
    Vision,
    Audio,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMultimodalOwnedAllocation {
    kind: NativeMultimodalOwnedAllocationKind,
    address: usize,
    resident_bytes: u64,
}

impl NativeMultimodalOwnedAllocation {
    pub const fn kind(&self) -> NativeMultimodalOwnedAllocationKind {
        self.kind
    }

    pub const fn address(&self) -> usize {
        self.address
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug)]
struct NativeQwenVisionBlock {
    normalization_one: NativeModule,
    query_key_value: NativeModule,
    attention_output: NativeModule,
    normalization_two: NativeModule,
    feed_forward_up: NativeModule,
    feed_forward_activation: NativeModule,
    feed_forward_down: NativeModule,
}

#[derive(Clone, Debug)]
struct NativeQwenVisionMerger {
    normalization: NativeModule,
    first: NativeModule,
    activation: NativeModule,
    second: NativeModule,
    normalization_after_merge: bool,
}

#[derive(Clone, Debug)]
pub struct NativeQwenVisionEncoder {
    configuration: QwenVisionConfiguration,
    patch_projection: NativeModule,
    position_embedding: Tensor,
    blocks: Vec<NativeQwenVisionBlock>,
    merger: NativeQwenVisionMerger,
    deepstack_mergers: Vec<NativeQwenVisionMerger>,
}

#[derive(Clone, Debug)]
pub struct QwenVisionProjection {
    pub embedding: Tensor,
    pub deepstack: Vec<Tensor>,
    pub grid_thw: [usize; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3VlMarkerPlan {
    expanded_tokens: Vec<i64>,
    spans: Vec<MultimodalSpan>,
    visual_position_mask: Vec<bool>,
}

pub struct QwenMultimodalGenerationRequest<'a> {
    pub text: NativeTextGenerationRequest<'a>,
    pub prepared_images: &'a [Qwen3VlPreparedImage],
    pub transaction: &'a RngTransaction,
}

pub struct Qwen25ConditioningRequest<'a> {
    pub prompt: &'a str,
    pub prepared_images: &'a [Qwen25PreparedImage],
    pub custom_template: Option<&'a str>,
    pub capture_layer: Option<isize>,
}

#[derive(Clone, Debug)]
pub struct QwenConditioningOutput {
    hidden: Tensor,
    attention_mask: Option<Tensor>,
}

pub struct GemmaConditioningRequest<'a> {
    pub prompt: &'a str,
    pub prepared_visuals: &'a [GemmaPreparedVisual],
    pub prepared_audio: Option<&'a GemmaPreparedAudio>,
    pub use_default_template: bool,
    pub thinking: bool,
}

#[derive(Clone, Debug)]
pub struct GemmaConditioningOutput {
    hidden: Tensor,
    attention_mask: Tensor,
}

impl GemmaConditioningOutput {
    pub fn hidden(&self) -> &Tensor {
        &self.hidden
    }

    pub fn attention_mask(&self) -> &Tensor {
        &self.attention_mask
    }
}

impl QwenConditioningOutput {
    pub fn hidden(&self) -> &Tensor {
        &self.hidden
    }

    pub fn attention_mask(&self) -> Option<&Tensor> {
        self.attention_mask.as_ref()
    }
}

pub struct GemmaMultimodalGenerationRequest<'a> {
    pub text: NativeTextGenerationRequest<'a>,
    pub prepared_visuals: &'a [GemmaPreparedVisual],
    pub prepared_audio: Option<&'a GemmaPreparedAudio>,
    pub use_default_template: bool,
    pub thinking: bool,
    pub transaction: &'a RngTransaction,
}

impl Qwen3VlMarkerPlan {
    pub fn expanded_tokens(&self) -> &[i64] {
        &self.expanded_tokens
    }

    pub fn spans(&self) -> &[MultimodalSpan] {
        &self.spans
    }

    pub fn visual_position_mask(&self) -> &[bool] {
        &self.visual_position_mask
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sam3Prompt {
    pub text: String,
    pub maximum_detections: usize,
}

#[derive(Clone, Debug)]
pub struct Sam3EncodedCondition {
    pub condition: Tensor,
    pub attention_mask: Option<Tensor>,
    pub maximum_detections: usize,
}

#[derive(Clone, Debug)]
pub struct Sam3ConditionPack {
    pub main_condition: Tensor,
    pub first_pooled: Option<Tensor>,
    pub conditions: Vec<Sam3EncodedCondition>,
}

#[derive(Debug, Error)]
pub enum MultimodalTextError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    OperatorIndirection(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Indexing(#[from] IndexingMaskingPartOneError),
    #[error(transparent)]
    ShapeLayoutTwo(#[from] ShapeLayoutTransformPartTwoError),
    #[error(transparent)]
    ShapeLayoutThree(#[from] ShapeLayoutTransformPartThreeError),
    #[error(transparent)]
    Bidirectional(Box<BidirectionalTextError>),
    #[error(transparent)]
    Decoder(Box<DecoderTextError>),
    #[error(transparent)]
    Tokenizer(#[from] NativeTokenizerError),
    #[error(transparent)]
    ClipText(#[from] ClipTextError),
    #[error(transparent)]
    ClipVision(#[from] ClipVisionError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    NativeOps(#[from] NativeOpsError),
    #[error(transparent)]
    NativeTensor(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Spatial(#[from] SpatialFunctionalKernelError),
    #[error(transparent)]
    Spectral(#[from] SpectralTransformError),
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraPartOneError),
    #[error("multimodal text input is invalid: {0}")]
    InvalidInput(&'static str),
    #[error("multimodal text arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("SAM3 prompt limit {0:?} is not a finite nonnegative number")]
    InvalidPromptLimit(String),
    #[error("multimodal text execution was cancelled")]
    Cancelled,
}

impl From<sim_types::CancellationError> for MultimodalTextError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<BidirectionalTextError> for MultimodalTextError {
    fn from(error: BidirectionalTextError) -> Self {
        Self::Bidirectional(Box::new(error))
    }
}

impl From<DecoderTextError> for MultimodalTextError {
    fn from(error: DecoderTextError) -> Self {
        Self::Decoder(Box::new(error))
    }
}

pub fn run_bidirectional_text_owner(
    owner: &NativeT5TextEncoder,
    backend: &CpuBackend,
    request: BidirectionalTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<BidirectionalTextOutput, MultimodalTextError> {
    context.cancellation.check()?;
    Ok(owner.forward(backend, request, context)?)
}

pub fn run_decoder_text_owner(
    owner: &NativeDecoderTextEncoder,
    backend: &CpuBackend,
    request: DecoderTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<DecoderTextOutput, MultimodalTextError> {
    context.cancellation.check()?;
    Ok(owner.forward(backend, request, context)?)
}

pub fn run_clip_text_owner(
    owner: &NativeClipText,
    backend: &CpuBackend,
    request: ClipTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<ClipTextOutput, MultimodalTextError> {
    context.cancellation.check()?;
    Ok(owner.forward(backend, request, context)?)
}

pub fn run_clip_vision_owner(
    owner: &mut NativeClipVision,
    backend: &CpuBackend,
    image: &Tensor,
    crop: bool,
    intermediate: ClipVisionIntermediate,
    context: &ExecutionContext<'_>,
) -> Result<ClipVisionOutput, MultimodalTextError> {
    context.cancellation.check()?;
    let pixels = owner.preprocess(backend, image, crop, context)?;
    Ok(owner.forward(backend, &pixels, intermediate, context)?)
}

pub fn qwen2vl_mrope_position_ids(
    sequence_length: usize,
    spans: &[MultimodalSpan],
    cancellation: &sim_types::CancellationToken,
) -> Result<Option<MultimodalPositionIds>, MultimodalTextError> {
    cancellation.check()?;
    if spans.is_empty() {
        return Ok(None);
    }
    let mut temporal = sequential_positions(sequence_length)?;
    let mut height = clone_positions(&temporal)?;
    let mut width = clone_positions(&temporal)?;
    let mut previous_end = 0_usize;
    let mut offset = 0_i64;
    for (span_index, span) in spans.iter().enumerate() {
        if span_index.is_multiple_of(32) {
            cancellation.check()?;
        }
        let end = span
            .start
            .checked_add(span.size)
            .ok_or(MultimodalTextError::Overflow("image span end"))?;
        if span.size == 0
            || span.start < previous_end
            || end > sequence_length
            || span.grid_thw.contains(&0)
            || span.grid_thw[1] / 2 == 0
            || span.grid_thw[2] / 2 == 0
        {
            return Err(MultimodalTextError::InvalidInput(
                "image spans must be ordered, nonempty, in bounds, and have positive half grids",
            ));
        }
        let start = usize_to_i64(span.start, "image start")?;
        let size = usize_to_i64(span.size, "image size")?;
        let len_max = usize_to_i64(
            *span
                .grid_thw
                .iter()
                .max()
                .ok_or(MultimodalTextError::InvalidInput("image grid is empty"))?
                / 2,
            "image grid maximum",
        )?;
        let start_value = start
            .checked_add(offset)
            .ok_or(MultimodalTextError::Overflow("image position start"))?;
        let height_extent = span.grid_thw[1] / 2;
        let width_extent = span.grid_thw[2] / 2;
        let height_repeat = span
            .size
            .checked_add(height_extent - 1)
            .and_then(|value| value.checked_div(height_extent))
            .ok_or(MultimodalTextError::Overflow("height repeat"))?;
        for local in 0..span.size {
            let index = span.start + local;
            temporal[index] = start_value;
            height[index] = start_value
                .checked_add(usize_to_i64(local / height_repeat, "height position")?)
                .ok_or(MultimodalTextError::Overflow("height position"))?;
            width[index] = start_value
                .checked_add(usize_to_i64(local % width_extent, "width position")?)
                .ok_or(MultimodalTextError::Overflow("width position"))?;
        }
        offset = offset
            .checked_add(len_max)
            .and_then(|value| value.checked_sub(size))
            .ok_or(MultimodalTextError::Overflow("multimodal position offset"))?;
        let suffix_start = start
            .checked_add(len_max)
            .and_then(|value| value.checked_add(offset - (len_max - size)))
            .ok_or(MultimodalTextError::Overflow("suffix position"))?;
        for index in end..sequence_length {
            let delta = usize_to_i64(index - end, "suffix offset")?;
            let value = suffix_start
                .checked_add(delta)
                .ok_or(MultimodalTextError::Overflow("suffix position"))?;
            temporal[index] = value;
            height[index] = value;
            width[index] = value;
        }
        previous_end = end;
    }
    cancellation.check()?;
    Ok(Some(MultimodalPositionIds {
        temporal,
        height,
        width,
    }))
}

pub fn qwen3vl_target_dimensions(
    height: u64,
    width: u64,
) -> Result<(u64, u64), MultimodalTextError> {
    if height == 0 || width == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL images require nonzero dimensions",
        ));
    }
    let factor = u64::try_from(
        QWEN3VL_IMAGE_PATCH_SIZE
            .checked_mul(QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE)
            .ok_or(MultimodalTextError::Overflow("Qwen3-VL resize factor"))?,
    )
    .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL resize factor"))?;
    let mut target_height = round_to_factor(height, factor)?;
    let mut target_width = round_to_factor(width, factor)?;
    let rounded_pixels = target_height
        .checked_mul(target_width)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL rounded pixels"))?;
    let source_pixels = height
        .checked_mul(width)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL source pixels"))?;
    if rounded_pixels > QWEN3VL_IMAGE_MAXIMUM_PIXELS {
        let beta = ((source_pixels as f64) / (QWEN3VL_IMAGE_MAXIMUM_PIXELS as f64)).sqrt();
        target_height = floor_scaled_to_factor(height, beta, factor)?;
        target_width = floor_scaled_to_factor(width, beta, factor)?;
    } else if rounded_pixels < QWEN3VL_IMAGE_MINIMUM_PIXELS {
        let beta = ((QWEN3VL_IMAGE_MINIMUM_PIXELS as f64) / (source_pixels as f64)).sqrt();
        target_height = ceil_scaled_to_factor(height, beta, factor)?;
        target_width = ceil_scaled_to_factor(width, beta, factor)?;
    }
    if target_height == 0
        || target_width == 0
        || !target_height.is_multiple_of(factor)
        || !target_width.is_multiple_of(factor)
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL target dimensions are not positive factor-aligned values",
        ));
    }
    if target_height
        .checked_mul(target_width)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL target pixels"))?
        > QWEN3VL_IMAGE_MAXIMUM_PIXELS
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL target dimensions exceed the native bounded pixel limit",
        ));
    }
    Ok((target_height, target_width))
}

pub fn gemma3_target_dimensions(
    height: u64,
    width: u64,
) -> Result<(u64, u64), MultimodalTextError> {
    let source_pixels = height
        .checked_mul(width)
        .ok_or(MultimodalTextError::Overflow("Gemma3 source pixels"))?;
    if source_pixels == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 images require nonzero dimensions",
        ));
    }
    let scale = ((GEMMA3_IMAGE_AREA_PIXELS as f64) / (source_pixels as f64)).sqrt();
    let target_height = checked_f64_to_u64(
        ((height as f64) * scale).round_ties_even().max(1.0),
        "Gemma3 target height",
    )?;
    let target_width = checked_f64_to_u64(
        ((width as f64) * scale).round_ties_even().max(1.0),
        "Gemma3 target width",
    )?;
    let target_pixels = target_height
        .checked_mul(target_width)
        .ok_or(MultimodalTextError::Overflow("Gemma3 target pixels"))?;
    if target_pixels > GEMMA3_MAXIMUM_PREPARED_PIXELS {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 target dimensions exceed the native bounded pixel limit",
        ));
    }
    Ok((target_height, target_width))
}

pub fn prepare_gemma3_image(
    backend: &CpuBackend,
    image: &ImageTensor,
    context: &ExecutionContext<'_>,
) -> Result<GemmaPreparedVisual, MultimodalTextError> {
    context.cancellation.check()?;
    let (batch, height, width, channels) = image.dimensions()?;
    if batch == 0 || channels < 3 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 requires a nonempty RGB or RGBA IMAGE batch",
        ));
    }
    let image = project_rgb_channels(backend, image, context)?;
    let (target_height, target_width) = gemma3_target_dimensions(height, width)?;
    let image = image.resize(
        target_width,
        target_height,
        ResizeMode::Area,
        ResizeCrop::Disabled,
        backend,
        context,
    )?;
    context.cancellation.check()?;
    Ok(GemmaPreparedVisual {
        image,
        kind: GemmaPreparedVisualKind::Gemma3Image,
        maximum_soft_tokens: GEMMA3_IMAGE_SOFT_TOKENS,
        source_frame_index: 0,
        timestamp_seconds: None,
    })
}

impl NativeGemma3VisionProjector {
    pub fn new(
        configuration: Gemma3VisionConfiguration,
        vision: Arc<NativeClipVision>,
        normalization_weight: Tensor,
        input_projection_weight: Tensor,
    ) -> Result<Self, MultimodalTextError> {
        configuration.validate()?;
        validate_gemma3_vision_owner(&configuration, &vision)?;
        let stream = normalization_weight.descriptor().stream();
        gemma3_require_parameter_shape(
            &normalization_weight,
            &[configuration.vision_hidden_size],
            stream,
        )?;
        gemma3_require_parameter_shape(
            &input_projection_weight,
            &[
                configuration.vision_hidden_size,
                configuration.output_hidden_size,
            ],
            stream,
        )?;
        let mut owner = Self {
            configuration,
            vision,
            normalization_weight,
            input_projection_weight,
            semantic_state_digest_sha256: String::new(),
        };
        owner.semantic_state_digest_sha256 =
            owner.project_semantic_state_digest(&sim_types::CancellationToken::default())?;
        owner.validate(&sim_types::CancellationToken::default())?;
        Ok(owner)
    }

    pub fn configuration(&self) -> &Gemma3VisionConfiguration {
        &self.configuration
    }

    pub fn vision(&self) -> &Arc<NativeClipVision> {
        &self.vision
    }

    pub fn execution_stream(&self) -> StreamId {
        self.normalization_weight.descriptor().stream()
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        self.configuration.validate()?;
        validate_gemma3_vision_owner(&self.configuration, &self.vision)?;
        self.vision.validate(cancellation)?;
        let stream = self.normalization_weight.descriptor().stream();
        gemma3_require_parameter_shape(
            &self.normalization_weight,
            &[self.configuration.vision_hidden_size],
            stream,
        )?;
        gemma3_require_parameter_shape(
            &self.input_projection_weight,
            &[
                self.configuration.vision_hidden_size,
                self.configuration.output_hidden_size,
            ],
            stream,
        )?;
        if self.semantic_state_digest_sha256 != self.project_semantic_state_digest(cancellation)? {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma3 vision semantic identity changed after admission",
            ));
        }
        self.resident_bytes()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, MultimodalTextError> {
        let mut allocations = Vec::new();
        for allocation in self.vision.resident_parts()?.tensor_allocations() {
            insert_gemma3_resident_allocation(
                &mut allocations,
                allocation.storage_id(),
                allocation.resident_bytes(),
            )?;
        }
        for tensor in [&self.normalization_weight, &self.input_projection_weight] {
            insert_gemma3_resident_allocation(
                &mut allocations,
                tensor.storage_id(),
                tensor.storage_byte_len(),
            )?;
        }
        Ok(allocations)
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        let vision_owned = self.vision.resident_parts()?.owned_bytes();
        u64::try_from(mem::size_of::<Self>())
            .ok()
            .and_then(|bytes| bytes.checked_add(vision_owned))
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(self.semantic_state_digest_sha256.capacity()).ok()?)
            })
            .ok_or(MultimodalTextError::Overflow(
                "Gemma3 vision owned residency",
            ))
    }

    pub fn resident_bytes(&self) -> Result<u64, MultimodalTextError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |bytes, (_, allocation)| {
                bytes
                    .checked_add(allocation)
                    .ok_or(MultimodalTextError::Overflow(
                        "Gemma3 vision total residency",
                    ))
            },
        )
    }

    pub fn project(
        &self,
        backend: &CpuBackend,
        prepared: &GemmaPreparedVisual,
        context: &ExecutionContext<'_>,
    ) -> Result<Gemma3VisionProjection, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        if prepared.kind() != GemmaPreparedVisualKind::Gemma3Image
            || prepared.maximum_soft_tokens() != GEMMA3_IMAGE_SOFT_TOKENS
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma3 vision requires canonical image preparation and token count",
            ));
        }
        let mut session = self.vision.execution_session(context.cancellation)?;
        let pixels = self
            .vision
            .preprocess(backend, prepared.image().tensor(), true, context)?;
        let output = session.forward(backend, &pixels, ClipVisionIntermediate::None, context)?;
        let hidden = &output.last_hidden_state;
        let shape = hidden.descriptor().shape();
        let patch_count = self
            .configuration
            .patches_per_side
            .checked_mul(self.configuration.patches_per_side)
            .ok_or(MultimodalTextError::Overflow("Gemma3 vision patches"))?;
        if shape.len() != 3
            || shape[0] == 0
            || shape[1] != usize_to_u64(patch_count, "Gemma3 patch count")?
            || shape[2]
                != usize_to_u64(
                    self.configuration.vision_hidden_size,
                    "Gemma3 vision hidden size",
                )?
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma3 CLIP vision output has the wrong shape",
            ));
        }
        let batch = u64_to_usize(shape[0], "Gemma3 vision batch")?;
        let hidden_values = tensor_to_f32(backend, hidden, context)?;
        let normalization_weight = tensor_to_f32(backend, &self.normalization_weight, context)?;
        let pooled = gemma3_pool_and_normalize(
            backend,
            &hidden_values,
            batch,
            &self.configuration,
            &normalization_weight,
            context,
        )?;
        let tokens_per_image = self
            .configuration
            .tokens_per_side
            .checked_mul(self.configuration.tokens_per_side)
            .ok_or(MultimodalTextError::Overflow("Gemma3 vision tokens"))?;
        let normalized_shape = [
            usize_to_u64(batch, "Gemma3 vision batch")?,
            usize_to_u64(tokens_per_image, "Gemma3 vision tokens")?,
            usize_to_u64(
                self.configuration.vision_hidden_size,
                "Gemma3 vision hidden size",
            )?,
        ];
        let normalized = tensor_from_f32(backend, &normalized_shape, &pooled, context)?;
        let embedding = matmul_with_context_exact_native(
            backend,
            &normalized,
            &self.input_projection_weight,
            context,
        )?;
        context.check()?;
        Ok(Gemma3VisionProjection {
            embedding,
            tokens_per_image,
        })
    }

    fn project_semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        cancellation.check()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.gemma3-vision-projector.v1");
        hasher.update(LLAMA_SOURCE_SHA256.as_bytes());
        hasher.update(GEMMA3_MULTIMODAL_SOURCE_SHA256.as_bytes());
        hasher.update(canonical_multimodal_bytes(&self.configuration)?);
        hasher.update(self.vision.semantic_digest_sha256().as_bytes());
        for tensor in [&self.normalization_weight, &self.input_projection_weight] {
            cancellation.check()?;
            hasher.update(tensor.contiguous_bytes()?);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn validate_gemma3_vision_owner(
    configuration: &Gemma3VisionConfiguration,
    vision: &NativeClipVision,
) -> Result<(), MultimodalTextError> {
    let vision_configuration = vision.configuration();
    if vision.is_training()
        || vision_configuration.dtype != DType::F32
        || vision_configuration.device != DeviceId::CPU
        || vision_configuration.model_type != ClipVisionModelType::Siglip
        || vision_configuration.hidden_size != configuration.vision_hidden_size
        || vision_configuration.image_size
            != configuration
                .patches_per_side
                .checked_mul(vision_configuration.patch_size)
                .ok_or(MultimodalTextError::Overflow("Gemma3 vision image size"))?
        || vision_configuration.projection_dimension.is_some()
        || vision_configuration.llava_projection_dimension.is_some()
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 projector requires an evaluation-only retained SigLIP vision owner",
        ));
    }
    if configuration.source_profile.is_some()
        && vision_configuration
            != &(ClipVisionConfiguration {
                model_type: ClipVisionModelType::Siglip,
                dtype: DType::F32,
                device: DeviceId::CPU,
                hidden_size: 1152,
                intermediate_size: 4304,
                attention_heads: 16,
                layer_count: 27,
                image_size: 896,
                patch_size: 14,
                num_channels: 3,
                max_num_patches: 0,
                activation: ClipVisionActivation::GeluTanh,
                projection_dimension: None,
                llava_projection_dimension: None,
            })
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 production projection requires the exact retained 896-square SigLIP graph",
        ));
    }
    Ok(())
}

fn gemma3_pool_and_normalize(
    backend: &CpuBackend,
    hidden: &[f32],
    batch: usize,
    configuration: &Gemma3VisionConfiguration,
    normalization_weight: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, MultimodalTextError> {
    let hidden_size = configuration.vision_hidden_size;
    let patches_per_side = configuration.patches_per_side;
    let tokens_per_side = configuration.tokens_per_side;
    let kernel = patches_per_side / tokens_per_side;
    let patch_count = patches_per_side
        .checked_mul(patches_per_side)
        .ok_or(MultimodalTextError::Overflow("Gemma3 patch count"))?;
    let tokens = tokens_per_side
        .checked_mul(tokens_per_side)
        .ok_or(MultimodalTextError::Overflow("Gemma3 token count"))?;
    if hidden.len()
        != batch
            .checked_mul(patch_count)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or(MultimodalTextError::Overflow("Gemma3 hidden values"))?
        || normalization_weight.len() != hidden_size
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 vision projection input storage is malformed",
        ));
    }
    let mut output = backend.workspace_vec(
        context,
        batch
            .checked_mul(tokens)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or(MultimodalTextError::Overflow("Gemma3 projected tokens"))?,
    )?;
    let divisor = (kernel * kernel) as f32;
    let epsilon = f32::from_bits(configuration.normalization_epsilon_bits);
    let weight_offset = f32::from_bits(configuration.normalization_weight_offset_bits);
    for batch_index in 0..batch {
        for token_y in 0..tokens_per_side {
            for token_x in 0..tokens_per_side {
                context.check()?;
                let output_start = output.len();
                for channel in 0..hidden_size {
                    let mut sum = 0.0_f32;
                    for kernel_y in 0..kernel {
                        for kernel_x in 0..kernel {
                            let patch_y = token_y * kernel + kernel_y;
                            let patch_x = token_x * kernel + kernel_x;
                            let index = ((batch_index * patch_count
                                + patch_y * patches_per_side
                                + patch_x)
                                * hidden_size)
                                + channel;
                            sum += hidden.get(index).copied().ok_or(
                                MultimodalTextError::InvalidInput(
                                    "Gemma3 vision patch storage is incomplete",
                                ),
                            )?;
                        }
                    }
                    output.try_push(sum / divisor)?;
                }
                let output_end = output_start
                    .checked_add(hidden_size)
                    .ok_or(MultimodalTextError::Overflow("Gemma3 pooled token storage"))?;
                let token = output.get_mut(output_start..output_end).ok_or(
                    MultimodalTextError::InvalidInput("Gemma3 pooled token storage is incomplete"),
                )?;
                let mean_square = token.iter().try_fold(0.0_f32, |sum, value| {
                    let square = value * value;
                    if square.is_finite() {
                        Ok(sum + square)
                    } else {
                        Err(MultimodalTextError::InvalidInput(
                            "Gemma3 pooled vision state is nonfinite",
                        ))
                    }
                })? / hidden_size as f32;
                let inverse = (mean_square + epsilon).sqrt().recip();
                for (value, weight) in token.iter_mut().zip(normalization_weight) {
                    *value *= inverse * (*weight + weight_offset);
                    if !value.is_finite() {
                        return Err(MultimodalTextError::InvalidInput(
                            "Gemma3 normalized vision state is nonfinite",
                        ));
                    }
                }
            }
        }
    }
    Ok(output)
}

fn gemma3_require_parameter_shape(
    tensor: &Tensor,
    expected: &[usize],
    stream: StreamId,
) -> Result<(), MultimodalTextError> {
    let expected = expected
        .iter()
        .map(|value| usize_to_u64(*value, "Gemma3 parameter shape"))
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.descriptor().shape() != expected
        || tensor.descriptor().dtype() != DType::F32
        || tensor.descriptor().device() != DeviceId::CPU
        || tensor.descriptor().stream() != stream
        || tensor
            .contiguous_bytes()?
            .chunks_exact(mem::size_of::<f32>())
            .any(|chunk| {
                let Ok(bytes) = <[u8; 4]>::try_from(chunk) else {
                    return true;
                };
                !f32::from_ne_bytes(bytes).is_finite()
            })
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 vision parameter shape, target, or values are invalid",
        ));
    }
    Ok(())
}

fn insert_gemma3_resident_allocation(
    allocations: &mut Vec<(sim_tensor::StorageId, u64)>,
    storage_id: sim_tensor::StorageId,
    resident_bytes: u64,
) -> Result<(), MultimodalTextError> {
    if let Some((_, existing_bytes)) = allocations
        .iter()
        .find(|(existing_id, _)| *existing_id == storage_id)
    {
        if *existing_bytes != resident_bytes {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma3 aliased tensor storage changed byte length",
            ));
        }
    } else {
        allocations
            .try_reserve(1)
            .map_err(|_| MultimodalTextError::Overflow("Gemma3 residency allocations"))?;
        allocations.push((storage_id, resident_bytes));
    }
    Ok(())
}

pub fn gemma4_target_dimensions(
    height: u64,
    width: u64,
    maximum_soft_tokens: usize,
) -> Result<(u64, u64), MultimodalTextError> {
    let source_pixels = height
        .checked_mul(width)
        .ok_or(MultimodalTextError::Overflow("Gemma4 source pixels"))?;
    if source_pixels == 0 || maximum_soft_tokens == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 visual dimensions and soft-token budget must be nonzero",
        ));
    }
    let maximum_soft_tokens = u64::try_from(maximum_soft_tokens)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 soft-token budget"))?;
    let side_multiple = GEMMA4_IMAGE_PATCH_SIZE
        .checked_mul(GEMMA4_IMAGE_POOLING_SIZE)
        .ok_or(MultimodalTextError::Overflow("Gemma4 side multiple"))?;
    let target_pixels = maximum_soft_tokens
        .checked_mul(GEMMA4_IMAGE_POOLING_SIZE)
        .and_then(|value| value.checked_mul(GEMMA4_IMAGE_POOLING_SIZE))
        .and_then(|value| value.checked_mul(GEMMA4_IMAGE_PATCH_SIZE))
        .and_then(|value| value.checked_mul(GEMMA4_IMAGE_PATCH_SIZE))
        .ok_or(MultimodalTextError::Overflow("Gemma4 target pixels"))?;
    let scale = ((target_pixels as f64) / (source_pixels as f64)).sqrt();
    let target_height = checked_f64_to_u64(
        (((height as f64) * scale) / (side_multiple as f64)).floor(),
        "Gemma4 target height units",
    )?
    .checked_mul(side_multiple)
    .ok_or(MultimodalTextError::Overflow("Gemma4 target height"))?
    .max(side_multiple);
    let target_width = checked_f64_to_u64(
        (((width as f64) * scale) / (side_multiple as f64)).floor(),
        "Gemma4 target width units",
    )?
    .checked_mul(side_multiple)
    .ok_or(MultimodalTextError::Overflow("Gemma4 target width"))?
    .max(side_multiple);
    let prepared_pixels = target_height
        .checked_mul(target_width)
        .ok_or(MultimodalTextError::Overflow("Gemma4 target dimensions"))?;
    let maximum_prepared_pixels =
        target_pixels
            .checked_mul(4)
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 maximum prepared pixels",
            ))?;
    if prepared_pixels > maximum_prepared_pixels {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 target dimensions exceed the native bounded pixel limit",
        ));
    }
    Ok((target_height, target_width))
}

pub fn prepare_gemma4_visuals(
    backend: &CpuBackend,
    image: Option<&ImageTensor>,
    video: Option<&ImageTensor>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<GemmaPreparedVisual>, MultimodalTextError> {
    context.cancellation.check()?;
    let (source, kind, maximum_soft_tokens, frame_step) = match (image, video) {
        (_, Some(video)) => (
            video,
            GemmaPreparedVisualKind::Gemma4VideoFrame,
            GEMMA4_VIDEO_SOFT_TOKENS,
            GEMMA4_VIDEO_SOURCE_FPS,
        ),
        (Some(image), None) => (
            image,
            GemmaPreparedVisualKind::Gemma4Image,
            GEMMA4_IMAGE_SOFT_TOKENS,
            1,
        ),
        (None, None) => return Ok(Vec::new()),
    };
    let (batch, height, width, channels) = source.dimensions()?;
    if batch == 0 || channels < 3 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 requires a nonempty RGB or RGBA IMAGE batch",
        ));
    }
    let source = project_rgb_channels(backend, source, context)?;
    let (target_height, target_width) =
        gemma4_target_dimensions(height, width, maximum_soft_tokens)?;
    let batch = u64_to_usize(batch, "Gemma4 visual batch")?;
    let frame_elements = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(MultimodalTextError::Overflow("Gemma4 frame elements"))?;
    let source_values = source.as_f32_slice()?;
    let selected_count = batch
        .checked_add(frame_step - 1)
        .ok_or(MultimodalTextError::Overflow("Gemma4 selected frame count"))?
        / frame_step;
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(selected_count)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 prepared frames"))?;
    for (timestamp_seconds, source_frame_index) in (0..batch).step_by(frame_step).enumerate() {
        context.cancellation.check()?;
        let start = source_frame_index
            .checked_mul(frame_elements)
            .ok_or(MultimodalTextError::Overflow("Gemma4 frame offset"))?;
        let end = start
            .checked_add(frame_elements)
            .ok_or(MultimodalTextError::Overflow("Gemma4 frame end"))?;
        let values = source_values
            .get(start..end)
            .ok_or(MultimodalTextError::InvalidInput(
                "Gemma4 frame storage is incomplete",
            ))?;
        let image = gemma4_quantized_bicubic_resize(
            backend,
            values,
            height,
            width,
            target_height,
            target_width,
            context,
        )?;
        prepared.push(GemmaPreparedVisual {
            image,
            kind,
            maximum_soft_tokens,
            source_frame_index,
            timestamp_seconds: (kind == GemmaPreparedVisualKind::Gemma4VideoFrame)
                .then_some(timestamp_seconds),
        });
    }
    context.cancellation.check()?;
    Ok(prepared)
}

pub fn prepare_gemma4_audio(
    backend: &CpuBackend,
    waveform: &Tensor,
    sample_rate: u32,
    context: &ExecutionContext<'_>,
) -> Result<GemmaPreparedAudio, MultimodalTextError> {
    context.cancellation.check()?;
    if !(GEMMA4_AUDIO_MINIMUM_SAMPLE_RATE..=GEMMA4_AUDIO_MAXIMUM_SAMPLE_RATE).contains(&sample_rate)
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio sample rate is outside the checked AUDIO range",
        ));
    }
    let descriptor = waveform.descriptor();
    let shape = descriptor.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio requires contiguous [1, channels, samples] AUDIO",
        ));
    }
    if descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != context.stream
        || !descriptor.is_contiguous()?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio must be contiguous CPU F32 on the execution stream",
        ));
    }
    let channels = u64_to_usize(shape[1], "Gemma4 audio channels")?;
    let original_samples = u64_to_usize(shape[2], "Gemma4 audio samples")?;
    let source = tensor_to_f32(backend, waveform, context)?;
    let expected_source_values = channels
        .checked_mul(original_samples)
        .ok_or(MultimodalTextError::Overflow("Gemma4 audio source values"))?;
    if source.len() != expected_source_values {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio storage does not match its descriptor",
        ));
    }
    let mut mono = backend.workspace_vec(context, original_samples)?;
    for sample in 0..original_samples {
        check_gemma_audio_periodically(sample, context)?;
        let mut sum = 0.0_f64;
        for channel in 0..channels {
            let index = channel
                .checked_mul(original_samples)
                .and_then(|value| value.checked_add(sample))
                .ok_or(MultimodalTextError::Overflow("Gemma4 mono sample index"))?;
            let value = *source.get(index).ok_or(MultimodalTextError::InvalidInput(
                "Gemma4 audio channel storage is incomplete",
            ))?;
            if !value.is_finite() {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma4 audio samples must be finite",
                ));
            }
            sum += f64::from(value);
        }
        mono.try_push((sum / channels as f64) as f32)?;
    }
    let resampled = gemma4_resample_polyphase(backend, &mono, sample_rate, context)?;
    let resampled_samples = resampled.len();
    let marker_tokens = gemma4_audio_marker_tokens(original_samples, sample_rate)?;
    let padded_samples = resampled_samples
        .checked_add(GEMMA4_AUDIO_PADDING_MULTIPLE - 1)
        .ok_or(MultimodalTextError::Overflow("Gemma4 padded audio samples"))?
        / GEMMA4_AUDIO_PADDING_MULTIPLE
        * GEMMA4_AUDIO_PADDING_MULTIPLE;
    let semicausal_samples = padded_samples.checked_add(GEMMA4_AUDIO_FRAME_STEP).ok_or(
        MultimodalTextError::Overflow("Gemma4 semicausal audio samples"),
    )?;
    let required_frame_samples = GEMMA4_AUDIO_FRAME_LENGTH
        .checked_add(1)
        .ok_or(MultimodalTextError::Overflow("Gemma4 frame samples"))?;
    let frame_count = if semicausal_samples < required_frame_samples {
        0
    } else {
        (semicausal_samples - required_frame_samples) / GEMMA4_AUDIO_FRAME_STEP + 1
    };
    let (log_mel, frame_mask) =
        gemma4_log_mel(backend, &resampled, resampled_samples, frame_count, context)?;
    context.cancellation.check()?;
    Ok(GemmaPreparedAudio {
        log_mel,
        frame_mask,
        marker_tokens,
        original_sample_rate: sample_rate,
        original_samples,
        resampled_samples,
    })
}

pub fn gemma4_audio_marker_tokens(
    original_samples: usize,
    sample_rate: u32,
) -> Result<usize, MultimodalTextError> {
    if !(GEMMA4_AUDIO_MINIMUM_SAMPLE_RATE..=GEMMA4_AUDIO_MAXIMUM_SAMPLE_RATE).contains(&sample_rate)
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio sample rate is outside the checked AUDIO range",
        ));
    }
    let projected_samples = if sample_rate == GEMMA4_AUDIO_SAMPLE_RATE {
        original_samples
    } else {
        original_samples
            .checked_mul(GEMMA4_AUDIO_SAMPLE_RATE as usize)
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 projected audio samples",
            ))?
            / sample_rate as usize
    };
    let projected_with_padding = projected_samples
        .checked_add(GEMMA4_AUDIO_FRAME_STEP)
        .ok_or(MultimodalTextError::Overflow(
            "Gemma4 projected audio frame count",
        ))?;
    let frame_count = if projected_with_padding < GEMMA4_AUDIO_FRAME_LENGTH + 1 {
        0
    } else {
        (projected_with_padding - (GEMMA4_AUDIO_FRAME_LENGTH + 1)) / GEMMA4_AUDIO_FRAME_STEP + 1
    };
    let once = frame_count.div_ceil(2);
    Ok(once.div_ceil(2).min(GEMMA4_AUDIO_MAXIMUM_TOKENS))
}

fn gemma4_resample_polyphase(
    backend: &CpuBackend,
    mono: &[f32],
    sample_rate: u32,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, MultimodalTextError> {
    if sample_rate == GEMMA4_AUDIO_SAMPLE_RATE {
        let mut output = backend.workspace_vec(context, mono.len())?;
        for (index, value) in mono.iter().copied().enumerate() {
            check_gemma_audio_periodically(index, context)?;
            output.try_push(value)?;
        }
        return Ok(output);
    }
    let divisor = greatest_common_divisor(sample_rate, GEMMA4_AUDIO_SAMPLE_RATE);
    let up = usize::try_from(GEMMA4_AUDIO_SAMPLE_RATE / divisor)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample up factor"))?;
    let down = usize::try_from(sample_rate / divisor)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample down factor"))?;
    let ratio_limit = up.max(down);
    let half_length = GEMMA4_AUDIO_FILTER_HALF_WIDTH
        .checked_mul(ratio_limit)
        .ok_or(MultimodalTextError::Overflow(
            "Gemma4 resample filter half length",
        ))?;
    let filter_length = half_length
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(MultimodalTextError::Overflow(
            "Gemma4 resample filter length",
        ))?;
    if filter_length > GEMMA4_AUDIO_MAXIMUM_FILTER_TAPS {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 resample ratio exceeds the native bounded filter limit",
        ));
    }
    let output_length = mono
        .len()
        .checked_mul(up)
        .and_then(|value| value.checked_add(down - 1))
        .ok_or(MultimodalTextError::Overflow("Gemma4 resampled length"))?
        / down;
    let taps_per_output = filter_length
        .checked_add(up - 1)
        .ok_or(MultimodalTextError::Overflow("Gemma4 resample work"))?
        / up;
    let multiply_adds = output_length
        .checked_mul(taps_per_output)
        .ok_or(MultimodalTextError::Overflow("Gemma4 resample work"))?;
    if multiply_adds > GEMMA4_AUDIO_MAXIMUM_RESAMPLE_MULTIPLY_ADDS {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio exceeds the native bounded resample work limit",
        ));
    }
    let cutoff = 0.96 / ratio_limit as f64;
    let kaiser_denominator = gemma4_bessel_i0(GEMMA4_AUDIO_KAISER_BETA);
    let mut normalization = 0.0_f64;
    for index in 0..filter_length {
        check_gemma_audio_periodically(index, context)?;
        normalization += gemma4_firwin_value(index, half_length, cutoff, kaiser_denominator);
    }
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 resample filter normalization is invalid",
        ));
    }
    let pre_padding = down - half_length % down;
    let pre_remove = (half_length + pre_padding) / down;
    let mut output = backend.workspace_vec(context, output_length)?;
    let up_i128 = i128::try_from(up)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample up factor"))?;
    let down_i128 = i128::try_from(down)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample down factor"))?;
    let pre_padding_i128 = i128::try_from(pre_padding)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample pre-padding"))?;
    let filter_last_i128 = i128::try_from(filter_length - 1)
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample filter length"))?;
    for output_index in 0..output_length {
        check_gemma_audio_periodically(output_index, context)?;
        let projected_output_index =
            output_index
                .checked_add(pre_remove)
                .ok_or(MultimodalTextError::Overflow(
                    "Gemma4 resample output index",
                ))?;
        let raw_index = i128::try_from(projected_output_index)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample output index"))?
            .checked_mul(down_i128)
            .and_then(|value| value.checked_sub(pre_padding_i128))
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 resample convolution index",
            ))?;
        let first_input = div_ceil_i128(raw_index - filter_last_i128, up_i128).max(0);
        let last_input = raw_index.div_euclid(up_i128).min(
            i128::try_from(mono.len() - 1)
                .map_err(|_| MultimodalTextError::Overflow("Gemma4 audio samples"))?,
        );
        let mut sum = 0.0_f64;
        if first_input <= last_input {
            for input_index in first_input..=last_input {
                let filter_index = raw_index
                    .checked_sub(input_index.checked_mul(up_i128).ok_or(
                        MultimodalTextError::Overflow("Gemma4 resample filter index"),
                    )?)
                    .ok_or(MultimodalTextError::Overflow(
                        "Gemma4 resample filter index",
                    ))?;
                let filter_index = usize::try_from(filter_index)
                    .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample filter index"))?;
                let input_index = usize::try_from(input_index)
                    .map_err(|_| MultimodalTextError::Overflow("Gemma4 resample input index"))?;
                let input = f64::from(*mono.get(input_index).ok_or(
                    MultimodalTextError::InvalidInput("Gemma4 resample input is incomplete"),
                )?);
                let coefficient =
                    gemma4_firwin_value(filter_index, half_length, cutoff, kaiser_denominator)
                        / normalization
                        * up as f64;
                sum += input * coefficient;
            }
        }
        output.try_push(sum as f32)?;
    }
    Ok(output)
}

fn gemma4_log_mel(
    backend: &CpuBackend,
    audio: &[f32],
    real_samples: usize,
    frame_count: usize,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), MultimodalTextError> {
    let frame_values = frame_count
        .checked_mul(GEMMA4_AUDIO_FFT_LENGTH)
        .ok_or(MultimodalTextError::Overflow("Gemma4 FFT frame values"))?;
    let mut framed = backend.workspace_vec(context, frame_values)?;
    let mut mask = backend.workspace_vec::<u8>(context, frame_count)?;
    for frame in 0..frame_count {
        check_gemma_audio_periodically(frame, context)?;
        let frame_start = frame
            .checked_mul(GEMMA4_AUDIO_FRAME_STEP)
            .ok_or(MultimodalTextError::Overflow("Gemma4 frame start"))?;
        let mask_index = frame_start
            .checked_add(GEMMA4_AUDIO_FRAME_LENGTH)
            .ok_or(MultimodalTextError::Overflow("Gemma4 frame mask index"))?;
        let source_mask_index = mask_index.checked_sub(GEMMA4_AUDIO_FRAME_STEP);
        mask.try_push(u8::from(
            source_mask_index.is_some_and(|index| index < real_samples),
        ))?;
        for offset in 0..GEMMA4_AUDIO_FFT_LENGTH {
            let value = if offset < GEMMA4_AUDIO_FRAME_LENGTH {
                let padded_index = frame_start
                    .checked_add(offset)
                    .ok_or(MultimodalTextError::Overflow("Gemma4 frame sample"))?;
                let source_index = padded_index.checked_sub(GEMMA4_AUDIO_FRAME_STEP);
                let sample = source_index
                    .and_then(|index| audio.get(index))
                    .copied()
                    .unwrap_or(0.0);
                let phase =
                    std::f64::consts::TAU * offset as f64 / GEMMA4_AUDIO_FRAME_LENGTH as f64;
                sample * (0.5 - 0.5 * phase.cos()) as f32
            } else {
                0.0
            };
            framed.try_push(value)?;
        }
    }
    let mut features = backend.workspace_vec(
        context,
        frame_count
            .checked_mul(GEMMA4_AUDIO_MEL_BINS)
            .ok_or(MultimodalTextError::Overflow("Gemma4 log-mel values"))?,
    )?;
    if frame_count > 0 {
        let framed = tensor_from_f32(
            backend,
            &[usize_to_u64(frame_count, "Gemma4 audio frames")?, 512],
            &framed,
            context,
        )?;
        // NumPy's pinned FFT promotes the already-windowed F32 samples to F64.
        let framed = cast_to_with_context_exact_native(
            backend,
            &framed,
            DType::F64,
            DeviceId::CPU,
            false,
            false,
            context,
        )?;
        let spectrum = fftn_with_context_exact_native(backend, &framed, &[1], context)?;
        let spectrum_bytes = spectrum.contiguous_bytes()?;
        let expected_complex = frame_count
            .checked_mul(GEMMA4_AUDIO_FFT_LENGTH)
            .ok_or(MultimodalTextError::Overflow("Gemma4 spectrum values"))?;
        if spectrum_bytes.len()
            != expected_complex
                .checked_mul(16)
                .ok_or(MultimodalTextError::Overflow("Gemma4 spectrum bytes"))?
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 spectrum storage is malformed",
            ));
        }
        let filterbank = gemma4_mel_filterbank(backend, context)?;
        let mut magnitudes = [0.0_f64; 257];
        for frame in 0..frame_count {
            check_gemma_audio_periodically(frame, context)?;
            if *mask.get(frame).ok_or(MultimodalTextError::InvalidInput(
                "Gemma4 frame mask is incomplete",
            ))? == 0
            {
                for _ in 0..GEMMA4_AUDIO_MEL_BINS {
                    features.try_push(0.0)?;
                }
                continue;
            }
            for (frequency, magnitude) in magnitudes.iter_mut().enumerate() {
                let complex_index = frame
                    .checked_mul(GEMMA4_AUDIO_FFT_LENGTH)
                    .and_then(|value| value.checked_add(frequency))
                    .ok_or(MultimodalTextError::Overflow("Gemma4 spectrum index"))?;
                let byte_index = complex_index
                    .checked_mul(16)
                    .ok_or(MultimodalTextError::Overflow("Gemma4 spectrum byte index"))?;
                let bytes = spectrum_bytes.get(byte_index..byte_index + 16).ok_or(
                    MultimodalTextError::InvalidInput("Gemma4 spectrum storage is incomplete"),
                )?;
                let real = f64::from_ne_bytes(bytes[0..8].try_into().map_err(|_| {
                    MultimodalTextError::InvalidInput("Gemma4 spectrum real value is malformed")
                })?);
                let imaginary = f64::from_ne_bytes(bytes[8..16].try_into().map_err(|_| {
                    MultimodalTextError::InvalidInput(
                        "Gemma4 spectrum imaginary value is malformed",
                    )
                })?);
                *magnitude = real.hypot(imaginary);
            }
            for mel in 0..GEMMA4_AUDIO_MEL_BINS {
                let mut sum = 0.0_f64;
                for (frequency, magnitude) in magnitudes.iter().copied().enumerate() {
                    let filter_index = frequency
                        .checked_mul(GEMMA4_AUDIO_MEL_BINS)
                        .and_then(|value| value.checked_add(mel))
                        .ok_or(MultimodalTextError::Overflow("Gemma4 mel filter index"))?;
                    sum += magnitude
                        * *filterbank.get(filter_index).ok_or(
                            MultimodalTextError::InvalidInput(
                                "Gemma4 mel filterbank is incomplete",
                            ),
                        )?;
                }
                features.try_push((sum + 0.001).ln() as f32)?;
            }
        }
    }
    let log_mel = tensor_from_f32(
        backend,
        &[
            1,
            usize_to_u64(frame_count, "Gemma4 audio frames")?,
            GEMMA4_AUDIO_MEL_BINS as u64,
        ],
        &features,
        context,
    )?;
    let mask_descriptor = TensorDescriptor::contiguous(
        vec![1, usize_to_u64(frame_count, "Gemma4 audio frames")?],
        DType::Bool,
        DeviceId::CPU,
        context.stream,
    )?;
    let (frame_mask, _) = backend.upload_bytes(mask_descriptor, &mask, context)?;
    Ok((log_mel, frame_mask))
}

fn gemma4_mel_filterbank(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f64>, MultimodalTextError> {
    let mut filter_frequencies = [0.0_f64; GEMMA4_AUDIO_MEL_BINS + 2];
    let maximum_mel = 2595.0 * (1.0_f64 + 8000.0 / 700.0).log10();
    for (index, value) in filter_frequencies.iter_mut().enumerate() {
        let mel = maximum_mel * index as f64 / (GEMMA4_AUDIO_MEL_BINS + 1) as f64;
        *value = 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
    }
    let mut filterbank = backend.workspace_vec(
        context,
        257_usize
            .checked_mul(GEMMA4_AUDIO_MEL_BINS)
            .ok_or(MultimodalTextError::Overflow("Gemma4 mel filterbank"))?,
    )?;
    for frequency in 0..257 {
        check_gemma_audio_periodically(frequency, context)?;
        let hertz = 8000.0 * frequency as f64 / 256.0;
        for mel in 0..GEMMA4_AUDIO_MEL_BINS {
            let lower = filter_frequencies[mel];
            let center = filter_frequencies[mel + 1];
            let upper = filter_frequencies[mel + 2];
            let rising = (hertz - lower) / (center - lower);
            let falling = (upper - hertz) / (upper - center);
            filterbank.try_push(rising.min(falling).max(0.0))?;
        }
    }
    Ok(filterbank)
}

fn gemma4_firwin_value(
    index: usize,
    half_length: usize,
    cutoff: f64,
    kaiser_denominator: f64,
) -> f64 {
    let distance = index as f64 - half_length as f64;
    let scaled = cutoff * distance;
    let sinc = if scaled == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * scaled).sin() / (std::f64::consts::PI * scaled)
    };
    let ratio = distance / half_length as f64;
    let window = gemma4_bessel_i0(GEMMA4_AUDIO_KAISER_BETA * (1.0 - ratio * ratio).max(0.0).sqrt())
        / kaiser_denominator;
    cutoff * sinc * window
}

fn gemma4_bessel_i0(value: f64) -> f64 {
    let argument = value * value / 4.0;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for order in 1..=64 {
        term *= argument / ((order * order) as f64);
        sum += term;
        if term <= sum * f64::EPSILON {
            break;
        }
    }
    sum
}

const fn greatest_common_divisor(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn div_ceil_i128(value: i128, divisor: i128) -> i128 {
    let quotient = value.div_euclid(divisor);
    if value.rem_euclid(divisor) == 0 {
        quotient
    } else {
        quotient + 1
    }
}

fn check_gemma_audio_periodically(
    index: usize,
    context: &ExecutionContext<'_>,
) -> Result<(), MultimodalTextError> {
    if index.is_multiple_of(1_024) {
        context.cancellation.check()?;
    }
    Ok(())
}

fn project_rgb_channels(
    backend: &CpuBackend,
    image: &ImageTensor,
    context: &ExecutionContext<'_>,
) -> Result<ImageTensor, MultimodalTextError> {
    let (batch, height, width, channels) = image.dimensions()?;
    if channels == 3 {
        return Ok(image.clone());
    }
    if channels < 3 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma visual preparation requires at least three channels",
        ));
    }
    let pixels = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(MultimodalTextError::Overflow("Gemma RGB pixels"))?;
    let channels = u64_to_usize(channels, "Gemma source channels")?;
    let source = image.as_f32_slice()?;
    let mut values = backend.workspace_vec(
        context,
        pixels
            .checked_mul(3)
            .ok_or(MultimodalTextError::Overflow("Gemma RGB values"))?,
    )?;
    for pixel in 0..pixels {
        context.cancellation.check()?;
        let start = pixel
            .checked_mul(channels)
            .ok_or(MultimodalTextError::Overflow("Gemma source pixel"))?;
        for channel in 0..3 {
            values.try_push(*source.get(start + channel).ok_or(
                MultimodalTextError::InvalidInput("Gemma source pixel storage is incomplete"),
            )?)?;
        }
    }
    Ok(ImageTensor::from_f32(
        backend, context, batch, height, width, 3, &values,
    )?)
}

fn gemma4_quantized_bicubic_resize(
    backend: &CpuBackend,
    values: &[f32],
    height: u64,
    width: u64,
    target_height: u64,
    target_width: u64,
    context: &ExecutionContext<'_>,
) -> Result<ImageTensor, MultimodalTextError> {
    let height = u64_to_usize(height, "Gemma4 source height")?;
    let width = u64_to_usize(width, "Gemma4 source width")?;
    let target_height = u64_to_usize(target_height, "Gemma4 target height")?;
    let target_width = u64_to_usize(target_width, "Gemma4 target width")?;
    let source_count = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(3))
        .ok_or(MultimodalTextError::Overflow("Gemma4 quantized source"))?;
    if values.len() != source_count {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 source frame has the wrong length",
        ));
    }
    let mut nchw = backend.workspace_vec(context, source_count)?;
    for channel in 0..3 {
        for y in 0..height {
            context.cancellation.check()?;
            for x in 0..width {
                let source = (y * width + x) * 3 + channel;
                let value = values
                    .get(source)
                    .copied()
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma4 source frame storage is incomplete",
                    ))?
                    .clamp(0.0, 1.0);
                nchw.try_push(f32::from((value * 255.0).trunc() as u8) / 255.0)?;
            }
        }
    }
    let resized = interpolate_with_context_exact_native(
        &nchw,
        &[1, 3, height, width],
        &InterpolateConfiguration {
            output_size: Some(vec![target_height, target_width]),
            scale_factor: None,
            mode: InterpolateMode::Bicubic,
            align_corners: Some(false),
            recompute_scale_factor: None,
            antialias: true,
        },
        DeviceId::CPU,
        context,
    )?;
    if resized.shape != [1, 3, target_height, target_width] {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 bicubic output has the wrong shape",
        ));
    }
    let output_count = target_height
        .checked_mul(target_width)
        .and_then(|value| value.checked_mul(3))
        .ok_or(MultimodalTextError::Overflow("Gemma4 resized output"))?;
    let mut bhwc = backend.workspace_vec(context, output_count)?;
    for y in 0..target_height {
        context.cancellation.check()?;
        for x in 0..target_width {
            for channel in 0..3 {
                let source = (channel * target_height + y) * target_width + x;
                let value = resized
                    .values
                    .get(source)
                    .copied()
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma4 bicubic output storage is incomplete",
                    ))?
                    .clamp(0.0, 1.0);
                bhwc.try_push(f32::from((value * 255.0).round_ties_even() as u8) / 255.0)?;
            }
        }
    }
    Ok(ImageTensor::from_f32(
        backend,
        context,
        1,
        u64::try_from(target_height)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 target height"))?,
        u64::try_from(target_width)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 target width"))?,
        3,
        &bhwc,
    )?)
}

pub fn prepare_qwen3vl_images(
    backend: &CpuBackend,
    images: &ImageTensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Qwen3VlPreparedImage>, MultimodalTextError> {
    prepare_qwen_images(backend, images, QwenVisionFamily::Qwen3Vl8B, context)
}

pub fn prepare_qwen25_images(
    backend: &CpuBackend,
    images: &ImageTensor,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Qwen25PreparedImage>, MultimodalTextError> {
    context.check()?;
    let (batch, height, width, channels) = images.dimensions()?;
    if batch == 0 || channels != 3 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen2.5 images must be a nonempty RGB batch",
        ));
    }
    let (target_height, target_width) = qwen25_target_dimensions(height, width)?;
    let source = images.as_f32_slice()?;
    let image_elements = usize::try_from(
        height
            .checked_mul(width)
            .and_then(|value| value.checked_mul(3))
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 source image"))?,
    )
    .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 source image"))?;
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(u64_to_usize(batch, "Qwen2.5 image batch")?)
        .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 image batch"))?;
    for batch_index in 0..u64_to_usize(batch, "Qwen2.5 image batch")? {
        context.check()?;
        let start = batch_index
            .checked_mul(image_elements)
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 image offset"))?;
        let end = start
            .checked_add(image_elements)
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 image end"))?;
        let singleton = ImageTensor::from_f32(
            backend,
            context,
            1,
            height,
            width,
            3,
            source
                .get(start..end)
                .ok_or(MultimodalTextError::InvalidInput(
                    "Qwen2.5 image storage is incomplete",
                ))?,
        )?;
        let resized = singleton.resize(
            target_width,
            target_height,
            ResizeMode::Bilinear,
            ResizeCrop::Disabled,
            backend,
            context,
        )?;
        prepared.push(prepare_qwen25_resized_image(
            backend,
            &resized,
            target_height,
            target_width,
            context,
        )?);
    }
    Ok(prepared)
}

fn qwen25_target_dimensions(height: u64, width: u64) -> Result<(u64, u64), MultimodalTextError> {
    if height == 0 || width == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen2.5 images require nonzero dimensions",
        ));
    }
    let factor = 28;
    let mut target_height = round_to_factor(height, factor)?;
    let mut target_width = round_to_factor(width, factor)?;
    let rounded = target_height
        .checked_mul(target_width)
        .ok_or(MultimodalTextError::Overflow("Qwen2.5 pixels"))?;
    let source = height
        .checked_mul(width)
        .ok_or(MultimodalTextError::Overflow("Qwen2.5 source pixels"))?;
    if rounded > 12_845_056 {
        let beta = ((source as f64) / 12_845_056_f64).sqrt();
        target_height = floor_scaled_to_factor(height, beta, factor)?;
        target_width = floor_scaled_to_factor(width, beta, factor)?;
    } else if rounded < 3_136 {
        let beta = (3_136_f64 / source as f64).sqrt();
        target_height = ceil_scaled_to_factor(height, beta, factor)?;
        target_width = ceil_scaled_to_factor(width, beta, factor)?;
    }
    Ok((target_height, target_width))
}

fn prepare_qwen25_resized_image(
    backend: &CpuBackend,
    image: &ImageTensor,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Qwen25PreparedImage, MultimodalTextError> {
    context.check()?;
    if image.dimensions()? != (1, height, width, 3)
        || !height.is_multiple_of(28)
        || !width.is_multiple_of(28)
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen2.5 resized image geometry is invalid",
        ));
    }
    let grid_height = u64_to_usize(height / 14, "Qwen2.5 grid height")?;
    let grid_width = u64_to_usize(width / 14, "Qwen2.5 grid width")?;
    let merged_height = grid_height / 2;
    let merged_width = grid_width / 2;
    let patch_count = grid_height
        .checked_mul(grid_width)
        .ok_or(MultimodalTextError::Overflow("Qwen2.5 patches"))?;
    let patch_width = 3 * 2 * 14 * 14;
    let mut patches = backend.workspace_vec(
        context,
        patch_count
            .checked_mul(patch_width)
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 patch storage"))?,
    )?;
    let values = image.as_f32_slice()?;
    let height = u64_to_usize(height, "Qwen2.5 height")?;
    let width = u64_to_usize(width, "Qwen2.5 width")?;
    let mean = [0.481_454_66_f32, 0.457_827_5, 0.408_210_73];
    let deviation = [0.268_629_54_f32, 0.261_302_6, 0.275_777_1];
    for block_y in 0..merged_height {
        for block_x in 0..merged_width {
            context.check()?;
            for merge_y in 0..2 {
                for merge_x in 0..2 {
                    let patch_y = block_y * 2 + merge_y;
                    let patch_x = block_x * 2 + merge_x;
                    for channel in 0..3 {
                        for _temporal in 0..2 {
                            for local_y in 0..14 {
                                for local_x in 0..14 {
                                    let y = patch_y * 14 + local_y;
                                    let x = patch_x * 14 + local_x;
                                    let index = y
                                        .checked_mul(width)
                                        .and_then(|value| value.checked_add(x))
                                        .and_then(|value| value.checked_mul(3))
                                        .and_then(|value| value.checked_add(channel))
                                        .ok_or(MultimodalTextError::Overflow("Qwen2.5 pixel"))?;
                                    let value = *values.get(index).ok_or(
                                        MultimodalTextError::InvalidInput(
                                            "Qwen2.5 resized pixels are incomplete",
                                        ),
                                    )?;
                                    if !value.is_finite() {
                                        return Err(MultimodalTextError::InvalidInput(
                                            "Qwen2.5 pixels must be finite",
                                        ));
                                    }
                                    patches
                                        .try_push((value - mean[channel]) / deviation[channel])?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if patches.len() != patch_count * patch_width || height != grid_height * 14 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen2.5 patch packing is incomplete",
        ));
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![usize_to_u64(patch_count, "Qwen2.5 patches")?, 3, 2, 14, 14],
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    let patches = backend.upload_f32(descriptor, &patches, context)?.0;
    Ok(Qwen25PreparedImage {
        patches,
        grid_thw: [1, grid_height, grid_width],
        merged_tokens: merged_height
            .checked_mul(merged_width)
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 merged tokens"))?,
    })
}

pub fn prepare_qwen_images(
    backend: &CpuBackend,
    images: &ImageTensor,
    family: QwenVisionFamily,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Qwen3VlPreparedImage>, MultimodalTextError> {
    context.cancellation.check()?;
    let (batch, height, width, channels) = images.dimensions()?;
    if batch == 0 || channels != 3 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL images must be a nonempty RGB IMAGE batch",
        ));
    }
    let (target_height, target_width) = qwen3vl_target_dimensions(height, width)?;
    let source = images.as_f32_slice()?;
    let image_elements = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(MultimodalTextError::Overflow(
            "Qwen3-VL source image elements",
        ))?;
    let batch_size = u64_to_usize(batch, "Qwen3-VL image batch")?;
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(batch_size)
        .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL prepared image batch"))?;
    for batch_index in 0..batch_size {
        context.cancellation.check()?;
        let start = batch_index
            .checked_mul(image_elements)
            .ok_or(MultimodalTextError::Overflow("Qwen3-VL image offset"))?;
        let end = start
            .checked_add(image_elements)
            .ok_or(MultimodalTextError::Overflow("Qwen3-VL image end"))?;
        let singleton = ImageTensor::from_f32(
            backend,
            context,
            1,
            height,
            width,
            channels,
            source
                .get(start..end)
                .ok_or(MultimodalTextError::InvalidInput(
                    "Qwen3-VL image batch storage is incomplete",
                ))?,
        )?;
        let resized = singleton.resize(
            target_width,
            target_height,
            ResizeMode::Bilinear,
            ResizeCrop::Disabled,
            backend,
            context,
        )?;
        prepared.push(prepare_qwen3vl_resized_image(
            backend,
            &resized,
            target_height,
            target_width,
            family,
            context,
        )?);
    }
    context.cancellation.check()?;
    Ok(prepared)
}

pub fn plan_qwen3vl_markers(
    tokens: &[i64],
    images: &[Qwen3VlPreparedImage],
    cancellation: &sim_types::CancellationToken,
) -> Result<Qwen3VlMarkerPlan, MultimodalTextError> {
    plan_qwen_markers(tokens, images, QwenVisionFamily::Qwen3Vl8B, cancellation)
}

pub fn plan_qwen_markers(
    tokens: &[i64],
    images: &[Qwen3VlPreparedImage],
    family: QwenVisionFamily,
    cancellation: &sim_types::CancellationToken,
) -> Result<Qwen3VlMarkerPlan, MultimodalTextError> {
    cancellation.check()?;
    if images.iter().any(|image| image.family != family) {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen prepared image family does not match the marker plan",
        ));
    }
    let image_pad_token = family.image_pad_token();
    let marker_count = tokens
        .iter()
        .filter(|token| **token == image_pad_token)
        .count();
    if marker_count != images.len() {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL image markers and prepared images must match exactly",
        ));
    }
    let mut expanded_length = 0_usize;
    let mut length_image_index = 0_usize;
    for token in tokens {
        let token_length = if *token == image_pad_token {
            let image = images
                .get(length_image_index)
                .ok_or(MultimodalTextError::InvalidInput(
                    "Qwen3-VL image marker has no prepared image",
                ))?;
            length_image_index = length_image_index
                .checked_add(1)
                .ok_or(MultimodalTextError::Overflow("Qwen3-VL image index"))?;
            image.merged_tokens
        } else {
            1
        };
        expanded_length = expanded_length
            .checked_add(token_length)
            .ok_or(MultimodalTextError::Overflow("Qwen3-VL expanded tokens"))?;
    }
    let mut expanded_tokens = Vec::new();
    expanded_tokens
        .try_reserve_exact(expanded_length)
        .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL expanded tokens"))?;
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(images.len())
        .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL image spans"))?;
    let mut image_index = 0_usize;
    for (token_index, token) in tokens.iter().copied().enumerate() {
        if token_index.is_multiple_of(256) {
            cancellation.check()?;
        }
        if token != image_pad_token {
            expanded_tokens.push(token);
            continue;
        }
        let image = images
            .get(image_index)
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen3-VL image marker has no prepared image",
            ))?;
        let start = expanded_tokens.len();
        expanded_tokens.extend(std::iter::repeat_n(token, image.merged_tokens));
        spans.push(MultimodalSpan {
            start,
            size: image.merged_tokens,
            grid_thw: image.grid_thw,
        });
        image_index = image_index
            .checked_add(1)
            .ok_or(MultimodalTextError::Overflow("Qwen3-VL image index"))?;
    }
    if expanded_tokens.is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL expanded prompt cannot be empty",
        ));
    }
    let mut visual_position_mask = Vec::new();
    visual_position_mask
        .try_reserve_exact(expanded_tokens.len())
        .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL visual position mask"))?;
    visual_position_mask.resize(expanded_tokens.len(), false);
    for span in &spans {
        let end = span
            .start
            .checked_add(span.size)
            .ok_or(MultimodalTextError::Overflow("Qwen3-VL visual mask"))?;
        visual_position_mask
            .get_mut(span.start..end)
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen3-VL visual span is outside the expanded prompt",
            ))?
            .fill(true);
    }
    cancellation.check()?;
    Ok(Qwen3VlMarkerPlan {
        expanded_tokens,
        spans,
        visual_position_mask,
    })
}

impl NativeGemma4VisionEncoder {
    pub fn new(
        configuration: Gemma4VisionConfiguration,
        weights: Gemma4VisionWeights,
    ) -> Result<Self, MultimodalTextError> {
        configuration.validate()?;
        if weights.blocks.len() != configuration.layer_count {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 vision block count does not match the closed profile",
            ));
        }
        let stream = weights.patch_projection_weight.descriptor().stream();
        let patch_width = 3_usize
            .checked_mul(configuration.patch_size)
            .and_then(|value| value.checked_mul(configuration.patch_size))
            .ok_or(MultimodalTextError::Overflow("Gemma4 patch width"))?;
        let patch_projection = gemma4_linear_module(
            "gemma4_vision.patch_projection",
            patch_width,
            configuration.hidden_size,
            weights.patch_projection_weight,
            stream,
        )?;
        gemma4_require_parameter_shape(
            &weights.position_embedding,
            &[
                2,
                configuration.position_embeddings,
                configuration.hidden_size,
            ],
            stream,
        )?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(configuration.layer_count)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 vision blocks"))?;
        for (index, weights) in weights.blocks.into_iter().enumerate() {
            blocks.push(gemma4_vision_block(index, &configuration, weights, stream)?);
        }
        let projector = gemma4_linear_module(
            "gemma4_vision.projector",
            configuration.hidden_size,
            configuration.output_hidden_size,
            weights.projector_weight,
            stream,
        )?;
        let mut owner = Self {
            configuration,
            patch_projection,
            position_embedding: weights.position_embedding,
            blocks,
            projector,
            semantic_state_digest_sha256: String::new(),
        };
        owner.semantic_state_digest_sha256 =
            owner.project_semantic_state_digest(&sim_types::CancellationToken::default())?;
        owner.validate(&sim_types::CancellationToken::default())?;
        Ok(owner)
    }

    pub fn configuration(&self) -> &Gemma4VisionConfiguration {
        &self.configuration
    }

    pub fn execution_stream(&self) -> StreamId {
        self.position_embedding.descriptor().stream()
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        if !mapped.unexpected_keys().is_empty() {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 vision mapped weights contain unexpected parameters",
            ));
        }
        let mut reconstructed = self.clone();
        let mut consumed = BTreeSet::new();
        reload_multimodal_module(
            &mut reconstructed.patch_projection,
            "patch_projection",
            mapped,
            &mut consumed,
        )?;
        reconstructed.position_embedding =
            take_multimodal_tensor(mapped, "position_embedding", &mut consumed)?;
        reload_multimodal_module(
            &mut reconstructed.projector,
            "projector",
            mapped,
            &mut consumed,
        )?;
        for (index, block) in reconstructed.blocks.iter_mut().enumerate() {
            cancellation.check()?;
            reload_gemma4_vision_block(block, &format!("blocks.{index}"), mapped, &mut consumed)?;
        }
        if consumed.len() != mapped.tensors().len() {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 vision mapped weights contain unconsumed parameters",
            ));
        }
        reconstructed.semantic_state_digest_sha256 =
            reconstructed.project_semantic_state_digest(cancellation)?;
        reconstructed.validate(cancellation)?;
        Ok(reconstructed)
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        self.configuration.validate()?;
        if self.blocks.len() != self.configuration.layer_count {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 retained vision block count changed after admission",
            ));
        }
        if self.semantic_state_digest_sha256 != self.project_semantic_state_digest(cancellation)? {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 vision semantic identity changed after admission",
            ));
        }
        self.resident_bytes()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, MultimodalTextError> {
        let mut allocations = Vec::new();
        insert_gemma4_resident_allocation(
            &mut allocations,
            self.position_embedding.storage_id(),
            self.position_embedding.storage_byte_len(),
        )?;
        for (_, module) in self.named_modules() {
            for (storage_id, bytes) in module.resident_tensor_allocations() {
                insert_gemma4_resident_allocation(&mut allocations, storage_id, bytes)?;
            }
        }
        for tensor in self.named_tensors() {
            insert_gemma4_resident_allocation(
                &mut allocations,
                tensor.storage_id(),
                tensor.storage_byte_len(),
            )?;
        }
        Ok(allocations)
    }

    pub(crate) fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        for (_, module) in self.named_modules() {
            requirements.extend(module.execution_requirements(dtype).iter());
        }
        requirements
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        let block_bytes = self
            .blocks
            .capacity()
            .checked_mul(mem::size_of::<NativeGemma4VisionBlock>())
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 vision block residency",
            ))?;
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .ok()
            .and_then(|bytes| {
                u64::try_from(block_bytes)
                    .ok()
                    .and_then(|part| bytes.checked_add(part))
            })
            .and_then(|bytes| {
                bytes.checked_add(u64::try_from(self.semantic_state_digest_sha256.capacity()).ok()?)
            })
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 vision owned residency",
            ))?;
        for (_, module) in self.named_modules() {
            let tensor_bytes = module.resident_tensor_allocations().into_iter().try_fold(
                0_u64,
                |total, (_, allocation)| {
                    total
                        .checked_add(allocation)
                        .ok_or(MultimodalTextError::Overflow(
                            "Gemma4 vision module tensor residency",
                        ))
                },
            )?;
            bytes = bytes
                .checked_add(
                    module
                        .resident_storage_bytes()?
                        .checked_sub(tensor_bytes)
                        .ok_or(MultimodalTextError::Overflow(
                            "Gemma4 vision module residency projection",
                        ))?,
                )
                .ok_or(MultimodalTextError::Overflow(
                    "Gemma4 vision module residency",
                ))?;
        }
        Ok(bytes)
    }

    pub fn resident_bytes(&self) -> Result<u64, MultimodalTextError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(MultimodalTextError::Overflow("Gemma4 vision residency"))
            },
        )
    }

    pub fn project(
        &self,
        backend: &CpuBackend,
        prepared: &GemmaPreparedVisual,
        context: &ExecutionContext<'_>,
    ) -> Result<Gemma4VisionProjection, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let expected_soft_tokens = match prepared.kind() {
            GemmaPreparedVisualKind::Gemma4Image => GEMMA4_IMAGE_SOFT_TOKENS,
            GemmaPreparedVisualKind::Gemma4VideoFrame => GEMMA4_VIDEO_SOFT_TOKENS,
            GemmaPreparedVisualKind::Gemma3Image => {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma4 vision cannot consume Gemma3 image preparation",
                ));
            }
        };
        if prepared.maximum_soft_tokens() != expected_soft_tokens {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 prepared visual carries the wrong soft-token budget",
            ));
        }
        let (batch, height, width, channels) = prepared.image().dimensions()?;
        if batch != 1
            || channels != 3
            || height == 0
            || width == 0
            || !height.is_multiple_of(usize_to_u64(
                self.configuration.patch_size,
                "Gemma4 patch size",
            )?)
            || !width.is_multiple_of(usize_to_u64(
                self.configuration.patch_size,
                "Gemma4 patch size",
            )?)
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 vision requires one canonically prepared RGB visual",
            ));
        }
        let height = u64_to_usize(height, "Gemma4 prepared height")?;
        let width = u64_to_usize(width, "Gemma4 prepared width")?;
        let patches_high = height / self.configuration.patch_size;
        let patches_wide = width / self.configuration.patch_size;
        if patches_high > self.configuration.position_embeddings
            || patches_wide > self.configuration.position_embeddings
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 prepared position exceeds the learned position table",
            ));
        }
        let patch_count = patches_high
            .checked_mul(patches_wide)
            .ok_or(MultimodalTextError::Overflow("Gemma4 patch count"))?;
        let padded_patch_count = expected_soft_tokens
            .checked_mul(self.configuration.pooling_kernel_size)
            .and_then(|value| value.checked_mul(self.configuration.pooling_kernel_size))
            .ok_or(MultimodalTextError::Overflow("Gemma4 padded patches"))?;
        if patch_count > padded_patch_count {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 prepared visual exceeds its soft-token budget",
            ));
        }
        let patch_width = 3_usize
            .checked_mul(self.configuration.patch_size)
            .and_then(|value| value.checked_mul(self.configuration.patch_size))
            .ok_or(MultimodalTextError::Overflow("Gemma4 patch width"))?;
        let patch_storage = padded_patch_count
            .checked_mul(patch_width)
            .ok_or(MultimodalTextError::Overflow("Gemma4 patch storage"))?;
        let mut patches = backend.workspace_vec(context, patch_storage)?;
        for _ in 0..patch_storage {
            patches.try_push(0.0)?;
        }
        gemma4_patchify(
            prepared.image().as_f32_slice()?,
            height,
            width,
            self.configuration.patch_size,
            &mut patches,
            context.cancellation,
        )?;
        for value in patches.iter_mut() {
            *value = 2.0 * (*value - 0.5);
        }
        let patch_tensor = qwen_tensor(
            backend,
            &[padded_patch_count, patch_width],
            &patches,
            context,
        )?;
        let mut patch_projection = self.patch_projection.clone();
        let projected = patch_projection.forward_with_context(backend, &patch_tensor, context)?;
        let mut hidden = tensor_to_f32(backend, &projected, context)?;
        gemma4_add_positions(
            &mut hidden,
            &tensor_to_f32(backend, &self.position_embedding, context)?,
            patches_high,
            patches_wide,
            padded_patch_count,
            self.configuration.hidden_size,
            self.configuration.position_embeddings,
            context.cancellation,
        )?;
        let mut hidden = qwen_tensor(
            backend,
            &[padded_patch_count, self.configuration.hidden_size],
            &hidden,
            context,
        )?;
        for block in &self.blocks {
            context.cancellation.check()?;
            hidden = block.forward(
                backend,
                &hidden,
                patches_high,
                patches_wide,
                patch_count,
                &self.configuration,
                context,
            )?;
        }
        let hidden_values = tensor_to_f32(backend, &hidden, context)?;
        let pooled = gemma4_pool(
            backend,
            &hidden_values,
            patches_high,
            patches_wide,
            self.configuration.hidden_size,
            self.configuration.pooling_kernel_size,
            context,
        )?;
        let tokens = (patches_high / self.configuration.pooling_kernel_size)
            .checked_mul(patches_wide / self.configuration.pooling_kernel_size)
            .ok_or(MultimodalTextError::Overflow("Gemma4 pooled tokens"))?;
        let normalized = gemma4_parameterless_rms_norm(
            backend,
            &pooled,
            tokens,
            self.configuration.hidden_size,
            context,
        )?;
        let normalized = qwen_tensor(
            backend,
            &[tokens, self.configuration.hidden_size],
            &normalized,
            context,
        )?;
        let mut projector = self.projector.clone();
        let embedding = projector.forward_with_context(backend, &normalized, context)?;
        context.check()?;
        Ok(Gemma4VisionProjection {
            embedding,
            tokens,
            kind: prepared.kind(),
        })
    }

    fn project_semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        cancellation.check()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.gemma4-vision.v1");
        hasher.update(GEMMA4_MULTIMODAL_SOURCE_SHA256.as_bytes());
        hasher.update(canonical_multimodal_bytes(&self.configuration)?);
        hasher.update(self.position_embedding.contiguous_bytes()?);
        for (name, module) in self.named_modules() {
            cancellation.check()?;
            hasher.update(name.as_bytes());
            hasher.update(module.semantic_state_digest(cancellation)?.as_bytes());
        }
        for tensor in self.named_tensors() {
            cancellation.check()?;
            hasher.update(tensor.contiguous_bytes()?);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn named_modules(&self) -> Vec<(String, &NativeModule)> {
        let mut modules = vec![
            ("patch_projection".to_owned(), &self.patch_projection),
            ("projector".to_owned(), &self.projector),
        ];
        for (index, block) in self.blocks.iter().enumerate() {
            for (name, module) in block.named_modules() {
                modules.push((format!("blocks.{index}.{name}"), module));
            }
        }
        modules
    }

    fn named_tensors(&self) -> Vec<&Tensor> {
        let mut tensors = Vec::new();
        for block in &self.blocks {
            tensors.extend(block.named_tensors());
        }
        tensors
    }
}

impl NativeGemma4AudioEncoder {
    pub fn new(
        configuration: Gemma4AudioConfiguration,
        weights: Gemma4AudioWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        configuration.validate()?;
        if weights.blocks.len() != configuration.layer_count {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio block count does not match the configuration",
            ));
        }
        let stream = weights.first_convolution_weight.descriptor().stream();
        gemma4_require_parameter_shape(
            &weights.first_convolution_weight,
            &[configuration.first_convolution_channels, 1, 3, 3],
            stream,
        )?;
        gemma4_require_parameter_shape(
            &weights.first_convolution_normalization_weight,
            &[configuration.first_convolution_channels],
            stream,
        )?;
        gemma4_require_parameter_shape(
            &weights.second_convolution_weight,
            &[
                configuration.second_convolution_channels,
                configuration.first_convolution_channels,
                3,
                3,
            ],
            stream,
        )?;
        gemma4_require_parameter_shape(
            &weights.second_convolution_normalization_weight,
            &[configuration.second_convolution_channels],
            stream,
        )?;
        let subsample_projection = gemma4_linear_module(
            "gemma4_audio.subsample_projection",
            configuration.subsampled_feature_width()?,
            configuration.hidden_size,
            weights.subsample_projection_weight,
            stream,
        )?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(configuration.layer_count)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 audio blocks"))?;
        for (index, block) in weights.blocks.into_iter().enumerate() {
            cancellation.check()?;
            blocks.push(gemma4_audio_block(index, &configuration, block, stream)?);
        }
        gemma4_require_parameter_shape(
            &weights.encoder_output_bias,
            &[configuration.encoder_output_size],
            stream,
        )?;
        let encoder_output = qwen_linear_module(
            "gemma4_audio.encoder_output",
            configuration.hidden_size,
            configuration.encoder_output_size,
            weights.encoder_output_weight,
            Some(weights.encoder_output_bias),
            stream,
        )?;
        let projector = gemma4_linear_module(
            "gemma4_audio.projector",
            configuration.encoder_output_size,
            configuration.output_hidden_size,
            weights.projector_weight,
            stream,
        )?;
        let mut owner = Self {
            configuration,
            first_convolution_weight: weights.first_convolution_weight,
            first_convolution_normalization_weight: weights.first_convolution_normalization_weight,
            second_convolution_weight: weights.second_convolution_weight,
            second_convolution_normalization_weight: weights
                .second_convolution_normalization_weight,
            subsample_projection,
            blocks,
            encoder_output,
            projector,
            semantic_state_digest_sha256: String::new(),
        };
        owner.semantic_state_digest_sha256 = owner.project_semantic_state_digest(cancellation)?;
        owner.validate(cancellation)?;
        Ok(owner)
    }

    pub fn configuration(&self) -> &Gemma4AudioConfiguration {
        &self.configuration
    }

    pub fn execution_stream(&self) -> StreamId {
        self.first_convolution_weight.descriptor().stream()
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        if !mapped.unexpected_keys().is_empty() {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio mapped weights contain unexpected parameters",
            ));
        }
        let mut reconstructed = self.clone();
        let mut consumed = BTreeSet::new();
        reconstructed.first_convolution_weight =
            take_multimodal_tensor(mapped, "first_convolution.weight", &mut consumed)?;
        reconstructed.first_convolution_normalization_weight = take_multimodal_tensor(
            mapped,
            "first_convolution.normalization.weight",
            &mut consumed,
        )?;
        reconstructed.second_convolution_weight =
            take_multimodal_tensor(mapped, "second_convolution.weight", &mut consumed)?;
        reconstructed.second_convolution_normalization_weight = take_multimodal_tensor(
            mapped,
            "second_convolution.normalization.weight",
            &mut consumed,
        )?;
        reload_multimodal_module(
            &mut reconstructed.subsample_projection,
            "subsample_projection",
            mapped,
            &mut consumed,
        )?;
        reload_multimodal_module(
            &mut reconstructed.encoder_output,
            "encoder_output",
            mapped,
            &mut consumed,
        )?;
        reload_multimodal_module(
            &mut reconstructed.projector,
            "projector",
            mapped,
            &mut consumed,
        )?;
        for (index, block) in reconstructed.blocks.iter_mut().enumerate() {
            cancellation.check()?;
            reload_gemma4_audio_block(block, &format!("blocks.{index}"), mapped, &mut consumed)?;
        }
        if consumed.len() != mapped.tensors().len() {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio mapped weights contain unconsumed parameters",
            ));
        }
        reconstructed.semantic_state_digest_sha256 =
            reconstructed.project_semantic_state_digest(cancellation)?;
        reconstructed.validate(cancellation)?;
        Ok(reconstructed)
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        self.configuration.validate()?;
        if self.blocks.len() != self.configuration.layer_count
            || self.semantic_state_digest_sha256
                != self.project_semantic_state_digest(cancellation)?
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio retained semantic state changed",
            ));
        }
        self.resident_tensor_allocations()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn project(
        &self,
        backend: &CpuBackend,
        prepared: &GemmaPreparedAudio,
        context: &ExecutionContext<'_>,
    ) -> Result<Gemma4AudioProjection, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let shape = prepared.log_mel().descriptor().shape();
        if shape.len() != 3
            || shape[0] != 1
            || shape[2] != usize_to_u64(self.configuration.mel_bins, "Gemma4 mel bins")?
            || prepared.log_mel().descriptor().dtype() != DType::F32
            || prepared.log_mel().descriptor().device() != DeviceId::CPU
            || prepared.log_mel().descriptor().stream() != context.stream
            || prepared.frame_mask().descriptor().shape() != [shape[0], shape[1]]
            || prepared.frame_mask().descriptor().dtype() != DType::Bool
            || prepared.frame_mask().descriptor().stream() != context.stream
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio encoder requires canonical log-mel and frame-mask preparation",
            ));
        }
        let frames = u64_to_usize(shape[1], "Gemma4 audio frames")?;
        if frames == 0 {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio encoder requires at least one prepared frame",
            ));
        }
        let mask_bytes = prepared.frame_mask().contiguous_bytes()?;
        if mask_bytes.len() != frames {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio frame mask storage is malformed",
            ));
        }
        let mut mask = Vec::new();
        mask.try_reserve_exact(frames)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 audio frame mask"))?;
        mask.extend_from_slice(mask_bytes);
        let values = tensor_to_f32(backend, prepared.log_mel(), context)?;
        let (values, first_frames, first_bins) = gemma4_audio_conv2d_layer(
            backend,
            &values,
            1,
            frames,
            self.configuration.mel_bins,
            &tensor_to_f32(backend, &self.first_convolution_weight, context)?,
            self.configuration.first_convolution_channels,
            &tensor_to_f32(
                backend,
                &self.first_convolution_normalization_weight,
                context,
            )?,
            &mask,
            f32::from_bits(self.configuration.normalization_epsilon_bits),
            context,
        )?;
        mask = gemma4_audio_subsample_mask(&mask)?;
        let (values, second_frames, second_bins) = gemma4_audio_conv2d_layer(
            backend,
            &values,
            self.configuration.first_convolution_channels,
            first_frames,
            first_bins,
            &tensor_to_f32(backend, &self.second_convolution_weight, context)?,
            self.configuration.second_convolution_channels,
            &tensor_to_f32(
                backend,
                &self.second_convolution_normalization_weight,
                context,
            )?,
            &mask,
            f32::from_bits(self.configuration.normalization_epsilon_bits),
            context,
        )?;
        mask = gemma4_audio_subsample_mask(&mask)?;
        if mask.len() != second_frames
            || second_bins.checked_mul(self.configuration.second_convolution_channels)
                != Some(self.configuration.subsampled_feature_width()?)
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio subsampling produced an invalid geometry",
            ));
        }
        let flattened = gemma4_audio_flatten_convolution(
            backend,
            &values,
            second_frames,
            second_bins,
            self.configuration.second_convolution_channels,
            context,
        )?;
        let mut subsample_projection = self.subsample_projection.clone();
        let mut hidden = subsample_projection.forward_with_context(backend, &flattened, context)?;
        for block in &self.blocks {
            context.cancellation.check()?;
            hidden = block.forward(backend, &hidden, &mask, &self.configuration, context)?;
        }
        let mut encoder_output = self.encoder_output.clone();
        let encoded = encoder_output.forward_with_context(backend, &hidden, context)?;
        let normalized = gemma4_parameterless_rms_norm(
            backend,
            &tensor_to_f32(backend, &encoded, context)?,
            second_frames,
            self.configuration.encoder_output_size,
            context,
        )?;
        let normalized = qwen_tensor(
            backend,
            &[second_frames, self.configuration.encoder_output_size],
            &normalized,
            context,
        )?;
        let mut projector = self.projector.clone();
        let embedding = projector.forward_with_context(backend, &normalized, context)?;
        if prepared.marker_tokens() != second_frames.min(GEMMA4_AUDIO_MAXIMUM_TOKENS) {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 audio marker span does not match encoder subsampling",
            ));
        }
        context.check()?;
        Ok(Gemma4AudioProjection {
            embedding,
            tokens: second_frames,
        })
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, MultimodalTextError> {
        let mut allocations = Vec::new();
        for tensor in self.named_tensors() {
            insert_gemma4_resident_allocation(
                &mut allocations,
                tensor.storage_id(),
                tensor.storage_byte_len(),
            )?;
        }
        for (_, module) in self.named_modules() {
            for (storage_id, bytes) in module.resident_tensor_allocations() {
                insert_gemma4_resident_allocation(&mut allocations, storage_id, bytes)?;
            }
        }
        Ok(allocations)
    }

    pub(crate) fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        for (_, module) in self.named_modules() {
            requirements.extend(module.execution_requirements(dtype).iter());
        }
        requirements
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        let block_bytes = self
            .blocks
            .capacity()
            .checked_mul(mem::size_of::<NativeGemma4AudioBlock>())
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 audio block residency",
            ))?;
        let mut bytes = u64::try_from(mem::size_of::<Self>() + block_bytes)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 audio owned residency"))?;
        for (_, module) in self.named_modules() {
            let tensor_bytes = module.resident_tensor_allocations().into_iter().try_fold(
                0_u64,
                |total, (_, allocation)| {
                    total
                        .checked_add(allocation)
                        .ok_or(MultimodalTextError::Overflow(
                            "Gemma4 audio module tensor residency",
                        ))
                },
            )?;
            bytes = bytes
                .checked_add(
                    module
                        .resident_storage_bytes()?
                        .checked_sub(tensor_bytes)
                        .ok_or(MultimodalTextError::Overflow(
                            "Gemma4 audio module residency projection",
                        ))?,
                )
                .ok_or(MultimodalTextError::Overflow(
                    "Gemma4 audio module residency",
                ))?;
        }
        Ok(bytes)
    }

    pub fn resident_bytes(&self) -> Result<u64, MultimodalTextError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(MultimodalTextError::Overflow("Gemma4 audio residency"))
            },
        )
    }

    fn project_semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        cancellation.check()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.gemma4-audio.v1");
        hasher.update(GEMMA4_MULTIMODAL_SOURCE_SHA256.as_bytes());
        hasher.update(canonical_multimodal_bytes(&self.configuration)?);
        for (name, module) in self.named_modules() {
            cancellation.check()?;
            hasher.update(name.as_bytes());
            hasher.update(module.semantic_state_digest(cancellation)?.as_bytes());
        }
        for tensor in self.named_tensors() {
            cancellation.check()?;
            hasher.update(tensor.contiguous_bytes()?);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn named_modules(&self) -> Vec<(String, &NativeModule)> {
        let mut modules = vec![
            (
                "subsample_projection".to_owned(),
                &self.subsample_projection,
            ),
            ("encoder_output".to_owned(), &self.encoder_output),
            ("projector".to_owned(), &self.projector),
        ];
        for (index, block) in self.blocks.iter().enumerate() {
            for (name, module) in block.named_modules() {
                modules.push((format!("blocks.{index}.{name}"), module));
            }
        }
        modules
    }

    fn named_tensors(&self) -> Vec<&Tensor> {
        let mut tensors = vec![
            &self.first_convolution_weight,
            &self.first_convolution_normalization_weight,
            &self.second_convolution_weight,
            &self.second_convolution_normalization_weight,
        ];
        for block in &self.blocks {
            tensors.extend(block.named_tensors());
        }
        tensors
    }
}

impl NativeGemma4AudioBlock {
    fn named_modules(&self) -> Vec<(&'static str, &NativeModule)> {
        vec![
            ("ff1.first", &self.feed_forward_one.first.linear),
            ("ff1.second", &self.feed_forward_one.second.linear),
            ("query", &self.query.linear),
            ("key", &self.key.linear),
            ("value", &self.value.linear),
            ("attention_output", &self.attention_output.linear),
            ("relative_key_projection", &self.relative_key_projection),
            ("convolution_start", &self.convolution_start.linear),
            ("convolution_end", &self.convolution_end.linear),
            ("ff2.first", &self.feed_forward_two.first.linear),
            ("ff2.second", &self.feed_forward_two.second.linear),
        ]
    }

    fn named_tensors(&self) -> Vec<&Tensor> {
        let mut tensors = vec![
            &self.feed_forward_one.pre_normalization_weight,
            &self.feed_forward_one.post_normalization_weight,
            &self.attention_scale,
            &self.pre_attention_normalization_weight,
            &self.post_attention_normalization_weight,
            &self.convolution_pre_normalization_weight,
            &self.depthwise_convolution_weight,
            &self.convolution_normalization_weight,
            &self.feed_forward_two.pre_normalization_weight,
            &self.feed_forward_two.post_normalization_weight,
            &self.output_normalization_weight,
        ];
        for linear in [
            &self.feed_forward_one.first,
            &self.feed_forward_one.second,
            &self.query,
            &self.key,
            &self.value,
            &self.attention_output,
            &self.convolution_start,
            &self.convolution_end,
            &self.feed_forward_two.first,
            &self.feed_forward_two.second,
        ] {
            tensors.extend(linear.named_tensors());
        }
        tensors
    }

    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        mask: &[u8],
        configuration: &Gemma4AudioConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        let tokens = mask.len();
        let mut hidden =
            self.feed_forward_one
                .forward(backend, input, tokens, configuration, context)?;
        let residual = hidden.clone();
        let normalized = gemma4_weighted_rms_norm(
            backend,
            &hidden,
            &self.pre_attention_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        let attention =
            gemma4_audio_attention(backend, &normalized, mask, self, configuration, context)?;
        let attention = gemma4_weighted_rms_norm(
            backend,
            &attention,
            &self.post_attention_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        hidden = qwen_add_tensors(backend, &residual, &attention, context)?;
        hidden = gemma4_audio_convolution(backend, &hidden, self, configuration, context)?;
        hidden = self
            .feed_forward_two
            .forward(backend, &hidden, tokens, configuration, context)?;
        gemma4_weighted_rms_norm(
            backend,
            &hidden,
            &self.output_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )
    }
}

impl NativeGemma4AudioFeedForward {
    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        tokens: usize,
        configuration: &Gemma4AudioConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        let normalized = gemma4_weighted_rms_norm(
            backend,
            input,
            &self.pre_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        let first = self.first.forward(backend, &normalized, context)?;
        let mut values = tensor_to_f32(backend, &first, context)?;
        for value in values.iter_mut() {
            *value *= 1.0 / (1.0 + (-*value).exp());
        }
        let activated = qwen_tensor(
            backend,
            &[tokens, configuration.intermediate_size],
            &values,
            context,
        )?;
        let second = self.second.forward(backend, &activated, context)?;
        let normalized = gemma4_weighted_rms_norm(
            backend,
            &second,
            &self.post_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        let mut values = tensor_to_f32(backend, &normalized, context)?;
        let residual_weight = f32::from_bits(configuration.residual_weight_bits);
        for value in values.iter_mut() {
            *value *= residual_weight;
        }
        let scaled = qwen_tensor(
            backend,
            &[tokens, configuration.hidden_size],
            &values,
            context,
        )?;
        qwen_add_tensors(backend, input, &scaled, context)
    }
}

impl NativeQwen25VisionEncoder {
    pub fn new(weights: Qwen25VisionWeights) -> Result<Self, MultimodalTextError> {
        if weights.blocks.len() != 32 {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 vision requires exactly 32 blocks",
            ));
        }
        let stream = weights.patch_weight.descriptor().stream();
        let mut patch_projection = NativeModule::linear(
            "qwen25_vision.patch_projection",
            3 * 2 * 14 * 14,
            1_280,
            false,
            false,
        )?;
        patch_projection.load_dense_parameters(weights.patch_weight, None)?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(32)
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 vision blocks"))?;
        for (index, weights) in weights.blocks.into_iter().enumerate() {
            blocks.push(NativeQwen25VisionBlock {
                normalization_one_weight: weights.normalization_one_weight,
                query_key_value: qwen_linear_module(
                    &format!("qwen25_vision.blocks.{index}.qkv"),
                    1_280,
                    3 * 1_280,
                    weights.query_key_value_weight,
                    Some(weights.query_key_value_bias),
                    stream,
                )?,
                attention_output: qwen_linear_module(
                    &format!("qwen25_vision.blocks.{index}.attention_output"),
                    1_280,
                    1_280,
                    weights.attention_output_weight,
                    Some(weights.attention_output_bias),
                    stream,
                )?,
                normalization_two_weight: weights.normalization_two_weight,
                feed_forward_gate: qwen_linear_module(
                    &format!("qwen25_vision.blocks.{index}.feed_forward_gate"),
                    1_280,
                    3_420,
                    weights.feed_forward_gate_weight,
                    Some(weights.feed_forward_gate_bias),
                    stream,
                )?,
                feed_forward_up: qwen_linear_module(
                    &format!("qwen25_vision.blocks.{index}.feed_forward_up"),
                    1_280,
                    3_420,
                    weights.feed_forward_up_weight,
                    Some(weights.feed_forward_up_bias),
                    stream,
                )?,
                feed_forward_down: qwen_linear_module(
                    &format!("qwen25_vision.blocks.{index}.feed_forward_down"),
                    3_420,
                    1_280,
                    weights.feed_forward_down_weight,
                    Some(weights.feed_forward_down_bias),
                    stream,
                )?,
            });
        }
        let mut merger_first =
            NativeModule::linear("qwen25_vision.merger.first", 5_120, 5_120, true, false)?;
        merger_first
            .load_dense_parameters(weights.merger_first_weight, Some(weights.merger_first_bias))?;
        let mut merger_second =
            NativeModule::linear("qwen25_vision.merger.second", 5_120, 3_584, true, false)?;
        merger_second.load_dense_parameters(
            weights.merger_second_weight,
            Some(weights.merger_second_bias),
        )?;
        let owner = Self {
            patch_projection,
            blocks,
            merger_normalization_weight: weights.merger_normalization_weight,
            merger_first,
            merger_second,
            stream,
        };
        owner.validate(&sim_types::CancellationToken::default())?;
        Ok(owner)
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        if !mapped.unexpected_keys().is_empty() {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 vision mapped weights contain unexpected parameters",
            ));
        }
        let mut reconstructed = self.clone();
        let mut consumed = BTreeSet::new();
        reload_multimodal_module(
            &mut reconstructed.patch_projection,
            "patch_projection",
            mapped,
            &mut consumed,
        )?;
        for (index, block) in reconstructed.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{index}");
            block.normalization_one_weight = take_multimodal_tensor(
                mapped,
                &format!("{prefix}.normalization_one.weight"),
                &mut consumed,
            )?;
            reload_multimodal_module(
                &mut block.query_key_value,
                &format!("{prefix}.query_key_value"),
                mapped,
                &mut consumed,
            )?;
            reload_multimodal_module(
                &mut block.attention_output,
                &format!("{prefix}.attention_output"),
                mapped,
                &mut consumed,
            )?;
            block.normalization_two_weight = take_multimodal_tensor(
                mapped,
                &format!("{prefix}.normalization_two.weight"),
                &mut consumed,
            )?;
            reload_multimodal_module(
                &mut block.feed_forward_gate,
                &format!("{prefix}.feed_forward_gate"),
                mapped,
                &mut consumed,
            )?;
            reload_multimodal_module(
                &mut block.feed_forward_up,
                &format!("{prefix}.feed_forward_up"),
                mapped,
                &mut consumed,
            )?;
            reload_multimodal_module(
                &mut block.feed_forward_down,
                &format!("{prefix}.feed_forward_down"),
                mapped,
                &mut consumed,
            )?;
        }
        reconstructed.merger_normalization_weight =
            take_multimodal_tensor(mapped, "merger.normalization.weight", &mut consumed)?;
        reload_multimodal_module(
            &mut reconstructed.merger_first,
            "merger.first",
            mapped,
            &mut consumed,
        )?;
        reload_multimodal_module(
            &mut reconstructed.merger_second,
            "merger.second",
            mapped,
            &mut consumed,
        )?;
        if consumed.len() != mapped.tensors().len() {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 vision mapped weights contain unconsumed parameters",
            ));
        }
        reconstructed.validate(cancellation)?;
        Ok(reconstructed)
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        if self.blocks.len() != 32
            || self
                .named_tensors()
                .iter()
                .any(|tensor| tensor.descriptor().stream() != self.stream)
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 vision retained state is invalid",
            ));
        }
        self.semantic_state_digest(cancellation)?;
        Ok(())
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        cancellation.check()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.qwen25-vision.v1\0hidden=1280\0output=3584\0intermediate=3420\0layers=32\0heads=16\0patch=14\0temporal=2\0merge=2");
        for (name, module) in self.named_modules() {
            cancellation.check()?;
            hasher.update(
                u64::try_from(name.len())
                    .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 digest"))?
                    .to_be_bytes(),
            );
            hasher.update(name.as_bytes());
            hasher.update(module.semantic_state_digest(cancellation)?.as_bytes());
        }
        for tensor in self.named_tensors() {
            hasher.update(tensor.contiguous_bytes()?);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub(crate) fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = NativeExecutionRequirements::new();
        for (_, module) in self.named_modules() {
            requirements.extend(module.execution_requirements(dtype).iter());
        }
        requirements
    }

    pub fn resident_tensor_allocations(&self) -> Vec<(sim_tensor::StorageId, u64)> {
        let mut allocations = Vec::new();
        for (_, module) in self.named_modules() {
            for allocation in module.resident_tensor_allocations() {
                if !allocations.iter().any(|(id, _)| *id == allocation.0) {
                    allocations.push(allocation);
                }
            }
        }
        for tensor in self.named_tensors() {
            if !allocations.iter().any(|(id, _)| *id == tensor.storage_id()) {
                allocations.push((tensor.storage_id(), tensor.storage_byte_len()));
            }
        }
        allocations
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 vision residency"))?;
        bytes = bytes
            .checked_add(
                u64::try_from(
                    self.blocks
                        .capacity()
                        .checked_mul(mem::size_of::<NativeQwen25VisionBlock>())
                        .ok_or(MultimodalTextError::Overflow("Qwen2.5 blocks"))?,
                )
                .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 blocks"))?,
            )
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 vision residency"))?;
        for (_, module) in self.named_modules() {
            let tensors = module.resident_tensor_allocations().into_iter().try_fold(
                0_u64,
                |total, (_, value)| {
                    total
                        .checked_add(value)
                        .ok_or(MultimodalTextError::Overflow("Qwen2.5 module tensors"))
                },
            )?;
            bytes = bytes
                .checked_add(
                    module
                        .resident_storage_bytes()?
                        .checked_sub(tensors)
                        .ok_or(MultimodalTextError::Overflow("Qwen2.5 module projection"))?,
                )
                .ok_or(MultimodalTextError::Overflow("Qwen2.5 vision residency"))?;
        }
        Ok(bytes)
    }

    fn named_modules(&self) -> Vec<(String, &NativeModule)> {
        let mut modules = vec![
            ("patch_projection".to_owned(), &self.patch_projection),
            ("merger.first".to_owned(), &self.merger_first),
            ("merger.second".to_owned(), &self.merger_second),
        ];
        for (index, block) in self.blocks.iter().enumerate() {
            for (suffix, module) in [
                ("query_key_value", &block.query_key_value),
                ("attention_output", &block.attention_output),
                ("feed_forward_gate", &block.feed_forward_gate),
                ("feed_forward_up", &block.feed_forward_up),
                ("feed_forward_down", &block.feed_forward_down),
            ] {
                modules.push((format!("blocks.{index}.{suffix}"), module));
            }
        }
        modules
    }

    fn named_tensors(&self) -> Vec<&Tensor> {
        let mut tensors = vec![&self.merger_normalization_weight];
        for block in &self.blocks {
            tensors.push(&block.normalization_one_weight);
            tensors.push(&block.normalization_two_weight);
        }
        tensors
    }

    pub fn project(
        &self,
        backend: &CpuBackend,
        prepared: &Qwen25PreparedImage,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let shape = prepared.patches.descriptor().shape();
        if shape.len() != 5 || shape[1..] != [3, 2, 14, 14] {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 prepared patches have invalid geometry",
            ));
        }
        let patch_count = u64_to_usize(shape[0], "Qwen2.5 patches")?;
        let flattened = qwen_tensor(
            backend,
            &[patch_count, 3 * 2 * 14 * 14],
            &tensor_to_f32(backend, &prepared.patches, context)?,
            context,
        )?;
        let mut projection = self.patch_projection.clone();
        let mut hidden = projection.forward_with_context(backend, &flattened, context)?;
        for block in &self.blocks {
            hidden = block.forward(backend, &hidden, prepared.grid_thw, context)?;
        }
        let normalized = gemma4_weighted_rms_norm(
            backend,
            &hidden,
            &self.merger_normalization_weight,
            patch_count,
            1_280,
            1.0e-6,
            context,
        )?;
        let values = tensor_to_f32(backend, &normalized, context)?;
        let merged = qwen_tensor(backend, &[prepared.merged_tokens, 5_120], &values, context)?;
        let mut first = self.merger_first.clone();
        let first = first.forward_with_context(backend, &merged, context)?;
        let mut values = tensor_to_f32(backend, &first, context)?;
        for value in values.iter_mut() {
            *value = 0.5
                * *value
                * (1.0
                    + std::f32::consts::FRAC_2_SQRT_PI
                        .mul_add(*value + 0.044_715 * *value * *value * *value, 0.0)
                        .tanh());
        }
        let activated = qwen_tensor(backend, &[prepared.merged_tokens, 5_120], &values, context)?;
        let mut second = self.merger_second.clone();
        Ok(second.forward_with_context(backend, &activated, context)?)
    }
}

impl NativeQwen25VisionBlock {
    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        grid_thw: [usize; 3],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        let tokens = grid_thw[0]
            .checked_mul(grid_thw[1])
            .and_then(|value| value.checked_mul(grid_thw[2]))
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 vision tokens"))?;
        let normalized = gemma4_weighted_rms_norm(
            backend,
            input,
            &self.normalization_one_weight,
            tokens,
            1_280,
            1.0e-6,
            context,
        )?;
        let mut qkv = self.query_key_value.clone();
        let qkv = qkv.forward_with_context(backend, &normalized, context)?;
        let attention = qwen_vision_attention(
            backend,
            &tensor_to_f32(backend, &qkv, context)?,
            grid_thw,
            1_280,
            16,
            context,
        )?;
        let attention = qwen_tensor(backend, &[tokens, 1_280], &attention, context)?;
        let mut output = self.attention_output.clone();
        let hidden = qwen_add_tensors(
            backend,
            input,
            &output.forward_with_context(backend, &attention, context)?,
            context,
        )?;
        let normalized = gemma4_weighted_rms_norm(
            backend,
            &hidden,
            &self.normalization_two_weight,
            tokens,
            1_280,
            1.0e-6,
            context,
        )?;
        let mut gate = self.feed_forward_gate.clone();
        let mut up = self.feed_forward_up.clone();
        let gate = gate.forward_with_context(backend, &normalized, context)?;
        let up = up.forward_with_context(backend, &normalized, context)?;
        let mut gate_values = tensor_to_f32(backend, &gate, context)?;
        let up_values = tensor_to_f32(backend, &up, context)?;
        for (gate, up) in gate_values.iter_mut().zip(up_values.iter()) {
            *gate = *gate / (1.0 + (-*gate).exp()) * *up;
        }
        let gated = qwen_tensor(backend, &[tokens, 3_420], &gate_values, context)?;
        let mut down = self.feed_forward_down.clone();
        qwen_add_tensors(
            backend,
            &hidden,
            &down.forward_with_context(backend, &gated, context)?,
            context,
        )
    }
}

impl NativeQwenVisionEncoder {
    pub fn new(
        configuration: QwenVisionConfiguration,
        weights: QwenVisionWeights,
    ) -> Result<Self, MultimodalTextError> {
        configuration.validate()?;
        if weights.blocks.len() != configuration.layer_count
            || weights.deepstack_mergers.len() != configuration.family.deepstack_layers().len()
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen vision block or deepstack weight count does not match the profile",
            ));
        }
        let stream = weights.patch_weight.descriptor().stream();
        let patch_width = 3
            * QWEN3VL_IMAGE_TEMPORAL_PATCH_SIZE
            * QWEN3VL_IMAGE_PATCH_SIZE
            * QWEN3VL_IMAGE_PATCH_SIZE;
        let patch_projection = qwen_linear_module(
            "qwen_vision.patch_projection",
            patch_width,
            configuration.hidden_size,
            weights.patch_weight,
            Some(weights.patch_bias),
            stream,
        )?;
        qwen_require_parameter_shape(
            &weights.position_embedding,
            &[configuration.position_embeddings, configuration.hidden_size],
            stream,
        )?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(configuration.layer_count)
            .map_err(|_| MultimodalTextError::Overflow("Qwen vision blocks"))?;
        for (index, weights) in weights.blocks.into_iter().enumerate() {
            blocks.push(qwen_vision_block(index, &configuration, weights, stream)?);
        }
        let merger = qwen_vision_merger(
            "qwen_vision.merger",
            &configuration,
            weights.merger,
            false,
            stream,
        )?;
        let mut deepstack_mergers = Vec::new();
        deepstack_mergers
            .try_reserve_exact(weights.deepstack_mergers.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen deepstack mergers"))?;
        for (index, weights) in weights.deepstack_mergers.into_iter().enumerate() {
            deepstack_mergers.push(qwen_vision_merger(
                &format!("qwen_vision.deepstack.{index}"),
                &configuration,
                weights,
                true,
                stream,
            )?);
        }
        Ok(Self {
            configuration,
            patch_projection,
            position_embedding: weights.position_embedding,
            blocks,
            merger,
            deepstack_mergers,
        })
    }

    pub fn configuration(&self) -> &QwenVisionConfiguration {
        &self.configuration
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        if !mapped.unexpected_keys().is_empty() {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen vision mapped weights contain unexpected parameters",
            ));
        }
        let mut reconstructed = self.clone();
        let mut consumed = BTreeSet::new();
        reload_multimodal_module(
            &mut reconstructed.patch_projection,
            "patch_projection",
            mapped,
            &mut consumed,
        )?;
        reconstructed.position_embedding =
            take_multimodal_tensor(mapped, "position_embedding", &mut consumed)?;
        for (index, block) in reconstructed.blocks.iter_mut().enumerate() {
            cancellation.check()?;
            let prefix = format!("blocks.{index}");
            for (suffix, module) in [
                ("normalization_one", &mut block.normalization_one),
                ("query_key_value", &mut block.query_key_value),
                ("attention_output", &mut block.attention_output),
                ("normalization_two", &mut block.normalization_two),
                ("feed_forward_up", &mut block.feed_forward_up),
                ("feed_forward_down", &mut block.feed_forward_down),
            ] {
                reload_multimodal_module(
                    module,
                    &format!("{prefix}.{suffix}"),
                    mapped,
                    &mut consumed,
                )?;
            }
        }
        reload_qwen_merger(&mut reconstructed.merger, "merger", mapped, &mut consumed)?;
        for (index, merger) in reconstructed.deepstack_mergers.iter_mut().enumerate() {
            reload_qwen_merger(merger, &format!("deepstack.{index}"), mapped, &mut consumed)?;
        }
        if consumed.len() != mapped.tensors().len() {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen vision mapped weights contain unconsumed parameters",
            ));
        }
        reconstructed.semantic_state_digest(cancellation)?;
        Ok(reconstructed)
    }

    pub fn execution_stream(&self) -> StreamId {
        self.position_embedding.descriptor().stream()
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        cancellation.check()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.qwen-vision.v1");
        hasher.update(format!("{:?}", self.configuration).as_bytes());
        hasher.update(self.position_embedding.contiguous_bytes()?);
        for (name, module) in self.named_modules() {
            cancellation.check()?;
            hasher.update([0]);
            hasher.update(name.as_bytes());
            hasher.update(module.semantic_state_digest(cancellation)?.as_bytes());
        }
        cancellation.check()?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_tensor_allocations(&self) -> Vec<(sim_tensor::StorageId, u64)> {
        let mut allocations = vec![(
            self.position_embedding.storage_id(),
            self.position_embedding.storage_byte_len(),
        )];
        for (_, module) in self.named_modules() {
            for (storage_id, bytes) in module.resident_tensor_allocations() {
                if !allocations
                    .iter()
                    .any(|(existing, _)| *existing == storage_id)
                {
                    allocations.push((storage_id, bytes));
                }
            }
        }
        allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, MultimodalTextError> {
        self.resident_tensor_allocations().into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(MultimodalTextError::Overflow("Qwen vision residency"))
            },
        )
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        let block_bytes = self
            .blocks
            .capacity()
            .checked_mul(mem::size_of::<NativeQwenVisionBlock>())
            .ok_or(MultimodalTextError::Overflow("Qwen vision block residency"))?;
        let merger_bytes = self
            .deepstack_mergers
            .capacity()
            .checked_mul(mem::size_of::<NativeQwenVisionMerger>())
            .ok_or(MultimodalTextError::Overflow(
                "Qwen deepstack merger residency",
            ))?;
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .ok()
            .and_then(|bytes| {
                u64::try_from(block_bytes)
                    .ok()
                    .and_then(|part| bytes.checked_add(part))
            })
            .and_then(|bytes| {
                u64::try_from(merger_bytes)
                    .ok()
                    .and_then(|part| bytes.checked_add(part))
            })
            .ok_or(MultimodalTextError::Overflow("Qwen vision owned residency"))?;
        for (_, module) in self.named_modules() {
            let tensor_bytes = module.resident_tensor_allocations().into_iter().try_fold(
                0_u64,
                |total, (_, allocation)| {
                    total
                        .checked_add(allocation)
                        .ok_or(MultimodalTextError::Overflow(
                            "Qwen vision tensor residency",
                        ))
                },
            )?;
            let module_bytes = module.resident_storage_bytes()?;
            bytes = bytes
                .checked_add(module_bytes.checked_sub(tensor_bytes).ok_or(
                    MultimodalTextError::Overflow("Qwen vision module residency projection"),
                )?)
                .ok_or(MultimodalTextError::Overflow(
                    "Qwen vision module residency",
                ))?;
        }
        Ok(bytes)
    }

    pub fn project(
        &self,
        backend: &CpuBackend,
        prepared: &Qwen3VlPreparedImage,
        context: &ExecutionContext<'_>,
    ) -> Result<QwenVisionProjection, MultimodalTextError> {
        context.cancellation.check()?;
        if prepared.family != self.configuration.family {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen prepared image family does not match the retained vision encoder",
            ));
        }
        let patch_shape = prepared.patches.descriptor().shape();
        let patch_count = patch_shape
            .first()
            .copied()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen prepared patch count is invalid",
            ))?;
        if patch_shape != [usize_to_u64(patch_count, "Qwen patch count")?, 3, 2, 16, 16] {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen prepared patches must use the exact flattened Conv3d geometry",
            ));
        }
        let flattened = qwen_tensor(
            backend,
            &[patch_count, 3 * 2 * 16 * 16],
            &tensor_to_f32(backend, &prepared.patches, context)?,
            context,
        )?;
        let mut patch_projection = self.patch_projection.clone();
        let hidden = patch_projection.forward_with_context(backend, &flattened, context)?;
        let mut hidden_values = tensor_to_f32(backend, &hidden, context)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        qwen_add_interpolated_positions(
            &mut hidden_values,
            prepared.grid_thw,
            &tensor_to_f32(backend, &self.position_embedding, context)?,
            self.configuration.hidden_size,
            self.configuration.spatial_merge_size,
            context.cancellation,
        )?;
        let mut hidden = qwen_tensor(
            backend,
            &[patch_count, self.configuration.hidden_size],
            &hidden_values,
            context,
        )?;
        let mut captured = Vec::new();
        captured
            .try_reserve_exact(self.deepstack_mergers.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen deepstack captures"))?;
        for (index, block) in self.blocks.iter().enumerate() {
            context.cancellation.check()?;
            hidden = block.forward(
                backend,
                &hidden,
                prepared.grid_thw,
                &self.configuration,
                context,
            )?;
            if let Some(capture_index) = self
                .configuration
                .family
                .deepstack_layers()
                .iter()
                .position(|layer| *layer == index)
            {
                let merger = self.deepstack_mergers.get(capture_index).ok_or(
                    MultimodalTextError::InvalidInput("Qwen deepstack merger is missing"),
                )?;
                captured.push(merger.forward(backend, &hidden, &self.configuration, context)?);
            }
        }
        let embedding = self
            .merger
            .forward(backend, &hidden, &self.configuration, context)?;
        if embedding.descriptor().shape()
            != [
                usize_to_u64(prepared.merged_tokens, "Qwen merged tokens")?,
                usize_to_u64(self.configuration.output_hidden_size, "Qwen output hidden")?,
            ]
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen merged vision output does not match the prepared span",
            ));
        }
        context.cancellation.check()?;
        Ok(QwenVisionProjection {
            embedding,
            deepstack: captured,
            grid_thw: prepared.grid_thw,
        })
    }

    fn named_modules(&self) -> Vec<(String, &NativeModule)> {
        let mut modules = vec![("patch_projection".to_owned(), &self.patch_projection)];
        for (index, block) in self.blocks.iter().enumerate() {
            for (name, module) in block.modules() {
                modules.push((format!("blocks.{index}.{name}"), module));
            }
        }
        for (child, module) in self.merger.modules() {
            modules.push((format!("merger.{child}"), module));
        }
        for (index, merger) in self.deepstack_mergers.iter().enumerate() {
            for (child, module) in merger.modules() {
                modules.push((format!("deepstack.{index}.{child}"), module));
            }
        }
        modules
    }
}

#[derive(Clone)]
pub struct NativeQwenMultimodal {
    tokenizer: Arc<NativePromptTokenizer>,
    decoder: Arc<NativeDecoderTextEncoder>,
    vision: Arc<NativeQwenVisionEncoder>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum GemmaMultimodalFamily {
    Gemma3FourBVision,
    Gemma3TwelveB,
    Gemma4E2B,
    Gemma4E4B,
    Gemma4ThirtyOneB,
}

#[derive(Clone)]
enum NativeGemmaVisionResource {
    Gemma3(Arc<NativeGemma3VisionProjector>),
    Gemma4(Arc<NativeGemma4VisionEncoder>),
}

#[derive(Clone)]
pub struct NativeGemmaMultimodal {
    family: GemmaMultimodalFamily,
    tokenizer: Arc<NativePromptTokenizer>,
    decoder: Arc<NativeDecoderTextEncoder>,
    vision: NativeGemmaVisionResource,
    audio: Option<Arc<NativeGemma4AudioEncoder>>,
}

pub fn qwen_multimodal_tokenizer_profile(family: QwenVisionFamily) -> Qwen2PretokenizerProfile {
    match family {
        QwenVisionFamily::Qwen3Vl4B | QwenVisionFamily::Qwen3Vl8B => {
            Qwen2PretokenizerProfile::Qwen2
        }
        QwenVisionFamily::Qwen35_08B
        | QwenVisionFamily::Qwen35_2B
        | QwenVisionFamily::Qwen35_4B
        | QwenVisionFamily::Qwen35_9B
        | QwenVisionFamily::Qwen35_27B => Qwen2PretokenizerProfile::Qwen35Declared,
    }
}

pub fn qwen_multimodal_decoder_configuration(
    family: QwenVisionFamily,
) -> Result<DecoderTextConfiguration, MultimodalTextError> {
    let profile = decoder_profile_fact(qwen_decoder_profile_name(family)).ok_or(
        MultimodalTextError::InvalidInput("Qwen decoder source profile is missing"),
    )?;
    let layer_kinds = if profile.architecture == DecoderArchitecture::Qwen35 {
        let period = profile
            .linear_attention_period
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen3.5 linear-attention period is missing",
            ))?;
        (0..profile.hidden_layers)
            .map(|index| {
                if (index + 1).is_multiple_of(period) {
                    DecoderLayerKind::FullAttention
                } else {
                    DecoderLayerKind::LinearAttention
                }
            })
            .collect()
    } else {
        vec![DecoderLayerKind::FullAttention; profile.hidden_layers]
    };
    let qwen35_linear = match (
        profile.linear_key_heads,
        profile.linear_value_heads,
        profile.linear_key_head_dimension,
        profile.linear_value_head_dimension,
        profile.convolution_kernel_size,
    ) {
        (
            Some(key_heads),
            Some(value_heads),
            Some(key_head_dimension),
            Some(value_head_dimension),
            Some(convolution_kernel_size),
        ) => Some(crate::Qwen35LinearConfiguration {
            key_heads,
            value_heads,
            key_head_dimension,
            value_head_dimension,
            convolution_kernel_size,
        }),
        (None, None, None, None, None) => None,
        _ => {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen decoder linear-attention profile is incomplete",
            ));
        }
    };
    let theta = profile
        .rope_theta()
        .next()
        .ok_or(MultimodalTextError::InvalidInput(
            "Qwen decoder RoPE theta is missing",
        ))?;
    let rotary_dimension = if profile.architecture == DecoderArchitecture::Qwen35 {
        ((profile.head_dimension as f32) * profile.partial_rotary_factor()).round() as usize
    } else {
        profile.head_dimension
    };
    let configuration = DecoderTextConfiguration {
        architecture: profile.architecture,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: profile.vocabulary_size,
        maximum_tokens: profile.maximum_positions,
        hidden_size: profile.hidden_size,
        feed_forward_size: profile.intermediate_size,
        layer_kinds,
        attention_heads: profile.attention_heads,
        key_value_heads: profile.key_value_heads,
        head_dimension: profile.head_dimension,
        query_key_norm: profile.query_key_norm,
        qwen35_linear,
        gemma3: None,
        gemma4: None,
        normalization_epsilon_bits: profile.normalization_epsilon_bits,
        rope: crate::DecoderRopeConfiguration {
            theta,
            rotary_dimension,
            interleaved_sections: profile.rope_sections.to_vec(),
            scaling: crate::RopeScaling::None,
        },
        sliding_window: None,
        activation: profile.activation,
        embedding_scale_bits: 1.0_f32.to_bits(),
        residual_scale_bits: 1.0_f32.to_bits(),
        norm_weight_offset_bits: if profile.rms_norm_add {
            1.0_f32.to_bits()
        } else {
            0.0_f32.to_bits()
        },
        logits_soft_cap_bits: profile.final_logit_soft_cap_bits,
        tied_output_head: !profile.untied_output_head,
        stop_tokens: profile.stop_tokens.to_vec(),
    };
    configuration.validate()?;
    Ok(configuration)
}

impl NativeQwenMultimodal {
    pub fn new(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeQwenVisionEncoder>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        if !vision.configuration().source_exact_profile {
            return Err(MultimodalTextError::InvalidInput(
                "production Qwen multimodal resources require closed source-exact profiles",
            ));
        }
        if vision.configuration() != &QwenVisionConfiguration::source(vision.configuration().family)
        {
            return Err(MultimodalTextError::InvalidInput(
                "production Qwen multimodal vision configuration is not the closed source profile",
            ));
        }
        Self::checked(tokenizer, decoder, vision, cancellation)
    }

    #[doc(hidden)]
    pub fn reduced_fixture(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeQwenVisionEncoder>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        if vision.configuration().source_exact_profile {
            return Err(MultimodalTextError::InvalidInput(
                "reduced Qwen fixture resources require a reduced vision profile",
            ));
        }
        Self::checked(tokenizer, decoder, vision, cancellation)
    }

    fn checked(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeQwenVisionEncoder>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        let resource = Self {
            tokenizer,
            decoder,
            vision,
        };
        resource.validate(cancellation)?;
        Ok(resource)
    }

    pub fn family(&self) -> QwenVisionFamily {
        self.vision.configuration().family
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        let (decoder_weights, vision_weights) = split_multimodal_weights(mapped)?;
        let decoder = Arc::new(
            self.decoder
                .reconstruct_from_mapped_weights(&decoder_weights, cancellation)?,
        );
        let vision = Arc::new(
            self.vision
                .reconstruct_from_mapped_weights(&vision_weights, cancellation)?,
        );
        let reconstructed = Self {
            tokenizer: self.tokenizer.clone(),
            decoder,
            vision,
        };
        reconstructed.validate(cancellation)?;
        Ok(reconstructed)
    }

    pub fn tokenizer(&self) -> &Arc<NativePromptTokenizer> {
        &self.tokenizer
    }

    pub fn decoder(&self) -> &Arc<NativeDecoderTextEncoder> {
        &self.decoder
    }

    pub fn vision(&self) -> &Arc<NativeQwenVisionEncoder> {
        &self.vision
    }

    pub fn is_source_exact_profile(&self) -> bool {
        self.vision.configuration().source_exact_profile
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        let family = self.family();
        let expected_tokenizer_profile = qwen_multimodal_tokenizer_profile(family);
        let expected_tokenizer_digest = match expected_tokenizer_profile {
            Qwen2PretokenizerProfile::Qwen2 => QWEN25_TOKENIZER_ARTIFACT_DIGEST,
            Qwen2PretokenizerProfile::Qwen35Declared => QWEN35_TOKENIZER_ARTIFACT_DIGEST,
        };
        if self.tokenizer.qwen2_profile() != Some(expected_tokenizer_profile)
            || self.tokenizer.qwen2_artifact_digest() != Some(expected_tokenizer_digest)
            || self.tokenizer.has_textual_inversion_embeddings()
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen multimodal resources require the exact canonical Qwen2 tokenizer family",
            ));
        }
        let tokenizer_configuration = self.tokenizer.configuration();
        let expected_pad = match expected_tokenizer_profile {
            Qwen2PretokenizerProfile::Qwen2 => 151_643,
            Qwen2PretokenizerProfile::Qwen35Declared => 248_044,
        };
        if tokenizer_configuration.pad_token != expected_pad
            || tokenizer_configuration.minimum_length != Some(1)
            || tokenizer_configuration.minimum_padding.is_some()
            || tokenizer_configuration.pad_to_maximum_length
            || tokenizer_configuration.pad_left
            || tokenizer_configuration.start_token.is_some()
            || tokenizer_configuration.end_token.is_some()
            || !tokenizer_configuration.disable_weights
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen multimodal tokenizer configuration is not source-compatible",
            ));
        }
        let marker = u32::try_from(family.image_pad_token())
            .map_err(|_| MultimodalTextError::Overflow("Qwen image marker"))?;
        if self
            .tokenizer
            .encode_numeric("<|image_pad|>", cancellation)?
            != [marker]
            || self.tokenizer.encode_numeric("", cancellation)? != [expected_pad]
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen multimodal tokenizer control-token projection changed",
            ));
        }
        let decoder_configuration = self.decoder.configuration();
        if decoder_configuration.hidden_size != self.vision.configuration().output_hidden_size
            || tokenizer_configuration.maximum_length < decoder_configuration.maximum_tokens
            || tokenizer_configuration.maximum_word_length != 8
            || tokenizer_configuration.embedding_width.is_some()
            || self.decoder.execution_stream() != self.vision.execution_stream()
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen tokenizer, decoder, and vision sequence or hidden dimensions differ",
            ));
        }
        if self.vision.configuration().source_exact_profile {
            if decoder_configuration != &qwen_multimodal_decoder_configuration(family)? {
                return Err(MultimodalTextError::InvalidInput(
                    "Qwen decoder configuration does not match the closed source profile",
                ));
            }
        } else if !qwen_reduced_decoder_is_compatible(decoder_configuration, family) {
            return Err(MultimodalTextError::InvalidInput(
                "reduced Qwen decoder fixture is not structurally compatible with its family",
            ));
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        self.validate(cancellation)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.qwen-multimodal-resource.v2");
        hasher.update(b"standard-sim-text-generation-adapter");
        hasher.update(format!("{:?}", self.family()).as_bytes());
        hasher.update(QWEN_MULTIMODAL_ROUTING_SOURCE_SHA256.as_bytes());
        hasher.update(crate::LLAMA_SOURCE_SHA256.as_bytes());
        hasher.update(QWEN35_SOURCE_SHA256.as_bytes());
        hasher.update(
            match self.family() {
                QwenVisionFamily::Qwen3Vl4B | QwenVisionFamily::Qwen3Vl8B => QWEN3VL_SOURCE_SHA256,
                QwenVisionFamily::Qwen35_08B
                | QwenVisionFamily::Qwen35_2B
                | QwenVisionFamily::Qwen35_4B
                | QwenVisionFamily::Qwen35_9B
                | QwenVisionFamily::Qwen35_27B => QWEN35_SOURCE_SHA256,
            }
            .as_bytes(),
        );
        hasher.update(QWEN_VL_SOURCE_SHA256.as_bytes());
        hasher.update(SD1_CLIP_SOURCE_SHA256.as_bytes());
        hasher.update(
            self.tokenizer
                .qwen2_artifact_digest()
                .ok_or(MultimodalTextError::InvalidInput(
                    "Qwen tokenizer artifact identity is unavailable",
                ))?
                .as_bytes(),
        );
        hasher.update(self.tokenizer.semantic_digest(cancellation)?.as_bytes());
        hasher.update(self.decoder.semantic_state_digest(cancellation)?.as_bytes());
        hasher.update(self.vision.semantic_state_digest(cancellation)?.as_bytes());
        cancellation.check()?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        u64::try_from(mem::size_of::<Self>())
            .map_err(|_| MultimodalTextError::Overflow("Qwen multimodal owner residency"))
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, MultimodalTextError> {
        let mut allocations = self.decoder.resident_tensor_allocations();
        for (storage_id, resident_bytes) in self.vision.resident_tensor_allocations() {
            if let Some((_, existing_bytes)) = allocations
                .iter()
                .find(|(existing, _)| *existing == storage_id)
            {
                if *existing_bytes != resident_bytes {
                    return Err(MultimodalTextError::InvalidInput(
                        "shared Qwen tensor storage changed resident size",
                    ));
                }
            } else {
                allocations.push((storage_id, resident_bytes));
            }
        }
        Ok(allocations)
    }

    pub fn resident_bytes(&self) -> Result<u64, MultimodalTextError> {
        let mut bytes = self.resident_owned_bytes()?;
        bytes = bytes.checked_add(self.tokenizer.resident_bytes()?).ok_or(
            MultimodalTextError::Overflow("Qwen multimodal backing residency"),
        )?;
        bytes = bytes
            .checked_add(self.decoder.resident_owned_bytes()?)
            .ok_or(MultimodalTextError::Overflow(
                "Qwen multimodal backing residency",
            ))?;
        bytes = bytes
            .checked_add(self.vision.resident_owned_bytes()?)
            .ok_or(MultimodalTextError::Overflow(
                "Qwen multimodal backing residency",
            ))?;
        self.resident_tensor_allocations()?.into_iter().try_fold(
            bytes,
            |total, (_, resident_bytes)| {
                total
                    .checked_add(resident_bytes)
                    .ok_or(MultimodalTextError::Overflow(
                        "Qwen multimodal tensor residency",
                    ))
            },
        )
    }

    pub fn generate(
        &self,
        backend: &CpuBackend,
        request: QwenMultimodalGenerationRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeTextGenerationResult, MultimodalTextError> {
        context.cancellation.check()?;
        self.validate(context.cancellation)?;
        let configuration = self.decoder.generation_configuration(&request.text)?;
        let prompt_tokens = self
            .tokenizer
            .encode_numeric(request.text.formatted_prompt, context.cancellation)?;
        let prompt_tokens = prompt_tokens.into_iter().map(i64::from).collect::<Vec<_>>();
        let marker_plan = plan_qwen_markers(
            &prompt_tokens,
            request.prepared_images,
            self.family(),
            context.cancellation,
        )?;
        if marker_plan
            .expanded_tokens()
            .len()
            .checked_add(configuration.maximum_new_tokens)
            .ok_or(MultimodalTextError::Overflow(
                "Qwen prompt and generation length",
            ))?
            > self.decoder.configuration().maximum_tokens
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen prompt and generation exceed the decoder token limit",
            ));
        }
        let text_embeddings =
            self.decoder
                .embed_token_values(backend, marker_plan.expanded_tokens(), context)?;
        let mut projections = Vec::new();
        projections
            .try_reserve_exact(request.prepared_images.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen vision projections"))?;
        for prepared in request.prepared_images {
            context.cancellation.check()?;
            projections.push(self.vision.project(backend, prepared, context)?);
        }
        let mut image_embeddings = Vec::new();
        image_embeddings
            .try_reserve_exact(projections.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen image embeddings"))?;
        for (span, projection) in marker_plan.spans().iter().zip(&projections) {
            if projection.grid_thw != span.grid_thw {
                return Err(MultimodalTextError::InvalidInput(
                    "Qwen projected image grid does not match its marker span",
                ));
            }
            image_embeddings.push(MultimodalImageEmbedding {
                span: *span,
                embedding: &projection.embedding,
                deepstack: &projection.deepstack,
            });
        }
        if image_embeddings.len() != marker_plan.spans().len() {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen projected images do not cover every marker span",
            ));
        }
        let embeddings =
            join_multimodal_embeddings(backend, &text_embeddings, &image_embeddings, context)?;
        let sequence_length = marker_plan.expanded_tokens().len();
        let causal_positions = decoder_causal_positions(sequence_length)?;
        let multidimensional_positions = if matches!(
            self.family(),
            QwenVisionFamily::Qwen3Vl4B | QwenVisionFamily::Qwen3Vl8B
        ) {
            qwen2vl_mrope_position_ids(sequence_length, marker_plan.spans(), context.cancellation)?
                .map(|positions| qwen_decoder_position_axes(&positions, context.cancellation))
                .transpose()?
        } else {
            None
        };
        let rope_positions = multidimensional_positions.as_ref().map_or(
            DecoderRopePositions::Scalar(&causal_positions),
            |positions| DecoderRopePositions::Multidimensional(positions),
        );
        let deepstack_join = if matches!(
            self.family(),
            QwenVisionFamily::Qwen3Vl4B | QwenVisionFamily::Qwen3Vl8B
        ) && !image_embeddings.is_empty()
        {
            join_qwen3vl_deepstack(backend, sequence_length, &image_embeddings, context)?
        } else {
            if image_embeddings
                .iter()
                .any(|image| !image.deepstack.is_empty())
            {
                return Err(MultimodalTextError::InvalidInput(
                    "Qwen3.5 generation cannot admit deepstack inputs",
                ));
            }
            None
        };
        if deepstack_join
            .as_ref()
            .is_some_and(|joined| joined.visual_position_mask != marker_plan.visual_position_mask())
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen deepstack mask does not match the marker plan",
            ));
        }
        let deepstack = deepstack_join
            .as_ref()
            .map(|joined| DecoderPreparedDeepstack {
                visual_position_mask: &joined.visual_position_mask,
                layers: &joined.layers,
            });
        let outcome = self.decoder.generate_prepared(
            backend,
            DecoderPreparedGenerationPrompt {
                embeddings: &embeddings,
                sampling_history: &[],
                attention_mask: None,
                rope_positions,
                causal_positions: &causal_positions,
                deepstack,
                initial_input_ids: None,
            },
            &configuration,
            request.transaction,
            context,
        )?;
        context.cancellation.check()?;
        Ok(self
            .decoder
            .finish_prepared_generation(&self.tokenizer, outcome, context)?)
    }
}

impl NativeQwen25Multimodal {
    pub fn new(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeQwen25VisionEncoder>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        let owner = Self {
            tokenizer,
            decoder,
            vision,
        };
        owner.validate(cancellation)?;
        Ok(owner)
    }

    pub fn tokenizer(&self) -> &Arc<NativePromptTokenizer> {
        &self.tokenizer
    }
    pub fn decoder(&self) -> &Arc<NativeDecoderTextEncoder> {
        &self.decoder
    }
    pub fn vision(&self) -> &Arc<NativeQwen25VisionEncoder> {
        &self.vision
    }

    pub(crate) fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = self.decoder.execution_requirements();
        requirements.extend(self.vision.execution_requirements(dtype).iter());
        requirements
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        let tokenizer = self.tokenizer.configuration();
        let decoder = self.decoder.configuration();
        if self.tokenizer.qwen2_profile() != Some(Qwen2PretokenizerProfile::Qwen2)
            || self.tokenizer.qwen2_artifact_digest() != Some(QWEN25_TOKENIZER_ARTIFACT_DIGEST)
            || tokenizer.pad_token != 151_643
            || tokenizer.start_token.is_some()
            || tokenizer.end_token.is_some()
            || tokenizer.minimum_length != Some(1)
            || tokenizer.pad_to_maximum_length
            || tokenizer.embedding_width != Some(3_584)
            || decoder.vocabulary_size != 152_064
            || decoder.hidden_size != 3_584
            || decoder.feed_forward_size != 18_944
            || decoder.layer_kinds.len() != 28
            || decoder.attention_heads != 28
            || decoder.key_value_heads != 4
            || decoder.head_dimension != 128
            || decoder.maximum_tokens != 128_000
            || self.decoder.execution_stream() != self.vision.stream
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 multimodal owners do not match the pinned profile",
            ));
        }
        self.vision.validate(cancellation)?;
        Ok(())
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        let (decoder, vision) = split_multimodal_weights(mapped)?;
        Self::new(
            self.tokenizer.clone(),
            Arc::new(
                self.decoder
                    .reconstruct_from_mapped_weights(&decoder, cancellation)?,
            ),
            Arc::new(
                self.vision
                    .reconstruct_from_mapped_weights(&vision, cancellation)?,
            ),
            cancellation,
        )
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        self.validate(cancellation)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.qwen25-image-resource.v1");
        hasher.update(self.tokenizer.semantic_digest(cancellation)?.as_bytes());
        hasher.update(self.decoder.semantic_state_digest(cancellation)?.as_bytes());
        hasher.update(self.vision.semantic_state_digest(cancellation)?.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        u64::try_from(mem::size_of::<Self>())
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 multimodal residency"))
    }

    pub fn resident_owned_allocations(
        &self,
    ) -> Result<Vec<NativeMultimodalOwnedAllocation>, MultimodalTextError> {
        Ok(vec![
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Resource,
                address: self as *const Self as usize,
                resident_bytes: self.resident_owned_bytes()?,
            },
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Tokenizer,
                address: Arc::as_ptr(&self.tokenizer) as usize,
                resident_bytes: self.tokenizer.resident_bytes()?,
            },
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Decoder,
                address: Arc::as_ptr(&self.decoder) as usize,
                resident_bytes: self.decoder.resident_owned_bytes()?,
            },
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Vision,
                address: Arc::as_ptr(&self.vision) as usize,
                resident_bytes: self.vision.resident_owned_bytes()?,
            },
        ])
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, MultimodalTextError> {
        let mut allocations = self.decoder.resident_tensor_allocations();
        for allocation in self.vision.resident_tensor_allocations() {
            if let Some((_, bytes)) = allocations.iter().find(|(id, _)| *id == allocation.0) {
                if *bytes != allocation.1 {
                    return Err(MultimodalTextError::InvalidInput(
                        "Qwen2.5 aliased storage changed size",
                    ));
                }
            } else {
                allocations.push(allocation);
            }
        }
        Ok(allocations)
    }

    pub fn encode_conditioning(
        &self,
        backend: &CpuBackend,
        request: Qwen25ConditioningRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<QwenConditioningOutput, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let formatted = format_qwen25_image_conditioning_prompt(
            request.prompt,
            !request.prepared_images.is_empty(),
            request.custom_template,
        )?;
        let tokens = self
            .tokenizer
            .encode_numeric(&formatted, context.cancellation)?
            .into_iter();
        let mut projected_tokens = Vec::new();
        projected_tokens
            .try_reserve_exact(tokens.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 prompt tokens"))?;
        projected_tokens.extend(tokens.map(i64::from));
        let mut image_layout = Vec::new();
        image_layout
            .try_reserve_exact(request.prepared_images.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 image layout"))?;
        image_layout.extend(
            request
                .prepared_images
                .iter()
                .map(|image| (image.grid_thw, image.merged_tokens)),
        );
        let marker_plan =
            plan_qwen25_marker_layout(&projected_tokens, &image_layout, context.cancellation)?;
        let text_embeddings =
            self.decoder
                .embed_token_values(backend, &marker_plan.expanded_tokens, context)?;
        let mut projections = Vec::new();
        projections
            .try_reserve_exact(marker_plan.spans.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 vision projections"))?;
        for image in request.prepared_images.iter().take(marker_plan.spans.len()) {
            context.cancellation.check()?;
            projections.push(self.vision.project(backend, image, context)?);
        }
        let mut image_embeddings = Vec::new();
        image_embeddings
            .try_reserve_exact(projections.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 image embeddings"))?;
        for (span, projection) in marker_plan.spans.iter().zip(&projections) {
            image_embeddings.push(MultimodalImageEmbedding {
                span: *span,
                embedding: projection,
                deepstack: &[],
            });
        }
        let embeddings =
            join_multimodal_embeddings(backend, &text_embeddings, &image_embeddings, context)?;
        let position_values = if !marker_plan.spans.is_empty() {
            let values = qwen2vl_mrope_position_ids(
                marker_plan.expanded_tokens.len(),
                &marker_plan.spans,
                context.cancellation,
            )?
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen2.5 image positions are missing",
            ))?;
            vec![values.temporal, values.height, values.width]
        } else {
            vec![sequential_positions(marker_plan.expanded_tokens.len())?]
        };
        let causal = decoder_causal_positions(marker_plan.expanded_tokens.len())?;
        let mut position_axes = Vec::new();
        if position_values.len() > 1 {
            position_axes
                .try_reserve_exact(position_values.len())
                .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 position axes"))?;
            for axis in &position_values {
                position_axes.push(
                    axis.iter()
                        .map(|value| {
                            usize::try_from(*value).map_err(|_| {
                                MultimodalTextError::InvalidInput("Qwen2.5 position is negative")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        let rope_positions = if position_axes.is_empty() {
            DecoderRopePositions::Scalar(&causal)
        } else {
            DecoderRopePositions::Multidimensional(&position_axes)
        };
        let output = self.decoder.forward_prepared(
            backend,
            crate::DecoderPreparedTextRequest {
                embeddings: &embeddings,
                attention_mask: None,
                rope_positions,
                causal_positions: &causal,
                cache: None,
                capture_layer: request.capture_layer,
                deepstack: None,
                initial_input_ids: None,
            },
            context,
        )?;
        let hidden = request
            .capture_layer
            .and_then(|_| output.intermediate())
            .unwrap_or(output.last_hidden_state());
        let trim = qwen_conditioning_prefix_trim(&projected_tokens)?;
        let remaining = hidden
            .descriptor()
            .shape()
            .get(1)
            .copied()
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen2.5 hidden tokens are missing",
            ))?
            .checked_sub(
                u64::try_from(trim).map_err(|_| MultimodalTextError::Overflow("Qwen2.5 trim"))?,
            )
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen2.5 trim removes every token",
            ))?;
        let hidden = narrow_method_exact_native(
            hidden,
            1,
            i64::try_from(trim).map_err(|_| MultimodalTextError::Overflow("Qwen2.5 trim"))?,
            remaining,
            context.cancellation,
        )?;
        let trimmed_tokens =
            marker_plan
                .expanded_tokens
                .get(trim..)
                .ok_or(MultimodalTextError::InvalidInput(
                    "Qwen2.5 attention-mask trim is invalid",
                ))?;
        let attention_mask = if trimmed_tokens.iter().all(|token| *token != 151_643) {
            None
        } else {
            let mut mask = backend.workspace_vec(context, trimmed_tokens.len())?;
            for token in trimmed_tokens {
                mask.try_push(if *token == 151_643 { 0.0 } else { 1.0 })?;
            }
            Some(qwen_tensor(
                backend,
                &[1, trimmed_tokens.len()],
                &mask,
                context,
            )?)
        };
        Ok(QwenConditioningOutput {
            hidden,
            attention_mask,
        })
    }
}

struct Qwen25MarkerPlan {
    expanded_tokens: Vec<i64>,
    spans: Vec<MultimodalSpan>,
}

fn plan_qwen25_marker_layout(
    tokens: &[i64],
    images: &[([usize; 3], usize)],
    cancellation: &sim_types::CancellationToken,
) -> Result<Qwen25MarkerPlan, MultimodalTextError> {
    cancellation.check()?;
    let mapped_images = tokens
        .iter()
        .filter(|token| **token == 151_655)
        .count()
        .min(images.len());
    let mut expanded_length = tokens.len();
    let mut length_image_index = 0_usize;
    for token in tokens {
        if *token != 151_655 || length_image_index >= mapped_images {
            continue;
        }
        expanded_length = expanded_length
            .checked_add(images[length_image_index].1.saturating_sub(1))
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 expanded prompt"))?;
        length_image_index = length_image_index
            .checked_add(1)
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 image index"))?;
    }
    let mut expanded_tokens = Vec::new();
    expanded_tokens
        .try_reserve_exact(expanded_length)
        .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 expanded prompt"))?;
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(mapped_images)
        .map_err(|_| MultimodalTextError::Overflow("Qwen2.5 marker spans"))?;
    let mut image_index = 0_usize;
    for (token_index, token) in tokens.iter().copied().enumerate() {
        if token_index.is_multiple_of(256) {
            cancellation.check()?;
        }
        let Some((grid_thw, merged_tokens)) = images.get(image_index) else {
            expanded_tokens.push(token);
            continue;
        };
        if token != 151_655 {
            expanded_tokens.push(token);
            continue;
        }
        if *merged_tokens == 0 {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen2.5 prepared image has no merged tokens",
            ));
        }
        let start = expanded_tokens.len();
        expanded_tokens.extend(std::iter::repeat_n(token, *merged_tokens));
        spans.push(MultimodalSpan {
            start,
            size: *merged_tokens,
            grid_thw: *grid_thw,
        });
        image_index = image_index
            .checked_add(1)
            .ok_or(MultimodalTextError::Overflow("Qwen2.5 image index"))?;
    }
    if expanded_tokens.is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen2.5 expanded prompt cannot be empty",
        ));
    }
    cancellation.check()?;
    Ok(Qwen25MarkerPlan {
        expanded_tokens,
        spans,
    })
}

fn format_qwen25_image_conditioning_prompt(
    prompt: &str,
    has_images: bool,
    custom_template: Option<&str>,
) -> Result<String, MultimodalTextError> {
    if prompt.starts_with("<|im_start|>") || prompt.starts_with("<|start_header_id|>") {
        return Ok(prompt.to_owned());
    }
    let template = custom_template.unwrap_or(if has_images {
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    } else {
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    });
    let (prefix, suffix) = template
        .split_once("{}")
        .ok_or(MultimodalTextError::InvalidInput(
            "Qwen conditioning template must contain exactly one prompt placeholder",
        ))?;
    if suffix.contains("{}") {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen conditioning template contains multiple prompt placeholders",
        ));
    }
    let mut formatted = String::new();
    formatted
        .try_reserve_exact(
            prefix
                .len()
                .checked_add(prompt.len())
                .and_then(|length| length.checked_add(suffix.len()))
                .ok_or(MultimodalTextError::Overflow("Qwen conditioning template"))?,
        )
        .map_err(|_| MultimodalTextError::Overflow("Qwen conditioning template"))?;
    formatted.push_str(prefix);
    formatted.push_str(prompt);
    formatted.push_str(suffix);
    Ok(formatted)
}

fn qwen_conditioning_prefix_trim(tokens: &[i64]) -> Result<usize, MultimodalTextError> {
    let mut template_end = None;
    let mut count = 0;
    for (index, token) in tokens.iter().copied().enumerate() {
        if token == 151_644 && count < 2 {
            template_end = Some(index);
            count += 1;
        }
    }
    let mut template_end = template_end.ok_or(MultimodalTextError::InvalidInput(
        "Qwen conditioning template has no second conversation marker",
    ))?;
    if tokens.get(template_end + 1) == Some(&872) && tokens.get(template_end + 2) == Some(&198) {
        template_end = template_end
            .checked_add(3)
            .ok_or(MultimodalTextError::Overflow("Qwen conditioning prefix"))?;
    }
    Ok(template_end)
}

impl GemmaMultimodalFamily {
    pub const fn tokenizer_profile(self) -> GemmaTokenizerProfile {
        match self {
            Self::Gemma3FourBVision | Self::Gemma3TwelveB => {
                GemmaTokenizerProfile::Gemma3SentencePiece
            }
            Self::Gemma4E2B | Self::Gemma4E4B | Self::Gemma4ThirtyOneB => {
                GemmaTokenizerProfile::Gemma4TokenizerJson
            }
        }
    }

    pub fn decoder_configuration(self) -> DecoderTextConfiguration {
        match self {
            Self::Gemma3FourBVision => {
                gemma3_decoder_configuration(Gemma3DecoderProfile::FourBVision)
            }
            Self::Gemma3TwelveB => gemma3_decoder_configuration(Gemma3DecoderProfile::TwelveB),
            Self::Gemma4E2B => gemma4_decoder_configuration(Gemma4DecoderProfile::E2B),
            Self::Gemma4E4B => gemma4_decoder_configuration(Gemma4DecoderProfile::E4B),
            Self::Gemma4ThirtyOneB => {
                gemma4_decoder_configuration(Gemma4DecoderProfile::ThirtyOneB)
            }
        }
    }

    pub const fn supports_audio(self) -> bool {
        matches!(self, Self::Gemma4E2B | Self::Gemma4E4B)
    }

    pub const fn source_sha256(self) -> &'static str {
        match self {
            Self::Gemma3FourBVision => GEMMA3_FOUR_B_MULTIMODAL_SOURCE_SHA256,
            Self::Gemma3TwelveB => GEMMA3_MULTIMODAL_SOURCE_SHA256,
            Self::Gemma4E2B | Self::Gemma4E4B | Self::Gemma4ThirtyOneB => {
                GEMMA4_MULTIMODAL_SOURCE_SHA256
            }
        }
    }
}

impl NativeGemmaMultimodal {
    pub fn gemma3(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeGemma3VisionProjector>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        let family = match vision.configuration().source_profile {
            Some(Gemma3VisionProfile::FourBVision) => GemmaMultimodalFamily::Gemma3FourBVision,
            Some(Gemma3VisionProfile::TwelveB) => GemmaMultimodalFamily::Gemma3TwelveB,
            None => {
                return Err(MultimodalTextError::InvalidInput(
                    "production Gemma3 resources require a closed source vision profile",
                ));
            }
        };
        Self::checked(
            family,
            tokenizer,
            decoder,
            NativeGemmaVisionResource::Gemma3(vision),
            None,
            true,
            cancellation,
        )
    }

    pub fn gemma4(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeGemma4VisionEncoder>,
        audio: Option<Arc<NativeGemma4AudioEncoder>>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        if !vision.configuration().source_exact_profile {
            return Err(MultimodalTextError::InvalidInput(
                "production Gemma4 resources require a closed source vision profile",
            ));
        }
        let family = match vision.configuration().profile {
            Gemma4VisionProfile::E2B => GemmaMultimodalFamily::Gemma4E2B,
            Gemma4VisionProfile::E4B => GemmaMultimodalFamily::Gemma4E4B,
            Gemma4VisionProfile::ThirtyOneB => GemmaMultimodalFamily::Gemma4ThirtyOneB,
        };
        Self::checked(
            family,
            tokenizer,
            decoder,
            NativeGemmaVisionResource::Gemma4(vision),
            audio,
            true,
            cancellation,
        )
    }

    #[doc(hidden)]
    pub fn reduced_gemma3_fixture(
        family: GemmaMultimodalFamily,
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeGemma3VisionProjector>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        if !matches!(
            family,
            GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB
        ) || vision.configuration().source_profile.is_some()
        {
            return Err(MultimodalTextError::InvalidInput(
                "reduced Gemma3 fixtures require a reduced Gemma3 vision profile",
            ));
        }
        Self::checked(
            family,
            tokenizer,
            decoder,
            NativeGemmaVisionResource::Gemma3(vision),
            None,
            false,
            cancellation,
        )
    }

    #[doc(hidden)]
    pub fn reduced_gemma4_fixture(
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: Arc<NativeGemma4VisionEncoder>,
        audio: Option<Arc<NativeGemma4AudioEncoder>>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        if vision.configuration().source_exact_profile
            || audio
                .as_ref()
                .is_some_and(|owner| owner.configuration().source_exact_profile)
        {
            return Err(MultimodalTextError::InvalidInput(
                "reduced Gemma4 fixtures require reduced vision and audio profiles",
            ));
        }
        let family = match vision.configuration().profile {
            Gemma4VisionProfile::E2B => GemmaMultimodalFamily::Gemma4E2B,
            Gemma4VisionProfile::E4B => GemmaMultimodalFamily::Gemma4E4B,
            Gemma4VisionProfile::ThirtyOneB => GemmaMultimodalFamily::Gemma4ThirtyOneB,
        };
        Self::checked(
            family,
            tokenizer,
            decoder,
            NativeGemmaVisionResource::Gemma4(vision),
            audio,
            false,
            cancellation,
        )
    }

    fn checked(
        family: GemmaMultimodalFamily,
        tokenizer: Arc<NativePromptTokenizer>,
        decoder: Arc<NativeDecoderTextEncoder>,
        vision: NativeGemmaVisionResource,
        audio: Option<Arc<NativeGemma4AudioEncoder>>,
        source_exact: bool,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        let resource = Self {
            family,
            tokenizer,
            decoder,
            vision,
            audio,
        };
        resource.validate(cancellation)?;
        if resource.is_source_exact_profile() != source_exact {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma retained components mix production and reduced profiles",
            ));
        }
        Ok(resource)
    }

    pub const fn family(&self) -> GemmaMultimodalFamily {
        self.family
    }

    pub fn tokenizer(&self) -> &Arc<NativePromptTokenizer> {
        &self.tokenizer
    }

    pub fn decoder(&self) -> &Arc<NativeDecoderTextEncoder> {
        &self.decoder
    }

    pub fn gemma3_vision(&self) -> Option<&Arc<NativeGemma3VisionProjector>> {
        match &self.vision {
            NativeGemmaVisionResource::Gemma3(vision) => Some(vision),
            NativeGemmaVisionResource::Gemma4(_) => None,
        }
    }

    pub fn gemma4_vision(&self) -> Option<&Arc<NativeGemma4VisionEncoder>> {
        match &self.vision {
            NativeGemmaVisionResource::Gemma3(_) => None,
            NativeGemmaVisionResource::Gemma4(vision) => Some(vision),
        }
    }

    pub fn audio(&self) -> Option<&Arc<NativeGemma4AudioEncoder>> {
        self.audio.as_ref()
    }

    pub(crate) fn execution_requirements(&self, dtype: DType) -> NativeExecutionRequirements {
        let mut requirements = self.decoder.execution_requirements();
        if let Some(vision) = self.gemma4_vision() {
            requirements.extend(vision.execution_requirements(dtype).iter());
        }
        if let Some(audio) = &self.audio {
            requirements.extend(audio.execution_requirements(dtype).iter());
        }
        requirements
    }

    pub fn reconstruct_from_mapped_weights(
        &self,
        mapped: &MappedModelWeights,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, MultimodalTextError> {
        cancellation.check()?;
        let NativeGemmaVisionResource::Gemma4(vision) = &self.vision else {
            return Err(MultimodalTextError::InvalidInput(
                "scheduled CLIP reconstruction is pinned to the Gemma4 multimodal owner",
            ));
        };
        let (decoder_weights, vision_weights, audio_weights) =
            split_gemma4_multimodal_weights(mapped, self.audio.is_some())?;
        let decoder = Arc::new(
            self.decoder
                .reconstruct_from_mapped_weights(&decoder_weights, cancellation)?,
        );
        let vision =
            Arc::new(vision.reconstruct_from_mapped_weights(&vision_weights, cancellation)?);
        let audio = match (&self.audio, audio_weights) {
            (Some(owner), Some(weights)) => Some(Arc::new(
                owner.reconstruct_from_mapped_weights(&weights, cancellation)?,
            )),
            (None, None) => None,
            _ => {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma4 scheduled weights and retained audio capability disagree",
                ));
            }
        };
        Self::checked(
            self.family,
            self.tokenizer.clone(),
            decoder,
            NativeGemmaVisionResource::Gemma4(vision),
            audio,
            self.is_source_exact_profile(),
            cancellation,
        )
    }

    pub fn is_source_exact_profile(&self) -> bool {
        let decoder_exact = self.decoder.configuration() == &self.family.decoder_configuration();
        let vision_exact = match (&self.vision, self.family) {
            (
                NativeGemmaVisionResource::Gemma3(vision),
                GemmaMultimodalFamily::Gemma3FourBVision,
            ) => {
                vision.configuration()
                    == &Gemma3VisionConfiguration::source(Gemma3VisionProfile::FourBVision)
            }
            (NativeGemmaVisionResource::Gemma3(vision), GemmaMultimodalFamily::Gemma3TwelveB) => {
                vision.configuration()
                    == &Gemma3VisionConfiguration::source(Gemma3VisionProfile::TwelveB)
            }
            (NativeGemmaVisionResource::Gemma4(vision), family) => {
                let expected = match family {
                    GemmaMultimodalFamily::Gemma4E2B => Some(Gemma4VisionProfile::E2B),
                    GemmaMultimodalFamily::Gemma4E4B => Some(Gemma4VisionProfile::E4B),
                    GemmaMultimodalFamily::Gemma4ThirtyOneB => {
                        Some(Gemma4VisionProfile::ThirtyOneB)
                    }
                    GemmaMultimodalFamily::Gemma3FourBVision
                    | GemmaMultimodalFamily::Gemma3TwelveB => None,
                };
                expected.is_some_and(|profile| {
                    vision.configuration() == &Gemma4VisionConfiguration::source(profile)
                })
            }
            _ => false,
        };
        let audio_exact = match (self.family, &self.audio) {
            (GemmaMultimodalFamily::Gemma4E2B, Some(audio)) => {
                Gemma4AudioConfiguration::source(Gemma4AudioProfile::E2B)
                    .is_ok_and(|expected| audio.configuration() == &expected)
            }
            (GemmaMultimodalFamily::Gemma4E4B, Some(audio)) => {
                Gemma4AudioConfiguration::source(Gemma4AudioProfile::E4B)
                    .is_ok_and(|expected| audio.configuration() == &expected)
            }
            (GemmaMultimodalFamily::Gemma4ThirtyOneB, None)
            | (GemmaMultimodalFamily::Gemma3FourBVision, None)
            | (GemmaMultimodalFamily::Gemma3TwelveB, None) => true,
            _ => false,
        };
        decoder_exact && vision_exact && audio_exact
    }

    pub fn validate(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<(), MultimodalTextError> {
        cancellation.check()?;
        if self.tokenizer.gemma_profile() != Some(self.family.tokenizer_profile())
            || self.tokenizer.gemma_artifact_digest().is_none()
            || self.tokenizer.has_textual_inversion_embeddings()
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma multimodal resources require a canonical checkpoint-bound Gemma tokenizer",
            ));
        }
        let tokenizer = self.tokenizer.configuration();
        let decoder = self.decoder.configuration();
        if tokenizer.pad_token != 0
            || tokenizer.start_token != Some(2)
            || tokenizer.end_token.is_some()
            || tokenizer.minimum_length != Some(1)
            || tokenizer.minimum_padding.is_some()
            || tokenizer.pad_to_maximum_length
            || !tokenizer.pad_left
            || !tokenizer.disable_weights
            || tokenizer.maximum_word_length != 8
            || tokenizer.embedding_width != Some(decoder.hidden_size)
            || tokenizer.maximum_length < decoder.maximum_tokens
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma multimodal tokenizer sequence and embedding configuration changed",
            ));
        }
        let marker = match self.family.tokenizer_profile() {
            GemmaTokenizerProfile::Gemma3SentencePiece => crate::GEMMA3_IMAGE_TOKEN,
            GemmaTokenizerProfile::Gemma4TokenizerJson => crate::GEMMA4_IMAGE_TOKEN,
        };
        if self
            .tokenizer
            .encode_numeric(
                if self.family.tokenizer_profile() == GemmaTokenizerProfile::Gemma3SentencePiece {
                    "<image_soft_token>"
                } else {
                    "<|image|>"
                },
                cancellation,
            )?
            .last()
            != Some(&marker)
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma tokenizer marker projection changed",
            ));
        }
        let (vision_hidden, vision_stream) = match (&self.vision, self.family) {
            (
                NativeGemmaVisionResource::Gemma3(vision),
                GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB,
            ) => {
                vision.validate(cancellation)?;
                (
                    vision.configuration().output_hidden_size,
                    vision.execution_stream(),
                )
            }
            (
                NativeGemmaVisionResource::Gemma4(vision),
                GemmaMultimodalFamily::Gemma4E2B
                | GemmaMultimodalFamily::Gemma4E4B
                | GemmaMultimodalFamily::Gemma4ThirtyOneB,
            ) => {
                vision.validate(cancellation)?;
                (
                    vision.configuration().output_hidden_size,
                    vision.execution_stream(),
                )
            }
            _ => {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma multimodal family and retained vision owner disagree",
                ));
            }
        };
        if decoder.hidden_size != vision_hidden
            || self.decoder.execution_stream() != vision_stream
            || decoder.architecture != DecoderArchitecture::Gemma
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma decoder and vision execution profiles disagree",
            ));
        }
        match (self.family, &self.audio) {
            (GemmaMultimodalFamily::Gemma4E2B, Some(audio))
                if audio.configuration().profile == Gemma4AudioProfile::E2B =>
            {
                audio.validate(cancellation)?;
                if audio.configuration().output_hidden_size != decoder.hidden_size
                    || audio.execution_stream() != self.decoder.execution_stream()
                {
                    return Err(MultimodalTextError::InvalidInput(
                        "Gemma4 audio and decoder execution profiles disagree",
                    ));
                }
            }
            (GemmaMultimodalFamily::Gemma4E4B, Some(audio))
                if audio.configuration().profile == Gemma4AudioProfile::E4B =>
            {
                audio.validate(cancellation)?;
                if audio.configuration().output_hidden_size != decoder.hidden_size
                    || audio.execution_stream() != self.decoder.execution_stream()
                {
                    return Err(MultimodalTextError::InvalidInput(
                        "Gemma4 audio and decoder execution profiles disagree",
                    ));
                }
            }
            (GemmaMultimodalFamily::Gemma4ThirtyOneB, None)
            | (GemmaMultimodalFamily::Gemma3FourBVision, None)
            | (GemmaMultimodalFamily::Gemma3TwelveB, None) => {}
            _ => {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma multimodal audio capability does not match its family",
                ));
            }
        }
        if self.is_source_exact_profile() {
            if decoder != &self.family.decoder_configuration() {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma decoder configuration does not match the closed source family",
                ));
            }
        } else if !gemma_reduced_decoder_is_compatible(decoder, self.family) {
            return Err(MultimodalTextError::InvalidInput(
                "reduced Gemma decoder is incompatible with its retained family",
            ));
        }
        self.resident_tensor_allocations()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn semantic_state_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, MultimodalTextError> {
        self.validate(cancellation)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.gemma-multimodal-resource.v1");
        hasher.update(b"standard-sim-text-generation-adapter");
        hasher.update(canonical_multimodal_bytes(&self.family)?);
        hasher.update(QWEN_MULTIMODAL_ROUTING_SOURCE_SHA256.as_bytes());
        hasher.update(LLAMA_SOURCE_SHA256.as_bytes());
        hasher.update(SD1_CLIP_SOURCE_SHA256.as_bytes());
        hasher.update(self.family.source_sha256().as_bytes());
        if matches!(
            self.family,
            GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB
        ) {
            hasher.update(SPIECE_TOKENIZER_SOURCE_SHA256.as_bytes());
        }
        hasher.update(
            self.tokenizer
                .gemma_artifact_digest()
                .ok_or(MultimodalTextError::InvalidInput(
                    "Gemma tokenizer artifact identity is unavailable",
                ))?
                .as_bytes(),
        );
        hasher.update(self.tokenizer.semantic_digest(cancellation)?.as_bytes());
        hasher.update(self.decoder.semantic_state_digest(cancellation)?.as_bytes());
        match &self.vision {
            NativeGemmaVisionResource::Gemma3(vision) => {
                hasher.update(vision.semantic_state_digest_sha256().as_bytes())
            }
            NativeGemmaVisionResource::Gemma4(vision) => {
                hasher.update(vision.semantic_state_digest_sha256().as_bytes())
            }
        }
        if let Some(audio) = &self.audio {
            hasher.update(b"audio");
            hasher.update(audio.semantic_state_digest_sha256().as_bytes());
        } else {
            hasher.update(b"no-audio");
        }
        cancellation.check()?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, MultimodalTextError> {
        u64::try_from(mem::size_of::<Self>())
            .map_err(|_| MultimodalTextError::Overflow("Gemma multimodal owner residency"))
    }

    pub fn resident_owned_allocations(
        &self,
    ) -> Result<Vec<NativeMultimodalOwnedAllocation>, MultimodalTextError> {
        let mut allocations = vec![
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Resource,
                address: self as *const Self as usize,
                resident_bytes: self.resident_owned_bytes()?,
            },
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Tokenizer,
                address: Arc::as_ptr(&self.tokenizer) as usize,
                resident_bytes: self.tokenizer.resident_bytes()?,
            },
            NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Decoder,
                address: Arc::as_ptr(&self.decoder) as usize,
                resident_bytes: self.decoder.resident_owned_bytes()?,
            },
        ];
        let (vision_address, vision_bytes) = match &self.vision {
            NativeGemmaVisionResource::Gemma3(vision) => {
                (Arc::as_ptr(vision) as usize, vision.resident_owned_bytes()?)
            }
            NativeGemmaVisionResource::Gemma4(vision) => {
                (Arc::as_ptr(vision) as usize, vision.resident_owned_bytes()?)
            }
        };
        allocations.push(NativeMultimodalOwnedAllocation {
            kind: NativeMultimodalOwnedAllocationKind::Vision,
            address: vision_address,
            resident_bytes: vision_bytes,
        });
        if let Some(audio) = &self.audio {
            allocations.push(NativeMultimodalOwnedAllocation {
                kind: NativeMultimodalOwnedAllocationKind::Audio,
                address: Arc::as_ptr(audio) as usize,
                resident_bytes: audio.resident_owned_bytes()?,
            });
        }
        Ok(allocations)
    }

    pub fn resident_tensor_allocations(
        &self,
    ) -> Result<Vec<(sim_tensor::StorageId, u64)>, MultimodalTextError> {
        let mut allocations = Vec::new();
        for (storage_id, bytes) in self.decoder.resident_tensor_allocations() {
            insert_gemma4_resident_allocation(&mut allocations, storage_id, bytes)?;
        }
        let vision_allocations = match &self.vision {
            NativeGemmaVisionResource::Gemma3(vision) => vision.resident_tensor_allocations()?,
            NativeGemmaVisionResource::Gemma4(vision) => vision.resident_tensor_allocations()?,
        };
        for (storage_id, bytes) in vision_allocations {
            insert_gemma4_resident_allocation(&mut allocations, storage_id, bytes)?;
        }
        if let Some(audio) = &self.audio {
            for (storage_id, bytes) in audio.resident_tensor_allocations()? {
                insert_gemma4_resident_allocation(&mut allocations, storage_id, bytes)?;
            }
        }
        Ok(allocations)
    }

    pub fn resident_bytes(&self) -> Result<u64, MultimodalTextError> {
        let mut bytes = self.resident_owned_bytes()?;
        bytes = bytes.checked_add(self.tokenizer.resident_bytes()?).ok_or(
            MultimodalTextError::Overflow("Gemma multimodal retained owner residency"),
        )?;
        bytes = bytes
            .checked_add(self.decoder.resident_owned_bytes()?)
            .ok_or(MultimodalTextError::Overflow(
                "Gemma multimodal retained owner residency",
            ))?;
        bytes = bytes
            .checked_add(match &self.vision {
                NativeGemmaVisionResource::Gemma3(vision) => vision.resident_owned_bytes()?,
                NativeGemmaVisionResource::Gemma4(vision) => vision.resident_owned_bytes()?,
            })
            .ok_or(MultimodalTextError::Overflow(
                "Gemma multimodal vision residency",
            ))?;
        if let Some(audio) = &self.audio {
            bytes = bytes.checked_add(audio.resident_owned_bytes()?).ok_or(
                MultimodalTextError::Overflow("Gemma multimodal audio residency"),
            )?;
        }
        self.resident_tensor_allocations()?
            .into_iter()
            .try_fold(bytes, |total, (_, allocation)| {
                total
                    .checked_add(allocation)
                    .ok_or(MultimodalTextError::Overflow(
                        "Gemma multimodal tensor residency",
                    ))
            })
    }

    pub fn encode_conditioning(
        &self,
        backend: &CpuBackend,
        request: GemmaConditioningRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<GemmaConditioningOutput, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let formatted_prompt = format_gemma_multimodal_prompt(
            self.family,
            request.prompt,
            request.prepared_visuals,
            request.prepared_audio,
            request.use_default_template,
            request.thinking,
            context.cancellation,
        )?;
        let prompt_tokens = self
            .tokenizer
            .encode_numeric(&formatted_prompt, context.cancellation)?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        let mut media = Vec::new();
        media
            .try_reserve_exact(
                request
                    .prepared_visuals
                    .len()
                    .checked_add(usize::from(request.prepared_audio.is_some()))
                    .ok_or(MultimodalTextError::Overflow(
                        "Gemma conditioning projected media",
                    ))?,
            )
            .map_err(|_| MultimodalTextError::Overflow("Gemma conditioning projected media"))?;
        match self.family {
            GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB => {
                let vision = self
                    .gemma3_vision()
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma3 conditioning requires its retained vision owner",
                    ))?;
                for prepared in request.prepared_visuals {
                    context.check()?;
                    let (embedding, tokens) = flatten_gemma3_projection(
                        backend,
                        vision.project(backend, prepared, context)?,
                        context,
                    )?;
                    media.push(GemmaProjectedMedia {
                        marker: i64::from(crate::GEMMA3_IMAGE_TOKEN),
                        source_markers: 1,
                        tokens,
                        embedding,
                    });
                }
            }
            GemmaMultimodalFamily::Gemma4E2B
            | GemmaMultimodalFamily::Gemma4E4B
            | GemmaMultimodalFamily::Gemma4ThirtyOneB => {
                let vision = self
                    .gemma4_vision()
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma4 conditioning requires its retained vision owner",
                    ))?;
                for prepared in request.prepared_visuals {
                    context.check()?;
                    let projection = vision.project(backend, prepared, context)?;
                    let marker = match projection.kind {
                        GemmaPreparedVisualKind::Gemma4Image => crate::GEMMA4_IMAGE_TOKEN,
                        GemmaPreparedVisualKind::Gemma4VideoFrame => crate::GEMMA4_VIDEO_TOKEN,
                        GemmaPreparedVisualKind::Gemma3Image => {
                            return Err(MultimodalTextError::InvalidInput(
                                "Gemma4 conditioning cannot consume Gemma3 visual projection",
                            ));
                        }
                    };
                    media.push(GemmaProjectedMedia {
                        marker: i64::from(marker),
                        source_markers: 1,
                        tokens: projection.tokens,
                        embedding: projection.embedding,
                    });
                }
                if let Some(prepared_audio) = request.prepared_audio {
                    let audio = self
                        .audio
                        .as_ref()
                        .ok_or(MultimodalTextError::InvalidInput(
                            "Gemma family does not admit audio conditioning",
                        ))?;
                    let projection = audio.project(backend, prepared_audio, context)?;
                    media.push(GemmaProjectedMedia {
                        marker: i64::from(crate::GEMMA4_AUDIO_TOKEN),
                        source_markers: prepared_audio.marker_tokens(),
                        tokens: projection.tokens,
                        embedding: projection.embedding,
                    });
                }
            }
        }
        let marker_plan =
            plan_gemma_markers(&prompt_tokens, &media, self.family, context.cancellation)?;
        if marker_plan.expanded_tokens.len() > self.decoder.configuration().maximum_tokens {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma conditioning exceeds the decoder token limit",
            ));
        }
        let text_embeddings =
            self.decoder
                .embed_token_values(backend, &marker_plan.expanded_tokens, context)?;
        let mut projected_embeddings = Vec::new();
        projected_embeddings
            .try_reserve_exact(marker_plan.entries.len())
            .map_err(|_| MultimodalTextError::Overflow("Gemma conditioning media"))?;
        for entry in &marker_plan.entries {
            let projected =
                media
                    .get(entry.media_index)
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma conditioning marker projection is unavailable",
                    ))?;
            projected_embeddings.push(MultimodalImageEmbedding {
                span: entry.span,
                embedding: &projected.embedding,
                deepstack: &[],
            });
        }
        let embeddings =
            join_multimodal_embeddings(backend, &text_embeddings, &projected_embeddings, context)?;
        let causal_positions = decoder_causal_positions(marker_plan.expanded_tokens.len())?;
        let gemma4 = matches!(
            self.family,
            GemmaMultimodalFamily::Gemma4E2B
                | GemmaMultimodalFamily::Gemma4E4B
                | GemmaMultimodalFamily::Gemma4ThirtyOneB
        );
        let output = self.decoder.forward_prepared_all_layers(
            backend,
            crate::DecoderPreparedTextRequest {
                embeddings: &embeddings,
                attention_mask: None,
                rope_positions: DecoderRopePositions::Scalar(&causal_positions),
                causal_positions: &causal_positions,
                cache: None,
                capture_layer: None,
                deepstack: None,
                initial_input_ids: gemma4.then_some(marker_plan.expanded_tokens.as_slice()),
            },
            context,
        )?;
        let hidden = output
            .all_intermediate()
            .ok_or(MultimodalTextError::InvalidInput(
                "Gemma conditioning all-layer output is missing",
            ))?
            .clone();
        let mut mask_values = backend.workspace_vec(context, marker_plan.expanded_tokens.len())?;
        for _ in 0..marker_plan.expanded_tokens.len() {
            mask_values.try_push(1.0)?;
        }
        let attention_mask = qwen_tensor(
            backend,
            &[1, marker_plan.expanded_tokens.len()],
            &mask_values,
            context,
        )?;
        context.check()?;
        Ok(GemmaConditioningOutput {
            hidden,
            attention_mask,
        })
    }

    pub fn generate(
        &self,
        backend: &CpuBackend,
        request: GemmaMultimodalGenerationRequest<'_>,
        context: &ExecutionContext<'_>,
    ) -> Result<NativeTextGenerationResult, MultimodalTextError> {
        context.check()?;
        self.validate(context.cancellation)?;
        let configuration = self.decoder.generation_configuration(&request.text)?;
        let formatted_prompt = format_gemma_multimodal_prompt(
            self.family,
            request.text.formatted_prompt,
            request.prepared_visuals,
            request.prepared_audio,
            request.use_default_template,
            request.thinking,
            context.cancellation,
        )?;
        let prompt_tokens = self
            .tokenizer
            .encode_numeric(&formatted_prompt, context.cancellation)?
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();

        let mut media = Vec::new();
        media
            .try_reserve_exact(
                request
                    .prepared_visuals
                    .len()
                    .checked_add(usize::from(request.prepared_audio.is_some()))
                    .ok_or(MultimodalTextError::Overflow("Gemma projected media"))?,
            )
            .map_err(|_| MultimodalTextError::Overflow("Gemma projected media"))?;
        match self.family {
            GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB => {
                let vision = self
                    .gemma3_vision()
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma3 generation requires the retained Gemma3 vision owner",
                    ))?;
                for prepared in request.prepared_visuals {
                    context.check()?;
                    let projection = vision.project(backend, prepared, context)?;
                    let (embedding, tokens) =
                        flatten_gemma3_projection(backend, projection, context)?;
                    media.push(GemmaProjectedMedia {
                        marker: i64::from(crate::GEMMA3_IMAGE_TOKEN),
                        source_markers: 1,
                        tokens,
                        embedding,
                    });
                }
            }
            GemmaMultimodalFamily::Gemma4E2B
            | GemmaMultimodalFamily::Gemma4E4B
            | GemmaMultimodalFamily::Gemma4ThirtyOneB => {
                let vision = self
                    .gemma4_vision()
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma4 generation requires the retained Gemma4 vision owner",
                    ))?;
                for prepared in request.prepared_visuals {
                    context.check()?;
                    let projection = vision.project(backend, prepared, context)?;
                    let marker = match projection.kind {
                        GemmaPreparedVisualKind::Gemma4Image => crate::GEMMA4_IMAGE_TOKEN,
                        GemmaPreparedVisualKind::Gemma4VideoFrame => crate::GEMMA4_VIDEO_TOKEN,
                        GemmaPreparedVisualKind::Gemma3Image => {
                            return Err(MultimodalTextError::InvalidInput(
                                "Gemma4 generation cannot consume Gemma3 visual projection",
                            ));
                        }
                    };
                    media.push(GemmaProjectedMedia {
                        marker: i64::from(marker),
                        source_markers: 1,
                        tokens: projection.tokens,
                        embedding: projection.embedding,
                    });
                }
                if let Some(prepared_audio) = request.prepared_audio {
                    let audio = self
                        .audio
                        .as_ref()
                        .ok_or(MultimodalTextError::InvalidInput(
                            "Gemma family does not admit audio generation",
                        ))?;
                    let projection = audio.project(backend, prepared_audio, context)?;
                    media.push(GemmaProjectedMedia {
                        marker: i64::from(crate::GEMMA4_AUDIO_TOKEN),
                        source_markers: prepared_audio.marker_tokens(),
                        tokens: projection.tokens,
                        embedding: projection.embedding,
                    });
                }
            }
        }

        let marker_plan =
            plan_gemma_markers(&prompt_tokens, &media, self.family, context.cancellation)?;
        if marker_plan
            .expanded_tokens
            .len()
            .checked_add(configuration.maximum_new_tokens)
            .ok_or(MultimodalTextError::Overflow(
                "Gemma prompt and generation length",
            ))?
            > self.decoder.configuration().maximum_tokens
        {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma prompt and generation exceed the decoder token limit",
            ));
        }
        let text_embeddings =
            self.decoder
                .embed_token_values(backend, &marker_plan.expanded_tokens, context)?;
        let mut projected_embeddings = Vec::new();
        projected_embeddings
            .try_reserve_exact(marker_plan.entries.len())
            .map_err(|_| MultimodalTextError::Overflow("Gemma joined media"))?;
        for entry in &marker_plan.entries {
            let projected =
                media
                    .get(entry.media_index)
                    .ok_or(MultimodalTextError::InvalidInput(
                        "Gemma marker projection is unavailable",
                    ))?;
            projected_embeddings.push(MultimodalImageEmbedding {
                span: entry.span,
                embedding: &projected.embedding,
                deepstack: &[],
            });
        }
        let embeddings =
            join_multimodal_embeddings(backend, &text_embeddings, &projected_embeddings, context)?;
        let causal_positions = decoder_causal_positions(marker_plan.expanded_tokens.len())?;
        let gemma4 = matches!(
            self.family,
            GemmaMultimodalFamily::Gemma4E2B
                | GemmaMultimodalFamily::Gemma4E4B
                | GemmaMultimodalFamily::Gemma4ThirtyOneB
        );
        let sampling_history = if gemma4 {
            marker_plan.expanded_tokens.as_slice()
        } else {
            &[]
        };
        let initial_input_ids = gemma4.then_some(marker_plan.expanded_tokens.as_slice());
        let outcome = self.decoder.generate_prepared(
            backend,
            DecoderPreparedGenerationPrompt {
                embeddings: &embeddings,
                sampling_history,
                attention_mask: None,
                rope_positions: DecoderRopePositions::Scalar(&causal_positions),
                causal_positions: &causal_positions,
                deepstack: None,
                initial_input_ids,
            },
            &configuration,
            request.transaction,
            context,
        )?;
        let mut generated_tokens = Vec::new();
        generated_tokens
            .try_reserve_exact(outcome.generated_tokens.len())
            .map_err(|_| MultimodalTextError::Overflow("Gemma generated tokens"))?;
        for token in outcome.generated_tokens {
            generated_tokens.push(u32::try_from(token).map_err(|_| {
                MultimodalTextError::InvalidInput("Gemma generated token is out of range")
            })?);
        }
        let text = self
            .tokenizer
            .decode_generated(&generated_tokens, context.cancellation)?;
        context.check()?;
        Ok(NativeTextGenerationResult {
            text,
            generated_tokens,
            transaction: outcome.transaction,
        })
    }
}

struct GemmaProjectedMedia {
    marker: i64,
    source_markers: usize,
    tokens: usize,
    embedding: Tensor,
}

struct GemmaMarkerEntry {
    span: MultimodalSpan,
    media_index: usize,
}

struct GemmaMarkerPlan {
    expanded_tokens: Vec<i64>,
    entries: Vec<GemmaMarkerEntry>,
}

pub fn format_gemma_multimodal_prompt(
    family: GemmaMultimodalFamily,
    prompt: &str,
    visuals: &[GemmaPreparedVisual],
    audio: Option<&GemmaPreparedAudio>,
    use_default_template: bool,
    thinking: bool,
    cancellation: &sim_types::CancellationToken,
) -> Result<String, MultimodalTextError> {
    cancellation.check()?;
    match family {
        GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB => {
            if visuals.len() > 1
                || visuals
                    .iter()
                    .any(|visual| visual.kind() != GemmaPreparedVisualKind::Gemma3Image)
                || audio.is_some()
            {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma3 generation admits one image and no video or audio",
                ));
            }
            if !use_default_template || prompt.starts_with("<start_of_turn>") {
                return Ok(prompt.to_owned());
            }
            let template_prefix = if visuals.is_empty() {
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n"
            } else {
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n\n<image_soft_token>"
            };
            let template_suffix = if visuals.is_empty() {
                "<end_of_turn>\n<start_of_turn>model\n"
            } else {
                "<end_of_turn>\n\n<start_of_turn>model\n"
            };
            let capacity = template_prefix
                .len()
                .checked_add(prompt.len())
                .and_then(|value| value.checked_add(template_suffix.len()))
                .ok_or(MultimodalTextError::Overflow("Gemma3 prompt template"))?;
            let mut formatted = String::new();
            formatted
                .try_reserve_exact(capacity)
                .map_err(|_| MultimodalTextError::Overflow("Gemma3 prompt template"))?;
            formatted.push_str(template_prefix);
            formatted.push_str(prompt);
            formatted.push_str(template_suffix);
            cancellation.check()?;
            Ok(formatted)
        }
        GemmaMultimodalFamily::Gemma4E2B
        | GemmaMultimodalFamily::Gemma4E4B
        | GemmaMultimodalFamily::Gemma4ThirtyOneB => {
            let mut visual_kind = None;
            for visual in visuals {
                cancellation.check()?;
                if !matches!(
                    visual.kind(),
                    GemmaPreparedVisualKind::Gemma4Image
                        | GemmaPreparedVisualKind::Gemma4VideoFrame
                ) {
                    return Err(MultimodalTextError::InvalidInput(
                        "Gemma4 generation requires canonical Gemma4 visual preparation",
                    ));
                }
                if visual_kind.is_some_and(|kind| kind != visual.kind()) {
                    return Err(MultimodalTextError::InvalidInput(
                        "Gemma4 generation cannot mix prepared image and video frames",
                    ));
                }
                visual_kind = Some(visual.kind());
                match visual.kind() {
                    GemmaPreparedVisualKind::Gemma4Image
                        if visual.timestamp_seconds().is_some() =>
                    {
                        return Err(MultimodalTextError::InvalidInput(
                            "Gemma4 prepared images cannot carry video timestamps",
                        ));
                    }
                    GemmaPreparedVisualKind::Gemma4VideoFrame
                        if visual.timestamp_seconds().is_none() =>
                    {
                        return Err(MultimodalTextError::InvalidInput(
                            "Gemma4 prepared video frames require timestamps",
                        ));
                    }
                    _ => {}
                }
            }
            if matches!(family, GemmaMultimodalFamily::Gemma4ThirtyOneB) && audio.is_some() {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma4 31B does not admit audio generation",
                ));
            }
            if audio.is_some_and(|prepared| prepared.marker_tokens() == 0) {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma4 audio preparation requires at least one marker token",
                ));
            }
            if !use_default_template || prompt.starts_with("<|turn>") {
                return Ok(prompt.to_owned());
            }
            let audio_markers = audio.map_or(0, GemmaPreparedAudio::marker_tokens);
            let capacity = prompt
                .len()
                .checked_add(160)
                .and_then(|value| value.checked_add(visuals.len().checked_mul(64)?))
                .and_then(|value| value.checked_add(audio_markers.checked_mul(11)?))
                .ok_or(MultimodalTextError::Overflow("Gemma4 prompt template"))?;
            let mut media = String::new();
            media
                .try_reserve_exact(capacity)
                .map_err(|_| MultimodalTextError::Overflow("Gemma4 prompt template"))?;
            match visual_kind {
                Some(GemmaPreparedVisualKind::Gemma4VideoFrame) => {
                    media.push_str("\n\n");
                    for (index, visual) in visuals.iter().enumerate() {
                        cancellation.check()?;
                        if index > 0 {
                            media.push(' ');
                        }
                        let seconds =
                            visual
                                .timestamp_seconds()
                                .ok_or(MultimodalTextError::InvalidInput(
                                    "Gemma4 prepared video timestamp is unavailable",
                                ))?;
                        write!(
                            &mut media,
                            "{:02}:{:02} <|image><|video|><image|>",
                            seconds / 60,
                            seconds % 60
                        )
                        .map_err(|_| {
                            MultimodalTextError::InvalidInput(
                                "Gemma4 video template could not be formatted",
                            )
                        })?;
                    }
                    media.push_str("\n\n");
                }
                Some(GemmaPreparedVisualKind::Gemma4Image) => {
                    media.push_str("\n\n");
                    for (index, _) in visuals.iter().enumerate() {
                        cancellation.check()?;
                        if index > 0 {
                            media.push_str("\n\n\n\n");
                        }
                        media.push_str("<|image><|image|><image|>");
                    }
                    media.push_str("\n\n");
                }
                None => {}
                Some(GemmaPreparedVisualKind::Gemma3Image) => {
                    return Err(MultimodalTextError::InvalidInput(
                        "Gemma4 template cannot contain Gemma3 images",
                    ));
                }
            }
            if let Some(prepared) = audio {
                media.push_str("<|audio>");
                for _ in 0..prepared.marker_tokens() {
                    cancellation.check()?;
                    media.push_str("<|audio|>");
                }
                media.push_str("<audio|>");
            }
            let system = if thinking {
                "<|turn>system\n<|think|><turn|>\n"
            } else {
                ""
            };
            let mut formatted = String::new();
            formatted
                .try_reserve_exact(capacity)
                .map_err(|_| MultimodalTextError::Overflow("Gemma4 prompt template"))?;
            formatted.push_str(system);
            formatted.push_str("<|turn>user\n");
            formatted.push_str(&media);
            formatted.push_str(prompt);
            formatted.push_str("<turn|>\n<|turn>model\n");
            cancellation.check()?;
            Ok(formatted)
        }
    }
}

fn flatten_gemma3_projection(
    backend: &CpuBackend,
    projection: Gemma3VisionProjection,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, usize), MultimodalTextError> {
    let shape = projection.embedding.descriptor().shape();
    if shape.len() != 3
        || shape[0] == 0
        || shape[1] != usize_to_u64(projection.tokens_per_image, "Gemma3 projected image tokens")?
        || shape[2] == 0
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma3 projected image embedding has the wrong shape",
        ));
    }
    let tokens = u64_to_usize(shape[0], "Gemma3 projected image batch")?
        .checked_mul(projection.tokens_per_image)
        .ok_or(MultimodalTextError::Overflow(
            "Gemma3 projected image tokens",
        ))?;
    let embedding = torch_reshape_with_context_exact_native(
        backend,
        &projection.embedding,
        &[
            usize_to_i64(tokens, "Gemma3 projected image tokens")?,
            u64_to_i64(shape[2], "Gemma3 projected image hidden size")?,
        ],
        context,
    )?;
    Ok((embedding, tokens))
}

fn plan_gemma_markers(
    prompt_tokens: &[i64],
    media: &[GemmaProjectedMedia],
    family: GemmaMultimodalFamily,
    cancellation: &sim_types::CancellationToken,
) -> Result<GemmaMarkerPlan, MultimodalTextError> {
    cancellation.check()?;
    let mut removed_markers = 0_usize;
    let mut inserted_tokens = 0_usize;
    for projected in media {
        if projected.source_markers == 0 || projected.tokens == 0 {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma projected media requires nonempty marker and embedding spans",
            ));
        }
        removed_markers = removed_markers
            .checked_add(projected.source_markers)
            .ok_or(MultimodalTextError::Overflow("Gemma marker count"))?;
        inserted_tokens = inserted_tokens
            .checked_add(projected.tokens)
            .ok_or(MultimodalTextError::Overflow("Gemma projected token count"))?;
    }
    let expanded_length = prompt_tokens
        .len()
        .checked_sub(removed_markers)
        .and_then(|value| value.checked_add(inserted_tokens))
        .ok_or(MultimodalTextError::InvalidInput(
            "Gemma prompt does not contain every required media marker",
        ))?;
    let mut expanded_tokens = Vec::new();
    expanded_tokens
        .try_reserve_exact(expanded_length)
        .map_err(|_| MultimodalTextError::Overflow("Gemma expanded prompt"))?;
    let mut consumed = Vec::new();
    consumed
        .try_reserve_exact(media.len())
        .map_err(|_| MultimodalTextError::Overflow("Gemma marker state"))?;
    consumed.resize(media.len(), false);
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(media.len())
        .map_err(|_| MultimodalTextError::Overflow("Gemma marker spans"))?;
    let mut source_index = 0_usize;
    while source_index < prompt_tokens.len() {
        cancellation.check()?;
        let token = *prompt_tokens
            .get(source_index)
            .ok_or(MultimodalTextError::InvalidInput(
                "Gemma prompt token storage is incomplete",
            ))?;
        if gemma_media_marker(family, token) {
            let media_index = media
                .iter()
                .enumerate()
                .find_map(|(index, projected)| {
                    (!consumed[index] && projected.marker == token).then_some(index)
                })
                .ok_or(MultimodalTextError::InvalidInput(
                    "Gemma prompt contains an unmatched media marker",
                ))?;
            let projected = &media[media_index];
            let source_end = source_index
                .checked_add(projected.source_markers)
                .ok_or(MultimodalTextError::Overflow("Gemma media marker run"))?;
            if source_end > prompt_tokens.len()
                || prompt_tokens[source_index..source_end]
                    .iter()
                    .any(|candidate| *candidate != token)
            {
                return Err(MultimodalTextError::InvalidInput(
                    "Gemma media marker run does not match its prepared projection",
                ));
            }
            let start = expanded_tokens.len();
            expanded_tokens.extend(std::iter::repeat_n(0, projected.tokens));
            entries.push(GemmaMarkerEntry {
                span: MultimodalSpan {
                    start,
                    size: projected.tokens,
                    grid_thw: [1, 1, projected.tokens],
                },
                media_index,
            });
            consumed[media_index] = true;
            source_index = source_end;
        } else {
            expanded_tokens.push(token);
            source_index += 1;
        }
    }
    if consumed.iter().any(|value| !*value) || expanded_tokens.len() != expanded_length {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma prepared media does not cover the prompt marker plan",
        ));
    }
    Ok(GemmaMarkerPlan {
        expanded_tokens,
        entries,
    })
}

fn gemma_media_marker(family: GemmaMultimodalFamily, token: i64) -> bool {
    match family {
        GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB => {
            token == i64::from(crate::GEMMA3_IMAGE_TOKEN)
        }
        GemmaMultimodalFamily::Gemma4E2B
        | GemmaMultimodalFamily::Gemma4E4B
        | GemmaMultimodalFamily::Gemma4ThirtyOneB => [
            crate::GEMMA4_IMAGE_TOKEN,
            crate::GEMMA4_VIDEO_TOKEN,
            crate::GEMMA4_AUDIO_TOKEN,
        ]
        .into_iter()
        .map(i64::from)
        .any(|marker| marker == token),
    }
}

fn gemma_reduced_decoder_is_compatible(
    configuration: &DecoderTextConfiguration,
    family: GemmaMultimodalFamily,
) -> bool {
    if configuration.architecture != DecoderArchitecture::Gemma
        || configuration.hidden_size == 0
        || !configuration.query_key_norm
    {
        return false;
    }
    match family {
        GemmaMultimodalFamily::Gemma3FourBVision | GemmaMultimodalFamily::Gemma3TwelveB => {
            configuration.gemma3.is_some() && configuration.gemma4.is_none()
        }
        GemmaMultimodalFamily::Gemma4E2B
        | GemmaMultimodalFamily::Gemma4E4B
        | GemmaMultimodalFamily::Gemma4ThirtyOneB => {
            configuration
                .gemma4
                .as_ref()
                .is_some_and(|gemma4| gemma4.source_profile.is_none())
                && configuration.gemma3.is_none()
        }
    }
}

fn decoder_causal_positions(sequence_length: usize) -> Result<Vec<usize>, MultimodalTextError> {
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(sequence_length)
        .map_err(|_| MultimodalTextError::Overflow("decoder causal positions"))?;
    positions.extend(0..sequence_length);
    Ok(positions)
}

fn qwen_decoder_position_axes(
    positions: &MultimodalPositionIds,
    cancellation: &sim_types::CancellationToken,
) -> Result<Vec<Vec<usize>>, MultimodalTextError> {
    let mut axes = Vec::new();
    axes.try_reserve_exact(3)
        .map_err(|_| MultimodalTextError::Overflow("Qwen position axes"))?;
    for source in [positions.temporal(), positions.height(), positions.width()] {
        cancellation.check()?;
        let mut axis = Vec::new();
        axis.try_reserve_exact(source.len())
            .map_err(|_| MultimodalTextError::Overflow("Qwen position axis"))?;
        for value in source {
            axis.push(usize::try_from(*value).map_err(|_| {
                MultimodalTextError::InvalidInput("Qwen position IDs must be nonnegative")
            })?);
        }
        axes.push(axis);
    }
    Ok(axes)
}

fn qwen_decoder_profile_name(family: QwenVisionFamily) -> &'static str {
    match family {
        QwenVisionFamily::Qwen3Vl4B => "Qwen3VL_4BConfig",
        QwenVisionFamily::Qwen3Vl8B => "Qwen3VL_8BConfig",
        QwenVisionFamily::Qwen35_08B => "qwen35_08b",
        QwenVisionFamily::Qwen35_2B => "qwen35_2b",
        QwenVisionFamily::Qwen35_4B => "qwen35_4b",
        QwenVisionFamily::Qwen35_9B => "qwen35_9b",
        QwenVisionFamily::Qwen35_27B => "qwen35_27b",
    }
}

fn qwen_reduced_decoder_is_compatible(
    configuration: &DecoderTextConfiguration,
    family: QwenVisionFamily,
) -> bool {
    let is_qwen35 = matches!(
        family,
        QwenVisionFamily::Qwen35_08B
            | QwenVisionFamily::Qwen35_2B
            | QwenVisionFamily::Qwen35_4B
            | QwenVisionFamily::Qwen35_9B
            | QwenVisionFamily::Qwen35_27B
    );
    if is_qwen35 {
        configuration.architecture == DecoderArchitecture::Qwen35
            && configuration.query_key_norm
            && configuration.qwen35_linear.is_some()
            && configuration
                .layer_kinds
                .contains(&DecoderLayerKind::LinearAttention)
    } else {
        configuration.architecture == DecoderArchitecture::Llama
            && configuration.query_key_norm
            && configuration.qwen35_linear.is_none()
            && configuration
                .layer_kinds
                .iter()
                .all(|kind| *kind == DecoderLayerKind::FullAttention)
            && configuration.layer_kinds.len() >= family.deepstack_layers().len()
    }
}

impl NativeGemma4ClippedLinear {
    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        context.cancellation.check()?;
        let input_minimum = gemma4_scalar(&self.input_minimum)?;
        let input_maximum = gemma4_scalar(&self.input_maximum)?;
        let output_minimum = gemma4_scalar(&self.output_minimum)?;
        let output_maximum = gemma4_scalar(&self.output_maximum)?;
        if input_minimum > input_maximum || output_minimum > output_maximum {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 clipped-linear bounds are inverted",
            ));
        }
        let mut input_values = tensor_to_f32(backend, input, context)?;
        for value in input_values.iter_mut() {
            *value = value.clamp(input_minimum, input_maximum);
        }
        let input = tensor_from_f32(backend, input.descriptor().shape(), &input_values, context)?;
        let mut linear = self.linear.clone();
        let output = linear.forward_with_context(backend, &input, context)?;
        let mut output_values = tensor_to_f32(backend, &output, context)?;
        for value in output_values.iter_mut() {
            *value = value.clamp(output_minimum, output_maximum);
        }
        context.cancellation.check()?;
        Ok(tensor_from_f32(
            backend,
            output.descriptor().shape(),
            &output_values,
            context,
        )?)
    }

    fn named_tensors(&self) -> [&Tensor; 4] {
        [
            &self.input_minimum,
            &self.input_maximum,
            &self.output_minimum,
            &self.output_maximum,
        ]
    }
}

impl NativeGemma4VisionBlock {
    fn named_modules(&self) -> [(&'static str, &NativeModule); 7] {
        [
            ("query", &self.query.linear),
            ("key", &self.key.linear),
            ("value", &self.value.linear),
            ("attention_output", &self.attention_output.linear),
            ("feed_forward_gate", &self.feed_forward_gate.linear),
            ("feed_forward_up", &self.feed_forward_up.linear),
            ("feed_forward_down", &self.feed_forward_down.linear),
        ]
    }

    fn named_tensors(&self) -> Vec<&Tensor> {
        let mut tensors = vec![
            &self.input_normalization_weight,
            &self.query_normalization_weight,
            &self.key_normalization_weight,
            &self.post_attention_normalization_weight,
            &self.pre_feed_forward_normalization_weight,
            &self.post_feed_forward_normalization_weight,
        ];
        for linear in [
            &self.query,
            &self.key,
            &self.value,
            &self.attention_output,
            &self.feed_forward_gate,
            &self.feed_forward_up,
            &self.feed_forward_down,
        ] {
            tensors.extend(linear.named_tensors());
        }
        tensors
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        patches_high: usize,
        patches_wide: usize,
        valid_patches: usize,
        configuration: &Gemma4VisionConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        let tokens = input
            .descriptor()
            .shape()
            .first()
            .copied()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(MultimodalTextError::InvalidInput(
                "Gemma4 vision block token count is invalid",
            ))?;
        let normalized = gemma4_weighted_rms_norm(
            backend,
            input,
            &self.input_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        let query = self.query.forward(backend, &normalized, context)?;
        let key = self.key.forward(backend, &normalized, context)?;
        let value = self.value.forward(backend, &normalized, context)?;
        let attention = gemma4_vision_attention(
            backend,
            &query,
            &key,
            &value,
            &self.query_normalization_weight,
            &self.key_normalization_weight,
            patches_high,
            patches_wide,
            valid_patches,
            configuration,
            context,
        )?;
        let attention = self
            .attention_output
            .forward(backend, &attention, context)?;
        let attention = gemma4_weighted_rms_norm(
            backend,
            &attention,
            &self.post_attention_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        let residual = qwen_add_tensors(backend, input, &attention, context)?;
        let normalized = gemma4_weighted_rms_norm(
            backend,
            &residual,
            &self.pre_feed_forward_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        let gate = self
            .feed_forward_gate
            .forward(backend, &normalized, context)?;
        let up = self
            .feed_forward_up
            .forward(backend, &normalized, context)?;
        let gate_values = tensor_to_f32(backend, &gate, context)?;
        let up_values = tensor_to_f32(backend, &up, context)?;
        if gate_values.len() != up_values.len() {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 feed-forward projections have inconsistent storage",
            ));
        }
        let mut activated = backend.workspace_vec(context, gate_values.len())?;
        for (gate, up) in gate_values.iter().zip(up_values.iter()) {
            context.cancellation.check()?;
            activated.try_push(gemma4_gelu_tanh(*gate) * *up)?;
        }
        let activated = qwen_tensor(
            backend,
            &[tokens, configuration.intermediate_size],
            &activated,
            context,
        )?;
        let down = self
            .feed_forward_down
            .forward(backend, &activated, context)?;
        let down = gemma4_weighted_rms_norm(
            backend,
            &down,
            &self.post_feed_forward_normalization_weight,
            tokens,
            configuration.hidden_size,
            f32::from_bits(configuration.normalization_epsilon_bits),
            context,
        )?;
        qwen_add_tensors(backend, &residual, &down, context)
    }
}

fn gemma4_audio_feed_forward(
    name: &str,
    configuration: &Gemma4AudioConfiguration,
    weights: Gemma4AudioFeedForwardWeights,
    stream: StreamId,
) -> Result<NativeGemma4AudioFeedForward, MultimodalTextError> {
    gemma4_require_parameter_shape(
        &weights.pre_normalization_weight,
        &[configuration.hidden_size],
        stream,
    )?;
    gemma4_require_parameter_shape(
        &weights.post_normalization_weight,
        &[configuration.hidden_size],
        stream,
    )?;
    Ok(NativeGemma4AudioFeedForward {
        pre_normalization_weight: weights.pre_normalization_weight,
        first: gemma4_clipped_linear(
            &format!("{name}.first"),
            configuration.hidden_size,
            configuration.intermediate_size,
            weights.first,
            stream,
        )?,
        second: gemma4_clipped_linear(
            &format!("{name}.second"),
            configuration.intermediate_size,
            configuration.hidden_size,
            weights.second,
            stream,
        )?,
        post_normalization_weight: weights.post_normalization_weight,
    })
}

fn gemma4_audio_block(
    index: usize,
    configuration: &Gemma4AudioConfiguration,
    weights: Gemma4AudioBlockWeights,
    stream: StreamId,
) -> Result<NativeGemma4AudioBlock, MultimodalTextError> {
    let head_dimension = configuration.hidden_size / configuration.attention_heads;
    for (tensor, width) in [
        (&weights.attention_scale, head_dimension),
        (
            &weights.pre_attention_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.post_attention_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.convolution_pre_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.convolution_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.output_normalization_weight,
            configuration.hidden_size,
        ),
    ] {
        gemma4_require_parameter_shape(tensor, &[width], stream)?;
    }
    gemma4_require_parameter_shape(
        &weights.depthwise_convolution_weight,
        &[
            configuration.hidden_size,
            1,
            configuration.convolution_kernel_size,
        ],
        stream,
    )?;
    let prefix = format!("gemma4_audio.blocks.{index}");
    Ok(NativeGemma4AudioBlock {
        feed_forward_one: gemma4_audio_feed_forward(
            &format!("{prefix}.feed_forward_one"),
            configuration,
            weights.feed_forward_one,
            stream,
        )?,
        query: gemma4_clipped_linear(
            &format!("{prefix}.query"),
            configuration.hidden_size,
            configuration.hidden_size,
            weights.query,
            stream,
        )?,
        key: gemma4_clipped_linear(
            &format!("{prefix}.key"),
            configuration.hidden_size,
            configuration.hidden_size,
            weights.key,
            stream,
        )?,
        value: gemma4_clipped_linear(
            &format!("{prefix}.value"),
            configuration.hidden_size,
            configuration.hidden_size,
            weights.value,
            stream,
        )?,
        attention_output: gemma4_clipped_linear(
            &format!("{prefix}.attention_output"),
            configuration.hidden_size,
            configuration.hidden_size,
            weights.attention_output,
            stream,
        )?,
        attention_scale: weights.attention_scale,
        relative_key_projection: gemma4_linear_module(
            &format!("{prefix}.relative_key_projection"),
            configuration.hidden_size,
            configuration.hidden_size,
            weights.relative_key_projection_weight,
            stream,
        )?,
        pre_attention_normalization_weight: weights.pre_attention_normalization_weight,
        post_attention_normalization_weight: weights.post_attention_normalization_weight,
        convolution_pre_normalization_weight: weights.convolution_pre_normalization_weight,
        convolution_start: gemma4_clipped_linear(
            &format!("{prefix}.convolution_start"),
            configuration.hidden_size,
            configuration
                .hidden_size
                .checked_mul(2)
                .ok_or(MultimodalTextError::Overflow(
                    "Gemma4 audio convolution gate width",
                ))?,
            weights.convolution_start,
            stream,
        )?,
        depthwise_convolution_weight: weights.depthwise_convolution_weight,
        convolution_normalization_weight: weights.convolution_normalization_weight,
        convolution_end: gemma4_clipped_linear(
            &format!("{prefix}.convolution_end"),
            configuration.hidden_size,
            configuration.hidden_size,
            weights.convolution_end,
            stream,
        )?,
        feed_forward_two: gemma4_audio_feed_forward(
            &format!("{prefix}.feed_forward_two"),
            configuration,
            weights.feed_forward_two,
            stream,
        )?,
        output_normalization_weight: weights.output_normalization_weight,
    })
}

fn gemma4_audio_subsample_mask(mask: &[u8]) -> Result<Vec<u8>, MultimodalTextError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(mask.len().div_ceil(2))
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 audio subsampled mask"))?;
    output.extend(mask.iter().step_by(2).copied());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn gemma4_audio_conv2d_layer(
    backend: &CpuBackend,
    input: &[f32],
    input_channels: usize,
    input_frames: usize,
    input_bins: usize,
    weights: &[f32],
    output_channels: usize,
    normalization_weight: &[f32],
    mask: &[u8],
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<(CpuWorkspaceVec<f32>, usize, usize), MultimodalTextError> {
    if input.len()
        != input_channels
            .checked_mul(input_frames)
            .and_then(|value| value.checked_mul(input_bins))
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 audio convolution input",
            ))?
        || weights.len()
            != output_channels
                .checked_mul(input_channels)
                .and_then(|value| value.checked_mul(9))
                .ok_or(MultimodalTextError::Overflow(
                    "Gemma4 audio convolution weights",
                ))?
        || normalization_weight.len() != output_channels
        || mask.len() != input_frames
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 audio convolution storage is malformed",
        ));
    }
    let output_frames = input_frames.div_ceil(2);
    let output_bins = input_bins.div_ceil(2);
    let mut output = backend.workspace_vec(
        context,
        output_channels
            .checked_mul(output_frames)
            .and_then(|value| value.checked_mul(output_bins))
            .ok_or(MultimodalTextError::Overflow(
                "Gemma4 audio convolution output",
            ))?,
    )?;
    for channel in 0..output_channels {
        for frame in 0..output_frames {
            for bin in 0..output_bins {
                context.cancellation.check()?;
                let mut sum = 0.0_f32;
                for input_channel in 0..input_channels {
                    for kernel_frame in 0..3 {
                        for kernel_bin in 0..3 {
                            let source_frame = frame * 2 + kernel_frame;
                            let source_bin = bin * 2 + kernel_bin;
                            let Some(source_frame) = source_frame.checked_sub(1) else {
                                continue;
                            };
                            let Some(source_bin) = source_bin.checked_sub(1) else {
                                continue;
                            };
                            if source_frame >= input_frames || source_bin >= input_bins {
                                continue;
                            }
                            let source = (input_channel * input_frames + source_frame) * input_bins
                                + source_bin;
                            let weight =
                                ((channel * input_channels + input_channel) * 3 + kernel_frame) * 3
                                    + kernel_bin;
                            let mask_value = if mask[source_frame] == 0 { 0.0 } else { 1.0 };
                            sum += input[source] * mask_value * weights[weight];
                        }
                    }
                }
                output.try_push(sum)?;
            }
        }
    }
    for frame in 0..output_frames {
        for bin in 0..output_bins {
            context.cancellation.check()?;
            let mut mean = 0.0_f32;
            for channel in 0..output_channels {
                mean += output[(channel * output_frames + frame) * output_bins + bin];
            }
            mean /= output_channels as f32;
            let mut variance = 0.0_f32;
            for channel in 0..output_channels {
                let value = output[(channel * output_frames + frame) * output_bins + bin] - mean;
                variance += value * value;
            }
            variance /= output_channels as f32;
            let inverse = (variance + epsilon).sqrt().recip();
            for channel in 0..output_channels {
                let index = (channel * output_frames + frame) * output_bins + bin;
                output[index] =
                    ((output[index] - mean) * inverse * normalization_weight[channel]).max(0.0);
            }
        }
    }
    Ok((output, output_frames, output_bins))
}

fn gemma4_audio_flatten_convolution(
    backend: &CpuBackend,
    values: &[f32],
    frames: usize,
    bins: usize,
    channels: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let mut output = backend.workspace_vec(context, values.len())?;
    for frame in 0..frames {
        context.cancellation.check()?;
        for bin in 0..bins {
            for channel in 0..channels {
                output.try_push(values[(channel * frames + frame) * bins + bin])?;
            }
        }
    }
    qwen_tensor(backend, &[frames, bins * channels], &output, context)
}

fn gemma4_audio_relative_encoding(
    backend: &CpuBackend,
    configuration: &Gemma4AudioConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let half = configuration.hidden_size / 2;
    let logarithmic_increment = 10_000.0_f32.ln() / (half.saturating_sub(1).max(1)) as f32;
    let positions = configuration.attention_chunk_size + 1;
    let mut output = backend.workspace_vec(context, positions * configuration.hidden_size)?;
    for position in (0..positions).rev() {
        context.cancellation.check()?;
        for timescale in 0..half {
            let angle = position as f32 * (-(timescale as f32) * logarithmic_increment).exp();
            output.try_push(angle.sin())?;
        }
        for timescale in 0..half {
            let angle = position as f32 * (-(timescale as f32) * logarithmic_increment).exp();
            output.try_push(angle.cos())?;
        }
    }
    qwen_tensor(
        backend,
        &[positions, configuration.hidden_size],
        &output,
        context,
    )
}

fn gemma4_audio_attention(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &[u8],
    block: &NativeGemma4AudioBlock,
    configuration: &Gemma4AudioConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let tokens = mask.len();
    let heads = configuration.attention_heads;
    let head_dimension = configuration.hidden_size / heads;
    let query = tensor_to_f32(
        backend,
        &block.query.forward(backend, input, context)?,
        context,
    )?;
    let key = tensor_to_f32(
        backend,
        &block.key.forward(backend, input, context)?,
        context,
    )?;
    let value = tensor_to_f32(
        backend,
        &block.value.forward(backend, input, context)?,
        context,
    )?;
    let scales = tensor_to_f32(backend, &block.attention_scale, context)?;
    let relative = gemma4_audio_relative_encoding(backend, configuration, context)?;
    let mut relative_key_projection = block.relative_key_projection.clone();
    let relative = tensor_to_f32(
        backend,
        &relative_key_projection.forward_with_context(backend, &relative, context)?,
        context,
    )?;
    let chunk = configuration.attention_chunk_size;
    let past = configuration.attention_context_left - 1;
    let future = configuration.attention_context_right;
    let context_size = chunk + past + future;
    let relative_positions = chunk + 1;
    let blocks = tokens.div_ceil(chunk);
    let q_scale = (head_dimension as f32).powf(-0.5) / 2.0_f32.ln();
    let k_scale = (1.0_f32 + std::f32::consts::E).ln() / 2.0_f32.ln();
    let softcap = f32::from_bits(configuration.attention_logit_cap_bits);
    let mut output = backend.workspace_vec(context, tokens * configuration.hidden_size)?;
    for _ in 0..tokens * configuration.hidden_size {
        output.try_push(0.0)?;
    }
    let mut logits = backend.workspace_vec(context, context_size)?;
    for _ in 0..context_size {
        logits.try_push(0.0)?;
    }
    for block_index in 0..blocks {
        for query_local in 0..chunk {
            let query_index = block_index * chunk + query_local;
            if query_index >= tokens {
                continue;
            }
            for head in 0..heads {
                context.cancellation.check()?;
                let mut maximum = f32::NEG_INFINITY;
                for context_index in 0..context_size {
                    let key_signed = (block_index * chunk + context_index) as i128 - past as i128;
                    let valid_key = usize::try_from(key_signed).ok().filter(|key| *key < tokens);
                    let attend = valid_key.is_some_and(|key_index| {
                        let distance = query_index as i128 - key_index as i128;
                        ((distance >= 0 && distance < past as i128)
                            || (distance < 0 && -distance < future as i128))
                            && mask[key_index] != 0
                    });
                    let mut score = if let Some(key_index) = valid_key.filter(|_| attend) {
                        let mut content = 0.0_f32;
                        for dimension in 0..head_dimension {
                            let query_offset = query_index * configuration.hidden_size
                                + head * head_dimension
                                + dimension;
                            let key_offset = key_index * configuration.hidden_size
                                + head * head_dimension
                                + dimension;
                            let softplus = (1.0 + scales[dimension].exp()).ln();
                            content += query[query_offset]
                                * q_scale
                                * softplus
                                * key[key_offset]
                                * k_scale;
                        }
                        let flattened = query_local * context_size + context_index;
                        let source_row = flattened / (context_size + 1);
                        let source_column = flattened % (context_size + 1);
                        let mut position = 0.0_f32;
                        if source_row < chunk && source_column < relative_positions {
                            for dimension in 0..head_dimension {
                                let query_offset = query_index * configuration.hidden_size
                                    + head * head_dimension
                                    + dimension;
                                let relative_offset = source_column * configuration.hidden_size
                                    + head * head_dimension
                                    + dimension;
                                let softplus = (1.0 + scales[dimension].exp()).ln();
                                position += query[query_offset]
                                    * q_scale
                                    * softplus
                                    * relative[relative_offset];
                            }
                        }
                        let raw = content + position;
                        (raw / softcap).tanh() * softcap
                    } else {
                        -1.0e9
                    };
                    if !score.is_finite() {
                        score = -1.0e9;
                    }
                    maximum = maximum.max(score);
                    logits[context_index] = score;
                }
                let denominator = logits
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .sum::<f32>();
                for context_index in 0..context_size {
                    let key_signed = (block_index * chunk + context_index) as i128 - past as i128;
                    let Some(key_index) =
                        usize::try_from(key_signed).ok().filter(|key| *key < tokens)
                    else {
                        continue;
                    };
                    let probability = (logits[context_index] - maximum).exp() / denominator;
                    for dimension in 0..head_dimension {
                        output[query_index * configuration.hidden_size
                            + head * head_dimension
                            + dimension] += probability
                            * value[key_index * configuration.hidden_size
                                + head * head_dimension
                                + dimension];
                    }
                }
            }
        }
    }
    let output = qwen_tensor(
        backend,
        &[tokens, configuration.hidden_size],
        &output,
        context,
    )?;
    block.attention_output.forward(backend, &output, context)
}

fn gemma4_audio_convolution(
    backend: &CpuBackend,
    input: &Tensor,
    block: &NativeGemma4AudioBlock,
    configuration: &Gemma4AudioConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let tokens = u64_to_usize(input.descriptor().shape()[0], "Gemma4 audio tokens")?;
    let residual = input.clone();
    let normalized = gemma4_weighted_rms_norm(
        backend,
        input,
        &block.convolution_pre_normalization_weight,
        tokens,
        configuration.hidden_size,
        f32::from_bits(configuration.normalization_epsilon_bits),
        context,
    )?;
    let start = tensor_to_f32(
        backend,
        &block
            .convolution_start
            .forward(backend, &normalized, context)?,
        context,
    )?;
    let weights = tensor_to_f32(backend, &block.depthwise_convolution_weight, context)?;
    let mut convolved = backend.workspace_vec(context, tokens * configuration.hidden_size)?;
    for token in 0..tokens {
        context.cancellation.check()?;
        for feature in 0..configuration.hidden_size {
            let mut sum = 0.0_f32;
            for kernel in 0..configuration.convolution_kernel_size {
                let Some(source) = token
                    .checked_add(kernel + 1)
                    .and_then(|value| value.checked_sub(configuration.convolution_kernel_size))
                else {
                    continue;
                };
                let left = start[source * configuration.hidden_size * 2 + feature];
                let right = start
                    [source * configuration.hidden_size * 2 + configuration.hidden_size + feature];
                sum += left
                    * (1.0 / (1.0 + (-right).exp()))
                    * weights[feature * configuration.convolution_kernel_size + kernel];
            }
            convolved.try_push(sum)?;
        }
    }
    let convolved = qwen_tensor(
        backend,
        &[tokens, configuration.hidden_size],
        &convolved,
        context,
    )?;
    let convolved = gemma4_weighted_rms_norm(
        backend,
        &convolved,
        &block.convolution_normalization_weight,
        tokens,
        configuration.hidden_size,
        f32::from_bits(configuration.normalization_epsilon_bits),
        context,
    )?;
    let mut values = tensor_to_f32(backend, &convolved, context)?;
    for value in values.iter_mut() {
        *value *= 1.0 / (1.0 + (-*value).exp());
    }
    let activated = qwen_tensor(
        backend,
        &[tokens, configuration.hidden_size],
        &values,
        context,
    )?;
    let output = block
        .convolution_end
        .forward(backend, &activated, context)?;
    qwen_add_tensors(backend, &residual, &output, context)
}

fn gemma4_linear_module(
    name: &str,
    input_features: usize,
    output_features: usize,
    weight: Tensor,
    stream: StreamId,
) -> Result<NativeModule, MultimodalTextError> {
    gemma4_require_parameter_shape(&weight, &[output_features, input_features], stream)?;
    let mut module = NativeModule::linear(name, input_features, output_features, false, false)?;
    module.load_dense_parameters(weight, None)?;
    Ok(module)
}

fn gemma4_clipped_linear(
    name: &str,
    input_features: usize,
    output_features: usize,
    weights: Gemma4ClippedLinearWeights,
    stream: StreamId,
) -> Result<NativeGemma4ClippedLinear, MultimodalTextError> {
    for bound in [
        &weights.input_minimum,
        &weights.input_maximum,
        &weights.output_minimum,
        &weights.output_maximum,
    ] {
        gemma4_require_scalar(bound, stream)?;
    }
    if gemma4_scalar(&weights.input_minimum)? > gemma4_scalar(&weights.input_maximum)?
        || gemma4_scalar(&weights.output_minimum)? > gemma4_scalar(&weights.output_maximum)?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 clipped-linear bounds are inverted",
        ));
    }
    Ok(NativeGemma4ClippedLinear {
        linear: gemma4_linear_module(
            name,
            input_features,
            output_features,
            weights.weight,
            stream,
        )?,
        input_minimum: weights.input_minimum,
        input_maximum: weights.input_maximum,
        output_minimum: weights.output_minimum,
        output_maximum: weights.output_maximum,
    })
}

fn gemma4_vision_block(
    index: usize,
    configuration: &Gemma4VisionConfiguration,
    weights: Gemma4VisionBlockWeights,
    stream: StreamId,
) -> Result<NativeGemma4VisionBlock, MultimodalTextError> {
    for (tensor, width) in [
        (
            &weights.input_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.query_normalization_weight,
            configuration.head_dimension,
        ),
        (
            &weights.key_normalization_weight,
            configuration.head_dimension,
        ),
        (
            &weights.post_attention_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.pre_feed_forward_normalization_weight,
            configuration.hidden_size,
        ),
        (
            &weights.post_feed_forward_normalization_weight,
            configuration.hidden_size,
        ),
    ] {
        gemma4_require_parameter_shape(tensor, &[width], stream)?;
    }
    let prefix = format!("gemma4_vision.blocks.{index}");
    let attention_width = configuration
        .attention_heads
        .checked_mul(configuration.head_dimension)
        .ok_or(MultimodalTextError::Overflow(
            "Gemma4 vision attention width",
        ))?;
    Ok(NativeGemma4VisionBlock {
        input_normalization_weight: weights.input_normalization_weight,
        query: gemma4_clipped_linear(
            &format!("{prefix}.query"),
            configuration.hidden_size,
            attention_width,
            weights.query,
            stream,
        )?,
        key: gemma4_clipped_linear(
            &format!("{prefix}.key"),
            configuration.hidden_size,
            attention_width,
            weights.key,
            stream,
        )?,
        value: gemma4_clipped_linear(
            &format!("{prefix}.value"),
            configuration.hidden_size,
            attention_width,
            weights.value,
            stream,
        )?,
        query_normalization_weight: weights.query_normalization_weight,
        key_normalization_weight: weights.key_normalization_weight,
        attention_output: gemma4_clipped_linear(
            &format!("{prefix}.attention_output"),
            attention_width,
            configuration.hidden_size,
            weights.attention_output,
            stream,
        )?,
        post_attention_normalization_weight: weights.post_attention_normalization_weight,
        pre_feed_forward_normalization_weight: weights.pre_feed_forward_normalization_weight,
        feed_forward_gate: gemma4_clipped_linear(
            &format!("{prefix}.feed_forward_gate"),
            configuration.hidden_size,
            configuration.intermediate_size,
            weights.feed_forward_gate,
            stream,
        )?,
        feed_forward_up: gemma4_clipped_linear(
            &format!("{prefix}.feed_forward_up"),
            configuration.hidden_size,
            configuration.intermediate_size,
            weights.feed_forward_up,
            stream,
        )?,
        feed_forward_down: gemma4_clipped_linear(
            &format!("{prefix}.feed_forward_down"),
            configuration.intermediate_size,
            configuration.hidden_size,
            weights.feed_forward_down,
            stream,
        )?,
        post_feed_forward_normalization_weight: weights.post_feed_forward_normalization_weight,
    })
}

fn gemma4_require_parameter_shape(
    tensor: &Tensor,
    expected: &[usize],
    stream: StreamId,
) -> Result<(), MultimodalTextError> {
    let mut expected_shape = Vec::new();
    expected_shape
        .try_reserve_exact(expected.len())
        .map_err(|_| MultimodalTextError::Overflow("Gemma4 parameter shape"))?;
    for dimension in expected {
        expected_shape.push(usize_to_u64(*dimension, "Gemma4 parameter shape")?);
    }
    let descriptor = tensor.descriptor();
    if descriptor.shape() != expected_shape
        || descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != stream
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 vision parameter shape or execution target is invalid",
        ));
    }
    for chunk in tensor
        .contiguous_bytes()?
        .chunks_exact(std::mem::size_of::<f32>())
    {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
            MultimodalTextError::InvalidInput("Gemma4 parameter storage is malformed")
        })?;
        if !f32::from_ne_bytes(bytes).is_finite() {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 learned parameters must be finite",
            ));
        }
    }
    Ok(())
}

fn gemma4_require_scalar(tensor: &Tensor, stream: StreamId) -> Result<(), MultimodalTextError> {
    let descriptor = tensor.descriptor();
    if !(descriptor.shape().is_empty() || descriptor.shape() == [1])
        || descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != stream
        || gemma4_scalar(tensor)?.is_nan()
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 clipped-linear bound must be one non-NaN CPU F32 scalar",
        ));
    }
    Ok(())
}

fn gemma4_scalar(tensor: &Tensor) -> Result<f32, MultimodalTextError> {
    let bytes = tensor.contiguous_bytes()?;
    let bytes: [u8; 4] = bytes.try_into().map_err(|_| {
        MultimodalTextError::InvalidInput("Gemma4 clipped-linear scalar storage is malformed")
    })?;
    Ok(f32::from_ne_bytes(bytes))
}

fn insert_gemma4_resident_allocation(
    allocations: &mut Vec<(sim_tensor::StorageId, u64)>,
    storage_id: sim_tensor::StorageId,
    bytes: u64,
) -> Result<(), MultimodalTextError> {
    if let Some((_, existing_bytes)) = allocations
        .iter()
        .find(|(existing, _)| *existing == storage_id)
    {
        if *existing_bytes != bytes {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 aliased tensor storage has inconsistent residency",
            ));
        }
    } else {
        allocations
            .try_reserve(1)
            .map_err(|_| MultimodalTextError::Overflow("Gemma4 resident allocations"))?;
        allocations.push((storage_id, bytes));
    }
    Ok(())
}

fn gemma4_patchify(
    pixels: &[f32],
    height: usize,
    width: usize,
    patch_size: usize,
    output: &mut [f32],
    cancellation: &sim_types::CancellationToken,
) -> Result<(), MultimodalTextError> {
    let patch_width = 3_usize
        .checked_mul(patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .ok_or(MultimodalTextError::Overflow("Gemma4 patch width"))?;
    let patch_count = (height / patch_size)
        .checked_mul(width / patch_size)
        .ok_or(MultimodalTextError::Overflow("Gemma4 patch count"))?;
    if pixels.len()
        != height
            .checked_mul(width)
            .and_then(|value| value.checked_mul(3))
            .ok_or(MultimodalTextError::Overflow("Gemma4 input pixels"))?
        || output.len()
            < patch_count
                .checked_mul(patch_width)
                .ok_or(MultimodalTextError::Overflow("Gemma4 patch storage"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 patch input storage is malformed",
        ));
    }
    let mut output_index = 0_usize;
    for patch_y in 0..height / patch_size {
        for patch_x in 0..width / patch_size {
            cancellation.check()?;
            for pixel_y in 0..patch_size {
                for pixel_x in 0..patch_size {
                    let source_pixel = (patch_y * patch_size + pixel_y)
                        .checked_mul(width)
                        .and_then(|value| value.checked_add(patch_x * patch_size + pixel_x))
                        .and_then(|value| value.checked_mul(3))
                        .ok_or(MultimodalTextError::Overflow("Gemma4 patch source"))?;
                    for channel in 0..3 {
                        *output.get_mut(output_index).ok_or(
                            MultimodalTextError::InvalidInput(
                                "Gemma4 patch output storage is incomplete",
                            ),
                        )? = *pixels.get(source_pixel + channel).ok_or(
                            MultimodalTextError::InvalidInput("Gemma4 pixel storage is incomplete"),
                        )?;
                        output_index += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemma4_add_positions(
    hidden: &mut [f32],
    positions: &[f32],
    patches_high: usize,
    patches_wide: usize,
    padded_patches: usize,
    hidden_size: usize,
    position_embeddings: usize,
    cancellation: &sim_types::CancellationToken,
) -> Result<(), MultimodalTextError> {
    let patch_count = patches_high
        .checked_mul(patches_wide)
        .ok_or(MultimodalTextError::Overflow("Gemma4 position patches"))?;
    if hidden.len()
        != padded_patches
            .checked_mul(hidden_size)
            .ok_or(MultimodalTextError::Overflow("Gemma4 hidden positions"))?
        || positions.len()
            != 2_usize
                .checked_mul(position_embeddings)
                .and_then(|value| value.checked_mul(hidden_size))
                .ok_or(MultimodalTextError::Overflow("Gemma4 position table"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 position storage is malformed",
        ));
    }
    for token in 0..patch_count {
        cancellation.check()?;
        let y = token / patches_wide;
        let x = token % patches_wide;
        for feature in 0..hidden_size {
            let output = token * hidden_size + feature;
            let x_index = x * hidden_size + feature;
            let y_index = (position_embeddings + y) * hidden_size + feature;
            hidden[output] += positions[x_index] + positions[y_index];
        }
    }
    Ok(())
}

fn gemma4_weighted_rms_norm(
    backend: &CpuBackend,
    input: &Tensor,
    weight: &Tensor,
    rows: usize,
    width: usize,
    epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let values = tensor_to_f32(backend, input, context)?;
    let weights = tensor_to_f32(backend, weight, context)?;
    if values.len()
        != rows
            .checked_mul(width)
            .ok_or(MultimodalTextError::Overflow("Gemma4 RMS values"))?
        || weights.len() != width
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 RMS normalization storage is malformed",
        ));
    }
    let mut output = backend.workspace_vec(context, values.len())?;
    for row in 0..rows {
        context.cancellation.check()?;
        let start = row * width;
        let slice = &values[start..start + width];
        let mean_square =
            slice.iter().fold(0.0_f32, |sum, value| sum + value * value) / width as f32;
        let inverse = (mean_square + epsilon).sqrt().recip();
        for feature in 0..width {
            output.try_push(slice[feature] * inverse * weights[feature])?;
        }
    }
    Ok(tensor_from_f32(
        backend,
        input.descriptor().shape(),
        &output,
        context,
    )?)
}

fn gemma4_parameterless_rms_norm(
    backend: &CpuBackend,
    values: &[f32],
    rows: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, MultimodalTextError> {
    if values.len()
        != rows
            .checked_mul(width)
            .ok_or(MultimodalTextError::Overflow("Gemma4 projector RMS values"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 parameterless RMS storage is malformed",
        ));
    }
    let mut output = backend.workspace_vec(context, values.len())?;
    for row in 0..rows {
        context.cancellation.check()?;
        let start = row * width;
        let slice = &values[start..start + width];
        let mean_square =
            slice.iter().fold(0.0_f32, |sum, value| sum + value * value) / width as f32;
        let inverse = (mean_square + 1.0e-6).sqrt().recip();
        for feature in 0..width {
            output.try_push(slice[feature] * inverse)?;
        }
    }
    Ok(output)
}

fn gemma4_gelu_tanh(value: f32) -> f32 {
    0.5 * value
        * (1.0
            + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044_715 * value * value * value))
                .tanh())
}

fn gemma4_pool(
    backend: &CpuBackend,
    hidden: &[f32],
    patches_high: usize,
    patches_wide: usize,
    hidden_size: usize,
    kernel: usize,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, MultimodalTextError> {
    if kernel == 0
        || !patches_high.is_multiple_of(kernel)
        || !patches_wide.is_multiple_of(kernel)
        || hidden.len()
            < patches_high
                .checked_mul(patches_wide)
                .and_then(|value| value.checked_mul(hidden_size))
                .ok_or(MultimodalTextError::Overflow("Gemma4 pool input"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 pooling geometry is invalid",
        ));
    }
    let pooled_high = patches_high / kernel;
    let pooled_wide = patches_wide / kernel;
    let tokens = pooled_high
        .checked_mul(pooled_wide)
        .ok_or(MultimodalTextError::Overflow("Gemma4 pooled tokens"))?;
    let mut output = backend.workspace_vec(
        context,
        tokens
            .checked_mul(hidden_size)
            .ok_or(MultimodalTextError::Overflow("Gemma4 pool output"))?,
    )?;
    for _ in 0..tokens
        .checked_mul(hidden_size)
        .ok_or(MultimodalTextError::Overflow("Gemma4 pool output"))?
    {
        output.try_push(0.0)?;
    }
    let scale = (hidden_size as f32).sqrt() / (kernel * kernel) as f32;
    for pooled_y in 0..pooled_high {
        for pooled_x in 0..pooled_wide {
            context.cancellation.check()?;
            let token = pooled_y * pooled_wide + pooled_x;
            for kernel_y in 0..kernel {
                for kernel_x in 0..kernel {
                    let source_token = (pooled_y * kernel + kernel_y) * patches_wide
                        + pooled_x * kernel
                        + kernel_x;
                    for feature in 0..hidden_size {
                        output[token * hidden_size + feature] +=
                            hidden[source_token * hidden_size + feature] * scale;
                    }
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn gemma4_vision_attention(
    backend: &CpuBackend,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    query_weight: &Tensor,
    key_weight: &Tensor,
    patches_high: usize,
    patches_wide: usize,
    valid_patches: usize,
    configuration: &Gemma4VisionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let tokens = query
        .descriptor()
        .shape()
        .first()
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(MultimodalTextError::InvalidInput(
            "Gemma4 attention token count is invalid",
        ))?;
    let width = configuration
        .attention_heads
        .checked_mul(configuration.head_dimension)
        .ok_or(MultimodalTextError::Overflow("Gemma4 attention width"))?;
    let expected_shape = [
        usize_to_u64(tokens, "Gemma4 attention tokens")?,
        usize_to_u64(width, "Gemma4 attention width")?,
    ];
    if query.descriptor().shape() != expected_shape
        || key.descriptor().shape() != expected_shape
        || value.descriptor().shape() != expected_shape
        || valid_patches
            != patches_high
                .checked_mul(patches_wide)
                .ok_or(MultimodalTextError::Overflow("Gemma4 valid patches"))?
        || valid_patches > tokens
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 attention geometry is invalid",
        ));
    }
    let query_values = tensor_to_f32(backend, query, context)?;
    let key_values = tensor_to_f32(backend, key, context)?;
    let value_values = tensor_to_f32(backend, value, context)?;
    let query_weight = tensor_to_f32(backend, query_weight, context)?;
    let key_weight = tensor_to_f32(backend, key_weight, context)?;
    let vector_values = tokens
        .checked_mul(width)
        .ok_or(MultimodalTextError::Overflow("Gemma4 attention vectors"))?;
    let mut normalized_query = backend.workspace_vec(context, vector_values)?;
    let mut normalized_key = backend.workspace_vec(context, vector_values)?;
    let mut normalized_value = backend.workspace_vec(context, vector_values)?;
    for _ in 0..vector_values {
        normalized_query.try_push(0.0)?;
        normalized_key.try_push(0.0)?;
        normalized_value.try_push(0.0)?;
    }
    let epsilon = f32::from_bits(configuration.normalization_epsilon_bits);
    for token in 0..tokens {
        context.cancellation.check()?;
        let coordinate = if token < valid_patches {
            Some((token % patches_wide, token / patches_wide))
        } else {
            None
        };
        for head in 0..configuration.attention_heads {
            let start = token * width + head * configuration.head_dimension;
            let end = start + configuration.head_dimension;
            gemma4_normalize_head(
                &query_values[start..end],
                Some(&query_weight),
                epsilon,
                &mut normalized_query[start..end],
            )?;
            gemma4_normalize_head(
                &key_values[start..end],
                Some(&key_weight),
                epsilon,
                &mut normalized_key[start..end],
            )?;
            gemma4_normalize_head(
                &value_values[start..end],
                None,
                epsilon,
                &mut normalized_value[start..end],
            )?;
            let (x, y) = coordinate
                .map(|(x, y)| (x as i64, y as i64))
                .unwrap_or((-1, -1));
            gemma4_apply_vision_rope(
                &mut normalized_query[start..end],
                x,
                y,
                f32::from_bits(configuration.rotary_theta_bits),
            )?;
            gemma4_apply_vision_rope(
                &mut normalized_key[start..end],
                x,
                y,
                f32::from_bits(configuration.rotary_theta_bits),
            )?;
        }
    }
    let mut mask = backend.workspace_vec(context, tokens)?;
    for key_token in 0..tokens {
        mask.try_push(key_token < valid_patches)?;
    }
    let workspace_limit_bytes = tokens
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(MultimodalTextError::Overflow("Gemma4 attention workspace"))?;
    let outcome = scaled_dot_product_attention_with_context(
        backend,
        AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: 1,
            query_tokens: tokens,
            key_tokens: tokens,
            heads: configuration.attention_heads,
            head_dimension: configuration.head_dimension,
            value_dimension: configuration.head_dimension,
            scale: Some(1.0),
            workspace_limit_bytes,
        },
        &normalized_query,
        &normalized_key,
        &normalized_value,
        Some(AttentionMask::Boolean {
            values: &mask,
            shape: AttentionMaskShape::KeyTokens,
        }),
        context,
    )?;
    qwen_tensor(backend, &[tokens, width], &outcome.values, context)
}

fn gemma4_normalize_head(
    input: &[f32],
    weight: Option<&[f32]>,
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), MultimodalTextError> {
    if input.is_empty()
        || input.len() != output.len()
        || weight.is_some_and(|weight| weight.len() != input.len())
    {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 attention head normalization is malformed",
        ));
    }
    let mean_square =
        input.iter().fold(0.0_f32, |sum, value| sum + value * value) / input.len() as f32;
    let inverse = (mean_square + epsilon).sqrt().recip();
    for index in 0..input.len() {
        output[index] = input[index] * inverse * weight.map_or(1.0, |weight| weight[index]);
    }
    Ok(())
}

fn gemma4_apply_vision_rope(
    values: &mut [f32],
    x: i64,
    y: i64,
    theta: f32,
) -> Result<(), MultimodalTextError> {
    if values.is_empty() || !values.len().is_multiple_of(4) || !theta.is_finite() || theta <= 0.0 {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 vision rotary geometry is invalid",
        ));
    }
    let axis_width = values.len() / 2;
    let pair_width = axis_width / 2;
    for (axis, coordinate) in [(0_usize, x), (1_usize, y)] {
        let axis_start = axis * axis_width;
        for frequency in 0..pair_width {
            let exponent = (frequency * 2) as f32 / axis_width as f32;
            let angle = coordinate as f32 / theta.powf(exponent);
            let cosine = angle.cos();
            let sine = angle.sin();
            let first_index = axis_start + frequency;
            let second_index = first_index + pair_width;
            let first = values[first_index];
            let second = values[second_index];
            values[first_index] = first * cosine - second * sine;
            values[second_index] = second * cosine + first * sine;
        }
    }
    Ok(())
}

impl NativeQwenVisionBlock {
    fn modules(&self) -> [(&'static str, &NativeModule); 7] {
        [
            ("normalization_one", &self.normalization_one),
            ("query_key_value", &self.query_key_value),
            ("attention_output", &self.attention_output),
            ("normalization_two", &self.normalization_two),
            ("feed_forward_up", &self.feed_forward_up),
            ("feed_forward_activation", &self.feed_forward_activation),
            ("feed_forward_down", &self.feed_forward_down),
        ]
    }

    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        grid_thw: [usize; 3],
        configuration: &QwenVisionConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        let mut normalization_one = self.normalization_one.clone();
        let normalized = normalization_one.forward_with_context(backend, input, context)?;
        let mut query_key_value = self.query_key_value.clone();
        let projected = query_key_value.forward_with_context(backend, &normalized, context)?;
        let projected = tensor_to_f32(backend, &projected, context)?;
        let attention = qwen_vision_attention(
            backend,
            &projected,
            grid_thw,
            configuration.hidden_size,
            configuration.attention_heads,
            context,
        )?;
        let attention = qwen_tensor(
            backend,
            &[
                grid_thw[0] * grid_thw[1] * grid_thw[2],
                configuration.hidden_size,
            ],
            &attention,
            context,
        )?;
        let mut attention_output = self.attention_output.clone();
        let attention = attention_output.forward_with_context(backend, &attention, context)?;
        let residual = qwen_add_tensors(backend, input, &attention, context)?;
        let mut normalization_two = self.normalization_two.clone();
        let normalized = normalization_two.forward_with_context(backend, &residual, context)?;
        let mut feed_forward_up = self.feed_forward_up.clone();
        let up = feed_forward_up.forward_with_context(backend, &normalized, context)?;
        let mut activation = self.feed_forward_activation.clone();
        let up = activation.forward_with_context(backend, &up, context)?;
        let mut feed_forward_down = self.feed_forward_down.clone();
        let down = feed_forward_down.forward_with_context(backend, &up, context)?;
        qwen_add_tensors(backend, &residual, &down, context)
    }
}

impl NativeQwenVisionMerger {
    fn modules(&self) -> [(&'static str, &NativeModule); 4] {
        [
            ("normalization", &self.normalization),
            ("first", &self.first),
            ("activation", &self.activation),
            ("second", &self.second),
        ]
    }

    fn forward(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        configuration: &QwenVisionConfiguration,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, MultimodalTextError> {
        let shape = input.descriptor().shape();
        let tokens = shape
            .first()
            .copied()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen merger input token count is invalid",
            ))?;
        let merge_unit = configuration
            .spatial_merge_size
            .checked_mul(configuration.spatial_merge_size)
            .ok_or(MultimodalTextError::Overflow("Qwen merger unit"))?;
        if shape
            != [
                usize_to_u64(tokens, "Qwen merger tokens")?,
                usize_to_u64(configuration.hidden_size, "Qwen merger hidden")?,
            ]
            || !tokens.is_multiple_of(merge_unit)
        {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen merger input does not contain complete spatial groups",
            ));
        }
        let merge_width = configuration
            .hidden_size
            .checked_mul(merge_unit)
            .ok_or(MultimodalTextError::Overflow("Qwen merger width"))?;
        let mut normalization = self.normalization.clone();
        let merged = if self.normalization_after_merge {
            let reshaped = qwen_tensor(
                backend,
                &[tokens / merge_unit, merge_width],
                &tensor_to_f32(backend, input, context)?,
                context,
            )?;
            normalization.forward_with_context(backend, &reshaped, context)?
        } else {
            let normalized = normalization.forward_with_context(backend, input, context)?;
            qwen_tensor(
                backend,
                &[tokens / merge_unit, merge_width],
                &tensor_to_f32(backend, &normalized, context)?,
                context,
            )?
        };
        let mut first = self.first.clone();
        let hidden = first.forward_with_context(backend, &merged, context)?;
        let mut activation = self.activation.clone();
        let hidden = activation.forward_with_context(backend, &hidden, context)?;
        let mut second = self.second.clone();
        Ok(second.forward_with_context(backend, &hidden, context)?)
    }
}

fn qwen_linear_module(
    name: &str,
    input_features: usize,
    output_features: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    stream: sim_tensor::StreamId,
) -> Result<NativeModule, MultimodalTextError> {
    qwen_require_parameter_shape(&weight, &[output_features, input_features], stream)?;
    if let Some(bias) = bias.as_ref() {
        qwen_require_parameter_shape(bias, &[output_features], stream)?;
    }
    let mut module =
        NativeModule::linear(name, input_features, output_features, bias.is_some(), false)?;
    module.load_dense_parameters(weight, bias)?;
    Ok(module)
}

fn split_multimodal_weights(
    mapped: &MappedModelWeights,
) -> Result<(MappedModelWeights, MappedModelWeights), MultimodalTextError> {
    if !mapped.unexpected_keys().is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "multimodal mapped weights contain unexpected parameters",
        ));
    }
    let mut decoder = BTreeMap::new();
    let mut vision = BTreeMap::new();
    for (name, tensor) in mapped.tensors() {
        if let Some(name) = name.strip_prefix("decoder.") {
            decoder.insert(name.to_owned(), tensor.clone());
        } else if let Some(name) = name.strip_prefix("vision.") {
            vision.insert(name.to_owned(), tensor.clone());
        } else {
            return Err(MultimodalTextError::InvalidInput(
                "multimodal mapped weight has no canonical owner prefix",
            ));
        }
    }
    if decoder.is_empty() || vision.is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "multimodal mapped weights omit a canonical owner",
        ));
    }
    Ok((
        MappedModelWeights::from_parts(
            mapped.base_artifact_digest().to_owned(),
            decoder,
            Vec::new(),
        ),
        MappedModelWeights::from_parts(
            mapped.base_artifact_digest().to_owned(),
            vision,
            Vec::new(),
        ),
    ))
}

fn canonical_multimodal_bytes(value: &impl Serialize) -> Result<Vec<u8>, MultimodalTextError> {
    serde_json::to_vec(value).map_err(|_| {
        MultimodalTextError::InvalidInput(
            "multimodal configuration cannot be canonically serialized",
        )
    })
}

fn split_gemma4_multimodal_weights(
    mapped: &MappedModelWeights,
    audio_required: bool,
) -> Result<
    (
        MappedModelWeights,
        MappedModelWeights,
        Option<MappedModelWeights>,
    ),
    MultimodalTextError,
> {
    if !mapped.unexpected_keys().is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 mapped weights contain unexpected parameters",
        ));
    }
    let mut decoder = BTreeMap::new();
    let mut vision = BTreeMap::new();
    let mut audio = BTreeMap::new();
    for (name, tensor) in mapped.tensors() {
        if let Some(name) = name.strip_prefix("decoder.") {
            decoder.insert(name.to_owned(), tensor.clone());
        } else if let Some(name) = name.strip_prefix("vision.") {
            vision.insert(name.to_owned(), tensor.clone());
        } else if let Some(name) = name.strip_prefix("audio.") {
            audio.insert(name.to_owned(), tensor.clone());
        } else {
            return Err(MultimodalTextError::InvalidInput(
                "Gemma4 mapped weight has no canonical owner prefix",
            ));
        }
    }
    if decoder.is_empty() || vision.is_empty() || audio_required != !audio.is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "Gemma4 mapped weights omit or add a canonical retained owner",
        ));
    }
    let artifact = mapped.base_artifact_digest().to_owned();
    Ok((
        MappedModelWeights::from_parts(artifact.clone(), decoder, Vec::new()),
        MappedModelWeights::from_parts(artifact.clone(), vision, Vec::new()),
        (!audio.is_empty()).then(|| MappedModelWeights::from_parts(artifact, audio, Vec::new())),
    ))
}

fn take_multimodal_tensor(
    mapped: &MappedModelWeights,
    name: &str,
    consumed: &mut BTreeSet<String>,
) -> Result<Tensor, MultimodalTextError> {
    let tensor = mapped
        .tensors()
        .get(name)
        .cloned()
        .ok_or(MultimodalTextError::InvalidInput(
            "multimodal mapped weight is missing",
        ))?;
    consumed.insert(name.to_owned());
    Ok(tensor)
}

fn reload_multimodal_module(
    module: &mut NativeModule,
    name: &str,
    mapped: &MappedModelWeights,
    consumed: &mut BTreeSet<String>,
) -> Result<(), MultimodalTextError> {
    let (_, existing_bias) = module.dense_parameters()?;
    let weight = take_multimodal_tensor(mapped, &format!("{name}.weight"), consumed)?;
    let bias = if existing_bias.is_some() {
        Some(take_multimodal_tensor(
            mapped,
            &format!("{name}.bias"),
            consumed,
        )?)
    } else {
        None
    };
    module.load_dense_parameters(weight, bias)?;
    Ok(())
}

fn reload_gemma4_clipped_linear(
    linear: &mut NativeGemma4ClippedLinear,
    prefix: &str,
    mapped: &MappedModelWeights,
    consumed: &mut BTreeSet<String>,
) -> Result<(), MultimodalTextError> {
    reload_multimodal_module(&mut linear.linear, prefix, mapped, consumed)?;
    linear.input_minimum =
        take_multimodal_tensor(mapped, &format!("{prefix}.input_minimum"), consumed)?;
    linear.input_maximum =
        take_multimodal_tensor(mapped, &format!("{prefix}.input_maximum"), consumed)?;
    linear.output_minimum =
        take_multimodal_tensor(mapped, &format!("{prefix}.output_minimum"), consumed)?;
    linear.output_maximum =
        take_multimodal_tensor(mapped, &format!("{prefix}.output_maximum"), consumed)?;
    Ok(())
}

fn reload_gemma4_vision_block(
    block: &mut NativeGemma4VisionBlock,
    prefix: &str,
    mapped: &MappedModelWeights,
    consumed: &mut BTreeSet<String>,
) -> Result<(), MultimodalTextError> {
    block.input_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.input_normalization.weight"),
        consumed,
    )?;
    block.query_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.query_normalization.weight"),
        consumed,
    )?;
    block.key_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.key_normalization.weight"),
        consumed,
    )?;
    block.post_attention_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.post_attention_normalization.weight"),
        consumed,
    )?;
    block.pre_feed_forward_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.pre_feed_forward_normalization.weight"),
        consumed,
    )?;
    block.post_feed_forward_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.post_feed_forward_normalization.weight"),
        consumed,
    )?;
    for (suffix, linear) in [
        ("query", &mut block.query),
        ("key", &mut block.key),
        ("value", &mut block.value),
        ("attention_output", &mut block.attention_output),
        ("feed_forward_gate", &mut block.feed_forward_gate),
        ("feed_forward_up", &mut block.feed_forward_up),
        ("feed_forward_down", &mut block.feed_forward_down),
    ] {
        reload_gemma4_clipped_linear(linear, &format!("{prefix}.{suffix}"), mapped, consumed)?;
    }
    Ok(())
}

fn reload_gemma4_audio_feed_forward(
    feed_forward: &mut NativeGemma4AudioFeedForward,
    prefix: &str,
    mapped: &MappedModelWeights,
    consumed: &mut BTreeSet<String>,
) -> Result<(), MultimodalTextError> {
    feed_forward.pre_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.pre_normalization.weight"),
        consumed,
    )?;
    feed_forward.post_normalization_weight = take_multimodal_tensor(
        mapped,
        &format!("{prefix}.post_normalization.weight"),
        consumed,
    )?;
    reload_gemma4_clipped_linear(
        &mut feed_forward.first,
        &format!("{prefix}.first"),
        mapped,
        consumed,
    )?;
    reload_gemma4_clipped_linear(
        &mut feed_forward.second,
        &format!("{prefix}.second"),
        mapped,
        consumed,
    )?;
    Ok(())
}

fn reload_gemma4_audio_block(
    block: &mut NativeGemma4AudioBlock,
    prefix: &str,
    mapped: &MappedModelWeights,
    consumed: &mut BTreeSet<String>,
) -> Result<(), MultimodalTextError> {
    reload_gemma4_audio_feed_forward(
        &mut block.feed_forward_one,
        &format!("{prefix}.ff1"),
        mapped,
        consumed,
    )?;
    reload_gemma4_audio_feed_forward(
        &mut block.feed_forward_two,
        &format!("{prefix}.ff2"),
        mapped,
        consumed,
    )?;
    for (suffix, linear) in [
        ("query", &mut block.query),
        ("key", &mut block.key),
        ("value", &mut block.value),
        ("attention_output", &mut block.attention_output),
        ("convolution_start", &mut block.convolution_start),
        ("convolution_end", &mut block.convolution_end),
    ] {
        reload_gemma4_clipped_linear(linear, &format!("{prefix}.{suffix}"), mapped, consumed)?;
    }
    reload_multimodal_module(
        &mut block.relative_key_projection,
        &format!("{prefix}.relative_key_projection"),
        mapped,
        consumed,
    )?;
    for (suffix, target) in [
        ("attention_scale", &mut block.attention_scale),
        (
            "pre_attention_normalization.weight",
            &mut block.pre_attention_normalization_weight,
        ),
        (
            "post_attention_normalization.weight",
            &mut block.post_attention_normalization_weight,
        ),
        (
            "convolution_pre_normalization.weight",
            &mut block.convolution_pre_normalization_weight,
        ),
        (
            "depthwise_convolution.weight",
            &mut block.depthwise_convolution_weight,
        ),
        (
            "convolution_normalization.weight",
            &mut block.convolution_normalization_weight,
        ),
        (
            "output_normalization.weight",
            &mut block.output_normalization_weight,
        ),
    ] {
        *target = take_multimodal_tensor(mapped, &format!("{prefix}.{suffix}"), consumed)?;
    }
    Ok(())
}

fn reload_qwen_merger(
    merger: &mut NativeQwenVisionMerger,
    prefix: &str,
    mapped: &MappedModelWeights,
    consumed: &mut BTreeSet<String>,
) -> Result<(), MultimodalTextError> {
    reload_multimodal_module(
        &mut merger.normalization,
        &format!("{prefix}.normalization"),
        mapped,
        consumed,
    )?;
    reload_multimodal_module(
        &mut merger.first,
        &format!("{prefix}.first"),
        mapped,
        consumed,
    )?;
    reload_multimodal_module(
        &mut merger.second,
        &format!("{prefix}.second"),
        mapped,
        consumed,
    )?;
    Ok(())
}

fn qwen_layer_norm_module(
    name: &str,
    width: usize,
    weight: Tensor,
    bias: Tensor,
    stream: sim_tensor::StreamId,
) -> Result<NativeModule, MultimodalTextError> {
    qwen_require_parameter_shape(&weight, &[width], stream)?;
    qwen_require_parameter_shape(&bias, &[width], stream)?;
    let mut module = NativeModule::layer_norm(name, vec![width], 1.0e-6, true, true, false)?;
    module.load_dense_parameters(weight, Some(bias))?;
    Ok(module)
}

fn qwen_vision_block(
    index: usize,
    configuration: &QwenVisionConfiguration,
    weights: QwenVisionBlockWeights,
    stream: sim_tensor::StreamId,
) -> Result<NativeQwenVisionBlock, MultimodalTextError> {
    let hidden_size = configuration.hidden_size;
    let intermediate_size = configuration.intermediate_size;
    let prefix = format!("qwen_vision.blocks.{index}");
    Ok(NativeQwenVisionBlock {
        normalization_one: qwen_layer_norm_module(
            &format!("{prefix}.norm1"),
            hidden_size,
            weights.normalization_one_weight,
            weights.normalization_one_bias,
            stream,
        )?,
        query_key_value: qwen_linear_module(
            &format!("{prefix}.qkv"),
            hidden_size,
            hidden_size
                .checked_mul(3)
                .ok_or(MultimodalTextError::Overflow("Qwen vision QKV width"))?,
            weights.query_key_value_weight,
            Some(weights.query_key_value_bias),
            stream,
        )?,
        attention_output: qwen_linear_module(
            &format!("{prefix}.attention_output"),
            hidden_size,
            hidden_size,
            weights.attention_output_weight,
            Some(weights.attention_output_bias),
            stream,
        )?,
        normalization_two: qwen_layer_norm_module(
            &format!("{prefix}.norm2"),
            hidden_size,
            weights.normalization_two_weight,
            weights.normalization_two_bias,
            stream,
        )?,
        feed_forward_up: qwen_linear_module(
            &format!("{prefix}.mlp_up"),
            hidden_size,
            intermediate_size,
            weights.feed_forward_up_weight,
            Some(weights.feed_forward_up_bias),
            stream,
        )?,
        feed_forward_activation: NativeModule::gelu(
            format!("{prefix}.mlp_activation"),
            GeluApproximation::Tanh,
        )?,
        feed_forward_down: qwen_linear_module(
            &format!("{prefix}.mlp_down"),
            intermediate_size,
            hidden_size,
            weights.feed_forward_down_weight,
            Some(weights.feed_forward_down_bias),
            stream,
        )?,
    })
}

fn qwen_vision_merger(
    name: &str,
    configuration: &QwenVisionConfiguration,
    weights: QwenVisionMergerWeights,
    normalization_after_merge: bool,
    stream: sim_tensor::StreamId,
) -> Result<NativeQwenVisionMerger, MultimodalTextError> {
    let merge_unit = configuration
        .spatial_merge_size
        .checked_mul(configuration.spatial_merge_size)
        .ok_or(MultimodalTextError::Overflow("Qwen vision merge unit"))?;
    let merge_width = configuration
        .hidden_size
        .checked_mul(merge_unit)
        .ok_or(MultimodalTextError::Overflow("Qwen vision merge width"))?;
    let normalization_width = if normalization_after_merge {
        merge_width
    } else {
        configuration.hidden_size
    };
    Ok(NativeQwenVisionMerger {
        normalization: qwen_layer_norm_module(
            &format!("{name}.norm"),
            normalization_width,
            weights.normalization_weight,
            weights.normalization_bias,
            stream,
        )?,
        first: qwen_linear_module(
            &format!("{name}.linear_one"),
            merge_width,
            merge_width,
            weights.first_weight,
            Some(weights.first_bias),
            stream,
        )?,
        activation: NativeModule::gelu(format!("{name}.activation"), GeluApproximation::None)?,
        second: qwen_linear_module(
            &format!("{name}.linear_two"),
            merge_width,
            configuration.output_hidden_size,
            weights.second_weight,
            Some(weights.second_bias),
            stream,
        )?,
        normalization_after_merge,
    })
}

fn qwen_require_parameter_shape(
    tensor: &Tensor,
    expected: &[usize],
    stream: sim_tensor::StreamId,
) -> Result<(), MultimodalTextError> {
    let mut expected_shape = Vec::new();
    expected_shape
        .try_reserve_exact(expected.len())
        .map_err(|_| MultimodalTextError::Overflow("Qwen parameter shape"))?;
    for value in expected {
        expected_shape.push(usize_to_u64(*value, "Qwen parameter shape")?);
    }
    let descriptor = tensor.descriptor();
    if descriptor.shape() != expected_shape
        || descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != stream
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen vision parameter shape or execution target is invalid",
        ));
    }
    for chunk in tensor
        .contiguous_bytes()?
        .chunks_exact(std::mem::size_of::<f32>())
    {
        let bytes: [u8; 4] = chunk.try_into().map_err(|_| {
            MultimodalTextError::InvalidInput("Qwen vision parameter storage is malformed")
        })?;
        if !f32::from_ne_bytes(bytes).is_finite() {
            return Err(MultimodalTextError::InvalidInput(
                "Qwen vision parameters must be finite",
            ));
        }
    }
    Ok(())
}

fn qwen_tensor(
    backend: &CpuBackend,
    shape: &[usize],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    let mut tensor_shape = Vec::new();
    tensor_shape
        .try_reserve_exact(shape.len())
        .map_err(|_| MultimodalTextError::Overflow("Qwen tensor shape"))?;
    for value in shape {
        tensor_shape.push(usize_to_u64(*value, "Qwen tensor shape")?);
    }
    Ok(tensor_from_f32(backend, &tensor_shape, values, context)?)
}

fn qwen_add_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    Ok(native_tensor_add(backend, left, right, context)?)
}

fn qwen_add_interpolated_positions(
    hidden: &mut [f32],
    grid_thw: [usize; 3],
    position_embedding: &[f32],
    hidden_size: usize,
    merge_size: usize,
    cancellation: &sim_types::CancellationToken,
) -> Result<(), MultimodalTextError> {
    let [frames, height, width] = grid_thw;
    if frames == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(merge_size)
        || !width.is_multiple_of(merge_size)
        || position_embedding.len()
            != 48_usize
                .checked_mul(48)
                .and_then(|value| value.checked_mul(hidden_size))
                .ok_or(MultimodalTextError::Overflow("Qwen position table"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen vision position interpolation geometry is invalid",
        ));
    }
    let token_count = frames
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or(MultimodalTextError::Overflow("Qwen position token count"))?;
    if hidden.len()
        != token_count
            .checked_mul(hidden_size)
            .ok_or(MultimodalTextError::Overflow("Qwen hidden positions"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen hidden state does not match its image grid",
        ));
    }
    let denominator_height = height.saturating_sub(1).max(1) as f32;
    let denominator_width = width.saturating_sub(1).max(1) as f32;
    let merged_height = height / merge_size;
    let merged_width = width / merge_size;
    let mut output_token = 0_usize;
    for _frame in 0..frames {
        for block_y in 0..merged_height {
            for block_x in 0..merged_width {
                cancellation.check()?;
                for merge_y in 0..merge_size {
                    for merge_x in 0..merge_size {
                        let y = block_y * merge_size + merge_y;
                        let x = block_x * merge_size + merge_x;
                        let source_y = y as f32 * 47.0 / denominator_height;
                        let source_x = x as f32 * 47.0 / denominator_width;
                        let low_y = source_y.floor() as usize;
                        let low_x = source_x.floor() as usize;
                        let high_y = low_y.saturating_add(1).min(47);
                        let high_x = low_x.saturating_add(1).min(47);
                        let fraction_y = source_y - low_y as f32;
                        let fraction_x = source_x - low_x as f32;
                        let neighbors = [
                            (low_y, low_x, (1.0 - fraction_y) * (1.0 - fraction_x)),
                            (low_y, high_x, (1.0 - fraction_y) * fraction_x),
                            (high_y, low_x, fraction_y * (1.0 - fraction_x)),
                            (high_y, high_x, fraction_y * fraction_x),
                        ];
                        let output_start = output_token
                            .checked_mul(hidden_size)
                            .ok_or(MultimodalTextError::Overflow("Qwen position output"))?;
                        for hidden_index in 0..hidden_size {
                            let mut position_value = 0.0_f32;
                            for (source_y, source_x, weight) in neighbors {
                                let source_index = source_y
                                    .checked_mul(48)
                                    .and_then(|value| value.checked_add(source_x))
                                    .and_then(|value| value.checked_mul(hidden_size))
                                    .and_then(|value| value.checked_add(hidden_index))
                                    .ok_or(MultimodalTextError::Overflow("Qwen position source"))?;
                                position_value += *position_embedding.get(source_index).ok_or(
                                    MultimodalTextError::InvalidInput(
                                        "Qwen position table storage is incomplete",
                                    ),
                                )? * weight;
                            }
                            let output_index = output_start
                                .checked_add(hidden_index)
                                .ok_or(MultimodalTextError::Overflow("Qwen position output"))?;
                            *hidden.get_mut(output_index).ok_or(
                                MultimodalTextError::InvalidInput(
                                    "Qwen hidden state storage is incomplete",
                                ),
                            )? += position_value;
                        }
                        output_token += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

fn qwen_vision_attention(
    backend: &CpuBackend,
    projected: &[f32],
    grid_thw: [usize; 3],
    hidden_size: usize,
    heads: usize,
    context: &ExecutionContext<'_>,
) -> Result<Vec<f32>, MultimodalTextError> {
    let token_count = grid_thw[0]
        .checked_mul(grid_thw[1])
        .and_then(|value| value.checked_mul(grid_thw[2]))
        .ok_or(MultimodalTextError::Overflow("Qwen attention tokens"))?;
    let head_dimension = hidden_size
        .checked_div(heads)
        .ok_or(MultimodalTextError::Overflow(
            "Qwen attention head dimension",
        ))?;
    if token_count == 0
        || heads == 0
        || head_dimension == 0
        || !head_dimension.is_multiple_of(4)
        || projected.len()
            != token_count
                .checked_mul(hidden_size)
                .and_then(|value| value.checked_mul(3))
                .ok_or(MultimodalTextError::Overflow("Qwen QKV projection"))?
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen vision attention geometry is invalid",
        ));
    }
    let vector_count = token_count
        .checked_mul(hidden_size)
        .ok_or(MultimodalTextError::Overflow("Qwen attention vectors"))?;
    let mut query = Vec::new();
    let mut key = Vec::new();
    let mut value = Vec::new();
    for output in [&mut query, &mut key, &mut value] {
        output
            .try_reserve_exact(vector_count)
            .map_err(|_| MultimodalTextError::Overflow("Qwen attention vectors"))?;
    }
    let [frames, height, width] = grid_thw;
    let merge_size = QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE;
    let merged_height = height / merge_size;
    let merged_width = width / merge_size;
    let frequency_width = head_dimension / 4;
    let mut token_index = 0_usize;
    for _frame in 0..frames {
        for block_y in 0..merged_height {
            for block_x in 0..merged_width {
                context.cancellation.check()?;
                for merge_y in 0..merge_size {
                    for merge_x in 0..merge_size {
                        let row = block_y * merge_size + merge_y;
                        let column = block_x * merge_size + merge_x;
                        let row_offset = token_index
                            .checked_mul(hidden_size)
                            .and_then(|value| value.checked_mul(3))
                            .ok_or(MultimodalTextError::Overflow("Qwen QKV row"))?;
                        for head in 0..heads {
                            let head_offset = head
                                .checked_mul(head_dimension)
                                .ok_or(MultimodalTextError::Overflow("Qwen attention head"))?;
                            let mut query_head = Vec::new();
                            let mut key_head = Vec::new();
                            query_head
                                .try_reserve_exact(head_dimension)
                                .map_err(|_| MultimodalTextError::Overflow("Qwen query head"))?;
                            key_head
                                .try_reserve_exact(head_dimension)
                                .map_err(|_| MultimodalTextError::Overflow("Qwen key head"))?;
                            for dimension in 0..head_dimension {
                                let query_index = row_offset
                                    .checked_add(head_offset)
                                    .and_then(|value| value.checked_add(dimension))
                                    .ok_or(MultimodalTextError::Overflow("Qwen query index"))?;
                                let key_index = row_offset
                                    .checked_add(hidden_size)
                                    .and_then(|value| value.checked_add(head_offset))
                                    .and_then(|value| value.checked_add(dimension))
                                    .ok_or(MultimodalTextError::Overflow("Qwen key index"))?;
                                query_head.push(*projected.get(query_index).ok_or(
                                    MultimodalTextError::InvalidInput(
                                        "Qwen query projection storage is incomplete",
                                    ),
                                )?);
                                key_head.push(*projected.get(key_index).ok_or(
                                    MultimodalTextError::InvalidInput(
                                        "Qwen key projection storage is incomplete",
                                    ),
                                )?);
                            }
                            qwen_apply_vision_rope(&mut query_head, row, column, frequency_width)?;
                            qwen_apply_vision_rope(&mut key_head, row, column, frequency_width)?;
                            query.extend_from_slice(&query_head);
                            key.extend_from_slice(&key_head);
                            let value_start = row_offset
                                .checked_add(hidden_size.checked_mul(2).ok_or(
                                    MultimodalTextError::Overflow("Qwen value projection"),
                                )?)
                                .and_then(|value| value.checked_add(head_offset))
                                .ok_or(MultimodalTextError::Overflow("Qwen value head"))?;
                            let value_end = value_start
                                .checked_add(head_dimension)
                                .ok_or(MultimodalTextError::Overflow("Qwen value head"))?;
                            value.extend_from_slice(projected.get(value_start..value_end).ok_or(
                                MultimodalTextError::InvalidInput(
                                    "Qwen value projection storage is incomplete",
                                ),
                            )?);
                        }
                        token_index += 1;
                    }
                }
            }
        }
    }
    let workspace_limit_bytes = token_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(MultimodalTextError::Overflow("Qwen attention workspace"))?;
    let outcome = scaled_dot_product_attention_with_context(
        backend,
        AttentionRequest {
            backend: AttentionBackend::PytorchSdp,
            fallback: AttentionFallbackPolicy::AllowExactNative,
            batch: 1,
            query_tokens: token_count,
            key_tokens: token_count,
            heads,
            head_dimension,
            value_dimension: head_dimension,
            scale: None,
            workspace_limit_bytes,
        },
        &query,
        &key,
        &value,
        None,
        context,
    )?;
    Ok(outcome.values)
}

fn qwen_apply_vision_rope(
    values: &mut [f32],
    row: usize,
    column: usize,
    frequency_width: usize,
) -> Result<(), MultimodalTextError> {
    let expected_width = frequency_width
        .checked_mul(4)
        .ok_or(MultimodalTextError::Overflow("Qwen rotary width"))?;
    if frequency_width == 0 || values.len() != expected_width {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen rotary vector width is invalid",
        ));
    }
    let half = values.len() / 2;
    for index in 0..half {
        let coordinate = if index < frequency_width { row } else { column };
        let frequency_index = index % frequency_width;
        let exponent = (frequency_index * 2) as f32 / (frequency_width * 2) as f32;
        let angle = coordinate as f32 / 10_000.0_f32.powf(exponent);
        let cosine = angle.cos();
        let sine = angle.sin();
        let second_index = index
            .checked_add(half)
            .ok_or(MultimodalTextError::Overflow("Qwen rotary index"))?;
        let first = *values.get(index).ok_or(MultimodalTextError::InvalidInput(
            "Qwen rotary vector storage is incomplete",
        ))?;
        let second = *values
            .get(second_index)
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen rotary vector storage is incomplete",
            ))?;
        *values
            .get_mut(index)
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen rotary vector storage is incomplete",
            ))? = first * cosine - second * sine;
        *values
            .get_mut(second_index)
            .ok_or(MultimodalTextError::InvalidInput(
                "Qwen rotary vector storage is incomplete",
            ))? = second * cosine + first * sine;
    }
    Ok(())
}

pub fn join_multimodal_embeddings(
    backend: &CpuBackend,
    text_embeddings: &Tensor,
    images: &[MultimodalImageEmbedding<'_>],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    context.cancellation.check()?;
    let shape = text_embeddings.descriptor().shape();
    if shape.len() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "text embeddings must use nonempty [batch, tokens, hidden] shape",
        ));
    }
    require_cpu_f32(text_embeddings, context)?;
    if images.is_empty() {
        return Ok(text_embeddings.clone());
    }
    let sequence_length = u64_to_usize(shape[1], "sequence length")?;
    let mut pieces = Vec::new();
    pieces
        .try_reserve_exact(images.len().saturating_mul(2).saturating_add(1))
        .map_err(|_| MultimodalTextError::Overflow("joined embedding pieces"))?;
    let mut cursor = 0_usize;
    for image in images {
        context.cancellation.check()?;
        let end = image
            .span
            .start
            .checked_add(image.span.size)
            .ok_or(MultimodalTextError::Overflow("image span end"))?;
        if image.span.size == 0 || image.span.start < cursor || end > sequence_length {
            return Err(MultimodalTextError::InvalidInput(
                "image spans must be ordered, nonempty, and in bounds",
            ));
        }
        if image.span.start > cursor {
            pieces.push(narrow_method_exact_native(
                text_embeddings,
                1,
                usize_to_i64(cursor, "text prefix")?,
                usize_to_u64(image.span.start - cursor, "text prefix")?,
                context.cancellation,
            )?);
        }
        pieces.push(normalize_image_embedding(
            backend,
            image.embedding,
            shape[0],
            usize_to_u64(image.span.size, "image span")?,
            shape[2],
            context,
        )?);
        cursor = end;
    }
    if cursor < sequence_length {
        pieces.push(narrow_method_exact_native(
            text_embeddings,
            1,
            usize_to_i64(cursor, "text suffix")?,
            usize_to_u64(sequence_length - cursor, "text suffix")?,
            context.cancellation,
        )?);
    }
    if pieces.is_empty() {
        return Err(MultimodalTextError::InvalidInput(
            "joined embedding graph produced no pieces",
        ));
    }
    let joined = torch_cat_with_context_exact_native(backend, &pieces, 1, context)?;
    context.cancellation.check()?;
    Ok(joined)
}

pub fn join_qwen3vl_deepstack(
    backend: &CpuBackend,
    sequence_length: usize,
    images: &[MultimodalImageEmbedding<'_>],
    context: &ExecutionContext<'_>,
) -> Result<Option<MultimodalDeepstackJoin>, MultimodalTextError> {
    context.cancellation.check()?;
    if images.is_empty() {
        return Ok(None);
    }
    let layer_count = images[0].deepstack.len();
    if layer_count == 0 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL images require deepstack layers",
        ));
    }
    let mut mask = vec![false; sequence_length];
    let mut per_layer = (0..layer_count)
        .map(|_| Vec::<Tensor>::new())
        .collect::<Vec<_>>();
    let mut previous_end = 0_usize;
    for image in images {
        context.cancellation.check()?;
        let end = image
            .span
            .start
            .checked_add(image.span.size)
            .ok_or(MultimodalTextError::Overflow("deepstack image span"))?;
        if image.span.size == 0
            || image.span.start < previous_end
            || end > sequence_length
            || image.deepstack.len() != layer_count
        {
            return Err(MultimodalTextError::InvalidInput(
                "deepstack spans or layer counts are invalid",
            ));
        }
        mask[image.span.start..end].fill(true);
        for (layer, tensor) in image.deepstack.iter().enumerate() {
            require_cpu_f32(tensor, context)?;
            let shape = tensor.descriptor().shape();
            if shape.len() != 2 || shape[0] != usize_to_u64(image.span.size, "deepstack span")? {
                return Err(MultimodalTextError::InvalidInput(
                    "deepstack layers must use [image tokens, hidden] shape",
                ));
            }
            per_layer[layer].push(tensor.clone());
        }
        previous_end = end;
    }
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(layer_count)
        .map_err(|_| MultimodalTextError::Overflow("deepstack layers"))?;
    for inputs in per_layer {
        layers.push(torch_cat_with_context_exact_native(
            backend, &inputs, 0, context,
        )?);
    }
    context.cancellation.check()?;
    Ok(Some(MultimodalDeepstackJoin {
        visual_position_mask: mask,
        layers,
    }))
}

pub fn ideogram4_project_taps(
    backend: &CpuBackend,
    tapped_layers: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    context.cancellation.check()?;
    require_cpu_f32(tapped_layers, context)?;
    let shape = tapped_layers.descriptor().shape();
    if shape.len() != 4 || shape[1] != IDEOGRAM4_TAP_LAYERS.len() as u64 {
        return Err(MultimodalTextError::InvalidInput(
            "Ideogram4 taps must use [batch, 13, tokens, hidden] shape",
        ));
    }
    let projected_width = shape[3]
        .checked_mul(shape[1])
        .ok_or(MultimodalTextError::Overflow("Ideogram4 projection width"))?;
    let permuted = tensor_permute_exact_native(tapped_layers, &[0, 2, 3, 1], context.cancellation)?;
    let projected = torch_reshape_with_context_exact_native(
        backend,
        &permuted,
        &[
            u64_to_i64(shape[0], "Ideogram4 batch")?,
            u64_to_i64(shape[2], "Ideogram4 tokens")?,
            u64_to_i64(projected_width, "Ideogram4 width")?,
        ],
        context,
    )?;
    context.cancellation.check()?;
    Ok(projected)
}

pub fn ovis_template_end(tokens: &[i64]) -> Result<usize, MultimodalTextError> {
    let mut start =
        tokens
            .iter()
            .position(|token| *token == 4004)
            .ok_or(MultimodalTextError::InvalidInput(
                "Ovis prompt is missing the first im_start token",
            ))?;
    if tokens.get(start + 1) == Some(&25) {
        start += 1;
    }
    Ok(start)
}

pub fn trim_ovis_conditioning(
    conditioning: &Tensor,
    template_end: usize,
    cancellation: &sim_types::CancellationToken,
) -> Result<Tensor, MultimodalTextError> {
    cancellation.check()?;
    let shape = conditioning.descriptor().shape();
    if shape.len() != 3 || template_end >= u64_to_usize(shape[1], "Ovis tokens")? {
        return Err(MultimodalTextError::InvalidInput(
            "Ovis conditioning must be rank three and retain at least one token",
        ));
    }
    Ok(narrow_method_exact_native(
        conditioning,
        1,
        usize_to_i64(template_end, "Ovis template end")?,
        shape[1] - usize_to_u64(template_end, "Ovis template end")?,
        cancellation,
    )?)
}

pub fn parse_sam3_prompts(text: &str) -> Result<Vec<Sam3Prompt>, MultimodalTextError> {
    const MAX_PROMPTS: usize = 1024;
    const MAX_PROMPT_BYTES: usize = 1 << 20;
    if text.len() > MAX_PROMPT_BYTES {
        return Err(MultimodalTextError::InvalidInput(
            "SAM3 prompt bytes exceed the native bound",
        ));
    }
    let normalized = text.replace(['(', ')'], "");
    let mut prompts = Vec::new();
    for part in normalized
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if prompts.len() == MAX_PROMPTS {
            return Err(MultimodalTextError::InvalidInput(
                "SAM3 prompt count exceeds the native bound",
            ));
        }
        let (prompt, maximum_detections) = parse_sam3_prompt_part(part)?;
        prompts.push(Sam3Prompt {
            text: prompt.to_owned(),
            maximum_detections,
        });
    }
    Ok(prompts)
}

pub fn pack_sam3_conditions(
    conditions: Vec<Sam3EncodedCondition>,
    first_pooled: Option<Tensor>,
) -> Result<Sam3ConditionPack, MultimodalTextError> {
    let first = conditions.first().ok_or(MultimodalTextError::InvalidInput(
        "SAM3 condition packing requires at least one prompt",
    ))?;
    if conditions
        .iter()
        .any(|condition| condition.maximum_detections == 0)
    {
        return Err(MultimodalTextError::InvalidInput(
            "SAM3 maximum detections must be nonzero",
        ));
    }
    Ok(Sam3ConditionPack {
        main_condition: first.condition.clone(),
        first_pooled,
        conditions,
    })
}

pub fn format_ideogram4_prompt(text: &str) -> String {
    if text.starts_with("<|im_start|>") {
        text.to_owned()
    } else {
        format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    }
}

pub fn format_ovis_prompt(text: &str) -> String {
    format!(
        "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: {text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

pub fn format_qwen3vl_prompt(text: &str, image_count: usize, thinking: bool) -> String {
    if text.starts_with("<|im_start|>") {
        return text.to_owned();
    }
    let vision = "<|vision_start|><|image_pad|><|vision_end|>".repeat(image_count);
    let mut prompt = if image_count == 0 {
        format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
    } else {
        format!("<|im_start|>user\n{vision}{text}<|im_end|>\n<|im_start|>assistant\n")
    };
    if !thinking {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}

pub fn format_qwen_multimodal_prompt(
    family: QwenVisionFamily,
    text: &str,
    image_count: usize,
    thinking: bool,
    _use_default_template: bool,
    cancellation: &sim_types::CancellationToken,
) -> Result<String, MultimodalTextError> {
    cancellation.check()?;
    if text.starts_with("<|im_start|>") {
        return Ok(text.to_owned());
    }
    const PREFIX: &str = "<|im_start|>user\n";
    const IMAGE_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
    let thinking_suffix = if thinking {
        ""
    } else {
        match family {
            QwenVisionFamily::Qwen3Vl4B | QwenVisionFamily::Qwen3Vl8B => "<think>\n\n</think>\n\n",
            QwenVisionFamily::Qwen35_08B
            | QwenVisionFamily::Qwen35_2B
            | QwenVisionFamily::Qwen35_4B
            | QwenVisionFamily::Qwen35_9B
            | QwenVisionFamily::Qwen35_27B => "<think>\n</think>\n",
        }
    };
    let marker_bytes = IMAGE_MARKER
        .len()
        .checked_mul(image_count)
        .ok_or(MultimodalTextError::Overflow("Qwen image prompt markers"))?;
    let capacity = PREFIX
        .len()
        .checked_add(marker_bytes)
        .and_then(|value| value.checked_add(text.len()))
        .and_then(|value| value.checked_add(SUFFIX.len()))
        .and_then(|value| value.checked_add(thinking_suffix.len()))
        .ok_or(MultimodalTextError::Overflow("Qwen prompt template"))?;
    if capacity > crate::MAX_NATIVE_PROMPT_BYTES {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen prompt template exceeds the native bounded prompt size",
        ));
    }
    let mut prompt = String::new();
    prompt
        .try_reserve_exact(capacity)
        .map_err(|_| MultimodalTextError::Overflow("Qwen prompt template"))?;
    prompt.push_str(PREFIX);
    for index in 0..image_count {
        if index.is_multiple_of(16) {
            cancellation.check()?;
        }
        prompt.push_str(IMAGE_MARKER);
    }
    prompt.push_str(text);
    prompt.push_str(SUFFIX);
    prompt.push_str(thinking_suffix);
    cancellation.check()?;
    Ok(prompt)
}

fn parse_sam3_prompt_part(part: &str) -> Result<(&str, usize), MultimodalTextError> {
    let Some((text, raw_limit)) = part.rsplit_once(':') else {
        return Ok((part, 1));
    };
    let text = text.trim();
    let raw_limit = raw_limit.trim();
    if text.is_empty()
        || raw_limit.is_empty()
        || !raw_limit
            .chars()
            .all(|character| character.is_ascii_digit() || character == '.')
    {
        return Ok((part, 1));
    }
    let value = raw_limit
        .parse::<f64>()
        .map_err(|_| MultimodalTextError::InvalidPromptLimit(raw_limit.to_owned()))?;
    if !value.is_finite() || value < 0.0 || value > usize::MAX as f64 {
        return Err(MultimodalTextError::InvalidPromptLimit(
            raw_limit.to_owned(),
        ));
    }
    let rounded = value.round_ties_even().max(1.0);
    Ok((text, rounded as usize))
}

fn prepare_qwen3vl_resized_image(
    backend: &CpuBackend,
    image: &ImageTensor,
    height: u64,
    width: u64,
    family: QwenVisionFamily,
    context: &ExecutionContext<'_>,
) -> Result<Qwen3VlPreparedImage, MultimodalTextError> {
    context.cancellation.check()?;
    let (batch, actual_height, actual_width, channels) = image.dimensions()?;
    if (batch, actual_height, actual_width, channels) != (1, height, width, 3) {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL resized image geometry is inconsistent",
        ));
    }
    let patch_size = u64::try_from(QWEN3VL_IMAGE_PATCH_SIZE)
        .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL patch size"))?;
    let merge_size = u64::try_from(QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE)
        .map_err(|_| MultimodalTextError::Overflow("Qwen3-VL merge size"))?;
    let grid_height = height
        .checked_div(patch_size)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL grid height"))?;
    let grid_width = width
        .checked_div(patch_size)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL grid width"))?;
    if grid_height == 0
        || grid_width == 0
        || !grid_height.is_multiple_of(merge_size)
        || !grid_width.is_multiple_of(merge_size)
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL patch grid must contain complete spatial merge blocks",
        ));
    }
    let patch_count = grid_height
        .checked_mul(grid_width)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL patch count"))?;
    let patch_width = 3_usize
        .checked_mul(QWEN3VL_IMAGE_TEMPORAL_PATCH_SIZE)
        .and_then(|value| value.checked_mul(QWEN3VL_IMAGE_PATCH_SIZE))
        .and_then(|value| value.checked_mul(QWEN3VL_IMAGE_PATCH_SIZE))
        .ok_or(MultimodalTextError::Overflow(
            "Qwen3-VL flattened patch width",
        ))?;
    let patch_elements = patch_count
        .checked_mul(patch_width)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL patch elements"))?;
    let values = image.as_f32_slice()?;
    let (mean, standard_deviation) = family.normalization();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL image pixels must be finite",
        ));
    }
    let height_usize = u64_to_usize(height, "Qwen3-VL target height")?;
    let width_usize = u64_to_usize(width, "Qwen3-VL target width")?;
    let grid_height_usize = u64_to_usize(grid_height, "Qwen3-VL grid height")?;
    let grid_width_usize = u64_to_usize(grid_width, "Qwen3-VL grid width")?;
    let merged_height = grid_height_usize / QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE;
    let merged_width = grid_width_usize / QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE;
    let mut patches = backend.workspace_vec(context, patch_elements)?;
    for block_y in 0..merged_height {
        for block_x in 0..merged_width {
            context.cancellation.check()?;
            for merge_y in 0..QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE {
                for merge_x in 0..QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE {
                    let patch_y = block_y * QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE + merge_y;
                    let patch_x = block_x * QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE + merge_x;
                    for channel in 0..3_usize {
                        for _temporal in 0..QWEN3VL_IMAGE_TEMPORAL_PATCH_SIZE {
                            for local_y in 0..QWEN3VL_IMAGE_PATCH_SIZE {
                                for local_x in 0..QWEN3VL_IMAGE_PATCH_SIZE {
                                    let y = patch_y * QWEN3VL_IMAGE_PATCH_SIZE + local_y;
                                    let x = patch_x * QWEN3VL_IMAGE_PATCH_SIZE + local_x;
                                    let index = y
                                        .checked_mul(width_usize)
                                        .and_then(|value| value.checked_add(x))
                                        .and_then(|value| value.checked_mul(3))
                                        .and_then(|value| value.checked_add(channel))
                                        .ok_or(MultimodalTextError::Overflow(
                                            "Qwen3-VL normalized pixel index",
                                        ))?;
                                    let value = values.get(index).copied().ok_or(
                                        MultimodalTextError::InvalidInput(
                                            "Qwen3-VL resized image storage is incomplete",
                                        ),
                                    )?;
                                    patches.try_push(
                                        (value - mean[channel]) / standard_deviation[channel],
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let reconstructed_height = merged_height
        .checked_mul(
            QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE
                .checked_mul(QWEN3VL_IMAGE_PATCH_SIZE)
                .ok_or(MultimodalTextError::Overflow(
                    "Qwen3-VL reconstructed height",
                ))?,
        )
        .ok_or(MultimodalTextError::Overflow(
            "Qwen3-VL reconstructed height",
        ))?;
    let reconstructed_width = merged_width
        .checked_mul(
            QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE
                .checked_mul(QWEN3VL_IMAGE_PATCH_SIZE)
                .ok_or(MultimodalTextError::Overflow(
                    "Qwen3-VL reconstructed width",
                ))?,
        )
        .ok_or(MultimodalTextError::Overflow(
            "Qwen3-VL reconstructed width",
        ))?;
    if patches.len() != patch_elements
        || height_usize != reconstructed_height
        || width_usize != reconstructed_width
    {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL patch packing did not cover the resized image exactly",
        ));
    }
    let descriptor = TensorDescriptor::contiguous(
        vec![
            usize_to_u64(patch_count, "Qwen3-VL patch count")?,
            3,
            usize_to_u64(
                QWEN3VL_IMAGE_TEMPORAL_PATCH_SIZE,
                "Qwen3-VL temporal patch size",
            )?,
            usize_to_u64(QWEN3VL_IMAGE_PATCH_SIZE, "Qwen3-VL patch height")?,
            usize_to_u64(QWEN3VL_IMAGE_PATCH_SIZE, "Qwen3-VL patch width")?,
        ],
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    let patches = backend.upload_f32(descriptor, &patches, context)?.0;
    let merged_tokens = merged_height
        .checked_mul(merged_width)
        .ok_or(MultimodalTextError::Overflow("Qwen3-VL merged tokens"))?;
    context.cancellation.check()?;
    Ok(Qwen3VlPreparedImage {
        patches,
        grid_thw: [1, grid_height_usize, grid_width_usize],
        merged_tokens,
        family,
    })
}

fn round_to_factor(value: u64, factor: u64) -> Result<u64, MultimodalTextError> {
    let units = ((value as f64) / (factor as f64)).round_ties_even();
    scaled_units_to_dimension(units, factor, false)
}

fn floor_scaled_to_factor(value: u64, beta: f64, factor: u64) -> Result<u64, MultimodalTextError> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL resize scale must be finite and positive",
        ));
    }
    let units = ((value as f64) / beta / (factor as f64)).floor();
    scaled_units_to_dimension(units, factor, true)
}

fn ceil_scaled_to_factor(value: u64, beta: f64, factor: u64) -> Result<u64, MultimodalTextError> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(MultimodalTextError::InvalidInput(
            "Qwen3-VL resize scale must be finite and positive",
        ));
    }
    let units = ((value as f64) * beta / (factor as f64)).ceil();
    scaled_units_to_dimension(units, factor, false)
}

fn scaled_units_to_dimension(
    units: f64,
    factor: u64,
    minimum_one: bool,
) -> Result<u64, MultimodalTextError> {
    if !units.is_finite() || units < 0.0 || units > (u64::MAX / factor) as f64 {
        return Err(MultimodalTextError::Overflow(
            "Qwen3-VL scaled image dimension",
        ));
    }
    let units = units as u64;
    let units = if minimum_one { units.max(1) } else { units };
    units
        .checked_mul(factor)
        .ok_or(MultimodalTextError::Overflow(
            "Qwen3-VL scaled image dimension",
        ))
}

fn sequential_positions(length: usize) -> Result<Vec<i64>, MultimodalTextError> {
    let mut positions = Vec::new();
    positions
        .try_reserve_exact(length)
        .map_err(|_| MultimodalTextError::Overflow("sequential positions"))?;
    for index in 0..length {
        positions.push(usize_to_i64(index, "sequential position")?);
    }
    Ok(positions)
}

fn clone_positions(positions: &[i64]) -> Result<Vec<i64>, MultimodalTextError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(positions.len())
        .map_err(|_| MultimodalTextError::Overflow("multimodal positions"))?;
    cloned.extend_from_slice(positions);
    Ok(cloned)
}

fn normalize_image_embedding(
    backend: &CpuBackend,
    embedding: &Tensor,
    batch: u64,
    tokens: u64,
    hidden: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, MultimodalTextError> {
    require_cpu_f32(embedding, context)?;
    let shape = embedding.descriptor().shape();
    match shape {
        [actual_tokens, actual_hidden]
            if batch == 1 && *actual_tokens == tokens && *actual_hidden == hidden =>
        {
            Ok(torch_reshape_with_context_exact_native(
                backend,
                embedding,
                &[
                    1,
                    u64_to_i64(tokens, "image tokens")?,
                    u64_to_i64(hidden, "hidden")?,
                ],
                context,
            )?)
        }
        [actual_batch, actual_tokens, actual_hidden]
            if *actual_batch == batch && *actual_tokens == tokens && *actual_hidden == hidden =>
        {
            Ok(embedding.clone())
        }
        _ => Err(MultimodalTextError::InvalidInput(
            "image embeddings must match [batch, image tokens, hidden]",
        )),
    }
}

fn require_cpu_f32(
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<(), MultimodalTextError> {
    let descriptor = tensor.descriptor();
    if descriptor.dtype() != DType::F32
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != context.stream
    {
        return Err(MultimodalTextError::InvalidInput(
            "multimodal tensors must be CPU F32 values on the caller stream",
        ));
    }
    Ok(())
}

fn usize_to_i64(value: usize, name: &'static str) -> Result<i64, MultimodalTextError> {
    i64::try_from(value).map_err(|_| MultimodalTextError::Overflow(name))
}

fn checked_f64_to_u64(value: f64, name: &'static str) -> Result<u64, MultimodalTextError> {
    if !value.is_finite() || value < 0.0 || value >= 18_446_744_073_709_551_616.0 {
        return Err(MultimodalTextError::Overflow(name));
    }
    Ok(value as u64)
}

fn usize_to_u64(value: usize, name: &'static str) -> Result<u64, MultimodalTextError> {
    u64::try_from(value).map_err(|_| MultimodalTextError::Overflow(name))
}

fn u64_to_usize(value: u64, name: &'static str) -> Result<usize, MultimodalTextError> {
    usize::try_from(value).map_err(|_| MultimodalTextError::Overflow(name))
}

fn u64_to_i64(value: u64, name: &'static str) -> Result<i64, MultimodalTextError> {
    i64::try_from(value).map_err(|_| MultimodalTextError::Overflow(name))
}

#[cfg(test)]
mod native_qwen25_conditioning_tests {
    use super::*;

    #[test]
    fn qwen25_markers_map_successive_images_and_leave_unmatched_inputs_source_exact()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = sim_types::CancellationToken::default();
        let plan = plan_qwen25_marker_layout(
            &[10, 151_655, 20, 151_655, 151_655, 30],
            &[
                ([1, 2, 2], 2),
                ([1, 4, 2], 3),
                ([1, 6, 2], 1),
                ([1, 8, 2], 9),
            ],
            &cancellation,
        )?;
        assert_eq!(
            plan.expanded_tokens,
            [
                10, 151_655, 151_655, 20, 151_655, 151_655, 151_655, 151_655, 30
            ]
        );
        assert_eq!(plan.spans.len(), 3);
        assert_eq!((plan.spans[0].start, plan.spans[0].size), (1, 2));
        assert_eq!((plan.spans[1].start, plan.spans[1].size), (4, 3));
        assert_eq!((plan.spans[2].start, plan.spans[2].size), (7, 1));

        let unmatched = plan_qwen25_marker_layout(&[151_655, 7], &[], &cancellation)?;
        assert_eq!(unmatched.expanded_tokens, [151_655, 7]);
        assert!(unmatched.spans.is_empty());
        Ok(())
    }
}
