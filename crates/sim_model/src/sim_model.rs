pub mod alias_free_activation;
pub mod artifact_index;
pub mod attention;
pub mod audio_encoder;
pub mod background_removal;
pub mod clip;
pub mod clip_text;
pub mod clip_text_encoder_composite;
pub mod clip_text_encoder_decoder;
pub mod clip_text_encoder_multimodal;
pub mod clip_text_encoder_t5;
pub mod clip_text_encoders;
pub mod clip_tokenizer;
pub mod clip_vision;
pub mod cogvideox_family;
pub mod conditioning;
pub mod conditioning_resources;
pub mod controlnet;
pub mod cosmos_family;
pub mod depth_anything_3;
pub mod descriptor;
mod dino2;
pub mod flux_chroma_family;
pub mod formats;
pub mod frame_interpolation;
pub mod hidream_o1_family;
pub mod hunyuan3d_family;
pub mod hunyuan_video_family;
pub mod hunyuandit_family;
pub mod kandinsky5_family;
pub mod latent_format;
pub mod latent_operation;
pub mod latent_upscale_model;
pub mod ltx_family;
pub mod lumina_zimage_family;
pub mod model_family;
pub mod model_store;
pub mod moge;
pub mod native_node_payload;
pub mod native_ops;
pub mod omnigen2_boogu_family;
pub mod parser_limits;
pub mod patch_graph;
pub mod patches;
pub mod pixart_family;
pub mod pixeldit_pid_family;
pub mod quantization;
pub mod quantized_autograd;
pub mod qwen_image_family;
pub mod registry_generator;
pub mod restricted_pickle;
pub mod sd2_family;
pub mod sdpose;
pub mod sdxl_family;
pub mod upscale_contract;
pub mod upscale_model;
pub mod vae;
pub mod vae_architecture;
pub mod vae_audio;
pub mod vae_image;
pub mod vae_structured;
mod vae_tiling;
pub mod vae_video;
pub mod vision_models;
pub mod weight_adapter;

pub use latent_operation::{
    NativeLatentOperation, NativeLatentOperationError, NativeLatentOperationKind,
};

pub use alias_free_activation::{
    NativeAliasFreeActivation1d, PeriodicActivation, alias_free_activation_1d_exact_native,
};
pub use artifact_index::{
    ARTIFACT_INDEX_VERSION, ArtifactAvailability, ArtifactChange, ArtifactChangeKind,
    ArtifactIndex, ArtifactIndexError, ArtifactKey, ArtifactRecord, ArtifactRoot,
    ArtifactWritePolicy, VerifiedArtifactFile,
};
pub use attention::{
    AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionMask, AttentionMaskShape,
    AttentionOutcome, AttentionRequest, MathSdpReductionPolicy, MathSdpSelection, SdpaBackend,
    SdpaKernelSelection, allow_fp16_bf16_reduction_math_sdp_exact_native,
    enable_flash_sdp_exact_native, enable_math_sdp_exact_native,
    enable_mem_efficient_sdp_exact_native, scaled_dot_product_attention_with_context,
    sdpa_kernel_exact_native,
};
pub use audio_encoder::{
    AUDIO_ENCODERS_SOURCE_PATH, AUDIO_ENCODERS_SOURCE_SHA256, AudioEncoderError,
    AudioEncoderProfile, NativeAudioEncoder, NativeAudioEncoderArchitecture,
    NativeAudioEncoderCheckpoint, NativeAudioEncoderDiagnostic, NormalizedAudioEncoderCheckpoint,
    WAV2VEC2_SOURCE_PATH, WAV2VEC2_SOURCE_SHA256, WHISPER_SOURCE_PATH, WHISPER_SOURCE_SHA256,
    normalize_and_select_architecture, normalize_and_select_profile,
};
pub use background_removal::{
    BACKGROUND_REMOVAL_MODEL_SOURCE_SHA256, BIREFNET_CONFIG_SOURCE_SHA256, BIREFNET_MARKER,
    BIREFNET_SOURCE_SHA256, CLIP_MODEL_SOURCE_SHA256, MODEL_MANAGEMENT_SOURCE_SHA256,
    NODES_BACKGROUND_REMOVAL_SOURCE_SHA256, NativeBackgroundRemovalCheckpoint,
    NativeBackgroundRemovalError, NativeBackgroundRemovalResource, SIM_OPS_SOURCE_SHA256,
};
#[cfg(any(test, feature = "test-support"))]
pub use background_removal::{
    BackgroundRemovalFixtureMutation, deterministic_reduced_test_checkpoint,
};
pub use clip_text::{
    CLIP_TEXT_CATALOG_SYMBOLS, CLIP_TEXT_SOURCE_PATH, CLIP_TEXT_SOURCE_SHA256, ClipTextActivation,
    ClipTextConfiguration, ClipTextError, ClipTextInput, ClipTextIntermediate,
    ClipTextLayerWeights, ClipTextOutput, ClipTextRequest, ClipTextWeights, NativeClipText,
    SD1_CLIP_SOURCE_PATH, SD1_CLIP_SOURCE_SHA256,
};
pub use clip_text_encoder_composite::{
    AudioSamplingOptions, COMPOSITE_TEXT_ENCODER_CONTRACT_COUNT, COMPOSITE_TEXT_ENCODER_CONTRACTS,
    CompositeConditioningInput, CompositeConditioningOutput, CompositeContractFact,
    CompositeExecutionPlan, CompositeHiddenJoin, CompositeOwner, CompositePooledPolicy,
    CompositeSymbolBehavior, CompositeTextEncoderError, QuotedPromptPart, basic_cleaners,
    collapse_whitespace, compose_conditioning, composite_contract_fact, composite_execution_plan,
    composite_symbol_behavior, delegate_bidirectional_text, delegate_clip_text,
    delegate_clip_vision, delegate_decoder_text, expand_abbreviations_multilingual,
    expand_numbers_multilingual, expand_symbols_multilingual, generate_audio_codes,
    japanese_to_romaji, multilingual_cleaners, number_to_text_i64, sample_audio_token,
    split_quotation,
};
pub use clip_text_encoder_decoder::{
    DECODER_PROFILE_FACTS, DECODER_TEXT_ENCODER_CATALOG_SYMBOLS, DecoderActivation,
    DecoderArchitecture, DecoderAttentionCache, DecoderAttentionWeights, DecoderAudioProfileFact,
    DecoderGenerationConfiguration, DecoderGenerationOutcome, DecoderKvState, DecoderLayerCache,
    DecoderLayerKind, DecoderLayerWeights, DecoderPreparedDeepstack,
    DecoderPreparedGenerationOutcome, DecoderPreparedGenerationPrompt, DecoderPreparedTextRequest,
    DecoderProfileFact, DecoderRopeConfiguration, DecoderRopePositions, DecoderSymbolBehavior,
    DecoderTextConfiguration, DecoderTextError, DecoderTextOutput, DecoderTextRequest,
    DecoderTextWeights, DecoderVisionProfileFact, GEMMA4_SOURCE_PATH, GEMMA4_SOURCE_SHA256,
    GPT_OSS_SOURCE_PATH, GPT_OSS_SOURCE_SHA256, Gemma3DecoderConfiguration, Gemma3DecoderProfile,
    Gemma4DecoderConfiguration, Gemma4DecoderProfile, Gemma4LayerInputWeights,
    Gemma4PerLayerWeights, LLAMA_SOURCE_PATH, LLAMA_SOURCE_SHA256, NativeDecoderTextEncoder,
    NativeTextGenerationRequest, NativeTextGenerationResult, QWEN35_SOURCE_PATH,
    QWEN35_SOURCE_SHA256, Qwen35LinearCache, Qwen35LinearConfiguration, Qwen35LinearWeights,
    RopeScaling, TEXT_GENERATION_SOURCE_PATH, TEXT_GENERATION_SOURCE_SHA256, apply_rope,
    decoder_profile_fact, decoder_symbol_behavior, gemma3_decoder_configuration,
    gemma4_audio_conv2d_subsample, gemma4_audio_relative_positions, gemma4_clipped_linear,
    gemma4_decoder_configuration, gemma4_vision_patch_embed, gemma4_vision_rope, gpt_oss_moe,
    gpt_oss_top_k_route, precompute_multidimensional_rope, precompute_rope,
    qwen35_causal_conv1d_update, qwen35_causal_conv1d_update_exact, qwen35_chunk_gated_delta_rule,
    qwen35_chunk_gated_delta_rule_exact, qwen35_vision_patch_embed, qwen35_vision_patch_merge,
    tokenize_decoder_prompt,
};
pub use clip_text_encoder_multimodal::{
    GEMMA3_FOUR_B_MULTIMODAL_SOURCE_PATH, GEMMA3_FOUR_B_MULTIMODAL_SOURCE_SHA256,
    GEMMA3_IMAGE_AREA_PIXELS, GEMMA3_IMAGE_SOFT_TOKENS, GEMMA3_MAXIMUM_PREPARED_PIXELS,
    GEMMA3_MULTIMODAL_SOURCE_PATH, GEMMA3_MULTIMODAL_SOURCE_SHA256, GEMMA4_AUDIO_FFT_LENGTH,
    GEMMA4_AUDIO_FRAME_LENGTH, GEMMA4_AUDIO_FRAME_STEP, GEMMA4_AUDIO_MAXIMUM_SAMPLE_RATE,
    GEMMA4_AUDIO_MAXIMUM_TOKENS, GEMMA4_AUDIO_MEL_BINS, GEMMA4_AUDIO_MINIMUM_SAMPLE_RATE,
    GEMMA4_AUDIO_SAMPLE_RATE, GEMMA4_IMAGE_PATCH_SIZE, GEMMA4_IMAGE_POOLING_SIZE,
    GEMMA4_IMAGE_SOFT_TOKENS, GEMMA4_MULTIMODAL_SOURCE_PATH, GEMMA4_MULTIMODAL_SOURCE_SHA256,
    GEMMA4_VIDEO_SOFT_TOKENS, GEMMA4_VIDEO_SOURCE_FPS, Gemma3VisionConfiguration,
    Gemma3VisionProfile, Gemma3VisionProjection, Gemma4AudioBlockWeights, Gemma4AudioConfiguration,
    Gemma4AudioFeedForwardWeights, Gemma4AudioProfile, Gemma4AudioProjection, Gemma4AudioWeights,
    Gemma4ClippedLinearWeights, Gemma4VisionBlockWeights, Gemma4VisionConfiguration,
    Gemma4VisionProfile, Gemma4VisionProjection, Gemma4VisionWeights, GemmaMultimodalFamily,
    GemmaMultimodalGenerationRequest, GemmaPreparedAudio, GemmaPreparedVisual,
    GemmaPreparedVisualKind, IDEOGRAM4_SOURCE_PATH, IDEOGRAM4_SOURCE_SHA256, IDEOGRAM4_TAP_LAYERS,
    JINA_CLIP2_SOURCE_PATH, JINA_CLIP2_SOURCE_SHA256, MULTIMODAL_PROFILE_FACTS,
    MULTIMODAL_TEXT_ENCODER_CATALOG_SYMBOLS, MultimodalDeepstackJoin, MultimodalFamily,
    MultimodalImageEmbedding, MultimodalPositionIds, MultimodalProfileFact, MultimodalSpan,
    MultimodalSymbolBehavior, MultimodalTextError, MultimodalTextOwner,
    NativeGemma3VisionProjector, NativeGemma4AudioEncoder, NativeGemma4VisionEncoder,
    NativeGemmaMultimodal, NativeQwenMultimodal, NativeQwenVisionEncoder, OVIS_SOURCE_PATH,
    OVIS_SOURCE_SHA256, QWEN_MULTIMODAL_ROUTING_SOURCE_PATH, QWEN_MULTIMODAL_ROUTING_SOURCE_SHA256,
    QWEN_VL_SOURCE_PATH, QWEN_VL_SOURCE_SHA256, QWEN2VL_FULL_ATTENTION_LAYERS,
    QWEN3VL_4B_DEEPSTACK_LAYERS, QWEN3VL_8B_DEEPSTACK_LAYERS, QWEN3VL_IMAGE_MAXIMUM_PIXELS,
    QWEN3VL_IMAGE_MINIMUM_PIXELS, QWEN3VL_IMAGE_PAD_TOKEN, QWEN3VL_IMAGE_PATCH_SIZE,
    QWEN3VL_IMAGE_SPATIAL_MERGE_SIZE, QWEN3VL_IMAGE_TEMPORAL_PATCH_SIZE, QWEN3VL_SOURCE_PATH,
    QWEN3VL_SOURCE_SHA256, QWEN35_IMAGE_MEAN, QWEN35_IMAGE_PAD_TOKEN,
    QWEN35_IMAGE_STANDARD_DEVIATION, Qwen3VlMarkerPlan, Qwen3VlPreparedImage,
    QwenMultimodalGenerationRequest, QwenVisionBlockWeights, QwenVisionConfiguration,
    QwenVisionFamily, QwenVisionMergerWeights, QwenVisionProjection, QwenVisionWeights,
    SAM3_CLIP_SOURCE_PATH, SAM3_CLIP_SOURCE_SHA256, Sam3ConditionPack, Sam3EncodedCondition,
    Sam3Prompt, format_gemma_multimodal_prompt, format_ideogram4_prompt, format_ovis_prompt,
    format_qwen_multimodal_prompt, format_qwen3vl_prompt, gemma3_target_dimensions,
    gemma4_audio_marker_tokens, gemma4_target_dimensions, ideogram4_project_taps,
    join_multimodal_embeddings, join_qwen3vl_deepstack, multimodal_profile,
    multimodal_symbol_behavior, ovis_template_end, pack_sam3_conditions, parse_sam3_prompts,
    plan_qwen_markers, plan_qwen3vl_markers, prepare_gemma3_image, prepare_gemma4_audio,
    prepare_gemma4_visuals, prepare_qwen_images, prepare_qwen3vl_images,
    qwen_multimodal_decoder_configuration, qwen_multimodal_tokenizer_profile,
    qwen2vl_mrope_position_ids, qwen3vl_target_dimensions, run_bidirectional_text_owner,
    run_clip_text_owner, run_clip_vision_owner, run_decoder_text_owner, trim_ovis_conditioning,
};
pub use clip_text_encoder_t5::{
    BERT_SOURCE_PATH, BERT_SOURCE_SHA256, BidirectionalFeedForwardActivation,
    BidirectionalLayerWeights, BidirectionalPooling, BidirectionalTextArchitecture,
    BidirectionalTextConfiguration, BidirectionalTextError, BidirectionalTextInput,
    BidirectionalTextOutput, BidirectionalTextRequest, BidirectionalTextWeights,
    NativeT5TextEncoder, SPIECE_TOKENIZER_SOURCE_PATH, SPIECE_TOKENIZER_SOURCE_SHA256,
    T5_BIDIRECTIONAL_CATALOG_SYMBOLS, T5_SOURCE_PATH, T5_SOURCE_SHA256, relative_position_bucket,
    tokenize_bidirectional_prompt,
};
pub use clip_text_encoders::{
    TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT, TEXT_ENCODER_ARCHITECTURE_REGISTRY_VERSION,
    TEXT_ENCODER_OWNER_FACTS, TEXT_ENCODER_SOURCE_SEGMENTS, TextEncoderArchitectureOwner,
    TextEncoderArchitectureRegistry, TextEncoderOwnerFact, TextEncoderRegistryError,
    TextEncoderSourceSegment,
};
pub use clip_tokenizer::{
    CLIP_TOKENIZER_SOURCE_ROWS, ClipBpeTokenizer, GEMMA3_END_OF_TURN_TOKEN, GEMMA3_IMAGE_TOKEN,
    GEMMA4_AUDIO_TOKEN, GEMMA4_IMAGE_TOKEN, GEMMA4_PAD_TOKEN, GEMMA4_START_TOKEN,
    GEMMA4_VIDEO_TOKEN, GemmaTokenizer, GemmaTokenizerProfile, MAX_GEMMA_TOKENIZER_JSON_BYTES,
    MAX_GEMMA_TOKENIZER_VOCABULARY, MAX_NATIVE_EMBEDDING_VALUES, MAX_NATIVE_PROMPT_BATCH,
    MAX_NATIVE_PROMPT_BYTES, MAX_NATIVE_TOKEN_SECTIONS, MAX_NATIVE_WEIGHT_SEGMENTS,
    NativePromptTokenizer, NativeTokenSection, NativeTokenValue, NativeTokenizedPrompt,
    NativeTokenizerError, NativeTokenizerFamily, NativeWeightedToken, PromptWeightSegment,
    QWEN25_TOKENIZER_ARTIFACT_DIGEST, QWEN35_TOKENIZER_ARTIFACT_DIGEST, Qwen2BpeTokenizer,
    Qwen2PretokenizerProfile, SentencePieceTokenizer, TextualInversionEmbedding,
    TokenizerConfiguration, apply_empty_baseline_token_weights, escape_important,
    generate_empty_tokens, parse_parentheses, parse_prompt_weights, token_weights,
    unescape_important,
};
pub use clip_vision::{
    CLIP_VISION_CATALOG_SYMBOLS, CLIP_VISION_RESOURCE_ROLE, CLIP_VISION_SOURCE_PATH,
    CLIP_VISION_SOURCE_SHA256, CLIP_VISION_SOURCE_TYPE_ID, ClipVisionActivation,
    ClipVisionConfiguration, ClipVisionError, ClipVisionIntermediate, ClipVisionLayerWeights,
    ClipVisionModelType, ClipVisionOutput, ClipVisionResidentParts,
    ClipVisionTensorResidentAllocation, ClipVisionWeights, NativeClipVision,
    NativeClipVisionExecutionSession, clip_preprocess_with_context, siglip2_flex_resolution,
    siglip2_preprocess_with_context,
};
pub use cogvideox_family::{
    COGVIDEOX_DETECTION_MARKER_KEYS, COGVIDEOX_LAYOUT_SIGNATURES, COGVIDEOX_PATCH_PROJECTION_KEYS,
    CogVideoXConfiguration, CogVideoXLatentVariant, CogVideoXLayout,
    configuration_for_probe as cogvideox_configuration_for_probe,
};
pub use conditioning_resources::{
    FLUX_REDUX_SOURCE_SHA256, GLIGEN_ATTENTION_SOURCE_SHA256, GLIGEN_OPENAIMODEL_SOURCE_SHA256,
    GLIGEN_SAMPLERS_SOURCE_SHA256, GLIGEN_SOURCE_SHA256, NativeGligenCheckpoint, NativeGligenError,
    NativeGligenFuserLocation, NativeGligenPositionParameter, NativeGligenPreparedPositions,
    NativeGligenRegion, NativeGligenResource, NativePhotoMakerCheckpoint,
    NativePhotoMakerCheckpointEntry, NativePhotoMakerError, NativePhotoMakerResource,
    NativeStyleModelCheckpoint, NativeStyleModelError, NativeStyleModelResource,
    PHOTOMAKER_CLIP_VISION_SOURCE_SHA256, PHOTOMAKER_SOURCE_SHA256, STYLE_ADAPTER_SOURCE_SHA256,
    STYLE_MODEL_NODES_SOURCE_SHA256, STYLE_MODEL_OPS_SOURCE_SHA256, STYLE_MODEL_SD_SOURCE_SHA256,
};
pub use cosmos_family::{
    COSMOS_ANIMA_DETECTION_MARKER_KEYS, COSMOS_CLIP_CANDIDATES, COSMOS_CLIP_CONFIGURATION,
    COSMOS_CLIP_TARGET, COSMOS_GENERAL_DETECTION_MARKER_KEYS, COSMOS_GENERAL_STATE_PLAN,
    COSMOS_GENERAL_STATE_PLAN_CASES, COSMOS_LAYOUT_SIGNATURES, COSMOS_PATCH_PROJECTION_KEYS,
    COSMOS_PREDICT2_DETECTION_MARKER_KEYS, COSMOS_PREDICT2_STATE_PLAN,
    COSMOS_PREDICT2_STATE_PLAN_CASES, COSMOS_SUPPORTED_DEVICES, COSMOS_SUPPORTED_DTYPES,
    COSMOS_WEIGHT_RULES, CosmosArchitecture, CosmosConfiguration, CosmosModelSize, CosmosRatio,
    configuration_for_probe as cosmos_configuration_for_probe,
};
pub use depth_anything_3::{
    DEPTH_ANYTHING_3_CAMERA_SOURCE_SHA256, DEPTH_ANYTHING_3_DINO2_SOURCE_SHA256,
    DEPTH_ANYTHING_3_DPT_SOURCE_SHA256, DEPTH_ANYTHING_3_MODEL_DETECTION_SOURCE_SHA256,
    DEPTH_ANYTHING_3_MODEL_SOURCE_SHA256, DEPTH_ANYTHING_3_PREPROCESS_SOURCE_SHA256,
    DEPTH_ANYTHING_3_RAY_POSE_SOURCE_SHA256, DEPTH_ANYTHING_3_REFERENCE_VIEW_SOURCE_SHA256,
    DEPTH_ANYTHING_3_TRANSFORM_SOURCE_SHA256, NODES_DEPTH_ANYTHING_3_SOURCE_SHA256,
    NativeDepthAnything3Checkpoint, NativeDepthAnything3Error, NativeDepthAnything3Geometry,
    NativeDepthAnything3Invocation, NativeDepthAnything3ReferenceStrategy,
    NativeDepthAnything3ResizeMethod, NativeDepthAnything3Resource,
};
#[cfg(any(test, feature = "test-support"))]
pub use depth_anything_3::{
    DepthAnything3FixtureCheckpointParity, DepthAnything3FixtureMutation,
    DepthAnything3FixtureProfile, DepthAnything3FixtureStateParity,
    NativeDepthAnything3RansacViewTrace, NativeDepthAnything3TestTrace,
    deterministic_reduced_depth_anything_3_checkpoint, mutate_reduced_depth_anything_3_checkpoint,
    reduced_depth_anything_3_checkpoint_parity_for_fixture,
    select_reduced_depth_anything_3_reference_for_fixture,
};
pub use descriptor::{
    CatalogModelDescriptor, MODEL_DESCRIPTOR_SCHEMA_VERSION, MemoryEstimatorDescriptor,
    ModelCatalogAvailability, ModelCatalogConfidence, ModelCatalogKey, ModelCatalogKind,
    ModelComponentDescriptor, ModelDescriptor, ModelDescriptorError, ModelEvidenceLevel,
    ModelParityStatus, TensorKeyRule,
};
pub use flux_chroma_family::{
    CHROMA_LAYOUT_SIGNATURES, FLUX_DIFFUSERS_KEY_NORM_KEYS, FLUX_DIFFUSERS_STATE_PLAN,
    FLUX_GUIDANCE_PROJECTION_KEYS, FLUX_INPUT_PROJECTION_KEYS, FLUX_LAYOUT_SIGNATURES,
    FLUX_NATIVE_KEY_NORM_KEYS, FLUX_STATE_PLAN_CASES, FLUX_TEXT_PROJECTION_KEYS,
    FLUX2_DISCRIMINATOR_KEYS, FluxChromaConfiguration, FluxChromaFinalHead, FluxChromaLayout,
    FluxChromaVariant, configuration_for_probe as flux_chroma_configuration_for_probe,
};
pub use formats::{
    ArchiveEntry, FileSlice, GgufValue, MAX_EMBEDDING_ARCHIVE_VALUES, ModelFormat,
    ModelFormatError, ParsedModel, ParsedModelPayload, SentencePieceType, SentencePieceVocabulary,
    SentencePieceVocabularyEntry, TensorMetadata, TorchArchiveFileLoader, detect_model_format,
    load_torch_archive_file, parse_model_file, parse_safetensors_bytes,
};
pub use frame_interpolation::{
    FrameInterpolationError, FrameInterpolationFallbackState, FrameInterpolationInvocationPlan,
    FrameInterpolationProfile, NativeFrameInterpolationModel,
};
pub use hidream_o1_family::{
    HIDREAM_O1_ARCHITECTURE_VERSION, HIDREAM_O1_ASSISTANT_TOKEN_ID, HIDREAM_O1_BOI_TOKEN_ID,
    HIDREAM_O1_BOR_TOKEN_ID, HIDREAM_O1_BOT_TOKEN_ID, HIDREAM_O1_CLIP_CANDIDATES,
    HIDREAM_O1_CLIP_TARGET, HIDREAM_O1_COMPONENT_STATE_SCHEMAS, HIDREAM_O1_COMPONENTS,
    HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT, HIDREAM_O1_EOR_TOKEN_ID, HIDREAM_O1_IM_END_TOKEN_ID,
    HIDREAM_O1_IM_START_TOKEN_ID, HIDREAM_O1_IMAGE_TOKEN_ID, HIDREAM_O1_LATENT_FEATURE_ID,
    HIDREAM_O1_LATENT_FORMAT, HIDREAM_O1_LATENT_IDENTIFIER, HIDREAM_O1_LAYOUT_SIGNATURES,
    HIDREAM_O1_MEMORY_USAGE_FACTOR, HIDREAM_O1_NATIVE_STATE_PLAN, HIDREAM_O1_NEWLINE_TOKEN_ID,
    HIDREAM_O1_PAD_TOKEN_ID, HIDREAM_O1_PATCH_SIZE, HIDREAM_O1_PIXEL_VAE_SENTINEL,
    HIDREAM_O1_STATE_PLAN_CASES, HIDREAM_O1_SUPPORTED_DEVICES, HIDREAM_O1_SUPPORTED_DTYPES,
    HIDREAM_O1_TEXT_ENCODER_SENTINEL, HIDREAM_O1_TMS_TOKEN_ID, HIDREAM_O1_UNPREFIXED_STATE_PLAN,
    HIDREAM_O1_USER_TOKEN_ID, HIDREAM_O1_VIDEO_TOKEN_ID, HIDREAM_O1_VISION_END_TOKEN_ID,
    HIDREAM_O1_VISION_IMAGE_MEAN, HIDREAM_O1_VISION_IMAGE_STD, HIDREAM_O1_VISION_MERGE_SIZE,
    HIDREAM_O1_VISION_PATCH_SIZE, HIDREAM_O1_VISION_START_TOKEN_ID, HIDREAM_O1_WEIGHT_RULES,
    HiDreamO1Configuration, HiDreamO1Layout,
    configuration_for_probe as hidream_o1_configuration_for_probe,
};
pub use hunyuan_video_family::{
    HUNYUAN_IMAGE_CLIP_CANDIDATES, HUNYUAN_IMAGE_CLIP_CONFIGURATION, HUNYUAN_IMAGE_CLIP_TARGET,
    HUNYUAN_IMAGE21_LATENT_FORMAT, HUNYUAN_IMAGE21_REFINER_LATENT_FORMAT,
    HUNYUAN_REFINER_IMAGE_SCALE, HUNYUAN_REFINER_SEED_OFFSET, HUNYUAN_VIDEO_BYT5_INPUT_DIMENSION,
    HUNYUAN_VIDEO_BYT5_INTERMEDIATE_DIMENSION, HUNYUAN_VIDEO_CLIP_CANDIDATES,
    HUNYUAN_VIDEO_CLIP_CONFIGURATION, HUNYUAN_VIDEO_CLIP_TARGET,
    HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS, HUNYUAN_VIDEO_COMPONENTS, HUNYUAN_VIDEO_FORWARD_PROGRAM,
    HUNYUAN_VIDEO_HEAD_DIMENSION, HUNYUAN_VIDEO_LATENT_FORMAT, HUNYUAN_VIDEO_MLP_RATIO,
    HUNYUAN_VIDEO_MODEL_REQUIRED_KEYS, HUNYUAN_VIDEO_PREFIXED_STATE_PLAN,
    HUNYUAN_VIDEO_SAVE_PREFIX, HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN,
    HUNYUAN_VIDEO_STANDALONE_STATE_PLAN, HUNYUAN_VIDEO_SUPPORTED_DEVICES,
    HUNYUAN_VIDEO_SUPPORTED_DTYPES, HUNYUAN_VIDEO_THETA, HUNYUAN_VIDEO_VECTOR_INPUT_DIMENSION,
    HUNYUAN_VIDEO15_CLIP_CANDIDATES, HUNYUAN_VIDEO15_CLIP_TARGET, HUNYUAN_VIDEO15_LATENT_FORMAT,
    HUNYUAN_VIDEO15_SUPPORTED_DTYPES, HUNYUAN_VIDEO15_VISION_INPUT_DIMENSION,
    HunyuanVideoConfiguration, HunyuanVideoLayout, HunyuanVideoVariant,
    augment_refiner_conditioning, configuration_for_probe as hunyuan_video_configuration_for_probe,
    state_plan_for_layout as hunyuan_video_state_plan_for_layout,
};
pub use hunyuan3d_family::{
    HUNYUAN3D_COMMON_MAPPING, HUNYUAN3D_COMPONENTS, HUNYUAN3D_MEMORY_USAGE_FACTOR,
    HUNYUAN3D_MINI_DEPTH, HUNYUAN3D_MINI_LATENT_FORMAT, HUNYUAN3D_MLP_RATIO,
    HUNYUAN3D_NUMBER_OF_HEADS, HUNYUAN3D_PREFIXED_STATE_PLAN, HUNYUAN3D_SAVED_MODEL_STATE_PLAN,
    HUNYUAN3D_SCALE_SUFFIX, HUNYUAN3D_STANDALONE_STATE_PLAN, HUNYUAN3D_STANDARD_STATE_PLAN_CASES,
    HUNYUAN3D_SUPPORTED_DEVICES, HUNYUAN3D_SUPPORTED_DTYPES, HUNYUAN3D_V2_LATENT_FORMAT,
    HUNYUAN3D_V21_CONTEXT_DIMENSION, HUNYUAN3D_V21_LATENT_FORMAT, HUNYUAN3D_WEIGHT_SUFFIX,
    Hunyuan3DCommonMapping, Hunyuan3DConfiguration, Hunyuan3DLayout, Hunyuan3DVariant,
    common_mapping as hunyuan3d_common_mapping,
    configuration_for_probe as hunyuan3d_configuration_for_probe,
    state_plan_for_layout as hunyuan3d_state_plan_for_layout,
};
pub use hunyuandit_family::{
    HUNYUANDIT_BASE_EXTRA_INPUT, HUNYUANDIT_CLIP_CANDIDATES, HUNYUANDIT_CLIP_TARGET,
    HUNYUANDIT_CLIP_TEXT_DIMENSION, HUNYUANDIT_CLIP_TEXT_LENGTH, HUNYUANDIT_COMMON_MAPPING,
    HUNYUANDIT_COMPONENTS, HUNYUANDIT_DEFAULT_MLP_RATIO, HUNYUANDIT_DIT1_EXTRA_INPUT,
    HUNYUANDIT_FORWARD_PROGRAM, HUNYUANDIT_G_DEPTH, HUNYUANDIT_G_HIDDEN_SIZE,
    HUNYUANDIT_G_MLP_RATIO, HUNYUANDIT_IMAGE_META_DIMENSION,
    HUNYUANDIT_IMAGE_META_EMBEDDING_DIMENSION, HUNYUANDIT_INPUT_CHANNELS, HUNYUANDIT_LATENT_FORMAT,
    HUNYUANDIT_LINEAR_END, HUNYUANDIT_LINEAR_START, HUNYUANDIT_MEMORY_USAGE_FACTOR,
    HUNYUANDIT_NUMBER_OF_HEADS, HUNYUANDIT_PATCH_SIZE, HUNYUANDIT_PREFIXED_STATE_PLAN,
    HUNYUANDIT_SAVED_MODEL_STATE_PLAN, HUNYUANDIT_STANDALONE_STATE_PLAN,
    HUNYUANDIT_STANDARD_STATE_PLAN_CASES, HUNYUANDIT_SUPPORTED_DEVICES,
    HUNYUANDIT_SUPPORTED_DTYPES, HUNYUANDIT_T5_TEXT_DIMENSION, HUNYUANDIT_T5_TEXT_LENGTH,
    HUNYUANDIT1_LINEAR_END, HunyuanDiTAttentionPrecision, HunyuanDiTCommonMapping,
    HunyuanDiTConfiguration, HunyuanDiTLayout, HunyuanDiTVariant,
    common_mapping as hunyuandit_common_mapping,
    configuration_for_probe as hunyuandit_configuration_for_probe,
    state_plan_for_layout as hunyuandit_state_plan_for_layout,
};
pub use kandinsky5_family::{
    KANDINSKY5_CLIP_CONFIGURATION, KANDINSKY5_COMMON_MAPPING, KANDINSKY5_COMPONENT_STATE_SCHEMAS,
    KANDINSKY5_COMPONENTS, KANDINSKY5_DIFFUSERS_MARKER, KANDINSKY5_FORWARD_PROGRAM,
    KANDINSKY5_IMAGE_CLIP_CANDIDATES, KANDINSKY5_IMAGE_CLIP_TARGET, KANDINSKY5_IMAGE_CONDITIONING,
    KANDINSKY5_IMAGE_LATENT_FORMAT, KANDINSKY5_IMAGE_LITE_MODEL_DIMENSION,
    KANDINSKY5_IMAGE_ROPE_SCALE_FACTOR, KANDINSKY5_IMAGE_SAMPLING_SHIFT,
    KANDINSKY5_IMAGE_VISUAL_EMBED_DIMENSION, KANDINSKY5_LAYOUT_SIGNATURES,
    KANDINSKY5_MEMORY_USAGE_FACTOR, KANDINSKY5_MODEL_OPTIONAL_KEYS, KANDINSKY5_MODEL_REQUIRED_KEYS,
    KANDINSKY5_OUTPUT_CHANNELS, KANDINSKY5_PATCH_SIZE, KANDINSKY5_POOLED_TEXT_INPUT_DIMENSION,
    KANDINSKY5_PREFIXED_STATE_PLAN, KANDINSKY5_ROPE_THETA, KANDINSKY5_STANDALONE_STATE_PLAN,
    KANDINSKY5_STATE_PLAN_CASES, KANDINSKY5_SUPPORTED_DEVICES, KANDINSKY5_SUPPORTED_DTYPES,
    KANDINSKY5_TEXT_BLOCK_COUNT, KANDINSKY5_TEXT_INPUT_DIMENSION, KANDINSKY5_TIME_DIMENSION,
    KANDINSKY5_VIDEO_CLIP_CANDIDATES, KANDINSKY5_VIDEO_CLIP_TARGET, KANDINSKY5_VIDEO_CONDITIONING,
    KANDINSKY5_VIDEO_LATENT_FORMAT, KANDINSKY5_VIDEO_LITE_AXES_DIMENSIONS,
    KANDINSKY5_VIDEO_LITE_MODEL_DIMENSION, KANDINSKY5_VIDEO_PRO_MODEL_DIMENSION,
    KANDINSKY5_VIDEO_ROPE_SCALE_FACTOR, KANDINSKY5_VIDEO_SAMPLING_SHIFT,
    KANDINSKY5_VIDEO_VISUAL_EMBED_DIMENSION, KANDINSKY5_VISUAL_BLOCK_COUNT,
    KANDINSKY5_WIDE_AXES_DIMENSIONS, Kandinsky5CommonMapping, Kandinsky5ConditioningFact,
    Kandinsky5Configuration, Kandinsky5Layout, Kandinsky5Variant,
    common_mapping as kandinsky5_common_mapping,
    configuration_for_probe as kandinsky5_configuration_for_probe,
    state_plan_for_layout as kandinsky5_state_plan_for_layout,
};
pub use latent_format::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDefinition, LatentFormatDescriptor,
    LatentFormatError, LatentFormatIdentity, LatentFormatRegistry, LatentTensorLayout,
    LatentTransform, PreviewReshape, empty_latent, process_latent_in, process_latent_out,
    project_latent_preview,
};
pub use latent_upscale_model::{
    HUNYUAN_UPSAMPLER_SOURCE_SHA256, HunyuanLatentUpscaleCrop, HunyuanLatentUpscaleMode,
    LTX_UPSAMPLER_SOURCE_SHA256, NODES_HUNYUAN_SOURCE_SHA256, NativeLatentUpscaleArchitecture,
    NativeLatentUpscaleCheckpoint, NativeLatentUpscaleModelError, NativeLatentUpscaleModelResource,
    NativeLtxUpscaleConfiguration,
};
pub use ltx_family::{
    LTX_AUDIO_MARKER, LTX_CLIP_CANDIDATES, LTX_CLIP_CONFIGURATION, LTX_CLIP_TARGET,
    LTX_COMMON_MAPPING, LTX_COMPONENT_STATE_SCHEMAS, LTX_COMPONENTS, LTX_DIFFUSERS_MARKER,
    LTX_FORWARD_PROGRAM, LTX_MAX_TRANSFORMER_CONFIG_BYTES, LTX_MODEL_OPTIONAL_KEYS,
    LTX_MODEL_REQUIRED_KEYS, LTX_PIXART_COLLISION_MARKER, LTX_PREFIXED_STATE_PLAN,
    LTX_SAVED_MODEL_STATE_PLAN, LTX_STANDALONE_STATE_PLAN, LTX_SUPPORTED_DEVICES,
    LTX_SUPPORTED_DTYPES, LTX_TIMESTEP_MARKER, LTXAV_CONDITIONING, LTXAV_LATENT_FORMAT,
    LTXAV_MEMORY_USAGE_FACTOR, LTXV_BASE_MEMORY_USAGE_FACTOR, LTXV_CONDITIONING,
    LTXV_LATENT_FORMAT, LTXV_SAMPLING_SHIFT, LtxConditioningFact, LtxConfiguration, LtxLayout,
    LtxVariant, common_mapping as ltx_common_mapping,
    configuration_for_probe as ltx_configuration_for_probe, ltxv_memory_usage_factor,
    state_plan_for_layout as ltx_state_plan_for_layout,
};
pub use lumina_zimage_family::{
    LUMINA_AXES_DIMENSIONS, LUMINA_AXES_LENGTHS, LUMINA_CLIP_CANDIDATES, LUMINA_CLIP_CONFIGURATION,
    LUMINA_CLIP_TARGET, LUMINA_DIMENSION, LUMINA_HEAD_COUNT, LUMINA_INPUT_CHANNELS,
    LUMINA_KV_HEAD_COUNT, LUMINA_MEMORY_USAGE_FACTOR, LUMINA_PATCH_SIZE, LUMINA_ROPE_THETA,
    LUMINA_SAMPLING_SHIFT, LUMINA_ZIMAGE_COMMON_MAPPING, LUMINA_ZIMAGE_COMPONENT_STATE_SCHEMAS,
    LUMINA_ZIMAGE_COMPONENTS, LUMINA_ZIMAGE_CONDITIONING, LUMINA_ZIMAGE_FORWARD_PROGRAM,
    LUMINA_ZIMAGE_LATENT_FORMAT, LUMINA_ZIMAGE_MODEL_OPTIONAL_KEYS,
    LUMINA_ZIMAGE_MODEL_REQUIRED_KEYS, LUMINA_ZIMAGE_PREFIXED_STATE_PLAN,
    LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN, LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
    LUMINA_ZIMAGE_SUPPORTED_DEVICES, LUMINA_ZIMAGE_SUPPORTED_DTYPES, LuminaZImageCommonMapping,
    LuminaZImageConditioningFact, LuminaZImageConfiguration, LuminaZImageLayout,
    LuminaZImageVariant, ZIMAGE_AXES_DIMENSIONS, ZIMAGE_AXES_LENGTHS, ZIMAGE_CLIP_CANDIDATES,
    ZIMAGE_CLIP_CONFIGURATION, ZIMAGE_CLIP_TARGET, ZIMAGE_DIFFUSERS_STATE_PLAN, ZIMAGE_DIMENSION,
    ZIMAGE_HEAD_COUNT, ZIMAGE_KV_HEAD_COUNT, ZIMAGE_MEMORY_USAGE_FACTOR,
    ZIMAGE_PAD_TOKENS_MULTIPLE, ZIMAGE_PIXEL_LATENT_FORMAT, ZIMAGE_PIXEL_MEMORY_USAGE_FACTOR,
    ZIMAGE_ROPE_THETA, ZIMAGE_SAMPLING_SHIFT, ZIMAGE_TIME_SCALE, ZImagePixelDecoderConfiguration,
    common_mapping as lumina_zimage_common_mapping,
    configuration_for_probe as lumina_zimage_configuration_for_probe,
    state_plan_for_layout as lumina_zimage_state_plan_for_layout,
};
pub use model_family::{
    MODEL_CLIP_TARGET_SCHEMA_VERSION, MODEL_FAMILY_SCHEMA_VERSION,
    MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION, MappedModelComponents, MappedModelWeights,
    ModelBaseFallback, ModelClipConfigurationFact, ModelClipConfigurationFactDefinition,
    ModelClipModelDescriptor, ModelClipModelInvocation, ModelClipModelInvocationDefinition,
    ModelClipTargetCandidateDefinition, ModelClipTargetCandidateDescriptor, ModelClipTargetCase,
    ModelClipTargetDefinition, ModelClipTargetDescriptor, ModelClipTargetSelector,
    ModelConfigurationKind, ModelConfigurationValue, ModelDetection, ModelDetectionOutcome,
    ModelDetectionPolicy, ModelDetectionRule, ModelDimensionEvaluationContext,
    ModelDimensionExpression, ModelFamilyComponent, ModelFamilyComponentStateSchema,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyIdentity, ModelFamilyProfile,
    ModelFamilyProfileSelector, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelFamilyStatePlanCase, ModelFamilyStatePlanProbeSelector, ModelFamilyStatePlanSelector,
    ModelFamilyWeightBinding, ModelForwardCheckpoint, ModelForwardOperation, ModelForwardStep,
    ModelKeyPredicate, ModelKeyRewrite, ModelKeySelector, ModelLayoutSignature,
    ModelMemoryEstimate, ModelNativeTargetIdentifier, ModelNormalizedConfiguration,
    ModelOptionalKeyReplacement, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact,
    ModelPerTensorTransform, ModelProbe, ModelRoundCondition, ModelSourceConfigurationRule,
    ModelSplitOutputRule, ModelStateLayout, ModelStateTarget, ModelStateTensorReference,
    ModelStateTransaction, ModelStateTransformOperation, ModelStateTransformPlan,
    ModelStateTransformPlanDefinition, ModelStorageDType, ModelTensorFactPredicate,
    ModelTensorFactRelation, ModelTensorFactSubject, ModelTokenizerDescriptor,
    ModelTransformBranchOutputRule, ModelUnetPrefixSelection, ModelUnmatchedKeyDisposition,
    ModelWeightRule, NativeFamilyBuildOptions, NativeFamilyModel, ResolvedModelFamily,
    build_model_family, build_model_family_for_probe, describe_model_family,
    detect_model_family_rules, estimate_model_memory, map_model_weights,
};
#[cfg(feature = "test-support")]
pub use model_family::{NativeModelRecipeSourceForm, auraflow_recipe_fixture_state_schema};
#[cfg(unix)]
pub use model_store::ReadOnlyTensorMapping;
pub use model_store::{
    LoadedModel, ModelFamilyProbeError, ModelFamilyProbeErrorKind, ModelLoadAccounting,
    ModelOperationRecord, ModelOperationStage, ModelStore, ModelStoreError,
    ModelStreamNestedStateDisposition, VerifiedEmbeddingArchivePayload,
    VerifiedModelStreamArtifact, VerifiedModelStreamSource, VerifiedModelStreamTensor,
    VerifiedModelTensor, VerifiedModelTensorPayload, VerifiedSentencePieceVocabulary,
};
pub use moge::{
    MOGE_DINO2_SOURCE_SHA256, MOGE_GEOMETRY_SOURCE_SHA256, MOGE_MODEL_SOURCE_SHA256,
    MOGE_MODULES_SOURCE_SHA256, NODES_MOGE_SOURCE_SHA256, NativeMogeCheckpoint, NativeMogeError,
    NativeMogeGeometry, NativeMogeInvocation, NativeMogeNestedCheckpoint, NativeMogeResource,
    NativeMogeVersion,
};
#[cfg(any(test, feature = "test-support"))]
pub use moge::{
    MogeFixtureMutation, MogeFixtureProfile, deterministic_reduced_moge_checkpoint,
    mutate_reduced_moge_checkpoint,
};
pub use native_node_payload::{
    AudioEncoderOutput, AudioEncoderOutputKind, IcLoraParameters, LossMap,
    NativeLoraApplicationMode, NativeLoraModelResource, NativeModelBackingKind, NativeModelPayload,
    NativeModelPayloadError, NativeModelResidentAllocation, NativeModelResidentParts,
    NativeModelResourceIdentity, NativeModelResourceRole, NativeModelTensorResidentAllocation,
    NativeStructuredResidentParts,
};
pub use native_ops::{
    CastedParameters, ConvolutionAutopad, EmbeddingOptions, GeluApproximation, LossReduction,
    NativeExecutionRequirements, NativeModelExecutionAdmission, NativeModelExecutionRole,
    NativeModule, NativeModuleSpec, NativeOperationSet, NativeOpsError, PrefetchReceipt,
    RngAwareModuleForward, UpsampleMode, WeightCastLease,
    adaptive_average_pool_2d_module_exact_native, admit_native_model_resource_execution,
    average_pool_1d_module_exact_native, average_pool_2d_module_exact_native,
    average_pool_3d_module_exact_native, batch_norm_1d_module_exact_native,
    batch_norm_2d_module_exact_native, buffer_module_exact_native,
    cast_modules_with_vbar_with_context_exact_native, conv_2d_module_exact_native,
    conv_3d_module_exact_native, conv1d_module_exact_native,
    disable_weight_init_conv1d_exact_native, disable_weight_init_convolution_exact_native,
    disable_weight_init_group_norm_exact_native, disable_weight_init_layer_norm_exact_native,
    disable_weight_init_linear_exact_native, dropout_module_exact_native, elu_module_exact_native,
    embedding_module_exact_native, gelu_module_exact_native, group_norm_module_exact_native,
    huber_loss_module_exact_native, identity_module_exact_native,
    instance_norm_2d_module_exact_native, l1_loss_module_exact_native,
    layer_norm_module_exact_native, leaky_relu_module_exact_native, linear_module_exact_native,
    manual_cast_layer_norm_exact_native, manual_cast_linear_exact_native,
    max_pool_2d_module_exact_native, mixed_precision_ops_exact_native, module_dict_exact_native,
    module_exact_native, module_init_exact_native, module_list_exact_native,
    mse_loss_module_exact_native, multihead_attention_module_exact_native,
    pick_operations_exact_native, pixel_shuffle_module_exact_native,
    pixel_unshuffle_module_exact_native, prelu_module_exact_native, relu_6_module_exact_native,
    relu_module_exact_native, remove_parametrizations_with_context_exact_native,
    replication_pad_2d_module_exact_native, sequential_module_exact_native,
    sigmoid_module_exact_native, silu_module_exact_native, smooth_l1_loss_module_exact_native,
    softmax_module_exact_native, spectral_norm_exact_native, tanh_module_exact_native,
    upsample_module_exact_native, weight_norm_exact_native, zero_pad_2d_module_exact_native,
};
pub use omnigen2_boogu_family::{
    BOOGU_AXES_DIMENSIONS, BOOGU_AXES_LENGTHS, BOOGU_CLIP_CANDIDATES, BOOGU_CLIP_CONFIGURATION,
    BOOGU_CLIP_TARGET, BOOGU_COMPONENT_STATE_SCHEMAS, BOOGU_DETECTION_MARKER_KEYS,
    BOOGU_DETECTION_RULES, BOOGU_FORWARD_PROGRAM, BOOGU_HEAD_COUNT, BOOGU_INPUT_CHANNELS,
    BOOGU_KV_HEAD_COUNT, BOOGU_MEMORY_ESTIMATOR, BOOGU_MEMORY_USAGE_FACTOR,
    BOOGU_MODEL_OPTIONAL_KEYS, BOOGU_MODEL_REQUIRED_KEYS, BOOGU_MULTIPLE_OF, BOOGU_PATCH_SIZE,
    BOOGU_SAMPLING_SHIFT, BOOGU_SUPPORTED_DTYPES, BOOGU_TIMESTEP_SCALE, OMNIGEN2_AXES_DIMENSIONS,
    OMNIGEN2_AXES_LENGTHS, OMNIGEN2_BASE_SUPPORTED_DTYPES, OMNIGEN2_BOOGU_COMPONENTS,
    OMNIGEN2_BOOGU_CONDITIONING, OMNIGEN2_BOOGU_LATENT_FORMAT, OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN,
    OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN, OMNIGEN2_BOOGU_SUPPORTED_DEVICES,
    OMNIGEN2_CLIP_CANDIDATES, OMNIGEN2_CLIP_CONFIGURATION, OMNIGEN2_CLIP_TARGET,
    OMNIGEN2_DETECTION_MARKER_KEYS, OMNIGEN2_EXTENDED_SUPPORTED_DTYPES, OMNIGEN2_FORWARD_PROGRAM,
    OMNIGEN2_HEAD_COUNT, OMNIGEN2_HIDDEN_SIZE, OMNIGEN2_INPUT_CHANNELS, OMNIGEN2_KV_HEAD_COUNT,
    OMNIGEN2_LAYER_COUNT, OMNIGEN2_MEMORY_ESTIMATOR, OMNIGEN2_MEMORY_USAGE_FACTOR,
    OMNIGEN2_MODEL_OPTIONAL_KEYS, OMNIGEN2_MODEL_REQUIRED_KEYS, OMNIGEN2_MULTIPLE_OF,
    OMNIGEN2_PATCH_SIZE, OMNIGEN2_REFINER_LAYER_COUNT, OMNIGEN2_SAMPLING_SHIFT,
    OMNIGEN2_TEXT_FEATURE_DIMENSION, OMNIGEN2_TIMESTEP_SCALE, Omnigen2BooguConditioningFact,
    Omnigen2BooguConfiguration, Omnigen2BooguLayout, Omnigen2BooguVariant,
    configuration_for_probe as omnigen2_boogu_configuration_for_probe,
    state_plan_for_layout as omnigen2_boogu_state_plan_for_layout,
    supported_dtypes_for_capabilities as omnigen2_boogu_supported_dtypes_for_capabilities,
};
pub use parser_limits::{
    PARSER_DECODED_ALLOCATION_MULTIPLIER, PARSER_LIMITS_VERSION, ParserLimitError, ParserLimits,
};
pub use patch_graph::{
    NestedPatch, PATCH_GRAPH_SCHEMA_VERSION, PatchApplication, PatchComputeBoundary, PatchGraph,
    PatchGraphError, PatchGraphIdentity, PatchGraphIdentityError, PatchKind, PatchOperation,
    PatchPayload, PatchSlice, PatchTarget, PatchTensor, PatchValueTransform,
    SemanticPatchOperation, factorize_patch_dimension,
};
pub use pixart_family::{
    PIXART_ALPHA_CONDITIONING_KEYS, PIXART_ALPHA_LATENT_FORMAT, PIXART_BETA_SCHEDULE,
    PIXART_CAPTION_CHANNELS, PIXART_CLIP_CANDIDATES, PIXART_CLIP_TARGET,
    PIXART_COMPONENT_STATE_SCHEMAS, PIXART_COMPONENTS, PIXART_FORWARD_PROGRAM, PIXART_HEAD_COUNT,
    PIXART_HIDDEN_SIZE, PIXART_INPUT_CHANNELS, PIXART_LINEAR_END, PIXART_LINEAR_START,
    PIXART_MAX_DEPTH, PIXART_MAX_MODEL_LENGTH, PIXART_MEMORY_ESTIMATOR, PIXART_MEMORY_USAGE_FACTOR,
    PIXART_MLP_RATIO, PIXART_MODEL_OPTIONAL_KEYS, PIXART_MODEL_REQUIRED_KEYS, PIXART_PATCH_SIZE,
    PIXART_PREFIXED_NATIVE_STATE_PLAN, PIXART_SIGMA_CONDITIONING_KEYS, PIXART_SIGMA_LATENT_FORMAT,
    PIXART_STANDALONE_NATIVE_STATE_PLAN, PIXART_SUPPORTED_DEVICES, PIXART_SUPPORTED_DTYPES,
    PIXART_TIMESTEPS, PixArtConditioningKey, PixArtConfiguration, PixArtLayout, PixArtVariant,
    conditioning_keys_for_variant as pixart_conditioning_keys_for_variant,
    configuration_for_probe as pixart_configuration_for_probe,
    diffusers_state_plan as pixart_diffusers_state_plan,
    native_state_plan_for_layout as pixart_native_state_plan_for_layout,
};
pub use pixeldit_pid_family::{
    PID_CONDITIONING_KEYS, PID_FORWARD_PROGRAM, PID_SAMPLING_SHIFT, PID_SR_SCALE,
    PIXELDIT_CLIP_CANDIDATES, PIXELDIT_CLIP_TARGET, PIXELDIT_CONDITIONING_KEYS,
    PIXELDIT_CORE_STATE_PLAN, PIXELDIT_FORWARD_PROGRAM, PIXELDIT_GROUP_COUNT, PIXELDIT_HIDDEN_SIZE,
    PIXELDIT_INPUT_CHANNELS, PIXELDIT_NET_STATE_PLAN, PIXELDIT_PATCH_DEPTH, PIXELDIT_PATCH_SIZE,
    PIXELDIT_PID_COMPONENT_STATE_SCHEMAS, PIXELDIT_PID_COMPONENTS, PIXELDIT_PID_LATENT_FORMAT,
    PIXELDIT_PID_MEMORY_USAGE_FACTOR, PIXELDIT_PID_MODEL_OPTIONAL_KEYS,
    PIXELDIT_PID_MODEL_REQUIRED_KEYS, PIXELDIT_PID_SUPPORTED_DEVICES,
    PIXELDIT_PID_SUPPORTED_DTYPES, PIXELDIT_PIXEL_ATTENTION_HIDDEN_SIZE, PIXELDIT_PIXEL_DEPTH,
    PIXELDIT_PIXEL_GROUP_COUNT, PIXELDIT_PIXEL_HIDDEN_SIZE, PIXELDIT_SAMPLING_SHIFT,
    PIXELDIT_TEXT_FEATURE_DIMENSION, PIXELDIT_TEXT_MAX_LENGTH, PIXELDIT_TEXT_ROPE_THETA,
    PiDConfiguration, PixelDitPidConditioningKey, PixelDitPidConfiguration, PixelDitPidLayout,
    PixelDitPidVariant,
    conditioning_keys_for_variant as pixeldit_pid_conditioning_keys_for_variant,
    configuration_for_probe as pixeldit_pid_configuration_for_probe,
    forward_program_for_variant as pixeldit_pid_forward_program_for_variant,
    state_plan_for_layout as pixeldit_pid_state_plan_for_layout,
};
pub use quantization::{
    LayerQuantizationV1, QuantLinearLayout, QuantLinearScale, QuantizationError, QuantizationKind,
    QuantizationMetadataV1, QuantizedContentIdentity, QuantizedLinearMatrix,
    QuantizedMaterialization, QuantizedMatrix, QuantizedSourceIdentity, quantize_linear_matrix,
    quantize_matrix,
};
pub use quantized_autograd::{
    QuantLinearError, QuantLinearExecution, QuantLinearGradients, QuantLinearOptions,
    QuantLinearWeight, quant_linear_forward_exact_native,
};
pub use qwen_image_family::{
    QWEN_IMAGE_ATTENTION_HEAD_DIMENSION, QWEN_IMAGE_AXES_DIMENSIONS,
    QWEN_IMAGE_BASE_CONDITIONING_KEYS, QWEN_IMAGE_BLOCK_PREFIXES, QWEN_IMAGE_CLIP_CANDIDATES,
    QWEN_IMAGE_CLIP_CONFIGURATION, QWEN_IMAGE_CLIP_TARGET, QWEN_IMAGE_COMPONENT_STATE_SCHEMAS,
    QWEN_IMAGE_COMPONENTS, QWEN_IMAGE_INNER_DIMENSION, QWEN_IMAGE_JOINT_ATTENTION_DIMENSION,
    QWEN_IMAGE_LATENT_FORMAT, QWEN_IMAGE_LAYERED_CONDITIONING_KEYS, QWEN_IMAGE_MAXIMUM_DEPTH,
    QWEN_IMAGE_MAXIMUM_LAYERED_SLICES, QWEN_IMAGE_MEMORY_ESTIMATOR, QWEN_IMAGE_MEMORY_USAGE_FACTOR,
    QWEN_IMAGE_MODEL_OPTIONAL_KEYS, QWEN_IMAGE_MODEL_REQUIRED_KEYS,
    QWEN_IMAGE_NUMBER_OF_ATTENTION_HEADS, QWEN_IMAGE_PATCH_SIZE,
    QWEN_IMAGE_POOLED_PROJECTION_DIMENSION, QWEN_IMAGE_SAMPLING_SHIFT,
    QWEN_IMAGE_SUPPORTED_DEVICES, QWEN_IMAGE_SUPPORTED_DTYPES, QwenImageBlockPrefix,
    QwenImageConditioningKey, QwenImageConfiguration, QwenImageReferenceMethod,
    checked_patch_graph as qwen_image_checked_patch_graph,
    configuration_for_probe as qwen_image_configuration_for_probe,
    layered_latent_extent as qwen_image_layered_latent_extent,
};
pub use registry_generator::{
    MODEL_CATALOG, ModelRegistry, ModelRegistryError, ModelRegistryGenerator,
};
pub use restricted_pickle::{
    ALLOWED_PICKLE_TARGETS, AllowedPickleTarget, PickleValue, RESTRICTED_PICKLE_ALLOWLIST_VERSION,
    RESTRICTED_PICKLE_DECODED_ALLOCATION_MULTIPLIER, RestrictedPickleError, SafeGlobalsAdmission,
    add_safe_globals_exact_native, parse_restricted_pickle, parse_restricted_pickle_cancellable,
};
pub use sd2_family::{
    LOTUS_CONDITIONING, SD2_ATTENTION_HEAD_CHANNELS, SD2_CHANNEL_MULTIPLIERS, SD2_CLIP_CANDIDATES,
    SD2_CLIP_TARGET, SD2_COMMON_MAPPING, SD2_COMPONENT_STATE_SCHEMAS, SD2_COMPONENTS,
    SD2_CONDITIONING, SD2_CONTEXT_DIMENSION, SD2_DIFFUSERS_STATE_PLAN, SD2_FORWARD_PROGRAM,
    SD2_LATENT_FORMAT, SD2_LAYOUT_SIGNATURES, SD2_MEMORY_USAGE_FACTOR, SD2_MODEL_CHANNELS,
    SD2_MODEL_OPTIONAL_KEYS, SD2_MODEL_REQUIRED_KEYS, SD2_NUM_RES_BLOCKS, SD2_PREFIXED_STATE_PLAN,
    SD2_SUPPORTED_DEVICES, SD2_SUPPORTED_DTYPES, SD2_TRANSFORMER_DEPTH,
    SD2_TRANSFORMER_DEPTH_OUTPUT, SD2_UNCLIP_BETA_SCHEDULE, SD2_UNCLIP_H_CONFIGURATION,
    SD2_UNCLIP_L_CONFIGURATION, SD2_UNCLIP_NOISE_AUGMENT_MERGE, SD2_UNCLIP_SEED_OFFSET,
    SD2_UNCLIP_TIMESTEPS, SD2_V_PREDICTION_THRESHOLD, Sd2CommonMapping, Sd2ConditioningFact,
    Sd2Configuration, Sd2Layout, Sd2ModelType, Sd2UnclipConfiguration, Sd2Variant,
    UNCLIP_CONDITIONING, common_mapping as sd2_common_mapping,
    configuration_for_probe as sd2_configuration_for_probe, lotus_task_embedding,
    state_plan_for_layout as sd2_state_plan_for_layout,
    weight_statistic_request_for_probe as sd2_weight_statistic_request_for_probe,
};
pub use sdpose::{
    NativeSdPoseHeatmapHead, NativeSdPoseModel, NativeSdPoseSd2Denoiser, SDPOSE_HEAD_SOURCE_SHA256,
    SDPOSE_HEATMAP_CHANNELS, SDPOSE_HEATMAP_HEIGHT, SDPOSE_HEATMAP_WIDTH, SDPOSE_INPUT_HEIGHT,
    SDPOSE_INPUT_WIDTH, SDPOSE_MODEL_DETECTION_SOURCE_SHA256, SdPoseHeatmapHeadConfiguration,
    SdPoseHeatmapHeadWeightSpec, SdPoseModelError, SdPoseProjectionError, SdPoseRawKeypoint,
    SdPoseSd2Configuration, SdPoseSd2Error, SdPoseSd2ForwardOutput, SdPoseSd2WeightSpec,
    decode_sdpose_heatmaps, project_sdpose_openpose_person, sdpose_heatmap_head_weight_manifest,
    sdpose_sd2_weight_manifest,
};
pub use sdxl_family::{
    SDXL_ADM_INPUT_DIMENSION, SDXL_ATTENTION_HEAD_CHANNELS, SDXL_CLIP_CANDIDATES, SDXL_CLIP_TARGET,
    SDXL_COMMON_MAPPING, SDXL_COMPONENT_STATE_SCHEMAS, SDXL_COMPONENTS, SDXL_CONTEXT_DIMENSION,
    SDXL_DIFFUSERS_STATE_PLAN, SDXL_FORWARD_PROGRAM, SDXL_KOALA_1B_TRANSFORMER_DEPTH,
    SDXL_KOALA_1B_TRANSFORMER_DEPTH_OUTPUT, SDXL_KOALA_700M_TRANSFORMER_DEPTH,
    SDXL_KOALA_700M_TRANSFORMER_DEPTH_OUTPUT, SDXL_LATENT_FORMAT, SDXL_LAYOUT_SIGNATURES,
    SDXL_MEMORY_USAGE_FACTOR, SDXL_MODEL_CHANNELS, SDXL_MODEL_OPTIONAL_KEYS,
    SDXL_MODEL_REQUIRED_KEYS, SDXL_PREFIXED_STATE_PLAN, SDXL_REFINER_ADM_INPUT_DIMENSION,
    SDXL_REFINER_CLIP_CANDIDATES, SDXL_REFINER_CLIP_TARGET, SDXL_REFINER_CONTEXT_DIMENSION,
    SDXL_REFINER_MEMORY_USAGE_FACTOR, SDXL_REFINER_MODEL_CHANNELS, SDXL_REFINER_TRANSFORMER_DEPTH,
    SDXL_REFINER_TRANSFORMER_DEPTH_OUTPUT, SDXL_SEGMIND_TRANSFORMER_DEPTH,
    SDXL_SEGMIND_TRANSFORMER_DEPTH_OUTPUT, SDXL_SSD1B_TRANSFORMER_DEPTH,
    SDXL_SSD1B_TRANSFORMER_DEPTH_OUTPUT, SDXL_STANDALONE_STATE_PLAN, SDXL_STATE_PLAN_CASES,
    SDXL_SUPPORTED_DEVICES, SDXL_SUPPORTED_DTYPES, SDXL_TRANSFORMER_DEPTH,
    SDXL_TRANSFORMER_DEPTH_OUTPUT, SdxlCommonMapping, SdxlConfiguration, SdxlLayout, SdxlVariant,
    common_mapping as sdxl_common_mapping, configuration_for_probe as sdxl_configuration_for_probe,
    state_plan_for_layout as sdxl_state_plan_for_layout,
};
pub use upscale_contract::{
    NATIVE_UPSCALE_ADMITTED_ARCHITECTURE_COUNT, NATIVE_UPSCALE_ARCHITECTURE_COUNT,
    NATIVE_UPSCALE_CONTRACT_ID, NATIVE_UPSCALE_CONTRACT_SCHEMA_VERSION,
    NATIVE_UPSCALE_CONTRACT_SHA256, NativeUpscaleArchitectureContract,
    NativeUpscaleCanonicalStateKeys, NativeUpscaleContractError, NativeUpscaleDetection,
    NativeUpscaleRuntimeContract, NativeUpscaleSourceFile, NativeUpscaleStateDictionaryLayout,
    compiled_native_upscale_contract, validate_native_upscale_contract_candidate,
};
pub use upscale_model::{
    NativeUpscaleModelError, NativeUpscaleModelResource, NativeUpscaleUnavailable,
    NativeUpscaleUnavailableReason,
};
pub use vae::validate_native_vae_backend_target;
pub use vae::{
    LtxLatentStatisticsDirection, NativeStructuredVae, NativeVae, VAE_SCHEMA_VERSION,
    VaeArchitectureIdentity, VaeBoundary, VaeBoundaryKind, VaeCanonicalCompatibility,
    VaeDescriptor, VaeError, VaeGaussianSplatBatch, VaeIdentity, VaeKernelProfile, VaeOperation,
    VaeShapeField, VaeStructuredDecodeRequest, VaeStructuredOutputKind, VaeStructuredResult,
    VaeTileAxisFormula, VaeTilePlan,
};
pub use vae_architecture::{
    VAE_AUTOMATIC_ROW_ID, VAE_DIFFUSERS_ROW_ID, VAE_DIFFUSERS_SENTINEL, VAE_SELECTOR_BRANCH_COUNT,
    VAE_SELECTOR_CATALOG_ROWS, VAE_SELECTOR_ROW_COUNT, VAE_SELECTOR_SOURCE_PATH,
    VAE_SELECTOR_SOURCE_SHA256, VAE_UNBOUND_ROW_ID, VaeArchitectureError, VaeArchitectureRegistry,
    VaeArchitectureSelection, VaeBoundaryDomain, VaeCatalogRowKind, VaeExecutionTarget,
    VaeLoaderConfiguration, VaeSelectorCatalogRow,
};
pub use vae_audio::{
    AudioVaeError, AudioVaeSourceCheckpoint, NativeAudioVaeArchitecture, audio_vae_source_plan,
    audio_vae_source_state_schema, inspect_audio_vae_architecture,
    load_audio_vae_from_model_store_with_context,
};
pub use vae_image::{
    ImageVaeError, NativeImageVaeArchitecture, image_vae_source_state_schema,
    inspect_image_vae_architecture,
};
pub use vae_structured::{
    HUNYUAN_SHAPE_ARCHITECTURE, NativeStructuredVaeArchitecture, StructuredVaeError,
    StructuredVaeStateCheckpoint, TRIPO_GAUSSIAN_FEATURES_PER_TOKEN, TRIPO_GAUSSIANS_PER_TOKEN,
    TRIPO_MAX_OCTREE_LEVEL, TRIPO_SPLAT_ARCHITECTURE, hammersley_3d, level_embedding,
    load_structured_vae_from_model_store_with_context, radical_inverse, shape_grid_coordinates,
    shape_output_from_logits, structured_vae_source_plan, structured_vae_source_state_count,
    structured_vae_source_state_schema, systematic_sample_counts,
    tripo_gaussian_output_from_predictions,
};
pub use vae_video::{
    NativeVideoVaeArchitecture, VideoVaeError, VideoVaeSourceCheckpoint,
    inspect_video_vae_architecture, load_video_vae_from_model_store_with_context,
    video_vae_source_plan, video_vae_source_state_schema,
};
pub use vision_models::{
    EFFICIENTNET_V2_S_OPERATION_ID, NativeEfficientNetBlockKind, NativeEfficientNetStage,
    NativeEfficientNetV2S, NativeEfficientNetV2SFeatureSource, NativeRaftLarge,
    NativeRaftLargeExecutionSession, NativeRaftResidentParts, NativeRaftTensorResidentAllocation,
    NativeVisionModelError, NativeVisionStateKind, NativeVisionStateSpec, RAFT_LARGE_OPERATION_ID,
    RAFT_LARGE_RESOURCE_ROLE, RAFT_LARGE_SOURCE_TYPE_ID, efficientnet_v2_s_exact_native,
    efficientnet_v2_s_features_from_module_with_context,
    load_stage_c_efficientnet_feature_module_from_model_store_with_context,
    load_vision_state_from_model_store_with_context,
    load_vision_state_with_sibling_namespaces_from_model_store_with_context,
    raft_large_exact_native,
};
pub use weight_adapter::{
    ADAPTER_MAP_ORDER, AdapterFamily, AdapterTensor, BypassBinding, BypassForwardHook,
    BypassInjectionManager, BypassPatch, BypassRuntimePlan, LayerKind, LoadedWeightAdapter,
    ModuleTypeInfo, NativeWeightAdapter, TrainableAdapterKind, TrainableWeightOutput,
    WEIGHT_ADAPTER_ORDER, WeightAdapterError, WeightAdapterLoadRequest, WeightAdapterRegistry,
};

include!(concat!(env!("OUT_DIR"), "/generated_modules.rs"));
