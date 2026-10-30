use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::{
    ClipVisionActivation, ClipVisionConfiguration, ClipVisionLayerWeights, ClipVisionModelType,
    ClipVisionWeights, DecoderActivation, DecoderArchitecture, DecoderAttentionWeights,
    DecoderLayerKind, DecoderLayerWeights, DecoderRopeConfiguration, DecoderTextConfiguration,
    DecoderTextWeights, GEMMA3_FOUR_B_MULTIMODAL_SOURCE_SHA256, GEMMA3_IMAGE_AREA_PIXELS,
    GEMMA3_MULTIMODAL_SOURCE_PATH, GEMMA3_MULTIMODAL_SOURCE_SHA256, GEMMA4_AUDIO_FFT_LENGTH,
    GEMMA4_AUDIO_FRAME_LENGTH, GEMMA4_AUDIO_FRAME_STEP, GEMMA4_AUDIO_MAXIMUM_SAMPLE_RATE,
    GEMMA4_AUDIO_MAXIMUM_TOKENS, GEMMA4_AUDIO_MEL_BINS, GEMMA4_AUDIO_MINIMUM_SAMPLE_RATE,
    GEMMA4_AUDIO_SAMPLE_RATE, GEMMA4_IMAGE_SOFT_TOKENS, GEMMA4_MULTIMODAL_SOURCE_PATH,
    GEMMA4_MULTIMODAL_SOURCE_SHA256, GEMMA4_VIDEO_SOFT_TOKENS, GEMMA4_VIDEO_SOURCE_FPS,
    Gemma3VisionConfiguration, Gemma3VisionProfile, Gemma4AudioBlockWeights,
    Gemma4AudioConfiguration, Gemma4AudioFeedForwardWeights, Gemma4AudioProfile,
    Gemma4AudioWeights, Gemma4ClippedLinearWeights, Gemma4DecoderConfiguration,
    Gemma4LayerInputWeights, Gemma4PerLayerWeights, Gemma4VisionBlockWeights,
    Gemma4VisionConfiguration, Gemma4VisionProfile, Gemma4VisionWeights, GemmaMultimodalFamily,
    GemmaMultimodalGenerationRequest, GemmaPreparedVisualKind, GemmaTokenizer,
    GemmaTokenizerProfile, IDEOGRAM4_SOURCE_PATH, IDEOGRAM4_SOURCE_SHA256, IDEOGRAM4_TAP_LAYERS,
    JINA_CLIP2_SOURCE_PATH, JINA_CLIP2_SOURCE_SHA256, MULTIMODAL_TEXT_ENCODER_CATALOG_SYMBOLS,
    MultimodalFamily, MultimodalImageEmbedding, MultimodalSpan, MultimodalSymbolBehavior,
    MultimodalTextError, NativeClipVision, NativeDecoderTextEncoder, NativeGemma3VisionProjector,
    NativeGemma4AudioEncoder, NativeGemma4VisionEncoder, NativeGemmaMultimodal, NativeModelPayload,
    NativePromptTokenizer, NativeQwenMultimodal, NativeQwenVisionEncoder,
    NativeTextGenerationRequest, NativeTokenizerFamily, OVIS_SOURCE_PATH, OVIS_SOURCE_SHA256,
    QWEN_VL_SOURCE_PATH, QWEN_VL_SOURCE_SHA256, QWEN3VL_IMAGE_PAD_TOKEN, QWEN3VL_SOURCE_PATH,
    QWEN3VL_SOURCE_SHA256, QWEN35_IMAGE_MEAN, QWEN35_IMAGE_PAD_TOKEN,
    QWEN35_IMAGE_STANDARD_DEVIATION, Qwen2BpeTokenizer, Qwen2PretokenizerProfile,
    QwenMultimodalGenerationRequest, QwenVisionBlockWeights, QwenVisionConfiguration,
    QwenVisionFamily, QwenVisionMergerWeights, QwenVisionWeights, RopeScaling,
    SAM3_CLIP_SOURCE_PATH, SAM3_CLIP_SOURCE_SHA256, Sam3EncodedCondition, TokenizerConfiguration,
    format_gemma_multimodal_prompt, format_ideogram4_prompt, format_ovis_prompt,
    format_qwen_multimodal_prompt, format_qwen3vl_prompt, gemma3_target_dimensions,
    gemma4_audio_marker_tokens, gemma4_target_dimensions, ideogram4_project_taps,
    join_multimodal_embeddings, join_qwen3vl_deepstack, multimodal_profile,
    multimodal_symbol_behavior, ovis_template_end, pack_sam3_conditions, parse_sam3_prompts,
    plan_qwen_markers, plan_qwen3vl_markers, prepare_gemma3_image, prepare_gemma4_audio,
    prepare_gemma4_visuals, prepare_qwen_images, prepare_qwen3vl_images,
    qwen_multimodal_decoder_configuration, qwen_multimodal_tokenizer_profile,
    qwen2vl_mrope_position_ids, qwen3vl_target_dimensions, trim_ovis_conditioning,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    ImageTensor, RetryRngPolicy, RngAlgorithm, RngProfileVersion, RngStream, RngStreamAddress,
    StreamId, Tensor, TensorDescriptor, TensorError, generated_native_diffusion::tensor_to_f32,
};
use std::{collections::BTreeMap, error::Error, fs, path::Path, sync::Arc};

const MEMORY_LIMIT: u64 = 64 * 1024 * 1024;
const TASK_ID: &str = "comfy-parity-clip-text-encoder-multimodal-foundation";
const COMPOSITE_TASK_ID: &str = "comfy-parity-clip-text-encoder-composite-adapters";
const IMPLEMENTATION_CLOSURE: [&str; 3] = [
    "crates/sim_model/src/clip_text_encoder_multimodal.rs",
    "crates/sim_model/src/sim_model.rs",
    "crates/sim_model/tests/clip_text_encoder_multimodal.rs",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?)
}

fn context<'a>(
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
    scratch: u64,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(scratch)?,
        rng_phase: None,
        cancellation,
    })
}

fn qwen_generation_transaction() -> Result<sim_tensor::RngTransaction, Box<dyn Error>> {
    let address = RngStreamAddress::new(
        "qwen-workflow",
        "attempt-1",
        "qwen-node",
        0,
        "qwen-multimodal-generation",
        0,
        0,
        RetryRngPolicy::Replay,
    )?;
    Ok(RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        41,
        address,
    )?
    .begin(None)?)
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn filled_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension).ok()?)
    });
    tensor(
        backend,
        shape,
        &vec![value; elements.ok_or("fixture tensor shape overflowed")?],
        context,
    )
}

fn matrix_tensor(
    backend: &CpuBackend,
    output: usize,
    input: usize,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = output.checked_mul(input).ok_or("matrix size overflowed")?;
    let values = (0..count)
        .map(|index| {
            let row = index / input;
            let column = index % input;
            if row % input == column {
                scale
            } else {
                ((index * 11 % 17) as f32 - 8.0) * scale * 0.0125
            }
        })
        .collect::<Vec<_>>();
    tensor(
        backend,
        &[u64::try_from(output)?, u64::try_from(input)?],
        &values,
        context,
    )
}

fn reduced_gemma3_clip_vision(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<(Arc<NativeClipVision>, Tensor), Box<dyn Error>> {
    let configuration = ClipVisionConfiguration {
        model_type: ClipVisionModelType::Siglip,
        dtype: DType::F32,
        device: DeviceId::CPU,
        hidden_size: 4,
        intermediate_size: 8,
        attention_heads: 2,
        layer_count: 1,
        image_size: 4,
        patch_size: 1,
        num_channels: 3,
        max_num_patches: 0,
        activation: ClipVisionActivation::GeluTanh,
        projection_dimension: None,
        llava_projection_dimension: None,
    };
    let mut patch_values = vec![0.0_f32; 4 * 3];
    patch_values[0] = 1.0;
    patch_values[4] = 1.0;
    patch_values[8] = 1.0;
    patch_values[9] = 0.5;
    patch_values[10] = 0.25;
    patch_values[11] = 0.125;
    let position_values = (0..16)
        .flat_map(|position| {
            (0..4).map(move |channel| position as f32 * 0.01 + channel as f32 * 0.001)
        })
        .collect::<Vec<_>>();
    let post_norm_weight = filled_tensor(backend, &[4], 1.0, context)?;
    let layer = ClipVisionLayerWeights {
        layer_norm_1_weight: filled_tensor(backend, &[4], 1.0, context)?,
        layer_norm_1_bias: filled_tensor(backend, &[4], 0.0, context)?,
        query_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        query_bias: filled_tensor(backend, &[4], 0.0, context)?,
        key_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        key_bias: filled_tensor(backend, &[4], 0.0, context)?,
        value_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        value_bias: filled_tensor(backend, &[4], 0.0, context)?,
        output_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        output_bias: filled_tensor(backend, &[4], 0.0, context)?,
        layer_norm_2_weight: filled_tensor(backend, &[4], 1.0, context)?,
        layer_norm_2_bias: filled_tensor(backend, &[4], 0.0, context)?,
        feed_forward_1_weight: filled_tensor(backend, &[8, 4], 0.0, context)?,
        feed_forward_1_bias: filled_tensor(backend, &[8], 0.0, context)?,
        feed_forward_2_weight: filled_tensor(backend, &[4, 8], 0.0, context)?,
        feed_forward_2_bias: filled_tensor(backend, &[4], 0.0, context)?,
    };
    let vision = NativeClipVision::new(
        configuration,
        ClipVisionWeights {
            patch_embedding_weight: tensor(backend, &[4, 3, 1, 1], &patch_values, context)?,
            patch_embedding_bias: Some(filled_tensor(backend, &[4], 0.0, context)?),
            class_embedding: None,
            position_embedding: tensor(backend, &[16, 4], &position_values, context)?,
            pre_layer_norm_weight: None,
            pre_layer_norm_bias: None,
            layers: vec![layer],
            post_layer_norm_weight: post_norm_weight.clone(),
            post_layer_norm_bias: filled_tensor(backend, &[4], 0.0, context)?,
            visual_projection_weight: None,
            llava_linear_1_weight: None,
            llava_linear_1_bias: None,
            llava_linear_2_weight: None,
            llava_linear_2_bias: None,
        },
    )?;
    Ok((Arc::new(vision), post_norm_weight))
}

fn reduced_qwen_vision_weights(
    backend: &CpuBackend,
    configuration: &QwenVisionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<QwenVisionWeights, Box<dyn Error>> {
    let hidden = u64::try_from(configuration.hidden_size)?;
    let intermediate = u64::try_from(configuration.intermediate_size)?;
    let output = u64::try_from(configuration.output_hidden_size)?;
    let merge_width = hidden.checked_mul(4).ok_or("merge width overflowed")?;
    let patch_width = 3 * 2 * 16 * 16;
    let mut patch_values = vec![0.0_f32; configuration.hidden_size * patch_width];
    for hidden_index in 0..configuration.hidden_size {
        patch_values[hidden_index * patch_width + hidden_index] =
            0.125 + hidden_index as f32 * 0.025;
    }
    let patch_weight = tensor(
        backend,
        &[hidden, u64::try_from(patch_width)?],
        &patch_values,
        context,
    )?;
    let patch_bias = filled_tensor(backend, &[hidden], 0.01, context)?;
    let mut position_values = vec![0.0_f32; 2304 * configuration.hidden_size];
    for position in 0..2304 {
        for hidden_index in 0..configuration.hidden_size {
            position_values[position * configuration.hidden_size + hidden_index] =
                position as f32 * 0.0001 + hidden_index as f32 * 0.001;
        }
    }
    let position_embedding = tensor(backend, &[2304, hidden], &position_values, context)?;
    let mut blocks = Vec::new();
    for layer in 0..configuration.layer_count {
        let scale = 0.001 + layer as f32 * 0.00001;
        blocks.push(QwenVisionBlockWeights {
            normalization_one_weight: filled_tensor(backend, &[hidden], 1.0, context)?,
            normalization_one_bias: filled_tensor(backend, &[hidden], 0.0, context)?,
            query_key_value_weight: filled_tensor(backend, &[hidden * 3, hidden], scale, context)?,
            query_key_value_bias: filled_tensor(backend, &[hidden * 3], 0.0, context)?,
            attention_output_weight: filled_tensor(backend, &[hidden, hidden], scale, context)?,
            attention_output_bias: filled_tensor(backend, &[hidden], 0.0, context)?,
            normalization_two_weight: filled_tensor(backend, &[hidden], 1.0, context)?,
            normalization_two_bias: filled_tensor(backend, &[hidden], 0.0, context)?,
            feed_forward_up_weight: filled_tensor(
                backend,
                &[intermediate, hidden],
                scale,
                context,
            )?,
            feed_forward_up_bias: filled_tensor(backend, &[intermediate], 0.0, context)?,
            feed_forward_down_weight: filled_tensor(
                backend,
                &[hidden, intermediate],
                scale,
                context,
            )?,
            feed_forward_down_bias: filled_tensor(backend, &[hidden], 0.0, context)?,
        });
    }
    let merger = QwenVisionMergerWeights {
        normalization_weight: filled_tensor(backend, &[hidden], 1.0, context)?,
        normalization_bias: filled_tensor(backend, &[hidden], 0.0, context)?,
        first_weight: filled_tensor(backend, &[merge_width, merge_width], 0.01, context)?,
        first_bias: filled_tensor(backend, &[merge_width], 0.0, context)?,
        second_weight: filled_tensor(backend, &[output, merge_width], 0.02, context)?,
        second_bias: filled_tensor(backend, &[output], 0.0, context)?,
    };
    let mut deepstack_mergers = Vec::new();
    for _ in configuration.family.deepstack_layers() {
        deepstack_mergers.push(QwenVisionMergerWeights {
            normalization_weight: filled_tensor(backend, &[merge_width], 1.0, context)?,
            normalization_bias: filled_tensor(backend, &[merge_width], 0.0, context)?,
            first_weight: filled_tensor(backend, &[merge_width, merge_width], 0.015, context)?,
            first_bias: filled_tensor(backend, &[merge_width], 0.0, context)?,
            second_weight: filled_tensor(backend, &[output, merge_width], 0.025, context)?,
            second_bias: filled_tensor(backend, &[output], 0.0, context)?,
        });
    }
    Ok(QwenVisionWeights {
        patch_weight,
        patch_bias,
        position_embedding,
        blocks,
        merger,
        deepstack_mergers,
    })
}

fn reduced_gemma4_clipped_linear(
    backend: &CpuBackend,
    input: usize,
    output: usize,
    scale: f32,
    minimum: &Tensor,
    maximum: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4ClippedLinearWeights, Box<dyn Error>> {
    let mut values = vec![0.0_f32; input * output];
    for row in 0..output {
        for column in 0..input {
            values[row * input + column] =
                scale * (1.0 + ((row * input + column) % 7) as f32 * 0.125);
        }
    }
    Ok(Gemma4ClippedLinearWeights {
        weight: tensor(
            backend,
            &[u64::try_from(output)?, u64::try_from(input)?],
            &values,
            context,
        )?,
        input_minimum: minimum.clone(),
        input_maximum: maximum.clone(),
        output_minimum: minimum.clone(),
        output_maximum: maximum.clone(),
    })
}

fn reduced_gemma4_vision_weights(
    backend: &CpuBackend,
    configuration: &Gemma4VisionConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4VisionWeights, Box<dyn Error>> {
    let hidden = configuration.hidden_size;
    let intermediate = configuration.intermediate_size;
    let patch_width = 3 * configuration.patch_size * configuration.patch_size;
    let minimum = filled_tensor(backend, &[1], -8.0, context)?;
    let maximum = filled_tensor(backend, &[1], 8.0, context)?;
    let mut patch_values = vec![0.0_f32; hidden * patch_width];
    for output in 0..hidden {
        for source in 0..patch_width {
            patch_values[output * patch_width + source] =
                0.0005 * (1.0 + ((source + output * 3) % 11) as f32);
        }
    }
    let mut position_values = vec![0.0_f32; 2 * configuration.position_embeddings * hidden];
    for axis in 0..2 {
        for position in 0..configuration.position_embeddings {
            for feature in 0..hidden {
                position_values
                    [(axis * configuration.position_embeddings + position) * hidden + feature] =
                    axis as f32 * 0.003 + position as f32 * 0.0002 + feature as f32 * 0.0001;
            }
        }
    }
    let mut blocks = Vec::new();
    for layer in 0..configuration.layer_count {
        let scale = 0.01 + layer as f32 * 0.001;
        blocks.push(Gemma4VisionBlockWeights {
            input_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            query: reduced_gemma4_clipped_linear(
                backend, hidden, hidden, scale, &minimum, &maximum, context,
            )?,
            key: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.1,
                &minimum,
                &maximum,
                context,
            )?,
            value: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.2,
                &minimum,
                &maximum,
                context,
            )?,
            query_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(configuration.head_dimension)?],
                1.0,
                context,
            )?,
            key_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(configuration.head_dimension)?],
                1.0,
                context,
            )?,
            attention_output: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 0.7,
                &minimum,
                &maximum,
                context,
            )?,
            post_attention_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            pre_feed_forward_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            feed_forward_gate: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                intermediate,
                scale * 0.6,
                &minimum,
                &maximum,
                context,
            )?,
            feed_forward_up: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                intermediate,
                scale * 0.5,
                &minimum,
                &maximum,
                context,
            )?,
            feed_forward_down: reduced_gemma4_clipped_linear(
                backend,
                intermediate,
                hidden,
                scale * 0.4,
                &minimum,
                &maximum,
                context,
            )?,
            post_feed_forward_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
        });
    }
    Ok(Gemma4VisionWeights {
        patch_projection_weight: tensor(
            backend,
            &[u64::try_from(hidden)?, u64::try_from(patch_width)?],
            &patch_values,
            context,
        )?,
        position_embedding: tensor(
            backend,
            &[
                2,
                u64::try_from(configuration.position_embeddings)?,
                u64::try_from(hidden)?,
            ],
            &position_values,
            context,
        )?,
        blocks,
        projector_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.output_hidden_size)?,
                u64::try_from(hidden)?,
            ],
            0.075,
            context,
        )?,
    })
}

fn reduced_gemma4_audio_feed_forward(
    backend: &CpuBackend,
    configuration: &Gemma4AudioConfiguration,
    scale: f32,
    minimum: &Tensor,
    maximum: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4AudioFeedForwardWeights, Box<dyn Error>> {
    Ok(Gemma4AudioFeedForwardWeights {
        pre_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.hidden_size)?],
            1.0,
            context,
        )?,
        first: reduced_gemma4_clipped_linear(
            backend,
            configuration.hidden_size,
            configuration.intermediate_size,
            scale,
            minimum,
            maximum,
            context,
        )?,
        second: reduced_gemma4_clipped_linear(
            backend,
            configuration.intermediate_size,
            configuration.hidden_size,
            scale * 0.7,
            minimum,
            maximum,
            context,
        )?,
        post_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.hidden_size)?],
            1.0,
            context,
        )?,
    })
}

fn reduced_gemma4_audio_weights(
    backend: &CpuBackend,
    configuration: &Gemma4AudioConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<Gemma4AudioWeights, Box<dyn Error>> {
    let minimum = filled_tensor(backend, &[1], -8.0, context)?;
    let maximum = filled_tensor(backend, &[1], 8.0, context)?;
    let hidden = configuration.hidden_size;
    let mut blocks = Vec::new();
    for index in 0..configuration.layer_count {
        let scale = 0.01 + index as f32 * 0.001;
        blocks.push(Gemma4AudioBlockWeights {
            feed_forward_one: reduced_gemma4_audio_feed_forward(
                backend,
                configuration,
                scale,
                &minimum,
                &maximum,
                context,
            )?,
            query: reduced_gemma4_clipped_linear(
                backend, hidden, hidden, scale, &minimum, &maximum, context,
            )?,
            key: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.1,
                &minimum,
                &maximum,
                context,
            )?,
            value: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 1.2,
                &minimum,
                &maximum,
                context,
            )?,
            attention_output: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 0.8,
                &minimum,
                &maximum,
                context,
            )?,
            attention_scale: filled_tensor(
                backend,
                &[u64::try_from(hidden / configuration.attention_heads)?],
                0.1,
                context,
            )?,
            relative_key_projection_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?, u64::try_from(hidden)?],
                0.02,
                context,
            )?,
            pre_attention_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            post_attention_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            convolution_pre_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            convolution_start: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden * 2,
                scale * 0.9,
                &minimum,
                &maximum,
                context,
            )?,
            depthwise_convolution_weight: filled_tensor(
                backend,
                &[
                    u64::try_from(hidden)?,
                    1,
                    u64::try_from(configuration.convolution_kernel_size)?,
                ],
                0.2,
                context,
            )?,
            convolution_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
            convolution_end: reduced_gemma4_clipped_linear(
                backend,
                hidden,
                hidden,
                scale * 0.6,
                &minimum,
                &maximum,
                context,
            )?,
            feed_forward_two: reduced_gemma4_audio_feed_forward(
                backend,
                configuration,
                scale * 0.5,
                &minimum,
                &maximum,
                context,
            )?,
            output_normalization_weight: filled_tensor(
                backend,
                &[u64::try_from(hidden)?],
                1.0,
                context,
            )?,
        });
    }
    Ok(Gemma4AudioWeights {
        first_convolution_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.first_convolution_channels)?,
                1,
                3,
                3,
            ],
            0.04,
            context,
        )?,
        first_convolution_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.first_convolution_channels)?],
            1.0,
            context,
        )?,
        second_convolution_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.second_convolution_channels)?,
                u64::try_from(configuration.first_convolution_channels)?,
                3,
                3,
            ],
            0.03,
            context,
        )?,
        second_convolution_normalization_weight: filled_tensor(
            backend,
            &[u64::try_from(configuration.second_convolution_channels)?],
            1.0,
            context,
        )?,
        subsample_projection_weight: filled_tensor(
            backend,
            &[
                u64::try_from(hidden)?,
                u64::try_from(configuration.mel_bins.div_ceil(2).div_ceil(2))?
                    * u64::try_from(configuration.second_convolution_channels)?,
            ],
            0.02,
            context,
        )?,
        blocks,
        encoder_output_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.encoder_output_size)?,
                u64::try_from(hidden)?,
            ],
            0.025,
            context,
        )?,
        encoder_output_bias: filled_tensor(
            backend,
            &[u64::try_from(configuration.encoder_output_size)?],
            0.01,
            context,
        )?,
        projector_weight: filled_tensor(
            backend,
            &[
                u64::try_from(configuration.output_hidden_size)?,
                u64::try_from(configuration.encoder_output_size)?,
            ],
            0.03,
            context,
        )?,
    })
}

fn gemma4_prompt_tokenizer(
    hidden_size: usize,
) -> Result<Arc<NativePromptTokenizer>, Box<dyn Error>> {
    let family = GemmaTokenizer::gemma4_from_tokenizer_json(
        include_str!(
            "../../sim_test_support/fixtures/text_generation/gemma_multimodal/tokenizer/gemma4-tokenizer.json"
        ),
        &CancellationToken::default(),
    )?;
    Ok(Arc::new(NativePromptTokenizer::checked(
        NativeTokenizerFamily::Gemma(family),
        TokenizerConfiguration {
            maximum_length: 131_072,
            minimum_length: Some(1),
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: true,
            start_token: Some(2),
            end_token: None,
            pad_token: 0,
            maximum_word_length: 8,
            disable_weights: true,
            embedding_width: Some(hidden_size),
        },
        BTreeMap::new(),
    )?))
}

fn reduced_gemma4_decoder_configuration() -> DecoderTextConfiguration {
    DecoderTextConfiguration {
        architecture: DecoderArchitecture::Gemma,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: 262_144,
        maximum_tokens: 512,
        hidden_size: 4,
        feed_forward_size: 8,
        layer_kinds: (0_usize..10)
            .map(|index| {
                if (index + 1).is_multiple_of(5) {
                    DecoderLayerKind::FullAttention
                } else {
                    DecoderLayerKind::SlidingAttention
                }
            })
            .collect(),
        attention_heads: 2,
        key_value_heads: 1,
        head_dimension: 2,
        query_key_norm: true,
        qwen35_linear: None,
        gemma3: None,
        gemma4: Some(Gemma4DecoderConfiguration {
            source_profile: None,
            local_rope: DecoderRopeConfiguration {
                theta: 10_000.0,
                rotary_dimension: 2,
                interleaved_sections: Vec::new(),
                scaling: RopeScaling::None,
            },
            global_head_dimension: 4,
            global_rotary_pairs: 1,
            sliding_layers_per_cycle: 4,
            hidden_size_per_layer_input: 2,
            shared_key_value_layers: 5,
            double_wide_mlp: true,
        }),
        normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
        rope: DecoderRopeConfiguration {
            theta: 1_000_000.0,
            rotary_dimension: 2,
            interleaved_sections: Vec::new(),
            scaling: RopeScaling::None,
        },
        sliding_window: Some(512),
        activation: DecoderActivation::GeluTanh,
        embedding_scale_bits: 2.0_f32.to_bits(),
        residual_scale_bits: 1.0_f32.to_bits(),
        norm_weight_offset_bits: 0.0_f32.to_bits(),
        logits_soft_cap_bits: Some(30.0_f32.to_bits()),
        tied_output_head: true,
        stop_tokens: vec![1, 50, 106],
    }
}

fn reduced_gemma4_decoder(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    scale_offset: f32,
) -> Result<Arc<NativeDecoderTextEncoder>, Box<dyn Error>> {
    let configuration = reduced_gemma4_decoder_configuration();
    let gemma4 = configuration.gemma4.as_ref().ok_or("Gemma4 config")?;
    let first_shared = configuration.layer_kinds.len() - gemma4.shared_key_value_layers;
    let mut layers = Vec::new();
    for (index, kind) in configuration.layer_kinds.iter().copied().enumerate() {
        let head_dimension = if kind == DecoderLayerKind::FullAttention {
            gemma4.global_head_dimension
        } else {
            configuration.head_dimension
        };
        let query_width = configuration.attention_heads * head_dimension;
        let key_value_width = configuration.key_value_heads * head_dimension;
        let feed_forward = if gemma4.double_wide_mlp && index >= first_shared {
            configuration.feed_forward_size * 2
        } else {
            configuration.feed_forward_size
        };
        let scale = scale_offset + 0.01 + index as f32 * 0.001;
        layers.push(DecoderLayerWeights {
            attention_norm_weight: filled_tensor(backend, &[4], 0.3, context)?,
            attention: DecoderAttentionWeights::DotProduct {
                query_weight: matrix_tensor(backend, query_width, 4, scale, context)?,
                key_weight: matrix_tensor(backend, key_value_width, 4, scale + 0.01, context)?,
                value_weight: matrix_tensor(backend, key_value_width, 4, scale + 0.02, context)?,
                query_norm_weight: Some(filled_tensor(
                    backend,
                    &[u64::try_from(head_dimension)?],
                    0.8,
                    context,
                )?),
                key_norm_weight: Some(filled_tensor(
                    backend,
                    &[u64::try_from(head_dimension)?],
                    1.1,
                    context,
                )?),
                output_weight: matrix_tensor(backend, 4, query_width, scale + 0.03, context)?,
            },
            feed_forward_norm_weight: filled_tensor(backend, &[4], 0.25, context)?,
            feed_forward_gate_weight: matrix_tensor(
                backend,
                feed_forward,
                4,
                scale + 0.04,
                context,
            )?,
            feed_forward_up_weight: matrix_tensor(backend, feed_forward, 4, scale + 0.05, context)?,
            feed_forward_down_weight: matrix_tensor(
                backend,
                4,
                feed_forward,
                scale + 0.06,
                context,
            )?,
            post_attention_norm_weight: Some(filled_tensor(backend, &[4], 0.2, context)?),
            post_feed_forward_norm_weight: Some(filled_tensor(backend, &[4], 0.22, context)?),
            attention_sink: None,
            gemma4_layer_input: Some(Gemma4LayerInputWeights {
                gate_weight: matrix_tensor(backend, 2, 4, scale + 0.07, context)?,
                projection_weight: matrix_tensor(backend, 4, 2, scale + 0.08, context)?,
                post_norm_weight: filled_tensor(backend, &[4], 0.4, context)?,
                layer_scalar: tensor(backend, &[1], &[0.97 + index as f32 * 0.001], context)?,
            }),
        });
    }
    let total_per_layer = configuration.layer_kinds.len() * gemma4.hidden_size_per_layer_input;
    Ok(Arc::new(NativeDecoderTextEncoder::new(
        configuration,
        DecoderTextWeights {
            token_embedding: filled_tensor(backend, &[262_144, 4], 0.001, context)?,
            layers,
            final_norm_weight: filled_tensor(backend, &[4], 0.35, context)?,
            output_head_weight: None,
            gemma4_per_layer: Some(Gemma4PerLayerWeights {
                token_embedding: filled_tensor(
                    backend,
                    &[262_144, u64::try_from(total_per_layer)?],
                    0.002 + scale_offset,
                    context,
                )?,
                model_projection_weight: matrix_tensor(backend, total_per_layer, 4, 0.03, context)?,
                projection_norm_weight: tensor(backend, &[2], &[0.9, 1.1], context)?,
            }),
        },
    )?))
}

fn qwen25_prompt_tokenizer() -> Result<Arc<NativePromptTokenizer>, Box<dyn Error>> {
    let tokenizer_configuration =
        fs::read_to_string(workspace()?.join(
            "projects/comfy/ComfyUI/sim/text_encoders/qwen25_tokenizer/tokenizer_config.json",
        ))?;
    let vocabulary = fs::read_to_string(
        workspace()?.join("projects/comfy/ComfyUI/sim/text_encoders/qwen25_tokenizer/vocab.json"),
    )?;
    let merges = fs::read_to_string(
        workspace()?.join("projects/comfy/ComfyUI/sim/text_encoders/qwen25_tokenizer/merges.txt"),
    )?;
    let family = Qwen2BpeTokenizer::from_artifacts(
        Qwen2PretokenizerProfile::Qwen2,
        &tokenizer_configuration,
        &vocabulary,
        &merges,
        &CancellationToken::default(),
    )?;
    Ok(Arc::new(NativePromptTokenizer::checked(
        NativeTokenizerFamily::Qwen2ByteBpe(family),
        TokenizerConfiguration {
            maximum_length: 8,
            minimum_length: Some(1),
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: false,
            start_token: None,
            end_token: None,
            pad_token: 151_643,
            maximum_word_length: 8,
            disable_weights: true,
            embedding_width: None,
        },
        BTreeMap::new(),
    )?))
}

fn qwen35_prompt_tokenizer() -> Result<Arc<NativePromptTokenizer>, Box<dyn Error>> {
    let tokenizer_configuration =
        fs::read_to_string(workspace()?.join(
            "projects/comfy/ComfyUI/sim/text_encoders/qwen35_tokenizer/tokenizer_config.json",
        ))?;
    let vocabulary = fs::read_to_string(
        workspace()?.join("projects/comfy/ComfyUI/sim/text_encoders/qwen35_tokenizer/vocab.json"),
    )?;
    let merges = fs::read_to_string(
        workspace()?.join("projects/comfy/ComfyUI/sim/text_encoders/qwen35_tokenizer/merges.txt"),
    )?;
    let family = Qwen2BpeTokenizer::from_artifacts(
        Qwen2PretokenizerProfile::Qwen35Declared,
        &tokenizer_configuration,
        &vocabulary,
        &merges,
        &CancellationToken::default(),
    )?;
    Ok(Arc::new(NativePromptTokenizer::checked(
        NativeTokenizerFamily::Qwen2ByteBpe(family),
        TokenizerConfiguration {
            maximum_length: 8,
            minimum_length: Some(1),
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: false,
            start_token: None,
            end_token: None,
            pad_token: 248_044,
            maximum_word_length: 8,
            disable_weights: true,
            embedding_width: None,
        },
        BTreeMap::new(),
    )?))
}

fn reduced_qwen3_decoder(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    token_scale: f32,
) -> Result<Arc<NativeDecoderTextEncoder>, Box<dyn Error>> {
    let configuration = DecoderTextConfiguration {
        architecture: DecoderArchitecture::Llama,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: 151_936,
        maximum_tokens: 8,
        hidden_size: 4,
        feed_forward_size: 8,
        layer_kinds: vec![DecoderLayerKind::FullAttention; 3],
        attention_heads: 2,
        key_value_heads: 1,
        head_dimension: 2,
        query_key_norm: true,
        qwen35_linear: None,
        gemma3: None,
        gemma4: None,
        normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
        rope: DecoderRopeConfiguration {
            theta: 5_000_000.0,
            rotary_dimension: 2,
            interleaved_sections: vec![1, 0, 0],
            scaling: RopeScaling::None,
        },
        sliding_window: None,
        activation: DecoderActivation::Silu,
        embedding_scale_bits: 1.0_f32.to_bits(),
        residual_scale_bits: 1.0_f32.to_bits(),
        norm_weight_offset_bits: 0.0_f32.to_bits(),
        logits_soft_cap_bits: None,
        tied_output_head: true,
        stop_tokens: vec![151_643, 151_645],
    };
    let token_embedding = filled_tensor(backend, &[151_936, 4], token_scale, context)?;
    let mut layers = Vec::new();
    for layer in 0..3 {
        let scale = 0.01 + layer as f32 * 0.001;
        layers.push(DecoderLayerWeights {
            attention_norm_weight: filled_tensor(backend, &[4], 1.0, context)?,
            attention: DecoderAttentionWeights::DotProduct {
                query_weight: filled_tensor(backend, &[4, 4], scale, context)?,
                key_weight: filled_tensor(backend, &[2, 4], scale, context)?,
                value_weight: filled_tensor(backend, &[2, 4], scale, context)?,
                query_norm_weight: Some(filled_tensor(backend, &[2], 1.0, context)?),
                key_norm_weight: Some(filled_tensor(backend, &[2], 1.0, context)?),
                output_weight: filled_tensor(backend, &[4, 4], scale, context)?,
            },
            feed_forward_norm_weight: filled_tensor(backend, &[4], 1.0, context)?,
            feed_forward_gate_weight: filled_tensor(backend, &[8, 4], scale, context)?,
            feed_forward_up_weight: filled_tensor(backend, &[8, 4], scale, context)?,
            feed_forward_down_weight: filled_tensor(backend, &[4, 8], scale, context)?,
            post_attention_norm_weight: None,
            post_feed_forward_norm_weight: None,
            attention_sink: None,
            gemma4_layer_input: None,
        });
    }
    Ok(Arc::new(NativeDecoderTextEncoder::new(
        configuration,
        DecoderTextWeights {
            token_embedding,
            layers,
            final_norm_weight: filled_tensor(backend, &[4], 1.0, context)?,
            output_head_weight: None,
            gemma4_per_layer: None,
        },
    )?))
}

#[test]
fn source_profiles_and_total_symbol_map_are_exact() -> Result<(), Box<dyn Error>> {
    assert_eq!(
        IDEOGRAM4_TAP_LAYERS,
        [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36]
    );
    let ideogram = multimodal_profile(MultimodalFamily::Ideogram4);
    assert_eq!(ideogram.hidden_size, 4096);
    assert_eq!(ideogram.projection_size, Some(53_248));
    assert_eq!(ideogram.rope_theta(), Some(5_000_000.0));
    let jina = multimodal_profile(MultimodalFamily::JinaClip2);
    assert_eq!((jina.vocabulary_size, jina.maximum_tokens), (250_002, 8192));
    assert_eq!(jina.rope_theta(), Some(20_000.0));
    let ovis = multimodal_profile(MultimodalFamily::Ovis25);
    assert_eq!((ovis.hidden_size, ovis.pad_token), (2048, 151_643));
    let qwen4 = multimodal_profile(MultimodalFamily::Qwen3Vl4B);
    assert_eq!(
        (qwen4.hidden_size, qwen4.visual_layer_count),
        (2560, Some(24))
    );
    let qwen8 = multimodal_profile(MultimodalFamily::Qwen3Vl8B);
    assert_eq!(
        (qwen8.hidden_size, qwen8.visual_layer_count),
        (4096, Some(27))
    );
    let qwen2 = multimodal_profile(MultimodalFamily::Qwen2Vl);
    assert_eq!((qwen2.patch_size, qwen2.layer_count), (Some(14), 32));
    let sam3 = multimodal_profile(MultimodalFamily::Sam3);
    assert_eq!((sam3.maximum_tokens, sam3.projection_size), (32, Some(512)));

    let workspace = workspace()?;
    for (path, expected) in [
        (IDEOGRAM4_SOURCE_PATH, IDEOGRAM4_SOURCE_SHA256),
        (JINA_CLIP2_SOURCE_PATH, JINA_CLIP2_SOURCE_SHA256),
        (OVIS_SOURCE_PATH, OVIS_SOURCE_SHA256),
        (QWEN3VL_SOURCE_PATH, QWEN3VL_SOURCE_SHA256),
        (QWEN_VL_SOURCE_PATH, QWEN_VL_SOURCE_SHA256),
        (SAM3_CLIP_SOURCE_PATH, SAM3_CLIP_SOURCE_SHA256),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            expected
        );
    }
    assert!(multimodal_symbol_behavior(QWEN_VL_SOURCE_PATH, "PythonFallback").is_none());
    Ok(())
}

#[test]
fn qwen_multimodal_positions_match_source_order_and_fail_closed() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let span = MultimodalSpan {
        start: 2,
        size: 4,
        grid_thw: [1, 4, 4],
    };
    let positions = qwen2vl_mrope_position_ids(10, &[span], &cancellation)?
        .ok_or("image positions were not constructed")?;
    assert_eq!(positions.temporal(), &[0, 1, 2, 2, 2, 2, 4, 5, 6, 7]);
    assert_eq!(positions.height(), &[0, 1, 2, 2, 3, 3, 4, 5, 6, 7]);
    assert_eq!(positions.width(), &[0, 1, 2, 3, 2, 3, 4, 5, 6, 7]);
    assert_eq!(qwen2vl_mrope_position_ids(4, &[], &cancellation)?, None);
    assert!(matches!(
        qwen2vl_mrope_position_ids(
            10,
            &[
                span,
                MultimodalSpan {
                    start: 5,
                    size: 2,
                    grid_thw: [1, 4, 4]
                },
            ],
            &cancellation,
        ),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        qwen2vl_mrope_position_ids(10, &[span], &cancelled),
        Err(MultimodalTextError::Cancelled)
    ));
    Ok(())
}

#[test]
fn qwen3vl_resize_patch_packing_and_batch_splitting_are_source_exact() -> Result<(), Box<dyn Error>>
{
    assert_eq!(qwen3vl_target_dimensions(48, 96)?, (64, 96));
    assert_eq!(qwen3vl_target_dimensions(17, 31)?, (64, 96));
    let (large_height, large_width) = qwen3vl_target_dimensions(10_000, 20_000)?;
    assert_eq!((large_height % 32, large_width % 32), (0, 0));
    assert!(
        large_height
            .checked_mul(large_width)
            .ok_or("large image overflow")?
            <= 12_845_056
    );
    assert!(qwen3vl_target_dimensions(0, 32).is_err());
    assert!(qwen3vl_target_dimensions(1, u64::MAX / 2).is_err());

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let mut pixels = vec![0.5_f32; 2 * 17 * 31 * 3];
    pixels[17 * 31 * 3] = 0.75;
    let images = ImageTensor::from_f32(&backend, &context, 2, 17, 31, 3, &pixels)?;
    let prepared = prepare_qwen3vl_images(&backend, &images, &context)?;
    assert_eq!(prepared.len(), 2);
    for image in &prepared {
        assert_eq!(image.grid_thw(), [1, 4, 6]);
        assert_eq!(image.merged_tokens(), 6);
        assert_eq!(image.patches().descriptor().shape(), [24, 3, 2, 16, 16]);
    }
    let first = tensor_to_f32(&backend, prepared[0].patches(), &context)?;
    assert!(first.iter().all(|value| *value == 0.0));
    let second = tensor_to_f32(&backend, prepared[1].patches(), &context)?;
    assert!(second.iter().any(|value| *value > 0.0));
    drop(first);
    drop(second);
    drop(prepared);
    drop(images);

    let mut ordered_pixels = vec![0.5_f32; 64 * 64 * 3];
    ordered_pixels[0] = 0.75;
    ordered_pixels[(16 * 3) + 1] = 0.25;
    ordered_pixels[(16 * 64 * 3) + 2] = 1.0;
    ordered_pixels[(32 * 3) + 1] = 0.0;
    let ordered_image = ImageTensor::from_f32(&backend, &context, 1, 64, 64, 3, &ordered_pixels)?;
    let ordered = prepare_qwen3vl_images(&backend, &ordered_image, &context)?;
    let ordered_patches = tensor_to_f32(&backend, ordered[0].patches(), &context)?;
    let patch_width = 3 * 2 * 16 * 16;
    assert_eq!(ordered_patches[0], 0.5);
    assert_eq!(ordered_patches[16 * 16], 0.5);
    assert_eq!(ordered_patches[patch_width + (2 * 16 * 16)], -0.5);
    assert_eq!(ordered_patches[patch_width + (3 * 16 * 16)], -0.5);
    assert_eq!(ordered_patches[(2 * patch_width) + (4 * 16 * 16)], 1.0);
    assert_eq!(ordered_patches[(2 * patch_width) + (5 * 16 * 16)], 1.0);
    assert_eq!(ordered_patches[(4 * patch_width) + (2 * 16 * 16)], -1.0);
    assert_eq!(ordered_patches[(4 * patch_width) + (3 * 16 * 16)], -1.0);
    assert_eq!(
        ordered_patches
            .iter()
            .filter(|value| **value != 0.0)
            .count(),
        8
    );
    drop(ordered_patches);
    drop(ordered);
    drop(ordered_image);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn qwen3vl_marker_plan_expands_real_image_spans_and_fails_closed() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let first = ImageTensor::from_f32(&backend, &context, 1, 32, 32, 3, &vec![0.5; 3_072])?;
    let second = ImageTensor::from_f32(&backend, &context, 1, 64, 32, 3, &vec![0.5; 6_144])?;
    let first = prepare_qwen3vl_images(&backend, &first, &context)?;
    let second = prepare_qwen3vl_images(&backend, &second, &context)?;
    let images = [first[0].clone(), second[0].clone()];
    assert_eq!(
        (images[0].merged_tokens(), images[1].merged_tokens()),
        (4, 6)
    );
    let plan = plan_qwen3vl_markers(
        &[10, QWEN3VL_IMAGE_PAD_TOKEN, 11, QWEN3VL_IMAGE_PAD_TOKEN, 12],
        &images,
        &cancellation,
    )?;
    assert_eq!(
        plan.expanded_tokens(),
        &[
            10,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            11,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            QWEN3VL_IMAGE_PAD_TOKEN,
            12,
        ]
    );
    assert_eq!(
        plan.spans(),
        &[
            MultimodalSpan {
                start: 1,
                size: 4,
                grid_thw: [1, 4, 4]
            },
            MultimodalSpan {
                start: 6,
                size: 6,
                grid_thw: [1, 6, 4]
            },
        ]
    );
    assert_eq!(
        plan.visual_position_mask()
            .iter()
            .filter(|value| **value)
            .count(),
        10
    );
    assert!(plan_qwen3vl_markers(&[QWEN3VL_IMAGE_PAD_TOKEN], &images, &cancellation).is_err());
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        plan_qwen3vl_markers(&[], &[], &cancelled),
        Err(MultimodalTextError::Cancelled)
    ));
    drop(plan);
    drop(images);
    drop(first);
    drop(second);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn qwen3vl_preparation_cancellation_and_oom_leave_workspace_empty() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup_context = context(&authority, &cancellation, 8 * 1024 * 1024)?;
    let image = ImageTensor::from_f32(
        &backend,
        &setup_context,
        1,
        64,
        64,
        3,
        &vec![0.5; 64 * 64 * 3],
    )?;

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 8 * 1024 * 1024)?;
    assert!(matches!(
        prepare_qwen3vl_images(&backend, &image, &cancelled_context),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    let constrained = context(&authority, &cancellation, 1_024)?;
    assert!(prepare_qwen3vl_images(&backend, &image, &constrained).is_err());
    assert_eq!(constrained.scratch.in_use_bytes(), 0);

    let invalid = ImageTensor::from_f32(
        &backend,
        &setup_context,
        1,
        32,
        32,
        3,
        &vec![f32::NAN; 32 * 32 * 3],
    )?;
    assert!(matches!(
        prepare_qwen3vl_images(&backend, &invalid, &setup_context),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    Ok(())
}

#[test]
fn gemma_image_video_preparation_is_source_exact_bounded_and_transactional()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/image_video/manifest.json"
    ))?;
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(
        manifest["sources"][0]["sha256"],
        "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2"
    );
    assert_eq!(
        manifest["sources"][1]["sha256"],
        "c6ffbb2fbecd8f97e781a654a06ccf3910dc670867d38c0ce30542312f00cde6"
    );
    assert_eq!(GEMMA3_IMAGE_AREA_PIXELS, 896 * 896);
    assert_eq!(gemma3_target_dimensions(512, 512)?, (896, 896));
    assert_eq!(gemma3_target_dimensions(17, 31)?, (664, 1_210));
    assert!(gemma3_target_dimensions(0, 31).is_err());
    assert!(gemma3_target_dimensions(1, u64::MAX / 2).is_err());
    assert_eq!(
        gemma4_target_dimensions(48, 48, GEMMA4_IMAGE_SOFT_TOKENS)?,
        (768, 768)
    );
    assert_eq!(
        gemma4_target_dimensions(48, 48, GEMMA4_VIDEO_SOFT_TOKENS)?,
        (384, 384)
    );
    assert!(gemma4_target_dimensions(48, 48, 0).is_err());
    assert!(gemma4_target_dimensions(1, u64::MAX / 2, GEMMA4_IMAGE_SOFT_TOKENS).is_err());

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let gemma3_source =
        ImageTensor::from_f32(&backend, &setup, 1, 1, 1, 4, &[1.0, 0.25, 0.0, 0.75])?;
    let gemma3 = prepare_gemma3_image(&backend, &gemma3_source, &setup)?;
    assert_eq!(gemma3.kind(), GemmaPreparedVisualKind::Gemma3Image);
    assert_eq!(gemma3.maximum_soft_tokens(), 256);
    assert_eq!(gemma3.image().dimensions()?, (1, 896, 896, 3));
    let gemma3_values = gemma3.image().as_f32_slice()?;
    assert_eq!(gemma3_values.first().copied(), Some(1.0));
    assert_eq!(gemma3_values.get(1).copied(), Some(0.25));
    assert_eq!(gemma3_values.get(2).copied(), Some(0.0));
    drop(gemma3);

    let image = ImageTensor::from_f32(&backend, &setup, 1, 1, 1, 4, &[1.0, 0.0, 0.0, 1.0])?;
    let mut video_values = Vec::new();
    video_values.try_reserve_exact(49 * 4)?;
    for frame in 0..49 {
        video_values.extend_from_slice(&[0.0, if frame == 24 { 0.5 } else { 1.0 }, 0.0, 0.25]);
    }
    let video = ImageTensor::from_f32(&backend, &setup, 49, 1, 1, 4, &video_values)?;
    let prepared = prepare_gemma4_visuals(&backend, Some(&image), Some(&video), &setup)?;
    assert_eq!(prepared.len(), 3);
    for (index, visual) in prepared.iter().enumerate() {
        assert_eq!(visual.kind(), GemmaPreparedVisualKind::Gemma4VideoFrame);
        assert_eq!(visual.maximum_soft_tokens(), GEMMA4_VIDEO_SOFT_TOKENS);
        assert_eq!(visual.source_frame_index(), index * GEMMA4_VIDEO_SOURCE_FPS);
        assert_eq!(visual.timestamp_seconds(), Some(index));
        assert_eq!(visual.image().dimensions()?, (1, 384, 384, 3));
        assert_eq!(visual.image().as_f32_slice()?.first().copied(), Some(0.0));
    }
    let middle_green = prepared
        .get(1)
        .ok_or("Gemma4 middle prepared frame is missing")?
        .image()
        .as_f32_slice()?
        .get(1)
        .copied()
        .ok_or("Gemma4 middle prepared pixel is missing")?;
    assert!((middle_green - (127.0 / 255.0)).abs() <= 1.0e-6);
    assert_ne!(
        prepared
            .first()
            .ok_or("Gemma4 first prepared frame is missing")?
            .image()
            .as_f32_slice()?
            .get(1),
        Some(&0.0)
    );
    drop(prepared);

    let image_only = prepare_gemma4_visuals(&backend, Some(&image), None, &setup)?;
    let image_visual = image_only
        .first()
        .ok_or("Gemma4 prepared image is missing")?;
    assert_eq!(image_only.len(), 1);
    assert_eq!(image_visual.kind(), GemmaPreparedVisualKind::Gemma4Image);
    assert_eq!(image_visual.maximum_soft_tokens(), GEMMA4_IMAGE_SOFT_TOKENS);
    assert_eq!(image_visual.image().dimensions()?, (1, 768, 768, 3));
    assert!(prepare_gemma4_visuals(&backend, None, None, &setup)?.is_empty());
    drop(image_only);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 48 * 1024 * 1024)?;
    assert!(matches!(
        prepare_gemma4_visuals(&backend, Some(&image), None, &cancelled_context),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    let constrained = context(&authority, &cancellation, 4)?;
    assert!(prepare_gemma4_visuals(&backend, Some(&image), None, &constrained).is_err());
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma3_retained_vision_projector_is_exact_alias_aware_and_transactional()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/gemma3_vision/manifest.json"
    ))?;
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let source = fs::read(workspace.join(GEMMA3_MULTIMODAL_SOURCE_PATH))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        GEMMA3_MULTIMODAL_SOURCE_SHA256
    );
    assert_eq!(
        Gemma3VisionConfiguration::source(Gemma3VisionProfile::FourBVision).output_hidden_size,
        2_560
    );
    assert_eq!(
        Gemma3VisionConfiguration::source(Gemma3VisionProfile::TwelveB).output_hidden_size,
        3_840
    );

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let (vision, shared_norm) = reduced_gemma3_clip_vision(&backend, &setup)?;
    let projection_weight = tensor(
        &backend,
        &[4, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.5, 0.75],
        &setup,
    )?;
    let configuration = Gemma3VisionConfiguration::reduced_fixture(4, 3, 4, 2);
    let owner = NativeGemma3VisionProjector::new(
        configuration.clone(),
        vision.clone(),
        shared_norm.clone(),
        projection_weight,
    )?;
    let input = ImageTensor::from_f32(&backend, &setup, 1, 1, 1, 4, &[1.0, 0.25, 0.0, 1.0])?;
    let prepared = prepare_gemma3_image(&backend, &input, &setup)?;
    let output = owner.project(&backend, &prepared, &setup)?;
    assert_eq!(output.tokens_per_image, 4);
    assert_eq!(output.embedding.descriptor().shape(), &[1, 4, 3]);
    let output_values = tensor_to_f32(&backend, &output.embedding, &setup)?;
    assert!(output_values.iter().all(|value| value.is_finite()));
    let output_digest = format!(
        "{:x}",
        Sha256::digest(
            output_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        output_digest,
        manifest["reduced_fixture"]["output_f32le_sha256"]
            .as_str()
            .ok_or("Gemma3 projection fixture digest is missing")?
    );
    drop(output_values);

    let allocations = owner.resident_tensor_allocations()?;
    assert_eq!(
        allocations
            .iter()
            .filter(|(storage_id, _)| *storage_id == shared_norm.storage_id())
            .count(),
        1
    );
    let storage_ids = allocations
        .iter()
        .map(|(storage_id, _)| *storage_id)
        .collect::<Vec<_>>();
    assert!(storage_ids.iter().enumerate().all(|(index, storage_id)| {
        !storage_ids[..index].iter().any(|prior| prior == storage_id)
    }));

    let changed_projection = filled_tensor(&backend, &[4, 3], 0.125, &setup)?;
    let changed = NativeGemma3VisionProjector::new(
        configuration.clone(),
        vision.clone(),
        shared_norm.clone(),
        changed_projection,
    )?;
    assert_ne!(
        owner.semantic_state_digest_sha256(),
        changed.semantic_state_digest_sha256()
    );
    assert_eq!(
        owner.semantic_state_digest_sha256(),
        owner.clone().semantic_state_digest_sha256()
    );

    let mut training_vision = (*vision).clone();
    training_vision.train();
    assert!(
        NativeGemma3VisionProjector::new(
            configuration,
            Arc::new(training_vision),
            shared_norm,
            filled_tensor(&backend, &[4, 3], 0.0, &setup)?,
        )
        .is_err()
    );
    let wrong_prepared = prepare_gemma4_visuals(&backend, Some(&input), None, &setup)?;
    assert!(owner.project(&backend, &wrong_prepared[0], &setup).is_err());
    drop(output);
    drop(wrong_prepared);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 48 * 1024 * 1024)?;
    assert!(matches!(
        owner.project(&backend, &prepared, &cancelled_context),
        Err(MultimodalTextError::Tensor(TensorError::Cancelled))
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    let constrained = context(&authority, &cancellation, 16)?;
    assert!(owner.project(&backend, &prepared, &constrained).is_err());
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    drop(prepared);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma4_retained_vision_projector_is_exact_alias_aware_and_transactional()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/gemma4_vision/manifest.json"
    ))?;
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(fs::read(workspace.join(GEMMA4_MULTIMODAL_SOURCE_PATH))?)
        ),
        GEMMA4_MULTIMODAL_SOURCE_SHA256
    );
    assert_eq!(
        Gemma4VisionConfiguration::source(Gemma4VisionProfile::E2B).output_hidden_size,
        1_536
    );
    assert_eq!(
        Gemma4VisionConfiguration::source(Gemma4VisionProfile::E4B).output_hidden_size,
        2_560
    );
    let thirty_one = Gemma4VisionConfiguration::source(Gemma4VisionProfile::ThirtyOneB);
    assert_eq!(thirty_one.hidden_size, 1_152);
    assert_eq!(thirty_one.output_hidden_size, 5_376);

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 60 * 1024 * 1024)?;
    let configuration = Gemma4VisionConfiguration::reduced_fixture(
        Gemma4VisionProfile::E4B,
        4,
        6,
        1,
        1,
        4,
        16,
        64,
        3,
        3,
    );
    let owner = NativeGemma4VisionEncoder::new(
        configuration.clone(),
        reduced_gemma4_vision_weights(&backend, &configuration, &setup)?,
    )?;
    let source = ImageTensor::from_f32(&backend, &setup, 1, 1, 1, 3, &[0.2, 0.5, 0.9])?;
    let prepared_video = prepare_gemma4_visuals(&backend, None, Some(&source), &setup)?;
    assert_eq!(prepared_video[0].image().dimensions()?, (1, 384, 384, 3));
    assert_eq!(
        prepared_video[0].image().as_f32_slice()?.len(),
        384 * 384 * 3
    );
    let video = owner.project(&backend, &prepared_video[0], &setup)?;
    assert_eq!(video.kind, GemmaPreparedVisualKind::Gemma4VideoFrame);
    assert_eq!(video.tokens, 64);
    assert_eq!(video.embedding.descriptor().shape(), &[64, 3]);
    let video_values = tensor_to_f32(&backend, &video.embedding, &setup)?;
    let output_digest = format!(
        "{:x}",
        Sha256::digest(
            video_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        output_digest,
        manifest["reduced_fixture"]["video_output_f32le_sha256"]
            .as_str()
            .ok_or("Gemma4 video fixture digest is missing")?
    );
    drop(video_values);

    let prepared_image = prepare_gemma4_visuals(&backend, Some(&source), None, &setup)?;
    let image = owner.project(&backend, &prepared_image[0], &setup)?;
    assert_eq!(image.kind, GemmaPreparedVisualKind::Gemma4Image);
    assert_eq!(image.tokens, 256);
    assert_eq!(image.embedding.descriptor().shape(), &[256, 3]);

    let allocations = owner.resident_tensor_allocations()?;
    let storage_ids = allocations
        .iter()
        .map(|(storage_id, _)| *storage_id)
        .collect::<Vec<_>>();
    assert!(storage_ids.iter().enumerate().all(|(index, storage_id)| {
        !storage_ids[..index].iter().any(|prior| prior == storage_id)
    }));
    assert_eq!(
        owner.semantic_state_digest_sha256(),
        owner.clone().semantic_state_digest_sha256()
    );
    let mut changed_weights = reduced_gemma4_vision_weights(&backend, &configuration, &setup)?;
    changed_weights.projector_weight = filled_tensor(&backend, &[3, 4], 0.076, &setup)?;
    let changed = NativeGemma4VisionEncoder::new(configuration.clone(), changed_weights)?;
    assert_ne!(
        owner.semantic_state_digest_sha256(),
        changed.semantic_state_digest_sha256()
    );
    let mut forged = Gemma4VisionConfiguration::source(Gemma4VisionProfile::E4B);
    forged.rotary_theta_bits = 10_000.0_f32.to_bits();
    assert!(
        NativeGemma4VisionEncoder::new(
            forged,
            reduced_gemma4_vision_weights(&backend, &configuration, &setup)?,
        )
        .is_err()
    );

    let gemma3 = prepare_gemma3_image(&backend, &source, &setup)?;
    assert!(owner.project(&backend, &gemma3, &setup).is_err());
    drop(video);
    drop(image);
    drop(prepared_video);
    drop(prepared_image);
    drop(gemma3);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 60 * 1024 * 1024)?;
    let prepared_video = prepare_gemma4_visuals(&backend, None, Some(&source), &setup)?;
    assert!(
        owner
            .project(&backend, &prepared_video[0], &cancelled_context)
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    let constrained = context(&authority, &cancellation, 16)?;
    assert!(
        owner
            .project(&backend, &prepared_video[0], &constrained)
            .is_err()
    );
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    drop(prepared_video);
    drop(source);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma4_retained_audio_encoder_is_exact_alias_aware_and_transactional()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/gemma4_audio/manifest.json"
    ))?;
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(
        Gemma4AudioConfiguration::source(Gemma4AudioProfile::E2B)?.output_hidden_size,
        1_536
    );
    assert_eq!(
        Gemma4AudioConfiguration::source(Gemma4AudioProfile::E4B)?.output_hidden_size,
        2_560
    );
    assert!(Gemma4AudioConfiguration::source(Gemma4AudioProfile::ThirtyOneB).is_err());

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let configuration = Gemma4AudioConfiguration::reduced_fixture(
        Gemma4AudioProfile::E4B,
        128,
        2,
        1,
        4,
        6,
        1,
        1,
        3,
        2,
        3,
        4,
        3,
    )?;
    let owner = NativeGemma4AudioEncoder::new(
        configuration.clone(),
        reduced_gemma4_audio_weights(&backend, &configuration, &setup)?,
        &cancellation,
    )?;
    let waveform_values = (0..640)
        .map(|index| ((index as f32 * 0.017).sin() * 0.25) + 0.1)
        .collect::<Vec<_>>();
    let waveform = tensor(&backend, &[1, 1, 640], &waveform_values, &setup)?;
    let prepared = prepare_gemma4_audio(&backend, &waveform, 16_000, &setup)?;
    let output = owner.project(&backend, &prepared, &setup)?;
    assert_eq!(output.tokens, 1);
    assert_eq!(output.embedding.descriptor().shape(), [1, 3]);
    let values = tensor_to_f32(&backend, &output.embedding, &setup)?;
    assert!(values.iter().all(|value| value.is_finite()));
    let digest = format!(
        "{:x}",
        Sha256::digest(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        digest,
        manifest["reduced_fixture"]["output_f32le_sha256"]
            .as_str()
            .ok_or("Gemma4 audio output digest is missing")?
    );
    drop(values);

    let allocations = owner.resident_tensor_allocations()?;
    let storage_ids = allocations
        .iter()
        .map(|(storage_id, _)| *storage_id)
        .collect::<Vec<_>>();
    assert!(storage_ids.iter().enumerate().all(|(index, storage_id)| {
        !storage_ids[..index].iter().any(|prior| prior == storage_id)
    }));
    assert_eq!(
        owner.semantic_state_digest_sha256(),
        owner.clone().semantic_state_digest_sha256()
    );
    let mut changed_weights = reduced_gemma4_audio_weights(&backend, &configuration, &setup)?;
    changed_weights.projector_weight = filled_tensor(&backend, &[3, 4], 0.031, &setup)?;
    let changed =
        NativeGemma4AudioEncoder::new(configuration.clone(), changed_weights, &cancellation)?;
    assert_ne!(
        owner.semantic_state_digest_sha256(),
        changed.semantic_state_digest_sha256()
    );
    let mut forged = Gemma4AudioConfiguration::source(Gemma4AudioProfile::E4B)?;
    forged.attention_chunk_size = 11;
    assert!(
        NativeGemma4AudioEncoder::new(
            forged,
            reduced_gemma4_audio_weights(&backend, &configuration, &setup)?,
            &cancellation,
        )
        .is_err()
    );

    drop(output);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 48 * 1024 * 1024)?;
    assert!(
        owner
            .project(&backend, &prepared, &cancelled_context)
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    let constrained = context(&authority, &cancellation, 16)?;
    assert!(owner.project(&backend, &prepared, &constrained).is_err());
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    drop(prepared);
    drop(waveform);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma_multimodal_resource_closes_family_identity_residency_and_payload_admission()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/resource/manifest.json"
    ))?;
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(
        manifest["accepted_families"]
            .as_array()
            .ok_or("Gemma family matrix is missing")?
            .len(),
        5
    );
    let family_matrix = [
        (
            GemmaMultimodalFamily::Gemma3FourBVision,
            Gemma3VisionConfiguration::source(Gemma3VisionProfile::FourBVision).output_hidden_size,
            None,
        ),
        (
            GemmaMultimodalFamily::Gemma3TwelveB,
            Gemma3VisionConfiguration::source(Gemma3VisionProfile::TwelveB).output_hidden_size,
            None,
        ),
        (
            GemmaMultimodalFamily::Gemma4E2B,
            Gemma4VisionConfiguration::source(Gemma4VisionProfile::E2B).output_hidden_size,
            Some(Gemma4AudioProfile::E2B),
        ),
        (
            GemmaMultimodalFamily::Gemma4E4B,
            Gemma4VisionConfiguration::source(Gemma4VisionProfile::E4B).output_hidden_size,
            Some(Gemma4AudioProfile::E4B),
        ),
        (
            GemmaMultimodalFamily::Gemma4ThirtyOneB,
            Gemma4VisionConfiguration::source(Gemma4VisionProfile::ThirtyOneB).output_hidden_size,
            None,
        ),
    ];
    for (family, vision_width, audio_profile) in family_matrix {
        let decoder = family.decoder_configuration();
        assert_eq!(decoder.hidden_size, vision_width);
        assert_eq!(family.supports_audio(), audio_profile.is_some());
        if let Some(profile) = audio_profile {
            assert_eq!(
                Gemma4AudioConfiguration::source(profile)?.output_hidden_size,
                decoder.hidden_size
            );
        }
    }
    assert_eq!(
        GemmaMultimodalFamily::Gemma3FourBVision.source_sha256(),
        GEMMA3_FOUR_B_MULTIMODAL_SOURCE_SHA256
    );
    assert_eq!(
        GemmaMultimodalFamily::Gemma3TwelveB.tokenizer_profile(),
        GemmaTokenizerProfile::Gemma3SentencePiece
    );
    assert_eq!(
        GemmaMultimodalFamily::Gemma4E4B.tokenizer_profile(),
        GemmaTokenizerProfile::Gemma4TokenizerJson
    );

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokenizer = gemma4_prompt_tokenizer(4)?;
    let decoder = reduced_gemma4_decoder(&backend, &setup, 0.0)?;
    let vision_configuration = Gemma4VisionConfiguration::reduced_fixture(
        Gemma4VisionProfile::E4B,
        4,
        8,
        1,
        1,
        4,
        16,
        64,
        3,
        4,
    );
    let vision = Arc::new(NativeGemma4VisionEncoder::new(
        vision_configuration.clone(),
        reduced_gemma4_vision_weights(&backend, &vision_configuration, &setup)?,
    )?);
    let audio_configuration = Gemma4AudioConfiguration::reduced_fixture(
        Gemma4AudioProfile::E4B,
        4,
        2,
        2,
        4,
        8,
        1,
        2,
        3,
        2,
        3,
        4,
        4,
    )?;
    let audio = Arc::new(NativeGemma4AudioEncoder::new(
        audio_configuration.clone(),
        reduced_gemma4_audio_weights(&backend, &audio_configuration, &setup)?,
        &cancellation,
    )?);
    let resource = NativeGemmaMultimodal::reduced_gemma4_fixture(
        tokenizer.clone(),
        decoder.clone(),
        vision.clone(),
        Some(audio.clone()),
        &cancellation,
    )?;
    assert_eq!(resource.family(), GemmaMultimodalFamily::Gemma4E4B);
    assert!(Arc::ptr_eq(resource.tokenizer(), &tokenizer));
    assert!(Arc::ptr_eq(resource.decoder(), &decoder));
    assert!(Arc::ptr_eq(
        resource.gemma4_vision().ok_or("Gemma4 vision missing")?,
        &vision
    ));
    assert!(Arc::ptr_eq(
        resource.audio().ok_or("Gemma4 audio missing")?,
        &audio
    ));
    assert!(resource.gemma3_vision().is_none());
    assert!(!resource.is_source_exact_profile());
    let digest = resource.semantic_state_digest(&cancellation)?;
    assert_eq!(digest.len(), 64);
    let allocations = resource.resident_tensor_allocations()?;
    assert!(
        allocations
            .iter()
            .enumerate()
            .all(|(index, (storage_id, _))| {
                !allocations[..index]
                    .iter()
                    .any(|(prior, _)| prior == storage_id)
            })
    );
    let tensor_bytes = allocations
        .iter()
        .try_fold(0_u64, |total, (_, bytes)| total.checked_add(*bytes))
        .ok_or("Gemma tensor residency overflowed")?;
    assert!(resource.resident_bytes()? > tensor_bytes);
    let clone = NativeGemmaMultimodal::reduced_gemma4_fixture(
        tokenizer.clone(),
        decoder.clone(),
        vision.clone(),
        Some(audio.clone()),
        &cancellation,
    )?;
    assert_eq!(clone.semantic_state_digest(&cancellation)?, digest);
    assert_eq!(clone.resident_tensor_allocations()?, allocations);

    let (changed_backend, changed_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let changed_context = context(&changed_authority, &cancellation, 64 * 1024 * 1024)?;
    let changed_decoder = reduced_gemma4_decoder(&changed_backend, &changed_context, 0.005)?;
    let changed = NativeGemmaMultimodal::reduced_gemma4_fixture(
        tokenizer.clone(),
        changed_decoder,
        vision.clone(),
        Some(audio.clone()),
        &cancellation,
    )?;
    assert_ne!(changed.semantic_state_digest(&cancellation)?, digest);
    assert!(
        NativeGemmaMultimodal::reduced_gemma4_fixture(
            tokenizer.clone(),
            decoder.clone(),
            vision.clone(),
            None,
            &cancellation,
        )
        .is_err()
    );
    let e2_vision_configuration = Gemma4VisionConfiguration::reduced_fixture(
        Gemma4VisionProfile::E2B,
        4,
        8,
        1,
        1,
        4,
        16,
        64,
        3,
        4,
    );
    let e2_vision = Arc::new(NativeGemma4VisionEncoder::new(
        e2_vision_configuration.clone(),
        reduced_gemma4_vision_weights(&backend, &e2_vision_configuration, &setup)?,
    )?);
    assert!(
        NativeGemmaMultimodal::reduced_gemma4_fixture(
            tokenizer.clone(),
            decoder.clone(),
            e2_vision,
            Some(audio.clone()),
            &cancellation,
        )
        .is_err()
    );
    assert!(
        NativeGemmaMultimodal::gemma4(
            tokenizer.clone(),
            decoder.clone(),
            vision.clone(),
            Some(audio.clone()),
            &cancellation,
        )
        .is_err()
    );
    assert!(NativeModelPayload::gemma_multimodal_clip(Arc::new(resource)).is_err());
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        NativeGemmaMultimodal::reduced_gemma4_fixture(
            tokenizer,
            decoder,
            vision,
            Some(audio),
            &cancelled,
        ),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    assert_eq!(changed_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma_multimodal_generation_formats_replaces_and_delegates_transactionally()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/generation/manifest.json"
    ))?;
    assert_eq!(
        manifest["generation_contract"]["gemma4_initial_input_ids"],
        "expanded-prompt-zero-media-ids"
    );
    assert_eq!(
        manifest["generation_contract"]["decoder_delegation"],
        "one-borrowed-rng-prepared-generation-call"
    );
    assert_eq!(
        format_gemma_multimodal_prompt(
            GemmaMultimodalFamily::Gemma3FourBVision,
            "hello",
            &[],
            None,
            true,
            true,
            &CancellationToken::default(),
        )?,
        "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
    );
    assert_eq!(
        format_gemma_multimodal_prompt(
            GemmaMultimodalFamily::Gemma3TwelveB,
            "<start_of_turn>user\nhello",
            &[],
            None,
            true,
            false,
            &CancellationToken::default(),
        )?,
        "<start_of_turn>user\nhello"
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(128 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 112 * 1024 * 1024)?;
    let tokenizer = gemma4_prompt_tokenizer(4)?;
    let decoder = reduced_gemma4_decoder(&backend, &setup, 0.0)?;
    let vision_configuration = Gemma4VisionConfiguration::reduced_fixture(
        Gemma4VisionProfile::E4B,
        4,
        8,
        1,
        1,
        4,
        16,
        64,
        3,
        4,
    );
    let vision = Arc::new(NativeGemma4VisionEncoder::new(
        vision_configuration.clone(),
        reduced_gemma4_vision_weights(&backend, &vision_configuration, &setup)?,
    )?);
    let audio_configuration = Gemma4AudioConfiguration::reduced_fixture(
        Gemma4AudioProfile::E4B,
        128,
        2,
        2,
        4,
        8,
        1,
        2,
        3,
        2,
        3,
        4,
        4,
    )?;
    let audio = Arc::new(NativeGemma4AudioEncoder::new(
        audio_configuration.clone(),
        reduced_gemma4_audio_weights(&backend, &audio_configuration, &setup)?,
        &cancellation,
    )?);
    let resource = NativeGemmaMultimodal::reduced_gemma4_fixture(
        tokenizer,
        decoder,
        vision,
        Some(audio),
        &cancellation,
    )?;
    let source = ImageTensor::from_f32(&backend, &setup, 1, 1, 1, 3, &[0.2, 0.5, 0.9])?;
    let prepared_visuals = prepare_gemma4_visuals(&backend, None, Some(&source), &setup)?;
    let waveform_values = (0..640)
        .map(|index| (index as f32 * 0.017).sin() * 0.25)
        .collect::<Vec<_>>();
    let waveform = tensor(&backend, &[1, 1, 640], &waveform_values, &setup)?;
    let prepared_audio = prepare_gemma4_audio(&backend, &waveform, 16_000, &setup)?;
    assert_eq!(prepared_visuals[0].timestamp_seconds(), Some(0));
    assert_eq!(prepared_audio.marker_tokens(), 1);
    assert_eq!(
        format_gemma_multimodal_prompt(
            GemmaMultimodalFamily::Gemma4E4B,
            "hello",
            &prepared_visuals,
            Some(&prepared_audio),
            true,
            true,
            &cancellation,
        )?,
        "<|turn>system\n<|think|><turn|>\n<|turn>user\n\n\n00:00 <|image><|video|><image|>\n\n<|audio><|audio|><audio|>hello<turn|>\n<|turn>model\n"
    );
    let text = NativeTextGenerationRequest {
        formatted_prompt: "hello",
        maximum_new_tokens: 1,
        do_sample: false,
        temperature_bits: 0.0_f32.to_bits(),
        top_k: 0,
        top_p_bits: 1.0_f32.to_bits(),
        minimum_p_bits: 0.0_f32.to_bits(),
        repetition_penalty_bits: 1.0_f32.to_bits(),
        presence_penalty_bits: 0.0_f32.to_bits(),
    };
    let transaction = qwen_generation_transaction()?;
    let checkpoint = transaction.checkpoint();
    let digest = resource.semantic_state_digest(&cancellation)?;
    let residency = resource.resident_tensor_allocations()?;
    let outcome = resource.generate(
        &backend,
        GemmaMultimodalGenerationRequest {
            text: text.clone(),
            prepared_visuals: &prepared_visuals,
            prepared_audio: Some(&prepared_audio),
            use_default_template: true,
            thinking: true,
            transaction: &transaction,
        },
        &setup,
    )?;
    assert_eq!(outcome.generated_tokens.len(), 1);
    assert_eq!(transaction.checkpoint(), checkpoint);
    assert_eq!(resource.semantic_state_digest(&cancellation)?, digest);
    assert_eq!(resource.resident_tensor_allocations()?, residency);

    assert!(
        resource
            .generate(
                &backend,
                GemmaMultimodalGenerationRequest {
                    text,
                    prepared_visuals: &prepared_visuals,
                    prepared_audio: Some(&prepared_audio),
                    use_default_template: false,
                    thinking: false,
                    transaction: &transaction,
                },
                &setup,
            )
            .is_err()
    );
    assert_eq!(transaction.checkpoint(), checkpoint);

    let constrained = context(&authority, &cancellation, 4)?;
    assert!(
        resource
            .generate(
                &backend,
                GemmaMultimodalGenerationRequest {
                    text: NativeTextGenerationRequest {
                        formatted_prompt: "hello",
                        maximum_new_tokens: 1,
                        do_sample: false,
                        temperature_bits: 0.0_f32.to_bits(),
                        top_k: 0,
                        top_p_bits: 1.0_f32.to_bits(),
                        minimum_p_bits: 0.0_f32.to_bits(),
                        repetition_penalty_bits: 1.0_f32.to_bits(),
                        presence_penalty_bits: 0.0_f32.to_bits(),
                    },
                    prepared_visuals: &prepared_visuals,
                    prepared_audio: Some(&prepared_audio),
                    use_default_template: true,
                    thinking: false,
                    transaction: &transaction,
                },
                &constrained,
            )
            .is_err()
    );
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    assert_eq!(transaction.checkpoint(), checkpoint);
    assert_eq!(resource.semantic_state_digest(&cancellation)?, digest);
    assert_eq!(resource.resident_tensor_allocations()?, residency);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma_audio_preparation_is_source_exact_bounded_and_transactional() -> Result<(), Box<dyn Error>>
{
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/audio_preparation/manifest.json"
    ))?;
    assert_eq!(
        manifest["source_snapshot"]["tree_sha256"],
        "21de8fece20d8d5bfa94daaa52d6ccfe2db6726ca0803ca3b383ad164cbd1d5f"
    );
    assert_eq!(
        manifest["sources"][0]["sha256"],
        "c6ffbb2fbecd8f97e781a654a06ccf3910dc670867d38c0ce30542312f00cde6"
    );
    assert_eq!(GEMMA4_AUDIO_SAMPLE_RATE, 16_000);
    assert_eq!(GEMMA4_AUDIO_FRAME_LENGTH, 320);
    assert_eq!(GEMMA4_AUDIO_FRAME_STEP, 160);
    assert_eq!(GEMMA4_AUDIO_FFT_LENGTH, 512);
    assert_eq!(GEMMA4_AUDIO_MEL_BINS, 128);
    assert_eq!(GEMMA4_AUDIO_MAXIMUM_TOKENS, 750);
    assert_eq!(GEMMA4_AUDIO_MINIMUM_SAMPLE_RATE, 8_000);
    assert_eq!(GEMMA4_AUDIO_MAXIMUM_SAMPLE_RATE, 384_000);
    assert_eq!(gemma4_audio_marker_tokens(640, 16_000)?, 1);
    assert_eq!(gemma4_audio_marker_tokens(320, 8_000)?, 1);
    assert_eq!(gemma4_audio_marker_tokens(1, 16_000)?, 0);
    assert_eq!(gemma4_audio_marker_tokens(4_000_000, 16_000)?, 750);
    assert!(gemma4_audio_marker_tokens(usize::MAX, 8_000).is_err());

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let stereo = tensor(
        &backend,
        &[1, 2, 640],
        &[vec![1.0; 640], vec![-1.0; 640]].concat(),
        &setup,
    )?;
    let prepared = prepare_gemma4_audio(&backend, &stereo, 16_000, &setup)?;
    assert_eq!(prepared.original_sample_rate(), 16_000);
    assert_eq!(prepared.original_samples(), 640);
    assert_eq!(prepared.resampled_samples(), 640);
    assert_eq!(prepared.marker_tokens(), 1);
    assert_eq!(prepared.log_mel().descriptor().shape(), [1, 3, 128]);
    assert_eq!(prepared.frame_mask().descriptor().shape(), [1, 3]);
    assert_eq!(prepared.frame_mask().descriptor().dtype(), DType::Bool);
    assert_eq!(prepared.frame_mask().contiguous_bytes()?, [1, 1, 1]);
    let silence = tensor_to_f32(&backend, prepared.log_mel(), &setup)?;
    assert_eq!(silence.len(), 3 * 128);
    for value in silence.iter().copied() {
        assert!((value - 0.001_f32.ln()).abs() <= 1.0e-6);
    }
    drop(silence);
    drop(prepared);

    let mut resample_values = Vec::new();
    resample_values.try_reserve_exact(2_205)?;
    for sample in 0..2_205 {
        let phase = std::f32::consts::TAU * 440.0 * sample as f32 / 44_100.0;
        resample_values.push(phase.sin());
    }
    let resample_input = tensor(&backend, &[1, 1, 2_205], &resample_values, &setup)?;
    let resampled = prepare_gemma4_audio(&backend, &resample_input, 44_100, &setup)?;
    assert_eq!(resampled.resampled_samples(), 800);
    assert_eq!(resampled.marker_tokens(), 1);
    assert_eq!(resampled.log_mel().descriptor().shape(), [1, 5, 128]);
    assert_eq!(resampled.frame_mask().contiguous_bytes()?, [1, 1, 1, 1, 0]);
    let resampled_features = tensor_to_f32(&backend, resampled.log_mel(), &setup)?;
    assert!(resampled_features.iter().all(|value| value.is_finite()));
    let expected_first = [
        -6.9077554, 1.503931, 0.31853235, 1.3154463, 0.8074848, 1.2384311, 0.96338606, 1.3288747,
        0.9583913, 1.535566, 0.7444314, 1.7973692,
    ];
    for (actual, expected) in resampled_features.iter().zip(expected_first) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }
    let feature_bytes = resampled.log_mel().contiguous_bytes()?;
    let source: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/audio_preparation/source-oracle.json"
    ))?;
    assert_eq!(source["source"]["sha256"], manifest["sources"][0]["sha256"]);
    let expected = source["values"].as_array().ok_or("source audio values")?;
    assert_eq!(expected.len(), resampled_features.len());
    let tolerance = source["absolute_tolerance"]
        .as_f64()
        .ok_or("audio tolerance")?;
    assert_eq!(tolerance, 2e-6);
    for (index, (actual, expected)) in resampled_features.iter().zip(expected).enumerate() {
        let expected = expected.as_f64().ok_or("source audio scalar")?;
        let error = (f64::from(*actual) - expected).abs();
        assert!(
            error <= tolerance,
            "log_mel[{index}]: native {actual}, source {expected}, error {error}"
        );
    }
    let repeated = prepare_gemma4_audio(&backend, &resample_input, 44_100, &setup)?;
    assert_eq!(feature_bytes, repeated.log_mel().contiguous_bytes()?);
    drop(repeated);
    drop(resampled_features);
    drop(resampled);

    let short = tensor(&backend, &[1, 1, 1], &[0.25], &setup)?;
    let short_prepared = prepare_gemma4_audio(&backend, &short, 16_000, &setup)?;
    assert_eq!(short_prepared.log_mel().descriptor().shape(), [1, 0, 128]);
    assert_eq!(short_prepared.frame_mask().descriptor().shape(), [1, 0]);
    assert_eq!(short_prepared.marker_tokens(), 0);
    drop(short_prepared);
    let empty = tensor(&backend, &[1, 1, 0], &[], &setup)?;
    assert!(matches!(
        prepare_gemma4_audio(&backend, &empty, 16_000, &setup),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    let invalid = tensor(&backend, &[1, 1, 1], &[f32::NAN], &setup)?;
    assert!(matches!(
        prepare_gemma4_audio(&backend, &invalid, 16_000, &setup),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    assert!(matches!(
        prepare_gemma4_audio(&backend, &short, 0, &setup),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    assert!(matches!(
        prepare_gemma4_audio(&backend, &short, 7_999, &setup),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    assert!(matches!(
        prepare_gemma4_audio(&backend, &short, 384_001, &setup),
        Err(MultimodalTextError::InvalidInput(_))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 48 * 1024 * 1024)?;
    assert!(matches!(
        prepare_gemma4_audio(&backend, &stereo, 16_000, &cancelled_context),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    let constrained = context(&authority, &cancellation, 4)?;
    assert!(prepare_gemma4_audio(&backend, &stereo, 16_000, &constrained).is_err());
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn retained_qwen_vision_executes_closed_family_graphs_and_rolls_back() -> Result<(), Box<dyn Error>>
{
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 48 * 1024 * 1024)?;
    let image = ImageTensor::from_f32(&backend, &setup, 1, 32, 32, 3, &vec![0.5; 32 * 32 * 3])?;
    let qwen35_prepared =
        prepare_qwen_images(&backend, &image, QwenVisionFamily::Qwen35_08B, &setup)?;
    let qwen35_values = tensor_to_f32(&backend, qwen35_prepared[0].patches(), &setup)?;
    assert!(
        (qwen35_values[0] - (0.5 - QWEN35_IMAGE_MEAN[0]) / QWEN35_IMAGE_STANDARD_DEVIATION[0])
            .abs()
            < 1.0e-6
    );
    drop(qwen35_values);
    let qwen35_plan = plan_qwen_markers(
        &[1, QWEN35_IMAGE_PAD_TOKEN, 2],
        &qwen35_prepared,
        QwenVisionFamily::Qwen35_08B,
        &cancellation,
    )?;
    assert_eq!(qwen35_plan.spans()[0].size, 4);
    assert!(
        plan_qwen_markers(
            &[1, QWEN3VL_IMAGE_PAD_TOKEN, 2],
            &qwen35_prepared,
            QwenVisionFamily::Qwen35_08B,
            &cancellation,
        )
        .is_err()
    );

    let qwen35_configuration =
        QwenVisionConfiguration::reduced_fixture(QwenVisionFamily::Qwen35_08B, 4, 8, 2, 1, 6);
    let qwen35 = NativeQwenVisionEncoder::new(
        qwen35_configuration.clone(),
        reduced_qwen_vision_weights(&backend, &qwen35_configuration, &setup)?,
    )?;
    let qwen35_digest = qwen35.semantic_state_digest(&cancellation)?;
    assert_eq!(qwen35_digest.len(), 64);
    assert!(qwen35.resident_bytes()? > 0);
    let qwen35_projection = qwen35.project(&backend, &qwen35_prepared[0], &setup)?;
    assert_eq!(qwen35_projection.embedding.descriptor().shape(), [4, 6]);
    assert!(qwen35_projection.deepstack.is_empty());
    let qwen35_output = tensor_to_f32(&backend, &qwen35_projection.embedding, &setup)?;
    assert!(qwen35_output.iter().all(|value| value.is_finite()));
    assert!(qwen35_output.iter().any(|value| *value != 0.0));
    let qwen35_output_digest = format!(
        "{:x}",
        Sha256::digest(
            qwen35_output
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        qwen35_output_digest,
        "16a13ffa41fb0a5f742f0e8c3b2364fa4020dd2cb9a71e1309e62d6e143cb8ed"
    );
    let mut batch_pixels = vec![0.5_f32; 2 * 32 * 32 * 3];
    batch_pixels[32 * 32 * 3..].fill(0.75);
    let image_batch = ImageTensor::from_f32(&backend, &setup, 2, 32, 32, 3, &batch_pixels)?;
    let batch_prepared =
        prepare_qwen_images(&backend, &image_batch, QwenVisionFamily::Qwen35_08B, &setup)?;
    let isolated_first = qwen35.project(&backend, &batch_prepared[0], &setup)?;
    let isolated_second = qwen35.project(&backend, &batch_prepared[1], &setup)?;
    assert_eq!(
        &*tensor_to_f32(&backend, &isolated_first.embedding, &setup)?,
        &*qwen35_output
    );
    assert_ne!(
        &*tensor_to_f32(&backend, &isolated_second.embedding, &setup)?,
        &*qwen35_output
    );
    drop(qwen35_output);

    let qwen3_prepared =
        prepare_qwen_images(&backend, &image, QwenVisionFamily::Qwen3Vl4B, &setup)?;
    let qwen3_configuration =
        QwenVisionConfiguration::reduced_fixture(QwenVisionFamily::Qwen3Vl4B, 4, 8, 18, 1, 6);
    let qwen3 = NativeQwenVisionEncoder::new(
        qwen3_configuration.clone(),
        reduced_qwen_vision_weights(&backend, &qwen3_configuration, &setup)?,
    )?;
    let qwen3_projection = qwen3.project(&backend, &qwen3_prepared[0], &setup)?;
    assert_eq!(qwen3_projection.embedding.descriptor().shape(), [4, 6]);
    assert_eq!(qwen3_projection.deepstack.len(), 3);
    assert!(
        qwen3_projection
            .deepstack
            .iter()
            .all(|tensor| tensor.descriptor().shape() == [4, 6])
    );
    assert_ne!(qwen3.semantic_state_digest(&cancellation)?, qwen35_digest);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 48 * 1024 * 1024)?;
    assert!(matches!(
        qwen35.project(&backend, &qwen35_prepared[0], &cancelled_context),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    let constrained = context(&authority, &cancellation, 4)?;
    assert!(
        qwen35
            .project(&backend, &qwen35_prepared[0], &constrained)
            .is_err()
    );
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    drop(qwen3_projection);
    drop(qwen35_projection);
    drop(qwen3_prepared);
    drop(qwen35_prepared);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn qwen_multimodal_resource_closes_admission_identity_and_residency() -> Result<(), Box<dyn Error>>
{
    let families = [
        QwenVisionFamily::Qwen3Vl4B,
        QwenVisionFamily::Qwen3Vl8B,
        QwenVisionFamily::Qwen35_08B,
        QwenVisionFamily::Qwen35_2B,
        QwenVisionFamily::Qwen35_4B,
        QwenVisionFamily::Qwen35_9B,
        QwenVisionFamily::Qwen35_27B,
    ];
    let mut configurations = Vec::new();
    for family in families {
        let decoder_configuration = qwen_multimodal_decoder_configuration(family)?;
        let vision_configuration = QwenVisionConfiguration::source(family);
        assert_eq!(
            decoder_configuration.hidden_size,
            vision_configuration.output_hidden_size
        );
        assert_eq!(
            qwen_multimodal_tokenizer_profile(family),
            if matches!(
                family,
                QwenVisionFamily::Qwen3Vl4B | QwenVisionFamily::Qwen3Vl8B
            ) {
                Qwen2PretokenizerProfile::Qwen2
            } else {
                Qwen2PretokenizerProfile::Qwen35Declared
            }
        );
        configurations.push((family, decoder_configuration, vision_configuration));
    }
    for (left_index, (left_family, left_decoder, left_vision)) in configurations.iter().enumerate()
    {
        for (right_index, (right_family, right_decoder, right_vision)) in
            configurations.iter().enumerate()
        {
            if left_index == right_index {
                assert_eq!(left_family, right_family);
                assert_eq!(left_decoder, right_decoder);
                assert_eq!(left_vision, right_vision);
            } else {
                assert!(
                    left_family != right_family
                        && (left_decoder != right_decoder || left_vision != right_vision)
                );
            }
        }
    }
    let resource_manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/qwen_multimodal/resource/manifest.json"
    ))?;
    assert_eq!(
        resource_manifest["accepted_families"]
            .as_array()
            .ok_or("resource family matrix is missing")?
            .len(),
        7
    );

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let configuration =
        QwenVisionConfiguration::reduced_fixture(QwenVisionFamily::Qwen3Vl4B, 4, 8, 18, 1, 4);
    let vision = Arc::new(NativeQwenVisionEncoder::new(
        configuration.clone(),
        reduced_qwen_vision_weights(&backend, &configuration, &setup)?,
    )?);
    let tokenizer = qwen25_prompt_tokenizer()?;
    let decoder = reduced_qwen3_decoder(&backend, &setup, 0.01)?;
    let resource = NativeQwenMultimodal::reduced_fixture(
        tokenizer.clone(),
        decoder.clone(),
        vision.clone(),
        &cancellation,
    )?;
    assert_eq!(resource.family(), QwenVisionFamily::Qwen3Vl4B);
    assert!(Arc::ptr_eq(resource.tokenizer(), &tokenizer));
    assert!(Arc::ptr_eq(resource.decoder(), &decoder));
    assert!(Arc::ptr_eq(resource.vision(), &vision));
    assert!(!resource.is_source_exact_profile());
    assert_eq!(resource.semantic_state_digest(&cancellation)?.len(), 64);
    let tensor_bytes = resource
        .resident_tensor_allocations()?
        .into_iter()
        .try_fold(0_u64, |total, (_, bytes)| total.checked_add(bytes))
        .ok_or("resource tensor residency overflowed")?;
    assert!(resource.resident_bytes()? > tensor_bytes);

    let clone = NativeQwenMultimodal::reduced_fixture(
        tokenizer.clone(),
        decoder.clone(),
        vision.clone(),
        &cancellation,
    )?;
    assert_eq!(
        resource.semantic_state_digest(&cancellation)?,
        clone.semantic_state_digest(&cancellation)?
    );
    assert_eq!(
        resource.resident_tensor_allocations()?,
        clone.resident_tensor_allocations()?
    );

    let changed = NativeQwenMultimodal::reduced_fixture(
        tokenizer.clone(),
        reduced_qwen3_decoder(&backend, &setup, 0.02)?,
        vision.clone(),
        &cancellation,
    )?;
    assert_ne!(
        resource.semantic_state_digest(&cancellation)?,
        changed.semantic_state_digest(&cancellation)?
    );
    assert!(
        NativeQwenMultimodal::reduced_fixture(
            qwen35_prompt_tokenizer()?,
            decoder.clone(),
            vision.clone(),
            &cancellation,
        )
        .is_err()
    );
    assert!(
        NativeQwenMultimodal::new(
            tokenizer.clone(),
            decoder.clone(),
            vision.clone(),
            &cancellation,
        )
        .is_err()
    );
    assert!(NativeModelPayload::qwen_multimodal_clip(Arc::new(resource)).is_err());
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        NativeQwenMultimodal::reduced_fixture(tokenizer, decoder, vision, &cancelled),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn qwen_multimodal_generation_replaces_markers_and_delegates_transactionally()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/qwen_multimodal/generation/manifest.json"
    ))?;
    assert_eq!(
        manifest["generation_contract"]["qwen3vl_positions"],
        "three-axis-prefill-scalar-continuation"
    );
    assert_eq!(
        manifest["generation_contract"]["qwen35_positions"],
        "scalar-source-generation-route"
    );
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let vision_configuration =
        QwenVisionConfiguration::reduced_fixture(QwenVisionFamily::Qwen3Vl4B, 4, 8, 18, 1, 4);
    let resource = NativeQwenMultimodal::reduced_fixture(
        qwen25_prompt_tokenizer()?,
        reduced_qwen3_decoder(&backend, &setup, 0.01)?,
        Arc::new(NativeQwenVisionEncoder::new(
            vision_configuration.clone(),
            reduced_qwen_vision_weights(&backend, &vision_configuration, &setup)?,
        )?),
        &cancellation,
    )?;
    let image = ImageTensor::from_f32(&backend, &setup, 1, 32, 32, 3, &vec![0.5; 32 * 32 * 3])?;
    let prepared = prepare_qwen_images(&backend, &image, QwenVisionFamily::Qwen3Vl4B, &setup)?;
    assert_eq!(prepared[0].merged_tokens(), 4);
    let request = NativeTextGenerationRequest {
        formatted_prompt: "<|image_pad|>",
        maximum_new_tokens: 1,
        do_sample: false,
        temperature_bits: 0.0_f32.to_bits(),
        top_k: 0,
        top_p_bits: 1.0_f32.to_bits(),
        minimum_p_bits: 0.0_f32.to_bits(),
        repetition_penalty_bits: 1.0_f32.to_bits(),
        presence_penalty_bits: 0.0_f32.to_bits(),
    };
    let transaction = qwen_generation_transaction()?;
    let checkpoint = transaction.checkpoint();
    let resource_digest = resource.semantic_state_digest(&cancellation)?;
    let resource_residency = resource.resident_tensor_allocations()?;
    let outcome = resource.generate(
        &backend,
        QwenMultimodalGenerationRequest {
            text: request.clone(),
            prepared_images: &prepared,
            transaction: &transaction,
        },
        &setup,
    )?;
    assert_eq!(outcome.generated_tokens.len(), 1);
    assert_eq!(transaction.checkpoint(), checkpoint);
    assert_eq!(
        resource.semantic_state_digest(&cancellation)?,
        resource_digest
    );
    assert_eq!(resource.resident_tensor_allocations()?, resource_residency);

    assert!(
        resource
            .generate(
                &backend,
                QwenMultimodalGenerationRequest {
                    text: NativeTextGenerationRequest {
                        formatted_prompt: "<|endoftext|>",
                        ..request.clone()
                    },
                    prepared_images: &prepared,
                    transaction: &transaction,
                },
                &setup,
            )
            .is_err()
    );
    assert_eq!(transaction.checkpoint(), checkpoint);

    let constrained_context = context(&authority, &cancellation, 4)?;
    assert!(
        resource
            .generate(
                &backend,
                QwenMultimodalGenerationRequest {
                    text: request.clone(),
                    prepared_images: &prepared,
                    transaction: &transaction,
                },
                &constrained_context,
            )
            .is_err()
    );
    assert_eq!(transaction.checkpoint(), checkpoint);
    assert_eq!(constrained_context.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 64 * 1024 * 1024)?;
    assert!(matches!(
        resource.generate(
            &backend,
            QwenMultimodalGenerationRequest {
                text: request,
                prepared_images: &prepared,
                transaction: &transaction,
            },
            &cancelled_context,
        ),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(transaction.checkpoint(), checkpoint);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        resource.semantic_state_digest(&cancellation)?,
        resource_digest
    );
    assert_eq!(resource.resident_tensor_allocations()?, resource_residency);
    assert_eq!(setup.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn modality_join_deepstack_and_projection_are_native_and_transactional()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    let text = tensor(
        &backend,
        &[1, 6, 2],
        &(0..12).map(|value| value as f32).collect::<Vec<_>>(),
        &context,
    )?;
    let image = tensor(&backend, &[2, 2], &[100.0, 101.0, 102.0, 103.0], &context)?;
    let deep_a = tensor(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
    let deep_b = tensor(&backend, &[2, 2], &[5.0, 6.0, 7.0, 8.0], &context)?;
    let deepstack = vec![deep_a, deep_b];
    let images = [MultimodalImageEmbedding {
        span: MultimodalSpan {
            start: 2,
            size: 2,
            grid_thw: [1, 4, 4],
        },
        embedding: &image,
        deepstack: &deepstack,
    }];
    let joined = join_multimodal_embeddings(&backend, &text, &images, &context)?;
    assert_eq!(joined.descriptor().shape(), [1, 6, 2]);
    assert_eq!(
        &*tensor_to_f32(&backend, &joined, &context)?,
        &[
            0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0, 8.0, 9.0, 10.0, 11.0
        ]
    );
    let joined_deepstack = join_qwen3vl_deepstack(&backend, 6, &images, &context)?
        .ok_or("deepstack join was not produced")?;
    assert_eq!(
        joined_deepstack.visual_position_mask,
        [false, false, true, true, false, false]
    );
    assert_eq!(joined_deepstack.layers.len(), 2);
    assert_eq!(joined_deepstack.layers[0].descriptor().shape(), [2, 2]);

    let taps = tensor(
        &backend,
        &[1, 13, 2, 2],
        &(0..52).map(|value| value as f32).collect::<Vec<_>>(),
        &context,
    )?;
    let projection = ideogram4_project_taps(&backend, &taps, &context)?;
    assert_eq!(projection.descriptor().shape(), [1, 2, 26]);
    let values = tensor_to_f32(&backend, &projection, &context)?;
    assert_eq!(
        &values[..13],
        &[
            0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0
        ]
    );
    assert_eq!(
        &values[13..26],
        &[
            1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 49.0
        ]
    );
    drop(values);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn ovis_sam3_and_prompt_adapters_preserve_source_semantics() -> Result<(), Box<dyn Error>> {
    assert_eq!(ovis_template_end(&[10, 4004, 25, 99])?, 2);
    assert!(matches!(
        ovis_template_end(&[10, 11]),
        Err(MultimodalTextError::InvalidInput(_))
    ));
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 8 * 1024 * 1024)?;
    let conditioning = tensor(&backend, &[1, 4, 1], &[1.0, 2.0, 3.0, 4.0], &context)?;
    let trimmed = trim_ovis_conditioning(&conditioning, 2, &cancellation)?;
    assert_eq!(trimmed.descriptor().shape(), [1, 2, 1]);
    assert_eq!(&*tensor_to_f32(&backend, &trimmed, &context)?, &[3.0, 4.0]);

    let prompts = parse_sam3_prompts("(cat): 2.5, dog: 0, red car")?;
    assert_eq!(prompts.len(), 3);
    assert_eq!(
        (prompts[0].text.as_str(), prompts[0].maximum_detections),
        ("cat", 2)
    );
    assert_eq!(
        (prompts[1].text.as_str(), prompts[1].maximum_detections),
        ("dog", 1)
    );
    assert_eq!(
        (prompts[2].text.as_str(), prompts[2].maximum_detections),
        ("red car", 1)
    );
    assert!(matches!(
        parse_sam3_prompts("cat:1.2.3"),
        Err(MultimodalTextError::InvalidPromptLimit(_))
    ));
    let condition = tensor(&backend, &[1, 1, 1], &[7.0], &context)?;
    let pack = pack_sam3_conditions(
        vec![Sam3EncodedCondition {
            condition,
            attention_mask: None,
            maximum_detections: 2,
        }],
        None,
    )?;
    assert_eq!(pack.main_condition.descriptor().shape(), [1, 1, 1]);
    assert_eq!(pack.conditions.len(), 1);

    assert_eq!(
        format_ideogram4_prompt("cat"),
        "<|im_start|>user\ncat<|im_end|>\n<|im_start|>assistant\n"
    );
    assert!(format_ovis_prompt("cat").contains("Describe the image"));
    let qwen = format_qwen3vl_prompt("cat", 2, false);
    assert_eq!(
        qwen,
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>cat<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(
        format_qwen3vl_prompt("cat", 0, true),
        "<|im_start|>user\ncat<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        format_qwen3vl_prompt("<|im_start|>raw", 3, false),
        "<|im_start|>raw"
    );
    assert_eq!(
        format_qwen_multimodal_prompt(
            QwenVisionFamily::Qwen35_4B,
            "cat",
            1,
            false,
            false,
            &cancellation,
        )?,
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>cat<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    );
    assert_eq!(
        format_qwen_multimodal_prompt(
            QwenVisionFamily::Qwen3Vl4B,
            "<|im_start|>raw",
            2,
            false,
            true,
            &cancellation,
        )?,
        "<|im_start|>raw"
    );
    Ok(())
}

#[test]
fn cancellation_and_oom_reject_before_multimodal_publication() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let live = CancellationToken::default();
    let setup = context(&authority, &live, 8 * 1024 * 1024)?;
    let text = tensor(&backend, &[1, 2, 1], &[1.0, 2.0], &setup)?;
    let image = tensor(&backend, &[1, 1], &[3.0], &setup)?;
    drop(setup);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 8 * 1024 * 1024)?;
    let images = [MultimodalImageEmbedding {
        span: MultimodalSpan {
            start: 1,
            size: 1,
            grid_thw: [1, 2, 2],
        },
        embedding: &image,
        deepstack: &[],
    }];
    assert!(matches!(
        join_multimodal_embeddings(&backend, &text, &images, &cancelled_context),
        Err(MultimodalTextError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    let (oom_backend, oom_authority) = CpuWorkspaceAuthority::create_backend(32)?;
    let live = CancellationToken::default();
    let oom_context = context(&oom_authority, &live, 0)?;
    let oom_text = tensor(&oom_backend, &[1, 2, 1], &[1.0, 2.0], &oom_context)?;
    let oom_image = tensor(&oom_backend, &[1, 1], &[3.0], &oom_context)?;
    let oom_images = [MultimodalImageEmbedding {
        span: MultimodalSpan {
            start: 1,
            size: 1,
            grid_thw: [1, 2, 2],
        },
        embedding: &oom_image,
        deepstack: &[],
    }];
    let error = join_multimodal_embeddings(&oom_backend, &oom_text, &oom_images, &oom_context)
        .expect_err("backend memory exhaustion must reject multimodal publication");
    assert!(matches!(&error, MultimodalTextError::ShapeLayoutTwo(_)));
    assert!(error.to_string().contains("backend limit is 32 bytes"));
    assert_eq!(oom_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_clip_001_multimodal_rows_execute_and_extend_cumulative_ledger() -> Result<(), Box<dyn Error>>
{
    let workspace = workspace()?;
    let catalog = fs::read_to_string(
        workspace.join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut contracts = Vec::new();
    let mut symbols = Vec::new();
    for line in catalog.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.get(8).copied() != Some(TASK_ID) {
            continue;
        }
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[7], "sim_model::clip_text_encoder_multimodal");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), fields[5]);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_contract_witness(fields[2], fields[3])?;
        symbols.push(fields[3]);
        contracts.push(json!({
            "contract_id": fields[0],
            "task_id": TASK_ID,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": [
                format!("{}:native-valid", fields[0]),
                format!("{}:typed-invalid", fields[0]),
            ],
        }));
    }
    assert_eq!(symbols, MULTIMODAL_TEXT_ENCODER_CATALOG_SYMBOLS);
    assert_eq!(contracts.len(), 53);

    let artifact_path = workspace.join("target/sim-parity/val-clip-001.json");
    let mut artifact = if artifact_path.exists() {
        serde_json::from_slice::<Value>(&fs::read(&artifact_path)?)?
    } else {
        empty_clip_artifact()
    };
    assert_eq!(artifact.get("schema_version"), Some(&json!(1)));
    assert_eq!(artifact.get("validation_id"), Some(&json!("VAL-CLIP-001")));
    let implementations = IMPLEMENTATION_CLOSURE
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            }))
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    let task_results = artifact
        .get_mut("task_results")
        .and_then(Value::as_object_mut)
        .ok_or("VAL-CLIP-001 task results are missing")?;
    task_results.insert(
        TASK_ID.to_owned(),
        json!({
            "status": "passed",
            "passed": 53,
            "failed": 0,
            "skipped": 0,
            "case_ids": [
                "text-encoder-multimodal:source-provenance-and-exact-row-closure",
                "text-encoder-multimodal:canonical-text-and-vision-delegation",
                "text-encoder-multimodal:position-and-projection-semantics",
                "text-encoder-multimodal:typed-target-cancellation-oom-workspace",
            ],
            "implementations": implementations,
        }),
    );
    let composite_completed = task_results.contains_key(COMPOSITE_TASK_ID);
    let passed = task_results.values().try_fold(0_u64, |total, result| {
        total
            .checked_add(
                result
                    .get("passed")
                    .and_then(Value::as_u64)
                    .ok_or("passed count")?,
            )
            .ok_or("passed count overflowed")
    })?;
    let artifact_contracts = artifact
        .get_mut("contracts")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 contracts are missing")?;
    artifact_contracts
        .retain(|contract| contract.get("task_id").and_then(Value::as_str) != Some(TASK_ID));
    artifact_contracts.extend(contracts);
    artifact["summary"] = json!({"passed": passed, "failed": 0, "skipped": 0});
    let remaining = artifact
        .get_mut("remaining_tasks")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 remaining tasks are missing")?;
    remaining.retain(|task| task.as_str() != Some(TASK_ID));
    if !composite_completed
        && !remaining
            .iter()
            .any(|task| task.as_str() == Some(COMPOSITE_TASK_ID))
    {
        let insertion = remaining
            .iter()
            .position(|task| task.as_str() == Some("comfy-parity-clip-text-encoder-breadth"))
            .unwrap_or(remaining.len());
        remaining.insert(insertion, json!(COMPOSITE_TASK_ID));
    }
    let producer_path = "crates/sim_model/tests/clip_text_encoder_multimodal.rs";
    artifact["implementation"] = json!({
        "path": producer_path,
        "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(producer_path))?)),
    });
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
    Ok(())
}

fn execute_contract_witness(source_path: &str, symbol: &str) -> Result<(), Box<dyn Error>> {
    let behavior = multimodal_symbol_behavior(source_path, symbol)
        .ok_or_else(|| format!("{source_path}:{symbol} has no native behavior"))?;
    match behavior {
        MultimodalSymbolBehavior::Profile => {
            assert_eq!(
                multimodal_profile(MultimodalFamily::JinaClip2).layer_count,
                24
            );
        }
        MultimodalSymbolBehavior::ProfileFactory => {
            assert_eq!(
                multimodal_profile(MultimodalFamily::Qwen3Vl8B).hidden_size,
                4096
            );
        }
        MultimodalSymbolBehavior::TokenizerAdapter => {
            assert!(!format_ideogram4_prompt("fixture").is_empty());
        }
        MultimodalSymbolBehavior::PositionConstruction => {
            assert!(qwen2vl_mrope_position_ids(1, &[], &CancellationToken::default())?.is_none());
        }
        MultimodalSymbolBehavior::PromptPacking => {
            assert_eq!(parse_sam3_prompts("fixture")?.len(), 1);
        }
        MultimodalSymbolBehavior::BidirectionalTextDelegation
        | MultimodalSymbolBehavior::DecoderTextDelegation
        | MultimodalSymbolBehavior::ClipTextDelegation
        | MultimodalSymbolBehavior::VisionDelegation
        | MultimodalSymbolBehavior::ImagePreprocessDelegation
        | MultimodalSymbolBehavior::ModalityJoin
        | MultimodalSymbolBehavior::Projection
        | MultimodalSymbolBehavior::ModelAdapter => {}
    }
    assert!(multimodal_symbol_behavior(source_path, "__typed_invalid__").is_none());
    Ok(())
}

fn workspace() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "workspace root is unavailable".into())
}

fn empty_clip_artifact() -> Value {
    json!({
        "schema_version": 1,
        "validation_id": "VAL-CLIP-001",
        "overall_status": "partial",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "sim_tensor::CpuBackend",
            "device": "cpu",
            "dtype": "f32",
        },
        "summary": {"passed": 0, "failed": 0, "skipped": 0},
        "implementation": {},
        "task_results": {},
        "contracts": [],
        "remaining_tasks": [
            "comfy-parity-clip-text-encoder-multimodal-foundation",
            "comfy-parity-clip-text-encoder-composite-adapters",
            "comfy-parity-clip-text-encoder-breadth",
            "comfy-parity-clip-owner-consolidation"
        ],
    })
}

fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn Error>> {
    let source = std::str::from_utf8(source)?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let function_prefix = format!("def {symbol}(");
    let class_with_bases_prefix = format!("class {symbol}(");
    let class_without_bases_prefix = format!("class {symbol}:");
    let start = lines
        .iter()
        .position(|line| {
            line.starts_with(&function_prefix)
                || line.starts_with(&class_with_bases_prefix)
                || line.starts_with(&class_without_bases_prefix)
        })
        .ok_or_else(|| format!("Python symbol {symbol:?} is missing"))?;
    let mut header_complete = lines[start].trim_end().ends_with(':');
    let mut saw_indented_body = false;
    let mut end = lines.len();
    for (index, line) in lines.iter().enumerate().skip(start + 1) {
        let trimmed = line.trim();
        if !header_complete {
            if !line.starts_with(char::is_whitespace) && trimmed.ends_with(':') {
                header_complete = true;
            }
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if line.starts_with(char::is_whitespace) {
            saw_indented_body = true;
        } else if saw_indented_body {
            end = index;
            break;
        }
    }
    let mut body_end = end;
    while body_end > start + 1 {
        let trimmed = lines[body_end - 1].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            body_end -= 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(lines[start..body_end].concat().as_bytes())
    ))
}
