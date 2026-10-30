use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::{
    ClipBpeTokenizer, DECODER_PROFILE_FACTS, DECODER_TEXT_ENCODER_CATALOG_SYMBOLS,
    DecoderActivation, DecoderArchitecture, DecoderAttentionWeights,
    DecoderGenerationConfiguration, DecoderLayerKind, DecoderLayerWeights,
    DecoderPreparedDeepstack, DecoderPreparedGenerationPrompt, DecoderPreparedTextRequest,
    DecoderRopeConfiguration, DecoderRopePositions, DecoderSymbolBehavior,
    DecoderTextConfiguration, DecoderTextError, DecoderTextRequest, DecoderTextWeights,
    GEMMA4_SOURCE_PATH, GEMMA4_SOURCE_SHA256, GPT_OSS_SOURCE_PATH, GPT_OSS_SOURCE_SHA256,
    Gemma3DecoderConfiguration, Gemma3DecoderProfile, Gemma4DecoderConfiguration,
    Gemma4DecoderProfile, Gemma4LayerInputWeights, Gemma4PerLayerWeights, LLAMA_SOURCE_PATH,
    LLAMA_SOURCE_SHA256, ModelTokenizerDescriptor, NativeDecoderTextEncoder, NativeModelPayload,
    NativePromptTokenizer, NativeTextGenerationRequest, NativeTokenizerFamily, QWEN35_SOURCE_PATH,
    QWEN35_SOURCE_SHA256, Qwen35LinearConfiguration, Qwen35LinearWeights, RopeScaling,
    TEXT_GENERATION_SOURCE_PATH, TEXT_GENERATION_SOURCE_SHA256, TokenizerConfiguration, apply_rope,
    decoder_profile_fact, decoder_symbol_behavior, gemma3_decoder_configuration,
    gemma4_audio_conv2d_subsample, gemma4_audio_relative_positions, gemma4_clipped_linear,
    gemma4_decoder_configuration, gemma4_vision_patch_embed, gemma4_vision_rope, gpt_oss_moe,
    gpt_oss_top_k_route, precompute_multidimensional_rope, precompute_rope,
    qwen35_causal_conv1d_update, qwen35_causal_conv1d_update_exact, qwen35_chunk_gated_delta_rule,
    qwen35_chunk_gated_delta_rule_exact, qwen35_vision_patch_embed, qwen35_vision_patch_merge,
    tokenize_decoder_prompt,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    RetryRngPolicy, RngAlgorithm, RngProfileVersion, RngStream, RngStreamAddress, StreamId, Tensor,
    TensorDescriptor, generated_native_diffusion::tensor_to_f32,
};
use std::{collections::BTreeMap, error::Error, fs, path::Path, sync::Arc};

const MEMORY_LIMIT: u64 = 128 * 1024 * 1024;
const TASK_ID: &str = "comfy-parity-clip-text-encoder-decoder-foundation";
const IMPLEMENTATION_CLOSURE: [&str; 4] = [
    "crates/sim_model/src/clip_text_encoder_decoder.rs",
    "crates/sim_model/src/sim_model.rs",
    "crates/sim_model/tests/clip_backend_admission.rs",
    "crates/sim_model/tests/clip_text_encoder_decoder.rs",
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

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
    Ok(tensor)
}

fn i64_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, context.stream)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let (tensor, _) = backend.upload_bytes(descriptor, &bytes, context)?;
    Ok(tensor)
}

fn filled(
    backend: &CpuBackend,
    shape: &[u64],
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = shape.iter().try_fold(1_usize, |count, dimension| {
        count.checked_mul(usize::try_from(*dimension).ok()?)
    });
    tensor(
        backend,
        shape,
        &vec![value; count.ok_or("tensor shape overflow")?],
        context,
    )
}

fn matrix(
    backend: &CpuBackend,
    output: usize,
    input: usize,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = output.checked_mul(input).ok_or("matrix size overflow")?;
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

fn configuration(architecture: DecoderArchitecture) -> DecoderTextConfiguration {
    let qwen = architecture == DecoderArchitecture::Qwen35;
    DecoderTextConfiguration {
        architecture,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: 8,
        maximum_tokens: 8,
        hidden_size: 4,
        feed_forward_size: 8,
        layer_kinds: if qwen {
            vec![
                DecoderLayerKind::LinearAttention,
                DecoderLayerKind::LinearAttention,
                DecoderLayerKind::LinearAttention,
                DecoderLayerKind::FullAttention,
            ]
        } else {
            vec![
                DecoderLayerKind::FullAttention,
                DecoderLayerKind::SlidingAttention,
            ]
        },
        attention_heads: 2,
        key_value_heads: if qwen { 2 } else { 1 },
        head_dimension: 2,
        query_key_norm: qwen,
        qwen35_linear: qwen.then_some(Qwen35LinearConfiguration {
            key_heads: 1,
            value_heads: 2,
            key_head_dimension: 2,
            value_head_dimension: 2,
            convolution_kernel_size: 3,
        }),
        gemma3: None,
        gemma4: None,
        normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
        rope: DecoderRopeConfiguration {
            theta: 10_000.0,
            rotary_dimension: 2,
            interleaved_sections: Vec::new(),
            scaling: if architecture == DecoderArchitecture::GptOss {
                RopeScaling::Yarn {
                    factor: 2.0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                }
            } else {
                RopeScaling::None
            },
        },
        sliding_window: Some(3),
        activation: if architecture == DecoderArchitecture::GptOss {
            DecoderActivation::GeluTanh
        } else {
            DecoderActivation::Silu
        },
        embedding_scale_bits: if architecture == DecoderArchitecture::Gemma {
            2.0_f32.to_bits()
        } else {
            1.0_f32.to_bits()
        },
        residual_scale_bits: 1.0_f32.to_bits(),
        norm_weight_offset_bits: if architecture == DecoderArchitecture::Gemma || qwen {
            1.0_f32.to_bits()
        } else {
            0.0_f32.to_bits()
        },
        logits_soft_cap_bits: (architecture == DecoderArchitecture::Gemma)
            .then_some(4.0_f32.to_bits()),
        tied_output_head: true,
        stop_tokens: vec![7],
    }
}

fn reduced_gemma3_configuration() -> DecoderTextConfiguration {
    let mut configuration = configuration(DecoderArchitecture::Gemma);
    configuration.layer_kinds = vec![
        DecoderLayerKind::SlidingAttention,
        DecoderLayerKind::SlidingAttention,
        DecoderLayerKind::SlidingAttention,
        DecoderLayerKind::SlidingAttention,
        DecoderLayerKind::SlidingAttention,
        DecoderLayerKind::FullAttention,
    ];
    configuration.query_key_norm = true;
    configuration.gemma3 = Some(Gemma3DecoderConfiguration {
        local_rope: DecoderRopeConfiguration {
            theta: 10_000.0,
            rotary_dimension: 2,
            interleaved_sections: Vec::new(),
            scaling: RopeScaling::None,
        },
    });
    configuration.rope = DecoderRopeConfiguration {
        theta: 1_000_000.0,
        rotary_dimension: 2,
        interleaved_sections: Vec::new(),
        scaling: RopeScaling::Linear { factor: 8.0 },
    };
    configuration.sliding_window = Some(1024);
    configuration.activation = DecoderActivation::GeluTanh;
    configuration.logits_soft_cap_bits = None;
    configuration.stop_tokens = vec![1, 106];
    configuration
}

fn reduced_gemma4_configuration() -> DecoderTextConfiguration {
    let mut configuration = configuration(DecoderArchitecture::Gemma);
    configuration.layer_kinds = (0_usize..10)
        .map(|index| {
            if (index + 1).is_multiple_of(5) {
                DecoderLayerKind::FullAttention
            } else {
                DecoderLayerKind::SlidingAttention
            }
        })
        .collect();
    configuration.query_key_norm = true;
    configuration.gemma3 = None;
    configuration.gemma4 = Some(Gemma4DecoderConfiguration {
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
    });
    configuration.rope = DecoderRopeConfiguration {
        theta: 1_000_000.0,
        rotary_dimension: 2,
        interleaved_sections: Vec::new(),
        scaling: RopeScaling::None,
    };
    configuration.sliding_window = Some(512);
    configuration.activation = DecoderActivation::GeluTanh;
    configuration.norm_weight_offset_bits = 0.0_f32.to_bits();
    configuration.logits_soft_cap_bits = Some(30.0_f32.to_bits());
    configuration.stop_tokens = vec![1, 50, 106];
    configuration
}

fn gemma4_weights(
    backend: &CpuBackend,
    configuration: &DecoderTextConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<DecoderTextWeights, Box<dyn Error>> {
    let gemma4 = configuration.gemma4.as_ref().ok_or("Gemma4 config")?;
    let token_embedding = tensor(
        backend,
        &[8, 4],
        &(0..32)
            .map(|index| ((index * 11 % 23) as f32 - 11.0) / 17.0)
            .collect::<Vec<_>>(),
        context,
    )?;
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
        let scale = 0.09 + index as f32 * 0.007;
        layers.push(DecoderLayerWeights {
            attention_norm_weight: filled(backend, &[4], 0.3, context)?,
            attention: DecoderAttentionWeights::DotProduct {
                query_weight: matrix(backend, query_width, 4, scale, context)?,
                key_weight: matrix(backend, key_value_width, 4, scale + 0.03, context)?,
                value_weight: matrix(backend, key_value_width, 4, scale + 0.05, context)?,
                query_norm_weight: Some(filled(
                    backend,
                    &[u64::try_from(head_dimension)?],
                    0.8,
                    context,
                )?),
                key_norm_weight: Some(filled(
                    backend,
                    &[u64::try_from(head_dimension)?],
                    1.1,
                    context,
                )?),
                output_weight: matrix(backend, 4, query_width, scale + 0.07, context)?,
            },
            feed_forward_norm_weight: filled(backend, &[4], 0.25, context)?,
            feed_forward_gate_weight: matrix(backend, feed_forward, 4, scale + 0.11, context)?,
            feed_forward_up_weight: matrix(backend, feed_forward, 4, scale + 0.13, context)?,
            feed_forward_down_weight: matrix(backend, 4, feed_forward, scale + 0.17, context)?,
            post_attention_norm_weight: Some(filled(backend, &[4], 0.2, context)?),
            post_feed_forward_norm_weight: Some(filled(backend, &[4], 0.22, context)?),
            attention_sink: None,
            gemma4_layer_input: Some(Gemma4LayerInputWeights {
                gate_weight: matrix(backend, 2, 4, scale + 0.19, context)?,
                projection_weight: matrix(backend, 4, 2, scale + 0.23, context)?,
                post_norm_weight: filled(backend, &[4], 0.4, context)?,
                layer_scalar: tensor(backend, &[1], &[0.97 + index as f32 * 0.001], context)?,
            }),
        });
    }
    let total_per_layer = configuration.layer_kinds.len() * gemma4.hidden_size_per_layer_input;
    Ok(DecoderTextWeights {
        token_embedding,
        layers,
        final_norm_weight: filled(backend, &[4], 0.35, context)?,
        output_head_weight: None,
        gemma4_per_layer: Some(Gemma4PerLayerWeights {
            token_embedding: matrix(backend, 8, total_per_layer, 0.31, context)?,
            model_projection_weight: matrix(backend, total_per_layer, 4, 0.37, context)?,
            projection_norm_weight: tensor(backend, &[2], &[0.9, 1.1], context)?,
        }),
    })
}

fn weights(
    backend: &CpuBackend,
    configuration: &DecoderTextConfiguration,
    context: &ExecutionContext<'_>,
) -> Result<DecoderTextWeights, Box<dyn Error>> {
    let token_embedding = tensor(
        backend,
        &[8, 4],
        &(0..32)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 14.0)
            .collect::<Vec<_>>(),
        context,
    )?;
    let query_width = configuration.attention_heads * configuration.head_dimension;
    let key_value_width = configuration.key_value_heads * configuration.head_dimension;
    let mut layers = Vec::new();
    for (layer, kind) in configuration.layer_kinds.iter().copied().enumerate() {
        let attention = if kind == DecoderLayerKind::LinearAttention {
            let linear = configuration
                .qwen35_linear
                .as_ref()
                .ok_or("linear profile missing")?;
            let key_width = linear.key_heads * linear.key_head_dimension;
            let value_width = linear.value_heads * linear.value_head_dimension;
            let convolution_width = key_width * 2 + value_width;
            DecoderAttentionWeights::Qwen35Linear(Qwen35LinearWeights {
                mixed_query_key_value_weight: matrix(
                    backend,
                    convolution_width,
                    4,
                    0.31 + layer as f32 * 0.01,
                    context,
                )?,
                gate_weight: matrix(backend, value_width, 4, 0.29, context)?,
                beta_weight: matrix(backend, linear.value_heads, 4, 0.17, context)?,
                alpha_weight: matrix(backend, linear.value_heads, 4, -0.13, context)?,
                output_weight: matrix(backend, 4, value_width, 0.41, context)?,
                time_bias: tensor(backend, &[2], &[0.2, -0.3], context)?,
                log_decay: tensor(backend, &[2], &[-0.4, 0.1], context)?,
                convolution_weight: matrix(backend, convolution_width, 3, 0.23, context)?,
                normalization_weight: tensor(backend, &[2], &[0.1, -0.2], context)?,
            })
        } else {
            DecoderAttentionWeights::DotProduct {
                query_weight: matrix(
                    backend,
                    if configuration.architecture == DecoderArchitecture::Qwen35 {
                        query_width * 2
                    } else {
                        query_width
                    },
                    4,
                    0.31 + layer as f32 * 0.01,
                    context,
                )?,
                key_weight: matrix(
                    backend,
                    key_value_width,
                    4,
                    0.29 + layer as f32 * 0.01,
                    context,
                )?,
                value_weight: matrix(
                    backend,
                    key_value_width,
                    4,
                    0.37 + layer as f32 * 0.01,
                    context,
                )?,
                query_norm_weight: configuration
                    .query_key_norm
                    .then(|| tensor(backend, &[2], &[0.75, 1.25], context))
                    .transpose()?,
                key_norm_weight: configuration
                    .query_key_norm
                    .then(|| tensor(backend, &[2], &[1.1, 0.8], context))
                    .transpose()?,
                output_weight: matrix(backend, 4, query_width, 0.41, context)?,
            }
        };
        layers.push(DecoderLayerWeights {
            attention_norm_weight: filled(backend, &[4], 0.2, context)?,
            attention,
            feed_forward_norm_weight: filled(backend, &[4], 0.25, context)?,
            feed_forward_gate_weight: matrix(backend, 8, 4, 0.23, context)?,
            feed_forward_up_weight: matrix(backend, 8, 4, 0.27, context)?,
            feed_forward_down_weight: matrix(backend, 4, 8, 0.33, context)?,
            post_attention_norm_weight: (configuration.architecture == DecoderArchitecture::Gemma)
                .then(|| filled(backend, &[4], 0.15, context))
                .transpose()?,
            post_feed_forward_norm_weight: (configuration.architecture
                == DecoderArchitecture::Gemma)
                .then(|| filled(backend, &[4], 0.18, context))
                .transpose()?,
            attention_sink: (configuration.architecture == DecoderArchitecture::GptOss)
                .then(|| tensor(backend, &[2], &[0.2, -0.1], context))
                .transpose()?,
            gemma4_layer_input: None,
        });
    }
    Ok(DecoderTextWeights {
        token_embedding,
        layers,
        final_norm_weight: filled(backend, &[4], 0.3, context)?,
        output_head_weight: None,
        gemma4_per_layer: None,
    })
}

fn model(
    architecture: DecoderArchitecture,
) -> Result<
    (
        CpuBackend,
        CpuWorkspaceAuthority,
        NativeDecoderTextEncoder,
        CancellationToken,
    ),
    Box<dyn Error>,
> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let configuration = configuration(architecture);
    let model = {
        let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
        NativeDecoderTextEncoder::new(
            configuration.clone(),
            weights(&backend, &configuration, &context)?,
        )?
    };
    Ok((backend, authority, model, cancellation))
}

#[test]
fn decoder_graph_executes_causal_gqa_sliding_cache_and_batch_safe_append()
-> Result<(), Box<dyn Error>> {
    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Llama)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[2, 2], &[1, 2, 3, 4], &context)?;
    let first = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: None,
            cache: None,
            capture_layer: Some(-1),
        },
        &context,
    )?;
    assert_eq!(first.last_hidden_state().descriptor().shape(), [2, 2, 4]);
    assert_eq!(first.logits().descriptor().shape(), [2, 2, 8]);
    assert!(first.intermediate().is_some());
    for layer in first.cache().layers().iter().flatten() {
        let sim_model::DecoderLayerCache::Attention(cache) = layer else {
            return Err("Llama layer produced a non-attention cache".into());
        };
        assert_eq!(cache.tokens(), 2);
        assert_eq!(cache.keys().descriptor().shape(), [2, 2, 1, 2]);
    }

    let next_tokens = i64_tensor(&backend, &[2, 1], &[5, 6], &context)?;
    let second = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &next_tokens,
            attention_mask: None,
            positions: Some(&[2]),
            cache: Some(first.cache()),
            capture_layer: None,
        },
        &context,
    )?;
    assert_eq!(second.logits().descriptor().shape(), [2, 1, 8]);
    for layer in second.cache().layers().iter().flatten() {
        let sim_model::DecoderLayerCache::Attention(cache) = layer else {
            return Err("Llama layer produced a non-attention cache".into());
        };
        assert_eq!(cache.tokens(), 3);
        assert_eq!(cache.keys().descriptor().shape(), [2, 3, 1, 2]);
    }
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn qwen3_query_key_norm_is_per_head_pre_rope_checkpoint_backed_and_cache_exact()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution_context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let mut configuration = configuration(DecoderArchitecture::Llama);
    configuration.query_key_norm = true;
    configuration.layer_kinds = vec![DecoderLayerKind::FullAttention];
    configuration.rope.theta = 5_000_000.0;
    configuration.rope.interleaved_sections = vec![1];
    let admitted_weights = weights(&backend, &configuration, &execution_context)?;
    let model = NativeDecoderTextEncoder::new(configuration.clone(), admitted_weights)?;

    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &execution_context)?;
    let full = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: Some(&[0, 1]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let first_token = i64_tensor(&backend, &[1, 1], &[1], &execution_context)?;
    let first = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &first_token,
            attention_mask: None,
            positions: Some(&[0]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let second_token = i64_tensor(&backend, &[1, 1], &[2], &execution_context)?;
    let second = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &second_token,
            attention_mask: None,
            positions: Some(&[1]),
            cache: Some(first.cache()),
            capture_layer: None,
        },
        &execution_context,
    )?;
    let full_logits = tensor_to_f32(&backend, full.logits(), &execution_context)?;
    let second_logits = tensor_to_f32(&backend, second.logits(), &execution_context)?;
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(
                full_logits
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            )
        ),
        "e2f7a9dc822b118de4e2b20f5db96609c9d4bb0ab1d7c557fef3ba3b76f3d0f1"
    );
    for (expected, actual) in full_logits[8..].iter().zip(second_logits.iter()) {
        assert!((expected - actual).abs() < 1.0e-5, "{expected} != {actual}");
    }

    let mut changed_weights = weights(&backend, &configuration, &execution_context)?;
    let DecoderAttentionWeights::DotProduct {
        query_norm_weight, ..
    } = &mut changed_weights.layers[0].attention
    else {
        return Err("expected dot-product weights".into());
    };
    *query_norm_weight = Some(tensor(&backend, &[2], &[1.5, 0.5], &execution_context)?);
    let changed = NativeDecoderTextEncoder::new(configuration.clone(), changed_weights)?;
    let changed_output = changed.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: Some(&[0, 1]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let baseline_values = tensor_to_f32(&backend, full.logits(), &execution_context)?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let changed_values = tensor_to_f32(&backend, changed_output.logits(), &execution_context)?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_ne!(baseline_values, changed_values);
    assert_ne!(
        model.semantic_state_digest(&cancellation)?,
        changed.semantic_state_digest(&cancellation)?
    );

    let mut missing_weights = weights(&backend, &configuration, &execution_context)?;
    let DecoderAttentionWeights::DotProduct {
        key_norm_weight, ..
    } = &mut missing_weights.layers[0].attention
    else {
        return Err("expected dot-product weights".into());
    };
    *key_norm_weight = None;
    assert!(matches!(
        NativeDecoderTextEncoder::new(configuration, missing_weights),
        Err(DecoderTextError::InvalidConfiguration(
            "query/key normalization weights must exactly match the decoder profile"
        ))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 64 * 1024 * 1024)?;
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: Some(&[0, 1]),
                    cache: None,
                    capture_layer: None,
                },
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma3_alternating_rope_norm_scaling_cache_and_generation_are_exact()
-> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/gemma3_decoder/manifest.json"
    ))?;
    assert_eq!(
        manifest["task_id"],
        "comfy-parity-native-gemma3-decoder-exactness-foundation"
    );
    assert_eq!(manifest["source"]["sha256"], LLAMA_SOURCE_SHA256);
    assert_eq!(manifest["source_snapshot"]["version"], "0.27.1");
    assert_eq!(manifest["source_snapshot"]["file_count"], 949);
    assert_eq!(manifest["claims"]["licensed_checkpoint_included"], false);
    assert_eq!(manifest["claims"]["runtime_oracle_observed"], false);

    let four_b = gemma3_decoder_configuration(Gemma3DecoderProfile::FourBVision);
    four_b.validate()?;
    assert_eq!(four_b.vocabulary_size, 262_208);
    assert_eq!(four_b.hidden_size, 2560);
    assert_eq!(four_b.feed_forward_size, 10_240);
    assert_eq!(four_b.layer_kinds.len(), 34);
    assert_eq!(four_b.layer_kinds[5], DecoderLayerKind::FullAttention);
    assert_eq!(four_b.layer_kinds[33], DecoderLayerKind::SlidingAttention);
    let twelve_b = gemma3_decoder_configuration(Gemma3DecoderProfile::TwelveB);
    twelve_b.validate()?;
    assert_eq!(twelve_b.hidden_size, 3840);
    assert_eq!(twelve_b.feed_forward_size, 15_360);
    assert_eq!(twelve_b.layer_kinds.len(), 48);
    assert_eq!(twelve_b.attention_heads, 16);
    assert_eq!(twelve_b.key_value_heads, 8);

    let cancellation = CancellationToken::default();
    let global_rope = precompute_rope(&[0, 8], &four_b.rope, &cancellation)?;
    let local_rope = precompute_rope(
        &[0, 8],
        &four_b
            .gemma3
            .as_ref()
            .ok_or("Gemma3 profile missing")?
            .local_rope,
        &cancellation,
    )?;
    assert_ne!(global_rope, local_rope);
    assert!((global_rope[128][0] - 1.0_f32.cos()).abs() < 1.0e-6);
    assert!((global_rope[128][1] - 1.0_f32.sin()).abs() < 1.0e-6);
    assert!((local_rope[128][0] - 8.0_f32.cos()).abs() < 1.0e-6);
    assert!((local_rope[128][1] - 8.0_f32.sin()).abs() < 1.0e-6);

    let (backend, authority) = backend()?;
    let execution_context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let reduced = reduced_gemma3_configuration();
    reduced.validate()?;
    let admitted_weights = weights(&backend, &reduced, &execution_context)?;
    let model = NativeDecoderTextEncoder::new(reduced.clone(), admitted_weights)?;
    let tokens = i64_tensor(&backend, &[1, 3], &[1, 2, 3], &execution_context)?;
    let full = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: Some(&[0, 1, 2]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let first_token = i64_tensor(&backend, &[1, 1], &[1], &execution_context)?;
    let first = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &first_token,
            attention_mask: None,
            positions: Some(&[0]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let second_token = i64_tensor(&backend, &[1, 1], &[2], &execution_context)?;
    let second = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &second_token,
            attention_mask: None,
            positions: Some(&[1]),
            cache: Some(first.cache()),
            capture_layer: None,
        },
        &execution_context,
    )?;
    let third_token = i64_tensor(&backend, &[1, 1], &[3], &execution_context)?;
    let third = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &third_token,
            attention_mask: None,
            positions: Some(&[2]),
            cache: Some(second.cache()),
            capture_layer: None,
        },
        &execution_context,
    )?;
    let full_logits = tensor_to_f32(&backend, full.logits(), &execution_context)?;
    let third_logits = tensor_to_f32(&backend, third.logits(), &execution_context)?;
    for (expected, actual) in full_logits[16..].iter().zip(third_logits.iter()) {
        assert!((expected - actual).abs() < 1.0e-5, "{expected} != {actual}");
    }
    let full_digest = format!(
        "{:x}",
        Sha256::digest(
            full_logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        manifest["reduced_fixture"]["full_logits_sha256"],
        "30447eeb298bdaa2edcf48e1b2407903f2da625da3e2bdf0313acc0bc1b046a8"
    );
    assert_eq!(
        full_digest,
        "30447eeb298bdaa2edcf48e1b2407903f2da625da3e2bdf0313acc0bc1b046a8"
    );
    drop(full_logits);
    drop(third_logits);
    for layer in third.cache().layers().iter().flatten() {
        let sim_model::DecoderLayerCache::Attention(cache) = layer else {
            return Err("Gemma3 produced a non-attention cache".into());
        };
        assert_eq!(cache.tokens(), 3);
    }

    let transaction = generation_transaction()?;
    let original_checkpoint = transaction.checkpoint();
    let generation_configuration = DecoderGenerationConfiguration {
        maximum_new_tokens: 2,
        temperature_bits: 0.8_f32.to_bits(),
        top_k: Some(4),
        top_p_bits: Some(0.9_f32.to_bits()),
        minimum_p_bits: Some(0.05_f32.to_bits()),
        repetition_penalty_bits: 1.0_f32.to_bits(),
        presence_penalty_bits: 0.0_f32.to_bits(),
    };
    let generated = model.generate(
        &backend,
        &tokens,
        &generation_configuration,
        &transaction,
        &execution_context,
    )?;
    let replayed = model.generate(
        &backend,
        &tokens,
        &generation_configuration,
        &transaction,
        &execution_context,
    )?;
    assert_eq!(generated.tokens, replayed.tokens);
    assert_eq!(
        generated.transaction.checkpoint(),
        replayed.transaction.checkpoint()
    );
    assert_eq!(transaction.checkpoint(), original_checkpoint);

    let mut changed_weights = weights(&backend, &reduced, &execution_context)?;
    changed_weights.layers[5].post_feed_forward_norm_weight =
        Some(filled(&backend, &[4], 0.27, &execution_context)?);
    let changed_model = NativeDecoderTextEncoder::new(reduced.clone(), changed_weights)?;
    assert_ne!(
        model.semantic_state_digest(&cancellation)?,
        changed_model.semantic_state_digest(&cancellation)?
    );

    let mut invalid_rope = reduced.clone();
    invalid_rope
        .gemma3
        .as_mut()
        .ok_or("Gemma3 profile missing")?
        .local_rope
        .theta = 1_000_000.0;
    assert!(matches!(
        invalid_rope.validate(),
        Err(DecoderTextError::InvalidConfiguration(
            "Gemma3 requires the exact alternating local/global attention, RoPE, normalization, scaling, tied-head, and stop-token profile"
        ))
    ));
    let mut missing_post_norm = weights(&backend, &reduced, &execution_context)?;
    missing_post_norm.layers[0].post_attention_norm_weight = None;
    assert!(matches!(
        NativeDecoderTextEncoder::new(reduced, missing_post_norm),
        Err(DecoderTextError::InvalidConfiguration(
            "Gemma layers require post-attention and post-feed-forward normalization weights"
        ))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 64 * 1024 * 1024)?;
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: Some(&[0, 1, 2]),
                    cache: None,
                    capture_layer: None,
                },
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    let constrained_context = context(&authority, &cancellation, 4)?;
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: Some(&[0, 1, 2]),
                    cache: None,
                    capture_layer: None,
                },
                &constrained_context,
            )
            .is_err()
    );
    assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
    assert_eq!(execution_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn gemma4_global_shared_per_layer_cache_and_generation_are_exact() -> Result<(), Box<dyn Error>> {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/text_generation/gemma_multimodal/gemma4_decoder/manifest.json"
    ))?;
    assert_eq!(
        manifest["task_id"],
        "comfy-parity-native-gemma4-decoder-exactness-foundation"
    );
    assert_eq!(manifest["source"]["sha256"], GEMMA4_SOURCE_SHA256);
    assert_eq!(manifest["source_snapshot"]["version"], "0.27.1");
    assert_eq!(manifest["source_snapshot"]["file_count"], 949);
    assert_eq!(manifest["claims"]["licensed_checkpoint_included"], false);
    assert_eq!(manifest["claims"]["runtime_oracle_observed"], false);

    let e2b = gemma4_decoder_configuration(Gemma4DecoderProfile::E2B);
    e2b.validate()?;
    assert_eq!(e2b.hidden_size, 1536);
    assert_eq!(e2b.feed_forward_size, 6144);
    assert_eq!(e2b.layer_kinds.len(), 35);
    let e2b_profile = e2b.gemma4.as_ref().ok_or("E2B Gemma4 profile")?;
    assert_eq!(e2b_profile.global_head_dimension, 512);
    assert_eq!(e2b_profile.hidden_size_per_layer_input, 256);
    assert_eq!(e2b_profile.shared_key_value_layers, 20);
    assert!(e2b_profile.double_wide_mlp);
    assert_eq!(e2b.layer_kinds[4], DecoderLayerKind::FullAttention);
    let e4b = gemma4_decoder_configuration(Gemma4DecoderProfile::E4B);
    e4b.validate()?;
    assert_eq!(e4b.hidden_size, 2560);
    assert_eq!(e4b.layer_kinds.len(), 42);
    assert_eq!(
        e4b.gemma4
            .as_ref()
            .ok_or("E4B Gemma4 profile")?
            .shared_key_value_layers,
        18
    );
    let thirty_one_b = gemma4_decoder_configuration(Gemma4DecoderProfile::ThirtyOneB);
    thirty_one_b.validate()?;
    assert_eq!(thirty_one_b.hidden_size, 5376);
    assert_eq!(thirty_one_b.feed_forward_size, 21_504);
    assert_eq!(thirty_one_b.layer_kinds.len(), 60);
    let thirty_one_b_profile = thirty_one_b.gemma4.as_ref().ok_or("31B Gemma4 profile")?;
    assert_eq!(thirty_one_b_profile.hidden_size_per_layer_input, 0);
    assert_eq!(thirty_one_b_profile.shared_key_value_layers, 0);

    let cancellation = CancellationToken::default();
    let (backend, authority) = backend()?;
    let execution_context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let reduced = reduced_gemma4_configuration();
    reduced.validate()?;
    let admitted_weights = gemma4_weights(&backend, &reduced, &execution_context)?;
    let model = NativeDecoderTextEncoder::new(reduced.clone(), admitted_weights)?;
    let tokens = i64_tensor(&backend, &[1, 3], &[1, 2, 3], &execution_context)?;
    let full = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: Some(&[0, 1, 2]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let first_token = i64_tensor(&backend, &[1, 1], &[1], &execution_context)?;
    let first = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &first_token,
            attention_mask: None,
            positions: Some(&[0]),
            cache: None,
            capture_layer: None,
        },
        &execution_context,
    )?;
    let second_token = i64_tensor(&backend, &[1, 1], &[2], &execution_context)?;
    let second = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &second_token,
            attention_mask: None,
            positions: Some(&[1]),
            cache: Some(first.cache()),
            capture_layer: None,
        },
        &execution_context,
    )?;
    let third_token = i64_tensor(&backend, &[1, 1], &[3], &execution_context)?;
    let third = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &third_token,
            attention_mask: None,
            positions: Some(&[2]),
            cache: Some(second.cache()),
            capture_layer: None,
        },
        &execution_context,
    )?;
    let full_logits = tensor_to_f32(&backend, full.logits(), &execution_context)?;
    let third_logits = tensor_to_f32(&backend, third.logits(), &execution_context)?;
    for (expected, actual) in full_logits[16..].iter().zip(third_logits.iter()) {
        assert!((expected - actual).abs() < 1.0e-5, "{expected} != {actual}");
    }
    assert!(full_logits.iter().all(|value| value.abs() <= 30.0));
    let full_digest = format!(
        "{:x}",
        Sha256::digest(
            full_logits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    assert_eq!(
        full_digest,
        "10080fb49e529b5b7f341c47c112dba564b067c6f908d7604c200c3b78d02e16"
    );
    drop(full_logits);
    drop(third_logits);
    let sim_model::DecoderLayerCache::Attention(sliding_cache) = third.cache().layers()[0]
        .as_ref()
        .ok_or("Gemma4 sliding cache")?
    else {
        return Err("Gemma4 sliding layer has wrong cache kind".into());
    };
    assert_eq!(sliding_cache.head_dimension(), 2);
    let sim_model::DecoderLayerCache::Attention(global_cache) = third.cache().layers()[4]
        .as_ref()
        .ok_or("Gemma4 global cache")?
    else {
        return Err("Gemma4 global layer has wrong cache kind".into());
    };
    assert_eq!(global_cache.head_dimension(), 4);
    assert!(third.cache().layers()[5..].iter().all(Option::is_none));

    let embeddings = model.embed_tokens(&backend, &tokens, &execution_context)?;
    let prepared_with_ids = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: Some(&[1, 2, 3]),
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&[0, 1, 2]),
            causal_positions: &[0, 1, 2],
            cache: None,
            capture_layer: None,
            deepstack: None,
        },
        &execution_context,
    )?;
    let prepared_without_ids = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&[0, 1, 2]),
            causal_positions: &[0, 1, 2],
            cache: None,
            capture_layer: None,
            deepstack: None,
        },
        &execution_context,
    )?;
    assert_ne!(
        tensor_to_f32(&backend, prepared_with_ids.logits(), &execution_context)?.as_ref(),
        tensor_to_f32(&backend, prepared_without_ids.logits(), &execution_context)?.as_ref()
    );

    let transaction = generation_transaction()?;
    let checkpoint = transaction.checkpoint();
    let generation_configuration = DecoderGenerationConfiguration {
        maximum_new_tokens: 2,
        temperature_bits: 0.8_f32.to_bits(),
        top_k: Some(4),
        top_p_bits: Some(0.9_f32.to_bits()),
        minimum_p_bits: Some(0.05_f32.to_bits()),
        repetition_penalty_bits: 1.0_f32.to_bits(),
        presence_penalty_bits: 0.0_f32.to_bits(),
    };
    let prompt = DecoderPreparedGenerationPrompt {
        initial_input_ids: Some(&[1, 2, 3]),
        embeddings: &embeddings,
        sampling_history: &[1, 2, 3],
        attention_mask: None,
        rope_positions: DecoderRopePositions::Scalar(&[0, 1, 2]),
        causal_positions: &[0, 1, 2],
        deepstack: None,
    };
    let generated = model.generate_prepared(
        &backend,
        prompt,
        &generation_configuration,
        &transaction,
        &execution_context,
    )?;
    let replayed = model.generate_prepared(
        &backend,
        prompt,
        &generation_configuration,
        &transaction,
        &execution_context,
    )?;
    assert_eq!(generated.generated_tokens, replayed.generated_tokens);
    assert_eq!(
        generated.transaction.checkpoint(),
        replayed.transaction.checkpoint()
    );
    assert_eq!(transaction.checkpoint(), checkpoint);

    let mut changed_weights = gemma4_weights(&backend, &reduced, &execution_context)?;
    changed_weights.layers[0]
        .gemma4_layer_input
        .as_mut()
        .ok_or("Gemma4 layer input")?
        .layer_scalar = tensor(&backend, &[1], &[0.91], &execution_context)?;
    let changed = NativeDecoderTextEncoder::new(reduced.clone(), changed_weights)?;
    assert_ne!(
        model.semantic_state_digest(&cancellation)?,
        changed.semantic_state_digest(&cancellation)?
    );
    let mut invalid_profile = reduced.clone();
    invalid_profile
        .gemma4
        .as_mut()
        .ok_or("Gemma4 profile")?
        .source_profile = Some(Gemma4DecoderProfile::E2B);
    assert!(invalid_profile.validate().is_err());
    let mut missing_per_layer = gemma4_weights(&backend, &reduced, &execution_context)?;
    missing_per_layer.gemma4_per_layer = None;
    assert!(NativeDecoderTextEncoder::new(reduced, missing_per_layer).is_err());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 64 * 1024 * 1024)?;
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: Some(&[0, 1, 2]),
                    cache: None,
                    capture_layer: None,
                },
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    let constrained_context = context(&authority, &cancellation, 4)?;
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: Some(&[0, 1, 2]),
                    cache: None,
                    capture_layer: None,
                },
                &constrained_context,
            )
            .is_err()
    );
    assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
    assert_eq!(execution_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn rope_regular_linear_yarn_partial_and_multidimensional_variants_execute()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let base = DecoderRopeConfiguration {
        theta: 10_000.0,
        rotary_dimension: 4,
        interleaved_sections: Vec::new(),
        scaling: RopeScaling::None,
    };
    let regular = precompute_rope(&[0, 1, 2], &base, &cancellation)?;
    assert_eq!(regular.len(), 6);
    assert_eq!(regular[0], [1.0, 0.0]);
    let linear = precompute_rope(
        &[0, 1, 2],
        &DecoderRopeConfiguration {
            scaling: RopeScaling::Linear { factor: 2.0 },
            ..base.clone()
        },
        &cancellation,
    )?;
    let yarn = precompute_rope(
        &[0, 1, 2],
        &DecoderRopeConfiguration {
            scaling: RopeScaling::Yarn {
                factor: 2.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
            ..base.clone()
        },
        &cancellation,
    )?;
    assert_ne!(regular, linear);
    assert_ne!(regular, yarn);
    let rotated = apply_rope(
        &[1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5],
        1,
        1,
        1,
        8,
        &[2],
        &base,
        &cancellation,
    )?;
    assert_eq!(&rotated[4..], &[0.5, -0.5, 1.5, -1.5]);
    let multidimensional = precompute_multidimensional_rope(
        &[vec![0, 1], vec![2, 3]],
        &DecoderRopeConfiguration {
            interleaved_sections: vec![1, 1],
            ..base
        },
        &cancellation,
    )?;
    assert_eq!(multidimensional.len(), 4);
    Ok(())
}

#[test]
fn gpt_oss_sinks_yarn_router_and_experts_execute() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let routes = gpt_oss_top_k_route(&[1.0, 3.0, 2.0, -1.0, 2.0, 0.5], 2, 3, 2, &cancellation)?;
    assert_eq!(routes[0][0].0, 1);
    assert_eq!(routes[1][0].0, 1);
    let output = gpt_oss_moe(
        &[0.5, -0.25, 0.75, 0.125],
        &[1.0, 3.0, 2.0, -1.0],
        &[0.2; 8],
        &[0.3; 8],
        &[0.4; 8],
        2,
        2,
        2,
        2,
        2,
        &cancellation,
    )?;
    assert_eq!(output.len(), 4);
    assert!(output.iter().all(|value| value.is_finite()));

    let (backend, authority, model, cancellation) = model(DecoderArchitecture::GptOss)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let output = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: None,
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    assert!(
        tensor_to_f32(&backend, output.logits(), &context)?
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn qwen35_linear_recurrent_convolution_and_hybrid_graph_execute() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let query = vec![0.5, -0.25, 0.75, 0.125, -0.5, 0.4, 0.2, 0.1];
    let key = vec![0.3; 8];
    let value = vec![0.8; 8];
    let (first, state) =
        qwen35_chunk_gated_delta_rule(&query, &key, &value, &[], 1, 2, 4, &cancellation)?;
    let (convolved, convolution_state) =
        qwen35_causal_conv1d_update(&first, &[], 1, 2, 4, &[0.25, 0.5, 0.25], &cancellation)?;
    assert_eq!(state.len(), 4);
    assert_eq!(convolved.len(), 8);
    assert_eq!(convolution_state.len(), 8);
    let (exact_delta, exact_state) = qwen35_chunk_gated_delta_rule_exact(
        &[1.0, 0.0],
        &[1.0, 0.0],
        &[2.0],
        &[0.0],
        &[1.0],
        &[],
        1,
        1,
        1,
        2,
        1,
        &cancellation,
    )?;
    assert_eq!(exact_state, vec![2.0, 0.0]);
    assert!((exact_delta[0] - 2.0_f32.sqrt()).abs() < 1.0e-6);
    let (exact_convolution, exact_convolution_state) = qwen35_causal_conv1d_update_exact(
        &[2.0],
        &[1.0, 0.0],
        &[1.0, 2.0, 3.0],
        None,
        1,
        1,
        1,
        3,
        &cancellation,
    )?;
    assert!((exact_convolution[0] - 7.0 / (1.0 + (-7.0_f32).exp())).abs() < 1.0e-6);
    assert_eq!(exact_convolution_state, vec![0.0, 2.0]);
    let embedded = qwen35_vision_patch_embed(
        &[0.1, 0.2, 0.3, 0.4],
        &[0.5, 0.0, 0.0, 0.5],
        &[0.1, -0.1],
        2,
        2,
        2,
        &cancellation,
    )?;
    assert_eq!(embedded.len(), 4);
    let merged = qwen35_vision_patch_merge(
        &embedded,
        2,
        2,
        1,
        &[1.0, 0.0, 0.0, 1.0],
        None,
        &[1.0, 0.0, 0.0, 1.0],
        None,
        2,
        &cancellation,
    )?;
    assert_eq!(merged.len(), 4);

    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Qwen35)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let output = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: None,
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    assert!(matches!(
        output.cache().layers().first().and_then(Option::as_ref),
        Some(sim_model::DecoderLayerCache::Linear(_))
    ));
    let next_token = i64_tensor(&backend, &[1, 1], &[3], &context)?;
    let continued = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &next_token,
            attention_mask: None,
            positions: Some(&[2]),
            cache: Some(output.cache()),
            capture_layer: None,
        },
        &context,
    )?;
    let Some(sim_model::DecoderLayerCache::Linear(linear_cache)) =
        continued.cache().layers().first().and_then(Option::as_ref)
    else {
        return Err("Qwen3.5 recurrent layer cache is missing".into());
    };
    assert_eq!(
        linear_cache.recurrent_state.descriptor().shape(),
        [1, 2, 2, 2]
    );
    assert_eq!(
        linear_cache.convolution_state.descriptor().shape(),
        [1, 8, 2]
    );
    assert_eq!(linear_cache.step_index, 3);
    let full_tokens = i64_tensor(&backend, &[1, 3], &[1, 2, 3], &context)?;
    let full = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &full_tokens,
            attention_mask: None,
            positions: Some(&[0, 1, 2]),
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    let full_logits = tensor_to_f32(&backend, full.logits(), &context)?;
    let continued_logits = tensor_to_f32(&backend, continued.logits(), &context)?;
    for (expected, actual) in full_logits[16..].iter().zip(continued_logits.iter()) {
        assert!((expected - actual).abs() < 1.0e-5, "{expected} != {actual}");
    }
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(
                full_logits
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            )
        ),
        "02f4e8afd51de3d86100655bdb69f8c9b673fe42dfc242d47840b74f408ad53c"
    );
    let mut changed_weights = weights(&backend, model.configuration(), &context)?;
    let DecoderAttentionWeights::Qwen35Linear(linear_weights) =
        &mut changed_weights.layers[0].attention
    else {
        return Err("Qwen3.5 linear weights are missing".into());
    };
    linear_weights.log_decay = tensor(&backend, &[2], &[0.8, -0.7], &context)?;
    let changed = NativeDecoderTextEncoder::new(model.configuration().clone(), changed_weights)?;
    assert_ne!(
        model.semantic_state_digest(&cancellation)?,
        changed.semantic_state_digest(&cancellation)?
    );
    let changed_output = changed.forward(
        &backend,
        DecoderTextRequest {
            tokens: &full_tokens,
            attention_mask: None,
            positions: Some(&[0, 1, 2]),
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    assert_ne!(
        full_logits.iter().copied().collect::<Vec<_>>(),
        tensor_to_f32(&backend, changed_output.logits(), &context)?
            .iter()
            .copied()
            .collect::<Vec<_>>()
    );

    let mut malformed_weights = weights(&backend, model.configuration(), &context)?;
    let DecoderAttentionWeights::Qwen35Linear(linear_weights) =
        &mut malformed_weights.layers[0].attention
    else {
        return Err("Qwen3.5 linear weights are missing".into());
    };
    linear_weights.convolution_weight = filled(&backend, &[8, 2], 0.2, &context)?;
    assert!(matches!(
        NativeDecoderTextEncoder::new(model.configuration().clone(), malformed_weights),
        Err(DecoderTextError::InvalidConfiguration(
            "parameter shape does not match the decoder profile"
        ))
    ));
    Ok(())
}

#[test]
fn gemma4_scaled_double_norm_softcap_vision_and_audio_equations_execute()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    assert_eq!(
        gemma4_vision_rope(2, 3, 8, 10_000.0, &cancellation)?.len(),
        24
    );
    assert_eq!(
        gemma4_audio_relative_positions(2, 3, 2)?,
        vec![2, 3, 4, 1, 2, 3]
    );
    assert_eq!(
        gemma4_clipped_linear(
            &[-2.0, 2.0],
            &[1.0, 1.0],
            Some(&[0.25]),
            1,
            2,
            1,
            [-1.0, 1.0],
            [-0.5, 0.5],
            &cancellation,
        )?,
        vec![0.25]
    );
    let patches = gemma4_vision_patch_embed(
        &[0.0, 0.5, 1.0, 0.25],
        &[[0, 0], [-1, -1]],
        &[0.5, 0.5, -0.5, 0.5],
        &[0.1, 0.2, 0.3, 0.4],
        1,
        2,
        2,
        2,
        1,
        &cancellation,
    )?;
    assert_eq!(patches.len(), 4);
    let (audio, audio_shape) = gemma4_audio_conv2d_subsample(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.0, 0.0, 0.0, 1.0],
        1,
        1,
        2,
        2,
        1,
        2,
        1,
        0,
        &cancellation,
    )?;
    assert_eq!(audio_shape, [1, 1]);
    assert_eq!(audio, vec![5.0]);
    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Gemma)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let output = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: None,
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    assert!(
        tensor_to_f32(&backend, output.logits(), &context)?
            .iter()
            .all(|value| value.abs() <= 4.0)
    );
    Ok(())
}

fn generation_transaction() -> Result<sim_tensor::RngTransaction, Box<dyn Error>> {
    let address = RngStreamAddress::new(
        "decoder-workflow",
        "attempt-1",
        "decoder-node",
        0,
        "generation",
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

fn generation_tokenizer() -> Result<NativePromptTokenizer, Box<dyn Error>> {
    let vocabulary = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
    ))?;
    let merges = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
    ))?;
    Ok(NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.decoder.generation.tokenizer")?,
            &vocabulary,
            &merges,
        )?),
        TokenizerConfiguration {
            maximum_length: 8,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: false,
            pad_left: false,
            start_token: Some(1),
            end_token: Some(2),
            pad_token: 2,
            maximum_word_length: 32,
            disable_weights: true,
            embedding_width: None,
        },
        BTreeMap::new(),
    )?)
}

#[test]
fn caller_addressed_generation_is_deterministic_and_does_not_mutate_input_transaction()
-> Result<(), Box<dyn Error>> {
    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Llama)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let prompt = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let generation = DecoderGenerationConfiguration {
        maximum_new_tokens: 3,
        temperature_bits: 0.8_f32.to_bits(),
        top_k: Some(4),
        top_p_bits: Some(0.9_f32.to_bits()),
        minimum_p_bits: Some(0.05_f32.to_bits()),
        repetition_penalty_bits: 1.1_f32.to_bits(),
        presence_penalty_bits: 0.1_f32.to_bits(),
    };
    let transaction = generation_transaction()?;
    let first = model.generate(&backend, &prompt, &generation, &transaction, &context)?;
    let second = model.generate(&backend, &prompt, &generation, &transaction, &context)?;
    assert_eq!(first.tokens, second.tokens);
    assert!(first.tokens.len() >= 3 && first.tokens.len() <= 5);
    Ok(())
}

#[test]
fn retained_decoder_clip_generation_decodes_only_new_tokens_and_reuses_backing()
-> Result<(), Box<dyn Error>> {
    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Llama)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokenizer = Arc::new(generation_tokenizer()?);
    let model = Arc::new(model);
    let transaction = generation_transaction()?;
    let original_checkpoint = transaction.checkpoint();
    let request = NativeTextGenerationRequest {
        formatted_prompt: "",
        maximum_new_tokens: 1,
        do_sample: false,
        temperature_bits: 1.0_f32.to_bits(),
        top_k: 50,
        top_p_bits: 1.0_f32.to_bits(),
        minimum_p_bits: 0.0_f32.to_bits(),
        repetition_penalty_bits: 1.0_f32.to_bits(),
        presence_penalty_bits: 0.0_f32.to_bits(),
    };
    for invalid_limit in [0, 32_769] {
        let mut invalid = request.clone();
        invalid.maximum_new_tokens = invalid_limit;
        assert!(matches!(
            model.generate_text(&tokenizer, &backend, invalid, &transaction, &context),
            Err(DecoderTextError::InvalidInput(
                "maximum new tokens must be between 1 and 32768"
            ))
        ));
        assert_eq!(transaction.checkpoint(), original_checkpoint);
    }
    let outcome = model.generate_text(&tokenizer, &backend, request, &transaction, &context)?;
    assert_eq!(outcome.generated_tokens.len(), 1);
    assert_eq!(transaction.checkpoint(), original_checkpoint);

    let first = NativeModelPayload::decoder_clip(tokenizer.clone(), model.clone())?;
    let second = NativeModelPayload::decoder_clip(tokenizer.clone(), model.clone())?;
    assert_eq!(first.identity(), second.identity());
    assert_eq!(first.resident_parts()?, second.resident_parts()?);
    let shared_tensor_model = Arc::new(model.as_ref().clone());
    let shared_tensor_payload = NativeModelPayload::decoder_clip(tokenizer, shared_tensor_model)?;
    assert_eq!(first.identity(), shared_tensor_payload.identity());
    assert!(!first.resident_parts()?.tensor_allocations().is_empty());
    assert_eq!(
        first.resident_parts()?.tensor_allocations(),
        shared_tensor_payload.resident_parts()?.tensor_allocations()
    );
    assert!(first.decoder_clip_resource().is_some());
    first.validate()?;
    Ok(())
}

#[test]
fn target_shape_cache_generation_cancellation_and_oom_fail_typed_and_atomic()
-> Result<(), Box<dyn Error>> {
    let mut invalid = configuration(DecoderArchitecture::Llama);
    invalid.dtype = DType::F16;
    assert!(matches!(
        invalid.validate(),
        Err(DecoderTextError::UnsupportedTarget { .. })
    ));
    let cancellation = CancellationToken::default();
    assert!(
        apply_rope(
            &[1.0],
            1,
            1,
            1,
            2,
            &[0],
            &DecoderRopeConfiguration {
                theta: 10_000.0,
                rotary_dimension: 2,
                interleaved_sections: Vec::new(),
                scaling: RopeScaling::None,
            },
            &cancellation,
        )
        .is_err()
    );
    assert!(gpt_oss_top_k_route(&[f32::NAN], 1, 1, 1, &cancellation).is_err());
    assert!(qwen35_causal_conv1d_update(&[1.0], &[], 1, 1, 1, &[], &cancellation).is_err());
    assert!(gemma4_vision_rope(1, 1, 6, 10_000.0, &cancellation).is_err());

    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Llama)?;
    let setup = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &setup)?;
    let first = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: None,
            cache: None,
            capture_layer: None,
        },
        &setup,
    )?;
    let original_cache = first.cache().clone();
    let original_cache_storage = attention_cache_storage_ids(&original_cache)?;
    cancellation.cancel();
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: None,
                    cache: Some(&original_cache),
                    capture_layer: None,
                },
                &setup,
            )
            .is_err()
    );
    assert_eq!(
        attention_cache_storage_ids(first.cache())?,
        original_cache_storage
    );
    drop(setup);

    let fresh_cancellation = CancellationToken::default();
    let insufficient = context(&authority, &fresh_cancellation, 64)?;
    assert!(
        model
            .forward(
                &backend,
                DecoderTextRequest {
                    tokens: &tokens,
                    attention_mask: None,
                    positions: None,
                    cache: Some(&original_cache),
                    capture_layer: None,
                },
                &insufficient,
            )
            .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    assert_eq!(
        attention_cache_storage_ids(first.cache())?,
        original_cache_storage
    );
    Ok(())
}

#[test]
fn prepared_prefill_shares_generation_rng_cache_and_multidimensional_rope()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let mut prepared_configuration = configuration(DecoderArchitecture::Llama);
    prepared_configuration.stop_tokens.clear();
    let model = NativeDecoderTextEncoder::new(
        prepared_configuration.clone(),
        weights(&backend, &prepared_configuration, &context)?,
    )?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let embeddings = model.embed_tokens(&backend, &tokens, &context)?;
    let scalar_positions = [0_usize, 1];
    let prepared = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&scalar_positions),
            causal_positions: &scalar_positions,
            cache: None,
            capture_layer: None,
            deepstack: None,
        },
        &context,
    )?;
    let numeric = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: Some(&scalar_positions),
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    let prepared_logits = tensor_to_f32(&backend, prepared.logits(), &context)?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let numeric_logits = tensor_to_f32(&backend, numeric.logits(), &context)?
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(prepared_logits, numeric_logits);

    let transaction = generation_transaction()?;
    let original_checkpoint = transaction.checkpoint();
    let attention_mask = tensor(&backend, &[1, 2], &[1.0, 0.0], &context)?;
    let generation = model.generate_prepared(
        &backend,
        DecoderPreparedGenerationPrompt {
            initial_input_ids: None,
            embeddings: &embeddings,
            sampling_history: &[1, 2],
            attention_mask: Some(&attention_mask),
            rope_positions: DecoderRopePositions::Scalar(&scalar_positions),
            causal_positions: &scalar_positions,
            deepstack: None,
        },
        &DecoderGenerationConfiguration {
            maximum_new_tokens: 2,
            temperature_bits: 0.0_f32.to_bits(),
            top_k: None,
            top_p_bits: None,
            minimum_p_bits: None,
            repetition_penalty_bits: 1.0_f32.to_bits(),
            presence_penalty_bits: 0.0_f32.to_bits(),
        },
        &transaction,
        &context,
    )?;
    assert_eq!(generation.generated_tokens.len(), 2);
    assert_eq!(transaction.checkpoint(), original_checkpoint);
    assert_eq!(context.scratch.in_use_bytes(), 0);

    let mut qwen_configuration = configuration(DecoderArchitecture::Llama);
    qwen_configuration.rope.interleaved_sections = vec![1, 0, 0];
    let qwen_model = NativeDecoderTextEncoder::new(
        qwen_configuration.clone(),
        weights(&backend, &qwen_configuration, &context)?,
    )?;
    let qwen_embeddings = qwen_model.embed_tokens(&backend, &tokens, &context)?;
    let axes = vec![vec![0, 1], vec![0, 2], vec![0, 3]];
    let qwen_output = qwen_model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &qwen_embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Multidimensional(&axes),
            causal_positions: &scalar_positions,
            cache: None,
            capture_layer: None,
            deepstack: None,
        },
        &context,
    )?;
    assert_eq!(qwen_output.logits().descriptor().shape(), [1, 2, 8]);
    Ok(())
}

#[test]
fn prepared_deepstack_is_exact_post_layer_prefill_only_and_transactional()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let mut decoder_configuration = configuration(DecoderArchitecture::Llama);
    decoder_configuration.stop_tokens.clear();
    let model = NativeDecoderTextEncoder::new(
        decoder_configuration.clone(),
        weights(&backend, &decoder_configuration, &context)?,
    )?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let embeddings = model.embed_tokens(&backend, &tokens, &context)?;
    let positions = [0_usize, 1];
    let baseline = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            cache: None,
            capture_layer: Some(0),
            deepstack: None,
        },
        &context,
    )?;
    let layer = tensor(&backend, &[1, 4], &[1.0, -2.0, 3.0, -4.0], &context)?;
    let mask = [true, false];
    let layers = [layer];
    let deepstack = DecoderPreparedDeepstack {
        visual_position_mask: &mask,
        layers: &layers,
    };
    let injected = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            cache: None,
            capture_layer: Some(0),
            deepstack: Some(deepstack),
        },
        &context,
    )?;
    let baseline_intermediate = tensor_to_f32(
        &backend,
        baseline.intermediate().ok_or("baseline capture missing")?,
        &context,
    )?;
    let injected_intermediate = tensor_to_f32(
        &backend,
        injected.intermediate().ok_or("deepstack capture missing")?,
        &context,
    )?;
    let differences = injected_intermediate
        .iter()
        .zip(baseline_intermediate.iter())
        .map(|(injected, baseline)| injected - baseline)
        .collect::<Vec<_>>();
    assert_eq!(&differences[..4], &[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(&differences[4..], &[0.0; 4]);

    let second_layer = tensor(&backend, &[1, 4], &[-0.5, 1.5, -2.5, 3.5], &context)?;
    let first_two_layers = [layers[0].clone(), second_layer];
    let before_second = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            cache: None,
            capture_layer: Some(1),
            deepstack: Some(deepstack),
        },
        &context,
    )?;
    let after_second = model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            cache: None,
            capture_layer: Some(1),
            deepstack: Some(DecoderPreparedDeepstack {
                visual_position_mask: &mask,
                layers: &first_two_layers,
            }),
        },
        &context,
    )?;
    let before_second_values = tensor_to_f32(
        &backend,
        before_second
            .intermediate()
            .ok_or("second baseline missing")?,
        &context,
    )?;
    let after_second_values = tensor_to_f32(
        &backend,
        after_second
            .intermediate()
            .ok_or("second capture missing")?,
        &context,
    )?;
    let second_difference = after_second_values
        .iter()
        .zip(before_second_values.iter())
        .map(|(after, before)| after - before)
        .collect::<Vec<_>>();
    assert_eq!(&second_difference[..4], &[-0.5, 1.5, -2.5, 3.5]);
    assert_eq!(&second_difference[4..], &[0.0; 4]);

    let mut three_layer_configuration = decoder_configuration.clone();
    three_layer_configuration
        .layer_kinds
        .push(DecoderLayerKind::FullAttention);
    let three_layer_model = NativeDecoderTextEncoder::new(
        three_layer_configuration.clone(),
        weights(&backend, &three_layer_configuration, &context)?,
    )?;
    let three_layer_embeddings = three_layer_model.embed_tokens(&backend, &tokens, &context)?;
    let third_layer = tensor(&backend, &[1, 4], &[0.25, -0.75, 1.25, -1.75], &context)?;
    let first_three_layers = [
        first_two_layers[0].clone(),
        first_two_layers[1].clone(),
        third_layer,
    ];
    let before_third = three_layer_model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &three_layer_embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            cache: None,
            capture_layer: Some(2),
            deepstack: Some(DecoderPreparedDeepstack {
                visual_position_mask: &mask,
                layers: &first_two_layers,
            }),
        },
        &context,
    )?;
    let after_third = three_layer_model.forward_prepared(
        &backend,
        DecoderPreparedTextRequest {
            initial_input_ids: None,
            embeddings: &three_layer_embeddings,
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            cache: None,
            capture_layer: Some(2),
            deepstack: Some(DecoderPreparedDeepstack {
                visual_position_mask: &mask,
                layers: &first_three_layers,
            }),
        },
        &context,
    )?;
    let before_third_values = tensor_to_f32(
        &backend,
        before_third
            .intermediate()
            .ok_or("third baseline missing")?,
        &context,
    )?;
    let after_third_values = tensor_to_f32(
        &backend,
        after_third.intermediate().ok_or("third capture missing")?,
        &context,
    )?;
    let third_difference = after_third_values
        .iter()
        .zip(before_third_values.iter())
        .map(|(after, before)| after - before)
        .collect::<Vec<_>>();
    assert_eq!(&third_difference[..4], &[0.25, -0.75, 1.25, -1.75]);
    assert_eq!(&third_difference[4..], &[0.0; 4]);

    let transaction = generation_transaction()?;
    let checkpoint = transaction.checkpoint();
    let generated = model.generate_prepared(
        &backend,
        DecoderPreparedGenerationPrompt {
            initial_input_ids: None,
            embeddings: &embeddings,
            sampling_history: &[],
            attention_mask: None,
            rope_positions: DecoderRopePositions::Scalar(&positions),
            causal_positions: &positions,
            deepstack: Some(deepstack),
        },
        &DecoderGenerationConfiguration {
            maximum_new_tokens: 2,
            temperature_bits: 0.0_f32.to_bits(),
            top_k: None,
            top_p_bits: None,
            minimum_p_bits: None,
            repetition_penalty_bits: 1.0_f32.to_bits(),
            presence_penalty_bits: 0.0_f32.to_bits(),
        },
        &transaction,
        &context,
    )?;
    assert_eq!(generated.generated_tokens.len(), 2);
    assert_eq!(transaction.checkpoint(), checkpoint);

    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: Some(baseline.cache()),
                    capture_layer: None,
                    deepstack: Some(deepstack),
                },
                &context,
            )
            .is_err()
    );
    let too_many_layers = first_three_layers.clone();
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(DecoderPreparedDeepstack {
                        visual_position_mask: &mask,
                        layers: &too_many_layers,
                    }),
                },
                &context,
            )
            .is_err()
    );

    let no_visual = [false, false];
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(DecoderPreparedDeepstack {
                        visual_position_mask: &no_visual,
                        layers: &layers,
                    }),
                },
                &context,
            )
            .is_err()
    );
    let short_mask = [true];
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(DecoderPreparedDeepstack {
                        visual_position_mask: &short_mask,
                        layers: &layers,
                    }),
                },
                &context,
            )
            .is_err()
    );
    let wrong_width = tensor(&backend, &[1, 3], &[0.0; 3], &context)?;
    let wrong_layers = [wrong_width];
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(DecoderPreparedDeepstack {
                        visual_position_mask: &mask,
                        layers: &wrong_layers,
                    }),
                },
                &context,
            )
            .is_err()
    );
    let wrong_dtype = i64_tensor(&backend, &[1, 4], &[0, 0, 0, 0], &context)?;
    let wrong_dtype_layers = [wrong_dtype];
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(DecoderPreparedDeepstack {
                        visual_position_mask: &mask,
                        layers: &wrong_dtype_layers,
                    }),
                },
                &context,
            )
            .is_err()
    );
    let foreign_context = ExecutionContext {
        stream: StreamId::new(9),
        scratch: authority.authorize_workspace(64 * 1024 * 1024)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let foreign_stream_layer = tensor(&backend, &[1, 4], &[0.0; 4], &foreign_context)?;
    let foreign_stream_layers = [foreign_stream_layer];
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(DecoderPreparedDeepstack {
                        visual_position_mask: &mask,
                        layers: &foreign_stream_layers,
                    }),
                },
                &context,
            )
            .is_err()
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(64 * 1024 * 1024)?,
        rng_phase: None,
        cancellation: &cancelled,
    };
    assert!(
        model
            .generate_prepared(
                &backend,
                DecoderPreparedGenerationPrompt {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    sampling_history: &[],
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    deepstack: Some(deepstack),
                },
                &DecoderGenerationConfiguration {
                    maximum_new_tokens: 1,
                    temperature_bits: 0.0_f32.to_bits(),
                    top_k: None,
                    top_p_bits: None,
                    minimum_p_bits: None,
                    repetition_penalty_bits: 1.0_f32.to_bits(),
                    presence_penalty_bits: 0.0_f32.to_bits(),
                },
                &transaction,
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(transaction.checkpoint(), checkpoint);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    let constrained = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(8)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: Some(deepstack),
                },
                &constrained,
            )
            .is_err()
    );
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    drop(baseline_intermediate);
    drop(injected_intermediate);
    drop(before_second_values);
    drop(after_second_values);
    drop(before_third_values);
    drop(after_third_values);
    drop(generated);
    drop(after_third);
    drop(before_third);
    drop(three_layer_embeddings);
    drop(first_three_layers);
    drop(after_second);
    drop(before_second);
    drop(first_two_layers);
    drop(injected);
    drop(baseline);
    drop(layers);
    drop(too_many_layers);
    drop(wrong_layers);
    drop(wrong_dtype_layers);
    drop(foreign_stream_layers);
    assert_eq!(foreign_context.scratch.in_use_bytes(), 0);
    drop(embeddings);
    drop(tokens);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn prepared_prefill_rejects_shape_cache_and_cancellation_without_rng_mutation()
-> Result<(), Box<dyn Error>> {
    let (backend, authority, model, cancellation) = model(DecoderArchitecture::Llama)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let embeddings = model.embed_tokens(&backend, &tokens, &context)?;
    let positions = [0_usize, 1];
    let first = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: Some(&positions),
            cache: None,
            capture_layer: None,
        },
        &context,
    )?;
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &embeddings,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: Some(first.cache()),
                    capture_layer: None,
                    deepstack: None,
                },
                &context,
            )
            .is_err()
    );
    let malformed = tensor(&backend, &[1, 2, 3], &[0.0; 6], &context)?;
    assert!(
        model
            .forward_prepared(
                &backend,
                DecoderPreparedTextRequest {
                    initial_input_ids: None,
                    embeddings: &malformed,
                    attention_mask: None,
                    rope_positions: DecoderRopePositions::Scalar(&positions),
                    causal_positions: &positions,
                    cache: None,
                    capture_layer: None,
                    deepstack: None,
                },
                &context,
            )
            .is_err()
    );
    cancellation.cancel();
    assert!(model.embed_tokens(&backend, &tokens, &context).is_err());
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

fn attention_cache_storage_ids(
    cache: &sim_model::DecoderKvState,
) -> Result<Vec<(sim_tensor::StorageId, sim_tensor::StorageId)>, Box<dyn Error>> {
    cache
        .layers()
        .iter()
        .map(|layer| match layer {
            Some(sim_model::DecoderLayerCache::Attention(cache)) => {
                Ok((cache.keys().storage_id(), cache.values().storage_id()))
            }
            _ => Err("expected an attention cache layer".into()),
        })
        .collect()
}

fn architecture_for_contract(source_path: &str, symbol: &str) -> DecoderArchitecture {
    decoder_profile_fact(symbol)
        .map(|profile| profile.architecture)
        .unwrap_or(match source_path {
            GEMMA4_SOURCE_PATH => DecoderArchitecture::Gemma,
            GPT_OSS_SOURCE_PATH => DecoderArchitecture::GptOss,
            QWEN35_SOURCE_PATH => DecoderArchitecture::Qwen35,
            _ => DecoderArchitecture::Llama,
        })
}

fn execute_tiny_decoder_graph(architecture: DecoderArchitecture) -> Result<(), Box<dyn Error>> {
    let (backend, authority, model, cancellation) = model(architecture)?;
    let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 2], &[1, 2], &context)?;
    let output = model.forward(
        &backend,
        DecoderTextRequest {
            tokens: &tokens,
            attention_mask: None,
            positions: None,
            cache: None,
            capture_layer: Some(-1),
        },
        &context,
    )?;
    assert_eq!(output.last_hidden_state().descriptor().shape(), [1, 2, 4]);
    assert_eq!(output.logits().descriptor().shape(), [1, 2, 8]);
    assert!(output.intermediate().is_some());
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

fn execute_tokenizer_adapter() -> Result<(), Box<dyn Error>> {
    let vocabulary = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
    ))?;
    let merges = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
    ))?;
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.decoder.contract.tokenizer")?,
            &vocabulary,
            &merges,
        )?),
        TokenizerConfiguration {
            maximum_length: 6,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: true,
            pad_left: false,
            start_token: Some(1),
            end_token: Some(2),
            pad_token: 2,
            maximum_word_length: 32,
            disable_weights: false,
            embedding_width: None,
        },
        BTreeMap::new(),
    )?;
    let prompt = tokenize_decoder_prompt(&tokenizer, "a test", &CancellationToken::default())?;
    assert_eq!(prompt.sections().len(), 1);
    assert_eq!(prompt.sections()[0].tokens().len(), 6);
    Ok(())
}

fn execute_valid_contract(source_path: &str, symbol: &str) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let behavior = decoder_symbol_behavior(symbol)
        .ok_or_else(|| format!("unaccounted decoder symbol {symbol}"))?;
    match behavior {
        DecoderSymbolBehavior::Profile => {
            let profile = decoder_profile_fact(symbol)
                .ok_or_else(|| format!("decoder profile {symbol} is missing"))?;
            assert!(profile.vocabulary_size > 0);
            assert!(profile.hidden_size > 0);
            assert!(profile.hidden_layers > 0);
            assert!(
                profile
                    .rope_theta()
                    .all(|theta| theta.is_finite() && theta > 0.0)
            );
        }
        DecoderSymbolBehavior::ProfileFactory => match symbol {
            "_make_config" => {
                for model_type in [
                    "qwen35_08b",
                    "qwen35_2b",
                    "qwen35_4b",
                    "qwen35_9b",
                    "qwen35_27b",
                ] {
                    assert_eq!(
                        decoder_profile_fact(model_type)
                            .ok_or("Qwen3.5 factory profile is missing")?
                            .transformer_type,
                        model_type
                    );
                }
            }
            "_qwen35_layer_types" => {
                let profile =
                    decoder_profile_fact("Qwen35Config").ok_or("Qwen3.5 profile is missing")?;
                let period = profile
                    .linear_attention_period
                    .ok_or("Qwen3.5 layer period is missing")?;
                let layers = (0..profile.hidden_layers)
                    .map(|index| {
                        if (index + 1).is_multiple_of(period) {
                            DecoderLayerKind::FullAttention
                        } else {
                            DecoderLayerKind::LinearAttention
                        }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(layers.len(), 24);
                assert_eq!(layers.get(3), Some(&DecoderLayerKind::FullAttention));
                assert_eq!(layers.get(23), Some(&DecoderLayerKind::FullAttention));
            }
            "_make_variant" => {
                assert!(decoder_profile_fact("Gemma4_E2B_Config").is_some());
                assert!(decoder_profile_fact("Gemma4_31B_Config").is_some());
            }
            _ => return Err(format!("unaccounted profile factory {symbol}").into()),
        },
        DecoderSymbolBehavior::Normalization
        | DecoderSymbolBehavior::Attention
        | DecoderSymbolBehavior::FeedForward
        | DecoderSymbolBehavior::DecoderGraph
        | DecoderSymbolBehavior::Mask => {
            execute_tiny_decoder_graph(architecture_for_contract(source_path, symbol))?;
        }
        DecoderSymbolBehavior::Generation => {
            let (backend, authority, model, cancellation) = model(DecoderArchitecture::Llama)?;
            let context = context(&authority, &cancellation, 64 * 1024 * 1024)?;
            let tokens = i64_tensor(&backend, &[1, 1], &[1], &context)?;
            let generated = model.generate(
                &backend,
                &tokens,
                &DecoderGenerationConfiguration {
                    maximum_new_tokens: 1,
                    temperature_bits: 0.0_f32.to_bits(),
                    top_k: None,
                    top_p_bits: None,
                    minimum_p_bits: None,
                    repetition_penalty_bits: 1.0_f32.to_bits(),
                    presence_penalty_bits: 0.0_f32.to_bits(),
                },
                &generation_transaction()?,
                &context,
            )?;
            assert_eq!(generated.tokens.len(), 2);
        }
        DecoderSymbolBehavior::Router => {
            gpt_oss_top_k_route(&[1.0, 0.0], 1, 2, 1, &cancellation)?;
        }
        DecoderSymbolBehavior::Experts => {
            gpt_oss_moe(
                &[0.5, -0.25],
                &[1.0, 0.0],
                &[0.2; 8],
                &[0.3; 8],
                &[0.4; 8],
                1,
                2,
                2,
                2,
                1,
                &cancellation,
            )?;
        }
        DecoderSymbolBehavior::Projection => {
            assert_eq!(
                gemma4_clipped_linear(
                    &[2.0, -2.0],
                    &[1.0, 0.0, 0.0, 1.0],
                    None,
                    1,
                    2,
                    2,
                    [-1.0, 1.0],
                    [-0.5, 0.5],
                    &cancellation,
                )?,
                vec![0.5, -0.5]
            );
        }
        DecoderSymbolBehavior::Rope => {
            let scaling = if source_path == GPT_OSS_SOURCE_PATH {
                RopeScaling::Yarn {
                    factor: 32.0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                }
            } else {
                RopeScaling::None
            };
            precompute_rope(
                &[0, 1],
                &DecoderRopeConfiguration {
                    theta: 10_000.0,
                    rotary_dimension: 2,
                    interleaved_sections: Vec::new(),
                    scaling,
                },
                &cancellation,
            )?;
        }
        DecoderSymbolBehavior::VisionRope => {
            gemma4_vision_rope(1, 2, 8, 10_000.0, &cancellation)?;
        }
        DecoderSymbolBehavior::VisionPatch => {
            if source_path == QWEN35_SOURCE_PATH {
                qwen35_vision_patch_embed(
                    &[0.1, 0.2],
                    &[0.5, -0.5],
                    &[0.0],
                    1,
                    2,
                    1,
                    &cancellation,
                )?;
            } else {
                gemma4_vision_patch_embed(
                    &[0.0, 1.0],
                    &[[0, 0]],
                    &[0.5, -0.5],
                    &[0.1, 0.2],
                    1,
                    1,
                    2,
                    1,
                    1,
                    &cancellation,
                )?;
            }
        }
        DecoderSymbolBehavior::VisionFeedForward | DecoderSymbolBehavior::VisionMerge => {
            qwen35_vision_patch_merge(
                &[0.25],
                1,
                1,
                1,
                &[1.0],
                None,
                &[1.0],
                None,
                1,
                &cancellation,
            )?;
        }
        DecoderSymbolBehavior::VisionAttention | DecoderSymbolBehavior::VisionGraph => {
            execute_tiny_decoder_graph(architecture_for_contract(source_path, symbol))?;
            gemma4_vision_rope(1, 2, 8, 10_000.0, &cancellation)?;
        }
        DecoderSymbolBehavior::AudioSubsample | DecoderSymbolBehavior::AudioConvolution => {
            gemma4_audio_conv2d_subsample(&[1.0], &[0.5], 1, 1, 1, 1, 1, 1, 1, 0, &cancellation)?;
        }
        DecoderSymbolBehavior::AudioRelativePosition => {
            gemma4_audio_relative_positions(2, 2, 2)?;
        }
        DecoderSymbolBehavior::AudioFeedForward
        | DecoderSymbolBehavior::AudioAttention
        | DecoderSymbolBehavior::AudioGraph => {
            execute_tiny_decoder_graph(DecoderArchitecture::Gemma)?;
            gemma4_audio_relative_positions(2, 2, 2)?;
        }
        DecoderSymbolBehavior::LinearRecurrence => {
            qwen35_chunk_gated_delta_rule_exact(
                &[0.5, 0.25],
                &[0.2, 0.1],
                &[0.8],
                &[0.0],
                &[0.5],
                &[],
                1,
                1,
                1,
                2,
                1,
                &cancellation,
            )?;
        }
        DecoderSymbolBehavior::CausalConvolution => {
            qwen35_causal_conv1d_update_exact(
                &[0.5],
                &[0.0, 0.0],
                &[0.25, 0.5, 0.25],
                None,
                1,
                1,
                1,
                3,
                &cancellation,
            )?;
        }
        DecoderSymbolBehavior::TokenizerAdapter => execute_tokenizer_adapter()?,
        DecoderSymbolBehavior::ModelAdapter => {
            execute_tiny_decoder_graph(architecture_for_contract(source_path, symbol))?;
            execute_tokenizer_adapter()?;
        }
    }
    Ok(())
}

fn execute_invalid_contract(source_path: &str, symbol: &str) -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let behavior = decoder_symbol_behavior(symbol)
        .ok_or_else(|| format!("unaccounted decoder symbol {symbol}"))?;
    let rejected = match behavior {
        DecoderSymbolBehavior::Router | DecoderSymbolBehavior::Experts => {
            gpt_oss_top_k_route(&[f32::NAN], 1, 1, 1, &cancellation).is_err()
        }
        DecoderSymbolBehavior::LinearRecurrence | DecoderSymbolBehavior::CausalConvolution => {
            qwen35_causal_conv1d_update(&[1.0], &[], 1, 1, 1, &[], &cancellation).is_err()
        }
        DecoderSymbolBehavior::VisionRope
        | DecoderSymbolBehavior::VisionPatch
        | DecoderSymbolBehavior::VisionFeedForward
        | DecoderSymbolBehavior::VisionAttention
        | DecoderSymbolBehavior::VisionGraph
        | DecoderSymbolBehavior::VisionMerge => {
            gemma4_vision_rope(1, 1, 6, 10_000.0, &cancellation).is_err()
        }
        DecoderSymbolBehavior::AudioSubsample
        | DecoderSymbolBehavior::AudioFeedForward
        | DecoderSymbolBehavior::AudioRelativePosition
        | DecoderSymbolBehavior::AudioAttention
        | DecoderSymbolBehavior::AudioConvolution
        | DecoderSymbolBehavior::AudioGraph => {
            gemma4_audio_conv2d_subsample(&[], &[0.5], 1, 1, 1, 1, 1, 1, 1, 0, &cancellation)
                .is_err()
        }
        DecoderSymbolBehavior::Projection => gemma4_clipped_linear(
            &[1.0],
            &[],
            None,
            1,
            1,
            1,
            [-1.0, 1.0],
            [-1.0, 1.0],
            &cancellation,
        )
        .is_err(),
        DecoderSymbolBehavior::TokenizerAdapter | DecoderSymbolBehavior::ModelAdapter => {
            ModelTokenizerDescriptor::checked("").is_err()
        }
        _ => {
            let mut invalid = configuration(architecture_for_contract(source_path, symbol));
            invalid.head_dimension = 0;
            invalid.validate().is_err()
        }
    };
    if !rejected {
        return Err(
            format!("decoder invalid fixture was accepted for {source_path}:{symbol}").into(),
        );
    }
    Ok(())
}

#[test]
fn source_profile_facts_cover_every_static_and_factory_decoder_profile_exactly() {
    for symbol in DECODER_TEXT_ENCODER_CATALOG_SYMBOLS {
        assert!(
            decoder_symbol_behavior(symbol).is_some(),
            "decoder symbol {symbol} has no executable behavior owner"
        );
    }
    let expected = [
        ("Llama2Config", 128_320, 4096, 14_336, 32, 32, 8, 128, 8192),
        (
            "Mistral3Small24BConfig",
            131_072,
            5120,
            32_768,
            40,
            32,
            8,
            128,
            8192,
        ),
        (
            "Ministral3_3BConfig",
            131_072,
            3072,
            9216,
            26,
            32,
            8,
            128,
            262_144,
        ),
        (
            "Qwen25_3BConfig",
            151_936,
            2048,
            11_008,
            36,
            16,
            2,
            128,
            128_000,
        ),
        (
            "Qwen3_06BConfig",
            151_936,
            1024,
            3072,
            28,
            16,
            8,
            128,
            32_768,
        ),
        (
            "Qwen3_06B_ACE15_Config",
            151_669,
            1024,
            3072,
            28,
            16,
            8,
            128,
            32_768,
        ),
        (
            "Qwen3_2B_ACE15_lm_Config",
            217_204,
            2048,
            6144,
            28,
            16,
            8,
            128,
            40_960,
        ),
        (
            "Qwen3_4B_ACE15_lm_Config",
            217_204,
            2560,
            9728,
            36,
            32,
            8,
            128,
            40_960,
        ),
        (
            "Qwen3_4BConfig",
            151_936,
            2560,
            9728,
            36,
            32,
            8,
            128,
            40_960,
        ),
        (
            "Qwen3_8BConfig",
            151_936,
            4096,
            12_288,
            36,
            32,
            8,
            128,
            40_960,
        ),
        (
            "Qwen3VL_8BConfig",
            151_936,
            4096,
            12_288,
            36,
            32,
            8,
            128,
            262_144,
        ),
        (
            "Qwen3VL_4BConfig",
            151_936,
            2560,
            9728,
            36,
            32,
            8,
            128,
            262_144,
        ),
        (
            "Ovis25_2BConfig",
            151_936,
            2048,
            6144,
            28,
            16,
            8,
            128,
            40_960,
        ),
        (
            "Qwen25_7BVLI_Config",
            152_064,
            3584,
            18_944,
            28,
            28,
            4,
            128,
            128_000,
        ),
        ("Gemma2_2B_Config", 256_000, 2304, 9216, 26, 8, 4, 256, 8192),
        (
            "Gemma3_4B_Config",
            262_208,
            2560,
            10_240,
            34,
            8,
            4,
            256,
            131_072,
        ),
        (
            "Gemma3_4B_Vision_Config",
            262_208,
            2560,
            10_240,
            34,
            8,
            4,
            256,
            131_072,
        ),
        (
            "Gemma3_12B_Config",
            262_208,
            3840,
            15_360,
            48,
            16,
            8,
            256,
            131_072,
        ),
        (
            "Gemma4Config",
            262_144,
            2560,
            10_240,
            42,
            8,
            2,
            256,
            131_072,
        ),
        (
            "Gemma4_E2B_Config",
            262_144,
            1536,
            6144,
            35,
            8,
            1,
            256,
            131_072,
        ),
        (
            "Gemma4_31B_Config",
            262_144,
            5376,
            21_504,
            60,
            32,
            16,
            256,
            131_072,
        ),
        ("GptOss20BConfig", 201_088, 2880, 2880, 24, 64, 8, 64, 4096),
        ("Qwen35Config", 248_320, 2048, 6144, 24, 8, 2, 256, 32_768),
        (
            "_make_config:qwen35_08b",
            248_320,
            1024,
            3584,
            24,
            8,
            2,
            256,
            32_768,
        ),
        (
            "_make_config:qwen35_4b",
            248_320,
            2560,
            9216,
            32,
            16,
            4,
            256,
            32_768,
        ),
        (
            "_make_config:qwen35_9b",
            248_320,
            4096,
            12_288,
            32,
            16,
            4,
            256,
            32_768,
        ),
        (
            "_make_config:qwen35_27b",
            248_320,
            5120,
            17_408,
            64,
            24,
            4,
            256,
            32_768,
        ),
    ];
    assert_eq!(DECODER_PROFILE_FACTS.len(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        let profile = DECODER_PROFILE_FACTS
            .get(index)
            .expect("the expected profile index is in range");
        assert_eq!(
            (
                profile.source_symbol,
                profile.vocabulary_size,
                profile.hidden_size,
                profile.intermediate_size,
                profile.hidden_layers,
                profile.attention_heads,
                profile.key_value_heads,
                profile.head_dimension,
                profile.maximum_positions,
            ),
            *expected
        );
        assert!(profile.normalization_epsilon().is_normal());
        assert_eq!(
            profile.rope_theta().count(),
            if profile.source_symbol == "" {
                0
            } else {
                1 + usize::from(
                    profile.transformer_type == "gemma3" || profile.transformer_type == "gemma4",
                )
            }
        );
    }

    let qwen_vl = decoder_profile_fact("Qwen3VL_8BConfig").expect("Qwen3 VL profile");
    assert_eq!(qwen_vl.rope_sections, &[24, 20, 20]);
    assert!(qwen_vl.interleaved_multidimensional_rope);
    assert_eq!(qwen_vl.rope_theta().collect::<Vec<_>>(), vec![5_000_000.0]);

    let gemma4 = decoder_profile_fact("Gemma4Config").expect("Gemma4 profile");
    assert_eq!(gemma4.sliding_pattern, &[512, 512, 512, 512, 512, 0]);
    assert_eq!(gemma4.global_head_dimension, Some(512));
    assert_eq!(gemma4.final_logit_soft_cap(), Some(30.0));
    assert_eq!(gemma4.hidden_size_per_layer_input, 256);
    assert_eq!(gemma4.shared_key_value_layers, 18);
    assert_eq!(
        gemma4.vision,
        Some(sim_model::DecoderVisionProfileFact {
            hidden_size: 768,
            intermediate_size: 3072,
            layers: 16,
            attention_heads: 12,
            head_dimension: Some(64),
            image_size: Some(896),
            patch_size: 16,
            temporal_patch_size: None,
            spatial_merge_size: None,
            position_embeddings: Some(10_240),
            pooling_kernel_size: Some(3),
        })
    );
    assert!(gemma4.audio.is_some());

    let gpt_oss = decoder_profile_fact("GptOss20BConfig").expect("GPT-OSS profile");
    assert_eq!(gpt_oss.local_experts, Some(32));
    assert_eq!(gpt_oss.experts_per_token, Some(4));
    assert_eq!(gpt_oss.rope_scale().collect::<Vec<_>>(), vec![32.0]);

    for model in [
        "qwen35_08b",
        "qwen35_2b",
        "qwen35_4b",
        "qwen35_9b",
        "qwen35_27b",
    ] {
        let profile = decoder_profile_fact(model).expect("Qwen3.5 factory profile");
        assert_eq!(profile.linear_attention_period, Some(4));
        assert_eq!(profile.linear_key_heads, Some(16));
        assert_eq!(profile.linear_key_head_dimension, Some(128));
        assert_eq!(profile.linear_value_head_dimension, Some(128));
        assert_eq!(profile.convolution_kernel_size, Some(4));
    }
}

#[test]
fn source_constants_match_the_pinned_decoder_files() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    for (path, expected) in [
        (LLAMA_SOURCE_PATH, LLAMA_SOURCE_SHA256),
        (GEMMA4_SOURCE_PATH, GEMMA4_SOURCE_SHA256),
        (GPT_OSS_SOURCE_PATH, GPT_OSS_SOURCE_SHA256),
        (QWEN35_SOURCE_PATH, QWEN35_SOURCE_SHA256),
        (TEXT_GENERATION_SOURCE_PATH, TEXT_GENERATION_SOURCE_SHA256),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            expected
        );
    }
    Ok(())
}

#[test]
fn decoder_tokenizer_wrappers_delegate_the_verified_canonical_prompt_owner()
-> Result<(), Box<dyn Error>> {
    let vocabulary = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/vocab.json"
    ))?;
    let merges = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/models/sd15-tiny-v1/merges.txt"
    ))?;
    let tokenizer = NativePromptTokenizer::checked(
        NativeTokenizerFamily::ClipBpe(ClipBpeTokenizer::from_json_and_merges(
            ModelTokenizerDescriptor::checked("sim.decoder.test.tokenizer")?,
            &vocabulary,
            &merges,
        )?),
        TokenizerConfiguration {
            maximum_length: 6,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: true,
            pad_left: false,
            start_token: Some(1),
            end_token: Some(2),
            pad_token: 2,
            maximum_word_length: 32,
            disable_weights: false,
            embedding_width: None,
        },
        BTreeMap::new(),
    )?;
    let prompt = tokenize_decoder_prompt(&tokenizer, "a test", &CancellationToken::default())?;
    assert_eq!(prompt.sections().len(), 1);
    assert_eq!(prompt.sections()[0].tokens().len(), 6);
    Ok(())
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
            "comfy-parity-clip-execution-foundation",
            "comfy-parity-clip-text-transformer-foundation",
            "comfy-parity-clip-vision-foundation",
            "comfy-parity-clip-text-encoder-breadth",
            "comfy-parity-clip-owner-consolidation"
        ],
    })
}

#[test]
fn val_clip_001_decoder_rows_execute_and_extend_cumulative_ledger() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut contracts = Vec::new();
    let mut symbols = Vec::new();
    for line in catalog.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.get(8).copied() != Some(TASK_ID) {
            continue;
        }
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[7], "sim_model::clip_text_encoder_decoder");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-CLIP-001");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), fields[5]);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_valid_contract(fields[2], fields[3])?;
        execute_invalid_contract(fields[2], fields[3])?;
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
    assert_eq!(symbols, DECODER_TEXT_ENCODER_CATALOG_SYMBOLS);
    assert_eq!(contracts.len(), 127);

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
            "passed": 127,
            "failed": 0,
            "skipped": 0,
            "case_ids": [
                "text-encoder-decoder:source-provenance-and-exact-row-closure",
                "text-encoder-decoder:rope-gqa-rmsnorm-and-gated-mlp",
                "text-encoder-decoder:causal-cache-and-generation-semantics",
                "text-encoder-decoder:typed-target-cancellation-oom-workspace",
            ],
            "implementations": implementations,
        }),
    );
    let passed = task_results.values().try_fold(0_u64, |total, result| {
        total
            .checked_add(
                result
                    .get("passed")
                    .and_then(Value::as_u64)
                    .ok_or("task result passed count is missing")?,
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
    let producer_path = "crates/sim_model/tests/clip_text_encoder_decoder.rs";
    artifact["implementation"] = json!({
        "path": producer_path,
        "sha256": format!(
            "{:x}",
            Sha256::digest(fs::read(workspace.join(producer_path))?)
        ),
    });
    if let Some(parent) = artifact_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
    Ok(())
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
    let mut saw_indented_body = false;
    let mut end = lines.len();
    for (index, line) in lines.iter().enumerate().skip(start + 1) {
        let trimmed = line.trim();
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
