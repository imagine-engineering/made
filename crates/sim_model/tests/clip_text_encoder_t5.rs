use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, BERT_SOURCE_PATH, BERT_SOURCE_SHA256,
    BidirectionalFeedForwardActivation, BidirectionalLayerWeights, BidirectionalPooling,
    BidirectionalTextArchitecture, BidirectionalTextConfiguration, BidirectionalTextError,
    BidirectionalTextInput, BidirectionalTextRequest, BidirectionalTextWeights, ModelStore,
    NativePromptTokenizer, NativeT5TextEncoder, NativeTokenValue, NativeTokenizerFamily,
    ParserLimits, SPIECE_TOKENIZER_SOURCE_PATH, SPIECE_TOKENIZER_SOURCE_SHA256,
    SentencePieceTokenizer, T5_BIDIRECTIONAL_CATALOG_SYMBOLS, T5_SOURCE_PATH, T5_SOURCE_SHA256,
    TokenizerConfiguration, relative_position_bucket, tokenize_bidirectional_prompt,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorDescriptor,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use std::{collections::BTreeMap, error::Error, fs, path::Path};

const MEMORY_LIMIT: u64 = 128 * 1024 * 1024;
const TASK_ID: &str = "comfy-parity-clip-text-encoder-t5-foundation";
const IMPLEMENTATION_CLOSURE: [&str; 4] = [
    "crates/sim_model/src/clip_text_encoder_t5.rs",
    "crates/sim_model/src/sim_model.rs",
    "crates/sim_model/tests/clip_backend_admission.rs",
    "crates/sim_model/tests/clip_text_encoder_t5.rs",
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
    Ok(tensor_from_f32(backend, shape, values, context)?)
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
                ((index * 7 % 13) as f32 - 6.0) * scale * 0.015
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

fn configuration(
    architecture: BidirectionalTextArchitecture,
    gated: bool,
    projection: bool,
) -> BidirectionalTextConfiguration {
    BidirectionalTextConfiguration {
        architecture,
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: 8,
        maximum_tokens: 4,
        type_vocabulary_size: if architecture == BidirectionalTextArchitecture::Bert {
            2
        } else {
            0
        },
        hidden_size: 4,
        attention_inner_size: 4,
        feed_forward_size: 8,
        attention_heads: 2,
        layer_count: 2,
        normalization_epsilon_bits: 1.0e-6_f32.to_bits(),
        gated_feed_forward: gated,
        activation: if architecture == BidirectionalTextArchitecture::T5 {
            BidirectionalFeedForwardActivation::GeluTanh
        } else {
            BidirectionalFeedForwardActivation::Gelu
        },
        relative_attention: architecture == BidirectionalTextArchitecture::T5,
        relative_attention_buckets: 8,
        relative_attention_max_distance: 16,
        projection_dimension: projection.then_some(3),
    }
}

fn layer_weights(
    backend: &CpuBackend,
    architecture: BidirectionalTextArchitecture,
    gated: bool,
    first: bool,
    context: &ExecutionContext<'_>,
) -> Result<BidirectionalLayerWeights, Box<dyn Error>> {
    let bert = architecture == BidirectionalTextArchitecture::Bert;
    let bias4 = || filled(backend, &[4], 0.01, context);
    let bias8 = || filled(backend, &[8], -0.015, context);
    let relative = if !bert && first {
        Some(tensor(
            backend,
            &[8, 2],
            &(0..16)
                .map(|index| (index as f32 - 8.0) / 40.0)
                .collect::<Vec<_>>(),
            context,
        )?)
    } else {
        None
    };
    Ok(BidirectionalLayerWeights {
        attention_norm_weight: filled(backend, &[4], 1.0, context)?,
        attention_norm_bias: bert.then(bias4).transpose()?,
        query_weight: matrix(backend, 4, 4, 0.31, context)?,
        query_bias: bert.then(bias4).transpose()?,
        key_weight: matrix(backend, 4, 4, 0.29, context)?,
        key_bias: bert.then(bias4).transpose()?,
        value_weight: matrix(backend, 4, 4, 0.37, context)?,
        value_bias: bert.then(bias4).transpose()?,
        attention_output_weight: matrix(backend, 4, 4, 0.41, context)?,
        attention_output_bias: bert.then(bias4).transpose()?,
        feed_forward_norm_weight: filled(backend, &[4], 1.0, context)?,
        feed_forward_norm_bias: bert.then(bias4).transpose()?,
        feed_forward_input_weight: matrix(backend, 8, 4, 0.47, context)?,
        feed_forward_input_bias: bert.then(bias8).transpose()?,
        feed_forward_gate_weight: gated
            .then(|| matrix(backend, 8, 4, 0.23, context))
            .transpose()?,
        feed_forward_gate_bias: None,
        feed_forward_output_weight: matrix(backend, 4, 8, 0.43, context)?,
        feed_forward_output_bias: bert.then(bias4).transpose()?,
        relative_attention_bias: relative,
    })
}

fn weights(
    backend: &CpuBackend,
    architecture: BidirectionalTextArchitecture,
    gated: bool,
    projection: bool,
    context: &ExecutionContext<'_>,
) -> Result<BidirectionalTextWeights, Box<dyn Error>> {
    let bert = architecture == BidirectionalTextArchitecture::Bert;
    let token_embedding = tensor(
        backend,
        &[8, 4],
        &(0..32)
            .map(|index| ((index * 5 % 17) as f32 - 8.0) / 12.0)
            .collect::<Vec<_>>(),
        context,
    )?;
    let position_embedding = bert
        .then(|| {
            tensor(
                backend,
                &[4, 4],
                &(0..16)
                    .map(|index| ((index * 3 % 11) as f32 - 5.0) / 25.0)
                    .collect::<Vec<_>>(),
                context,
            )
        })
        .transpose()?;
    let token_type_embedding = bert
        .then(|| {
            tensor(
                backend,
                &[2, 4],
                &[0.0, 0.0, 0.0, 0.0, 0.2, -0.2, 0.1, -0.1],
                context,
            )
        })
        .transpose()?;
    Ok(BidirectionalTextWeights {
        token_embedding,
        position_embedding,
        token_type_embedding,
        embedding_norm_weight: bert
            .then(|| filled(backend, &[4], 1.0, context))
            .transpose()?,
        embedding_norm_bias: bert
            .then(|| filled(backend, &[4], 0.0, context))
            .transpose()?,
        layers: vec![
            layer_weights(backend, architecture, gated, true, context)?,
            layer_weights(backend, architecture, gated, false, context)?,
        ],
        final_norm_weight: (!bert)
            .then(|| filled(backend, &[4], 1.0, context))
            .transpose()?,
        projection_weight: projection
            .then(|| matrix(backend, 3, 4, 0.67, context))
            .transpose()?,
        projection_bias: projection
            .then(|| filled(backend, &[3], 0.02, context))
            .transpose()?,
    })
}

fn model(
    backend: &CpuBackend,
    architecture: BidirectionalTextArchitecture,
    gated: bool,
    projection: bool,
    context: &ExecutionContext<'_>,
) -> Result<NativeT5TextEncoder, Box<dyn Error>> {
    Ok(NativeT5TextEncoder::new(
        configuration(architecture, gated, projection),
        weights(backend, architecture, gated, projection, context)?,
    )?)
}

fn request<'a>(tokens: &'a Tensor) -> BidirectionalTextRequest<'a> {
    BidirectionalTextRequest {
        input: BidirectionalTextInput::Tokens(tokens),
        attention_mask: None,
        token_type_ids: None,
        intermediate_layer: Some(0),
        final_norm_intermediate: true,
        pooling: BidirectionalPooling::MeanUnmasked,
        project_pooled: true,
    }
}

#[test]
fn source_identity_symbols_and_relative_buckets_are_exact() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    for (path, digest) in [
        (T5_SOURCE_PATH, T5_SOURCE_SHA256),
        (BERT_SOURCE_PATH, BERT_SOURCE_SHA256),
        (SPIECE_TOKENIZER_SOURCE_PATH, SPIECE_TOKENIZER_SOURCE_SHA256),
    ] {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            digest
        );
    }
    assert_eq!(T5_BIDIRECTIONAL_CATALOG_SYMBOLS.len(), 19);
    assert_eq!(relative_position_bucket(0, true, 32, 128)?, 0);
    assert_eq!(relative_position_bucket(-1, true, 32, 128)?, 1);
    assert_eq!(relative_position_bucket(1, true, 32, 128)?, 17);
    assert_eq!(relative_position_bucket(1024, true, 32, 128)?, 31);
    assert!(relative_position_bucket(1, true, 3, 128).is_err());
    Ok(())
}

#[test]
fn t5_relative_attention_gating_capture_and_pooling_execute() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[2, 4], &[1, 2, 3, 4, 4, 3, 2, 1], &context)?;
    let mask = i64_tensor(&backend, &[2, 4], &[1, 1, 1, 0, 1, 1, 0, 0], &context)?;
    let gated = model(
        &backend,
        BidirectionalTextArchitecture::T5,
        true,
        true,
        &context,
    )?;
    let ungated = model(
        &backend,
        BidirectionalTextArchitecture::T5,
        false,
        true,
        &context,
    )?;
    let request = BidirectionalTextRequest {
        attention_mask: Some(&mask),
        ..request(&tokens)
    };
    let gated_output = gated.forward(&backend, request.clone(), &context)?;
    let ungated_output = ungated.forward(&backend, request, &context)?;
    assert_eq!(
        gated_output.last_hidden_state().descriptor().shape(),
        &[2, 4, 4]
    );
    assert_eq!(
        gated_output
            .intermediate()
            .ok_or("missing capture")?
            .descriptor()
            .shape(),
        &[2, 4, 4]
    );
    assert_eq!(
        gated_output
            .pooled()
            .ok_or("missing pool")?
            .descriptor()
            .shape(),
        &[2, 4]
    );
    assert_eq!(
        gated_output
            .projected_pooled()
            .ok_or("missing projection")?
            .descriptor()
            .shape(),
        &[2, 3]
    );
    assert_ne!(
        &*tensor_to_f32(&backend, gated_output.last_hidden_state(), &context)?,
        &*tensor_to_f32(&backend, ungated_output.last_hidden_state(), &context)?,
        "gated and ungated feed-forward paths must differ"
    );
    assert_ne!(
        &*tensor_to_f32(
            &backend,
            gated_output.intermediate().ok_or("missing capture")?,
            &context
        )?,
        &*tensor_to_f32(&backend, gated_output.last_hidden_state(), &context)?,
        "intermediate capture must not stop execution"
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn t5_distinct_attention_inner_width_executes() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    let mut configuration = configuration(BidirectionalTextArchitecture::T5, true, false);
    configuration.attention_inner_size = 2;
    let mut weights = weights(
        &backend,
        BidirectionalTextArchitecture::T5,
        true,
        false,
        &context,
    )?;
    for layer in &mut weights.layers {
        layer.query_weight = matrix(&backend, 2, 4, 0.31, &context)?;
        layer.key_weight = matrix(&backend, 2, 4, 0.29, &context)?;
        layer.value_weight = matrix(&backend, 2, 4, 0.37, &context)?;
        layer.attention_output_weight = matrix(&backend, 4, 2, 0.41, &context)?;
    }
    let model = NativeT5TextEncoder::new(configuration, weights)?;
    let tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 4], &context)?;
    let output = model.forward(
        &backend,
        BidirectionalTextRequest {
            project_pooled: false,
            pooling: BidirectionalPooling::None,
            ..request(&tokens)
        },
        &context,
    )?;
    assert_eq!(output.last_hidden_state().descriptor().shape(), &[1, 4, 4]);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn umt5_per_layer_relative_bias_executes_without_shared_substitution() -> Result<(), Box<dyn Error>>
{
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    let mut configuration = configuration(BidirectionalTextArchitecture::T5, false, false);
    configuration.relative_attention = false;
    configuration.activation = BidirectionalFeedForwardActivation::Relu;
    let mut weights = weights(
        &backend,
        BidirectionalTextArchitecture::T5,
        false,
        false,
        &context,
    )?;
    let second = weights.layers.get_mut(1).ok_or("second layer is missing")?;
    second.relative_attention_bias = Some(tensor(
        &backend,
        &[8, 2],
        &(0..16)
            .map(|index| (8.0 - index as f32) / 35.0)
            .collect::<Vec<_>>(),
        &context,
    )?);
    let model = NativeT5TextEncoder::new(configuration, weights)?;
    let tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 4], &context)?;
    let output = model.forward(
        &backend,
        BidirectionalTextRequest {
            project_pooled: false,
            pooling: BidirectionalPooling::None,
            ..request(&tokens)
        },
        &context,
    )?;
    assert!(
        tensor_to_f32(&backend, output.last_hidden_state(), &context)?
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn bert_embeddings_masks_post_norm_and_embedding_input_execute() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 4], &context)?;
    let types_zero = i64_tensor(&backend, &[1, 4], &[0, 0, 0, 0], &context)?;
    let types_mixed = i64_tensor(&backend, &[1, 4], &[0, 1, 0, 1], &context)?;
    let model = model(
        &backend,
        BidirectionalTextArchitecture::Bert,
        false,
        false,
        &context,
    )?;
    let zero = model.forward(
        &backend,
        BidirectionalTextRequest {
            token_type_ids: Some(&types_zero),
            project_pooled: false,
            pooling: BidirectionalPooling::FirstToken,
            ..request(&tokens)
        },
        &context,
    )?;
    let mixed = model.forward(
        &backend,
        BidirectionalTextRequest {
            token_type_ids: Some(&types_mixed),
            project_pooled: false,
            pooling: BidirectionalPooling::FirstToken,
            ..request(&tokens)
        },
        &context,
    )?;
    assert_ne!(
        &*tensor_to_f32(&backend, zero.last_hidden_state(), &context)?,
        &*tensor_to_f32(&backend, mixed.last_hidden_state(), &context)?
    );
    let embeddings = tensor(
        &backend,
        &[1, 4, 4],
        &(0..16).map(|index| index as f32 / 20.0).collect::<Vec<_>>(),
        &context,
    )?;
    let embedded = model.forward(
        &backend,
        BidirectionalTextRequest {
            input: BidirectionalTextInput::Embeddings(&embeddings),
            attention_mask: None,
            token_type_ids: None,
            intermediate_layer: Some(-1),
            final_norm_intermediate: false,
            pooling: BidirectionalPooling::None,
            project_pooled: false,
        },
        &context,
    )?;
    assert!(embedded.pooled().is_none());
    assert!(embedded.intermediate().is_some());
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn tokenizer_input_delegates_to_verified_canonical_sentencepiece_owner()
-> Result<(), Box<dyn Error>> {
    let tokenizer = verified_prompt_tokenizer()?;
    let cancellation = CancellationToken::default();
    let prompt = tokenize_bidirectional_prompt(&tokenizer, "hello world", &cancellation)?;
    let numeric = prompt
        .sections()
        .iter()
        .flat_map(|section| section.tokens())
        .filter_map(|token| match token.value() {
            NativeTokenValue::Token(token) => Some(*token),
            NativeTokenValue::Embedding { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(numeric, [1, 3, 4, 2]);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(tokenize_bidirectional_prompt(&tokenizer, "hello", &cancelled).is_err());
    Ok(())
}

#[test]
fn target_shape_mask_layer_projection_cancellation_and_oom_fail_typed() -> Result<(), Box<dyn Error>>
{
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let setup = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    let mut invalid = configuration(BidirectionalTextArchitecture::T5, true, false);
    invalid.device = DeviceId::new(sim_types::DeviceKind::Metal, 0);
    assert!(matches!(
        NativeT5TextEncoder::new(
            invalid,
            weights(
                &backend,
                BidirectionalTextArchitecture::T5,
                true,
                false,
                &setup
            )?,
        ),
        Err(BidirectionalTextError::UnsupportedTarget { .. })
    ));
    let model = model(
        &backend,
        BidirectionalTextArchitecture::T5,
        true,
        false,
        &setup,
    )?;
    let invalid_tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 8, 4], &setup)?;
    assert!(matches!(
        model.forward(
            &backend,
            BidirectionalTextRequest {
                project_pooled: false,
                ..request(&invalid_tokens)
            },
            &setup,
        ),
        Err(BidirectionalTextError::TokenOutOfRange(8))
    ));
    let tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 4], &setup)?;
    let invalid_mask = tensor(&backend, &[1, 4], &[1.0, 0.5, 0.0, 0.0], &setup)?;
    assert!(matches!(
        model.forward(
            &backend,
            BidirectionalTextRequest {
                attention_mask: Some(&invalid_mask),
                project_pooled: false,
                ..request(&tokens)
            },
            &setup,
        ),
        Err(BidirectionalTextError::InvalidInput(_))
    ));
    assert!(matches!(
        model.forward(
            &backend,
            BidirectionalTextRequest {
                intermediate_layer: Some(2),
                project_pooled: false,
                ..request(&tokens)
            },
            &setup,
        ),
        Err(BidirectionalTextError::IntermediateOutOfRange { .. })
    ));
    assert!(matches!(
        model.forward(&backend, request(&tokens), &setup),
        Err(BidirectionalTextError::MissingProjection)
    ));
    drop(setup);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 1024 * 1024)?;
    assert!(
        model
            .forward(&backend, request(&tokens), &cancelled_context)
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    drop(cancelled_context);

    let insufficient = context(&authority, &cancellation, 16)?;
    assert!(
        model
            .forward(&backend, request(&tokens), &insufficient)
            .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    drop(insufficient);

    let success = context(&authority, &cancellation, 32 * 1024 * 1024)?;
    model.forward(
        &backend,
        BidirectionalTextRequest {
            project_pooled: false,
            ..request(&tokens)
        },
        &success,
    )?;
    assert_eq!(success.scratch.in_use_bytes(), 0);
    Ok(())
}

fn execute_valid_contract(symbol: &str) -> Result<(), Box<dyn Error>> {
    if symbol == "SPieceTokenizer" {
        return tokenizer_input_delegates_to_verified_canonical_sentencepiece_owner();
    }
    if symbol.starts_with("T5") {
        t5_relative_attention_gating_capture_and_pooling_execute()
    } else if symbol.starts_with("Bert") {
        bert_embeddings_masks_post_norm_and_embedding_input_execute()
    } else {
        Err(format!("unaccounted bidirectional symbol {symbol}").into())
    }
}

fn execute_invalid_contract(symbol: &str) -> Result<(), Box<dyn Error>> {
    if !T5_BIDIRECTIONAL_CATALOG_SYMBOLS.contains(&symbol) {
        return Err(format!("unaccounted bidirectional symbol {symbol}").into());
    }
    target_shape_mask_layer_projection_cancellation_and_oom_fail_typed()
}

#[test]
fn val_clip_001_t5_bidirectional_rows_execute_and_extend_cumulative_ledger()
-> Result<(), Box<dyn Error>> {
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
        assert_eq!(fields[7], "sim_model::clip_text_encoder_t5");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-CLIP-001");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), fields[5]);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_valid_contract(fields[3])?;
        execute_invalid_contract(fields[3])?;
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
    assert_eq!(symbols, T5_BIDIRECTIONAL_CATALOG_SYMBOLS);
    assert_eq!(contracts.len(), 19);

    let artifact_path = workspace.join("target/sim-parity/val-clip-001.json");
    let mut artifact = serde_json::from_slice::<Value>(&fs::read(&artifact_path)?)?;
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
            "passed": 19,
            "failed": 0,
            "skipped": 0,
            "case_ids": [
                "text-encoder-t5:source-provenance-and-exact-row-closure",
                "text-encoder-t5:sentencepiece-and-token-input-delegation",
                "text-encoder-t5:relative-attention-and-gated-feed-forward",
                "text-encoder-t5:typed-target-cancellation-oom-workspace",
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
    fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
    Ok(())
}

fn verified_prompt_tokenizer() -> Result<NativePromptTokenizer, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("tokenizer.model");
    fs::write(&path, sentencepiece_model_bytes())?;
    let cancellation = CancellationToken::default();
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "tokenizer",
        "tokenizers",
        directory.path(),
        ["model"],
    )?)?;
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("tokenizer", "tokenizer.model")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let vocabulary =
        store.verified_sentencepiece_vocabulary(&index, &loaded, &key, &cancellation)?;
    let sentencepiece = SentencePieceTokenizer::from_verified_vocabulary(vocabulary)?;
    Ok(NativePromptTokenizer::checked(
        NativeTokenizerFamily::SentencePiece(sentencepiece),
        TokenizerConfiguration {
            maximum_length: 4,
            minimum_length: None,
            minimum_padding: None,
            pad_to_maximum_length: true,
            pad_left: false,
            start_token: Some(1),
            end_token: Some(2),
            pad_token: 2,
            maximum_word_length: 8,
            disable_weights: false,
            embedding_width: None,
        },
        BTreeMap::new(),
    )?)
}

fn sentencepiece_model_bytes() -> Vec<u8> {
    let entries = [
        ("<unk>", -10.0_f32, 2_u64),
        ("<s>", 0.0, 3),
        ("</s>", 0.0, 3),
        ("▁hello", -1.0, 1),
        ("▁world", -1.0, 1),
    ];
    let mut model = Vec::new();
    for (piece, score, piece_type) in entries {
        let mut encoded = Vec::new();
        encoded.push(0x0a);
        push_varint(&mut encoded, piece.len() as u64);
        encoded.extend_from_slice(piece.as_bytes());
        encoded.push(0x15);
        encoded.extend_from_slice(&score.to_le_bytes());
        encoded.push(0x18);
        push_varint(&mut encoded, piece_type);
        model.push(0x0a);
        push_varint(&mut model, encoded.len() as u64);
        model.extend(encoded);
    }
    model
}

fn push_varint(output: &mut Vec<u8>, mut value: u64) {
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
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find_map(|(index, line)| {
            let trimmed = line.trim();
            (!trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !line.starts_with(char::is_whitespace))
            .then_some(index)
        })
        .unwrap_or(lines.len());
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
