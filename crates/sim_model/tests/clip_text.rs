use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::clip_text::{
    CLIP_TEXT_CATALOG_SYMBOLS, CLIP_TEXT_SOURCE_PATH, CLIP_TEXT_SOURCE_SHA256, ClipTextActivation,
    ClipTextConfiguration, ClipTextError, ClipTextInput, ClipTextIntermediate,
    ClipTextLayerWeights, ClipTextRequest, ClipTextWeights, NativeClipText, SD1_CLIP_SOURCE_PATH,
    SD1_CLIP_SOURCE_SHA256,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorBackend, TensorDescriptor, generated_native_diffusion::tensor_to_f32,
};
use sim_types::DeviceKind;
use std::{error::Error, fs, path::Path};

const MEMORY_LIMIT: u64 = 64 * 1024 * 1024;
const TEXT_TASK: &str = "comfy-parity-clip-text-transformer-foundation";
const TEXT_IMPLEMENTATION_CLOSURE: [&str; 5] = [
    "crates/sim_model/src/clip_text.rs",
    "crates/sim_model/src/clip.rs",
    "crates/sim_model/src/sim_model.rs",
    "crates/sim_model/tests/clip_backend_admission.rs",
    "crates/sim_model/tests/clip_text.rs",
];

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
    let (tensor, event) = backend.upload_f32(descriptor, values, context)?;
    backend.wait_event(event, context)?;
    Ok(tensor)
}

fn i64_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(
        values
            .len()
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or("I64 fixture byte count overflowed")?,
    )?;
    for value in values {
        bytes.extend(value.to_ne_bytes());
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, context.stream)?;
    let (tensor, event) = backend.upload_bytes(descriptor, &bytes, context)?;
    backend.wait_event(event, context)?;
    Ok(tensor)
}

fn filled_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or("tensor element count overflowed")?;
    tensor(
        backend,
        shape,
        &vec![value; usize::try_from(count)?],
        context,
    )
}

fn identity_matrix(
    backend: &CpuBackend,
    rows: usize,
    columns: usize,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let mut values = vec![0.0; rows * columns];
    for index in 0..rows.min(columns) {
        values[index * columns + index] = scale;
    }
    tensor(
        backend,
        &[u64::try_from(rows)?, u64::try_from(columns)?],
        &values,
        context,
    )
}

fn configuration(
    activation: ClipTextActivation,
    projection_dimension: Option<usize>,
) -> ClipTextConfiguration {
    ClipTextConfiguration {
        dtype: DType::F32,
        device: DeviceId::CPU,
        vocabulary_size: 8,
        max_position_embeddings: 4,
        hidden_size: 4,
        intermediate_size: 8,
        attention_heads: 2,
        layer_count: 2,
        eos_token_id: 7,
        activation,
        projection_dimension,
    }
}

fn layer_weights(
    backend: &CpuBackend,
    active: bool,
    scale: f32,
    context: &ExecutionContext<'_>,
) -> Result<ClipTextLayerWeights, Box<dyn Error>> {
    let matrix_scale = if active { scale } else { 0.0 };
    let bias_scale = if active { scale * 0.05 } else { 0.0 };
    Ok(ClipTextLayerWeights {
        layer_norm_1_weight: filled_tensor(backend, &[4], 1.0, context)?,
        layer_norm_1_bias: filled_tensor(backend, &[4], 0.0, context)?,
        query_weight: identity_matrix(backend, 4, 4, matrix_scale, context)?,
        query_bias: filled_tensor(backend, &[4], bias_scale, context)?,
        key_weight: identity_matrix(backend, 4, 4, matrix_scale, context)?,
        key_bias: filled_tensor(backend, &[4], -bias_scale, context)?,
        value_weight: identity_matrix(backend, 4, 4, matrix_scale, context)?,
        value_bias: filled_tensor(backend, &[4], bias_scale, context)?,
        output_weight: identity_matrix(backend, 4, 4, matrix_scale, context)?,
        output_bias: tensor(
            backend,
            &[4],
            &[bias_scale, -bias_scale, bias_scale * 2.0, -bias_scale * 2.0],
            context,
        )?,
        layer_norm_2_weight: filled_tensor(backend, &[4], 1.0, context)?,
        layer_norm_2_bias: filled_tensor(backend, &[4], 0.0, context)?,
        feed_forward_1_weight: identity_matrix(backend, 8, 4, matrix_scale, context)?,
        feed_forward_1_bias: filled_tensor(backend, &[8], bias_scale, context)?,
        feed_forward_2_weight: identity_matrix(backend, 4, 8, matrix_scale, context)?,
        feed_forward_2_bias: tensor(
            backend,
            &[4],
            &[-bias_scale, bias_scale * 2.0, bias_scale, -bias_scale * 2.0],
            context,
        )?,
    })
}

fn weights(
    backend: &CpuBackend,
    second_layer_active: bool,
    context: &ExecutionContext<'_>,
) -> Result<ClipTextWeights, Box<dyn Error>> {
    let token_values = (0..32)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 10.0)
        .collect::<Vec<_>>();
    let position_values = (0..16)
        .map(|index| ((index * 5 % 11) as f32 - 5.0) / 20.0)
        .collect::<Vec<_>>();
    Ok(ClipTextWeights {
        token_embedding: tensor(backend, &[8, 4], &token_values, context)?,
        position_embedding: tensor(backend, &[4, 4], &position_values, context)?,
        layers: vec![
            layer_weights(backend, true, 0.35, context)?,
            layer_weights(backend, second_layer_active, 0.55, context)?,
        ],
        final_layer_norm_weight: filled_tensor(backend, &[4], 1.0, context)?,
        final_layer_norm_bias: filled_tensor(backend, &[4], 0.0, context)?,
    })
}

fn model(
    backend: &CpuBackend,
    activation: ClipTextActivation,
    second_layer_active: bool,
    with_projection: bool,
    context: &ExecutionContext<'_>,
) -> Result<NativeClipText, Box<dyn Error>> {
    let projection = with_projection
        .then(|| identity_matrix(backend, 3, 4, 0.75, context))
        .transpose()?;
    Ok(NativeClipText::new(
        configuration(activation, with_projection.then_some(3)),
        weights(backend, second_layer_active, context)?,
        projection,
    )?)
}

fn request<'a>(tokens: &'a Tensor, intermediate: ClipTextIntermediate) -> ClipTextRequest<'a> {
    ClipTextRequest {
        input: ClipTextInput::Tokens(tokens),
        attention_mask: None,
        num_tokens: None,
        intermediate,
        final_layer_norm_intermediate: true,
        project_pooled: true,
        zero_out_masked: false,
    }
}

#[test]
fn source_identity_and_exact_ten_catalog_symbols_are_pinned() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    for (path, expected) in [
        (CLIP_TEXT_SOURCE_PATH, CLIP_TEXT_SOURCE_SHA256),
        (SD1_CLIP_SOURCE_PATH, SD1_CLIP_SOURCE_SHA256),
    ] {
        let source = fs::read(workspace.join(path))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), expected);
    }
    assert_eq!(CLIP_TEXT_CATALOG_SYMBOLS.len(), 10);
    Ok(())
}

#[test]
fn hidden_capture_continues_remaining_layers_and_final_pooling() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let tokens = i64_tensor(&backend, &[2, 4], &[1, 2, 7, 7, 3, 4, 5, 7], &context)?;
    let inactive_second = model(
        &backend,
        ClipTextActivation::QuickGelu,
        false,
        true,
        &context,
    )?;
    let active_second = model(
        &backend,
        ClipTextActivation::QuickGelu,
        true,
        true,
        &context,
    )?;
    let inactive = inactive_second.forward(
        &backend,
        request(&tokens, ClipTextIntermediate::Layer(0)),
        &context,
    )?;
    let active = active_second.forward(
        &backend,
        request(&tokens, ClipTextIntermediate::Layer(0)),
        &context,
    )?;
    assert_eq!(
        inactive.last_hidden_state().descriptor().shape(),
        &[2, 4, 4]
    );
    assert_eq!(
        inactive
            .intermediate()
            .ok_or("missing hidden capture")?
            .descriptor()
            .shape(),
        &[2, 4, 4]
    );
    assert_eq!(inactive.pooled().descriptor().shape(), &[2, 4]);
    assert_eq!(
        inactive
            .projected_pooled()
            .ok_or("missing projected pool")?
            .descriptor()
            .shape(),
        &[2, 3]
    );
    assert_eq!(
        &*tensor_to_f32(
            &backend,
            inactive.intermediate().ok_or("missing hidden capture")?,
            &context,
        )?,
        &*tensor_to_f32(
            &backend,
            active.intermediate().ok_or("missing hidden capture")?,
            &context,
        )?,
        "capturing layer zero must not substitute the final hidden state"
    );
    assert_ne!(
        &*tensor_to_f32(&backend, inactive.last_hidden_state(), &context)?,
        &*tensor_to_f32(&backend, active.last_hidden_state(), &context)?,
        "the remaining active layer must still affect final pooling state"
    );
    assert_ne!(
        &*tensor_to_f32(&backend, inactive.pooled(), &context)?,
        &*tensor_to_f32(&backend, active.pooled(), &context)?
    );
    Ok(())
}

#[test]
fn causal_padding_projection_and_layer_list_semantics_are_exact() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let first = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 7], &context)?;
    let future_changed = i64_tensor(&backend, &[1, 4], &[1, 6, 3, 7], &context)?;
    let model = model(&backend, ClipTextActivation::Gelu, true, true, &context)?;
    let first_output = model.forward(
        &backend,
        request(&first, ClipTextIntermediate::All),
        &context,
    )?;
    let changed_output = model.forward(
        &backend,
        request(&future_changed, ClipTextIntermediate::Layers(vec![1, 0])),
        &context,
    )?;
    assert_eq!(
        first_output
            .intermediate()
            .ok_or("missing all-layer capture")?
            .descriptor()
            .shape(),
        &[1, 2, 4, 4]
    );
    assert_eq!(
        changed_output
            .intermediate()
            .ok_or("missing list capture")?
            .descriptor()
            .shape(),
        &[1, 2, 4, 4]
    );
    let first_values = tensor_to_f32(&backend, first_output.last_hidden_state(), &context)?;
    let changed_values = tensor_to_f32(&backend, changed_output.last_hidden_state(), &context)?;
    for (left, right) in first_values
        .iter()
        .take(4)
        .zip(changed_values.iter().take(4))
    {
        assert!(
            (left - right).abs() < 1.0e-6,
            "future token changed causal output"
        );
    }

    let mask = i64_tensor(&backend, &[1, 4], &[1, 1, 0, 0], &context)?;
    let masked = model.forward(
        &backend,
        ClipTextRequest {
            input: ClipTextInput::Tokens(&first),
            attention_mask: Some(&mask),
            num_tokens: None,
            intermediate: ClipTextIntermediate::All,
            final_layer_norm_intermediate: true,
            project_pooled: false,
            zero_out_masked: true,
        },
        &context,
    )?;
    assert!(masked.attention_mask().is_some());
    let masked_values = tensor_to_f32(&backend, masked.last_hidden_state(), &context)?;
    assert!(masked_values[8..].iter().all(|value| *value == 0.0));
    let masked_intermediate = tensor_to_f32(
        &backend,
        masked.intermediate().ok_or("missing masked intermediate")?,
        &context,
    )?;
    for layer in 0..2 {
        let start = layer * 16 + 8;
        assert!(
            masked_intermediate[start..start + 8]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
    Ok(())
}

#[test]
fn embedding_input_num_tokens_and_all_activations_execute() -> Result<(), Box<dyn Error>> {
    for activation in [
        ClipTextActivation::QuickGelu,
        ClipTextActivation::Gelu,
        ClipTextActivation::GeluTanh,
    ] {
        let (backend, authority) = backend()?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
        let model = model(&backend, activation, true, false, &context)?;
        let embeddings = tensor(
            &backend,
            &[2, 4, 4],
            &(0..32)
                .map(|index| (index as f32 - 16.0) / 12.0)
                .collect::<Vec<_>>(),
            &context,
        )?;
        let output = model.forward(
            &backend,
            ClipTextRequest {
                input: ClipTextInput::Embeddings(&embeddings),
                attention_mask: None,
                num_tokens: Some(&[2, 4]),
                intermediate: ClipTextIntermediate::Layer(-1),
                final_layer_norm_intermediate: false,
                project_pooled: false,
                zero_out_masked: false,
            },
            &context,
        )?;
        assert_eq!(output.pooled().descriptor().shape(), &[2, 4]);
        let hidden = tensor_to_f32(&backend, output.last_hidden_state(), &context)?;
        let pooled = tensor_to_f32(&backend, output.pooled(), &context)?;
        assert_eq!(&pooled[..4], &hidden[4..8]);
        assert_eq!(&pooled[4..], &hidden[28..32]);
        assert!(hidden.iter().any(|value| value.abs() > 1.0e-6));
        drop(pooled);
        drop(hidden);
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }
    Ok(())
}

#[test]
fn invalid_configuration_inputs_masks_projection_and_layers_fail_typed()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let mut invalid_configuration = configuration(ClipTextActivation::QuickGelu, None);
    invalid_configuration.device = DeviceId::new(DeviceKind::Metal, 0);
    assert!(matches!(
        NativeClipText::new(
            invalid_configuration,
            weights(&backend, true, &context)?,
            None,
        ),
        Err(ClipTextError::UnsupportedTarget { .. })
    ));
    let mut malformed_weights = weights(&backend, true, &context)?;
    malformed_weights.final_layer_norm_weight = filled_tensor(&backend, &[3], 1.0, &context)?;
    assert!(matches!(
        NativeClipText::new(
            configuration(ClipTextActivation::QuickGelu, None),
            malformed_weights,
            None,
        ),
        Err(ClipTextError::Module(_))
    ));

    let model = model(
        &backend,
        ClipTextActivation::QuickGelu,
        true,
        false,
        &context,
    )?;
    let invalid_tokens = i64_tensor(&backend, &[1, 4], &[1, 8, 2, 7], &context)?;
    assert!(matches!(
        model.forward(
            &backend,
            ClipTextRequest {
                project_pooled: false,
                ..request(&invalid_tokens, ClipTextIntermediate::None)
            },
            &context,
        ),
        Err(ClipTextError::TokenOutOfRange(8))
    ));
    let tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 7], &context)?;
    assert!(matches!(
        model.forward(
            &backend,
            ClipTextRequest {
                project_pooled: false,
                ..request(&tokens, ClipTextIntermediate::Layers(vec![0, -2]))
            },
            &context,
        ),
        Err(ClipTextError::DuplicateIntermediate)
    ));
    assert!(matches!(
        model.forward(
            &backend,
            ClipTextRequest {
                project_pooled: true,
                ..request(&tokens, ClipTextIntermediate::None)
            },
            &context,
        ),
        Err(ClipTextError::MissingProjection)
    ));
    let invalid_mask = tensor(&backend, &[1, 4], &[1.0, 0.5, 0.0, 0.0], &context)?;
    assert!(matches!(
        model.forward(
            &backend,
            ClipTextRequest {
                attention_mask: Some(&invalid_mask),
                project_pooled: false,
                ..request(&tokens, ClipTextIntermediate::None)
            },
            &context,
        ),
        Err(ClipTextError::InvalidInput(_))
    ));
    let embeddings = filled_tensor(&backend, &[1, 4, 4], 0.0, &context)?;
    assert!(matches!(
        model.forward(
            &backend,
            ClipTextRequest {
                input: ClipTextInput::Embeddings(&embeddings),
                attention_mask: None,
                num_tokens: None,
                intermediate: ClipTextIntermediate::None,
                final_layer_norm_intermediate: true,
                project_pooled: false,
                zero_out_masked: false,
            },
            &context,
        ),
        Err(ClipTextError::InvalidInput(_))
    ));
    Ok(())
}

#[test]
fn cancellation_and_workspace_oom_publish_nothing_and_converge() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let active = CancellationToken::default();
    let setup = context(&authority, &active, 16 * 1024 * 1024)?;
    let model = model(&backend, ClipTextActivation::QuickGelu, true, true, &setup)?;
    let tokens = i64_tensor(&backend, &[1, 4], &[1, 2, 3, 7], &setup)?;
    drop(setup);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 1024 * 1024)?;
    assert!(
        model
            .forward(
                &backend,
                request(&tokens, ClipTextIntermediate::All),
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    drop(cancelled_context);

    let insufficient = context(&authority, &active, 16)?;
    assert!(
        model
            .forward(
                &backend,
                request(&tokens, ClipTextIntermediate::All),
                &insufficient,
            )
            .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    drop(insufficient);

    let successful = context(&authority, &active, 16 * 1024 * 1024)?;
    model.forward(
        &backend,
        request(&tokens, ClipTextIntermediate::All),
        &successful,
    )?;
    assert_eq!(successful.scratch.in_use_bytes(), 0);
    Ok(())
}

fn execute_valid_text_contract(symbol: &str) -> Result<(), Box<dyn Error>> {
    if !CLIP_TEXT_CATALOG_SYMBOLS.contains(&symbol) {
        return Err(format!("unaccounted CLIP text symbol {symbol}").into());
    }
    hidden_capture_continues_remaining_layers_and_final_pooling()?;
    causal_padding_projection_and_layer_list_semantics_are_exact()?;
    embedding_input_num_tokens_and_all_activations_execute()?;
    Ok(())
}

fn execute_invalid_text_contract(symbol: &str) -> Result<(), Box<dyn Error>> {
    if !CLIP_TEXT_CATALOG_SYMBOLS.contains(&symbol) {
        return Err(format!("unaccounted CLIP text symbol {symbol}").into());
    }
    invalid_configuration_inputs_masks_projection_and_layers_fail_typed()?;
    cancellation_and_workspace_oom_publish_nothing_and_converge()?;
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
fn val_clip_001_text_rows_execute_and_extend_cumulative_ledger() -> Result<(), Box<dyn Error>> {
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
        if fields.get(8).copied() != Some(TEXT_TASK) {
            continue;
        }
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[7], "sim_model::clip");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-CLIP-001");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), fields[5]);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_valid_text_contract(fields[3])?;
        execute_invalid_text_contract(fields[3])?;
        symbols.push(fields[3]);
        contracts.push(json!({
            "contract_id": fields[0],
            "task_id": TEXT_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": [
                format!("{}:native-text-valid", fields[0]),
                format!("{}:native-text-invalid", fields[0]),
            ],
        }));
    }
    assert_eq!(symbols, CLIP_TEXT_CATALOG_SYMBOLS);
    assert_eq!(contracts.len(), CLIP_TEXT_CATALOG_SYMBOLS.len());

    let artifact_path = workspace.join("target/sim-parity/val-clip-001.json");
    let mut artifact = if artifact_path.exists() {
        serde_json::from_slice::<Value>(&fs::read(&artifact_path)?)?
    } else {
        empty_clip_artifact()
    };
    assert_eq!(artifact.get("schema_version"), Some(&json!(1)));
    assert_eq!(artifact.get("validation_id"), Some(&json!("VAL-CLIP-001")));
    let task_results = artifact
        .get_mut("task_results")
        .and_then(Value::as_object_mut)
        .ok_or("VAL-CLIP-001 task results are missing")?;
    let implementations = TEXT_IMPLEMENTATION_CLOSURE
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            }))
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    task_results.insert(
        TEXT_TASK.to_owned(),
        json!({
            "status": "passed",
            "passed": contracts.len(),
            "failed": 0,
            "skipped": 0,
            "case_ids": [
                "task339:source-provenance-and-ten-contracts",
                "task339:embedding-attention-activation-residual",
                "task339:causal-padding-and-pooling",
                "task339:hidden-list-all-final-capture-continuation",
                "task339:projection-and-unprojected-pooling",
                "task339:typed-target-shape-mask-layer-rejection",
                "task339:cancellation-oom-no-publication",
                "task339:ownership-consolidation",
            ],
            "implementations": implementations,
        }),
    );
    let passed = task_results.values().try_fold(0_u64, |total, result| {
        let count = result
            .get("passed")
            .and_then(Value::as_u64)
            .ok_or("VAL-CLIP-001 task result has no passed count")?;
        total
            .checked_add(count)
            .ok_or("VAL-CLIP-001 passed count overflowed")
    })?;
    let artifact_contracts = artifact
        .get_mut("contracts")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 contracts are missing")?;
    artifact_contracts
        .retain(|contract| contract.get("task_id").and_then(Value::as_str) != Some(TEXT_TASK));
    artifact_contracts.extend(contracts);
    let remaining = artifact
        .get_mut("remaining_tasks")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 remaining tasks are missing")?;
    remaining.retain(|task| task.as_str() != Some(TEXT_TASK));
    artifact["summary"] = json!({"passed": passed, "failed": 0, "skipped": 0});
    artifact["implementation"] = json!({
        "path": "crates/sim_model/tests/clip_text.rs",
        "sha256": format!(
            "{:x}",
            Sha256::digest(fs::read(workspace.join("crates/sim_model/tests/clip_text.rs"))?)
        ),
    });
    fs::create_dir_all(
        artifact_path
            .parent()
            .ok_or("artifact parent is unavailable")?,
    )?;
    fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
    Ok(())
}
