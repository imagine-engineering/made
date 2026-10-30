use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_model::audio_encoder::{
    AUDIO_ENCODERS_SOURCE_PATH, AUDIO_ENCODERS_SOURCE_SHA256, AudioEncoderError,
    NODES_AUDIO_ENCODER_SOURCE_PATH, NODES_AUDIO_ENCODER_SOURCE_SHA256,
    NativeAudioEncoderCheckpoint, WAV2VEC2_SOURCE_PATH, WAV2VEC2_SOURCE_SHA256,
    WHISPER_SOURCE_PATH, WHISPER_SOURCE_SHA256, normalize_and_select_architecture,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId,
    TensorBackend,
    generated_sim_operator_indirection_01::tensor_from_f32_with_context_exact_native,
};
use std::{error::Error, fs, path::Path, sync::Arc};

const MEMORY_LIMIT: u64 = 256 * 1024 * 1024;
const ORACLE: &str = include_str!(
    "../../sim_test_support/fixtures/models/audio-encoder-resource-foundation/oracle.json"
);
const GENERATOR: &[u8] = include_bytes!(
    "../../sim_test_support/fixtures/models/audio-encoder-resource-foundation/generate_oracle.py"
);
const ORACLE_SHA256: &str = "873985e347289929433a8a66918ab17db19f9309e9ae13521ef7ef7d8943a218";

fn backend() -> Result<(Arc<CpuBackend>, CpuWorkspaceAuthority), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    Ok((Arc::new(backend), authority))
}

fn context<'a>(
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(MEMORY_LIMIT)?,
        rng_phase: None,
        cancellation,
    })
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, Box<dyn Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        DType::F32,
        backend.device(),
        context,
    )?)
}

fn tensor_with_dtype(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, Box<dyn Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        dtype,
        backend.device(),
        context,
    )?)
}

fn workspace_root() -> Result<&'static Path, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| "workspace root is unavailable".into())
}

fn assert_pinned_source(path: &str, expected: &str) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(workspace_root()?.join(path))?;
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
    Ok(())
}

#[test]
fn audio_encoder_profiles_normalize_and_select_source_exactly() -> Result<(), Box<dyn Error>> {
    assert_pinned_source(AUDIO_ENCODERS_SOURCE_PATH, AUDIO_ENCODERS_SOURCE_SHA256)?;
    assert_pinned_source(
        NODES_AUDIO_ENCODER_SOURCE_PATH,
        NODES_AUDIO_ENCODER_SOURCE_SHA256,
    )?;
    assert_pinned_source(WAV2VEC2_SOURCE_PATH, WAV2VEC2_SOURCE_SHA256)?;
    assert_pinned_source(WHISPER_SOURCE_PATH, WHISPER_SOURCE_SHA256)?;

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation)?;
    let marker = tensor(&backend, &[768], &vec![0.0; 768], &context)?;
    let collision = normalize_and_select_architecture(
        NativeAudioEncoderCheckpoint {
            artifact_sha256: "11".repeat(32),
            ordered_state: vec![
                (
                    "wav2vec2.encoder.layer_norm.bias".to_owned(),
                    marker.clone(),
                ),
                ("encoder.layer_norm.bias".to_owned(), marker.clone()),
            ],
            memory_budget_bytes: MEMORY_LIMIT,
        },
        &context,
    );
    assert!(
        matches!(collision, Err(AudioEncoderError::InvalidCheckpoint(message)) if message.contains("collides"))
    );

    let mixed = normalize_and_select_architecture(
        NativeAudioEncoderCheckpoint {
            artifact_sha256: "22".repeat(32),
            ordered_state: vec![
                ("wav2vec2.encoder.layer_norm.bias".to_owned(), marker),
                (
                    "model.encoder.embed_positions.weight".to_owned(),
                    tensor(&backend, &[1], &[0.0], &context)?,
                ),
            ],
            memory_budget_bytes: MEMORY_LIMIT,
        },
        &context,
    );
    assert!(matches!(
        mixed,
        Err(AudioEncoderError::MissingState(key)) if key == "encoder.layer_norm.weight"
    ));

    let marker = tensor(&backend, &[768], &vec![0.0; 768], &context)?;
    let bad_shape = normalize_and_select_architecture(
        NativeAudioEncoderCheckpoint {
            artifact_sha256: "33".repeat(32),
            ordered_state: vec![
                (
                    "wav2vec2.encoder.layer_norm.bias".to_owned(),
                    marker.clone(),
                ),
                (
                    "wav2vec2.encoder.layer_norm.weight".to_owned(),
                    tensor(&backend, &[1], &[0.0], &context)?,
                ),
            ],
            memory_budget_bytes: MEMORY_LIMIT,
        },
        &context,
    );
    assert!(matches!(
        bad_shape,
        Err(AudioEncoderError::StateShape { key, .. }) if key == "encoder.layer_norm.weight"
    ));

    let bad_dtype = normalize_and_select_architecture(
        NativeAudioEncoderCheckpoint {
            artifact_sha256: "44".repeat(32),
            ordered_state: vec![
                (
                    "wav2vec2.encoder.layer_norm.bias".to_owned(),
                    marker.clone(),
                ),
                (
                    "wav2vec2.encoder.layer_norm.weight".to_owned(),
                    tensor_with_dtype(&backend, &[768], &vec![0.0; 768], DType::F16, &context)?,
                ),
            ],
            memory_budget_bytes: MEMORY_LIMIT,
        },
        &context,
    );
    assert!(matches!(
        bad_dtype,
        Err(AudioEncoderError::StatePlacement { key, .. }) if key == "encoder.layer_norm.weight"
    ));

    let other_context = ExecutionContext {
        stream: StreamId::new(7),
        scratch: authority.authorize_workspace(MEMORY_LIMIT)?,
        rng_phase: None,
        cancellation: &cancellation,
    };
    let bad_stream = normalize_and_select_architecture(
        NativeAudioEncoderCheckpoint {
            artifact_sha256: "55".repeat(32),
            ordered_state: vec![
                ("wav2vec2.encoder.layer_norm.bias".to_owned(), marker),
                (
                    "wav2vec2.encoder.layer_norm.weight".to_owned(),
                    tensor(&backend, &[768], &vec![0.0; 768], &other_context)?,
                ),
            ],
            memory_budget_bytes: MEMORY_LIMIT,
        },
        &context,
    );
    assert!(matches!(
        bad_stream,
        Err(AudioEncoderError::StatePlacement { key, .. }) if key == "encoder.layer_norm.weight"
    ));

    let whisper_collision = normalize_and_select_architecture(
        NativeAudioEncoderCheckpoint {
            artifact_sha256: "66".repeat(32),
            ordered_state: vec![
                (
                    "model.encoder.embed_positions.weight".to_owned(),
                    tensor(&backend, &[1], &[0.0], &context)?,
                ),
                (
                    "model.encoder.conv1.bias".to_owned(),
                    tensor(&backend, &[1], &[0.0], &context)?,
                ),
                (
                    "encoder.conv1.bias".to_owned(),
                    tensor(&backend, &[1], &[0.0], &context)?,
                ),
            ],
            memory_budget_bytes: MEMORY_LIMIT,
        },
        &context,
    );
    assert!(matches!(
        whisper_collision,
        Err(AudioEncoderError::InvalidCheckpoint(message)) if message.contains("Whisper")
            && message.contains("collides")
    ));
    Ok(())
}

#[test]
fn audio_encoder_reduced_oracles_are_exact_and_transactional() -> Result<(), Box<dyn Error>> {
    let document: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(
        document.get("format").and_then(Value::as_str),
        Some("zed.sim.audio-encoder-reduced-oracle.v1"),
    );
    let generator_sha256 = format!("{:x}", Sha256::digest(GENERATOR));
    assert_eq!(
        document.get("generator_sha256").and_then(Value::as_str),
        Some(generator_sha256.as_str()),
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(ORACLE.as_bytes())),
        ORACLE_SHA256
    );
    let pinned_sources = document
        .get("pinned_sources")
        .and_then(Value::as_object)
        .ok_or("oracle source provenance is missing")?;
    for (path, expected) in [
        (
            NODES_AUDIO_ENCODER_SOURCE_PATH,
            NODES_AUDIO_ENCODER_SOURCE_SHA256,
        ),
        (AUDIO_ENCODERS_SOURCE_PATH, AUDIO_ENCODERS_SOURCE_SHA256),
        (WAV2VEC2_SOURCE_PATH, WAV2VEC2_SOURCE_SHA256),
        (WHISPER_SOURCE_PATH, WHISPER_SOURCE_SHA256),
    ] {
        assert_eq!(
            pinned_sources.get(path).and_then(Value::as_str),
            Some(expected)
        );
    }
    for identifier in ["wav2vec2-base", "wav2vec2-large", "whisper-large-v3"] {
        let result = document
            .get("results")
            .and_then(|results| results.get(identifier))
            .ok_or("oracle result is missing")?;
        let shape = result
            .get("shape")
            .and_then(Value::as_array)
            .ok_or("oracle shape is missing")?;
        assert_eq!(
            shape.as_slice(),
            [Value::from(2), Value::from(4), Value::from(2)]
        );
        let values = result
            .get("values")
            .and_then(Value::as_array)
            .ok_or("oracle values are missing")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(values.len() * std::mem::size_of::<f32>())?;
        for value in values {
            bytes.extend_from_slice(
                &(value.as_f64().ok_or("oracle value is invalid")? as f32).to_le_bytes(),
            );
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            result
                .get("raw_f32_sha256")
                .and_then(Value::as_str)
                .ok_or("raw hash is missing")?,
        );
    }
    Ok(())
}

#[test]
fn audio_encoder_identity_residency_reconstruction_and_failures_are_atomic()
-> Result<(), Box<dyn Error>> {
    let source = include_str!("../src/audio_encoder.rs");
    for seam in [
        "pub fn from_checkpoint(",
        "pub fn encode(",
        "pub fn reconstruct(",
        "pub const fn identifier(",
        "pub fn artifact_sha256(",
        "pub fn semantic_state_digest_sha256(",
        "pub fn resident_owned_bytes(",
        "pub fn resident_tensor_allocations(",
        "pub fn resident_bytes(",
        "pub fn deterministic_reduced_audio_encoder_fixture(",
    ] {
        assert!(source.contains(seam), "missing ownership seam {seam}");
    }
    assert!(source.contains("AudioEncoderOutput::layered"));
    assert!(source.contains("tensor_var_with_context_exact_native"));
    assert!(source.contains("mel_spectrogram_with_context_exact_native"));
    assert!(source.contains("scaled_dot_product_attention_with_context"));
    Ok(())
}
