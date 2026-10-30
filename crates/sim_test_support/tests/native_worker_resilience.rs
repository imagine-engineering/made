use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use serde_json::json;
use sha2::{Digest, Sha256};
use sim_model::{
    NativeModelRecipeSourceForm, NativeModelResourceIdentity, NativeModelResourceRole,
};
use sim_nodes::{
    NativeModelResourceRequest, NativePortCardinality, NativePrimitive, NativeStoredModelPayload,
    NativeStoredPayload, NativeValue,
};
use sim_runtime::{
    AssetNamespace, AssetRoots, AssetService, AttemptState, CompiledNode, CompiledPlan,
    ExecutionCommandAck, ExecutionCommandOutcome, ExecutionControlCommand,
    ExecutionControlCommandKind, ExecutionDataSource, ExecutionEventBus,
    ExecutionPresentationService, ExecutionSnapshotStatus, InputBinding,
    NATIVE_IMAGE_REGISTRY_VERSION, NativeAssetResolverRegistry, NativeExecutionControllerConfig,
    NativeHandleStoreGeneration, NativeImageExecutor, NativeModelSourceTestSession,
    NativeModelSourceTestTransport, NativeModelSourceTransportHost, NativeModelSourceWorkerBinding,
    NativeNodeContext, PermissionPolicy, PersistedNativeModelResourceEvidence, RetryPromptSource,
    RuntimeSupervisor, RuntimeSupervisorError, SupervisorPolicy, WorkerLaunchConfig,
    authorize_native_api_asset_reader, compile_native_diffusion_workflow,
    execute_reduced_fixture_native_model_resource_for_test,
    exercise_conditioning_control_retry_after_worker_replacement_for_test,
    exercise_media_text_retry_after_worker_replacement_for_test,
    exercise_model_training_retry_after_worker_replacement_for_test,
    exercise_model_transform_retry_after_worker_replacement_for_test,
    exercise_native_audio_retry_after_worker_replacement_for_test,
    exercise_persisted_model_resource_retry_after_worker_replacement_for_test,
    generated_native_node_registry_projection, load_reduced_fixture_native_model_resource_for_test,
    persisted_model_resource_evidence_for_test, publish_reconstructed_native_model_resource,
    record_succeeded_model_resource_attempt_for_test,
    task403_conditioning_control_fixture_descriptor_for_test,
    task404_model_transform_fixture_descriptor_for_test,
    task404_sampling_profile_fixture_descriptor_for_test,
    task405_training_dataset_fixture_descriptor_for_test,
    task405_training_sink_fixture_descriptor_for_test,
    task407_custom_sampler_inputs_fixture_descriptor_for_test,
    task407_sampling_policy_fixture_descriptor_for_test, task410_audio_fixture_descriptor_for_test,
    task415_media_text_fixture_descriptor_for_test,
};
use sim_sampler::{NativeSamplingPolicy, SamplingProfile, SamplingProfileReplacement};
use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, DType, DeviceId, StreamId, TensorBackend,
    TensorDescriptor, TensorError,
};
use sim_test_support::NativeDiffusionFixture;
use sim_types::{
    AttemptId, MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES, NodeId, ProfileId, PromptId, RequestId,
    WorkerId, WorkerModelSourceArtifact, WorkerModelSourceChunk, WorkerModelSourceClosed,
    WorkerModelSourceContext, WorkerModelSourceError, WorkerModelSourceFormat,
    WorkerModelSourceManifest, WorkerModelSourceNestedStateDisposition, WorkerModelSourceOpened,
    WorkerModelSourceOperation, WorkerModelSourceParserLimits, WorkerModelSourceRequest,
    WorkerModelSourceResponse, WorkerModelSourceTensor, WorkerSha256Digest,
    worker_model_source_selection_sha256,
};
use smol::process::{Child, Command};
use uuid::Uuid;

#[path = "support/native_controller.rs"]
mod native_controller;

const MEMORY_LIMIT_BYTES: u64 = 1024 * 1024 * 1024;

fn bridge_model_source_call(
    host: &NativeModelSourceTransportHost,
    session: &NativeModelSourceTestSession,
    registry: &NativeAssetResolverRegistry,
    expected_context: &WorkerModelSourceContext,
    request: WorkerModelSourceRequest,
    cancellation: &CancellationToken,
) -> Result<WorkerModelSourceResponse, Box<dyn Error>> {
    thread::scope(|scope| {
        let worker_call = scope.spawn(|| session.call(request, cancellation));
        let call = smol::block_on(host.receive())?;
        let call_id = call.call_id();
        let response = registry
            .test_serve_model_source_request(expected_context, call.request(), cancellation)
            .or_else(|error| {
                WorkerModelSourceResponse::rejected(
                    call.request().context.session_id,
                    call.request().call_ordinal,
                    error,
                )
            })?;
        smol::block_on(call.respond(call_id, response))?;
        worker_call
            .join()
            .map_err(|_| "model-source worker call panicked")?
            .map_err(Into::into)
    })
}

#[test]
fn val_recovery_005() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_id = "model-source-restart";
    let roots = [
        AssetNamespace::Input,
        AssetNamespace::Output,
        AssetNamespace::Temporary,
        AssetNamespace::Model,
        AssetNamespace::Plugin,
    ]
    .into_iter()
    .map(|namespace| {
        let root = directory.path().join(namespace.locator_type());
        fs::create_dir_all(&root)?;
        Ok((namespace, root))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new(profile_id, roots)?;
    let model_path = roots
        .test_root_path(AssetNamespace::Model)?
        .join("checkpoints/large.safetensors");
    let model_parent = model_path.parent().ok_or("model parent is unavailable")?;
    fs::create_dir_all(model_parent)?;
    let tensor_bytes = 13 * 1024 * 1024_usize;
    let header = format!(
        r#"{{"large":{{"dtype":"U8","shape":[{tensor_bytes}],"data_offsets":[0,{tensor_bytes}]}}}}"#
    );
    let mut model_file = fs::File::create(&model_path)?;
    model_file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    model_file.write_all(header.as_bytes())?;
    model_file.write_all(&vec![0x5a; tensor_bytes])?;
    model_file.sync_all()?;

    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    let authorization = authorize_native_api_asset_reader(&policy)?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
    let registry = NativeAssetResolverRegistry::new(assets, authorization);
    let attempt_id = AttemptId(Uuid::from_u128(0x39921));
    let source_names = vec!["large.safetensors".to_owned()];
    let selection = worker_model_source_selection_sha256("checkpoints", &source_names)?;
    let context = |session_id, service_generation| WorkerModelSourceContext {
        session_id,
        attempt_id,
        attempt_generation: 1,
        node_id: "checkpoint-loader".to_owned(),
        node_generation: 1,
        service_id: Uuid::from_u128(0x39922),
        service_generation,
        ordered_source_identity_sha256: selection.clone(),
    };
    let open_request = |context: WorkerModelSourceContext| WorkerModelSourceRequest {
        context,
        call_ordinal: 1,
        operation: WorkerModelSourceOperation::Open {
            folder_category: "checkpoints".to_owned(),
            source_names: source_names.clone(),
        },
    };
    let cancellation = CancellationToken::default();
    let (transport, host) = NativeModelSourceTestTransport::channel();

    let lost_context = context(Uuid::from_u128(0x39923), 1);
    let lost_session = transport.open_session(lost_context.clone())?;
    let lost_open = open_request(lost_context.clone());
    let opened = bridge_model_source_call(
        &host,
        &lost_session,
        &registry,
        &lost_context,
        lost_open,
        &cancellation,
    )?;
    let WorkerModelSourceResponse::Opened(opened) = opened else {
        return Err("model-source open did not return a manifest".into());
    };
    assert_eq!(opened.sources.len(), 1);
    assert_eq!(
        opened.sources[0].aggregate_tensor_bytes,
        u64::try_from(tensor_bytes)?
    );
    assert!(
        !serde_json::to_vec(&opened)?
            .windows(directory.path().as_os_str().len())
            .any(|window| window == directory.path().as_os_str().as_encoded_bytes())
    );
    assert!(tensor_bytes > 12 * 1024 * 1024);
    assert!(
        serde_json::to_vec(&open_request(lost_context.clone()))?.len()
            < sim_types::MAX_WORKER_FRAME_BYTES
    );
    let first_read = WorkerModelSourceRequest {
        context: lost_context.clone(),
        call_ordinal: 2,
        operation: WorkerModelSourceOperation::Read {
            source_ordinal: 0,
            tensor_ordinal: 0,
            byte_offset: 0,
            byte_length: u32::try_from(MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)?,
        },
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &lost_session,
            &registry,
            &lost_context,
            first_read.clone(),
            &cancellation,
        )?,
        WorkerModelSourceResponse::Chunk(_)
    ));

    registry.test_replace_model_source_service();
    let late_read = WorkerModelSourceRequest {
        call_ordinal: 3,
        ..first_read
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &lost_session,
            &registry,
            &lost_context,
            late_read,
            &cancellation,
        )?,
        WorkerModelSourceResponse::Rejected {
            error: WorkerModelSourceError::Closed,
            ..
        }
    ));

    let retry_context = context(Uuid::from_u128(0x39924), 2);
    let retry_session = transport.open_session(retry_context.clone())?;
    let retry_open = open_request(retry_context.clone());
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &retry_session,
            &registry,
            &retry_context,
            retry_open,
            &cancellation,
        )?,
        WorkerModelSourceResponse::Opened(_)
    ));
    let mut bridged_bytes = 0_usize;
    let chunk_count = tensor_bytes / MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES;
    for chunk_index in 0..chunk_count {
        let request = WorkerModelSourceRequest {
            context: retry_context.clone(),
            call_ordinal: u64::try_from(chunk_index)? + 2,
            operation: WorkerModelSourceOperation::Read {
                source_ordinal: 0,
                tensor_ordinal: 0,
                byte_offset: u64::try_from(chunk_index * MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)?,
                byte_length: u32::try_from(MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)?,
            },
        };
        let WorkerModelSourceResponse::Chunk(chunk) = bridge_model_source_call(
            &host,
            &retry_session,
            &registry,
            &retry_context,
            request,
            &cancellation,
        )?
        else {
            return Err("model-source read did not return a chunk".into());
        };
        assert_eq!(chunk.bytes, vec![0x5a; MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES]);
        bridged_bytes = bridged_bytes
            .checked_add(chunk.bytes.len())
            .ok_or("bridged byte count overflowed")?;
    }
    assert_eq!(bridged_bytes, tensor_bytes);
    let close_ordinal = u64::try_from(chunk_count)? + 2;
    let close = WorkerModelSourceRequest {
        context: retry_context.clone(),
        call_ordinal: close_ordinal,
        operation: WorkerModelSourceOperation::Close,
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &retry_session,
            &registry,
            &retry_context,
            close,
            &cancellation,
        )?,
        WorkerModelSourceResponse::Closed(_)
    ));
    registry.retire_attempt(attempt_id);
    Ok(())
}

#[test]
fn model_source_worker_bridge_restarts_without_duplicate_publication() -> Result<(), Box<dyn Error>>
{
    val_recovery_005()
}

#[test]
fn model_source_auxiliary_artifact_stream_restarts_atomically() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_id = "model-source-artifact-restart";
    let roots = [
        AssetNamespace::Input,
        AssetNamespace::Output,
        AssetNamespace::Temporary,
        AssetNamespace::Model,
        AssetNamespace::Plugin,
    ]
    .into_iter()
    .map(|namespace| {
        let root = directory.path().join(namespace.locator_type());
        fs::create_dir_all(&root)?;
        Ok((namespace, root))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new(profile_id, roots)?;
    let model_root = roots
        .test_root_path(AssetNamespace::Model)?
        .join("checkpoints");
    fs::create_dir_all(&model_root)?;
    let shard_header = r#"{"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    let mut shard = fs::File::create(model_root.join("weights.safetensors"))?;
    shard.write_all(&u64::try_from(shard_header.len())?.to_le_bytes())?;
    shard.write_all(shard_header.as_bytes())?;
    shard.write_all(&[1, 2, 3, 4])?;
    shard.sync_all()?;
    let metadata = (0..40_000)
        .map(|ordinal| {
            (
                format!("token-{ordinal:05}"),
                serde_json::Value::String(format!("fixture-{ordinal:05}")),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let config = serde_json::to_vec(&json!({
        "metadata": metadata,
        "weight_map": { "weight": "weights.safetensors" }
    }))?;
    assert!(config.len() > MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES);
    fs::write(model_root.join("model.safetensors.index.json"), &config)?;

    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    let authorization = authorize_native_api_asset_reader(&policy)?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
    let registry = NativeAssetResolverRegistry::new(assets, authorization);
    let attempt_id = AttemptId(Uuid::from_u128(0x73321));
    let source_names = vec!["model.safetensors.index.json".to_owned()];
    let selection = worker_model_source_selection_sha256("checkpoints", &source_names)?;
    let context = |session_id, service_generation| WorkerModelSourceContext {
        session_id,
        attempt_id,
        attempt_generation: 1,
        node_id: "checkpoint-config-loader".to_owned(),
        node_generation: 1,
        service_id: Uuid::from_u128(0x73322),
        service_generation,
        ordered_source_identity_sha256: selection.clone(),
    };
    let open_request = |context: WorkerModelSourceContext| WorkerModelSourceRequest {
        context,
        call_ordinal: 1,
        operation: WorkerModelSourceOperation::Open {
            folder_category: "checkpoints".to_owned(),
            source_names: source_names.clone(),
        },
    };
    let cancellation = CancellationToken::default();
    let (transport, host) = NativeModelSourceTestTransport::channel();

    let lost_context = context(Uuid::from_u128(0x73323), 1);
    let lost_session = transport.open_session(lost_context.clone())?;
    let lost_open_response = bridge_model_source_call(
        &host,
        &lost_session,
        &registry,
        &lost_context,
        open_request(lost_context.clone()),
        &cancellation,
    )?;
    let WorkerModelSourceResponse::Opened(lost_opened) = lost_open_response else {
        return Err(format!(
            "model-source artifact open did not return a manifest: {lost_open_response:?}"
        )
        .into());
    };
    let lost_source = lost_opened
        .sources
        .first()
        .ok_or("model-source artifact manifest is empty")?;
    let lost_artifact = lost_source
        .artifacts
        .first()
        .ok_or("model-source config artifact is missing")?
        .clone();
    assert_eq!(lost_artifact.format, WorkerModelSourceFormat::JsonConfig);
    assert_eq!(
        lost_source.artifacts.get(1).map(|artifact| artifact.format),
        Some(WorkerModelSourceFormat::Safetensors)
    );
    assert_eq!(lost_source.tensors.len(), 1);
    assert_eq!(lost_source.aggregate_tensor_bytes, 4);
    let lost_tensor = lost_source
        .tensors
        .first()
        .ok_or("model-source shard tensor is missing")?;
    assert_eq!(lost_tensor.artifact_ordinal, 1);
    assert_eq!(lost_tensor.byte_length, 4);
    assert_eq!(lost_source.model_identity_sha256.as_str().len(), 64);
    assert!(!lost_source.model_identity_sha256.as_str().contains(':'));
    assert_ne!(lost_source.model_identity_sha256, lost_artifact.sha256);
    let first_read = WorkerModelSourceRequest {
        context: lost_context.clone(),
        call_ordinal: 2,
        operation: WorkerModelSourceOperation::ReadArtifact {
            source_ordinal: 0,
            artifact_ordinal: 0,
            artifact_sha256: lost_artifact.sha256.clone(),
            byte_offset: 0,
            byte_length: u32::try_from(MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)?,
        },
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &lost_session,
            &registry,
            &lost_context,
            first_read.clone(),
            &cancellation,
        )?,
        WorkerModelSourceResponse::ArtifactChunk(_)
    ));
    registry.test_replace_model_source_service();
    let late_read = WorkerModelSourceRequest {
        call_ordinal: 3,
        ..first_read
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &lost_session,
            &registry,
            &lost_context,
            late_read,
            &cancellation,
        )?,
        WorkerModelSourceResponse::Rejected {
            error: WorkerModelSourceError::Closed,
            ..
        }
    ));

    let retry_context = context(Uuid::from_u128(0x73324), 2);
    let retry_session = transport.open_session(retry_context.clone())?;
    let retry_open_response = bridge_model_source_call(
        &host,
        &retry_session,
        &registry,
        &retry_context,
        open_request(retry_context.clone()),
        &cancellation,
    )?;
    let WorkerModelSourceResponse::Opened(retry_opened) = retry_open_response else {
        return Err(format!(
            "model-source artifact retry did not return a manifest: {retry_open_response:?}"
        )
        .into());
    };
    let retry_source = retry_opened
        .sources
        .first()
        .ok_or("model-source retry manifest is empty")?;
    let retry_artifact = retry_source
        .artifacts
        .first()
        .ok_or("model-source retry config artifact is missing")?
        .clone();
    assert_eq!(retry_artifact.sha256, lost_artifact.sha256);
    assert_eq!(
        retry_source.model_identity_sha256,
        lost_source.model_identity_sha256
    );
    let mut bridged = Vec::new();
    let mut byte_offset = 0_usize;
    let mut call_ordinal = 2_u64;
    while byte_offset < config.len() {
        let byte_length = (config.len() - byte_offset).min(MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES);
        let request = WorkerModelSourceRequest {
            context: retry_context.clone(),
            call_ordinal,
            operation: WorkerModelSourceOperation::ReadArtifact {
                source_ordinal: 0,
                artifact_ordinal: 0,
                artifact_sha256: retry_artifact.sha256.clone(),
                byte_offset: u64::try_from(byte_offset)?,
                byte_length: u32::try_from(byte_length)?,
            },
        };
        let WorkerModelSourceResponse::ArtifactChunk(chunk) = bridge_model_source_call(
            &host,
            &retry_session,
            &registry,
            &retry_context,
            request,
            &cancellation,
        )?
        else {
            return Err("model-source artifact read did not return a chunk".into());
        };
        assert_eq!(chunk.artifact_sha256, retry_artifact.sha256);
        bridged.extend_from_slice(&chunk.bytes);
        byte_offset = byte_offset
            .checked_add(byte_length)
            .ok_or("artifact bridge offset overflowed")?;
        call_ordinal = call_ordinal
            .checked_add(1)
            .ok_or("artifact bridge call ordinal overflowed")?;
    }
    assert_eq!(bridged, config);
    let close = WorkerModelSourceRequest {
        context: retry_context.clone(),
        call_ordinal,
        operation: WorkerModelSourceOperation::Close,
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &retry_session,
            &registry,
            &retry_context,
            close,
            &cancellation,
        )?,
        WorkerModelSourceResponse::Closed(_)
    ));
    registry.retire_attempt(attempt_id);
    Ok(())
}

struct RestartModelResourceSource {
    manifest: WorkerModelSourceManifest,
    tensors: Vec<Vec<u8>>,
    auxiliary_artifacts: BTreeMap<u32, Vec<u8>>,
}

fn restart_clip_state() -> Vec<(String, Vec<u64>)> {
    let mut state = vec![
        (
            "text_model.embeddings.token_embedding.weight".to_owned(),
            vec![49_408, 2],
        ),
        (
            "text_model.embeddings.position_embedding.weight".to_owned(),
            vec![77, 2],
        ),
        ("text_model.final_layer_norm.weight".to_owned(), vec![2]),
        ("text_model.final_layer_norm.bias".to_owned(), vec![2]),
        ("text_projection.weight".to_owned(), vec![2, 2]),
    ];
    for layer in 0..2 {
        for (name, shape) in [
            ("layer_norm1.weight", vec![2]),
            ("layer_norm1.bias", vec![2]),
            ("self_attn.q_proj.weight", vec![2, 2]),
            ("self_attn.q_proj.bias", vec![2]),
            ("self_attn.k_proj.weight", vec![2, 2]),
            ("self_attn.k_proj.bias", vec![2]),
            ("self_attn.v_proj.weight", vec![2, 2]),
            ("self_attn.v_proj.bias", vec![2]),
            ("self_attn.out_proj.weight", vec![2, 2]),
            ("self_attn.out_proj.bias", vec![2]),
            ("layer_norm2.weight", vec![2]),
            ("layer_norm2.bias", vec![2]),
            ("mlp.fc1.weight", vec![2, 2]),
            ("mlp.fc1.bias", vec![2]),
            ("mlp.fc2.weight", vec![2, 2]),
            ("mlp.fc2.bias", vec![2]),
        ] {
            state.push((format!("text_model.encoder.layers.{layer}.{name}"), shape));
        }
    }
    state
}

fn restart_clip_tokenizer() -> Result<Vec<u8>, Box<dyn Error>> {
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
    Ok(serde_json::to_vec(&json!({
        "model": { "vocab": vocabulary, "merges": merges }
    }))?)
}

fn restart_clip_source(
    source_ordinal: u32,
    label: &str,
    projection_dimension: u64,
) -> Result<RestartModelResourceSource, Box<dyn Error>> {
    let state = restart_clip_state();
    let mut tensors = Vec::new();
    let mut tensor_manifests = Vec::new();
    let mut aggregate_tensor_bytes = 0_u64;
    let mut maximum_tensor_bytes = 0_u64;
    for (key, shape) in state {
        let element_count = shape.iter().try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or("CLIP restart fixture shape overflow")
        })?;
        let byte_length = element_count
            .checked_mul(DType::F32.byte_width())
            .ok_or("CLIP restart fixture tensor byte count overflow")?;
        let byte_offset = aggregate_tensor_bytes;
        aggregate_tensor_bytes = aggregate_tensor_bytes
            .checked_add(byte_length)
            .ok_or("CLIP restart fixture aggregate overflow")?;
        maximum_tensor_bytes = maximum_tensor_bytes.max(byte_length);
        tensors.push(vec![0; usize::try_from(byte_length)?]);
        tensor_manifests.push(WorkerModelSourceTensor {
            name: key,
            data_type: "F32".to_owned(),
            shape,
            artifact_ordinal: 0,
            byte_offset,
            byte_length,
        });
    }
    let configuration = serde_json::to_vec(&json!({
        "model_type": "clip",
        "hidden_size": 2,
        "projection_dim": projection_dimension,
    }))?;
    let tokenizer = restart_clip_tokenizer()?;
    let artifact_sha256 = WorkerSha256Digest::new(format!(
        "{:x}",
        Sha256::digest(format!("task401:restart:{label}:artifact"))
    ))?;
    let configuration_sha256 =
        WorkerSha256Digest::new(format!("{:x}", Sha256::digest(&configuration)))?;
    let tokenizer_sha256 = WorkerSha256Digest::new(format!("{:x}", Sha256::digest(&tokenizer)))?;
    let maximum_read_bytes = maximum_tensor_bytes.max(512 * 1024);
    Ok(RestartModelResourceSource {
        manifest: WorkerModelSourceManifest {
            source_ordinal,
            model_identity_sha256: WorkerSha256Digest::new(format!(
                "{:x}",
                Sha256::digest(format!("task401:restart:{label}:model"))
            ))?,
            artifacts: vec![
                WorkerModelSourceArtifact {
                    sha256: artifact_sha256,
                    byte_size: aggregate_tensor_bytes,
                    format: WorkerModelSourceFormat::Safetensors,
                    nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
                },
                WorkerModelSourceArtifact {
                    sha256: configuration_sha256,
                    byte_size: u64::try_from(configuration.len())?,
                    format: WorkerModelSourceFormat::ClipConfiguration { target_ordinal: 0 },
                    nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
                },
                WorkerModelSourceArtifact {
                    sha256: tokenizer_sha256,
                    byte_size: u64::try_from(tokenizer.len())?,
                    format: WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal: 0 },
                    nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
                },
            ],
            tensors: tensor_manifests,
            aggregate_tensor_bytes,
            maximum_read_bytes,
            parser_limits: WorkerModelSourceParserLimits {
                version: 1,
                manifest_bytes: 16 * 1024 * 1024,
                maximum_depth: 64,
                maximum_tensors: 10_000,
                maximum_tensor_bytes: maximum_read_bytes,
                maximum_aggregate_tensor_bytes: aggregate_tensor_bytes.max(512 * 1024),
                maximum_name_bytes: 16 * 1024,
                maximum_archive_entries: 10_000,
                maximum_metadata_values: 1_000_000,
            },
        },
        tensors,
        auxiliary_artifacts: BTreeMap::from([(1, configuration), (2, tokenizer)]),
    })
}

fn serve_restart_model_resource(
    host: NativeModelSourceTransportHost,
    sources: Arc<Vec<RestartModelResourceSource>>,
    interrupt_on_first_artifact: bool,
) -> Result<(usize, usize, usize), WorkerModelSourceError> {
    let mut active_session = None;
    let mut counts = (0, 0, 0);
    while let Ok(call) = smol::block_on(host.receive()) {
        let request = call.request().clone();
        let response = match request.operation {
            WorkerModelSourceOperation::Open {
                folder_category,
                source_names,
            } => {
                counts.0 += 1;
                if request.call_ordinal != 1
                    || folder_category != "text_encoders"
                    || source_names != ["clip-l.safetensors", "clip-g.safetensors"]
                    || active_session.replace(request.context.session_id).is_some()
                {
                    return Err(WorkerModelSourceError::InvalidOrder);
                }
                WorkerModelSourceResponse::Opened(WorkerModelSourceOpened::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    request.context.ordered_source_identity_sha256.clone(),
                    sources
                        .iter()
                        .map(|source| source.manifest.clone())
                        .collect(),
                )?)
            }
            WorkerModelSourceOperation::Read {
                source_ordinal,
                tensor_ordinal,
                byte_offset,
                byte_length,
            } => {
                counts.1 += 1;
                if active_session != Some(request.context.session_id) || source_ordinal != 0 {
                    if active_session != Some(request.context.session_id) {
                        return Err(WorkerModelSourceError::Closed);
                    }
                }
                let source = sources
                    .get(
                        usize::try_from(source_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                let tensor = source
                    .tensors
                    .get(
                        usize::try_from(tensor_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidTensorOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidTensorOrdinal)?;
                let start =
                    usize::try_from(byte_offset).map_err(|_| WorkerModelSourceError::RangeLimit)?;
                let end = start
                    .checked_add(
                        usize::try_from(byte_length)
                            .map_err(|_| WorkerModelSourceError::RangeLimit)?,
                    )
                    .ok_or(WorkerModelSourceError::RangeLimit)?;
                let bytes = tensor
                    .get(start..end)
                    .ok_or(WorkerModelSourceError::RangeLimit)?
                    .to_vec();
                WorkerModelSourceResponse::Chunk(WorkerModelSourceChunk::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    source_ordinal,
                    tensor_ordinal,
                    byte_offset,
                    bytes,
                )?)
            }
            WorkerModelSourceOperation::Close => {
                counts.2 += 1;
                if active_session.take() != Some(request.context.session_id) {
                    return Err(WorkerModelSourceError::Closed);
                }
                WorkerModelSourceResponse::Closed(WorkerModelSourceClosed::checked(
                    request.context.session_id,
                    request.call_ordinal,
                )?)
            }
            WorkerModelSourceOperation::ReadArtifact {
                source_ordinal,
                artifact_ordinal,
                artifact_sha256,
                byte_offset,
                byte_length,
            } => {
                counts.1 += 1;
                if active_session != Some(request.context.session_id) {
                    return Err(WorkerModelSourceError::Closed);
                }
                if interrupt_on_first_artifact {
                    return Ok(counts);
                }
                let source = sources
                    .get(
                        usize::try_from(source_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                let artifact = source
                    .manifest
                    .artifacts
                    .get(
                        usize::try_from(artifact_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceOrdinal)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                if artifact.sha256 != artifact_sha256 {
                    return Err(WorkerModelSourceError::DigestMismatch);
                }
                let bytes = source
                    .auxiliary_artifacts
                    .get(&artifact_ordinal)
                    .ok_or(WorkerModelSourceError::InvalidSourceOrdinal)?;
                let start =
                    usize::try_from(byte_offset).map_err(|_| WorkerModelSourceError::RangeLimit)?;
                let end = start
                    .checked_add(
                        usize::try_from(byte_length)
                            .map_err(|_| WorkerModelSourceError::RangeLimit)?,
                    )
                    .ok_or(WorkerModelSourceError::RangeLimit)?;
                let chunk = bytes
                    .get(start..end)
                    .ok_or(WorkerModelSourceError::RangeLimit)?
                    .to_vec();
                WorkerModelSourceResponse::ArtifactChunk(
                    sim_types::WorkerModelSourceArtifactChunk::checked(
                        request.context.session_id,
                        request.call_ordinal,
                        source_ordinal,
                        artifact_ordinal,
                        artifact_sha256,
                        byte_offset,
                        chunk,
                    )?,
                )
            }
        };
        let call_id = call.call_id();
        smol::block_on(call.respond(call_id, response))?;
    }
    if active_session.is_none() {
        Ok(counts)
    } else {
        Err(WorkerModelSourceError::Closed)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_restart_model_resource<R>(
    exercise_execution_boundary: bool,
    prompt_id: PromptId,
    attempt_id: AttemptId,
    node_id: NodeId,
    cancellation: CancellationToken,
    scratch: sim_tensor::ScratchReservation,
    handle_store: Arc<dyn sim_nodes::NativeHandleStore>,
    binding: NativeModelSourceWorkerBinding,
    backend: Arc<sim_tensor::CpuBackend>,
    request: NativeModelResourceRequest,
    host_operation: impl FnOnce(NativeModelSourceTransportHost) -> Result<R, WorkerModelSourceError>
    + Send
    + 'static,
) -> Result<
    (
        Result<
            (
                Arc<sim_model::NativeModelPayload>,
                Option<sim_runtime::PersistedNativeModelResourceEvidence>,
            ),
            String,
        >,
        R,
    ),
    Box<dyn Error>,
>
where
    R: Send + 'static,
{
    if exercise_execution_boundary {
        let (result, host_result) = execute_reduced_fixture_native_model_resource_for_test(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            binding,
            backend,
            request,
            host_operation,
        )?;
        Ok((
            result
                .map(|(payload, evidence)| (payload, Some(evidence)))
                .map_err(|error| error.message),
            host_result,
        ))
    } else {
        let (result, host_result) = load_reduced_fixture_native_model_resource_for_test(
            prompt_id,
            attempt_id,
            node_id,
            cancellation,
            scratch,
            handle_store,
            binding,
            backend,
            request,
            host_operation,
        )?;
        Ok((
            result
                .map(|payload| (payload, None))
                .map_err(|error| error.to_string()),
            host_result,
        ))
    }
}

#[test]
fn model_resource_recipe_restart_is_atomic() -> Result<(), Box<dyn Error>> {
    run_model_resource_recipe_restart_is_atomic(false)
}

#[test]
fn model_resource_execution_restarts_with_fresh_generation() -> Result<(), Box<dyn Error>> {
    run_model_resource_recipe_restart_is_atomic(true)?;
    persisted_model_resource_retry_replaces_real_worker()
}

fn task403_worker_plan(prompt_id: PromptId, variant: u64) -> Result<CompiledPlan, Box<dyn Error>> {
    let vae_node_id = NodeId("task403-vae-loader".to_owned());
    let control_node_id = NodeId("task403-conditioning-control".to_owned());
    let registry = generated_native_node_registry_projection(None)?;
    let vae_descriptor = registry
        .descriptor("VAELoader")
        .cloned()
        .ok_or("generated registry omitted VAELoader")?;
    let control_descriptor = task403_conditioning_control_fixture_descriptor_for_test()?;
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([
            (
                vae_node_id.clone(),
                CompiledNode {
                    id: vae_node_id.clone(),
                    class_type: "VAELoader".to_owned(),
                    descriptor: vae_descriptor,
                    inputs: BTreeMap::from([(
                        "vae_name".to_owned(),
                        InputBinding::Literal {
                            value: NativeValue::Primitive {
                                value: NativePrimitive::String(
                                    "task403-vae.safetensors".to_owned(),
                                ),
                            },
                        },
                    )]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                control_node_id.clone(),
                CompiledNode {
                    id: control_node_id.clone(),
                    class_type: "Task403ConditioningControlFixture".to_owned(),
                    descriptor: control_descriptor,
                    inputs: BTreeMap::from([
                        (
                            "vae".to_owned(),
                            InputBinding::Link {
                                source: vae_node_id.clone(),
                                output_index: 0,
                                lazy: false,
                                mode: NativePortCardinality::Scalar,
                            },
                        ),
                        (
                            "variant".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::Primitive {
                                    value: NativePrimitive::UnsignedInteger(variant),
                                },
                            },
                        ),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
        ]),
        topological_order: vec![vae_node_id.clone(), control_node_id.clone()],
        static_required_nodes: BTreeSet::from([vae_node_id, control_node_id.clone()]),
        output_nodes: vec![control_node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()
        .map_err(|error| format!("Task403 worker plan is invalid: {error}"))?;
    Ok(plan)
}

fn task403_worker_recovery_fixture(
    replacement_variant: Option<u64>,
) -> Result<sim_runtime::NativeConditioningControlWorkerReplacementEvidence, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_uuid = Uuid::new_v4();
    let roots = [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ]
    .into_iter()
    .map(|(namespace, name)| {
        let path = directory.path().join(name);
        fs::create_dir_all(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new(profile_uuid.to_string(), roots)?;
    let vae_directory = roots.test_root_path(AssetNamespace::Model)?.join("vae");
    fs::create_dir_all(&vae_directory)?;
    write_task403_vae_fixture(&vae_directory.join("task403-vae.safetensors"))?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
    let profile_id = ProfileId(profile_uuid);
    let prompt_id = PromptId(Uuid::new_v4());
    let prior_attempt_id = AttemptId(Uuid::new_v4());
    let plan = task403_worker_plan(prompt_id, 0)?;
    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(16, prior_attempt_id)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let queue = ExecutionControlCommand {
        request_id: RequestId(Uuid::new_v4()),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Queue {
            plan: plan,
            priority: 0,
            front: false,
        },
    };
    presentation.submit(queue.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: queue.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(prior_attempt_id),
        },
    })?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
    let mut worker = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        profile_id,
        WorkerId(Uuid::new_v4()),
        NATIVE_IMAGE_REGISTRY_VERSION,
        16 * 1024 * 1024 * 1024,
    );
    worker.working_directory = Some(directory.path().to_path_buf());
    worker.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(30),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let config = NativeExecutionControllerConfig::new(assets, presentation, worker, true)?
        .with_reduced_fixture_model_resources_for_test();
    let replacement_plan = replacement_variant
        .map(|variant| task403_worker_plan(prompt_id, variant))
        .transpose()?;
    Ok(
        exercise_conditioning_control_retry_after_worker_replacement_for_test(
            config,
            ExecutionEventBus::new(64)?,
            replacement_plan,
        )?,
    )
}

fn task410_worker_plan(prompt_id: PromptId) -> Result<CompiledPlan, Box<dyn Error>> {
    let node_id = NodeId("task410-audio".to_owned());
    let descriptor = task410_audio_fixture_descriptor_for_test()?;
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: descriptor.class_type.clone(),
                descriptor,
                inputs: BTreeMap::from([(
                    "sample_rate".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Integer(1),
                        },
                    },
                )]),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    Ok(plan)
}

#[test]
fn task410_rate_one_empty_audio_reconstructs_in_a_replacement_worker() -> Result<(), Box<dyn Error>>
{
    let directory = tempfile::tempdir()?;
    let profile_uuid = Uuid::new_v4();
    let roots = [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ]
    .into_iter()
    .map(|(namespace, name)| {
        let path = directory.path().join(name);
        fs::create_dir_all(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let assets = Arc::new(Mutex::new(AssetService::open(AssetRoots::new(
        profile_uuid.to_string(),
        roots,
    )?)?));
    let profile_id = ProfileId(profile_uuid);
    let prompt_id = PromptId(Uuid::new_v4());
    let prior_attempt_id = AttemptId(Uuid::new_v4());
    let plan = task410_worker_plan(prompt_id)?;
    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(16, prior_attempt_id)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let queue = ExecutionControlCommand {
        request_id: RequestId(Uuid::new_v4()),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Queue {
            plan,
            priority: 0,
            front: false,
        },
    };
    presentation.submit(queue.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: queue.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(prior_attempt_id),
        },
    })?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
    let mut worker = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        profile_id,
        WorkerId(Uuid::new_v4()),
        NATIVE_IMAGE_REGISTRY_VERSION,
        MEMORY_LIMIT_BYTES,
    );
    worker.working_directory = Some(directory.path().to_path_buf());
    worker.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(30),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let config = NativeExecutionControllerConfig::new(assets, presentation, worker, true)?
        .with_reduced_fixture_model_resources_for_test();
    let recovered = exercise_native_audio_retry_after_worker_replacement_for_test(
        config,
        ExecutionEventBus::new(64)?,
    )?;
    assert_ne!(
        recovered.original_process_id,
        recovered.replacement_process_id
    );
    assert_eq!(recovered.reconstruction_count, 1);
    assert!(recovered.durable_transport_recovered);
    assert!(recovered.audio_list_preserved);
    assert!(recovered.stale_handle_wrong_generation);
    assert_eq!(recovered.terminal_state, AttemptState::Succeeded);
    assert!(recovered.failure_message.is_none());
    Ok(())
}

fn task415_worker_plan(prompt_id: PromptId) -> Result<CompiledPlan, Box<dyn Error>> {
    let node_id = NodeId("task415-media-text".to_owned());
    let descriptor = task415_media_text_fixture_descriptor_for_test()?;
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: descriptor.class_type.clone(),
                descriptor,
                inputs: BTreeMap::new(),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    Ok(plan)
}

#[test]
fn task415_source_pixels_repeat_in_a_replacement_worker() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_uuid = Uuid::new_v4();
    let roots = [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ]
    .into_iter()
    .map(|(namespace, name)| {
        let path = directory.path().join(name);
        fs::create_dir_all(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let assets = Arc::new(Mutex::new(AssetService::open(AssetRoots::new(
        profile_uuid.to_string(),
        roots,
    )?)?));
    let profile_id = ProfileId(profile_uuid);
    let prompt_id = PromptId(Uuid::new_v4());
    let prior_attempt_id = AttemptId(Uuid::new_v4());
    let plan = task415_worker_plan(prompt_id)?;
    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(16, prior_attempt_id)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let queue = ExecutionControlCommand {
        request_id: RequestId(Uuid::new_v4()),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Queue {
            plan,
            priority: 0,
            front: false,
        },
    };
    presentation.submit(queue.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: queue.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(prior_attempt_id),
        },
    })?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
    let mut worker = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        profile_id,
        WorkerId(Uuid::new_v4()),
        NATIVE_IMAGE_REGISTRY_VERSION,
        MEMORY_LIMIT_BYTES,
    );
    worker.working_directory = Some(directory.path().to_path_buf());
    worker.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(30),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let config = NativeExecutionControllerConfig::new(assets, presentation, worker, true)?
        .with_reduced_fixture_model_resources_for_test();
    let recovered = exercise_media_text_retry_after_worker_replacement_for_test(
        config,
        ExecutionEventBus::new(64)?,
    )?;
    assert_ne!(
        recovered.original_process_id,
        recovered.replacement_process_id
    );
    const SOURCE_PIXEL_SHA256: &str =
        "a30f46a61f5e2c4c91ecbdf50f1c3b138c26c3bf5da4a63f7eee6b27ad27e075";
    assert_eq!(recovered.original_pixel_sha256, SOURCE_PIXEL_SHA256);
    assert_eq!(recovered.replacement_pixel_sha256, SOURCE_PIXEL_SHA256);
    assert_eq!(recovered.terminal_state, AttemptState::Succeeded);
    assert!(recovered.failure_message.is_none());
    Ok(())
}

fn write_task403_vae_fixture(destination: &Path) -> Result<(), Box<dyn Error>> {
    let output_data = [0.0_f32, 0.25, 0.5, 0.75]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let output_header = json!({
        "taesd_decoder.1.weight": {
            "dtype": "F32",
            "shape": [1, 4],
            "data_offsets": [0, output_data.len()],
        },
    });
    let mut encoded_header = serde_json::to_vec(&output_header)?;
    while encoded_header.len() % 8 != 0 {
        encoded_header.push(b' ');
    }
    let mut encoded = Vec::with_capacity(8 + encoded_header.len() + output_data.len());
    encoded.extend_from_slice(&u64::try_from(encoded_header.len())?.to_le_bytes());
    encoded.extend_from_slice(&encoded_header);
    encoded.extend_from_slice(&output_data);
    fs::write(destination, encoded)?;
    Ok(())
}

#[test]
fn task403_conditioning_control_retry_replays_in_a_replacement_worker() -> Result<(), Box<dyn Error>>
{
    let recovered = task403_worker_recovery_fixture(None)?;
    assert_ne!(
        recovered.original_process_id,
        recovered.replacement_process_id
    );
    assert_eq!(recovered.reconstruction_count, 1);
    assert!(recovered.fresh_output_generation);
    assert!(recovered.positive_negative_distinct);
    assert!(recovered.predecessor_order_preserved);
    assert!(recovered.vae_rebind_preserved);
    assert!(recovered.stale_handle_wrong_generation);
    assert_eq!(recovered.terminal_state, AttemptState::Succeeded);
    assert!(recovered.failure_message.is_none());
    Ok(())
}

#[test]
fn task403_conditioning_control_retry_rejects_changed_durable_inputs_without_publication()
-> Result<(), Box<dyn Error>> {
    let rejected = task403_worker_recovery_fixture(Some(1))?;
    assert_ne!(
        rejected.original_process_id,
        rejected.replacement_process_id
    );
    assert_eq!(rejected.reconstruction_count, 1);
    assert_eq!(
        rejected.terminal_state,
        AttemptState::Failed,
        "unexpected Task403 rejection evidence: {rejected:#?}"
    );
    assert!(!rejected.fresh_output_generation);
    assert!(!rejected.positive_negative_distinct);
    assert!(rejected.no_publication_on_mismatch);
    assert!(
        rejected
            .failure_message
            .as_deref()
            .is_some_and(|message| message.contains("identity changed"))
    );
    Ok(())
}

fn task404_worker_plan(prompt_id: PromptId, variant: u64) -> Result<CompiledPlan, Box<dyn Error>> {
    let loader_id = NodeId("task404-checkpoint-loader".to_owned());
    let transform_id = NodeId("task404-model-transform".to_owned());
    let registry = generated_native_node_registry_projection(None)?;
    let loader_descriptor = registry
        .descriptor("UNETLoader")
        .cloned()
        .ok_or("generated registry omitted UNETLoader")?;
    let transform_descriptor = task404_model_transform_fixture_descriptor_for_test()?;
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([
            (
                loader_id.clone(),
                CompiledNode {
                    id: loader_id.clone(),
                    class_type: "UNETLoader".to_owned(),
                    descriptor: loader_descriptor,
                    inputs: BTreeMap::from([
                        (
                            "unet_name".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::Primitive {
                                    value: NativePrimitive::String(
                                        "task404-model.safetensors".to_owned(),
                                    ),
                                },
                            },
                        ),
                        (
                            "weight_dtype".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::Primitive {
                                    value: NativePrimitive::String("default".to_owned()),
                                },
                            },
                        ),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                transform_id.clone(),
                CompiledNode {
                    id: transform_id.clone(),
                    class_type: "Task404ModelTransformFixture".to_owned(),
                    descriptor: transform_descriptor,
                    inputs: BTreeMap::from([
                        (
                            "model".to_owned(),
                            InputBinding::Link {
                                source: loader_id.clone(),
                                output_index: 0,
                                lazy: false,
                                mode: NativePortCardinality::Scalar,
                            },
                        ),
                        (
                            "variant".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::Primitive {
                                    value: NativePrimitive::UnsignedInteger(variant),
                                },
                            },
                        ),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
        ]),
        topological_order: vec![loader_id.clone(), transform_id.clone()],
        static_required_nodes: BTreeSet::from([loader_id, transform_id.clone()]),
        output_nodes: vec![transform_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()
        .map_err(|error| format!("Task404 worker plan is invalid: {error}"))?;
    Ok(plan)
}

fn task405_worker_plan(prompt_id: PromptId, variant: u64) -> Result<CompiledPlan, Box<dyn Error>> {
    let loader_id = NodeId("task405-checkpoint-loader".to_owned());
    let dataset_id = NodeId("task405-training-dataset".to_owned());
    let training_id = NodeId("task405-training".to_owned());
    let sink_id = NodeId("task405-training-sink".to_owned());
    let registry = generated_native_node_registry_projection(None)?;
    let loader_descriptor = registry
        .descriptor("UNETLoader")
        .cloned()
        .ok_or("generated registry omitted UNETLoader")?;
    let training_descriptor = registry
        .descriptor("TrainLoraNode")
        .cloned()
        .ok_or("generated registry omitted TrainLoraNode")?;
    let list_string = |value: &str| InputBinding::Literal {
        value: NativeValue::List {
            values: vec![NativeValue::Primitive {
                value: NativePrimitive::String(value.to_owned()),
            }],
        },
    };
    let list_unsigned = |value| InputBinding::Literal {
        value: NativeValue::List {
            values: vec![NativeValue::Primitive {
                value: NativePrimitive::UnsignedInteger(value),
            }],
        },
    };
    let list_boolean = |value| InputBinding::Literal {
        value: NativeValue::List {
            values: vec![NativeValue::Primitive {
                value: NativePrimitive::Boolean(value),
            }],
        },
    };
    let list_dynamic_combo = |value: &str| InputBinding::Literal {
        value: NativeValue::List {
            values: vec![NativeValue::PreservedUnknown {
                type_name: "COMBO".to_owned(),
                value: json!(value),
            }],
        },
    };
    let plan = CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([
            (
                loader_id.clone(),
                CompiledNode {
                    id: loader_id.clone(),
                    class_type: "UNETLoader".to_owned(),
                    descriptor: loader_descriptor,
                    inputs: BTreeMap::from([
                        (
                            "unet_name".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::Primitive {
                                    value: NativePrimitive::String(
                                        "task405-model.safetensors".to_owned(),
                                    ),
                                },
                            },
                        ),
                        (
                            "weight_dtype".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::Primitive {
                                    value: NativePrimitive::String("default".to_owned()),
                                },
                            },
                        ),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                dataset_id.clone(),
                CompiledNode {
                    id: dataset_id.clone(),
                    class_type: "Task405TrainingDatasetFixture".to_owned(),
                    descriptor: task405_training_dataset_fixture_descriptor_for_test()?,
                    inputs: BTreeMap::from([(
                        "model".to_owned(),
                        InputBinding::Link {
                            source: loader_id.clone(),
                            output_index: 0,
                            lazy: false,
                            mode: NativePortCardinality::Scalar,
                        },
                    )]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                training_id.clone(),
                CompiledNode {
                    id: training_id.clone(),
                    class_type: "TrainLoraNode".to_owned(),
                    descriptor: training_descriptor,
                    inputs: BTreeMap::from([
                        (
                            "model".to_owned(),
                            InputBinding::Link {
                                source: loader_id.clone(),
                                output_index: 0,
                                lazy: false,
                                mode: NativePortCardinality::List,
                            },
                        ),
                        (
                            "latents".to_owned(),
                            InputBinding::Link {
                                source: dataset_id.clone(),
                                output_index: 0,
                                lazy: false,
                                mode: NativePortCardinality::List,
                            },
                        ),
                        (
                            "positive".to_owned(),
                            InputBinding::Link {
                                source: dataset_id.clone(),
                                output_index: 1,
                                lazy: false,
                                mode: NativePortCardinality::List,
                            },
                        ),
                        ("batch_size".to_owned(), list_unsigned(1)),
                        ("grad_accumulation_steps".to_owned(), list_unsigned(1)),
                        ("steps".to_owned(), list_unsigned(2)),
                        (
                            "learning_rate".to_owned(),
                            InputBinding::Literal {
                                value: NativeValue::List {
                                    values: vec![NativeValue::Primitive {
                                        value: NativePrimitive::Number(0.005),
                                    }],
                                },
                            },
                        ),
                        ("rank".to_owned(), list_unsigned(2)),
                        ("optimizer".to_owned(), list_string("AdamW")),
                        ("loss_function".to_owned(), list_string("SmoothL1")),
                        ("seed".to_owned(), list_unsigned(1_405 + variant)),
                        ("training_dtype".to_owned(), list_string("bf16")),
                        ("lora_dtype".to_owned(), list_string("fp32")),
                        ("quantized_backward".to_owned(), list_boolean(false)),
                        ("algorithm".to_owned(), list_dynamic_combo("LoRA")),
                        ("gradient_checkpointing".to_owned(), list_boolean(true)),
                        ("checkpoint_depth".to_owned(), list_unsigned(1)),
                        ("offloading".to_owned(), list_boolean(true)),
                        ("existing_lora".to_owned(), list_string("[None]")),
                        ("bucket_mode".to_owned(), list_boolean(false)),
                        ("bypass_mode".to_owned(), list_boolean(false)),
                    ]),
                    unknown: BTreeMap::new(),
                },
            ),
            (
                sink_id.clone(),
                CompiledNode {
                    id: sink_id.clone(),
                    class_type: "Task405TrainingSinkFixture".to_owned(),
                    descriptor: task405_training_sink_fixture_descriptor_for_test()?,
                    inputs: BTreeMap::from([(
                        "lora".to_owned(),
                        InputBinding::Link {
                            source: training_id.clone(),
                            output_index: 0,
                            lazy: false,
                            mode: NativePortCardinality::Scalar,
                        },
                    )]),
                    unknown: BTreeMap::new(),
                },
            ),
        ]),
        topological_order: vec![
            loader_id.clone(),
            dataset_id.clone(),
            training_id.clone(),
            sink_id.clone(),
        ],
        static_required_nodes: BTreeSet::from([
            loader_id,
            dataset_id,
            training_id,
            sink_id.clone(),
        ]),
        output_nodes: vec![sink_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity().map_err(|error| {
        let diagnostics = plan
            .nodes
            .iter()
            .map(|(node_id, node)| {
                let descriptor_error = node
                    .descriptor
                    .validate()
                    .err()
                    .map_or_else(|| "valid".to_owned(), |error| error.to_string());
                let bindings = node
                    .inputs
                    .iter()
                    .map(|(name, binding)| {
                        let input = node
                            .descriptor
                            .inputs
                            .iter()
                            .find(|input| input.name == *name);
                        match (input, binding) {
                            (Some(input), InputBinding::Literal { value }) => format!(
                                "{name}=literal(cardinality={:?}, accepted={}, schema_match={}, schema={:?}, value={value:?})",
                                input.cardinality,
                                match value {
                                    NativeValue::List { values } => values
                                        .iter()
                                        .all(|value| input.accepted_types.accepts(value)),
                                    value => input.accepted_types.accepts(value),
                                },
                                node.descriptor
                                    .source_schema
                                    .as_ref()
                                    .and_then(|schema| {
                                        schema.inputs.iter().find(|schema| schema.name == *name)
                                    })
                                    .is_none_or(|schema| {
                                        sim_nodes::native_value_matches_input_schema(value, schema)
                                    }),
                                node.descriptor.source_schema.as_ref().and_then(|schema| {
                                    schema.inputs.iter().find(|schema| schema.name == *name)
                                })
                            ),
                            (
                                Some(input),
                                InputBinding::Link {
                                    source,
                                    output_index,
                                    lazy,
                                    mode,
                                },
                            ) => {
                                let output = plan.nodes.get(source).and_then(|source_node| {
                                    source_node.descriptor.outputs.get(*output_index)
                                });
                                format!(
                                    "{name}=link(source={source:?}, output={output_index}, input_cardinality={:?}, mode={mode:?}, input_lazy={}, link_lazy={lazy}, output={output:?})",
                                    input.cardinality, input.lazy
                                )
                            }
                            (None, _) => format!("{name}=unknown-input({binding:?})"),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "node={node_id:?}, class={}, descriptor={descriptor_error}, output_node={}, bindings=[{bindings}]",
                    node.class_type, node.descriptor.output_node
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "Task405 worker plan is invalid: {error}; output_nodes={:?}; topological_order={:?}; static_required_nodes={:?}\n{diagnostics}",
            plan.output_nodes, plan.topological_order, plan.static_required_nodes
        )
    })?;
    Ok(plan)
}

fn task404_worker_recovery_fixture(
    replacement_variant: Option<u64>,
) -> Result<sim_runtime::NativeModelTransformWorkerReplacementEvidence, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_uuid = Uuid::new_v4();
    let roots = [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ]
    .into_iter()
    .map(|(namespace, name)| {
        let path = directory.path().join(name);
        fs::create_dir_all(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new(profile_uuid.to_string(), roots)?;
    let checkpoint_directory = roots
        .test_root_path(AssetNamespace::Model)?
        .join("diffusion_models");
    fs::create_dir_all(&checkpoint_directory)?;
    write_task404_model_fixture(&checkpoint_directory.join("task404-model.safetensors"))?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
    let profile_id = ProfileId(profile_uuid);
    let prompt_id = PromptId(Uuid::new_v4());
    let prior_attempt_id = AttemptId(Uuid::new_v4());
    let plan = task404_worker_plan(prompt_id, 0)?;
    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(16, prior_attempt_id)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let queue = ExecutionControlCommand {
        request_id: RequestId(Uuid::new_v4()),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Queue {
            plan: plan,
            priority: 0,
            front: false,
        },
    };
    presentation.submit(queue.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: queue.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(prior_attempt_id),
        },
    })?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
    let mut worker = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        profile_id,
        WorkerId(Uuid::new_v4()),
        NATIVE_IMAGE_REGISTRY_VERSION,
        2 * 1024 * 1024 * 1024,
    );
    worker.working_directory = Some(directory.path().to_path_buf());
    worker.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(30),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let config = NativeExecutionControllerConfig::new(assets, presentation, worker, true)?
        .with_memory_policy(sim_runtime::MemoryPolicy::Performance)
        .with_reduced_fixture_model_resources_for_test();
    let replacement_plan = replacement_variant
        .map(|variant| task404_worker_plan(prompt_id, variant))
        .transpose()?;
    Ok(
        exercise_model_transform_retry_after_worker_replacement_for_test(
            config,
            ExecutionEventBus::new(64)?,
            replacement_plan,
        )?,
    )
}

fn task405_worker_recovery_fixture(
    replacement_variant: Option<u64>,
) -> Result<sim_runtime::NativeModelTrainingWorkerReplacementEvidence, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_uuid = Uuid::new_v4();
    let roots = [
        (AssetNamespace::Input, "input"),
        (AssetNamespace::Output, "output"),
        (AssetNamespace::Temporary, "temporary"),
        (AssetNamespace::Model, "model"),
        (AssetNamespace::Plugin, "plugin"),
    ]
    .into_iter()
    .map(|(namespace, name)| {
        let path = directory.path().join(name);
        fs::create_dir_all(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new(profile_uuid.to_string(), roots)?;
    let checkpoint_directory = roots
        .test_root_path(AssetNamespace::Model)?
        .join("diffusion_models");
    fs::create_dir_all(&checkpoint_directory)?;
    write_task404_model_fixture(&checkpoint_directory.join("task405-model.safetensors"))?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
    let profile_id = ProfileId(profile_uuid);
    let prompt_id = PromptId(Uuid::new_v4());
    let prior_attempt_id = AttemptId(Uuid::new_v4());
    let plan = task405_worker_plan(prompt_id, 0)?;
    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(16, prior_attempt_id)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let queue = ExecutionControlCommand {
        request_id: RequestId(Uuid::new_v4()),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Queue {
            plan: plan,
            priority: 0,
            front: false,
        },
    };
    presentation.submit(queue.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: queue.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(prior_attempt_id),
        },
    })?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
    let mut worker = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        profile_id,
        WorkerId(Uuid::new_v4()),
        NATIVE_IMAGE_REGISTRY_VERSION,
        2 * 1024 * 1024 * 1024,
    );
    worker.working_directory = Some(directory.path().to_path_buf());
    worker.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(30),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let config = NativeExecutionControllerConfig::new(assets, presentation, worker, true)?
        .with_memory_policy(sim_runtime::MemoryPolicy::Performance)
        .with_reduced_fixture_model_resources_for_test();
    let replacement_plan = replacement_variant
        .map(|variant| task405_worker_plan(prompt_id, variant))
        .transpose()?;
    Ok(
        exercise_model_training_retry_after_worker_replacement_for_test(
            config,
            ExecutionEventBus::new(64)?,
            replacement_plan,
        )?,
    )
}

fn write_task404_model_fixture(destination: &Path) -> Result<(), Box<dyn Error>> {
    let mut header = BTreeMap::new();
    let mut payload = Vec::new();
    for (name, shape) in
        sim_model::auraflow_recipe_fixture_state_schema(NativeModelRecipeSourceForm::Standalone)?
    {
        let new_start = payload.len();
        let elements = shape.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(usize::try_from(*dimension).ok()?)
        });
        let byte_length = elements
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("Task404 fixture tensor size overflowed")?;
        payload.resize(
            payload
                .len()
                .checked_add(byte_length)
                .ok_or("Task404 fixture payload size overflowed")?,
            0,
        );
        let new_end = payload.len();
        header.insert(
            name,
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [new_start, new_end],
            }),
        );
    }
    if header.is_empty() {
        return Err("Task404 source MODEL state is empty".into());
    }
    let mut encoded_header = serde_json::to_vec(&header)?;
    while !encoded_header.len().is_multiple_of(8) {
        encoded_header.push(b' ');
    }
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&u64::try_from(encoded_header.len())?.to_le_bytes());
    encoded.extend_from_slice(&encoded_header);
    encoded.extend_from_slice(&payload);
    fs::write(destination, encoded)?;
    Ok(())
}

#[test]
fn task404_model_transform_retry_replays_in_a_replacement_worker() -> Result<(), Box<dyn Error>> {
    let recovered = task404_worker_recovery_fixture(None)?;
    assert_ne!(
        recovered.original_process_id,
        recovered.replacement_process_id
    );
    assert_eq!(recovered.reconstruction_count, 1);
    assert!(recovered.fresh_output_generation);
    assert!(recovered.selector_identity_bound);
    assert!(recovered.patch_operation_count > 2);
    assert!(recovered.conditioning_identity_bound);
    assert!(recovered.sampling_policy_identity_bound);
    assert!(recovered.guidance_modifier_identity_bound);
    assert!(recovered.stale_handle_wrong_generation);
    assert_eq!(recovered.terminal_state, AttemptState::Succeeded);
    assert!(recovered.failure_message.is_none());
    Ok(())
}

#[test]
fn task407_policy_and_modifiers_retry_in_a_replacement_worker() -> Result<(), Box<dyn Error>> {
    let recovered = task404_worker_recovery_fixture(None)?;
    assert_ne!(
        recovered.original_process_id,
        recovered.replacement_process_id
    );
    assert_eq!(recovered.reconstruction_count, 1);
    assert!(recovered.fresh_output_generation);
    assert!(recovered.sampling_policy_identity_bound);
    assert!(recovered.guidance_modifier_identity_bound);
    assert!(recovered.stale_handle_wrong_generation);
    assert_eq!(recovered.terminal_state, AttemptState::Succeeded);
    assert!(recovered.failure_message.is_none());
    let rejected = task404_worker_recovery_fixture(Some(1))?;
    assert_ne!(
        rejected.original_process_id,
        rejected.replacement_process_id
    );
    assert_eq!(rejected.reconstruction_count, 1);
    assert_eq!(rejected.terminal_state, AttemptState::Failed);
    assert!(!rejected.fresh_output_generation);
    assert!(rejected.no_publication_on_mismatch);
    assert!(
        rejected
            .failure_message
            .as_deref()
            .is_some_and(|message| message.contains("identity changed"))
    );
    Ok(())
}

#[test]
fn task404_model_transform_retry_rejects_changed_recipe_without_publication()
-> Result<(), Box<dyn Error>> {
    let rejected = task404_worker_recovery_fixture(Some(1))?;
    assert_ne!(
        rejected.original_process_id,
        rejected.replacement_process_id
    );
    assert_eq!(rejected.reconstruction_count, 1);
    assert_eq!(rejected.terminal_state, AttemptState::Failed);
    assert!(!rejected.fresh_output_generation);
    assert!(rejected.no_publication_on_mismatch);
    assert!(
        rejected
            .failure_message
            .as_deref()
            .is_some_and(|message| message.contains("identity changed"))
    );
    Ok(())
}

#[test]
fn task405_model_training_retry_replays_in_a_replacement_worker() -> Result<(), Box<dyn Error>> {
    let recovered = task405_worker_recovery_fixture(None)?;
    assert_ne!(
        recovered.original_process_id,
        recovered.replacement_process_id
    );
    assert_eq!(recovered.reconstruction_count, 1);
    assert!(recovered.fresh_output_generation);
    assert!(recovered.exact_steps);
    assert!(recovered.exact_residency);
    assert!(recovered.retained_state_bound);
    assert!(recovered.stale_handle_wrong_generation);
    assert_eq!(recovered.terminal_state, AttemptState::Succeeded);
    assert!(recovered.failure_message.is_none());
    Ok(())
}

#[test]
fn task405_model_training_retry_rejects_changed_seed_without_publication()
-> Result<(), Box<dyn Error>> {
    let rejected = task405_worker_recovery_fixture(Some(1))?;
    assert_ne!(
        rejected.original_process_id,
        rejected.replacement_process_id
    );
    assert_eq!(rejected.reconstruction_count, 1);
    assert_eq!(rejected.terminal_state, AttemptState::Failed);
    assert!(!rejected.fresh_output_generation);
    assert!(rejected.no_publication_on_mismatch);
    assert!(
        rejected
            .failure_message
            .as_deref()
            .is_some_and(|message| message.contains("identity changed"))
    );
    Ok(())
}

#[test]
fn task404_sampling_replacement_runs_through_production_sd15_ksampler() -> Result<(), Box<dyn Error>>
{
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) =
        CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture = Arc::new(NativeDiffusionFixture::checked_in());
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> = fixture;
    let mut baseline_plan = compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &BTreeSet::new(),
        provider.clone(),
    )?;
    baseline_plan.prompt_id = PromptId(Uuid::new_v4());
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::new_v4()),
        BTreeMap::new(),
        true,
        backend,
        provider,
    )?;
    let baseline = executor.execute_blocking(
        &baseline_plan,
        AttemptId(Uuid::new_v4()),
        Default::default(),
        0,
        workspace_authority.authorize_workspace(2 * 1024 * 1024 * 1024)?,
    )?;
    assert_eq!(
        baseline.report.state,
        AttemptState::Succeeded,
        "baseline SD15 KSampler failed: {:?}",
        baseline.report.error
    );

    let loader_id = NodeId("1".to_owned());
    let sampler_id = NodeId("5".to_owned());
    let replacement_id = NodeId("task404-sampling-profile".to_owned());
    let mut replacement_plan = baseline_plan;
    replacement_plan.prompt_id = PromptId(Uuid::new_v4());
    replacement_plan.nodes.insert(
        replacement_id.clone(),
        CompiledNode {
            id: replacement_id.clone(),
            class_type: "Task404SamplingProfileFixture".to_owned(),
            descriptor: task404_sampling_profile_fixture_descriptor_for_test()?,
            inputs: BTreeMap::from([(
                "model".to_owned(),
                InputBinding::Link {
                    source: loader_id,
                    output_index: 0,
                    lazy: false,
                    mode: NativePortCardinality::Scalar,
                },
            )]),
            unknown: BTreeMap::new(),
        },
    );
    let InputBinding::Link { source, .. } = replacement_plan
        .nodes
        .get_mut(&sampler_id)
        .ok_or("compiled workflow omitted KSampler")?
        .inputs
        .get_mut("model")
        .ok_or("compiled KSampler omitted model")?
    else {
        return Err("compiled KSampler model is not a link".into());
    };
    *source = replacement_id.clone();
    let sampler_position = replacement_plan
        .topological_order
        .iter()
        .position(|node_id| node_id == &sampler_id)
        .ok_or("compiled workflow omitted KSampler order")?;
    replacement_plan
        .topological_order
        .insert(sampler_position, replacement_id.clone());
    replacement_plan
        .static_required_nodes
        .insert(replacement_id);
    replacement_plan.validate_integrity()?;
    let replacement = executor.execute_blocking(
        &replacement_plan,
        AttemptId(Uuid::new_v4()),
        Default::default(),
        0,
        workspace_authority.authorize_workspace(2 * 1024 * 1024 * 1024)?,
    )?;
    assert_eq!(
        replacement.report.state,
        AttemptState::Succeeded,
        "replacement SD15 KSampler failed: {:?}",
        replacement.report.error
    );
    let baseline_sampler = baseline
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("baseline KSampler UI")?;
    let replacement_sampler = replacement
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("replacement KSampler UI")?;
    assert_eq!(
        baseline_sampler
            .get("sampling_profile")
            .and_then(serde_json::Value::as_str),
        Some(NativeSamplingPolicy::sd15()?.identity().as_str())
    );
    assert_eq!(
        replacement_sampler
            .get("sampling_profile")
            .and_then(serde_json::Value::as_str),
        Some(
            SamplingProfileReplacement::DiscreteVPrediction
                .resolve()?
                .identity()
                .as_str()
        )
    );
    assert_ne!(
        baseline_sampler.get("denoiser_sha256"),
        replacement_sampler.get("denoiser_sha256")
    );
    assert_ne!(
        baseline_sampler.get("latent_sha256"),
        replacement_sampler.get("latent_sha256")
    );
    cancellation.check()?;
    Ok(())
}

#[test]
fn task407_attached_policy_and_modifiers_run_through_production_sd15_ksampler()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) =
        CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture = Arc::new(NativeDiffusionFixture::checked_in());
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> = fixture;
    let mut baseline_plan = compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &BTreeSet::new(),
        provider.clone(),
    )?;
    baseline_plan.prompt_id = PromptId(Uuid::new_v4());
    let registry = generated_native_node_registry_projection(Some(provider.clone()))?;
    let sampling_descriptor = registry
        .descriptor("ModelSamplingContinuousV")
        .cloned()
        .ok_or("generated registry omitted ModelSamplingContinuousV")?;
    let modifier_descriptor = registry
        .descriptor("APG")
        .cloned()
        .ok_or("generated registry omitted APG")?;
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::new_v4()),
        BTreeMap::new(),
        true,
        backend,
        provider,
    )?;
    let baseline = executor.execute_blocking(
        &baseline_plan,
        AttemptId(Uuid::new_v4()),
        Default::default(),
        0,
        workspace_authority.authorize_workspace(2 * 1024 * 1024 * 1024)?,
    )?;
    assert_eq!(
        baseline.report.state,
        AttemptState::Succeeded,
        "baseline SD15 KSampler failed: {:?}",
        baseline.report.error
    );

    let loader_id = NodeId("1".to_owned());
    let sampler_id = NodeId("5".to_owned());
    let sampling_id = NodeId("task407-model-sampling".to_owned());
    let modifier_id = NodeId("task407-apg".to_owned());
    let mut replacement_plan = baseline_plan;
    replacement_plan.prompt_id = PromptId(Uuid::new_v4());
    replacement_plan.nodes.insert(
        sampling_id.clone(),
        CompiledNode {
            id: sampling_id.clone(),
            class_type: "ModelSamplingContinuousV".to_owned(),
            descriptor: sampling_descriptor,
            inputs: BTreeMap::from([
                (
                    "model".to_owned(),
                    InputBinding::Link {
                        source: loader_id,
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "sampling".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::String("v_prediction".to_owned()),
                        },
                    },
                ),
                (
                    "sigma_max".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(80.0),
                        },
                    },
                ),
                (
                    "sigma_min".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(0.02),
                        },
                    },
                ),
            ]),
            unknown: BTreeMap::new(),
        },
    );
    replacement_plan.nodes.insert(
        modifier_id.clone(),
        CompiledNode {
            id: modifier_id.clone(),
            class_type: "APG".to_owned(),
            descriptor: modifier_descriptor,
            inputs: BTreeMap::from([
                (
                    "model".to_owned(),
                    InputBinding::Link {
                        source: sampling_id.clone(),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "eta".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(0.65),
                        },
                    },
                ),
                (
                    "norm_threshold".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(0.8),
                        },
                    },
                ),
                (
                    "momentum".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(0.2),
                        },
                    },
                ),
            ]),
            unknown: BTreeMap::new(),
        },
    );
    let InputBinding::Link { source, .. } = replacement_plan
        .nodes
        .get_mut(&sampler_id)
        .ok_or("compiled workflow omitted KSampler")?
        .inputs
        .get_mut("model")
        .ok_or("compiled KSampler omitted model")?
    else {
        return Err("compiled KSampler model is not a link".into());
    };
    *source = modifier_id.clone();
    let sampler_position = replacement_plan
        .topological_order
        .iter()
        .position(|node_id| node_id == &sampler_id)
        .ok_or("compiled workflow omitted KSampler order")?;
    replacement_plan.topological_order.splice(
        sampler_position..sampler_position,
        [sampling_id.clone(), modifier_id.clone()],
    );
    replacement_plan.static_required_nodes.insert(sampling_id);
    replacement_plan.static_required_nodes.insert(modifier_id);
    replacement_plan.validate_integrity()?;
    let replacement = executor.execute_blocking(
        &replacement_plan,
        AttemptId(Uuid::new_v4()),
        Default::default(),
        0,
        workspace_authority.authorize_workspace(2 * 1024 * 1024 * 1024)?,
    )?;
    assert_eq!(
        replacement.report.state,
        AttemptState::Succeeded,
        "Task407 SD15 KSampler failed: {:?}",
        replacement.report.error
    );
    let baseline_sampler = baseline
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("baseline KSampler UI")?;
    let replacement_sampler = replacement
        .report
        .ui_outputs
        .get(&sampler_id)
        .ok_or("Task407 KSampler UI")?;
    assert_ne!(
        baseline_sampler.get("sampling_profile"),
        replacement_sampler.get("sampling_profile")
    );
    assert_ne!(
        baseline_sampler.get("sampling_policy_digest"),
        replacement_sampler.get("sampling_policy_digest")
    );
    assert_ne!(
        baseline_sampler.get("guidance_modifier_digest"),
        replacement_sampler.get("guidance_modifier_digest")
    );
    assert_ne!(
        baseline_sampler.get("denoiser_sha256"),
        replacement_sampler.get("denoiser_sha256")
    );
    assert_ne!(
        baseline_sampler.get("latent_sha256"),
        replacement_sampler.get("latent_sha256")
    );
    Ok(())
}

#[test]
fn task407_cfg_guider_runs_through_production_custom_sampler() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) =
        CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024 * 1024)?;
    let backend = Arc::new(backend);
    let fixture = Arc::new(NativeDiffusionFixture::checked_in());
    let provider: Arc<dyn sim_runtime::NativeDiffusionProvider> = fixture;
    let mut plan = compile_native_diffusion_workflow(
        include_bytes!("../fixtures/native_diffusion/workflow.json"),
        &BTreeSet::new(),
        provider.clone(),
    )?;
    plan.prompt_id = PromptId(Uuid::new_v4());
    let registry = generated_native_node_registry_projection(Some(provider.clone()))?;
    let guider_descriptor = registry
        .descriptor("CFGGuider")
        .cloned()
        .ok_or("generated registry omitted CFGGuider")?;
    let custom_descriptor = registry
        .descriptor("SamplerCustomAdvanced")
        .cloned()
        .ok_or("generated registry omitted SamplerCustomAdvanced")?;
    let inputs_descriptor = task407_custom_sampler_inputs_fixture_descriptor_for_test()?;
    let transform_descriptor = task407_sampling_policy_fixture_descriptor_for_test()?;
    let loader_id = NodeId("1".to_owned());
    let positive_id = NodeId("2".to_owned());
    let negative_id = NodeId("3".to_owned());
    let latent_id = NodeId("4".to_owned());
    let sampler_id = NodeId("5".to_owned());
    let decode_id = NodeId("6".to_owned());
    let inputs_id = NodeId("task407-custom-inputs".to_owned());
    let transform_id = NodeId("task407-model-hooks-and-modifiers".to_owned());
    let guider_id = NodeId("task407-cfg-guider".to_owned());
    let custom_id = NodeId("task407-custom-sampler".to_owned());
    plan.nodes.insert(
        inputs_id.clone(),
        CompiledNode {
            id: inputs_id.clone(),
            class_type: "Task407CustomSamplerInputsFixture".to_owned(),
            descriptor: inputs_descriptor,
            inputs: BTreeMap::new(),
            unknown: BTreeMap::new(),
        },
    );
    plan.nodes.insert(
        transform_id.clone(),
        CompiledNode {
            id: transform_id.clone(),
            class_type: "Task407SamplingPolicyFixture".to_owned(),
            descriptor: transform_descriptor,
            inputs: BTreeMap::from([(
                "model".to_owned(),
                InputBinding::Link {
                    source: loader_id,
                    output_index: 0,
                    lazy: false,
                    mode: NativePortCardinality::Scalar,
                },
            )]),
            unknown: BTreeMap::new(),
        },
    );
    plan.nodes.insert(
        guider_id.clone(),
        CompiledNode {
            id: guider_id.clone(),
            class_type: "CFGGuider".to_owned(),
            descriptor: guider_descriptor,
            inputs: BTreeMap::from([
                (
                    "model".to_owned(),
                    InputBinding::Link {
                        source: transform_id.clone(),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "positive".to_owned(),
                    InputBinding::Link {
                        source: positive_id,
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "negative".to_owned(),
                    InputBinding::Link {
                        source: negative_id,
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "cfg".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::Number(5.5),
                        },
                    },
                ),
            ]),
            unknown: BTreeMap::new(),
        },
    );
    plan.nodes.insert(
        custom_id.clone(),
        CompiledNode {
            id: custom_id.clone(),
            class_type: "SamplerCustomAdvanced".to_owned(),
            descriptor: custom_descriptor,
            inputs: BTreeMap::from([
                (
                    "noise".to_owned(),
                    InputBinding::Link {
                        source: inputs_id.clone(),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "guider".to_owned(),
                    InputBinding::Link {
                        source: guider_id.clone(),
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "sampler".to_owned(),
                    InputBinding::Link {
                        source: inputs_id.clone(),
                        output_index: 1,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "sigmas".to_owned(),
                    InputBinding::Link {
                        source: inputs_id.clone(),
                        output_index: 2,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
                (
                    "latent_image".to_owned(),
                    InputBinding::Link {
                        source: latent_id,
                        output_index: 0,
                        lazy: false,
                        mode: NativePortCardinality::Scalar,
                    },
                ),
            ]),
            unknown: BTreeMap::new(),
        },
    );
    plan.nodes.remove(&sampler_id);
    plan.static_required_nodes.remove(&sampler_id);
    let decode_node = plan
        .nodes
        .get_mut(&decode_id)
        .ok_or("compiled workflow omitted VAEDecode")?;
    let samples_input = decode_node
        .descriptor
        .inputs
        .iter()
        .find(|input| input.name == "samples")
        .ok_or("VAEDecode descriptor omitted samples")?;
    let samples_lazy = samples_input.lazy;
    let samples_cardinality = samples_input.cardinality;
    decode_node.inputs.insert(
        "samples".to_owned(),
        InputBinding::Link {
            source: custom_id.clone(),
            output_index: 0,
            lazy: samples_lazy,
            mode: samples_cardinality,
        },
    );
    let sampler_position = plan
        .topological_order
        .iter()
        .position(|node_id| node_id == &sampler_id)
        .ok_or("compiled workflow omitted KSampler order")?;
    plan.topological_order.splice(
        sampler_position..=sampler_position,
        [
            inputs_id.clone(),
            transform_id.clone(),
            guider_id.clone(),
            custom_id.clone(),
        ],
    );
    plan.static_required_nodes
        .extend([inputs_id, transform_id, guider_id, custom_id.clone()]);
    plan.validate_integrity()?;
    let executor = NativeImageExecutor::new_with_diffusion_provider(
        ProfileId(Uuid::new_v4()),
        BTreeMap::new(),
        true,
        backend,
        provider,
    )?;
    let execution = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::new_v4()),
        Default::default(),
        0,
        workspace_authority.authorize_workspace(2 * 1024 * 1024 * 1024)?,
    )?;
    assert_eq!(
        execution.report.state,
        AttemptState::Succeeded,
        "Task407 custom sampler failed: {:?}",
        execution.report.error
    );
    let sampler = execution
        .report
        .ui_outputs
        .get(&custom_id)
        .ok_or("Task407 custom sampler UI")?;
    assert_eq!(
        sampler
            .get("guider_kind")
            .and_then(serde_json::Value::as_str),
        Some("cfg")
    );
    assert!(
        sampler
            .get("guider_semantic_sha256")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|digest| digest.len() == 64)
    );
    assert_eq!(
        sampler
            .get("execution_hook_count")
            .and_then(serde_json::Value::as_u64),
        Some(2)
    );
    assert_eq!(
        sampler
            .get("guidance_modifier_count")
            .and_then(serde_json::Value::as_u64),
        Some(2)
    );
    assert!(
        sampler
            .get("execution_hook_digest")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|digest| digest.len() == 64)
    );
    Ok(())
}

fn write_task402_style_fixture(roots: &AssetRoots) -> Result<(), Box<dyn Error>> {
    let oracle: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../fixtures/models/conditioning-auxiliary-resource-foundation/oracle.json"
    ))?;
    let state = oracle
        .get("style")
        .and_then(|value| value.get("dtypes"))
        .and_then(|value| value.get("float32"))
        .and_then(|value| value.get("state"))
        .and_then(serde_json::Value::as_array)
        .ok_or("style fixture state is missing")?;
    let mut header = serde_json::Map::new();
    let mut byte_offset = 0_u64;
    for entry in state {
        let key = entry
            .get("key")
            .and_then(serde_json::Value::as_str)
            .ok_or("style fixture key is missing")?;
        let shape = entry
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or("style fixture shape is missing")?;
        let element_count = shape.iter().try_fold(1_u64, |count, dimension| {
            count
                .checked_mul(
                    dimension
                        .as_u64()
                        .ok_or("style fixture dimension is invalid")?,
                )
                .ok_or("style fixture element count overflow")
        })?;
        let byte_length = element_count
            .checked_mul(4)
            .ok_or("style fixture byte length overflow")?;
        let byte_end = byte_offset
            .checked_add(byte_length)
            .ok_or("style fixture byte range overflow")?;
        header.insert(
            key.to_owned(),
            json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [byte_offset, byte_end],
            }),
        );
        byte_offset = byte_end;
    }
    let header = serde_json::to_vec(&header)?;
    let path = roots
        .test_root_path(AssetNamespace::Model)?
        .join("style_models/style.safetensors");
    fs::create_dir_all(path.parent().ok_or("style fixture parent is missing")?)?;
    let mut file = fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&vec![0_u8; usize::try_from(byte_offset)?])?;
    file.sync_all()?;
    Ok(())
}

fn task402_style_plan(prompt_id: PromptId) -> Result<CompiledPlan, Box<dyn Error>> {
    let node_id = NodeId("model-resource-loader".to_owned());
    let registry = generated_native_node_registry_projection(None)?;
    let mut descriptor = registry
        .descriptor("StyleModelLoader")
        .cloned()
        .ok_or("generated registry omitted StyleModelLoader")?;
    descriptor.output_node = true;
    Ok(CompiledPlan {
        prompt_id,
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes: BTreeMap::from([(
            node_id.clone(),
            CompiledNode {
                id: node_id.clone(),
                class_type: "StyleModelLoader".to_owned(),
                descriptor,
                inputs: BTreeMap::from([(
                    "style_model_name".to_owned(),
                    InputBinding::Literal {
                        value: NativeValue::Primitive {
                            value: NativePrimitive::String("style.safetensors".to_owned()),
                        },
                    },
                )]),
                unknown: BTreeMap::new(),
            },
        )]),
        topological_order: vec![node_id.clone()],
        static_required_nodes: BTreeSet::from([node_id.clone()]),
        output_nodes: vec![node_id],
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    })
}

fn task402_persisted_style_evidence(
    assets: Arc<Mutex<AssetService>>,
    profile_id: &str,
    attempt_id: AttemptId,
    node_id: &NodeId,
) -> Result<PersistedNativeModelResourceEvidence, Box<dyn Error>> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    let authorization = authorize_native_api_asset_reader(&policy)?;
    let registry = NativeAssetResolverRegistry::new(assets, authorization);
    let request = NativeModelResourceRequest::checked(
        NativeModelResourceRole::StyleModel,
        "style_models",
        vec!["style.safetensors".to_owned()],
    )?;
    let selection =
        worker_model_source_selection_sha256(request.folder_category(), request.source_names())?;
    let context = WorkerModelSourceContext {
        session_id: Uuid::from_u128(0x402_321),
        attempt_id,
        attempt_generation: 1,
        node_id: node_id.0.clone(),
        node_generation: 1,
        service_id: Uuid::from_u128(0x402_322),
        service_generation: 1,
        ordered_source_identity_sha256: selection,
    };
    let (transport, host) = NativeModelSourceTestTransport::channel();
    let session = transport.open_session(context.clone())?;
    let open = WorkerModelSourceRequest {
        context: context.clone(),
        call_ordinal: 1,
        operation: WorkerModelSourceOperation::Open {
            folder_category: request.folder_category().to_owned(),
            source_names: request.source_names().to_vec(),
        },
    };
    let WorkerModelSourceResponse::Opened(opened) = bridge_model_source_call(
        &host,
        &session,
        &registry,
        &context,
        open,
        &CancellationToken::default(),
    )?
    else {
        return Err("style source did not open for persisted evidence".into());
    };
    let close = WorkerModelSourceRequest {
        context: context.clone(),
        call_ordinal: 2,
        operation: WorkerModelSourceOperation::Close,
    };
    assert!(matches!(
        bridge_model_source_call(
            &host,
            &session,
            &registry,
            &context,
            close,
            &CancellationToken::default(),
        )?,
        WorkerModelSourceResponse::Closed(_)
    ));
    let deliberately_changed_identity = NativeModelResourceIdentity::checked(
        NativeModelResourceRole::StyleModel,
        "persisted-style-identity-mismatch",
        "zed-native-style-model-v1",
        "a".repeat(64),
        "b".repeat(64),
    )?;
    Ok(persisted_model_resource_evidence_for_test(
        node_id.clone(),
        &request,
        &opened,
        deliberately_changed_identity,
    )?)
}

fn persisted_model_resource_retry_replaces_real_worker() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let profile_uuid = Uuid::from_u128(0x402_323);
    let roots = [
        AssetNamespace::Input,
        AssetNamespace::Output,
        AssetNamespace::Temporary,
        AssetNamespace::Model,
        AssetNamespace::Plugin,
    ]
    .into_iter()
    .map(|namespace| {
        let path = directory.path().join(namespace.locator_type());
        fs::create_dir_all(&path)?;
        Ok((namespace, path))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let roots = AssetRoots::new(profile_uuid.to_string(), roots)?;
    write_task402_style_fixture(&roots)?;
    let assets = Arc::new(Mutex::new(AssetService::open(roots)?));
    let profile_id = ProfileId(profile_uuid);
    let prompt_id = PromptId(Uuid::from_u128(0x402_324));
    let prior_attempt_id = AttemptId(Uuid::from_u128(0x402_325));
    let retry_attempt_id = AttemptId(Uuid::from_u128(0x402_326));
    let node_id = NodeId("model-resource-loader".to_owned());
    let plan = task402_style_plan(prompt_id)?;
    let evidence = task402_persisted_style_evidence(
        assets.clone(),
        &profile_uuid.to_string(),
        prior_attempt_id,
        &node_id,
    )?;
    let mut presentation =
        ExecutionPresentationService::new_with_first_attempt_id(16, prior_attempt_id)?;
    presentation.initialize_profile(
        profile_id,
        ExecutionDataSource::Live,
        ExecutionSnapshotStatus::Ready,
    )?;
    let queue = ExecutionControlCommand {
        request_id: RequestId(Uuid::from_u128(0x402_327)),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Queue {
            plan: plan,
            priority: 0,
            front: false,
        },
    };
    presentation.submit(queue.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: queue.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(prior_attempt_id),
        },
    })?;
    record_succeeded_model_resource_attempt_for_test(
        &mut presentation,
        profile_id,
        prompt_id,
        prior_attempt_id,
        evidence,
    )?;
    let retry = ExecutionControlCommand {
        request_id: RequestId(Uuid::from_u128(0x402_328)),
        profile_id,
        expected_revision: None,
        kind: ExecutionControlCommandKind::Retry {
            attempt_id: prior_attempt_id,
            source: RetryPromptSource::OriginalPrompt,
            replacement_plan: None,
        },
    };
    presentation.submit(retry.clone())?;
    presentation.apply_ack(ExecutionCommandAck {
        request_id: retry.request_id,
        profile_id,
        outcome: ExecutionCommandOutcome::Accepted {
            assigned_attempt_id: Some(retry_attempt_id),
        },
    })?;
    let presentation = sim_runtime::ExecutionPresentationOwner::ephemeral(presentation);
    let mut worker = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_native_image_worker_fixture"),
        profile_id,
        WorkerId(Uuid::from_u128(0x402_329)),
        NATIVE_IMAGE_REGISTRY_VERSION,
        MEMORY_LIMIT_BYTES,
    );
    worker.working_directory = Some(directory.path().to_path_buf());
    worker.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(30),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(3),
        ready_timeout: Duration::from_secs(30),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    let config = NativeExecutionControllerConfig::new(assets, presentation.clone(), worker, true)?
        .with_reduced_fixture_model_resources_for_test();
    let replacement = exercise_persisted_model_resource_retry_after_worker_replacement_for_test(
        config,
        ExecutionEventBus::new(64)?,
    )?;
    assert_ne!(
        replacement.original_process_id,
        replacement.replacement_process_id
    );
    assert_eq!(replacement.service_generation, 2);
    assert_eq!(replacement.attempt_generation, 1);
    assert_eq!(replacement.reconstruction_count, 1);
    assert_eq!(
        replacement.original_handle_store_identity.store_id,
        replacement.replacement_handle_store_identity.store_id
    );
    assert_ne!(
        replacement.original_handle_store_identity.generation_id,
        replacement.replacement_handle_store_identity.generation_id
    );
    assert!(replacement.worker_store_id_preserved);
    assert!(replacement.worker_generation_changed);
    assert!(replacement.worker_old_handle_wrong_generation);
    let snapshot = presentation.snapshot(profile_id)?;
    let retry = snapshot
        .attempts
        .iter()
        .find(|attempt| attempt.attempt_id == retry_attempt_id)
        .ok_or("persisted retry disappeared")?;
    assert_eq!(retry.state, AttemptState::Failed);
    assert!(retry.outputs.is_empty());
    Ok(())
}

fn run_model_resource_recipe_restart_is_atomic(
    exercise_handle_restart: bool,
) -> Result<(), Box<dyn Error>> {
    let sources = Arc::new(vec![
        restart_clip_source(0, "clip-l", 768)?,
        restart_clip_source(1, "clip-g", 1_280)?,
    ]);
    let workspace_bytes = 1024 * 1024 * 1024;
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let backend = Arc::new(backend);
    let attempt_id = AttemptId(Uuid::from_u128(0x40101));
    let handle_store_generation = NativeHandleStoreGeneration::new()?;
    let handle_store = handle_store_generation.handle_store_for_attempt(attempt_id);
    let request = NativeModelResourceRequest::checked(
        NativeModelResourceRole::Clip,
        "text_encoders",
        vec![
            "clip-l.safetensors".to_owned(),
            "clip-g.safetensors".to_owned(),
        ],
    )?;
    let interrupted_sources = sources.clone();
    let (interrupted, interrupted_counts) = load_restart_model_resource(
        exercise_handle_restart,
        PromptId(Uuid::from_u128(0x40102)),
        attempt_id,
        NodeId("resource-restart".to_owned()),
        CancellationToken::default(),
        workspace_authority.authorize_workspace(workspace_bytes)?,
        handle_store.clone(),
        NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x40103), 1, 1, 1)?,
        backend.clone(),
        request.clone(),
        move |host| serve_restart_model_resource(host, interrupted_sources, true),
    )?;
    assert!(interrupted.is_err());
    assert_eq!(
        interrupted_counts.0,
        if exercise_handle_restart { 2 } else { 1 }
    );
    assert!(interrupted_counts.1 > 0);
    assert_eq!(
        interrupted_counts.2,
        if exercise_handle_restart { 1 } else { 0 }
    );
    assert_eq!(handle_store_generation.len(), 0);

    let retry_sources = sources.clone();
    let (payload, retry_counts) = load_restart_model_resource(
        exercise_handle_restart,
        PromptId(Uuid::from_u128(0x40102)),
        attempt_id,
        NodeId("resource-restart".to_owned()),
        CancellationToken::default(),
        workspace_authority.authorize_workspace(workspace_bytes)?,
        handle_store.clone(),
        NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x40103), 2, 2, 2)?,
        backend.clone(),
        request.clone(),
        move |host| serve_restart_model_resource(host, retry_sources, false),
    )?;
    let (payload, retry_evidence) = payload.map_err(|error| {
        format!("ordered CLIP construction after worker replacement failed: {error}")
    })?;
    let clip = payload
        .native_clip_resource()
        .ok_or("replacement CLIP payload omitted concrete backing")?;
    assert_eq!(clip.components().len(), 2);
    assert_eq!(retry_counts.0, if exercise_handle_restart { 2 } else { 1 });
    assert!(retry_counts.1 > interrupted_counts.1);
    assert_eq!(retry_counts.2, if exercise_handle_restart { 2 } else { 1 });
    assert_eq!(handle_store_generation.len(), 0);
    if exercise_handle_restart {
        let cancellation = CancellationToken::default();
        let stored = Arc::new(NativeStoredModelPayload::model_resource(payload.clone())?);
        let old_handle = handle_store.publish(NativeStoredPayload::Model(stored), &cancellation)?;
        let semantic_digest = payload.identity().digest_sha256().to_owned();

        let restarted_generation = handle_store_generation.fresh_generation()?;
        let restarted_store = restarted_generation.handle_store_for_attempt(attempt_id);
        let restarted_sources = sources;
        let (reconstructed, reconstructed_counts) = load_restart_model_resource(
            true,
            PromptId(Uuid::from_u128(0x40102)),
            attempt_id,
            NodeId("resource-restart".to_owned()),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            restarted_store.clone(),
            NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x40103), 3, 3, 3)?,
            backend.clone(),
            request,
            move |host| serve_restart_model_resource(host, restarted_sources, false),
        )?;
        let (reconstructed, reconstructed_evidence) = reconstructed.map_err(|error| {
            format!("ordered CLIP reconstruction in fresh generation failed: {error}")
        })?;
        assert_eq!(reconstructed.identity().digest_sha256(), semantic_digest);
        assert_eq!(reconstructed_evidence, retry_evidence);
        assert_eq!(reconstructed_counts.0, 2);
        assert!(reconstructed_counts.1 > 0);
        assert_eq!(reconstructed_counts.2, 2);
        let stored = Arc::new(NativeStoredModelPayload::model_resource(reconstructed)?);
        let fresh_handle =
            restarted_store.publish(NativeStoredPayload::Model(stored), &cancellation)?;
        assert!(matches!(
            restarted_store.resolve(&old_handle, old_handle.handle_type(), &cancellation),
            Err(sim_nodes::NativeHandleStoreError::WrongGeneration)
        ));
        let resolved =
            restarted_store.resolve(&fresh_handle, fresh_handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Model(resolved_model) = resolved.as_ref() else {
            return Err("fresh model handle resolved to another payload kind".into());
        };
        assert_eq!(resolved_model.digest_sha256(), semantic_digest);
        drop(resolved);
        restarted_store.revoke(&fresh_handle, &cancellation)?;

        let changed_sources = Arc::new(vec![
            restart_clip_source(0, "clip-l", 768)?,
            restart_clip_source(1, "clip-g-changed", 1_280)?,
        ]);
        let (changed, changed_counts) = load_restart_model_resource(
            true,
            PromptId(Uuid::from_u128(0x40102)),
            attempt_id,
            NodeId("resource-restart".to_owned()),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            restarted_store.clone(),
            NativeModelSourceWorkerBinding::checked_for_test(Uuid::from_u128(0x40103), 4, 4, 4)?,
            backend,
            NativeModelResourceRequest::checked(
                NativeModelResourceRole::Clip,
                "text_encoders",
                vec![
                    "clip-l.safetensors".to_owned(),
                    "clip-g.safetensors".to_owned(),
                ],
            )?,
            move |host| serve_restart_model_resource(host, changed_sources, false),
        )?;
        let (changed, changed_evidence) = changed.map_err(|error| {
            format!("changed ordered CLIP reconstruction did not reach identity check: {error}")
        })?;
        assert_ne!(changed_evidence, retry_evidence);
        assert_ne!(changed.identity().digest_sha256(), semantic_digest);
        assert_eq!(changed_counts.0, 2);
        assert_eq!(changed_counts.2, 2);
        let changed_context = NativeNodeContext::new(
            PromptId(Uuid::from_u128(0x40102)),
            attempt_id,
            NodeId("resource-restart".to_owned()),
            CancellationToken::default(),
            workspace_authority.authorize_workspace(workspace_bytes)?,
            restarted_store.clone(),
        )?;
        let failure = publish_reconstructed_native_model_resource(
            changed,
            &semantic_digest,
            &changed_context,
        )
        .expect_err("changed source reconstruction unexpectedly published a fresh handle");
        assert_eq!(failure.code, "native_model_resource_reconstruction_changed");
        assert!(restarted_generation.is_empty());
        handle_store.revoke(&old_handle, &cancellation)?;
        assert!(restarted_generation.is_empty());
        assert!(handle_store_generation.is_empty());
    }
    Ok(())
}

#[test]
fn val_recovery_008() -> Result<(), Box<dyn Error>> {
    let mut cases = native_controller::run_native_controller_e2e()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64)?;
    let workspace = workspace_authority.authorize_workspace(64)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
    let descriptor =
        TensorDescriptor::contiguous(vec![16], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let cancelled_allocation = backend.allocate(descriptor, &context);
    cases.insert(
        "cancelled_device_allocation_releases_memory_before_fence",
        matches!(cancelled_allocation, Err(TensorError::Cancelled))
            && backend.memory_snapshot().current_bytes == 0,
    );
    cases.insert(
        "canonical_cancellation_state_is_monotonic",
        cancellation.is_cancelled() && cancellation.check().is_err(),
    );

    assert_cases("VAL-RECOVERY-008", &cases);
    write_artifact(
        "val-recovery-008.json",
        "VAL-RECOVERY-008",
        "native-worker-controller-cancellation-convergence",
        &cases,
        json!({
            "native_controller_source_sha256": format!(
                "{:x}",
                Sha256::digest(include_bytes!("support/native_controller.rs"))
            ),
            "worker_source_sha256": format!(
                "{:x}",
                Sha256::digest(include_bytes!("../../sim_worker/src/sim_worker.rs"))
            )
        }),
    )?;
    Ok(())
}

#[test]
fn val_recovery_009() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut foreign = spawn_foreign_process()?;
    let foreign_process_id = foreign.id();
    let before_shutdown = run_orphan_case(directory.path(), "before-shutdown")?;
    let during_shutdown = run_orphan_case(directory.path(), "during-shutdown")?;
    let foreign_untouched = foreign.try_status()?.is_none() && process_exists(foreign_process_id)?;

    let mut supervisor = smol::block_on(RuntimeSupervisor::start(worker_config()))?;
    let replacement_ready = supervisor.worker_process_id().is_some();
    let first_status = smol::block_on(supervisor.terminate())?;
    let mut recovered = smol::block_on(supervisor.recover())?;
    let recovered_once = recovered.worker_process_id().is_some();
    let second_status = smol::block_on(recovered.terminate())?;
    let restart_loop_prevented = matches!(
        smol::block_on(recovered.recover()),
        Err(RuntimeSupervisorError::RecoveryBudgetExhausted { maximum: 1 })
    );

    terminate_foreign_process(&mut foreign)?;
    let cases = BTreeMap::from([
        (
            "parent_crash_before_shutdown_leaves_no_owned_worker",
            before_shutdown,
        ),
        (
            "parent_crash_during_shutdown_leaves_no_owned_worker",
            during_shutdown,
        ),
        (
            "foreign_process_is_never_selected_or_terminated",
            foreign_untouched,
        ),
        (
            "replacement_worker_starts_after_orphan_reconciliation",
            replacement_ready,
        ),
        (
            "bounded_recovery_runs_once_without_restart_loop",
            !first_status.success()
                && recovered_once
                && !second_status.success()
                && restart_loop_prevented,
        ),
    ]);
    assert_cases("VAL-RECOVERY-009", &cases);
    write_artifact(
        "val-recovery-009.json",
        "VAL-RECOVERY-009",
        "native-worker-parent-crash-orphan-containment",
        &cases,
        json!({
            "orphan_parent_fixture_sha256": format!(
                "{:x}",
                Sha256::digest(include_bytes!("../src/bin/sim_test_worker_fixture.rs"))
            ),
            "process_owner": "RuntimeSupervisor",
            "foreign_process_id_was_distinct": true
        }),
    )?;
    Ok(())
}

fn run_orphan_case(directory: &Path, phase: &str) -> Result<bool, Box<dyn Error>> {
    let pid_file = directory.join(format!("{phase}.pid"));
    let status = smol::block_on(async {
        Command::new(env!("CARGO_BIN_EXE_sim_test_worker_fixture"))
            .arg("--orphan-parent")
            .arg(env!("CARGO_BIN_EXE_sim_test_worker_fixture"))
            .arg(&pid_file)
            .arg(phase)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
    })?;
    let worker_process_id = fs::read_to_string(&pid_file)?.trim().parse::<u32>()?;
    Ok(!status.success() && wait_for_process_exit(worker_process_id, Duration::from_secs(5))?)
}

fn worker_config() -> WorkerLaunchConfig {
    let mut config = WorkerLaunchConfig::new(
        env!("CARGO_BIN_EXE_sim_test_worker_fixture"),
        ProfileId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_09f3)),
        WorkerId(Uuid::from_u128(0x5349_4d00_0000_0000_0000_0000_0000_09f4)),
        "orphan-resilience-v1",
        MEMORY_LIMIT_BYTES,
    );
    config.policy = SupervisorPolicy {
        heartbeat_interval: Duration::from_secs(2),
        missed_heartbeat_limit: 3,
        shutdown_timeout: Duration::from_secs(5),
        ready_timeout: Duration::from_secs(5),
        maximum_automatic_restarts: 1,
        restart_backoff: Duration::from_millis(1),
    };
    config
}

fn wait_for_process_exit(process_id: u32, timeout: Duration) -> Result<bool, Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !process_exists(process_id)? {
            return Ok(true);
        }
        thread::sleep(Duration::from_millis(10));
    }
    Ok(!process_exists(process_id)?)
}

#[cfg(unix)]
fn process_exists(process_id: u32) -> Result<bool, Box<dyn Error>> {
    Ok(smol::block_on(async {
        Command::new("/bin/kill")
            .args(["-0", &process_id.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
    })?
    .success())
}

#[cfg(windows)]
fn process_exists(process_id: u32) -> Result<bool, Box<dyn Error>> {
    let output = smol::block_on(async {
        Command::new("tasklist.exe")
            .args(["/FI", &format!("PID eq {process_id}"), "/FO", "CSV", "/NH"])
            .output()
            .await
    })?;
    Ok(output.status.success()
        && String::from_utf8_lossy(&output.stdout).contains(&process_id.to_string()))
}

#[cfg(unix)]
fn spawn_foreign_process() -> Result<Child, Box<dyn Error>> {
    Ok(Command::new("/bin/sleep")
        .arg("30")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?)
}

#[cfg(windows)]
fn spawn_foreign_process() -> Result<Child, Box<dyn Error>> {
    Ok(Command::new("ping.exe")
        .args(["-n", "30", "127.0.0.1"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?)
}

fn terminate_foreign_process(child: &mut Child) -> Result<(), Box<dyn Error>> {
    if child.try_status()?.is_none() {
        child.kill()?;
    }
    smol::block_on(child.status())?;
    Ok(())
}

fn assert_cases(validation: &str, cases: &BTreeMap<&str, bool>) {
    assert!(
        cases.values().all(|passed| *passed),
        "{validation} cases failed: {cases:#?}"
    );
}

fn target_directory() -> Result<PathBuf, Box<dyn Error>> {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    Ok(std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target")))
}

fn write_artifact(
    filename: &str,
    validation: &str,
    scope: &str,
    cases: &BTreeMap<&str, bool>,
    fixture_digests: serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let directory = target_directory()?.join("comfy-parity");
    fs::create_dir_all(&directory)?;
    let artifact = json!({
        "validation_id": validation,
        "validation": validation,
        "scope": scope,
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "process_containment": if cfg!(windows) { "job_object_and_stdin_eof" } else { "owned_process_group_and_stdin_eof" },
            "python_or_javascript_runtime": false
        },
        "fixture_digests": fixture_digests,
        "summary": {
            "passed": cases.len(),
            "failed": 0,
            "skipped": 0
        },
        "cases": cases,
        "skipped": []
    });
    fs::write(
        directory.join(filename),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
