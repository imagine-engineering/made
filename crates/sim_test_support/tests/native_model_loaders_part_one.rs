use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_nodes::{NativeCachePolicy, NativeEffectClass, NativePrimitive, NativeValue};
use sim_runtime::{
    AttemptState, CompiledPlan, InputBinding, NativeImageExecutor, NativeModelSourceTransportHost,
    NativeModelSourceWorkerBinding, NativeNodeRegistry, PromptCompileError, PromptCompiler,
    generated_native_node_registry_projection,
};
use sim_tensor::{CancellationToken, CpuWorkspaceAuthority};
use sim_types::{
    ApiPrompt, AttemptId, NodeId, ProfileId, PromptId, PromptNode, PromptSubmission,
    WorkerModelSourceArtifact, WorkerModelSourceChunk, WorkerModelSourceClosed,
    WorkerModelSourceError, WorkerModelSourceFormat, WorkerModelSourceManifest,
    WorkerModelSourceNestedStateDisposition, WorkerModelSourceOpened, WorkerModelSourceOperation,
    WorkerModelSourceParserLimits, WorkerModelSourceResponse, WorkerModelSourceTensor,
    WorkerSha256Digest,
};
use std::{collections::BTreeMap, error::Error, sync::Arc};
use uuid::Uuid;

fn loader_plan(source: Value) -> Result<CompiledPlan, Box<dyn Error>> {
    resource_loader_plan("AudioEncoderLoader", "audio_encoder_name", source)
}

fn resource_loader_plan(
    class_type: &str,
    source_input: &str,
    source: Value,
) -> Result<CompiledPlan, Box<dyn Error>> {
    let projection = generated_native_node_registry_projection(None)?;
    let mut descriptor = projection
        .descriptor(class_type)
        .cloned()
        .ok_or("production registry omitted AudioEncoderLoader")?;
    // Scheduling this otherwise non-output loader directly isolates its real production binding.
    descriptor.output_node = true;
    let mut registry = NativeNodeRegistry::default();
    registry.register_descriptor(descriptor)?;
    Ok(PromptCompiler::new(&registry).compile(PromptSubmission {
        prompt: ApiPrompt(BTreeMap::from([(
            NodeId::from("audio-loader"),
            PromptNode {
                class_type: class_type.to_owned(),
                inputs: BTreeMap::from([(source_input.to_owned(), source)]),
                unknown: BTreeMap::new(),
            },
        )])),
        prompt_id: Some(PromptId(Uuid::from_u128(0x634_12))),
        client_id: None,
        number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
    })?)
}

#[test]
fn audio_encoder_loader_uses_existing_production_combo_projection() -> Result<(), Box<dyn Error>> {
    let plan = loader_plan(json!("nested/encoder.safetensors"))?;
    plan.validate_integrity()?;
    let node = plan
        .nodes
        .get(&NodeId::from("audio-loader"))
        .ok_or("loader absent")?;
    assert_eq!(node.descriptor.effect, NativeEffectClass::ReadsArtifact);
    assert_eq!(node.descriptor.cache, NativeCachePolicy::InputIdentity);
    assert_eq!(node.descriptor.outputs.len(), 1);
    assert_eq!(
        node.descriptor.outputs.first().ok_or("output absent")?.name,
        "output_0"
    );
    let InputBinding::Literal { value } =
        node.inputs.get("audio_encoder_name").ok_or("name absent")?
    else {
        return Err("filename became a graph link".into());
    };
    assert!(
        matches!(value, NativeValue::PreservedUnknown { type_name, value }
        if type_name == "COMBO" && value == "nested/encoder.safetensors")
    );
    for invalid in [json!(42), json!(false), json!({"filename": "encoder.pt"})] {
        execute_rejected_loader_value(
            invalid,
            "unused.safetensors",
            768,
            false,
            Some("invalid_native_diffusion_input: `audio_encoder_name` must be a string"),
        )?;
    }
    Ok(())
}

fn marker_manifest(width: u64) -> Result<WorkerModelSourceManifest, Box<dyn Error>> {
    let byte_length = width.checked_mul(4).ok_or("marker size overflow")?;
    let bytes = vec![0_u8; usize::try_from(byte_length)?];
    let digest = WorkerSha256Digest::new(format!("{:x}", Sha256::digest(&bytes)))?;
    Ok(WorkerModelSourceManifest {
        source_ordinal: 0,
        model_identity_sha256: WorkerSha256Digest::new(format!(
            "{:x}",
            Sha256::digest(b"task634:malformed-marker")
        ))?,
        artifacts: vec![WorkerModelSourceArtifact {
            sha256: digest,
            byte_size: byte_length,
            format: WorkerModelSourceFormat::Safetensors,
            nested_state_disposition: WorkerModelSourceNestedStateDisposition::Flat,
        }],
        tensors: vec![WorkerModelSourceTensor {
            name: "encoder.layer_norm.bias".to_owned(),
            data_type: "F32".to_owned(),
            shape: vec![width],
            artifact_ordinal: 0,
            byte_offset: 0,
            byte_length,
        }],
        aggregate_tensor_bytes: byte_length,
        maximum_read_bytes: 512 * 1024,
        parser_limits: WorkerModelSourceParserLimits {
            version: 1,
            manifest_bytes: 64 * 1024,
            maximum_depth: 64,
            maximum_tensors: 2048,
            maximum_tensor_bytes: 512 * 1024,
            maximum_aggregate_tensor_bytes: 512 * 1024,
            maximum_name_bytes: 1024,
            maximum_archive_entries: 2048,
            maximum_metadata_values: 2048,
        },
    })
}

fn serve_resource_marker_source(
    host: NativeModelSourceTransportHost,
    expected_folder: &str,
    source_name: String,
    manifest: WorkerModelSourceManifest,
) -> Result<(usize, usize, usize), Box<dyn Error + Send + Sync>> {
    let mut counts = (0, 0, 0);
    while let Ok(call) = smol::block_on(host.receive()) {
        let request = call.request().clone();
        let response = match request.operation {
            WorkerModelSourceOperation::Open {
                folder_category,
                source_names,
            } => {
                assert_eq!(folder_category, expected_folder);
                assert_eq!(source_names.as_slice(), std::slice::from_ref(&source_name));
                counts.0 += 1;
                WorkerModelSourceResponse::Opened(WorkerModelSourceOpened::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    request.context.ordered_source_identity_sha256,
                    vec![manifest.clone()],
                )?)
            }
            WorkerModelSourceOperation::Read {
                source_ordinal,
                tensor_ordinal,
                byte_offset,
                byte_length,
            } => {
                assert_eq!(source_ordinal, 0);
                assert_eq!(tensor_ordinal, 0);
                assert!(
                    byte_offset
                        .checked_add(u64::from(byte_length))
                        .is_some_and(|end| end <= manifest.aggregate_tensor_bytes)
                );
                counts.1 += 1;
                WorkerModelSourceResponse::Chunk(WorkerModelSourceChunk::checked(
                    request.context.session_id,
                    request.call_ordinal,
                    source_ordinal,
                    tensor_ordinal,
                    byte_offset,
                    vec![0; usize::try_from(byte_length)?],
                )?)
            }
            WorkerModelSourceOperation::Close => {
                counts.2 += 1;
                WorkerModelSourceResponse::Closed(WorkerModelSourceClosed::checked(
                    request.context.session_id,
                    request.call_ordinal,
                )?)
            }
            WorkerModelSourceOperation::ReadArtifact { .. } => {
                return Err(WorkerModelSourceError::InvalidOrder.into());
            }
        };
        let call_id = call.call_id();
        smol::block_on(call.respond(call_id, response))?;
    }
    Ok(counts)
}

fn execute_rejected_loader(
    source_name: &str,
    width: u64,
    expect_transport: bool,
) -> Result<(), Box<dyn Error>> {
    execute_rejected_loader_value(
        json!(source_name),
        source_name,
        width,
        expect_transport,
        None,
    )
}

fn execute_rejected_loader_value(
    source: Value,
    source_name: &str,
    width: u64,
    expect_transport: bool,
    expected_error: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    execute_rejected_resource_loader(
        &ResourceLoaderContract {
            class_type: "AudioEncoderLoader",
            source_input: "audio_encoder_name",
            source_folder: "audio_encoders",
            marker_name: "encoder.layer_norm.bias",
        },
        source,
        source_name,
        width,
        expect_transport,
        expected_error,
    )
}

struct ResourceLoaderContract {
    class_type: &'static str,
    source_input: &'static str,
    source_folder: &'static str,
    marker_name: &'static str,
}

fn execute_rejected_resource_loader(
    contract: &ResourceLoaderContract,
    source: Value,
    source_name: &str,
    width: u64,
    expect_transport: bool,
    expected_error: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let plan = resource_loader_plan(contract.class_type, contract.source_input, source)?;
    let workspace_bytes = 128 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let (executor, host) = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x634_120)),
        BTreeMap::new(),
        true,
        Arc::new(backend),
    )?
    .with_model_source_worker(NativeModelSourceWorkerBinding::checked_for_test(
        Uuid::from_u128(0x634_121),
        1,
        1,
        1,
    )?)?;
    let generation = executor.handle_store_generation_for_test();
    let mut manifest = marker_manifest(width)?;
    manifest
        .tensors
        .first_mut()
        .ok_or("marker tensor absent")?
        .name = contract.marker_name.to_owned();
    let source_name = source_name.to_owned();
    let source_folder = contract.source_folder;
    let host_thread = std::thread::spawn(move || {
        serve_resource_marker_source(host, source_folder, source_name, manifest)
    });
    let execution = executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x634_122)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(workspace_bytes)?,
    )?;
    assert_eq!(execution.report.state, AttemptState::Failed);
    assert!(execution.report.outputs.is_empty());
    assert!(execution.report.ui_outputs.is_empty());
    assert!(execution.output_proposals.is_empty());
    assert!(
        execution
            .report
            .error
            .as_deref()
            .is_some_and(|message| !message.is_empty())
    );
    if let Some(expected_error) = expected_error {
        assert!(
            execution
                .report
                .error
                .as_deref()
                .is_some_and(|message| message.contains(expected_error)),
            "production rejection omitted its actual typed source-string failure: {:?}",
            execution.report.error,
        );
    }
    assert!(generation.is_empty());
    assert_eq!(executor.cache_len_for_test(), 0);
    drop(execution);
    drop(executor);
    let counts = host_thread
        .join()
        .map_err(|_| "audio source host panicked")?
        .map_err(|error| error.to_string())?;
    if expect_transport {
        assert!(
            counts.0 > 0,
            "the production COMBO input did not reach its canonical source owner"
        );
        assert!(
            counts.1 > 0,
            "the actual canonical checkpoint constructor did not receive tensor bytes"
        );
        assert_eq!(
            counts.0, counts.2,
            "all opened source sessions must close on failure"
        );
    } else {
        assert_eq!(counts, (0, 0, 0));
    }
    Ok(())
}

#[test]
fn audio_encoder_loader_combo_reaches_canonical_checkpoint_rejection() -> Result<(), Box<dyn Error>>
{
    for width in [512, 768, 1024] {
        execute_rejected_loader("nested/encoder.safetensors", width, true)?;
    }
    Ok(())
}

#[test]
fn audio_encoder_loader_source_name_policy_precedes_transport() -> Result<(), Box<dyn Error>> {
    for name in ["./base.pt", "/base.pt", "../base.pt", "nested/../base.pt"] {
        execute_rejected_loader(name, 768, false)?;
    }
    Ok(())
}

#[test]
fn audio_encoder_loader_original_source_contract_is_preparation_only() -> Result<(), Box<dyn Error>>
{
    let contract: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-loaders-comfy-node-0012/source-contract.json"
    ))?;
    assert_eq!(contract.get("claim"), Some(&json!("COMFY-NODE-0012")));
    let cases = contract
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("source cases missing")?;
    assert_eq!(cases.len(), 15);
    let selected = |label: &str| {
        cases
            .iter()
            .find(|case| case.get("label") == Some(&json!(label)))
            .ok_or_else(|| format!("source case missing: {label}"))
    };
    for (label, model_type, width) in [
        ("base configuration", "wav2vec2", Some(768)),
        ("large prefixed configuration", "wav2vec2", Some(1024)),
        ("whisper configuration", "whisper3", None),
    ] {
        let case = selected(label)?;
        let configuration = case
            .get("source_preparation_gate")
            .and_then(|gate| gate.get("configuration"))
            .ok_or("constructor gate absent")?;
        assert_eq!(configuration.get("model_type"), Some(&json!(model_type)));
        if let Some(width) = width {
            assert_eq!(configuration.get("embed_dim"), Some(&json!(width)));
        }
        let events = case
            .get("events")
            .and_then(Value::as_array)
            .ok_or("source events absent")?;
        assert!(events.iter().any(
            |event| event.get("phase") == Some(&json!("source_torch_load"))
                && event.get("safe_load") == Some(&json!(true))
        ));
        assert_eq!(
            events.last().and_then(|event| event.get("phase")),
            Some(&json!("original_model_constructor_boundary"))
        );
    }
    for label in [
        "missing file",
        "corrupt safe torch file",
        "unsupported embedding width",
        "wrong marker rank",
        "unrecognized state",
        "empty state",
    ] {
        assert!(selected(label)?.get("source_error").is_some());
    }
    for label in [
        "normalized source filename",
        "original absolute prefix normalization",
        "original traversal normalization",
    ] {
        assert!(selected(label)?.get("source_preparation_gate").is_some());
    }
    Ok(())
}

#[test]
fn audio_encoder_loader_cancelled_attempt_does_not_open_source() -> Result<(), Box<dyn Error>> {
    execute_cancelled_resource_loader("AudioEncoderLoader", "audio_encoder_name", "audio_encoders")
}

fn execute_cancelled_resource_loader(
    class_type: &str,
    source_input: &str,
    source_folder: &'static str,
) -> Result<(), Box<dyn Error>> {
    let plan = resource_loader_plan(class_type, source_input, json!("encoder.safetensors"))?;
    let workspace_bytes = 128 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let (executor, host) = NativeImageExecutor::new_with_generated_registry(
        ProfileId(Uuid::from_u128(0x634_130)),
        BTreeMap::new(),
        true,
        Arc::new(backend),
    )?
    .with_model_source_worker(NativeModelSourceWorkerBinding::checked_for_test(
        Uuid::from_u128(0x634_131),
        1,
        1,
        1,
    )?)?;
    let generation = executor.handle_store_generation_for_test();
    let manifest = marker_manifest(768)?;
    let host_thread = std::thread::spawn(move || {
        serve_resource_marker_source(
            host,
            source_folder,
            "encoder.safetensors".to_owned(),
            manifest,
        )
    });
    let cancellation = CancellationToken::default();
    assert!(cancellation.cancel());
    match executor.execute_blocking(
        &plan,
        AttemptId(Uuid::from_u128(0x634_132)),
        cancellation,
        0,
        authority.authorize_workspace(workspace_bytes)?,
    ) {
        Ok(execution) => {
            assert_eq!(execution.report.state, AttemptState::Cancelled);
            assert!(execution.report.outputs.is_empty());
            assert!(execution.report.ui_outputs.is_empty());
            assert!(execution.output_proposals.is_empty());
        }
        Err(error) => assert!(error.to_string().to_ascii_lowercase().contains("cancel")),
    }
    assert!(generation.is_empty());
    assert_eq!(executor.cache_len_for_test(), 0);
    drop(executor);
    assert_eq!(
        host_thread
            .join()
            .map_err(|_| "cancelled source host panicked")?
            .map_err(|error| error.to_string())?,
        (0, 0, 0)
    );
    Ok(())
}

#[test]
fn reduced_audio_resource_handle_roundtrip_consumes_real_tensors() -> Result<(), Box<dyn Error>> {
    use sim_media::NativeAudioPayload;
    use sim_model::audio_encoder::deterministic_reduced_audio_encoder_fixture;
    use sim_model::{NativeAudioEncoderArchitecture, NativeModelPayload};
    use sim_nodes::{NativeStoredModelPayload, NativeStoredPayload};
    use sim_runtime::NativeHandleStoreGeneration;
    use sim_tensor::generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    };
    use sim_tensor::{DType, StreamId, TensorBackend};

    let oracle: Value = serde_json::from_str(include_str!(
        "../fixtures/models/audio-encoder-resource-foundation/oracle.json"
    ))?;
    let workspace_bytes = 128 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(workspace_bytes)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(workspace_bytes)?,
        &cancellation,
    );
    for architecture in [
        NativeAudioEncoderArchitecture::Wav2Vec2Base,
        NativeAudioEncoderArchitecture::Wav2Vec2Large,
        NativeAudioEncoderArchitecture::WhisperLargeV3,
    ] {
        let resource = Arc::new(deterministic_reduced_audio_encoder_fixture(
            &backend,
            &context,
            architecture,
            0,
        )?);
        let model = Arc::new(NativeModelPayload::audio_encoder(resource.clone())?);
        let stored =
            NativeStoredPayload::Model(Arc::new(NativeStoredModelPayload::model_resource(model)?));
        let resident_bytes = stored.resident_bytes()?;
        let generation = NativeHandleStoreGeneration::with_capacities(1, resident_bytes)?;
        let store = generation.handle_store_for_attempt(AttemptId(Uuid::new_v4()));
        let handle = store.publish(stored, &cancellation)?;
        assert_eq!(handle.handle_type().type_id, "AUDIO_ENCODER");
        let resolved = store.resolve(&handle, handle.handle_type(), &cancellation)?;
        let NativeStoredPayload::Model(stored) = resolved.as_ref() else {
            return Err("AUDIO_ENCODER was not a concrete native model resource".into());
        };
        let loaded = stored
            .model_payload()
            .audio_encoder_resource()
            .ok_or("audio resource absent")?;
        assert!(Arc::ptr_eq(loaded, &resource));
        let samples: usize = if architecture == NativeAudioEncoderArchitecture::WhisperLargeV3 {
            29
        } else {
            1600
        };
        let values = (0..4 * samples)
            .map(|index| {
                let batch = index / (2 * samples);
                let channel = (index / samples) % 2;
                let sample = index % samples;
                ((sample as f32 * 0.013 + channel as f32 * 0.31).sin() * (1.0 + batch as f32 * 1.7))
                    + batch as f32 * 0.2
            })
            .collect::<Vec<_>>();
        let waveform = tensor_from_f32_with_context_exact_native(
            &backend,
            &[2, 2, u64::try_from(samples)?],
            &values,
            DType::F32,
            backend.device(),
            &context,
        )?;
        let audio = NativeAudioPayload::checked(waveform, 16000)?;
        let output = loaded.encode(&backend, &audio, &context)?;
        let encoded = output.encoded_audio().ok_or("real encoder output absent")?;
        let actual = tensor_to_f32_with_context_exact_native(&backend, encoded, &context)?;
        let expected = oracle
            .get("results")
            .and_then(|results| results.get(architecture.identifier()))
            .and_then(|result| result.get("values"))
            .and_then(Value::as_array)
            .ok_or("reduced oracle absent")?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let expected = expected.as_f64().ok_or("non-numeric reduced oracle")? as f32;
            let tolerance = 0.0002_f32 + 0.0002_f32 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{}[{index}] expected{expected}, actual{actual}",
                architecture.identifier()
            );
        }
        assert_eq!(output.audio_samples(), Some(u64::try_from(samples)?));
        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(
            store
                .resolve(&handle, handle.handle_type(), &cancelled)
                .is_err()
        );
        drop(output);
        drop(resolved);
        store.revoke(&handle, &cancellation)?;
        assert!(generation.is_empty());
    }
    Ok(())
}

const CLIP_VISION_LOADER_CONTRACT: ResourceLoaderContract = ResourceLoaderContract {
    class_type: "CLIPVisionLoader",
    source_input: "clip_name",
    source_folder: "clip_vision",
    marker_name: "vision_model.encoder.layers.22.layer_norm1.weight",
};

#[test]
fn clip_vision_loader_uses_existing_production_combo_projection() -> Result<(), Box<dyn Error>> {
    let plan = resource_loader_plan("CLIPVisionLoader", "clip_name", json!("nested/vision.pt"))?;
    plan.validate_integrity()?;
    let node = plan
        .nodes
        .get(&NodeId::from("audio-loader"))
        .ok_or("vision loader absent")?;
    assert_eq!(node.descriptor.effect, NativeEffectClass::ReadsArtifact);
    assert_eq!(node.descriptor.cache, NativeCachePolicy::InputIdentity);
    assert_eq!(node.descriptor.outputs.len(), 1);
    let InputBinding::Literal { value } = node
        .inputs
        .get("clip_name")
        .ok_or("vision filename absent")?
    else {
        return Err("vision filename became a graph link".into());
    };
    assert!(
        matches!(value, NativeValue::Primitive { value: NativePrimitive::String(value) }
        if value == "nested/vision.pt")
    );
    for invalid in [json!(42), json!(false), json!({"filename": "vision.pt"})] {
        assert_clip_vision_literal_rejected(invalid)?;
    }
    Ok(())
}

#[test]
fn clip_vision_loader_source_configuration_gap_is_atomic() -> Result<(), Box<dyn Error>> {
    execute_rejected_resource_loader(
        &CLIP_VISION_LOADER_CONTRACT,
        json!("nested/vision.pt"),
        "nested/vision.pt",
        2,
        true,
        Some("CLIP vision source omitted its semantic configuration"),
    )
}

#[test]
fn clip_vision_loader_rejects_literal_lists_before_execution() -> Result<(), Box<dyn Error>> {
    assert_clip_vision_literal_rejected(json!(["vision.pt", "vision.pt"]))
}

fn assert_clip_vision_literal_rejected(value: Value) -> Result<(), Box<dyn Error>> {
    let error = resource_loader_plan("CLIPVisionLoader", "clip_name", value)
        .err()
        .ok_or("invalid V1 filename literal compiled")?;
    assert!(matches!(error.downcast_ref::<PromptCompileError>(),
        Some(PromptCompileError::InvalidLiteral { node, input })
        if node == &NodeId::from("audio-loader") && input == "clip_name"));
    Ok(())
}

#[test]
fn clip_vision_loader_source_name_policy_precedes_transport() -> Result<(), Box<dyn Error>> {
    for name in [
        "./vision.pt",
        "/vision.pt",
        "../vision.pt",
        "nested/../vision.pt",
    ] {
        execute_rejected_resource_loader(
            &CLIP_VISION_LOADER_CONTRACT,
            json!(name),
            name,
            2,
            false,
            None,
        )?;
    }
    Ok(())
}

#[test]
fn clip_vision_loader_cancelled_attempt_does_not_open_source() -> Result<(), Box<dyn Error>> {
    execute_cancelled_resource_loader("CLIPVisionLoader", "clip_name", "clip_vision")
}

#[test]
fn clip_vision_loader_original_source_contract_preserves_configuration_gates()
-> Result<(), Box<dyn Error>> {
    let contract: Value = serde_json::from_str(include_str!(
        "../fixtures/nodes/model-loaders-comfy-node-0076/source-contract.json"
    ))?;
    assert_eq!(contract.get("claim"), Some(&json!("COMFY-NODE-0076")));
    let cases = contract
        .get("cases")
        .and_then(Value::as_array)
        .ok_or("vision source cases absent")?;
    assert_eq!(cases.len(), 23);
    assert_eq!(
        cases
            .iter()
            .filter(|case| case.get("source_preparation_gate").is_some())
            .count(),
        16
    );
    assert_eq!(
        cases
            .iter()
            .filter(|case| case.get("source_error").is_some())
            .count(),
        7
    );
    let selected = |label: &str| {
        cases
            .iter()
            .find(|case| case.get("label") == Some(&json!(label)))
            .ok_or_else(|| format!("vision source case missing: {label}"))
    };
    for (label, configuration) in [
        (
            "CLIP L source selection",
            "comfy/clip_vision_config_vitl.json",
        ),
        ("CLIP H source selection", "comfy/clip_vision_config_h.json"),
        ("CLIP G source selection", "comfy/clip_vision_config_g.json"),
        (
            "SigLIP2 source selection",
            "comfy/clip_vision_siglip2_base_naflex.json",
        ),
        (
            "OpenAI key conversion",
            "comfy/clip_vision_config_vitl.json",
        ),
    ] {
        let case = selected(label)?;
        assert_eq!(
            case.get("source_preparation_gate")
                .and_then(|gate| gate.get("configuration"))
                .and_then(|configuration| configuration.get("source")),
            Some(&json!(configuration))
        );
        let events = case
            .get("events")
            .and_then(Value::as_array)
            .ok_or("vision source events absent")?;
        assert!(events.iter().any(
            |event| event.get("phase") == Some(&json!("source_torch_load"))
                && event.get("safe_load_argument") == Some(&json!(false))
                && event.get("actual_weights_only") == Some(&json!(true))
        ));
    }
    for label in ["unrecognized original state", "empty original state"] {
        assert_eq!(
            selected(label)?
                .get("source_error")
                .and_then(|error| error.get("message")),
            Some(&json!(
                "ERROR: clip vision file is invalid and does not contain a valid vision model."
            ))
        );
    }
    assert_eq!(
        selected("unsupported SigLIP positions")?
            .get("source_error")
            .and_then(|error| error.get("type")),
        Some(&json!("UnboundLocalError"))
    );
    Ok(())
}
