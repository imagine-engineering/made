use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, MappedModelWeights, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_stableaudio_sim_model_0132::{
        MODEL_FAMILY, MODEL_FAMILY_FEATURE_ID, MODEL_FAMILY_FIXTURE, MODEL_FAMILY_IDENTIFIER,
        MODEL_FAMILY_MEMORY_USAGE_FACTOR, MODEL_FAMILY_PROJECTION_SHA256,
        MODEL_FAMILY_REGISTRATION, MODEL_FAMILY_SIGMA_MAX, MODEL_FAMILY_SIGMA_MIN,
        MODEL_FAMILY_SOURCE_ORDINAL, MODEL_FAMILY_SOURCE_PATH, MODEL_FAMILY_SOURCE_SHA256,
        SOURCE_ARCHITECTURE, StableAudioAttentionConfiguration, StableAudioTimestepFeatures,
        configuration_for_probe,
    },
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        cast_to_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9132",
    identifier: "StableAudioAmbiguousFixture",
    ..MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 121,
        source_architecture: "model_base.StableAudioAmbiguousFixture",
        ..MODEL_FAMILY_REGISTRATION
    },
];

#[derive(Debug, Deserialize)]
struct FamilyFixture {
    fixture_id: String,
    feature_id: String,
    detector: DetectorFixture,
    base_artifact_digest: String,
    source_weights: Vec<TensorFixture>,
    input: TensorFixture,
    dtype: DType,
    device: DeviceKind,
    activation_elements: u64,
    memory_budget_bytes: u64,
    expected_memory_bytes: u64,
    checkpoints: Vec<CheckpointFixture>,
    patches: Vec<PatchOperation>,
    patched_checkpoints: Vec<CheckpointFixture>,
}

#[derive(Debug, Deserialize)]
struct DetectorFixture {
    tensor_shapes: BTreeMap<String, Vec<u64>>,
    metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize)]
struct TensorFixture {
    key: String,
    shape: Vec<u64>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CheckpointFixture {
    name: String,
    values: Vec<f32>,
}

#[test]
fn val_model_family_row_001_stableaudio_source_projection_descriptor_and_store_adapter()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    verify_source_projection()?;
    verify_descriptor(&fixture)?;

    let probe = probe_through_model_store(&fixture)?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    let configuration = configuration_for_probe(&probe)?;
    assert_eq!(configuration.global_condition_dimension, 2);
    assert!(!configuration.project_condition_tokens);
    assert_eq!(configuration.embedding_dimension, 2);
    assert_eq!(configuration.memory_tokens, None);
    assert_eq!(
        configuration.attention,
        StableAudioAttentionConfiguration::LayerNormalized {
            feature_scale: true
        }
    );
    assert_eq!(
        configuration.timestep_features,
        StableAudioTimestepFeatures::Learned
    );
    assert_eq!(configuration.io_channels, 2);
    assert_eq!(configuration.input_concat_dimension, 1);
    assert_eq!(configuration.local_add_condition_dimension, None);
    assert_eq!(configuration.depth, 1);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().identity.identifier(), MODEL_FAMILY_IDENTIFIER);
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(resolved.source_architecture(), SOURCE_ARCHITECTURE);
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(
        resolved.clip_target().candidates()[0].tokenizer().identifier(),
        "sim.text_encoders.sa_t5.SAT5Tokenizer"
    );
    assert_eq!(
        resolved.clip_target().candidates()[0].clip_model().target().as_str(),
        "sim.text_encoders.sa_t5.SAT5Model"
    );

    verify_transactional_mapping(&fixture, &probe)?;
    verify_owner_delegation()?;
    super::write_model_family_row_artifact(
        MODEL_FAMILY_FIXTURE,
        MODEL_FAMILY_FEATURE_ID,
        MODEL_FAMILY_IDENTIFIER,
        MODEL_FAMILY_SOURCE_ORDINAL,
        "stableaudio_comfy_model_0132",
        &[
            "source-and-catalog-provenance",
            "model-store-authoritative-probe",
            "typed-audio-configuration",
            "sat5-target-and-stableaudio1-latent",
            "transactional-component-routing-and-zero-beta-drop",
            "native-forward-conditioning-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "diffusers-stableaudio3-partial-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_stableaudio_forward_patch_memory_and_failure_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    let probe = fixture_probe(&fixture);
    verify_execution(&fixture, &probe)?;
    verify_failures(&fixture, &probe)?;
    Ok(())
}

fn verify_source_projection() -> Result<(), Box<dyn std::error::Error>> {
    let provenance: serde_json::Value = serde_json::from_slice(&fs::read(provenance_path())?)?;
    assert_eq!(provenance["schema_version"], 1);
    assert_eq!(provenance["fixture_id"], MODEL_FAMILY_FIXTURE);
    assert_eq!(provenance["feature_id"], MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(provenance["source_architecture"], SOURCE_ARCHITECTURE);
    assert_eq!(provenance["latent_feature_id"], "COMFY-MODEL-0050");
    assert_eq!(provenance["latent_identifier"], "StableAudio1");
    assert_eq!(
        provenance["catalog_projection_sha256"],
        MODEL_FAMILY_PROJECTION_SHA256
    );
    assert!(
        provenance["alternate_layout"]
            .as_str()
            .is_some_and(|value| value.contains("fails closed"))
    );
    assert!(
        provenance["oracle"]
            .as_str()
            .is_some_and(|value| value.contains("no production Python dependency"))
    );
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("missing StableAudio source projection")?;
    assert!(projection.contains("unet_config.audio_model=dit1.0"));
    assert!(projection.contains("conditioning=seconds_start[0,512],seconds_total[0,512]"));
    assert!(projection.contains("diffusers_support=none"));
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
            .as_str()
            .ok_or("missing StableAudio projection digest")?
    );

    let repository_root = repository_root();
    for source in provenance["source_files"]
        .as_array()
        .ok_or("missing StableAudio source files")?
    {
        let path = source["path"].as_str().ok_or("missing source path")?;
        let expected = source["sha256"].as_str().ok_or("missing source digest")?;
        assert_eq!(sha256(&fs::read(repository_root.join(path))?), expected);
    }
    assert_eq!(MODEL_FAMILY_SOURCE_PATH, provenance["source_files"][0]["path"]);
    assert_eq!(MODEL_FAMILY_SOURCE_SHA256, provenance["source_files"][0]["sha256"]);

    let catalog: serde_json::Value = serde_json::from_slice(&fs::read(
        repository_root.join("crates/sim_model/catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .and_then(|models| {
            models
                .iter()
                .find(|model| model["source_symbol"] == MODEL_FAMILY_IDENTIFIER)
        })
        .ok_or("StableAudio catalog row is absent")?;
    assert_eq!(sha256(&serde_json::to_vec(row)?), MODEL_FAMILY_PROJECTION_SHA256);
    assert_eq!(row["source_ordinal"], MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(row["static"]["unet_config"]["value"]["audio_model"], "dit1.0");
    Ok(())
}

fn verify_descriptor(fixture: &FamilyFixture) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(fixture.fixture_id, MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture.feature_id, MODEL_FAMILY_FEATURE_ID);
    assert_eq!(fixture.dtype, DType::F32);
    assert_eq!(fixture.device, DeviceKind::Cpu);
    assert_eq!(fixture.activation_elements, 2);
    assert_eq!(fixture.memory_budget_bytes, fixture.expected_memory_bytes);
    assert_eq!(MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);
    assert_eq!(MODEL_FAMILY_SIGMA_MAX, 500.0);
    assert_eq!(MODEL_FAMILY_SIGMA_MIN, 0.03);
    assert_eq!(MODEL_FAMILY.supported_dtypes, [DType::F16, DType::Bf16, DType::F32]);
    assert_eq!(MODEL_FAMILY.supported_devices, [DeviceKind::Cpu]);
    let descriptor = describe_model_family(&MODEL_FAMILY)?;
    assert_eq!(descriptor.family, MODEL_FAMILY_FEATURE_ID);
    assert_eq!(descriptor.identifier, MODEL_FAMILY_IDENTIFIER);
    assert_eq!(descriptor.latent_format, "StableAudio1");
    assert_eq!(descriptor.component_graph.len(), 5);
    assert_eq!(descriptor.component_graph[0].identifier, "denoiser");
    assert_eq!(descriptor.component_graph[1].identifier, "seconds_start_conditioner");
    assert_eq!(descriptor.component_graph[2].identifier, "seconds_total_conditioner");
    assert!(descriptor.component_graph[..3].iter().all(|component| component.required));
    assert!(!descriptor.component_graph[3].required);
    assert!(!descriptor.component_graph[4].required);
    Ok(())
}

fn verify_transactional_mapping(
    fixture: &FamilyFixture,
    probe: &ModelProbe,
) -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(fixture, DType::F32, &backend, &context)?;
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let mapped = registry.resolve(probe)?.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 5);
    let denoiser = mapped.component("denoiser").ok_or("missing StableAudio denoiser")?;
    assert!(denoiser.contains_key("native.to_global_embed.0.weight"));
    assert!(!denoiser.keys().any(|key| key.ends_with(".beta")));
    assert!(
        mapped
            .component("seconds_start_conditioner")
            .is_some_and(|component| component.contains_key("embedder.embedding.0.weights"))
    );
    assert!(
        mapped
            .component("seconds_total_conditioner")
            .is_some_and(|component| component.contains_key("embedder.embedding.0.weights"))
    );
    assert!(
        mapped
            .component("text_encoder")
            .is_some_and(|component| component.contains_key("model.t5.encoder.weight"))
    );
    assert!(
        mapped
            .component("vae")
            .is_some_and(|component| component.contains_key("model.decoder.weight"))
    );
    assert!(mapped.binding().is_some());
    Ok(())
}

fn verify_execution(
    fixture: &FamilyFixture,
    probe: &ModelProbe,
) -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(probe)?;
    let source = source_tensors(fixture, DType::F32, &backend, &context)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    let options = NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: fixture.activation_elements,
        memory_budget_bytes: 1024 * 1024,
        allow_unexpected_weights: true,
    };
    let model = build_model_family_for_probe(&registry, probe, weights.clone(), options)?;
    assert_eq!(model.memory_estimate().parameter_elements, 38);
    assert_eq!(model.memory_estimate().weight_bytes, 152);
    assert_eq!(model.memory_estimate().activation_bytes, 8);
    assert_eq!(model.memory_estimate().total_bytes, 160);

    let input = tensor(
        &backend,
        &fixture.input.shape,
        &fixture.input.values,
        DType::F32,
        &context,
    )?;
    assert_checkpoints(
        &backend,
        model.forward_checkpoints(&backend, &input, &context)?,
        &fixture.checkpoints,
        &context,
    )?;

    let patch = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?;
    let patched_weights = patch.apply(&backend, &weights, &context)?;
    assert_ne!(patched_weights.cache_identity(), weights.cache_identity());
    let patched = model.with_weights(patched_weights)?;
    assert_checkpoints(
        &backend,
        patched.forward_checkpoints(&backend, &input, &context)?,
        &fixture.patched_checkpoints,
        &context,
    )?;

    let add = fixture.patches[0].clone();
    let replace = PatchOperation {
        identifier: "replace-condition-token-projection".to_owned(),
        kind: PatchKind::Adapter,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.to_cond_embed.0.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![4.0, 0.0, 0.0, 4.0],
            application: PatchApplication::Replace,
        }],
    };
    let ordered = PatchGraph::checked(
        &fixture.base_artifact_digest,
        vec![replace.clone(), add.clone()],
    )?
    .apply(&backend, &weights, &context)?;
    let reversed = PatchGraph::checked(&fixture.base_artifact_digest, vec![add, replace])?
        .apply(&backend, &weights, &context)?;
    assert_ne!(
        tensor_values(&backend, &ordered, "native.to_cond_embed.0.weight", &context)?,
        tensor_values(&backend, &reversed, "native.to_cond_embed.0.weight", &context)?
    );

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let typed_source = source_tensors(fixture, dtype, &backend, &context)?;
        let typed_weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            &fixture.base_artifact_digest,
            &typed_source,
        )?;
        let mut typed_options = options;
        typed_options.dtype = dtype;
        assert!(
            build_model_family_for_probe(&registry, probe, typed_weights, typed_options).is_ok()
        );
    }
    let mut unsupported_dtype = options;
    unsupported_dtype.dtype = DType::F64;
    assert!(matches!(
        build_model_family_for_probe(&registry, probe, weights.clone(), unsupported_dtype),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
    let mut unsupported_device = options;
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, probe, weights.clone(), unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));
    let mut oom = options;
    oom.memory_budget_bytes = model.memory_estimate().total_bytes - 1;
    assert!(matches!(
        build_model_family_for_probe(&registry, probe, weights, oom),
        Err(ModelFamilyError::OutOfMemory { required: 160, budget: 159 })
    ));
    Ok(())
}

fn verify_failures(
    fixture: &FamilyFixture,
    probe: &ModelProbe,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;

    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("conv_in.weight".to_owned(), vec![2, 2, 1]),
            ("conv_out.weight".to_owned(), vec![2, 2, 1]),
        ]),
        metadata: BTreeMap::from([("model_layout".to_owned(), "diffusers".to_owned())]),
    };
    assert!(matches!(
        configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("no StableAudio entry")
    ));
    assert!(matches!(registry.detect(&diffusers), Err(ModelFamilyError::NoDetectionMatch)));

    let mut misleading_metadata = probe.clone();
    misleading_metadata
        .metadata
        .insert("audio_model".to_owned(), "not-dit1.0".to_owned());
    misleading_metadata.metadata.insert(
        "model_layout".to_owned(),
        "prefixed-native".to_owned(),
    );
    assert_eq!(
        registry
            .resolve(&misleading_metadata)?
            .detection()
            .identity
            .feature_id(),
        MODEL_FAMILY_FEATURE_ID
    );

    let mut stable_audio_3 = probe.clone();
    stable_audio_3.tensor_shapes.insert(
        "model.model.transformer.global_cond_embedder.0.weight".to_owned(),
        vec![2, 2],
    );
    assert!(matches!(
        registry.resolve(&stable_audio_3),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("StableAudio3")
    ));

    let mut partial = probe.clone();
    partial.tensor_shapes.remove(
        "conditioner.conditioners.seconds_start.embedder.embedding.0.weights",
    );
    assert!(matches!(registry.detect(&partial), Err(ModelFamilyError::NoDetectionMatch)));

    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "model.model.transformer.project_in.weight".to_owned(),
        vec![2, 1],
    );
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("narrower than audio I/O")
    ));

    let mut mixed_norms = probe.clone();
    mixed_norms.tensor_shapes.insert(
        "model.model.transformer.layers.0.self_attn.q_norm.gamma".to_owned(),
        vec![2],
    );
    assert!(matches!(
        registry.resolve(&mixed_norms),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("both layer-normalized and RMS-normalized")
    ));

    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(fixture, DType::F32, &backend, &context)?;
    cancellation.cancel();
    assert!(
        registry
            .resolve(probe)?
            .map_state_dictionary(
                &ModelStateTransaction::new(&backend, &context),
                &fixture.base_artifact_digest,
                &source,
            )
            .is_err()
    );
    Ok(())
}

fn fixture() -> Result<FamilyFixture, Box<dyn std::error::Error>> {
    let fixture: FamilyFixture = serde_json::from_slice(&fs::read(fixture_path())?)?;
    assert_eq!(fixture.fixture_id, MODEL_FAMILY_FIXTURE);
    assert!(!fixture.patches.is_empty());
    assert_eq!(fixture.checkpoints.len(), fixture.patched_checkpoints.len());
    Ok(fixture)
}

fn fixture_probe(fixture: &FamilyFixture) -> ModelProbe {
    ModelProbe {
        tensor_shapes: fixture.detector.tensor_shapes.clone(),
        metadata: fixture.detector.metadata.clone(),
    }
}

fn probe_through_model_store(
    fixture: &FamilyFixture,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("stableaudio.safetensors");
    let tensors = probe_tensor_fixtures(fixture)?;
    write_safetensors(&model_path, &tensors)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "stableaudio-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("stableaudio-row", "stableaudio.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let probe = store.family_probe(&loaded, &cancellation)?;
    assert_eq!(probe.tensor_shapes(), &fixture.detector.tensor_shapes);
    Ok(probe)
}

fn source_tensors(
    fixture: &FamilyFixture,
    dtype: DType,
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe_tensor_fixtures(fixture)?
        .iter()
        .map(|fixture| {
            Ok((
                fixture.key.clone(),
                tensor(
                    backend,
                    &fixture.shape,
                    &fixture.values,
                    dtype,
                    context,
                )?,
            ))
        })
        .collect()
}

fn probe_tensor_fixtures(
    fixture: &FamilyFixture,
) -> Result<Vec<TensorFixture>, Box<dyn std::error::Error>> {
    fixture
        .detector
        .tensor_shapes
        .iter()
        .map(|(key, shape)| {
            let count = usize::try_from(shape.iter().try_fold(1_u64, |total, dimension| {
                total.checked_mul(*dimension).ok_or("fixture shape overflow")
            })?)?;
            let values = fixture
                .source_weights
                .iter()
                .find(|weight| weight.key == *key)
                .map_or_else(|| vec![0.0; count], |weight| weight.values.clone());
            if values.len() != count {
                return Err(format!("{key} value count mismatch").into());
            }
            Ok(TensorFixture {
                key: key.clone(),
                shape: shape.clone(),
                values,
            })
        })
        .collect()
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
    if dtype == DType::F32 {
        Ok(tensor)
    } else {
        Ok(cast_to_with_context_exact_native(
            backend,
            &tensor,
            dtype,
            backend.device(),
            false,
            false,
            context,
        )?)
    }
}

fn tensor_values(
    backend: &CpuBackend,
    weights: &MappedModelWeights,
    key: &str,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        weights.tensors().get(key).ok_or("missing patched weight")?,
        context,
    )?)
}

fn assert_checkpoints(
    backend: &CpuBackend,
    actual: Vec<sim_model::ModelForwardCheckpoint>,
    expected: &[CheckpointFixture],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.name, expected.name);
        let values =
            tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
        assert_eq!(values.len(), expected.values.len());
        for (index, (actual_value, expected_value)) in
            values.iter().zip(&expected.values).enumerate()
        {
            if (actual_value - expected_value).abs() > 1.0e-5 {
                return Err(format!(
                    "checkpoint {} value {index}: expected {expected_value}, got {actual_value}",
                    actual.name
                )
                .into());
            }
        }
    }
    Ok(())
}

fn write_safetensors(
    path: &Path,
    tensors: &[TensorFixture],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let start = data.len();
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            tensor.key.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn verify_owner_delegation() -> Result<(), Box<dyn std::error::Error>> {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/families/stableaudio_sim_model_0132.rs"
    ));
    for canonical in [
        "ModelFamilyRegistration",
        "ModelFamilyStatePlanSelector",
        "ModelStateTransformPlanDefinition",
        "ModelProbe",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(source.contains(canonical));
    }
    for forbidden_owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "std::fs",
        "File::",
        "unsafe ",
        "std::process",
        "Command::",
        "python",
    ] {
        assert!(!source.contains(forbidden_owner));
    }
    Ok(())
}

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../sim_test_support/fixtures/models/stableaudio-comfy-model-0132/family.json",
    )
}

fn provenance_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../sim_test_support/fixtures/models/stableaudio-comfy-model-0132/provenance.json",
    )
}

fn repository_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
