use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_stableaudio3_sim_model_0133 as sa3,
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

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [sa3::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9133",
    identifier: "StableAudio3AmbiguousFixture",
    ..sa3::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sa3::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 120,
        source_architecture: "model_base.StableAudio3AmbiguousFixture",
        ..sa3::MODEL_FAMILY_REGISTRATION
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
fn val_model_family_row_001_stableaudio3_source_configuration_and_state_transform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    verify_source_projection()?;
    assert_eq!(fixture.fixture_id, sa3::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture.feature_id, sa3::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(fixture.dtype, DType::F32);
    assert_eq!(fixture.device, DeviceKind::Cpu);
    assert_eq!(fixture.memory_budget_bytes, fixture.expected_memory_bytes);
    assert_eq!(sa3::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 7.0);
    assert_eq!(sa3::MODEL_FAMILY_SHIFT, 2.0);
    assert_eq!(sa3::MODEL_FAMILY_MULTIPLIER, 1.0);

    let descriptor = describe_model_family(&sa3::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "StableAudio3");
    assert_eq!(descriptor.component_graph.len(), 5);
    assert_eq!(descriptor.component_graph[0].identifier, "denoiser");
    assert_eq!(descriptor.component_graph[1].identifier, "seconds_total_conditioner");
    assert_eq!(descriptor.component_graph[2].identifier, "prompt_conditioner");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);

    let probe = probe_through_model_store(&fixture)?;
    let configuration = sa3::configuration_for_probe(&probe)?;
    assert_eq!(configuration.global_condition_dimension, 2);
    assert!(!configuration.project_condition_tokens);
    assert_eq!(configuration.embedding_dimension, 2);
    assert_eq!(configuration.memory_tokens, None);
    assert_eq!(
        configuration.attention,
        sa3::StableAudio3AttentionConfiguration::LayerNormalized
    );
    assert!(configuration.learned_timestep_features);
    assert_eq!(configuration.io_channels, 2);
    assert_eq!(configuration.input_concat_dimension, 1);
    assert_eq!(configuration.local_add_condition_dimension, None);
    assert_eq!(configuration.depth, 1);
    assert!(configuration.shared_global_embedding);
    assert_eq!(configuration.max_text_tokens, 256);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), sa3::MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(resolved.source_architecture(), sa3::SOURCE_ARCHITECTURE);
    let clip = &resolved.clip_target().candidates()[0];
    assert_eq!(
        clip.tokenizer().identifier(),
        "sim.text_encoders.sa3.SAT5GemmaTokenizer"
    );
    assert_eq!(
        clip.clip_model().target().as_str(),
        "sim.text_encoders.sa3.SAT5GemmaModel"
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&fixture, DType::F32, &backend, &context)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 5);
    let denoiser = mapped.component("denoiser").ok_or("missing denoiser")?;
    assert!(denoiser.contains_key("native.transformer.global_cond_embedder.0.weight"));
    assert!(!denoiser.keys().any(|key| key.ends_with(".beta")));
    assert!(mapped.component("seconds_total_conditioner").is_some_and(|state| {
        state.contains_key("embedder.embedding.0.weights")
    }));
    assert!(mapped.component("prompt_conditioner").is_some_and(|state| {
        state.contains_key("padding_embedding")
    }));
    assert!(mapped.component("text_encoder").is_some_and(|state| {
        state.contains_key("model.gemma.encoder.weight")
    }));
    assert!(mapped.component("vae").is_some_and(|state| {
        state.contains_key("model.decoder.weight")
    }));

    verify_failures(&fixture, &registry, &probe)?;
    verify_owner_delegation()?;
    super::write_model_family_row_artifact(
        sa3::MODEL_FAMILY_FIXTURE,
        sa3::MODEL_FAMILY_FEATURE_ID,
        sa3::MODEL_FAMILY_IDENTIFIER,
        sa3::MODEL_FAMILY_SOURCE_ORDINAL,
        "stableaudio3_comfy_model_0133",
        &[
            "source-and-catalog-provenance",
            "typed-shared-global-audio-configuration",
            "sat5-gemma-target-and-stableaudio3-latent",
            "transactional-duration-padding-text-vae-routing",
            "zero-beta-filter-and-native-forward",
            "memory-oom-dtype-device-cancellation",
            "diffusers-stableaudio1-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_stableaudio3_forward_patch_memory_and_platform_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    let probe = fixture_probe(&fixture);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&fixture, DType::F32, &backend, &context)?;
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
    let model = build_model_family_for_probe(&registry, &probe, weights.clone(), options)?;
    assert_eq!(model.memory_estimate().total_bytes, 308);
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
    assert_ne!(weights.cache_identity(), patched_weights.cache_identity());
    assert_checkpoints(
        &backend,
        model
            .with_weights(patched_weights)?
            .forward_checkpoints(&backend, &input, &context)?,
        &fixture.patched_checkpoints,
        &context,
    )?;

    let add = fixture.patches[0].clone();
    let replace = PatchOperation {
        identifier: "replace-stableaudio3-token-projection".to_owned(),
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
        tensor_to_f32_with_context_exact_native(
            &backend,
            ordered
                .tensors()
                .get("native.to_cond_embed.0.weight")
                .ok_or("missing ordered token projection")?,
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            reversed
                .tensors()
                .get("native.to_cond_embed.0.weight")
                .ok_or("missing reversed token projection")?,
            &context,
        )?
    );

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let source = source_tensors(&fixture, dtype, &backend, &context)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            &fixture.base_artifact_digest,
            &source,
        )?;
        let mut typed = options;
        typed.dtype = dtype;
        assert!(build_model_family_for_probe(&registry, &probe, weights, typed).is_ok());
    }
    let mut unsupported_dtype = options;
    unsupported_dtype.dtype = DType::F64;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights.clone(), unsupported_dtype),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
    let mut unsupported_device = options;
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights.clone(), unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));
    let mut oom = options;
    oom.memory_budget_bytes = model.memory_estimate().total_bytes - 1;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, oom),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));
    Ok(())
}

fn verify_source_projection() -> Result<(), Box<dyn std::error::Error>> {
    let provenance: serde_json::Value = serde_json::from_slice(&fs::read(provenance_path())?)?;
    assert_eq!(provenance["feature_id"], sa3::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], sa3::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], sa3::MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(provenance["source_architecture"], sa3::SOURCE_ARCHITECTURE);
    assert_eq!(provenance["latent_feature_id"], "COMFY-MODEL-0051");
    assert_eq!(provenance["latent_identifier"], "StableAudio3");
    assert_eq!(provenance["catalog_projection_sha256"], sa3::MODEL_FAMILY_PROJECTION_SHA256);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("missing source projection")?;
    assert!(projection.contains("global_cond_shared_embed=true"));
    assert!(projection.contains("conditioning=seconds_total[0,384]"));
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    let root = repository_root();
    for source in provenance["source_files"].as_array().ok_or("missing source files")? {
        let path = source["path"].as_str().ok_or("missing source path")?;
        let digest = source["sha256"].as_str().ok_or("missing source digest")?;
        assert_eq!(sha256(&fs::read(root.join(path))?), digest);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&fs::read(
        root.join("crates/sim_model/catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["feature_id"] == sa3::MODEL_FAMILY_FEATURE_ID))
        .ok_or("missing catalog row")?;
    assert_eq!(sha256(&serde_json::to_vec(row)?), sa3::MODEL_FAMILY_PROJECTION_SHA256);
    assert_eq!(row["static"]["unet_config"]["value"]["global_cond_shared_embed"], true);
    Ok(())
}

fn verify_failures(
    fixture: &FamilyFixture,
    registry: &ModelFamilyRegistry,
    probe: &ModelProbe,
) -> Result<(), Box<dyn std::error::Error>> {
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([("conv_in.weight".to_owned(), vec![2, 2, 1])]),
        metadata: BTreeMap::from([("model_layout".to_owned(), "diffusers".to_owned())]),
    };
    assert!(matches!(
        sa3::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut stable_audio_1 = probe.clone();
    stable_audio_1.tensor_shapes.insert(
        "conditioner.conditioners.seconds_start.embedder.embedding.0.weights".to_owned(),
        vec![1],
    );
    assert!(matches!(
        registry.resolve(&stable_audio_1),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("seconds_start")
    ));
    let mut partial = probe.clone();
    partial.tensor_shapes.remove(
        "model.model.transformer.global_cond_embedder.0.weight",
    );
    assert!(registry.resolve(&partial).is_err());
    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "model.model.transformer.global_cond_embedder.0.weight".to_owned(),
        vec![2, 3],
    );
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("shared global")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    let mut misleading = probe.clone();
    misleading
        .metadata
        .insert("audio_model".to_owned(), "not-dit1.0".to_owned());
    assert_eq!(
        registry
            .resolve(&misleading)?
            .detection()
            .identity
            .feature_id(),
        sa3::MODEL_FAMILY_FEATURE_ID
    );

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
    Ok(serde_json::from_slice(&fs::read(fixture_path())?)?)
}

fn fixture_probe(fixture: &FamilyFixture) -> ModelProbe {
    ModelProbe {
        tensor_shapes: fixture.detector.tensor_shapes.clone(),
        metadata: fixture.detector.metadata.clone(),
    }
}

fn source_tensors(
    fixture: &FamilyFixture,
    dtype: DType,
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe_tensor_fixtures(fixture)?
        .iter()
        .map(|weight| {
            Ok((
                weight.key.clone(),
                tensor(backend, &weight.shape, &weight.values, dtype, context)?,
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
            let count = usize::try_from(shape.iter().product::<u64>())?;
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

fn probe_through_model_store(
    fixture: &FamilyFixture,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("stableaudio3.safetensors");
    write_safetensors(&path, &probe_tensor_fixtures(fixture)?)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "stableaudio3-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(
        &index,
        &ArtifactKey::new("stableaudio3-row", "stableaudio3.safetensors")?,
        &cancellation,
    )?;
    let probe = store.family_probe(&loaded, &cancellation)?;
    assert_eq!(probe.tensor_shapes(), &fixture.detector.tensor_shapes);
    Ok(probe)
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
    Ok(if dtype == DType::F32 {
        tensor
    } else {
        cast_to_with_context_exact_native(
            backend,
            &tensor,
            dtype,
            backend.device(),
            false,
            false,
            context,
        )?
    })
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
        let values = tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
        for (index, (actual_value, expected_value)) in
            values.iter().zip(&expected.values).enumerate()
        {
            if (actual_value - expected_value).abs() > 1.0e-5 {
                return Err(format!(
                    "{}[{index}]: expected {expected_value}, got {actual_value}",
                    expected.name
                )
                .into());
            }
        }
    }
    Ok(())
}

fn verify_owner_delegation() -> Result<(), Box<dyn std::error::Error>> {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/families/stableaudio3_sim_model_0133.rs"
    ));
    for owner in [
        "ModelFamilyRegistration",
        "ModelFamilyStatePlanSelector",
        "ModelStateTransformPlanDefinition",
        "ModelProbe",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(source.contains(owner));
    }
    for forbidden in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "std::fs",
        "unsafe ",
        "std::process",
        "Command::",
        "python",
    ] {
        assert!(!source.contains(forbidden));
    }
    Ok(())
}

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../sim_test_support/fixtures/models/stableaudio3-comfy-model-0133/family.json",
    )
}

fn provenance_path() -> std::path::PathBuf {
    fixture_path().with_file_name("provenance.json")
}

fn repository_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
