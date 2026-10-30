use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_acestep_sim_model_0061::{
        MODEL_FAMILY, MODEL_FAMILY_FEATURE_ID, MODEL_FAMILY_FIXTURE, MODEL_FAMILY_IDENTIFIER,
        MODEL_FAMILY_MEMORY_USAGE_FACTOR, MODEL_FAMILY_REGISTRATION, MODEL_FAMILY_SAMPLING_SHIFT,
        MODEL_FAMILY_SOURCE_ORDINAL,
    },
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Write};

const ARTIFACT_DIGEST: &str = "a61ea61ea61ea61ea61ea61ea61ea61ea61ea61ea61ea61ea61ea61ea61ea61e";

static AMBIGUOUS_MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9061",
    identifier: "ACEStepAmbiguousFixture",
    ..MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_MODEL_FAMILY,
        source_ordinal: 74,
        source_architecture: "ACEStepAmbiguousFixture",
        ..MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_acestep_source_projection_registration_and_descriptor()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(MODEL_FAMILY_IDENTIFIER, "ACEStep");
    assert_eq!(MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0061");
    assert_eq!(MODEL_FAMILY_FIXTURE, "acestep-comfy-model-0061");
    assert_eq!(MODEL_FAMILY_SOURCE_ORDINAL, 73);
    assert_eq!(MODEL_FAMILY_REGISTRATION.source_ordinal, 73);
    assert_eq!(MODEL_FAMILY_REGISTRATION.source_architecture, "ACEStep");
    assert_eq!(MODEL_FAMILY_SAMPLING_SHIFT, 3.0);
    assert_eq!(MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.5);

    let descriptor = describe_model_family(&MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "ACEStep");
    assert_eq!(descriptor.family, "COMFY-MODEL-0061");
    assert_eq!(
        descriptor.architecture_version,
        "ace-step-transformer-2d-v1"
    );
    assert_eq!(descriptor.latent_format, "ACEAudio");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    let row_source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/acestep_sim_model_0061.rs"),
    )?;
    for canonical_owner in [
        "ModelFamilyRegistration",
        "ModelStateTransformPlanDefinition",
        "ModelFamilyComponentStateSchema",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(row_source.contains(canonical_owner));
    }
    for competing_owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(competing_owner));
    }

    let provenance_path = fixture_directory().join("provenance.json");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(provenance_path)?)?;
    assert_eq!(provenance["feature_id"], MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 73);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for source in provenance["source_files"]
        .as_array()
        .ok_or("source_files must be an array")?
    {
        let path = source["path"]
            .as_str()
            .ok_or("source path must be a string")?;
        let expected = source["sha256"]
            .as_str()
            .ok_or("source digest must be a string")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), expected);
    }
    super::write_model_family_row_artifact(
        MODEL_FAMILY_FIXTURE,
        MODEL_FAMILY_FEATURE_ID,
        MODEL_FAMILY_IDENTIFIER,
        MODEL_FAMILY_SOURCE_ORDINAL,
        "acestep_comfy_model_0061",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-and-diffusers-detection",
            "transactional-component-mapping",
            "named-forward-checkpoints-and-patching",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_acestep_parsed_detection_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store("native")?;
    assert_eq!(
        probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(resolved.source_ordinal(), 73);
    assert_eq!(resolved.source_architecture(), "ACEStep");
    assert_eq!(resolved.profile().latent_identifier, "ACEAudio");
    let clip = resolved.clip_target();
    assert!(!clip.dynamic_selection());
    assert_eq!(clip.candidates().len(), 1);
    assert_eq!(
        clip.candidates()[0].tokenizer().identifier(),
        "sim.text_encoders.ace.AceT5Tokenizer"
    );
    assert_eq!(
        clip.candidates()[0].clip_model().target().as_str(),
        "sim.text_encoders.ace.AceT5Model"
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&backend, &context, "native", DType::F32, false)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 4);
    assert!(mapped.component("denoiser").is_some());
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            mapped
                .component("conditioning")
                .and_then(|component| component.get("speaker_embeds"))
                .ok_or("generated speaker embedding is missing")?,
            &context,
        )?,
        [0.0, 0.0]
    );

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let build_options = options(DType::F32, 44, true);
    let model = build_model_family_for_probe(&registry, &probe, weights, build_options)?;
    assert_eq!(model.source_ordinal(), Some(73));
    assert_eq!(model.memory_estimate().total_bytes, 44);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "genre_projection",
        &[1.0, -1.0],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "audio_output",
        &[0.62371254, -0.26263955],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "row-genre-bias".to_owned(),
            kind: PatchKind::Adapter,
            scale: 0.5,
            targets: vec![PatchTarget {
                key: "genre_embedder.bias".to_owned(),
                expected_shape: vec![2],
                values: vec![1.0, 1.0],
                application: PatchApplication::Add,
            }],
        }],
    )?;
    let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
    let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &patched_checkpoints,
        "audio_output",
        &[0.8415208, -0.18655962],
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 43, true)),
        Err(ModelFamilyError::OutOfMemory {
            required: 44,
            budget: 43,
        })
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_acestep_diffusers_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );

    let probe = probe_through_model_store("diffusers")?;
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "model.");
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, "diffusers", DType::Bf16, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model =
        build_model_family_for_probe(&registry, &probe, weights, options(DType::Bf16, 44, true))?;
    assert_eq!(model.profile().supported_dtypes, &[DType::Bf16, DType::F32]);

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported_device = options(DType::Bf16, 44, true);
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let missing_probe = ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, true))?;
    assert!(matches!(
        registry.resolve(&missing_probe),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut unexpected_facts = parsed_facts("native", DType::F32, false);
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: "F32".to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected_source = source_tensors(&backend, &context, "native", DType::F32, false)?;
    unexpected_source.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &unexpected_source,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let bad_probe = ModelProbe {
        tensor_shapes: BTreeMap::new(),
        metadata: BTreeMap::from([("audio_model".to_owned(), "not-ace".to_owned())]),
    };
    assert!(matches!(
        registry.detect(&bad_probe),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut misleading_facts = parsed_facts("native", DType::F32, false);
    misleading_facts.formats[0]
        .metadata
        .insert("audio_model".to_owned(), "not-ace".to_owned());
    let misleading_probe = ModelProbe::from_parsed_facts(misleading_facts)?;
    assert_eq!(
        registry
            .resolve(&misleading_probe)?
            .detection()
            .identity
            .feature_id(),
        MODEL_FAMILY_FEATURE_ID
    );

    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &cancelled_context),
            ARTIFACT_DIGEST,
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

fn parsed_facts(layout: &str, dtype: DType, omit_genre_weight: bool) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in model_shapes(omit_genre_weight) {
        tensors.insert(
            format!("{prefix}{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    tensors.insert(
        "vae.decoder.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: dtype.catalog_name().to_owned(),
        },
    );
    tensors.insert(
        "text_encoders.embedding.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: dtype.catalog_name().to_owned(),
        },
    );
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::from([("audio_model".to_owned(), "ace".to_owned())]),
        }],
    }
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_genre_weight: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(omit_genre_weight) {
        let values = match key.as_str() {
            "genre_embedder.weight" | "final_layer.linear.weight" => {
                vec![1.0, 0.0, 0.0, 1.0]
            }
            "genre_embedder.bias" => vec![1.0, -1.0],
            "final_layer.linear.bias" => vec![0.0, 0.0],
            _ => vec![0.0; shape.iter().product::<u64>() as usize],
        };
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, &shape, &values, dtype, context)?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    source.insert(
        "text_encoders.embedding.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store(layout: &str) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("acestep.safetensors");
    write_safetensors(&model_path, layout)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "acestep-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("acestep-row", "acestep.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(
    path: &std::path::Path,
    layout: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut shapes = model_shapes(false);
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.embedding.weight".to_owned(), vec![1]),
    ]);
    for (key, shape) in shapes {
        let name = if key.starts_with("vae.") || key.starts_with("text_encoders.") {
            key
        } else {
            format!("{prefix}{key}")
        };
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or("fixture shape overflow")
        })?;
        for _ in 0..elements {
            data.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn model_shapes(omit_genre_weight: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("genre_embedder.bias".to_owned(), vec![2]),
        ("final_layer.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.bias".to_owned(), vec![2]),
        (
            "transformer_blocks.0.attn.to_q.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "transformer_blocks.0.attn.to_k.weight".to_owned(),
            vec![2, 2],
        ),
    ];
    if !omit_genre_weight {
        shapes.push(("genre_embedder.weight".to_owned(), vec![2, 2]));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        dtype,
        backend.device(),
        context,
    )?)
}

fn options(
    dtype: DType,
    memory_budget_bytes: u64,
    allow_unexpected_weights: bool,
) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes,
        allow_unexpected_weights,
    }
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoints
        .iter()
        .find(|checkpoint| checkpoint.name == name)
        .ok_or("checkpoint is missing")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{name}: {actual} != {expected}"
        );
    }
    Ok(())
}

fn fixture_directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
