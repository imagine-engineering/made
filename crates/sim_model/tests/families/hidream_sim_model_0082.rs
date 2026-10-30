use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_hidream_sim_model_0082 as hidream,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Write, path::Path};

const ARTIFACT_DIGEST: &str = "0082008200820082008200820082008200820082008200820082008200820082";
const EXPECTED_MEMORY_BYTES: u64 = 164;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9082",
    identifier: "HiDream_AmbiguousFixture",
    ..hidream::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    hidream::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 169,
        source_architecture: "model_base.HiDreamAmbiguousFixture",
        ..hidream::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_hidream_source_configuration_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(hidream::MODEL_FAMILY_IDENTIFIER, "HiDream");
    assert_eq!(hidream::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0082");
    assert_eq!(hidream::MODEL_FAMILY_SOURCE_ORDINAL, 69);
    assert_eq!(
        hidream::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.HiDream"
    );
    assert_eq!(hidream::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);

    let descriptor = describe_model_family(&hidream::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 2);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 2);

    let registry =
        ModelFamilyRegistry::checked_registrations(&[hidream::MODEL_FAMILY_REGISTRATION])?;
    let native_probe = ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32))?;
    let native = hidream::configuration_for_probe(&native_probe)?;
    assert_eq!(native.layout, hidream::HiDreamLayout::Native);
    assert_source_configuration(native);
    assert_eq!(
        registry
            .resolve(&native_probe)?
            .profile()
            .latent_identifier,
        "Flux"
    );

    let diffusers_probe = ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::Bf16))?;
    let diffusers = hidream::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(diffusers.layout, hidream::HiDreamLayout::Diffusers);
    assert_source_configuration(diffusers);

    let mut misleading = parsed_facts("native", DType::F32);
    misleading.formats[0]
        .metadata
        .insert("image_model".to_owned(), "hidream_o1".to_owned());
    let misleading = ModelProbe::from_parsed_facts(misleading)?;
    assert_eq!(
        registry.detect(&misleading)?.identity.feature_id(),
        hidream::MODEL_FAMILY_FEATURE_ID
    );

    let mut partial = parsed_facts("native", DType::F32);
    partial
        .tensors
        .remove("model.diffusion_model.final_layer.linear.weight");
    let partial = ModelProbe::from_parsed_facts(partial)?;
    assert!(matches!(
        registry.detect(&partial),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut malformed = parsed_facts("native", DType::F32);
    malformed
        .tensors
        .get_mut("model.diffusion_model.x_embedder.proj.weight")
        .ok_or("HiDream patch projection must exist")?
        .shape = vec![3, 2];
    let malformed = ModelProbe::from_parsed_facts(malformed)?;
    assert!(matches!(
        hidream::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("patch projection shape")
    ));

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("HiDream source projection must be a string")?;
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    assert_eq!(
        hidream::MODEL_FAMILY_PROJECTION_SHA256,
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("HiDream source_files must be an array")?
    {
        let path = source["path"].as_str().ok_or("source path must be a string")?;
        let digest = source["sha256"]
            .as_str()
            .ok_or("source digest must be a string")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), digest);
    }

    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog must contain models")?
        .iter()
        .find(|row| row["feature_id"] == hidream::MODEL_FAMILY_FEATURE_ID)
        .ok_or("HiDream catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 69);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "hidream");
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 2.0);
    assert_eq!(
        row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.Flux"
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/hidream_sim_model_0082.rs"),
    )?;
    for owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct ModelProbe",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "struct MemoryEstimator",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(owner));
    }

    super::write_model_family_row_artifact(
        hidream::MODEL_FAMILY_FIXTURE,
        hidream::MODEL_FAMILY_FEATURE_ID,
        hidream::MODEL_FAMILY_IDENTIFIER,
        hidream::MODEL_FAMILY_SOURCE_ORDINAL,
        "hidream_comfy_model_0082",
        &[
            "source-provenance-and-ownership",
            "native-and-diffusers-key-derived-configuration",
            "canonical-flux-latent-and-no-clip-target",
            "misleading-metadata-and-cross-family-rejection",
            "transactional-mapping-forward-patching-memory",
            "dtype-device-cancellation-and-typed-failures",
            "double-single-stream-and-conditioning-checkpoints",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_hidream_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[hidream::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        hidream::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(resolved.source_ordinal(), 69);
    assert!(resolved.clip_target().candidates().is_empty());

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&backend, &context, "native", DType::F32)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("model").is_some());
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model = build_model_family_for_probe(
        &registry,
        &probe,
        weights,
        options(DType::F32, EXPECTED_MEMORY_BYTES),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, EXPECTED_MEMORY_BYTES);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "caption_projection",
        &[0.9810586, -0.5189414],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "image_output",
        &[0.8004987, -0.8004987],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "hidream-image-output-bias".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.final_layer.linear.bias".to_owned(),
                expected_shape: vec![2],
                values: vec![0.5, 0.5],
                application: PatchApplication::Add,
            }],
        }],
    )?;
    let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
    let checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "image_output",
        &[0.9216684, -0.53704894],
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(DType::F32, EXPECTED_MEMORY_BYTES - 1),
        ),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == EXPECTED_MEMORY_BYTES && budget == EXPECTED_MEMORY_BYTES - 1
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_hidream_diffusers_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[hidream::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::Bf16))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, "diffusers", DType::Bf16)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    build_model_family_for_probe(
        &registry,
        &probe,
        weights,
        options(DType::Bf16, EXPECTED_MEMORY_BYTES),
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(DType::F16, EXPECTED_MEMORY_BYTES),
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported = options(DType::Bf16, EXPECTED_MEMORY_BYTES);
    unsupported.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut partial_facts = parsed_facts("diffusers", DType::Bf16);
    partial_facts
        .tensors
        .remove("t_embedder.timestep_embedder.linear_2.weight");
    let partial_probe = ModelProbe::from_parsed_facts(partial_facts)?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let mut partial = source_tensors(&backend, &context, "diffusers", DType::Bf16)?;
    partial.remove("t_embedder.timestep_embedder.linear_2.weight");
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model"
                && key == "native.t_embedder.timestep_embedder.linear_2.weight"
    ));

    let mut unexpected_facts = parsed_facts("diffusers", DType::Bf16);
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: DType::Bf16.catalog_name().to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected = source_tensors(&backend, &context, "diffusers", DType::Bf16)?;
    unexpected.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::Bf16, &context)?,
    );
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &unexpected,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

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

fn assert_source_configuration(configuration: hidream::HiDreamConfiguration) {
    assert_eq!(configuration.patch_size, 2);
    assert_eq!(configuration.in_channels, 16);
    assert_eq!(configuration.out_channels, 16);
    assert_eq!(configuration.number_of_layers, 16);
    assert_eq!(configuration.number_of_single_layers, 32);
    assert_eq!(configuration.attention_head_dimension, 128);
    assert_eq!(configuration.number_of_attention_heads, 20);
    assert_eq!(configuration.inner_dimension, 2_560);
    assert_eq!(configuration.caption_channels, [4_096, 4_096]);
    assert_eq!(configuration.text_embedding_dimension, 2_048);
    assert_eq!(configuration.number_of_routed_experts, 4);
    assert_eq!(configuration.number_of_activated_experts, 2);
    assert_eq!(configuration.rope_axes_dimensions, [64, 32, 32]);
    assert_eq!(configuration.maximum_resolution, [128, 128]);
    assert_eq!(configuration.llama_layer_count, 48);
}

fn parsed_facts(layout: &str, dtype: DType) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = model_shapes()
        .into_iter()
        .map(|(key, shape)| {
            (
                format!("{prefix}{key}"),
                ModelParsedTensorFact {
                    shape,
                    storage_dtype: dtype.catalog_name().to_owned(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for index in 1..16 {
        tensors.insert(
            format!("{prefix}double_stream_blocks.{index}.block.attn1.to_q.weight"),
            ModelParsedTensorFact {
                shape: vec![1],
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for index in 1..32 {
        tensors.insert(
            format!("{prefix}single_stream_blocks.{index}.block.attn1.to_q.weight"),
            ModelParsedTensorFact {
                shape: vec![1],
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for key in ["vae.decoder.weight", "text_encoders.llama3.transformer.weight"] {
        tensors.insert(
            key.to_owned(),
            ModelParsedTensorFact {
                shape: vec![1],
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::new(),
        }],
    }
}

fn model_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("x_embedder.proj.weight".to_owned(), vec![2, 2]),
        (
            "t_embedder.timestep_embedder.linear_1.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "t_embedder.timestep_embedder.linear_1.bias".to_owned(),
            vec![2],
        ),
        (
            "t_embedder.timestep_embedder.linear_2.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "t_embedder.timestep_embedder.linear_2.bias".to_owned(),
            vec![2],
        ),
        ("caption_projection.0.linear.weight".to_owned(), vec![2, 2]),
        (
            "double_stream_blocks.0.block.attn1.to_out.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_stream_blocks.0.block.attn1.to_out.weight".to_owned(),
            vec![2, 2],
        ),
        ("final_layer.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.bias".to_owned(), vec![2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes() {
        let values = match key.as_str() {
            "x_embedder.proj.weight"
            | "t_embedder.timestep_embedder.linear_1.weight"
            | "t_embedder.timestep_embedder.linear_2.weight"
            | "caption_projection.0.linear.weight"
            | "double_stream_blocks.0.block.attn1.to_out.weight"
            | "single_stream_blocks.0.block.attn1.to_out.weight"
            | "final_layer.linear.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "t_embedder.timestep_embedder.linear_1.bias" => vec![1.0, -1.0],
            "t_embedder.timestep_embedder.linear_2.bias" => vec![0.25, -0.25],
            "final_layer.linear.bias" => vec![0.1, -0.1],
            _ => return Err(format!("missing source values for {key}").into()),
        };
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, &shape, &values, dtype, context)?,
        );
    }
    for index in 1..16 {
        source.insert(
            format!("{prefix}double_stream_blocks.{index}.block.attn1.to_q.weight"),
            tensor(backend, &[1], &[0.0], dtype, context)?,
        );
    }
    for index in 1..32 {
        source.insert(
            format!("{prefix}single_stream_blocks.{index}.block.attn1.to_q.weight"),
            tensor(backend, &[1], &[0.0], dtype, context)?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    source.insert(
        "text_encoders.llama3.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("hidream.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "hidream-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("hidream-row", "hidream.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let facts = parsed_facts("native", DType::F32);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, tensor) in facts.tensors {
        let start = data.len();
        let elements = tensor.shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        for _ in 0..elements {
            data.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        header.insert(
            key,
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, data.len()]
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

fn options(dtype: DType, memory_budget_bytes: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes,
        allow_unexpected_weights: true,
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
        .ok_or("HiDream checkpoint is missing")?;
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
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(hidream::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
