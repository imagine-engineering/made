use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelFamilyRegistry,
    ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits, PatchApplication,
    PatchGraph, PatchKind, PatchOperation, PatchTarget, build_model_family_for_probe,
    describe_model_family, generated_krea2_sim_model_0101 as krea,
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

const ARTIFACT_DIGEST: &str = "0101010101010101010101010101010101010101010101010101010101010101";

#[test]
fn val_model_family_row_001_krea2_source_configuration_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(krea::MODEL_FAMILY_IDENTIFIER, "Krea2");
    assert_eq!(krea::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0101");
    assert_eq!(krea::MODEL_FAMILY_SOURCE_ORDINAL, 79);
    assert_eq!(krea::MODEL_FAMILY_REGISTRATION.source_ordinal, 79);
    assert_eq!(krea::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.Krea2");
    assert_eq!(krea::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.2);
    assert_eq!(krea::MODEL_FAMILY_SAMPLING_SHIFT, 1.15);

    let descriptor = describe_model_family(&krea::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Krea2");
    assert_eq!(descriptor.family, "COMFY-MODEL-0101");
    assert_eq!(descriptor.architecture_version, "krea2-single-stream-dit-v1");
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    let native = ModelProbe::from_parsed_facts(parsed_facts("native", "safetensors", DType::F32))?;
    let configuration = krea::configuration_for_probe(&native)?;
    assert_eq!(configuration.layout, krea::Krea2Layout::PrefixedNative);
    assert_eq!(configuration.feature_dimension, 128);
    assert_eq!(configuration.channels, 1);
    assert_eq!(configuration.patch_size, 2);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.key_value_heads, 1);
    assert_eq!(configuration.text_layer_count, 12);
    assert_eq!(configuration.text_feature_dimension, 2);
    assert!(configuration.supports_temporal_batches);

    let standalone = ModelProbe::from_parsed_facts(parsed_facts(
        "standalone",
        "safetensors",
        DType::Bf16,
    ))?;
    assert_eq!(
        krea::configuration_for_probe(&standalone)?.layout,
        krea::Krea2Layout::StandaloneNative
    );
    let diffusers = ModelProbe::from_parsed_facts(parsed_facts(
        "standalone",
        "diffusers",
        DType::F32,
    ))?;
    assert!(matches!(
        krea::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));

    let store_probe = probe_through_model_store()?;
    let resolved = ModelFamilyRegistry::checked_registrations(&[krea::MODEL_FAMILY_REGISTRATION])?
        .resolve(&store_probe)?;
    assert_eq!(resolved.detection().identity.feature_id(), krea::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(store_probe.unet_prefix_selection()?.prefix(), "model.diffusion_model.");
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(
        resolved.clip_target().candidates()[0].tokenizer().identifier(),
        "sim.text_encoders.krea2.Krea2Tokenizer"
    );

    validate_provenance_and_catalog()?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("family.json"))?)?;
    assert_eq!(fixture["fixture_id"], krea::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture["feature_id"], krea::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(fixture["dtype"], "f32");
    assert_eq!(fixture["device"], "cpu");
    assert_eq!(
        fixture["source_weights"].as_array().map(Vec::len),
        Some(10)
    );
    assert_eq!(fixture["checkpoints"].as_array().map(Vec::len), Some(6));
    assert_eq!(fixture["patches"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        fixture["patched_checkpoints"].as_array().map(Vec::len),
        Some(6)
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/krea2_sim_model_0101.rs"),
    )?;
    for canonical_owner in ["ModelProbe", "ModelStateTransformPlanDefinition", "ModelForwardOperation"] {
        assert!(row_source.contains(canonical_owner));
    }
    for competing_owner in [
        "struct Tensor", "struct ModelStore", "struct PatchGraph", "struct CancellationToken",
        "struct ArtifactIndex", "std::fs", "unsafe ",
    ] {
        assert!(!row_source.contains(competing_owner));
    }
    super::write_model_family_row_artifact(
        krea::MODEL_FAMILY_FIXTURE,
        krea::MODEL_FAMILY_FEATURE_ID,
        krea::MODEL_FAMILY_IDENTIFIER,
        krea::MODEL_FAMILY_SOURCE_ORDINAL,
        "krea2_comfy_model_0101",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-and-standalone-detection",
            "source-exact-krea-configuration-and-diffusers-rejection",
            "transactional-component-mapping-and-generated-conditioning",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-misleading-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_krea2_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[krea::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );

    for (layout, dtype) in [
        ("native", DType::F32),
        ("standalone", DType::F32),
        ("native", DType::Bf16),
        ("native", DType::F16),
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(layout, "safetensors", dtype))?;
        let resolved = registry.resolve(&probe)?;
        let source = source_tensors(&backend, &context, layout, dtype, false)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert_eq!(mapped.components().len(), 4);
        assert!(mapped.component("runtime_conditioning").is_some_and(|state| {
            state.contains_key("sampling_multiplier") && state.contains_key("sampling_shift")
        }));
        if dtype == DType::F32 {
            let weights = resolved.map_primary_weights(
                &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &source,
            )?;
            let model = build_model_family_for_probe(
                &registry, &probe, weights, options(dtype, DeviceKind::Cpu, u64::MAX),
            )?;
            let input = tensor(&backend, &[1, 2], &[0.0, 0.0], dtype, &context)?;
            let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(&backend, &context, &checkpoints, "timestep_projection", &[1.0, -1.0])?;
            assert_checkpoint(&backend, &context, &checkpoints, "image_output", &[0.800_498_7, -0.800_498_7])?;

            let patch = PatchGraph::checked(
                ARTIFACT_DIGEST,
                vec![PatchOperation {
                    identifier: "krea2-output-bias".to_owned(),
                    kind: PatchKind::Lora,
                    scale: 1.0,
                    targets: vec![PatchTarget {
                        key: "native.last.linear.bias".to_owned(),
                        expected_shape: vec![2],
                        values: vec![0.5, 0.5],
                        application: PatchApplication::Add,
                    }],
                }],
            )?;
            let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
            let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(
                &backend, &context, &patched_checkpoints, "image_output", &[0.921_668_4, -0.537_048_94],
            )?;

            let required = model.memory_estimate().total_bytes;
            let fresh_source = source_tensors(&backend, &context, layout, dtype, false)?;
            let fresh_weights = resolved.map_primary_weights(
                &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &fresh_source,
            )?;
            assert!(matches!(
                build_model_family_for_probe(
                    &registry, &probe, fresh_weights,
                    options(dtype, DeviceKind::Cpu, required.saturating_sub(1)),
                ),
                Err(ModelFamilyError::OutOfMemory { required: actual, budget })
                    if actual == required && budget == required - 1
            ));
        }
    }

    let valid = ModelProbe::from_parsed_facts(parsed_facts("native", "safetensors", DType::F32))?;
    let resolved = registry.resolve(&valid)?;
    let source = source_tensors(&backend, &context, "native", DType::F32, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &source,
    )?;
    assert!(build_model_family_for_probe(
        &registry, &valid, weights, options(DType::F32, DeviceKind::Metal, u64::MAX),
    ).is_err());

    let mut partial = parsed_facts("native", "safetensors", DType::F32);
    partial.tensors.remove("model.diffusion_model.txtfusion.projector.weight");
    assert!(registry.resolve(&ModelProbe::from_parsed_facts(partial)?).is_err());

    let mut ambiguous = parsed_facts("native", "safetensors", DType::F32);
    for (key, shape) in model_shapes() {
        ambiguous.tensors.insert(
            key,
            ModelParsedTensorFact {
                shape,
                storage_dtype: DType::F32.catalog_name().to_owned(),
            },
        );
    }
    assert!(krea::configuration_for_probe(&ModelProbe::from_parsed_facts(ambiguous)?).is_err());

    let mut misleading = parsed_facts("native", "safetensors", DType::F32);
    misleading.tensors.get_mut("model.diffusion_model.first.weight")
        .ok_or("first weight must exist")?.shape = vec![128, 3];
    assert!(matches!(
        registry.resolve(&ModelProbe::from_parsed_facts(misleading)?),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));

    let missing = source_tensors(&backend, &context, "native", DType::F32, true)?;
    assert!(resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &missing,
    ).is_err());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::new(10), authority.authorize_workspace(8 * 1024 * 1024)?, &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &cancelled_context), ARTIFACT_DIGEST, &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

fn parsed_facts(layout: &str, format: &str, dtype: DType) -> ModelParsedFacts {
    let prefix = if layout == "native" { "model.diffusion_model." } else { "" };
    let mut tensors = model_shapes().into_iter().map(|(key, shape)| {
        (format!("{prefix}{key}"), ModelParsedTensorFact {
            shape, storage_dtype: dtype.catalog_name().to_owned(),
        })
    }).collect::<BTreeMap<_, _>>();
    for key in ["vae.decoder.weight", "text_encoders.qwen3vl_4b.transformer.weight"] {
        tensors.insert(key.to_owned(), ModelParsedTensorFact {
            shape: vec![1], storage_dtype: dtype.catalog_name().to_owned(),
        });
    }
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::from([
                ("image_model".to_owned(), "krea2".to_owned()),
                ("model_layout".to_owned(), format.to_owned()),
            ]),
        }],
    }
}

fn model_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("first.weight".to_owned(), vec![128, 4]),
        ("txtfusion.projector.weight".to_owned(), vec![1, 12]),
        ("txtfusion.layerwise_blocks.0.prenorm.scale".to_owned(), vec![2]),
        ("blocks.0.attn.wq.weight".to_owned(), vec![128, 128]),
        ("blocks.0.attn.wk.weight".to_owned(), vec![128, 128]),
        ("tmlp.0.weight".to_owned(), vec![2, 2]),
        ("tmlp.0.bias".to_owned(), vec![2]),
        ("blocks.0.mlp.down.weight".to_owned(), vec![2, 2]),
        ("last.linear.weight".to_owned(), vec![2, 2]),
        ("last.linear.bias".to_owned(), vec![2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend, context: &ExecutionContext<'_>, layout: &str, dtype: DType,
    omit_output: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" { "model.diffusion_model." } else { "" };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes() {
        if omit_output && key == "last.linear.weight" { continue; }
        let values = match key.as_str() {
            "tmlp.0.weight" | "blocks.0.mlp.down.weight" | "last.linear.weight" => {
                vec![1.0, 0.0, 0.0, 1.0]
            }
            "tmlp.0.bias" => vec![1.0, -1.0],
            "last.linear.bias" => vec![0.1, -0.1],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
        };
        source.insert(format!("{prefix}{key}"), tensor(backend, &shape, &values, dtype, context)?);
    }
    source.insert("vae.decoder.weight".to_owned(), tensor(backend, &[1], &[1.0], dtype, context)?);
    source.insert(
        "text_encoders.qwen3vl_4b.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("krea2.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "krea2-row", "checkpoints", directory.path(), ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("krea2-row", "krea2.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::json!({"image_model":"krea2"}));
    let mut data = Vec::new();
    for (key, shape) in model_shapes() {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        for _ in 0..elements { data.extend_from_slice(&0.0_f32.to_le_bytes()); }
        header.insert(format!("model.diffusion_model.{key}"),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}));
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn validate_provenance_and_catalog() -> Result<(), Box<dyn std::error::Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(sha256(&std::fs::read(repository.join(krea::MODEL_FAMILY_SOURCE_PATH))?), krea::MODEL_FAMILY_SOURCE_SHA256);
    let provenance: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture_directory().join("provenance.json"))?,
    )?;
    let projection = provenance["source_projection"].as_str().ok_or("projection must be text")?;
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    for source in provenance["source_files"].as_array().ok_or("source_files must be an array")? {
        let path = source["path"].as_str().ok_or("source path must be text")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), source["sha256"]);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"].as_array().ok_or("models must be an array")?.iter()
        .find(|row| row["feature_id"] == krea::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Krea2 catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 79);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "krea2");
    assert_eq!(sha256(&serde_json::to_vec(row)?), krea::MODEL_FAMILY_PROJECTION_SHA256);
    Ok(())
}

fn tensor(
    backend: &CpuBackend, shape: &[u64], values: &[f32], dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend, shape, values, dtype, backend.device(), context,
    )?)
}

fn options(dtype: DType, device: DeviceKind, memory_budget_bytes: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype, device, activation_elements: 2, memory_budget_bytes, allow_unexpected_weights: false,
    }
}

fn assert_checkpoint(
    backend: &CpuBackend, context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint], name: &str, expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoints.iter().find(|checkpoint| checkpoint.name == name)
        .ok_or("Krea2 checkpoint is missing")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{name}: {actual} != {expected}");
    }
    Ok(())
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../sim_test_support/fixtures/models")
        .join(krea::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
