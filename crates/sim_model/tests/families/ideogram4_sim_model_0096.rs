use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_ideogram4_sim_model_0096 as ideogram,
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

const ARTIFACT_DIGEST: &str = "0096009600960096009600960096009600960096009600960096009600960096";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9096",
    identifier: "Ideogram4AmbiguousFixture",
    ..ideogram::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    ideogram::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 178,
        source_architecture: "model_base.Ideogram4AmbiguousFixture",
        ..ideogram::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_ideogram4_source_descriptor_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(ideogram::MODEL_FAMILY_IDENTIFIER, "Ideogram4");
    assert_eq!(ideogram::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0096");
    assert_eq!(ideogram::MODEL_FAMILY_SOURCE_ORDINAL, 78);
    assert_eq!(ideogram::MODEL_FAMILY_REGISTRATION.source_ordinal, 78);
    assert_eq!(
        ideogram::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Ideogram4"
    );
    assert_eq!(ideogram::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 11.6);
    assert_eq!(ideogram::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(ideogram::MODEL_FAMILY_SAMPLING_SHIFT, 1.0);

    let descriptor = describe_model_family(&ideogram::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Ideogram4");
    assert_eq!(descriptor.family, "COMFY-MODEL-0096");
    assert_eq!(descriptor.architecture_version, "ideogram4-nextdit-v1");
    assert_eq!(descriptor.latent_format, "Flux2");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    let native = ModelProbe::from_parsed_facts(parsed_facts("native", "safetensors", DType::F32))?;
    let native_configuration = ideogram::configuration_for_probe(&native)?;
    assert_eq!(
        native_configuration.layout,
        ideogram::Ideogram4Layout::PrefixedNative
    );
    assert_eq!(native_configuration.hidden_size, 2);
    assert_eq!(native_configuration.in_channels, 4);
    assert_eq!(native_configuration.layer_count, 1);
    assert_eq!(native_configuration.attention_heads, 2);
    assert_eq!(native_configuration.attention_head_dimension, 1);
    assert_eq!(native_configuration.intermediate_size, 2);
    assert_eq!(native_configuration.adaln_dimension, 2);
    assert_eq!(native_configuration.llm_feature_dimension, 2);
    assert_eq!(native_configuration.patch_size, 2);
    assert_eq!(native_configuration.autoencoder_channels, 1);
    assert_eq!(native_configuration.rope_theta, 5_000_000);
    assert_eq!(native_configuration.mrope_sections, [24, 20, 20]);

    let standalone =
        ModelProbe::from_parsed_facts(parsed_facts("standalone", "safetensors", DType::Bf16))?;
    assert_eq!(
        ideogram::configuration_for_probe(&standalone)?.layout,
        ideogram::Ideogram4Layout::StandaloneNative
    );
    let diffusers =
        ModelProbe::from_parsed_facts(parsed_facts("standalone", "diffusers", DType::F32))?;
    assert!(matches!(
        ideogram::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));

    let store_probe = probe_through_model_store()?;
    assert_eq!(store_probe.format_identities(), ["safetensors"]);
    assert_eq!(
        store_probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    assert_eq!(
        ModelFamilyRegistry::checked_registrations(&[ideogram::MODEL_FAMILY_REGISTRATION])?
            .resolve(&store_probe)?
            .detection()
            .identity
            .feature_id(),
        ideogram::MODEL_FAMILY_FEATURE_ID
    );

    validate_provenance_and_catalog()?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("family.json"))?)?;
    assert_eq!(fixture["fixture_id"], ideogram::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture["feature_id"], ideogram::MODEL_FAMILY_FEATURE_ID);
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
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/ideogram4_sim_model_0096.rs"),
    )?;
    for canonical_owner in [
        "ModelProbe",
        "ModelStateTransformPlanDefinition",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(row_source.contains(canonical_owner));
    }
    for competing_owner in [
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct CancellationToken",
        "struct ArtifactIndex",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(competing_owner));
    }
    super::write_model_family_row_artifact(
        ideogram::MODEL_FAMILY_FIXTURE,
        ideogram::MODEL_FAMILY_FEATURE_ID,
        ideogram::MODEL_FAMILY_IDENTIFIER,
        ideogram::MODEL_FAMILY_SOURCE_ORDINAL,
        "ideogram4_comfy_model_0096",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-and-standalone-detection",
            "source-exact-configuration-and-diffusers-rejection",
            "transactional-component-mapping-and-generated-conditioning",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-misleading-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_ideogram4_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[ideogram::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );

    for (layout, dtype) in [
        ("native", DType::F32),
        ("standalone", DType::F32),
        ("native", DType::Bf16),
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(layout, "safetensors", dtype))?;
        let resolved = registry.resolve(&probe)?;
        let source = source_tensors(&backend, &context, layout, dtype, false)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert!(mapped.component("model").is_some());
        assert!(mapped.component("vae").is_some());
        assert!(mapped.component("text_encoder").is_some());
        assert!(mapped.component("runtime_conditioning").is_some_and(|state| {
            state.contains_key("sampling_multiplier") && state.contains_key("sampling_shift")
        }));
        if dtype == DType::F32 {
            let weights = resolved.map_primary_weights(
                &ModelStateTransaction::new(&backend, &context),
                ARTIFACT_DIGEST,
                &source,
            )?;
            let model = build_model_family_for_probe(
                &registry,
                &probe,
                weights,
                options(dtype, u64::MAX),
            )?;
            let input = tensor(&backend, &[1, 2], &[0.0, 0.0], dtype, &context)?;
            let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(&backend, &context, &checkpoints, "conditioning_projection", &[0.0, 0.0])?;
            assert_checkpoint(&backend, &context, &checkpoints, "image_output", &[0.099_667_996, -0.099_667_996])?;

            let patch = PatchGraph::checked(
                ARTIFACT_DIGEST,
                vec![PatchOperation {
                    identifier: "ideogram4-output-bias".to_owned(),
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
            let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(&backend, &context, &patched_checkpoints, "image_output", &[0.537_049_6, 0.379_949])?;

            let required = model.memory_estimate().total_bytes;
            let fresh_source = source_tensors(&backend, &context, layout, dtype, false)?;
            let fresh_weights = resolved.map_primary_weights(
                &ModelStateTransaction::new(&backend, &context),
                ARTIFACT_DIGEST,
                &fresh_source,
            )?;
            assert!(matches!(
                build_model_family_for_probe(
                    &registry,
                    &probe,
                    fresh_weights,
                    options(dtype, required.saturating_sub(1)),
                ),
                Err(ModelFamilyError::OutOfMemory { required: actual, budget })
                    if actual == required && budget == required - 1
            ));
        }
    }

    let f16_probe = ModelProbe::from_parsed_facts(parsed_facts("native", "safetensors", DType::F16))?;
    let f16_resolved = registry.resolve(&f16_probe)?;
    let f16_source = source_tensors(&backend, &context, "native", DType::F16, false)?;
    let f16_weights = f16_resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &f16_source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &f16_probe, f16_weights, options(DType::F16, u64::MAX)),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));

    let mut partial = parsed_facts("native", "safetensors", DType::F32);
    partial.tensors.remove("model.diffusion_model.embed_image_indicator.weight");
    let partial = ModelProbe::from_parsed_facts(partial)?;
    assert!(registry.resolve(&partial).is_err());

    let ambiguous_registry = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    let valid = ModelProbe::from_parsed_facts(parsed_facts("native", "safetensors", DType::F32))?;
    assert!(matches!(
        ambiguous_registry.resolve(&valid),
        Err(ModelFamilyError::AmbiguousDetection { .. })
    ));

    let mut misleading = parsed_facts("native", "safetensors", DType::F32);
    misleading
        .tensors
        .get_mut("model.diffusion_model.input_proj.weight")
        .ok_or("input projection must exist")?
        .shape = vec![2, 3];
    let misleading = ModelProbe::from_parsed_facts(misleading)?;
    assert!(matches!(
        registry.resolve(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));

    let source = source_tensors(&backend, &context, "native", DType::F32, true)?;
    let resolved = registry.resolve(&valid)?;
    assert!(resolved
        .map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )
        .is_err());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::new(9),
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancelled,
    );
    let source = source_tensors(&backend, &context, "native", DType::F32, false)?;
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

fn parsed_facts(layout: &str, format: &str, dtype: DType) -> ModelParsedFacts {
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
    for key in ["vae.decoder.weight", "text_encoders.qwen3vl_8b.transformer.weight"] {
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
            metadata: BTreeMap::from([
                ("image_model".to_owned(), "ideogram4".to_owned()),
                ("model_layout".to_owned(), format.to_owned()),
            ]),
        }],
    }
}

fn model_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("input_proj.weight".to_owned(), vec![2, 4]),
        ("input_proj.bias".to_owned(), vec![2]),
        ("llm_cond_proj.weight".to_owned(), vec![2, 2]),
        ("layers.0.attention.qkv.weight".to_owned(), vec![6, 2]),
        ("layers.0.attention.norm_q.weight".to_owned(), vec![1]),
        ("layers.0.feed_forward.w2.weight".to_owned(), vec![2, 2]),
        ("adaln_proj.weight".to_owned(), vec![2, 2]),
        ("embed_image_indicator.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.bias".to_owned(), vec![2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_output: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" { "model.diffusion_model." } else { "" };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes() {
        if omit_output && key == "final_layer.linear.weight" {
            continue;
        }
        let values = match key.as_str() {
            "llm_cond_proj.weight" | "layers.0.feed_forward.w2.weight" | "final_layer.linear.weight" => {
                vec![1.0, 0.0, 0.0, 1.0]
            }
            "final_layer.linear.bias" => vec![0.1, -0.1],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
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
        "text_encoders.qwen3vl_8b.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("ideogram4.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "ideogram4-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("ideogram4-row", "ideogram4.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::json!({"image_model":"ideogram4"}));
    let mut data = Vec::new();
    for (key, shape) in model_shapes() {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        for _ in 0..elements {
            data.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        header.insert(
            format!("model.diffusion_model.{key}"),
            serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
        );
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
    assert_eq!(
        sha256(&std::fs::read(repository.join(ideogram::MODEL_FAMILY_SOURCE_PATH))?),
        ideogram::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], ideogram::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], ideogram::MODEL_FAMILY_IDENTIFIER);
    let projection = provenance["source_projection"].as_str().ok_or("projection must be text")?;
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    for source in provenance["source_files"].as_array().ok_or("source_files must be an array")? {
        let path = source["path"].as_str().ok_or("source path must be text")?;
        let expected = source["sha256"].as_str().ok_or("source digest must be text")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), expected);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"].as_array().ok_or("models must be an array")?
        .iter().find(|row| row["feature_id"] == ideogram::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Ideogram4 catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 78);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "ideogram4");
    assert_eq!(sha256(&serde_json::to_vec(row)?), ideogram::MODEL_FAMILY_PROJECTION_SHA256);
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
        backend, shape, values, dtype, backend.device(), context,
    )?)
}

fn options(dtype: DType, memory_budget_bytes: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes,
        allow_unexpected_weights: false,
    }
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoints.iter().find(|checkpoint| checkpoint.name == name)
        .ok_or("Ideogram4 checkpoint is missing")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{name}: {actual} != {expected}");
    }
    Ok(())
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(ideogram::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
