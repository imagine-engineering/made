use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipModelInvocation, ModelFamilyError,
    ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact,
    ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits,
    PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family, generated_lens_sim_model_0104 as lens,
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

const ARTIFACT_DIGEST: &str = "0104010401040104010401040104010401040104010401040104010401040104";

#[derive(Clone, Copy)]
enum ClipFixture {
    Default,
    GptOssPrefixed,
    GptOssStandalone,
}

#[test]
fn val_model_family_row_001_lens_source_configuration_descriptor_clip_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(lens::MODEL_FAMILY_IDENTIFIER, "Lens");
    assert_eq!(lens::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0104");
    assert_eq!(lens::MODEL_FAMILY_SOURCE_ORDINAL, 81);
    assert_eq!(lens::MODEL_FAMILY_REGISTRATION.source_ordinal, 81);
    assert_eq!(lens::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.Lens");
    assert_eq!(lens::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 4.0);
    assert_eq!(lens::MODEL_FAMILY_SAMPLING_SHIFT, 1.829);

    let descriptor = describe_model_family(&lens::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Lens");
    assert_eq!(descriptor.family, "COMFY-MODEL-0104");
    assert_eq!(descriptor.architecture_version, "lens-dual-stream-mmdit-v1");
    assert_eq!(descriptor.latent_format, "Flux2");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    let native = ModelProbe::from_parsed_facts(parsed_facts(
        "native", "safetensors", DType::F32, ClipFixture::GptOssPrefixed,
    ))?;
    let configuration = lens::configuration_for_probe(&native)?;
    assert_eq!(configuration.layout, lens::LensLayout::PrefixedNative);
    assert_eq!(configuration.hidden_size, 2);
    assert_eq!(configuration.in_channels, 2);
    assert_eq!(configuration.out_channels, 1);
    assert_eq!(configuration.patch_size, 2);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.attention_heads, 2);
    assert_eq!(configuration.attention_head_dimension, 1);
    assert_eq!(configuration.text_feature_dimension, 2);
    assert_eq!(configuration.selected_text_layer_count, 2);
    assert!(configuration.multi_layer_text_features);
    assert_eq!(configuration.rope_axes_dimensions, [8, 28, 28]);

    let standalone = ModelProbe::from_parsed_facts(parsed_facts(
        "standalone", "safetensors", DType::Bf16, ClipFixture::GptOssStandalone,
    ))?;
    assert_eq!(
        lens::configuration_for_probe(&standalone)?.layout,
        lens::LensLayout::StandaloneNative
    );
    let diffusers = ModelProbe::from_parsed_facts(parsed_facts(
        "standalone", "diffusers", DType::F32, ClipFixture::Default,
    ))?;
    assert!(matches!(
        lens::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));

    let registry = ModelFamilyRegistry::checked_registrations(&[lens::MODEL_FAMILY_REGISTRATION])?;
    for (clip, expected_configuration_count) in [
        (ClipFixture::GptOssPrefixed, 1),
        (ClipFixture::GptOssStandalone, 1),
        (ClipFixture::Default, 0),
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(
            "native", "safetensors", DType::F32, clip,
        ))?;
        let resolved = registry.resolve(&probe)?;
        let candidate = &resolved.clip_target().candidates()[0];
        assert_eq!(
            candidate.tokenizer().identifier(),
            "sim.text_encoders.gpt_oss.LensTokenizer"
        );
        assert_eq!(candidate.clip_model().target().as_str(), "sim.text_encoders.gpt_oss.lens_te");
        assert!(matches!(
            candidate.clip_model().invocation(),
            ModelClipModelInvocation::Factory { configuration }
                if configuration.len() == expected_configuration_count
        ));
    }

    let store_probe = probe_through_model_store()?;
    assert_eq!(store_probe.unet_prefix_selection()?.prefix(), "model.diffusion_model.");
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        lens::MODEL_FAMILY_FEATURE_ID
    );

    validate_provenance_and_catalog()?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("family.json"))?)?;
    assert_eq!(fixture["fixture_id"], lens::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture["feature_id"], lens::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(fixture["dtype"], "f32");
    assert_eq!(fixture["device"], "cpu");
    assert_eq!(
        fixture["source_weights"].as_array().map(Vec::len),
        Some(8)
    );
    assert!(
        fixture["detector"]["tensor_shapes"]
            .get("text_encoders.gpt_oss.transformer.layers.0.self_attn.sinks")
            .is_some()
    );
    assert_eq!(fixture["checkpoints"].as_array().map(Vec::len), Some(6));
    assert_eq!(fixture["patches"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        fixture["patched_checkpoints"].as_array().map(Vec::len),
        Some(6)
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/lens_sim_model_0104.rs"),
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
        lens::MODEL_FAMILY_FIXTURE,
        lens::MODEL_FAMILY_FEATURE_ID,
        lens::MODEL_FAMILY_IDENTIFIER,
        lens::MODEL_FAMILY_SOURCE_ORDINAL,
        "lens_comfy_model_0104",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-and-standalone-detection",
            "source-exact-lens-configuration-and-diffusers-rejection",
            "source-exact-gpt-oss-prefix-precedence-and-default",
            "transactional-component-mapping-and-generated-conditioning",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation-partial-ambiguous-ownership",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_lens_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[lens::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT, authority.authorize_workspace(4 * 1024 * 1024)?, &cancellation,
    );

    for (layout, dtype) in [
        ("native", DType::F32), ("standalone", DType::F32), ("native", DType::Bf16),
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(
            layout, "safetensors", dtype, ClipFixture::GptOssPrefixed,
        ))?;
        let resolved = registry.resolve(&probe)?;
        let source = source_tensors(&backend, &context, layout, dtype, false)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &source,
        )?;
        assert_eq!(mapped.components().len(), 4);
        assert!(mapped.component("runtime_conditioning").is_some_and(|state| state.contains_key("sampling_shift")));
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
            assert_checkpoint(&backend, &context, &checkpoints, "image_output", &[0.761_593_8, -0.761_593_8])?;

            let patch = PatchGraph::checked(
                ARTIFACT_DIGEST,
                vec![PatchOperation {
                    identifier: "lens-image-output-scale".to_owned(),
                    kind: PatchKind::Lora,
                    scale: 1.0,
                    targets: vec![PatchTarget {
                        key: "native.transformer_blocks.0.attn.to_out.0.weight".to_owned(),
                        expected_shape: vec![2, 2],
                        values: vec![0.5, 0.0, 0.0, 0.5],
                        application: PatchApplication::Add,
                    }],
                }],
            )?;
            let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
            let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(
                &backend, &context, &patched_checkpoints, "image_output", &[0.905_147_9, -0.905_147_9],
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

    let valid = ModelProbe::from_parsed_facts(parsed_facts(
        "native", "safetensors", DType::F32, ClipFixture::GptOssPrefixed,
    ))?;
    let resolved = registry.resolve(&valid)?;
    let source = source_tensors(&backend, &context, "native", DType::F32, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &source,
    )?;
    assert!(build_model_family_for_probe(
        &registry, &valid, weights, options(DType::F32, DeviceKind::Metal, u64::MAX),
    ).is_err());

    let f16_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "native", "safetensors", DType::F16, ClipFixture::GptOssPrefixed,
    ))?;
    let f16_resolved = registry.resolve(&f16_probe)?;
    let f16_source = source_tensors(&backend, &context, "native", DType::F16, false)?;
    let f16_weights = f16_resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &f16_source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry, &f16_probe, f16_weights, options(DType::F16, DeviceKind::Cpu, u64::MAX),
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));

    let mut partial = parsed_facts("native", "safetensors", DType::F32, ClipFixture::Default);
    partial.tensors.remove("model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight");
    assert!(registry.resolve(&ModelProbe::from_parsed_facts(partial)?).is_err());

    let mut ambiguous = parsed_facts("native", "safetensors", DType::F32, ClipFixture::Default);
    for (key, shape) in model_shapes() {
        ambiguous.tensors.insert(
            key,
            ModelParsedTensorFact {
                shape,
                storage_dtype: DType::F32.catalog_name().to_owned(),
            },
        );
    }
    assert!(lens::configuration_for_probe(&ModelProbe::from_parsed_facts(ambiguous)?).is_err());

    let mut misleading = parsed_facts("native", "safetensors", DType::F32, ClipFixture::Default);
    misleading.tensors.get_mut("model.diffusion_model.proj_out.weight")
        .ok_or("output projection must exist")?.shape = vec![3, 2];
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
        StreamId::new(11), authority.authorize_workspace(4 * 1024 * 1024)?, &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &cancelled_context), ARTIFACT_DIGEST, &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

fn parsed_facts(layout: &str, format: &str, dtype: DType, clip: ClipFixture) -> ModelParsedFacts {
    let prefix = if layout == "native" { "model.diffusion_model." } else { "" };
    let mut tensors = model_shapes().into_iter().map(|(key, shape)| {
        (format!("{prefix}{key}"), ModelParsedTensorFact {
            shape, storage_dtype: dtype.catalog_name().to_owned(),
        })
    }).collect::<BTreeMap<_, _>>();
    let clip_key = match clip {
        ClipFixture::Default => None,
        ClipFixture::GptOssPrefixed => Some("text_encoders.gpt_oss.transformer.layers.0.self_attn.sinks"),
        ClipFixture::GptOssStandalone => Some("text_encoders.layers.0.self_attn.sinks"),
    };
    if let Some(clip_key) = clip_key {
        tensors.insert(clip_key.to_owned(), ModelParsedTensorFact {
            shape: vec![64], storage_dtype: dtype.catalog_name().to_owned(),
        });
    }
    tensors.insert("vae.decoder.weight".to_owned(), ModelParsedTensorFact {
        shape: vec![1], storage_dtype: dtype.catalog_name().to_owned(),
    });
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::from([
                ("image_model".to_owned(), "lens".to_owned()),
                ("model_layout".to_owned(), format.to_owned()),
            ]),
        }],
    }
}

fn model_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("img_in.weight".to_owned(), vec![2, 2]),
        ("proj_out.weight".to_owned(), vec![4, 2]),
        ("transformer_blocks.0.attn.norm_added_q.weight".to_owned(), vec![1]),
        ("transformer_blocks.0.img_mlp.w1.weight".to_owned(), vec![2, 2]),
        ("time_text_embed.timestep_embedder.linear_1.weight".to_owned(), vec![2, 2]),
        ("time_text_embed.timestep_embedder.linear_1.bias".to_owned(), vec![2]),
        ("transformer_blocks.0.img_mlp.w2.weight".to_owned(), vec![2, 2]),
        ("transformer_blocks.0.attn.to_out.0.weight".to_owned(), vec![2, 2]),
        ("txt_norm.0.weight".to_owned(), vec![2]),
        ("txt_norm.1.weight".to_owned(), vec![2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend, context: &ExecutionContext<'_>, layout: &str, dtype: DType,
    omit_output: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" { "model.diffusion_model." } else { "" };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes() {
        if omit_output && key == "transformer_blocks.0.attn.to_out.0.weight" { continue; }
        let values = match key.as_str() {
            "time_text_embed.timestep_embedder.linear_1.weight"
            | "transformer_blocks.0.img_mlp.w2.weight"
            | "transformer_blocks.0.attn.to_out.0.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "time_text_embed.timestep_embedder.linear_1.bias" => vec![1.0, -1.0],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
        };
        source.insert(format!("{prefix}{key}"), tensor(backend, &shape, &values, dtype, context)?);
    }
    source.insert("vae.decoder.weight".to_owned(), tensor(backend, &[1], &[1.0], dtype, context)?);
    source.insert(
        "text_encoders.gpt_oss.transformer.layers.0.self_attn.sinks".to_owned(),
        tensor(backend, &[64], &[0.0; 64], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("lens.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "lens-row", "checkpoints", directory.path(), ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("lens-row", "lens.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::json!({"image_model":"lens"}));
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
    assert_eq!(sha256(&std::fs::read(repository.join(lens::MODEL_FAMILY_SOURCE_PATH))?), lens::MODEL_FAMILY_SOURCE_SHA256);
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
        .find(|row| row["feature_id"] == lens::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Lens catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 81);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "lens");
    assert_eq!(sha256(&serde_json::to_vec(row)?), lens::MODEL_FAMILY_PROJECTION_SHA256);
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
        .ok_or("Lens checkpoint is missing")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{name}: {actual} != {expected}");
    }
    Ok(())
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../sim_test_support/fixtures/models")
        .join(lens::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
