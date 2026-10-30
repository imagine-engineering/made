use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_anima_sim_model_0063 as anima,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

const DIGEST: &str = "0630630630630630630630630630630630630630630630630630630630630630";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9063",
    identifier: "AnimaAmbiguousFixture",
    ..anima::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    anima::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 85,
        source_architecture: "model_base.AnimaAmbiguousFixture",
        ..anima::MODEL_FAMILY_REGISTRATION
    },
];

fn probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "model.diffusion_model.llm_adapter.in_proj.weight".to_owned(),
                vec![2, 2],
            ),
            (
                "model.diffusion_model.blocks.0.mlp.layer1.weight".to_owned(),
                vec![1],
            ),
            (
                "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight".to_owned(),
                vec![1],
            ),
            (
                "model.diffusion_model.x_embedder.proj.1.weight".to_owned(),
                vec![2_048, 68],
            ),
            ("first_stage_model.decoder.weight".to_owned(), vec![1]),
            (
                "cond_stage_model.qwen3_06b.transformer.weight".to_owned(),
                vec![1],
            ),
        ]),
        metadata: BTreeMap::from([
            ("image_model".to_owned(), "not-anima".to_owned()),
            ("model_layout".to_owned(), "diffusers".to_owned()),
        ]),
    }
}

#[test]
fn val_model_family_row_001_anima_source_bound_transactional_and_failure_typed()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(anima::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0063");
    assert_eq!(anima::MODEL_FAMILY_IDENTIFIER, "Anima");
    assert_eq!(anima::MODEL_FAMILY_REGISTRATION.source_ordinal, 84);
    assert_eq!(
        anima::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Anima"
    );

    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = fs::read(repository_root.join(anima::MODEL_FAMILY_SOURCE_PATH))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        anima::MODEL_FAMILY_SOURCE_SHA256
    );
    let row_source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/anima_sim_model_0063.rs"),
    )?;
    for forbidden_owner in [
        "pub struct ",
        "pub enum ",
        "std::process::Command",
        "fn commit(",
        "fn allocate(",
    ] {
        assert!(
            !row_source.contains(forbidden_owner),
            "Anima row must delegate foundational ownership: {forbidden_owner}"
        );
    }
    let provenance: serde_json::Value = serde_json::from_slice(&fs::read(
        repository_root
            .join("crates/sim_test_support/fixtures/models/anima-comfy-model-0063/provenance.json"),
    )?)?;
    assert_eq!(provenance["feature_id"], anima::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_ordinal"], 84);
    let source_projection = provenance["source_projection"]
        .as_str()
        .ok_or("Anima provenance projection missing")?;
    assert_eq!(
        format!("{:x}", Sha256::digest(source_projection.as_bytes())),
        provenance["source_projection_sha256"]
    );
    for source_file in provenance["source_files"]
        .as_array()
        .ok_or("Anima provenance source files missing")?
    {
        let relative_path = source_file["path"]
            .as_str()
            .ok_or("Anima provenance source path missing")?;
        let expected = source_file["sha256"]
            .as_str()
            .ok_or("Anima provenance source digest missing")?;
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(repository_root.join(relative_path))?)
            ),
            expected
        );
    }
    let catalog: serde_json::Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let projection = catalog["models"]
        .as_array()
        .ok_or("model-family catalog models missing")?
        .iter()
        .find(|model| model["feature_id"] == anima::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Anima source projection missing")?;
    assert_eq!(projection["source_ordinal"].as_u64(), Some(84));
    assert_eq!(projection["source_symbol"].as_str(), Some("Anima"));
    assert_eq!(projection["static"]["unet_config"]["value"]["image_model"], "anima");
    assert_eq!(
        format!("{:x}", Sha256::digest(serde_json::to_vec(projection)?)),
        anima::MODEL_FAMILY_PROJECTION_SHA256
    );

    let descriptor = describe_model_family(&anima::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);

    let registry = ModelFamilyRegistry::checked_registrations(&[
        anima::MODEL_FAMILY_REGISTRATION,
    ])?;
    let resolved = registry.resolve(&probe())?;
    assert_eq!(resolved.detection().identity.feature_id(), "COMFY-MODEL-0063");
    assert_eq!(resolved.source_ordinal(), 84);
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(
        resolved.clip_target().candidates()[0].tokenizer().identifier(),
        "sim.text_encoders.anima.AnimaTokenizer"
    );

    assert!(matches!(
        registry.detect(&ModelProbe {
            tensor_shapes: BTreeMap::new(),
            metadata: BTreeMap::new(),
        }),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let misleading = probe();
    assert_eq!(
        registry.detect(&misleading)?.identity.feature_id(),
        anima::MODEL_FAMILY_FEATURE_ID
    );
    let mut cosmos_predict2 = probe();
    cosmos_predict2.tensor_shapes.remove(
        "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
    );
    assert!(matches!(
        registry.detect(&cosmos_predict2),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let mut malformed = probe();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.x_embedder.proj.1.weight".to_owned(),
        vec![1_024, 68],
    );
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("model channels")
    ));
    let mut mixed = probe();
    mixed.tensor_shapes.extend([
        ("llm_adapter.in_proj.weight".to_owned(), vec![2, 2]),
        ("blocks.0.mlp.layer1.weight".to_owned(), vec![1]),
        (
            "llm_adapter.blocks.0.cross_attn.q_proj.weight".to_owned(),
            vec![1],
        ),
        ("x_embedder.proj.1.weight".to_owned(), vec![2_048, 68]),
    ]);
    assert!(matches!(
        registry.resolve(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe()),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_weights(&backend, DType::F32, &context)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &source,
    )?;
    assert_eq!(
        mapped.components().keys().map(String::as_str).collect::<Vec<_>>(),
        ["model", "text_encoder", "vae"]
    );
    assert!(mapped.components()["model"].contains_key("native.llm_adapter.in_proj.weight"));
    assert!(mapped.components()["vae"].contains_key("vae.decoder.weight"));
    assert!(mapped.components()["text_encoder"].contains_key("text_encoder.qwen3_06b.transformer.weight"));

    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let typed_source = source_weights(&backend, dtype, &context)?;
        let typed_weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            DIGEST,
            &typed_source,
        )?;
        let model = build_model_family_for_probe(
            &registry,
            &probe(),
            typed_weights,
            NativeFamilyBuildOptions {
                dtype,
                device: DeviceKind::Cpu,
                activation_elements: 2,
                memory_budget_bytes: u64::MAX,
                allow_unexpected_weights: false,
            },
        )?;
        assert_eq!(model.memory_estimate().weight_bytes, 278_540);
        assert_eq!(model.memory_estimate().activation_bytes, 4);
        assert_eq!(model.memory_estimate().total_bytes, 278_544);
    }
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe(),
            weights.clone(),
            NativeFamilyBuildOptions {
                dtype: DType::F32,
                device: DeviceKind::Cpu,
                activation_elements: 2,
                memory_budget_bytes: 278_543,
                allow_unexpected_weights: false,
            },
        ),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe(),
            weights.clone(),
            NativeFamilyBuildOptions {
                dtype: DType::F32,
                device: DeviceKind::Cuda,
                activation_elements: 2,
                memory_budget_bytes: 278_544,
                allow_unexpected_weights: false,
            },
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Cuda))
    ));

    let model = build_model_family_for_probe(
        &registry,
        &probe(),
        weights,
        NativeFamilyBuildOptions {
            dtype: DType::F32,
            device: DeviceKind::Cpu,
            activation_elements: 2,
            memory_budget_bytes: 278_544,
            allow_unexpected_weights: false,
        },
    )?;
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "llm_adapter.in_proj",
        &[5.0, 11.0],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "llm_adapter.normalized",
        &[-0.999_999_94, 0.999_999_94],
    )?;
    let patch = PatchGraph::checked(
        DIGEST,
        vec![PatchOperation {
            identifier: "anima-llm-adapter-delta".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.llm_adapter.in_proj.weight".to_owned(),
                expected_shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
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
        "llm_adapter.in_proj",
        &[6.0, 13.0],
    )?;
    let replacement = PatchOperation {
        identifier: "anima-llm-adapter-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.llm_adapter.in_proj.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![0.0, 0.0, 0.0, 0.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "anima-llm-adapter-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.llm_adapter.in_proj.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Add,
        }],
    };
    let replacement_then_addition = PatchGraph::checked(
        DIGEST,
        vec![replacement.clone(), addition.clone()],
    )?
    .apply(&backend, model.weights(), &context)?;
    let addition_then_replacement = PatchGraph::checked(DIGEST, vec![addition, replacement])?
        .apply(&backend, model.weights(), &context)?;
    assert_ne!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            replacement_then_addition
                .tensors()
                .get("native.llm_adapter.in_proj.weight")
                .ok_or("ordered Anima patch output is missing")?,
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            addition_then_replacement
                .tensors()
                .get("native.llm_adapter.in_proj.weight")
                .ok_or("reversed Anima patch output is missing")?,
            &context,
        )?,
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &cancelled_context),
            DIGEST,
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    super::write_model_family_row_artifact(
        anima::MODEL_FAMILY_FIXTURE,
        anima::MODEL_FAMILY_FEATURE_ID,
        anima::MODEL_FAMILY_IDENTIFIER,
        anima::MODEL_FAMILY_REGISTRATION.source_ordinal,
        "anima_comfy_model_0063",
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
fn val_model_family_row_001_anima_model_store_probes_preserve_mapping_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        anima::MODEL_FAMILY_REGISTRATION,
    ])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );

    let native_probe = probe_through_model_store("native")?;
    assert_eq!(native_probe.unet_prefix_selection()?.prefix(), "model.");
    let native = registry.resolve(&native_probe)?;
    let native_source = source_weights(&backend, DType::F32, &context)?;
    let native_weights = native.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &native_source,
    )?;
    assert!(native_weights.tensors().contains_key("native.llm_adapter.in_proj.weight"));

    let diffusers_probe = probe_through_model_store("diffusers")?;
    assert_eq!(diffusers_probe.unet_prefix_selection()?.prefix(), "model.");
    let diffusers = registry.resolve(&diffusers_probe)?;
    let diffusers_source = BTreeMap::from([
        (
            "llm_adapter.in_proj.weight".to_owned(),
            tensor(
                &backend,
                &[2, 2],
                &[1.0, 2.0, 3.0, 4.0],
                DType::F32,
                &context,
            )?,
        ),
        (
            "blocks.0.mlp.layer1.weight".to_owned(),
            tensor(&backend, &[1], &[0.0], DType::F32, &context)?,
        ),
        (
            "llm_adapter.blocks.0.cross_attn.q_proj.weight".to_owned(),
            tensor(&backend, &[1], &[0.0], DType::F32, &context)?,
        ),
        (
            "x_embedder.proj.1.weight".to_owned(),
            tensor(
                &backend,
                &[2_048, 68],
                &vec![0.0; 2_048 * 68],
                DType::F32,
                &context,
            )?,
        ),
        (
            "vae.decoder.weight".to_owned(),
            tensor(&backend, &[1], &[5.0], DType::F32, &context)?,
        ),
        (
            "text_encoders.qwen3_06b.transformer.weight".to_owned(),
            tensor(&backend, &[1], &[6.0], DType::F32, &context)?,
        ),
    ]);
    let diffusers_weights = diffusers.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &diffusers_source,
    )?;
    assert_eq!(
        diffusers_weights.tensors().keys().map(String::as_str).collect::<Vec<_>>(),
        [
            "native.blocks.0.mlp.layer1.weight",
            "native.llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "native.llm_adapter.in_proj.weight",
            "native.x_embedder.proj.1.weight",
        ]
    );
    Ok(())
}

fn parsed_facts(layout: &str) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let (vae_key, text_key) = if layout == "native" {
        (
            "first_stage_model.decoder.weight",
            "cond_stage_model.qwen3_06b.transformer.weight",
        )
    } else {
        (
            "vae.decoder.weight",
            "text_encoders.qwen3_06b.transformer.weight",
        )
    };
    ModelParsedFacts {
        tensors: BTreeMap::from([
            (
                format!("{prefix}llm_adapter.in_proj.weight"),
                ModelParsedTensorFact {
                    shape: vec![2, 2],
                    storage_dtype: "F32".to_owned(),
                },
            ),
            (
                format!("{prefix}blocks.0.mlp.layer1.weight"),
                ModelParsedTensorFact {
                    shape: vec![1],
                    storage_dtype: "F32".to_owned(),
                },
            ),
            (
                format!("{prefix}llm_adapter.blocks.0.cross_attn.q_proj.weight"),
                ModelParsedTensorFact {
                    shape: vec![1],
                    storage_dtype: "F32".to_owned(),
                },
            ),
            (
                format!("{prefix}x_embedder.proj.1.weight"),
                ModelParsedTensorFact {
                    shape: vec![2_048, 68],
                    storage_dtype: "F32".to_owned(),
                },
            ),
            (
                vae_key.to_owned(),
                ModelParsedTensorFact {
                    shape: vec![1],
                    storage_dtype: "F32".to_owned(),
                },
            ),
            (
                text_key.to_owned(),
                ModelParsedTensorFact {
                    shape: vec![1],
                    storage_dtype: "F32".to_owned(),
                },
            ),
        ]),
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::from([("image_model".to_owned(), "not-anima".to_owned())]),
        }],
    }
}

fn probe_through_model_store(layout: &str) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("anima.safetensors");
    write_safetensors(&model_path, &parsed_facts(layout))?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "anima-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("anima-row", "anima.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(
    path: &Path,
    facts: &ModelParsedFacts,
) -> Result<(), Box<dyn std::error::Error>> {
    let metadata = facts
        .formats
        .first()
        .ok_or("Anima safetensors fixture format is missing")?;
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::to_value(&metadata.metadata)?,
    );
    let mut data = Vec::new();
    for (name, tensor) in &facts.tensors {
        let start = data.len();
        let elements = tensor.shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or("Anima fixture shape overflow")
        })?;
        for _ in 0..elements {
            data.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": tensor.storage_dtype,
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
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let tensor = backend.upload_f32(descriptor, values, context)?.0;
    if dtype == DType::F32 {
        Ok(tensor)
    } else {
        Ok(sim_tensor::generated_sim_operator_indirection_01::cast_to_with_context_exact_native(
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

fn source_weights(
    backend: &CpuBackend,
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, sim_tensor::Tensor>, Box<dyn std::error::Error>> {
    Ok(BTreeMap::from([
        (
            "model.diffusion_model.llm_adapter.in_proj.weight".to_owned(),
            tensor(
                backend,
                &[2, 2],
                &[1.0, 2.0, 3.0, 4.0],
                dtype,
                context,
            )?,
        ),
        (
            "model.diffusion_model.blocks.0.mlp.layer1.weight".to_owned(),
            tensor(backend, &[1], &[0.0], dtype, context)?,
        ),
        (
            "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight".to_owned(),
            tensor(backend, &[1], &[0.0], dtype, context)?,
        ),
        (
            "model.diffusion_model.x_embedder.proj.1.weight".to_owned(),
            tensor(
                backend,
                &[2_048, 68],
                &vec![0.0; 2_048 * 68],
                dtype,
                context,
            )?,
        ),
        (
            "first_stage_model.decoder.weight".to_owned(),
            tensor(backend, &[1], &[5.0], dtype, context)?,
        ),
        (
            "cond_stage_model.qwen3_06b.transformer.weight".to_owned(),
            tensor(backend, &[1], &[6.0], dtype, context)?,
        ),
    ]))
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoints
        .iter()
        .find(|checkpoint| checkpoint.name == name)
        .ok_or_else(|| format!("missing Anima checkpoint {name}"))?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    if actual.len() != expected.len() {
        return Err(format!("Anima checkpoint {name} length mismatch").into());
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if (actual - expected).abs() > 1.0e-5 {
            return Err(format!(
                "Anima checkpoint {name} value {index} differs: {actual} versus {expected}"
            )
            .into());
        }
    }
    Ok(())
}
