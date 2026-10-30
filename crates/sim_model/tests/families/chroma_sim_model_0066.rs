use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipConfigurationFact, ModelClipModelInvocation,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits, PatchApplication,
    PatchGraph, PatchKind, PatchOperation, PatchTarget, build_model_family_for_probe,
    describe_model_family, generated_chroma_sim_model_0066 as chroma,
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

const ARTIFACT_DIGEST: &str = "0660660660660660660660660660660660660660660660660660660660660660";
const RESOLVED_MEMORY_BYTES: u64 = 21_064;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9066",
    identifier: "ChromaAmbiguousFixture",
    ..chroma::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    chroma::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 90,
        source_architecture: "model_base.ChromaAmbiguousFixture",
        ..chroma::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_chroma_source_projection_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(chroma::MODEL_FAMILY_IDENTIFIER, "Chroma");
    assert_eq!(chroma::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0066");
    assert_eq!(chroma::MODEL_FAMILY_FIXTURE, "chroma-comfy-model-0066");
    assert_eq!(chroma::MODEL_FAMILY_SOURCE_ORDINAL, 71);
    assert_eq!(chroma::MODEL_FAMILY_REGISTRATION.source_ordinal, 71);
    assert_eq!(
        chroma::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Chroma"
    );
    assert_eq!(chroma::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(chroma::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 3.2);

    let descriptor = describe_model_family(&chroma::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Chroma");
    assert_eq!(descriptor.family, "COMFY-MODEL-0066");
    assert_eq!(
        descriptor.architecture_version,
        "chroma-flux-transformer-v1"
    );
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(chroma::MODEL_FAMILY_SOURCE_PATH)
        )?),
        chroma::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], chroma::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], chroma::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 71);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("Chroma source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("Chroma source_files must be an array")?
    {
        let path = source["path"]
            .as_str()
            .ok_or("source path must be a string")?;
        let expected = source["sha256"]
            .as_str()
            .ok_or("source digest must be a string")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), expected);
    }

    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let catalog_row = catalog["models"]
        .as_array()
        .ok_or("model-family catalog models must be an array")?
        .iter()
        .find(|row| row["feature_id"] == chroma::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Chroma catalog row is missing")?;
    assert_eq!(catalog_row["source_ordinal"], 71);
    assert_eq!(catalog_row["source_symbol"], "Chroma");
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["image_model"],
        "chroma"
    );
    assert_eq!(
        catalog_row["static"]["latent_format"]["value"]["symbol"],
        "sim.latent_formats.Flux"
    );
    assert_eq!(
        sha256(&serde_json::to_vec(catalog_row)?),
        chroma::MODEL_FAMILY_PROJECTION_SHA256
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/chroma_sim_model_0066.rs"),
    )?;
    for canonical_adapter in [
        "ModelFamilyRegistration",
        "ModelStateTransformPlanDefinition",
        "ModelFamilyComponentStateSchema",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
        "ModelProbe",
    ] {
        assert!(row_source.contains(canonical_adapter));
    }
    for competing_owner in [
        "pub struct ",
        "pub enum ",
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
    super::write_model_family_row_artifact(
        chroma::MODEL_FAMILY_FIXTURE,
        chroma::MODEL_FAMILY_FEATURE_ID,
        chroma::MODEL_FAMILY_IDENTIFIER,
        chroma::MODEL_FAMILY_SOURCE_ORDINAL,
        "chroma_comfy_model_0066",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-and-unprefixed-detection",
            "transactional-component-and-scale-mapping",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-radiance-exclusion-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_chroma_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[chroma::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store("native")?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert!(!probe.metadata.contains_key("image_model"));
    assert!(!probe.metadata.contains_key("model_layout"));
    assert_eq!(
        probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0066"
    );
    assert_eq!(resolved.source_ordinal(), 71);
    assert_eq!(resolved.profile().latent_identifier, "Flux");
    assert_eq!(resolved.detection().evidence.len(), 3);
    assert!(
        resolved
            .detection()
            .evidence
            .iter()
            .all(|evidence| evidence.contains("AnyKeyPresent"))
    );

    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.pixart_t5.PixArtTokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.pixart_t5.pixart_te"
    );
    assert!(matches!(
        candidates[0].clip_model().invocation(),
        ModelClipModelInvocation::Factory { configuration }
            if matches!(configuration.as_slice(), [ModelClipConfigurationFact::Expand { source }]
                if source.as_str() == "sim.text_encoders.sd3_clip.t5_xxl_detect")
    ));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&backend, &context, "native", DType::F32, false)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 4);
    assert!(mapped.component("model").is_some_and(|model| {
        model.contains_key("native.double_blocks.0.img_attn.norm.key_norm.weight")
    }));
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());
    let guidance = mapped
        .component("runtime_conditioning")
        .and_then(|component| component.get("guidance_default"))
        .ok_or("generated Chroma guidance is missing")?;
    assert_eq!(guidance.descriptor().shape(), &[1]);
    assert_eq!(guidance.descriptor().dtype(), DType::F32);

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let build_options = options(DType::F32, RESOLVED_MEMORY_BYTES);
    let model = build_model_family_for_probe(&registry, &probe, weights, build_options)?;
    assert_eq!(model.memory_estimate().total_bytes, RESOLVED_MEMORY_BYTES);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "single_stream_projection",
        &[1.4621172, 0.8807971],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "flow_output",
        &[0.0, 0.96402675],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "chroma-single-stream-delta".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.single_blocks.0.linear2.weight".to_owned(),
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
        "flow_output",
        &[0.0, -0.9640262],
    )?;

    let replace_then_add = ordered_patch_graph(true)?;
    let add_then_replace = ordered_patch_graph(false)?;
    assert_ne!(
        replace_then_add.identity().ordered_digest,
        add_then_replace.identity().ordered_digest
    );
    let first = replace_then_add.apply(&backend, model.weights(), &context)?;
    let second = add_then_replace.apply(&backend, model.weights(), &context)?;
    assert_ne!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            &first.tensors()["native.single_blocks.0.linear2.weight"],
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            &second.tensors()["native.single_blocks.0.linear2.weight"],
            &context,
        )?
    );

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
            options(DType::F32, RESOLVED_MEMORY_BYTES - 1),
        ),
        Err(ModelFamilyError::OutOfMemory {
            required: RESOLVED_MEMORY_BYTES,
            budget,
        }) if budget == RESOLVED_MEMORY_BYTES - 1
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_chroma_unprefixed_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[chroma::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = probe_through_model_store("unprefixed")?;
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "model.");
    let resolved = registry.resolve(&probe)?;
    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let source = source_tensors(&backend, &context, "unprefixed", dtype, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert!(
            build_model_family_for_probe(
                &registry,
                &probe,
                weights,
                options(dtype, RESOLVED_MEMORY_BYTES),
            )
            .is_ok()
        );
    }

    let source = source_tensors(&backend, &context, "unprefixed", DType::F32, false)?;
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
            options(DType::F64, RESOLVED_MEMORY_BYTES),
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported_device = options(DType::F32, RESOLVED_MEMORY_BYTES);
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let partial_probe =
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, true, false, false))?;
    assert!(matches!(
        registry.detect(&partial_probe),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let malformed =
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, false, false, true))?;
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("expected [128]")
    ));
    let radiance =
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, false, true, false))?;
    assert!(matches!(
        registry.resolve(&radiance),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("ChromaRadiance")
    ));

    let mut unexpected_facts = parsed_facts("native", DType::F32, false, false, false);
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

    let no_match = ModelProbe {
        tensor_shapes: BTreeMap::new(),
        metadata: BTreeMap::from([("image_model".to_owned(), "chroma_radiance".to_owned())]),
    };
    assert!(matches!(
        registry.detect(&no_match),
        Err(ModelFamilyError::NoDetectionMatch)
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
        authority.authorize_workspace(4 * 1024 * 1024)?,
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

fn parsed_facts(
    layout: &str,
    dtype: DType,
    omit_final: bool,
    radiance: bool,
    malformed_norm: bool,
) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in model_shapes(omit_final, radiance, malformed_norm) {
        tensors.insert(
            format!("{prefix}{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for (key, shape) in [
        ("first_stage_model.decoder.weight", vec![1]),
        (
            "cond_stage_model.t5xxl.transformer.encoder.final_layer_norm.weight",
            vec![1],
        ),
    ] {
        tensors.insert(
            key.to_owned(),
            ModelParsedTensorFact {
                shape,
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

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_final: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(omit_final, false, false) {
        let values = match key.as_str() {
            "double_blocks.0.img_attn.proj.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "single_blocks.0.linear2.weight" => vec![2.0, 0.0, 0.0, 0.5],
            "final_layer.linear.weight" => vec![1.0, 1.0, 1.0, -1.0],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
        };
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, &shape, &values, dtype, context)?,
        );
    }
    source.insert(
        "first_stage_model.decoder.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    source.insert(
        "cond_stage_model.t5xxl.transformer.encoder.final_layer_norm.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn model_shapes(omit_final: bool, radiance: bool, malformed_norm: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "double_blocks.0.img_attn.norm.key_norm.scale".to_owned(),
            vec![if malformed_norm { 127 } else { 128 }],
        ),
        (
            "distilled_guidance_layer.norms.0.scale".to_owned(),
            vec![5_120],
        ),
        (
            "distilled_guidance_layer.in_proj.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "double_blocks.0.img_attn.proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("single_blocks.0.linear2.weight".to_owned(), vec![2, 2]),
    ];
    if !omit_final {
        shapes.push(("final_layer.linear.weight".to_owned(), vec![2, 2]));
    }
    if radiance {
        shapes.push(("nerf_blocks.0.norm.scale".to_owned(), vec![64]));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn probe_through_model_store(layout: &str) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("chroma.safetensors");
    write_safetensors(&model_path, layout)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "chroma-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("chroma-row", "chroma.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path, layout: &str) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(false, false, false);
    shapes.extend([
        ("first_stage_model.decoder.weight".to_owned(), vec![1]),
        (
            "cond_stage_model.t5xxl.transformer.encoder.final_layer_norm.weight".to_owned(),
            vec![1],
        ),
    ]);
    let mut data = Vec::new();
    for (key, shape) in shapes {
        let name = if key.starts_with("first_stage_model.") || key.starts_with("cond_stage_model.")
        {
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
    let checkpoint = checkpoints
        .iter()
        .find(|checkpoint| checkpoint.name == name)
        .ok_or("Chroma checkpoint is missing")?;
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

fn ordered_patch_graph(replace_first: bool) -> Result<PatchGraph, Box<dyn std::error::Error>> {
    let replacement = PatchOperation {
        identifier: "chroma-ordered-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.single_blocks.0.linear2.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "chroma-ordered-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.single_blocks.0.linear2.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Add,
        }],
    };
    Ok(PatchGraph::checked(
        ARTIFACT_DIGEST,
        if replace_first {
            vec![replacement, addition]
        } else {
            vec![addition, replacement]
        },
    )?)
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(chroma::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
