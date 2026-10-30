use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_cogvideox_i2v_sim_model_0068 as cogvideo,
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

const ARTIFACT_DIGEST: &str = "0068006800680068006800680068006800680068006800680068006800680068";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9068",
    identifier: "CogVideoX_I2V_AmbiguousFixture",
    ..cogvideo::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    cogvideo::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 92,
        source_architecture: "model_base.CogVideoXAmbiguousFixture",
        ..cogvideo::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_cogvideox_i2v_source_configuration_profiles_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(cogvideo::MODEL_FAMILY_IDENTIFIER, "CogVideoX_I2V");
    assert_eq!(cogvideo::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0068");
    assert_eq!(cogvideo::MODEL_FAMILY_SOURCE_ORDINAL, 90);
    assert_eq!(
        cogvideo::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.CogVideoX(image_to_video=True)"
    );
    assert_eq!(cogvideo::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);

    let descriptor = describe_model_family(&cogvideo::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "CogVideoX_I2V");
    assert_eq!(descriptor.latent_format, "CogVideoX");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );

    let registry =
        ModelFamilyRegistry::checked_registrations(&[cogvideo::MODEL_FAMILY_REGISTRATION])?;
    let spatial_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "native",
        DType::F32,
        PatchVariant::Spatial { channels: 32 },
        1,
        false,
    ))?;
    let spatial = cogvideo::configuration_for_probe(&spatial_probe)?;
    assert_eq!(spatial.layout, cogvideo::CogVideoXLayout::Native);
    assert_eq!(spatial.in_channels, 32);
    assert_eq!(spatial.number_of_attention_heads, 1);
    assert_eq!(spatial.temporal_patch_size, None);
    assert_eq!(
        (
            spatial.sample_height,
            spatial.sample_width,
            spatial.sample_frames
        ),
        (60, 90, 49)
    );
    assert_eq!(
        spatial.latent_variant,
        cogvideo::CogVideoXLatentVariant::CogVideoX
    );
    assert_eq!(
        registry
            .resolve(&spatial_probe)?
            .profile()
            .latent_identifier,
        "CogVideoX"
    );
    assert!(cogvideo::MODEL_FAMILY_REGISTRATION
        .source_configuration
        .is_empty());
    let mut misleading_probe = spatial_probe;
    misleading_probe.metadata.extend([
        ("image_model".to_owned(), "not-cogvideox".to_owned()),
        ("in_channels".to_owned(), "48".to_owned()),
        ("model_layout".to_owned(), "diffusers".to_owned()),
    ]);
    let misleading = registry.resolve(&misleading_probe)?;
    assert_eq!(
        misleading.detection().identity.feature_id(),
        cogvideo::MODEL_FAMILY_FEATURE_ID
    );
    let misleading_configuration = cogvideo::configuration_for_probe(&misleading_probe)?;
    assert_eq!(
        misleading_configuration.layout,
        cogvideo::CogVideoXLayout::Native
    );
    assert_eq!(misleading_configuration.in_channels, 32);

    let temporal_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "diffusers",
        DType::F32,
        PatchVariant::Temporal { channels: 32 },
        48,
        true,
    ))?;
    let temporal = cogvideo::configuration_for_probe(&temporal_probe)?;
    assert_eq!(temporal.layout, cogvideo::CogVideoXLayout::Diffusers);
    assert_eq!(temporal.temporal_patch_size, Some(2));
    assert_eq!(
        (
            temporal.sample_height,
            temporal.sample_width,
            temporal.sample_frames
        ),
        (96, 170, 81)
    );
    assert_eq!(temporal.text_embedding_dimension, Some(4_096));
    assert_eq!(temporal.ofs_embedding_dimension, Some(2));
    assert!(temporal.learned_positional_embeddings);
    assert_eq!(
        temporal.latent_variant,
        cogvideo::CogVideoXLatentVariant::CogVideoX1_5
    );
    assert_eq!(
        registry
            .resolve(&temporal_probe)?
            .profile()
            .latent_identifier,
        "CogVideoX1_5"
    );

    for channels in [16, 48] {
        let facts = parsed_facts(
            "native",
            DType::F32,
            PatchVariant::Spatial { channels },
            1,
            false,
        );
        let probe = ModelProbe::from_parsed_facts(facts)?;
        assert!(matches!(
            cogvideo::configuration_for_probe(&probe),
            Err(ModelFamilyError::InvalidSelectorOutput(message))
                if message.contains("requires 32")
        ));
        assert!(matches!(
            registry.detect(&probe),
            Err(ModelFamilyError::NoDetectionMatch)
        ));
    }

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("CogVideoX I2V source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("CogVideoX I2V source_files must be an array")?
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
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog must contain models")?
        .iter()
        .find(|row| row["feature_id"] == cogvideo::MODEL_FAMILY_FEATURE_ID)
        .ok_or("CogVideoX I2V catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 90);
    assert_eq!(row["static"]["unet_config"]["value"]["in_channels"], 32);

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/cogvideox_i2v_sim_model_0068.rs"),
    )?;
    for owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(owner));
    }
    super::write_model_family_row_artifact(
        cogvideo::MODEL_FAMILY_FIXTURE,
        cogvideo::MODEL_FAMILY_FEATURE_ID,
        cogvideo::MODEL_FAMILY_IDENTIFIER,
        cogvideo::MODEL_FAMILY_SOURCE_ORDINAL,
        "cogvideox_i2v_comfy_model_0068",
        &[
            "source-provenance-and-ownership",
            "spatial-and-temporal-patch-configuration",
            "dynamic-cogvideox-latent-profile",
            "i2v-versus-t2v-and-inpaint-detection",
            "transactional-mapping-forward-patching-memory",
            "dtype-device-cancellation-and-typed-failures",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_cogvideox_i2v_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[cogvideo::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    assert!(!probe.metadata.contains_key("image_model"));
    assert!(!probe.metadata.contains_key("in_channels"));
    assert!(!probe.metadata.contains_key("model_layout"));
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0068"
    );
    assert_eq!(resolved.source_ordinal(), 90);
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.detection().evidence.len(), 2);
    assert!(
        resolved
            .detection()
            .evidence
            .iter()
            .any(|evidence| evidence.contains("AnyKeyPresent"))
    );
    assert!(
        resolved
            .detection()
            .evidence
            .iter()
            .any(|evidence| evidence.contains("AnyTensorDimensionValue"))
    );
    let candidates = resolved.clip_target().candidates();
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.cogvideo.CogVideoXT5Tokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.sd3_clip.T5XXLModel"
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
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("model").is_some());
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model =
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 2_640))?;
    assert_eq!(model.memory_estimate().total_bytes, 2_640);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "video_output",
        &[0.8004987, -0.8004987],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "cogvideox-i2v-output-bias".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.proj_out.bias".to_owned(),
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
        "video_output",
        &[0.9216684, -0.53704894],
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 2_639)),
        Err(ModelFamilyError::OutOfMemory {
            required: 2_640,
            budget: 2_639
        })
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_cogvideox_i2v_diffusers_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[cogvideo::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = ModelProbe::from_parsed_facts(parsed_facts(
        "diffusers",
        DType::Bf16,
        PatchVariant::Spatial { channels: 32 },
        1,
        false,
    ))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, "diffusers", DType::Bf16, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    build_model_family_for_probe(&registry, &probe, weights, options(DType::Bf16, 2_640))?;

    let f16_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "diffusers",
        DType::F16,
        PatchVariant::Spatial { channels: 32 },
        1,
        false,
    ))?;
    let f16_resolved = registry.resolve(&f16_probe)?;
    let source_f16 = source_tensors(&backend, &context, "diffusers", DType::F16, false)?;
    let weights = f16_resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source_f16,
    )?;
    build_model_family_for_probe(&registry, &f16_probe, weights, options(DType::F16, 2_640))?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported = options(DType::Bf16, 2_640);
    unsupported.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut partial_facts = parsed_facts(
        "diffusers",
        DType::F32,
        PatchVariant::Spatial { channels: 32 },
        1,
        false,
    );
    partial_facts.tensors.remove("proj_out.weight");
    let partial_probe = ModelProbe::from_parsed_facts(partial_facts)?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let mut partial = source_tensors(&backend, &context, "diffusers", DType::F32, false)?;
    partial.remove("proj_out.weight");
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.proj_out.weight"
    ));

    let mut unexpected_facts = parsed_facts(
        "diffusers",
        DType::F32,
        PatchVariant::Spatial { channels: 32 },
        1,
        false,
    );
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected = source_tensors(&backend, &context, "diffusers", DType::F32, false)?;
    unexpected.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
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

#[derive(Clone, Copy)]
enum PatchVariant {
    Spatial { channels: u64 },
    Temporal { channels: u64 },
}

fn parsed_facts(
    layout: &str,
    dtype: DType,
    patch: PatchVariant,
    heads: u64,
    temporal_options: bool,
) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut shapes = model_shapes(patch, heads);
    if temporal_options {
        shapes.extend([
            ("patch_embed.text_proj.weight".to_owned(), vec![2, 4_096]),
            ("patch_embed.pos_embedding".to_owned(), vec![1, 2, 2]),
            ("ofs_embedding_linear_1.weight".to_owned(), vec![2, 2]),
        ]);
    }
    let mut tensors: BTreeMap<String, ModelParsedTensorFact> = shapes
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
        .collect();
    for key in [
        "vae.decoder.weight",
        "text_encoders.t5xxl.transformer.weight",
    ] {
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

fn model_shapes(patch: PatchVariant, heads: u64) -> Vec<(String, Vec<u64>)> {
    let patch_shape = match patch {
        PatchVariant::Spatial { channels } => vec![2, channels, 2, 2],
        PatchVariant::Temporal { channels } => vec![2, channels * 8],
    };
    vec![
        ("patch_embed.proj.weight".to_owned(), patch_shape),
        (
            "blocks.0.norm1.linear.weight".to_owned(),
            vec![heads * 64 * 6, 1],
        ),
        ("time_embedding_linear_1.weight".to_owned(), vec![2, 2]),
        ("time_embedding_linear_1.bias".to_owned(), vec![2]),
        ("time_embedding_linear_2.weight".to_owned(), vec![2, 2]),
        ("time_embedding_linear_2.bias".to_owned(), vec![2]),
        ("proj_out.weight".to_owned(), vec![2, 2]),
        ("proj_out.bias".to_owned(), vec![2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_projection: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(PatchVariant::Spatial { channels: 32 }, 1) {
        if omit_projection && key == "proj_out.weight" {
            continue;
        }
        let values = match key.as_str() {
            "time_embedding_linear_1.weight"
            | "time_embedding_linear_2.weight"
            | "proj_out.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "time_embedding_linear_1.bias" => vec![1.0, -1.0],
            "time_embedding_linear_2.bias" => vec![0.25, -0.25],
            "proj_out.bias" => vec![0.1, -0.1],
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
        "text_encoders.t5xxl.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("cogvideox-i2v.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "cogvideox-i2v-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("cogvideox-i2v-row", "cogvideox-i2v.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(PatchVariant::Spatial { channels: 32 }, 1);
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.t5xxl.transformer.weight".to_owned(), vec![1]),
    ]);
    let mut data = Vec::new();
    for (key, shape) in shapes {
        let name = if key.starts_with("vae.") || key.starts_with("text_encoders.") {
            key
        } else {
            format!("model.diffusion_model.{key}")
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
        .ok_or("CogVideoX I2V checkpoint is missing")?;
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
        .join(cogvideo::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
