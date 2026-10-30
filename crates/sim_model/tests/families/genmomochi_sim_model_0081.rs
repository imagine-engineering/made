use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipConfigurationFact,
    ModelClipModelInvocation, ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration,
    ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits, PatchApplication,
    PatchGraph, PatchKind, PatchOperation, PatchTarget, build_model_family_for_probe,
    describe_model_family, generated_genmomochi_sim_model_0081 as mochi,
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

const ARTIFACT_DIGEST: &str = "0081008100810081008100810081008100810081008100810081008100810081";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9081",
    identifier: "GenmoMochi_AmbiguousFixture",
    ..mochi::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    mochi::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 131,
        source_architecture: "model_base.GenmoMochiAmbiguousFixture",
        ..mochi::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_genmomochi_source_configuration_profiles_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(mochi::MODEL_FAMILY_IDENTIFIER, "GenmoMochi");
    assert_eq!(mochi::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0081");
    assert_eq!(mochi::MODEL_FAMILY_SOURCE_ORDINAL, 31);
    assert_eq!(
        mochi::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.GenmoMochi"
    );
    assert_eq!(mochi::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);
    assert_eq!(mochi::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(mochi::MODEL_FAMILY_SAMPLING_SHIFT, 6.0);

    let descriptor = describe_model_family(&mochi::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "GenmoMochi");
    assert_eq!(descriptor.latent_format, "Mochi");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 2);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 2);

    let registry =
        ModelFamilyRegistry::checked_registrations(&[mochi::MODEL_FAMILY_REGISTRATION])?;
    let native_probe = ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32))?;
    let native = mochi::configuration_for_probe(&native_probe)?;
    assert_eq!(native.layout, mochi::GenmoMochiLayout::Native);
    assert_eq!(native.depth, 48);
    assert_eq!(native.patch_size, 2);
    assert_eq!(native.number_of_attention_heads, 24);
    assert_eq!(native.hidden_size_x, 2);
    assert_eq!(native.hidden_size_y, 2);
    assert_eq!(native.text_feature_dimension, 2);
    assert_eq!(native.in_channels, 12);
    assert_eq!(native.out_channels, 12);
    assert_eq!((native.mlp_ratio_x, native.mlp_ratio_y), (4, 4));
    assert!(native.qk_normalization);
    assert!(!native.qkv_bias);
    assert!(native.output_bias);
    assert!(native.patch_embedding_bias);
    assert!(native.positional_encoding_preserves_area);
    assert!(native.timestep_mlp_bias);
    assert!(!native.attends_to_padding);
    assert_eq!(
        registry
            .resolve(&native_probe)?
            .profile()
            .latent_identifier,
        "Mochi"
    );

    let source_probe = ModelProbe::from_parsed_facts(parsed_facts_with_dimensions(
        "diffusers",
        DType::Bf16,
        3_072,
        1_536,
        4_096,
        12,
    ))?;
    let source = mochi::configuration_for_probe(&source_probe)?;
    assert_eq!(source.layout, mochi::GenmoMochiLayout::Diffusers);
    assert_eq!(source.hidden_size_x, 3_072);
    assert_eq!(source.hidden_size_y, 1_536);
    assert_eq!(source.text_feature_dimension, 4_096);

    let mut invalid_channels = parsed_facts("native", DType::F32);
    invalid_channels
        .tensors
        .get_mut("model.diffusion_model.x_embedder.proj.weight")
        .ok_or("Mochi patch projection must exist")?
        .shape = vec![2, 16, 2, 2];
    let invalid_channels = ModelProbe::from_parsed_facts(invalid_channels)?;
    assert!(matches!(
        mochi::configuration_for_probe(&invalid_channels),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("12, 2, 2")
    ));

    let mut misleading_metadata = parsed_facts("native", DType::F32);
    misleading_metadata.formats[0]
        .metadata
        .insert("image_model".to_owned(), "ltxv".to_owned());
    let misleading_probe = ModelProbe::from_parsed_facts(misleading_metadata)?;
    assert_eq!(
        registry.detect(&misleading_probe)?.identity.feature_id(),
        mochi::MODEL_FAMILY_FEATURE_ID
    );

    let mut partial_family = parsed_facts("native", DType::F32);
    partial_family
        .tensors
        .remove("model.diffusion_model.final_layer.linear.weight");
    let partial_probe = ModelProbe::from_parsed_facts(partial_family)?;
    assert!(matches!(
        registry.detect(&partial_probe),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("GenmoMochi source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    assert_eq!(
        mochi::MODEL_FAMILY_PROJECTION_SHA256,
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("GenmoMochi source_files must be an array")?
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
        .find(|row| row["feature_id"] == mochi::MODEL_FAMILY_FEATURE_ID)
        .ok_or("GenmoMochi catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 31);
    assert_eq!(
        row["static"]["unet_config"]["value"]["image_model"],
        "mochi_preview"
    );
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 2.0);
    assert_eq!(
        row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.Mochi"
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/genmomochi_sim_model_0081.rs"),
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
        mochi::MODEL_FAMILY_FIXTURE,
        mochi::MODEL_FAMILY_FEATURE_ID,
        mochi::MODEL_FAMILY_IDENTIFIER,
        mochi::MODEL_FAMILY_SOURCE_ORDINAL,
        "genmomochi_comfy_model_0081",
        &[
            "source-provenance-and-ownership",
            "native-and-diffusers-source-configuration",
            "canonical-mochi-latent-and-t5-target",
            "mochi-marker-and-cross-family-rejection",
            "transactional-mapping-forward-patching-memory",
            "dtype-device-cancellation-and-typed-failures",
            "generated-token-count-conditioning",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_genmomochi_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[mochi::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0081"
    );
    assert_eq!(resolved.source_ordinal(), 31);
    let candidates = resolved.clip_target().candidates();
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.genmo.MochiT5Tokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.genmo.mochi_te"
    );
    assert!(matches!(
        candidates[0].clip_model().invocation(),
        ModelClipModelInvocation::Factory { configuration }
            if matches!(configuration.as_slice(), [ModelClipConfigurationFact::Expand { source }]
                if source.as_str() == "sim.text_encoders.sd3_clip.t5_xxl_detect")
    ));

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
    assert!(mapped.component("model").is_some());
    assert!(mapped.component("runtime_conditioning").is_some_and(|state| {
        state.contains_key("num_tokens_default")
    }));
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model =
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 252))?;
    assert_eq!(model.memory_estimate().total_bytes, 252);
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
            identifier: "genmomochi-video-output-bias".to_owned(),
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
        "video_output",
        &[0.9216684, -0.53704894],
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 251)),
        Err(ModelFamilyError::OutOfMemory {
            required: 252,
            budget: 251
        })
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_genmomochi_diffusers_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[mochi::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::Bf16))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, "diffusers", DType::Bf16, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    build_model_family_for_probe(&registry, &probe, weights, options(DType::Bf16, 252))?;

    let f32_probe = ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::F32))?;
    let f32_resolved = registry.resolve(&f32_probe)?;
    let source_f32 = source_tensors(&backend, &context, "diffusers", DType::F32, false)?;
    let weights = f32_resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source_f32,
    )?;
    build_model_family_for_probe(&registry, &f32_probe, weights, options(DType::F32, 252))?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F16, 252)),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported = options(DType::Bf16, 252);
    unsupported.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut partial_facts = parsed_facts("diffusers", DType::F32);
    partial_facts.tensors.remove("t_embedder.mlp.2.weight");
    let partial_probe = ModelProbe::from_parsed_facts(partial_facts)?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let mut partial = source_tensors(&backend, &context, "diffusers", DType::F32, false)?;
    partial.remove("t_embedder.mlp.2.weight");
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.t_embedder.mlp.2.weight"
    ));

    let mut unexpected_facts = parsed_facts("diffusers", DType::F32);
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

fn parsed_facts(layout: &str, dtype: DType) -> ModelParsedFacts {
    parsed_facts_with_dimensions(layout, dtype, 2, 2, 2, 12)
}

fn parsed_facts_with_dimensions(
    layout: &str,
    dtype: DType,
    hidden_size_x: u64,
    hidden_size_y: u64,
    text_feature_dimension: u64,
    in_channels: u64,
) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors: BTreeMap<String, ModelParsedTensorFact> = model_shapes(
        hidden_size_x,
        hidden_size_y,
        text_feature_dimension,
        in_channels,
    )
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

fn model_shapes(
    hidden_size_x: u64,
    hidden_size_y: u64,
    text_feature_dimension: u64,
    in_channels: u64,
) -> Vec<(String, Vec<u64>)> {
    vec![
        (
            "x_embedder.proj.weight".to_owned(),
            vec![hidden_size_x, in_channels, 2, 2],
        ),
        (
            "t_embedder.mlp.0.weight".to_owned(),
            vec![hidden_size_x, 2],
        ),
        ("t_embedder.mlp.0.bias".to_owned(), vec![hidden_size_x]),
        (
            "t_embedder.mlp.2.weight".to_owned(),
            vec![hidden_size_x, hidden_size_x],
        ),
        ("t_embedder.mlp.2.bias".to_owned(), vec![hidden_size_x]),
        (
            "t5_yproj.weight".to_owned(),
            vec![hidden_size_y, text_feature_dimension],
        ),
        (
            "blocks.0.attn.proj_x.weight".to_owned(),
            vec![hidden_size_x, hidden_size_x],
        ),
        (
            "blocks.0.attn.proj_x.bias".to_owned(),
            vec![hidden_size_x],
        ),
        (
            "final_layer.linear.weight".to_owned(),
            vec![hidden_size_x, hidden_size_x],
        ),
        (
            "final_layer.linear.bias".to_owned(),
            vec![hidden_size_x],
        ),
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
    for (key, shape) in model_shapes(2, 2, 2, 12) {
        if omit_projection && key == "final_layer.linear.weight" {
            continue;
        }
        let values = match key.as_str() {
            "t_embedder.mlp.0.weight"
            | "t_embedder.mlp.2.weight"
            | "blocks.0.attn.proj_x.weight"
            | "final_layer.linear.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "t_embedder.mlp.0.bias" => vec![1.0, -1.0],
            "t_embedder.mlp.2.bias" => vec![0.25, -0.25],
            "blocks.0.attn.proj_x.bias" => vec![0.0, 0.0],
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
        "text_encoders.t5xxl.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("genmomochi.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "genmomochi-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("genmomochi-row", "genmomochi.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(2, 2, 2, 12);
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
        .ok_or("GenmoMochi checkpoint is missing")?;
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
        .join(mochi::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
