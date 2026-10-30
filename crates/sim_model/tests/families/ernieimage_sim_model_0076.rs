use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_ernieimage_sim_model_0076 as ernie,
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

const ARTIFACT_DIGEST: &str = "0076007600760076007600760076007600760076007600760076007600760076";
const EXPECTED_MEMORY_BYTES: u64 = 1_164;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9076",
    identifier: "ErnieImageAmbiguousFixture",
    ..ernie::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    ernie::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 88,
        source_architecture: "model_base.ErnieImageAmbiguousFixture",
        ..ernie::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_ernieimage_source_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(ernie::MODEL_FAMILY_IDENTIFIER, "ErnieImage");
    assert_eq!(ernie::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0076");
    assert_eq!(ernie::MODEL_FAMILY_SOURCE_ORDINAL, 86);
    assert_eq!(
        ernie::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.ErnieImage"
    );
    assert_eq!(ernie::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 10.0);
    assert_eq!(ernie::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1_000.0);
    assert_eq!(ernie::MODEL_FAMILY_SAMPLING_SHIFT, 3.0);

    let descriptor = describe_model_family(&ernie::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "ErnieImage");
    assert_eq!(descriptor.latent_format, "Flux2");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);

    let registry = ModelFamilyRegistry::checked_registrations(&[ernie::MODEL_FAMILY_REGISTRATION])?;
    let native_probe =
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, tiny_shape()))?;
    let native = ernie::configuration_for_probe(&native_probe)?;
    assert_eq!(native.layout, ernie::ErnieImageLayout::Native);
    assert_eq!(native.hidden_size, 2);
    assert_eq!(native.number_of_attention_heads, 2);
    assert_eq!(native.number_of_layers, 1);
    assert_eq!(native.feed_forward_hidden_size, 2);
    assert_eq!(native.in_channels, 128);
    assert_eq!(native.out_channels, 2);
    assert_eq!(native.patch_size, 1);
    assert_eq!(native.text_input_dimension, 3);
    assert_eq!(native.rope_theta, 256);
    assert_eq!(native.rope_axes_dimensions, [32, 48, 48]);
    assert!(native.qk_layer_normalization);
    assert_eq!(
        registry.resolve(&native_probe)?.profile().latent_identifier,
        "Flux2"
    );

    let source_shape = ErnieShape {
        hidden_size: 4_096,
        attention_heads: 32,
        layers: 36,
        feed_forward_hidden_size: 12_288,
        in_channels: 128,
        out_channels: 128,
        patch_size: 1,
        text_input_dimension: 3_072,
    };
    let diffusers_probe =
        ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::Bf16, source_shape))?;
    let diffusers = ernie::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(diffusers.layout, ernie::ErnieImageLayout::Diffusers);
    assert_eq!(diffusers.hidden_size, 4_096);
    assert_eq!(diffusers.number_of_attention_heads, 32);
    assert_eq!(diffusers.number_of_layers, 36);
    assert_eq!(diffusers.feed_forward_hidden_size, 12_288);
    assert_eq!(diffusers.in_channels, 128);
    assert_eq!(diffusers.out_channels, 128);
    assert_eq!(diffusers.text_input_dimension, 3_072);

    let mut malformed = parsed_facts("native", DType::F32, tiny_shape());
    malformed
        .tensors
        .get_mut("model.diffusion_model.layers.0.self_attention.norm_q.weight")
        .ok_or("query normalization fact must exist")?
        .shape = vec![3];
    let malformed_probe = ModelProbe::from_parsed_facts(malformed)?;
    assert!(matches!(
        ernie::configuration_for_probe(&malformed_probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("attention head dimension")
    ));

    let mut misleading_metadata = parsed_facts("native", DType::F32, tiny_shape());
    misleading_metadata.formats[0]
        .metadata
        .insert("image_model".to_owned(), "flux2".to_owned());
    let misleading_probe = ModelProbe::from_parsed_facts(misleading_metadata)?;
    assert_eq!(
        registry.detect(&misleading_probe)?.identity.feature_id(),
        ernie::MODEL_FAMILY_FEATURE_ID
    );

    let mut partial_family = parsed_facts("native", DType::F32, tiny_shape());
    partial_family
        .tensors
        .remove("model.diffusion_model.final_linear.weight");
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
        .ok_or("ErnieImage source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("ErnieImage source_files must be an array")?
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
        .find(|row| row["feature_id"] == ernie::MODEL_FAMILY_FEATURE_ID)
        .ok_or("ErnieImage catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 86);
    assert_eq!(
        row["static"]["unet_config"]["value"]["image_model"],
        "ernie"
    );
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 10.0);
    assert_eq!(
        row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.Flux2"
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/ernieimage_sim_model_0076.rs"),
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
        ernie::MODEL_FAMILY_FIXTURE,
        ernie::MODEL_FAMILY_FEATURE_ID,
        ernie::MODEL_FAMILY_IDENTIFIER,
        ernie::MODEL_FAMILY_SOURCE_ORDINAL,
        "ernieimage_comfy_model_0076",
        &[
            "source-provenance-and-ownership",
            "native-and-diffusers-configuration",
            "flux2-latent-and-ministral-clip-target",
            "ernie-marker-and-cross-family-rejection",
            "transactional-mapping-forward-patching-memory",
            "dtype-device-cancellation-and-typed-failures",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_ernieimage_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[ernie::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        ernie::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(resolved.source_ordinal(), 86);
    let candidates = resolved.clip_target().candidates();
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.ernie.ErnieTokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.ernie.te"
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
        "image_output",
        &[0.8004987, -0.8004987],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "ernieimage-output-bias".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.final_linear.bias".to_owned(),
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
        Err(ModelFamilyError::OutOfMemory {
            required: EXPECTED_MEMORY_BYTES,
            budget
        }) if budget == EXPECTED_MEMORY_BYTES - 1
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_ernieimage_diffusers_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[ernie::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let probe =
        ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::Bf16, tiny_shape()))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, "diffusers", DType::Bf16, false)?;
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

    let f16_probe =
        ModelProbe::from_parsed_facts(parsed_facts("diffusers", DType::F16, tiny_shape()))?;
    let f16_resolved = registry.resolve(&f16_probe)?;
    let source_f16 = source_tensors(&backend, &context, "diffusers", DType::F16, false)?;
    let weights = f16_resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source_f16,
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &f16_probe,
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

    let mut partial_facts = parsed_facts("diffusers", DType::F32, tiny_shape());
    partial_facts
        .tensors
        .remove("time_embedding.linear_1.bias");
    let partial_probe = ModelProbe::from_parsed_facts(partial_facts)?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let mut partial = source_tensors(&backend, &context, "diffusers", DType::F32, false)?;
    partial.remove("time_embedding.linear_1.bias");
    let partial_result = partial_resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &partial,
    );
    assert!(matches!(
        &partial_result,
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.time_embedding.linear_1.bias"
    ), "unexpected partial mapping result: {partial_result:?}");

    let mut unexpected_facts = parsed_facts("diffusers", DType::F32, tiny_shape());
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
struct ErnieShape {
    hidden_size: u64,
    attention_heads: u64,
    layers: usize,
    feed_forward_hidden_size: u64,
    in_channels: u64,
    out_channels: u64,
    patch_size: u64,
    text_input_dimension: u64,
}

fn tiny_shape() -> ErnieShape {
    ErnieShape {
        hidden_size: 2,
        attention_heads: 2,
        layers: 1,
        feed_forward_hidden_size: 2,
        in_channels: 128,
        out_channels: 2,
        patch_size: 1,
        text_input_dimension: 3,
    }
}

fn parsed_facts(layout: &str, dtype: DType, shape: ErnieShape) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors: BTreeMap<String, ModelParsedTensorFact> = model_shapes(shape)
        .into_iter()
        .map(|(key, dimensions)| {
            (
                format!("{prefix}{key}"),
                ModelParsedTensorFact {
                    shape: dimensions,
                    storage_dtype: dtype.catalog_name().to_owned(),
                },
            )
        })
        .collect();
    for key in [
        "vae.decoder.weight",
        "text_encoders.ministral3_3b.transformer.weight",
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

fn model_shapes(shape: ErnieShape) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "x_embedder.proj.weight".to_owned(),
            vec![
                shape.hidden_size,
                shape.in_channels,
                shape.patch_size,
                shape.patch_size,
            ],
        ),
        (
            "text_proj.weight".to_owned(),
            vec![shape.hidden_size, shape.text_input_dimension],
        ),
        (
            "time_embedding.linear_1.weight".to_owned(),
            vec![shape.hidden_size, shape.hidden_size],
        ),
        (
            "time_embedding.linear_1.bias".to_owned(),
            vec![shape.hidden_size],
        ),
        (
            "time_embedding.linear_2.weight".to_owned(),
            vec![shape.hidden_size, shape.hidden_size],
        ),
        (
            "time_embedding.linear_2.bias".to_owned(),
            vec![shape.hidden_size],
        ),
        (
            "layers.0.self_attention.to_q.weight".to_owned(),
            vec![shape.hidden_size, shape.hidden_size],
        ),
        (
            "layers.0.self_attention.norm_q.weight".to_owned(),
            vec![shape.hidden_size / shape.attention_heads],
        ),
        (
            "final_linear.weight".to_owned(),
            vec![
                shape.patch_size * shape.patch_size * shape.out_channels,
                shape.hidden_size,
            ],
        ),
        (
            "final_linear.bias".to_owned(),
            vec![shape.patch_size * shape.patch_size * shape.out_channels],
        ),
    ];
    for layer in 0..shape.layers {
        shapes.push((
            format!("layers.{layer}.mlp.linear_fc2.weight"),
            vec![shape.hidden_size, shape.feed_forward_hidden_size],
        ));
    }
    shapes
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
    for (key, shape) in model_shapes(tiny_shape()) {
        if omit_projection && key == "final_linear.weight" {
            continue;
        }
        let values = match key.as_str() {
            "time_embedding.linear_1.weight"
            | "time_embedding.linear_2.weight"
            | "layers.0.mlp.linear_fc2.weight"
            | "final_linear.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "time_embedding.linear_1.bias" => vec![1.0, -1.0],
            "time_embedding.linear_2.bias" => vec![0.25, -0.25],
            "final_linear.bias" => vec![0.1, -0.1],
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
        "text_encoders.ministral3_3b.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("ernieimage.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "ernieimage-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("ernieimage-row", "ernieimage.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(tiny_shape());
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        (
            "text_encoders.ministral3_3b.transformer.weight".to_owned(),
            vec![1],
        ),
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
        .ok_or("ErnieImage checkpoint is missing")?;
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
        .join(ernie::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
