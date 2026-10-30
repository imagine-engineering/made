use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_sd_x4upscaler_sim_model_0126 as sd_x4,
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

const ARTIFACT_DIGEST: &str = "1261261261261261261261261261261261261261261261261261261261261261";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9126",
    identifier: "SDX4UpscalerAmbiguousFixture",
    ..sd_x4::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sd_x4::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 114,
        source_architecture: "model_base.SDX4UpscalerAmbiguousFixture",
        ..sd_x4::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sd_x4upscaler_source_projection_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(sd_x4::MODEL_FAMILY_IDENTIFIER, "SD_X4Upscaler");
    assert_eq!(sd_x4::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0126");
    assert_eq!(sd_x4::MODEL_FAMILY_SOURCE_ORDINAL, 14);
    assert_eq!(sd_x4::MODEL_FAMILY_REGISTRATION.source_ordinal, 14);
    assert_eq!(
        sd_x4::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.SD_X4Upscaler"
    );
    assert_eq!(sd_x4::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.0);
    assert_eq!(sd_x4::MODEL_FAMILY_LINEAR_START, 0.0001);
    assert_eq!(sd_x4::MODEL_FAMILY_LINEAR_END, 0.02);

    let descriptor = describe_model_family(&sd_x4::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "SD_X4Upscaler");
    assert_eq!(descriptor.family, "COMFY-MODEL-0126");
    assert_eq!(descriptor.architecture_version, "sd-x4-upscaler-unet-v1");
    assert_eq!(descriptor.latent_format, "SD_X4");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);

    let probe = ModelProbe::from_parsed_facts(parsed_facts(DType::F32, false, false))?;
    let configuration = sd_x4::configuration_for_probe(&probe)?;
    assert_eq!(configuration.model_channels, 256);
    assert_eq!(configuration.input_channels, 7);
    assert_eq!(configuration.context_dimension, 1_024);
    assert!(configuration.linear_transformer_projection);
    assert!(!configuration.temporal_attention);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(repository.join(sd_x4::MODEL_FAMILY_SOURCE_PATH))?),
        sd_x4::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], sd_x4::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], sd_x4::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 14);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("SD_X4Upscaler source projection must be a string")?;
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    for source in provenance["source_files"]
        .as_array()
        .ok_or("source_files must be an array")?
    {
        let path = source["path"].as_str().ok_or("source path must be a string")?;
        let expected = source["sha256"].as_str().ok_or("source sha256 must be a string")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), expected);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog must contain models")?
        .iter()
        .find(|row| row["feature_id"] == sd_x4::MODEL_FAMILY_FEATURE_ID)
        .ok_or("SD_X4Upscaler catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 14);
    assert_eq!(row["static"]["unet_config"]["value"]["in_channels"], 7);
    assert_eq!(row["static"]["latent_format"]["value"]["symbol"], "latent_formats.SD_X4");
    assert_eq!(sha256(&serde_json::to_vec(row)?), sd_x4::MODEL_FAMILY_PROJECTION_SHA256);
    assert_eq!(provenance["catalog_projection_sha256"], sd_x4::MODEL_FAMILY_PROJECTION_SHA256);

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/sd_x4upscaler_sim_model_0126.rs"),
    )?;
    for owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(owner));
    }
    super::write_model_family_row_artifact(
        sd_x4::MODEL_FAMILY_FIXTURE,
        sd_x4::MODEL_FAMILY_FEATURE_ID,
        sd_x4::MODEL_FAMILY_IDENTIFIER,
        sd_x4::MODEL_FAMILY_SOURCE_ORDINAL,
        "sd_x4upscaler_comfy_model_0126",
        &[
            "source-provenance-catalog-and-ownership",
            "native-key-derived-normalized-configuration",
            "model-store-detection-and-native-layout",
            "transactional-component-mapping-and-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-misleading-metadata-and-diffusers-rejection",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sd_x4upscaler_execution_failures_and_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[sd_x4::MODEL_FAMILY_REGISTRATION])?;
    let model_store_probe = probe_through_model_store()?;
    assert_eq!(model_store_probe.unet_prefix_selection()?.prefix(), "model.diffusion_model.");
    assert_eq!(registry.detect(&model_store_probe)?.identity.feature_id(), "COMFY-MODEL-0126");

    let mut facts = parsed_facts(DType::F32, false, false);
    facts.formats[0]
        .metadata
        .insert("model_family".to_owned(), "misleading-svd".to_owned());
    let probe = ModelProbe::from_parsed_facts(facts)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.source_ordinal(), 14);
    assert_eq!(resolved.profile().latent_identifier, "SD_X4");
    assert_eq!(resolved.clip_target().candidates().len(), 1);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&backend, &context, DType::F32, false)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(mapped.component("model").is_some_and(|model| {
        model.contains_key("native.input_blocks.1.0.in_layers.2.weight")
    }));
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
        options(DType::F32, u64::MAX, DeviceKind::Cpu),
    )?;
    let required_memory = model.memory_estimate().total_bytes;
    assert!(required_memory > 0);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(&backend, &context, &checkpoints, "noise_level_conditioning", &[1.0, 2.0])?;
    assert_checkpoint(&backend, &context, &checkpoints, "upscaled_latent", &[0.0, 0.96402675])?;

    let replace_then_add = ordered_patch_graph(true)?.apply(&backend, model.weights(), &context)?;
    let add_then_replace = ordered_patch_graph(false)?.apply(&backend, model.weights(), &context)?;
    let first = model.with_weights(replace_then_add)?.forward_checkpoints(&backend, &input, &context)?;
    let second = model.with_weights(add_then_replace)?.forward_checkpoints(&backend, &input, &context)?;
    assert_ne!(
        checkpoint_values(&backend, &context, &first, "upscaled_latent")?,
        checkpoint_values(&backend, &context, &second, "upscaled_latent")?
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
            options(DType::F32, required_memory - 1, DeviceKind::Cpu),
        ),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == required_memory && budget == required_memory - 1
    ));

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let typed_probe = ModelProbe::from_parsed_facts(parsed_facts(dtype, false, false))?;
        let typed_resolved = registry.resolve(&typed_probe)?;
        let typed_source = source_tensors(&backend, &context, dtype, false)?;
        let typed_weights = typed_resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &typed_source,
        )?;
        build_model_family_for_probe(
            &registry,
            &typed_probe,
            typed_weights,
            options(dtype, u64::MAX, DeviceKind::Cpu),
        )?;
    }
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
            options(DType::F32, u64::MAX, DeviceKind::Metal),
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut partial = parsed_facts(DType::F32, true, false);
    let partial_probe = ModelProbe::from_parsed_facts(std::mem::take(&mut partial))?;
    assert!(matches!(registry.resolve(&partial_probe), Err(ModelFamilyError::ModelLayoutSelection(_))));
    let invalid_probe = ModelProbe::from_parsed_facts(parsed_facts(DType::F32, false, true))?;
    assert!(matches!(registry.detect(&invalid_probe), Err(ModelFamilyError::NoDetectionMatch)));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(ambiguous.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })));

    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts())?;
    assert!(matches!(registry.detect(&diffusers_probe), Err(ModelFamilyError::NoDetectionMatch)));
    assert!(matches!(
        sd_x4::configuration_for_probe(&diffusers_probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers detector table")
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
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

fn parsed_facts(dtype: DType, omit_forward_weight: bool, invalid_input: bool) -> ModelParsedFacts {
    let tensors = model_shapes(omit_forward_weight, invalid_input)
        .into_iter()
        .map(|(key, shape)| {
            (
                key,
                ModelParsedTensorFact {
                    shape,
                    storage_dtype: dtype.catalog_name().to_owned(),
                },
            )
        })
        .collect();
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::new(),
        }],
    }
}

fn model_shapes(omit_forward_weight: bool, invalid_input: bool) -> BTreeMap<String, Vec<u64>> {
    let prefix = "model.diffusion_model.";
    let mut shapes = BTreeMap::from([
        (format!("{prefix}input_blocks.0.0.weight"), if invalid_input { vec![256, 8, 3, 3] } else { vec![256, 7, 3, 3] }),
        (format!("{prefix}input_blocks.1.0.in_layers.0.weight"), vec![2]),
        (format!("{prefix}input_blocks.1.0.out_layers.3.weight"), vec![256, 2]),
        (format!("{prefix}input_blocks.1.0.in_layers.2.weight"), vec![2, 2]),
        (format!("{prefix}input_blocks.1.1.proj_in.weight"), vec![2, 2]),
        (format!("{prefix}input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"), vec![2, 1_024]),
        (format!("{prefix}middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"), vec![2, 2]),
        (format!("{prefix}output_blocks.0.0.in_layers.0.weight"), vec![2]),
        (format!("{prefix}output_blocks.0.0.in_layers.2.weight"), vec![2, 2]),
        (format!("{prefix}out.2.weight"), vec![4, 256, 3, 3]),
        ("first_stage_model.decoder.weight".to_owned(), vec![1]),
        ("cond_stage_model.model.weight".to_owned(), vec![1]),
    ]);
    if omit_forward_weight {
        shapes.remove(&format!("{prefix}output_blocks.0.0.in_layers.2.weight"));
    }
    shapes
}

fn diffusers_facts() -> ModelParsedFacts {
    let tensors = BTreeMap::from([
        ("conv_in.weight".to_owned(), ModelParsedTensorFact { shape: vec![256, 7, 3, 3], storage_dtype: "float32".to_owned() }),
        ("conv_out.weight".to_owned(), ModelParsedTensorFact { shape: vec![4, 256, 3, 3], storage_dtype: "float32".to_owned() }),
        ("down_blocks.0.resnets.0.conv1.weight".to_owned(), ModelParsedTensorFact { shape: vec![2, 2], storage_dtype: "float32".to_owned() }),
    ]);
    ModelParsedFacts { tensors, formats: vec![ModelParsedFormatFact { identity: "safetensors".to_owned(), metadata: BTreeMap::new() }] }
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    omit_forward_weight: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    model_shapes(omit_forward_weight, false)
        .into_iter()
        .map(|(key, shape)| {
            let elements = usize::try_from(shape.iter().product::<u64>())?;
            let values = match key.as_str() {
                "model.diffusion_model.input_blocks.1.0.in_layers.2.weight" => vec![1.0, 0.0, 0.0, 1.0],
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight" => vec![2.0, 0.0, 0.0, 0.5],
                "model.diffusion_model.output_blocks.0.0.in_layers.2.weight" => vec![1.0, 1.0, 1.0, -1.0],
                _ => vec![0.0; elements],
            };
            Ok((key, tensor(backend, &shape, &values, dtype, context)?))
        })
        .collect()
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("sd-x4.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical("sd-x4-row", "checkpoints", directory.path(), ["safetensors"])?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("sd-x4-row", "sd-x4.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::json!({"model_family": "misleading-svd"}));
    let mut data = Vec::new();
    for (key, shape) in model_shapes(false, false) {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        data.resize(data.len().checked_add(usize::try_from(elements.checked_mul(4).ok_or("fixture byte overflow")?)?) .ok_or("fixture length overflow")?, 0);
        header.insert(key, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
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
    Ok(tensor_from_f32_with_context_exact_native(backend, shape, values, dtype, backend.device(), context)?)
}

fn options(dtype: DType, memory_budget_bytes: u64, device: DeviceKind) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions { dtype, device, activation_elements: 2, memory_budget_bytes, allow_unexpected_weights: false }
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = checkpoint_values(backend, context, checkpoints, name)?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{name}: {actual} != {expected}");
    }
    Ok(())
}

fn checkpoint_values(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let checkpoint = checkpoints.iter().find(|checkpoint| checkpoint.name == name).ok_or("checkpoint is missing")?;
    Ok(tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?)
}

fn ordered_patch_graph(replace_first: bool) -> Result<PatchGraph, Box<dyn std::error::Error>> {
    let replacement = PatchOperation {
        identifier: "sd-x4-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "sd-x4-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Add,
        }],
    };
    Ok(PatchGraph::checked(ARTIFACT_DIGEST, if replace_first { vec![replacement, addition] } else { vec![addition, replacement] })?)
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(sd_x4::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
