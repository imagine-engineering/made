use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_sd15_sim_model_0117 as sd15,
    generated_sd15_instructpix2pix_sim_model_0118 as instruct,
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

const DIGEST: &str = "0117011701170117011701170117011701170117011701170117011701170117";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9117",
    identifier: "SD15_AmbiguousFixture",
    ..sd15::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sd15::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 103,
        source_architecture: "model_base.SD15_AmbiguousFixture",
        ..sd15::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sd15_source_configuration_provenance_store_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(sd15::MODEL_FAMILY_IDENTIFIER, "SD15");
    assert_eq!(sd15::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0117");
    assert_eq!(sd15::MODEL_FAMILY_SOURCE_ORDINAL, 3);
    assert_eq!(sd15::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.BaseModel");
    assert_eq!(sd15::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.0);
    let descriptor = describe_model_family(&sd15::MODEL_FAMILY)?;
    assert_eq!(descriptor.architecture_version, "sd15-v1");
    assert_eq!(descriptor.latent_format, "SD15");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);

    let registry = ModelFamilyRegistry::checked_registrations(&[
        instruct::MODEL_FAMILY_REGISTRATION,
        sd15::MODEL_FAMILY_REGISTRATION,
    ])?;
    for (probe, layout) in [
        (native_probe(false), sim_model::ModelStateLayout::PrefixedNative),
        (diffusers_probe(), sim_model::ModelStateLayout::Diffusers),
    ] {
        assert_eq!(registry.detect(&probe)?.identity.feature_id(), sd15::MODEL_FAMILY_FEATURE_ID);
        let configuration = sd15::configuration_for_probe(&probe)?;
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, 4);
        assert_eq!(configuration.model_channels, 320);
        assert_eq!(configuration.context_dimension, 768);
        assert_eq!(configuration.attention_heads, 8);
        assert!(!configuration.uses_linear_transformer_projection);
        assert!(!configuration.uses_temporal_attention);
        assert_eq!(configuration.adm_in_channels, None);
    }
    let stored = probe_through_model_store()?;
    assert_eq!(registry.detect(&stored)?.identity.feature_id(), sd15::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(sd15::configuration_for_probe(&stored)?.in_channels, 4);

    let mut misleading = native_probe(false);
    misleading.metadata.insert("model_layout".into(), "diffusers".into());
    misleading.metadata.insert("model_family".into(), "SD3".into());
    assert_eq!(registry.detect(&misleading)?.identity.feature_id(), sd15::MODEL_FAMILY_FEATURE_ID);
    let mut partial = native_probe(false);
    partial.tensor_shapes.remove("model.diffusion_model.input_blocks.0.0.weight");
    assert!(matches!(registry.detect(&partial), Err(ModelFamilyError::NoDetectionMatch)));
    let mut malformed = native_probe(false);
    malformed.tensor_shapes.get_mut("model.diffusion_model.input_blocks.0.0.weight").ok_or("input")?[0] = 319;
    assert!(matches!(
        sd15::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("expected [320, 4]")
    ));

    verify_provenance_and_catalog()?;
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/sd15_sim_model_0117.rs"),
    )?;
    for forbidden in ["struct Tensor", "struct ModelStore", "struct ModelProbe", "struct PatchGraph", "std::fs", "unsafe "] {
        assert!(!source.contains(forbidden));
    }
    super::write_model_family_row_artifact(
        sd15::MODEL_FAMILY_FIXTURE,
        sd15::MODEL_FAMILY_FEATURE_ID,
        sd15::MODEL_FAMILY_IDENTIFIER,
        sd15::MODEL_FAMILY_SOURCE_ORDINAL,
        "sd15_comfy_model_0117",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-and-diffusers-detection",
            "exact-source-configuration-and-sd1-clip-target",
            "transactional-unet-clip-vae-mapping",
            "named-forward-conditioning-checkpoints-and-patching",
            "memory-oom-dtype-device-cancellation",
            "partial-malformed-ambiguous-unexpected-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sd15_state_plans_round_clip_and_map_diffusers_exactly()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[sd15::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );

    let probe = native_probe(true);
    let source = native_source(&backend, &context, DType::F32, true)?;
    let resolved = registry.resolve(&probe)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &source,
    )?;
    let clip = mapped.component("text_encoder").ok_or("clip")?;
    assert_values(
        &backend,
        &context,
        &clip["clip_l.transformer.text_model.embeddings.position_ids"],
        &[0.0, 2.0, 2.0],
    )?;
    assert!(mapped.component("denoiser").ok_or("denoiser")?.contains_key("native.input_blocks.0.0.weight"));
    assert!(mapped.component("vae").ok_or("vae")?.contains_key("native.decoder.weight"));

    let diffusers_probe = diffusers_probe();
    let diffusers_source = diffusers_source(&backend, &context, DType::F32)?;
    let mapped = registry.resolve(&diffusers_probe)?.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &diffusers_source,
    )?;
    let denoiser = mapped.component("denoiser").ok_or("denoiser")?;
    for key in sd15::REQUIRED_KEYS {
        assert!(denoiser.contains_key(*key), "missing mapped {key}");
    }

    let mut unexpected_probe = native_probe(false);
    unexpected_probe.tensor_shapes.insert("legacy.python.extension".into(), vec![1]);
    let mut unexpected_source = native_source(&backend, &context, DType::F32, false)?;
    unexpected_source.insert(
        "legacy.python.extension".into(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    let unexpected = registry.resolve(&unexpected_probe)?;
    assert!(matches!(
        unexpected.map_state_dictionary(&ModelStateTransaction::new(&backend, &context), DIGEST, &unexpected_source),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["legacy.python.extension"]
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &cancelled_context), DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_sd15_forward_patch_memory_dtype_device_and_ambiguity()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[sd15::MODEL_FAMILY_REGISTRATION])?;
    let probe = native_probe(false);
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let source = native_source(&backend, &context, DType::F32, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &source,
    )?;
    let model = build_model_family_for_probe(
        &registry,
        &probe,
        weights,
        options(DType::F32, 1_000_000),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, 49_180);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &model.forward_checkpoints(&backend, &input, &context)?,
        "epsilon_prediction",
        &[0.9934323, 0.6237125],
    )?;
    let patch = PatchGraph::checked(
        DIGEST,
        vec![PatchOperation {
            identifier: "sd15-timestep-delta".into(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.time_embed.0.weight".into(),
                expected_shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 0.0],
                application: PatchApplication::Add,
            }],
        }],
    )?;
    let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
    assert_checkpoint(
        &backend,
        &context,
        &patched.forward_checkpoints(&backend, &input, &context)?,
        "epsilon_prediction",
        &[0.9999032, 0.9934323],
    )?;

    for dtype in [DType::F16, DType::Bf16] {
        let source = native_source(&backend, &context, dtype, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
        )?;
        build_model_family_for_probe(&registry, &probe, weights, options(dtype, 1_000_000))?;
    }
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 49_179)),
        Err(ModelFamilyError::OutOfMemory { required: 49_180, .. })
    ));
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    let mut metal = options(DType::F32, 1_000_000);
    metal.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, metal),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { .. })
    ));
    Ok(())
}

fn native_probe(include_components: bool) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        ("model.diffusion_model.input_blocks.0.0.weight".into(), vec![320, 4, 3, 3]),
        ("model.diffusion_model.time_embed.0.weight".into(), vec![2, 2]),
        ("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight".into(), vec![2, 2]),
        ("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight".into(), vec![2, 768]),
        ("model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight".into(), vec![2, 2]),
        ("model.diffusion_model.out.2.weight".into(), vec![4, 320, 3, 3]),
    ]);
    if include_components {
        tensor_shapes.insert("cond_stage_model.transformer.embeddings.position_ids".into(), vec![3]);
        tensor_shapes.insert("first_stage_model.decoder.weight".into(), vec![1]);
    }
    ModelProbe { tensor_shapes, metadata: BTreeMap::new() }
}

fn diffusers_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("conv_in.weight".into(), vec![320, 4, 3, 3]),
            ("time_embedding.linear_1.weight".into(), vec![2, 2]),
            ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".into(), vec![2, 2]),
            ("down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight".into(), vec![2, 768]),
            ("mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight".into(), vec![2, 2]),
            ("conv_out.weight".into(), vec![4, 320, 3, 3]),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn native_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    include_components: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    source_for_probe(backend, context, dtype, native_probe(include_components), true)
}

fn diffusers_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    source_for_probe(backend, context, dtype, diffusers_probe(), false)
}

fn source_for_probe(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    probe: ModelProbe,
    native: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe.tensor_shapes.into_iter().map(|(key, shape)| {
        let count = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key.ends_with("position_ids") {
            vec![0.2, 1.6, 2.4]
        } else if key.ends_with("time_embed.0.weight") || key.ends_with("time_embedding.linear_1.weight") {
            vec![1.0, 0.0, 0.0, 1.0]
        } else if key.contains("attn1.to_q.weight") {
            vec![2.0, 0.0, 0.0, 0.5]
        } else if key.contains("middle_block.1") || key.contains("mid_block.attentions") {
            vec![1.0, 1.0, 1.0, -1.0]
        } else {
            vec![0.0; count]
        };
        let _ = native;
        Ok((key, tensor(backend, &shape, &values, dtype, context)?))
    }).collect()
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sd15.safetensors");
    write_safetensors(&path, &native_probe(false))?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical("sd15-row", "checkpoints", directory.path(), ["safetensors"])?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("sd15-row", "sd15.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path, probe: &ModelProbe) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, shape) in &probe.tensor_shapes {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |a, b| a.checked_mul(*b).ok_or("overflow"))?;
        data.resize(data.len() + usize::try_from(elements)?.checked_mul(4).ok_or("overflow")?, 0);
        header.insert(key.clone(), serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}));
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

fn assert_values(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    tensor: &Tensor,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = tensor_to_f32_with_context_exact_native(backend, tensor, context)?;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = &checkpoints.iter().find(|checkpoint| checkpoint.name == name).ok_or("checkpoint")?.tensor;
    assert_values(backend, context, tensor, expected)
}

fn verify_provenance_and_catalog() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.join("crates/sim_test_support/fixtures/models")
            .join(sd15::MODEL_FAMILY_FIXTURE)
            .join("provenance.json"),
    )?)?;
    assert_eq!(sha256(provenance["source_projection"].as_str().ok_or("projection")?.as_bytes()), sd15::MODEL_FAMILY_PROJECTION_SHA256);
    assert_eq!(provenance["source_projection_sha256"], sd15::MODEL_FAMILY_PROJECTION_SHA256);
    for source in provenance["source_files"].as_array().ok_or("sources")? {
        assert_eq!(sha256(&std::fs::read(root.join(source["path"].as_str().ok_or("path")?))?), source["sha256"]);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("crates/sim_model/catalog/model-families-v1.json"))?)?;
    let row = catalog["models"].as_array().ok_or("models")?.iter().find(|row| row["feature_id"] == sd15::MODEL_FAMILY_FEATURE_ID).ok_or("row")?;
    assert_eq!(row["source_ordinal"], 3);
    assert_eq!(row["static"]["unet_config"]["value"]["context_dim"], 768);
    assert_eq!(row["static"]["unet_config"]["value"]["model_channels"], 320);
    assert_eq!(row["clip_target"]["calls"][0]["tokenizer"], "sd1_clip.SD1Tokenizer");
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
