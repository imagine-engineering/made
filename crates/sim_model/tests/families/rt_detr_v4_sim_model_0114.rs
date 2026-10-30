use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_rt_detr_v4_sim_model_0114 as rt_detr,
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

const DIGEST: &str = "0114011401140114011401140114011401140114011401140114011401140114";
const MEMORY_BYTES: u64 = 48;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9114",
    identifier: "RT_DETR_v4_AmbiguousFixture",
    ..rt_detr::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    rt_detr::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 185,
        source_architecture: "model_base.RT_DETR_v4_AmbiguousFixture",
        ..rt_detr::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_rt_detr_v4_source_configuration_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(rt_detr::MODEL_FAMILY_IDENTIFIER, "RT_DETR_v4");
    assert_eq!(rt_detr::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0114");
    assert_eq!(rt_detr::MODEL_FAMILY_SOURCE_ORDINAL, 85);
    assert_eq!(rt_detr::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.RT_DETR_v4");
    assert_eq!(rt_detr::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);

    let descriptor = describe_model_family(&rt_detr::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "LatentFormat");
    assert_eq!(descriptor.supported_dtypes, ["float16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);

    for (layout, expected) in [
        ("prefixed", rt_detr::RtDetrV4Layout::PrefixedNative),
        ("standalone", rt_detr::RtDetrV4Layout::StandaloneNative),
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(layout, DType::F32))?;
        let configuration = rt_detr::configuration_for_probe(&probe)?;
        assert_eq!(configuration.layout, expected);
        assert_eq!(configuration.encoder_hidden_size, 2);
        assert_eq!(configuration.input_channels, 2);
        assert_eq!(configuration.class_count, 2);
        assert_eq!(configuration.decoder_hidden_size, 2);
        assert_eq!(configuration.query_position_dimensions, 4);
        assert_eq!(configuration.decoder_layer_count, 1);
        assert_eq!(configuration.query_count, 300);
        assert_eq!(configuration.feature_strides, [8, 16, 32]);
    }

    let registry = ModelFamilyRegistry::checked_registrations(&[rt_detr::MODEL_FAMILY_REGISTRATION])?;
    let clip_probe = ModelProbe::from_parsed_facts(parsed_facts("standalone", DType::F32))?;
    assert!(registry.resolve(&clip_probe)?.clip_target().candidates().is_empty());
    let mut misleading = parsed_facts("standalone", DType::F32);
    misleading.formats[0].metadata.insert("image_model".into(), "SAM3".into());
    assert_eq!(
        registry.detect(&ModelProbe::from_parsed_facts(misleading)?)?.identity.feature_id(),
        rt_detr::MODEL_FAMILY_FEATURE_ID
    );
    let mut partial = parsed_facts("standalone", DType::F32);
    partial.tensors.remove("decoder.query_pos_head.layers.0.weight");
    assert!(matches!(
        registry.detect(&ModelProbe::from_parsed_facts(partial)?),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let mut malformed = parsed_facts("standalone", DType::F32);
    malformed.tensors.get_mut("encoder.pan_blocks.1.cv4.conv.weight").ok_or("encoder")?.shape = vec![2, 2];
    assert!(matches!(
        rt_detr::configuration_for_probe(&ModelProbe::from_parsed_facts(malformed)?),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("convolution shape")
    ));
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("transformer.encoder.pan_blocks.1.cv4.conv.weight".into(), vec![2, 2, 1, 1]),
            ("transformer.decoder.enc_score_head.weight".into(), vec![2, 2]),
            ("transformer.decoder.query_pos_head.layers.0.weight".into(), vec![2, 4]),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        rt_detr::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));

    verify_provenance_and_catalog()?;
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/rt_detr_v4_sim_model_0114.rs"),
    )?;
    for forbidden in ["struct Tensor", "struct ModelStore", "struct ModelProbe", "struct PatchGraph", "std::fs", "unsafe "] {
        assert!(!source.contains(forbidden));
    }

    super::write_model_family_row_artifact(
        rt_detr::MODEL_FAMILY_FIXTURE,
        rt_detr::MODEL_FAMILY_FEATURE_ID,
        rt_detr::MODEL_FAMILY_IDENTIFIER,
        rt_detr::MODEL_FAMILY_SOURCE_ORDINAL,
        "rt_detr_v4_comfy_model_0114",
        &[
            "source-provenance-and-ownership",
            "prefixed-and-standalone-native-configuration",
            "diffusers-fail-closed",
            "persisted-model-store-probe",
            "transactional-mapping-forward-and-patching",
            "dtype-device-cancellation-oom-and-invalid-inputs",
            "deterministic-decoder-checkpoints",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_rt_detr_v4_store_mapping_forward_patch_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[rt_detr::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    for layout in ["prefixed", "standalone"] {
        let facts = parsed_facts(layout, DType::F32);
        let layout_probe = ModelProbe::from_parsed_facts(facts)?;
        let layout_resolved = registry.resolve(&layout_probe)?;
        let source = source_tensors(&backend, &context, layout, DType::F32)?;
        let mapped = layout_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
        )?;
        assert!(mapped.component("model").ok_or("model")?.contains_key("native.decoder.enc_score_head.weight"));
    }

    let source = source_tensors(&backend, &context, "standalone", DType::F32)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    let model = build_model_family_for_probe(
        &registry, &probe, weights, options(DType::F32, MEMORY_BYTES),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, MEMORY_BYTES);
    let input = tensor(&backend, &[1, 2], &[0.25, -0.5], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(&backend, &context, &checkpoints, "decoder_class_logits", &[0.35, -0.7])?;
    assert_checkpoint(&backend, &context, &checkpoints, "decoder_probabilities", &[0.33637553, -0.6043678])?;

    let patch = PatchGraph::checked(DIGEST, vec![PatchOperation {
        identifier: "rt-detr-score-bias".into(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.decoder.enc_score_head.bias".into(),
            expected_shape: vec![2],
            values: vec![0.5, 0.5],
            application: PatchApplication::Add,
        }],
    }])?;
    let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
    assert_checkpoint(
        &backend, &context,
        &patched.forward_checkpoints(&backend, &input, &context)?,
        "decoder_probabilities", &[0.6910695, -0.19737533],
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, MEMORY_BYTES - 1)),
        Err(ModelFamilyError::OutOfMemory { required: MEMORY_BYTES, budget }) if budget == MEMORY_BYTES - 1
    ));
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::Bf16, MEMORY_BYTES)),
        Err(ModelFamilyError::UnsupportedDType(DType::Bf16))
    ));
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    let mut metal = options(DType::F32, MEMORY_BYTES);
    metal.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, metal),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut unexpected = source.clone();
    unexpected.insert("diffusers.transformer.weight".into(), tensor(&backend, &[1], &[1.0], DType::F32, &context)?);
    let mut unexpected_facts = parsed_facts("standalone", DType::F32);
    unexpected_facts.tensors.insert(
        "diffusers.transformer.weight".into(),
        ModelParsedTensorFact { shape: vec![1], storage_dtype: DType::F32.catalog_name().into() },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let unexpected_result = unexpected_resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &unexpected,
    );
    assert!(
        matches!(
            unexpected_result,
            Err(ModelFamilyError::UnexpectedKeys(ref keys)) if keys == &["diffusers.transformer.weight"]
        ),
        "unexpected result: {unexpected_result:?}"
    );
    let mut partial = source.clone();
    partial.remove("decoder.enc_score_head.bias");
    let mut partial_facts = parsed_facts("standalone", DType::F32);
    partial_facts.tensors.remove("decoder.enc_score_head.bias");
    let partial_probe = ModelProbe::from_parsed_facts(partial_facts)?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    assert!(matches!(
        partial_resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &context), DIGEST, &partial),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.decoder.enc_score_head.bias"
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT, authority.authorize_workspace(2 * 1024 * 1024)?, &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &cancelled_context), DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(ambiguous.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { .. })));
    Ok(())
}

fn parsed_facts(layout: &str, dtype: DType) -> ModelParsedFacts {
    let prefix = if layout == "prefixed" { "model.diffusion_model." } else { "" };
    let tensors = model_shapes().into_iter().map(|(key, shape)| (
        format!("{prefix}{key}"),
        ModelParsedTensorFact { shape, storage_dtype: dtype.catalog_name().into() },
    )).collect();
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact { identity: "safetensors".into(), metadata: BTreeMap::new() }],
    }
}

fn model_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("encoder.pan_blocks.1.cv4.conv.weight".into(), vec![2, 2, 1, 1]),
        ("decoder.enc_score_head.weight".into(), vec![2, 2]),
        ("decoder.enc_score_head.bias".into(), vec![2]),
        ("decoder.query_pos_head.layers.0.weight".into(), vec![2, 4]),
        ("decoder.decoder.layers.0.self_attn.q_proj.weight".into(), vec![2, 2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend, context: &ExecutionContext<'_>, layout: &str, dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "prefixed" { "model.diffusion_model." } else { "" };
    let values = [
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.1, -0.2],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ];
    model_shapes().into_iter().zip(values).map(|((key, shape), values)| {
        Ok((format!("{prefix}{key}"), tensor(backend, &shape, &values, dtype, context)?))
    }).collect()
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("rt-detr-v4.safetensors");
    write_safetensors(&path, parsed_facts("standalone", DType::F32))?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical("rt-detr-row", "checkpoints", directory.path(), ["safetensors"])?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("rt-detr-row", "rt-detr-v4.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path, facts: ModelParsedFacts) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, tensor) in facts.tensors {
        let start = data.len();
        let elements = tensor.shape.iter().try_fold(1_u64, |a, b| a.checked_mul(*b).ok_or("overflow"))?;
        for _ in 0..elements { data.extend_from_slice(&0.0_f32.to_le_bytes()); }
        header.insert(key, serde_json::json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[start,data.len()]}));
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn tensor(
    backend: &CpuBackend, shape: &[u64], values: &[f32], dtype: DType, context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(tensor_from_f32_with_context_exact_native(backend, shape, values, dtype, backend.device(), context)?)
}

fn options(dtype: DType, memory_budget_bytes: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype, device: DeviceKind::Cpu, activation_elements: 2, memory_budget_bytes,
        allow_unexpected_weights: false,
    }
}

fn assert_checkpoint(
    backend: &CpuBackend, context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint], name: &str, expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoints.iter().find(|checkpoint| checkpoint.name == name).ok_or("checkpoint")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) { assert!((actual - expected).abs() <= 1.0e-5); }
    Ok(())
}

fn verify_provenance_and_catalog() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("crates/sim_test_support/fixtures/models").join(rt_detr::MODEL_FAMILY_FIXTURE).join("provenance.json");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let projection = provenance["source_projection"].as_str().ok_or("projection")?;
    assert_eq!(sha256(projection.as_bytes()), rt_detr::MODEL_FAMILY_PROJECTION_SHA256);
    for source in provenance["source_files"].as_array().ok_or("sources")? {
        assert_eq!(sha256(&std::fs::read(root.join(source["path"].as_str().ok_or("path")?))?), source["sha256"]);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("crates/sim_model/catalog/model-families-v1.json"))?)?;
    let row = catalog["models"].as_array().ok_or("models")?.iter().find(|row| row["feature_id"] == rt_detr::MODEL_FAMILY_FEATURE_ID).ok_or("row")?;
    assert_eq!(row["source_ordinal"], 85);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "RT_DETR_v4");
    assert_eq!(row["static"]["supported_inference_dtypes"]["value"].as_array().ok_or("dtypes")?.len(), 2);
    assert_eq!(row["clip_target"]["calls"].as_array().ok_or("clip")?.len(), 0);
    Ok(())
}

fn sha256(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
