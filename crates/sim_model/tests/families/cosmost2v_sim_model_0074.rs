use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, CosmosArchitecture, CosmosRatio,
    ModelClipConfigurationFact, ModelClipModelInvocation, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_cosmost2v_sim_model_0074 as cosmos,
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

const ARTIFACT_DIGEST: &str = "0740740740740740740740740740740740740740740740740740740740740740";
const RESOLVED_MEMORY_BYTES: u64 = 8_368;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9074",
    identifier: "CosmosT2VAmbiguousFixture",
    ..cosmos::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    cosmos::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 141,
        source_architecture: "model_base.CosmosVideoAmbiguousFixture",
        ..cosmos::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_cosmost2v_source_projection_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(cosmos::MODEL_FAMILY_IDENTIFIER, "CosmosT2V");
    assert_eq!(cosmos::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0074");
    assert_eq!(cosmos::MODEL_FAMILY_FIXTURE, "cosmost2v-comfy-model-0074");
    assert_eq!(cosmos::MODEL_FAMILY_SOURCE_ORDINAL, 41);
    assert_eq!(cosmos::MODEL_FAMILY_REGISTRATION.source_ordinal, 41);
    assert_eq!(
        cosmos::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.CosmosVideo"
    );
    assert!(cosmos::MODEL_FAMILY_REGISTRATION.source_configuration.is_empty());
    assert_eq!(cosmos::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.6);
    assert_eq!(cosmos::MODEL_FAMILY_SIGMA_DATA, 0.5);
    assert_eq!(cosmos::MODEL_FAMILY_SIGMA_MAX, 80.0);
    assert_eq!(cosmos::MODEL_FAMILY_SIGMA_MIN, 0.002);

    let descriptor = describe_model_family(&cosmos::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "CosmosT2V");
    assert_eq!(descriptor.family, "COMFY-MODEL-0074");
    assert_eq!(descriptor.architecture_version, "cosmos-general-dit-t2v-v1");
    assert_eq!(descriptor.latent_format, "Cosmos1CV8x8x8");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);

    let seven_b =
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 16, 4_096, false, false, "native"))?;
    let configuration = cosmos::configuration_for_probe(&seven_b)?;
    assert_eq!(configuration.architecture, CosmosArchitecture::GeneralDit);
    assert_eq!(configuration.in_channels, 16);
    assert_eq!(configuration.out_channels, 16);
    assert_eq!(configuration.model_channels, 4_096);
    assert_eq!(configuration.number_of_blocks, 28);
    assert_eq!(configuration.number_of_heads, 32);
    assert_eq!(configuration.maximum_image_height, 240);
    assert_eq!(configuration.maximum_image_width, 240);
    assert_eq!(configuration.maximum_frames, 128);
    assert_eq!(configuration.spatial_patch_size, 2);
    assert_eq!(configuration.temporal_patch_size, 1);
    assert!(configuration.concatenate_padding_mask);
    assert!(!configuration.image_to_video);
    assert!(!configuration.positional_embeddings_learnable);
    assert_eq!(
        configuration.rope_extrapolation,
        [ratio(1, 1), ratio(1, 1), ratio(2, 1)]
    );
    assert_eq!(configuration.extra_extrapolation, None);
    assert_eq!(configuration.extra_per_block_absolute_position, Some(true));
    assert_eq!(configuration.cross_attention_embedding_channels, None);
    assert_eq!(configuration.minimum_frames_per_second, None);
    assert_eq!(configuration.maximum_frames_per_second, None);
    assert_eq!(configuration.adaln_lora_dimension, 256);
    assert_eq!(configuration.memory_usage_factor, 1.6);
    assert_eq!(configuration.model_size, cosmos::CosmosT2VModelSize::SevenB);

    let fourteen_b =
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 16, 5_120, false, false, "native"))?;
    let configuration = cosmos::configuration_for_probe(&fourteen_b)?;
    assert_eq!(configuration.number_of_blocks, 36);
    assert_eq!(configuration.number_of_heads, 40);
    assert_eq!(
        configuration.rope_extrapolation,
        [ratio(2, 1), ratio(2, 1), ratio(2, 1)]
    );
    assert_eq!(
        configuration.extra_extrapolation,
        Some([ratio(2, 1), ratio(2, 1), ratio(2, 1)])
    );
    assert_eq!(
        configuration.model_size,
        cosmos::CosmosT2VModelSize::FourteenB
    );

    let mut misleading_facts = parsed_facts(DType::F32, 16, 4_096, false, false, "native");
    misleading_facts.formats[0].metadata.extend([
        ("image_model".to_owned(), "cosmos_predict2".to_owned()),
        ("in_channels".to_owned(), "17".to_owned()),
        ("model_layout".to_owned(), "diffusers".to_owned()),
    ]);
    let misleading = ModelProbe::from_parsed_facts(misleading_facts)?;
    let registry =
        ModelFamilyRegistry::checked_registrations(&[cosmos::MODEL_FAMILY_REGISTRATION])?;
    let misleading_resolved = registry.resolve(&misleading)?;
    assert_eq!(
        misleading_resolved.detection().identity.feature_id(),
        cosmos::MODEL_FAMILY_FEATURE_ID
    );
    let misleading_configuration = cosmos::configuration_for_probe(&misleading)?;
    assert_eq!(misleading_configuration.architecture, CosmosArchitecture::GeneralDit);
    assert_eq!(misleading_configuration.in_channels, 16);
    assert!(!misleading_configuration.image_to_video);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(cosmos::MODEL_FAMILY_SOURCE_PATH)
        )?),
        cosmos::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], cosmos::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], cosmos::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 41);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("CosmosT2V source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("CosmosT2V source_files must be an array")?
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
        .find(|row| row["feature_id"] == cosmos::MODEL_FAMILY_FEATURE_ID)
        .ok_or("CosmosT2V catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 41);
    assert_eq!(row["static"]["unet_config"]["value"]["in_channels"], 16);
    assert_eq!(
        sha256(&serde_json::to_vec(row)?),
        cosmos::MODEL_FAMILY_PROJECTION_SHA256
    );
    assert_eq!(
        provenance["catalog_projection_sha256"],
        cosmos::MODEL_FAMILY_PROJECTION_SHA256
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/cosmost2v_sim_model_0074.rs"),
    )?;
    for owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct ModelProbe",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct MemoryEstimator",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(owner));
    }
    assert!(row_source.contains("cosmos_configuration_for_probe("));
    assert!(!row_source.contains("fn required_shape("));
    assert!(!row_source.contains("ModelDetectionRule::Metadata"));
    assert!(!row_source.contains("ModelSourceConfigurationRule"));
    assert!(row_source.contains("ModelDetectionRule::AnyKeyPresent"));
    assert!(row_source.contains("ModelDetectionRule::AnyTensorDimensionValue"));
    assert!(row_source.contains("source_configuration: &[]"));
    super::write_model_family_row_artifact(
        cosmos::MODEL_FAMILY_FIXTURE,
        cosmos::MODEL_FAMILY_FEATURE_ID,
        cosmos::MODEL_FAMILY_IDENTIFIER,
        cosmos::MODEL_FAMILY_SOURCE_ORDINAL,
        "cosmost2v_comfy_model_0074",
        &[
            "source-provenance-catalog-and-ownership",
            "source-exact-seven-b-and-fourteen-b-profiles",
            "model-store-net-prefix-detection",
            "transactional-component-mapping-and-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "i2v-predict2-partial-ambiguous-and-layout-rejection",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_cosmost2v_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[cosmos::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "net.");
    assert!(!probe.metadata.contains_key("image_model"));
    assert!(!probe.metadata.contains_key("in_channels"));
    assert!(!probe.metadata.contains_key("model_layout"));
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0074"
    );
    assert_eq!(resolved.source_ordinal(), 41);
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
    assert_eq!(resolved.profile().latent_identifier, "Cosmos1CV8x8x8");

    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.cosmos.CosmosT5Tokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.cosmos.te"
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
    let source = source_tensors(&backend, &context, DType::F32, false)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("model").is_some_and(|model| {
        model.contains_key("native.blocks.block0.blocks.0.block.attn.to_q.0.weight")
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
        options(DType::F32, RESOLVED_MEMORY_BYTES),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, RESOLVED_MEMORY_BYTES);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "transformer_block_projection",
        &[1.4621172, 0.8807971],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "video_output",
        &[0.0, 0.96402675],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "cosmost2v-transformer-block-delta".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.blocks.block0.blocks.2.block.layer2.weight".to_owned(),
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
        "video_output",
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
            &first.tensors()["native.blocks.block0.blocks.2.block.layer2.weight"],
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            &second.tensors()["native.blocks.block0.blocks.2.block.layer2.weight"],
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
fn val_model_family_row_001_cosmost2v_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[cosmos::MODEL_FAMILY_REGISTRATION])?;
    let probe =
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 16, 4_096, false, false, "native"))?;
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );

    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let source = source_tensors(&backend, &context, dtype, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(dtype, RESOLVED_MEMORY_BYTES),
        )?;
    }

    let source = source_tensors(&backend, &context, DType::F32, false)?;
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
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 16, 4_096, true, false, "native"))?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let partial = source_tensors(&backend, &context, DType::F32, true)?;
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.final_layer.linear.weight"
    ));

    let i2v_profile =
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 17, 4_096, false, false, "native"))?;
    assert!(matches!(
        registry.detect(&i2v_profile),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let mut mismatched_facts = parsed_facts(DType::F32, 17, 4_096, false, false, "native");
    mismatched_facts.formats[0]
        .metadata
        .insert("in_channels".to_owned(), "16".to_owned());
    let mismatch = ModelProbe::from_parsed_facts(mismatched_facts)?;
    assert!(matches!(
        registry.resolve(&mismatch),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let malformed =
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 16, 2_048, false, false, "native"))?;
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("expected 4096 or 5120")
    ));
    let predict2 =
        ModelProbe::from_parsed_facts(parsed_facts(DType::F32, 16, 4_096, false, true, "native"))?;
    assert!(matches!(
        registry.resolve(&predict2),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("Predict2 marker")
    ));
    let spoofed_layout = ModelProbe::from_parsed_facts(parsed_facts(
        DType::F32,
        16,
        4_096,
        false,
        false,
        "diffusers",
    ))?;
    assert!(registry.resolve(&spoofed_layout).is_ok());

    let mut unexpected_facts = parsed_facts(DType::F32, 16, 4_096, false, false, "native");
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected = source_tensors(&backend, &context, DType::F32, false)?;
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

    let no_match = ModelProbe {
        tensor_shapes: BTreeMap::new(),
        metadata: BTreeMap::from([("image_model".to_owned(), "cosmos_predict2".to_owned())]),
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
    dtype: DType,
    in_channels: u64,
    model_channels: u64,
    omit_final: bool,
    predict2: bool,
    layout: &str,
) -> ModelParsedFacts {
    let mut tensors = BTreeMap::new();
    for (key, shape) in model_shapes(in_channels, model_channels, omit_final, predict2) {
        tensors.insert(
            format!("net.{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
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
    let mut metadata = BTreeMap::new();
    if layout != "native" {
        metadata.insert("model_layout".to_owned(), layout.to_owned());
    }
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata,
        }],
    }
}

fn model_shapes(
    in_channels: u64,
    model_channels: u64,
    omit_final: bool,
    predict2: bool,
) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "x_embedder.proj.1.weight".to_owned(),
            vec![1, (in_channels + 1) * 4],
        ),
        (
            "blocks.block0.blocks.0.block.attn.to_q.0.weight".to_owned(),
            vec![model_channels, 1],
        ),
        ("t_embedder.1.linear_1.weight".to_owned(), vec![2, 2]),
        ("t_embedder.1.linear_2.weight".to_owned(), vec![2, 2]),
        (
            "blocks.block0.blocks.2.block.layer2.weight".to_owned(),
            vec![2, 2],
        ),
        ("affline_norm.weight".to_owned(), vec![2]),
    ];
    if !omit_final {
        shapes.push(("final_layer.linear.weight".to_owned(), vec![2, 2]));
    }
    if predict2 {
        shapes.push(("blocks.0.mlp.layer1.weight".to_owned(), vec![2, 2]));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    omit_final: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(16, 4_096, omit_final, false) {
        let values = match key.as_str() {
            "t_embedder.1.linear_1.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "blocks.block0.blocks.2.block.layer2.weight" => vec![2.0, 0.0, 0.0, 0.5],
            "final_layer.linear.weight" => vec![1.0, 1.0, 1.0, -1.0],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
        };
        source.insert(
            format!("net.{key}"),
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
    let model_path = directory.path().join("cosmost2v.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "cosmost2v-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("cosmost2v-row", "cosmost2v.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(16, 4_096, false, false)
        .into_iter()
        .map(|(key, shape)| (format!("net.{key}"), shape))
        .collect::<Vec<_>>();
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.t5xxl.transformer.weight".to_owned(), vec![1]),
    ]);
    let mut data = Vec::new();
    for (key, shape) in shapes {
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
            key,
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
        .ok_or("CosmosT2V checkpoint is missing")?;
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
        identifier: "cosmost2v-ordered-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.blocks.block0.blocks.2.block.layer2.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "cosmost2v-ordered-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.blocks.block0.blocks.2.block.layer2.weight".to_owned(),
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
        .join(cosmos::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

const fn ratio(numerator: u64, denominator: u64) -> CosmosRatio {
    CosmosRatio {
        numerator,
        denominator,
    }
}
