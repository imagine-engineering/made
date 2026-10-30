use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, HIDREAM_O1_ARCHITECTURE_VERSION,
    HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT, HIDREAM_O1_LATENT_FORMAT, HIDREAM_O1_PATCH_SIZE,
    HIDREAM_O1_PIXEL_VAE_SENTINEL, HIDREAM_O1_TEXT_ENCODER_SENTINEL,
    HIDREAM_O1_UNPREFIXED_STATE_PLAN, HiDreamO1Configuration, HiDreamO1Layout,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits,
    PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_hidreamo1_sim_model_0083 as hidream_o1,
    hidream_o1_configuration_for_probe,
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

const ARTIFACT_DIGEST: &str = "0083008300830083008300830083008300830083008300830083008300830083";
const EXPECTED_MEMORY_BYTES: u64 = 8_388_664;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9083",
    identifier: "HiDreamO1_AmbiguousFixture",
    ..hidream_o1::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    hidream_o1::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 170,
        source_architecture: "model_base.HiDreamO1AmbiguousFixture",
        ..hidream_o1::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_hidreamo1_source_configuration_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(hidream_o1::MODEL_FAMILY_IDENTIFIER, "HiDreamO1");
    assert_eq!(hidream_o1::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0083");
    assert_eq!(hidream_o1::MODEL_FAMILY_SOURCE_ORDINAL, 70);
    assert_eq!(
        hidream_o1::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.HiDreamO1"
    );
    assert!(
        hidream_o1::MODEL_FAMILY_REGISTRATION
            .source_configuration
            .is_empty()
    );
    assert_eq!(hidream_o1::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.033);
    assert_eq!(hidream_o1::MODEL_FAMILY_SAMPLING_SHIFT, 3.0);
    assert_eq!(hidream_o1::MODEL_FAMILY_NOISE_SCALE, 8.0);

    let descriptor = describe_model_family(&hidream_o1::MODEL_FAMILY)?;
    assert_eq!(descriptor.architecture_version, HIDREAM_O1_ARCHITECTURE_VERSION);
    assert_eq!(descriptor.latent_format, "HiDreamO1Pixel");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 2);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 2);
    assert_eq!(HIDREAM_O1_LATENT_FORMAT.channels, 3);
    assert_eq!(HIDREAM_O1_LATENT_FORMAT.spatial_downscale_ratio, 1);

    let registry =
        ModelFamilyRegistry::checked_registrations(&[hidream_o1::MODEL_FAMILY_REGISTRATION])?;
    for layout in ["native", "standalone"] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(layout, DType::F32))?;
        let resolved = registry.resolve(&probe)?;
        assert_eq!(
            resolved.detection().identity.feature_id(),
            hidream_o1::MODEL_FAMILY_FEATURE_ID
        );
        assert_eq!(resolved.detection().score, 1_000);
        assert_eq!(resolved.detection().evidence.len(), 4);
        assert!(
            resolved
                .detection()
                .evidence
                .iter()
                .all(|evidence| evidence.contains("AnyTensorDimensionValue"))
        );
        assert_eq!(resolved.profile().latent_identifier, "HiDreamO1Pixel");
        let configuration = hidream_o1_configuration_for_probe(&probe)?;
        assert_eq!(
            configuration.layout,
            if layout == "native" {
                HiDreamO1Layout::Native
            } else {
                HiDreamO1Layout::Unprefixed
            }
        );
        assert_source_configuration(configuration);
    }

    let mut misleading = parsed_facts("native", DType::F32);
    misleading.formats[0]
        .metadata
        .insert("image_model".to_owned(), "hidream".to_owned());
    misleading.formats[0]
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    let misleading = ModelProbe::from_parsed_facts(misleading)?;
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        hidream_o1::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(
        hidream_o1_configuration_for_probe(&misleading)?.layout,
        HiDreamO1Layout::Native
    );

    let mut partial = parsed_facts("native", DType::F32);
    partial
        .tensors
        .remove("model.diffusion_model.x_embedder.proj1.weight");
    assert!(matches!(
        registry.detect(&ModelProbe::from_parsed_facts(partial)?),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut malformed = parsed_facts("native", DType::F32);
    malformed
        .tensors
        .get_mut("model.diffusion_model.t_embedder1.mlp.0.weight")
        .ok_or("HiDreamO1 timestep projection must exist")?
        .shape = vec![4_096];
    let malformed = ModelProbe::from_parsed_facts(malformed)?;
    assert!(matches!(
        registry.detect(&malformed),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    assert!(matches!(
        hidream_o1_configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("t_embedder1.mlp.0.weight shape")
    ));

    let diffusers_probe = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("transformer_blocks.0.attn.to_q.weight".to_owned(), vec![4_096, 4_096]),
            ("proj_in.weight".to_owned(), vec![1_024, 3_072]),
        ]),
        metadata: BTreeMap::from([("image_model".to_owned(), "hidream_o1".to_owned())]),
    };
    assert!(matches!(
        registry.detect(&diffusers_probe),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let store_probe = probe_through_model_store()?;
    for key in ["image_model", "model_layout", "guidance", "in_channels"] {
        assert!(!store_probe.metadata.contains_key(key));
    }
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        hidream_o1::MODEL_FAMILY_FEATURE_ID
    );

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("HiDreamO1 source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    assert_eq!(
        hidream_o1::MODEL_FAMILY_PROJECTION_SHA256,
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("HiDreamO1 source_files must be an array")?
    {
        let path = source["path"].as_str().ok_or("source path must be a string")?;
        let digest = source["sha256"]
            .as_str()
            .ok_or("source digest must be a string")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), digest);
    }

    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog must contain models")?
        .iter()
        .find(|row| row["feature_id"] == hidream_o1::MODEL_FAMILY_FEATURE_ID)
        .ok_or("HiDreamO1 catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 70);
    assert_eq!(
        row["static"]["unet_config"]["value"]["image_model"],
        "hidream_o1"
    );
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 0.033);
    assert_eq!(
        row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.HiDreamO1Pixel"
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/hidreamo1_sim_model_0083.rs"),
    )?;
    for canonical_import in [
        "HIDREAM_O1_ARCHITECTURE_VERSION",
        "HIDREAM_O1_CLIP_TARGET",
        "HIDREAM_O1_COMPONENT_STATE_SCHEMAS",
        "HIDREAM_O1_LAYOUT_SIGNATURES",
        "HIDREAM_O1_STATE_PLAN_CASES",
        "hidream_o1_configuration_for_probe",
    ] {
        assert!(row_source.contains(canonical_import));
    }
    for forbidden in [
        "ModelDetectionRule::Metadata",
        "ModelSourceConfigurationRule",
        "HIDREAM_O1_NATIVE_STATE_PLAN",
        "HIDREAM_O1_UNPREFIXED_STATE_PLAN",
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
        assert!(!row_source.contains(forbidden));
    }

    super::write_model_family_row_artifact(
        hidream_o1::MODEL_FAMILY_FIXTURE,
        hidream_o1::MODEL_FAMILY_FEATURE_ID,
        hidream_o1::MODEL_FAMILY_IDENTIFIER,
        hidream_o1::MODEL_FAMILY_SOURCE_ORDINAL,
        "hidreamo1_comfy_model_0083",
        &[
            "source-provenance-registration-and-ownership",
            "key-derived-native-and-standalone-layouts",
            "canonical-hidream-o1-configuration-latent-and-clip",
            "misleading-metadata-partial-malformed-and-diffusers-rejection",
            "transactional-state-mapping-and-source-sentinels",
            "named-forward-patch-memory-and-oom",
            "bf16-f32-cpu-and-fail-closed-dtype-device",
            "ambiguity-cross-family-unexpected-and-cancellation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_hidreamo1_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[hidream_o1::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.source_ordinal(), 70);
    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.hidream_o1.HiDreamO1Tokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.hidream_o1.HiDreamO1TE"
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(48 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&backend, &context, "native", DType::F32, None)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    let model_state = mapped.component("model").ok_or("missing HiDreamO1 model state")?;
    assert_eq!(model_state.len(), 10);
    assert!(
        model_state
            .keys()
            .all(|key| !key.contains(HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT))
    );
    assert_sentinel(
        &backend,
        &context,
        &mapped,
        "vae",
        HIDREAM_O1_PIXEL_VAE_SENTINEL,
        1.0,
    )?;
    assert_sentinel(
        &backend,
        &context,
        &mapped,
        "text_encoder",
        HIDREAM_O1_TEXT_ENCODER_SENTINEL,
        0.0,
    )?;

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
    )
    .map_err(|error| format!("HiDreamO1 build failed: {error}"))?;
    assert_eq!(model.memory_estimate().total_bytes, EXPECTED_MEMORY_BYTES);
    for key in [
        "native.t_embedder1.mlp.2.weight",
        "native.x_embedder.proj2.weight",
        "native.visual.patch_embed.proj.weight",
        "native.language_model.layers.0.self_attn.q_proj.weight",
        "native.final_layer2.linear.weight",
    ] {
        assert_eq!(
            model
                .weights()
                .tensors()
                .get(key)
                .ok_or_else(|| format!("missing reduced forward weight {key}"))?
                .descriptor()
                .shape(),
            &[2, 2],
            "{key}"
        );
    }
    for key in [
        "native.t_embedder1.mlp.2.bias",
        "native.x_embedder.proj2.bias",
        "native.final_layer2.linear.bias",
    ] {
        assert_eq!(
            model
                .weights()
                .tensors()
                .get(key)
                .ok_or_else(|| format!("missing reduced forward bias {key}"))?
                .descriptor()
                .shape(),
            &[2],
            "{key}"
        );
    }
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model
        .forward_checkpoints(&backend, &input, &context)
        .map_err(|error| format!("HiDreamO1 forward failed: {error}"))?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "language_model_projection",
        &[0.9810586, -0.5189414],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "pixel_output",
        &[0.8004987, -0.8004987],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "hidream-o1-pixel-output-bias".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.final_layer2.linear.bias".to_owned(),
                expected_shape: vec![2],
                values: vec![0.5, 0.5],
                application: PatchApplication::Add,
            }],
        }],
    )?;
    let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
    assert_checkpoint(
        &backend,
        &context,
        &patched.forward_checkpoints(&backend, &input, &context)?,
        "pixel_output",
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
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == EXPECTED_MEMORY_BYTES && budget == EXPECTED_MEMORY_BYTES - 1
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_hidreamo1_standalone_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[hidream_o1::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(48 * 1024 * 1024)?,
        &cancellation,
    );

    for dtype in [DType::Bf16, DType::F32] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts("standalone", dtype))?;
        let resolved = registry.resolve(&probe)?;
        let source = source_tensors(&backend, &context, "standalone", dtype, None)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(dtype, EXPECTED_MEMORY_BYTES),
        )?;
    }

    let probe = ModelProbe::from_parsed_facts(parsed_facts("standalone", DType::Bf16))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, "standalone", DType::Bf16, None)?;
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
            options(DType::F16, EXPECTED_MEMORY_BYTES),
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported_device = options(DType::Bf16, EXPECTED_MEMORY_BYTES);
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let partial = source_tensors(
        &backend,
        &context,
        "standalone",
        DType::Bf16,
        Some("x_embedder.proj1.weight"),
    )?;
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(message))
            if message.contains("key count changed")
    ));

    let mut unexpected =
        source_tensors(&backend, &context, "standalone", DType::Bf16, None)?;
    unexpected.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::Bf16, &context)?,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &unexpected,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(message))
            if message.contains("key count changed")
    ));
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &HIDREAM_O1_UNPREFIXED_STATE_PLAN.compile()?,
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

    let mut mixed = parsed_facts("native", DType::F32);
    mixed
        .tensors
        .extend(parsed_facts("standalone", DType::F32).tensors);
    assert!(matches!(
        registry.resolve(&ModelProbe::from_parsed_facts(mixed)?),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));

    let cross_family = ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "model.diffusion_model.caption_projection.0.linear.weight".to_owned(),
                vec![2_560, 4_096],
            ),
            (
                "model.diffusion_model.double_stream_blocks.0.block.attn1.to_out.weight"
                    .to_owned(),
                vec![2_560, 2_560],
            ),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        registry.detect(&cross_family),
        Err(ModelFamilyError::NoDetectionMatch)
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

fn assert_source_configuration(configuration: HiDreamO1Configuration) {
    assert_eq!(configuration.patch_size, HIDREAM_O1_PATCH_SIZE);
    assert_eq!(configuration.input_channels, 3);
    assert_eq!(configuration.patch_dimension, 3_072);
    assert_eq!(configuration.bottleneck_dimension, 1_024);
    assert_eq!(configuration.hidden_size, 4_096);
    assert_eq!(configuration.intermediate_size, 12_288);
    assert_eq!(configuration.hidden_layer_count, 36);
    assert_eq!(configuration.attention_head_count, 32);
    assert_eq!(configuration.key_value_head_count, 8);
    assert_eq!(configuration.attention_head_dimension, 128);
    assert_eq!(configuration.maximum_position_embeddings, 128_000);
    assert_eq!(configuration.vision_hidden_size, 1_152);
    assert_eq!(configuration.vision_intermediate_size, 4_304);
    assert_eq!(configuration.vision_depth, 27);
    assert_eq!(configuration.vision_head_count, 16);
    assert_eq!(configuration.vision_position_embedding_count, 2_304);
}

fn parsed_facts(layout: &str, dtype: DType) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in probe_shapes() {
        tensors.insert(
            format!("{prefix}{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for key in [
        "vae.decoder.weight",
        "text_encoders.hidream_o1.transformer.weight",
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

fn probe_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("t_embedder1.mlp.0.weight".to_owned(), vec![4_096, 256]),
        ("t_embedder1.mlp.2.weight".to_owned(), vec![2, 2]),
        ("t_embedder1.mlp.2.bias".to_owned(), vec![2]),
        ("x_embedder.proj1.weight".to_owned(), vec![1_024, 3_072]),
        ("x_embedder.proj2.weight".to_owned(), vec![2, 2]),
        ("x_embedder.proj2.bias".to_owned(), vec![2]),
        ("visual.patch_embed.proj.weight".to_owned(), vec![2, 2]),
        (
            "language_model.layers.0.self_attn.q_proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("final_layer2.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer2.linear.bias".to_owned(), vec![2]),
        (deepstack_weight_key(), vec![1]),
    ]
}

fn source_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("t_embedder1.mlp.0.weight".to_owned(), vec![4_096, 256]),
        ("t_embedder1.mlp.2.weight".to_owned(), vec![2, 2]),
        ("t_embedder1.mlp.2.bias".to_owned(), vec![2]),
        ("x_embedder.proj1.weight".to_owned(), vec![1_024, 3_072]),
        ("x_embedder.proj2.weight".to_owned(), vec![2, 2]),
        ("x_embedder.proj2.bias".to_owned(), vec![2]),
        ("visual.patch_embed.proj.weight".to_owned(), vec![2, 2]),
        (
            "language_model.layers.0.self_attn.q_proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("final_layer2.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer2.linear.bias".to_owned(), vec![2]),
        (deepstack_weight_key(), vec![1]),
    ]
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omitted: Option<&str>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in source_shapes() {
        if omitted == Some(key.as_str()) {
            continue;
        }
        let values = match key.as_str() {
            "t_embedder1.mlp.0.weight" | "x_embedder.proj1.weight" => {
                vec![0.0; usize::try_from(shape.iter().product::<u64>())?]
            }
            "t_embedder1.mlp.2.weight"
            | "x_embedder.proj2.weight"
            | "visual.patch_embed.proj.weight"
            | "language_model.layers.0.self_attn.q_proj.weight"
            | "final_layer2.linear.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "t_embedder1.mlp.2.bias" => vec![1.0, -1.0],
            "x_embedder.proj2.bias" => vec![0.25, -0.25],
            "final_layer2.linear.bias" => vec![0.1, -0.1],
            key if key == deepstack_weight_key() => vec![9.0],
            _ => return Err(format!("missing source values for {key}").into()),
        };
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, &shape, &values, dtype, context)?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, &[1], &[7.0], dtype, context)?,
    );
    source.insert(
        "text_encoders.hidream_o1.transformer.weight".to_owned(),
        tensor(backend, &[1], &[8.0], dtype, context)?,
    );
    Ok(source)
}

fn deepstack_weight_key() -> String {
    format!("visual.{}", "deepstack_merger_list.0.weight")
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("hidream-o1.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "hidream-o1-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("hidream-o1-row", "hidream-o1.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, tensor) in parsed_facts("native", DType::F32).tensors {
        let shape = tensor.shape;
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or("fixture shape overflow")
        })?;
        let bytes = usize::try_from(elements.checked_mul(4).ok_or("fixture byte overflow")?)?;
        data.resize(data.len().checked_add(bytes).ok_or("fixture data overflow")?, 0);
        header.insert(
            key,
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()]
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
        .ok_or("HiDreamO1 checkpoint is missing")?;
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

fn assert_sentinel(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    mapped: &sim_model::MappedModelComponents,
    component: &str,
    key: &str,
    expected: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let component = mapped.component(component).ok_or("missing sentinel component")?;
    assert_eq!(component.len(), 1);
    let tensor = component.get(key).ok_or("missing sentinel tensor")?;
    assert_eq!(tensor.descriptor().shape(), &[1]);
    assert_eq!(tensor.descriptor().dtype(), DType::F32);
    assert_eq!(
        &*tensor_to_f32_with_context_exact_native(backend, tensor, context)?,
        &[expected]
    );
    Ok(())
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(hidream_o1::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
