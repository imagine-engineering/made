use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipConfigurationFact,
    ModelClipModelInvocation, ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration,
    ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, MappedModelWeights, NativeFamilyBuildOptions, ParserLimits,
    PatchApplication, Omnigen2BooguLayout, Omnigen2BooguVariant, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family, describe_model_family,
    generated_boogu_sim_model_0065 as boogu,
    map_model_weights,
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

const ARTIFACT_DIGEST: &str =
    "0065006500650065006500650065006500650065006500650065006500650065";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9065",
    identifier: "BooguAmbiguousFixture",
    ..boogu::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    boogu::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 77,
        source_architecture: "model_base.BooguAmbiguousFixture",
        ..boogu::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_boogu_source_projection_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(boogu::MODEL_FAMILY_IDENTIFIER, "Boogu");
    assert_eq!(boogu::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0065");
    assert_eq!(boogu::MODEL_FAMILY_FIXTURE, "boogu-comfy-model-0065");
    assert_eq!(boogu::MODEL_FAMILY_SOURCE_ORDINAL, 76);
    assert_eq!(
        boogu::MODEL_FAMILY_SOURCE_PATH,
        "projects/comfy/ComfyUI/sim/supported_models.py"
    );
    assert_eq!(boogu::MODEL_FAMILY_REGISTRATION.source_ordinal, 76);
    assert_eq!(
        boogu::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Boogu"
    );
    assert_eq!(boogu::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(boogu::MODEL_FAMILY_SAMPLING_SHIFT, 3.16);
    assert_eq!(boogu::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.15);

    let descriptor = describe_model_family(&boogu::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Boogu");
    assert_eq!(descriptor.family, "COMFY-MODEL-0065");
    assert_eq!(descriptor.architecture_version, "boogu-transformer-2d-v1");
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 4);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(boogu::MODEL_FAMILY_SOURCE_PATH)
        )?),
        boogu::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(
        fixture_directory().join("provenance.json"),
    )?)?;
    assert_eq!(provenance["feature_id"], boogu::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], boogu::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 76);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("Boogu source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        boogu::MODEL_FAMILY_PROJECTION_SHA256
    );
    assert_eq!(
        provenance["source_projection_sha256"],
        boogu::MODEL_FAMILY_PROJECTION_SHA256
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("Boogu source_files must be an array")?
    {
        let path = source["path"].as_str().ok_or("source path must be a string")?;
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
        .find(|row| row["feature_id"] == boogu::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Boogu catalog row is missing")?;
    assert_eq!(catalog_row["source_ordinal"], 76);
    assert_eq!(catalog_row["source_symbol"], "Boogu");
    assert_eq!(catalog_row["static"]["unet_config"]["value"]["image_model"], "boogu");
    assert_eq!(catalog_row["static"]["latent_format"]["value"]["symbol"], "latent_formats.Flux");

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/boogu_sim_model_0065.rs"),
    )?;
    for canonical_adapter in [
        "ModelFamilyRegistration",
        "omnigen2_boogu_family",
        "OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN",
        "BOOGU_COMPONENT_STATE_SCHEMAS",
        "BOOGU_FORWARD_PROGRAM",
        "BOOGU_MEMORY_ESTIMATOR",
        "omnigen2_boogu_configuration_for_probe",
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
    assert!(!row_source.contains("Diffusers"));
    assert!(row_source.contains("ModelStateLayout::StandaloneNative"));
    assert!(boogu::MODEL_FAMILY_REGISTRATION.profile_selector.is_some());
    for (layout, expected_layout) in [
        ("native", Omnigen2BooguLayout::PrefixedNative),
        ("standalone", Omnigen2BooguLayout::StandaloneNative),
    ] {
        let probe = ModelProbe::from_parsed_facts(source_exact_parsed_facts(layout, DType::F32))?;
        let configuration = boogu::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, Omnigen2BooguVariant::Boogu);
        assert_eq!(configuration.layout, expected_layout);
        assert_eq!(configuration.hidden_size, 3_360);
        assert_eq!(configuration.number_of_layers, 32);
        assert_eq!(configuration.number_of_double_stream_layers, 8);
        assert_eq!(configuration.number_of_refiner_layers, 2);
        assert_eq!(configuration.number_of_attention_heads, 28);
        assert_eq!(configuration.number_of_kv_heads, 7);
        assert_eq!(configuration.instruction_feature_dimension, 3_360);
        assert_eq!(configuration.patch_size, 2);
        assert_eq!(configuration.input_channels, 16);
        assert_eq!(configuration.output_channels, 16);
        assert_eq!(configuration.multiple_of, 256);
        assert_eq!(configuration.axes_dimensions, [40, 40, 40]);
        assert_eq!(configuration.axes_lengths, [2_048, 1_664, 1_664]);
        assert_eq!(configuration.timestep_scale, 1_000.0);
        assert_eq!(configuration.sampling_shift, 3.16);
        assert_eq!(configuration.memory_usage_factor, 2.15);
        assert_eq!(configuration.base_supported_dtypes, [DType::Bf16, DType::F32]);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0029");
        assert_eq!(configuration.latent_format.identifier, "Flux");
    }
    super::write_model_family_row_artifact(
        boogu::MODEL_FAMILY_FIXTURE,
        boogu::MODEL_FAMILY_FEATURE_ID,
        boogu::MODEL_FAMILY_IDENTIFIER,
        boogu::MODEL_FAMILY_SOURCE_ORDINAL,
        "boogu_comfy_model_0065",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-and-standalone-native-detection",
            "transactional-component-mapping",
            "named-forward-checkpoints-and-patching",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_boogu_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        boogu::MODEL_FAMILY_REGISTRATION,
    ])?;
    let probe = probe_through_model_store("native")?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "model.diffusion_model.");
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().identity.feature_id(), "COMFY-MODEL-0065");
    assert_eq!(resolved.source_ordinal(), 76);
    assert_eq!(resolved.profile().latent_identifier, "Flux");
    assert!(
        resolved
            .detection()
            .evidence
            .iter()
            .any(|evidence| evidence.contains("double_stream_layers.0"))
    );
    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.boogu.BooguTokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.boogu.te"
    );
    assert!(matches!(
        candidates[0].clip_model().invocation(),
        ModelClipModelInvocation::Factory { configuration }
            if matches!(configuration.as_slice(), [ModelClipConfigurationFact::Expand { source }]
                if source.as_str() == "sim.text_encoders.hunyuan_video.llama_detect")
    ));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let mapped_source = probe_source_tensors(&backend, &context, &probe, DType::F32)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &mapped_source,
    )?;
    assert_eq!(mapped.components().len(), 4);
    assert!(mapped.component("model").is_some());
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());
    let reference_count = mapped
        .component("runtime_conditioning")
        .and_then(|component| component.get("reference_latent_count"))
        .ok_or("generated reference count is missing")?;
    assert_eq!(reference_count.descriptor().shape(), &[1]);
    assert_eq!(reference_count.descriptor().dtype(), DType::I64);

    let weights = legacy_weights(&backend, &context, DType::F32)?;
    let build_options = options(DType::F32, 130);
    let model = build_model_family(&boogu::MODEL_FAMILY, weights, build_options)?;
    assert_eq!(model.memory_estimate().total_bytes, 130);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "double_stream_query",
        &[2.0, -0.5],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "negated_output",
        &[-2.011594, 0.43877035],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "boogu-row-patch".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.x_embedder.bias".to_owned(),
                expected_shape: vec![2],
                values: vec![0.5, 0.5],
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
        "negated_output",
        &[-3.1077223, 0.35945588],
    )?;

    let weights = legacy_weights(&backend, &context, DType::F32)?;
    assert!(matches!(
        build_model_family(&boogu::MODEL_FAMILY, weights, options(DType::F32, 129)),
        Err(ModelFamilyError::OutOfMemory {
            required: 130,
            budget: 129,
        })
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_boogu_standalone_native_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        boogu::MODEL_FAMILY_REGISTRATION,
    ])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );

    let probe = ModelProbe::from_parsed_facts(parsed_facts("standalone", DType::Bf16, false))?;
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "model.");
    let resolved = registry.resolve(&probe)?;
    let source = probe_source_tensors(&backend, &context, &probe, DType::Bf16)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 4);
    let weights = legacy_weights(&backend, &context, DType::Bf16)?;
    let model = build_model_family(
        &boogu::MODEL_FAMILY,
        weights,
        options(DType::Bf16, 130),
    )?;
    assert_eq!(model.profile().supported_dtypes, &[DType::Bf16, DType::F32]);

    let weights = legacy_weights(&backend, &context, DType::F16)?;
    assert!(matches!(
        build_model_family(&boogu::MODEL_FAMILY, weights, options(DType::F16, 130)),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));
    let weights = legacy_weights(&backend, &context, DType::Bf16)?;
    let mut unsupported_device = options(DType::Bf16, 130);
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family(&boogu::MODEL_FAMILY, weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let partial_probe = ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, true))?;
    assert!(matches!(
        registry.resolve(&partial_probe),
        Err(
            ModelFamilyError::ModelLayoutSelection(_)
                | ModelFamilyError::InvalidSelectorOutput(_)
        )
    ));

    let mut unexpected_facts = parsed_facts("native", DType::F32, false);
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: "F32".to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let unexpected_source =
        probe_source_tensors(&backend, &context, &unexpected_probe, DType::F32)?;
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &unexpected_source,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let no_match = ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "transformer_blocks.0.attn.to_q.weight".to_owned(),
            vec![2, 2],
        )]),
        metadata: BTreeMap::from([("image_model".to_owned(), "boogu".to_owned())]),
    };
    assert!(matches!(registry.detect(&no_match), Err(ModelFamilyError::NoDetectionMatch)));
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

fn parsed_facts(layout: &str, dtype: DType, omit_x_weight: bool) -> ModelParsedFacts {
    parsed_facts_with_shape_mode(layout, dtype, omit_x_weight, false)
}

fn source_exact_parsed_facts(layout: &str, dtype: DType) -> ModelParsedFacts {
    parsed_facts_with_shape_mode(layout, dtype, false, true)
}

fn parsed_facts_with_shape_mode(
    layout: &str,
    dtype: DType,
    omit_x_weight: bool,
    source_exact: bool,
) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in probe_shapes(omit_x_weight, source_exact) {
        tensors.insert(
            format!("{prefix}{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for (key, shape) in [
        ("vae.decoder.weight", vec![1]),
        ("text_encoders.qwen3vl_8b.transformer.weight", vec![1]),
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

fn probe_shapes(omit_x_weight: bool, source_exact: bool) -> Vec<(String, Vec<u64>)> {
    let (hidden_size, layers, double_layers, refiners) = if source_exact {
        (3_360, 32, 8, 2)
    } else {
        (56, 1, 1, 1)
    };
    let mut shapes = vec![
        ("x_embedder.bias".to_owned(), vec![hidden_size]),
        ("norm_out.linear_2.weight".to_owned(), vec![64, hidden_size]),
        ("norm_out.linear_2.bias".to_owned(), vec![64]),
        (
            "time_caption_embed.caption_embedder.0.weight".to_owned(),
            vec![hidden_size, 4_096],
        ),
    ];
    if !omit_x_weight {
        shapes.push(("x_embedder.weight".to_owned(), vec![hidden_size, 64]));
    }
    for index in 0..layers {
        shapes.push((
            format!("single_stream_layers.{index}.attn.to_q.weight"),
            vec![hidden_size, hidden_size],
        ));
    }
    for index in 0..double_layers {
        shapes.push((
            format!(
                "double_stream_layers.{index}.img_instruct_attn.processor.img_to_q.weight"
            ),
            vec![hidden_size, hidden_size],
        ));
    }
    for index in 0..refiners {
        shapes.push((
            format!("noise_refiner.{index}.attn.to_q.weight"),
            vec![hidden_size, hidden_size],
        ));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_x_weight: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(omit_x_weight) {
        let values = match key.as_str() {
            "x_embedder.weight" | "norm_out.linear_2.weight" => {
                vec![1.0, 0.0, 0.0, 1.0]
            }
            "x_embedder.bias" => vec![1.0, -1.0],
            "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight" => {
                vec![2.0, 0.0, 0.0, 0.5]
            }
            "norm_out.linear_2.bias" => vec![0.25, -0.25],
            _ => vec![0.0; shape.iter().product::<u64>() as usize],
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
        "text_encoders.qwen3vl_8b.transformer.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    probe: &ModelProbe,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe
        .tensor_shapes
        .iter()
        .map(|(key, shape)| {
            let elements = usize::try_from(shape.iter().product::<u64>())?;
            Ok((
                key.clone(),
                tensor(backend, shape, &vec![0.0; elements], dtype, context)?,
            ))
        })
        .collect()
}

fn legacy_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
) -> Result<MappedModelWeights, Box<dyn std::error::Error>> {
    let mut source = source_tensors(backend, context, "native", dtype, false)?;
    source.retain(|key, _| key.starts_with("model.diffusion_model."));
    Ok(map_model_weights(
        &boogu::MODEL_FAMILY,
        ARTIFACT_DIGEST,
        source,
    )?)
}

fn model_shapes(omit_x_weight: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("x_embedder.bias".to_owned(), vec![2]),
        (
            "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight".to_owned(),
            vec![2, 2],
        ),
        ("norm_out.linear_2.weight".to_owned(), vec![2, 2]),
        ("norm_out.linear_2.bias".to_owned(), vec![2]),
        (
            "time_caption_embed.caption_embedder.0.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_stream_layers.0.attn.to_q.weight".to_owned(),
            vec![2, 2],
        ),
        ("noise_refiner.0.attn.to_q.weight".to_owned(), vec![2, 2]),
    ];
    if !omit_x_weight {
        shapes.push(("x_embedder.weight".to_owned(), vec![2, 2]));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn probe_through_model_store(layout: &str) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("boogu.safetensors");
    write_safetensors(&model_path, layout)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "boogu-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("boogu-row", "boogu.safetensors")?;
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
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({}),
    );
    let mut shapes = probe_shapes(false, false);
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.qwen3vl_8b.transformer.weight".to_owned(), vec![1]),
    ]);
    let mut data = Vec::new();
    for (key, shape) in shapes {
        let name = if key.starts_with("vae.") || key.starts_with("text_encoders.") {
            key
        } else {
            format!("{prefix}{key}")
        };
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
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
        .ok_or("Boogu checkpoint is missing")?;
    let actual = tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{name}: {actual} != {expected}");
    }
    Ok(())
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(boogu::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
