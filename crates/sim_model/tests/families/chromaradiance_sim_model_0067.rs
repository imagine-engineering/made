use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipConfigurationFact,
    ModelClipModelInvocation, ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration,
    ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits, PatchApplication,
    PatchGraph, PatchKind, PatchOperation, PatchTarget, build_model_family_for_probe,
    describe_model_family, generated_chromaradiance_sim_model_0067 as radiance,
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
    "0067006700670067006700670067006700670067006700670067006700670067";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9067",
    identifier: "ChromaRadianceAmbiguousFixture",
    ..radiance::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    radiance::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 73,
        source_architecture: "model_base.ChromaRadianceAmbiguousFixture",
        ..radiance::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_chromaradiance_source_projection_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(radiance::MODEL_FAMILY_IDENTIFIER, "ChromaRadiance");
    assert_eq!(radiance::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0067");
    assert_eq!(
        radiance::MODEL_FAMILY_FIXTURE,
        "chromaradiance-comfy-model-0067"
    );
    assert_eq!(radiance::MODEL_FAMILY_SOURCE_ORDINAL, 72);
    assert_eq!(radiance::MODEL_FAMILY_REGISTRATION.source_ordinal, 72);
    assert_eq!(
        radiance::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.ChromaRadiance"
    );
    assert_eq!(radiance::SOURCE_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(radiance::SOURCE_MEMORY_USAGE_FACTOR, 0.044);
    assert_eq!(radiance::SOURCE_NERF_HIDDEN_SIZE, 64);
    assert_eq!(radiance::SOURCE_NERF_MLP_RATIO, 4);
    assert_eq!(radiance::SOURCE_NERF_DEPTH, 4);
    assert_eq!(radiance::SOURCE_NERF_MAX_FREQUENCIES, 8);
    assert_eq!(radiance::SOURCE_NERF_TILE_SIZE, 512);

    let descriptor = describe_model_family(&radiance::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "ChromaRadiance");
    assert_eq!(descriptor.family, "COMFY-MODEL-0067");
    assert_eq!(descriptor.architecture_version, "chroma-radiance-v1");
    assert_eq!(descriptor.latent_format, "ChromaRadiance");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 1);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 1);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(
        fixture_directory().join("provenance.json"),
    )?)?;
    assert_eq!(provenance["feature_id"], radiance::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], radiance::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 72);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("ChromaRadiance source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("ChromaRadiance source_files must be an array")?
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
        .find(|row| row["feature_id"] == radiance::MODEL_FAMILY_FEATURE_ID)
        .ok_or("ChromaRadiance catalog row is missing")?;
    assert_eq!(catalog_row["source_ordinal"], 72);
    assert_eq!(catalog_row["source_symbol"], "ChromaRadiance");
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["image_model"],
        "chroma_radiance"
    );
    assert_eq!(
        catalog_row["static"]["latent_format"]["value"]["symbol"],
        "sim.latent_formats.ChromaRadiance"
    );
    assert_eq!(
        catalog_row["static"]["memory_usage_factor"]["value"],
        0.044
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/chromaradiance_sim_model_0067.rs"),
    )?;
    for canonical_adapter in [
        "ModelFamilyRegistration",
        "ModelStateTransformPlanDefinition",
        "ModelFamilyComponentStateSchema",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(row_source.contains(canonical_adapter));
    }
    for competing_owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct ModelFamilyRegistry",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct MemoryEstimator",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(competing_owner));
    }

    super::write_model_family_row_artifact(
        radiance::MODEL_FAMILY_FIXTURE,
        radiance::MODEL_FAMILY_FEATURE_ID,
        radiance::MODEL_FAMILY_IDENTIFIER,
        radiance::MODEL_FAMILY_SOURCE_ORDINAL,
        "chromaradiance_comfy_model_0067",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-and-unprefixed-detection",
            "scale-to-weight-transactional-component-mapping",
            "linear-and-convolution-native-program-selection",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation-and-ownership",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_chromaradiance_native_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry()?;
    let probe = probe_through_model_store()?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert!(!probe.metadata.contains_key("image_model"));
    assert!(!probe.metadata.contains_key("model_layout"));
    assert_eq!(
        probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().identity.feature_id(), "COMFY-MODEL-0067");
    assert_eq!(resolved.source_ordinal(), 72);
    assert_eq!(resolved.profile().latent_identifier, "ChromaRadiance");
    assert_eq!(resolved.detection().evidence.len(), 4);
    assert!(
        resolved
            .detection()
            .evidence
            .iter()
            .all(|evidence| evidence.contains("AnyKeyPresent"))
    );
    let configuration = radiance::configuration_for_probe(&probe)?;
    assert_eq!(configuration.layout, radiance::ChromaRadianceLayout::Native);
    assert_eq!(configuration.patch_size, 1);
    assert_eq!(configuration.hidden_size, 128);
    assert_eq!(configuration.context_input_dimension, 4);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.double_block_count, 1);
    assert_eq!(configuration.single_block_count, 1);
    assert_eq!(
        configuration.final_head,
        radiance::ChromaRadianceFinalHead::Linear
    );
    assert!(!configuration.use_x0_prediction);
    assert!(!configuration.use_sequential_text_ids);

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

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(
        &backend,
        &context,
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    )?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("denoiser").is_some());
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());
    let denoiser = mapped.component("denoiser").ok_or("denoiser is missing")?;
    assert!(denoiser.contains_key("native.nerf_blocks.0.norm.weight"));
    assert!(!denoiser.contains_key("native.nerf_blocks.0.norm.scale"));

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model = build_model_family_for_probe(
        &registry,
        &probe,
        weights,
        options(DType::F32, 550),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, 550);
    let input = tensor(&backend, &[1, 2], &[2.0, -1.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "nerf_final_normalization",
        &[0.99999976, -0.99999976],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "linear_radiance_output",
        &[1.2499998, -1.2499998],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "chromaradiance-final-bias".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.nerf_final_layer.linear.bias".to_owned(),
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
        "linear_radiance_output",
        &[1.7499998, -0.74999976],
    )?;

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, 549)),
        Err(ModelFamilyError::OutOfMemory {
            required: 550,
            budget: 549,
        })
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_chromaradiance_unprefixed_conv_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = ModelProbe::from_parsed_facts(parsed_facts(
        radiance::ChromaRadianceLayout::Unprefixed,
        radiance::ChromaRadianceFinalHead::Convolution,
        DType::F32,
    ))?;
    let resolved = registry.resolve(&probe)?;
    let configuration = radiance::configuration_for_probe(&probe)?;
    assert_eq!(
        configuration.layout,
        radiance::ChromaRadianceLayout::Unprefixed
    );
    assert_eq!(
        configuration.final_head,
        radiance::ChromaRadianceFinalHead::Convolution
    );
    assert!(configuration.use_x0_prediction);
    assert!(configuration.use_sequential_text_ids);

    let source = source_tensors(
        &backend,
        &context,
        radiance::ChromaRadianceLayout::Unprefixed,
        radiance::ChromaRadianceFinalHead::Convolution,
        DType::F32,
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
        options(DType::F32, 1_024),
    )?;
    let input = tensor(
        &backend,
        &[1, 2, 1, 1],
        &[1.0, 2.0],
        DType::F32,
        &context,
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &model.forward_checkpoints(&backend, &input, &context)?,
        "convolution_radiance_output",
        &[1.1, 1.9, 3.0],
    )?;

    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let typed_source = source_tensors(
            &backend,
            &context,
            radiance::ChromaRadianceLayout::Native,
            radiance::ChromaRadianceFinalHead::Linear,
            dtype,
        )?;
        let typed_probe = ModelProbe::from_parsed_facts(parsed_facts(
            radiance::ChromaRadianceLayout::Native,
            radiance::ChromaRadianceFinalHead::Linear,
            dtype,
        ))?;
        let typed_resolved = registry.resolve(&typed_probe)?;
        let weights = typed_resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &typed_source,
        )?;
        build_model_family_for_probe(
            &registry,
            &typed_probe,
            weights,
            options(dtype, 1_024),
        )?;
    }

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F64, 1_024)),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported_device = options(DType::F32, 1_024);
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut malformed = parsed_facts(
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    );
    malformed
        .tensors
        .get_mut("model.diffusion_model.img_in_patch.weight")
        .ok_or("img_in_patch fixture is missing")?
        .shape = vec![2, 4, 1, 1];
    let malformed = ModelProbe::from_parsed_facts(malformed)?;
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("img_in_patch.weight shape")
    ));

    let mut partial = parsed_facts(
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    );
    partial
        .tensors
        .remove("model.diffusion_model.nerf_blocks.3.norm.scale");
    let partial = ModelProbe::from_parsed_facts(partial)?;
    assert!(matches!(
        registry.resolve(&partial),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("NeRF depth")
    ));

    let mut missing_head = parsed_facts(
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    );
    missing_head
        .tensors
        .remove("model.diffusion_model.nerf_final_layer.linear.weight");
    let missing_head = ModelProbe::from_parsed_facts(missing_head)?;
    assert!(matches!(
        registry.detect(&missing_head),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut duplicate_facts = parsed_facts(
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    );
    duplicate_facts.tensors.insert(
        "model.diffusion_model.nerf_blocks.0.norm.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![2],
            storage_dtype: "F32".to_owned(),
        },
    );
    let duplicate_probe = ModelProbe::from_parsed_facts(duplicate_facts)?;
    let duplicate_resolved = registry.resolve(&duplicate_probe)?;
    let mut duplicate_source = source_tensors(
        &backend,
        &context,
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    )?;
    duplicate_source.insert(
        "model.diffusion_model.nerf_blocks.0.norm.weight".to_owned(),
        tensor(&backend, &[2], &[1.0, 1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        duplicate_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &duplicate_source,
        ),
        Err(ModelFamilyError::DuplicateComponentKey { component, key })
            if component == "denoiser" && key == "native.nerf_blocks.0.norm.weight"
    ));

    let mut unexpected_facts = parsed_facts(
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    );
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: "F32".to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected_source = source_tensors(
        &backend,
        &context,
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
        DType::F32,
    )?;
    unexpected_source.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[0.0], DType::F32, &context)?,
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
        metadata: BTreeMap::from([("image_model".to_owned(), "chroma".to_owned())]),
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

fn registry() -> Result<ModelFamilyRegistry, ModelFamilyError> {
    ModelFamilyRegistry::checked_registrations(&[radiance::MODEL_FAMILY_REGISTRATION])
}

fn parsed_facts(
    layout: radiance::ChromaRadianceLayout,
    head: radiance::ChromaRadianceFinalHead,
    dtype: DType,
) -> ModelParsedFacts {
    let tensors = model_shapes(layout, head)
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

fn model_shapes(
    layout: radiance::ChromaRadianceLayout,
    head: radiance::ChromaRadianceFinalHead,
) -> BTreeMap<String, Vec<u64>> {
    let prefix = match layout {
        radiance::ChromaRadianceLayout::Native => "model.diffusion_model.",
        radiance::ChromaRadianceLayout::Unprefixed => "",
        radiance::ChromaRadianceLayout::Diffusers => {
            panic!("ChromaRadiance has no lossless generic Diffusers layout")
        }
    };
    let norm_suffix = match layout {
        radiance::ChromaRadianceLayout::Native => "scale",
        radiance::ChromaRadianceLayout::Unprefixed => "weight",
        radiance::ChromaRadianceLayout::Diffusers => {
            panic!("ChromaRadiance has no lossless generic Diffusers layout")
        }
    };
    let mut shapes = BTreeMap::from([
        (format!("{prefix}img_in_patch.weight"), vec![2, 3, 1, 1]),
        (format!("{prefix}img_in_patch.bias"), vec![2]),
        (format!("{prefix}txt_in.weight"), vec![128, 4]),
        (
            format!("{prefix}double_blocks.0.img_attn.norm.key_norm.{norm_suffix}"),
            vec![2],
        ),
        (
            format!("{prefix}single_blocks.0.linear1.weight"),
            vec![2, 2],
        ),
        (
            format!("{prefix}distilled_guidance_layer.norms.0.{norm_suffix}"),
            vec![2],
        ),
        (
            format!("{prefix}nerf_blocks.0.param_generator.weight"),
            vec![2, 2],
        ),
    ]);
    for index in 0..radiance::SOURCE_NERF_DEPTH {
        shapes.insert(
            format!("{prefix}nerf_blocks.{index}.norm.{norm_suffix}"),
            vec![2],
        );
    }
    match head {
        radiance::ChromaRadianceFinalHead::Linear => {
            shapes.extend([
                (
                    format!("{prefix}nerf_final_layer.norm.{norm_suffix}"),
                    vec![2],
                ),
                (
                    format!("{prefix}nerf_final_layer.linear.weight"),
                    vec![2, 2],
                ),
                (
                    format!("{prefix}nerf_final_layer.linear.bias"),
                    vec![2],
                ),
            ]);
        }
        radiance::ChromaRadianceFinalHead::Convolution => {
            shapes.extend([
                (
                    format!("{prefix}nerf_final_layer_conv.norm.{norm_suffix}"),
                    vec![2],
                ),
                (
                    format!("{prefix}nerf_final_layer_conv.conv.weight"),
                    vec![3, 2, 1, 1],
                ),
                (
                    format!("{prefix}nerf_final_layer_conv.conv.bias"),
                    vec![3],
                ),
                (format!("{prefix}__x0__"), vec![1]),
                (format!("{prefix}__sequential__"), vec![1]),
            ]);
        }
    }
    shapes.insert("first_stage_model.decoder.weight".to_owned(), vec![1]);
    shapes.insert(
        "cond_stage_model.t5xxl.transformer.weight".to_owned(),
        vec![1],
    );
    shapes
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: radiance::ChromaRadianceLayout,
    head: radiance::ChromaRadianceFinalHead,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    model_shapes(layout, head)
        .into_iter()
        .map(|(key, shape)| {
            let values = if key.ends_with("nerf_final_layer.norm.scale")
                || key.ends_with("nerf_final_layer.norm.weight")
                || key.ends_with("nerf_blocks.0.norm.scale")
                || key.ends_with("nerf_blocks.0.norm.weight")
            {
                vec![1.0, 1.0]
            } else if key.ends_with("nerf_final_layer.linear.weight") {
                vec![1.0, 0.0, 0.0, 1.0]
            } else if key.ends_with("nerf_final_layer.linear.bias") {
                vec![0.25, -0.25]
            } else if key.ends_with("nerf_final_layer_conv.conv.weight") {
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            } else if key.ends_with("nerf_final_layer_conv.conv.bias") {
                vec![0.1, -0.1, 0.0]
            } else {
                vec![0.0; shape.iter().product::<u64>() as usize]
            };
            Ok((
                key,
                tensor(backend, &shape, &values, dtype, context)?,
            ))
        })
        .collect()
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("chromaradiance.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "chromaradiance-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("chromaradiance-row", "chromaradiance.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let shapes = model_shapes(
        radiance::ChromaRadianceLayout::Native,
        radiance::ChromaRadianceFinalHead::Linear,
    );
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, shape) in shapes {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
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
        .ok_or("ChromaRadiance checkpoint is missing")?;
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
        .join(radiance::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
