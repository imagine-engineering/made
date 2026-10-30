use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, FluxChromaLayout, FluxChromaVariant,
    ModelClipConfigurationFact, ModelClipModelInvocation, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_flux_sim_model_0077 as flux,
    generated_longcatimage_sim_model_0105 as longcat,
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

const ARTIFACT_DIGEST: &str = "0105010501050105010501050105010501050105010501050105010501050105";
const RESOLVED_MEMORY_BYTES: u64 = 1_868_344;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9105",
    identifier: "LongCatImageAmbiguousFixture",
    ..longcat::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    longcat::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 9_105,
        source_architecture: "model_base.LongCatImageAmbiguousFixture",
        ..longcat::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_longcat_source_projection_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(longcat::MODEL_FAMILY_IDENTIFIER, "LongCatImage");
    assert_eq!(longcat::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0105");
    assert_eq!(
        longcat::MODEL_FAMILY_FIXTURE,
        "longcatimage-comfy-model-0105"
    );
    assert_eq!(longcat::MODEL_FAMILY_SOURCE_ORDINAL, 29);
    assert_eq!(longcat::MODEL_FAMILY_REGISTRATION.source_ordinal, 29);
    assert_eq!(
        longcat::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.LongCatImage"
    );
    assert!(
        longcat::MODEL_FAMILY_REGISTRATION
            .source_configuration
            .is_empty()
    );
    assert_eq!(longcat::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.5);

    let descriptor = describe_model_family(&longcat::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "LongCatImage");
    assert_eq!(descriptor.family, "COMFY-MODEL-0105");
    assert_eq!(descriptor.architecture_version, "longcat-image-flux-v1");
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);

    for layout in [NativeLayout::Prefixed, NativeLayout::Standalone] {
        let probe = ModelProbe::from_parsed_facts(native_facts(layout, DType::F32, true))?;
        let configuration = longcat::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, FluxChromaVariant::LongCatImage);
        assert_eq!(
            configuration.layout,
            match layout {
                NativeLayout::Prefixed => FluxChromaLayout::Native,
                NativeLayout::Standalone => FluxChromaLayout::Unprefixed,
            }
        );
        assert_longcat_configuration(configuration);
    }
    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts(DType::F32, true))?;
    let diffusers_configuration = longcat::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(diffusers_configuration.layout, FluxChromaLayout::Diffusers);
    assert_longcat_configuration(diffusers_configuration);

    let registry = ModelFamilyRegistry::checked_registrations(&[
        flux::MODEL_FAMILY_REGISTRATION,
        longcat::MODEL_FAMILY_REGISTRATION,
    ])?;
    let longcat_resolved = registry.resolve(&ModelProbe::from_parsed_facts(native_facts(
        NativeLayout::Prefixed,
        DType::F32,
        true,
    ))?)?;
    assert_eq!(
        longcat_resolved.detection().identity.feature_id(),
        longcat::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(longcat_resolved.detection().score, 1_200);
    let flux_resolved = registry.resolve(&ModelProbe::from_parsed_facts(flux_facts())?)?;
    assert_eq!(
        flux_resolved.detection().identity.feature_id(),
        flux::MODEL_FAMILY_FEATURE_ID
    );

    let candidate = &longcat_resolved.clip_target().candidates()[0];
    assert_eq!(
        candidate.tokenizer().identifier(),
        "sim.text_encoders.longcat_image.LongCatImageTokenizer"
    );
    assert_eq!(
        candidate.clip_model().target().as_str(),
        "sim.text_encoders.longcat_image.te"
    );
    assert!(matches!(
        candidate.clip_model().invocation(),
        ModelClipModelInvocation::Factory { configuration }
            if matches!(configuration.as_slice(), [ModelClipConfigurationFact::Expand { source }]
                if source.as_str()
                    == "sim.text_encoders.hunyuan_video.llama_detect.qwen25_7b")
    ));

    let store_probe = probe_through_model_store(NativeLayout::Prefixed)?;
    assert_eq!(
        store_probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    assert_eq!(store_probe.metadata["image_model"], "flux");
    assert_eq!(store_probe.metadata["context_in_dim"], "4096");
    assert_eq!(store_probe.metadata["guidance_embed"], "true");
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        longcat::MODEL_FAMILY_FEATURE_ID
    );

    validate_provenance_and_catalog()?;
    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/longcatimage_sim_model_0105.rs"),
    )?;
    for shared_owner in [
        "FluxChromaVariant::LongCatImage",
        "FLUX_COMPONENT_STATE_SCHEMAS",
        "FLUX_FORWARD_PROGRAM",
        "FLUX_LAYOUT_SIGNATURES",
        "FLUX_MEMORY_ESTIMATOR",
        "FLUX_STATE_PLAN_CASES",
    ] {
        assert!(row_source.contains(shared_owner));
    }
    for competing_owner in [
        "pub struct ",
        "pub enum ",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct CancellationToken",
        "struct ArtifactIndex",
        "std::fs",
        "unsafe ",
        "ModelStateTransformPlanDefinition",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor {",
    ] {
        assert!(!row_source.contains(competing_owner));
    }
    assert!(!row_source.contains("ModelDetectionRule::Metadata"));
    assert!(row_source.contains("FLUX_TEXT_PROJECTION_KEYS"));
    assert!(row_source.contains("FLUX_INPUT_PROJECTION_KEYS"));

    super::write_model_family_row_artifact(
        longcat::MODEL_FAMILY_FIXTURE,
        longcat::MODEL_FAMILY_FEATURE_ID,
        longcat::MODEL_FAMILY_IDENTIFIER,
        longcat::MODEL_FAMILY_SOURCE_ORDINAL,
        "longcatimage_comfy_model_0105",
        &[
            "source-provenance-registration-descriptor",
            "model-store-prefixed-standalone-diffusers-detection",
            "source-exact-longcat-configuration-and-qwen-clip-target",
            "transactional-generic-flux-component-mapping",
            "named-forward-rope-conditioning-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "partial-malformed-ambiguous-misleading-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_longcat_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        flux::MODEL_FAMILY_REGISTRATION,
        longcat::MODEL_FAMILY_REGISTRATION,
    ])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );

    for (layout, dtype) in [
        (NativeLayout::Prefixed, DType::F32),
        (NativeLayout::Standalone, DType::F32),
        (NativeLayout::Prefixed, DType::Bf16),
        (NativeLayout::Prefixed, DType::F16),
    ] {
        let probe = ModelProbe::from_parsed_facts(native_facts(layout, dtype, false))?;
        let resolved = registry.resolve(&probe)?;
        let source = native_source_tensors(&backend, &context, layout, dtype, false)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert_eq!(mapped.components().len(), 3);
        assert!(mapped.component("model").is_some_and(|model| {
            model.contains_key("native.double_blocks.0.img_attn.norm.key_norm.weight")
                && !model.contains_key("native.vector_in.in_layer.weight")
                && !model.contains_key("native.guidance_in.in_layer.weight")
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
            options(dtype, DeviceKind::Cpu, RESOLVED_MEMORY_BYTES),
        )?;
        assert_eq!(model.memory_estimate().total_bytes, RESOLVED_MEMORY_BYTES);

        if dtype == DType::F32 && layout == NativeLayout::Prefixed {
            let input = tensor(&backend, &[1, 2], &[1.0, 2.0], dtype, &context)?;
            let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
            assert_checkpoint(
                &backend,
                &context,
                &checkpoints,
                "single_stream_projection",
                &[1.4621172, 0.8807971],
            )?;
            assert_checkpoint(
                &backend,
                &context,
                &checkpoints,
                "flow_output",
                &[0.0, 0.96402675],
            )?;

            let patch = PatchGraph::checked(
                ARTIFACT_DIGEST,
                vec![PatchOperation {
                    identifier: "longcat-single-stream-delta".to_owned(),
                    kind: PatchKind::Lora,
                    scale: 1.0,
                    targets: vec![PatchTarget {
                        key: "native.single_blocks.0.linear2.weight".to_owned(),
                        expected_shape: vec![2, 2],
                        values: vec![1.0, 0.0, 0.0, 1.0],
                        application: PatchApplication::Add,
                    }],
                }],
            )?;
            let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
            assert_checkpoint(
                &backend,
                &context,
                &patched.forward_checkpoints(&backend, &input, &context)?,
                "flow_output",
                &[0.0, -0.9640262],
            )?;

            let first = ordered_patch_graph(true)?;
            let second = ordered_patch_graph(false)?;
            assert_ne!(
                first.identity().ordered_digest,
                second.identity().ordered_digest
            );
            let first_weights = first.apply(&backend, model.weights(), &context)?;
            let second_weights = second.apply(&backend, model.weights(), &context)?;
            assert_ne!(
                tensor_to_f32_with_context_exact_native(
                    &backend,
                    &first_weights.tensors()["native.single_blocks.0.linear2.weight"],
                    &context,
                )?,
                tensor_to_f32_with_context_exact_native(
                    &backend,
                    &second_weights.tensors()["native.single_blocks.0.linear2.weight"],
                    &context,
                )?
            );
        }
    }

    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts(DType::F32, false))?;
    let diffusers_resolved = registry.resolve(&diffusers_probe)?;
    let diffusers_source = diffusers_source_tensors(&backend, &context)?;
    let diffusers_mapped = diffusers_resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &diffusers_source,
    )?;
    assert!(diffusers_mapped.component("model").is_some_and(|model| {
        model.contains_key("native.double_blocks.0.img_attn.qkv.weight")
            && model.contains_key("native.single_blocks.0.linear1.weight")
            && model.contains_key("native.final_layer.linear.weight")
    }));

    let probe = ModelProbe::from_parsed_facts(native_facts(
        NativeLayout::Prefixed,
        DType::F32,
        false,
    ))?;
    let resolved = registry.resolve(&probe)?;
    let source = native_source_tensors(
        &backend,
        &context,
        NativeLayout::Prefixed,
        DType::F32,
        false,
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
            options(DType::F32, DeviceKind::Cpu, RESOLVED_MEMORY_BYTES - 1),
        ),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == RESOLVED_MEMORY_BYTES && budget == RESOLVED_MEMORY_BYTES - 1
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_longcat_typed_failures_cancellation_and_ambiguity()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        flux::MODEL_FAMILY_REGISTRATION,
        longcat::MODEL_FAMILY_REGISTRATION,
    ])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = ModelProbe::from_parsed_facts(native_facts(
        NativeLayout::Prefixed,
        DType::F32,
        false,
    ))?;
    let resolved = registry.resolve(&probe)?;

    let source = native_source_tensors(
        &backend,
        &context,
        NativeLayout::Prefixed,
        DType::F32,
        false,
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
            options(DType::F64, DeviceKind::Cpu, RESOLVED_MEMORY_BYTES),
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
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
            options(DType::F32, DeviceKind::Metal, RESOLVED_MEMORY_BYTES),
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let mut partial_facts = native_facts(NativeLayout::Prefixed, DType::F32, false);
    partial_facts
        .tensors
        .remove("model.diffusion_model.final_layer.linear.weight");
    let partial_probe = ModelProbe::from_parsed_facts(partial_facts)?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let partial_source = native_source_tensors(
        &backend,
        &context,
        NativeLayout::Prefixed,
        DType::F32,
        true,
    )?;
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial_source,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.final_layer.linear.weight"
    ));

    let mut unexpected_facts = native_facts(NativeLayout::Prefixed, DType::F32, false);
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected_source = source.clone();
    unexpected_source.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &unexpected_source,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let flux_probe = ModelProbe::from_parsed_facts(flux_facts())?;
    assert_eq!(
        registry
            .resolve(&flux_probe)?
            .detection()
            .identity
            .feature_id(),
        flux::MODEL_FAMILY_FEATURE_ID
    );
    for mutation in [InvalidProbe::Vector, InvalidProbe::Guidance] {
        let invalid_probe = ModelProbe::from_parsed_facts(invalid_native_facts(mutation))?;
        assert!(matches!(
            registry.resolve(&invalid_probe),
            Err(ModelFamilyError::InvalidSelectorOutput(message))
                if message.contains("LongCatImage") || message.contains("Flux")
        ));
    }

    let mut malformed = native_facts(NativeLayout::Prefixed, DType::F32, false);
    malformed
        .tensors
        .get_mut("model.diffusion_model.img_in.weight")
        .ok_or("LongCat image projection must exist")?
        .shape = vec![64];
    malformed
        .tensors
        .get_mut("model.diffusion_model.txt_in.weight")
        .ok_or("LongCat text projection must exist")?
        .shape = vec![3_584];
    assert!(matches!(
        registry.resolve(&ModelProbe::from_parsed_facts(malformed)?),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::new(105),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeLayout {
    Prefixed,
    Standalone,
}

#[derive(Clone, Copy)]
enum InvalidProbe {
    Vector,
    Guidance,
}

fn assert_longcat_configuration(configuration: sim_model::FluxChromaConfiguration) {
    assert_eq!(configuration.in_channels, 16);
    assert_eq!(configuration.out_channels, 16);
    assert_eq!(configuration.patch_size, 2);
    assert_eq!(configuration.hidden_size, 128);
    assert_eq!(configuration.context_input_dimension, 3_584);
    assert_eq!(configuration.vector_input_dimension, None);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.double_block_count, 19);
    assert_eq!(configuration.single_block_count, 38);
    assert!(!configuration.guidance_embedding);
    assert_eq!(configuration.text_id_dimensions, [1, 2]);
}

fn native_facts(layout: NativeLayout, dtype: DType, full_depth: bool) -> ModelParsedFacts {
    let prefix = native_prefix(layout);
    let mut tensors = native_shapes(false, full_depth)
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
        .collect::<BTreeMap<_, _>>();
    for (key, shape) in [
        ("vae.decoder.weight", vec![1]),
        (
            "text_encoders.qwen25_7b.transformer.layers.0.input_layernorm.weight",
            vec![1],
        ),
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

fn invalid_native_facts(mutation: InvalidProbe) -> ModelParsedFacts {
    let mut facts = native_facts(NativeLayout::Prefixed, DType::F32, true);
    match mutation {
        InvalidProbe::Vector => {
            facts.tensors.insert(
                "model.diffusion_model.vector_in.in_layer.weight".to_owned(),
                ModelParsedTensorFact {
                    shape: vec![128, 768],
                    storage_dtype: DType::F32.catalog_name().to_owned(),
                },
            );
        }
        InvalidProbe::Guidance => {
            facts.tensors.insert(
                "model.diffusion_model.guidance_in.in_layer.weight".to_owned(),
                ModelParsedTensorFact {
                    shape: vec![128, 256],
                    storage_dtype: DType::F32.catalog_name().to_owned(),
                },
            );
        }
    }
    facts
}

fn flux_facts() -> ModelParsedFacts {
    let mut facts = native_facts(NativeLayout::Prefixed, DType::F32, true);
    facts
        .tensors
        .get_mut("model.diffusion_model.txt_in.weight")
        .expect("test fixture text projection must exist")
        .shape = vec![128, 4_096];
    for key in [
        "model.diffusion_model.vector_in.in_layer.weight",
        "model.diffusion_model.guidance_in.in_layer.weight",
    ] {
        facts.tensors.insert(
            key.to_owned(),
            ModelParsedTensorFact {
                shape: vec![128, 768],
                storage_dtype: DType::F32.catalog_name().to_owned(),
            },
        );
    }
    facts
}

fn diffusers_facts(dtype: DType, full_depth: bool) -> ModelParsedFacts {
    ModelParsedFacts {
        tensors: diffusers_shapes(full_depth)
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
            .collect(),
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::new(),
        }],
    }
}

fn native_shapes(omit_final: bool, full_depth: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "double_blocks.0.img_attn.norm.key_norm.scale".to_owned(),
            vec![128],
        ),
        ("img_in.weight".to_owned(), vec![128, 64]),
        ("txt_in.weight".to_owned(), vec![128, 3_584]),
        (
            "double_blocks.0.img_attn.proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("single_blocks.0.linear2.weight".to_owned(), vec![2, 2]),
    ];
    if !omit_final {
        shapes.push(("final_layer.linear.weight".to_owned(), vec![2, 2]));
    }
    if full_depth {
        for index in 1..longcat::SOURCE_DOUBLE_BLOCK_COUNT {
            shapes.push((
                format!("double_blocks.{index}.img_attn.norm.key_norm.scale"),
                vec![1],
            ));
        }
        for index in 1..longcat::SOURCE_SINGLE_BLOCK_COUNT {
            shapes.push((
                format!("single_blocks.{index}.modulation.lin.weight"),
                vec![1],
            ));
        }
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn diffusers_shapes(full_depth: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("x_embedder.weight".to_owned(), vec![128, 64]),
        ("x_embedder.bias".to_owned(), vec![128]),
        ("context_embedder.weight".to_owned(), vec![128, 3_584]),
        (
            "transformer_blocks.0.attn.norm_k.weight".to_owned(),
            vec![128],
        ),
        (
            "transformer_blocks.0.attn.to_q.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "transformer_blocks.0.attn.to_k.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "transformer_blocks.0.attn.to_v.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "transformer_blocks.0.attn.to_out.0.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_transformer_blocks.0.attn.norm_k.weight".to_owned(),
            vec![128],
        ),
        (
            "single_transformer_blocks.0.attn.to_q.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_transformer_blocks.0.attn.to_k.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_transformer_blocks.0.attn.to_v.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_transformer_blocks.0.proj_mlp.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "single_transformer_blocks.0.proj_out.weight".to_owned(),
            vec![2, 2],
        ),
        ("proj_out.weight".to_owned(), vec![2, 2]),
    ];
    if full_depth {
        for index in 1..longcat::SOURCE_DOUBLE_BLOCK_COUNT {
            shapes.push((
                format!("transformer_blocks.{index}.attn.norm_k.weight"),
                vec![1],
            ));
        }
        for index in 1..longcat::SOURCE_SINGLE_BLOCK_COUNT {
            shapes.push((
                format!("single_transformer_blocks.{index}.attn.norm_k.weight"),
                vec![1],
            ));
        }
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn native_source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: NativeLayout,
    dtype: DType,
    omit_final: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = native_prefix(layout);
    let mut source = BTreeMap::new();
    for (key, shape) in native_shapes(omit_final, false) {
        let values = model_values(&key, &shape)?;
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
        "text_encoders.qwen25_7b.transformer.layers.0.input_layernorm.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn diffusers_source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut source = BTreeMap::new();
    for (key, shape) in diffusers_shapes(false) {
        let values = match key.as_str() {
            "transformer_blocks.0.attn.to_out.0.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "single_transformer_blocks.0.proj_out.weight" => vec![2.0, 0.0, 0.0, 0.5],
            "proj_out.weight" => vec![1.0, 1.0, 1.0, -1.0],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
        };
        source.insert(
            key,
            tensor(backend, &shape, &values, DType::F32, context)?,
        );
    }
    Ok(source)
}

fn model_values(key: &str, shape: &[u64]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(match key {
        "double_blocks.0.img_attn.proj.weight" => vec![1.0, 0.0, 0.0, 1.0],
        "single_blocks.0.linear2.weight" => vec![2.0, 0.0, 0.0, 0.5],
        "final_layer.linear.weight" => vec![1.0, 1.0, 1.0, -1.0],
        _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
    })
}

fn probe_through_model_store(
    layout: NativeLayout,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("longcat.safetensors");
    write_safetensors(&model_path, layout)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "longcat-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("longcat-row", "longcat.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(
    path: &Path,
    layout: NativeLayout,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = native_prefix(layout);
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            "image_model": "flux",
            "context_in_dim": "4096",
            "guidance_embed": "true"
        }),
    );
    let mut data = Vec::new();
    for (key, shape) in native_shapes(false, true) {
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
            format!("{prefix}{key}"),
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

fn validate_provenance_and_catalog() -> Result<(), Box<dyn std::error::Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(longcat::MODEL_FAMILY_SOURCE_PATH)
        )?),
        longcat::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], longcat::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], longcat::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 29);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("LongCatImage source projection must be text")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("LongCatImage source_files must be an array")?
    {
        let path = source["path"]
            .as_str()
            .ok_or("source path must be text")?;
        assert_eq!(
            sha256(&std::fs::read(repository.join(path))?),
            source["sha256"]
        );
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("model-family catalog models must be an array")?
        .iter()
        .find(|row| row["feature_id"] == longcat::MODEL_FAMILY_FEATURE_ID)
        .ok_or("LongCatImage catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 29);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "flux");
    assert_eq!(row["static"]["unet_config"]["value"]["context_in_dim"], 3_584);
    assert_eq!(
        sha256(&serde_json::to_vec(row)?),
        longcat::MODEL_FAMILY_PROJECTION_SHA256
    );
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

fn options(
    dtype: DType,
    device: DeviceKind,
    memory_budget_bytes: u64,
) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device,
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
        .ok_or("LongCatImage checkpoint is missing")?;
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
        identifier: "longcat-ordered-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.single_blocks.0.linear2.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "longcat-ordered-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.single_blocks.0.linear2.weight".to_owned(),
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

fn native_prefix(layout: NativeLayout) -> &'static str {
    match layout {
        NativeLayout::Prefixed => "model.diffusion_model.",
        NativeLayout::Standalone => "",
    }
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(longcat::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
