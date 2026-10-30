use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, FluxChromaFinalHead, FluxChromaLayout,
    FluxChromaVariant, ModelClipConfigurationFact, ModelClipModelInvocation, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts,
    ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_flux_sim_model_0077 as base_flux,
    generated_fluxschnell_sim_model_0080 as schnell,
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

const ARTIFACT_DIGEST: &str = "0800800800800800800800800800800800800800800800800800800800800800";
const RESOLVED_MEMORY_BYTES: u64 = 35_384;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9080",
    identifier: "FluxSchnellAmbiguousFixture",
    ..schnell::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    schnell::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 130,
        source_architecture: "model_base.FluxSchnellAmbiguousFixture",
        ..schnell::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_fluxschnell_source_projection_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(schnell::MODEL_FAMILY_IDENTIFIER, "FluxSchnell");
    assert_eq!(schnell::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0080");
    assert_eq!(
        schnell::MODEL_FAMILY_FIXTURE,
        "fluxschnell-comfy-model-0080"
    );
    assert_eq!(schnell::MODEL_FAMILY_SOURCE_ORDINAL, 30);
    assert_eq!(schnell::MODEL_FAMILY_REGISTRATION.source_ordinal, 30);
    assert_eq!(
        schnell::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Flux(model_type=model_base.ModelType.FLOW)"
    );
    assert!(schnell::MODEL_FAMILY_REGISTRATION.source_configuration.is_empty());
    assert_eq!(schnell::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(schnell::MODEL_FAMILY_SAMPLING_SHIFT, 1.0);
    assert_eq!(schnell::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 3.1);

    let descriptor = describe_model_family(&schnell::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "FluxSchnell");
    assert_eq!(descriptor.family, "COMFY-MODEL-0080");
    assert_eq!(descriptor.architecture_version, "flux-schnell-flow-v1");
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(
        schnell::MODEL_FAMILY.latent_feature_id,
        base_flux::MODEL_FAMILY.latent_feature_id
    );
    assert_eq!(
        schnell::MODEL_FAMILY.latent_identifier,
        base_flux::MODEL_FAMILY.latent_identifier
    );
    assert!(std::ptr::eq(
        schnell::MODEL_FAMILY.clip_target,
        base_flux::MODEL_FAMILY.clip_target
    ));
    assert!(std::ptr::eq(
        schnell::MODEL_FAMILY.weight_rules,
        base_flux::MODEL_FAMILY.weight_rules
    ));
    assert!(std::ptr::eq(
        schnell::MODEL_FAMILY.forward_program,
        base_flux::MODEL_FAMILY.forward_program
    ));

    let configuration = schnell::configuration_for_probe(&ModelProbe::from_parsed_facts(
        parsed_facts("native", DType::F32, false, false, false),
    )?)?;
    assert_eq!(configuration.variant, FluxChromaVariant::Flux);
    assert_eq!(configuration.layout, FluxChromaLayout::Native);
    assert_eq!(
        (configuration.in_channels, configuration.out_channels),
        (16, 16)
    );
    assert_eq!(
        (configuration.patch_size, configuration.hidden_size),
        (2, 128)
    );
    assert_eq!(configuration.context_input_dimension, 2);
    assert_eq!(configuration.vector_input_dimension, Some(2));
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(
        (
            configuration.double_block_count,
            configuration.single_block_count
        ),
        (1, 1)
    );
    assert!(!configuration.guidance_embedding);
    assert_eq!(configuration.final_head, FluxChromaFinalHead::Linear);
    assert!(!configuration.use_x0_prediction);
    assert!(!configuration.use_sequential_text_ids);

    let mut diffusers_facts = diffusers_facts(DType::F32);
    diffusers_facts.formats[0].metadata.extend([
        ("image_model".to_owned(), "flux2".to_owned()),
        ("guidance_embed".to_owned(), "true".to_owned()),
        ("in_channels".to_owned(), "96".to_owned()),
        ("model_layout".to_owned(), "native".to_owned()),
    ]);
    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts)?;
    let registry =
        ModelFamilyRegistry::checked_registrations(&[schnell::MODEL_FAMILY_REGISTRATION])?;
    let diffusers_resolved = registry.resolve(&diffusers_probe)?;
    assert_eq!(
        diffusers_resolved.detection().identity.feature_id(),
        schnell::MODEL_FAMILY_FEATURE_ID
    );
    let configuration = schnell::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(configuration.layout, FluxChromaLayout::Diffusers);
    assert_eq!(configuration.in_channels, 16);
    assert!(!configuration.guidance_embedding);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(schnell::MODEL_FAMILY_SOURCE_PATH)
        )?),
        schnell::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], schnell::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(
        provenance["source_symbol"],
        schnell::MODEL_FAMILY_IDENTIFIER
    );
    assert_eq!(provenance["source_ordinal"], 30);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("FluxSchnell source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("FluxSchnell source_files must be an array")?
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
    let catalog_row = catalog["models"]
        .as_array()
        .ok_or("model-family catalog models must be an array")?
        .iter()
        .find(|row| row["feature_id"] == schnell::MODEL_FAMILY_FEATURE_ID)
        .ok_or("FluxSchnell catalog row is missing")?;
    assert_eq!(catalog_row["source_ordinal"], 30);
    assert_eq!(catalog_row["source_symbol"], "FluxSchnell");
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["image_model"],
        "flux"
    );
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["guidance_embed"],
        false
    );
    assert_eq!(
        catalog_row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.Flux"
    );
    assert_eq!(
        sha256(&serde_json::to_vec(catalog_row)?),
        schnell::MODEL_FAMILY_PROJECTION_SHA256
    );
    assert_eq!(
        provenance["catalog_projection_sha256"],
        schnell::MODEL_FAMILY_PROJECTION_SHA256
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/fluxschnell_sim_model_0080.rs"),
    )?;
    for canonical_adapter in [
        "ModelFamilyRegistration",
        "FLUX_COMPONENT_STATE_SCHEMAS",
        "FLUX_STATE_PLAN_CASES",
        "FLUX_FORWARD_PROGRAM",
        "flux_chroma_configuration_for_probe",
        "ModelProbe",
    ] {
        assert!(row_source.contains(canonical_adapter));
    }
    for competing_owner in [
        "pub struct ",
        "pub enum ",
        "ModelStateTransformPlanDefinition",
        "ModelFamilyComponentStateSchema",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
        "ModelClipTargetDefinition",
        "const WEIGHT_RULES",
        "const FORWARD_PROGRAM",
        "const SUPPORTED_DTYPES",
        "const SUPPORTED_DEVICES",
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
    assert!(!row_source.contains("ModelDetectionRule::Metadata"));
    assert!(!row_source.contains("ModelSourceConfigurationRule"));
    assert!(row_source.contains("FLUX_INPUT_PROJECTION_KEYS"));
    assert!(row_source.contains("source_configuration: &[]"));
    super::write_model_family_row_artifact(
        schnell::MODEL_FAMILY_FIXTURE,
        schnell::MODEL_FAMILY_FEATURE_ID,
        schnell::MODEL_FAMILY_IDENTIFIER,
        schnell::MODEL_FAMILY_SOURCE_ORDINAL,
        "fluxschnell_comfy_model_0080",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-unprefixed-and-diffusers-detection",
            "transactional-component-and-scale-mapping",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-guided-flux2-exclusion-and-owner-delegation",
            "guidance-false-flow-sampling-and-base-flux-separation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_fluxschnell_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        base_flux::MODEL_FAMILY_REGISTRATION,
        schnell::MODEL_FAMILY_REGISTRATION,
    ])?;
    let probe = probe_through_model_store("native")?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert!(!probe.metadata.contains_key("image_model"));
    assert!(!probe.metadata.contains_key("guidance_embed"));
    assert!(!probe.metadata.contains_key("in_channels"));
    assert!(!probe.metadata.contains_key("model_layout"));
    assert_eq!(
        probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 700);
    assert_eq!(resolved.detection().evidence.len(), 1);
    assert!(resolved.detection().evidence[0].contains("AnyTensorDimensionValue"));
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0080"
    );
    assert_eq!(resolved.source_ordinal(), 30);
    assert_eq!(resolved.profile().latent_identifier, "Flux");

    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].tokenizer().identifier(),
        "sim.text_encoders.flux.FluxTokenizer"
    );
    assert_eq!(
        candidates[0].clip_model().target().as_str(),
        "sim.text_encoders.flux.flux_clip"
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
    let source = source_tensors(&backend, &context, "native", DType::F32, false)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("model").is_some_and(|model| {
        model.contains_key("native.double_blocks.0.img_attn.norm.key_norm.weight")
    }));
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("text_encoder").is_some());
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let build_options = options(DType::F32, RESOLVED_MEMORY_BYTES);
    let model = build_model_family_for_probe(&registry, &probe, weights, build_options)?;
    assert_eq!(model.memory_estimate().total_bytes, RESOLVED_MEMORY_BYTES);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
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
            identifier: "fluxschnell-single-stream-delta".to_owned(),
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
    let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &patched_checkpoints,
        "flow_output",
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
            &first.tensors()["native.single_blocks.0.linear2.weight"],
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            &second.tensors()["native.single_blocks.0.linear2.weight"],
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
fn val_model_family_row_001_fluxschnell_unprefixed_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[schnell::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = probe_through_model_store("unprefixed")?;
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "model.");
    let resolved = registry.resolve(&probe)?;
    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let source = source_tensors(&backend, &context, "unprefixed", dtype, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert!(
            build_model_family_for_probe(
                &registry,
                &probe,
                weights,
                options(dtype, RESOLVED_MEMORY_BYTES),
            )
            .is_ok()
        );
    }

    let source = source_tensors(&backend, &context, "unprefixed", DType::F32, false)?;
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
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, true, false, false))?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let partial_source = source_tensors(&backend, &context, "native", DType::F32, true)?;
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial_source,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.final_layer.linear.weight"
    ));

    let malformed =
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, false, false, true))?;
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let flux2 =
        ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, false, true, false))?;
    assert!(matches!(
        registry.resolve(&flux2),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Flux2")
    ));

    let mut guided_facts = parsed_facts("native", DType::F32, false, false, false);
    guided_facts.tensors.insert(
        "model.diffusion_model.guidance_in.in_layer.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![128, 2],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let guided_probe = ModelProbe::from_parsed_facts(guided_facts)?;
    assert!(matches!(
        registry.resolve(&guided_probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("guidance embedding must be disabled")
    ));

    let mut inpaint_facts = parsed_facts("native", DType::F32, false, false, false);
    inpaint_facts.tensors.insert(
        "model.diffusion_model.img_in.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![128, 384],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let inpaint_probe = ModelProbe::from_parsed_facts(inpaint_facts)?;
    assert!(matches!(
        registry.resolve(&inpaint_probe),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut base_facts = parsed_facts("native", DType::F32, false, false, false);
    base_facts.formats[0]
        .metadata
        .insert("guidance_embed".to_owned(), "true".to_owned());
    base_facts.tensors.insert(
        "model.diffusion_model.guidance_in.in_layer.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![128, 2],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let base_probe = ModelProbe::from_parsed_facts(base_facts)?;
    let combined = ModelFamilyRegistry::checked_registrations(&[
        base_flux::MODEL_FAMILY_REGISTRATION,
        schnell::MODEL_FAMILY_REGISTRATION,
    ])?;
    assert_eq!(
        combined
            .resolve(&base_probe)?
            .detection()
            .identity
            .feature_id(),
        base_flux::MODEL_FAMILY_FEATURE_ID
    );

    let mut unexpected_facts = parsed_facts("native", DType::F32, false, false, false);
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: "F32".to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected_source = source_tensors(&backend, &context, "native", DType::F32, false)?;
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

    let no_match = ModelProbe {
        tensor_shapes: BTreeMap::new(),
        metadata: BTreeMap::from([("image_model".to_owned(), "flux2".to_owned())]),
    };
    assert!(matches!(
        registry.detect(&no_match),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 700, .. })
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
    layout: &str,
    dtype: DType,
    omit_final: bool,
    flux2: bool,
    malformed_input: bool,
) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in model_shapes(omit_final, flux2, malformed_input) {
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
        (
            "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight",
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

fn diffusers_facts(dtype: DType) -> ModelParsedFacts {
    let mut tensors = BTreeMap::new();
    for (key, shape) in diffusers_shapes() {
        tensors.insert(
            key,
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

fn diffusers_shapes() -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("x_embedder.weight".to_owned(), vec![128, 64]),
        ("x_embedder.bias".to_owned(), vec![128]),
        ("context_embedder.weight".to_owned(), vec![128, 2]),
        ("transformer_blocks.0.attn.norm_k.weight".to_owned(), vec![128]),
        ("transformer_blocks.0.attn.to_q.weight".to_owned(), vec![128, 128]),
        ("transformer_blocks.0.attn.to_k.weight".to_owned(), vec![128, 128]),
        ("transformer_blocks.0.attn.to_v.weight".to_owned(), vec![128, 128]),
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
            vec![128, 128],
        ),
        (
            "single_transformer_blocks.0.attn.to_k.weight".to_owned(),
            vec![128, 128],
        ),
        (
            "single_transformer_blocks.0.attn.to_v.weight".to_owned(),
            vec![128, 128],
        ),
        (
            "single_transformer_blocks.0.proj_mlp.weight".to_owned(),
            vec![512, 128],
        ),
        (
            "single_transformer_blocks.0.proj_out.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "time_text_embed.text_embedder.linear_1.weight".to_owned(),
            vec![128, 2],
        ),
        ("proj_out.weight".to_owned(), vec![2, 2]),
    ];
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_final: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(omit_final, false, false) {
        let values = match key.as_str() {
            "double_blocks.0.img_attn.proj.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "single_blocks.0.linear2.weight" => vec![2.0, 0.0, 0.0, 0.5],
            "final_layer.linear.weight" => vec![1.0, 1.0, 1.0, -1.0],
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
        "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn model_shapes(omit_final: bool, flux2: bool, malformed_input: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "double_blocks.0.img_attn.norm.key_norm.scale".to_owned(),
            vec![128],
        ),
        (
            "img_in.weight".to_owned(),
            if malformed_input {
                vec![64]
            } else {
                vec![128, 64]
            },
        ),
        ("txt_in.weight".to_owned(), vec![128, 2]),
        ("vector_in.in_layer.weight".to_owned(), vec![128, 2]),
        (
            "double_blocks.0.img_attn.proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("single_blocks.0.linear2.weight".to_owned(), vec![2, 2]),
    ];
    if !omit_final {
        shapes.push(("final_layer.linear.weight".to_owned(), vec![2, 2]));
    }
    if flux2 {
        shapes.push((
            "double_stream_modulation_img.lin.weight".to_owned(),
            vec![2, 2],
        ));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn probe_through_model_store(layout: &str) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("fluxschnell.safetensors");
    write_safetensors(&model_path, layout)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "fluxschnell-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("fluxschnell-row", "fluxschnell.safetensors")?;
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
    let mut shapes = model_shapes(false, false, false);
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        (
            "text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight".to_owned(),
            vec![1],
        ),
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
            total
                .checked_mul(*dimension)
                .ok_or("fixture shape overflow")
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
        .ok_or("FluxSchnell checkpoint is missing")?;
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
        identifier: "fluxschnell-ordered-replacement".to_owned(),
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
        identifier: "fluxschnell-ordered-addition".to_owned(),
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

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(schnell::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
