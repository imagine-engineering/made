use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, FluxChromaFinalHead, FluxChromaLayout,
    FluxChromaVariant, ModelClipConfigurationFact, ModelClipModelInvocation, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts,
    ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    flux_chroma_family, generated_flux_sim_model_0077 as base_flux,
    generated_fluxinpaint_sim_model_0079 as inpaint,
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

const ARTIFACT_DIGEST: &str = "0079007900790079007900790079007900790079007900790079007900790079";
const RESOLVED_MEMORY_BYTES: u64 = 200_248;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9079",
    identifier: "FluxInpaintAmbiguousFixture",
    ..inpaint::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    inpaint::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 127,
        source_architecture: "model_base.FluxInpaintAmbiguousFixture",
        ..inpaint::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_fluxinpaint_source_projection_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(inpaint::MODEL_FAMILY_IDENTIFIER, "FluxInpaint");
    assert_eq!(inpaint::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0079");
    assert_eq!(
        inpaint::MODEL_FAMILY_FIXTURE,
        "fluxinpaint-comfy-model-0079"
    );
    assert_eq!(inpaint::MODEL_FAMILY_SOURCE_ORDINAL, 27);
    assert_eq!(inpaint::MODEL_FAMILY_REGISTRATION.source_ordinal, 27);
    assert_eq!(
        inpaint::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Flux"
    );
    assert!(inpaint::MODEL_FAMILY_REGISTRATION.source_configuration.is_empty());
    assert_eq!(inpaint::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 3.1);
    assert_eq!(inpaint::MODEL_FAMILY_IN_CHANNELS, 96);

    let descriptor = describe_model_family(&inpaint::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "FluxInpaint");
    assert_eq!(descriptor.family, "COMFY-MODEL-0079");
    assert_eq!(descriptor.architecture_version, "flux-transformer-v1");
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);
    assert!(std::ptr::eq(
        inpaint::MODEL_FAMILY.clip_target,
        &flux_chroma_family::FLUX_CLIP_TARGET
    ));
    assert!(std::ptr::eq(
        inpaint::MODEL_FAMILY.components,
        flux_chroma_family::FLUX_COMPONENTS
    ));
    assert!(std::ptr::eq(
        inpaint::MODEL_FAMILY.weight_rules,
        flux_chroma_family::FLUX_WEIGHT_RULES
    ));
    assert!(std::ptr::eq(
        inpaint::MODEL_FAMILY.forward_program,
        flux_chroma_family::FLUX_FORWARD_PROGRAM
    ));

    for layout in ["native", "unprefixed"] {
        let configuration = inpaint::configuration_for_probe(&ModelProbe::from_parsed_facts(
            parsed_facts(ProbeCase {
                layout,
                ..ProbeCase::default()
            }),
        )?)?;
        assert_eq!(configuration.variant, FluxChromaVariant::Flux);
        assert_eq!(
            configuration.layout,
            if layout == "native" {
                FluxChromaLayout::Native
            } else {
                FluxChromaLayout::Unprefixed
            }
        );
        assert_eq!(
            (configuration.in_channels, configuration.out_channels),
            (96, 16)
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
        assert!(configuration.guidance_embedding);
        assert_eq!(configuration.final_head, FluxChromaFinalHead::Linear);
        assert!(!configuration.use_x0_prediction);
        assert!(!configuration.use_sequential_text_ids);
    }

    let mut diffusers_facts = diffusers_facts();
    diffusers_facts.formats[0].metadata.extend([
        ("image_model".to_owned(), "flux2".to_owned()),
        ("guidance_embed".to_owned(), "false".to_owned()),
        ("in_channels".to_owned(), "16".to_owned()),
        ("model_layout".to_owned(), "native".to_owned()),
    ]);
    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts)?;
    let registry =
        ModelFamilyRegistry::checked_registrations(&[inpaint::MODEL_FAMILY_REGISTRATION])?;
    let diffusers_resolved = registry.resolve(&diffusers_probe)?;
    assert_eq!(
        diffusers_resolved.detection().identity.feature_id(),
        inpaint::MODEL_FAMILY_FEATURE_ID
    );
    let configuration = inpaint::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(configuration.layout, FluxChromaLayout::Diffusers);
    assert_eq!(configuration.in_channels, 96);
    assert!(configuration.guidance_embedding);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(inpaint::MODEL_FAMILY_SOURCE_PATH)
        )?),
        inpaint::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], inpaint::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(
        provenance["source_symbol"],
        inpaint::MODEL_FAMILY_IDENTIFIER
    );
    assert_eq!(provenance["source_ordinal"], 27);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("FluxInpaint source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("FluxInpaint source_files must be an array")?
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
        .find(|row| row["feature_id"] == inpaint::MODEL_FAMILY_FEATURE_ID)
        .ok_or("FluxInpaint catalog row is missing")?;
    assert_eq!(catalog_row["source_ordinal"], 27);
    assert_eq!(catalog_row["source_symbol"], "FluxInpaint");
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["image_model"],
        "flux"
    );
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["guidance_embed"],
        true
    );
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["in_channels"],
        96
    );
    assert_eq!(
        catalog_row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.Flux"
    );
    assert_eq!(
        sha256(&serde_json::to_vec(catalog_row)?),
        inpaint::MODEL_FAMILY_PROJECTION_SHA256
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/fluxinpaint_sim_model_0079.rs"),
    )?;
    for canonical_adapter in [
        "flux_chroma_family",
        "FLUX_CLIP_TARGET",
        "FLUX_COMPONENT_STATE_SCHEMAS",
        "FLUX_FORWARD_PROGRAM",
        "flux_chroma_configuration_for_probe",
    ] {
        assert!(row_source.contains(canonical_adapter));
    }
    for competing_owner in [
        "generated_flux_comfy_model_0077",
        "pub struct ",
        "pub enum ",
        "ModelStateTransformPlanDefinition",
        "ModelFamilyComponentStateSchema",
        "ModelForwardOperation",
        "ModelClipTargetDefinition",
        "const WEIGHT_RULES",
        "const FORWARD_PROGRAM",
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
    assert!(row_source.contains("FLUX_GUIDANCE_PROJECTION_KEYS"));
    assert!(row_source.contains("source_configuration: &[]"));
    super::write_model_family_row_artifact(
        inpaint::MODEL_FAMILY_FIXTURE,
        inpaint::MODEL_FAMILY_FEATURE_ID,
        inpaint::MODEL_FAMILY_IDENTIFIER,
        inpaint::MODEL_FAMILY_SOURCE_ORDINAL,
        "fluxinpaint_comfy_model_0079",
        &[
            "source-provenance-registration-descriptor",
            "canonical-flux-owner-adapter",
            "model-store-native-unprefixed-and-diffusers-detection",
            "source-exact-channel-and-guidance-precedence",
            "transactional-component-and-scale-mapping",
            "named-forward-checkpoints-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "partial-ambiguous-layout-flux2-and-guidance-failures",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_fluxinpaint_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[inpaint::MODEL_FAMILY_REGISTRATION])?;
    for layout in ["native", "unprefixed"] {
        let probe = probe_through_model_store(layout)?;
        assert_eq!(probe.format_identities(), ["safetensors"]);
        assert!(!probe.metadata.contains_key("image_model"));
        assert!(!probe.metadata.contains_key("guidance_embed"));
        assert!(!probe.metadata.contains_key("in_channels"));
        assert!(!probe.metadata.contains_key("model_layout"));
        let resolved = registry.resolve(&probe)?;
        assert_eq!(
            resolved.detection().identity.feature_id(),
            "COMFY-MODEL-0079"
        );
        assert_eq!(resolved.detection().score, 1_200);
        assert_eq!(resolved.detection().evidence.len(), 2);
        assert!(
            resolved
                .detection()
                .evidence
                .iter()
                .any(|evidence| evidence.contains("AnyTensorDimensionValue"))
        );
        assert!(
            resolved
                .detection()
                .evidence
                .iter()
                .any(|evidence| evidence.contains("AnyKeyPresent"))
        );
        assert_eq!(resolved.source_ordinal(), 27);
        assert_eq!(resolved.profile().latent_identifier, "Flux");
    }

    let probe = probe_through_model_store("native")?;
    assert_eq!(
        probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    let resolved = registry.resolve(&probe)?;
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
    let source = source_tensors(&backend, &context, "native", DType::F32, false, false)?;
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

    let patched =
        model.with_weights(single_patch()?.apply(&backend, model.weights(), &context)?)?;
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
fn val_model_family_row_001_fluxinpaint_precedence_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[inpaint::MODEL_FAMILY_REGISTRATION])?;
    let combined = ModelFamilyRegistry::checked_registrations(&[
        inpaint::MODEL_FAMILY_REGISTRATION,
        base_flux::MODEL_FAMILY_REGISTRATION,
    ])?;
    let inpaint_probe = ModelProbe::from_parsed_facts(parsed_facts(ProbeCase::default()))?;
    let inpaint_detection = combined.detect(&inpaint_probe)?;
    assert_eq!(inpaint_detection.identity.feature_id(), "COMFY-MODEL-0079");
    assert_eq!(inpaint_detection.score, 1_200);
    let base_probe = ModelProbe::from_parsed_facts(parsed_facts(ProbeCase {
        in_channels: 16,
        ..ProbeCase::default()
    }))?;
    let base_detection = combined.detect(&base_probe)?;
    assert_eq!(base_detection.identity.feature_id(), "COMFY-MODEL-0077");
    assert_eq!(base_detection.score, 1_000);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let resolved = registry.resolve(&inpaint_probe)?;
    for dtype in [DType::Bf16, DType::F32] {
        let source = source_tensors(&backend, &context, "native", dtype, false, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert!(
            build_model_family_for_probe(
                &registry,
                &inpaint_probe,
                weights,
                options(dtype, RESOLVED_MEMORY_BYTES),
            )
            .is_ok()
        );
    }
    for dtype in [DType::F16, DType::F64] {
        let source = source_tensors(&backend, &context, "native", dtype, false, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert!(matches!(
            build_model_family_for_probe(
                &registry,
                &inpaint_probe,
                weights,
                options(dtype, RESOLVED_MEMORY_BYTES),
            ),
            Err(ModelFamilyError::UnsupportedDType(rejected)) if rejected == dtype
        ));
    }

    let source = source_tensors(&backend, &context, "native", DType::F32, false, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let mut unsupported_device = options(DType::F32, RESOLVED_MEMORY_BYTES);
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &inpaint_probe, weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let partial_probe = ModelProbe::from_parsed_facts(parsed_facts(ProbeCase {
        omit_final: true,
        ..ProbeCase::default()
    }))?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let partial_source = source_tensors(&backend, &context, "native", DType::F32, true, false)?;
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial_source,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.final_layer.linear.weight"
    ));

    for probe_case in [
        ProbeCase {
            input_width: Some(64),
            ..ProbeCase::default()
        },
        ProbeCase {
            guidance_tensor: false,
            ..ProbeCase::default()
        },
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(probe_case))?;
        assert!(matches!(
            registry.resolve(&probe),
            Err(ModelFamilyError::NoDetectionMatch)
        ));
    }
    let flux2_probe = ModelProbe::from_parsed_facts(parsed_facts(ProbeCase {
        flux2: true,
        ..ProbeCase::default()
    }))?;
    assert!(matches!(
        registry.resolve(&flux2_probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Flux2")
    ));

    let mut unexpected_facts = parsed_facts(ProbeCase::default());
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: "F32".to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected_source =
        source_tensors(&backend, &context, "native", DType::F32, false, false)?;
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

    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&inpaint_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
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

#[derive(Clone, Copy)]
struct ProbeCase<'a> {
    layout: &'a str,
    in_channels: u64,
    input_width: Option<u64>,
    guidance_tensor: bool,
    omit_final: bool,
    flux2: bool,
}

impl Default for ProbeCase<'_> {
    fn default() -> Self {
        Self {
            layout: "native",
            in_channels: 96,
            input_width: None,
            guidance_tensor: true,
            omit_final: false,
            flux2: false,
        }
    }
}

fn parsed_facts(probe_case: ProbeCase<'_>) -> ModelParsedFacts {
    let prefix = if probe_case.layout == "unprefixed" {
        ""
    } else {
        "model.diffusion_model."
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in model_shapes(
        probe_case.in_channels,
        probe_case.input_width,
        probe_case.guidance_tensor,
        probe_case.omit_final,
        probe_case.flux2,
    ) {
        tensors.insert(
            format!("{prefix}{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: "F32".to_owned(),
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
                storage_dtype: "F32".to_owned(),
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

fn diffusers_facts() -> ModelParsedFacts {
    let mut tensors = BTreeMap::new();
    for (key, shape) in diffusers_shapes() {
        tensors.insert(
            key,
            ModelParsedTensorFact {
                shape,
                storage_dtype: DType::F32.catalog_name().to_owned(),
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
        ("x_embedder.weight".to_owned(), vec![128, 384]),
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
        (
            "time_text_embed.guidance_embedder.linear_1.weight".to_owned(),
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
    unexpected: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(96, None, true, omit_final, false) {
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
    if unexpected {
        source.insert(
            "unexpected.weight".to_owned(),
            tensor(backend, &[1], &[1.0], dtype, context)?,
        );
    }
    Ok(source)
}

fn model_shapes(
    in_channels: u64,
    input_width: Option<u64>,
    guidance_tensor: bool,
    omit_final: bool,
    flux2: bool,
) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "double_blocks.0.img_attn.norm.key_norm.scale".to_owned(),
            vec![128],
        ),
        (
            "img_in.weight".to_owned(),
            vec![128, input_width.unwrap_or(in_channels * 4)],
        ),
        ("txt_in.weight".to_owned(), vec![128, 2]),
        ("vector_in.in_layer.weight".to_owned(), vec![128, 2]),
        (
            "double_blocks.0.img_attn.proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("single_blocks.0.linear2.weight".to_owned(), vec![2, 2]),
    ];
    if guidance_tensor {
        shapes.push(("guidance_in.in_layer.weight".to_owned(), vec![128, 2]));
    }
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
    let model_path = directory.path().join("fluxinpaint.safetensors");
    write_safetensors(&model_path, layout)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "fluxinpaint-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("fluxinpaint-row", "fluxinpaint.safetensors")?;
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
    let mut shapes = model_shapes(96, None, true, false, false);
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
        .ok_or("FluxInpaint checkpoint is missing")?;
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

fn single_patch() -> Result<PatchGraph, Box<dyn std::error::Error>> {
    Ok(PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "fluxinpaint-single-stream-delta".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.single_blocks.0.linear2.weight".to_owned(),
                expected_shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
                application: PatchApplication::Add,
            }],
        }],
    )?)
}

fn ordered_patch_graph(replace_first: bool) -> Result<PatchGraph, Box<dyn std::error::Error>> {
    let replacement = PatchOperation {
        identifier: "fluxinpaint-ordered-replacement".to_owned(),
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
        identifier: "fluxinpaint-ordered-addition".to_owned(),
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
        .join(inpaint::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
