use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipConfigurationFact, ModelClipModelInvocation,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe,
    ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits, PatchApplication,
    PatchGraph, PatchKind, PatchOperation, PatchTarget, build_model_family_for_probe,
    describe_model_family, generated_flux2_sim_model_0078 as flux2,
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

const ARTIFACT_DIGEST: &str = "0670670670670670670670670670670670670670670670670670670670670670";
const RESOLVED_MEMORY_BYTES: u64 = 739_470;

#[derive(Clone, Copy)]
enum ClipFixture {
    None,
    Qwen4,
    Qwen8,
    Qwen4And8,
    MistralPruned,
    MistralFull,
    QuantizedQwen4,
}

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9078",
    identifier: "Flux2AmbiguousFixture",
    ..flux2::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    flux2::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 180,
        source_architecture: "model_base.Flux2AmbiguousFixture",
        ..flux2::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_flux2_source_projection_descriptor_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(flux2::MODEL_FAMILY_IDENTIFIER, "Flux2");
    assert_eq!(flux2::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0078");
    assert_eq!(flux2::MODEL_FAMILY_FIXTURE, "flux2-comfy-model-0078");
    assert_eq!(flux2::MODEL_FAMILY_SOURCE_ORDINAL, 80);
    assert_eq!(flux2::MODEL_FAMILY_REGISTRATION.source_ordinal, 80);
    assert_eq!(
        flux2::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Flux2"
    );
    assert!(flux2::MODEL_FAMILY_REGISTRATION.source_configuration.is_empty());
    assert_eq!(flux2::MODEL_FAMILY_SAMPLING_SHIFT, 2.02);
    assert_eq!(flux2::MODEL_FAMILY_INHERITED_MEMORY_USAGE_FACTOR, 3.1);
    assert!((flux2::memory_usage_factor(3_072) - 14.628_571_428_571_428).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&flux2::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Flux2");
    assert_eq!(descriptor.family, "COMFY-MODEL-0078");
    assert_eq!(descriptor.architecture_version, "flux2-transformer-v1");
    assert_eq!(descriptor.latent_format, "Flux2");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 15);

    let mut diffusers_facts = diffusers_facts(DType::F32);
    diffusers_facts.formats[0].metadata.extend([
        ("image_model".to_owned(), "flux".to_owned()),
        ("guidance_embed".to_owned(), "true".to_owned()),
        ("in_channels".to_owned(), "96".to_owned()),
        ("model_layout".to_owned(), "native".to_owned()),
    ]);
    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts)?;
    let registry =
        ModelFamilyRegistry::checked_registrations(&[flux2::MODEL_FAMILY_REGISTRATION])?;
    let diffusers_resolved = registry.resolve(&diffusers_probe)?;
    assert_eq!(
        diffusers_resolved.detection().identity.feature_id(),
        flux2::MODEL_FAMILY_FEATURE_ID
    );
    let configuration = flux2::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(configuration.layout, sim_model::FluxChromaLayout::Diffusers);
    assert_eq!(configuration.variant, sim_model::FluxChromaVariant::Flux2);
    assert_eq!(configuration.in_channels, 16);

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(flux2::MODEL_FAMILY_SOURCE_PATH)
        )?),
        flux2::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], flux2::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], flux2::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], 80);
    assert_eq!(provenance["latent_feature_id"], "COMFY-MODEL-0030");
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("Flux2 source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("Flux2 source_files must be an array")?
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
        .find(|row| row["feature_id"] == flux2::MODEL_FAMILY_FEATURE_ID)
        .ok_or("Flux2 catalog row is missing")?;
    assert_eq!(catalog_row["source_ordinal"], 80);
    assert_eq!(catalog_row["source_symbol"], "Flux2");
    assert_eq!(
        catalog_row["static"]["unet_config"]["value"]["image_model"],
        "flux2"
    );
    assert_eq!(
        catalog_row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.Flux2"
    );
    assert_eq!(
        catalog_row["clip_target"]["calls"].as_array().map(Vec::len),
        Some(3)
    );
    assert_eq!(
        sha256(&serde_json::to_vec(catalog_row)?),
        flux2::MODEL_FAMILY_PROJECTION_SHA256
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/flux2_sim_model_0078.rs"),
    )?;
    for canonical_adapter in [
        "flux_chroma_configuration_for_probe",
        "FLUX_STATE_PLAN_CASES",
        "FLUX_COMPONENT_STATE_SCHEMAS",
        "FLUX_FORWARD_PROGRAM",
        "FLUX_MEMORY_USAGE_FACTOR",
        "MemoryEstimatorDescriptor",
        "ModelProbe",
    ] {
        assert!(
            row_source.contains(canonical_adapter),
            "missing {canonical_adapter}"
        );
    }
    for competing_owner in [
        "pub struct ",
        "pub enum ",
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "ModelStateTransformPlanDefinition",
        "std::fs",
        "unsafe ",
    ] {
        assert!(
            !row_source.contains(competing_owner),
            "found {competing_owner}"
        );
    }
    assert!(!row_source.contains("ModelDetectionRule::Metadata"));
    assert!(!row_source.contains("ModelSourceConfigurationRule"));
    assert!(row_source.contains("FLUX_INPUT_PROJECTION_KEYS"));
    assert!(row_source.contains("FLUX2_DISCRIMINATOR_KEYS"));
    assert!(row_source.contains("source_configuration: &[]"));
    super::write_model_family_row_artifact(
        flux2::MODEL_FAMILY_FIXTURE,
        flux2::MODEL_FAMILY_FEATURE_ID,
        flux2::MODEL_FAMILY_IDENTIFIER,
        flux2::MODEL_FAMILY_SOURCE_ORDINAL,
        "flux2_comfy_model_0078",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-unprefixed-and-diffusers-detection",
            "canonical-flux-chroma-configuration-and-state-mapping",
            "source-exact-dynamic-clip-precedence-and-pruning",
            "named-forward-checkpoints-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-cross-family-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_flux2_mapping_forward_patch_memory_and_clip()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[flux2::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store("native", ClipFixture::Qwen4)?;
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
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0078"
    );
    assert_eq!(resolved.source_ordinal(), 80);
    assert_eq!(resolved.detection().score, 1_000);
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
    assert_eq!(resolved.profile().latent_identifier, "Flux2");
    assert_eq!(resolved.profile().memory_estimator.bytes_per_parameter, 15);
    assert_clip(&resolved, "KleinTokenizer", "klein_te", "qwen3_4b", false)?;

    let configuration = flux2::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, sim_model::FluxChromaVariant::Flux2);
    assert_eq!(configuration.in_channels, 16);
    assert_eq!(configuration.out_channels, 128);
    assert_eq!(configuration.patch_size, 1);
    assert_eq!(configuration.hidden_size, 3_072);
    assert_eq!(configuration.context_input_dimension, 4_096);
    assert_eq!(configuration.attention_heads, 24);
    assert_eq!(configuration.double_block_count, 1);
    assert_eq!(configuration.single_block_count, 1);
    assert!(!configuration.guidance_embedding);

    for (clip, tokenizer, model_type, pruned) in [
        (ClipFixture::Qwen8, "KleinTokenizer8B", "qwen3_8b", false),
        (
            ClipFixture::MistralPruned,
            "Flux2Tokenizer",
            "mistral3_24b",
            true,
        ),
        (
            ClipFixture::MistralFull,
            "Flux2Tokenizer",
            "mistral3_24b",
            false,
        ),
        (
            ClipFixture::QuantizedQwen4,
            "KleinTokenizer",
            "qwen3_4b",
            false,
        ),
    ] {
        let candidate_probe =
            ModelProbe::from_parsed_facts(parsed_facts("native", DType::F32, false, false, clip))?;
        let candidate = registry.resolve(&candidate_probe)?;
        assert_clip(
            &candidate,
            tokenizer,
            if tokenizer == "Flux2Tokenizer" {
                "flux2_te"
            } else {
                "klein_te"
            },
            model_type,
            pruned,
        )?;
    }
    let precedence_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "native",
        DType::F32,
        false,
        false,
        ClipFixture::Qwen4And8,
    ))?;
    assert_clip(
        &registry.resolve(&precedence_probe)?,
        "KleinTokenizer",
        "klein_te",
        "qwen3_4b",
        false,
    )?;
    let no_clip_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "native",
        DType::F32,
        false,
        false,
        ClipFixture::None,
    ))?;
    assert!(
        registry
            .resolve(&no_clip_probe)?
            .clip_target()
            .candidates()
            .is_empty()
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(
        &backend,
        &context,
        "native",
        DType::F32,
        false,
        ClipFixture::Qwen4,
    )?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("model").is_some_and(|model| {
        model.contains_key("native.double_blocks.0.img_attn.norm.key_norm.weight")
            && model.contains_key("native.double_stream_modulation_img.lin.weight")
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
    assert_eq!(model.memory_estimate().parameter_elements, 49_296);
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

    let patched = model.with_weights(ordered_patch_graph(true)?.apply(
        &backend,
        model.weights(),
        &context,
    )?)?;
    assert_checkpoint(
        &backend,
        &context,
        &patched.forward_checkpoints(&backend, &input, &context)?,
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
            &context
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            &second.tensors()["native.single_blocks.0.linear2.weight"],
            &context
        )?
    );
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, RESOLVED_MEMORY_BYTES - 1)),
        Err(ModelFamilyError::OutOfMemory { required: RESOLVED_MEMORY_BYTES, budget })
            if budget == RESOLVED_MEMORY_BYTES - 1
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_flux2_unprefixed_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[flux2::MODEL_FAMILY_REGISTRATION])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let probe = probe_through_model_store("unprefixed", ClipFixture::MistralPruned)?;
    assert_eq!(probe.unet_prefix_selection()?.prefix(), "model.");
    let resolved = registry.resolve(&probe)?;
    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let source = source_tensors(
            &backend,
            &context,
            "unprefixed",
            dtype,
            false,
            ClipFixture::MistralPruned,
        )?;
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
                options(dtype, RESOLVED_MEMORY_BYTES)
            )
            .is_ok()
        );
    }

    let source = source_tensors(
        &backend,
        &context,
        "unprefixed",
        DType::F32,
        false,
        ClipFixture::MistralPruned,
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
            options(DType::F64, RESOLVED_MEMORY_BYTES)
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

    let partial_probe = ModelProbe::from_parsed_facts(parsed_facts(
        "native",
        DType::F32,
        true,
        false,
        ClipFixture::None,
    ))?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let partial_source = source_tensors(
        &backend,
        &context,
        "native",
        DType::F32,
        true,
        ClipFixture::None,
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

    let malformed = ModelProbe::from_parsed_facts(parsed_facts(
        "native",
        DType::F32,
        false,
        true,
        ClipFixture::None,
    ))?;
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    let flux_probe = ModelProbe::from_parsed_facts(flux_facts())?;
    assert!(matches!(
        flux2::configuration_for_probe(&flux_probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("expected Flux2")
    ));

    let mut unexpected_facts = parsed_facts("native", DType::F32, false, false, ClipFixture::None);
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
        "native",
        DType::F32,
        false,
        ClipFixture::None,
    )?;
    unexpected_source.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context), ARTIFACT_DIGEST, &unexpected_source,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let no_match = ModelProbe {
        tensor_shapes: BTreeMap::new(),
        metadata: BTreeMap::from([("image_model".to_owned(), "flux".to_owned())]),
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

fn assert_clip(
    resolved: &sim_model::ResolvedModelFamily,
    tokenizer_suffix: &str,
    model_suffix: &str,
    detection_name: &str,
    pruned: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert!(
        candidates[0]
            .tokenizer()
            .identifier()
            .ends_with(tokenizer_suffix)
    );
    assert!(
        candidates[0]
            .clip_model()
            .target()
            .as_str()
            .ends_with(model_suffix)
    );
    let ModelClipModelInvocation::Factory { configuration } =
        candidates[0].clip_model().invocation()
    else {
        return Err("Flux2 CLIP model must be a factory".into());
    };
    assert!(configuration.iter().any(|fact| matches!(
        fact,
        ModelClipConfigurationFact::Expand { source } if source.as_str().ends_with(detection_name)
    )));
    assert_eq!(
        configuration.iter().any(|fact| matches!(
            fact,
            ModelClipConfigurationFact::Bind { parameter, source }
                if parameter.as_str() == "pruned" && source.as_str() == "true"
        )),
        pruned
    );
    Ok(())
}

fn parsed_facts(
    layout: &str,
    dtype: DType,
    omit_final: bool,
    malformed_input: bool,
    clip: ClipFixture,
) -> ModelParsedFacts {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensors = BTreeMap::new();
    for (key, shape) in model_shapes(omit_final, malformed_input) {
        tensors.insert(
            format!("{prefix}{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for (key, shape) in clip_shapes(clip) {
        tensors.insert(
            key.to_owned(),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for (key, shape) in [
        ("vae.decoder.weight", vec![1]),
        ("text_encoders.shared.weight", vec![1]),
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
        ("x_embedder.weight".to_owned(), vec![128, 16]),
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
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight".to_owned(),
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
        ("proj_out.weight".to_owned(), vec![2, 2]),
    ];
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn flux_facts() -> ModelParsedFacts {
    let mut facts = parsed_facts("native", DType::F32, false, false, ClipFixture::None);
    facts
        .tensors
        .remove("model.diffusion_model.double_stream_modulation_img.lin.weight");
    facts
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: &str,
    dtype: DType,
    omit_final: bool,
    clip: ClipFixture,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(omit_final, false) {
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
    for (key, shape) in clip_shapes(clip) {
        source.insert(
            key.to_owned(),
            tensor(
                backend,
                &shape,
                &vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
                dtype,
                context,
            )?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    source.insert(
        "text_encoders.shared.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn model_shapes(omit_final: bool, malformed_input: bool) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "double_blocks.0.img_attn.norm.key_norm.scale".to_owned(),
            vec![128],
        ),
        (
            "img_in.weight".to_owned(),
            if malformed_input {
                vec![3_072]
            } else {
                vec![3_072, 16]
            },
        ),
        (
            "double_stream_modulation_img.lin.weight".to_owned(),
            vec![2, 2],
        ),
        (
            "double_blocks.0.img_attn.proj.weight".to_owned(),
            vec![2, 2],
        ),
        ("single_blocks.0.linear2.weight".to_owned(), vec![2, 2]),
    ];
    if !omit_final {
        shapes.push(("final_layer.linear.weight".to_owned(), vec![2, 2]));
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn clip_shapes(clip: ClipFixture) -> Vec<(&'static str, Vec<u64>)> {
    match clip {
        ClipFixture::None => Vec::new(),
        ClipFixture::Qwen4 => vec![(
            "text_encoders.qwen3_4b.transformer.model.norm.weight",
            vec![1],
        )],
        ClipFixture::Qwen8 => vec![(
            "text_encoders.qwen3_8b.transformer.model.layers.0.input_layernorm.weight",
            vec![1],
        )],
        ClipFixture::Qwen4And8 => vec![
            (
                "text_encoders.qwen3_4b.transformer.model.norm.weight",
                vec![1],
            ),
            (
                "text_encoders.qwen3_8b.transformer.model.norm.weight",
                vec![1],
            ),
        ],
        ClipFixture::MistralPruned => vec![(
            "text_encoders.mistral3_24b.transformer.model.norm.weight",
            vec![1],
        )],
        ClipFixture::MistralFull => vec![
            (
                "text_encoders.mistral3_24b.transformer.model.norm.weight",
                vec![1],
            ),
            (
                "text_encoders.mistral3_24b.transformer.model.layers.39.post_attention_layernorm.weight",
                vec![1],
            ),
        ],
        ClipFixture::QuantizedQwen4 => vec![(
            "text_encoders.qwen3_4b.transformer.model.layers.0.sim_quant",
            vec![1],
        )],
    }
}

fn probe_through_model_store(
    layout: &str,
    clip: ClipFixture,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("flux2.safetensors");
    write_safetensors(&model_path, layout, clip)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "flux2-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("flux2-row", "flux2.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(
    path: &Path,
    layout: &str,
    clip: ClipFixture,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(false, false)
        .into_iter()
        .map(|(key, shape)| (format!("{prefix}{key}"), shape))
        .collect::<Vec<_>>();
    shapes.extend(
        clip_shapes(clip)
            .into_iter()
            .map(|(key, shape)| (key.to_owned(), shape)),
    );
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.shared.weight".to_owned(), vec![1]),
    ]);
    let mut data = Vec::new();
    for (name, shape) in shapes {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or("fixture shape overflow")
        })?;
        for _ in 0..elements {
            data.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        header.insert(name, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
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
        .ok_or("Flux2 checkpoint is missing")?;
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
        identifier: "flux2-ordered-replacement".to_owned(),
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
        identifier: "flux2-ordered-addition".to_owned(),
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
        .join(flux2::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
