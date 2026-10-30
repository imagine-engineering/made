use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, HUNYUANDIT_BASE_EXTRA_INPUT,
    HUNYUANDIT_DIT1_EXTRA_INPUT, HUNYUANDIT_G_DEPTH,
    HUNYUANDIT_G_HIDDEN_SIZE, HUNYUANDIT_G_MLP_RATIO, HUNYUANDIT_LATENT_FORMAT,
    HUNYUANDIT_LINEAR_END, HUNYUANDIT_MEMORY_USAGE_FACTOR, HUNYUANDIT_NUMBER_OF_HEADS,
    HUNYUANDIT_SUPPORTED_DEVICES, HUNYUANDIT_SUPPORTED_DTYPES, HUNYUANDIT1_LINEAR_END,
    HunyuanDiTAttentionPrecision, HunyuanDiTConfiguration, HunyuanDiTLayout,
    HunyuanDiTVariant, ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration,
    ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact,
    ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits,
    PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family, describe_model_family,
    generated_hunyuandit_sim_model_0087 as row_dit,
    generated_hunyuandit1_sim_model_0088 as row_dit1, hunyuandit_configuration_for_probe,
    hunyuandit_state_plan_for_layout, map_model_weights,
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

#[derive(Clone, Copy)]
pub(super) struct RowSpec {
    pub feature_id: &'static str,
    pub identifier: &'static str,
    pub fixture: &'static str,
    pub module: &'static str,
    pub source_ordinal: u16,
    pub architecture_version: &'static str,
    pub image_model: &'static str,
    pub projection_sha256: &'static str,
    pub variant: HunyuanDiTVariant,
    pub expected_memory: u64,
}

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9087",
    identifier: "HunyuanDiTAmbiguousFixture",
    ..row_dit::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_dit::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 125,
        source_architecture: "model_base.HunyuanDiTAmbiguousFixture",
        ..row_dit::MODEL_FAMILY_REGISTRATION
    },
];

const SPEC: RowSpec = RowSpec {
    feature_id: row_dit::MODEL_FAMILY_FEATURE_ID,
    identifier: row_dit::MODEL_FAMILY_IDENTIFIER,
    fixture: row_dit::MODEL_FAMILY_FIXTURE,
    module: "hunyuandit_comfy_model_0087",
    source_ordinal: row_dit::MODEL_FAMILY_SOURCE_ORDINAL,
    architecture_version: "hunyuandit-v-prediction-transformer-v1",
    image_model: "hydit",
    projection_sha256: row_dit::MODEL_FAMILY_PROJECTION_SHA256,
    variant: HunyuanDiTVariant::DiT,
    expected_memory: 58,
};

#[test]
fn val_model_family_row_001_hunyuandit_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuandit_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuandit_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}

pub(super) fn assert_source_contract(spec: RowSpec) -> Result<(), Box<dyn std::error::Error>> {
    let registration = registration(spec.variant);
    assert_eq!(registration.definition.feature_id, spec.feature_id);
    assert_eq!(registration.definition.identifier, spec.identifier);
    assert_eq!(registration.source_ordinal, spec.source_ordinal);
    assert_eq!(registration.source_architecture, "model_base.HunyuanDiT");
    assert!(registration.source_configuration.is_empty());

    let descriptor = describe_model_family(registration.definition)?;
    assert_eq!(descriptor.architecture_version, spec.architecture_version);
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 2);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 2);
    assert_eq!(registration.definition.latent_feature_id, "COMFY-MODEL-0047");
    assert_eq!(registration.definition.components.len(), 3);
    assert_eq!(registration.definition.clip_target.candidates.len(), 1);
    assert_eq!(
        registration.definition.clip_target.candidates[0].tokenizer,
        "sim.text_encoders.hydit.HyditTokenizer"
    );
    assert_eq!(
        registration.definition.clip_target.candidates[0].clip_model,
        "sim.text_encoders.hydit.HyditModel"
    );
    assert_eq!(HUNYUANDIT_MEMORY_USAGE_FACTOR, 1.3);
    assert_eq!(HUNYUANDIT_NUMBER_OF_HEADS, 16);
    assert_eq!(HUNYUANDIT_SUPPORTED_DTYPES.len(), 3);
    assert_eq!(HUNYUANDIT_SUPPORTED_DEVICES, &[DeviceKind::Cpu]);
    assert_eq!(HUNYUANDIT_LATENT_FORMAT.feature_id, "COMFY-MODEL-0047");

    let registry = all_rows_registry()?;
    for layout in layouts() {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32, false))?;
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), spec.feature_id);
        assert_eq!(resolved.detection().score, 1_000);
        assert_eq!(resolved.detection().evidence.len(), 3);
        assert!(
            resolved
                .detection()
                .evidence
                .iter()
                .all(|evidence| evidence.contains("AnyTensorFact"))
        );
        assert_eq!(resolved.profile().latent_feature_id, "COMFY-MODEL-0047");
        assert_eq!(resolved.profile().latent_identifier, "SDXL");
        let configuration = hunyuandit_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, spec.variant);
        assert_eq!(configuration.layout, layout);
        assert_shape_reduced_configuration(spec, configuration);
    }

    let exact = ModelProbe::from_parsed_facts(parsed_facts(
        spec,
        HunyuanDiTLayout::StandaloneNative,
        DType::F32,
        true,
    ))?;
    assert_source_exact_configuration(spec, hunyuandit_configuration_for_probe(&exact)?);

    let mut misleading = parsed_facts(spec, HunyuanDiTLayout::PrefixedNative, DType::F32, false);
    misleading.formats[0]
        .metadata
        .insert("image_model".to_owned(), "pixart_sigma".to_owned());
    misleading.formats[0]
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    let misleading = ModelProbe::from_parsed_facts(misleading)?;
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        spec.feature_id
    );

    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "transformer_blocks.0.attn1.to_q.weight".to_owned(),
                vec![1_152, 1_152],
            ),
            ("pos_embed.proj.weight".to_owned(), vec![1_152, 4, 2, 2]),
        ]),
        metadata: BTreeMap::from([("image_model".to_owned(), spec.image_model.to_owned())]),
    };
    assert!(matches!(
        registry.detect(&diffusers),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let store_probe = probe_through_model_store(spec)?;
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        spec.feature_id
    );
    for key in ["image_model", "model_layout", "depth", "variant"] {
        assert!(!store_probe.metadata.contains_key(key));
    }

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory(spec).join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("HunyuanDiT source projection must be a string")?;
    assert_eq!(sha256(projection.as_bytes()), spec.projection_sha256);
    assert_eq!(provenance["source_projection_sha256"], spec.projection_sha256);
    for source in provenance["source_files"]
        .as_array()
        .ok_or("HunyuanDiT source_files must be an array")?
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
        .find(|row| row["feature_id"] == spec.feature_id)
        .ok_or("HunyuanDiT catalog row is missing")?;
    assert_eq!(row["source_ordinal"], spec.source_ordinal);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], spec.image_model);
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 1.3);
    assert_eq!(
        row["static"]["latent_format"]["value"]["symbol"],
        "latent_formats.SDXL"
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families")
            .join(format!("{}.rs", spec.module)),
    )?;
    for canonical_import in [
        "HUNYUANDIT_CLIP_TARGET",
        "HUNYUANDIT_COMPONENTS",
        "HUNYUANDIT_FORWARD_PROGRAM",
        "HUNYUANDIT_PREFIXED_STATE_PLAN",
        "HUNYUANDIT_SAVED_MODEL_STATE_PLAN",
        "HUNYUANDIT_STANDALONE_STATE_PLAN",
        "HUNYUANDIT_SUPPORTED_DTYPES",
        "hunyuandit_configuration_for_probe",
    ] {
        assert!(row_source.contains(canonical_import), "{canonical_import}");
    }
    assert_eq!(row_source.matches("ModelDetectionRule::AnyTensorFact").count(), 3);
    for forbidden in [
        "ModelDetectionRule::Metadata",
        "ModelSourceConfigurationRule",
        "MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION",
        "struct HunyuanDiTConfiguration",
        "struct ModelStore",
        "struct ModelProbe",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(forbidden), "{forbidden}");
    }

    super::write_model_family_row_artifact(
        spec.fixture,
        spec.feature_id,
        spec.identifier,
        spec.source_ordinal,
        spec.module,
        &[
            "source-provenance-registration-and-canonical-ownership",
            "any-tensor-fact-prefixed-saved-and-standalone-detection",
            "variant-configuration-conditioning-and-latent-identity",
            "misleading-metadata-and-unsupported-diffusers-rejection",
            "transactional-layout-and-component-routing",
            "model-store-forward-patch-memory-and-oom",
            "bf16-f16-f32-cpu-and-fail-closed-device",
            "partial-mixed-ambiguous-unexpected-and-cancellation-failures",
        ],
    )?;
    Ok(())
}

pub(super) fn assert_execution_contract(
    spec: RowSpec,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = all_rows_registry()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16 * 1024 * 1024)?,
        &cancellation,
    );

    for layout in layouts() {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32, false))?;
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), spec.feature_id);
        let source = mapping_tensors(&backend, &context, spec, layout, DType::F32, None)?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuandit_state_plan_for_layout(layout).compile()?,
            artifact_digest(spec),
            &source,
        )?;
        assert_eq!(mapped.components().len(), 3);
        let model = mapped.component("model").ok_or("missing HunyuanDiT model")?;
        assert!(model.keys().all(|key| key.starts_with("native.")));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }

    let weights = legacy_weights(&backend, &context, spec, DType::F32, None)?;
    let model = build_model_family(
        registration(spec.variant).definition,
        weights,
        options(DType::F32, DeviceKind::Cpu, spec.expected_memory),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, spec.expected_memory);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "conditioning.t5_projection",
        &[1.0, -1.0],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "conditioning.extra_projection",
        &[0.9810586, -0.5189414],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "transformer.block_0_attention",
        &[0.9810586, -0.5189414],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "output.final_projection",
        &[1.0810586, -0.6189414],
    )?;

    let patch = PatchGraph::checked(
        artifact_digest(spec),
        vec![PatchOperation {
            identifier: format!("{}-output-bias", spec.fixture),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.final_layer.linear.bias".to_owned(),
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
        "output.final_projection",
        &[1.5810586, -0.1189414],
    )?;

    let weights = legacy_weights(&backend, &context, spec, DType::F32, None)?;
    assert!(matches!(
        build_model_family(
            registration(spec.variant).definition,
            weights,
            options(
                DType::F32,
                DeviceKind::Cpu,
                spec.expected_memory.saturating_sub(1)
            ),
        ),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == spec.expected_memory && budget == spec.expected_memory - 1
    ));
    Ok(())
}

pub(super) fn assert_failure_contract(
    spec: RowSpec,
    ambiguous_registrations: &'static [ModelFamilyRegistration],
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = all_rows_registry()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16 * 1024 * 1024)?,
        &cancellation,
    );
    let layout = HunyuanDiTLayout::StandaloneNative;

    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let weights = legacy_weights(&backend, &context, spec, dtype, None)?;
        build_model_family(
            registration(spec.variant).definition,
            weights,
            options(dtype, DeviceKind::Cpu, spec.expected_memory),
        )?;
    }

    let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32, false))?;
    let resolved = registry.resolve(&probe)?;
    let source = mapping_tensors(&backend, &context, spec, layout, DType::F32, None)?;
    let weights = legacy_weights(&backend, &context, spec, DType::F32, None)?;
    assert!(matches!(
        build_model_family(
            registration(spec.variant).definition,
            weights,
            options(DType::F32, DeviceKind::Metal, spec.expected_memory),
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let partial_source =
        mapping_tensors(&backend, &context, spec, layout, DType::F32, Some("mlp_t5.0.weight"))?;
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            artifact_digest(spec),
            &partial_source,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(_))
    ));

    let missing = legacy_weights(
        &backend,
        &context,
        spec,
        DType::F32,
        Some("final_layer.linear.bias"),
    );
    assert!(matches!(
        missing,
        Err(ModelFamilyError::MissingRequiredKey(key))
            if key == "native.final_layer.linear.bias"
    ));

    let mut unexpected = source.clone();
    unexpected.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &hunyuandit_state_plan_for_layout(layout).compile()?,
            artifact_digest(spec),
            &unexpected,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let ambiguous = ModelFamilyRegistry::checked_registrations(ambiguous_registrations)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let mut mixed = parsed_facts(
        spec,
        HunyuanDiTLayout::PrefixedNative,
        DType::F32,
        false,
    );
    mixed.tensors.extend(
        parsed_facts(
            spec,
            HunyuanDiTLayout::StandaloneNative,
            DType::F32,
            false,
        )
        .tensors,
    );
    assert!(matches!(
        registry.resolve(&ModelProbe::from_parsed_facts(mixed)?),
        Err(ModelFamilyError::ModelLayoutSelection(_))
            | Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));

    let mut partial = parsed_facts(spec, layout, DType::F32, false);
    partial
        .tensors
        .remove(&format!("{}extra_embedder.0.weight", layout_prefix(layout)));
    let partial = ModelProbe::from_parsed_facts(partial)?;
    assert!(matches!(
        registry.detect(&partial),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    assert!(matches!(
        hunyuandit_configuration_for_probe(&partial),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("partial")
    ));

    let cross_family = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("y_embedder.y_embedding".to_owned(), vec![1, 1_152]),
            ("adaln_single.emb.timestep_embedder.linear_1.weight".to_owned(), vec![2, 2]),
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
        ModelStateTransaction::new(&backend, &cancelled_context).execute(
            &hunyuandit_state_plan_for_layout(layout).compile()?,
            artifact_digest(spec),
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

fn all_rows_registry() -> Result<ModelFamilyRegistry, ModelFamilyError> {
    ModelFamilyRegistry::checked_registrations(&[
        row_dit::MODEL_FAMILY_REGISTRATION,
        row_dit1::MODEL_FAMILY_REGISTRATION,
    ])
}

fn registration(variant: HunyuanDiTVariant) -> ModelFamilyRegistration {
    match variant {
        HunyuanDiTVariant::DiT => row_dit::MODEL_FAMILY_REGISTRATION,
        HunyuanDiTVariant::DiT1 => row_dit1::MODEL_FAMILY_REGISTRATION,
    }
}

fn layouts() -> [HunyuanDiTLayout; 3] {
    [
        HunyuanDiTLayout::PrefixedNative,
        HunyuanDiTLayout::SavedModel,
        HunyuanDiTLayout::StandaloneNative,
    ]
}

fn parsed_facts(
    spec: RowSpec,
    layout: HunyuanDiTLayout,
    dtype: DType,
    source_exact: bool,
) -> ModelParsedFacts {
    let prefix = layout_prefix(layout);
    let tensors = detector_shapes(spec, source_exact)
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
        .collect();
    ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata: BTreeMap::new(),
        }],
    }
}

fn detector_shapes(spec: RowSpec, source_exact: bool) -> Vec<(String, Vec<u64>)> {
    let (hidden_size, depth) = if source_exact {
        match spec.variant {
            HunyuanDiTVariant::DiT => (1_152, 28),
            HunyuanDiTVariant::DiT1 => (HUNYUANDIT_G_HIDDEN_SIZE, HUNYUANDIT_G_DEPTH),
        }
    } else {
        match spec.variant {
            HunyuanDiTVariant::DiT => (32, 2),
            HunyuanDiTVariant::DiT1 => (HUNYUANDIT_G_HIDDEN_SIZE, 2),
        }
    };
    let extra_input = match spec.variant {
        HunyuanDiTVariant::DiT => HUNYUANDIT_BASE_EXTRA_INPUT,
        HunyuanDiTVariant::DiT1 => HUNYUANDIT_DIT1_EXTRA_INPUT,
    };
    let mut shapes = vec![
        ("mlp_t5.0.weight".to_owned(), vec![8_192, 2_048]),
        ("mlp_t5.2.weight".to_owned(), vec![1_024, 8_192]),
        (
            "x_embedder.proj.weight".to_owned(),
            vec![hidden_size, 4, 2, 2],
        ),
        (
            "extra_embedder.0.weight".to_owned(),
            vec![hidden_size * 4, extra_input],
        ),
    ];
    for ordinal in 0..depth {
        shapes.push((
            format!("blocks.{ordinal}.attn1.qkv.weight"),
            vec![hidden_size * 3, hidden_size],
        ));
    }
    shapes
}

fn mapping_shapes(spec: RowSpec) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("mlp_t5.0.weight".to_owned(), vec![2, 2]),
        ("mlp_t5.0.bias".to_owned(), vec![2]),
        ("mlp_t5.2.weight".to_owned(), vec![2, 2]),
        ("text_embedding_padding".to_owned(), vec![1]),
        ("pooler.q_proj.weight".to_owned(), vec![1]),
        ("x_embedder.proj.weight".to_owned(), vec![1]),
        ("t_embedder.mlp.0.weight".to_owned(), vec![1]),
        ("extra_embedder.0.weight".to_owned(), vec![2, 2]),
        ("extra_embedder.0.bias".to_owned(), vec![2]),
        ("blocks.0.attn1.qkv.weight".to_owned(), vec![1]),
        ("final_layer.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.bias".to_owned(), vec![2]),
    ];
    if spec.variant == HunyuanDiTVariant::DiT1 {
        shapes.insert(5, ("style_embedder.weight".to_owned(), vec![1]));
    }
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.hydit.weight".to_owned(), vec![1]),
    ]);
    shapes
}

fn mapping_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    spec: RowSpec,
    layout: HunyuanDiTLayout,
    dtype: DType,
    omitted: Option<&str>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let mut source = BTreeMap::new();
    for (key, shape) in mapping_shapes(spec) {
        if omitted == Some(key.as_str()) {
            continue;
        }
        let source_key = if key.starts_with("vae.") || key.starts_with("text_encoders.") {
            key.clone()
        } else {
            format!("{prefix}{key}")
        };
        let elements = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key == "mlp_t5.0.bias" {
            vec![1.0, -1.0]
        } else if key == "extra_embedder.0.bias" {
            vec![0.25, -0.25]
        } else if key == "final_layer.linear.bias" {
            vec![0.1, -0.1]
        } else if shape == [2, 2]
            && matches!(
                key.as_str(),
                "mlp_t5.0.weight"
                    | "mlp_t5.2.weight"
                    | "extra_embedder.0.weight"
                    | "final_layer.linear.weight"
            )
        {
            vec![1.0, 0.0, 0.0, 1.0]
        } else if key == "vae.decoder.weight" {
            vec![7.0]
        } else if key.starts_with("text_encoders.") {
            vec![8.0]
        } else {
            vec![1.0; elements]
        };
        source.insert(source_key, tensor(backend, &shape, &values, dtype, context)?);
    }
    Ok(source)
}

fn legacy_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    spec: RowSpec,
    dtype: DType,
    omitted: Option<&str>,
) -> Result<sim_model::MappedModelWeights, ModelFamilyError> {
    let mut source = mapping_tensors(
        backend,
        context,
        spec,
        HunyuanDiTLayout::PrefixedNative,
        dtype,
        omitted,
    )
    .map_err(|error| ModelFamilyError::InvalidSelectorOutput(error.to_string()))?;
    source.retain(|key, _| key.starts_with("model.diffusion_model."));
    map_model_weights(
        registration(spec.variant).definition,
        artifact_digest(spec),
        source,
    )
}

fn assert_shape_reduced_configuration(spec: RowSpec, configuration: HunyuanDiTConfiguration) {
    assert_eq!(configuration.in_channels, 4);
    assert_eq!(configuration.patch_size, 2);
    assert_eq!(configuration.number_of_heads, 16);
    assert_eq!(configuration.depth, 2);
    assert!(configuration.qk_normalization);
    assert!(configuration.learn_sigma);
    assert_eq!(configuration.memory_usage_factor, 1.3);
    match spec.variant {
        HunyuanDiTVariant::DiT => {
            assert_eq!(configuration.hidden_size, 32);
            assert_eq!(configuration.extra_input_dimension, 1_024);
            assert_eq!(configuration.mlp_ratio, 4.0);
            assert!(!configuration.size_conditioning);
            assert!(!configuration.style_conditioning);
            assert_eq!(
                configuration.attention_precision,
                HunyuanDiTAttentionPrecision::Float32
            );
            assert_eq!(configuration.sampling_linear_end, HUNYUANDIT_LINEAR_END);
        }
        HunyuanDiTVariant::DiT1 => {
            assert_eq!(configuration.hidden_size, HUNYUANDIT_G_HIDDEN_SIZE);
            assert_eq!(configuration.extra_input_dimension, HUNYUANDIT_DIT1_EXTRA_INPUT);
            assert_eq!(configuration.mlp_ratio, 4.0);
            assert!(configuration.size_conditioning);
            assert!(configuration.style_conditioning);
            assert_eq!(
                configuration.attention_precision,
                HunyuanDiTAttentionPrecision::Inherited
            );
            assert_eq!(configuration.sampling_linear_end, HUNYUANDIT1_LINEAR_END);
        }
    }
}

fn assert_source_exact_configuration(spec: RowSpec, configuration: HunyuanDiTConfiguration) {
    assert_eq!(configuration.variant, spec.variant);
    assert_eq!(configuration.in_channels, 4);
    assert_eq!(configuration.patch_size, 2);
    assert_eq!(configuration.number_of_heads, 16);
    assert!(configuration.qk_normalization);
    assert!(configuration.learn_sigma);
    match spec.variant {
        HunyuanDiTVariant::DiT => {
            assert_eq!(configuration.hidden_size, 1_152);
            assert_eq!(configuration.depth, 28);
            assert_eq!(configuration.mlp_ratio, 4.0);
            assert_eq!(configuration.extra_input_dimension, 1_024);
        }
        HunyuanDiTVariant::DiT1 => {
            assert_eq!(configuration.hidden_size, HUNYUANDIT_G_HIDDEN_SIZE);
            assert_eq!(configuration.depth, HUNYUANDIT_G_DEPTH);
            assert_eq!(configuration.mlp_ratio, HUNYUANDIT_G_MLP_RATIO);
            assert_eq!(configuration.extra_input_dimension, HUNYUANDIT_DIT1_EXTRA_INPUT);
        }
    }
}

fn layout_prefix(layout: HunyuanDiTLayout) -> &'static str {
    match layout {
        HunyuanDiTLayout::PrefixedNative => "model.diffusion_model.",
        HunyuanDiTLayout::SavedModel => "model.",
        HunyuanDiTLayout::StandaloneNative => "",
    }
}

fn options(dtype: DType, device: DeviceKind, budget: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device,
        activation_elements: 2,
        memory_budget_bytes: budget,
        allow_unexpected_weights: true,
    }
}

fn artifact_digest(spec: RowSpec) -> &'static str {
    match spec.variant {
        HunyuanDiTVariant::DiT => {
            "0087008700870087008700870087008700870087008700870087008700870087"
        }
        HunyuanDiTVariant::DiT1 => {
            "0088008800880088008800880088008800880088008800880088008800880088"
        }
    }
}

fn probe_through_model_store(spec: RowSpec) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("hunyuandit.safetensors");
    write_sparse_safetensors(&model_path, spec)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        spec.fixture,
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new(spec.fixture, "hunyuandit.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_sparse_safetensors(
    path: &Path,
    spec: RowSpec,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data_bytes = 0_u64;
    for (key, shape) in detector_shapes(spec, false) {
        let start = data_bytes;
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        data_bytes = data_bytes
            .checked_add(elements.checked_mul(2).ok_or("fixture byte overflow")?)
            .ok_or("fixture data overflow")?;
        header.insert(
            format!("model.diffusion_model.{key}"),
            serde_json::json!({
                "dtype": "F16",
                "shape": shape,
                "data_offsets": [start, data_bytes]
            }),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    let file_length = 8_u64
        .checked_add(u64::try_from(header.len())?)
        .and_then(|value| value.checked_add(data_bytes))
        .ok_or("safetensors length overflow")?;
    file.set_len(file_length)?;
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
        .ok_or("HunyuanDiT checkpoint is missing")?;
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

fn fixture_directory(spec: RowSpec) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(spec.fixture)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
