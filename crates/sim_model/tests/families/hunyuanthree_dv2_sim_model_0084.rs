use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, HUNYUAN3D_MEMORY_USAGE_FACTOR,
    HUNYUAN3D_MINI_DEPTH, HUNYUAN3D_MLP_RATIO, HUNYUAN3D_NUMBER_OF_HEADS,
    HUNYUAN3D_SUPPORTED_DEVICES, HUNYUAN3D_SUPPORTED_DTYPES, HUNYUAN3D_V21_CONTEXT_DIMENSION,
    Hunyuan3DConfiguration, Hunyuan3DLayout, Hunyuan3DVariant, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts,
    ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_hunyuanthree_dv2_1_sim_model_0085 as row_v21,
    generated_hunyuanthree_dv2_sim_model_0084 as row_v2,
    generated_hunyuanthree_dv2mini_sim_model_0086 as row_mini,
    hunyuan3d_configuration_for_probe, hunyuan3d_state_plan_for_layout,
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

const ARTIFACT_DIGEST: &str = "0084008400840084008400840084008400840084008400840084008400840084";

#[derive(Clone, Copy)]
pub(super) struct RowSpec {
    pub feature_id: &'static str,
    pub identifier: &'static str,
    pub fixture: &'static str,
    pub module: &'static str,
    pub source_ordinal: u16,
    pub source_architecture: &'static str,
    pub architecture_version: &'static str,
    pub latent_feature_id: &'static str,
    pub latent_identifier: &'static str,
    pub image_model: &'static str,
    pub projection_sha256: &'static str,
    pub variant: Hunyuan3DVariant,
    pub detection_score: u32,
}

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9084",
    identifier: "Hunyuan3Dv2AmbiguousFixture",
    ..row_v2::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_v2::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 166,
        source_architecture: "model_base.Hunyuan3Dv2AmbiguousFixture",
        ..row_v2::MODEL_FAMILY_REGISTRATION
    },
];

const SPEC: RowSpec = RowSpec {
    feature_id: row_v2::MODEL_FAMILY_FEATURE_ID,
    identifier: row_v2::MODEL_FAMILY_IDENTIFIER,
    fixture: row_v2::MODEL_FAMILY_FIXTURE,
    module: "hunyuanthree_dv2_comfy_model_0084",
    source_ordinal: row_v2::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.Hunyuan3Dv2",
    architecture_version: "hunyuan3d-v2-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0032",
    latent_identifier: "Hunyuan3Dv2",
    image_model: "hunyuan3d2",
    projection_sha256: row_v2::MODEL_FAMILY_PROJECTION_SHA256,
    variant: Hunyuan3DVariant::V2,
    detection_score: 1_000,
};

#[test]
fn val_model_family_row_001_hunyuan3dv2_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuan3dv2_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuan3dv2_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}

pub(super) fn assert_source_contract(spec: RowSpec) -> Result<(), Box<dyn std::error::Error>> {
    let registration = registration(spec.variant);
    assert_eq!(registration.definition.feature_id, spec.feature_id);
    assert_eq!(registration.definition.identifier, spec.identifier);
    assert_eq!(registration.source_ordinal, spec.source_ordinal);
    assert_eq!(registration.source_architecture, spec.source_architecture);
    assert!(registration.source_configuration.is_empty());
    assert_eq!(HUNYUAN3D_MEMORY_USAGE_FACTOR, 3.5);
    assert_eq!(HUNYUAN3D_MLP_RATIO, 4.0);
    assert_eq!(HUNYUAN3D_NUMBER_OF_HEADS, 16);

    let descriptor = describe_model_family(registration.definition)?;
    assert_eq!(descriptor.architecture_version, spec.architecture_version);
    assert_eq!(descriptor.latent_format, spec.latent_identifier);
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 2);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 2);
    assert_eq!(registration.definition.latent_feature_id, spec.latent_feature_id);
    assert_eq!(registration.definition.components.len(), 3);
    assert_eq!(HUNYUAN3D_SUPPORTED_DTYPES.len(), 3);
    assert_eq!(HUNYUAN3D_SUPPORTED_DEVICES, &[DeviceKind::Cpu]);

    let registry = all_rows_registry()?;
    for layout in [
        Hunyuan3DLayout::PrefixedNative,
        Hunyuan3DLayout::SavedModel,
        Hunyuan3DLayout::StandaloneNative,
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), spec.feature_id);
        assert_eq!(resolved.detection().score, spec.detection_score);
        assert_eq!(resolved.detection().evidence.len(), 4);
        assert!(
            resolved
                .detection()
                .evidence
                .iter()
                .all(|evidence| evidence.contains("AnyKeyPresent"))
        );
        assert_eq!(resolved.profile().latent_feature_id, spec.latent_feature_id);
        assert_eq!(resolved.profile().latent_identifier, spec.latent_identifier);
        let configuration = hunyuan3d_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, spec.variant);
        assert_eq!(configuration.layout, layout);
        assert_shape_reduced_configuration(spec, configuration);
    }

    let source_configuration =
        hunyuan3d_configuration_for_probe(&source_exact_probe(spec, Hunyuan3DLayout::StandaloneNative))?;
    assert_source_exact_configuration(spec, source_configuration);

    let mut misleading = parsed_facts(spec, Hunyuan3DLayout::PrefixedNative, DType::F32);
    misleading.formats[0]
        .metadata
        .insert("image_model".to_owned(), "genmo_mochi".to_owned());
    misleading.formats[0]
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    let misleading = ModelProbe::from_parsed_facts(misleading)?;
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        spec.feature_id
    );
    assert_eq!(
        hunyuan3d_configuration_for_probe(&misleading)?.layout,
        Hunyuan3DLayout::PrefixedNative
    );

    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("transformer_blocks.0.attn.to_q.weight".to_owned(), vec![32, 32]),
            ("proj_in.weight".to_owned(), vec![32, 4]),
        ]),
        metadata: BTreeMap::from([("image_model".to_owned(), spec.image_model.to_owned())]),
    };
    assert!(matches!(
        registry.detect(&diffusers),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let store_probe = probe_through_model_store(spec)?;
    for key in ["image_model", "model_layout", "depth", "guidance"] {
        assert!(!store_probe.metadata.contains_key(key));
    }
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        spec.feature_id
    );

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory(spec).join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("Hunyuan3D source projection must be a string")?;
    assert_eq!(sha256(projection.as_bytes()), spec.projection_sha256);
    assert_eq!(provenance["source_projection_sha256"], spec.projection_sha256);
    for source in provenance["source_files"]
        .as_array()
        .ok_or("Hunyuan3D source_files must be an array")?
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
        .ok_or("Hunyuan3D catalog row is missing")?;
    assert_eq!(row["source_ordinal"], spec.source_ordinal);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], spec.image_model);
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 3.5);
    assert_eq!(
        row["static"]["latent_format"]["value"]["symbol"],
        format!("latent_formats.{}", spec.latent_identifier)
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families")
            .join(format!("{}.rs", spec.module)),
    )?;
    for canonical_import in [
        "HUNYUAN3D_COMPONENTS",
        "HUNYUAN3D_MEMORY_USAGE_FACTOR",
        "HUNYUAN3D_PREFIXED_STATE_PLAN",
        "HUNYUAN3D_SAVED_MODEL_STATE_PLAN",
        "HUNYUAN3D_STANDALONE_STATE_PLAN",
        "HUNYUAN3D_SUPPORTED_DTYPES",
        "hunyuan3d_configuration_for_probe",
    ] {
        assert!(row_source.contains(canonical_import), "{canonical_import}");
    }
    for forbidden in [
        "ModelDetectionRule::Metadata",
        "ModelSourceConfigurationRule",
        "MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION",
        "struct Hunyuan3DConfiguration",
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
            "key-derived-prefixed-saved-and-standalone-layouts",
            "variant-depth-configuration-and-latent-identity",
            "misleading-metadata-and-unsupported-diffusers-rejection",
            "transactional-scale-normalization-and-component-routing",
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

    for layout in [
        Hunyuan3DLayout::PrefixedNative,
        Hunyuan3DLayout::SavedModel,
        Hunyuan3DLayout::StandaloneNative,
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
        let resolved = registry.resolve(&probe)?;
        let source = source_tensors(&backend, &context, spec, layout, DType::F32, None)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        assert_eq!(mapped.components().len(), 3);
        let model = mapped.component("model").ok_or("missing Hunyuan3D model")?;
        assert!(!model.is_empty());
        assert!(model.keys().all(|key| key.starts_with("native.")));
        assert!(model.keys().all(|key| !key.ends_with(".scale")));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("clip_vision").map(BTreeMap::len), Some(1));

        let scale_source = scale_mapping_source(&backend, &context, spec, layout)?;
        let scale_mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan3d_state_plan_for_layout(layout).compile()?,
            ARTIFACT_DIGEST,
            &scale_source,
        )?;
        assert!(
            scale_mapped
                .component("model")
                .ok_or("missing scale-normalized model")?
                .contains_key("native.final_layer.linear.weight")
        );
    }

    let layout = Hunyuan3DLayout::PrefixedNative;
    let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, spec, layout, DType::F32, None)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let expected_memory = expected_memory_bytes(spec);
    let model = build_model_family_for_probe(
        &registry,
        &probe,
        weights,
        options(DType::F32, DeviceKind::Cpu, expected_memory),
    )
    .map_err(|error| format!("{} build failed: {error}", spec.identifier))?;
    assert_eq!(model.memory_estimate().total_bytes, expected_memory);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model
        .forward_checkpoints(&backend, &input, &context)
        .map_err(|error| format!("{} forward failed: {error}", spec.identifier))?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        final_checkpoint(spec),
        &[0.8004987, -0.8004987],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
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
        final_checkpoint(spec),
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
            options(DType::F32, DeviceKind::Cpu, expected_memory - 1),
        ),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == expected_memory && budget == expected_memory - 1
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
    let expected_memory = expected_memory_bytes(spec);

    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let layout = Hunyuan3DLayout::StandaloneNative;
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, dtype))?;
        let resolved = registry.resolve(&probe)?;
        let source = source_tensors(&backend, &context, spec, layout, dtype, None)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(dtype, DeviceKind::Cpu, expected_memory),
        )?;
    }

    let layout = Hunyuan3DLayout::StandaloneNative;
    let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
    let resolved = registry.resolve(&probe)?;
    let source = source_tensors(&backend, &context, spec, layout, DType::F32, None)?;
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
            options(DType::F32, DeviceKind::Metal, expected_memory),
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let omitted = primary_marker(spec);
    let partial_source = source_tensors(
        &backend,
        &context,
        spec,
        layout,
        DType::F32,
        Some(omitted),
    )?;
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial_source,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(message)) if message.contains("key count changed")
    ));

    let mut unexpected = source.clone();
    unexpected.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &unexpected,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(message)) if message.contains("key count changed")
    ));
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan3d_state_plan_for_layout(layout).compile()?,
            ARTIFACT_DIGEST,
            &unexpected,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let ambiguous = ModelFamilyRegistry::checked_registrations(ambiguous_registrations)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score, .. }) if score == spec.detection_score
    ));

    let mut mixed = parsed_facts(spec, Hunyuan3DLayout::PrefixedNative, DType::F32);
    mixed.tensors.extend(
        parsed_facts(spec, Hunyuan3DLayout::StandaloneNative, DType::F32).tensors,
    );
    assert!(matches!(
        registry.resolve(&ModelProbe::from_parsed_facts(mixed)?),
        Err(ModelFamilyError::ModelLayoutSelection(_))
            | Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));

    let cross_family = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("patch_embed.proj.weight".to_owned(), vec![32, 16]),
            ("time_embedding_linear_1.weight".to_owned(), vec![2, 2]),
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

fn all_rows_registry() -> Result<ModelFamilyRegistry, ModelFamilyError> {
    ModelFamilyRegistry::checked_registrations(&[
        row_mini::MODEL_FAMILY_REGISTRATION,
        row_v2::MODEL_FAMILY_REGISTRATION,
        row_v21::MODEL_FAMILY_REGISTRATION,
    ])
}

fn registration(variant: Hunyuan3DVariant) -> ModelFamilyRegistration {
    match variant {
        Hunyuan3DVariant::V2 => row_v2::MODEL_FAMILY_REGISTRATION,
        Hunyuan3DVariant::V2_1 => row_v21::MODEL_FAMILY_REGISTRATION,
        Hunyuan3DVariant::V2Mini => row_mini::MODEL_FAMILY_REGISTRATION,
    }
}

fn parsed_facts(spec: RowSpec, layout: Hunyuan3DLayout, dtype: DType) -> ModelParsedFacts {
    let prefix = layout_prefix(layout);
    let tensors = source_shapes(spec)
        .into_iter()
        .map(|(key, shape)| {
            (
                if key.starts_with("vae.") || key.starts_with("conditioner.") {
                    key
                } else {
                    format!("{prefix}{key}")
                },
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

fn source_shapes(spec: RowSpec) -> Vec<(String, Vec<u64>)> {
    let mut shapes = Vec::new();
    match spec.variant {
        Hunyuan3DVariant::V2 | Hunyuan3DVariant::V2Mini => {
            shapes.extend([
                ("latent_in.weight".to_owned(), vec![32, 4]),
                ("cond_in.weight".to_owned(), vec![32, 6]),
            ]);
            let depth = if spec.variant == Hunyuan3DVariant::V2 {
                9
            } else {
                HUNYUAN3D_MINI_DEPTH
            };
            for ordinal in 0..depth {
                shapes.push((
                    format!("double_blocks.{ordinal}.img_attn.proj.weight"),
                    vec![2, 2],
                ));
            }
            for ordinal in 0..2 {
                shapes.push((
                    format!("single_blocks.{ordinal}.linear1.weight"),
                    vec![2, 2],
                ));
            }
            shapes.extend(classic_forward_shapes());
        }
        Hunyuan3DVariant::V2_1 => {
            shapes.extend([
                ("x_embedder.weight".to_owned(), vec![32, 4]),
                ("t_embedder.mlp.2.weight".to_owned(), vec![128, 32]),
                ("t_embedder.mlp.2.bias".to_owned(), vec![2]),
                ("blocks.0.attn1.k_norm.weight".to_owned(), vec![2]),
            ]);
            for ordinal in 0..2 {
                shapes.push((
                    format!("blocks.{ordinal}.attn1.q_proj.weight"),
                    vec![2, 2],
                ));
            }
            shapes.extend([
                ("t_embedder.mlp.0.weight".to_owned(), vec![2, 2]),
                ("t_embedder.mlp.0.bias".to_owned(), vec![2]),
                ("t_embedder.cond_proj.weight".to_owned(), vec![2, 2]),
                ("t_embedder.cond_proj.bias".to_owned(), vec![2]),
                ("final_layer.linear.weight".to_owned(), vec![2, 2]),
                ("final_layer.linear.bias".to_owned(), vec![2]),
            ]);
        }
    }
    shapes.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        (
            "conditioner.main_image_encoder.model.visual.weight".to_owned(),
            vec![1],
        ),
    ]);
    shapes
}

fn classic_forward_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("time_in.in_layer.weight".to_owned(), vec![2, 2]),
        ("time_in.in_layer.bias".to_owned(), vec![2]),
        ("time_in.out_layer.weight".to_owned(), vec![2, 2]),
        ("time_in.out_layer.bias".to_owned(), vec![2]),
        ("final_layer.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.bias".to_owned(), vec![2]),
    ]
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    spec: RowSpec,
    layout: Hunyuan3DLayout,
    dtype: DType,
    omitted: Option<&str>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let mut source = BTreeMap::new();
    for (key, shape) in source_shapes(spec) {
        if omitted == Some(key.as_str()) {
            continue;
        }
        let source_key = if key.starts_with("vae.") || key.starts_with("conditioner.") {
            key.clone()
        } else {
            format!("{prefix}{key}")
        };
        let elements = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key.ends_with("in_layer.bias") || key.ends_with("mlp.0.bias") {
            vec![1.0, -1.0]
        } else if key.ends_with("out_layer.bias") || key.ends_with("cond_proj.bias") {
            vec![0.25, -0.25]
        } else if key == "final_layer.linear.bias" {
            vec![0.1, -0.1]
        } else if shape == [2, 2]
            && (key.contains("time_in")
                || key.contains("t_embedder.mlp.0")
                || key.contains("t_embedder.cond_proj")
                || key.contains("double_blocks.0")
                || key.contains("single_blocks.0")
                || key == "blocks.0.attn1.q_proj.weight"
                || key == "final_layer.linear.weight")
        {
            vec![1.0, 0.0, 0.0, 1.0]
        } else if key == "vae.decoder.weight" {
            vec![7.0]
        } else if key.starts_with("conditioner.") {
            vec![8.0]
        } else if key.ends_with("k_norm.weight") {
            vec![1.0; elements]
        } else {
            vec![0.0; elements]
        };
        source.insert(source_key, tensor(backend, &shape, &values, dtype, context)?);
    }
    Ok(source)
}

fn scale_mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    spec: RowSpec,
    layout: Hunyuan3DLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let marker = if spec.variant == Hunyuan3DVariant::V2_1 {
        "x_embedder.weight"
    } else {
        "latent_in.weight"
    };
    Ok(BTreeMap::from([
        (
            format!("{prefix}{marker}"),
            tensor(backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], DType::F32, context)?,
        ),
        (
            format!("{prefix}final_layer.linear.scale"),
            tensor(backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], DType::F32, context)?,
        ),
        (
            "vae.decoder.weight".to_owned(),
            tensor(backend, &[1], &[7.0], DType::F32, context)?,
        ),
        (
            "conditioner.main_image_encoder.model.visual.weight".to_owned(),
            tensor(backend, &[1], &[8.0], DType::F32, context)?,
        ),
    ]))
}

fn source_exact_probe(spec: RowSpec, layout: Hunyuan3DLayout) -> ModelProbe {
    let prefix = layout_prefix(layout);
    let mut tensor_shapes = BTreeMap::new();
    match spec.variant {
        Hunyuan3DVariant::V2 | Hunyuan3DVariant::V2Mini => {
            tensor_shapes.insert(format!("{prefix}latent_in.weight"), vec![1_024, 64]);
            tensor_shapes.insert(format!("{prefix}cond_in.weight"), vec![1_024, 1_536]);
            let depth = if spec.variant == Hunyuan3DVariant::V2 {
                16
            } else {
                HUNYUAN3D_MINI_DEPTH
            };
            for ordinal in 0..depth {
                tensor_shapes.insert(
                    format!("{prefix}double_blocks.{ordinal}.img_attn.proj.weight"),
                    vec![1_024, 1_024],
                );
            }
            for ordinal in 0..32 {
                tensor_shapes.insert(
                    format!("{prefix}single_blocks.{ordinal}.linear1.weight"),
                    vec![1_024, 1_024],
                );
            }
        }
        Hunyuan3DVariant::V2_1 => {
            tensor_shapes.extend([
                (format!("{prefix}x_embedder.weight"), vec![2_048, 64]),
                (
                    format!("{prefix}t_embedder.mlp.2.weight"),
                    vec![8_192, 2_048],
                ),
                (format!("{prefix}blocks.0.attn1.k_norm.weight"), vec![128]),
            ]);
            for ordinal in 0..21 {
                tensor_shapes.insert(
                    format!("{prefix}blocks.{ordinal}.attn1.q_proj.weight"),
                    vec![2_048, 2_048],
                );
            }
        }
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn assert_shape_reduced_configuration(spec: RowSpec, configuration: Hunyuan3DConfiguration) {
    assert_eq!(configuration.in_channels, 4);
    assert_eq!(configuration.hidden_size, 32);
    assert_eq!(configuration.number_of_heads, 16);
    assert_eq!(configuration.mlp_ratio, 4.0);
    assert_eq!(configuration.memory_usage_factor, 3.5);
    match spec.variant {
        Hunyuan3DVariant::V2 => {
            assert_eq!(configuration.context_dimension, 6);
            assert_eq!(configuration.depth, 9);
            assert_eq!(configuration.single_block_depth, 2);
            assert!(configuration.qkv_bias);
        }
        Hunyuan3DVariant::V2Mini => {
            assert_eq!(configuration.context_dimension, 6);
            assert_eq!(configuration.depth, 8);
            assert_eq!(configuration.single_block_depth, 2);
            assert!(configuration.qkv_bias);
        }
        Hunyuan3DVariant::V2_1 => {
            assert_eq!(configuration.context_dimension, HUNYUAN3D_V21_CONTEXT_DIMENSION);
            assert_eq!(configuration.depth, 2);
            assert_eq!(configuration.single_block_depth, 0);
            assert!(!configuration.qkv_bias);
        }
    }
}

fn assert_source_exact_configuration(spec: RowSpec, configuration: Hunyuan3DConfiguration) {
    assert_eq!(configuration.variant, spec.variant);
    assert_eq!(configuration.in_channels, 64);
    assert_eq!(configuration.number_of_heads, 16);
    assert_eq!(configuration.mlp_ratio, 4.0);
    match spec.variant {
        Hunyuan3DVariant::V2 => {
            assert_eq!(configuration.context_dimension, 1_536);
            assert_eq!(configuration.hidden_size, 1_024);
            assert_eq!(configuration.depth, 16);
            assert_eq!(configuration.single_block_depth, 32);
        }
        Hunyuan3DVariant::V2Mini => {
            assert_eq!(configuration.context_dimension, 1_536);
            assert_eq!(configuration.hidden_size, 1_024);
            assert_eq!(configuration.depth, 8);
            assert_eq!(configuration.single_block_depth, 32);
        }
        Hunyuan3DVariant::V2_1 => {
            assert_eq!(configuration.context_dimension, 1_024);
            assert_eq!(configuration.hidden_size, 2_048);
            assert_eq!(configuration.depth, 21);
            assert_eq!(configuration.single_block_depth, 0);
        }
    }
}

fn layout_prefix(layout: Hunyuan3DLayout) -> &'static str {
    match layout {
        Hunyuan3DLayout::PrefixedNative => "model.diffusion_model.",
        Hunyuan3DLayout::SavedModel => "model.",
        Hunyuan3DLayout::StandaloneNative => "",
    }
}

fn primary_marker(spec: RowSpec) -> &'static str {
    if spec.variant == Hunyuan3DVariant::V2_1 {
        "x_embedder.weight"
    } else {
        "latent_in.weight"
    }
}

fn expected_memory_bytes(spec: RowSpec) -> u64 {
    let parameters = source_shapes(spec)
        .into_iter()
        .filter(|(key, _)| !key.starts_with("vae.") && !key.starts_with("conditioner."))
        .map(|(_, shape)| shape.into_iter().product::<u64>())
        .sum::<u64>();
    parameters * 2 + 4
}

fn final_checkpoint(_spec: RowSpec) -> &'static str {
    "latent_output"
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

fn probe_through_model_store(spec: RowSpec) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("hunyuan3d.safetensors");
    write_safetensors(&model_path, spec)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        spec.fixture,
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new(spec.fixture, "hunyuan3d.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path, spec: RowSpec) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, tensor) in parsed_facts(spec, Hunyuan3DLayout::PrefixedNative, DType::F32).tensors {
        let start = data.len();
        let elements = tensor.shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        let bytes = usize::try_from(elements.checked_mul(4).ok_or("fixture byte overflow")?)?;
        data.resize(data.len().checked_add(bytes).ok_or("fixture data overflow")?, 0);
        header.insert(
            key,
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
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
        .ok_or("Hunyuan3D checkpoint is missing")?;
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
