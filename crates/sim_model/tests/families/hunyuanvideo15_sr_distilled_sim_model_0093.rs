use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, HUNYUAN_REFINER_IMAGE_SCALE,
    HUNYUAN_REFINER_SEED_OFFSET, HUNYUAN_VIDEO_BYT5_INPUT_DIMENSION,
    HUNYUAN_VIDEO_BYT5_INTERMEDIATE_DIMENSION, HUNYUAN_VIDEO_HEAD_DIMENSION,
    HUNYUAN_VIDEO_MLP_RATIO, HUNYUAN_VIDEO_SAVE_PREFIX, HUNYUAN_VIDEO_SUPPORTED_DEVICES,
    HUNYUAN_VIDEO_THETA, HUNYUAN_VIDEO_VECTOR_INPUT_DIMENSION,
    HUNYUAN_VIDEO15_VISION_INPUT_DIMENSION, HunyuanVideoConfiguration, HunyuanVideoLayout,
    HunyuanVideoVariant, ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration,
    ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact,
    ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits,
    PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    augment_refiner_conditioning, build_model_family, describe_model_family,
    generated_hunyuanimage21_sim_model_0089 as row_image,
    generated_hunyuanimage21refiner_sim_model_0090 as row_refiner,
    generated_hunyuanvideo_sim_model_0091 as row_video,
    generated_hunyuanvideo15_sim_model_0092 as row_video15,
    generated_hunyuanvideo15_sr_distilled_sim_model_0093 as row_sr,
    generated_hunyuanvideoi2v_sim_model_0094 as row_i2v,
    hunyuan_video_configuration_for_probe, hunyuan_video_state_plan_for_layout,
    map_model_weights,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, RetryRngPolicy, RngAlgorithm,
    RngProfileVersion, RngStream, RngStreamAddress, StreamId, Tensor, TensorBackend,
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
    pub source_architecture: &'static str,
    pub architecture_version: &'static str,
    pub latent_feature_id: &'static str,
    pub latent_identifier: &'static str,
    pub latent_symbol: &'static str,
    pub clip_tokenizer: &'static str,
    pub clip_model: &'static str,
    pub projection_sha256: &'static str,
    pub registration: ModelFamilyRegistration,
    pub artifact_digest: &'static str,
    pub variant: HunyuanVideoVariant,
    pub memory_usage_factor: f64,
    pub sampling_shift: f64,
    pub supported_dtypes: &'static [DType],
    pub detector_rule_count: usize,
    pub detector_in_channels: u64,
    pub detector_patch: &'static [u64],
    pub detector_context_input: u64,
}

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9093",
    identifier: "HunyuanVideo15SrDistilledAmbiguousFixture",
    ..row_sr::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_sr::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 134,
        source_architecture: "model_base.HunyuanVideo15SrDistilledAmbiguousFixture",
        ..row_sr::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: RowSpec = RowSpec {
    feature_id: row_sr::MODEL_FAMILY_FEATURE_ID,
    identifier: row_sr::MODEL_FAMILY_IDENTIFIER,
    fixture: row_sr::MODEL_FAMILY_FIXTURE,
    module: "hunyuanvideo15_sr_distilled_comfy_model_0093",
    source_ordinal: row_sr::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.HunyuanVideo15_SR_Distilled",
    architecture_version: "hunyuan-video-1.5-sr-distilled-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0038",
    latent_identifier: "HunyuanVideo15",
    latent_symbol: "latent_formats.HunyuanVideo15",
    clip_tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer",
    clip_model: "sim.text_encoders.hunyuan_image.te",
    projection_sha256: row_sr::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_sr::MODEL_FAMILY_REGISTRATION,
    artifact_digest: "0093009300930093009300930093009300930093009300930093009300930093",
    variant: HunyuanVideoVariant::Video15SrDistilled,
    memory_usage_factor: 4.0,
    sampling_shift: 2.0,
    supported_dtypes: &[DType::F16, DType::Bf16, DType::F32],
    detector_rule_count: 5,
    detector_in_channels: 98,
    detector_patch: &[1, 2, 2],
    detector_context_input: 3_584,
};

#[test]
fn val_model_family_row_001_hunyuanvideo15_sr_distilled_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideo15_sr_distilled_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideo15_sr_distilled_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}

pub(super) fn assert_source_contract(spec: RowSpec) -> Result<(), Box<dyn std::error::Error>> {
    let registration = spec.registration;
    assert_eq!(registration.definition.feature_id, spec.feature_id);
    assert_eq!(registration.definition.identifier, spec.identifier);
    assert_eq!(registration.source_ordinal, spec.source_ordinal);
    assert_eq!(registration.source_architecture, spec.source_architecture);
    assert!(registration.source_configuration.is_empty());

    let descriptor = describe_model_family(registration.definition)?;
    assert_eq!(descriptor.architecture_version, spec.architecture_version);
    assert_eq!(descriptor.latent_format, spec.latent_identifier);
    assert_eq!(registration.definition.latent_feature_id, spec.latent_feature_id);
    assert_eq!(registration.definition.components.len(), 3);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(registration.definition.supported_dtypes, spec.supported_dtypes);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 2);
    assert_eq!(descriptor.memory_estimator.activation_bytes_per_element, 2);
    assert_eq!(registration.definition.clip_target.candidates.len(), 1);
    assert_eq!(
        registration.definition.clip_target.candidates[0].tokenizer,
        spec.clip_tokenizer
    );
    assert_eq!(
        registration.definition.clip_target.candidates[0].clip_model,
        spec.clip_model
    );
    assert_eq!(HUNYUAN_VIDEO_SUPPORTED_DEVICES, &[DeviceKind::Cpu]);
    assert_eq!(HUNYUAN_VIDEO_THETA, 256);
    assert_eq!(HUNYUAN_VIDEO_HEAD_DIMENSION, 128);
    assert_eq!(HUNYUAN_VIDEO_MLP_RATIO, 4.0);
    assert_eq!(HUNYUAN_VIDEO_VECTOR_INPUT_DIMENSION, 768);
    assert_eq!(HUNYUAN_VIDEO_BYT5_INPUT_DIMENSION, 1_472);
    assert_eq!(HUNYUAN_VIDEO_BYT5_INTERMEDIATE_DIMENSION, 2_048);
    assert_eq!(HUNYUAN_VIDEO15_VISION_INPUT_DIMENSION, 1_152);
    assert_eq!(HUNYUAN_VIDEO_SAVE_PREFIX, "model.model.");
    assert_eq!(HUNYUAN_REFINER_IMAGE_SCALE, 0.75);
    assert_eq!(HUNYUAN_REFINER_SEED_OFFSET, -10);

    let registry = all_rows_registry()?;
    for layout in layouts() {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), spec.feature_id);
        assert_eq!(resolved.detection().score, 1_000);
        assert_eq!(resolved.detection().evidence.len(), spec.detector_rule_count);
        assert!(
            resolved
                .detection()
                .evidence
                .iter()
                .all(|evidence| evidence.contains("AnyTensorFact"))
        );
        assert_eq!(resolved.profile().latent_feature_id, spec.latent_feature_id);
        assert_eq!(resolved.profile().latent_identifier, spec.latent_identifier);
        let configuration = hunyuan_video_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, spec.variant);
        assert_eq!(configuration.layout, layout);
        assert_configuration(spec, configuration)?;
    }

    for row_spec in all_specs() {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(
            row_spec,
            HunyuanVideoLayout::StandaloneNative,
            DType::F32,
        ))?;
        assert_eq!(
            registry.resolve(&probe)?.detection().identity.feature_id(),
            row_spec.feature_id
        );
    }

    let mut misleading = parsed_facts(spec, HunyuanVideoLayout::PrefixedNative, DType::F32);
    misleading.formats[0]
        .metadata
        .insert("image_model".to_owned(), "pixart_sigma".to_owned());
    misleading.formats[0]
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert_eq!(
        registry
            .resolve(&ModelProbe::from_parsed_facts(misleading)?)?
            .detection()
            .identity
            .feature_id(),
        spec.feature_id
    );

    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("transformer_blocks.0.attn.to_q.weight".to_owned(), vec![256, 256]),
            ("proj_in.weight".to_owned(), vec![256, 64]),
        ]),
        metadata: BTreeMap::from([("image_model".to_owned(), "hunyuan_video".to_owned())]),
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
    for key in ["image_model", "model_layout", "variant"] {
        assert!(!store_probe.metadata.contains_key(key));
    }

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory(spec).join("provenance.json"))?)?;
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("Hunyuan image/video source projection must be a string")?;
    assert_eq!(sha256(projection.as_bytes()), spec.projection_sha256);
    assert_eq!(provenance["source_projection_sha256"], spec.projection_sha256);
    match spec.variant {
        HunyuanVideoVariant::Video15SrDistilled => {
            assert!(projection.contains("unet_config.in_channels=98"));
            assert!(projection.contains("configuration.meanflow_sum=true"));
        }
        HunyuanVideoVariant::VideoI2V => {
            assert!(projection.contains("unet_config.in_channels=33"));
            assert!(projection.contains("conditioning.concat_keys=concat_image,mask_inverted"));
        }
        _ => {}
    }
    for source in provenance["source_files"]
        .as_array()
        .ok_or("Hunyuan image/video source_files must be an array")?
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
        .ok_or("Hunyuan image/video catalog row is missing")?;
    assert_eq!(row["source_ordinal"], spec.source_ordinal);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "hunyuan_video");
    assert_eq!(row["static"]["memory_usage_factor"]["value"], spec.memory_usage_factor);
    assert_eq!(row["static"]["latent_format"]["value"]["symbol"], spec.latent_symbol);

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families")
            .join(format!("{}.rs", spec.module)),
    )?;
    for canonical_import in [
        "HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS",
        "HUNYUAN_VIDEO_COMPONENTS",
        "HUNYUAN_VIDEO_FORWARD_PROGRAM",
        "HUNYUAN_VIDEO_PREFIXED_STATE_PLAN",
        "HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN",
        "HUNYUAN_VIDEO_STANDALONE_STATE_PLAN",
        "hunyuan_video_configuration_for_probe",
    ] {
        assert!(row_source.contains(canonical_import), "{canonical_import}");
    }
    assert_eq!(
        row_source.matches("ModelDetectionRule::AnyTensorFact").count(),
        spec.detector_rule_count
    );
    for forbidden in [
        "ModelDetectionRule::Metadata",
        "ModelSourceConfigurationRule",
        "MODEL_STATE_TRANSFORM_PLAN_SCHEMA_VERSION",
        "struct HunyuanVideoConfiguration",
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
            "variant-configuration-conditioning-clip-and-latent-identity",
            "misleading-metadata-and-unsupported-diffusers-rejection",
            "transactional-layout-rewrites-and-component-routing",
            "model-store-forward-patch-memory-and-oom",
            "declared-dtype-cpu-and-fail-closed-device",
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
        let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
        let resolved = registry.resolve(&probe)?;
        let source = mapping_tensors(&backend, &context, spec, layout, DType::F32, None)?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan_video_state_plan_for_layout(layout).compile()?,
            spec.artifact_digest,
            &source,
        )?;
        assert_eq!(resolved.detection().identity.feature_id(), spec.feature_id);
        assert_eq!(mapped.components().len(), 3);
        let model = mapped.component("model").ok_or("missing Hunyuan model")?;
        assert!(model.keys().all(|key| key.starts_with("native.")));
        assert!(model.contains_key("native.txt_in.t_embedder.in_layer.weight"));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }

    let weights = legacy_weights(&backend, &context, spec, DType::F32, None)?;
    let model = build_model_family(
        spec.registration.definition,
        weights,
        options(DType::F32, DeviceKind::Cpu, 32),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, 32);
    let input = tensor(&backend, &[1, 2], &[0.0, 0.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "conditioning.context_projection",
        &[1.0, -1.0],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "conditioning.context_activation",
        &[0.7310586, -0.26894143],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "transformer.double_block_0_attention",
        &[0.7310586, -0.26894143],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "output.final_projection",
        &[0.8310586, -0.36894143],
    )?;

    let patch = PatchGraph::checked(
        spec.artifact_digest,
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
        &[1.3310586, 0.13105857],
    )?;

    if spec.variant == HunyuanVideoVariant::Image21Refiner {
        let latent = tensor(
            &backend,
            &[1, 1, 2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            DType::F32,
            &context,
        )?;
        let mut transaction = rng_transaction()?;
        let checkpoint = transaction.checkpoint();
        let conditioned = augment_refiner_conditioning(
            &backend,
            &latent,
            4,
            4,
            0.0,
            &mut transaction,
            &context,
        )?;
        let values = tensor_to_f32_with_context_exact_native(&backend, &conditioned, &context)?;
        assert!((values[0] - 0.75).abs() <= 1.0e-6);
        assert!((values[15] - 3.0).abs() <= 1.0e-6);
        assert_eq!(transaction.checkpoint(), checkpoint);
    }

    let weights = legacy_weights(&backend, &context, spec, DType::F32, None)?;
    assert!(matches!(
        build_model_family(
            spec.registration.definition,
            weights,
            options(DType::F32, DeviceKind::Cpu, 31),
        ),
        Err(ModelFamilyError::OutOfMemory {
            required: 32,
            budget: 31
        })
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
    let layout = HunyuanVideoLayout::StandaloneNative;

    for dtype in spec.supported_dtypes {
        let weights = legacy_weights(&backend, &context, spec, *dtype, None)?;
        build_model_family(
            spec.registration.definition,
            weights,
            options(*dtype, DeviceKind::Cpu, 32),
        )?;
    }
    if !spec.supported_dtypes.contains(&DType::F16) {
        let weights = legacy_weights(&backend, &context, spec, DType::F16, None)?;
        assert!(matches!(
            build_model_family(
                spec.registration.definition,
                weights,
                options(DType::F16, DeviceKind::Cpu, 32),
            ),
            Err(ModelFamilyError::UnsupportedDType(DType::F16))
        ));
    }

    let weights = legacy_weights(&backend, &context, spec, DType::F32, None)?;
    assert!(matches!(
        build_model_family(
            spec.registration.definition,
            weights,
            options(DType::F32, DeviceKind::Metal, 32),
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let probe = ModelProbe::from_parsed_facts(parsed_facts(spec, layout, DType::F32))?;
    let resolved = registry.resolve(&probe)?;
    let source = mapping_tensors(&backend, &context, spec, layout, DType::F32, None)?;
    let partial_source = mapping_tensors(
        &backend,
        &context,
        spec,
        layout,
        DType::F32,
        Some("img_in.proj.weight"),
    )?;
    assert!(matches!(
        resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            spec.artifact_digest,
            &partial_source,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(_))
    ));

    assert!(matches!(
        legacy_weights(
            &backend,
            &context,
            spec,
            DType::F32,
            Some("final_layer.linear.bias"),
        ),
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
            &hunyuan_video_state_plan_for_layout(layout).compile()?,
            spec.artifact_digest,
            &unexpected,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["unexpected.weight"]
    ));

    let ambiguous = ModelFamilyRegistry::checked_registrations(ambiguous_registrations)?;
    assert!(matches!(
        ambiguous.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let mut mixed = parsed_facts(spec, HunyuanVideoLayout::PrefixedNative, DType::F32);
    mixed.tensors.extend(
        parsed_facts(spec, HunyuanVideoLayout::StandaloneNative, DType::F32).tensors,
    );
    assert!(matches!(
        registry.resolve(&ModelProbe::from_parsed_facts(mixed)?),
        Err(ModelFamilyError::ModelLayoutSelection(_))
            | Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));

    let mut partial = parsed_facts(spec, layout, DType::F32);
    partial
        .tensors
        .remove(&format!("{}final_layer.linear.weight", layout_prefix(layout)));
    let partial = ModelProbe::from_parsed_facts(partial)?;
    assert!(matches!(
        registry.detect(&partial),
        Err(ModelFamilyError::NoDetectionMatch)
    ));
    assert!(matches!(
        hunyuan_video_configuration_for_probe(&partial),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("partial")
    ));

    let pixart = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("t_block.1.weight".to_owned(), vec![1]),
            ("y_embedder.y_embedding".to_owned(), vec![1, 256]),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        registry.detect(&pixart),
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
            &hunyuan_video_state_plan_for_layout(layout).compile()?,
            spec.artifact_digest,
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

fn all_specs() -> [RowSpec; 2] {
    [
        SPEC,
        super::generated_hunyuanvideoi2v_sim_model_0094::SPEC,
    ]
}

fn all_rows_registry() -> Result<ModelFamilyRegistry, ModelFamilyError> {
    ModelFamilyRegistry::checked_registrations(&[
        row_sr::MODEL_FAMILY_REGISTRATION,
        row_video15::MODEL_FAMILY_REGISTRATION,
        row_refiner::MODEL_FAMILY_REGISTRATION,
        row_image::MODEL_FAMILY_REGISTRATION,
        row_i2v::MODEL_FAMILY_REGISTRATION,
        row_video::MODEL_FAMILY_REGISTRATION,
    ])
}

fn layouts() -> [HunyuanVideoLayout; 3] {
    [
        HunyuanVideoLayout::PrefixedNative,
        HunyuanVideoLayout::SavedModel,
        HunyuanVideoLayout::StandaloneNative,
    ]
}

fn parsed_facts(spec: RowSpec, layout: HunyuanVideoLayout, dtype: DType) -> ModelParsedFacts {
    let prefix = layout_prefix(layout);
    let tensors = detector_shapes(spec)
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

fn detector_shapes(spec: RowSpec) -> Vec<(String, Vec<u64>)> {
    let hidden = 256;
    let in_channels = spec.detector_in_channels;
    let patch = spec.detector_patch;
    let context_input = spec.detector_context_input;
    let patch_volume = patch.iter().product::<u64>();
    let mut projection = vec![hidden, in_channels];
    projection.extend_from_slice(patch);
    let mut shapes = vec![
        (
            "txt_in.individual_token_refiner.blocks.0.norm1.weight".to_owned(),
            vec![hidden],
        ),
        ("img_in.proj.weight".to_owned(), projection),
        (
            "final_layer.linear.weight".to_owned(),
            vec![in_channels * patch_volume, hidden],
        ),
        (
            "txt_in.input_embedder.weight".to_owned(),
            vec![hidden, context_input],
        ),
        ("double_blocks.0.attn.weight".to_owned(), vec![hidden, hidden]),
        ("double_blocks.1.attn.weight".to_owned(), vec![hidden, hidden]),
        ("single_blocks.0.attn.weight".to_owned(), vec![hidden, hidden]),
    ];
    if matches!(
        spec.variant,
        HunyuanVideoVariant::Video
            | HunyuanVideoVariant::VideoI2V
            | HunyuanVideoVariant::Video15
            | HunyuanVideoVariant::Video15SrDistilled
    ) {
        shapes.extend([
            ("vector_in.in_layer.weight".to_owned(), vec![hidden, 768]),
            ("guidance_in.in_layer.weight".to_owned(), vec![hidden, 256]),
            ("byt5_in.fc1.weight".to_owned(), vec![2_048, 1_472]),
            ("time_r_in.in_layer.weight".to_owned(), vec![hidden, 256]),
        ]);
    }
    if spec.variant == HunyuanVideoVariant::Image21 {
        shapes.extend([
            ("guidance_in.in_layer.weight".to_owned(), vec![hidden, 256]),
            ("byt5_in.fc1.weight".to_owned(), vec![2_048, 1_472]),
            ("time_r_in.in_layer.weight".to_owned(), vec![hidden, 256]),
        ]);
    }
    if matches!(
        spec.variant,
        HunyuanVideoVariant::Video15 | HunyuanVideoVariant::Video15SrDistilled
    ) {
        shapes.extend([
            ("vision_in.proj.0.weight".to_owned(), vec![1_152]),
            ("cond_type_embedding.weight".to_owned(), vec![3, hidden]),
        ]);
    }
    shapes
}

fn mapping_shapes() -> Vec<(String, Vec<u64>)> {
    vec![
        ("img_in.proj.weight".to_owned(), vec![1]),
        ("final_layer.linear.weight".to_owned(), vec![2, 2]),
        ("final_layer.linear.bias".to_owned(), vec![2]),
        ("txt_in.input_embedder.weight".to_owned(), vec![2, 2]),
        ("txt_in.input_embedder.bias".to_owned(), vec![2]),
        (
            "txt_in.individual_token_refiner.blocks.0.norm1.weight".to_owned(),
            vec![1],
        ),
        ("txt_in.t_embedder.mlp.0.weight".to_owned(), vec![1]),
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.hunyuan.weight".to_owned(), vec![1]),
    ]
}

fn mapping_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    _spec: RowSpec,
    layout: HunyuanVideoLayout,
    dtype: DType,
    omitted: Option<&str>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let mut source = BTreeMap::new();
    for (key, shape) in mapping_shapes() {
        if omitted == Some(key.as_str()) {
            continue;
        }
        let source_key = if key.starts_with("vae.") || key.starts_with("text_encoders.") {
            key.clone()
        } else {
            format!("{prefix}{key}")
        };
        let elements = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key == "txt_in.input_embedder.bias" {
            vec![1.0, -1.0]
        } else if key == "final_layer.linear.bias" {
            vec![0.1, -0.1]
        } else if shape == [2, 2] {
            vec![1.0, 0.0, 0.0, 1.0]
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
        HunyuanVideoLayout::PrefixedNative,
        dtype,
        omitted,
    )
    .map_err(|error| ModelFamilyError::InvalidSelectorOutput(error.to_string()))?;
    source.retain(|key, _| {
        key.starts_with("model.diffusion_model.") && !key.contains("txt_in.t_embedder")
    });
    map_model_weights(
        spec.registration.definition,
        spec.artifact_digest,
        source,
    )
}

fn assert_configuration(
    spec: RowSpec,
    configuration: HunyuanVideoConfiguration,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(configuration.variant, spec.variant);
    assert_eq!(configuration.hidden_size, 256);
    assert_eq!(configuration.number_of_heads, 2);
    assert_eq!(configuration.double_block_depth, 2);
    assert_eq!(configuration.single_block_depth, 1);
    assert_eq!(configuration.out_channels, configuration.in_channels);
    assert_eq!(configuration.sampling_shift, spec.sampling_shift);
    assert_eq!(configuration.memory_usage_factor, spec.memory_usage_factor);
    assert_eq!(configuration.supported_dtypes, spec.supported_dtypes);
    match spec.variant {
        HunyuanVideoVariant::Image21 => {
            assert_eq!(configuration.in_channels, 64);
            assert_eq!(configuration.patch_size, [2, 2, 1]);
            assert_eq!(configuration.patch_rank, 2);
            assert_eq!(configuration.context_input_dimension, 3_584);
            assert_eq!(configuration.axes_dimensions, [0, 64, 64]);
            assert_eq!(configuration.vector_input_dimension, None);
            assert!(configuration.guidance_embedding);
            assert!(configuration.byt5_conditioning);
            assert!(configuration.mean_flow);
        }
        HunyuanVideoVariant::Image21Refiner => {
            assert_eq!(configuration.in_channels, 64);
            assert_eq!(configuration.patch_size, [1, 1, 1]);
            assert_eq!(configuration.patch_rank, 3);
            assert_eq!(configuration.context_input_dimension, 4_096);
            assert_eq!(configuration.axes_dimensions, [16, 56, 56]);
            assert_eq!(configuration.vector_input_dimension, None);
            assert!(!configuration.guidance_embedding);
            assert!(!configuration.byt5_conditioning);
            assert!(!configuration.mean_flow);
        }
        HunyuanVideoVariant::Video => {
            assert_eq!(configuration.in_channels, 16);
            assert_eq!(configuration.patch_size, [1, 2, 2]);
            assert_eq!(configuration.patch_rank, 3);
            assert_eq!(configuration.context_input_dimension, 4_096);
            assert_eq!(configuration.axes_dimensions, [16, 56, 56]);
            assert_eq!(configuration.vector_input_dimension, Some(768));
            assert!(configuration.guidance_embedding);
            assert!(configuration.byt5_conditioning);
            assert!(configuration.mean_flow);
        }
        HunyuanVideoVariant::Video15 => {
            assert_eq!(configuration.in_channels, 32);
            assert_eq!(configuration.patch_size, [1, 2, 2]);
            assert_eq!(configuration.patch_rank, 3);
            assert_eq!(configuration.context_input_dimension, 3_584);
            assert_eq!(configuration.vector_input_dimension, Some(768));
            assert_eq!(configuration.vision_input_dimension, Some(1_152));
            assert!(configuration.condition_type_embedding);
            assert!(configuration.mean_flow_sum);
            assert!(configuration.guidance_embedding);
            assert!(configuration.byt5_conditioning);
            assert!(configuration.mean_flow);
        }
        HunyuanVideoVariant::Video15SrDistilled => {
            assert_eq!(configuration.in_channels, 98);
            assert_eq!(configuration.patch_size, [1, 2, 2]);
            assert_eq!(configuration.patch_rank, 3);
            assert_eq!(configuration.context_input_dimension, 3_584);
            assert_eq!(configuration.axes_dimensions, [16, 56, 56]);
            assert_eq!(configuration.vector_input_dimension, Some(768));
            assert_eq!(configuration.vision_input_dimension, Some(1_152));
            assert!(configuration.condition_type_embedding);
            assert!(configuration.mean_flow_sum);
            assert!(configuration.guidance_embedding);
            assert!(configuration.byt5_conditioning);
            assert!(configuration.mean_flow);
        }
        HunyuanVideoVariant::VideoI2V => {
            assert_eq!(configuration.in_channels, 33);
            assert_eq!(configuration.patch_size, [1, 2, 2]);
            assert_eq!(configuration.patch_rank, 3);
            assert_eq!(configuration.context_input_dimension, 4_096);
            assert_eq!(configuration.axes_dimensions, [16, 56, 56]);
            assert_eq!(configuration.vector_input_dimension, Some(768));
            assert_eq!(configuration.vision_input_dimension, None);
            assert!(!configuration.condition_type_embedding);
            assert!(!configuration.mean_flow_sum);
            assert!(configuration.guidance_embedding);
            assert!(configuration.byt5_conditioning);
            assert!(configuration.mean_flow);
        }
        unsupported => {
            return Err(format!("unsupported focused Hunyuan variant {unsupported:?}").into());
        }
    }
    Ok(())
}

fn layout_prefix(layout: HunyuanVideoLayout) -> &'static str {
    match layout {
        HunyuanVideoLayout::PrefixedNative => "model.diffusion_model.",
        HunyuanVideoLayout::SavedModel => "model.",
        HunyuanVideoLayout::StandaloneNative => "",
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

fn probe_through_model_store(spec: RowSpec) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("hunyuan.safetensors");
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
    let key = ArtifactKey::new(spec.fixture, "hunyuan.safetensors")?;
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
    for (key, shape) in detector_shapes(spec) {
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
        .ok_or("Hunyuan image/video checkpoint is missing")?;
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

fn rng_transaction() -> Result<sim_tensor::RngTransaction, Box<dyn std::error::Error>> {
    let address = RngStreamAddress::new(
        "hunyuan-refiner",
        "attempt-1",
        "conditioning",
        0,
        "noise-augmentation",
        0,
        0,
        RetryRngPolicy::Replay,
    )?;
    Ok(RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        13,
        address,
    )?
    .begin(None)?)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
