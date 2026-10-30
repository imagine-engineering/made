use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_depthanything3_sim_model_0075 as depthanything3,
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

const ARTIFACT_DIGEST: &str = "0750750750750750750750750750750750750750750750750750750750750750";
const RESOLVED_MEMORY_BYTES: u64 = 903_320;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9075",
    identifier: "DepthAnything3AmbiguousFixture",
    ..depthanything3::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    depthanything3::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 193,
        source_architecture: "model_base.DepthAnything3AmbiguousFixture",
        ..depthanything3::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_depthanything3_source_profiles_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(depthanything3::MODEL_FAMILY_IDENTIFIER, "DepthAnything3");
    assert_eq!(depthanything3::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0075");
    assert_eq!(
        depthanything3::MODEL_FAMILY_FIXTURE,
        "depthanything3-comfy-model-0075"
    );
    assert_eq!(depthanything3::MODEL_FAMILY_SOURCE_ORDINAL, 93);
    assert_eq!(depthanything3::MODEL_FAMILY_REGISTRATION.source_ordinal, 93);
    assert_eq!(
        depthanything3::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.DepthAnything3"
    );
    assert_eq!(depthanything3::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);
    assert_eq!(depthanything3::MODEL_FAMILY_PATCH_SIZE, 14);
    assert_eq!(depthanything3::MODEL_FAMILY_IMAGE_SIZE, 518);

    let descriptor = describe_model_family(&depthanything3::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "DepthAnything3");
    assert_eq!(descriptor.family, "COMFY-MODEL-0075");
    assert_eq!(
        descriptor.architecture_version,
        "depth-anything-3-dinov2-v1"
    );
    assert_eq!(descriptor.latent_format, "LatentFormat");
    assert_eq!(
        descriptor.supported_dtypes,
        ["bfloat16", "float16", "float32"]
    );
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);

    let mono = ModelProbe::from_parsed_facts(parsed_facts(
        DType::F32,
        depthanything3::DepthAnything3Backbone::VitLarge,
        ProbeOptions::default().sky_head(true),
    ))?;
    let mono = depthanything3::configuration_for_probe(&mono)?;
    assert_eq!(
        mono.backbone,
        depthanything3::DepthAnything3Backbone::VitLarge
    );
    assert_eq!(mono.backbone.identifier(), "vitl");
    assert_eq!(mono.hidden_size, 1_024);
    assert_eq!(mono.layer_count, 24);
    assert_eq!(mono.attention_heads, 16);
    assert_eq!(mono.patch_size, 14);
    assert_eq!(mono.image_size, 518);
    assert_eq!(mono.head, depthanything3::DepthAnything3Head::Dpt);
    assert_eq!(mono.head_output_dimension, 2);
    assert_eq!(mono.output_layers, [4, 11, 17, 23]);
    assert!(mono.use_sky_head);
    assert!(!mono.concatenate_camera_token);
    assert_eq!(mono.qknorm_start, None);

    let multiview = ModelProbe::from_parsed_facts(parsed_facts(
        DType::F32,
        depthanything3::DepthAnything3Backbone::VitSmall,
        ProbeOptions::default()
            .auxiliary_head(true)
            .camera(true)
            .qknorm_start(Some(6)),
    ))?;
    let multiview = depthanything3::configuration_for_probe(&multiview)?;
    assert_eq!(
        multiview.backbone,
        depthanything3::DepthAnything3Backbone::VitSmall
    );
    assert_eq!(multiview.attention_heads, 6);
    assert_eq!(multiview.head, depthanything3::DepthAnything3Head::DualDpt);
    assert_eq!(multiview.head_output_dimension, 2);
    assert_eq!(multiview.output_layers, [5, 7, 9, 11]);
    assert!(!multiview.use_sky_head);
    assert!(multiview.concatenate_camera_token);
    assert_eq!(multiview.qknorm_start, Some(6));
    assert_eq!(multiview.alternate_attention_start, Some(6));
    assert_eq!(multiview.rope_start, Some(6));
    assert!(multiview.has_camera_encoder);
    assert_eq!(multiview.camera_encoder_dimension, Some(384));
    assert!(multiview.has_camera_decoder);
    assert_eq!(multiview.camera_decoder_dimension, Some(768));

    for backbone in [
        depthanything3::DepthAnything3Backbone::VitBase,
        depthanything3::DepthAnything3Backbone::VitGiant,
    ] {
        let probe = ModelProbe::from_parsed_facts(parsed_facts(
            DType::F32,
            backbone,
            ProbeOptions::default(),
        ))?;
        let configuration = depthanything3::configuration_for_probe(&probe)?;
        assert_eq!(configuration.backbone, backbone);
        assert_eq!(configuration.hidden_size, backbone.hidden_size());
        assert_eq!(configuration.layer_count, backbone.layer_count());
        assert_eq!(configuration.attention_heads, backbone.attention_heads());
    }

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(
            repository.join(depthanything3::MODEL_FAMILY_SOURCE_PATH)
        )?),
        depthanything3::MODEL_FAMILY_SOURCE_SHA256
    );
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory().join("provenance.json"))?)?;
    assert_eq!(
        provenance["feature_id"],
        depthanything3::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(provenance["source_symbol"], "DepthAnything3");
    assert_eq!(provenance["source_ordinal"], 93);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("DepthAnything3 source projection must be a string")?;
    assert_eq!(
        sha256(projection.as_bytes()),
        provenance["source_projection_sha256"]
    );
    for source in provenance["source_files"]
        .as_array()
        .ok_or("DepthAnything3 source_files must be an array")?
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
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog must contain models")?
        .iter()
        .find(|row| row["feature_id"] == depthanything3::MODEL_FAMILY_FEATURE_ID)
        .ok_or("DepthAnything3 catalog row is missing")?;
    assert_eq!(row["source_ordinal"], 93);
    assert_eq!(
        row["static"]["unet_config"]["value"]["image_model"],
        "DepthAnything3"
    );
    assert_eq!(
        sha256(&serde_json::to_vec(row)?),
        depthanything3::MODEL_FAMILY_PROJECTION_SHA256
    );
    assert_eq!(
        provenance["catalog_projection_sha256"],
        depthanything3::MODEL_FAMILY_PROJECTION_SHA256
    );

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/depthanything3_sim_model_0075.rs"),
    )?;
    for owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct ModelProbe",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct MemoryEstimator",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(owner));
    }
    super::write_model_family_row_artifact(
        depthanything3::MODEL_FAMILY_FIXTURE,
        depthanything3::MODEL_FAMILY_FEATURE_ID,
        depthanything3::MODEL_FAMILY_IDENTIFIER,
        depthanything3::MODEL_FAMILY_SOURCE_ORDINAL,
        "depthanything3_comfy_model_0075",
        &[
            "source-provenance-catalog-and-ownership",
            "four-source-exact-dinov2-backbone-profiles",
            "dpt-dualdpt-camera-and-qknorm-configuration",
            "model-store-native-prefix-detection",
            "transactional-component-mapping-and-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-vanilla-dinov2-ambiguous-and-layout-rejection",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_depthanything3_model_store_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[depthanything3::MODEL_FAMILY_REGISTRATION])?;
    let probe = probe_through_model_store()?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert_eq!(
        probe.unet_prefix_selection()?.prefix(),
        "model.diffusion_model."
    );
    let resolved = registry.resolve(&probe)?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        "COMFY-MODEL-0075"
    );
    assert_eq!(resolved.source_ordinal(), 93);
    assert_eq!(resolved.profile().latent_identifier, "LatentFormat");
    assert!(resolved.clip_target().candidates().is_empty());

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&backend, &context, DType::F32, ProbeOptions::default())?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(mapped.component("model").is_some_and(|model| {
        model.contains_key("native.backbone.embeddings.patch_embeddings.projection.weight")
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
        "depth_refinement",
        &[1.4621172, 0.8807971],
    )?;
    assert_checkpoint(
        &backend,
        &context,
        &checkpoints,
        "depth_output",
        &[0.0, 0.96402675],
    )?;

    let patch = PatchGraph::checked(
        ARTIFACT_DIGEST,
        vec![PatchOperation {
            identifier: "depthanything3-refinement-delta".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.head.scratch.refinenet1.out_conv.weight".to_owned(),
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
        "depth_output",
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
            &first.tensors()["native.head.scratch.refinenet1.out_conv.weight"],
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            &second.tensors()["native.head.scratch.refinenet1.out_conv.weight"],
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
fn val_model_family_row_001_depthanything3_dtype_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[depthanything3::MODEL_FAMILY_REGISTRATION])?;
    let probe = ModelProbe::from_parsed_facts(parsed_facts(
        DType::F32,
        depthanything3::DepthAnything3Backbone::VitSmall,
        ProbeOptions::default(),
    ))?;
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );

    for dtype in [DType::Bf16, DType::F16, DType::F32] {
        let source = source_tensors(&backend, &context, dtype, ProbeOptions::default())?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &source,
        )?;
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            options(dtype, RESOLVED_MEMORY_BYTES),
        )?;
    }

    let source = source_tensors(&backend, &context, DType::F32, ProbeOptions::default())?;
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

    let partial_options = ProbeOptions::default()
        .auxiliary_head(true)
        .omit_output(true);
    let partial_probe = ModelProbe::from_parsed_facts(parsed_facts(
        DType::F32,
        depthanything3::DepthAnything3Backbone::VitSmall,
        partial_options,
    ))?;
    let partial_resolved = registry.resolve(&partial_probe)?;
    let partial = source_tensors(&backend, &context, DType::F32, partial_options)?;
    assert!(matches!(
        partial_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &partial,
        ),
        Err(ModelFamilyError::MissingComponentKey { component, key })
            if component == "model" && key == "native.head.scratch.output_conv2.2.weight"
    ));

    let invalid_hidden = ModelProbe::from_parsed_facts(parsed_facts_with_hidden_size(512, 12))?;
    assert!(matches!(
        registry.resolve(&invalid_hidden),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("unsupported backbone hidden size")
    ));
    let invalid_depth = ModelProbe::from_parsed_facts(parsed_facts_with_hidden_size(384, 11))?;
    assert!(matches!(
        registry.resolve(&invalid_depth),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("requires 12 consecutive layers")
    ));
    let qknorm_gap = ModelProbe::from_parsed_facts(parsed_facts(
        DType::F32,
        depthanything3::DepthAnything3Backbone::VitSmall,
        ProbeOptions::default()
            .camera(true)
            .qknorm_start(Some(6))
            .qknorm_gap(Some(8)),
    ))?;
    assert!(matches!(
        registry.resolve(&qknorm_gap),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("contiguous suffix")
    ));

    let vanilla_dinov2 = ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "model.diffusion_model.backbone.embeddings.patch_embeddings.projection.weight"
                .to_owned(),
            vec![384, 3, 14, 14],
        )]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        registry.detect(&vanilla_dinov2),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut unexpected_facts = parsed_facts(
        DType::F32,
        depthanything3::DepthAnything3Backbone::VitSmall,
        ProbeOptions::default(),
    );
    unexpected_facts.tensors.insert(
        "unexpected.weight".to_owned(),
        ModelParsedTensorFact {
            shape: vec![1],
            storage_dtype: DType::F32.catalog_name().to_owned(),
        },
    );
    let unexpected_probe = ModelProbe::from_parsed_facts(unexpected_facts)?;
    let unexpected_resolved = registry.resolve(&unexpected_probe)?;
    let mut unexpected = source.clone();
    unexpected.insert(
        "unexpected.weight".to_owned(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        unexpected_resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
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

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
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

#[derive(Clone, Copy, Default)]
struct ProbeOptions {
    auxiliary_head: bool,
    camera: bool,
    sky_head: bool,
    omit_output: bool,
    qknorm_start: Option<usize>,
    qknorm_gap: Option<usize>,
}

impl ProbeOptions {
    fn auxiliary_head(mut self, value: bool) -> Self {
        self.auxiliary_head = value;
        self
    }

    fn camera(mut self, value: bool) -> Self {
        self.camera = value;
        self
    }

    fn sky_head(mut self, value: bool) -> Self {
        self.sky_head = value;
        self
    }

    fn omit_output(mut self, value: bool) -> Self {
        self.omit_output = value;
        self
    }

    fn qknorm_start(mut self, value: Option<usize>) -> Self {
        self.qknorm_start = value;
        self
    }

    fn qknorm_gap(mut self, value: Option<usize>) -> Self {
        self.qknorm_gap = value;
        self
    }
}

fn parsed_facts(
    dtype: DType,
    backbone: depthanything3::DepthAnything3Backbone,
    options: ProbeOptions,
) -> ModelParsedFacts {
    parsed_facts_from_shapes(dtype, model_shapes(backbone, options))
}

fn parsed_facts_with_hidden_size(hidden_size: u64, layer_count: usize) -> ModelParsedFacts {
    let mut shapes = base_model_shapes(hidden_size, layer_count, ProbeOptions::default());
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    parsed_facts_from_shapes(DType::F32, shapes)
}

fn parsed_facts_from_shapes(dtype: DType, shapes: Vec<(String, Vec<u64>)>) -> ModelParsedFacts {
    let mut tensors = BTreeMap::new();
    for (key, shape) in shapes {
        tensors.insert(
            format!("model.diffusion_model.{key}"),
            ModelParsedTensorFact {
                shape,
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }
    for key in [
        "first_stage_model.decoder.weight",
        "cond_stage_model.legacy.weight",
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

fn model_shapes(
    backbone: depthanything3::DepthAnything3Backbone,
    options: ProbeOptions,
) -> Vec<(String, Vec<u64>)> {
    base_model_shapes(backbone.hidden_size(), backbone.layer_count(), options)
}

fn base_model_shapes(
    hidden_size: u64,
    layer_count: usize,
    options: ProbeOptions,
) -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        (
            "backbone.embeddings.patch_embeddings.projection.weight".to_owned(),
            vec![hidden_size, 3, 14, 14],
        ),
        ("head.projects.0.weight".to_owned(), vec![2, 2]),
        ("head.projects.1.weight".to_owned(), vec![2, 2]),
        ("head.projects.2.weight".to_owned(), vec![2, 2]),
        ("head.projects.3.weight".to_owned(), vec![2, 2]),
        (
            "head.scratch.refinenet1.out_conv.weight".to_owned(),
            vec![2, 2],
        ),
    ];
    if !options.omit_output {
        shapes.push(("head.scratch.output_conv2.2.weight".to_owned(), vec![2, 2]));
    }
    for index in 0..layer_count {
        shapes.push((
            format!("backbone.encoder.layer.{index}.layer_scale1.lambda1"),
            vec![1],
        ));
    }
    if let Some(start) = options.qknorm_start {
        for index in start..layer_count {
            if options.qknorm_gap != Some(index) {
                shapes.push((
                    format!("backbone.encoder.layer.{index}.attention.q_norm.weight"),
                    vec![hidden_size],
                ));
            }
        }
    }
    if options.auxiliary_head {
        shapes.push((
            "head.scratch.refinenet1_aux.out_conv.weight".to_owned(),
            vec![2, 2],
        ));
    }
    if options.sky_head {
        shapes.push(("head.scratch.sky_output_conv2.0.weight".to_owned(), vec![1]));
    }
    if options.camera {
        shapes.extend([
            (
                "backbone.embeddings.camera_token".to_owned(),
                vec![1, 1, hidden_size],
            ),
            ("cam_enc.token_norm.weight".to_owned(), vec![hidden_size]),
            (
                "cam_enc.pose_branch.fc2.weight".to_owned(),
                vec![hidden_size, 2],
            ),
            ("cam_dec.fc_t.weight".to_owned(), vec![3, hidden_size * 2]),
        ]);
    }
    shapes.sort_by(|left, right| left.0.cmp(&right.0));
    shapes
}

fn source_tensors(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    options: ProbeOptions,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut source = BTreeMap::new();
    for (key, shape) in model_shapes(depthanything3::DepthAnything3Backbone::VitSmall, options) {
        let values = match key.as_str() {
            "head.projects.0.weight" => vec![1.0, 0.0, 0.0, 1.0],
            "head.scratch.refinenet1.out_conv.weight" => vec![2.0, 0.0, 0.0, 0.5],
            "head.scratch.output_conv2.2.weight" => vec![1.0, 1.0, 1.0, -1.0],
            _ => vec![0.0; usize::try_from(shape.iter().product::<u64>())?],
        };
        source.insert(
            format!("model.diffusion_model.{key}"),
            tensor(backend, &shape, &values, dtype, context)?,
        );
    }
    source.insert(
        "first_stage_model.decoder.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    source.insert(
        "cond_stage_model.legacy.weight".to_owned(),
        tensor(backend, &[1], &[1.0], dtype, context)?,
    );
    Ok(source)
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("depthanything3.safetensors");
    write_safetensors(&model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "depthanything3-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("depthanything3-row", "depthanything3.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut shapes = model_shapes(
        depthanything3::DepthAnything3Backbone::VitSmall,
        ProbeOptions::default(),
    )
    .into_iter()
    .map(|(key, shape)| (format!("model.diffusion_model.{key}"), shape))
    .collect::<Vec<_>>();
    shapes.extend([
        ("first_stage_model.decoder.weight".to_owned(), vec![1]),
        ("cond_stage_model.legacy.weight".to_owned(), vec![1]),
    ]);
    let mut data = Vec::new();
    for (key, shape) in shapes {
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
        allow_unexpected_weights: true,
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
        .ok_or("DepthAnything3 checkpoint is missing")?;
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
        identifier: "depthanything3-ordered-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.head.scratch.refinenet1.out_conv.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "depthanything3-ordered-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.head.scratch.refinenet1.out_conv.weight".to_owned(),
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
        .join(depthanything3::MODEL_FAMILY_FIXTURE)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
