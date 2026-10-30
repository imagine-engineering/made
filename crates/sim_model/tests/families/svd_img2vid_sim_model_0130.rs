use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions,
    ParserLimits, PatchApplication, PatchGraph, PatchKind, PatchOperation, PatchTarget,
    build_model_family_for_probe, describe_model_family,
    generated_svd_img2vid_sim_model_0130 as svd,
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

const ARTIFACT_DIGEST: &str = "1301301301301301301301301301301301301301301301301301301301301301";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9130",
    identifier: "SVDImg2VidAmbiguousFixture",
    ..svd::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    svd::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 192,
        source_architecture: "model_base.SVDImg2VidAmbiguousFixture",
        ..svd::MODEL_FAMILY_REGISTRATION
    },
];
static SVD_IMG2VID_REGISTRATIONS: [ModelFamilyRegistration; 1] =
    [svd::MODEL_FAMILY_REGISTRATION];

pub(super) struct VideoFamilyCase {
    pub definition: &'static ModelFamilyDefinition,
    pub registration: ModelFamilyRegistration,
    pub registrations: &'static [ModelFamilyRegistration],
    pub ambiguous_registrations: &'static [ModelFamilyRegistration],
    pub identifier: &'static str,
    pub feature_id: &'static str,
    pub fixture: &'static str,
    pub module: &'static str,
    pub source_ordinal: u16,
    pub source_architecture: &'static str,
    pub architecture_version: &'static str,
    pub projection_sha256: &'static str,
    pub adm_input_channels: u64,
    pub vae_source_prefix: &'static str,
    pub conditioning_checkpoint: &'static str,
    pub output_checkpoint: &'static str,
    pub validate_configuration: fn(&ModelProbe) -> Result<(), ModelFamilyError>,
}

#[test]
fn val_model_family_row_001_svd_img2vid_source_projection_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    run_source_validation(&case())
}

#[test]
fn val_model_family_row_001_svd_img2vid_execution_failures_and_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    run_execution_validation(&case())
}

fn case() -> VideoFamilyCase {
    VideoFamilyCase {
        definition: &svd::MODEL_FAMILY,
        registration: svd::MODEL_FAMILY_REGISTRATION,
        registrations: &SVD_IMG2VID_REGISTRATIONS,
        ambiguous_registrations: &AMBIGUOUS_REGISTRATIONS,
        identifier: svd::MODEL_FAMILY_IDENTIFIER,
        feature_id: svd::MODEL_FAMILY_FEATURE_ID,
        fixture: svd::MODEL_FAMILY_FIXTURE,
        module: "svd_img2vid_comfy_model_0130",
        source_ordinal: svd::MODEL_FAMILY_SOURCE_ORDINAL,
        source_architecture: "model_base.SVD_img2vid",
        architecture_version: "svd-image-to-video-unet-v1",
        projection_sha256: svd::MODEL_FAMILY_PROJECTION_SHA256,
        adm_input_channels: svd::MODEL_FAMILY_ADM_IN_CHANNELS,
        vae_source_prefix: "first_stage_model.",
        conditioning_checkpoint: "adm_conditioning_projection",
        output_checkpoint: "video_latent_output",
        validate_configuration: |probe| svd::configuration_for_probe(probe).map(|_| ()),
    }
}

pub(super) fn run_source_validation(
    case: &VideoFamilyCase,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(case.definition.identifier, case.identifier);
    assert_eq!(case.definition.feature_id, case.feature_id);
    assert_eq!(case.registration.source_ordinal, case.source_ordinal);
    assert_eq!(case.registration.source_architecture, case.source_architecture);
    assert_eq!(case.definition.architecture_version, case.architecture_version);
    assert_eq!(case.definition.latent_feature_id, "COMFY-MODEL-0045");
    assert_eq!(case.definition.latent_identifier, "SD15");

    let descriptor = describe_model_family(case.definition)?;
    assert_eq!(descriptor.identifier, case.identifier);
    assert_eq!(descriptor.family, case.feature_id);
    assert_eq!(descriptor.architecture_version, case.architecture_version);
    assert_eq!(descriptor.latent_format, "SD15");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);
    assert!(descriptor.component_graph.iter().any(|component| component.identifier == "vision_encoder"));

    let probe = ModelProbe::from_parsed_facts(parsed_facts(case, DType::F32, false, false, case.adm_input_channels))?;
    (case.validate_configuration)(&probe)?;
    let normalized = probe.normalized_configuration()?;
    assert_eq!(normalized.fact("model_channels"), Some(&sim_model::ModelConfigurationValue::Unsigned(320)));
    assert_eq!(normalized.fact("in_channels"), Some(&sim_model::ModelConfigurationValue::Unsigned(8)));
    assert_eq!(normalized.fact("context_dim"), Some(&sim_model::ModelConfigurationValue::Unsigned(1_024)));
    assert_eq!(normalized.fact("adm_in_channels"), Some(&sim_model::ModelConfigurationValue::Unsigned(case.adm_input_channels)));
    assert_eq!(normalized.fact("use_temporal_attention"), Some(&sim_model::ModelConfigurationValue::Boolean(true)));
    assert_eq!(normalized.fact("use_temporal_resblock"), Some(&sim_model::ModelConfigurationValue::Boolean(true)));

    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        sha256(&std::fs::read(repository.join(svd::MODEL_FAMILY_SOURCE_PATH))?),
        svd::MODEL_FAMILY_SOURCE_SHA256
    );
    let fixture_directory = fixture_directory(case);
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_directory.join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], case.feature_id);
    assert_eq!(provenance["source_symbol"], case.identifier);
    assert_eq!(provenance["source_ordinal"], case.source_ordinal);
    assert_eq!(provenance["source_architecture"], case.source_architecture);
    assert_eq!(provenance["catalog_projection_sha256"], case.projection_sha256);
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("video-family source projection must be a string")?;
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    for source in provenance["source_files"]
        .as_array()
        .ok_or("source_files must be an array")?
    {
        let path = source["path"].as_str().ok_or("source path must be a string")?;
        let expected = source["sha256"].as_str().ok_or("source sha256 must be a string")?;
        assert_eq!(sha256(&std::fs::read(repository.join(path))?), expected);
    }

    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog must contain models")?
        .iter()
        .find(|row| row["feature_id"] == case.feature_id)
        .ok_or("video-family catalog row is missing")?;
    assert_eq!(row["source_ordinal"], case.source_ordinal);
    assert_eq!(row["static"]["unet_config"]["value"]["in_channels"], 8);
    assert_eq!(row["static"]["unet_config"]["value"]["adm_in_channels"], case.adm_input_channels);
    assert_eq!(sha256(&serde_json::to_vec(row)?), case.projection_sha256);

    let row_source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("src/families/{}.rs", case.module)),
    )?;
    for owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "struct CpuWorkspaceAuthority",
        "std::fs",
        "unsafe ",
    ] {
        assert!(!row_source.contains(owner));
    }
    super::write_model_family_row_artifact(
        case.fixture,
        case.feature_id,
        case.identifier,
        case.source_ordinal,
        case.module,
        &[
            "source-provenance-catalog-and-ownership",
            "native-key-derived-normalized-video-configuration",
            "model-store-detection-and-native-layout",
            "transactional-model-vae-vision-mapping-and-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-family-exclusion-and-diffusers-rejection",
        ],
    )?;
    Ok(())
}

pub(super) fn run_execution_validation(
    case: &VideoFamilyCase,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(case.registrations)?;
    let model_store_probe = probe_through_model_store(case)?;
    assert_eq!(model_store_probe.unet_prefix_selection()?.prefix(), "model.diffusion_model.");
    assert_eq!(registry.detect(&model_store_probe)?.identity.feature_id(), case.feature_id);

    let probe = ModelProbe::from_parsed_facts(parsed_facts(case, DType::F32, false, false, case.adm_input_channels))?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.source_ordinal(), case.source_ordinal);
    assert_eq!(resolved.profile().latent_identifier, "SD15");
    assert!(resolved.clip_target().candidates().is_empty());

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(case, &backend, &context, DType::F32, false)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    assert!(mapped.component("model").is_some_and(|model| model.contains_key("native.input_blocks.1.0.in_layers.2.weight")));
    assert!(mapped.component("vae").is_some());
    assert!(mapped.component("vision_encoder").is_some());

    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        ARTIFACT_DIGEST,
        &source,
    )?;
    let model = build_model_family_for_probe(
        &registry,
        &probe,
        weights,
        options(DType::F32, u64::MAX, DeviceKind::Cpu),
    )?;
    let required_memory = model.memory_estimate().total_bytes;
    assert!(required_memory > 0);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_checkpoint(&backend, &context, &checkpoints, case.conditioning_checkpoint, &[1.0, 2.0])?;
    assert_checkpoint(&backend, &context, &checkpoints, case.output_checkpoint, &[0.0, 0.96402675])?;

    let replace_then_add = ordered_patch_graph(case, true)?.apply(&backend, model.weights(), &context)?;
    let add_then_replace = ordered_patch_graph(case, false)?.apply(&backend, model.weights(), &context)?;
    let first = model.with_weights(replace_then_add)?.forward_checkpoints(&backend, &input, &context)?;
    let second = model.with_weights(add_then_replace)?.forward_checkpoints(&backend, &input, &context)?;
    assert_ne!(
        checkpoint_values(&backend, &context, &first, case.output_checkpoint)?,
        checkpoint_values(&backend, &context, &second, case.output_checkpoint)?
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
            options(DType::F32, required_memory - 1, DeviceKind::Cpu),
        ),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == required_memory && budget == required_memory - 1
    ));

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let typed_probe = ModelProbe::from_parsed_facts(parsed_facts(case, dtype, false, false, case.adm_input_channels))?;
        let typed_resolved = registry.resolve(&typed_probe)?;
        let typed_source = source_tensors(case, &backend, &context, dtype, false)?;
        let typed_weights = typed_resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &typed_source,
        )?;
        build_model_family_for_probe(
            &registry,
            &typed_probe,
            typed_weights,
            options(dtype, u64::MAX, DeviceKind::Cpu),
        )?;
    }
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
            options(DType::F32, u64::MAX, DeviceKind::Metal),
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let partial_probe = ModelProbe::from_parsed_facts(parsed_facts(case, DType::F32, true, false, case.adm_input_channels))?;
    assert!(matches!(registry.resolve(&partial_probe), Err(ModelFamilyError::ModelLayoutSelection(_))));
    let invalid_probe = ModelProbe::from_parsed_facts(parsed_facts(case, DType::F32, false, true, case.adm_input_channels))?;
    assert!(matches!(registry.detect(&invalid_probe), Err(ModelFamilyError::NoDetectionMatch)));
    let wrong_adm = if case.adm_input_channels == 256 { 768 } else { 256 };
    let other_family_probe = ModelProbe::from_parsed_facts(parsed_facts(case, DType::F32, false, false, wrong_adm))?;
    assert!(matches!(registry.detect(&other_family_probe), Err(ModelFamilyError::NoDetectionMatch)));
    let ambiguous = ModelFamilyRegistry::checked_registrations(case.ambiguous_registrations)?;
    assert!(matches!(ambiguous.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })));

    let diffusers_probe = ModelProbe::from_parsed_facts(diffusers_facts(case.adm_input_channels))?;
    assert!(matches!(registry.detect(&diffusers_probe), Err(ModelFamilyError::NoDetectionMatch)));
    assert!(matches!(
        (case.validate_configuration)(&diffusers_probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers detector table")
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

fn parsed_facts(
    case: &VideoFamilyCase,
    dtype: DType,
    omit_forward_weight: bool,
    invalid_input: bool,
    adm_input_channels: u64,
) -> ModelParsedFacts {
    let tensors = model_shapes(case, omit_forward_weight, invalid_input, adm_input_channels)
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
            metadata: BTreeMap::from([("model_family".to_owned(), "misleading-sd-x4".to_owned())]),
        }],
    }
}

fn model_shapes(
    case: &VideoFamilyCase,
    omit_forward_weight: bool,
    invalid_input: bool,
    adm_input_channels: u64,
) -> BTreeMap<String, Vec<u64>> {
    let prefix = "model.diffusion_model.";
    let mut shapes = BTreeMap::from([
        (format!("{prefix}input_blocks.0.0.weight"), if invalid_input { vec![320, 9, 3, 3] } else { vec![320, 8, 3, 3] }),
        (format!("{prefix}label_emb.0.0.weight"), vec![2, adm_input_channels]),
        (format!("{prefix}input_blocks.1.0.in_layers.0.weight"), vec![2]),
        (format!("{prefix}input_blocks.1.0.out_layers.3.weight"), vec![320, 2]),
        (format!("{prefix}input_blocks.1.0.in_layers.2.weight"), vec![2, 2]),
        (format!("{prefix}input_blocks.1.1.proj_in.weight"), vec![2, 2]),
        (format!("{prefix}input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"), vec![2, 1_024]),
        (format!("{prefix}input_blocks.1.1.time_stack.0.attn1.to_q.weight"), vec![2, 2]),
        (format!("{prefix}input_blocks.1.1.time_stack.0.attn2.to_q.weight"), vec![2, 2]),
        (format!("{prefix}middle_block.1.transformer_blocks.0.attn1.to_out.0.weight"), vec![2, 2]),
        (format!("{prefix}output_blocks.0.0.in_layers.0.weight"), vec![2]),
        (format!("{prefix}output_blocks.0.0.in_layers.2.weight"), vec![2, 2]),
        (format!("{prefix}out.2.weight"), vec![4, 320, 3, 3]),
        (format!("{}decoder.weight", case.vae_source_prefix), vec![1]),
        ("conditioner.embedders.0.open_clip.model.visual.proj.weight".to_owned(), vec![1]),
    ]);
    if omit_forward_weight {
        shapes.remove(&format!("{prefix}output_blocks.0.0.in_layers.2.weight"));
    }
    shapes
}

fn diffusers_facts(adm_input_channels: u64) -> ModelParsedFacts {
    let tensors = BTreeMap::from([
        ("conv_in.weight".to_owned(), ModelParsedTensorFact { shape: vec![320, 8, 3, 3], storage_dtype: "float32".to_owned() }),
        ("conv_out.weight".to_owned(), ModelParsedTensorFact { shape: vec![4, 320, 3, 3], storage_dtype: "float32".to_owned() }),
        ("class_embedding.linear_1.weight".to_owned(), ModelParsedTensorFact { shape: vec![2, adm_input_channels], storage_dtype: "float32".to_owned() }),
        ("down_blocks.0.resnets.0.conv1.weight".to_owned(), ModelParsedTensorFact { shape: vec![2, 2], storage_dtype: "float32".to_owned() }),
    ]);
    ModelParsedFacts { tensors, formats: vec![ModelParsedFormatFact { identity: "safetensors".to_owned(), metadata: BTreeMap::new() }] }
}

fn source_tensors(
    case: &VideoFamilyCase,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    omit_forward_weight: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    model_shapes(case, omit_forward_weight, false, case.adm_input_channels)
        .into_iter()
        .map(|(key, shape)| {
            let elements = usize::try_from(shape.iter().product::<u64>())?;
            let values = match key.as_str() {
                "model.diffusion_model.input_blocks.1.0.in_layers.2.weight" => vec![1.0, 0.0, 0.0, 1.0],
                "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight" => vec![2.0, 0.0, 0.0, 0.5],
                "model.diffusion_model.output_blocks.0.0.in_layers.2.weight" => vec![1.0, 1.0, 1.0, -1.0],
                _ => vec![0.0; elements],
            };
            Ok((key, tensor(backend, &shape, &values, dtype, context)?))
        })
        .collect()
}

fn probe_through_model_store(case: &VideoFamilyCase) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let file_name = format!("{}.safetensors", case.fixture);
    let model_path = directory.path().join(&file_name);
    write_safetensors(case, &model_path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical("video-family-row", "checkpoints", directory.path(), ["safetensors"])?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("video-family-row", &file_name)?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(case: &VideoFamilyCase, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::json!({"model_family": "misleading-sd-x4"}));
    let mut data = Vec::new();
    for (key, shape) in model_shapes(case, false, false, case.adm_input_channels) {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("fixture shape overflow")
        })?;
        let bytes = usize::try_from(elements.checked_mul(4).ok_or("fixture byte overflow")?)?;
        data.resize(data.len().checked_add(bytes).ok_or("fixture length overflow")?, 0);
        header.insert(key, serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
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
    Ok(tensor_from_f32_with_context_exact_native(backend, shape, values, dtype, backend.device(), context)?)
}

fn options(dtype: DType, memory_budget_bytes: u64, device: DeviceKind) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions { dtype, device, activation_elements: 2, memory_budget_bytes, allow_unexpected_weights: false }
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = checkpoint_values(backend, context, checkpoints, name)?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5, "{name}: {actual} != {expected}");
    }
    Ok(())
}

fn checkpoint_values(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let checkpoint = checkpoints.iter().find(|checkpoint| checkpoint.name == name).ok_or("checkpoint is missing")?;
    Ok(tensor_to_f32_with_context_exact_native(backend, &checkpoint.tensor, context)?)
}

fn ordered_patch_graph(case: &VideoFamilyCase, replace_first: bool) -> Result<PatchGraph, Box<dyn std::error::Error>> {
    let replacement = PatchOperation {
        identifier: format!("{}-replacement", case.fixture),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: format!("{}-addition", case.fixture),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Add,
        }],
    };
    Ok(PatchGraph::checked(ARTIFACT_DIGEST, if replace_first { vec![replacement, addition] } else { vec![addition, replacement] })?)
}

fn fixture_directory(case: &VideoFamilyCase) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(case.fixture)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
