use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelClipModelInvocation, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyRegistry, ModelProbe, ModelStateTransaction,
    ModelStore, NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_acestep15_sim_model_0062::{
        MODEL_FAMILY, MODEL_FAMILY_FEATURE_ID, MODEL_FAMILY_FIXTURE, MODEL_FAMILY_IDENTIFIER,
        MODEL_FAMILY_REGISTRATION, SOURCE_ARCHITECTURE, SOURCE_MEMORY_USAGE_FACTOR,
        SOURCE_SAMPLING_MULTIPLIER, SOURCE_SAMPLING_SHIFT,
    },
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        cast_to_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

static REGISTRATIONS: [sim_model::ModelFamilyRegistration; 1] =
    [MODEL_FAMILY_REGISTRATION];
static TIED_DEFINITIONS: [ModelFamilyDefinition; 2] = [
    MODEL_FAMILY,
    ModelFamilyDefinition {
        feature_id: "COMFY-MODEL-9062",
        identifier: "ACEStep15TieFixture",
        ..MODEL_FAMILY
    },
];

#[derive(Debug, Deserialize)]
struct FamilyFixture {
    fixture_id: String,
    feature_id: String,
    detector: DetectorFixture,
    base_artifact_digest: String,
    source_weights: Vec<TensorFixture>,
    input: TensorFixture,
    dtype: DType,
    device: DeviceKind,
    activation_elements: u64,
    memory_budget_bytes: u64,
    expected_memory_bytes: u64,
    checkpoints: Vec<CheckpointFixture>,
    patches: Vec<PatchOperation>,
    patched_checkpoints: Vec<CheckpointFixture>,
}

#[derive(Debug, Deserialize)]
struct DetectorFixture {
    tensor_shapes: BTreeMap<String, Vec<u64>>,
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ProvenanceFixture {
    schema_version: u16,
    fixture_id: String,
    feature_id: String,
    source_symbol: String,
    source_ordinal: u16,
    source_architecture: String,
    latent_feature_id: String,
    latent_identifier: String,
    source_constants: SourceConstants,
    source_files: Vec<SourceFileFixture>,
    source_projection: String,
    source_projection_sha256: String,
    oracle: String,
}

#[derive(Debug, Deserialize)]
struct SourceConstants {
    audio_model: String,
    memory_usage_factor: f64,
    sampling_multiplier: f64,
    sampling_shift: f64,
    supported_dtypes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SourceFileFixture {
    path: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct TensorFixture {
    key: String,
    shape: Vec<u64>,
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CheckpointFixture {
    name: String,
    values: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum FixtureLayout {
    PrefixedNative,
    Unprefixed,
}

#[test]
fn val_model_family_row_001_acestep15_sim_model_0062()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    let provenance = provenance()?;
    verify_provenance(&provenance)?;
    verify_identity_descriptor_and_profile(&fixture, &provenance)?;
    let unprefixed_probe = probe_through_model_store(&fixture, FixtureLayout::Unprefixed)?;
    verify_detection_and_mapping(&fixture, &unprefixed_probe, FixtureLayout::Unprefixed)?;
    let native_probe = probe_through_model_store(&fixture, FixtureLayout::PrefixedNative)?;
    verify_detection_and_mapping(&fixture, &native_probe, FixtureLayout::PrefixedNative)?;
    verify_forward_patch_memory_dtype_device_and_failures(&fixture, &unprefixed_probe)?;
    verify_invalid_partial_ambiguous_and_cancellation(&fixture, &unprefixed_probe)?;
    verify_owner_delegation()?;
    super::write_model_family_row_artifact(
        MODEL_FAMILY_FIXTURE,
        MODEL_FAMILY_FEATURE_ID,
        MODEL_FAMILY_IDENTIFIER,
        MODEL_FAMILY_REGISTRATION.source_ordinal,
        "acestep15_comfy_model_0062",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-and-diffusers-detection",
            "qwen3-profile-selection-and-ambiguity",
            "transactional-component-mapping",
            "named-forward-checkpoints-and-patching",
            "memory-oom-dtype-device-cancellation",
            "partial-malformed-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_acestep15_clip_selector_distinguishes_qwen3_variants_and_rejects_ambiguity()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture()?;
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let probe_2b = fixture_probe(&fixture, FixtureLayout::Unprefixed)?;
    let resolved_2b = registry.resolve(&probe_2b)?;
    assert!(!resolved_2b.clip_target().dynamic_selection());
    assert_clip_expansion(resolved_2b.clip_target(), "detect_qwen3_2b")?;

    let mut probe_4b = probe_2b;
    let shape = probe_4b
        .tensor_shapes
        .remove("text_encoders.qwen3_2b.transformer.model.norm.weight")
        .ok_or("missing qwen3_2b fixture marker")?;
    probe_4b.tensor_shapes.insert(
        "text_encoders.qwen3_4b.transformer.model.norm.weight".to_owned(),
        shape,
    );
    let resolved_4b = registry.resolve(&probe_4b)?;
    assert_clip_expansion(resolved_4b.clip_target(), "detect_qwen3_4b")?;

    probe_4b.tensor_shapes.insert(
        "text_encoders.qwen3_2b.transformer.model.norm.weight".to_owned(),
        vec![4],
    );
    assert!(matches!(
        registry.resolve(&probe_4b),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("both qwen3_2b and qwen3_4b")
    ));
    Ok(())
}

fn verify_identity_descriptor_and_profile(
    fixture: &FamilyFixture,
    provenance: &ProvenanceFixture,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(MODEL_FAMILY_FEATURE_ID, fixture.feature_id);
    assert_eq!(MODEL_FAMILY_IDENTIFIER, provenance.source_symbol);
    assert_eq!(MODEL_FAMILY_FIXTURE, fixture.fixture_id);
    assert_eq!(MODEL_FAMILY_REGISTRATION.source_ordinal, provenance.source_ordinal);
    assert_eq!(SOURCE_ARCHITECTURE, provenance.source_architecture);
    assert_eq!(MODEL_FAMILY.latent_feature_id, provenance.latent_feature_id);
    assert_eq!(MODEL_FAMILY.latent_identifier, provenance.latent_identifier);
    assert_eq!(provenance.source_constants.audio_model, "ace1.5");
    assert_eq!(SOURCE_MEMORY_USAGE_FACTOR, provenance.source_constants.memory_usage_factor);
    assert_eq!(SOURCE_SAMPLING_MULTIPLIER, provenance.source_constants.sampling_multiplier);
    assert_eq!(SOURCE_SAMPLING_SHIFT, provenance.source_constants.sampling_shift);
    assert_eq!(
        provenance.source_constants.supported_dtypes,
        ["bfloat16", "float32"]
    );
    assert_eq!(MODEL_FAMILY.supported_dtypes, [DType::Bf16, DType::F32]);
    assert_eq!(MODEL_FAMILY.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(MODEL_FAMILY.memory_estimator.bytes_per_parameter, 5);
    assert_eq!(MODEL_FAMILY.memory_estimator.activation_bytes_per_element, 5);

    let descriptor = describe_model_family(&MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, provenance.source_symbol);
    assert_eq!(descriptor.family, fixture.feature_id);
    assert_eq!(descriptor.latent_format, provenance.latent_identifier);
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.component_graph[0].identifier, "denoiser");
    assert!(descriptor.component_graph[0].required);
    assert_eq!(descriptor.component_graph[1].identifier, "text_encoder");
    assert!(!descriptor.component_graph[1].required);
    assert_eq!(descriptor.component_graph[2].identifier, "vae");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    Ok(())
}

fn verify_detection_and_mapping(
    fixture: &FamilyFixture,
    probe: &ModelProbe,
    layout: FixtureLayout,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert_eq!(
        probe.storage_dtype(&fixture_key(
            "decoder.layers.0.self_attn_norm.weight",
            layout,
        )),
        Some(sim_model::ModelStorageDType::Tensor(DType::F32))
    );
    let layer_pattern = match layout {
        FixtureLayout::PrefixedNative => "model.diffusion_model.decoder.layers.{}.",
        FixtureLayout::Unprefixed => "decoder.layers.{}.",
    };
    assert_eq!(probe.consecutive_block_count(layer_pattern)?, 2);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(probe)?;
    assert_eq!(resolved.detection().identity.identifier(), MODEL_FAMILY_IDENTIFIER);
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), MODEL_FAMILY_REGISTRATION.source_ordinal);
    assert_eq!(resolved.source_architecture(), SOURCE_ARCHITECTURE);
    assert_clip_expansion(resolved.clip_target(), "detect_qwen3_2b")?;

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(fixture, layout, DType::F32, &backend, &context)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    assert_eq!(mapped.components().len(), 3);
    assert!(
        mapped
            .component("denoiser")
            .is_some_and(|component| component.contains_key("decoder.condition_embedder.weight"))
    );
    assert!(
        mapped
            .component("text_encoder")
            .is_some_and(|component| component.contains_key("model.qwen3_2b.transformer.model.norm.weight"))
    );
    assert!(
        mapped
            .component("vae")
            .is_some_and(|component| component.contains_key("model.decoder.weight"))
    );
    assert!(mapped.binding().is_some());
    Ok(())
}

fn verify_forward_patch_memory_dtype_device_and_failures(
    fixture: &FamilyFixture,
    probe: &ModelProbe,
) -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(probe)?;
    let source = source_tensors(
        fixture,
        FixtureLayout::Unprefixed,
        DType::F32,
        &backend,
        &context,
    )?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    let options = NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes: 1024 * 1024,
        allow_unexpected_weights: true,
    };
    let model = build_model_family_for_probe(&registry, probe, weights.clone(), options)?;
    let input = tensor(
        &backend,
        &fixture.input.shape,
        &fixture.input.values,
        DType::F32,
        &context,
    )?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_eq!(checkpoints.len(), fixture.checkpoints.len());
    for (actual, expected) in checkpoints.iter().zip(&fixture.checkpoints) {
        assert_eq!(actual.name, expected.name);
        assert_close(
            &tensor_to_f32_with_context_exact_native(&backend, &actual.tensor, &context)?,
            &expected.values,
            1.0e-5,
        )?;
    }

    let memory = model.memory_estimate();
    assert_eq!(memory.weight_bytes, memory.parameter_elements * 5);
    assert_eq!(memory.activation_bytes, 10);
    assert_eq!(
        MODEL_FAMILY.memory_estimator.bytes_per_parameter,
        SOURCE_MEMORY_USAGE_FACTOR.ceil() as u32
    );
    let mut oom = options;
    oom.memory_budget_bytes = memory
        .total_bytes
        .checked_sub(1)
        .ok_or("ACE-Step 1.5 memory estimate unexpectedly has no bytes")?;
    assert!(matches!(
        build_model_family_for_probe(&registry, probe, weights.clone(), oom),
        Err(ModelFamilyError::OutOfMemory { required, budget })
            if required == memory.total_bytes && budget + 1 == required
    ));
    let mut unexpected_forbidden = options;
    unexpected_forbidden.allow_unexpected_weights = false;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            probe,
            weights.clone(),
            unexpected_forbidden,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if !keys.is_empty()
    ));

    let bf16_source = source_tensors(
        fixture,
        FixtureLayout::Unprefixed,
        DType::Bf16,
        &backend,
        &context,
    )?;
    let bf16_weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &bf16_source,
    )?;
    let mut bf16_options = options;
    bf16_options.dtype = DType::Bf16;
    assert!(build_model_family_for_probe(&registry, probe, bf16_weights.clone(), bf16_options)
        .is_ok());
    let mut unsupported_dtype = bf16_options;
    unsupported_dtype.dtype = DType::F16;
    assert!(matches!(
        build_model_family_for_probe(&registry, probe, bf16_weights.clone(), unsupported_dtype),
        Err(ModelFamilyError::UnsupportedDType(DType::F16))
    ));
    let mut unsupported_device = bf16_options;
    unsupported_device.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, probe, bf16_weights, unsupported_device),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));

    let patch = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?;
    let patched_weights = patch.apply(&backend, &weights, &context)?;
    assert_ne!(patched_weights.cache_identity(), weights.cache_identity());
    let patched = model.with_weights(patched_weights)?;
    let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
    for (actual, expected) in patched_checkpoints.iter().zip(&fixture.patched_checkpoints) {
        assert_eq!(actual.name, expected.name);
        assert_close(
            &tensor_to_f32_with_context_exact_native(&backend, &actual.tensor, &context)?,
            &expected.values,
            1.0e-5,
        )?;
    }

    let add = PatchOperation {
        identifier: "add-condition-projection".to_owned(),
        kind: PatchKind::Adapter,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "decoder.condition_embedder.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Add,
        }],
    };
    let replace = fixture
        .patches
        .first()
        .cloned()
        .ok_or("ACE-Step 1.5 fixture is missing its replacement patch")?;
    let ordered = PatchGraph::checked(
        &fixture.base_artifact_digest,
        vec![replace.clone(), add.clone()],
    )?
    .apply(&backend, &weights, &context)?;
    let reversed = PatchGraph::checked(&fixture.base_artifact_digest, vec![add, replace])?
        .apply(&backend, &weights, &context)?;
    assert_ne!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            ordered
                .tensors()
                .get("decoder.condition_embedder.weight")
                .ok_or("ordered patch output is missing the condition projection")?,
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            reversed
                .tensors()
                .get("decoder.condition_embedder.weight")
                .ok_or("reversed patch output is missing the condition projection")?,
            &context,
        )?,
    );
    Ok(())
}

fn verify_invalid_partial_ambiguous_and_cancellation(
    fixture: &FamilyFixture,
    probe: &ModelProbe,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let mut partial = probe.clone();
    partial
        .tensor_shapes
        .remove("encoder.lyric_encoder.layers.0.input_layernorm.weight");
    partial.metadata.remove(
        "__sim.model_probe.v1.dtype.encoder.lyric_encoder.layers.0.input_layernorm.weight",
    );
    let partial_error = registry
        .detect(&partial)
        .expect_err("partial ACE-Step 1.5 probe unexpectedly matched");
    assert!(
        matches!(
            &partial_error,
            ModelFamilyError::NoDetectionMatch
                | ModelFamilyError::MissingRequiredStateKey(_)
        ),
        "unexpected partial-probe error: {partial_error:?}"
    );

    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "decoder.layers.0.self_attn.q_proj.weight".to_owned(),
        vec![255, 4],
    );
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("decoder attention dimensions")
    ));

    let cross_family = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("conv_in.weight".to_owned(), vec![2, 2, 1, 1]),
            ("conv_out.weight".to_owned(), vec![2, 2, 1, 1]),
        ]),
        metadata: BTreeMap::from([
            ("audio_model".to_owned(), "ace1.5".to_owned()),
            ("model_layout".to_owned(), "diffusers".to_owned()),
        ]),
    };
    assert!(matches!(
        registry.detect(&cross_family),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut misleading = probe.clone();
    misleading
        .metadata
        .insert("audio_model".to_owned(), "not-ace1.5".to_owned());
    misleading
        .metadata
        .insert("model_layout".to_owned(), "prefixed-native".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        MODEL_FAMILY_FEATURE_ID
    );

    let native = fixture_probe(fixture, FixtureLayout::PrefixedNative)?;
    let mut mixed = fixture_probe(fixture, FixtureLayout::Unprefixed)?;
    mixed.tensor_shapes.extend(native.tensor_shapes);
    let mixed_error = registry
        .resolve(&mixed)
        .expect_err("mixed ACE-Step 1.5 layouts unexpectedly resolved");
    assert!(
        matches!(mixed_error, ModelFamilyError::ModelLayoutSelection(_)),
        "unexpected mixed-layout error: {mixed_error:?}"
    );
    assert!(matches!(
        ModelFamilyRegistry::checked(&TIED_DEFINITIONS)?.detect(probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(
        fixture,
        FixtureLayout::Unprefixed,
        DType::F32,
        &backend,
        &context,
    )?;
    cancellation.cancel();
    assert!(registry.resolve(probe)?.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    ).is_err());
    Ok(())
}

fn verify_owner_delegation() -> Result<(), Box<dyn std::error::Error>> {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/families/acestep15_sim_model_0062.rs"
    ));
    for canonical in [
        "ModelFamilyRegistration",
        "ModelFamilyStatePlanSelector",
        "ModelStateTransformPlanDefinition",
        "ModelProbe",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(source.contains(canonical));
    }
    for forbidden_owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "std::fs",
        "File::",
        "unsafe ",
    ] {
        assert!(!source.contains(forbidden_owner));
    }
    Ok(())
}

fn fixture() -> Result<FamilyFixture, Box<dyn std::error::Error>> {
    let bytes = fs::read(fixture_path())?;
    let fixture: FamilyFixture = serde_json::from_slice(&bytes)?;
    assert_eq!(fixture.fixture_id, MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture.dtype, DType::F32);
    assert_eq!(fixture.device, DeviceKind::Cpu);
    assert_eq!(fixture.activation_elements, 2);
    assert_eq!(fixture.memory_budget_bytes, fixture.expected_memory_bytes);
    assert!(!fixture.patches.is_empty());
    assert_eq!(fixture.patched_checkpoints.len(), fixture.checkpoints.len());
    Ok(fixture)
}

fn provenance() -> Result<ProvenanceFixture, Box<dyn std::error::Error>> {
    let bytes = fs::read(provenance_path())?;
    let provenance: ProvenanceFixture = serde_json::from_slice(&bytes)?;
    assert_eq!(provenance.schema_version, 1);
    assert_eq!(provenance.fixture_id, MODEL_FAMILY_FIXTURE);
    assert_eq!(provenance.feature_id, MODEL_FAMILY_FEATURE_ID);
    Ok(provenance)
}

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../sim_test_support/fixtures/models/acestep15-comfy-model-0062/family.json",
    )
}

fn provenance_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../sim_test_support/fixtures/models/acestep15-comfy-model-0062/provenance.json",
    )
}

fn fixture_probe(
    fixture: &FamilyFixture,
    layout: FixtureLayout,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    Ok(ModelProbe {
        tensor_shapes: probe_tensor_fixtures(fixture, layout)?
            .into_iter()
            .map(|tensor| (tensor.key, tensor.shape))
            .collect(),
        metadata: fixture.detector.metadata.clone(),
    })
}

fn probe_through_model_store(
    fixture: &FamilyFixture,
    layout: FixtureLayout,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let model_path = directory.path().join("acestep15.safetensors");
    let tensors = probe_tensor_fixtures(fixture, layout)?;
    write_safetensors(&model_path, &tensors)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "acestep15-row",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("acestep15-row", "acestep15.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let probe = store.family_probe(&loaded, &cancellation)?;
    assert_eq!(probe.tensor_shapes(), &fixture_probe(fixture, layout)?.tensor_shapes);
    Ok(probe)
}

fn source_tensors(
    fixture: &FamilyFixture,
    layout: FixtureLayout,
    dtype: DType,
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe_tensor_fixtures(fixture, layout)?
        .iter()
        .map(|tensor_fixture| {
            Ok((
                tensor_fixture.key.clone(),
                tensor(
                    backend,
                    &tensor_fixture.shape,
                    &tensor_fixture.values,
                    dtype,
                    context,
                )?,
            ))
        })
        .collect()
}

fn probe_tensor_fixtures(
    fixture: &FamilyFixture,
    layout: FixtureLayout,
) -> Result<Vec<TensorFixture>, Box<dyn std::error::Error>> {
    fixture
        .detector
        .tensor_shapes
        .iter()
        .map(|(key, shape)| {
            let count = shape.iter().try_fold(1_u64, |total, dimension| {
                total.checked_mul(*dimension).ok_or("fixture shape overflow")
            })?;
            let count = usize::try_from(count)?;
            let source = fixture.source_weights.iter().find(|weight| weight.key == *key);
            let values = source.map_or_else(
                || vec![0.0; count],
                |weight| weight.values.clone(),
            );
            if values.len() != count {
                return Err(format!("{key} value count mismatch").into());
            }
            Ok(TensorFixture {
                key: fixture_key(key, layout),
                shape: shape.clone(),
                values,
            })
        })
        .collect()
}

fn fixture_key(key: &str, layout: FixtureLayout) -> String {
    if matches!(layout, FixtureLayout::PrefixedNative)
        && !key.starts_with("text_encoders.")
        && !key.starts_with("vae.")
    {
        format!("model.diffusion_model.{key}")
    } else {
        key.to_owned()
    }
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
    if dtype == DType::F32 {
        Ok(tensor)
    } else {
        Ok(cast_to_with_context_exact_native(
            backend,
            &tensor,
            dtype,
            backend.device(),
            false,
            false,
            context,
        )?)
    }
}

fn write_safetensors(
    path: &Path,
    tensors: &[TensorFixture],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for tensor in tensors {
        let start = data.len();
        for value in &tensor.values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            tensor.key.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn assert_clip_expansion(
    target: &sim_model::ModelClipTargetDescriptor,
    expected_source: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(target.candidates().len(), 1);
    let invocation = target.candidates()[0].clip_model().invocation();
    let ModelClipModelInvocation::Factory { configuration } = invocation else {
        return Err("ACE-Step 1.5 CLIP target is not a factory".into());
    };
    assert!(matches!(
        configuration.as_slice(),
        [sim_model::ModelClipConfigurationFact::Expand { source }]
            if source.as_str() == expected_source
    ));
    Ok(())
}

fn verify_provenance(provenance: &ProvenanceFixture) -> Result<(), Box<dyn std::error::Error>> {
    assert!(provenance.oracle.contains("no production Python dependency"));
    assert_eq!(
        sha256(provenance.source_projection.as_bytes()),
        provenance.source_projection_sha256
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for source in &provenance.source_files {
        assert_eq!(sha256(&fs::read(root.join(&source.path))?), source.sha256);
    }
    Ok(())
}

fn assert_close(
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err("checkpoint length mismatch".into());
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if (actual - expected).abs() > tolerance {
            return Err(format!(
                "checkpoint value {index} differs: actual={actual}, expected={expected}, tolerance={tolerance}"
            )
            .into());
        }
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
