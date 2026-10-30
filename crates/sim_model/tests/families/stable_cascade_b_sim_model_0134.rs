use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_stable_cascade_b_sim_model_0134 as cascade_b,
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

static B_REGISTRATIONS: [ModelFamilyRegistration; 1] =
    [cascade_b::MODEL_FAMILY_REGISTRATION];
static B_AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9134",
    identifier: "StableCascadeBAmbiguousFixture",
    ..cascade_b::MODEL_FAMILY
};
static B_AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    cascade_b::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &B_AMBIGUOUS_DEFINITION,
        source_ordinal: 116,
        source_architecture: "model_base.StableCascadeBAmbiguousFixture",
        ..cascade_b::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) struct CascadeFamilyCase {
    pub definition: &'static ModelFamilyDefinition,
    pub registration: ModelFamilyRegistration,
    pub registrations: &'static [ModelFamilyRegistration],
    pub identifier: &'static str,
    pub feature_id: &'static str,
    pub fixture: &'static str,
    pub module: &'static str,
    pub source_ordinal: u16,
    pub source_architecture: &'static str,
    pub projection_sha256: &'static str,
    pub latent_feature_id: &'static str,
    pub latent_identifier: &'static str,
    pub architecture_version: &'static str,
    pub variant_marker: &'static str,
    pub variant_dimension: usize,
    pub supported_dtypes: &'static [DType],
    pub component_count: usize,
    pub has_vision: bool,
    pub malformed_width: u64,
    pub patch_key: &'static str,
    pub focused_memory_bytes: u64,
    pub validate_configuration: fn(&ModelProbe) -> Result<(), ModelFamilyError>,
}

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

#[test]
fn val_model_family_row_001_stable_cascade_b_source_configuration_and_state_transform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture(&case())?;
    let configuration = cascade_b::configuration_for_probe(&fixture_probe(&fixture))?;
    assert_eq!(configuration.variant, cascade_b::StableCascadeBVariant::Full);
    assert_eq!(configuration.hidden_dimensions, [320, 640, 1_280, 1_280]);
    assert_eq!(configuration.attention_heads, [-1, -1, 20, 20]);
    assert_eq!(configuration.down_blocks, [2, 6, 28, 6]);
    assert_eq!(configuration.up_blocks, [6, 28, 6, 2]);
    assert_eq!(configuration.down_repeats, [1, 1, 1, 1]);
    assert_eq!(configuration.up_repeats, [3, 3, 2, 2]);

    let mut lite = fixture_probe(&fixture);
    lite.tensor_shapes.insert(
        cascade_b_variant_marker().to_owned(),
        vec![2, 576],
    );
    let lite = cascade_b::configuration_for_probe(&lite)?;
    assert_eq!(lite.variant, cascade_b::StableCascadeBVariant::Lite);
    assert_eq!(lite.hidden_dimensions, [320, 576, 1_152, 1_152]);
    assert_eq!(lite.attention_heads, [-1, 9, 18, 18]);
    assert_eq!(lite.down_blocks, [2, 4, 14, 4]);
    assert_eq!(lite.up_repeats, [2, 2, 2, 2]);
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&B_AMBIGUOUS_REGISTRATIONS)?
            .detect(&fixture_probe(&fixture)),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    run_source_validation(&case())
}

#[test]
fn val_model_family_row_001_stable_cascade_b_forward_patch_memory_and_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    run_execution_validation(&case())
}

fn case() -> CascadeFamilyCase {
    CascadeFamilyCase {
        definition: &cascade_b::MODEL_FAMILY,
        registration: cascade_b::MODEL_FAMILY_REGISTRATION,
        registrations: &B_REGISTRATIONS,
        identifier: cascade_b::MODEL_FAMILY_IDENTIFIER,
        feature_id: cascade_b::MODEL_FAMILY_FEATURE_ID,
        fixture: cascade_b::MODEL_FAMILY_FIXTURE,
        module: "stable_cascade_b_comfy_model_0134",
        source_ordinal: cascade_b::MODEL_FAMILY_SOURCE_ORDINAL,
        source_architecture: cascade_b::SOURCE_ARCHITECTURE,
        projection_sha256: cascade_b::MODEL_FAMILY_PROJECTION_SHA256,
        latent_feature_id: "COMFY-MODEL-0043",
        latent_identifier: "SC_B",
        architecture_version: "stable-cascade-stage-b-v1",
        variant_marker: cascade_b_variant_marker(),
        variant_dimension: 1,
        supported_dtypes: &[DType::F16, DType::Bf16, DType::F32],
        component_count: 3,
        has_vision: false,
        malformed_width: 641,
        patch_key: "native.clip_mapper.weight",
        focused_memory_bytes: 5_248,
        validate_configuration: |probe| cascade_b::configuration_for_probe(probe).map(|_| ()),
    }
}

const fn cascade_b_variant_marker() -> &'static str {
    "model.diffusion_model.down_blocks.1.0.channelwise.0.weight"
}

pub(super) fn run_source_validation(
    case: &CascadeFamilyCase,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture(case)?;
    assert_eq!(fixture.fixture_id, case.fixture);
    assert_eq!(fixture.feature_id, case.feature_id);
    assert_eq!(fixture.dtype, DType::F32);
    assert_eq!(fixture.device, DeviceKind::Cpu);
    assert_eq!(fixture.expected_memory_bytes, 56);
    assert_eq!(case.definition.identifier, case.identifier);
    assert_eq!(case.definition.feature_id, case.feature_id);
    assert_eq!(case.registration.source_ordinal, case.source_ordinal);
    assert_eq!(case.registration.source_architecture, case.source_architecture);
    assert_eq!(case.definition.latent_feature_id, case.latent_feature_id);
    assert_eq!(case.definition.latent_identifier, case.latent_identifier);
    assert_eq!(case.definition.architecture_version, case.architecture_version);

    let descriptor = describe_model_family(case.definition)?;
    assert_eq!(descriptor.component_graph.len(), case.component_count);
    assert_eq!(descriptor.latent_format, case.latent_identifier);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(
        descriptor
            .component_graph
            .iter()
            .any(|component| component.identifier == "vision_encoder"),
        case.has_vision
    );
    verify_provenance(case)?;
    let probe = probe_through_model_store(case, &fixture)?;
    (case.validate_configuration)(&probe)?;
    let registry = ModelFamilyRegistry::checked_registrations(case.registrations)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), case.source_ordinal);
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(
        resolved.clip_target().candidates()[0]
            .tokenizer()
            .identifier(),
        "sdxl_clip.StableCascadeTokenizer"
    );
    assert_eq!(
        resolved.clip_target().candidates()[0]
            .clip_model()
            .target()
            .as_str(),
        "sdxl_clip.StableCascadeClipModel"
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16 * 1024 * 1024)?,
        &cancellation,
    );
    let mut source = source_tensors(&fixture, DType::F32, &backend, &context)?;
    source.insert(
        "model.diffusion_model.down_blocks.0.0.attention.attn.in_proj_weight".to_owned(),
        tensor(
            &backend,
            &[6, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            DType::F32,
            &context,
        )?,
    );
    source.insert(
        "model.diffusion_model.down_blocks.0.0.attention.attn.in_proj_bias".to_owned(),
        tensor(
            &backend,
            &[6],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            DType::F32,
            &context,
        )?,
    );
    source.insert(
        "text_encoder.clip_g.text_projection".to_owned(),
        tensor(
            &backend,
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            DType::F32,
            &context,
        )?,
    );
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    assert_eq!(mapped.components().len(), case.component_count);
    let denoiser = mapped.component("denoiser").ok_or("missing denoiser")?;
    for (suffix, expected) in [
        ("to_q.weight", vec![1.0, 2.0, 3.0, 4.0]),
        ("to_k.weight", vec![5.0, 6.0, 7.0, 8.0]),
        ("to_v.weight", vec![9.0, 10.0, 11.0, 12.0]),
    ] {
        let key = format!("native.down_blocks.0.0.attention.attn.{suffix}");
        assert_eq!(
            tensor_to_f32_with_context_exact_native(
                &backend,
                denoiser.get(&key).ok_or("missing split QKV weight")?,
                &context,
            )?,
            expected
        );
    }
    for (suffix, expected) in [
        ("to_q.bias", vec![1.0, 2.0]),
        ("to_k.bias", vec![3.0, 4.0]),
        ("to_v.bias", vec![5.0, 6.0]),
    ] {
        let key = format!("native.down_blocks.0.0.attention.attn.{suffix}");
        assert_eq!(
            tensor_to_f32_with_context_exact_native(
                &backend,
                denoiser.get(&key).ok_or("missing split QKV bias")?,
                &context,
            )?,
            expected
        );
    }
    let text = mapped.component("text_encoder").ok_or("missing text encoder")?;
    let projection = text
        .get("clip_g.transformer.text_projection.weight")
        .ok_or("missing transposed CLIP-G projection")?;
    assert_eq!(projection.descriptor().shape(), [3, 2]);
    assert_eq!(
        tensor_to_f32_with_context_exact_native(&backend, projection, &context)?,
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    assert!(text.contains_key("clip_g.transformer.encoder.weight"));
    assert!(mapped.component("vae").is_some_and(|state| state.contains_key("decoder.weight")));
    if case.has_vision {
        assert!(mapped.component("vision_encoder").is_some_and(|state| {
            state.contains_key("visual.encoder.weight")
        }));
    }

    verify_failures(case, &fixture, &registry)?;
    verify_owner_delegation(case)?;
    super::write_model_family_row_artifact(
        case.fixture,
        case.feature_id,
        case.identifier,
        case.source_ordinal,
        case.module,
        &[
            "source-and-catalog-provenance",
            "typed-full-lite-shape-derived-configuration",
            "stable-cascade-clip-target-and-latent",
            "transactional-qkv-split-and-clip-projection-transpose",
            "text-vision-vae-component-routing",
            "native-forward-patch-memory-dtype-device-oom",
            "diffusers-partial-malformed-cancellation-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}

pub(super) fn run_execution_validation(
    case: &CascadeFamilyCase,
) -> Result<(), Box<dyn std::error::Error>> {
    let fixture = fixture(case)?;
    let probe = fixture_probe(&fixture);
    let registry = ModelFamilyRegistry::checked_registrations(case.registrations)?;
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(&fixture, DType::F32, &backend, &context)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    )?;
    let options = NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: fixture.activation_elements,
        memory_budget_bytes: 32 * 1024 * 1024,
        allow_unexpected_weights: true,
    };
    let model = build_model_family_for_probe(&registry, &probe, weights.clone(), options)?;
    assert_eq!(model.memory_estimate().total_bytes, case.focused_memory_bytes);
    let input = tensor(
        &backend,
        &fixture.input.shape,
        &fixture.input.values,
        DType::F32,
        &context,
    )?;
    assert_checkpoints(
        &backend,
        model.forward_checkpoints(&backend, &input, &context)?,
        &fixture.checkpoints,
        &context,
    )?;
    let patch = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?;
    let patched = patch.apply(&backend, &weights, &context)?;
    assert_ne!(patched.cache_identity(), weights.cache_identity());
    assert_checkpoints(
        &backend,
        model
            .with_weights(patched)?
            .forward_checkpoints(&backend, &input, &context)?,
        &fixture.patched_checkpoints,
        &context,
    )?;

    let add = fixture.patches[0].clone();
    let replace = PatchOperation {
        identifier: "replace-cascade-projection".to_owned(),
        kind: PatchKind::Adapter,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: case.patch_key.to_owned(),
            expected_shape: vec![2, 2],
            values: vec![4.0, 0.0, 0.0, 4.0],
            application: PatchApplication::Replace,
        }],
    };
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
            ordered.tensors().get(case.patch_key).ok_or("missing ordered patch")?,
            &context,
        )?,
        tensor_to_f32_with_context_exact_native(
            &backend,
            reversed.tensors().get(case.patch_key).ok_or("missing reversed patch")?,
            &context,
        )?
    );

    for &dtype in case.supported_dtypes {
        let source = source_tensors(&fixture, dtype, &backend, &context)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            &fixture.base_artifact_digest,
            &source,
        )?;
        let mut typed = options;
        typed.dtype = dtype;
        assert!(build_model_family_for_probe(&registry, &probe, weights, typed).is_ok());
    }
    if !case.supported_dtypes.contains(&DType::F16) {
        let mut f16 = options;
        f16.dtype = DType::F16;
        assert!(matches!(
            build_model_family_for_probe(&registry, &probe, weights.clone(), f16),
            Err(ModelFamilyError::UnsupportedDType(DType::F16))
        ));
    }
    let mut f64 = options;
    f64.dtype = DType::F64;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights.clone(), f64),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));
    let mut metal = options;
    metal.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights.clone(), metal),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));
    let mut oom = options;
    oom.memory_budget_bytes = model.memory_estimate().total_bytes - 1;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, oom),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));
    Ok(())
}

fn verify_provenance(case: &CascadeFamilyCase) -> Result<(), Box<dyn std::error::Error>> {
    let provenance: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture_directory(case).join("provenance.json"))?)?;
    assert_eq!(provenance["feature_id"], case.feature_id);
    assert_eq!(provenance["source_symbol"], case.identifier);
    assert_eq!(provenance["source_ordinal"], case.source_ordinal);
    assert_eq!(provenance["source_architecture"], case.source_architecture);
    assert_eq!(provenance["latent_feature_id"], case.latent_feature_id);
    assert_eq!(provenance["latent_identifier"], case.latent_identifier);
    assert_eq!(provenance["catalog_projection_sha256"], case.projection_sha256);
    let projection = provenance["source_projection"].as_str().ok_or("missing projection")?;
    assert!(projection.contains("attention_conversion=in_proj_weight_and_bias_split_to_q_k_v"));
    assert!(projection.contains("diffusers_support=none"));
    assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
    let root = repository_root();
    for source in provenance["source_files"].as_array().ok_or("missing source files")? {
        let path = source["path"].as_str().ok_or("missing source path")?;
        assert_eq!(
            sha256(&fs::read(root.join(path))?),
            source["sha256"].as_str().ok_or("missing digest")?
        );
    }
    let catalog: serde_json::Value = serde_json::from_slice(&fs::read(
        root.join("crates/sim_model/catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .and_then(|rows| rows.iter().find(|row| row["feature_id"] == case.feature_id))
        .ok_or("missing catalog row")?;
    assert_eq!(sha256(&serde_json::to_vec(row)?), case.projection_sha256);
    Ok(())
}

fn verify_failures(
    case: &CascadeFamilyCase,
    fixture: &FamilyFixture,
    registry: &ModelFamilyRegistry,
) -> Result<(), Box<dyn std::error::Error>> {
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([("conv_in.weight".to_owned(), vec![2, 2, 1])]),
        metadata: BTreeMap::from([("model_layout".to_owned(), "diffusers".to_owned())]),
    };
    assert!(matches!(
        (case.validate_configuration)(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut partial = fixture_probe(fixture);
    partial.tensor_shapes.remove(case.variant_marker);
    assert!(registry.resolve(&partial).is_err());
    let mut malformed = fixture_probe(fixture);
    let shape = malformed
        .tensor_shapes
        .get_mut(case.variant_marker)
        .ok_or("missing variant marker")?;
    *shape
        .get_mut(case.variant_dimension)
        .ok_or("missing variant dimension")? = case.malformed_width;
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("unsupported")
    ));
    let mut misleading = fixture_probe(fixture);
    misleading.metadata.insert(
        "stable_cascade_stage".to_owned(),
        "not-this-stage".to_owned(),
    );
    assert_eq!(
        registry
            .resolve(&misleading)?
            .detection()
            .identity
            .feature_id(),
        case.feature_id
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16 * 1024 * 1024)?,
        &cancellation,
    );
    let source = source_tensors(fixture, DType::F32, &backend, &context)?;
    cancellation.cancel();
    assert!(registry.resolve(&fixture_probe(fixture))?.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        &fixture.base_artifact_digest,
        &source,
    ).is_err());
    Ok(())
}

fn fixture(case: &CascadeFamilyCase) -> Result<FamilyFixture, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(fixture_directory(case).join("family.json"))?)?)
}

fn fixture_probe(fixture: &FamilyFixture) -> ModelProbe {
    ModelProbe {
        tensor_shapes: fixture.detector.tensor_shapes.clone(),
        metadata: fixture.detector.metadata.clone(),
    }
}

fn source_tensors(
    fixture: &FamilyFixture,
    dtype: DType,
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe_tensor_fixtures(fixture)?
        .iter()
        .map(|weight| {
            Ok((
                weight.key.clone(),
                tensor(backend, &weight.shape, &weight.values, dtype, context)?,
            ))
        })
        .collect()
}

fn probe_tensor_fixtures(
    fixture: &FamilyFixture,
) -> Result<Vec<TensorFixture>, Box<dyn std::error::Error>> {
    fixture
        .detector
        .tensor_shapes
        .iter()
        .map(|(key, shape)| {
            let count = usize::try_from(shape.iter().product::<u64>())?;
            let values = fixture
                .source_weights
                .iter()
                .find(|weight| weight.key == *key)
                .map_or_else(|| vec![0.0; count], |weight| weight.values.clone());
            if values.len() != count {
                return Err(format!("{key} value count mismatch").into());
            }
            Ok(TensorFixture {
                key: key.clone(),
                shape: shape.clone(),
                values,
            })
        })
        .collect()
}

fn probe_through_model_store(
    case: &CascadeFamilyCase,
    fixture: &FamilyFixture,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let filename = format!("{}.safetensors", case.fixture);
    write_safetensors(
        &directory.path().join(&filename),
        &probe_tensor_fixtures(fixture)?,
    )?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        case.fixture,
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(
        &index,
        &ArtifactKey::new(case.fixture, &filename)?,
        &cancellation,
    )?;
    let probe = store.family_probe(&loaded, &cancellation)?;
    assert_eq!(probe.tensor_shapes(), &fixture.detector.tensor_shapes);
    Ok(probe)
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
    Ok(if dtype == DType::F32 {
        tensor
    } else {
        cast_to_with_context_exact_native(
            backend,
            &tensor,
            dtype,
            backend.device(),
            false,
            false,
            context,
        )?
    })
}

fn assert_checkpoints(
    backend: &CpuBackend,
    actual: Vec<sim_model::ModelForwardCheckpoint>,
    expected: &[CheckpointFixture],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.name, expected.name);
        let values = tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
        for (index, (actual_value, expected_value)) in
            values.iter().zip(&expected.values).enumerate()
        {
            if (actual_value - expected_value).abs() > 1.0e-5 {
                return Err(format!(
                    "{}[{index}]: expected {expected_value}, got {actual_value}",
                    expected.name
                )
                .into());
            }
        }
    }
    Ok(())
}

fn verify_owner_delegation(case: &CascadeFamilyCase) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/families")
        .join(format!("{}.rs", case.module));
    let source = fs::read_to_string(path)?;
    for owner in [
        "ModelFamilyRegistration",
        "ModelFamilyStatePlanSelector",
        "ModelStateTransformPlanDefinition",
        "ModelProbe",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(source.contains(owner));
    }
    for forbidden in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "std::fs",
        "unsafe ",
        "std::process",
        "Command::",
        "python",
    ] {
        assert!(!source.contains(forbidden));
    }
    Ok(())
}

fn fixture_directory(case: &CascadeFamilyCase) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(case.fixture)
}

fn repository_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
