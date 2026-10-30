use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_stable_zero123_sim_model_0136 as zero123,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [zero123::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9136",
    identifier: "Stable_Zero123_AmbiguousFixture",
    ..zero123::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    zero123::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 901,
        source_architecture: "model_base.Stable_Zero123_AmbiguousFixture",
        ..zero123::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_stable_zero123_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(zero123::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        zero123::MODEL_FAMILY_FIXTURE,
        zero123::MODEL_FAMILY_FEATURE_ID,
        zero123::MODEL_FAMILY_IDENTIFIER,
        zero123::MODEL_FAMILY_SOURCE_ORDINAL,
        zero123::SOURCE_ARCHITECTURE,
        zero123::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe(&fixture);
    let store_probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(probe.tensor_shapes(), store_probe.tensor_shapes());
    let configuration = zero123::configuration_for_probe(&store_probe)?;
    assert_eq!(configuration.model_channels, 320);
    assert_eq!(configuration.input_channels, 8);
    assert_eq!(configuration.context_dimension, 768);
    assert!(!configuration.linear_transformer_projection);
    assert_eq!(configuration.adm_input_channels, None);
    assert!(!configuration.temporal_attention);
    assert_eq!(configuration.attention_heads, 8);
    assert_eq!(configuration.projection_input_dimension, 1_024);
    assert_eq!(configuration.projection_output_dimension, 768);

    let descriptor = describe_model_family(&zero123::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SD15");
    assert_eq!(descriptor.component_graph.len(), 4);
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    let resolved = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?.resolve(&probe)?;
    assert!(resolved.clip_target().candidates().is_empty());
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 1);
    assert_eq!(resolved.source_architecture(), zero123::SOURCE_ARCHITECTURE);

    let mut partial = probe.clone();
    partial.tensor_shapes.remove("cc_projection.bias");
    assert!(ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?.resolve(&partial).is_err());
    let mut malformed = probe.clone();
    malformed
        .tensor_shapes
        .insert("cc_projection.weight".to_owned(), vec![767, 1_024]);
    assert!(matches!(
        zero123::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        zero123::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut misleading = probe.clone();
    misleading
        .metadata
        .insert("model_family".to_owned(), "SVD_img2vid".to_owned());
    assert_eq!(
        ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?
            .resolve(&misleading)?
            .detection()
            .identity
            .feature_id(),
        zero123::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("stable_zero123_comfy_model_0136")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_stable_zero123_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(zero123::MODEL_FAMILY_FIXTURE)?;
    let extras = [
        support::TensorFixture::new("cond_stage_model.model.visual.proj.weight", &[1], &[1.0]),
        support::TensorFixture::new("first_stage_model.decoder.weight", &[1], &[1.0]),
    ];
    support::exercise_family(
        &fixture,
        &REGISTRATIONS,
        &extras,
        &["model", "cc_projection", "vision_encoder", "vae"],
        "native.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight",
    )?;
    super::write_model_family_row_artifact(
        zero123::MODEL_FAMILY_FIXTURE,
        zero123::MODEL_FAMILY_FEATURE_ID,
        zero123::MODEL_FAMILY_IDENTIFIER,
        zero123::MODEL_FAMILY_SOURCE_ORDINAL,
        "stable_zero123_comfy_model_0136",
        &[
            "source-and-catalog-provenance",
            "paired-cc-projection-native-detection",
            "source-exact-zero123-configuration",
            "transactional-denoiser-projection-vision-vae-routing",
            "named-native-forward-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "diffusers-partial-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}

pub(super) mod support {
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelFamilyRegistration,
        ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
        NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
        PatchOperation, PatchTarget, build_model_family_for_probe,
    };
    use sim_tensor::{
        CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend,
        TensorDescriptor,
        generated_sim_operator_indirection_01::{
            cast_to_with_context_exact_native, tensor_to_f32_with_context_exact_native,
        },
    };
    use sim_types::{CancellationToken, DeviceKind};
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, fs, io::Write, path::Path};

    #[derive(Debug, Deserialize)]
    pub(crate) struct FamilyFixture {
        pub(crate) fixture_id: String,
        pub(crate) feature_id: String,
        pub(crate) detector: DetectorFixture,
        pub(crate) base_artifact_digest: String,
        pub(crate) source_weights: Vec<TensorFixture>,
        pub(crate) input: TensorFixture,
        pub(crate) dtype: DType,
        pub(crate) device: DeviceKind,
        pub(crate) activation_elements: u64,
        pub(crate) memory_budget_bytes: u64,
        pub(crate) expected_memory_bytes: u64,
        pub(crate) checkpoints: Vec<CheckpointFixture>,
        pub(crate) patches: Vec<PatchOperation>,
        pub(crate) patched_checkpoints: Vec<CheckpointFixture>,
    }

    #[derive(Debug, Deserialize)]
    pub(crate) struct DetectorFixture {
        pub(crate) tensor_shapes: BTreeMap<String, Vec<u64>>,
        pub(crate) metadata: BTreeMap<String, String>,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub(crate) struct TensorFixture {
        pub(crate) key: String,
        pub(crate) shape: Vec<u64>,
        pub(crate) values: Vec<f32>,
    }

    impl TensorFixture {
        pub(crate) fn new(key: &str, shape: &[u64], values: &[f32]) -> Self {
            Self {
                key: key.to_owned(),
                shape: shape.to_vec(),
                values: values.to_vec(),
            }
        }
    }

    #[derive(Debug, Deserialize)]
    pub(crate) struct CheckpointFixture {
        name: String,
        values: Vec<f32>,
    }

    pub(crate) fn load_fixture(
        fixture: &str,
    ) -> Result<FamilyFixture, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(fixture_directory(fixture).join("family.json"))?)?)
    }

    pub(crate) fn probe(fixture: &FamilyFixture) -> ModelProbe {
        ModelProbe {
            tensor_shapes: fixture.detector.tensor_shapes.clone(),
            metadata: fixture.detector.metadata.clone(),
        }
    }

    pub(crate) fn probe_through_model_store(
        fixture: &FamilyFixture,
    ) -> Result<ModelProbe, Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("family.safetensors");
        write_safetensors(&path, fixture)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "model-family-row",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("model-family-row", "family.safetensors")?,
            &cancellation,
        )?;
        let probe = store.family_probe(&loaded, &cancellation)?;
        assert_eq!(probe.tensor_shapes(), &fixture.detector.tensor_shapes);
        for (key, value) in &fixture.detector.metadata {
            assert_eq!(probe.metadata().get(key), Some(value));
        }
        Ok(probe)
    }

    pub(crate) fn exercise_family(
        fixture: &FamilyFixture,
        registrations: &'static [ModelFamilyRegistration],
        extras: &[TensorFixture],
        expected_components: &[&str],
        patch_target: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let registration = registrations.first().ok_or("missing registration")?;
        assert!(!fixture.fixture_id.is_empty());
        assert_eq!(fixture.feature_id, registration.definition.feature_id);
        assert_eq!(fixture.dtype, DType::F32);
        assert_eq!(fixture.device, DeviceKind::Cpu);
        assert_eq!(fixture.memory_budget_bytes, fixture.expected_memory_bytes);
        let probe = probe(fixture);
        let registry = ModelFamilyRegistry::checked_registrations(registrations)?;
        let resolved = registry.resolve(&probe)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(16 * 1024 * 1024)?,
            &cancellation,
        );
        let source = source_tensors(fixture, extras, DType::F32, &backend, &context)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            &fixture.base_artifact_digest,
            &source,
        )?;
        for component in expected_components {
            assert!(mapped.component(component).is_some(), "missing {component}");
        }
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            &fixture.base_artifact_digest,
            &source,
        )?;
        let options = NativeFamilyBuildOptions {
            dtype: DType::F32,
            device: DeviceKind::Cpu,
            activation_elements: fixture.activation_elements,
            memory_budget_bytes: u64::MAX,
            allow_unexpected_weights: false,
        };
        let model = build_model_family_for_probe(&registry, &probe, weights.clone(), options)?;
        let required_memory_bytes = model.memory_estimate().total_bytes;
        assert!(required_memory_bytes >= fixture.expected_memory_bytes);
        let input = tensor(&backend, &fixture.input, DType::F32, &context)?;
        assert_checkpoints(
            &backend,
            model.forward_checkpoints(&backend, &input, &context)?,
            &fixture.checkpoints,
            &context,
        )?;
        let patch = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?;
        let patched_weights = patch.apply(&backend, &weights, &context)?;
        assert_checkpoints(
            &backend,
            model
                .with_weights(patched_weights)?
                .forward_checkpoints(&backend, &input, &context)?,
            &fixture.patched_checkpoints,
            &context,
        )?;
        let add = fixture.patches[0].clone();
        let replace = PatchOperation {
            identifier: "ordered-replacement".to_owned(),
            kind: PatchKind::Adapter,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: patch_target.to_owned(),
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
                ordered.tensors().get(patch_target).ok_or("ordered patch target")?,
                &context,
            )?,
            tensor_to_f32_with_context_exact_native(
                &backend,
                reversed.tensors().get(patch_target).ok_or("reversed patch target")?,
                &context,
            )?
        );

        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let source = source_tensors(fixture, extras, dtype, &backend, &context)?;
            let weights = resolved.map_primary_weights(
                &ModelStateTransaction::new(&backend, &context),
                &fixture.base_artifact_digest,
                &source,
            )?;
            let mut typed = options;
            typed.dtype = dtype;
            assert!(build_model_family_for_probe(&registry, &probe, weights, typed).is_ok());
        }
        let mut unsupported_dtype = options;
        unsupported_dtype.dtype = DType::F64;
        assert!(matches!(
            build_model_family_for_probe(&registry, &probe, weights.clone(), unsupported_dtype),
            Err(ModelFamilyError::UnsupportedDType(DType::F64))
        ));
        let mut unsupported_device = options;
        unsupported_device.device = DeviceKind::Metal;
        assert!(matches!(
            build_model_family_for_probe(&registry, &probe, weights.clone(), unsupported_device),
            Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
        ));
        let mut oom = options;
        oom.memory_budget_bytes = required_memory_bytes - 1;
        assert!(matches!(
            build_model_family_for_probe(&registry, &probe, weights, oom),
            Err(ModelFamilyError::OutOfMemory { .. })
        ));
        cancellation.cancel();
        assert!(matches!(
            resolved.map_state_dictionary(
                &ModelStateTransaction::new(&backend, &context),
                &fixture.base_artifact_digest,
                &source,
            ),
            Err(ModelFamilyError::Cancelled(_))
        ));
        Ok(())
    }

    pub(crate) fn verify_provenance(
        fixture: &str,
        feature_id: &str,
        identifier: &str,
        source_ordinal: u16,
        architecture: &str,
        projection_sha256: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let provenance: serde_json::Value = serde_json::from_slice(&fs::read(
            fixture_directory(fixture).join("provenance.json"),
        )?)?;
        assert_eq!(provenance["fixture_id"], fixture);
        assert_eq!(provenance["feature_id"], feature_id);
        assert_eq!(provenance["source_symbol"], identifier);
        assert_eq!(provenance["source_ordinal"], source_ordinal);
        assert_eq!(provenance["source_architecture"], architecture);
        assert_eq!(provenance["catalog_projection_sha256"], projection_sha256);
        let projection = provenance["source_projection"]
            .as_str()
            .ok_or("source projection")?;
        assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
        for source in provenance["source_files"].as_array().ok_or("source files")? {
            let path = source["path"].as_str().ok_or("source path")?;
            let digest = source["sha256"].as_str().ok_or("source digest")?;
            assert_eq!(sha256(&fs::read(repository_root().join(path))?), digest);
        }
        let catalog: serde_json::Value = serde_json::from_slice(&fs::read(
            repository_root().join("crates/sim_model/catalog/model-families-v1.json"),
        )?)?;
        let row = catalog["models"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["feature_id"] == feature_id))
            .ok_or("catalog row")?;
        assert_eq!(sha256(&serde_json::to_vec(row)?), projection_sha256);
        Ok(())
    }

    pub(crate) fn verify_owner_delegation(module: &str) -> Result<(), Box<dyn std::error::Error>> {
        let source = fs::read_to_string(
            repository_root().join(format!("crates/sim_model/src/families/{module}.rs")),
        )?;
        for owner in [
            "ModelFamilyRegistration",
            "ModelFamilyStatePlanSelector",
            "ModelStateTransformPlanDefinition",
            "ModelProbe",
            "ModelForwardOperation",
            "MemoryEstimatorDescriptor",
        ] {
            assert!(source.contains(owner), "missing owner {owner}");
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
            assert!(!source.contains(forbidden), "forbidden owner {forbidden}");
        }
        Ok(())
    }

    fn source_tensors(
        fixture: &FamilyFixture,
        extras: &[TensorFixture],
        dtype: DType,
        backend: &CpuBackend,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        let supplied = fixture
            .source_weights
            .iter()
            .chain(extras)
            .map(|weight| (weight.key.as_str(), weight))
            .collect::<BTreeMap<_, _>>();
        for (key, weight) in &supplied {
            let expected = fixture
                .detector
                .tensor_shapes
                .get(*key)
                .ok_or_else(|| format!("source tensor {key} is absent from detector facts"))?;
            if expected != &weight.shape {
                return Err(format!(
                    "source tensor {key} shape {:?} differs from detector {expected:?}",
                    weight.shape
                )
                .into());
            }
        }
        fixture
            .detector
            .tensor_shapes
            .iter()
            .map(|(key, shape)| {
                let synthesized;
                let weight = if let Some(weight) = supplied.get(key.as_str()) {
                    *weight
                } else {
                    let count = usize::try_from(shape.iter().try_fold(
                        1_u64,
                        |count, value| count.checked_mul(*value).ok_or("tensor size overflow"),
                    )?)?;
                    synthesized = TensorFixture::new(key, shape, &vec![0.0; count]);
                    &synthesized
                };
                Ok((key.clone(), tensor(backend, weight, dtype, context)?))
            })
            .collect()
    }

    fn tensor(
        backend: &CpuBackend,
        fixture: &TensorFixture,
        dtype: DType,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            fixture.shape.clone(),
            DType::F32,
            backend.device(),
            context.stream,
        )?;
        let (tensor, _) = backend.upload_f32(descriptor, &fixture.values, context)?;
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
        for (actual, checkpoint) in actual.iter().zip(expected) {
            assert_eq!(actual.name, checkpoint.name);
            let values = tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
            assert_eq!(values.len(), checkpoint.values.len());
            for (index, (actual_value, expected_value)) in
                values.iter().zip(&checkpoint.values).enumerate()
            {
                if (actual_value - expected_value).abs() > 1.0e-5 {
                    return Err(format!(
                        "{}[{index}] expected {expected_value}, got {actual_value}",
                        checkpoint.name
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn write_safetensors(
        path: &Path,
        fixture: &FamilyFixture,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut header = serde_json::Map::new();
        if !fixture.detector.metadata.is_empty() {
            header.insert(
                "__metadata__".to_owned(),
                serde_json::to_value(&fixture.detector.metadata)?,
            );
        }
        let mut data = Vec::new();
        for (key, shape) in &fixture.detector.tensor_shapes {
            let count = usize::try_from(shape.iter().try_fold(1_u64, |count, value| {
                count.checked_mul(*value).ok_or("tensor size overflow")
            })?)?;
            let start = data.len();
            data.resize(start + count * std::mem::size_of::<f32>(), 0);
            header.insert(
                key.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
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

    fn fixture_directory(fixture: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sim_test_support/fixtures/models")
            .join(fixture)
    }

    fn repository_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
}
