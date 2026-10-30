use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    SdxlLayout, SdxlVariant, describe_model_family,
    generated_sdxl_instructpix2pix_sim_model_0125 as ip2p,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [ip2p::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9125",
    identifier: "SDXLInstructPix2PixAmbiguousFixture",
    ..ip2p::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    ip2p::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 907,
        source_architecture: "model_base.SDXLInstructPix2PixAmbiguousFixture",
        ..ip2p::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sdxl_instructpix2pix_source_layouts_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    support::verify_provenance(
        ip2p::MODEL_FAMILY_FIXTURE,
        ip2p::MODEL_FAMILY_FEATURE_ID,
        ip2p::MODEL_FAMILY_IDENTIFIER,
        ip2p::MODEL_FAMILY_SOURCE_ORDINAL,
        ip2p::SOURCE_ARCHITECTURE,
        ip2p::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    assert_eq!(ip2p::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.8);
    let descriptor = describe_model_family(&ip2p::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    for layout in [
        SdxlLayout::PrefixedNative,
        SdxlLayout::StandaloneNative,
        SdxlLayout::Diffusers,
    ] {
        let probe = support::variant_probe(layout, 8, 10);
        let configuration = ip2p::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, SdxlVariant::InstructPix2Pix);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, 8);
        assert_eq!(configuration.transformer_depth_middle, 10);
        assert_eq!(registry.resolve(&probe)?.detection().score, 1_100);
    }

    let store_probe = support::probe_through_model_store(ip2p::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(
        registry.resolve(&store_probe)?.detection().identity.feature_id(),
        ip2p::MODEL_FAMILY_FEATURE_ID
    );
    let mut misleading = store_probe.clone();
    misleading
        .metadata
        .insert("model_family".to_owned(), "SSD1B".to_owned());
    assert_eq!(registry.resolve(&misleading)?.profile().latent_identifier, "SDXL");
    let mut partial = store_probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.label_emb.0.0.weight");
    assert!(registry.resolve(&partial).is_err());
    assert!(matches!(
        ip2p::configuration_for_probe(&support::variant_probe(SdxlLayout::PrefixedNative, 4, 10)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("8-channel/depth-10")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_100, .. })
    ));
    support::verify_owner_delegation("sdxl_instructpix2pix_comfy_model_0125")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sdxl_instructpix2pix_native_execution_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_sdxl(ip2p::MODEL_FAMILY_FIXTURE, &REGISTRATIONS)?;
    super::write_model_family_row_artifact(
        ip2p::MODEL_FAMILY_FIXTURE,
        ip2p::MODEL_FAMILY_FEATURE_ID,
        ip2p::MODEL_FAMILY_IDENTIFIER,
        ip2p::MODEL_FAMILY_SOURCE_ORDINAL,
        "sdxl_instructpix2pix_comfy_model_0125",
        &[
            "source-and-catalog-provenance",
            "native-standalone-and-diffusers-key-detection",
            "source-exact-instruct-pix2pix-profile",
            "transactional-sdxl-component-routing",
            "named-native-forward-and-conditioning-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-misleading-and-cross-variant-failures",
            "canonical-sdxl-owner-delegation",
        ],
    )?;
    Ok(())
}

pub(super) mod support {
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelFamilyRegistration,
        ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
        NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
        PatchOperation, PatchTarget, SdxlLayout, build_model_family, map_model_weights,
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

    pub(crate) fn variant_probe(layout: SdxlLayout, input_channels: u64, depth: usize) -> ModelProbe {
        let (input, time, adm, output, residual, deep) = match layout {
            SdxlLayout::PrefixedNative => (
                "model.diffusion_model.input_blocks.0.0.weight",
                "model.diffusion_model.time_embed.0.weight",
                "model.diffusion_model.label_emb.0.0.weight",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.input_blocks.2.0.in_layers.0.weight",
                "model.diffusion_model.input_blocks.7.1.transformer_blocks.",
            ),
            SdxlLayout::StandaloneNative => (
                "input_blocks.0.0.weight",
                "time_embed.0.weight",
                "label_emb.0.0.weight",
                "out.2.weight",
                "input_blocks.2.0.in_layers.0.weight",
                "input_blocks.7.1.transformer_blocks.",
            ),
            SdxlLayout::Diffusers => (
                "conv_in.weight",
                "time_embedding.linear_1.weight",
                "add_embedding.linear_1.weight",
                "conv_out.weight",
                "down_blocks.0.resnets.1.conv1.weight",
                "down_blocks.2.attentions.0.transformer_blocks.",
            ),
        };
        let mut shapes = BTreeMap::from([
            (input.to_owned(), vec![320, input_channels, 3, 3]),
            (time.to_owned(), vec![1_280, 320]),
            (adm.to_owned(), vec![1, 2_816]),
            (output.to_owned(), vec![4, 320, 3, 3]),
            (residual.to_owned(), vec![1]),
        ]);
        for index in 0..depth {
            shapes.insert(format!("{deep}{index}.attn2.to_k.weight"), vec![1, 2_048]);
        }
        ModelProbe {
            tensor_shapes: shapes,
            metadata: BTreeMap::new(),
        }
    }

    pub(crate) fn probe_through_model_store(
        fixture_id: &str,
    ) -> Result<ModelProbe, Box<dyn std::error::Error>> {
        let fixture = fixture(fixture_id)?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("family.safetensors");
        write_safetensors(&path, &fixture.detector)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "sdxl-family-row",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("sdxl-family-row", "family.safetensors")?,
            &cancellation,
        )?;
        let probe = store.family_probe(&loaded, &cancellation)?;
        assert_eq!(probe.tensor_shapes(), &fixture.detector.tensor_shapes);
        Ok(probe)
    }

    pub(crate) fn exercise_sdxl(
        fixture_id: &str,
        registrations: &'static [ModelFamilyRegistration],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = fixture(fixture_id)?;
        let registration = registrations.first().ok_or("missing registration")?;
        assert_eq!(fixture.fixture_id, fixture_id);
        assert_eq!(fixture.feature_id, registration.definition.feature_id);
        assert_eq!(fixture.dtype, DType::F32);
        assert_eq!(fixture.device, DeviceKind::Cpu);
        assert_eq!(fixture.memory_budget_bytes, fixture.expected_memory_bytes);
        let probe = ModelProbe {
            tensor_shapes: fixture.detector.tensor_shapes.clone(),
            metadata: fixture.detector.metadata.clone(),
        };
        let registry = ModelFamilyRegistry::checked_registrations(registrations)?;
        let resolved = registry.resolve(&probe)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024 * 1024)?,
            &cancellation,
        );
        let source = detector_tensors(&fixture, &backend, &context)?;
        let mapped = resolved.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            &fixture.base_artifact_digest,
            &source,
        )?;
        for component in ["denoiser", "text_encoder", "vae"] {
            assert!(mapped.component(component).is_some(), "missing {component}");
        }
        let compact_source = compact_source_tensors(&fixture, DType::F32, &backend, &context)?;
        let weights = map_model_weights(
            registration.definition,
            &fixture.base_artifact_digest,
            compact_source,
        )?;
        let options = NativeFamilyBuildOptions {
            dtype: DType::F32,
            device: DeviceKind::Cpu,
            activation_elements: fixture.activation_elements,
            memory_budget_bytes: u64::MAX,
            allow_unexpected_weights: false,
        };
        let model = build_model_family(registration.definition, weights.clone(), options)?;
        assert_eq!(model.memory_estimate().total_bytes, fixture.expected_memory_bytes);
        let input = tensor(&backend, &fixture.input, DType::F32, &context)?;
        assert_checkpoints(
            &backend,
            &context,
            model.forward_checkpoints(&backend, &input, &context)?,
            &fixture.checkpoints,
        )?;
        let patch = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?;
        let patched = patch.apply(&backend, &weights, &context)?;
        assert_checkpoints(
            &backend,
            &context,
            model
                .with_weights(patched)?
                .forward_checkpoints(&backend, &input, &context)?,
            &fixture.patched_checkpoints,
        )?;

        let add = fixture.patches[0].clone();
        let replace = PatchOperation {
            identifier: "ordered-sdxl-replacement".to_owned(),
            kind: PatchKind::Adapter,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.out.2.weight".to_owned(),
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
                ordered.tensors().get("native.out.2.weight").ok_or("ordered output")?,
                &context,
            )?,
            tensor_to_f32_with_context_exact_native(
                &backend,
                reversed.tensors().get("native.out.2.weight").ok_or("reversed output")?,
                &context,
            )?
        );

        for dtype in [DType::F16, DType::Bf16, DType::F32] {
            let typed_source = compact_source_tensors(&fixture, dtype, &backend, &context)?;
            let typed_weights = map_model_weights(
                registration.definition,
                &fixture.base_artifact_digest,
                typed_source,
            )?;
            let mut typed = options;
            typed.dtype = dtype;
            build_model_family(registration.definition, typed_weights, typed)?;
        }
        let mut unsupported_dtype = options;
        unsupported_dtype.dtype = DType::F64;
        assert!(matches!(
            build_model_family(registration.definition, weights.clone(), unsupported_dtype),
            Err(ModelFamilyError::UnsupportedDType(DType::F64))
        ));
        let mut unsupported_device = options;
        unsupported_device.device = DeviceKind::Metal;
        assert!(matches!(
            build_model_family(registration.definition, weights.clone(), unsupported_device),
            Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
        ));
        let mut oom = options;
        oom.memory_budget_bytes = fixture.expected_memory_bytes - 1;
        assert!(matches!(
            build_model_family(registration.definition, weights, oom),
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
        fixture_id: &str,
        feature_id: &str,
        identifier: &str,
        source_ordinal: u16,
        architecture: &str,
        projection_sha256: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let provenance: serde_json::Value = serde_json::from_slice(&fs::read(
            fixture_directory(fixture_id).join("provenance.json"),
        )?)?;
        assert_eq!(provenance["fixture_id"], fixture_id);
        assert_eq!(provenance["feature_id"], feature_id);
        assert_eq!(provenance["source_symbol"], identifier);
        assert_eq!(provenance["source_ordinal"], source_ordinal);
        assert_eq!(provenance["source_architecture"], architecture);
        assert_eq!(provenance["catalog_projection_sha256"], projection_sha256);
        let projection = provenance["source_projection"].as_str().ok_or("source projection")?;
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
        for delegation in [
            "sdxl_family::configuration_for_probe",
            "sdxl_family::SDXL_LAYOUT_SIGNATURES",
            "sdxl_family::SDXL_STATE_PLAN_CASES",
            "sdxl_family::SDXL_FORWARD_PROGRAM",
            "sdxl_family::SDXL_COMPONENT_STATE_SCHEMAS",
        ] {
            assert!(source.contains(delegation), "missing delegation {delegation}");
        }
        for forbidden in [
            "struct CancellationToken",
            "struct Tensor",
            "struct ModelStore",
            "struct ModelStateTransaction",
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

    fn fixture(fixture_id: &str) -> Result<FamilyFixture, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(
            fixture_directory(fixture_id).join("family.json"),
        )?)?)
    }

    fn compact_source_tensors(
        fixture: &FamilyFixture,
        dtype: DType,
        backend: &CpuBackend,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        fixture
            .source_weights
            .iter()
            .map(|weight| Ok((weight.key.clone(), tensor(backend, weight, dtype, context)?)))
            .collect()
    }

    fn detector_tensors(
        fixture: &FamilyFixture,
        backend: &CpuBackend,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        fixture
            .detector
            .tensor_shapes
            .iter()
            .map(|(key, shape)| {
                let elements = usize::try_from(shape.iter().try_fold(1_u64, |count, value| {
                    count.checked_mul(*value).ok_or("tensor size overflow")
                })?)?;
                let weight = TensorFixture {
                    key: key.clone(),
                    shape: shape.clone(),
                    values: vec![0.0; elements],
                };
                Ok((key.clone(), tensor(backend, &weight, DType::F32, context)?))
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
        context: &sim_tensor::ExecutionContext<'_>,
        actual: Vec<sim_model::ModelForwardCheckpoint>,
        expected: &[CheckpointFixture],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.name, expected.name);
            let values =
                tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
            assert_eq!(values.len(), expected.values.len());
            for (value, expected) in values.iter().zip(&expected.values) {
                assert!((value - expected).abs() <= 1.0e-5);
            }
        }
        Ok(())
    }

    fn write_safetensors(
        path: &Path,
        detector: &DetectorFixture,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut header = serde_json::Map::new();
        if !detector.metadata.is_empty() {
            header.insert("__metadata__".to_owned(), serde_json::to_value(&detector.metadata)?);
        }
        let mut data = Vec::new();
        for (key, shape) in &detector.tensor_shapes {
            let count = usize::try_from(shape.iter().try_fold(1_u64, |count, value| {
                count.checked_mul(*value).ok_or("tensor size overflow")
            })?)?;
            let start = data.len();
            data.resize(start + count * std::mem::size_of::<f32>(), 0);
            header.insert(
                key.clone(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header)?;
        let mut file = fs::File::create(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(&data)?;
        Ok(())
    }

    fn fixture_directory(fixture_id: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sim_test_support/fixtures/models")
            .join(fixture_id)
    }

    fn repository_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
}
