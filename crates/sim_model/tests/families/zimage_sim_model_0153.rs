use super::generated_stable_zero123_sim_model_0136::support as row_support;
use sim_model::{
    LUMINA_ZIMAGE_STANDALONE_STATE_PLAN, LuminaZImageLayout, LuminaZImageVariant,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ZIMAGE_DIFFUSERS_STATE_PLAN, describe_model_family,
    generated_zimage_sim_model_0153 as zimage,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [zimage::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9153",
    identifier: "ZImage_AmbiguousFixture",
    ..zimage::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    zimage::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 946,
        source_architecture: "model_base.ZImage_AmbiguousFixture",
        ..zimage::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_zimage_source_configuration_and_layouts()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(zimage::MODEL_FAMILY_FIXTURE)?;
    row_support::verify_provenance(
        zimage::MODEL_FAMILY_FIXTURE,
        zimage::MODEL_FAMILY_FEATURE_ID,
        zimage::MODEL_FAMILY_IDENTIFIER,
        zimage::MODEL_FAMILY_SOURCE_ORDINAL,
        zimage::SOURCE_ARCHITECTURE,
        zimage::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let direct_probe = support::probe(&fixture);
    let store_probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(direct_probe.tensor_shapes(), store_probe.tensor_shapes());

    let configuration = zimage::configuration_for_probe(&store_probe)?;
    assert_eq!(configuration.variant, LuminaZImageVariant::ZImage);
    assert_eq!(configuration.layout, LuminaZImageLayout::PrefixedNative);
    assert_eq!(configuration.dimension, 3_840);
    assert_eq!(configuration.caption_feature_dimension, 2);
    assert_eq!(configuration.number_of_layers, 2);
    assert_eq!(configuration.number_of_heads, 30);
    assert_eq!(configuration.number_of_kv_heads, 30);
    assert_eq!(configuration.axes_dimensions, &[32, 48, 48]);
    assert_eq!(configuration.axes_lengths, &[1_536, 512, 512]);
    assert_eq!(configuration.rope_theta, 256.0);
    assert_eq!(configuration.feed_forward_multiplier, 8.0 / 3.0);
    assert_eq!(configuration.patch_size, 2);
    assert_eq!((configuration.input_channels, configuration.output_channels), (16, 16));
    assert!(configuration.qk_norm);
    assert!(configuration.zimage_modulation);
    assert_eq!(configuration.time_scale, Some(1_000.0));
    assert_eq!(configuration.pad_tokens_multiple, Some(32));
    assert_eq!(configuration.sampling_shift, 3.0);
    assert_eq!(configuration.memory_usage_factor, 2.8);
    assert_eq!(configuration.supported_dtypes, &[DType::Bf16, DType::F32]);

    let descriptor = describe_model_family(&zimage::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Flux");
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 3);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&store_probe)?;
    assert_eq!(resolved.detection().score, 900);
    assert_eq!(resolved.source_ordinal(), 46);
    assert_eq!(resolved.source_architecture(), zimage::SOURCE_ARCHITECTURE);
    assert_eq!(resolved.clip_target().candidates().len(), 1);

    let standalone = support::standalone_probe(&direct_probe);
    let standalone_configuration = zimage::configuration_for_probe(&standalone)?;
    assert_eq!(standalone_configuration.layout, LuminaZImageLayout::StandaloneNative);
    support::assert_selected_plan_identity(
        &registry,
        &standalone,
        &LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
    )?;

    let diffusers = support::zimage_diffusers_probe(2);
    let diffusers_configuration = zimage::configuration_for_probe(&diffusers)?;
    assert_eq!(diffusers_configuration.layout, LuminaZImageLayout::Diffusers);
    support::assert_selected_plan_identity(&registry, &diffusers, &ZIMAGE_DIFFUSERS_STATE_PLAN)?;
    support::exercise_selected_state_plan(
        &registry,
        &diffusers,
        &support::diffusers_state_source(),
        &["model", "text_encoder", "vae"],
        &[
            "native.x_embedder.weight",
            "native.cap_embedder.1.weight",
            "native.noise_refiner.0.attention.k_norm.weight",
            "native.final_layer.linear.weight",
        ],
    )?;

    let mut partial = direct_probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.x_embedder.weight");
    assert!(ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?.resolve(&partial).is_err());

    let mut malformed = direct_probe.clone();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.final_layer.linear.weight".to_owned(),
        vec![63, 3_840],
    );
    assert!(matches!(
        zimage::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("must both equal 64")
    ));

    let mut misleading = direct_probe.clone();
    misleading
        .metadata
        .insert("image_model".to_owned(), "zimage_pixel".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        zimage::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&direct_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 900, .. })
    ));
    row_support::verify_owner_delegation("zimage_comfy_model_0153")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_zimage_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(zimage::MODEL_FAMILY_FIXTURE)?;
    let extras = [
        support::TensorFixture::new("text_encoders.qwen3_4b.transformer.weight", &[1], &[1.0]),
        support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0]),
    ];
    support::exercise_family(
        &fixture,
        &REGISTRATIONS,
        &extras,
        &["model", "text_encoder", "vae"],
        "native.cap_embedder.1.weight",
    )?;
    super::write_model_family_row_artifact(
        zimage::MODEL_FAMILY_FIXTURE,
        zimage::MODEL_FAMILY_FEATURE_ID,
        zimage::MODEL_FAMILY_IDENTIFIER,
        zimage::MODEL_FAMILY_SOURCE_ORDINAL,
        "zimage_comfy_model_0153",
        &[
            "source-and-catalog-provenance",
            "model-store-prefixed-native-probe",
            "standalone-native-and-pinned-diffusers-selection",
            "checked-zimage-geometry-and-conditioning",
            "transactional-model-text-vae-routing-and-unmatched-rejection",
            "named-native-forward-and-patch-order",
            "exact-memory-oom-dtype-device-cancellation",
            "partial-malformed-metadata-and-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}

pub(crate) mod support {
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelFamilyRegistration,
        ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStateTransformPlanDefinition,
        ModelStore, NativeFamilyBuildOptions, PatchApplication, PatchGraph, PatchKind,
        PatchOperation, PatchTarget, build_model_family, map_model_weights,
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
    use std::{
        collections::BTreeMap,
        fs::{self, OpenOptions},
        io::Write,
        path::Path,
    };

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct FamilyFixture {
        pub(crate) fixture_id: String,
        pub(crate) feature_id: String,
        pub(crate) detector: ModelProbeFixture,
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

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct ModelProbeFixture {
        pub(crate) tensor_shapes: BTreeMap<String, Vec<u64>>,
        pub(crate) metadata: BTreeMap<String, String>,
    }

    #[derive(Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
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

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct CheckpointFixture {
        pub(crate) name: String,
        pub(crate) values: Vec<f32>,
    }

    pub(crate) fn load_fixture(
        fixture: &str,
    ) -> Result<FamilyFixture, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(
            fixture_directory(fixture).join("family.json"),
        )?)?)
    }

    pub(crate) fn probe(fixture: &FamilyFixture) -> ModelProbe {
        ModelProbe {
            tensor_shapes: fixture.detector.tensor_shapes.clone(),
            metadata: fixture.detector.metadata.clone(),
        }
    }

    pub(crate) fn standalone_probe(prefixed: &ModelProbe) -> ModelProbe {
        ModelProbe {
            tensor_shapes: prefixed
                .tensor_shapes
                .iter()
                .map(|(key, shape)| {
                    (
                        key.strip_prefix("model.diffusion_model.")
                            .unwrap_or(key)
                            .to_owned(),
                        shape.clone(),
                    )
                })
                .collect(),
            metadata: prefixed.metadata.clone(),
        }
    }

    pub(crate) fn zimage_diffusers_probe(layers: usize) -> ModelProbe {
        let mut tensor_shapes = BTreeMap::from([
            ("cap_embedder.1.weight".to_owned(), vec![3_840, 2]),
            (
                "noise_refiner.0.attention.norm_k.weight".to_owned(),
                vec![128],
            ),
            (
                "noise_refiner.0.attention.to_q.weight".to_owned(),
                vec![3_840, 3_840],
            ),
            (
                "noise_refiner.0.attention.to_k.weight".to_owned(),
                vec![3_840, 3_840],
            ),
            (
                "noise_refiner.0.attention.to_v.weight".to_owned(),
                vec![3_840, 3_840],
            ),
            ("all_x_embedder.2-1.weight".to_owned(), vec![3_840, 64]),
            (
                "all_final_layer.2-1.linear.weight".to_owned(),
                vec![64, 3_840],
            ),
        ]);
        for index in 0..layers {
            tensor_shapes.insert(
                format!("layers.{index}.attention.to_q.weight"),
                vec![3_840, 3_840],
            );
        }
        ModelProbe {
            tensor_shapes,
            metadata: BTreeMap::new(),
        }
    }

    pub(crate) fn probe_through_model_store(
        fixture: &FamilyFixture,
    ) -> Result<ModelProbe, Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("family.safetensors");
        write_sparse_safetensors(&path, &fixture.detector)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "model-family-row",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(sim_model::ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("model-family-row", "family.safetensors")?,
            &cancellation,
        )?;
        let probe = store.family_probe(&loaded, &cancellation)?;
        assert_eq!(probe.tensor_shapes(), &fixture.detector.tensor_shapes);
        Ok(probe)
    }

    pub(crate) fn assert_selected_plan_identity(
        registry: &ModelFamilyRegistry,
        probe: &ModelProbe,
        expected: &ModelStateTransformPlanDefinition,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            registry
                .resolve(probe)?
                .state_plan()
                .ok_or("missing selected state plan")?
                .identity(),
            expected.compile()?.identity()
        );
        Ok(())
    }

    pub(crate) fn diffusers_state_source() -> Vec<TensorFixture> {
        [
            "all_x_embedder.2-1.weight",
            "all_x_embedder.2-1.bias",
            "cap_embedder.1.weight",
            "cap_embedder.1.bias",
            "noise_refiner.0.attention.norm_k.weight",
            "noise_refiner.0.attention.to_q.weight",
            "layers.0.attention.to_q.weight",
            "all_final_layer.2-1.linear.weight",
            "all_final_layer.2-1.linear.bias",
            "text_encoders.qwen3_4b.transformer.weight",
            "vae.decoder.weight",
        ]
        .into_iter()
        .enumerate()
        .map(|(index, key)| TensorFixture::new(key, &[1], &[index as f32 + 1.0]))
        .collect()
    }

    pub(crate) fn exercise_selected_state_plan(
        registry: &ModelFamilyRegistry,
        probe: &ModelProbe,
        source: &[TensorFixture],
        expected_components: &[&str],
        expected_model_keys: &[&str],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let resolved = registry.resolve(probe)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(512 * 1024)?,
            &cancellation,
        );
        let source = tensor_map(source, DType::F32, &backend, &context)?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            resolved.state_plan().ok_or("missing state plan")?,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            &source,
        )?;
        for component in expected_components {
            assert!(mapped.component(component).is_some(), "missing {component}");
        }
        let model = mapped.component("model").ok_or("missing model component")?;
        for key in expected_model_keys {
            assert!(model.contains_key(*key), "missing {key}");
        }
        Ok(())
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
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4 * 1024 * 1024)?,
            &cancellation,
        );

        let state_source = fixture
            .source_weights
            .iter()
            .chain(extras)
            .cloned()
            .collect::<Vec<_>>();
        let state_source = tensor_map(&state_source, DType::F32, &backend, &context)?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            resolved.state_plan().ok_or("missing state plan")?,
            &fixture.base_artifact_digest,
            &state_source,
        )?;
        for component in expected_components {
            assert!(mapped.component(component).is_some(), "missing {component}");
        }
        let mut unmatched = state_source.clone();
        unmatched.insert(
            "rogue.weight".to_owned(),
            tensor_from_values(&backend, &[1], &[1.0], DType::F32, &context)?,
        );
        assert!(ModelStateTransaction::new(&backend, &context)
            .execute(
                resolved.state_plan().ok_or("missing state plan")?,
                &fixture.base_artifact_digest,
                &unmatched,
            )
            .is_err());

        let weights = map_model_weights(
            registration.definition,
            &fixture.base_artifact_digest,
            tensor_map(&fixture.source_weights, DType::F32, &backend, &context)?,
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
        let input = tensor_fixture(&backend, &fixture.input, DType::F32, &context)?;
        assert_checkpoints(
            &backend,
            model.forward_checkpoints(&backend, &input, &context)?,
            &fixture.checkpoints,
            &context,
        )?;
        let patched = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?
            .apply(&backend, &weights, &context)?;
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
        let reversed =
            PatchGraph::checked(&fixture.base_artifact_digest, vec![add, replace])?.apply(
                &backend,
                &weights,
                &context,
            )?;
        assert_ne!(
            tensor_to_f32_with_context_exact_native(
                &backend,
                ordered.tensors().get(patch_target).ok_or("ordered target")?,
                &context,
            )?,
            tensor_to_f32_with_context_exact_native(
                &backend,
                reversed.tensors().get(patch_target).ok_or("reversed target")?,
                &context,
            )?
        );

        for dtype in [DType::Bf16, DType::F32] {
            let weights = map_model_weights(
                registration.definition,
                &fixture.base_artifact_digest,
                tensor_map(&fixture.source_weights, dtype, &backend, &context)?,
            )?;
            let mut typed = options;
            typed.dtype = dtype;
            assert!(build_model_family(registration.definition, weights, typed).is_ok());
        }
        for dtype in [DType::F16, DType::F64] {
            let mut unsupported = options;
            unsupported.dtype = dtype;
            assert!(matches!(
                build_model_family(registration.definition, weights.clone(), unsupported),
                Err(ModelFamilyError::UnsupportedDType(actual)) if actual == dtype
            ));
        }
        let mut unsupported_device = options;
        unsupported_device.device = DeviceKind::Metal;
        assert!(matches!(
            build_model_family(
                registration.definition,
                weights.clone(),
                unsupported_device
            ),
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
            ModelStateTransaction::new(&backend, &context).execute(
                resolved.state_plan().ok_or("missing state plan")?,
                &fixture.base_artifact_digest,
                &state_source,
            ),
            Err(ModelFamilyError::Cancelled(_))
        ));
        Ok(())
    }

    fn tensor_map(
        fixtures: &[TensorFixture],
        dtype: DType,
        backend: &CpuBackend,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        fixtures
            .iter()
            .map(|fixture| {
                Ok((
                    fixture.key.clone(),
                    tensor_fixture(backend, fixture, dtype, context)?,
                ))
            })
            .collect()
    }

    fn tensor_fixture(
        backend: &CpuBackend,
        fixture: &TensorFixture,
        dtype: DType,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        tensor_from_values(backend, &fixture.shape, &fixture.values, dtype, context)
    }

    fn tensor_from_values(
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
        let tensor = backend.upload_f32(descriptor, values, context)?.0;
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
            let values =
                tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
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

    fn write_sparse_safetensors(
        path: &Path,
        detector: &ModelProbeFixture,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut header = serde_json::Map::new();
        if !detector.metadata.is_empty() {
            header.insert(
                "__metadata__".to_owned(),
                serde_json::to_value(&detector.metadata)?,
            );
        }
        let mut offset = 0_u64;
        for (key, shape) in &detector.tensor_shapes {
            let size = shape
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .ok_or("safetensors tensor size overflow")?;
            let end = offset.checked_add(size).ok_or("safetensors size overflow")?;
            header.insert(
                key.clone(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let header = serde_json::to_vec(&header)?;
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
        file.write_all(&header)?;
        file.set_len(
            8_u64
                .checked_add(u64::try_from(header.len())?)
                .and_then(|value| value.checked_add(offset))
                .ok_or("safetensors file size overflow")?,
        )?;
        Ok(())
    }

    fn fixture_directory(fixture: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sim_test_support/fixtures/models")
            .join(fixture)
    }
}
