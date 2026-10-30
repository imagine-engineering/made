use sim_model::{
    ModelFamilyDefinition, ModelFamilyRegistration, SdxlVariant,
    generated_koala_1b_sim_model_0097 as row_1b,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9097",
    identifier: "KOALA_1B_AmbiguousFixture",
    ..row_1b::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_1b::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 912,
        source_architecture: "model_base.KOALA_1B_AmbiguousFixture",
        ..row_1b::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: support::KoalaSpec = support::KoalaSpec {
    feature_id: row_1b::MODEL_FAMILY_FEATURE_ID,
    identifier: row_1b::MODEL_FAMILY_IDENTIFIER,
    fixture: row_1b::MODEL_FAMILY_FIXTURE,
    module: "koala_1b_comfy_model_0097",
    source_ordinal: row_1b::MODEL_FAMILY_SOURCE_ORDINAL,
    architecture_version: "koala-1b-sdxl-unet-v1",
    variant: SdxlVariant::Koala1B,
    depth: 6,
    middle_depth: 6,
    projection_sha256: row_1b::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_1b::MODEL_FAMILY_REGISTRATION,
};

#[test]
fn val_model_family_row_001_koala_1b_source_detection_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    support::assert_source_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}

#[test]
fn val_model_family_row_001_koala_1b_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::assert_execution_contract(SPEC)
}

pub(super) mod support {
    use sim_model::{
        ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyError, ModelFamilyRegistration,
        ModelFamilyRegistry, ModelParsedFacts, ModelParsedFormatFact, ModelParsedTensorFact,
        ModelProbe, ModelStateTransaction, ModelStore, NativeFamilyBuildOptions, ParserLimits,
        PatchGraph, SdxlLayout, SdxlVariant, build_model_family, describe_model_family,
        generated_koala_1b_sim_model_0097 as row_1b,
        generated_koala_700m_sim_model_0098 as row_700m, map_model_weights,
        sdxl_configuration_for_probe, sdxl_state_plan_for_layout,
    };
    use sim_tensor::{
        CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor,
        TensorBackend, TensorDescriptor,
        generated_sim_operator_indirection_01::{
            cast_to_with_context_exact_native, tensor_to_f32_with_context_exact_native,
        },
    };
    use sim_types::{CancellationToken, DeviceKind};
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::{
        collections::BTreeMap,
        fs,
        io::{Seek, SeekFrom, Write},
        path::Path,
    };

    #[derive(Clone, Copy)]
    pub(crate) struct KoalaSpec {
        pub(crate) feature_id: &'static str,
        pub(crate) identifier: &'static str,
        pub(crate) fixture: &'static str,
        pub(crate) module: &'static str,
        pub(crate) source_ordinal: u16,
        pub(crate) architecture_version: &'static str,
        pub(crate) variant: SdxlVariant,
        pub(crate) depth: usize,
        pub(crate) middle_depth: isize,
        pub(crate) projection_sha256: &'static str,
        pub(crate) registration: ModelFamilyRegistration,
    }

    #[derive(Debug, Deserialize)]
    pub(crate) struct FamilyFixture {
        pub(crate) fixture_id: String,
        pub(crate) feature_id: String,
        pub(crate) detector: DetectorFixture,
        pub(crate) base_artifact_digest: String,
        source_weights: Vec<TensorFixture>,
        input: TensorFixture,
        dtype: DType,
        device: DeviceKind,
        activation_elements: u64,
        memory_budget_bytes: u64,
        expected_memory_bytes: u64,
        checkpoints: Vec<CheckpointFixture>,
        patches: Vec<sim_model::PatchOperation>,
        patched_checkpoints: Vec<CheckpointFixture>,
    }

    #[derive(Debug, Deserialize)]
    pub(crate) struct DetectorFixture {
        pub(crate) tensor_shapes: BTreeMap<String, Vec<u64>>,
        pub(crate) metadata: BTreeMap<String, String>,
    }

    #[derive(Debug, Deserialize)]
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

    pub(crate) fn assert_source_contract(
        spec: KoalaSpec,
        ambiguous_registrations: &'static [ModelFamilyRegistration],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = load_fixture(spec.fixture)?;
        assert_eq!(fixture.fixture_id, spec.fixture);
        assert_eq!(fixture.feature_id, spec.feature_id);
        verify_provenance(spec)?;
        let descriptor = describe_model_family(spec.registration.definition)?;
        assert_eq!(descriptor.identifier, spec.identifier);
        assert_eq!(descriptor.architecture_version, spec.architecture_version);
        assert_eq!(descriptor.latent_format, "SDXL");
        assert_eq!(descriptor.component_graph.len(), 3);
        assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
        assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
        assert_eq!(spec.registration.source_ordinal, spec.source_ordinal);
        assert_eq!(spec.registration.source_architecture, "model_base.SDXL");

        let registry = ModelFamilyRegistry::checked_registrations(&[
            row_700m::MODEL_FAMILY_REGISTRATION,
            row_1b::MODEL_FAMILY_REGISTRATION,
        ])?;
        for layout in [
            SdxlLayout::PrefixedNative,
            SdxlLayout::StandaloneNative,
            SdxlLayout::Diffusers,
        ] {
            let probe = koala_probe(layout, spec.depth, DType::F32);
            let configuration = sdxl_configuration_for_probe(&probe)?;
            assert_eq!(configuration.variant, spec.variant);
            assert_eq!(configuration.layout, layout);
            assert_eq!(configuration.model_channels, 320);
            assert_eq!(configuration.in_channels, 4);
            assert_eq!(configuration.out_channels, 4);
            assert_eq!(configuration.context_dimension, 2_048);
            assert_eq!(configuration.adm_in_channels, 2_816);
            assert_eq!(configuration.num_res_blocks, [1, 1, 1]);
            assert_eq!(configuration.transformer_depth_middle, spec.middle_depth);
            assert!(configuration.uses_linear_transformer_projection);
            assert!(!configuration.uses_temporal_attention);
            assert_eq!(configuration.memory_usage_factor, 0.8);
            assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0047");
            assert_eq!(
                registry.resolve(&probe)?.detection().identity.feature_id(),
                spec.feature_id
            );
        }

        let stored = probe_through_model_store(&fixture)?;
        assert_eq!(
            registry.resolve(&stored)?.detection().identity.feature_id(),
            spec.feature_id
        );
        let mut misleading = koala_probe(SdxlLayout::PrefixedNative, spec.depth, DType::F32);
        misleading
            .metadata
            .insert("image_model".to_owned(), "sd15".to_owned());
        assert_eq!(
            registry.resolve(&misleading)?.detection().identity.feature_id(),
            spec.feature_id
        );
        let mut partial = koala_probe(SdxlLayout::PrefixedNative, spec.depth, DType::F32);
        partial
            .tensor_shapes
            .remove("model.diffusion_model.label_emb.0.0.weight");
        assert!(registry.detect(&partial).is_err());
        let mut malformed = koala_probe(SdxlLayout::PrefixedNative, spec.depth, DType::F32);
        malformed.tensor_shapes.insert(
            "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
            vec![320, 8, 3, 3],
        );
        assert!(matches!(
            sdxl_configuration_for_probe(&malformed),
            Err(ModelFamilyError::InvalidSelectorOutput(_))
        ));
        let valid = koala_probe(SdxlLayout::PrefixedNative, spec.depth, DType::F32);
        assert!(matches!(
            ModelFamilyRegistry::checked_registrations(ambiguous_registrations)?.detect(&valid),
            Err(ModelFamilyError::AmbiguousDetection { .. })
        ));
        verify_owner_delegation(spec.module)?;
        Ok(())
    }

    pub(crate) fn assert_execution_contract(
        spec: KoalaSpec,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = load_fixture(spec.fixture)?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024 * 1024)?,
            &cancellation,
        );

        for layout in [
            SdxlLayout::PrefixedNative,
            SdxlLayout::StandaloneNative,
            SdxlLayout::Diffusers,
        ] {
            let mapped = ModelStateTransaction::new(&backend, &context).execute(
                &sdxl_state_plan_for_layout(layout).compile()?,
                &fixture.base_artifact_digest,
                &mapping_source(&backend, &context, layout)?,
            )?;
            let denoiser = mapped.component("denoiser").ok_or("missing denoiser")?;
            for key in [
                "native.input_blocks.0.0.weight",
                "native.time_embed.0.weight",
                "native.label_emb.0.0.weight",
                "native.out.2.weight",
            ] {
                assert!(denoiser.contains_key(key), "{layout:?}: {key}");
            }
            assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(2));
            assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        }

        exercise_legacy(&fixture, spec.registration, &backend, &context)?;
        let cancellation_source =
            mapping_source(&backend, &context, SdxlLayout::PrefixedNative)?;
        cancellation.cancel();
        assert!(matches!(
            ModelStateTransaction::new(&backend, &context).execute(
                &sdxl_state_plan_for_layout(SdxlLayout::PrefixedNative).compile()?,
                &fixture.base_artifact_digest,
                &cancellation_source,
            ),
            Err(ModelFamilyError::Cancelled(_))
        ));
        super::super::write_model_family_row_artifact(
            spec.fixture,
            spec.feature_id,
            spec.identifier,
            spec.source_ordinal,
            spec.module,
            &[
                "source-catalog-provenance-and-registration",
                "prefixed-standalone-and-diffusers-key-detection",
                "koala-depth-profile-clip-and-sdxl-latent",
                "transactional-denoiser-text-and-vae-routing",
                "named-native-forward-and-patch-order",
                "memory-oom-dtype-device-and-cancellation",
                "partial-malformed-ambiguous-and-misleading-failures",
                "canonical-sdxl-owner-delegation",
            ],
        )?;
        Ok(())
    }

    pub(crate) fn exercise_legacy(
        fixture: &FamilyFixture,
        registration: ModelFamilyRegistration,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(fixture.dtype, DType::F32);
        assert_eq!(fixture.device, DeviceKind::Cpu);
        assert_eq!(fixture.memory_budget_bytes, fixture.expected_memory_bytes);
        let source = fixture_tensors(fixture, DType::F32, backend, context)?;
        let weights = map_model_weights(
            registration.definition,
            &fixture.base_artifact_digest,
            source,
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
        let input = tensor_from_fixture(backend, context, &fixture.input, DType::F32)?;
        assert_checkpoints(
            backend,
            context,
            model.forward_checkpoints(backend, &input, context)?,
            &fixture.checkpoints,
        )?;
        let patched = PatchGraph::checked(
            &fixture.base_artifact_digest,
            fixture.patches.clone(),
        )?
        .apply(backend, &weights, context)?;
        assert_checkpoints(
            backend,
            context,
            model
                .with_weights(patched)?
                .forward_checkpoints(backend, &input, context)?,
            &fixture.patched_checkpoints,
        )?;
        for dtype in registration.definition.supported_dtypes {
            let source = fixture_tensors(fixture, *dtype, backend, context)?;
            let weights = map_model_weights(
                registration.definition,
                &fixture.base_artifact_digest,
                source,
            )?;
            let mut typed = options;
            typed.dtype = *dtype;
            build_model_family(registration.definition, weights, typed)?;
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
        Ok(())
    }

    fn koala_probe(layout: SdxlLayout, depth: usize, dtype: DType) -> ModelProbe {
        let mut tensors = BTreeMap::new();
        match layout {
            SdxlLayout::PrefixedNative | SdxlLayout::StandaloneNative => {
                let prefix = if layout == SdxlLayout::PrefixedNative {
                    "model.diffusion_model."
                } else {
                    ""
                };
                add_fact(&mut tensors, format!("{prefix}input_blocks.0.0.weight"), &[320, 4, 3, 3], dtype);
                add_fact(&mut tensors, format!("{prefix}time_embed.0.weight"), &[1_280, 320], dtype);
                add_fact(&mut tensors, format!("{prefix}label_emb.0.0.weight"), &[1_280, 2_816], dtype);
                add_fact(&mut tensors, format!("{prefix}out.2.weight"), &[4, 320, 3, 3], dtype);
                add_depth(&mut tensors, &format!("{prefix}input_blocks.3.1.transformer_blocks."), 2, dtype);
                add_depth(&mut tensors, &format!("{prefix}input_blocks.5.1.transformer_blocks."), depth, dtype);
            }
            SdxlLayout::Diffusers => {
                add_fact(&mut tensors, "conv_in.weight".to_owned(), &[320, 4, 3, 3], dtype);
                add_fact(&mut tensors, "time_embedding.linear_1.weight".to_owned(), &[1_280, 320], dtype);
                add_fact(&mut tensors, "add_embedding.linear_1.weight".to_owned(), &[1_280, 2_816], dtype);
                add_fact(&mut tensors, "conv_out.weight".to_owned(), &[4, 320, 3, 3], dtype);
                add_depth(&mut tensors, "down_blocks.1.attentions.0.transformer_blocks.", 2, dtype);
                add_depth(&mut tensors, "down_blocks.2.attentions.0.transformer_blocks.", depth, dtype);
            }
        }
        ModelProbe::from_parsed_facts(ModelParsedFacts {
            tensors,
            formats: vec![ModelParsedFormatFact {
                identity: "safetensors".to_owned(),
                metadata: BTreeMap::new(),
            }],
        })
        .expect("KOALA probe fixture must be valid")
    }

    fn add_depth(
        tensors: &mut BTreeMap<String, ModelParsedTensorFact>,
        prefix: &str,
        depth: usize,
        dtype: DType,
    ) {
        for index in 0..depth {
            add_fact(
                tensors,
                format!("{prefix}{index}.attn2.to_k.weight"),
                &[320, 2_048],
                dtype,
            );
        }
    }

    fn add_fact(
        tensors: &mut BTreeMap<String, ModelParsedTensorFact>,
        key: String,
        shape: &[u64],
        dtype: DType,
    ) {
        tensors.insert(
            key,
            ModelParsedTensorFact {
                shape: shape.to_vec(),
                storage_dtype: dtype.catalog_name().to_owned(),
            },
        );
    }

    fn mapping_source(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        layout: SdxlLayout,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        let keys: &[(&str, &[u64], &[f32])] = match layout {
            SdxlLayout::PrefixedNative => &[
                ("model.diffusion_model.input_blocks.0.0.weight", &[1], &[1.0]),
                ("model.diffusion_model.time_embed.0.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                ("model.diffusion_model.label_emb.0.0.weight", &[1], &[1.0]),
                ("model.diffusion_model.out.2.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
                ("conditioner.embedders.0.transformer.text_model.weight", &[1], &[1.0]),
                ("conditioner.embedders.1.model.weight", &[1], &[1.0]),
                ("first_stage_model.decoder.weight", &[1], &[1.0]),
            ],
            SdxlLayout::StandaloneNative => &[
                ("input_blocks.0.0.weight", &[1], &[1.0]),
                ("time_embed.0.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                ("label_emb.0.0.weight", &[1], &[1.0]),
                ("out.2.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
                ("text_encoders.clip_l.weight", &[1], &[1.0]),
                ("text_encoders.clip_g.weight", &[1], &[1.0]),
                ("vae.decoder.weight", &[1], &[1.0]),
            ],
            SdxlLayout::Diffusers => &[
                ("conv_in.weight", &[1], &[1.0]),
                ("time_embedding.linear_1.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                ("add_embedding.linear_1.weight", &[1], &[1.0]),
                ("conv_out.weight", &[2, 2], &[1.0, 1.0, 1.0, -1.0]),
                ("text_encoder.model.weight", &[1], &[1.0]),
                ("text_encoder_2.model.weight", &[1], &[1.0]),
                ("vae.decoder.weight", &[1], &[1.0]),
            ],
        };
        keys.iter()
            .map(|(key, shape, values)| {
                Ok((
                    (*key).to_owned(),
                    tensor(backend, context, shape, values, DType::F32)?,
                ))
            })
            .collect()
    }

    pub(crate) fn load_fixture(
        fixture: &str,
    ) -> Result<FamilyFixture, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(
            fixture_directory(fixture).join("family.json"),
        )?)?)
    }

    pub(crate) fn probe_through_model_store(
        fixture: &FamilyFixture,
    ) -> Result<ModelProbe, Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("koala.safetensors");
        write_sparse_safetensors(&path, &fixture.detector)?;
        let mut index = ArtifactIndex::default();
        index.add_root(ArtifactRoot::canonical(
            "koala-row",
            "checkpoints",
            directory.path(),
            ["safetensors"],
        )?)?;
        let cancellation = CancellationToken::default();
        index.refresh(&cancellation)?;
        let mut store = ModelStore::new(ParserLimits::default())?;
        let loaded = store.load(
            &index,
            &ArtifactKey::new("koala-row", "koala.safetensors")?,
            &cancellation,
        )?;
        Ok(store.family_probe(&loaded, &cancellation)?)
    }

    fn write_sparse_safetensors(
        path: &Path,
        detector: &DetectorFixture,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for (key, shape) in &detector.tensor_shapes {
            let bytes = shape
                .iter()
                .try_fold(4_u64, |bytes, value| bytes.checked_mul(*value).ok_or("tensor size overflow"))?;
            let next = offset.checked_add(bytes).ok_or("safetensors offset overflow")?;
            header.insert(
                key.clone(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[offset,next]}),
            );
            offset = next;
        }
        if !detector.metadata.is_empty() {
            header.insert("__metadata__".to_owned(), serde_json::to_value(&detector.metadata)?);
        }
        let encoded = serde_json::to_vec(&header)?;
        let mut file = fs::File::create(path)?;
        file.write_all(&u64::try_from(encoded.len())?.to_le_bytes())?;
        file.write_all(&encoded)?;
        file.seek(SeekFrom::Start(8 + u64::try_from(encoded.len())? + offset))?;
        file.write_all(&[])?;
        file.set_len(8 + u64::try_from(encoded.len())? + offset)?;
        Ok(())
    }

    fn fixture_tensors(
        fixture: &FamilyFixture,
        dtype: DType,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        fixture
            .source_weights
            .iter()
            .map(|entry| {
                Ok((
                    entry.key.clone(),
                    tensor_from_fixture(backend, context, entry, dtype)?,
                ))
            })
            .collect()
    }

    fn tensor_from_fixture(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        fixture: &TensorFixture,
        dtype: DType,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        tensor(backend, context, &fixture.shape, &fixture.values, dtype)
    }

    pub(crate) fn tensor(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        shape: &[u64],
        values: &[f32],
        dtype: DType,
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
        context: &ExecutionContext<'_>,
        actual: Vec<sim_model::ModelForwardCheckpoint>,
        expected: &[CheckpointFixture],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(actual.len(), expected.len());
        for (actual, checkpoint) in actual.iter().zip(expected) {
            assert_eq!(actual.name, checkpoint.name);
            let values = tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
            for (index, (actual, expected)) in values.iter().zip(&checkpoint.values).enumerate() {
                if (actual - expected).abs() > 1.0e-5 {
                    return Err(format!(
                        "{}[{index}] expected {expected}, got {actual}",
                        checkpoint.name
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn verify_provenance(spec: KoalaSpec) -> Result<(), Box<dyn std::error::Error>> {
        let provenance: serde_json::Value = serde_json::from_slice(&fs::read(
            fixture_directory(spec.fixture).join("provenance.json"),
        )?)?;
        assert_eq!(provenance["feature_id"], spec.feature_id);
        assert_eq!(provenance["source_symbol"], spec.identifier);
        assert_eq!(provenance["source_ordinal"], spec.source_ordinal);
        assert_eq!(provenance["source_architecture"], "model_base.SDXL");
        assert_eq!(provenance["catalog_projection_sha256"], spec.projection_sha256);
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
            .and_then(|rows| rows.iter().find(|row| row["feature_id"] == spec.feature_id))
            .ok_or("catalog row")?;
        assert_eq!(sha256(&serde_json::to_vec(row)?), spec.projection_sha256);
        Ok(())
    }

    fn verify_owner_delegation(module: &str) -> Result<(), Box<dyn std::error::Error>> {
        let source = fs::read_to_string(
            repository_root().join(format!("crates/sim_model/src/families/{module}.rs")),
        )?;
        for owner in [
            "ModelFamilyRegistration",
            "ModelFamilyStatePlanSelector",
            "ModelProbe",
            "MemoryEstimatorDescriptor",
            "sdxl_configuration_for_probe",
            "SDXL_STATE_PLAN_CASES",
        ] {
            assert!(source.contains(owner), "missing canonical delegation {owner}");
        }
        for forbidden in [
            "struct Tensor",
            "struct ModelStore",
            "struct PatchGraph",
            "struct ModelStateTransaction",
            "std::fs",
            "std::process",
            "Command::",
            "unsafe ",
            "python",
        ] {
            assert!(!source.contains(forbidden), "forbidden owner {forbidden}");
        }
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
