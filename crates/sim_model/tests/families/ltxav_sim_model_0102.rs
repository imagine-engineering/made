use sim_model::{
    LTXAV_CONDITIONING, LTX_CLIP_TARGET, LtxLayout, LtxVariant, ModelFamilyError,
    ModelFamilyRegistry, ModelProbe, generated_ltxav_sim_model_0102 as ltxav,
    generated_ltxv_sim_model_0103 as ltxv, ltx_configuration_for_probe,
};
use std::collections::BTreeMap;

static REGISTRATIONS: [sim_model::ModelFamilyRegistration; 2] = [
    ltxv::MODEL_FAMILY_REGISTRATION,
    ltxav::MODEL_FAMILY_REGISTRATION,
];

#[test]
fn val_model_family_row_001_ltxav_source_detection_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(ltxav::MODEL_FAMILY_IDENTIFIER, "LTXAV");
    assert_eq!(ltxav::MODEL_FAMILY_SOURCE_ORDINAL, 33);
    assert_eq!(ltxav::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.LTXAV");
    let fixture = support::load_fixture(ltxav::MODEL_FAMILY_FIXTURE)?;
    let probe = support::probe(&fixture);
    let configuration = ltx_configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, LtxVariant::AudioVideo);
    assert_eq!(configuration.layout, LtxLayout::PrefixedNative);
    assert_eq!(configuration.input_channels, 4);
    assert_eq!(configuration.inner_dimension, 64);
    assert_eq!(configuration.number_of_layers, 1);
    assert_eq!(configuration.attention_head_dimension, 2);
    assert_eq!(configuration.number_of_attention_heads, 32);
    assert_eq!(configuration.cross_attention_dimension, 2_048);
    assert_eq!(configuration.audio_input_channels, Some(8));
    assert_eq!(configuration.audio_inner_dimension, Some(32));
    assert_eq!(configuration.memory_usage_factor, 0.077);
    assert_eq!(configuration.conditioning, LTXAV_CONDITIONING);
    assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0039");
    assert!(std::ptr::eq(configuration.clip_target, &LTX_CLIP_TARGET));
    for target_prefix in ["model.diffusion_model.", "model.", ""] {
        let layout_probe = rewrite_model_probe(&probe, target_prefix);
        assert_eq!(
            ltx_configuration_for_probe(&layout_probe)?.variant,
            LtxVariant::AudioVideo
        );
        support::exercise_state_plan(
            &REGISTRATIONS,
            ltxav::MODEL_FAMILY_FIXTURE,
            &layout_probe,
            |key| rewrite_model_key(key, target_prefix),
        )?;
    }

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let detection = registry.detect(&probe)?;
    assert_eq!(detection.identity.feature_id(), ltxav::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(detection.score, 1_400);
    let mut video_only = probe.clone();
    video_only.tensor_shapes.remove(
        "model.diffusion_model.audio_adaln_single.linear.weight",
    );
    video_only
        .tensor_shapes
        .remove("model.diffusion_model.audio_patchify_proj.weight");
    assert_eq!(
        registry.detect(&video_only)?.identity.feature_id(),
        ltxv::MODEL_FAMILY_FEATURE_ID
    );

    let mut malformed = probe;
    malformed.tensor_shapes.insert(
        "model.diffusion_model.audio_patchify_proj.weight".to_owned(),
        vec![32, 0],
    );
    assert!(matches!(
        registry.resolve(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "transformer_blocks.0.attn1.to_q.weight".to_owned(),
            vec![64, 64],
        )]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        ltx_configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    support::validate_provenance(
        &ltxav::MODEL_FAMILY,
        ltxav::MODEL_FAMILY_FIXTURE,
        ltxav::MODEL_FAMILY_SOURCE_ORDINAL,
        "model_base.LTXAV",
        ltxav::MODEL_FAMILY_PROJECTION_SHA256,
        ltxav::MODEL_FAMILY_SOURCE_PATH,
        ltxav::MODEL_FAMILY_SOURCE_SHA256,
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_ltxav_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_registration(
        &REGISTRATIONS,
        ltxav::MODEL_FAMILY_FIXTURE,
        &[sim_tensor::DType::Bf16, sim_tensor::DType::F32],
    )?;
    super::write_model_family_row_artifact(
        ltxav::MODEL_FAMILY_FIXTURE,
        ltxav::MODEL_FAMILY_FEATURE_ID,
        ltxav::MODEL_FAMILY_IDENTIFIER,
        ltxav::MODEL_FAMILY_SOURCE_ORDINAL,
        "ltxav_comfy_model_0102",
        &[
            "source-provenance-registration-descriptor",
            "source-exact-audio-video-configuration-and-profile",
            "ltxav-ltxv-registry-precedence",
            "prefixed-saved-standalone-native-state-plans",
            "forward-checkpoints-and-patch-order",
            "bf16-f32-memory-device-oom-cancellation",
            "partial-malformed-diffusers-and-owner-delegation",
        ],
    )?;
    Ok(())
}

fn rewrite_model_probe(probe: &ModelProbe, target_prefix: &str) -> ModelProbe {
    ModelProbe {
        tensor_shapes: probe
            .tensor_shapes
            .iter()
            .map(|(key, shape)| (rewrite_model_key(key, target_prefix), shape.clone()))
            .collect(),
        metadata: probe.metadata.clone(),
    }
}

fn rewrite_model_key(key: &str, target_prefix: &str) -> String {
    key.strip_prefix("model.diffusion_model.")
        .map_or_else(|| key.to_owned(), |suffix| format!("{target_prefix}{suffix}"))
}

pub(crate) mod support {
    use sim_model::{
        ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
        ModelProbe, ModelStateTransaction, NativeFamilyBuildOptions, PatchGraph,
        build_model_family, describe_model_family, map_model_weights,
    };
    use sim_tensor::{
        CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend,
        generated_sim_operator_indirection_01::{
            tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
        },
    };
    use sim_types::{CancellationToken, DeviceKind};
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, path::Path};

    #[derive(Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct FamilyFixture {
        pub(crate) fixture_id: String,
        pub(crate) feature_id: String,
        pub(crate) detector: ProbeFixture,
        pub(crate) base_artifact_digest: String,
        pub(crate) source_weights: Vec<TensorFixture>,
        pub(crate) input: TensorFixture,
        pub(crate) dtype: DType,
        pub(crate) device: DeviceKind,
        pub(crate) activation_elements: u64,
        pub(crate) memory_budget_bytes: u64,
        pub(crate) expected_memory_bytes: u64,
        pub(crate) checkpoints: Vec<CheckpointFixture>,
        pub(crate) patches: Vec<sim_model::PatchOperation>,
        pub(crate) patched_checkpoints: Vec<CheckpointFixture>,
    }

    #[derive(Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct ProbeFixture {
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

    #[derive(Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct CheckpointFixture {
        pub(crate) name: String,
        pub(crate) values: Vec<f32>,
    }

    pub(crate) fn load_fixture(
        fixture: &str,
    ) -> Result<FamilyFixture, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&std::fs::read(
            fixture_directory(fixture).join("family.json"),
        )?)?)
    }

    pub(crate) fn probe(fixture: &FamilyFixture) -> ModelProbe {
        ModelProbe {
            tensor_shapes: fixture.detector.tensor_shapes.clone(),
            metadata: fixture.detector.metadata.clone(),
        }
    }

    pub(crate) fn validate_provenance(
        definition: &ModelFamilyDefinition,
        fixture: &str,
        ordinal: u16,
        architecture: &str,
        catalog_projection: &str,
        source_path: &str,
        source_sha256: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        describe_model_family(definition)?;
        let repository = repository_root();
        assert_eq!(sha256(&std::fs::read(repository.join(source_path))?), source_sha256);
        let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(
            fixture_directory(fixture).join("provenance.json"),
        )?)?;
        assert_eq!(provenance["fixture_id"], fixture);
        assert_eq!(provenance["feature_id"], definition.feature_id);
        assert_eq!(provenance["source_ordinal"], ordinal);
        assert_eq!(provenance["source_architecture"], architecture);
        assert_eq!(provenance["catalog_projection_sha256"], catalog_projection);
        let projection = provenance["source_projection"]
            .as_str()
            .ok_or("source projection must be text")?;
        assert_eq!(sha256(projection.as_bytes()), provenance["source_projection_sha256"]);
        for source in provenance["source_files"]
            .as_array()
            .ok_or("source files must be an array")?
        {
            let path = source["path"].as_str().ok_or("source path must be text")?;
            assert_eq!(sha256(&std::fs::read(repository.join(path))?), source["sha256"]);
        }
        let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
            repository.join("crates/sim_model/catalog/model-families-v1.json"),
        )?)?;
        let row = catalog["models"]
            .as_array()
            .ok_or("catalog models must be an array")?
            .iter()
            .find(|row| row["feature_id"] == definition.feature_id)
            .ok_or("catalog row is missing")?;
        assert_eq!(row["source_ordinal"], ordinal);
        assert_eq!(sha256(&serde_json::to_vec(row)?), catalog_projection);
        Ok(())
    }

    pub(crate) fn exercise_registration(
        registrations: &'static [ModelFamilyRegistration],
        fixture_name: &str,
        dtypes: &[DType],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = load_fixture(fixture_name)?;
        assert_eq!(fixture.fixture_id, fixture_name);
        assert!(dtypes.contains(&fixture.dtype));
        let registry = ModelFamilyRegistry::checked_registrations(registrations)?;
        let probe = probe(&fixture);
        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), fixture.feature_id);
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        for dtype in dtypes {
            let cancellation = CancellationToken::default();
            let context = backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(32 * 1024 * 1024)?,
                &cancellation,
            );
            let source = tensor_map(&fixture.source_weights, *dtype, &backend, &context)?;
            let transaction = ModelStateTransaction::new(&backend, &context);
            let components = transaction.execute(
                resolved.state_plan().ok_or("missing state plan")?,
                &fixture.base_artifact_digest,
                &source,
            )?;
            let primary = resolved
                .definition()
                .components
                .iter()
                .find(|component| component.required)
                .ok_or("missing required primary component")?
                .identifier;
            assert!(components
                .component(primary)
                .is_some_and(|component| !component.is_empty()));
            let weights = map_model_weights(
                resolved.definition(),
                &fixture.base_artifact_digest,
                source.clone(),
            )?;
            let options = NativeFamilyBuildOptions {
                dtype: *dtype,
                device: fixture.device,
                activation_elements: fixture.activation_elements,
                memory_budget_bytes: fixture.memory_budget_bytes,
                allow_unexpected_weights: false,
            };
            let model = build_model_family(resolved.definition(), weights.clone(), options)?;
            assert_eq!(model.memory_estimate().total_bytes, fixture.expected_memory_bytes);
            let input = tensor(&fixture.input, *dtype, &backend, &context)?;
            assert_checkpoints(
                &backend,
                &context,
                &model.forward_checkpoints(&backend, &input, &context)?,
                &fixture.checkpoints,
            )?;
            let patched =
                PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?
                    .apply(&backend, model.weights(), &context)?;
            let patched_model = model.with_weights(patched)?;
            assert_checkpoints(
                &backend,
                &context,
                &patched_model.forward_checkpoints(&backend, &input, &context)?,
                &fixture.patched_checkpoints,
            )?;
            let mut unsupported = options;
            unsupported.device = DeviceKind::Metal;
            assert!(matches!(
                build_model_family(resolved.definition(), weights.clone(), unsupported),
                Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
            ));
            let mut oom = options;
            oom.memory_budget_bytes = fixture.expected_memory_bytes.saturating_sub(1);
            assert!(matches!(
                build_model_family(resolved.definition(), weights, oom),
                Err(ModelFamilyError::OutOfMemory { .. })
            ));
            cancellation.cancel();
            assert!(matches!(
                ModelStateTransaction::new(&backend, &context).execute(
                    resolved.state_plan().ok_or("missing state plan")?,
                    &fixture.base_artifact_digest,
                    &source,
                ),
                Err(ModelFamilyError::Cancelled(_))
            ));
        }
        Ok(())
    }

    pub(crate) fn exercise_state_plan(
        registrations: &'static [ModelFamilyRegistration],
        fixture_name: &str,
        probe: &ModelProbe,
        rewrite_key: impl Fn(&str) -> String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture = load_fixture(fixture_name)?;
        let resolved = ModelFamilyRegistry::checked_registrations(registrations)?.resolve(probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), fixture.feature_id);
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32 * 1024 * 1024)?,
            &cancellation,
        );
        let source = fixture
            .source_weights
            .iter()
            .map(|fixture| {
                Ok((
                    rewrite_key(&fixture.key),
                    tensor(fixture, DType::F32, &backend, &context)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
        let components = ModelStateTransaction::new(&backend, &context).execute(
            resolved.state_plan().ok_or("missing state plan")?,
            &fixture.base_artifact_digest,
            &source,
        )?;
        let primary = resolved
            .definition()
            .components
            .iter()
            .find(|component| component.required)
            .ok_or("missing required primary component")?
            .identifier;
        assert!(components
            .component(primary)
            .is_some_and(|component| !component.is_empty()));
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
                    tensor(fixture, dtype, backend, context)?,
                ))
            })
            .collect()
    }

    fn tensor(
        fixture: &TensorFixture,
        dtype: DType,
        backend: &CpuBackend,
        context: &sim_tensor::ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        Ok(tensor_from_f32_with_context_exact_native(
            backend,
            &fixture.shape,
            &fixture.values,
            dtype,
            backend.device(),
            context,
        )?)
    }

    fn assert_checkpoints(
        backend: &CpuBackend,
        context: &sim_tensor::ExecutionContext<'_>,
        actual: &[sim_model::ModelForwardCheckpoint],
        expected: &[CheckpointFixture],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(actual.len(), expected.len());
        for (checkpoint, expected) in actual.iter().zip(expected) {
            assert_eq!(checkpoint.name, expected.name);
            let values = tensor_to_f32_with_context_exact_native(
                backend,
                &checkpoint.tensor,
                context,
            )?;
            assert_eq!(values.len(), expected.values.len());
            for (actual, expected) in values.iter().zip(&expected.values) {
                assert!(
                    (actual - expected).abs() <= 2.0e-2,
                    "{}: {actual} != {expected}",
                    checkpoint.name
                );
            }
        }
        Ok(())
    }

    fn fixture_directory(fixture: &str) -> std::path::PathBuf {
        repository_root()
            .join("crates/sim_test_support/fixtures/models")
            .join(fixture)
    }

    fn repository_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
}
