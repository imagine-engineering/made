use sim_model::{
    LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN, LuminaZImageLayout, LuminaZImageVariant,
    ModelFamilyError, ModelProbe, generated_lumina2_sim_model_0107 as lumina,
    lumina_zimage_state_plan_for_layout,
};
use std::collections::BTreeMap;

#[test]
fn source_projection_descriptor_fixture_and_fail_closed_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(lumina::MODEL_FAMILY_IDENTIFIER, "Lumina2");
    assert_eq!(lumina::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0107");
    assert_eq!(lumina::MODEL_FAMILY_SOURCE_ORDINAL, 49);
    assert_eq!(lumina::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.Lumina2");
    assert_eq!(lumina::MODEL_FAMILY_SAMPLING_SHIFT, 6.0);
    assert_eq!(lumina::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.4);
    support::validate_provenance(
        lumina::MODEL_FAMILY_FIXTURE,
        lumina::MODEL_FAMILY_FEATURE_ID,
        lumina::MODEL_FAMILY_IDENTIFIER,
        lumina::MODEL_FAMILY_SOURCE_ORDINAL,
        lumina::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    support::exercise_fixture(
        &lumina::MODEL_FAMILY,
        lumina::MODEL_FAMILY_FIXTURE,
        "lumina2_comfy_model_0107",
        49,
    )?;
    Ok(())
}

#[test]
fn exact_native_layouts_execute_and_invalid_or_ambiguous_probes_fail()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        LuminaZImageLayout::PrefixedNative,
        LuminaZImageLayout::SavedModel,
        LuminaZImageLayout::StandaloneNative,
    ] {
        let probe = probe(layout);
        let configuration = lumina::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, LuminaZImageVariant::Lumina2);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.dimension, 2_304);
        assert_eq!(configuration.number_of_heads, 24);
        assert_eq!(configuration.number_of_kv_heads, 8);
        support::exercise_plan(
            lumina_zimage_state_plan_for_layout(layout),
            &mapping_keys(layout),
            &["native.x_embedder.weight", "native.final_layer.linear.weight"],
        )?;
    }
    assert_eq!(
        lumina_zimage_state_plan_for_layout(LuminaZImageLayout::SavedModel).encoded_plan,
        LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN.encoded_plan
    );

    let mut partial = probe(LuminaZImageLayout::StandaloneNative);
    partial.tensor_shapes.remove("x_embedder.weight");
    assert!(matches!(
        lumina::configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut ambiguous = probe(LuminaZImageLayout::PrefixedNative);
    ambiguous
        .tensor_shapes
        .extend(probe(LuminaZImageLayout::StandaloneNative).tensor_shapes);
    assert!(matches!(
        lumina::configuration_for_probe(&ambiguous),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
    let zimage = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("cap_embedder.1.weight".to_owned(), vec![3_840, 2_048]),
            ("noise_refiner.0.attention.k_norm.weight".to_owned(), vec![128]),
            ("noise_refiner.0.attention.qkv.weight".to_owned(), vec![11_520, 3_840]),
            ("x_embedder.weight".to_owned(), vec![3_840, 64]),
            ("final_layer.linear.weight".to_owned(), vec![64, 3_840]),
            ("layers.0.attention.qkv.weight".to_owned(), vec![11_520, 3_840]),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        lumina::configuration_for_probe(&zimage),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("cannot admit")
    ));
    support::assert_leaf_owner("lumina2_comfy_model_0107", "lumina_zimage_family")?;
    Ok(())
}

fn probe(layout: LuminaZImageLayout) -> ModelProbe {
    let prefix = match layout {
        LuminaZImageLayout::PrefixedNative => "model.diffusion_model.",
        LuminaZImageLayout::SavedModel => "model.",
        LuminaZImageLayout::StandaloneNative => "",
        LuminaZImageLayout::Diffusers => unreachable!(),
    };
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (format!("{prefix}cap_embedder.1.weight"), vec![2_304, 2_048]),
            (format!("{prefix}noise_refiner.0.attention.k_norm.weight"), vec![96]),
            (format!("{prefix}noise_refiner.0.attention.qkv.weight"), vec![3_840, 2_304]),
            (format!("{prefix}x_embedder.weight"), vec![2_304, 64]),
            (format!("{prefix}final_layer.linear.weight"), vec![64, 2_304]),
            (format!("{prefix}layers.0.attention.qkv.weight"), vec![3_840, 2_304]),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn mapping_keys(layout: LuminaZImageLayout) -> Vec<(String, Vec<u64>)> {
    let prefix = match layout {
        LuminaZImageLayout::PrefixedNative => "model.diffusion_model.",
        LuminaZImageLayout::SavedModel => "model.",
        LuminaZImageLayout::StandaloneNative => "",
        LuminaZImageLayout::Diffusers => unreachable!(),
    };
    [
        "cap_embedder.1.weight",
        "noise_refiner.0.attention.k_norm.weight",
        "noise_refiner.0.attention.qkv.weight",
        "x_embedder.weight",
        "final_layer.linear.weight",
        "layers.0.attention.qkv.weight",
    ]
    .into_iter()
    .map(|key| (format!("{prefix}{key}"), vec![2, 2]))
    .chain([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.gemma.weight".to_owned(), vec![1]),
    ])
    .collect()
}

pub(crate) mod support {
    use sim_model::{
        ModelFamilyDefinition, ModelFamilyError, ModelProbe, ModelStateTransaction,
        ModelStateTransformPlanDefinition, NativeFamilyBuildOptions, PatchGraph,
        build_model_family, describe_model_family, map_model_weights,
    };
    use sim_tensor::{
        CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor,
        TensorBackend,
        generated_sim_operator_indirection_01::{
            tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
        },
    };
    use sim_types::{CancellationToken, DeviceKind};
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, path::Path};

    #[derive(Deserialize)]
    struct FamilyFixture {
        fixture_id: String,
        feature_id: String,
        detector: ProbeFixture,
        base_artifact_digest: String,
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

    #[derive(Deserialize)]
    struct ProbeFixture {
        tensor_shapes: BTreeMap<String, Vec<u64>>,
        metadata: BTreeMap<String, String>,
    }

    #[derive(Deserialize)]
    struct TensorFixture {
        key: String,
        shape: Vec<u64>,
        values: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct CheckpointFixture {
        name: String,
        values: Vec<f32>,
    }

    pub(crate) fn validate_provenance(
        fixture_id: &str,
        feature_id: &str,
        identifier: &str,
        ordinal: u16,
        projection_sha256: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let repository = repository_root();
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
            fixture_directory(fixture_id).join("provenance.json"),
        )?)?;
        assert_eq!(value["feature_id"], feature_id);
        assert_eq!(value["source_symbol"], identifier);
        assert_eq!(value["source_ordinal"], ordinal);
        let projection = value["source_projection"]
            .as_str()
            .ok_or("source_projection must be a string")?;
        assert_eq!(sha256(projection.as_bytes()), projection_sha256);
        assert_eq!(value["source_projection_sha256"], projection_sha256);
        for source in value["source_files"]
            .as_array()
            .ok_or("source_files must be an array")?
        {
            let path = source["path"].as_str().ok_or("source path must be a string")?;
            let digest = source["sha256"].as_str().ok_or("source digest must be a string")?;
            assert_eq!(sha256(&std::fs::read(repository.join(path))?), digest, "{path}");
        }
        Ok(())
    }

    pub(crate) fn exercise_fixture(
        definition: &'static ModelFamilyDefinition,
        fixture_id: &str,
        module: &str,
        ordinal: u16,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fixture: FamilyFixture = serde_json::from_slice(&std::fs::read(
            fixture_directory(fixture_id).join("family.json"),
        )?)?;
        assert_eq!(fixture.fixture_id, fixture_id);
        assert_eq!(fixture.feature_id, definition.feature_id);
        assert_eq!(fixture.dtype, DType::F32);
        assert_eq!(fixture.device, DeviceKind::Cpu);
        let descriptor = describe_model_family(definition)?;
        assert_eq!(descriptor.family, definition.feature_id);
        let probe = ModelProbe {
            tensor_shapes: fixture.detector.tensor_shapes.clone(),
            metadata: fixture.detector.metadata.clone(),
        };
        assert!(sim_model::ModelFamilyRegistry::checked(sim_model::GENERATED_MODEL_FAMILIES)?
            .detect(&probe)?
            .identity
            .feature_id()
            == definition.feature_id);

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4 * 1024 * 1024)?,
            &cancellation,
        );
        let weights = fixture_weights(&backend, &context, &fixture, DType::F32)?;
        let mapped = map_model_weights(definition, &fixture.base_artifact_digest, weights)?;
        let options = NativeFamilyBuildOptions {
            dtype: DType::F32,
            device: DeviceKind::Cpu,
            activation_elements: fixture.activation_elements,
            memory_budget_bytes: fixture.memory_budget_bytes,
            allow_unexpected_weights: false,
        };
        let model = build_model_family(definition, mapped.clone(), options)?;
        assert_eq!(model.memory_estimate().total_bytes, fixture.expected_memory_bytes);
        let input = tensor(
            &backend,
            &context,
            &fixture.input.shape,
            &fixture.input.values,
            DType::F32,
        )?;
        assert_checkpoints(
            &backend,
            &context,
            &model.forward_checkpoints(&backend, &input, &context)?,
            &fixture.checkpoints,
        )?;
        let patched = PatchGraph::checked(&fixture.base_artifact_digest, fixture.patches.clone())?
            .apply(&backend, model.weights(), &context)?;
        assert_checkpoints(
            &backend,
            &context,
            &model.with_weights(patched)?.forward_checkpoints(&backend, &input, &context)?,
            &fixture.patched_checkpoints,
        )?;
        assert!(matches!(
            build_model_family(
                definition,
                mapped.clone(),
                NativeFamilyBuildOptions { memory_budget_bytes: fixture.expected_memory_bytes - 1, ..options }
            ),
            Err(ModelFamilyError::OutOfMemory { .. })
        ));
        assert!(matches!(
            build_model_family(
                definition,
                mapped.clone(),
                NativeFamilyBuildOptions { dtype: DType::F64, ..options }
            ),
            Err(ModelFamilyError::UnsupportedDType(DType::F64))
        ));
        assert!(matches!(
            build_model_family(
                definition,
                mapped,
                NativeFamilyBuildOptions { device: DeviceKind::Metal, ..options }
            ),
            Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
        ));

        for dtype in definition.supported_dtypes {
            let typed = fixture_weights(&backend, &context, &fixture, *dtype)?;
            let typed = map_model_weights(definition, &fixture.base_artifact_digest, typed)?;
            build_model_family(
                definition,
                typed,
                NativeFamilyBuildOptions { dtype: *dtype, ..options },
            )?;
        }

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(4 * 1024 * 1024)?,
            &cancelled,
        );
        assert!(model
            .forward_checkpoints(&backend, &input, &cancelled_context)
            .is_err());

        super::super::write_model_family_row_artifact(
            fixture_id,
            definition.feature_id,
            definition.identifier,
            ordinal,
            module,
            &[
                "source-provenance-registration-descriptor",
                "all-declared-layout-transactions",
                "named-forward-and-conditioning-checkpoints",
                "patch-application",
                "memory-oom-dtype-device-cancellation",
                "partial-ambiguous-invalid-probes",
                "canonical-owner-delegation",
            ],
        )?;
        Ok(())
    }

    pub(crate) fn exercise_plan(
        plan: &ModelStateTransformPlanDefinition,
        keys: &[(String, Vec<u64>)],
        expected: &[&str],
    ) -> Result<(), Box<dyn std::error::Error>> {
        exercise_compiled_plan(&plan.compile()?, keys, expected)
    }

    pub(crate) fn exercise_compiled_plan(
        plan: &sim_model::ModelStateTransformPlan,
        keys: &[(String, Vec<u64>)],
        expected: &[&str],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let source = keys
            .iter()
            .map(|(key, shape)| {
                let count = shape.iter().product::<u64>() as usize;
                Ok((
                    key.clone(),
                    tensor(&backend, &context, shape, &vec![1.0; count], DType::F32)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            plan,
            "0107010701070107010701070107010701070107010701070107010701070107",
            &source,
        )?;
        let model = mapped.component("model").ok_or("mapped model component is missing")?;
        for key in expected {
            assert!(model.contains_key(*key), "missing {key}");
        }
        Ok(())
    }

    pub(crate) fn assert_leaf_owner(
        module: &str,
        adapter: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let source = std::fs::read_to_string(
            repository_root().join(format!("crates/sim_model/src/families/{module}.rs")),
        )?;
        assert!(source.contains(adapter));
        for forbidden in ["pub struct ", "pub enum ", "unsafe ", "Command::new", "std::fs"] {
            assert!(!source.contains(forbidden), "row contains competing owner: {forbidden}");
        }
        Ok(())
    }

    fn fixture_weights(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        fixture: &FamilyFixture,
        dtype: DType,
    ) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
        fixture
            .source_weights
            .iter()
            .map(|weight| {
                Ok((
                    weight.key.clone(),
                    tensor(backend, context, &weight.shape, &weight.values, dtype)?,
                ))
            })
            .collect()
    }

    fn assert_checkpoints(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        actual: &[sim_model::ModelForwardCheckpoint],
        expected: &[CheckpointFixture],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(actual.len(), expected.len());
        for expected in expected {
            let actual = actual
                .iter()
                .find(|checkpoint| checkpoint.name == expected.name)
                .ok_or("checkpoint is missing")?;
            let values = tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
            assert_eq!(values.len(), expected.values.len(), "{}", expected.name);
            for (actual, expected_value) in values.iter().zip(&expected.values) {
                assert!(
                    (actual - expected_value).abs() <= 1.0e-5,
                    "{}: {actual} != {expected_value}",
                    expected.name
                );
            }
        }
        Ok(())
    }

    fn tensor(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        shape: &[u64],
        values: &[f32],
        dtype: DType,
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
