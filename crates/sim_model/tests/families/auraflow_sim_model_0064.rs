use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_auraflow_sim_model_0064 as aura,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        cast_to_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path};

const DIGEST: &str = "0640640640640640640640640640640640640640640640640640640640640640";

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9064",
    identifier: "AuraFlowAmbiguousFixture",
    ..aura::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    aura::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 86,
        source_architecture: "model_base.AuraFlowAmbiguousFixture",
        ..aura::MODEL_FAMILY_REGISTRATION
    },
];

#[derive(Debug, Deserialize)]
struct FamilyFixture {
    fixture_id: String,
    feature_id: String,
    checkpoints: Vec<CheckpointFixture>,
    patches: Vec<PatchOperation>,
    patched_checkpoints: Vec<CheckpointFixture>,
}

#[derive(Debug, Deserialize)]
struct CheckpointFixture {
    name: String,
    values: Vec<f32>,
}

fn native_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "model.diffusion_model.cond_seq_linear.weight".to_owned(),
                vec![2, 2_048],
            ),
            (
                "model.diffusion_model.positional_encoding".to_owned(),
                vec![1, 4, 2],
            ),
            (
                "model.diffusion_model.double_layers.0.attn.w1q.weight".to_owned(),
                vec![2, 2],
            ),
            (
                "model.diffusion_model.single_layers.0.attn.w1q.weight".to_owned(),
                vec![2, 2],
            ),
            (
                "model.diffusion_model.final_linear.weight".to_owned(),
                vec![2, 2],
            ),
        ]),
        metadata: BTreeMap::from([
            ("cond_seq_dim".to_owned(), "4096".to_owned()),
            ("model_layout".to_owned(), "diffusers".to_owned()),
        ]),
    }
}

fn diffusers_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("context_embedder.weight".to_owned(), vec![2, 2_048]),
            ("pos_embed.pos_embed".to_owned(), vec![1, 4, 2]),
            (
                "joint_transformer_blocks.0.attn.add_q_proj.weight".to_owned(),
                vec![2, 2],
            ),
            (
                "joint_transformer_blocks.0.attn.add_k_proj.weight".to_owned(),
                vec![2, 2],
            ),
            (
                "single_transformer_blocks.0.attn.to_q.weight".to_owned(),
                vec![2, 2],
            ),
            ("proj_out.weight".to_owned(), vec![2, 2]),
        ]),
        metadata: BTreeMap::from([
            ("cond_seq_dim".to_owned(), "4096".to_owned()),
            ("model_layout".to_owned(), "native".to_owned()),
        ]),
    }
}

#[test]
fn val_model_family_row_001_auraflow_source_bound_and_profiled_for_both_layouts()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(aura::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0064");
    assert_eq!(aura::MODEL_FAMILY_IDENTIFIER, "AuraFlow");
    assert_eq!(aura::MODEL_FAMILY_REGISTRATION.source_ordinal, 22);
    assert_eq!(aura::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.AuraFlow");

    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(fs::read(repository_root.join(aura::MODEL_FAMILY_SOURCE_PATH))?)
        ),
        aura::MODEL_FAMILY_SOURCE_SHA256
    );
    let catalog: serde_json::Value = serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let projection = catalog["models"]
        .as_array()
        .ok_or("model catalog has no rows")?
        .iter()
        .find(|row| row["feature_id"] == aura::MODEL_FAMILY_FEATURE_ID)
        .ok_or("AuraFlow catalog row missing")?;
    assert_eq!(projection["source_ordinal"].as_u64(), Some(22));
    assert_eq!(projection["source_symbol"], "AuraFlow");
    assert_eq!(projection["static"]["unet_config"]["value"]["cond_seq_dim"], 2_048);
    assert_eq!(
        format!("{:x}", Sha256::digest(serde_json::to_vec(projection)?)),
        aura::MODEL_FAMILY_PROJECTION_SHA256
    );
    verify_provenance(&repository_root)?;

    let descriptor = describe_model_family(&aura::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    let registry = ModelFamilyRegistry::checked_registrations(&[
        aura::MODEL_FAMILY_REGISTRATION,
    ])?;
    let resolved = registry.resolve(&native_probe())?;
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(
        resolved.clip_target().candidates()[0]
            .tokenizer()
            .identifier(),
        "sim.text_encoders.aura_t5.AuraT5Tokenizer"
    );

    let native_configuration = aura::configuration_for_probe(&native_probe())?;
    assert_eq!(native_configuration.layout, sim_model::ModelStateLayout::PrefixedNative);
    assert_eq!(native_configuration.maximum_sequence_length, 4);
    assert_eq!(native_configuration.conditioning_dimension, 2_048);
    assert_eq!(native_configuration.double_layer_count, 1);
    assert_eq!(native_configuration.layer_count, 2);
    let diffusers_configuration = aura::configuration_for_probe(&diffusers_probe())?;
    assert_eq!(
        diffusers_configuration.layout,
        sim_model::ModelStateLayout::Diffusers
    );
    assert_eq!(diffusers_configuration, AuraFlowConfigurationFixture::expected_diffusers());

    let stored_native = probe_through_model_store("native", &native_probe())?;
    assert_eq!(stored_native.tensor_shapes(), native_probe().tensor_shapes());
    assert_eq!(
        registry.resolve(&stored_native)?.detection().identity.feature_id(),
        aura::MODEL_FAMILY_FEATURE_ID
    );
    let stored_diffusers = probe_through_model_store("diffusers", &diffusers_probe())?;
    assert_eq!(stored_diffusers.tensor_shapes(), diffusers_probe().tensor_shapes());
    assert_eq!(
        aura::configuration_for_probe(&stored_diffusers)?.layout,
        sim_model::ModelStateLayout::Diffusers
    );
    verify_owner_delegation()?;
    super::write_model_family_row_artifact(
        aura::MODEL_FAMILY_FIXTURE,
        aura::MODEL_FAMILY_FEATURE_ID,
        aura::MODEL_FAMILY_IDENTIFIER,
        aura::MODEL_FAMILY_REGISTRATION.source_ordinal,
        "auraflow_comfy_model_0064",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-and-diffusers-detection",
            "exact-source-configuration-and-profile",
            "transactional-component-mapping",
            "named-forward-checkpoints-and-patching",
            "memory-oom-dtype-device-cancellation",
            "partial-malformed-and-owner-delegation",
        ],
    )?;
    Ok(())
}

struct AuraFlowConfigurationFixture;

impl AuraFlowConfigurationFixture {
    fn expected_diffusers() -> aura::AuraFlowConfiguration {
        aura::AuraFlowConfiguration {
            layout: sim_model::ModelStateLayout::Diffusers,
            maximum_sequence_length: 4,
            conditioning_dimension: 2_048,
            double_layer_count: 1,
            layer_count: 2,
        }
    }
}

#[test]
fn val_model_family_row_001_auraflow_mapping_forward_patch_and_failures_are_native()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        aura::MODEL_FAMILY_REGISTRATION,
    ])?;
    let probe = native_probe();
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 22);

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(256 * 1024)?,
        &cancellation,
    );
    let source = native_source(&backend, DType::F32, &context)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &source,
    )?;
    let options = NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes: 8_236,
        allow_unexpected_weights: false,
    };
    let model = build_model_family_for_probe(&registry, &probe, weights.clone(), options)?;
    assert_eq!(model.memory_estimate().total_bytes, 8_236);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    let fixture = fixture()?;
    assert_eq!(fixture.fixture_id, aura::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture.feature_id, aura::MODEL_FAMILY_FEATURE_ID);
    assert_checkpoints(&backend, &context, &checkpoints, &fixture.checkpoints)?;

    let patch = PatchGraph::checked(
        DIGEST,
        vec![PatchOperation {
            identifier: "auraflow-single-stream-attention-delta".to_owned(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.single_layers.0.attn.w1q.weight".to_owned(),
                expected_shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
                application: PatchApplication::Add,
            }],
        }],
    )?;
    let patched = model.with_weights(patch.apply(&backend, &weights, &context)?)?;
    let patched_checkpoints = patched.forward_checkpoints(&backend, &input, &context)?;
    assert_eq!(fixture.patches.len(), 1);
    assert_checkpoints(
        &backend,
        &context,
        &patched_checkpoints,
        &fixture.patched_checkpoints,
    )?;

    let replace_then_add = ordered_patch_graph(true)?;
    let add_then_replace = ordered_patch_graph(false)?;
    assert_ne!(
        replace_then_add.identity().ordered_digest,
        add_then_replace.identity().ordered_digest
    );
    let replace_then_add_weights = replace_then_add.apply(&backend, &weights, &context)?;
    let add_then_replace_weights = add_then_replace.apply(&backend, &weights, &context)?;
    let first = replace_then_add_weights
        .tensors()
        .get("native.single_layers.0.attn.w1q.weight")
        .ok_or("ordered patch target missing")?;
    let second = add_then_replace_weights
        .tensors()
        .get("native.single_layers.0.attn.w1q.weight")
        .ok_or("reverse ordered patch target missing")?;
    assert_ne!(
        tensor_to_f32_with_context_exact_native(&backend, first, &context)?,
        tensor_to_f32_with_context_exact_native(&backend, second, &context)?
    );

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let typed_source = native_source(&backend, dtype, &context)?;
        let typed_weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            DIGEST,
            &typed_source,
        )?;
        assert!(build_model_family_for_probe(
            &registry,
            &probe,
            typed_weights,
            NativeFamilyBuildOptions { dtype, ..options },
        )
        .is_ok());
    }

    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe,
            weights.clone(),
            NativeFamilyBuildOptions { memory_budget_bytes: 8_235, ..options },
        ),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe,
            weights,
            NativeFamilyBuildOptions { device: DeviceKind::Cuda, ..options },
        ),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Cuda))
    ));
    let f64_source = native_source(&backend, DType::F32, &context)?;
    let f64_weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &f64_source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(
            &registry,
            &probe,
            f64_weights,
            NativeFamilyBuildOptions { dtype: DType::F64, ..options },
        ),
        Err(ModelFamilyError::UnsupportedDType(DType::F64))
    ));

    let mut wrong_dimension = probe.clone();
    wrong_dimension.tensor_shapes.insert(
        "model.diffusion_model.cond_seq_linear.weight".to_owned(),
        vec![2, 2_047],
    );
    assert!(matches!(
        registry.resolve(&wrong_dimension),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("expected 2048")
    ));
    let mut partial = probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.single_layers.0.attn.w1q.weight");
    assert!(registry.resolve(&partial).is_err());
    let mut wrong_metadata = probe.clone();
    wrong_metadata.metadata.clear();
    wrong_metadata
        .metadata
        .insert("cond_seq_dim".to_owned(), "1".to_owned());
    wrong_metadata
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert_eq!(
        registry.resolve(&wrong_metadata)?.detection().identity.feature_id(),
        aura::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    cancellation.cancel();
    assert!(resolved
        .map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context),
            DIGEST,
            &source,
        )
        .is_err());
    Ok(())
}

#[test]
fn val_model_family_row_001_auraflow_native_and_diffusers_plans_route_canonical_components()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        aura::MODEL_FAMILY_REGISTRATION,
    ])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(256 * 1024)?,
        &cancellation,
    );

    let mut component_probe = native_probe();
    component_probe
        .tensor_shapes
        .insert("vae.decoder.weight".to_owned(), vec![1]);
    component_probe
        .tensor_shapes
        .insert("text_encoders.t5.weight".to_owned(), vec![1]);
    let mut component_source = native_source(&backend, DType::F32, &context)?;
    component_source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(&backend, &[1], &[5.0], DType::F32, &context)?,
    );
    component_source.insert(
        "text_encoders.t5.weight".to_owned(),
        tensor(&backend, &[1], &[6.0], DType::F32, &context)?,
    );
    let mapped = registry.resolve(&component_probe)?.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &component_source,
    )?;
    assert!(mapped.components()["denoiser"].contains_key("native.final_linear.weight"));
    assert!(mapped.components()["vae"].contains_key("vae.decoder.weight"));
    assert!(mapped.components()["text_encoder"].contains_key("text_encoder.t5.weight"));

    let diffusers = diffusers_probe();
    let diffusers_weights = registry.resolve(&diffusers)?.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &diffusers_source(&backend, DType::F32, &context)?,
    )?;
    assert_eq!(
        diffusers_weights
            .tensors()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "native.cond_seq_linear.weight",
            "native.double_layers.0.attn.w1k.weight",
            "native.double_layers.0.attn.w1q.weight",
            "native.final_linear.weight",
            "native.positional_encoding",
            "native.single_layers.0.attn.w1q.weight",
        ]
    );
    Ok(())
}

fn native_source(
    backend: &CpuBackend,
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, sim_tensor::Tensor>, Box<dyn std::error::Error>> {
    let conditioning_values = vec![0.0; 2 * 2_048];
    Ok(BTreeMap::from([
        (
            "model.diffusion_model.cond_seq_linear.weight".to_owned(),
            tensor(
                backend,
                &[2, 2_048],
                &conditioning_values,
                dtype,
                context,
            )?,
        ),
        (
            "model.diffusion_model.positional_encoding".to_owned(),
            tensor(
                backend,
                &[1, 4, 2],
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype,
                context,
            )?,
        ),
        (
            "model.diffusion_model.double_layers.0.attn.w1q.weight".to_owned(),
            tensor(backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], dtype, context)?,
        ),
        (
            "model.diffusion_model.single_layers.0.attn.w1q.weight".to_owned(),
            tensor(backend, &[2, 2], &[2.0, 0.0, 0.0, 0.5], dtype, context)?,
        ),
        (
            "model.diffusion_model.final_linear.weight".to_owned(),
            tensor(backend, &[2, 2], &[1.0, 1.0, 1.0, -1.0], dtype, context)?,
        ),
    ]))
}

fn diffusers_source(
    backend: &CpuBackend,
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<BTreeMap<String, sim_tensor::Tensor>, Box<dyn std::error::Error>> {
    let conditioning_values = vec![0.0; 2 * 2_048];
    Ok(BTreeMap::from([
        (
            "context_embedder.weight".to_owned(),
            tensor(
                backend,
                &[2, 2_048],
                &conditioning_values,
                dtype,
                context,
            )?,
        ),
        (
            "pos_embed.pos_embed".to_owned(),
            tensor(
                backend,
                &[1, 4, 2],
                &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype,
                context,
            )?,
        ),
        (
            "joint_transformer_blocks.0.attn.add_q_proj.weight".to_owned(),
            tensor(backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0], dtype, context)?,
        ),
        (
            "joint_transformer_blocks.0.attn.add_k_proj.weight".to_owned(),
            tensor(backend, &[2, 2], &[0.5, 0.0, 0.0, 0.5], dtype, context)?,
        ),
        (
            "single_transformer_blocks.0.attn.to_q.weight".to_owned(),
            tensor(backend, &[2, 2], &[2.0, 0.0, 0.0, 0.5], dtype, context)?,
        ),
        (
            "proj_out.weight".to_owned(),
            tensor(backend, &[2, 2], &[1.0, 1.0, 1.0, -1.0], dtype, context)?,
        ),
    ]))
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let tensor = backend.upload_f32(descriptor, values, context)?.0;
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

fn assert_checkpoints(
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    actual: &[sim_model::ModelForwardCheckpoint],
    expected: &[CheckpointFixture],
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.name, expected.name);
        let values = tensor_to_f32_with_context_exact_native(backend, &actual.tensor, context)?;
        assert_eq!(values.len(), expected.values.len());
        for (actual, expected) in values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() <= 1.0e-5, "{actual} != {expected}");
        }
    }
    Ok(())
}

fn ordered_patch_graph(replace_first: bool) -> Result<PatchGraph, Box<dyn std::error::Error>> {
    let replacement = PatchOperation {
        identifier: "auraflow-ordered-replacement".to_owned(),
        kind: PatchKind::Replacement,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.single_layers.0.attn.w1q.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Replace,
        }],
    };
    let addition = PatchOperation {
        identifier: "auraflow-ordered-addition".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.single_layers.0.attn.w1q.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![1.0, 0.0, 0.0, 1.0],
            application: PatchApplication::Add,
        }],
    };
    let operations = if replace_first {
        vec![replacement, addition]
    } else {
        vec![addition, replacement]
    };
    Ok(PatchGraph::checked(DIGEST, operations)?)
}

fn fixture() -> Result<FamilyFixture, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../sim_test_support/fixtures/models/auraflow-comfy-model-0064/family.json",
        ),
    )?)?)
}

fn verify_provenance(repository_root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let provenance: serde_json::Value = serde_json::from_slice(&fs::read(repository_root.join(
        "crates/sim_test_support/fixtures/models/auraflow-comfy-model-0064/provenance.json",
    ))?)?;
    assert_eq!(provenance["feature_id"], aura::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], aura::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"].as_u64(), Some(22));
    let projection = provenance["source_projection"]
        .as_str()
        .ok_or("AuraFlow provenance projection missing")?;
    assert_eq!(
        format!("{:x}", Sha256::digest(projection.as_bytes())),
        provenance["source_projection_sha256"]
    );
    assert!(provenance["oracle"]
        .as_str()
        .is_some_and(|oracle| oracle.contains("no production Python dependency")));
    for source in provenance["source_files"]
        .as_array()
        .ok_or("AuraFlow provenance sources missing")?
    {
        let path = source["path"].as_str().ok_or("source path missing")?;
        let sha256 = source["sha256"].as_str().ok_or("source digest missing")?;
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(repository_root.join(path))?)),
            sha256
        );
    }
    Ok(())
}

fn probe_through_model_store(
    name: &str,
    expected: &ModelProbe,
) -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let file_name = format!("auraflow-{name}.safetensors");
    let model_path = directory.path().join(&file_name);
    write_safetensors(&model_path, expected)?;
    let root_identifier = format!("auraflow-{name}-row");
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        &root_identifier,
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new(&root_identifier, &file_name)?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path, probe: &ModelProbe) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, shape) in &probe.tensor_shapes {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total.checked_mul(*dimension).ok_or("probe tensor shape overflow")
        })?;
        data.resize(
            data.len()
                .checked_add(usize::try_from(elements)?.checked_mul(4).ok_or("byte overflow")?)
                .ok_or("data length overflow")?,
            0,
        );
        header.insert(
            key.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    header.insert(
        "__metadata__".to_owned(),
        serde_json::to_value(&probe.metadata)?,
    );
    let header = serde_json::to_vec(&header)?;
    let mut file = fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn verify_owner_delegation() -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/auraflow_sim_model_0064.rs"),
    )?;
    for canonical_owner in [
        "ModelFamilyRegistration",
        "ModelFamilyStatePlanSelector",
        "ModelStateTransformPlanDefinition",
        "ModelProbe",
        "ModelForwardOperation",
        "MemoryEstimatorDescriptor",
    ] {
        assert!(source.contains(canonical_owner));
    }
    for duplicate_owner in [
        "struct CancellationToken",
        "struct Tensor",
        "struct ModelStore",
        "struct PatchGraph",
        "struct ArtifactIndex",
        "std::process::Command",
        "unsafe ",
    ] {
        assert!(!source.contains(duplicate_owner));
    }
    Ok(())
}
