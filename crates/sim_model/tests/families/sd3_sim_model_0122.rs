use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction, ModelStore,
    NativeFamilyBuildOptions, ParserLimits, PatchApplication, PatchGraph, PatchKind,
    PatchOperation, PatchTarget, build_model_family_for_probe, describe_model_family,
    generated_sd3_sim_model_0122 as sd3,
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

const DIGEST: &str = "0122012201220122012201220122012201220122012201220122012201220122";
const MEMORY_BYTES: u64 = 16_460;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9122",
    identifier: "SD3_AmbiguousFixture",
    ..sd3::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sd3::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 119,
        source_architecture: "model_base.SD3_AmbiguousFixture",
        ..sd3::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sd3_source_configuration_dynamic_clip_provenance_and_store()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(sd3::MODEL_FAMILY_IDENTIFIER, "SD3");
    assert_eq!(sd3::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0122");
    assert_eq!(sd3::MODEL_FAMILY_SOURCE_ORDINAL, 19);
    assert_eq!(sd3::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.SD3");
    assert_eq!(sd3::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.6);
    assert_eq!(sd3::MODEL_FAMILY_SAMPLING_SHIFT, 3.0);
    let descriptor = describe_model_family(&sd3::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SD3");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);

    let registry = ModelFamilyRegistry::checked_registrations(&[sd3::MODEL_FAMILY_REGISTRATION])?;
    for (probe, layout) in [
        (native_probe(true), sim_model::ModelStateLayout::PrefixedNative),
        (diffusers_probe(), sim_model::ModelStateLayout::Diffusers),
    ] {
        let configuration = sd3::configuration_for_probe(&probe)?;
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, 16);
        assert_eq!(configuration.patch_size, 2);
        assert_eq!(configuration.hidden_size, 128);
        assert_eq!(configuration.attention_head_count, 2);
        assert_eq!(configuration.block_count, 1);
        assert_eq!(registry.detect(&probe)?.identity.feature_id(), sd3::MODEL_FAMILY_FEATURE_ID);
    }
    let native_configuration = sd3::configuration_for_probe(&native_probe(true))?;
    assert!(native_configuration.clip_l);
    assert!(native_configuration.clip_g);
    assert!(native_configuration.t5xxl);
    let resolved = registry.resolve(&native_probe(true))?;
    assert!(resolved.clip_target().dynamic_selection());
    let candidates = resolved.clip_target().candidates();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].tokenizer().identifier(), "sim.text_encoders.sd3_clip.SD3Tokenizer");
    assert_eq!(candidates[0].clip_model().target().as_str(), "sim.text_encoders.sd3_clip.sd3_clip");
    let clip_json = serde_json::to_string(resolved.clip_target())?;
    for fact in ["clip_l", "clip_g", "t5", "t5_xxl_detect"] {
        assert!(clip_json.contains(fact));
    }
    let stored = probe_through_model_store()?;
    assert_eq!(registry.detect(&stored)?.identity.feature_id(), sd3::MODEL_FAMILY_FEATURE_ID);

    let mut misleading = native_probe(false);
    misleading.metadata.insert("model_layout".into(), "diffusers".into());
    misleading.metadata.insert("model_family".into(), "AuraFlow".into());
    assert_eq!(registry.detect(&misleading)?.identity.feature_id(), sd3::MODEL_FAMILY_FEATURE_ID);
    let mut partial = native_probe(false);
    partial.tensor_shapes.remove("model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight");
    assert!(matches!(registry.detect(&partial), Err(ModelFamilyError::NoDetectionMatch)));
    let mut malformed = native_probe(false);
    malformed.tensor_shapes.get_mut("model.diffusion_model.x_embedder.proj.weight").ok_or("input")?[0] = 127;
    assert!(matches!(
        sd3::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("heads*64")
    ));

    verify_provenance_and_catalog()?;
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/sd3_sim_model_0122.rs"),
    )?;
    for forbidden in ["struct Tensor", "struct ModelStore", "struct ModelProbe", "struct PatchGraph", "std::fs", "unsafe "] {
        assert!(!source.contains(forbidden));
    }
    super::write_model_family_row_artifact(
        sd3::MODEL_FAMILY_FIXTURE,
        sd3::MODEL_FAMILY_FEATURE_ID,
        sd3::MODEL_FAMILY_IDENTIFIER,
        sd3::MODEL_FAMILY_SOURCE_ORDINAL,
        "sd3_comfy_model_0122",
        &[
            "source-provenance-registration-descriptor",
            "model-store-native-and-diffusers-detection",
            "source-exact-mmdit-configuration-and-dynamic-clip",
            "transactional-qkv-assembly-and-component-routing",
            "named-forward-conditioning-checkpoints-and-patching",
            "memory-oom-dtype-device-cancellation",
            "partial-malformed-ambiguous-unexpected-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sd3_diffusers_qkv_assembly_is_exact_and_transactional()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[sd3::MODEL_FAMILY_REGISTRATION])?;
    let probe = diffusers_probe();
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let source = diffusers_source(&backend, &context, DType::F32)?;
    let mapped = resolved.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    let denoiser = mapped.component("denoiser").ok_or("denoiser")?;
    assert_values(
        &backend,
        &context,
        &denoiser["native.joint_blocks.0.x_block.attn.qkv.weight"],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )?;
    assert_values(
        &backend,
        &context,
        &denoiser["native.joint_blocks.0.context_block.attn.qkv.weight"],
        &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    )?;

    let mut unexpected_probe = diffusers_probe();
    unexpected_probe.tensor_shapes.insert("legacy.javascript.extension".into(), vec![1]);
    let mut unexpected_source = source.clone();
    unexpected_source.insert(
        "legacy.javascript.extension".into(),
        tensor(&backend, &[1], &[1.0], DType::F32, &context)?,
    );
    assert!(matches!(
        registry.resolve(&unexpected_probe)?.map_state_dictionary(
            &ModelStateTransaction::new(&backend, &context), DIGEST, &unexpected_source,
        ),
        Err(ModelFamilyError::UnexpectedKeys(keys)) if keys == ["legacy.javascript.extension"]
    ));
    let mut partial_probe = diffusers_probe();
    partial_probe.tensor_shapes.remove("transformer_blocks.0.attn.to_v.weight");
    assert!(matches!(
        registry.resolve(&partial_probe),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        resolved.map_state_dictionary(&ModelStateTransaction::new(&backend, &cancelled_context), DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_sd3_forward_patch_memory_dtype_device_and_ambiguity()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[sd3::MODEL_FAMILY_REGISTRATION])?;
    let probe = native_probe(false);
    let resolved = registry.resolve(&probe)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancellation,
    );
    let source = native_source(&backend, &context, DType::F32, false)?;
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    let model = build_model_family_for_probe(
        &registry, &probe, weights, options(DType::F32, MEMORY_BYTES),
    )?;
    assert_eq!(model.memory_estimate().total_bytes, MEMORY_BYTES);
    let input = tensor(&backend, &[1, 2], &[1.0, 2.0], DType::F32, &context)?;
    assert_checkpoint(
        &backend,
        &context,
        &model.forward_checkpoints(&backend, &input, &context)?,
        "sd3_flow_prediction",
        &[0.9950548, 0.7615942],
    )?;
    let patch = PatchGraph::checked(
        DIGEST,
        vec![PatchOperation {
            identifier: "sd3-image-attention-delta".into(),
            kind: PatchKind::Lora,
            scale: 1.0,
            targets: vec![PatchTarget {
                key: "native.joint_blocks.0.x_block.attn.proj.weight".into(),
                expected_shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
                application: PatchApplication::Add,
            }],
        }],
    )?;
    let patched = model.with_weights(patch.apply(&backend, model.weights(), &context)?)?;
    assert_checkpoint(
        &backend,
        &context,
        &patched.forward_checkpoints(&backend, &input, &context)?,
        "sd3_flow_prediction",
        &[0.9999877, 0.0],
    )?;

    for dtype in [DType::F16, DType::Bf16] {
        let source = native_source(&backend, &context, dtype, false)?;
        let weights = resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
        )?;
        build_model_family_for_probe(&registry, &probe, weights, options(dtype, MEMORY_BYTES))?;
    }
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, options(DType::F32, MEMORY_BYTES - 1)),
        Err(ModelFamilyError::OutOfMemory { required: MEMORY_BYTES, .. })
    ));
    let weights = resolved.map_primary_weights(
        &ModelStateTransaction::new(&backend, &context), DIGEST, &source,
    )?;
    let mut metal = options(DType::F32, MEMORY_BYTES);
    metal.device = DeviceKind::Metal;
    assert!(matches!(
        build_model_family_for_probe(&registry, &probe, weights, metal),
        Err(ModelFamilyError::UnsupportedDevice(DeviceKind::Metal))
    ));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { .. })
    ));
    Ok(())
}

fn native_probe(include_text: bool) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        ("model.diffusion_model.x_embedder.proj.weight".into(), vec![128, 16, 2, 2]),
        ("model.diffusion_model.t_embedder.mlp.0.weight".into(), vec![2, 2]),
        ("model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight".into(), vec![6, 2]),
        ("model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight".into(), vec![6, 2]),
        ("model.diffusion_model.joint_blocks.0.x_block.attn.proj.weight".into(), vec![2, 2]),
        ("model.diffusion_model.final_layer.linear.weight".into(), vec![2, 2]),
    ]);
    if include_text {
        tensor_shapes.insert("text_encoders.clip_l.transformer.text_model.final_layer_norm.weight".into(), vec![2]);
        tensor_shapes.insert("text_encoders.clip_g.transformer.text_model.final_layer_norm.weight".into(), vec![2]);
        tensor_shapes.insert("text_encoders.t5xxl.transformer.encoder.final_layer_norm.weight".into(), vec![2]);
    }
    ModelProbe { tensor_shapes, metadata: BTreeMap::new() }
}

fn diffusers_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("pos_embed.proj.weight".into(), vec![128, 16, 2, 2]),
            ("time_text_embed.timestep_embedder.linear_1.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.to_q.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.to_k.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.to_v.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.add_q_proj.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.add_k_proj.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.add_v_proj.weight".into(), vec![2, 2]),
            ("transformer_blocks.0.attn.to_out.0.weight".into(), vec![2, 2]),
            ("proj_out.weight".into(), vec![2, 2]),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn native_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    include_text: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    native_probe(include_text).tensor_shapes.into_iter().map(|(key, shape)| {
        let count = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key.ends_with("t_embedder.mlp.0.weight") {
            vec![1.0, 0.0, 0.0, 1.0]
        } else if key.ends_with("x_block.attn.proj.weight") {
            vec![2.0, 0.0, 0.0, 0.5]
        } else if key.ends_with("final_layer.linear.weight") {
            vec![1.0, 1.0, 1.0, -1.0]
        } else {
            vec![0.0; count]
        };
        Ok((key, tensor(backend, &shape, &values, dtype, context)?))
    }).collect()
}

fn diffusers_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let qkv_values = BTreeMap::from([
        ("transformer_blocks.0.attn.to_q.weight", vec![1.0, 2.0, 3.0, 4.0]),
        ("transformer_blocks.0.attn.to_k.weight", vec![5.0, 6.0, 7.0, 8.0]),
        ("transformer_blocks.0.attn.to_v.weight", vec![9.0, 10.0, 11.0, 12.0]),
        ("transformer_blocks.0.attn.add_q_proj.weight", vec![13.0, 14.0, 15.0, 16.0]),
        ("transformer_blocks.0.attn.add_k_proj.weight", vec![17.0, 18.0, 19.0, 20.0]),
        ("transformer_blocks.0.attn.add_v_proj.weight", vec![21.0, 22.0, 23.0, 24.0]),
    ]);
    diffusers_probe().tensor_shapes.into_iter().map(|(key, shape)| {
        let count = usize::try_from(shape.iter().product::<u64>())?;
        let values = qkv_values.get(key.as_str()).cloned().unwrap_or_else(|| {
            if key.ends_with("timestep_embedder.linear_1.weight") {
                vec![1.0, 0.0, 0.0, 1.0]
            } else if key.ends_with("to_out.0.weight") {
                vec![2.0, 0.0, 0.0, 0.5]
            } else if key == "proj_out.weight" {
                vec![1.0, 1.0, 1.0, -1.0]
            } else {
                vec![0.0; count]
            }
        });
        Ok((key, tensor(backend, &shape, &values, dtype, context)?))
    }).collect()
}

fn probe_through_model_store() -> Result<ModelProbe, Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("sd3.safetensors");
    write_safetensors(&path, &native_probe(false))?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical("sd3-row", "checkpoints", directory.path(), ["safetensors"])?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("sd3-row", "sd3.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    Ok(store.family_probe(&loaded, &cancellation)?)
}

fn write_safetensors(path: &Path, probe: &ModelProbe) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, shape) in &probe.tensor_shapes {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |a, b| a.checked_mul(*b).ok_or("overflow"))?;
        data.resize(data.len() + usize::try_from(elements)?.checked_mul(4).ok_or("overflow")?, 0);
        header.insert(key.clone(), serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}));
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
    Ok(tensor_from_f32_with_context_exact_native(
        backend, shape, values, dtype, backend.device(), context,
    )?)
}

fn options(dtype: DType, memory_budget_bytes: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype,
        device: DeviceKind::Cpu,
        activation_elements: 2,
        memory_budget_bytes,
        allow_unexpected_weights: false,
    }
}

fn assert_values(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    tensor: &Tensor,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = tensor_to_f32_with_context_exact_native(backend, tensor, context)?;
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn assert_checkpoint(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    checkpoints: &[sim_model::ModelForwardCheckpoint],
    name: &str,
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = &checkpoints.iter().find(|checkpoint| checkpoint.name == name).ok_or("checkpoint")?.tensor;
    assert_values(backend, context, tensor, expected)
}

fn verify_provenance_and_catalog() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.join("crates/sim_test_support/fixtures/models")
            .join(sd3::MODEL_FAMILY_FIXTURE)
            .join("provenance.json"),
    )?)?;
    assert_eq!(sha256(provenance["source_projection"].as_str().ok_or("projection")?.as_bytes()), sd3::MODEL_FAMILY_PROJECTION_SHA256);
    for source in provenance["source_files"].as_array().ok_or("sources")? {
        assert_eq!(sha256(&std::fs::read(root.join(source["path"].as_str().ok_or("path")?))?), source["sha256"]);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("crates/sim_model/catalog/model-families-v1.json"))?)?;
    let row = catalog["models"].as_array().ok_or("models")?.iter().find(|row| row["feature_id"] == sd3::MODEL_FAMILY_FEATURE_ID).ok_or("row")?;
    assert_eq!(row["source_ordinal"], 19);
    assert_eq!(row["static"]["unet_config"]["value"]["in_channels"], 16);
    assert_eq!(row["static"]["memory_usage_factor"]["value"], 1.6);
    assert!(row["clip_target"]["has_dynamic_control_flow"].as_bool().ok_or("dynamic")?);
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
