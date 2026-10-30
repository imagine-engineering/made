use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelProbe, ModelStateTransaction, NativeFamilyBuildOptions, build_model_family_for_probe,
    describe_model_family, generated_sd15_sim_model_0117 as sd15,
    generated_sd15_instructpix2pix_sim_model_0118 as instruct,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

const DIGEST: &str = "0118011801180118011801180118011801180118011801180118011801180118";
const MEMORY_BYTES: u64 = 72_220;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9118",
    identifier: "SD15_instructpix2pix_AmbiguousFixture",
    ..instruct::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    instruct::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 102,
        source_architecture: "model_base.SD15_instructpix2pix_AmbiguousFixture",
        ..instruct::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sd15_instructpix2pix_inheritance_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(instruct::MODEL_FAMILY_IDENTIFIER, "SD15_instructpix2pix");
    assert_eq!(instruct::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0118");
    assert_eq!(instruct::MODEL_FAMILY_SOURCE_ORDINAL, 2);
    assert_eq!(instruct::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.SD15_instructpix2pix");
    assert_eq!(instruct::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.0);
    assert_eq!(instruct::MODEL_FAMILY.latent_feature_id, sd15::MODEL_FAMILY.latent_feature_id);
    assert_eq!(instruct::MODEL_FAMILY.clip_target, sd15::MODEL_FAMILY.clip_target);
    assert_eq!(instruct::MODEL_FAMILY.forward_program, sd15::MODEL_FAMILY.forward_program);
    let descriptor = describe_model_family(&instruct::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SD15");
    assert_eq!(descriptor.supported_dtypes, ["float16", "bfloat16", "float32"]);

    let registry = combined_registry()?;
    for (probe, layout) in [
        (native_probe(false), sim_model::ModelStateLayout::PrefixedNative),
        (diffusers_probe(), sim_model::ModelStateLayout::Diffusers),
    ] {
        assert_eq!(registry.detect(&probe)?.identity.feature_id(), instruct::MODEL_FAMILY_FEATURE_ID);
        let configuration = instruct::configuration_for_probe(&probe)?;
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, 8);
        assert_eq!(configuration.model_channels, 320);
        assert_eq!(configuration.context_dimension, 768);
    }
    assert_eq!(registry.detect(&sd15_probe())?.identity.feature_id(), sd15::MODEL_FAMILY_FEATURE_ID);
    let resolved = registry.resolve(&native_probe(false))?;
    assert_eq!(resolved.clip_target().candidates()[0].tokenizer().identifier(), "sd1_clip.SD1Tokenizer");
    assert_eq!(resolved.clip_target().candidates()[0].clip_model().target().as_str(), "sd1_clip.SD1ClipModel");

    let mut malformed = native_probe(false);
    malformed.tensor_shapes.get_mut("model.diffusion_model.input_blocks.0.0.weight").ok_or("input")?[1] = 4;
    assert!(matches!(
        instruct::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("expected [320, 8]")
    ));
    let mut partial = native_probe(false);
    partial.tensor_shapes.remove("model.diffusion_model.input_blocks.0.0.weight");
    assert!(matches!(registry.detect(&partial), Err(ModelFamilyError::NoDetectionMatch)));

    verify_provenance_and_catalog()?;
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/sd15_instructpix2pix_sim_model_0118.rs"),
    )?;
    for forbidden in ["struct Tensor", "struct ModelStore", "struct ModelProbe", "struct PatchGraph", "std::fs", "unsafe "] {
        assert!(!source.contains(forbidden));
    }
    super::write_model_family_row_artifact(
        instruct::MODEL_FAMILY_FIXTURE,
        instruct::MODEL_FAMILY_FEATURE_ID,
        instruct::MODEL_FAMILY_IDENTIFIER,
        instruct::MODEL_FAMILY_SOURCE_ORDINAL,
        "sd15_instructpix2pix_comfy_model_0118",
        &[
            "source-provenance-and-sd15-inheritance",
            "native-and-diffusers-eight-channel-detection",
            "sd1-clip-and-sd15-latent-profile",
            "transactional-component-mapping",
            "native-forward-and-conditioning-checkpoints",
            "memory-oom-dtype-device-cancellation",
            "partial-malformed-ambiguous-unexpected-and-owner-delegation",
        ],
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sd15_instructpix2pix_mapping_forward_and_failures_are_native()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[instruct::MODEL_FAMILY_REGISTRATION])?;
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
    let checkpoints = model.forward_checkpoints(&backend, &input, &context)?;
    assert_values(
        &backend,
        &context,
        &checkpoints.iter().find(|checkpoint| checkpoint.name == "epsilon_prediction").ok_or("checkpoint")?.tensor,
        &[0.9934323, 0.6237125],
    )?;

    let mapped = registry.resolve(&diffusers_probe())?.map_state_dictionary(
        &ModelStateTransaction::new(&backend, &context),
        DIGEST,
        &diffusers_source(&backend, &context, DType::F32)?,
    )?;
    for key in sd15::REQUIRED_KEYS {
        assert!(mapped.component("denoiser").ok_or("denoiser")?.contains_key(*key));
    }

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

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(2 * 1024 * 1024)?,
        &cancelled,
    );
    assert!(matches!(
        resolved.map_primary_weights(&ModelStateTransaction::new(&backend, &cancelled_context), DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));
    let ambiguous = ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?;
    assert!(matches!(
        ambiguous.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { .. })
    ));
    Ok(())
}

fn combined_registry() -> Result<ModelFamilyRegistry, ModelFamilyError> {
    ModelFamilyRegistry::checked_registrations(&[
        instruct::MODEL_FAMILY_REGISTRATION,
        sd15::MODEL_FAMILY_REGISTRATION,
    ])
}

fn native_probe(include_clip: bool) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        ("model.diffusion_model.input_blocks.0.0.weight".into(), vec![320, 8, 3, 3]),
        ("model.diffusion_model.time_embed.0.weight".into(), vec![2, 2]),
        ("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight".into(), vec![2, 2]),
        ("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight".into(), vec![2, 768]),
        ("model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight".into(), vec![2, 2]),
        ("model.diffusion_model.out.2.weight".into(), vec![4, 320, 3, 3]),
    ]);
    if include_clip {
        tensor_shapes.insert("cond_stage_model.transformer.embeddings.position_ids".into(), vec![3]);
    }
    ModelProbe { tensor_shapes, metadata: BTreeMap::from([("model_family".into(), "SD15".into())]) }
}

fn sd15_probe() -> ModelProbe {
    let mut probe = native_probe(false);
    probe.tensor_shapes.get_mut("model.diffusion_model.input_blocks.0.0.weight").expect("input")[1] = 4;
    probe
}

fn diffusers_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("conv_in.weight".into(), vec![320, 8, 3, 3]),
            ("time_embedding.linear_1.weight".into(), vec![2, 2]),
            ("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight".into(), vec![2, 2]),
            ("down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight".into(), vec![2, 768]),
            ("mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight".into(), vec![2, 2]),
            ("conv_out.weight".into(), vec![4, 320, 3, 3]),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn native_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    include_clip: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    source_for_probe(backend, context, dtype, native_probe(include_clip))
}

fn diffusers_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    source_for_probe(backend, context, dtype, diffusers_probe())
}

fn source_for_probe(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    dtype: DType,
    probe: ModelProbe,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    probe.tensor_shapes.into_iter().map(|(key, shape)| {
        let count = usize::try_from(shape.iter().product::<u64>())?;
        let values = if key.ends_with("position_ids") {
            vec![0.2, 1.6, 2.4]
        } else if key.ends_with("time_embed.0.weight") || key.ends_with("time_embedding.linear_1.weight") {
            vec![1.0, 0.0, 0.0, 1.0]
        } else if key.contains("attn1.to_q.weight") {
            vec![2.0, 0.0, 0.0, 0.5]
        } else if key.contains("middle_block.1") || key.contains("mid_block.attentions") {
            vec![1.0, 1.0, 1.0, -1.0]
        } else {
            vec![0.0; count]
        };
        Ok((key, tensor(backend, &shape, &values, dtype, context)?))
    }).collect()
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
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn verify_provenance_and_catalog() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(
        root.join("crates/sim_test_support/fixtures/models")
            .join(instruct::MODEL_FAMILY_FIXTURE)
            .join("provenance.json"),
    )?)?;
    assert_eq!(sha256(provenance["source_projection"].as_str().ok_or("projection")?.as_bytes()), instruct::MODEL_FAMILY_PROJECTION_SHA256);
    for source in provenance["source_files"].as_array().ok_or("sources")? {
        assert_eq!(sha256(&std::fs::read(root.join(source["path"].as_str().ok_or("path")?))?), source["sha256"]);
    }
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("crates/sim_model/catalog/model-families-v1.json"))?)?;
    let row = catalog["models"].as_array().ok_or("models")?.iter().find(|row| row["feature_id"] == instruct::MODEL_FAMILY_FEATURE_ID).ok_or("row")?;
    assert_eq!(row["source_ordinal"], 2);
    assert_eq!(row["inheritance_chain"][1], "SD15");
    assert_eq!(row["static"]["unet_config"]["value"]["in_channels"], 8);
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
