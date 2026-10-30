use sim_model::{
    Kandinsky5Layout, Kandinsky5Variant, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction,
    describe_model_family, generated_kandinsky5_sim_model_0099 as kandinsky,
    kandinsky5_configuration_for_probe, kandinsky5_state_plan_for_layout,
};
use sim_tensor::{CpuWorkspaceAuthority, DType, StreamId};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

use super::generated_koala_1b_sim_model_0097::support;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9099",
    identifier: "Kandinsky5AmbiguousFixture",
    ..kandinsky::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    kandinsky::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 983,
        source_architecture: "model_base.Kandinsky5AmbiguousFixture",
        ..kandinsky::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_kandinsky5_source_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(kandinsky::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(fixture.fixture_id, kandinsky::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture.feature_id, kandinsky::MODEL_FAMILY_FEATURE_ID);
    verify_provenance()?;
    let descriptor = describe_model_family(&kandinsky::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "Kandinsky5");
    assert_eq!(descriptor.architecture_version, "kandinsky5-video-transformer-v1");
    assert_eq!(descriptor.latent_format, "HunyuanVideo");
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(kandinsky::MODEL_FAMILY_SOURCE_ORDINAL, 83);
    assert_eq!(kandinsky::MODEL_FAMILY_REGISTRATION.source_ordinal, 83);
    assert_eq!(
        kandinsky::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Kandinsky5"
    );
    assert_eq!(kandinsky::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.25);
    assert_eq!(kandinsky::MODEL_FAMILY_SAMPLING_SHIFT, 10.0);

    let registry = ModelFamilyRegistry::checked_registrations(&[
        kandinsky::MODEL_FAMILY_REGISTRATION,
    ])?;
    for layout in [
        Kandinsky5Layout::PrefixedNative,
        Kandinsky5Layout::StandaloneNative,
    ] {
        let probe = layout_probe(&fixture.detector.tensor_shapes, layout);
        let configuration = kandinsky5_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, Kandinsky5Variant::VideoLite);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.input_visual_channels, 33);
        assert_eq!(configuration.output_visual_channels, 16);
        assert_eq!(configuration.model_dimension, 1_792);
        assert_eq!(configuration.time_dimension, 512);
        assert_eq!(configuration.feed_forward_dimension, 7_168);
        assert_eq!(configuration.visual_embed_dimension, 132);
        assert_eq!(configuration.patch_size, [1, 2, 2]);
        assert_eq!(configuration.text_block_count, 2);
        assert_eq!(configuration.visual_block_count, 32);
        assert_eq!(configuration.axes_dimensions, [16, 24, 24]);
        assert_eq!(configuration.attention_head_dimension, 64);
        assert_eq!(configuration.attention_head_count, 28);
        assert!(configuration.concat_conditioning);
        assert_eq!(configuration.sampling_shift, 10.0);
        assert_eq!(configuration.memory_usage_factor, 1.25);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0037");
        assert_eq!(
            registry.resolve(&probe)?.detection().identity.feature_id(),
            kandinsky::MODEL_FAMILY_FEATURE_ID
        );
    }

    let mut pro = layout_probe(
        &fixture.detector.tensor_shapes,
        Kandinsky5Layout::StandaloneNative,
    );
    set_shape(&mut pro, "visual_embeddings.in_layer.bias", &[4_096]);
    set_shape(&mut pro, "visual_embeddings.in_layer.weight", &[4_096, 132]);
    set_shape(
        &mut pro,
        "visual_transformer_blocks.0.feed_forward.in_layer.weight",
        &[16_384, 4_096],
    );
    set_shape(
        &mut pro,
        "visual_transformer_blocks.0.cross_attention.key_norm.weight",
        &[128],
    );
    let pro_configuration = kandinsky5_configuration_for_probe(&pro)?;
    assert_eq!(pro_configuration.variant, Kandinsky5Variant::VideoPro);
    assert_eq!(pro_configuration.attention_head_count, 32);
    assert_eq!(
        registry.resolve(&pro)?.detection().identity.feature_id(),
        kandinsky::MODEL_FAMILY_FEATURE_ID
    );

    let store_probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(
        registry
            .resolve(&store_probe)?
            .detection()
            .identity
            .feature_id(),
        kandinsky::MODEL_FAMILY_FEATURE_ID
    );
    let mut misleading = store_probe.clone();
    misleading
        .metadata
        .insert("image_model".to_owned(), "flux".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        kandinsky::MODEL_FAMILY_FEATURE_ID
    );
    let mut image = layout_probe(
        &fixture.detector.tensor_shapes,
        Kandinsky5Layout::StandaloneNative,
    );
    set_shape(&mut image, "visual_embeddings.in_layer.bias", &[2_560]);
    set_shape(&mut image, "visual_embeddings.in_layer.weight", &[2_560, 64]);
    set_shape(
        &mut image,
        "visual_transformer_blocks.0.feed_forward.in_layer.weight",
        &[10_240, 2_560],
    );
    set_shape(
        &mut image,
        "visual_transformer_blocks.0.cross_attention.key_norm.weight",
        &[128],
    );
    assert_eq!(
        kandinsky5_configuration_for_probe(&image)?.variant,
        Kandinsky5Variant::ImageLite
    );
    assert!(registry.detect(&image).is_err());
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "transformer_blocks.0.cross_attention.key_norm.weight".to_owned(),
            vec![128],
        )]),
        metadata: BTreeMap::from([("model_layout".to_owned(), "diffusers".to_owned())]),
    };
    assert!(matches!(
        kandinsky5_configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut partial = store_probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.visual_embeddings.in_layer.weight");
    assert!(registry.detect(&partial).is_err());
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { .. })
    ));
    verify_owner_delegation()?;
    Ok(())
}

#[test]
fn val_model_family_row_001_kandinsky5_mapping_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(kandinsky::MODEL_FAMILY_FIXTURE)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        Kandinsky5Layout::PrefixedNative,
        Kandinsky5Layout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &kandinsky5_state_plan_for_layout(layout).compile()?,
            &fixture.base_artifact_digest,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model component")?;
        for key in [
            "native.visual_embeddings.in_layer.weight",
            "native.time_embeddings.in_layer.weight",
            "native.text_embeddings.in_layer.weight",
            "native.visual_transformer_blocks.0.cross_attention.key_norm.weight",
            "native.out_layer.out_layer.weight",
        ] {
            assert!(model.contains_key(key), "{layout:?}: {key}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }

    support::exercise_legacy(
        &fixture,
        kandinsky::MODEL_FAMILY_REGISTRATION,
        &backend,
        &context,
    )?;
    let cancellation_source =
        mapping_source(&backend, &context, Kandinsky5Layout::PrefixedNative)?;
    cancellation.cancel();
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &kandinsky5_state_plan_for_layout(Kandinsky5Layout::PrefixedNative).compile()?,
            &fixture.base_artifact_digest,
            &cancellation_source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    super::write_model_family_row_artifact(
        kandinsky::MODEL_FAMILY_FIXTURE,
        kandinsky::MODEL_FAMILY_FEATURE_ID,
        kandinsky::MODEL_FAMILY_IDENTIFIER,
        kandinsky::MODEL_FAMILY_SOURCE_ORDINAL,
        "kandinsky5_comfy_model_0099",
        &[
            "source-catalog-provenance-and-registration",
            "prefixed-and-standalone-video-key-detection",
            "video-lite-pro-configuration-clip-and-latent",
            "image-precedence-and-diffusers-rejection",
            "transactional-model-text-and-vae-routing",
            "named-native-forward-and-patch-order",
            "memory-oom-dtype-device-and-cancellation",
            "partial-ambiguous-misleading-and-owner-delegation",
        ],
    )?;
    Ok(())
}

fn layout_probe(
    native: &BTreeMap<String, Vec<u64>>,
    layout: Kandinsky5Layout,
) -> ModelProbe {
    let target_prefix = match layout {
        Kandinsky5Layout::PrefixedNative => "model.diffusion_model.",
        Kandinsky5Layout::StandaloneNative => "",
    };
    ModelProbe {
        tensor_shapes: native
            .iter()
            .map(|(key, shape)| {
                let suffix = key
                    .strip_prefix("model.diffusion_model.")
                    .unwrap_or(key);
                (format!("{target_prefix}{suffix}"), shape.clone())
            })
            .collect(),
        metadata: BTreeMap::new(),
    }
}

fn set_shape(probe: &mut ModelProbe, key: &str, shape: &[u64]) {
    probe.tensor_shapes.insert(key.to_owned(), shape.to_vec());
}

fn mapping_source(
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    layout: Kandinsky5Layout,
) -> Result<BTreeMap<String, sim_tensor::Tensor>, Box<dyn std::error::Error>> {
    let prefix = match layout {
        Kandinsky5Layout::PrefixedNative => "model.diffusion_model.",
        Kandinsky5Layout::StandaloneNative => "",
    };
    let entries: &[(&str, &[u64], &[f32])] = &[
        ("visual_embeddings.in_layer.weight", &[1], &[1.0]),
        ("visual_embeddings.in_layer.bias", &[1], &[0.0]),
        ("time_embeddings.in_layer.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("time_embeddings.in_layer.bias", &[2], &[0.0, 0.0]),
        ("text_embeddings.in_layer.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("text_embeddings.in_layer.bias", &[2], &[0.0, 0.0]),
        ("visual_transformer_blocks.0.cross_attention.key_norm.weight", &[1], &[1.0]),
        ("visual_transformer_blocks.0.feed_forward.in_layer.weight", &[1], &[1.0]),
        ("out_layer.out_layer.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("out_layer.out_layer.bias", &[2], &[0.1, -0.1]),
    ];
    let mut source = entries
        .iter()
        .map(|(key, shape, values)| {
            Ok((
                format!("{prefix}{key}"),
                support::tensor(backend, context, shape, values, DType::F32)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
    source.insert(
        "vae.decoder.weight".to_owned(),
        support::tensor(backend, context, &[1], &[1.0], DType::F32)?,
    );
    source.insert(
        "text_encoders.qwen25_7b.weight".to_owned(),
        support::tensor(backend, context, &[1], &[1.0], DType::F32)?,
    );
    Ok(source)
}

fn verify_provenance() -> Result<(), Box<dyn std::error::Error>> {
    let provenance: serde_json::Value = serde_json::from_slice(&fs::read(
        fixture_directory().join("provenance.json"),
    )?)?;
    assert_eq!(provenance["feature_id"], kandinsky::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], kandinsky::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], kandinsky::MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(
        provenance["catalog_projection_sha256"],
        kandinsky::MODEL_FAMILY_PROJECTION_SHA256
    );
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
        .and_then(|rows| {
            rows.iter()
                .find(|row| row["feature_id"] == kandinsky::MODEL_FAMILY_FEATURE_ID)
        })
        .ok_or("catalog row")?;
    assert_eq!(
        sha256(&serde_json::to_vec(row)?),
        kandinsky::MODEL_FAMILY_PROJECTION_SHA256
    );
    Ok(())
}

fn verify_owner_delegation() -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/kandinsky5_sim_model_0099.rs"),
    )?;
    for owner in [
        "KANDINSKY5_COMPONENT_STATE_SCHEMAS",
        "KANDINSKY5_FORWARD_PROGRAM",
        "KANDINSKY5_LAYOUT_SIGNATURES",
        "KANDINSKY5_STATE_PLAN_CASES",
        "kandinsky5_configuration_for_probe",
        "ModelProbe",
        "MemoryEstimatorDescriptor",
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

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sim_test_support/fixtures/models")
        .join(kandinsky::MODEL_FAMILY_FIXTURE)
}

fn repository_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
