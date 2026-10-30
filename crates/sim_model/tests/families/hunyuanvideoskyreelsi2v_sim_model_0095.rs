use sim_model::{
    HunyuanVideoLayout, HunyuanVideoVariant, ModelFamilyDefinition, ModelFamilyError,
    ModelFamilyRegistration, ModelFamilyRegistry, ModelProbe, ModelStateTransaction,
    describe_model_family, generated_hunyuanvideo15_sim_model_0092 as video15,
    generated_hunyuanvideoskyreelsi2v_sim_model_0095 as skyreels,
    hunyuan_video_configuration_for_probe, hunyuan_video_state_plan_for_layout,
};
use sim_tensor::{CpuWorkspaceAuthority, DType, StreamId};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

use super::generated_koala_1b_sim_model_0097::support;

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9095",
    identifier: "HunyuanVideoSkyreelsI2VAmbiguousFixture",
    ..skyreels::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    skyreels::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 938,
        source_architecture: "model_base.HunyuanVideoSkyreelsI2VAmbiguousFixture",
        ..skyreels::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_hunyuanvideoskyreelsi2v_source_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(skyreels::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(fixture.fixture_id, skyreels::MODEL_FAMILY_FIXTURE);
    assert_eq!(fixture.feature_id, skyreels::MODEL_FAMILY_FEATURE_ID);
    verify_provenance()?;
    let descriptor = describe_model_family(&skyreels::MODEL_FAMILY)?;
    assert_eq!(descriptor.identifier, "HunyuanVideoSkyreelsI2V");
    assert_eq!(
        descriptor.architecture_version,
        "hunyuan-video-skyreels-i2v-flow-transformer-v1"
    );
    assert_eq!(descriptor.latent_format, "HunyuanVideo");
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(skyreels::MODEL_FAMILY_SOURCE_ORDINAL, 38);
    assert_eq!(skyreels::MODEL_FAMILY_REGISTRATION.source_ordinal, 38);
    assert_eq!(
        skyreels::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.HunyuanVideoSkyreelsI2V"
    );
    assert_eq!(skyreels::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.8);
    assert_eq!(skyreels::MODEL_FAMILY_SAMPLING_SHIFT, 7.0);

    let registry = ModelFamilyRegistry::checked_registrations(&[
        video15::MODEL_FAMILY_REGISTRATION,
        skyreels::MODEL_FAMILY_REGISTRATION,
    ])?;
    for layout in [
        HunyuanVideoLayout::PrefixedNative,
        HunyuanVideoLayout::SavedModel,
        HunyuanVideoLayout::StandaloneNative,
    ] {
        let probe = layout_probe(&fixture.detector.tensor_shapes, layout);
        let configuration = hunyuan_video_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, HunyuanVideoVariant::VideoSkyreelsI2V);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, 32);
        assert_eq!(configuration.out_channels, 32);
        assert_eq!(configuration.patch_size, [1, 2, 2]);
        assert_eq!(configuration.context_input_dimension, 4_096);
        assert_eq!(configuration.hidden_size, 256);
        assert_eq!(configuration.number_of_heads, 2);
        assert_eq!(configuration.axes_dimensions, [16, 56, 56]);
        assert_eq!(configuration.sampling_shift, 7.0);
        assert_eq!(configuration.memory_usage_factor, 1.8);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0037");
        assert_eq!(
            registry.resolve(&probe)?.detection().identity.feature_id(),
            skyreels::MODEL_FAMILY_FEATURE_ID
        );
    }

    let store_probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(
        registry
            .resolve(&store_probe)?
            .detection()
            .identity
            .feature_id(),
        skyreels::MODEL_FAMILY_FEATURE_ID
    );
    let mut video15_probe = store_probe.clone();
    video15_probe.tensor_shapes.insert(
        "model.diffusion_model.txt_in.input_embedder.weight".to_owned(),
        vec![256, 3_584],
    );
    video15_probe.tensor_shapes.insert(
        "model.diffusion_model.vision_in.proj.0.weight".to_owned(),
        vec![1_152],
    );
    assert_eq!(
        registry
            .resolve(&video15_probe)?
            .detection()
            .identity
            .feature_id(),
        video15::MODEL_FAMILY_FEATURE_ID
    );
    let mut misleading = store_probe.clone();
    misleading
        .metadata
        .insert("image_model".to_owned(), "sdxl".to_owned());
    misleading
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        skyreels::MODEL_FAMILY_FEATURE_ID
    );
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("transformer_blocks.0.attn.to_q.weight".to_owned(), vec![256, 256]),
            ("proj_in.weight".to_owned(), vec![256, 32]),
        ]),
        metadata: BTreeMap::from([("model_layout".to_owned(), "diffusers".to_owned())]),
    };
    assert!(registry.detect(&diffusers).is_err());
    assert!(hunyuan_video_configuration_for_probe(&diffusers).is_err());
    let mut partial = store_probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.final_layer.linear.weight");
    assert!(registry.detect(&partial).is_err());
    assert!(matches!(
        hunyuan_video_configuration_for_probe(&partial),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("partial")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { .. })
    ));
    verify_owner_delegation()?;
    Ok(())
}

#[test]
fn val_model_family_row_001_hunyuanvideoskyreelsi2v_mapping_forward_patch_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(skyreels::MODEL_FAMILY_FIXTURE)?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        HunyuanVideoLayout::PrefixedNative,
        HunyuanVideoLayout::SavedModel,
        HunyuanVideoLayout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan_video_state_plan_for_layout(layout).compile()?,
            &fixture.base_artifact_digest,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model component")?;
        for key in [
            "native.img_in.proj.weight",
            "native.final_layer.linear.weight",
            "native.final_layer.linear.bias",
            "native.txt_in.input_embedder.weight",
            "native.txt_in.individual_token_refiner.blocks.0.norm1.weight",
            "native.txt_in.t_embedder.in_layer.weight",
        ] {
            assert!(model.contains_key(key), "{layout:?}: {key}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }

    support::exercise_legacy(
        &fixture,
        skyreels::MODEL_FAMILY_REGISTRATION,
        &backend,
        &context,
    )?;
    let cancellation_source =
        mapping_source(&backend, &context, HunyuanVideoLayout::PrefixedNative)?;
    cancellation.cancel();
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan_video_state_plan_for_layout(HunyuanVideoLayout::PrefixedNative).compile()?,
            &fixture.base_artifact_digest,
            &cancellation_source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    super::write_model_family_row_artifact(
        skyreels::MODEL_FAMILY_FIXTURE,
        skyreels::MODEL_FAMILY_FEATURE_ID,
        skyreels::MODEL_FAMILY_IDENTIFIER,
        skyreels::MODEL_FAMILY_SOURCE_ORDINAL,
        "hunyuanvideoskyreelsi2v_comfy_model_0095",
        &[
            "source-catalog-provenance-and-registration",
            "prefixed-saved-and-standalone-key-detection",
            "skyreels-i2v-configuration-clip-and-latent",
            "explicit-unsupported-diffusers-rejection",
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
    layout: HunyuanVideoLayout,
) -> ModelProbe {
    let target_prefix = match layout {
        HunyuanVideoLayout::PrefixedNative => "model.diffusion_model.",
        HunyuanVideoLayout::SavedModel => "model.",
        HunyuanVideoLayout::StandaloneNative => "",
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

fn mapping_source(
    backend: &sim_tensor::CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    layout: HunyuanVideoLayout,
) -> Result<BTreeMap<String, sim_tensor::Tensor>, Box<dyn std::error::Error>> {
    let prefix = match layout {
        HunyuanVideoLayout::PrefixedNative => "model.diffusion_model.",
        HunyuanVideoLayout::SavedModel => "model.",
        HunyuanVideoLayout::StandaloneNative => "",
    };
    let entries: &[(&str, &[u64], &[f32])] = &[
        ("img_in.proj.weight", &[1], &[1.0]),
        ("final_layer.linear.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("final_layer.linear.bias", &[2], &[0.1, -0.1]),
        ("txt_in.input_embedder.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
        ("txt_in.input_embedder.bias", &[2], &[1.0, -1.0]),
        ("txt_in.individual_token_refiner.blocks.0.norm1.weight", &[1], &[1.0]),
        ("txt_in.t_embedder.mlp.0.weight", &[1], &[1.0]),
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
        "text_encoders.hunyuan.weight".to_owned(),
        support::tensor(backend, context, &[1], &[1.0], DType::F32)?,
    );
    Ok(source)
}

fn verify_provenance() -> Result<(), Box<dyn std::error::Error>> {
    let provenance: serde_json::Value = serde_json::from_slice(&fs::read(
        fixture_directory().join("provenance.json"),
    )?)?;
    assert_eq!(provenance["feature_id"], skyreels::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(provenance["source_symbol"], skyreels::MODEL_FAMILY_IDENTIFIER);
    assert_eq!(provenance["source_ordinal"], skyreels::MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(
        provenance["catalog_projection_sha256"],
        skyreels::MODEL_FAMILY_PROJECTION_SHA256
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
                .find(|row| row["feature_id"] == skyreels::MODEL_FAMILY_FEATURE_ID)
        })
        .ok_or("catalog row")?;
    assert_eq!(
        sha256(&serde_json::to_vec(row)?),
        skyreels::MODEL_FAMILY_PROJECTION_SHA256
    );
    Ok(())
}

fn verify_owner_delegation() -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/families/hunyuanvideoskyreelsi2v_sim_model_0095.rs"),
    )?;
    for owner in [
        "HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS",
        "HUNYUAN_VIDEO_FORWARD_PROGRAM",
        "HUNYUAN_VIDEO_PREFIXED_STATE_PLAN",
        "HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN",
        "HUNYUAN_VIDEO_STANDALONE_STATE_PLAN",
        "hunyuan_video_configuration_for_probe",
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
        .join(skyreels::MODEL_FAMILY_FIXTURE)
}

fn repository_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
