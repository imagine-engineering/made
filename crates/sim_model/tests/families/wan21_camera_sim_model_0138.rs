use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_camera_sim_model_0138 as camera,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [camera::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9138",
    identifier: "WAN21_Camera_AmbiguousFixture",
    ..camera::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    camera::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 956,
        source_architecture: "model_base.WAN21_Camera_AmbiguousFixture",
        ..camera::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_camera_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(camera::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        camera::MODEL_FAMILY_FIXTURE,
        camera::MODEL_FAMILY_FEATURE_ID,
        camera::MODEL_FAMILY_IDENTIFIER,
        camera::MODEL_FAMILY_SOURCE_ORDINAL,
        camera::SOURCE_ARCHITECTURE,
        camera::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = camera::configuration_for_probe(&probe)?;
    assert_eq!(configuration.image_model, "wan2.1");
    assert_eq!(configuration.model_type, "camera");
    assert!(!configuration.causal_ar);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 32);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.feed_forward_dimension, 256);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.patch_size, [1, 2, 2]);
    assert_eq!(configuration.camera_condition_channels, Some(24));
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&camera::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    let clip = &resolved.clip_target().candidates()[0];
    assert_eq!(
        clip.tokenizer().identifier(),
        "sim.text_encoders.wan.WanT5Tokenizer"
    );
    assert_eq!(clip.clip_model().target().as_str(), "sim.text_encoders.wan.te");
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 56);

    let mut partial = probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.img_emb.proj.0.bias");
    assert!(registry.resolve(&partial).is_err());
    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.control_adapter.conv.weight".to_owned(),
        vec![128, 1_535, 2, 2],
    );
    assert!(matches!(
        camera::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        camera::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut misleading = probe.clone();
    misleading
        .metadata
        .insert("model_type".to_owned(), "t2v".to_owned());
    assert_eq!(
        registry
            .resolve(&misleading)?
            .detection()
            .identity
            .feature_id(),
        camera::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("wan21_camera_comfy_model_0138")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_camera_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(camera::MODEL_FAMILY_FIXTURE)?;
    let extras = [
        support::TensorFixture::new("text_encoders.umt5xxl.transformer.block.weight", &[1], &[1.0]),
        support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0]),
    ];
    support::exercise_family(
        &fixture,
        &REGISTRATIONS,
        &extras,
        &["model", "runtime_conditioning", "text_encoder", "vae"],
        "native.blocks.0.ffn.2.weight",
    )?;
    super::write_model_family_row_artifact(
        camera::MODEL_FAMILY_FIXTURE,
        camera::MODEL_FAMILY_FEATURE_ID,
        camera::MODEL_FAMILY_IDENTIFIER,
        camera::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_camera_comfy_model_0138",
        &[
            "source-and-catalog-provenance",
            "camera-control-key-derived-detection",
            "shape-reduced-wan-camera-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-partial-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
