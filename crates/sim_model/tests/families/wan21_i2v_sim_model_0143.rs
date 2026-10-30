use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_i2v_sim_model_0143 as i2v,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [i2v::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9143",
    identifier: "WAN21_I2V_AmbiguousFixture",
    ..i2v::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    i2v::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 953,
        source_architecture: "model_base.WAN21_I2V_AmbiguousFixture",
        ..i2v::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_i2v_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(i2v::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        i2v::MODEL_FAMILY_FIXTURE,
        i2v::MODEL_FAMILY_FEATURE_ID,
        i2v::MODEL_FAMILY_IDENTIFIER,
        i2v::MODEL_FAMILY_SOURCE_ORDINAL,
        i2v::SOURCE_ARCHITECTURE,
        i2v::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = i2v::configuration_for_probe(&probe)?;
    assert_eq!(configuration.image_model, "wan2.1");
    assert_eq!(configuration.model_type, "i2v");
    assert_eq!(configuration.architecture_model_type, "i2v");
    assert!(configuration.image_to_video);
    assert!(!configuration.audio_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 36);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.feed_forward_dimension, 256);
    assert_eq!(configuration.layer_count, 1);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&i2v::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 53);

    let mut missing_image = probe.clone();
    missing_image
        .tensor_shapes
        .remove("model.diffusion_model.img_emb.proj.0.bias");
    assert!(registry.resolve(&missing_image).is_err());
    let mut wrong_channels = probe.clone();
    wrong_channels.tensor_shapes.insert(
        "model.diffusion_model.patch_embedding.weight".to_owned(),
        vec![128, 48, 1, 2, 2],
    );
    assert!(registry.resolve(&wrong_channels).is_err());
    let mut humo = probe.clone();
    humo.tensor_shapes.insert(
        "model.diffusion_model.audio_proj.audio_proj_glob_1.layer.bias".to_owned(),
        vec![128],
    );
    assert!(matches!(
        i2v::configuration_for_probe(&humo),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut misleading = probe.clone();
    misleading.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"in_dim\":48}}".to_owned(),
    );
    assert!(matches!(
        i2v::configuration_for_probe(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        i2v::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("wan21_i2v_comfy_model_0143")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_i2v_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(i2v::MODEL_FAMILY_FIXTURE)?;
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
        i2v::MODEL_FAMILY_FIXTURE,
        i2v::MODEL_FAMILY_FEATURE_ID,
        i2v::MODEL_FAMILY_IDENTIFIER,
        i2v::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_i2v_comfy_model_0143",
        &[
            "source-and-catalog-provenance",
            "image-key-and-in-dim-36-detection",
            "shape-reduced-image-to-video-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-humo-partial-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
