use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_flowrvs_sim_model_0140 as flow,
    generated_wan21_funcontrol2v_sim_model_0141 as fun,
    generated_wan21_humo_sim_model_0142 as humo,
    generated_wan21_i2v_sim_model_0143 as i2v,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [humo::MODEL_FAMILY_REGISTRATION];
static BATCH_REGISTRATIONS: [ModelFamilyRegistration; 4] = [
    i2v::MODEL_FAMILY_REGISTRATION,
    fun::MODEL_FAMILY_REGISTRATION,
    humo::MODEL_FAMILY_REGISTRATION,
    flow::MODEL_FAMILY_REGISTRATION,
];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9142",
    identifier: "WAN21_HuMo_AmbiguousFixture",
    ..humo::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    humo::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 959,
        source_architecture: "model_base.WAN21_HuMo_AmbiguousFixture",
        ..humo::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_humo_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(humo::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        humo::MODEL_FAMILY_FIXTURE,
        humo::MODEL_FAMILY_FEATURE_ID,
        humo::MODEL_FAMILY_IDENTIFIER,
        humo::MODEL_FAMILY_SOURCE_ORDINAL,
        humo::SOURCE_ARCHITECTURE,
        humo::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = humo::configuration_for_probe(&probe)?;
    assert_eq!(configuration.image_model, "wan2.1");
    assert_eq!(configuration.model_type, "humo");
    assert_eq!(configuration.architecture_model_type, "humo");
    assert!(!configuration.image_to_video);
    assert!(configuration.audio_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 36);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.feed_forward_dimension, 256);
    assert_eq!(configuration.layer_count, 1);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&humo::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_200);
    assert_eq!(resolved.source_ordinal(), 59);
    let batch = ModelFamilyRegistry::checked_registrations(&BATCH_REGISTRATIONS)?;
    assert_eq!(
        batch.resolve(&probe)?.detection().identity.feature_id(),
        humo::MODEL_FAMILY_FEATURE_ID
    );

    let mut missing_audio = probe.clone();
    missing_audio.tensor_shapes.remove(
        "model.diffusion_model.audio_proj.audio_proj_glob_1.layer.bias",
    );
    assert!(registry.resolve(&missing_audio).is_err());
    let mut malformed_audio = probe.clone();
    malformed_audio.tensor_shapes.insert(
        "model.diffusion_model.audio_proj.audio_proj_glob_1.layer.bias".to_owned(),
        vec![127],
    );
    assert!(matches!(
        humo::configuration_for_probe(&malformed_audio),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut camera = probe.clone();
    camera.tensor_shapes.insert(
        "model.diffusion_model.control_adapter.conv.weight".to_owned(),
        vec![128, 1_536, 2, 2],
    );
    assert!(matches!(
        humo::configuration_for_probe(&camera),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut misleading = probe.clone();
    misleading.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"model_type\":\"i2v\"}}".to_owned(),
    );
    assert!(matches!(
        humo::configuration_for_probe(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        humo::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
    ));
    support::verify_owner_delegation("wan21_humo_comfy_model_0142")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_humo_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(humo::MODEL_FAMILY_FIXTURE)?;
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
        humo::MODEL_FAMILY_FIXTURE,
        humo::MODEL_FAMILY_FEATURE_ID,
        humo::MODEL_FAMILY_IDENTIFIER,
        humo::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_humo_comfy_model_0142",
        &[
            "source-and-catalog-provenance",
            "audio-projection-precedence-detection",
            "shape-reduced-humo-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-adapter-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
