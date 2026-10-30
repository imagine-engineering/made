use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as t2v,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [t2v::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9146",
    identifier: "WAN21_T2V_AmbiguousFixture",
    ..t2v::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    t2v::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 952,
        source_architecture: "model_base.WAN21_AmbiguousFixture",
        ..t2v::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_t2v_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(t2v::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        t2v::MODEL_FAMILY_FIXTURE,
        t2v::MODEL_FAMILY_FEATURE_ID,
        t2v::MODEL_FAMILY_IDENTIFIER,
        t2v::MODEL_FAMILY_SOURCE_ORDINAL,
        t2v::SOURCE_ARCHITECTURE,
        t2v::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = t2v::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, t2v::Wan21BatchVariant::T2V);
    assert_eq!(configuration.image_model, "wan2.1");
    assert_eq!(configuration.model_type, "t2v");
    assert_eq!(configuration.architecture_model_type, "t2v");
    assert!(!configuration.image_to_video);
    assert!(!configuration.reference_conditioning);
    assert!(!configuration.pose_conditioning);
    assert!(!configuration.mask_conditioning);
    assert!(!configuration.vace_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 16);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.feed_forward_dimension, 256);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.auxiliary_input_channels, None);
    assert_eq!(configuration.auxiliary_layer_count, None);
    assert_eq!(configuration.patch_size, [1, 2, 2]);
    assert_eq!(configuration.frequency_dimension, 256);
    assert!(configuration.qk_norm);
    assert!(configuration.cross_attention_norm);
    assert_eq!(configuration.epsilon_millionths, 1);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&t2v::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 500);
    assert_eq!(resolved.source_ordinal(), 52);
    assert_eq!(resolved.clip_target().candidates().len(), 1);

    let mut specialized = probe.clone();
    specialized.tensor_shapes.insert(
        t2v::POSE_PATCH_WEIGHT.to_owned(),
        vec![128, 16, 1, 2, 2],
    );
    assert!(registry.resolve(&specialized).is_err());
    let mut malformed = probe.clone();
    malformed
        .tensor_shapes
        .insert(t2v::HEAD_MODULATION.to_owned(), vec![2, 128]);
    assert!(matches!(
        t2v::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut misleading = probe.clone();
    misleading.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"model_type\":\"scail\"}}".to_owned(),
    );
    assert!(matches!(
        t2v::configuration_for_probe(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        t2v::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 500, .. })
    ));
    support::verify_owner_delegation("wan21_t2v_comfy_model_0146")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_t2v_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(t2v::MODEL_FAMILY_FIXTURE)?;
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
        t2v::MODEL_FAMILY_FIXTURE,
        t2v::MODEL_FAMILY_FEATURE_ID,
        t2v::MODEL_FAMILY_IDENTIFIER,
        t2v::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_t2v_comfy_model_0146",
        &[
            "source-and-catalog-provenance",
            "generic-wan-detector-fallback",
            "shape-reduced-text-to-video-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-specialized-metadata-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
