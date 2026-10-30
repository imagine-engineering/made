use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_scail_sim_model_0144 as scail,
    generated_wan21_t2v_sim_model_0146 as t2v,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [scail::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9144",
    identifier: "WAN21_SCAIL_AmbiguousFixture",
    ..scail::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    scail::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 962,
        source_architecture: "model_base.WAN21_SCAIL_AmbiguousFixture",
        ..scail::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_scail_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(scail::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        scail::MODEL_FAMILY_FIXTURE,
        scail::MODEL_FAMILY_FEATURE_ID,
        scail::MODEL_FAMILY_IDENTIFIER,
        scail::MODEL_FAMILY_SOURCE_ORDINAL,
        scail::SOURCE_ARCHITECTURE,
        scail::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = scail::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, t2v::Wan21BatchVariant::Scail);
    assert_eq!(configuration.model_type, "scail");
    assert_eq!(configuration.architecture_model_type, "i2v");
    assert!(!configuration.image_to_video);
    assert!(configuration.reference_conditioning);
    assert!(configuration.pose_conditioning);
    assert!(!configuration.mask_conditioning);
    assert!(!configuration.vace_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 20);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.auxiliary_input_channels, None);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&scail::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 62);

    let mut missing_pose = probe.clone();
    missing_pose.tensor_shapes.remove(t2v::POSE_PATCH_WEIGHT);
    assert!(registry.resolve(&missing_pose).is_err());
    let mut wrong_pose = probe.clone();
    wrong_pose.tensor_shapes.insert(
        t2v::POSE_PATCH_WEIGHT.to_owned(),
        vec![128, 19, 1, 2, 2],
    );
    assert!(matches!(
        scail::configuration_for_probe(&wrong_pose),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut scail2 = probe.clone();
    scail2.tensor_shapes.insert(
        t2v::MASK_PATCH_WEIGHT.to_owned(),
        vec![128, 28, 1, 2, 2],
    );
    assert!(matches!(
        scail::configuration_for_probe(&scail2),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut misleading = probe.clone();
    misleading.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"model_type\":\"t2v\"}}".to_owned(),
    );
    assert!(matches!(
        scail::configuration_for_probe(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        scail::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("wan21_scail_comfy_model_0144")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_scail_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(scail::MODEL_FAMILY_FIXTURE)?;
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
        scail::MODEL_FAMILY_FIXTURE,
        scail::MODEL_FAMILY_FEATURE_ID,
        scail::MODEL_FAMILY_IDENTIFIER,
        scail::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_scail_comfy_model_0144",
        &[
            "source-and-catalog-provenance",
            "pose-key-source-precedence-detection",
            "shape-reduced-reference-pose-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-mask-partial-metadata-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
