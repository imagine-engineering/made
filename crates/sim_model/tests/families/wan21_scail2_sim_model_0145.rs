use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_scail2_sim_model_0145 as scail2,
    generated_wan21_scail_sim_model_0144 as scail,
    generated_wan21_t2v_sim_model_0146 as t2v,
    generated_wan21_vace_sim_model_0147 as vace,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [scail2::MODEL_FAMILY_REGISTRATION];
static BATCH_REGISTRATIONS: [ModelFamilyRegistration; 4] = [
    t2v::MODEL_FAMILY_REGISTRATION,
    vace::MODEL_FAMILY_REGISTRATION,
    scail::MODEL_FAMILY_REGISTRATION,
    scail2::MODEL_FAMILY_REGISTRATION,
];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9145",
    identifier: "WAN21_SCAIL2_AmbiguousFixture",
    ..scail2::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    scail2::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 963,
        source_architecture: "model_base.WAN21_SCAIL2_AmbiguousFixture",
        ..scail2::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_scail2_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(scail2::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        scail2::MODEL_FAMILY_FIXTURE,
        scail2::MODEL_FAMILY_FEATURE_ID,
        scail2::MODEL_FAMILY_IDENTIFIER,
        scail2::MODEL_FAMILY_SOURCE_ORDINAL,
        scail2::SOURCE_ARCHITECTURE,
        scail2::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = scail2::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, t2v::Wan21BatchVariant::Scail2);
    assert_eq!(configuration.model_type, "scail2");
    assert_eq!(configuration.architecture_model_type, "i2v");
    assert!(configuration.reference_conditioning);
    assert!(configuration.pose_conditioning);
    assert!(configuration.mask_conditioning);
    assert!(!configuration.vace_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 20);
    assert_eq!(configuration.auxiliary_input_channels, Some(28));
    assert_eq!(configuration.auxiliary_layer_count, None);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&scail2::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_200);
    assert_eq!(resolved.source_ordinal(), 63);
    let batch = ModelFamilyRegistry::checked_registrations(&BATCH_REGISTRATIONS)?;
    assert_eq!(
        batch.resolve(&probe)?.detection().identity.feature_id(),
        scail2::MODEL_FAMILY_FEATURE_ID
    );

    let mut missing_mask = probe.clone();
    missing_mask.tensor_shapes.remove(t2v::MASK_PATCH_WEIGHT);
    assert!(registry.resolve(&missing_mask).is_err());
    let mut missing_pose = probe.clone();
    missing_pose.tensor_shapes.remove(t2v::POSE_PATCH_WEIGHT);
    assert!(registry.resolve(&missing_pose).is_err());
    let mut wrong_mask = probe.clone();
    wrong_mask.tensor_shapes.insert(
        t2v::MASK_PATCH_WEIGHT.to_owned(),
        vec![128, 27, 1, 2, 2],
    );
    assert!(matches!(
        scail2::configuration_for_probe(&wrong_mask),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut vace_overlap = probe.clone();
    vace_overlap.tensor_shapes.insert(
        t2v::VACE_PATCH_WEIGHT.to_owned(),
        vec![128, 96, 1, 2, 2],
    );
    assert!(matches!(
        scail2::configuration_for_probe(&vace_overlap),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut misleading = probe.clone();
    misleading.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"model_type\":\"scail2\",\"mask_in_dim\":27}}".to_owned(),
    );
    assert!(matches!(
        scail2::configuration_for_probe(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        scail2::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
    ));
    support::verify_owner_delegation("wan21_scail2_comfy_model_0145")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_scail2_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(scail2::MODEL_FAMILY_FIXTURE)?;
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
        scail2::MODEL_FAMILY_FIXTURE,
        scail2::MODEL_FAMILY_FEATURE_ID,
        scail2::MODEL_FAMILY_IDENTIFIER,
        scail2::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_scail2_comfy_model_0145",
        &[
            "source-and-catalog-provenance",
            "mask-before-pose-source-precedence-detection",
            "shape-reduced-multi-reference-mask-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-adapter-partial-metadata-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
