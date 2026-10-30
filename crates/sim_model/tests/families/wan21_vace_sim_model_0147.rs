use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as t2v,
    generated_wan21_vace_sim_model_0147 as vace,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [vace::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9147",
    identifier: "WAN21_Vace_AmbiguousFixture",
    ..vace::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    vace::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 955,
        source_architecture: "model_base.WAN21_Vace_AmbiguousFixture",
        ..vace::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_vace_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(vace::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        vace::MODEL_FAMILY_FIXTURE,
        vace::MODEL_FAMILY_FEATURE_ID,
        vace::MODEL_FAMILY_IDENTIFIER,
        vace::MODEL_FAMILY_SOURCE_ORDINAL,
        vace::SOURCE_ARCHITECTURE,
        vace::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = vace::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, t2v::Wan21BatchVariant::Vace);
    assert_eq!(configuration.model_type, "vace");
    assert_eq!(configuration.architecture_model_type, "t2v");
    assert!(!configuration.image_to_video);
    assert!(!configuration.reference_conditioning);
    assert!(!configuration.pose_conditioning);
    assert!(!configuration.mask_conditioning);
    assert!(configuration.vace_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 16);
    assert_eq!(configuration.auxiliary_input_channels, Some(96));
    assert_eq!(configuration.auxiliary_layer_count, Some(1));
    assert!((configuration.memory_usage_factor - 1.2 * 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&vace::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_200);
    assert_eq!(resolved.source_ordinal(), 55);

    let mut missing_patch = probe.clone();
    missing_patch.tensor_shapes.remove(t2v::VACE_PATCH_WEIGHT);
    assert!(registry.resolve(&missing_patch).is_err());
    let mut malformed_patch = probe.clone();
    malformed_patch.tensor_shapes.insert(
        t2v::VACE_PATCH_WEIGHT.to_owned(),
        vec![128, 96, 2, 2],
    );
    assert!(matches!(
        vace::configuration_for_probe(&malformed_patch),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut too_many_layers = probe.clone();
    too_many_layers.tensor_shapes.insert(
        "model.diffusion_model.vace_blocks.1.ffn.0.weight".to_owned(),
        vec![256, 128],
    );
    assert!(matches!(
        vace::configuration_for_probe(&too_many_layers),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut misleading = probe.clone();
    misleading.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"model_type\":\"vace\",\"vace_in_dim\":95}}".to_owned(),
    );
    assert!(matches!(
        vace::configuration_for_probe(&misleading),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        vace::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
    ));
    support::verify_owner_delegation("wan21_vace_comfy_model_0147")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_vace_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(vace::MODEL_FAMILY_FIXTURE)?;
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
        vace::MODEL_FAMILY_FIXTURE,
        vace::MODEL_FAMILY_FEATURE_ID,
        vace::MODEL_FAMILY_IDENTIFIER,
        vace::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_vace_comfy_model_0147",
        &[
            "source-and-catalog-provenance",
            "vace-first-source-precedence-detection",
            "shape-reduced-vace-context-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-layer-partial-metadata-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
