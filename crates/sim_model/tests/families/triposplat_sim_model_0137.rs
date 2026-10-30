use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_triposplat_sim_model_0137 as tripo,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [tripo::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9137",
    identifier: "TripoSplatAmbiguousFixture",
    ..tripo::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    tripo::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 968,
        source_architecture: "model_base.TripoSplatAmbiguousFixture",
        ..tripo::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_triposplat_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(tripo::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        tripo::MODEL_FAMILY_FIXTURE,
        tripo::MODEL_FAMILY_FEATURE_ID,
        tripo::MODEL_FAMILY_IDENTIFIER,
        tripo::MODEL_FAMILY_SOURCE_ORDINAL,
        tripo::SOURCE_ARCHITECTURE,
        tripo::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = tripo::configuration_for_probe(&probe)?;
    assert_eq!(configuration.query_token_length, 8_192);
    assert_eq!(configuration.latent_channels, 16);
    assert_eq!(configuration.model_channels, 1_024);
    assert_eq!(configuration.conditioning_channels, 1_280);
    assert_eq!(configuration.secondary_conditioning_channels, 128);
    assert_eq!(configuration.camera_channels, 5);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.block_count, 24);
    assert_eq!(configuration.refiner_block_count, 2);
    assert_eq!(configuration.attention_heads, 16);
    assert_eq!(configuration.attention_head_channels, 64);
    assert!(configuration.shared_modulation);
    assert!(configuration.qk_rms_norm);

    let descriptor = describe_model_family(&tripo::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "TripoSplat");
    assert_eq!(descriptor.component_graph.len(), 3);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert!(resolved.clip_target().candidates().is_empty());
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 68);

    let mut partial = probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.repo_layers.0.final_map.weight");
    assert!(registry.resolve(&partial).is_err());
    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.cam_out_layer.weight".to_owned(),
        vec![6, 1_024],
    );
    assert!(matches!(
        tripo::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        tripo::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut misleading = probe.clone();
    misleading
        .metadata
        .insert("image_model".to_owned(), "hidream".to_owned());
    assert_eq!(
        registry
            .resolve(&misleading)?
            .detection()
            .identity
            .feature_id(),
        tripo::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("triposplat_comfy_model_0137")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_triposplat_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(tripo::MODEL_FAMILY_FIXTURE)?;
    let extras = [support::TensorFixture::new(
        "first_stage_model.octree.out_proj.weight",
        &[1],
        &[1.0],
    )];
    support::exercise_family(
        &fixture,
        &REGISTRATIONS,
        &extras,
        &["model", "runtime_conditioning", "vae"],
        "native.blocks.0.mlp.mlp.2.weight",
    )?;
    super::write_model_family_row_artifact(
        tripo::MODEL_FAMILY_FIXTURE,
        tripo::MODEL_FAMILY_FEATURE_ID,
        tripo::MODEL_FAMILY_IDENTIFIER,
        tripo::MODEL_FAMILY_SOURCE_ORDINAL,
        "triposplat_comfy_model_0137",
        &[
            "source-and-catalog-provenance",
            "paired-triposplat-key-detection",
            "source-exact-latent-sequence-configuration",
            "transactional-model-runtime-vae-routing",
            "named-native-forward-and-patch-order",
            "memory-oom-dtype-device-cancellation",
            "diffusers-partial-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
