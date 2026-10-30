use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    SdxlLayout, SdxlVariant, describe_model_family,
    generated_sdxl_instructpix2pix_sim_model_0125 as ip2p,
    generated_segmind_vega_sim_model_0131 as vega, generated_ssd1b_sim_model_0127 as ssd1b,
};

use super::generated_sdxl_instructpix2pix_sim_model_0125::support;

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [vega::MODEL_FAMILY_REGISTRATION];
static SDXL_BATCH_REGISTRATIONS: [ModelFamilyRegistration; 3] = [
    ip2p::MODEL_FAMILY_REGISTRATION,
    ssd1b::MODEL_FAMILY_REGISTRATION,
    vega::MODEL_FAMILY_REGISTRATION,
];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9131",
    identifier: "SegmindVegaAmbiguousFixture",
    ..vega::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    vega::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 913,
        source_architecture: "model_base.SegmindVegaAmbiguousFixture",
        ..vega::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_segmind_vega_source_layouts_precedence_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    support::verify_provenance(
        vega::MODEL_FAMILY_FIXTURE,
        vega::MODEL_FAMILY_FEATURE_ID,
        vega::MODEL_FAMILY_IDENTIFIER,
        vega::MODEL_FAMILY_SOURCE_ORDINAL,
        vega::SOURCE_ARCHITECTURE,
        vega::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    assert_eq!(vega::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.8);
    let descriptor = describe_model_family(&vega::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.component_graph.len(), 3);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    for layout in [
        SdxlLayout::PrefixedNative,
        SdxlLayout::StandaloneNative,
        SdxlLayout::Diffusers,
    ] {
        let probe = support::variant_probe(layout, 4, 2);
        let configuration = vega::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, SdxlVariant::SegmindVega);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.transformer_depth_middle, -1);
        assert_eq!(configuration.transformer_depth, [0, 0, 1, 1, 2, 2]);
        assert_eq!(registry.resolve(&probe)?.detection().score, 1_000);
    }
    let store_probe = support::probe_through_model_store(vega::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(registry.resolve(&store_probe)?.source_ordinal(), 13);
    let batch = ModelFamilyRegistry::checked_registrations(&SDXL_BATCH_REGISTRATIONS)?;
    assert_eq!(
        batch
            .resolve(&support::variant_probe(SdxlLayout::PrefixedNative, 8, 10))?
            .detection()
            .identity
            .feature_id(),
        ip2p::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(
        batch
            .resolve(&support::variant_probe(SdxlLayout::PrefixedNative, 4, 4))?
            .detection()
            .identity
            .feature_id(),
        ssd1b::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(batch.resolve(&store_probe)?.detection().identity.feature_id(), vega::MODEL_FAMILY_FEATURE_ID);
    let mut misleading = store_probe.clone();
    misleading
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert_eq!(registry.resolve(&misleading)?.source_ordinal(), 13);
    assert!(matches!(
        vega::configuration_for_probe(&support::variant_probe(SdxlLayout::StandaloneNative, 4, 4)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("depth-2")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("segmind_vega_comfy_model_0131")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_segmind_vega_native_execution_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_sdxl(vega::MODEL_FAMILY_FIXTURE, &REGISTRATIONS)?;
    super::write_model_family_row_artifact(
        vega::MODEL_FAMILY_FIXTURE,
        vega::MODEL_FAMILY_FEATURE_ID,
        vega::MODEL_FAMILY_IDENTIFIER,
        vega::MODEL_FAMILY_SOURCE_ORDINAL,
        "segmind_vega_comfy_model_0131",
        &[
            "source-and-catalog-provenance",
            "native-standalone-and-diffusers-key-detection",
            "source-exact-segmind-depth-profile",
            "sdxl-batch-source-precedence",
            "transactional-sdxl-component-routing",
            "named-native-forward-and-conditioning-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "ambiguous-misleading-and-cross-variant-failures",
            "canonical-sdxl-owner-delegation",
        ],
    )?;
    Ok(())
}
