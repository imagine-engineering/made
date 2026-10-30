use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    SdxlLayout, SdxlVariant, describe_model_family, generated_ssd1b_sim_model_0127 as ssd1b,
};

use super::generated_sdxl_instructpix2pix_sim_model_0125::support;

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [ssd1b::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9127",
    identifier: "SSD1BAmbiguousFixture",
    ..ssd1b::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    ssd1b::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 910,
        source_architecture: "model_base.SSD1BAmbiguousFixture",
        ..ssd1b::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_ssd1b_source_layouts_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    support::verify_provenance(
        ssd1b::MODEL_FAMILY_FIXTURE,
        ssd1b::MODEL_FAMILY_FEATURE_ID,
        ssd1b::MODEL_FAMILY_IDENTIFIER,
        ssd1b::MODEL_FAMILY_SOURCE_ORDINAL,
        ssd1b::SOURCE_ARCHITECTURE,
        ssd1b::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    assert_eq!(ssd1b::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.8);
    let descriptor = describe_model_family(&ssd1b::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.component_graph.len(), 3);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    for layout in [
        SdxlLayout::PrefixedNative,
        SdxlLayout::StandaloneNative,
        SdxlLayout::Diffusers,
    ] {
        let probe = support::variant_probe(layout, 4, 4);
        let configuration = ssd1b::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, SdxlVariant::Ssd1B);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.transformer_depth_middle, -1);
        assert_eq!(configuration.transformer_depth, [0, 0, 2, 2, 4, 4]);
        assert_eq!(registry.resolve(&probe)?.detection().score, 1_100);
    }
    let store_probe = support::probe_through_model_store(ssd1b::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(registry.resolve(&store_probe)?.source_ordinal(), 10);
    let mut partial = store_probe.clone();
    partial.tensor_shapes.remove(
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_k.weight",
    );
    assert!(registry.resolve(&partial).is_err());
    assert!(matches!(
        ssd1b::configuration_for_probe(&support::variant_probe(SdxlLayout::Diffusers, 4, 2)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("depth-4")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_100, .. })
    ));
    support::verify_owner_delegation("ssd1b_comfy_model_0127")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_ssd1b_native_execution_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_sdxl(ssd1b::MODEL_FAMILY_FIXTURE, &REGISTRATIONS)?;
    super::write_model_family_row_artifact(
        ssd1b::MODEL_FAMILY_FIXTURE,
        ssd1b::MODEL_FAMILY_FEATURE_ID,
        ssd1b::MODEL_FAMILY_IDENTIFIER,
        ssd1b::MODEL_FAMILY_SOURCE_ORDINAL,
        "ssd1b_comfy_model_0127",
        &[
            "source-and-catalog-provenance",
            "native-standalone-and-diffusers-key-detection",
            "source-exact-ssd1b-depth-profile",
            "transactional-sdxl-component-routing",
            "named-native-forward-and-conditioning-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-and-cross-variant-failures",
            "canonical-sdxl-owner-delegation",
        ],
    )?;
    Ok(())
}
