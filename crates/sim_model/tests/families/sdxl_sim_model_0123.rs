use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    SdxlLayout, SdxlVariant, describe_model_family, generated_sdxl_sim_model_0123 as sdxl,
};

use super::generated_sdxl_instructpix2pix_sim_model_0125::support;

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [sdxl::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9123",
    identifier: "SDXLAmbiguousFixture",
    ..sdxl::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sdxl::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 909,
        source_architecture: "model_base.SDXLAmbiguousFixture",
        ..sdxl::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sdxl_source_layouts_precedence_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    support::verify_provenance(
        sdxl::MODEL_FAMILY_FIXTURE,
        sdxl::MODEL_FAMILY_FEATURE_ID,
        sdxl::MODEL_FAMILY_IDENTIFIER,
        sdxl::MODEL_FAMILY_SOURCE_ORDINAL,
        sdxl::SOURCE_ARCHITECTURE,
        sdxl::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    assert_eq!(sdxl::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.8);
    let descriptor = describe_model_family(&sdxl::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.component_graph.len(), 3);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    for layout in [
        SdxlLayout::PrefixedNative,
        SdxlLayout::StandaloneNative,
        SdxlLayout::Diffusers,
    ] {
        let probe = support::variant_probe(layout, 4, 10);
        let configuration = sdxl::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, SdxlVariant::Base);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, 4);
        assert_eq!(configuration.transformer_depth_middle, 10);
        assert_eq!(registry.resolve(&probe)?.detection().score, 1_300);
    }
    let store_probe = support::probe_through_model_store(sdxl::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(registry.resolve(&store_probe)?.source_ordinal(), 9);
    let mut misleading = store_probe.clone();
    misleading
        .metadata
        .insert("model_family".to_owned(), "SSD1B".to_owned());
    assert_eq!(registry.resolve(&misleading)?.profile().latent_identifier, "SDXL");
    let mut partial = store_probe.clone();
    partial.tensor_shapes.remove(
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.9.attn2.to_k.weight",
    );
    assert!(registry.resolve(&partial).is_err());
    assert!(matches!(
        sdxl::configuration_for_probe(&support::variant_probe(SdxlLayout::Diffusers, 4, 4)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("depth-10")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_300, .. })
    ));
    support::verify_owner_delegation("sdxl_comfy_model_0123")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sdxl_native_execution_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_sdxl(sdxl::MODEL_FAMILY_FIXTURE, &REGISTRATIONS)?;
    super::write_model_family_row_artifact(
        sdxl::MODEL_FAMILY_FIXTURE,
        sdxl::MODEL_FAMILY_FEATURE_ID,
        sdxl::MODEL_FAMILY_IDENTIFIER,
        sdxl::MODEL_FAMILY_SOURCE_ORDINAL,
        "sdxl_comfy_model_0123",
        &[
            "source-and-catalog-provenance",
            "native-standalone-and-diffusers-key-detection",
            "source-exact-sdxl-base-depth-profile",
            "sdxl-variant-detection-precedence",
            "transactional-sdxl-component-routing",
            "named-native-forward-and-conditioning-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-misleading-and-cross-variant-failures",
            "canonical-sdxl-owner-delegation",
        ],
    )?;
    Ok(())
}
