use super::{
    generated_stable_zero123_sim_model_0136::support as row_support,
    generated_zimage_sim_model_0153::support,
};
use sim_model::{
    LUMINA_ZIMAGE_STANDALONE_STATE_PLAN, LuminaZImageLayout, LuminaZImageVariant,
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ZIMAGE_PIXEL_LATENT_FORMAT, describe_model_family,
    generated_zimage_sim_model_0153 as zimage,
    generated_zimagepixelspace_sim_model_0154 as pixel,
};
use sim_tensor::DType;
use sim_types::DeviceKind;

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [pixel::MODEL_FAMILY_REGISTRATION];
static COMPLETE_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    pixel::MODEL_FAMILY_REGISTRATION,
    zimage::MODEL_FAMILY_REGISTRATION,
];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9154",
    identifier: "ZImagePixelSpace_AmbiguousFixture",
    ..pixel::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    pixel::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 945,
        source_architecture: "model_base.ZImagePixelSpace_AmbiguousFixture",
        ..pixel::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_zimage_pixel_space_configuration_and_layouts()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(pixel::MODEL_FAMILY_FIXTURE)?;
    row_support::verify_provenance(
        pixel::MODEL_FAMILY_FIXTURE,
        pixel::MODEL_FAMILY_FEATURE_ID,
        pixel::MODEL_FAMILY_IDENTIFIER,
        pixel::MODEL_FAMILY_SOURCE_ORDINAL,
        pixel::SOURCE_ARCHITECTURE,
        pixel::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let direct_probe = support::probe(&fixture);
    let store_probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(direct_probe.tensor_shapes(), store_probe.tensor_shapes());

    let configuration = pixel::configuration_for_probe(&store_probe)?;
    assert_eq!(configuration.variant, LuminaZImageVariant::ZImagePixelSpace);
    assert_eq!(configuration.layout, LuminaZImageLayout::PrefixedNative);
    assert_eq!(configuration.dimension, 3_840);
    assert_eq!(configuration.number_of_layers, 1);
    assert_eq!(configuration.patch_size, 4);
    assert_eq!((configuration.input_channels, configuration.output_channels), (3, 3));
    assert_eq!(configuration.memory_usage_factor, 0.03);
    assert_eq!(configuration.supported_dtypes, &[DType::Bf16, DType::F32]);
    assert_eq!(
        configuration.latent_format.feature_id,
        ZIMAGE_PIXEL_LATENT_FORMAT.feature_id
    );
    assert_eq!(
        configuration.latent_format.identifier,
        ZIMAGE_PIXEL_LATENT_FORMAT.identifier
    );
    let decoder = configuration.pixel_decoder.ok_or("missing pixel decoder")?;
    assert_eq!(decoder.input_channels, 48);
    assert_eq!(decoder.hidden_size, 512);
    assert_eq!(decoder.number_of_residual_blocks, 1);
    assert_eq!(decoder.maximum_frequencies, 8);
    assert!(decoder.uses_x0);

    let descriptor = describe_model_family(&pixel::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "ZImagePixelSpace");
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.memory_estimator.bytes_per_parameter, 1);

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&store_probe)?;
    assert_eq!(resolved.detection().score, 1_200);
    assert_eq!(resolved.source_ordinal(), 45);
    assert_eq!(resolved.source_architecture(), pixel::SOURCE_ARCHITECTURE);

    let complete = ModelFamilyRegistry::checked_registrations(&COMPLETE_REGISTRATIONS)?;
    assert_eq!(
        complete.resolve(&direct_probe)?.detection().identity.feature_id(),
        pixel::MODEL_FAMILY_FEATURE_ID
    );
    let latent_fixture = support::load_fixture(zimage::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(
        complete
            .resolve(&support::probe(&latent_fixture))?
            .detection()
            .identity
            .feature_id(),
        zimage::MODEL_FAMILY_FEATURE_ID
    );

    let standalone = support::standalone_probe(&direct_probe);
    let standalone_configuration = pixel::configuration_for_probe(&standalone)?;
    assert_eq!(standalone_configuration.layout, LuminaZImageLayout::StandaloneNative);
    support::assert_selected_plan_identity(
        &registry,
        &standalone,
        &LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
    )?;
    support::exercise_selected_state_plan(
        &registry,
        &standalone,
        &standalone_state_source(),
        &["model", "text_encoder"],
        &[
            "native.x_embedder.weight",
            "native.cap_embedder.1.weight",
            "native.noise_refiner.0.attention.k_norm.weight",
            "native.final_layer.linear.weight",
            "native.dec_net.cond_embed.weight",
        ],
    )?;

    let mut diffusers = support::zimage_diffusers_probe(1);
    diffusers
        .tensor_shapes
        .insert("dec_net.cond_embed.weight".to_owned(), vec![512, 48]);
    assert!(matches!(
        pixel::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("does not admit the pinned Diffusers")
    ));
    assert!(registry.resolve(&diffusers).is_err());

    let mut malformed = direct_probe.clone();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.dec_net.input_embedder.embedder.0.weight".to_owned(),
        vec![512, 111],
    );
    assert!(matches!(
        pixel::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("not a perfect square")
    ));

    let mut partial = direct_probe.clone();
    partial
        .tensor_shapes
        .remove("model.diffusion_model.dec_net.cond_embed.weight");
    assert!(registry.resolve(&partial).is_err());
    let mut misleading = direct_probe.clone();
    misleading
        .metadata
        .insert("image_model".to_owned(), "lumina2".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        pixel::MODEL_FAMILY_FEATURE_ID
    );
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&direct_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })
    ));
    row_support::verify_owner_delegation("zimagepixelspace_comfy_model_0154")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_zimage_pixel_space_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(pixel::MODEL_FAMILY_FIXTURE)?;
    let extras = [support::TensorFixture::new(
        "text_encoders.qwen3_4b.transformer.weight",
        &[1],
        &[1.0],
    )];
    support::exercise_family(
        &fixture,
        &REGISTRATIONS,
        &extras,
        &["model", "text_encoder"],
        "native.cap_embedder.1.weight",
    )?;
    super::write_model_family_row_artifact(
        pixel::MODEL_FAMILY_FIXTURE,
        pixel::MODEL_FAMILY_FEATURE_ID,
        pixel::MODEL_FAMILY_IDENTIFIER,
        pixel::MODEL_FAMILY_SOURCE_ORDINAL,
        "zimagepixelspace_comfy_model_0154",
        &[
            "source-and-catalog-provenance",
            "model-store-prefixed-native-probe",
            "standalone-native-and-diffusers-rejection",
            "checked-rgb-patch-and-pixel-decoder-geometry",
            "pixel-latent-and-reference-memory-identity",
            "transactional-model-text-routing-and-unmatched-rejection",
            "named-native-forward-and-patch-order",
            "exact-memory-oom-dtype-device-cancellation",
            "sibling-precedence-partial-malformed-metadata-and-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}

fn standalone_state_source() -> Vec<support::TensorFixture> {
    [
        "x_embedder.weight",
        "x_embedder.bias",
        "cap_embedder.1.weight",
        "cap_embedder.1.bias",
        "noise_refiner.0.attention.k_norm.weight",
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "dec_net.cond_embed.weight",
        "dec_net.final_layer.linear.weight",
        "dec_net.input_embedder.embedder.0.weight",
        "text_encoders.qwen3_4b.transformer.weight",
    ]
    .into_iter()
    .enumerate()
    .map(|(index, key)| support::TensorFixture::new(key, &[1], &[index as f32 + 1.0]))
    .collect()
}
