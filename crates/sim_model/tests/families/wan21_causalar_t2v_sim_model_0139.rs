use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_causalar_t2v_sim_model_0139 as causal,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [causal::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9139",
    identifier: "WAN21_CausalAR_T2V_AmbiguousFixture",
    ..causal::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    causal::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 951,
        source_architecture: "model_base.WAN21_CausalAR_AmbiguousFixture",
        ..causal::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_causalar_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(causal::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        causal::MODEL_FAMILY_FIXTURE,
        causal::MODEL_FAMILY_FEATURE_ID,
        causal::MODEL_FAMILY_IDENTIFIER,
        causal::MODEL_FAMILY_SOURCE_ORDINAL,
        causal::SOURCE_ARCHITECTURE,
        causal::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(
        probe.metadata().get("config").map(String::as_str),
        Some(causal::CAUSAL_CONFIG_METADATA)
    );
    let configuration = causal::configuration_for_probe(&probe)?;
    assert_eq!(configuration.image_model, "wan2.1");
    assert_eq!(configuration.model_type, "t2v");
    assert!(configuration.causal_ar);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 16);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.feed_forward_dimension, 256);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.camera_condition_channels, None);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&causal::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.clip_target().candidates().len(), 1);
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 51);

    let mut missing_metadata = probe.clone();
    missing_metadata.metadata.remove("config");
    assert!(registry.resolve(&missing_metadata).is_err());
    let mut false_metadata = probe.clone();
    false_metadata.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"causal_ar\":false}}".to_owned(),
    );
    assert!(registry.resolve(&false_metadata).is_err());
    let mut camera = probe.clone();
    camera.tensor_shapes.insert(
        "model.diffusion_model.control_adapter.conv.weight".to_owned(),
        vec![128, 1_536, 2, 2],
    );
    assert!(matches!(
        causal::configuration_for_probe(&camera),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("camera")
    ));
    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.patch_embedding.weight".to_owned(),
        vec![128, 32, 1, 2, 2],
    );
    assert!(matches!(
        causal::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        causal::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("wan21_causalar_t2v_comfy_model_0139")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_causalar_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(causal::MODEL_FAMILY_FIXTURE)?;
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
        causal::MODEL_FAMILY_FIXTURE,
        causal::MODEL_FAMILY_FEATURE_ID,
        causal::MODEL_FAMILY_IDENTIFIER,
        causal::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_causalar_t2v_comfy_model_0139",
        &[
            "source-and-catalog-provenance",
            "explicit-causal-config-selection",
            "shape-reduced-wan-causal-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-metadata-camera-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
