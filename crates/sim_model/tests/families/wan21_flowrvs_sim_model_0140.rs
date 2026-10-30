use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_flowrvs_sim_model_0140 as flow,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [flow::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9140",
    identifier: "WAN21_FlowRVS_AmbiguousFixture",
    ..flow::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    flow::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 961,
        source_architecture: "model_base.WAN21_FlowRVS_AmbiguousFixture",
        ..flow::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_wan21_flowrvs_source_configuration_and_state_plan()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(flow::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(
        flow::MODEL_FAMILY_FIXTURE,
        flow::MODEL_FAMILY_FEATURE_ID,
        flow::MODEL_FAMILY_IDENTIFIER,
        flow::MODEL_FAMILY_SOURCE_ORDINAL,
        flow::SOURCE_ARCHITECTURE,
        flow::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    let probe = support::probe_through_model_store(&fixture)?;
    assert_eq!(
        probe.metadata().get("config").map(String::as_str),
        Some(flow::FLOW_RVS_CONFIG_METADATA)
    );
    let configuration = flow::configuration_for_probe(&probe)?;
    assert_eq!(configuration.image_model, "wan2.1");
    assert_eq!(configuration.model_type, "flow_rvs");
    assert_eq!(configuration.architecture_model_type, "t2v");
    assert!(configuration.image_to_video);
    assert!(!configuration.audio_conditioning);
    assert_eq!(configuration.dimension, 128);
    assert_eq!(configuration.input_channels, 16);
    assert_eq!(configuration.output_channels, 16);
    assert_eq!(configuration.attention_heads, 1);
    assert_eq!(configuration.feed_forward_dimension, 256);
    assert_eq!(configuration.layer_count, 1);
    assert_eq!(configuration.patch_size, [1, 2, 2]);
    assert_eq!(configuration.frequency_dimension, 256);
    assert!(configuration.qk_norm);
    assert!(configuration.cross_attention_norm);
    assert_eq!(configuration.epsilon_millionths, 1);
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);

    let descriptor = describe_model_family(&flow::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.component_graph.len(), 4);
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let resolved = registry.resolve(&probe)?;
    assert_eq!(resolved.detection().score, 1_000);
    assert_eq!(resolved.source_ordinal(), 61);
    assert_eq!(resolved.clip_target().candidates().len(), 1);

    let mut missing_metadata = probe.clone();
    missing_metadata.metadata.remove("config");
    assert!(registry.resolve(&missing_metadata).is_err());
    let mut wrong_metadata = probe.clone();
    wrong_metadata.metadata.insert(
        "config".to_owned(),
        "{\"transformer\":{\"model_type\":\"t2v\"}}".to_owned(),
    );
    assert!(registry.resolve(&wrong_metadata).is_err());
    let mut malformed = probe.clone();
    malformed.tensor_shapes.insert(
        "model.diffusion_model.patch_embedding.weight".to_owned(),
        vec![128, 16, 2, 2],
    );
    assert!(matches!(
        flow::configuration_for_probe(&malformed),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    let mut diffusers = probe.clone();
    diffusers
        .metadata
        .insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(
        flow::configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("wan21_flowrvs_comfy_model_0140")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan21_flowrvs_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(flow::MODEL_FAMILY_FIXTURE)?;
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
        flow::MODEL_FAMILY_FIXTURE,
        flow::MODEL_FAMILY_FEATURE_ID,
        flow::MODEL_FAMILY_IDENTIFIER,
        flow::MODEL_FAMILY_SOURCE_ORDINAL,
        "wan21_flowrvs_comfy_model_0140",
        &[
            "source-and-catalog-provenance",
            "explicit-flow-rvs-config-selection",
            "shape-reduced-reverse-flow-configuration",
            "transactional-model-runtime-text-vae-routing",
            "named-native-forward-and-patch-order",
            "dynamic-memory-oom-dtype-device-cancellation",
            "diffusers-metadata-malformed-ambiguity-failures",
            "authoritative-owner-delegation",
        ],
    )?;
    Ok(())
}
