use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as wan,
    generated_wan22_animate_sim_model_0148 as animate,
};

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [animate::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9148", identifier: "WAN22_Animate_AmbiguousFixture", ..animate::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    animate::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration { definition: &AMBIGUOUS_DEFINITION, source_ordinal: 960, source_architecture: "model_base.WAN22_Animate_AmbiguousFixture", ..animate::MODEL_FAMILY_REGISTRATION },
];

#[test]
fn val_model_family_row_001_wan22_animate_source_configuration_and_state_plan() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(animate::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(animate::MODEL_FAMILY_FIXTURE, animate::MODEL_FAMILY_FEATURE_ID, animate::MODEL_FAMILY_IDENTIFIER, animate::MODEL_FAMILY_SOURCE_ORDINAL, animate::SOURCE_ARCHITECTURE, animate::MODEL_FAMILY_PROJECTION_SHA256)?;
    let probe = support::probe_through_model_store(&fixture)?;
    let configuration = animate::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, animate::Wan22BatchVariant::Animate);
    assert_eq!(configuration.model_type, "animate");
    assert_eq!(configuration.architecture_model_type, "i2v");
    assert!(!configuration.image_to_video);
    assert!(configuration.face_conditioning && configuration.pose_conditioning);
    assert_eq!((configuration.dimension, configuration.input_channels, configuration.output_channels), (128, 16, 16));
    assert!((configuration.memory_usage_factor - 128.0 / 2_222.0).abs() < f64::EPSILON);
    assert_eq!(describe_model_family(&animate::MODEL_FAMILY)?.latent_format, "Wan21");
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    assert_eq!(registry.resolve(&probe)?.detection().score, 1_200);
    let mut missing = probe.clone(); missing.tensor_shapes.remove(wan::FACE_ADAPTER_WEIGHT); assert!(registry.resolve(&missing).is_err());
    let mut malformed = probe.clone(); malformed.tensor_shapes.insert(animate::ANIMATE_POSE_WEIGHT.to_owned(), vec![128, 15, 1, 2, 2]);
    assert!(matches!(animate::configuration_for_probe(&malformed), Err(ModelFamilyError::InvalidSelectorOutput(_))));
    let mut conflicting = probe.clone(); conflicting.tensor_shapes.insert(wan::CONTROL_WEIGHT.to_owned(), vec![128, 24, 2, 2]);
    assert!(animate::configuration_for_probe(&conflicting).is_err());
    let mut diffusers = probe.clone(); diffusers.metadata.insert("model_layout".to_owned(), "diffusers".to_owned());
    assert!(matches!(animate::configuration_for_probe(&diffusers), Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")));
    assert!(matches!(ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })));
    support::verify_owner_delegation("wan22_animate_comfy_model_0148")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_wan22_animate_forward_patch_memory_and_platform() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(animate::MODEL_FAMILY_FIXTURE)?;
    let extras = [support::TensorFixture::new("text_encoders.umt5xxl.transformer.block.weight", &[1], &[1.0]), support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0])];
    support::exercise_family(&fixture, &REGISTRATIONS, &extras, &["model", "runtime_conditioning", "text_encoder", "vae"], "native.blocks.0.ffn.2.weight")?;
    super::write_model_family_row_artifact(animate::MODEL_FAMILY_FIXTURE, animate::MODEL_FAMILY_FEATURE_ID, animate::MODEL_FAMILY_IDENTIFIER, animate::MODEL_FAMILY_SOURCE_ORDINAL, "wan22_animate_comfy_model_0148", &["source-and-catalog-provenance", "face-adapter-source-precedence-detection", "shape-reduced-face-pose-configuration", "transactional-model-runtime-text-vae-routing", "named-native-forward-and-patch-order", "dynamic-memory-oom-dtype-device-cancellation", "diffusers-conflicting-malformed-ambiguity-failures", "authoritative-owner-delegation"])?;
    Ok(())
}
