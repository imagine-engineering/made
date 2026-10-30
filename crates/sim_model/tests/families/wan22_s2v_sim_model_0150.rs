use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as wan,
    generated_wan22_s2v_sim_model_0150 as s2v,
};
static REGISTRATIONS: [ModelFamilyRegistration; 1] = [s2v::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition { feature_id: "COMFY-MODEL-9150", identifier: "WAN22_S2V_AmbiguousFixture", ..s2v::MODEL_FAMILY };
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [s2v::MODEL_FAMILY_REGISTRATION, ModelFamilyRegistration { definition: &AMBIGUOUS_DEFINITION, source_ordinal: 958, source_architecture: "model_base.WAN22_S2V_AmbiguousFixture", ..s2v::MODEL_FAMILY_REGISTRATION }];

#[test]
fn val_model_family_row_001_wan22_s2v_source_configuration_and_state_plan() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(s2v::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(s2v::MODEL_FAMILY_FIXTURE, s2v::MODEL_FAMILY_FEATURE_ID, s2v::MODEL_FAMILY_IDENTIFIER, s2v::MODEL_FAMILY_SOURCE_ORDINAL, s2v::SOURCE_ARCHITECTURE, s2v::MODEL_FAMILY_PROJECTION_SHA256)?;
    let probe = support::probe_through_model_store(&fixture)?; let configuration = s2v::configuration_for_probe(&probe)?;
    assert_eq!(configuration.model_type, "s2v"); assert_eq!(configuration.architecture_model_type, "t2v"); assert!(!configuration.image_to_video);
    assert!(configuration.audio_conditioning && configuration.reference_conditioning && configuration.motion_conditioning && configuration.control_conditioning);
    assert_eq!((configuration.input_channels, configuration.output_channels), (16, 16)); assert_eq!(describe_model_family(&s2v::MODEL_FAMILY)?.latent_format, "Wan21");
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?; assert_eq!(registry.resolve(&probe)?.detection().score, 1_000);
    let mut missing = probe.clone(); missing.tensor_shapes.remove(wan::CAUSAL_AUDIO_WEIGHT); assert!(registry.resolve(&missing).is_err());
    let mut malformed = probe.clone(); malformed.tensor_shapes.insert(wan::CAUSAL_AUDIO_WEIGHT.to_owned(), vec![127, 64]); assert!(s2v::configuration_for_probe(&malformed).is_err());
    let mut conflicting = probe.clone(); conflicting.tensor_shapes.insert(wan::AUDIO_BIAS.to_owned(), vec![128]); assert!(s2v::configuration_for_probe(&conflicting).is_err());
    let mut diffusers = probe.clone(); diffusers.metadata.insert("model_layout".to_owned(), "diffusers".to_owned()); assert!(matches!(s2v::configuration_for_probe(&diffusers), Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")));
    assert!(matches!(ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })));
    support::verify_owner_delegation("wan22_s2v_comfy_model_0150")?; Ok(())
}
#[test]
fn val_model_family_row_001_wan22_s2v_forward_patch_memory_and_platform() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(s2v::MODEL_FAMILY_FIXTURE)?; let extras = [support::TensorFixture::new("text_encoders.umt5xxl.transformer.block.weight", &[1], &[1.0]), support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0])];
    support::exercise_family(&fixture, &REGISTRATIONS, &extras, &["model", "runtime_conditioning", "text_encoder", "vae"], "native.blocks.0.ffn.2.weight")?;
    super::write_model_family_row_artifact(s2v::MODEL_FAMILY_FIXTURE, s2v::MODEL_FAMILY_FEATURE_ID, s2v::MODEL_FAMILY_IDENTIFIER, s2v::MODEL_FAMILY_SOURCE_ORDINAL, "wan22_s2v_comfy_model_0150", &["source-and-catalog-provenance", "causal-audio-source-precedence-detection", "shape-reduced-audio-reference-motion-configuration", "transactional-model-runtime-text-vae-routing", "named-native-forward-and-patch-order", "dynamic-memory-oom-dtype-device-cancellation", "diffusers-conflicting-malformed-ambiguity-failures", "authoritative-owner-delegation"])?; Ok(())
}
