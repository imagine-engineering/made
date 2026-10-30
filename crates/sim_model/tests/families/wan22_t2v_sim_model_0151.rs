use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as wan,
    generated_wan22_t2v_sim_model_0151 as t2v,
};
static REGISTRATIONS: [ModelFamilyRegistration; 1] = [t2v::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition { feature_id: "COMFY-MODEL-9151", identifier: "WAN22_T2V_AmbiguousFixture", ..t2v::MODEL_FAMILY };
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [t2v::MODEL_FAMILY_REGISTRATION, ModelFamilyRegistration { definition: &AMBIGUOUS_DEFINITION, source_ordinal: 950, source_architecture: "model_base.WAN22_AmbiguousFixture", ..t2v::MODEL_FAMILY_REGISTRATION }];

#[test]
fn val_model_family_row_001_wan22_t2v_source_configuration_and_state_plan() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(t2v::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(t2v::MODEL_FAMILY_FIXTURE, t2v::MODEL_FAMILY_FEATURE_ID, t2v::MODEL_FAMILY_IDENTIFIER, t2v::MODEL_FAMILY_SOURCE_ORDINAL, t2v::SOURCE_ARCHITECTURE, t2v::MODEL_FAMILY_PROJECTION_SHA256)?;
    let probe = support::probe_through_model_store(&fixture)?; let configuration = t2v::configuration_for_probe(&probe)?;
    assert_eq!(configuration.model_type, "t2v"); assert_eq!(configuration.architecture_model_type, "t2v"); assert!(configuration.image_to_video);
    assert_eq!((configuration.input_channels, configuration.output_channels), (16, 48)); assert_eq!(describe_model_family(&t2v::MODEL_FAMILY)?.latent_format, "Wan22");
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?; assert_eq!(registry.resolve(&probe)?.source_ordinal(), 50);
    let mut wan21 = probe.clone(); wan21.tensor_shapes.insert(wan::HEAD_WEIGHT.to_owned(), vec![64, 128]); assert!(t2v::configuration_for_probe(&wan21).is_err());
    let mut specialized = probe.clone(); specialized.tensor_shapes.insert(wan::FACE_ADAPTER_WEIGHT.to_owned(), vec![128]); assert!(t2v::configuration_for_probe(&specialized).is_err());
    let mut misleading = probe.clone(); misleading.metadata.insert("config".to_owned(), "{\"transformer\":{\"out_dim\":16}}".to_owned()); assert!(t2v::configuration_for_probe(&misleading).is_err());
    let mut diffusers = probe.clone(); diffusers.metadata.insert("model_layout".to_owned(), "diffusers".to_owned()); assert!(matches!(t2v::configuration_for_probe(&diffusers), Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")));
    assert!(matches!(ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })));
    support::verify_owner_delegation("wan22_t2v_comfy_model_0151")?; Ok(())
}
#[test]
fn val_model_family_row_001_wan22_t2v_forward_patch_memory_and_platform() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(t2v::MODEL_FAMILY_FIXTURE)?; let extras = [support::TensorFixture::new("text_encoders.umt5xxl.transformer.block.weight", &[1], &[1.0]), support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0])];
    support::exercise_family(&fixture, &REGISTRATIONS, &extras, &["model", "runtime_conditioning", "text_encoder", "vae"], "native.blocks.0.ffn.2.weight")?;
    super::write_model_family_row_artifact(t2v::MODEL_FAMILY_FIXTURE, t2v::MODEL_FAMILY_FEATURE_ID, t2v::MODEL_FAMILY_IDENTIFIER, t2v::MODEL_FAMILY_SOURCE_ORDINAL, "wan22_t2v_comfy_model_0151", &["source-and-catalog-provenance", "wan22-output-geometry-detection", "wan22-latent-image-to-video-configuration", "transactional-model-runtime-text-vae-routing", "named-native-forward-and-patch-order", "dynamic-memory-oom-dtype-device-cancellation", "diffusers-adapter-wan21-metadata-ambiguity-failures", "authoritative-owner-delegation"])?; Ok(())
}
