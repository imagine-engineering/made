use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as wan,
    generated_wan22_camera_sim_model_0149 as camera,
};
static REGISTRATIONS: [ModelFamilyRegistration; 1] = [camera::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition { feature_id: "COMFY-MODEL-9149", identifier: "WAN22_Camera_AmbiguousFixture", ..camera::MODEL_FAMILY };
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [camera::MODEL_FAMILY_REGISTRATION, ModelFamilyRegistration { definition: &AMBIGUOUS_DEFINITION, source_ordinal: 957, source_architecture: "model_base.WAN22_Camera_AmbiguousFixture", ..camera::MODEL_FAMILY_REGISTRATION }];

#[test]
fn val_model_family_row_001_wan22_camera_source_configuration_and_state_plan() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(camera::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(camera::MODEL_FAMILY_FIXTURE, camera::MODEL_FAMILY_FEATURE_ID, camera::MODEL_FAMILY_IDENTIFIER, camera::MODEL_FAMILY_SOURCE_ORDINAL, camera::SOURCE_ARCHITECTURE, camera::MODEL_FAMILY_PROJECTION_SHA256)?;
    let probe = support::probe_through_model_store(&fixture)?; let configuration = camera::configuration_for_probe(&probe)?;
    assert_eq!(configuration.model_type, "camera_2.2"); assert_eq!(configuration.architecture_model_type, "t2v"); assert!(!configuration.image_to_video);
    assert_eq!((configuration.input_channels, configuration.output_channels, configuration.camera_condition_channels), (36, 16, Some(24)));
    assert!(configuration.control_conditioning); assert_eq!(describe_model_family(&camera::MODEL_FAMILY)?.latent_format, "Wan21");
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?; assert_eq!(registry.resolve(&probe)?.source_ordinal(), 57);
    let mut image_camera = probe.clone(); image_camera.tensor_shapes.insert(wan::IMAGE_BIAS.to_owned(), vec![128]); assert!(camera::configuration_for_probe(&image_camera).is_err());
    let mut malformed = probe.clone(); malformed.tensor_shapes.insert(wan::CONTROL_WEIGHT.to_owned(), vec![128, 25, 2, 2]); assert!(camera::configuration_for_probe(&malformed).is_err());
    let mut misleading = probe.clone(); misleading.metadata.insert("config".to_owned(), "{\"transformer\":{\"model_type\":\"camera\"}}".to_owned()); assert!(camera::configuration_for_probe(&misleading).is_err());
    let mut diffusers = probe.clone(); diffusers.metadata.insert("model_layout".to_owned(), "diffusers".to_owned()); assert!(matches!(camera::configuration_for_probe(&diffusers), Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")));
    assert!(matches!(ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })));
    support::verify_owner_delegation("wan22_camera_comfy_model_0149")?; Ok(())
}
#[test]
fn val_model_family_row_001_wan22_camera_forward_patch_memory_and_platform() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(camera::MODEL_FAMILY_FIXTURE)?; let extras = [support::TensorFixture::new("text_encoders.umt5xxl.transformer.block.weight", &[1], &[1.0]), support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0])];
    support::exercise_family(&fixture, &REGISTRATIONS, &extras, &["model", "runtime_conditioning", "text_encoder", "vae"], "native.blocks.0.ffn.2.weight")?;
    super::write_model_family_row_artifact(camera::MODEL_FAMILY_FIXTURE, camera::MODEL_FAMILY_FEATURE_ID, camera::MODEL_FAMILY_IDENTIFIER, camera::MODEL_FAMILY_SOURCE_ORDINAL, "wan22_camera_comfy_model_0149", &["source-and-catalog-provenance", "camera22-absence-discriminator-detection", "shape-reduced-camera22-configuration", "transactional-model-runtime-text-vae-routing", "named-native-forward-and-patch-order", "dynamic-memory-oom-dtype-device-cancellation", "diffusers-image-bias-malformed-ambiguity-failures", "authoritative-owner-delegation"])?; Ok(())
}
