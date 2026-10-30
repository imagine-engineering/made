use super::generated_stable_zero123_sim_model_0136::support;
use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    describe_model_family, generated_wan21_t2v_sim_model_0146 as wan,
    generated_wan22_wandancer_sim_model_0152 as dancer,
};
static REGISTRATIONS: [ModelFamilyRegistration; 1] = [dancer::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition { feature_id: "COMFY-MODEL-9152", identifier: "WAN22_WanDancer_AmbiguousFixture", ..dancer::MODEL_FAMILY };
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [dancer::MODEL_FAMILY_REGISTRATION, ModelFamilyRegistration { definition: &AMBIGUOUS_DEFINITION, source_ordinal: 964, source_architecture: "model_base.WAN22_WanDancer_AmbiguousFixture", ..dancer::MODEL_FAMILY_REGISTRATION }];

#[test]
fn val_model_family_row_001_wan22_wandancer_source_configuration_and_state_plan() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(dancer::MODEL_FAMILY_FIXTURE)?;
    support::verify_provenance(dancer::MODEL_FAMILY_FIXTURE, dancer::MODEL_FAMILY_FEATURE_ID, dancer::MODEL_FAMILY_IDENTIFIER, dancer::MODEL_FAMILY_SOURCE_ORDINAL, dancer::SOURCE_ARCHITECTURE, dancer::MODEL_FAMILY_PROJECTION_SHA256)?;
    let probe = support::probe_through_model_store(&fixture)?; let configuration = dancer::configuration_for_probe(&probe)?;
    assert_eq!(configuration.model_type, "wandancer"); assert_eq!(configuration.architecture_model_type, "i2v"); assert!(configuration.image_to_video);
    assert!(configuration.audio_conditioning && configuration.reference_conditioning && configuration.music_conditioning);
    assert_eq!((configuration.input_channels, configuration.output_channels), (36, 16)); assert_eq!(configuration.memory_usage_factor, 1.8); assert_eq!(describe_model_family(&dancer::MODEL_FAMILY)?.latent_format, "Wan21");
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?; assert_eq!(registry.resolve(&probe)?.detection().score, 1_200);
    let mut malformed = probe.clone(); malformed.tensor_shapes.insert(wan::GLOBAL_PATCH_WEIGHT.to_owned(), vec![128, 35, 1, 2, 2]); assert!(dancer::configuration_for_probe(&malformed).is_err());
    let mut conflicting = probe.clone(); conflicting.tensor_shapes.insert(wan::MASK_PATCH_WEIGHT.to_owned(), vec![128, 28, 1, 2, 2]); assert!(dancer::configuration_for_probe(&conflicting).is_err());
    let mut diffusers = probe.clone(); diffusers.metadata.insert("model_layout".to_owned(), "diffusers".to_owned()); assert!(matches!(dancer::configuration_for_probe(&diffusers), Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")));
    assert!(matches!(ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&probe), Err(ModelFamilyError::AmbiguousDetection { score: 1_200, .. })));
    support::verify_owner_delegation("wan22_wandancer_comfy_model_0152")?; Ok(())
}
#[test]
fn val_model_family_row_001_wan22_wandancer_forward_patch_memory_and_platform() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::load_fixture(dancer::MODEL_FAMILY_FIXTURE)?; let extras = [support::TensorFixture::new("text_encoders.umt5xxl.transformer.block.weight", &[1], &[1.0]), support::TensorFixture::new("vae.decoder.weight", &[1], &[1.0]), support::TensorFixture::new("model.diffusion_model.music_encoder.0.self_attn.in_proj_weight", &[6, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])];
    support::exercise_family(&fixture, &REGISTRATIONS, &extras, &["model", "runtime_conditioning", "text_encoder", "vae"], "native.blocks.0.ffn.2.weight")?;
    super::write_model_family_row_artifact(dancer::MODEL_FAMILY_FIXTURE, dancer::MODEL_FAMILY_FEATURE_ID, dancer::MODEL_FAMILY_IDENTIFIER, dancer::MODEL_FAMILY_SOURCE_ORDINAL, "wan22_wandancer_comfy_model_0152", &["source-and-catalog-provenance", "global-patch-source-precedence-detection", "shape-reduced-music-image-configuration", "transactional-qkv-model-runtime-text-vae-routing", "named-native-forward-and-patch-order", "fixed-memory-oom-dtype-device-cancellation", "diffusers-conflicting-malformed-ambiguity-failures", "authoritative-owner-delegation"])?; Ok(())
}
