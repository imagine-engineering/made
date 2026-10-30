use sim_model::{
    LTXV_CONDITIONING, LTX_CLIP_TARGET, LtxLayout, LtxVariant, ModelFamilyError,
    ModelFamilyRegistry, generated_ltxav_sim_model_0102 as ltxav,
    generated_ltxv_sim_model_0103 as ltxv, ltx_configuration_for_probe,
};

use super::generated_ltxav_sim_model_0102::support;

static REGISTRATIONS: [sim_model::ModelFamilyRegistration; 2] = [
    ltxv::MODEL_FAMILY_REGISTRATION,
    ltxav::MODEL_FAMILY_REGISTRATION,
];

#[test]
fn val_model_family_row_001_ltxv_source_detection_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(ltxv::MODEL_FAMILY_IDENTIFIER, "LTXV");
    assert_eq!(ltxv::MODEL_FAMILY_SOURCE_ORDINAL, 32);
    assert_eq!(ltxv::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.LTXV");
    let fixture = support::load_fixture(ltxv::MODEL_FAMILY_FIXTURE)?;
    let prefixed = support::probe(&fixture);
    for (probe, layout, target_prefix) in [
        (
            prefixed.clone(),
            LtxLayout::PrefixedNative,
            "model.diffusion_model.",
        ),
        (
            rewritten_probe(&prefixed, "model."),
            LtxLayout::SavedModel,
            "model.",
        ),
        (
            rewritten_probe(&prefixed, ""),
            LtxLayout::StandaloneNative,
            "",
        ),
    ] {
        let configuration = ltx_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, LtxVariant::Video);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.input_channels, 4);
        assert_eq!(configuration.inner_dimension, 64);
        assert_eq!(configuration.number_of_layers, 1);
        assert_eq!(configuration.attention_head_dimension, 2);
        assert_eq!(configuration.number_of_attention_heads, 32);
        assert_eq!(configuration.cross_attention_dimension, 2_048);
        assert_eq!(configuration.audio_input_channels, None);
        assert_eq!(configuration.memory_usage_factor, 5.5);
        assert_eq!(configuration.conditioning, LTXV_CONDITIONING);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0040");
        assert!(std::ptr::eq(configuration.clip_target, &LTX_CLIP_TARGET));
        support::exercise_state_plan(
            &REGISTRATIONS,
            ltxv::MODEL_FAMILY_FIXTURE,
            &probe,
            |key| rewrite_model_key(key, target_prefix),
        )?;
    }

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let detection = registry.detect(&prefixed)?;
    assert_eq!(detection.identity.feature_id(), ltxv::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(detection.score, 1_000);
    let mut pixart = prefixed.clone();
    pixart.tensor_shapes.insert(
        "model.diffusion_model.pos_embed.proj.bias".to_owned(),
        vec![64],
    );
    assert!(matches!(
        registry.resolve(&pixart),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("PixArt")
    ));
    let mut contradictory = prefixed;
    contradictory.metadata.insert(
        "config".to_owned(),
        r#"{"transformer":{"cross_attention_dim":4096}}"#.to_owned(),
    );
    assert!(matches!(
        registry.resolve(&contradictory),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("contradicts")
    ));
    support::validate_provenance(
        &ltxv::MODEL_FAMILY,
        ltxv::MODEL_FAMILY_FIXTURE,
        ltxv::MODEL_FAMILY_SOURCE_ORDINAL,
        "model_base.LTXV",
        ltxv::MODEL_FAMILY_PROJECTION_SHA256,
        ltxv::MODEL_FAMILY_SOURCE_PATH,
        ltxv::MODEL_FAMILY_SOURCE_SHA256,
    )?;
    Ok(())
}

#[test]
fn val_model_family_row_001_ltxv_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_registration(
        &REGISTRATIONS,
        ltxv::MODEL_FAMILY_FIXTURE,
        &[sim_tensor::DType::Bf16, sim_tensor::DType::F32],
    )?;
    super::write_model_family_row_artifact(
        ltxv::MODEL_FAMILY_FIXTURE,
        ltxv::MODEL_FAMILY_FEATURE_ID,
        ltxv::MODEL_FAMILY_IDENTIFIER,
        ltxv::MODEL_FAMILY_SOURCE_ORDINAL,
        "ltxv_comfy_model_0103",
        &[
            "source-provenance-registration-descriptor",
            "source-exact-video-configuration-and-profile",
            "prefixed-saved-model-and-standalone-layouts",
            "ltxv-ltxav-registry-precedence",
            "forward-checkpoints-and-patch-order",
            "bf16-f32-memory-device-oom-cancellation",
            "pixart-metadata-diffusers-and-owner-delegation",
        ],
    )?;
    Ok(())
}

fn rewritten_probe(probe: &sim_model::ModelProbe, target_prefix: &str) -> sim_model::ModelProbe {
    sim_model::ModelProbe {
        tensor_shapes: probe
            .tensor_shapes
            .iter()
            .map(|(key, shape)| {
                let key = key
                    .strip_prefix("model.diffusion_model.")
                    .map_or_else(|| key.clone(), |suffix| format!("{target_prefix}{suffix}"));
                (key, shape.clone())
            })
            .collect(),
        metadata: probe.metadata.clone(),
    }
}

fn rewrite_model_key(key: &str, target_prefix: &str) -> String {
    key.strip_prefix("model.diffusion_model.")
        .map_or_else(|| key.to_owned(), |suffix| format!("{target_prefix}{suffix}"))
}
