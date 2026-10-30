use sim_model::{
    KANDINSKY5_IMAGE_CLIP_TARGET, KANDINSKY5_IMAGE_CONDITIONING,
    KANDINSKY5_IMAGE_LATENT_FORMAT, Kandinsky5Layout, Kandinsky5Variant, ModelFamilyError,
    ModelFamilyRegistry, ModelProbe, generated_kandinsky5image_sim_model_0100 as image,
    kandinsky5_configuration_for_probe,
};
use std::collections::BTreeMap;

use super::generated_ltxav_sim_model_0102::support;

static REGISTRATIONS: [sim_model::ModelFamilyRegistration; 1] =
    [image::MODEL_FAMILY_REGISTRATION];

#[test]
fn val_model_family_row_001_kandinsky5image_source_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(image::MODEL_FAMILY_IDENTIFIER, "Kandinsky5Image");
    assert_eq!(image::MODEL_FAMILY_SOURCE_ORDINAL, 82);
    assert_eq!(
        image::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.Kandinsky5Image"
    );
    let fixture = support::load_fixture(image::MODEL_FAMILY_FIXTURE)?;
    let prefixed = support::probe(&fixture);
    for (probe, layout, target_prefix) in [
        (
            prefixed.clone(),
            Kandinsky5Layout::PrefixedNative,
            "model.diffusion_model.",
        ),
        (
            standalone_probe(&prefixed),
            Kandinsky5Layout::StandaloneNative,
            "",
        ),
    ] {
        let configuration = kandinsky5_configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, Kandinsky5Variant::ImageLite);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.model_dimension, 2_560);
        assert_eq!(configuration.time_dimension, 512);
        assert_eq!(configuration.feed_forward_dimension, 10_240);
        assert_eq!(configuration.visual_embed_dimension, 64);
        assert_eq!(configuration.patch_size, [1, 2, 2]);
        assert_eq!(configuration.text_block_count, 2);
        assert_eq!(configuration.visual_block_count, 32);
        assert_eq!(configuration.axes_dimensions, [32, 48, 48]);
        assert_eq!(configuration.rope_scale_factor, [1.0, 1.0, 1.0]);
        assert!(!configuration.concat_conditioning);
        assert_eq!(configuration.conditioning, KANDINSKY5_IMAGE_CONDITIONING);
        assert_eq!(configuration.sampling_shift, 3.0);
        assert_eq!(configuration.memory_usage_factor, 1.25);
        assert_eq!(
            configuration.latent_format.feature_id,
            KANDINSKY5_IMAGE_LATENT_FORMAT.feature_id
        );
        assert_eq!(
            configuration.latent_format.identifier,
            KANDINSKY5_IMAGE_LATENT_FORMAT.identifier
        );
        assert!(std::ptr::eq(
            configuration.clip_target,
            &KANDINSKY5_IMAGE_CLIP_TARGET
        ));
        support::exercise_state_plan(
            &REGISTRATIONS,
            image::MODEL_FAMILY_FIXTURE,
            &probe,
            |key| rewrite_model_prefix(key, target_prefix),
        )?;
    }
    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    let detection = registry.detect(&prefixed)?;
    assert_eq!(detection.identity.feature_id(), image::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(detection.score, 1_000);
    let mut misleading = prefixed;
    misleading
        .metadata
        .insert("image_model".to_owned(), "kandinsky5-video".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        image::MODEL_FAMILY_FEATURE_ID
    );
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "transformer_blocks.0.cross_attention.key_norm.weight".to_owned(),
            vec![64],
        )]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        kandinsky5_configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    support::validate_provenance(
        &image::MODEL_FAMILY,
        image::MODEL_FAMILY_FIXTURE,
        image::MODEL_FAMILY_SOURCE_ORDINAL,
        "model_base.Kandinsky5Image",
        image::MODEL_FAMILY_PROJECTION_SHA256,
        image::MODEL_FAMILY_SOURCE_PATH,
        image::MODEL_FAMILY_SOURCE_SHA256,
    )?;
    Ok(())
}

fn rewrite_model_prefix(key: &str, target_prefix: &str) -> String {
    key.strip_prefix("model.diffusion_model.")
        .map_or_else(|| key.to_owned(), |suffix| format!("{target_prefix}{suffix}"))
}

#[test]
fn val_model_family_row_001_kandinsky5image_mapping_forward_patch_memory_and_platform()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_registration(
        &REGISTRATIONS,
        image::MODEL_FAMILY_FIXTURE,
        &[sim_tensor::DType::Bf16, sim_tensor::DType::F32],
    )?;
    super::write_model_family_row_artifact(
        image::MODEL_FAMILY_FIXTURE,
        image::MODEL_FAMILY_FEATURE_ID,
        image::MODEL_FAMILY_IDENTIFIER,
        image::MODEL_FAMILY_SOURCE_ORDINAL,
        "kandinsky5image_comfy_model_0100",
        &[
            "source-provenance-registration-descriptor",
            "source-exact-image-configuration-and-profile",
            "prefixed-and-standalone-native-layouts",
            "flux-latent-and-image-tokenizer-selection",
            "forward-checkpoints-and-patch-order",
            "bf16-f32-memory-device-oom-cancellation",
            "video-diffusers-misleading-and-owner-delegation",
        ],
    )?;
    Ok(())
}

fn standalone_probe(probe: &ModelProbe) -> ModelProbe {
    ModelProbe {
        tensor_shapes: probe
            .tensor_shapes
            .iter()
            .map(|(key, shape)| {
                (
                    key.strip_prefix("model.diffusion_model.")
                        .unwrap_or(key)
                        .to_owned(),
                    shape.clone(),
                )
            })
            .collect(),
        metadata: probe.metadata.clone(),
    }
}
