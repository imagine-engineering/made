use sim_model::{
    LOTUS_CONDITIONING, ModelFamilyRegistry, ModelProbe, Sd2Layout, Sd2ModelType, Sd2Variant,
    generated_lotusd_sim_model_0106 as lotus,
    generated_sd20_sim_model_0119 as sd20, lotus_task_embedding,
};

use super::{
    generated_ltxav_sim_model_0102::support,
    generated_sd20_sim_model_0119::{standard_diffusers_probe, standard_native_probe},
};

static LOTUS_REGISTRATIONS: [sim_model::ModelFamilyRegistration; 1] =
    [lotus::MODEL_FAMILY_REGISTRATION];
static SD2_REGISTRATIONS: [sim_model::ModelFamilyRegistration; 2] = [
    lotus::MODEL_FAMILY_REGISTRATION,
    sd20::MODEL_FAMILY_REGISTRATION,
];

#[test]
fn val_model_family_row_001_lotusd_source_detection_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(lotus::MODEL_FAMILY_IDENTIFIER, "LotusD");
    assert_eq!(lotus::MODEL_FAMILY_SOURCE_ORDINAL, 0);
    assert_eq!(lotus::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.Lotus");
    for (probe, layout) in [
        (standard_native_probe(Some(4), false, 4), Sd2Layout::PrefixedNative),
        (standard_diffusers_probe(Some(4), false), Sd2Layout::Diffusers),
    ] {
        let configuration = lotus::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, Sd2Variant::LotusD);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.model_type, Sd2ModelType::ImgToImg);
        assert_eq!(configuration.input_channels, 4);
        assert_eq!(configuration.output_channels, 4);
        assert_eq!(configuration.model_channels, 320);
        assert_eq!(configuration.context_dimension, 1_024);
        assert_eq!(configuration.adm_in_channels, Some(4));
        assert_eq!(configuration.conditioning, LOTUS_CONDITIONING);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0045");
    }
    assert_eq!(lotus_task_embedding(), [1.0_f32.sin(), 0.0, 1.0_f32.cos(), 1.0]);

    let standalone = standalone_probe(&standard_native_probe(Some(4), false, 4));
    let standalone_configuration = lotus::configuration_for_probe(&standalone)?;
    assert_eq!(standalone_configuration.variant, Sd2Variant::LotusD);
    assert_eq!(standalone_configuration.model_type, Sd2ModelType::ImgToImg);
    assert_eq!(standalone_configuration.conditioning, LOTUS_CONDITIONING);

    let fixture = support::load_fixture(lotus::MODEL_FAMILY_FIXTURE)?;
    let probe = support::probe(&fixture);
    let registry = ModelFamilyRegistry::checked_registrations(&SD2_REGISTRATIONS)?;
    let detection = registry.detect(&probe)?;
    assert_eq!(detection.identity.feature_id(), lotus::MODEL_FAMILY_FEATURE_ID);
    assert_eq!(detection.score, 1_400);
    assert_eq!(
        registry.resolve(&standalone)?.detection().identity.feature_id(),
        lotus::MODEL_FAMILY_FEATURE_ID
    );
    support::exercise_state_plan(
        &LOTUS_REGISTRATIONS,
        lotus::MODEL_FAMILY_FIXTURE,
        &probe,
        str::to_owned,
    )?;
    support::exercise_state_plan(
        &LOTUS_REGISTRATIONS,
        lotus::MODEL_FAMILY_FIXTURE,
        &standalone,
        standalone_key,
    )?;
    support::exercise_state_plan(
        &LOTUS_REGISTRATIONS,
        lotus::MODEL_FAMILY_FIXTURE,
        &standard_diffusers_probe(Some(4), false),
        diffusers_key,
    )?;
    let sd20_probe = standard_native_probe(None, false, 4);
    assert_eq!(
        registry.detect(&sd20_probe)?.identity.feature_id(),
        sd20::MODEL_FAMILY_FEATURE_ID
    );
    let mut malformed = probe;
    malformed.tensor_shapes.insert(
        "model.diffusion_model.label_emb.0.0.weight".to_owned(),
        vec![1_280, 5],
    );
    assert_eq!(
        registry.detect(&malformed)?.identity.feature_id(),
        sd20::MODEL_FAMILY_FEATURE_ID
    );
    support::validate_provenance(
        &lotus::MODEL_FAMILY,
        lotus::MODEL_FAMILY_FIXTURE,
        lotus::MODEL_FAMILY_SOURCE_ORDINAL,
        "model_base.Lotus",
        lotus::MODEL_FAMILY_PROJECTION_SHA256,
        lotus::MODEL_FAMILY_SOURCE_PATH,
        lotus::MODEL_FAMILY_SOURCE_SHA256,
    )?;
    Ok(())
}

fn standalone_probe(probe: &ModelProbe) -> ModelProbe {
    ModelProbe {
        tensor_shapes: probe
            .tensor_shapes
            .iter()
            .map(|(key, shape)| (standalone_key(key), shape.clone()))
            .collect(),
        metadata: probe.metadata.clone(),
    }
}

fn standalone_key(key: &str) -> String {
    key.strip_prefix("model.diffusion_model.")
        .unwrap_or(key)
        .to_owned()
}

fn diffusers_key(key: &str) -> String {
    match key {
        "model.diffusion_model.input_blocks.0.0.weight" => "conv_in.weight",
        "model.diffusion_model.time_embed.0.weight" => "time_embedding.linear_1.weight",
        "model.diffusion_model.time_embed.0.bias" => "time_embedding.linear_1.bias",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight" => {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        }
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight" => {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"
        }
        "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight" => {
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight"
        }
        "model.diffusion_model.out.2.weight" => "conv_out.weight",
        "model.diffusion_model.label_emb.0.0.weight" => "class_embedding.linear_1.weight",
        _ => key,
    }
    .to_owned()
}

#[test]
fn val_model_family_row_001_lotusd_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_registration(
        &LOTUS_REGISTRATIONS,
        lotus::MODEL_FAMILY_FIXTURE,
        &[
            sim_tensor::DType::F16,
            sim_tensor::DType::Bf16,
            sim_tensor::DType::F32,
        ],
    )?;
    super::write_model_family_row_artifact(
        lotus::MODEL_FAMILY_FIXTURE,
        lotus::MODEL_FAMILY_FEATURE_ID,
        lotus::MODEL_FAMILY_IDENTIFIER,
        lotus::MODEL_FAMILY_SOURCE_ORDINAL,
        "lotusd_comfy_model_0106",
        &[
            "source-provenance-registration-descriptor",
            "source-exact-lotus-configuration-and-profile",
            "prefixed-standalone-and-diffusers-state-plans",
            "lotusd-sd20-registry-precedence",
            "task-embedding-forward-and-patch-order",
            "f16-bf16-f32-memory-device-oom-cancellation",
            "partial-malformed-statistic-and-owner-delegation",
        ],
    )?;
    Ok(())
}
