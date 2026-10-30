use sim_model::{
    ModelFamilyError, ModelProbe, PixelDitPidLayout, PixelDitPidVariant,
    generated_pixelditt2i_sim_model_0112 as pixeldit, pixeldit_pid_state_plan_for_layout,
};
use std::collections::BTreeMap;

use super::generated_lumina2_sim_model_0107::support;

#[test]
fn source_projection_descriptor_fixture_and_fail_closed_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(pixeldit::MODEL_FAMILY_IDENTIFIER, "PixelDiTT2I");
    assert_eq!(pixeldit::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0112");
    assert_eq!(pixeldit::MODEL_FAMILY_SOURCE_ORDINAL, 48);
    assert_eq!(
        pixeldit::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.PixelDiTT2I"
    );
    assert_eq!(pixeldit::MODEL_FAMILY_SAMPLING_SHIFT, 4.0);
    support::validate_provenance(
        pixeldit::MODEL_FAMILY_FIXTURE,
        pixeldit::MODEL_FAMILY_FEATURE_ID,
        pixeldit::MODEL_FAMILY_IDENTIFIER,
        48,
        pixeldit::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    support::exercise_fixture(
        &pixeldit::MODEL_FAMILY,
        pixeldit::MODEL_FAMILY_FIXTURE,
        "pixelditt2i_comfy_model_0112",
        48,
    )?;
    Ok(())
}

#[test]
fn core_and_net_layouts_execute_branching_and_pid_or_partial_probes_fail()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        PixelDitPidLayout::CoreNative,
        PixelDitPidLayout::NetNative,
    ] {
        let configuration = pixeldit::configuration_for_probe(&probe(layout, false))?;
        assert_eq!(configuration.variant, PixelDitPidVariant::PixelDitT2I);
        assert_eq!(configuration.layout, layout);
        assert!(configuration.pid.is_none());
        assert_eq!(configuration.sampling_shift, 4.0);
        assert_eq!(configuration.conditioning_keys.len(), 1);
        support::exercise_plan(
            pixeldit_pid_state_plan_for_layout(layout),
            &mapping_keys(layout),
            &[
                "native.pixel_blocks.0.adaLN_modulation_msa.weight",
                "native.pixel_blocks.0.adaLN_modulation_mlp.weight",
                "native.final_layer.linear.weight",
            ],
        )?;
    }
    assert!(matches!(
        pixeldit::configuration_for_probe(&probe(PixelDitPidLayout::CoreNative, true)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("cannot admit")
    ));
    let mut partial = probe(PixelDitPidLayout::NetNative, false);
    partial.tensor_shapes.remove("net.final_layer.linear.weight");
    assert!(matches!(
        pixeldit::configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut ambiguous = probe(PixelDitPidLayout::CoreNative, false);
    ambiguous
        .tensor_shapes
        .extend(probe(PixelDitPidLayout::NetNative, false).tensor_shapes);
    assert!(matches!(
        pixeldit::configuration_for_probe(&ambiguous),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
    support::assert_leaf_owner("pixelditt2i_comfy_model_0112", "pixeldit_pid_family")?;
    Ok(())
}

fn probe(layout: PixelDitPidLayout, include_lq: bool) -> ModelProbe {
    let prefix = prefix(layout);
    let modulation = 6 * 16 * 16 * 16;
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}pixel_embedder.proj.weight"),
            vec![16, 3],
        ),
        (
            format!("{prefix}s_embedder.proj.weight"),
            vec![1_536, 768],
        ),
        (
            format!("{prefix}y_embedder.proj.weight"),
            vec![1_536, 2_304],
        ),
        (
            format!("{prefix}final_layer.linear.weight"),
            vec![3, 16],
        ),
    ]);
    for index in 0..14 {
        tensor_shapes.insert(
            format!("{prefix}patch_blocks.{index}.attn.qkv_x.weight"),
            vec![4_608, 1_536],
        );
    }
    for index in 0..2 {
        tensor_shapes.insert(
            format!("{prefix}pixel_blocks.{index}.adaLN_modulation.0.weight"),
            vec![modulation, 1_536],
        );
    }
    if include_lq {
        tensor_shapes.insert(
            format!("{prefix}lq_proj.latent_proj.0.weight"),
            vec![512, 64, 3, 3],
        );
        for index in 0..7 {
            tensor_shapes.insert(
                format!("{prefix}lq_proj.gate_modules.{index}.content_proj.weight"),
                vec![1_536, 3_072],
            );
        }
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn mapping_keys(layout: PixelDitPidLayout) -> Vec<(String, Vec<u64>)> {
    let prefix = prefix(layout);
    vec![
        (
            format!("{prefix}pixel_embedder.proj.weight"),
            vec![2, 3],
        ),
        (format!("{prefix}s_embedder.proj.weight"), vec![2, 2]),
        (format!("{prefix}y_embedder.proj.weight"), vec![2, 2]),
        (
            format!("{prefix}patch_blocks.0.attn.qkv_x.weight"),
            vec![2, 2],
        ),
        (
            format!("{prefix}pixel_blocks.0.adaLN_modulation.0.weight"),
            vec![24, 2],
        ),
        (
            format!("{prefix}pixel_blocks.0.adaLN_modulation.0.bias"),
            vec![24],
        ),
        (
            format!("{prefix}final_layer.linear.weight"),
            vec![2, 2],
        ),
        ("_repa_projector.weight".to_owned(), vec![1]),
        ("net_ema.shadow".to_owned(), vec![1]),
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.gemma.weight".to_owned(), vec![1]),
    ]
}

fn prefix(layout: PixelDitPidLayout) -> &'static str {
    match layout {
        PixelDitPidLayout::CoreNative => "core.",
        PixelDitPidLayout::NetNative => "net.",
    }
}
