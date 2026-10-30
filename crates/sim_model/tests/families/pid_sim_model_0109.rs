use sim_model::{
    ModelFamilyError, ModelProbe, PixelDitPidLayout, PixelDitPidVariant,
    generated_pid_sim_model_0109 as pid, pixeldit_pid_state_plan_for_layout,
};
use std::collections::BTreeMap;

use super::generated_lumina2_sim_model_0107::support;

#[test]
fn source_projection_descriptor_fixture_and_fail_closed_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(pid::MODEL_FAMILY_IDENTIFIER, "PiD");
    assert_eq!(pid::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0109");
    assert_eq!(pid::MODEL_FAMILY_SOURCE_ORDINAL, 47);
    assert_eq!(pid::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.PiD");
    assert_eq!(pid::MODEL_FAMILY_SAMPLING_SHIFT, 1.5);
    support::validate_provenance(
        pid::MODEL_FAMILY_FIXTURE,
        pid::MODEL_FAMILY_FEATURE_ID,
        pid::MODEL_FAMILY_IDENTIFIER,
        47,
        pid::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    support::exercise_fixture(
        &pid::MODEL_FAMILY,
        pid::MODEL_FAMILY_FIXTURE,
        "pid_comfy_model_0109",
        47,
    )?;
    Ok(())
}

#[test]
fn core_and_net_layouts_execute_branching_and_non_pid_or_partial_probes_fail()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [PixelDitPidLayout::CoreNative, PixelDitPidLayout::NetNative] {
        let configuration = pid::configuration_for_probe(&probe(layout, true))?;
        assert_eq!(configuration.variant, PixelDitPidVariant::PiD);
        assert_eq!(configuration.layout, layout);
        let pid_configuration = configuration.pid.ok_or("PiD configuration is missing")?;
        assert_eq!(pid_configuration.lq_gate_count, 7);
        assert_eq!(pid_configuration.lq_interval, 2);
        assert_eq!(pid_configuration.latent_spatial_down_factor, 16);
        support::exercise_plan(
            pixeldit_pid_state_plan_for_layout(layout),
            &mapping_keys(layout),
            &[
                "native.pixel_blocks.0.adaLN_modulation_msa.weight",
                "native.pixel_blocks.0.adaLN_modulation_mlp.weight",
                "native.lq_proj.latent_proj.0.weight",
            ],
        )?;
    }
    assert!(matches!(
        pid::configuration_for_probe(&probe(PixelDitPidLayout::CoreNative, false)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("cannot admit")
    ));
    let mut partial = probe(PixelDitPidLayout::NetNative, true);
    partial.tensor_shapes.remove("net.final_layer.linear.weight");
    assert!(matches!(
        pid::configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut ambiguous = probe(PixelDitPidLayout::CoreNative, true);
    ambiguous
        .tensor_shapes
        .extend(probe(PixelDitPidLayout::NetNative, true).tensor_shapes);
    assert!(matches!(
        pid::configuration_for_probe(&ambiguous),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
    support::assert_leaf_owner("pid_comfy_model_0109", "pixeldit_pid_family")?;
    Ok(())
}

fn probe(layout: PixelDitPidLayout, include_lq: bool) -> ModelProbe {
    let prefix = prefix(layout);
    let modulation = 6 * 16 * 16 * 16;
    let mut tensor_shapes = BTreeMap::from([
        (format!("{prefix}pixel_embedder.proj.weight"), vec![16, 3]),
        (format!("{prefix}s_embedder.proj.weight"), vec![1_536, 768]),
        (format!("{prefix}y_embedder.proj.weight"), vec![1_536, 2_304]),
        (format!("{prefix}final_layer.linear.weight"), vec![3, 16]),
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
    ModelProbe { tensor_shapes, metadata: BTreeMap::new() }
}

fn mapping_keys(layout: PixelDitPidLayout) -> Vec<(String, Vec<u64>)> {
    let prefix = prefix(layout);
    vec![
        (format!("{prefix}pixel_embedder.proj.weight"), vec![2, 3]),
        (format!("{prefix}s_embedder.proj.weight"), vec![2, 2]),
        (format!("{prefix}y_embedder.proj.weight"), vec![2, 2]),
        (format!("{prefix}patch_blocks.0.attn.qkv_x.weight"), vec![2, 2]),
        (
            format!("{prefix}pixel_blocks.0.adaLN_modulation.0.weight"),
            vec![24, 2],
        ),
        (
            format!("{prefix}pixel_blocks.0.adaLN_modulation.0.bias"),
            vec![24],
        ),
        (format!("{prefix}lq_proj.latent_proj.0.weight"), vec![2, 2]),
        (
            format!("{prefix}lq_proj.gate_modules.0.content_proj.weight"),
            vec![2, 2],
        ),
        (format!("{prefix}final_layer.linear.weight"), vec![2, 2]),
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
