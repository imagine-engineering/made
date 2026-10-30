use sim_model::{
    ModelFamilyError, ModelProbe, Omnigen2BooguLayout, Omnigen2BooguVariant,
    generated_omnigen2_sim_model_0108 as omnigen2, omnigen2_boogu_state_plan_for_layout,
};
use std::collections::BTreeMap;

use super::generated_lumina2_sim_model_0107::support;

#[test]
fn source_projection_descriptor_fixture_and_fail_closed_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(omnigen2::MODEL_FAMILY_IDENTIFIER, "Omnigen2");
    assert_eq!(omnigen2::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0108");
    assert_eq!(omnigen2::MODEL_FAMILY_SOURCE_ORDINAL, 75);
    assert_eq!(omnigen2::MODEL_FAMILY_REGISTRATION.source_architecture, "model_base.Omnigen2");
    assert_eq!(omnigen2::MODEL_FAMILY_SAMPLING_SHIFT, 2.6);
    support::validate_provenance(
        omnigen2::MODEL_FAMILY_FIXTURE,
        omnigen2::MODEL_FAMILY_FEATURE_ID,
        omnigen2::MODEL_FAMILY_IDENTIFIER,
        75,
        omnigen2::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    support::exercise_fixture(
        &omnigen2::MODEL_FAMILY,
        omnigen2::MODEL_FAMILY_FIXTURE,
        "omnigen2_comfy_model_0108",
        75,
    )?;
    Ok(())
}

#[test]
fn both_native_layouts_execute_and_sibling_or_malformed_probes_fail()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        Omnigen2BooguLayout::PrefixedNative,
        Omnigen2BooguLayout::StandaloneNative,
    ] {
        let configuration = omnigen2::configuration_for_probe(&probe(layout))?;
        assert_eq!(configuration.variant, Omnigen2BooguVariant::Omnigen2);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.hidden_size, 2_520);
        assert_eq!(configuration.number_of_layers, 32);
        assert_eq!(configuration.number_of_refiner_layers, 2);
        support::exercise_plan(
            omnigen2_boogu_state_plan_for_layout(layout),
            &mapping_keys(layout),
            &["native.x_embedder.weight", "native.layers.0.attn.to_q.weight"],
        )?;
    }
    let mut partial = probe(Omnigen2BooguLayout::StandaloneNative);
    partial.tensor_shapes.remove("norm_out.linear_2.weight");
    assert!(matches!(
        omnigen2::configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut ambiguous = probe(Omnigen2BooguLayout::PrefixedNative);
    ambiguous.tensor_shapes.extend(
        probe(Omnigen2BooguLayout::StandaloneNative).tensor_shapes,
    );
    assert!(matches!(
        omnigen2::configuration_for_probe(&ambiguous),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
    let mut boogu = probe(Omnigen2BooguLayout::StandaloneNative);
    boogu.tensor_shapes.insert(
        "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight".to_owned(),
        vec![3_360, 3_360],
    );
    assert!(matches!(
        omnigen2::configuration_for_probe(&boogu),
        Err(ModelFamilyError::InvalidSelectorOutput(_))
    ));
    support::assert_leaf_owner("omnigen2_comfy_model_0108", "omnigen2_boogu_family")?;
    Ok(())
}

fn probe(layout: Omnigen2BooguLayout) -> ModelProbe {
    let prefix = match layout {
        Omnigen2BooguLayout::PrefixedNative => "model.diffusion_model.",
        Omnigen2BooguLayout::StandaloneNative => "",
    };
    let mut tensor_shapes = BTreeMap::from([
        (format!("{prefix}x_embedder.weight"), vec![2_520, 64]),
        (format!("{prefix}norm_out.linear_2.weight"), vec![64, 2_520]),
        (
            format!("{prefix}time_caption_embed.timestep_embedder.linear_1.bias"),
            vec![2_520],
        ),
    ]);
    for index in 0..32 {
        tensor_shapes.insert(
            format!("{prefix}layers.{index}.attn.to_q.weight"),
            vec![2_520, 2_520],
        );
    }
    for index in 0..2 {
        tensor_shapes.insert(
            format!("{prefix}noise_refiner.{index}.attn.to_q.weight"),
            vec![2_520, 2_520],
        );
    }
    ModelProbe { tensor_shapes, metadata: BTreeMap::new() }
}

fn mapping_keys(layout: Omnigen2BooguLayout) -> Vec<(String, Vec<u64>)> {
    let prefix = match layout {
        Omnigen2BooguLayout::PrefixedNative => "model.diffusion_model.",
        Omnigen2BooguLayout::StandaloneNative => "",
    };
    [
        "x_embedder.weight",
        "x_embedder.bias",
        "time_caption_embed.timestep_embedder.linear_1.bias",
        "layers.0.attn.to_q.weight",
        "noise_refiner.0.attn.to_q.weight",
        "norm_out.linear_2.weight",
        "norm_out.linear_2.bias",
    ]
    .into_iter()
    .map(|key| (format!("{prefix}{key}"), vec![2, 2]))
    .chain([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.qwen.weight".to_owned(), vec![1]),
    ])
    .collect()
}
