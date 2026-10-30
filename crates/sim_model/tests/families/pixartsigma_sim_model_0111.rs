use sim_model::{
    ModelFamilyError, ModelFamilyRegistry, ModelProbe, PixArtLayout, PixArtVariant,
    generated_pixartsigma_sim_model_0111 as pixart,
};
use std::collections::BTreeMap;

use super::generated_lumina2_sim_model_0107::support;

#[test]
fn source_projection_descriptor_fixture_and_fail_closed_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(pixart::MODEL_FAMILY_IDENTIFIER, "PixArtSigma");
    assert_eq!(pixart::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0111");
    assert_eq!(pixart::MODEL_FAMILY_SOURCE_ORDINAL, 24);
    assert_eq!(
        pixart::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.PixArt"
    );
    assert_eq!(pixart::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 0.5);
    support::validate_provenance(
        pixart::MODEL_FAMILY_FIXTURE,
        pixart::MODEL_FAMILY_FEATURE_ID,
        pixart::MODEL_FAMILY_IDENTIFIER,
        24,
        pixart::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    support::exercise_fixture(
        &pixart::MODEL_FAMILY,
        pixart::MODEL_FAMILY_FIXTURE,
        "pixartsigma_comfy_model_0111",
        24,
    )?;
    Ok(())
}

#[test]
fn native_and_pinned_diffusers_layouts_execute_and_alpha_or_partial_probes_fail()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        pixart::MODEL_FAMILY_REGISTRATION,
    ])?;
    for layout in [
        PixArtLayout::PrefixedNative,
        PixArtLayout::StandaloneNative,
    ] {
        let native_probe = probe(layout, false);
        let configuration = pixart::configuration_for_probe(&native_probe)?;
        assert_eq!(configuration.variant, PixArtVariant::Sigma);
        assert_eq!(configuration.layout, layout);
        assert!(!configuration.micro_conditioning);
        assert_eq!(configuration.latent_format.identifier, "SDXL");
        support::exercise_compiled_plan(
            registry
                .resolve(&native_probe)?
                .state_plan()
                .ok_or("registry omitted the native state plan")?,
            &native_mapping_keys(layout),
            &[
                "native.x_embedder.proj.weight",
                "native.blocks.0.attn.qkv.weight",
                "native.final_layer.linear.weight",
            ],
        )?;
    }
    let diffusers_probe = probe(PixArtLayout::Diffusers, false);
    let configuration = pixart::configuration_for_probe(&diffusers_probe)?;
    assert_eq!(configuration.layout, PixArtLayout::Diffusers);
    let resolved = registry.resolve(&diffusers_probe)?;
    let plan = resolved
        .state_plan()
        .ok_or("registry omitted the probe-derived Diffusers state plan")?;
    support::exercise_compiled_plan(
        plan,
        &diffusers_mapping_keys(),
        &[
            "native.x_embedder.proj.weight",
            "native.blocks.0.attn.qkv.weight",
            "native.blocks.0.cross_attn.kv_linear.weight",
            "native.final_layer.linear.weight",
        ],
    )?;

    assert!(matches!(
        pixart::configuration_for_probe(&probe(PixArtLayout::StandaloneNative, true)),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("cannot admit")
    ));
    let mut partial = probe(PixArtLayout::Diffusers, false);
    partial.tensor_shapes.remove("proj_out.weight");
    assert!(matches!(
        pixart::configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut ambiguous = probe(PixArtLayout::PrefixedNative, false);
    ambiguous
        .tensor_shapes
        .extend(probe(PixArtLayout::StandaloneNative, false).tensor_shapes);
    assert!(matches!(
        pixart::configuration_for_probe(&ambiguous),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
    support::assert_leaf_owner("pixartsigma_comfy_model_0111", "pixart_family")?;
    Ok(())
}

fn probe(layout: PixArtLayout, alpha: bool) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::new();
    if layout == PixArtLayout::Diffusers {
        tensor_shapes.extend(BTreeMap::from([
            (
                "adaln_single.emb.timestep_embedder.linear_1.bias".to_owned(),
                vec![1_152],
            ),
            ("adaln_single.linear.weight".to_owned(), vec![6_912, 1_152]),
            ("pos_embed.proj.bias".to_owned(), vec![1_152]),
            ("pos_embed.proj.weight".to_owned(), vec![1_152, 4, 2, 2]),
            (
                "caption_projection.y_embedding".to_owned(),
                vec![120, 4_096],
            ),
            ("proj_out.weight".to_owned(), vec![32, 1_152]),
            (
                "transformer_blocks.0.attn1.to_q.weight".to_owned(),
                vec![1_152, 1_152],
            ),
            (
                "transformer_blocks.0.attn1.to_k.weight".to_owned(),
                vec![1_152, 1_152],
            ),
            (
                "transformer_blocks.0.attn1.to_v.weight".to_owned(),
                vec![1_152, 1_152],
            ),
        ]));
        if alpha {
            tensor_shapes.insert(
                "adaln_single.emb.resolution_embedder.linear_1.weight".to_owned(),
                vec![384, 256],
            );
            tensor_shapes.insert(
                "adaln_single.emb.aspect_ratio_embedder.linear_1.weight".to_owned(),
                vec![384, 256],
            );
        }
    } else {
        let prefix = match layout {
            PixArtLayout::PrefixedNative => "model.diffusion_model.",
            PixArtLayout::StandaloneNative => "",
            PixArtLayout::Diffusers => unreachable!(),
        };
        tensor_shapes.extend(BTreeMap::from([
            (format!("{prefix}t_block.1.weight"), vec![6_912, 1_152]),
            (
                format!("{prefix}x_embedder.proj.weight"),
                vec![1_152, 4, 2, 2],
            ),
            (
                format!("{prefix}y_embedder.y_embedding"),
                vec![120, 4_096],
            ),
            (
                format!("{prefix}blocks.0.attn.qkv.weight"),
                vec![3_456, 1_152],
            ),
            (
                format!("{prefix}final_layer.linear.weight"),
                vec![32, 1_152],
            ),
            (format!("{prefix}pos_embed"), vec![1, 64, 1_152]),
        ]));
        if alpha {
            tensor_shapes.insert(
                format!("{prefix}csize_embedder.mlp.0.weight"),
                vec![384, 256],
            );
            tensor_shapes.insert(
                format!("{prefix}ar_embedder.mlp.0.weight"),
                vec![384, 256],
            );
        }
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn native_mapping_keys(layout: PixArtLayout) -> Vec<(String, Vec<u64>)> {
    let prefix = match layout {
        PixArtLayout::PrefixedNative => "model.diffusion_model.",
        PixArtLayout::StandaloneNative => "",
        PixArtLayout::Diffusers => unreachable!(),
    };
    [
        "t_block.1.weight",
        "t_embedder.mlp.0.weight",
        "x_embedder.proj.weight",
        "y_embedder.y_proj.fc1.weight",
        "blocks.0.attn.qkv.weight",
        "final_layer.linear.weight",
    ]
    .into_iter()
    .map(|key| (format!("{prefix}{key}"), vec![2, 2]))
    .chain([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.t5.weight".to_owned(), vec![1]),
    ])
    .collect()
}

fn diffusers_mapping_keys() -> Vec<(String, Vec<u64>)> {
    let mut keys = [
        "pos_embed.proj.weight",
        "pos_embed.proj.bias",
        "caption_projection.y_embedding",
        "caption_projection.linear_1.weight",
        "caption_projection.linear_1.bias",
        "caption_projection.linear_2.weight",
        "caption_projection.linear_2.bias",
        "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias",
        "adaln_single.linear.weight",
        "adaln_single.linear.bias",
        "proj_out.weight",
        "proj_out.bias",
        "scale_shift_table",
        "transformer_blocks.0.attn1.to_q.weight",
        "transformer_blocks.0.attn1.to_k.weight",
        "transformer_blocks.0.attn1.to_v.weight",
        "transformer_blocks.0.attn1.to_q.bias",
        "transformer_blocks.0.attn1.to_k.bias",
        "transformer_blocks.0.attn1.to_v.bias",
        "transformer_blocks.0.attn2.to_q.weight",
        "transformer_blocks.0.attn2.to_q.bias",
        "transformer_blocks.0.attn2.to_k.weight",
        "transformer_blocks.0.attn2.to_v.weight",
        "transformer_blocks.0.attn2.to_k.bias",
        "transformer_blocks.0.attn2.to_v.bias",
        "transformer_blocks.0.scale_shift_table",
        "transformer_blocks.0.attn1.to_out.0.weight",
        "transformer_blocks.0.attn1.to_out.0.bias",
        "transformer_blocks.0.ff.net.0.proj.weight",
        "transformer_blocks.0.ff.net.0.proj.bias",
        "transformer_blocks.0.ff.net.2.weight",
        "transformer_blocks.0.ff.net.2.bias",
        "transformer_blocks.0.attn2.to_out.0.weight",
        "transformer_blocks.0.attn2.to_out.0.bias",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), vec![2, 2]))
    .collect::<Vec<_>>();
    keys.extend([
        ("vae.decoder.weight".to_owned(), vec![1]),
        ("text_encoders.t5.weight".to_owned(), vec![1]),
    ]);
    keys
}
