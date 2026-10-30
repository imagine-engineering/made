use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelProbe, SdxlLayout, SdxlVariant, describe_model_family,
    generated_sdxlrefiner_sim_model_0124 as refiner,
};
use std::collections::BTreeMap;

use super::generated_sdxl_instructpix2pix_sim_model_0125::support;

static REGISTRATIONS: [ModelFamilyRegistration; 1] = [refiner::MODEL_FAMILY_REGISTRATION];
static AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9124",
    identifier: "SDXLRefinerAmbiguousFixture",
    ..refiner::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    refiner::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_DEFINITION,
        source_ordinal: 908,
        source_architecture: "model_base.SDXLRefinerAmbiguousFixture",
        ..refiner::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_sdxlrefiner_source_layouts_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    support::verify_provenance(
        refiner::MODEL_FAMILY_FIXTURE,
        refiner::MODEL_FAMILY_FEATURE_ID,
        refiner::MODEL_FAMILY_IDENTIFIER,
        refiner::MODEL_FAMILY_SOURCE_ORDINAL,
        refiner::SOURCE_ARCHITECTURE,
        refiner::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    assert_eq!(refiner::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.0);
    let descriptor = describe_model_family(&refiner::MODEL_FAMILY)?;
    assert_eq!(descriptor.latent_format, "SDXL");
    assert_eq!(descriptor.component_graph.len(), 3);
    assert_eq!(
        refiner::MODEL_FAMILY.clip_target.candidates[0].clip_model,
        "sim.sdxl_clip.SDXLRefinerClipModel"
    );

    let registry = ModelFamilyRegistry::checked_registrations(&REGISTRATIONS)?;
    for layout in [
        SdxlLayout::PrefixedNative,
        SdxlLayout::StandaloneNative,
        SdxlLayout::Diffusers,
    ] {
        let probe = refiner_probe(layout, 4);
        let configuration = refiner::configuration_for_probe(&probe)?;
        assert_eq!(configuration.variant, SdxlVariant::Refiner);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.model_channels, 384);
        assert_eq!(configuration.context_dimension, 1_280);
        assert_eq!(configuration.adm_in_channels, 2_560);
        assert_eq!(configuration.transformer_depth_middle, 4);
        assert_eq!(registry.resolve(&probe)?.detection().score, 1_000);
    }
    let store_probe = support::probe_through_model_store(refiner::MODEL_FAMILY_FIXTURE)?;
    assert_eq!(registry.resolve(&store_probe)?.source_ordinal(), 8);
    let mut partial = store_probe.clone();
    partial.tensor_shapes.remove(
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.3.attn2.to_k.weight",
    );
    assert!(registry.resolve(&partial).is_err());
    assert!(matches!(
        refiner::configuration_for_probe(&support::variant_probe(
            SdxlLayout::StandaloneNative,
            4,
            10,
        )),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("384-channel/depth-4")
    ));
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&AMBIGUOUS_REGISTRATIONS)?.detect(&store_probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    support::verify_owner_delegation("sdxlrefiner_comfy_model_0124")?;
    Ok(())
}

#[test]
fn val_model_family_row_001_sdxlrefiner_native_execution_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    support::exercise_sdxl(refiner::MODEL_FAMILY_FIXTURE, &REGISTRATIONS)?;
    super::write_model_family_row_artifact(
        refiner::MODEL_FAMILY_FIXTURE,
        refiner::MODEL_FAMILY_FEATURE_ID,
        refiner::MODEL_FAMILY_IDENTIFIER,
        refiner::MODEL_FAMILY_SOURCE_ORDINAL,
        "sdxlrefiner_comfy_model_0124",
        &[
            "source-and-catalog-provenance",
            "native-standalone-and-diffusers-key-detection",
            "source-exact-sdxl-refiner-profile",
            "refiner-clip-target-selection",
            "transactional-sdxl-component-routing",
            "named-native-forward-and-conditioning-checkpoints",
            "patch-order-memory-oom-dtype-device-cancellation",
            "partial-ambiguous-and-cross-variant-failures",
            "canonical-sdxl-owner-delegation",
        ],
    )?;
    Ok(())
}

fn refiner_probe(layout: SdxlLayout, depth: usize) -> ModelProbe {
    let (input, time, adm, output, residual, deep) = match layout {
        SdxlLayout::PrefixedNative => (
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "model.diffusion_model.out.2.weight",
            "model.diffusion_model.input_blocks.2.0.in_layers.0.weight",
            "model.diffusion_model.input_blocks.7.1.transformer_blocks.",
        ),
        SdxlLayout::StandaloneNative => (
            "input_blocks.0.0.weight",
            "time_embed.0.weight",
            "label_emb.0.0.weight",
            "out.2.weight",
            "input_blocks.2.0.in_layers.0.weight",
            "input_blocks.7.1.transformer_blocks.",
        ),
        SdxlLayout::Diffusers => (
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "add_embedding.linear_1.weight",
            "conv_out.weight",
            "down_blocks.0.resnets.1.conv1.weight",
            "down_blocks.2.attentions.0.transformer_blocks.",
        ),
    };
    let mut tensor_shapes = BTreeMap::from([
        (input.to_owned(), vec![384, 4, 3, 3]),
        (time.to_owned(), vec![1_536, 384]),
        (adm.to_owned(), vec![1, 2_560]),
        (output.to_owned(), vec![4, 384, 3, 3]),
        (residual.to_owned(), vec![1]),
    ]);
    for index in 0..depth {
        tensor_shapes.insert(format!("{deep}{index}.attn2.to_k.weight"), vec![1, 1_280]);
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}
