use sim_model::{
    COSMOS_GENERAL_STATE_PLAN, COSMOS_PREDICT2_STATE_PLAN, CosmosArchitecture, CosmosModelSize,
    CosmosRatio, ModelFamilyError, ModelFamilyRegistry, ModelProbe, cosmos_configuration_for_probe,
    generated_cosmosi2v_sim_model_0071 as cosmos_i2v,
};
use std::{collections::BTreeMap, fs, path::Path};

#[test]
fn val_model_family_row_001_cosmos_adapter_preserves_general_and_predict2_profiles() {
    let general = cosmos_configuration_for_probe(
        &general_probe(17, 4_096),
        CosmosArchitecture::GeneralDit,
        true,
        "CosmosI2V",
    )
    .expect("GeneralDIT I2V configuration");
    assert_eq!(general.in_channels, 17);
    assert_eq!(general.model_size, CosmosModelSize::SevenB);
    assert_eq!(
        (general.number_of_blocks, general.number_of_heads),
        (28, 32)
    );
    assert_eq!(
        general.rope_extrapolation,
        [ratio(1, 1), ratio(1, 1), ratio(2, 1)]
    );
    assert_eq!(general.extra_per_block_absolute_position, Some(true));
    assert_eq!(general.memory_usage_factor, 1.6);

    let predict_t2i = cosmos_configuration_for_probe(
        &predict2_probe(16, 2_048, false),
        CosmosArchitecture::Predict2,
        false,
        "CosmosT2IPredict2",
    )
    .expect("Predict2 T2I configuration");
    assert_eq!(predict_t2i.model_size, CosmosModelSize::Predict2TwoB);
    assert_eq!(
        (predict_t2i.number_of_blocks, predict_t2i.number_of_heads),
        (28, 16)
    );
    assert_eq!(
        predict_t2i.rope_extrapolation,
        [ratio(4, 1), ratio(4, 1), ratio(1, 1)]
    );
    assert_eq!(predict_t2i.memory_usage_factor, 0.95);

    let predict_i2v = cosmos_configuration_for_probe(
        &predict2_probe(17, 5_120, false),
        CosmosArchitecture::Predict2,
        true,
        "CosmosI2VPredict2",
    )
    .expect("Predict2 I2V configuration");
    assert_eq!(predict_i2v.model_size, CosmosModelSize::FourteenB);
    assert_eq!(
        predict_i2v.rope_extrapolation,
        [ratio(2, 1), ratio(2, 1), ratio(5, 6)]
    );
    assert_eq!(predict_i2v.extra_per_block_absolute_position, None);
    assert_eq!(predict_i2v.memory_usage_factor, 2.375);
}

#[test]
fn val_model_family_row_001_cosmos_adapter_separates_markers_channels_and_shapes() {
    assert!(matches!(
        cosmos_configuration_for_probe(
            &predict2_probe(17, 2_048, false),
            CosmosArchitecture::GeneralDit,
            true,
            "CosmosI2V",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("Cosmos Predict2 marker")
    ));
    assert!(matches!(
        cosmos_configuration_for_probe(
            &predict2_probe(16, 2_048, true),
            CosmosArchitecture::Predict2,
            false,
            "CosmosT2IPredict2",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("Anima marker")
    ));
    assert!(matches!(
        cosmos_configuration_for_probe(
            &general_probe(16, 4_096),
            CosmosArchitecture::GeneralDit,
            true,
            "CosmosI2V",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("in_channels 16; requires 17")
    ));

    let mut malformed = general_probe(17, 4_096);
    malformed
        .tensor_shapes
        .insert("net.x_embedder.proj.1.weight".to_string(), vec![2, 71]);
    assert!(matches!(
        cosmos_configuration_for_probe(
            &malformed,
            CosmosArchitecture::GeneralDit,
            true,
            "CosmosI2V",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("x_embedder.proj.1.weight shape")
    ));
}

#[test]
fn val_model_detection_001_cosmos_i2v_registration_is_key_derived_and_metadata_independent()
-> Result<(), Box<dyn std::error::Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(&[cosmos_i2v::MODEL_FAMILY_REGISTRATION])?;
    let mut model_probe = general_probe(17, 4_096);
    model_probe.metadata.extend([
        ("image_model".to_owned(), "anima".to_owned()),
        ("in_channels".to_owned(), "999".to_owned()),
    ]);
    assert_eq!(
        registry.detect(&model_probe)?.identity.feature_id(),
        cosmos_i2v::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(
        registry
            .resolve(&model_probe)?
            .detection()
            .identity
            .feature_id(),
        cosmos_i2v::MODEL_FAMILY_FEATURE_ID
    );

    let mut missing_marker = general_probe(17, 4_096);
    missing_marker
        .tensor_shapes
        .remove("net.blocks.block0.blocks.0.block.attn.to_q.0.weight");
    assert!(matches!(
        registry.detect(&missing_marker),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let wrong_channels = general_probe(16, 4_096);
    assert!(matches!(
        registry.detect(&wrong_channels),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut cross_family = general_probe(17, 4_096);
    cross_family
        .tensor_shapes
        .insert("net.blocks.0.mlp.layer1.weight".to_owned(), vec![2, 2]);
    assert!(matches!(
        registry.resolve(&cross_family),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("Cosmos Predict2 marker")
    ));
    Ok(())
}

#[test]
fn val_model_family_row_001_cosmos_adapter_is_the_single_shared_owner() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let owner = fs::read_to_string(crate_root.join("src/cosmos_family.rs"))
        .expect("canonical Cosmos adapter source");
    assert_eq!(owner.matches("pub enum CosmosArchitecture").count(), 1);
    assert_eq!(owner.matches("pub struct CosmosConfiguration").count(), 1);
    assert_eq!(owner.matches("pub enum CosmosModelSize").count(), 1);
    assert!(owner.contains("pub const COSMOS_GENERAL_DETECTION_MARKER_KEYS"));
    assert!(owner.contains("pub const COSMOS_PREDICT2_DETECTION_MARKER_KEYS"));
    assert!(owner.contains("pub const COSMOS_ANIMA_DETECTION_MARKER_KEYS"));
    assert!(owner.contains("pub const COSMOS_PATCH_PROJECTION_KEYS"));

    let row = fs::read_to_string(crate_root.join("src/families/cosmosi2v_sim_model_0071.rs"))
        .expect("CosmosI2V row source");
    assert!(!row.contains("pub struct CosmosI2VConfiguration"));
    assert!(!row.contains("pub enum CosmosI2VModelSize"));
    assert!(!row.contains("fn shape("));
    assert!(!row.contains("ModelDetectionRule::Metadata"));
    assert!(!row.contains("ModelSourceConfigurationRule"));
    assert!(row.contains("ModelDetectionRule::AnyTensorDimensionValue"));
    assert!(row.contains("source_configuration: &[]"));
    assert!(row.contains("cosmos_configuration_for_probe("));

    let general = COSMOS_GENERAL_STATE_PLAN
        .compile()
        .expect("GeneralDIT state plan");
    let predict2 = COSMOS_PREDICT2_STATE_PLAN
        .compile()
        .expect("Predict2 state plan");
    assert_ne!(general.identity(), predict2.identity());
    assert_eq!(general.operations().len(), 3);
    assert_eq!(predict2.operations().len(), 3);
}

fn general_probe(in_channels: u64, model_channels: u64) -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "net.x_embedder.proj.1.weight".to_string(),
                vec![2, (in_channels + 1) * 4],
            ),
            (
                "net.blocks.block0.blocks.0.block.attn.to_q.0.weight".to_string(),
                vec![model_channels, 2],
            ),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn predict2_probe(in_channels: u64, model_channels: u64, anima: bool) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        (
            "net.x_embedder.proj.1.weight".to_string(),
            vec![model_channels, (in_channels + 1) * 4],
        ),
        ("net.blocks.0.mlp.layer1.weight".to_string(), vec![2, 2]),
    ]);
    if anima {
        tensor_shapes.insert(
            "net.llm_adapter.blocks.0.cross_attn.q_proj.weight".to_string(),
            vec![2, 2],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

const fn ratio(numerator: u64, denominator: u64) -> CosmosRatio {
    CosmosRatio {
        numerator,
        denominator,
    }
}
