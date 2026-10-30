use sim_model::{
    CogVideoXLatentVariant, CogVideoXLayout, ModelDetectionRule, ModelFamilyError,
    ModelFamilyIdentity, ModelFamilyRegistry, ModelProbe, cogvideox_configuration_for_probe,
    detect_model_family_rules, generated_cogvideox_i2v_sim_model_0068 as i2v,
    generated_cogvideox_inpaint_sim_model_0069 as inpaint,
    generated_cogvideox_t2v_sim_model_0070 as t2v,
};
use std::{collections::BTreeMap, fs, path::Path};

#[test]
fn val_model_family_row_001_cogvideox_adapter_preserves_spatial_and_temporal_profiles() {
    let spatial = cogvideox_configuration_for_probe(
        &probe("native", 32, 1, PatchVariant::Spatial, false),
        32,
        "CogVideoX_I2V",
    )
    .expect("native spatial configuration");
    assert_eq!(spatial.layout, CogVideoXLayout::Native);
    assert_eq!(spatial.number_of_attention_heads, 1);
    assert_eq!(spatial.temporal_patch_size, None);
    assert_eq!(
        (
            spatial.sample_height,
            spatial.sample_width,
            spatial.sample_frames
        ),
        (60, 90, 49)
    );
    assert_eq!(spatial.latent_variant, CogVideoXLatentVariant::CogVideoX);

    let temporal = cogvideox_configuration_for_probe(
        &probe("diffusers", 48, 48, PatchVariant::Temporal, true),
        48,
        "CogVideoX_Inpaint",
    )
    .expect("diffusers temporal configuration");
    assert_eq!(temporal.layout, CogVideoXLayout::Diffusers);
    assert_eq!(temporal.temporal_patch_size, Some(2));
    assert_eq!(
        (
            temporal.sample_height,
            temporal.sample_width,
            temporal.sample_frames
        ),
        (96, 170, 81)
    );
    assert_eq!(temporal.text_embedding_dimension, Some(4_096));
    assert_eq!(temporal.ofs_embedding_dimension, Some(2));
    assert!(temporal.learned_positional_embeddings);
    assert_eq!(
        temporal.latent_variant,
        CogVideoXLatentVariant::CogVideoX1_5
    );
}

#[test]
fn val_model_family_row_001_cogvideox_adapter_rejects_malformed_and_cross_family_probes() {
    let wrong_channel = probe("native", 16, 1, PatchVariant::Spatial, false);
    assert!(matches!(
        cogvideox_configuration_for_probe(&wrong_channel, 32, "CogVideoX_I2V"),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message == "CogVideoX_I2V configuration in_channels 16; requires 32"
    ));

    let mut malformed = probe("diffusers", 16, 1, PatchVariant::Spatial, false);
    malformed
        .tensor_shapes
        .insert("blocks.0.norm1.linear.weight".to_string(), vec![65, 2]);
    assert!(matches!(
        cogvideox_configuration_for_probe(&malformed, 16, "CogVideoX_T2V"),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message == "CogVideoX_T2V configuration blocks.0.norm1.linear.weight shape"
    ));

    let mut malformed_optional = probe("diffusers", 16, 1, PatchVariant::Spatial, false);
    malformed_optional
        .tensor_shapes
        .insert("patch_embed.text_proj.weight".to_string(), vec![2]);
    assert!(matches!(
        cogvideox_configuration_for_probe(&malformed_optional, 16, "CogVideoX_T2V"),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("missing dimension 1 for patch_embed.text_proj.weight")
    ));
}

#[test]
fn val_model_detection_001_cogvideox_registry_is_key_derived_and_metadata_independent()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        i2v::MODEL_FAMILY_REGISTRATION,
        inpaint::MODEL_FAMILY_REGISTRATION,
        t2v::MODEL_FAMILY_REGISTRATION,
    ])?;
    for (layout, channels, heads, patch, feature_id) in [
        (
            "native",
            32,
            1,
            PatchVariant::Spatial,
            i2v::MODEL_FAMILY_FEATURE_ID,
        ),
        (
            "diffusers",
            32,
            48,
            PatchVariant::Temporal,
            i2v::MODEL_FAMILY_FEATURE_ID,
        ),
        (
            "native",
            48,
            1,
            PatchVariant::Spatial,
            inpaint::MODEL_FAMILY_FEATURE_ID,
        ),
        (
            "diffusers",
            48,
            48,
            PatchVariant::Temporal,
            inpaint::MODEL_FAMILY_FEATURE_ID,
        ),
        (
            "native",
            16,
            1,
            PatchVariant::Spatial,
            t2v::MODEL_FAMILY_FEATURE_ID,
        ),
        (
            "diffusers",
            16,
            48,
            PatchVariant::Temporal,
            t2v::MODEL_FAMILY_FEATURE_ID,
        ),
    ] {
        let mut model_probe = probe(layout, channels, heads, patch, true);
        model_probe.metadata.extend([
            ("image_model".to_owned(), "not-cogvideox".to_owned()),
            ("in_channels".to_owned(), "999".to_owned()),
        ]);
        assert_eq!(
            registry.detect(&model_probe)?.identity.feature_id(),
            feature_id
        );
        assert_eq!(
            registry
                .resolve(&model_probe)?
                .detection()
                .identity
                .feature_id(),
            feature_id
        );
    }

    let mut missing_projection = probe("native", 32, 1, PatchVariant::Spatial, false);
    missing_projection
        .tensor_shapes
        .remove("model.diffusion_model.patch_embed.proj.weight");
    assert!(matches!(
        registry.detect(&missing_projection),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let wrong_channels = probe("diffusers", 24, 1, PatchVariant::Spatial, false);
    assert!(matches!(
        registry.detect(&wrong_channels),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    let mut cross_family = probe("native", 32, 1, PatchVariant::Spatial, false);
    cross_family
        .tensor_shapes
        .extend(probe("diffusers", 48, 1, PatchVariant::Spatial, false).tensor_shapes);
    assert!(matches!(
        registry.detect(&cross_family),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));

    let mut mixed_layout = probe("native", 16, 1, PatchVariant::Spatial, false);
    mixed_layout
        .tensor_shapes
        .extend(probe("diffusers", 16, 1, PatchVariant::Spatial, false).tensor_shapes);
    assert!(registry.resolve(&mixed_layout).is_err());
    Ok(())
}

#[test]
fn val_model_detection_001_tensor_dimension_rule_is_bounded_and_shape_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let identity = ModelFamilyIdentity::new("COMFY-MODEL-9068", "CogVideoXRule", "rule-v1")?;
    let model_probe = ModelProbe {
        tensor_shapes: BTreeMap::from([("patch.weight".to_owned(), vec![2, 256])]),
        metadata: BTreeMap::new(),
    };
    let detection = detect_model_family_rules(
        identity.clone(),
        &[ModelDetectionRule::AnyTensorDimensionValue {
            keys: &["native.patch.weight", "patch.weight"],
            dimension: 1,
            values: &[32, 256],
            score: 300,
        }],
        &model_probe,
    )?;
    assert_eq!(detection.score, 300);

    for invalid_rule in [
        ModelDetectionRule::AnyTensorDimensionValue {
            keys: &[],
            dimension: 1,
            values: &[32],
            score: 300,
        },
        ModelDetectionRule::AnyTensorDimensionValue {
            keys: &["patch.weight"],
            dimension: 32,
            values: &[32],
            score: 300,
        },
        ModelDetectionRule::AnyTensorDimensionValue {
            keys: &["patch.weight"],
            dimension: 1,
            values: &[],
            score: 300,
        },
        ModelDetectionRule::AnyTensorDimensionValue {
            keys: &["patch.weight"],
            dimension: 1,
            values: &[32, 32],
            score: 300,
        },
    ] {
        assert!(matches!(
            detect_model_family_rules(identity.clone(), &[invalid_rule], &model_probe),
            Err(ModelFamilyError::InvalidDefinition(_))
                | Err(ModelFamilyError::DuplicateDefinitionValue(_))
        ));
    }
    Ok(())
}

#[test]
fn val_model_family_row_001_cogvideox_adapter_is_the_single_configuration_owner() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let owner = fs::read_to_string(crate_root.join("src/cogvideox_family.rs"))
        .expect("canonical CogVideoX adapter source");
    assert_eq!(owner.matches("pub enum CogVideoXLayout").count(), 1);
    assert_eq!(owner.matches("pub enum CogVideoXLatentVariant").count(), 1);
    assert_eq!(
        owner.matches("pub struct CogVideoXConfiguration").count(),
        1
    );
    assert!(owner.contains("pub const COGVIDEOX_DETECTION_MARKER_KEYS"));
    assert!(owner.contains("pub const COGVIDEOX_PATCH_PROJECTION_KEYS"));

    for row in [
        "cogvideox_i2v_sim_model_0068.rs",
        "cogvideox_inpaint_sim_model_0069.rs",
        "cogvideox_t2v_sim_model_0070.rs",
    ] {
        let source = fs::read_to_string(crate_root.join("src/families").join(row))
            .expect("CogVideoX family row source");
        assert!(!source.contains("pub enum CogVideoXLayout"));
        assert!(!source.contains("pub enum CogVideoXLatentVariant"));
        assert!(!source.contains("pub enum CogVideoXT2VLayout"));
        assert!(!source.contains("pub enum CogVideoXT2VLatentVariant"));
        assert!(!source.contains("fn optional_dimension("));
        assert!(!source.contains("ModelDetectionRule::Metadata"));
        assert!(!source.contains("ModelSourceConfigurationRule"));
        assert!(source.contains("ModelDetectionRule::AnyTensorDimensionValue"));
        assert!(source.contains("source_configuration: &[]"));
        assert!(source.contains("cogvideox_configuration_for_probe("));
    }
}

#[derive(Clone, Copy)]
enum PatchVariant {
    Spatial,
    Temporal,
}

fn probe(
    layout: &str,
    channels: u64,
    heads: u64,
    patch: PatchVariant,
    optional_dimensions: bool,
) -> ModelProbe {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let patch_shape = match patch {
        PatchVariant::Spatial => vec![2, channels, 2, 2],
        PatchVariant::Temporal => vec![2, channels * 8],
    };
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}blocks.0.norm1.linear.weight"),
            vec![heads * 64 * 6, 2],
        ),
        (format!("{prefix}patch_embed.proj.weight"), patch_shape),
    ]);
    if optional_dimensions {
        tensor_shapes.extend([
            (
                format!("{prefix}patch_embed.text_proj.weight"),
                vec![2, 4_096],
            ),
            (format!("{prefix}ofs_embedding_linear_1.weight"), vec![2, 2]),
            (format!("{prefix}patch_embed.pos_embedding"), vec![1, 2, 2]),
        ]);
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}
