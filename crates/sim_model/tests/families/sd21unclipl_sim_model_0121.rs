use sim_model::{
    ModelFamilyError, Sd2ConditioningFact, Sd2Layout, Sd2ModelType, Sd2Variant,
    describe_model_family,
    generated_sd20_sim_model_0119 as sd20,
    generated_sd21unclipl_sim_model_0121 as unclip_l,
};
use sim_types::DeviceKind;

use super::generated_sd20_sim_model_0119 as sd20_test;

#[test]
fn val_model_family_row_001_sd21_unclip_l_configuration_precedence_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(unclip_l::MODEL_FAMILY_IDENTIFIER, "SD21UnclipL");
    assert_eq!(unclip_l::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0121");
    assert_eq!(unclip_l::MODEL_FAMILY_SOURCE_ORDINAL, 5);
    assert_eq!(unclip_l::MODEL_FAMILY_REGISTRATION.source_ordinal, 5);
    assert_eq!(
        unclip_l::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.SD21UNCLIP"
    );
    assert_eq!(unclip_l::SOURCE_ADM_IN_CHANNELS, 1_536);
    assert_eq!(unclip_l::SOURCE_TIMESTEP_DIMENSION, 768);
    let descriptor = describe_model_family(&unclip_l::MODEL_FAMILY)?;
    assert_eq!(descriptor.architecture_version, "sd21-unclip-l-native-v1");
    assert_eq!(descriptor.latent_format, "SD15");

    for (probe, layout) in [
        (
            sd20_test::standard_native_probe(Some(1_536), false, 4),
            Sd2Layout::PrefixedNative,
        ),
        (
            sd20_test::standard_diffusers_probe(Some(1_536), false),
            Sd2Layout::Diffusers,
        ),
    ] {
        let configuration = unclip_l::configuration_for_probe(&probe, None)?;
        assert_eq!(configuration.variant, Sd2Variant::Sd21UnclipL);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.model_type, Sd2ModelType::Eps);
        assert_eq!(configuration.adm_in_channels, Some(1_536));
        let unclip = configuration.unclip.ok_or("unCLIP-L config is missing")?;
        assert_eq!(unclip.timestep_dimension, 768);
        assert_eq!(unclip.timesteps, 1_000);
        assert_eq!(unclip.beta_schedule, "squaredcos_cap_v2");
        assert_eq!(unclip.seed_offset, -10);
        assert!(
            configuration
                .conditioning
                .contains(&Sd2ConditioningFact::UnclipVisionEmbedding)
        );
        assert!(
            configuration
                .conditioning
                .contains(&Sd2ConditioningFact::UnclipNoiseLevelEmbedding)
        );
        assert!(
            configuration
                .conditioning
                .contains(&Sd2ConditioningFact::UnclipZeroFallback)
        );
    }
    let registry = sd20_test::registry()?;
    let resolved = registry.resolve(&sd20_test::execution_probe(Some(1_536)))?;
    assert_eq!(
        resolved.detection().identity.feature_id(),
        unclip_l::MODEL_FAMILY_FEATURE_ID
    );
    assert_eq!(resolved.detection().score, 1_600);
    assert_eq!(resolved.profile().supported_devices, [DeviceKind::Cpu]);
    assert_eq!(
        registry
            .resolve(&sd20_test::probe_through_model_store(Some(1_536))?)?
            .detection()
            .identity
            .feature_id(),
        unclip_l::MODEL_FAMILY_FEATURE_ID
    );
    sd20_test::exercise_registered_runtime(1_536, unclip_l::MODEL_FAMILY_FEATURE_ID)?;

    let statistic_probe = sd20_test::standard_native_probe(Some(1_536), true, 4);
    assert!(unclip_l::weight_statistic_request_for_probe(&statistic_probe)?.is_some());
    let high = sd20_test::observe_statistic(&[-0.15, -0.05, 0.05, 0.15])?;
    assert_eq!(
        unclip_l::configuration_for_probe(&statistic_probe, Some(&high))?.model_type,
        Sd2ModelType::VPrediction
    );

    assert!(matches!(
        unclip_l::configuration_for_probe(
            &sd20_test::standard_native_probe(Some(2_048), false, 4),
            None,
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("expected Sd21UnclipL")
    ));
    assert!(matches!(
        registry.resolve(&sd20_test::execution_probe(Some(2_048))),
        Ok(resolved)
            if resolved.detection().identity.feature_id() != unclip_l::MODEL_FAMILY_FEATURE_ID
    ));
    assert!(matches!(
        sd20::configuration_for_probe(
            &sd20_test::standard_native_probe(Some(1_536), false, 4),
            None,
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("expected Sd20")
    ));

    sd20_test::validate_provenance_and_catalog(
        unclip_l::MODEL_FAMILY_FIXTURE,
        unclip_l::MODEL_FAMILY_IDENTIFIER,
        unclip_l::MODEL_FAMILY_FEATURE_ID,
        unclip_l::MODEL_FAMILY_SOURCE_ORDINAL,
        unclip_l::MODEL_FAMILY_PROJECTION_SHA256,
        "model_base.SD21UNCLIP",
        Some(1_536),
        Some(768),
    )?;
    super::write_model_family_row_artifact(
        unclip_l::MODEL_FAMILY_FIXTURE,
        unclip_l::MODEL_FAMILY_FEATURE_ID,
        unclip_l::MODEL_FAMILY_IDENTIFIER,
        unclip_l::MODEL_FAMILY_SOURCE_ORDINAL,
        "sd21unclipl_comfy_model_0121",
        &[
            "source-provenance-catalog-registration",
            "native-and-diffusers-adm-1536-configuration",
            "unclip-vision-noise-zero-fallback-conditioning",
            "sd20-unclip-l-unclip-h-precedence-and-fail-closed-routing",
            "shared-sd2-state-forward-statistic-memory-device-semantics",
        ],
    )?;
    Ok(())
}
