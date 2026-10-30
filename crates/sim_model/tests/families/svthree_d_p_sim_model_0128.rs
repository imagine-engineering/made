use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelProbe,
    generated_svthree_d_p_sim_model_0128 as sv3d_p,
};

use super::generated_svd_img2vid_sim_model_0130::{
    VideoFamilyCase, run_execution_validation, run_source_validation,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9128",
    identifier: "SV3DPAmbiguousFixture",
    ..sv3d_p::MODEL_FAMILY
};
static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    sv3d_p::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 118,
        source_architecture: "model_base.SV3DPAmbiguousFixture",
        ..sv3d_p::MODEL_FAMILY_REGISTRATION
    },
];
static SVTHREE_D_P_REGISTRATIONS: [ModelFamilyRegistration; 1] =
    [sv3d_p::MODEL_FAMILY_REGISTRATION];

#[test]
fn val_model_family_row_001_svthree_d_p_source_projection_configuration_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(sv3d_p::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 2.0);
    assert_eq!(sv3d_p::MODEL_FAMILY_SIGMA_MAX, 700.0);
    assert_eq!(sv3d_p::MODEL_FAMILY_SIGMA_MIN, 0.002);
    run_source_validation(&case())
}

#[test]
fn val_model_family_row_001_svthree_d_p_execution_failures_and_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    run_execution_validation(&case())
}

fn case() -> VideoFamilyCase {
    VideoFamilyCase {
        definition: &sv3d_p::MODEL_FAMILY,
        registration: sv3d_p::MODEL_FAMILY_REGISTRATION,
        registrations: &SVTHREE_D_P_REGISTRATIONS,
        ambiguous_registrations: &AMBIGUOUS_REGISTRATIONS,
        identifier: sv3d_p::MODEL_FAMILY_IDENTIFIER,
        feature_id: sv3d_p::MODEL_FAMILY_FEATURE_ID,
        fixture: sv3d_p::MODEL_FAMILY_FIXTURE,
        module: "svthree_d_p_comfy_model_0128",
        source_ordinal: sv3d_p::MODEL_FAMILY_SOURCE_ORDINAL,
        source_architecture: "model_base.SV3D_p",
        architecture_version: "sv3d-pose-conditioned-unet-v1",
        projection_sha256: sv3d_p::MODEL_FAMILY_PROJECTION_SHA256,
        adm_input_channels: sv3d_p::MODEL_FAMILY_ADM_IN_CHANNELS,
        vae_source_prefix: "conditioner.embedders.1.encoder.",
        conditioning_checkpoint: "pose_conditioning_projection",
        output_checkpoint: "view_latent_output",
        validate_configuration,
    }
}

fn validate_configuration(probe: &ModelProbe) -> Result<(), ModelFamilyError> {
    let configuration = sv3d_p::configuration_for_probe(probe)?;
    if configuration.model_channels != 320
        || configuration.input_channels != 8
        || configuration.context_dimension != 1_024
        || configuration.adm_input_channels != 1_280
        || !configuration.temporal_attention
        || !configuration.temporal_residual_blocks
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "SV3D_p typed configuration did not preserve source values".to_owned(),
        ));
    }
    Ok(())
}
