use super::generated_hunyuanthree_dv2_sim_model_0084::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};
use sim_model::{
    Hunyuan3DVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuanthree_dv2mini_sim_model_0086 as row,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9086",
    identifier: "Hunyuan3Dv2miniAmbiguousFixture",
    ..row::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 165,
        source_architecture: "model_base.Hunyuan3Dv2miniAmbiguousFixture",
        ..row::MODEL_FAMILY_REGISTRATION
    },
];

const SPEC: RowSpec = RowSpec {
    feature_id: row::MODEL_FAMILY_FEATURE_ID,
    identifier: row::MODEL_FAMILY_IDENTIFIER,
    fixture: row::MODEL_FAMILY_FIXTURE,
    module: "hunyuanthree_dv2mini_comfy_model_0086",
    source_ordinal: row::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.Hunyuan3Dv2",
    architecture_version: "hunyuan3d-v2-mini-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0034",
    latent_identifier: "Hunyuan3Dv2mini",
    image_model: "hunyuan3d2",
    projection_sha256: row::MODEL_FAMILY_PROJECTION_SHA256,
    variant: Hunyuan3DVariant::V2Mini,
    detection_score: 900,
};

#[test]
fn val_model_family_row_001_hunyuan3dv2mini_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuan3dv2mini_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuan3dv2mini_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
