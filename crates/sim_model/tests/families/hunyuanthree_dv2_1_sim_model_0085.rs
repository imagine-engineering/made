use super::generated_hunyuanthree_dv2_sim_model_0084::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};
use sim_model::{
    Hunyuan3DVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuanthree_dv2_1_sim_model_0085 as row,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9085",
    identifier: "Hunyuan3Dv2_1AmbiguousFixture",
    ..row::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 167,
        source_architecture: "model_base.Hunyuan3Dv2_1AmbiguousFixture",
        ..row::MODEL_FAMILY_REGISTRATION
    },
];

const SPEC: RowSpec = RowSpec {
    feature_id: row::MODEL_FAMILY_FEATURE_ID,
    identifier: row::MODEL_FAMILY_IDENTIFIER,
    fixture: row::MODEL_FAMILY_FIXTURE,
    module: "hunyuanthree_dv2_1_comfy_model_0085",
    source_ordinal: row::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.Hunyuan3Dv2_1",
    architecture_version: "hunyuan3d-v2-1-flow-dit-v1",
    latent_feature_id: "COMFY-MODEL-0033",
    latent_identifier: "Hunyuan3Dv2_1",
    image_model: "hunyuan3d2_1",
    projection_sha256: row::MODEL_FAMILY_PROJECTION_SHA256,
    variant: Hunyuan3DVariant::V2_1,
    detection_score: 1_000,
};

#[test]
fn val_model_family_row_001_hunyuan3dv2_1_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuan3dv2_1_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuan3dv2_1_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
