use sim_model::{
    HunyuanDiTVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuandit1_sim_model_0088 as row_dit1,
};

use super::generated_hunyuandit_sim_model_0087::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9088",
    identifier: "HunyuanDiT1AmbiguousFixture",
    ..row_dit1::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_dit1::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 126,
        source_architecture: "model_base.HunyuanDiT1AmbiguousFixture",
        ..row_dit1::MODEL_FAMILY_REGISTRATION
    },
];

const SPEC: RowSpec = RowSpec {
    feature_id: row_dit1::MODEL_FAMILY_FEATURE_ID,
    identifier: row_dit1::MODEL_FAMILY_IDENTIFIER,
    fixture: row_dit1::MODEL_FAMILY_FIXTURE,
    module: "hunyuandit1_comfy_model_0088",
    source_ordinal: row_dit1::MODEL_FAMILY_SOURCE_ORDINAL,
    architecture_version: "hunyuandit1-v-prediction-transformer-v1",
    image_model: "hydit1",
    projection_sha256: row_dit1::MODEL_FAMILY_PROJECTION_SHA256,
    variant: HunyuanDiTVariant::DiT1,
    expected_memory: 60,
};

#[test]
fn val_model_family_row_001_hunyuandit1_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuandit1_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuandit1_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
