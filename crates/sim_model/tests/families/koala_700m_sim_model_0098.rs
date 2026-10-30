use sim_model::{
    ModelFamilyDefinition, ModelFamilyRegistration, SdxlVariant,
    generated_koala_700m_sim_model_0098 as row_700m,
};

use super::generated_koala_1b_sim_model_0097::support::{
    KoalaSpec, assert_execution_contract, assert_source_contract,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9098",
    identifier: "KOALA_700M_AmbiguousFixture",
    ..row_700m::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_700m::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 911,
        source_architecture: "model_base.KOALA_700M_AmbiguousFixture",
        ..row_700m::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: KoalaSpec = KoalaSpec {
    feature_id: row_700m::MODEL_FAMILY_FEATURE_ID,
    identifier: row_700m::MODEL_FAMILY_IDENTIFIER,
    fixture: row_700m::MODEL_FAMILY_FIXTURE,
    module: "koala_700m_comfy_model_0098",
    source_ordinal: row_700m::MODEL_FAMILY_SOURCE_ORDINAL,
    architecture_version: "koala-700m-sdxl-unet-v1",
    variant: SdxlVariant::Koala700M,
    depth: 5,
    middle_depth: -2,
    projection_sha256: row_700m::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_700m::MODEL_FAMILY_REGISTRATION,
};

#[test]
fn val_model_family_row_001_koala_700m_source_detection_configuration_and_provenance()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}

#[test]
fn val_model_family_row_001_koala_700m_mapping_forward_patch_memory_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}
