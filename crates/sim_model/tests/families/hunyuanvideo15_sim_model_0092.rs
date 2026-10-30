use sim_model::{
    HunyuanVideoVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuanvideo15_sim_model_0092 as row_video15,
};
use sim_tensor::DType;

use super::generated_hunyuanimage21_sim_model_0089::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9092",
    identifier: "HunyuanVideo15AmbiguousFixture",
    ..row_video15::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_video15::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 135,
        source_architecture: "model_base.HunyuanVideo15AmbiguousFixture",
        ..row_video15::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: RowSpec = RowSpec {
    feature_id: row_video15::MODEL_FAMILY_FEATURE_ID,
    identifier: row_video15::MODEL_FAMILY_IDENTIFIER,
    fixture: row_video15::MODEL_FAMILY_FIXTURE,
    module: "hunyuanvideo15_comfy_model_0092",
    source_ordinal: row_video15::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.HunyuanVideo15",
    architecture_version: "hunyuan-video-1.5-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0038",
    latent_identifier: "HunyuanVideo15",
    latent_symbol: "latent_formats.HunyuanVideo15",
    clip_tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideo15Tokenizer",
    clip_model: "sim.text_encoders.hunyuan_image.te",
    projection_sha256: row_video15::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_video15::MODEL_FAMILY_REGISTRATION,
    artifact_digest: "0092009200920092009200920092009200920092009200920092009200920092",
    variant: HunyuanVideoVariant::Video15,
    memory_usage_factor: 4.0,
    sampling_shift: 7.0,
    supported_dtypes: &[DType::F16, DType::Bf16, DType::F32],
    detector_rule_count: 5,
    detector_in_channels: 32,
    detector_patch: &[1, 2, 2],
    detector_context_input: 3_584,
};

#[test]
fn val_model_family_row_001_hunyuanvideo15_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideo15_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideo15_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
