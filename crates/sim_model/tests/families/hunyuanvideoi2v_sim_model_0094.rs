use sim_model::{
    HunyuanVideoVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuanvideoi2v_sim_model_0094 as row_i2v,
};
use sim_tensor::DType;

use super::generated_hunyuanvideo15_sr_distilled_sim_model_0093::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9094",
    identifier: "HunyuanVideoI2VAmbiguousFixture",
    ..row_i2v::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_i2v::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 139,
        source_architecture: "model_base.HunyuanVideoI2VAmbiguousFixture",
        ..row_i2v::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: RowSpec = RowSpec {
    feature_id: row_i2v::MODEL_FAMILY_FEATURE_ID,
    identifier: row_i2v::MODEL_FAMILY_IDENTIFIER,
    fixture: row_i2v::MODEL_FAMILY_FIXTURE,
    module: "hunyuanvideoi2v_comfy_model_0094",
    source_ordinal: row_i2v::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.HunyuanVideoI2V",
    architecture_version: "hunyuan-video-i2v-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0037",
    latent_identifier: "HunyuanVideo",
    latent_symbol: "latent_formats.HunyuanVideo",
    clip_tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideoTokenizer",
    clip_model: "sim.text_encoders.hunyuan_video.hunyuan_video_clip",
    projection_sha256: row_i2v::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_i2v::MODEL_FAMILY_REGISTRATION,
    artifact_digest: "0094009400940094009400940094009400940094009400940094009400940094",
    variant: HunyuanVideoVariant::VideoI2V,
    memory_usage_factor: 1.8,
    sampling_shift: 7.0,
    supported_dtypes: &[DType::Bf16, DType::F32],
    detector_rule_count: 4,
    detector_in_channels: 33,
    detector_patch: &[1, 2, 2],
    detector_context_input: 4_096,
};

#[test]
fn val_model_family_row_001_hunyuanvideoi2v_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideoi2v_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideoi2v_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
