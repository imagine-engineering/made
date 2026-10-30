use sim_model::{
    HunyuanVideoVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuanvideo_sim_model_0091 as row_video,
};
use sim_tensor::DType;

use super::generated_hunyuanimage21_sim_model_0089::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9091",
    identifier: "HunyuanVideoAmbiguousFixture",
    ..row_video::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_video::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 140,
        source_architecture: "model_base.HunyuanVideoAmbiguousFixture",
        ..row_video::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: RowSpec = RowSpec {
    feature_id: row_video::MODEL_FAMILY_FEATURE_ID,
    identifier: row_video::MODEL_FAMILY_IDENTIFIER,
    fixture: row_video::MODEL_FAMILY_FIXTURE,
    module: "hunyuanvideo_comfy_model_0091",
    source_ordinal: row_video::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.HunyuanVideo",
    architecture_version: "hunyuan-video-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0037",
    latent_identifier: "HunyuanVideo",
    latent_symbol: "latent_formats.HunyuanVideo",
    clip_tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideoTokenizer",
    clip_model: "sim.text_encoders.hunyuan_video.hunyuan_video_clip",
    projection_sha256: row_video::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_video::MODEL_FAMILY_REGISTRATION,
    artifact_digest: "0091009100910091009100910091009100910091009100910091009100910091",
    variant: HunyuanVideoVariant::Video,
    memory_usage_factor: 1.8,
    sampling_shift: 7.0,
    supported_dtypes: &[DType::Bf16, DType::F32],
    detector_rule_count: 4,
    detector_in_channels: 16,
    detector_patch: &[1, 2, 2],
    detector_context_input: 4_096,
};

#[test]
fn val_model_family_row_001_hunyuanvideo_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideo_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanvideo_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
