use sim_model::{
    HunyuanVideoVariant, ModelFamilyDefinition, ModelFamilyRegistration,
    generated_hunyuanimage21refiner_sim_model_0090 as row_refiner,
};
use sim_tensor::DType;

use super::generated_hunyuanimage21_sim_model_0089::{
    RowSpec, assert_execution_contract, assert_failure_contract, assert_source_contract,
};

static AMBIGUOUS_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9090",
    identifier: "HunyuanImage21RefinerAmbiguousFixture",
    ..row_refiner::MODEL_FAMILY
};

static AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    row_refiner::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &AMBIGUOUS_FAMILY,
        source_ordinal: 136,
        source_architecture: "model_base.HunyuanImage21RefinerAmbiguousFixture",
        ..row_refiner::MODEL_FAMILY_REGISTRATION
    },
];

pub(super) const SPEC: RowSpec = RowSpec {
    feature_id: row_refiner::MODEL_FAMILY_FEATURE_ID,
    identifier: row_refiner::MODEL_FAMILY_IDENTIFIER,
    fixture: row_refiner::MODEL_FAMILY_FIXTURE,
    module: "hunyuanimage21refiner_comfy_model_0090",
    source_ordinal: row_refiner::MODEL_FAMILY_SOURCE_ORDINAL,
    source_architecture: "model_base.HunyuanImage21Refiner",
    architecture_version: "hunyuan-image-2.1-refiner-flow-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0036",
    latent_identifier: "HunyuanImage21Refiner",
    latent_symbol: "latent_formats.HunyuanImage21Refiner",
    clip_tokenizer: "sim.text_encoders.hunyuan_video.HunyuanVideoTokenizer",
    clip_model: "sim.text_encoders.hunyuan_video.hunyuan_video_clip",
    projection_sha256: row_refiner::MODEL_FAMILY_PROJECTION_SHA256,
    registration: row_refiner::MODEL_FAMILY_REGISTRATION,
    artifact_digest: "0090009000900090009000900090009000900090009000900090009000900090",
    variant: HunyuanVideoVariant::Image21Refiner,
    memory_usage_factor: 1.8,
    sampling_shift: 4.0,
    supported_dtypes: &[DType::Bf16, DType::F32],
    detector_rule_count: 4,
    detector_in_channels: 64,
    detector_patch: &[1, 1, 1],
    detector_context_input: 4_096,
};

#[test]
fn val_model_family_row_001_hunyuanimage21refiner_source_detection_provenance_and_ownership()
-> Result<(), Box<dyn std::error::Error>> {
    assert_source_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanimage21refiner_mapping_forward_patch_and_memory()
-> Result<(), Box<dyn std::error::Error>> {
    assert_execution_contract(SPEC)
}

#[test]
fn val_model_family_row_001_hunyuanimage21refiner_dtype_device_and_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    assert_failure_contract(SPEC, &AMBIGUOUS_REGISTRATIONS)
}
