use sim_model::{
    ModelFamilyDefinition, ModelFamilyError, ModelFamilyRegistration, ModelFamilyRegistry,
    ModelProbe,
    generated_stable_cascade_c_sim_model_0135 as cascade_c,
};
use sim_tensor::DType;

use super::generated_stable_cascade_b_sim_model_0134::{
    CascadeFamilyCase, run_execution_validation, run_source_validation,
};

static C_REGISTRATIONS: [ModelFamilyRegistration; 1] =
    [cascade_c::MODEL_FAMILY_REGISTRATION];
static C_AMBIGUOUS_DEFINITION: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: "COMFY-MODEL-9135",
    identifier: "StableCascadeCAmbiguousFixture",
    ..cascade_c::MODEL_FAMILY
};
static C_AMBIGUOUS_REGISTRATIONS: [ModelFamilyRegistration; 2] = [
    cascade_c::MODEL_FAMILY_REGISTRATION,
    ModelFamilyRegistration {
        definition: &C_AMBIGUOUS_DEFINITION,
        source_ordinal: 115,
        source_architecture: "model_base.StableCascadeCAmbiguousFixture",
        ..cascade_c::MODEL_FAMILY_REGISTRATION
    },
];

#[test]
fn val_model_family_row_001_stable_cascade_c_source_configuration_and_state_transform()
-> Result<(), Box<dyn std::error::Error>> {
    let probe = c_probe(2_048);
    let configuration = cascade_c::configuration_for_probe(&probe)?;
    assert_eq!(configuration.variant, cascade_c::StableCascadeCVariant::Full);
    assert_eq!(configuration.conditioning_dimension, 2_048);
    assert_eq!(configuration.hidden_dimensions, [2_048, 2_048]);
    assert_eq!(configuration.attention_heads, [32, 32]);
    assert_eq!(configuration.down_blocks, [8, 24]);
    assert_eq!(configuration.up_blocks, [24, 8]);

    let lite = cascade_c::configuration_for_probe(&c_probe(1_536))?;
    assert_eq!(lite.variant, cascade_c::StableCascadeCVariant::Lite);
    assert_eq!(lite.conditioning_dimension, 1_536);
    assert_eq!(lite.hidden_dimensions, [1_536, 1_536]);
    assert_eq!(lite.attention_heads, [24, 24]);
    assert_eq!(lite.down_blocks, [4, 12]);
    assert_eq!(lite.up_blocks, [12, 4]);
    assert!(matches!(
        ModelFamilyRegistry::checked_registrations(&C_AMBIGUOUS_REGISTRATIONS)?
            .detect(&probe),
        Err(ModelFamilyError::AmbiguousDetection { score: 1_000, .. })
    ));
    run_source_validation(&case())
}

#[test]
fn val_model_family_row_001_stable_cascade_c_forward_patch_memory_and_matrix()
-> Result<(), Box<dyn std::error::Error>> {
    run_execution_validation(&case())
}

fn case() -> CascadeFamilyCase {
    CascadeFamilyCase {
        definition: &cascade_c::MODEL_FAMILY,
        registration: cascade_c::MODEL_FAMILY_REGISTRATION,
        registrations: &C_REGISTRATIONS,
        identifier: cascade_c::MODEL_FAMILY_IDENTIFIER,
        feature_id: cascade_c::MODEL_FAMILY_FEATURE_ID,
        fixture: cascade_c::MODEL_FAMILY_FIXTURE,
        module: "stable_cascade_c_comfy_model_0135",
        source_ordinal: cascade_c::MODEL_FAMILY_SOURCE_ORDINAL,
        source_architecture: cascade_c::SOURCE_ARCHITECTURE,
        projection_sha256: cascade_c::MODEL_FAMILY_PROJECTION_SHA256,
        latent_feature_id: "COMFY-MODEL-0044",
        latent_identifier: "SC_Prior",
        architecture_version: "stable-cascade-stage-c-v1",
        variant_marker: "model.diffusion_model.clip_txt_mapper.weight",
        variant_dimension: 0,
        supported_dtypes: &[DType::Bf16, DType::F32],
        component_count: 4,
        has_vision: true,
        malformed_width: 1_537,
        patch_key: "native.embedding.1.weight",
        focused_memory_bytes: 12_448,
        validate_configuration: |probe| cascade_c::configuration_for_probe(probe).map(|_| ()),
    }
}

fn c_probe(width: u64) -> ModelProbe {
    ModelProbe {
        tensor_shapes: std::collections::BTreeMap::from([
            ("model.diffusion_model.clf.1.weight".to_owned(), vec![2, 2]),
            (
                "model.diffusion_model.clip_txt_mapper.weight".to_owned(),
                vec![width, 2],
            ),
            (
                "model.diffusion_model.embedding.1.weight".to_owned(),
                vec![2, 2],
            ),
            (
                "model.diffusion_model.down_blocks.0.0.channelwise.0.weight".to_owned(),
                vec![2, 2],
            ),
        ]),
        metadata: std::collections::BTreeMap::new(),
    }
}
