use crate::{
    LTX_CLIP_TARGET, LTX_COMPONENT_STATE_SCHEMAS, LTX_COMPONENTS, LTX_FORWARD_PROGRAM,
    LTX_MODEL_OPTIONAL_KEYS, LTX_MODEL_REQUIRED_KEYS, LTX_PREFIXED_STATE_PLAN,
    LTX_SAVED_MODEL_STATE_PLAN, LTX_STANDALONE_STATE_PLAN, LTX_SUPPORTED_DEVICES,
    LTX_SUPPORTED_DTYPES, LTXV_BASE_MEMORY_USAGE_FACTOR, LTXV_LATENT_FORMAT,
    MemoryEstimatorDescriptor, ModelClipTargetSelector, ModelDetectionRule, ModelFamilyDefinition,
    ModelFamilyError, ModelFamilyProfile, ModelFamilyRegistration, ModelFamilyStatePlanCase,
    ModelFamilyStatePlanSelector, ModelLayoutSignature, ModelProbe, ModelStateLayout,
    ModelWeightRule, LtxVariant, ltx_configuration_for_probe,
};

pub const MODEL_FAMILY_IDENTIFIER: &str = "LTXV";
pub const MODEL_FAMILY_FEATURE_ID: &str = "COMFY-MODEL-0103";
pub const MODEL_FAMILY_FIXTURE: &str = "ltxv-comfy-model-0103";
pub const MODEL_FAMILY_SOURCE_ORDINAL: u16 = 32;
pub const MODEL_FAMILY_SOURCE_PATH: &str = "projects/comfy/ComfyUI/comfy/supported_models.py";
pub const MODEL_FAMILY_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
pub const MODEL_FAMILY_PROJECTION_SHA256: &str =
    "00ddbb70c115677110385d245a5d59c87d2e49a9e3c9d09192c3ca74f2a40f4b";
pub const MODEL_FAMILY_MEMORY_USAGE_FACTOR: f64 = LTXV_BASE_MEMORY_USAGE_FACTOR;

pub const TIMESTEP_KEYS: &[&str] = &[
    "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias",
    "model.adaln_single.emb.timestep_embedder.linear_1.bias",
    "adaln_single.emb.timestep_embedder.linear_1.bias",
];
pub const PATCH_KEYS: &[&str] = &[
    "model.diffusion_model.patchify_proj.weight",
    "model.patchify_proj.weight",
    "patchify_proj.weight",
];
pub const ATTENTION_KEYS: &[&str] = &[
    "model.diffusion_model.transformer_blocks.0.attn2.to_k.weight",
    "model.transformer_blocks.0.attn2.to_k.weight",
    "transformer_blocks.0.attn2.to_k.weight",
];
pub const OUTPUT_KEYS: &[&str] = &[
    "model.diffusion_model.proj_out.weight",
    "model.proj_out.weight",
    "proj_out.weight",
];

pub const BASE_DETECTION_RULES: &[ModelDetectionRule] = &[
    ModelDetectionRule::AnyKeyPresent {
        keys: TIMESTEP_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: PATCH_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: ATTENTION_KEYS,
        score: 250,
    },
    ModelDetectionRule::AnyKeyPresent {
        keys: OUTPUT_KEYS,
        score: 250,
    },
];

pub const WEIGHT_RULES: &[ModelWeightRule] = &[ModelWeightRule {
    source_prefix: "model.diffusion_model.",
    target_prefix: "native.",
    required: true,
}];

pub const LAYOUT_SIGNATURES: &[ModelLayoutSignature] = &[
    ModelLayoutSignature {
        layout: ModelStateLayout::PrefixedNative,
        required_keys: &[
            "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias",
            "model.diffusion_model.patchify_proj.weight",
            "model.diffusion_model.transformer_blocks.0.attn2.to_k.weight",
            "model.diffusion_model.proj_out.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::Diffusers,
        required_keys: &[
            "model.adaln_single.emb.timestep_embedder.linear_1.bias",
            "model.patchify_proj.weight",
            "model.transformer_blocks.0.attn2.to_k.weight",
            "model.proj_out.weight",
        ],
        required_prefixes: &[],
    },
    ModelLayoutSignature {
        layout: ModelStateLayout::StandaloneNative,
        required_keys: &[
            "adaln_single.emb.timestep_embedder.linear_1.bias",
            "patchify_proj.weight",
            "transformer_blocks.0.attn2.to_k.weight",
            "proj_out.weight",
        ],
        required_prefixes: &[],
    },
];

pub const STATE_PLAN_CASES: &[ModelFamilyStatePlanCase] = &[
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::PrefixedNative,
        plan: &LTX_PREFIXED_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::Diffusers,
        plan: &LTX_SAVED_MODEL_STATE_PLAN,
    },
    ModelFamilyStatePlanCase {
        layout: ModelStateLayout::StandaloneNative,
        plan: &LTX_STANDALONE_STATE_PLAN,
    },
];

pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {
    feature_id: MODEL_FAMILY_FEATURE_ID,
    identifier: MODEL_FAMILY_IDENTIFIER,
    architecture_version: "ltxv-transformer-v1",
    latent_feature_id: "COMFY-MODEL-0040",
    latent_identifier: "LTXV",
    clip_target: &LTX_CLIP_TARGET,
    components: LTX_COMPONENTS,
    detection_rules: BASE_DETECTION_RULES,
    weight_rules: WEIGHT_RULES,
    required_keys: LTX_MODEL_REQUIRED_KEYS,
    optional_keys: LTX_MODEL_OPTIONAL_KEYS,
    supported_dtypes: LTX_SUPPORTED_DTYPES,
    supported_devices: LTX_SUPPORTED_DEVICES,
    memory_estimator: MemoryEstimatorDescriptor {
        fixed_bytes: 0,
        bytes_per_parameter: 2,
        activation_bytes_per_element: 2,
    },
    forward_program: LTX_FORWARD_PROGRAM,
};

pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {
    definition: &MODEL_FAMILY,
    source_ordinal: 32,
    source_architecture: "model_base.LTXV",
    source_configuration: &[],
    required_state_keys: &[],
    profile_selector: Some(select_profile),
    clip_target_selector: ModelClipTargetSelector::Static(&LTX_CLIP_TARGET),
    state_plan_selector: ModelFamilyStatePlanSelector::Layout {
        signatures: LAYOUT_SIGNATURES,
        cases: STATE_PLAN_CASES,
    },
    component_state_schemas: LTX_COMPONENT_STATE_SCHEMAS,
};

fn select_profile(probe: &ModelProbe) -> Result<ModelFamilyProfile, ModelFamilyError> {
    let configuration = ltx_configuration_for_probe(probe)?;
    if configuration.variant != LtxVariant::Video {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "LTXV row cannot admit audio-video state".to_owned(),
        ));
    }
    if configuration.latent_format.feature_id != LTXV_LATENT_FORMAT.feature_id
        || configuration.latent_format.identifier != LTXV_LATENT_FORMAT.identifier
    {
        return Err(ModelFamilyError::InvalidSelectorOutput(
            "LTXV latent selection drifted".to_owned(),
        ));
    }
    Ok(ModelFamilyProfile::from_definition(&MODEL_FAMILY))
}
