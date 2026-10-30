use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_model::{
    NativeVisionStateKind, NativeVisionStateSpec, VaeKernelProfile, VaeLoaderConfiguration,
    image_vae_source_state_schema,
};
use sim_tensor::DType;
use std::{error::Error, fs, path::Path};

#[derive(Deserialize)]
struct Provenance {
    fixture_id: String,
    oracle_kind: String,
    production_dependency: bool,
    sources: Vec<Source>,
}

#[derive(Deserialize)]
struct Source {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct ArchitectureFixture {
    schema_version: u16,
    fixture_id: String,
    oracle_kind: String,
    production_dependency: bool,
    provenance_sha256: String,
    cases: Vec<ArchitectureCase>,
}

#[derive(Deserialize)]
struct ArchitectureCase {
    id: String,
    catalog_contract_ids: Vec<String>,
    profile: String,
    loader_configuration: Value,
    architecture: String,
    encode: String,
    decode: String,
    spatial_ratios: SpatialRatios,
    state_checkpoints: Vec<StateCheckpoint>,
    equation_checkpoints: Vec<String>,
}

#[derive(Deserialize)]
struct SpatialRatios {
    encode: u64,
    decode: u64,
}

#[derive(Deserialize)]
struct StateCheckpoint {
    name: String,
    shape: Vec<u64>,
}

fn fixture_profile(name: &str) -> Result<VaeKernelProfile, Box<dyn Error>> {
    Ok(match name {
        "TemporalAutoencodingEngineV1" => VaeKernelProfile::TemporalAutoencodingEngineV1,
        "TaesdV1" => VaeKernelProfile::TaesdV1,
        "StableCascadeStageAV1" => VaeKernelProfile::StableCascadeStageAV1,
        "StableCascadeStageCEncoderV1" => VaeKernelProfile::StableCascadeStageCEncoderV1,
        "StableCascadeStageCPreviewerV1" => VaeKernelProfile::StableCascadeStageCPreviewerV1,
        "StableCascadeStageCCombinedV1" => VaeKernelProfile::StableCascadeStageCCombinedV1,
        "HunyuanImageV1" => VaeKernelProfile::HunyuanImageV1,
        "AutoencoderKlV1" => VaeKernelProfile::AutoencoderKlV1,
        "AutoencoderKlX4V1" => VaeKernelProfile::AutoencoderKlX4V1,
        "AutoencoderKlBatchNormV1" => VaeKernelProfile::AutoencoderKlBatchNormV1,
        "ExplicitAutoencoderKlV1" => VaeKernelProfile::ExplicitAutoencoderKlV1,
        "AutoencodingEngineV1" => VaeKernelProfile::AutoencodingEngineV1,
        "AutoencodingEngineX4V1" => VaeKernelProfile::AutoencodingEngineX4V1,
        "AutoencodingEngineBatchNormV1" => VaeKernelProfile::AutoencodingEngineBatchNormV1,
        "PixelSpaceV1" => VaeKernelProfile::PixelSpaceV1,
        other => return Err(format!("unknown fixture profile {other}").into()),
    })
}

fn profile_architecture(profile: &VaeKernelProfile) -> &'static str {
    match profile {
        VaeKernelProfile::PixelSpaceV1 => "sim.pixel_space_convert.PixelspaceConversionVAE.v1",
        VaeKernelProfile::TemporalAutoencodingEngineV1 => {
            "sim.ldm.models.autoencoder.AutoencodingEngine.temporal.v1"
        }
        VaeKernelProfile::TaesdV1 => "sim.taesd.TAESD.v1",
        VaeKernelProfile::StableCascadeStageAV1 => "sim.ldm.cascade.stage_a.StageA.v1",
        VaeKernelProfile::StableCascadeStageCEncoderV1 => {
            "sim.ldm.cascade.stage_c.StageCEncoder.v1"
        }
        VaeKernelProfile::StableCascadeStageCPreviewerV1 => {
            "sim.ldm.cascade.stage_c.StageCPreviewer.v1"
        }
        VaeKernelProfile::StableCascadeStageCCombinedV1 => {
            "sim.ldm.cascade.stage_c.StageCCombined.v1"
        }
        VaeKernelProfile::HunyuanImageV1 => "sim.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1",
        VaeKernelProfile::AutoencoderKlV1
        | VaeKernelProfile::AutoencoderKlX4V1
        | VaeKernelProfile::AutoencoderKlBatchNormV1
        | VaeKernelProfile::ExplicitAutoencoderKlV1
        | VaeKernelProfile::AutoencodingEngineV1
        | VaeKernelProfile::AutoencodingEngineX4V1
        | VaeKernelProfile::AutoencodingEngineBatchNormV1 => {
            "sim.ldm.models.autoencoder.AutoencoderKL.v1"
        }
        _ => "unsupported",
    }
}

fn expected_equation_checkpoints(case_id: &str) -> Option<&'static [&'static str]> {
    Some(match case_id {
        "temporal-autoencoding-engine-image" => &[
            "frame_batch_to_temporal_sequence",
            "temporal_residual_alpha_mix",
        ],
        "taesd-sd15" => &[
            "encoder_block_residual",
            "decoder_tanh_bound",
            "latent_scale_shift",
        ],
        "taesd-flux2-128" => &["flux2_pixel_unshuffle_encode", "flux2_pixel_shuffle_decode"],
        "stable-cascade-stage-a" => &[
            "pixel_unshuffle_encode",
            "depthwise_replication_pad",
            "gelu_exact",
            "pixel_shuffle_decode",
        ],
        "stable-cascade-stage-c-encoder" => &[
            "channel_standardization",
            "efficientnet_v2_s_features",
            "mapper_batch_norm",
        ],
        "stable-cascade-stage-c-previewer" => &[
            "conv_transpose_upsample",
            "gelu_exact",
            "batch_norm_inference",
        ],
        "stable-cascade-stage-c-combined" => {
            &["prefixed_encoder_state", "prefixed_previewer_state"]
        }
        "hunyuan-image" => &[
            "downsample_pixel_unshuffle_residual",
            "channel_group_mean",
            "upsample_pixel_shuffle_residual",
        ],
        "autoencoder-kl-standard" => &[
            "bottom_right_downsample_pad",
            "diagonal_gaussian_mode",
            "nearest_upsample",
            "unit_interval_process_output",
        ],
        "autoencoder-kl-x4" => &["x4_spatial_geometry"],
        "autoencoder-kl-batch-normalized" => &[
            "latent_pixel_unshuffle",
            "batch_norm_latent",
            "latent_pixel_shuffle",
        ],
        "autoencoder-kl-asymmetric-decoder" => &[
            "separate_decoder_base_channels",
            "embed_dimension_projection",
        ],
        "autoencoding-engine-standard" => {
            &["diagonal_gaussian_regularizer_mode", "no_quant_projection"]
        }
        "autoencoding-engine-x4" => &["x4_spatial_geometry", "no_quant_projection"],
        "autoencoding-engine-batch-normalized" => &[
            "latent_pixel_unshuffle",
            "batch_norm_latent",
            "no_quant_projection",
        ],
        "pixel-space" => &["unit_to_signed_encode", "signed_to_unit_decode"],
        "explicit-autoencoder-kl" => &[
            "average_pool_without_convolution",
            "nearest_upsample_without_convolution",
            "decoder_tanh",
            "attention_resolution",
        ],
        _ => return None,
    })
}

#[test]
fn val_vae_001_image_profiles_and_state_schema_are_typed() {
    let profiles = [
        VaeKernelProfile::TemporalAutoencodingEngineV1,
        VaeKernelProfile::TaesdV1,
        VaeKernelProfile::StableCascadeStageAV1,
        VaeKernelProfile::StableCascadeStageCEncoderV1,
        VaeKernelProfile::StableCascadeStageCPreviewerV1,
        VaeKernelProfile::StableCascadeStageCCombinedV1,
        VaeKernelProfile::HunyuanImageV1,
        VaeKernelProfile::AutoencoderKlV1,
        VaeKernelProfile::AutoencoderKlX4V1,
        VaeKernelProfile::AutoencoderKlBatchNormV1,
        VaeKernelProfile::AutoencodingEngineV1,
        VaeKernelProfile::AutoencodingEngineX4V1,
        VaeKernelProfile::AutoencodingEngineBatchNormV1,
        VaeKernelProfile::PixelSpaceV1,
    ];
    assert_eq!(profiles.len(), 14);
    let parameter = NativeVisionStateSpec {
        name: "encoder.conv_in.weight".to_owned(),
        shape: vec![4, 3, 3, 3],
        dtype: DType::F32,
        kind: NativeVisionStateKind::Parameter,
    };
    let buffer = NativeVisionStateSpec {
        name: "bn.running_mean".to_owned(),
        shape: vec![16],
        dtype: DType::F32,
        kind: NativeVisionStateKind::Buffer,
    };
    assert_ne!(parameter.kind, buffer.kind);
}

#[test]
fn val_vae_001_adapter_preserves_canonical_ownership_boundaries() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(root.join("src/vae_image.rs"))?;
    let source = production_vae_image_source(&source);
    for required in [
        "VaeModelBinding::checked",
        "NativeVae::checked_kernel",
        "load_vision_state_from_model_store_with_context",
        "load_vision_state_with_sibling_namespaces_from_model_store_with_context",
        "load_stage_c_efficientnet_feature_module_from_model_store_with_context",
        "efficientnet_v2_s_features_from_module_with_context",
        "canonical_vision_model_store_dtype",
        "source_state_manifest",
        "legacy_quantization_source_names",
        "admit_source_manifest",
        "gelu_scalar_exact_native",
        "backend.convolution",
        "backend.resize",
        "context.check()?",
    ] {
        assert!(
            source.contains(required),
            "missing canonical delegation: {required}"
        );
    }
    for forbidden in [
        "CancellationToken::default",
        "Command::new",
        "std::process",
        "python",
        "retry(",
        "workspace_vec",
        "NativeImageVaeStateKind",
        "NativeImageVaeStateSpec",
        "fn storage_dtype",
    ] {
        assert!(
            !source.contains(forbidden),
            "duplicate or forbidden owner: {forbidden}"
        );
    }
    Ok(())
}

fn production_vae_image_source(source: &str) -> &str {
    // Test-local cancellation tokens do not confer production ownership, but
    // an earlier isolated test attribute must not hide later production code.
    source
        .split_once("\n#[cfg(test)]\nmod tests {")
        .map_or(source, |(production, _)| production)
}

#[test]
fn vae_image_ownership_scan_excludes_only_the_terminal_test_module() {
    let source = "canonical();\n#[cfg(test)]\nfn fixture() {}\nCancellationToken::default();\n#[cfg(test)]\nmod tests {\nCommand::new();\n}";
    let production = production_vae_image_source(source);
    assert!(production.contains("canonical();"));
    assert!(production.contains("CancellationToken::default();"));
    assert!(!production.contains("Command::new();"));
    assert_eq!(
        production_vae_image_source("CancellationToken::default();"),
        "CancellationToken::default();"
    );
}

#[test]
fn val_vae_001_pinned_source_provenance_matches_checkout() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("crates/sim_test_support/fixtures/models/vae-image/provenance.json");
    let provenance: Provenance = serde_json::from_slice(&fs::read(fixture)?)?;
    assert_eq!(
        provenance.fixture_id,
        "sim-native-image-vae-source-manifests-v1"
    );
    assert_eq!(
        provenance.oracle_kind,
        "development-time-source-conformance"
    );
    assert!(!provenance.production_dependency);
    assert_eq!(provenance.sources.len(), 9);
    for source in provenance.sources {
        let bytes = fs::read(root.join(&source.path))?;
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), source.sha256);
    }
    Ok(())
}

#[test]
fn val_vae_001_image_source_fixture_covers_every_native_profile() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_root = root.join("crates/sim_test_support/fixtures/models/vae-image");
    let provenance_bytes = fs::read(fixture_root.join("provenance.json"))?;
    let fixture: ArchitectureFixture = serde_json::from_slice(&fs::read(
        fixture_root.join("architecture-checkpoints.json"),
    )?)?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.fixture_id,
        "sim-native-image-vae-source-checkpoints-v1"
    );
    assert_eq!(fixture.oracle_kind, "immutable-source-derived-checkpoints");
    assert!(!fixture.production_dependency);
    assert_eq!(
        fixture.provenance_sha256,
        format!("{:x}", Sha256::digest(provenance_bytes))
    );
    assert_eq!(fixture.cases.len(), 17);

    let catalog = fs::read_to_string(
        root.join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut covered_profiles = std::collections::BTreeSet::new();
    let mut covered_contracts = std::collections::BTreeSet::new();
    let mut equation_checkpoint_count = 0;
    let mut unique_equation_checkpoints = std::collections::BTreeSet::new();
    for case in fixture.cases {
        assert!(covered_profiles.insert(case.id.clone()));
        let profile = fixture_profile(&case.profile)?;
        let configuration: VaeLoaderConfiguration =
            serde_json::from_value(case.loader_configuration)?;
        let schema = image_vae_source_state_schema(&profile, &configuration, DType::F32)?;
        assert_eq!(case.architecture, profile_architecture(&profile));
        assert!(matches!(
            case.encode.as_str(),
            "available" | "typed_unavailable"
        ));
        assert!(matches!(
            case.decode.as_str(),
            "available" | "typed_unavailable"
        ));
        assert!(case.spatial_ratios.encode > 0 && case.spatial_ratios.decode > 0);
        assert!(!case.state_checkpoints.is_empty());
        let expected_equations = expected_equation_checkpoints(&case.id)
            .ok_or_else(|| format!("{} has no normative equation manifest", case.id))?;
        assert_eq!(
            case.equation_checkpoints, expected_equations,
            "{} equation checkpoints changed",
            case.id
        );
        equation_checkpoint_count += case.equation_checkpoints.len();
        unique_equation_checkpoints.extend(case.equation_checkpoints.iter().cloned());
        for checkpoint in case.state_checkpoints {
            let state = schema
                .iter()
                .find(|state| state.name == checkpoint.name)
                .ok_or_else(|| format!("{} is missing checkpoint {}", case.id, checkpoint.name))?;
            assert_eq!(state.shape, checkpoint.shape, "{}", case.id);
        }
        for contract_id in case.catalog_contract_ids {
            assert!(catalog.contains(&contract_id), "missing {contract_id}");
            covered_contracts.insert(contract_id);
        }
    }
    assert_eq!(covered_contracts.len(), 11);
    assert_eq!(covered_profiles.len(), 17);
    assert_eq!(equation_checkpoint_count, 45);
    assert_eq!(unique_equation_checkpoints.len(), 39);
    Ok(())
}
