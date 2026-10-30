use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_model::{VaeKernelProfile, audio_vae_source_plan};
use std::{collections::BTreeSet, error::Error, fs, path::Path};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u16,
    fixture_id: String,
    oracle_kind: String,
    production_dependency: bool,
    provenance_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    catalog_contract_ids: Vec<String>,
    profile: String,
    input_sample_rate: u32,
    output_sample_rate: u32,
    sample_ratio_numerator: u64,
    sample_ratio_denominator: u64,
    latent_dimensions: u8,
    latent_layout: String,
    equation_checkpoints: Vec<String>,
}

fn profile(name: &str) -> Result<VaeKernelProfile, Box<dyn Error>> {
    Ok(match name {
        "AudioOobleck44KhzV1" => VaeKernelProfile::AudioOobleck44KhzV1,
        "AudioOobleck48KhzV1" => VaeKernelProfile::AudioOobleck48KhzV1,
        "MusicDcaeV1" => VaeKernelProfile::MusicDcaeV1,
        "MmAudio16KhzV1" => VaeKernelProfile::MmAudio16KhzV1,
        "LtxAudioV1" => VaeKernelProfile::LtxAudioV1,
        "StableAudio3DeepV1" => VaeKernelProfile::StableAudio3DeepV1,
        "StableAudio3ShallowV1" => VaeKernelProfile::StableAudio3ShallowV1,
        other => return Err(format!("unknown audio VAE profile {other}").into()),
    })
}

#[test]
fn val_vae_001_audio_source_ledger_covers_every_catalog_contract() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = workspace.join("crates/sim_test_support/fixtures/models/vae-audio");
    let provenance = fs::read(root.join("provenance.json"))?;
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(root.join("architecture-checkpoints.json"))?)?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.fixture_id,
        "sim-native-audio-vae-source-checkpoints-v1"
    );
    assert_eq!(fixture.oracle_kind, "immutable-source-derived-checkpoints");
    assert!(!fixture.production_dependency);
    assert_eq!(
        fixture.provenance_sha256,
        format!("{:x}", Sha256::digest(provenance))
    );
    assert_eq!(fixture.cases.len(), 7);

    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut case_ids = BTreeSet::new();
    let mut contracts = BTreeSet::new();
    let mut layouts = BTreeSet::new();
    for case in fixture.cases {
        assert!(case_ids.insert(case.id));
        let plan = audio_vae_source_plan(&profile(&case.profile)?)?;
        assert_eq!(plan.input_sample_rate(), case.input_sample_rate);
        assert_eq!(plan.output_sample_rate(), case.output_sample_rate);
        assert_eq!(
            plan.sample_ratio(),
            (case.sample_ratio_numerator, case.sample_ratio_denominator)
        );
        assert_eq!(plan.latent_dimensions(), case.latent_dimensions);
        assert_eq!(plan.equation_checkpoints(), case.equation_checkpoints);
        for contract in case.catalog_contract_ids {
            assert!(catalog.contains(&contract));
            contracts.insert(contract);
        }
        layouts.insert(case.latent_layout);
    }
    assert_eq!(contracts.len(), 5);
    assert_eq!(
        layouts,
        BTreeSet::from(["bct".to_owned(), "bctf-16".to_owned()])
    );
    Ok(())
}

#[test]
fn val_ownership_001_audio_adapter_delegates_foundational_owners() -> Result<(), Box<dyn Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_root.join("src/vae_audio.rs"))?;
    let production = source
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(source.as_str(), |(production, _)| production);
    for forbidden in [
        "CancellationToken::default",
        "struct AudioModelStore",
        "struct AudioAssetService",
        "struct AudioTilePlan",
        "Command::new",
        "python",
        "retry",
        "store.read",
        "store.load",
    ] {
        assert!(
            !production.contains(forbidden),
            "duplicate or forbidden owner: {forbidden}"
        );
    }
    for required in [
        "VaeDescriptor",
        "LoadedModel",
        "VaeKernelProfile",
        "validate_native_vae_backend_binding",
        "load_vision_state_from_model_store_with_context",
        "VaeModelBinding::checked",
        "loaded_mel_spectrogram",
    ] {
        assert!(
            production.contains(required),
            "missing canonical delegation: {required}"
        );
    }
    assert_eq!(
        production
            .matches("load_vision_state_from_model_store_with_context(")
            .count(),
        1,
        "the audio adapter must enter the canonical ModelStore loader exactly once"
    );
    assert_eq!(
        production.matches("fn loaded_mel_spectrogram(").count(),
        1,
        "checkpoint-backed audio spectral projection must have one focused adapter"
    );

    let runtime = fs::read_to_string(crate_root.join("../sim_runtime/src/assets.rs"))?;
    let entry = runtime
        .find("pub fn load_audio_vae_with_context(")
        .ok_or("missing AssetService audio VAE entry point")?;
    let end = runtime[entry..]
        .find("pub fn load_and_execute_audio_vae_with_context(")
        .ok_or("missing AssetService audio execution entry point")?;
    let loader = &runtime[entry..entry + end];
    assert_eq!(loader.matches("self.load_model(").count(), 1);
    assert_eq!(
        loader
            .matches("load_audio_vae_from_model_store_with_context(")
            .count(),
        1
    );
    assert!(loader.contains("require_asset_authorization("));
    Ok(())
}
