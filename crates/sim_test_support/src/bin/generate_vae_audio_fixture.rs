use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Provenance {
    sources: Vec<Source>,
}

#[derive(Deserialize)]
struct Source {
    path: String,
    sha256: String,
}

fn case(
    id: &str,
    contract: &str,
    profile: &str,
    rates: [u32; 2],
    ratio: [u64; 2],
    latent_dimensions: u8,
    latent_layout: &str,
    equations: &[&str],
) -> Value {
    json!({
        "id": id,
        "catalog_contract_ids": [contract],
        "profile": profile,
        "input_sample_rate": rates[0],
        "output_sample_rate": rates[1],
        "sample_ratio_numerator": ratio[0],
        "sample_ratio_denominator": ratio[1],
        "latent_dimensions": latent_dimensions,
        "latent_layout": latent_layout,
        "equation_checkpoints": equations,
    })
}

fn main() -> Result<()> {
    let mode = match std::env::args().nth(1).as_deref() {
        None => "write",
        Some("--check") => "check",
        Some(argument) => bail!("unsupported argument {argument}"),
    };
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("workspace root is unavailable")?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/vae-audio");
    let provenance_bytes = fs::read(root.join("provenance.json"))?;
    let provenance: Provenance = serde_json::from_slice(&provenance_bytes)?;
    for source in provenance.sources {
        let bytes = fs::read(workspace.join(&source.path))?;
        if format!("{:x}", Sha256::digest(bytes)) != source.sha256 {
            bail!("pinned source digest changed for {}", source.path);
        }
    }

    const OOBLECK: &str = "conditioning-vae-architecture-sd-audiooobleckvae-6c64e9cf";
    const MUSIC: &str =
        "conditioning-vae-architecture-sd-comfy-ldm-ace-vae-music-dcae-pipeline-musicdcae-adb8d824";
    const MMAUDIO: &str = "conditioning-vae-architecture-sd-comfy-ldm-mmaudio-vae-autoencoder-audioautoencoder-929e9c62";
    const LTX: &str =
        "conditioning-vae-architecture-sd-comfy-ldm-lightricks-vae-audio-vae-audiovae-1024d260";
    const SA3: &str =
        "conditioning-vae-architecture-sd-comfy-ldm-audio-vae-sa3-sa3audiovae-637f1b5c";
    let oobleck = [
        "weight_normalized_conv1d_residual_stack",
        "snake_beta_alias_free_activation",
        "diagonal_gaussian_reparameterized_sample",
        "replicate_channel_padding",
    ];
    let music = [
        "waveform_resample_to_44100",
        "checkpoint_hann_window_and_mel_filter_bank",
        "log_mel_normalize_minus_11_to_3",
        "dcae_latent_affine_shift_minus_1_9091_scale_0_1786",
        "extra_1d_channel_reshape_16",
        "audio_chunk_multiple_4096",
    ];
    let mmaudio = [
        "stereo_mean_then_resample_44100_to_16000",
        "checkpoint_hann_window_and_mel_filter_bank",
        "stft_1024_hop_256_mel_80",
        "per_band_mean_std_normalization",
        "diagonal_gaussian_mode",
        "vocoder_then_resample_16000_to_44100",
    ];
    let ltx = [
        "source_rate_to_configured_sample_rate",
        "stft_configured_hop_and_mel_bins",
        "causal_latent_length_ceil",
        "per_channel_latent_statistics",
        "extra_1d_channel_reshape_16",
        "configured_vocoder_output_rate",
    ];
    let sa3 = [
        "zero_pad_to_patch_256",
        "patch_channels_2_times_256",
        "variable_stride_16",
        "softnorm_bottleneck",
        "bounded_transformer_chunks",
    ];
    let cases = vec![
        case(
            "oobleck-44k",
            OOBLECK,
            "AudioOobleck44KhzV1",
            [44_100, 44_100],
            [2_048, 1],
            1,
            "bct",
            &oobleck,
        ),
        case(
            "oobleck-48k",
            OOBLECK,
            "AudioOobleck48KhzV1",
            [48_000, 48_000],
            [1_920, 1],
            1,
            "bct",
            &oobleck,
        ),
        case(
            "music-dcae",
            MUSIC,
            "MusicDcaeV1",
            [44_100, 44_100],
            [4_096, 1],
            2,
            "bctf-16",
            &music,
        ),
        case(
            "mmaudio-16k",
            MMAUDIO,
            "MmAudio16KhzV1",
            [44_100, 44_100],
            [141_120, 100],
            1,
            "bct",
            &mmaudio,
        ),
        case(
            "ltx-audio",
            LTX,
            "LtxAudioV1",
            [44_100, 16_000],
            [1_764, 1],
            2,
            "bctf-16",
            &ltx,
        ),
        case(
            "stable-audio-3-deep",
            SA3,
            "StableAudio3DeepV1",
            [44_100, 44_100],
            [4_096, 1],
            1,
            "bct",
            &sa3,
        ),
        case(
            "stable-audio-3-shallow",
            SA3,
            "StableAudio3ShallowV1",
            [44_100, 44_100],
            [4_096, 1],
            1,
            "bct",
            &sa3,
        ),
    ];
    let artifact = json!({
        "schema_version": 1,
        "fixture_id": "comfy-native-audio-vae-source-checkpoints-v1",
        "oracle_kind": "immutable-source-derived-checkpoints",
        "production_dependency": false,
        "provenance_sha256": format!("{:x}", Sha256::digest(&provenance_bytes)),
        "cases": cases,
    });
    let output = root.join("architecture-checkpoints.json");
    let encoded = serde_json::to_vec_pretty(&artifact)?;
    if mode == "check" {
        if fs::read(&output)? != encoded {
            bail!("{} is stale", output.display());
        }
    } else {
        fs::write(output, encoded)?;
    }
    Ok(())
}
