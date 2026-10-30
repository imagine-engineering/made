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

fn checkpoint(name: &str, shape: &[u64]) -> Value {
    json!({"name": name, "shape": shape})
}

fn case(
    id: &str,
    contract_ids: &[&str],
    profile: &str,
    configuration: Value,
    architecture: &str,
    encode: &str,
    decode: &str,
    ratios: [u64; 2],
    checkpoints: Vec<Value>,
    equations: &[&str],
) -> Value {
    json!({
        "id": id,
        "catalog_contract_ids": contract_ids,
        "profile": profile,
        "loader_configuration": configuration,
        "architecture": architecture,
        "encode": encode,
        "decode": decode,
        "spatial_ratios": {"encode": ratios[0], "decode": ratios[1]},
        "state_checkpoints": checkpoints,
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
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/vae-image");
    let provenance_path = root.join("provenance.json");
    let provenance_bytes = fs::read(&provenance_path)?;
    let provenance: Provenance = serde_json::from_slice(&provenance_bytes)?;
    for source in provenance.sources {
        let bytes = fs::read(workspace.join(&source.path))
            .with_context(|| format!("read pinned source {}", source.path))?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != source.sha256 {
            bail!("pinned source digest changed for {}", source.path);
        }
    }

    let automatic = json!({"kind": "automatic"});
    let default = |x4: bool, batch_norm_latent: bool, asymmetric: Option<u64>, embed_dim| {
        json!({
            "kind": "default_kl",
            "x4": x4,
            "legacy_prefix_rewrite": true,
            "batch_norm_latent": batch_norm_latent,
            "asymmetric_decoder_channels": asymmetric,
            "embed_dim": embed_dim,
        })
    };
    let explicit_parameters = json!({
        "ddconfig": {
            "attn_resolutions": [4],
            "batch_norm_latent": true,
            "ch": 32,
            "ch_mult": [1, 2],
            "double_z": true,
            "in_channels": 3,
            "num_res_blocks": 1,
            "out_ch": 3,
            "resamp_with_conv": false,
            "resolution": 8,
            "z_channels": 4
        },
        "decoder_ddconfig": {
            "attn_resolutions": [],
            "ch": 64,
            "ch_mult": [1, 2],
            "double_z": true,
            "in_channels": 3,
            "num_res_blocks": 1,
            "out_ch": 3,
            "resamp_with_conv": false,
            "resolution": 8,
            "tanh_out": true,
            "z_channels": 4
        },
        "embed_dim": 4
    });
    let explicit_json = serde_json::to_string(&explicit_parameters)?;
    let explicit = json!({
        "kind": "explicit_autoencoder_kl",
        "params_sha256": format!("{:x}", Sha256::digest(explicit_json.as_bytes())),
        "params_json": explicit_json,
    });

    let temporal_contract = "conditioning-vae-architecture-sd-autoencodingengine-5bca3e44";
    let taesd_contract = "conditioning-vae-architecture-sd-comfy-taesd-taesd-taesd-d0ccd20f";
    let stage_a_contract = "conditioning-vae-architecture-sd-stagea-1b767df0";
    let stage_c_encoder_contract = "conditioning-vae-architecture-sd-stagec-coder-94678b59";
    let stage_c_previewer_contract = "conditioning-vae-architecture-sd-stagec-coder-84281997";
    let stage_c_combined_contract = "conditioning-vae-architecture-sd-stagec-coder-4801dafb";
    let hunyuan_contract = "conditioning-vae-architecture-sd-autoencodingengine-b7150e80";
    let kl_contract = "conditioning-vae-architecture-sd-autoencoderkl-69aa0015";
    let engine_contract = "conditioning-vae-architecture-sd-autoencodingengine-9f13006f";
    let pixel_contract = "conditioning-vae-architecture-sd-comfy-pixel-space-convert-pixelspaceconversionvae-3ec12255";
    let explicit_contract = "conditioning-vae-architecture-sd-autoencoderkl-61f06f31";

    let cases = vec![
        case(
            "temporal-autoencoding-engine-image",
            &[temporal_contract],
            "TemporalAutoencodingEngineV1",
            automatic.clone(),
            "comfy.ldm.models.autoencoder.AutoencodingEngine.temporal.v1",
            "available",
            "available",
            [8, 8],
            vec![
                checkpoint("decoder.mid.block_1.mix_factor", &[1]),
                checkpoint("decoder.conv_out.time_mix_conv.weight", &[3, 3, 3, 1, 1]),
            ],
            &[
                "frame_batch_to_temporal_sequence",
                "temporal_residual_alpha_mix",
            ],
        ),
        case(
            "taesd-sd15",
            &[taesd_contract],
            "TaesdV1",
            json!({"kind":"taesd","latent_channels":4,"metadata_override":false}),
            "comfy.taesd.TAESD.v1",
            "available",
            "available",
            [8, 8],
            vec![
                checkpoint("taesd_encoder.0.weight", &[64, 3, 3, 3]),
                checkpoint("taesd_encoder.14.weight", &[4, 64, 3, 3]),
                checkpoint("taesd_decoder.1.weight", &[64, 4, 3, 3]),
                checkpoint("taesd_decoder.19.weight", &[3, 64, 3, 3]),
            ],
            &[
                "encoder_block_residual",
                "decoder_tanh_bound",
                "latent_scale_shift",
            ],
        ),
        case(
            "taesd-flux2-128",
            &[taesd_contract],
            "TaesdV1",
            json!({"kind":"taesd","latent_channels":128,"metadata_override":true}),
            "comfy.taesd.TAESD.v1",
            "available",
            "available",
            [16, 16],
            vec![
                checkpoint("taesd_encoder.14.weight", &[32, 64, 3, 3]),
                checkpoint("taesd_encoder.11.pool.0.weight", &[256, 64, 1, 1]),
                checkpoint("taesd_decoder.1.weight", &[64, 32, 3, 3]),
            ],
            &["flux2_pixel_unshuffle_encode", "flux2_pixel_shuffle_decode"],
        ),
        case(
            "stable-cascade-stage-a",
            &[stage_a_contract],
            "StableCascadeStageAV1",
            automatic.clone(),
            "comfy.ldm.cascade.stage_a.StageA.v1",
            "available",
            "available",
            [4, 4],
            vec![
                checkpoint("down_blocks.0.depthwise.1.weight", &[192, 1, 3, 3]),
                checkpoint("down_blocks.0.channelwise.0.weight", &[768, 192]),
                checkpoint("vquantizer.codebook.weight", &[8192, 4]),
                checkpoint("up_blocks.13.weight", &[384, 192, 4, 4]),
            ],
            &[
                "pixel_unshuffle_encode",
                "depthwise_replication_pad",
                "gelu_exact",
                "pixel_shuffle_decode",
            ],
        ),
        case(
            "stable-cascade-stage-c-encoder",
            &[stage_c_encoder_contract],
            "StableCascadeStageCEncoderV1",
            automatic.clone(),
            "comfy.ldm.cascade.stage_c.StageCEncoder.v1",
            "available",
            "typed_unavailable",
            [32, 1],
            vec![
                checkpoint("mapper.0.weight", &[16, 1280, 1, 1]),
                checkpoint("mapper.1.running_mean", &[16]),
                checkpoint("mean", &[3]),
                checkpoint("std", &[3]),
            ],
            &[
                "channel_standardization",
                "efficientnet_v2_s_features",
                "mapper_batch_norm",
            ],
        ),
        case(
            "stable-cascade-stage-c-previewer",
            &[stage_c_previewer_contract],
            "StableCascadeStageCPreviewerV1",
            automatic.clone(),
            "comfy.ldm.cascade.stage_c.StageCPreviewer.v1",
            "typed_unavailable",
            "available",
            [1, 8],
            vec![
                checkpoint("blocks.0.weight", &[512, 16, 1, 1]),
                checkpoint("blocks.6.weight", &[512, 256, 2, 2]),
                checkpoint("blocks.24.weight", &[3, 128, 1, 1]),
            ],
            &[
                "conv_transpose_upsample",
                "gelu_exact",
                "batch_norm_inference",
            ],
        ),
        case(
            "stable-cascade-stage-c-combined",
            &[stage_c_combined_contract],
            "StableCascadeStageCCombinedV1",
            automatic.clone(),
            "comfy.ldm.cascade.stage_c.StageCCombined.v1",
            "available",
            "available",
            [32, 8],
            vec![
                checkpoint("encoder.mapper.0.weight", &[16, 1280, 1, 1]),
                checkpoint("previewer.blocks.24.weight", &[3, 128, 1, 1]),
            ],
            &["prefixed_encoder_state", "prefixed_previewer_state"],
        ),
        case(
            "hunyuan-image",
            &[hunyuan_contract],
            "HunyuanImageV1",
            automatic.clone(),
            "comfy.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1",
            "available",
            "available",
            [32, 32],
            vec![
                checkpoint("encoder.conv_in.weight", &[128, 3, 3, 3]),
                checkpoint("encoder.down.0.downsample.conv.weight", &[64, 128, 3, 3]),
                checkpoint("decoder.conv_in.weight", &[1024, 64, 3, 3]),
            ],
            &[
                "downsample_pixel_unshuffle_residual",
                "channel_group_mean",
                "upsample_pixel_shuffle_residual",
            ],
        ),
        case(
            "autoencoder-kl-standard",
            &[kl_contract],
            "AutoencoderKlV1",
            default(false, false, None, Some(4)),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [8, 8],
            vec![
                checkpoint("encoder.conv_in.weight", &[128, 3, 3, 3]),
                checkpoint("quant_conv.weight", &[8, 8, 1, 1]),
                checkpoint("post_quant_conv.weight", &[4, 4, 1, 1]),
                checkpoint("decoder.conv_out.weight", &[3, 128, 3, 3]),
            ],
            &[
                "bottom_right_downsample_pad",
                "diagonal_gaussian_mode",
                "nearest_upsample",
                "unit_interval_process_output",
            ],
        ),
        case(
            "autoencoder-kl-x4",
            &[kl_contract],
            "AutoencoderKlX4V1",
            default(true, false, None, Some(4)),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [4, 4],
            vec![
                checkpoint("encoder.down.1.downsample.conv.weight", &[256, 256, 3, 3]),
                checkpoint("decoder.up.2.upsample.conv.weight", &[512, 512, 3, 3]),
            ],
            &["x4_spatial_geometry"],
        ),
        case(
            "autoencoder-kl-batch-normalized",
            &[kl_contract],
            "AutoencoderKlBatchNormV1",
            default(false, true, None, Some(4)),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [16, 16],
            vec![
                checkpoint("bn.running_mean", &[16]),
                checkpoint("bn.num_batches_tracked", &[]),
            ],
            &[
                "latent_pixel_unshuffle",
                "batch_norm_latent",
                "latent_pixel_shuffle",
            ],
        ),
        case(
            "autoencoder-kl-asymmetric-decoder",
            &[kl_contract],
            "AutoencoderKlV1",
            default(false, false, Some(96), Some(6)),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [8, 8],
            vec![
                checkpoint("quant_conv.weight", &[12, 8, 1, 1]),
                checkpoint("post_quant_conv.weight", &[4, 6, 1, 1]),
                checkpoint("decoder.conv_in.weight", &[384, 4, 3, 3]),
            ],
            &[
                "separate_decoder_base_channels",
                "embed_dimension_projection",
            ],
        ),
        case(
            "autoencoding-engine-standard",
            &[engine_contract],
            "AutoencodingEngineV1",
            default(false, false, None, None),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [8, 8],
            vec![
                checkpoint("encoder.conv_out.weight", &[8, 512, 3, 3]),
                checkpoint("decoder.conv_in.weight", &[512, 4, 3, 3]),
            ],
            &["diagonal_gaussian_regularizer_mode", "no_quant_projection"],
        ),
        case(
            "autoencoding-engine-x4",
            &[engine_contract],
            "AutoencodingEngineX4V1",
            default(true, false, None, None),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [4, 4],
            vec![checkpoint(
                "decoder.up.2.upsample.conv.weight",
                &[512, 512, 3, 3],
            )],
            &["x4_spatial_geometry", "no_quant_projection"],
        ),
        case(
            "autoencoding-engine-batch-normalized",
            &[engine_contract],
            "AutoencodingEngineBatchNormV1",
            default(false, true, None, None),
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [16, 16],
            vec![checkpoint("bn.running_var", &[16])],
            &[
                "latent_pixel_unshuffle",
                "batch_norm_latent",
                "no_quant_projection",
            ],
        ),
        case(
            "pixel-space",
            &[pixel_contract],
            "PixelSpaceV1",
            automatic,
            "comfy.pixel_space_convert.PixelspaceConversionVAE.v1",
            "available",
            "available",
            [1, 1],
            vec![checkpoint("pixel_space_vae", &[])],
            &["unit_to_signed_encode", "signed_to_unit_decode"],
        ),
        case(
            "explicit-autoencoder-kl",
            &[explicit_contract],
            "ExplicitAutoencoderKlV1",
            explicit,
            "comfy.ldm.models.autoencoder.AutoencoderKL.v1",
            "available",
            "available",
            [2, 2],
            vec![
                checkpoint("encoder.conv_in.weight", &[32, 3, 3, 3]),
                checkpoint("decoder.conv_in.weight", &[128, 4, 3, 3]),
                checkpoint("bn.running_mean", &[16]),
            ],
            &[
                "average_pool_without_convolution",
                "nearest_upsample_without_convolution",
                "decoder_tanh",
                "attention_resolution",
            ],
        ),
    ];
    let artifact = json!({
        "schema_version": 1,
        "fixture_id": "comfy-native-image-vae-source-checkpoints-v1",
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
