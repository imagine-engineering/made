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
    contracts: &[&str],
    profile: &str,
    latent: &str,
    ratios: [u64; 2],
    state: &[(&str, u8)],
    equations: &[&str],
) -> Value {
    json!({
        "id": id,
        "catalog_contract_ids": contracts,
        "profile": profile,
        "latent": latent,
        "temporal_ratio": ratios[0],
        "spatial_ratio": ratios[1],
        "state_checkpoints": state.iter().map(|(name, rank)| json!({"name":name,"rank":rank})).collect::<Vec<_>>(),
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
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/vae-video");
    let provenance_bytes = fs::read(root.join("provenance.json"))?;
    let provenance: Provenance = serde_json::from_slice(&provenance_bytes)?;
    for source in provenance.sources {
        let bytes = fs::read(workspace.join(&source.path))?;
        if format!("{:x}", Sha256::digest(bytes)) != source.sha256 {
            bail!("pinned source digest changed for {}", source.path);
        }
    }

    const IMAGE_REFINER: &str = "conditioning-vae-architecture-sd-autoencodingengine-27dec63f";
    const MOCHI: &str =
        "conditioning-vae-architecture-sd-comfy-ldm-genmo-vae-model-videovae-b1621cfc";
    const LTX: &str = "conditioning-vae-architecture-sd-comfy-ldm-lightricks-vae-causal-video-autoencoder-videovae-238bf43d";
    const VIDEO_REFINER: &str = "conditioning-vae-architecture-sd-autoencodingengine-d7c8a707";
    const COG: &str =
        "conditioning-vae-architecture-sd-comfy-ldm-cogvideo-vae-autoencoderklcogvideox-18fb0b1a";
    const CAUSAL: &str = "conditioning-vae-architecture-sd-autoencoderkl-153e7ff2";
    const COSMOS: &str = "conditioning-vae-architecture-sd-comfy-ldm-cosmos-vae-causalcontinuousvideotokenizer-ba589e82";
    const WAN22: &str = "conditioning-vae-architecture-sd-comfy-ldm-wan-vae2-2-wanvae-69fa166b";
    const WAN21: &str = "conditioning-vae-architecture-sd-comfy-ldm-wan-vae-wanvae-ce05ff38";
    const TAEHV: &str = "conditioning-vae-architecture-sd-comfy-taesd-taehv-taehv-4688008a";
    const LIGHT_HV: &str = "conditioning-vae-architecture-sd-comfy-taesd-taehv-taehv-baed411e";
    const LIGHT_WAN: &str = "conditioning-vae-architecture-sd-comfy-taesd-taehv-taehv-ff3688fd";

    let mochi_equations = [
        "causal_temporal_encode_ceil_t_div_6",
        "causal_temporal_decode_t_mul_6_minus_5",
        "spatial_stride_8",
        "per_channel_latent_affine",
    ];
    let ltx_equations = [
        "causal_temporal_encode_ceil_t_div_8",
        "causal_temporal_decode_t_mul_8_minus_7",
        "spatial_stride_32",
        "caller_addressed_deterministic_decode_rng",
    ];
    let refiner_equations = [
        "causal_temporal_encode_ceil_t_div_4",
        "causal_temporal_decode_t_mul_4_minus_3",
        "spatial_stride_16",
        "first_frame_cache_boundary",
    ];
    let image_refiner_equations = [
        "single_frame_encode_expand_temporal_4",
        "single_frame_decode_select_last",
        "spatial_stride_16",
        "diagonal_gaussian_mode",
    ];
    let cog_equations = [
        "causal_conv3d_first_frame_replication",
        "rolling_temporal_encode_cache",
        "rolling_temporal_decode_cache",
        "diagonal_gaussian_mode",
    ];
    let causal_equations = [
        "causal_conv3d_prefix_cache",
        "temporal_compress_4",
        "diagonal_gaussian_mode",
    ];
    let cosmos_equations = [
        "causal_temporal_encode_ceil_t_div_8",
        "causal_temporal_decode_t_mul_8_minus_7",
        "haar_wavelet_patchify_3d",
        "haar_wavelet_unpatchify_3d",
    ];
    let wan_equations = [
        "causal_conv3d_two_frame_cache",
        "first_frame_separate_encode",
        "first_frame_separate_decode",
        "causal_temporal_index_4",
    ];
    let taehv_equations = [
        "frame_queue_dependency_order",
        "first_frame_zero_temporal_memory",
        "temporal_blend_memory",
        "bounded_frame_work_queue",
    ];
    let cases = vec![
        case(
            "hunyuan-image-refiner",
            &[IMAGE_REFINER],
            "HunyuanImageRefinerV1",
            "HunyuanImage21Refiner",
            [4, 16],
            &[("decoder.conv_in.weight", 5)],
            &image_refiner_equations,
        ),
        case(
            "mochi",
            &[MOCHI],
            "MochiV1",
            "Mochi",
            [6, 8],
            &[
                ("decoder.blocks.2.blocks.3.stack.5.weight", 5),
                ("encoder.layers.4.layers.1.attn_block.attn.qkv.weight", 2),
            ],
            &mochi_equations,
        ),
        case(
            "ltx-video-v0",
            &[LTX],
            "LtxVideoV0",
            "LTXV",
            [8, 32],
            &[("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight", 5)],
            &ltx_equations,
        ),
        case(
            "ltx-video-v1",
            &[LTX],
            "LtxVideoV1",
            "LTXV",
            [8, 32],
            &[("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight", 5)],
            &ltx_equations,
        ),
        case(
            "ltx-video-v2",
            &[LTX],
            "LtxVideoV2",
            "LTXV",
            [8, 32],
            &[("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight", 5)],
            &ltx_equations,
        ),
        case(
            "hunyuan-video-refiner",
            &[VIDEO_REFINER],
            "HunyuanVideoRefinerV1",
            "HunyuanVideo15",
            [4, 16],
            &[("decoder.conv_in.conv.weight", 5)],
            &refiner_equations,
        ),
        case(
            "cogvideox",
            &[COG],
            "CogVideoXV1",
            "CogVideoX",
            [4, 8],
            &[
                ("decoder.conv_in.conv.weight", 5),
                ("decoder.mid_block.resnets.0.norm1.norm_layer.weight", 1),
                ("encoder.conv_out.conv.weight", 5),
            ],
            &cog_equations,
        ),
        case(
            "cogvideox-1-5",
            &[COG],
            "CogVideoXV1",
            "CogVideoX1_5",
            [4, 8],
            &[
                ("decoder.conv_in.conv.weight", 5),
                ("decoder.mid_block.resnets.0.norm1.norm_layer.weight", 1),
                ("encoder.conv_out.conv.weight", 5),
            ],
            &cog_equations,
        ),
        case(
            "causal-3d",
            &[CAUSAL],
            "Causal3dV1",
            "unbound",
            [4, 8],
            &[
                ("decoder.conv_in.conv.weight", 5),
                ("post_quant_conv.weight", 5),
            ],
            &causal_equations,
        ),
        case(
            "cosmos",
            &[COSMOS],
            "CosmosV1",
            "Cosmos1CV8x8x8",
            [8, 8],
            &[("decoder.unpatcher3d.wavelets", 5)],
            &cosmos_equations,
        ),
        case(
            "wan-2-1",
            &[WAN21],
            "Wan21V1",
            "Wan21",
            [4, 8],
            &[
                ("decoder.middle.0.residual.0.gamma", 1),
                ("encoder.conv1.weight", 5),
                ("decoder.head.2.weight", 5),
            ],
            &wan_equations,
        ),
        case(
            "wan-2-2",
            &[WAN22],
            "Wan22V1",
            "Wan22",
            [4, 16],
            &[
                ("decoder.middle.0.residual.0.gamma", 1),
                ("decoder.upsamples.0.upsamples.0.residual.2.weight", 5),
            ],
            &wan_equations,
        ),
        case(
            "taehv-wan-2-2",
            &[TAEHV],
            "TaeHvWan22V1",
            "Wan22",
            [4, 16],
            &[("decoder.1.weight", 4), ("decoder.22.bias", 1)],
            &taehv_equations,
        ),
        case(
            "taehv-ltx2",
            &[TAEHV],
            "TaeHvLtx2V1",
            "LTXV",
            [8, 32],
            &[("decoder.1.weight", 4), ("decoder.22.bias", 1)],
            &taehv_equations,
        ),
        case(
            "lighttae-hunyuan-1-5",
            &[LIGHT_HV],
            "LightTaeHv15V1",
            "HunyuanVideo15",
            [4, 16],
            &[("decoder.1.weight", 4), ("decoder.22.bias", 1)],
            &taehv_equations,
        ),
        case(
            "taehv-hunyuan",
            &[LIGHT_WAN],
            "TaeHvHunyuanV1",
            "HunyuanVideo",
            [4, 8],
            &[("decoder.1.weight", 4), ("decoder.22.bias", 1)],
            &taehv_equations,
        ),
        case(
            "lighttae-wan-2-1",
            &[LIGHT_WAN],
            "LightTaeWan21V1",
            "Wan21",
            [4, 8],
            &[("decoder.1.weight", 4), ("decoder.22.bias", 1)],
            &taehv_equations,
        ),
    ];
    let artifact = json!({
        "schema_version": 1,
        "fixture_id": "comfy-native-video-vae-source-checkpoints-v1",
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
