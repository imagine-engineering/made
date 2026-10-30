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
    architecture: &str,
    structured_kind: &str,
    equations: &[&str],
) -> Value {
    json!({
        "id": id,
        "catalog_contract_id": contract,
        "profile": profile,
        "architecture": architecture,
        "structured_kind": structured_kind,
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
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/vae-structured");
    let provenance_bytes = fs::read(root.join("provenance.json"))?;
    let provenance: Provenance = serde_json::from_slice(&provenance_bytes)?;
    for source in provenance.sources {
        let bytes = fs::read(workspace.join(&source.path))?;
        if format!("{:x}", Sha256::digest(bytes)) != source.sha256 {
            bail!("pinned source digest changed for {}", source.path);
        }
    }

    let cases = vec![
        case(
            "hunyuan-shape-v1",
            "conditioning-vae-architecture-sd-comfy-ldm-hunyuan3d-vae-shapevae-e9c6b443",
            "HunyuanShapeV1",
            "comfy.ldm.hunyuan3d.vae.ShapeVAE.v1",
            "shape",
            &[
                "post_kl_channels_last_linear",
                "sixteen_residual_self_attention_blocks",
                "geo_final_layer_norm_eps_1e_5",
                "fourier_xyz_query_embedding",
                "chunked_cross_attention_occupancy_projection",
                "inclusive_resolution_plus_one_volume_grid",
            ],
        ),
        case(
            "tripo-splat-v1",
            "conditioning-vae-architecture-sd-comfy-ldm-triposplat-vae-octreegaussiandecoder-71f21635",
            "TripoSplatV1",
            "comfy.ldm.triposplat.vae.OctreeGaussianDecoder.v1",
            "gaussian_splats",
            &[
                "systematic_octree_probability_resampling",
                "caller_addressed_coordinate_jitter",
                "log2_absolute_position_embedding",
                "four_cross_only_octree_blocks",
                "sixteen_self_cross_gaussian_blocks",
                "octree_and_gaussian_final_layer_norm_eps_1e_5",
                "hammersley_atanh_offset_perturbation",
                "softplus_scale_and_sigmoid_opacity_activation",
                "y_up_position_and_quaternion_transform",
            ],
        ),
    ];
    let artifact = json!({
        "schema_version": 1,
        "fixture_id": "comfy-native-structured-vae-source-checkpoints-v1",
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
