use serde::Deserialize;
use sha2::{Digest, Sha256};
use sim_model::{VaeKernelProfile, video_vae_source_plan, video_vae_source_state_schema};
use sim_tensor::DType;
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
    latent: String,
    temporal_ratio: u64,
    spatial_ratio: u64,
    state_checkpoints: Vec<Checkpoint>,
    equation_checkpoints: Vec<String>,
}

#[derive(Deserialize)]
struct Checkpoint {
    name: String,
    rank: u8,
}

fn profile(name: &str) -> Result<VaeKernelProfile, Box<dyn Error>> {
    Ok(match name {
        "HunyuanImageRefinerV1" => VaeKernelProfile::HunyuanImageRefinerV1,
        "MochiV1" => VaeKernelProfile::MochiV1,
        "LtxVideoV0" => VaeKernelProfile::LtxVideoV0 {
            configuration_sha256: None,
        },
        "LtxVideoV1" => VaeKernelProfile::LtxVideoV1 {
            configuration_sha256: None,
        },
        "LtxVideoV2" => VaeKernelProfile::LtxVideoV2 {
            configuration_sha256: None,
        },
        "HunyuanVideoRefinerV1" => VaeKernelProfile::HunyuanVideoRefinerV1,
        "CogVideoXV1" => VaeKernelProfile::CogVideoXV1,
        "Causal3dV1" => VaeKernelProfile::Causal3dV1,
        "CosmosV1" => VaeKernelProfile::CosmosV1,
        "Wan21V1" => VaeKernelProfile::Wan21V1,
        "Wan22V1" => VaeKernelProfile::Wan22V1,
        "TaeHvWan22V1" => VaeKernelProfile::TaeHvWan22V1,
        "TaeHvLtx2V1" => VaeKernelProfile::TaeHvLtx2V1,
        "LightTaeHv15V1" => VaeKernelProfile::LightTaeHv15V1,
        "TaeHvHunyuanV1" => VaeKernelProfile::TaeHvHunyuanV1,
        "LightTaeWan21V1" => VaeKernelProfile::LightTaeWan21V1,
        other => return Err(format!("unknown video VAE profile {other}").into()),
    })
}

#[test]
fn val_vae_001_video_source_ledger_covers_all_rows_and_configurations() -> Result<(), Box<dyn Error>>
{
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let root = workspace.join("crates/sim_test_support/fixtures/models/vae-video");
    let provenance = fs::read(root.join("provenance.json"))?;
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(root.join("architecture-checkpoints.json"))?)?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.fixture_id,
        "sim-native-video-vae-source-checkpoints-v1"
    );
    assert_eq!(fixture.oracle_kind, "immutable-source-derived-checkpoints");
    assert!(!fixture.production_dependency);
    assert_eq!(
        fixture.provenance_sha256,
        format!("{:x}", Sha256::digest(provenance))
    );
    assert_eq!(fixture.cases.len(), 17);
    let catalog = fs::read_to_string(
        workspace
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut ids = BTreeSet::new();
    let mut contracts = BTreeSet::new();
    let mut latent_variants = BTreeSet::new();
    for case in fixture.cases {
        assert!(ids.insert(case.id));
        let plan = video_vae_source_plan(&profile(&case.profile)?)?;
        assert_eq!(plan.temporal_ratio(), case.temporal_ratio);
        assert_eq!(plan.spatial_ratio(), case.spatial_ratio);
        assert_eq!(plan.equation_checkpoints(), case.equation_checkpoints);
        for checkpoint in case.state_checkpoints {
            let actual = plan
                .state_checkpoints()
                .iter()
                .find(|actual| actual.name == checkpoint.name)
                .ok_or_else(|| format!("missing {}", checkpoint.name))?;
            assert_eq!(actual.rank, checkpoint.rank);
        }
        for contract in case.catalog_contract_ids {
            assert!(catalog.contains(&contract));
            contracts.insert(contract);
        }
        latent_variants.insert(case.latent);
    }
    assert_eq!(contracts.len(), 12);
    assert!(latent_variants.len() >= 10);
    Ok(())
}

#[test]
fn val_ownership_001_video_adapter_has_no_parallel_foundational_owner() -> Result<(), Box<dyn Error>>
{
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(crate_root.join("src/vae_video.rs"))?;
    let production = source.split("#[cfg(test)]").next().unwrap_or(&source);
    for forbidden in [
        "CancellationToken::default",
        "struct NativeVideoCancellation",
        "struct VideoTilePlan",
        "struct VideoModelStore",
        "Command::new",
        "python",
    ] {
        assert!(
            !production.contains(forbidden),
            "duplicate or forbidden owner: {forbidden}"
        );
    }
    for required in ["VaeDescriptor", "LoadedModel", "VaeKernelProfile"] {
        assert!(
            production.contains(required),
            "missing canonical domain delegation: {required}"
        );
    }
    assert!(production.contains("validate_native_vae_backend_binding"));
    assert_eq!(
        production
            .matches("load_projected_vision_state_from_model_store_with_context(")
            .count(),
        1,
        "the video adapter must call the canonical projected state loader exactly once"
    );
    assert!(!production.contains("store.read"));
    assert!(!production.contains("store.load"));
    assert!(!production.contains("process_latent_in"));
    assert!(!production.contains("process_latent_out"));

    let vision_source = fs::read_to_string(crate_root.join("src/vision_models.rs"))?;
    let vision_production = vision_source
        .split("#[cfg(test)]")
        .next()
        .unwrap_or(&vision_source);
    assert_eq!(
        vision_production
            .matches("fn load_projected_vision_state_from_model_store_with_context(")
            .count(),
        1,
        "projected ModelStore admission must have one authoritative entry point"
    );
    assert_eq!(
        vision_production
            .matches("fn load_projected_vision_state_from_model_store_impl(")
            .count(),
        1,
        "projected ModelStore validation must have one authoritative implementation"
    );
    Ok(())
}

#[test]
fn val_vae_001_taehv_topologies_are_complete_and_profile_derived() -> Result<(), Box<dyn Error>> {
    for (profile, encoder_input, latent_channels, first_temporal_pool, first_temporal_grow) in [
        (VaeKernelProfile::TaeHvWan22V1, 12, 48, 2, 1),
        (VaeKernelProfile::TaeHvLtx2V1, 48, 128, 2, 2),
        (VaeKernelProfile::LightTaeHv15V1, 12, 32, 2, 1),
        (VaeKernelProfile::TaeHvHunyuanV1, 3, 16, 2, 1),
        (VaeKernelProfile::LightTaeWan21V1, 3, 16, 2, 1),
    ] {
        let schema = video_vae_source_state_schema(&profile, DType::F32)?;
        assert_eq!(schema.len(), 128);
        let shape = |name: &str| {
            schema
                .iter()
                .find(|state| state.name == name)
                .map(|state| state.shape.as_slice())
        };
        assert_eq!(
            shape("encoder.0.weight"),
            Some([64, encoder_input, 3, 3].as_slice())
        );
        assert_eq!(
            shape("encoder.2.conv.weight"),
            Some([64, 64 * first_temporal_pool, 1, 1].as_slice())
        );
        assert_eq!(
            shape("encoder.17.weight"),
            Some([latent_channels, 64, 3, 3].as_slice())
        );
        assert_eq!(
            shape("decoder.1.weight"),
            Some([256, latent_channels, 3, 3].as_slice())
        );
        assert_eq!(
            shape("decoder.7.conv.weight"),
            Some([256 * first_temporal_grow, 256, 1, 1].as_slice())
        );
        assert_eq!(
            shape("decoder.22.weight"),
            Some([encoder_input, 64, 3, 3].as_slice())
        );
        assert_eq!(
            schema
                .iter()
                .filter(|state| state.name.ends_with(".bias"))
                .count(),
            58
        );
    }
    Ok(())
}

#[test]
fn val_vae_001_hunyuan_refiner_schemas_preserve_source_module_boundaries()
-> Result<(), Box<dyn Error>> {
    fn shape<'a>(schema: &'a [sim_model::NativeVisionStateSpec], name: &str) -> Option<&'a [u64]> {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.as_slice())
    }

    let image =
        video_vae_source_state_schema(&VaeKernelProfile::HunyuanImageRefinerV1, DType::F32)?;
    let video =
        video_vae_source_state_schema(&VaeKernelProfile::HunyuanVideoRefinerV1, DType::F32)?;
    assert_eq!(image.len(), 280);
    assert_eq!(video.len(), 218);

    for schema in [&image, &video] {
        let names = schema
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names.len(),
            schema.len(),
            "state schema contains duplicate keys"
        );
        assert!(schema.iter().all(|state| state.dtype == DType::F32));
    }

    assert_eq!(
        shape(&image, "decoder.conv_in.weight"),
        Some([1024, 32, 3, 3, 3].as_slice())
    );
    assert_eq!(
        shape(&image, "encoder.down.0.block.0.norm1.weight"),
        Some([128].as_slice())
    );
    assert_eq!(
        shape(&image, "encoder.down.0.downsample.conv.weight"),
        Some([64, 128, 3, 3, 3].as_slice())
    );
    assert_eq!(shape(&image, "decoder.conv_in.conv.weight"), None);

    assert_eq!(
        shape(&video, "decoder.conv_in.conv.weight"),
        Some([1024, 32, 3, 3, 3].as_slice())
    );
    assert_eq!(
        shape(&video, "encoder.down.0.block.0.norm1.gamma"),
        Some([128, 1, 1, 1].as_slice())
    );
    assert_eq!(
        shape(&video, "encoder.down.0.downsample.conv.conv.weight"),
        Some([64, 128, 3, 3, 3].as_slice())
    );
    assert_eq!(shape(&video, "decoder.conv_in.weight"), None);
    Ok(())
}

#[test]
fn val_vae_001_generic_causal3d_schema_preserves_classic_kl_topology() -> Result<(), Box<dyn Error>>
{
    let schema = video_vae_source_state_schema(&VaeKernelProfile::Causal3dV1, DType::F32)?;
    assert_eq!(schema.len(), 248);
    let names = schema
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), schema.len());
    let shape = |name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(
        shape("encoder.conv_in.conv.weight"),
        Some(vec![128, 3, 3, 3, 3])
    );
    assert_eq!(
        shape("encoder.down.1.block.0.nin_shortcut.conv.weight"),
        Some(vec![256, 128, 1, 1, 1])
    );
    assert_eq!(
        shape("encoder.down.0.downsample.conv.conv.weight"),
        Some(vec![128, 128, 3, 3, 3])
    );
    assert_eq!(shape("quant_conv.weight"), Some(vec![8, 8, 1, 1, 1]));
    assert_eq!(shape("post_quant_conv.weight"), Some(vec![4, 4, 1, 1, 1]));
    assert_eq!(shape("encoder.down.0.block.0.norm1.gamma"), None);
    assert_eq!(
        shape("encoder.down.0.block.0.norm1.weight"),
        Some(vec![128])
    );
    Ok(())
}

#[test]
fn val_vae_001_cogvideox_schema_preserves_conditioned_decoder_topology()
-> Result<(), Box<dyn Error>> {
    let schema = video_vae_source_state_schema(&VaeKernelProfile::CogVideoXV1, DType::F32)?;
    assert_eq!(schema.len(), 436);
    let names = schema
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), schema.len());
    let shape = |name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(
        shape("encoder.conv_in.conv.weight"),
        Some(vec![128, 3, 3, 3, 3])
    );
    assert_eq!(
        shape("encoder.down_blocks.0.downsamplers.0.conv.weight"),
        Some(vec![128, 128, 3, 3])
    );
    assert_eq!(
        shape("decoder.mid_block.resnets.0.norm1.norm_layer.weight"),
        Some(vec![512])
    );
    assert_eq!(
        shape("decoder.mid_block.resnets.0.norm1.conv_y.conv.weight"),
        Some(vec![512, 16, 1, 1, 1])
    );
    assert_eq!(
        shape("decoder.up_blocks.1.resnets.0.conv_shortcut.weight"),
        Some(vec![256, 512, 1, 1, 1])
    );
    assert_eq!(
        shape("decoder.conv_out.conv.weight"),
        Some(vec![3, 128, 3, 3, 3])
    );
    Ok(())
}

#[test]
fn val_vae_001_wan21_schema_preserves_source_sequential_indices() -> Result<(), Box<dyn Error>> {
    let schema = video_vae_source_state_schema(&VaeKernelProfile::Wan21V1, DType::F32)?;
    assert_eq!(schema.len(), 194);
    let names = schema
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), schema.len());
    let shape = |name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(shape("encoder.conv1.weight"), Some(vec![96, 3, 3, 3, 3]));
    assert_eq!(
        shape("encoder.downsamples.3.residual.2.weight"),
        Some(vec![192, 96, 3, 3, 3])
    );
    assert_eq!(
        shape("encoder.downsamples.5.time_conv.weight"),
        Some(vec![192, 192, 3, 1, 1])
    );
    assert_eq!(
        shape("decoder.upsamples.3.time_conv.weight"),
        Some(vec![768, 384, 3, 1, 1])
    );
    assert_eq!(
        shape("decoder.upsamples.4.shortcut.weight"),
        Some(vec![384, 192, 1, 1, 1])
    );
    assert_eq!(shape("decoder.head.0.gamma"), Some(vec![96, 1, 1, 1]));
    Ok(())
}

#[test]
fn val_vae_001_wan22_schema_preserves_nested_source_blocks() -> Result<(), Box<dyn Error>> {
    let schema = video_vae_source_state_schema(&VaeKernelProfile::Wan22V1, DType::F32)?;
    assert_eq!(schema.len(), 196);
    let names = schema
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), schema.len());
    let shape = |name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(shape("encoder.conv1.weight"), Some(vec![160, 12, 3, 3, 3]));
    assert_eq!(shape("encoder.head.2.weight"), Some(vec![96, 640, 3, 3, 3]));
    assert_eq!(shape("conv1.weight"), Some(vec![96, 96, 1, 1, 1]));
    assert_eq!(shape("conv2.weight"), Some(vec![48, 48, 1, 1, 1]));
    assert_eq!(shape("decoder.conv1.weight"), Some(vec![1024, 48, 3, 3, 3]));
    assert_eq!(
        shape("encoder.downsamples.1.downsamples.0.residual.2.weight"),
        Some(vec![320, 160, 3, 3, 3])
    );
    assert_eq!(
        shape("encoder.downsamples.1.downsamples.2.time_conv.weight"),
        Some(vec![320, 320, 3, 1, 1])
    );
    assert_eq!(
        shape("encoder.downsamples.2.downsamples.2.time_conv.weight"),
        Some(vec![640, 640, 3, 1, 1])
    );
    assert_eq!(
        shape("encoder.downsamples.0.downsamples.2.time_conv.weight"),
        None
    );
    assert_eq!(
        shape("decoder.upsamples.0.upsamples.0.residual.2.weight"),
        Some(vec![1024, 1024, 3, 3, 3])
    );
    assert_eq!(
        shape("decoder.upsamples.0.upsamples.3.time_conv.weight"),
        Some(vec![2048, 1024, 3, 1, 1])
    );
    assert_eq!(
        shape("decoder.upsamples.1.upsamples.3.time_conv.weight"),
        Some(vec![2048, 1024, 3, 1, 1])
    );
    assert_eq!(
        shape("decoder.upsamples.2.upsamples.3.time_conv.weight"),
        None
    );
    assert_eq!(shape("decoder.head.2.weight"), Some(vec![12, 256, 3, 3, 3]));
    Ok(())
}

#[test]
fn val_vae_001_cosmos_schema_preserves_factorized_source_blocks() -> Result<(), Box<dyn Error>> {
    let schema = video_vae_source_state_schema(&VaeKernelProfile::CosmosV1, DType::Bf16)?;
    let names = schema
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), schema.len());
    let shape = |name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(
        shape("encoder.conv_in.0.conv3d.weight"),
        Some(vec![128, 192, 1, 3, 3])
    );
    assert_eq!(
        shape("encoder.down.0.downsample.conv2.conv3d.weight"),
        Some(vec![256, 256, 3, 1, 1])
    );
    assert_eq!(
        shape("encoder.mid.attn_1.1.q.conv3d.weight"),
        Some(vec![512, 512, 1, 1, 1])
    );
    assert_eq!(
        shape("encoder.mid.attn_1.1.norm.norm.weight"),
        Some(vec![512])
    );
    assert_eq!(
        shape("decoder.up.1.upsample.conv1.conv3d.weight"),
        Some(vec![512, 512, 1, 3, 3])
    );
    assert_eq!(
        shape("decoder.up.0.block.0.nin_shortcut.conv3d.weight"),
        Some(vec![256, 512, 1, 1, 1])
    );
    assert_eq!(shape("latent_mean"), Some(vec![256]));
    assert_eq!(shape("latent_std"), Some(vec![256]));
    assert_eq!(shape("decoder.unpatcher3d.wavelets"), None);
    Ok(())
}

#[test]
fn val_vae_001_mochi_schema_preserves_attention_and_depth_to_space_time()
-> Result<(), Box<dyn Error>> {
    let schema = video_vae_source_state_schema(&VaeKernelProfile::MochiV1, DType::F16)?;
    let names = schema
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), schema.len());
    let shape = |name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(shape("encoder.layers.0.weight"), Some(vec![64, 15]));
    assert_eq!(
        shape("encoder.layers.4.layers.0.weight"),
        Some(vec![128, 64, 1, 2, 2])
    );
    assert_eq!(
        shape("encoder.layers.4.layers.1.attn_block.attn.qkv.weight"),
        Some(vec![384, 128])
    );
    assert_eq!(
        shape("encoder.layers.6.layers.6.stack.5.weight"),
        Some(vec![384, 384, 3, 3, 3])
    );
    assert_eq!(shape("encoder.output_proj.weight"), Some(vec![24, 384]));
    assert_eq!(shape("decoder.blocks.1.proj.weight"), Some(vec![6144, 768]));
    assert_eq!(
        shape("decoder.blocks.2.blocks.3.stack.5.weight"),
        Some(vec![512, 512, 3, 3, 3])
    );
    assert_eq!(shape("decoder.output_proj.weight"), Some(vec![3, 128]));
    Ok(())
}

#[test]
fn val_vae_001_ltx_schemas_preserve_all_three_source_topologies() -> Result<(), Box<dyn Error>> {
    let profiles = [
        VaeKernelProfile::LtxVideoV0 {
            configuration_sha256: None,
        },
        VaeKernelProfile::LtxVideoV1 {
            configuration_sha256: None,
        },
        VaeKernelProfile::LtxVideoV2 {
            configuration_sha256: None,
        },
    ];
    let mut schemas = Vec::with_capacity(profiles.len());
    for profile in &profiles {
        schemas.push(video_vae_source_state_schema(profile, DType::Bf16)?);
    }
    assert_eq!(
        schemas.iter().map(Vec::len).collect::<Vec<_>>(),
        [190, 294, 226]
    );
    for (schema, encoder_output_channels) in schemas.iter().zip([512, 512, 2048]) {
        let names = schema
            .iter()
            .map(|state| state.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), schema.len());
        assert_eq!(
            schema
                .iter()
                .find(|state| state.name == "encoder.conv_in.conv.weight")
                .map(|state| state.shape.as_slice()),
            Some([128, 48, 3, 3, 3].as_slice())
        );
        assert_eq!(
            schema
                .iter()
                .find(|state| state.name == "encoder.conv_out.conv.weight")
                .map(|state| state.shape.as_slice()),
            Some([129, encoder_output_channels, 3, 3, 3].as_slice())
        );
        for name in [
            "per_channel_statistics.std-of-means",
            "per_channel_statistics.mean-of-means",
        ] {
            let state = schema.iter().find(|state| state.name == name).ok_or(name)?;
            assert_eq!(state.shape, [128]);
            assert_eq!(state.dtype, DType::F32);
        }
    }
    let shape = |schema: &[sim_model::NativeVisionStateSpec], name: &str| {
        schema
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.shape.clone())
    };
    assert_eq!(
        shape(
            &schemas[0],
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"
        ),
        Some(vec![512, 512, 3, 3, 3])
    );
    assert_eq!(
        shape(
            &schemas[1],
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"
        ),
        Some(vec![1024, 1024, 3, 3, 3])
    );
    assert_eq!(
        shape(
            &schemas[2],
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"
        ),
        Some(vec![1024, 1024, 3, 3, 3])
    );
    assert_eq!(
        shape(&schemas[2], "encoder.down_blocks.1.conv.conv.weight"),
        Some(vec![64, 128, 3, 3, 3])
    );
    assert_eq!(
        shape(&schemas[2], "encoder.down_blocks.3.conv.conv.weight"),
        Some(vec![256, 256, 3, 3, 3])
    );
    assert_eq!(
        shape(&schemas[2], "encoder.down_blocks.5.conv.conv.weight"),
        Some(vec![128, 512, 3, 3, 3])
    );
    assert_eq!(
        shape(
            &schemas[1],
            "decoder.up_blocks.0.time_embedder.timestep_embedder.linear_1.weight"
        ),
        Some(vec![4096, 256])
    );
    assert_eq!(
        shape(&schemas[1], "decoder.up_blocks.1.conv.conv.weight"),
        Some(vec![4096, 1024, 3, 3, 3])
    );
    assert_eq!(
        shape(
            &schemas[1],
            "decoder.up_blocks.2.res_blocks.0.per_channel_scale1"
        ),
        Some(vec![512, 1, 1])
    );
    assert_eq!(
        shape(&schemas[0], "decoder.timestep_scale_multiplier"),
        None
    );
    assert_eq!(
        shape(&schemas[1], "decoder.timestep_scale_multiplier"),
        Some(vec![])
    );
    Ok(())
}
