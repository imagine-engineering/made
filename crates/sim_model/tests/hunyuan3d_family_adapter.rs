use sim_model::{
    HUNYUAN3D_COMMON_MAPPING, HUNYUAN3D_MEMORY_USAGE_FACTOR, HUNYUAN3D_MINI_DEPTH,
    HUNYUAN3D_MINI_LATENT_FORMAT, HUNYUAN3D_PREFIXED_STATE_PLAN, HUNYUAN3D_SAVED_MODEL_STATE_PLAN,
    HUNYUAN3D_STANDALONE_STATE_PLAN, HUNYUAN3D_V2_LATENT_FORMAT, HUNYUAN3D_V21_LATENT_FORMAT,
    Hunyuan3DLayout, Hunyuan3DVariant, ModelFamilyError, ModelProbe, ModelStateTransaction,
    hunyuan3d_common_mapping, hunyuan3d_configuration_for_probe, hunyuan3d_state_plan_for_layout,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    TensorDescriptor,
};
use sim_types::CancellationToken;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_hunyuan3d_variants_and_layouts_are_source_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let prefixed = hunyuan3d_configuration_for_probe(&classic_probe(
        Hunyuan3DLayout::PrefixedNative,
        16,
        true,
    ))?;
    assert_eq!(prefixed.variant, Hunyuan3DVariant::V2);
    assert_eq!(prefixed.layout, Hunyuan3DLayout::PrefixedNative);
    assert_eq!(prefixed.in_channels, 64);
    assert_eq!(prefixed.context_dimension, 1_536);
    assert_eq!(prefixed.hidden_size, 1_024);
    assert_eq!(prefixed.depth, 16);
    assert_eq!(prefixed.single_block_depth, 3);
    assert!(prefixed.guidance_embedding);
    assert!(prefixed.qkv_bias);
    assert_eq!(
        prefixed.latent_format.feature_id,
        HUNYUAN3D_V2_LATENT_FORMAT.feature_id
    );

    let saved = hunyuan3d_configuration_for_probe(&classic_probe(
        Hunyuan3DLayout::SavedModel,
        HUNYUAN3D_MINI_DEPTH,
        false,
    ))?;
    assert_eq!(saved.variant, Hunyuan3DVariant::V2Mini);
    assert_eq!(saved.layout, Hunyuan3DLayout::SavedModel);
    assert!(!saved.guidance_embedding);
    assert_eq!(
        saved.latent_format.feature_id,
        HUNYUAN3D_MINI_LATENT_FORMAT.feature_id
    );

    let standalone =
        hunyuan3d_configuration_for_probe(&v21_probe(Hunyuan3DLayout::StandaloneNative, 21))?;
    assert_eq!(standalone.variant, Hunyuan3DVariant::V2_1);
    assert_eq!(standalone.layout, Hunyuan3DLayout::StandaloneNative);
    assert_eq!(standalone.in_channels, 64);
    assert_eq!(standalone.context_dimension, 1_024);
    assert_eq!(standalone.hidden_size, 2_048);
    assert_eq!(standalone.depth, 21);
    assert_eq!(standalone.single_block_depth, 0);
    assert!(!standalone.guidance_embedding);
    assert!(!standalone.qkv_bias);
    assert_eq!(standalone.memory_usage_factor, 3.5);
    assert_eq!(
        standalone.latent_format.feature_id,
        HUNYUAN3D_V21_LATENT_FORMAT.feature_id
    );
    Ok(())
}

#[test]
fn val_model_detection_001_hunyuan3d_rejects_partial_mixed_genmo_and_malformed_probes() {
    let genmo = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("x_embedder.proj.weight".to_owned(), vec![3_072, 64]),
            ("t5_yproj.weight".to_owned(), vec![4_096, 4_096]),
            ("blocks.0.attn.proj_x.weight".to_owned(), vec![3_072, 3_072]),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        hunyuan3d_configuration_for_probe(&genmo),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no Hunyuan3D")
    ));

    let mut partial = v21_probe(Hunyuan3DLayout::StandaloneNative, 2);
    partial.tensor_shapes.remove("x_embedder.weight");
    assert_invalid(partial, "partial Hunyuan3D 2.1 marker set");

    let mut mixed_variant = classic_probe(Hunyuan3DLayout::StandaloneNative, 2, false);
    mixed_variant.tensor_shapes.extend([
        ("t_embedder.mlp.2.weight".to_owned(), vec![8_192, 2_048]),
        ("blocks.0.attn1.k_norm.weight".to_owned(), vec![128]),
        ("x_embedder.weight".to_owned(), vec![2_048, 64]),
    ]);
    assert_invalid(mixed_variant, "mixes classic and 2.1 markers");

    let mut mixed_layout = classic_probe(Hunyuan3DLayout::PrefixedNative, 2, false);
    mixed_layout
        .tensor_shapes
        .insert("latent_in.weight".to_owned(), vec![1_024, 64]);
    mixed_layout
        .tensor_shapes
        .insert("cond_in.weight".to_owned(), vec![1_024, 1_536]);
    assert!(matches!(
        hunyuan3d_configuration_for_probe(&mixed_layout),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut malformed = classic_probe(Hunyuan3DLayout::StandaloneNative, 2, false);
    malformed
        .tensor_shapes
        .insert("latent_in.weight".to_owned(), vec![1_023, 64]);
    assert_invalid(malformed, "hidden sizes differ");

    let mut non_consecutive = classic_probe(Hunyuan3DLayout::StandaloneNative, 2, false);
    non_consecutive
        .tensor_shapes
        .remove("double_blocks.1.img_attn.proj.weight");
    non_consecutive.tensor_shapes.insert(
        "double_blocks.2.img_attn.proj.weight".to_owned(),
        vec![1_024, 1_024],
    );
    assert_invalid(non_consecutive, "not consecutive");

    let mut misleading = classic_probe(Hunyuan3DLayout::PrefixedNative, 2, false);
    misleading
        .metadata
        .insert("model_layout".to_owned(), "standalone-native".to_owned());
    assert!(matches!(
        hunyuan3d_configuration_for_probe(&misleading),
        Ok(configuration) if configuration.layout == Hunyuan3DLayout::PrefixedNative
    ));
}

#[test]
fn val_model_family_row_001_hunyuan3d_plans_normalize_scale_and_layouts_transactionally()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        Hunyuan3DLayout::PrefixedNative,
        Hunyuan3DLayout::SavedModel,
        Hunyuan3DLayout::StandaloneNative,
    ] {
        let source = mapping_source(&backend, &context, layout)?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan3d_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &source,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        assert!(!model.is_empty());
        assert!(model.keys().all(|key| key.starts_with("native.")));
        assert!(model.keys().all(|key| !key.ends_with(".scale")));
        assert!(model.keys().any(|key| key.ends_with(".weight")));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("clip_vision").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        hunyuan3d_state_plan_for_layout(Hunyuan3DLayout::PrefixedNative).encoded_plan,
        HUNYUAN3D_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        hunyuan3d_state_plan_for_layout(Hunyuan3DLayout::SavedModel).encoded_plan,
        HUNYUAN3D_SAVED_MODEL_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        hunyuan3d_state_plan_for_layout(Hunyuan3DLayout::StandaloneNative).encoded_plan,
        HUNYUAN3D_STANDALONE_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_cancel_001_hunyuan3d_mapping_cancellation_is_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = mapping_source(&backend, &context, Hunyuan3DLayout::PrefixedNative)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    let result = ModelStateTransaction::new(&backend, &context).execute(
        &HUNYUAN3D_PREFIXED_STATE_PLAN.compile()?,
        DIGEST,
        &source,
    );
    assert!(matches!(result, Err(ModelFamilyError::Cancelled(_))));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_ownership_001_hunyuan3d_has_one_adapter_mapping_and_three_latent_owners()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(std::ptr::eq(
        hunyuan3d_common_mapping(),
        &HUNYUAN3D_COMMON_MAPPING
    ));
    assert_eq!(
        hunyuan3d_common_mapping().memory_usage_factor,
        HUNYUAN3D_MEMORY_USAGE_FACTOR
    );
    assert_eq!(hunyuan3d_common_mapping().components.len(), 3);
    assert_eq!(
        hunyuan3d_common_mapping().supported_dtypes,
        &[DType::Bf16, DType::F16, DType::F32]
    );

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let adapter_path = crate_root.join("src/hunyuan3d_family.rs");
    let test_path = crate_root.join("tests/hunyuan3d_family_adapter.rs");
    let rust_files = rust_files(repository_root)?;
    assert_eq!(
        count_in_files(
            &rust_files,
            "pub struct Hunyuan3DConfiguration",
            &[&test_path]
        )?,
        1
    );
    assert_eq!(
        files_containing(
            &rust_files,
            "pub static HUNYUAN3D_COMMON_MAPPING",
            &[&test_path]
        )?,
        vec![adapter_path]
    );
    for (identifier, expected) in [
        (
            "pub const LATENT_FORMAT_IDENTIFIER: &str = \"Hunyuan3Dv2\"",
            crate_root.join("src/latent_formats/hunyuanthree_dv2_sim_model_0032.rs"),
        ),
        (
            "pub const LATENT_FORMAT_IDENTIFIER: &str = \"Hunyuan3Dv2_1\"",
            crate_root.join("src/latent_formats/hunyuanthree_dv2_1_sim_model_0033.rs"),
        ),
        (
            "pub const LATENT_FORMAT_IDENTIFIER: &str = \"Hunyuan3Dv2mini\"",
            crate_root.join("src/latent_formats/hunyuanthree_dv2mini_sim_model_0034.rs"),
        ),
    ] {
        assert_eq!(
            files_containing(&rust_files, identifier, &[&test_path])?,
            vec![expected]
        );
    }
    Ok(())
}

fn classic_probe(layout: Hunyuan3DLayout, depth: usize, guidance: bool) -> ModelProbe {
    let prefix = layout_prefix(layout);
    let mut tensor_shapes = BTreeMap::from([
        (format!("{prefix}latent_in.weight"), vec![1_024, 64]),
        (format!("{prefix}cond_in.weight"), vec![1_024, 1_536]),
    ]);
    for ordinal in 0..depth {
        tensor_shapes.insert(
            format!("{prefix}double_blocks.{ordinal}.img_attn.proj.weight"),
            vec![1_024, 1_024],
        );
    }
    for ordinal in 0..3 {
        tensor_shapes.insert(
            format!("{prefix}single_blocks.{ordinal}.linear1.weight"),
            vec![1_024, 1_024],
        );
    }
    if guidance {
        tensor_shapes.insert(
            format!("{prefix}guidance_in.in_layer.weight"),
            vec![1_024, 256],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn v21_probe(layout: Hunyuan3DLayout, depth: usize) -> ModelProbe {
    let prefix = layout_prefix(layout);
    let mut tensor_shapes = BTreeMap::from([
        (format!("{prefix}x_embedder.weight"), vec![2_048, 64]),
        (
            format!("{prefix}t_embedder.mlp.2.weight"),
            vec![8_192, 2_048],
        ),
        (format!("{prefix}blocks.0.attn1.k_norm.weight"), vec![128]),
    ]);
    for ordinal in 0..depth {
        tensor_shapes.insert(
            format!("{prefix}blocks.{ordinal}.attn1.q_proj.weight"),
            vec![2_048, 2_048],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn layout_prefix(layout: Hunyuan3DLayout) -> &'static str {
    match layout {
        Hunyuan3DLayout::PrefixedNative => "model.diffusion_model.",
        Hunyuan3DLayout::SavedModel => "model.",
        Hunyuan3DLayout::StandaloneNative => "",
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        hunyuan3d_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: Hunyuan3DLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let keys: &[&str] = if layout == Hunyuan3DLayout::StandaloneNative {
        &[
            "x_embedder.scale",
            "t_embedder.mlp.2.weight",
            "blocks.0.attn1.k_norm.scale",
            "final_layer.linear.weight",
        ]
    } else {
        &[
            "latent_in.weight",
            "cond_in.scale",
            "double_blocks.0.img_attn.proj.scale",
            "single_blocks.0.linear1.weight",
            "final_layer.linear.weight",
        ]
    };
    let mut source = BTreeMap::new();
    for (index, key) in keys.iter().enumerate() {
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, context, index as f32 + 1.0)?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, context, 9.0)?,
    );
    source.insert(
        "conditioner.main_image_encoder.model.visual.weight".to_owned(),
        tensor(backend, context, 10.0)?,
    );
    Ok(source)
}

fn tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    value: f32,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(vec![1], DType::F32, backend.device(), context.stream)?;
    Ok(backend.upload_f32(descriptor, &[value], context)?.0)
}

fn rust_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("._"))
            {
                continue;
            }
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "target") {
                    continue;
                }
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn count_in_files(
    files: &[PathBuf],
    needle: &str,
    excluded: &[&PathBuf],
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut count = 0;
    for path in files {
        if excluded.contains(&path) {
            continue;
        }
        count += fs::read_to_string(path)?.matches(needle).count();
    }
    Ok(count)
}

fn files_containing(
    files: &[PathBuf],
    needle: &str,
    excluded: &[&PathBuf],
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut matching = Vec::new();
    for path in files {
        if excluded.contains(&path) {
            continue;
        }
        if fs::read_to_string(path)?.contains(needle) {
            matching.push(path.clone());
        }
    }
    Ok(matching)
}
