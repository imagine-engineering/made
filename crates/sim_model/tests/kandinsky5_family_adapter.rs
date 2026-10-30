use sim_model::{
    KANDINSKY5_COMMON_MAPPING, KANDINSKY5_DIFFUSERS_MARKER, KANDINSKY5_FORWARD_PROGRAM,
    KANDINSKY5_IMAGE_CLIP_TARGET, KANDINSKY5_IMAGE_CONDITIONING, KANDINSKY5_IMAGE_LATENT_FORMAT,
    KANDINSKY5_IMAGE_ROPE_SCALE_FACTOR, KANDINSKY5_MEMORY_USAGE_FACTOR,
    KANDINSKY5_PREFIXED_STATE_PLAN, KANDINSKY5_STANDALONE_STATE_PLAN, KANDINSKY5_VIDEO_CLIP_TARGET,
    KANDINSKY5_VIDEO_CONDITIONING, KANDINSKY5_VIDEO_LATENT_FORMAT,
    KANDINSKY5_VIDEO_ROPE_SCALE_FACTOR, KANDINSKY5_WIDE_AXES_DIMENSIONS,
    Kandinsky5ConditioningFact, Kandinsky5Layout, Kandinsky5Variant, ModelFamilyError, ModelProbe,
    ModelStateTransaction, kandinsky5_common_mapping, kandinsky5_configuration_for_probe,
    kandinsky5_state_plan_for_layout,
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
fn val_model_detection_001_kandinsky5_source_profiles_layouts_and_image_precedence_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let cases = [
        (
            Kandinsky5Variant::VideoLite,
            Kandinsky5Layout::PrefixedNative,
            1_792,
            132,
            [16, 24, 24],
            28,
            10.0,
        ),
        (
            Kandinsky5Variant::VideoLite,
            Kandinsky5Layout::StandaloneNative,
            1_792,
            132,
            [16, 24, 24],
            28,
            10.0,
        ),
        (
            Kandinsky5Variant::VideoPro,
            Kandinsky5Layout::PrefixedNative,
            4_096,
            132,
            KANDINSKY5_WIDE_AXES_DIMENSIONS,
            32,
            10.0,
        ),
        (
            Kandinsky5Variant::ImageLite,
            Kandinsky5Layout::StandaloneNative,
            2_560,
            64,
            KANDINSKY5_WIDE_AXES_DIMENSIONS,
            20,
            3.0,
        ),
    ];
    for (variant, layout, model, visual, axes, heads, shift) in cases {
        let configuration = kandinsky5_configuration_for_probe(&probe(layout, model, visual))?;
        assert_eq!(configuration.variant, variant);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.model_dimension, model);
        assert_eq!(configuration.visual_embed_dimension, visual);
        assert_eq!(configuration.input_visual_channels, visual / 4);
        assert_eq!(configuration.feed_forward_dimension, model * 4);
        assert_eq!(configuration.time_dimension, 512);
        assert_eq!(configuration.text_block_count, 2);
        assert_eq!(configuration.visual_block_count, 32);
        assert_eq!(configuration.axes_dimensions, axes);
        assert_eq!(configuration.attention_head_count, heads);
        assert_eq!(configuration.sampling_shift, shift);
        assert_eq!(
            configuration.memory_usage_factor,
            KANDINSKY5_MEMORY_USAGE_FACTOR
        );
        if variant == Kandinsky5Variant::ImageLite {
            assert!(!configuration.concat_conditioning);
            assert_eq!(
                configuration.rope_scale_factor,
                KANDINSKY5_IMAGE_ROPE_SCALE_FACTOR
            );
            assert_eq!(
                configuration.latent_format.feature_id,
                KANDINSKY5_IMAGE_LATENT_FORMAT.feature_id
            );
            assert!(std::ptr::eq(
                configuration.clip_target,
                &KANDINSKY5_IMAGE_CLIP_TARGET
            ));
        } else {
            assert!(configuration.concat_conditioning);
            assert_eq!(
                configuration.rope_scale_factor,
                KANDINSKY5_VIDEO_ROPE_SCALE_FACTOR
            );
            assert_eq!(
                configuration.latent_format.feature_id,
                KANDINSKY5_VIDEO_LATENT_FORMAT.feature_id
            );
            assert!(std::ptr::eq(
                configuration.clip_target,
                &KANDINSKY5_VIDEO_CLIP_TARGET
            ));
        }
    }
    Ok(())
}

#[test]
fn val_model_detection_001_kandinsky5_rejects_diffusers_partial_mixed_gaps_and_bad_profiles() {
    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([(KANDINSKY5_DIFFUSERS_MARKER.to_owned(), vec![128])]),
        metadata: BTreeMap::new(),
    };
    assert_invalid(diffusers, "Diffusers layout is unsupported");

    let mut partial = probe(Kandinsky5Layout::StandaloneNative, 1_792, 132);
    partial
        .tensor_shapes
        .remove("visual_embeddings.in_layer.weight");
    assert!(matches!(
        kandinsky5_configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no supported layout")
    ));

    let mut mixed = probe(Kandinsky5Layout::PrefixedNative, 1_792, 132);
    mixed
        .tensor_shapes
        .extend(probe(Kandinsky5Layout::StandaloneNative, 1_792, 132).tensor_shapes);
    assert!(matches!(
        kandinsky5_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = probe(Kandinsky5Layout::StandaloneNative, 1_792, 132);
    gap.tensor_shapes
        .remove("visual_transformer_blocks.7.feed_forward.in_layer.weight");
    assert_invalid(
        gap,
        "visual transformer blocks are not exactly 32 consecutive",
    );

    assert_invalid(
        probe(Kandinsky5Layout::PrefixedNative, 2_560, 132),
        "unsupported model/visual dimensions",
    );
    assert_invalid(
        probe(Kandinsky5Layout::StandaloneNative, 4_096, 64),
        "expected visual dimension 132",
    );
    assert_invalid(
        probe(Kandinsky5Layout::StandaloneNative, 3_072, 132),
        "unsupported model/visual dimensions",
    );
}

#[test]
fn val_model_family_row_001_kandinsky5_native_state_plans_commit_atomically()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        Kandinsky5Layout::PrefixedNative,
        Kandinsky5Layout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &kandinsky5_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        for required in [
            "native.visual_embeddings.in_layer.weight",
            "native.time_embeddings.in_layer.weight",
            "native.text_embeddings.in_layer.weight",
            "native.visual_transformer_blocks.0.cross_attention.key_norm.weight",
            "native.out_layer.out_layer.weight",
        ] {
            assert!(model.contains_key(required), "{layout:?}: {required}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        kandinsky5_state_plan_for_layout(Kandinsky5Layout::PrefixedNative).encoded_plan,
        KANDINSKY5_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        kandinsky5_state_plan_for_layout(Kandinsky5Layout::StandaloneNative).encoded_plan,
        KANDINSKY5_STANDALONE_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_memory_001_val_cancel_001_kandinsky5_conditioning_memory_and_cancellation_are_bounded()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(KANDINSKY5_MEMORY_USAGE_FACTOR, 1.25);
    assert_eq!(
        KANDINSKY5_VIDEO_CONDITIONING.last(),
        Some(&Kandinsky5ConditioningFact::ZeroImageAndInverseMaskVideoConcat)
    );
    assert!(
        !KANDINSKY5_IMAGE_CONDITIONING
            .contains(&Kandinsky5ConditioningFact::ZeroImageAndInverseMaskVideoConcat)
    );
    assert!(
        KANDINSKY5_VIDEO_CONDITIONING
            .contains(&Kandinsky5ConditioningFact::OptionalProcessedLatentTimeDimensionReplacement)
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let source = mapping_source(&backend, &context, Kandinsky5Layout::PrefixedNative)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    let result = ModelStateTransaction::new(&backend, &context).execute(
        &KANDINSKY5_PREFIXED_STATE_PLAN.compile()?,
        DIGEST,
        &source,
    );
    assert!(matches!(result, Err(ModelFamilyError::Cancelled(_))));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_ownership_001_kandinsky5_has_one_adapter_and_imports_canonical_latents()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(std::ptr::eq(
        kandinsky5_common_mapping(),
        &KANDINSKY5_COMMON_MAPPING
    ));
    assert_eq!(kandinsky5_common_mapping().components.len(), 3);
    assert_eq!(kandinsky5_common_mapping().component_state_schemas.len(), 3);
    assert_eq!(
        kandinsky5_common_mapping().forward_program,
        KANDINSKY5_FORWARD_PROGRAM
    );
    assert_eq!(
        KANDINSKY5_VIDEO_LATENT_FORMAT.feature_id,
        "COMFY-MODEL-0037"
    );
    assert_eq!(
        KANDINSKY5_IMAGE_LATENT_FORMAT.feature_id,
        "COMFY-MODEL-0029"
    );

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let test_path = crate_root.join("tests/kandinsky5_family_adapter.rs");
    let adapter_path = crate_root.join("src/kandinsky5_family.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let flux_latent_path = crate_root.join("src/latent_formats/flux_sim_model_0029.rs");
    let video_latent_path = crate_root.join("src/latent_formats/hunyuanvideo_sim_model_0037.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(&files, "pub struct Kandinsky5Configuration", &[&test_path])?,
        vec![adapter_path.clone()]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0029", &test_path)?,
        vec![flux_latent_path]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0037", &test_path)?,
        vec![video_latent_path]
    );
    let transaction_declaration = ["pub struct Model", "StateTransaction"].concat();
    assert_eq!(
        files_containing(&files, &transaction_declaration, &[&test_path])?,
        vec![foundation_path]
    );
    let adapter = fs::read_to_string(adapter_path)?;
    let forbidden = [
        ["struct Model", "StateTransaction"].concat(),
        ["struct Patch", "Graph"].concat(),
        ["fn estimate_model", "_memory"].concat(),
        ["struct Cancellation", "Token"].concat(),
    ];
    for forbidden in forbidden {
        assert!(!adapter.contains(&forbidden));
    }
    Ok(())
}

fn probe(layout: Kandinsky5Layout, model: u64, visual: u64) -> ModelProbe {
    let prefix = match layout {
        Kandinsky5Layout::PrefixedNative => "model.diffusion_model.",
        Kandinsky5Layout::StandaloneNative => "",
    };
    let head = if model == 1_792 { 64 } else { 128 };
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}visual_embeddings.in_layer.bias"),
            vec![model],
        ),
        (
            format!("{prefix}visual_embeddings.in_layer.weight"),
            vec![model, visual],
        ),
        (format!("{prefix}time_embeddings.in_layer.bias"), vec![512]),
        (
            format!("{prefix}time_embeddings.in_layer.weight"),
            vec![512, model],
        ),
        (
            format!("{prefix}text_embeddings.in_layer.weight"),
            vec![model, 3_584],
        ),
        (
            format!("{prefix}out_layer.out_layer.weight"),
            vec![64, model],
        ),
    ]);
    for index in 0..2 {
        tensor_shapes.insert(
            format!("{prefix}text_transformer_blocks.{index}.self_attention.query_norm.weight"),
            vec![head],
        );
    }
    for index in 0..32 {
        tensor_shapes.insert(
            format!("{prefix}visual_transformer_blocks.{index}.feed_forward.in_layer.weight"),
            vec![model * 4, model],
        );
    }
    tensor_shapes.insert(
        format!("{prefix}visual_transformer_blocks.0.cross_attention.key_norm.weight"),
        vec![head],
    );
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        kandinsky5_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: Kandinsky5Layout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = match layout {
        Kandinsky5Layout::PrefixedNative => "model.diffusion_model.",
        Kandinsky5Layout::StandaloneNative => "",
    };
    let model_keys = [
        "visual_embeddings.in_layer.weight",
        "time_embeddings.in_layer.weight",
        "text_embeddings.in_layer.weight",
        "visual_transformer_blocks.0.cross_attention.key_norm.weight",
        "out_layer.out_layer.weight",
    ];
    model_keys
        .into_iter()
        .map(|key| format!("{prefix}{key}"))
        .chain([
            "vae.decoder.weight".to_owned(),
            "text_encoders.qwen.weight".to_owned(),
        ])
        .enumerate()
        .map(|(index, key)| Ok((key, tensor(backend, context, index as f32 + 1.0)?)))
        .collect()
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

fn latent_owner(
    files: &[PathBuf],
    feature_id: &str,
    excluded: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    Ok(files_containing(
        files,
        "pub const LATENT_FORMAT: LatentFormatDefinition",
        &[excluded],
    )?
    .into_iter()
    .filter(|path| fs::read_to_string(path).is_ok_and(|source| source.contains(feature_id)))
    .collect())
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
                if path
                    .file_name()
                    .is_some_and(|name| name == "target" || name == "tests")
                {
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

fn files_containing(
    files: &[PathBuf],
    needle: &str,
    excluded: &[&Path],
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut matches = Vec::new();
    for path in files {
        if excluded.iter().any(|excluded| *excluded == path) {
            continue;
        }
        if fs::read_to_string(path)?.contains(needle) {
            matches.push(path.clone());
        }
    }
    matches.sort();
    Ok(matches)
}
