use sim_model::{
    LTX_CLIP_TARGET, LTX_COMMON_MAPPING, LTX_FORWARD_PROGRAM, LTX_MAX_TRANSFORMER_CONFIG_BYTES,
    LTX_PIXART_COLLISION_MARKER, LTX_PREFIXED_STATE_PLAN, LTX_SAVED_MODEL_STATE_PLAN,
    LTX_STANDALONE_STATE_PLAN, LTXAV_CONDITIONING, LTXAV_LATENT_FORMAT, LTXAV_MEMORY_USAGE_FACTOR,
    LTXV_BASE_MEMORY_USAGE_FACTOR, LTXV_CONDITIONING, LTXV_LATENT_FORMAT, LTXV_SAMPLING_SHIFT,
    LtxConditioningFact, LtxLayout, LtxVariant, ModelFamilyError, ModelProbe,
    ModelStateTransaction, ltx_common_mapping, ltx_configuration_for_probe,
    ltx_state_plan_for_layout, ltxv_memory_usage_factor,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, Scalar, StreamId, Tensor,
    TensorBackend, TensorDescriptor,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, greater_with_context_exact_native,
    },
};
use sim_types::CancellationToken;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_ltxav_precedes_ltxv_and_all_native_layouts_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        LtxLayout::PrefixedNative,
        LtxLayout::SavedModel,
        LtxLayout::StandaloneNative,
    ] {
        let video = ltx_configuration_for_probe(&probe(layout, LtxVariant::Video, 2_048, 64, 3))?;
        assert_eq!(video.variant, LtxVariant::Video);
        assert_eq!(video.layout, layout);
        assert_eq!(video.input_channels, 128);
        assert_eq!(video.inner_dimension, 2_048);
        assert_eq!(video.number_of_layers, 3);
        assert_eq!(video.attention_head_dimension, 64);
        assert_eq!(video.number_of_attention_heads, 32);
        assert_eq!(video.cross_attention_dimension, 2_048);
        assert_eq!(video.audio_input_channels, None);
        assert_eq!(video.sampling_shift, LTXV_SAMPLING_SHIFT);
        assert_eq!(video.memory_usage_factor, LTXV_BASE_MEMORY_USAGE_FACTOR);
        assert_eq!(video.conditioning, LTXV_CONDITIONING);
        assert_eq!(video.latent_format.feature_id, "COMFY-MODEL-0040");
        assert!(std::ptr::eq(video.clip_target, &LTX_CLIP_TARGET));

        let audio_video =
            ltx_configuration_for_probe(&probe(layout, LtxVariant::AudioVideo, 4_096, 128, 4))?;
        assert_eq!(audio_video.variant, LtxVariant::AudioVideo);
        assert_eq!(audio_video.layout, layout);
        assert_eq!(audio_video.input_channels, 128);
        assert_eq!(audio_video.inner_dimension, 4_096);
        assert_eq!(audio_video.number_of_layers, 4);
        assert_eq!(audio_video.attention_head_dimension, 128);
        assert_eq!(audio_video.cross_attention_dimension, 4_096);
        assert_eq!(audio_video.audio_input_channels, Some(128));
        assert_eq!(audio_video.audio_inner_dimension, Some(2_048));
        assert_eq!(audio_video.memory_usage_factor, LTXAV_MEMORY_USAGE_FACTOR);
        assert_eq!(audio_video.conditioning, LTXAV_CONDITIONING);
        assert_eq!(audio_video.latent_format.feature_id, "COMFY-MODEL-0039");
    }
    Ok(())
}

#[test]
fn val_model_detection_001_ltx_rejects_pixart_diffusers_partial_mixed_gaps_and_bad_overrides() {
    let mut pixart = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    pixart
        .tensor_shapes
        .insert(LTX_PIXART_COLLISION_MARKER.to_owned(), vec![2_048]);
    assert_invalid(pixart, "PixArt collision marker");

    let diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([(
            "transformer_blocks.0.attn1.to_q.weight".to_owned(),
            vec![2_048, 2_048],
        )]),
        metadata: BTreeMap::new(),
    };
    assert_invalid(diffusers, "Diffusers layout is unsupported");

    let mut partial = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    partial.tensor_shapes.remove("patchify_proj.weight");
    assert!(matches!(
        ltx_configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no supported")
    ));

    let mut mixed = probe(LtxLayout::PrefixedNative, LtxVariant::Video, 2_048, 64, 3);
    mixed
        .tensor_shapes
        .extend(probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3).tensor_shapes);
    assert!(matches!(
        ltx_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    gap.tensor_shapes
        .remove("transformer_blocks.1.attn2.to_k.weight");
    assert_invalid(gap, "not a consecutive bounded sequence");

    let mut bad_heads = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    bad_heads.tensor_shapes.insert(
        "transformer_blocks.0.attn2.to_k.weight".to_owned(),
        vec![2_047, 2_048],
    );
    assert_invalid(bad_heads, "not divisible by 32 heads");

    let mut contradictory = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    contradictory.metadata.insert(
        "config".to_owned(),
        r#"{"transformer":{"num_layers":4}}"#.to_owned(),
    );
    assert_invalid(contradictory, "contradicts detected value 3");

    let mut malformed = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    malformed
        .metadata
        .insert("config".to_owned(), "{".to_owned());
    assert_invalid(malformed, "invalid JSON");

    let mut oversized = probe(LtxLayout::StandaloneNative, LtxVariant::Video, 2_048, 64, 3);
    oversized.metadata.insert(
        "config".to_owned(),
        "x".repeat(LTX_MAX_TRANSFORMER_CONFIG_BYTES + 1),
    );
    assert_invalid(oversized, "maximum is");
}

#[test]
fn val_model_family_row_001_ltx_state_plans_are_source_native_and_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        LtxLayout::PrefixedNative,
        LtxLayout::SavedModel,
        LtxLayout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &ltx_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        for key in [
            "native.adaln_single.emb.timestep_embedder.linear_1.bias",
            "native.patchify_proj.weight",
            "native.transformer_blocks.0.attn2.to_k.weight",
            "native.proj_out.weight",
            "native.audio_adaln_single.linear.weight",
            "native.audio_patchify_proj.weight",
        ] {
            assert!(model.contains_key(key), "{layout:?}: {key}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        ltx_state_plan_for_layout(LtxLayout::PrefixedNative).encoded_plan,
        LTX_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        ltx_state_plan_for_layout(LtxLayout::SavedModel).encoded_plan,
        LTX_SAVED_MODEL_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        ltx_state_plan_for_layout(LtxLayout::StandaloneNative).encoded_plan,
        LTX_STANDALONE_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_memory_001_val_cancel_001_ltx_memory_conditioning_and_failure_paths_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(ltxv_memory_usage_factor(2_048)?, 5.5);
    assert_eq!(ltxv_memory_usage_factor(4_096)?, 11.0);
    assert!(ltxv_memory_usage_factor(0).is_err());
    assert_eq!(LTXAV_MEMORY_USAGE_FACTOR, 0.077);
    assert!(LTXV_CONDITIONING.contains(&LtxConditioningFact::FrameRateDefault25));
    assert!(!LTXV_CONDITIONING.contains(&LtxConditioningFact::OptionalAudioDenoiseMask));
    assert!(LTXAV_CONDITIONING.contains(&LtxConditioningFact::OptionalAudioDenoiseMask));
    assert!(LTXAV_CONDITIONING.contains(&LtxConditioningFact::OptionalReferenceAudio));
    assert!(LTXAV_CONDITIONING.contains(&LtxConditioningFact::AudioMaskedTimestepPatchification));

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let source = mapping_source(&backend, &context, LtxLayout::PrefixedNative)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    let result = ModelStateTransaction::new(&backend, &context).execute(
        &LTX_PREFIXED_STATE_PLAN.compile()?,
        DIGEST,
        &source,
    );
    assert!(matches!(result, Err(ModelFamilyError::Cancelled(_))));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let tiny_cancellation = CancellationToken::default();
    let tiny_authorization = authority.authorize_workspace(3)?;
    let tiny_context = backend.execution_context(
        StreamId::DEFAULT,
        tiny_authorization.clone(),
        &tiny_cancellation,
    );
    let oom_input = backend
        .upload_f32(
            TensorDescriptor::contiguous(vec![4], DType::F32, backend.device(), StreamId::DEFAULT)?,
            &[1.0, 2.0, 3.0, 4.0],
            &tiny_context,
        )?
        .0;
    let oom_baseline = backend.memory_snapshot().current_bytes;
    let oom = greater_with_context_exact_native(
        &backend,
        &oom_input,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        &tiny_context,
    );
    assert!(oom.is_err());
    assert_eq!(tiny_authorization.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, oom_baseline);
    Ok(())
}

#[test]
fn val_ownership_001_ltx_has_one_adapter_and_imports_canonical_latents()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(std::ptr::eq(ltx_common_mapping(), &LTX_COMMON_MAPPING));
    assert_eq!(ltx_common_mapping().components.len(), 3);
    assert_eq!(ltx_common_mapping().component_state_schemas.len(), 3);
    assert_eq!(ltx_common_mapping().forward_program, LTX_FORWARD_PROGRAM);
    assert_eq!(LTXV_LATENT_FORMAT.feature_id, "COMFY-MODEL-0040");
    assert_eq!(LTXAV_LATENT_FORMAT.feature_id, "COMFY-MODEL-0039");

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let test_path = crate_root.join("tests/ltx_family_adapter.rs");
    let adapter_path = crate_root.join("src/ltx_family.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let ltxv_latent_path = crate_root.join("src/latent_formats/ltxv_sim_model_0040.rs");
    let ltxav_latent_path = crate_root.join("src/latent_formats/ltxav_sim_model_0039.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(&files, "pub struct LtxConfiguration", &[&test_path])?,
        vec![adapter_path.clone()]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0040", &test_path)?,
        vec![ltxv_latent_path]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0039", &test_path)?,
        vec![ltxav_latent_path]
    );
    let transaction_declaration = ["pub struct Model", "StateTransaction"].concat();
    assert_eq!(
        files_containing(&files, &transaction_declaration, &[&test_path])?,
        vec![foundation_path]
    );
    let adapter = fs::read_to_string(adapter_path)?;
    for forbidden in [
        ["struct Model", "StateTransaction"].concat(),
        ["struct Patch", "Graph"].concat(),
        ["fn estimate_model", "_memory"].concat(),
        ["struct Cancellation", "Token"].concat(),
    ] {
        assert!(!adapter.contains(&forbidden));
    }
    Ok(())
}

fn probe(
    layout: LtxLayout,
    variant: LtxVariant,
    cross_attention_dimension: u64,
    head_dimension: u64,
    layers: usize,
) -> ModelProbe {
    let prefix = prefix(layout);
    let inner = head_dimension * 32;
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}adaln_single.emb.timestep_embedder.linear_1.bias"),
            vec![inner],
        ),
        (format!("{prefix}patchify_proj.weight"), vec![inner, 128]),
        (format!("{prefix}proj_out.weight"), vec![128, inner]),
    ]);
    for index in 0..layers {
        tensor_shapes.insert(
            format!("{prefix}transformer_blocks.{index}.attn2.to_k.weight"),
            vec![inner, cross_attention_dimension],
        );
    }
    if variant == LtxVariant::AudioVideo {
        tensor_shapes.insert(
            format!("{prefix}audio_adaln_single.linear.weight"),
            vec![2_048, 2_048],
        );
        tensor_shapes.insert(
            format!("{prefix}audio_patchify_proj.weight"),
            vec![2_048, 128],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        ltx_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: LtxLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = prefix(layout);
    let model_keys = [
        "adaln_single.emb.timestep_embedder.linear_1.bias",
        "patchify_proj.weight",
        "transformer_blocks.0.attn2.to_k.weight",
        "proj_out.weight",
        "audio_adaln_single.linear.weight",
        "audio_patchify_proj.weight",
    ];
    model_keys
        .into_iter()
        .map(|key| format!("{prefix}{key}"))
        .chain([
            "vae.decoder.weight".to_owned(),
            "text_encoders.t5xxl.weight".to_owned(),
        ])
        .enumerate()
        .map(|(index, key)| Ok((key, tensor(backend, context, index as f32 + 1.0)?)))
        .collect()
}

fn prefix(layout: LtxLayout) -> &'static str {
    match layout {
        LtxLayout::PrefixedNative => "model.diffusion_model.",
        LtxLayout::SavedModel => "model.",
        LtxLayout::StandaloneNative => "",
    }
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
