use sim_model::{
    LUMINA_AXES_DIMENSIONS, LUMINA_AXES_LENGTHS, LUMINA_CLIP_TARGET, LUMINA_DIMENSION,
    LUMINA_HEAD_COUNT, LUMINA_KV_HEAD_COUNT, LUMINA_MEMORY_USAGE_FACTOR, LUMINA_ROPE_THETA,
    LUMINA_SAMPLING_SHIFT, LUMINA_ZIMAGE_COMMON_MAPPING, LUMINA_ZIMAGE_CONDITIONING,
    LUMINA_ZIMAGE_FORWARD_PROGRAM, LUMINA_ZIMAGE_LATENT_FORMAT, LUMINA_ZIMAGE_PREFIXED_STATE_PLAN,
    LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN, LUMINA_ZIMAGE_STANDALONE_STATE_PLAN,
    LUMINA_ZIMAGE_SUPPORTED_DTYPES, LuminaZImageConditioningFact, LuminaZImageLayout,
    LuminaZImageVariant, ModelFamilyError, ModelProbe, ModelStateTransaction,
    ZIMAGE_AXES_DIMENSIONS, ZIMAGE_AXES_LENGTHS, ZIMAGE_CLIP_TARGET, ZIMAGE_DIFFUSERS_STATE_PLAN,
    ZIMAGE_DIMENSION, ZIMAGE_HEAD_COUNT, ZIMAGE_KV_HEAD_COUNT, ZIMAGE_MEMORY_USAGE_FACTOR,
    ZIMAGE_PAD_TOKENS_MULTIPLE, ZIMAGE_PIXEL_LATENT_FORMAT, ZIMAGE_PIXEL_MEMORY_USAGE_FACTOR,
    ZIMAGE_ROPE_THETA, ZIMAGE_SAMPLING_SHIFT, ZIMAGE_TIME_SCALE, lumina_zimage_common_mapping,
    lumina_zimage_configuration_for_probe, lumina_zimage_state_plan_for_layout,
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
fn val_model_detection_001_pixel_then_zimage_then_lumina_precedence_is_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        LuminaZImageLayout::PrefixedNative,
        LuminaZImageLayout::SavedModel,
        LuminaZImageLayout::StandaloneNative,
    ] {
        let lumina = lumina_zimage_configuration_for_probe(&native_probe(
            layout,
            LuminaZImageVariant::Lumina2,
            3,
        ))?;
        assert_eq!(lumina.variant, LuminaZImageVariant::Lumina2);
        assert_eq!(lumina.layout, layout);
        assert_eq!(lumina.dimension, LUMINA_DIMENSION);
        assert_eq!(lumina.number_of_layers, 3);
        assert_eq!(lumina.number_of_heads, LUMINA_HEAD_COUNT);
        assert_eq!(lumina.number_of_kv_heads, LUMINA_KV_HEAD_COUNT);
        assert_eq!(lumina.axes_dimensions, LUMINA_AXES_DIMENSIONS);
        assert_eq!(lumina.axes_lengths, LUMINA_AXES_LENGTHS);
        assert_eq!(lumina.rope_theta, LUMINA_ROPE_THETA);
        assert_eq!(lumina.feed_forward_multiplier, 4.0);
        assert_eq!(lumina.patch_size, 2);
        assert_eq!(lumina.input_channels, 16);
        assert_eq!(lumina.sampling_shift, LUMINA_SAMPLING_SHIFT);
        assert_eq!(lumina.memory_usage_factor, LUMINA_MEMORY_USAGE_FACTOR);
        assert_eq!(lumina.time_scale, None);
        assert_eq!(lumina.pad_tokens_multiple, None);
        assert_eq!(lumina.latent_format.feature_id, "COMFY-MODEL-0029");
        assert!(std::ptr::eq(lumina.clip_target, &LUMINA_CLIP_TARGET));

        let zimage = lumina_zimage_configuration_for_probe(&native_probe(
            layout,
            LuminaZImageVariant::ZImage,
            4,
        ))?;
        assert_eq!(zimage.variant, LuminaZImageVariant::ZImage);
        assert_eq!(zimage.dimension, ZIMAGE_DIMENSION);
        assert_eq!(zimage.number_of_heads, ZIMAGE_HEAD_COUNT);
        assert_eq!(zimage.number_of_kv_heads, ZIMAGE_KV_HEAD_COUNT);
        assert_eq!(zimage.axes_dimensions, ZIMAGE_AXES_DIMENSIONS);
        assert_eq!(zimage.axes_lengths, ZIMAGE_AXES_LENGTHS);
        assert_eq!(zimage.rope_theta, ZIMAGE_ROPE_THETA);
        assert_eq!(zimage.feed_forward_multiplier, 8.0 / 3.0);
        assert_eq!(zimage.sampling_shift, ZIMAGE_SAMPLING_SHIFT);
        assert_eq!(zimage.memory_usage_factor, ZIMAGE_MEMORY_USAGE_FACTOR);
        assert_eq!(zimage.time_scale, Some(ZIMAGE_TIME_SCALE));
        assert_eq!(zimage.pad_tokens_multiple, Some(ZIMAGE_PAD_TOKENS_MULTIPLE));
        assert_eq!(zimage.clip_text_dimension, Some(1_024));
        assert_eq!(zimage.siglip_feature_dimension, Some(1_152));
        assert!(std::ptr::eq(zimage.clip_target, &ZIMAGE_CLIP_TARGET));

        let pixel = lumina_zimage_configuration_for_probe(&native_probe(
            layout,
            LuminaZImageVariant::ZImagePixelSpace,
            2,
        ))?;
        assert_eq!(pixel.variant, LuminaZImageVariant::ZImagePixelSpace);
        assert_eq!(pixel.patch_size, 4);
        assert_eq!(pixel.input_channels, 3);
        assert_eq!(pixel.output_channels, 3);
        assert_eq!(pixel.memory_usage_factor, ZIMAGE_PIXEL_MEMORY_USAGE_FACTOR);
        assert_eq!(pixel.latent_format.feature_id, "COMFY-MODEL-0055");
        let decoder = pixel.pixel_decoder.ok_or("missing pixel decoder")?;
        assert_eq!(decoder.input_channels, 48);
        assert_eq!(decoder.hidden_size, 512);
        assert_eq!(decoder.number_of_residual_blocks, 2);
        assert_eq!(decoder.maximum_frequencies, 8);
        assert!(decoder.uses_x0);
    }
    Ok(())
}

#[test]
fn val_model_detection_001_only_zimage_admits_pinned_diffusers_and_malformed_layouts_fail()
-> Result<(), Box<dyn std::error::Error>> {
    let zimage = lumina_zimage_configuration_for_probe(&diffusers_probe(3))?;
    assert_eq!(zimage.variant, LuminaZImageVariant::ZImage);
    assert_eq!(zimage.layout, LuminaZImageLayout::Diffusers);
    assert_eq!(zimage.number_of_layers, 3);

    let mut lumina_diffusers = diffusers_probe(2);
    set_diffusers_dimension(&mut lumina_diffusers, LUMINA_DIMENSION);
    assert_invalid(lumina_diffusers, "Lumina2 does not admit");

    let mut pixel_diffusers = diffusers_probe(2);
    pixel_diffusers
        .tensor_shapes
        .insert("dec_net.cond_embed.weight".to_owned(), vec![512, 48]);
    assert_invalid(pixel_diffusers, "does not admit the pinned Diffusers");

    let mut partial = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::Lumina2,
        2,
    );
    partial.tensor_shapes.remove("x_embedder.weight");
    assert!(matches!(
        lumina_zimage_configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no exact")
    ));

    let mut mixed = native_probe(
        LuminaZImageLayout::PrefixedNative,
        LuminaZImageVariant::ZImage,
        2,
    );
    mixed.tensor_shapes.extend(
        native_probe(
            LuminaZImageLayout::StandaloneNative,
            LuminaZImageVariant::ZImage,
            2,
        )
        .tensor_shapes,
    );
    assert!(matches!(
        lumina_zimage_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::Lumina2,
        3,
    );
    gap.tensor_shapes.remove("layers.1.attention.qkv.weight");
    assert_invalid(gap, "not a consecutive bounded sequence");

    let mut unsupported = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::Lumina2,
        2,
    );
    unsupported
        .tensor_shapes
        .insert("cap_embedder.1.weight".to_owned(), vec![3_072, 4_096]);
    unsupported
        .tensor_shapes
        .insert("x_embedder.weight".to_owned(), vec![3_072, 64]);
    unsupported
        .tensor_shapes
        .insert("final_layer.linear.weight".to_owned(), vec![64, 3_072]);
    assert_invalid(unsupported, "neither Lumina2");
    Ok(())
}

#[test]
fn val_model_family_row_001_pixel_decoder_shapes_are_checked_not_guessed() {
    let mut bad_patch = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::ZImagePixelSpace,
        2,
    );
    bad_patch
        .tensor_shapes
        .insert("x_embedder.weight".to_owned(), vec![ZIMAGE_DIMENSION, 45]);
    assert_invalid(bad_patch, "not a perfect square");

    let mut bad_decoder_output = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::ZImagePixelSpace,
        2,
    );
    bad_decoder_output.tensor_shapes.insert(
        "dec_net.final_layer.linear.weight".to_owned(),
        vec![47, 512],
    );
    assert_invalid(bad_decoder_output, "contradicts patch width");

    let mut bad_frequencies = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::ZImagePixelSpace,
        2,
    );
    bad_frequencies.tensor_shapes.insert(
        "dec_net.input_embedder.embedder.0.weight".to_owned(),
        vec![512, 111],
    );
    assert_invalid(bad_frequencies, "not a perfect square");

    let mut bad_attention = native_probe(
        LuminaZImageLayout::StandaloneNative,
        LuminaZImageVariant::ZImage,
        2,
    );
    bad_attention.tensor_shapes.insert(
        "noise_refiner.0.attention.qkv.weight".to_owned(),
        vec![11_519, ZIMAGE_DIMENSION],
    );
    assert_invalid(bad_attention, "must be [11520, 3840]");
}

#[test]
fn val_model_format_001_state_plans_map_native_and_diffusers_transactionally()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        LuminaZImageLayout::PrefixedNative,
        LuminaZImageLayout::SavedModel,
        LuminaZImageLayout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &lumina_zimage_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        for key in [
            "native.cap_embedder.1.weight",
            "native.noise_refiner.0.attention.k_norm.weight",
            "native.noise_refiner.0.attention.qkv.weight",
            "native.x_embedder.weight",
            "native.final_layer.linear.weight",
            "native.dec_net.cond_embed.weight",
        ] {
            assert!(model.contains_key(key), "{layout:?}: {key}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }

    let diffusers = ModelStateTransaction::new(&backend, &context).execute(
        &ZIMAGE_DIFFUSERS_STATE_PLAN.compile()?,
        DIGEST,
        &mapping_source(&backend, &context, LuminaZImageLayout::Diffusers)?,
    )?;
    let model = diffusers.component("model").ok_or("missing model")?;
    for key in [
        "native.x_embedder.weight",
        "native.cap_embedder.1.weight",
        "native.noise_refiner.0.attention.k_norm.weight",
        "native.final_layer.linear.weight",
        "native.diffusers.layers.0.attention.to_q.weight",
        "native.diffusers.noise_refiner.0.attention.to_k.weight",
    ] {
        assert!(model.contains_key(key), "{key}");
    }
    assert_eq!(
        lumina_zimage_state_plan_for_layout(LuminaZImageLayout::PrefixedNative).encoded_plan,
        LUMINA_ZIMAGE_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        lumina_zimage_state_plan_for_layout(LuminaZImageLayout::SavedModel).encoded_plan,
        LUMINA_ZIMAGE_SAVED_MODEL_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        lumina_zimage_state_plan_for_layout(LuminaZImageLayout::StandaloneNative).encoded_plan,
        LUMINA_ZIMAGE_STANDALONE_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_memory_001_val_cancel_001_val_ownership_001_common_facts_have_one_owner()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LUMINA_MEMORY_USAGE_FACTOR, 1.4);
    assert_eq!(ZIMAGE_MEMORY_USAGE_FACTOR, 2.8);
    assert_eq!(ZIMAGE_PIXEL_MEMORY_USAGE_FACTOR, 0.03);
    assert_eq!(LUMINA_ZIMAGE_SUPPORTED_DTYPES, &[DType::Bf16, DType::F32]);
    assert!(
        LUMINA_ZIMAGE_CONDITIONING
            .contains(&LuminaZImageConditioningFact::ReferenceLatentsAffectMemoryEstimate)
    );
    assert!(
        LUMINA_ZIMAGE_CONDITIONING
            .contains(&LuminaZImageConditioningFact::OptionalSiglipVisionFeatures)
    );
    assert!(std::ptr::eq(
        lumina_zimage_common_mapping(),
        &LUMINA_ZIMAGE_COMMON_MAPPING
    ));
    assert_eq!(
        lumina_zimage_common_mapping().forward_program,
        LUMINA_ZIMAGE_FORWARD_PROGRAM
    );
    assert_eq!(LUMINA_ZIMAGE_LATENT_FORMAT.feature_id, "COMFY-MODEL-0029");
    assert_eq!(ZIMAGE_PIXEL_LATENT_FORMAT.feature_id, "COMFY-MODEL-0055");

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let source = mapping_source(&backend, &context, LuminaZImageLayout::PrefixedNative)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    let cancelled = ModelStateTransaction::new(&backend, &context).execute(
        &LUMINA_ZIMAGE_PREFIXED_STATE_PLAN.compile()?,
        DIGEST,
        &source,
    );
    assert!(matches!(cancelled, Err(ModelFamilyError::Cancelled(_))));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let tiny_cancellation = CancellationToken::default();
    let tiny_authorization = authority.authorize_workspace(3)?;
    let tiny_context = backend.execution_context(
        StreamId::DEFAULT,
        tiny_authorization.clone(),
        &tiny_cancellation,
    );
    let input = backend
        .upload_f32(
            TensorDescriptor::contiguous(vec![4], DType::F32, backend.device(), StreamId::DEFAULT)?,
            &[1.0, 2.0, 3.0, 4.0],
            &tiny_context,
        )?
        .0;
    let oom_baseline = backend.memory_snapshot().current_bytes;
    assert!(
        greater_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Scalar(Scalar::Float(2.0)),
            &tiny_context,
        )
        .is_err()
    );
    assert_eq!(tiny_authorization.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, oom_baseline);

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let test_path = crate_root.join("tests/lumina_zimage_family_adapter.rs");
    let adapter_path = crate_root.join("src/lumina_zimage_family.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let flux_latent_path = crate_root.join("src/latent_formats/flux_sim_model_0029.rs");
    let pixel_latent_path =
        crate_root.join("src/latent_formats/zimagepixelspace_sim_model_0055.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(
            &files,
            "pub struct LuminaZImageConfiguration",
            &[&test_path]
        )?,
        vec![adapter_path.clone()]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0029", &test_path)?,
        vec![flux_latent_path]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0055", &test_path)?,
        vec![pixel_latent_path]
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
        ["struct Cancellation", "Token"].concat(),
        ["Command::", "new(\"python"].concat(),
    ] {
        assert!(!adapter.contains(&forbidden));
    }
    Ok(())
}

fn native_probe(
    layout: LuminaZImageLayout,
    variant: LuminaZImageVariant,
    layers: usize,
) -> ModelProbe {
    let prefix = prefix(layout);
    let dimension = match variant {
        LuminaZImageVariant::Lumina2 => LUMINA_DIMENSION,
        LuminaZImageVariant::ZImage | LuminaZImageVariant::ZImagePixelSpace => ZIMAGE_DIMENSION,
    };
    let (heads, kv_heads) = match variant {
        LuminaZImageVariant::Lumina2 => (LUMINA_HEAD_COUNT, LUMINA_KV_HEAD_COUNT),
        _ => (ZIMAGE_HEAD_COUNT, ZIMAGE_KV_HEAD_COUNT),
    };
    let kv_dimension = dimension / heads * kv_heads;
    let qkv_dimension = dimension + kv_dimension * 2;
    let patch_width = if variant == LuminaZImageVariant::ZImagePixelSpace {
        4 * 4 * 3
    } else {
        2 * 2 * 16
    };
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}cap_embedder.1.weight"),
            vec![dimension, 4_096],
        ),
        (
            format!("{prefix}noise_refiner.0.attention.k_norm.weight"),
            vec![dimension / heads],
        ),
        (
            format!("{prefix}noise_refiner.0.attention.qkv.weight"),
            vec![qkv_dimension, dimension],
        ),
        (
            format!("{prefix}x_embedder.weight"),
            vec![dimension, patch_width],
        ),
        (
            format!("{prefix}final_layer.linear.weight"),
            vec![patch_width, dimension],
        ),
    ]);
    for index in 0..layers {
        tensor_shapes.insert(
            format!("{prefix}layers.{index}.attention.qkv.weight"),
            vec![qkv_dimension, dimension],
        );
    }
    if variant != LuminaZImageVariant::Lumina2 {
        tensor_shapes.insert(format!("{prefix}cap_pad_token"), vec![1, dimension]);
        tensor_shapes.insert(
            format!("{prefix}clip_text_pooled_proj.0.weight"),
            vec![1_024, dimension],
        );
        tensor_shapes.insert(
            format!("{prefix}siglip_embedder.0.weight"),
            vec![1_152, dimension],
        );
    }
    if variant == LuminaZImageVariant::ZImagePixelSpace {
        tensor_shapes.extend([
            (format!("{prefix}dec_net.cond_embed.weight"), vec![512, 48]),
            (
                format!("{prefix}dec_net.final_layer.linear.weight"),
                vec![48, 512],
            ),
            (
                format!("{prefix}dec_net.input_embedder.embedder.0.weight"),
                vec![512, 48 + 8 * 8],
            ),
            (
                format!("{prefix}dec_net.res_blocks.0.in_ln.weight"),
                vec![512],
            ),
            (
                format!("{prefix}dec_net.res_blocks.1.in_ln.weight"),
                vec![512],
            ),
            (format!("{prefix}__x0__"), vec![1]),
        ]);
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn diffusers_probe(layers: usize) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        (
            "cap_embedder.1.weight".to_owned(),
            vec![ZIMAGE_DIMENSION, 4_096],
        ),
        (
            "noise_refiner.0.attention.norm_k.weight".to_owned(),
            vec![128],
        ),
        (
            "noise_refiner.0.attention.to_q.weight".to_owned(),
            vec![ZIMAGE_DIMENSION, ZIMAGE_DIMENSION],
        ),
        (
            "noise_refiner.0.attention.to_k.weight".to_owned(),
            vec![ZIMAGE_DIMENSION, ZIMAGE_DIMENSION],
        ),
        (
            "noise_refiner.0.attention.to_v.weight".to_owned(),
            vec![ZIMAGE_DIMENSION, ZIMAGE_DIMENSION],
        ),
        (
            "all_x_embedder.2-1.weight".to_owned(),
            vec![ZIMAGE_DIMENSION, 64],
        ),
        (
            "all_final_layer.2-1.linear.weight".to_owned(),
            vec![64, ZIMAGE_DIMENSION],
        ),
    ]);
    for index in 0..layers {
        tensor_shapes.insert(
            format!("layers.{index}.attention.to_q.weight"),
            vec![ZIMAGE_DIMENSION, ZIMAGE_DIMENSION],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn set_diffusers_dimension(probe: &mut ModelProbe, dimension: u64) {
    for key in [
        "cap_embedder.1.weight",
        "noise_refiner.0.attention.to_q.weight",
        "noise_refiner.0.attention.to_k.weight",
        "noise_refiner.0.attention.to_v.weight",
        "all_x_embedder.2-1.weight",
        "all_final_layer.2-1.linear.weight",
    ] {
        let shape = probe.tensor_shapes.get_mut(key).expect("fixture key");
        match key {
            "cap_embedder.1.weight" => shape[0] = dimension,
            "all_x_embedder.2-1.weight" => shape[0] = dimension,
            "all_final_layer.2-1.linear.weight" => shape[1] = dimension,
            _ => {
                shape[0] = dimension;
                shape[1] = dimension;
            }
        }
    }
    for (key, shape) in &mut probe.tensor_shapes {
        if key.starts_with("layers.") {
            shape[0] = dimension;
            shape[1] = dimension;
        }
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        lumina_zimage_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: LuminaZImageLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let model_keys: Vec<String> = if layout == LuminaZImageLayout::Diffusers {
        [
            "all_x_embedder.2-1.weight",
            "cap_embedder.1.weight",
            "noise_refiner.0.attention.norm_k.weight",
            "noise_refiner.0.attention.to_q.weight",
            "noise_refiner.0.attention.to_k.weight",
            "noise_refiner.0.attention.to_v.weight",
            "layers.0.attention.to_q.weight",
            "all_final_layer.2-1.linear.weight",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    } else {
        let prefix = prefix(layout);
        [
            "cap_embedder.1.weight",
            "noise_refiner.0.attention.k_norm.weight",
            "noise_refiner.0.attention.qkv.weight",
            "layers.0.attention.qkv.weight",
            "x_embedder.weight",
            "final_layer.linear.weight",
            "dec_net.cond_embed.weight",
        ]
        .into_iter()
        .map(|key| format!("{prefix}{key}"))
        .collect()
    };
    model_keys
        .into_iter()
        .chain([
            "vae.decoder.weight".to_owned(),
            "text_encoders.language.weight".to_owned(),
        ])
        .enumerate()
        .map(|(index, key)| Ok((key, tensor(backend, context, index as f32 + 1.0)?)))
        .collect()
}

fn prefix(layout: LuminaZImageLayout) -> &'static str {
    match layout {
        LuminaZImageLayout::PrefixedNative => "model.diffusion_model.",
        LuminaZImageLayout::SavedModel => "model.",
        LuminaZImageLayout::StandaloneNative | LuminaZImageLayout::Diffusers => "",
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
