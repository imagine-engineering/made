use sim_model::{
    HUNYUAN_IMAGE_CLIP_TARGET, HUNYUAN_IMAGE21_LATENT_FORMAT,
    HUNYUAN_IMAGE21_REFINER_LATENT_FORMAT, HUNYUAN_REFINER_IMAGE_SCALE, HUNYUAN_VIDEO_CLIP_TARGET,
    HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS, HUNYUAN_VIDEO_COMPONENTS, HUNYUAN_VIDEO_FORWARD_PROGRAM,
    HUNYUAN_VIDEO_LATENT_FORMAT, HUNYUAN_VIDEO_PREFIXED_STATE_PLAN, HUNYUAN_VIDEO_SAVE_PREFIX,
    HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN, HUNYUAN_VIDEO_STANDALONE_STATE_PLAN,
    HUNYUAN_VIDEO_SUPPORTED_DEVICES, HUNYUAN_VIDEO_SUPPORTED_DTYPES, HUNYUAN_VIDEO_THETA,
    HUNYUAN_VIDEO15_CLIP_TARGET, HUNYUAN_VIDEO15_LATENT_FORMAT, HUNYUAN_VIDEO15_SUPPORTED_DTYPES,
    HunyuanVideoLayout, HunyuanVideoVariant, ModelFamilyError, ModelKeyRewrite,
    ModelOptionalKeyReplacement, ModelProbe, ModelStateTransaction, augment_refiner_conditioning,
    hunyuan_video_configuration_for_probe, hunyuan_video_state_plan_for_layout,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, RetryRngPolicy, RngAlgorithm,
    RngProfileVersion, RngStream, RngStreamAddress, StreamId, Tensor, TensorBackend,
    TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, fs, path::Path};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_hunyuan_variants_layouts_axes_and_condition_facts_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        HunyuanVideoLayout::PrefixedNative,
        HunyuanVideoLayout::SavedModel,
        HunyuanVideoLayout::StandaloneNative,
    ] {
        let video =
            hunyuan_video_configuration_for_probe(&probe(layout, HunyuanVideoVariant::Video))?;
        assert_eq!(video.variant, HunyuanVideoVariant::Video);
        assert_eq!(video.layout, layout);
        assert_eq!(video.patch_rank, 3);
        assert_eq!(video.axes_dimensions, [16, 56, 56]);
        assert_eq!(video.number_of_heads, 2);
        assert_eq!(video.double_block_depth, 2);
        assert_eq!(video.single_block_depth, 1);
        assert_eq!(video.vector_input_dimension, Some(768));
        assert!(video.guidance_embedding);
        assert!(video.byt5_conditioning);
        assert!(video.mean_flow);
        assert_eq!(video.memory_usage_factor, 1.8);
        assert_eq!(video.sampling_shift, 7.0);
        assert_eq!(
            video.latent_format.feature_id,
            HUNYUAN_VIDEO_LATENT_FORMAT.feature_id
        );
        assert!(std::ptr::eq(video.clip_target, &HUNYUAN_VIDEO_CLIP_TARGET));
    }

    let image = hunyuan_video_configuration_for_probe(&probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Image21,
    ))?;
    assert_eq!(image.variant, HunyuanVideoVariant::Image21);
    assert_eq!(image.patch_rank, 2);
    assert_eq!(image.axes_dimensions, [0, 64, 64]);
    assert_eq!(image.memory_usage_factor, 8.7);
    assert_eq!(image.sampling_shift, 5.0);
    assert_eq!(
        image.latent_format.feature_id,
        HUNYUAN_IMAGE21_LATENT_FORMAT.feature_id
    );
    assert!(std::ptr::eq(image.clip_target, &HUNYUAN_IMAGE_CLIP_TARGET));

    let refiner = hunyuan_video_configuration_for_probe(&probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Image21Refiner,
    ))?;
    assert_eq!(refiner.variant, HunyuanVideoVariant::Image21Refiner);
    assert_eq!(refiner.patch_size, [1, 1, 1]);
    assert_eq!(refiner.sampling_shift, 4.0);
    assert_eq!(
        refiner.latent_format.feature_id,
        HUNYUAN_IMAGE21_REFINER_LATENT_FORMAT.feature_id
    );

    for variant in [
        HunyuanVideoVariant::VideoI2V,
        HunyuanVideoVariant::VideoSkyreelsI2V,
        HunyuanVideoVariant::Video15,
        HunyuanVideoVariant::Video15SrDistilled,
    ] {
        let configuration = hunyuan_video_configuration_for_probe(&probe(
            HunyuanVideoLayout::PrefixedNative,
            variant,
        ))?;
        assert_eq!(configuration.variant, variant);
        if matches!(
            variant,
            HunyuanVideoVariant::Video15 | HunyuanVideoVariant::Video15SrDistilled
        ) {
            assert_eq!(configuration.vision_input_dimension, Some(1_152));
            assert!(configuration.mean_flow_sum);
            assert!(configuration.condition_type_embedding);
            assert_eq!(
                configuration.latent_format.feature_id,
                HUNYUAN_VIDEO15_LATENT_FORMAT.feature_id
            );
            assert!(std::ptr::eq(
                configuration.clip_target,
                &HUNYUAN_VIDEO15_CLIP_TARGET
            ));
        }
    }
    Ok(())
}

#[test]
fn val_model_detection_001_hunyuan_rejects_partial_mixed_pixart_gaps_and_bad_shapes() {
    let mut partial = probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Video,
    );
    partial.tensor_shapes.remove("final_layer.linear.weight");
    assert_invalid(partial, "partial marker set");

    let mut pixart = probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Video,
    );
    pixart
        .tensor_shapes
        .insert("t_block.1.weight".to_owned(), vec![1]);
    assert_invalid(pixart, "PixArt");

    let mut gap = probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Video,
    );
    gap.tensor_shapes.remove("double_blocks.1.attn.weight");
    gap.tensor_shapes
        .insert("double_blocks.2.attn.weight".to_owned(), vec![256, 256]);
    assert_invalid(gap, "not consecutive");

    let mut malformed = probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Video,
    );
    malformed
        .tensor_shapes
        .insert("img_in.proj.weight".to_owned(), vec![250, 16, 1, 2, 2]);
    assert_invalid(malformed, "not divisible");

    let mut vision = probe(
        HunyuanVideoLayout::StandaloneNative,
        HunyuanVideoVariant::Video15,
    );
    vision
        .tensor_shapes
        .insert("vision_in.proj.0.weight".to_owned(), vec![1_024]);
    assert_invalid(vision, "vision input dimension");

    let mut mixed = probe(
        HunyuanVideoLayout::PrefixedNative,
        HunyuanVideoVariant::Video,
    );
    mixed.tensor_shapes.extend(
        probe(
            HunyuanVideoLayout::StandaloneNative,
            HunyuanVideoVariant::Video,
        )
        .tensor_shapes,
    );
    assert!(matches!(
        hunyuan_video_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
}

#[test]
fn val_model_family_foundation_001_ordered_optional_rewrites_preserve_source_order_and_bounds()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        HunyuanVideoLayout::PrefixedNative,
        HunyuanVideoLayout::SavedModel,
        HunyuanVideoLayout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuan_video_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        assert!(model.contains_key("native.txt_in.t_embedder.in_layer.weight"));
        assert!(model.contains_key("native.double_blocks.0.img_attn.norm.query_norm.weight"));
        assert!(model.contains_key("native.double_blocks.0.img_attn_q_norm.weight"));
        assert!(model.keys().all(|key| key.starts_with("native.")));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        hunyuan_video_state_plan_for_layout(HunyuanVideoLayout::PrefixedNative).encoded_plan,
        HUNYUAN_VIDEO_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        hunyuan_video_state_plan_for_layout(HunyuanVideoLayout::SavedModel).encoded_plan,
        HUNYUAN_VIDEO_SAVED_MODEL_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        hunyuan_video_state_plan_for_layout(HunyuanVideoLayout::StandaloneNative).encoded_plan,
        HUNYUAN_VIDEO_STANDALONE_STATE_PLAN.encoded_plan
    );
    assert!(ModelKeyRewrite::ordered_optional(Vec::new()).is_err());
    assert!(
        ModelKeyRewrite::ordered_optional(
            (0..65)
                .map(|index| ModelOptionalKeyReplacement::Contains {
                    from: format!("source_{index}"),
                    to: format!("target_{index}"),
                })
                .collect()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn val_rng_001_refiner_augmentation_delegates_resize_and_commits_rng_only_on_success()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let input = tensor_shape(&backend, &context, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
    let mut no_noise_rng = rng_transaction(13)?;
    let checkpoint = no_noise_rng.checkpoint();
    let resized =
        augment_refiner_conditioning(&backend, &input, 4, 4, 0.0, &mut no_noise_rng, &context)?;
    assert_eq!(resized.descriptor().shape(), &[1, 1, 4, 4]);
    let values = tensor_to_f32_with_context_exact_native(&backend, &resized, &context)?;
    assert!((values[0] - HUNYUAN_REFINER_IMAGE_SCALE).abs() < 1.0e-6);
    assert!((values[15] - 4.0 * HUNYUAN_REFINER_IMAGE_SCALE).abs() < 1.0e-6);
    assert_eq!(no_noise_rng.checkpoint(), checkpoint);

    let mut first_rng = rng_transaction(27)?;
    let mut replay_rng = rng_transaction(27)?;
    let first =
        augment_refiner_conditioning(&backend, &input, 4, 4, 0.25, &mut first_rng, &context)?;
    let replay =
        augment_refiner_conditioning(&backend, &input, 4, 4, 0.25, &mut replay_rng, &context)?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(&backend, &first, &context)?,
        tensor_to_f32_with_context_exact_native(&backend, &replay, &context)?
    );
    assert_eq!(first_rng.checkpoint(), replay_rng.checkpoint());

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancelled,
    );
    let mut cancelled_rng = rng_transaction(41)?;
    let before = cancelled_rng.checkpoint();
    assert!(matches!(
        augment_refiner_conditioning(
            &backend,
            &input,
            4,
            4,
            0.5,
            &mut cancelled_rng,
            &cancelled_context,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    assert_eq!(cancelled_rng.checkpoint(), before);
    Ok(())
}

#[test]
fn val_ownership_001_hunyuan_video_has_one_adapter_and_canonical_owners()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(HUNYUAN_VIDEO_COMPONENTS.len(), 3);
    assert_eq!(HUNYUAN_VIDEO_COMPONENT_STATE_SCHEMAS.len(), 3);
    assert_eq!(HUNYUAN_VIDEO_FORWARD_PROGRAM.len(), 4);
    assert_eq!(HUNYUAN_VIDEO_SAVE_PREFIX, "model.model.");
    assert_eq!(HUNYUAN_VIDEO_THETA, 256);
    assert_eq!(HUNYUAN_VIDEO_SUPPORTED_DTYPES, &[DType::Bf16, DType::F32]);
    assert_eq!(
        HUNYUAN_VIDEO15_SUPPORTED_DTYPES,
        &[DType::F16, DType::Bf16, DType::F32]
    );
    assert_eq!(HUNYUAN_VIDEO_SUPPORTED_DEVICES.len(), 1);

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let adapter = crate_root.join("src/hunyuan_video_family.rs");
    let foundation = crate_root.join("src/model_family.rs");
    let adapter_source = fs::read_to_string(&adapter)?;
    let foundation_source = fs::read_to_string(&foundation)?;
    assert_eq!(
        adapter_source
            .matches("pub struct HunyuanVideoConfiguration")
            .count(),
        1
    );
    assert_eq!(
        foundation_source
            .matches("pub enum ModelOptionalKeyReplacement")
            .count(),
        1
    );
    for forbidden in [
        "struct ModelStateTransaction",
        "struct RngTransaction",
        "struct CpuWorkspaceAuthority",
        "struct CancellationToken",
        "fn resize_with_context_exact_native",
    ] {
        assert!(!adapter_source.contains(forbidden));
    }
    Ok(())
}

fn probe(layout: HunyuanVideoLayout, variant: HunyuanVideoVariant) -> ModelProbe {
    let prefix = layout_prefix(layout);
    let (in_channels, patch) = match variant {
        HunyuanVideoVariant::Image21 => (64, vec![2, 2]),
        HunyuanVideoVariant::Image21Refiner => (64, vec![1, 1, 1]),
        HunyuanVideoVariant::Video => (16, vec![1, 2, 2]),
        HunyuanVideoVariant::VideoI2V => (33, vec![1, 2, 2]),
        HunyuanVideoVariant::VideoSkyreelsI2V | HunyuanVideoVariant::Video15 => (32, vec![1, 2, 2]),
        HunyuanVideoVariant::Video15SrDistilled => (98, vec![1, 2, 2]),
    };
    let hidden = 256;
    let patch_volume = patch.iter().product::<u64>();
    let mut projection = vec![hidden, in_channels];
    projection.extend_from_slice(&patch);
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}txt_in.individual_token_refiner.blocks.0.norm1.weight"),
            vec![hidden],
        ),
        (format!("{prefix}img_in.proj.weight"), projection),
        (
            format!("{prefix}final_layer.linear.weight"),
            vec![in_channels * patch_volume, hidden],
        ),
        (
            format!("{prefix}txt_in.input_embedder.weight"),
            vec![hidden, 4_096],
        ),
        (
            format!("{prefix}double_blocks.0.attn.weight"),
            vec![hidden, hidden],
        ),
        (
            format!("{prefix}double_blocks.1.attn.weight"),
            vec![hidden, hidden],
        ),
        (
            format!("{prefix}single_blocks.0.attn.weight"),
            vec![hidden, hidden],
        ),
    ]);
    if !matches!(
        variant,
        HunyuanVideoVariant::Image21 | HunyuanVideoVariant::Image21Refiner
    ) {
        tensor_shapes.insert(
            format!("{prefix}vector_in.in_layer.weight"),
            vec![hidden, 768],
        );
        tensor_shapes.insert(
            format!("{prefix}guidance_in.in_layer.weight"),
            vec![hidden, 256],
        );
        tensor_shapes.insert(format!("{prefix}byt5_in.fc1.weight"), vec![2_048, 1_472]);
        tensor_shapes.insert(
            format!("{prefix}time_r_in.in_layer.weight"),
            vec![hidden, 256],
        );
    }
    if matches!(
        variant,
        HunyuanVideoVariant::Video15 | HunyuanVideoVariant::Video15SrDistilled
    ) {
        tensor_shapes.insert(format!("{prefix}vision_in.proj.0.weight"), vec![1_152]);
        tensor_shapes.insert(
            format!("{prefix}cond_type_embedding.weight"),
            vec![3, hidden],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn layout_prefix(layout: HunyuanVideoLayout) -> &'static str {
    match layout {
        HunyuanVideoLayout::PrefixedNative => "model.diffusion_model.",
        HunyuanVideoLayout::SavedModel => "model.",
        HunyuanVideoLayout::StandaloneNative => "",
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        hunyuan_video_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: HunyuanVideoLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let keys = [
        "img_in.proj.weight",
        "final_layer.linear.weight",
        "txt_in.input_embedder.weight",
        "txt_in.individual_token_refiner.blocks.0.norm1.weight",
        "txt_in.t_embedder.mlp.0.weight",
        "double_blocks.0.img_attn_q_norm.weight",
        "double_blocks.0.img_attn_q_norm.scale",
        "single_blocks.0.linear.weight",
    ];
    let mut source = BTreeMap::new();
    for (index, key) in keys.iter().enumerate() {
        source.insert(
            format!("{prefix}{key}"),
            scalar_tensor(backend, context, index as f32 + 1.0)?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        scalar_tensor(backend, context, 20.0)?,
    );
    source.insert(
        "text_encoders.qwen.weight".to_owned(),
        scalar_tensor(backend, context, 21.0)?,
    );
    Ok(source)
}

fn scalar_tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    value: f32,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    tensor_shape(backend, context, &[1], &[value])
}

fn tensor_shape(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
    values: &[f32],
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn rng_transaction(seed: u64) -> Result<sim_tensor::RngTransaction, Box<dyn std::error::Error>> {
    let address = RngStreamAddress::new(
        "hunyuan-refiner",
        "attempt-1",
        "conditioning",
        0,
        "noise-augmentation",
        0,
        0,
        RetryRngPolicy::Replay,
    )?;
    Ok(RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        seed,
        address,
    )?
    .begin(None)?)
}
