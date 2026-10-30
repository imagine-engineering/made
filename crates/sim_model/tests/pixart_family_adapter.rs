use sim_model::{
    ModelFamilyError, ModelProbe, ModelStateTransaction, PIXART_ALPHA_CONDITIONING_KEYS,
    PIXART_ALPHA_LATENT_FORMAT, PIXART_BETA_SCHEDULE, PIXART_CAPTION_CHANNELS, PIXART_CLIP_TARGET,
    PIXART_FORWARD_PROGRAM, PIXART_HEAD_COUNT, PIXART_HIDDEN_SIZE, PIXART_INPUT_CHANNELS,
    PIXART_LINEAR_END, PIXART_LINEAR_START, PIXART_MAX_DEPTH, PIXART_MEMORY_ESTIMATOR,
    PIXART_MEMORY_USAGE_FACTOR, PIXART_PATCH_SIZE, PIXART_PREFIXED_NATIVE_STATE_PLAN,
    PIXART_SIGMA_CONDITIONING_KEYS, PIXART_SIGMA_LATENT_FORMAT,
    PIXART_STANDALONE_NATIVE_STATE_PLAN, PIXART_SUPPORTED_DTYPES, PIXART_TIMESTEPS,
    PixArtConditioningKey, PixArtLayout, PixArtVariant, pixart_conditioning_keys_for_variant,
    pixart_configuration_for_probe, pixart_diffusers_state_plan,
    pixart_native_state_plan_for_layout,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, fs, path::Path};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_alpha_sigma_and_native_diffusers_layouts_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        PixArtLayout::PrefixedNative,
        PixArtLayout::StandaloneNative,
        PixArtLayout::Diffusers,
    ] {
        let alpha = pixart_configuration_for_probe(&probe(layout, PixArtVariant::Alpha, 2))?;
        assert_eq!(alpha.variant, PixArtVariant::Alpha);
        assert_eq!(alpha.layout, layout);
        assert_eq!(alpha.hidden_size, PIXART_HIDDEN_SIZE);
        assert_eq!(alpha.number_of_heads, PIXART_HEAD_COUNT);
        assert_eq!(alpha.depth, 2);
        assert_eq!(alpha.input_channels, PIXART_INPUT_CHANNELS);
        assert_eq!(alpha.patch_size, PIXART_PATCH_SIZE);
        assert_eq!(alpha.caption_channels, PIXART_CAPTION_CHANNELS);
        assert_eq!(alpha.model_max_length, 120);
        assert!(alpha.micro_conditioning);
        assert_eq!(alpha.conditioning_keys, PIXART_ALPHA_CONDITIONING_KEYS);
        assert_eq!(alpha.memory_usage_factor, PIXART_MEMORY_USAGE_FACTOR);
        assert_eq!(alpha.supported_dtypes, PIXART_SUPPORTED_DTYPES);
        assert_eq!(alpha.memory_estimator, PIXART_MEMORY_ESTIMATOR);
        assert_eq!(alpha.latent_format.feature_id, "COMFY-MODEL-0045");
        assert_eq!(
            alpha.latent_format.identifier,
            PIXART_ALPHA_LATENT_FORMAT.identifier
        );
        assert!(std::ptr::eq(alpha.clip_target, &PIXART_CLIP_TARGET));
        if layout == PixArtLayout::Diffusers {
            assert_eq!(alpha.input_size, None);
        } else {
            assert_eq!(alpha.input_size, Some(16));
            assert_eq!(alpha.positional_interpolation, Some(0));
        }

        let sigma = pixart_configuration_for_probe(&probe(layout, PixArtVariant::Sigma, 1))?;
        assert_eq!(sigma.variant, PixArtVariant::Sigma);
        assert!(!sigma.micro_conditioning);
        assert_eq!(sigma.conditioning_keys, PIXART_SIGMA_CONDITIONING_KEYS);
        assert_eq!(sigma.latent_format.feature_id, "COMFY-MODEL-0047");
        assert_eq!(
            sigma.latent_format.identifier,
            PIXART_SIGMA_LATENT_FORMAT.identifier
        );
    }
    assert_eq!(PIXART_BETA_SCHEDULE, "sqrt_linear");
    assert_eq!(PIXART_LINEAR_START, 0.0001);
    assert_eq!(PIXART_LINEAR_END, 0.02);
    assert_eq!(PIXART_TIMESTEPS, 1_000);
    Ok(())
}

#[test]
fn val_model_detection_001_partial_mixed_gapped_and_bad_geometry_fail_typed() {
    let mut mixed = probe(PixArtLayout::StandaloneNative, PixArtVariant::Alpha, 1);
    mixed
        .tensor_shapes
        .extend(probe(PixArtLayout::Diffusers, PixArtVariant::Alpha, 1).tensor_shapes);
    assert!(matches!(
        pixart_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = probe(PixArtLayout::StandaloneNative, PixArtVariant::Sigma, 2);
    gap.tensor_shapes.remove("blocks.1.attn.qkv.weight");
    gap.tensor_shapes.insert(
        "blocks.2.attn.qkv.weight".to_owned(),
        vec![3 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
    );
    assert_invalid(gap, "not a consecutive bounded sequence");

    let mut partial_micro = probe(PixArtLayout::StandaloneNative, PixArtVariant::Alpha, 1);
    partial_micro
        .tensor_shapes
        .remove("csize_embedder.mlp.0.weight");
    assert_invalid(partial_micro, "incomplete");

    let mut bad_patch = probe(PixArtLayout::Diffusers, PixArtVariant::Sigma, 1);
    bad_patch.tensor_shapes.insert(
        "pos_embed.proj.weight".to_owned(),
        vec![PIXART_HIDDEN_SIZE, PIXART_INPUT_CHANNELS, 3, 2],
    );
    assert_invalid(bad_patch, "must be");

    let mut bad_position = probe(PixArtLayout::PrefixedNative, PixArtVariant::Sigma, 1);
    bad_position.tensor_shapes.insert(
        "model.diffusion_model.pos_embed".to_owned(),
        vec![1, 63, PIXART_HIDDEN_SIZE],
    );
    assert_invalid(bad_position, "not square");

    assert!(matches!(
        pixart_diffusers_state_plan(PIXART_MAX_DEPTH + 1, PixArtVariant::Sigma),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("outside")
    ));
}

#[test]
fn val_tensor_001_native_mapping_and_diffusers_qkv_assembly_are_transactional()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    for layout in [PixArtLayout::PrefixedNative, PixArtLayout::StandaloneNative] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &pixart_native_state_plan_for_layout(layout)?.compile()?,
            DIGEST,
            &native_mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        assert!(model.contains_key("native.x_embedder.proj.weight"));
        assert!(model.contains_key("native.blocks.0.attn.qkv.weight"));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        pixart_native_state_plan_for_layout(PixArtLayout::PrefixedNative)?.encoded_plan,
        PIXART_PREFIXED_NATIVE_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        pixart_native_state_plan_for_layout(PixArtLayout::StandaloneNative)?.encoded_plan,
        PIXART_STANDALONE_NATIVE_STATE_PLAN.encoded_plan
    );

    let source = diffusers_mapping_source(&backend, &context, PixArtVariant::Sigma)?;
    let mapped = ModelStateTransaction::new(&backend, &context).execute(
        &pixart_diffusers_state_plan(1, PixArtVariant::Sigma)?,
        DIGEST,
        &source,
    )?;
    let model = mapped.component("model").ok_or("missing model")?;
    let qkv = model
        .get("native.blocks.0.attn.qkv.weight")
        .ok_or("missing qkv")?;
    let kv = model
        .get("native.blocks.0.cross_attn.kv_linear.weight")
        .ok_or("missing cross kv")?;
    assert_eq!(qkv.descriptor().shape(), [6, 2]);
    assert_eq!(kv.descriptor().shape(), [4, 2]);
    assert_eq!(
        &*tensor_to_f32_with_context_exact_native(&backend, qkv, &context)?,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
    assert_eq!(
        &*tensor_to_f32_with_context_exact_native(&backend, kv, &context)?,
        &[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
    );
    assert!(!model.keys().any(|key| key.starts_with("conversion.")));
    Ok(())
}

#[test]
fn val_memory_001_val_cancel_001_assembly_oom_and_cancellation_publish_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    const CAPACITY: u64 = 64 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(CAPACITY)?;
    let cancellation = CancellationToken::default();
    let upload_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(CAPACITY)?,
        &cancellation,
    );
    let source = diffusers_mapping_source(&backend, &upload_context, PixArtVariant::Sigma)?;
    let before_filler = backend.memory_snapshot().current_bytes;
    let filler_elements = (CAPACITY - before_filler - 1) / 4;
    let _filler = tensor(
        &backend,
        &upload_context,
        &[filler_elements],
        &vec![0.0; filler_elements as usize],
    )?;
    let baseline = backend.memory_snapshot().current_bytes;
    assert!(
        ModelStateTransaction::new(&backend, &upload_context)
            .execute(
                &pixart_diffusers_state_plan(1, PixArtVariant::Sigma)?,
                DIGEST,
                &source,
            )
            .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    cancellation.cancel();
    assert!(matches!(
        ModelStateTransaction::new(&backend, &upload_context).execute(
            &pixart_diffusers_state_plan(1, PixArtVariant::Sigma)?,
            DIGEST,
            &source,
        ),
        Err(ModelFamilyError::Cancelled(_))
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_latent_001_val_ownership_001_conditioning_program_and_owners_are_canonical()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(PIXART_ALPHA_LATENT_FORMAT.feature_id, "COMFY-MODEL-0045");
    assert_eq!(PIXART_SIGMA_LATENT_FORMAT.feature_id, "COMFY-MODEL-0047");
    assert_eq!(
        PixArtConditioningKey::CrossAttention.as_str(),
        "c_crossattn"
    );
    assert_eq!(PixArtConditioningKey::Size.as_str(), "c_size");
    assert_eq!(PixArtConditioningKey::AspectRatio.as_str(), "c_ar");
    assert_eq!(
        pixart_conditioning_keys_for_variant(PixArtVariant::Alpha),
        PIXART_ALPHA_CONDITIONING_KEYS
    );
    assert_eq!(
        pixart_conditioning_keys_for_variant(PixArtVariant::Sigma),
        PIXART_SIGMA_CONDITIONING_KEYS
    );
    assert_eq!(PIXART_FORWARD_PROGRAM.len(), 5);

    let adapter =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pixart_family.rs"))?;
    for required in [
        "ModelStateTransformOperation::Assemble",
        "generated_sd15_comfy_model_0045::LATENT_FORMAT",
        "generated_sdxl_comfy_model_0047::LATENT_FORMAT",
    ] {
        assert!(adapter.contains(required));
    }
    for forbidden in [
        ["struct Model", "StateTransaction"].concat(),
        ["struct Cancellation", "Token"].concat(),
        ["struct Tensor", "Backend"].concat(),
        ["Command::", "new(\"python"].concat(),
        "tensor_cat_exact_native".to_owned(),
        "reshape_with_context_exact_native".to_owned(),
    ] {
        assert!(!adapter.contains(&forbidden));
    }
    Ok(())
}

fn probe(layout: PixArtLayout, variant: PixArtVariant, depth: usize) -> ModelProbe {
    let prefix = match layout {
        PixArtLayout::PrefixedNative => "model.diffusion_model.",
        PixArtLayout::StandaloneNative | PixArtLayout::Diffusers => "",
    };
    let mut tensor_shapes = BTreeMap::new();
    if layout == PixArtLayout::Diffusers {
        tensor_shapes.extend(BTreeMap::from([
            (
                "adaln_single.emb.timestep_embedder.linear_1.bias".to_owned(),
                vec![PIXART_HIDDEN_SIZE],
            ),
            (
                "adaln_single.linear.weight".to_owned(),
                vec![6 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
            ),
            ("pos_embed.proj.bias".to_owned(), vec![PIXART_HIDDEN_SIZE]),
            (
                "pos_embed.proj.weight".to_owned(),
                vec![
                    PIXART_HIDDEN_SIZE,
                    PIXART_INPUT_CHANNELS,
                    PIXART_PATCH_SIZE,
                    PIXART_PATCH_SIZE,
                ],
            ),
            (
                "caption_projection.y_embedding".to_owned(),
                vec![120, PIXART_CAPTION_CHANNELS],
            ),
            (
                "proj_out.weight".to_owned(),
                vec![
                    2 * PIXART_INPUT_CHANNELS * PIXART_PATCH_SIZE * PIXART_PATCH_SIZE,
                    PIXART_HIDDEN_SIZE,
                ],
            ),
        ]));
        for index in 0..depth {
            for projection in ["to_q", "to_k", "to_v"] {
                tensor_shapes.insert(
                    format!("transformer_blocks.{index}.attn1.{projection}.weight"),
                    vec![PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
                );
            }
        }
        if variant == PixArtVariant::Alpha {
            tensor_shapes.insert(
                "adaln_single.emb.resolution_embedder.linear_1.weight".to_owned(),
                vec![PIXART_HIDDEN_SIZE / 3, 256],
            );
            tensor_shapes.insert(
                "adaln_single.emb.aspect_ratio_embedder.linear_1.weight".to_owned(),
                vec![PIXART_HIDDEN_SIZE / 3, 256],
            );
        }
    } else {
        tensor_shapes.extend(BTreeMap::from([
            (
                format!("{prefix}t_block.1.weight"),
                vec![6 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
            ),
            (
                format!("{prefix}x_embedder.proj.weight"),
                vec![
                    PIXART_HIDDEN_SIZE,
                    PIXART_INPUT_CHANNELS,
                    PIXART_PATCH_SIZE,
                    PIXART_PATCH_SIZE,
                ],
            ),
            (
                format!("{prefix}y_embedder.y_embedding"),
                vec![120, PIXART_CAPTION_CHANNELS],
            ),
            (
                format!("{prefix}final_layer.linear.weight"),
                vec![
                    2 * PIXART_INPUT_CHANNELS * PIXART_PATCH_SIZE * PIXART_PATCH_SIZE,
                    PIXART_HIDDEN_SIZE,
                ],
            ),
            (
                format!("{prefix}pos_embed"),
                vec![1, 64, PIXART_HIDDEN_SIZE],
            ),
        ]));
        for index in 0..depth {
            tensor_shapes.insert(
                format!("{prefix}blocks.{index}.attn.qkv.weight"),
                vec![3 * PIXART_HIDDEN_SIZE, PIXART_HIDDEN_SIZE],
            );
        }
        if variant == PixArtVariant::Alpha {
            tensor_shapes.insert(
                format!("{prefix}ar_embedder.mlp.0.weight"),
                vec![PIXART_HIDDEN_SIZE / 3, 256],
            );
            tensor_shapes.insert(
                format!("{prefix}csize_embedder.mlp.0.weight"),
                vec![PIXART_HIDDEN_SIZE / 3, 256],
            );
        }
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn native_mapping_source(
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    layout: PixArtLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = match layout {
        PixArtLayout::PrefixedNative => "model.diffusion_model.",
        PixArtLayout::StandaloneNative => "",
        PixArtLayout::Diffusers => return Err("Diffusers is not a native layout".into()),
    };
    let mut source = BTreeMap::new();
    for key in [
        "t_block.1.weight",
        "t_embedder.mlp.0.weight",
        "x_embedder.proj.weight",
        "y_embedder.y_proj.fc1.weight",
        "blocks.0.attn.qkv.weight",
        "final_layer.linear.weight",
    ] {
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, context, &[2, 2], &[1.0; 4])?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, context, &[1], &[7.0])?,
    );
    source.insert(
        "text_encoders.t5.weight".to_owned(),
        tensor(backend, context, &[1], &[8.0])?,
    );
    Ok(source)
}

fn diffusers_mapping_source(
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    variant: PixArtVariant,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut source = BTreeMap::new();
    let mut insert = |key: &str, values: &[f32]| -> Result<(), Box<dyn std::error::Error>> {
        source.insert(key.to_owned(), tensor(backend, context, &[2, 2], values)?);
        Ok(())
    };
    if variant == PixArtVariant::Alpha {
        for key in [
            "adaln_single.emb.resolution_embedder.linear_1.weight",
            "adaln_single.emb.resolution_embedder.linear_1.bias",
            "adaln_single.emb.resolution_embedder.linear_2.weight",
            "adaln_single.emb.resolution_embedder.linear_2.bias",
            "adaln_single.emb.aspect_ratio_embedder.linear_1.weight",
            "adaln_single.emb.aspect_ratio_embedder.linear_1.bias",
            "adaln_single.emb.aspect_ratio_embedder.linear_2.weight",
            "adaln_single.emb.aspect_ratio_embedder.linear_2.bias",
        ] {
            insert(key, &[1.0; 4])?;
        }
    }
    for key in [
        "pos_embed.proj.weight",
        "pos_embed.proj.bias",
        "caption_projection.y_embedding",
        "caption_projection.linear_1.weight",
        "caption_projection.linear_1.bias",
        "caption_projection.linear_2.weight",
        "caption_projection.linear_2.bias",
        "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias",
        "adaln_single.linear.weight",
        "adaln_single.linear.bias",
        "proj_out.weight",
        "proj_out.bias",
        "scale_shift_table",
    ] {
        insert(key, &[1.0; 4])?;
    }
    insert(
        "transformer_blocks.0.attn1.to_q.weight",
        &[0.0, 1.0, 2.0, 3.0],
    )?;
    insert(
        "transformer_blocks.0.attn1.to_k.weight",
        &[4.0, 5.0, 6.0, 7.0],
    )?;
    insert(
        "transformer_blocks.0.attn1.to_v.weight",
        &[8.0, 9.0, 10.0, 11.0],
    )?;
    for key in [
        "transformer_blocks.0.attn1.to_q.bias",
        "transformer_blocks.0.attn1.to_k.bias",
        "transformer_blocks.0.attn1.to_v.bias",
        "transformer_blocks.0.attn2.to_q.weight",
        "transformer_blocks.0.attn2.to_q.bias",
    ] {
        insert(key, &[1.0; 4])?;
    }
    insert(
        "transformer_blocks.0.attn2.to_k.weight",
        &[20.0, 21.0, 22.0, 23.0],
    )?;
    insert(
        "transformer_blocks.0.attn2.to_v.weight",
        &[24.0, 25.0, 26.0, 27.0],
    )?;
    for key in [
        "transformer_blocks.0.attn2.to_k.bias",
        "transformer_blocks.0.attn2.to_v.bias",
        "transformer_blocks.0.scale_shift_table",
        "transformer_blocks.0.attn1.to_out.0.weight",
        "transformer_blocks.0.attn1.to_out.0.bias",
        "transformer_blocks.0.ff.net.0.proj.weight",
        "transformer_blocks.0.ff.net.0.proj.bias",
        "transformer_blocks.0.ff.net.2.weight",
        "transformer_blocks.0.ff.net.2.bias",
        "transformer_blocks.0.attn2.to_out.0.weight",
        "transformer_blocks.0.attn2.to_out.0.bias",
    ] {
        insert(key, &[1.0; 4])?;
    }
    insert("vae.decoder.weight", &[7.0; 4])?;
    insert("text_encoders.t5.weight", &[8.0; 4])?;
    Ok(source)
}

fn tensor(
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    shape: &[u64],
    values: &[f32],
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        pixart_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}
