use sim_model::{
    ModelFamilyError, ModelProbe, ModelStateTransaction, PID_CONDITIONING_KEYS,
    PID_FORWARD_PROGRAM, PID_SAMPLING_SHIFT, PIXELDIT_CLIP_TARGET, PIXELDIT_CONDITIONING_KEYS,
    PIXELDIT_CORE_STATE_PLAN, PIXELDIT_FORWARD_PROGRAM, PIXELDIT_HIDDEN_SIZE,
    PIXELDIT_NET_STATE_PLAN, PIXELDIT_PATCH_DEPTH, PIXELDIT_PATCH_SIZE, PIXELDIT_PID_LATENT_FORMAT,
    PIXELDIT_PID_MEMORY_USAGE_FACTOR, PIXELDIT_PID_SUPPORTED_DTYPES, PIXELDIT_PIXEL_DEPTH,
    PIXELDIT_PIXEL_HIDDEN_SIZE, PIXELDIT_SAMPLING_SHIFT, PIXELDIT_TEXT_FEATURE_DIMENSION,
    PixelDitPidConditioningKey, PixelDitPidLayout, PixelDitPidVariant,
    pixeldit_pid_conditioning_keys_for_variant, pixeldit_pid_configuration_for_probe,
    pixeldit_pid_forward_program_for_variant, pixeldit_pid_state_plan_for_layout,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, StreamId, Tensor, TensorBackend, TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, fs, path::Path};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_pid_precedes_pixeldit_and_core_net_layouts_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [PixelDitPidLayout::CoreNative, PixelDitPidLayout::NetNative] {
        let base = pixeldit_pid_configuration_for_probe(&probe(
            layout,
            PixelDitPidVariant::PixelDitT2I,
            16,
        ))?;
        assert_eq!(base.variant, PixelDitPidVariant::PixelDitT2I);
        assert_eq!(base.layout, layout);
        assert_eq!(base.hidden_size, PIXELDIT_HIDDEN_SIZE);
        assert_eq!(base.pixel_hidden_size, PIXELDIT_PIXEL_HIDDEN_SIZE);
        assert_eq!(base.patch_depth, PIXELDIT_PATCH_DEPTH);
        assert_eq!(base.pixel_depth, PIXELDIT_PIXEL_DEPTH);
        assert_eq!(base.patch_size, PIXELDIT_PATCH_SIZE);
        assert_eq!(base.text_feature_dimension, PIXELDIT_TEXT_FEATURE_DIMENSION);
        assert_eq!(base.sampling_shift, PIXELDIT_SAMPLING_SHIFT);
        assert_eq!(base.memory_usage_factor, PIXELDIT_PID_MEMORY_USAGE_FACTOR);
        assert_eq!(base.supported_dtypes, PIXELDIT_PID_SUPPORTED_DTYPES);
        assert!(base.pid.is_none());
        assert_eq!(base.conditioning_keys, PIXELDIT_CONDITIONING_KEYS);
        assert_eq!(base.latent_format.feature_id, "COMFY-MODEL-0042");
        assert!(std::ptr::eq(base.clip_target, &PIXELDIT_CLIP_TARGET));

        let pid16 =
            pixeldit_pid_configuration_for_probe(&probe(layout, PixelDitPidVariant::PiD, 16))?;
        assert_eq!(pid16.variant, PixelDitPidVariant::PiD);
        assert_eq!(pid16.sampling_shift, PID_SAMPLING_SHIFT);
        assert_eq!(pid16.conditioning_keys, PID_CONDITIONING_KEYS);
        let pid16 = pid16.pid.ok_or("missing PiD configuration")?;
        assert_eq!(pid16.lq_latent_channels, 16);
        assert_eq!(pid16.latent_spatial_down_factor, 8);
        assert_eq!(pid16.lq_gate_count, 7);
        assert_eq!(pid16.lq_interval, 2);

        let pid128 =
            pixeldit_pid_configuration_for_probe(&probe(layout, PixelDitPidVariant::PiD, 128))?;
        assert_eq!(
            pid128
                .pid
                .ok_or("missing PiD configuration")?
                .latent_spatial_down_factor,
            16
        );
    }
    Ok(())
}

#[test]
fn val_model_detection_001_partial_mixed_gapped_and_bad_geometry_fail_typed() {
    let mut partial = probe(
        PixelDitPidLayout::CoreNative,
        PixelDitPidVariant::PixelDitT2I,
        16,
    );
    partial.tensor_shapes.remove("core.s_embedder.proj.weight");
    assert!(matches!(
        pixeldit_pid_configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no exact")
    ));

    let mut mixed = probe(PixelDitPidLayout::CoreNative, PixelDitPidVariant::PiD, 16);
    mixed
        .tensor_shapes
        .extend(probe(PixelDitPidLayout::NetNative, PixelDitPidVariant::PiD, 16).tensor_shapes);
    assert!(matches!(
        pixeldit_pid_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = probe(PixelDitPidLayout::NetNative, PixelDitPidVariant::PiD, 16);
    gap.tensor_shapes
        .remove("net.lq_proj.gate_modules.1.content_proj.weight");
    assert_invalid(gap, "not a consecutive bounded sequence");

    let mut bad_modulation = probe(
        PixelDitPidLayout::CoreNative,
        PixelDitPidVariant::PixelDitT2I,
        16,
    );
    bad_modulation.tensor_shapes.insert(
        "core.pixel_blocks.0.adaLN_modulation.0.weight".to_owned(),
        vec![95, PIXELDIT_HIDDEN_SIZE],
    );
    assert_invalid(bad_modulation, "must be");

    let mut bad_lq = probe(PixelDitPidLayout::CoreNative, PixelDitPidVariant::PiD, 16);
    bad_lq.tensor_shapes.insert(
        "core.lq_proj.latent_proj.0.weight".to_owned(),
        vec![512, 16, 1, 1],
    );
    assert_invalid(bad_lq, "must be [hidden, channels, 3, 3]");
}

#[test]
fn val_tensor_001_state_plans_strip_layout_drop_training_state_and_split_adaln_exactly()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    for layout in [PixelDitPidLayout::CoreNative, PixelDitPidLayout::NetNative] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &pixeldit_pid_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        let msa_weight = model
            .get("native.pixel_blocks.0.adaLN_modulation_msa.weight")
            .ok_or("missing msa weight")?;
        let mlp_weight = model
            .get("native.pixel_blocks.0.adaLN_modulation_mlp.weight")
            .ok_or("missing mlp weight")?;
        let msa_bias = model
            .get("native.pixel_blocks.0.adaLN_modulation_msa.bias")
            .ok_or("missing msa bias")?;
        let mlp_bias = model
            .get("native.pixel_blocks.0.adaLN_modulation_mlp.bias")
            .ok_or("missing mlp bias")?;
        assert_eq!(msa_weight.descriptor().shape(), [12, 2]);
        assert_eq!(mlp_weight.descriptor().shape(), [12, 2]);
        assert_eq!(msa_bias.descriptor().shape(), [12]);
        assert_eq!(mlp_bias.descriptor().shape(), [12]);
        assert_eq!(
            &*tensor_to_f32_with_context_exact_native(&backend, msa_weight, &context)?,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 24.0, 25.0, 26.0,
                27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
            ]
        );
        assert_eq!(
            &*tensor_to_f32_with_context_exact_native(&backend, mlp_weight, &context)?,
            &[
                12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 36.0, 37.0,
                38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
            ]
        );
        assert!(model.contains_key("native.pixel_embedder.proj.weight"));
        assert!(!model.keys().any(|key| key.contains("adaLN_modulation.0")));
        assert!(!model.keys().any(|key| key.starts_with("net_ema")));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        pixeldit_pid_state_plan_for_layout(PixelDitPidLayout::CoreNative).encoded_plan,
        PIXELDIT_CORE_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        pixeldit_pid_state_plan_for_layout(PixelDitPidLayout::NetNative).encoded_plan,
        PIXELDIT_NET_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_cancel_001_invalid_split_and_cancellation_publish_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let mut malformed = mapping_source(&backend, &context, PixelDitPidLayout::CoreNative)?;
    malformed.insert(
        "core.pixel_blocks.0.adaLN_modulation.0.weight".to_owned(),
        tensor(
            &backend,
            &context,
            &[25, 2],
            &(0..50).map(|v| v as f32).collect::<Vec<_>>(),
        )?,
    );
    assert!(
        ModelStateTransaction::new(&backend, &context)
            .execute(&PIXELDIT_CORE_STATE_PLAN.compile()?, DIGEST, &malformed)
            .is_err()
    );

    let source = mapping_source(&backend, &context, PixelDitPidLayout::CoreNative)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(
            &PIXELDIT_CORE_STATE_PLAN.compile()?,
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
    assert_eq!(PIXELDIT_PID_LATENT_FORMAT.feature_id, "COMFY-MODEL-0042");
    assert_eq!(PIXELDIT_PID_LATENT_FORMAT.channels, 3);
    assert_eq!(
        PixelDitPidConditioningKey::AttentionMask.as_str(),
        "attention_mask"
    );
    assert_eq!(PixelDitPidConditioningKey::LqLatent.as_str(), "lq_latent");
    assert_eq!(
        PixelDitPidConditioningKey::DegradeSigma.as_str(),
        "degrade_sigma"
    );
    assert_eq!(
        pixeldit_pid_conditioning_keys_for_variant(PixelDitPidVariant::PixelDitT2I),
        PIXELDIT_CONDITIONING_KEYS
    );
    assert_eq!(
        pixeldit_pid_conditioning_keys_for_variant(PixelDitPidVariant::PiD),
        PID_CONDITIONING_KEYS
    );
    assert_eq!(
        pixeldit_pid_forward_program_for_variant(PixelDitPidVariant::PixelDitT2I),
        PIXELDIT_FORWARD_PROGRAM
    );
    assert_eq!(
        pixeldit_pid_forward_program_for_variant(PixelDitPidVariant::PiD),
        PID_FORWARD_PROGRAM
    );

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let adapter_path = crate_root.join("src/pixeldit_pid_family.rs");
    let latent_path = crate_root.join("src/latent_formats/pixelditpixel_sim_model_0042.rs");
    let adapter = fs::read_to_string(&adapter_path)?;
    let latent = fs::read_to_string(latent_path)?;
    assert!(latent.contains("COMFY-MODEL-0042"));
    for required in [
        "TransformBranchesEach",
        "SourceDimension",
        "tensor_split_exact_native",
        "reshape_with_context_exact_native",
    ] {
        if required.ends_with("exact_native") {
            assert!(!adapter.contains(required));
        } else {
            assert!(adapter.contains(required));
        }
    }
    for forbidden in [
        ["struct Model", "StateTransaction"].concat(),
        ["struct Cancellation", "Token"].concat(),
        ["struct Tensor", "Backend"].concat(),
        ["Command::", "new(\"python"].concat(),
    ] {
        assert!(!adapter.contains(&forbidden));
    }
    Ok(())
}

fn probe(layout: PixelDitPidLayout, variant: PixelDitPidVariant, lq_channels: u64) -> ModelProbe {
    let prefix = prefix(layout);
    let modulation = 6 * PIXELDIT_PIXEL_HIDDEN_SIZE * PIXELDIT_PATCH_SIZE * PIXELDIT_PATCH_SIZE;
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}pixel_embedder.proj.weight"),
            vec![PIXELDIT_PIXEL_HIDDEN_SIZE, 3],
        ),
        (
            format!("{prefix}s_embedder.proj.weight"),
            vec![
                PIXELDIT_HIDDEN_SIZE,
                3 * PIXELDIT_PATCH_SIZE * PIXELDIT_PATCH_SIZE,
            ],
        ),
        (
            format!("{prefix}y_embedder.proj.weight"),
            vec![PIXELDIT_HIDDEN_SIZE, PIXELDIT_TEXT_FEATURE_DIMENSION],
        ),
        (
            format!("{prefix}final_layer.linear.weight"),
            vec![3, PIXELDIT_PIXEL_HIDDEN_SIZE],
        ),
    ]);
    for index in 0..PIXELDIT_PATCH_DEPTH {
        tensor_shapes.insert(
            format!("{prefix}patch_blocks.{index}.attn.qkv_x.weight"),
            vec![3 * PIXELDIT_HIDDEN_SIZE, PIXELDIT_HIDDEN_SIZE],
        );
    }
    for index in 0..PIXELDIT_PIXEL_DEPTH {
        tensor_shapes.insert(
            format!("{prefix}pixel_blocks.{index}.adaLN_modulation.0.weight"),
            vec![modulation, PIXELDIT_HIDDEN_SIZE],
        );
    }
    if variant == PixelDitPidVariant::PiD {
        tensor_shapes.insert(
            format!("{prefix}lq_proj.latent_proj.0.weight"),
            vec![512, lq_channels, 3, 3],
        );
        for index in 0..7 {
            tensor_shapes.insert(
                format!("{prefix}lq_proj.gate_modules.{index}.content_proj.weight"),
                vec![PIXELDIT_HIDDEN_SIZE, 2 * PIXELDIT_HIDDEN_SIZE],
            );
        }
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn mapping_source(
    backend: &CpuBackend,
    context: &sim_tensor::ExecutionContext<'_>,
    layout: PixelDitPidLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = prefix(layout);
    let mut source = BTreeMap::from([
        (
            format!("{prefix}pixel_embedder.proj.weight"),
            tensor(backend, context, &[2, 3], &[1.0; 6])?,
        ),
        (
            format!("{prefix}s_embedder.proj.weight"),
            tensor(backend, context, &[2, 2], &[1.0; 4])?,
        ),
        (
            format!("{prefix}y_embedder.proj.weight"),
            tensor(backend, context, &[2, 2], &[1.0; 4])?,
        ),
        (
            format!("{prefix}patch_blocks.0.attn.qkv_x.weight"),
            tensor(backend, context, &[2, 2], &[1.0; 4])?,
        ),
        (
            format!("{prefix}pixel_blocks.0.adaLN_modulation.0.weight"),
            tensor(
                backend,
                context,
                &[24, 2],
                &(0..48).map(|value| value as f32).collect::<Vec<_>>(),
            )?,
        ),
        (
            format!("{prefix}pixel_blocks.0.adaLN_modulation.0.bias"),
            tensor(
                backend,
                context,
                &[24],
                &(0..24).map(|value| value as f32).collect::<Vec<_>>(),
            )?,
        ),
        (
            format!("{prefix}final_layer.linear.weight"),
            tensor(backend, context, &[2, 2], &[1.0; 4])?,
        ),
        (
            "_repa_projector.weight".to_owned(),
            tensor(backend, context, &[1], &[9.0])?,
        ),
        (
            "net_ema.shadow".to_owned(),
            tensor(backend, context, &[1], &[9.0])?,
        ),
        (
            "vae.decoder.weight".to_owned(),
            tensor(backend, context, &[1], &[7.0])?,
        ),
        (
            "text_encoders.gemma.weight".to_owned(),
            tensor(backend, context, &[1], &[8.0])?,
        ),
    ]);
    if layout == PixelDitPidLayout::NetNative {
        source.insert(
            "net.lq_proj.latent_proj.0.weight".to_owned(),
            tensor(backend, context, &[1], &[6.0])?,
        );
    }
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

fn prefix(layout: PixelDitPidLayout) -> &'static str {
    match layout {
        PixelDitPidLayout::CoreNative => "core.",
        PixelDitPidLayout::NetNative => "net.",
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        pixeldit_pid_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}
