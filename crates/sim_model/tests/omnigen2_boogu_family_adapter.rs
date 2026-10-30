use sim_model::{
    BOOGU_AXES_DIMENSIONS, BOOGU_AXES_LENGTHS, BOOGU_CLIP_TARGET, BOOGU_FORWARD_PROGRAM,
    BOOGU_HEAD_COUNT, BOOGU_KV_HEAD_COUNT, BOOGU_MEMORY_ESTIMATOR, BOOGU_MEMORY_USAGE_FACTOR,
    BOOGU_SUPPORTED_DTYPES, ModelFamilyError, ModelProbe, ModelStateTransaction,
    OMNIGEN2_AXES_DIMENSIONS, OMNIGEN2_AXES_LENGTHS, OMNIGEN2_BASE_SUPPORTED_DTYPES,
    OMNIGEN2_BOOGU_CONDITIONING, OMNIGEN2_BOOGU_LATENT_FORMAT, OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN,
    OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN, OMNIGEN2_CLIP_TARGET, OMNIGEN2_EXTENDED_SUPPORTED_DTYPES,
    OMNIGEN2_FORWARD_PROGRAM, OMNIGEN2_HEAD_COUNT, OMNIGEN2_HIDDEN_SIZE, OMNIGEN2_KV_HEAD_COUNT,
    OMNIGEN2_LAYER_COUNT, OMNIGEN2_MEMORY_ESTIMATOR, OMNIGEN2_MEMORY_USAGE_FACTOR,
    OMNIGEN2_REFINER_LAYER_COUNT, Omnigen2BooguConditioningFact, Omnigen2BooguLayout,
    Omnigen2BooguVariant, omnigen2_boogu_configuration_for_probe,
    omnigen2_boogu_state_plan_for_layout, omnigen2_boogu_supported_dtypes_for_capabilities,
};
use sim_tensor::{
    BackendCapabilityMatrix, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    Layout, OperationSupport, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
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
fn val_model_detection_001_boogu_precedes_omnigen2_and_both_native_layouts_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        Omnigen2BooguLayout::PrefixedNative,
        Omnigen2BooguLayout::StandaloneNative,
    ] {
        let omnigen =
            omnigen2_boogu_configuration_for_probe(&probe(layout, Omnigen2BooguVariant::Omnigen2))?;
        assert_eq!(omnigen.variant, Omnigen2BooguVariant::Omnigen2);
        assert_eq!(omnigen.layout, layout);
        assert_eq!(omnigen.hidden_size, OMNIGEN2_HIDDEN_SIZE);
        assert_eq!(omnigen.number_of_layers, OMNIGEN2_LAYER_COUNT);
        assert_eq!(
            omnigen.number_of_refiner_layers,
            OMNIGEN2_REFINER_LAYER_COUNT
        );
        assert_eq!(omnigen.number_of_double_stream_layers, 0);
        assert_eq!(omnigen.number_of_attention_heads, OMNIGEN2_HEAD_COUNT);
        assert_eq!(omnigen.number_of_kv_heads, OMNIGEN2_KV_HEAD_COUNT);
        assert_eq!(omnigen.axes_dimensions, OMNIGEN2_AXES_DIMENSIONS);
        assert_eq!(omnigen.axes_lengths, OMNIGEN2_AXES_LENGTHS);
        assert_eq!(omnigen.memory_usage_factor, OMNIGEN2_MEMORY_USAGE_FACTOR);
        assert_eq!(omnigen.memory_estimator, OMNIGEN2_MEMORY_ESTIMATOR);
        assert_eq!(
            omnigen.base_supported_dtypes,
            OMNIGEN2_BASE_SUPPORTED_DTYPES
        );
        assert_eq!(omnigen.latent_format.feature_id, "COMFY-MODEL-0029");
        assert!(std::ptr::eq(omnigen.clip_target, &OMNIGEN2_CLIP_TARGET));

        let boogu =
            omnigen2_boogu_configuration_for_probe(&probe(layout, Omnigen2BooguVariant::Boogu))?;
        assert_eq!(boogu.variant, Omnigen2BooguVariant::Boogu);
        assert_eq!(boogu.hidden_size, 3_360);
        assert_eq!(boogu.number_of_layers, 32);
        assert_eq!(boogu.number_of_double_stream_layers, 8);
        assert_eq!(boogu.number_of_refiner_layers, 2);
        assert_eq!(boogu.number_of_attention_heads, BOOGU_HEAD_COUNT);
        assert_eq!(boogu.number_of_kv_heads, BOOGU_KV_HEAD_COUNT);
        assert_eq!(boogu.instruction_feature_dimension, 3_360);
        assert_eq!(boogu.axes_dimensions, BOOGU_AXES_DIMENSIONS);
        assert_eq!(boogu.axes_lengths, BOOGU_AXES_LENGTHS);
        assert_eq!(boogu.memory_usage_factor, BOOGU_MEMORY_USAGE_FACTOR);
        assert_eq!(boogu.memory_estimator, BOOGU_MEMORY_ESTIMATOR);
        assert_eq!(boogu.base_supported_dtypes, BOOGU_SUPPORTED_DTYPES);
        assert!(std::ptr::eq(boogu.clip_target, &BOOGU_CLIP_TARGET));
    }
    Ok(())
}

#[test]
fn val_model_detection_001_malformed_partial_mixed_gapped_and_pseudo_diffusers_fail() {
    let mut partial = probe(
        Omnigen2BooguLayout::StandaloneNative,
        Omnigen2BooguVariant::Omnigen2,
    );
    partial.tensor_shapes.remove("x_embedder.weight");
    assert!(matches!(
        omnigen2_boogu_configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no exact")
    ));

    let mut mixed = probe(
        Omnigen2BooguLayout::PrefixedNative,
        Omnigen2BooguVariant::Boogu,
    );
    mixed.tensor_shapes.extend(
        probe(
            Omnigen2BooguLayout::StandaloneNative,
            Omnigen2BooguVariant::Boogu,
        )
        .tensor_shapes,
    );
    assert!(matches!(
        omnigen2_boogu_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = probe(
        Omnigen2BooguLayout::StandaloneNative,
        Omnigen2BooguVariant::Boogu,
    );
    gap.tensor_shapes
        .remove("single_stream_layers.1.attn.to_q.weight");
    assert_invalid(gap, "not a consecutive bounded sequence");

    let mut wrong_omnigen = probe(
        Omnigen2BooguLayout::StandaloneNative,
        Omnigen2BooguVariant::Omnigen2,
    );
    wrong_omnigen
        .tensor_shapes
        .remove("layers.31.attn.to_q.weight");
    assert_invalid(wrong_omnigen, "requires hidden/layer/refiner values");

    let pseudo_diffusers = ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "transformer_blocks.0.attn.to_q.weight".to_owned(),
                vec![2, 2],
            ),
            ("proj_out.weight".to_owned(), vec![2, 2]),
        ]),
        metadata: BTreeMap::from([("image_model".to_owned(), "boogu".to_owned())]),
    };
    assert!(matches!(
        omnigen2_boogu_configuration_for_probe(&pseudo_diffusers),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("source-native")
    ));
}

#[test]
fn val_model_family_row_001_state_plans_are_source_native_complete_and_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        Omnigen2BooguLayout::PrefixedNative,
        Omnigen2BooguLayout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &omnigen2_boogu_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        for key in [
            "native.x_embedder.weight",
            "native.time_caption_embed.timestep_embedder.linear_1.bias",
            "native.layers.0.attn.to_q.weight",
            "native.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
            "native.single_stream_layers.0.attn.to_q.weight",
            "native.noise_refiner.0.attn.to_q.weight",
            "native.ref_image_refiner.0.attn.to_q.weight",
            "native.context_refiner.0.attn.to_q.weight",
            "native.norm_out.linear_2.weight",
        ] {
            assert!(model.contains_key(key), "{layout:?}: {key}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
        let reference_count = mapped
            .component("runtime_conditioning")
            .and_then(|component| component.get("reference_latent_count"))
            .ok_or("missing generated reference count")?;
        assert_eq!(reference_count.descriptor().dtype(), DType::I64);
    }
    assert_eq!(
        omnigen2_boogu_state_plan_for_layout(Omnigen2BooguLayout::PrefixedNative).encoded_plan,
        OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        omnigen2_boogu_state_plan_for_layout(Omnigen2BooguLayout::StandaloneNative).encoded_plan,
        OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_memory_001_canonical_capability_gates_only_omnigen2_f16_and_factors_drive_estimates()
-> Result<(), Box<dyn std::error::Error>> {
    let base = BackendCapabilityMatrix::new(DeviceId::CPU, Vec::new(), Vec::new())?;
    let fp16_support = OperationSupport::allocation(DType::F16, Layout::Contiguous);
    let extended =
        BackendCapabilityMatrix::new(DeviceId::CPU, vec![fp16_support], vec![fp16_support])?;
    assert_eq!(
        omnigen2_boogu_supported_dtypes_for_capabilities(Omnigen2BooguVariant::Omnigen2, &base),
        OMNIGEN2_BASE_SUPPORTED_DTYPES
    );
    assert_eq!(
        omnigen2_boogu_supported_dtypes_for_capabilities(Omnigen2BooguVariant::Omnigen2, &extended),
        OMNIGEN2_EXTENDED_SUPPORTED_DTYPES
    );
    assert_eq!(
        omnigen2_boogu_supported_dtypes_for_capabilities(Omnigen2BooguVariant::Boogu, &extended),
        BOOGU_SUPPORTED_DTYPES
    );
    assert_eq!(OMNIGEN2_MEMORY_USAGE_FACTOR, 1.95);
    assert_eq!(OMNIGEN2_MEMORY_ESTIMATOR.bytes_per_parameter, 4);
    assert_eq!(OMNIGEN2_MEMORY_ESTIMATOR.activation_bytes_per_element, 8);
    assert_eq!(BOOGU_MEMORY_USAGE_FACTOR, 2.15);
    assert_eq!(BOOGU_MEMORY_ESTIMATOR.bytes_per_parameter, 4);
    assert_eq!(BOOGU_MEMORY_ESTIMATOR.activation_bytes_per_element, 9);
    assert_eq!(OMNIGEN2_FORWARD_PROGRAM.len(), 5);
    assert_eq!(BOOGU_FORWARD_PROGRAM.len(), 5);
    Ok(())
}

#[test]
fn val_cancel_001_val_ownership_001_conditioning_latent_and_failures_have_one_owner()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(OMNIGEN2_BOOGU_LATENT_FORMAT.feature_id, "COMFY-MODEL-0029");
    assert!(
        OMNIGEN2_BOOGU_CONDITIONING
            .contains(&Omnigen2BooguConditioningFact::OptionalReferenceLatents)
    );
    assert!(
        OMNIGEN2_BOOGU_CONDITIONING
            .contains(&Omnigen2BooguConditioningFact::ReferenceLatentsAffectMemoryEstimate)
    );

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let source = mapping_source(&backend, &context, Omnigen2BooguLayout::PrefixedNative)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    let cancelled = ModelStateTransaction::new(&backend, &context).execute(
        &OMNIGEN2_BOOGU_PREFIXED_STATE_PLAN.compile()?,
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
    let test_path = crate_root.join("tests/omnigen2_boogu_family_adapter.rs");
    let adapter_path = crate_root.join("src/omnigen2_boogu_family.rs");
    let row_path = crate_root.join("src/families/boogu_sim_model_0065.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let flux_latent_path = crate_root.join("src/latent_formats/flux_sim_model_0029.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(
            &files,
            "pub struct Omnigen2BooguConfiguration",
            &[&test_path]
        )?,
        vec![adapter_path.clone()]
    );
    assert_eq!(
        latent_owner(&files, "COMFY-MODEL-0029", &test_path)?,
        vec![flux_latent_path]
    );
    let transaction_declaration = ["pub struct Model", "StateTransaction"].concat();
    assert_eq!(
        files_containing(&files, &transaction_declaration, &[&test_path])?,
        vec![foundation_path]
    );
    let row = fs::read_to_string(row_path)?;
    for imported in [
        "omnigen2_boogu_family",
        "BOOGU_MEMORY_ESTIMATOR",
        "BOOGU_FORWARD_PROGRAM",
        "BOOGU_COMPONENT_STATE_SCHEMAS",
        "OMNIGEN2_BOOGU_STANDALONE_STATE_PLAN",
    ] {
        assert!(row.contains(imported));
    }
    assert!(!row.contains("Diffusers"));
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

fn probe(layout: Omnigen2BooguLayout, variant: Omnigen2BooguVariant) -> ModelProbe {
    let prefix = prefix(layout);
    let (hidden_size, heads, layers, double_layers, refiners) = match variant {
        Omnigen2BooguVariant::Omnigen2 => (
            OMNIGEN2_HIDDEN_SIZE,
            OMNIGEN2_HEAD_COUNT,
            OMNIGEN2_LAYER_COUNT,
            0,
            OMNIGEN2_REFINER_LAYER_COUNT,
        ),
        Omnigen2BooguVariant::Boogu => (3_360, BOOGU_HEAD_COUNT, 32, 8, 2),
    };
    let mut tensor_shapes = BTreeMap::from([
        (format!("{prefix}x_embedder.weight"), vec![hidden_size, 64]),
        (
            format!("{prefix}norm_out.linear_2.weight"),
            vec![64, hidden_size],
        ),
        (
            format!("{prefix}time_caption_embed.timestep_embedder.linear_1.bias"),
            vec![hidden_size],
        ),
    ]);
    let head_dimension = hidden_size / heads;
    for index in 0..refiners {
        tensor_shapes.insert(
            format!("{prefix}noise_refiner.{index}.attn.to_q.weight"),
            vec![hidden_size, hidden_size],
        );
        tensor_shapes.insert(
            format!("{prefix}context_refiner.{index}.attn.to_q.weight"),
            vec![hidden_size, hidden_size],
        );
        tensor_shapes.insert(
            format!("{prefix}ref_image_refiner.{index}.attn.to_q.weight"),
            vec![hidden_size, hidden_size],
        );
    }
    match variant {
        Omnigen2BooguVariant::Omnigen2 => {
            for index in 0..layers {
                tensor_shapes.insert(
                    format!("{prefix}layers.{index}.attn.to_q.weight"),
                    vec![hidden_size, hidden_size],
                );
            }
        }
        Omnigen2BooguVariant::Boogu => {
            tensor_shapes.insert(
                format!("{prefix}time_caption_embed.caption_embedder.0.weight"),
                vec![hidden_size, 4_096],
            );
            for index in 0..layers {
                tensor_shapes.insert(
                    format!("{prefix}single_stream_layers.{index}.attn.to_q.weight"),
                    vec![hidden_size, hidden_size],
                );
            }
            for index in 0..double_layers {
                tensor_shapes.insert(
                    format!(
                        "{prefix}double_stream_layers.{index}.img_instruct_attn.processor.img_to_q.weight"
                    ),
                    vec![hidden_size, hidden_size],
                );
            }
        }
    }
    tensor_shapes.insert(
        format!("{prefix}noise_refiner.0.attn.q_norm.weight"),
        vec![head_dimension],
    );
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        omnigen2_boogu_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: Omnigen2BooguLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = prefix(layout);
    [
        "x_embedder.weight",
        "time_caption_embed.timestep_embedder.linear_1.bias",
        "layers.0.attn.to_q.weight",
        "double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
        "single_stream_layers.0.attn.to_q.weight",
        "noise_refiner.0.attn.to_q.weight",
        "ref_image_refiner.0.attn.to_q.weight",
        "context_refiner.0.attn.to_q.weight",
        "norm_out.linear_2.weight",
    ]
    .into_iter()
    .map(|key| format!("{prefix}{key}"))
    .chain([
        "vae.decoder.weight".to_owned(),
        "text_encoders.language.weight".to_owned(),
    ])
    .enumerate()
    .map(|(index, key)| Ok((key, tensor(backend, context, index as f32 + 1.0)?)))
    .collect()
}

fn prefix(layout: Omnigen2BooguLayout) -> &'static str {
    match layout {
        Omnigen2BooguLayout::PrefixedNative => "model.diffusion_model.",
        Omnigen2BooguLayout::StandaloneNative => "",
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
