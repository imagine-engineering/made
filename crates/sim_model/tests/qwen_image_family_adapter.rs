use sim_model::{
    ModelFamilyError, ModelProbe, ModelStateLayout, PatchApplication, PatchKind, PatchOperation,
    PatchTarget, QWEN_IMAGE_ATTENTION_HEAD_DIMENSION, QWEN_IMAGE_AXES_DIMENSIONS,
    QWEN_IMAGE_BASE_CONDITIONING_KEYS, QWEN_IMAGE_BLOCK_PREFIXES, QWEN_IMAGE_INNER_DIMENSION,
    QWEN_IMAGE_JOINT_ATTENTION_DIMENSION, QWEN_IMAGE_LATENT_FORMAT,
    QWEN_IMAGE_LAYERED_CONDITIONING_KEYS, QWEN_IMAGE_MEMORY_ESTIMATOR,
    QWEN_IMAGE_MEMORY_USAGE_FACTOR, QWEN_IMAGE_NUMBER_OF_ATTENTION_HEADS, QWEN_IMAGE_PATCH_SIZE,
    QWEN_IMAGE_SAMPLING_SHIFT, QWEN_IMAGE_SUPPORTED_DTYPES, QwenImageConditioningKey,
    QwenImageReferenceMethod, empty_latent, qwen_image_checked_patch_graph,
    qwen_image_configuration_for_probe, qwen_image_layered_latent_extent,
};
use sim_tensor::{CpuWorkspaceAuthority, DType, StreamId};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, fs, path::Path};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_native_layouts_and_source_configuration_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    for layout in [
        ModelStateLayout::PrefixedNative,
        ModelStateLayout::StandaloneNative,
    ] {
        let configuration = qwen_image_configuration_for_probe(&probe(layout, 2))?;
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.input_channels, 64);
        assert_eq!(configuration.output_channels, 16);
        assert_eq!(configuration.number_of_layers, 2);
        assert_eq!(configuration.inner_dimension, QWEN_IMAGE_INNER_DIMENSION);
        assert_eq!(
            configuration.number_of_attention_heads,
            QWEN_IMAGE_NUMBER_OF_ATTENTION_HEADS
        );
        assert_eq!(
            configuration.attention_head_dimension,
            QWEN_IMAGE_ATTENTION_HEAD_DIMENSION
        );
        assert_eq!(
            configuration.joint_attention_dimension,
            QWEN_IMAGE_JOINT_ATTENTION_DIMENSION
        );
        assert_eq!(configuration.patch_size, QWEN_IMAGE_PATCH_SIZE);
        assert_eq!(configuration.axes_dimensions, QWEN_IMAGE_AXES_DIMENSIONS);
        assert!(configuration.txt_norm);
        assert!(configuration.supports_reference_images);
        assert_eq!(
            configuration.reference_method,
            QwenImageReferenceMethod::Index
        );
        assert_eq!(
            configuration.conditioning_keys,
            QWEN_IMAGE_BASE_CONDITIONING_KEYS
        );
        assert_eq!(configuration.sampling_shift, QWEN_IMAGE_SAMPLING_SHIFT);
        assert_eq!(
            configuration.memory_usage_factor,
            QWEN_IMAGE_MEMORY_USAGE_FACTOR
        );
        assert_eq!(configuration.memory_estimator, QWEN_IMAGE_MEMORY_ESTIMATOR);
        assert_eq!(configuration.supported_dtypes, QWEN_IMAGE_SUPPORTED_DTYPES);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0053");
        assert_eq!(
            configuration.latent_format.identifier,
            QWEN_IMAGE_LATENT_FORMAT.identifier
        );
    }
    Ok(())
}

#[test]
fn val_model_detection_001_timestep_and_layered_facts_follow_source_precedence()
-> Result<(), Box<dyn std::error::Error>> {
    let mut marker = probe(ModelStateLayout::StandaloneNative, 1);
    marker
        .tensor_shapes
        .insert("__index_timestep_zero__".to_owned(), vec![]);
    let marker = qwen_image_configuration_for_probe(&marker)?;
    assert!(marker.timestep_zero_marker);
    assert!(!marker.use_additional_timestep_condition);
    assert_eq!(
        marker.reference_method,
        QwenImageReferenceMethod::IndexTimestepZero
    );

    let mut layered = probe(ModelStateLayout::StandaloneNative, 1);
    layered
        .tensor_shapes
        .insert("__index_timestep_zero__".to_owned(), vec![]);
    layered.tensor_shapes.insert(
        "time_text_embed.addition_t_embedding.weight".to_owned(),
        vec![2, QWEN_IMAGE_INNER_DIMENSION],
    );
    let layered = qwen_image_configuration_for_probe(&layered)?;
    assert!(layered.timestep_zero_marker);
    assert!(layered.use_additional_timestep_condition);
    assert_eq!(
        layered.reference_method,
        QwenImageReferenceMethod::NegativeIndex
    );
    assert_eq!(
        layered.conditioning_keys,
        QWEN_IMAGE_LAYERED_CONDITIONING_KEYS
    );
    assert_eq!(
        QwenImageConditioningKey::ReferenceLatents.as_str(),
        "ref_latents"
    );
    assert_eq!(
        QwenImageConditioningKey::ReferenceLatentsMethod.as_str(),
        "ref_latents_method"
    );
    assert_eq!(
        QwenImageConditioningKey::AdditionalTimestepCondition.as_str(),
        "additional_t_cond"
    );
    Ok(())
}

#[test]
fn val_model_detection_001_diffusers_mixed_gapped_and_invalid_geometry_fail_typed() {
    let diffusers = probe(ModelStateLayout::Diffusers, 1);
    assert!(matches!(
        qwen_image_configuration_for_probe(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));

    let mut mixed = probe(ModelStateLayout::PrefixedNative, 1);
    mixed
        .tensor_shapes
        .extend(probe(ModelStateLayout::StandaloneNative, 1).tensor_shapes);
    assert!(matches!(
        qwen_image_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let mut gap = probe(ModelStateLayout::StandaloneNative, 2);
    gap.tensor_shapes
        .retain(|key, _| !key.starts_with("transformer_blocks.1."));
    gap.tensor_shapes.insert(
        "transformer_blocks.2.img_mod.1.weight".to_owned(),
        vec![QWEN_IMAGE_INNER_DIMENSION * 6, QWEN_IMAGE_INNER_DIMENSION],
    );
    assert_invalid(gap, "not a consecutive bounded sequence");

    let mut bad_txt_norm = probe(ModelStateLayout::StandaloneNative, 1);
    bad_txt_norm.tensor_shapes.insert(
        "txt_norm.weight".to_owned(),
        vec![QWEN_IMAGE_JOINT_ATTENTION_DIMENSION - 1],
    );
    assert_invalid(bad_txt_norm, "must be 3584");

    let mut bad_channels = probe(ModelStateLayout::StandaloneNative, 1);
    bad_channels.tensor_shapes.insert(
        "img_in.weight".to_owned(),
        vec![QWEN_IMAGE_INNER_DIMENSION, 63],
    );
    assert_invalid(bad_channels, "divisible by four");

    let mut bad_addition = probe(ModelStateLayout::StandaloneNative, 1);
    bad_addition.tensor_shapes.insert(
        "time_text_embed.addition_t_embedding.weight".to_owned(),
        vec![1, QWEN_IMAGE_INNER_DIMENSION],
    );
    assert_invalid(bad_addition, "must be [2, 3072]");
}

#[test]
fn val_latent_001_layered_geometry_delegates_to_canonical_wan21_empty_latent()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );
    let tensor = empty_latent(
        QWEN_IMAGE_LATENT_FORMAT,
        &backend,
        qwen_image_layered_latent_extent(640, 640, 3, 1)?,
        DType::F32,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(tensor.descriptor().shape(), [1, 16, 4, 80, 80]);

    assert!(qwen_image_layered_latent_extent(641, 640, 3, 1).is_err());
    assert!(qwen_image_layered_latent_extent(640, 640, 3, 0).is_err());
    assert!(qwen_image_layered_latent_extent(640, 640, 4_096, 1).is_err());
    Ok(())
}

#[test]
fn val_memory_001_val_cancel_001_layered_allocation_fails_closed()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(128 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(128 * 1024)?,
        &cancellation,
    );
    let extent = qwen_image_layered_latent_extent(640, 640, 3, 1)?;
    let baseline = backend.memory_snapshot().current_bytes;
    assert!(
        empty_latent(
            QWEN_IMAGE_LATENT_FORMAT,
            &backend,
            extent,
            DType::F32,
            StreamId::DEFAULT,
            &context,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    cancellation.cancel();
    assert!(
        empty_latent(
            QWEN_IMAGE_LATENT_FORMAT,
            &backend,
            extent,
            DType::F32,
            StreamId::DEFAULT,
            &context,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_ownership_001_block_catalog_delegates_ordered_commits_to_patch_graph()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(QWEN_IMAGE_BLOCK_PREFIXES.len(), 3);
    assert_eq!(
        QWEN_IMAGE_BLOCK_PREFIXES[0].mapped_prefix,
        "native.transformer_blocks."
    );
    let first = operation("first", 1.0);
    let second = operation("second", 2.0);
    let forward = qwen_image_checked_patch_graph(DIGEST, vec![first.clone(), second.clone()])?;
    let reverse = qwen_image_checked_patch_graph(DIGEST, vec![second, first])?;
    assert_ne!(
        forward.identity().ordered_digest,
        reverse.identity().ordered_digest
    );
    assert_eq!(forward.semantic_operations().len(), 2);

    let invalid = PatchOperation {
        identifier: "invalid".to_owned(),
        kind: PatchKind::DenseDiff,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.img_in.weight".to_owned(),
            expected_shape: vec![1],
            values: vec![1.0],
            application: PatchApplication::Add,
        }],
    };
    assert!(qwen_image_checked_patch_graph(DIGEST, vec![invalid]).is_err());

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let adapter = fs::read_to_string(crate_root.join("src/qwen_image_family.rs"))?;
    for forbidden in [
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "fn estimate_model_memory",
        "struct CancellationToken",
        "struct Tensor",
    ] {
        assert!(!adapter.contains(forbidden));
    }
    assert_eq!(
        source_files_containing(&crate_root.join("src"), "pub struct QwenImageConfiguration")?,
        vec![crate_root.join("src/qwen_image_family.rs")]
    );
    Ok(())
}

#[test]
fn val_model_family_row_001_pinned_source_facts_remain_bound()
-> Result<(), Box<dyn std::error::Error>> {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let detection =
        fs::read_to_string(repository_root.join("projects/comfy/ComfyUI/sim/model_detection.py"))?;
    for fact in [
        "txt_norm.weight",
        "__index_timestep_zero__",
        "time_text_embed.addition_t_embedding.weight",
        "default_ref_method",
        "negative_index",
    ] {
        assert!(detection.contains(fact), "pinned detector lost {fact}");
    }
    let supported =
        fs::read_to_string(repository_root.join("projects/comfy/ComfyUI/sim/supported_models.py"))?;
    for fact in [
        "class QwenImage",
        "memory_usage_factor = 1.8",
        "latent_format = latent_formats.Wan21",
        "torch.bfloat16, torch.float32",
    ] {
        assert!(supported.contains(fact), "pinned family lost {fact}");
    }
    Ok(())
}

fn probe(layout: ModelStateLayout, depth: usize) -> ModelProbe {
    let prefix = match layout {
        ModelStateLayout::PrefixedNative => "model.diffusion_model.",
        ModelStateLayout::StandaloneNative => "",
        ModelStateLayout::Diffusers => "transformer.",
    };
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}txt_norm.weight"),
            vec![QWEN_IMAGE_JOINT_ATTENTION_DIMENSION],
        ),
        (
            format!("{prefix}img_in.weight"),
            vec![QWEN_IMAGE_INNER_DIMENSION, 64],
        ),
        (
            format!("{prefix}txt_in.weight"),
            vec![
                QWEN_IMAGE_INNER_DIMENSION,
                QWEN_IMAGE_JOINT_ATTENTION_DIMENSION,
            ],
        ),
        (
            format!("{prefix}proj_out.weight"),
            vec![
                16 * QWEN_IMAGE_PATCH_SIZE.pow(2),
                QWEN_IMAGE_INNER_DIMENSION,
            ],
        ),
    ]);
    for ordinal in 0..depth {
        let block = format!("{prefix}transformer_blocks.{ordinal}");
        tensor_shapes.insert(
            format!("{block}.img_mod.1.weight"),
            vec![QWEN_IMAGE_INNER_DIMENSION * 6, QWEN_IMAGE_INNER_DIMENSION],
        );
        tensor_shapes.insert(
            format!("{block}.txt_mod.1.weight"),
            vec![QWEN_IMAGE_INNER_DIMENSION * 6, QWEN_IMAGE_INNER_DIMENSION],
        );
        tensor_shapes.insert(
            format!("{block}.attn.to_q.weight"),
            vec![QWEN_IMAGE_INNER_DIMENSION, QWEN_IMAGE_INNER_DIMENSION],
        );
        tensor_shapes.insert(
            format!("{block}.attn.norm_q.weight"),
            vec![QWEN_IMAGE_ATTENTION_HEAD_DIMENSION],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn operation(identifier: &str, value: f32) -> PatchOperation {
    PatchOperation {
        identifier: identifier.to_owned(),
        kind: PatchKind::DenseDiff,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.transformer_blocks.0.attn.to_q.weight".to_owned(),
            expected_shape: vec![1],
            values: vec![value],
            application: PatchApplication::Add,
        }],
    }
}

fn assert_invalid(probe: ModelProbe, needle: &str) {
    match qwen_image_configuration_for_probe(&probe) {
        Err(ModelFamilyError::InvalidSelectorOutput(message)) => assert!(
            message.contains(needle),
            "expected {needle:?} in {message:?}"
        ),
        other => panic!("expected invalid selector containing {needle:?}, got {other:?}"),
    }
}

fn source_files_containing(
    directory: &Path,
    needle: &str,
) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                if fs::read_to_string(&path).is_ok_and(|source| source.contains(needle)) {
                    result.push(path);
                }
            }
        }
    }
    result.sort();
    Ok(result)
}
