use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_model::clip_vision::{
    CLIP_VISION_CATALOG_SYMBOLS, CLIP_VISION_SOURCE_PATH, CLIP_VISION_SOURCE_SHA256,
    ClipVisionActivation, ClipVisionConfiguration, ClipVisionError, ClipVisionIntermediate,
    ClipVisionLayerWeights, ClipVisionModelType, ClipVisionWeights, NativeClipVision,
    clip_preprocess_with_context, siglip2_flex_resolution, siglip2_preprocess_with_context,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorDescriptor, generated_native_diffusion::tensor_to_f32,
};
use std::{error::Error, fs, path::Path, sync::Arc};

const MEMORY_LIMIT: u64 = 64 * 1024 * 1024;
const VISION_TASK: &str = "comfy-parity-clip-vision-foundation";
const VISION_IMPLEMENTATION_CLOSURE: [&str; 3] = [
    "crates/sim_model/src/clip_vision.rs",
    "crates/sim_model/src/sim_model.rs",
    "crates/sim_model/tests/clip_vision.rs",
];

fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn Error>> {
    let source = std::str::from_utf8(source)?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let function_prefix = format!("def {symbol}(");
    let class_with_bases_prefix = format!("class {symbol}(");
    let class_without_bases_prefix = format!("class {symbol}:");
    let start = lines
        .iter()
        .position(|line| {
            line.starts_with(&function_prefix)
                || line.starts_with(&class_with_bases_prefix)
                || line.starts_with(&class_without_bases_prefix)
        })
        .ok_or_else(|| format!("Python symbol {symbol:?} is missing"))?;
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find_map(|(index, line)| {
            let trimmed = line.trim();
            (!trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !line.starts_with(char::is_whitespace))
            .then_some(index)
        })
        .unwrap_or(lines.len());
    let mut body_end = end;
    while body_end > start + 1 {
        let trimmed = lines[body_end - 1].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            body_end -= 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(lines[start..body_end].concat().as_bytes())
    ))
}

fn backend() -> Result<(Arc<CpuBackend>, CpuWorkspaceAuthority), Box<dyn Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    Ok((Arc::new(backend), authority))
}

fn context<'a>(
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
    scratch: u64,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority.authorize_workspace(scratch)?,
        rng_phase: None,
        cancellation,
    })
}

fn tensor(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn filled_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    value: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or("tensor element count overflowed")?;
    tensor(
        backend,
        shape,
        &vec![value; usize::try_from(count)?],
        context,
    )
}

fn identity_matrix(
    backend: &CpuBackend,
    rows: usize,
    columns: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let mut values = vec![0.0; rows * columns];
    for index in 0..rows.min(columns) {
        values[index * columns + index] = 1.0;
    }
    tensor(
        backend,
        &[u64::try_from(rows)?, u64::try_from(columns)?],
        &values,
        context,
    )
}

fn configuration(model_type: ClipVisionModelType) -> ClipVisionConfiguration {
    ClipVisionConfiguration {
        model_type,
        dtype: DType::F32,
        device: DeviceId::CPU,
        hidden_size: 4,
        intermediate_size: 8,
        attention_heads: 2,
        layer_count: 2,
        image_size: if model_type == ClipVisionModelType::Siglip2 {
            0
        } else {
            4
        },
        patch_size: 2,
        num_channels: 3,
        max_num_patches: 4,
        activation: match model_type {
            ClipVisionModelType::Clip => ClipVisionActivation::QuickGelu,
            ClipVisionModelType::Siglip => ClipVisionActivation::Gelu,
            ClipVisionModelType::Siglip2 => ClipVisionActivation::GeluTanh,
        },
        projection_dimension: Some(3),
        llava_projection_dimension: if model_type == ClipVisionModelType::Clip {
            Some(6)
        } else {
            None
        },
    }
}

fn layer_weights(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<ClipVisionLayerWeights, Box<dyn Error>> {
    Ok(ClipVisionLayerWeights {
        layer_norm_1_weight: filled_tensor(backend, &[4], 1.0, context)?,
        layer_norm_1_bias: filled_tensor(backend, &[4], 0.0, context)?,
        query_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        query_bias: filled_tensor(backend, &[4], 0.0, context)?,
        key_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        key_bias: filled_tensor(backend, &[4], 0.0, context)?,
        value_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        value_bias: filled_tensor(backend, &[4], 0.0, context)?,
        output_weight: filled_tensor(backend, &[4, 4], 0.0, context)?,
        output_bias: filled_tensor(backend, &[4], 0.0, context)?,
        layer_norm_2_weight: filled_tensor(backend, &[4], 1.0, context)?,
        layer_norm_2_bias: filled_tensor(backend, &[4], 0.0, context)?,
        feed_forward_1_weight: filled_tensor(backend, &[8, 4], 0.0, context)?,
        feed_forward_1_bias: filled_tensor(backend, &[8], 0.0, context)?,
        feed_forward_2_weight: filled_tensor(backend, &[4, 8], 0.0, context)?,
        feed_forward_2_bias: filled_tensor(backend, &[4], 0.0, context)?,
    })
}

fn weights(
    backend: &CpuBackend,
    model_type: ClipVisionModelType,
    context: &ExecutionContext<'_>,
) -> Result<ClipVisionWeights, Box<dyn Error>> {
    let patch_weight = match model_type {
        ClipVisionModelType::Siglip2 => identity_matrix(backend, 4, 12, context)?,
        ClipVisionModelType::Clip | ClipVisionModelType::Siglip => {
            let mut values = vec![0.0; 4 * 3 * 2 * 2];
            for output in 0..4 {
                values[output * 12 + output] = 1.0;
            }
            tensor(backend, &[4, 3, 2, 2], &values, context)?
        }
    };
    let patch_bias = match model_type {
        ClipVisionModelType::Clip => None,
        ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => {
            Some(filled_tensor(backend, &[4], 0.0, context)?)
        }
    };
    let class_embedding = match model_type {
        ClipVisionModelType::Clip => Some(tensor(backend, &[4], &[1.0, -1.0, 0.5, -0.5], context)?),
        ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => None,
    };
    let positions = match model_type {
        ClipVisionModelType::Clip => 5,
        ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => 4,
    };
    let (pre_layer_norm_weight, pre_layer_norm_bias) = match model_type {
        ClipVisionModelType::Clip => (
            Some(filled_tensor(backend, &[4], 1.0, context)?),
            Some(filled_tensor(backend, &[4], 0.0, context)?),
        ),
        ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => (None, None),
    };
    let (llava_linear_1_weight, llava_linear_1_bias, llava_linear_2_weight, llava_linear_2_bias) =
        match model_type {
            ClipVisionModelType::Clip => (
                Some(identity_matrix(backend, 6, 4, context)?),
                Some(filled_tensor(backend, &[6], 0.0, context)?),
                Some(identity_matrix(backend, 6, 6, context)?),
                Some(filled_tensor(backend, &[6], 0.0, context)?),
            ),
            ClipVisionModelType::Siglip | ClipVisionModelType::Siglip2 => (None, None, None, None),
        };
    Ok(ClipVisionWeights {
        patch_embedding_weight: patch_weight,
        patch_embedding_bias: patch_bias,
        class_embedding,
        position_embedding: filled_tensor(backend, &[u64::try_from(positions)?, 4], 0.0, context)?,
        pre_layer_norm_weight,
        pre_layer_norm_bias,
        layers: vec![
            layer_weights(backend, context)?,
            layer_weights(backend, context)?,
        ],
        post_layer_norm_weight: filled_tensor(backend, &[4], 1.0, context)?,
        post_layer_norm_bias: filled_tensor(backend, &[4], 0.0, context)?,
        visual_projection_weight: Some(identity_matrix(backend, 3, 4, context)?),
        llava_linear_1_weight,
        llava_linear_1_bias,
        llava_linear_2_weight,
        llava_linear_2_bias,
    })
}

fn nchw_input(
    backend: &CpuBackend,
    height: usize,
    width: usize,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let count = 3 * height * width;
    let values = (0..count)
        .map(|index| index as f32 / count as f32)
        .collect::<Vec<_>>();
    tensor(
        backend,
        &[1, 3, u64::try_from(height)?, u64::try_from(width)?],
        &values,
        context,
    )
}

#[test]
fn source_identity_and_catalog_rows_are_pinned() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let source = fs::read(root.join(CLIP_VISION_SOURCE_PATH))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&source)),
        CLIP_VISION_SOURCE_SHA256
    );
    let source = String::from_utf8(source)?;
    assert_eq!(CLIP_VISION_CATALOG_SYMBOLS.len(), 9);
    for symbol in CLIP_VISION_CATALOG_SYMBOLS {
        assert!(source.contains(symbol), "missing source symbol {symbol}");
    }
    Ok(())
}

#[test]
fn adapter_delegates_foundational_ownership_without_absorbing_other_vision_models()
-> Result<(), Box<dyn Error>> {
    let source =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/clip_vision.rs"))?;
    let source = source
        .split_once("\n#[cfg(test)]\n")
        .map_or(source.as_str(), |(production, _)| production);
    assert!(source.contains("NativeExecutionRequirements"));
    assert!(source.contains("ExecutionContext"));
    assert!(source.contains("resize_with_context_exact_native"));
    assert!(source.matches("admit_backend_target(").count() >= 4);
    assert!(!source.contains("require_supported(backend.capabilities())"));
    assert!(source.contains("backend.binary("));
    for delegated in [
        "narrow_method_exact_native",
        "tensor_permute_exact_native",
        "contiguous_with_context_exact_native",
        "torch_reshape_with_context_exact_native",
        "torch_cat_with_context_exact_native",
        "torch_stack_with_context_exact_native",
        "canonical_quick_gelu",
        "normalize_with_context_exact_native",
    ] {
        assert!(
            source.contains(delegated),
            "CLIP vision adapter must delegate {delegated}"
        );
    }
    assert_eq!(
        source
            .lines()
            .filter(|line| line.trim() == "pub struct NativeClipVision {")
            .count(),
        1
    );
    for forbidden in [
        "struct CancellationToken",
        "struct CpuBackend",
        "struct ModelStore",
        "struct ArtifactIndex",
        "NativeEfficientNet",
        "NativeRaft",
        "TypedRoot",
        "std::path",
        "std::fs",
        "create_backend(",
        "authorize_workspace(",
        "tensor_to_f32_with_context_exact_native",
        "tensor_from_f32_with_context_exact_native",
        "CpuWorkspaceVec",
        "fn center_crop_nchw",
    ] {
        assert!(
            !source.contains(forbidden),
            "CLIP vision adapter absorbed forbidden owner {forbidden}"
        );
    }
    Ok(())
}

#[test]
fn standard_preprocess_truncates_alpha_quantizes_and_normalizes() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 1024 * 1024)?;
    let image = tensor(&backend, &[1, 1, 1, 4], &[0.1, 0.5, 1.2, 0.9], &context)?;
    let output =
        clip_preprocess_with_context(&backend, &image, 1, [0.0; 3], [1.0; 3], true, &context)?;
    assert_eq!(output.descriptor().shape(), &[1, 3, 1, 1]);
    let values = tensor_to_f32(&backend, &output, &context)?;
    assert_eq!(&*values, &[26.0 / 255.0, 128.0 / 255.0, 1.0]);
    Ok(())
}

#[test]
fn standard_preprocess_covers_crop_and_stretch_geometry() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 2 * 1024 * 1024)?;
    let values = (0..24).map(|value| value as f32 / 24.0).collect::<Vec<_>>();
    let image = tensor(&backend, &[1, 2, 3, 4], &values, &context)?;
    let cropped =
        clip_preprocess_with_context(&backend, &image, 4, [0.0; 3], [1.0; 3], true, &context)?;
    let stretched =
        clip_preprocess_with_context(&backend, &image, 4, [0.0; 3], [1.0; 3], false, &context)?;
    assert_eq!(cropped.descriptor().shape(), &[1, 3, 4, 4]);
    assert_eq!(stretched.descriptor().shape(), &[1, 3, 4, 4]);
    let cropped = tensor_to_f32(&backend, &cropped, &context)?;
    let stretched = tensor_to_f32(&backend, &stretched, &context)?;
    assert_ne!(&*cropped, &*stretched);
    Ok(())
}

#[test]
fn siglip2_flexible_preprocess_uses_source_resolution_search() -> Result<(), Box<dyn Error>> {
    assert_eq!(siglip2_flex_resolution(4, 8, 2, 8)?, (4, 8));
    let (height, width) = siglip2_flex_resolution(9, 17, 4, 6)?;
    assert_eq!(height % 4, 0);
    assert_eq!(width % 4, 0);
    assert!((height / 4) * (width / 4) <= 6);

    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 4 * 1024 * 1024)?;
    let values = vec![0.25; 9 * 17 * 3];
    let image = tensor(&backend, &[1, 9, 17, 3], &values, &context)?;
    let output = siglip2_preprocess_with_context(
        &backend, &image, 0, 4, 6, [0.5; 3], [0.5; 3], true, &context,
    )?;
    assert_eq!(
        output.descriptor().shape(),
        &[1, 3, u64::try_from(height)?, u64::try_from(width)?]
    );
    assert!(
        tensor_to_f32(&backend, &output, &context)?
            .iter()
            .all(|value| (*value + 0.498_039_22).abs() < 1.0e-6)
    );
    Ok(())
}

#[test]
fn preprocess_invalid_configuration_and_shape_fail_typed() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 4 * 1024 * 1024)?;
    let image = tensor(&backend, &[1, 2, 2, 3], &[0.5; 12], &context)?;

    assert!(matches!(
        clip_preprocess_with_context(&backend, &image, 0, [0.0; 3], [1.0; 3], true, &context,),
        Err(ClipVisionError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        clip_preprocess_with_context(
            &backend,
            &image,
            2,
            [0.0; 3],
            [1.0, 0.0, 1.0],
            true,
            &context,
        ),
        Err(ClipVisionError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        siglip2_preprocess_with_context(
            &backend, &image, 0, 0, 4, [0.0; 3], [1.0; 3], true, &context,
        ),
        Err(ClipVisionError::InvalidConfiguration(_))
    ));
    for dimensions in [(0, 2, 2, 3), (1, 0, 2, 3), (1, 2, 0, 3), (1, 2, 2, 2)] {
        let count = dimensions.0 * dimensions.1 * dimensions.2 * dimensions.3;
        let invalid = tensor(
            &backend,
            &[
                u64::try_from(dimensions.0)?,
                u64::try_from(dimensions.1)?,
                u64::try_from(dimensions.2)?,
                u64::try_from(dimensions.3)?,
            ],
            &vec![0.0; count],
            &context,
        )?;
        assert!(matches!(
            clip_preprocess_with_context(&backend, &invalid, 2, [0.0; 3], [1.0; 3], true, &context,),
            Err(ClipVisionError::InvalidInput(_))
        ));
    }
    assert!(matches!(
        siglip2_flex_resolution(0, 2, 2, 4),
        Err(ClipVisionError::InvalidInput(_))
    ));
    assert!(matches!(
        siglip2_flex_resolution(2, 2, 0, 4),
        Err(ClipVisionError::InvalidInput(_))
    ));
    assert!(matches!(
        siglip2_flex_resolution(2, 2, 2, 0),
        Err(ClipVisionError::InvalidInput(_))
    ));
    Ok(())
}

#[test]
fn clip_embeddings_pool_projection_intermediates_and_llava_match_source_shapes()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let input = nchw_input(&backend, 4, 4, &context)?;
    let mut baseline = NativeClipVision::new(
        configuration(ClipVisionModelType::Clip),
        weights(&backend, ClipVisionModelType::Clip, &context)?,
    )?;
    let baseline_output = baseline.forward(
        &backend,
        &input,
        ClipVisionIntermediate::Layer(-1),
        &context,
    )?;
    let mut model_weights = weights(&backend, ClipVisionModelType::Clip, &context)?;
    let first_layer = model_weights
        .layers
        .first_mut()
        .ok_or("missing first CLIP vision layer")?;
    first_layer.query_weight = identity_matrix(&backend, 4, 4, &context)?;
    first_layer.key_weight = identity_matrix(&backend, 4, 4, &context)?;
    first_layer.value_weight = identity_matrix(&backend, 4, 4, &context)?;
    first_layer.output_weight = identity_matrix(&backend, 4, 4, &context)?;
    first_layer.feed_forward_1_weight = identity_matrix(&backend, 8, 4, &context)?;
    first_layer.feed_forward_2_weight = identity_matrix(&backend, 4, 8, &context)?;
    let mut model = NativeClipVision::new(configuration(ClipVisionModelType::Clip), model_weights)?;
    assert_eq!(model.configuration().model_type, ClipVisionModelType::Clip);
    let bhwc = tensor(&backend, &[1, 4, 4, 3], &[0.5; 48], &context)?;
    assert_eq!(
        model
            .preprocess(&backend, &bhwc, true, &context)?
            .descriptor()
            .shape(),
        &[1, 3, 4, 4]
    );
    let output = model.forward(
        &backend,
        &input,
        ClipVisionIntermediate::Layer(-1),
        &context,
    )?;
    assert_eq!(output.last_hidden_state.descriptor().shape(), &[1, 5, 4]);
    assert_eq!(
        output
            .intermediate
            .as_ref()
            .map(|value| value.descriptor().shape()),
        Some(&[1, 5, 4][..])
    );
    assert_eq!(output.image_embeds.descriptor().shape(), &[1, 3]);
    assert_eq!(
        output
            .projected_intermediate
            .as_ref()
            .map(|value| value.descriptor().shape()),
        Some(&[1, 4, 6][..])
    );
    assert!(
        tensor_to_f32(&backend, &output.last_hidden_state, &context)?
            .iter()
            .any(|value| value.abs() > 1.0e-6)
    );
    assert_ne!(
        &*tensor_to_f32(&backend, &output.last_hidden_state, &context)?,
        &*tensor_to_f32(&backend, &baseline_output.last_hidden_state, &context)?,
        "nonzero attention and MLP weights must change the visual transformer result"
    );
    assert!(
        tensor_to_f32(&backend, &output.image_embeds, &context)?
            .iter()
            .any(|value| value.abs() > 1.0e-6)
    );
    assert!(
        tensor_to_f32(
            &backend,
            output
                .projected_intermediate
                .as_ref()
                .ok_or("missing Llava projection")?,
            &context,
        )?
        .iter()
        .any(|value| value.abs() > 1.0e-6)
    );
    Ok(())
}

#[test]
fn siglip_and_siglip2_execute_distinct_embedding_paths() -> Result<(), Box<dyn Error>> {
    for (model_type, height, width) in [
        (ClipVisionModelType::Siglip, 4, 4),
        (ClipVisionModelType::Siglip2, 2, 4),
    ] {
        let (backend, authority) = backend()?;
        let cancellation = CancellationToken::default();
        let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
        let mut model = NativeClipVision::new(
            configuration(model_type),
            weights(&backend, model_type, &context)?,
        )?;
        let input = nchw_input(&backend, height, width, &context)?;
        let output = model.forward(&backend, &input, ClipVisionIntermediate::All, &context)?;
        let patches = (height / 2) * (width / 2);
        assert_eq!(
            output.last_hidden_state.descriptor().shape(),
            &[1, u64::try_from(patches)?, 4]
        );
        assert_eq!(
            output
                .intermediate
                .as_ref()
                .map(|value| value.descriptor().shape()),
            Some(&[1, 2, u64::try_from(patches)?, 4][..])
        );
        assert_eq!(
            output.image_embeds.descriptor().shape(),
            &[1, u64::try_from(patches)?, 3]
        );
        assert!(output.projected_intermediate.is_none());
        let intermediate_values = tensor_to_f32(
            &backend,
            output
                .intermediate
                .as_ref()
                .ok_or("missing all-layer output")?,
            &context,
        )?;
        let expected = match model_type {
            ClipVisionModelType::Siglip => [0.0, 1.0 / 48.0, 4.0 / 48.0, 5.0 / 48.0],
            ClipVisionModelType::Siglip2 => [0.0, 8.0 / 24.0, 16.0 / 24.0, 1.0 / 24.0],
            ClipVisionModelType::Clip => unreachable!(),
        };
        for (actual, expected) in intermediate_values.iter().take(4).zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
    }
    Ok(())
}

#[test]
fn siglip2_position_resize_adds_nonzero_source_ordered_embeddings() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let mut model_weights = weights(&backend, ClipVisionModelType::Siglip2, &context)?;
    model_weights.patch_embedding_weight = filled_tensor(&backend, &[4, 12], 0.0, &context)?;
    let positions = (0..16).map(|value| value as f32 / 16.0).collect::<Vec<_>>();
    model_weights.position_embedding = tensor(&backend, &[4, 4], &positions, &context)?;
    let mut model =
        NativeClipVision::new(configuration(ClipVisionModelType::Siglip2), model_weights)?;
    let input = nchw_input(&backend, 2, 4, &context)?;
    let output = model.forward(&backend, &input, ClipVisionIntermediate::Layer(0), &context)?;
    let actual = tensor_to_f32(
        &backend,
        output.intermediate.as_ref().ok_or("missing intermediate")?,
        &context,
    )?;
    let expected = (4..12).map(|value| value as f32 / 16.0).collect::<Vec<_>>();
    assert_eq!(&actual[..8], expected.as_slice());
    assert_ne!(&actual[..8], &positions[..8]);
    Ok(())
}

#[test]
fn unsupported_dtypes_and_devices_fail_typed_without_relabel_or_substitution()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 24 * 1024 * 1024)?;

    for dtype in [DType::F16, DType::Bf16] {
        let mut unsupported = configuration(ClipVisionModelType::Clip);
        unsupported.dtype = dtype;
        let error = NativeClipVision::new(
            unsupported,
            weights(&backend, ClipVisionModelType::Clip, &context)?,
        )
        .expect_err("an unadvertised CPU dtype must fail before execution");
        assert!(matches!(
            error,
            ClipVisionError::UnsupportedTarget {
                dtype: actual,
                device: DeviceId::CPU,
            } if actual == dtype
        ));
    }

    let mut unsupported = configuration(ClipVisionModelType::Clip);
    unsupported.device = DeviceId::from_source_device("metal")?;
    let error = NativeClipVision::new(
        unsupported,
        weights(&backend, ClipVisionModelType::Clip, &context)?,
    )
    .expect_err("a CPU adapter must not relabel Metal execution");
    assert!(matches!(
        error,
        ClipVisionError::UnsupportedTarget {
            dtype: DType::F32,
            device
        } if device == DeviceId::from_source_device("metal")?
    ));
    Ok(())
}

#[test]
fn invalid_resolution_channel_intermediate_and_weight_shapes_fail_typed()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let mut invalid_weights = weights(&backend, ClipVisionModelType::Clip, &context)?;
    invalid_weights.position_embedding = filled_tensor(&backend, &[4, 4], 0.0, &context)?;
    assert!(
        NativeClipVision::new(configuration(ClipVisionModelType::Clip), invalid_weights,).is_err()
    );

    let mut invalid_fixed_siglip2 = configuration(ClipVisionModelType::Siglip2);
    invalid_fixed_siglip2.image_size = 3;
    assert!(matches!(
        NativeClipVision::new(
            invalid_fixed_siglip2,
            weights(&backend, ClipVisionModelType::Siglip2, &context)?,
        ),
        Err(ClipVisionError::InvalidConfiguration(_))
    ));

    let mut invalid_siglip2_positions = weights(&backend, ClipVisionModelType::Siglip2, &context)?;
    invalid_siglip2_positions.position_embedding = filled_tensor(&backend, &[3, 4], 0.0, &context)?;
    let mut invalid_siglip2_configuration = configuration(ClipVisionModelType::Siglip2);
    invalid_siglip2_configuration.max_num_patches = 3;
    assert!(matches!(
        NativeClipVision::new(invalid_siglip2_configuration, invalid_siglip2_positions,),
        Err(ClipVisionError::InvalidConfiguration(_))
    ));

    let mut invalid_projection = weights(&backend, ClipVisionModelType::Clip, &context)?;
    invalid_projection.visual_projection_weight =
        Some(filled_tensor(&backend, &[4, 4], 0.0, &context)?);
    assert!(
        NativeClipVision::new(configuration(ClipVisionModelType::Clip), invalid_projection,)
            .is_err()
    );

    let mut model = NativeClipVision::new(
        configuration(ClipVisionModelType::Clip),
        weights(&backend, ClipVisionModelType::Clip, &context)?,
    )?;
    let wrong_channels = tensor(&backend, &[1, 4, 4, 4], &vec![0.0; 64], &context)?;
    assert!(
        model
            .forward(
                &backend,
                &wrong_channels,
                ClipVisionIntermediate::None,
                &context,
            )
            .is_err()
    );
    let wrong_resolution = nchw_input(&backend, 2, 4, &context)?;
    assert!(
        model
            .forward(
                &backend,
                &wrong_resolution,
                ClipVisionIntermediate::None,
                &context,
            )
            .is_err()
    );
    let input = nchw_input(&backend, 4, 4, &context)?;
    assert!(
        model
            .forward(&backend, &input, ClipVisionIntermediate::Layer(2), &context,)
            .is_err()
    );
    assert!(matches!(
        model.forward(&backend, &input, ClipVisionIntermediate::None, &context,),
        Err(ClipVisionError::MissingLlavaIntermediate)
    ));
    let descriptor =
        TensorDescriptor::contiguous(vec![1, 3, 4, 4], DType::I64, DeviceId::CPU, context.stream)?;
    let integer_input = backend
        .upload_bytes(descriptor, &vec![0; 3 * 4 * 4 * 8], &context)?
        .0;
    assert!(
        model
            .forward(
                &backend,
                &integer_input,
                ClipVisionIntermediate::None,
                &context,
            )
            .is_err()
    );

    let zero_batch = tensor(&backend, &[0, 3, 4, 4], &[], &context)?;
    assert!(matches!(
        model.forward(
            &backend,
            &zero_batch,
            ClipVisionIntermediate::None,
            &context,
        ),
        Err(ClipVisionError::InvalidInput(_))
    ));

    let mut siglip2 = NativeClipVision::new(
        configuration(ClipVisionModelType::Siglip2),
        weights(&backend, ClipVisionModelType::Siglip2, &context)?,
    )?;
    let too_many_patches = nchw_input(&backend, 4, 6, &context)?;
    assert!(matches!(
        siglip2.forward(
            &backend,
            &too_many_patches,
            ClipVisionIntermediate::None,
            &context,
        ),
        Err(ClipVisionError::InvalidInput(_))
    ));
    Ok(())
}

#[test]
fn cancellation_and_workspace_oom_publish_nothing_and_converge() -> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let construction = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let mut model = NativeClipVision::new(
        configuration(ClipVisionModelType::Clip),
        weights(&backend, ClipVisionModelType::Clip, &construction)?,
    )?;
    let input = nchw_input(&backend, 4, 4, &construction)?;
    drop(construction);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&authority, &cancelled, 1024 * 1024)?;
    assert!(
        model
            .forward(
                &backend,
                &input,
                ClipVisionIntermediate::None,
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    drop(cancelled_context);

    let active = CancellationToken::default();
    let insufficient = context(&authority, &active, 16)?;
    assert!(
        model
            .forward(&backend, &input, ClipVisionIntermediate::All, &insufficient,)
            .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    drop(insufficient);

    let successful = context(&authority, &active, 16 * 1024 * 1024)?;
    model.forward(
        &backend,
        &input,
        ClipVisionIntermediate::Layer(0),
        &successful,
    )?;
    assert_eq!(successful.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn caller_context_construction_reconstruction_and_public_forward_are_identity_stable()
-> Result<(), Box<dyn Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let context = context(&authority, &cancellation, 16 * 1024 * 1024)?;
    let clip_configuration = configuration(ClipVisionModelType::Clip);
    let legacy = NativeClipVision::new(
        clip_configuration.clone(),
        weights(&backend, ClipVisionModelType::Clip, &context)?,
    )?;
    let cancellable = NativeClipVision::new_with_cancellation(
        clip_configuration,
        weights(&backend, ClipVisionModelType::Clip, &context)?,
        &cancellation,
    )?;
    assert_eq!(legacy.configuration(), cancellable.configuration());
    assert_eq!(
        legacy.semantic_digest_sha256(),
        cancellable.semantic_digest_sha256()
    );

    let reconstructed = cancellable.reconstruct(&cancellation)?;
    assert_eq!(
        reconstructed.semantic_digest_sha256(),
        cancellable.semantic_digest_sha256()
    );
    assert_eq!(
        reconstructed.resident_parts()?,
        cancellable.resident_parts()?
    );
    let input = nchw_input(&backend, 4, 4, &context)?;
    let mut original_execution = cancellable;
    let original_output = original_execution.forward(
        &backend,
        &input,
        ClipVisionIntermediate::Layer(-1),
        &context,
    )?;
    let mut reconstructed_execution = reconstructed;
    let reconstructed_output = reconstructed_execution.forward(
        &backend,
        &input,
        ClipVisionIntermediate::Layer(-1),
        &context,
    )?;
    assert_eq!(
        original_output.semantic_digest_sha256(),
        reconstructed_output.semantic_digest_sha256()
    );
    assert_eq!(
        original_output.image_embeds.contiguous_bytes()?,
        reconstructed_output.image_embeds.contiguous_bytes()?
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        NativeClipVision::new_with_cancellation(
            configuration(ClipVisionModelType::Clip),
            weights(&backend, ClipVisionModelType::Clip, &context)?,
            &cancelled,
        ),
        Err(ClipVisionError::Cancelled)
    ));
    assert!(matches!(
        original_execution.reconstruct(&cancelled),
        Err(ClipVisionError::Cancelled)
    ));

    let source =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/clip_vision.rs"))?;
    let production = source
        .split_once("\n#[cfg(test)]\n")
        .map_or(source.as_str(), |(production, _)| production);
    assert_eq!(
        production.matches("pub fn new_with_cancellation(").count(),
        1
    );
    assert_eq!(production.matches("pub fn reconstruct(").count(), 1);
    assert_eq!(
        production.matches("pub(crate) fn forward_checked(").count(),
        1
    );
    let checked_forward = production
        .split_once("pub(crate) fn forward_checked(")
        .and_then(|(_, tail)| tail.split_once("\n    fn patch_embeddings("))
        .map(|(checked, _)| checked)
        .ok_or("checked CLIP vision forward boundary is missing")?;
    assert_eq!(checked_forward.matches(".post_layer_norm").count(), 2);
    assert_eq!(
        checked_forward
            .matches("self.visual_projection.as_mut()")
            .count(),
        1
    );
    Ok(())
}

fn execute_valid_vision_contract(symbol: &str) -> Result<(), Box<dyn Error>> {
    match symbol {
        "clip_preprocess" => {
            standard_preprocess_truncates_alpha_quantizes_and_normalizes()?;
            standard_preprocess_covers_crop_and_stretch_geometry()?;
        }
        "siglip2_flex_calc_resolution" | "siglip2_preprocess" => {
            siglip2_flexible_preprocess_uses_source_resolution_search()?;
        }
        "siglip2_pos_embed" => {
            siglip2_position_resize_adds_nonzero_source_ordered_embeddings()?;
        }
        "Siglip2Embeddings" | "CLIPVisionEmbeddings" => {
            siglip_and_siglip2_execute_distinct_embedding_paths()?;
        }
        "CLIPVision" | "LlavaProjector" | "CLIPVisionModelProjection" => {
            clip_embeddings_pool_projection_intermediates_and_llava_match_source_shapes()?;
        }
        unexpected => return Err(format!("unaccounted CLIP vision symbol {unexpected}").into()),
    }
    Ok(())
}

fn execute_invalid_vision_contract(symbol: &str) -> Result<(), Box<dyn Error>> {
    if !CLIP_VISION_CATALOG_SYMBOLS.contains(&symbol) {
        return Err(format!("unaccounted CLIP vision symbol {symbol}").into());
    }
    if matches!(
        symbol,
        "clip_preprocess" | "siglip2_flex_calc_resolution" | "siglip2_preprocess"
    ) {
        preprocess_invalid_configuration_and_shape_fail_typed()?;
    }
    invalid_resolution_channel_intermediate_and_weight_shapes_fail_typed()?;
    unsupported_dtypes_and_devices_fail_typed_without_relabel_or_substitution()?;
    Ok(())
}

#[test]
fn val_clip_001_vision_rows_execute_and_extend_cumulative_ledger() -> Result<(), Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let catalog = fs::read_to_string(
        workspace.join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let mut contracts = Vec::new();
    let mut symbols = Vec::new();
    for line in catalog.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.get(8).copied() != Some(VISION_TASK) {
            continue;
        }
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[7], "sim_model::clip");
        assert_eq!(fields[9], "VAL-CLIP-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-CLIP-001");
        let source = fs::read(workspace.join(fields[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), fields[5]);
        assert_eq!(python_symbol_sha256(&source, fields[3])?, fields[6]);
        execute_valid_vision_contract(fields[3])?;
        execute_invalid_vision_contract(fields[3])?;
        symbols.push(fields[3]);
        contracts.push(json!({
            "contract_id": fields[0],
            "task_id": VISION_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": [
                format!("{}:native-vision-valid", fields[0]),
                format!("{}:native-vision-invalid", fields[0]),
            ],
        }));
    }
    assert_eq!(symbols, CLIP_VISION_CATALOG_SYMBOLS);
    assert_eq!(contracts.len(), CLIP_VISION_CATALOG_SYMBOLS.len());
    cancellation_and_workspace_oom_publish_nothing_and_converge()?;
    adapter_delegates_foundational_ownership_without_absorbing_other_vision_models()?;

    let artifact_path = workspace.join("target/sim-parity/val-clip-001.json");
    let mut artifact: Value = serde_json::from_slice(&fs::read(&artifact_path)?)?;
    assert_eq!(artifact.get("schema_version"), Some(&json!(1)));
    assert_eq!(artifact.get("validation_id"), Some(&json!("VAL-CLIP-001")));
    let task_results = artifact
        .get_mut("task_results")
        .and_then(Value::as_object_mut)
        .ok_or("VAL-CLIP-001 task results are missing")?;
    let implementations = VISION_IMPLEMENTATION_CLOSURE
        .iter()
        .map(|path| {
            Ok(json!({
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
            }))
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    task_results.insert(
        VISION_TASK.to_owned(),
        json!({
            "status": "passed",
            "passed": contracts.len(),
            "failed": 0,
            "skipped": 0,
            "case_ids": [
                "task340:source-provenance-and-nine-contracts",
                "task340:standard-and-flexible-preprocess",
                "task340:embeddings-and-position-resize",
                "task340:attention-pooling-and-projections",
                "task340:typed-target-and-shape-rejection",
                "task340:cancellation-oom-no-publication",
                "task340:ownership-consolidation",
            ],
            "implementations": implementations,
        }),
    );
    let passed = task_results.values().try_fold(0_u64, |total, result| {
        let count = result
            .get("passed")
            .and_then(Value::as_u64)
            .ok_or("VAL-CLIP-001 task result has no passed count")?;
        total
            .checked_add(count)
            .ok_or("VAL-CLIP-001 passed count overflowed")
    })?;
    let artifact_contracts = artifact
        .get_mut("contracts")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 contracts are missing")?;
    artifact_contracts
        .retain(|contract| contract.get("task_id").and_then(Value::as_str) != Some(VISION_TASK));
    artifact_contracts.extend(contracts);
    let remaining = artifact
        .get_mut("remaining_tasks")
        .and_then(Value::as_array_mut)
        .ok_or("VAL-CLIP-001 remaining tasks are missing")?;
    remaining.retain(|task| task.as_str() != Some(VISION_TASK));
    artifact["summary"] = json!({"passed": passed, "failed": 0, "skipped": 0});
    let producer_path = "crates/sim_model/tests/clip_vision.rs";
    artifact["implementation"] = json!({
        "path": producer_path,
        "sha256": format!(
            "{:x}",
            Sha256::digest(fs::read(workspace.join(producer_path))?)
        ),
    });
    fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
    Ok(())
}
