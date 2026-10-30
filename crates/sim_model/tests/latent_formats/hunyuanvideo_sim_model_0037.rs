use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentFormatRegistry, LatentTensorLayout, LatentTransform,
    PreviewReshape, empty_latent,
    generated_hunyuanvideo_sim_model_0037::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

static REGISTRY: [sim_model::LatentFormatDefinition; 1] = [LATENT_FORMAT];

#[test]
fn val_latent_001_hunyuanvideo_exact_contract_and_stable_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "HunyuanVideo");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0037");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 16);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 4);
    assert_eq!(
        LATENT_FORMAT.scale_factor.to_bits(),
        0.476_986_f32.to_bits()
    );
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(
        LATENT_FORMAT.preview_factors,
        &[
            [-0.0395, -0.0331, 0.0445],
            [0.0696, 0.0795, 0.0518],
            [0.0135, -0.0945, -0.0282],
            [0.0108, -0.0250, -0.0765],
            [-0.0209, 0.0032, 0.0224],
            [-0.0804, -0.0254, -0.0639],
            [-0.0991, 0.0271, -0.0669],
            [-0.0646, -0.0422, -0.0400],
            [-0.0696, -0.0595, -0.0894],
            [-0.0799, -0.0208, -0.0375],
            [0.1166, 0.1627, 0.0962],
            [0.1165, 0.0432, 0.0407],
            [-0.2315, -0.1920, -0.1355],
            [-0.0270, 0.0401, -0.0821],
            [-0.0616, -0.0997, -0.0727],
            [0.0249, -0.0469, -0.1703],
        ]
    );
    assert_eq!(LATENT_FORMAT.preview_bias, Some([0.0259, -0.0192, -0.0761]));
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, Some("taehv"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Affine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    assert_eq!(descriptor.channels, 16);
    assert_eq!(descriptor.dimensions, 3);
    assert_eq!(descriptor.preview_factors, LATENT_FORMAT.preview_factors);
    assert_eq!(descriptor.preview_bias, LATENT_FORMAT.preview_bias);
    assert_eq!(descriptor.decoder_name.as_deref(), Some("taehv"));
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0037","identifier":"HunyuanVideo"}}"#
        )
    );
    let identity: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(identity, descriptor.identity);
    let registry = LatentFormatRegistry::checked(&REGISTRY)?;
    assert_eq!(
        registry.get(&identity).map(|format| format.identifier),
        Some(LATENT_FORMAT_IDENTIFIER)
    );
    Ok(())
}

#[test]
fn hunyuanvideo_batched_geometry_allocating_transform_round_trip_and_preview()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let stream = StreamId::new(37);
    let context =
        backend.execution_context(stream, authority.authorize_workspace(0)?, &cancellation);

    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional {
            batch: 3,
            frames: 5,
            width: 23,
            height: 17,
        },
        DType::F16,
        stream,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[3, 16, 2, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), stream);

    let input_values = (0..64)
        .map(|value| value as f32 / 9.0 - 3.0)
        .collect::<Vec<_>>();
    let input = tensor(
        &backend,
        &[2, 16, 2, 1, 1],
        &input_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.descriptor(), input.descriptor());
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    for (actual, source) in values(&model_facing)?.iter().zip(&input_values) {
        assert!((actual - source * 0.476_986_f32).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.descriptor(), input.descriptor());
    assert_ne!(restored.tensor_id(), model_facing.tensor_id());
    assert_ne!(restored.storage_id(), model_facing.storage_id());
    for (actual, expected) in values(&restored)?.iter().zip(&input_values) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }

    let mut preview_values = vec![0.0; 32];
    preview_values[0] = 2.0;
    preview_values[31] = -3.0;
    let preview_input = tensor(
        &backend,
        &[2, 16, 1, 1, 1],
        &preview_values,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[2, 3, 1, 1, 1]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), stream);
    let expected = [
        0.0259 + 2.0 * -0.0395,
        -0.0192 + 2.0 * -0.0331,
        -0.0761 + 2.0 * 0.0445,
        0.0259 - 3.0 * 0.0249,
        -0.0192 - 3.0 * -0.0469,
        -0.0761 - 3.0 * -0.1703,
    ];
    for (actual, expected) in values(&preview)?.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn hunyuanvideo_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );

    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::TwoDimensional {
                batch: 1,
                width: 8,
                height: 8,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions {
            identifier,
            expected: 3,
            actual: 2,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::ThreeDimensional {
                batch: 1,
                frames: 1,
                width: 7,
                height: 8,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent {
            identifier,
            field: "width",
            value: 7,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let wrong_channels = tensor(
        &backend,
        &[1, 15, 1, 1, 1],
        &[0.0; 15],
        DType::F32,
        &context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let wrong_rank = tensor(&backend, &[1, 16, 1, 1], &[0.0; 16], DType::F32, &context)?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &wrong_rank, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let f16 = tensor(&backend, &[1, 16, 1, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let alternate_stream = StreamId::new(38);
    let alternate_context = backend.execution_context(
        alternate_stream,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let alternate_input = tensor(
        &backend,
        &[1, 16, 1, 1, 1],
        &[0.0; 16],
        DType::F32,
        &alternate_context,
    )?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &alternate_input, &context),
        Err(LatentFormatError::Tensor(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        })) if actual == alternate_stream
    ));

    let input = tensor(
        &backend,
        &[1, 16, 1, 1, 1],
        &[0.0; 16],
        DType::F32,
        &context,
    )?;
    cancellation.cancel();
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::ThreeDimensional {
                batch: 1,
                frames: 1,
                width: 8,
                height: 8,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}

fn tensor(
    backend: &impl TensorBackend,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, backend.device(), context.stream)?;
    let (mut tensor, _) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    if dtype == DType::F32 {
        let mut write = tensor.write()?;
        let bytes = write.bytes_mut()?;
        if bytes.len() != std::mem::size_of_val(values) {
            return Err("fixture tensor length mismatch".into());
        }
        for (destination, value) in bytes.chunks_exact_mut(4).zip(values) {
            destination.copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(tensor)
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let count = tensor.descriptor().element_count()?;
    let mut values = Vec::with_capacity(usize::try_from(count)?);
    for index in 0..count {
        values.push(f32::from_le_bytes(
            tensor.linear_element_bytes(index)?.try_into()?,
        ));
    }
    Ok(values)
}
