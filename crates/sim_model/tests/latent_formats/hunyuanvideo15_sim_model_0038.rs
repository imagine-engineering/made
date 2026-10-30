use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_hunyuanvideo15_sim_model_0038::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_hunyuanvideo15_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "HunyuanVideo15");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0038");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 32);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 16);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 4);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.03682_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.preview_factors.len(), 32);
    assert_eq!(LATENT_FORMAT.preview_factors[0], [0.0568, -0.0521, -0.0131]);
    assert_eq!(LATENT_FORMAT.preview_factors[31], [0.0005, -0.0106, 0.0242]);
    assert_eq!(LATENT_FORMAT.preview_bias, Some([0.0456, -0.0202, -0.0644]));
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, Some("lighttaehy1_5"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Affine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0038","identifier":"HunyuanVideo15"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn hunyuanvideo15_geometry_affine_round_trip_and_preview_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );

    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional {
            batch: 2,
            frames: 9,
            width: 31,
            height: 17,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 32, 3, 1, 1]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);

    let source_values = (0..64)
        .map(|index| index as f32 / 17.0 - 1.5)
        .collect::<Vec<_>>();
    let input = tensor(&backend, &[2, 32, 1, 1, 1], &source_values, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.descriptor(), input.descriptor());
    for (actual, source) in values(&model_facing)?.iter().zip(&source_values) {
        assert!((actual - source * 1.03682_f32).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.descriptor(), input.descriptor());
    for (actual, expected) in values(&restored)?.iter().zip(&source_values) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }

    let preview_input = tensor(
        &backend,
        &[1, 32, 1, 1, 1],
        &(0..32).map(|index| index as f32 / 10.0).collect::<Vec<_>>(),
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 1, 1]);
    let preview_values = values(&preview)?;
    let preview_bias = LATENT_FORMAT
        .preview_bias
        .ok_or("HunyuanVideo15 preview bias is required")?;
    for (color, actual) in preview_values.iter().enumerate() {
        let expected = LATENT_FORMAT.preview_factors.iter().enumerate().fold(
            preview_bias[color],
            |value, (channel, factors)| {
                (channel as f32 / 10.0).mul_add(factors[color], value)
            },
        );
        assert!((actual - expected).abs() <= 1.0e-5);
    }
    Ok(())
}

#[test]
fn hunyuanvideo15_rejects_extent_shape_dtype_stream_and_cancellation()
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
            LatentExtent::TwoDimensional { batch: 1, width: 8, height: 8 },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions { expected: 3, actual: 2, .. })
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::ThreeDimensional { batch: 1, frames: 0, width: 8, height: 8 },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent { field: "frames", value: 0, .. })
    ));
    let wrong_channels = tensor(&backend, &[1, 31, 1, 1, 1], &[0.0; 31], &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { .. })
    ));

    let f16 = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional { batch: 1, frames: 1, width: 16, height: 16 },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(38);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_input = tensor(&backend, &[1, 32, 1, 1, 1], &[0.0; 32], &foreign_context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &foreign_input, &context),
        Err(LatentFormatError::Tensor(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        })) if actual == foreign_stream
    ));

    let input = tensor(&backend, &[1, 32, 1, 1, 1], &[0.0; 32], &context)?;
    cancellation.cancel();
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}

fn tensor(
    backend: &impl TensorBackend,
    shape: &[u64],
    values: &[f32],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (mut tensor, _) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    let mut write = tensor.write()?;
    let bytes = write.bytes_mut()?;
    if bytes.len() != std::mem::size_of_val(values) {
        return Err("fixture tensor length mismatch".into());
    }
    for (destination, value) in bytes.chunks_exact_mut(4).zip(values) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
    drop(write);
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
