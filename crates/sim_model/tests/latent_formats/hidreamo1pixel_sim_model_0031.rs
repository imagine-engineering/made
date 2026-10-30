use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_hidreamo1pixel_sim_model_0031::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

const EXPECTED_PREVIEW_FACTORS: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[test]
fn val_latent_001_hidreamo1pixel_exact_inherited_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "HiDreamO1Pixel");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0031");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 3);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(LATENT_FORMAT.preview_factors, &EXPECTED_PREVIEW_FACTORS);
    assert_eq!(LATENT_FORMAT.preview_bias, None);
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Identity);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    assert_eq!(descriptor.preview_factors, EXPECTED_PREVIEW_FACTORS);
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0031","identifier":"HiDreamO1Pixel"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn hidreamo1pixel_geometry_batching_f16_f32_alias_and_preview_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );

    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::TwoDimensional {
            batch: 2,
            width: 5,
            height: 3,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 3, 3, 5]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);
    let model_facing_f16 = process_latent_in(&LATENT_FORMAT, &backend, &empty, &context)?;
    assert_eq!(model_facing_f16.tensor_id(), empty.tensor_id());
    assert_eq!(model_facing_f16.storage_id(), empty.storage_id());
    assert_eq!(model_facing_f16.descriptor(), empty.descriptor());
    let restored_f16 = process_latent_out(&LATENT_FORMAT, &backend, &model_facing_f16, &context)?;
    assert_eq!(restored_f16.tensor_id(), empty.tensor_id());
    assert_eq!(restored_f16.storage_id(), empty.storage_id());

    let source_values = [
        1.0, 2.0, 10.0, 20.0, 100.0, 200.0, -1.0, -2.0, -10.0, -20.0, -100.0, -200.0,
    ];
    let input = tensor(
        &backend,
        &[2, 3, 1, 2],
        &source_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.tensor_id(), input.tensor_id());
    assert_eq!(model_facing.storage_id(), input.storage_id());
    assert_eq!(model_facing.descriptor(), input.descriptor());
    assert_eq!(values(&model_facing)?, source_values);
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.tensor_id(), input.tensor_id());
    assert_eq!(restored.storage_id(), input.storage_id());
    assert_eq!(values(&restored)?, source_values);

    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[2, 3, 1, 2]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), StreamId::DEFAULT);
    assert_eq!(values(&preview)?, source_values);
    Ok(())
}

#[test]
fn hidreamo1pixel_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
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
            LatentExtent::OneDimensional {
                batch: 1,
                length: 8,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions {
            identifier,
            expected: 2,
            actual: 1,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::TwoDimensional {
                batch: 1,
                width: 0,
                height: 8,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent {
            identifier,
            field: "width",
            value: 0,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let wrong_channels = tensor(&backend, &[1, 4, 1, 1], &[0.0; 4], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let wrong_rank = tensor(&backend, &[1, 3, 1], &[0.0; 3], DType::F32, &context)?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &wrong_rank, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let f16 = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::TwoDimensional {
            batch: 1,
            width: 1,
            height: 1,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(31);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_stream_input = tensor(
        &backend,
        &[1, 3, 1, 1],
        &[1.0, 2.0, 3.0],
        DType::F32,
        &foreign_context,
    )?;
    assert_eq!(foreign_stream_input.descriptor().device(), backend.device());
    assert!(matches!(
        process_latent_out(
            &LATENT_FORMAT,
            &backend,
            &foreign_stream_input,
            &context,
        ),
        Err(LatentFormatError::Tensor(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        })) if actual == foreign_stream
    ));

    let input = tensor(
        &backend,
        &[1, 3, 1, 1],
        &[1.0, 2.0, 3.0],
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
