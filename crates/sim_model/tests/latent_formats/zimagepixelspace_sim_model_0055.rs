use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_zimagepixelspace_sim_model_0055::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_zimage_pixel_space_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "ZImagePixelSpace");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0055");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 3);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(
        LATENT_FORMAT.preview_factors,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    assert_eq!(LATENT_FORMAT.preview_bias, None);
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Identity);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0055","identifier":"ZImagePixelSpace"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn zimage_pixel_space_preserves_alias_identity_and_projects_rgb()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let stream = StreamId::new(55);
    let context =
        backend.execution_context(stream, authority.authorize_workspace(0)?, &cancellation);
    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::TwoDimensional {
            batch: 2,
            width: 23,
            height: 17,
        },
        DType::F16,
        stream,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 3, 17, 23]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), stream);

    let f16_input = tensor(&backend, &[1, 3, 1, 1], &[], DType::F16, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &f16_input, &context)?;
    assert_eq!(model_facing.tensor_id(), f16_input.tensor_id());
    assert_eq!(model_facing.storage_id(), f16_input.storage_id());
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.tensor_id(), f16_input.tensor_id());
    assert_eq!(restored.storage_id(), f16_input.storage_id());

    let source = [0.25, -0.5, 1.75, 2.0, 3.0, -4.0];
    let preview_input = tensor(&backend, &[1, 3, 1, 2], &source, DType::F32, &context)?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 2]);
    assert_eq!(values(&preview)?, source);
    Ok(())
}

#[test]
fn zimage_pixel_space_rejects_invalid_extent_shape_stream_and_cancellation()
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
            LatentExtent::ThreeDimensional {
                batch: 1,
                frames: 1,
                width: 1,
                height: 1
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions {
            expected: 2,
            actual: 3,
            ..
        })
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::TwoDimensional {
                batch: 1,
                width: 0,
                height: 1
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent {
            field: "width",
            value: 0,
            ..
        })
    ));
    let wrong_channels = tensor(&backend, &[1, 2, 1, 1], &[0.0; 2], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { .. })
    ));
    let foreign_stream = StreamId::new(55);
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let foreign_input = tensor(
        &backend,
        &[1, 3, 1, 1],
        &[0.0; 3],
        DType::F32,
        &foreign_context,
    )?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &foreign_input, &context),
        Err(LatentFormatError::Tensor(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        })) if actual == foreign_stream
    ));
    let input = tensor(&backend, &[1, 3, 1, 1], &[0.0; 3], DType::F32, &context)?;
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
