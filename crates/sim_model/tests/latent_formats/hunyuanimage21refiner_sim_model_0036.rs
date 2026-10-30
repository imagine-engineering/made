use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_hunyuanimage21refiner_sim_model_0036::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_hunyuanimage21refiner_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "HunyuanImage21Refiner");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0036");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 64);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.03682_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert!(LATENT_FORMAT.preview_factors.is_empty());
    assert_eq!(LATENT_FORMAT.preview_bias, None);
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(
        LATENT_FORMAT.transform,
        LatentTransform::HunyuanImage21Refiner
    );

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    assert_eq!(descriptor.channels, 64);
    assert_eq!(descriptor.dimensions, 3);
    assert_eq!(descriptor.scale_factor.to_bits(), 1.03682_f32.to_bits());
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0036","identifier":"HunyuanImage21Refiner"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn hunyuanimage21refiner_empty_latent_uses_batched_three_dimensional_geometry()
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
        LatentExtent::ThreeDimensional {
            batch: 3,
            frames: 5,
            width: 17,
            height: 9,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[3, 64, 5, 1, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);
    Ok(())
}

#[test]
fn hunyuanimage21refiner_custom_transform_matches_frame_channel_indexing_and_round_trips()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );

    let shape = [2, 64, 3, 1, 2];
    let input_values = (0..768)
        .map(|index| index as f32 / 17.0 - 11.0)
        .collect::<Vec<_>>();
    let input = tensor(&backend, &shape, &input_values, DType::F32, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    assert_eq!(model_facing.descriptor().shape(), &[2, 128, 2, 1, 2]);
    assert!(model_facing.descriptor().is_contiguous()?);
    assert_eq!(model_facing.descriptor().dtype(), DType::F32);
    assert_eq!(model_facing.descriptor().device(), backend.device());
    assert_eq!(model_facing.descriptor().stream(), StreamId::DEFAULT);

    let model_values = values(&model_facing)?;
    for batch in 0..2_usize {
        for channel in 0..128_usize {
            for frame in 0..2_usize {
                for position in 0..2_usize {
                    let duplicated_frame = frame * 2 + channel / 64;
                    let source_frame = duplicated_frame.saturating_sub(1);
                    let source_channel = channel % 64;
                    let source_index =
                        (((batch * 64 + source_channel) * 3 + source_frame) * 2) + position;
                    let model_index = (((batch * 128 + channel) * 2 + frame) * 2) + position;
                    let expected = input_values[source_index] * 1.03682_f32;
                    assert!((model_values[model_index] - expected).abs() <= 1.0e-6);
                }
            }
        }
    }

    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_ne!(restored.tensor_id(), model_facing.tensor_id());
    assert_ne!(restored.storage_id(), model_facing.storage_id());
    assert_eq!(restored.descriptor(), input.descriptor());
    for (actual, expected) in values(&restored)?.iter().zip(&input_values) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }

    let arbitrary_values = (0..256)
        .map(|index| index as f32 / 9.0 - 7.0)
        .collect::<Vec<_>>();
    let arbitrary_model = tensor(
        &backend,
        &[1, 128, 2, 1, 1],
        &arbitrary_values,
        DType::F32,
        &context,
    )?;
    let unpacked = process_latent_out(&LATENT_FORMAT, &backend, &arbitrary_model, &context)?;
    assert_eq!(unpacked.descriptor().shape(), &[1, 64, 3, 1, 1]);
    let unpacked_values = values(&unpacked)?;
    for channel in 0..64_usize {
        for frame in 0..3_usize {
            let expanded_frame = frame + 1;
            let source_channel = channel + (expanded_frame % 2) * 64;
            let source_frame = expanded_frame / 2;
            let source_index = source_channel * 2 + source_frame;
            let output_index = channel * 3 + frame;
            let expected = arbitrary_values[source_index] / 1.03682_f32;
            assert!((unpacked_values[output_index] - expected).abs() <= 1.0e-6);
        }
    }
    Ok(())
}

#[test]
fn hunyuanimage21refiner_rejects_invalid_geometry_shape_dtype_stream_preview_and_cancellation()
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
                frames: 0,
                width: 8,
                height: 8,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent {
            identifier,
            field: "frames",
            value: 0,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let wrong_rank = tensor(&backend, &[1, 64, 3, 1], &[], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_rank, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let wrong_channels = tensor(&backend, &[1, 63, 3, 1, 1], &[], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let even_frames = tensor(&backend, &[1, 64, 2, 1, 1], &[], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &even_frames, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let wrong_model_channels = tensor(&backend, &[1, 64, 2, 1, 1], &[], DType::F32, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &wrong_model_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let f16 = tensor(&backend, &[1, 64, 1, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(36);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_stream_input = tensor(
        &backend,
        &[1, 64, 1, 1, 1],
        &[],
        DType::F32,
        &foreign_context,
    )?;
    assert!(matches!(
        process_latent_in(
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
        &[1, 64, 1, 1, 1],
        &[0.0; 64],
        DType::F32,
        &context,
    )?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::PreviewUnavailable { identifier })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    cancellation.cancel();
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &input, &context),
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
    if dtype == DType::F32 && !values.is_empty() {
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
