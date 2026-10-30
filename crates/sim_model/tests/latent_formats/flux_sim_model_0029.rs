use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_flux_sim_model_0029::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

const EXPECTED_PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [-0.0346, 0.0244, 0.0681],
    [0.0034, 0.0210, 0.0687],
    [0.0275, -0.0668, -0.0433],
    [-0.0174, 0.0160, 0.0617],
    [0.0859, 0.0721, 0.0329],
    [0.0004, 0.0383, 0.0115],
    [0.0405, 0.0861, 0.0915],
    [-0.0236, -0.0185, -0.0259],
    [-0.0245, 0.0250, 0.1180],
    [0.1008, 0.0755, -0.0421],
    [-0.0515, 0.0201, 0.0011],
    [0.0428, -0.0012, -0.0036],
    [0.0817, 0.0765, 0.0749],
    [-0.1264, -0.0522, -0.1103],
    [-0.0280, -0.0881, -0.0499],
    [-0.1262, -0.0982, -0.0778],
];

#[test]
fn val_latent_001_flux_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Flux");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0029");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 16);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 0.3611_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.1159_f32.to_bits());
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(LATENT_FORMAT.preview_factors, &EXPECTED_PREVIEW_FACTORS);
    assert_eq!(
        LATENT_FORMAT.preview_bias,
        Some([-0.0329, -0.0718, -0.0851])
    );
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, Some("taef1_decoder"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Affine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    assert_eq!(descriptor.preview_factors, EXPECTED_PREVIEW_FACTORS);
    assert_eq!(descriptor.preview_bias, Some([-0.0329, -0.0718, -0.0851]));
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0029","identifier":"Flux"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn flux_geometry_batching_affine_round_trip_and_preview_are_exact()
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
            width: 17,
            height: 24,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 16, 3, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);

    let input_values = (0..32)
        .map(|index| index as f32 / 7.0 - 2.0)
        .collect::<Vec<_>>();
    let input = tensor(
        &backend,
        &[2, 16, 1, 1],
        &input_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.descriptor().shape(), &[2, 16, 1, 1]);
    assert_eq!(model_facing.descriptor().dtype(), DType::F32);
    assert_eq!(model_facing.descriptor().device(), backend.device());
    assert_eq!(model_facing.descriptor().stream(), StreamId::DEFAULT);
    for (actual, source) in values(&model_facing)?.iter().zip(&input_values) {
        let expected = (*source - 0.1159_f32) * 0.3611_f32;
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    for (actual, expected) in values(&restored)?.iter().zip(&input_values) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }

    let mut preview_input_values = [0.0_f32; 16];
    preview_input_values[0] = 2.0;
    preview_input_values[9] = -1.5;
    let preview_input = tensor(
        &backend,
        &[1, 16, 1, 1],
        &preview_input_values,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 1]);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), StreamId::DEFAULT);
    let preview_values = values(&preview)?;
    let expected_preview = [-0.2533_f32, -0.13625_f32, 0.11425_f32];
    for (actual, expected) in preview_values.iter().zip(expected_preview) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn flux_rejects_invalid_extents_shapes_dtype_stream_and_cancellation()
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

    let wrong_channels = tensor(&backend, &[1, 15, 1, 1], &[0.0; 15], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let f16 = tensor(&backend, &[1, 16, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(7);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_stream_input = tensor(
        &backend,
        &[1, 16, 1, 1],
        &[0.0; 16],
        DType::F32,
        &foreign_context,
    )?;
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

    let input = tensor(&backend, &[1, 16, 1, 1], &[0.0; 16], DType::F32, &context)?;
    cancellation.cancel();
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &input, &context),
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
