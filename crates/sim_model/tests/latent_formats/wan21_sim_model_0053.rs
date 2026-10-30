use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_wan21_sim_model_0053::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

const SOURCE_MEANS: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const SOURCE_STDS: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];
const SOURCE_PREVIEW_FACTORS: [[f32; 3]; 16] = [
    [-0.1299, -0.1692, 0.2932],
    [0.0671, 0.0406, 0.0442],
    [0.3568, 0.2548, 0.1747],
    [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328],
    [0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364],
    [-0.1293, 0.0740, 0.1636],
    [0.0680, 0.3019, 0.1128],
    [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699],
    [0.0060, -0.0633, 0.0005],
    [0.3477, 0.2275, 0.2950],
    [0.1984, 0.0913, 0.1861],
];
const SOURCE_PREVIEW_BIAS: [f32; 3] = [-0.1835, -0.0868, -0.3360];

#[test]
fn val_latent_001_wan21_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Wan21");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0053");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 16);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 4);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.channel_means, &SOURCE_MEANS);
    assert_eq!(LATENT_FORMAT.channel_stds, &SOURCE_STDS);
    assert_eq!(LATENT_FORMAT.preview_factors, &SOURCE_PREVIEW_FACTORS);
    assert_eq!(LATENT_FORMAT.preview_bias, Some(SOURCE_PREVIEW_BIAS));
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, Some("lighttaew2_1"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::PerChannelAffine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0053","identifier":"Wan21"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn wan21_geometry_per_channel_round_trip_and_preview_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let stream = StreamId::new(53);
    let context =
        backend.execution_context(stream, authority.authorize_workspace(0)?, &cancellation);
    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional {
            batch: 2,
            frames: 5,
            width: 23,
            height: 17,
        },
        DType::F16,
        stream,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 16, 2, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), stream);

    let source = (0..16)
        .map(|index| index as f32 / 5.0 - 1.25)
        .collect::<Vec<_>>();
    let input = tensor(&backend, &[1, 16, 1, 1, 1], &source, DType::F32, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    for (channel, actual) in values(&model_facing)?.iter().enumerate() {
        let expected = (source[channel] - SOURCE_MEANS[channel]) / SOURCE_STDS[channel];
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    for (actual, expected) in values(&restored)?.iter().zip(&source) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }

    let mut preview_source = [0.0_f32; 16];
    preview_source[0] = 2.0;
    preview_source[15] = -3.0;
    let preview_input = tensor(
        &backend,
        &[1, 16, 1, 1, 1],
        &preview_source,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 1, 1]);
    let expected = [
        (-3.0_f32).mul_add(
            SOURCE_PREVIEW_FACTORS[15][0],
            2.0_f32.mul_add(SOURCE_PREVIEW_FACTORS[0][0], SOURCE_PREVIEW_BIAS[0]),
        ),
        (-3.0_f32).mul_add(
            SOURCE_PREVIEW_FACTORS[15][1],
            2.0_f32.mul_add(SOURCE_PREVIEW_FACTORS[0][1], SOURCE_PREVIEW_BIAS[1]),
        ),
        (-3.0_f32).mul_add(
            SOURCE_PREVIEW_FACTORS[15][2],
            2.0_f32.mul_add(SOURCE_PREVIEW_FACTORS[0][2], SOURCE_PREVIEW_BIAS[2]),
        ),
    ];
    assert_eq!(values(&preview)?, expected);
    Ok(())
}

#[test]
fn wan21_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    validate_failure_contract(53, 16, 8)
}

fn validate_failure_contract(
    stream_value: u64,
    channels: u64,
    spatial_ratio: u64,
) -> Result<(), Box<dyn std::error::Error>> {
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
                width: spatial_ratio,
                height: spatial_ratio
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::ThreeDimensional {
                batch: 1,
                frames: 1,
                width: spatial_ratio - 1,
                height: spatial_ratio,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent { field: "width", .. })
    ));
    let wrong_channels = tensor(
        &backend,
        &[1, channels - 1, 1, 1, 1],
        &vec![0.0; usize::try_from(channels - 1)?],
        DType::F32,
        &context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { .. })
    ));
    let f16 = tensor(&backend, &[1, channels, 1, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));
    let foreign_stream = StreamId::new(stream_value);
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let foreign_input = tensor(
        &backend,
        &[1, channels, 1, 1, 1],
        &vec![0.0; usize::try_from(channels)?],
        DType::F32,
        &foreign_context,
    )?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &foreign_input, &context),
        Err(LatentFormatError::Tensor(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        })) if actual == foreign_stream
    ));
    let input = tensor(
        &backend,
        &[1, channels, 1, 1, 1],
        &vec![0.0; usize::try_from(channels)?],
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
