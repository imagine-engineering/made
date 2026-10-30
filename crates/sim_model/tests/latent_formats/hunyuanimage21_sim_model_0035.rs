use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_hunyuanimage21_sim_model_0035::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

const EXPECTED_PREVIEW_FACTORS: [[f32; 3]; 64] = [
    [-0.0154, -0.0397, -0.0521],
    [0.0005, 0.0093, 0.0006],
    [-0.0805, -0.0773, -0.0586],
    [-0.0494, -0.0487, -0.0498],
    [-0.0212, -0.0076, -0.0261],
    [-0.0179, -0.0417, -0.0505],
    [0.0158, 0.0310, 0.0239],
    [0.0409, 0.0516, 0.0201],
    [0.0350, 0.0553, 0.0036],
    [-0.0447, -0.0327, -0.0479],
    [-0.0038, -0.0221, -0.0365],
    [-0.0423, -0.0718, -0.0654],
    [0.0039, 0.0368, 0.0104],
    [0.0655, 0.0217, 0.0122],
    [0.0490, 0.1638, 0.2053],
    [0.0932, 0.0829, 0.0650],
    [-0.0186, -0.0209, -0.0135],
    [-0.0080, -0.0076, -0.0148],
    [-0.0284, -0.0201, 0.0011],
    [-0.0642, -0.0294, -0.0777],
    [-0.0035, 0.0076, -0.0140],
    [0.0519, 0.0731, 0.0887],
    [-0.0102, 0.0095, 0.0704],
    [0.0068, 0.0218, -0.0023],
    [-0.0726, -0.0486, -0.0519],
    [0.0260, 0.0295, 0.0263],
    [0.0250, 0.0333, 0.0341],
    [0.0168, -0.0120, -0.0174],
    [0.0226, 0.1037, 0.0114],
    [0.2577, 0.1906, 0.1604],
    [-0.0646, -0.0137, -0.0018],
    [-0.0112, 0.0309, 0.0358],
    [-0.0347, 0.0146, -0.0481],
    [0.0234, 0.0179, 0.0201],
    [0.0157, 0.0313, 0.0225],
    [0.0423, 0.0675, 0.0524],
    [-0.0031, 0.0027, -0.0255],
    [0.0447, 0.0555, 0.0330],
    [-0.0152, 0.0103, 0.0299],
    [-0.0755, -0.0489, -0.0635],
    [0.0853, 0.0788, 0.1017],
    [-0.0272, -0.0294, -0.0471],
    [0.0440, 0.0400, -0.0137],
    [0.0335, 0.0317, -0.0036],
    [-0.0344, -0.0621, -0.0984],
    [-0.0127, -0.0630, -0.0620],
    [-0.0648, 0.0360, 0.0924],
    [-0.0781, -0.0801, -0.0409],
    [0.0363, 0.0613, 0.0499],
    [0.0238, 0.0034, 0.0041],
    [-0.0135, 0.0258, 0.0310],
    [0.0614, 0.1086, 0.0589],
    [0.0428, 0.0350, 0.0205],
    [0.0153, 0.0173, -0.0018],
    [-0.0288, -0.0455, -0.0091],
    [0.0344, 0.0109, -0.0157],
    [-0.0205, -0.0247, -0.0187],
    [0.0487, 0.0126, 0.0064],
    [-0.0220, -0.0013, 0.0074],
    [-0.0203, -0.0094, -0.0048],
    [-0.0719, 0.0429, -0.0442],
    [0.1042, 0.0497, 0.0356],
    [-0.0659, -0.0578, -0.0280],
    [-0.0060, -0.0322, -0.0234],
];

#[test]
fn val_latent_001_hunyuanimage21_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "HunyuanImage21");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0035");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 64);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 32);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 0.75289_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(LATENT_FORMAT.preview_factors, &EXPECTED_PREVIEW_FACTORS);
    assert_eq!(LATENT_FORMAT.preview_bias, Some([0.0007, -0.0256, -0.0206]));
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Affine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    assert_eq!(descriptor.preview_factors, EXPECTED_PREVIEW_FACTORS);
    assert_eq!(descriptor.preview_bias, Some([0.0007, -0.0256, -0.0206]));
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0035","identifier":"HunyuanImage21"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn hunyuanimage21_geometry_batching_affine_round_trip_and_preview_are_exact()
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
            width: 65,
            height: 97,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 64, 3, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);

    let input_values = (0..128)
        .map(|index| index as f32 / 13.0 - 4.0)
        .collect::<Vec<_>>();
    let input = tensor(
        &backend,
        &[2, 64, 1, 1],
        &input_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    assert_eq!(model_facing.descriptor().shape(), &[2, 64, 1, 1]);
    assert_eq!(model_facing.descriptor().dtype(), DType::F32);
    assert_eq!(model_facing.descriptor().device(), backend.device());
    assert_eq!(model_facing.descriptor().stream(), StreamId::DEFAULT);
    for (actual, source) in values(&model_facing)?.iter().zip(&input_values) {
        assert!((actual - source * 0.75289_f32).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_ne!(restored.tensor_id(), model_facing.tensor_id());
    assert_ne!(restored.storage_id(), model_facing.storage_id());
    for (actual, expected) in values(&restored)?.iter().zip(&input_values) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }

    let mut preview_input_values = [0.0_f32; 64];
    preview_input_values[0] = 2.0;
    preview_input_values[29] = -1.5;
    preview_input_values[61] = 0.25;
    let preview_input = tensor(
        &backend,
        &[1, 64, 1, 1],
        &preview_input_values,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 1]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), StreamId::DEFAULT);
    let expected_preview = [-0.3906_f32, -0.378475_f32, -0.3565_f32];
    for (actual, expected) in values(&preview)?.iter().zip(expected_preview) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn hunyuanimage21_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
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
                width: 32,
                height: 32,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions {
            identifier,
            expected: 2,
            actual: 3,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::TwoDimensional {
                batch: 1,
                width: 31,
                height: 32,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent {
            identifier,
            field: "width",
            value: 31,
        }) if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let wrong_channels = tensor(&backend, &[1, 63, 1, 1], &[0.0; 63], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let wrong_rank = tensor(&backend, &[1, 64, 1], &[0.0; 64], DType::F32, &context)?;
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
            width: 32,
            height: 32,
        },
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
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(35);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_stream_input = tensor(
        &backend,
        &[1, 64, 1, 1],
        &[0.0; 64],
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

    let input = tensor(&backend, &[1, 64, 1, 1], &[0.0; 64], DType::F32, &context)?;
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
