use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentFormatRegistry, LatentTensorLayout, LatentTransform,
    PreviewReshape, empty_latent,
    generated_cosmos1cv8x8x8_sim_model_0028::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

static REGISTRY: [sim_model::LatentFormatDefinition; 1] = [LATENT_FORMAT];

#[test]
fn val_latent_001_cosmos1cv8x8x8_exact_contract_and_stable_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Cosmos1CV8x8x8");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0028");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 16);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.scale_factor, 1.0);
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(
        LATENT_FORMAT.preview_factors,
        &[
            [0.1817, 0.2284, 0.2423],
            [-0.0586, -0.0862, -0.3108],
            [-0.4703, -0.4255, -0.3995],
            [0.0803, 0.1963, 0.1001],
            [-0.0820, -0.1050, 0.0400],
            [0.2511, 0.3098, 0.2787],
            [-0.1830, -0.2117, -0.0040],
            [-0.0621, -0.2187, -0.0939],
            [0.3619, 0.1082, 0.1455],
            [0.3164, 0.3922, 0.2575],
            [0.1152, 0.0231, -0.0462],
            [-0.1434, -0.3609, -0.3665],
            [0.0635, 0.1471, 0.1680],
            [-0.3635, -0.1963, -0.3248],
            [-0.1865, 0.0365, 0.2346],
            [0.0447, 0.0994, 0.0881],
        ]
    );
    assert_eq!(
        LATENT_FORMAT.preview_bias,
        Some([-0.1223, -0.1889, -0.1976])
    );
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Affine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0028","identifier":"Cosmos1CV8x8x8"}}"#
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
fn cosmos1cv8x8x8_batched_geometry_scaling_preview_and_metadata()
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
            frames: 9,
            width: 23,
            height: 17,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[3, 16, 2, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);

    let mut input_values = vec![0.0; 32];
    input_values[0] = 1.0;
    input_values[31] = -2.0;
    let input = tensor(
        &backend,
        &[2, 16, 1, 1, 1],
        &input_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.descriptor(), input.descriptor());
    assert_eq!(values(&model_facing)?, input_values);
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.descriptor(), input.descriptor());
    assert_eq!(values(&restored)?, input_values);

    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[2, 3, 1, 1, 1]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), StreamId::DEFAULT);
    let expected = [
        -0.1223 + 0.1817,
        -0.1889 + 0.2284,
        -0.1976 + 0.2423,
        -0.1223 - 2.0 * 0.0447,
        -0.1889 - 2.0 * 0.0994,
        -0.1976 - 2.0 * 0.0881,
    ];
    for (actual, expected) in values(&preview)?.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn cosmos1cv8x8x8_reports_extent_shape_dtype_stream_and_cancellation_errors()
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

    let invalid_shape = tensor(
        &backend,
        &[1, 15, 1, 1, 1],
        &[0.0; 15],
        DType::F32,
        &context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &invalid_shape, &context),
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

    let alternate_stream = StreamId::new(7);
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
