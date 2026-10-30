use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentFormatRegistry, LatentTensorLayout, LatentTransform,
    PreviewReshape, empty_latent,
    generated_ltxv_sim_model_0040::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

static REGISTRY: [sim_model::LatentFormatDefinition; 1] = [LATENT_FORMAT];

#[test]
fn val_latent_001_ltxv_exact_contract_and_stable_identity() -> Result<(), Box<dyn std::error::Error>>
{
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "LTXV");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0040");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 128);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 32);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(LATENT_FORMAT.preview_factors.len(), 128);
    assert_eq!(preview_factors_digest(), 0x3124_55d1_15ed_a3c9);
    assert_eq!(
        LATENT_FORMAT.preview_factors[0],
        [0.011202, -0.00063815, -0.010021]
    );
    assert_eq!(
        LATENT_FORMAT.preview_factors[31],
        [-0.0063704, -0.0084827, -0.0095483]
    );
    assert_eq!(
        LATENT_FORMAT.preview_factors[63],
        [-0.054229, 0.026644, 0.0063394]
    );
    assert_eq!(
        LATENT_FORMAT.preview_factors[95],
        [0.02054, 0.020729, 0.0064338]
    );
    assert_eq!(
        LATENT_FORMAT.preview_factors[127],
        [-0.014605, -0.0067032, 0.0039675]
    );
    assert_eq!(
        LATENT_FORMAT.preview_bias,
        Some([-0.0571, -0.1657, -0.2512])
    );
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Affine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    assert_eq!(descriptor.channels, 128);
    assert_eq!(descriptor.channel_means, Vec::<f32>::new());
    assert_eq!(descriptor.channel_stds, Vec::<f32>::new());
    assert_eq!(descriptor.preview_factors, LATENT_FORMAT.preview_factors);
    assert_eq!(descriptor.preview_bias, LATENT_FORMAT.preview_bias);
    assert_eq!(descriptor.decoder_name, None);
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0040","identifier":"LTXV"}}"#
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
fn ltxv_batched_geometry_allocating_round_trip_and_preview()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let stream = StreamId::new(40);
    let context =
        backend.execution_context(stream, authority.authorize_workspace(0)?, &cancellation);

    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional {
            batch: 3,
            frames: 17,
            width: 95,
            height: 65,
        },
        DType::F16,
        stream,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[3, 128, 3, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), stream);

    let input_values = (0..512)
        .map(|value| value as f32 / 29.0 - 7.0)
        .collect::<Vec<_>>();
    let input = tensor(
        &backend,
        &[2, 128, 2, 1, 1],
        &input_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.descriptor(), input.descriptor());
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    assert_eq!(values(&model_facing)?, input_values);
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.descriptor(), input.descriptor());
    assert_ne!(restored.tensor_id(), model_facing.tensor_id());
    assert_ne!(restored.storage_id(), model_facing.storage_id());
    assert_eq!(values(&restored)?, input_values);

    let mut preview_input_values = vec![0.0; 512];
    preview_input_values[0] = 2.0;
    preview_input_values[63 * 2] = -3.0;
    preview_input_values[(128 + 127) * 2 + 1] = 4.0;
    let preview_input = tensor(
        &backend,
        &[2, 128, 2, 1, 1],
        &preview_input_values,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[2, 3, 2, 1, 1]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), stream);
    let bias = LATENT_FORMAT
        .preview_bias
        .ok_or("LTXV preview bias required")?;
    let expected = [
        2.0_f32.mul_add(
            LATENT_FORMAT.preview_factors[0][0],
            (-3.0_f32).mul_add(LATENT_FORMAT.preview_factors[63][0], bias[0]),
        ),
        bias[0],
        2.0_f32.mul_add(
            LATENT_FORMAT.preview_factors[0][1],
            (-3.0_f32).mul_add(LATENT_FORMAT.preview_factors[63][1], bias[1]),
        ),
        bias[1],
        2.0_f32.mul_add(
            LATENT_FORMAT.preview_factors[0][2],
            (-3.0_f32).mul_add(LATENT_FORMAT.preview_factors[63][2], bias[2]),
        ),
        bias[2],
        bias[0],
        4.0_f32.mul_add(LATENT_FORMAT.preview_factors[127][0], bias[0]),
        bias[1],
        4.0_f32.mul_add(LATENT_FORMAT.preview_factors[127][1], bias[1]),
        bias[2],
        4.0_f32.mul_add(LATENT_FORMAT.preview_factors[127][2], bias[2]),
    ];
    for (actual, expected) in values(&preview)?.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn ltxv_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
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
                width: 32,
                height: 32,
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

    let wrong_channels = tensor(
        &backend,
        &[1, 127, 1, 1, 1],
        &[0.0; 127],
        DType::F32,
        &context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let wrong_rank = tensor(&backend, &[1, 128, 1, 1], &[0.0; 128], DType::F32, &context)?;
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &wrong_rank, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let f16 = tensor(&backend, &[1, 128, 1, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let alternate_stream = StreamId::new(41);
    let alternate_context = backend.execution_context(
        alternate_stream,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let alternate_input = tensor(
        &backend,
        &[1, 128, 1, 1, 1],
        &[0.0; 128],
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
        &[1, 128, 1, 1, 1],
        &[0.0; 128],
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
                width: 32,
                height: 32,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::Tensor(TensorError::Cancelled))
    ));
    Ok(())
}

fn preview_factors_digest() -> u64 {
    LATENT_FORMAT
        .preview_factors
        .iter()
        .flatten()
        .fold(0xcbf2_9ce4_8422_2325, |digest, value| {
            value
                .to_bits()
                .to_le_bytes()
                .iter()
                .fold(digest, |digest, byte| {
                    (digest ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        })
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
