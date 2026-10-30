use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_mochi_sim_model_0041::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_mochi_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Mochi");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0041");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 12);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 6);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    let expected_means = [
        -0.067_308_959_535_100_81_f64 as f32,
        -0.038_011_381_506_090_416_f64 as f32,
        -0.074_778_209_128_661_41_f64 as f32,
        -0.055_652_644_709_955_61_f64 as f32,
        0.012_767_231_469_026_969_f64 as f32,
        -0.047_035_427_462_464_19_f64 as f32,
        0.043_896_967_884_726_704_f64 as f32,
        -0.093_463_057_070_259_76_f64 as f32,
        -0.099_183_147_630_168_93_f64 as f32,
        -0.008_729_793_427_399_178_f64 as f32,
        -0.011_931_556_316_503_654_f64 as f32,
        -0.032_199_339_188_728_5_f64 as f32,
    ];
    assert_eq!(LATENT_FORMAT.channel_means, &expected_means);
    let expected_standard_deviations = [
        0.926_379_502_849_386_3_f64 as f32,
        0.924_889_454_319_376_6_f64 as f32,
        0.939_305_939_089_061_7_f64 as f32,
        0.959_253_732_819_592_f64 as f32,
        0.824_456_013_275_279_3_f64 as f32,
        0.917_259_975_397_747_f64 as f32,
        0.929_415_443_101_369_6_f64 as f32,
        1.372_094_235_778_852_1_f64 as f32,
        0.881_393_668_867_029_f64 as f32,
        0.916_831_569_212_434_8_f64 as f32,
        0.918_524_927_934_555_2_f64 as f32,
        0.927_475_757_080_504_1_f64 as f32,
    ];
    assert_eq!(LATENT_FORMAT.channel_stds, &expected_standard_deviations);
    assert_eq!(LATENT_FORMAT.preview_factors.len(), 12);
    assert_eq!(LATENT_FORMAT.preview_factors[0], [-0.0069, -0.0045, 0.0018]);
    assert_eq!(
        LATENT_FORMAT.preview_factors[11],
        [-0.0396, -0.0495, -0.0281]
    );
    assert_eq!(
        LATENT_FORMAT.preview_bias,
        Some([-0.0940, -0.1418, -0.1453])
    );
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::PerChannelAffine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0041","identifier":"Mochi"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn mochi_geometry_per_channel_round_trip_and_preview_are_exact()
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
            frames: 13,
            width: 17,
            height: 9,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 12, 3, 1, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);

    let source_values = (0..48)
        .map(|index| index as f32 / 13.0 - 1.5)
        .collect::<Vec<_>>();
    let input = tensor(
        &backend,
        &[2, 12, 2, 1, 1],
        &source_values,
        DType::F32,
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    assert_eq!(model_facing.descriptor(), input.descriptor());
    for (index, (actual, source)) in values(&model_facing)?
        .iter()
        .zip(&source_values)
        .enumerate()
    {
        let channel = (index / 2) % 12;
        let expected =
            (source - LATENT_FORMAT.channel_means[channel]) / LATENT_FORMAT.channel_stds[channel];
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_ne!(restored.tensor_id(), model_facing.tensor_id());
    assert_ne!(restored.storage_id(), model_facing.storage_id());
    assert_eq!(restored.descriptor(), input.descriptor());
    for (actual, expected) in values(&restored)?.iter().zip(&source_values) {
        assert!((actual - expected).abs() <= 2.0e-6);
    }

    let preview_values = (0..12).map(|index| index as f32 / 10.0).collect::<Vec<_>>();
    let preview_input = tensor(
        &backend,
        &[1, 12, 1, 1, 1],
        &preview_values,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 1, 1]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), StreamId::DEFAULT);
    let bias = LATENT_FORMAT
        .preview_bias
        .ok_or("Mochi preview bias is required")?;
    for (color, actual) in values(&preview)?.iter().enumerate() {
        let expected = LATENT_FORMAT
            .preview_factors
            .iter()
            .enumerate()
            .fold(bias[color], |value, (channel, factors)| {
                preview_values[channel].mul_add(factors[color], value)
            });
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    Ok(())
}

#[test]
fn mochi_rejects_invalid_geometry_shape_dtype_stream_and_cancellation()
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
        Err(LatentFormatError::ExtentDimensions { identifier, expected: 3, actual: 2 })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::ThreeDimensional { batch: 0, frames: 1, width: 8, height: 8 },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent { identifier, field: "batch", value: 0 })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));

    let wrong_channels = tensor(&backend, &[1, 11, 1, 1, 1], &[], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let f16 = tensor(&backend, &[1, 12, 1, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(41);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_input = tensor(
        &backend,
        &[1, 12, 1, 1, 1],
        &[0.0; 12],
        DType::F32,
        &foreign_context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &foreign_input, &context),
        Err(LatentFormatError::Tensor(TensorError::StreamMismatch {
            expected: StreamId::DEFAULT,
            actual,
        })) if actual == foreign_stream
    ));

    let input = tensor(
        &backend,
        &[1, 12, 1, 1, 1],
        &[0.0; 12],
        DType::F32,
        &context,
    )?;
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
