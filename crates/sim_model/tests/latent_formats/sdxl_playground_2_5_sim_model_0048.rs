use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentFormatRegistry, LatentTensorLayout, LatentTransform,
    PreviewReshape, empty_latent,
    generated_sdxl_playground_2_5_sim_model_0048::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

const SOURCE_MEANS: [f32; 4] = [-1.6574, 1.886, -1.383, 2.5155];
const SOURCE_STANDARD_DEVIATIONS: [f32; 4] = [8.4927, 5.9022, 6.5498, 5.2299];
const SOURCE_PREVIEW_FACTORS: [[f32; 3]; 4] = [
    [0.3920, 0.4054, 0.4549],
    [-0.2634, -0.0196, 0.0653],
    [0.0568, 0.1687, -0.0755],
    [-0.3112, -0.2359, -0.2076],
];
static REGISTRY: [sim_model::LatentFormatDefinition; 1] = [LATENT_FORMAT];

#[test]
fn val_latent_001_sdxl_playground_2_5_exact_contract_selector_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "SDXL_Playground_2_5");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0048");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 4);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 0.5_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.channel_means, &SOURCE_MEANS);
    assert_eq!(LATENT_FORMAT.channel_stds, &SOURCE_STANDARD_DEVIATIONS);
    assert_eq!(LATENT_FORMAT.preview_factors, &SOURCE_PREVIEW_FACTORS);
    assert_eq!(LATENT_FORMAT.preview_bias, None);
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, Some("taesdxl_decoder"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::PerChannelAffine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0048","identifier":"SDXL_Playground_2_5"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    let registry = LatentFormatRegistry::checked(&REGISTRY)?;
    assert_eq!(
        registry.get(&decoded).map(|format| format.feature_id),
        Some("COMFY-MODEL-0048")
    );
    Ok(())
}

#[test]
fn sdxl_playground_2_5_geometry_per_channel_round_trip_and_preview_are_source_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let stream = StreamId::new(48);
    let context =
        backend.execution_context(stream, authority.authorize_workspace(0)?, &cancellation);

    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::TwoDimensional {
            batch: 2,
            width: 17,
            height: 25,
        },
        DType::F16,
        stream,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 4, 3, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), stream);

    let source = [
        -2.0, 0.5, 1.0, 3.0, -1.5, 2.0, 0.25, -0.75, 4.0, -3.0, 1.25, 0.0, -0.5, 5.0, -4.0, 2.5,
    ];
    let input = tensor(&backend, &[2, 4, 1, 2], &source, DType::F32, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    assert_eq!(model_facing.descriptor(), input.descriptor());
    for (index, (actual, source)) in values(&model_facing)?.iter().zip(source).enumerate() {
        let channel = (index / 2) % 4;
        let expected =
            (source - SOURCE_MEANS[channel]) * 0.5_f32 / SOURCE_STANDARD_DEVIATIONS[channel];
        assert!((actual - expected).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_ne!(restored.tensor_id(), model_facing.tensor_id());
    assert_ne!(restored.storage_id(), model_facing.storage_id());
    for (actual, expected) in values(&restored)?.iter().zip(source) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }

    let preview_source = [1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 1_000.0, 2_000.0];
    let preview_input = tensor(
        &backend,
        &[1, 4, 1, 2],
        &preview_source,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_ne!(preview.tensor_id(), preview_input.tensor_id());
    assert_ne!(preview.storage_id(), preview_input.storage_id());
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 2]);
    assert_eq!(preview.descriptor().dtype(), DType::F32);
    assert_eq!(preview.descriptor().device(), backend.device());
    assert_eq!(preview.descriptor().stream(), stream);
    let preview_values = values(&preview)?;
    for color in 0..3 {
        for position in 0..2 {
            let expected = (0..4).fold(0.0_f32, |value, channel| {
                preview_source[channel * 2 + position]
                    .mul_add(SOURCE_PREVIEW_FACTORS[channel][color], value)
            });
            assert!((preview_values[color * 2 + position] - expected).abs() <= 1.0e-4);
        }
    }
    Ok(())
}

#[test]
fn sdxl_playground_2_5_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
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

    let wrong_channels = tensor(&backend, &[1, 3, 1, 1], &[0.0; 3], DType::F32, &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { identifier, .. })
            if identifier == LATENT_FORMAT_IDENTIFIER
    ));
    let f16 = tensor(&backend, &[1, 4, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));

    let foreign_stream = StreamId::new(49);
    let foreign_cancellation = CancellationToken::default();
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &foreign_cancellation,
    );
    let foreign_input = tensor(
        &backend,
        &[1, 4, 1, 1],
        &[0.0; 4],
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

    let input = tensor(&backend, &[1, 4, 1, 1], &[0.0; 4], DType::F32, &context)?;
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
