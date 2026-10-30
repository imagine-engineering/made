use sim_model::{
    LATENT_FORMAT_SCHEMA_VERSION, LatentExtent, LatentFormatDescriptor, LatentFormatError,
    LatentFormatIdentity, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_wan22_sim_model_0054::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

const SOURCE_MEANS: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
const SOURCE_STDS: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];
const SOURCE_PREVIEW_FACTORS: [[f32; 3]; 48] = [
    [0.0119, 0.0103, 0.0046],
    [-0.1062, -0.0504, 0.0165],
    [0.0140, 0.0409, 0.0491],
    [-0.0813, -0.0677, 0.0607],
    [0.0656, 0.0851, 0.0808],
    [0.0264, 0.0463, 0.0912],
    [0.0295, 0.0326, 0.0590],
    [-0.0244, -0.0270, 0.0025],
    [0.0443, -0.0102, 0.0288],
    [-0.0465, -0.0090, -0.0205],
    [0.0359, 0.0236, 0.0082],
    [-0.0776, 0.0854, 0.1048],
    [0.0564, 0.0264, 0.0561],
    [0.0006, 0.0594, 0.0418],
    [-0.0319, -0.0542, -0.0637],
    [-0.0268, 0.0024, 0.0260],
    [0.0539, 0.0265, 0.0358],
    [-0.0359, -0.0312, -0.0287],
    [-0.0285, -0.1032, -0.1237],
    [0.1041, 0.0537, 0.0622],
    [-0.0086, -0.0374, -0.0051],
    [0.0390, 0.0670, 0.2863],
    [0.0069, 0.0144, 0.0082],
    [0.0006, -0.0167, 0.0079],
    [0.0313, -0.0574, -0.0232],
    [-0.1454, -0.0902, -0.0481],
    [0.0714, 0.0827, 0.0447],
    [-0.0304, -0.0574, -0.0196],
    [0.0401, 0.0384, 0.0204],
    [-0.0758, -0.0297, -0.0014],
    [0.0568, 0.1307, 0.1372],
    [-0.0055, -0.0310, -0.0380],
    [0.0239, -0.0305, 0.0325],
    [-0.0663, -0.0673, -0.0140],
    [-0.0416, -0.0047, -0.0023],
    [0.0166, 0.0112, -0.0093],
    [-0.0211, 0.0011, 0.0331],
    [0.1833, 0.1466, 0.2250],
    [-0.0368, 0.0370, 0.0295],
    [-0.3441, -0.3543, -0.2008],
    [-0.0479, -0.0489, -0.0420],
    [-0.0660, -0.0153, 0.0800],
    [-0.0101, 0.0068, 0.0156],
    [-0.0690, -0.0452, -0.0927],
    [-0.0145, 0.0041, 0.0015],
    [0.0421, 0.0451, 0.0373],
    [0.0504, -0.0483, -0.0356],
    [-0.0837, 0.0168, 0.0055],
];
const SOURCE_PREVIEW_BIAS: [f32; 3] = [0.0317, -0.0878, -0.1388];

#[test]
fn val_latent_001_wan22_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Wan22");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0054");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 48);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 16);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 4);
    assert_eq!(LATENT_FORMAT.scale_factor.to_bits(), 1.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.shift_factor.to_bits(), 0.0_f32.to_bits());
    assert_eq!(LATENT_FORMAT.channel_means, &SOURCE_MEANS);
    assert_eq!(LATENT_FORMAT.channel_stds, &SOURCE_STDS);
    assert_eq!(LATENT_FORMAT.preview_factors, &SOURCE_PREVIEW_FACTORS);
    assert_eq!(LATENT_FORMAT.preview_bias, Some(SOURCE_PREVIEW_BIAS));
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, Some("lighttaew2_2"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::PerChannelAffine);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        format!(
            r#"{{"schema_version":{LATENT_FORMAT_SCHEMA_VERSION},"feature_id":"COMFY-MODEL-0054","identifier":"Wan22"}}"#
        )
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn wan22_geometry_per_channel_round_trip_and_preview_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let stream = StreamId::new(54);
    let context =
        backend.execution_context(stream, authority.authorize_workspace(0)?, &cancellation);
    let empty = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional {
            batch: 2,
            frames: 5,
            width: 47,
            height: 33,
        },
        DType::F16,
        stream,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 48, 2, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), stream);

    let source = (0..48)
        .map(|index| index as f32 / 11.0 - 1.75)
        .collect::<Vec<_>>();
    let input = tensor(&backend, &[1, 48, 1, 1, 1], &source, DType::F32, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_ne!(model_facing.tensor_id(), input.tensor_id());
    assert_ne!(model_facing.storage_id(), input.storage_id());
    for (channel, actual) in values(&model_facing)?.iter().enumerate() {
        let expected = (source[channel] - SOURCE_MEANS[channel]) / SOURCE_STDS[channel];
        assert!((actual - expected).abs() <= 2.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    for (actual, expected) in values(&restored)?.iter().zip(&source) {
        assert!((actual - expected).abs() <= 3.0e-6);
    }

    let mut preview_source = [0.0_f32; 48];
    preview_source[0] = 2.0;
    preview_source[47] = -3.0;
    let preview_input = tensor(
        &backend,
        &[1, 48, 1, 1, 1],
        &preview_source,
        DType::F32,
        &context,
    )?;
    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &preview_input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 1, 1]);
    let expected = [
        (-3.0_f32).mul_add(
            SOURCE_PREVIEW_FACTORS[47][0],
            2.0_f32.mul_add(SOURCE_PREVIEW_FACTORS[0][0], SOURCE_PREVIEW_BIAS[0]),
        ),
        (-3.0_f32).mul_add(
            SOURCE_PREVIEW_FACTORS[47][1],
            2.0_f32.mul_add(SOURCE_PREVIEW_FACTORS[0][1], SOURCE_PREVIEW_BIAS[1]),
        ),
        (-3.0_f32).mul_add(
            SOURCE_PREVIEW_FACTORS[47][2],
            2.0_f32.mul_add(SOURCE_PREVIEW_FACTORS[0][2], SOURCE_PREVIEW_BIAS[2]),
        ),
    ];
    assert_eq!(values(&preview)?, expected);
    Ok(())
}

#[test]
fn wan22_rejects_invalid_extent_shape_dtype_stream_and_cancellation()
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
                width: 16,
                height: 16
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
                width: 15,
                height: 16
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::InvalidExtent {
            field: "width",
            value: 15,
            ..
        })
    ));
    let wrong_channels = tensor(
        &backend,
        &[1, 47, 1, 1, 1],
        &[0.0; 47],
        DType::F32,
        &context,
    )?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &wrong_channels, &context),
        Err(LatentFormatError::InvalidShape { .. })
    ));
    let f16 = tensor(&backend, &[1, 48, 1, 1, 1], &[], DType::F16, &context)?;
    assert!(matches!(
        process_latent_out(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F16,
        }))
    ));
    let foreign_stream = StreamId::new(54);
    let foreign_context = backend.execution_context(
        foreign_stream,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let foreign_input = tensor(
        &backend,
        &[1, 48, 1, 1, 1],
        &[0.0; 48],
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
        &[1, 48, 1, 1, 1],
        &[0.0; 48],
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
