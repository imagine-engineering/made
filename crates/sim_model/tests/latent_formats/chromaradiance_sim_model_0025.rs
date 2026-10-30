use sim_model::generated_chromaradiance_sim_model_0025::{
    LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER,
};
use sim_model::{
    LatentExtent, LatentFormatDescriptor, LatentFormatError, LatentFormatIdentity,
    LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent, process_latent_in,
    process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_chromaradiance_exact_contract_and_stable_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "ChromaRadiance");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0025");
    assert_eq!(LATENT_FORMAT.identifier, LATENT_FORMAT_IDENTIFIER);
    assert_eq!(LATENT_FORMAT.channels, 3);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor, 1.0);
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert!(LATENT_FORMAT.channel_means.is_empty());
    assert!(LATENT_FORMAT.channel_stds.is_empty());
    assert_eq!(
        LATENT_FORMAT.preview_factors,
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],]
    );
    assert_eq!(LATENT_FORMAT.preview_bias, None);
    assert_eq!(LATENT_FORMAT.preview_reshape, PreviewReshape::None);
    assert_eq!(LATENT_FORMAT.decoder_name, None);
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Identity);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_vec(&descriptor.identity)?;
    let decoded: LatentFormatIdentity = serde_json::from_slice(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    assert_eq!(decoded.feature_id(), "COMFY-MODEL-0025");
    assert_eq!(decoded.identifier(), "ChromaRadiance");
    assert_eq!(
        String::from_utf8(encoded)?,
        r#"{"schema_version":1,"feature_id":"COMFY-MODEL-0025","identifier":"ChromaRadiance"}"#
    );
    Ok(())
}

#[test]
fn chromaradiance_shapes_identity_transform_preview_and_cancellation()
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
            width: 5,
            height: 3,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 3, 3, 5]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);

    let input = tensor(
        &backend,
        &[2, 3, 1, 2],
        &[
            1.0, 2.0, 10.0, 20.0, 100.0, 200.0, -1.0, -2.0, -10.0, -20.0, -100.0, -200.0,
        ],
        &context,
    )?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.tensor_id(), input.tensor_id());
    assert_eq!(model_facing.storage_id(), input.storage_id());
    assert_eq!(values(&model_facing)?, values(&input)?);
    assert_eq!(model_facing.descriptor().shape(), &[2, 3, 1, 2]);
    assert_eq!(model_facing.descriptor().dtype(), DType::F32);
    assert_eq!(model_facing.descriptor().device(), backend.device());
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.tensor_id(), input.tensor_id());
    assert_eq!(restored.storage_id(), input.storage_id());
    assert_eq!(values(&restored)?, values(&input)?);

    let f16_identity = process_latent_in(&LATENT_FORMAT, &backend, &empty, &context)?;
    assert_eq!(f16_identity.tensor_id(), empty.tensor_id());
    assert_eq!(f16_identity.storage_id(), empty.storage_id());
    assert_eq!(f16_identity.descriptor().dtype(), DType::F16);

    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[2, 3, 1, 2]);
    assert_eq!(values(&preview)?, values(&input)?);

    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::TwoDimensional {
                batch: 1,
                width: 0,
                height: 3,
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
    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::OneDimensional {
                batch: 1,
                length: 4,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions {
            expected: 2,
            actual: 1,
            ..
        })
    ));

    cancellation.cancel();
    assert!(process_latent_in(&LATENT_FORMAT, &backend, &input, &context).is_err());
    assert!(project_latent_preview(&LATENT_FORMAT, &backend, &input, &context).is_err());
    Ok(())
}

fn tensor(
    backend: &impl TensorBackend,
    shape: &[u64],
    values: &[f32],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    let (mut tensor, _) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    let mut write = tensor.write()?;
    let bytes = write.bytes_mut()?;
    if bytes.len() != values.len() * 4 {
        return Err("fixture tensor length mismatch".into());
    }
    for (destination, value) in bytes.chunks_exact_mut(4).zip(values) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
    drop(write);
    Ok(tensor)
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let count = tensor.descriptor().element_count()?;
    let mut values = Vec::with_capacity(usize::try_from(count)?);
    for index in 0..count {
        let bytes = tensor.linear_element_bytes(index)?;
        values.push(f32::from_le_bytes(bytes.try_into()?));
    }
    Ok(values)
}
