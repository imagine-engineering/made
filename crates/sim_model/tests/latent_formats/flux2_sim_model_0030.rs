use sim_model::{
    LatentExtent, LatentFormatDescriptor, LatentFormatError, LatentFormatIdentity,
    LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    generated_flux2_sim_model_0030::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_flux2_exact_contract_and_stable_identity()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Flux2");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0030");
    assert_eq!(LATENT_FORMAT.channels, 128);
    assert_eq!(LATENT_FORMAT.dimensions, 2);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 16);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(LATENT_FORMAT.scale_factor, 1.0);
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert_eq!(LATENT_FORMAT.preview_factors.len(), 32);
    assert_eq!(LATENT_FORMAT.preview_factors[0], [0.0058, 0.0113, 0.0073]);
    assert_eq!(
        LATENT_FORMAT.preview_factors[31],
        [-0.0111, -0.0460, -0.0614]
    );
    assert_eq!(
        LATENT_FORMAT.preview_bias,
        Some([-0.0329, -0.0718, -0.0851])
    );
    assert_eq!(
        LATENT_FORMAT.preview_reshape,
        PreviewReshape::Flux2Spatial2x
    );
    assert_eq!(LATENT_FORMAT.decoder_name, Some("taef2_decoder"));
    assert_eq!(LATENT_FORMAT.layout, LatentTensorLayout::ChannelsFirst);
    assert_eq!(LATENT_FORMAT.transform, LatentTransform::Identity);

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        r#"{"schema_version":1,"feature_id":"COMFY-MODEL-0030","identifier":"Flux2"}"#
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn flux2_geometry_aliasing_packed_preview_errors_and_cancellation()
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
        LatentExtent::TwoDimensional {
            batch: 2,
            width: 33,
            height: 32,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 128, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);
    let identity_f16 = process_latent_in(&LATENT_FORMAT, &backend, &empty, &context)?;
    assert_eq!(identity_f16.tensor_id(), empty.tensor_id());
    assert_eq!(identity_f16.storage_id(), empty.storage_id());

    let mut source = vec![0.0; 128];
    source[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let input = tensor(&backend, &[1, 128, 1, 1], &source, &context)?;
    let model_facing = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(model_facing.tensor_id(), input.tensor_id());
    assert_eq!(model_facing.storage_id(), input.storage_id());
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &model_facing, &context)?;
    assert_eq!(restored.tensor_id(), input.tensor_id());
    assert_eq!(restored.storage_id(), input.storage_id());

    let preview = project_latent_preview(&LATENT_FORMAT, &backend, &input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 2, 2]);
    let expected = [
        -0.0271, -0.0213, -0.0155, -0.0097, -0.0605, -0.0492, -0.0379, -0.0266, -0.0778, -0.0705,
        -0.0632, -0.0559,
    ];
    for (actual, expected) in values(&preview)?.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-6);
    }

    assert!(matches!(
        empty_latent(
            &LATENT_FORMAT,
            &backend,
            LatentExtent::OneDimensional {
                batch: 1,
                length: 16,
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions { .. })
    ));
    let invalid = tensor(&backend, &[1, 127, 1, 1], &[0.0; 127], &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &invalid, &context),
        Err(LatentFormatError::InvalidShape { .. })
    ));

    cancellation.cancel();
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
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    let (mut tensor, _) = backend.fill(Scalar::Float(0.0), descriptor, context)?;
    let mut write = tensor.write()?;
    let bytes = write.bytes_mut()?;
    if bytes.len() != std::mem::size_of_val(values) {
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
        values.push(f32::from_le_bytes(
            tensor.linear_element_bytes(index)?.try_into()?,
        ));
    }
    Ok(values)
}
