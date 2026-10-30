use sim_model::{
    LatentExtent, LatentFormatDescriptor, LatentFormatError, LatentFormatIdentity, empty_latent,
    generated_hunyuanthree_dv2mini_sim_model_0034::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

#[test]
fn val_latent_001_hunyuan3dv2mini_exact_contract_and_serialized_identity()
-> Result<(), Box<dyn std::error::Error>>
{
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "Hunyuan3Dv2mini");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0034");
    assert_eq!(LATENT_FORMAT.channels, 64);
    assert_eq!(LATENT_FORMAT.dimensions, 1);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 1);
    assert_eq!(
        LATENT_FORMAT.scale_factor.to_bits(),
        1.018_813_7_f32.to_bits()
    );
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert!(LATENT_FORMAT.preview_factors.is_empty());
    assert!(LATENT_FORMAT.decoder_name.is_none());

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let encoded = serde_json::to_string(&descriptor.identity)?;
    assert_eq!(
        encoded,
        r#"{"schema_version":1,"feature_id":"COMFY-MODEL-0034","identifier":"Hunyuan3Dv2mini"}"#
    );
    let decoded: LatentFormatIdentity = serde_json::from_str(&encoded)?;
    assert_eq!(decoded, descriptor.identity);
    Ok(())
}

#[test]
fn hunyuan3dv2mini_geometry_scaling_errors_and_cancellation()
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
        LatentExtent::OneDimensional {
            batch: 2,
            length: 17,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 64, 17]);
    assert_eq!(empty.descriptor().dtype(), DType::F16);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &empty, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch { .. }))
    ));

    let source = vec![2.0; 64];
    let input = tensor(&backend, &[1, 64, 1], &source, &context)?;
    let encoded = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    for value in values(&encoded)? {
        assert!((value - 2.0 * 1.018_813_7_f32).abs() <= 1.0e-6);
    }
    let restored = process_latent_out(&LATENT_FORMAT, &backend, &encoded, &context)?;
    for value in values(&restored)? {
        assert!((value - 2.0).abs() <= 1.0e-6);
    }
    assert!(matches!(
        project_latent_preview(&LATENT_FORMAT, &backend, &input, &context),
        Err(LatentFormatError::PreviewUnavailable { .. })
    ));
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
        Err(LatentFormatError::ExtentDimensions { .. })
    ));
    let invalid = tensor(&backend, &[1, 63, 1], &[0.0; 63], &context)?;
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
