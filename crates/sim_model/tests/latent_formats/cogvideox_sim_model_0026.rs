use sim_model::{
    LatentExtent, LatentFormatDescriptor, LatentFormatError, LatentFormatIdentity,
    LatentFormatRegistry, empty_latent,
    generated_cogvideox_sim_model_0026::{LATENT_FORMAT, LATENT_FORMAT_IDENTIFIER},
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
    TensorError,
};
use sim_types::CancellationToken;

static REGISTRY: [sim_model::LatentFormatDefinition; 1] = [LATENT_FORMAT];

#[test]
fn val_latent_001_cogvideox_definition_and_identity_match_the_source_contract()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(LATENT_FORMAT_IDENTIFIER, "CogVideoX");
    assert_eq!(LATENT_FORMAT.feature_id, "COMFY-MODEL-0026");
    assert_eq!(LATENT_FORMAT.channels, 16);
    assert_eq!(LATENT_FORMAT.dimensions, 3);
    assert_eq!(LATENT_FORMAT.spatial_downscale_ratio, 8);
    assert_eq!(LATENT_FORMAT.temporal_downscale_ratio, 4);
    assert_eq!(
        LATENT_FORMAT.scale_factor.to_bits(),
        1.152_584_3_f32.to_bits()
    );
    assert_eq!(LATENT_FORMAT.shift_factor, 0.0);
    assert!(LATENT_FORMAT.preview_factors.is_empty());
    assert!(LATENT_FORMAT.decoder_name.is_none());

    let descriptor = LatentFormatDescriptor::checked(&LATENT_FORMAT)?;
    let identity = LatentFormatIdentity::new("COMFY-MODEL-0026", "CogVideoX")?;
    assert_eq!(descriptor.identity, identity);
    assert_eq!(
        serde_json::to_string(&descriptor.identity)?,
        r#"{"schema_version":1,"feature_id":"COMFY-MODEL-0026","identifier":"CogVideoX"}"#
    );
    let registry = LatentFormatRegistry::checked(&REGISTRY)?;
    assert_eq!(
        registry.get(&identity).map(|format| format.identifier),
        Some("CogVideoX")
    );
    Ok(())
}

#[test]
fn cogvideox_geometry_transform_errors_and_cancellation_use_the_canonical_owner()
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
            batch: 2,
            frames: 5,
            width: 17,
            height: 16,
        },
        DType::F32,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 16, 2, 2, 2]);
    assert_eq!(empty.descriptor().dtype(), DType::F32);
    assert_eq!(empty.descriptor().device(), backend.device());
    assert!(values(&empty)?.iter().all(|value| *value == 0.0));

    let f16 = empty_latent(
        &LATENT_FORMAT,
        &backend,
        LatentExtent::ThreeDimensional {
            batch: 1,
            frames: 1,
            width: 8,
            height: 8,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(f16.descriptor().dtype(), DType::F16);
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(TensorError::DTypeMismatch { .. }))
    ));

    let input_values = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
    let input = tensor(&backend, &[1, 16, 1, 1, 1], &input_values, &context)?;
    let encoded = process_latent_in(&LATENT_FORMAT, &backend, &input, &context)?;
    for (actual, source) in values(&encoded)?.iter().zip(&input_values) {
        assert!((actual - source * 1.152_584_3_f32).abs() <= 1.0e-6);
    }
    let decoded = process_latent_out(&LATENT_FORMAT, &backend, &encoded, &context)?;
    for (actual, expected) in values(&decoded)?.iter().zip(&input_values) {
        assert!((actual - expected).abs() <= 1.0e-6);
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
    let invalid = tensor(&backend, &[1, 15, 1, 1, 1], &[0.0; 15], &context)?;
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &invalid, &context),
        Err(LatentFormatError::InvalidShape { .. })
    ));

    cancellation.cancel();
    assert!(matches!(
        process_latent_in(&LATENT_FORMAT, &backend, &input, &context),
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
        values.push(f32::from_le_bytes(
            tensor.linear_element_bytes(index)?.try_into()?,
        ));
    }
    Ok(values)
}
