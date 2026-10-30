#[path = "../build.rs"]
#[allow(dead_code)]
mod build_script;

use sim_model::{
    GENERATED_LATENT_FORMATS, GENERATED_MODULES, LATENT_FORMAT_SCHEMA_VERSION, LatentExtent,
    LatentFormatDefinition, LatentFormatDescriptor, LatentFormatError, LatentFormatIdentity,
    LatentFormatRegistry, LatentTensorLayout, LatentTransform, PreviewReshape, empty_latent,
    process_latent_in, process_latent_out, project_latent_preview,
};
use sim_tensor::{
    CpuWorkspaceAuthority, DType, Scalar, StreamId, Tensor, TensorBackend, TensorDescriptor,
};
use sim_types::CancellationToken;

const PREVIEW: [[f32; 3]; 2] = [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]];
const MEANS: [f32; 2] = [1.0, -2.0];
const STDS: [f32; 2] = [2.0, 4.0];

const AFFINE: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-9001",
    identifier: "FoundationAffine",
    channels: 2,
    dimensions: 2,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 0.5,
    shift_factor: 1.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &PREVIEW,
    preview_bias: Some([0.25, -0.25, 0.0]),
    preview_reshape: PreviewReshape::None,
    decoder_name: Some("foundation_decoder"),
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Affine,
};

const IDENTITY: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-9005",
    identifier: "FoundationIdentity",
    channels: 2,
    dimensions: 2,
    spatial_downscale_ratio: 1,
    temporal_downscale_ratio: 1,
    scale_factor: 1.0,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::Identity,
};

const PER_CHANNEL: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-9002",
    identifier: "FoundationPerChannel",
    channels: 2,
    dimensions: 1,
    spatial_downscale_ratio: 1,
    temporal_downscale_ratio: 4,
    scale_factor: 2.0,
    shift_factor: 0.0,
    channel_means: &MEANS,
    channel_stds: &STDS,
    preview_factors: &[],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::PerChannelAffine,
};

const REFINER: LatentFormatDefinition = LatentFormatDefinition {
    feature_id: "COMFY-MODEL-9003",
    identifier: "FoundationRefiner",
    channels: 2,
    dimensions: 3,
    spatial_downscale_ratio: 8,
    temporal_downscale_ratio: 1,
    scale_factor: 2.0,
    shift_factor: 0.0,
    channel_means: &[],
    channel_stds: &[],
    preview_factors: &[],
    preview_bias: None,
    preview_reshape: PreviewReshape::None,
    decoder_name: None,
    layout: LatentTensorLayout::ChannelsFirst,
    transform: LatentTransform::HunyuanImage21Refiner,
};

static REGISTRY_FORMATS: [LatentFormatDefinition; 2] = [AFFINE, PER_CHANNEL];
static DUPLICATE_IDENTIFIERS: [LatentFormatDefinition; 2] = [
    AFFINE,
    LatentFormatDefinition {
        feature_id: "COMFY-MODEL-9004",
        ..AFFINE
    },
];

#[test]
fn identity_and_registry_are_checked_and_stable() -> Result<(), Box<dyn std::error::Error>> {
    let identity = LatentFormatIdentity::new(AFFINE.feature_id, AFFINE.identifier)?;
    let encoded = serde_json::to_vec(&identity)?;
    let decoded: LatentFormatIdentity = serde_json::from_slice(&encoded)?;
    assert_eq!(decoded, identity);
    assert_eq!(decoded.feature_id(), AFFINE.feature_id);
    assert_eq!(decoded.identifier(), AFFINE.identifier);
    assert!(String::from_utf8(encoded)?.contains(&format!(
        "\"schema_version\":{LATENT_FORMAT_SCHEMA_VERSION}"
    )));
    assert!(serde_json::from_str::<LatentFormatIdentity>(
        r#"{"schema_version":2,"feature_id":"COMFY-MODEL-9001","identifier":"FoundationAffine"}"#,
    )
    .is_err());
    assert!(LatentFormatIdentity::new("MODEL-9001", "FoundationAffine").is_err());
    assert!(LatentFormatIdentity::new("COMFY-MODEL-9001", "bad identifier").is_err());

    let descriptor = LatentFormatDescriptor::checked(&AFFINE)?;
    assert_eq!(descriptor.identity, identity);
    assert_eq!(descriptor.preview_factors, PREVIEW);
    let registry = LatentFormatRegistry::checked(&REGISTRY_FORMATS)?;
    assert_eq!(registry.len(), 2);
    assert_eq!(
        registry.get(&identity).map(|value| value.identifier),
        Some(AFFINE.identifier)
    );
    assert!(matches!(
        LatentFormatRegistry::checked(&DUPLICATE_IDENTIFIERS),
        Err(LatentFormatError::DuplicateIdentifier { .. })
    ));
    Ok(())
}

#[test]
fn build_manifest_has_one_sorted_latent_registry_and_rejects_duplicates()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(GENERATED_MODULES.windows(2).all(|pair| pair[0] < pair[1]));
    let generated_registry = LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)?;
    let generated_latent_module_count = GENERATED_MODULES
        .iter()
        .filter(|module| module.starts_with("latent_formats/"))
        .count();
    assert_eq!(generated_registry.len(), generated_latent_module_count);
    assert!(build_script::valid_module_name(
        "ace_audio_comfy_model_0023"
    ));
    assert!(!build_script::valid_module_name("ACEAudio"));
    let source = "pub const LATENT_FORMAT_IDENTIFIER: &str = \"ACEAudio\";\n";
    assert_eq!(
        build_script::latent_format_identifier(source, std::path::Path::new("ace.rs"))?,
        "ACEAudio"
    );
    let mut identifiers = Vec::new();
    build_script::register_latent_format(&mut identifiers, "first", "ACEAudio")?;
    assert!(build_script::register_latent_format(&mut identifiers, "second", "ACEAudio").is_err());
    assert!(build_script::latent_format_test_names(&[]).is_err());

    let test_directory = tempfile::tempdir()?;
    let expected = vec![("ace_audio".to_owned(), "ACEAudio".to_owned())];
    assert!(build_script::latent_format_test_names_in(&expected, test_directory.path()).is_err());
    std::fs::write(test_directory.path().join("ace_audio.rs"), "")?;
    assert_eq!(
        build_script::latent_format_test_names_in(&expected, test_directory.path())?,
        vec!["ace_audio"]
    );
    std::fs::write(test_directory.path().join("orphan.rs"), "")?;
    assert!(build_script::latent_format_test_names_in(&expected, test_directory.path()).is_err());
    Ok(())
}

#[test]
fn val_latent_001_authoritative_foundation_empty_geometry_affine_preview_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );

    let empty = empty_latent(
        &AFFINE,
        &backend,
        LatentExtent::TwoDimensional {
            batch: 2,
            width: 17,
            height: 16,
        },
        DType::F32,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[2, 2, 2, 2]);
    assert_eq!(values(&empty)?, vec![0.0; 16]);
    assert!(matches!(
        empty_latent(
            &AFFINE,
            &backend,
            LatentExtent::OneDimensional {
                batch: 1,
                length: 8
            },
            DType::F32,
            StreamId::DEFAULT,
            &context,
        ),
        Err(LatentFormatError::ExtentDimensions { .. })
    ));

    let input = tensor(&backend, &[1, 2, 1, 2], &[1.0, 3.0, 2.0, 4.0], &context)?;
    let identity = process_latent_in(&IDENTITY, &backend, &input, &context)?;
    assert_eq!(identity.tensor_id(), input.tensor_id());
    assert_eq!(identity.storage_id(), input.storage_id());
    let identity_out = process_latent_out(&IDENTITY, &backend, &identity, &context)?;
    assert_eq!(identity_out.tensor_id(), input.tensor_id());
    assert_eq!(identity_out.storage_id(), input.storage_id());
    let encoded = process_latent_in(&AFFINE, &backend, &input, &context)?;
    assert_eq!(values(&encoded)?, vec![0.0, 1.0, 0.5, 1.5]);
    let decoded = process_latent_out(&AFFINE, &backend, &encoded, &context)?;
    assert_eq!(values(&decoded)?, values(&input)?);
    assert_eq!(decoded.descriptor().dtype(), DType::F32);
    assert_eq!(decoded.descriptor().device(), backend.device());

    let f16 = empty_latent(
        &AFFINE,
        &backend,
        LatentExtent::TwoDimensional {
            batch: 1,
            width: 8,
            height: 8,
        },
        DType::F16,
        StreamId::DEFAULT,
        &context,
    )?;
    let identity_f16 = process_latent_in(&IDENTITY, &backend, &f16, &context)?;
    assert_eq!(identity_f16.tensor_id(), f16.tensor_id());
    assert_eq!(identity_f16.storage_id(), f16.storage_id());
    assert_eq!(identity_f16.descriptor().dtype(), DType::F16);
    assert!(matches!(
        process_latent_in(&AFFINE, &backend, &f16, &context),
        Err(LatentFormatError::Tensor(
            sim_tensor::TensorError::DTypeMismatch { .. }
        ))
    ));

    let preview = project_latent_preview(&AFFINE, &backend, &input, &context)?;
    assert_eq!(preview.descriptor().shape(), &[1, 3, 1, 2]);
    assert_eq!(values(&preview)?, vec![1.25, 3.25, 1.75, 3.75, -0.5, -0.5]);

    cancellation.cancel();
    assert!(process_latent_in(&AFFINE, &backend, &input, &context).is_err());
    Ok(())
}

#[test]
fn per_channel_and_custom_transforms_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let empty = empty_latent(
        &PER_CHANNEL,
        &backend,
        LatentExtent::OneDimensional {
            batch: 1,
            length: 9,
        },
        DType::F32,
        StreamId::DEFAULT,
        &context,
    )?;
    assert_eq!(empty.descriptor().shape(), &[1, 2, 3]);

    let input = tensor(&backend, &[1, 2, 2], &[1.0, 3.0, -2.0, 2.0], &context)?;
    let encoded = process_latent_in(&PER_CHANNEL, &backend, &input, &context)?;
    assert_eq!(values(&encoded)?, vec![0.0, 2.0, 0.0, 2.0]);
    assert_eq!(
        values(&process_latent_out(
            &PER_CHANNEL,
            &backend,
            &encoded,
            &context
        )?)?,
        values(&input)?
    );

    let refiner_input = tensor(
        &backend,
        &[1, 2, 3, 1, 1],
        &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        &context,
    )?;
    let refiner_encoded = process_latent_in(&REFINER, &backend, &refiner_input, &context)?;
    assert_eq!(refiner_encoded.descriptor().shape(), &[1, 4, 2, 1, 1]);
    assert_eq!(
        values(&refiner_encoded)?,
        vec![2.0, 4.0, 20.0, 40.0, 2.0, 6.0, 20.0, 60.0]
    );
    let refiner_decoded = process_latent_out(&REFINER, &backend, &refiner_encoded, &context)?;
    assert_eq!(
        refiner_decoded.descriptor().shape(),
        refiner_input.descriptor().shape()
    );
    assert_eq!(values(&refiner_decoded)?, values(&refiner_input)?);
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
