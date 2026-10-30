include!(concat!(
    env!("OUT_DIR"),
    "/generated_latent_format_tests.rs"
));

use serde_json::json;
use sha2::{Digest, Sha256};
use sim_model::{
    GENERATED_LATENT_FORMAT_MANIFEST, LatentExtent, LatentFormatDescriptor, LatentFormatIdentity,
    LatentTensorLayout, LatentTransform, empty_latent, process_latent_in, process_latent_out,
    project_latent_preview,
};
use sim_tensor::{CpuWorkspaceAuthority, DType, StreamId, TensorBackend};
use sim_types::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[test]
fn val_latent_001_all_formats_emit_complete_artifact() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(GENERATED_LATENT_FORMAT_MANIFEST.len(), 33);
    assert!(
        GENERATED_LATENT_FORMAT_MANIFEST
            .windows(2)
            .all(|pair| pair[0].0 < pair[1].0)
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?;
    let catalog_path = workspace.join(".agents/specs/comfy-parity/catalogs/backend-models.csv");
    let catalog_bytes = fs::read(&catalog_path)?;
    let catalog_source = std::str::from_utf8(&catalog_bytes)?;
    let catalog_identities = latent_catalog_identities(catalog_source)?;
    let generated_identities = GENERATED_LATENT_FORMAT_MANIFEST
        .iter()
        .map(|(_, definition)| {
            (
                definition.identifier.to_owned(),
                definition.feature_id.to_owned(),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(catalog_identities.len(), 33);
    assert_eq!(generated_identities, catalog_identities);

    let release_gates = [
        "comfy-parity-native-diffusion-e2e",
        "comfy-parity-native-compute-breadth-integration",
        "comfy-parity-final-validation",
    ];
    let task_source = fs::read_to_string(workspace.join(".agents/specs/comfy-parity/tasks.md"))?;
    for gate in release_gates {
        assert!(
            task_source
                .lines()
                .any(|line| line == format!("  - _id: {gate}")),
            "VAL-LATENT-001 release gate is not a durable task id: {gate}"
        );
    }

    let (backend, authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let mut fixture_digests = BTreeMap::new();
    let mut cases = Vec::with_capacity(GENERATED_LATENT_FORMAT_MANIFEST.len());

    for (module, definition) in GENERATED_LATENT_FORMAT_MANIFEST {
        let source_path =
            workspace.join(format!("crates/sim_model/src/latent_formats/{module}.rs"));
        let test_path =
            workspace.join(format!("crates/sim_model/tests/latent_formats/{module}.rs"));
        let source_bytes = fs::read(&source_path)?;
        let test_bytes = fs::read(&test_path)?;
        let test_source = std::str::from_utf8(&test_bytes)?;
        assert!(
            test_source.contains("fn val_latent_001"),
            "{module} has no VAL-LATENT-001 row test"
        );
        let source_digest = format!("{:x}", Sha256::digest(&source_bytes));
        let test_digest = format!("{:x}", Sha256::digest(&test_bytes));
        fixture_digests.insert(
            module.to_string(),
            json!({"source": source_digest, "test": test_digest}),
        );

        let descriptor = LatentFormatDescriptor::checked(definition)?;
        let identity = LatentFormatIdentity::new(definition.feature_id, definition.identifier)?;
        let identity_bytes = serde_json::to_vec(&identity)?;
        let decoded_identity: LatentFormatIdentity = serde_json::from_slice(&identity_bytes)?;
        assert_eq!(decoded_identity, identity);
        assert_eq!(descriptor.identity, identity);
        assert_eq!(descriptor.channels, definition.channels);
        assert_eq!(descriptor.dimensions, definition.dimensions);
        assert_eq!(descriptor.scale_factor, definition.scale_factor);
        assert_eq!(descriptor.shift_factor, definition.shift_factor);

        let extent = representative_extent(
            definition.dimensions,
            definition.spatial_downscale_ratio,
            definition.temporal_downscale_ratio,
        )?;
        let empty = empty_latent(
            definition,
            &backend,
            extent,
            DType::F32,
            StreamId::DEFAULT,
            &context,
        )?;
        assert_eq!(empty.descriptor().dtype(), DType::F32);
        assert_eq!(empty.descriptor().device(), backend.device());
        assert_eq!(empty.descriptor().stream(), StreamId::DEFAULT);
        assert!(
            empty
                .descriptor()
                .shape()
                .iter()
                .all(|dimension| *dimension > 0)
        );
        assert_eq!(
            empty.descriptor().shape()
                [channel_index(definition.layout, empty.descriptor().shape())],
            definition.channels
        );

        let encoded = process_latent_in(definition, &backend, &empty, &context)?;
        let decoded = process_latent_out(definition, &backend, &encoded, &context)?;
        assert_eq!(decoded.descriptor().shape(), empty.descriptor().shape());
        assert_eq!(decoded.descriptor().dtype(), DType::F32);
        assert_eq!(decoded.descriptor().device(), backend.device());
        assert_values_close(&decoded, &empty)?;
        if definition.transform == LatentTransform::Identity {
            assert_eq!(encoded.tensor_id(), empty.tensor_id());
            assert_eq!(encoded.storage_id(), empty.storage_id());
        }

        let f16 = empty_latent(
            definition,
            &backend,
            extent,
            DType::F16,
            StreamId::DEFAULT,
            &context,
        )?;
        assert_eq!(f16.descriptor().dtype(), DType::F16);
        assert_eq!(f16.descriptor().device(), backend.device());
        let f16_behavior = match process_latent_in(definition, &backend, &f16, &context) {
            Ok(processed) => {
                assert_eq!(definition.transform, LatentTransform::Identity);
                assert_eq!(processed.tensor_id(), f16.tensor_id());
                "preserved"
            }
            Err(_) => {
                assert_ne!(definition.transform, LatentTransform::Identity);
                "typed_unavailable"
            }
        };

        let mut invalid_shape = empty.descriptor().shape().to_vec();
        let invalid_channel_index = channel_index(definition.layout, &invalid_shape);
        invalid_shape[invalid_channel_index] = definition
            .channels
            .checked_add(1)
            .ok_or("latent channel count overflow")?;
        let invalid_descriptor = sim_tensor::TensorDescriptor::contiguous(
            invalid_shape,
            DType::F32,
            backend.device(),
            StreamId::DEFAULT,
        )?;
        let (invalid, _) = backend.allocate(invalid_descriptor, &context)?;
        assert!(process_latent_in(definition, &backend, &invalid, &context).is_err());

        let preview_behavior = if definition.preview_factors.is_empty() {
            assert!(project_latent_preview(definition, &backend, &empty, &context).is_err());
            "typed_unavailable"
        } else {
            let preview = project_latent_preview(definition, &backend, &empty, &context)?;
            assert_eq!(preview.descriptor().shape()[1], 3);
            assert_eq!(preview.descriptor().dtype(), DType::F32);
            "projected"
        };

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024 * 1024)?,
            &cancelled,
        );
        assert!(
            empty_latent(
                definition,
                &backend,
                extent,
                DType::F32,
                StreamId::DEFAULT,
                &cancelled_context,
            )
            .is_err()
        );

        cases.push(json!({
            "channels": definition.channels,
            "dimensions": definition.dimensions,
            "empty_shape": empty.descriptor().shape(),
            "feature_id": definition.feature_id,
            "f16_behavior": f16_behavior,
            "identifier": definition.identifier,
            "module": module,
            "preview_behavior": preview_behavior,
            "preview_reshape": format!("{:?}", definition.preview_reshape),
            "scale_factor": definition.scale_factor,
            "shift_factor": definition.shift_factor,
            "source_sha256": source_digest,
            "spatial_downscale_ratio": definition.spatial_downscale_ratio,
            "temporal_downscale_ratio": definition.temporal_downscale_ratio,
            "test_sha256": test_digest,
            "transform": format!("{:?}", definition.transform),
        }));
    }

    let artifact = json!({
        "cases": cases,
        "environment": {
            "architecture": std::env::consts::ARCH,
            "backend": "native-rust-cpu",
            "operating_system": std::env::consts::OS,
        },
        "fixture_digests": {
            "backend_models_catalog": format!("{:x}", Sha256::digest(&catalog_bytes)),
            "row_sources_and_tests": fixture_digests,
        },
        "remaining_release_gates": release_gates,
        "scope": "all 33 generated native latent-format rows COMFY-MODEL-0023 through COMFY-MODEL-0055",
        "skipped": [],
        "summary": {"failed": 0, "passed": 33, "skipped": 0},
        "validation": "VAL-LATENT-001",
        "validation_id": "VAL-LATENT-001",
    });
    let artifact_directory = workspace.join("target/sim-parity");
    fs::create_dir_all(&artifact_directory)?;
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(artifact_directory.join("val-latent-001.json"), bytes)?;
    Ok(())
}

fn latent_catalog_identities(
    catalog: &str,
) -> Result<BTreeSet<(String, String)>, Box<dyn std::error::Error>> {
    let mut identities = BTreeSet::new();
    for line in catalog.lines() {
        let Some(row) = line.strip_prefix("latent format,") else {
            continue;
        };
        let (identifier, _) = row
            .split_once(',')
            .ok_or("latent-format catalog row has no identifier field")?;
        let feature_id = line
            .rsplit_once(',')
            .map(|(_, feature_id)| feature_id)
            .ok_or("latent-format catalog row has no feature-id field")?;
        if identifier.is_empty() || !feature_id.starts_with("COMFY-MODEL-") {
            return Err(format!("invalid latent-format catalog row: {line}").into());
        }
        if !identities.insert((identifier.to_owned(), feature_id.to_owned())) {
            return Err(format!(
                "duplicate latent-format catalog identity: {identifier}/{feature_id}"
            )
            .into());
        }
    }
    Ok(identities)
}

fn representative_extent(
    dimensions: u8,
    spatial_downscale_ratio: u64,
    temporal_downscale_ratio: u64,
) -> Result<LatentExtent, Box<dyn std::error::Error>> {
    let spatial = spatial_downscale_ratio
        .checked_mul(2)
        .ok_or("latent spatial extent overflow")?;
    let temporal = temporal_downscale_ratio
        .checked_mul(3)
        .ok_or("latent temporal extent overflow")?;
    match dimensions {
        1 => Ok(LatentExtent::OneDimensional {
            batch: 1,
            length: temporal,
        }),
        2 => Ok(LatentExtent::TwoDimensional {
            batch: 1,
            width: spatial,
            height: spatial,
        }),
        3 => Ok(LatentExtent::ThreeDimensional {
            batch: 1,
            frames: temporal,
            width: spatial,
            height: spatial,
        }),
        other => Err(format!("unsupported latent dimensions {other}").into()),
    }
}

fn channel_index(layout: LatentTensorLayout, shape: &[u64]) -> usize {
    match layout {
        LatentTensorLayout::ChannelsFirst => 1,
        LatentTensorLayout::SequenceChannelsLast => shape.len() - 1,
    }
}

fn assert_values_close(
    actual: &sim_tensor::Tensor,
    expected: &sim_tensor::Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = actual.descriptor().element_count()?;
    assert_eq!(count, expected.descriptor().element_count()?);
    for index in 0..count {
        let actual_value = f32::from_le_bytes(actual.linear_element_bytes(index)?.try_into()?);
        let expected_value = f32::from_le_bytes(expected.linear_element_bytes(index)?.try_into()?);
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-5,
            "latent round trip differed at element {index}: {actual_value} != {expected_value}"
        );
    }
    Ok(())
}
