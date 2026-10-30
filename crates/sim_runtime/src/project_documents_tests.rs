use crate::assets::{
    AssetIdentity, AssetNamespace, AssetRoots, AssetService, UploadRequest, sha256,
};
use crate::project_documents::{
    ProjectAssetVersion, ProjectAssetVersionReference, ProjectDocumentError,
    ProjectEditableSourceReference, SIM_DOCUMENT_SCHEMA_VERSION, SimDocument, SimEngineDocument,
};
use crate::{
    AssetOperation, AuthorizedCapabilities, Capability, CapabilitySet, OutputCommitReceipt,
    OutputCommitter, OutputOperationState, OutputProposal, PermissionGrant, PermissionPolicy,
};
use serde_json::json;
use sim_tensor::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::PathBuf,
};
use uuid::Uuid;

type TestResult = Result<(), Box<dyn Error>>;

struct Fixture {
    _directory: tempfile::TempDir,
    roots: AssetRoots,
    assets: AssetService,
    committer: OutputCommitter,
    authorization: AuthorizedCapabilities,
}

fn fixture() -> Result<Fixture, Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let mut paths = Vec::new();
    for namespace in [
        AssetNamespace::Input,
        AssetNamespace::Output,
        AssetNamespace::Temporary,
    ] {
        let path = directory.path().join(namespace.locator_type());
        fs::create_dir(&path)?;
        paths.push((namespace, path));
    }
    let roots = AssetRoots::new("e6e6e6e6-3333-4444-5555-666666666666", paths)?;
    let capabilities = CapabilitySet::new(roots.namespaces().flat_map(|namespace| {
        [
            AssetOperation::Read,
            AssetOperation::Write,
            AssetOperation::Delete,
        ]
        .into_iter()
        .map(move |action| Capability::Asset {
            namespace: namespace.locator_type().into(),
            action,
        })
    }));
    let grant = PermissionGrant::new(
        "e6e6e6e6-3333-4444-5555-666666666666",
        "project-document-test",
        capabilities.clone(),
        "explicit-project-png-test-grant",
    )?;
    let authorization = PermissionPolicy::new("e6e6e6e6-3333-4444-5555-666666666666", [grant])?
        .authorize("project-document-test", &capabilities)?;
    Ok(Fixture {
        assets: AssetService::open(roots.clone())?,
        committer: OutputCommitter::open(roots.clone())?,
        roots,
        authorization,
        _directory: directory,
    })
}

fn encoded_png(red: f32) -> Result<Vec<u8>, Box<dyn Error>> {
    let pixels = [
        red, 0.1, 0.9, 0., 0.2, 0.8, 0.3, 1., 0.1, 0.2, 0.7, 0.5, 0.9, 0.1, 0.2, 1.,
    ];
    let bytes = sim_media::encode_png_frame(
        &pixels,
        1,
        2,
        2,
        4,
        0,
        &BTreeMap::from([("workflow".into(), "project-document-test-only".into())]),
        sim_media::PngLimits::default(),
    )?;
    let decoded = sim_media::decode_png(&bytes, sim_media::PngLimits::default())?;
    assert_eq!((decoded.width, decoded.height), (2, 2));
    assert!(decoded.has_alpha);
    Ok(bytes)
}

fn upload(
    fixture: &mut Fixture,
    bytes: Vec<u8>,
    overwrite: bool,
) -> Result<AssetIdentity, Box<dyn Error>> {
    Ok(fixture
        .assets
        .upload(
            UploadRequest {
                namespace: AssetNamespace::Input,
                filename: "linked.png".into(),
                subfolder: PathBuf::from("editable-inputs"),
                bytes,
                overwrite,
                tags: BTreeSet::new(),
            },
            &fixture.authorization,
            &CancellationToken::default(),
        )?
        .record
        .identity)
}

fn pin(
    fixture: &mut Fixture,
    identity: &AssetIdentity,
    bytes: &[u8],
) -> Result<ProjectAssetVersion, Box<dyn Error>> {
    Ok(fixture.assets.pin_project_png_version(
        &mut fixture.committer,
        identity,
        &sha256(bytes),
        &fixture.authorization,
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
        1024 * 1024,
    )?)
}

fn publish(fixture: &mut Fixture, bytes: Vec<u8>) -> Result<OutputCommitReceipt, Box<dyn Error>> {
    let proposal = OutputProposal::new(
        Uuid::new_v4(),
        AssetNamespace::Output,
        "derived/graphite-baked",
        "png",
        0,
        2,
        2,
        bytes,
    )?
    .with_projection_metadata(serde_json::to_vec(
        &json!({"schema":"test-only-project-provenance", "ordered_inputs":[]}),
    )?)?;
    let proposal_id = proposal.proposal_id();
    let receipts = fixture.committer.commit_proposal_batch_and_register(
        &[proposal],
        chrono::DateTime::parse_from_rfc3339("2026-10-09T10:00:00+00:00")?,
        &mut fixture.assets,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(receipts.len(), 1);
    let receipt = receipts
        .into_iter()
        .next()
        .ok_or("missing canonical receipt")?;
    assert_eq!(receipt.proposal_id(), proposal_id);
    assert_eq!(receipt.operation().state, OutputOperationState::Committed);
    assert!(receipt.operation().execution_scope.is_none());
    Ok(receipt)
}

#[test]
fn project_png_pin_changed_input_retains_both_versions_after_service_reopen() -> TestResult {
    let mut fixture = fixture()?;
    let first_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, first_bytes.clone(), false)?;
    let first = pin(&mut fixture, &identity, &first_bytes)?;
    let repeated = pin(&mut fixture, &identity, &first_bytes)?;
    assert_eq!(first, repeated);
    let second_bytes = encoded_png(0.75)?;
    assert_ne!(first_bytes, second_bytes);
    let replaced_identity = upload(&mut fixture, second_bytes.clone(), true)?;
    assert_eq!(identity, replaced_identity);
    let second = pin(&mut fixture, &identity, &second_bytes)?;
    assert_eq!(first.reference.asset_id, second.reference.asset_id);
    assert_ne!(first.reference, second.reference);
    assert_ne!(first.stored_identity, second.stored_identity);
    assert_eq!(
        fixture.assets.read_verified(
            &identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        second_bytes
    );
    let reopened = AssetService::open(fixture.roots.clone())?;
    for (version, bytes) in [(&first, &first_bytes), (&second, &second_bytes)] {
        let resolved = reopened.resolve_project_asset_version(
            &version.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024,
        )?;
        assert_eq!(resolved.version(), version);
        assert_eq!(resolved.bytes(), bytes);
        assert_eq!(resolved.version().reference.content_sha256, sha256(bytes));
    }
    let reopened_committer = OutputCommitter::open(fixture.roots)?;
    assert!(
        reopened_committer
            .operations()
            .iter()
            .all(|operation| operation.state == OutputOperationState::Committed)
    );
    assert_eq!(reopened_committer.operations().len(), 2);
    Ok(())
}

#[test]
fn project_png_missing_and_tampered_versions_fail_without_reading_changed_input() -> TestResult {
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let version = pin(&mut fixture, &identity, &bytes)?;
    let absent = ProjectAssetVersionReference {
        asset_id: Uuid::new_v4(),
        version_id: Uuid::new_v4(),
        content_sha256: version.reference.content_sha256.clone(),
    };
    assert!(
        matches!(fixture.assets.resolve_project_asset_version(&absent, &fixture.authorization, &CancellationToken::default(), 1024 * 1024), Err(ProjectDocumentError::MissingVersion(reference)) if reference == absent)
    );
    let stored = fixture
        .roots
        .test_root_path(AssetNamespace::Output)?
        .join(&version.stored_identity.relative_path);
    fs::write(&stored, encoded_png(0.75)?)?;
    assert!(matches!(
        fixture.assets.resolve_project_asset_version(
            &version.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(_)) | Err(ProjectDocumentError::VersionIntegrity)
    ));
    fs::remove_file(&stored)?;
    assert!(matches!(
        fixture.assets.resolve_project_asset_version(
            &version.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(_))
    ));
    assert_eq!(
        fixture.assets.read_verified(
            &identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        bytes
    );
    Ok(())
}

fn document(reference: ProjectAssetVersionReference) -> SimDocument {
    SimDocument {
        format: "sim".into(),
        schema_version: SIM_DOCUMENT_SCHEMA_VERSION,
        project_id: Uuid::parse_str("e6e6e6e6-3333-4444-5555-666666666666")
            .expect("fixed project fixture UUID"),
        documents: vec![SimEngineDocument::new(
            Uuid::new_v4(),
            "test-only-opaque-engine".into(),
            "test-only-source-pin".into(),
            "opaque-test-fixture".into(),
            "{\"editable_test_value\":1}".into(),
            vec![reference.clone()],
        )],
        build_outputs: vec![reference],
        provenance: json!({"qualification":"container fixture; not Graphite legacy admission"}),
    }
}

#[test]
fn sim_container_roundtrip_rejects_schema_payload_and_dependency_tamper() -> TestResult {
    let reference = ProjectAssetVersionReference {
        asset_id: Uuid::new_v4(),
        version_id: Uuid::new_v4(),
        content_sha256: "a".repeat(64),
    };
    let original = document(reference);
    let encoded = original.encode()?;
    assert_eq!(SimDocument::decode(&encoded)?, original);
    let mut changed = original.clone();
    changed.schema_version += 1;
    assert!(matches!(
        SimDocument::decode(&serde_json::to_vec(&changed)?),
        Err(ProjectDocumentError::UnsupportedSchema)
    ));
    let mut changed = original.clone();
    changed
        .documents
        .first_mut()
        .ok_or("missing engine")?
        .payload
        .push('x');
    assert!(matches!(
        SimDocument::decode(&serde_json::to_vec(&changed)?),
        Err(ProjectDocumentError::InvalidDocument)
    ));
    let mut duplicate = original.clone();
    duplicate
        .documents
        .push(original.documents.first().ok_or("missing engine")?.clone());
    assert!(matches!(
        duplicate.encode(),
        Err(ProjectDocumentError::InvalidDocument)
    ));
    let mut duplicate = original.clone();
    let dependency = duplicate
        .documents
        .first()
        .ok_or("missing engine")?
        .dependencies
        .first()
        .ok_or("missing dependency")?
        .clone();
    duplicate
        .documents
        .first_mut()
        .ok_or("missing engine")?
        .dependencies
        .push(dependency);
    assert!(matches!(
        duplicate.encode(),
        Err(ProjectDocumentError::InvalidDocument)
    ));
    let mut malformed = serde_json::to_value(original)?;
    malformed
        .as_object_mut()
        .ok_or("missing object")?
        .remove("documents");
    assert!(matches!(
        SimDocument::decode(&serde_json::to_vec(&malformed)?),
        Err(ProjectDocumentError::Json(_))
    ));
    Ok(())
}

#[test]
fn sim_container_rejects_duplicate_build_output_provenance() -> TestResult {
    let reference = ProjectAssetVersionReference {
        asset_id: Uuid::new_v4(),
        version_id: Uuid::new_v4(),
        content_sha256: "a".repeat(64),
    };
    let mut container = document(reference.clone());
    container.build_outputs.push(reference);
    assert!(matches!(
        container.encode(),
        Err(ProjectDocumentError::InvalidDocument)
    ));
    Ok(())
}

#[test]
fn project_png_derived_version_requires_canonical_receipt_and_retains_ordered_provenance()
-> TestResult {
    let mut fixture = fixture()?;
    let first_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, first_bytes.clone(), false)?;
    let first = pin(&mut fixture, &identity, &first_bytes)?;
    let second_bytes = encoded_png(0.75)?;
    upload(&mut fixture, second_bytes.clone(), true)?;
    let second = pin(&mut fixture, &identity, &second_bytes)?;
    let derived_bytes = encoded_png(0.5)?;
    let receipt = publish(&mut fixture, derived_bytes.clone())?;
    let asset_id = Uuid::new_v4();
    let dependencies = vec![second.reference.clone(), first.reference.clone()];
    let provenance =
        json!({"kind":"derived-test", "ordered_inputs":[second.reference, first.reference]});
    let version = fixture.assets.register_project_png_version(
        &receipt,
        identity.clone(),
        asset_id,
        dependencies.clone(),
        None,
        provenance.clone(),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(version.stored_identity, receipt.operation().identity);
    assert_eq!(version.reference.content_sha256, receipt.operation().sha256);
    assert_eq!(version.byte_size, receipt.operation().byte_size);
    assert_eq!(
        fixture
            .assets
            .record(&version.stored_identity)
            .ok_or("missing canonical asset index record")?
            .sha256,
        version.reference.content_sha256
    );
    let repeated = fixture.assets.register_project_png_version(
        &receipt,
        identity.clone(),
        asset_id,
        dependencies.clone(),
        None,
        provenance.clone(),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(version, repeated);
    let resolved = AssetService::open(fixture.roots.clone())?.resolve_project_asset_version(
        &version.reference,
        &fixture.authorization,
        &CancellationToken::default(),
        1024 * 1024,
    )?;
    assert_eq!(resolved.bytes(), derived_bytes);
    assert_eq!(resolved.version().dependencies, dependencies);
    assert_eq!(resolved.version().provenance, provenance);
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            asset_id,
            vec![first.reference, second.reference],
            None,
            provenance,
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    Ok(())
}

#[test]
fn project_png_rejects_duplicate_derived_dependencies_without_catalog_poisoning() -> TestResult {
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &bytes)?;
    let receipt = publish(&mut fixture, encoded_png(0.5)?)?;
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            Uuid::new_v4(),
            vec![input.reference.clone(), input.reference.clone()],
            None,
            json!({"kind":"derived-test"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    assert_eq!(
        fixture
            .assets
            .resolve_project_asset_version(
                &input.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        bytes
    );
    Ok(())
}

#[test]
fn project_png_rejects_foreign_editable_source_before_catalog_mutation() -> TestResult {
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &bytes)?;
    let receipt = publish(&mut fixture, encoded_png(0.5)?)?;
    let mut foreign = receipt.operation().identity.clone();
    foreign.profile_id = "other-project".into();
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            Uuid::new_v4(),
            vec![input.reference.clone()],
            Some(ProjectEditableSourceReference {
                identity: foreign,
                content_sha256: "a".repeat(64)
            }),
            json!({"kind":"derived-test"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    assert_eq!(
        fixture
            .assets
            .resolve_project_asset_version(
                &input.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        bytes
    );
    Ok(())
}

#[test]
fn project_png_pre_cancelled_pin_and_canonical_publication_leave_no_committed_output() -> TestResult
{
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        fixture.assets.pin_project_png_version(
            &mut fixture.committer,
            &identity,
            &sha256(&bytes),
            &fixture.authorization,
            &fixture.authorization,
            &fixture.authorization,
            &cancelled,
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(
            crate::assets::AssetError::Cancelled
        ))
    ));
    assert!(fixture.committer.operations().is_empty());
    let proposal = OutputProposal::new(
        Uuid::new_v4(),
        AssetNamespace::Output,
        "cancelled/derived",
        "png",
        0,
        2,
        2,
        bytes,
    )?;
    assert!(matches!(
        fixture.committer.commit_proposal_batch_and_register(
            &[proposal],
            chrono::DateTime::parse_from_rfc3339("2026-10-09T10:00:00+00:00")?,
            &mut fixture.assets,
            &fixture.authorization,
            &cancelled
        ),
        Err(crate::OutputCommitError::Asset(
            crate::assets::AssetError::Cancelled
        ))
    ));
    assert!(fixture.committer.operations().is_empty());
    let reopened = OutputCommitter::open(fixture.roots)?;
    assert!(reopened.operations().is_empty());
    Ok(())
}

#[test]
fn project_png_retains_verified_editable_source_digest_and_container_dependencies() -> TestResult {
    let mut fixture = fixture()?;
    let input_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, input_bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &input_bytes)?;
    let container = document(input.reference.clone());
    let source_bytes = container.encode()?;
    let source_proposal = OutputProposal::new(
        Uuid::new_v4(),
        AssetNamespace::Output,
        "editable/source",
        "sim",
        0,
        0,
        0,
        source_bytes.clone(),
    )?;
    let source_receipts = fixture.committer.commit_proposal_batch_and_register(
        &[source_proposal],
        chrono::DateTime::parse_from_rfc3339("2026-10-09T10:00:00+00:00")?,
        &mut fixture.assets,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    let source_receipt = source_receipts
        .first()
        .ok_or("missing canonical source receipt")?;
    let editable = ProjectEditableSourceReference {
        identity: source_receipt.operation().identity.clone(),
        content_sha256: source_receipt.operation().sha256.clone(),
    };
    assert_eq!(editable.content_sha256, sha256(&source_bytes));
    let derived_receipt = publish(&mut fixture, encoded_png(0.5)?)?;
    let version = fixture.assets.register_project_png_version(
        &derived_receipt,
        identity.clone(),
        Uuid::new_v4(),
        vec![input.reference.clone()],
        Some(editable.clone()),
        json!({"kind":"derived-test"}),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    let reopened = AssetService::open(fixture.roots.clone())?;
    let resolved = reopened.resolve_project_asset_version(
        &version.reference,
        &fixture.authorization,
        &CancellationToken::default(),
        1024 * 1024,
    )?;
    assert_eq!(resolved.version().editable_source, Some(editable.clone()));
    let source = reopened.read_verified(
        &editable.identity,
        &fixture.authorization,
        &CancellationToken::default(),
        1024 * 1024,
    )?;
    assert_eq!(source, source_bytes);
    assert_eq!(SimDocument::decode(&source)?, container);
    let mut wrong_digest = editable;
    wrong_digest.content_sha256 = "0".repeat(64);
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &derived_receipt,
            identity,
            Uuid::new_v4(),
            vec![input.reference.clone()],
            Some(wrong_digest),
            json!({"kind":"derived-test"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::VersionIntegrity)
    ));
    assert!(
        fixture
            .assets
            .resolve_project_asset_version(
                &input.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )
            .is_ok()
    );
    Ok(())
}

#[test]
fn project_png_invalid_png_registration_and_pin_fail_without_creating_version() -> TestResult {
    let mut fixture = fixture()?;
    let bytes = b"not a PNG despite the extension".to_vec();
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    assert!(matches!(
        fixture.assets.pin_project_png_version(
            &mut fixture.committer,
            &identity,
            &sha256(&bytes),
            &fixture.authorization,
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    assert!(fixture.committer.operations().is_empty());
    let receipt = publish(&mut fixture, bytes)?;
    let asset_id = Uuid::new_v4();
    let reference = ProjectAssetVersionReference {
        asset_id,
        version_id: receipt.operation().operation_id,
        content_sha256: receipt.operation().sha256.clone(),
    };
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            asset_id,
            Vec::new(),
            None,
            json!({"kind":"invalid-test"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    assert!(matches!(
        fixture.assets.resolve_project_asset_version(
            &reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::MissingVersion(_))
    ));
    assert_eq!(receipt.operation().state, OutputOperationState::Committed);
    Ok(())
}

#[test]
fn project_png_catalog_write_failure_does_not_claim_rollback_of_existing_publication() -> TestResult
{
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &bytes)?;
    let derived_bytes = encoded_png(0.5)?;
    let receipt = publish(&mut fixture, derived_bytes.clone())?;
    let asset_id = Uuid::new_v4();
    let reference = ProjectAssetVersionReference {
        asset_id,
        version_id: receipt.operation().operation_id,
        content_sha256: receipt.operation().sha256.clone(),
    };
    let over_budget =
        json!({"diagnostic":"x".repeat(crate::project_documents::MAX_SIM_DOCUMENT_BYTES + 1)});
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            asset_id,
            vec![input.reference.clone()],
            None,
            over_budget,
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::TooLarge)
    ));
    let reopened = AssetService::open(fixture.roots.clone())?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::MissingVersion(_))
    ));
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &input.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        bytes
    );
    assert_eq!(
        reopened.read_verified(
            &receipt.operation().identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        derived_bytes
    );
    assert_eq!(
        OutputCommitter::open(fixture.roots)?
            .operation(receipt.operation().operation_id)
            .ok_or("missing journaled publication")?
            .state,
        OutputOperationState::Committed
    );
    Ok(())
}

#[test]
fn project_png_header_dimension_limit_and_truncated_png_are_rejected_before_version_registration()
-> TestResult {
    let mut fixture = fixture()?;
    let wide = sim_media::encode_png_frame(
        &vec![0.5; 32769 * 3],
        1,
        1,
        32769,
        3,
        0,
        &BTreeMap::new(),
        sim_media::PngLimits {
            max_dimension: 32769,
            ..sim_media::PngLimits::default()
        },
    )?;
    let decoded = sim_media::decode_png(
        &wide,
        sim_media::PngLimits {
            max_dimension: 32769,
            ..sim_media::PngLimits::default()
        },
    )?;
    assert_eq!((decoded.width, decoded.height), (32769, 1));
    drop(decoded);
    let identity = upload(&mut fixture, wide.clone(), false)?;
    assert!(matches!(
        fixture.assets.pin_project_png_version(
            &mut fixture.committer,
            &identity,
            &sha256(&wide),
            &fixture.authorization,
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    assert!(fixture.committer.operations().is_empty());
    let receipt = publish(&mut fixture, wide)?;
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            Uuid::new_v4(),
            Vec::new(),
            None,
            json!({"kind":"invalid-dimensions-test"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    let valid = encoded_png(0.5)?;
    let truncated = valid.get(..24).ok_or("real PNG header missing")?.to_vec();
    let identity = upload(&mut fixture, truncated.clone(), true)?;
    assert!(matches!(
        fixture.assets.pin_project_png_version(
            &mut fixture.committer,
            &identity,
            &sha256(&truncated),
            &fixture.authorization,
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    Ok(())
}

#[test]
fn project_png_version_and_catalog_storage_reject_generic_output_overwrite() -> TestResult {
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let version = pin(&mut fixture, &identity, &bytes)?;
    let filename = version
        .stored_identity
        .filename()
        .ok_or("version filename missing")?
        .to_owned();
    let subfolder = version
        .stored_identity
        .relative_path
        .parent()
        .ok_or("version directory missing")?
        .to_owned();
    let overwrite = fixture.assets.upload(
        UploadRequest {
            namespace: AssetNamespace::Output,
            filename,
            subfolder,
            bytes: encoded_png(0.75)?,
            overwrite: true,
            tags: BTreeSet::new(),
        },
        &fixture.authorization,
        &CancellationToken::default(),
    );
    assert!(matches!(
        overwrite,
        Err(crate::assets::AssetError::ReservedAsset(_))
    ));
    let overwrite_catalog = fixture.assets.upload(
        UploadRequest {
            namespace: AssetNamespace::Output,
            filename: "project-asset-versions.json".into(),
            subfolder: PathBuf::from(".sim"),
            bytes: b"{}".to_vec(),
            overwrite: true,
            tags: BTreeSet::new(),
        },
        &fixture.authorization,
        &CancellationToken::default(),
    );
    assert!(matches!(
        overwrite_catalog,
        Err(crate::assets::AssetError::ReservedAsset(_))
    ));
    assert_eq!(
        AssetService::open(fixture.roots)?
            .resolve_project_asset_version(
                &version.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        bytes
    );
    Ok(())
}

fn publish_editable_container(
    fixture: &mut Fixture,
    container: &SimDocument,
) -> Result<ProjectEditableSourceReference, Box<dyn Error>> {
    let proposal = OutputProposal::new(
        Uuid::new_v4(),
        AssetNamespace::Output,
        "editable/focused-source",
        "sim",
        0,
        0,
        0,
        container.encode()?,
    )?;
    let receipts = fixture.committer.commit_proposal_batch_and_register(
        &[proposal],
        chrono::DateTime::parse_from_rfc3339("2026-10-10T10:00:00+00:00")?,
        &mut fixture.assets,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    let receipt = receipts.first().ok_or("missing editable source receipt")?;
    Ok(ProjectEditableSourceReference {
        identity: receipt.operation().identity.clone(),
        content_sha256: receipt.operation().sha256.clone(),
    })
}

#[test]
fn project_png_canonical_removal_rejects_versions_sources_and_catalog_before_moving_bytes()
-> TestResult {
    let mut fixture = fixture()?;
    let input_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, input_bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &input_bytes)?;
    let container = document(input.reference.clone());
    let editable = publish_editable_container(&mut fixture, &container)?;
    let derived_bytes = encoded_png(0.5)?;
    let receipt = publish(&mut fixture, derived_bytes.clone())?;
    let derived = fixture.assets.register_project_png_version(
        &receipt,
        receipt.operation().identity.clone(),
        Uuid::new_v4(),
        vec![input.reference.clone()],
        Some(editable.clone()),
        json!({"kind":"removal-guard-control"}),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    let catalog = fixture
        .roots
        .identity(AssetNamespace::Output, ".sim/project-asset-versions.json")?;
    let protected = [
        input.stored_identity.clone(),
        derived.stored_identity.clone(),
        editable.identity,
        catalog,
    ];
    let operation_count = fixture.committer.operations().len();
    for identity in &protected {
        let path = fixture
            ._directory
            .path()
            .join(identity.namespace.locator_type())
            .join(&identity.relative_path);
        let before = fs::read(&path)?;
        assert!(matches!(
            fixture.committer.prepare_removal(
                identity,
                &fixture.authorization,
                &CancellationToken::default()
            ),
            Err(crate::OutputCommitError::Asset(crate::assets::AssetError::ReservedAsset(rejected)))
                if rejected == *identity
        ));
        assert_eq!(fs::read(path)?, before);
        assert!(fixture.committer.pending_removals().is_empty());
        assert_eq!(fixture.committer.operations().len(), operation_count);
    }
    let reopened = AssetService::open(fixture.roots.clone())?;
    for (version, bytes) in [(&input, &input_bytes), (&derived, &derived_bytes)] {
        assert_eq!(
            reopened
                .resolve_project_asset_version(
                    &version.reference,
                    &fixture.authorization,
                    &CancellationToken::default(),
                    1024 * 1024
                )?
                .bytes(),
            bytes
        );
    }
    assert!(
        OutputCommitter::open(fixture.roots.clone())?
            .pending_removals()
            .is_empty()
    );
    let ordinary = publish(&mut fixture, encoded_png(0.75)?)?;
    let removal = fixture.committer.prepare_removal(
        &ordinary.operation().identity,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    let committed = fixture.committer.commit_removal_and_register(
        removal.operation_id,
        &mut fixture.assets,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(committed.state, OutputOperationState::Committed);
    Ok(())
}

#[test]
fn project_png_distinct_same_content_receipts_retain_separate_versions_and_provenance() -> TestResult
{
    let mut fixture = fixture()?;
    let input_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, input_bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &input_bytes)?;
    let derived_bytes = encoded_png(0.5)?;
    let asset_id = Uuid::new_v4();
    let mut versions = Vec::new();
    for export_sequence in 0..2 {
        let receipt = publish(&mut fixture, derived_bytes.clone())?;
        let provenance = json!({"kind":"repeated-export-control", "sequence":export_sequence,
            "ordered_inputs":[input.reference.clone()]});
        let version = fixture.assets.register_project_png_version(
            &receipt,
            receipt.operation().identity.clone(),
            asset_id,
            vec![input.reference.clone()],
            None,
            provenance.clone(),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default(),
        )?;
        assert_eq!(
            version.reference.version_id,
            receipt.operation().operation_id
        );
        assert_eq!(version.reference.content_sha256, sha256(&derived_bytes));
        assert_eq!(version.provenance, provenance);
        versions.push(version);
    }
    let [first, second] = versions.as_slice() else {
        return Err("missing two canonical versions".into());
    };
    assert_eq!(first.reference.asset_id, second.reference.asset_id);
    assert_eq!(
        first.reference.content_sha256,
        second.reference.content_sha256
    );
    assert_ne!(first.reference.version_id, second.reference.version_id);
    assert_ne!(first.stored_identity, second.stored_identity);
    assert_ne!(first.provenance, second.provenance);
    let reopened = AssetService::open(fixture.roots.clone())?;
    for version in &versions {
        let resolved = reopened.resolve_project_asset_version(
            &version.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024,
        )?;
        assert_eq!(resolved.version(), version);
        assert_eq!(resolved.bytes(), derived_bytes);
        assert_eq!(
            resolved.version().dependencies,
            vec![input.reference.clone()]
        );
    }
    Ok(())
}

#[test]
fn project_png_rejects_same_root_editable_source_with_another_project_identity() -> TestResult {
    let mut fixture = fixture()?;
    let bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &bytes)?;
    let mut container = document(input.reference.clone());
    container.project_id = Uuid::new_v4();
    assert_ne!(container.project_id.to_string(), fixture.roots.profile_id);
    let editable = publish_editable_container(&mut fixture, &container)?;
    let receipt = publish(&mut fixture, encoded_png(0.5)?)?;
    let asset_id = Uuid::new_v4();
    let reference = ProjectAssetVersionReference {
        asset_id,
        version_id: receipt.operation().operation_id,
        content_sha256: receipt.operation().sha256.clone(),
    };
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            asset_id,
            vec![input.reference],
            Some(editable.clone()),
            json!({"kind":"mismatched-editable-project-control"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    let reopened = AssetService::open(fixture.roots.clone())?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::MissingVersion(_))
    ));
    assert_eq!(
        SimDocument::decode(&reopened.read_verified(
            &editable.identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?)?,
        container
    );
    Ok(())
}

#[test]
fn project_png_rejects_derived_dependency_absent_from_editable_engine_links() -> TestResult {
    let mut fixture = fixture()?;
    let first_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, first_bytes.clone(), false)?;
    let first = pin(&mut fixture, &identity, &first_bytes)?;
    let second_bytes = encoded_png(0.75)?;
    upload(&mut fixture, second_bytes.clone(), true)?;
    let second = pin(&mut fixture, &identity, &second_bytes)?;
    let mut container = document(first.reference.clone());
    container.build_outputs.push(second.reference.clone());
    let editable = publish_editable_container(&mut fixture, &container)?;
    let receipt = publish(&mut fixture, encoded_png(0.5)?)?;
    let asset_id = Uuid::new_v4();
    let reference = ProjectAssetVersionReference {
        asset_id,
        version_id: receipt.operation().operation_id,
        content_sha256: receipt.operation().sha256.clone(),
    };
    assert!(matches!(
        fixture.assets.register_project_png_version(
            &receipt,
            identity,
            asset_id,
            vec![second.reference.clone()],
            Some(editable.clone()),
            json!({"kind":"missing-editable-dependency-control"}),
            &fixture.authorization,
            &fixture.authorization,
            &CancellationToken::default()
        ),
        Err(ProjectDocumentError::InvalidVersion)
    ));
    let reopened = AssetService::open(fixture.roots.clone())?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::MissingVersion(_))
    ));
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &second.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        second_bytes
    );
    let valid = fixture.assets.register_project_png_version(
        &receipt,
        receipt.operation().identity.clone(),
        asset_id,
        vec![first.reference.clone()],
        Some(editable),
        json!({"kind":"matching-editable-dependency-recovery"}),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(valid.dependencies, vec![first.reference]);
    Ok(())
}

#[test]
fn project_png_writer_pool_reuses_a_b_a_handles_and_retains_recoverable_publications() -> TestResult
{
    let Fixture {
        _directory: first_directory,
        roots: first_roots,
        assets: mut first_assets,
        committer: first_unpooled,
        authorization: first_authorization,
    } = fixture()?;
    let Fixture {
        _directory: second_directory,
        roots: second_roots,
        assets: mut second_assets,
        committer: second_unpooled,
        authorization: second_authorization,
    } = fixture()?;
    drop(first_unpooled);
    drop(second_unpooled);
    assert_eq!(first_roots.profile_id, second_roots.profile_id);
    assert_ne!(first_roots, second_roots);
    let mut pool = crate::ProjectOutputCommitterPool::default();
    let first_writer = pool.for_roots(first_roots.clone())?;
    let second_writer = pool.for_roots(second_roots.clone())?;
    assert!(!std::sync::Arc::ptr_eq(&first_writer, &second_writer));
    let publish = |writer: &crate::SharedOutputCommitter,
                   assets: &mut AssetService,
                   authorization: &AuthorizedCapabilities,
                   bytes: Vec<u8>|
     -> Result<OutputCommitReceipt, Box<dyn Error>> {
        let proposal = OutputProposal::new(
            Uuid::new_v4(),
            AssetNamespace::Output,
            "pooled/shared-prefix",
            "png",
            0,
            2,
            2,
            bytes,
        )?;
        let mut committer = writer.lock().map_err(|_| "pooled writer was poisoned")?;
        let receipts = committer.commit_proposal_batch_and_register(
            &[proposal],
            chrono::DateTime::parse_from_rfc3339("2026-10-10T10:00:00+00:00")?,
            assets,
            authorization,
            &CancellationToken::default(),
        )?;
        assert_eq!(receipts.len(), 1);
        receipts
            .into_iter()
            .next()
            .ok_or_else(|| "missing pooled receipt".into())
    };
    let first_bytes = encoded_png(0.25)?;
    let second_bytes = encoded_png(0.5)?;
    let third_bytes = encoded_png(0.75)?;
    let first_receipt = publish(
        &first_writer,
        &mut first_assets,
        &first_authorization,
        first_bytes.clone(),
    )?;
    let second_receipt = publish(
        &second_writer,
        &mut second_assets,
        &second_authorization,
        second_bytes.clone(),
    )?;
    let returned_first_writer = pool.for_roots(first_roots.clone())?;
    assert!(std::sync::Arc::ptr_eq(
        &first_writer,
        &returned_first_writer
    ));
    let third_receipt = publish(
        &returned_first_writer,
        &mut first_assets,
        &first_authorization,
        third_bytes.clone(),
    )?;
    assert_ne!(
        first_receipt.operation().identity,
        third_receipt.operation().identity
    );
    assert!(
        third_receipt.operation().collision_counter > first_receipt.operation().collision_counter
    );
    for writer in [&first_writer, &returned_first_writer] {
        let committer = writer.lock().map_err(|_| "pooled writer was poisoned")?;
        assert_eq!(committer.operations().len(), 2);
        for receipt in [&first_receipt, &third_receipt] {
            let operation = committer
                .operation(receipt.operation().operation_id)
                .ok_or("first project journal lost an operation")?;
            assert_eq!(operation.state, OutputOperationState::Committed);
            assert_eq!(operation.identity, receipt.operation().identity);
            assert_eq!(operation.sha256, receipt.operation().sha256);
        }
        assert!(
            committer
                .operation(second_receipt.operation().operation_id)
                .is_none()
        );
    }
    assert_eq!(
        second_writer
            .lock()
            .map_err(|_| "pooled writer was poisoned")?
            .operations()
            .len(),
        1
    );
    drop(pool);
    drop(first_writer);
    drop(returned_first_writer);
    drop(second_writer);
    drop(first_assets);
    drop(second_assets);
    for (roots, authorization, expected) in [
        (
            first_roots,
            first_authorization,
            vec![(first_receipt, first_bytes), (third_receipt, third_bytes)],
        ),
        (
            second_roots,
            second_authorization,
            vec![(second_receipt, second_bytes)],
        ),
    ] {
        let recovered = OutputCommitter::open(roots.clone())?;
        let assets = AssetService::open(roots)?;
        assert_eq!(recovered.operations().len(), expected.len());
        for (receipt, bytes) in expected {
            let operation = recovered
                .operation(receipt.operation().operation_id)
                .ok_or("reopened project journal lost an operation")?;
            assert_eq!(operation.state, OutputOperationState::Committed);
            assert_eq!(operation.identity, receipt.operation().identity);
            assert_eq!(operation.sha256, sha256(&bytes));
            assert_eq!(
                assets.read_verified(
                    &operation.identity,
                    &authorization,
                    &CancellationToken::default(),
                    1024 * 1024,
                )?,
                bytes
            );
        }
    }
    drop(first_directory);
    drop(second_directory);
    Ok(())
}

#[test]
fn project_png_native_controller_retains_canonical_writer_and_rejects_other_physical_roots()
-> TestResult {
    let mut first_fixture = fixture()?;
    let first_receipt = publish(&mut first_fixture, encoded_png(0.25)?)?;
    let mut second_fixture = fixture()?;
    let second_receipt = publish(&mut second_fixture, encoded_png(0.5)?)?;
    let Fixture {
        _directory: first_directory,
        roots: first_roots,
        assets: first_assets,
        committer: first_unpooled,
        authorization: first_authorization,
    } = first_fixture;
    let Fixture {
        _directory: second_directory,
        roots: second_roots,
        assets: second_assets,
        committer: second_unpooled,
        authorization: second_authorization,
    } = second_fixture;
    drop(first_unpooled);
    drop(second_unpooled);
    assert_eq!(first_roots.profile_id, second_roots.profile_id);
    assert_ne!(first_roots, second_roots);
    let first_assets = std::sync::Arc::new(std::sync::Mutex::new(first_assets));
    let second_assets = std::sync::Arc::new(std::sync::Mutex::new(second_assets));
    let mut pool = crate::ProjectOutputCommitterPool::default();
    let first_writer = pool.for_roots(first_roots.clone())?;
    let second_writer = pool.for_roots(second_roots.clone())?;
    let profile_id = crate::ProfileId(Uuid::parse_str(&first_roots.profile_id)?);
    let mut presentation_service = crate::ExecutionPresentationService::new(8)?;
    presentation_service.initialize_profile(
        profile_id,
        crate::ExecutionDataSource::Live,
        crate::ExecutionSnapshotStatus::Ready,
    )?;
    let presentation = crate::ExecutionPresentationOwner::ephemeral(presentation_service);
    let unused_binary = first_directory.path().join("worker-must-not-be-launched");
    assert!(!unused_binary.exists());
    let worker = || {
        crate::WorkerLaunchConfig::new(
            unused_binary.clone(),
            profile_id,
            sim_types::WorkerId(Uuid::from_u128(0x5734)),
            crate::NATIVE_IMAGE_REGISTRY_VERSION,
            1024,
        )
    };
    let config = crate::NativeExecutionControllerConfig::new_with_output_committer(
        first_assets.clone(),
        presentation.clone(),
        worker(),
        true,
        first_writer.clone(),
    )?;
    assert!(std::sync::Arc::ptr_eq(
        &config.output_committer,
        &first_writer
    ));
    assert!(std::sync::Arc::ptr_eq(&config.assets, &first_assets));
    assert!(std::sync::Arc::ptr_eq(&config.presentation, &presentation));
    assert!(matches!(
        crate::NativeExecutionControllerConfig::new_with_output_committer(
            first_assets.clone(),
            presentation,
            worker(),
            true,
            second_writer.clone()
        ),
        Err(crate::NativeImageRuntimeError::Asset(message))
            if message == "canonical publisher and project asset roots differ"
    ));
    for (writer, receipt) in [
        (&first_writer, &first_receipt),
        (&second_writer, &second_receipt),
    ] {
        let committer = writer.lock().map_err(|_| "canonical writer was poisoned")?;
        assert_eq!(
            committer.operations(),
            std::slice::from_ref(receipt.operation())
        );
        assert_eq!(committer.operations().len(), 1);
        let operation = committer
            .operation(receipt.operation().operation_id)
            .ok_or("configuration changed the existing publisher journal")?;
        assert_eq!(operation.state, OutputOperationState::Committed);
        assert_eq!(operation.identity, receipt.operation().identity);
        assert_eq!(operation.sha256, receipt.operation().sha256);
        assert!(committer.pending_removals().is_empty());
    }
    let new_bytes = encoded_png(0.75)?;
    let proposal = OutputProposal::new(
        Uuid::new_v4(),
        AssetNamespace::Output,
        "injected/canonical-writer",
        "png",
        0,
        2,
        2,
        new_bytes.clone(),
    )?;
    let new_receipt = {
        let mut assets = config
            .assets
            .lock()
            .map_err(|_| "asset service was poisoned")?;
        let mut committer = config
            .output_committer
            .lock()
            .map_err(|_| "canonical writer was poisoned")?;
        let receipts = committer.commit_proposal_batch_and_register(
            &[proposal],
            chrono::DateTime::parse_from_rfc3339("2026-10-10T10:00:00+00:00")?,
            &mut assets,
            &first_authorization,
            &CancellationToken::default(),
        )?;
        assert_eq!(receipts.len(), 1);
        assert_eq!(committer.operations().len(), 2);
        assert!(
            committer
                .operation(first_receipt.operation().operation_id)
                .is_some()
        );
        receipts
            .into_iter()
            .next()
            .ok_or("missing injected-writer receipt")?
    };
    assert_eq!(
        first_writer
            .lock()
            .map_err(|_| "canonical writer was poisoned")?
            .operations()
            .len(),
        2
    );
    assert_eq!(
        second_writer
            .lock()
            .map_err(|_| "canonical writer was poisoned")?
            .operations()
            .len(),
        1
    );
    assert!(!unused_binary.exists());
    drop(config);
    drop(pool);
    drop(first_writer);
    drop(second_writer);
    drop(first_assets);
    drop(second_assets);
    let reopened_writer = OutputCommitter::open(first_roots.clone())?;
    let reopened_assets = AssetService::open(first_roots)?;
    assert_eq!(reopened_writer.operations().len(), 2);
    for receipt in [&first_receipt, &new_receipt] {
        assert_eq!(
            reopened_writer
                .operation(receipt.operation().operation_id)
                .ok_or("injected canonical journal failed recovery")?
                .sha256,
            receipt.operation().sha256
        );
    }
    assert_eq!(
        reopened_assets.read_verified(
            &new_receipt.operation().identity,
            &first_authorization,
            &CancellationToken::default(),
            1024 * 1024,
        )?,
        new_bytes
    );
    let second_reopened_writer = OutputCommitter::open(second_roots.clone())?;
    assert_eq!(second_reopened_writer.operations().len(), 1);
    let second_reopened_assets = AssetService::open(second_roots)?;
    assert_eq!(
        second_reopened_assets.read_verified(
            &second_receipt.operation().identity,
            &second_authorization,
            &CancellationToken::default(),
            1024 * 1024,
        )?,
        encoded_png(0.5)?
    );
    drop(first_directory);
    drop(second_directory);
    Ok(())
}

fn publish_linked_derived(
    fixture: &mut Fixture,
    input: &ProjectAssetVersion,
    bytes: Vec<u8>,
) -> Result<(ProjectAssetVersion, ProjectEditableSourceReference), Box<dyn Error>> {
    let editable = publish_editable_container(fixture, &document(input.reference.clone()))?;
    let receipt = publish(fixture, bytes)?;
    let derived = fixture.assets.register_project_png_version(
        &receipt,
        receipt.operation().identity.clone(),
        Uuid::new_v4(),
        vec![input.reference.clone()],
        Some(editable.clone()),
        json!({"kind":"retained-output-integrity-control", "input":input.reference}),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    Ok((derived, editable))
}

#[test]
fn project_png_retained_output_rejects_missing_or_changed_historical_input_after_current_update()
-> TestResult {
    let mut fixture = fixture()?;
    let first_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, first_bytes.clone(), false)?;
    let first = pin(&mut fixture, &identity, &first_bytes)?;
    let derived_bytes = encoded_png(0.5)?;
    let (derived, _) = publish_linked_derived(&mut fixture, &first, derived_bytes.clone())?;
    let second_bytes = encoded_png(0.75)?;
    upload(&mut fixture, second_bytes.clone(), true)?;
    let second = pin(&mut fixture, &identity, &second_bytes)?;
    let mut current = document(second.reference.clone());
    current.build_outputs.push(derived.reference.clone());
    let current_source = publish_editable_container(&mut fixture, &current)?;
    let mut reopened = AssetService::open(fixture.roots.clone())?;
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &derived.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        derived_bytes
    );
    let historical_path = fixture
        .roots
        .test_root_path(AssetNamespace::Output)?
        .join(&first.stored_identity.relative_path);
    fs::write(&historical_path, &second_bytes)?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &derived.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(_)) | Err(ProjectDocumentError::VersionIntegrity)
    ));
    fs::remove_file(&historical_path)?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &derived.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(_))
    ));
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &second.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        second_bytes
    );
    assert_eq!(
        SimDocument::decode(&reopened.read_verified(
            &current_source.identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?)?,
        current
    );
    fs::write(&historical_path, &first_bytes)?;
    reopened.scan_namespaces(
        &[AssetNamespace::Output],
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(
        reopened.read_verified(
            &first.stored_identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        first_bytes
    );
    let catalog_path = fixture
        .roots
        .test_root_path(AssetNamespace::Output)?
        .join(".sim/project-asset-versions.json");
    let catalog_bytes = fs::read(&catalog_path)?;
    let mut catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    let first_reference = serde_json::to_value(&first.reference)?;
    let versions = catalog
        .get_mut("versions")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or("missing real version catalog rows")?;
    let previous_count = versions.len();
    versions.retain(|version| version.get("reference") != Some(&first_reference));
    assert_eq!(versions.len() + 1, previous_count);
    fs::write(&catalog_path, serde_json::to_vec(&catalog)?)?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &derived.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::MissingVersion(reference)) if reference == first.reference
    ));
    fs::write(catalog_path, catalog_bytes)?;
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &derived.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .version(),
        &derived
    );
    Ok(())
}

#[test]
fn project_png_retained_output_rejects_changed_or_missing_editable_source_and_recovers()
-> TestResult {
    let mut fixture = fixture()?;
    let input_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, input_bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &input_bytes)?;
    let derived_bytes = encoded_png(0.5)?;
    let (derived, editable) = publish_linked_derived(&mut fixture, &input, derived_bytes.clone())?;
    let source_path = fixture
        .roots
        .test_root_path(AssetNamespace::Output)?
        .join(&editable.identity.relative_path);
    let source_bytes = fs::read(&source_path)?;
    let mut tampered = source_bytes.clone();
    tampered.push(b' ');
    assert!(SimDocument::decode(&tampered).is_ok());
    let mut reopened = AssetService::open(fixture.roots.clone())?;
    fs::write(&source_path, tampered)?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &derived.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(_)) | Err(ProjectDocumentError::VersionIntegrity)
    ));
    fs::remove_file(&source_path)?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &derived.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        ),
        Err(ProjectDocumentError::Asset(_))
    ));
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &input.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .bytes(),
        input_bytes
    );
    fs::write(source_path, &source_bytes)?;
    reopened.scan_namespaces(
        &[AssetNamespace::Output],
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(
        reopened.read_verified(
            &editable.identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        source_bytes
    );
    let recovered = reopened.resolve_project_asset_version(
        &derived.reference,
        &fixture.authorization,
        &CancellationToken::default(),
        1024 * 1024,
    )?;
    assert_eq!(recovered.bytes(), derived_bytes);
    assert_eq!(recovered.version(), &derived);
    assert_eq!(recovered.version().editable_source, Some(editable));
    Ok(())
}

#[test]
fn project_png_retained_output_rejects_catalog_dependency_cycle_deterministically() -> TestResult {
    let mut fixture = fixture()?;
    let input_bytes = encoded_png(0.25)?;
    let identity = upload(&mut fixture, input_bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &input_bytes)?;
    let derived_bytes = encoded_png(0.5)?;
    let (derived, _) = publish_linked_derived(&mut fixture, &input, derived_bytes.clone())?;
    let catalog_path = fixture
        .roots
        .test_root_path(AssetNamespace::Output)?
        .join(".sim/project-asset-versions.json");
    let catalog_bytes = fs::read(&catalog_path)?;
    let mut catalog: serde_json::Value = serde_json::from_slice(&catalog_bytes)?;
    let input_reference = serde_json::to_value(&input.reference)?;
    let input_row = catalog
        .get_mut("versions")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or("missing real version catalog rows")?
        .iter_mut()
        .find(|version| version.get("reference") == Some(&input_reference))
        .ok_or("missing real input version row")?;
    let dependencies = input_row
        .get_mut("dependencies")
        .ok_or("missing real dependency field")?;
    *dependencies = serde_json::to_value(vec![derived.reference.clone()])?;
    fs::write(&catalog_path, serde_json::to_vec(&catalog)?)?;
    let reopened = AssetService::open(fixture.roots.clone())?;
    for _ in 0..2 {
        assert!(matches!(
            reopened.resolve_project_asset_version(
                &derived.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            ),
            Err(ProjectDocumentError::InvalidVersion)
        ));
    }
    assert_eq!(
        reopened.read_verified(
            &input.stored_identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        input_bytes
    );
    assert_eq!(
        reopened.read_verified(
            &derived.stored_identity,
            &fixture.authorization,
            &CancellationToken::default(),
            1024 * 1024
        )?,
        derived_bytes
    );
    fs::write(catalog_path, catalog_bytes)?;
    assert_eq!(
        reopened
            .resolve_project_asset_version(
                &derived.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024
            )?
            .version(),
        &derived
    );
    Ok(())
}

#[test]
fn project_png_requested_output_cap_does_not_limit_larger_retained_input() -> TestResult {
    let mut fixture = fixture()?;
    let workflow = "larger-valid-source-metadata".repeat(256);
    let pixels = [
        0.25, 0.1, 0.9, 0., 0.2, 0.8, 0.3, 1., 0.1, 0.2, 0.7, 0.5, 0.9, 0.1, 0.2, 1.,
    ];
    let input_bytes = sim_media::encode_png_frame(
        &pixels,
        1,
        2,
        2,
        4,
        0,
        &BTreeMap::from([("workflow".into(), workflow.clone())]),
        sim_media::PngLimits::default(),
    )?;
    let decoded = sim_media::decode_png(&input_bytes, sim_media::PngLimits::default())?;
    assert_eq!((decoded.width, decoded.height), (2, 2));
    assert!(decoded.has_alpha);
    assert_eq!(decoded.metadata.sim_metadata().workflow, Some(workflow));
    let derived_bytes = encoded_png(0.5)?;
    assert!(input_bytes.len() > derived_bytes.len());
    let identity = upload(&mut fixture, input_bytes.clone(), false)?;
    let input = pin(&mut fixture, &identity, &input_bytes)?;
    let receipt = publish(&mut fixture, derived_bytes.clone())?;
    let asset_id = Uuid::new_v4();
    let dependencies = vec![input.reference.clone()];
    let provenance = json!({
        "schema": "test-only-project-provenance",
        "ordered_inputs": dependencies,
        "qualification": "requested-byte-cap-is-not-ancestor-byte-cap"
    });
    let derived = fixture.assets.register_project_png_version(
        &receipt,
        identity.clone(),
        asset_id,
        dependencies.clone(),
        None,
        provenance.clone(),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(input.byte_size, u64::try_from(input_bytes.len())?);
    assert_eq!(derived.byte_size, u64::try_from(derived_bytes.len())?);
    assert!(input.byte_size > derived.byte_size);
    let repeated = fixture.assets.register_project_png_version(
        &receipt,
        identity,
        asset_id,
        dependencies.clone(),
        None,
        provenance.clone(),
        &fixture.authorization,
        &fixture.authorization,
        &CancellationToken::default(),
    )?;
    assert_eq!(repeated, derived);
    let reopened = AssetService::open(fixture.roots.clone())?;
    let resolved = reopened.resolve_project_asset_version(
        &derived.reference,
        &fixture.authorization,
        &CancellationToken::default(),
        derived.byte_size,
    )?;
    assert_eq!(resolved.bytes(), derived_bytes);
    assert_eq!(resolved.version(), &derived);
    assert_eq!(resolved.version().dependencies, dependencies);
    assert_eq!(resolved.version().provenance, provenance);
    let too_small = derived
        .byte_size
        .checked_sub(1)
        .ok_or("derived PNG must contain bytes")?;
    assert!(matches!(
        reopened.resolve_project_asset_version(
            &derived.reference,
            &fixture.authorization,
            &CancellationToken::default(),
            too_small
        ),
        Err(ProjectDocumentError::Asset(crate::assets::AssetError::TooLarge {
            actual,
            limit
        })) if actual == derived.byte_size && limit == too_small
    ));
    Ok(())
}

#[test]
fn project_png_catalog_listing_preserves_version_contracts_and_requires_output_read() -> TestResult
{
    let mut fixture = fixture()?;
    assert!(
        fixture
            .assets
            .project_png_versions(&fixture.authorization)?
            .is_empty()
    );
    let bytes = encoded_png(0.4)?;
    let identity = upload(&mut fixture, bytes.clone(), false)?;
    let version = pin(&mut fixture, &identity, &bytes)?;
    let versions = fixture
        .assets
        .project_png_versions(&fixture.authorization)?;
    assert_eq!(versions, vec![version.clone()]);
    assert_eq!(fixture.committer.operations().len(), 1);
    assert_eq!(
        fixture
            .assets
            .resolve_project_asset_version(
                &versions.first().ok_or("no retained version")?.reference,
                &fixture.authorization,
                &CancellationToken::default(),
                1024 * 1024,
            )?
            .bytes(),
        bytes
    );

    let capabilities = CapabilitySet::new([Capability::Asset {
        namespace: AssetNamespace::Output.locator_type().into(),
        action: AssetOperation::Write,
    }]);
    let grant = PermissionGrant::new(
        fixture.roots.profile_id.clone(),
        "catalog-writer",
        capabilities.clone(),
        "write-only",
    )?;
    let write_only = PermissionPolicy::new(fixture.roots.profile_id.clone(), [grant])?
        .authorize("catalog-writer", &capabilities)?;
    assert!(fixture.assets.project_png_versions(&write_only).is_err());
    assert_eq!(
        fixture
            .assets
            .project_png_versions(&fixture.authorization)?,
        vec![version]
    );
    assert_eq!(fixture.committer.operations().len(), 1);
    Ok(())
}
