use collab::{
    migration::buzz::object_git_metadata::{
        BuzzObjectGitImportError, BuzzObjectGitImportOutcome, BuzzObjectKind,
        BuzzObjectMetadataRecord, import_object_git_metadata,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const MAIN_OID: &str = "1111111111111111111111111111111111111111";

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "buzz-object-inventory")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn record(
    community_id: CommunityId,
    source_sequence: u64,
    key: String,
    size: u64,
    bytes: Option<Vec<u8>>,
    observed_sha256: [u8; 32],
    etag: Option<&str>,
) -> BuzzObjectMetadataRecord {
    BuzzObjectMetadataRecord::new(
        community_id,
        source_sequence,
        key,
        size,
        observed_sha256,
        etag.map(str::to_owned),
        bytes,
    )
    .expect("valid source record")
}

fn fixture(community_id: CommunityId) -> (Vec<BuzzObjectMetadataRecord>, [u8; 32], [u8; 32]) {
    let owner = "aa".repeat(32);
    let blob_bytes = b"image";
    let blob_sha256 = sha256(blob_bytes);
    let blob_digest = hex::encode(blob_sha256);
    let pack_bytes = b"PACK fixture";
    let pack_sha256 = sha256(pack_bytes);
    let pack_digest = hex::encode(pack_sha256);
    let sidecar = serde_json::to_vec(&json!({
        "dim": "1x1",
        "blurhash": "",
        "thumb_url": "",
        "ext": "png",
        "mime_type": "image/png",
        "size": blob_bytes.len(),
        "uploaded_at": 1_700_000_000,
        "duration_secs": null
    }))
    .expect("sidecar JSON");
    let manifest = format!(
        "{{\"version\":1,\"head\":\"refs/heads/main\",\"refs\":{{\"refs/heads/main\":\"{MAIN_OID}\"}},\"packs\":[\"packs/{pack_digest}\"],\"parent\":null}}"
    )
    .into_bytes();
    let manifest_sha256 = sha256(&manifest);
    let manifest_digest = hex::encode(manifest_sha256);
    let pointer = manifest_digest.as_bytes().to_vec();

    (
        vec![
            record(
                community_id,
                1,
                format!("{blob_digest}.png"),
                blob_bytes.len() as u64,
                None,
                blob_sha256,
                Some("blob-etag"),
            ),
            record(
                community_id,
                2,
                format!("_meta/{community_id}/{blob_digest}.json"),
                sidecar.len() as u64,
                Some(sidecar.clone()),
                sha256(&sidecar),
                Some("sidecar-etag"),
            ),
            record(
                community_id,
                3,
                format!("packs/{pack_digest}"),
                pack_bytes.len() as u64,
                None,
                pack_sha256,
                Some("pack-etag"),
            ),
            record(
                community_id,
                4,
                format!("manifests/{manifest_digest}"),
                manifest.len() as u64,
                Some(manifest),
                manifest_sha256,
                Some("manifest-etag"),
            ),
            record(
                community_id,
                5,
                format!("repos/{community_id}/{owner}/zed/pointer"),
                pointer.len() as u64,
                Some(pointer.clone()),
                sha256(&pointer),
                Some("pointer-etag"),
            ),
        ],
        blob_sha256,
        pack_sha256,
    )
}

#[test]
fn fixture_inventory_preserves_content_and_ref_identity() {
    let community_id = community(1);
    let (records, blob_sha256, pack_sha256) = fixture(community_id);

    let outcome = import_object_git_metadata(&tenant(community_id), &records)
        .expect("inventory must validate");
    let BuzzObjectGitImportOutcome::Complete(inventory) = outcome else {
        panic!("complete fixture must not report missing objects");
    };

    assert_eq!(inventory.objects.len(), 5);
    assert!(inventory.objects.iter().any(|object| {
        object.kind == BuzzObjectKind::MediaBlob && object.observed_sha256 == blob_sha256
    }));
    assert!(inventory.objects.iter().any(|object| {
        object.kind == BuzzObjectKind::GitPack && object.observed_sha256 == pack_sha256
    }));
    let media = inventory.media_bindings.first().expect("media binding");
    assert_eq!(media.sha256, blob_sha256);
    assert_eq!(media.mime_type, "image/png");
    let repository = inventory.repositories.first().expect("repository");
    assert_eq!(repository.repository_name, "zed");
    assert_eq!(repository.head, "refs/heads/main");
    assert_eq!(
        repository.refs.get("refs/heads/main"),
        Some(&MAIN_OID.to_owned())
    );
    assert_eq!(repository.pack_sha256, vec![pack_sha256]);
    assert_eq!(
        hex::encode(repository.ref_state_hash),
        "25f908604744ead9f4f6262cd17f4f7a4217b57df1e20026ad4e3ee830847bf4"
    );
    let checkpoint = inventory.checkpoint_progress();
    assert_eq!(checkpoint.final_source_sequence, 5);
    assert_eq!(checkpoint.scanned, 5);
    assert_eq!(checkpoint.imported, 5);
    assert_eq!(checkpoint.skipped, 0);
}

#[test]
fn missing_referenced_pack_reports_without_checkpoint_progress() {
    let community_id = community(1);
    let (mut records, _, pack_sha256) = fixture(community_id);
    records.retain(|record| !record.key().starts_with("packs/"));

    let outcome = import_object_git_metadata(&tenant(community_id), &records)
        .expect("missing object is a report, not an importer failure");
    assert_eq!(outcome.checkpoint_progress(), None);
    let BuzzObjectGitImportOutcome::Missing(report) = outcome else {
        panic!("missing pack must block completion");
    };
    assert_eq!(report.missing.len(), 1);
    assert_eq!(
        report.missing[0].key,
        format!("packs/{}", hex::encode(pack_sha256))
    );
    assert_eq!(report.missing[0].expected_sha256, Some(pack_sha256));
    assert_eq!(report.last_scanned_source_sequence, 5);
}

#[test]
fn foreign_tenant_pointer_fails_closed() {
    let community_id = community(1);
    let foreign = community(2);
    let owner = "aa".repeat(32);
    let pointer = "11".repeat(32).into_bytes();
    let record = record(
        community_id,
        1,
        format!("repos/{foreign}/{owner}/zed/pointer"),
        pointer.len() as u64,
        Some(pointer.clone()),
        sha256(&pointer),
        Some("pointer-etag"),
    );

    let error = import_object_git_metadata(&tenant(community_id), &[record])
        .expect_err("foreign object binding must fail closed");
    assert!(matches!(
        error,
        BuzzObjectGitImportError::TenantBoundaryViolation
    ));
}
