use std::collections::{BTreeMap, BTreeSet, HashSet};

use collaboration_domain::{CommunityId, TenantContext};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const MAX_INVENTORY_RECORDS: usize = 100_000;
const MAX_METADATA_BYTES: usize = 2 * 1024 * 1024;
const MAX_MANIFEST_PACKS: usize = 128;
const MAX_MANIFEST_REFS: usize = 10_000;
const UPLOAD_RECORD_VERSION: u32 = 1;
const MANIFEST_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzObjectMetadataRecord {
    community_id: CommunityId,
    source_sequence: u64,
    key: String,
    size: u64,
    observed_sha256: [u8; 32],
    etag: Option<String>,
    metadata_bytes: Option<Vec<u8>>,
}

impl BuzzObjectMetadataRecord {
    pub fn new(
        community_id: CommunityId,
        source_sequence: u64,
        key: impl Into<String>,
        size: u64,
        observed_sha256: [u8; 32],
        etag: Option<String>,
        metadata_bytes: Option<Vec<u8>>,
    ) -> Result<Self, BuzzObjectGitImportError> {
        let key = key.into();
        if source_sequence == 0
            || key.is_empty()
            || key.len() > 1_024
            || key.trim() != key
            || key.chars().any(char::is_control)
            || etag.as_ref().is_some_and(|etag| {
                etag.is_empty()
                    || etag.len() > 1_024
                    || etag.trim() != etag
                    || etag.chars().any(char::is_control)
            })
            || metadata_bytes
                .as_ref()
                .is_some_and(|bytes| bytes.len() > MAX_METADATA_BYTES)
        {
            return Err(BuzzObjectGitImportError::InvalidSourceRecord);
        }
        if let Some(bytes) = &metadata_bytes
            && (sha256(bytes) != observed_sha256 || u64::try_from(bytes.len()).ok() != Some(size))
        {
            return Err(BuzzObjectGitImportError::MetadataDigestMismatch { key });
        }
        Ok(Self {
            community_id,
            source_sequence,
            key,
            size,
            observed_sha256,
            etag,
            metadata_bytes,
        })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub const fn source_sequence(&self) -> u64 {
        self.source_sequence
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BuzzObjectKind {
    MediaBlob,
    MediaThumbnail,
    MediaSidecar,
    MediaUploadRecord,
    GitPack,
    GitIndex,
    GitManifest,
    GitPointer,
}

impl BuzzObjectKind {
    const fn hash_label(self) -> &'static [u8] {
        match self {
            Self::MediaBlob => b"media_blob",
            Self::MediaThumbnail => b"media_thumbnail",
            Self::MediaSidecar => b"media_sidecar",
            Self::MediaUploadRecord => b"media_upload_record",
            Self::GitPack => b"git_pack",
            Self::GitIndex => b"git_index",
            Self::GitManifest => b"git_manifest",
            Self::GitPointer => b"git_pointer",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzObjectIdentity {
    pub key: String,
    pub kind: BuzzObjectKind,
    pub size: u64,
    pub observed_sha256: [u8; 32],
    pub etag: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzMediaBinding {
    pub community_id: CommunityId,
    pub sha256: [u8; 32],
    pub blob_key: String,
    pub sidecar_keys: Vec<String>,
    pub upload_record_keys: Vec<String>,
    pub thumbnail_key: Option<String>,
    pub mime_type: String,
    pub size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzGitRepositoryInventory {
    pub community_id: CommunityId,
    pub owner_public_key: String,
    pub repository_name: String,
    pub pointer_key: String,
    pub pointer_etag: String,
    pub manifest_sha256: [u8; 32],
    pub head: String,
    pub refs: BTreeMap<String, String>,
    pub pack_sha256: Vec<[u8; 32]>,
    pub ref_state_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzObjectGitCheckpointProgress {
    pub final_source_sequence: u64,
    pub source_hash: [u8; 32],
    pub target_hash: [u8; 32],
    pub scanned: u64,
    pub imported: u64,
    pub skipped: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzObjectGitInventory {
    pub objects: Vec<BuzzObjectIdentity>,
    pub media_bindings: Vec<BuzzMediaBinding>,
    pub repositories: Vec<BuzzGitRepositoryInventory>,
    checkpoint: BuzzObjectGitCheckpointProgress,
}

impl BuzzObjectGitInventory {
    pub const fn checkpoint_progress(&self) -> BuzzObjectGitCheckpointProgress {
        self.checkpoint
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct BuzzMissingObject {
    pub key: String,
    pub required_by: String,
    pub expected_sha256: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzMissingObjectReport {
    pub missing: Vec<BuzzMissingObject>,
    pub scanned: u64,
    pub last_scanned_source_sequence: u64,
    pub source_hash: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuzzObjectGitImportOutcome {
    Complete(BuzzObjectGitInventory),
    Missing(BuzzMissingObjectReport),
}

impl BuzzObjectGitImportOutcome {
    pub const fn checkpoint_progress(&self) -> Option<BuzzObjectGitCheckpointProgress> {
        match self {
            Self::Complete(inventory) => Some(inventory.checkpoint_progress()),
            Self::Missing(_) => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzObjectGitImportError {
    #[error("Buzz object metadata source record is invalid")]
    InvalidSourceRecord,
    #[error("Buzz object metadata body does not match its observed SHA-256 for {key}")]
    MetadataDigestMismatch { key: String },
    #[error("Buzz object metadata inventory crossed its tenant boundary")]
    TenantBoundaryViolation,
    #[error("Buzz object metadata inventory is empty, oversized or out of order")]
    InvalidBatch,
    #[error("Buzz object metadata inventory contains duplicate key {key}")]
    DuplicateKey { key: String },
    #[error("Buzz object metadata inventory contains unknown key {key}")]
    UnknownKey { key: String },
    #[error("Buzz content-addressed object {key} has a different observed SHA-256")]
    ContentIdentityMismatch { key: String },
    #[error("Buzz metadata body is required for {key}")]
    MissingMetadataBody { key: String },
    #[error("Buzz metadata document {key} is invalid: {reason}")]
    InvalidMetadataDocument { key: String, reason: &'static str },
    #[error("Buzz media records disagree about content {sha256}")]
    MediaBindingConflict { sha256: String },
    #[error("Buzz Git manifest ancestry contains a cycle at {key}")]
    ManifestCycle { key: String },
}

#[derive(Clone, Debug, Deserialize)]
struct BuzzBlobMeta {
    #[serde(default)]
    dim: String,
    #[serde(default)]
    blurhash: String,
    #[serde(default)]
    thumb_url: String,
    ext: String,
    mime_type: String,
    size: u64,
    #[serde(default)]
    uploaded_at: i64,
    #[serde(default)]
    duration_secs: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
struct BuzzUploadRecord {
    version: u32,
    event_id: String,
    sha256: String,
    ext: String,
    mime_type: String,
    size: u64,
    uploaded_at: i64,
    community_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BuzzGitManifest {
    version: u32,
    head: String,
    refs: BTreeMap<String, String>,
    packs: Vec<String>,
    parent: Option<String>,
}

#[derive(Clone, Debug)]
struct MediaBindingBuilder {
    sha256: [u8; 32],
    extension: String,
    mime_type: String,
    size: u64,
    sidecar_keys: BTreeSet<String>,
    upload_record_keys: BTreeSet<String>,
    thumbnail_required: bool,
}

#[derive(Clone, Debug)]
struct RepositoryPointer {
    owner_public_key: String,
    repository_name: String,
    pointer_key: String,
    pointer_etag: String,
    manifest_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ClassifiedKey {
    MediaBlob {
        sha256: [u8; 32],
        extension: String,
    },
    MediaThumbnail {
        source_sha256: [u8; 32],
    },
    MediaSidecar {
        community_id: Uuid,
        sha256: [u8; 32],
    },
    MediaUploadRecord {
        community_id: Uuid,
        sha256: [u8; 32],
        event_id: String,
    },
    GitPack {
        sha256: [u8; 32],
    },
    GitIndex {
        pack_sha256: [u8; 32],
    },
    GitManifest {
        sha256: [u8; 32],
    },
    GitPointer {
        community_id: Uuid,
        owner_public_key: String,
        repository_name: String,
    },
    Probe,
}

impl ClassifiedKey {
    fn object_kind(&self) -> Option<BuzzObjectKind> {
        match self {
            Self::MediaBlob { .. } => Some(BuzzObjectKind::MediaBlob),
            Self::MediaThumbnail { .. } => Some(BuzzObjectKind::MediaThumbnail),
            Self::MediaSidecar { .. } => Some(BuzzObjectKind::MediaSidecar),
            Self::MediaUploadRecord { .. } => Some(BuzzObjectKind::MediaUploadRecord),
            Self::GitPack { .. } => Some(BuzzObjectKind::GitPack),
            Self::GitIndex { .. } => Some(BuzzObjectKind::GitIndex),
            Self::GitManifest { .. } => Some(BuzzObjectKind::GitManifest),
            Self::GitPointer { .. } => Some(BuzzObjectKind::GitPointer),
            Self::Probe => None,
        }
    }
}

pub fn import_object_git_metadata(
    tenant: &TenantContext,
    records: &[BuzzObjectMetadataRecord],
) -> Result<BuzzObjectGitImportOutcome, BuzzObjectGitImportError> {
    validate_batch(tenant, records)?;

    let mut source_hasher = Sha256::new();
    let mut records_by_key = BTreeMap::new();
    let mut classes_by_key = BTreeMap::new();
    for record in records {
        hash_source_record(&mut source_hasher, record);
        if records_by_key.insert(record.key.as_str(), record).is_some() {
            return Err(BuzzObjectGitImportError::DuplicateKey {
                key: record.key.clone(),
            });
        }
        let class =
            classify_key(&record.key).ok_or_else(|| BuzzObjectGitImportError::UnknownKey {
                key: record.key.clone(),
            })?;
        validate_classified_record(tenant, record, &class)?;
        classes_by_key.insert(record.key.as_str(), class);
    }

    let source_hash = source_hasher.finalize().into();
    let scanned =
        u64::try_from(records.len()).map_err(|_| BuzzObjectGitImportError::InvalidBatch)?;
    let last_scanned_source_sequence = records
        .last()
        .map(|record| record.source_sequence)
        .ok_or(BuzzObjectGitImportError::InvalidBatch)?;
    let mut reachable_keys = BTreeSet::new();
    let mut missing = BTreeSet::new();
    let mut media = BTreeMap::<[u8; 32], MediaBindingBuilder>::new();
    let mut pointers = Vec::new();

    for (key, class) in &classes_by_key {
        let record = records_by_key
            .get(key)
            .copied()
            .ok_or(BuzzObjectGitImportError::InvalidSourceRecord)?;
        match class {
            ClassifiedKey::MediaSidecar { sha256, .. } => {
                reachable_keys.insert((*key).to_owned());
                let metadata: BuzzBlobMeta = parse_metadata(record)?;
                validate_media_fields(&record.key, &metadata.ext, &metadata.mime_type)?;
                let builder = media.entry(*sha256).or_insert_with(|| MediaBindingBuilder {
                    sha256: *sha256,
                    extension: metadata.ext.clone(),
                    mime_type: metadata.mime_type.clone(),
                    size: metadata.size,
                    sidecar_keys: BTreeSet::new(),
                    upload_record_keys: BTreeSet::new(),
                    thumbnail_required: false,
                });
                merge_media_binding(builder, &metadata.ext, &metadata.mime_type, metadata.size)?;
                builder.sidecar_keys.insert((*key).to_owned());
                builder.thumbnail_required |= !metadata.thumb_url.is_empty();
                let _ = (
                    &metadata.dim,
                    &metadata.blurhash,
                    metadata.uploaded_at,
                    metadata.duration_secs,
                );
            }
            ClassifiedKey::MediaUploadRecord {
                sha256, event_id, ..
            } => {
                reachable_keys.insert((*key).to_owned());
                let upload: BuzzUploadRecord = parse_metadata(record)?;
                validate_upload_record(tenant, record, sha256, event_id, &upload)?;
                let builder = media.entry(*sha256).or_insert_with(|| MediaBindingBuilder {
                    sha256: *sha256,
                    extension: upload.ext.clone(),
                    mime_type: upload.mime_type.clone(),
                    size: upload.size,
                    sidecar_keys: BTreeSet::new(),
                    upload_record_keys: BTreeSet::new(),
                    thumbnail_required: false,
                });
                merge_media_binding(builder, &upload.ext, &upload.mime_type, upload.size)?;
                builder.upload_record_keys.insert((*key).to_owned());
            }
            ClassifiedKey::GitPointer {
                owner_public_key,
                repository_name,
                ..
            } => {
                reachable_keys.insert((*key).to_owned());
                let body = required_metadata_bytes(record)?;
                let digest_text = std::str::from_utf8(body)
                    .ok()
                    .map(str::trim)
                    .ok_or_else(|| invalid_document(&record.key, "pointer is not UTF-8"))?;
                let manifest_sha256 = parse_sha256(digest_text)
                    .ok_or_else(|| invalid_document(&record.key, "pointer digest is malformed"))?;
                let pointer_etag = record
                    .etag
                    .clone()
                    .ok_or_else(|| invalid_document(&record.key, "pointer ETag is absent"))?;
                pointers.push(RepositoryPointer {
                    owner_public_key: owner_public_key.clone(),
                    repository_name: repository_name.clone(),
                    pointer_key: (*key).to_owned(),
                    pointer_etag,
                    manifest_sha256,
                });
            }
            ClassifiedKey::Probe
            | ClassifiedKey::MediaBlob { .. }
            | ClassifiedKey::MediaThumbnail { .. }
            | ClassifiedKey::GitPack { .. }
            | ClassifiedKey::GitIndex { .. }
            | ClassifiedKey::GitManifest { .. } => {}
        }
    }

    for builder in media.values() {
        let digest = hex::encode(builder.sha256);
        let blob_key = format!("{digest}.{}", builder.extension);
        require_object(
            &records_by_key,
            &blob_key,
            format!("media:{digest}"),
            Some(builder.sha256),
            &mut reachable_keys,
            &mut missing,
        );
        if let Some(record) = records_by_key.get(blob_key.as_str())
            && record.size != builder.size
        {
            return Err(BuzzObjectGitImportError::MediaBindingConflict { sha256: digest });
        }
        if builder.thumbnail_required {
            let thumbnail_key = format!("{digest}.thumb.jpg");
            require_object(
                &records_by_key,
                &thumbnail_key,
                format!("media-thumbnail:{digest}"),
                None,
                &mut reachable_keys,
                &mut missing,
            );
        }
    }

    let mut repositories = Vec::with_capacity(pointers.len());
    for pointer in pointers {
        let repository = inventory_repository(
            tenant,
            &pointer,
            &records_by_key,
            &classes_by_key,
            &mut reachable_keys,
            &mut missing,
        )?;
        if let Some(repository) = repository {
            repositories.push(repository);
        }
    }

    if !missing.is_empty() {
        return Ok(BuzzObjectGitImportOutcome::Missing(
            BuzzMissingObjectReport {
                missing: missing.into_iter().collect(),
                scanned,
                last_scanned_source_sequence,
                source_hash,
            },
        ));
    }

    let mut objects = Vec::with_capacity(reachable_keys.len());
    for key in &reachable_keys {
        let record = records_by_key
            .get(key.as_str())
            .copied()
            .ok_or(BuzzObjectGitImportError::InvalidSourceRecord)?;
        let kind = classes_by_key
            .get(key.as_str())
            .and_then(ClassifiedKey::object_kind)
            .ok_or(BuzzObjectGitImportError::InvalidSourceRecord)?;
        objects.push(BuzzObjectIdentity {
            key: key.clone(),
            kind,
            size: record.size,
            observed_sha256: record.observed_sha256,
            etag: record.etag.clone(),
        });
    }
    let media_bindings = media
        .into_values()
        .map(|builder| {
            let digest = hex::encode(builder.sha256);
            BuzzMediaBinding {
                community_id: tenant.community_id(),
                sha256: builder.sha256,
                blob_key: format!("{digest}.{}", builder.extension),
                sidecar_keys: builder.sidecar_keys.into_iter().collect(),
                upload_record_keys: builder.upload_record_keys.into_iter().collect(),
                thumbnail_key: builder
                    .thumbnail_required
                    .then(|| format!("{digest}.thumb.jpg")),
                mime_type: builder.mime_type,
                size: builder.size,
            }
        })
        .collect::<Vec<_>>();
    repositories.sort_by(|left, right| left.pointer_key.cmp(&right.pointer_key));
    let target_hash = hash_target(&objects, &media_bindings, &repositories);
    let imported =
        u64::try_from(objects.len()).map_err(|_| BuzzObjectGitImportError::InvalidSourceRecord)?;
    let skipped = scanned
        .checked_sub(imported)
        .ok_or(BuzzObjectGitImportError::InvalidSourceRecord)?;

    Ok(BuzzObjectGitImportOutcome::Complete(
        BuzzObjectGitInventory {
            objects,
            media_bindings,
            repositories,
            checkpoint: BuzzObjectGitCheckpointProgress {
                final_source_sequence: last_scanned_source_sequence,
                source_hash,
                target_hash,
                scanned,
                imported,
                skipped,
            },
        },
    ))
}

fn validate_batch(
    tenant: &TenantContext,
    records: &[BuzzObjectMetadataRecord],
) -> Result<(), BuzzObjectGitImportError> {
    if records.is_empty() || records.len() > MAX_INVENTORY_RECORDS {
        return Err(BuzzObjectGitImportError::InvalidBatch);
    }
    let mut prior_sequence = 0;
    for record in records {
        if record.community_id != tenant.community_id() {
            return Err(BuzzObjectGitImportError::TenantBoundaryViolation);
        }
        if record.source_sequence <= prior_sequence {
            return Err(BuzzObjectGitImportError::InvalidBatch);
        }
        prior_sequence = record.source_sequence;
    }
    Ok(())
}

fn validate_classified_record(
    tenant: &TenantContext,
    record: &BuzzObjectMetadataRecord,
    class: &ClassifiedKey,
) -> Result<(), BuzzObjectGitImportError> {
    let tenant_uuid = tenant.community_id().as_uuid();
    match class {
        ClassifiedKey::MediaSidecar { community_id, .. }
        | ClassifiedKey::MediaUploadRecord { community_id, .. }
        | ClassifiedKey::GitPointer { community_id, .. }
            if *community_id != tenant_uuid =>
        {
            return Err(BuzzObjectGitImportError::TenantBoundaryViolation);
        }
        ClassifiedKey::MediaBlob { sha256, .. }
        | ClassifiedKey::GitPack { sha256 }
        | ClassifiedKey::GitManifest { sha256 }
            if *sha256 != record.observed_sha256 =>
        {
            return Err(BuzzObjectGitImportError::ContentIdentityMismatch {
                key: record.key.clone(),
            });
        }
        _ => {}
    }
    Ok(())
}

fn inventory_repository(
    tenant: &TenantContext,
    pointer: &RepositoryPointer,
    records: &BTreeMap<&str, &BuzzObjectMetadataRecord>,
    classes: &BTreeMap<&str, ClassifiedKey>,
    reachable: &mut BTreeSet<String>,
    missing: &mut BTreeSet<BuzzMissingObject>,
) -> Result<Option<BuzzGitRepositoryInventory>, BuzzObjectGitImportError> {
    let current_manifest_key = format!("manifests/{}", hex::encode(pointer.manifest_sha256));
    if !records.contains_key(current_manifest_key.as_str()) {
        missing.insert(BuzzMissingObject {
            key: current_manifest_key,
            required_by: pointer.pointer_key.clone(),
            expected_sha256: Some(pointer.manifest_sha256),
        });
        return Ok(None);
    }

    let mut manifest_key = current_manifest_key;
    let mut visited = HashSet::new();
    let mut current_manifest = None;
    loop {
        if !visited.insert(manifest_key.clone()) {
            return Err(BuzzObjectGitImportError::ManifestCycle { key: manifest_key });
        }
        reachable.insert(manifest_key.clone());
        let record = records
            .get(manifest_key.as_str())
            .copied()
            .ok_or(BuzzObjectGitImportError::InvalidSourceRecord)?;
        if !matches!(
            classes.get(manifest_key.as_str()),
            Some(ClassifiedKey::GitManifest { .. })
        ) {
            return Err(invalid_document(
                &manifest_key,
                "manifest key has the wrong object class",
            ));
        }
        let manifest = parse_manifest(record)?;
        for pack_key in &manifest.packs {
            let pack_sha256 = parse_pack_key(pack_key)
                .ok_or_else(|| invalid_document(&manifest_key, "manifest pack key is malformed"))?;
            require_object(
                records,
                pack_key,
                manifest_key.clone(),
                Some(pack_sha256),
                reachable,
                missing,
            );
            let index_key = format!("idx/{}", hex::encode(pack_sha256));
            if records.contains_key(index_key.as_str()) {
                reachable.insert(index_key);
            }
        }
        if current_manifest.is_none() {
            current_manifest = Some(manifest.clone());
        }
        let Some(parent) = manifest.parent else {
            break;
        };
        let parent_sha256 = parse_sha256(&parent)
            .ok_or_else(|| invalid_document(&manifest_key, "manifest parent is malformed"))?;
        let parent_key = format!("manifests/{parent}");
        if !records.contains_key(parent_key.as_str()) {
            missing.insert(BuzzMissingObject {
                key: parent_key,
                required_by: manifest_key,
                expected_sha256: Some(parent_sha256),
            });
            break;
        }
        manifest_key = parent_key;
    }

    let Some(manifest) = current_manifest else {
        return Ok(None);
    };
    let pack_sha256 = manifest
        .packs
        .iter()
        .map(|key| {
            parse_pack_key(key).ok_or_else(|| {
                invalid_document(&pointer.pointer_key, "manifest pack key is malformed")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ref_state_hash = hash_ref_state(
        tenant.community_id(),
        &pointer.owner_public_key,
        &pointer.repository_name,
        &manifest,
    );
    Ok(Some(BuzzGitRepositoryInventory {
        community_id: tenant.community_id(),
        owner_public_key: pointer.owner_public_key.clone(),
        repository_name: pointer.repository_name.clone(),
        pointer_key: pointer.pointer_key.clone(),
        pointer_etag: pointer.pointer_etag.clone(),
        manifest_sha256: pointer.manifest_sha256,
        head: manifest.head,
        refs: manifest.refs,
        pack_sha256,
        ref_state_hash,
    }))
}

fn parse_manifest(
    record: &BuzzObjectMetadataRecord,
) -> Result<BuzzGitManifest, BuzzObjectGitImportError> {
    let bytes = required_metadata_bytes(record)?;
    let manifest: BuzzGitManifest = serde_json::from_slice(bytes)
        .map_err(|_| invalid_document(&record.key, "manifest JSON is malformed"))?;
    if manifest.version != MANIFEST_VERSION
        || manifest.packs.len() > MAX_MANIFEST_PACKS
        || manifest.refs.len() > MAX_MANIFEST_REFS
        || !is_safe_refname(&manifest.head)
        || manifest
            .refs
            .iter()
            .any(|(name, oid)| !is_safe_refname(name) || !is_git_oid(oid))
        || manifest
            .packs
            .iter()
            .any(|key| parse_pack_key(key).is_none())
        || !manifest.packs.windows(2).all(|pair| pair[0] < pair[1])
        || manifest
            .parent
            .as_ref()
            .is_some_and(|digest| parse_sha256(digest).is_none())
    {
        return Err(invalid_document(
            &record.key,
            "manifest violates Buzz schema v1",
        ));
    }
    let canonical = serde_json::to_vec(&manifest)
        .map_err(|_| invalid_document(&record.key, "manifest cannot be canonicalized"))?;
    if canonical != bytes {
        return Err(invalid_document(
            &record.key,
            "manifest bytes are not canonical",
        ));
    }
    Ok(manifest)
}

fn validate_media_fields(
    key: &str,
    extension: &str,
    mime_type: &str,
) -> Result<(), BuzzObjectGitImportError> {
    if !is_blob_extension(extension)
        || mime_type.is_empty()
        || mime_type.len() > 255
        || mime_type.trim() != mime_type
        || mime_type.chars().any(char::is_control)
    {
        return Err(invalid_document(key, "media fields are malformed"));
    }
    Ok(())
}

fn validate_upload_record(
    tenant: &TenantContext,
    record: &BuzzObjectMetadataRecord,
    key_sha256: &[u8; 32],
    key_event_id: &str,
    upload: &BuzzUploadRecord,
) -> Result<(), BuzzObjectGitImportError> {
    validate_media_fields(&record.key, &upload.ext, &upload.mime_type)?;
    if upload.version != UPLOAD_RECORD_VERSION
        || upload.event_id != key_event_id
        || parse_sha256(&upload.sha256) != Some(*key_sha256)
        || upload.community_id != tenant.community_id().to_string()
        || upload.uploaded_at < 0
    {
        return Err(invalid_document(
            &record.key,
            "upload record violates Buzz schema v1",
        ));
    }
    Ok(())
}

fn merge_media_binding(
    builder: &mut MediaBindingBuilder,
    extension: &str,
    mime_type: &str,
    size: u64,
) -> Result<(), BuzzObjectGitImportError> {
    if builder.extension != extension || builder.mime_type != mime_type || builder.size != size {
        return Err(BuzzObjectGitImportError::MediaBindingConflict {
            sha256: hex::encode(builder.sha256),
        });
    }
    Ok(())
}

fn require_object(
    records: &BTreeMap<&str, &BuzzObjectMetadataRecord>,
    key: &str,
    required_by: String,
    expected_sha256: Option<[u8; 32]>,
    reachable: &mut BTreeSet<String>,
    missing: &mut BTreeSet<BuzzMissingObject>,
) {
    if records.contains_key(key) {
        reachable.insert(key.to_owned());
    } else {
        missing.insert(BuzzMissingObject {
            key: key.to_owned(),
            required_by,
            expected_sha256,
        });
    }
}

fn parse_metadata<T: for<'de> Deserialize<'de>>(
    record: &BuzzObjectMetadataRecord,
) -> Result<T, BuzzObjectGitImportError> {
    serde_json::from_slice(required_metadata_bytes(record)?)
        .map_err(|_| invalid_document(&record.key, "JSON is malformed"))
}

fn required_metadata_bytes(
    record: &BuzzObjectMetadataRecord,
) -> Result<&[u8], BuzzObjectGitImportError> {
    record
        .metadata_bytes
        .as_deref()
        .ok_or_else(|| BuzzObjectGitImportError::MissingMetadataBody {
            key: record.key.clone(),
        })
}

fn invalid_document(key: &str, reason: &'static str) -> BuzzObjectGitImportError {
    BuzzObjectGitImportError::InvalidMetadataDocument {
        key: key.to_owned(),
        reason,
    }
}

fn classify_key(key: &str) -> Option<ClassifiedKey> {
    if key.starts_with("probe/") {
        return Some(ClassifiedKey::Probe);
    }
    if let Some(digest) = key.strip_prefix("packs/").and_then(parse_sha256) {
        return Some(ClassifiedKey::GitPack { sha256: digest });
    }
    if let Some(digest) = key.strip_prefix("idx/").and_then(parse_sha256) {
        return Some(ClassifiedKey::GitIndex {
            pack_sha256: digest,
        });
    }
    if let Some(digest) = key.strip_prefix("manifests/").and_then(parse_sha256) {
        return Some(ClassifiedKey::GitManifest { sha256: digest });
    }
    if let Some(pointer) = parse_pointer_key(key) {
        return Some(pointer);
    }
    if let Some(sidecar) = parse_sidecar_key(key) {
        return Some(sidecar);
    }
    if let Some(upload) = parse_upload_record_key(key) {
        return Some(upload);
    }
    if let Some(digest) = key.strip_suffix(".thumb.jpg").and_then(parse_sha256) {
        return Some(ClassifiedKey::MediaThumbnail {
            source_sha256: digest,
        });
    }
    let (digest, extension) = key.split_once('.')?;
    if extension.contains('.') || !is_blob_extension(extension) {
        return None;
    }
    Some(ClassifiedKey::MediaBlob {
        sha256: parse_sha256(digest)?,
        extension: extension.to_owned(),
    })
}

fn parse_sidecar_key(key: &str) -> Option<ClassifiedKey> {
    let suffix = key.strip_prefix("_meta/")?;
    let (community, file) = suffix.split_once('/')?;
    if file.contains('/') {
        return None;
    }
    let digest = file.strip_suffix(".json").and_then(parse_sha256)?;
    Some(ClassifiedKey::MediaSidecar {
        community_id: parse_canonical_uuid(community)?,
        sha256: digest,
    })
}

fn parse_upload_record_key(key: &str) -> Option<ClassifiedKey> {
    let suffix = key.strip_prefix("_uploads/")?;
    let mut parts = suffix.split('/');
    let community_id = parse_canonical_uuid(parts.next()?)?;
    let sha256 = parse_sha256(parts.next()?)?;
    let event_id = parts.next()?.strip_suffix(".json")?;
    if parts.next().is_some() || !is_ulid(event_id) {
        return None;
    }
    Some(ClassifiedKey::MediaUploadRecord {
        community_id,
        sha256,
        event_id: event_id.to_owned(),
    })
}

fn parse_pointer_key(key: &str) -> Option<ClassifiedKey> {
    let suffix = key.strip_prefix("repos/")?;
    let mut parts = suffix.split('/');
    let community_id = parse_canonical_uuid(parts.next()?)?;
    let owner_public_key = parts.next()?;
    let repository_name = parts.next()?;
    if parts.next()? != "pointer"
        || parts.next().is_some()
        || parse_public_key(owner_public_key).is_none()
        || !is_repository_name(repository_name)
    {
        return None;
    }
    Some(ClassifiedKey::GitPointer {
        community_id,
        owner_public_key: owner_public_key.to_owned(),
        repository_name: repository_name.to_owned(),
    })
}

fn parse_pack_key(key: &str) -> Option<[u8; 32]> {
    key.strip_prefix("packs/").and_then(parse_sha256)
}

fn parse_public_key(value: &str) -> Option<[u8; 32]> {
    parse_sha256(value)
}

fn parse_sha256(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let bytes = hex::decode(value).ok()?;
    bytes.try_into().ok()
}

fn parse_canonical_uuid(value: &str) -> Option<Uuid> {
    let parsed = Uuid::parse_str(value).ok()?;
    (parsed.to_string() == value).then_some(parsed)
}

fn is_blob_extension(value: &str) -> bool {
    !value.is_empty() && value.len() <= 8 && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

fn is_repository_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && !value.starts_with('.')
        && !value.contains("..")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn is_ulid(value: &str) -> bool {
    value.len() == 26
        && value.bytes().all(|byte| {
            byte.is_ascii_digit()
                || (b'A'..=b'H').contains(&byte)
                || matches!(byte, b'J' | b'K' | b'M' | b'N')
                || (b'P'..=b'T').contains(&byte)
                || (b'V'..=b'Z').contains(&byte)
        })
}

fn is_safe_refname(value: &str) -> bool {
    value.starts_with("refs/")
        && !value.ends_with('/')
        && !value.contains("..")
        && !value.contains("//")
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '/' | '_' | '.' | '-')
        })
}

fn is_git_oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hash_source_record(hasher: &mut Sha256, record: &BuzzObjectMetadataRecord) {
    hash_u64(hasher, record.source_sequence);
    hash_bytes(hasher, record.key.as_bytes());
    hash_u64(hasher, record.size);
    hasher.update(record.observed_sha256);
    hash_optional_bytes(hasher, record.etag.as_deref().map(str::as_bytes));
}

fn hash_ref_state(
    community_id: CommunityId,
    owner_public_key: &str,
    repository_name: &str,
    manifest: &BuzzGitManifest,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hash_bytes(&mut hasher, community_id.to_string().as_bytes());
    hash_bytes(&mut hasher, owner_public_key.as_bytes());
    hash_bytes(&mut hasher, repository_name.as_bytes());
    hash_bytes(&mut hasher, manifest.head.as_bytes());
    for (name, oid) in &manifest.refs {
        hash_bytes(&mut hasher, name.as_bytes());
        hash_bytes(&mut hasher, oid.as_bytes());
    }
    hasher.finalize().into()
}

fn hash_target(
    objects: &[BuzzObjectIdentity],
    media_bindings: &[BuzzMediaBinding],
    repositories: &[BuzzGitRepositoryInventory],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for object in objects {
        hash_bytes(&mut hasher, object.key.as_bytes());
        hash_bytes(&mut hasher, object.kind.hash_label());
        hash_u64(&mut hasher, object.size);
        hasher.update(object.observed_sha256);
        hash_optional_bytes(&mut hasher, object.etag.as_deref().map(str::as_bytes));
    }
    for media in media_bindings {
        hasher.update(media.sha256);
        hash_bytes(&mut hasher, media.blob_key.as_bytes());
        hash_bytes(&mut hasher, media.mime_type.as_bytes());
        hash_u64(&mut hasher, media.size);
        for key in &media.sidecar_keys {
            hash_bytes(&mut hasher, key.as_bytes());
        }
        for key in &media.upload_record_keys {
            hash_bytes(&mut hasher, key.as_bytes());
        }
        hash_optional_bytes(
            &mut hasher,
            media.thumbnail_key.as_deref().map(str::as_bytes),
        );
    }
    for repository in repositories {
        hash_bytes(&mut hasher, repository.pointer_key.as_bytes());
        hasher.update(repository.manifest_sha256);
        hasher.update(repository.ref_state_hash);
    }
    hasher.finalize().into()
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_u64(hasher, u64::try_from(bytes.len()).unwrap_or(u64::MAX));
    hasher.update(bytes);
}

fn hash_optional_bytes(hasher: &mut Sha256, bytes: Option<&[u8]>) {
    match bytes {
        Some(bytes) => {
            hasher.update([1]);
            hash_bytes(hasher, bytes);
        }
        None => hasher.update([0]),
    }
}
