use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::{
    media::object_store::{
        MediaCleanupCandidate, MediaListingSafety, MediaObjectBackend, MediaObjectBackendError,
        MediaObjectDeleteOutcome, MediaObjectPage, MediaObjectVersion, MediaObjectWriteOutcome,
        MediaOrphanDeletionLease, MediaOrphanFinalization, MediaResolvedRange,
    },
    retention::{
        media_cleanup::{
            RetentionMediaBackend, RetentionMediaBackendError, RetentionMediaCheckpoint,
            RetentionMediaCheckpointCommitOutcome, RetentionMediaCleanup,
            RetentionMediaCleanupError, RetentionMediaCleanupItem, RetentionMediaReferenceOutcome,
            RetentionMediaRemovalPlan,
        },
        worker::RetentionSourcePosition,
    },
};
use collaboration_domain::{
    AggregateId, CommunityId, MediaAttachmentLink, MediaByteSize, MediaContentHash,
    MediaDescriptor, MediaIdentity, OperationId, TenantContext, TrustedTenantRoute,
};
use uuid::Uuid;

#[derive(Clone)]
struct StoredObject {
    version: MediaObjectVersion,
}

#[derive(Default)]
struct ObjectState {
    objects: BTreeMap<MediaContentHash, StoredObject>,
    delete_attempts: BTreeMap<MediaContentHash, usize>,
    fail_once: BTreeSet<MediaContentHash>,
    version_artifacts: BTreeSet<MediaContentHash>,
}

#[derive(Clone, Default)]
struct ObjectBackend {
    state: Arc<Mutex<ObjectState>>,
}

impl ObjectBackend {
    fn insert(&self, content_hash: MediaContentHash, version: &str) {
        self.state.lock().expect("object state").objects.insert(
            content_hash,
            StoredObject {
                version: MediaObjectVersion::new(version).expect("version"),
            },
        );
    }

    fn contains(&self, content_hash: MediaContentHash) -> bool {
        self.state
            .lock()
            .expect("object state")
            .objects
            .contains_key(&content_hash)
    }

    fn fail_once(&self, content_hash: MediaContentHash) {
        self.state
            .lock()
            .expect("object state")
            .fail_once
            .insert(content_hash);
    }
}

#[async_trait]
impl MediaObjectBackend for ObjectBackend {
    async fn put_if_absent(
        &self,
        _descriptor: &MediaDescriptor,
        _reader: std::fs::File,
    ) -> Result<MediaObjectWriteOutcome, MediaObjectBackendError> {
        Err(MediaObjectBackendError::InvalidData)
    }

    async fn get_range(
        &self,
        _descriptor: &MediaDescriptor,
        _object_version: &MediaObjectVersion,
        _range: MediaResolvedRange,
    ) -> Result<Option<Vec<u8>>, MediaObjectBackendError> {
        Err(MediaObjectBackendError::InvalidData)
    }

    async fn list_page(
        &self,
        _after: Option<MediaContentHash>,
        _limit: u32,
    ) -> Result<MediaObjectPage, MediaObjectBackendError> {
        Ok(MediaObjectPage::new(
            Vec::new(),
            None,
            MediaListingSafety::KnownUnversioned,
        ))
    }

    async fn delete_if_match(
        &self,
        content_hash: MediaContentHash,
        object_version: &MediaObjectVersion,
    ) -> Result<MediaObjectDeleteOutcome, MediaObjectBackendError> {
        let mut state = self.state.lock().expect("object state");
        *state.delete_attempts.entry(content_hash).or_default() += 1;
        if state.fail_once.remove(&content_hash) {
            return Err(MediaObjectBackendError::Unavailable);
        }
        if state.version_artifacts.contains(&content_hash) {
            return Ok(MediaObjectDeleteOutcome::VersionArtifact);
        }
        let Some(object) = state.objects.get(&content_hash) else {
            return Ok(MediaObjectDeleteOutcome::AlreadyMissing);
        };
        if object.version != *object_version {
            return Ok(MediaObjectDeleteOutcome::PreconditionFailed);
        }
        state.objects.remove(&content_hash);
        Ok(MediaObjectDeleteOutcome::Deleted)
    }
}

#[derive(Clone)]
struct AuthorityObject {
    byte_size: MediaByteSize,
    version: MediaObjectVersion,
    created_at_millis: u64,
}

#[derive(Default)]
struct BackendState {
    checkpoint: Option<RetentionMediaCheckpoint>,
    items: Vec<RetentionMediaCleanupItem>,
    references: BTreeMap<MediaContentHash, u64>,
    objects: BTreeMap<MediaContentHash, AuthorityObject>,
    pending: BTreeMap<OperationId, Vec<MediaOrphanDeletionLease>>,
    detached: BTreeSet<OperationId>,
    detach_attempts: usize,
}

#[derive(Clone, Default)]
struct Backend {
    state: Arc<Mutex<BackendState>>,
}

impl Backend {
    fn new(items: Vec<RetentionMediaCleanupItem>) -> Self {
        Self {
            state: Arc::new(Mutex::new(BackendState {
                items,
                ..BackendState::default()
            })),
        }
    }

    fn register(&self, content_hash: MediaContentHash, references: u64, version: &str) {
        let mut state = self.state.lock().expect("backend state");
        state.references.insert(content_hash, references);
        state.objects.insert(
            content_hash,
            AuthorityObject {
                byte_size: MediaByteSize::new(16).expect("size"),
                version: MediaObjectVersion::new(version).expect("version"),
                created_at_millis: 1,
            },
        );
    }
}

#[async_trait]
impl RetentionMediaBackend for Backend {
    async fn load_checkpoint(
        &self,
        _tenant: &TenantContext,
    ) -> Result<Option<RetentionMediaCheckpoint>, RetentionMediaBackendError> {
        Ok(self.state.lock().expect("backend state").checkpoint)
    }

    async fn load_batch(
        &self,
        _tenant: &TenantContext,
        checkpoint: RetentionMediaCheckpoint,
        limit: usize,
    ) -> Result<Vec<RetentionMediaCleanupItem>, RetentionMediaBackendError> {
        Ok(self
            .state
            .lock()
            .expect("backend state")
            .items
            .iter()
            .filter(|item| {
                item.source_position().sequence() > checkpoint.source_position().sequence()
            })
            .take(limit)
            .cloned()
            .collect())
    }

    async fn detach_and_reserve_orphans(
        &self,
        tenant: &TenantContext,
        item: &RetentionMediaCleanupItem,
    ) -> Result<RetentionMediaRemovalPlan, RetentionMediaBackendError> {
        if item.community_id() != tenant.community_id() {
            return Err(RetentionMediaBackendError::Conflict);
        }
        let mut state = self.state.lock().expect("backend state");
        state.detach_attempts += 1;
        if state.detached.contains(&item.operation_id()) {
            return RetentionMediaRemovalPlan::new(
                item.operation_id(),
                RetentionMediaReferenceOutcome::AlreadyDetached,
                state
                    .pending
                    .get(&item.operation_id())
                    .cloned()
                    .unwrap_or_default(),
            )
            .map_err(|_| RetentionMediaBackendError::InvalidData);
        }
        let mut leases = Vec::new();
        for attachment in item.attachments() {
            let content_hash = attachment.media_identity().content_hash();
            let references = state
                .references
                .get_mut(&content_hash)
                .ok_or(RetentionMediaBackendError::InvalidData)?;
            *references = references
                .checked_sub(1)
                .ok_or(RetentionMediaBackendError::InvalidData)?;
            if *references == 0 {
                let object = state
                    .objects
                    .get(&content_hash)
                    .ok_or(RetentionMediaBackendError::InvalidData)?;
                let candidate = MediaCleanupCandidate::new(
                    content_hash,
                    object.byte_size,
                    object.version.clone(),
                    object.created_at_millis,
                )
                .map_err(|_| RetentionMediaBackendError::InvalidData)?;
                leases.push(
                    MediaOrphanDeletionLease::new(item.operation_id().as_uuid(), &candidate, 1)
                        .map_err(|_| RetentionMediaBackendError::InvalidData)?,
                );
            }
        }
        state.detached.insert(item.operation_id());
        state.pending.insert(item.operation_id(), leases.clone());
        RetentionMediaRemovalPlan::new(
            item.operation_id(),
            RetentionMediaReferenceOutcome::Detached,
            leases,
        )
        .map_err(|_| RetentionMediaBackendError::InvalidData)
    }

    async fn finalize_orphan_deletion(
        &self,
        tenant: &TenantContext,
        item: &RetentionMediaCleanupItem,
        lease: &MediaOrphanDeletionLease,
        _outcome: MediaOrphanFinalization,
    ) -> Result<(), RetentionMediaBackendError> {
        if tenant.community_id() != item.community_id() {
            return Err(RetentionMediaBackendError::Conflict);
        }
        let mut state = self.state.lock().expect("backend state");
        let leases = state
            .pending
            .get_mut(&item.operation_id())
            .ok_or(RetentionMediaBackendError::InvalidData)?;
        let Some(index) = leases.iter().position(|candidate| candidate == lease) else {
            return Err(RetentionMediaBackendError::InvalidData);
        };
        leases.remove(index);
        Ok(())
    }

    async fn advance_checkpoint(
        &self,
        _tenant: &TenantContext,
        expected: RetentionMediaCheckpoint,
        next: RetentionMediaCheckpoint,
    ) -> Result<RetentionMediaCheckpointCommitOutcome, RetentionMediaBackendError> {
        let mut state = self.state.lock().expect("backend state");
        let current = state
            .checkpoint
            .unwrap_or_else(|| RetentionMediaCheckpoint::initial(expected.community_id()));
        if current == next {
            return Ok(RetentionMediaCheckpointCommitOutcome::AlreadyCommitted);
        }
        if current != expected {
            return Err(RetentionMediaBackendError::StaleCheckpoint);
        }
        state.checkpoint = Some(next);
        Ok(RetentionMediaCheckpointCommitOutcome::Committed)
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "retention-media-cleanup")
                .expect("route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn hash(value: u8) -> MediaContentHash {
    MediaContentHash::from_digest([value; 32])
}

fn attachment(tenant: &TenantContext, content_hash: MediaContentHash) -> MediaAttachmentLink {
    MediaAttachmentLink::new(
        tenant,
        MediaIdentity::new(tenant.community_id(), content_hash).expect("identity"),
        AggregateId::from_uuid(Uuid::from_u128(20)),
        AggregateId::from_uuid(Uuid::from_u128(21)),
    )
    .expect("attachment")
}

fn item(
    tenant: &TenantContext,
    sequence: u64,
    operation: u128,
    hashes: &[MediaContentHash],
) -> RetentionMediaCleanupItem {
    RetentionMediaCleanupItem::new(
        tenant.community_id(),
        RetentionSourcePosition::new(sequence, [sequence as u8; 32]).expect("position"),
        OperationId::from_uuid(Uuid::from_u128(operation)),
        1_000,
        hashes
            .iter()
            .map(|content_hash| attachment(tenant, *content_hash))
            .collect(),
    )
    .expect("item")
}

#[tokio::test]
async fn shared_content_survives_while_the_last_reference_becomes_a_verified_orphan() {
    let tenant = tenant(community(1));
    let shared = hash(1);
    let orphan = hash(2);
    let backend = Backend::new(vec![item(&tenant, 1, 30, &[shared, orphan])]);
    backend.register(shared, 2, "shared-v1");
    backend.register(orphan, 1, "orphan-v1");
    let objects = ObjectBackend::default();
    objects.insert(shared, "shared-v1");
    objects.insert(orphan, "orphan-v1");

    let outcome = RetentionMediaCleanup::new(backend.clone(), objects.clone())
        .run_batch(&tenant, 10)
        .await
        .expect("cleanup");
    assert_eq!(outcome.checkpoint().source_position().sequence(), 1);
    assert_eq!(outcome.counts().detached_references, 2);
    assert_eq!(outcome.counts().deleted_objects, 1);
    assert!(outcome.completed_batch());
    assert!(objects.contains(shared));
    assert!(!objects.contains(orphan));
    let state = backend.state.lock().expect("backend state");
    assert_eq!(state.references[&shared], 1);
    assert_eq!(state.references[&orphan], 0);
}

#[tokio::test]
async fn object_store_failure_preserves_the_checkpoint_and_resumes_the_same_lease() {
    let tenant = tenant(community(1));
    let orphan = hash(2);
    let backend = Backend::new(vec![item(&tenant, 1, 31, &[orphan])]);
    backend.register(orphan, 1, "orphan-v1");
    let objects = ObjectBackend::default();
    objects.insert(orphan, "orphan-v1");
    objects.fail_once(orphan);
    let cleanup = RetentionMediaCleanup::new(backend.clone(), objects.clone());

    assert!(matches!(
        cleanup.run_batch(&tenant, 10).await,
        Err(RetentionMediaCleanupError::ObjectBackend(
            MediaObjectBackendError::Unavailable
        ))
    ));
    assert!(
        backend
            .state
            .lock()
            .expect("backend state")
            .checkpoint
            .is_none()
    );
    assert!(objects.contains(orphan));
    let resumed = cleanup
        .run_batch(&tenant, 10)
        .await
        .expect("resumed cleanup");
    assert_eq!(resumed.counts().already_detached_references, 1);
    assert_eq!(resumed.counts().deleted_objects, 1);
    assert!(!objects.contains(orphan));
    let state = backend.state.lock().expect("backend state");
    assert_eq!(state.detach_attempts, 2);
    assert!(state.pending[&OperationId::from_uuid(Uuid::from_u128(31))].is_empty());
}

#[tokio::test]
async fn changed_generation_is_preserved_for_the_next_global_orphan_sweep() {
    let tenant = tenant(community(1));
    let orphan = hash(2);
    let backend = Backend::new(vec![item(&tenant, 1, 32, &[orphan])]);
    backend.register(orphan, 1, "observed-v1");
    let objects = ObjectBackend::default();
    objects.insert(orphan, "replacement-v2");

    let outcome = RetentionMediaCleanup::new(backend, objects.clone())
        .run_batch(&tenant, 10)
        .await
        .expect("cleanup");
    assert_eq!(outcome.counts().concurrently_changed_objects, 1);
    assert!(objects.contains(orphan));
}

#[tokio::test]
async fn foreign_or_regressing_batches_fail_before_reference_or_object_mutation() {
    let local = tenant(community(1));
    let foreign = tenant(community(2));
    let content_hash = hash(2);
    let backend = Backend::new(vec![item(&foreign, 1, 33, &[content_hash])]);
    backend.register(content_hash, 1, "orphan-v1");
    let objects = ObjectBackend::default();
    objects.insert(content_hash, "orphan-v1");
    assert!(matches!(
        RetentionMediaCleanup::new(backend.clone(), objects.clone())
            .run_batch(&local, 10)
            .await,
        Err(RetentionMediaCleanupError::InvalidBatch)
    ));
    assert_eq!(
        backend.state.lock().expect("backend state").detach_attempts,
        0
    );
    assert!(objects.contains(content_hash));

    let backend = Backend::new(vec![
        item(&local, 2, 34, &[hash(3)]),
        item(&local, 1, 35, &[hash(4)]),
    ]);
    assert!(matches!(
        RetentionMediaCleanup::new(backend.clone(), objects)
            .run_batch(&local, 10)
            .await,
        Err(RetentionMediaCleanupError::InvalidBatch)
    ));
    assert_eq!(
        backend.state.lock().expect("backend state").detach_attempts,
        0
    );
}

#[test]
fn attachment_batches_reject_duplicate_or_mixed_message_links() {
    let tenant = tenant(community(1));
    let link = attachment(&tenant, hash(1));
    assert!(matches!(
        RetentionMediaCleanupItem::new(
            tenant.community_id(),
            RetentionSourcePosition::new(1, [1; 32]).expect("position"),
            OperationId::from_uuid(Uuid::from_u128(40)),
            1,
            vec![link, link],
        ),
        Err(RetentionMediaCleanupError::InvalidInput)
    ));

    let other_message = MediaAttachmentLink::new(
        &tenant,
        MediaIdentity::new(tenant.community_id(), hash(2)).expect("identity"),
        AggregateId::from_uuid(Uuid::from_u128(20)),
        AggregateId::from_uuid(Uuid::from_u128(22)),
    )
    .expect("other message");
    assert!(matches!(
        RetentionMediaCleanupItem::new(
            tenant.community_id(),
            RetentionSourcePosition::new(1, [1; 32]).expect("position"),
            OperationId::from_uuid(Uuid::from_u128(41)),
            1,
            vec![link, other_message],
        ),
        Err(RetentionMediaCleanupError::InvalidInput)
    ));
}
