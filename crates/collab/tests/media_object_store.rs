use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::media::{
    object_store::{
        AuthorizedMediaObject, MediaCheckpointCommitOutcome, MediaCleanupCandidate,
        MediaCleanupCheckpoint, MediaCleanupReport, MediaListingSafety, MediaObjectAuthority,
        MediaObjectAuthorityError, MediaObjectBackend, MediaObjectBackendError,
        MediaObjectDeleteOutcome, MediaObjectPage, MediaObjectStore, MediaObjectStoreError,
        MediaObjectStoreLimits, MediaObjectVersion, MediaObjectWriteOutcome,
        MediaOrphanDeletionLease, MediaOrphanFinalization, MediaOrphanReservationOutcome,
        MediaPublication, MediaPublicationOutcome, MediaRangeRequest, MediaResolvedRange,
        MediaStoreOutcome, StoredMediaObject,
    },
    upload_admission::{MediaUploadAdmission, MediaUploadRequest},
    validation::{
        MAX_DECODED_IMAGE_PIXELS, MAX_IMAGE_UPLOAD_BYTES, MAX_VIDEO_UPLOAD_BYTES,
        MediaValidationLimits, MediaValidationSession,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MediaByteSize, MediaContentHash, MediaContentType,
    MediaDescriptor, MediaIdentity, MediaMetadata, MediaObjectSelection, MediaTenantPath,
    MembershipRole, MembershipStatus, OperationId, PrincipalId, PrincipalScopes, ServiceAccountId,
    TenantContext, TrustedTenantRoute,
};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Clone)]
struct ObjectRecord {
    descriptor: MediaDescriptor,
    version: MediaObjectVersion,
    created_at_millis: u64,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct ObjectBackendState {
    objects: BTreeMap<MediaContentHash, ObjectRecord>,
    put_attempts: u64,
    range_attempts: u64,
    delete_attempts: u64,
    listing_safety: Option<MediaListingSafety>,
    fail_next_delete: bool,
}

#[derive(Clone, Default)]
struct ObjectBackend {
    state: Arc<Mutex<ObjectBackendState>>,
}

impl ObjectBackend {
    fn insert(&self, hash_byte: u8, created_at_millis: u64) -> MediaContentHash {
        let content_hash = MediaContentHash::from_digest([hash_byte; 32]);
        let bytes = vec![hash_byte; 16];
        let descriptor = MediaDescriptor::new(
            content_hash,
            MediaContentType::new("image/png").expect("content type"),
            MediaByteSize::new(bytes.len() as u64).expect("size"),
        );
        self.state.lock().expect("object state").objects.insert(
            content_hash,
            ObjectRecord {
                descriptor,
                version: MediaObjectVersion::new(format!("generation-{hash_byte}"))
                    .expect("version"),
                created_at_millis,
                bytes,
            },
        );
        content_hash
    }

    fn contains(&self, content_hash: MediaContentHash) -> bool {
        self.state
            .lock()
            .expect("object state")
            .objects
            .contains_key(&content_hash)
    }
}

#[async_trait]
impl MediaObjectBackend for ObjectBackend {
    async fn put_if_absent(
        &self,
        descriptor: &MediaDescriptor,
        mut reader: std::fs::File,
    ) -> Result<MediaObjectWriteOutcome, MediaObjectBackendError> {
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|_| MediaObjectBackendError::Unavailable)?;
        if bytes.len() as u64 != descriptor.byte_size().get()
            || MediaContentHash::from_digest(Sha256::digest(&bytes).into())
                != descriptor.content_hash()
        {
            return Err(MediaObjectBackendError::InvalidData);
        }
        let mut state = self.state.lock().expect("object state");
        state.put_attempts += 1;
        if let Some(existing) = state.objects.get(&descriptor.content_hash()) {
            return Ok(MediaObjectWriteOutcome::Existing(
                StoredMediaObject::new(
                    existing.descriptor.clone(),
                    existing.version.clone(),
                    existing.created_at_millis,
                )
                .map_err(|_| MediaObjectBackendError::InvalidData)?,
            ));
        }
        let record = ObjectRecord {
            descriptor: descriptor.clone(),
            version: MediaObjectVersion::new("generation-upload")
                .map_err(|_| MediaObjectBackendError::InvalidData)?,
            created_at_millis: 150,
            bytes,
        };
        let stored = StoredMediaObject::new(
            record.descriptor.clone(),
            record.version.clone(),
            record.created_at_millis,
        )
        .map_err(|_| MediaObjectBackendError::InvalidData)?;
        state.objects.insert(descriptor.content_hash(), record);
        Ok(MediaObjectWriteOutcome::Created(stored))
    }

    async fn get_range(
        &self,
        descriptor: &MediaDescriptor,
        object_version: &MediaObjectVersion,
        range: MediaResolvedRange,
    ) -> Result<Option<Vec<u8>>, MediaObjectBackendError> {
        let mut state = self.state.lock().expect("object state");
        state.range_attempts += 1;
        let Some(object) = state.objects.get(&descriptor.content_hash()) else {
            return Ok(None);
        };
        if object.descriptor != *descriptor || object.version != *object_version {
            return Err(MediaObjectBackendError::InvalidData);
        }
        let start =
            usize::try_from(range.start()).map_err(|_| MediaObjectBackendError::InvalidData)?;
        let end = usize::try_from(range.end_inclusive())
            .map_err(|_| MediaObjectBackendError::InvalidData)?;
        Ok(object.bytes.get(start..=end).map(ToOwned::to_owned))
    }

    async fn list_page(
        &self,
        after: Option<MediaContentHash>,
        limit: u32,
    ) -> Result<MediaObjectPage, MediaObjectBackendError> {
        let state = self.state.lock().expect("object state");
        let safety = state
            .listing_safety
            .unwrap_or(MediaListingSafety::KnownUnversioned);
        let mut eligible = state
            .objects
            .iter()
            .filter(|(content_hash, _)| after.is_none_or(|after| **content_hash > after));
        let mut objects = Vec::new();
        for _ in 0..limit {
            let Some((content_hash, object)) = eligible.next() else {
                break;
            };
            objects.push(
                MediaCleanupCandidate::new(
                    *content_hash,
                    object.descriptor.byte_size(),
                    object.version.clone(),
                    object.created_at_millis,
                )
                .map_err(|_| MediaObjectBackendError::InvalidData)?,
            );
        }
        let has_more = eligible.next().is_some();
        let next_after = has_more
            .then(|| objects.last().map(MediaCleanupCandidate::content_hash))
            .flatten();
        Ok(MediaObjectPage::new(objects, next_after, safety))
    }

    async fn delete_if_match(
        &self,
        content_hash: MediaContentHash,
        object_version: &MediaObjectVersion,
    ) -> Result<MediaObjectDeleteOutcome, MediaObjectBackendError> {
        let mut state = self.state.lock().expect("object state");
        state.delete_attempts += 1;
        if state.fail_next_delete {
            state.fail_next_delete = false;
            return Err(MediaObjectBackendError::Unavailable);
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

#[derive(Default)]
struct AuthorityState {
    publications: BTreeMap<(CommunityId, OperationId), MediaPublication>,
    referenced: BTreeSet<MediaContentHash>,
    checkpoints: BTreeMap<Uuid, MediaCleanupCheckpoint>,
    leases: BTreeMap<MediaContentHash, MediaOrphanDeletionLease>,
}

#[derive(Clone, Default)]
struct Authority {
    state: Arc<Mutex<AuthorityState>>,
}

impl Authority {
    fn set_referenced(&self, content_hash: MediaContentHash) {
        self.state
            .lock()
            .expect("authority state")
            .referenced
            .insert(content_hash);
    }
}

#[async_trait]
impl MediaObjectAuthority for Authority {
    async fn publish(
        &self,
        tenant: &TenantContext,
        publication: &MediaPublication,
    ) -> Result<MediaPublicationOutcome, MediaObjectAuthorityError> {
        if publication.metadata().fields().identity.community_id() != tenant.community_id() {
            return Err(MediaObjectAuthorityError::Conflict);
        }
        let mut state = self.state.lock().expect("authority state");
        let key = (tenant.community_id(), publication.operation_id());
        if let Some(existing) = state.publications.get(&key) {
            return Ok(MediaPublicationOutcome::Existing(existing.clone()));
        }
        if state
            .leases
            .contains_key(&publication.metadata().fields().identity.content_hash())
        {
            return Err(MediaObjectAuthorityError::Conflict);
        }
        state.publications.insert(key, publication.clone());
        state
            .referenced
            .insert(publication.metadata().fields().identity.content_hash());
        Ok(MediaPublicationOutcome::Published)
    }

    async fn resolve_for_read(
        &self,
        tenant: &TenantContext,
        _principal_id: PrincipalId,
        _resource: AuthorizationResource,
        path: MediaTenantPath,
    ) -> Result<Option<AuthorizedMediaObject>, MediaObjectAuthorityError> {
        if tenant.community_id() != path.community_id() {
            return Ok(None);
        }
        let state = self.state.lock().expect("authority state");
        let publication = state.publications.values().find(|publication| {
            publication.metadata().fields().identity.community_id() == tenant.community_id()
                && publication.metadata().fields().identity.content_hash() == path.content_hash()
        });
        let Some(publication) = publication else {
            return Ok(None);
        };
        let fields = publication.metadata().fields();
        let descriptor = MediaDescriptor::new(
            fields.identity.content_hash(),
            fields.content_type.clone(),
            fields.byte_size,
        );
        AuthorizedMediaObject::new(path, descriptor, publication.object_version().clone())
            .map(Some)
            .map_err(|_| MediaObjectAuthorityError::InvalidData)
    }

    async fn load_cleanup_checkpoint(
        &self,
        job_id: Uuid,
        scan_started_at_millis: u64,
    ) -> Result<MediaCleanupCheckpoint, MediaObjectAuthorityError> {
        let mut state = self.state.lock().expect("authority state");
        if let Some(checkpoint) = state.checkpoints.get(&job_id) {
            return Ok(*checkpoint);
        }
        let checkpoint = MediaCleanupCheckpoint::initial(job_id, scan_started_at_millis)
            .map_err(|_| MediaObjectAuthorityError::InvalidData)?;
        state.checkpoints.insert(job_id, checkpoint);
        Ok(checkpoint)
    }

    async fn reserve_orphan_deletion(
        &self,
        checkpoint: MediaCleanupCheckpoint,
        candidate: &MediaCleanupCandidate,
        orphan_grace_millis: u64,
    ) -> Result<MediaOrphanReservationOutcome, MediaObjectAuthorityError> {
        let mut state = self.state.lock().expect("authority state");
        if state.referenced.contains(&candidate.content_hash()) {
            return Ok(MediaOrphanReservationOutcome::Referenced);
        }
        if candidate
            .created_at_millis()
            .checked_add(orphan_grace_millis)
            .is_none_or(|eligible_at| eligible_at > checkpoint.scan_started_at_millis())
        {
            return Ok(MediaOrphanReservationOutcome::Indeterminate);
        }
        if let Some(lease) = state.leases.get(&candidate.content_hash()) {
            return Ok(MediaOrphanReservationOutcome::Reserved(lease.clone()));
        }
        let lease = MediaOrphanDeletionLease::new(checkpoint.job_id(), candidate, 1)
            .map_err(|_| MediaObjectAuthorityError::InvalidData)?;
        state.leases.insert(candidate.content_hash(), lease.clone());
        Ok(MediaOrphanReservationOutcome::Reserved(lease))
    }

    async fn finalize_orphan_deletion(
        &self,
        lease: &MediaOrphanDeletionLease,
        _outcome: MediaOrphanFinalization,
    ) -> Result<(), MediaObjectAuthorityError> {
        let mut state = self.state.lock().expect("authority state");
        let Some(existing) = state.leases.get(&lease.content_hash()) else {
            return Err(MediaObjectAuthorityError::InvalidData);
        };
        if existing != lease {
            return Err(MediaObjectAuthorityError::Conflict);
        }
        state.leases.remove(&lease.content_hash());
        Ok(())
    }

    async fn commit_cleanup_checkpoint(
        &self,
        expected: MediaCleanupCheckpoint,
        next: MediaCleanupCheckpoint,
    ) -> Result<MediaCheckpointCommitOutcome, MediaObjectAuthorityError> {
        let mut state = self.state.lock().expect("authority state");
        let Some(current) = state.checkpoints.get(&expected.job_id()).copied() else {
            return Err(MediaObjectAuthorityError::InvalidData);
        };
        if current == next {
            return Ok(MediaCheckpointCommitOutcome::AlreadyCommitted);
        }
        if current != expected {
            return Err(MediaObjectAuthorityError::Conflict);
        }
        state.checkpoints.insert(expected.job_id(), next);
        Ok(MediaCheckpointCommitOutcome::Committed)
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(TrustedTenantRoute::from_listener(community_id, "media-store").expect("route")),
        &[],
    )
    .expect("tenant")
}

fn png_bytes() -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([17, 31, 47, 255])))
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("png");
    bytes.into_inner()
}

fn validated(
    tenant: &TenantContext,
    principal_id: PrincipalId,
    operation_id: OperationId,
    bytes: &[u8],
) -> collab::media::validation::ValidatedMedia {
    let content_hash = MediaContentHash::from_digest(Sha256::digest(bytes).into());
    let admission = MediaUploadAdmission::restore(
        tenant,
        principal_id,
        MediaUploadRequest::new(operation_id, content_hash, bytes.len() as u64).expect("request"),
        100,
        10_000,
    )
    .expect("admission");
    let limits = MediaValidationLimits::new(
        MAX_IMAGE_UPLOAD_BYTES,
        MAX_VIDEO_UPLOAD_BYTES,
        MAX_DECODED_IMAGE_PIXELS,
    )
    .expect("validation limits");
    let mut session = MediaValidationSession::begin(
        admission,
        tenant,
        principal_id,
        MediaContentType::new("image/png").expect("content type"),
        200,
        limits,
    )
    .expect("validation session");
    session.write_chunk(bytes).expect("media bytes");
    session
        .finish(tenant, principal_id, 200)
        .expect("validated media")
}

fn media_path(community_id: CommunityId, content_hash: MediaContentHash) -> MediaTenantPath {
    MediaMetadata::new(
        MediaIdentity::new(community_id, content_hash).expect("identity"),
        principal(2),
        MediaContentType::new("image/png").expect("content type"),
        MediaByteSize::new(1).expect("size"),
        1,
    )
    .expect("metadata")
    .tenant_path(&tenant(community_id), MediaObjectSelection::Original)
    .expect("path")
}

struct ReadAuthorizationFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
    membership: CommunityMembership,
}

impl ReadAuthorizationFixture {
    fn new(community_id: CommunityId) -> Self {
        let principal_id = principal(2);
        let scope = AuthorizationScope::new("media:read").expect("scope");
        Self {
            tenant: tenant(community_id),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(3),
                PrincipalScopes::new([scope.clone()]).expect("scopes"),
            ),
            scope,
            membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        }
    }

    fn request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Read,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Media,
                resource_id: AggregateId::from_uuid(Uuid::from_u128(50)),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 300,
        }
    }
}

fn store(
    object_backend: ObjectBackend,
    authority: Authority,
) -> MediaObjectStore<ObjectBackend, Authority> {
    MediaObjectStore::new(object_backend, authority, MediaObjectStoreLimits::default())
        .expect("store")
}

#[tokio::test]
async fn duplicate_content_is_written_once_and_published_per_tenant() {
    let object_backend = ObjectBackend::default();
    let authority = Authority::default();
    let store = store(object_backend.clone(), authority.clone());
    let bytes = png_bytes();
    for (community_id, operation_id) in [
        (community(1), OperationId::from_uuid(Uuid::from_u128(11))),
        (community(2), OperationId::from_uuid(Uuid::from_u128(12))),
    ] {
        let tenant = tenant(community_id);
        assert_eq!(
            store
                .store_validated(
                    &tenant,
                    principal(2),
                    300,
                    validated(&tenant, principal(2), operation_id, &bytes),
                )
                .await,
            Ok(MediaStoreOutcome::Published)
        );
    }
    let first_tenant = tenant(community(1));
    assert_eq!(
        store
            .store_validated(
                &first_tenant,
                principal(2),
                400,
                validated(
                    &first_tenant,
                    principal(2),
                    OperationId::from_uuid(Uuid::from_u128(11)),
                    &bytes,
                ),
            )
            .await,
        Ok(MediaStoreOutcome::Replayed)
    );
    let object_state = object_backend.state.lock().expect("object state");
    assert_eq!(object_state.objects.len(), 1);
    assert_eq!(object_state.put_attempts, 3);
    drop(object_state);
    assert_eq!(
        authority
            .state
            .lock()
            .expect("authority state")
            .publications
            .len(),
        2
    );
}

#[tokio::test]
async fn authorized_ranges_are_bounded_and_foreign_tenants_never_reach_storage() {
    let object_backend = ObjectBackend::default();
    let authority = Authority::default();
    let store = store(object_backend.clone(), authority);
    let local_tenant = tenant(community(1));
    let bytes = png_bytes();
    store
        .store_validated(
            &local_tenant,
            principal(2),
            300,
            validated(
                &local_tenant,
                principal(2),
                OperationId::from_uuid(Uuid::from_u128(11)),
                &bytes,
            ),
        )
        .await
        .expect("stored");
    let content_hash = MediaContentHash::from_digest(Sha256::digest(&bytes).into());
    let path = media_path(community(1), content_hash);
    let authorization = ReadAuthorizationFixture::new(community(1));
    let response = store
        .read_range(
            &authorization.request(),
            path,
            MediaRangeRequest::new(2, 5).expect("range"),
        )
        .await
        .expect("range response");
    assert_eq!(response.bytes(), &bytes[2..=5]);
    assert_eq!(response.range().byte_length(), Ok(4));
    assert_eq!(response.total_size().get(), bytes.len() as u64);

    let foreign = ReadAuthorizationFixture::new(community(2));
    let attempts_before = object_backend
        .state
        .lock()
        .expect("object state")
        .range_attempts;
    assert_eq!(
        store
            .read_range(
                &foreign.request(),
                path,
                MediaRangeRequest::new(2, 5).expect("range"),
            )
            .await,
        Err(MediaObjectStoreError::UnauthorizedOrNotFound)
    );
    assert_eq!(
        object_backend
            .state
            .lock()
            .expect("object state")
            .range_attempts,
        attempts_before
    );
}

#[tokio::test]
async fn missing_published_object_fails_without_fabricating_a_range() {
    let object_backend = ObjectBackend::default();
    let authority = Authority::default();
    let store = store(object_backend.clone(), authority);
    let local_tenant = tenant(community(1));
    let bytes = png_bytes();
    store
        .store_validated(
            &local_tenant,
            principal(2),
            300,
            validated(
                &local_tenant,
                principal(2),
                OperationId::from_uuid(Uuid::from_u128(11)),
                &bytes,
            ),
        )
        .await
        .expect("stored");
    object_backend
        .state
        .lock()
        .expect("object state")
        .objects
        .clear();
    let content_hash = MediaContentHash::from_digest(Sha256::digest(&bytes).into());
    let authorization = ReadAuthorizationFixture::new(community(1));
    assert_eq!(
        store
            .read_range(
                &authorization.request(),
                media_path(community(1), content_hash),
                MediaRangeRequest::new(0, 1).expect("range"),
            )
            .await,
        Err(MediaObjectStoreError::BackendUnavailable)
    );
}

#[tokio::test]
async fn cleanup_preserves_shared_and_young_objects_and_deletes_only_verified_orphans() {
    let object_backend = ObjectBackend::default();
    let authority = Authority::default();
    let referenced = object_backend.insert(1, 1);
    let orphan = object_backend.insert(2, 1);
    let young = object_backend.insert(3, 9_000_000);
    authority.set_referenced(referenced);
    let store = store(object_backend.clone(), authority);
    let MediaCleanupReport {
        checkpoint,
        reached_run_limit,
    } = store
        .cleanup_orphans(Uuid::from_u128(70), 10_000_000)
        .await
        .expect("cleanup");
    assert!(checkpoint.completed());
    assert_eq!(checkpoint.inspected_objects(), 3);
    assert_eq!(checkpoint.deleted_objects(), 1);
    assert!(!reached_run_limit);
    assert!(object_backend.contains(referenced));
    assert!(!object_backend.contains(orphan));
    assert!(object_backend.contains(young));
}

#[tokio::test]
async fn cleanup_halts_on_unknown_taxonomy_and_resumes_after_storage_failure() {
    let object_backend = ObjectBackend::default();
    let authority = Authority::default();
    let orphan = object_backend.insert(2, 1);
    object_backend
        .state
        .lock()
        .expect("object state")
        .listing_safety = Some(MediaListingSafety::UnknownObjectShape);
    let store = store(object_backend.clone(), authority.clone());
    let job_id = Uuid::from_u128(71);
    assert_eq!(
        store.cleanup_orphans(job_id, 10_000_000).await,
        Err(MediaObjectStoreError::UnsafeCleanup)
    );
    assert!(object_backend.contains(orphan));
    assert_eq!(
        authority.state.lock().expect("authority state").checkpoints[&job_id].inspected_objects(),
        0
    );

    {
        let mut state = object_backend.state.lock().expect("object state");
        state.listing_safety = None;
        state.fail_next_delete = true;
    }
    assert_eq!(
        store.cleanup_orphans(job_id, 10_000_001).await,
        Err(MediaObjectStoreError::BackendUnavailable)
    );
    assert!(object_backend.contains(orphan));
    let report = store
        .cleanup_orphans(job_id, 10_000_002)
        .await
        .expect("resumed cleanup");
    assert!(report.checkpoint.completed());
    assert_eq!(report.checkpoint.deleted_objects(), 1);
    assert!(!object_backend.contains(orphan));
    assert_eq!(
        object_backend
            .state
            .lock()
            .expect("object state")
            .delete_attempts,
        2
    );
}
