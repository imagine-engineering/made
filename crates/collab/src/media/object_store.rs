use std::{error::Error, fmt, fs::File};

use async_trait::async_trait;
use collaboration_domain::{
    AuthenticatedPrincipalKind, AuthorizationAction, AuthorizationDecision, AuthorizationRequest,
    AuthorizationResource, AuthorizationResourceKind, MediaByteSize, MediaContentHash,
    MediaContentType, MediaDescriptor, MediaIdentity, MediaMetadata, MediaTenantPath, OperationId,
    PrincipalId, TenantContext, authorize,
};
use uuid::Uuid;

use super::{
    upload_admission::{MAX_MEDIA_UPLOAD_ADMISSION_MILLIS, MediaUploadAdmissionError},
    validation::{MediaValidationError, ValidatedMedia},
};

pub const MEDIA_READ_SCOPE: &str = "media:read";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaObjectStoreLimits {
    max_range_bytes: u64,
    cleanup_page_size: u32,
    max_cleanup_objects: u64,
    orphan_grace_millis: u64,
}

impl MediaObjectStoreLimits {
    pub fn new(
        max_range_bytes: u64,
        cleanup_page_size: u32,
        max_cleanup_objects: u64,
        orphan_grace_millis: u64,
    ) -> Result<Self, MediaObjectStoreError> {
        if max_range_bytes == 0
            || cleanup_page_size == 0
            || max_cleanup_objects < u64::from(cleanup_page_size)
            || orphan_grace_millis <= MAX_MEDIA_UPLOAD_ADMISSION_MILLIS
        {
            return Err(MediaObjectStoreError::InvalidConfiguration);
        }
        Ok(Self {
            max_range_bytes,
            cleanup_page_size,
            max_cleanup_objects,
            orphan_grace_millis,
        })
    }

    pub const fn max_range_bytes(self) -> u64 {
        self.max_range_bytes
    }

    pub const fn cleanup_page_size(self) -> u32 {
        self.cleanup_page_size
    }

    pub const fn max_cleanup_objects(self) -> u64 {
        self.max_cleanup_objects
    }

    pub const fn orphan_grace_millis(self) -> u64 {
        self.orphan_grace_millis
    }
}

impl Default for MediaObjectStoreLimits {
    fn default() -> Self {
        Self {
            max_range_bytes: 8 * 1024 * 1024,
            cleanup_page_size: 1_000,
            max_cleanup_objects: 100_000,
            orphan_grace_millis: 2 * 60 * 60 * 1_000,
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct MediaObjectVersion(String);

impl MediaObjectVersion {
    pub fn new(value: impl Into<String>) -> Result<Self, MediaObjectStoreError> {
        let value = value.into();
        if value.is_empty() || value.len() > 1_024 || value.chars().any(char::is_control) {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for MediaObjectVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("MediaObjectVersion([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredMediaObject {
    descriptor: MediaDescriptor,
    version: MediaObjectVersion,
    created_at_millis: u64,
}

impl StoredMediaObject {
    pub fn new(
        descriptor: MediaDescriptor,
        version: MediaObjectVersion,
        created_at_millis: u64,
    ) -> Result<Self, MediaObjectStoreError> {
        if created_at_millis == 0 {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(Self {
            descriptor,
            version,
            created_at_millis,
        })
    }

    pub const fn descriptor(&self) -> &MediaDescriptor {
        &self.descriptor
    }

    pub const fn version(&self) -> &MediaObjectVersion {
        &self.version
    }

    pub const fn created_at_millis(&self) -> u64 {
        self.created_at_millis
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MediaObjectWriteOutcome {
    Created(StoredMediaObject),
    Existing(StoredMediaObject),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaResolvedRange {
    start: u64,
    end_inclusive: u64,
}

impl MediaResolvedRange {
    pub const fn start(self) -> u64 {
        self.start
    }

    pub const fn end_inclusive(self) -> u64 {
        self.end_inclusive
    }

    pub fn byte_length(self) -> Result<u64, MediaObjectStoreError> {
        self.end_inclusive
            .checked_sub(self.start)
            .and_then(|length| length.checked_add(1))
            .ok_or(MediaObjectStoreError::InvalidRange)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaRangeRequest {
    start: u64,
    end_inclusive: u64,
}

impl MediaRangeRequest {
    pub fn new(start: u64, end_inclusive: u64) -> Result<Self, MediaObjectStoreError> {
        if start > end_inclusive {
            return Err(MediaObjectStoreError::InvalidRange);
        }
        Ok(Self {
            start,
            end_inclusive,
        })
    }

    fn resolve(
        self,
        total_size: MediaByteSize,
        max_range_bytes: u64,
    ) -> Result<MediaResolvedRange, MediaObjectStoreError> {
        let range = MediaResolvedRange {
            start: self.start,
            end_inclusive: self.end_inclusive,
        };
        if self.end_inclusive >= total_size.get() || range.byte_length()? > max_range_bytes {
            return Err(MediaObjectStoreError::InvalidRange);
        }
        Ok(range)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaRangeResponse {
    content_type: MediaContentType,
    range: MediaResolvedRange,
    total_size: MediaByteSize,
    bytes: Vec<u8>,
}

impl MediaRangeResponse {
    pub const fn content_type(&self) -> &MediaContentType {
        &self.content_type
    }

    pub const fn range(&self) -> MediaResolvedRange {
        self.range
    }

    pub const fn total_size(&self) -> MediaByteSize {
        self.total_size
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaPublication {
    operation_id: OperationId,
    metadata: MediaMetadata,
    object_version: MediaObjectVersion,
}

impl MediaPublication {
    pub const fn operation_id(&self) -> OperationId {
        self.operation_id
    }

    pub const fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    pub const fn object_version(&self) -> &MediaObjectVersion {
        &self.object_version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MediaPublicationOutcome {
    Published,
    Existing(MediaPublication),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizedMediaObject {
    path: MediaTenantPath,
    descriptor: MediaDescriptor,
    object_version: MediaObjectVersion,
}

impl AuthorizedMediaObject {
    pub fn new(
        path: MediaTenantPath,
        descriptor: MediaDescriptor,
        object_version: MediaObjectVersion,
    ) -> Result<Self, MediaObjectStoreError> {
        if path.content_hash() != descriptor.content_hash() {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(Self {
            path,
            descriptor,
            object_version,
        })
    }

    pub const fn path(&self) -> MediaTenantPath {
        self.path
    }

    pub const fn descriptor(&self) -> &MediaDescriptor {
        &self.descriptor
    }

    pub const fn object_version(&self) -> &MediaObjectVersion {
        &self.object_version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaCleanupCandidate {
    content_hash: MediaContentHash,
    byte_size: MediaByteSize,
    object_version: MediaObjectVersion,
    created_at_millis: u64,
}

impl MediaCleanupCandidate {
    pub fn new(
        content_hash: MediaContentHash,
        byte_size: MediaByteSize,
        object_version: MediaObjectVersion,
        created_at_millis: u64,
    ) -> Result<Self, MediaObjectStoreError> {
        if created_at_millis == 0 {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(Self {
            content_hash,
            byte_size,
            object_version,
            created_at_millis,
        })
    }

    pub const fn content_hash(&self) -> MediaContentHash {
        self.content_hash
    }

    pub const fn byte_size(&self) -> MediaByteSize {
        self.byte_size
    }

    pub const fn object_version(&self) -> &MediaObjectVersion {
        &self.object_version
    }

    pub const fn created_at_millis(&self) -> u64 {
        self.created_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaListingSafety {
    KnownUnversioned,
    UnknownObjectShape,
    Incomplete,
    Versioned,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaObjectPage {
    objects: Vec<MediaCleanupCandidate>,
    next_after: Option<MediaContentHash>,
    safety: MediaListingSafety,
}

impl MediaObjectPage {
    pub fn new(
        objects: Vec<MediaCleanupCandidate>,
        next_after: Option<MediaContentHash>,
        safety: MediaListingSafety,
    ) -> Self {
        Self {
            objects,
            next_after,
            safety,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaCleanupCheckpointFields {
    pub job_id: Uuid,
    pub checkpoint_version: u64,
    pub scan_started_at_millis: u64,
    pub cursor: Option<MediaContentHash>,
    pub inspected_objects: u64,
    pub deleted_objects: u64,
    pub completed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaCleanupCheckpoint {
    job_id: Uuid,
    checkpoint_version: u64,
    scan_started_at_millis: u64,
    cursor: Option<MediaContentHash>,
    inspected_objects: u64,
    deleted_objects: u64,
    completed: bool,
}

impl MediaCleanupCheckpoint {
    pub fn initial(
        job_id: Uuid,
        scan_started_at_millis: u64,
    ) -> Result<Self, MediaObjectStoreError> {
        Self::from_record(MediaCleanupCheckpointFields {
            job_id,
            checkpoint_version: 0,
            scan_started_at_millis,
            cursor: None,
            inspected_objects: 0,
            deleted_objects: 0,
            completed: false,
        })
    }

    pub fn from_record(
        fields: MediaCleanupCheckpointFields,
    ) -> Result<Self, MediaObjectStoreError> {
        if fields.job_id.is_nil()
            || fields.scan_started_at_millis == 0
            || fields.deleted_objects > fields.inspected_objects
            || fields.checkpoint_version != fields.inspected_objects + u64::from(fields.completed)
            || (fields.inspected_objects == 0 && fields.cursor.is_some())
            || (fields.inspected_objects > 0 && fields.cursor.is_none())
        {
            return Err(MediaObjectStoreError::InvalidCheckpoint);
        }
        Ok(Self {
            job_id: fields.job_id,
            checkpoint_version: fields.checkpoint_version,
            scan_started_at_millis: fields.scan_started_at_millis,
            cursor: fields.cursor,
            inspected_objects: fields.inspected_objects,
            deleted_objects: fields.deleted_objects,
            completed: fields.completed,
        })
    }

    pub const fn job_id(self) -> Uuid {
        self.job_id
    }

    pub const fn checkpoint_version(self) -> u64 {
        self.checkpoint_version
    }

    pub const fn scan_started_at_millis(self) -> u64 {
        self.scan_started_at_millis
    }

    pub const fn cursor(self) -> Option<MediaContentHash> {
        self.cursor
    }

    pub const fn inspected_objects(self) -> u64 {
        self.inspected_objects
    }

    pub const fn deleted_objects(self) -> u64 {
        self.deleted_objects
    }

    pub const fn completed(self) -> bool {
        self.completed
    }

    fn advance(
        self,
        content_hash: MediaContentHash,
        deleted: bool,
    ) -> Result<Self, MediaObjectStoreError> {
        if self.completed || self.cursor.is_some_and(|cursor| content_hash <= cursor) {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(Self {
            checkpoint_version: self
                .checkpoint_version
                .checked_add(1)
                .ok_or(MediaObjectStoreError::InvalidCheckpoint)?,
            cursor: Some(content_hash),
            inspected_objects: self
                .inspected_objects
                .checked_add(1)
                .ok_or(MediaObjectStoreError::InvalidCheckpoint)?,
            deleted_objects: self
                .deleted_objects
                .checked_add(u64::from(deleted))
                .ok_or(MediaObjectStoreError::InvalidCheckpoint)?,
            ..self
        })
    }

    fn complete(self) -> Result<Self, MediaObjectStoreError> {
        if self.completed {
            return Ok(self);
        }
        Ok(Self {
            checkpoint_version: self
                .checkpoint_version
                .checked_add(1)
                .ok_or(MediaObjectStoreError::InvalidCheckpoint)?,
            completed: true,
            ..self
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaOrphanDeletionLease {
    job_id: Uuid,
    content_hash: MediaContentHash,
    object_version: MediaObjectVersion,
    reference_version: u64,
}

impl MediaOrphanDeletionLease {
    pub fn new(
        job_id: Uuid,
        candidate: &MediaCleanupCandidate,
        reference_version: u64,
    ) -> Result<Self, MediaObjectStoreError> {
        if job_id.is_nil() || reference_version == 0 {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(Self {
            job_id,
            content_hash: candidate.content_hash,
            object_version: candidate.object_version.clone(),
            reference_version,
        })
    }

    pub const fn job_id(&self) -> Uuid {
        self.job_id
    }

    pub const fn content_hash(&self) -> MediaContentHash {
        self.content_hash
    }

    pub const fn object_version(&self) -> &MediaObjectVersion {
        &self.object_version
    }

    pub const fn reference_version(&self) -> u64 {
        self.reference_version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MediaOrphanReservationOutcome {
    Referenced,
    Reserved(MediaOrphanDeletionLease),
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaObjectDeleteOutcome {
    Deleted,
    AlreadyMissing,
    PreconditionFailed,
    VersionArtifact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaOrphanFinalization {
    Deleted,
    Preserved,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaCheckpointCommitOutcome {
    Committed,
    AlreadyCommitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaCleanupReport {
    pub checkpoint: MediaCleanupCheckpoint,
    pub reached_run_limit: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaObjectBackendError {
    Unavailable,
    InvalidData,
}

impl fmt::Display for MediaObjectBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("media object backend is unavailable")
    }
}

impl Error for MediaObjectBackendError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaObjectAuthorityError {
    Unavailable,
    Conflict,
    InvalidData,
    OutcomeUnknown,
}

impl fmt::Display for MediaObjectAuthorityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("media object authority is unavailable")
    }
}

impl Error for MediaObjectAuthorityError {}

#[async_trait]
pub trait MediaObjectBackend: Send + Sync {
    async fn put_if_absent(
        &self,
        descriptor: &MediaDescriptor,
        reader: File,
    ) -> Result<MediaObjectWriteOutcome, MediaObjectBackendError>;

    async fn get_range(
        &self,
        descriptor: &MediaDescriptor,
        object_version: &MediaObjectVersion,
        range: MediaResolvedRange,
    ) -> Result<Option<Vec<u8>>, MediaObjectBackendError>;

    async fn list_page(
        &self,
        after: Option<MediaContentHash>,
        limit: u32,
    ) -> Result<MediaObjectPage, MediaObjectBackendError>;

    async fn delete_if_match(
        &self,
        content_hash: MediaContentHash,
        object_version: &MediaObjectVersion,
    ) -> Result<MediaObjectDeleteOutcome, MediaObjectBackendError>;
}

#[async_trait]
pub trait MediaObjectAuthority: Send + Sync {
    async fn publish(
        &self,
        tenant: &TenantContext,
        publication: &MediaPublication,
    ) -> Result<MediaPublicationOutcome, MediaObjectAuthorityError>;

    async fn resolve_for_read(
        &self,
        tenant: &TenantContext,
        principal_id: PrincipalId,
        resource: AuthorizationResource,
        path: MediaTenantPath,
    ) -> Result<Option<AuthorizedMediaObject>, MediaObjectAuthorityError>;

    async fn load_cleanup_checkpoint(
        &self,
        job_id: Uuid,
        scan_started_at_millis: u64,
    ) -> Result<MediaCleanupCheckpoint, MediaObjectAuthorityError>;

    async fn reserve_orphan_deletion(
        &self,
        checkpoint: MediaCleanupCheckpoint,
        candidate: &MediaCleanupCandidate,
        orphan_grace_millis: u64,
    ) -> Result<MediaOrphanReservationOutcome, MediaObjectAuthorityError>;

    async fn finalize_orphan_deletion(
        &self,
        lease: &MediaOrphanDeletionLease,
        outcome: MediaOrphanFinalization,
    ) -> Result<(), MediaObjectAuthorityError>;

    async fn commit_cleanup_checkpoint(
        &self,
        expected: MediaCleanupCheckpoint,
        next: MediaCleanupCheckpoint,
    ) -> Result<MediaCheckpointCommitOutcome, MediaObjectAuthorityError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaStoreOutcome {
    Published,
    Replayed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaObjectStoreError {
    InvalidConfiguration,
    InvalidRequest,
    UnauthorizedOrNotFound,
    InvalidRange,
    PublicationConflict,
    InvalidBackendData,
    InvalidCheckpoint,
    UnsafeCleanup,
    BackendUnavailable,
    AuthorityUnavailable,
    Admission(MediaUploadAdmissionError),
    Validation(MediaValidationError),
}

impl fmt::Display for MediaObjectStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "media object-store configuration is invalid",
            Self::InvalidRequest => "media object-store request is invalid",
            Self::UnauthorizedOrNotFound => "media object is unavailable",
            Self::InvalidRange => "media byte range is invalid or unsatisfiable",
            Self::PublicationConflict => "media publication conflicts with durable state",
            Self::InvalidBackendData => "media object backend data is invalid",
            Self::InvalidCheckpoint => "media cleanup checkpoint is invalid",
            Self::UnsafeCleanup => "media cleanup cannot safely continue",
            Self::BackendUnavailable => "media object backend is unavailable",
            Self::AuthorityUnavailable => "media object authority is unavailable",
            Self::Admission(error) => return error.fmt(formatter),
            Self::Validation(error) => return error.fmt(formatter),
        })
    }
}

impl Error for MediaObjectStoreError {}

impl From<MediaUploadAdmissionError> for MediaObjectStoreError {
    fn from(error: MediaUploadAdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl From<MediaValidationError> for MediaObjectStoreError {
    fn from(error: MediaValidationError) -> Self {
        Self::Validation(error)
    }
}

pub struct MediaObjectStore<ObjectBackend, Authority> {
    object_backend: ObjectBackend,
    authority: Authority,
    limits: MediaObjectStoreLimits,
}

impl<ObjectBackend, Authority> MediaObjectStore<ObjectBackend, Authority>
where
    ObjectBackend: MediaObjectBackend,
    Authority: MediaObjectAuthority,
{
    pub fn new(
        object_backend: ObjectBackend,
        authority: Authority,
        limits: MediaObjectStoreLimits,
    ) -> Result<Self, MediaObjectStoreError> {
        let limits = MediaObjectStoreLimits::new(
            limits.max_range_bytes,
            limits.cleanup_page_size,
            limits.max_cleanup_objects,
            limits.orphan_grace_millis,
        )?;
        Ok(Self {
            object_backend,
            authority,
            limits,
        })
    }

    pub async fn store_validated(
        &self,
        tenant: &TenantContext,
        principal_id: PrincipalId,
        now_millis: u64,
        validated: ValidatedMedia,
    ) -> Result<MediaStoreOutcome, MediaObjectStoreError> {
        let admission =
            validated
                .admission()
                .validate_for_processing(tenant, principal_id, now_millis)?;
        let descriptor = MediaDescriptor::new(
            admission.content_hash(),
            validated.content_type().clone(),
            admission.byte_size(),
        );
        let metadata = MediaMetadata::new(
            MediaIdentity::new(admission.community_id(), admission.content_hash())
                .map_err(|_| MediaObjectStoreError::InvalidRequest)?,
            principal_id,
            validated.content_type().clone(),
            admission.byte_size(),
            admission.admitted_at_millis(),
        )
        .map_err(|_| MediaObjectStoreError::InvalidRequest)?;
        let reader = validated.into_reader()?;
        let stored = match self
            .object_backend
            .put_if_absent(&descriptor, reader)
            .await
            .map_err(map_object_backend_error)?
        {
            MediaObjectWriteOutcome::Created(stored)
            | MediaObjectWriteOutcome::Existing(stored) => stored,
        };
        validate_stored_object(&stored, &descriptor, now_millis)?;
        let publication = MediaPublication {
            operation_id: admission.operation_id(),
            metadata,
            object_version: stored.version,
        };
        match self
            .authority
            .publish(tenant, &publication)
            .await
            .map_err(map_authority_error)?
        {
            MediaPublicationOutcome::Published => Ok(MediaStoreOutcome::Published),
            MediaPublicationOutcome::Existing(existing) if existing == publication => {
                Ok(MediaStoreOutcome::Replayed)
            }
            MediaPublicationOutcome::Existing(_) => Err(MediaObjectStoreError::PublicationConflict),
        }
    }

    pub async fn read_range(
        &self,
        authorization: &AuthorizationRequest<'_>,
        path: MediaTenantPath,
        requested_range: MediaRangeRequest,
    ) -> Result<MediaRangeResponse, MediaObjectStoreError> {
        let principal_id = authorize_media_read(authorization, path)?;
        let Some(object) = self
            .authority
            .resolve_for_read(
                authorization.tenant,
                principal_id,
                authorization.resource,
                path,
            )
            .await
            .map_err(map_authority_error)?
        else {
            return Err(MediaObjectStoreError::UnauthorizedOrNotFound);
        };
        validate_authorized_object(&object, path)?;
        let range =
            requested_range.resolve(object.descriptor.byte_size(), self.limits.max_range_bytes)?;
        let expected_length = usize::try_from(range.byte_length()?)
            .map_err(|_| MediaObjectStoreError::InvalidRange)?;
        let Some(bytes) = self
            .object_backend
            .get_range(&object.descriptor, &object.object_version, range)
            .await
            .map_err(map_object_backend_error)?
        else {
            return Err(MediaObjectStoreError::BackendUnavailable);
        };
        if bytes.len() != expected_length {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        Ok(MediaRangeResponse {
            content_type: object.descriptor.content_type().clone(),
            range,
            total_size: object.descriptor.byte_size(),
            bytes,
        })
    }

    pub async fn cleanup_orphans(
        &self,
        job_id: Uuid,
        now_millis: u64,
    ) -> Result<MediaCleanupReport, MediaObjectStoreError> {
        if job_id.is_nil() || now_millis == 0 {
            return Err(MediaObjectStoreError::InvalidRequest);
        }
        let mut checkpoint = self
            .authority
            .load_cleanup_checkpoint(job_id, now_millis)
            .await
            .map_err(map_authority_error)?;
        validate_checkpoint(checkpoint, job_id, now_millis)?;
        if checkpoint.completed {
            return Ok(MediaCleanupReport {
                checkpoint,
                reached_run_limit: false,
            });
        }
        let run_start_count = checkpoint.inspected_objects;
        loop {
            if checkpoint.inspected_objects - run_start_count >= self.limits.max_cleanup_objects {
                return Ok(MediaCleanupReport {
                    checkpoint,
                    reached_run_limit: true,
                });
            }
            let page = self
                .object_backend
                .list_page(checkpoint.cursor, self.limits.cleanup_page_size)
                .await
                .map_err(map_object_backend_error)?;
            validate_cleanup_page(&page, checkpoint.cursor, self.limits.cleanup_page_size)?;
            if page.safety != MediaListingSafety::KnownUnversioned {
                return Err(MediaObjectStoreError::UnsafeCleanup);
            }
            if page.objects.is_empty() {
                if page.next_after.is_some() {
                    return Err(MediaObjectStoreError::UnsafeCleanup);
                }
                let completed = checkpoint.complete()?;
                commit_checkpoint(&self.authority, checkpoint, completed).await?;
                return Ok(MediaCleanupReport {
                    checkpoint: completed,
                    reached_run_limit: false,
                });
            }
            let has_more = page.next_after.is_some();
            for candidate in page.objects {
                if checkpoint.inspected_objects - run_start_count >= self.limits.max_cleanup_objects
                {
                    return Ok(MediaCleanupReport {
                        checkpoint,
                        reached_run_limit: true,
                    });
                }
                let old_checkpoint = checkpoint;
                let mut deleted = false;
                let old_enough = candidate
                    .created_at_millis
                    .checked_add(self.limits.orphan_grace_millis)
                    .is_some_and(|eligible_at| eligible_at <= checkpoint.scan_started_at_millis);
                if old_enough {
                    match self
                        .authority
                        .reserve_orphan_deletion(
                            checkpoint,
                            &candidate,
                            self.limits.orphan_grace_millis,
                        )
                        .await
                        .map_err(map_authority_error)?
                    {
                        MediaOrphanReservationOutcome::Referenced => {}
                        MediaOrphanReservationOutcome::Indeterminate => {
                            return Err(MediaObjectStoreError::UnsafeCleanup);
                        }
                        MediaOrphanReservationOutcome::Reserved(lease) => {
                            validate_deletion_lease(checkpoint, &candidate, &lease)?;
                            match self
                                .object_backend
                                .delete_if_match(candidate.content_hash, &candidate.object_version)
                                .await
                                .map_err(map_object_backend_error)?
                            {
                                MediaObjectDeleteOutcome::Deleted
                                | MediaObjectDeleteOutcome::AlreadyMissing => {
                                    self.authority
                                        .finalize_orphan_deletion(
                                            &lease,
                                            MediaOrphanFinalization::Deleted,
                                        )
                                        .await
                                        .map_err(map_authority_error)?;
                                    deleted = true;
                                }
                                MediaObjectDeleteOutcome::PreconditionFailed => {
                                    self.authority
                                        .finalize_orphan_deletion(
                                            &lease,
                                            MediaOrphanFinalization::Preserved,
                                        )
                                        .await
                                        .map_err(map_authority_error)?;
                                }
                                MediaObjectDeleteOutcome::VersionArtifact => {
                                    return Err(MediaObjectStoreError::UnsafeCleanup);
                                }
                            }
                        }
                    }
                }
                checkpoint = checkpoint.advance(candidate.content_hash, deleted)?;
                commit_checkpoint(&self.authority, old_checkpoint, checkpoint).await?;
            }
            if has_more && page.next_after != checkpoint.cursor {
                return Err(MediaObjectStoreError::UnsafeCleanup);
            }
            if !has_more {
                let completed = checkpoint.complete()?;
                commit_checkpoint(&self.authority, checkpoint, completed).await?;
                return Ok(MediaCleanupReport {
                    checkpoint: completed,
                    reached_run_limit: false,
                });
            }
        }
    }
}

fn authorize_media_read(
    authorization: &AuthorizationRequest<'_>,
    path: MediaTenantPath,
) -> Result<PrincipalId, MediaObjectStoreError> {
    if authorization.tenant.community_id() != path.community_id()
        || authorization.tenant.community_id() != authorization.principal.community_id()
        || authorization.resource.community_id != authorization.tenant.community_id()
        || authorization.required_scope.as_str() != MEDIA_READ_SCOPE
        || authorization.action != AuthorizationAction::Read
        || authorization.resource.kind != AuthorizationResourceKind::Media
        || authorization.resource.resource_id.as_uuid().is_nil()
    {
        return Err(MediaObjectStoreError::UnauthorizedOrNotFound);
    }
    if authorize(authorization) != AuthorizationDecision::Allowed {
        return Err(MediaObjectStoreError::UnauthorizedOrNotFound);
    }
    Ok(match authorization.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => authorization.principal.principal_id(),
    })
}

fn validate_stored_object(
    stored: &StoredMediaObject,
    descriptor: &MediaDescriptor,
    now_millis: u64,
) -> Result<(), MediaObjectStoreError> {
    if stored.descriptor != *descriptor || stored.created_at_millis > now_millis {
        return Err(MediaObjectStoreError::InvalidBackendData);
    }
    Ok(())
}

fn validate_authorized_object(
    object: &AuthorizedMediaObject,
    path: MediaTenantPath,
) -> Result<(), MediaObjectStoreError> {
    if object.path != path || object.descriptor.content_hash() != path.content_hash() {
        return Err(MediaObjectStoreError::InvalidBackendData);
    }
    Ok(())
}

fn validate_checkpoint(
    checkpoint: MediaCleanupCheckpoint,
    job_id: Uuid,
    now_millis: u64,
) -> Result<(), MediaObjectStoreError> {
    MediaCleanupCheckpoint::from_record(MediaCleanupCheckpointFields {
        job_id: checkpoint.job_id,
        checkpoint_version: checkpoint.checkpoint_version,
        scan_started_at_millis: checkpoint.scan_started_at_millis,
        cursor: checkpoint.cursor,
        inspected_objects: checkpoint.inspected_objects,
        deleted_objects: checkpoint.deleted_objects,
        completed: checkpoint.completed,
    })?;
    if checkpoint.job_id != job_id || checkpoint.scan_started_at_millis > now_millis {
        return Err(MediaObjectStoreError::InvalidCheckpoint);
    }
    Ok(())
}

fn validate_cleanup_page(
    page: &MediaObjectPage,
    after: Option<MediaContentHash>,
    page_size: u32,
) -> Result<(), MediaObjectStoreError> {
    if page.objects.len() > usize::try_from(page_size).unwrap_or(usize::MAX) {
        return Err(MediaObjectStoreError::InvalidBackendData);
    }
    let mut previous = after;
    for object in &page.objects {
        if previous.is_some_and(|hash| object.content_hash <= hash) {
            return Err(MediaObjectStoreError::InvalidBackendData);
        }
        previous = Some(object.content_hash);
    }
    if page.next_after.is_some() && page.next_after != previous {
        return Err(MediaObjectStoreError::InvalidBackendData);
    }
    Ok(())
}

fn validate_deletion_lease(
    checkpoint: MediaCleanupCheckpoint,
    candidate: &MediaCleanupCandidate,
    lease: &MediaOrphanDeletionLease,
) -> Result<(), MediaObjectStoreError> {
    if lease.job_id != checkpoint.job_id
        || lease.content_hash != candidate.content_hash
        || lease.object_version != candidate.object_version
        || lease.reference_version == 0
    {
        return Err(MediaObjectStoreError::InvalidBackendData);
    }
    Ok(())
}

async fn commit_checkpoint<Authority>(
    authority: &Authority,
    expected: MediaCleanupCheckpoint,
    next: MediaCleanupCheckpoint,
) -> Result<(), MediaObjectStoreError>
where
    Authority: MediaObjectAuthority,
{
    match authority
        .commit_cleanup_checkpoint(expected, next)
        .await
        .map_err(map_authority_error)?
    {
        MediaCheckpointCommitOutcome::Committed => Ok(()),
        MediaCheckpointCommitOutcome::AlreadyCommitted if next.checkpoint_version > 0 => Ok(()),
        MediaCheckpointCommitOutcome::AlreadyCommitted => {
            Err(MediaObjectStoreError::InvalidCheckpoint)
        }
    }
}

fn map_object_backend_error(error: MediaObjectBackendError) -> MediaObjectStoreError {
    match error {
        MediaObjectBackendError::Unavailable => MediaObjectStoreError::BackendUnavailable,
        MediaObjectBackendError::InvalidData => MediaObjectStoreError::InvalidBackendData,
    }
}

fn map_authority_error(error: MediaObjectAuthorityError) -> MediaObjectStoreError {
    match error {
        MediaObjectAuthorityError::Unavailable | MediaObjectAuthorityError::OutcomeUnknown => {
            MediaObjectStoreError::AuthorityUnavailable
        }
        MediaObjectAuthorityError::Conflict => MediaObjectStoreError::PublicationConflict,
        MediaObjectAuthorityError::InvalidData => MediaObjectStoreError::InvalidBackendData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_resolution_uses_checked_inclusive_arithmetic() {
        let size = MediaByteSize::new(10).expect("size");
        assert_eq!(
            MediaRangeRequest::new(2, 5)
                .expect("range")
                .resolve(size, 4)
                .expect("resolved")
                .byte_length(),
            Ok(4)
        );
        assert_eq!(
            MediaRangeRequest::new(2, 6)
                .expect("range")
                .resolve(size, 4),
            Err(MediaObjectStoreError::InvalidRange)
        );
        assert_eq!(
            MediaRangeRequest::new(9, u64::MAX)
                .expect("range")
                .resolve(size, u64::MAX),
            Err(MediaObjectStoreError::InvalidRange)
        );
    }

    #[test]
    fn cleanup_checkpoint_rejects_regression_and_overcount() {
        let job_id = Uuid::from_u128(1);
        let checkpoint = MediaCleanupCheckpoint::initial(job_id, 10).expect("checkpoint");
        let hash = MediaContentHash::from_digest([1; 32]);
        let next = checkpoint.advance(hash, true).expect("advance");
        assert_eq!(next.checkpoint_version(), 1);
        assert_eq!(next.inspected_objects(), 1);
        assert_eq!(next.deleted_objects(), 1);
        assert_eq!(
            next.advance(hash, false),
            Err(MediaObjectStoreError::InvalidBackendData)
        );
        assert_eq!(
            MediaCleanupCheckpoint::from_record(MediaCleanupCheckpointFields {
                job_id,
                checkpoint_version: 1,
                scan_started_at_millis: 10,
                cursor: Some(hash),
                inspected_objects: 1,
                deleted_objects: 2,
                completed: false,
            }),
            Err(MediaObjectStoreError::InvalidCheckpoint)
        );
    }
}
