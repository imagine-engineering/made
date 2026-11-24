use std::{error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AuthenticatedPrincipalKind, AuthorizationAction, AuthorizationDecision,
    AuthorizationDenial, AuthorizationRequest, AuthorizationResourceKind, CommunityId,
    MediaByteSize, MediaContentHash, OperationId, PrincipalId, TenantContext, authorize,
};

pub const MEDIA_UPLOAD_SCOPE: &str = "media:write";
pub const MAX_MEDIA_UPLOAD_ADMISSION_MILLIS: u64 = 60 * 60 * 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaUploadAdmissionLimits {
    max_upload_bytes: MediaByteSize,
    admission_lifetime_millis: u64,
}

impl MediaUploadAdmissionLimits {
    pub fn new(
        max_upload_bytes: u64,
        admission_lifetime_millis: u64,
    ) -> Result<Self, MediaUploadAdmissionError> {
        if admission_lifetime_millis == 0
            || admission_lifetime_millis > MAX_MEDIA_UPLOAD_ADMISSION_MILLIS
        {
            return Err(MediaUploadAdmissionError::InvalidConfiguration);
        }
        Ok(Self {
            max_upload_bytes: MediaByteSize::new(max_upload_bytes)
                .map_err(|_| MediaUploadAdmissionError::InvalidConfiguration)?,
            admission_lifetime_millis,
        })
    }

    pub const fn max_upload_bytes(self) -> MediaByteSize {
        self.max_upload_bytes
    }

    pub const fn admission_lifetime_millis(self) -> u64 {
        self.admission_lifetime_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaUploadRequest {
    operation_id: OperationId,
    content_hash: MediaContentHash,
    byte_size: MediaByteSize,
}

impl MediaUploadRequest {
    pub fn new(
        operation_id: OperationId,
        content_hash: MediaContentHash,
        byte_size: u64,
    ) -> Result<Self, MediaUploadAdmissionError> {
        if operation_id.as_uuid().is_nil() {
            return Err(MediaUploadAdmissionError::InvalidRequest);
        }
        Ok(Self {
            operation_id,
            content_hash,
            byte_size: MediaByteSize::new(byte_size)
                .map_err(|_| MediaUploadAdmissionError::InvalidRequest)?,
        })
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn content_hash(self) -> MediaContentHash {
        self.content_hash
    }

    pub const fn byte_size(self) -> MediaByteSize {
        self.byte_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MediaUploadAdmission {
    community_id: CommunityId,
    principal_id: PrincipalId,
    operation_id: OperationId,
    content_hash: MediaContentHash,
    byte_size: MediaByteSize,
    admitted_at_millis: u64,
    expires_at_millis: u64,
}

impl MediaUploadAdmission {
    pub fn restore(
        tenant: &TenantContext,
        principal_id: PrincipalId,
        request: MediaUploadRequest,
        admitted_at_millis: u64,
        expires_at_millis: u64,
    ) -> Result<Self, MediaUploadAdmissionError> {
        let admission = Self {
            community_id: tenant.community_id(),
            principal_id,
            operation_id: request.operation_id,
            content_hash: request.content_hash,
            byte_size: request.byte_size,
            admitted_at_millis,
            expires_at_millis,
        };
        admission.validate_record()?;
        Ok(admission)
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn principal_id(self) -> PrincipalId {
        self.principal_id
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn content_hash(self) -> MediaContentHash {
        self.content_hash
    }

    pub const fn byte_size(self) -> MediaByteSize {
        self.byte_size
    }

    pub const fn admitted_at_millis(self) -> u64 {
        self.admitted_at_millis
    }

    pub const fn expires_at_millis(self) -> u64 {
        self.expires_at_millis
    }

    pub fn validate_for_processing(
        self,
        tenant: &TenantContext,
        principal_id: PrincipalId,
        now_millis: u64,
    ) -> Result<Self, MediaUploadAdmissionError> {
        self.validate_record()?;
        if tenant.community_id() != self.community_id || principal_id != self.principal_id {
            return Err(MediaUploadAdmissionError::TenantMismatch);
        }
        if now_millis == 0 {
            return Err(MediaUploadAdmissionError::InvalidRequest);
        }
        if now_millis < self.admitted_at_millis {
            return Err(MediaUploadAdmissionError::InvalidBackendData);
        }
        if now_millis >= self.expires_at_millis {
            return Err(MediaUploadAdmissionError::Expired);
        }
        Ok(self)
    }

    fn validate_record(self) -> Result<(), MediaUploadAdmissionError> {
        if self.community_id.as_uuid().is_nil()
            || self.principal_id.as_uuid().is_nil()
            || self.operation_id.as_uuid().is_nil()
            || self.admitted_at_millis == 0
            || self.expires_at_millis <= self.admitted_at_millis
            || self.expires_at_millis - self.admitted_at_millis > MAX_MEDIA_UPLOAD_ADMISSION_MILLIS
        {
            return Err(MediaUploadAdmissionError::InvalidBackendData);
        }
        Ok(())
    }

    fn same_operation(self, candidate: Self) -> bool {
        self.community_id == candidate.community_id
            && self.principal_id == candidate.principal_id
            && self.operation_id == candidate.operation_id
            && self.content_hash == candidate.content_hash
            && self.byte_size == candidate.byte_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaUploadAdmissionOutcome {
    Issued(MediaUploadAdmission),
    Replayed(MediaUploadAdmission),
}

impl MediaUploadAdmissionOutcome {
    pub const fn admission(self) -> MediaUploadAdmission {
        match self {
            Self::Issued(admission) | Self::Replayed(admission) => admission,
        }
    }

    pub const fn replayed(self) -> bool {
        matches!(self, Self::Replayed(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaUploadReservationOutcome {
    Reserved,
    Existing(MediaUploadAdmission),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaUploadAdmissionBackendError {
    Unavailable,
    Conflict,
    OutcomeUnknown,
}

impl fmt::Display for MediaUploadAdmissionBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "media upload admission backend is unavailable",
            Self::Conflict => "media upload admission conflicts with durable state",
            Self::OutcomeUnknown => "media upload admission outcome is unknown",
        })
    }
}

impl Error for MediaUploadAdmissionBackendError {}

#[async_trait]
pub trait MediaUploadAdmissionBackend: Send + Sync {
    async fn reserve(
        &self,
        tenant: &TenantContext,
        admission: MediaUploadAdmission,
    ) -> Result<MediaUploadReservationOutcome, MediaUploadAdmissionBackendError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaUploadAdmissionError {
    InvalidConfiguration,
    InvalidRequest,
    Unauthorized,
    TenantMismatch,
    PayloadTooLarge,
    Expired,
    ReplayConflict,
    InvalidBackendData,
    Backend(MediaUploadAdmissionBackendError),
}

impl fmt::Display for MediaUploadAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "media upload admission configuration is invalid",
            Self::InvalidRequest => "media upload request is invalid",
            Self::Unauthorized => "media upload is not authorized",
            Self::TenantMismatch => "media upload tenant does not match",
            Self::PayloadTooLarge => "media upload exceeds its configured size limit",
            Self::Expired => "media upload admission is expired",
            Self::ReplayConflict => "media upload operation was reused with different input",
            Self::InvalidBackendData => "media upload admission backend data is invalid",
            Self::Backend(error) => return error.fmt(formatter),
        })
    }
}

impl Error for MediaUploadAdmissionError {}

impl From<MediaUploadAdmissionBackendError> for MediaUploadAdmissionError {
    fn from(error: MediaUploadAdmissionBackendError) -> Self {
        Self::Backend(error)
    }
}

pub struct MediaUploadAdmissionService<Backend> {
    backend: Backend,
    limits: MediaUploadAdmissionLimits,
}

impl<Backend> MediaUploadAdmissionService<Backend>
where
    Backend: MediaUploadAdmissionBackend,
{
    pub const fn new(backend: Backend, limits: MediaUploadAdmissionLimits) -> Self {
        Self { backend, limits }
    }

    pub async fn admit(
        &self,
        authorization: &AuthorizationRequest<'_>,
        request: MediaUploadRequest,
    ) -> Result<MediaUploadAdmissionOutcome, MediaUploadAdmissionError> {
        let principal_id = validate_authorization_shape(authorization, request)?;
        match authorize(authorization) {
            AuthorizationDecision::Allowed => {}
            AuthorizationDecision::Denied(AuthorizationDenial::TenantMismatch) => {
                return Err(MediaUploadAdmissionError::TenantMismatch);
            }
            AuthorizationDecision::Denied(_) => {
                return Err(MediaUploadAdmissionError::Unauthorized);
            }
        }
        if request.byte_size > self.limits.max_upload_bytes {
            return Err(MediaUploadAdmissionError::PayloadTooLarge);
        }
        if authorization.now_millis == 0 {
            return Err(MediaUploadAdmissionError::InvalidRequest);
        }
        let expires_at_millis = authorization
            .now_millis
            .checked_add(self.limits.admission_lifetime_millis)
            .ok_or(MediaUploadAdmissionError::InvalidRequest)?;
        let candidate = MediaUploadAdmission::restore(
            authorization.tenant,
            principal_id,
            request,
            authorization.now_millis,
            expires_at_millis,
        )?;
        match self
            .backend
            .reserve(authorization.tenant, candidate)
            .await?
        {
            MediaUploadReservationOutcome::Reserved => {
                Ok(MediaUploadAdmissionOutcome::Issued(candidate))
            }
            MediaUploadReservationOutcome::Existing(existing) => {
                existing.validate_record()?;
                if !existing.same_operation(candidate) {
                    return Err(MediaUploadAdmissionError::ReplayConflict);
                }
                let existing = existing.validate_for_processing(
                    authorization.tenant,
                    principal_id,
                    authorization.now_millis,
                )?;
                Ok(MediaUploadAdmissionOutcome::Replayed(existing))
            }
        }
    }
}

fn validate_authorization_shape(
    authorization: &AuthorizationRequest<'_>,
    request: MediaUploadRequest,
) -> Result<PrincipalId, MediaUploadAdmissionError> {
    if authorization.tenant.community_id() != authorization.principal.community_id()
        || authorization.resource.community_id != authorization.tenant.community_id()
    {
        return Err(MediaUploadAdmissionError::TenantMismatch);
    }
    let principal_id = match authorization.principal.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => authorization.principal.principal_id(),
    };
    if principal_id.as_uuid().is_nil()
        || authorization.required_scope.as_str() != MEDIA_UPLOAD_SCOPE
        || authorization.action != AuthorizationAction::Write
        || authorization.resource.kind != AuthorizationResourceKind::Media
        || authorization.resource.resource_id
            != AggregateId::from_uuid(request.operation_id.as_uuid())
        || authorization.resource.owner_principal_id != Some(principal_id)
        || authorization.resource.channel_id.is_some()
    {
        return Err(MediaUploadAdmissionError::Unauthorized);
    }
    Ok(principal_id)
}
