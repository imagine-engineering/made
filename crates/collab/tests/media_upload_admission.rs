use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use collab::media::upload_admission::{
    MediaUploadAdmission, MediaUploadAdmissionBackend, MediaUploadAdmissionBackendError,
    MediaUploadAdmissionError, MediaUploadAdmissionLimits, MediaUploadAdmissionOutcome,
    MediaUploadAdmissionService, MediaUploadRequest, MediaUploadReservationOutcome,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MediaContentHash, MembershipRole, MembershipStatus,
    OperationId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use uuid::Uuid;

#[derive(Clone, Default)]
struct Backend {
    admissions: Arc<Mutex<BTreeMap<(CommunityId, OperationId), MediaUploadAdmission>>>,
    attempts: Arc<Mutex<usize>>,
}

#[async_trait]
impl MediaUploadAdmissionBackend for Backend {
    async fn reserve(
        &self,
        tenant: &TenantContext,
        admission: MediaUploadAdmission,
    ) -> Result<MediaUploadReservationOutcome, MediaUploadAdmissionBackendError> {
        *self.attempts.lock().expect("attempts") += 1;
        if tenant.community_id() != admission.community_id() {
            return Err(MediaUploadAdmissionBackendError::Conflict);
        }
        let mut admissions = self.admissions.lock().expect("admissions");
        let key = (admission.community_id(), admission.operation_id());
        if let Some(existing) = admissions.get(&key) {
            return Ok(MediaUploadReservationOutcome::Existing(*existing));
        }
        admissions.insert(key, admission);
        Ok(MediaUploadReservationOutcome::Reserved)
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
        Some(TrustedTenantRoute::from_listener(community_id, "media-upload").expect("route")),
        &[],
    )
    .expect("tenant")
}

fn operation(value: u128) -> OperationId {
    OperationId::from_uuid(Uuid::from_u128(value))
}

struct AuthorizationFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
    membership: Option<CommunityMembership>,
    operation_id: OperationId,
    now_millis: u64,
    resource_community_id: CommunityId,
}

impl AuthorizationFixture {
    fn new(community_id: CommunityId, now_millis: u64) -> Self {
        let principal_id = principal(2);
        let scope = AuthorizationScope::new("media:write").expect("scope");
        Self {
            tenant: tenant(community_id),
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(3),
                PrincipalScopes::new([scope.clone()]).expect("scopes"),
            ),
            scope,
            membership: Some(CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            operation_id: operation(4),
            now_millis,
            resource_community_id: community_id,
        }
    }

    fn request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: self.resource_community_id,
                kind: AuthorizationResourceKind::Media,
                resource_id: AggregateId::from_uuid(self.operation_id.as_uuid()),
                owner_principal_id: Some(principal(2)),
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: self.membership,
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: self.now_millis,
        }
    }

    fn upload(&self, hash: u8, byte_size: u64) -> MediaUploadRequest {
        MediaUploadRequest::new(
            self.operation_id,
            MediaContentHash::from_digest([hash; 32]),
            byte_size,
        )
        .expect("upload request")
    }
}

fn service(backend: Backend) -> MediaUploadAdmissionService<Backend> {
    MediaUploadAdmissionService::new(
        backend,
        MediaUploadAdmissionLimits::new(1_024, 1_000).expect("limits"),
    )
}

#[tokio::test]
async fn authenticated_member_receives_a_bounded_content_bound_operation() {
    let fixture = AuthorizationFixture::new(community(1), 100);
    let backend = Backend::default();
    let outcome = service(backend.clone())
        .admit(&fixture.request(), fixture.upload(7, 512))
        .await
        .expect("admission");
    let MediaUploadAdmissionOutcome::Issued(admission) = outcome else {
        panic!("first admission must be issued");
    };
    assert_eq!(admission.community_id(), community(1));
    assert_eq!(admission.principal_id(), principal(2));
    assert_eq!(admission.content_hash().as_bytes(), [7; 32]);
    assert_eq!(admission.byte_size().get(), 512);
    assert_eq!(admission.expires_at_millis(), 1_100);
    assert_eq!(*backend.attempts.lock().expect("attempts"), 1);
    let diagnostics = format!("{admission:?}");
    assert!(!diagnostics.contains(&"07".repeat(32)));
    assert!(!diagnostics.contains("credential"));
}

#[tokio::test]
async fn unauthorized_and_wrong_tenant_requests_fail_before_reservation() {
    let mut unauthorized = AuthorizationFixture::new(community(1), 100);
    unauthorized.membership = None;
    let backend = Backend::default();
    assert_eq!(
        service(backend.clone())
            .admit(&unauthorized.request(), unauthorized.upload(1, 2_048))
            .await,
        Err(MediaUploadAdmissionError::Unauthorized)
    );

    let mut foreign = AuthorizationFixture::new(community(1), 100);
    foreign.resource_community_id = community(2);
    assert_eq!(
        service(backend.clone())
            .admit(&foreign.request(), foreign.upload(1, 512))
            .await,
        Err(MediaUploadAdmissionError::TenantMismatch)
    );
    assert_eq!(*backend.attempts.lock().expect("attempts"), 0);
}

#[tokio::test]
async fn oversized_upload_fails_before_reservation() {
    let fixture = AuthorizationFixture::new(community(1), 100);
    let backend = Backend::default();
    assert_eq!(
        service(backend.clone())
            .admit(&fixture.request(), fixture.upload(1, 1_025))
            .await,
        Err(MediaUploadAdmissionError::PayloadTooLarge)
    );
    assert_eq!(*backend.attempts.lock().expect("attempts"), 0);
}

#[tokio::test]
async fn exact_replay_reuses_admission_while_changed_input_conflicts() {
    let fixture = AuthorizationFixture::new(community(1), 100);
    let backend = Backend::default();
    let service = service(backend.clone());
    let issued = service
        .admit(&fixture.request(), fixture.upload(1, 512))
        .await
        .expect("issued")
        .admission();
    let replayed = service
        .admit(&fixture.request(), fixture.upload(1, 512))
        .await
        .expect("replayed");
    assert!(replayed.replayed());
    assert_eq!(replayed.admission(), issued);
    assert_eq!(
        service
            .admit(&fixture.request(), fixture.upload(2, 512))
            .await,
        Err(MediaUploadAdmissionError::ReplayConflict)
    );
    assert_eq!(*backend.attempts.lock().expect("attempts"), 3);
}

#[tokio::test]
async fn expired_admission_cannot_be_replayed_or_processed() {
    let first = AuthorizationFixture::new(community(1), 100);
    let backend = Backend::default();
    let service = service(backend);
    let admission = service
        .admit(&first.request(), first.upload(1, 512))
        .await
        .expect("issued")
        .admission();
    let expired = AuthorizationFixture::new(community(1), admission.expires_at_millis());
    assert_eq!(
        service
            .admit(&expired.request(), expired.upload(1, 512))
            .await,
        Err(MediaUploadAdmissionError::Expired)
    );
    assert_eq!(
        admission.validate_for_processing(
            &expired.tenant,
            expired.principal.principal_id(),
            admission.expires_at_millis(),
        ),
        Err(MediaUploadAdmissionError::Expired)
    );
    let future = MediaUploadAdmission::restore(
        &expired.tenant,
        expired.principal.principal_id(),
        expired.upload(1, 512),
        admission.expires_at_millis() + 1,
        admission.expires_at_millis() + 500,
    )
    .expect("structurally valid future admission");
    assert_eq!(
        future.validate_for_processing(
            &expired.tenant,
            expired.principal.principal_id(),
            admission.expires_at_millis(),
        ),
        Err(MediaUploadAdmissionError::InvalidBackendData)
    );
}
