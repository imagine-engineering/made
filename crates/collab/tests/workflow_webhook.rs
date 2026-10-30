use std::{
    net::{IpAddr, Ipv4Addr},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use collab::workflows::{
    repository::{
        StoredWorkflowDefinition, WorkflowIdentity, WorkflowLifecycle, WorkflowProvenance,
        WorkflowRepositoryError, WorkflowRunRequest, WorkflowScope, WorkflowStoreOutcome,
    },
    triggers::{WORKFLOW_RUN_SCOPE, WorkflowRunClaimer, WorkflowTriggerAdmissionStatus},
    webhook::{
        MAX_WEBHOOK_BODY_BYTES, ResolvedWebhookCredential, WebhookAdmissionError,
        WebhookAuthentication, WebhookBodyReadError, WebhookCredentialReference,
        WebhookCredentialResolver, WebhookDnsResolver, WebhookIngressLimits, WebhookNetworkPolicy,
        WebhookRedirectPolicy, WebhookTransportPolicyError, WorkflowWebhookAdmission,
        webhook_signature_v1,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, PrincipalId,
    PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use collaboration_workflow::definition::WorkflowDefinition;
use futures::stream;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const NOW: u64 = 1_900_000_320_000;
const SECRET: &[u8; 32] = b"0123456789abcdef0123456789abcdef";

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "workflow-webhook-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn stored_definition(community_id: CommunityId) -> StoredWorkflowDefinition {
    let identity =
        WorkflowIdentity::new(community_id, Uuid::from_u128(10)).expect("workflow identity");
    StoredWorkflowDefinition {
        identity,
        definition_version: 1,
        definition: WorkflowDefinition::parse_yaml(
            r#"
version: 1
name: Webhook test
enabled: true
trigger:
  on: webhook
steps:
  - id: announce
    action: send_message
    text: admitted
"#,
        )
        .expect("workflow definition"),
        definition_sha256: [1; 32],
        creator_principal_id: principal(20),
        author_principal_id: principal(20),
        scope: WorkflowScope::Community,
        current_definition_version: 1,
        head_revision: 1,
        lifecycle: WorkflowLifecycle::Active,
        provenance: WorkflowProvenance::new("zed", "workflow:10:1", "1", NOW, None)
            .expect("provenance"),
        created_at_millis: NOW,
    }
}

struct OwnerAuthorization {
    tenant: TenantContext,
    scope: AuthorizationScope,
    principal: AuthenticatedPrincipal,
    membership: CommunityMembership,
}

impl OwnerAuthorization {
    fn new(community_id: CommunityId) -> Self {
        let owner_principal_id = principal(20);
        let scope = AuthorizationScope::new(WORKFLOW_RUN_SCOPE).expect("scope");
        Self {
            tenant: tenant(community_id),
            principal: AuthenticatedPrincipal::zed_account(
                owner_principal_id,
                community_id,
                ServiceAccountId::new(20),
                PrincipalScopes::new([scope.clone()]).expect("scopes"),
            ),
            membership: CommunityMembership {
                community_id,
                principal_id: owner_principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            scope,
        }
    }

    fn request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Workflow,
                resource_id: aggregate(10),
                owner_principal_id: Some(principal(20)),
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: NOW,
        }
    }
}

#[derive(Clone, Default)]
struct FakeRunClaimer {
    requests: Arc<Mutex<Vec<WorkflowRunRequest>>>,
}

impl FakeRunClaimer {
    fn requests(&self) -> Vec<WorkflowRunRequest> {
        self.requests.lock().expect("request lock").clone()
    }
}

#[async_trait]
impl WorkflowRunClaimer for FakeRunClaimer {
    async fn claim_run(
        &self,
        _tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        let mut requests = self.requests.lock().expect("request lock");
        if let Some(existing) = requests
            .iter()
            .find(|existing| existing.trigger_operation_id == request.trigger_operation_id)
        {
            return if existing == request {
                Ok(WorkflowStoreOutcome::Duplicate)
            } else {
                Err(WorkflowRepositoryError::IdempotencyConflict)
            };
        }
        requests.push(request.clone());
        Ok(WorkflowStoreOutcome::Applied)
    }
}

#[derive(Clone, Default)]
struct FakeCredentialResolver {
    resolutions: Arc<AtomicUsize>,
}

#[async_trait]
impl WebhookCredentialResolver for FakeCredentialResolver {
    async fn resolve(
        &self,
        _tenant: &TenantContext,
        _workflow: WorkflowIdentity,
        reference: &WebhookCredentialReference,
    ) -> Result<ResolvedWebhookCredential, WebhookAdmissionError> {
        if reference.as_str() != "credentials://workflow/webhook" {
            return Err(WebhookAdmissionError::CredentialUnavailable);
        }
        self.resolutions.fetch_add(1, Ordering::SeqCst);
        ResolvedWebhookCredential::new("current", SECRET.to_vec())
    }
}

fn credential_reference() -> WebhookCredentialReference {
    WebhookCredentialReference::new("credentials://workflow/webhook").expect("reference")
}

fn authentication(
    tenant: &TenantContext,
    definition: &StoredWorkflowDefinition,
    body: &[u8],
    idempotency_key: &str,
) -> WebhookAuthentication {
    let content_sha256: [u8; 32] = Sha256::digest(body).into();
    let signature = webhook_signature_v1(
        SECRET,
        tenant,
        definition.identity,
        NOW,
        idempotency_key,
        content_sha256,
    )
    .expect("signature");
    WebhookAuthentication::new(
        NOW,
        idempotency_key,
        &hex::encode(content_sha256),
        &hex::encode(signature),
    )
    .expect("authentication")
}

#[tokio::test]
async fn signed_webhook_replay_returns_one_stable_run() {
    let community_id = community(1);
    let owner = OwnerAuthorization::new(community_id);
    let definition = stored_definition(community_id);
    let body = br#"{"message":"ship","count":2}"#;
    let authentication = authentication(&owner.tenant, &definition, body, "delivery-42");
    let claimer = FakeRunClaimer::default();
    let admission =
        WorkflowWebhookAdmission::new(claimer.clone(), FakeCredentialResolver::default());

    let first = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &authentication,
            Some(body.len() as u64),
            stream::iter([Ok(Bytes::from_static(body))]),
            &owner.request(),
            NOW,
        )
        .await
        .expect("first admission");
    let duplicate = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &authentication,
            Some(body.len() as u64),
            stream::iter([Ok(Bytes::from_static(body))]),
            &owner.request(),
            NOW,
        )
        .await
        .expect("duplicate admission");

    assert_eq!(first.status, WorkflowTriggerAdmissionStatus::Claimed);
    assert_eq!(duplicate.status, WorkflowTriggerAdmissionStatus::Duplicate);
    assert_eq!(first.run_identity, duplicate.run_identity);
    let requests = claimer.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].trigger_source_id, "webhook:delivery-42");
    assert_eq!(requests[0].trigger_context["webhook"]["message"], "ship");
}

#[tokio::test]
async fn invalid_signature_is_rejected_before_body_polling() {
    let community_id = community(1);
    let owner = OwnerAuthorization::new(community_id);
    let definition = stored_definition(community_id);
    let body = br#"{"message":"ship"}"#;
    let content_sha256: [u8; 32] = Sha256::digest(body).into();
    let authentication = WebhookAuthentication::new(
        NOW,
        "delivery-43",
        &hex::encode(content_sha256),
        &hex::encode([0; 32]),
    )
    .expect("authentication");
    let body_polls = Arc::new(AtomicUsize::new(0));
    let body_stream = stream::poll_fn({
        let body_polls = Arc::clone(&body_polls);
        move |_| {
            body_polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(None)
        }
    });
    let claimer = FakeRunClaimer::default();
    let admission =
        WorkflowWebhookAdmission::new(claimer.clone(), FakeCredentialResolver::default());

    let result = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &authentication,
            Some(body.len() as u64),
            body_stream,
            &owner.request(),
            NOW,
        )
        .await;

    assert!(matches!(
        result,
        Err(WebhookAdmissionError::InvalidAuthentication)
    ));
    assert_eq!(body_polls.load(Ordering::SeqCst), 0);
    assert!(claimer.requests().is_empty());
}

#[tokio::test]
async fn revoked_owner_is_rejected_before_body_polling() {
    let community_id = community(1);
    let mut owner = OwnerAuthorization::new(community_id);
    owner.membership.status = MembershipStatus::Revoked;
    let definition = stored_definition(community_id);
    let body = br#"{"message":"ship"}"#;
    let authentication = authentication(&owner.tenant, &definition, body, "delivery-44");
    let body_polls = Arc::new(AtomicUsize::new(0));
    let body_stream = stream::poll_fn({
        let body_polls = Arc::clone(&body_polls);
        move |_| {
            body_polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(None)
        }
    });
    let claimer = FakeRunClaimer::default();
    let admission =
        WorkflowWebhookAdmission::new(claimer.clone(), FakeCredentialResolver::default());

    let result = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &authentication,
            Some(body.len() as u64),
            body_stream,
            &owner.request(),
            NOW,
        )
        .await;

    assert!(matches!(result, Err(WebhookAdmissionError::Trigger(_))));
    assert_eq!(body_polls.load(Ordering::SeqCst), 0);
    assert!(claimer.requests().is_empty());
}

#[tokio::test]
async fn oversized_or_reserved_body_is_rejected_without_a_run() {
    let community_id = community(1);
    let owner = OwnerAuthorization::new(community_id);
    let definition = stored_definition(community_id);
    let empty_authentication = authentication(&owner.tenant, &definition, b"", "delivery-45");
    let claimer = FakeRunClaimer::default();
    let admission =
        WorkflowWebhookAdmission::new(claimer.clone(), FakeCredentialResolver::default());

    let oversized = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &empty_authentication,
            Some((MAX_WEBHOOK_BODY_BYTES + 1) as u64),
            stream::empty(),
            &owner.request(),
            NOW,
        )
        .await;
    assert!(matches!(
        oversized,
        Err(WebhookAdmissionError::BodyTooLarge)
    ));

    let reserved_body = br#"{"trigger_author":"spoof"}"#;
    let reserved_authentication =
        authentication(&owner.tenant, &definition, reserved_body, "delivery-46");
    let reserved = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &reserved_authentication,
            None,
            stream::iter([Ok(Bytes::from_static(reserved_body))]),
            &owner.request(),
            NOW,
        )
        .await;
    assert!(matches!(reserved, Err(WebhookAdmissionError::InvalidBody)));
    assert!(claimer.requests().is_empty());
}

#[tokio::test]
async fn signed_digest_rejects_body_tampering_without_a_run() {
    let community_id = community(1);
    let owner = OwnerAuthorization::new(community_id);
    let definition = stored_definition(community_id);
    let authentication = authentication(
        &owner.tenant,
        &definition,
        br#"{"message":"approved"}"#,
        "delivery-48",
    );
    let tampered_body = br#"{"message":"tampered"}"#;
    let claimer = FakeRunClaimer::default();
    let admission =
        WorkflowWebhookAdmission::new(claimer.clone(), FakeCredentialResolver::default());

    let result = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &authentication,
            Some(tampered_body.len() as u64),
            stream::iter([Ok(Bytes::from_static(tampered_body))]),
            &owner.request(),
            NOW,
        )
        .await;

    assert!(matches!(
        result,
        Err(WebhookAdmissionError::BodyDigestMismatch)
    ));
    assert!(claimer.requests().is_empty());
}

#[tokio::test]
async fn slow_body_times_out_before_run_claim() {
    let community_id = community(1);
    let owner = OwnerAuthorization::new(community_id);
    let definition = stored_definition(community_id);
    let authentication = authentication(&owner.tenant, &definition, b"", "delivery-47");
    let claimer = FakeRunClaimer::default();
    let limits =
        WebhookIngressLimits::with_total_timeout(Duration::from_millis(10)).expect("limits");
    let admission = WorkflowWebhookAdmission::with_limits(
        claimer.clone(),
        FakeCredentialResolver::default(),
        limits,
    );

    let result = admission
        .admit(
            &owner.tenant,
            &definition,
            &credential_reference(),
            &authentication,
            None,
            stream::pending::<Result<Bytes, WebhookBodyReadError>>(),
            &owner.request(),
            NOW,
        )
        .await;

    assert!(matches!(result, Err(WebhookAdmissionError::Timeout)));
    assert!(claimer.requests().is_empty());
}

#[derive(Clone)]
struct FixedDns(Vec<IpAddr>);

#[async_trait]
impl WebhookDnsResolver for FixedDns {
    async fn resolve(
        &self,
        _host: &str,
        _port: u16,
    ) -> Result<Vec<IpAddr>, WebhookTransportPolicyError> {
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn webhook_network_policy_rejects_mixed_private_dns_and_redirects() {
    let mixed = WebhookNetworkPolicy::new(FixedDns(vec![
        IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34)),
        IpAddr::V4(Ipv4Addr::LOCALHOST),
    ]));
    let result = mixed.pin("https://example.com/hook").await;
    assert!(matches!(
        result,
        Err(WebhookTransportPolicyError::UnsafeAddress)
    ));
    assert!(matches!(
        mixed.pin("https://127.0.0.1/hook").await,
        Err(WebhookTransportPolicyError::UnsafeAddress)
    ));
    assert!(matches!(
        mixed.pin("http://example.com/hook").await,
        Err(WebhookTransportPolicyError::InvalidTarget)
    ));

    let public =
        WebhookNetworkPolicy::new(FixedDns(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]));
    let destination = public
        .pin("https://example.com/hook")
        .await
        .expect("public destination");
    assert_eq!(public.redirect_policy(), WebhookRedirectPolicy::Reject);
    assert!(!public.proxies_enabled());
    assert_eq!(
        destination.socket_address().ip(),
        Ipv4Addr::new(93, 184, 216, 34)
    );
    destination.build_client().expect("pinned client");
}
