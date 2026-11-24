use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use collab::{
    collaboration_command::{
        CURRENT_COMMAND_CONTRACT_VERSION, CommandAdapter, DomainCommand, DomainCommandReceipt,
        DomainCommandSink, DomainCommandSubmissionError,
    },
    nostr::ingress::{
        CURRENT_NOSTR_INGRESS_VERSION, NostrIngressDeployment, NostrIngressError,
        NostrIngressRequest, VersionedNostrIngress,
    },
    tenant_admission::{AuthorizedRpcRequest, bind_rpc_tenant},
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, OperationId, PrincipalId,
    PrincipalScopes, ServiceAccountId, TrustedTenantRoute,
};
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn admission() -> AuthorizedRpcRequest {
    let community_id = community(1);
    let principal_id = principal(2);
    let tenant = bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "nostr-ingress-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant");
    let required_scope = AuthorizationScope::new("events:write").expect("scope");
    let authenticated = AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(3),
        PrincipalScopes::new([required_scope.clone()]).expect("scopes"),
    );
    AuthorizedRpcRequest::authorize(&AuthorizationRequest {
        tenant: &tenant,
        principal: &authenticated,
        required_scope: &required_scope,
        action: AuthorizationAction::Write,
        resource: AuthorizationResource {
            community_id,
            kind: AuthorizationResourceKind::Community,
            resource_id: AggregateId::from_uuid(Uuid::from_u128(4)),
            owner_principal_id: None,
            channel_id: None,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(CommunityMembership {
            community_id,
            principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: 100,
    })
    .expect("authorized admission")
}

#[derive(Default)]
struct RecordingState {
    writes: AtomicUsize,
    commands: Mutex<Vec<DomainCommand<String>>>,
}

#[derive(Clone, Default)]
struct RecordingSink(Arc<RecordingState>);

#[async_trait]
impl DomainCommandSink<String> for RecordingSink {
    async fn submit(
        &self,
        command: DomainCommand<String>,
    ) -> Result<DomainCommandReceipt, DomainCommandSubmissionError> {
        self.0.writes.fetch_add(1, Ordering::SeqCst);
        let receipt = DomainCommandReceipt::new(command.operation_id(), AggregateVersion::FIRST);
        self.0
            .commands
            .lock()
            .expect("recording sink lock")
            .push(command);
        Ok(receipt)
    }
}

fn request(version: u16, deployment: NostrIngressDeployment) -> NostrIngressRequest<String> {
    NostrIngressRequest::new(
        version,
        CURRENT_NOSTR_INGRESS_VERSION,
        OperationId::from_uuid(Uuid::from_u128(5)),
        Some(AggregateVersion::FIRST),
        None,
        deployment,
        "verified-event".to_owned(),
    )
}

#[tokio::test]
async fn nostr_ingress_version_rejects_unsupported_versions_before_a_write() {
    let sink = RecordingSink::default();
    let ingress = VersionedNostrIngress::new(sink.clone());

    let error = ingress
        .submit(
            admission(),
            request(
                CURRENT_NOSTR_INGRESS_VERSION + 1,
                NostrIngressDeployment::TemporarySidecar,
            ),
        )
        .await
        .expect_err("future version must be rejected");
    assert_eq!(
        error,
        NostrIngressError::UnsupportedVersion {
            minimum: CURRENT_NOSTR_INGRESS_VERSION,
            current: CURRENT_NOSTR_INGRESS_VERSION,
        }
    );
    assert_eq!(sink.0.writes.load(Ordering::SeqCst), 0);

    let receipt = ingress
        .submit(
            admission(),
            request(
                CURRENT_NOSTR_INGRESS_VERSION,
                NostrIngressDeployment::TemporarySidecar,
            ),
        )
        .await
        .expect("current version accepted");
    assert_eq!(
        receipt.operation_id(),
        OperationId::from_uuid(Uuid::from_u128(5))
    );
    assert_eq!(sink.0.writes.load(Ordering::SeqCst), 1);
    let commands = sink.0.commands.lock().expect("recording sink lock");
    let command = commands.first().expect("one command");
    assert_eq!(command.contract_version(), CURRENT_COMMAND_CONTRACT_VERSION);
    assert_eq!(command.tenant().community_id(), community(1));
    assert_eq!(command.principal().principal_id(), principal(2));
    assert_eq!(
        command.originating_adapter(),
        CommandAdapter::NostrTemporarySidecar
    );
    assert_eq!(command.payload(), "verified-event");
}
