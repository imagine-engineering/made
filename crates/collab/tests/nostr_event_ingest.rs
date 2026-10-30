use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use collab::{
    collaboration_command::{
        DomainCommand, DomainCommandReceipt, DomainCommandSink, DomainCommandSubmissionError,
    },
    nostr::{
        event_ingest::{
            NostrEventCommand, NostrEventFrameErrorKind, NostrEventIngestStatus, NostrEventIngress,
        },
        ingress::NostrIngressDeployment,
    },
    tenant_admission::{AuthorizedRpcRequest, bind_rpc_tenant},
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, NostrAuthenticationMethod,
    NostrPublicKey, PrincipalId, PrincipalScopes, TrustedTenantRoute,
};
use serde_json::{Value, json};
use uuid::Uuid;

const EVENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
));

fn community() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn event_fixture(name: &str) -> Value {
    let fixture: Value = serde_json::from_str(EVENTS).expect("valid frozen event corpus");
    fixture["events"][name].clone()
}

fn event_frame(event: &Value) -> String {
    json!(["EVENT", event]).to_string()
}

fn admission(public_key: NostrPublicKey) -> AuthorizedRpcRequest {
    let community_id = community();
    let principal_id = PrincipalId::from_uuid(Uuid::from_u128(2));
    let tenant = bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "nostr-event-ingest-test")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant");
    let required_scope = AuthorizationScope::new("events:write").expect("scope");
    let principal = AuthenticatedPrincipal::nostr_identity(
        principal_id,
        community_id,
        public_key,
        NostrAuthenticationMethod::Nip42,
        PrincipalScopes::new([required_scope.clone()]).expect("principal scopes"),
    );
    AuthorizedRpcRequest::authorize(&AuthorizationRequest {
        tenant: &tenant,
        principal: &principal,
        required_scope: &required_scope,
        action: AuthorizationAction::Write,
        resource: AuthorizationResource {
            community_id,
            kind: AuthorizationResourceKind::Community,
            resource_id: AggregateId::from_uuid(Uuid::from_u128(3)),
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

fn event_author(event: &Value) -> NostrPublicKey {
    let bytes =
        hex::decode(event["pubkey"].as_str().expect("event public key")).expect("hex public key");
    NostrPublicKey::from_bytes(bytes.try_into().expect("32-byte public key"))
}

#[derive(Clone, Copy, Default)]
enum SinkMode {
    #[default]
    Applied,
    Duplicate,
    Rejected,
    Unavailable,
}

#[derive(Default)]
struct SinkState {
    writes: AtomicUsize,
    mode: Mutex<SinkMode>,
    commands: Mutex<Vec<DomainCommand<NostrEventCommand>>>,
}

#[derive(Clone, Default)]
struct RecordingSink(Arc<SinkState>);

impl RecordingSink {
    fn set_mode(&self, mode: SinkMode) {
        *self.0.mode.lock().expect("sink mode lock") = mode;
    }
}

#[async_trait]
impl DomainCommandSink<NostrEventCommand> for RecordingSink {
    async fn submit(
        &self,
        command: DomainCommand<NostrEventCommand>,
    ) -> Result<DomainCommandReceipt, DomainCommandSubmissionError> {
        self.0.writes.fetch_add(1, Ordering::SeqCst);
        let operation_id = command.operation_id();
        self.0
            .commands
            .lock()
            .expect("recording commands lock")
            .push(command);
        match *self.0.mode.lock().expect("sink mode lock") {
            SinkMode::Applied => Ok(DomainCommandReceipt::new(
                operation_id,
                AggregateVersion::FIRST,
            )),
            SinkMode::Duplicate => Ok(DomainCommandReceipt::duplicate(
                operation_id,
                AggregateVersion::FIRST,
            )),
            SinkMode::Rejected => Err(DomainCommandSubmissionError::Rejected),
            SinkMode::Unavailable => Err(DomainCommandSubmissionError::Unavailable),
        }
    }
}

#[tokio::test]
async fn nostr_event_ingest_accepts_and_idempotently_submits_frozen_events() {
    let event = event_fixture("legacy_message");
    let sink = RecordingSink::default();
    let ingress = NostrEventIngress::new(sink.clone(), NostrIngressDeployment::InProcess);
    let now = event["created_at"].as_u64().expect("created_at");

    let accepted = ingress
        .handle_frame(admission(event_author(&event)), &event_frame(&event), now)
        .await
        .expect("accepted EVENT");
    assert_eq!(accepted.status(), NostrEventIngestStatus::Accepted);
    assert_eq!(
        accepted.frame(),
        format!(r#"["OK","{}",true,""]"#, event["id"].as_str().expect("id"))
    );

    sink.set_mode(SinkMode::Duplicate);
    let duplicate = ingress
        .handle_frame(admission(event_author(&event)), &event_frame(&event), now)
        .await
        .expect("duplicate EVENT");
    assert_eq!(duplicate.status(), NostrEventIngestStatus::Duplicate);
    assert_eq!(
        duplicate.frame(),
        format!(
            r#"["OK","{}",true,"duplicate:"]"#,
            event["id"].as_str().expect("id")
        )
    );

    let commands = sink.0.commands.lock().expect("recording commands lock");
    assert_eq!(commands.len(), 2);
    assert_eq!(commands[0].operation_id(), commands[1].operation_id());
    assert_eq!(commands[0].tenant().community_id(), community());
    assert_eq!(
        commands[0].payload().signed_event().claimed_id.to_hex(),
        event["id"].as_str().expect("id")
    );
    assert_eq!(commands[0].payload().wire_event(), &event);
}

#[tokio::test]
async fn nostr_event_ingest_matches_frozen_tampered_event_rejection() {
    let event = event_fixture("malformed_tampered_content");
    let sink = RecordingSink::default();
    let ingress = NostrEventIngress::new(sink.clone(), NostrIngressDeployment::InProcess);
    let outcome = ingress
        .handle_frame(
            admission(event_author(&event)),
            &event_frame(&event),
            event["created_at"].as_u64().expect("created_at"),
        )
        .await
        .expect("negative OK");

    assert_eq!(outcome.status(), NostrEventIngestStatus::Invalid);
    assert_eq!(
        outcome.frame(),
        r#"["OK","4fe01b3f32599b1c541190751dc0f4dfa15361b602332231ae9c1fe91c80ec4c",false,"invalid: invalid event id: computed 0c3a3f53d89b5dbb338fc7ecf0a6af734c3ec54b1ce28e3c4961a71342aeb904, got 4fe01b3f32599b1c541190751dc0f4dfa15361b602332231ae9c1fe91c80ec4c"]"#
    );
    assert_eq!(sink.0.writes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn nostr_event_ingest_rejects_unauthorized_author_before_submission() {
    let event = event_fixture("legacy_message");
    let other_bytes =
        hex::decode("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9")
            .expect("other public key");
    let other_author =
        NostrPublicKey::from_bytes(other_bytes.try_into().expect("32-byte other public key"));
    let sink = RecordingSink::default();
    let ingress = NostrEventIngress::new(sink.clone(), NostrIngressDeployment::InProcess);
    let outcome = ingress
        .handle_frame(
            admission(other_author),
            &event_frame(&event),
            event["created_at"].as_u64().expect("created_at"),
        )
        .await
        .expect("unauthorized EVENT");

    assert_eq!(outcome.status(), NostrEventIngestStatus::Unauthorized);
    assert_eq!(
        outcome.frame(),
        format!(
            r#"["OK","{}",false,"invalid: event pubkey does not match authenticated identity"]"#,
            event["id"].as_str().expect("id")
        )
    );
    assert_eq!(sink.0.writes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn nostr_event_ingest_bounds_malformed_frames_and_sanitizes_sink_failures() {
    let event = event_fixture("legacy_message");
    let sink = RecordingSink::default();
    let ingress = NostrEventIngress::new(sink.clone(), NostrIngressDeployment::InProcess);
    let error = ingress
        .handle_frame(admission(event_author(&event)), "{}", 1)
        .await
        .expect_err("non-array frame");
    assert_eq!(error.kind(), NostrEventFrameErrorKind::InvalidFrame);

    let mut malformed = event.clone();
    malformed["sig"] = Value::String("bad".into());
    let malformed_outcome = ingress
        .handle_frame(
            admission(event_author(&event)),
            &event_frame(&malformed),
            event["created_at"].as_u64().expect("created_at"),
        )
        .await
        .expect("malformed event with usable id");
    assert_eq!(malformed_outcome.status(), NostrEventIngestStatus::Invalid);
    assert!(
        malformed_outcome
            .frame()
            .ends_with("false,\"invalid: malformed event\"]")
    );
    assert_eq!(sink.0.writes.load(Ordering::SeqCst), 0);

    sink.set_mode(SinkMode::Rejected);
    let rejected = ingress
        .handle_frame(
            admission(event_author(&event)),
            &event_frame(&event),
            event["created_at"].as_u64().expect("created_at"),
        )
        .await
        .expect("rejected command");
    assert_eq!(rejected.status(), NostrEventIngestStatus::Unauthorized);
    assert!(
        rejected
            .frame()
            .ends_with("false,\"restricted: event rejected\"]")
    );

    sink.set_mode(SinkMode::Unavailable);
    let unavailable = ingress
        .handle_frame(
            admission(event_author(&event)),
            &event_frame(&event),
            event["created_at"].as_u64().expect("created_at"),
        )
        .await
        .expect("unavailable command service");
    assert_eq!(unavailable.status(), NostrEventIngestStatus::Unavailable);
    assert!(
        unavailable
            .frame()
            .ends_with("false,\"error: internal server error\"]")
    );
}
