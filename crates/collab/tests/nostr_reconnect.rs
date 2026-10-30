use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use collab::{
    collaboration_command::{
        DomainCommand, DomainCommandReceipt, DomainCommandSink, DomainCommandSubmissionError,
    },
    nostr::{
        auth::{
            NostrAuthChallenge, NostrAuthReplayStore, NostrAuthenticationInfrastructureError,
            NostrConnectionAuthenticator, NostrPrincipalResolutionError, NostrPrincipalResolver,
            ReplayClaim, VerifiedNip42Identity,
        },
        event_ingest::{NostrEventCommand, NostrEventIngestStatus, NostrEventIngress},
        ingress::NostrIngressDeployment,
        subscriptions::{
            NostrStoredEvent, NostrSubscriptionFilter, NostrSubscriptionQuery,
            NostrSubscriptionResources, NostrSubscriptionServiceError, NostrSubscriptionSession,
            SubscriptionId, SubscriptionResourceToken,
        },
    },
    tenant_admission::{AuthorizedRpcRequest, bind_rpc_tenant},
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, NostrAuthenticationMethod,
    NostrPublicKey, PrincipalId, PrincipalScopes, TenantContext, TrustedTenantRoute,
};
use nostr_compat::{
    CanonicalEvent, EventId, EventSignature, PublicKey, SignedEvent, generated_kinds::KIND_AUTH,
};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};
use serde_json::{Value, json};
use uuid::Uuid;

const RELAY_URL: &str = "wss://relay.example.test";
const NOW: u64 = 1_900_000_000;
const EVENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
));

fn community() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn principal() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(2))
}

fn tenant() -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community(), "nostr-reconnect")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn scopes() -> PrincipalScopes {
    PrincipalScopes::new([
        AuthorizationScope::new("events:read").expect("read scope"),
        AuthorizationScope::new("events:write").expect("write scope"),
    ])
    .expect("principal scopes")
}

fn authenticated(public_key: NostrPublicKey) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::nostr_identity(
        principal(),
        community(),
        public_key,
        NostrAuthenticationMethod::Nip42,
        scopes(),
    )
}

fn challenge(byte: u8) -> NostrAuthChallenge {
    NostrAuthChallenge::parse(format!("{byte:02x}").repeat(32)).expect("challenge")
}

fn signed_event(
    secret_byte: u8,
    kind: u16,
    created_at: u64,
    tags: Vec<Vec<String>>,
    content: String,
) -> SignedEvent {
    let secret = SecretKey::from_slice(&[secret_byte; 32]).expect("fixture secret");
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        created_at,
        kind,
        tags,
        content,
    );
    let claimed_id = event.event_id().expect("event id");
    let signature =
        secp.sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
    SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

fn signed_auth_event(
    secret_byte: u8,
    challenge: &NostrAuthChallenge,
    created_at: u64,
) -> SignedEvent {
    signed_event(
        secret_byte,
        KIND_AUTH as u16,
        created_at,
        vec![
            vec!["relay".to_owned(), RELAY_URL.to_owned()],
            vec!["challenge".to_owned(), challenge.as_str().to_owned()],
        ],
        String::new(),
    )
}

fn fixture_event() -> Value {
    let fixture: Value = serde_json::from_str(EVENTS).expect("valid event corpus");
    fixture["events"]["legacy_message"].clone()
}

fn event_author(event: &Value) -> NostrPublicKey {
    let bytes =
        hex::decode(event["pubkey"].as_str().expect("event public key")).expect("hex public key");
    NostrPublicKey::from_bytes(bytes.try_into().expect("32-byte public key"))
}

fn admission(public_key: NostrPublicKey) -> AuthorizedRpcRequest {
    let tenant = tenant();
    let authenticated_principal = authenticated(public_key);
    let required_scope = AuthorizationScope::new("events:write").expect("scope");
    AuthorizedRpcRequest::authorize(&AuthorizationRequest {
        tenant: &tenant,
        principal: &authenticated_principal,
        required_scope: &required_scope,
        action: AuthorizationAction::Write,
        resource: AuthorizationResource {
            community_id: community(),
            kind: AuthorizationResourceKind::Community,
            resource_id: AggregateId::from_uuid(Uuid::from_u128(3)),
            owner_principal_id: None,
            channel_id: None,
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(CommunityMembership {
            community_id: community(),
            principal_id: principal(),
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

#[derive(Clone, Default)]
struct ReplayStore(Arc<Mutex<BTreeSet<(CommunityId, EventId)>>>);

#[async_trait]
impl NostrAuthReplayStore for ReplayStore {
    async fn claim(
        &self,
        community_id: CommunityId,
        event_id: EventId,
        _expires_at_seconds: u64,
    ) -> Result<ReplayClaim, NostrAuthenticationInfrastructureError> {
        let inserted = self
            .0
            .lock()
            .expect("replay store lock")
            .insert((community_id, event_id));
        Ok(if inserted {
            ReplayClaim::Claimed
        } else {
            ReplayClaim::Replay
        })
    }
}

#[derive(Clone, Copy)]
struct PrincipalResolver;

#[async_trait]
impl NostrPrincipalResolver for PrincipalResolver {
    async fn resolve(
        &self,
        tenant: &TenantContext,
        identity: &VerifiedNip42Identity,
    ) -> Result<AuthenticatedPrincipal, NostrPrincipalResolutionError> {
        Ok(AuthenticatedPrincipal::nostr_identity(
            principal(),
            tenant.community_id(),
            NostrPublicKey::from_bytes(*identity.public_key().as_bytes()),
            NostrAuthenticationMethod::Nip42,
            scopes(),
        ))
    }
}

fn authenticator(
    challenge: NostrAuthChallenge,
    issued_at: u64,
    replay_store: ReplayStore,
) -> NostrConnectionAuthenticator<ReplayStore, PrincipalResolver> {
    NostrConnectionAuthenticator::new(
        tenant(),
        RELAY_URL,
        challenge,
        issued_at,
        replay_store,
        PrincipalResolver,
    )
    .expect("authenticator")
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueryWindow {
    since: Option<u64>,
    until: Option<u64>,
}

#[derive(Default)]
struct QueryState {
    calls: Mutex<Vec<QueryWindow>>,
    fail_older_window: AtomicBool,
}

#[derive(Clone, Default)]
struct Query(Arc<QueryState>);

impl Query {
    fn calls(&self) -> Vec<QueryWindow> {
        self.0.calls.lock().expect("query calls lock").clone()
    }
}

#[async_trait]
impl NostrSubscriptionQuery for Query {
    async fn historical(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        filters: &[NostrSubscriptionFilter],
    ) -> Result<Vec<NostrStoredEvent>, NostrSubscriptionServiceError> {
        if tenant.community_id() != principal.community_id() || filters.len() != 1 {
            return Err(NostrSubscriptionServiceError::Denied);
        }
        let filter = filters[0].event_filter();
        self.0
            .calls
            .lock()
            .expect("query calls lock")
            .push(QueryWindow {
                since: filter.since,
                until: filter.until,
            });
        if filter.until.is_some() && self.0.fail_older_window.load(Ordering::SeqCst) {
            return Err(NostrSubscriptionServiceError::Unavailable);
        }
        let content = if filter.since.is_some() {
            "authoritative-head"
        } else {
            "older-window"
        };
        Ok(vec![
            NostrStoredEvent::new(json!({
                "id": format!("{content}-event"),
                "content": content,
            }))
            .expect("stored event"),
        ])
    }

    async fn count(
        &self,
        _tenant: &TenantContext,
        _principal: &AuthenticatedPrincipal,
        _filters: &[NostrSubscriptionFilter],
    ) -> Result<u64, NostrSubscriptionServiceError> {
        Ok(0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Activation {
    connection_id: Uuid,
    subscription_id: String,
}

#[derive(Default)]
struct ResourceState {
    next_token: AtomicU64,
    active: Mutex<BTreeSet<u64>>,
    activations: Mutex<Vec<Activation>>,
}

#[derive(Clone, Default)]
struct Resources(Arc<ResourceState>);

impl Resources {
    fn active_count(&self) -> usize {
        self.0.active.lock().expect("active resources lock").len()
    }

    fn activations(&self) -> Vec<Activation> {
        self.0.activations.lock().expect("activations lock").clone()
    }
}

#[async_trait]
impl NostrSubscriptionResources for Resources {
    async fn activate(
        &self,
        tenant: &TenantContext,
        connection_id: Uuid,
        subscription_id: &SubscriptionId,
        _filters: &[NostrSubscriptionFilter],
    ) -> Result<SubscriptionResourceToken, NostrSubscriptionServiceError> {
        if tenant.community_id() != community() {
            return Err(NostrSubscriptionServiceError::Denied);
        }
        let token = self.0.next_token.fetch_add(1, Ordering::SeqCst) + 1;
        self.0
            .active
            .lock()
            .expect("active resources lock")
            .insert(token);
        self.0
            .activations
            .lock()
            .expect("activations lock")
            .push(Activation {
                connection_id,
                subscription_id: subscription_id.as_str().to_owned(),
            });
        Ok(SubscriptionResourceToken::new(token))
    }

    async fn release(
        &self,
        token: SubscriptionResourceToken,
    ) -> Result<(), NostrSubscriptionServiceError> {
        self.0
            .active
            .lock()
            .expect("active resources lock")
            .remove(&token.get());
        Ok(())
    }
}

fn session(
    principal: AuthenticatedPrincipal,
    connection_id: Uuid,
    query: Query,
    resources: Resources,
) -> NostrSubscriptionSession<Query, Resources> {
    NostrSubscriptionSession::new(tenant(), principal, connection_id, query, resources)
        .expect("subscription session")
}

#[tokio::test]
async fn nostr_reconnect_reauthenticates_refetches_windows_and_rearms_subscriptions() {
    let replay_store = ReplayStore::default();
    let initial_challenge = challenge(0x11);
    let reconnect_challenge = challenge(0x22);
    let initial_auth = signed_auth_event(1, &initial_challenge, NOW);
    let reconnect_auth = signed_auth_event(1, &reconnect_challenge, NOW + 1);

    let initial_principal = authenticator(initial_challenge, NOW, replay_store.clone())
        .authenticate(&initial_auth, NOW)
        .await
        .expect("initial authentication")
        .clone();
    let query = Query::default();
    let resources = Resources::default();
    let initial_connection = Uuid::from_u128(10);
    let mut initial_session = session(
        initial_principal,
        initial_connection,
        query.clone(),
        resources.clone(),
    );
    initial_session
        .handle_frame(r#"["REQ","timeline",{"since":200,"limit":100}]"#)
        .await
        .expect("initial head subscription");
    assert_eq!(resources.active_count(), 1);
    initial_session.cancel().await.expect("disconnect cleanup");
    assert_eq!(resources.active_count(), 0);

    let reconnect_principal = authenticator(reconnect_challenge, NOW + 1, replay_store)
        .authenticate(&reconnect_auth, NOW + 1)
        .await
        .expect("fresh reconnect authentication")
        .clone();
    let reconnect_connection = Uuid::from_u128(11);
    let mut reconnect_session = session(
        reconnect_principal,
        reconnect_connection,
        query.clone(),
        resources.clone(),
    );
    let head = reconnect_session
        .handle_frame(r#"["REQ","timeline",{"since":200,"limit":100}]"#)
        .await
        .expect("authoritative head refetch");
    let older = reconnect_session
        .handle_frame(r#"["REQ","timeline-window",{"until":199,"limit":100}]"#)
        .await
        .expect("authoritative window refetch");

    assert_eq!(head.frames().len(), 2);
    assert!(head.frames()[0].contains("authoritative-head"));
    assert_eq!(head.frames()[1], r#"["EOSE","timeline"]"#);
    assert!(older.frames()[0].contains("older-window"));
    assert_eq!(older.frames()[1], r#"["EOSE","timeline-window"]"#);
    assert_eq!(
        query.calls(),
        vec![
            QueryWindow {
                since: Some(200),
                until: None,
            },
            QueryWindow {
                since: Some(200),
                until: None,
            },
            QueryWindow {
                since: None,
                until: Some(199),
            },
        ]
    );
    assert_eq!(
        resources.activations(),
        vec![
            Activation {
                connection_id: initial_connection,
                subscription_id: "timeline".to_owned(),
            },
            Activation {
                connection_id: reconnect_connection,
                subscription_id: "timeline".to_owned(),
            },
            Activation {
                connection_id: reconnect_connection,
                subscription_id: "timeline-window".to_owned(),
            },
        ]
    );
    assert_eq!(resources.active_count(), 2);
    reconnect_session.cancel().await.expect("reconnect cleanup");
    assert_eq!(resources.active_count(), 0);
}

#[tokio::test]
async fn nostr_reconnect_exposes_partial_window_freshness_without_dropping_the_head() {
    let query = Query::default();
    query.0.fail_older_window.store(true, Ordering::SeqCst);
    let resources = Resources::default();
    let public_key = NostrPublicKey::from_bytes([7; 32]);
    let mut reconnect_session = session(
        authenticated(public_key),
        Uuid::from_u128(12),
        query,
        resources.clone(),
    );

    let head = reconnect_session
        .handle_frame(r#"["REQ","head",{"since":200}]"#)
        .await
        .expect("head remains available");
    let older = reconnect_session
        .handle_frame(r#"["REQ","older",{"until":199}]"#)
        .await
        .expect("window failure is protocol-visible");

    assert!(head.frames()[0].contains("authoritative-head"));
    assert_eq!(head.frames()[1], r#"["EOSE","head"]"#);
    assert_eq!(
        older.frames(),
        &[r#"["CLOSED","older","error: query unavailable"]"#]
    );
    assert_eq!(resources.active_count(), 1);
    reconnect_session
        .cancel()
        .await
        .expect("partial reconnect cleanup");
    assert_eq!(resources.active_count(), 0);
}

#[derive(Default)]
struct DeduplicatingSinkState {
    operations: Mutex<BTreeMap<collaboration_domain::OperationId, AggregateVersion>>,
}

#[derive(Clone, Default)]
struct DeduplicatingSink(Arc<DeduplicatingSinkState>);

#[async_trait]
impl DomainCommandSink<NostrEventCommand> for DeduplicatingSink {
    async fn submit(
        &self,
        command: DomainCommand<NostrEventCommand>,
    ) -> Result<DomainCommandReceipt, DomainCommandSubmissionError> {
        let mut operations = self
            .0
            .operations
            .lock()
            .expect("deduplication operations lock");
        if let Some(version) = operations.get(&command.operation_id()).copied() {
            return Ok(DomainCommandReceipt::duplicate(
                command.operation_id(),
                version,
            ));
        }
        operations.insert(command.operation_id(), AggregateVersion::FIRST);
        Ok(DomainCommandReceipt::new(
            command.operation_id(),
            AggregateVersion::FIRST,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LocalItemState {
    Optimistic,
    Authoritative,
}

#[tokio::test]
async fn nostr_reconnect_reconciles_an_optimistic_event_without_a_duplicate_echo() {
    let event = fixture_event();
    let event_id = event["id"].as_str().expect("event id").to_owned();
    let event_frame = json!(["EVENT", event]).to_string();
    for deployment in [
        NostrIngressDeployment::InProcess,
        NostrIngressDeployment::TemporarySidecar,
    ] {
        let sink = DeduplicatingSink::default();
        let ingress = NostrEventIngress::new(sink, deployment);
        let mut local_items = BTreeMap::from([(event_id.clone(), LocalItemState::Optimistic)]);

        let accepted = ingress
            .handle_frame(
                admission(event_author(&event)),
                &event_frame,
                event["created_at"].as_u64().expect("created_at"),
            )
            .await
            .expect("initial optimistic publish");
        assert_eq!(accepted.status(), NostrEventIngestStatus::Accepted);
        local_items.insert(event_id.clone(), LocalItemState::Authoritative);

        let duplicate = ingress
            .handle_frame(
                admission(event_author(&event)),
                &event_frame,
                event["created_at"].as_u64().expect("created_at"),
            )
            .await
            .expect("replayed publish after uncertain disconnect");
        assert_eq!(duplicate.status(), NostrEventIngestStatus::Duplicate);
        assert!(duplicate.frame().ends_with("true,\"duplicate:\"]"));

        let historical_echo = NostrStoredEvent::new(event.clone()).expect("historical event");
        let echoed_id = historical_echo.wire_event()["id"]
            .as_str()
            .expect("historical event id")
            .to_owned();
        local_items.insert(echoed_id, LocalItemState::Authoritative);

        assert_eq!(local_items.len(), 1);
        assert_eq!(
            local_items.get(&event_id),
            Some(&LocalItemState::Authoritative)
        );
    }
}
