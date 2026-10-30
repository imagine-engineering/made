use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use collab::{
    nostr::subscriptions::{
        MAX_ACTIVE_SUBSCRIPTIONS, MAX_FILTER_LIMIT, MAX_FILTERS_PER_REQUEST, MAX_NOSTR_FRAME_BYTES,
        MAX_SUBSCRIPTION_ID_BYTES, NostrStoredEvent, NostrSubscriptionError,
        NostrSubscriptionFailure, NostrSubscriptionFilter, NostrSubscriptionFrame,
        NostrSubscriptionQuery, NostrSubscriptionResources, NostrSubscriptionServiceError,
        NostrSubscriptionSession, SubscriptionId, SubscriptionResourceToken,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AuthenticatedPrincipal, CommunityId, NostrAuthenticationMethod, NostrPublicKey, PrincipalId,
    PrincipalScopes, TenantContext, TrustedTenantRoute,
};
use serde_json::json;
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "nostr-subscriptions")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn authenticated(community_id: CommunityId) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::nostr_identity(
        principal(9),
        community_id,
        NostrPublicKey::from_bytes([7; 32]),
        NostrAuthenticationMethod::Nip42,
        PrincipalScopes::default(),
    )
}

#[derive(Default)]
struct QueryState {
    counts: Mutex<BTreeMap<CommunityId, u64>>,
    fail_historical: AtomicBool,
}

#[derive(Clone, Default)]
struct Query(Arc<QueryState>);

#[async_trait]
impl NostrSubscriptionQuery for Query {
    async fn historical(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        _filters: &[NostrSubscriptionFilter],
    ) -> Result<Vec<NostrStoredEvent>, NostrSubscriptionServiceError> {
        if tenant.community_id() != principal.community_id() {
            return Err(NostrSubscriptionServiceError::Denied);
        }
        if self.0.fail_historical.load(Ordering::SeqCst) {
            return Err(NostrSubscriptionServiceError::Unavailable);
        }
        Ok(vec![
            NostrStoredEvent::new(json!({
                "id": format!("event-{}", tenant.community_id()),
                "content": "visible",
            }))
            .expect("stored event"),
        ])
    }

    async fn count(
        &self,
        tenant: &TenantContext,
        principal: &AuthenticatedPrincipal,
        _filters: &[NostrSubscriptionFilter],
    ) -> Result<u64, NostrSubscriptionServiceError> {
        if tenant.community_id() != principal.community_id() {
            return Err(NostrSubscriptionServiceError::Denied);
        }
        Ok(*self
            .0
            .counts
            .lock()
            .expect("query count lock")
            .get(&tenant.community_id())
            .unwrap_or(&0))
    }
}

#[derive(Default)]
struct ResourceState {
    next_token: AtomicU64,
    active: Mutex<BTreeSet<u64>>,
    released: Mutex<Vec<u64>>,
    fail_activate: AtomicBool,
}

#[derive(Clone, Default)]
struct Resources(Arc<ResourceState>);

impl Resources {
    fn active_count(&self) -> usize {
        self.0.active.lock().expect("active resource lock").len()
    }

    fn released_count(&self) -> usize {
        self.0
            .released
            .lock()
            .expect("released resource lock")
            .len()
    }
}

#[async_trait]
impl NostrSubscriptionResources for Resources {
    async fn activate(
        &self,
        tenant: &TenantContext,
        _connection_id: Uuid,
        _subscription_id: &SubscriptionId,
        _filters: &[NostrSubscriptionFilter],
    ) -> Result<SubscriptionResourceToken, NostrSubscriptionServiceError> {
        if self.0.fail_activate.load(Ordering::SeqCst) {
            return Err(NostrSubscriptionServiceError::Unavailable);
        }
        assert!(
            matches!(tenant.community_id(), value if value == community(1) || value == community(2))
        );
        let token = self.0.next_token.fetch_add(1, Ordering::SeqCst) + 1;
        self.0
            .active
            .lock()
            .expect("active resource lock")
            .insert(token);
        Ok(SubscriptionResourceToken::new(token))
    }

    async fn release(
        &self,
        token: SubscriptionResourceToken,
    ) -> Result<(), NostrSubscriptionServiceError> {
        self.0
            .active
            .lock()
            .expect("active resource lock")
            .remove(&token.get());
        self.0
            .released
            .lock()
            .expect("released resource lock")
            .push(token.get());
        Ok(())
    }
}

fn session(
    community_id: CommunityId,
    query: Query,
    resources: Resources,
) -> NostrSubscriptionSession<Query, Resources> {
    NostrSubscriptionSession::new(
        tenant(community_id),
        authenticated(community_id),
        Uuid::from_u128(50),
        query,
        resources,
    )
    .expect("session")
}

#[test]
fn nostr_subscriptions_bound_frames_ids_filters_values_and_limits() {
    assert_eq!(
        NostrSubscriptionFrame::parse(&"x".repeat(MAX_NOSTR_FRAME_BYTES + 1)),
        Err(NostrSubscriptionError::FrameTooLarge)
    );
    assert_eq!(
        NostrSubscriptionFrame::parse(
            &json!(["REQ", "x".repeat(MAX_SUBSCRIPTION_ID_BYTES + 1)]).to_string()
        ),
        Err(NostrSubscriptionError::InvalidSubscriptionId)
    );
    let filters = vec![json!({}); MAX_FILTERS_PER_REQUEST + 1];
    let mut request = vec![json!("REQ"), json!("sub")];
    request.extend(filters);
    assert_eq!(
        NostrSubscriptionFrame::parse(&serde_json::to_string(&request).expect("request")),
        Err(NostrSubscriptionError::TooManyFilters)
    );
    let parsed = NostrSubscriptionFrame::parse(
        &json!(["REQ", "sub", {"kinds": [1], "limit": 50_000}]).to_string(),
    )
    .expect("bounded filter");
    let NostrSubscriptionFrame::Req { filters, .. } = parsed else {
        panic!("expected REQ")
    };
    assert_eq!(filters[0].limit(), MAX_FILTER_LIMIT);
}

#[tokio::test]
async fn nostr_subscriptions_emit_event_eose_replace_and_closed_frames() {
    let query = Query::default();
    let resources = Resources::default();
    let mut session = session(community(1), query, resources.clone());

    let outcome = session
        .handle_frame(r#"["REQ","sub",{"kinds":[1]}]"#)
        .await
        .expect("REQ");
    assert_eq!(outcome.failure_reason(), None);
    assert_eq!(outcome.frames().len(), 2);
    let event_frame: serde_json::Value =
        serde_json::from_str(&outcome.frames()[0]).expect("EVENT frame");
    assert_eq!(event_frame[0], "EVENT");
    assert_eq!(event_frame[1], "sub");
    assert_eq!(event_frame[2]["content"], "visible");
    assert!(
        event_frame[2]["id"]
            .as_str()
            .is_some_and(|value| value.starts_with("event-"))
    );
    assert_eq!(outcome.frames()[1], r#"["EOSE","sub"]"#);
    assert_eq!(session.active_subscription_count(), 1);
    assert_eq!(resources.active_count(), 1);

    session
        .handle_frame(r#"["REQ","sub",{"kinds":[2]}]"#)
        .await
        .expect("replacement REQ");
    assert_eq!(session.active_subscription_count(), 1);
    assert_eq!(resources.active_count(), 1);
    assert_eq!(resources.released_count(), 1);

    let closed = session
        .handle_frame(r#"["CLOSE","sub"]"#)
        .await
        .expect("CLOSE");
    assert_eq!(closed.frames(), &[r#"["CLOSED","sub",""]"#]);
    assert_eq!(session.active_subscription_count(), 0);
    assert_eq!(resources.active_count(), 0);
}

#[tokio::test]
async fn nostr_subscriptions_count_is_tenant_private_and_never_registers() {
    let query = Query::default();
    query
        .0
        .counts
        .lock()
        .expect("query count lock")
        .extend([(community(1), 2), (community(2), 7)]);
    let resources = Resources::default();
    let mut first = session(community(1), query.clone(), resources.clone());
    let mut second = session(community(2), query, resources.clone());

    let first_count = first
        .handle_frame(r#"["COUNT","count",{"kinds":[1]}]"#)
        .await
        .expect("first count");
    let second_count = second
        .handle_frame(r#"["COUNT","count",{"kinds":[1]}]"#)
        .await
        .expect("second count");
    assert_eq!(first_count.frames(), &[r#"["COUNT","count",{"count":2}]"#]);
    assert_eq!(second_count.frames(), &[r#"["COUNT","count",{"count":7}]"#]);
    assert_eq!(resources.active_count(), 0);
}

#[tokio::test]
async fn nostr_subscriptions_cancel_releases_every_resource() {
    let resources = Resources::default();
    let mut session = session(community(1), Query::default(), resources.clone());
    session
        .handle_frame(r#"["REQ","one"]"#)
        .await
        .expect("first REQ");
    session
        .handle_frame(r#"["REQ","two"]"#)
        .await
        .expect("second REQ");

    session.cancel().await.expect("cancel cleanup");
    assert_eq!(session.active_subscription_count(), 0);
    assert_eq!(resources.active_count(), 0);
    assert_eq!(resources.released_count(), 2);
}

#[tokio::test]
async fn nostr_subscriptions_query_failure_closes_and_cleans_the_subscription() {
    let query = Query::default();
    query.0.fail_historical.store(true, Ordering::SeqCst);
    let resources = Resources::default();
    let mut session = session(community(1), query, resources.clone());

    let outcome = session
        .handle_frame(r#"["REQ","failed",{"kinds":[1]}]"#)
        .await
        .expect("handled failure");
    assert_eq!(
        outcome.failure_reason(),
        Some(NostrSubscriptionFailure::QueryUnavailable)
    );
    assert_eq!(
        outcome.frames(),
        &[r#"["CLOSED","failed","error: query unavailable"]"#]
    );
    assert_eq!(session.active_subscription_count(), 0);
    assert_eq!(resources.active_count(), 0);
}

#[tokio::test]
async fn nostr_subscriptions_enforce_the_per_connection_active_limit() {
    let resources = Resources::default();
    let mut session = session(community(1), Query::default(), resources.clone());
    for index in 0..MAX_ACTIVE_SUBSCRIPTIONS {
        session
            .handle_frame(&json!(["REQ", format!("sub-{index}")]).to_string())
            .await
            .expect("bounded subscription");
    }
    let outcome = session
        .handle_frame(r#"["REQ","overflow"]"#)
        .await
        .expect("limit response");
    assert_eq!(
        outcome.failure_reason(),
        Some(NostrSubscriptionFailure::TooManySubscriptions)
    );
    assert_eq!(
        session.active_subscription_count(),
        MAX_ACTIVE_SUBSCRIPTIONS
    );
    assert_eq!(resources.active_count(), MAX_ACTIVE_SUBSCRIPTIONS);
    session.cancel().await.expect("limit cleanup");
    assert_eq!(resources.active_count(), 0);
}
