use std::collections::BTreeMap;

use collab::messages::window_repository::{
    ChannelWindowPage, ChannelWindowQuery, MessageProjectionLifecycle, MessageWindowRepository,
    MessageWindowRow, StableChannelWindow, ThreadWindowQuery, WindowAccess, WindowError,
    WindowSnapshot,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    ChannelMembership, CommunityId, CommunityMembership, MembershipRole, MembershipStatus,
    NostrEventId, PrincipalId, PrincipalScopes, ServiceAccountId, TenantContext,
    TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(1))
}

fn channel_id() -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(2))
}

fn principal_id() -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(3))
}

fn message_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn event_id(value: u8) -> NostrEventId {
    NostrEventId::from_bytes([value; 32])
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id(), "message-window-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn scope() -> AuthorizationScope {
    AuthorizationScope::new("collaboration:messages:read").expect("scope")
}

fn principal(scope: &AuthorizationScope) -> AuthenticatedPrincipal {
    AuthenticatedPrincipal::zed_account(
        principal_id(),
        community_id(),
        ServiceAccountId::new(4),
        PrincipalScopes::new([scope.clone()]).expect("scopes"),
    )
}

fn authorization<'a>(
    tenant: &'a TenantContext,
    principal: &'a AuthenticatedPrincipal,
    scope: &'a AuthorizationScope,
    include_channel_membership: bool,
) -> AuthorizationRequest<'a> {
    AuthorizationRequest {
        tenant,
        principal,
        required_scope: scope,
        action: AuthorizationAction::Read,
        resource: AuthorizationResource {
            community_id: community_id(),
            kind: AuthorizationResourceKind::Channel,
            resource_id: channel_id(),
            owner_principal_id: None,
            channel_id: Some(channel_id()),
        },
        current_membership_version: AggregateVersion::FIRST,
        community_membership: Some(CommunityMembership {
            community_id: community_id(),
            principal_id: principal_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        current_channel_membership_version: include_channel_membership
            .then_some(AggregateVersion::FIRST),
        channel_membership: include_channel_membership.then_some(ChannelMembership {
            community_id: community_id(),
            channel_id: channel_id(),
            principal_id: principal_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }),
        delegation: None,
        now_millis: 1_900_000_000_000,
    }
}

fn snapshot_row(micros: i64) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([("snapshot_micros".to_owned(), micros.into())])
}

fn message_row(value: u8, created_at: u64) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("community_id".to_owned(), community_id().as_uuid().into()),
        (
            "message_id".to_owned(),
            message_id(u128::from(value) + 10).as_uuid().into(),
        ),
        ("channel_id".to_owned(), channel_id().as_uuid().into()),
        ("source_event_id".to_owned(), vec![value; 32].into()),
        ("current_event_id".to_owned(), vec![value; 32].into()),
        (
            "author_principal_id".to_owned(),
            principal_id().as_uuid().into(),
        ),
        (
            "message_created_at_text".to_owned(),
            created_at.to_string().into(),
        ),
        ("lifecycle_state".to_owned(), "active".to_owned().into()),
        ("message_version_text".to_owned(), "1".to_owned().into()),
        (
            "projected_at_micros".to_owned(),
            (900_i64 + i64::from(value)).into(),
        ),
    ])
}

fn thread_row(value: u8, parent: u8, created_at: u64, depth: i32) -> BTreeMap<String, SeaValue> {
    let mut row = message_row(value, created_at);
    row.insert("parent_event_id".to_owned(), vec![parent; 32].into());
    row.insert("depth".to_owned(), depth.into());
    row
}

fn window_row(value: u8, created_at: u64) -> MessageWindowRow {
    MessageWindowRow {
        community_id: community_id(),
        message_id: message_id(u128::from(value) + 10),
        channel_id: channel_id(),
        source_event_id: event_id(value),
        current_event_id: event_id(value),
        author_principal_id: principal_id(),
        message_created_at: created_at,
        lifecycle: MessageProjectionLifecycle::Active,
        message_version: 1,
        projected_at_micros: 900 + u64::from(value),
    }
}

fn exec_result() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

#[tokio::test]
async fn channel_window_dense_second_cursor_returns_each_authorized_row_once() {
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([exec_result(), exec_result()])
        .append_query_results([
            vec![snapshot_row(1_000)],
            vec![
                message_row(1, 100),
                message_row(2, 100),
                message_row(3, 100),
            ],
            vec![message_row(3, 100)],
        ])
        .into_connection();
    let repository = MessageWindowRepository::new(database).expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = authorization(&tenant, &principal, &scope, true);

    let first = repository
        .channel_page(
            WindowAccess {
                authorization: &authorization,
            },
            &ChannelWindowQuery::head(channel_id(), 2).expect("head query"),
        )
        .await
        .expect("first page");
    assert!(first.has_more);
    assert_eq!(first.rows.len(), 2);
    let cursor = first.next_cursor.expect("continuation cursor");
    assert_eq!(cursor.source_event_id, event_id(2));

    let second = repository
        .channel_page(
            WindowAccess {
                authorization: &authorization,
            },
            &ChannelWindowQuery::continuation(channel_id(), 2, cursor, first.snapshot)
                .expect("continuation query"),
        )
        .await
        .expect("second page");
    assert!(!second.has_more);
    let ids = first
        .rows
        .iter()
        .chain(&second.rows)
        .map(|row| row.source_event_id)
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![event_id(1), event_id(2), event_id(3)]);

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("message.projected_at <= to_timestamp"));
    assert!(log.contains("row.source_event_id >"));
    assert!(log.contains("row.broadcast"));
}

#[tokio::test]
async fn thread_window_dense_second_cursor_and_depth_are_exact() {
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([exec_result(), exec_result()])
        .append_query_results([
            vec![snapshot_row(2_000)],
            vec![
                thread_row(2, 1, 200, 1),
                thread_row(3, 2, 200, 2),
                thread_row(4, 3, 200, 3),
            ],
            vec![thread_row(4, 3, 200, 3)],
        ])
        .into_connection();
    let repository = MessageWindowRepository::new(database).expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = authorization(&tenant, &principal, &scope, true);
    let first_query =
        ThreadWindowQuery::head(channel_id(), event_id(1), 2, Some(3)).expect("thread head query");
    let first = repository
        .thread_page(
            WindowAccess {
                authorization: &authorization,
            },
            &first_query,
        )
        .await
        .expect("first thread page");
    let cursor = first.next_cursor.expect("thread cursor");
    let second_query = ThreadWindowQuery::continuation(
        channel_id(),
        event_id(1),
        2,
        Some(3),
        cursor,
        first.snapshot,
    )
    .expect("thread continuation query");
    let second = repository
        .thread_page(
            WindowAccess {
                authorization: &authorization,
            },
            &second_query,
        )
        .await
        .expect("second thread page");
    let replies = first
        .replies
        .iter()
        .chain(&second.replies)
        .map(|row| (row.message.source_event_id, row.depth))
        .collect::<Vec<_>>();
    assert_eq!(
        replies,
        vec![(event_id(2), 1), (event_id(3), 2), (event_id(4), 3)]
    );
    assert!(!second.has_more);

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    assert!(log.contains("WITH RECURSIVE"));
    assert!(log.contains("NOT child.source_event_id = ANY(thread.path)"));
    assert!(log.contains("row.message_created_at >"));
}

#[test]
fn concurrent_live_rows_overlay_history_and_reconcile_on_head_refetch() {
    let mut window = StableChannelWindow::new(channel_id()).expect("window");
    window
        .replace_head(ChannelWindowPage {
            rows: vec![window_row(3, 100), window_row(4, 99)],
            has_more: false,
            next_cursor: None,
            request_cursor: None,
            snapshot: WindowSnapshot::from_micros(1_000),
        })
        .expect("initial head");
    assert!(
        window
            .push_live(window_row(1, 102))
            .expect("first live row")
    );
    assert!(
        window
            .push_live(window_row(2, 101))
            .expect("second live row")
    );
    assert!(!window.push_live(window_row(2, 101)).expect("live retry"));
    assert_eq!(
        window
            .ordered_rows()
            .iter()
            .map(|row| row.source_event_id)
            .collect::<Vec<_>>(),
        vec![event_id(1), event_id(2), event_id(3), event_id(4)]
    );

    window
        .replace_head(ChannelWindowPage {
            rows: vec![window_row(1, 102), window_row(2, 101)],
            has_more: true,
            next_cursor: Some(window_row(2, 101).cursor()),
            request_cursor: None,
            snapshot: WindowSnapshot::from_micros(2_000),
        })
        .expect("reconnected head");
    window
        .append_page(ChannelWindowPage {
            rows: vec![window_row(3, 100), window_row(4, 99)],
            has_more: false,
            next_cursor: None,
            request_cursor: Some(window_row(2, 101).cursor()),
            snapshot: WindowSnapshot::from_micros(2_000),
        })
        .expect("reconnected continuation");
    assert_eq!(
        window
            .ordered_rows()
            .iter()
            .map(|row| row.source_event_id)
            .collect::<Vec<_>>(),
        vec![event_id(1), event_id(2), event_id(3), event_id(4)]
    );
}

#[tokio::test]
async fn authorization_failure_happens_before_window_database_work() {
    let repository = MessageWindowRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("repository");
    let tenant = tenant();
    let scope = scope();
    let principal = principal(&scope);
    let authorization = authorization(&tenant, &principal, &scope, false);
    let result = repository
        .channel_page(
            WindowAccess {
                authorization: &authorization,
            },
            &ChannelWindowQuery::head(channel_id(), 20).expect("query"),
        )
        .await;

    assert!(matches!(result, Err(WindowError::Unauthorized(_))));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}
