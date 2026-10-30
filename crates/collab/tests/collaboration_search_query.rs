use std::collections::BTreeMap;

use collab::{
    search::{
        indexer::{SearchDocumentType, SearchProjectionOperation},
        query::{
            CollaborationSearchQueries, CollaborationSearchResultClass,
            CollaborationSearchResultIdentity, CollaborationSearchResultReference,
        },
        repository::{
            CollaborationSearchQuery, CollaborationSearchRepository, SearchAccess, SearchMode,
            SearchProjectionFreshness, SearchRepositoryError,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateVersion, AuthenticatedPrincipal, AuthorizationScope, CommunityId, CommunityMembership,
    MembershipRole, MembershipStatus, PrincipalId, PrincipalScopes, ServiceAccountId,
    TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use uuid::Uuid;

const CHANNEL_SEARCH_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000100_collaboration_search_channels.up.sql"
));
const CHANNEL_SEARCH_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000100_collaboration_search_channels.down.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "search-query")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn principal(community_id: CommunityId, search_scope: bool) -> AuthenticatedPrincipal {
    let scopes = if search_scope {
        PrincipalScopes::new([AuthorizationScope::new("collaboration:search").expect("scope")])
            .expect("scopes")
    } else {
        PrincipalScopes::default()
    };
    AuthenticatedPrincipal::zed_account(
        PrincipalId::from_uuid(Uuid::from_u128(2)),
        community_id,
        ServiceAccountId::new(3),
        scopes,
    )
}

fn membership(
    community_id: CommunityId,
    principal: &AuthenticatedPrincipal,
) -> CommunityMembership {
    CommunityMembership {
        community_id,
        principal_id: principal.principal_id(),
        role: MembershipRole::Member,
        status: MembershipStatus::Active,
        version: AggregateVersion::FIRST,
    }
}

fn document_row(
    document_type: &str,
    source_record_id: &str,
    source_version: &str,
    rank: f32,
) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("record_type".into(), "canonical_document".to_owned().into()),
        ("source_system".into(), "zed".to_owned().into()),
        (
            "source_record_id".into(),
            source_record_id.to_owned().into(),
        ),
        ("source_version".into(), source_version.to_owned().into()),
        ("document_type".into(), document_type.to_owned().into()),
        (
            "observed_at_millis".into(),
            "1900000000000".to_owned().into(),
        ),
        ("rank".into(), rank.into()),
    ])
}

fn event_row(event_byte: u8, author_byte: u8, kind: i32, rank: f32) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("record_type".into(), "signed_event".to_owned().into()),
        ("source_system".into(), "nostr".to_owned().into()),
        (
            "source_record_id".into(),
            format!("{event_byte:02x}").repeat(32).into(),
        ),
        (
            "source_version".into(),
            format!("{event_byte:02x}").repeat(32).into(),
        ),
        ("event_id".into(), vec![event_byte; 32].into()),
        ("author_public_key".into(), vec![author_byte; 32].into()),
        ("event_kind".into(), kind.into()),
        (
            "observed_at_millis".into(),
            "1900000000000".to_owned().into(),
        ),
        ("rank".into(), rank.into()),
    ])
}

fn current_freshness_row() -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("checkpoint_count".into(), 1_i64.into()),
        ("all_clean".into(), true.into()),
        (
            "oldest_projected_at_millis".into(),
            1_900_000_000_000_i64.into(),
        ),
        ("affected_count".into(), 0_i64.into()),
    ])
}

#[tokio::test]
async fn collaboration_search_query_authorizes_before_rank_and_limit() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let principal = principal(community_id, false);
    let repository = CollaborationSearchRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("repository");
    let queries = CollaborationSearchQueries::new(repository);
    let query =
        CollaborationSearchQuery::new("private", SearchMode::FullText, 1, 1).expect("query");

    let result = queries
        .query(
            SearchAccess {
                tenant: &tenant,
                principal: &principal,
                current_membership_version: AggregateVersion::FIRST,
                community_membership: Some(membership(community_id, &principal)),
                now_millis: 1_900_000_000_000,
            },
            &query,
        )
        .await;

    assert!(matches!(
        result,
        Err(
            collab::search::query::CollaborationSearchQueryError::Repository(
                SearchRepositoryError::Unauthorized(_)
            )
        )
    ));
    assert!(
        queries
            .into_repository()
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}

#[tokio::test]
async fn collaboration_search_query_returns_typed_stable_identities_and_freshness() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let principal = principal(community_id, true);
    let rows = vec![
        document_row("community", "community:one", "1", 1.0),
        document_row("channel", "channel:one", "3", 0.9),
        document_row("profile", "profile:one", "7", 0.8),
        document_row("profile", "profile:one", "8", 0.7),
        document_row("project", "project:one", "2", 0.6),
        event_row(10, 20, 0, 0.5),
        event_row(11, 20, 0, 0.4),
        event_row(12, 30, 9, 0.3),
    ];
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .append_query_results([rows, vec![current_freshness_row()]])
        .into_connection();
    let repository = CollaborationSearchRepository::new(database).expect("repository");
    let queries = CollaborationSearchQueries::new(repository);
    let query = CollaborationSearchQuery::new("alpha", SearchMode::Prefix, 1, 20).expect("query");

    let result = queries
        .query(
            SearchAccess {
                tenant: &tenant,
                principal: &principal,
                current_membership_version: AggregateVersion::FIRST,
                community_membership: Some(membership(community_id, &principal)),
                now_millis: 1_900_000_000_000,
            },
            &query,
        )
        .await
        .expect("typed search");

    assert_eq!(
        result.hits.iter().map(|hit| hit.class).collect::<Vec<_>>(),
        vec![
            CollaborationSearchResultClass::Community,
            CollaborationSearchResultClass::Channel,
            CollaborationSearchResultClass::Member,
            CollaborationSearchResultClass::Member,
            CollaborationSearchResultClass::Project,
            CollaborationSearchResultClass::Member,
            CollaborationSearchResultClass::Member,
            CollaborationSearchResultClass::Message,
        ]
    );
    assert_eq!(result.hits[2].identity, result.hits[3].identity);
    assert_eq!(result.hits[5].identity, result.hits[6].identity);
    assert_ne!(result.hits[6].identity, result.hits[7].identity);
    assert!(matches!(
        result.hits[1].identity,
        CollaborationSearchResultIdentity::Canonical(_)
    ));
    assert!(matches!(
        result.hits[5].reference,
        CollaborationSearchResultReference::MemberProfile { .. }
    ));
    assert!(matches!(
        result.hits[7].reference,
        CollaborationSearchResultReference::Message { kind: 9, .. }
    ));
    assert_eq!(
        result.projection_freshness,
        SearchProjectionFreshness::Current {
            oldest_projected_at_millis: 1_900_000_000_000,
        }
    );

    let log = format!(
        "{:#?}",
        queries
            .into_repository()
            .into_connection()
            .into_transaction_log()
    );
    let policy = log
        .find("document.visibility_scope = 'community'")
        .expect("visibility policy");
    let rank = log.find("ORDER BY rank DESC").expect("rank ordering");
    let limit = log.find("LIMIT $4").expect("limit");
    assert!(policy < rank && rank < limit);
}

#[test]
fn collaboration_search_channel_projection_extension_is_reversible() {
    let payload = SearchProjectionOperation::upsert_community(
        SearchDocumentType::Channel,
        "general",
        "General discussion",
    )
    .expect("channel operation")
    .encode()
    .expect("channel payload");
    let payload: serde_json::Value = serde_json::from_slice(&payload).expect("projection json");
    assert_eq!(payload["document_type"], "channel");
    assert!(CHANNEL_SEARCH_UP.contains("'community', 'channel', 'project'"));
    assert!(
        CHANNEL_SEARCH_DOWN
            .starts_with("ALTER TABLE public.collaboration_search_documents NO FORCE")
    );
    assert!(CHANNEL_SEARCH_DOWN.contains("FORCE ROW LEVEL SECURITY;"));
    assert_eq!(CHANNEL_SEARCH_DOWN.matches("'channel'").count(), 1);
}
