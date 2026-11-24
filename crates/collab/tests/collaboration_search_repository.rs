use std::collections::BTreeMap;

use collab::{
    search::repository::{
        CollaborationSearchQuery, CollaborationSearchRepository, SearchAccess, SearchMode,
        SearchProjectionFreshness, SearchRecordReference, SearchRepositoryError,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateVersion, AuthenticatedPrincipal, AuthorizationScope, CommunityId, CommunityMembership,
    MembershipRole, MembershipStatus, PrincipalId, PrincipalScopes, ServiceAccountId,
    TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use sqlx::PgPool;
use uuid::Uuid;

const EVENTS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const PROJECTIONS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000300_collaboration_projections.up.sql"
));
const SEARCH_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000500_collaboration_search.up.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "search-repository")
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

fn event_row(rank: f32) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("record_type".into(), "signed_event".to_owned().into()),
        ("source_system".into(), "nostr".to_owned().into()),
        ("source_record_id".into(), "01".repeat(32).into()),
        ("source_version".into(), "01".repeat(32).into()),
        ("event_id".into(), vec![1_u8; 32].into()),
        ("author_public_key".into(), vec![2_u8; 32].into()),
        ("event_kind".into(), 9_i32.into()),
        (
            "observed_at_millis".into(),
            "1900000000000".to_owned().into(),
        ),
        ("rank".into(), rank.into()),
    ])
}

fn document_row(rank: f32) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("record_type".into(), "canonical_document".to_owned().into()),
        ("source_system".into(), "zed".to_owned().into()),
        ("source_record_id".into(), "project:7".to_owned().into()),
        ("source_version".into(), "4".to_owned().into()),
        ("document_type".into(), "project".to_owned().into()),
        (
            "observed_at_millis".into(),
            "1900000001000".to_owned().into(),
        ),
        ("rank".into(), rank.into()),
    ])
}

fn lagging_freshness_row() -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("checkpoint_count".into(), 2_i64.into()),
        ("all_clean".into(), false.into()),
        (
            "oldest_projected_at_millis".into(),
            1_900_000_000_000_i64.into(),
        ),
        ("affected_count".into(), 1_i64.into()),
    ])
}

#[tokio::test]
async fn collaboration_search_authorizes_before_database_or_limit_work() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let principal = principal(community_id, false);
    let repository = CollaborationSearchRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("repository");
    let query =
        CollaborationSearchQuery::new("private", SearchMode::FullText, 1, 1).expect("query");

    let result = repository
        .search(
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
        Err(SearchRepositoryError::Unauthorized(_))
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}

#[tokio::test]
async fn collaboration_search_orders_authorized_references_and_exposes_lag() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let principal = principal(community_id, true);
    let database = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .append_query_results([
            vec![event_row(0.9), document_row(0.4)],
            vec![lagging_freshness_row()],
        ])
        .into_connection();
    let repository = CollaborationSearchRepository::new(database).expect("repository");
    let query =
        CollaborationSearchQuery::new("  alpha\0beta  ", SearchMode::Prefix, 1, 2).expect("query");

    let result = repository
        .search(
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
        .expect("authorized search");

    assert_eq!(query.text(), "alpha beta");
    assert_eq!(result.hits.len(), 2);
    assert!(result.hits[0].rank > result.hits[1].rank);
    assert!(matches!(
        result.hits[0].record,
        SearchRecordReference::SignedEvent { kind: 9, .. }
    ));
    assert!(matches!(
        &result.hits[1].record,
        SearchRecordReference::CanonicalDocument { document_type, .. }
            if document_type == "project"
    ));
    assert_eq!(
        result.projection_freshness,
        SearchProjectionFreshness::Lagging {
            oldest_projected_at_millis: 1_900_000_000_000,
            affected_checkpoints: 1,
        }
    );

    let log = format!("{:#?}", repository.into_connection().into_transaction_log());
    let visibility = log
        .find("document.visibility_scope = 'community'")
        .expect("visibility predicate");
    let order = log.find("ORDER BY rank DESC").expect("rank ordering");
    let limit = log.find("LIMIT $4").expect("bound limit");
    assert!(visibility < order && order < limit);
    assert!(log.contains("event.search_tsv IS NOT NULL"));
    assert!(log.contains("document.search_tsv IS NOT NULL"));
    assert!(log.contains("projection_name = 'collaboration_search'"));
}

#[test]
fn collaboration_search_query_is_bounded_and_rejects_empty_input() {
    assert!(CollaborationSearchQuery::new(" \0 ", SearchMode::FullText, 1, 10).is_err());
    let query = CollaborationSearchQuery::new(
        format!("  {}  ", "x".repeat(5_000)),
        SearchMode::FullText,
        u32::MAX,
        u32::MAX,
    )
    .expect("bounded query");
    assert_eq!(query.text().chars().count(), 4_096);
    assert_eq!(query.page(), 1_000);
    assert_eq!(query.results_per_page(), 500);
}

#[tokio::test]
async fn collaboration_search_live_query_ranks_only_authorized_candidates_and_marks_lag() {
    let Some(database_url) = std::env::var("COLLAB_SEARCH_REPOSITORY_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_SEARCH_REPOSITORY_TEST_DATABASE_URL is unset; live repository test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    for migration in [EVENTS_UP, PROJECTIONS_UP, SEARCH_UP] {
        sqlx::raw_sql(migration)
            .execute(&pool)
            .await
            .expect("apply collaboration migration");
    }
    let community_id = community(1);
    for (event_id, kind, content) in [
        (vec![1_u8; 32], 9_i32, "alpha alpha alpha public"),
        (
            vec![2_u8; 32],
            1059_i32,
            "alpha alpha alpha alpha alpha private ciphertext",
        ),
    ] {
        sqlx::query(
            r#"
INSERT INTO public.collaboration_events (
    community_id, event_id, author_public_key, event_created_at, kind, tags,
    content, canonical_event_bytes, signature, signature_state, verified_at,
    persistence_class, discriminator
) VALUES ($1, $2, $3, 1900000000, $4, '[]'::jsonb, $5, $6, $7,
          'verified_historical', clock_timestamp(), 'regular', NULL)
"#,
        )
        .bind(community_id.as_uuid())
        .bind(event_id)
        .bind(vec![3_u8; 32])
        .bind(kind)
        .bind(content)
        .bind(content.as_bytes())
        .bind(vec![4_u8; 64])
        .execute(&pool)
        .await
        .expect("insert event");
    }
    sqlx::query(
        r#"
INSERT INTO public.collaboration_search_documents (
    community_id, source_system, source_record_id, source_version,
    source_observed_at, projection_version, document_type, visibility_scope,
    title, body
) VALUES
    ($1, 'zed', 'project:public', '1', clock_timestamp(), 1, 'project',
     'community', 'alpha public project', 'alpha'),
    ($1, 'zed', 'task:restricted', '1', clock_timestamp(), 1, 'task',
     'authorized_restricted', 'alpha restricted task', 'alpha alpha alpha alpha')
"#,
    )
    .bind(community_id.as_uuid())
    .execute(&pool)
    .await
    .expect("insert canonical documents");
    sqlx::query(
        r#"
INSERT INTO public.collaboration_projection_checkpoints (
    community_id, projection_name, source_system, source_record_id,
    source_version, source_observed_at, projection_version, reset_generation,
    drift_state, authoritative_hash, projection_hash, projected_at, last_error
) VALUES ($1, 'collaboration_search', 'zed', 'project:public', '1',
          clock_timestamp(), 1, 1, 'diverged', $2, $3,
          clock_timestamp(), 'seeded drift')
"#,
    )
    .bind(community_id.as_uuid())
    .bind(vec![5_u8; 32])
    .bind(vec![6_u8; 32])
    .execute(&pool)
    .await
    .expect("insert lagging search checkpoint");

    let tenant = tenant(community_id);
    let principal = principal(community_id, true);
    let repository = CollaborationSearchRepository::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("connect repository"),
    )
    .expect("repository");
    let query = CollaborationSearchQuery::new("alpha", SearchMode::FullText, 1, 10).expect("query");
    let result = repository
        .search(
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
        .expect("live search");

    assert_eq!(result.hits.len(), 2);
    assert!(
        result
            .hits
            .windows(2)
            .all(|pair| pair[0].rank >= pair[1].rank)
    );
    assert!(result.hits.iter().any(|hit| matches!(
        hit.record,
        SearchRecordReference::SignedEvent { kind: 9, .. }
    )));
    assert!(result.hits.iter().any(|hit| matches!(
        &hit.record,
        SearchRecordReference::CanonicalDocument { provenance, .. }
            if provenance.source_record_id.as_str() == "project:public"
    )));
    assert!(!result.hits.iter().any(|hit| matches!(
        hit.record,
        SearchRecordReference::SignedEvent { kind: 1059, .. }
    )));
    assert!(!result.hits.iter().any(|hit| matches!(
        &hit.record,
        SearchRecordReference::CanonicalDocument { provenance, .. }
            if provenance.source_record_id.as_str() == "task:restricted"
    )));
    assert!(matches!(
        result.projection_freshness,
        SearchProjectionFreshness::Lagging {
            affected_checkpoints: 1,
            ..
        }
    ));

    let prefix_query =
        CollaborationSearchQuery::new("alp", SearchMode::Prefix, 1, 10).expect("prefix query");
    let prefix_result = repository
        .search(
            SearchAccess {
                tenant: &tenant,
                principal: &principal,
                current_membership_version: AggregateVersion::FIRST,
                community_membership: Some(membership(community_id, &principal)),
                now_millis: 1_900_000_000_000,
            },
            &prefix_query,
        )
        .await
        .expect("live prefix search");
    assert_eq!(prefix_result.hits.len(), 2);
    pool.close().await;
}
