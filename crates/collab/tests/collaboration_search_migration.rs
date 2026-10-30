use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_820_000_500;
const EVENTS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const SEARCH_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000500_collaboration_search.up.sql"
));
const SEARCH_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000500_collaboration_search.down.sql"
));

#[test]
fn collaboration_search_schema_is_tenant_fenced_and_privacy_positive() {
    for required in [
        "PRIMARY KEY (community_id, source_system, source_record_id)",
        "ALTER TABLE public.collaboration_events",
        "visibility_scope IN ('community', 'authorized_restricted', 'excluded')",
        "kind IN (0, 9, 40002, 45001, 45003)",
        "WHEN visibility_scope = 'community'",
        "WHERE search_tsv IS NOT NULL",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(
            SEARCH_UP.contains(required),
            "missing search invariant {required}"
        );
    }
    assert!(SEARCH_DOWN.starts_with("DROP TABLE public.collaboration_search_documents;"));
    assert!(!SEARCH_DOWN.contains("CASCADE"));
}

#[tokio::test]
async fn collaboration_search_migration_has_stable_up_and_down_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let search_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(search_migrations.len(), 2);
    let up = search_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("search up migration");
    let down = search_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("search down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(SEARCH_UP).as_slice());
    assert_eq!(
        down.checksum.as_ref(),
        Sha384::digest(SEARCH_DOWN).as_slice()
    );
}

#[tokio::test]
async fn collaboration_search_excludes_private_content_from_vector_and_partial_index() {
    let Some(database_url) = std::env::var("COLLAB_SEARCH_TEST_DATABASE_URL").ok() else {
        eprintln!("COLLAB_SEARCH_TEST_DATABASE_URL is unset; live search migration test skipped");
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect to isolated PostgreSQL database");
    sqlx::raw_sql(EVENTS_UP)
        .execute(&pool)
        .await
        .expect("apply events migration");
    sqlx::raw_sql(SEARCH_UP)
        .execute(&pool)
        .await
        .expect("apply search migration");

    let community_id = Uuid::from_u128(1);
    for (event_id, kind, content) in [
        (vec![1_u8; 32], 9_i32, "public searchable marker"),
        (vec![2_u8; 32], 1059_i32, "private searchable marker"),
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
        .bind(community_id)
        .bind(&event_id)
        .bind(vec![3_u8; 32])
        .bind(kind)
        .bind(content)
        .bind(content.as_bytes())
        .bind(vec![4_u8; 64])
        .execute(&pool)
        .await
        .expect("insert authoritative event");
    }

    sqlx::query(
        r#"
INSERT INTO public.collaboration_search_documents (
    community_id, source_system, source_record_id, source_version,
    source_observed_at, projection_version, document_type, visibility_scope,
    title, body
) VALUES
    ($1, 'zed', 'project:public', '1', clock_timestamp(), 1, 'project',
     'community', 'public project', 'canonical searchable marker'),
    ($1, 'zed', 'project:restricted', '1', clock_timestamp(), 1, 'project',
     'authorized_restricted', 'restricted project', 'canonical searchable marker')
"#,
    )
    .bind(community_id)
    .execute(&pool)
    .await
    .expect("insert canonical search projections");

    let vectors = sqlx::query_as::<_, (i32, bool)>(
        "SELECT kind, search_tsv IS NULL FROM public.collaboration_events ORDER BY kind",
    )
    .fetch_all(&pool)
    .await
    .expect("load generated vectors");
    assert_eq!(vectors, vec![(9, false), (1059, true)]);

    let matches: Vec<i32> = sqlx::query_scalar(
        r#"
SELECT kind
FROM public.collaboration_events
WHERE search_tsv @@ websearch_to_tsquery('simple', 'searchable marker')
ORDER BY kind
"#,
    )
    .fetch_all(&pool)
    .await
    .expect("query generated vectors");
    assert_eq!(matches, vec![9]);

    let canonical_vectors = sqlx::query_as::<_, (String, bool)>(
        "SELECT source_record_id, search_tsv IS NULL FROM public.collaboration_search_documents ORDER BY source_record_id",
    )
    .fetch_all(&pool)
    .await
    .expect("load canonical generated vectors");
    assert_eq!(
        canonical_vectors,
        vec![
            ("project:public".to_owned(), false),
            ("project:restricted".to_owned(), true),
        ]
    );

    let index_predicate: String = sqlx::query_scalar(
        r#"
SELECT pg_get_expr(index_definition.indpred, index_definition.indrelid)
FROM pg_index AS index_definition
JOIN pg_class AS index_relation ON index_relation.oid = index_definition.indexrelid
WHERE index_relation.relname = 'collaboration_events_search_fts'
"#,
    )
    .fetch_one(&pool)
    .await
    .expect("load partial index predicate");
    assert_eq!(index_predicate, "(search_tsv IS NOT NULL)");

    sqlx::raw_sql(SEARCH_DOWN)
        .execute(&pool)
        .await
        .expect("roll back search migration");
    pool.close().await;
}
