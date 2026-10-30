use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_820_000_700;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.down.sql"
));

#[test]
fn collaboration_channel_schema_is_tenant_fenced_and_provenance_indexed() {
    for table in [
        "collaboration_communities",
        "collaboration_community_memberships",
        "collaboration_join_policy_acceptances",
        "collaboration_channels",
        "collaboration_channel_invites",
        "collaboration_channel_memberships",
    ] {
        assert!(UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(UP.contains(&format!("'{table}'")));
        assert!(UP.contains(&format!("{table}_provenance")));
        assert!(DOWN.contains(&format!("DROP TABLE public.{table};")));
    }
    for required in [
        "PRIMARY KEY (community_id, principal_id)",
        "PRIMARY KEY (community_id, channel_id)",
        "PRIMARY KEY (community_id, channel_id, principal_id)",
        "FOREIGN KEY (community_id, creator_principal_id)",
        "FOREIGN KEY (community_id, principal_id)",
        "source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')",
        "source_record_id",
        "source_version",
        "source_observed_at",
        "integrity_algorithm",
        "aggregate_version numeric(20, 0)",
        "membership_version numeric(20, 0)",
        "channel_version numeric(20, 0)",
        "invite_version numeric(20, 0)",
        "UNIQUE (community_id, token_hash)",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE FOR ALL",
        "AS RESTRICTIVE FOR ALL",
        "current_setting(''app.community_id'', true)",
    ] {
        assert!(UP.contains(required), "missing schema invariant {required}");
    }
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), 6);
}

#[tokio::test]
async fn collaboration_channel_schema_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let channel_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(channel_migrations.len(), 2);
    let up = channel_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("channel up migration");
    let down = channel_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("channel down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_channel_schema_enforces_live_tenant_isolation_and_rolls_down() {
    let Some(database_url) = std::env::var("COLLAB_CHANNEL_MIGRATION_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_CHANNEL_MIGRATION_TEST_DATABASE_URL is unset; live channel migration test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply channel migration");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_channel_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_channel_request;",
    )
    .execute(&pool)
    .await
    .expect("create least-privilege request role");

    let community_a = Uuid::from_u128(1);
    let community_b = Uuid::from_u128(2);
    let mut transaction = pool.begin().await.expect("begin tenant A transaction");
    sqlx::query("SET LOCAL ROLE collaboration_channel_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, 'a.example', 'active', 1, 'buzz', 'communities:1', '30', now(), now(), now())",
    )
    .bind(community_a)
    .execute(&mut *transaction)
    .await
    .expect("insert tenant A community");
    transaction.commit().await.expect("commit tenant A");

    let mut transaction = pool.begin().await.expect("begin tenant B transaction");
    sqlx::query("SET LOCAL ROLE collaboration_channel_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_b.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant B");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_communities WHERE community_id = $1",
    )
    .bind(community_a)
    .fetch_one(&mut *transaction)
    .await
    .expect("query foreign tenant");
    assert_eq!(visible, 0);
    let foreign_insert = sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'foreign.example', 'active', 1, 'buzz', 'communities:foreign', now(), now(), now())",
    )
    .bind(community_a)
    .execute(&mut *transaction)
    .await;
    let error = foreign_insert.expect_err("foreign tenant insert must fail");
    assert_eq!(
        error
            .as_database_error()
            .and_then(|error| error.code())
            .as_deref(),
        Some("42501"),
        "the insert must reach the tenant policy rather than fail SQL syntax or shape"
    );
    transaction.rollback().await.expect("rollback tenant B");

    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
    let remaining: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.collaboration_communities')::text")
            .fetch_one(&pool)
            .await
            .expect("query rolled-down table");
    assert_eq!(remaining, None);
    println!("community schema: RLS rejection code and reversible cleanup verified");
}
