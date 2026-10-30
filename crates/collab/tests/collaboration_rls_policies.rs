use sqlx::PgPool;
use uuid::Uuid;

const IDENTITY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260815000100_collaboration_identity_bindings.up.sql"
));
const EVENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const HEADS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000200_collaboration_event_heads.up.sql"
));
const PROJECTIONS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000300_collaboration_projections.up.sql"
));
const OUTBOX: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000400_collaboration_outbox.up.sql"
));
const SEARCH: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000500_collaboration_search.up.sql"
));
const CHECKPOINTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000600_collaboration_migration_checkpoints.up.sql"
));

#[test]
fn collaboration_restrictive_policies_have_explicit_permissive_candidates() {
    for (name, migration) in [
        ("identity", IDENTITY),
        ("events", EVENTS),
        ("heads", HEADS),
        ("projections", PROJECTIONS),
        ("outbox", OUTBOX),
        ("search", SEARCH),
        ("checkpoints", CHECKPOINTS),
    ] {
        let restrictive = migration.matches("AS RESTRICTIVE").count();
        let permissive = migration.matches("AS PERMISSIVE").count();
        assert!(restrictive > 0, "{name} must retain a restrictive policy");
        assert_eq!(
            permissive, restrictive,
            "{name} must pair every restrictive policy with one candidate policy"
        );
        assert_eq!(migration.matches("USING (true)").count(), permissive);
        assert_eq!(migration.matches("WITH CHECK (true)").count(), permissive);
    }
}

#[tokio::test]
async fn collaboration_rls_allows_one_tenant_and_hides_it_from_another_request_role() {
    let Some(database_url) = std::env::var("COLLAB_RLS_POLICY_TEST_DATABASE_URL").ok() else {
        eprintln!("COLLAB_RLS_POLICY_TEST_DATABASE_URL is unset; live RLS policy test skipped");
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    for migration in [
        IDENTITY,
        EVENTS,
        HEADS,
        PROJECTIONS,
        OUTBOX,
        SEARCH,
        CHECKPOINTS,
    ] {
        sqlx::raw_sql(migration)
            .execute(&pool)
            .await
            .expect("apply collaboration migration");
    }
    let unpaired: Vec<String> = sqlx::query_scalar(
        "SELECT restrictive.tablename FROM (SELECT tablename FROM pg_policies WHERE schemaname = 'public' GROUP BY tablename HAVING bool_or(permissive = 'RESTRICTIVE')) AS restrictive WHERE NOT EXISTS (SELECT 1 FROM pg_policies AS permissive WHERE permissive.schemaname = 'public' AND permissive.tablename = restrictive.tablename AND permissive.permissive = 'PERMISSIVE') ORDER BY restrictive.tablename",
    )
    .fetch_all(&pool)
    .await
    .expect("inspect installed policies");
    assert!(unpaired.is_empty(), "restrictive-only tables: {unpaired:?}");

    sqlx::raw_sql(
        "CREATE ROLE collaboration_rls_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_rls_request;",
    )
    .execute(&pool)
    .await
    .expect("create least-privilege request role");
    let community_a = Uuid::from_u128(1);
    let community_b = Uuid::from_u128(2);
    let run_id = Uuid::from_u128(10);

    let mut transaction = pool.begin().await.expect("begin tenant A transaction");
    sqlx::query("SET LOCAL ROLE collaboration_rls_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "INSERT INTO public.collaboration_migration_runs (run_id, community_id, source_revision) VALUES ($1, $2, 'revision-a')",
    )
    .bind(run_id)
    .bind(community_a)
    .execute(&mut *transaction)
    .await
    .expect("insert tenant A through paired policies");
    transaction.commit().await.expect("commit tenant A");

    let mut transaction = pool.begin().await.expect("begin tenant B transaction");
    sqlx::query("SET LOCAL ROLE collaboration_rls_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_b.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant B");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_migration_runs WHERE run_id = $1",
    )
    .bind(run_id)
    .fetch_one(&mut *transaction)
    .await
    .expect("query tenant A run as tenant B");
    assert_eq!(visible, 0);
    let foreign_insert = sqlx::query(
        "INSERT INTO public.collaboration_migration_runs (run_id, community_id, source_revision) VALUES ($1, $2, 'foreign')",
    )
    .bind(Uuid::from_u128(11))
    .bind(community_a)
    .execute(&mut *transaction)
    .await;
    assert!(foreign_insert.is_err());
    transaction.rollback().await.expect("rollback tenant B");
}
