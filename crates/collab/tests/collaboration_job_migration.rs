use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_823_000_200;
const CHANNELS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260823000200_collaboration_jobs.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260823000200_collaboration_jobs.down.sql"
));

#[test]
fn collaboration_job_schema_versions_every_canonical_command() {
    for required in [
        "PRIMARY KEY (community_id, job_id)",
        "PRIMARY KEY (community_id, job_id, version)",
        "UNIQUE (community_id, operation_id)",
        "command_type IN ('request', 'accept', 'progress', 'result', 'cancel', 'error')",
        "command_type IN ('accept', 'progress', 'result')",
        "current_state IN (",
        "current_executor_principal_id = target_executor_principal_id",
        "version BETWEEN 1 AND 18446744073709551615",
        "recorded_at >= occurred_at",
    ] {
        assert!(
            UP.contains(required),
            "missing job-version invariant {required}"
        );
    }
}

#[test]
fn collaboration_job_schema_indexes_bounded_tree_ancestry() {
    for required in [
        "CREATE TABLE public.collaboration_job_ancestry",
        "depth smallint NOT NULL CHECK (depth BETWEEN 1 AND 8)",
        "PRIMARY KEY (community_id, ancestor_job_id, descendant_job_id)",
        "UNIQUE (community_id, descendant_job_id, depth)",
        "CHECK (ancestor_job_id <> descendant_job_id)",
        "CREATE INDEX collaboration_job_ancestry_descendants",
        "CREATE INDEX collaboration_job_ancestry_direct_children",
        "WHERE depth = 1",
    ] {
        assert!(
            UP.contains(required),
            "missing ancestry invariant {required}"
        );
    }
}

#[test]
fn collaboration_job_schema_allows_exactly_one_recoverable_active_lease() {
    for required in [
        "PRIMARY KEY (community_id, job_id, lease_generation)",
        "UNIQUE (community_id, lease_id)",
        "FOREIGN KEY (community_id, job_id, job_version)",
        "CREATE UNIQUE INDEX collaboration_job_executor_leases_one_active",
        "WHERE state = 'active'",
        "CREATE INDEX collaboration_job_executor_leases_recovery",
        "recovery_after >= expires_at",
        "release_reason IN ('completed', 'cancelled', 'failed', 'expired', 'replaced')",
    ] {
        assert!(
            UP.contains(required),
            "missing executor-lease invariant {required}"
        );
    }
}

#[test]
fn collaboration_job_schema_is_tenant_fenced_and_exactly_reversible() {
    for table in [
        "collaboration_jobs",
        "collaboration_job_versions",
        "collaboration_job_ancestry",
        "collaboration_job_executor_leases",
    ] {
        assert!(UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(UP.contains(&format!("'{table}'")));
        assert!(DOWN.contains(&format!("DROP TABLE public.{table};")));
    }
    for required in [
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE FOR ALL",
        "AS RESTRICTIVE FOR ALL",
        "current_setting(''app.community_id'', true)",
        "REFERENCES public.collaboration_communities (community_id)",
        "REFERENCES public.collaboration_jobs (community_id, job_id)",
    ] {
        assert!(UP.contains(required), "missing tenant invariant {required}");
    }
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), 4);
}

#[tokio::test]
async fn collaboration_job_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let job_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(job_migrations.len(), 2);
    let up = job_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("job up migration");
    let down = job_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("job down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_job_schema_enforces_live_tenant_ancestry_and_lease_constraints() {
    let Some(database_url) = std::env::var("COLLAB_JOB_MIGRATION_TEST_DATABASE_URL").ok() else {
        eprintln!(
            "COLLAB_JOB_MIGRATION_TEST_DATABASE_URL is unset; live job migration test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(CHANNELS_UP)
        .execute(&pool)
        .await
        .expect("apply channel migration");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply job migration");

    let community_id = Uuid::from_u128(1);
    let requester_id = Uuid::from_u128(2);
    let executor_id = Uuid::from_u128(3);
    let parent_job_id = Uuid::from_u128(4);
    let child_job_id = Uuid::from_u128(5);
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'jobs.example', 'active', 1, 'zed', 'community:jobs', now(), now(), now())",
    )
    .bind(community_id)
    .execute(&pool)
    .await
    .expect("insert community");
    for job_id in [parent_job_id, child_job_id] {
        sqlx::query(
            "INSERT INTO public.collaboration_jobs (community_id, job_id, requester_principal_id, target_executor_principal_id, current_version, current_state, requested_at, updated_at) VALUES ($1, $2, $3, $4, 1, 'requested', now(), now())",
        )
        .bind(community_id)
        .bind(job_id)
        .bind(requester_id)
        .bind(executor_id)
        .execute(&pool)
        .await
        .expect("insert requested job");
        sqlx::query(
            "INSERT INTO public.collaboration_job_versions (community_id, job_id, version, operation_id, command_type, actor_principal_id, occurred_at) VALUES ($1, $2, 1, $3, 'request', $4, now())",
        )
        .bind(community_id)
        .bind(job_id)
        .bind(Uuid::from_u128(job_id.as_u128() + 100))
        .bind(requester_id)
        .execute(&pool)
        .await
        .expect("insert request version");
    }
    sqlx::query(
        "INSERT INTO public.collaboration_job_ancestry (community_id, ancestor_job_id, descendant_job_id, depth, created_at) VALUES ($1, $2, $3, 1, now())",
    )
    .bind(community_id)
    .bind(parent_job_id)
    .bind(child_job_id)
    .execute(&pool)
    .await
    .expect("insert direct ancestry");
    assert!(
        sqlx::query(
            "INSERT INTO public.collaboration_job_ancestry (community_id, ancestor_job_id, descendant_job_id, depth, created_at) VALUES ($1, $2, $2, 1, now())",
        )
        .bind(community_id)
        .bind(parent_job_id)
        .execute(&pool)
        .await
        .is_err(),
        "self-ancestry must fail"
    );

    sqlx::query(
        "INSERT INTO public.collaboration_job_versions (community_id, job_id, version, operation_id, command_type, actor_principal_id, executor_principal_id, occurred_at) VALUES ($1, $2, 2, $3, 'accept', $4, $4, now())",
    )
    .bind(community_id)
    .bind(parent_job_id)
    .bind(Uuid::from_u128(200))
    .bind(executor_id)
    .execute(&pool)
    .await
    .expect("insert accepted version");
    sqlx::query(
        "UPDATE public.collaboration_jobs SET current_version = 2, current_state = 'accepted', current_executor_principal_id = $3, updated_at = now() WHERE community_id = $1 AND job_id = $2",
    )
    .bind(community_id)
    .bind(parent_job_id)
    .bind(executor_id)
    .execute(&pool)
    .await
    .expect("advance job head");
    let insert_lease = |generation: i64, lease_id: Uuid| {
        sqlx::query(
            "INSERT INTO public.collaboration_job_executor_leases (community_id, job_id, job_version, lease_generation, lease_id, executor_principal_id, state, acquired_at, last_heartbeat_at, expires_at, recovery_after) VALUES ($1, $2, 2, $3, $4, $5, 'active', now(), now(), now() + interval '30 seconds', now() + interval '60 seconds')",
        )
        .bind(community_id)
        .bind(parent_job_id)
        .bind(generation)
        .bind(lease_id)
        .bind(executor_id)
    };
    insert_lease(1, Uuid::from_u128(300))
        .execute(&pool)
        .await
        .expect("insert active executor lease");
    assert!(
        insert_lease(2, Uuid::from_u128(301))
            .execute(&pool)
            .await
            .is_err(),
        "a job must not have two active executor leases"
    );
    sqlx::query(
        "UPDATE public.collaboration_job_executor_leases SET state = 'released', released_at = now(), release_reason = 'expired' WHERE community_id = $1 AND job_id = $2 AND lease_generation = 1",
    )
    .bind(community_id)
    .bind(parent_job_id)
    .execute(&pool)
    .await
    .expect("release expired lease");
    insert_lease(2, Uuid::from_u128(301))
        .execute(&pool)
        .await
        .expect("insert recovery lease");

    sqlx::raw_sql(
        "CREATE ROLE collaboration_job_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON public.collaboration_jobs, \
         public.collaboration_job_versions, public.collaboration_job_ancestry, \
         public.collaboration_job_executor_leases TO collaboration_job_request;",
    )
    .execute(&pool)
    .await
    .expect("create request role");
    let mut transaction = pool.begin().await.expect("begin tenant transaction");
    sqlx::query("SET LOCAL ROLE collaboration_job_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(Uuid::from_u128(99).to_string())
        .execute(&mut *transaction)
        .await
        .expect("set foreign tenant");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_jobs WHERE community_id = $1",
    )
    .bind(community_id)
    .fetch_one(&mut *transaction)
    .await
    .expect("query foreign tenant");
    assert_eq!(visible, 0);
    transaction.rollback().await.expect("rollback tenant query");

    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll job migration down");
    let remaining: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.collaboration_jobs')::text")
            .fetch_one(&pool)
            .await
            .expect("query rolled-down table");
    assert_eq!(remaining, None);
    pool.close().await;
}
