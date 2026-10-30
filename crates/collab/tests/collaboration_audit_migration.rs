use std::path::Path;

use serde_json::json;
use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_824_000_400;
const CHANNELS_UP: &str =
    include_str!("../migrations/20260820000700_collaboration_channels.up.sql");
const CHANNELS_DOWN: &str =
    include_str!("../migrations/20260820000700_collaboration_channels.down.sql");
const UP: &str = include_str!("../migrations/20260824000400_collaboration_audit.up.sql");
const DOWN: &str = include_str!("../migrations/20260824000400_collaboration_audit.down.sql");

#[test]
fn collaboration_audit_schema_is_append_only_and_chain_linked() {
    for required in [
        "PRIMARY KEY (community_id, sequence)",
        "UNIQUE (community_id, entry_hash)",
        "UNIQUE (community_id, operation_id)",
        "octet_length(entry_hash) = 32",
        "previous_sequence numeric(20, 0) GENERATED ALWAYS AS",
        "FOREIGN KEY (community_id, previous_sequence, previous_hash)",
        "sequence = bridge_source_sequence + 1",
        "bridge_source = 'buzz_v1'",
        "BEFORE UPDATE OR DELETE ON public.collaboration_audit_entries",
        "collaboration audit entries are immutable",
    ] {
        assert!(
            UP.contains(required),
            "missing audit-entry invariant {required}"
        );
    }
}

#[test]
fn collaboration_audit_schema_has_one_head_and_monotonic_export_cursors() {
    for required in [
        "CREATE TABLE public.collaboration_audit_heads",
        "community_id uuid PRIMARY KEY",
        "NEW.sequence <> OLD.sequence + 1",
        "entry.previous_hash = OLD.entry_hash",
        "CREATE TABLE public.collaboration_audit_export_cursors",
        "PRIMARY KEY (community_id, exporter_id)",
        "NEW.cursor_version <> OLD.cursor_version + 1",
        "NEW.exported_through_sequence <= OLD.exported_through_sequence",
        "FOREIGN KEY (community_id, exported_through_sequence, exported_through_hash)",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE FOR ALL",
        "AS RESTRICTIVE FOR ALL",
        "current_setting(''app.community_id'', true)",
    ] {
        assert!(
            UP.contains(required),
            "missing audit-head invariant {required}"
        );
    }
}

#[test]
fn collaboration_audit_schema_rolls_back_in_dependency_order() {
    let expected = [
        "DROP TABLE public.collaboration_audit_export_cursors;",
        "DROP FUNCTION public.guard_collaboration_audit_export_cursor_mutation();",
        "DROP TABLE public.collaboration_audit_heads;",
        "DROP FUNCTION public.guard_collaboration_audit_head_mutation();",
        "DROP TABLE public.collaboration_audit_entries;",
        "DROP FUNCTION public.reject_collaboration_audit_entry_mutation();",
    ];
    assert_eq!(DOWN.lines().collect::<Vec<_>>(), expected);
    assert!(!DOWN.contains("CASCADE"));
}

#[tokio::test]
async fn collaboration_audit_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(file!())
        .parent()
        .expect("audit migration test has a parent directory")
        .join("../migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let audit_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(audit_migrations.len(), 2);
    let up = audit_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("audit up migration");
    let down = audit_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("audit down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_audit_schema_enforces_live_immutability_heads_and_tenants() {
    let Some(database_url) = std::env::var("COLLAB_AUDIT_MIGRATION_TEST_DATABASE_URL").ok() else {
        eprintln!(
            "COLLAB_AUDIT_MIGRATION_TEST_DATABASE_URL is unset; live audit migration test skipped"
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
        .expect("apply audit migration");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_audit_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_audit_request;",
    )
    .execute(&pool)
    .await
    .expect("create least-privilege request role");

    let community_a = Uuid::from_u128(1);
    let community_b = Uuid::from_u128(2);
    for (community_id, host) in [
        (community_a, "audit-a.example"),
        (community_b, "audit-b.example"),
    ] {
        let mut transaction = pool.begin().await.expect("begin community transaction");
        assume_tenant(&mut transaction, community_id).await;
        sqlx::query(
            "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'active', 1, 'zed', $3, now(), now(), now())",
        )
        .bind(community_id)
        .bind(host)
        .bind(format!("community:{community_id}"))
        .execute(&mut *transaction)
        .await
        .expect("insert community");
        transaction.commit().await.expect("commit community");
    }

    let first_hash = vec![1_u8; 32];
    let second_hash = vec![2_u8; 32];
    let mut transaction = pool.begin().await.expect("begin chain transaction");
    assume_tenant(&mut transaction, community_a).await;
    insert_entry(
        &mut transaction,
        community_a,
        1,
        Uuid::from_u128(11),
        &first_hash,
        None,
    )
    .await;
    sqlx::query(
        "INSERT INTO public.collaboration_audit_heads (community_id, sequence, entry_hash) VALUES ($1, 1, $2)",
    )
    .bind(community_a)
    .bind(&first_hash)
    .execute(&mut *transaction)
    .await
    .expect("insert first head");
    insert_entry(
        &mut transaction,
        community_a,
        2,
        Uuid::from_u128(12),
        &second_hash,
        Some(&first_hash),
    )
    .await;
    sqlx::query(
        "UPDATE public.collaboration_audit_heads SET sequence = 2, entry_hash = $2 WHERE community_id = $1 AND sequence = 1 AND entry_hash = $3",
    )
    .bind(community_a)
    .bind(&second_hash)
    .bind(&first_hash)
    .execute(&mut *transaction)
    .await
    .expect("advance head once");
    sqlx::query(
        "INSERT INTO public.collaboration_audit_export_cursors (community_id, exporter_id, cursor_version, exported_through_sequence, exported_through_hash) VALUES ($1, 'operator_archive', 1, 2, $2)",
    )
    .bind(community_a)
    .bind(&second_hash)
    .execute(&mut *transaction)
    .await
    .expect("insert export cursor");
    transaction.commit().await.expect("commit chain");

    let mut transaction = pool.begin().await.expect("begin immutable update");
    assume_tenant(&mut transaction, community_a).await;
    let mutation = sqlx::query(
        "UPDATE public.collaboration_audit_entries SET outcome = 'failed' WHERE community_id = $1 AND sequence = 1",
    )
    .bind(community_a)
    .execute(&mut *transaction)
    .await;
    assert!(mutation.is_err());
    transaction.rollback().await.expect("rollback mutation");

    let mut transaction = pool.begin().await.expect("begin immutable delete");
    assume_tenant(&mut transaction, community_a).await;
    let deletion = sqlx::query(
        "DELETE FROM public.collaboration_audit_entries WHERE community_id = $1 AND sequence = 1",
    )
    .bind(community_a)
    .execute(&mut *transaction)
    .await;
    assert!(deletion.is_err());
    transaction.rollback().await.expect("rollback deletion");

    let mut transaction = pool.begin().await.expect("begin cursor regression");
    assume_tenant(&mut transaction, community_a).await;
    let regression = sqlx::query(
        "UPDATE public.collaboration_audit_export_cursors SET cursor_version = 2, exported_through_sequence = 1, exported_through_hash = $2 WHERE community_id = $1 AND exporter_id = 'operator_archive'",
    )
    .bind(community_a)
    .bind(&first_hash)
    .execute(&mut *transaction)
    .await;
    assert!(regression.is_err());
    transaction
        .rollback()
        .await
        .expect("rollback cursor regression");

    let mut transaction = pool.begin().await.expect("begin tenant B read");
    assume_tenant(&mut transaction, community_b).await;
    for table in [
        "collaboration_audit_entries",
        "collaboration_audit_heads",
        "collaboration_audit_export_cursors",
    ] {
        let count: i64 = sqlx::query_scalar(&format!(
            "SELECT count(*) FROM public.{table} WHERE community_id = $1"
        ))
        .bind(community_a)
        .fetch_one(&mut *transaction)
        .await
        .expect("query foreign tenant");
        assert_eq!(count, 0, "foreign tenant saw rows in {table}");
    }
    transaction
        .rollback()
        .await
        .expect("rollback tenant B read");

    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll audit migration down");
    for table in [
        "collaboration_audit_entries",
        "collaboration_audit_heads",
        "collaboration_audit_export_cursors",
    ] {
        let remaining: Option<String> =
            sqlx::query_scalar(&format!("SELECT to_regclass('public.{table}')::text"))
                .fetch_one(&pool)
                .await
                .expect("query rolled-down table");
        assert_eq!(remaining, None);
    }
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
}

async fn assume_tenant(
    transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    community_id: Uuid,
) {
    sqlx::query("SET LOCAL ROLE collaboration_audit_request")
        .execute(&mut **transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_id.to_string())
        .execute(&mut **transaction)
        .await
        .expect("set admitted community");
}

async fn insert_entry(
    transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    community_id: Uuid,
    sequence: i64,
    operation_id: Uuid,
    entry_hash: &[u8],
    previous_hash: Option<&[u8]>,
) {
    sqlx::query(
        "INSERT INTO public.collaboration_audit_entries (community_id, sequence, entry_hash, previous_hash, operation_id, action, actor_principal_id, outcome, occurred_at_millis, fields) VALUES ($1, $2, $3, $4, $5, 'workflow.action.completed', NULL, 'succeeded', 1900000000000, $6)",
    )
    .bind(community_id)
    .bind(sequence)
    .bind(entry_hash)
    .bind(previous_hash)
    .bind(operation_id)
    .bind(json!([]))
    .execute(&mut **transaction)
    .await
    .expect("insert audit entry");
}
