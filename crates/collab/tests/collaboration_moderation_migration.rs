use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_824_000_500;
const CHANNELS_UP: &str =
    include_str!("../migrations/20260820000700_collaboration_channels.up.sql");
const CHANNELS_DOWN: &str =
    include_str!("../migrations/20260820000700_collaboration_channels.down.sql");
const AUDIT_UP: &str = include_str!("../migrations/20260824000400_collaboration_audit.up.sql");
const AUDIT_DOWN: &str = include_str!("../migrations/20260824000400_collaboration_audit.down.sql");
const UP: &str = include_str!("../migrations/20260824000500_collaboration_moderation.up.sql");
const DOWN: &str = include_str!("../migrations/20260824000500_collaboration_moderation.down.sql");

const TABLES: [&str; 7] = [
    "collaboration_moderation_actions",
    "collaboration_moderation_reports",
    "collaboration_moderation_report_resolutions",
    "collaboration_moderation_restriction_versions",
    "collaboration_personal_mute_versions",
    "collaboration_identity_archive_versions",
    "collaboration_community_archive_versions",
];

#[test]
fn collaboration_moderation_schema_preserves_private_reports_and_attributed_actions() {
    for required in [
        "PRIMARY KEY (community_id, report_id)",
        "UNIQUE (community_id, filed_operation_id)",
        "target_kind IN ('event', 'principal', 'blob')",
        "target_kind = 'event'",
        "target_kind = 'principal'",
        "target_kind = 'blob'",
        "octet_length(private_context) BETWEEN 1 AND 4096",
        "'nudity', 'malware'",
        "PRIMARY KEY (community_id, action_id)",
        "UNIQUE (community_id, operation_id)",
        "REFERENCES public.collaboration_audit_entries (community_id, operation_id)",
        "PRIMARY KEY (community_id, report_id)",
        "resulting_version = 2",
        "collaboration_moderation_report_resolutions_immutable",
    ] {
        assert!(
            UP.contains(required),
            "missing report/action invariant {required}"
        );
    }
}

#[test]
fn collaboration_moderation_schema_keeps_current_state_unique_and_history_recoverable() {
    for required in [
        "PRIMARY KEY (community_id, target_principal_id, restriction_version)",
        "collaboration_moderation_restrictions_current",
        "PRIMARY KEY (community_id, owner_principal_id, muted_principal_id, mute_version)",
        "collaboration_personal_mutes_current",
        "collaboration_personal_mutes_active",
        "PRIMARY KEY (community_id, identity_public_key, archive_version)",
        "collaboration_identity_archives_current",
        "collaboration_identity_archives_active",
        "PRIMARY KEY (community_id, archive_version)",
        "collaboration_community_archives_current",
        "collaboration_community_archives_active",
        "WHERE is_current",
        "OLD.is_current",
        "NOT NEW.is_current",
        "to_jsonb(NEW) - 'is_current'",
        "collaboration moderation versions cannot be deleted",
    ] {
        assert!(
            UP.contains(required),
            "missing state-history invariant {required}"
        );
    }
    assert_eq!(UP.matches("WHERE is_current;").count(), 4);
}

#[test]
fn collaboration_moderation_schema_is_tenant_fenced_and_provenance_complete() {
    for table in TABLES {
        assert!(UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(UP.contains(&format!("'{table}'")));
    }
    for required in [
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE FOR ALL",
        "AS RESTRICTIVE FOR ALL",
        "current_setting(''app.community_id'', true)",
        "REFERENCES public.collaboration_communities (community_id)",
        "REFERENCES public.collaboration_community_memberships (community_id, principal_id)",
        "source_system",
        "source_record_id",
        "source_version",
        "source_observed_at",
        "integrity_algorithm",
        "integrity_value",
    ] {
        assert!(
            UP.contains(required),
            "missing tenant/provenance invariant {required}"
        );
    }
    assert_eq!(
        UP.matches("source_system text NOT NULL").count(),
        TABLES.len()
    );
    assert_eq!(
        UP.matches("source_record_id text NOT NULL").count(),
        TABLES.len()
    );
    assert_eq!(
        UP.matches("source_observed_at timestamptz NOT NULL")
            .count(),
        TABLES.len()
    );
    assert_eq!(
        UP.matches("CHECK ((integrity_algorithm IS NULL) = (integrity_value IS NULL))")
            .count(),
        TABLES.len()
    );
}

#[test]
fn collaboration_moderation_schema_rolls_back_in_dependency_order() {
    let expected = [
        "DROP TABLE public.collaboration_community_archive_versions;",
        "DROP TABLE public.collaboration_identity_archive_versions;",
        "DROP TABLE public.collaboration_personal_mute_versions;",
        "DROP TABLE public.collaboration_moderation_restriction_versions;",
        "DROP TABLE public.collaboration_moderation_report_resolutions;",
        "DROP TABLE public.collaboration_moderation_reports;",
        "DROP TABLE public.collaboration_moderation_actions;",
        "DROP FUNCTION public.guard_collaboration_moderation_version_retirement();",
        "DROP FUNCTION public.reject_collaboration_moderation_history_mutation();",
    ];
    assert_eq!(DOWN.lines().collect::<Vec<_>>(), expected);
    assert!(!DOWN.contains("CASCADE"));
}

#[tokio::test]
async fn collaboration_moderation_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(file!())
        .parent()
        .expect("moderation migration test has a parent directory")
        .join("../migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let moderation_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(moderation_migrations.len(), 2);
    let up = moderation_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("moderation up migration");
    let down = moderation_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("moderation down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_moderation_schema_enforces_live_uniqueness_history_tenants_and_rollback() {
    let Some(database_url) = std::env::var("COLLAB_MODERATION_MIGRATION_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_MODERATION_MIGRATION_TEST_DATABASE_URL is unset; live moderation migration test skipped"
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
    sqlx::raw_sql(AUDIT_UP)
        .execute(&pool)
        .await
        .expect("apply audit migration");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply moderation migration");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_moderation_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_moderation_request;",
    )
    .execute(&pool)
    .await
    .expect("create request role");

    let community_id = Uuid::from_u128(1);
    let foreign_community_id = Uuid::from_u128(2);
    let actor_principal_id = Uuid::from_u128(3);
    let target_principal_id = Uuid::from_u128(4);
    for (id, host, source_record_id) in [
        (community_id, "moderation.example", "community:moderation"),
        (
            foreign_community_id,
            "moderation-foreign.example",
            "community:moderation-foreign",
        ),
    ] {
        sqlx::query(
            "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'active', 1, 'zed', $3, '1', now(), now(), now())",
        )
        .bind(id)
        .bind(host)
        .bind(source_record_id)
        .execute(&pool)
        .await
        .expect("insert community");
    }
    for (principal_id, role) in [
        (actor_principal_id, "admin"),
        (target_principal_id, "member"),
    ] {
        sqlx::query(
            "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1, $2, $3, 'active', 1, now(), now(), 'zed', $4, '1', now())",
        )
        .bind(community_id)
        .bind(principal_id)
        .bind(role)
        .bind(format!("membership:{principal_id}"))
        .execute(&pool)
        .await
        .expect("insert membership");
    }

    let first_operation_id = Uuid::from_u128(10);
    let second_operation_id = Uuid::from_u128(11);
    let first_hash = vec![1_u8; 32];
    let second_hash = vec![2_u8; 32];
    for (sequence, operation_id, entry_hash, previous_hash) in [
        (1_i64, first_operation_id, &first_hash, None),
        (2_i64, second_operation_id, &second_hash, Some(&first_hash)),
    ] {
        sqlx::query(
            "INSERT INTO public.collaboration_audit_entries (community_id, sequence, entry_hash, previous_hash, operation_id, action, actor_principal_id, outcome, occurred_at_millis, fields) VALUES ($1, $2, $3, $4, $5, 'moderation.apply_restriction', $6, 'succeeded', $7, '[]'::jsonb)",
        )
        .bind(community_id)
        .bind(sequence)
        .bind(entry_hash)
        .bind(previous_hash)
        .bind(operation_id)
        .bind(actor_principal_id)
        .bind(100_i64 + sequence)
        .execute(&pool)
        .await
        .expect("insert audit attribution");
    }

    for (action_id, operation_id, action_kind, source_record_id) in [
        (
            Uuid::from_u128(20),
            first_operation_id,
            "apply_ban",
            "action:apply-ban",
        ),
        (
            Uuid::from_u128(21),
            second_operation_id,
            "lift_ban",
            "action:lift-ban",
        ),
    ] {
        let mut transaction = pool.begin().await.expect("begin action transaction");
        assume_tenant(&mut transaction, community_id).await;
        sqlx::query(
            "INSERT INTO public.collaboration_moderation_actions (community_id, action_id, operation_id, actor_principal_id, action_kind, record_kind, record_id, target_principal_id, occurred_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1, $2, $3, $4, $5, 'restriction', $6, $6, now(), 'zed', $7, '1', now())",
        )
        .bind(community_id)
        .bind(action_id)
        .bind(operation_id)
        .bind(actor_principal_id)
        .bind(action_kind)
        .bind(target_principal_id)
        .bind(source_record_id)
        .execute(&mut *transaction)
        .await
        .expect("insert attributed action");
        transaction.commit().await.expect("commit action");
    }

    let mut transaction = pool.begin().await.expect("begin first restriction");
    assume_tenant(&mut transaction, community_id).await;
    insert_restriction_version(
        &mut transaction,
        community_id,
        target_principal_id,
        2,
        true,
        "active",
        "apply_ban",
        actor_principal_id,
        first_operation_id,
        "restriction:2",
    )
    .await;
    transaction
        .commit()
        .await
        .expect("commit first restriction");

    let mut transaction = pool.begin().await.expect("begin duplicate current");
    assume_tenant(&mut transaction, community_id).await;
    let duplicate_current = sqlx::query(
        "INSERT INTO public.collaboration_moderation_restriction_versions (community_id, target_principal_id, restriction_version, is_current, ban_state, timeout_state, transition_kind, actor_principal_id, operation_id, occurred_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1, $2, 3, true, 'none', 'none', 'lift_ban', $3, $4, now(), 'zed', 'restriction:3', '1', now())",
    )
    .bind(community_id)
    .bind(target_principal_id)
    .bind(actor_principal_id)
    .bind(second_operation_id)
    .execute(&mut *transaction)
    .await;
    assert!(duplicate_current.is_err());
    transaction
        .rollback()
        .await
        .expect("rollback duplicate current");

    let mut transaction = pool.begin().await.expect("begin version advance");
    assume_tenant(&mut transaction, community_id).await;
    sqlx::query(
        "UPDATE public.collaboration_moderation_restriction_versions SET is_current = false WHERE community_id = $1 AND target_principal_id = $2 AND restriction_version = 2 AND is_current",
    )
    .bind(community_id)
    .bind(target_principal_id)
    .execute(&mut *transaction)
    .await
    .expect("retire prior restriction version");
    insert_restriction_version(
        &mut transaction,
        community_id,
        target_principal_id,
        3,
        true,
        "none",
        "lift_ban",
        actor_principal_id,
        second_operation_id,
        "restriction:3",
    )
    .await;
    transaction.commit().await.expect("commit version advance");

    let mut transaction = pool.begin().await.expect("begin history check");
    assume_tenant(&mut transaction, community_id).await;
    let versions: Vec<(i64, bool, String)> = sqlx::query_as(
        "SELECT restriction_version::bigint, is_current, ban_state FROM public.collaboration_moderation_restriction_versions WHERE community_id = $1 AND target_principal_id = $2 ORDER BY restriction_version",
    )
    .bind(community_id)
    .bind(target_principal_id)
    .fetch_all(&mut *transaction)
    .await
    .expect("load restriction history");
    assert_eq!(
        versions,
        vec![
            (2, false, "active".to_owned()),
            (3, true, "none".to_owned())
        ]
    );
    transaction.commit().await.expect("commit history check");

    let mut transaction = pool.begin().await.expect("begin history mutation");
    assume_tenant(&mut transaction, community_id).await;
    let history_mutation = sqlx::query(
        "UPDATE public.collaboration_moderation_restriction_versions SET actor_principal_id = $3 WHERE community_id = $1 AND target_principal_id = $2 AND restriction_version = 2",
    )
    .bind(community_id)
    .bind(target_principal_id)
    .bind(target_principal_id)
    .execute(&mut *transaction)
    .await;
    assert!(history_mutation.is_err());
    transaction
        .rollback()
        .await
        .expect("rollback history mutation");

    let mut transaction = pool.begin().await.expect("begin foreign tenant read");
    assume_tenant(&mut transaction, foreign_community_id).await;
    for table in [
        "collaboration_moderation_actions",
        "collaboration_moderation_restriction_versions",
    ] {
        let count: i64 = sqlx::query_scalar(&format!(
            "SELECT count(*) FROM public.{table} WHERE community_id = $1"
        ))
        .bind(community_id)
        .fetch_one(&mut *transaction)
        .await
        .expect("query foreign tenant");
        assert_eq!(count, 0, "foreign tenant saw rows in {table}");
    }
    transaction
        .rollback()
        .await
        .expect("rollback foreign tenant read");

    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll moderation migration down");
    for table in TABLES {
        let remaining: Option<String> =
            sqlx::query_scalar(&format!("SELECT to_regclass('public.{table}')::text"))
                .fetch_one(&pool)
                .await
                .expect("query rolled-down table");
        assert_eq!(remaining, None);
    }
    sqlx::raw_sql(
        "DROP OWNED BY collaboration_moderation_request; \
         DROP ROLE collaboration_moderation_request;",
    )
    .execute(&pool)
    .await
    .expect("drop request role");
    sqlx::raw_sql(AUDIT_DOWN)
        .execute(&pool)
        .await
        .expect("roll audit migration down");
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
}

async fn assume_tenant(
    transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    community_id: Uuid,
) {
    sqlx::query("SET LOCAL ROLE collaboration_moderation_request")
        .execute(&mut **transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_id.to_string())
        .execute(&mut **transaction)
        .await
        .expect("set admitted community");
}

#[allow(clippy::too_many_arguments)]
async fn insert_restriction_version(
    transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    community_id: Uuid,
    target_principal_id: Uuid,
    version: i64,
    is_current: bool,
    ban_state: &str,
    transition_kind: &str,
    actor_principal_id: Uuid,
    operation_id: Uuid,
    source_record_id: &str,
) {
    sqlx::query(
        "INSERT INTO public.collaboration_moderation_restriction_versions (community_id, target_principal_id, restriction_version, is_current, ban_state, timeout_state, transition_kind, actor_principal_id, operation_id, occurred_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1, $2, $3, $4, $5, 'none', $6, $7, $8, now(), 'zed', $9, '1', now())",
    )
    .bind(community_id)
    .bind(target_principal_id)
    .bind(version)
    .bind(is_current)
    .bind(ban_state)
    .bind(transition_kind)
    .bind(actor_principal_id)
    .bind(operation_id)
    .bind(source_record_id)
    .execute(&mut **transaction)
    .await
    .expect("insert restriction version");
}
