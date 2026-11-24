use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_822_000_400;
const CHANNELS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const CHANNELS_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.down.sql"
));
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000400_collaboration_git.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000400_collaboration_git.down.sql"
));

#[test]
fn collaboration_git_schema_has_one_authority_and_explicit_grants() {
    for table in [
        "collaboration_hosted_repositories",
        "collaboration_git_storage_handles",
        "collaboration_git_repository_grants",
    ] {
        assert!(UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(UP.contains(&format!("'{table}'")));
        assert!(DOWN.contains(&format!("DROP TABLE public.{table};")));
    }
    for required in [
        "authority_kind IN ('sim_hosted_nip34', 'external_provider')",
        "authority_version numeric(20, 0)",
        "authority_kind = 'sim_hosted_nip34'",
        "permission IN ('read', 'write', 'admin')",
        "PRIMARY KEY (\n        community_id, repository_id, grantee_principal_id, permission",
        "FOREIGN KEY (community_id, grantee_principal_id)",
        "FOREIGN KEY (community_id, granted_by_principal_id)",
        "REFERENCES public.collaboration_community_memberships",
    ] {
        assert!(
            UP.contains(required),
            "missing authority invariant {required}"
        );
    }
    for forbidden in [
        "project_signer_public_key",
        "channel_id",
        "remote_url",
        "filesystem_path",
        "local_path",
        "credential",
        "secret",
    ] {
        assert!(
            !UP.contains(forbidden),
            "hosted authority must not acquire {forbidden}"
        );
    }
}

#[test]
fn collaboration_git_schema_is_tenant_fenced_archivable_and_reversible() {
    for required in [
        "REFERENCES public.collaboration_communities (community_id)",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE FOR ALL",
        "AS RESTRICTIVE FOR ALL",
        "current_setting(''app.community_id'', true)",
        "lifecycle_state IN ('active', 'archived')",
        "grant_state IN ('active', 'revoked')",
        "lifecycle_state = 'archived' AND archived_at IS NOT NULL",
        "grant_state = 'revoked' AND revoked_at IS NOT NULL",
    ] {
        assert!(
            UP.contains(required),
            "missing lifecycle invariant {required}"
        );
    }
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), 3);
}

#[tokio::test]
async fn collaboration_git_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let git_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(git_migrations.len(), 2);
    let up = git_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("Git up migration");
    let down = git_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("Git down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_git_schema_enforces_tenants_grants_archive_and_rollback() {
    let Some(database_url) = std::env::var("COLLAB_GIT_MIGRATION_TEST_DATABASE_URL").ok() else {
        eprintln!(
            "COLLAB_GIT_MIGRATION_TEST_DATABASE_URL is unset; live Git migration test skipped"
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
        .expect("apply Git migration");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_git_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_git_request;",
    )
    .execute(&pool)
    .await
    .expect("create least-privilege request role");

    let community_a = Uuid::from_u128(1);
    let community_b = Uuid::from_u128(2);
    let repository_id = Uuid::from_u128(3);
    let storage_handle_id = Uuid::from_u128(4);
    let owner_principal_id = Uuid::from_u128(5);
    let writer_principal_id = Uuid::from_u128(6);
    let repository_owner = vec![0xaa_u8; 32];

    for (community_id, host) in [
        (community_a, "git-a.example"),
        (community_b, "git-b.example"),
    ] {
        sqlx::query(
            "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'active', 1, 'zed', $2, now(), now(), now())",
        )
        .bind(community_id)
        .bind(host)
        .execute(&pool)
        .await
        .expect("insert community");
    }
    for principal_id in [owner_principal_id, writer_principal_id] {
        sqlx::query(
            "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, 'member', 'active', 1, now(), now(), 'zed', $2::text, now())",
        )
        .bind(community_a)
        .bind(principal_id)
        .execute(&pool)
        .await
        .expect("insert community member");
    }

    let mut transaction = pool.begin().await.expect("begin tenant A transaction");
    sqlx::query("SET LOCAL ROLE collaboration_git_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "INSERT INTO public.collaboration_hosted_repositories (community_id, repository_id, repository_owner_public_key, repository_discriminator, authority_kind, authority_version, lifecycle_state, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 'alpha', 'sim_hosted_nip34', 1, 'active', 'buzz', '30617:alpha', now(), now(), now())",
    )
    .bind(community_a)
    .bind(repository_id)
    .bind(&repository_owner)
    .execute(&mut *transaction)
    .await
    .expect("insert hosted repository");
    sqlx::query(
        "INSERT INTO public.collaboration_git_storage_handles (community_id, storage_handle_id, repository_id, handle_version, lifecycle_state, created_at, updated_at) VALUES ($1, $2, $3, 1, 'active', now(), now())",
    )
    .bind(community_a)
    .bind(storage_handle_id)
    .bind(repository_id)
    .execute(&mut *transaction)
    .await
    .expect("insert opaque storage handle");
    for permission in ["read", "write", "admin"] {
        sqlx::query(
            "INSERT INTO public.collaboration_git_repository_grants (community_id, repository_id, grantee_principal_id, permission, grant_version, grant_state, granted_by_principal_id, created_at, updated_at) VALUES ($1, $2, $3, $4, 1, 'active', $5, now(), now())",
        )
        .bind(community_a)
        .bind(repository_id)
        .bind(writer_principal_id)
        .bind(permission)
        .bind(owner_principal_id)
        .execute(&mut *transaction)
        .await
        .expect("insert explicit permission");
    }
    let duplicate = sqlx::query(
        "INSERT INTO public.collaboration_git_repository_grants (community_id, repository_id, grantee_principal_id, permission, grant_version, grant_state, granted_by_principal_id, created_at, updated_at) VALUES ($1, $2, $3, 'write', 1, 'active', $4, now(), now())",
    )
    .bind(community_a)
    .bind(repository_id)
    .bind(writer_principal_id)
    .bind(owner_principal_id)
    .execute(&mut *transaction)
    .await;
    assert!(duplicate.is_err(), "duplicate exact grants must fail");
    transaction
        .rollback()
        .await
        .expect("clear failed transaction");

    let mut transaction = pool
        .begin()
        .await
        .expect("begin committed tenant A transaction");
    sqlx::query("SET LOCAL ROLE collaboration_git_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "INSERT INTO public.collaboration_hosted_repositories (community_id, repository_id, repository_owner_public_key, repository_discriminator, authority_kind, authority_version, lifecycle_state, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 'alpha', 'sim_hosted_nip34', 1, 'active', 'buzz', '30617:alpha', now(), now(), now())",
    )
    .bind(community_a)
    .bind(repository_id)
    .bind(&repository_owner)
    .execute(&mut *transaction)
    .await
    .expect("insert hosted repository");
    sqlx::query(
        "INSERT INTO public.collaboration_git_storage_handles (community_id, storage_handle_id, repository_id, handle_version, lifecycle_state, created_at, updated_at) VALUES ($1, $2, $3, 1, 'active', now(), now())",
    )
    .bind(community_a)
    .bind(storage_handle_id)
    .bind(repository_id)
    .execute(&mut *transaction)
    .await
    .expect("insert opaque storage handle");
    for permission in ["read", "write", "admin"] {
        sqlx::query(
            "INSERT INTO public.collaboration_git_repository_grants (community_id, repository_id, grantee_principal_id, permission, grant_version, grant_state, granted_by_principal_id, created_at, updated_at) VALUES ($1, $2, $3, $4, 1, 'active', $5, now(), now())",
        )
        .bind(community_a)
        .bind(repository_id)
        .bind(writer_principal_id)
        .bind(permission)
        .bind(owner_principal_id)
        .execute(&mut *transaction)
        .await
        .expect("insert explicit permission");
    }
    transaction.commit().await.expect("commit tenant A state");

    let invalid_archive = sqlx::query(
        "UPDATE public.collaboration_hosted_repositories SET lifecycle_state = 'archived', updated_at = now() WHERE community_id = $1 AND repository_id = $2",
    )
    .bind(community_a)
    .bind(repository_id)
    .execute(&pool)
    .await;
    assert!(invalid_archive.is_err(), "archive requires archived_at");
    sqlx::query(
        "UPDATE public.collaboration_git_repository_grants SET grant_state = 'revoked', grant_version = 2, revoked_at = now(), updated_at = now() WHERE community_id = $1 AND repository_id = $2",
    )
    .bind(community_a)
    .bind(repository_id)
    .execute(&pool)
    .await
    .expect("revoke grants");
    sqlx::query(
        "UPDATE public.collaboration_git_storage_handles SET lifecycle_state = 'archived', handle_version = 2, archived_at = now(), updated_at = now() WHERE community_id = $1 AND repository_id = $2",
    )
    .bind(community_a)
    .bind(repository_id)
    .execute(&pool)
    .await
    .expect("archive storage handle");
    sqlx::query(
        "UPDATE public.collaboration_hosted_repositories SET lifecycle_state = 'archived', authority_version = 2, archived_at = now(), updated_at = now() WHERE community_id = $1 AND repository_id = $2",
    )
    .bind(community_a)
    .bind(repository_id)
    .execute(&pool)
    .await
    .expect("archive hosted repository");

    let mut transaction = pool.begin().await.expect("begin tenant B transaction");
    sqlx::query("SET LOCAL ROLE collaboration_git_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_b.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant B");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_hosted_repositories WHERE community_id = $1",
    )
    .bind(community_a)
    .fetch_one(&mut *transaction)
    .await
    .expect("query foreign repository");
    assert_eq!(visible, 0);
    let foreign_insert = sqlx::query(
        "INSERT INTO public.collaboration_hosted_repositories (community_id, repository_id, repository_owner_public_key, repository_discriminator, authority_kind, authority_version, lifecycle_state, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 'foreign', 'sim_hosted_nip34', 1, 'active', 'zed', 'foreign', now(), now(), now())",
    )
    .bind(community_a)
    .bind(Uuid::from_u128(9))
    .bind(&repository_owner)
    .execute(&mut *transaction)
    .await;
    assert!(foreign_insert.is_err());
    transaction.rollback().await.expect("rollback tenant B");

    sqlx::raw_sql("DROP OWNED BY collaboration_git_request; DROP ROLE collaboration_git_request;")
        .execute(&pool)
        .await
        .expect("remove request role");
    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll Git migration down");
    for table in [
        "collaboration_hosted_repositories",
        "collaboration_git_storage_handles",
        "collaboration_git_repository_grants",
    ] {
        let remaining: Option<String> = sqlx::query_scalar("SELECT to_regclass($1)::text")
            .bind(format!("public.{table}"))
            .fetch_one(&pool)
            .await
            .expect("query rolled-down Git table");
        assert_eq!(remaining, None);
    }
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
}
