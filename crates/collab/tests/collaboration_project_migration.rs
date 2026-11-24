use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_822_000_300;
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
    "/migrations/20260822000300_collaboration_projects.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000300_collaboration_projects.down.sql"
));

#[test]
fn collaboration_project_migration_is_tenant_fenced_cross_owner_and_local_state_free() {
    for table in [
        "collaboration_project_groups",
        "collaboration_project_repository_bindings",
        "collaboration_project_channel_bindings",
    ] {
        assert!(UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(UP.contains(&format!("'{table}'")));
        assert!(DOWN.contains(&format!("DROP TABLE public.{table};")));
    }
    for required in [
        "PRIMARY KEY (\n        community_id,\n        project_signer_public_key",
        "UNIQUE (community_id, source_event_id)",
        "FOREIGN KEY (community_id)\n        REFERENCES public.collaboration_communities",
        "FOREIGN KEY (community_id, channel_id)\n        REFERENCES public.collaboration_channels",
        "repository_kind integer NOT NULL DEFAULT 30617 CHECK (repository_kind = 30617)",
        "repository_owner_public_key bytea NOT NULL",
        "binding_state IN ('active', 'deleted')",
        "project_record_version numeric(20, 0)",
        "WHERE is_current",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS PERMISSIVE FOR ALL",
        "AS RESTRICTIVE FOR ALL",
        "current_setting(''app.community_id'', true)",
    ] {
        assert!(UP.contains(required), "missing schema invariant {required}");
    }
    assert!(!UP.contains("repository_owner_public_key = project_signer_public_key"));
    for forbidden in ["worktree_id", "remote_url", "filesystem_path", "local_path"] {
        assert!(
            !UP.contains(forbidden),
            "collaboration schema must not own local state {forbidden}"
        );
    }
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), 3);
}

#[tokio::test]
async fn collaboration_project_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let project_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(project_migrations.len(), 2);
    let up = project_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("project up migration");
    let down = project_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("project down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_project_migration_enforces_tenants_cross_owner_and_binding_deletion() {
    let Some(database_url) = std::env::var("COLLAB_PROJECT_MIGRATION_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_PROJECT_MIGRATION_TEST_DATABASE_URL is unset; live project migration test skipped"
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
        .expect("apply project migration");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_project_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_project_request;",
    )
    .execute(&pool)
    .await
    .expect("create least-privilege request role");

    let community_a = Uuid::from_u128(1);
    let community_b = Uuid::from_u128(2);
    let creator = Uuid::from_u128(3);
    let channel = Uuid::from_u128(4);
    let project_signer = vec![0xaa; 32];
    let repository_owner = vec![0xbb; 32];
    let first_event = vec![0x11_u8; 32];
    let second_event = vec![0x22_u8; 32];

    let mut transaction = pool.begin().await.expect("begin tenant A transaction");
    sqlx::query("SET LOCAL ROLE collaboration_project_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, 'projects-a.example', 'active', 1, 'nostr', 'community:a', '1', now(), now(), now())",
    )
    .bind(community_a)
    .execute(&mut *transaction)
    .await
    .expect("insert tenant A community");
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1, $2, 'owner', 'active', 1, now(), now(), 'zed', 'member:creator', '1', now())",
    )
    .bind(community_a)
    .bind(creator)
    .execute(&mut *transaction)
    .await
    .expect("insert channel creator");
    sqlx::query(
        "INSERT INTO public.collaboration_channels (community_id, channel_id, name, channel_type, visibility, lifecycle_state, creator_principal_id, channel_version, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'project', 'stream', 'private', 'active', $3, 1, 'zed', 'channel:project', '1', now(), now(), now())",
    )
    .bind(community_a)
    .bind(channel)
    .bind(creator)
    .execute(&mut *transaction)
    .await
    .expect("insert project channel");
    sqlx::query(
        "INSERT INTO public.collaboration_project_groups (community_id, project_signer_public_key, project_slug, record_version, is_current, source_event_id, source_created_at, name, visibility, channel_reference, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'multi-owner', 1, true, $3, 100, 'Multi owner', 'listed', 'project-channel', now(), now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(&first_event)
    .execute(&mut *transaction)
    .await
    .expect("insert signed project group");
    sqlx::query(
        "INSERT INTO public.collaboration_project_repository_bindings (community_id, project_signer_public_key, project_slug, repository_owner_public_key, repository_discriminator, binding_version, project_record_version, is_current, binding_state, relay_hint, created_at, updated_at) VALUES ($1, $2, 'multi-owner', $3, 'other/repository', 1, 1, true, 'active', 'wss://relay.example', now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(&repository_owner)
    .execute(&mut *transaction)
    .await
    .expect("bind cross-owner repository");
    sqlx::query(
        "INSERT INTO public.collaboration_project_channel_bindings (community_id, project_signer_public_key, project_slug, binding_version, project_record_version, channel_id, is_current, binding_state, created_at, updated_at) VALUES ($1, $2, 'multi-owner', 1, 1, $3, true, 'active', now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(channel)
    .execute(&mut *transaction)
    .await
    .expect("bind project channel");
    transaction.commit().await.expect("commit active bindings");

    let mut transaction = pool.begin().await.expect("begin deletion transaction");
    sqlx::query("SET LOCAL ROLE collaboration_project_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "UPDATE public.collaboration_project_groups SET is_current = false, updated_at = now() WHERE community_id = $1 AND project_signer_public_key = $2 AND project_slug = 'multi-owner' AND is_current",
    )
    .bind(community_a)
    .bind(&project_signer)
    .execute(&mut *transaction)
    .await
    .expect("retire first project head");
    sqlx::query(
        "INSERT INTO public.collaboration_project_groups (community_id, project_signer_public_key, project_slug, record_version, is_current, source_event_id, source_created_at, name, visibility, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'multi-owner', 2, true, $3, 101, 'Multi owner', 'listed', now(), now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(&second_event)
    .execute(&mut *transaction)
    .await
    .expect("insert replacement project head");
    sqlx::query(
        "UPDATE public.collaboration_project_repository_bindings SET is_current = false, updated_at = now() WHERE community_id = $1 AND project_signer_public_key = $2 AND project_slug = 'multi-owner' AND repository_owner_public_key = $3 AND repository_discriminator = 'other/repository' AND is_current",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(&repository_owner)
    .execute(&mut *transaction)
    .await
    .expect("retire repository binding");
    sqlx::query(
        "INSERT INTO public.collaboration_project_repository_bindings (community_id, project_signer_public_key, project_slug, repository_owner_public_key, repository_discriminator, binding_version, project_record_version, is_current, binding_state, deleted_at, created_at, updated_at) VALUES ($1, $2, 'multi-owner', $3, 'other/repository', 2, 2, true, 'deleted', now(), now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(&repository_owner)
    .execute(&mut *transaction)
    .await
    .expect("insert repository binding tombstone");
    sqlx::query(
        "UPDATE public.collaboration_project_channel_bindings SET is_current = false, updated_at = now() WHERE community_id = $1 AND project_signer_public_key = $2 AND project_slug = 'multi-owner' AND is_current",
    )
    .bind(community_a)
    .bind(&project_signer)
    .execute(&mut *transaction)
    .await
    .expect("retire channel binding");
    sqlx::query(
        "INSERT INTO public.collaboration_project_channel_bindings (community_id, project_signer_public_key, project_slug, binding_version, project_record_version, channel_id, is_current, binding_state, deleted_at, created_at, updated_at) VALUES ($1, $2, 'multi-owner', 2, 2, $3, true, 'deleted', now(), now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(channel)
    .execute(&mut *transaction)
    .await
    .expect("insert channel binding tombstone");
    transaction.commit().await.expect("commit binding deletion");

    let mut transaction = pool.begin().await.expect("begin tenant A read");
    sqlx::query("SET LOCAL ROLE collaboration_project_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    let stored_owner: Vec<u8> = sqlx::query_scalar(
        "SELECT repository_owner_public_key FROM public.collaboration_project_repository_bindings WHERE community_id = $1 AND project_signer_public_key = $2 AND project_slug = 'multi-owner' AND binding_version = 1",
    )
    .bind(community_a)
    .bind(&project_signer)
    .fetch_one(&mut *transaction)
    .await
    .expect("read cross-owner binding");
    assert_eq!(stored_owner, repository_owner);
    assert_ne!(stored_owner, project_signer);
    let repository_state: String = sqlx::query_scalar(
        "SELECT binding_state FROM public.collaboration_project_repository_bindings WHERE community_id = $1 AND project_signer_public_key = $2 AND project_slug = 'multi-owner' AND is_current",
    )
    .bind(community_a)
    .bind(&project_signer)
    .fetch_one(&mut *transaction)
    .await
    .expect("read current repository binding");
    let channel_state: String = sqlx::query_scalar(
        "SELECT binding_state FROM public.collaboration_project_channel_bindings WHERE community_id = $1 AND project_signer_public_key = $2 AND project_slug = 'multi-owner' AND is_current",
    )
    .bind(community_a)
    .bind(&project_signer)
    .fetch_one(&mut *transaction)
    .await
    .expect("read current channel binding");
    assert_eq!(repository_state, "deleted");
    assert_eq!(channel_state, "deleted");
    transaction.commit().await.expect("commit tenant A read");

    let mut transaction = pool.begin().await.expect("begin tenant B transaction");
    sqlx::query("SET LOCAL ROLE collaboration_project_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_b.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant B");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_project_groups WHERE community_id = $1",
    )
    .bind(community_a)
    .fetch_one(&mut *transaction)
    .await
    .expect("query foreign project group");
    assert_eq!(visible, 0);
    let foreign_insert = sqlx::query(
        "INSERT INTO public.collaboration_project_groups (community_id, project_signer_public_key, project_slug, record_version, is_current, source_event_id, source_created_at, visibility, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'foreign', 1, true, $3, 1, 'listed', now(), now(), now())",
    )
    .bind(community_a)
    .bind(&project_signer)
    .bind(vec![0x33_u8; 32])
    .execute(&mut *transaction)
    .await;
    assert!(foreign_insert.is_err());
    transaction.rollback().await.expect("rollback tenant B");

    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll project migration down");
    for table in [
        "collaboration_project_groups",
        "collaboration_project_repository_bindings",
        "collaboration_project_channel_bindings",
    ] {
        let remaining: Option<String> = sqlx::query_scalar("SELECT to_regclass($1)::text")
            .bind(format!("public.{table}"))
            .fetch_one(&pool)
            .await
            .expect("query rolled-down project table");
        assert_eq!(remaining, None);
    }
    sqlx::raw_sql(CHANNELS_DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
}
