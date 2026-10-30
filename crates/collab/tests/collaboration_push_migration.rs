use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_822_000_200;
const EVENTS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const CHANNELS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000200_collaboration_push.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000200_collaboration_push.down.sql"
));

#[test]
fn collaboration_push_schema_encrypts_active_authority_and_carries_no_wake_payload() {
    for required in [
        "capability_ciphertext bytea",
        "subscription_policy_ciphertext bytea",
        "custody_key_id text",
        "capability_reference bytea",
        "capability_ciphertext IS NULL",
        "subscription_policy_ciphertext IS NULL",
        "custody_key_id IS NULL",
        "octet_length(capability_ciphertext) BETWEEN 1 AND 16384",
        "octet_length(subscription_policy_ciphertext) BETWEEN 1 AND 1048576",
    ] {
        assert!(
            UP.contains(required),
            "missing encryption invariant {required}"
        );
    }
    let wake_schema = UP
        .split_once("CREATE TABLE public.collaboration_push_wake_jobs")
        .map(|(_, schema)| schema)
        .expect("wake-job schema");
    for forbidden in [
        "payload ",
        "content ",
        "title ",
        "body ",
        "subtitle ",
        "deep_link",
        "url ",
        "ciphertext",
    ] {
        assert!(
            !wake_schema.contains(forbidden),
            "wake job must not carry {forbidden}"
        );
    }
}

#[test]
fn collaboration_push_schema_fences_generations_and_deduplicates_wakes() {
    for required in [
        "PRIMARY KEY (community_id, owner_principal_id, installation_id)",
        "UNIQUE (community_id, owner_principal_id, installation_id, generation)",
        "generation BETWEEN 1 AND 9007199254740991",
        "lease_generation BETWEEN 1 AND 9007199254740991",
        "endpoint_generation BETWEEN 1 AND 9007199254740991",
        "UNIQUE (community_id, request_id)",
        "UNIQUE (community_id, capability_reference, source_event_id)",
        "FOREIGN KEY (community_id, owner_principal_id, installation_id)",
        "state IN ('pending', 'leased', 'delivered', 'failed', 'suppressed')",
    ] {
        assert!(UP.contains(required), "missing queue invariant {required}");
    }
}

#[test]
fn collaboration_push_schema_is_tenant_fenced_and_exactly_reversible() {
    for table in ["collaboration_push_leases", "collaboration_push_wake_jobs"] {
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
        "REFERENCES public.collaboration_community_memberships",
        "REFERENCES public.collaboration_events",
    ] {
        assert!(UP.contains(required), "missing tenant invariant {required}");
    }
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), 2);
}

#[tokio::test]
async fn collaboration_push_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let push_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(push_migrations.len(), 2);
    let up = push_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("push up migration");
    let down = push_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("push down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_push_schema_enforces_live_tenant_and_idempotency_constraints() {
    let Some(database_url) = std::env::var("COLLAB_PUSH_MIGRATION_TEST_DATABASE_URL").ok() else {
        eprintln!(
            "COLLAB_PUSH_MIGRATION_TEST_DATABASE_URL is unset; live push migration test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(EVENTS_UP)
        .execute(&pool)
        .await
        .expect("apply event migration");
    sqlx::raw_sql(CHANNELS_UP)
        .execute(&pool)
        .await
        .expect("apply channel migration");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply push migration");

    let community_id = Uuid::from_u128(1);
    let owner_principal_id = Uuid::from_u128(2);
    let event_id = vec![3_u8; 32];
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'push.example', 'active', 1, 'buzz', 'community:push', now(), now(), now())",
    )
    .bind(community_id)
    .execute(&pool)
    .await
    .expect("insert community");
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, 'member', 'active', 1, now(), now(), 'buzz', 'membership:push', now())",
    )
    .bind(community_id)
    .bind(owner_principal_id)
    .execute(&pool)
    .await
    .expect("insert owner membership");
    sqlx::query(
        "INSERT INTO public.collaboration_events (community_id, event_id, author_public_key, event_created_at, kind, tags, content, canonical_event_bytes, signature, signature_state, verified_at, persistence_class, discriminator) VALUES ($1, $2, $3, 1900000000, 30350, '[]'::jsonb, 'encrypted', $4, $5, 'verified_historical', now(), 'parameterized_replaceable', 'installation-one')",
    )
    .bind(community_id)
    .bind(&event_id)
    .bind(vec![4_u8; 32])
    .bind(b"encrypted".as_slice())
    .bind(vec![5_u8; 64])
    .execute(&pool)
    .await
    .expect("insert push lease event");
    sqlx::query(
        "INSERT INTO public.collaboration_push_leases (community_id, owner_principal_id, installation_id, source_event_id, source_created_at, generation, active, expires_at, last_active_expires_at, endpoint_generation, capability_reference, capability_ciphertext, subscription_policy_ciphertext, custody_key_id, endpoint_enabled, accepted_at, updated_at) VALUES ($1, $2, 'installation-one', $3, 1900000000, 1, true, now() + interval '1 day', now() + interval '1 day', 1, $4, $5, $6, 'push-key-1', true, now(), now())",
    )
    .bind(community_id)
    .bind(owner_principal_id)
    .bind(&event_id)
    .bind(vec![6_u8; 32])
    .bind(vec![7_u8; 64])
    .bind(vec![8_u8; 64])
    .execute(&pool)
    .await
    .expect("insert encrypted active lease");

    let insert_wake = |wake_id: Uuid| {
        sqlx::query(
            "INSERT INTO public.collaboration_push_wake_jobs (community_id, wake_id, request_id, owner_principal_id, installation_id, lease_generation, endpoint_generation, capability_reference, source_event_id, expires_at) VALUES ($1, $2, $2, $3, 'installation-one', 1, 1, $4, $5, now() + interval '1 hour')",
        )
        .bind(community_id)
        .bind(wake_id)
        .bind(owner_principal_id)
        .bind(vec![6_u8; 32])
        .bind(&event_id)
    };
    insert_wake(Uuid::from_u128(10))
        .execute(&pool)
        .await
        .expect("insert wake");
    assert!(
        insert_wake(Uuid::from_u128(11))
            .execute(&pool)
            .await
            .is_err(),
        "the same capability/event pair must be idempotent"
    );

    sqlx::raw_sql(
        "CREATE ROLE collaboration_push_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON public.collaboration_push_leases, \
         public.collaboration_push_wake_jobs TO collaboration_push_request;",
    )
    .execute(&pool)
    .await
    .expect("create request role");
    let mut transaction = pool
        .begin()
        .await
        .expect("begin foreign tenant transaction");
    sqlx::query("SET LOCAL ROLE collaboration_push_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(Uuid::from_u128(99).to_string())
        .execute(&mut *transaction)
        .await
        .expect("set foreign tenant");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_push_leases WHERE community_id = $1",
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
        .expect("roll push migration down");
    let remaining: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.collaboration_push_leases')::text")
            .fetch_one(&pool)
            .await
            .expect("query rolled-down table");
    assert_eq!(remaining, None);
    pool.close().await;
}
