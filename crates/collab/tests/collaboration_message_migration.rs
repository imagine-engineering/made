use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_820_000_800;
const EVENT_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const EVENT_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.down.sql"
));
const CHANNEL_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const CHANNEL_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.down.sql"
));
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000800_collaboration_messages.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000800_collaboration_messages.down.sql"
));

#[test]
fn collaboration_message_schema_preserves_order_tombstones_and_provenance() {
    for table in [
        "collaboration_messages",
        "collaboration_message_auxiliary_events",
    ] {
        assert!(UP.contains(&format!("CREATE TABLE public.{table}")));
        assert!(UP.contains(&format!(
            "ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"
        )));
        assert!(UP.contains(&format!(
            "ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY"
        )));
        assert!(UP.contains(&format!("{table}_provenance")));
        assert!(DOWN.contains(&format!("DROP TABLE public.{table};")));
    }
    for invariant in [
        "message_created_at DESC,\n        source_event_id ASC",
        "event_created_at ASC,\n        auxiliary_event_id ASC",
        "UNIQUE (community_id, source_event_id)",
        "PRIMARY KEY (community_id, auxiliary_event_id)",
        "WHERE lifecycle_state = 'deleted'",
        "WHERE is_tombstone",
        "'edit'",
        "'delete'",
        "'reaction_add'",
        "'reaction_remove'",
        "'pin'",
        "'unpin'",
        "'bookmark'",
        "'unbookmark'",
        "'schedule'",
        "'schedule_cancel'",
        "'schedule_publish'",
        "octet_length(emoji) BETWEEN 1 AND 4096",
        "source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')",
        "source_record_id",
        "source_version",
        "source_observed_at",
        "AS PERMISSIVE",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(
            UP.contains(invariant),
            "missing schema invariant {invariant}"
        );
    }
    assert!(!UP.contains("content text"));
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), 2);
}

#[tokio::test]
async fn collaboration_message_schema_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let message_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(message_migrations.len(), 2);
    let up = message_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("message up migration");
    let down = message_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("message down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_message_schema_enforces_live_order_uniqueness_and_tenant_fences() {
    let Some(database_url) = std::env::var("COLLAB_MESSAGE_MIGRATION_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_MESSAGE_MIGRATION_TEST_DATABASE_URL is unset; live message migration test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(EVENT_UP)
        .execute(&pool)
        .await
        .expect("apply event migration");
    sqlx::raw_sql(CHANNEL_UP)
        .execute(&pool)
        .await
        .expect("apply channel migration");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply message migration");
    sqlx::raw_sql(
        "CREATE ROLE collaboration_message_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public \
         TO collaboration_message_request;",
    )
    .execute(&pool)
    .await
    .expect("create least-privilege request role");

    let community_a = Uuid::from_u128(1);
    let community_b = Uuid::from_u128(2);
    let principal = Uuid::from_u128(3);
    let channel = Uuid::from_u128(4);
    let source_events = [
        vec![1_u8; 32],
        vec![2_u8; 32],
        vec![3_u8; 32],
        vec![4_u8; 32],
    ];
    let mut transaction = pool.begin().await.expect("begin tenant A setup");
    sqlx::query("SET LOCAL ROLE collaboration_message_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    sqlx::query(
        "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, 'a.example', 'active', 1, 'buzz', 'communities:1', now(), now(), now())",
    )
    .bind(community_a)
    .execute(&mut *transaction)
    .await
    .expect("insert community");
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_observed_at) VALUES ($1, $2, 'owner', 'active', 1, now(), now(), 'buzz', 'members:1', now())",
    )
    .bind(community_a)
    .bind(principal)
    .execute(&mut *transaction)
    .await
    .expect("insert member");
    sqlx::query(
        "INSERT INTO public.collaboration_channels (community_id, channel_id, name, channel_type, visibility, lifecycle_state, creator_principal_id, channel_version, source_system, source_record_id, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'general', 'stream', 'open', 'active', $3, 1, 'buzz', 'channels:1', now(), now(), now())",
    )
    .bind(community_a)
    .bind(channel)
    .bind(principal)
    .execute(&mut *transaction)
    .await
    .expect("insert channel");
    for (index, event_id) in source_events.iter().enumerate() {
        sqlx::query(
            "INSERT INTO public.collaboration_events (community_id, event_id, author_public_key, event_created_at, kind, tags, content, canonical_event_bytes, signature, signature_state, verified_at, persistence_class) VALUES ($1, $2, $3, 100, $4, '[]', '', $5, $6, 'verified_historical', now(), 'regular')",
        )
        .bind(community_a)
        .bind(event_id)
        .bind(vec![9_u8; 32])
        .bind(if index < 2 { 9_i32 } else { 5_i32 })
        .bind(vec![index as u8 + 1])
        .bind(vec![index as u8 + 1; 64])
        .execute(&mut *transaction)
        .await
        .expect("insert signed event authority");
    }
    for (index, event_id) in source_events[..2].iter().enumerate() {
        sqlx::query(
            "INSERT INTO public.collaboration_messages (community_id, message_id, channel_id, source_event_id, current_event_id, author_principal_id, message_created_at, lifecycle_state, message_version, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, $4, $4, $5, 100, 'active', 1, 'buzz', $6, now())",
        )
        .bind(community_a)
        .bind(Uuid::from_u128(index as u128 + 10))
        .bind(channel)
        .bind(event_id)
        .bind(principal)
        .bind(format!("messages:{}", index + 1))
        .execute(&mut *transaction)
        .await
        .expect("insert message projection");
    }
    sqlx::query(
        "INSERT INTO public.collaboration_message_auxiliary_events (community_id, auxiliary_event_id, channel_id, target_message_event_id, actor_principal_id, auxiliary_kind, event_created_at, is_tombstone, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, $4, $5, 'delete', 101, true, 'buzz', 'deletes:missing-target', now())",
    )
    .bind(community_a)
    .bind(&source_events[2])
    .bind(channel)
    .bind(vec![8_u8; 32])
    .bind(principal)
    .execute(&mut *transaction)
    .await
    .expect("insert out-of-order tombstone");
    transaction.commit().await.expect("commit tenant A setup");

    let mut transaction = pool.begin().await.expect("begin tenant A read");
    sqlx::query("SET LOCAL ROLE collaboration_message_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    let ordered: Vec<Vec<u8>> = sqlx::query_scalar(
        "SELECT source_event_id FROM public.collaboration_messages WHERE community_id = $1 AND channel_id = $2 ORDER BY message_created_at DESC, source_event_id ASC",
    )
    .bind(community_a)
    .bind(channel)
    .fetch_all(&mut *transaction)
    .await
    .expect("read same-second window");
    assert_eq!(ordered, source_events[..2]);
    let tombstones: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_message_auxiliary_events WHERE community_id = $1 AND is_tombstone",
    )
    .bind(community_a)
    .fetch_one(&mut *transaction)
    .await
    .expect("count tombstones");
    assert_eq!(tombstones, 1);
    transaction.commit().await.expect("commit tenant A read");

    let mut transaction = pool.begin().await.expect("begin duplicate check");
    sqlx::query("SET LOCAL ROLE collaboration_message_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    let duplicate = sqlx::query(
        "INSERT INTO public.collaboration_messages (community_id, message_id, channel_id, source_event_id, current_event_id, author_principal_id, message_created_at, lifecycle_state, message_version, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, $4, $4, $5, 100, 'active', 1, 'buzz', 'messages:duplicate', now())",
    )
    .bind(community_a)
    .bind(Uuid::from_u128(99))
    .bind(channel)
    .bind(&source_events[0])
    .bind(principal)
    .execute(&mut *transaction)
    .await;
    assert!(duplicate.is_err());
    transaction.rollback().await.expect("rollback duplicate");

    let mut transaction = pool.begin().await.expect("begin tombstone check");
    sqlx::query("SET LOCAL ROLE collaboration_message_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_a.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant A");
    let false_tombstone = sqlx::query(
        "INSERT INTO public.collaboration_message_auxiliary_events (community_id, auxiliary_event_id, channel_id, target_message_event_id, actor_principal_id, auxiliary_kind, emoji, event_created_at, is_tombstone, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, $4, $5, 'reaction_add', '+1', 102, true, 'buzz', 'reactions:false-tombstone', now())",
    )
    .bind(community_a)
    .bind(&source_events[3])
    .bind(channel)
    .bind(&source_events[0])
    .bind(principal)
    .execute(&mut *transaction)
    .await;
    assert!(false_tombstone.is_err());
    transaction
        .rollback()
        .await
        .expect("rollback false tombstone");

    let mut transaction = pool.begin().await.expect("begin tenant B check");
    sqlx::query("SET LOCAL ROLE collaboration_message_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(community_b.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set tenant B");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_messages WHERE community_id = $1",
    )
    .bind(community_a)
    .fetch_one(&mut *transaction)
    .await
    .expect("query foreign tenant");
    assert_eq!(visible, 0);
    let foreign_insert = sqlx::query(
        "INSERT INTO public.collaboration_messages (community_id, message_id, channel_id, source_event_id, current_event_id, author_principal_id, message_created_at, lifecycle_state, message_version, source_system, source_record_id, source_observed_at) VALUES ($1, $2, $3, $4, $4, $5, 100, 'active', 1, 'buzz', 'messages:foreign', now())",
    )
    .bind(community_a)
    .bind(Uuid::from_u128(100))
    .bind(channel)
    .bind(&source_events[1])
    .bind(principal)
    .execute(&mut *transaction)
    .await;
    assert!(foreign_insert.is_err());
    transaction
        .rollback()
        .await
        .expect("rollback tenant B check");

    sqlx::raw_sql(
        "DROP OWNED BY collaboration_message_request; DROP ROLE collaboration_message_request;",
    )
    .execute(&pool)
    .await
    .expect("drop request role");
    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll message migration down");
    sqlx::raw_sql(CHANNEL_DOWN)
        .execute(&pool)
        .await
        .expect("roll channel migration down");
    sqlx::raw_sql(EVENT_DOWN)
        .execute(&pool)
        .await
        .expect("roll event migration down");
    let remaining: Option<String> =
        sqlx::query_scalar("SELECT to_regclass('public.collaboration_messages')::text")
            .fetch_one(&pool)
            .await
            .expect("query rolled-down table");
    assert_eq!(remaining, None);
}
