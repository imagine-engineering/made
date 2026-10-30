use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::migrate::{Migration, MigrationSource, MigrationType};

const VERSION: i64 = 20_260_820_000_100;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.down.sql"
));

fn migration(migrations: &[Migration], migration_type: MigrationType) -> &Migration {
    migrations
        .iter()
        .find(|migration| {
            migration.version == VERSION && migration.migration_type == migration_type
        })
        .expect("collaboration event migration")
}

#[tokio::test]
async fn collaboration_event_migration_has_checksum_stable_forward_and_rollback_paths() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let event_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(event_migrations.len(), 2);

    let up = migration(&migrations, MigrationType::ReversibleUp);
    let down = migration(&migrations, MigrationType::ReversibleDown);
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
    assert!(up.sql.contains("CREATE TABLE public.collaboration_events"));
    assert!(down.sql.contains("DROP TABLE public.collaboration_events"));
    assert!(
        down.sql
            .contains("DROP FUNCTION public.reject_collaboration_event_update")
    );
}

#[test]
fn collaboration_event_schema_is_partitioned_tenant_fenced_and_bounded() {
    for required in [
        "PRIMARY KEY (community_id, event_id)",
        "PARTITION BY HASH (community_id)",
        "FOR partition_index IN 0..15 LOOP",
        "MODULUS 16, REMAINDER %s",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
        "octet_length(event_id) = 32",
        "octet_length(author_public_key) = 32",
        "event_created_at BETWEEN 0 AND 18446744073709551615",
        "kind BETWEEN 0 AND 65535",
        "octet_length(content) <= 262144",
        "octet_length(canonical_event_bytes) BETWEEN 1 AND 524288",
        "octet_length(signature) = 64",
    ] {
        assert!(UP.contains(required), "missing schema invariant {required}");
    }
    assert_eq!(
        UP.matches("CREATE POLICY collaboration_events_community")
            .count(),
        2,
        "the parent and every generated partition must receive a tenant policy"
    );
}

#[test]
fn collaboration_event_schema_preserves_bytes_and_head_order() {
    for required in [
        "signature_state IN ('verified_live', 'verified_historical')",
        "persistence_class IN ('regular', 'replaceable', 'parameterized_replaceable')",
        "collaboration_events_chronological",
        "collaboration_events_kind_chronological",
        "collaboration_events_author_kind_chronological",
        "collaboration_events_addressable_head",
        "community_id,\n        kind,\n        author_public_key,\n        discriminator,\n        event_created_at DESC,\n        event_id ASC",
        "BEFORE UPDATE ON public.collaboration_events",
        "collaboration event records are immutable",
    ] {
        assert!(UP.contains(required), "missing event invariant {required}");
    }
    assert!(
        !UP.contains("persistence_class IN ('regular', 'ephemeral'"),
        "ephemeral events must not enter authoritative storage"
    );
    assert!(
        !DOWN.contains("CASCADE"),
        "rollback must name exact ownership"
    );
}
