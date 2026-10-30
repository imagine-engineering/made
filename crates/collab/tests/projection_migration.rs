use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::migrate::{MigrationSource, MigrationType};

const VERSION: i64 = 20_260_820_000_300;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000300_collaboration_projections.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000300_collaboration_projections.down.sql"
));

#[tokio::test]
async fn projection_migration_has_checksum_stable_forward_and_rollback_paths() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let projection_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(projection_migrations.len(), 2);
    let up = projection_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("projection up migration");
    let down = projection_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("projection down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
    assert!(
        down.sql
            .contains("DROP TABLE public.collaboration_projection_checkpoints;")
    );
    assert!(
        down.sql
            .contains("DROP FUNCTION public.guard_collaboration_projection_checkpoint_update();")
    );
    assert!(!down.sql.contains("CASCADE"));
}

#[test]
fn projection_migration_preserves_provenance_resume_and_version_fences() {
    for required in [
        "PRIMARY KEY (community_id, projection_name, source_system, source_record_id)",
        "source_system IN ('zed', 'buzz', 'nostr', 'acp', 'external_git')",
        "octet_length(source_record_id) BETWEEN 1 AND 1024",
        "source_version text",
        "source_observed_at timestamptz NOT NULL",
        "source_integrity_algorithm",
        "(source_integrity_algorithm IS NULL) = (source_integrity_value IS NULL)",
        "projection_version BETWEEN 1 AND 18446744073709551615",
        "NEW.projection_version <> OLD.projection_version + 1",
        "projection checkpoint version conflict",
        "ERRCODE = 'serialization_failure'",
        "reset_generation BETWEEN 1 AND 18446744073709551615",
        "cursor bytea CHECK (cursor IS NULL OR octet_length(cursor) <= 65536)",
        "projected_at timestamptz NOT NULL",
    ] {
        assert!(
            UP.contains(required),
            "missing checkpoint invariant {required}"
        );
    }
}

#[test]
fn projection_migration_scopes_drift_and_reset_to_one_tenant() {
    for required in [
        "drift_state IN ('clean', 'suspect', 'diverged', 'rebuilding', 'reset_pending')",
        "authoritative_hash IS NOT NULL",
        "projection_hash IS NOT NULL",
        "authoritative_hash <> projection_hash",
        "drift_state <> 'reset_pending' OR cursor IS NULL",
        "(reset_at IS NULL) = (reset_generation = 1)",
        "NEW.reset_generation <> OLD.reset_generation + 1",
        "NEW.drift_state <> 'reset_pending'",
        "projection reset must fence stale work and clear its cursor",
        "collaboration_projection_checkpoints_scan",
        "collaboration_projection_checkpoints_source",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(
            UP.contains(required),
            "missing drift/reset invariant {required}"
        );
    }
    for index in [
        "collaboration_projection_checkpoints_scan",
        "collaboration_projection_checkpoints_source",
    ] {
        let start = UP.find(index).expect("index exists");
        let tail = &UP[start..];
        assert!(
            tail.find("community_id").is_some_and(|offset| offset < 160),
            "index {index} must lead with community_id"
        );
    }
}
