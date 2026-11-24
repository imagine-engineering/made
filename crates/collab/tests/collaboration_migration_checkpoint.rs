use std::path::Path;

use collab::{
    migration::buzz::checkpoint::{
        MigrationCheckpoint, MigrationCheckpointError, MigrationCheckpointRepository,
        MigrationCheckpointStatus, MigrationCheckpointUpdate, MigrationCounts, MigrationCursor,
        MigrationStream, RollbackBoundary,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use sea_orm::{DatabaseBackend, MockDatabase};
use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_820_000_600;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000600_collaboration_migration_checkpoints.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000600_collaboration_migration_checkpoints.down.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "migration-checkpoint")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn checkpoint(community_id: CommunityId) -> MigrationCheckpoint {
    MigrationCheckpoint::new(
        community_id,
        Uuid::from_u128(10),
        "buzz-revision-abc",
        MigrationStream::SignedEvents,
        "events-p0",
        RollbackBoundary::reversible("source-read-only").expect("rollback boundary"),
    )
    .expect("checkpoint")
}

fn update(
    status: MigrationCheckpointStatus,
    sequence: u64,
    token: &[u8],
    counts: MigrationCounts,
    last_error: Option<&str>,
) -> MigrationCheckpointUpdate {
    MigrationCheckpointUpdate {
        status,
        source_cursor: MigrationCursor::new(sequence, Some(token.to_vec())).expect("source cursor"),
        target_cursor: MigrationCursor::new(sequence, Some(token.to_vec())).expect("target cursor"),
        counts,
        source_hash: (counts.scanned > 0).then_some([5; 32]),
        target_hash: (counts.scanned > 0).then_some([6; 32]),
        rollback_boundary: RollbackBoundary::reversible("source-read-only")
            .expect("rollback boundary"),
        last_error: last_error.map(str::to_owned),
    }
}

#[test]
fn migration_checkpoint_interrupts_and_resumes_monotonically() {
    let initial = checkpoint(community(1));
    let counts = MigrationCounts {
        scanned: 10,
        imported: 8,
        skipped: 2,
        failed: 0,
    };
    let running = initial
        .transition(update(
            MigrationCheckpointStatus::Running,
            10,
            b"cursor-10",
            counts,
            None,
        ))
        .expect("start import");
    let interrupted = running
        .transition(update(
            MigrationCheckpointStatus::Interrupted,
            10,
            b"cursor-10",
            counts,
            Some("worker stopped"),
        ))
        .expect("record interruption");
    let resumed = interrupted
        .transition(update(
            MigrationCheckpointStatus::Running,
            10,
            b"cursor-10",
            counts,
            None,
        ))
        .expect("resume from exact cursor");
    assert_eq!(resumed.status(), MigrationCheckpointStatus::Running);
    assert_eq!(resumed.source_cursor().sequence(), 10);
    assert_eq!(resumed.counts(), counts);

    assert!(matches!(
        resumed.transition(update(
            MigrationCheckpointStatus::Running,
            9,
            b"cursor-9",
            counts,
            None,
        )),
        Err(MigrationCheckpointError::ProgressRegression)
    ));
    assert!(matches!(
        resumed.transition(update(
            MigrationCheckpointStatus::Running,
            10,
            b"different-token",
            counts,
            None,
        )),
        Err(MigrationCheckpointError::ProgressRegression)
    ));

    let crossed = running
        .transition(MigrationCheckpointUpdate {
            status: MigrationCheckpointStatus::Interrupted,
            source_cursor: MigrationCursor::new(10, Some(b"cursor-10".to_vec()))
                .expect("source cursor"),
            target_cursor: MigrationCursor::new(10, Some(b"cursor-10".to_vec()))
                .expect("target cursor"),
            counts,
            source_hash: Some([5; 32]),
            target_hash: Some([6; 32]),
            rollback_boundary: RollbackBoundary::irreversible("new-only-write", 100)
                .expect("irreversible boundary"),
            last_error: Some("stopped after boundary".to_owned()),
        })
        .expect("cross irreversible boundary");
    assert!(matches!(
        crossed.transition(MigrationCheckpointUpdate {
            status: MigrationCheckpointStatus::RolledBack,
            source_cursor: MigrationCursor::new(10, Some(b"cursor-10".to_vec()))
                .expect("source cursor"),
            target_cursor: MigrationCursor::new(10, Some(b"cursor-10".to_vec()))
                .expect("target cursor"),
            counts,
            source_hash: Some([5; 32]),
            target_hash: Some([6; 32]),
            rollback_boundary: RollbackBoundary::irreversible("new-only-write", 100)
                .expect("irreversible boundary"),
            last_error: None,
        }),
        Err(MigrationCheckpointError::ProgressRegression)
    ));
}

#[tokio::test]
async fn migration_checkpoint_rejects_cross_tenant_reuse_before_database_io() {
    let checkpoint = checkpoint(community(1));
    let repository = MigrationCheckpointRepository::new(
        MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
    )
    .expect("repository");
    let result = repository
        .save_transition(
            &tenant(community(2)),
            &checkpoint,
            update(
                MigrationCheckpointStatus::Running,
                0,
                b"",
                MigrationCounts::default(),
                None,
            ),
        )
        .await;
    assert!(matches!(
        result,
        Err(MigrationCheckpointError::TenantBoundaryViolation)
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}

#[test]
fn migration_checkpoint_schema_fences_progress_tenant_and_rollback() {
    for required in [
        "run_id uuid PRIMARY KEY",
        "UNIQUE (community_id, run_id)",
        "PRIMARY KEY (community_id, run_id, stream_name, shard_id)",
        "checkpoint_version <> OLD.checkpoint_version + 1",
        "migration checkpoint progress cannot regress",
        "migration checkpoint cursor token changed without progress",
        "migration checkpoint integrity changed without progress",
        "migration rollback boundary cannot become reversible",
        "migration irreversible boundary is immutable",
        "migration checkpoint status transition is invalid",
        "imported_count + skipped_count + failed_count <= scanned_count",
        "source_hash IS NULL OR octet_length(source_hash) = 32",
        "target_hash IS NULL OR octet_length(target_hash) = 32",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(
            UP.contains(required),
            "missing checkpoint invariant {required}"
        );
    }
    assert!(DOWN.starts_with("DROP TABLE public.collaboration_migration_checkpoints;"));
    assert!(!DOWN.contains("CASCADE"));
}

#[tokio::test]
async fn migration_checkpoint_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let checkpoint_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(checkpoint_migrations.len(), 2);
    let up = checkpoint_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("checkpoint up migration");
    let down = checkpoint_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("checkpoint down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn migration_checkpoint_persists_interruption_resume_and_version_fence() {
    let Some(database_url) = std::env::var("COLLAB_MIGRATION_CHECKPOINT_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_MIGRATION_CHECKPOINT_TEST_DATABASE_URL is unset; live checkpoint test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply checkpoint migration");
    let community_id = community(1);
    let tenant = tenant(community_id);
    let initial = checkpoint(community_id);
    let repository = MigrationCheckpointRepository::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("connect repository"),
    )
    .expect("repository");
    repository
        .create(&tenant, &initial)
        .await
        .expect("create checkpoint");
    assert_eq!(
        repository
            .load(
                &tenant,
                initial.run_id(),
                initial.stream(),
                initial.shard_id(),
            )
            .await
            .expect("load initial"),
        Some(initial.clone())
    );

    let counts = MigrationCounts {
        scanned: 10,
        imported: 10,
        skipped: 0,
        failed: 0,
    };
    let running = repository
        .save_transition(
            &tenant,
            &initial,
            update(
                MigrationCheckpointStatus::Running,
                10,
                b"cursor-10",
                counts,
                None,
            ),
        )
        .await
        .expect("persist running");
    let interrupted = repository
        .save_transition(
            &tenant,
            &running,
            update(
                MigrationCheckpointStatus::Interrupted,
                10,
                b"cursor-10",
                counts,
                Some("worker stopped"),
            ),
        )
        .await
        .expect("persist interruption");

    let resumed_repository = MigrationCheckpointRepository::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("reconnect repository"),
    )
    .expect("resumed repository");
    let loaded = resumed_repository
        .load(
            &tenant,
            interrupted.run_id(),
            interrupted.stream(),
            interrupted.shard_id(),
        )
        .await
        .expect("load interruption")
        .expect("checkpoint exists");
    assert_eq!(loaded, interrupted);
    let resumed = resumed_repository
        .save_transition(
            &tenant,
            &loaded,
            update(
                MigrationCheckpointStatus::Running,
                10,
                b"cursor-10",
                counts,
                None,
            ),
        )
        .await
        .expect("persist resume");
    assert_eq!(
        resumed.source_cursor().token(),
        Some(b"cursor-10".as_slice())
    );

    let stale_write = resumed_repository
        .save_transition(
            &tenant,
            &loaded,
            update(
                MigrationCheckpointStatus::Running,
                10,
                b"cursor-10",
                counts,
                None,
            ),
        )
        .await;
    assert!(matches!(
        stale_write,
        Err(MigrationCheckpointError::VersionConflict)
    ));
    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll back checkpoint migration");
    pool.close().await;
}
