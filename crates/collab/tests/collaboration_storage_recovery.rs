use async_trait::async_trait;
use collab::{
    collaboration_command::{
        CommandAdapter, DomainCommand, DomainCommandDisposition, DomainCommandSink,
        DomainCommandSubmissionError,
    },
    db::collaboration::{
        outbox::{
            AppliedCommand, CommandFingerprint, OutboxOperation, TransactionalCommandMutation,
            TransactionalCommandOutbox,
        },
        rebuild::{
            ProjectionDriftState, ProjectionMaterialization, ProjectionRebuildAdapter,
            ProjectionRebuildError, ProjectionRebuilder, ProjectionRow, ProjectionRows,
            ProjectionSource,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateVersion, AuthenticatedPrincipal, CommunityId, OperationId, PrincipalId,
    PrincipalScopes, Provenance, SourceRecordId, SourceSystem, TenantContext, TrustedTenantRoute,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseTransaction, QueryResult, Statement, Value,
};
use sqlx::{Executor, PgPool, postgres::PgPoolOptions};
use url::Url;
use uuid::Uuid;

const EVENTS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const HEADS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000200_collaboration_event_heads.up.sql"
));
const PROJECTIONS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000300_collaboration_projections.up.sql"
));
const PROJECTIONS_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000300_collaboration_projections.down.sql"
));
const OUTBOX_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000400_collaboration_outbox.up.sql"
));

fn tenant() -> TenantContext {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "storage-recovery")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn command(tenant: &TenantContext) -> DomainCommand<Vec<u8>> {
    command_with_id(tenant, 3)
}

fn command_with_id(tenant: &TenantContext, operation: u128) -> DomainCommand<Vec<u8>> {
    let principal = AuthenticatedPrincipal::service(
        PrincipalId::from_uuid(Uuid::from_u128(2)),
        tenant.community_id(),
        "storage-recovery",
        PrincipalScopes::default(),
    )
    .expect("service principal");
    DomainCommand::new(
        OperationId::from_uuid(Uuid::from_u128(operation)),
        tenant.clone(),
        principal,
        None,
        None,
        CommandAdapter::NostrInProcess,
        b"accepted-command".to_vec(),
    )
}

fn projection_source() -> ProjectionSource {
    ProjectionSource::new(
        "conversation_activity",
        Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new("conversation:42").expect("source ID"),
            1_900_000_000_001,
        )
        .with_source_version("2"),
    )
    .expect("projection source")
}

struct RecoveryMutation {
    topic: &'static str,
}

impl Default for RecoveryMutation {
    fn default() -> Self {
        Self {
            topic: "recovery.accepted",
        }
    }
}

#[async_trait]
impl TransactionalCommandMutation<Vec<u8>> for RecoveryMutation {
    fn fingerprint(
        &self,
        command: &DomainCommand<Vec<u8>>,
    ) -> Result<CommandFingerprint, DomainCommandSubmissionError> {
        CommandFingerprint::new("recovery.accept", command.payload())
    }

    async fn apply(
        &self,
        transaction: &DatabaseTransaction,
        command: &DomainCommand<Vec<u8>>,
    ) -> Result<AppliedCommand, DomainCommandSubmissionError> {
        let result = transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
INSERT INTO public.collaboration_recovery_commands (
    community_id, operation_id, authoritative_version, payload
) VALUES ($1, $2, 1, $3)
"#,
                [
                    Value::Uuid(Some(Box::new(command.tenant().community_id().as_uuid()))),
                    Value::Uuid(Some(Box::new(command.operation_id().as_uuid()))),
                    command.payload().clone().into(),
                ],
            ))
            .await
            .map_err(|_| DomainCommandSubmissionError::Unavailable)?;
        if result.rows_affected() != 1 {
            return Err(DomainCommandSubmissionError::Unavailable);
        }
        let operation = OutboxOperation::new(
            self.topic,
            Provenance::new(
                SourceSystem::Zed,
                SourceRecordId::new(command.operation_id().to_string()).expect("source ID"),
                1_900_000_000_000,
            )
            .with_source_version("1"),
            command.payload().clone(),
        )?;
        Ok(AppliedCommand::new(AggregateVersion::FIRST, operation))
    }
}

struct RecoveryProjectionAdapter;

#[async_trait]
impl ProjectionRebuildAdapter for RecoveryProjectionAdapter {
    async fn load_authority(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
    ) -> Result<ProjectionRows, ProjectionRebuildError> {
        let rows = transaction
            .query_all(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
SELECT record_key, payload
FROM public.collaboration_recovery_authority
WHERE community_id = $1 AND source_record_id = $2
ORDER BY record_key
"#,
                [
                    Value::Uuid(Some(Box::new(tenant.community_id().as_uuid()))),
                    source.provenance().source_record_id.as_str().into(),
                ],
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        projection_rows(rows)
    }

    async fn replace_projection(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
        rows: &ProjectionRows,
    ) -> Result<(), ProjectionRebuildError> {
        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
DELETE FROM public.collaboration_recovery_projection
WHERE community_id = $1 AND source_record_id = $2
"#,
                [
                    Value::Uuid(Some(Box::new(tenant.community_id().as_uuid()))),
                    source.provenance().source_record_id.as_str().into(),
                ],
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        for row in rows.as_slice() {
            transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    r#"
INSERT INTO public.collaboration_recovery_projection (
    community_id, source_record_id, source_version, record_key, payload
) VALUES ($1, $2, $3, $4, $5)
"#,
                    [
                        Value::Uuid(Some(Box::new(tenant.community_id().as_uuid()))),
                        source.provenance().source_record_id.as_str().into(),
                        source.provenance().source_version.as_deref().into(),
                        row.key().into(),
                        row.payload().to_vec().into(),
                    ],
                ))
                .await
                .map_err(ProjectionRebuildError::Unavailable)?;
        }
        Ok(())
    }

    async fn load_projection(
        &self,
        transaction: &DatabaseTransaction,
        tenant: &TenantContext,
        source: &ProjectionSource,
    ) -> Result<ProjectionMaterialization, ProjectionRebuildError> {
        let rows = transaction
            .query_all(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                r#"
SELECT source_version, record_key, payload
FROM public.collaboration_recovery_projection
WHERE community_id = $1 AND source_record_id = $2
ORDER BY record_key
"#,
                [
                    Value::Uuid(Some(Box::new(tenant.community_id().as_uuid()))),
                    source.provenance().source_record_id.as_str().into(),
                ],
            ))
            .await
            .map_err(ProjectionRebuildError::Unavailable)?;
        let source_version = rows
            .first()
            .map(|row| row.try_get("", "source_version"))
            .transpose()
            .map_err(ProjectionRebuildError::Unavailable)?;
        ProjectionMaterialization::new(source_version, projection_rows(rows)?)
    }
}

fn projection_rows(rows: Vec<QueryResult>) -> Result<ProjectionRows, ProjectionRebuildError> {
    let rows = rows
        .into_iter()
        .map(|row| {
            let key: String = row
                .try_get("", "record_key")
                .map_err(ProjectionRebuildError::Unavailable)?;
            let payload: Vec<u8> = row
                .try_get("", "payload")
                .map_err(ProjectionRebuildError::Unavailable)?;
            ProjectionRow::new(key, payload)
        })
        .collect::<Result<Vec<_>, _>>()?;
    ProjectionRows::new(rows)
}

fn admin_database_url() -> String {
    std::env::var("COLLAB_TEST_DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres@localhost/postgres".to_owned())
}

fn isolated_database_url(admin_url: &str, database_name: &str) -> anyhow::Result<String> {
    let mut url = Url::parse(admin_url)?;
    url.set_path(&format!("/{database_name}"));
    Ok(url.to_string())
}

async fn apply_recovery_migrations(pool: &PgPool) -> anyhow::Result<()> {
    for migration in [EVENTS_UP, HEADS_UP, PROJECTIONS_UP, OUTBOX_UP] {
        sqlx::raw_sql(migration).execute(pool).await?;
    }
    Ok(())
}

async fn exercise_projection_checkpoint_fences(pool: &PgPool) -> anyhow::Result<()> {
    let first_community = Uuid::new_v4();
    let second_community = Uuid::new_v4();
    let role = format!("checkpoint_probe_{}", Uuid::new_v4().simple());
    // Rollback removes both probe rows and the role, including on assertion failure.
    let mut transaction = pool.begin().await?;
    for community in [first_community, second_community] {
        sqlx::query(
            r#"
INSERT INTO public.collaboration_projection_checkpoints (
    community_id, projection_name, source_system, source_record_id,
    source_observed_at, projection_version, reset_generation, cursor,
    drift_state, projected_at
) VALUES ($1, 'checkpoint-probe', 'zed', 'probe:1', clock_timestamp(),
          1, 1, $2, 'rebuilding', clock_timestamp())
"#,
        )
        .bind(community)
        .bind(b"page:1".as_slice())
        .execute(&mut *transaction)
        .await?;
    }
    sqlx::raw_sql(&format!(
        "CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS; \
         GRANT USAGE ON SCHEMA public TO {role}; \
         GRANT SELECT, UPDATE ON public.collaboration_projection_checkpoints TO {role}; \
         SET LOCAL ROLE {role};"
    ))
    .execute(&mut *transaction)
    .await?;
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(first_community.to_string())
        .execute(&mut *transaction)
        .await?;
    let bypasses_tenant_fence: bool = sqlx::query_scalar(
        "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user",
    )
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(!bypasses_tenant_fence, "checkpoint probe must exercise RLS");
    let visible: (i64, Vec<u8>) = sqlx::query_as(
        "SELECT projection_version::bigint, cursor FROM public.collaboration_projection_checkpoints",
    )
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(visible == (1, b"page:1".to_vec()));
    let resumed = sqlx::query(
        "UPDATE public.collaboration_projection_checkpoints \
         SET projection_version = 2, cursor = $1 WHERE projection_version = 1",
    )
    .bind(b"page:2".as_slice())
    .execute(&mut *transaction)
    .await?;
    anyhow::ensure!(
        resumed.rows_affected() == 1,
        "resume crossed the tenant fence"
    );
    let stale_resume = sqlx::query(
        "UPDATE public.collaboration_projection_checkpoints \
         SET projection_version = 2, cursor = $1 WHERE projection_version = 1",
    )
    .bind(b"stale-page".as_slice())
    .execute(&mut *transaction)
    .await?;
    anyhow::ensure!(
        stale_resume.rows_affected() == 0,
        "stale cursor was accepted"
    );
    let resumed_cursor: (i64, Vec<u8>) = sqlx::query_as(
        "SELECT projection_version::bigint, cursor FROM public.collaboration_projection_checkpoints",
    )
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(resumed_cursor == (2, b"page:2".to_vec()));

    for (statement, expected_code) in [
        (
            "UPDATE public.collaboration_projection_checkpoints SET projection_version = 2",
            "40001",
        ),
        (
            "UPDATE public.collaboration_projection_checkpoints \
             SET projection_version = 3, reset_generation = 2, \
                 drift_state = 'reset_pending', reset_at = clock_timestamp()",
            "23514",
        ),
    ] {
        sqlx::query("SAVEPOINT checkpoint_rejection")
            .execute(&mut *transaction)
            .await?;
        let rejected = sqlx::query(statement).execute(&mut *transaction).await;
        let error = rejected.err().ok_or_else(|| {
            anyhow::anyhow!("checkpoint fence accepted invalid update: {statement}")
        })?;
        anyhow::ensure!(
            error
                .as_database_error()
                .and_then(|error| error.code())
                .as_deref()
                == Some(expected_code),
            "unexpected checkpoint rejection: {error}"
        );
        sqlx::query("ROLLBACK TO SAVEPOINT checkpoint_rejection")
            .execute(&mut *transaction)
            .await?;
        sqlx::query("RELEASE SAVEPOINT checkpoint_rejection")
            .execute(&mut *transaction)
            .await?;
    }

    let reset = sqlx::query(
        "UPDATE public.collaboration_projection_checkpoints \
         SET projection_version = 3, reset_generation = 2, cursor = NULL, \
             drift_state = 'reset_pending', reset_at = clock_timestamp()",
    )
    .execute(&mut *transaction)
    .await?;
    anyhow::ensure!(reset.rows_affected() == 1, "reset crossed the tenant fence");
    let reset_state: (i64, i64, Option<Vec<u8>>, String) = sqlx::query_as(
        "SELECT projection_version::bigint, reset_generation::bigint, cursor, drift_state \
         FROM public.collaboration_projection_checkpoints",
    )
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(reset_state == (3, 2, None, "reset_pending".to_owned()));
    sqlx::query("RESET ROLE").execute(&mut *transaction).await?;
    let untouched: (i64, i64, Vec<u8>) = sqlx::query_as(
        "SELECT projection_version::bigint, reset_generation::bigint, cursor \
         FROM public.collaboration_projection_checkpoints WHERE community_id = $1",
    )
    .bind(second_community)
    .fetch_one(&mut *transaction)
    .await?;
    anyhow::ensure!(untouched == (1, 1, b"page:1".to_vec()));
    transaction.rollback().await?;
    eprintln!("projection checkpoints: resume, version rejection and tenant-scoped reset verified");
    Ok(())
}

async fn prepare_recovery_tables(pool: &PgPool) -> anyhow::Result<()> {
    sqlx::raw_sql(
        r#"
CREATE TABLE public.collaboration_recovery_commands (
    community_id uuid NOT NULL,
    operation_id uuid NOT NULL,
    authoritative_version bigint NOT NULL,
    payload bytea NOT NULL,
    PRIMARY KEY (community_id, operation_id)
);
CREATE TABLE public.collaboration_recovery_authority (
    community_id uuid NOT NULL,
    source_record_id text NOT NULL,
    record_key text NOT NULL,
    payload bytea NOT NULL,
    PRIMARY KEY (community_id, source_record_id, record_key)
);
CREATE TABLE public.collaboration_recovery_projection (
    community_id uuid NOT NULL,
    source_record_id text NOT NULL,
    source_version text NOT NULL,
    record_key text NOT NULL,
    payload bytea NOT NULL,
    PRIMARY KEY (community_id, source_record_id, record_key)
);
CREATE FUNCTION public.interrupt_collaboration_outbox() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'seeded outbox interruption';
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER interrupt_collaboration_outbox
    BEFORE INSERT ON public.collaboration_outbox
    FOR EACH ROW EXECUTE FUNCTION public.interrupt_collaboration_outbox();
"#,
    )
    .execute(pool)
    .await?;
    Ok(())
}

async fn wait_for_blocked_outbox_writers(pool: &PgPool, expected: i64) -> anyhow::Result<()> {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let waiting: i64 = sqlx::query_scalar(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted AND database = (SELECT oid FROM pg_database WHERE datname = current_database())",
            )
            .fetch_one(pool)
            .await?;
            if waiting == expected {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await??;
    Ok(())
}

async fn exercise_outbox_commit_order(pool: &PgPool, database_url: &str) -> anyhow::Result<()> {
    sqlx::raw_sql(
        r#"
CREATE FUNCTION public.hold_first_outbox_commit() RETURNS trigger AS $$
BEGIN
    IF NEW.operation_id = '00000000-0000-0000-0000-000000000064'::uuid THEN
        PERFORM pg_advisory_xact_lock(1223344::bigint);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER hold_first_outbox_commit AFTER INSERT ON public.collaboration_outbox
FOR EACH ROW EXECUTE FUNCTION public.hold_first_outbox_commit();
"#,
    )
    .execute(pool)
    .await?;
    let mut gate = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock(1223344::bigint)")
        .execute(&mut *gate)
        .await?;
    let connection = sea_orm::Database::connect(database_url).await?;
    let other_topic_store = TransactionalCommandOutbox::new(
        sea_orm::Database::connect(database_url).await?,
        RecoveryMutation {
            topic: "recovery.independent",
        },
    )
    .map_err(|backend| anyhow::anyhow!("unexpected database backend: {backend:?}"))?;
    let store = TransactionalCommandOutbox::new(connection, RecoveryMutation::default())
        .map_err(|backend| anyhow::anyhow!("unexpected database backend: {backend:?}"))?;
    let tenant = tenant();
    let first = store.submit(command_with_id(&tenant, 100));
    tokio::pin!(first);
    tokio::select! {
        result = &mut first => anyhow::bail!("first writer bypassed the commit gate: {result:?}"),
        result = wait_for_blocked_outbox_writers(pool, 1) => result?,
    }
    let second = store.submit(command_with_id(&tenant, 101));
    tokio::pin!(second);
    tokio::select! {
        result = &mut first => anyhow::bail!("first writer bypassed the commit gate: {result:?}"),
        result = &mut second => anyhow::bail!("same-topic writer overtook the pending commit: {result:?}"),
        result = wait_for_blocked_outbox_writers(pool, 2) => result?,
    }
    let other_tenant = bind_rpc_tenant(
        Some(TrustedTenantRoute::from_listener(
            CommunityId::new(),
            "independent-outbox",
        )?),
        &[],
    )
    .map_err(|error| anyhow::anyhow!("independent tenant admission failed: {error:?}"))?;
    let independent = store.submit(command_with_id(&other_tenant, 102));
    tokio::pin!(independent);
    tokio::select! {
        result = &mut first => anyhow::bail!("first writer bypassed the commit gate: {result:?}"),
        result = &mut second => anyhow::bail!("same-topic writer overtook the pending commit: {result:?}"),
        result = &mut independent => anyhow::ensure!(result?.disposition() == DomainCommandDisposition::Applied),
    }
    let independent_topic = other_topic_store.submit(command_with_id(&tenant, 103));
    tokio::pin!(independent_topic);
    tokio::select! {
        result = &mut first => anyhow::bail!("first writer bypassed the commit gate: {result:?}"),
        result = &mut second => anyhow::bail!("same-topic writer overtook the pending commit: {result:?}"),
        result = &mut independent_topic => anyhow::ensure!(result?.disposition() == DomainCommandDisposition::Applied),
    }
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_outbox WHERE operation_id IN ($1, $2)",
    )
    .bind(Uuid::from_u128(100))
    .bind(Uuid::from_u128(101))
    .fetch_one(pool)
    .await?;
    anyhow::ensure!(
        visible == 0,
        "uncommitted topic must not expose a higher replay cursor"
    );
    gate.rollback().await?;
    let (first, second) = tokio::try_join!(first, second)?;
    anyhow::ensure!(first.disposition() == DomainCommandDisposition::Applied);
    anyhow::ensure!(second.disposition() == DomainCommandDisposition::Applied);
    let ordered: Vec<Uuid> = sqlx::query_scalar(
        "SELECT operation_id FROM public.collaboration_outbox WHERE operation_id IN ($1, $2) ORDER BY outbox_sequence",
    )
    .bind(Uuid::from_u128(100))
    .bind(Uuid::from_u128(101))
    .fetch_all(pool)
    .await?;
    anyhow::ensure!(ordered == vec![Uuid::from_u128(100), Uuid::from_u128(101)]);
    sqlx::raw_sql("DROP TRIGGER hold_first_outbox_commit ON public.collaboration_outbox; DROP FUNCTION public.hold_first_outbox_commit();")
        .execute(pool)
        .await?;
    Ok(())
}

async fn wait_for_replica_replay(primary: &PgPool, replica: &PgPool) -> anyhow::Result<()> {
    let target: String = sqlx::query_scalar("SELECT pg_current_wal_lsn()::text")
        .fetch_one(primary)
        .await?;
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            let caught_up: bool = sqlx::query_scalar(
                "SELECT COALESCE(pg_last_wal_replay_lsn() >= $1::text::pg_lsn, false)",
            )
            .bind(&target)
            .fetch_one(replica)
            .await?;
            if caught_up {
                return Ok::<_, anyhow::Error>(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await??;
    Ok(())
}

async fn recovery_counts(pool: &PgPool) -> anyhow::Result<(i64, i64, i64)> {
    Ok(sqlx::query_as(
        "SELECT (SELECT count(*) FROM collaboration_recovery_commands), \
         (SELECT count(*) FROM collaboration_command_receipts), \
         (SELECT count(*) FROM collaboration_outbox)",
    )
    .fetch_one(pool)
    .await?)
}

async fn exercise_physical_replica(
    primary: &PgPool,
    database_url: &str,
    replica_admin_url: &str,
) -> anyhow::Result<()> {
    let replica_admin = PgPoolOptions::new()
        .max_connections(1)
        .connect(replica_admin_url)
        .await?;
    let in_recovery: bool = sqlx::query_scalar("SELECT pg_is_in_recovery()")
        .fetch_one(&replica_admin)
        .await?;
    anyhow::ensure!(in_recovery, "physical replica URL must point to a standby");
    wait_for_replica_replay(primary, &replica_admin).await?;
    let database_name = Url::parse(database_url)?
        .path()
        .trim_start_matches('/')
        .to_owned();
    let replica = PgPoolOptions::new()
        .max_connections(2)
        .connect(&isolated_database_url(replica_admin_url, &database_name)?)
        .await?;
    let before = recovery_counts(primary).await?;
    anyhow::ensure!(recovery_counts(&replica).await? == before);
    sqlx::query("SELECT pg_wal_replay_pause()")
        .execute(&replica_admin)
        .await?;
    let result = async {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let state: String = sqlx::query_scalar("SELECT pg_get_wal_replay_pause_state()")
                    .fetch_one(&replica_admin)
                    .await?;
                if state == "paused" {
                    return Ok::<_, anyhow::Error>(());
                }
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await??;
        let store = TransactionalCommandOutbox::new(
            sea_orm::Database::connect(database_url).await?,
            RecoveryMutation::default(),
        )
        .map_err(|backend| anyhow::anyhow!("unexpected database backend: {backend:?}"))?;
        let accepted = store.submit(command_with_id(&tenant(), 200)).await?;
        anyhow::ensure!(accepted.disposition() == DomainCommandDisposition::Applied);
        anyhow::ensure!(
            recovery_counts(primary).await? == (before.0 + 1, before.1 + 1, before.2 + 1)
        );
        anyhow::ensure!(
            recovery_counts(&replica).await? == before,
            "paused standby exposed a new commit"
        );
        Ok::<_, anyhow::Error>(())
    }
    .await;
    // Resume even when an assertion fails so isolated database cleanup can replay.
    let resume = sqlx::query("SELECT pg_wal_replay_resume()")
        .execute(&replica_admin)
        .await;
    if let Err(error) = resume {
        return Err(match result {
            Ok(()) => error.into(),
            Err(recovery_error) => recovery_error.context(format!(
                "standby replay also failed to resume after the recovery error: {error}"
            )),
        });
    }
    result?;
    wait_for_replica_replay(primary, &replica_admin).await?;
    anyhow::ensure!(recovery_counts(&replica).await? == (before.0 + 1, before.1 + 1, before.2 + 1));
    let store = TransactionalCommandOutbox::new(
        sea_orm::Database::connect(database_url).await?,
        RecoveryMutation::default(),
    )
    .map_err(|backend| anyhow::anyhow!("unexpected database backend: {backend:?}"))?;
    let duplicate = store.submit(command_with_id(&tenant(), 200)).await?;
    anyhow::ensure!(duplicate.disposition() == DomainCommandDisposition::Duplicate);
    wait_for_replica_replay(primary, &replica_admin).await?;
    anyhow::ensure!(recovery_counts(&replica).await? == (before.0 + 1, before.1 + 1, before.2 + 1));
    replica.close().await;
    replica_admin.close().await;
    eprintln!("physical standby: paused lag, atomic catch-up and duplicate retry verified");
    Ok(())
}

async fn exercise_storage_recovery(database_url: &str) -> anyhow::Result<()> {
    let pool = PgPoolOptions::new()
        .max_connections(8)
        .connect(database_url)
        .await?;
    apply_recovery_migrations(&pool).await?;
    exercise_projection_checkpoint_fences(&pool).await?;
    prepare_recovery_tables(&pool).await?;
    let tenant = tenant();

    {
        let connection = sea_orm::Database::connect(database_url).await?;
        let store = TransactionalCommandOutbox::new(connection, RecoveryMutation::default())
            .map_err(|backend| anyhow::anyhow!("unexpected database backend: {backend:?}"))?;
        let interrupted = store.submit(command(&tenant)).await;
        anyhow::ensure!(
            interrupted == Err(DomainCommandSubmissionError::Unavailable),
            "outbox interruption must fail the command"
        );
        let command_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_recovery_commands")
                .fetch_one(&pool)
                .await?;
        let receipt_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_command_receipts")
                .fetch_one(&pool)
                .await?;
        let outbox_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_outbox")
                .fetch_one(&pool)
                .await?;
        anyhow::ensure!(
            (command_count, receipt_count, outbox_count) == (0, 0, 0),
            "interrupted transaction leaked authority, receipt or outbox state"
        );

        sqlx::raw_sql(
            "DROP TRIGGER interrupt_collaboration_outbox ON public.collaboration_outbox;\
             DROP FUNCTION public.interrupt_collaboration_outbox();",
        )
        .execute(&pool)
        .await?;
        let applied = store.submit(command(&tenant)).await?;
        let duplicate = store.submit(command(&tenant)).await?;
        anyhow::ensure!(applied.disposition() == DomainCommandDisposition::Applied);
        anyhow::ensure!(duplicate.disposition() == DomainCommandDisposition::Duplicate);
        let command_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_recovery_commands")
                .fetch_one(&pool)
                .await?;
        let receipt_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_command_receipts")
                .fetch_one(&pool)
                .await?;
        let outbox_count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM public.collaboration_outbox")
                .fetch_one(&pool)
                .await?;
        anyhow::ensure!((command_count, receipt_count, outbox_count) == (1, 1, 1));
    }

    sqlx::query(
        r#"
INSERT INTO public.collaboration_events (
    community_id, event_id, author_public_key, event_created_at, kind, tags,
    content, canonical_event_bytes, signature, signature_state, verified_at,
    persistence_class, discriminator
) VALUES ($1, $2, $3, 1900000000, 1, '[]'::jsonb, 'authority survives',
          $4, $5, 'verified_historical', clock_timestamp(), 'regular', NULL)
"#,
    )
    .bind(tenant.community_id().as_uuid())
    .bind(vec![21_u8; 32])
    .bind(vec![22_u8; 32])
    .bind(b"canonical-authority".as_slice())
    .bind(vec![23_u8; 64])
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
INSERT INTO public.collaboration_recovery_authority (
    community_id, source_record_id, record_key, payload
) VALUES ($1, 'conversation:42', 'activity:1', 'v1')
"#,
    )
    .bind(tenant.community_id().as_uuid())
    .execute(&pool)
    .await?;
    sqlx::query(
        r#"
INSERT INTO public.collaboration_recovery_projection (
    community_id, source_record_id, source_version, record_key, payload
) VALUES ($1, 'conversation:42', '1', 'activity:1', 'v1')
"#,
    )
    .bind(tenant.community_id().as_uuid())
    .execute(&pool)
    .await?;

    let mut stale_reader = pool.begin().await?;
    stale_reader
        .execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        .await?;
    let stale_before: String = sqlx::query_scalar(
        "SELECT source_version FROM public.collaboration_recovery_projection WHERE community_id = $1 AND source_record_id = 'conversation:42'",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(&mut *stale_reader)
    .await?;
    anyhow::ensure!(stale_before == "1");

    sqlx::query(
        "UPDATE public.collaboration_recovery_authority SET payload = 'v2' WHERE community_id = $1 AND source_record_id = 'conversation:42'",
    )
    .bind(tenant.community_id().as_uuid())
    .execute(&pool)
    .await?;
    {
        let connection = sea_orm::Database::connect(database_url).await?;
        let rebuilder = ProjectionRebuilder::new(connection, RecoveryProjectionAdapter)?;
        let diagnostic = rebuilder.rebuild(&tenant, &projection_source()).await?;
        anyhow::ensure!(diagnostic.state() == ProjectionDriftState::Clean);
    }
    let stale_after: String = sqlx::query_scalar(
        "SELECT source_version FROM public.collaboration_recovery_projection WHERE community_id = $1 AND source_record_id = 'conversation:42'",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(&mut *stale_reader)
    .await?;
    anyhow::ensure!(stale_after == "1", "repeatable-read lag fixture advanced");
    stale_reader.rollback().await?;
    let fresh_version: String = sqlx::query_scalar(
        "SELECT source_version FROM public.collaboration_recovery_projection WHERE community_id = $1 AND source_record_id = 'conversation:42'",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(&pool)
    .await?;
    let authoritative_payload: Vec<u8> = sqlx::query_scalar(
        "SELECT payload FROM public.collaboration_recovery_authority WHERE community_id = $1 AND source_record_id = 'conversation:42'",
    )
    .bind(tenant.community_id().as_uuid())
    .fetch_one(&pool)
    .await?;
    anyhow::ensure!(fresh_version == "2");
    anyhow::ensure!(authoritative_payload == b"v2");

    sqlx::raw_sql(PROJECTIONS_DOWN).execute(&pool).await?;
    let event_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_events")
        .fetch_one(&pool)
        .await?;
    let command_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.collaboration_recovery_commands")
            .fetch_one(&pool)
            .await?;
    let outbox_count: i64 = sqlx::query_scalar("SELECT count(*) FROM public.collaboration_outbox")
        .fetch_one(&pool)
        .await?;
    anyhow::ensure!((event_count, command_count, outbox_count) == (1, 1, 1));
    sqlx::raw_sql(PROJECTIONS_UP).execute(&pool).await?;
    {
        let connection = sea_orm::Database::connect(database_url).await?;
        let rebuilder = ProjectionRebuilder::new(connection, RecoveryProjectionAdapter)?;
        let diagnostic = rebuilder.rebuild(&tenant, &projection_source()).await?;
        anyhow::ensure!(diagnostic.state() == ProjectionDriftState::Clean);
    }
    let recovered_checkpoint_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM public.collaboration_projection_checkpoints")
            .fetch_one(&pool)
            .await?;
    anyhow::ensure!(recovered_checkpoint_count == 1);

    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        exercise_outbox_commit_order(&pool, database_url),
    )
    .await??;
    match std::env::var("COLLAB_TEST_REPLICA_DATABASE_URL") {
        Ok(replica_url) => exercise_physical_replica(&pool, database_url, &replica_url).await?,
        Err(std::env::VarError::NotPresent) => {
            eprintln!("physical standby not configured; this run covers same-server recovery only");
        }
        Err(error) => return Err(error.into()),
    }
    pool.close().await;
    Ok(())
}

#[tokio::test]
async fn collaboration_storage_recovery_preserves_authority_across_failures() {
    let admin_url = admin_database_url();
    let database_name = format!("zed_collaboration_recovery_{}", Uuid::new_v4().simple());
    let database_url = isolated_database_url(&admin_url, &database_name)
        .expect("isolated database URL must be valid");
    let admin_pool = PgPoolOptions::new()
        .max_connections(2)
        .connect(&admin_url)
        .await
        .expect("connect to COLLAB_TEST_DATABASE_URL admin database");
    admin_pool
        .execute(format!("CREATE DATABASE {database_name}").as_str())
        .await
        .expect("create isolated collaboration recovery database");

    let result = exercise_storage_recovery(&database_url).await;
    admin_pool
        .execute(
            format!(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{database_name}' AND pid <> pg_backend_pid()"
            )
            .as_str(),
        )
        .await
        .expect("terminate isolated database sessions");
    admin_pool
        .execute(format!("DROP DATABASE {database_name}").as_str())
        .await
        .expect("drop isolated collaboration recovery database");
    admin_pool.close().await;

    result.expect("isolated PostgreSQL recovery scenarios pass");
}
