use std::{collections::BTreeMap, path::Path, str::FromStr};

use collab::workflows::repository::{
    WorkflowIdentity, WorkflowProvenance, WorkflowRepositoryError, WorkflowRunIdentity,
    WorkflowRunRequest, WorkflowTriggerKind,
};
use collab::workflows::scheduler::{
    MAX_CONCURRENT_RUNS_PER_COMMUNITY, MAX_CONCURRENT_RUNS_PER_DEFINITION,
    MAX_QUEUED_RUNS_PER_COMMUNITY, MAX_QUEUED_RUNS_PER_DEPLOYMENT, WorkflowCapacityScope,
    WorkflowQueueAdmission, WorkflowScheduler,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use collaboration_workflow::definition::WorkflowDefinition;
use sea_orm::{DatabaseBackend, DbErr, MockDatabase, MockExecResult};
use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
    postgres::{PgConnectOptions, PgPoolOptions},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_825_000_100;
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../collab/migrations/20260825000100_collaboration_workflow_scheduler_admission.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../collab/migrations/20260825000100_collaboration_workflow_scheduler_admission.down.sql"
));

fn tenant() -> TenantContext {
    let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "workflow-scheduler")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn run_request() -> WorkflowRunRequest {
    let community_id = tenant().community_id();
    WorkflowRunRequest {
        identity: WorkflowRunIdentity::new(community_id, Uuid::from_u128(20))
            .expect("run identity"),
        workflow: WorkflowIdentity::new(community_id, Uuid::from_u128(10))
            .expect("workflow identity"),
        definition_version: 1,
        trigger_operation_id: Uuid::from_u128(30),
        trigger_kind: WorkflowTriggerKind::Webhook,
        trigger_source_id: "webhook:operation-30".to_owned(),
        trigger_context: serde_json::json!({}),
        step_operation_ids: vec![Uuid::from_u128(40)],
        provenance: WorkflowProvenance::new("zed", "run:20", "1", 1_000, None)
            .expect("run provenance"),
        created_at_millis: 1_000,
    }
}

fn definition_row() -> BTreeMap<String, sea_orm::Value> {
    let definition = WorkflowDefinition::parse_yaml(
        r#"
version: 1
name: Queue admission
enabled: true
trigger:
  on: webhook
steps:
  - id: announce
    action: send_message
    text: queued
"#,
    )
    .expect("workflow definition");
    let encoded = serde_json::to_string(&definition).expect("canonical definition");
    BTreeMap::from([
        (
            "creator_principal_id".to_owned(),
            Uuid::from_u128(50).into(),
        ),
        ("scope_kind".to_owned(), "community".into()),
        (
            "project_signer_public_key".to_owned(),
            Option::<Vec<u8>>::None.into(),
        ),
        ("project_slug".to_owned(), Option::<String>::None.into()),
        (
            "project_record_version_text".to_owned(),
            Option::<String>::None.into(),
        ),
        ("definition_version_text".to_owned(), "1".into()),
        ("definition_schema_version".to_owned(), 1_i32.into()),
        ("name".to_owned(), "Queue admission".into()),
        ("definition_json".to_owned(), encoded.clone().into()),
        (
            "definition_sha256".to_owned(),
            sha2::Sha256::digest(encoded.as_bytes()).to_vec().into(),
        ),
        ("author_principal_id".to_owned(), Uuid::from_u128(50).into()),
        ("source_system".to_owned(), "zed".into()),
        ("source_record_id".to_owned(), "workflow:10".into()),
        ("source_version".to_owned(), "1".into()),
        ("source_observed_at_millis".to_owned(), 1_000_i64.into()),
        (
            "source_integrity_sha256".to_owned(),
            Option::<Vec<u8>>::None.into(),
        ),
        ("created_at_millis".to_owned(), 1_000_i64.into()),
        ("current_definition_version_text".to_owned(), "1".into()),
        ("head_revision_text".to_owned(), "1".into()),
        ("lifecycle_state".to_owned(), "active".into()),
    ])
}

fn stored_run_row() -> BTreeMap<String, sea_orm::Value> {
    let request = run_request();
    BTreeMap::from([
        (
            "workflow_id".to_owned(),
            request.workflow.workflow_id().into(),
        ),
        ("definition_version_text".to_owned(), "1".into()),
        (
            "trigger_operation_id".to_owned(),
            request.trigger_operation_id.into(),
        ),
        ("trigger_kind".to_owned(), "webhook".into()),
        (
            "trigger_source_id".to_owned(),
            request.trigger_source_id.into(),
        ),
        ("trigger_context_json".to_owned(), "{}".into()),
        ("run_version_text".to_owned(), "1".into()),
        ("status".to_owned(), "queued".into()),
        ("current_step_index".to_owned(), 0_i16.into()),
        ("error_code".to_owned(), Option::<String>::None.into()),
        ("error_message".to_owned(), Option::<String>::None.into()),
        ("source_system".to_owned(), "zed".into()),
        ("source_record_id".to_owned(), "run:20".into()),
        ("source_version".to_owned(), "1".into()),
        ("source_observed_at_millis".to_owned(), 1_000_i64.into()),
        ("created_at_millis".to_owned(), 1_000_i64.into()),
        ("started_at_millis".to_owned(), Option::<i64>::None.into()),
        ("completed_at_millis".to_owned(), Option::<i64>::None.into()),
        ("updated_at_millis".to_owned(), 1_000_i64.into()),
    ])
}

fn stored_step_row() -> BTreeMap<String, sea_orm::Value> {
    BTreeMap::from([
        ("workflow_id".to_owned(), Uuid::from_u128(10).into()),
        ("definition_version_text".to_owned(), "1".into()),
        ("step_index".to_owned(), 0_i16.into()),
        ("step_id".to_owned(), "announce".into()),
        ("operation_id".to_owned(), Uuid::from_u128(40).into()),
        ("state".to_owned(), "pending".into()),
        ("attempt_count".to_owned(), 0_i16.into()),
        ("output_json".to_owned(), Option::<String>::None.into()),
        ("error_code".to_owned(), Option::<String>::None.into()),
        ("error_message".to_owned(), Option::<String>::None.into()),
        ("source_system".to_owned(), "zed".into()),
        ("source_record_id".to_owned(), "run:20:step:announce".into()),
        ("source_version".to_owned(), "1".into()),
        ("source_observed_at_millis".to_owned(), 1_000_i64.into()),
        ("created_at_millis".to_owned(), 1_000_i64.into()),
        ("started_at_millis".to_owned(), Option::<i64>::None.into()),
        ("completed_at_millis".to_owned(), Option::<i64>::None.into()),
        ("updated_at_millis".to_owned(), 1_000_i64.into()),
    ])
}

#[test]
fn workflow_scheduler_limits_match_ol_exe_04() {
    assert_eq!(MAX_QUEUED_RUNS_PER_COMMUNITY, 1_000);
    assert_eq!(MAX_QUEUED_RUNS_PER_DEPLOYMENT, 10_000);
    assert_eq!(MAX_CONCURRENT_RUNS_PER_COMMUNITY, 16);
    assert_eq!(MAX_CONCURRENT_RUNS_PER_DEFINITION, 4);

    for invariant in [
        "community_queue_depth >= 1000",
        "deployment_queue_depth >= 10000",
        "community_execution_count >= 16",
        "definition_execution_count >= 4",
        "workflow_scheduler_capacity_unavailable:community_queue",
        "workflow_scheduler_capacity_unavailable:deployment_queue",
        "workflow_scheduler_capacity_unavailable:community_execution",
        "workflow_scheduler_capacity_unavailable:definition_execution",
    ] {
        assert!(
            UP.contains(invariant),
            "missing scheduler invariant {invariant}"
        );
    }
}

#[test]
fn workflow_scheduler_queue_index_is_opaque_and_transactional() {
    assert!(UP.contains("sha256(uuid_send(community_id) || uuid_send(run_id))"));
    assert!(UP.contains("sha256(uuid_send(NEW.community_id) || uuid_send(NEW.run_id))"));
    assert!(UP.contains("pg_advisory_xact_lock(7449358843737115665)"));
    assert!(
        UP.contains("REVOKE ALL ON public.collaboration_workflow_ready_queue_index FROM PUBLIC")
    );
    assert_eq!(UP.matches("SECURITY DEFINER").count(), 3);
    assert!(UP.contains("workflow_scheduler_tenant_context_mismatch"));

    let table_definition = UP
        .split("REVOKE ALL")
        .next()
        .expect("ready queue table definition");
    assert!(!table_definition.contains("community_id"));
    assert!(!table_definition.contains("run_id"));
}

#[test]
fn workflow_scheduler_migration_is_exactly_reversible() {
    for object in [
        "collaboration_workflow_runs_ready_queue_admission",
        "collaboration_workflow_runs_ready_queue_release",
        "collaboration_workflow_run_leases_execution_admission",
        "collaboration_workflow_admit_ready_queue",
        "collaboration_workflow_release_ready_queue",
        "collaboration_workflow_observe_ready_queue",
        "collaboration_workflow_admit_execution",
        "collaboration_workflow_ready_queue_index",
    ] {
        assert!(UP.contains(object), "up migration omits {object}");
        assert!(DOWN.contains(object), "down migration omits {object}");
    }
    assert!(!DOWN.contains("CASCADE"));
}

#[tokio::test]
async fn workflow_scheduler_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../collab/migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let scheduler_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(scheduler_migrations.len(), 2);
    let up = scheduler_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("scheduler up migration");
    let down = scheduler_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("scheduler down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn workflow_scheduler_observation_exposes_only_bounded_depth_and_age() {
    let row = BTreeMap::from([
        ("community_queue_depth".to_owned(), 12_i64.into()),
        ("deployment_queue_depth".to_owned(), 34_i64.into()),
        (
            "community_oldest_at_millis".to_owned(),
            Some(1_000_i64).into(),
        ),
        (
            "deployment_oldest_at_millis".to_owned(),
            Some(500_i64).into(),
        ),
    ]);
    let connection = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .append_query_results([vec![row]])
        .into_connection();
    let scheduler = WorkflowScheduler::new(connection).expect("scheduler");

    let observation = scheduler
        .observe_queue(&tenant(), 10_500)
        .await
        .expect("queue observation");

    assert_eq!(observation.community_queue_depth, 12);
    assert_eq!(observation.deployment_queue_depth, 34);
    assert_eq!(observation.oldest_queued_seconds, Some(10));
}

#[tokio::test]
async fn workflow_scheduler_exact_queue_retry_is_duplicate_without_new_admission() {
    let request = run_request();
    let connection = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([MockExecResult {
            last_insert_id: 0,
            rows_affected: 1,
        }])
        .append_query_results([
            vec![BTreeMap::from([(
                "run_id".to_owned(),
                request.identity.run_id().into(),
            )])],
            vec![stored_run_row()],
            vec![stored_step_row()],
            vec![],
        ])
        .into_connection();
    let scheduler = WorkflowScheduler::new(connection).expect("scheduler");

    assert_eq!(
        scheduler
            .queue_run(&tenant(), &request)
            .await
            .expect("exact queue retry"),
        WorkflowQueueAdmission::Duplicate
    );
}

#[tokio::test]
async fn workflow_scheduler_new_run_returns_typed_queued_admission() {
    let request = run_request();
    let connection = MockDatabase::new(DatabaseBackend::Postgres)
        .append_exec_results([
            MockExecResult {
                last_insert_id: 0,
                rows_affected: 1,
            },
            MockExecResult {
                last_insert_id: 0,
                rows_affected: 1,
            },
            MockExecResult {
                last_insert_id: 0,
                rows_affected: 1,
            },
        ])
        .append_query_results([vec![], vec![definition_row()]])
        .into_connection();
    let scheduler = WorkflowScheduler::new(connection).expect("scheduler");

    assert_eq!(
        scheduler
            .queue_run(&tenant(), &request)
            .await
            .expect("new queue admission"),
        WorkflowQueueAdmission::Queued
    );
}

#[tokio::test]
async fn workflow_scheduler_maps_each_database_cap_to_a_typed_scope() {
    for (database_scope, expected_scope) in [
        ("community_queue", WorkflowCapacityScope::CommunityQueue),
        ("deployment_queue", WorkflowCapacityScope::DeploymentQueue),
        (
            "community_execution",
            WorkflowCapacityScope::CommunityExecution,
        ),
        (
            "definition_execution",
            WorkflowCapacityScope::DefinitionExecution,
        ),
    ] {
        let connection = MockDatabase::new(DatabaseBackend::Postgres)
            .append_exec_results([MockExecResult {
                last_insert_id: 0,
                rows_affected: 1,
            }])
            .append_exec_errors([DbErr::Custom(format!(
                "workflow_scheduler_capacity_unavailable:{database_scope}"
            ))])
            .append_query_results([vec![], vec![definition_row()]])
            .into_connection();
        let scheduler = WorkflowScheduler::new(connection).expect("scheduler");

        let error = scheduler
            .queue_run(&tenant(), &run_request())
            .await
            .expect_err("capacity must fail closed");
        assert!(matches!(
            error,
            WorkflowRepositoryError::CapacityUnavailable(scope)
                if scope == expected_scope
        ));
    }
}

#[tokio::test]
async fn workflow_scheduler_live_admission_enforces_all_caps_and_releases_after_restart() {
    let Some(database_url) = std::env::var("COLLAB_WORKFLOW_SCHEDULER_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_WORKFLOW_SCHEDULER_TEST_DATABASE_URL is unset; live scheduler test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(
        r#"
CREATE TABLE public.collaboration_workflow_runs (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL,
    workflow_id uuid NOT NULL,
    status text NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    PRIMARY KEY (community_id, run_id)
);
CREATE TABLE public.collaboration_workflow_run_leases (
    community_id uuid NOT NULL,
    run_id uuid NOT NULL,
    lease_id uuid NOT NULL,
    state text NOT NULL,
    acquired_at timestamptz NOT NULL,
    recovery_after timestamptz NOT NULL,
    PRIMARY KEY (community_id, lease_id)
);
ALTER TABLE public.collaboration_workflow_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_run_leases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.collaboration_workflow_run_leases FORCE ROW LEVEL SECURITY;
CREATE POLICY scheduler_runs_admission ON public.collaboration_workflow_runs
    AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY scheduler_runs_community ON public.collaboration_workflow_runs
    AS RESTRICTIVE FOR ALL
    USING (community_id = NULLIF(current_setting('app.community_id', true), '')::uuid)
    WITH CHECK (community_id = NULLIF(current_setting('app.community_id', true), '')::uuid);
CREATE POLICY scheduler_leases_admission ON public.collaboration_workflow_run_leases
    AS PERMISSIVE FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY scheduler_leases_community ON public.collaboration_workflow_run_leases
    AS RESTRICTIVE FOR ALL
    USING (community_id = NULLIF(current_setting('app.community_id', true), '')::uuid)
    WITH CHECK (community_id = NULLIF(current_setting('app.community_id', true), '')::uuid);
"#,
    )
    .execute(&pool)
    .await
    .expect("create scheduler fixture schema");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply scheduler migration");

    sqlx::raw_sql(
        r#"
CREATE ROLE collaboration_scheduler_runtime LOGIN PASSWORD 'scheduler-runtime';
GRANT USAGE ON SCHEMA public TO collaboration_scheduler_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON public.collaboration_workflow_runs,
       public.collaboration_workflow_run_leases
    TO collaboration_scheduler_runtime;
GRANT EXECUTE
    ON FUNCTION public.collaboration_workflow_observe_ready_queue(uuid)
    TO collaboration_scheduler_runtime;
"#,
    )
    .execute(&pool)
    .await
    .expect("grant the isolated runtime role only canonical workflow access");
    let runtime_options = PgConnectOptions::from_str(&database_url)
        .expect("parse isolated PostgreSQL URL")
        .username("collaboration_scheduler_runtime")
        .password("scheduler-runtime");
    let runtime_pool = PgPoolOptions::new()
        .max_connections(1)
        .connect_with(runtime_options)
        .await
        .expect("connect isolated runtime role");
    let runtime_community = Uuid::from_u128(90);
    let runtime_run = Uuid::from_u128(90_001);
    let mut runtime_transaction = runtime_pool
        .begin()
        .await
        .expect("begin runtime admission transaction");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(runtime_community.to_string())
        .execute(&mut *runtime_transaction)
        .await
        .expect("set runtime tenant");
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_runs (community_id, run_id, workflow_id, status, created_at, updated_at) VALUES ($1, $2, $3, 'queued', now(), now())",
    )
    .bind(runtime_community)
    .bind(runtime_run)
    .bind(Uuid::from_u128(900))
    .execute(&mut *runtime_transaction)
    .await
    .expect("security-definer trigger admits runtime queue insert");
    let runtime_observation: (i64, Option<i64>, i64, Option<i64>) =
        sqlx::query_as("SELECT * FROM public.collaboration_workflow_observe_ready_queue($1)")
            .bind(runtime_community)
            .fetch_one(&mut *runtime_transaction)
            .await
            .expect("runtime can read only bounded queue observation");
    assert_eq!(runtime_observation.0, 1);
    assert_eq!(runtime_observation.2, 1);
    assert!(
        sqlx::query_scalar::<_, i64>(
            "SELECT count(*)::bigint FROM public.collaboration_workflow_ready_queue_index",
        )
        .fetch_one(&mut *runtime_transaction)
        .await
        .expect_err("runtime must not read opaque deployment index directly")
        .to_string()
        .contains("permission denied")
    );
    runtime_transaction
        .rollback()
        .await
        .expect("rollback runtime privilege probe");
    runtime_pool.close().await;

    let workflow_id = Uuid::from_u128(100);
    for community_number in 1_u128..=10 {
        insert_queued_runs(
            &pool,
            Uuid::from_u128(community_number),
            workflow_id,
            u64::try_from((community_number - 1) * 1_000).expect("queue offset"),
            MAX_QUEUED_RUNS_PER_COMMUNITY,
        )
        .await;
    }

    let community_overflow =
        insert_queued_runs_result(&pool, Uuid::from_u128(1), workflow_id, 20_000, 1)
            .await
            .expect_err("community queue overflow must fail");
    assert!(community_overflow.to_string().contains("community_queue"));

    let deployment_overflow =
        insert_queued_runs_result(&pool, Uuid::from_u128(11), workflow_id, 21_000, 1)
            .await
            .expect_err("deployment queue overflow must fail");
    assert!(deployment_overflow.to_string().contains("deployment_queue"));

    let queue_depth: i64 = sqlx::query_scalar(
        "SELECT count(*)::bigint FROM public.collaboration_workflow_ready_queue_index",
    )
    .fetch_one(&pool)
    .await
    .expect("deployment queue depth");
    assert_eq!(queue_depth, i64::from(MAX_QUEUED_RUNS_PER_DEPLOYMENT));

    sqlx::query("DELETE FROM public.collaboration_workflow_runs")
        .execute(&pool)
        .await
        .expect("drain durable queue");

    let execution_community = Uuid::from_u128(30);
    let definition_ids = (0_u128..5)
        .map(|offset| Uuid::from_u128(300 + offset))
        .collect::<Vec<_>>();
    let mut run_ids = Vec::new();
    for (definition_index, definition_id) in definition_ids.iter().enumerate() {
        for run_index in 0_u128..5 {
            let run_id = Uuid::from_u128(
                30_000
                    + u128::try_from(definition_index).expect("definition index") * 10
                    + run_index,
            );
            sqlx::query(
                "INSERT INTO public.collaboration_workflow_runs (community_id, run_id, workflow_id, status, created_at, updated_at) VALUES ($1, $2, $3, 'queued', now(), now())",
            )
            .bind(execution_community)
            .bind(run_id)
            .bind(definition_id)
            .execute(&pool)
            .await
            .expect("insert execution candidate");
            run_ids.push((definition_index, run_id));
        }
    }
    sqlx::query(
        "UPDATE public.collaboration_workflow_runs SET status = 'running' WHERE community_id = $1",
    )
    .bind(execution_community)
    .execute(&pool)
    .await
    .expect("dispatch execution candidates");

    for (_, run_id) in run_ids
        .iter()
        .filter(|(definition, _)| *definition == 0)
        .take(4)
    {
        insert_active_lease(&pool, execution_community, *run_id).await;
    }
    let definition_overflow = insert_active_lease_result(&pool, execution_community, run_ids[4].1)
        .await
        .expect_err("definition concurrency overflow must fail");
    assert!(
        definition_overflow
            .to_string()
            .contains("definition_execution")
    );

    for definition_index in 1..4 {
        for (_, run_id) in run_ids
            .iter()
            .filter(|(definition, _)| *definition == definition_index)
            .take(4)
        {
            insert_active_lease(&pool, execution_community, *run_id).await;
        }
    }
    let community_overflow = insert_active_lease_result(
        &pool,
        execution_community,
        run_ids
            .iter()
            .find(|(definition, _)| *definition == 4)
            .expect("fifth definition run")
            .1,
    )
    .await
    .expect_err("community concurrency overflow must fail");
    assert!(
        community_overflow
            .to_string()
            .contains("community_execution")
    );

    let released_lease_id = run_ids[0].1;
    drop(pool);
    let restarted_pool = PgPool::connect(&database_url)
        .await
        .expect("reconnect scheduler after restart");
    sqlx::query(
        "UPDATE public.collaboration_workflow_run_leases SET state = 'released' WHERE community_id = $1 AND lease_id = $2",
    )
    .bind(execution_community)
    .bind(released_lease_id)
    .execute(&restarted_pool)
    .await
    .expect("release durable execution slot");
    insert_active_lease(
        &restarted_pool,
        execution_community,
        run_ids
            .iter()
            .find(|(definition, _)| *definition == 4)
            .expect("replacement execution")
            .1,
    )
    .await;

    sqlx::raw_sql(DOWN)
        .execute(&restarted_pool)
        .await
        .expect("roll scheduler migration down");
    let queue_index: Option<String> = sqlx::query_scalar(
        "SELECT to_regclass('public.collaboration_workflow_ready_queue_index')::text",
    )
    .fetch_one(&restarted_pool)
    .await
    .expect("inspect rollback");
    assert_eq!(queue_index, None);
    sqlx::raw_sql(
        "DROP TABLE public.collaboration_workflow_run_leases; DROP TABLE public.collaboration_workflow_runs; DROP OWNED BY collaboration_scheduler_runtime; DROP ROLE collaboration_scheduler_runtime;",
    )
    .execute(&restarted_pool)
    .await
    .expect("drop scheduler fixture schema");
}

async fn insert_queued_runs(
    pool: &PgPool,
    community_id: Uuid,
    workflow_id: Uuid,
    offset: u64,
    count: u32,
) {
    insert_queued_runs_result(pool, community_id, workflow_id, offset, count)
        .await
        .expect("insert queued runs");
}

async fn insert_queued_runs_result(
    pool: &PgPool,
    community_id: Uuid,
    workflow_id: Uuid,
    offset: u64,
    count: u32,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        r#"
INSERT INTO public.collaboration_workflow_runs (
    community_id, run_id, workflow_id, status, created_at, updated_at
)
SELECT
    $1,
    ('00000000-0000-0000-0000-' || lpad(to_hex($2::bigint + value), 12, '0'))::uuid,
    $3,
    'queued',
    now(),
    now()
FROM generate_series(1, $4::bigint) AS value
"#,
    )
    .bind(community_id)
    .bind(i64::try_from(offset).expect("queue offset fits i64"))
    .bind(workflow_id)
    .bind(i64::from(count))
    .execute(pool)
    .await?;
    Ok(())
}

async fn insert_active_lease(pool: &PgPool, community_id: Uuid, run_id: Uuid) {
    insert_active_lease_result(pool, community_id, run_id)
        .await
        .expect("insert active lease");
}

async fn insert_active_lease_result(
    pool: &PgPool,
    community_id: Uuid,
    run_id: Uuid,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_run_leases (community_id, run_id, lease_id, state, acquired_at, recovery_after) VALUES ($1, $2, $2, 'active', now(), now() + interval '1 minute')",
    )
    .bind(community_id)
    .bind(run_id)
    .execute(pool)
    .await?;
    Ok(())
}
