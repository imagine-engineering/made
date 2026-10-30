use std::path::Path;

use sha2::{Digest, Sha384};
use sqlx::{
    PgPool,
    migrate::{MigrationSource, MigrationType},
};
use uuid::Uuid;

const VERSION: i64 = 20_260_824_000_100;
const CHANNELS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000700_collaboration_channels.up.sql"
));
const PROJECTS_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260822000300_collaboration_projects.up.sql"
));
const UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260824000100_collaboration_workflows.up.sql"
));
const DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260824000100_collaboration_workflows.down.sql"
));

const TABLES: [&str; 6] = [
    "collaboration_workflow_definitions",
    "collaboration_workflow_definition_versions",
    "collaboration_workflow_definition_heads",
    "collaboration_workflow_runs",
    "collaboration_workflow_steps",
    "collaboration_workflow_retries",
];

#[test]
fn collaboration_workflow_schema_keeps_immutable_version_and_project_keys() {
    for required in [
        "PRIMARY KEY (community_id, workflow_id)",
        "PRIMARY KEY (community_id, workflow_id, definition_version)",
        "UNIQUE (community_id, workflow_id, definition_sha256)",
        "definition_schema_version = 1",
        "current_definition_version",
        "head_revision",
        "REFERENCES public.collaboration_workflow_definition_versions",
        "scope_kind IN ('community', 'project')",
        "REFERENCES public.collaboration_project_groups",
        "project_record_version",
        "source_integrity_sha256",
        "source_system",
        "source_record_id",
        "source_version",
        "source_observed_at",
    ] {
        assert!(
            UP.contains(required),
            "missing definition invariant {required}"
        );
    }
}

#[test]
fn collaboration_workflow_schema_normalizes_runs_steps_and_bounded_retries() {
    for required in [
        "PRIMARY KEY (community_id, run_id)",
        "UNIQUE (community_id, trigger_operation_id)",
        "UNIQUE (community_id, run_id, workflow_id, definition_version)",
        "REFERENCES public.collaboration_workflow_definition_versions",
        "PRIMARY KEY (community_id, run_id, step_index)",
        "UNIQUE (community_id, run_id, step_id)",
        "REFERENCES public.collaboration_workflow_runs",
        "attempt_count BETWEEN 0 AND 8",
        "PRIMARY KEY (community_id, run_id, step_index, attempt_number)",
        "attempt_number BETWEEN 2 AND 8",
        "failure_class IN (",
        "'rate_limited', 'temporary_unavailable', 'timeout', 'transport'",
        "REFERENCES public.collaboration_workflow_steps",
        "WHERE state = 'scheduled'",
    ] {
        assert!(
            UP.contains(required),
            "missing execution invariant {required}"
        );
    }
    for unbounded_blob in [
        "octet_length(definition::text) <= 65536",
        "octet_length(trigger_context::text) <= 1048576",
        "octet_length(output::text) <= 65536",
        "octet_length(error_message) <= 4096",
    ] {
        assert!(
            UP.contains(unbounded_blob),
            "missing byte bound {unbounded_blob}"
        );
    }
}

#[test]
fn collaboration_workflow_schema_is_tenant_fenced_and_exactly_reversible() {
    for table in TABLES {
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
    ] {
        assert!(UP.contains(required), "missing tenant invariant {required}");
    }
    assert!(!DOWN.contains("CASCADE"));
    assert_eq!(DOWN.lines().count(), TABLES.len());
}

#[tokio::test]
async fn collaboration_workflow_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let workflow_migrations = migrations
        .iter()
        .filter(|migration| migration.version == VERSION)
        .collect::<Vec<_>>();
    assert_eq!(workflow_migrations.len(), 2);
    let up = workflow_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("workflow up migration");
    let down = workflow_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("workflow down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(UP).as_slice());
    assert_eq!(down.checksum.as_ref(), Sha384::digest(DOWN).as_slice());
}

#[tokio::test]
async fn collaboration_workflow_schema_enforces_live_version_relations_and_tenant_fences() {
    let Some(database_url) = std::env::var("COLLAB_WORKFLOW_MIGRATION_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_WORKFLOW_MIGRATION_TEST_DATABASE_URL is unset; live workflow migration test skipped"
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
    sqlx::raw_sql(PROJECTS_UP)
        .execute(&pool)
        .await
        .expect("apply project migration");
    sqlx::raw_sql(UP)
        .execute(&pool)
        .await
        .expect("apply workflow migration");

    let community_id = Uuid::from_u128(1);
    let foreign_community_id = Uuid::from_u128(2);
    let creator_id = Uuid::from_u128(3);
    let workflow_id = Uuid::from_u128(4);
    let run_id = Uuid::from_u128(5);
    let trigger_operation_id = Uuid::from_u128(6);
    let step_operation_id = Uuid::from_u128(7);
    let retry_operation_id = Uuid::from_u128(8);
    let project_signer = vec![0xaa_u8; 32];
    let project_event = vec![0xbb_u8; 32];
    let definition_hash = vec![0xcc_u8; 32];

    for (id, host, record_id) in [
        (community_id, "workflow.example", "community:workflow"),
        (
            foreign_community_id,
            "workflow-foreign.example",
            "community:foreign",
        ),
    ] {
        sqlx::query(
            "INSERT INTO public.collaboration_communities (community_id, host, lifecycle_state, aggregate_version, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'active', 1, 'zed', $3, '1', now(), now(), now())",
        )
        .bind(id)
        .bind(host)
        .bind(record_id)
        .execute(&pool)
        .await
        .expect("insert community");
    }
    sqlx::query(
        "INSERT INTO public.collaboration_community_memberships (community_id, principal_id, role, status, membership_version, joined_at, updated_at, source_system, source_record_id, source_version, source_observed_at) VALUES ($1, $2, 'owner', 'active', 1, now(), now(), 'zed', 'member:workflow-owner', '1', now())",
    )
    .bind(community_id)
    .bind(creator_id)
    .execute(&pool)
    .await
    .expect("insert workflow creator");
    sqlx::query(
        "INSERT INTO public.collaboration_project_groups (community_id, project_signer_public_key, project_slug, record_version, is_current, source_event_id, source_created_at, name, visibility, source_observed_at, created_at, updated_at) VALUES ($1, $2, 'workflow-project', 1, true, $3, 100, 'Workflow project', 'listed', now(), now(), now())",
    )
    .bind(community_id)
    .bind(&project_signer)
    .bind(&project_event)
    .execute(&pool)
    .await
    .expect("insert project scope");
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_definitions (community_id, workflow_id, creator_principal_id, scope_kind, project_signer_public_key, project_slug, project_record_version, created_at) VALUES ($1, $2, $3, 'project', $4, 'workflow-project', 1, now())",
    )
    .bind(community_id)
    .bind(workflow_id)
    .bind(creator_id)
    .bind(&project_signer)
    .execute(&pool)
    .await
    .expect("insert project-scoped workflow");
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_definition_versions (community_id, workflow_id, definition_version, definition_schema_version, name, definition, definition_sha256, author_principal_id, source_system, source_record_id, source_version, source_observed_at, created_at) VALUES ($1, $2, 1, 1, 'Deploy', '{\"version\":1}'::jsonb, $3, $4, 'zed', 'workflow:deploy', '1', now(), now())",
    )
    .bind(community_id)
    .bind(workflow_id)
    .bind(&definition_hash)
    .bind(creator_id)
    .execute(&pool)
    .await
    .expect("insert immutable definition version");
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_definition_heads (community_id, workflow_id, current_definition_version, head_revision, lifecycle_state, source_system, source_record_id, source_version, source_observed_at, updated_at) VALUES ($1, $2, 1, 1, 'active', 'zed', 'workflow-head:deploy', '1', now(), now())",
    )
    .bind(community_id)
    .bind(workflow_id)
    .execute(&pool)
    .await
    .expect("insert current definition head");

    assert!(
        sqlx::query(
            "INSERT INTO public.collaboration_workflow_runs (community_id, run_id, workflow_id, definition_version, trigger_operation_id, trigger_kind, trigger_source_id, trigger_context, run_version, status, current_step_index, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 2, $4, 'manual', 'manual:bad-version', '{}'::jsonb, 1, 'queued', 0, 'zed', 'run:bad-version', '1', now(), now(), now())",
        )
        .bind(community_id)
        .bind(Uuid::from_u128(20))
        .bind(workflow_id)
        .bind(Uuid::from_u128(21))
        .execute(&pool)
        .await
        .is_err(),
        "a run must reference an existing immutable definition version"
    );
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_runs (community_id, run_id, workflow_id, definition_version, trigger_operation_id, trigger_kind, trigger_source_id, trigger_context, run_version, status, current_step_index, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 1, $4, 'manual', 'manual:deploy', '{}'::jsonb, 1, 'queued', 0, 'zed', 'run:deploy', '1', now(), now(), now())",
    )
    .bind(community_id)
    .bind(run_id)
    .bind(workflow_id)
    .bind(trigger_operation_id)
    .execute(&pool)
    .await
    .expect("insert definition-bound run");
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_steps (community_id, run_id, workflow_id, definition_version, step_index, step_id, operation_id, state, attempt_count, source_system, source_record_id, source_version, source_observed_at, created_at, updated_at) VALUES ($1, $2, $3, 1, 0, 'deploy', $4, 'retry_scheduled', 1, 'zed', 'step:deploy', '1', now(), now(), now())",
    )
    .bind(community_id)
    .bind(run_id)
    .bind(workflow_id)
    .bind(step_operation_id)
    .execute(&pool)
    .await
    .expect("insert run-bound step");
    sqlx::query(
        "INSERT INTO public.collaboration_workflow_retries (community_id, run_id, step_index, attempt_number, retry_operation_id, failure_class, state, scheduled_at, due_at, source_system, source_record_id, source_version, source_observed_at, created_at) VALUES ($1, $2, 0, 2, $3, 'temporary_unavailable', 'scheduled', now(), now() + interval '1 second', 'zed', 'retry:deploy:2', '1', now(), now())",
    )
    .bind(community_id)
    .bind(run_id)
    .bind(retry_operation_id)
    .execute(&pool)
    .await
    .expect("insert step-bound retry");

    sqlx::raw_sql(
        "CREATE ROLE collaboration_workflow_request NOLOGIN NOBYPASSRLS; \
         GRANT SELECT, INSERT, UPDATE, DELETE ON public.collaboration_workflow_definitions, \
         public.collaboration_workflow_definition_versions, \
         public.collaboration_workflow_definition_heads, public.collaboration_workflow_runs, \
         public.collaboration_workflow_steps, public.collaboration_workflow_retries \
         TO collaboration_workflow_request;",
    )
    .execute(&pool)
    .await
    .expect("create request role");
    let mut transaction = pool
        .begin()
        .await
        .expect("begin foreign tenant transaction");
    sqlx::query("SET LOCAL ROLE collaboration_workflow_request")
        .execute(&mut *transaction)
        .await
        .expect("assume request role");
    sqlx::query("SELECT set_config('app.community_id', $1, true)")
        .bind(foreign_community_id.to_string())
        .execute(&mut *transaction)
        .await
        .expect("set foreign tenant");
    let visible: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_workflow_runs WHERE community_id = $1",
    )
    .bind(community_id)
    .fetch_one(&mut *transaction)
    .await
    .expect("query foreign workflow tenant");
    assert_eq!(visible, 0);
    transaction
        .rollback()
        .await
        .expect("rollback foreign tenant transaction");

    sqlx::raw_sql(DOWN)
        .execute(&pool)
        .await
        .expect("roll workflow migration down");
    for table in TABLES {
        let remaining: Option<String> = sqlx::query_scalar("SELECT to_regclass($1)::text")
            .bind(format!("public.{table}"))
            .fetch_one(&pool)
            .await
            .expect("query rolled-down workflow table");
        assert_eq!(remaining, None, "{table} must be removed by rollback");
    }
    pool.close().await;
}
