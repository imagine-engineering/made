use std::{collections::BTreeMap, path::Path};

use collab::workflows::repository::{
    DefinitionVersionWrite, RetryFailureClass, WorkflowIdentity, WorkflowLifecycle,
    WorkflowProvenance, WorkflowRepository, WorkflowRepositoryError, WorkflowRunIdentity,
    WorkflowRunLeaseFence, WorkflowRunLeaseRequest, WorkflowRunRequest, WorkflowRunState,
    WorkflowScope, WorkflowStepCheckpoint, WorkflowStepState, WorkflowStoreOutcome,
    WorkflowTriggerKind,
};
use collaboration_domain::{CommunityId, PrincipalId, TenantContext, TrustedTenantRoute};
use collaboration_workflow::definition::WorkflowDefinition;
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value};
use sha2::{Digest, Sha256, Sha384};
use sqlx::migrate::{MigrationSource, MigrationType};
use uuid::Uuid;

const NOW: i64 = 1_900_000_000_000;
const LEASE_VERSION: i64 = 20_260_824_000_200;
const LEASE_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260824000200_collaboration_workflow_run_leases.up.sql"
));
const LEASE_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260824000200_collaboration_workflow_run_leases.down.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "workflow-repository")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn definition() -> WorkflowDefinition {
    WorkflowDefinition::parse_yaml(
        r#"
version: 1
name: Deploy
enabled: true
trigger:
  on: webhook
steps:
  - id: announce
    action: send_message
    text: deployment started
  - id: approve
    action: request_approval
    from: owners
    message: approve deployment
    timeout: 1h
"#,
    )
    .expect("valid workflow definition")
}

fn provenance(record: &str) -> WorkflowProvenance {
    WorkflowProvenance::new(
        "zed",
        record,
        "1",
        u64::try_from(NOW).expect("timestamp"),
        None,
    )
    .expect("valid provenance")
}

fn workflow_identity(community_id: CommunityId) -> WorkflowIdentity {
    WorkflowIdentity::new(community_id, Uuid::from_u128(10)).expect("workflow identity")
}

fn run_identity(community_id: CommunityId) -> WorkflowRunIdentity {
    WorkflowRunIdentity::new(community_id, Uuid::from_u128(20)).expect("run identity")
}

fn definition_write(community_id: CommunityId) -> DefinitionVersionWrite {
    DefinitionVersionWrite {
        identity: workflow_identity(community_id),
        definition_version: 1,
        definition: definition(),
        creator_principal_id: principal(30),
        author_principal_id: principal(30),
        scope: WorkflowScope::Community,
        lifecycle: WorkflowLifecycle::Active,
        expected_head_revision: None,
        provenance: provenance("workflow:deploy:v1"),
        created_at_millis: u64::try_from(NOW).expect("timestamp"),
    }
}

fn definition_row(community_id: CommunityId) -> BTreeMap<String, Value> {
    let write = definition_write(community_id);
    let encoded = serde_json::to_string(&write.definition).expect("canonical definition JSON");
    BTreeMap::from([
        (
            "creator_principal_id".into(),
            write.creator_principal_id.as_uuid().into(),
        ),
        ("scope_kind".into(), "community".into()),
        (
            "project_signer_public_key".into(),
            Option::<Vec<u8>>::None.into(),
        ),
        ("project_slug".into(), Option::<String>::None.into()),
        (
            "project_record_version_text".into(),
            Option::<String>::None.into(),
        ),
        ("definition_version_text".into(), "1".into()),
        ("definition_schema_version".into(), 1_i32.into()),
        ("name".into(), "Deploy".into()),
        ("definition_json".into(), encoded.clone().into()),
        (
            "definition_sha256".into(),
            Sha256::digest(encoded.as_bytes()).to_vec().into(),
        ),
        (
            "author_principal_id".into(),
            write.author_principal_id.as_uuid().into(),
        ),
        ("source_system".into(), "zed".into()),
        ("source_record_id".into(), "workflow:deploy:v1".into()),
        ("source_version".into(), "1".into()),
        ("source_observed_at_millis".into(), NOW.into()),
        (
            "source_integrity_sha256".into(),
            Option::<Vec<u8>>::None.into(),
        ),
        ("created_at_millis".into(), NOW.into()),
        ("current_definition_version_text".into(), "1".into()),
        ("head_revision_text".into(), "1".into()),
        ("lifecycle_state".into(), "active".into()),
    ])
}

fn run_request(community_id: CommunityId) -> WorkflowRunRequest {
    WorkflowRunRequest {
        identity: run_identity(community_id),
        workflow: workflow_identity(community_id),
        definition_version: 1,
        trigger_operation_id: Uuid::from_u128(40),
        trigger_kind: WorkflowTriggerKind::Webhook,
        trigger_source_id: "webhook:request-1".to_owned(),
        trigger_context: serde_json::json!({"request_id": "request-1"}),
        step_operation_ids: vec![Uuid::from_u128(50), Uuid::from_u128(51)],
        provenance: provenance("run:deploy:1"),
        created_at_millis: u64::try_from(NOW).expect("timestamp"),
    }
}

fn run_row(community_id: CommunityId, state: &str, version: u64) -> BTreeMap<String, Value> {
    let request = run_request(community_id);
    BTreeMap::from([
        ("workflow_id".into(), request.workflow.workflow_id().into()),
        ("definition_version_text".into(), "1".into()),
        (
            "trigger_operation_id".into(),
            request.trigger_operation_id.into(),
        ),
        ("trigger_kind".into(), "webhook".into()),
        ("trigger_source_id".into(), request.trigger_source_id.into()),
        (
            "trigger_context_json".into(),
            serde_json::to_string(&request.trigger_context)
                .expect("trigger JSON")
                .into(),
        ),
        ("run_version_text".into(), version.to_string().into()),
        ("status".into(), state.to_owned().into()),
        ("current_step_index".into(), 0_i16.into()),
        ("error_code".into(), Option::<String>::None.into()),
        ("error_message".into(), Option::<String>::None.into()),
        ("source_system".into(), "zed".into()),
        ("source_record_id".into(), "run:deploy:1".into()),
        ("source_version".into(), "1".into()),
        ("source_observed_at_millis".into(), NOW.into()),
        ("created_at_millis".into(), NOW.into()),
        (
            "started_at_millis".into(),
            (state != "queued").then_some(NOW + 1).into(),
        ),
        ("completed_at_millis".into(), Option::<i64>::None.into()),
        (
            "updated_at_millis".into(),
            (NOW + i64::try_from(version).expect("small version") - 1).into(),
        ),
    ])
}

fn step_row(index: i16, state: &str, updated_at: i64) -> BTreeMap<String, Value> {
    let (step_id, operation_id) = if index == 0 {
        ("announce", Uuid::from_u128(50))
    } else {
        ("approve", Uuid::from_u128(51))
    };
    BTreeMap::from([
        ("workflow_id".into(), Uuid::from_u128(10).into()),
        ("definition_version_text".into(), "1".into()),
        ("step_index".into(), index.into()),
        ("step_id".into(), step_id.into()),
        ("operation_id".into(), operation_id.into()),
        ("state".into(), state.to_owned().into()),
        (
            "attempt_count".into(),
            if state == "pending" { 0_i16 } else { 1_i16 }.into(),
        ),
        ("output_json".into(), Option::<String>::None.into()),
        ("error_code".into(), Option::<String>::None.into()),
        ("error_message".into(), Option::<String>::None.into()),
        ("source_system".into(), "zed".into()),
        (
            "source_record_id".into(),
            format!("run:{}:step:{step_id}", Uuid::from_u128(20)).into(),
        ),
        ("source_version".into(), "1".into()),
        ("source_observed_at_millis".into(), NOW.into()),
        ("created_at_millis".into(), NOW.into()),
        (
            "started_at_millis".into(),
            (state != "pending").then_some(NOW + 1).into(),
        ),
        ("completed_at_millis".into(), Option::<i64>::None.into()),
        ("updated_at_millis".into(), updated_at.into()),
    ])
}

fn lease_row(community_id: CommunityId, version: u64) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("run_id".into(), run_identity(community_id).run_id().into()),
        ("run_version_text".into(), version.to_string().into()),
        ("lease_generation_text".into(), "1".into()),
        ("lease_id".into(), Uuid::from_u128(60).into()),
        ("worker_id".into(), "worker-a".into()),
        ("state".into(), "active".into()),
        ("acquired_at_millis".into(), NOW.into()),
        ("last_heartbeat_at_millis".into(), NOW.into()),
        ("expires_at_millis".into(), (NOW + 100).into()),
        ("recovery_after_millis".into(), (NOW + 200).into()),
        ("released_at_millis".into(), Option::<i64>::None.into()),
        ("release_reason".into(), Option::<String>::None.into()),
    ])
}

fn repository(
    query_results: Vec<Vec<BTreeMap<String, Value>>>,
    affected_rows: &[u64],
) -> WorkflowRepository {
    let connection =
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
    WorkflowRepository::new(connection).expect("Postgres workflow repository")
}

fn log(repository: WorkflowRepository) -> String {
    format!("{:#?}", repository.into_connection().into_transaction_log())
}

#[tokio::test]
async fn workflow_definition_versions_are_immutable_and_reconstructable() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let write = definition_write(community_id);
    let store = repository(vec![vec![], vec![], vec![]], &[1, 1, 1, 1]);
    assert_eq!(
        store
            .store_definition(&tenant, &write)
            .await
            .expect("store definition"),
        WorkflowStoreOutcome::Applied
    );
    let store_log = log(store);
    assert!(store_log.contains("collaboration_workflow_definition_versions"));
    assert!(store_log.contains("collaboration_workflow_definition_heads"));

    let load = repository(vec![vec![definition_row(community_id)]], &[1]);
    let stored = load
        .load_definition(&tenant, write.identity, 1)
        .await
        .expect("load definition")
        .expect("stored definition");
    assert_eq!(stored.definition, write.definition);
    assert_eq!(stored.definition_version, 1);
    assert_eq!(stored.current_definition_version, 1);
    assert_eq!(stored.head_revision, 1);
    assert_eq!(stored.lifecycle, WorkflowLifecycle::Active);
}

#[tokio::test]
async fn duplicate_trigger_returns_the_existing_definition_pinned_run() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let request = run_request(community_id);
    let create = repository(
        vec![vec![], vec![definition_row(community_id)]],
        &[1, 1, 1, 1],
    );
    assert_eq!(
        create
            .claim_run(&tenant, &request)
            .await
            .expect("claim trigger"),
        WorkflowStoreOutcome::Applied
    );
    let create_log = log(create);
    assert!(create_log.contains("INSERT INTO public.collaboration_workflow_runs"));
    assert_eq!(
        create_log
            .matches("INSERT INTO public.collaboration_workflow_steps")
            .count(),
        2
    );

    let duplicate = repository(
        vec![
            vec![BTreeMap::from([(
                "run_id".into(),
                request.identity.run_id().into(),
            )])],
            vec![run_row(community_id, "queued", 1)],
            vec![step_row(0, "pending", NOW), step_row(1, "pending", NOW)],
            vec![],
        ],
        &[1],
    );
    assert_eq!(
        duplicate
            .claim_run(&tenant, &request)
            .await
            .expect("replay trigger"),
        WorkflowStoreOutcome::Duplicate
    );
    assert!(!log(duplicate).contains("INSERT INTO public.collaboration_workflow_runs"));
}

#[tokio::test]
async fn trigger_claim_rejects_a_superseded_definition_version() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let request = run_request(community_id);
    let mut superseded = definition_row(community_id);
    superseded.insert("current_definition_version_text".into(), "2".into());
    let repository = repository(vec![vec![], vec![superseded]], &[1]);

    let result = repository.claim_run(&tenant, &request).await;

    assert!(matches!(
        result,
        Err(WorkflowRepositoryError::TransitionConflict)
    ));
    assert!(!log(repository).contains("INSERT INTO public.collaboration_workflow_runs"));
}

#[tokio::test]
async fn approval_wait_is_lease_fenced_and_restarts_from_durable_state() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity = run_identity(community_id);
    let acquire = repository(
        vec![
            vec![run_row(community_id, "running", 1)],
            vec![step_row(0, "running", NOW), step_row(1, "pending", NOW)],
            vec![],
            vec![],
            vec![],
            vec![BTreeMap::from([(
                "lease_generation_text".into(),
                "0".into(),
            )])],
        ],
        &[1, 1],
    );
    let lease = acquire
        .acquire_run_lease(
            &tenant,
            &WorkflowRunLeaseRequest {
                identity,
                expected_run_version: 1,
                lease_id: Uuid::from_u128(60),
                worker_id: "worker-a".to_owned(),
                acquired_at_millis: u64::try_from(NOW).expect("timestamp"),
                expires_at_millis: u64::try_from(NOW + 100).expect("timestamp"),
                recovery_after_millis: u64::try_from(NOW + 200).expect("timestamp"),
            },
        )
        .await
        .expect("acquire run lease")
        .lease;
    let fence = WorkflowRunLeaseFence::from(&lease);
    assert!(log(acquire).contains("collaboration_workflow_run_leases"));

    let checkpoint = repository(
        vec![
            vec![lease_row(community_id, 1)],
            vec![run_row(community_id, "running", 1)],
            vec![step_row(0, "running", NOW), step_row(1, "pending", NOW)],
            vec![],
        ],
        &[1, 1, 1],
    );
    assert_eq!(
        checkpoint
            .checkpoint_step(
                &tenant,
                &WorkflowStepCheckpoint {
                    identity,
                    expected_run_version: 1,
                    step_index: 0,
                    operation_id: Uuid::from_u128(50),
                    expected_step_state: WorkflowStepState::Running,
                    next_step_state: WorkflowStepState::WaitingApproval,
                    next_run_state: WorkflowRunState::WaitingApproval,
                    next_step_index: 0,
                    attempt_count: 1,
                    output: None,
                    error_code: None,
                    error_message: None,
                    occurred_at_millis: u64::try_from(NOW + 1).expect("timestamp"),
                    lease: fence.clone(),
                },
            )
            .await
            .expect("checkpoint approval wait"),
        WorkflowStoreOutcome::Applied
    );
    let checkpoint_log = log(checkpoint);
    assert!(checkpoint_log.contains("lease_generation"));
    assert!(checkpoint_log.contains("run_version = CAST"));

    let stale = repository(vec![vec![]], &[1]);
    assert!(matches!(
        stale
            .checkpoint_step(
                &tenant,
                &WorkflowStepCheckpoint {
                    identity,
                    expected_run_version: 1,
                    step_index: 0,
                    operation_id: Uuid::from_u128(50),
                    expected_step_state: WorkflowStepState::Running,
                    next_step_state: WorkflowStepState::WaitingApproval,
                    next_run_state: WorkflowRunState::WaitingApproval,
                    next_step_index: 0,
                    attempt_count: 1,
                    output: None,
                    error_code: None,
                    error_message: None,
                    occurred_at_millis: u64::try_from(NOW + 1).expect("timestamp"),
                    lease: fence,
                },
            )
            .await,
        Err(WorkflowRepositoryError::LeaseFenceLost)
    ));
    assert!(!log(stale).contains("UPDATE public.collaboration_workflow_runs"));

    let restart = repository(
        vec![
            vec![run_row(community_id, "waiting_approval", 2)],
            vec![
                step_row(0, "waiting_approval", NOW + 1),
                step_row(1, "pending", NOW),
            ],
            vec![],
        ],
        &[1],
    );
    let restored = restart
        .load_run(&tenant, identity)
        .await
        .expect("load waiting run")
        .expect("stored waiting run");
    assert_eq!(restored.state, WorkflowRunState::WaitingApproval);
    assert_eq!(restored.steps[0].state, WorkflowStepState::WaitingApproval);
    assert_eq!(restored.run_version, 2);
}

#[tokio::test]
async fn retry_operations_are_idempotent_and_bounded() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let retry = collab::workflows::repository::WorkflowRetryWrite {
        identity: run_identity(community_id),
        step_index: 0,
        attempt_number: 2,
        retry_operation_id: Uuid::from_u128(70),
        failure_class: RetryFailureClass::Timeout,
        scheduled_at_millis: u64::try_from(NOW).expect("timestamp"),
        due_at_millis: u64::try_from(NOW + 100).expect("timestamp"),
        provenance: provenance("retry:deploy:2"),
        created_at_millis: u64::try_from(NOW).expect("timestamp"),
    };
    let first = repository(vec![vec![]], &[1, 1]);
    assert_eq!(
        first
            .record_retry(&tenant, &retry)
            .await
            .expect("record retry"),
        WorkflowStoreOutcome::Applied
    );
    let duplicate = repository(
        vec![vec![BTreeMap::from([
            ("run_id".into(), retry.identity.run_id().into()),
            ("step_index".into(), 0_i16.into()),
            ("attempt_number".into(), 2_i16.into()),
            ("failure_class".into(), "timeout".into()),
            ("state".into(), "scheduled".into()),
            ("scheduled_at_millis".into(), NOW.into()),
            ("due_at_millis".into(), (NOW + 100).into()),
            ("source_system".into(), "zed".into()),
            ("source_record_id".into(), "retry:deploy:2".into()),
            ("source_version".into(), "1".into()),
            ("source_observed_at_millis".into(), NOW.into()),
            ("created_at_millis".into(), NOW.into()),
        ])]],
        &[1],
    );
    assert_eq!(
        duplicate
            .record_retry(&tenant, &retry)
            .await
            .expect("replay retry"),
        WorkflowStoreOutcome::Duplicate
    );
}

#[tokio::test]
async fn tenant_mismatch_is_rejected_before_database_access() {
    let repository = repository(vec![], &[]);
    let result = repository
        .load_run(&tenant(community(1)), run_identity(community(2)))
        .await;
    assert!(matches!(
        result,
        Err(WorkflowRepositoryError::TenantBoundaryViolation)
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}

#[test]
fn workflow_run_lease_schema_is_fenced_and_reversible() {
    for required in [
        "PRIMARY KEY (community_id, run_id, lease_generation)",
        "UNIQUE (community_id, lease_id)",
        "run_version numeric(20, 0)",
        "WHERE state = 'active'",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE FOR ALL",
        "current_setting('app.community_id', true)",
    ] {
        assert!(
            LEASE_UP.contains(required),
            "missing lease invariant {required}"
        );
    }
    assert_eq!(
        LEASE_DOWN,
        "DROP TABLE public.collaboration_workflow_run_leases;\n"
    );
}

#[tokio::test]
async fn workflow_run_lease_migration_has_stable_reversible_checksums() {
    let migrations_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let migrations = MigrationSource::resolve(migrations_path.as_path())
        .await
        .expect("migration directory resolves");
    let lease_migrations = migrations
        .iter()
        .filter(|migration| migration.version == LEASE_VERSION)
        .collect::<Vec<_>>();
    assert_eq!(lease_migrations.len(), 2);
    let up = lease_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleUp)
        .expect("workflow lease up migration");
    let down = lease_migrations
        .iter()
        .find(|migration| migration.migration_type == MigrationType::ReversibleDown)
        .expect("workflow lease down migration");
    assert_eq!(up.checksum.as_ref(), Sha384::digest(LEASE_UP).as_slice());
    assert_eq!(
        down.checksum.as_ref(),
        Sha384::digest(LEASE_DOWN).as_slice()
    );
}
