use std::collections::BTreeMap;

use collab::workflows::{
    approval::{
        ApprovalCapability, ApprovalDecision, ApprovalDecisionWrite, ApprovalExpiryWrite,
        ApprovalOutboxKind, ApprovalRequestWrite, WorkflowApprovalDisposition,
        WorkflowApprovalError, WorkflowApprovalRepository,
    },
    repository::{
        StoredWorkflowDefinition, StoredWorkflowRun, StoredWorkflowStep, WorkflowIdentity,
        WorkflowLifecycle, WorkflowProvenance, WorkflowRunIdentity, WorkflowRunLease,
        WorkflowRunLeaseState, WorkflowRunState, WorkflowScope, WorkflowStepState,
        WorkflowTriggerKind,
    },
};
use collaboration_domain::{
    AuthenticatedPrincipal, AuthorizationScope, CommunityId, PrincipalId, PrincipalScopes,
    ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use collaboration_workflow::definition::WorkflowDefinition;
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const NOW: u64 = 1_900_000_000_000;
const APPROVAL_UP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260824000300_collaboration_workflow_approvals.up.sql"
));
const APPROVAL_DOWN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260824000300_collaboration_workflow_approvals.down.sql"
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
            TrustedTenantRoute::from_listener(community_id, "workflow-approval")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn workflow_definition() -> WorkflowDefinition {
    WorkflowDefinition::parse_yaml(
        r#"
version: 1
name: Release
enabled: true
trigger:
  on: webhook
steps:
  - id: approve
    action: request_approval
    from: owners
    message: approve release
    timeout: 1h
  - id: announce
    action: send_message
    text: released
"#,
    )
    .expect("workflow definition")
}

fn provenance(record: &str) -> WorkflowProvenance {
    WorkflowProvenance::new("zed", record, "1", NOW, None).expect("provenance")
}

fn definition(community_id: CommunityId) -> StoredWorkflowDefinition {
    let definition = workflow_definition();
    let encoded = serde_json::to_vec(&definition).expect("definition JSON");
    StoredWorkflowDefinition {
        identity: WorkflowIdentity::new(community_id, Uuid::from_u128(10)).expect("identity"),
        definition_version: 1,
        definition,
        definition_sha256: Sha256::digest(encoded).into(),
        creator_principal_id: principal(30),
        author_principal_id: principal(30),
        scope: WorkflowScope::Community,
        current_definition_version: 1,
        head_revision: 1,
        lifecycle: WorkflowLifecycle::Active,
        provenance: provenance("workflow:release:1"),
        created_at_millis: NOW - 100,
    }
}

fn step(state: WorkflowStepState) -> StoredWorkflowStep {
    StoredWorkflowStep {
        index: 0,
        step_id: "approve".to_owned(),
        operation_id: Uuid::from_u128(50),
        state,
        attempt_count: 1,
        output: None,
        error_code: None,
        error_message: None,
        created_at_millis: NOW - 100,
        started_at_millis: Some(NOW - 50),
        completed_at_millis: None,
        updated_at_millis: NOW - 50,
    }
}

fn run(community_id: CommunityId, step: StoredWorkflowStep) -> StoredWorkflowRun {
    let workflow = WorkflowIdentity::new(community_id, Uuid::from_u128(10)).expect("identity");
    StoredWorkflowRun {
        identity: WorkflowRunIdentity::new(community_id, Uuid::from_u128(20))
            .expect("run identity"),
        workflow,
        definition_version: 1,
        trigger_operation_id: Uuid::from_u128(40),
        trigger_kind: WorkflowTriggerKind::Webhook,
        trigger_source_id: "webhook:release".to_owned(),
        trigger_context: serde_json::json!({}),
        run_version: 2,
        state: WorkflowRunState::Running,
        current_step_index: 0,
        error_code: None,
        error_message: None,
        provenance: provenance("run:release:1"),
        created_at_millis: NOW - 100,
        started_at_millis: Some(NOW - 50),
        completed_at_millis: None,
        updated_at_millis: NOW - 50,
        steps: vec![step],
        retries: vec![],
    }
}

fn lease(community_id: CommunityId) -> WorkflowRunLease {
    WorkflowRunLease {
        identity: WorkflowRunIdentity::new(community_id, Uuid::from_u128(20))
            .expect("run identity"),
        admitted_run_version: 2,
        generation: 3,
        lease_id: Uuid::from_u128(60),
        worker_id: "worker-a".to_owned(),
        state: WorkflowRunLeaseState::Active,
        acquired_at_millis: NOW - 50,
        last_heartbeat_at_millis: NOW - 10,
        expires_at_millis: NOW + 10_000,
        recovery_after_millis: NOW + 20_000,
        released_at_millis: None,
        release_reason: None,
    }
}

fn actor(community_id: CommunityId, principal_id: PrincipalId) -> AuthenticatedPrincipal {
    let scope = AuthorizationScope::new("workflows:approve").expect("scope");
    AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(principal_id.as_uuid().as_u128() as u64),
        PrincipalScopes::new([scope]).expect("scopes"),
    )
}

fn repository(
    query_results: Vec<Vec<BTreeMap<String, Value>>>,
    affected_rows: &[u64],
) -> WorkflowApprovalRepository {
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
    WorkflowApprovalRepository::new(connection).expect("approval repository")
}

fn log(repository: WorkflowApprovalRepository) -> String {
    format!("{:#?}", repository.into_connection().into_transaction_log())
}

fn creation_state_row() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("run_version_text".into(), "2".into()),
        ("run_state".into(), "running".into()),
        ("step_state".into(), "running".into()),
        ("step_operation_id".into(), Uuid::from_u128(50).into()),
        ("current_definition_version_text".into(), "1".into()),
        ("lifecycle_state".into(), "active".into()),
        (
            "creator_principal_id".into(),
            principal(30).as_uuid().into(),
        ),
        ("lease_run_version_text".into(), "2".into()),
        ("lease_generation_text".into(), "3".into()),
        ("lease_id".into(), Uuid::from_u128(60).into()),
        ("worker_id".into(), "worker-a".into()),
        ("lease_state".into(), "active".into()),
        (
            "lease_expires_at_millis".into(),
            i64::try_from(NOW + 10_000).expect("timestamp").into(),
        ),
    ])
}

fn approval_row(state: &str, decision_operation_id: Option<Uuid>) -> BTreeMap<String, Value> {
    let terminal = state != "pending";
    let human_decision = matches!(state, "granted" | "denied");
    BTreeMap::from([
        ("approval_id".into(), Uuid::from_u128(70).into()),
        ("run_id".into(), Uuid::from_u128(20).into()),
        ("workflow_id".into(), Uuid::from_u128(10).into()),
        ("definition_version_text".into(), "1".into()),
        (
            "workflow_creator_principal_id".into(),
            principal(30).as_uuid().into(),
        ),
        ("step_index".into(), 0_i16.into()),
        ("step_operation_id".into(), Uuid::from_u128(50).into()),
        ("capability_sha256".into(), vec![8_u8; 32].into()),
        ("eligibility_kind".into(), "owner".into()),
        ("eligible_principal_id".into(), Option::<Uuid>::None.into()),
        ("request_message".into(), "approve release".into()),
        ("state".into(), state.to_owned().into()),
        ("decision_operation_id".into(), decision_operation_id.into()),
        (
            "decided_by_principal_id".into(),
            human_decision.then_some(principal(31).as_uuid()).into(),
        ),
        (
            "decision_note".into(),
            human_decision.then_some("approved".to_owned()).into(),
        ),
        (
            "expires_at_millis".into(),
            i64::try_from(NOW + 100_000).expect("timestamp").into(),
        ),
        (
            "created_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
        (
            "decided_at_millis".into(),
            terminal
                .then_some(i64::try_from(NOW + 10).expect("timestamp"))
                .into(),
        ),
        (
            "updated_at_millis".into(),
            i64::try_from(NOW + if terminal { 10 } else { 0 })
                .expect("timestamp")
                .into(),
        ),
    ])
}

fn membership_row(role: &str, status: &str, version: u64) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("role".into(), role.to_owned().into()),
        ("status".into(), status.to_owned().into()),
        ("membership_version_text".into(), version.to_string().into()),
    ])
}

fn waiting_state_row() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("run_version_text".into(), "3".into()),
        ("run_state".into(), "waiting_approval".into()),
        ("current_step_index".into(), 0_i16.into()),
        ("step_state".into(), "waiting_approval".into()),
        ("step_operation_id".into(), Uuid::from_u128(50).into()),
    ])
}

fn outbox_row() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("outbox_id".into(), Uuid::from_u128(80).into()),
        ("approval_id".into(), Uuid::from_u128(70).into()),
        ("run_id".into(), Uuid::from_u128(20).into()),
        ("step_index".into(), 0_i16.into()),
        ("operation_id".into(), Uuid::from_u128(70).into()),
        ("intent_kind".into(), "notify".into()),
        ("state".into(), "pending".into()),
        ("attempt_count".into(), 0_i16.into()),
        (
            "available_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
        (
            "created_at_millis".into(),
            i64::try_from(NOW).expect("timestamp").into(),
        ),
    ])
}

#[tokio::test]
async fn request_suspend_and_restart_outbox_are_durable() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let definition = definition(community_id);
    let step = step(WorkflowStepState::Running);
    let run = run(community_id, step.clone());
    let lease = lease(community_id);
    let capability = ApprovalCapability::from_bytes([7; 32]).expect("capability");
    let store = repository(
        vec![vec![], vec![creation_state_row()]],
        &[1, 1, 1, 1, 1, 1],
    );
    let (disposition, approval) = store
        .create_request(&ApprovalRequestWrite {
            tenant: &tenant,
            definition: &definition,
            run: &run,
            step: &step,
            lease: &lease,
            eligibility_spec: "owners",
            message: "approve release",
            capability: &capability,
            expires_at_millis: NOW + 100_000,
            created_at_millis: NOW,
        })
        .await
        .expect("create request");
    assert_eq!(disposition, WorkflowApprovalDisposition::Applied);
    assert_eq!(approval.workflow_creator_principal_id, principal(30));
    let store_log = log(store);
    assert!(store_log.contains("collaboration_workflow_approvals"));
    assert!(store_log.contains("status = 'waiting_approval'"));
    assert!(store_log.contains("state = 'released'"));
    assert!(store_log.contains("collaboration_workflow_approval_outbox"));

    let restarted = repository(vec![vec![outbox_row()]], &[1]);
    let pending = restarted
        .pending_outbox(&tenant, NOW + 1, 16)
        .await
        .expect("load pending outbox after restart");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].kind, ApprovalOutboxKind::Notify);
}

#[tokio::test]
async fn grant_wins_a_grant_deny_race_and_conflicting_deny_is_rejected() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let approver = actor(community_id, principal(31));
    let operation_id = Uuid::from_u128(90);
    let grant = repository(
        vec![
            vec![approval_row("pending", None)],
            vec![membership_row("owner", "active", 4)],
            vec![waiting_state_row()],
        ],
        &[1, 1, 1, 1, 1],
    );
    let (disposition, outbox) = grant
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: operation_id,
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect("grant approval");
    assert_eq!(disposition, WorkflowApprovalDisposition::Applied);
    assert_eq!(outbox.kind, ApprovalOutboxKind::Resume);
    let grant_log = log(grant);
    assert!(grant_log.contains("state = 'completed'"));
    assert!(grant_log.contains("status = 'queued'"));

    let deny = repository(
        vec![vec![approval_row("granted", Some(operation_id))]],
        &[1],
    );
    let error = deny
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(91),
            decision: ApprovalDecision::Deny,
            actor: &approver,
            note: Some("denied"),
            decided_at_millis: NOW + 11,
        })
        .await
        .expect_err("losing decision must conflict");
    assert!(matches!(error, WorkflowApprovalError::DecisionConflict));
    assert!(!log(deny).contains("UPDATE public.collaboration_workflow_approvals"));
}

#[tokio::test]
async fn stale_membership_cannot_approve() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let approver = actor(community_id, principal(31));
    let store = repository(
        vec![
            vec![approval_row("pending", None)],
            vec![membership_row("owner", "revoked", 5)],
        ],
        &[1],
    );
    let error = store
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(90),
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect_err("revoked approver must fail closed");
    assert!(matches!(error, WorkflowApprovalError::Unauthorized));
    assert!(!log(store).contains("UPDATE public.collaboration_workflow_approvals"));
}

#[tokio::test]
async fn exact_duplicate_decision_is_idempotent_without_second_transition() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let approver = actor(community_id, principal(31));
    let operation_id = Uuid::from_u128(90);
    let store = repository(
        vec![vec![approval_row("granted", Some(operation_id))]],
        &[1],
    );
    let (disposition, outbox) = store
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: operation_id,
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: Some("approved"),
            decided_at_millis: NOW + 10,
        })
        .await
        .expect("idempotent duplicate");
    assert_eq!(disposition, WorkflowApprovalDisposition::Duplicate);
    assert_eq!(outbox.kind, ApprovalOutboxKind::Resume);
    assert!(!log(store).contains("UPDATE public.collaboration_workflow_approvals"));
}

#[tokio::test]
async fn expiry_wins_once_and_fences_a_late_grant() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let expiry_operation_id = Uuid::from_u128(92);
    let expiry = repository(
        vec![
            vec![approval_row("pending", None)],
            vec![waiting_state_row()],
        ],
        &[1, 1, 1, 1, 1],
    );
    let (disposition, outbox) = expiry
        .expire(&ApprovalExpiryWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            expiry_operation_id,
            expired_at_millis: NOW + 100_000,
        })
        .await
        .expect("expire approval");
    assert_eq!(disposition, WorkflowApprovalDisposition::Applied);
    assert_eq!(outbox.kind, ApprovalOutboxKind::Cancel);
    let expiry_log = log(expiry);
    assert!(expiry_log.contains("state = 'cancelled'"));
    assert!(expiry_log.contains("status = 'cancelled'"));

    let approver = actor(community_id, principal(31));
    let late_grant = repository(
        vec![vec![approval_row("expired", Some(expiry_operation_id))]],
        &[1],
    );
    let error = late_grant
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(93),
            decision: ApprovalDecision::Grant,
            actor: &approver,
            note: None,
            decided_at_millis: NOW + 100_001,
        })
        .await
        .expect_err("expired approval fences grant");
    assert!(matches!(error, WorkflowApprovalError::DecisionConflict));
}

#[tokio::test]
async fn self_approval_and_waiting_version_mismatch_fail_closed() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let creator = actor(community_id, principal(30));
    let self_approval = repository(
        vec![
            vec![approval_row("pending", None)],
            vec![membership_row("owner", "active", 6)],
        ],
        &[1],
    );
    let error = self_approval
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(94),
            decision: ApprovalDecision::Grant,
            actor: &creator,
            note: None,
            decided_at_millis: NOW + 10,
        })
        .await
        .expect_err("creator cannot approve own workflow");
    assert!(matches!(error, WorkflowApprovalError::Unauthorized));

    let mut mismatched_waiting = waiting_state_row();
    mismatched_waiting.insert("step_operation_id".into(), Uuid::from_u128(999).into());
    let approver = actor(community_id, principal(31));
    let mismatch = repository(
        vec![
            vec![approval_row("pending", None)],
            vec![membership_row("owner", "active", 6)],
            vec![mismatched_waiting],
        ],
        &[1],
    );
    let error = mismatch
        .decide(&ApprovalDecisionWrite {
            tenant: &tenant,
            approval_id: Uuid::from_u128(70),
            decision_operation_id: Uuid::from_u128(95),
            decision: ApprovalDecision::Deny,
            actor: &approver,
            note: None,
            decided_at_millis: NOW + 10,
        })
        .await
        .expect_err("mismatched waiting checkpoint must not decide");
    assert!(matches!(error, WorkflowApprovalError::StaleRequest));
    assert!(!log(mismatch).contains("UPDATE public.collaboration_workflow_approvals"));
}

#[test]
fn approval_migration_has_tenant_isolation_and_reversible_schema() {
    assert!(APPROVAL_UP.contains("FORCE ROW LEVEL SECURITY"));
    assert!(APPROVAL_UP.contains("collaboration_workflow_approvals_community"));
    assert!(APPROVAL_UP.contains("UNIQUE (community_id, run_id, step_index)"));
    assert!(APPROVAL_UP.contains("UNIQUE (community_id, decision_operation_id)"));
    assert!(APPROVAL_UP.contains("workflow_creator_principal_id uuid NOT NULL"));
    assert!(APPROVAL_DOWN.contains("DROP TABLE public.collaboration_workflow_approval_outbox"));
    assert!(APPROVAL_DOWN.contains("DROP TABLE public.collaboration_workflow_approvals"));
}
