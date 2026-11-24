use std::fmt;

use collab::workflows::{
    approval::{ApprovalDecision, ApprovalState, StoredWorkflowApproval},
    repository::{
        RetryFailureClass, RetryState, StoredWorkflowDefinition, StoredWorkflowRetry,
        StoredWorkflowRun, WorkflowIdentity, WorkflowLifecycle, WorkflowRunIdentity,
        WorkflowRunState, WorkflowScope, WorkflowStepState,
    },
};
use collaboration_domain::{AggregateId, CommunityId, OperationId};
use collaboration_workflow::definition::WorkflowDefinition;
use serde_json::{Value, json};
use uuid::Uuid;

use super::contracts::{ErrorClass, error_contract};

const MAX_INPUT_BYTES: usize = 65_536;
const MAX_NOTE_BYTES: usize = 4_096;
const MAX_RUNS: u32 = 100;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowInputs(Value);

impl WorkflowInputs {
    pub fn new(value: Value) -> Result<Self, WorkflowsCliError> {
        if !value.is_object()
            || serde_json::to_vec(&value).map_or(true, |bytes| bytes.len() > MAX_INPUT_BYTES)
        {
            return Err(WorkflowsCliError::InvalidRequest);
        }
        Ok(Self(value))
    }

    pub fn value(&self) -> &Value {
        &self.0
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ApprovalNote(String);

impl ApprovalNote {
    pub fn new(value: impl Into<String>) -> Result<Self, WorkflowsCliError> {
        let value = value.into();
        if value.len() > MAX_NOTE_BYTES || value.chars().any(char::is_control) {
            return Err(WorkflowsCliError::InvalidRequest);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApprovalNote {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApprovalNote(<redacted>)")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: Uuid,
    pub version: u64,
}

#[derive(Clone)]
pub enum WorkflowsCliCommand {
    List {
        community_id: CommunityId,
        channel_id: AggregateId,
    },
    Get {
        identity: WorkflowIdentity,
    },
    Create {
        identity: WorkflowIdentity,
        channel_id: AggregateId,
        definition: WorkflowDefinition,
        operation_id: OperationId,
    },
    Update {
        identity: WorkflowIdentity,
        channel_id: AggregateId,
        definition: WorkflowDefinition,
        expected_head_revision: u64,
        operation_id: OperationId,
    },
    Delete {
        identity: WorkflowIdentity,
        expected_head_revision: u64,
        operation_id: OperationId,
    },
    Trigger {
        identity: WorkflowIdentity,
        inputs: WorkflowInputs,
        operation_id: OperationId,
    },
    Runs {
        identity: WorkflowIdentity,
        limit: u32,
    },
    DecideApproval {
        community_id: CommunityId,
        approval_id: Uuid,
        decision: ApprovalDecision,
        note: Option<ApprovalNote>,
        operation_id: OperationId,
    },
    RetryRun {
        identity: WorkflowRunIdentity,
        expected_run_version: u64,
        operation_id: OperationId,
    },
    CancelRun {
        identity: WorkflowRunIdentity,
        expected_run_version: u64,
        operation_id: OperationId,
    },
}

impl WorkflowsCliCommand {
    pub fn definition_from_yaml(yaml: &str) -> Result<WorkflowDefinition, WorkflowsCliError> {
        WorkflowDefinition::parse_yaml(yaml).map_err(|_| WorkflowsCliError::InvalidRequest)
    }

    const fn verb(&self) -> WorkflowsCliVerb {
        match self {
            Self::List { .. } => WorkflowsCliVerb::List,
            Self::Get { .. } => WorkflowsCliVerb::Get,
            Self::Create { .. } => WorkflowsCliVerb::Create,
            Self::Update { .. } => WorkflowsCliVerb::Update,
            Self::Delete { .. } => WorkflowsCliVerb::Delete,
            Self::Trigger { .. } => WorkflowsCliVerb::Trigger,
            Self::Runs { .. } => WorkflowsCliVerb::Runs,
            Self::DecideApproval { .. } => WorkflowsCliVerb::Approve,
            Self::RetryRun { .. } => WorkflowsCliVerb::Retry,
            Self::CancelRun { .. } => WorkflowsCliVerb::Cancel,
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::Runs { limit, .. } => (1..=MAX_RUNS).contains(limit),
            Self::DecideApproval { approval_id, .. } => !approval_id.is_nil(),
            Self::Update {
                expected_head_revision,
                ..
            }
            | Self::Delete {
                expected_head_revision,
                ..
            } => *expected_head_revision > 0,
            Self::RetryRun {
                expected_run_version,
                ..
            }
            | Self::CancelRun {
                expected_run_version,
                ..
            } => *expected_run_version > 0,
            _ => true,
        }
    }
}

impl fmt::Debug for WorkflowsCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkflowsCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkflowsCliVerb {
    List,
    Get,
    Create,
    Update,
    Delete,
    Trigger,
    Runs,
    Approve,
    Retry,
    Cancel,
}

impl WorkflowsCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::List => "workflows.list",
            Self::Get => "workflows.get",
            Self::Create => "workflows.create",
            Self::Update => "workflows.update",
            Self::Delete => "workflows.delete",
            Self::Trigger => "workflows.trigger",
            Self::Runs => "workflows.runs",
            Self::Approve => "workflows.approve",
            Self::Retry => "workflows.retry",
            Self::Cancel => "workflows.cancel",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum WorkflowsCliOutcome {
    Definition(StoredWorkflowDefinition),
    Definitions(Vec<StoredWorkflowDefinition>),
    Run(StoredWorkflowRun),
    Runs(Vec<StoredWorkflowRun>),
    Approval(StoredWorkflowApproval),
    Applied(WorkflowWriteReceipt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowsCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    PermissionDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl WorkflowsCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "workflows_cli_invalid_request",
            Self::NotFound => "workflows_cli_not_found",
            Self::Unavailable => "workflows_cli_unavailable",
            Self::PermissionDenied => "workflows_cli_permission_denied",
            Self::PartialFailure => "workflows_cli_completion_unknown",
            Self::Unexpected => "workflows_cli_unexpected_response",
            Self::Conflict => "workflows_cli_stale_version",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::PermissionDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait WorkflowsCliExecutor {
    fn execute(
        &self,
        command: WorkflowsCliCommand,
    ) -> Result<WorkflowsCliOutcome, WorkflowsCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowsCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_workflows_command(
    executor: &impl WorkflowsCliExecutor,
    command: WorkflowsCliCommand,
) -> WorkflowsCliExecution {
    let verb = command.verb();
    if !command.is_valid() {
        return error_output(verb, WorkflowsCliError::InvalidRequest);
    }
    match executor.execute(command) {
        Ok(outcome) => match success_output(verb, outcome) {
            Some(output) => WorkflowsCliExecution {
                stdout: format!("{output}\n"),
                stderr: String::new(),
                exit_code: 0,
            },
            None => error_output(verb, WorkflowsCliError::Unexpected),
        },
        Err(error) => error_output(verb, error),
    }
}

fn error_output(verb: WorkflowsCliVerb, error: WorkflowsCliError) -> WorkflowsCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    WorkflowsCliExecution {
        stdout: String::new(),
        stderr: format!(
            "{}\n",
            json!({
                "command": verb.as_str(),
                "error": contract.category,
                "error_code": diagnostic,
                "message": diagnostic,
                "ok": false,
                "retryable": contract.retryable,
            })
        ),
        exit_code: contract.exit_class as i32,
    }
}

fn success_output(verb: WorkflowsCliVerb, outcome: WorkflowsCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (WorkflowsCliVerb::Get, WorkflowsCliOutcome::Definition(definition)) => {
            Some(definition_output(verb, &definition))
        }
        (WorkflowsCliVerb::List, WorkflowsCliOutcome::Definitions(definitions)) => Some(json!({
            "command": verb.as_str(),
            "ok": true,
            "workflows": definitions.iter().map(|definition| definition_output(WorkflowsCliVerb::Get, definition)).collect::<Vec<_>>(),
        })),
        (WorkflowsCliVerb::Runs, WorkflowsCliOutcome::Runs(runs)) => Some(json!({
            "command": verb.as_str(),
            "ok": true,
            "runs": runs.iter().map(run_output).collect::<Vec<_>>(),
        })),
        (WorkflowsCliVerb::Approve, WorkflowsCliOutcome::Approval(approval)) => {
            Some(approval_output(verb, &approval))
        }
        (
            WorkflowsCliVerb::Create
            | WorkflowsCliVerb::Update
            | WorkflowsCliVerb::Delete
            | WorkflowsCliVerb::Trigger
            | WorkflowsCliVerb::Approve
            | WorkflowsCliVerb::Retry
            | WorkflowsCliVerb::Cancel,
            WorkflowsCliOutcome::Applied(receipt),
        ) => Some(receipt_output(verb, receipt)),
        _ => None,
    }
}

fn definition_output(verb: WorkflowsCliVerb, definition: &StoredWorkflowDefinition) -> Value {
    json!({
        "author_principal_id": definition.author_principal_id,
        "command": verb.as_str(),
        "created_at_millis": definition.created_at_millis,
        "creator_principal_id": definition.creator_principal_id,
        "definition": definition.definition,
        "definition_sha256": hex_hash(&definition.definition_sha256),
        "definition_version": definition.definition_version,
        "head_revision": definition.head_revision,
        "lifecycle": lifecycle_name(definition.lifecycle),
        "ok": true,
        "scope": scope_output(&definition.scope),
        "workflow_id": definition.identity.workflow_id(),
    })
}

fn run_output(run: &StoredWorkflowRun) -> Value {
    json!({
        "completed_at_millis": run.completed_at_millis,
        "current_step_index": run.current_step_index,
        "definition_version": run.definition_version,
        "error_code": run.error_code,
        "run_id": run.identity.run_id(),
        "run_version": run.run_version,
        "started_at_millis": run.started_at_millis,
        "state": run_state_name(run.state),
        "steps": run.steps.iter().map(|step| json!({
            "attempt_count": step.attempt_count,
            "completed_at_millis": step.completed_at_millis,
            "error_code": step.error_code,
            "index": step.index,
            "started_at_millis": step.started_at_millis,
            "state": step_state_name(step.state),
            "step_id": step.step_id,
        })).collect::<Vec<_>>(),
        "retries": run.retries.iter().map(retry_output).collect::<Vec<_>>(),
        "updated_at_millis": run.updated_at_millis,
        "workflow_id": run.workflow.workflow_id(),
    })
}

fn retry_output(retry: &StoredWorkflowRetry) -> Value {
    json!({
        "attempt_number": retry.attempt_number,
        "due_at_millis": retry.due_at_millis,
        "failure_class": retry_failure_name(retry.failure_class),
        "scheduled_at_millis": retry.scheduled_at_millis,
        "state": retry_state_name(retry.state),
        "step_index": retry.step_index,
    })
}

fn approval_output(verb: WorkflowsCliVerb, approval: &StoredWorkflowApproval) -> Value {
    json!({
        "approval_id": approval.approval_id,
        "command": verb.as_str(),
        "decided_at_millis": approval.decided_at_millis,
        "decided_by_principal_id": approval.decided_by_principal_id,
        "definition_version": approval.definition_version,
        "expires_at_millis": approval.expires_at_millis,
        "ok": true,
        "run_id": approval.run_id,
        "state": approval_state_name(approval.state),
        "step_index": approval.step_index,
        "workflow_id": approval.workflow_id,
    })
}

fn receipt_output(verb: WorkflowsCliVerb, receipt: WorkflowWriteReceipt) -> Value {
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": receipt.resource_id,
        "version": receipt.version,
    })
}

const fn lifecycle_name(lifecycle: WorkflowLifecycle) -> &'static str {
    match lifecycle {
        WorkflowLifecycle::Draft => "draft",
        WorkflowLifecycle::Active => "active",
        WorkflowLifecycle::Disabled => "disabled",
        WorkflowLifecycle::Archived => "archived",
    }
}

fn scope_output(scope: &WorkflowScope) -> Value {
    match scope {
        WorkflowScope::Community => json!({ "kind": "community" }),
        WorkflowScope::Project {
            signer_public_key,
            slug,
            record_version,
        } => json!({
            "kind": "project",
            "record_version": record_version,
            "signer_public_key": hex_hash(signer_public_key),
            "slug": slug,
        }),
    }
}

const fn run_state_name(state: WorkflowRunState) -> &'static str {
    match state {
        WorkflowRunState::Claimed => "claimed",
        WorkflowRunState::Queued => "queued",
        WorkflowRunState::Running => "running",
        WorkflowRunState::WaitingApproval => "waiting_approval",
        WorkflowRunState::RetryScheduled => "retry_scheduled",
        WorkflowRunState::RepairRequired => "repair_required",
        WorkflowRunState::Completed => "completed",
        WorkflowRunState::Failed => "failed",
        WorkflowRunState::Cancelled => "cancelled",
    }
}

const fn step_state_name(state: WorkflowStepState) -> &'static str {
    match state {
        WorkflowStepState::Pending => "pending",
        WorkflowStepState::Running => "running",
        WorkflowStepState::WaitingApproval => "waiting_approval",
        WorkflowStepState::RetryScheduled => "retry_scheduled",
        WorkflowStepState::RepairRequired => "repair_required",
        WorkflowStepState::Completed => "completed",
        WorkflowStepState::Skipped => "skipped",
        WorkflowStepState::Failed => "failed",
        WorkflowStepState::Cancelled => "cancelled",
    }
}

const fn retry_failure_name(failure: RetryFailureClass) -> &'static str {
    match failure {
        RetryFailureClass::RateLimited => "rate_limited",
        RetryFailureClass::TemporaryUnavailable => "temporary_unavailable",
        RetryFailureClass::Timeout => "timeout",
        RetryFailureClass::Transport => "transport",
    }
}

const fn retry_state_name(state: RetryState) -> &'static str {
    match state {
        RetryState::Scheduled => "scheduled",
        RetryState::Claimed => "claimed",
        RetryState::Completed => "completed",
        RetryState::Exhausted => "exhausted",
        RetryState::Cancelled => "cancelled",
    }
}

const fn approval_state_name(state: ApprovalState) -> &'static str {
    match state {
        ApprovalState::Pending => "pending",
        ApprovalState::Granted => "granted",
        ApprovalState::Denied => "denied",
        ApprovalState::Expired => "expired",
        ApprovalState::Cancelled => "cancelled",
    }
}

fn hex_hash(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use collab::workflows::repository::{
        StoredWorkflowStep, WorkflowProvenance, WorkflowTriggerKind,
    };
    use collaboration_domain::PrincipalId;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<WorkflowsCliCommand>>,
        result: RefCell<Option<Result<WorkflowsCliOutcome, WorkflowsCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<WorkflowsCliOutcome, WorkflowsCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl WorkflowsCliExecutor for TestExecutor {
        fn execute(
            &self,
            command: WorkflowsCliCommand,
        ) -> Result<WorkflowsCliOutcome, WorkflowsCliError> {
            self.command.replace(Some(command));
            self.result.borrow_mut().take().expect("called once")
        }
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(2))
    }

    fn workflow_identity() -> WorkflowIdentity {
        WorkflowIdentity::new(community_id(), Uuid::from_u128(3)).expect("workflow")
    }

    fn run_identity() -> WorkflowRunIdentity {
        WorkflowRunIdentity::new(community_id(), Uuid::from_u128(4)).expect("run")
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn definition() -> WorkflowDefinition {
        WorkflowsCliCommand::definition_from_yaml(
            "version: 1\nname: test\ntrigger:\n  on: schedule\n  cron: '0 9 * * 1-5'\nsteps:\n  - id: done\n    action: delay\n    duration: 1s\n",
        )
        .expect("definition")
    }

    fn provenance() -> WorkflowProvenance {
        WorkflowProvenance::new("zed", "workflow", "1", 10, None).expect("provenance")
    }

    fn stored_definition() -> StoredWorkflowDefinition {
        StoredWorkflowDefinition {
            identity: workflow_identity(),
            definition_version: 1,
            definition: definition(),
            definition_sha256: [5; 32],
            creator_principal_id: principal_id(6),
            author_principal_id: principal_id(6),
            scope: WorkflowScope::Community,
            current_definition_version: 1,
            head_revision: 1,
            lifecycle: WorkflowLifecycle::Active,
            provenance: provenance(),
            created_at_millis: 10,
        }
    }

    fn stored_run(state: WorkflowRunState) -> StoredWorkflowRun {
        StoredWorkflowRun {
            identity: run_identity(),
            workflow: workflow_identity(),
            definition_version: 1,
            trigger_operation_id: Uuid::from_u128(7),
            trigger_kind: WorkflowTriggerKind::Manual,
            trigger_source_id: "cli".into(),
            trigger_context: json!({ "private": "not-output" }),
            run_version: 3,
            state,
            current_step_index: 0,
            error_code: Some("temporary_unavailable".into()),
            error_message: Some("PRIVATE-PROVIDER-DETAIL".into()),
            provenance: provenance(),
            created_at_millis: 10,
            started_at_millis: Some(11),
            completed_at_millis: None,
            updated_at_millis: 12,
            steps: vec![StoredWorkflowStep {
                index: 0,
                step_id: "done".into(),
                operation_id: Uuid::from_u128(8),
                state: WorkflowStepState::RetryScheduled,
                attempt_count: 1,
                output: Some(json!({ "private": "not-output" })),
                error_code: Some("temporary_unavailable".into()),
                error_message: Some("PRIVATE-STEP-DETAIL".into()),
                created_at_millis: 10,
                started_at_millis: Some(11),
                completed_at_millis: None,
                updated_at_millis: 12,
            }],
            retries: vec![StoredWorkflowRetry {
                step_index: 0,
                attempt_number: 2,
                retry_operation_id: Uuid::from_u128(9),
                failure_class: RetryFailureClass::TemporaryUnavailable,
                state: RetryState::Scheduled,
                scheduled_at_millis: 12,
                due_at_millis: 20,
                claimed_at_millis: None,
                completed_at_millis: None,
            }],
        }
    }

    fn receipt(version: u64) -> WorkflowWriteReceipt {
        WorkflowWriteReceipt {
            operation_id: operation_id(),
            resource_id: workflow_identity().workflow_id(),
            version,
        }
    }

    #[test]
    fn definitions_use_canonical_parser_and_stable_projection() {
        let executor =
            TestExecutor::returning(Ok(WorkflowsCliOutcome::Definition(stored_definition())));
        let output = execute_workflows_command(
            &executor,
            WorkflowsCliCommand::Get {
                identity: workflow_identity(),
            },
        );
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("\"name\":\"test\""));
        assert!(output.stdout.contains(&"05".repeat(32)));
        assert!(WorkflowsCliCommand::definition_from_yaml("steps: []").is_err());
    }

    #[test]
    fn trigger_forwards_bounded_object_and_operation() {
        let inputs = WorkflowInputs::new(json!({ "branch": "main" })).expect("inputs");
        let command = WorkflowsCliCommand::Trigger {
            identity: workflow_identity(),
            inputs: inputs.clone(),
            operation_id: operation_id(),
        };
        let executor = TestExecutor::returning(Ok(WorkflowsCliOutcome::Applied(receipt(1))));
        let output = execute_workflows_command(&executor, command);
        assert_eq!(output.exit_code, 0);
        assert!(matches!(
            executor.command.take(),
            Some(WorkflowsCliCommand::Trigger { inputs: actual, operation_id: actual_operation, .. })
                if actual == inputs && actual_operation == operation_id()
        ));
        assert!(WorkflowInputs::new(json!(["not", "an", "object"])).is_err());
    }

    #[test]
    fn approval_grant_and_deny_preserve_exact_decision_without_token_output() {
        for decision in [ApprovalDecision::Grant, ApprovalDecision::Deny] {
            let command = WorkflowsCliCommand::DecideApproval {
                community_id: community_id(),
                approval_id: Uuid::from_u128(10),
                decision,
                note: Some(ApprovalNote::new("reviewed").expect("note")),
                operation_id: operation_id(),
            };
            let executor = TestExecutor::returning(Ok(WorkflowsCliOutcome::Applied(receipt(2))));
            let output = execute_workflows_command(&executor, command);
            assert_eq!(output.exit_code, 0);
            assert!(!output.stdout.contains("capability"));
            assert!(matches!(
                executor.command.take(),
                Some(WorkflowsCliCommand::DecideApproval { decision: actual, .. }) if actual == decision
            ));
        }
    }

    #[test]
    fn waiting_retry_failure_and_cancellation_are_distinct_and_redacted() {
        for (state, expected) in [
            (WorkflowRunState::WaitingApproval, "waiting_approval"),
            (WorkflowRunState::RetryScheduled, "retry_scheduled"),
            (WorkflowRunState::Failed, "failed"),
            (WorkflowRunState::Cancelled, "cancelled"),
        ] {
            let output = execute_workflows_command(
                &TestExecutor::returning(Ok(WorkflowsCliOutcome::Runs(vec![stored_run(state)]))),
                WorkflowsCliCommand::Runs {
                    identity: workflow_identity(),
                    limit: 20,
                },
            );
            assert!(output.stdout.contains(expected));
            assert!(output.stdout.contains("temporary_unavailable"));
            assert!(!output.stdout.contains("PRIVATE-PROVIDER-DETAIL"));
            assert!(!output.stdout.contains("PRIVATE-STEP-DETAIL"));
            assert!(!output.stdout.contains("not-output"));
        }
    }

    #[test]
    fn stable_errors_invalid_requests_and_mismatched_outcomes_fail_closed() {
        let invalid = execute_workflows_command(
            &TestExecutor::returning(Ok(WorkflowsCliOutcome::Runs(Vec::new()))),
            WorkflowsCliCommand::Runs {
                identity: workflow_identity(),
                limit: 0,
            },
        );
        assert_eq!(invalid.exit_code, 1);

        let cases = [
            (WorkflowsCliError::InvalidRequest, "user_error", 1, false),
            (WorkflowsCliError::NotFound, "not_found", 1, false),
            (WorkflowsCliError::Unavailable, "network_error", 2, true),
            (
                WorkflowsCliError::PartialFailure,
                "delivery_unknown",
                2,
                false,
            ),
            (WorkflowsCliError::PermissionDenied, "auth_error", 3, false),
            (WorkflowsCliError::Unexpected, "error", 4, false),
            (WorkflowsCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_workflows_command(
                &TestExecutor::returning(Err(error)),
                WorkflowsCliCommand::Get {
                    identity: workflow_identity(),
                },
            );
            assert_eq!(output.exit_code, exit_code);
            let envelope: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(envelope["error"], category);
            assert_eq!(envelope["retryable"], retryable);
        }

        let mismatch = execute_workflows_command(
            &TestExecutor::returning(Ok(WorkflowsCliOutcome::Applied(receipt(1)))),
            WorkflowsCliCommand::Get {
                identity: workflow_identity(),
            },
        );
        assert_eq!(mismatch.exit_code, 4);
    }
}
