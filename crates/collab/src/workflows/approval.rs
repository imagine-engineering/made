use std::{fmt, str::FromStr};

use collaboration_domain::{
    AuthenticatedPrincipal, AuthenticatedPrincipalKind, AuthorizationScope, CommunityId,
    PrincipalId, TenantContext,
};
use rand::RngCore;
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, ExecResult,
    QueryResult, Statement, TransactionTrait,
};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::repository::{
    StoredWorkflowDefinition, StoredWorkflowRun, StoredWorkflowStep, WorkflowLifecycle,
    WorkflowRunLease, WorkflowRunLeaseState, WorkflowRunState, WorkflowStepState,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const SELECT_APPROVAL_BY_STEP_SQL: &str = r#"
SELECT approval_id, run_id, workflow_id,
       definition_version::text AS definition_version_text,
       workflow_creator_principal_id,
       step_index, step_operation_id, capability_sha256,
       eligibility_kind, eligible_principal_id, request_message, state,
       decision_operation_id, decided_by_principal_id, decision_note,
       floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
       floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis,
       floor(extract(epoch FROM decided_at) * 1000)::bigint AS decided_at_millis,
       floor(extract(epoch FROM updated_at) * 1000)::bigint AS updated_at_millis
FROM public.collaboration_workflow_approvals
WHERE community_id = $1 AND run_id = $2 AND step_index = $3
FOR UPDATE
"#;
const SELECT_APPROVAL_BY_ID_SQL: &str = r#"
SELECT approval_id, run_id, workflow_id,
       definition_version::text AS definition_version_text,
       workflow_creator_principal_id,
       step_index, step_operation_id, capability_sha256,
       eligibility_kind, eligible_principal_id, request_message, state,
       decision_operation_id, decided_by_principal_id, decision_note,
       floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
       floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis,
       floor(extract(epoch FROM decided_at) * 1000)::bigint AS decided_at_millis,
       floor(extract(epoch FROM updated_at) * 1000)::bigint AS updated_at_millis
FROM public.collaboration_workflow_approvals
WHERE community_id = $1 AND approval_id = $2
FOR UPDATE
"#;
const SELECT_CREATION_STATE_SQL: &str = r#"
SELECT run.run_version::text AS run_version_text,
       run.status AS run_state,
       step.state AS step_state,
       step.operation_id AS step_operation_id,
       definition.current_definition_version::text AS current_definition_version_text,
       definition.lifecycle_state,
       definition.creator_principal_id,
       lease.run_version::text AS lease_run_version_text,
       lease.lease_generation::text AS lease_generation_text,
       lease.lease_id, lease.worker_id, lease.state AS lease_state,
       floor(extract(epoch FROM lease.expires_at) * 1000)::bigint
           AS lease_expires_at_millis
FROM public.collaboration_workflow_runs AS run
JOIN public.collaboration_workflow_steps AS step
  ON step.community_id = run.community_id AND step.run_id = run.run_id
JOIN public.collaboration_workflow_definition_heads AS definition
  ON definition.community_id = run.community_id
 AND definition.workflow_id = run.workflow_id
JOIN public.collaboration_workflow_run_leases AS lease
  ON lease.community_id = run.community_id AND lease.run_id = run.run_id
WHERE run.community_id = $1 AND run.run_id = $2
  AND step.step_index = $3 AND lease.lease_generation = CAST($4 AS numeric)
FOR UPDATE OF run, step, definition, lease
"#;
const INSERT_APPROVAL_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_approvals (
    community_id, approval_id, run_id, workflow_id, definition_version,
    workflow_creator_principal_id,
    step_index, step_operation_id, capability_sha256,
    eligibility_kind, eligible_principal_id, request_message, state,
    expires_at, created_at, updated_at
) VALUES (
    $1, $2, $3, $4, CAST($5 AS numeric), $6, $7, $8, $9,
    $10, $11, $12, 'pending',
    to_timestamp($13::double precision / 1000),
    to_timestamp($14::double precision / 1000),
    to_timestamp($14::double precision / 1000)
)
"#;
const UPDATE_STEP_WAITING_SQL: &str = r#"
UPDATE public.collaboration_workflow_steps
SET state = 'waiting_approval', attempt_count = $5,
    updated_at = to_timestamp($6::double precision / 1000)
WHERE community_id = $1 AND run_id = $2 AND step_index = $3
  AND operation_id = $4 AND state = 'running'
"#;
const UPDATE_RUN_WAITING_SQL: &str = r#"
UPDATE public.collaboration_workflow_runs
SET run_version = run_version + 1, status = 'waiting_approval',
    current_step_index = $4,
    updated_at = to_timestamp($5::double precision / 1000)
WHERE community_id = $1 AND run_id = $2
  AND run_version = CAST($3 AS numeric) AND status = 'running'
"#;
const RELEASE_LEASE_WAITING_SQL: &str = r#"
UPDATE public.collaboration_workflow_run_leases
SET state = 'released', released_at = to_timestamp($7::double precision / 1000),
    release_reason = 'replaced'
WHERE community_id = $1 AND run_id = $2
  AND run_version = CAST($3 AS numeric)
  AND lease_generation = CAST($4 AS numeric)
  AND lease_id = $5 AND worker_id = $6 AND state = 'active'
"#;
const INSERT_OUTBOX_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_approval_outbox (
    community_id, outbox_id, approval_id, run_id, step_index,
    operation_id, intent_kind, state, available_at, created_at, updated_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, 'pending',
    to_timestamp($8::double precision / 1000),
    to_timestamp($8::double precision / 1000),
    to_timestamp($8::double precision / 1000)
)
"#;
const SELECT_MEMBERSHIP_SQL: &str = r#"
SELECT role, status, membership_version::text AS membership_version_text
FROM public.collaboration_community_memberships
WHERE community_id = $1 AND principal_id = $2
FOR SHARE
"#;
const SELECT_WAITING_STATE_SQL: &str = r#"
SELECT run.run_version::text AS run_version_text,
       run.status AS run_state, run.current_step_index,
       step.state AS step_state, step.operation_id AS step_operation_id
FROM public.collaboration_workflow_runs AS run
JOIN public.collaboration_workflow_steps AS step
  ON step.community_id = run.community_id AND step.run_id = run.run_id
WHERE run.community_id = $1 AND run.run_id = $2 AND step.step_index = $3
FOR UPDATE OF run, step
"#;
const UPDATE_APPROVAL_DECISION_SQL: &str = r#"
UPDATE public.collaboration_workflow_approvals
SET state = $3, decision_operation_id = $4,
    decided_by_principal_id = $5, decision_note = $6,
    decided_at = to_timestamp($7::double precision / 1000),
    updated_at = to_timestamp($7::double precision / 1000)
WHERE community_id = $1 AND approval_id = $2 AND state = 'pending'
"#;
const UPDATE_STEP_GRANTED_SQL: &str = r#"
UPDATE public.collaboration_workflow_steps
SET state = 'completed', output = CAST($5 AS jsonb),
    completed_at = to_timestamp($6::double precision / 1000),
    updated_at = to_timestamp($6::double precision / 1000)
WHERE community_id = $1 AND run_id = $2 AND step_index = $3
  AND operation_id = $4 AND state = 'waiting_approval'
"#;
const UPDATE_STEP_DENIED_SQL: &str = r#"
UPDATE public.collaboration_workflow_steps
SET state = 'cancelled',
    completed_at = to_timestamp($5::double precision / 1000),
    updated_at = to_timestamp($5::double precision / 1000)
WHERE community_id = $1 AND run_id = $2 AND step_index = $3
  AND operation_id = $4 AND state = 'waiting_approval'
"#;
const UPDATE_RUN_GRANTED_SQL: &str = r#"
UPDATE public.collaboration_workflow_runs
SET run_version = run_version + 1, status = 'queued',
    current_step_index = $4,
    updated_at = to_timestamp($5::double precision / 1000)
WHERE community_id = $1 AND run_id = $2
  AND run_version = CAST($3 AS numeric) AND status = 'waiting_approval'
"#;
const UPDATE_RUN_DENIED_SQL: &str = r#"
UPDATE public.collaboration_workflow_runs
SET run_version = run_version + 1, status = 'cancelled',
    completed_at = to_timestamp($4::double precision / 1000),
    updated_at = to_timestamp($4::double precision / 1000)
WHERE community_id = $1 AND run_id = $2
  AND run_version = CAST($3 AS numeric) AND status = 'waiting_approval'
"#;
const SELECT_PENDING_OUTBOX_SQL: &str = r#"
SELECT outbox_id, approval_id, run_id, step_index, operation_id,
       intent_kind, state, attempt_count,
       floor(extract(epoch FROM available_at) * 1000)::bigint AS available_at_millis,
       floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis
FROM public.collaboration_workflow_approval_outbox
WHERE community_id = $1 AND state = 'pending'
  AND available_at <= to_timestamp($2::double precision / 1000)
ORDER BY available_at, created_at, outbox_id
LIMIT $3
"#;

const APPROVAL_NAMESPACE: Uuid = Uuid::from_u128(0xa451_7ae1_9e40_5d9f_b56d_6506_29af_3141);
const OUTBOX_NAMESPACE: Uuid = Uuid::from_u128(0xbb18_68a5_82e5_579e_9c08_3cf3_da32_568e);
const MAX_APPROVAL_MESSAGE_BYTES: usize = 16 * 1024;
const MAX_DECISION_NOTE_BYTES: usize = 4 * 1024;
const MAX_OUTBOX_BATCH: u16 = 256;
const MAX_APPROVAL_LIFETIME_MILLIS: u64 = 30 * 24 * 60 * 60 * 1000;
const WORKFLOW_APPROVE_SCOPE: &str = "workflows:approve";

pub struct ApprovalCapability([u8; 32]);

impl ApprovalCapability {
    pub fn generate() -> Self {
        let mut value = [0_u8; 32];
        rand::rng().fill_bytes(&mut value);
        Self(value)
    }

    pub fn from_bytes(value: [u8; 32]) -> Result<Self, WorkflowApprovalError> {
        if value == [0; 32] {
            return Err(WorkflowApprovalError::InvalidInput);
        }
        Ok(Self(value))
    }

    fn sha256(&self) -> [u8; 32] {
        Sha256::digest(self.0).into()
    }
}

impl fmt::Debug for ApprovalCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApprovalCapability([REDACTED])")
    }
}

impl Drop for ApprovalCapability {
    fn drop(&mut self) {
        self.0.fill(0);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalEligibility {
    AnyMember,
    Owner,
    Administrator,
    Principal(PrincipalId),
}

impl ApprovalEligibility {
    pub fn parse(value: &str) -> Result<Self, WorkflowApprovalError> {
        let value = value.trim();
        match value {
            "any" | "any_member" => Ok(Self::AnyMember),
            "owner" | "owners" | "@owner" | "@owners" => Ok(Self::Owner),
            "admin" | "admins" | "administrator" | "administrators" | "@admin" | "@admins"
            | "@administrator" | "@administrators" => Ok(Self::Administrator),
            _ => {
                let value = value.strip_prefix("principal:").unwrap_or(value);
                let id = Uuid::from_str(value).map_err(|_| WorkflowApprovalError::InvalidInput)?;
                if id.is_nil() || id.to_string() != value {
                    return Err(WorkflowApprovalError::InvalidInput);
                }
                Ok(Self::Principal(PrincipalId::from_uuid(id)))
            }
        }
    }

    fn database_kind(self) -> &'static str {
        match self {
            Self::AnyMember => "any_member",
            Self::Owner => "owner",
            Self::Administrator => "administrator",
            Self::Principal(_) => "principal",
        }
    }

    fn principal_id(self) -> Option<PrincipalId> {
        match self {
            Self::Principal(principal_id) => Some(principal_id),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalState {
    Pending,
    Granted,
    Denied,
    Expired,
    Cancelled,
}

impl ApprovalState {
    fn database_name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Granted => "granted",
            Self::Denied => "denied",
            Self::Expired => "expired",
            Self::Cancelled => "cancelled",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowApprovalError> {
        match value {
            "pending" => Ok(Self::Pending),
            "granted" => Ok(Self::Granted),
            "denied" => Ok(Self::Denied),
            "expired" => Ok(Self::Expired),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(WorkflowApprovalError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredWorkflowApproval {
    pub community_id: CommunityId,
    pub approval_id: Uuid,
    pub run_id: Uuid,
    pub workflow_id: Uuid,
    pub definition_version: u64,
    pub workflow_creator_principal_id: PrincipalId,
    pub step_index: u16,
    pub step_operation_id: Uuid,
    pub capability_sha256: [u8; 32],
    pub eligibility: ApprovalEligibility,
    pub request_message: String,
    pub state: ApprovalState,
    pub decision_operation_id: Option<Uuid>,
    pub decided_by_principal_id: Option<PrincipalId>,
    pub decision_note: Option<String>,
    pub expires_at_millis: u64,
    pub created_at_millis: u64,
    pub decided_at_millis: Option<u64>,
    pub updated_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalOutboxKind {
    Notify,
    Resume,
    Cancel,
}

impl ApprovalOutboxKind {
    fn database_name(self) -> &'static str {
        match self {
            Self::Notify => "notify",
            Self::Resume => "resume",
            Self::Cancel => "cancel",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowApprovalError> {
        match value {
            "notify" => Ok(Self::Notify),
            "resume" => Ok(Self::Resume),
            "cancel" => Ok(Self::Cancel),
            _ => Err(WorkflowApprovalError::InvalidRecord),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalOutboxState {
    Pending,
    Claimed,
    Completed,
    Failed,
}

impl ApprovalOutboxState {
    fn from_database(value: &str) -> Result<Self, WorkflowApprovalError> {
        match value {
            "pending" => Ok(Self::Pending),
            "claimed" => Ok(Self::Claimed),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            _ => Err(WorkflowApprovalError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowApprovalOutboxItem {
    pub outbox_id: Uuid,
    pub approval_id: Uuid,
    pub run_id: Uuid,
    pub step_index: u16,
    pub operation_id: Uuid,
    pub kind: ApprovalOutboxKind,
    pub state: ApprovalOutboxState,
    pub attempt_count: u16,
    pub available_at_millis: u64,
    pub created_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowApprovalDisposition {
    Applied,
    Duplicate,
}

pub struct ApprovalRequestWrite<'a> {
    pub tenant: &'a TenantContext,
    pub definition: &'a StoredWorkflowDefinition,
    pub run: &'a StoredWorkflowRun,
    pub step: &'a StoredWorkflowStep,
    pub lease: &'a WorkflowRunLease,
    pub eligibility_spec: &'a str,
    pub message: &'a str,
    pub capability: &'a ApprovalCapability,
    pub expires_at_millis: u64,
    pub created_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApprovalDecision {
    Grant,
    Deny,
}

impl ApprovalDecision {
    fn approval_state(self) -> ApprovalState {
        match self {
            Self::Grant => ApprovalState::Granted,
            Self::Deny => ApprovalState::Denied,
        }
    }

    fn outbox_kind(self) -> ApprovalOutboxKind {
        match self {
            Self::Grant => ApprovalOutboxKind::Resume,
            Self::Deny => ApprovalOutboxKind::Cancel,
        }
    }
}

pub struct ApprovalDecisionWrite<'a> {
    pub tenant: &'a TenantContext,
    pub approval_id: Uuid,
    pub decision_operation_id: Uuid,
    pub decision: ApprovalDecision,
    pub actor: &'a AuthenticatedPrincipal,
    pub note: Option<&'a str>,
    pub decided_at_millis: u64,
}

pub struct ApprovalExpiryWrite<'a> {
    pub tenant: &'a TenantContext,
    pub approval_id: Uuid,
    pub expiry_operation_id: Uuid,
    pub expired_at_millis: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkflowApprovalError {
    #[error("workflow approval repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("workflow approval request is invalid")]
    InvalidInput,
    #[error("workflow approval record is invalid")]
    InvalidRecord,
    #[error("workflow approval request is stale")]
    StaleRequest,
    #[error("workflow approval decision is not authorized")]
    Unauthorized,
    #[error("workflow approval has already been decided")]
    DecisionConflict,
    #[error("workflow approval repository is unavailable")]
    Unavailable(#[source] sea_orm::DbErr),
}

pub struct WorkflowApprovalRepository {
    connection: DatabaseConnection,
}

impl WorkflowApprovalRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, WorkflowApprovalError> {
        if connection.get_database_backend() != DatabaseBackend::Postgres {
            return Err(WorkflowApprovalError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn create_request(
        &self,
        write: &ApprovalRequestWrite<'_>,
    ) -> Result<(WorkflowApprovalDisposition, StoredWorkflowApproval), WorkflowApprovalError> {
        let eligibility = validate_request_write(write)?;
        let approval_id = approval_id(write);
        let capability_sha256 = write.capability.sha256();
        let transaction = self.begin().await?;
        if let Err(error) = set_tenant(&transaction, write.tenant.community_id()).await {
            return finish_transaction(transaction, Err(error)).await;
        }
        let result = async {
            if let Some(existing) = query_approval_by_step(
                &transaction,
                write.tenant.community_id(),
                write.run.identity.run_id(),
                write.step.index,
            )
            .await?
            {
                if exact_request_replay(
                    &existing,
                    write,
                    approval_id,
                    capability_sha256,
                    eligibility,
                ) {
                    return Ok((WorkflowApprovalDisposition::Duplicate, existing));
                }
                return Err(WorkflowApprovalError::DecisionConflict);
            }

            let state = query_creation_state(&transaction, write).await?;
            validate_creation_state(&state, write)?;
            require_one(
                transaction
                    .execute(insert_approval_statement(
                        write,
                        approval_id,
                        capability_sha256,
                        eligibility,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            require_one(
                transaction
                    .execute(update_step_waiting_statement(write)?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            require_one(
                transaction
                    .execute(update_run_waiting_statement(write)?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            require_one(
                transaction
                    .execute(release_lease_statement(write)?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            let notification_operation_id = approval_id;
            require_one(
                transaction
                    .execute(insert_outbox_statement(
                        write.tenant.community_id(),
                        approval_id,
                        write.run.identity.run_id(),
                        write.step.index,
                        notification_operation_id,
                        ApprovalOutboxKind::Notify,
                        write.created_at_millis,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            Ok((
                WorkflowApprovalDisposition::Applied,
                StoredWorkflowApproval {
                    community_id: write.tenant.community_id(),
                    approval_id,
                    run_id: write.run.identity.run_id(),
                    workflow_id: write.definition.identity.workflow_id(),
                    definition_version: write.definition.definition_version,
                    workflow_creator_principal_id: write.definition.creator_principal_id,
                    step_index: write.step.index,
                    step_operation_id: write.step.operation_id,
                    capability_sha256,
                    eligibility,
                    request_message: write.message.to_owned(),
                    state: ApprovalState::Pending,
                    decision_operation_id: None,
                    decided_by_principal_id: None,
                    decision_note: None,
                    expires_at_millis: write.expires_at_millis,
                    created_at_millis: write.created_at_millis,
                    decided_at_millis: None,
                    updated_at_millis: write.created_at_millis,
                },
            ))
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn decide(
        &self,
        write: &ApprovalDecisionWrite<'_>,
    ) -> Result<(WorkflowApprovalDisposition, WorkflowApprovalOutboxItem), WorkflowApprovalError>
    {
        let actor_principal_id = validate_decision_write(write)?;
        let transaction = self.begin().await?;
        if let Err(error) = set_tenant(&transaction, write.tenant.community_id()).await {
            return finish_transaction(transaction, Err(error)).await;
        }
        let result = async {
            let approval =
                query_approval_by_id(&transaction, write.tenant.community_id(), write.approval_id)
                    .await?
                    .ok_or(WorkflowApprovalError::StaleRequest)?;
            let target_state = write.decision.approval_state();
            let outbox_kind = write.decision.outbox_kind();
            if approval.state != ApprovalState::Pending {
                if approval.state == target_state
                    && approval.decision_operation_id == Some(write.decision_operation_id)
                    && approval.decided_by_principal_id == Some(actor_principal_id)
                    && approval.decision_note.as_deref() == write.note
                {
                    return Ok((
                        WorkflowApprovalDisposition::Duplicate,
                        decision_outbox_item(&approval, write, outbox_kind),
                    ));
                }
                return Err(WorkflowApprovalError::DecisionConflict);
            }
            if write.decided_at_millis >= approval.expires_at_millis {
                return Err(WorkflowApprovalError::StaleRequest);
            }
            let membership = query_membership(
                &transaction,
                write.tenant.community_id(),
                actor_principal_id,
            )
            .await?
            .ok_or(WorkflowApprovalError::Unauthorized)?;
            authorize_approver(&approval, actor_principal_id, &membership)?;
            let waiting = query_waiting_state(&transaction, &approval)
                .await?
                .ok_or(WorkflowApprovalError::StaleRequest)?;
            validate_waiting_state(&waiting, &approval)?;

            require_one(
                transaction
                    .execute(update_approval_decision_statement(
                        &approval,
                        write,
                        actor_principal_id,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            let output = serde_json::to_string(&serde_json::json!({
                "approval_id": approval.approval_id,
                "decision": target_state.database_name(),
                "decided_by_principal_id": actor_principal_id,
            }))
            .map_err(|_| WorkflowApprovalError::InvalidInput)?;
            let next_step_index = approval
                .step_index
                .checked_add(1)
                .ok_or(WorkflowApprovalError::InvalidInput)?;
            match write.decision {
                ApprovalDecision::Grant => {
                    require_one(
                        transaction
                            .execute(update_step_granted_statement(
                                &approval,
                                &output,
                                write.decided_at_millis,
                            )?)
                            .await
                            .map_err(WorkflowApprovalError::Unavailable)?,
                    )?;
                    require_one(
                        transaction
                            .execute(update_run_granted_statement(
                                &approval,
                                waiting.run_version,
                                next_step_index,
                                write.decided_at_millis,
                            )?)
                            .await
                            .map_err(WorkflowApprovalError::Unavailable)?,
                    )?;
                }
                ApprovalDecision::Deny => {
                    require_one(
                        transaction
                            .execute(update_step_denied_statement(
                                &approval,
                                write.decided_at_millis,
                            )?)
                            .await
                            .map_err(WorkflowApprovalError::Unavailable)?,
                    )?;
                    require_one(
                        transaction
                            .execute(update_run_denied_statement(
                                &approval,
                                waiting.run_version,
                                write.decided_at_millis,
                            )?)
                            .await
                            .map_err(WorkflowApprovalError::Unavailable)?,
                    )?;
                }
            }
            require_one(
                transaction
                    .execute(insert_outbox_statement(
                        approval.community_id,
                        approval.approval_id,
                        approval.run_id,
                        approval.step_index,
                        write.decision_operation_id,
                        outbox_kind,
                        write.decided_at_millis,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            Ok((
                WorkflowApprovalDisposition::Applied,
                decision_outbox_item(&approval, write, outbox_kind),
            ))
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn expire(
        &self,
        write: &ApprovalExpiryWrite<'_>,
    ) -> Result<(WorkflowApprovalDisposition, WorkflowApprovalOutboxItem), WorkflowApprovalError>
    {
        if write.approval_id.is_nil()
            || write.expiry_operation_id.is_nil()
            || write.expiry_operation_id == write.approval_id
            || write.expired_at_millis == 0
        {
            return Err(WorkflowApprovalError::InvalidInput);
        }
        let transaction = self.begin().await?;
        if let Err(error) = set_tenant(&transaction, write.tenant.community_id()).await {
            return finish_transaction(transaction, Err(error)).await;
        }
        let result = async {
            let approval =
                query_approval_by_id(&transaction, write.tenant.community_id(), write.approval_id)
                    .await?
                    .ok_or(WorkflowApprovalError::StaleRequest)?;
            if approval.state != ApprovalState::Pending {
                if approval.state == ApprovalState::Expired
                    && approval.decision_operation_id == Some(write.expiry_operation_id)
                    && approval.decided_by_principal_id.is_none()
                {
                    return Ok((
                        WorkflowApprovalDisposition::Duplicate,
                        outbox_item(
                            &approval,
                            write.expiry_operation_id,
                            ApprovalOutboxKind::Cancel,
                            approval
                                .decided_at_millis
                                .unwrap_or(write.expired_at_millis),
                        ),
                    ));
                }
                return Err(WorkflowApprovalError::DecisionConflict);
            }
            if write.expired_at_millis < approval.expires_at_millis {
                return Err(WorkflowApprovalError::StaleRequest);
            }
            let waiting = query_waiting_state(&transaction, &approval)
                .await?
                .ok_or(WorkflowApprovalError::StaleRequest)?;
            validate_waiting_state(&waiting, &approval)?;
            require_one(
                transaction
                    .execute(update_approval_expiry_statement(&approval, write)?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            require_one(
                transaction
                    .execute(update_step_denied_statement(
                        &approval,
                        write.expired_at_millis,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            require_one(
                transaction
                    .execute(update_run_denied_statement(
                        &approval,
                        waiting.run_version,
                        write.expired_at_millis,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            require_one(
                transaction
                    .execute(insert_outbox_statement(
                        approval.community_id,
                        approval.approval_id,
                        approval.run_id,
                        approval.step_index,
                        write.expiry_operation_id,
                        ApprovalOutboxKind::Cancel,
                        write.expired_at_millis,
                    )?)
                    .await
                    .map_err(WorkflowApprovalError::Unavailable)?,
            )?;
            Ok((
                WorkflowApprovalDisposition::Applied,
                outbox_item(
                    &approval,
                    write.expiry_operation_id,
                    ApprovalOutboxKind::Cancel,
                    write.expired_at_millis,
                ),
            ))
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn pending_outbox(
        &self,
        tenant: &TenantContext,
        now_millis: u64,
        limit: u16,
    ) -> Result<Vec<WorkflowApprovalOutboxItem>, WorkflowApprovalError> {
        if now_millis == 0 || limit == 0 || limit > MAX_OUTBOX_BATCH {
            return Err(WorkflowApprovalError::InvalidInput);
        }
        let transaction = self.begin().await?;
        if let Err(error) = set_tenant(&transaction, tenant.community_id()).await {
            return finish_transaction(transaction, Err(error)).await;
        }
        let result = async {
            let rows = transaction
                .query_all(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_PENDING_OUTBOX_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        millis_value(now_millis)?,
                        i32::from(limit).into(),
                    ],
                ))
                .await
                .map_err(WorkflowApprovalError::Unavailable)?;
            rows.into_iter().map(|row| outbox_from_row(&row)).collect()
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, WorkflowApprovalError> {
        self.connection
            .begin()
            .await
            .map_err(WorkflowApprovalError::Unavailable)
    }
}

#[derive(Clone, Debug)]
struct CreationState {
    run_version: u64,
    run_state: String,
    step_state: String,
    step_operation_id: Uuid,
    current_definition_version: u64,
    lifecycle_state: String,
    creator_principal_id: PrincipalId,
    lease_run_version: u64,
    lease_generation: u64,
    lease_id: Uuid,
    worker_id: String,
    lease_state: String,
    lease_expires_at_millis: u64,
}

#[derive(Clone, Debug)]
struct MembershipRow {
    role: String,
    status: String,
    version: u64,
}

#[derive(Clone, Debug)]
struct WaitingState {
    run_version: u64,
    run_state: String,
    current_step_index: u16,
    step_state: String,
    step_operation_id: Uuid,
}

fn validate_request_write(
    write: &ApprovalRequestWrite<'_>,
) -> Result<ApprovalEligibility, WorkflowApprovalError> {
    let community_id = write.tenant.community_id();
    if write.definition.identity.community_id() != community_id
        || write.run.identity.community_id() != community_id
        || write.run.workflow != write.definition.identity
        || write.run.definition_version != write.definition.definition_version
        || write.run.state != WorkflowRunState::Running
        || write.run.current_step_index != write.step.index
        || write.step.state != WorkflowStepState::Running
        || write.step.operation_id.is_nil()
        || write.step.attempt_count == 0
        || write.lease.identity != write.run.identity
        || write.lease.admitted_run_version != write.run.run_version
        || write.lease.generation == 0
        || write.lease.lease_id.is_nil()
        || write.lease.worker_id.is_empty()
        || write.lease.state != WorkflowRunLeaseState::Active
        || write.lease.expires_at_millis < write.created_at_millis
        || write.definition.current_definition_version != write.definition.definition_version
        || write.definition.lifecycle != WorkflowLifecycle::Active
        || write.message.is_empty()
        || write.message.len() > MAX_APPROVAL_MESSAGE_BYTES
        || write
            .message
            .chars()
            .any(|character| character.is_control() && character != '\n' && character != '\t')
        || write.created_at_millis == 0
        || write.expires_at_millis <= write.created_at_millis
        || write.expires_at_millis - write.created_at_millis > MAX_APPROVAL_LIFETIME_MILLIS
    {
        return Err(WorkflowApprovalError::InvalidInput);
    }
    ApprovalEligibility::parse(write.eligibility_spec)
}

fn validate_decision_write(
    write: &ApprovalDecisionWrite<'_>,
) -> Result<PrincipalId, WorkflowApprovalError> {
    if write.approval_id.is_nil()
        || write.decision_operation_id.is_nil()
        || write.decision_operation_id == write.approval_id
        || write.decided_at_millis == 0
        || write.actor.community_id() != write.tenant.community_id()
        || write.note.is_some_and(|note| {
            note.len() > MAX_DECISION_NOTE_BYTES
                || note.chars().any(|character| {
                    character.is_control() && character != '\n' && character != '\t'
                })
        })
    {
        return Err(WorkflowApprovalError::InvalidInput);
    }
    let required_scope = AuthorizationScope::new(WORKFLOW_APPROVE_SCOPE)
        .map_err(|_| WorkflowApprovalError::InvalidInput)?;
    if !write.actor.scopes().contains(&required_scope) {
        return Err(WorkflowApprovalError::Unauthorized);
    }
    let principal_id = match write.actor.kind() {
        AuthenticatedPrincipalKind::ScopedToken {
            subject_principal_id,
            ..
        } => *subject_principal_id,
        _ => write.actor.principal_id(),
    };
    if principal_id.as_uuid().is_nil() {
        return Err(WorkflowApprovalError::Unauthorized);
    }
    Ok(principal_id)
}

fn approval_id(write: &ApprovalRequestWrite<'_>) -> Uuid {
    let identity = format!(
        "{}:{}:{}:{}:{}:{}",
        write.tenant.community_id(),
        write.definition.identity.workflow_id(),
        write.definition.definition_version,
        write.run.identity.run_id(),
        write.step.index,
        write.step.operation_id,
    );
    Uuid::new_v5(&APPROVAL_NAMESPACE, identity.as_bytes())
}

fn outbox_id(
    community_id: CommunityId,
    approval_id: Uuid,
    operation_id: Uuid,
    kind: ApprovalOutboxKind,
) -> Uuid {
    let identity = format!(
        "{}:{}:{}:{}",
        community_id,
        approval_id,
        operation_id,
        kind.database_name(),
    );
    Uuid::new_v5(&OUTBOX_NAMESPACE, identity.as_bytes())
}

fn exact_request_replay(
    approval: &StoredWorkflowApproval,
    write: &ApprovalRequestWrite<'_>,
    expected_approval_id: Uuid,
    capability_sha256: [u8; 32],
    eligibility: ApprovalEligibility,
) -> bool {
    approval.approval_id == expected_approval_id
        && approval.workflow_id == write.definition.identity.workflow_id()
        && approval.definition_version == write.definition.definition_version
        && approval.workflow_creator_principal_id == write.definition.creator_principal_id
        && approval.step_operation_id == write.step.operation_id
        && approval.capability_sha256 == capability_sha256
        && approval.eligibility == eligibility
        && approval.request_message == write.message
        && approval.expires_at_millis == write.expires_at_millis
        && approval.created_at_millis == write.created_at_millis
}

fn validate_creation_state(
    state: &CreationState,
    write: &ApprovalRequestWrite<'_>,
) -> Result<(), WorkflowApprovalError> {
    if state.run_version != write.run.run_version
        || state.run_state != "running"
        || state.step_state != "running"
        || state.step_operation_id != write.step.operation_id
        || state.current_definition_version != write.definition.definition_version
        || state.lifecycle_state != "active"
        || state.creator_principal_id != write.definition.creator_principal_id
        || state.lease_run_version != write.run.run_version
        || state.lease_generation != write.lease.generation
        || state.lease_id != write.lease.lease_id
        || state.worker_id != write.lease.worker_id
        || state.lease_state != "active"
        || state.lease_expires_at_millis < write.created_at_millis
    {
        return Err(WorkflowApprovalError::StaleRequest);
    }
    Ok(())
}

fn authorize_approver(
    approval: &StoredWorkflowApproval,
    actor_principal_id: PrincipalId,
    membership: &MembershipRow,
) -> Result<(), WorkflowApprovalError> {
    if membership.status != "active"
        || membership.version == 0
        || membership.role == "bot"
        || actor_principal_id == approval.workflow_creator_principal_id
    {
        return Err(WorkflowApprovalError::Unauthorized);
    }
    let eligible = match approval.eligibility {
        ApprovalEligibility::AnyMember => {
            matches!(
                membership.role.as_str(),
                "owner" | "admin" | "member" | "guest"
            )
        }
        ApprovalEligibility::Owner => membership.role == "owner",
        ApprovalEligibility::Administrator => {
            matches!(membership.role.as_str(), "owner" | "admin")
        }
        ApprovalEligibility::Principal(principal_id) => principal_id == actor_principal_id,
    };
    eligible
        .then_some(())
        .ok_or(WorkflowApprovalError::Unauthorized)
}

fn validate_waiting_state(
    state: &WaitingState,
    approval: &StoredWorkflowApproval,
) -> Result<(), WorkflowApprovalError> {
    if state.run_version == 0
        || state.run_state != "waiting_approval"
        || state.current_step_index != approval.step_index
        || state.step_state != "waiting_approval"
        || state.step_operation_id != approval.step_operation_id
    {
        return Err(WorkflowApprovalError::StaleRequest);
    }
    Ok(())
}

fn decision_outbox_item(
    approval: &StoredWorkflowApproval,
    write: &ApprovalDecisionWrite<'_>,
    kind: ApprovalOutboxKind,
) -> WorkflowApprovalOutboxItem {
    let created_at_millis = approval
        .decided_at_millis
        .unwrap_or(write.decided_at_millis);
    outbox_item(
        approval,
        write.decision_operation_id,
        kind,
        created_at_millis,
    )
}

fn outbox_item(
    approval: &StoredWorkflowApproval,
    operation_id: Uuid,
    kind: ApprovalOutboxKind,
    created_at_millis: u64,
) -> WorkflowApprovalOutboxItem {
    WorkflowApprovalOutboxItem {
        outbox_id: outbox_id(
            approval.community_id,
            approval.approval_id,
            operation_id,
            kind,
        ),
        approval_id: approval.approval_id,
        run_id: approval.run_id,
        step_index: approval.step_index,
        operation_id,
        kind,
        state: ApprovalOutboxState::Pending,
        attempt_count: 0,
        available_at_millis: created_at_millis,
        created_at_millis,
    }
}

async fn query_approval_by_step(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    run_id: Uuid,
    step_index: u16,
) -> Result<Option<StoredWorkflowApproval>, WorkflowApprovalError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_APPROVAL_BY_STEP_SQL,
            vec![
                community_id.as_uuid().into(),
                run_id.into(),
                i16::try_from(step_index)
                    .map_err(|_| WorkflowApprovalError::InvalidInput)?
                    .into(),
            ],
        ))
        .await
        .map_err(WorkflowApprovalError::Unavailable)?
        .map(|row| approval_from_row(&row, community_id))
        .transpose()
}

async fn query_approval_by_id(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    approval_id: Uuid,
) -> Result<Option<StoredWorkflowApproval>, WorkflowApprovalError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_APPROVAL_BY_ID_SQL,
            vec![community_id.as_uuid().into(), approval_id.into()],
        ))
        .await
        .map_err(WorkflowApprovalError::Unavailable)?
        .map(|row| approval_from_row(&row, community_id))
        .transpose()
}

async fn query_creation_state(
    transaction: &DatabaseTransaction,
    write: &ApprovalRequestWrite<'_>,
) -> Result<CreationState, WorkflowApprovalError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_CREATION_STATE_SQL,
            vec![
                write.tenant.community_id().as_uuid().into(),
                write.run.identity.run_id().into(),
                i16::try_from(write.step.index)
                    .map_err(|_| WorkflowApprovalError::InvalidInput)?
                    .into(),
                write.lease.generation.to_string().into(),
            ],
        ))
        .await
        .map_err(WorkflowApprovalError::Unavailable)?
        .ok_or(WorkflowApprovalError::StaleRequest)?;
    Ok(CreationState {
        run_version: parse_u64(row_value(&row, "run_version_text")?)?,
        run_state: row_value(&row, "run_state")?,
        step_state: row_value(&row, "step_state")?,
        step_operation_id: row_value(&row, "step_operation_id")?,
        current_definition_version: parse_u64(row_value(&row, "current_definition_version_text")?)?,
        lifecycle_state: row_value(&row, "lifecycle_state")?,
        creator_principal_id: PrincipalId::from_uuid(row_value(&row, "creator_principal_id")?),
        lease_run_version: parse_u64(row_value(&row, "lease_run_version_text")?)?,
        lease_generation: parse_u64(row_value(&row, "lease_generation_text")?)?,
        lease_id: row_value(&row, "lease_id")?,
        worker_id: row_value(&row, "worker_id")?,
        lease_state: row_value(&row, "lease_state")?,
        lease_expires_at_millis: parse_millis(row_value(&row, "lease_expires_at_millis")?)?,
    })
}

async fn query_membership(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    principal_id: PrincipalId,
) -> Result<Option<MembershipRow>, WorkflowApprovalError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_MEMBERSHIP_SQL,
            vec![community_id.as_uuid().into(), principal_id.as_uuid().into()],
        ))
        .await
        .map_err(WorkflowApprovalError::Unavailable)?
        .map(|row| {
            Ok(MembershipRow {
                role: row_value(&row, "role")?,
                status: row_value(&row, "status")?,
                version: parse_u64(row_value(&row, "membership_version_text")?)?,
            })
        })
        .transpose()
}

async fn query_waiting_state(
    transaction: &DatabaseTransaction,
    approval: &StoredWorkflowApproval,
) -> Result<Option<WaitingState>, WorkflowApprovalError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_WAITING_STATE_SQL,
            vec![
                approval.community_id.as_uuid().into(),
                approval.run_id.into(),
                i16::try_from(approval.step_index)
                    .map_err(|_| WorkflowApprovalError::InvalidInput)?
                    .into(),
            ],
        ))
        .await
        .map_err(WorkflowApprovalError::Unavailable)?
        .map(|row| {
            Ok(WaitingState {
                run_version: parse_u64(row_value(&row, "run_version_text")?)?,
                run_state: row_value(&row, "run_state")?,
                current_step_index: parse_u16(row_value(&row, "current_step_index")?)?,
                step_state: row_value(&row, "step_state")?,
                step_operation_id: row_value(&row, "step_operation_id")?,
            })
        })
        .transpose()
}

fn insert_approval_statement(
    write: &ApprovalRequestWrite<'_>,
    approval_id: Uuid,
    capability_sha256: [u8; 32],
    eligibility: ApprovalEligibility,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_APPROVAL_SQL,
        vec![
            write.tenant.community_id().as_uuid().into(),
            approval_id.into(),
            write.run.identity.run_id().into(),
            write.definition.identity.workflow_id().into(),
            write.definition.definition_version.to_string().into(),
            write.definition.creator_principal_id.as_uuid().into(),
            i16::try_from(write.step.index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            write.step.operation_id.into(),
            capability_sha256.to_vec().into(),
            eligibility.database_kind().into(),
            eligibility.principal_id().map(PrincipalId::as_uuid).into(),
            write.message.into(),
            millis_value(write.expires_at_millis)?,
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn update_step_waiting_statement(
    write: &ApprovalRequestWrite<'_>,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_STEP_WAITING_SQL,
        vec![
            write.tenant.community_id().as_uuid().into(),
            write.run.identity.run_id().into(),
            i16::try_from(write.step.index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            write.step.operation_id.into(),
            i16::try_from(write.step.attempt_count)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn update_run_waiting_statement(
    write: &ApprovalRequestWrite<'_>,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_RUN_WAITING_SQL,
        vec![
            write.tenant.community_id().as_uuid().into(),
            write.run.identity.run_id().into(),
            write.run.run_version.to_string().into(),
            i16::try_from(write.step.index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn release_lease_statement(
    write: &ApprovalRequestWrite<'_>,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        RELEASE_LEASE_WAITING_SQL,
        vec![
            write.tenant.community_id().as_uuid().into(),
            write.run.identity.run_id().into(),
            write.run.run_version.to_string().into(),
            write.lease.generation.to_string().into(),
            write.lease.lease_id.into(),
            write.lease.worker_id.as_str().into(),
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn insert_outbox_statement(
    community_id: CommunityId,
    approval_id: Uuid,
    run_id: Uuid,
    step_index: u16,
    operation_id: Uuid,
    kind: ApprovalOutboxKind,
    available_at_millis: u64,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_OUTBOX_SQL,
        vec![
            community_id.as_uuid().into(),
            outbox_id(community_id, approval_id, operation_id, kind).into(),
            approval_id.into(),
            run_id.into(),
            i16::try_from(step_index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            operation_id.into(),
            kind.database_name().into(),
            millis_value(available_at_millis)?,
        ],
    ))
}

fn update_approval_decision_statement(
    approval: &StoredWorkflowApproval,
    write: &ApprovalDecisionWrite<'_>,
    actor_principal_id: PrincipalId,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_APPROVAL_DECISION_SQL,
        vec![
            approval.community_id.as_uuid().into(),
            approval.approval_id.into(),
            write.decision.approval_state().database_name().into(),
            write.decision_operation_id.into(),
            actor_principal_id.as_uuid().into(),
            write.note.into(),
            millis_value(write.decided_at_millis)?,
        ],
    ))
}

fn update_approval_expiry_statement(
    approval: &StoredWorkflowApproval,
    write: &ApprovalExpiryWrite<'_>,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_APPROVAL_DECISION_SQL,
        vec![
            approval.community_id.as_uuid().into(),
            approval.approval_id.into(),
            ApprovalState::Expired.database_name().into(),
            write.expiry_operation_id.into(),
            Option::<Uuid>::None.into(),
            Option::<String>::None.into(),
            millis_value(write.expired_at_millis)?,
        ],
    ))
}

fn update_step_granted_statement(
    approval: &StoredWorkflowApproval,
    output: &str,
    decided_at_millis: u64,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_STEP_GRANTED_SQL,
        vec![
            approval.community_id.as_uuid().into(),
            approval.run_id.into(),
            i16::try_from(approval.step_index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            approval.step_operation_id.into(),
            output.into(),
            millis_value(decided_at_millis)?,
        ],
    ))
}

fn update_step_denied_statement(
    approval: &StoredWorkflowApproval,
    decided_at_millis: u64,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_STEP_DENIED_SQL,
        vec![
            approval.community_id.as_uuid().into(),
            approval.run_id.into(),
            i16::try_from(approval.step_index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            approval.step_operation_id.into(),
            millis_value(decided_at_millis)?,
        ],
    ))
}

fn update_run_granted_statement(
    approval: &StoredWorkflowApproval,
    run_version: u64,
    next_step_index: u16,
    decided_at_millis: u64,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_RUN_GRANTED_SQL,
        vec![
            approval.community_id.as_uuid().into(),
            approval.run_id.into(),
            run_version.to_string().into(),
            i16::try_from(next_step_index)
                .map_err(|_| WorkflowApprovalError::InvalidInput)?
                .into(),
            millis_value(decided_at_millis)?,
        ],
    ))
}

fn update_run_denied_statement(
    approval: &StoredWorkflowApproval,
    run_version: u64,
    decided_at_millis: u64,
) -> Result<Statement, WorkflowApprovalError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_RUN_DENIED_SQL,
        vec![
            approval.community_id.as_uuid().into(),
            approval.run_id.into(),
            run_version.to_string().into(),
            millis_value(decided_at_millis)?,
        ],
    ))
}

fn approval_from_row(
    row: &QueryResult,
    community_id: CommunityId,
) -> Result<StoredWorkflowApproval, WorkflowApprovalError> {
    let eligibility_kind = row_value::<String>(row, "eligibility_kind")?;
    let eligible_principal_id =
        row_value::<Option<Uuid>>(row, "eligible_principal_id")?.map(PrincipalId::from_uuid);
    let eligibility = match (eligibility_kind.as_str(), eligible_principal_id) {
        ("any_member", None) => ApprovalEligibility::AnyMember,
        ("owner", None) => ApprovalEligibility::Owner,
        ("administrator", None) => ApprovalEligibility::Administrator,
        ("principal", Some(principal_id)) if !principal_id.as_uuid().is_nil() => {
            ApprovalEligibility::Principal(principal_id)
        }
        _ => return Err(WorkflowApprovalError::InvalidRecord),
    };
    let state = ApprovalState::from_database(&row_value::<String>(row, "state")?)?;
    let decision_operation_id: Option<Uuid> = row_value(row, "decision_operation_id")?;
    let decided_by_principal_id =
        row_value::<Option<Uuid>>(row, "decided_by_principal_id")?.map(PrincipalId::from_uuid);
    let decision_note: Option<String> = row_value(row, "decision_note")?;
    let decided_at_millis = parse_optional_millis(row_value(row, "decided_at_millis")?)?;
    if (state == ApprovalState::Pending)
        != (decision_operation_id.is_none()
            && decided_by_principal_id.is_none()
            && decided_at_millis.is_none())
    {
        return Err(WorkflowApprovalError::InvalidRecord);
    }
    Ok(StoredWorkflowApproval {
        community_id,
        approval_id: row_value(row, "approval_id")?,
        run_id: row_value(row, "run_id")?,
        workflow_id: row_value(row, "workflow_id")?,
        definition_version: parse_u64(row_value(row, "definition_version_text")?)?,
        workflow_creator_principal_id: PrincipalId::from_uuid(row_value(
            row,
            "workflow_creator_principal_id",
        )?),
        step_index: parse_u16(row_value(row, "step_index")?)?,
        step_operation_id: row_value(row, "step_operation_id")?,
        capability_sha256: bytes32(row_value(row, "capability_sha256")?)?,
        eligibility,
        request_message: row_value(row, "request_message")?,
        state,
        decision_operation_id,
        decided_by_principal_id,
        decision_note,
        expires_at_millis: parse_millis(row_value(row, "expires_at_millis")?)?,
        created_at_millis: parse_millis(row_value(row, "created_at_millis")?)?,
        decided_at_millis,
        updated_at_millis: parse_millis(row_value(row, "updated_at_millis")?)?,
    })
}

fn outbox_from_row(row: &QueryResult) -> Result<WorkflowApprovalOutboxItem, WorkflowApprovalError> {
    Ok(WorkflowApprovalOutboxItem {
        outbox_id: row_value(row, "outbox_id")?,
        approval_id: row_value(row, "approval_id")?,
        run_id: row_value(row, "run_id")?,
        step_index: parse_u16(row_value(row, "step_index")?)?,
        operation_id: row_value(row, "operation_id")?,
        kind: ApprovalOutboxKind::from_database(&row_value::<String>(row, "intent_kind")?)?,
        state: ApprovalOutboxState::from_database(&row_value::<String>(row, "state")?)?,
        attempt_count: parse_nonnegative_u16(row_value(row, "attempt_count")?)?,
        available_at_millis: parse_millis(row_value(row, "available_at_millis")?)?,
        created_at_millis: parse_millis(row_value(row, "created_at_millis")?)?,
    })
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, WorkflowApprovalError>,
) -> Result<T, WorkflowApprovalError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(WorkflowApprovalError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(WorkflowApprovalError::Unavailable)?;
            Err(error)
        }
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), WorkflowApprovalError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            vec![community_id.to_string().into()],
        ))
        .await
        .map_err(WorkflowApprovalError::Unavailable)?;
    Ok(())
}

fn require_one(result: ExecResult) -> Result<(), WorkflowApprovalError> {
    (result.rows_affected() == 1)
        .then_some(())
        .ok_or(WorkflowApprovalError::StaleRequest)
}

fn parse_u64(value: String) -> Result<u64, WorkflowApprovalError> {
    let value = value
        .parse::<u64>()
        .map_err(|_| WorkflowApprovalError::InvalidRecord)?;
    (value > 0)
        .then_some(value)
        .ok_or(WorkflowApprovalError::InvalidRecord)
}

fn parse_u16(value: i16) -> Result<u16, WorkflowApprovalError> {
    u16::try_from(value).map_err(|_| WorkflowApprovalError::InvalidRecord)
}

fn parse_nonnegative_u16(value: i16) -> Result<u16, WorkflowApprovalError> {
    u16::try_from(value).map_err(|_| WorkflowApprovalError::InvalidRecord)
}

fn parse_millis(value: i64) -> Result<u64, WorkflowApprovalError> {
    u64::try_from(value).map_err(|_| WorkflowApprovalError::InvalidRecord)
}

fn parse_optional_millis(value: Option<i64>) -> Result<Option<u64>, WorkflowApprovalError> {
    value.map(parse_millis).transpose()
}

fn millis_value(value: u64) -> Result<sea_orm::Value, WorkflowApprovalError> {
    i64::try_from(value)
        .map(Into::into)
        .map_err(|_| WorkflowApprovalError::InvalidInput)
}

fn bytes32(value: Vec<u8>) -> Result<[u8; 32], WorkflowApprovalError> {
    value
        .try_into()
        .map_err(|_| WorkflowApprovalError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, WorkflowApprovalError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| WorkflowApprovalError::InvalidRecord)
}
