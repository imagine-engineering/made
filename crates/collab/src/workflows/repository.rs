use std::collections::BTreeSet;

use collaboration_domain::{CommunityId, PrincipalId, TenantContext};
use collaboration_workflow::definition::WorkflowDefinition;
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::scheduler::{
    SELECT_QUEUE_OBSERVATION_SQL, WorkflowCapacityScope, WorkflowQueueObservation,
    capacity_scope_from_database_error,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const SELECT_DEFINITION_VERSION_SQL: &str = r#"
SELECT
    definition.creator_principal_id,
    definition.scope_kind,
    definition.project_signer_public_key,
    definition.project_slug,
    definition.project_record_version::text AS project_record_version_text,
    version.definition_version::text AS definition_version_text,
    version.definition_schema_version,
    version.name,
    version.definition::text AS definition_json,
    version.definition_sha256,
    version.author_principal_id,
    version.source_system,
    version.source_record_id,
    version.source_version,
    floor(extract(epoch FROM version.source_observed_at) * 1000)::bigint
        AS source_observed_at_millis,
    version.source_integrity_sha256,
    floor(extract(epoch FROM version.created_at) * 1000)::bigint AS created_at_millis,
    head.current_definition_version::text AS current_definition_version_text,
    head.head_revision::text AS head_revision_text,
    head.lifecycle_state
FROM public.collaboration_workflow_definition_versions AS version
JOIN public.collaboration_workflow_definitions AS definition
    USING (community_id, workflow_id)
JOIN public.collaboration_workflow_definition_heads AS head
    USING (community_id, workflow_id)
WHERE version.community_id = $1
  AND version.workflow_id = $2
  AND version.definition_version = CAST($3 AS numeric)
"#;
const SELECT_DEFINITION_CONFLICT_SQL: &str = r#"
SELECT
    version.definition_version::text AS definition_version_text,
    version.definition_sha256,
    version.author_principal_id,
    version.source_system,
    version.source_record_id,
    version.source_version,
    floor(extract(epoch FROM version.source_observed_at) * 1000)::bigint
        AS source_observed_at_millis,
    version.source_integrity_sha256,
    floor(extract(epoch FROM version.created_at) * 1000)::bigint AS created_at_millis,
    definition.creator_principal_id,
    definition.scope_kind,
    definition.project_signer_public_key,
    definition.project_slug,
    definition.project_record_version::text AS project_record_version_text
FROM public.collaboration_workflow_definition_versions AS version
JOIN public.collaboration_workflow_definitions AS definition
    USING (community_id, workflow_id)
WHERE version.community_id = $1
  AND version.workflow_id = $2
  AND (version.definition_version = CAST($3 AS numeric)
       OR version.definition_sha256 = $4)
FOR UPDATE
"#;
const SELECT_DEFINITION_IDENTITY_SQL: &str = r#"
SELECT creator_principal_id, scope_kind, project_signer_public_key, project_slug,
       project_record_version::text AS project_record_version_text
FROM public.collaboration_workflow_definitions
WHERE community_id = $1 AND workflow_id = $2
FOR UPDATE
"#;
const SELECT_DEFINITION_HEAD_SQL: &str = r#"
SELECT current_definition_version::text AS current_definition_version_text,
       head_revision::text AS head_revision_text,
       lifecycle_state
FROM public.collaboration_workflow_definition_heads
WHERE community_id = $1 AND workflow_id = $2
FOR UPDATE
"#;
const INSERT_DEFINITION_IDENTITY_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_definitions (
    community_id, workflow_id, creator_principal_id, scope_kind,
    project_signer_public_key, project_slug, project_record_version, created_at
) VALUES (
    $1, $2, $3, $4, $5, $6,
    CASE WHEN $7::text IS NULL THEN NULL ELSE CAST($7 AS numeric) END,
    to_timestamp($8::double precision / 1000)
)
"#;
const INSERT_DEFINITION_VERSION_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_definition_versions (
    community_id, workflow_id, definition_version, definition_schema_version,
    name, definition, definition_sha256, author_principal_id,
    source_system, source_record_id, source_version, source_observed_at,
    source_integrity_sha256, created_at
) VALUES (
    $1, $2, CAST($3 AS numeric), $4, $5, CAST($6 AS jsonb), $7, $8,
    $9, $10, $11, to_timestamp($12::double precision / 1000), $13,
    to_timestamp($14::double precision / 1000)
)
"#;
const INSERT_DEFINITION_HEAD_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_definition_heads (
    community_id, workflow_id, current_definition_version, head_revision,
    lifecycle_state, source_system, source_record_id, source_version,
    source_observed_at, updated_at
) VALUES (
    $1, $2, CAST($3 AS numeric), 1, $4, $5, $6, $7,
    to_timestamp($8::double precision / 1000),
    to_timestamp($9::double precision / 1000)
)
"#;
const UPDATE_DEFINITION_HEAD_SQL: &str = r#"
UPDATE public.collaboration_workflow_definition_heads
SET current_definition_version = CAST($3 AS numeric),
    head_revision = head_revision + 1,
    lifecycle_state = $4,
    source_system = $5,
    source_record_id = $6,
    source_version = $7,
    source_observed_at = to_timestamp($8::double precision / 1000),
    updated_at = to_timestamp($9::double precision / 1000)
WHERE community_id = $1
  AND workflow_id = $2
  AND head_revision = CAST($10 AS numeric)
"#;
const SELECT_TRIGGER_RUN_SQL: &str = r#"
SELECT run_id
FROM public.collaboration_workflow_runs
WHERE community_id = $1 AND trigger_operation_id = $2
FOR UPDATE
"#;
const SELECT_RUN_SQL: &str = r#"
SELECT
    workflow_id,
    definition_version::text AS definition_version_text,
    trigger_operation_id,
    trigger_kind,
    trigger_source_id,
    trigger_context::text AS trigger_context_json,
    run_version::text AS run_version_text,
    status,
    current_step_index,
    error_code,
    error_message,
    source_system,
    source_record_id,
    source_version,
    floor(extract(epoch FROM source_observed_at) * 1000)::bigint
        AS source_observed_at_millis,
    floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis,
    floor(extract(epoch FROM started_at) * 1000)::bigint AS started_at_millis,
    floor(extract(epoch FROM completed_at) * 1000)::bigint AS completed_at_millis,
    floor(extract(epoch FROM updated_at) * 1000)::bigint AS updated_at_millis
FROM public.collaboration_workflow_runs
WHERE community_id = $1 AND run_id = $2
"#;
const SELECT_STEPS_SQL: &str = r#"
SELECT
    workflow_id,
    definition_version::text AS definition_version_text,
    step_index,
    step_id,
    operation_id,
    state,
    attempt_count,
    output::text AS output_json,
    error_code,
    error_message,
    source_system,
    source_record_id,
    source_version,
    floor(extract(epoch FROM source_observed_at) * 1000)::bigint
        AS source_observed_at_millis,
    floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis,
    floor(extract(epoch FROM started_at) * 1000)::bigint AS started_at_millis,
    floor(extract(epoch FROM completed_at) * 1000)::bigint AS completed_at_millis,
    floor(extract(epoch FROM updated_at) * 1000)::bigint AS updated_at_millis
FROM public.collaboration_workflow_steps
WHERE community_id = $1 AND run_id = $2
ORDER BY step_index
"#;
const SELECT_RETRIES_SQL: &str = r#"
SELECT
    step_index,
    attempt_number,
    retry_operation_id,
    failure_class,
    state,
    floor(extract(epoch FROM scheduled_at) * 1000)::bigint AS scheduled_at_millis,
    floor(extract(epoch FROM due_at) * 1000)::bigint AS due_at_millis,
    floor(extract(epoch FROM claimed_at) * 1000)::bigint AS claimed_at_millis,
    floor(extract(epoch FROM completed_at) * 1000)::bigint AS completed_at_millis,
    source_system,
    source_record_id,
    source_version,
    floor(extract(epoch FROM source_observed_at) * 1000)::bigint
        AS source_observed_at_millis,
    floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis
FROM public.collaboration_workflow_retries
WHERE community_id = $1 AND run_id = $2
ORDER BY step_index, attempt_number
"#;
const INSERT_RUN_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_runs (
    community_id, run_id, workflow_id, definition_version,
    trigger_operation_id, trigger_kind, trigger_source_id, trigger_context,
    run_version, status, current_step_index,
    source_system, source_record_id, source_version, source_observed_at,
    created_at, updated_at
) VALUES (
    $1, $2, $3, CAST($4 AS numeric), $5, $6, $7, CAST($8 AS jsonb),
    1, 'queued', 0, $9, $10, $11,
    to_timestamp($12::double precision / 1000),
    to_timestamp($13::double precision / 1000),
    to_timestamp($13::double precision / 1000)
)
"#;
const INSERT_STEP_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_steps (
    community_id, run_id, workflow_id, definition_version,
    step_index, step_id, operation_id, state, attempt_count,
    source_system, source_record_id, source_version, source_observed_at,
    created_at, updated_at
) VALUES (
    $1, $2, $3, CAST($4 AS numeric), $5, $6, $7, 'pending', 0,
    $8, $9, $10, to_timestamp($11::double precision / 1000),
    to_timestamp($12::double precision / 1000),
    to_timestamp($12::double precision / 1000)
)
"#;
const UPDATE_STEP_SQL: &str = r#"
UPDATE public.collaboration_workflow_steps
SET state = $5,
    attempt_count = $6,
    output = CASE WHEN $7::text IS NULL THEN NULL ELSE CAST($7 AS jsonb) END,
    error_code = $8,
    error_message = $9,
    started_at = CASE WHEN $5 = 'running'
        THEN COALESCE(started_at, to_timestamp($10::double precision / 1000))
        ELSE started_at END,
    completed_at = CASE WHEN $5 IN ('completed', 'skipped', 'failed', 'cancelled')
        THEN to_timestamp($10::double precision / 1000) ELSE NULL END,
    updated_at = to_timestamp($10::double precision / 1000)
WHERE community_id = $1
  AND run_id = $2
  AND step_index = $3
  AND operation_id = $4
  AND state = $11
"#;
const UPDATE_RUN_SQL: &str = r#"
UPDATE public.collaboration_workflow_runs
SET run_version = run_version + 1,
    status = $4,
    current_step_index = $5,
    error_code = $6,
    error_message = $7,
    started_at = CASE WHEN $4 = 'running'
        THEN COALESCE(started_at, to_timestamp($8::double precision / 1000))
        ELSE started_at END,
    completed_at = CASE WHEN $4 IN ('completed', 'failed', 'cancelled')
        THEN to_timestamp($8::double precision / 1000) ELSE NULL END,
    updated_at = to_timestamp($8::double precision / 1000)
WHERE community_id = $1
  AND run_id = $2
  AND run_version = CAST($3 AS numeric)
"#;
const SELECT_RETRY_OPERATION_SQL: &str = r#"
SELECT run_id, step_index, attempt_number, failure_class, state,
       floor(extract(epoch FROM scheduled_at) * 1000)::bigint AS scheduled_at_millis,
       floor(extract(epoch FROM due_at) * 1000)::bigint AS due_at_millis,
       source_system, source_record_id, source_version,
       floor(extract(epoch FROM source_observed_at) * 1000)::bigint
           AS source_observed_at_millis,
       floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis
FROM public.collaboration_workflow_retries
WHERE community_id = $1 AND retry_operation_id = $2
FOR UPDATE
"#;
const INSERT_RETRY_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_retries (
    community_id, run_id, step_index, attempt_number, retry_operation_id,
    failure_class, state, scheduled_at, due_at,
    source_system, source_record_id, source_version, source_observed_at, created_at
) VALUES (
    $1, $2, $3, $4, $5, $6, 'scheduled',
    to_timestamp($7::double precision / 1000),
    to_timestamp($8::double precision / 1000),
    $9, $10, $11, to_timestamp($12::double precision / 1000),
    to_timestamp($13::double precision / 1000)
)
"#;
const SELECT_LEASE_BY_ID_SQL: &str = r#"
SELECT
    run_id,
    run_version::text AS run_version_text,
    lease_generation::text AS lease_generation_text,
    lease_id,
    worker_id,
    state,
    floor(extract(epoch FROM acquired_at) * 1000)::bigint AS acquired_at_millis,
    floor(extract(epoch FROM last_heartbeat_at) * 1000)::bigint
        AS last_heartbeat_at_millis,
    floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
    floor(extract(epoch FROM recovery_after) * 1000)::bigint AS recovery_after_millis,
    floor(extract(epoch FROM released_at) * 1000)::bigint AS released_at_millis,
    release_reason
FROM public.collaboration_workflow_run_leases
WHERE community_id = $1 AND lease_id = $2
FOR UPDATE
"#;
const SELECT_ACTIVE_LEASE_SQL: &str = r#"
SELECT
    run_id,
    run_version::text AS run_version_text,
    lease_generation::text AS lease_generation_text,
    lease_id,
    worker_id,
    state,
    floor(extract(epoch FROM acquired_at) * 1000)::bigint AS acquired_at_millis,
    floor(extract(epoch FROM last_heartbeat_at) * 1000)::bigint
        AS last_heartbeat_at_millis,
    floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
    floor(extract(epoch FROM recovery_after) * 1000)::bigint AS recovery_after_millis,
    floor(extract(epoch FROM released_at) * 1000)::bigint AS released_at_millis,
    release_reason
FROM public.collaboration_workflow_run_leases
WHERE community_id = $1 AND run_id = $2 AND state = 'active'
FOR UPDATE
"#;
const SELECT_MAXIMUM_LEASE_GENERATION_SQL: &str = r#"
SELECT COALESCE(max(lease_generation), 0)::text AS lease_generation_text
FROM public.collaboration_workflow_run_leases
WHERE community_id = $1 AND run_id = $2
"#;
const EXPIRE_LEASE_SQL: &str = r#"
UPDATE public.collaboration_workflow_run_leases
SET state = 'released',
    released_at = to_timestamp($5::double precision / 1000),
    release_reason = 'expired'
WHERE community_id = $1
  AND run_id = $2
  AND lease_generation = CAST($3 AS numeric)
  AND lease_id = $4
  AND state = 'active'
"#;
const INSERT_LEASE_SQL: &str = r#"
INSERT INTO public.collaboration_workflow_run_leases (
    community_id, run_id, run_version, lease_generation, lease_id, worker_id,
    state, acquired_at, last_heartbeat_at, expires_at, recovery_after
) VALUES (
    $1, $2, CAST($3 AS numeric), CAST($4 AS numeric), $5, $6, 'active',
    to_timestamp($7::double precision / 1000),
    to_timestamp($7::double precision / 1000),
    to_timestamp($8::double precision / 1000),
    to_timestamp($9::double precision / 1000)
)
"#;
const HEARTBEAT_LEASE_SQL: &str = r#"
UPDATE public.collaboration_workflow_run_leases
SET last_heartbeat_at = to_timestamp($7::double precision / 1000),
    expires_at = to_timestamp($8::double precision / 1000),
    recovery_after = to_timestamp($9::double precision / 1000)
WHERE community_id = $1
  AND run_id = $2
  AND lease_generation = CAST($3 AS numeric)
  AND lease_id = $4
  AND worker_id = $5
  AND state = 'active'
  AND expires_at >= to_timestamp($6::double precision / 1000)
"#;
const RELEASE_LEASE_SQL: &str = r#"
UPDATE public.collaboration_workflow_run_leases
SET state = 'released',
    released_at = to_timestamp($6::double precision / 1000),
    release_reason = $7
WHERE community_id = $1
  AND run_id = $2
  AND lease_generation = CAST($3 AS numeric)
  AND lease_id = $4
  AND worker_id = $5
  AND state = 'active'
"#;

const MAX_SOURCE_SYSTEM_BYTES: usize = 64;
const MAX_SOURCE_RECORD_BYTES: usize = 512;
const MAX_SOURCE_VERSION_BYTES: usize = 128;
const MAX_TRIGGER_SOURCE_BYTES: usize = 512;
const MAX_TRIGGER_CONTEXT_BYTES: usize = 1024 * 1024;
const MAX_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_ERROR_CODE_BYTES: usize = 64;
const MAX_ERROR_MESSAGE_BYTES: usize = 4 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkflowIdentity {
    community_id: CommunityId,
    workflow_id: Uuid,
}

impl WorkflowIdentity {
    pub fn new(
        community_id: CommunityId,
        workflow_id: Uuid,
    ) -> Result<Self, WorkflowRepositoryError> {
        if workflow_id.is_nil() {
            return Err(WorkflowRepositoryError::InvalidInput);
        }
        Ok(Self {
            community_id,
            workflow_id,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn workflow_id(self) -> Uuid {
        self.workflow_id
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkflowRunIdentity {
    community_id: CommunityId,
    run_id: Uuid,
}

impl WorkflowRunIdentity {
    pub fn new(community_id: CommunityId, run_id: Uuid) -> Result<Self, WorkflowRepositoryError> {
        if run_id.is_nil() {
            return Err(WorkflowRepositoryError::InvalidInput);
        }
        Ok(Self {
            community_id,
            run_id,
        })
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn run_id(self) -> Uuid {
        self.run_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkflowScope {
    Community,
    Project {
        signer_public_key: [u8; 32],
        slug: String,
        record_version: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowLifecycle {
    Draft,
    Active,
    Disabled,
    Archived,
}

impl WorkflowLifecycle {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Active => "active",
            Self::Disabled => "disabled",
            Self::Archived => "archived",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "draft" => Ok(Self::Draft),
            "active" => Ok(Self::Active),
            "disabled" => Ok(Self::Disabled),
            "archived" => Ok(Self::Archived),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowProvenance {
    source_system: String,
    source_record_id: String,
    source_version: String,
    observed_at_millis: u64,
    integrity_sha256: Option<[u8; 32]>,
}

impl WorkflowProvenance {
    pub fn new(
        source_system: impl Into<String>,
        source_record_id: impl Into<String>,
        source_version: impl Into<String>,
        observed_at_millis: u64,
        integrity_sha256: Option<[u8; 32]>,
    ) -> Result<Self, WorkflowRepositoryError> {
        let provenance = Self {
            source_system: source_system.into(),
            source_record_id: source_record_id.into(),
            source_version: source_version.into(),
            observed_at_millis,
            integrity_sha256,
        };
        validate_provenance(&provenance)?;
        Ok(provenance)
    }

    pub fn source_system(&self) -> &str {
        &self.source_system
    }

    pub fn source_record_id(&self) -> &str {
        &self.source_record_id
    }

    pub fn source_version(&self) -> &str {
        &self.source_version
    }

    pub const fn observed_at_millis(&self) -> u64 {
        self.observed_at_millis
    }

    pub const fn integrity_sha256(&self) -> Option<[u8; 32]> {
        self.integrity_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DefinitionVersionWrite {
    pub identity: WorkflowIdentity,
    pub definition_version: u64,
    pub definition: WorkflowDefinition,
    pub creator_principal_id: PrincipalId,
    pub author_principal_id: PrincipalId,
    pub scope: WorkflowScope,
    pub lifecycle: WorkflowLifecycle,
    pub expected_head_revision: Option<u64>,
    pub provenance: WorkflowProvenance,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredWorkflowDefinition {
    pub identity: WorkflowIdentity,
    pub definition_version: u64,
    pub definition: WorkflowDefinition,
    pub definition_sha256: [u8; 32],
    pub creator_principal_id: PrincipalId,
    pub author_principal_id: PrincipalId,
    pub scope: WorkflowScope,
    pub current_definition_version: u64,
    pub head_revision: u64,
    pub lifecycle: WorkflowLifecycle,
    pub provenance: WorkflowProvenance,
    pub created_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowStoreOutcome {
    Applied,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowTriggerKind {
    Event,
    Schedule,
    Webhook,
    Manual,
}

impl WorkflowTriggerKind {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Event => "event",
            Self::Schedule => "schedule",
            Self::Webhook => "webhook",
            Self::Manual => "manual",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "event" => Ok(Self::Event),
            "schedule" => Ok(Self::Schedule),
            "webhook" => Ok(Self::Webhook),
            "manual" => Ok(Self::Manual),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowRunState {
    Claimed,
    Queued,
    Running,
    WaitingApproval,
    RetryScheduled,
    RepairRequired,
    Completed,
    Failed,
    Cancelled,
}

impl WorkflowRunState {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Claimed => "claimed",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::WaitingApproval => "waiting_approval",
            Self::RetryScheduled => "retry_scheduled",
            Self::RepairRequired => "repair_required",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "claimed" => Ok(Self::Claimed),
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "waiting_approval" => Ok(Self::WaitingApproval),
            "retry_scheduled" => Ok(Self::RetryScheduled),
            "repair_required" => Ok(Self::RepairRequired),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }

    const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowStepState {
    Pending,
    Running,
    WaitingApproval,
    RetryScheduled,
    RepairRequired,
    Completed,
    Skipped,
    Failed,
    Cancelled,
}

impl WorkflowStepState {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::WaitingApproval => "waiting_approval",
            Self::RetryScheduled => "retry_scheduled",
            Self::RepairRequired => "repair_required",
            Self::Completed => "completed",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "waiting_approval" => Ok(Self::WaitingApproval),
            "retry_scheduled" => Ok(Self::RetryScheduled),
            "repair_required" => Ok(Self::RepairRequired),
            "completed" => Ok(Self::Completed),
            "skipped" => Ok(Self::Skipped),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }

    const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Skipped | Self::Failed | Self::Cancelled
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorkflowRunRequest {
    pub identity: WorkflowRunIdentity,
    pub workflow: WorkflowIdentity,
    pub definition_version: u64,
    pub trigger_operation_id: Uuid,
    pub trigger_kind: WorkflowTriggerKind,
    pub trigger_source_id: String,
    pub trigger_context: JsonValue,
    pub step_operation_ids: Vec<Uuid>,
    pub provenance: WorkflowProvenance,
    pub created_at_millis: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredWorkflowRun {
    pub identity: WorkflowRunIdentity,
    pub workflow: WorkflowIdentity,
    pub definition_version: u64,
    pub trigger_operation_id: Uuid,
    pub trigger_kind: WorkflowTriggerKind,
    pub trigger_source_id: String,
    pub trigger_context: JsonValue,
    pub run_version: u64,
    pub state: WorkflowRunState,
    pub current_step_index: u16,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub provenance: WorkflowProvenance,
    pub created_at_millis: u64,
    pub started_at_millis: Option<u64>,
    pub completed_at_millis: Option<u64>,
    pub updated_at_millis: u64,
    pub steps: Vec<StoredWorkflowStep>,
    pub retries: Vec<StoredWorkflowRetry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoredWorkflowStep {
    pub index: u16,
    pub step_id: String,
    pub operation_id: Uuid,
    pub state: WorkflowStepState,
    pub attempt_count: u16,
    pub output: Option<JsonValue>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub created_at_millis: u64,
    pub started_at_millis: Option<u64>,
    pub completed_at_millis: Option<u64>,
    pub updated_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryFailureClass {
    RateLimited,
    TemporaryUnavailable,
    Timeout,
    Transport,
}

impl RetryFailureClass {
    const fn database_name(self) -> &'static str {
        match self {
            Self::RateLimited => "rate_limited",
            Self::TemporaryUnavailable => "temporary_unavailable",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "rate_limited" => Ok(Self::RateLimited),
            "temporary_unavailable" => Ok(Self::TemporaryUnavailable),
            "timeout" => Ok(Self::Timeout),
            "transport" => Ok(Self::Transport),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryState {
    Scheduled,
    Claimed,
    Completed,
    Exhausted,
    Cancelled,
}

impl RetryState {
    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "scheduled" => Ok(Self::Scheduled),
            "claimed" => Ok(Self::Claimed),
            "completed" => Ok(Self::Completed),
            "exhausted" => Ok(Self::Exhausted),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredWorkflowRetry {
    pub step_index: u16,
    pub attempt_number: u16,
    pub retry_operation_id: Uuid,
    pub failure_class: RetryFailureClass,
    pub state: RetryState,
    pub scheduled_at_millis: u64,
    pub due_at_millis: u64,
    pub claimed_at_millis: Option<u64>,
    pub completed_at_millis: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowStepCheckpoint {
    pub identity: WorkflowRunIdentity,
    pub expected_run_version: u64,
    pub step_index: u16,
    pub operation_id: Uuid,
    pub expected_step_state: WorkflowStepState,
    pub next_step_state: WorkflowStepState,
    pub next_run_state: WorkflowRunState,
    pub next_step_index: u16,
    pub attempt_count: u16,
    pub output: Option<JsonValue>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub occurred_at_millis: u64,
    pub lease: WorkflowRunLeaseFence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRetryWrite {
    pub identity: WorkflowRunIdentity,
    pub step_index: u16,
    pub attempt_number: u16,
    pub retry_operation_id: Uuid,
    pub failure_class: RetryFailureClass,
    pub scheduled_at_millis: u64,
    pub due_at_millis: u64,
    pub provenance: WorkflowProvenance,
    pub created_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowRunLeaseState {
    Active,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowRunLeaseReleaseReason {
    Completed,
    Cancelled,
    Failed,
    Expired,
    Replaced,
}

impl WorkflowRunLeaseReleaseReason {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Replaced => "replaced",
        }
    }

    fn from_database(value: &str) -> Result<Self, WorkflowRepositoryError> {
        match value {
            "completed" => Ok(Self::Completed),
            "cancelled" => Ok(Self::Cancelled),
            "failed" => Ok(Self::Failed),
            "expired" => Ok(Self::Expired),
            "replaced" => Ok(Self::Replaced),
            _ => Err(WorkflowRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRunLease {
    pub identity: WorkflowRunIdentity,
    pub admitted_run_version: u64,
    pub generation: u64,
    pub lease_id: Uuid,
    pub worker_id: String,
    pub state: WorkflowRunLeaseState,
    pub acquired_at_millis: u64,
    pub last_heartbeat_at_millis: u64,
    pub expires_at_millis: u64,
    pub recovery_after_millis: u64,
    pub released_at_millis: Option<u64>,
    pub release_reason: Option<WorkflowRunLeaseReleaseReason>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRunLeaseRequest {
    pub identity: WorkflowRunIdentity,
    pub expected_run_version: u64,
    pub lease_id: Uuid,
    pub worker_id: String,
    pub acquired_at_millis: u64,
    pub expires_at_millis: u64,
    pub recovery_after_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRunLeaseAcquisition {
    pub outcome: WorkflowStoreOutcome,
    pub lease: WorkflowRunLease,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowRunLeaseFence {
    identity: WorkflowRunIdentity,
    generation: u64,
    lease_id: Uuid,
    worker_id: String,
}

impl From<&WorkflowRunLease> for WorkflowRunLeaseFence {
    fn from(lease: &WorkflowRunLease) -> Self {
        Self {
            identity: lease.identity,
            generation: lease.generation,
            lease_id: lease.lease_id,
            worker_id: lease.worker_id.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WorkflowRepositoryError {
    #[error("workflow repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("workflow repository request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("workflow repository input is invalid")]
    InvalidInput,
    #[error("workflow record does not exist")]
    NotFound,
    #[error("workflow operation conflicts with an existing idempotency record")]
    IdempotencyConflict,
    #[error("workflow state transition lost its optimistic fence")]
    TransitionConflict,
    #[error("workflow run lease is invalid")]
    InvalidLease,
    #[error("workflow run lease is not recoverable yet")]
    LeaseUnavailable,
    #[error("workflow run lease generation is no longer authoritative")]
    LeaseFenceLost,
    #[error("workflow scheduler capacity is unavailable for {0:?}")]
    CapacityUnavailable(WorkflowCapacityScope),
    #[error("workflow repository record is invalid")]
    InvalidRecord,
    #[error("workflow definition could not be encoded")]
    DefinitionEncoding(#[source] serde_json::Error),
    #[error("workflow repository is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct WorkflowRepository {
    connection: DatabaseConnection,
}

impl WorkflowRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, WorkflowRepositoryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(WorkflowRepositoryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn store_definition(
        &self,
        tenant: &TenantContext,
        write: &DefinitionVersionWrite,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        validate_workflow_identity(tenant, write.identity)?;
        let encoded = validate_definition_write(write)?;
        let definition_sha256: [u8; 32] = Sha256::digest(encoded.as_bytes()).into();
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let conflicts = transaction
                .query_all(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_DEFINITION_CONFLICT_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        write.identity.workflow_id.into(),
                        write.definition_version.to_string().into(),
                        definition_sha256.to_vec().into(),
                    ],
                ))
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if !conflicts.is_empty() {
                if conflicts.len() == 1
                    && parse_u64(row_value(&conflicts[0], "definition_version_text")?)?
                        == write.definition_version
                    && bytes32(row_value(&conflicts[0], "definition_sha256")?)? == definition_sha256
                    && immutable_definition_fields_match(&conflicts[0], write)?
                {
                    return Ok(WorkflowStoreOutcome::Duplicate);
                }
                return Err(WorkflowRepositoryError::IdempotencyConflict);
            }

            let identity_row = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_DEFINITION_IDENTITY_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        write.identity.workflow_id.into(),
                    ],
                ))
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if let Some(identity_row) = identity_row {
                validate_definition_identity(&identity_row, write)?;
            } else {
                transaction
                    .execute(insert_definition_identity_statement(write)?)
                    .await
                    .map_err(WorkflowRepositoryError::Unavailable)?;
            }

            let head = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_DEFINITION_HEAD_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        write.identity.workflow_id.into(),
                    ],
                ))
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            let previous_revision = if let Some(head) = head {
                let current_version =
                    parse_u64(row_value(&head, "current_definition_version_text")?)?;
                let head_revision = parse_u64(row_value(&head, "head_revision_text")?)?;
                if write.definition_version != current_version.saturating_add(1)
                    || write.expected_head_revision != Some(head_revision)
                {
                    return Err(WorkflowRepositoryError::TransitionConflict);
                }
                Some(head_revision)
            } else {
                if write.definition_version != 1 || write.expected_head_revision.is_some() {
                    return Err(WorkflowRepositoryError::TransitionConflict);
                }
                None
            };

            transaction
                .execute(insert_definition_version_statement(
                    write,
                    &encoded,
                    definition_sha256,
                )?)
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if let Some(previous_revision) = previous_revision {
                let updated = transaction
                    .execute(update_definition_head_statement(write, previous_revision)?)
                    .await
                    .map_err(WorkflowRepositoryError::Unavailable)?;
                if updated.rows_affected() != 1 {
                    return Err(WorkflowRepositoryError::TransitionConflict);
                }
            } else {
                transaction
                    .execute(insert_definition_head_statement(write)?)
                    .await
                    .map_err(WorkflowRepositoryError::Unavailable)?;
            }
            Ok(WorkflowStoreOutcome::Applied)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn load_definition(
        &self,
        tenant: &TenantContext,
        identity: WorkflowIdentity,
        definition_version: u64,
    ) -> Result<Option<StoredWorkflowDefinition>, WorkflowRepositoryError> {
        validate_workflow_identity(tenant, identity)?;
        if definition_version == 0 {
            return Err(WorkflowRepositoryError::InvalidInput);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            query_definition(&transaction, identity, definition_version, "FOR SHARE").await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn claim_run(
        &self,
        tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        validate_run_request(tenant, request)?;
        let context_json = serde_json::to_string(&request.trigger_context)
            .map_err(WorkflowRepositoryError::DefinitionEncoding)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            if let Some(row) = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_TRIGGER_RUN_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        request.trigger_operation_id.into(),
                    ],
                ))
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?
            {
                let run_id: Uuid = row_value(&row, "run_id")?;
                if run_id != request.identity.run_id {
                    return Err(WorkflowRepositoryError::IdempotencyConflict);
                }
                let stored = load_stored_run(&transaction, request.identity, "FOR UPDATE").await?;
                return if run_matches_request(&stored, request) {
                    Ok(WorkflowStoreOutcome::Duplicate)
                } else {
                    Err(WorkflowRepositoryError::IdempotencyConflict)
                };
            }

            let definition = query_definition(
                &transaction,
                request.workflow,
                request.definition_version,
                "FOR SHARE",
            )
            .await?
            .ok_or(WorkflowRepositoryError::NotFound)?;
            if definition.lifecycle != WorkflowLifecycle::Active
                || !definition.definition.enabled()
                || definition.definition_version != definition.current_definition_version
            {
                return Err(WorkflowRepositoryError::TransitionConflict);
            }
            if definition.definition.steps().len() != request.step_operation_ids.len() {
                return Err(WorkflowRepositoryError::InvalidInput);
            }
            transaction
                .execute(insert_run_statement(request, &context_json)?)
                .await
                .map_err(map_scheduler_database_error)?;
            for (index, (step, operation_id)) in definition
                .definition
                .steps()
                .iter()
                .zip(&request.step_operation_ids)
                .enumerate()
            {
                transaction
                    .execute(insert_step_statement(
                        request,
                        u16::try_from(index).map_err(|_| WorkflowRepositoryError::InvalidInput)?,
                        step.id(),
                        *operation_id,
                    )?)
                    .await
                    .map_err(WorkflowRepositoryError::Unavailable)?;
            }
            Ok(WorkflowStoreOutcome::Applied)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn load_run(
        &self,
        tenant: &TenantContext,
        identity: WorkflowRunIdentity,
    ) -> Result<Option<StoredWorkflowRun>, WorkflowRepositoryError> {
        validate_run_identity(tenant, identity)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            load_optional_stored_run(&transaction, identity, "FOR SHARE").await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn checkpoint_step(
        &self,
        tenant: &TenantContext,
        checkpoint: &WorkflowStepCheckpoint,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        validate_run_identity(tenant, checkpoint.identity)?;
        validate_checkpoint(checkpoint)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            validate_active_lease(
                &transaction,
                &checkpoint.lease,
                checkpoint.occurred_at_millis,
            )
            .await?;
            let stored = load_stored_run(&transaction, checkpoint.identity, "FOR UPDATE").await?;
            let step = stored
                .steps
                .get(usize::from(checkpoint.step_index))
                .filter(|step| step.index == checkpoint.step_index)
                .ok_or(WorkflowRepositoryError::NotFound)?;
            if checkpoint.operation_id != step.operation_id {
                return Err(WorkflowRepositoryError::IdempotencyConflict);
            }
            if checkpoint_is_duplicate(&stored, step, checkpoint) {
                return Ok(WorkflowStoreOutcome::Duplicate);
            }
            if stored.run_version != checkpoint.expected_run_version
                || step.state != checkpoint.expected_step_state
                || !valid_step_transition(step.state, checkpoint.next_step_state)
                || !valid_attempt_transition(step, checkpoint)
                || !valid_run_step_pair(&stored, checkpoint)
                || checkpoint.occurred_at_millis < stored.updated_at_millis
                || checkpoint.occurred_at_millis < step.updated_at_millis
            {
                return Err(WorkflowRepositoryError::TransitionConflict);
            }
            let output_json = checkpoint
                .output
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(WorkflowRepositoryError::DefinitionEncoding)?;
            let step_update = transaction
                .execute(update_step_statement(checkpoint, output_json.as_deref())?)
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if step_update.rows_affected() != 1 {
                return Err(WorkflowRepositoryError::TransitionConflict);
            }
            let run_update = transaction
                .execute(update_run_statement(checkpoint)?)
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if run_update.rows_affected() != 1 {
                return Err(WorkflowRepositoryError::TransitionConflict);
            }
            Ok(WorkflowStoreOutcome::Applied)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn record_retry(
        &self,
        tenant: &TenantContext,
        retry: &WorkflowRetryWrite,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        validate_retry_write(tenant, retry)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            if let Some(row) = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_RETRY_OPERATION_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        retry.retry_operation_id.into(),
                    ],
                ))
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?
            {
                return if retry_row_matches(&row, retry)? {
                    Ok(WorkflowStoreOutcome::Duplicate)
                } else {
                    Err(WorkflowRepositoryError::IdempotencyConflict)
                };
            }
            transaction
                .execute(insert_retry_statement(retry)?)
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            Ok(WorkflowStoreOutcome::Applied)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn acquire_run_lease(
        &self,
        tenant: &TenantContext,
        request: &WorkflowRunLeaseRequest,
    ) -> Result<WorkflowRunLeaseAcquisition, WorkflowRepositoryError> {
        validate_lease_request(tenant, request)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let run = load_stored_run(&transaction, request.identity, "FOR UPDATE").await?;
            if run.run_version != request.expected_run_version
                || !matches!(
                    run.state,
                    WorkflowRunState::Claimed
                        | WorkflowRunState::Queued
                        | WorkflowRunState::Running
                        | WorkflowRunState::RetryScheduled
                        | WorkflowRunState::RepairRequired
                )
            {
                return Err(WorkflowRepositoryError::InvalidLease);
            }
            if let Some(existing) = query_lease_by_id(
                &transaction,
                request.identity.community_id,
                request.lease_id,
            )
            .await?
            {
                if lease_matches_request(&existing, request)
                    && existing.state == WorkflowRunLeaseState::Active
                {
                    return Ok(WorkflowRunLeaseAcquisition {
                        outcome: WorkflowStoreOutcome::Duplicate,
                        lease: existing,
                    });
                }
                return Err(WorkflowRepositoryError::LeaseFenceLost);
            }
            if let Some(active) = query_active_lease(&transaction, request.identity).await? {
                if active.recovery_after_millis > request.acquired_at_millis {
                    return Err(WorkflowRepositoryError::LeaseUnavailable);
                }
                let expired = transaction
                    .execute(expire_lease_statement(&active, request.acquired_at_millis)?)
                    .await
                    .map_err(WorkflowRepositoryError::Unavailable)?;
                if expired.rows_affected() != 1 {
                    return Err(WorkflowRepositoryError::LeaseFenceLost);
                }
            }
            let maximum_generation =
                query_maximum_lease_generation(&transaction, request.identity).await?;
            let generation = maximum_generation
                .checked_add(1)
                .filter(|generation| *generation > 0)
                .ok_or(WorkflowRepositoryError::InvalidLease)?;
            transaction
                .execute(insert_lease_statement(request, generation)?)
                .await
                .map_err(map_scheduler_database_error)?;
            Ok(WorkflowRunLeaseAcquisition {
                outcome: WorkflowStoreOutcome::Applied,
                lease: WorkflowRunLease {
                    identity: request.identity,
                    admitted_run_version: request.expected_run_version,
                    generation,
                    lease_id: request.lease_id,
                    worker_id: request.worker_id.clone(),
                    state: WorkflowRunLeaseState::Active,
                    acquired_at_millis: request.acquired_at_millis,
                    last_heartbeat_at_millis: request.acquired_at_millis,
                    expires_at_millis: request.expires_at_millis,
                    recovery_after_millis: request.recovery_after_millis,
                    released_at_millis: None,
                    release_reason: None,
                },
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn heartbeat_run_lease(
        &self,
        tenant: &TenantContext,
        fence: &WorkflowRunLeaseFence,
        heartbeat_at_millis: u64,
        expires_at_millis: u64,
        recovery_after_millis: u64,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        validate_lease_fence(tenant, fence)?;
        if heartbeat_at_millis > expires_at_millis || expires_at_millis > recovery_after_millis {
            return Err(WorkflowRepositoryError::InvalidLease);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let updated = transaction
                .execute(heartbeat_lease_statement(
                    fence,
                    heartbeat_at_millis,
                    expires_at_millis,
                    recovery_after_millis,
                )?)
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if updated.rows_affected() == 1 {
                return Ok(WorkflowStoreOutcome::Applied);
            }
            let existing =
                query_lease_by_id(&transaction, fence.identity.community_id, fence.lease_id)
                    .await?
                    .ok_or(WorkflowRepositoryError::LeaseFenceLost)?;
            if lease_matches_fence(&existing, fence)
                && existing.state == WorkflowRunLeaseState::Active
                && existing.last_heartbeat_at_millis == heartbeat_at_millis
                && existing.expires_at_millis == expires_at_millis
                && existing.recovery_after_millis == recovery_after_millis
            {
                Ok(WorkflowStoreOutcome::Duplicate)
            } else {
                Err(WorkflowRepositoryError::LeaseFenceLost)
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn release_run_lease(
        &self,
        tenant: &TenantContext,
        fence: &WorkflowRunLeaseFence,
        released_at_millis: u64,
        reason: WorkflowRunLeaseReleaseReason,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        validate_lease_fence(tenant, fence)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let updated = transaction
                .execute(release_lease_statement(fence, released_at_millis, reason)?)
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            if updated.rows_affected() == 1 {
                return Ok(WorkflowStoreOutcome::Applied);
            }
            let existing =
                query_lease_by_id(&transaction, fence.identity.community_id, fence.lease_id)
                    .await?
                    .ok_or(WorkflowRepositoryError::LeaseFenceLost)?;
            if lease_matches_fence(&existing, fence)
                && existing.state == WorkflowRunLeaseState::Released
                && existing.released_at_millis == Some(released_at_millis)
                && existing.release_reason == Some(reason)
            {
                Ok(WorkflowStoreOutcome::Duplicate)
            } else {
                Err(WorkflowRepositoryError::LeaseFenceLost)
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub(super) async fn observe_queue(
        &self,
        tenant: &TenantContext,
        now_millis: u64,
    ) -> Result<WorkflowQueueObservation, WorkflowRepositoryError> {
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let row = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_QUEUE_OBSERVATION_SQL,
                    [tenant.community_id().as_uuid().into()],
                ))
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?
                .ok_or(WorkflowRepositoryError::InvalidRecord)?;
            let community_queue_depth =
                u32::try_from(row_value::<i64>(&row, "community_queue_depth")?)
                    .map_err(|_| WorkflowRepositoryError::InvalidRecord)?;
            let deployment_queue_depth =
                u32::try_from(row_value::<i64>(&row, "deployment_queue_depth")?)
                    .map_err(|_| WorkflowRepositoryError::InvalidRecord)?;
            let community_oldest_at_millis =
                parse_optional_millis(row_value(&row, "community_oldest_at_millis")?)?;
            let deployment_oldest_at_millis =
                parse_optional_millis(row_value(&row, "deployment_oldest_at_millis")?)?;
            let oldest_queued_at_millis =
                match (community_oldest_at_millis, deployment_oldest_at_millis) {
                    (Some(community), Some(deployment)) => Some(community.min(deployment)),
                    (Some(community), None) => Some(community),
                    (None, Some(deployment)) => Some(deployment),
                    (None, None) => None,
                };
            Ok(WorkflowQueueObservation {
                community_queue_depth,
                deployment_queue_depth,
                oldest_queued_seconds: oldest_queued_at_millis
                    .map(|queued_at| now_millis.saturating_sub(queued_at) / 1_000),
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, WorkflowRepositoryError> {
        self.connection
            .begin()
            .await
            .map_err(WorkflowRepositoryError::Unavailable)
    }
}

fn map_scheduler_database_error(error: DbErr) -> WorkflowRepositoryError {
    match capacity_scope_from_database_error(&error) {
        Some(scope) => WorkflowRepositoryError::CapacityUnavailable(scope),
        None => WorkflowRepositoryError::Unavailable(error),
    }
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, WorkflowRepositoryError>,
) -> Result<T, WorkflowRepositoryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(WorkflowRepositoryError::Unavailable)?;
            Err(error)
        }
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), WorkflowRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?;
    Ok(())
}

async fn query_definition(
    transaction: &DatabaseTransaction,
    identity: WorkflowIdentity,
    definition_version: u64,
    lock_clause: &str,
) -> Result<Option<StoredWorkflowDefinition>, WorkflowRepositoryError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            format!("{SELECT_DEFINITION_VERSION_SQL}{lock_clause}"),
            [
                identity.community_id.as_uuid().into(),
                identity.workflow_id.into(),
                definition_version.to_string().into(),
            ],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?;
    row.map(|row| definition_from_row(row, identity))
        .transpose()
}

fn definition_from_row(
    row: QueryResult,
    identity: WorkflowIdentity,
) -> Result<StoredWorkflowDefinition, WorkflowRepositoryError> {
    let definition_json: String = row_value(&row, "definition_json")?;
    let definition = WorkflowDefinition::parse_canonical_json(&definition_json)
        .map_err(|_| WorkflowRepositoryError::InvalidRecord)?;
    let scope = scope_from_row(&row)?;
    let provenance = provenance_from_row(&row)?;
    let definition_sha256 = bytes32(row_value(&row, "definition_sha256")?)?;
    let canonical_json =
        serde_json::to_string(&definition).map_err(WorkflowRepositoryError::DefinitionEncoding)?;
    if Sha256::digest(canonical_json.as_bytes()).as_slice() != definition_sha256 {
        return Err(WorkflowRepositoryError::InvalidRecord);
    }
    Ok(StoredWorkflowDefinition {
        identity,
        definition_version: parse_u64(row_value(&row, "definition_version_text")?)?,
        definition,
        definition_sha256,
        creator_principal_id: PrincipalId::from_uuid(row_value(&row, "creator_principal_id")?),
        author_principal_id: PrincipalId::from_uuid(row_value(&row, "author_principal_id")?),
        scope,
        current_definition_version: parse_u64(row_value(&row, "current_definition_version_text")?)?,
        head_revision: parse_u64(row_value(&row, "head_revision_text")?)?,
        lifecycle: WorkflowLifecycle::from_database(&row_value::<String>(
            &row,
            "lifecycle_state",
        )?)?,
        provenance,
        created_at_millis: parse_millis(row_value(&row, "created_at_millis")?)?,
    })
}

async fn load_stored_run(
    transaction: &DatabaseTransaction,
    identity: WorkflowRunIdentity,
    lock_clause: &str,
) -> Result<StoredWorkflowRun, WorkflowRepositoryError> {
    load_optional_stored_run(transaction, identity, lock_clause)
        .await?
        .ok_or(WorkflowRepositoryError::NotFound)
}

async fn load_optional_stored_run(
    transaction: &DatabaseTransaction,
    identity: WorkflowRunIdentity,
    lock_clause: &str,
) -> Result<Option<StoredWorkflowRun>, WorkflowRepositoryError> {
    let Some(run_row) = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            format!("{SELECT_RUN_SQL}{lock_clause}"),
            [
                identity.community_id.as_uuid().into(),
                identity.run_id.into(),
            ],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?
    else {
        return Ok(None);
    };
    let step_rows = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            format!("{SELECT_STEPS_SQL}{lock_clause}"),
            [
                identity.community_id.as_uuid().into(),
                identity.run_id.into(),
            ],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?;
    let retry_rows = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            format!("{SELECT_RETRIES_SQL}{lock_clause}"),
            [
                identity.community_id.as_uuid().into(),
                identity.run_id.into(),
            ],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?;
    stored_run_from_rows(identity, run_row, step_rows, retry_rows).map(Some)
}

fn stored_run_from_rows(
    identity: WorkflowRunIdentity,
    run_row: QueryResult,
    step_rows: Vec<QueryResult>,
    retry_rows: Vec<QueryResult>,
) -> Result<StoredWorkflowRun, WorkflowRepositoryError> {
    let workflow =
        WorkflowIdentity::new(identity.community_id, row_value(&run_row, "workflow_id")?)?;
    let definition_version = parse_u64(row_value(&run_row, "definition_version_text")?)?;
    let steps = step_rows
        .into_iter()
        .enumerate()
        .map(|(expected_index, row)| {
            step_from_row(&row, workflow, definition_version, expected_index)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let retries = retry_rows
        .into_iter()
        .map(|row| retry_from_row(&row, steps.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let run = StoredWorkflowRun {
        identity,
        workflow,
        definition_version,
        trigger_operation_id: row_value(&run_row, "trigger_operation_id")?,
        trigger_kind: WorkflowTriggerKind::from_database(&row_value::<String>(
            &run_row,
            "trigger_kind",
        )?)?,
        trigger_source_id: row_value(&run_row, "trigger_source_id")?,
        trigger_context: parse_json(row_value(&run_row, "trigger_context_json")?)?,
        run_version: parse_u64(row_value(&run_row, "run_version_text")?)?,
        state: WorkflowRunState::from_database(&row_value::<String>(&run_row, "status")?)?,
        current_step_index: parse_u16(row_value(&run_row, "current_step_index")?)?,
        error_code: row_value(&run_row, "error_code")?,
        error_message: row_value(&run_row, "error_message")?,
        provenance: provenance_without_integrity_from_row(&run_row)?,
        created_at_millis: parse_millis(row_value(&run_row, "created_at_millis")?)?,
        started_at_millis: parse_optional_millis(row_value(&run_row, "started_at_millis")?)?,
        completed_at_millis: parse_optional_millis(row_value(&run_row, "completed_at_millis")?)?,
        updated_at_millis: parse_millis(row_value(&run_row, "updated_at_millis")?)?,
        steps,
        retries,
    };
    validate_stored_run(&run)?;
    Ok(run)
}

fn step_from_row(
    row: &QueryResult,
    workflow: WorkflowIdentity,
    definition_version: u64,
    expected_index: usize,
) -> Result<StoredWorkflowStep, WorkflowRepositoryError> {
    let row_workflow_id: Uuid = row_value(row, "workflow_id")?;
    let row_definition_version = parse_u64(row_value(row, "definition_version_text")?)?;
    let index = parse_u16(row_value(row, "step_index")?)?;
    if row_workflow_id != workflow.workflow_id
        || row_definition_version != definition_version
        || usize::from(index) != expected_index
    {
        return Err(WorkflowRepositoryError::InvalidRecord);
    }
    let step = StoredWorkflowStep {
        index,
        step_id: row_value(row, "step_id")?,
        operation_id: row_value(row, "operation_id")?,
        state: WorkflowStepState::from_database(&row_value::<String>(row, "state")?)?,
        attempt_count: parse_u16(row_value(row, "attempt_count")?)?,
        output: row_value::<Option<String>>(row, "output_json")?
            .map(parse_json)
            .transpose()?,
        error_code: row_value(row, "error_code")?,
        error_message: row_value(row, "error_message")?,
        created_at_millis: parse_millis(row_value(row, "created_at_millis")?)?,
        started_at_millis: parse_optional_millis(row_value(row, "started_at_millis")?)?,
        completed_at_millis: parse_optional_millis(row_value(row, "completed_at_millis")?)?,
        updated_at_millis: parse_millis(row_value(row, "updated_at_millis")?)?,
    };
    if step.operation_id.is_nil()
        || step.step_id.is_empty()
        || step.step_id.len() > 64
        || step.attempt_count > 8
        || step.created_at_millis > step.updated_at_millis
        || step.state.is_terminal() != step.completed_at_millis.is_some()
        || matches!(
            step.state,
            WorkflowStepState::Failed | WorkflowStepState::RepairRequired
        ) != step.error_code.is_some()
    {
        return Err(WorkflowRepositoryError::InvalidRecord);
    }
    Ok(step)
}

fn retry_from_row(
    row: &QueryResult,
    step_count: usize,
) -> Result<StoredWorkflowRetry, WorkflowRepositoryError> {
    let retry = StoredWorkflowRetry {
        step_index: parse_u16(row_value(row, "step_index")?)?,
        attempt_number: parse_u16(row_value(row, "attempt_number")?)?,
        retry_operation_id: row_value(row, "retry_operation_id")?,
        failure_class: RetryFailureClass::from_database(&row_value::<String>(
            row,
            "failure_class",
        )?)?,
        state: RetryState::from_database(&row_value::<String>(row, "state")?)?,
        scheduled_at_millis: parse_millis(row_value(row, "scheduled_at_millis")?)?,
        due_at_millis: parse_millis(row_value(row, "due_at_millis")?)?,
        claimed_at_millis: parse_optional_millis(row_value(row, "claimed_at_millis")?)?,
        completed_at_millis: parse_optional_millis(row_value(row, "completed_at_millis")?)?,
    };
    if usize::from(retry.step_index) >= step_count
        || !(2..=8).contains(&retry.attempt_number)
        || retry.retry_operation_id.is_nil()
        || retry.scheduled_at_millis >= retry.due_at_millis
    {
        return Err(WorkflowRepositoryError::InvalidRecord);
    }
    Ok(retry)
}

fn validate_stored_run(run: &StoredWorkflowRun) -> Result<(), WorkflowRepositoryError> {
    if run.trigger_operation_id.is_nil()
        || run.trigger_source_id.is_empty()
        || run.trigger_source_id.len() > MAX_TRIGGER_SOURCE_BYTES
        || run.run_version == 0
        || usize::from(run.current_step_index) > run.steps.len()
        || run.created_at_millis > run.updated_at_millis
        || run.state.is_terminal() != run.completed_at_millis.is_some()
        || matches!(
            run.state,
            WorkflowRunState::Failed | WorkflowRunState::RepairRequired
        ) != run.error_code.is_some()
    {
        return Err(WorkflowRepositoryError::InvalidRecord);
    }
    Ok(())
}

fn validate_definition_write(
    write: &DefinitionVersionWrite,
) -> Result<String, WorkflowRepositoryError> {
    validate_provenance(&write.provenance)?;
    validate_scope(&write.scope)?;
    if write.definition_version == 0
        || write.creator_principal_id.as_uuid().is_nil()
        || write.author_principal_id.as_uuid().is_nil()
        || write.created_at_millis < write.provenance.observed_at_millis
    {
        return Err(WorkflowRepositoryError::InvalidInput);
    }
    serde_json::to_string(&write.definition).map_err(WorkflowRepositoryError::DefinitionEncoding)
}

fn validate_run_request(
    tenant: &TenantContext,
    request: &WorkflowRunRequest,
) -> Result<(), WorkflowRepositoryError> {
    validate_run_identity(tenant, request.identity)?;
    validate_workflow_identity(tenant, request.workflow)?;
    validate_provenance(&request.provenance)?;
    let context_bytes = serde_json::to_vec(&request.trigger_context)
        .map_err(WorkflowRepositoryError::DefinitionEncoding)?;
    if request.identity.community_id != request.workflow.community_id
        || request.definition_version == 0
        || request.trigger_operation_id.is_nil()
        || request.provenance.integrity_sha256.is_some()
        || request.trigger_source_id.is_empty()
        || request.trigger_source_id.len() > MAX_TRIGGER_SOURCE_BYTES
        || !request.trigger_context.is_object()
        || context_bytes.len() > MAX_TRIGGER_CONTEXT_BYTES
        || request.step_operation_ids.is_empty()
        || request.step_operation_ids.len() > 64
        || request.step_operation_ids.iter().any(Uuid::is_nil)
        || request
            .step_operation_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != request.step_operation_ids.len()
        || request.created_at_millis < request.provenance.observed_at_millis
    {
        return Err(WorkflowRepositoryError::InvalidInput);
    }
    Ok(())
}

fn validate_checkpoint(checkpoint: &WorkflowStepCheckpoint) -> Result<(), WorkflowRepositoryError> {
    let output_bytes = checkpoint
        .output
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map_err(WorkflowRepositoryError::DefinitionEncoding)?
        .map_or(0, |bytes| bytes.len());
    if checkpoint.lease.identity != checkpoint.identity
        || checkpoint.expected_run_version == 0
        || checkpoint.operation_id.is_nil()
        || checkpoint.next_step_index > 64
        || checkpoint.attempt_count > 8
        || output_bytes > MAX_OUTPUT_BYTES
        || !valid_optional_text(checkpoint.error_code.as_deref(), MAX_ERROR_CODE_BYTES)
        || !valid_optional_text(checkpoint.error_message.as_deref(), MAX_ERROR_MESSAGE_BYTES)
        || matches!(
            checkpoint.next_step_state,
            WorkflowStepState::Failed | WorkflowStepState::RepairRequired
        ) != checkpoint.error_code.is_some()
        || matches!(
            checkpoint.next_run_state,
            WorkflowRunState::Failed | WorkflowRunState::RepairRequired
        ) != checkpoint.error_code.is_some()
    {
        return Err(WorkflowRepositoryError::InvalidInput);
    }
    Ok(())
}

fn validate_retry_write(
    tenant: &TenantContext,
    retry: &WorkflowRetryWrite,
) -> Result<(), WorkflowRepositoryError> {
    validate_run_identity(tenant, retry.identity)?;
    validate_provenance(&retry.provenance)?;
    if retry.provenance.integrity_sha256.is_some()
        || !(2..=8).contains(&retry.attempt_number)
        || retry.retry_operation_id.is_nil()
        || retry.scheduled_at_millis >= retry.due_at_millis
        || retry.created_at_millis < retry.provenance.observed_at_millis
    {
        return Err(WorkflowRepositoryError::InvalidInput);
    }
    Ok(())
}

fn validate_lease_request(
    tenant: &TenantContext,
    request: &WorkflowRunLeaseRequest,
) -> Result<(), WorkflowRepositoryError> {
    validate_run_identity(tenant, request.identity)?;
    if request.expected_run_version == 0
        || request.lease_id.is_nil()
        || !valid_worker_id(&request.worker_id)
        || request.acquired_at_millis > request.expires_at_millis
        || request.expires_at_millis > request.recovery_after_millis
    {
        return Err(WorkflowRepositoryError::InvalidLease);
    }
    Ok(())
}

fn validate_lease_fence(
    tenant: &TenantContext,
    fence: &WorkflowRunLeaseFence,
) -> Result<(), WorkflowRepositoryError> {
    validate_run_identity(tenant, fence.identity)?;
    if fence.generation == 0 || fence.lease_id.is_nil() || !valid_worker_id(&fence.worker_id) {
        return Err(WorkflowRepositoryError::InvalidLease);
    }
    Ok(())
}

fn valid_worker_id(worker_id: &str) -> bool {
    !worker_id.is_empty()
        && worker_id.len() <= 128
        && worker_id.trim() == worker_id
        && !worker_id.chars().any(char::is_control)
}

fn validate_workflow_identity(
    tenant: &TenantContext,
    identity: WorkflowIdentity,
) -> Result<(), WorkflowRepositoryError> {
    if tenant.community_id() != identity.community_id {
        return Err(WorkflowRepositoryError::TenantBoundaryViolation);
    }
    Ok(())
}

fn validate_run_identity(
    tenant: &TenantContext,
    identity: WorkflowRunIdentity,
) -> Result<(), WorkflowRepositoryError> {
    if tenant.community_id() != identity.community_id {
        return Err(WorkflowRepositoryError::TenantBoundaryViolation);
    }
    Ok(())
}

fn validate_scope(scope: &WorkflowScope) -> Result<(), WorkflowRepositoryError> {
    match scope {
        WorkflowScope::Community => Ok(()),
        WorkflowScope::Project {
            slug,
            record_version,
            ..
        } if !slug.is_empty() && slug.len() <= 1024 && *record_version > 0 => Ok(()),
        WorkflowScope::Project { .. } => Err(WorkflowRepositoryError::InvalidInput),
    }
}

fn validate_provenance(provenance: &WorkflowProvenance) -> Result<(), WorkflowRepositoryError> {
    if provenance.source_system.is_empty()
        || provenance.source_system.len() > MAX_SOURCE_SYSTEM_BYTES
        || provenance.source_record_id.is_empty()
        || provenance.source_record_id.len() > MAX_SOURCE_RECORD_BYTES
        || provenance.source_version.is_empty()
        || provenance.source_version.len() > MAX_SOURCE_VERSION_BYTES
    {
        return Err(WorkflowRepositoryError::InvalidInput);
    }
    Ok(())
}

fn valid_optional_text(value: Option<&str>, maximum: usize) -> bool {
    value.is_none_or(|value| !value.is_empty() && value.len() <= maximum)
}

fn valid_step_transition(from: WorkflowStepState, to: WorkflowStepState) -> bool {
    matches!(
        (from, to),
        (WorkflowStepState::Pending, WorkflowStepState::Running)
            | (WorkflowStepState::Pending, WorkflowStepState::Skipped)
            | (WorkflowStepState::Pending, WorkflowStepState::Cancelled)
            | (WorkflowStepState::Running, WorkflowStepState::Completed)
            | (
                WorkflowStepState::Running,
                WorkflowStepState::WaitingApproval
            )
            | (
                WorkflowStepState::Running,
                WorkflowStepState::RetryScheduled
            )
            | (
                WorkflowStepState::Running,
                WorkflowStepState::RepairRequired
            )
            | (WorkflowStepState::Running, WorkflowStepState::Failed)
            | (WorkflowStepState::Running, WorkflowStepState::Cancelled)
            | (
                WorkflowStepState::WaitingApproval,
                WorkflowStepState::Completed
            )
            | (
                WorkflowStepState::WaitingApproval,
                WorkflowStepState::Failed
            )
            | (
                WorkflowStepState::WaitingApproval,
                WorkflowStepState::Cancelled
            )
            | (
                WorkflowStepState::RetryScheduled,
                WorkflowStepState::Running
            )
            | (
                WorkflowStepState::RetryScheduled,
                WorkflowStepState::Cancelled
            )
            | (
                WorkflowStepState::RepairRequired,
                WorkflowStepState::Running
            )
            | (WorkflowStepState::RepairRequired, WorkflowStepState::Failed)
            | (
                WorkflowStepState::RepairRequired,
                WorkflowStepState::Cancelled
            )
    )
}

fn valid_attempt_transition(
    step: &StoredWorkflowStep,
    checkpoint: &WorkflowStepCheckpoint,
) -> bool {
    match (step.state, checkpoint.next_step_state) {
        (WorkflowStepState::Pending, WorkflowStepState::Running) => checkpoint.attempt_count == 1,
        (WorkflowStepState::Pending, WorkflowStepState::Skipped | WorkflowStepState::Cancelled) => {
            checkpoint.attempt_count == 0
        }
        (WorkflowStepState::RetryScheduled, WorkflowStepState::Running) => {
            checkpoint.attempt_count == step.attempt_count.saturating_add(1)
        }
        _ => checkpoint.attempt_count == step.attempt_count,
    }
}

fn valid_run_step_pair(run: &StoredWorkflowRun, checkpoint: &WorkflowStepCheckpoint) -> bool {
    if usize::from(checkpoint.next_step_index) > run.steps.len() {
        return false;
    }
    match checkpoint.next_step_state {
        WorkflowStepState::Running => {
            checkpoint.next_run_state == WorkflowRunState::Running
                && checkpoint.next_step_index == checkpoint.step_index
        }
        WorkflowStepState::WaitingApproval => {
            checkpoint.next_run_state == WorkflowRunState::WaitingApproval
                && checkpoint.next_step_index == checkpoint.step_index
        }
        WorkflowStepState::RetryScheduled => {
            checkpoint.next_run_state == WorkflowRunState::RetryScheduled
                && checkpoint.next_step_index == checkpoint.step_index
        }
        WorkflowStepState::RepairRequired => {
            checkpoint.next_run_state == WorkflowRunState::RepairRequired
                && checkpoint.next_step_index == checkpoint.step_index
        }
        WorkflowStepState::Failed => checkpoint.next_run_state == WorkflowRunState::Failed,
        WorkflowStepState::Cancelled => checkpoint.next_run_state == WorkflowRunState::Cancelled,
        WorkflowStepState::Completed | WorkflowStepState::Skipped => {
            let expected_next = checkpoint.step_index.saturating_add(1);
            checkpoint.next_step_index == expected_next
                && if usize::from(expected_next) == run.steps.len() {
                    checkpoint.next_run_state == WorkflowRunState::Completed
                } else {
                    checkpoint.next_run_state == WorkflowRunState::Queued
                }
        }
        WorkflowStepState::Pending => false,
    }
}

fn checkpoint_is_duplicate(
    run: &StoredWorkflowRun,
    step: &StoredWorkflowStep,
    checkpoint: &WorkflowStepCheckpoint,
) -> bool {
    run.run_version == checkpoint.expected_run_version.saturating_add(1)
        && run.state == checkpoint.next_run_state
        && run.current_step_index == checkpoint.next_step_index
        && step.state == checkpoint.next_step_state
        && step.attempt_count == checkpoint.attempt_count
        && step.output == checkpoint.output
        && step.error_code == checkpoint.error_code
        && step.error_message == checkpoint.error_message
        && step.updated_at_millis == checkpoint.occurred_at_millis
        && run.updated_at_millis == checkpoint.occurred_at_millis
}

fn run_matches_request(run: &StoredWorkflowRun, request: &WorkflowRunRequest) -> bool {
    run.identity == request.identity
        && run.workflow == request.workflow
        && run.definition_version == request.definition_version
        && run.trigger_operation_id == request.trigger_operation_id
        && run.trigger_kind == request.trigger_kind
        && run.trigger_source_id == request.trigger_source_id
        && run.trigger_context == request.trigger_context
        && run.provenance == request.provenance
        && run.created_at_millis == request.created_at_millis
        && run.steps.len() == request.step_operation_ids.len()
        && run
            .steps
            .iter()
            .zip(&request.step_operation_ids)
            .all(|(step, operation_id)| step.operation_id == *operation_id)
}

fn retry_row_matches(
    row: &QueryResult,
    retry: &WorkflowRetryWrite,
) -> Result<bool, WorkflowRepositoryError> {
    Ok(row_value::<Uuid>(row, "run_id")? == retry.identity.run_id
        && parse_u16(row_value(row, "step_index")?)? == retry.step_index
        && parse_u16(row_value(row, "attempt_number")?)? == retry.attempt_number
        && row_value::<String>(row, "failure_class")? == retry.failure_class.database_name()
        && row_value::<String>(row, "state")? == "scheduled"
        && parse_millis(row_value(row, "scheduled_at_millis")?)? == retry.scheduled_at_millis
        && parse_millis(row_value(row, "due_at_millis")?)? == retry.due_at_millis
        && provenance_without_integrity_from_row(row)? == retry.provenance
        && parse_millis(row_value(row, "created_at_millis")?)? == retry.created_at_millis)
}

fn immutable_definition_fields_match(
    row: &QueryResult,
    write: &DefinitionVersionWrite,
) -> Result<bool, WorkflowRepositoryError> {
    Ok(
        row_value::<Uuid>(row, "creator_principal_id")? == write.creator_principal_id.as_uuid()
            && row_value::<Uuid>(row, "author_principal_id")?
                == write.author_principal_id.as_uuid()
            && scope_from_row(row)? == write.scope
            && provenance_from_row(row)? == write.provenance
            && parse_millis(row_value(row, "created_at_millis")?)? == write.created_at_millis,
    )
}

fn validate_definition_identity(
    row: &QueryResult,
    write: &DefinitionVersionWrite,
) -> Result<(), WorkflowRepositoryError> {
    let creator: Uuid = row_value(row, "creator_principal_id")?;
    if creator != write.creator_principal_id.as_uuid() || scope_from_row(row)? != write.scope {
        return Err(WorkflowRepositoryError::IdempotencyConflict);
    }
    Ok(())
}

fn scope_from_row(row: &QueryResult) -> Result<WorkflowScope, WorkflowRepositoryError> {
    match row_value::<String>(row, "scope_kind")?.as_str() {
        "community" => {
            if row_value::<Option<Vec<u8>>>(row, "project_signer_public_key")?.is_some()
                || row_value::<Option<String>>(row, "project_slug")?.is_some()
                || row_value::<Option<String>>(row, "project_record_version_text")?.is_some()
            {
                return Err(WorkflowRepositoryError::InvalidRecord);
            }
            Ok(WorkflowScope::Community)
        }
        "project" => {
            let signer = bytes32(
                row_value::<Option<Vec<u8>>>(row, "project_signer_public_key")?
                    .ok_or(WorkflowRepositoryError::InvalidRecord)?,
            )?;
            let slug = row_value::<Option<String>>(row, "project_slug")?
                .ok_or(WorkflowRepositoryError::InvalidRecord)?;
            let record_version = parse_u64(
                row_value::<Option<String>>(row, "project_record_version_text")?
                    .ok_or(WorkflowRepositoryError::InvalidRecord)?,
            )?;
            let scope = WorkflowScope::Project {
                signer_public_key: signer,
                slug,
                record_version,
            };
            validate_scope(&scope).map_err(|_| WorkflowRepositoryError::InvalidRecord)?;
            Ok(scope)
        }
        _ => Err(WorkflowRepositoryError::InvalidRecord),
    }
}

fn provenance_from_row(row: &QueryResult) -> Result<WorkflowProvenance, WorkflowRepositoryError> {
    let integrity_sha256 = row_value::<Option<Vec<u8>>>(row, "source_integrity_sha256")?
        .map(bytes32)
        .transpose()?;
    WorkflowProvenance::new(
        row_value::<String>(row, "source_system")?,
        row_value::<String>(row, "source_record_id")?,
        row_value::<String>(row, "source_version")?,
        parse_millis(row_value(row, "source_observed_at_millis")?)?,
        integrity_sha256,
    )
    .map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

fn provenance_without_integrity_from_row(
    row: &QueryResult,
) -> Result<WorkflowProvenance, WorkflowRepositoryError> {
    WorkflowProvenance::new(
        row_value::<String>(row, "source_system")?,
        row_value::<String>(row, "source_record_id")?,
        row_value::<String>(row, "source_version")?,
        parse_millis(row_value(row, "source_observed_at_millis")?)?,
        None,
    )
    .map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

async fn validate_active_lease(
    transaction: &DatabaseTransaction,
    fence: &WorkflowRunLeaseFence,
    occurred_at_millis: u64,
) -> Result<(), WorkflowRepositoryError> {
    let lease = query_lease_by_id(transaction, fence.identity.community_id, fence.lease_id)
        .await?
        .ok_or(WorkflowRepositoryError::LeaseFenceLost)?;
    if !lease_matches_fence(&lease, fence)
        || lease.state != WorkflowRunLeaseState::Active
        || lease.expires_at_millis < occurred_at_millis
    {
        return Err(WorkflowRepositoryError::LeaseFenceLost);
    }
    Ok(())
}

async fn query_lease_by_id(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    lease_id: Uuid,
) -> Result<Option<WorkflowRunLease>, WorkflowRepositoryError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_LEASE_BY_ID_SQL,
            [community_id.as_uuid().into(), lease_id.into()],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?
        .map(|row| lease_from_row(row, community_id))
        .transpose()
}

async fn query_active_lease(
    transaction: &DatabaseTransaction,
    identity: WorkflowRunIdentity,
) -> Result<Option<WorkflowRunLease>, WorkflowRepositoryError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_ACTIVE_LEASE_SQL,
            [
                identity.community_id.as_uuid().into(),
                identity.run_id.into(),
            ],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?
        .map(|row| lease_from_row(row, identity.community_id))
        .transpose()
}

async fn query_maximum_lease_generation(
    transaction: &DatabaseTransaction,
    identity: WorkflowRunIdentity,
) -> Result<u64, WorkflowRepositoryError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_MAXIMUM_LEASE_GENERATION_SQL,
            [
                identity.community_id.as_uuid().into(),
                identity.run_id.into(),
            ],
        ))
        .await
        .map_err(WorkflowRepositoryError::Unavailable)?
        .ok_or(WorkflowRepositoryError::InvalidRecord)?;
    row_value::<String>(&row, "lease_generation_text")?
        .parse()
        .map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

fn lease_from_row(
    row: QueryResult,
    community_id: CommunityId,
) -> Result<WorkflowRunLease, WorkflowRepositoryError> {
    let state = match row_value::<String>(&row, "state")?.as_str() {
        "active" => WorkflowRunLeaseState::Active,
        "released" => WorkflowRunLeaseState::Released,
        _ => return Err(WorkflowRepositoryError::InvalidRecord),
    };
    let release_reason = row_value::<Option<String>>(&row, "release_reason")?
        .as_deref()
        .map(WorkflowRunLeaseReleaseReason::from_database)
        .transpose()?;
    let lease = WorkflowRunLease {
        identity: WorkflowRunIdentity::new(community_id, row_value(&row, "run_id")?)?,
        admitted_run_version: parse_u64(row_value(&row, "run_version_text")?)?,
        generation: parse_u64(row_value(&row, "lease_generation_text")?)?,
        lease_id: row_value(&row, "lease_id")?,
        worker_id: row_value(&row, "worker_id")?,
        state,
        acquired_at_millis: parse_millis(row_value(&row, "acquired_at_millis")?)?,
        last_heartbeat_at_millis: parse_millis(row_value(&row, "last_heartbeat_at_millis")?)?,
        expires_at_millis: parse_millis(row_value(&row, "expires_at_millis")?)?,
        recovery_after_millis: parse_millis(row_value(&row, "recovery_after_millis")?)?,
        released_at_millis: parse_optional_millis(row_value(&row, "released_at_millis")?)?,
        release_reason,
    };
    if lease.lease_id.is_nil()
        || !valid_worker_id(&lease.worker_id)
        || lease.acquired_at_millis > lease.last_heartbeat_at_millis
        || lease.last_heartbeat_at_millis > lease.expires_at_millis
        || lease.expires_at_millis > lease.recovery_after_millis
        || !matches!(
            (lease.state, lease.released_at_millis, lease.release_reason),
            (WorkflowRunLeaseState::Active, None, None)
                | (WorkflowRunLeaseState::Released, Some(_), Some(_))
        )
    {
        return Err(WorkflowRepositoryError::InvalidRecord);
    }
    Ok(lease)
}

fn lease_matches_request(lease: &WorkflowRunLease, request: &WorkflowRunLeaseRequest) -> bool {
    lease.identity == request.identity
        && lease.admitted_run_version == request.expected_run_version
        && lease.lease_id == request.lease_id
        && lease.worker_id == request.worker_id
        && lease.acquired_at_millis == request.acquired_at_millis
        && lease.expires_at_millis == request.expires_at_millis
        && lease.recovery_after_millis == request.recovery_after_millis
}

fn lease_matches_fence(lease: &WorkflowRunLease, fence: &WorkflowRunLeaseFence) -> bool {
    lease.identity == fence.identity
        && lease.generation == fence.generation
        && lease.lease_id == fence.lease_id
        && lease.worker_id == fence.worker_id
}

fn expire_lease_statement(
    lease: &WorkflowRunLease,
    expired_at_millis: u64,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        EXPIRE_LEASE_SQL,
        [
            lease.identity.community_id.as_uuid().into(),
            lease.identity.run_id.into(),
            lease.generation.to_string().into(),
            lease.lease_id.into(),
            millis_value(expired_at_millis)?,
        ],
    ))
}

fn insert_lease_statement(
    request: &WorkflowRunLeaseRequest,
    generation: u64,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_LEASE_SQL,
        [
            request.identity.community_id.as_uuid().into(),
            request.identity.run_id.into(),
            request.expected_run_version.to_string().into(),
            generation.to_string().into(),
            request.lease_id.into(),
            request.worker_id.as_str().into(),
            millis_value(request.acquired_at_millis)?,
            millis_value(request.expires_at_millis)?,
            millis_value(request.recovery_after_millis)?,
        ],
    ))
}

fn heartbeat_lease_statement(
    fence: &WorkflowRunLeaseFence,
    heartbeat_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        HEARTBEAT_LEASE_SQL,
        [
            fence.identity.community_id.as_uuid().into(),
            fence.identity.run_id.into(),
            fence.generation.to_string().into(),
            fence.lease_id.into(),
            fence.worker_id.as_str().into(),
            millis_value(heartbeat_at_millis)?,
            millis_value(heartbeat_at_millis)?,
            millis_value(expires_at_millis)?,
            millis_value(recovery_after_millis)?,
        ],
    ))
}

fn release_lease_statement(
    fence: &WorkflowRunLeaseFence,
    released_at_millis: u64,
    reason: WorkflowRunLeaseReleaseReason,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        RELEASE_LEASE_SQL,
        [
            fence.identity.community_id.as_uuid().into(),
            fence.identity.run_id.into(),
            fence.generation.to_string().into(),
            fence.lease_id.into(),
            fence.worker_id.as_str().into(),
            millis_value(released_at_millis)?,
            reason.database_name().into(),
        ],
    ))
}

fn insert_definition_identity_statement(
    write: &DefinitionVersionWrite,
) -> Result<Statement, WorkflowRepositoryError> {
    let (scope_kind, signer, slug, record_version) = scope_values(&write.scope);
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_DEFINITION_IDENTITY_SQL,
        [
            write.identity.community_id.as_uuid().into(),
            write.identity.workflow_id.into(),
            write.creator_principal_id.as_uuid().into(),
            scope_kind.into(),
            signer.into(),
            slug.into(),
            record_version.into(),
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn insert_definition_version_statement(
    write: &DefinitionVersionWrite,
    encoded: &str,
    definition_sha256: [u8; 32],
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_DEFINITION_VERSION_SQL,
        [
            write.identity.community_id.as_uuid().into(),
            write.identity.workflow_id.into(),
            write.definition_version.to_string().into(),
            i32::try_from(write.definition.version())
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            write.definition.name().into(),
            encoded.into(),
            definition_sha256.to_vec().into(),
            write.author_principal_id.as_uuid().into(),
            write.provenance.source_system.as_str().into(),
            write.provenance.source_record_id.as_str().into(),
            write.provenance.source_version.as_str().into(),
            millis_value(write.provenance.observed_at_millis)?,
            write
                .provenance
                .integrity_sha256
                .map(|value| value.to_vec())
                .into(),
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn insert_definition_head_statement(
    write: &DefinitionVersionWrite,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_DEFINITION_HEAD_SQL,
        [
            write.identity.community_id.as_uuid().into(),
            write.identity.workflow_id.into(),
            write.definition_version.to_string().into(),
            write.lifecycle.database_name().into(),
            write.provenance.source_system.as_str().into(),
            write.provenance.source_record_id.as_str().into(),
            write.provenance.source_version.as_str().into(),
            millis_value(write.provenance.observed_at_millis)?,
            millis_value(write.created_at_millis)?,
        ],
    ))
}

fn update_definition_head_statement(
    write: &DefinitionVersionWrite,
    previous_revision: u64,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_DEFINITION_HEAD_SQL,
        [
            write.identity.community_id.as_uuid().into(),
            write.identity.workflow_id.into(),
            write.definition_version.to_string().into(),
            write.lifecycle.database_name().into(),
            write.provenance.source_system.as_str().into(),
            write.provenance.source_record_id.as_str().into(),
            write.provenance.source_version.as_str().into(),
            millis_value(write.provenance.observed_at_millis)?,
            millis_value(write.created_at_millis)?,
            previous_revision.to_string().into(),
        ],
    ))
}

fn insert_run_statement(
    request: &WorkflowRunRequest,
    context_json: &str,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_RUN_SQL,
        [
            request.identity.community_id.as_uuid().into(),
            request.identity.run_id.into(),
            request.workflow.workflow_id.into(),
            request.definition_version.to_string().into(),
            request.trigger_operation_id.into(),
            request.trigger_kind.database_name().into(),
            request.trigger_source_id.as_str().into(),
            context_json.into(),
            request.provenance.source_system.as_str().into(),
            request.provenance.source_record_id.as_str().into(),
            request.provenance.source_version.as_str().into(),
            millis_value(request.provenance.observed_at_millis)?,
            millis_value(request.created_at_millis)?,
        ],
    ))
}

fn insert_step_statement(
    request: &WorkflowRunRequest,
    step_index: u16,
    step_id: &str,
    operation_id: Uuid,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_STEP_SQL,
        [
            request.identity.community_id.as_uuid().into(),
            request.identity.run_id.into(),
            request.workflow.workflow_id.into(),
            request.definition_version.to_string().into(),
            i16::try_from(step_index)
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            step_id.into(),
            operation_id.into(),
            request.provenance.source_system.as_str().into(),
            format!("run:{}:step:{step_id}", request.identity.run_id).into(),
            request.provenance.source_version.as_str().into(),
            millis_value(request.provenance.observed_at_millis)?,
            millis_value(request.created_at_millis)?,
        ],
    ))
}

fn update_step_statement(
    checkpoint: &WorkflowStepCheckpoint,
    output_json: Option<&str>,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_STEP_SQL,
        [
            checkpoint.identity.community_id.as_uuid().into(),
            checkpoint.identity.run_id.into(),
            i16::try_from(checkpoint.step_index)
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            checkpoint.operation_id.into(),
            checkpoint.next_step_state.database_name().into(),
            i16::try_from(checkpoint.attempt_count)
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            output_json.into(),
            checkpoint.error_code.as_deref().into(),
            checkpoint.error_message.as_deref().into(),
            millis_value(checkpoint.occurred_at_millis)?,
            checkpoint.expected_step_state.database_name().into(),
        ],
    ))
}

fn update_run_statement(
    checkpoint: &WorkflowStepCheckpoint,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_RUN_SQL,
        [
            checkpoint.identity.community_id.as_uuid().into(),
            checkpoint.identity.run_id.into(),
            checkpoint.expected_run_version.to_string().into(),
            checkpoint.next_run_state.database_name().into(),
            i16::try_from(checkpoint.next_step_index)
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            checkpoint.error_code.as_deref().into(),
            checkpoint.error_message.as_deref().into(),
            millis_value(checkpoint.occurred_at_millis)?,
        ],
    ))
}

fn insert_retry_statement(
    retry: &WorkflowRetryWrite,
) -> Result<Statement, WorkflowRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_RETRY_SQL,
        [
            retry.identity.community_id.as_uuid().into(),
            retry.identity.run_id.into(),
            i16::try_from(retry.step_index)
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            i16::try_from(retry.attempt_number)
                .map_err(|_| WorkflowRepositoryError::InvalidInput)?
                .into(),
            retry.retry_operation_id.into(),
            retry.failure_class.database_name().into(),
            millis_value(retry.scheduled_at_millis)?,
            millis_value(retry.due_at_millis)?,
            retry.provenance.source_system.as_str().into(),
            retry.provenance.source_record_id.as_str().into(),
            retry.provenance.source_version.as_str().into(),
            millis_value(retry.provenance.observed_at_millis)?,
            millis_value(retry.created_at_millis)?,
        ],
    ))
}

fn scope_values(
    scope: &WorkflowScope,
) -> (&'static str, Option<Vec<u8>>, Option<&str>, Option<String>) {
    match scope {
        WorkflowScope::Community => ("community", None, None, None),
        WorkflowScope::Project {
            signer_public_key,
            slug,
            record_version,
        } => (
            "project",
            Some(signer_public_key.to_vec()),
            Some(slug.as_str()),
            Some(record_version.to_string()),
        ),
    }
}

fn parse_json(value: String) -> Result<JsonValue, WorkflowRepositoryError> {
    serde_json::from_str(&value).map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

fn parse_u64(value: String) -> Result<u64, WorkflowRepositoryError> {
    let value = value
        .parse::<u64>()
        .map_err(|_| WorkflowRepositoryError::InvalidRecord)?;
    (value > 0)
        .then_some(value)
        .ok_or(WorkflowRepositoryError::InvalidRecord)
}

fn parse_u16(value: i16) -> Result<u16, WorkflowRepositoryError> {
    u16::try_from(value).map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

fn parse_millis(value: i64) -> Result<u64, WorkflowRepositoryError> {
    u64::try_from(value).map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

fn parse_optional_millis(value: Option<i64>) -> Result<Option<u64>, WorkflowRepositoryError> {
    value.map(parse_millis).transpose()
}

fn millis_value(value: u64) -> Result<sea_orm::Value, WorkflowRepositoryError> {
    i64::try_from(value)
        .map(Into::into)
        .map_err(|_| WorkflowRepositoryError::InvalidInput)
}

fn bytes32(value: Vec<u8>) -> Result<[u8; 32], WorkflowRepositoryError> {
    value
        .try_into()
        .map_err(|_| WorkflowRepositoryError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, WorkflowRepositoryError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| WorkflowRepositoryError::InvalidRecord)
}
