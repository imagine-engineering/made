use collaboration_domain::TenantContext;
use sea_orm::{DatabaseConnection, DbErr};

use super::repository::{
    WorkflowRepository, WorkflowRepositoryError, WorkflowRunLeaseAcquisition,
    WorkflowRunLeaseFence, WorkflowRunLeaseReleaseReason, WorkflowRunLeaseRequest,
    WorkflowRunRequest, WorkflowStoreOutcome,
};

pub const MAX_QUEUED_RUNS_PER_COMMUNITY: u32 = 1_000;
pub const MAX_QUEUED_RUNS_PER_DEPLOYMENT: u32 = 10_000;
pub const MAX_CONCURRENT_RUNS_PER_COMMUNITY: u32 = 16;
pub const MAX_CONCURRENT_RUNS_PER_DEFINITION: u32 = 4;

pub(super) const SELECT_QUEUE_OBSERVATION_SQL: &str = r#"
SELECT
    community_queue_depth,
    community_oldest_at_millis,
    deployment_queue_depth,
    deployment_oldest_at_millis
FROM public.collaboration_workflow_observe_ready_queue($1)
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowCapacityScope {
    CommunityQueue,
    DeploymentQueue,
    CommunityExecution,
    DefinitionExecution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkflowQueueAdmission {
    Queued,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkflowQueueObservation {
    pub community_queue_depth: u32,
    pub deployment_queue_depth: u32,
    pub oldest_queued_seconds: Option<u64>,
}

pub struct WorkflowScheduler {
    repository: WorkflowRepository,
}

impl WorkflowScheduler {
    pub fn new(connection: DatabaseConnection) -> Result<Self, WorkflowRepositoryError> {
        Ok(Self {
            repository: WorkflowRepository::new(connection)?,
        })
    }

    pub async fn queue_run(
        &self,
        tenant: &TenantContext,
        request: &WorkflowRunRequest,
    ) -> Result<WorkflowQueueAdmission, WorkflowRepositoryError> {
        match self.repository.claim_run(tenant, request).await? {
            WorkflowStoreOutcome::Applied => Ok(WorkflowQueueAdmission::Queued),
            WorkflowStoreOutcome::Duplicate => Ok(WorkflowQueueAdmission::Duplicate),
        }
    }

    pub async fn acquire_execution(
        &self,
        tenant: &TenantContext,
        request: &WorkflowRunLeaseRequest,
    ) -> Result<WorkflowRunLeaseAcquisition, WorkflowRepositoryError> {
        self.repository.acquire_run_lease(tenant, request).await
    }

    pub async fn release_execution(
        &self,
        tenant: &TenantContext,
        fence: &WorkflowRunLeaseFence,
        released_at_millis: u64,
        reason: WorkflowRunLeaseReleaseReason,
    ) -> Result<WorkflowStoreOutcome, WorkflowRepositoryError> {
        self.repository
            .release_run_lease(tenant, fence, released_at_millis, reason)
            .await
    }

    pub async fn observe_queue(
        &self,
        tenant: &TenantContext,
        now_millis: u64,
    ) -> Result<WorkflowQueueObservation, WorkflowRepositoryError> {
        self.repository.observe_queue(tenant, now_millis).await
    }
}

pub(crate) fn capacity_scope_from_database_error(error: &DbErr) -> Option<WorkflowCapacityScope> {
    let message = error.to_string();
    if message.contains("community_queue") {
        Some(WorkflowCapacityScope::CommunityQueue)
    } else if message.contains("deployment_queue") {
        Some(WorkflowCapacityScope::DeploymentQueue)
    } else if message.contains("community_execution") {
        Some(WorkflowCapacityScope::CommunityExecution)
    } else if message.contains("definition_execution") {
        Some(WorkflowCapacityScope::DefinitionExecution)
    } else {
        None
    }
}
