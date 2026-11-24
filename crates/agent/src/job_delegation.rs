use std::collections::BTreeSet;

use async_trait::async_trait;
use collaboration_domain::{
    Job, JobAuthorizationDecision, JobAuthorizationDenial, JobAuthorizationRequest, JobCommand,
    JobCommandKind, JobCommandOutcome, JobCommandType, JobIdentity, JobState, JobStateKind,
    MAX_ACTIVE_CHILD_JOBS, MAX_JOB_DELEGATION_DEPTH, OperationId, PrincipalId, TenantContext,
    authorize_job_transition,
};
use uuid::Uuid;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DelegatedStoredJob {
    job: Job,
    ancestry: Vec<JobIdentity>,
}

impl DelegatedStoredJob {
    pub fn new(job: Job, ancestry: Vec<JobIdentity>) -> Result<Self, JobDelegationError> {
        validate_ancestry(job.identity(), &ancestry)?;
        Ok(Self { job, ancestry })
    }

    pub const fn job(&self) -> &Job {
        &self.job
    }

    pub fn ancestry(&self) -> &[JobIdentity] {
        &self.ancestry
    }

    pub fn apply(&mut self, command: JobCommand) -> Result<JobCommandOutcome, JobDelegationError> {
        self.job
            .apply(command)
            .map_err(|_| JobDelegationError::InvalidTransition)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DelegatedJobStoreOutcome {
    Applied,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum JobDelegationAuthorityError {
    #[error("delegated job authority rejected a conflicting operation")]
    Conflict,
    #[error("delegated job authority is unavailable")]
    Unavailable,
}

#[async_trait(?Send)]
pub trait JobDelegationAuthority {
    async fn load_job(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<DelegatedStoredJob>, JobDelegationAuthorityError>;

    async fn create_child(
        &self,
        tenant: &TenantContext,
        command: JobCommand,
        ancestry: Vec<JobIdentity>,
    ) -> Result<DelegatedJobStoreOutcome, JobDelegationAuthorityError>;

    async fn transition(
        &self,
        tenant: &TenantContext,
        command: JobCommand,
    ) -> Result<JobCommandOutcome, JobDelegationAuthorityError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthorizedChildJobRequest {
    command: JobCommand,
    ancestry: Vec<JobIdentity>,
}

impl AuthorizedChildJobRequest {
    pub fn authorize(request: &JobAuthorizationRequest<'_>) -> Result<Self, JobDelegationError> {
        if request.command.kind().command_type() != JobCommandType::Request
            || request.job.history() != std::slice::from_ref(request.command)
            || request.delegation.is_none()
        {
            return Err(JobDelegationError::InvalidChild);
        }
        match authorize_job_transition(request) {
            JobAuthorizationDecision::Allowed => {}
            JobAuthorizationDecision::Denied(denial) => {
                return Err(JobDelegationError::Unauthorized(denial));
            }
        }
        if request.ancestry.is_empty() {
            return Err(JobDelegationError::InvalidTree);
        }
        validate_ancestry(request.job.identity(), request.ancestry)?;
        Ok(Self {
            command: request.command.clone(),
            ancestry: request.ancestry.to_vec(),
        })
    }

    pub fn command(&self) -> &JobCommand {
        &self.command
    }

    pub fn ancestry(&self) -> &[JobIdentity] {
        &self.ancestry
    }

    pub fn parent_identity(&self) -> Option<JobIdentity> {
        self.ancestry.last().copied()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildCreationDisposition {
    Created,
    Existing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParentAggregationDisposition {
    Pending { remaining_children: usize },
    Completed,
    Failed,
    AlreadyTerminal(JobStateKind),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeCancellationDisposition {
    pub cancelled_children: usize,
    pub already_terminal_children: usize,
    pub parent_cancelled: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum JobDelegationError {
    #[error("delegated child job is invalid")]
    InvalidChild,
    #[error("delegated child job was not authorized")]
    Unauthorized(JobAuthorizationDenial),
    #[error("delegated job tree is invalid")]
    InvalidTree,
    #[error("delegated job does not exist")]
    NotFound,
    #[error("delegated parent job is not executable")]
    ParentNotExecutable,
    #[error("delegated job transition is invalid")]
    InvalidTransition,
    #[error(transparent)]
    Authority(#[from] JobDelegationAuthorityError),
}

#[derive(Default)]
pub struct DelegatedJobOrchestrator;

impl DelegatedJobOrchestrator {
    pub async fn create_child<A: JobDelegationAuthority>(
        &self,
        authority: &A,
        tenant: &TenantContext,
        request: &AuthorizedChildJobRequest,
    ) -> Result<ChildCreationDisposition, JobDelegationError> {
        if tenant.community_id() != request.command.identity().community_id() {
            return Err(JobDelegationError::InvalidTree);
        }
        let (parent_identity, parent_ancestry) = request
            .ancestry
            .split_last()
            .ok_or(JobDelegationError::InvalidTree)?;
        let parent = authority
            .load_job(tenant, *parent_identity)
            .await?
            .ok_or(JobDelegationError::NotFound)?;
        if parent.ancestry != parent_ancestry {
            return Err(JobDelegationError::InvalidTree);
        }
        if !matches!(
            parent.job.state(),
            JobState::Accepted { .. } | JobState::InProgress { .. }
        ) {
            return Err(JobDelegationError::ParentNotExecutable);
        }
        let outcome = authority
            .create_child(tenant, request.command.clone(), request.ancestry.clone())
            .await?;
        let stored = authority
            .load_job(tenant, request.command.identity())
            .await?
            .ok_or(JobDelegationError::NotFound)?;
        if stored.job.history() != std::slice::from_ref(&request.command)
            || stored.ancestry != request.ancestry
            || !is_direct_child(&parent, &stored)
        {
            return Err(JobDelegationError::InvalidTree);
        }
        Ok(match outcome {
            DelegatedJobStoreOutcome::Applied => ChildCreationDisposition::Created,
            DelegatedJobStoreOutcome::Duplicate => ChildCreationDisposition::Existing,
        })
    }

    pub async fn aggregate_parent<A: JobDelegationAuthority>(
        &self,
        authority: &A,
        tenant: &TenantContext,
        parent_identity: JobIdentity,
        child_identities: &[JobIdentity],
        actor_principal_id: PrincipalId,
        occurred_at_millis: u64,
    ) -> Result<ParentAggregationDisposition, JobDelegationError> {
        validate_children(parent_identity, child_identities)?;
        let parent = authority
            .load_job(tenant, parent_identity)
            .await?
            .ok_or(JobDelegationError::NotFound)?;
        if parent.job.state().is_terminal() {
            return Ok(ParentAggregationDisposition::AlreadyTerminal(
                parent.job.state().kind(),
            ));
        }
        if !matches!(
            parent.job.state(),
            JobState::Accepted { .. } | JobState::InProgress { .. }
        ) {
            return Err(JobDelegationError::ParentNotExecutable);
        }

        let mut remaining_children = 0;
        let mut has_failed_child = false;
        for child_identity in child_identities {
            let child = authority
                .load_job(tenant, *child_identity)
                .await?
                .ok_or(JobDelegationError::NotFound)?;
            if !is_direct_child(&parent, &child) {
                return Err(JobDelegationError::InvalidTree);
            }
            match child.job.state() {
                JobState::Completed { .. } => {}
                JobState::Cancelled { .. } | JobState::Failed { .. } => {
                    has_failed_child = true;
                }
                JobState::Requested | JobState::Accepted { .. } | JobState::InProgress { .. } => {
                    remaining_children += 1;
                }
            }
        }
        if remaining_children > 0 {
            return Ok(ParentAggregationDisposition::Pending { remaining_children });
        }

        let (operation_name, kind, disposition) = if has_failed_child {
            (
                b"children-failed".as_slice(),
                JobCommandKind::Error { actor_principal_id },
                ParentAggregationDisposition::Failed,
            )
        } else {
            let executor_principal_id = parent
                .job
                .state()
                .executor_principal_id()
                .ok_or(JobDelegationError::ParentNotExecutable)?;
            (
                b"children-completed".as_slice(),
                JobCommandKind::Result {
                    executor_principal_id,
                },
                ParentAggregationDisposition::Completed,
            )
        };
        let command = tree_command(
            &parent.job,
            parent_identity.job_id().as_uuid(),
            operation_name,
            occurred_at_millis,
            kind,
        )?;
        authority.transition(tenant, command).await?;
        Ok(disposition)
    }

    pub async fn cancel_tree<A: JobDelegationAuthority>(
        &self,
        authority: &A,
        tenant: &TenantContext,
        parent_identity: JobIdentity,
        child_identities: &[JobIdentity],
        actor_principal_id: PrincipalId,
        occurred_at_millis: u64,
    ) -> Result<TreeCancellationDisposition, JobDelegationError> {
        validate_children(parent_identity, child_identities)?;
        let parent = authority
            .load_job(tenant, parent_identity)
            .await?
            .ok_or(JobDelegationError::NotFound)?;
        let mut children = Vec::with_capacity(child_identities.len());
        for child_identity in child_identities {
            let child = authority
                .load_job(tenant, *child_identity)
                .await?
                .ok_or(JobDelegationError::NotFound)?;
            if !is_direct_child(&parent, &child) {
                return Err(JobDelegationError::InvalidTree);
            }
            children.push(child);
        }

        let mut cancelled_children = 0;
        let mut already_terminal_children = 0;
        for child in children {
            if child.job.state().is_terminal() {
                already_terminal_children += 1;
                continue;
            }
            let command = tree_command(
                &child.job,
                child.job.identity().job_id().as_uuid(),
                b"parent-cancel",
                occurred_at_millis,
                JobCommandKind::Cancel { actor_principal_id },
            )?;
            authority.transition(tenant, command).await?;
            cancelled_children += 1;
        }

        let parent_cancelled = if parent.job.state().is_terminal() {
            false
        } else {
            let command = tree_command(
                &parent.job,
                parent_identity.job_id().as_uuid(),
                b"tree-cancel",
                occurred_at_millis,
                JobCommandKind::Cancel { actor_principal_id },
            )?;
            authority.transition(tenant, command).await?;
            true
        };
        Ok(TreeCancellationDisposition {
            cancelled_children,
            already_terminal_children,
            parent_cancelled,
        })
    }
}

fn is_direct_child(parent: &DelegatedStoredJob, child: &DelegatedStoredJob) -> bool {
    match child.ancestry.split_last() {
        Some((parent_identity, ancestry)) => {
            *parent_identity == parent.job.identity() && ancestry == parent.ancestry
        }
        None => false,
    }
}

fn validate_ancestry(
    identity: JobIdentity,
    ancestry: &[JobIdentity],
) -> Result<(), JobDelegationError> {
    if ancestry.len() > MAX_JOB_DELEGATION_DEPTH
        || ancestry.iter().any(|ancestor| {
            ancestor.community_id() != identity.community_id() || ancestor == &identity
        })
        || ancestry.iter().copied().collect::<BTreeSet<_>>().len() != ancestry.len()
    {
        return Err(JobDelegationError::InvalidTree);
    }
    Ok(())
}

fn validate_children(
    parent_identity: JobIdentity,
    child_identities: &[JobIdentity],
) -> Result<(), JobDelegationError> {
    if child_identities.is_empty()
        || child_identities.len() > MAX_ACTIVE_CHILD_JOBS
        || child_identities.iter().any(|child| {
            child.community_id() != parent_identity.community_id() || child == &parent_identity
        })
        || child_identities
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != child_identities.len()
    {
        return Err(JobDelegationError::InvalidTree);
    }
    Ok(())
}

fn tree_command(
    job: &Job,
    namespace: Uuid,
    operation_name: &[u8],
    occurred_at_millis: u64,
    kind: JobCommandKind,
) -> Result<JobCommand, JobDelegationError> {
    let version = job
        .version()
        .next()
        .ok_or(JobDelegationError::InvalidTransition)?;
    JobCommand::new(
        job.identity(),
        OperationId::from_uuid(Uuid::new_v5(&namespace, operation_name)),
        version,
        occurred_at_millis,
        kind,
    )
    .map_err(|_| JobDelegationError::InvalidTransition)
}
