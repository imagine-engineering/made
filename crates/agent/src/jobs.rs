use agent_client_protocol::schema::v1 as acp;
use async_trait::async_trait;
use collaboration_domain::{
    AggregateVersion, Job, JobCommand, JobCommandKind, JobCommandOutcome, JobIdentity, JobState,
    JobStateKind, OperationId, PrincipalId, TenantContext,
};
use uuid::Uuid;

use crate::collaboration_session::{
    CollaborationExecutorId, CollaborationSessionError, CollaborationSessionIdentity,
    CollaborationSessionRegistry, CollaborationSessionResolution, CollaborationSessionScope,
};

const MAX_JOB_PROMPT_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobExecutorLease {
    identity: JobIdentity,
    job_version: AggregateVersion,
    generation: AggregateVersion,
    lease_id: Uuid,
    executor_principal_id: PrincipalId,
    acquired_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
}

impl JobExecutorLease {
    pub fn new(
        request: JobExecutorLeaseRequest,
        generation: AggregateVersion,
    ) -> Result<Self, JobExecutionError> {
        request.validate()?;
        Ok(Self {
            identity: request.identity,
            job_version: request.job_version,
            generation,
            lease_id: request.lease_id,
            executor_principal_id: request.executor_principal_id,
            acquired_at_millis: request.acquired_at_millis,
            expires_at_millis: request.expires_at_millis,
            recovery_after_millis: request.recovery_after_millis,
        })
    }

    pub const fn identity(self) -> JobIdentity {
        self.identity
    }

    pub const fn job_version(self) -> AggregateVersion {
        self.job_version
    }

    pub const fn generation(self) -> AggregateVersion {
        self.generation
    }

    pub const fn lease_id(self) -> Uuid {
        self.lease_id
    }

    pub const fn executor_principal_id(self) -> PrincipalId {
        self.executor_principal_id
    }

    pub const fn acquired_at_millis(self) -> u64 {
        self.acquired_at_millis
    }

    pub const fn expires_at_millis(self) -> u64 {
        self.expires_at_millis
    }

    pub const fn recovery_after_millis(self) -> u64 {
        self.recovery_after_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobExecutorLeaseRequest {
    identity: JobIdentity,
    job_version: AggregateVersion,
    lease_id: Uuid,
    executor_principal_id: PrincipalId,
    acquired_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
}

impl JobExecutorLeaseRequest {
    pub const fn identity(self) -> JobIdentity {
        self.identity
    }

    pub const fn job_version(self) -> AggregateVersion {
        self.job_version
    }

    pub const fn lease_id(self) -> Uuid {
        self.lease_id
    }

    pub const fn executor_principal_id(self) -> PrincipalId {
        self.executor_principal_id
    }

    pub const fn acquired_at_millis(self) -> u64 {
        self.acquired_at_millis
    }

    pub const fn expires_at_millis(self) -> u64 {
        self.expires_at_millis
    }

    pub const fn recovery_after_millis(self) -> u64 {
        self.recovery_after_millis
    }

    fn validate(self) -> Result<(), JobExecutionError> {
        if self.lease_id.is_nil()
            || self.executor_principal_id.as_uuid().is_nil()
            || self.acquired_at_millis > self.expires_at_millis
            || self.expires_at_millis > self.recovery_after_millis
        {
            return Err(JobExecutionError::InvalidRequest);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobLeaseReleaseReason {
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobLeaseMutationOutcome {
    Applied,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum JobExecutionAuthorityError {
    #[error("job execution authority rejected a conflicting transition")]
    Conflict,
    #[error("job executor lease is not recoverable yet")]
    LeaseUnavailable,
    #[error("job executor lease is no longer authoritative")]
    LeaseLost,
    #[error("job execution authority is unavailable")]
    Unavailable,
}

#[async_trait(?Send)]
pub trait JobExecutionAuthority {
    async fn load_job(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<Job>, JobExecutionAuthorityError>;

    async fn acquire_executor_lease(
        &self,
        tenant: &TenantContext,
        request: JobExecutorLeaseRequest,
    ) -> Result<JobExecutorLease, JobExecutionAuthorityError>;

    async fn active_executor_lease(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<JobExecutorLease>, JobExecutionAuthorityError>;

    async fn transition(
        &self,
        tenant: &TenantContext,
        command: JobCommand,
    ) -> Result<JobCommandOutcome, JobExecutionAuthorityError>;

    async fn release_executor_lease(
        &self,
        tenant: &TenantContext,
        lease: JobExecutorLease,
        released_at_millis: u64,
        reason: JobLeaseReleaseReason,
    ) -> Result<JobLeaseMutationOutcome, JobExecutionAuthorityError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum NativeJobRuntimeError {
    #[error("native job runtime is unavailable")]
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeJobRunOutcome {
    Completed {
        completed_at_millis: u64,
    },
    Failed {
        failed_at_millis: u64,
    },
    CancellationRequested {
        actor_principal_id: PrincipalId,
        cancelled_at_millis: u64,
    },
    Crashed,
}

#[async_trait(?Send)]
pub trait NativeJobRuntime {
    async fn create_session(
        &mut self,
        identity: CollaborationSessionIdentity,
    ) -> Result<acp::SessionId, NativeJobRuntimeError>;

    async fn resume_session(
        &mut self,
        session_id: &acp::SessionId,
    ) -> Result<(), NativeJobRuntimeError>;

    async fn run_job(
        &mut self,
        request: acp::PromptRequest,
    ) -> Result<NativeJobRunOutcome, NativeJobRuntimeError>;

    async fn cancel_session(
        &mut self,
        session_id: &acp::SessionId,
    ) -> Result<(), NativeJobRuntimeError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobExecutionRequest {
    identity: JobIdentity,
    executor_principal_id: PrincipalId,
    channel_id: Uuid,
    thread_id: Option<Uuid>,
    lease_id: Uuid,
    acquired_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
    prompt: String,
}

impl JobExecutionRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: JobIdentity,
        executor_principal_id: PrincipalId,
        channel_id: Uuid,
        thread_id: Option<Uuid>,
        lease_id: Uuid,
        acquired_at_millis: u64,
        expires_at_millis: u64,
        recovery_after_millis: u64,
        prompt: impl Into<String>,
    ) -> Result<Self, JobExecutionError> {
        let prompt = prompt.into();
        if executor_principal_id.as_uuid().is_nil()
            || channel_id.is_nil()
            || thread_id.is_some_and(|thread_id| thread_id.is_nil())
            || lease_id.is_nil()
            || acquired_at_millis > expires_at_millis
            || expires_at_millis > recovery_after_millis
            || prompt.trim().is_empty()
            || prompt.len() > MAX_JOB_PROMPT_BYTES
        {
            return Err(JobExecutionError::InvalidRequest);
        }
        Ok(Self {
            identity,
            executor_principal_id,
            channel_id,
            thread_id,
            lease_id,
            acquired_at_millis,
            expires_at_millis,
            recovery_after_millis,
            prompt,
        })
    }

    pub const fn identity(&self) -> JobIdentity {
        self.identity
    }

    pub const fn executor_principal_id(&self) -> PrincipalId {
        self.executor_principal_id
    }

    pub const fn channel_id(&self) -> Uuid {
        self.channel_id
    }

    pub const fn thread_id(&self) -> Option<Uuid> {
        self.thread_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobExecutionDisposition {
    Completed,
    Failed,
    Cancelled,
    Crashed,
    AlreadyTerminal(JobStateKind),
}

#[derive(Debug, thiserror::Error)]
pub enum JobExecutionError {
    #[error("job execution request is invalid")]
    InvalidRequest,
    #[error("job execution request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("job does not exist")]
    NotFound,
    #[error("job is not assigned to this executor")]
    ExecutorMismatch,
    #[error("job is not executable in its current state")]
    InvalidState,
    #[error("job executor lease expired before terminal publication")]
    LeaseExpired,
    #[error("job executor lease is no longer authoritative")]
    LeaseLost,
    #[error(transparent)]
    Authority(#[from] JobExecutionAuthorityError),
    #[error(transparent)]
    Runtime(#[from] NativeJobRuntimeError),
    #[error(transparent)]
    Session(#[from] CollaborationSessionError),
    #[error("job transition could not be represented")]
    InvalidTransition,
}

#[derive(Default)]
pub struct JobExecutionCoordinator {
    sessions: CollaborationSessionRegistry,
}

impl JobExecutionCoordinator {
    pub async fn execute_once<A, R>(
        &mut self,
        authority: &A,
        runtime: &mut R,
        tenant: &TenantContext,
        request: &JobExecutionRequest,
    ) -> Result<JobExecutionDisposition, JobExecutionError>
    where
        A: JobExecutionAuthority,
        R: NativeJobRuntime,
    {
        if tenant.community_id() != request.identity.community_id() {
            return Err(JobExecutionError::TenantBoundaryViolation);
        }
        let mut job = authority
            .load_job(tenant, request.identity)
            .await?
            .ok_or(JobExecutionError::NotFound)?;
        if job.state().is_terminal() {
            if let Some(active_lease) = authority
                .active_executor_lease(tenant, request.identity)
                .await?
                .filter(|lease| lease.executor_principal_id() == request.executor_principal_id)
            {
                match authority
                    .release_executor_lease(
                        tenant,
                        active_lease,
                        job.updated_at_millis(),
                        terminal_release_reason(job.state())?,
                    )
                    .await
                {
                    Ok(_) | Err(JobExecutionAuthorityError::LeaseLost) => {}
                    Err(error) => return Err(error.into()),
                }
            }
            return Ok(JobExecutionDisposition::AlreadyTerminal(job.state().kind()));
        }
        if !matches!(
            job.state(),
            JobState::Accepted { .. } | JobState::InProgress { .. }
        ) {
            return Err(JobExecutionError::InvalidState);
        }
        if job.state().executor_principal_id() != Some(request.executor_principal_id) {
            return Err(JobExecutionError::ExecutorMismatch);
        }

        let lease_request = JobExecutorLeaseRequest {
            identity: request.identity,
            job_version: job.version(),
            lease_id: request.lease_id,
            executor_principal_id: request.executor_principal_id,
            acquired_at_millis: request.acquired_at_millis,
            expires_at_millis: request.expires_at_millis,
            recovery_after_millis: request.recovery_after_millis,
        };
        let lease = authority
            .acquire_executor_lease(tenant, lease_request)
            .await?;
        if !lease_matches_request(lease, lease_request) {
            return Err(JobExecutionError::LeaseLost);
        }

        let session_identity = CollaborationSessionIdentity::new(
            request.identity.community_id().as_uuid(),
            CollaborationSessionScope::Job {
                channel_id: request.channel_id,
                thread_id: request.thread_id,
                job_id: request.identity.job_id().as_uuid(),
            },
        )?;
        let executor_id = CollaborationExecutorId::new(request.executor_principal_id.as_uuid())?;
        let resolution = self.sessions.resolve(session_identity, executor_id)?;
        let session_lease = resolution.lease().clone();
        let session_id = match resolution {
            CollaborationSessionResolution::Create(_) => {
                let session_id = runtime.create_session(session_identity).await?;
                self.sessions.activate(&session_lease, session_id.clone())?;
                session_id
            }
            CollaborationSessionResolution::Resume { session_id, .. } => {
                runtime.resume_session(&session_id).await?;
                session_id
            }
        };

        if matches!(job.state(), JobState::Accepted { .. }) {
            let progress = transition_command(
                &job,
                lease,
                b"progress",
                request.acquired_at_millis,
                JobCommandKind::Progress {
                    executor_principal_id: request.executor_principal_id,
                },
            )?;
            authority.transition(tenant, progress.clone()).await?;
            job.apply(progress)
                .map_err(|_| JobExecutionError::InvalidTransition)?;
        }

        let outcome = runtime
            .run_job(acp::PromptRequest::new(
                session_id.clone(),
                vec![request.prompt.clone().into()],
            ))
            .await?;
        let (terminal_at_millis, kind, release_reason, disposition, cancel_session) = match outcome
        {
            NativeJobRunOutcome::Completed {
                completed_at_millis,
            } => (
                completed_at_millis,
                JobCommandKind::Result {
                    executor_principal_id: request.executor_principal_id,
                },
                JobLeaseReleaseReason::Completed,
                JobExecutionDisposition::Completed,
                false,
            ),
            NativeJobRunOutcome::Failed { failed_at_millis } => (
                failed_at_millis,
                JobCommandKind::Error {
                    actor_principal_id: request.executor_principal_id,
                },
                JobLeaseReleaseReason::Failed,
                JobExecutionDisposition::Failed,
                false,
            ),
            NativeJobRunOutcome::CancellationRequested {
                actor_principal_id,
                cancelled_at_millis,
            } => (
                cancelled_at_millis,
                JobCommandKind::Cancel { actor_principal_id },
                JobLeaseReleaseReason::Cancelled,
                JobExecutionDisposition::Cancelled,
                true,
            ),
            NativeJobRunOutcome::Crashed => return Ok(JobExecutionDisposition::Crashed),
        };

        if terminal_at_millis >= lease.recovery_after_millis() {
            return Err(JobExecutionError::LeaseExpired);
        }
        let active_lease = authority
            .active_executor_lease(tenant, request.identity)
            .await?
            .ok_or(JobExecutionError::LeaseLost)?;
        if active_lease != lease {
            return Err(JobExecutionError::LeaseLost);
        }
        let cancellation = if cancel_session {
            let cancellation = self.sessions.authorize_cancellation(&session_lease)?;
            runtime.cancel_session(cancellation.session_id()).await?;
            Some(cancellation)
        } else {
            None
        };
        let terminal = transition_command(
            &job,
            lease,
            terminal_operation_name(kind),
            terminal_at_millis,
            kind,
        )?;
        authority.transition(tenant, terminal).await?;
        authority
            .release_executor_lease(tenant, lease, terminal_at_millis, release_reason)
            .await?;
        if let Some(cancellation) = cancellation {
            self.sessions.complete_cancellation(&cancellation)?;
        }
        Ok(disposition)
    }
}

fn lease_matches_request(lease: JobExecutorLease, request: JobExecutorLeaseRequest) -> bool {
    lease.identity == request.identity
        && lease.job_version == request.job_version
        && lease.lease_id == request.lease_id
        && lease.executor_principal_id == request.executor_principal_id
        && lease.acquired_at_millis == request.acquired_at_millis
        && lease.expires_at_millis == request.expires_at_millis
        && lease.recovery_after_millis == request.recovery_after_millis
}

fn transition_command(
    job: &Job,
    lease: JobExecutorLease,
    operation_name: &[u8],
    occurred_at_millis: u64,
    kind: JobCommandKind,
) -> Result<JobCommand, JobExecutionError> {
    let version = job
        .version()
        .next()
        .ok_or(JobExecutionError::InvalidTransition)?;
    JobCommand::new(
        job.identity(),
        OperationId::from_uuid(Uuid::new_v5(&lease.lease_id(), operation_name)),
        version,
        occurred_at_millis,
        kind,
    )
    .map_err(|_| JobExecutionError::InvalidTransition)
}

const fn terminal_operation_name(kind: JobCommandKind) -> &'static [u8] {
    match kind {
        JobCommandKind::Result { .. } => b"result",
        JobCommandKind::Cancel { .. } => b"cancel",
        JobCommandKind::Error { .. } => b"error",
        JobCommandKind::Request { .. }
        | JobCommandKind::Accept { .. }
        | JobCommandKind::Progress { .. } => b"invalid",
    }
}

const fn terminal_release_reason(
    state: JobState,
) -> Result<JobLeaseReleaseReason, JobExecutionError> {
    match state {
        JobState::Completed { .. } => Ok(JobLeaseReleaseReason::Completed),
        JobState::Cancelled { .. } => Ok(JobLeaseReleaseReason::Cancelled),
        JobState::Failed { .. } => Ok(JobLeaseReleaseReason::Failed),
        JobState::Requested | JobState::Accepted { .. } | JobState::InProgress { .. } => {
            Err(JobExecutionError::InvalidState)
        }
    }
}
