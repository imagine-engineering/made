use std::{cell::RefCell, collections::VecDeque};

use agent::jobs::{
    JobExecutionAuthority, JobExecutionAuthorityError, JobExecutionCoordinator,
    JobExecutionDisposition, JobExecutionError, JobExecutionRequest, JobExecutorLease,
    JobExecutorLeaseRequest, JobLeaseMutationOutcome, JobLeaseReleaseReason, NativeJobRunOutcome,
    NativeJobRuntime, NativeJobRuntimeError,
};
use agent_client_protocol::schema::v1 as acp;
use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, Job, JobCommand, JobCommandKind, JobCommandOutcome,
    JobIdentity, JobState, OperationId, PrincipalId, TenantContext, TrustedTenantRoute,
};
use futures::executor::block_on;
use uuid::Uuid;

const COMMUNITY: u128 = 1;
const JOB: u128 = 2;
const REQUESTER: u128 = 3;
const EXECUTOR: u128 = 4;
const CHANNEL: u128 = 5;

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn identity() -> JobIdentity {
    JobIdentity::new(
        CommunityId::from_uuid(Uuid::from_u128(COMMUNITY)),
        AggregateId::from_uuid(Uuid::from_u128(JOB)),
    )
    .expect("job identity")
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(
                CommunityId::from_uuid(Uuid::from_u128(COMMUNITY)),
                "job-execution-test",
            )
            .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn command(version: u64, occurred_at_millis: u64, kind: JobCommandKind) -> JobCommand {
    JobCommand::new(
        identity(),
        OperationId::from_uuid(Uuid::from_u128(100 + u128::from(version))),
        AggregateVersion::new(version).expect("positive version"),
        occurred_at_millis,
        kind,
    )
    .expect("job command")
}

fn accepted_job() -> Job {
    let mut job = Job::request(command(
        1,
        10,
        JobCommandKind::Request {
            requester_principal_id: principal(REQUESTER),
            target_executor_principal_id: principal(EXECUTOR),
        },
    ))
    .expect("requested job");
    job.apply(command(
        2,
        20,
        JobCommandKind::Accept {
            executor_principal_id: principal(EXECUTOR),
        },
    ))
    .expect("accepted job");
    job
}

fn request(
    lease_id: u128,
    acquired_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
) -> JobExecutionRequest {
    JobExecutionRequest::new(
        identity(),
        principal(EXECUTOR),
        Uuid::from_u128(CHANNEL),
        None,
        Uuid::from_u128(lease_id),
        acquired_at_millis,
        expires_at_millis,
        recovery_after_millis,
        "Run the accepted collaboration job",
    )
    .expect("execution request")
}

struct AuthorityState {
    job: Job,
    active_lease: Option<JobExecutorLease>,
    next_generation: u64,
    released: Vec<(JobExecutorLease, JobLeaseReleaseReason)>,
    result_publications: usize,
}

struct FakeAuthority {
    state: RefCell<AuthorityState>,
}

impl FakeAuthority {
    fn new(job: Job) -> Self {
        Self {
            state: RefCell::new(AuthorityState {
                job,
                active_lease: None,
                next_generation: 0,
                released: Vec::new(),
                result_publications: 0,
            }),
        }
    }
}

#[async_trait(?Send)]
impl JobExecutionAuthority for FakeAuthority {
    async fn load_job(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<Job>, JobExecutionAuthorityError> {
        if tenant.community_id() != identity.community_id() {
            return Err(JobExecutionAuthorityError::Unavailable);
        }
        let state = self.state.borrow();
        Ok((state.job.identity() == identity).then(|| state.job.clone()))
    }

    async fn acquire_executor_lease(
        &self,
        _tenant: &TenantContext,
        request: JobExecutorLeaseRequest,
    ) -> Result<JobExecutorLease, JobExecutionAuthorityError> {
        let mut state = self.state.borrow_mut();
        if let Some(active) = state.active_lease {
            if active.identity() == request.identity()
                && active.job_version() == request.job_version()
                && active.lease_id() == request.lease_id()
                && active.executor_principal_id() == request.executor_principal_id()
                && active.acquired_at_millis() == request.acquired_at_millis()
                && active.expires_at_millis() == request.expires_at_millis()
                && active.recovery_after_millis() == request.recovery_after_millis()
            {
                return Ok(active);
            }
            if request.acquired_at_millis() < active.recovery_after_millis() {
                return Err(JobExecutionAuthorityError::LeaseUnavailable);
            }
        }
        state.next_generation = state
            .next_generation
            .checked_add(1)
            .ok_or(JobExecutionAuthorityError::Unavailable)?;
        let generation = AggregateVersion::new(state.next_generation)
            .ok_or(JobExecutionAuthorityError::Unavailable)?;
        let lease = JobExecutorLease::new(request, generation)
            .map_err(|_| JobExecutionAuthorityError::Unavailable)?;
        state.active_lease = Some(lease);
        Ok(lease)
    }

    async fn active_executor_lease(
        &self,
        _tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<JobExecutorLease>, JobExecutionAuthorityError> {
        Ok(self
            .state
            .borrow()
            .active_lease
            .filter(|lease| lease.identity() == identity))
    }

    async fn transition(
        &self,
        _tenant: &TenantContext,
        command: JobCommand,
    ) -> Result<JobCommandOutcome, JobExecutionAuthorityError> {
        let mut state = self.state.borrow_mut();
        let outcome = state
            .job
            .apply(command.clone())
            .map_err(|_| JobExecutionAuthorityError::Conflict)?;
        if outcome == JobCommandOutcome::Applied
            && matches!(command.kind(), JobCommandKind::Result { .. })
        {
            state.result_publications += 1;
        }
        Ok(outcome)
    }

    async fn release_executor_lease(
        &self,
        _tenant: &TenantContext,
        lease: JobExecutorLease,
        _released_at_millis: u64,
        reason: JobLeaseReleaseReason,
    ) -> Result<JobLeaseMutationOutcome, JobExecutionAuthorityError> {
        let mut state = self.state.borrow_mut();
        if state.active_lease == Some(lease) {
            state.active_lease = None;
            state.released.push((lease, reason));
            Ok(JobLeaseMutationOutcome::Applied)
        } else if state.released.contains(&(lease, reason)) {
            Ok(JobLeaseMutationOutcome::Duplicate)
        } else {
            Err(JobExecutionAuthorityError::LeaseLost)
        }
    }
}

struct FakeRuntime {
    outcomes: VecDeque<NativeJobRunOutcome>,
    create_count: usize,
    resume_count: usize,
    run_count: usize,
    cancel_count: usize,
    prompts: Vec<acp::PromptRequest>,
}

impl FakeRuntime {
    fn new(outcomes: impl IntoIterator<Item = NativeJobRunOutcome>) -> Self {
        Self {
            outcomes: outcomes.into_iter().collect(),
            create_count: 0,
            resume_count: 0,
            run_count: 0,
            cancel_count: 0,
            prompts: Vec::new(),
        }
    }
}

#[async_trait(?Send)]
impl NativeJobRuntime for FakeRuntime {
    async fn create_session(
        &mut self,
        _identity: agent::collaboration_session::CollaborationSessionIdentity,
    ) -> Result<acp::SessionId, NativeJobRuntimeError> {
        self.create_count += 1;
        Ok(acp::SessionId::new("native-job-session"))
    }

    async fn resume_session(
        &mut self,
        session_id: &acp::SessionId,
    ) -> Result<(), NativeJobRuntimeError> {
        assert_eq!(session_id, &acp::SessionId::new("native-job-session"));
        self.resume_count += 1;
        Ok(())
    }

    async fn run_job(
        &mut self,
        request: acp::PromptRequest,
    ) -> Result<NativeJobRunOutcome, NativeJobRuntimeError> {
        self.run_count += 1;
        self.prompts.push(request);
        self.outcomes
            .pop_front()
            .ok_or(NativeJobRuntimeError::Unavailable)
    }

    async fn cancel_session(
        &mut self,
        session_id: &acp::SessionId,
    ) -> Result<(), NativeJobRuntimeError> {
        assert_eq!(session_id, &acp::SessionId::new("native-job-session"));
        self.cancel_count += 1;
        Ok(())
    }
}

#[test]
fn accepted_job_creates_one_session_and_publishes_exactly_one_result() {
    block_on(async {
        let authority = FakeAuthority::new(accepted_job());
        let mut runtime = FakeRuntime::new([NativeJobRunOutcome::Completed {
            completed_at_millis: 35,
        }]);
        let mut coordinator = JobExecutionCoordinator::default();
        let request = request(200, 30, 40, 50);
        assert_eq!(
            coordinator
                .execute_once(&authority, &mut runtime, &tenant(), &request)
                .await
                .expect("complete job"),
            JobExecutionDisposition::Completed
        );
        assert_eq!(runtime.create_count, 1);
        assert_eq!(runtime.run_count, 1);
        assert_eq!(
            runtime.prompts[0].session_id,
            acp::SessionId::new("native-job-session")
        );
        {
            let state = authority.state.borrow();
            assert!(matches!(state.job.state(), JobState::Completed { .. }));
            assert_eq!(state.result_publications, 1);
            assert_eq!(state.released.len(), 1);
        }

        assert_eq!(
            coordinator
                .execute_once(&authority, &mut runtime, &tenant(), &request)
                .await
                .expect("terminal retry"),
            JobExecutionDisposition::AlreadyTerminal(collaboration_domain::JobStateKind::Completed)
        );
        assert_eq!(runtime.create_count, 1);
        assert_eq!(runtime.run_count, 1);
        assert_eq!(authority.state.borrow().result_publications, 1);
    });
}

#[test]
fn cancellation_stops_the_owned_native_session_and_publishes_cancel() {
    block_on(async {
        let authority = FakeAuthority::new(accepted_job());
        let mut runtime = FakeRuntime::new([NativeJobRunOutcome::CancellationRequested {
            actor_principal_id: principal(REQUESTER),
            cancelled_at_millis: 35,
        }]);
        let mut coordinator = JobExecutionCoordinator::default();
        assert_eq!(
            coordinator
                .execute_once(
                    &authority,
                    &mut runtime,
                    &tenant(),
                    &request(201, 30, 40, 50),
                )
                .await
                .expect("cancel job"),
            JobExecutionDisposition::Cancelled
        );
        assert_eq!(runtime.cancel_count, 1);
        let state = authority.state.borrow();
        assert!(matches!(state.job.state(), JobState::Cancelled { .. }));
        assert_eq!(state.released[0].1, JobLeaseReleaseReason::Cancelled);
    });
}

#[test]
fn crash_keeps_the_lease_until_expiry_then_resumes_the_same_session() {
    block_on(async {
        let authority = FakeAuthority::new(accepted_job());
        let mut runtime = FakeRuntime::new([
            NativeJobRunOutcome::Crashed,
            NativeJobRunOutcome::Completed {
                completed_at_millis: 65,
            },
        ]);
        let mut coordinator = JobExecutionCoordinator::default();
        assert_eq!(
            coordinator
                .execute_once(
                    &authority,
                    &mut runtime,
                    &tenant(),
                    &request(202, 30, 40, 50),
                )
                .await
                .expect("crashed run"),
            JobExecutionDisposition::Crashed
        );
        assert!(matches!(
            coordinator
                .execute_once(
                    &authority,
                    &mut runtime,
                    &tenant(),
                    &request(203, 45, 48, 60),
                )
                .await,
            Err(JobExecutionError::Authority(
                JobExecutionAuthorityError::LeaseUnavailable
            ))
        ));
        assert_eq!(
            coordinator
                .execute_once(
                    &authority,
                    &mut runtime,
                    &tenant(),
                    &request(204, 50, 60, 70),
                )
                .await
                .expect("recovered run"),
            JobExecutionDisposition::Completed
        );
        assert_eq!(runtime.create_count, 1);
        assert_eq!(runtime.resume_count, 1);
        assert_eq!(runtime.run_count, 2);
        assert_eq!(authority.state.borrow().result_publications, 1);
    });
}

#[test]
fn completion_at_the_recovery_boundary_is_fenced_without_a_result() {
    block_on(async {
        let authority = FakeAuthority::new(accepted_job());
        let mut runtime = FakeRuntime::new([NativeJobRunOutcome::Completed {
            completed_at_millis: 50,
        }]);
        let mut coordinator = JobExecutionCoordinator::default();
        assert!(matches!(
            coordinator
                .execute_once(
                    &authority,
                    &mut runtime,
                    &tenant(),
                    &request(205, 30, 40, 50),
                )
                .await,
            Err(JobExecutionError::LeaseExpired)
        ));
        let state = authority.state.borrow();
        assert!(matches!(state.job.state(), JobState::InProgress { .. }));
        assert_eq!(state.result_publications, 0);
        assert!(state.active_lease.is_some());
    });
}
