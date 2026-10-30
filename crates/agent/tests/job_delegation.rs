use std::{cell::RefCell, collections::BTreeMap};

use agent::job_delegation::{
    AuthorizedChildJobRequest, ChildCreationDisposition, DelegatedJobOrchestrator,
    DelegatedJobStoreOutcome, DelegatedStoredJob, JobDelegationAuthority,
    JobDelegationAuthorityError, JobDelegationError, ParentAggregationDisposition,
    TreeCancellationDisposition,
};
use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationScope, CommunityId, Job,
    JobAuthorizationDenial, JobAuthorizationRequest, JobCommand, JobCommandKind, JobCommandOutcome,
    JobDelegationGrant, JobIdentity, JobState, JobTransitionSet, OperationId, PrincipalId,
    PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use futures::executor::block_on;
use uuid::Uuid;

const COMMUNITY: u128 = 1;
const PARENT: u128 = 10;
const REQUESTER: u128 = 20;
const EXECUTOR: u128 = 30;
const DELEGATOR: u128 = 40;

fn community_id() -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(COMMUNITY))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn identity(value: u128) -> JobIdentity {
    JobIdentity::new(
        community_id(),
        AggregateId::from_uuid(Uuid::from_u128(value)),
    )
    .expect("job identity")
}

fn tenant() -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id(), "job-delegation-test")
                .expect("trusted route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn command(
    identity: JobIdentity,
    version: u64,
    occurred_at_millis: u64,
    kind: JobCommandKind,
) -> JobCommand {
    JobCommand::new(
        identity,
        OperationId::from_uuid(Uuid::from_u128(
            identity.job_id().as_uuid().as_u128() + u128::from(version),
        )),
        AggregateVersion::new(version).expect("positive version"),
        occurred_at_millis,
        kind,
    )
    .expect("job command")
}

fn requested_job(identity: JobIdentity) -> Job {
    Job::request(command(
        identity,
        1,
        10,
        JobCommandKind::Request {
            requester_principal_id: principal(REQUESTER),
            target_executor_principal_id: principal(EXECUTOR),
        },
    ))
    .expect("requested job")
}

fn job_in_state(identity: JobIdentity, state: JobState) -> Job {
    let mut job = requested_job(identity);
    if matches!(state, JobState::Requested) {
        return job;
    }
    job.apply(command(
        identity,
        2,
        20,
        JobCommandKind::Accept {
            executor_principal_id: principal(EXECUTOR),
        },
    ))
    .expect("accept job");
    match state {
        JobState::Accepted { .. } => {}
        JobState::InProgress { .. } => {
            job.apply(command(
                identity,
                3,
                30,
                JobCommandKind::Progress {
                    executor_principal_id: principal(EXECUTOR),
                },
            ))
            .expect("progress job");
        }
        JobState::Completed { .. } => {
            job.apply(command(
                identity,
                3,
                30,
                JobCommandKind::Result {
                    executor_principal_id: principal(EXECUTOR),
                },
            ))
            .expect("complete job");
        }
        JobState::Cancelled { .. } => {
            job.apply(command(
                identity,
                3,
                30,
                JobCommandKind::Cancel {
                    actor_principal_id: principal(REQUESTER),
                },
            ))
            .expect("cancel job");
        }
        JobState::Failed { .. } => {
            job.apply(command(
                identity,
                3,
                30,
                JobCommandKind::Error {
                    actor_principal_id: principal(EXECUTOR),
                },
            ))
            .expect("fail job");
        }
        JobState::Requested => unreachable!("requested returned above"),
    }
    job
}

fn authorize_child(
    child_identity: JobIdentity,
    ancestry: &[JobIdentity],
) -> Result<AuthorizedChildJobRequest, JobDelegationError> {
    let tenant = tenant();
    let command = command(
        child_identity,
        1,
        40,
        JobCommandKind::Request {
            requester_principal_id: principal(REQUESTER),
            target_executor_principal_id: principal(EXECUTOR),
        },
    );
    let child = Job::request(command.clone()).expect("child request");
    let scope = AuthorizationScope::new("jobs:write").expect("scope");
    let authenticated_principal = AuthenticatedPrincipal::zed_account(
        principal(REQUESTER),
        community_id(),
        ServiceAccountId::new(1),
        PrincipalScopes::new([scope]).expect("scopes"),
    );
    let delegation = JobDelegationGrant::new(
        community_id(),
        identity(PARENT).job_id(),
        child_identity.job_id(),
        principal(DELEGATOR),
        principal(REQUESTER),
        AggregateVersion::FIRST,
        JobTransitionSet::new([collaboration_domain::JobCommandType::Request]),
        [],
        1_000,
    )
    .expect("delegation grant");
    AuthorizedChildJobRequest::authorize(&JobAuthorizationRequest {
        tenant: &tenant,
        principal: &authenticated_principal,
        job: &child,
        command: &command,
        current_membership_version: AggregateVersion::FIRST,
        community_membership: None,
        team_id: None,
        current_team_version: None,
        team_membership: None,
        service_grant: None,
        delegation: Some(&delegation),
        ancestry,
        authorized_resource_ids: &[],
        job_resource_ids: &[],
        active_child_jobs: 0,
        active_community_jobs: 1,
        now_millis: 100,
    })
}

struct FakeAuthority {
    jobs: RefCell<BTreeMap<JobIdentity, DelegatedStoredJob>>,
    applied_creates: RefCell<usize>,
}

impl FakeAuthority {
    fn new(records: impl IntoIterator<Item = DelegatedStoredJob>) -> Self {
        Self {
            jobs: RefCell::new(
                records
                    .into_iter()
                    .map(|record| (record.job().identity(), record))
                    .collect(),
            ),
            applied_creates: RefCell::new(0),
        }
    }

    fn record(
        identity: JobIdentity,
        state: JobState,
        ancestry: Vec<JobIdentity>,
    ) -> DelegatedStoredJob {
        DelegatedStoredJob::new(job_in_state(identity, state), ancestry).expect("stored job")
    }
}

#[async_trait(?Send)]
impl JobDelegationAuthority for FakeAuthority {
    async fn load_job(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<DelegatedStoredJob>, JobDelegationAuthorityError> {
        if tenant.community_id() != identity.community_id() {
            return Err(JobDelegationAuthorityError::Unavailable);
        }
        Ok(self.jobs.borrow().get(&identity).cloned())
    }

    async fn create_child(
        &self,
        _tenant: &TenantContext,
        command: JobCommand,
        ancestry: Vec<JobIdentity>,
    ) -> Result<DelegatedJobStoreOutcome, JobDelegationAuthorityError> {
        let mut jobs = self.jobs.borrow_mut();
        if let Some(existing) = jobs.get(&command.identity()) {
            return if existing.job().history() == std::slice::from_ref(&command)
                && existing.ancestry() == ancestry
            {
                Ok(DelegatedJobStoreOutcome::Duplicate)
            } else {
                Err(JobDelegationAuthorityError::Conflict)
            };
        }
        let job = Job::request(command).map_err(|_| JobDelegationAuthorityError::Conflict)?;
        let record = DelegatedStoredJob::new(job, ancestry)
            .map_err(|_| JobDelegationAuthorityError::Conflict)?;
        jobs.insert(record.job().identity(), record);
        *self.applied_creates.borrow_mut() += 1;
        Ok(DelegatedJobStoreOutcome::Applied)
    }

    async fn transition(
        &self,
        _tenant: &TenantContext,
        command: JobCommand,
    ) -> Result<JobCommandOutcome, JobDelegationAuthorityError> {
        let mut jobs = self.jobs.borrow_mut();
        let record = jobs
            .get_mut(&command.identity())
            .ok_or(JobDelegationAuthorityError::Conflict)?;
        record
            .apply(command)
            .map_err(|_| JobDelegationAuthorityError::Conflict)
    }
}

#[test]
fn delegated_tree_completes_only_after_every_child_completes() {
    block_on(async {
        let parent = identity(PARENT);
        let first_child = identity(11);
        let second_child = identity(12);
        let authority = FakeAuthority::new([
            FakeAuthority::record(
                parent,
                JobState::InProgress {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![],
            ),
            FakeAuthority::record(
                first_child,
                JobState::Completed {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![parent],
            ),
            FakeAuthority::record(
                second_child,
                JobState::InProgress {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![parent],
            ),
        ]);
        let orchestrator = DelegatedJobOrchestrator;
        assert_eq!(
            orchestrator
                .aggregate_parent(
                    &authority,
                    &tenant(),
                    parent,
                    &[first_child, second_child],
                    principal(EXECUTOR),
                    100,
                )
                .await
                .expect("pending tree"),
            ParentAggregationDisposition::Pending {
                remaining_children: 1
            }
        );
        authority
            .transition(
                &tenant(),
                command(
                    second_child,
                    4,
                    90,
                    JobCommandKind::Result {
                        executor_principal_id: principal(EXECUTOR),
                    },
                ),
            )
            .await
            .expect("complete second child");
        assert_eq!(
            orchestrator
                .aggregate_parent(
                    &authority,
                    &tenant(),
                    parent,
                    &[first_child, second_child],
                    principal(EXECUTOR),
                    100,
                )
                .await
                .expect("complete tree"),
            ParentAggregationDisposition::Completed
        );
        assert!(matches!(
            authority.jobs.borrow()[&parent].job().state(),
            JobState::Completed { .. }
        ));
    });
}

#[test]
fn partial_child_failure_fails_the_parent() {
    block_on(async {
        let parent = identity(PARENT);
        let authority = FakeAuthority::new([
            FakeAuthority::record(
                parent,
                JobState::Accepted {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![],
            ),
            FakeAuthority::record(
                identity(11),
                JobState::Completed {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![parent],
            ),
            FakeAuthority::record(
                identity(12),
                JobState::Failed {
                    executor_principal_id: Some(principal(EXECUTOR)),
                    reported_by_principal_id: principal(EXECUTOR),
                },
                vec![parent],
            ),
        ]);
        assert_eq!(
            DelegatedJobOrchestrator
                .aggregate_parent(
                    &authority,
                    &tenant(),
                    parent,
                    &[identity(11), identity(12)],
                    principal(EXECUTOR),
                    100,
                )
                .await
                .expect("aggregate failed child"),
            ParentAggregationDisposition::Failed
        );
        assert!(matches!(
            authority.jobs.borrow()[&parent].job().state(),
            JobState::Failed { .. }
        ));
    });
}

#[test]
fn parent_cancel_is_idempotently_propagated_to_nonterminal_children() {
    block_on(async {
        let parent = identity(PARENT);
        let first_child = identity(11);
        let second_child = identity(12);
        let authority = FakeAuthority::new([
            FakeAuthority::record(
                parent,
                JobState::InProgress {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![],
            ),
            FakeAuthority::record(
                first_child,
                JobState::InProgress {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![parent],
            ),
            FakeAuthority::record(
                second_child,
                JobState::Completed {
                    executor_principal_id: principal(EXECUTOR),
                },
                vec![parent],
            ),
        ]);
        let orchestrator = DelegatedJobOrchestrator;
        assert_eq!(
            orchestrator
                .cancel_tree(
                    &authority,
                    &tenant(),
                    parent,
                    &[first_child, second_child],
                    principal(REQUESTER),
                    100,
                )
                .await
                .expect("cancel tree"),
            TreeCancellationDisposition {
                cancelled_children: 1,
                already_terminal_children: 1,
                parent_cancelled: true,
            }
        );
        assert_eq!(
            orchestrator
                .cancel_tree(
                    &authority,
                    &tenant(),
                    parent,
                    &[first_child, second_child],
                    principal(REQUESTER),
                    100,
                )
                .await
                .expect("retry cancel tree"),
            TreeCancellationDisposition {
                cancelled_children: 0,
                already_terminal_children: 2,
                parent_cancelled: false,
            }
        );
    });
}

#[test]
fn authorized_child_creation_is_retry_safe_and_cycles_are_rejected() {
    block_on(async {
        let parent = identity(PARENT);
        let child = identity(11);
        let authority = FakeAuthority::new([FakeAuthority::record(
            parent,
            JobState::InProgress {
                executor_principal_id: principal(EXECUTOR),
            },
            vec![],
        )]);
        let request = authorize_child(child, &[parent]).expect("authorized child");
        let orchestrator = DelegatedJobOrchestrator;
        assert_eq!(
            orchestrator
                .create_child(&authority, &tenant(), &request)
                .await
                .expect("create child"),
            ChildCreationDisposition::Created
        );
        assert_eq!(
            orchestrator
                .create_child(&authority, &tenant(), &request)
                .await
                .expect("retry child"),
            ChildCreationDisposition::Existing
        );
        assert_eq!(*authority.applied_creates.borrow(), 1);
        assert_eq!(authority.jobs.borrow()[&child].ancestry(), &[parent]);

        let mismatched_ancestry =
            authorize_child(identity(12), &[identity(99), parent]).expect("authorized chain");
        assert!(matches!(
            orchestrator
                .create_child(&authority, &tenant(), &mismatched_ancestry)
                .await,
            Err(JobDelegationError::InvalidTree)
        ));
        assert!(matches!(
            authorize_child(identity(13), &[parent, identity(13)]),
            Err(JobDelegationError::Unauthorized(
                JobAuthorizationDenial::DelegationCycle
            ))
        ));
    });
}
