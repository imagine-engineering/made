use std::collections::BTreeMap;

use collab::jobs::repository::{
    ExecutorLeaseAcquireOutcome, ExecutorLeaseFence, ExecutorLeaseReleaseReason,
    ExecutorLeaseRequest, JobRepository, JobRepositoryError, JobStoreOutcome, LeaseMutationOutcome,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, JobCommand, JobCommandKind, JobError, JobIdentity,
    OperationId, PrincipalId, TenantContext, TrustedTenantRoute,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value};
use uuid::Uuid;

const REQUESTED_AT: i64 = 1_900_000_000_000;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn identity(community_id: CommunityId, value: u128) -> JobIdentity {
    JobIdentity::new(community_id, AggregateId::from_uuid(Uuid::from_u128(value)))
        .expect("valid job identity")
}

fn tenant(community_id: CommunityId) -> TenantContext {
    TenantContext::establish(
        Some(
            TrustedTenantRoute::from_listener(community_id, "job-repository")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn command(
    identity: JobIdentity,
    operation: u128,
    version: u64,
    kind: JobCommandKind,
) -> JobCommand {
    JobCommand::new(
        identity,
        OperationId::from_uuid(Uuid::from_u128(operation)),
        AggregateVersion::new(version).expect("positive version"),
        u64::try_from(REQUESTED_AT).expect("positive timestamp") + version - 1,
        kind,
    )
    .expect("valid job command")
}

fn accept(identity: JobIdentity, operation: u128) -> JobCommand {
    command(
        identity,
        operation,
        2,
        JobCommandKind::Accept {
            executor_principal_id: principal(20),
        },
    )
}

fn request(identity: JobIdentity) -> JobCommand {
    command(
        identity,
        100,
        1,
        JobCommandKind::Request {
            requester_principal_id: principal(10),
            target_executor_principal_id: principal(20),
        },
    )
}

fn head_row(state: &str, version: u64) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("requester_principal_id".into(), Uuid::from_u128(10).into()),
        (
            "target_executor_principal_id".into(),
            Uuid::from_u128(20).into(),
        ),
        (
            "current_executor_principal_id".into(),
            if state == "requested" {
                Option::<Uuid>::None.into()
            } else {
                Some(Uuid::from_u128(20)).into()
            },
        ),
        ("current_version_text".into(), version.to_string().into()),
        ("current_state".into(), state.to_owned().into()),
        ("requested_at_millis".into(), REQUESTED_AT.into()),
        (
            "updated_at_millis".into(),
            (REQUESTED_AT + i64::try_from(version).expect("small version") - 1).into(),
        ),
    ])
}

fn version_row(
    operation: u128,
    version: u64,
    command_type: &str,
    actor: u128,
    executor: Option<u128>,
) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("version_text".into(), version.to_string().into()),
        ("operation_id".into(), Uuid::from_u128(operation).into()),
        ("command_type".into(), command_type.to_owned().into()),
        ("actor_principal_id".into(), Uuid::from_u128(actor).into()),
        (
            "executor_principal_id".into(),
            executor.map(Uuid::from_u128).into(),
        ),
        (
            "occurred_at_millis".into(),
            (REQUESTED_AT + i64::try_from(version).expect("small version") - 1).into(),
        ),
    ])
}

fn request_row() -> BTreeMap<String, Value> {
    version_row(100, 1, "request", 10, None)
}

fn accept_row(operation: u128) -> BTreeMap<String, Value> {
    version_row(operation, 2, "accept", 20, Some(20))
}

fn lease_row(
    job_id: Uuid,
    generation: u64,
    lease_id: Uuid,
    recovery_after: i64,
) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("job_id".into(), job_id.into()),
        ("job_version_text".into(), "2".to_owned().into()),
        (
            "lease_generation_text".into(),
            generation.to_string().into(),
        ),
        ("lease_id".into(), lease_id.into()),
        ("executor_principal_id".into(), Uuid::from_u128(20).into()),
        ("state".into(), "active".to_owned().into()),
        ("acquired_at_millis".into(), REQUESTED_AT.into()),
        ("last_heartbeat_at_millis".into(), REQUESTED_AT.into()),
        ("expires_at_millis".into(), (REQUESTED_AT + 10).into()),
        ("recovery_after_millis".into(), recovery_after.into()),
        ("released_at_millis".into(), Option::<i64>::None.into()),
        ("release_reason".into(), Option::<String>::None.into()),
    ])
}

fn ancestry_row(job_id: AggregateId, depth: i16) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("ancestor_job_id".into(), job_id.as_uuid().into()),
        ("depth".into(), depth.into()),
    ])
}

fn repository(
    query_results: Vec<Vec<BTreeMap<String, Value>>>,
    affected_rows: &[u64],
) -> JobRepository {
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
    JobRepository::new(connection).expect("Postgres job repository")
}

fn log(repository: JobRepository) -> String {
    format!("{:#?}", repository.into_connection().into_transaction_log())
}

#[tokio::test]
async fn job_creation_persists_and_reconstructs_ordered_ancestry() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let child_identity = identity(community_id, 4);
    let root = identity(community_id, 2);
    let parent = identity(community_id, 3);
    let creation_repository = repository(vec![vec![]], &[1, 1, 1, 1, 1]);
    assert_eq!(
        creation_repository
            .create(&tenant, request(child_identity), vec![root, parent])
            .await
            .expect("create child job"),
        JobStoreOutcome::Applied
    );
    let create_log = log(creation_repository);
    assert!(create_log.contains("INSERT INTO public.collaboration_job_ancestry"));

    let repository = repository(
        vec![
            vec![head_row("requested", 1)],
            vec![request_row()],
            vec![
                ancestry_row(root.job_id(), 2),
                ancestry_row(parent.job_id(), 1),
            ],
        ],
        &[1],
    );
    let stored = repository
        .load(&tenant, child_identity)
        .await
        .expect("load child job")
        .expect("stored child job");
    assert_eq!(stored.job().history(), &[request(child_identity)]);
    assert_eq!(stored.ancestry(), &[root, parent]);
}

#[tokio::test]
async fn concurrent_accepts_serialize_and_only_one_transition_wins() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity = identity(community_id, 2);
    let first_accept = accept(identity, 101);
    let first = repository(
        vec![
            vec![],
            vec![head_row("requested", 1)],
            vec![request_row()],
            vec![],
        ],
        &[1, 1, 1],
    );
    assert_eq!(
        first
            .transition(&tenant, first_accept)
            .await
            .expect("first accept"),
        JobStoreOutcome::Applied
    );
    let first_log = log(first);
    assert!(first_log.contains("FOR UPDATE"));
    assert!(first_log.contains("current_version = CAST"));

    let second = repository(
        vec![
            vec![],
            vec![head_row("accepted", 2)],
            vec![request_row(), accept_row(101)],
            vec![],
        ],
        &[1],
    );
    assert!(matches!(
        second.transition(&tenant, accept(identity, 102)).await,
        Err(JobRepositoryError::Domain(JobError::VersionConflict { .. }))
    ));
}

#[tokio::test]
async fn exact_transition_retry_is_idempotent() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity = identity(community_id, 2);
    let operation = BTreeMap::from([
        ("job_id".into(), identity.job_id().as_uuid().into()),
        ("version_text".into(), "2".to_owned().into()),
    ]);
    let repository = repository(
        vec![
            vec![operation],
            vec![head_row("accepted", 2)],
            vec![request_row(), accept_row(101)],
            vec![],
        ],
        &[1],
    );
    assert_eq!(
        repository
            .transition(&tenant, accept(identity, 101))
            .await
            .expect("retry accepted command"),
        JobStoreOutcome::Duplicate
    );
    let transaction_log = log(repository);
    assert!(!transaction_log.contains("UPDATE public.collaboration_jobs\nSET"));
}

#[tokio::test]
async fn expired_executor_lease_is_recovered_with_a_new_generation() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity = identity(community_id, 2);
    let previous_lease_id = Uuid::from_u128(500);
    let next_lease_id = Uuid::from_u128(501);
    let acquisition_time = u64::try_from(REQUESTED_AT + 30).expect("positive timestamp");
    let repository = repository(
        vec![
            vec![head_row("accepted", 2)],
            vec![request_row(), accept_row(101)],
            vec![],
            vec![],
            vec![lease_row(
                identity.job_id().as_uuid(),
                1,
                previous_lease_id,
                REQUESTED_AT + 20,
            )],
            vec![BTreeMap::from([(
                "lease_generation_text".into(),
                "1".to_owned().into(),
            )])],
        ],
        &[1, 1, 1],
    );
    let request = ExecutorLeaseRequest::new(
        identity,
        AggregateVersion::new(2).expect("positive version"),
        next_lease_id,
        principal(20),
        acquisition_time,
        acquisition_time + 10,
        acquisition_time + 20,
    )
    .expect("valid lease request");
    let acquired = repository
        .acquire_executor_lease(&tenant, &request)
        .await
        .expect("recover expired lease");
    assert_eq!(acquired.outcome(), ExecutorLeaseAcquireOutcome::Acquired);
    assert_eq!(acquired.lease().generation().get(), 2);
    let transaction_log = log(repository);
    assert!(transaction_log.contains("release_reason = 'expired'"));
    assert!(transaction_log.contains("INSERT INTO public.collaboration_job_executor_leases"));
}

#[tokio::test]
async fn heartbeat_and_release_require_the_exact_lease_fence() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity = identity(community_id, 2);
    let acquired_at = u64::try_from(REQUESTED_AT + 30).expect("positive timestamp");
    let lease_id = Uuid::from_u128(501);
    let acquisition_repository = repository(
        vec![
            vec![head_row("accepted", 2)],
            vec![request_row(), accept_row(101)],
            vec![],
            vec![],
            vec![],
            vec![BTreeMap::from([(
                "lease_generation_text".into(),
                "0".to_owned().into(),
            )])],
        ],
        &[1, 1],
    );
    let request = ExecutorLeaseRequest::new(
        identity,
        AggregateVersion::new(2).expect("positive version"),
        lease_id,
        principal(20),
        acquired_at,
        acquired_at + 10,
        acquired_at + 20,
    )
    .expect("valid lease request");
    let acquisition = acquisition_repository
        .acquire_executor_lease(&tenant, &request)
        .await
        .expect("acquire lease");
    let fence = ExecutorLeaseFence::from(acquisition.lease());

    let heartbeat_repository = repository(vec![], &[1, 1]);
    assert_eq!(
        heartbeat_repository
            .heartbeat_executor_lease(
                &tenant,
                fence,
                acquired_at + 5,
                acquired_at + 15,
                acquired_at + 25,
            )
            .await
            .expect("heartbeat fenced lease"),
        LeaseMutationOutcome::Applied
    );
    assert!(log(heartbeat_repository).contains("lease_generation = CAST"));

    let release_repository = repository(vec![], &[1, 1]);
    assert_eq!(
        release_repository
            .release_executor_lease(
                &tenant,
                fence,
                acquired_at + 6,
                ExecutorLeaseReleaseReason::Completed,
            )
            .await
            .expect("release fenced lease"),
        LeaseMutationOutcome::Applied
    );
    assert!(log(release_repository).contains("release_reason"));
}

#[tokio::test]
async fn transition_reports_compare_and_swap_conflict() {
    let community_id = community(1);
    let tenant = tenant(community_id);
    let identity = identity(community_id, 2);
    let repository = repository(
        vec![
            vec![],
            vec![head_row("requested", 1)],
            vec![request_row()],
            vec![],
        ],
        &[1, 1, 0],
    );
    assert!(matches!(
        repository.transition(&tenant, accept(identity, 101)).await,
        Err(JobRepositoryError::TransitionConflict)
    ));
}

#[tokio::test]
async fn tenant_identity_mismatch_is_rejected_before_database_access() {
    let tenant = tenant(community(1));
    let other_identity = identity(community(2), 2);
    let repository = repository(vec![], &[]);
    assert!(matches!(
        repository.load(&tenant, other_identity).await,
        Err(JobRepositoryError::TenantBoundaryViolation)
    ));
    assert!(
        repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}
