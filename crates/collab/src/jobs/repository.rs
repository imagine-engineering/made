use std::collections::BTreeSet;

use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, Job, JobCommand, JobCommandKind, JobCommandOutcome,
    JobCommandType, JobError, JobIdentity, JobState, OperationId, PrincipalId, TenantContext,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use uuid::Uuid;

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const SELECT_HEAD_SQL: &str = r#"
SELECT
    requester_principal_id,
    target_executor_principal_id,
    current_executor_principal_id,
    current_version::text AS current_version_text,
    current_state,
    floor(extract(epoch FROM requested_at) * 1000)::bigint AS requested_at_millis,
    floor(extract(epoch FROM updated_at) * 1000)::bigint AS updated_at_millis
FROM public.collaboration_jobs
WHERE community_id = $1 AND job_id = $2
"#;
const SELECT_VERSIONS_SQL: &str = r#"
SELECT
    version::text AS version_text,
    operation_id,
    command_type,
    actor_principal_id,
    executor_principal_id,
    floor(extract(epoch FROM occurred_at) * 1000)::bigint AS occurred_at_millis
FROM public.collaboration_job_versions
WHERE community_id = $1 AND job_id = $2
ORDER BY version
"#;
const SELECT_ANCESTRY_SQL: &str = r#"
SELECT ancestor_job_id, depth
FROM public.collaboration_job_ancestry
WHERE community_id = $1 AND descendant_job_id = $2
ORDER BY depth DESC
"#;
const SELECT_OPERATION_SQL: &str = r#"
SELECT job_id, version::text AS version_text
FROM public.collaboration_job_versions
WHERE community_id = $1 AND operation_id = $2
FOR UPDATE
"#;
const INSERT_HEAD_SQL: &str = r#"
INSERT INTO public.collaboration_jobs (
    community_id,
    job_id,
    requester_principal_id,
    target_executor_principal_id,
    current_executor_principal_id,
    current_version,
    current_state,
    requested_at,
    updated_at
) VALUES (
    $1, $2, $3, $4, NULL, CAST($5 AS numeric), 'requested',
    to_timestamp($6::double precision / 1000),
    to_timestamp($6::double precision / 1000)
)
ON CONFLICT (community_id, job_id) DO NOTHING
"#;
const INSERT_VERSION_SQL: &str = r#"
INSERT INTO public.collaboration_job_versions (
    community_id,
    job_id,
    version,
    operation_id,
    command_type,
    actor_principal_id,
    executor_principal_id,
    occurred_at
) VALUES (
    $1, $2, CAST($3 AS numeric), $4, $5, $6, $7,
    to_timestamp($8::double precision / 1000)
)
"#;
const UPDATE_HEAD_SQL: &str = r#"
UPDATE public.collaboration_jobs
SET
    current_executor_principal_id = $3,
    current_version = CAST($4 AS numeric),
    current_state = $5,
    updated_at = to_timestamp($6::double precision / 1000)
WHERE community_id = $1
  AND job_id = $2
  AND current_version = CAST($7 AS numeric)
"#;
const INSERT_ANCESTRY_SQL: &str = r#"
INSERT INTO public.collaboration_job_ancestry (
    community_id, ancestor_job_id, descendant_job_id, depth, created_at
) VALUES (
    $1, $2, $3, $4, to_timestamp($5::double precision / 1000)
)
"#;
const SELECT_LEASE_BY_ID_SQL: &str = r#"
SELECT
    job_id,
    job_version::text AS job_version_text,
    lease_generation::text AS lease_generation_text,
    lease_id,
    executor_principal_id,
    state,
    floor(extract(epoch FROM acquired_at) * 1000)::bigint AS acquired_at_millis,
    floor(extract(epoch FROM last_heartbeat_at) * 1000)::bigint AS last_heartbeat_at_millis,
    floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
    floor(extract(epoch FROM recovery_after) * 1000)::bigint AS recovery_after_millis,
    floor(extract(epoch FROM released_at) * 1000)::bigint AS released_at_millis,
    release_reason
FROM public.collaboration_job_executor_leases
WHERE community_id = $1 AND lease_id = $2
FOR UPDATE
"#;
const SELECT_ACTIVE_LEASE_SQL: &str = r#"
SELECT
    job_id,
    job_version::text AS job_version_text,
    lease_generation::text AS lease_generation_text,
    lease_id,
    executor_principal_id,
    state,
    floor(extract(epoch FROM acquired_at) * 1000)::bigint AS acquired_at_millis,
    floor(extract(epoch FROM last_heartbeat_at) * 1000)::bigint AS last_heartbeat_at_millis,
    floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
    floor(extract(epoch FROM recovery_after) * 1000)::bigint AS recovery_after_millis,
    floor(extract(epoch FROM released_at) * 1000)::bigint AS released_at_millis,
    release_reason
FROM public.collaboration_job_executor_leases
WHERE community_id = $1 AND job_id = $2 AND state = 'active'
FOR UPDATE
"#;
const SELECT_MAX_LEASE_GENERATION_SQL: &str = r#"
SELECT COALESCE(max(lease_generation), 0)::text AS lease_generation_text
FROM public.collaboration_job_executor_leases
WHERE community_id = $1 AND job_id = $2
"#;
const EXPIRE_LEASE_SQL: &str = r#"
UPDATE public.collaboration_job_executor_leases
SET state = 'released', released_at = to_timestamp($4::double precision / 1000), release_reason = 'expired'
WHERE community_id = $1 AND job_id = $2 AND lease_generation = CAST($3 AS numeric) AND state = 'active'
"#;
const INSERT_LEASE_SQL: &str = r#"
INSERT INTO public.collaboration_job_executor_leases (
    community_id,
    job_id,
    job_version,
    lease_generation,
    lease_id,
    executor_principal_id,
    state,
    acquired_at,
    last_heartbeat_at,
    expires_at,
    recovery_after
) VALUES (
    $1, $2, CAST($3 AS numeric), CAST($4 AS numeric), $5, $6, 'active',
    to_timestamp($7::double precision / 1000),
    to_timestamp($7::double precision / 1000),
    to_timestamp($8::double precision / 1000),
    to_timestamp($9::double precision / 1000)
)
"#;
const HEARTBEAT_LEASE_SQL: &str = r#"
UPDATE public.collaboration_job_executor_leases
SET
    last_heartbeat_at = to_timestamp($7::double precision / 1000),
    expires_at = to_timestamp($8::double precision / 1000),
    recovery_after = to_timestamp($9::double precision / 1000)
WHERE community_id = $1
  AND job_id = $2
  AND lease_generation = CAST($3 AS numeric)
  AND lease_id = $4
  AND executor_principal_id = $5
  AND job_version = CAST($6 AS numeric)
  AND state = 'active'
  AND expires_at >= to_timestamp($7::double precision / 1000)
"#;
const RELEASE_LEASE_SQL: &str = r#"
UPDATE public.collaboration_job_executor_leases
SET state = 'released', released_at = to_timestamp($7::double precision / 1000), release_reason = $8
WHERE community_id = $1
  AND job_id = $2
  AND lease_generation = CAST($3 AS numeric)
  AND lease_id = $4
  AND executor_principal_id = $5
  AND job_version = CAST($6 AS numeric)
  AND state = 'active'
"#;

pub const MAX_JOB_ANCESTRY_DEPTH: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobStoreOutcome {
    Applied,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredJob {
    job: Job,
    ancestry: Vec<JobIdentity>,
}

impl StoredJob {
    pub const fn job(&self) -> &Job {
        &self.job
    }

    pub fn ancestry(&self) -> &[JobIdentity] {
        &self.ancestry
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorLeaseState {
    Active,
    Released,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorLeaseReleaseReason {
    Completed,
    Cancelled,
    Failed,
    Expired,
    Replaced,
}

impl ExecutorLeaseReleaseReason {
    const fn database_name(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Replaced => "replaced",
        }
    }

    fn from_database(value: &str) -> Result<Self, JobRepositoryError> {
        match value {
            "completed" => Ok(Self::Completed),
            "cancelled" => Ok(Self::Cancelled),
            "failed" => Ok(Self::Failed),
            "expired" => Ok(Self::Expired),
            "replaced" => Ok(Self::Replaced),
            _ => Err(JobRepositoryError::InvalidRecord),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorLease {
    identity: JobIdentity,
    job_version: AggregateVersion,
    generation: AggregateVersion,
    lease_id: Uuid,
    executor_principal_id: PrincipalId,
    state: ExecutorLeaseState,
    acquired_at_millis: u64,
    last_heartbeat_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
    released_at_millis: Option<u64>,
    release_reason: Option<ExecutorLeaseReleaseReason>,
}

impl ExecutorLease {
    pub const fn identity(&self) -> JobIdentity {
        self.identity
    }

    pub const fn job_version(&self) -> AggregateVersion {
        self.job_version
    }

    pub const fn generation(&self) -> AggregateVersion {
        self.generation
    }

    pub const fn lease_id(&self) -> Uuid {
        self.lease_id
    }

    pub const fn executor_principal_id(&self) -> PrincipalId {
        self.executor_principal_id
    }

    pub const fn state(&self) -> ExecutorLeaseState {
        self.state
    }

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }

    pub const fn recovery_after_millis(&self) -> u64 {
        self.recovery_after_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutorLeaseAcquireOutcome {
    Acquired,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorLeaseAcquisition {
    outcome: ExecutorLeaseAcquireOutcome,
    lease: ExecutorLease,
}

impl ExecutorLeaseAcquisition {
    pub const fn outcome(&self) -> ExecutorLeaseAcquireOutcome {
        self.outcome
    }

    pub const fn lease(&self) -> &ExecutorLease {
        &self.lease
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutorLeaseRequest {
    identity: JobIdentity,
    job_version: AggregateVersion,
    lease_id: Uuid,
    executor_principal_id: PrincipalId,
    acquired_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
}

impl ExecutorLeaseRequest {
    pub fn new(
        identity: JobIdentity,
        job_version: AggregateVersion,
        lease_id: Uuid,
        executor_principal_id: PrincipalId,
        acquired_at_millis: u64,
        expires_at_millis: u64,
        recovery_after_millis: u64,
    ) -> Result<Self, JobRepositoryError> {
        if lease_id.is_nil()
            || executor_principal_id.as_uuid().is_nil()
            || acquired_at_millis > expires_at_millis
            || expires_at_millis > recovery_after_millis
        {
            return Err(JobRepositoryError::InvalidLease);
        }
        Ok(Self {
            identity,
            job_version,
            lease_id,
            executor_principal_id,
            acquired_at_millis,
            expires_at_millis,
            recovery_after_millis,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutorLeaseFence {
    identity: JobIdentity,
    job_version: AggregateVersion,
    generation: AggregateVersion,
    lease_id: Uuid,
    executor_principal_id: PrincipalId,
}

impl From<&ExecutorLease> for ExecutorLeaseFence {
    fn from(lease: &ExecutorLease) -> Self {
        Self {
            identity: lease.identity,
            job_version: lease.job_version,
            generation: lease.generation,
            lease_id: lease.lease_id,
            executor_principal_id: lease.executor_principal_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseMutationOutcome {
    Applied,
    Duplicate,
}

#[derive(Debug, thiserror::Error)]
pub enum JobRepositoryError {
    #[error("job repository requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("job repository request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("job does not exist")]
    NotFound,
    #[error("job transition lost optimistic concurrency")]
    TransitionConflict,
    #[error("job ancestry is invalid")]
    InvalidAncestry,
    #[error("job executor lease is invalid")]
    InvalidLease,
    #[error("job executor lease is not recoverable yet")]
    LeaseUnavailable,
    #[error("job executor lease generation is no longer authoritative")]
    LeaseFenceLost,
    #[error("job repository record is invalid")]
    InvalidRecord,
    #[error(transparent)]
    Domain(#[from] JobError),
    #[error("job repository is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct JobRepository {
    connection: DatabaseConnection,
}

impl JobRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, JobRepositoryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(JobRepositoryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn create(
        &self,
        tenant: &TenantContext,
        command: JobCommand,
        ancestry: Vec<JobIdentity>,
    ) -> Result<JobStoreOutcome, JobRepositoryError> {
        validate_identity(tenant, command.identity())?;
        validate_ancestry(command.identity(), &ancestry)?;
        let requested_job = Job::request(command.clone())?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            if let Some(row) = transaction
                .query_one(operation_statement(
                    tenant.community_id(),
                    command.operation_id(),
                ))
                .await
                .map_err(JobRepositoryError::Unavailable)?
            {
                let job_id: Uuid = row_value(&row, "job_id")?;
                if job_id != command.identity().job_id().as_uuid() {
                    return Err(JobError::IdempotencyConflict.into());
                }
                let stored =
                    load_stored_job(&transaction, command.identity(), "FOR UPDATE").await?;
                return duplicate_create(stored, command, &ancestry);
            }

            let inserted = transaction
                .execute(insert_head_statement(&requested_job)?)
                .await
                .map_err(JobRepositoryError::Unavailable)?;
            if inserted.rows_affected() == 0 {
                let stored =
                    load_stored_job(&transaction, command.identity(), "FOR UPDATE").await?;
                return duplicate_create(stored, command, &ancestry);
            }

            insert_version(&transaction, &command).await?;
            for (index, ancestor) in ancestry.iter().enumerate() {
                let depth = i16::try_from(ancestry.len() - index)
                    .map_err(|_| JobRepositoryError::InvalidAncestry)?;
                transaction
                    .execute(Statement::from_sql_and_values(
                        DatabaseBackend::Postgres,
                        INSERT_ANCESTRY_SQL,
                        [
                            tenant.community_id().as_uuid().into(),
                            ancestor.job_id().as_uuid().into(),
                            command.identity().job_id().as_uuid().into(),
                            depth.into(),
                            millis_value(command.occurred_at_millis())?,
                        ],
                    ))
                    .await
                    .map_err(JobRepositoryError::Unavailable)?;
            }
            Ok(JobStoreOutcome::Applied)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn transition(
        &self,
        tenant: &TenantContext,
        command: JobCommand,
    ) -> Result<JobStoreOutcome, JobRepositoryError> {
        validate_identity(tenant, command.identity())?;
        if command.kind().command_type() == JobCommandType::Request {
            return Err(JobRepositoryError::TransitionConflict);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            if let Some(row) = transaction
                .query_one(operation_statement(
                    tenant.community_id(),
                    command.operation_id(),
                ))
                .await
                .map_err(JobRepositoryError::Unavailable)?
            {
                let job_id: Uuid = row_value(&row, "job_id")?;
                if job_id != command.identity().job_id().as_uuid() {
                    return Err(JobError::IdempotencyConflict.into());
                }
            }
            let stored = load_stored_job(&transaction, command.identity(), "FOR UPDATE").await?;
            let mut job = stored.job;
            let previous_version = job.version();
            match job.apply(command.clone())? {
                JobCommandOutcome::Unchanged => Ok(JobStoreOutcome::Duplicate),
                JobCommandOutcome::Applied => {
                    insert_version(&transaction, &command).await?;
                    let updated = transaction
                        .execute(update_head_statement(&job, previous_version)?)
                        .await
                        .map_err(JobRepositoryError::Unavailable)?;
                    if updated.rows_affected() != 1 {
                        return Err(JobRepositoryError::TransitionConflict);
                    }
                    Ok(JobStoreOutcome::Applied)
                }
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn load(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<StoredJob>, JobRepositoryError> {
        validate_identity(tenant, identity)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            load_optional_stored_job(&transaction, identity, "FOR SHARE").await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn acquire_executor_lease(
        &self,
        tenant: &TenantContext,
        request: &ExecutorLeaseRequest,
    ) -> Result<ExecutorLeaseAcquisition, JobRepositoryError> {
        validate_identity(tenant, request.identity)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let stored = load_stored_job(&transaction, request.identity, "FOR UPDATE").await?;
            if stored.job.version() != request.job_version
                || stored.job.state().executor_principal_id() != Some(request.executor_principal_id)
                || !matches!(
                    stored.job.state(),
                    JobState::Accepted { .. } | JobState::InProgress { .. }
                )
            {
                return Err(JobRepositoryError::InvalidLease);
            }

            if let Some(existing) =
                query_lease_by_id(&transaction, tenant.community_id(), request.lease_id).await?
            {
                if lease_matches_request(&existing, request)
                    && existing.state == ExecutorLeaseState::Active
                {
                    return Ok(ExecutorLeaseAcquisition {
                        outcome: ExecutorLeaseAcquireOutcome::Duplicate,
                        lease: existing,
                    });
                }
                return Err(JobRepositoryError::LeaseFenceLost);
            }

            if let Some(active) = query_active_lease(
                &transaction,
                tenant.community_id(),
                request.identity.job_id(),
            )
            .await?
            {
                if active.recovery_after_millis > request.acquired_at_millis {
                    return Err(JobRepositoryError::LeaseUnavailable);
                }
                let expired = transaction
                    .execute(Statement::from_sql_and_values(
                        DatabaseBackend::Postgres,
                        EXPIRE_LEASE_SQL,
                        [
                            tenant.community_id().as_uuid().into(),
                            request.identity.job_id().as_uuid().into(),
                            active.generation.to_string().into(),
                            millis_value(request.acquired_at_millis)?,
                        ],
                    ))
                    .await
                    .map_err(JobRepositoryError::Unavailable)?;
                if expired.rows_affected() != 1 {
                    return Err(JobRepositoryError::LeaseFenceLost);
                }
            }

            let maximum_generation = query_maximum_generation(
                &transaction,
                tenant.community_id(),
                request.identity.job_id(),
            )
            .await?;
            let generation = maximum_generation
                .map_or(Some(AggregateVersion::FIRST), AggregateVersion::next)
                .ok_or(JobRepositoryError::InvalidLease)?;
            transaction
                .execute(insert_lease_statement(request, generation)?)
                .await
                .map_err(JobRepositoryError::Unavailable)?;
            let lease = ExecutorLease {
                identity: request.identity,
                job_version: request.job_version,
                generation,
                lease_id: request.lease_id,
                executor_principal_id: request.executor_principal_id,
                state: ExecutorLeaseState::Active,
                acquired_at_millis: request.acquired_at_millis,
                last_heartbeat_at_millis: request.acquired_at_millis,
                expires_at_millis: request.expires_at_millis,
                recovery_after_millis: request.recovery_after_millis,
                released_at_millis: None,
                release_reason: None,
            };
            Ok(ExecutorLeaseAcquisition {
                outcome: ExecutorLeaseAcquireOutcome::Acquired,
                lease,
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn heartbeat_executor_lease(
        &self,
        tenant: &TenantContext,
        fence: ExecutorLeaseFence,
        heartbeat_at_millis: u64,
        expires_at_millis: u64,
        recovery_after_millis: u64,
    ) -> Result<LeaseMutationOutcome, JobRepositoryError> {
        validate_identity(tenant, fence.identity)?;
        if heartbeat_at_millis > expires_at_millis || expires_at_millis > recovery_after_millis {
            return Err(JobRepositoryError::InvalidLease);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let updated = transaction
                .execute(heartbeat_statement(
                    tenant.community_id(),
                    fence,
                    heartbeat_at_millis,
                    expires_at_millis,
                    recovery_after_millis,
                )?)
                .await
                .map_err(JobRepositoryError::Unavailable)?;
            if updated.rows_affected() == 1 {
                return Ok(LeaseMutationOutcome::Applied);
            }
            let existing = query_lease_by_id(&transaction, tenant.community_id(), fence.lease_id)
                .await?
                .ok_or(JobRepositoryError::LeaseFenceLost)?;
            if lease_matches_fence(&existing, fence)
                && existing.state == ExecutorLeaseState::Active
                && existing.last_heartbeat_at_millis == heartbeat_at_millis
                && existing.expires_at_millis == expires_at_millis
                && existing.recovery_after_millis == recovery_after_millis
            {
                Ok(LeaseMutationOutcome::Duplicate)
            } else {
                Err(JobRepositoryError::LeaseFenceLost)
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn release_executor_lease(
        &self,
        tenant: &TenantContext,
        fence: ExecutorLeaseFence,
        released_at_millis: u64,
        reason: ExecutorLeaseReleaseReason,
    ) -> Result<LeaseMutationOutcome, JobRepositoryError> {
        validate_identity(tenant, fence.identity)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let updated = transaction
                .execute(release_statement(
                    tenant.community_id(),
                    fence,
                    released_at_millis,
                    reason,
                )?)
                .await
                .map_err(JobRepositoryError::Unavailable)?;
            if updated.rows_affected() == 1 {
                return Ok(LeaseMutationOutcome::Applied);
            }
            let existing = query_lease_by_id(&transaction, tenant.community_id(), fence.lease_id)
                .await?
                .ok_or(JobRepositoryError::LeaseFenceLost)?;
            if lease_matches_fence(&existing, fence)
                && existing.state == ExecutorLeaseState::Released
                && existing.released_at_millis == Some(released_at_millis)
                && existing.release_reason == Some(reason)
            {
                Ok(LeaseMutationOutcome::Duplicate)
            } else {
                Err(JobRepositoryError::LeaseFenceLost)
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn active_executor_lease(
        &self,
        tenant: &TenantContext,
        identity: JobIdentity,
    ) -> Result<Option<ExecutorLease>, JobRepositoryError> {
        validate_identity(tenant, identity)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            query_active_lease(&transaction, tenant.community_id(), identity.job_id()).await
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, JobRepositoryError> {
        self.connection
            .begin()
            .await
            .map_err(JobRepositoryError::Unavailable)
    }
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, JobRepositoryError>,
) -> Result<T, JobRepositoryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(JobRepositoryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(JobRepositoryError::Unavailable)?;
            Err(error)
        }
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), JobRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?;
    Ok(())
}

fn validate_identity(
    tenant: &TenantContext,
    identity: JobIdentity,
) -> Result<(), JobRepositoryError> {
    if tenant.community_id() != identity.community_id() {
        return Err(JobRepositoryError::TenantBoundaryViolation);
    }
    Ok(())
}

fn validate_ancestry(
    identity: JobIdentity,
    ancestry: &[JobIdentity],
) -> Result<(), JobRepositoryError> {
    if ancestry.len() > MAX_JOB_ANCESTRY_DEPTH
        || ancestry.iter().any(|ancestor| {
            ancestor.community_id() != identity.community_id() || ancestor == &identity
        })
        || ancestry.iter().copied().collect::<BTreeSet<_>>().len() != ancestry.len()
    {
        return Err(JobRepositoryError::InvalidAncestry);
    }
    Ok(())
}

fn duplicate_create(
    mut stored: StoredJob,
    command: JobCommand,
    ancestry: &[JobIdentity],
) -> Result<JobStoreOutcome, JobRepositoryError> {
    if stored.job.apply(command)? != JobCommandOutcome::Unchanged || stored.ancestry != ancestry {
        return Err(JobRepositoryError::InvalidAncestry);
    }
    Ok(JobStoreOutcome::Duplicate)
}

async fn load_stored_job(
    transaction: &DatabaseTransaction,
    identity: JobIdentity,
    lock_clause: &str,
) -> Result<StoredJob, JobRepositoryError> {
    load_optional_stored_job(transaction, identity, lock_clause)
        .await?
        .ok_or(JobRepositoryError::NotFound)
}

async fn load_optional_stored_job(
    transaction: &DatabaseTransaction,
    identity: JobIdentity,
    lock_clause: &str,
) -> Result<Option<StoredJob>, JobRepositoryError> {
    let head = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            format!("{SELECT_HEAD_SQL} {lock_clause}"),
            [
                identity.community_id().as_uuid().into(),
                identity.job_id().as_uuid().into(),
            ],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?;
    let Some(head) = head else {
        return Ok(None);
    };
    let requester_principal_id =
        PrincipalId::from_uuid(row_value(&head, "requester_principal_id")?);
    let target_executor_principal_id =
        PrincipalId::from_uuid(row_value(&head, "target_executor_principal_id")?);
    let version_rows = transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_VERSIONS_SQL,
            [
                identity.community_id().as_uuid().into(),
                identity.job_id().as_uuid().into(),
            ],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?;
    let history = version_rows
        .into_iter()
        .map(|row| {
            command_from_row(
                row,
                identity,
                requester_principal_id,
                target_executor_principal_id,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let job = Job::from_history(history)?;
    validate_head(&head, &job)?;
    let ancestry = load_ancestry(transaction, identity).await?;
    Ok(Some(StoredJob { job, ancestry }))
}

async fn load_ancestry(
    transaction: &DatabaseTransaction,
    identity: JobIdentity,
) -> Result<Vec<JobIdentity>, JobRepositoryError> {
    transaction
        .query_all(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_ANCESTRY_SQL,
            [
                identity.community_id().as_uuid().into(),
                identity.job_id().as_uuid().into(),
            ],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?
        .into_iter()
        .map(|row| {
            let _: i16 = row_value(&row, "depth")?;
            JobIdentity::new(
                identity.community_id(),
                AggregateId::from_uuid(row_value(&row, "ancestor_job_id")?),
            )
            .map_err(JobRepositoryError::Domain)
        })
        .collect()
}

fn command_from_row(
    row: QueryResult,
    identity: JobIdentity,
    requester_principal_id: PrincipalId,
    target_executor_principal_id: PrincipalId,
) -> Result<JobCommand, JobRepositoryError> {
    let command_type: String = row_value(&row, "command_type")?;
    let actor_principal_id = PrincipalId::from_uuid(row_value(&row, "actor_principal_id")?);
    let executor_principal_id =
        row_value::<Option<Uuid>>(&row, "executor_principal_id")?.map(PrincipalId::from_uuid);
    let kind = match command_type.as_str() {
        "request"
            if actor_principal_id == requester_principal_id && executor_principal_id.is_none() =>
        {
            JobCommandKind::Request {
                requester_principal_id,
                target_executor_principal_id,
            }
        }
        "accept" => JobCommandKind::Accept {
            executor_principal_id: executor_principal_id
                .ok_or(JobRepositoryError::InvalidRecord)?,
        },
        "progress" => JobCommandKind::Progress {
            executor_principal_id: executor_principal_id
                .ok_or(JobRepositoryError::InvalidRecord)?,
        },
        "result" => JobCommandKind::Result {
            executor_principal_id: executor_principal_id
                .ok_or(JobRepositoryError::InvalidRecord)?,
        },
        "cancel" if executor_principal_id.is_none() => {
            JobCommandKind::Cancel { actor_principal_id }
        }
        "error" if executor_principal_id.is_none() => JobCommandKind::Error { actor_principal_id },
        _ => return Err(JobRepositoryError::InvalidRecord),
    };
    if kind.command_type() != JobCommandType::Request
        && matches!(kind, JobCommandKind::Accept { executor_principal_id } | JobCommandKind::Progress { executor_principal_id } | JobCommandKind::Result { executor_principal_id } if executor_principal_id != actor_principal_id)
    {
        return Err(JobRepositoryError::InvalidRecord);
    }
    JobCommand::new(
        identity,
        OperationId::from_uuid(row_value(&row, "operation_id")?),
        parse_version(row_value(&row, "version_text")?)?,
        parse_millis(row_value(&row, "occurred_at_millis")?)?,
        kind,
    )
    .map_err(JobRepositoryError::Domain)
}

fn validate_head(row: &QueryResult, job: &Job) -> Result<(), JobRepositoryError> {
    let state: String = row_value(row, "current_state")?;
    let executor = row_value::<Option<Uuid>>(row, "current_executor_principal_id")?
        .map(PrincipalId::from_uuid);
    if parse_version(row_value(row, "current_version_text")?)? != job.version()
        || parse_millis(row_value(row, "requested_at_millis")?)? != job.requested_at_millis()
        || parse_millis(row_value(row, "updated_at_millis")?)? != job.updated_at_millis()
        || state != state_name(job.state())
        || executor != job.state().executor_principal_id()
    {
        return Err(JobRepositoryError::InvalidRecord);
    }
    Ok(())
}

fn operation_statement(community_id: CommunityId, operation_id: OperationId) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        SELECT_OPERATION_SQL,
        [community_id.as_uuid().into(), operation_id.as_uuid().into()],
    )
}

fn insert_head_statement(job: &Job) -> Result<Statement, JobRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_HEAD_SQL,
        [
            job.identity().community_id().as_uuid().into(),
            job.identity().job_id().as_uuid().into(),
            job.requester_principal_id().as_uuid().into(),
            job.target_executor_principal_id().as_uuid().into(),
            job.version().to_string().into(),
            millis_value(job.requested_at_millis())?,
        ],
    ))
}

async fn insert_version(
    transaction: &DatabaseTransaction,
    command: &JobCommand,
) -> Result<(), JobRepositoryError> {
    let (actor_principal_id, executor_principal_id) = command_principals(command.kind());
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            INSERT_VERSION_SQL,
            [
                command.identity().community_id().as_uuid().into(),
                command.identity().job_id().as_uuid().into(),
                command.version().to_string().into(),
                command.operation_id().as_uuid().into(),
                command_type_name(command.kind().command_type()).into(),
                actor_principal_id.as_uuid().into(),
                executor_principal_id.map(PrincipalId::as_uuid).into(),
                millis_value(command.occurred_at_millis())?,
            ],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?;
    Ok(())
}

fn update_head_statement(
    job: &Job,
    previous_version: AggregateVersion,
) -> Result<Statement, JobRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPDATE_HEAD_SQL,
        [
            job.identity().community_id().as_uuid().into(),
            job.identity().job_id().as_uuid().into(),
            job.state()
                .executor_principal_id()
                .map(PrincipalId::as_uuid)
                .into(),
            job.version().to_string().into(),
            state_name(job.state()).into(),
            millis_value(job.updated_at_millis())?,
            previous_version.to_string().into(),
        ],
    ))
}

fn command_principals(kind: JobCommandKind) -> (PrincipalId, Option<PrincipalId>) {
    match kind {
        JobCommandKind::Request {
            requester_principal_id,
            ..
        } => (requester_principal_id, None),
        JobCommandKind::Accept {
            executor_principal_id,
        }
        | JobCommandKind::Progress {
            executor_principal_id,
        }
        | JobCommandKind::Result {
            executor_principal_id,
        } => (executor_principal_id, Some(executor_principal_id)),
        JobCommandKind::Cancel { actor_principal_id }
        | JobCommandKind::Error { actor_principal_id } => (actor_principal_id, None),
    }
}

const fn command_type_name(command_type: JobCommandType) -> &'static str {
    match command_type {
        JobCommandType::Request => "request",
        JobCommandType::Accept => "accept",
        JobCommandType::Progress => "progress",
        JobCommandType::Result => "result",
        JobCommandType::Cancel => "cancel",
        JobCommandType::Error => "error",
    }
}

const fn state_name(state: JobState) -> &'static str {
    match state {
        JobState::Requested => "requested",
        JobState::Accepted { .. } => "accepted",
        JobState::InProgress { .. } => "in_progress",
        JobState::Completed { .. } => "completed",
        JobState::Cancelled { .. } => "cancelled",
        JobState::Failed { .. } => "failed",
    }
}

async fn query_lease_by_id(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    lease_id: Uuid,
) -> Result<Option<ExecutorLease>, JobRepositoryError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_LEASE_BY_ID_SQL,
            [community_id.as_uuid().into(), lease_id.into()],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?
        .map(|row| lease_from_row(row, community_id))
        .transpose()
}

async fn query_active_lease(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    job_id: AggregateId,
) -> Result<Option<ExecutorLease>, JobRepositoryError> {
    transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_ACTIVE_LEASE_SQL,
            [community_id.as_uuid().into(), job_id.as_uuid().into()],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?
        .map(|row| lease_from_row(row, community_id))
        .transpose()
}

async fn query_maximum_generation(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
    job_id: AggregateId,
) -> Result<Option<AggregateVersion>, JobRepositoryError> {
    let row = transaction
        .query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SELECT_MAX_LEASE_GENERATION_SQL,
            [community_id.as_uuid().into(), job_id.as_uuid().into()],
        ))
        .await
        .map_err(JobRepositoryError::Unavailable)?
        .ok_or(JobRepositoryError::InvalidRecord)?;
    let value: String = row_value(&row, "lease_generation_text")?;
    let value = value
        .parse::<u64>()
        .map_err(|_| JobRepositoryError::InvalidRecord)?;
    Ok(AggregateVersion::new(value))
}

fn lease_from_row(
    row: QueryResult,
    community_id: CommunityId,
) -> Result<ExecutorLease, JobRepositoryError> {
    let state: String = row_value(&row, "state")?;
    let state = match state.as_str() {
        "active" => ExecutorLeaseState::Active,
        "released" => ExecutorLeaseState::Released,
        _ => return Err(JobRepositoryError::InvalidRecord),
    };
    let released_at_millis = row_value::<Option<i64>>(&row, "released_at_millis")?
        .map(parse_millis)
        .transpose()?;
    let release_reason = row_value::<Option<String>>(&row, "release_reason")?
        .as_deref()
        .map(ExecutorLeaseReleaseReason::from_database)
        .transpose()?;
    if !matches!(
        (state, released_at_millis, release_reason),
        (ExecutorLeaseState::Active, None, None) | (ExecutorLeaseState::Released, Some(_), Some(_))
    ) {
        return Err(JobRepositoryError::InvalidRecord);
    }
    let identity = JobIdentity::new(
        community_id,
        AggregateId::from_uuid(row_value(&row, "job_id")?),
    )?;
    let lease = ExecutorLease {
        identity,
        job_version: parse_version(row_value(&row, "job_version_text")?)?,
        generation: parse_version(row_value(&row, "lease_generation_text")?)?,
        lease_id: row_value(&row, "lease_id")?,
        executor_principal_id: PrincipalId::from_uuid(row_value(&row, "executor_principal_id")?),
        state,
        acquired_at_millis: parse_millis(row_value(&row, "acquired_at_millis")?)?,
        last_heartbeat_at_millis: parse_millis(row_value(&row, "last_heartbeat_at_millis")?)?,
        expires_at_millis: parse_millis(row_value(&row, "expires_at_millis")?)?,
        recovery_after_millis: parse_millis(row_value(&row, "recovery_after_millis")?)?,
        released_at_millis,
        release_reason,
    };
    if lease.lease_id.is_nil()
        || lease.executor_principal_id.as_uuid().is_nil()
        || lease.acquired_at_millis > lease.last_heartbeat_at_millis
        || lease.last_heartbeat_at_millis > lease.expires_at_millis
        || lease.expires_at_millis > lease.recovery_after_millis
        || lease
            .released_at_millis
            .is_some_and(|released| released < lease.acquired_at_millis)
    {
        return Err(JobRepositoryError::InvalidRecord);
    }
    Ok(lease)
}

fn insert_lease_statement(
    request: &ExecutorLeaseRequest,
    generation: AggregateVersion,
) -> Result<Statement, JobRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        INSERT_LEASE_SQL,
        [
            request.identity.community_id().as_uuid().into(),
            request.identity.job_id().as_uuid().into(),
            request.job_version.to_string().into(),
            generation.to_string().into(),
            request.lease_id.into(),
            request.executor_principal_id.as_uuid().into(),
            millis_value(request.acquired_at_millis)?,
            millis_value(request.expires_at_millis)?,
            millis_value(request.recovery_after_millis)?,
        ],
    ))
}

fn heartbeat_statement(
    community_id: CommunityId,
    fence: ExecutorLeaseFence,
    heartbeat_at_millis: u64,
    expires_at_millis: u64,
    recovery_after_millis: u64,
) -> Result<Statement, JobRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        HEARTBEAT_LEASE_SQL,
        [
            community_id.as_uuid().into(),
            fence.identity.job_id().as_uuid().into(),
            fence.generation.to_string().into(),
            fence.lease_id.into(),
            fence.executor_principal_id.as_uuid().into(),
            fence.job_version.to_string().into(),
            millis_value(heartbeat_at_millis)?,
            millis_value(expires_at_millis)?,
            millis_value(recovery_after_millis)?,
        ],
    ))
}

fn release_statement(
    community_id: CommunityId,
    fence: ExecutorLeaseFence,
    released_at_millis: u64,
    reason: ExecutorLeaseReleaseReason,
) -> Result<Statement, JobRepositoryError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        RELEASE_LEASE_SQL,
        [
            community_id.as_uuid().into(),
            fence.identity.job_id().as_uuid().into(),
            fence.generation.to_string().into(),
            fence.lease_id.into(),
            fence.executor_principal_id.as_uuid().into(),
            fence.job_version.to_string().into(),
            millis_value(released_at_millis)?,
            reason.database_name().into(),
        ],
    ))
}

fn lease_matches_request(lease: &ExecutorLease, request: &ExecutorLeaseRequest) -> bool {
    lease.identity == request.identity
        && lease.job_version == request.job_version
        && lease.lease_id == request.lease_id
        && lease.executor_principal_id == request.executor_principal_id
        && lease.acquired_at_millis == request.acquired_at_millis
        && lease.expires_at_millis == request.expires_at_millis
        && lease.recovery_after_millis == request.recovery_after_millis
}

fn lease_matches_fence(lease: &ExecutorLease, fence: ExecutorLeaseFence) -> bool {
    lease.identity == fence.identity
        && lease.job_version == fence.job_version
        && lease.generation == fence.generation
        && lease.lease_id == fence.lease_id
        && lease.executor_principal_id == fence.executor_principal_id
}

fn parse_version(value: String) -> Result<AggregateVersion, JobRepositoryError> {
    value
        .parse::<u64>()
        .ok()
        .and_then(AggregateVersion::new)
        .ok_or(JobRepositoryError::InvalidRecord)
}

fn parse_millis(value: i64) -> Result<u64, JobRepositoryError> {
    u64::try_from(value).map_err(|_| JobRepositoryError::InvalidRecord)
}

fn millis_value(value: u64) -> Result<sea_orm::Value, JobRepositoryError> {
    i64::try_from(value)
        .map(Into::into)
        .map_err(|_| JobRepositoryError::InvalidRecord)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, JobRepositoryError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| JobRepositoryError::InvalidRecord)
}
