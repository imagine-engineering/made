use std::fmt;

use collaboration_domain::{
    CommunityId, PrincipalId, PushCapabilityReference, PushEndpointGeneration, PushInstallationId,
    PushLease, PushLeaseAddress, PushLeaseError, PushLeaseGeneration, PushLeaseRecordFields,
    PushLeaseState, PushWake, PushWakePayload, TenantContext,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};
use uuid::Uuid;

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const UPSERT_LEASE_SQL: &str = r#"
INSERT INTO public.collaboration_push_leases (
    community_id, owner_principal_id, installation_id,
    source_event_id, source_created_at, generation, active,
    expires_at, last_active_expires_at, revoked_at,
    endpoint_generation, capability_reference, capability_ciphertext,
    subscription_policy_ciphertext, custody_key_id,
    endpoint_enabled, endpoint_disabled_at, accepted_at, updated_at
) VALUES (
    $1, $2, $3, $4, CAST($5 AS numeric), CAST($6 AS numeric), $7,
    to_timestamp($8::double precision / 1000),
    to_timestamp($9::double precision / 1000),
    to_timestamp($10::double precision / 1000),
    CAST($11 AS numeric), $12, $13, $14, $15, $16,
    to_timestamp($17::double precision / 1000),
    to_timestamp($18::double precision / 1000),
    to_timestamp($19::double precision / 1000)
)
ON CONFLICT (community_id, owner_principal_id, installation_id) DO UPDATE SET
    source_event_id = EXCLUDED.source_event_id,
    source_created_at = EXCLUDED.source_created_at,
    generation = EXCLUDED.generation,
    active = EXCLUDED.active,
    expires_at = EXCLUDED.expires_at,
    last_active_expires_at = EXCLUDED.last_active_expires_at,
    revoked_at = EXCLUDED.revoked_at,
    endpoint_generation = EXCLUDED.endpoint_generation,
    capability_reference = EXCLUDED.capability_reference,
    capability_ciphertext = EXCLUDED.capability_ciphertext,
    subscription_policy_ciphertext = EXCLUDED.subscription_policy_ciphertext,
    custody_key_id = EXCLUDED.custody_key_id,
    endpoint_enabled = EXCLUDED.endpoint_enabled,
    endpoint_disabled_at = EXCLUDED.endpoint_disabled_at,
    accepted_at = EXCLUDED.accepted_at,
    updated_at = EXCLUDED.updated_at
WHERE EXCLUDED.generation > public.collaboration_push_leases.generation
"#;
const SELECT_LEASE_SQL: &str = r#"
SELECT
    community_id, owner_principal_id, installation_id,
    source_event_id, source_created_at::text AS source_created_at_text,
    generation::text AS generation_text, active,
    floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
    floor(extract(epoch FROM last_active_expires_at) * 1000)::bigint
        AS last_active_expires_at_millis,
    floor(extract(epoch FROM revoked_at) * 1000)::bigint AS revoked_at_millis,
    endpoint_generation::text AS endpoint_generation_text,
    capability_reference, capability_ciphertext, subscription_policy_ciphertext,
    custody_key_id, endpoint_enabled,
    floor(extract(epoch FROM endpoint_disabled_at) * 1000)::bigint
        AS endpoint_disabled_at_millis,
    floor(extract(epoch FROM accepted_at) * 1000)::bigint AS accepted_at_millis,
    floor(extract(epoch FROM updated_at) * 1000)::bigint AS updated_at_millis
FROM public.collaboration_push_leases
WHERE community_id = $1 AND owner_principal_id = $2 AND installation_id = $3
"#;
const ENQUEUE_WAKE_SQL: &str = r#"
INSERT INTO public.collaboration_push_wake_jobs (
    community_id, wake_id, request_id, owner_principal_id, installation_id,
    lease_generation, endpoint_generation, capability_reference, source_event_id,
    expires_at, available_at, created_at
)
SELECT
    $1, $2, $3, $4, $5, CAST($6 AS numeric), CAST($7 AS numeric), $8, $9,
    to_timestamp($10::double precision / 1000),
    to_timestamp($11::double precision / 1000),
    to_timestamp($12::double precision / 1000)
FROM public.collaboration_push_leases AS lease
WHERE lease.community_id = $1
  AND lease.owner_principal_id = $4
  AND lease.installation_id = $5
  AND lease.active
  AND lease.endpoint_enabled
  AND lease.generation = CAST($6 AS numeric)
  AND lease.endpoint_generation = CAST($7 AS numeric)
  AND lease.capability_reference = $8
  AND lease.expires_at = to_timestamp($10::double precision / 1000)
  AND lease.expires_at >= to_timestamp($13::double precision / 1000)
ON CONFLICT DO NOTHING
"#;
const SELECT_WAKE_IDENTITY_SQL: &str = r#"
SELECT
    wake_id, request_id, owner_principal_id, installation_id,
    lease_generation::text AS lease_generation_text,
    endpoint_generation::text AS endpoint_generation_text,
    capability_reference, source_event_id,
    floor(extract(epoch FROM expires_at) * 1000)::bigint AS expires_at_millis,
    floor(extract(epoch FROM available_at) * 1000)::bigint AS available_at_millis,
    floor(extract(epoch FROM created_at) * 1000)::bigint AS created_at_millis
FROM public.collaboration_push_wake_jobs
WHERE community_id = $1
  AND (
      wake_id = $2
      OR request_id = $3
      OR (capability_reference = $4 AND source_event_id = $5)
  )
LIMIT 1
"#;
const CLAIM_WAKES_SQL: &str = r#"
WITH candidates AS (
    SELECT job.wake_id
    FROM public.collaboration_push_wake_jobs AS job
    JOIN public.collaboration_push_leases AS lease
      ON lease.community_id = job.community_id
     AND lease.owner_principal_id = job.owner_principal_id
     AND lease.installation_id = job.installation_id
     AND lease.active
     AND lease.endpoint_enabled
     AND lease.generation = job.lease_generation
     AND lease.endpoint_generation = job.endpoint_generation
     AND lease.capability_reference = job.capability_reference
     AND lease.expires_at = job.expires_at
    WHERE job.community_id = $1
      AND job.expires_at >= to_timestamp($2::double precision / 1000)
      AND (
          (job.state = 'pending'
              AND job.available_at <= to_timestamp($2::double precision / 1000))
          OR (job.state = 'leased'
              AND job.claim_expires_at <= to_timestamp($2::double precision / 1000))
    )
    ORDER BY job.available_at, job.created_at, job.wake_id
    LIMIT $3
    FOR UPDATE OF job SKIP LOCKED
)
UPDATE public.collaboration_push_wake_jobs AS job
SET state = 'leased',
    attempt_count = job.attempt_count + 1,
    claim_id = $4,
    claim_expires_at = to_timestamp($5::double precision / 1000)
FROM candidates
WHERE job.community_id = $1 AND job.wake_id = candidates.wake_id
RETURNING
    job.community_id, job.wake_id, job.request_id,
    job.owner_principal_id, job.installation_id,
    job.lease_generation::text AS lease_generation_text,
    job.endpoint_generation::text AS endpoint_generation_text,
    job.capability_reference, job.source_event_id,
    floor(extract(epoch FROM job.expires_at) * 1000)::bigint AS expires_at_millis,
    job.attempt_count, job.claim_id,
    floor(extract(epoch FROM job.claim_expires_at) * 1000)::bigint
        AS claim_expires_at_millis
"#;
const COMPLETE_WAKE_SQL: &str = r#"
UPDATE public.collaboration_push_wake_jobs
SET state = $4,
    claim_id = NULL,
    claim_expires_at = NULL,
    terminal_outcome = $5,
    completed_at = to_timestamp($6::double precision / 1000)
WHERE community_id = $1
  AND wake_id = $2
  AND state = 'leased'
  AND claim_id = $3
  AND claim_expires_at >= to_timestamp($6::double precision / 1000)
"#;
const REVALIDATE_WAKE_SQL: &str = r#"
SELECT EXISTS (
    SELECT 1
    FROM public.collaboration_push_wake_jobs AS job
    JOIN public.collaboration_push_leases AS lease
      ON lease.community_id = job.community_id
     AND lease.owner_principal_id = job.owner_principal_id
     AND lease.installation_id = job.installation_id
     AND lease.active
     AND lease.endpoint_enabled
     AND lease.generation = job.lease_generation
     AND lease.endpoint_generation = job.endpoint_generation
     AND lease.capability_reference = job.capability_reference
     AND lease.expires_at = job.expires_at
    WHERE job.community_id = $1
      AND job.wake_id = $2
      AND job.state = 'leased'
      AND job.claim_id = $3
      AND job.claim_expires_at >= to_timestamp($4::double precision / 1000)
      AND job.expires_at >= to_timestamp($4::double precision / 1000)
) AS authorized
"#;
const RETRY_WAKE_SQL: &str = r#"
UPDATE public.collaboration_push_wake_jobs
SET state = 'pending',
    available_at = to_timestamp($4::double precision / 1000),
    claim_id = NULL,
    claim_expires_at = NULL
WHERE community_id = $1
  AND wake_id = $2
  AND state = 'leased'
  AND claim_id = $3
  AND claim_expires_at >= to_timestamp($5::double precision / 1000)
  AND expires_at >= to_timestamp($4::double precision / 1000)
"#;
const DISABLE_WAKE_ENDPOINT_SQL: &str = r#"
UPDATE public.collaboration_push_leases AS lease
SET endpoint_enabled = false,
    endpoint_disabled_at = to_timestamp($4::double precision / 1000),
    updated_at = GREATEST(
        lease.updated_at,
        to_timestamp($4::double precision / 1000)
    )
FROM public.collaboration_push_wake_jobs AS job
WHERE job.community_id = $1
  AND job.wake_id = $2
  AND job.state = 'leased'
  AND job.claim_id = $3
  AND job.claim_expires_at >= to_timestamp($5::double precision / 1000)
  AND lease.community_id = job.community_id
  AND lease.owner_principal_id = job.owner_principal_id
  AND lease.installation_id = job.installation_id
  AND lease.active
  AND lease.endpoint_enabled
  AND lease.generation = job.lease_generation
  AND lease.endpoint_generation = job.endpoint_generation
  AND lease.capability_reference = job.capability_reference
"#;
const CANCEL_RETAINED_SOURCE_WAKES_SQL: &str = r#"
DELETE FROM public.collaboration_push_wake_jobs
WHERE community_id = $1
  AND source_event_id = $2
"#;

pub const MAX_PUSH_CLAIM_BATCH: u32 = 100;
pub const MAX_PUSH_CLAIM_MILLIS: u64 = 5 * 60 * 1_000;
const MAX_SAFE_GENERATION: u64 = 9_007_199_254_740_991;
const MAX_CAPABILITY_CIPHERTEXT_BYTES: usize = 16 * 1024;
const MAX_SUBSCRIPTION_POLICY_CIPHERTEXT_BYTES: usize = 1024 * 1024;
const MAX_CUSTODY_KEY_ID_BYTES: usize = 128;

#[derive(Clone, Eq, PartialEq)]
pub struct EncryptedPushAuthority {
    capability_ciphertext: Vec<u8>,
    subscription_policy_ciphertext: Vec<u8>,
    custody_key_id: String,
}

impl EncryptedPushAuthority {
    pub fn new(
        capability_ciphertext: Vec<u8>,
        subscription_policy_ciphertext: Vec<u8>,
        custody_key_id: impl Into<String>,
    ) -> Result<Self, PushOutboxError> {
        let custody_key_id = custody_key_id.into();
        if capability_ciphertext.is_empty()
            || capability_ciphertext.len() > MAX_CAPABILITY_CIPHERTEXT_BYTES
            || subscription_policy_ciphertext.is_empty()
            || subscription_policy_ciphertext.len() > MAX_SUBSCRIPTION_POLICY_CIPHERTEXT_BYTES
            || custody_key_id.is_empty()
            || custody_key_id.len() > MAX_CUSTODY_KEY_ID_BYTES
            || custody_key_id.trim() != custody_key_id
            || custody_key_id.chars().any(char::is_control)
        {
            return Err(PushOutboxError::InvalidRecord);
        }
        Ok(Self {
            capability_ciphertext,
            subscription_policy_ciphertext,
            custody_key_id,
        })
    }

    pub fn capability_ciphertext(&self) -> &[u8] {
        &self.capability_ciphertext
    }

    pub fn subscription_policy_ciphertext(&self) -> &[u8] {
        &self.subscription_policy_ciphertext
    }

    pub fn custody_key_id(&self) -> &str {
        &self.custody_key_id
    }
}

impl fmt::Debug for EncryptedPushAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("EncryptedPushAuthority([redacted])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushLeaseEventReference {
    event_id: [u8; 32],
    created_at: u64,
}

impl PushLeaseEventReference {
    pub const fn new(event_id: [u8; 32], created_at: u64) -> Self {
        Self {
            event_id,
            created_at,
        }
    }

    pub const fn event_id(self) -> [u8; 32] {
        self.event_id
    }

    pub const fn created_at(self) -> u64 {
        self.created_at
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushEndpointAuthorityState {
    Enabled,
    Disabled { disabled_at_millis: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushLeasePersistenceRecord {
    lease: PushLease,
    source_event: PushLeaseEventReference,
    encrypted_authority: Option<EncryptedPushAuthority>,
    endpoint_authority: PushEndpointAuthorityState,
    accepted_at_millis: u64,
    updated_at_millis: u64,
}

impl PushLeasePersistenceRecord {
    pub fn new(
        lease: PushLease,
        source_event: PushLeaseEventReference,
        encrypted_authority: Option<EncryptedPushAuthority>,
        endpoint_authority: PushEndpointAuthorityState,
        accepted_at_millis: u64,
        updated_at_millis: u64,
    ) -> Result<Self, PushOutboxError> {
        let fields = lease.fields();
        validate_generation(fields.generation.get())?;
        validate_millis(fields.expires_at_millis)?;
        validate_millis(fields.last_active_expires_at_millis)?;
        validate_millis(accepted_at_millis)?;
        validate_millis(updated_at_millis)?;
        if updated_at_millis < accepted_at_millis {
            return Err(PushOutboxError::InvalidRecord);
        }
        match (fields.state, &encrypted_authority, endpoint_authority) {
            (
                PushLeaseState::Active {
                    endpoint_generation,
                    ..
                },
                Some(_),
                PushEndpointAuthorityState::Enabled,
            ) => validate_generation(endpoint_generation.get())?,
            (
                PushLeaseState::Active {
                    endpoint_generation,
                    ..
                },
                Some(_),
                PushEndpointAuthorityState::Disabled { disabled_at_millis },
            ) => {
                validate_generation(endpoint_generation.get())?;
                validate_millis(disabled_at_millis)?;
            }
            (
                PushLeaseState::Revoked { revoked_at_millis },
                None,
                PushEndpointAuthorityState::Disabled { disabled_at_millis },
            ) => {
                validate_millis(revoked_at_millis)?;
                validate_millis(disabled_at_millis)?;
            }
            _ => return Err(PushOutboxError::InvalidRecord),
        }
        Ok(Self {
            lease,
            source_event,
            encrypted_authority,
            endpoint_authority,
            accepted_at_millis,
            updated_at_millis,
        })
    }

    pub const fn lease(&self) -> &PushLease {
        &self.lease
    }

    pub const fn source_event(&self) -> PushLeaseEventReference {
        self.source_event
    }

    pub const fn encrypted_authority(&self) -> Option<&EncryptedPushAuthority> {
        self.encrypted_authority.as_ref()
    }

    pub const fn endpoint_authority(&self) -> PushEndpointAuthorityState {
        self.endpoint_authority
    }

    pub const fn accepted_at_millis(&self) -> u64 {
        self.accepted_at_millis
    }

    pub const fn updated_at_millis(&self) -> u64 {
        self.updated_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushLeasePersistenceOutcome {
    Applied,
    Stale,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushWakeJobRequest {
    wake_id: Uuid,
    request_id: Uuid,
    wake: PushWake,
    source_event_id: [u8; 32],
    available_at_millis: u64,
    created_at_millis: u64,
}

impl PushWakeJobRequest {
    pub fn new(
        wake_id: Uuid,
        request_id: Uuid,
        wake: PushWake,
        source_event_id: [u8; 32],
        available_at_millis: u64,
        created_at_millis: u64,
    ) -> Result<Self, PushOutboxError> {
        validate_generation(wake.lease_generation().get())?;
        validate_generation(wake.endpoint_generation().get())?;
        validate_millis(wake.expires_at_millis())?;
        validate_millis(available_at_millis)?;
        validate_millis(created_at_millis)?;
        if wake_id.is_nil()
            || request_id.is_nil()
            || available_at_millis < created_at_millis
            || wake.expires_at_millis() < created_at_millis
        {
            return Err(PushOutboxError::InvalidRecord);
        }
        Ok(Self {
            wake_id,
            request_id,
            wake,
            source_event_id,
            available_at_millis,
            created_at_millis,
        })
    }

    pub const fn wake_id(&self) -> Uuid {
        self.wake_id
    }

    pub const fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub const fn wake(&self) -> &PushWake {
        &self.wake
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushWakeEnqueueOutcome {
    Enqueued,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushRetentionCancellationOutcome {
    Cancelled(u64),
    AlreadyClear,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimedPushWakeJob {
    community_id: CommunityId,
    wake_id: Uuid,
    request_id: Uuid,
    address: PushLeaseAddress,
    lease_generation: PushLeaseGeneration,
    endpoint_generation: PushEndpointGeneration,
    capability_reference: PushCapabilityReference,
    source_event_id: [u8; 32],
    expires_at_millis: u64,
    attempt_count: u32,
    claim_id: Uuid,
    claim_expires_at_millis: u64,
}

impl ClaimedPushWakeJob {
    pub const fn payload(&self) -> PushWakePayload {
        PushWakePayload::Reconnect
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn wake_id(&self) -> Uuid {
        self.wake_id
    }

    pub const fn request_id(&self) -> Uuid {
        self.request_id
    }

    pub const fn address(&self) -> &PushLeaseAddress {
        &self.address
    }

    pub const fn lease_generation(&self) -> PushLeaseGeneration {
        self.lease_generation
    }

    pub const fn endpoint_generation(&self) -> PushEndpointGeneration {
        self.endpoint_generation
    }

    pub const fn capability_reference(&self) -> PushCapabilityReference {
        self.capability_reference
    }

    pub const fn source_event_id(&self) -> [u8; 32] {
        self.source_event_id
    }

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }

    pub const fn attempt_count(&self) -> u32 {
        self.attempt_count
    }

    pub const fn claim_id(&self) -> Uuid {
        self.claim_id
    }

    pub const fn claim_expires_at_millis(&self) -> u64 {
        self.claim_expires_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushWakeClaim {
    claim_id: Uuid,
    now_millis: u64,
    claim_expires_at_millis: u64,
    limit: u32,
}

impl PushWakeClaim {
    pub fn new(
        claim_id: Uuid,
        now_millis: u64,
        claim_expires_at_millis: u64,
        limit: u32,
    ) -> Result<Self, PushOutboxError> {
        validate_millis(now_millis)?;
        validate_millis(claim_expires_at_millis)?;
        let claim_duration = claim_expires_at_millis
            .checked_sub(now_millis)
            .ok_or(PushOutboxError::InvalidClaim)?;
        if claim_id.is_nil()
            || limit == 0
            || limit > MAX_PUSH_CLAIM_BATCH
            || claim_duration == 0
            || claim_duration > MAX_PUSH_CLAIM_MILLIS
        {
            return Err(PushOutboxError::InvalidClaim);
        }
        Ok(Self {
            claim_id,
            now_millis,
            claim_expires_at_millis,
            limit,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushWakeTerminalOutcome {
    Accepted,
    InvalidEndpoint,
    RetryExhausted,
    LeaseUnavailable,
    AuthorizationLost,
    Expired,
}

impl PushWakeTerminalOutcome {
    const fn database_values(self) -> (&'static str, &'static str) {
        match self {
            Self::Accepted => ("delivered", "accepted"),
            Self::InvalidEndpoint => ("failed", "invalid_endpoint"),
            Self::RetryExhausted => ("failed", "retry_exhausted"),
            Self::LeaseUnavailable => ("suppressed", "lease_unavailable"),
            Self::AuthorizationLost => ("suppressed", "authorization_lost"),
            Self::Expired => ("suppressed", "expired"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PushOutboxError {
    #[error("push outbox persistence requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("push outbox request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("push outbox record is invalid or exceeds a bound")]
    InvalidRecord,
    #[error("push wake claim is invalid or exceeds a bound")]
    InvalidClaim,
    #[error("push wake idempotency key collides with different work")]
    IdempotencyCollision,
    #[error("current push device authority does not authorize the wake")]
    AuthorityUnavailable,
    #[error("push wake claim is no longer current")]
    ClaimLost,
    #[error(transparent)]
    Domain(#[from] PushLeaseError),
    #[error("push outbox persistence is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct PushOutboxRepository {
    connection: DatabaseConnection,
}

impl PushOutboxRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, PushOutboxError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(PushOutboxError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    pub async fn load_lease(
        &self,
        tenant: &TenantContext,
        address: &PushLeaseAddress,
    ) -> Result<Option<PushLeasePersistenceRecord>, PushOutboxError> {
        require_tenant(tenant, address.community_id)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let row = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SELECT_LEASE_SQL,
                    [
                        address.community_id.as_uuid().into(),
                        address.owner_principal_id.as_uuid().into(),
                        address.installation_id.as_str().to_owned().into(),
                    ],
                ))
                .await
                .map_err(PushOutboxError::Unavailable)?;
            let record = row.map(lease_from_row).transpose()?;
            if record
                .as_ref()
                .is_some_and(|record| record.lease.fields().address != *address)
            {
                return Err(PushOutboxError::TenantBoundaryViolation);
            }
            Ok(record)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn upsert_lease(
        &self,
        tenant: &TenantContext,
        record: &PushLeasePersistenceRecord,
    ) -> Result<PushLeasePersistenceOutcome, PushOutboxError> {
        require_tenant(tenant, record.lease.fields().address.community_id)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let result = transaction
                .execute(upsert_lease_statement(record)?)
                .await
                .map_err(PushOutboxError::Unavailable)?;
            Ok(if result.rows_affected() == 1 {
                PushLeasePersistenceOutcome::Applied
            } else {
                PushLeasePersistenceOutcome::Stale
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn enqueue_wake(
        &self,
        tenant: &TenantContext,
        request: &PushWakeJobRequest,
        now_millis: u64,
    ) -> Result<PushWakeEnqueueOutcome, PushOutboxError> {
        require_tenant(tenant, request.wake.address().community_id)?;
        validate_millis(now_millis)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let insert = transaction
                .execute(enqueue_wake_statement(request, now_millis)?)
                .await
                .map_err(PushOutboxError::Unavailable)?;
            if insert.rows_affected() == 1 {
                return Ok(PushWakeEnqueueOutcome::Enqueued);
            }
            let duplicate = transaction
                .query_one(wake_identity_statement(request))
                .await
                .map_err(PushOutboxError::Unavailable)?;
            match duplicate {
                Some(row) if wake_identity_matches(&row, request)? => {
                    Ok(PushWakeEnqueueOutcome::Duplicate)
                }
                Some(_) => Err(PushOutboxError::IdempotencyCollision),
                None => Err(PushOutboxError::AuthorityUnavailable),
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn claim_wakes(
        &self,
        tenant: &TenantContext,
        claim: PushWakeClaim,
    ) -> Result<Vec<ClaimedPushWakeJob>, PushOutboxError> {
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let rows = transaction
                .query_all(claim_wakes_statement(tenant.community_id(), claim)?)
                .await
                .map_err(PushOutboxError::Unavailable)?;
            let mut claimed = Vec::with_capacity(rows.len());
            for row in rows {
                let wake = claimed_wake_from_row(row)?;
                require_tenant(tenant, wake.community_id)?;
                claimed.push(wake);
            }
            Ok(claimed)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn complete_wake(
        &self,
        tenant: &TenantContext,
        wake: &ClaimedPushWakeJob,
        outcome: PushWakeTerminalOutcome,
        completed_at_millis: u64,
    ) -> Result<(), PushOutboxError> {
        require_tenant(tenant, wake.community_id)?;
        self.complete_claim(
            tenant,
            wake.wake_id,
            wake.claim_id,
            outcome,
            completed_at_millis,
        )
        .await
    }

    pub async fn revalidate_claim(
        &self,
        tenant: &TenantContext,
        wake_id: Uuid,
        claim_id: Uuid,
        now_millis: u64,
    ) -> Result<bool, PushOutboxError> {
        validate_claim_identity(wake_id, claim_id)?;
        validate_millis(now_millis)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let row = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    REVALIDATE_WAKE_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        wake_id.into(),
                        claim_id.into(),
                        millis_i64(now_millis)?.into(),
                    ],
                ))
                .await
                .map_err(PushOutboxError::Unavailable)?
                .ok_or(PushOutboxError::InvalidRecord)?;
            row_value(&row, "authorized")
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn retry_claim(
        &self,
        tenant: &TenantContext,
        wake_id: Uuid,
        claim_id: Uuid,
        available_at_millis: u64,
        now_millis: u64,
    ) -> Result<(), PushOutboxError> {
        validate_claim_identity(wake_id, claim_id)?;
        validate_millis(available_at_millis)?;
        validate_millis(now_millis)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let retried = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    RETRY_WAKE_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        wake_id.into(),
                        claim_id.into(),
                        millis_i64(available_at_millis)?.into(),
                        millis_i64(now_millis)?.into(),
                    ],
                ))
                .await
                .map_err(PushOutboxError::Unavailable)?;
            if retried.rows_affected() == 1 {
                Ok(())
            } else {
                Err(PushOutboxError::ClaimLost)
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn disable_claimed_endpoint(
        &self,
        tenant: &TenantContext,
        wake_id: Uuid,
        claim_id: Uuid,
        disabled_at_millis: u64,
        now_millis: u64,
    ) -> Result<bool, PushOutboxError> {
        validate_claim_identity(wake_id, claim_id)?;
        validate_millis(disabled_at_millis)?;
        validate_millis(now_millis)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let disabled = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    DISABLE_WAKE_ENDPOINT_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        wake_id.into(),
                        claim_id.into(),
                        millis_i64(disabled_at_millis)?.into(),
                        millis_i64(now_millis)?.into(),
                    ],
                ))
                .await
                .map_err(PushOutboxError::Unavailable)?;
            Ok(disabled.rows_affected() == 1)
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn complete_claim(
        &self,
        tenant: &TenantContext,
        wake_id: Uuid,
        claim_id: Uuid,
        outcome: PushWakeTerminalOutcome,
        completed_at_millis: u64,
    ) -> Result<(), PushOutboxError> {
        validate_claim_identity(wake_id, claim_id)?;
        validate_millis(completed_at_millis)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let (state, terminal_outcome) = outcome.database_values();
            let completed = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    COMPLETE_WAKE_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        wake_id.into(),
                        claim_id.into(),
                        state.to_owned().into(),
                        terminal_outcome.to_owned().into(),
                        millis_i64(completed_at_millis)?.into(),
                    ],
                ))
                .await
                .map_err(PushOutboxError::Unavailable)?;
            if completed.rows_affected() == 1 {
                Ok(())
            } else {
                Err(PushOutboxError::ClaimLost)
            }
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn cancel_source_wakes_after_retention(
        &self,
        tenant: &TenantContext,
        source_event_id: [u8; 32],
    ) -> Result<PushRetentionCancellationOutcome, PushOutboxError> {
        if source_event_id == [0; 32] {
            return Err(PushOutboxError::InvalidRecord);
        }
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, tenant.community_id()).await?;
            let cancelled = transaction
                .execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    CANCEL_RETAINED_SOURCE_WAKES_SQL,
                    [
                        tenant.community_id().as_uuid().into(),
                        source_event_id.to_vec().into(),
                    ],
                ))
                .await
                .map_err(PushOutboxError::Unavailable)?;
            Ok(if cancelled.rows_affected() == 0 {
                PushRetentionCancellationOutcome::AlreadyClear
            } else {
                PushRetentionCancellationOutcome::Cancelled(cancelled.rows_affected())
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    async fn begin(&self) -> Result<DatabaseTransaction, PushOutboxError> {
        self.connection
            .begin()
            .await
            .map_err(PushOutboxError::Unavailable)
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    community_id: CommunityId,
) -> Result<(), PushOutboxError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [community_id.to_string().into()],
        ))
        .await
        .map_err(PushOutboxError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, PushOutboxError>,
) -> Result<T, PushOutboxError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(PushOutboxError::Unavailable)?;
            Ok(value)
        }
        Err(error) => match transaction.rollback().await {
            Ok(()) => Err(error),
            Err(rollback_error) => Err(PushOutboxError::Unavailable(rollback_error)),
        },
    }
}

fn require_tenant(
    tenant: &TenantContext,
    community_id: CommunityId,
) -> Result<(), PushOutboxError> {
    if tenant.community_id() != community_id {
        return Err(PushOutboxError::TenantBoundaryViolation);
    }
    Ok(())
}

fn validate_claim_identity(wake_id: Uuid, claim_id: Uuid) -> Result<(), PushOutboxError> {
    if wake_id.is_nil() || claim_id.is_nil() {
        return Err(PushOutboxError::InvalidClaim);
    }
    Ok(())
}

fn upsert_lease_statement(
    record: &PushLeasePersistenceRecord,
) -> Result<Statement, PushOutboxError> {
    let fields = record.lease.fields();
    let (active, revoked_at, endpoint_generation, capability_reference) = match fields.state {
        PushLeaseState::Active {
            capability_reference,
            endpoint_generation,
        } => (
            true,
            None,
            Some(endpoint_generation.get().to_string()),
            Some(capability_reference.as_digest().to_vec()),
        ),
        PushLeaseState::Revoked { revoked_at_millis } => {
            (false, Some(millis_i64(revoked_at_millis)?), None, None)
        }
    };
    let (capability_ciphertext, subscription_policy_ciphertext, custody_key_id) = record
        .encrypted_authority
        .as_ref()
        .map(|authority| {
            (
                Some(authority.capability_ciphertext.clone()),
                Some(authority.subscription_policy_ciphertext.clone()),
                Some(authority.custody_key_id.clone()),
            )
        })
        .unwrap_or((None, None, None));
    let (endpoint_enabled, endpoint_disabled_at) = match record.endpoint_authority {
        PushEndpointAuthorityState::Enabled => (true, None),
        PushEndpointAuthorityState::Disabled { disabled_at_millis } => {
            (false, Some(millis_i64(disabled_at_millis)?))
        }
    };
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        UPSERT_LEASE_SQL,
        [
            fields.address.community_id.as_uuid().into(),
            fields.address.owner_principal_id.as_uuid().into(),
            fields.address.installation_id.as_str().to_owned().into(),
            record.source_event.event_id.to_vec().into(),
            record.source_event.created_at.to_string().into(),
            fields.generation.get().to_string().into(),
            active.into(),
            millis_i64(fields.expires_at_millis)?.into(),
            millis_i64(fields.last_active_expires_at_millis)?.into(),
            revoked_at.into(),
            endpoint_generation.into(),
            capability_reference.into(),
            capability_ciphertext.into(),
            subscription_policy_ciphertext.into(),
            custody_key_id.into(),
            endpoint_enabled.into(),
            endpoint_disabled_at.into(),
            millis_i64(record.accepted_at_millis)?.into(),
            millis_i64(record.updated_at_millis)?.into(),
        ],
    ))
}

fn enqueue_wake_statement(
    request: &PushWakeJobRequest,
    now_millis: u64,
) -> Result<Statement, PushOutboxError> {
    let wake = &request.wake;
    let address = wake.address();
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        ENQUEUE_WAKE_SQL,
        [
            address.community_id.as_uuid().into(),
            request.wake_id.into(),
            request.request_id.into(),
            address.owner_principal_id.as_uuid().into(),
            address.installation_id.as_str().to_owned().into(),
            wake.lease_generation().get().to_string().into(),
            wake.endpoint_generation().get().to_string().into(),
            wake.capability_reference().as_digest().to_vec().into(),
            request.source_event_id.to_vec().into(),
            millis_i64(wake.expires_at_millis())?.into(),
            millis_i64(request.available_at_millis)?.into(),
            millis_i64(request.created_at_millis)?.into(),
            millis_i64(now_millis)?.into(),
        ],
    ))
}

fn wake_identity_statement(request: &PushWakeJobRequest) -> Statement {
    Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        SELECT_WAKE_IDENTITY_SQL,
        [
            request.wake.address().community_id.as_uuid().into(),
            request.wake_id.into(),
            request.request_id.into(),
            request
                .wake
                .capability_reference()
                .as_digest()
                .to_vec()
                .into(),
            request.source_event_id.to_vec().into(),
        ],
    )
}

fn wake_identity_matches(
    row: &QueryResult,
    request: &PushWakeJobRequest,
) -> Result<bool, PushOutboxError> {
    let address = request.wake.address();
    Ok(row_value::<Uuid>(row, "wake_id")? == request.wake_id
        && row_value::<Uuid>(row, "request_id")? == request.request_id
        && row_value::<Uuid>(row, "owner_principal_id")? == address.owner_principal_id.as_uuid()
        && row_value::<String>(row, "installation_id")? == address.installation_id.as_str()
        && parse_generation(row_value(row, "lease_generation_text")?)?
            == request.wake.lease_generation().get()
        && parse_generation(row_value(row, "endpoint_generation_text")?)?
            == request.wake.endpoint_generation().get()
        && fixed_bytes::<32>(row_value(row, "capability_reference")?)?
            == *request.wake.capability_reference().as_digest()
        && fixed_bytes::<32>(row_value(row, "source_event_id")?)? == request.source_event_id
        && nonnegative_millis(row_value(row, "expires_at_millis")?)?
            == request.wake.expires_at_millis()
        && nonnegative_millis(row_value(row, "available_at_millis")?)?
            == request.available_at_millis
        && nonnegative_millis(row_value(row, "created_at_millis")?)? == request.created_at_millis)
}

fn claim_wakes_statement(
    community_id: CommunityId,
    claim: PushWakeClaim,
) -> Result<Statement, PushOutboxError> {
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        CLAIM_WAKES_SQL,
        [
            community_id.as_uuid().into(),
            millis_i64(claim.now_millis)?.into(),
            i64::from(claim.limit).into(),
            claim.claim_id.into(),
            millis_i64(claim.claim_expires_at_millis)?.into(),
        ],
    ))
}

fn lease_from_row(row: QueryResult) -> Result<PushLeasePersistenceRecord, PushOutboxError> {
    let community_id = CommunityId::from_uuid(row_value(&row, "community_id")?);
    let owner_principal_id = PrincipalId::from_uuid(row_value(&row, "owner_principal_id")?);
    let installation_id = PushInstallationId::new(row_value::<String>(&row, "installation_id")?)
        .ok_or(PushOutboxError::InvalidRecord)?;
    let generation =
        PushLeaseGeneration::new(parse_generation(row_value(&row, "generation_text")?)?)
            .ok_or(PushOutboxError::InvalidRecord)?;
    let expires_at_millis = nonnegative_millis(row_value(&row, "expires_at_millis")?)?;
    let last_active_expires_at_millis =
        nonnegative_millis(row_value(&row, "last_active_expires_at_millis")?)?;
    let active: bool = row_value(&row, "active")?;
    let state = if active {
        PushLeaseState::Active {
            capability_reference: PushCapabilityReference::from_digest(fixed_bytes(row_value(
                &row,
                "capability_reference",
            )?)?)
            .ok_or(PushOutboxError::InvalidRecord)?,
            endpoint_generation: PushEndpointGeneration::new(parse_generation(row_value(
                &row,
                "endpoint_generation_text",
            )?)?)
            .ok_or(PushOutboxError::InvalidRecord)?,
        }
    } else {
        PushLeaseState::Revoked {
            revoked_at_millis: nonnegative_millis(row_value(&row, "revoked_at_millis")?)?,
        }
    };
    let lease = PushLease::from_record(PushLeaseRecordFields {
        address: PushLeaseAddress {
            community_id,
            owner_principal_id,
            installation_id,
        },
        generation,
        expires_at_millis,
        last_active_expires_at_millis,
        state,
    })?;
    let encrypted_authority = if active {
        Some(EncryptedPushAuthority::new(
            row_value(&row, "capability_ciphertext")?,
            row_value(&row, "subscription_policy_ciphertext")?,
            row_value::<String>(&row, "custody_key_id")?,
        )?)
    } else {
        None
    };
    let endpoint_authority = if row_value(&row, "endpoint_enabled")? {
        PushEndpointAuthorityState::Enabled
    } else {
        PushEndpointAuthorityState::Disabled {
            disabled_at_millis: nonnegative_millis(row_value(
                &row,
                "endpoint_disabled_at_millis",
            )?)?,
        }
    };
    PushLeasePersistenceRecord::new(
        lease,
        PushLeaseEventReference::new(
            fixed_bytes(row_value(&row, "source_event_id")?)?,
            row_value::<String>(&row, "source_created_at_text")?
                .parse()
                .map_err(|_| PushOutboxError::InvalidRecord)?,
        ),
        encrypted_authority,
        endpoint_authority,
        nonnegative_millis(row_value(&row, "accepted_at_millis")?)?,
        nonnegative_millis(row_value(&row, "updated_at_millis")?)?,
    )
}

fn claimed_wake_from_row(row: QueryResult) -> Result<ClaimedPushWakeJob, PushOutboxError> {
    let community_id = CommunityId::from_uuid(row_value(&row, "community_id")?);
    let lease_generation =
        PushLeaseGeneration::new(parse_generation(row_value(&row, "lease_generation_text")?)?)
            .ok_or(PushOutboxError::InvalidRecord)?;
    let endpoint_generation = PushEndpointGeneration::new(parse_generation(row_value(
        &row,
        "endpoint_generation_text",
    )?)?)
    .ok_or(PushOutboxError::InvalidRecord)?;
    let attempt_count: i32 = row_value(&row, "attempt_count")?;
    Ok(ClaimedPushWakeJob {
        community_id,
        wake_id: row_value(&row, "wake_id")?,
        request_id: row_value(&row, "request_id")?,
        address: PushLeaseAddress {
            community_id,
            owner_principal_id: PrincipalId::from_uuid(row_value(&row, "owner_principal_id")?),
            installation_id: PushInstallationId::new(row_value::<String>(&row, "installation_id")?)
                .ok_or(PushOutboxError::InvalidRecord)?,
        },
        lease_generation,
        endpoint_generation,
        capability_reference: PushCapabilityReference::from_digest(fixed_bytes(row_value(
            &row,
            "capability_reference",
        )?)?)
        .ok_or(PushOutboxError::InvalidRecord)?,
        source_event_id: fixed_bytes(row_value(&row, "source_event_id")?)?,
        expires_at_millis: nonnegative_millis(row_value(&row, "expires_at_millis")?)?,
        attempt_count: u32::try_from(attempt_count).map_err(|_| PushOutboxError::InvalidRecord)?,
        claim_id: row_value(&row, "claim_id")?,
        claim_expires_at_millis: nonnegative_millis(row_value(&row, "claim_expires_at_millis")?)?,
    })
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, PushOutboxError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| PushOutboxError::InvalidRecord)
}

fn fixed_bytes<const LENGTH: usize>(value: Vec<u8>) -> Result<[u8; LENGTH], PushOutboxError> {
    value.try_into().map_err(|_| PushOutboxError::InvalidRecord)
}

fn parse_generation(value: String) -> Result<u64, PushOutboxError> {
    let value = value.parse().map_err(|_| PushOutboxError::InvalidRecord)?;
    validate_generation(value)?;
    Ok(value)
}

fn validate_generation(value: u64) -> Result<(), PushOutboxError> {
    if value == 0 || value > MAX_SAFE_GENERATION {
        return Err(PushOutboxError::InvalidRecord);
    }
    Ok(())
}

fn validate_millis(value: u64) -> Result<(), PushOutboxError> {
    millis_i64(value).map(|_| ())
}

fn millis_i64(value: u64) -> Result<i64, PushOutboxError> {
    i64::try_from(value).map_err(|_| PushOutboxError::InvalidRecord)
}

fn nonnegative_millis(value: i64) -> Result<u64, PushOutboxError> {
    u64::try_from(value).map_err(|_| PushOutboxError::InvalidRecord)
}
