use async_trait::async_trait;
use collab::push::outbox::{
    ClaimedPushWakeJob, PushOutboxError, PushOutboxRepository, PushWakeClaim,
    PushWakeTerminalOutcome,
};
use collaboration_domain::{
    CommunityId, PushCapabilityReference, PushEndpointGeneration, PushLeaseAddress,
    PushLeaseGeneration, PushWakePayload, TenantContext,
};
use uuid::Uuid;

pub const PUSH_WAKE_BATCH_LIMIT: u32 = 16;
pub const PUSH_CLAIM_MILLIS: u64 = 30_000;
pub const PUSH_MAX_ATTEMPTS: u32 = 8;
pub const PUSH_DEFAULT_RETRY_MILLIS: u64 = 2_000;
pub const PUSH_MIN_RETRY_MILLIS: u64 = 1_000;
pub const PUSH_MAX_RETRY_MILLIS: u64 = 3_600_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushGatewayWake {
    address: PushLeaseAddress,
    wake_id: Uuid,
    request_id: Uuid,
    lease_generation: PushLeaseGeneration,
    endpoint_generation: PushEndpointGeneration,
    capability_reference: PushCapabilityReference,
    expires_at_millis: u64,
    attempt_count: u32,
    claim_id: Uuid,
    claim_expires_at_millis: u64,
}

impl PushGatewayWake {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        address: PushLeaseAddress,
        wake_id: Uuid,
        request_id: Uuid,
        lease_generation: PushLeaseGeneration,
        endpoint_generation: PushEndpointGeneration,
        capability_reference: PushCapabilityReference,
        expires_at_millis: u64,
        attempt_count: u32,
        claim_id: Uuid,
        claim_expires_at_millis: u64,
    ) -> Result<Self, PushGatewayExecutorError> {
        if address.community_id.as_uuid().is_nil()
            || wake_id.is_nil()
            || request_id.is_nil()
            || claim_id.is_nil()
            || expires_at_millis == 0
            || attempt_count == 0
            || claim_expires_at_millis == 0
        {
            return Err(PushGatewayExecutorError::InvalidWork);
        }
        Ok(Self {
            address,
            wake_id,
            request_id,
            lease_generation,
            endpoint_generation,
            capability_reference,
            expires_at_millis,
            attempt_count,
            claim_id,
            claim_expires_at_millis,
        })
    }

    fn from_claimed(wake: ClaimedPushWakeJob) -> Result<Self, PushWakeStoreError> {
        Self::new(
            wake.address().clone(),
            wake.wake_id(),
            wake.request_id(),
            wake.lease_generation(),
            wake.endpoint_generation(),
            wake.capability_reference(),
            wake.expires_at_millis(),
            wake.attempt_count(),
            wake.claim_id(),
            wake.claim_expires_at_millis(),
        )
        .map_err(|_| PushWakeStoreError::InvalidRecord)
    }

    pub const fn community_id(&self) -> CommunityId {
        self.address.community_id
    }

    pub const fn address(&self) -> &PushLeaseAddress {
        &self.address
    }

    pub const fn wake_id(&self) -> Uuid {
        self.wake_id
    }

    pub const fn request_id(&self) -> Uuid {
        self.request_id
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

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }

    pub const fn attempt_count(&self) -> u32 {
        self.attempt_count
    }

    pub const fn claim_id(&self) -> Uuid {
        self.claim_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushGatewayClaim {
    claim_id: Uuid,
    now_millis: u64,
    claim_expires_at_millis: u64,
    limit: u32,
}

impl PushGatewayClaim {
    fn new(claim_id: Uuid, now_millis: u64) -> Result<Self, PushGatewayExecutorError> {
        let claim_expires_at_millis = now_millis
            .checked_add(PUSH_CLAIM_MILLIS)
            .ok_or(PushGatewayExecutorError::InvalidTime)?;
        if claim_id.is_nil() || i64::try_from(claim_expires_at_millis).is_err() {
            return Err(PushGatewayExecutorError::InvalidTime);
        }
        Ok(Self {
            claim_id,
            now_millis,
            claim_expires_at_millis,
            limit: PUSH_WAKE_BATCH_LIMIT,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PushWakeStoreError {
    #[error("push wake persistence is unavailable")]
    Unavailable,
    #[error("push wake claim is no longer current")]
    ClaimLost,
    #[error("push wake persistence returned an invalid record")]
    InvalidRecord,
}

#[async_trait]
pub trait PushWakeStore: Send + Sync {
    async fn claim(
        &self,
        tenant: &TenantContext,
        claim: PushGatewayClaim,
    ) -> Result<Vec<PushGatewayWake>, PushWakeStoreError>;

    async fn revalidate(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        now_millis: u64,
    ) -> Result<bool, PushWakeStoreError>;

    async fn retry(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        available_at_millis: u64,
        now_millis: u64,
    ) -> Result<(), PushWakeStoreError>;

    async fn disable_endpoint(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        disabled_at_millis: u64,
        now_millis: u64,
    ) -> Result<bool, PushWakeStoreError>;

    async fn complete(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        outcome: PushWakeTerminalOutcome,
        completed_at_millis: u64,
    ) -> Result<(), PushWakeStoreError>;
}

#[async_trait]
impl PushWakeStore for PushOutboxRepository {
    async fn claim(
        &self,
        tenant: &TenantContext,
        claim: PushGatewayClaim,
    ) -> Result<Vec<PushGatewayWake>, PushWakeStoreError> {
        let claim = PushWakeClaim::new(
            claim.claim_id,
            claim.now_millis,
            claim.claim_expires_at_millis,
            claim.limit,
        )
        .map_err(map_store_error)?;
        self.claim_wakes(tenant, claim)
            .await
            .map_err(map_store_error)?
            .into_iter()
            .map(PushGatewayWake::from_claimed)
            .collect()
    }

    async fn revalidate(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        now_millis: u64,
    ) -> Result<bool, PushWakeStoreError> {
        self.revalidate_claim(tenant, wake.wake_id, wake.claim_id, now_millis)
            .await
            .map_err(map_store_error)
    }

    async fn retry(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        available_at_millis: u64,
        now_millis: u64,
    ) -> Result<(), PushWakeStoreError> {
        self.retry_claim(
            tenant,
            wake.wake_id,
            wake.claim_id,
            available_at_millis,
            now_millis,
        )
        .await
        .map_err(map_store_error)
    }

    async fn disable_endpoint(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        disabled_at_millis: u64,
        now_millis: u64,
    ) -> Result<bool, PushWakeStoreError> {
        self.disable_claimed_endpoint(
            tenant,
            wake.wake_id,
            wake.claim_id,
            disabled_at_millis,
            now_millis,
        )
        .await
        .map_err(map_store_error)
    }

    async fn complete(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        outcome: PushWakeTerminalOutcome,
        completed_at_millis: u64,
    ) -> Result<(), PushWakeStoreError> {
        self.complete_claim(
            tenant,
            wake.wake_id,
            wake.claim_id,
            outcome,
            completed_at_millis,
        )
        .await
        .map_err(map_store_error)
    }
}

fn map_store_error(error: PushOutboxError) -> PushWakeStoreError {
    match error {
        PushOutboxError::ClaimLost => PushWakeStoreError::ClaimLost,
        PushOutboxError::InvalidRecord
        | PushOutboxError::InvalidClaim
        | PushOutboxError::TenantBoundaryViolation => PushWakeStoreError::InvalidRecord,
        PushOutboxError::UnsupportedDatabase
        | PushOutboxError::IdempotencyCollision
        | PushOutboxError::AuthorityUnavailable
        | PushOutboxError::Domain(_)
        | PushOutboxError::Unavailable(_) => PushWakeStoreError::Unavailable,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushAuthorizationDecision {
    Authorized,
    Denied,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("current push authorization is unavailable")]
pub struct PushAuthorizationError;

#[async_trait]
pub trait PushWakeAuthorization: Send + Sync {
    async fn authorize(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
    ) -> Result<PushAuthorizationDecision, PushAuthorizationError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushDeliveryRequest {
    request_id: Uuid,
    lease_generation: PushLeaseGeneration,
    endpoint_generation: PushEndpointGeneration,
    capability_reference: PushCapabilityReference,
    expires_at_millis: u64,
    payload: PushWakePayload,
}

impl PushDeliveryRequest {
    fn from_wake(wake: &PushGatewayWake) -> Self {
        Self {
            request_id: wake.request_id,
            lease_generation: wake.lease_generation,
            endpoint_generation: wake.endpoint_generation,
            capability_reference: wake.capability_reference,
            expires_at_millis: wake.expires_at_millis,
            payload: PushWakePayload::Reconnect,
        }
    }

    #[cfg(test)]
    pub(crate) const fn for_test(
        request_id: Uuid,
        lease_generation: PushLeaseGeneration,
        endpoint_generation: PushEndpointGeneration,
        capability_reference: PushCapabilityReference,
        expires_at_millis: u64,
    ) -> Self {
        Self {
            request_id,
            lease_generation,
            endpoint_generation,
            capability_reference,
            expires_at_millis,
            payload: PushWakePayload::Reconnect,
        }
    }

    pub const fn request_id(&self) -> Uuid {
        self.request_id
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

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }

    pub const fn payload(&self) -> PushWakePayload {
        self.payload
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushProviderOutcome {
    Accepted,
    InvalidEndpoint {
        endpoint_generation: PushEndpointGeneration,
        invalid_at_millis: Option<u64>,
    },
    Retry {
        retry_after_millis: Option<u64>,
    },
    ConfigurationFault,
    PermanentRequestFault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("push provider is temporarily unavailable")]
pub struct PushProviderError;

#[async_trait]
pub trait PushProvider: Send + Sync {
    async fn deliver(
        &self,
        request: PushDeliveryRequest,
    ) -> Result<PushProviderOutcome, PushProviderError>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PushExecutionSummary {
    pub claimed: u32,
    pub delivered: u32,
    pub retried: u32,
    pub failed: u32,
    pub suppressed: u32,
    pub exhausted: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum PushGatewayExecutorError {
    #[error("push gateway received invalid claimed work")]
    InvalidWork,
    #[error("push gateway time is invalid")]
    InvalidTime,
    #[error(transparent)]
    Store(#[from] PushWakeStoreError),
}

pub trait PushGatewayClock: Send + Sync {
    fn now_millis(&self) -> Result<u64, PushGatewayExecutorError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemPushGatewayClock;

impl PushGatewayClock for SystemPushGatewayClock {
    fn now_millis(&self) -> Result<u64, PushGatewayExecutorError> {
        let duration = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| PushGatewayExecutorError::InvalidTime)?;
        u64::try_from(duration.as_millis()).map_err(|_| PushGatewayExecutorError::InvalidTime)
    }
}

pub struct PushGatewayExecutor<S, A, P, C = SystemPushGatewayClock> {
    store: S,
    authorization: A,
    provider: P,
    clock: C,
}

impl<S, A, P> PushGatewayExecutor<S, A, P, SystemPushGatewayClock> {
    pub const fn new(store: S, authorization: A, provider: P) -> Self {
        Self {
            store,
            authorization,
            provider,
            clock: SystemPushGatewayClock,
        }
    }
}

impl<S, A, P, C> PushGatewayExecutor<S, A, P, C>
where
    S: PushWakeStore,
    A: PushWakeAuthorization,
    P: PushProvider,
    C: PushGatewayClock,
{
    pub const fn with_clock(store: S, authorization: A, provider: P, clock: C) -> Self {
        Self {
            store,
            authorization,
            provider,
            clock,
        }
    }

    pub async fn run_once(
        &self,
        tenant: &TenantContext,
        claim_id: Uuid,
    ) -> Result<PushExecutionSummary, PushGatewayExecutorError> {
        let now_millis = self.clock.now_millis()?;
        if i64::try_from(now_millis).is_err() {
            return Err(PushGatewayExecutorError::InvalidTime);
        }
        let claim = PushGatewayClaim::new(claim_id, now_millis)?;
        let wakes = self.store.claim(tenant, claim).await?;
        let claimed =
            u32::try_from(wakes.len()).map_err(|_| PushGatewayExecutorError::InvalidWork)?;
        if claimed > PUSH_WAKE_BATCH_LIMIT {
            return Err(PushGatewayExecutorError::InvalidWork);
        }
        let mut summary = PushExecutionSummary {
            claimed,
            ..PushExecutionSummary::default()
        };
        for wake in wakes {
            if wake.community_id() != tenant.community_id()
                || wake.claim_id != claim_id
                || wake.claim_expires_at_millis != claim.claim_expires_at_millis
            {
                return Err(PushGatewayExecutorError::InvalidWork);
            }
            let now_millis = self.clock.now_millis()?;
            match self.execute_one(tenant, &wake, now_millis).await? {
                PushExecutionDisposition::Delivered => summary.delivered += 1,
                PushExecutionDisposition::Retried => summary.retried += 1,
                PushExecutionDisposition::Failed => summary.failed += 1,
                PushExecutionDisposition::Suppressed => summary.suppressed += 1,
                PushExecutionDisposition::Exhausted => summary.exhausted += 1,
            }
        }
        Ok(summary)
    }

    async fn execute_one(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        now_millis: u64,
    ) -> Result<PushExecutionDisposition, PushGatewayExecutorError> {
        if now_millis > wake.expires_at_millis {
            self.store
                .complete(tenant, wake, PushWakeTerminalOutcome::Expired, now_millis)
                .await?;
            return Ok(PushExecutionDisposition::Suppressed);
        }
        match self.authorization.authorize(tenant, wake).await {
            Ok(PushAuthorizationDecision::Authorized) => {}
            Ok(PushAuthorizationDecision::Denied) => {
                self.store
                    .complete(
                        tenant,
                        wake,
                        PushWakeTerminalOutcome::AuthorizationLost,
                        now_millis,
                    )
                    .await?;
                return Ok(PushExecutionDisposition::Suppressed);
            }
            Err(_) => return self.retry_or_exhaust(tenant, wake, now_millis, None).await,
        }
        if !self.store.revalidate(tenant, wake, now_millis).await? {
            self.store
                .complete(
                    tenant,
                    wake,
                    PushWakeTerminalOutcome::LeaseUnavailable,
                    now_millis,
                )
                .await?;
            return Ok(PushExecutionDisposition::Suppressed);
        }
        match self
            .provider
            .deliver(PushDeliveryRequest::from_wake(wake))
            .await
        {
            Ok(PushProviderOutcome::Accepted) => {
                self.store
                    .complete(tenant, wake, PushWakeTerminalOutcome::Accepted, now_millis)
                    .await?;
                Ok(PushExecutionDisposition::Delivered)
            }
            Ok(PushProviderOutcome::InvalidEndpoint {
                endpoint_generation,
                invalid_at_millis,
            }) if endpoint_generation == wake.endpoint_generation => {
                let disabled_at_millis = invalid_at_millis.unwrap_or(now_millis).min(now_millis);
                if self
                    .store
                    .disable_endpoint(tenant, wake, disabled_at_millis, now_millis)
                    .await?
                {
                    self.store
                        .complete(
                            tenant,
                            wake,
                            PushWakeTerminalOutcome::InvalidEndpoint,
                            now_millis,
                        )
                        .await?;
                    Ok(PushExecutionDisposition::Failed)
                } else {
                    self.store
                        .complete(
                            tenant,
                            wake,
                            PushWakeTerminalOutcome::LeaseUnavailable,
                            now_millis,
                        )
                        .await?;
                    Ok(PushExecutionDisposition::Suppressed)
                }
            }
            Ok(PushProviderOutcome::InvalidEndpoint { .. })
            | Ok(PushProviderOutcome::PermanentRequestFault) => {
                self.store
                    .complete(
                        tenant,
                        wake,
                        PushWakeTerminalOutcome::RetryExhausted,
                        now_millis,
                    )
                    .await?;
                Ok(PushExecutionDisposition::Failed)
            }
            Ok(PushProviderOutcome::Retry { retry_after_millis }) => {
                self.retry_or_exhaust(tenant, wake, now_millis, retry_after_millis)
                    .await
            }
            Ok(PushProviderOutcome::ConfigurationFault) | Err(_) => {
                self.retry_or_exhaust(tenant, wake, now_millis, None).await
            }
        }
    }

    async fn retry_or_exhaust(
        &self,
        tenant: &TenantContext,
        wake: &PushGatewayWake,
        now_millis: u64,
        retry_after_millis: Option<u64>,
    ) -> Result<PushExecutionDisposition, PushGatewayExecutorError> {
        if wake.attempt_count >= PUSH_MAX_ATTEMPTS {
            self.store
                .complete(
                    tenant,
                    wake,
                    PushWakeTerminalOutcome::RetryExhausted,
                    now_millis,
                )
                .await?;
            return Ok(PushExecutionDisposition::Exhausted);
        }
        let delay = retry_delay_millis(wake, retry_after_millis);
        let Some(available_at_millis) = now_millis.checked_add(delay) else {
            return Err(PushGatewayExecutorError::InvalidTime);
        };
        if available_at_millis > wake.expires_at_millis {
            self.store
                .complete(tenant, wake, PushWakeTerminalOutcome::Expired, now_millis)
                .await?;
            return Ok(PushExecutionDisposition::Suppressed);
        }
        self.store
            .retry(tenant, wake, available_at_millis, now_millis)
            .await?;
        Ok(PushExecutionDisposition::Retried)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PushExecutionDisposition {
    Delivered,
    Retried,
    Failed,
    Suppressed,
    Exhausted,
}

fn retry_delay_millis(wake: &PushGatewayWake, retry_after_millis: Option<u64>) -> u64 {
    let base = retry_after_millis
        .unwrap_or(PUSH_DEFAULT_RETRY_MILLIS)
        .clamp(PUSH_MIN_RETRY_MILLIS, PUSH_MAX_RETRY_MILLIS);
    let exponent = wake.attempt_count.saturating_sub(1).min(6);
    let exponential = base
        .checked_mul(1_u64 << exponent)
        .unwrap_or(PUSH_MAX_RETRY_MILLIS)
        .min(PUSH_MAX_RETRY_MILLIS);
    let jitter_window = exponential / 4;
    let jitter = if jitter_window == 0 {
        0
    } else {
        (wake.request_id.as_u128() % u128::from(jitter_window + 1)) as u64
    };
    exponential
        .saturating_add(jitter)
        .min(PUSH_MAX_RETRY_MILLIS)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex, MutexGuard},
    };

    use collaboration_domain::{PrincipalId, PushInstallationId, TrustedTenantRoute};

    use super::*;

    #[derive(Clone, Copy)]
    struct FixedClock(u64);

    impl PushGatewayClock for FixedClock {
        fn now_millis(&self) -> Result<u64, PushGatewayExecutorError> {
            Ok(self.0)
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum StoreAction {
        Retried(u64),
        Disabled(u64),
        Completed(PushWakeTerminalOutcome),
    }

    #[derive(Default)]
    struct StoreState {
        wakes: Vec<PushGatewayWake>,
        revalidations: VecDeque<Result<bool, PushWakeStoreError>>,
        disable_result: bool,
        actions: Vec<StoreAction>,
    }

    #[derive(Clone, Default)]
    struct FakeStore(Arc<Mutex<StoreState>>);

    impl FakeStore {
        fn state(&self) -> MutexGuard<'_, StoreState> {
            self.0.lock().expect("fake store lock")
        }
    }

    #[async_trait]
    impl PushWakeStore for FakeStore {
        async fn claim(
            &self,
            _tenant: &TenantContext,
            claim: PushGatewayClaim,
        ) -> Result<Vec<PushGatewayWake>, PushWakeStoreError> {
            let mut state = self.state();
            assert!(
                state
                    .wakes
                    .iter()
                    .all(|wake| wake.claim_id == claim.claim_id)
            );
            Ok(std::mem::take(&mut state.wakes))
        }

        async fn revalidate(
            &self,
            _tenant: &TenantContext,
            _wake: &PushGatewayWake,
            _now_millis: u64,
        ) -> Result<bool, PushWakeStoreError> {
            self.state().revalidations.pop_front().unwrap_or(Ok(true))
        }

        async fn retry(
            &self,
            _tenant: &TenantContext,
            _wake: &PushGatewayWake,
            available_at_millis: u64,
            _now_millis: u64,
        ) -> Result<(), PushWakeStoreError> {
            self.state()
                .actions
                .push(StoreAction::Retried(available_at_millis));
            Ok(())
        }

        async fn disable_endpoint(
            &self,
            _tenant: &TenantContext,
            _wake: &PushGatewayWake,
            disabled_at_millis: u64,
            _now_millis: u64,
        ) -> Result<bool, PushWakeStoreError> {
            let mut state = self.state();
            state
                .actions
                .push(StoreAction::Disabled(disabled_at_millis));
            Ok(state.disable_result)
        }

        async fn complete(
            &self,
            _tenant: &TenantContext,
            _wake: &PushGatewayWake,
            outcome: PushWakeTerminalOutcome,
            _completed_at_millis: u64,
        ) -> Result<(), PushWakeStoreError> {
            self.state().actions.push(StoreAction::Completed(outcome));
            Ok(())
        }
    }

    #[derive(Clone)]
    struct FakeAuthorization(
        Arc<Mutex<VecDeque<Result<PushAuthorizationDecision, PushAuthorizationError>>>>,
    );

    impl FakeAuthorization {
        fn authorized() -> Self {
            Self(Arc::new(Mutex::new(VecDeque::from([Ok(
                PushAuthorizationDecision::Authorized,
            )]))))
        }
    }

    #[async_trait]
    impl PushWakeAuthorization for FakeAuthorization {
        async fn authorize(
            &self,
            _tenant: &TenantContext,
            _wake: &PushGatewayWake,
        ) -> Result<PushAuthorizationDecision, PushAuthorizationError> {
            self.0
                .lock()
                .expect("authorization lock")
                .pop_front()
                .unwrap_or(Ok(PushAuthorizationDecision::Authorized))
        }
    }

    #[derive(Default)]
    struct ProviderState {
        outcomes: VecDeque<Result<PushProviderOutcome, PushProviderError>>,
        requests: Vec<PushDeliveryRequest>,
    }

    #[derive(Clone, Default)]
    struct FakeProvider(Arc<Mutex<ProviderState>>);

    #[async_trait]
    impl PushProvider for FakeProvider {
        async fn deliver(
            &self,
            request: PushDeliveryRequest,
        ) -> Result<PushProviderOutcome, PushProviderError> {
            let mut state = self.0.lock().expect("provider lock");
            state.requests.push(request);
            state
                .outcomes
                .pop_front()
                .unwrap_or(Ok(PushProviderOutcome::Accepted))
        }
    }

    fn tenant() -> TenantContext {
        let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "push-gateway-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn wake(attempt_count: u32, claim_id: Uuid) -> PushGatewayWake {
        PushGatewayWake::new(
            PushLeaseAddress {
                community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
                owner_principal_id: PrincipalId::from_uuid(Uuid::from_u128(2)),
                installation_id: PushInstallationId::new("private-installation-marker")
                    .expect("installation"),
            },
            Uuid::from_u128(10),
            Uuid::from_u128(11),
            PushLeaseGeneration::new(3).expect("generation"),
            PushEndpointGeneration::new(4).expect("endpoint generation"),
            PushCapabilityReference::from_digest([5_u8; 32]).expect("capability"),
            4_000_000,
            attempt_count,
            claim_id,
            31_000,
        )
        .expect("wake")
    }

    fn executor(
        attempt_count: u32,
        outcome: Result<PushProviderOutcome, PushProviderError>,
    ) -> (
        PushGatewayExecutor<FakeStore, FakeAuthorization, FakeProvider, FixedClock>,
        FakeStore,
        FakeProvider,
        Uuid,
    ) {
        let claim_id = Uuid::from_u128(20);
        let store = FakeStore::default();
        store.state().wakes.push(wake(attempt_count, claim_id));
        let provider = FakeProvider::default();
        provider
            .0
            .lock()
            .expect("provider lock")
            .outcomes
            .push_back(outcome);
        (
            PushGatewayExecutor::with_clock(
                store.clone(),
                FakeAuthorization::authorized(),
                provider.clone(),
                FixedClock(1_000),
            ),
            store,
            provider,
            claim_id,
        )
    }

    #[tokio::test]
    async fn transient_provider_failure_releases_a_bounded_jittered_retry() {
        let (executor, store, _, claim_id) = executor(
            1,
            Ok(PushProviderOutcome::Retry {
                retry_after_millis: Some(2_000),
            }),
        );

        let summary = executor
            .run_once(&tenant(), claim_id)
            .await
            .expect("transient retry");

        assert_eq!(summary.retried, 1);
        let StoreAction::Retried(available_at_millis) = store.state().actions[0] else {
            panic!("expected retry action")
        };
        assert!((3_000..=3_500).contains(&available_at_millis));
    }

    #[tokio::test]
    async fn matching_permanent_endpoint_failure_disables_only_that_generation() {
        let (executor, store, _, claim_id) = executor(
            2,
            Ok(PushProviderOutcome::InvalidEndpoint {
                endpoint_generation: PushEndpointGeneration::new(4).expect("endpoint generation"),
                invalid_at_millis: Some(900),
            }),
        );
        store.state().disable_result = true;

        let summary = executor
            .run_once(&tenant(), claim_id)
            .await
            .expect("permanent failure");

        assert_eq!(summary.failed, 1);
        assert_eq!(
            store.state().actions,
            vec![
                StoreAction::Disabled(900),
                StoreAction::Completed(PushWakeTerminalOutcome::InvalidEndpoint),
            ]
        );
    }

    #[tokio::test]
    async fn stale_permanent_endpoint_failure_never_disables_the_current_generation() {
        let (executor, store, _, claim_id) = executor(
            2,
            Ok(PushProviderOutcome::InvalidEndpoint {
                endpoint_generation: PushEndpointGeneration::new(5)
                    .expect("stale endpoint generation"),
                invalid_at_millis: Some(900),
            }),
        );
        store.state().disable_result = true;

        let summary = executor
            .run_once(&tenant(), claim_id)
            .await
            .expect("stale permanent failure");

        assert_eq!(summary.failed, 1);
        assert_eq!(
            store.state().actions,
            vec![StoreAction::Completed(
                PushWakeTerminalOutcome::RetryExhausted
            )]
        );
    }

    #[tokio::test]
    async fn revocation_race_suppresses_before_provider_send() {
        let (executor, store, provider, claim_id) = executor(1, Ok(PushProviderOutcome::Accepted));
        store.state().revalidations.push_back(Ok(false));

        let summary = executor
            .run_once(&tenant(), claim_id)
            .await
            .expect("revoked wake");

        assert_eq!(summary.suppressed, 1);
        assert!(
            provider
                .0
                .lock()
                .expect("provider lock")
                .requests
                .is_empty()
        );
        assert_eq!(
            store.state().actions,
            vec![StoreAction::Completed(
                PushWakeTerminalOutcome::LeaseUnavailable
            )]
        );
    }

    #[tokio::test]
    async fn provider_boundary_exposes_only_fixed_reconnect_and_redacted_authority() {
        let (executor, _, provider, claim_id) = executor(1, Ok(PushProviderOutcome::Accepted));

        let summary = executor
            .run_once(&tenant(), claim_id)
            .await
            .expect("accepted wake");

        assert_eq!(summary.delivered, 1);
        let provider = provider.0.lock().expect("provider lock");
        let request = provider.requests.first().expect("provider request");
        assert_eq!(request.payload(), PushWakePayload::Reconnect);
        let diagnostics = format!("{request:?}");
        for private_marker in [
            "private-installation-marker",
            "event_content",
            "source_event_id",
            "ciphertext",
            "title",
            "body",
            "url",
        ] {
            assert!(!diagnostics.contains(private_marker), "{diagnostics}");
        }
    }

    #[tokio::test]
    async fn retry_exhaustion_becomes_terminal_without_another_queue_release() {
        let (executor, store, _, claim_id) = executor(8, Err(PushProviderError));

        let summary = executor
            .run_once(&tenant(), claim_id)
            .await
            .expect("retry exhaustion");

        assert_eq!(summary.exhausted, 1);
        assert_eq!(
            store.state().actions,
            vec![StoreAction::Completed(
                PushWakeTerminalOutcome::RetryExhausted
            )]
        );
    }
}
