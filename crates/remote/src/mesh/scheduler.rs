use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap, VecDeque},
};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateVersion, CommunityId, JobIdentity, PrincipalId, TenantContext,
};
use iroh_base::EndpointId;
use thiserror::Error;
use uuid::Uuid;

use super::advertisement::{
    AdvertisedResourceLimits, ComputeAdvertisementRegistry, ComputeCapability, MAX_MODEL_ID_BYTES,
    MeshDeviceId, ModelArtifactDigest, ProviderTrustClass, ValidatedComputeAdvertisement,
};

pub const MAX_COMMUNITY_QUEUE_DEPTH: usize = 1024;
pub const MAX_REQUESTER_QUEUE_DEPTH: usize = 64;
pub const MAX_REQUESTER_CONCURRENCY: u16 = 64;
pub const MAX_REQUESTER_WEIGHT: u16 = 16;
pub const MAX_POLICY_REQUESTERS: usize = 1024;
pub const MAX_OWNER_RESERVED_SLOTS: u16 = 8;
pub const MAX_MESH_LEASE_LIFETIME_MILLIS: u64 = 60_000;
pub const MAX_CANCELLATION_GRACE_MILLIS: u64 = 30_000;
pub const FAIRNESS_AGING_INTERVAL_MILLIS: u64 = 30_000;
pub const MAX_FAIRNESS_AGING_STEPS: u64 = 16;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TransferableContextClass {
    Public,
    Community,
    ExplicitPrivate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshResourceRequest {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub accelerator_memory_bytes: u64,
    pub model_cache_bytes: u64,
    pub network_bytes_per_second: u64,
    pub context_bytes: u64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub wall_clock_millis: u64,
    pub idle_millis: u64,
}

impl MeshResourceRequest {
    fn validate(self) -> Result<(), MeshSchedulerError> {
        if self.cpu_millicores == 0
            || self.memory_bytes == 0
            || self.accelerator_memory_bytes == 0
            || self.model_cache_bytes == 0
            || self.network_bytes_per_second == 0
            || self.context_bytes == 0
            || self.prompt_tokens == 0
            || self.output_tokens == 0
            || self.wall_clock_millis == 0
            || self.idle_millis == 0
        {
            return Err(MeshSchedulerError::InvalidRequest);
        }
        Ok(())
    }

    fn is_within(self, limits: AdvertisedResourceLimits) -> bool {
        self.cpu_millicores <= limits.cpu_millicores
            && self.memory_bytes <= limits.memory_bytes
            && self.accelerator_memory_bytes <= limits.accelerator_memory_bytes
            && self.model_cache_bytes <= limits.model_cache_bytes
            && self.network_bytes_per_second <= limits.network_bytes_per_second
            && self.context_bytes <= limits.maximum_context_bytes
            && self.prompt_tokens <= limits.maximum_prompt_tokens
            && self.output_tokens <= limits.maximum_output_tokens
            && self.wall_clock_millis <= limits.maximum_wall_clock_millis
            && self.idle_millis <= limits.maximum_idle_millis
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshScheduleRequest {
    job_identity: JobIdentity,
    job_version: AggregateVersion,
    attempt_id: Uuid,
    requester_principal_id: PrincipalId,
    executor_principal_id: PrincipalId,
    model_id: String,
    artifact_digest: ModelArtifactDigest,
    capability: ComputeCapability,
    context_class: TransferableContextClass,
    resources: MeshResourceRequest,
    lease_id: Uuid,
    lease_nonce: [u8; 16],
    enqueued_at_millis: u64,
    lease_expires_at_millis: u64,
    cancellation_grace_millis: u64,
}

impl MeshScheduleRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        job_identity: JobIdentity,
        job_version: AggregateVersion,
        attempt_id: Uuid,
        requester_principal_id: PrincipalId,
        executor_principal_id: PrincipalId,
        model_id: impl Into<String>,
        artifact_digest: ModelArtifactDigest,
        capability: ComputeCapability,
        context_class: TransferableContextClass,
        resources: MeshResourceRequest,
        lease_id: Uuid,
        lease_nonce: [u8; 16],
        enqueued_at_millis: u64,
        lease_expires_at_millis: u64,
        cancellation_grace_millis: u64,
    ) -> Result<Self, MeshSchedulerError> {
        let model_id = model_id.into();
        resources.validate()?;
        let lifetime = lease_expires_at_millis
            .checked_sub(enqueued_at_millis)
            .ok_or(MeshSchedulerError::InvalidRequest)?;
        if attempt_id.is_nil()
            || requester_principal_id.as_uuid().is_nil()
            || executor_principal_id.as_uuid().is_nil()
            || lease_id.is_nil()
            || lease_nonce == [0; 16]
            || model_id.is_empty()
            || model_id.len() > MAX_MODEL_ID_BYTES
            || model_id.trim() != model_id
            || model_id.chars().any(char::is_control)
            || lifetime == 0
            || lifetime > MAX_MESH_LEASE_LIFETIME_MILLIS
            || cancellation_grace_millis == 0
            || cancellation_grace_millis > MAX_CANCELLATION_GRACE_MILLIS
        {
            return Err(MeshSchedulerError::InvalidRequest);
        }
        Ok(Self {
            job_identity,
            job_version,
            attempt_id,
            requester_principal_id,
            executor_principal_id,
            model_id,
            artifact_digest,
            capability,
            context_class,
            resources,
            lease_id,
            lease_nonce,
            enqueued_at_millis,
            lease_expires_at_millis,
            cancellation_grace_millis,
        })
    }

    pub const fn job_identity(&self) -> JobIdentity {
        self.job_identity
    }

    pub const fn attempt_id(&self) -> Uuid {
        self.attempt_id
    }

    pub const fn requester_principal_id(&self) -> PrincipalId {
        self.requester_principal_id
    }
}

#[derive(Clone, Debug)]
pub struct MeshFairnessPolicy {
    version: AggregateVersion,
    audit_actor_principal_id: PrincipalId,
    requester_weights: HashMap<PrincipalId, u16>,
    maximum_requester_concurrency: u16,
    maximum_requester_queue_depth: usize,
    maximum_community_queue_depth: usize,
    owner_reserved_slots: u16,
    approved_trust_classes: BTreeSet<ProviderTrustClass>,
    approved_context_classes: BTreeSet<TransferableContextClass>,
}

impl MeshFairnessPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        version: AggregateVersion,
        audit_actor_principal_id: PrincipalId,
        requester_weights: HashMap<PrincipalId, u16>,
        maximum_requester_concurrency: u16,
        maximum_requester_queue_depth: usize,
        maximum_community_queue_depth: usize,
        owner_reserved_slots: u16,
        approved_trust_classes: BTreeSet<ProviderTrustClass>,
        approved_context_classes: BTreeSet<TransferableContextClass>,
    ) -> Result<Self, MeshSchedulerError> {
        if audit_actor_principal_id.as_uuid().is_nil()
            || requester_weights.len() > MAX_POLICY_REQUESTERS
            || requester_weights
                .keys()
                .any(|requester| requester.as_uuid().is_nil())
            || requester_weights
                .values()
                .any(|weight| *weight == 0 || *weight > MAX_REQUESTER_WEIGHT)
            || maximum_requester_concurrency == 0
            || maximum_requester_concurrency > MAX_REQUESTER_CONCURRENCY
            || maximum_requester_queue_depth == 0
            || maximum_requester_queue_depth > MAX_REQUESTER_QUEUE_DEPTH
            || maximum_community_queue_depth == 0
            || maximum_community_queue_depth > MAX_COMMUNITY_QUEUE_DEPTH
            || owner_reserved_slots > MAX_OWNER_RESERVED_SLOTS
            || approved_trust_classes.is_empty()
            || approved_trust_classes.contains(&ProviderTrustClass::ThirdParty)
            || approved_context_classes.is_empty()
        {
            return Err(MeshSchedulerError::InvalidPolicy);
        }
        Ok(Self {
            version,
            audit_actor_principal_id,
            requester_weights,
            maximum_requester_concurrency,
            maximum_requester_queue_depth,
            maximum_community_queue_depth,
            owner_reserved_slots,
            approved_trust_classes,
            approved_context_classes,
        })
    }

    fn weight(&self, requester_principal_id: PrincipalId) -> u16 {
        self.requester_weights
            .get(&requester_principal_id)
            .copied()
            .unwrap_or(1)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MeshProviderSelection {
    pub community_id: CommunityId,
    pub owner_principal_id: PrincipalId,
    pub device_id: MeshDeviceId,
    pub endpoint_id: EndpointId,
    pub trust_class: ProviderTrustClass,
    pub membership_version: AggregateVersion,
    pub runtime_generation: u64,
    pub sharing_generation: u64,
    pub advertisement_record_version: u64,
    pub advertisement_expires_at_millis: u64,
    pub model_id: String,
    pub artifact_digest: ModelArtifactDigest,
}

impl MeshProviderSelection {
    fn from_advertisement(
        advertisement: &ValidatedComputeAdvertisement,
        request: &MeshScheduleRequest,
        approved_trust_classes: &BTreeSet<ProviderTrustClass>,
        now_millis: u64,
    ) -> Result<Self, MeshCandidateIneligibility> {
        let advertisement = advertisement.advertisement();
        if advertisement.expires_at_millis <= now_millis
            || advertisement.expires_at_millis < request.lease_expires_at_millis
        {
            return Err(MeshCandidateIneligibility::Stale);
        }
        if !approved_trust_classes.contains(&advertisement.trust_class)
            || advertisement.trust_class == ProviderTrustClass::ThirdParty
        {
            return Err(MeshCandidateIneligibility::Trust);
        }
        let resources = advertisement
            .resources
            .ok_or(MeshCandidateIneligibility::ResourcePolicy)?;
        if !request.resources.is_within(resources) {
            return Err(MeshCandidateIneligibility::ResourcePolicy);
        }
        let model = advertisement
            .models
            .iter()
            .find(|model| model.model_id == request.model_id)
            .ok_or(MeshCandidateIneligibility::Model)?;
        if model.artifact_digest != request.artifact_digest {
            return Err(MeshCandidateIneligibility::Model);
        }
        if !model.capabilities.contains(&request.capability) {
            return Err(MeshCandidateIneligibility::Capability);
        }
        Ok(Self {
            community_id: advertisement.community_id,
            owner_principal_id: advertisement.owner_principal_id,
            device_id: advertisement.device_id,
            endpoint_id: advertisement.endpoint_id,
            trust_class: advertisement.trust_class,
            membership_version: advertisement.membership_version,
            runtime_generation: advertisement.runtime_generation,
            sharing_generation: advertisement.sharing_generation,
            advertisement_record_version: advertisement.record_version,
            advertisement_expires_at_millis: advertisement.expires_at_millis,
            model_id: model.model_id.clone(),
            artifact_digest: model.artifact_digest,
        })
    }

    fn sort_key(&self) -> (ProviderTrustClass, PrincipalId, MeshDeviceId, EndpointId) {
        (
            self.trust_class,
            self.owner_principal_id,
            self.device_id,
            self.endpoint_id,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurrentMeshCapacity {
    pub membership_version: AggregateVersion,
    pub runtime_generation: u64,
    pub sharing_generation: u64,
    pub advertisement_record_version: u64,
    pub installed_artifact_digest: ModelArtifactDigest,
    pub enforced_resource_ceiling: AdvertisedResourceLimits,
    pub available_concurrent_slots: u16,
    pub device_queued_requests: u16,
    pub requester_active_leases: u16,
}

impl CurrentMeshCapacity {
    fn admits(
        self,
        request: &MeshScheduleRequest,
        selection: &MeshProviderSelection,
        policy: &MeshFairnessPolicy,
    ) -> Result<(), MeshCandidateIneligibility> {
        if self.membership_version != selection.membership_version
            || self.runtime_generation != selection.runtime_generation
            || self.sharing_generation != selection.sharing_generation
            || self.advertisement_record_version != selection.advertisement_record_version
        {
            return Err(MeshCandidateIneligibility::Stale);
        }
        if self.installed_artifact_digest != request.artifact_digest {
            return Err(MeshCandidateIneligibility::Model);
        }
        if !request.resources.is_within(self.enforced_resource_ceiling) {
            return Err(MeshCandidateIneligibility::ResourcePolicy);
        }
        if self.requester_active_leases >= policy.maximum_requester_concurrency {
            return Err(MeshCandidateIneligibility::RequesterConcurrency);
        }
        if self.available_concurrent_slots == 0
            || self.device_queued_requests >= self.enforced_resource_ceiling.maximum_queued_requests
            || (request.requester_principal_id != selection.owner_principal_id
                && self.available_concurrent_slots <= policy.owner_reserved_slots)
        {
            return Err(MeshCandidateIneligibility::NoCapacity);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshCandidateIneligibility {
    Consent,
    Membership,
    RequesterAuthorization,
    Delegation,
    Trust,
    Model,
    Capability,
    Context,
    Revoked,
    Draining,
    Quarantined,
    Stale,
    Sandbox,
    ResourcePolicy,
    RequesterConcurrency,
    NoCapacity,
    JobNotExecutable,
}

impl MeshCandidateIneligibility {
    const fn is_capacity(self) -> bool {
        matches!(self, Self::RequesterConcurrency | Self::NoCapacity)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshCandidateEligibility {
    Eligible(CurrentMeshCapacity),
    Ineligible(MeshCandidateIneligibility),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshExecutorResourceLeaseRequest {
    pub job_identity: JobIdentity,
    pub job_version: AggregateVersion,
    pub attempt_id: Uuid,
    pub lease_id: Uuid,
    pub lease_nonce: [u8; 16],
    pub requester_principal_id: PrincipalId,
    pub executor_principal_id: PrincipalId,
    pub provider: MeshProviderSelection,
    pub capability: ComputeCapability,
    pub context_class: TransferableContextClass,
    pub resources: MeshResourceRequest,
    pub policy_version: AggregateVersion,
    pub policy_audit_actor_principal_id: PrincipalId,
    pub acquired_at_millis: u64,
    pub expires_at_millis: u64,
    pub cancellation_grace_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshExecutorResourceLease {
    request: MeshExecutorResourceLeaseRequest,
    executor_generation: AggregateVersion,
}

impl MeshExecutorResourceLease {
    pub fn new(
        request: MeshExecutorResourceLeaseRequest,
        executor_generation: AggregateVersion,
    ) -> Result<Self, MeshSchedulerError> {
        validate_lease_request(&request)?;
        Ok(Self {
            request,
            executor_generation,
        })
    }

    pub const fn request(&self) -> &MeshExecutorResourceLeaseRequest {
        &self.request
    }

    pub const fn executor_generation(&self) -> AggregateVersion {
        self.executor_generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshLeaseReleaseReason {
    Completed,
    Failed,
    Cancelled,
    Expired,
    Revoked,
    Drained,
    NodeLost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshLeaseMutationOutcome {
    Applied,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MeshSchedulingAuthorityError {
    #[error("mesh scheduling authority is unavailable")]
    Unavailable,
    #[error("mesh scheduling authority rejected a conflicting transition")]
    Conflict,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MeshLeaseAcquireError {
    #[error("the canonical job already has an executor lease")]
    JobAlreadyLeased,
    #[error("mesh capacity changed before lease acquisition")]
    CapacityChanged,
    #[error("mesh candidate became stale before lease acquisition")]
    StaleCandidate,
    #[error("mesh policy changed before lease acquisition")]
    PolicyChanged,
    #[error("mesh lease authority is unavailable")]
    AuthorityUnavailable,
}

#[async_trait(?Send)]
pub trait MeshSchedulingAuthority {
    async fn evaluate_candidate(
        &self,
        tenant: &TenantContext,
        request: &MeshScheduleRequest,
        provider: &MeshProviderSelection,
        policy_version: AggregateVersion,
        policy_audit_actor_principal_id: PrincipalId,
        now_millis: u64,
    ) -> Result<MeshCandidateEligibility, MeshSchedulingAuthorityError>;

    async fn acquire_executor_resource_lease(
        &self,
        tenant: &TenantContext,
        request: MeshExecutorResourceLeaseRequest,
    ) -> Result<MeshExecutorResourceLease, MeshLeaseAcquireError>;

    async fn release_executor_resource_lease(
        &self,
        tenant: &TenantContext,
        lease: &MeshExecutorResourceLease,
        released_at_millis: u64,
        reason: MeshLeaseReleaseReason,
    ) -> Result<MeshLeaseMutationOutcome, MeshSchedulingAuthorityError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshQueueReason {
    Fairness,
    RequesterConcurrency,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshScheduleOutcome {
    Idle,
    Acquired(MeshExecutorResourceLease),
    Queued {
        reason: MeshQueueReason,
    },
    NoCapacity {
        provider: Option<MeshProviderSelection>,
    },
    PolicyDenied {
        reason: MeshCandidateIneligibility,
        provider: Option<MeshProviderSelection>,
    },
    ProviderUnavailable {
        provider: MeshProviderSelection,
    },
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MeshSchedulerError {
    #[error("mesh scheduling request is invalid")]
    InvalidRequest,
    #[error("mesh fairness policy is invalid")]
    InvalidPolicy,
    #[error("mesh scheduling request crossed its typed community boundary")]
    TenantBoundaryViolation,
    #[error("mesh scheduler queue is full")]
    QueueFull,
    #[error("mesh scheduling attempt is already queued or leased")]
    DuplicateAttempt,
    #[error("mesh scheduling authority is unavailable")]
    AuthorityUnavailable,
    #[error("mesh scheduling authority returned a mismatched lease")]
    LeaseMismatch,
    #[error("mesh executor/resource lease is not active")]
    LeaseNotActive,
}

#[derive(Clone, Debug)]
struct QueuedMeshRequest {
    request: MeshScheduleRequest,
    sequence: u64,
    selected_provider: Option<MeshProviderSelection>,
}

#[derive(Clone, Debug, Default)]
struct RequesterQueue {
    requests: VecDeque<QueuedMeshRequest>,
    completed_service: u64,
    active_leases: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AttemptKey {
    job_identity: JobIdentity,
    attempt_id: Uuid,
}

#[derive(Clone, Debug)]
struct ActiveLease {
    requester_principal_id: PrincipalId,
    lease: MeshExecutorResourceLease,
}

pub struct MeshScheduler {
    community_id: CommunityId,
    policy: MeshFairnessPolicy,
    requester_queues: HashMap<PrincipalId, RequesterQueue>,
    attempts: HashMap<AttemptKey, Option<MeshProviderSelection>>,
    active_leases: HashMap<Uuid, ActiveLease>,
    next_sequence: u64,
    queued_count: usize,
}

impl MeshScheduler {
    pub fn new(
        community_id: CommunityId,
        policy: MeshFairnessPolicy,
    ) -> Result<Self, MeshSchedulerError> {
        if community_id.as_uuid().is_nil() {
            return Err(MeshSchedulerError::InvalidPolicy);
        }
        Ok(Self {
            community_id,
            policy,
            requester_queues: HashMap::new(),
            attempts: HashMap::new(),
            active_leases: HashMap::new(),
            next_sequence: 0,
            queued_count: 0,
        })
    }

    pub fn enqueue(&mut self, request: MeshScheduleRequest) -> Result<(), MeshSchedulerError> {
        if request.job_identity.community_id() != self.community_id {
            return Err(MeshSchedulerError::TenantBoundaryViolation);
        }
        if request.enqueued_at_millis >= request.lease_expires_at_millis {
            return Err(MeshSchedulerError::InvalidRequest);
        }
        let attempt = AttemptKey {
            job_identity: request.job_identity,
            attempt_id: request.attempt_id,
        };
        if self.attempts.contains_key(&attempt) {
            return Err(MeshSchedulerError::DuplicateAttempt);
        }
        if self.queued_count >= self.policy.maximum_community_queue_depth {
            return Err(MeshSchedulerError::QueueFull);
        }
        let requester_queue = self
            .requester_queues
            .entry(request.requester_principal_id)
            .or_default();
        if requester_queue.requests.len() >= self.policy.maximum_requester_queue_depth {
            return Err(MeshSchedulerError::QueueFull);
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(MeshSchedulerError::QueueFull)?;
        requester_queue.requests.push_back(QueuedMeshRequest {
            request,
            sequence,
            selected_provider: None,
        });
        self.attempts.insert(attempt, None);
        self.queued_count += 1;
        Ok(())
    }

    pub const fn queued_count(&self) -> usize {
        self.queued_count
    }

    pub async fn schedule_next<A: MeshSchedulingAuthority>(
        &mut self,
        tenant: &TenantContext,
        advertisements: &mut ComputeAdvertisementRegistry,
        authority: &A,
        now_millis: u64,
    ) -> Result<MeshScheduleOutcome, MeshSchedulerError> {
        if tenant.community_id() != self.community_id {
            return Err(MeshSchedulerError::TenantBoundaryViolation);
        }
        if self.queued_count == 0 {
            return Ok(MeshScheduleOutcome::Idle);
        }
        let Some(requester_principal_id) = self.next_requester(now_millis) else {
            return Ok(MeshScheduleOutcome::Queued {
                reason: MeshQueueReason::RequesterConcurrency,
            });
        };
        let queued = self
            .requester_queues
            .get(&requester_principal_id)
            .and_then(|queue| queue.requests.front())
            .cloned()
            .ok_or(MeshSchedulerError::InvalidRequest)?;

        if now_millis >= queued.request.lease_expires_at_millis {
            self.remove_front(requester_principal_id, false)?;
            return Ok(MeshScheduleOutcome::PolicyDenied {
                reason: MeshCandidateIneligibility::Stale,
                provider: queued.selected_provider,
            });
        }
        if now_millis < queued.request.enqueued_at_millis {
            self.remove_front(requester_principal_id, false)?;
            return Ok(MeshScheduleOutcome::PolicyDenied {
                reason: MeshCandidateIneligibility::Stale,
                provider: queued.selected_provider,
            });
        }

        if !self
            .policy
            .approved_context_classes
            .contains(&queued.request.context_class)
        {
            self.remove_front(requester_principal_id, false)?;
            return Ok(MeshScheduleOutcome::PolicyDenied {
                reason: MeshCandidateIneligibility::Context,
                provider: queued.selected_provider,
            });
        }

        let active_hints = advertisements.active_hints(now_millis);
        let mut candidates = Vec::new();
        let mut static_denial = None;
        for advertisement in active_hints {
            if advertisement.advertisement().community_id != self.community_id {
                continue;
            }
            match MeshProviderSelection::from_advertisement(
                advertisement,
                &queued.request,
                &self.policy.approved_trust_classes,
                now_millis,
            ) {
                Ok(candidate) => candidates.push(candidate),
                Err(reason) => {
                    static_denial.get_or_insert(reason);
                }
            }
        }
        candidates.sort_by_key(MeshProviderSelection::sort_key);

        let (selection, capacity) = if let Some(selected) = &queued.selected_provider {
            let Some(selection) = candidates
                .into_iter()
                .find(|candidate| candidate == selected)
            else {
                self.remove_front(requester_principal_id, false)?;
                return Ok(MeshScheduleOutcome::ProviderUnavailable {
                    provider: selected.clone(),
                });
            };
            match self
                .evaluate(authority, tenant, &queued.request, &selection, now_millis)
                .await?
            {
                Ok(capacity) => (selection, capacity),
                Err(reason) if reason.is_capacity() => {
                    return Ok(MeshScheduleOutcome::NoCapacity {
                        provider: Some(selection),
                    });
                }
                Err(reason) => {
                    self.remove_front(requester_principal_id, false)?;
                    return Ok(MeshScheduleOutcome::PolicyDenied {
                        reason,
                        provider: Some(selection),
                    });
                }
            }
        } else {
            let mut first_denial = None;
            let mut saw_capacity_denial = false;
            let mut eligible = None;
            for selection in candidates {
                match self
                    .evaluate(authority, tenant, &queued.request, &selection, now_millis)
                    .await?
                {
                    Ok(capacity) => {
                        eligible = Some((selection, capacity));
                        break;
                    }
                    Err(reason) if reason.is_capacity() => saw_capacity_denial = true,
                    Err(reason) => {
                        first_denial.get_or_insert(reason);
                    }
                }
            }
            let Some((selection, capacity)) = eligible else {
                let denial = first_denial.or(static_denial);
                if saw_capacity_denial || denial.is_none() {
                    return Ok(MeshScheduleOutcome::NoCapacity { provider: None });
                }
                self.remove_front(requester_principal_id, false)?;
                return Ok(MeshScheduleOutcome::PolicyDenied {
                    reason: denial.unwrap_or(MeshCandidateIneligibility::Trust),
                    provider: None,
                });
            };
            self.bind_front_provider(requester_principal_id, selection.clone())?;
            (selection, capacity)
        };

        capacity
            .admits(&queued.request, &selection, &self.policy)
            .map_err(|reason| match reason {
                MeshCandidateIneligibility::NoCapacity
                | MeshCandidateIneligibility::RequesterConcurrency => {
                    MeshSchedulerError::AuthorityUnavailable
                }
                _ => MeshSchedulerError::LeaseMismatch,
            })?;
        let lease_request = MeshExecutorResourceLeaseRequest {
            job_identity: queued.request.job_identity,
            job_version: queued.request.job_version,
            attempt_id: queued.request.attempt_id,
            lease_id: queued.request.lease_id,
            lease_nonce: queued.request.lease_nonce,
            requester_principal_id: queued.request.requester_principal_id,
            executor_principal_id: queued.request.executor_principal_id,
            provider: selection.clone(),
            capability: queued.request.capability,
            context_class: queued.request.context_class,
            resources: queued.request.resources,
            policy_version: self.policy.version,
            policy_audit_actor_principal_id: self.policy.audit_actor_principal_id,
            acquired_at_millis: now_millis,
            expires_at_millis: queued.request.lease_expires_at_millis,
            cancellation_grace_millis: queued.request.cancellation_grace_millis,
        };
        let lease = match authority
            .acquire_executor_resource_lease(tenant, lease_request.clone())
            .await
        {
            Ok(lease) => lease,
            Err(MeshLeaseAcquireError::CapacityChanged) => {
                return Ok(MeshScheduleOutcome::NoCapacity {
                    provider: Some(selection),
                });
            }
            Err(MeshLeaseAcquireError::StaleCandidate) => {
                self.remove_front(requester_principal_id, false)?;
                return Ok(MeshScheduleOutcome::ProviderUnavailable {
                    provider: selection,
                });
            }
            Err(MeshLeaseAcquireError::PolicyChanged) => {
                self.remove_front(requester_principal_id, false)?;
                return Ok(MeshScheduleOutcome::PolicyDenied {
                    reason: MeshCandidateIneligibility::Consent,
                    provider: Some(selection),
                });
            }
            Err(MeshLeaseAcquireError::JobAlreadyLeased) => {
                self.remove_front(requester_principal_id, false)?;
                return Ok(MeshScheduleOutcome::PolicyDenied {
                    reason: MeshCandidateIneligibility::JobNotExecutable,
                    provider: Some(selection),
                });
            }
            Err(MeshLeaseAcquireError::AuthorityUnavailable) => {
                return Err(MeshSchedulerError::AuthorityUnavailable);
            }
        };
        if lease.request != lease_request {
            return Err(MeshSchedulerError::LeaseMismatch);
        }
        self.remove_front(requester_principal_id, true)?;
        self.active_leases.insert(
            lease.request.lease_id,
            ActiveLease {
                requester_principal_id,
                lease: lease.clone(),
            },
        );
        Ok(MeshScheduleOutcome::Acquired(lease))
    }

    pub async fn release<A: MeshSchedulingAuthority>(
        &mut self,
        tenant: &TenantContext,
        authority: &A,
        lease_id: Uuid,
        released_at_millis: u64,
        reason: MeshLeaseReleaseReason,
    ) -> Result<MeshLeaseMutationOutcome, MeshSchedulerError> {
        if tenant.community_id() != self.community_id {
            return Err(MeshSchedulerError::TenantBoundaryViolation);
        }
        let active = self
            .active_leases
            .get(&lease_id)
            .ok_or(MeshSchedulerError::LeaseNotActive)?
            .clone();
        let outcome = authority
            .release_executor_resource_lease(tenant, &active.lease, released_at_millis, reason)
            .await
            .map_err(|_| MeshSchedulerError::AuthorityUnavailable)?;
        self.active_leases.remove(&lease_id);
        if let Some(queue) = self
            .requester_queues
            .get_mut(&active.requester_principal_id)
        {
            queue.active_leases = queue.active_leases.saturating_sub(1);
        }
        Ok(outcome)
    }

    async fn evaluate<A: MeshSchedulingAuthority>(
        &self,
        authority: &A,
        tenant: &TenantContext,
        request: &MeshScheduleRequest,
        selection: &MeshProviderSelection,
        now_millis: u64,
    ) -> Result<Result<CurrentMeshCapacity, MeshCandidateIneligibility>, MeshSchedulerError> {
        let eligibility = authority
            .evaluate_candidate(
                tenant,
                request,
                selection,
                self.policy.version,
                self.policy.audit_actor_principal_id,
                now_millis,
            )
            .await
            .map_err(|_| MeshSchedulerError::AuthorityUnavailable)?;
        match eligibility {
            MeshCandidateEligibility::Eligible(capacity) => Ok(capacity
                .admits(request, selection, &self.policy)
                .map(|()| capacity)),
            MeshCandidateEligibility::Ineligible(reason) => Ok(Err(reason)),
        }
    }

    fn next_requester(&self, now_millis: u64) -> Option<PrincipalId> {
        self.requester_queues
            .iter()
            .filter_map(|(requester, queue)| {
                let front = queue.requests.front()?;
                if queue.active_leases >= self.policy.maximum_requester_concurrency {
                    return None;
                }
                Some((*requester, queue, front))
            })
            .min_by(
                |(left_id, left_queue, left), (right_id, right_queue, right)| {
                    compare_fairness(
                        left_queue.completed_service,
                        self.policy.weight(*left_id),
                        left.request.enqueued_at_millis,
                        left.sequence,
                        right_queue.completed_service,
                        self.policy.weight(*right_id),
                        right.request.enqueued_at_millis,
                        right.sequence,
                        now_millis,
                    )
                },
            )
            .map(|(requester, _, _)| requester)
    }

    fn bind_front_provider(
        &mut self,
        requester_principal_id: PrincipalId,
        provider: MeshProviderSelection,
    ) -> Result<(), MeshSchedulerError> {
        let front = self
            .requester_queues
            .get_mut(&requester_principal_id)
            .and_then(|queue| queue.requests.front_mut())
            .ok_or(MeshSchedulerError::InvalidRequest)?;
        front.selected_provider = Some(provider.clone());
        let attempt = AttemptKey {
            job_identity: front.request.job_identity,
            attempt_id: front.request.attempt_id,
        };
        self.attempts.insert(attempt, Some(provider));
        Ok(())
    }

    fn remove_front(
        &mut self,
        requester_principal_id: PrincipalId,
        acquired: bool,
    ) -> Result<(), MeshSchedulerError> {
        let queue = self
            .requester_queues
            .get_mut(&requester_principal_id)
            .ok_or(MeshSchedulerError::InvalidRequest)?;
        queue
            .requests
            .pop_front()
            .ok_or(MeshSchedulerError::InvalidRequest)?;
        self.queued_count = self.queued_count.saturating_sub(1);
        if acquired {
            queue.completed_service = queue.completed_service.saturating_add(1);
            queue.active_leases = queue.active_leases.saturating_add(1);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_fairness(
    left_service: u64,
    left_weight: u16,
    left_enqueued_at_millis: u64,
    left_sequence: u64,
    right_service: u64,
    right_weight: u16,
    right_enqueued_at_millis: u64,
    right_sequence: u64,
    now_millis: u64,
) -> Ordering {
    let left_score = aged_service(left_service, left_enqueued_at_millis, now_millis);
    let right_score = aged_service(right_service, right_enqueued_at_millis, now_millis);
    (u128::from(left_score) * u128::from(right_weight))
        .cmp(&(u128::from(right_score) * u128::from(left_weight)))
        .then_with(|| left_sequence.cmp(&right_sequence))
}

fn aged_service(service: u64, enqueued_at_millis: u64, now_millis: u64) -> u64 {
    let aging_steps = now_millis
        .saturating_sub(enqueued_at_millis)
        .checked_div(FAIRNESS_AGING_INTERVAL_MILLIS)
        .unwrap_or(0)
        .min(MAX_FAIRNESS_AGING_STEPS);
    service.saturating_sub(aging_steps)
}

fn validate_lease_request(
    request: &MeshExecutorResourceLeaseRequest,
) -> Result<(), MeshSchedulerError> {
    request.resources.validate()?;
    let lifetime = request
        .expires_at_millis
        .checked_sub(request.acquired_at_millis)
        .ok_or(MeshSchedulerError::InvalidRequest)?;
    if request.job_identity.community_id() != request.provider.community_id
        || request.attempt_id.is_nil()
        || request.lease_id.is_nil()
        || request.lease_nonce == [0; 16]
        || request.requester_principal_id.as_uuid().is_nil()
        || request.executor_principal_id.as_uuid().is_nil()
        || request.policy_audit_actor_principal_id.as_uuid().is_nil()
        || lifetime == 0
        || lifetime > MAX_MESH_LEASE_LIFETIME_MILLIS
        || request.expires_at_millis > request.provider.advertisement_expires_at_millis
        || request.cancellation_grace_millis == 0
        || request.cancellation_grace_millis > MAX_CANCELLATION_GRACE_MILLIS
    {
        return Err(MeshSchedulerError::InvalidRequest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeSet, net::SocketAddr};

    use collaboration_domain::{
        AggregateId, CommunityMembership, MembershipRole, MembershipStatus, TrustedTenantRoute,
    };
    use iroh_base::{EndpointAddr, SecretKey, TransportAddr};

    use super::*;
    use crate::mesh::{
        advertisement::{
            ADVERTISEMENT_VERSION, AdvertisementState, ApprovedComputeModel,
            AttestedComputeAdvertisement, ComputeAdvertisement, ComputeAdvertisementPolicy,
            ComputeModelAdvertisement, SharingPolicyGates,
        },
        protocol::{
            AttestedPeerMembership, DeploymentId, FrameNonce, MeshAdmissionContext, MeshAuthority,
            MeshAuthorityUnavailable, MeshFrame, MeshProtocol, PeerMembershipRecord,
            PeerMembershipState, SessionFence, StreamHello, StreamRole, WireMessage, encode_frame,
        },
    };

    const NOW: u64 = 1_800_000_000_000;

    struct ProtocolAuthority {
        membership: CommunityMembership,
        endpoint_id: EndpointId,
    }

    impl MeshAuthority for ProtocolAuthority {
        fn membership(
            &self,
            community_id: CommunityId,
            principal_id: PrincipalId,
        ) -> Result<Option<CommunityMembership>, MeshAuthorityUnavailable> {
            Ok((self.membership.community_id == community_id
                && self.membership.principal_id == principal_id)
                .then_some(self.membership))
        }

        fn runtime_generation(
            &self,
            community_id: CommunityId,
            endpoint_id: EndpointId,
        ) -> Result<Option<u64>, MeshAuthorityUnavailable> {
            Ok(
                (self.membership.community_id == community_id && self.endpoint_id == endpoint_id)
                    .then_some(7),
            )
        }

        fn session_fence(
            &self,
            _community_id: CommunityId,
            _session_id: Uuid,
        ) -> Result<Option<SessionFence>, MeshAuthorityUnavailable> {
            Ok(None)
        }
    }

    struct Fixture {
        tenant: TenantContext,
        registry: ComputeAdvertisementRegistry,
        policy: MeshFairnessPolicy,
        digest: ModelArtifactDigest,
        resources: AdvertisedResourceLimits,
        owner_principal_id: PrincipalId,
        deployment_id: DeploymentId,
        trust_secret: SecretKey,
        advertisement_policy: ComputeAdvertisementPolicy,
    }

    impl Fixture {
        fn new() -> Self {
            let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
            let owner_principal_id = PrincipalId::from_uuid(Uuid::from_u128(2));
            let tenant = TenantContext::establish(
                Some(
                    TrustedTenantRoute::from_deployment(community_id, "test/deployment")
                        .expect("valid route"),
                ),
                &[],
            )
            .expect("trusted tenant");
            let peer_secret = SecretKey::from_bytes(&[4; 32]);
            let trust_secret = SecretKey::from_bytes(&[5; 32]);
            let deployment_id =
                DeploymentId::from_bytes([3; 32]).expect("valid deployment identity");
            let transport = TransportAddr::Ip(
                "203.0.113.10:443"
                    .parse::<SocketAddr>()
                    .expect("valid address"),
            );
            let membership = CommunityMembership {
                community_id,
                principal_id: owner_principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            };
            let protocol_authority = ProtocolAuthority {
                membership,
                endpoint_id: peer_secret.public(),
            };
            let admission = MeshAdmissionContext::new(
                &tenant,
                deployment_id,
                peer_secret.public(),
                Some(trust_secret.public()),
                BTreeSet::from([transport.clone()]),
            )
            .expect("valid admission");
            let peer_record = PeerMembershipRecord {
                deployment_id,
                community_id,
                owner_principal_id,
                endpoint: EndpointAddr::from_parts(peer_secret.public(), [transport]),
                runtime_generation: 7,
                membership_version: AggregateVersion::FIRST,
                state: PeerMembershipState::Active,
                record_version: 1,
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
            };
            let membership = AttestedPeerMembership::sign(peer_record, &trust_secret)
                .expect("attested membership");
            let hello = MeshFrame {
                deployment_id,
                community_id,
                sender_endpoint_id: peer_secret.public(),
                runtime_generation: 7,
                nonce: FrameNonce::from_bytes([6; 16]).expect("valid nonce"),
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
                message: WireMessage::Hello(StreamHello {
                    membership,
                    role: StreamRole::Control,
                }),
            };
            let mut protocol = MeshProtocol::new(admission);
            protocol
                .accept_frame(
                    &encode_frame(&hello).expect("encoded hello"),
                    NOW,
                    &protocol_authority,
                )
                .expect("authenticated hello");
            let peer = protocol.authenticated_peer().expect("authenticated peer");
            let digest = ModelArtifactDigest::from_bytes([7; 32]).expect("valid digest");
            let resources = resource_limits(8);
            let advertisement_policy = ComputeAdvertisementPolicy::new(
                SharingPolicyGates {
                    deployment_enabled: true,
                    community_enabled: true,
                    user_enabled: true,
                    device_enabled: true,
                },
                BTreeSet::from([ProviderTrustClass::CommunityMemberOwned]),
                vec![ApprovedComputeModel {
                    model_id: "org/model@sha256".to_string(),
                    artifact_digest: digest,
                    capabilities: BTreeSet::from([ComputeCapability::TextGeneration]),
                }],
                resources,
            )
            .expect("valid advertisement policy");
            let advertisement = ComputeAdvertisement {
                version: ADVERTISEMENT_VERSION,
                deployment_id,
                community_id,
                owner_principal_id,
                membership_version: AggregateVersion::FIRST,
                endpoint_id: peer_secret.public(),
                runtime_generation: 7,
                device_id: MeshDeviceId::from_bytes([8; 32]).expect("valid device"),
                sharing_generation: 1,
                record_version: 1,
                state: AdvertisementState::Available,
                trust_class: ProviderTrustClass::CommunityMemberOwned,
                models: vec![ComputeModelAdvertisement {
                    model_id: "org/model@sha256".to_string(),
                    artifact_digest: digest,
                    capabilities: BTreeSet::from([ComputeCapability::TextGeneration]),
                }],
                resources: Some(resources),
                issued_at_millis: NOW,
                expires_at_millis: NOW + 60_000,
            };
            let attested = AttestedComputeAdvertisement::sign(advertisement, &peer_secret)
                .expect("signed advertisement");
            let mut registry = ComputeAdvertisementRegistry::default();
            registry
                .apply(&attested, peer, &advertisement_policy, NOW)
                .expect("registered advertisement");
            let policy = MeshFairnessPolicy::new(
                AggregateVersion::FIRST,
                PrincipalId::from_uuid(Uuid::from_u128(3)),
                HashMap::new(),
                4,
                8,
                32,
                1,
                BTreeSet::from([ProviderTrustClass::CommunityMemberOwned]),
                BTreeSet::from([TransferableContextClass::Community]),
            )
            .expect("valid scheduling policy");
            Self {
                tenant,
                registry,
                policy,
                digest,
                resources,
                owner_principal_id,
                deployment_id,
                trust_secret,
                advertisement_policy,
            }
        }

        fn add_provider(
            &mut self,
            owner: u128,
            peer_key_byte: u8,
            device_byte: u8,
            issued_at_millis: u64,
        ) {
            let community_id = self.tenant.community_id();
            let owner_principal_id = PrincipalId::from_uuid(Uuid::from_u128(owner));
            let peer_secret = SecretKey::from_bytes(&[peer_key_byte; 32]);
            let transport =
                TransportAddr::Ip(SocketAddr::from(([203, 0, 113, peer_key_byte], 443)));
            let membership = CommunityMembership {
                community_id,
                principal_id: owner_principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            };
            let protocol_authority = ProtocolAuthority {
                membership,
                endpoint_id: peer_secret.public(),
            };
            let admission = MeshAdmissionContext::new(
                &self.tenant,
                self.deployment_id,
                peer_secret.public(),
                Some(self.trust_secret.public()),
                BTreeSet::from([transport.clone()]),
            )
            .expect("valid admission");
            let peer_record = PeerMembershipRecord {
                deployment_id: self.deployment_id,
                community_id,
                owner_principal_id,
                endpoint: EndpointAddr::from_parts(peer_secret.public(), [transport]),
                runtime_generation: 7,
                membership_version: AggregateVersion::FIRST,
                state: PeerMembershipState::Active,
                record_version: 1,
                issued_at_millis: issued_at_millis.saturating_sub(1_000),
                expires_at_millis: issued_at_millis + 44_000,
            };
            let membership = AttestedPeerMembership::sign(peer_record, &self.trust_secret)
                .expect("attested membership");
            let hello = MeshFrame {
                deployment_id: self.deployment_id,
                community_id,
                sender_endpoint_id: peer_secret.public(),
                runtime_generation: 7,
                nonce: FrameNonce::from_bytes([peer_key_byte.saturating_add(1); 16])
                    .expect("valid nonce"),
                issued_at_millis: issued_at_millis.saturating_sub(1_000),
                expires_at_millis: issued_at_millis + 44_000,
                message: WireMessage::Hello(StreamHello {
                    membership,
                    role: StreamRole::Control,
                }),
            };
            let mut protocol = MeshProtocol::new(admission);
            protocol
                .accept_frame(
                    &encode_frame(&hello).expect("encoded hello"),
                    issued_at_millis,
                    &protocol_authority,
                )
                .expect("authenticated hello");
            let peer = protocol.authenticated_peer().expect("authenticated peer");
            let advertisement = ComputeAdvertisement {
                version: ADVERTISEMENT_VERSION,
                deployment_id: self.deployment_id,
                community_id,
                owner_principal_id,
                membership_version: AggregateVersion::FIRST,
                endpoint_id: peer_secret.public(),
                runtime_generation: 7,
                device_id: MeshDeviceId::from_bytes([device_byte; 32]).expect("valid device"),
                sharing_generation: 1,
                record_version: 1,
                state: AdvertisementState::Available,
                trust_class: ProviderTrustClass::CommunityMemberOwned,
                models: vec![ComputeModelAdvertisement {
                    model_id: "org/model@sha256".to_string(),
                    artifact_digest: self.digest,
                    capabilities: BTreeSet::from([ComputeCapability::TextGeneration]),
                }],
                resources: Some(self.resources),
                issued_at_millis,
                expires_at_millis: issued_at_millis + 60_000,
            };
            let attested = AttestedComputeAdvertisement::sign(advertisement, &peer_secret)
                .expect("signed advertisement");
            self.registry
                .apply(
                    &attested,
                    peer,
                    &self.advertisement_policy,
                    issued_at_millis,
                )
                .expect("registered advertisement");
        }

        fn request(&self, requester: u128, attempt: u128) -> MeshScheduleRequest {
            MeshScheduleRequest::new(
                JobIdentity::new(
                    self.tenant.community_id(),
                    AggregateId::from_uuid(Uuid::from_u128(100 + attempt)),
                )
                .expect("valid job"),
                AggregateVersion::FIRST,
                Uuid::from_u128(attempt),
                PrincipalId::from_uuid(Uuid::from_u128(requester)),
                PrincipalId::from_uuid(Uuid::from_u128(50)),
                "org/model@sha256",
                self.digest,
                ComputeCapability::TextGeneration,
                TransferableContextClass::Community,
                resource_request(1),
                Uuid::from_u128(1_000 + attempt),
                [9; 16],
                NOW,
                NOW + 30_000,
                5_000,
            )
            .expect("valid request")
        }
    }

    struct FakeAuthority {
        eligibility: RefCell<MeshCandidateEligibility>,
        acquired_requesters: RefCell<Vec<PrincipalId>>,
        acquire_error: RefCell<Option<MeshLeaseAcquireError>>,
    }

    impl FakeAuthority {
        fn eligible(fixture: &Fixture) -> Self {
            Self {
                eligibility: RefCell::new(MeshCandidateEligibility::Eligible(
                    CurrentMeshCapacity {
                        membership_version: AggregateVersion::FIRST,
                        runtime_generation: 7,
                        sharing_generation: 1,
                        advertisement_record_version: 1,
                        installed_artifact_digest: fixture.digest,
                        enforced_resource_ceiling: fixture.resources,
                        available_concurrent_slots: 4,
                        device_queued_requests: 0,
                        requester_active_leases: 0,
                    },
                )),
                acquired_requesters: RefCell::new(Vec::new()),
                acquire_error: RefCell::new(None),
            }
        }
    }

    #[async_trait(?Send)]
    impl MeshSchedulingAuthority for FakeAuthority {
        async fn evaluate_candidate(
            &self,
            _tenant: &TenantContext,
            _request: &MeshScheduleRequest,
            _provider: &MeshProviderSelection,
            _policy_version: AggregateVersion,
            _policy_audit_actor_principal_id: PrincipalId,
            _now_millis: u64,
        ) -> Result<MeshCandidateEligibility, MeshSchedulingAuthorityError> {
            Ok(*self.eligibility.borrow())
        }

        async fn acquire_executor_resource_lease(
            &self,
            _tenant: &TenantContext,
            request: MeshExecutorResourceLeaseRequest,
        ) -> Result<MeshExecutorResourceLease, MeshLeaseAcquireError> {
            if let Some(error) = self.acquire_error.borrow_mut().take() {
                return Err(error);
            }
            self.acquired_requesters
                .borrow_mut()
                .push(request.requester_principal_id);
            MeshExecutorResourceLease::new(request, AggregateVersion::FIRST)
                .map_err(|_| MeshLeaseAcquireError::AuthorityUnavailable)
        }

        async fn release_executor_resource_lease(
            &self,
            _tenant: &TenantContext,
            _lease: &MeshExecutorResourceLease,
            _released_at_millis: u64,
            _reason: MeshLeaseReleaseReason,
        ) -> Result<MeshLeaseMutationOutcome, MeshSchedulingAuthorityError> {
            Ok(MeshLeaseMutationOutcome::Applied)
        }
    }

    fn resource_limits(scale: u64) -> AdvertisedResourceLimits {
        AdvertisedResourceLimits {
            cpu_millicores: u32::try_from(1_000 * scale).expect("test scale fits"),
            memory_bytes: 1024 * scale,
            accelerator_memory_bytes: 1024 * scale,
            model_cache_bytes: 1024 * scale,
            network_bytes_per_second: 1024 * scale,
            maximum_context_bytes: 1024 * scale,
            maximum_prompt_tokens: u32::try_from(100 * scale).expect("test scale fits"),
            maximum_output_tokens: u32::try_from(100 * scale).expect("test scale fits"),
            maximum_wall_clock_millis: 1_000 * scale,
            maximum_idle_millis: 100 * scale,
            maximum_concurrent_requests: u16::try_from(scale).expect("test scale fits"),
            maximum_queued_requests: u16::try_from(scale * 2).expect("test scale fits"),
        }
    }

    fn resource_request(scale: u64) -> MeshResourceRequest {
        MeshResourceRequest {
            cpu_millicores: u32::try_from(1_000 * scale).expect("test scale fits"),
            memory_bytes: 1024 * scale,
            accelerator_memory_bytes: 1024 * scale,
            model_cache_bytes: 1024 * scale,
            network_bytes_per_second: 1024 * scale,
            context_bytes: 1024 * scale,
            prompt_tokens: u32::try_from(100 * scale).expect("test scale fits"),
            output_tokens: u32::try_from(100 * scale).expect("test scale fits"),
            wall_clock_millis: 1_000 * scale,
            idle_millis: 100 * scale,
        }
    }

    fn acquired(outcome: MeshScheduleOutcome) -> MeshExecutorResourceLease {
        match outcome {
            MeshScheduleOutcome::Acquired(lease) => lease,
            other => panic!("expected acquired lease, got {other:?}"),
        }
    }

    #[test]
    fn mesh_scheduler_rechecks_eligibility_and_acquires_exact_canonical_lease() {
        smol::block_on(async {
            let mut fixture = Fixture::new();
            let authority = FakeAuthority::eligible(&fixture);
            *authority.eligibility.borrow_mut() = MeshCandidateEligibility::Ineligible(
                MeshCandidateIneligibility::RequesterAuthorization,
            );
            let mut scheduler =
                MeshScheduler::new(fixture.tenant.community_id(), fixture.policy.clone())
                    .expect("valid scheduler");
            scheduler.enqueue(fixture.request(10, 1)).expect("queued");
            assert_eq!(
                scheduler
                    .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                    .await,
                Ok(MeshScheduleOutcome::PolicyDenied {
                    reason: MeshCandidateIneligibility::RequesterAuthorization,
                    provider: None,
                })
            );
            assert!(authority.acquired_requesters.borrow().is_empty());
        });
    }

    #[test]
    fn mesh_scheduler_applies_weighted_fairness_with_bounded_requester_queues() {
        smol::block_on(async {
            let mut fixture = Fixture::new();
            fixture.policy.requester_weights = HashMap::from([
                (PrincipalId::from_uuid(Uuid::from_u128(10)), 1),
                (PrincipalId::from_uuid(Uuid::from_u128(11)), 2),
            ]);
            fixture.policy.maximum_requester_concurrency = 4;
            let authority = FakeAuthority::eligible(&fixture);
            let mut scheduler =
                MeshScheduler::new(fixture.tenant.community_id(), fixture.policy.clone())
                    .expect("valid scheduler");
            scheduler
                .enqueue(fixture.request(10, 1))
                .expect("queued one");
            scheduler
                .enqueue(fixture.request(10, 2))
                .expect("queued two");
            scheduler
                .enqueue(fixture.request(11, 3))
                .expect("queued three");
            scheduler
                .enqueue(fixture.request(11, 4))
                .expect("queued four");
            scheduler
                .enqueue(fixture.request(11, 5))
                .expect("queued five");
            for _ in 0..4 {
                acquired(
                    scheduler
                        .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                        .await
                        .expect("scheduled"),
                );
            }
            assert_eq!(
                *authority.acquired_requesters.borrow(),
                vec![
                    PrincipalId::from_uuid(Uuid::from_u128(10)),
                    PrincipalId::from_uuid(Uuid::from_u128(11)),
                    PrincipalId::from_uuid(Uuid::from_u128(11)),
                    PrincipalId::from_uuid(Uuid::from_u128(10)),
                ]
            );
        });
    }

    #[test]
    fn mesh_scheduler_keeps_capacity_loss_visible_and_retries_only_same_provider() {
        smol::block_on(async {
            let mut fixture = Fixture::new();
            let authority = FakeAuthority::eligible(&fixture);
            *authority.acquire_error.borrow_mut() = Some(MeshLeaseAcquireError::CapacityChanged);
            let mut scheduler =
                MeshScheduler::new(fixture.tenant.community_id(), fixture.policy.clone())
                    .expect("valid scheduler");
            scheduler.enqueue(fixture.request(10, 1)).expect("queued");
            let first = scheduler
                .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                .await
                .expect("visible capacity result");
            let selected = match first {
                MeshScheduleOutcome::NoCapacity {
                    provider: Some(provider),
                } => provider,
                other => panic!("expected selected no-capacity result, got {other:?}"),
            };
            assert_eq!(scheduler.queued_count(), 1);
            let lease = acquired(
                scheduler
                    .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW + 1)
                    .await
                    .expect("retried same provider"),
            );
            assert_eq!(lease.request().provider, selected);
        });
    }

    #[test]
    fn mesh_scheduler_does_not_fallback_after_selected_peer_is_revoked() {
        smol::block_on(async {
            let mut fixture = Fixture::new();
            fixture.add_provider(12, 14, 18, NOW + 1);
            let authority = FakeAuthority::eligible(&fixture);
            *authority.acquire_error.borrow_mut() = Some(MeshLeaseAcquireError::CapacityChanged);
            let mut scheduler =
                MeshScheduler::new(fixture.tenant.community_id(), fixture.policy.clone())
                    .expect("valid scheduler");
            scheduler.enqueue(fixture.request(10, 1)).expect("queued");
            let selected = match scheduler
                .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                .await
                .expect("capacity result")
            {
                MeshScheduleOutcome::NoCapacity {
                    provider: Some(provider),
                } => provider,
                other => panic!("expected selected no-capacity result, got {other:?}"),
            };
            fixture.registry.expire(NOW + 60_000);
            assert_eq!(fixture.registry.active_hints(NOW + 1).len(), 1);
            assert_eq!(
                scheduler
                    .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW + 1,)
                    .await,
                Ok(MeshScheduleOutcome::ProviderUnavailable { provider: selected })
            );
            assert_eq!(authority.acquired_requesters.borrow().len(), 0);
        });
    }

    #[test]
    fn mesh_scheduler_enforces_local_capacity_and_owner_reservation() {
        smol::block_on(async {
            let mut fixture = Fixture::new();
            let authority = FakeAuthority::eligible(&fixture);
            let mut capacity = match *authority.eligibility.borrow() {
                MeshCandidateEligibility::Eligible(capacity) => capacity,
                MeshCandidateEligibility::Ineligible(_) => unreachable!(),
            };
            capacity.available_concurrent_slots = 1;
            *authority.eligibility.borrow_mut() = MeshCandidateEligibility::Eligible(capacity);
            let mut scheduler =
                MeshScheduler::new(fixture.tenant.community_id(), fixture.policy.clone())
                    .expect("valid scheduler");
            scheduler.enqueue(fixture.request(10, 1)).expect("queued");
            assert!(matches!(
                scheduler
                    .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                    .await,
                Ok(MeshScheduleOutcome::NoCapacity { provider: None })
            ));

            let mut owner_scheduler =
                MeshScheduler::new(fixture.tenant.community_id(), scheduler.policy.clone())
                    .expect("valid scheduler");
            owner_scheduler
                .enqueue(fixture.request(fixture.owner_principal_id.as_uuid().as_u128(), 2))
                .expect("owner queued");
            capacity.available_concurrent_slots = 1;
            *authority.eligibility.borrow_mut() = MeshCandidateEligibility::Eligible(capacity);
            let lease = acquired(
                owner_scheduler
                    .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                    .await
                    .expect("owner reservation admitted"),
            );
            assert_eq!(
                lease.request().requester_principal_id,
                fixture.owner_principal_id
            );
        });
    }
}
