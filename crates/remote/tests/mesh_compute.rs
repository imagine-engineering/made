use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    net::SocketAddr,
};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, CommunityMembership, JobIdentity, MembershipRole,
    MembershipStatus, PrincipalId, TenantContext, TrustedTenantRoute,
};
use iroh_base::{EndpointAddr, EndpointId, SecretKey, TransportAddr};
use remote::mesh::{
    advertisement::{
        ADVERTISEMENT_VERSION, AdvertisedResourceLimits, AdvertisementState, AdvertisementUpdate,
        ApprovedComputeModel, AttestedComputeAdvertisement, ComputeAdvertisement,
        ComputeAdvertisementPolicy, ComputeAdvertisementRegistry, ComputeCapability,
        ComputeModelAdvertisement, MeshDeviceId, ModelArtifactDigest, ProviderTrustClass,
        SharingPolicyGates,
    },
    protocol::{
        AttestedPeerMembership, AuthenticatedMeshPeer, DeploymentId, FrameNonce, GOSSIP_VERSION,
        GossipMessage, MeshAdmissionContext, MeshAuthority, MeshAuthorityUnavailable, MeshFrame,
        MeshProtocol, PeerMembershipRecord, PeerMembershipState, ProtocolError, SessionFence,
        StreamHello, StreamRole, WireMessage, encode_frame,
    },
    scheduler::{
        CurrentMeshCapacity, MeshCandidateEligibility, MeshCandidateIneligibility,
        MeshExecutorResourceLease, MeshExecutorResourceLeaseRequest, MeshFairnessPolicy,
        MeshLeaseAcquireError, MeshLeaseMutationOutcome, MeshLeaseReleaseReason,
        MeshProviderSelection, MeshResourceRequest, MeshScheduleOutcome, MeshScheduleRequest,
        MeshScheduler, MeshSchedulerError, MeshSchedulingAuthority, MeshSchedulingAuthorityError,
        TransferableContextClass,
    },
};
use uuid::Uuid;

const NOW: u64 = 1_800_000_000_000;
const MODEL_ID: &str = "org/model@sha256";
const FAIRNESS_WAIT_BUDGET_MILLIS: u64 = 30_000;
const MAX_EQUAL_WEIGHT_SERVICE_RATIO: f64 = 1.25;
const LOAD_REQUESTERS: u128 = 16;
const REQUESTS_PER_REQUESTER: u128 = 16;
const LOAD_ADMISSION_INTERVAL_MILLIS: u64 = 100;

struct ProtocolAuthority {
    memberships: RefCell<HashMap<(CommunityId, PrincipalId), CommunityMembership>>,
    runtime_generations: HashMap<(CommunityId, EndpointId), u64>,
    unavailable: Cell<bool>,
}

impl MeshAuthority for ProtocolAuthority {
    fn membership(
        &self,
        community_id: CommunityId,
        principal_id: PrincipalId,
    ) -> Result<Option<CommunityMembership>, MeshAuthorityUnavailable> {
        if self.unavailable.get() {
            return Err(MeshAuthorityUnavailable);
        }
        Ok(self
            .memberships
            .borrow()
            .get(&(community_id, principal_id))
            .copied())
    }

    fn runtime_generation(
        &self,
        community_id: CommunityId,
        endpoint_id: EndpointId,
    ) -> Result<Option<u64>, MeshAuthorityUnavailable> {
        if self.unavailable.get() {
            return Err(MeshAuthorityUnavailable);
        }
        Ok(self
            .runtime_generations
            .get(&(community_id, endpoint_id))
            .copied())
    }

    fn session_fence(
        &self,
        _community_id: CommunityId,
        _session_id: Uuid,
    ) -> Result<Option<SessionFence>, MeshAuthorityUnavailable> {
        if self.unavailable.get() {
            return Err(MeshAuthorityUnavailable);
        }
        Ok(None)
    }
}

struct ProtocolFixture {
    community_id: CommunityId,
    owner_principal_id: PrincipalId,
    deployment_id: DeploymentId,
    peer_secret: SecretKey,
    trust_secret: SecretKey,
    transport: TransportAddr,
    tenant: TenantContext,
    authority: ProtocolAuthority,
}

impl ProtocolFixture {
    fn new(community: u128, owner: u128, peer_key_byte: u8) -> Self {
        let community_id = CommunityId::from_uuid(Uuid::from_u128(community));
        let owner_principal_id = PrincipalId::from_uuid(Uuid::from_u128(owner));
        let deployment_id = DeploymentId::from_bytes([3; 32]).expect("valid deployment identity");
        let peer_secret = SecretKey::from_bytes(&[peer_key_byte; 32]);
        let peer_endpoint_id = peer_secret.public();
        let trust_secret = SecretKey::from_bytes(&[5; 32]);
        let transport = TransportAddr::Ip(SocketAddr::from(([203, 0, 113, peer_key_byte], 443)));
        let tenant = TenantContext::establish(
            Some(
                TrustedTenantRoute::from_deployment(community_id, "test/deployment")
                    .expect("valid tenant route"),
            ),
            &[],
        )
        .expect("trusted tenant");
        let membership = CommunityMembership {
            community_id,
            principal_id: owner_principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        };
        Self {
            community_id,
            owner_principal_id,
            deployment_id,
            peer_secret,
            trust_secret,
            transport,
            tenant,
            authority: ProtocolAuthority {
                memberships: RefCell::new(HashMap::from([(
                    (community_id, owner_principal_id),
                    membership,
                )])),
                runtime_generations: HashMap::from([((community_id, peer_endpoint_id), 7)]),
                unavailable: Cell::new(false),
            },
        }
    }

    fn admission(&self) -> MeshAdmissionContext {
        MeshAdmissionContext::new(
            &self.tenant,
            self.deployment_id,
            self.peer_secret.public(),
            Some(self.trust_secret.public()),
            BTreeSet::from([self.transport.clone()]),
        )
        .expect("valid mesh admission")
    }

    fn membership_record(
        &self,
        record_version: u64,
        state: PeerMembershipState,
    ) -> PeerMembershipRecord {
        PeerMembershipRecord {
            deployment_id: self.deployment_id,
            community_id: self.community_id,
            owner_principal_id: self.owner_principal_id,
            endpoint: EndpointAddr::from_parts(self.peer_secret.public(), [self.transport.clone()]),
            runtime_generation: 7,
            membership_version: AggregateVersion::FIRST,
            state,
            record_version,
            issued_at_millis: NOW - 1_000,
            expires_at_millis: NOW + 44_000,
        }
    }

    fn frame(&self, nonce: u128, message: WireMessage) -> MeshFrame {
        MeshFrame {
            deployment_id: self.deployment_id,
            community_id: self.community_id,
            sender_endpoint_id: self.peer_secret.public(),
            runtime_generation: 7,
            nonce: FrameNonce::from_bytes(nonce.to_le_bytes()).expect("nonzero nonce"),
            issued_at_millis: NOW - 1_000,
            expires_at_millis: NOW + 44_000,
            message,
        }
    }

    fn hello(&self, nonce: u128) -> MeshFrame {
        let membership = AttestedPeerMembership::sign(
            self.membership_record(1, PeerMembershipState::Active),
            &self.trust_secret,
        )
        .expect("attested membership");
        self.frame(
            nonce,
            WireMessage::Hello(StreamHello {
                membership,
                role: StreamRole::Control,
            }),
        )
    }
}

#[derive(Clone)]
struct RegisteredProvider {
    peer: AuthenticatedMeshPeer,
    secret: SecretKey,
    owner_principal_id: PrincipalId,
    device_id: MeshDeviceId,
}

struct MeshFixture {
    tenant: TenantContext,
    deployment_id: DeploymentId,
    trust_secret: SecretKey,
    digest: ModelArtifactDigest,
    advertised_resources: AdvertisedResourceLimits,
    advertisement_policy: ComputeAdvertisementPolicy,
    registry: ComputeAdvertisementRegistry,
}

impl MeshFixture {
    fn new(advertised_resources: AdvertisedResourceLimits) -> Self {
        let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
        let tenant = TenantContext::establish(
            Some(
                TrustedTenantRoute::from_deployment(community_id, "test/deployment")
                    .expect("valid tenant route"),
            ),
            &[],
        )
        .expect("trusted tenant");
        let deployment_id = DeploymentId::from_bytes([3; 32]).expect("valid deployment identity");
        let trust_secret = SecretKey::from_bytes(&[5; 32]);
        let digest = ModelArtifactDigest::from_bytes([7; 32]).expect("valid model digest");
        let advertisement_policy = ComputeAdvertisementPolicy::new(
            SharingPolicyGates {
                deployment_enabled: true,
                community_enabled: true,
                user_enabled: true,
                device_enabled: true,
            },
            BTreeSet::from([ProviderTrustClass::CommunityMemberOwned]),
            vec![ApprovedComputeModel {
                model_id: MODEL_ID.to_string(),
                artifact_digest: digest,
                capabilities: BTreeSet::from([ComputeCapability::TextGeneration]),
            }],
            advertised_resources,
        )
        .expect("valid advertisement policy");
        Self {
            tenant,
            deployment_id,
            trust_secret,
            digest,
            advertised_resources,
            advertisement_policy,
            registry: ComputeAdvertisementRegistry::default(),
        }
    }

    fn register_provider(
        &mut self,
        owner: u128,
        peer_key_byte: u8,
        device_byte: u8,
    ) -> RegisteredProvider {
        let owner_principal_id = PrincipalId::from_uuid(Uuid::from_u128(owner));
        let peer_secret = SecretKey::from_bytes(&[peer_key_byte; 32]);
        let transport = TransportAddr::Ip(SocketAddr::from(([203, 0, 113, peer_key_byte], 443)));
        let membership = CommunityMembership {
            community_id: self.tenant.community_id(),
            principal_id: owner_principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        };
        let authority = ProtocolAuthority {
            memberships: RefCell::new(HashMap::from([(
                (self.tenant.community_id(), owner_principal_id),
                membership,
            )])),
            runtime_generations: HashMap::from([(
                (self.tenant.community_id(), peer_secret.public()),
                7,
            )]),
            unavailable: Cell::new(false),
        };
        let admission = MeshAdmissionContext::new(
            &self.tenant,
            self.deployment_id,
            peer_secret.public(),
            Some(self.trust_secret.public()),
            BTreeSet::from([transport.clone()]),
        )
        .expect("valid admission");
        let membership = AttestedPeerMembership::sign(
            PeerMembershipRecord {
                deployment_id: self.deployment_id,
                community_id: self.tenant.community_id(),
                owner_principal_id,
                endpoint: EndpointAddr::from_parts(peer_secret.public(), [transport]),
                runtime_generation: 7,
                membership_version: AggregateVersion::FIRST,
                state: PeerMembershipState::Active,
                record_version: 1,
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
            },
            &self.trust_secret,
        )
        .expect("attested membership");
        let mut protocol = MeshProtocol::new(admission);
        protocol
            .accept_frame(
                &encode_frame(&MeshFrame {
                    deployment_id: self.deployment_id,
                    community_id: self.tenant.community_id(),
                    sender_endpoint_id: peer_secret.public(),
                    runtime_generation: 7,
                    nonce: FrameNonce::from_bytes([peer_key_byte.saturating_add(1); 16])
                        .expect("valid nonce"),
                    issued_at_millis: NOW - 1_000,
                    expires_at_millis: NOW + 44_000,
                    message: WireMessage::Hello(StreamHello {
                        membership,
                        role: StreamRole::Control,
                    }),
                })
                .expect("encoded hello"),
                NOW,
                &authority,
            )
            .expect("authenticated hello");
        let peer = protocol.authenticated_peer().expect("authenticated peer");
        let device_id = MeshDeviceId::from_bytes([device_byte; 32]).expect("valid device");
        let advertisement = self.advertisement(
            owner_principal_id,
            &peer_secret,
            device_id,
            AdvertisementState::Available,
            1,
        );
        let attested = AttestedComputeAdvertisement::sign(advertisement, &peer_secret)
            .expect("signed advertisement");
        assert_eq!(
            self.registry
                .apply(&attested, peer, &self.advertisement_policy, NOW),
            Ok(AdvertisementUpdate::Available)
        );
        RegisteredProvider {
            peer,
            secret: peer_secret,
            owner_principal_id,
            device_id,
        }
    }

    fn revoke_provider(&mut self, provider: &RegisteredProvider) {
        let advertisement = self.advertisement(
            provider.owner_principal_id,
            &provider.secret,
            provider.device_id,
            AdvertisementState::Revoked,
            2,
        );
        let attested = AttestedComputeAdvertisement::sign(advertisement, &provider.secret)
            .expect("signed revocation");
        assert_eq!(
            self.registry.apply(
                &attested,
                provider.peer,
                &self.advertisement_policy,
                NOW + 1,
            ),
            Ok(AdvertisementUpdate::Revoked)
        );
    }

    fn advertisement(
        &self,
        owner_principal_id: PrincipalId,
        secret: &SecretKey,
        device_id: MeshDeviceId,
        state: AdvertisementState,
        record_version: u64,
    ) -> ComputeAdvertisement {
        let (models, resources) = if state == AdvertisementState::Revoked {
            (Vec::new(), None)
        } else {
            (
                vec![ComputeModelAdvertisement {
                    model_id: MODEL_ID.to_string(),
                    artifact_digest: self.digest,
                    capabilities: BTreeSet::from([ComputeCapability::TextGeneration]),
                }],
                Some(self.advertised_resources),
            )
        };
        ComputeAdvertisement {
            version: ADVERTISEMENT_VERSION,
            deployment_id: self.deployment_id,
            community_id: self.tenant.community_id(),
            owner_principal_id,
            membership_version: AggregateVersion::FIRST,
            endpoint_id: secret.public(),
            runtime_generation: 7,
            device_id,
            sharing_generation: 1,
            record_version,
            state,
            trust_class: ProviderTrustClass::CommunityMemberOwned,
            models,
            resources,
            issued_at_millis: NOW,
            expires_at_millis: NOW + 60_000,
        }
    }

    fn request(
        &self,
        requester: u128,
        attempt: u128,
        resources: MeshResourceRequest,
    ) -> MeshScheduleRequest {
        MeshScheduleRequest::new(
            JobIdentity::new(
                self.tenant.community_id(),
                AggregateId::from_uuid(Uuid::from_u128(10_000 + attempt)),
            )
            .expect("valid job identity"),
            AggregateVersion::FIRST,
            Uuid::from_u128(attempt),
            PrincipalId::from_uuid(Uuid::from_u128(requester)),
            PrincipalId::from_uuid(Uuid::from_u128(999)),
            MODEL_ID,
            self.digest,
            ComputeCapability::TextGeneration,
            TransferableContextClass::Community,
            resources,
            Uuid::from_u128(20_000 + attempt),
            attempt.to_le_bytes(),
            NOW,
            NOW + 30_000,
            5_000,
        )
        .expect("valid schedule request")
    }

    fn policy(
        &self,
        requester_weights: HashMap<PrincipalId, u16>,
        maximum_requester_concurrency: u16,
        maximum_requester_queue_depth: usize,
        maximum_community_queue_depth: usize,
    ) -> MeshFairnessPolicy {
        MeshFairnessPolicy::new(
            AggregateVersion::FIRST,
            PrincipalId::from_uuid(Uuid::from_u128(3)),
            requester_weights,
            maximum_requester_concurrency,
            maximum_requester_queue_depth,
            maximum_community_queue_depth,
            0,
            BTreeSet::from([ProviderTrustClass::CommunityMemberOwned]),
            BTreeSet::from([TransferableContextClass::Community]),
        )
        .expect("valid fairness policy")
    }
}

struct SchedulingAuthority {
    eligibility: RefCell<MeshCandidateEligibility>,
    acquire_errors: RefCell<VecDeque<MeshLeaseAcquireError>>,
    acquired_requesters: RefCell<Vec<PrincipalId>>,
    active_lease_ids: RefCell<HashSet<Uuid>>,
    release_count: Cell<usize>,
}

impl SchedulingAuthority {
    fn new(capacity: CurrentMeshCapacity) -> Self {
        Self {
            eligibility: RefCell::new(MeshCandidateEligibility::Eligible(capacity)),
            acquire_errors: RefCell::new(VecDeque::new()),
            acquired_requesters: RefCell::new(Vec::new()),
            active_lease_ids: RefCell::new(HashSet::new()),
            release_count: Cell::new(0),
        }
    }

    fn set_capacity(&self, capacity: CurrentMeshCapacity) {
        *self.eligibility.borrow_mut() = MeshCandidateEligibility::Eligible(capacity);
    }

    fn fail_next_acquire(&self, error: MeshLeaseAcquireError) {
        self.acquire_errors.borrow_mut().push_back(error);
    }
}

#[async_trait(?Send)]
impl MeshSchedulingAuthority for SchedulingAuthority {
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
        if let Some(error) = self.acquire_errors.borrow_mut().pop_front() {
            return Err(error);
        }
        if !self.active_lease_ids.borrow_mut().insert(request.lease_id) {
            return Err(MeshLeaseAcquireError::JobAlreadyLeased);
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
        lease: &MeshExecutorResourceLease,
        _released_at_millis: u64,
        _reason: MeshLeaseReleaseReason,
    ) -> Result<MeshLeaseMutationOutcome, MeshSchedulingAuthorityError> {
        let outcome = if self
            .active_lease_ids
            .borrow_mut()
            .remove(&lease.request().lease_id)
        {
            self.release_count
                .set(self.release_count.get().saturating_add(1));
            MeshLeaseMutationOutcome::Applied
        } else {
            MeshLeaseMutationOutcome::Duplicate
        };
        Ok(outcome)
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
        maximum_queued_requests: u16::try_from(scale * 4).expect("test scale fits"),
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

fn current_capacity(
    fixture: &MeshFixture,
    enforced_resource_ceiling: AdvertisedResourceLimits,
    available_concurrent_slots: u16,
) -> CurrentMeshCapacity {
    CurrentMeshCapacity {
        membership_version: AggregateVersion::FIRST,
        runtime_generation: 7,
        sharing_generation: 1,
        advertisement_record_version: 1,
        installed_artifact_digest: fixture.digest,
        enforced_resource_ceiling,
        available_concurrent_slots,
        device_queued_requests: 0,
        requester_active_leases: 0,
    }
}

fn acquired(outcome: MeshScheduleOutcome) -> MeshExecutorResourceLease {
    match outcome {
        MeshScheduleOutcome::Acquired(lease) => lease,
        other => panic!("expected acquired mesh lease, got {other:?}"),
    }
}

#[test]
fn mesh_partition_recovery_rechecks_authority_before_consuming_replay_nonce() {
    let fixture = ProtocolFixture::new(1, 2, 4);
    let mut protocol = MeshProtocol::new(fixture.admission());
    protocol
        .accept_frame(
            &encode_frame(&fixture.hello(1)).expect("encoded hello"),
            NOW,
            &fixture.authority,
        )
        .expect("authenticated hello");
    let gossip = fixture.frame(
        2,
        WireMessage::Gossip(GossipMessage::Digest {
            version: GOSSIP_VERSION,
            entries: Vec::new(),
        }),
    );
    let encoded = encode_frame(&gossip).expect("encoded gossip");

    fixture.authority.unavailable.set(true);
    assert_eq!(
        protocol.accept_frame(&encoded, NOW, &fixture.authority),
        Err(ProtocolError::AuthorityUnavailable)
    );
    fixture.authority.unavailable.set(false);
    protocol
        .accept_frame(&encoded, NOW, &fixture.authority)
        .expect("same frame recovers after canonical authority returns");
    assert_eq!(
        protocol.accept_frame(&encoded, NOW, &fixture.authority),
        Err(ProtocolError::Replay)
    );

    fixture
        .authority
        .memberships
        .borrow_mut()
        .get_mut(&(fixture.community_id, fixture.owner_principal_id))
        .expect("membership")
        .status = MembershipStatus::Revoked;
    let post_revocation = encode_frame(&fixture.frame(
        3,
        WireMessage::Gossip(GossipMessage::Digest {
            version: GOSSIP_VERSION,
            entries: Vec::new(),
        }),
    ))
    .expect("encoded post-revocation frame");
    assert_eq!(
        protocol.accept_frame(&post_revocation, NOW, &fixture.authority),
        Err(ProtocolError::RevokedMembership)
    );
}

#[test]
fn mesh_revocation_removes_a_bound_provider_without_fallback() {
    smol::block_on(async {
        let mut fixture = MeshFixture::new(resource_limits(8));
        let selected_provider = fixture.register_provider(2, 4, 8);
        fixture.register_provider(12, 14, 18);
        let authority =
            SchedulingAuthority::new(current_capacity(&fixture, fixture.advertised_resources, 8));
        authority.fail_next_acquire(MeshLeaseAcquireError::CapacityChanged);
        let mut scheduler = MeshScheduler::new(
            fixture.tenant.community_id(),
            fixture.policy(HashMap::new(), 2, 16, 256),
        )
        .expect("valid scheduler");
        scheduler
            .enqueue(fixture.request(20, 1, resource_request(1)))
            .expect("queued request");
        let bound = match scheduler
            .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
            .await
            .expect("visible capacity race")
        {
            MeshScheduleOutcome::NoCapacity {
                provider: Some(provider),
            } => provider,
            other => panic!("expected bound no-capacity outcome, got {other:?}"),
        };
        assert_eq!(
            bound.owner_principal_id,
            selected_provider.owner_principal_id
        );

        fixture.revoke_provider(&selected_provider);
        assert_eq!(fixture.registry.active_hints(NOW + 2).len(), 1);
        assert_eq!(
            scheduler
                .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW + 2)
                .await,
            Ok(MeshScheduleOutcome::ProviderUnavailable { provider: bound })
        );
        assert!(authority.acquired_requesters.borrow().is_empty());
    });
}

#[test]
fn mesh_false_capacity_and_resource_claims_fail_closed_then_recover() {
    smol::block_on(async {
        let mut fixture = MeshFixture::new(resource_limits(8));
        fixture.register_provider(2, 4, 8);
        let locally_enforced = resource_limits(1);
        let authority = SchedulingAuthority::new(current_capacity(&fixture, locally_enforced, 1));
        let policy = fixture.policy(HashMap::new(), 2, 16, 256);
        let mut oversized_scheduler =
            MeshScheduler::new(fixture.tenant.community_id(), policy.clone())
                .expect("valid scheduler");
        oversized_scheduler
            .enqueue(fixture.request(20, 1, resource_request(2)))
            .expect("advertisement admits the claimed request");
        assert_eq!(
            oversized_scheduler
                .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
                .await,
            Ok(MeshScheduleOutcome::PolicyDenied {
                reason: MeshCandidateIneligibility::ResourcePolicy,
                provider: None,
            })
        );
        assert!(authority.acquired_requesters.borrow().is_empty());

        let mut recovery_scheduler =
            MeshScheduler::new(fixture.tenant.community_id(), policy).expect("valid scheduler");
        recovery_scheduler
            .enqueue(fixture.request(20, 2, resource_request(1)))
            .expect("queued bounded request");
        authority.fail_next_acquire(MeshLeaseAcquireError::CapacityChanged);
        let bound_provider = match recovery_scheduler
            .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW)
            .await
            .expect("visible atomic capacity race")
        {
            MeshScheduleOutcome::NoCapacity {
                provider: Some(provider),
            } => provider,
            other => panic!("expected bound capacity failure, got {other:?}"),
        };
        authority.set_capacity(current_capacity(&fixture, locally_enforced, 1));
        let lease = acquired(
            recovery_scheduler
                .schedule_next(&fixture.tenant, &mut fixture.registry, &authority, NOW + 1)
                .await
                .expect("same-provider recovery"),
        );
        assert_eq!(lease.request().provider, bound_provider);
        assert_eq!(authority.active_lease_ids.borrow().len(), 1);
        assert_eq!(
            recovery_scheduler
                .release(
                    &fixture.tenant,
                    &authority,
                    lease.request().lease_id,
                    NOW + 2,
                    MeshLeaseReleaseReason::Completed,
                )
                .await,
            Ok(MeshLeaseMutationOutcome::Applied)
        );
        assert!(authority.active_lease_ids.borrow().is_empty());
    });
}

#[test]
fn mesh_equal_weight_load_meets_fairness_wait_and_cleanup_budgets() {
    smol::block_on(async {
        let mut fixture = MeshFixture::new(resource_limits(64));
        fixture.register_provider(2, 4, 8);
        let authority =
            SchedulingAuthority::new(current_capacity(&fixture, fixture.advertised_resources, 64));
        let mut scheduler = MeshScheduler::new(
            fixture.tenant.community_id(),
            fixture.policy(HashMap::new(), 2, 16, 256),
        )
        .expect("valid scheduler");
        let mut attempt = 1_u128;
        for requester in 1..=LOAD_REQUESTERS {
            for _ in 0..REQUESTS_PER_REQUESTER {
                scheduler
                    .enqueue(fixture.request(100 + requester, attempt, resource_request(1)))
                    .expect("load request fits approved queue caps");
                attempt += 1;
            }
        }
        assert_eq!(scheduler.queued_count(), 256);
        assert_eq!(
            scheduler.enqueue(fixture.request(200, 257, resource_request(1))),
            Err(MeshSchedulerError::QueueFull)
        );

        let mut service_counts = HashMap::<PrincipalId, usize>::new();
        let mut waits_millis = Vec::with_capacity(256);
        for sequence in 0..256_u64 {
            let now_millis = NOW + sequence * LOAD_ADMISSION_INTERVAL_MILLIS;
            let lease = acquired(
                scheduler
                    .schedule_next(
                        &fixture.tenant,
                        &mut fixture.registry,
                        &authority,
                        now_millis,
                    )
                    .await
                    .expect("load request scheduled"),
            );
            *service_counts
                .entry(lease.request().requester_principal_id)
                .or_default() += 1;
            let maximum = service_counts.values().copied().max().unwrap_or(0);
            let minimum = (1..=LOAD_REQUESTERS)
                .map(|requester| {
                    service_counts
                        .get(&PrincipalId::from_uuid(Uuid::from_u128(100 + requester)))
                        .copied()
                        .unwrap_or(0)
                })
                .min()
                .unwrap_or(0);
            assert!(maximum.saturating_sub(minimum) <= 1);
            waits_millis.push(now_millis - NOW);
            assert_eq!(
                scheduler
                    .release(
                        &fixture.tenant,
                        &authority,
                        lease.request().lease_id,
                        now_millis + 1,
                        MeshLeaseReleaseReason::Completed,
                    )
                    .await,
                Ok(MeshLeaseMutationOutcome::Applied)
            );
        }

        assert_eq!(scheduler.queued_count(), 0);
        assert_eq!(service_counts.len(), 16);
        assert!(
            service_counts
                .values()
                .all(|count| *count == usize::try_from(REQUESTS_PER_REQUESTER).expect("fits"))
        );
        let maximum_service = service_counts.values().copied().max().unwrap_or(0) as f64;
        let minimum_service = service_counts.values().copied().min().unwrap_or(1) as f64;
        assert!(maximum_service / minimum_service <= MAX_EQUAL_WEIGHT_SERVICE_RATIO);
        let percentile_index = (waits_millis.len() * 95).div_ceil(100).saturating_sub(1);
        assert!(waits_millis[percentile_index] <= FAIRNESS_WAIT_BUDGET_MILLIS);
        assert_eq!(authority.release_count.get(), 256);
        assert!(authority.active_lease_ids.borrow().is_empty());
    });
}

#[test]
fn mesh_gossip_load_rejects_replay_without_crossing_the_bounded_window() {
    let fixture = ProtocolFixture::new(1, 2, 4);
    let mut protocol = MeshProtocol::new(fixture.admission());
    protocol
        .accept_frame(
            &encode_frame(&fixture.hello(1)).expect("encoded hello"),
            NOW,
            &fixture.authority,
        )
        .expect("authenticated hello");

    let mut first_encoded = None;
    for nonce in 2..=1_025_u128 {
        let encoded = encode_frame(&fixture.frame(
            nonce,
            WireMessage::Gossip(GossipMessage::Digest {
                version: GOSSIP_VERSION,
                entries: Vec::new(),
            }),
        ))
        .expect("encoded gossip");
        protocol
            .accept_frame(&encoded, NOW, &fixture.authority)
            .expect("bounded unique gossip frame");
        first_encoded.get_or_insert(encoded);
    }
    assert_eq!(
        protocol.accept_frame(
            first_encoded.as_deref().expect("first gossip frame"),
            NOW,
            &fixture.authority,
        ),
        Err(ProtocolError::Replay)
    );
}
