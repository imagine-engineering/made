use std::collections::{BTreeSet, HashMap, HashSet};

use collaboration_domain::{AggregateVersion, CommunityId, PrincipalId};
use iroh_base::{EndpointId, SecretKey, Signature};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::protocol::{AuthenticatedMeshPeer, DeploymentId, MAX_CLOCK_SKEW_MILLIS};

pub const ADVERTISEMENT_VERSION: u8 = 1;
pub const MAX_ADVERTISEMENT_BYTES: usize = 1024 * 1024;
pub const MAX_ADVERTISEMENT_LIFETIME_MILLIS: u64 = 60_000;
pub const MAX_ADVERTISED_MODELS: usize = 16;
pub const MAX_APPROVED_MODELS: usize = 64;
pub const MAX_MODEL_ID_BYTES: usize = 256;
pub const MAX_CAPABILITIES_PER_MODEL: usize = 8;
pub const MAX_ADVERTISEMENT_REGISTRY_ENTRIES: usize = 1024;

const ADVERTISEMENT_ATTESTATION_CONTEXT: &[u8] = b"zed-mesh-compute-advertisement-v1\0";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct MeshDeviceId([u8; 32]);

impl MeshDeviceId {
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, AdvertisementError> {
        if bytes == [0; 32] {
            return Err(AdvertisementError::InvalidIdentity);
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ModelArtifactDigest([u8; 32]);

impl ModelArtifactDigest {
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, AdvertisementError> {
        if bytes == [0; 32] {
            return Err(AdvertisementError::InvalidModel);
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeCapability {
    TextGeneration,
    ImageInput,
    StructuredOutput,
    ToolCalling,
    Embeddings,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderTrustClass {
    DeploymentManaged,
    CommunityMemberOwned,
    ThirdParty,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvertisementState {
    Available,
    Draining,
    Revoked,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AdvertisedResourceLimits {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub accelerator_memory_bytes: u64,
    pub model_cache_bytes: u64,
    pub network_bytes_per_second: u64,
    pub maximum_context_bytes: u64,
    pub maximum_prompt_tokens: u32,
    pub maximum_output_tokens: u32,
    pub maximum_wall_clock_millis: u64,
    pub maximum_idle_millis: u64,
    pub maximum_concurrent_requests: u16,
    pub maximum_queued_requests: u16,
}

impl AdvertisedResourceLimits {
    fn validate(self) -> Result<(), AdvertisementError> {
        if self.cpu_millicores == 0
            || self.memory_bytes == 0
            || self.accelerator_memory_bytes == 0
            || self.model_cache_bytes == 0
            || self.network_bytes_per_second == 0
            || self.maximum_context_bytes == 0
            || self.maximum_prompt_tokens == 0
            || self.maximum_output_tokens == 0
            || self.maximum_wall_clock_millis == 0
            || self.maximum_idle_millis == 0
            || self.maximum_concurrent_requests == 0
            || self.maximum_queued_requests == 0
        {
            return Err(AdvertisementError::InvalidResourceClaim);
        }
        Ok(())
    }

    fn is_within(self, maximum: Self) -> bool {
        self.cpu_millicores <= maximum.cpu_millicores
            && self.memory_bytes <= maximum.memory_bytes
            && self.accelerator_memory_bytes <= maximum.accelerator_memory_bytes
            && self.model_cache_bytes <= maximum.model_cache_bytes
            && self.network_bytes_per_second <= maximum.network_bytes_per_second
            && self.maximum_context_bytes <= maximum.maximum_context_bytes
            && self.maximum_prompt_tokens <= maximum.maximum_prompt_tokens
            && self.maximum_output_tokens <= maximum.maximum_output_tokens
            && self.maximum_wall_clock_millis <= maximum.maximum_wall_clock_millis
            && self.maximum_idle_millis <= maximum.maximum_idle_millis
            && self.maximum_concurrent_requests <= maximum.maximum_concurrent_requests
            && self.maximum_queued_requests <= maximum.maximum_queued_requests
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ComputeModelAdvertisement {
    pub model_id: String,
    pub artifact_digest: ModelArtifactDigest,
    pub capabilities: BTreeSet<ComputeCapability>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ComputeAdvertisement {
    pub version: u8,
    pub deployment_id: DeploymentId,
    pub community_id: CommunityId,
    pub owner_principal_id: PrincipalId,
    pub membership_version: AggregateVersion,
    pub endpoint_id: EndpointId,
    pub runtime_generation: u64,
    pub device_id: MeshDeviceId,
    pub sharing_generation: u64,
    pub record_version: u64,
    pub state: AdvertisementState,
    pub trust_class: ProviderTrustClass,
    pub models: Vec<ComputeModelAdvertisement>,
    pub resources: Option<AdvertisedResourceLimits>,
    pub issued_at_millis: u64,
    pub expires_at_millis: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttestedComputeAdvertisement {
    advertisement: ComputeAdvertisement,
    signature: Signature,
}

impl AttestedComputeAdvertisement {
    pub fn sign(
        advertisement: ComputeAdvertisement,
        endpoint_secret: &SecretKey,
    ) -> Result<Self, AdvertisementError> {
        if endpoint_secret.public() != advertisement.endpoint_id {
            return Err(AdvertisementError::PeerMismatch);
        }
        let preimage = advertisement_preimage(&advertisement)?;
        Ok(Self {
            advertisement,
            signature: endpoint_secret.sign(&preimage),
        })
    }

    pub const fn advertisement(&self) -> &ComputeAdvertisement {
        &self.advertisement
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SharingPolicyGates {
    pub deployment_enabled: bool,
    pub community_enabled: bool,
    pub user_enabled: bool,
    pub device_enabled: bool,
}

impl SharingPolicyGates {
    fn all_enabled(self) -> bool {
        self.deployment_enabled
            && self.community_enabled
            && self.user_enabled
            && self.device_enabled
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ApprovedComputeModel {
    pub model_id: String,
    pub artifact_digest: ModelArtifactDigest,
    pub capabilities: BTreeSet<ComputeCapability>,
}

#[derive(Clone, Debug)]
pub struct ComputeAdvertisementPolicy {
    gates: SharingPolicyGates,
    approved_trust_classes: BTreeSet<ProviderTrustClass>,
    approved_models: HashMap<String, ApprovedComputeModel>,
    maximum_resources: AdvertisedResourceLimits,
}

impl ComputeAdvertisementPolicy {
    pub fn new(
        gates: SharingPolicyGates,
        approved_trust_classes: BTreeSet<ProviderTrustClass>,
        approved_models: Vec<ApprovedComputeModel>,
        maximum_resources: AdvertisedResourceLimits,
    ) -> Result<Self, AdvertisementError> {
        maximum_resources.validate()?;
        if approved_trust_classes.is_empty()
            || approved_trust_classes.contains(&ProviderTrustClass::ThirdParty)
        {
            return Err(AdvertisementError::UnapprovedTrustClass);
        }
        if approved_models.is_empty() || approved_models.len() > MAX_APPROVED_MODELS {
            return Err(AdvertisementError::InvalidModel);
        }
        let mut models = HashMap::with_capacity(approved_models.len());
        for model in approved_models {
            validate_model_id(&model.model_id)?;
            if model.capabilities.is_empty()
                || model.capabilities.len() > MAX_CAPABILITIES_PER_MODEL
            {
                return Err(AdvertisementError::InvalidCapability);
            }
            if models.insert(model.model_id.clone(), model).is_some() {
                return Err(AdvertisementError::DuplicateModel);
            }
        }
        Ok(Self {
            gates,
            approved_trust_classes,
            approved_models: models,
            maximum_resources,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedComputeAdvertisement {
    advertisement: ComputeAdvertisement,
}

impl ValidatedComputeAdvertisement {
    pub const fn advertisement(&self) -> &ComputeAdvertisement {
        &self.advertisement
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdvertisementUpdate {
    Available,
    Draining,
    Revoked,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AdvertisementKey {
    owner_principal_id: PrincipalId,
    device_id: MeshDeviceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RevocationFloor {
    sharing_generation: u64,
    record_version: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ComputeAdvertisementRegistry {
    entries: HashMap<AdvertisementKey, ValidatedComputeAdvertisement>,
    endpoint_owners: HashMap<EndpointId, AdvertisementKey>,
    revocation_floors: HashMap<AdvertisementKey, RevocationFloor>,
}

impl ComputeAdvertisementRegistry {
    pub fn apply(
        &mut self,
        attested: &AttestedComputeAdvertisement,
        peer: AuthenticatedMeshPeer,
        policy: &ComputeAdvertisementPolicy,
        now_millis: u64,
    ) -> Result<AdvertisementUpdate, AdvertisementError> {
        self.expire(now_millis);
        let validated = validate_advertisement(attested, peer, policy, now_millis)?;
        let advertisement = validated.advertisement();
        let key = AdvertisementKey {
            owner_principal_id: advertisement.owner_principal_id,
            device_id: advertisement.device_id,
        };

        if advertisement.state == AdvertisementState::Revoked {
            self.apply_revocation(key, advertisement)?;
            return Ok(AdvertisementUpdate::Revoked);
        }
        if self
            .revocation_floors
            .get(&key)
            .is_some_and(|floor| advertisement.sharing_generation <= floor.sharing_generation)
        {
            return Err(AdvertisementError::RevokedGeneration);
        }
        if let Some(existing) = self.entries.get(&key) {
            let existing = existing.advertisement();
            if advertisement.sharing_generation < existing.sharing_generation
                || (advertisement.sharing_generation == existing.sharing_generation
                    && advertisement.record_version <= existing.record_version)
            {
                return Err(AdvertisementError::StaleAdvertisement);
            }
        } else if self.entries.len() >= MAX_ADVERTISEMENT_REGISTRY_ENTRIES {
            return Err(AdvertisementError::RegistryFull);
        }
        if self
            .endpoint_owners
            .get(&advertisement.endpoint_id)
            .is_some_and(|owner| *owner != key)
        {
            return Err(AdvertisementError::DuplicateEndpoint);
        }

        if let Some(previous) = self.entries.get(&key) {
            self.endpoint_owners
                .remove(&previous.advertisement.endpoint_id);
        }
        self.endpoint_owners.insert(advertisement.endpoint_id, key);
        let update = if advertisement.state == AdvertisementState::Available {
            AdvertisementUpdate::Available
        } else {
            AdvertisementUpdate::Draining
        };
        self.entries.insert(key, validated);
        Ok(update)
    }

    pub fn active_hints(&mut self, now_millis: u64) -> Vec<&ValidatedComputeAdvertisement> {
        self.expire(now_millis);
        let mut entries: Vec<_> = self
            .entries
            .values()
            .filter(|entry| entry.advertisement.state == AdvertisementState::Available)
            .collect();
        entries.sort_by_key(|entry| {
            (
                entry.advertisement.owner_principal_id,
                entry.advertisement.device_id,
            )
        });
        entries
    }

    pub fn expire(&mut self, now_millis: u64) -> usize {
        let previous_len = self.entries.len();
        self.entries
            .retain(|_, entry| entry.advertisement.expires_at_millis > now_millis);
        self.rebuild_endpoint_owners();
        previous_len.saturating_sub(self.entries.len())
    }

    fn apply_revocation(
        &mut self,
        key: AdvertisementKey,
        advertisement: &ComputeAdvertisement,
    ) -> Result<(), AdvertisementError> {
        if !self.revocation_floors.contains_key(&key)
            && self.revocation_floors.len() >= MAX_ADVERTISEMENT_REGISTRY_ENTRIES
        {
            return Err(AdvertisementError::RegistryFull);
        }
        if let Some(floor) = self.revocation_floors.get(&key)
            && (advertisement.sharing_generation < floor.sharing_generation
                || (advertisement.sharing_generation == floor.sharing_generation
                    && advertisement.record_version <= floor.record_version))
        {
            return Err(AdvertisementError::StaleAdvertisement);
        }
        if let Some(existing) = self.entries.get(&key) {
            let existing = existing.advertisement();
            if advertisement.sharing_generation < existing.sharing_generation
                || (advertisement.sharing_generation == existing.sharing_generation
                    && advertisement.record_version <= existing.record_version)
            {
                return Err(AdvertisementError::StaleAdvertisement);
            }
        }
        self.entries.remove(&key);
        self.rebuild_endpoint_owners();
        self.revocation_floors.insert(
            key,
            RevocationFloor {
                sharing_generation: advertisement.sharing_generation,
                record_version: advertisement.record_version,
            },
        );
        Ok(())
    }

    fn rebuild_endpoint_owners(&mut self) {
        self.endpoint_owners.clear();
        self.endpoint_owners.extend(
            self.entries
                .iter()
                .map(|(key, value)| (value.advertisement.endpoint_id, *key)),
        );
    }
}

pub fn encode_advertisement(
    advertisement: &AttestedComputeAdvertisement,
) -> Result<Vec<u8>, AdvertisementError> {
    let bytes = postcard::to_stdvec(advertisement).map_err(|_| AdvertisementError::Malformed)?;
    if bytes.len() > MAX_ADVERTISEMENT_BYTES {
        return Err(AdvertisementError::AdvertisementTooLarge);
    }
    Ok(bytes)
}

pub fn decode_advertisement(
    bytes: &[u8],
) -> Result<AttestedComputeAdvertisement, AdvertisementError> {
    if bytes.len() > MAX_ADVERTISEMENT_BYTES {
        return Err(AdvertisementError::AdvertisementTooLarge);
    }
    postcard::from_bytes(bytes).map_err(|_| AdvertisementError::Malformed)
}

pub fn validate_advertisement(
    attested: &AttestedComputeAdvertisement,
    peer: AuthenticatedMeshPeer,
    policy: &ComputeAdvertisementPolicy,
    now_millis: u64,
) -> Result<ValidatedComputeAdvertisement, AdvertisementError> {
    encode_advertisement(attested)?;
    let advertisement = &attested.advertisement;
    if advertisement.version != ADVERTISEMENT_VERSION {
        return Err(AdvertisementError::UnsupportedVersion);
    }
    if advertisement.deployment_id != peer.deployment_id()
        || advertisement.community_id != peer.community_id()
        || advertisement.owner_principal_id != peer.owner_principal_id()
        || advertisement.membership_version != peer.membership_version()
        || advertisement.endpoint_id != peer.endpoint_id()
        || advertisement.runtime_generation != peer.runtime_generation()
    {
        return Err(AdvertisementError::PeerMismatch);
    }
    let preimage = advertisement_preimage(advertisement)?;
    advertisement
        .endpoint_id
        .verify(&preimage, &attested.signature)
        .map_err(|_| AdvertisementError::InvalidSignature)?;
    validate_time_window(advertisement, now_millis)?;
    if advertisement.sharing_generation == 0 || advertisement.record_version == 0 {
        return Err(AdvertisementError::InvalidGeneration);
    }
    if !policy.gates.all_enabled() {
        return Err(AdvertisementError::SharingDisabled);
    }
    if !policy
        .approved_trust_classes
        .contains(&advertisement.trust_class)
        || advertisement.trust_class == ProviderTrustClass::ThirdParty
    {
        return Err(AdvertisementError::UnapprovedTrustClass);
    }

    match advertisement.state {
        AdvertisementState::Available | AdvertisementState::Draining => {
            validate_available_advertisement(advertisement, policy)?;
        }
        AdvertisementState::Revoked => {
            if !advertisement.models.is_empty() || advertisement.resources.is_some() {
                return Err(AdvertisementError::InvalidRevocation);
            }
        }
    }
    Ok(ValidatedComputeAdvertisement {
        advertisement: advertisement.clone(),
    })
}

fn validate_available_advertisement(
    advertisement: &ComputeAdvertisement,
    policy: &ComputeAdvertisementPolicy,
) -> Result<(), AdvertisementError> {
    if advertisement.models.is_empty() || advertisement.models.len() > MAX_ADVERTISED_MODELS {
        return Err(AdvertisementError::InvalidModel);
    }
    let resources = advertisement
        .resources
        .ok_or(AdvertisementError::InvalidResourceClaim)?;
    resources.validate()?;
    if !resources.is_within(policy.maximum_resources) {
        return Err(AdvertisementError::ResourceClaimExceedsPolicy);
    }

    let mut model_ids = HashSet::with_capacity(advertisement.models.len());
    for model in &advertisement.models {
        validate_model_id(&model.model_id)?;
        if !model_ids.insert(model.model_id.as_str()) {
            return Err(AdvertisementError::DuplicateModel);
        }
        if model.capabilities.is_empty() || model.capabilities.len() > MAX_CAPABILITIES_PER_MODEL {
            return Err(AdvertisementError::InvalidCapability);
        }
        let approved = policy
            .approved_models
            .get(&model.model_id)
            .ok_or(AdvertisementError::UnapprovedModel)?;
        if model.artifact_digest != approved.artifact_digest {
            return Err(AdvertisementError::ArtifactDigestMismatch);
        }
        if !model.capabilities.is_subset(&approved.capabilities) {
            return Err(AdvertisementError::CapabilityNotApproved);
        }
    }
    Ok(())
}

fn validate_model_id(model_id: &str) -> Result<(), AdvertisementError> {
    if model_id.is_empty()
        || model_id.len() > MAX_MODEL_ID_BYTES
        || model_id.trim() != model_id
        || model_id.chars().any(char::is_control)
    {
        return Err(AdvertisementError::InvalidModel);
    }
    Ok(())
}

fn validate_time_window(
    advertisement: &ComputeAdvertisement,
    now_millis: u64,
) -> Result<(), AdvertisementError> {
    let lifetime = advertisement
        .expires_at_millis
        .checked_sub(advertisement.issued_at_millis)
        .ok_or(AdvertisementError::InvalidLifetime)?;
    if lifetime == 0 || lifetime > MAX_ADVERTISEMENT_LIFETIME_MILLIS {
        return Err(AdvertisementError::InvalidLifetime);
    }
    if advertisement.issued_at_millis > now_millis.saturating_add(MAX_CLOCK_SKEW_MILLIS) {
        return Err(AdvertisementError::NotYetValid);
    }
    if advertisement.expires_at_millis <= now_millis {
        return Err(AdvertisementError::Expired);
    }
    Ok(())
}

fn advertisement_preimage(
    advertisement: &ComputeAdvertisement,
) -> Result<Vec<u8>, AdvertisementError> {
    postcard::to_extend(advertisement, ADVERTISEMENT_ATTESTATION_CONTEXT.to_vec())
        .map_err(|_| AdvertisementError::Malformed)
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum AdvertisementError {
    #[error("mesh advertisement identity is invalid")]
    InvalidIdentity,
    #[error("mesh advertisement model is invalid")]
    InvalidModel,
    #[error("mesh advertisement capability is invalid")]
    InvalidCapability,
    #[error("mesh advertisement contains a duplicate model")]
    DuplicateModel,
    #[error("mesh advertisement resource claim is invalid")]
    InvalidResourceClaim,
    #[error("mesh advertisement resource claim exceeds policy")]
    ResourceClaimExceedsPolicy,
    #[error("mesh advertisement trust class is not approved")]
    UnapprovedTrustClass,
    #[error("mesh sharing is disabled by policy")]
    SharingDisabled,
    #[error("mesh advertisement version is unsupported")]
    UnsupportedVersion,
    #[error("mesh advertisement is malformed")]
    Malformed,
    #[error("mesh advertisement exceeds its size limit")]
    AdvertisementTooLarge,
    #[error("mesh advertisement lifetime is invalid")]
    InvalidLifetime,
    #[error("mesh advertisement is not yet valid")]
    NotYetValid,
    #[error("mesh advertisement has expired")]
    Expired,
    #[error("mesh advertisement peer binding does not match")]
    PeerMismatch,
    #[error("mesh advertisement signature is invalid")]
    InvalidSignature,
    #[error("mesh advertisement generation is invalid")]
    InvalidGeneration,
    #[error("mesh advertisement model is not approved")]
    UnapprovedModel,
    #[error("mesh advertisement artifact digest does not match policy")]
    ArtifactDigestMismatch,
    #[error("mesh advertisement capability is not approved")]
    CapabilityNotApproved,
    #[error("mesh advertisement revocation is invalid")]
    InvalidRevocation,
    #[error("mesh advertisement is stale or duplicated")]
    StaleAdvertisement,
    #[error("mesh advertisement generation was revoked")]
    RevokedGeneration,
    #[error("mesh endpoint is already advertised by another device")]
    DuplicateEndpoint,
    #[error("mesh advertisement registry is full")]
    RegistryFull,
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, net::SocketAddr};

    use collaboration_domain::{
        CommunityMembership, MembershipRole, MembershipStatus, TenantContext, TrustedTenantRoute,
    };
    use iroh_base::{EndpointAddr, TransportAddr};
    use uuid::Uuid;

    use super::*;
    use crate::mesh::protocol::{
        AttestedPeerMembership, FrameNonce, MeshAdmissionContext, MeshAuthority,
        MeshAuthorityUnavailable, MeshFrame, MeshProtocol, PeerMembershipRecord,
        PeerMembershipState, StreamHello, StreamRole, WireMessage, encode_frame,
    };

    const NOW: u64 = 1_800_000_000_000;

    struct TestAuthority {
        membership: CommunityMembership,
        endpoint_id: EndpointId,
    }

    impl MeshAuthority for TestAuthority {
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
        ) -> Result<Option<crate::mesh::protocol::SessionFence>, MeshAuthorityUnavailable> {
            Ok(None)
        }
    }

    struct Fixture {
        peer: AuthenticatedMeshPeer,
        peer_secret: SecretKey,
        policy: ComputeAdvertisementPolicy,
        advertisement: ComputeAdvertisement,
    }

    impl Fixture {
        fn new() -> Self {
            let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
            let principal_id = PrincipalId::from_uuid(Uuid::from_u128(2));
            let deployment_id =
                DeploymentId::from_bytes([3; 32]).expect("valid deployment identity");
            let peer_secret = SecretKey::from_bytes(&[4; 32]);
            let trust_secret = SecretKey::from_bytes(&[5; 32]);
            let transport = TransportAddr::Ip(
                "203.0.113.10:443"
                    .parse::<SocketAddr>()
                    .expect("valid address"),
            );
            let membership = CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            };
            let authority = TestAuthority {
                membership,
                endpoint_id: peer_secret.public(),
            };
            let tenant = TenantContext::establish(
                Some(
                    TrustedTenantRoute::from_deployment(community_id, "test/deployment")
                        .expect("valid route"),
                ),
                &[],
            )
            .expect("trusted tenant");
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
                owner_principal_id: principal_id,
                endpoint: EndpointAddr::from_parts(peer_secret.public(), [transport]),
                runtime_generation: 7,
                membership_version: AggregateVersion::FIRST,
                state: PeerMembershipState::Active,
                record_version: 1,
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
            };
            let attested_peer =
                AttestedPeerMembership::sign(peer_record, &trust_secret).expect("attested peer");
            let hello = MeshFrame {
                deployment_id,
                community_id,
                sender_endpoint_id: peer_secret.public(),
                runtime_generation: 7,
                nonce: FrameNonce::from_bytes([6; 16]).expect("nonce"),
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
                message: WireMessage::Hello(StreamHello {
                    membership: attested_peer,
                    role: StreamRole::Control,
                }),
            };
            let mut protocol = MeshProtocol::new(admission);
            protocol
                .accept_frame(
                    &encode_frame(&hello).expect("encoded hello"),
                    NOW,
                    &authority,
                )
                .expect("authenticated hello");
            let peer = protocol.authenticated_peer().expect("authenticated peer");

            let resources = resource_limits(4);
            let digest = ModelArtifactDigest::from_bytes([7; 32]).expect("model digest");
            let capabilities = BTreeSet::from([ComputeCapability::TextGeneration]);
            let policy = ComputeAdvertisementPolicy::new(
                SharingPolicyGates {
                    deployment_enabled: true,
                    community_enabled: true,
                    user_enabled: true,
                    device_enabled: true,
                },
                BTreeSet::from([
                    ProviderTrustClass::DeploymentManaged,
                    ProviderTrustClass::CommunityMemberOwned,
                ]),
                vec![ApprovedComputeModel {
                    model_id: "org/model@sha256".to_string(),
                    artifact_digest: digest,
                    capabilities: capabilities.clone(),
                }],
                resource_limits(8),
            )
            .expect("valid policy");
            let advertisement = ComputeAdvertisement {
                version: ADVERTISEMENT_VERSION,
                deployment_id,
                community_id,
                owner_principal_id: principal_id,
                membership_version: AggregateVersion::FIRST,
                endpoint_id: peer_secret.public(),
                runtime_generation: 7,
                device_id: MeshDeviceId::from_bytes([8; 32]).expect("device id"),
                sharing_generation: 1,
                record_version: 1,
                state: AdvertisementState::Available,
                trust_class: ProviderTrustClass::CommunityMemberOwned,
                models: vec![ComputeModelAdvertisement {
                    model_id: "org/model@sha256".to_string(),
                    artifact_digest: digest,
                    capabilities,
                }],
                resources: Some(resources),
                issued_at_millis: NOW,
                expires_at_millis: NOW + MAX_ADVERTISEMENT_LIFETIME_MILLIS,
            };
            Self {
                peer,
                peer_secret,
                policy,
                advertisement,
            }
        }

        fn sign(&self, advertisement: ComputeAdvertisement) -> AttestedComputeAdvertisement {
            AttestedComputeAdvertisement::sign(advertisement, &self.peer_secret)
                .expect("signed advertisement")
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

    #[test]
    fn compute_advertisement_rejects_spoofed_resource_and_artifact_claims() {
        let fixture = Fixture::new();
        let mut oversized = fixture.advertisement.clone();
        oversized.resources = Some(resource_limits(9));
        assert_eq!(
            validate_advertisement(&fixture.sign(oversized), fixture.peer, &fixture.policy, NOW),
            Err(AdvertisementError::ResourceClaimExceedsPolicy)
        );

        let mut wrong_artifact = fixture.advertisement.clone();
        wrong_artifact.models[0].artifact_digest =
            ModelArtifactDigest::from_bytes([9; 32]).expect("digest");
        assert_eq!(
            validate_advertisement(
                &fixture.sign(wrong_artifact),
                fixture.peer,
                &fixture.policy,
                NOW,
            ),
            Err(AdvertisementError::ArtifactDigestMismatch)
        );
    }

    #[test]
    fn compute_advertisement_expires_at_the_exact_boundary() {
        let fixture = Fixture::new();
        let mut registry = ComputeAdvertisementRegistry::default();
        registry
            .apply(
                &fixture.sign(fixture.advertisement.clone()),
                fixture.peer,
                &fixture.policy,
                NOW,
            )
            .expect("active advertisement");
        assert_eq!(
            registry
                .active_hints(NOW + MAX_ADVERTISEMENT_LIFETIME_MILLIS - 1)
                .len(),
            1
        );
        assert!(
            registry
                .active_hints(NOW + MAX_ADVERTISEMENT_LIFETIME_MILLIS)
                .is_empty()
        );
    }

    #[test]
    fn compute_advertisement_revocation_removes_and_fences_generation() {
        let fixture = Fixture::new();
        let mut registry = ComputeAdvertisementRegistry::default();
        registry
            .apply(
                &fixture.sign(fixture.advertisement.clone()),
                fixture.peer,
                &fixture.policy,
                NOW,
            )
            .expect("active advertisement");
        let mut revoked = fixture.advertisement.clone();
        revoked.record_version = 2;
        revoked.state = AdvertisementState::Revoked;
        revoked.models.clear();
        revoked.resources = None;
        assert_eq!(
            registry.apply(&fixture.sign(revoked), fixture.peer, &fixture.policy, NOW,),
            Ok(AdvertisementUpdate::Revoked)
        );
        assert!(registry.active_hints(NOW).is_empty());

        let mut replay = fixture.advertisement.clone();
        replay.record_version = 3;
        assert_eq!(
            registry.apply(&fixture.sign(replay), fixture.peer, &fixture.policy, NOW,),
            Err(AdvertisementError::RevokedGeneration)
        );
    }

    #[test]
    fn compute_advertisement_rejects_duplicate_endpoint_for_another_device() {
        let fixture = Fixture::new();
        let mut registry = ComputeAdvertisementRegistry::default();
        registry
            .apply(
                &fixture.sign(fixture.advertisement.clone()),
                fixture.peer,
                &fixture.policy,
                NOW,
            )
            .expect("first device");
        let mut duplicate = fixture.advertisement.clone();
        duplicate.device_id = MeshDeviceId::from_bytes([10; 32]).expect("second device");
        duplicate.record_version = 2;
        assert_eq!(
            registry.apply(&fixture.sign(duplicate), fixture.peer, &fixture.policy, NOW,),
            Err(AdvertisementError::DuplicateEndpoint)
        );
        assert_eq!(registry.active_hints(NOW).len(), 1);
    }

    #[test]
    fn compute_advertisement_enforces_encoded_and_structural_bounds() {
        let fixture = Fixture::new();
        assert_eq!(
            decode_advertisement(&vec![0; MAX_ADVERTISEMENT_BYTES + 1]),
            Err(AdvertisementError::AdvertisementTooLarge)
        );
        let mut invalid_model = fixture.advertisement.clone();
        invalid_model.models[0].model_id = "x".repeat(MAX_MODEL_ID_BYTES + 1);
        assert_eq!(
            validate_advertisement(
                &fixture.sign(invalid_model),
                fixture.peer,
                &fixture.policy,
                NOW,
            ),
            Err(AdvertisementError::InvalidModel)
        );
    }

    #[test]
    fn compute_advertisement_requires_every_opt_in_without_granting_a_lease() {
        let fixture = Fixture::new();
        let disabled = ComputeAdvertisementPolicy::new(
            SharingPolicyGates {
                deployment_enabled: true,
                community_enabled: true,
                user_enabled: true,
                device_enabled: false,
            },
            BTreeSet::from([ProviderTrustClass::CommunityMemberOwned]),
            fixture.policy.approved_models.values().cloned().collect(),
            fixture.policy.maximum_resources,
        )
        .expect("valid disabled policy");
        assert_eq!(
            validate_advertisement(
                &fixture.sign(fixture.advertisement.clone()),
                fixture.peer,
                &disabled,
                NOW,
            ),
            Err(AdvertisementError::SharingDisabled)
        );
    }
}
