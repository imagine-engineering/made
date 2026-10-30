use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::{Arc, Mutex},
};

use collaboration_domain::{
    AggregateVersion, CommunityId, CommunityMembership, MembershipStatus, PrincipalId,
    TenantContext,
};
use iroh_base::{EndpointAddr, EndpointId, SecretKey, Signature, TransportAddr};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub const ALPN: &[u8] = b"zed/mesh/1";
pub const WIRE_VERSION: u8 = 1;
pub const GOSSIP_VERSION: u8 = 1;
pub const MAX_STREAM_FRAME_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_CONTROL_FRAME_BYTES: usize = 1024 * 1024;
pub const MAX_ENDPOINT_IP_ADDRESSES: usize = 8;
pub const MAX_ENDPOINT_TRANSPORTS: usize = 16;
pub const MAX_GOSSIP_RECORDS: usize = 256;
pub const MAX_TRACKED_MEMBERSHIPS: usize = 4096;
pub const MAX_REPLAY_NONCES: usize = 4096;
pub const MAX_FRAME_LIFETIME_MILLIS: u64 = 45_000;
pub const MAX_CLOCK_SKEW_MILLIS: u64 = 5_000;

const MEMBERSHIP_ATTESTATION_CONTEXT: &[u8] = b"zed-mesh-membership-v1\0";
const CONTROL_FRAME_CLASS: u8 = 0;
const SESSION_FRAME_CLASS: u8 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DeploymentId([u8; 32]);

impl DeploymentId {
    pub fn from_bytes(bytes: [u8; 32]) -> Result<Self, ProtocolError> {
        if bytes == [0; 32] {
            return Err(ProtocolError::InvalidDeploymentIdentity);
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FrameNonce([u8; 16]);

impl FrameNonce {
    pub fn from_bytes(bytes: [u8; 16]) -> Result<Self, ProtocolError> {
        if bytes == [0; 16] {
            return Err(ProtocolError::InvalidNonce);
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PeerMembershipState {
    Active,
    Draining,
    Revoked,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PeerMembershipRecord {
    pub deployment_id: DeploymentId,
    pub community_id: CommunityId,
    pub owner_principal_id: PrincipalId,
    pub endpoint: EndpointAddr,
    pub runtime_generation: u64,
    pub membership_version: AggregateVersion,
    pub state: PeerMembershipState,
    pub record_version: u64,
    pub issued_at_millis: u64,
    pub expires_at_millis: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AttestedPeerMembership {
    record: PeerMembershipRecord,
    signature: Signature,
}

impl AttestedPeerMembership {
    pub fn sign(
        record: PeerMembershipRecord,
        trust_root: &SecretKey,
    ) -> Result<Self, ProtocolError> {
        let preimage = membership_attestation_preimage(&record)?;
        Ok(Self {
            record,
            signature: trust_root.sign(&preimage),
        })
    }

    pub const fn record(&self) -> &PeerMembershipRecord {
        &self.record
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SessionFence {
    pub session_id: Uuid,
    pub generation: u64,
    pub owner_endpoint_id: EndpointId,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionProfile {
    ReliableStream,
    RealtimeMedia,
    HuddleControl,
    SharedCompute,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum StreamRole {
    Control,
    Session {
        fence: SessionFence,
        profile: SessionProfile,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamHello {
    pub membership: AttestedPeerMembership,
    pub role: StreamRole,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GossipDigestEntry {
    pub endpoint_id: EndpointId,
    pub record_version: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum GossipMessage {
    Digest {
        version: u8,
        entries: Vec<GossipDigestEntry>,
    },
    Delta {
        version: u8,
        records: Vec<AttestedPeerMembership>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GoodbyeReason {
    SessionEnded,
    Draining,
    StaleGeneration,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum SessionMessage {
    Data {
        fence: SessionFence,
        payload: Vec<u8>,
    },
    Goodbye {
        fence: SessionFence,
        reason: GoodbyeReason,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum WireMessage {
    Hello(StreamHello),
    Gossip(GossipMessage),
    Session(SessionMessage),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MeshFrame {
    pub deployment_id: DeploymentId,
    pub community_id: CommunityId,
    pub sender_endpoint_id: EndpointId,
    pub runtime_generation: u64,
    pub nonce: FrameNonce,
    pub issued_at_millis: u64,
    pub expires_at_millis: u64,
    pub message: WireMessage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeshAuthorityUnavailable;

pub trait MeshAuthority {
    fn membership(
        &self,
        community_id: CommunityId,
        principal_id: PrincipalId,
    ) -> Result<Option<CommunityMembership>, MeshAuthorityUnavailable>;

    fn runtime_generation(
        &self,
        community_id: CommunityId,
        endpoint_id: EndpointId,
    ) -> Result<Option<u64>, MeshAuthorityUnavailable>;

    fn session_fence(
        &self,
        community_id: CommunityId,
        session_id: Uuid,
    ) -> Result<Option<SessionFence>, MeshAuthorityUnavailable>;
}

#[derive(Clone, Debug)]
pub struct MeshAdmissionContext {
    community_id: CommunityId,
    deployment_id: DeploymentId,
    transport_peer_id: EndpointId,
    trust_root: EndpointId,
    approved_transports: BTreeSet<TransportAddr>,
    replay_nonces: Arc<Mutex<HashMap<FrameNonce, u64>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthenticatedMeshPeer {
    deployment_id: DeploymentId,
    community_id: CommunityId,
    owner_principal_id: PrincipalId,
    endpoint_id: EndpointId,
    runtime_generation: u64,
    membership_version: AggregateVersion,
}

impl AuthenticatedMeshPeer {
    fn from_record(record: &PeerMembershipRecord) -> Self {
        Self {
            deployment_id: record.deployment_id,
            community_id: record.community_id,
            owner_principal_id: record.owner_principal_id,
            endpoint_id: record.endpoint.id,
            runtime_generation: record.runtime_generation,
            membership_version: record.membership_version,
        }
    }

    pub const fn deployment_id(self) -> DeploymentId {
        self.deployment_id
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn owner_principal_id(self) -> PrincipalId {
        self.owner_principal_id
    }

    pub const fn endpoint_id(self) -> EndpointId {
        self.endpoint_id
    }

    pub const fn runtime_generation(self) -> u64 {
        self.runtime_generation
    }

    pub const fn membership_version(self) -> AggregateVersion {
        self.membership_version
    }
}

impl MeshAdmissionContext {
    pub fn new(
        tenant: &TenantContext,
        deployment_id: DeploymentId,
        transport_peer_id: EndpointId,
        trust_root: Option<EndpointId>,
        approved_transports: BTreeSet<TransportAddr>,
    ) -> Result<Self, ProtocolError> {
        let trust_root = trust_root.ok_or(ProtocolError::MissingTrustRoot)?;
        if approved_transports.is_empty() {
            return Err(ProtocolError::MissingApprovedTransport);
        }
        Ok(Self {
            community_id: tenant.community_id(),
            deployment_id,
            transport_peer_id,
            trust_root,
            approved_transports,
            replay_nonces: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

#[derive(Clone, Debug)]
enum ProtocolState {
    AwaitingHello,
    Control {
        peer: PeerMembershipRecord,
    },
    Session {
        peer: PeerMembershipRecord,
        fence: SessionFence,
    },
    Closed,
}

#[derive(Clone, Debug)]
pub struct MeshProtocol {
    admission: MeshAdmissionContext,
    state: ProtocolState,
    membership_versions: HashMap<EndpointId, u64>,
}

impl MeshProtocol {
    pub fn new(admission: MeshAdmissionContext) -> Self {
        Self {
            admission,
            state: ProtocolState::AwaitingHello,
            membership_versions: HashMap::new(),
        }
    }

    pub fn authenticated_peer(&self) -> Option<AuthenticatedMeshPeer> {
        match &self.state {
            ProtocolState::Control { peer } | ProtocolState::Session { peer, .. } => {
                Some(AuthenticatedMeshPeer::from_record(peer))
            }
            ProtocolState::AwaitingHello | ProtocolState::Closed => None,
        }
    }

    pub fn accept_frame(
        &mut self,
        bytes: &[u8],
        now_millis: u64,
        authority: &impl MeshAuthority,
    ) -> Result<MeshFrame, ProtocolError> {
        let state_maximum = match &self.state {
            ProtocolState::AwaitingHello | ProtocolState::Control { .. } => MAX_CONTROL_FRAME_BYTES,
            ProtocolState::Session { .. } => MAX_STREAM_FRAME_BYTES,
            ProtocolState::Closed => return Err(ProtocolError::StreamClosed),
        };
        if bytes.len() > state_maximum {
            return Err(if state_maximum == MAX_CONTROL_FRAME_BYTES {
                ProtocolError::ControlFrameTooLarge
            } else {
                ProtocolError::FrameTooLarge
            });
        }
        let frame = decode_frame(bytes)?;
        self.validate_common(&frame, now_millis)?;

        let transition = match &self.state {
            ProtocolState::AwaitingHello => {
                let WireMessage::Hello(hello) = &frame.message else {
                    return Err(ProtocolError::FirstFrameMustBeHello);
                };
                self.validate_membership(
                    &hello.membership,
                    frame.sender_endpoint_id,
                    now_millis,
                    authority,
                    false,
                )?;
                let record = hello.membership.record();
                if record.runtime_generation != frame.runtime_generation {
                    return Err(ProtocolError::RuntimeGenerationMismatch);
                }
                match hello.role {
                    StreamRole::Control => ProtocolState::Control {
                        peer: record.clone(),
                    },
                    StreamRole::Session { fence, profile } => {
                        if profile == SessionProfile::RealtimeMedia {
                            return Err(ProtocolError::RealtimeMediaRequiresDatagram);
                        }
                        self.validate_session_fence(fence, authority)?;
                        ProtocolState::Session {
                            peer: record.clone(),
                            fence,
                        }
                    }
                }
            }
            ProtocolState::Control { peer } => {
                self.revalidate_peer(peer, now_millis, authority)?;
                let WireMessage::Gossip(gossip) = &frame.message else {
                    return Err(if matches!(&frame.message, WireMessage::Hello(_)) {
                        ProtocolError::DuplicateHello
                    } else {
                        ProtocolError::StreamRoleMismatch
                    });
                };
                self.validate_gossip(gossip, now_millis, authority)?;
                self.state.clone()
            }
            ProtocolState::Session { peer, fence } => {
                self.revalidate_peer(peer, now_millis, authority)?;
                let WireMessage::Session(session) = &frame.message else {
                    return Err(if matches!(&frame.message, WireMessage::Hello(_)) {
                        ProtocolError::DuplicateHello
                    } else {
                        ProtocolError::StreamRoleMismatch
                    });
                };
                let received_fence = match session {
                    SessionMessage::Data { fence, .. } | SessionMessage::Goodbye { fence, .. } => {
                        *fence
                    }
                };
                if received_fence != *fence {
                    return Err(ProtocolError::SessionFenceMismatch);
                }
                self.validate_session_fence(received_fence, authority)?;
                if matches!(session, SessionMessage::Goodbye { .. }) {
                    ProtocolState::Closed
                } else {
                    self.state.clone()
                }
            }
            ProtocolState::Closed => return Err(ProtocolError::StreamClosed),
        };

        if let WireMessage::Hello(hello) = &frame.message {
            self.membership_versions.insert(
                hello.membership.record.endpoint.id,
                hello.membership.record.record_version,
            );
        } else if let WireMessage::Gossip(GossipMessage::Delta { records, .. }) = &frame.message {
            for record in records {
                self.membership_versions
                    .insert(record.record.endpoint.id, record.record.record_version);
            }
        }
        {
            let mut replay_nonces = self
                .admission
                .replay_nonces
                .lock()
                .map_err(|_| ProtocolError::ProtocolStateUnavailable)?;
            replay_nonces.retain(|_, expires_at_millis| *expires_at_millis > now_millis);
            if replay_nonces.contains_key(&frame.nonce) {
                return Err(ProtocolError::Replay);
            }
            if replay_nonces.len() >= MAX_REPLAY_NONCES {
                return Err(ProtocolError::ReplayWindowFull);
            }
            replay_nonces.insert(frame.nonce, frame.expires_at_millis);
        }
        self.state = transition;
        Ok(frame)
    }

    fn validate_common(&self, frame: &MeshFrame, now_millis: u64) -> Result<(), ProtocolError> {
        if frame.deployment_id != self.admission.deployment_id {
            return Err(ProtocolError::DeploymentMismatch);
        }
        if frame.community_id != self.admission.community_id {
            return Err(ProtocolError::TenantMismatch);
        }
        if frame.sender_endpoint_id != self.admission.transport_peer_id {
            return Err(ProtocolError::TransportPeerMismatch);
        }
        validate_time_window(frame.issued_at_millis, frame.expires_at_millis, now_millis)?;
        Ok(())
    }

    fn revalidate_peer(
        &self,
        peer: &PeerMembershipRecord,
        now_millis: u64,
        authority: &impl MeshAuthority,
    ) -> Result<(), ProtocolError> {
        self.validate_membership_record(peer, now_millis, authority, false)
    }

    fn validate_membership(
        &self,
        attested: &AttestedPeerMembership,
        expected_endpoint_id: EndpointId,
        now_millis: u64,
        authority: &impl MeshAuthority,
        allow_revoked: bool,
    ) -> Result<(), ProtocolError> {
        if attested.record.endpoint.id != expected_endpoint_id {
            return Err(ProtocolError::TransportPeerMismatch);
        }
        let preimage = membership_attestation_preimage(&attested.record)?;
        self.admission
            .trust_root
            .verify(&preimage, &attested.signature)
            .map_err(|_| ProtocolError::InvalidAttestation)?;
        self.validate_membership_record(&attested.record, now_millis, authority, allow_revoked)
    }

    fn validate_membership_record(
        &self,
        record: &PeerMembershipRecord,
        now_millis: u64,
        authority: &impl MeshAuthority,
        allow_revoked: bool,
    ) -> Result<(), ProtocolError> {
        if record.deployment_id != self.admission.deployment_id {
            return Err(ProtocolError::DeploymentMismatch);
        }
        if record.community_id != self.admission.community_id {
            return Err(ProtocolError::TenantMismatch);
        }
        if record.runtime_generation == 0 || record.record_version == 0 {
            return Err(ProtocolError::InvalidGeneration);
        }
        validate_time_window(
            record.issued_at_millis,
            record.expires_at_millis,
            now_millis,
        )?;
        self.validate_endpoint(record)?;

        let membership = authority
            .membership(record.community_id, record.owner_principal_id)
            .map_err(|_| ProtocolError::AuthorityUnavailable)?
            .ok_or(ProtocolError::MissingMembership)?;
        if membership.community_id != record.community_id
            || membership.principal_id != record.owner_principal_id
        {
            return Err(ProtocolError::TenantMismatch);
        }
        if membership.version != record.membership_version {
            return Err(ProtocolError::StaleMembership);
        }
        match record.state {
            PeerMembershipState::Active | PeerMembershipState::Draining => {
                if membership.status != MembershipStatus::Active {
                    return Err(ProtocolError::RevokedMembership);
                }
                if record.state == PeerMembershipState::Draining && !allow_revoked {
                    return Err(ProtocolError::DrainingPeer);
                }
            }
            PeerMembershipState::Revoked => {
                if !allow_revoked || membership.status == MembershipStatus::Active {
                    return Err(ProtocolError::RevokedMembership);
                }
            }
        }

        let current_generation = authority
            .runtime_generation(record.community_id, record.endpoint.id)
            .map_err(|_| ProtocolError::AuthorityUnavailable)?
            .ok_or(ProtocolError::MissingRuntimeAuthority)?;
        if current_generation != record.runtime_generation {
            return Err(ProtocolError::RuntimeGenerationMismatch);
        }
        Ok(())
    }

    fn validate_endpoint(&self, record: &PeerMembershipRecord) -> Result<(), ProtocolError> {
        let transport_count = record.endpoint.addrs.len();
        if transport_count > MAX_ENDPOINT_TRANSPORTS {
            return Err(ProtocolError::TooManyEndpointTransports);
        }
        let ip_count = record.endpoint.ip_addrs().count();
        if ip_count > MAX_ENDPOINT_IP_ADDRESSES {
            return Err(ProtocolError::TooManyEndpointAddresses);
        }
        if record.state != PeerMembershipState::Revoked && transport_count == 0 {
            return Err(ProtocolError::MissingEndpointTransport);
        }
        if record.endpoint.addrs.iter().any(|transport| {
            matches!(transport, TransportAddr::Custom(_))
                || !self.admission.approved_transports.contains(transport)
        }) {
            return Err(ProtocolError::UnapprovedEndpointTransport);
        }
        Ok(())
    }

    fn validate_gossip(
        &self,
        gossip: &GossipMessage,
        now_millis: u64,
        authority: &impl MeshAuthority,
    ) -> Result<(), ProtocolError> {
        match gossip {
            GossipMessage::Digest { version, entries } => {
                validate_gossip_version(*version)?;
                if entries.len() > MAX_GOSSIP_RECORDS {
                    return Err(ProtocolError::TooManyGossipRecords);
                }
                let mut endpoints = HashSet::with_capacity(entries.len());
                for entry in entries {
                    if entry.record_version == 0 {
                        return Err(ProtocolError::InvalidGeneration);
                    }
                    if !endpoints.insert(entry.endpoint_id) {
                        return Err(ProtocolError::DuplicateGossipPeer);
                    }
                }
            }
            GossipMessage::Delta { version, records } => {
                validate_gossip_version(*version)?;
                if records.len() > MAX_GOSSIP_RECORDS {
                    return Err(ProtocolError::TooManyGossipRecords);
                }
                let mut endpoints = HashSet::with_capacity(records.len());
                let new_endpoint_count = records
                    .iter()
                    .filter(|record| {
                        !self
                            .membership_versions
                            .contains_key(&record.record.endpoint.id)
                    })
                    .count();
                if self
                    .membership_versions
                    .len()
                    .saturating_add(new_endpoint_count)
                    > MAX_TRACKED_MEMBERSHIPS
                {
                    return Err(ProtocolError::MembershipWindowFull);
                }
                for record in records {
                    let endpoint_id = record.record.endpoint.id;
                    if !endpoints.insert(endpoint_id) {
                        return Err(ProtocolError::DuplicateGossipPeer);
                    }
                    if self
                        .membership_versions
                        .get(&endpoint_id)
                        .is_some_and(|version| record.record.record_version <= *version)
                    {
                        return Err(ProtocolError::StaleGossipRecord);
                    }
                    self.validate_membership(record, endpoint_id, now_millis, authority, true)?;
                }
            }
        }
        Ok(())
    }

    fn validate_session_fence(
        &self,
        received: SessionFence,
        authority: &impl MeshAuthority,
    ) -> Result<(), ProtocolError> {
        if received.generation == 0 {
            return Err(ProtocolError::InvalidGeneration);
        }
        let current = authority
            .session_fence(self.admission.community_id, received.session_id)
            .map_err(|_| ProtocolError::AuthorityUnavailable)?
            .ok_or(ProtocolError::MissingSessionFence)?;
        if current != received {
            return Err(ProtocolError::SessionFenceMismatch);
        }
        Ok(())
    }
}

pub fn encode_frame(frame: &MeshFrame) -> Result<Vec<u8>, ProtocolError> {
    let frame_class = frame_class(&frame.message);
    let bytes = postcard::to_extend(frame, vec![WIRE_VERSION, frame_class])
        .map_err(|_| ProtocolError::MalformedFrame)?;
    validate_encoded_size(bytes.len(), frame_class)?;
    Ok(bytes)
}

pub fn decode_frame(bytes: &[u8]) -> Result<MeshFrame, ProtocolError> {
    if bytes.len() > MAX_STREAM_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let (version, remainder) = bytes.split_first().ok_or(ProtocolError::MalformedFrame)?;
    if *version != WIRE_VERSION {
        return Err(ProtocolError::UnsupportedWireVersion);
    }
    let (encoded_class, payload) = remainder
        .split_first()
        .ok_or(ProtocolError::MalformedFrame)?;
    validate_encoded_size(bytes.len(), *encoded_class)?;
    let frame: MeshFrame =
        postcard::from_bytes(payload).map_err(|_| ProtocolError::MalformedFrame)?;
    if frame_class(&frame.message) != *encoded_class {
        return Err(ProtocolError::FrameClassMismatch);
    }
    Ok(frame)
}

fn frame_class(message: &WireMessage) -> u8 {
    match message {
        WireMessage::Hello(_) | WireMessage::Gossip(_) => CONTROL_FRAME_CLASS,
        WireMessage::Session(_) => SESSION_FRAME_CLASS,
    }
}

fn validate_encoded_size(size: usize, frame_class: u8) -> Result<(), ProtocolError> {
    let maximum = match frame_class {
        CONTROL_FRAME_CLASS => MAX_CONTROL_FRAME_BYTES,
        SESSION_FRAME_CLASS => MAX_STREAM_FRAME_BYTES,
        _ => return Err(ProtocolError::MalformedFrame),
    };
    if size > maximum {
        return Err(if maximum == MAX_CONTROL_FRAME_BYTES {
            ProtocolError::ControlFrameTooLarge
        } else {
            ProtocolError::FrameTooLarge
        });
    }
    Ok(())
}

fn membership_attestation_preimage(
    record: &PeerMembershipRecord,
) -> Result<Vec<u8>, ProtocolError> {
    postcard::to_extend(record, MEMBERSHIP_ATTESTATION_CONTEXT.to_vec())
        .map_err(|_| ProtocolError::MalformedFrame)
}

fn validate_time_window(
    issued_at_millis: u64,
    expires_at_millis: u64,
    now_millis: u64,
) -> Result<(), ProtocolError> {
    let lifetime = expires_at_millis
        .checked_sub(issued_at_millis)
        .ok_or(ProtocolError::InvalidLifetime)?;
    if lifetime == 0 || lifetime > MAX_FRAME_LIFETIME_MILLIS {
        return Err(ProtocolError::InvalidLifetime);
    }
    if issued_at_millis > now_millis.saturating_add(MAX_CLOCK_SKEW_MILLIS) {
        return Err(ProtocolError::FrameFromFuture);
    }
    if expires_at_millis <= now_millis {
        return Err(ProtocolError::FrameExpired);
    }
    Ok(())
}

fn validate_gossip_version(version: u8) -> Result<(), ProtocolError> {
    if version != GOSSIP_VERSION {
        return Err(ProtocolError::UnsupportedGossipVersion);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ProtocolError {
    #[error("mesh deployment identity is invalid")]
    InvalidDeploymentIdentity,
    #[error("mesh frame nonce is invalid")]
    InvalidNonce,
    #[error("mesh trust root is not configured")]
    MissingTrustRoot,
    #[error("mesh transport policy is not configured")]
    MissingApprovedTransport,
    #[error("mesh frame is malformed")]
    MalformedFrame,
    #[error("mesh frame class does not match its body")]
    FrameClassMismatch,
    #[error("mesh wire version is unsupported")]
    UnsupportedWireVersion,
    #[error("mesh gossip version is unsupported")]
    UnsupportedGossipVersion,
    #[error("mesh frame exceeds its size limit")]
    FrameTooLarge,
    #[error("mesh control frame exceeds its size limit")]
    ControlFrameTooLarge,
    #[error("mesh frame lifetime is invalid")]
    InvalidLifetime,
    #[error("mesh frame is not yet valid")]
    FrameFromFuture,
    #[error("mesh frame has expired")]
    FrameExpired,
    #[error("mesh deployment does not match the trusted route")]
    DeploymentMismatch,
    #[error("mesh tenant does not match the trusted route")]
    TenantMismatch,
    #[error("mesh sender does not match the authenticated Iroh peer")]
    TransportPeerMismatch,
    #[error("mesh frame was replayed")]
    Replay,
    #[error("mesh replay window is full")]
    ReplayWindowFull,
    #[error("mesh protocol state is unavailable")]
    ProtocolStateUnavailable,
    #[error("mesh stream must begin with hello")]
    FirstFrameMustBeHello,
    #[error("mesh stream received a duplicate hello")]
    DuplicateHello,
    #[error("mesh stream role does not permit this frame")]
    StreamRoleMismatch,
    #[error("realtime media must use an Iroh datagram")]
    RealtimeMediaRequiresDatagram,
    #[error("mesh stream is closed")]
    StreamClosed,
    #[error("mesh peer attestation is invalid")]
    InvalidAttestation,
    #[error("canonical membership is unavailable")]
    MissingMembership,
    #[error("canonical membership is revoked")]
    RevokedMembership,
    #[error("mesh membership version is stale")]
    StaleMembership,
    #[error("mesh peer is draining")]
    DrainingPeer,
    #[error("mesh generation is invalid")]
    InvalidGeneration,
    #[error("canonical runtime authority is unavailable")]
    MissingRuntimeAuthority,
    #[error("mesh runtime generation does not match canonical authority")]
    RuntimeGenerationMismatch,
    #[error("mesh endpoint has no approved transport")]
    MissingEndpointTransport,
    #[error("mesh endpoint has too many direct addresses")]
    TooManyEndpointAddresses,
    #[error("mesh endpoint has too many transports")]
    TooManyEndpointTransports,
    #[error("mesh endpoint transport is not approved")]
    UnapprovedEndpointTransport,
    #[error("mesh gossip contains too many records")]
    TooManyGossipRecords,
    #[error("mesh gossip contains the same peer more than once")]
    DuplicateGossipPeer,
    #[error("mesh gossip record is stale or replayed")]
    StaleGossipRecord,
    #[error("mesh membership tracking window is full")]
    MembershipWindowFull,
    #[error("canonical mesh authority is unavailable")]
    AuthorityUnavailable,
    #[error("canonical session fence is unavailable")]
    MissingSessionFence,
    #[error("mesh session fence does not match canonical authority")]
    SessionFenceMismatch,
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, net::SocketAddr};

    use collaboration_domain::{MembershipRole, TrustedTenantRoute};

    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    #[derive(Default)]
    struct TestAuthority {
        memberships: HashMap<(CommunityId, PrincipalId), CommunityMembership>,
        runtime_generations: HashMap<(CommunityId, EndpointId), u64>,
        session_fences: HashMap<(CommunityId, Uuid), SessionFence>,
        unavailable: bool,
    }

    impl MeshAuthority for TestAuthority {
        fn membership(
            &self,
            community_id: CommunityId,
            principal_id: PrincipalId,
        ) -> Result<Option<CommunityMembership>, MeshAuthorityUnavailable> {
            if self.unavailable {
                return Err(MeshAuthorityUnavailable);
            }
            Ok(self.memberships.get(&(community_id, principal_id)).copied())
        }

        fn runtime_generation(
            &self,
            community_id: CommunityId,
            endpoint_id: EndpointId,
        ) -> Result<Option<u64>, MeshAuthorityUnavailable> {
            if self.unavailable {
                return Err(MeshAuthorityUnavailable);
            }
            Ok(self
                .runtime_generations
                .get(&(community_id, endpoint_id))
                .copied())
        }

        fn session_fence(
            &self,
            community_id: CommunityId,
            session_id: Uuid,
        ) -> Result<Option<SessionFence>, MeshAuthorityUnavailable> {
            if self.unavailable {
                return Err(MeshAuthorityUnavailable);
            }
            Ok(self
                .session_fences
                .get(&(community_id, session_id))
                .copied())
        }
    }

    struct Fixture {
        community_id: CommunityId,
        principal_id: PrincipalId,
        deployment_id: DeploymentId,
        peer_secret: SecretKey,
        trust_secret: SecretKey,
        transport: TransportAddr,
        authority: TestAuthority,
    }

    impl Fixture {
        fn new() -> Self {
            let community_id = CommunityId::from_uuid(Uuid::from_u128(1));
            let principal_id = PrincipalId::from_uuid(Uuid::from_u128(2));
            let peer_secret = SecretKey::from_bytes(&[3; 32]);
            let trust_secret = SecretKey::from_bytes(&[4; 32]);
            let transport = TransportAddr::Ip(
                "203.0.113.10:443"
                    .parse::<SocketAddr>()
                    .expect("valid test address"),
            );
            let membership = CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            };
            let mut authority = TestAuthority::default();
            authority
                .memberships
                .insert((community_id, principal_id), membership);
            authority
                .runtime_generations
                .insert((community_id, peer_secret.public()), 7);
            Self {
                community_id,
                principal_id,
                deployment_id: DeploymentId::from_bytes([5; 32])
                    .expect("valid deployment identity"),
                peer_secret,
                trust_secret,
                transport,
                authority,
            }
        }

        fn admission(&self) -> MeshAdmissionContext {
            let tenant = TenantContext::establish(
                Some(
                    TrustedTenantRoute::from_deployment(self.community_id, "test/deployment")
                        .expect("valid route"),
                ),
                &[],
            )
            .expect("trusted tenant");
            MeshAdmissionContext::new(
                &tenant,
                self.deployment_id,
                self.peer_secret.public(),
                Some(self.trust_secret.public()),
                BTreeSet::from([self.transport.clone()]),
            )
            .expect("valid admission")
        }

        fn protocol(&self) -> MeshProtocol {
            MeshProtocol::new(self.admission())
        }

        fn record(&self, record_version: u64, state: PeerMembershipState) -> PeerMembershipRecord {
            PeerMembershipRecord {
                deployment_id: self.deployment_id,
                community_id: self.community_id,
                owner_principal_id: self.principal_id,
                endpoint: EndpointAddr::from_parts(
                    self.peer_secret.public(),
                    [self.transport.clone()],
                ),
                runtime_generation: 7,
                membership_version: AggregateVersion::FIRST,
                state,
                record_version,
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
            }
        }

        fn frame(&self, nonce: u8, message: WireMessage) -> MeshFrame {
            MeshFrame {
                deployment_id: self.deployment_id,
                community_id: self.community_id,
                sender_endpoint_id: self.peer_secret.public(),
                runtime_generation: 7,
                nonce: FrameNonce::from_bytes([nonce; 16]).expect("nonzero nonce"),
                issued_at_millis: NOW - 1_000,
                expires_at_millis: NOW + 44_000,
                message,
            }
        }

        fn hello(&self, nonce: u8) -> MeshFrame {
            let membership = AttestedPeerMembership::sign(
                self.record(1, PeerMembershipState::Active),
                &self.trust_secret,
            )
            .expect("attestation");
            self.frame(
                nonce,
                WireMessage::Hello(StreamHello {
                    membership,
                    role: StreamRole::Control,
                }),
            )
        }

        fn gossip(&self, nonce: u8, gossip: GossipMessage) -> MeshFrame {
            self.frame(nonce, WireMessage::Gossip(gossip))
        }
    }

    #[test]
    fn mesh_protocol_rejects_unsupported_wire_and_gossip_versions() {
        let fixture = Fixture::new();
        let mut protocol = fixture.protocol();
        let mut hello = encode_frame(&fixture.hello(1)).expect("encode hello");
        hello[0] = WIRE_VERSION + 1;
        assert_eq!(
            protocol.accept_frame(&hello, NOW, &fixture.authority),
            Err(ProtocolError::UnsupportedWireVersion)
        );

        protocol
            .accept_frame(
                &encode_frame(&fixture.hello(2)).expect("encode hello"),
                NOW,
                &fixture.authority,
            )
            .expect("supported hello");
        let gossip = fixture.gossip(
            3,
            GossipMessage::Digest {
                version: GOSSIP_VERSION + 1,
                entries: Vec::new(),
            },
        );
        assert_eq!(
            protocol.accept_frame(
                &encode_frame(&gossip).expect("encode gossip"),
                NOW,
                &fixture.authority,
            ),
            Err(ProtocolError::UnsupportedGossipVersion)
        );
    }

    #[test]
    fn mesh_protocol_rejects_replayed_frame_across_peer_streams() {
        let fixture = Fixture::new();
        let admission = fixture.admission();
        let mut first_stream = MeshProtocol::new(admission.clone());
        let mut second_stream = MeshProtocol::new(admission);
        let hello = encode_frame(&fixture.hello(1)).expect("encode hello");
        first_stream
            .accept_frame(&hello, NOW, &fixture.authority)
            .expect("first delivery");
        assert_eq!(
            second_stream.accept_frame(&hello, NOW, &fixture.authority),
            Err(ProtocolError::Replay)
        );
    }

    #[test]
    fn mesh_protocol_rejects_revoked_membership_and_cross_tenant_frames() {
        let mut fixture = Fixture::new();
        fixture
            .authority
            .memberships
            .get_mut(&(fixture.community_id, fixture.principal_id))
            .expect("membership")
            .status = MembershipStatus::Revoked;
        let mut protocol = fixture.protocol();
        assert_eq!(
            protocol.accept_frame(
                &encode_frame(&fixture.hello(1)).expect("encode hello"),
                NOW,
                &fixture.authority,
            ),
            Err(ProtocolError::RevokedMembership)
        );

        let fixture = Fixture::new();
        let mut protocol = fixture.protocol();
        let mut foreign = fixture.hello(2);
        foreign.community_id = CommunityId::from_uuid(Uuid::from_u128(99));
        assert_eq!(
            protocol.accept_frame(
                &encode_frame(&foreign).expect("encode foreign frame"),
                NOW,
                &fixture.authority,
            ),
            Err(ProtocolError::TenantMismatch)
        );
    }

    #[test]
    fn mesh_protocol_partition_fails_closed_and_does_not_burn_nonce() {
        let mut fixture = Fixture::new();
        let mut protocol = fixture.protocol();
        protocol
            .accept_frame(
                &encode_frame(&fixture.hello(1)).expect("encode hello"),
                NOW,
                &fixture.authority,
            )
            .expect("hello");
        let gossip = fixture.gossip(
            2,
            GossipMessage::Digest {
                version: GOSSIP_VERSION,
                entries: Vec::new(),
            },
        );
        let encoded = encode_frame(&gossip).expect("encode gossip");
        fixture.authority.unavailable = true;
        assert_eq!(
            protocol.accept_frame(&encoded, NOW, &fixture.authority),
            Err(ProtocolError::AuthorityUnavailable)
        );
        fixture.authority.unavailable = false;
        protocol
            .accept_frame(&encoded, NOW, &fixture.authority)
            .expect("same frame is admissible after authority recovery");
    }

    #[test]
    fn mesh_protocol_rejects_malformed_duplicate_and_stale_gossip() {
        let fixture = Fixture::new();
        let mut protocol = fixture.protocol();
        assert_eq!(
            protocol.accept_frame(&[WIRE_VERSION, 0xff], NOW, &fixture.authority),
            Err(ProtocolError::MalformedFrame)
        );
        let mut oversized_control = vec![0; MAX_CONTROL_FRAME_BYTES + 1];
        oversized_control[0] = WIRE_VERSION;
        oversized_control[1] = CONTROL_FRAME_CLASS;
        assert_eq!(
            protocol.accept_frame(&oversized_control, NOW, &fixture.authority),
            Err(ProtocolError::ControlFrameTooLarge)
        );
        protocol
            .accept_frame(
                &encode_frame(&fixture.hello(1)).expect("encode hello"),
                NOW,
                &fixture.authority,
            )
            .expect("hello");
        let entry = GossipDigestEntry {
            endpoint_id: fixture.peer_secret.public(),
            record_version: 2,
        };
        let duplicate = fixture.gossip(
            2,
            GossipMessage::Digest {
                version: GOSSIP_VERSION,
                entries: vec![entry, entry],
            },
        );
        assert_eq!(
            protocol.accept_frame(
                &encode_frame(&duplicate).expect("encode duplicate digest"),
                NOW,
                &fixture.authority,
            ),
            Err(ProtocolError::DuplicateGossipPeer)
        );

        let replayed_record = AttestedPeerMembership::sign(
            fixture.record(1, PeerMembershipState::Active),
            &fixture.trust_secret,
        )
        .expect("attestation");
        let stale = fixture.gossip(
            3,
            GossipMessage::Delta {
                version: GOSSIP_VERSION,
                records: vec![replayed_record],
            },
        );
        assert_eq!(
            protocol.accept_frame(
                &encode_frame(&stale).expect("encode stale delta"),
                NOW,
                &fixture.authority,
            ),
            Err(ProtocolError::StaleGossipRecord)
        );
    }

    #[test]
    fn mesh_protocol_checks_the_canonical_session_fence_on_every_frame() {
        let mut fixture = Fixture::new();
        let fence = SessionFence {
            session_id: Uuid::from_u128(50),
            generation: 4,
            owner_endpoint_id: fixture.peer_secret.public(),
        };
        fixture
            .authority
            .session_fences
            .insert((fixture.community_id, fence.session_id), fence);
        let membership = AttestedPeerMembership::sign(
            fixture.record(1, PeerMembershipState::Active),
            &fixture.trust_secret,
        )
        .expect("attestation");
        let hello = fixture.frame(
            1,
            WireMessage::Hello(StreamHello {
                membership,
                role: StreamRole::Session {
                    fence,
                    profile: SessionProfile::SharedCompute,
                },
            }),
        );
        let mut protocol = fixture.protocol();
        protocol
            .accept_frame(
                &encode_frame(&hello).expect("encode hello"),
                NOW,
                &fixture.authority,
            )
            .expect("session hello");

        fixture.authority.session_fences.insert(
            (fixture.community_id, fence.session_id),
            SessionFence {
                generation: fence.generation + 1,
                ..fence
            },
        );
        let data = fixture.frame(
            2,
            WireMessage::Session(SessionMessage::Data {
                fence,
                payload: b"opaque".to_vec(),
            }),
        );
        assert_eq!(
            protocol.accept_frame(
                &encode_frame(&data).expect("encode data"),
                NOW,
                &fixture.authority,
            ),
            Err(ProtocolError::SessionFenceMismatch)
        );
    }
}
