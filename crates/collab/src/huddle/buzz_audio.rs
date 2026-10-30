use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    num::NonZeroU64,
};

use collaboration_domain::{
    Huddle, HuddleCommandOutcome, HuddleError, HuddleIdentity, HuddleLifecycleState,
    HuddleParticipantPresence, HuddleParticipantRole, OperationId, PrincipalId,
};

pub const BUZZ_AUDIO_PROTOCOL_V1: u8 = 1;
pub const BUZZ_AUDIO_PROTOCOL_V2: u8 = 2;
pub const MAX_BUZZ_AUDIO_FRAME_BYTES: usize = 4_096;
pub const MAX_BUZZ_AUDIO_TEXT_BYTES: usize = 8_192;
pub const MAX_BUZZ_AUDIO_PEERS: usize = 25;
pub const BUZZ_AUDIO_V2_HEADER_BYTES: usize = 8;
pub const BUZZ_AUDIO_FLAG_DTX: u8 = 0x01;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BuzzAudioCompatibilityPhase {
    Migration,
    DrainOnly,
    Retired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAudioCompatibilityWindow {
    revision: NonZeroU64,
    phase: BuzzAudioCompatibilityPhase,
}

impl BuzzAudioCompatibilityWindow {
    pub const fn new(revision: NonZeroU64, phase: BuzzAudioCompatibilityPhase) -> Self {
        Self { revision, phase }
    }

    pub const fn revision(self) -> NonZeroU64 {
        self.revision
    }

    pub const fn phase(self) -> BuzzAudioCompatibilityPhase {
        self.phase
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BuzzAudioProtocolVersion {
    V1,
    V2,
}

impl BuzzAudioProtocolVersion {
    pub fn negotiate(requested: Option<u8>) -> Result<Self, BuzzAudioGatewayError> {
        match requested.unwrap_or(BUZZ_AUDIO_PROTOCOL_V1) {
            BUZZ_AUDIO_PROTOCOL_V1 => Ok(Self::V1),
            BUZZ_AUDIO_PROTOCOL_V2 => Ok(Self::V2),
            _ => Err(BuzzAudioGatewayError::UnsupportedVersion),
        }
    }

    pub const fn wire_value(self) -> u8 {
        match self {
            Self::V1 => BUZZ_AUDIO_PROTOCOL_V1,
            Self::V2 => BUZZ_AUDIO_PROTOCOL_V2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAudioV2Header {
    sequence: u16,
    timestamp_48khz: u32,
    level_dbov: i8,
    flags: u8,
}

impl BuzzAudioV2Header {
    pub const fn sequence(self) -> u16 {
        self.sequence
    }

    pub const fn timestamp_48khz(self) -> u32 {
        self.timestamp_48khz
    }

    pub const fn level_dbov(self) -> i8 {
        self.level_dbov
    }

    pub const fn flags(self) -> u8 {
        self.flags
    }

    pub const fn is_dtx(self) -> bool {
        self.flags & BUZZ_AUDIO_FLAG_DTX != 0
    }

    fn parse(frame: &[u8]) -> Result<(Self, &[u8]), BuzzAudioGatewayError> {
        if frame.len() <= BUZZ_AUDIO_V2_HEADER_BYTES {
            return Err(BuzzAudioGatewayError::InvalidFrame);
        }
        let raw_level = frame[6] as i8;
        let level_dbov = if (-127..=0).contains(&raw_level) {
            raw_level
        } else {
            -127
        };
        Ok((
            Self {
                sequence: u16::from_be_bytes([frame[0], frame[1]]),
                timestamp_48khz: u32::from_be_bytes([frame[2], frame[3], frame[4], frame[5]]),
                level_dbov,
                flags: frame[7],
            },
            &frame[BUZZ_AUDIO_V2_HEADER_BYTES..],
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAudioFrame<'a> {
    protocol_version: BuzzAudioProtocolVersion,
    peer_index: u8,
    header: Option<BuzzAudioV2Header>,
    opus_payload: &'a [u8],
}

impl<'a> BuzzAudioFrame<'a> {
    pub const fn protocol_version(self) -> BuzzAudioProtocolVersion {
        self.protocol_version
    }

    pub const fn peer_index(self) -> u8 {
        self.peer_index
    }

    pub const fn header(self) -> Option<BuzzAudioV2Header> {
        self.header
    }

    pub const fn opus_payload(self) -> &'a [u8] {
        self.opus_payload
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("Buzz audio native bridge failed")]
pub struct BuzzAudioBridgeError;

pub trait BuzzAudioNativeBridge {
    fn attach_peer(
        &mut self,
        identity: HuddleIdentity,
        principal_id: PrincipalId,
        peer_index: u8,
        protocol_version: BuzzAudioProtocolVersion,
    ) -> Result<(), BuzzAudioBridgeError>;

    fn forward_opus(
        &mut self,
        identity: HuddleIdentity,
        principal_id: PrincipalId,
        frame: BuzzAudioFrame<'_>,
    ) -> Result<(), BuzzAudioBridgeError>;

    fn detach_peer(
        &mut self,
        identity: HuddleIdentity,
        principal_id: PrincipalId,
        peer_index: u8,
    ) -> Result<(), BuzzAudioBridgeError>;

    fn close_generation(&mut self, identity: HuddleIdentity) -> Result<(), BuzzAudioBridgeError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAudioPeer {
    principal_id: PrincipalId,
    peer_index: u8,
    protocol_version: BuzzAudioProtocolVersion,
}

impl BuzzAudioPeer {
    pub const fn principal_id(self) -> PrincipalId {
        self.principal_id
    }

    pub const fn peer_index(self) -> u8 {
        self.peer_index
    }

    pub const fn protocol_version(self) -> BuzzAudioProtocolVersion {
        self.protocol_version
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzAudioRoster {
    revision: u64,
    peers: Vec<BuzzAudioPeer>,
}

impl BuzzAudioRoster {
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub fn peers(&self) -> &[BuzzAudioPeer] {
        &self.peers
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuzzAudioJoinOutcome {
    canonical_outcome: HuddleCommandOutcome,
    peer: BuzzAudioPeer,
    roster_revision: u64,
}

impl BuzzAudioJoinOutcome {
    pub const fn canonical_outcome(self) -> HuddleCommandOutcome {
        self.canonical_outcome
    }

    pub const fn peer(self) -> BuzzAudioPeer {
        self.peer
    }

    pub const fn roster_revision(self) -> u64 {
        self.roster_revision
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BuzzAudioCleanupReport {
    detached_peers: usize,
    failed_peer_detaches: Vec<u8>,
    generation_close_attempted: bool,
    generation_close_failed: bool,
}

impl BuzzAudioCleanupReport {
    pub const fn detached_peers(&self) -> usize {
        self.detached_peers
    }

    pub fn failed_peer_detaches(&self) -> &[u8] {
        &self.failed_peer_detaches
    }

    pub const fn generation_close_attempted(&self) -> bool {
        self.generation_close_attempted
    }

    pub const fn generation_close_failed(&self) -> bool {
        self.generation_close_failed
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuzzAudioLeaveOutcome {
    canonical_outcome: HuddleCommandOutcome,
    roster_revision: u64,
    cleanup: BuzzAudioCleanupReport,
}

impl BuzzAudioLeaveOutcome {
    pub const fn canonical_outcome(&self) -> HuddleCommandOutcome {
        self.canonical_outcome
    }

    pub const fn roster_revision(&self) -> u64 {
        self.roster_revision
    }

    pub const fn cleanup(&self) -> &BuzzAudioCleanupReport {
        &self.cleanup
    }
}

pub struct BuzzAudioCompatibilityAdapter<Bridge: BuzzAudioNativeBridge> {
    identity: HuddleIdentity,
    compatibility_window: BuzzAudioCompatibilityWindow,
    pinned_version: Option<BuzzAudioProtocolVersion>,
    roster_revision: u64,
    peers: BTreeMap<PrincipalId, BuzzAudioPeer>,
    used_peer_indices: BTreeSet<u8>,
    generation_closed: bool,
    bridge: Bridge,
}

impl<Bridge> Drop for BuzzAudioCompatibilityAdapter<Bridge>
where
    Bridge: BuzzAudioNativeBridge,
{
    fn drop(&mut self) {
        let mut report = BuzzAudioCleanupReport::default();
        self.close_all_peers(&mut report);
        for peer_index in report.failed_peer_detaches {
            log::error!(
                "failed to detach Buzz compatibility peer {peer_index} while dropping adapter"
            );
        }
    }
}

impl<Bridge> BuzzAudioCompatibilityAdapter<Bridge>
where
    Bridge: BuzzAudioNativeBridge,
{
    pub fn new(
        huddle: &Huddle,
        compatibility_window: BuzzAudioCompatibilityWindow,
        bridge: Bridge,
    ) -> Result<Self, BuzzAudioGatewayError> {
        if !matches!(huddle.lifecycle(), HuddleLifecycleState::Active) {
            return Err(BuzzAudioGatewayError::HuddleEnded);
        }
        Ok(Self {
            identity: huddle.identity(),
            compatibility_window,
            pinned_version: None,
            roster_revision: 0,
            peers: BTreeMap::new(),
            used_peer_indices: BTreeSet::new(),
            generation_closed: false,
            bridge,
        })
    }

    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn compatibility_window(&self) -> BuzzAudioCompatibilityWindow {
        self.compatibility_window
    }

    pub const fn pinned_version(&self) -> Option<BuzzAudioProtocolVersion> {
        self.pinned_version
    }

    pub fn bridge(&self) -> &Bridge {
        &self.bridge
    }

    pub fn bridge_mut(&mut self) -> &mut Bridge {
        &mut self.bridge
    }

    pub fn roster(&self) -> BuzzAudioRoster {
        let mut peers: Vec<_> = self.peers.values().copied().collect();
        peers.sort_by_key(|peer| peer.peer_index);
        BuzzAudioRoster {
            revision: self.roster_revision,
            peers,
        }
    }

    pub fn update_compatibility_window(
        &mut self,
        next: BuzzAudioCompatibilityWindow,
    ) -> Result<(), BuzzAudioGatewayError> {
        if next.revision <= self.compatibility_window.revision
            || next.phase < self.compatibility_window.phase
        {
            return Err(BuzzAudioGatewayError::InvalidCompatibilityWindow);
        }
        if next.phase == BuzzAudioCompatibilityPhase::Retired && !self.peers.is_empty() {
            return Err(BuzzAudioGatewayError::CompatibilitySessionsActive);
        }
        self.compatibility_window = next;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn join(
        &mut self,
        huddle: &mut Huddle,
        principal_id: PrincipalId,
        role: HuddleParticipantRole,
        requested_version: Option<u8>,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<BuzzAudioJoinOutcome, BuzzAudioGatewayError> {
        self.validate_active_huddle(huddle)?;
        let protocol_version = BuzzAudioProtocolVersion::negotiate(requested_version)?;
        if let Some(existing) = self.peers.get(&principal_id).copied() {
            if existing.protocol_version != protocol_version {
                return Err(BuzzAudioGatewayError::UpgradeRequired {
                    pinned: existing.protocol_version,
                    requested: protocol_version,
                });
            }
            let canonical_outcome = join_canonical_participant(
                huddle,
                principal_id,
                role,
                operation_id,
                occurred_at_millis,
            )?;
            return Ok(BuzzAudioJoinOutcome {
                canonical_outcome,
                peer: existing,
                roster_revision: self.roster_revision,
            });
        }
        if self.compatibility_window.phase != BuzzAudioCompatibilityPhase::Migration {
            return Err(BuzzAudioGatewayError::CompatibilityCreationDisabled);
        }
        if self.peers.len() >= MAX_BUZZ_AUDIO_PEERS {
            return Err(BuzzAudioGatewayError::RoomFull);
        }
        if let Some(pinned) = self.pinned_version
            && pinned != protocol_version
        {
            return Err(BuzzAudioGatewayError::UpgradeRequired {
                pinned,
                requested: protocol_version,
            });
        }
        let next_revision = self
            .roster_revision
            .checked_add(1)
            .ok_or(BuzzAudioGatewayError::RosterRevisionExhausted)?;
        let peer_index = self.allocate_peer_index()?;
        let canonical_outcome = join_canonical_participant(
            huddle,
            principal_id,
            role,
            operation_id,
            occurred_at_millis,
        )?;
        if self
            .bridge
            .attach_peer(self.identity, principal_id, peer_index, protocol_version)
            .is_err()
        {
            return Err(BuzzAudioGatewayError::NativeBridgeUnavailable);
        }
        let peer = BuzzAudioPeer {
            principal_id,
            peer_index,
            protocol_version,
        };
        self.pinned_version.get_or_insert(protocol_version);
        self.used_peer_indices.insert(peer_index);
        self.peers.insert(principal_id, peer);
        self.roster_revision = next_revision;
        Ok(BuzzAudioJoinOutcome {
            canonical_outcome,
            peer,
            roster_revision: next_revision,
        })
    }

    pub fn forward_frame(
        &mut self,
        huddle: &Huddle,
        principal_id: PrincipalId,
        frame: &[u8],
    ) -> Result<Option<BuzzAudioV2Header>, BuzzAudioGatewayError> {
        self.validate_active_huddle(huddle)?;
        let peer = self
            .peers
            .get(&principal_id)
            .copied()
            .ok_or(BuzzAudioGatewayError::PeerNotAdmitted)?;
        if frame.is_empty() || frame.len() > MAX_BUZZ_AUDIO_FRAME_BYTES {
            return Err(BuzzAudioGatewayError::InvalidFrame);
        }
        let (header, opus_payload) = match peer.protocol_version {
            BuzzAudioProtocolVersion::V1 => (None, frame),
            BuzzAudioProtocolVersion::V2 => {
                let (header, payload) = BuzzAudioV2Header::parse(frame)?;
                (Some(header), payload)
            }
        };
        self.bridge
            .forward_opus(
                self.identity,
                principal_id,
                BuzzAudioFrame {
                    protocol_version: peer.protocol_version,
                    peer_index: peer.peer_index,
                    header,
                    opus_payload,
                },
            )
            .map_err(|_| BuzzAudioGatewayError::NativeBridgeUnavailable)?;
        Ok(header)
    }

    pub fn leave(
        &mut self,
        huddle: &mut Huddle,
        principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<BuzzAudioLeaveOutcome, BuzzAudioGatewayError> {
        self.validate_huddle(huddle)?;
        let canonical_outcome = huddle.leave(principal_id, operation_id, occurred_at_millis)?;
        let next_revision = self
            .roster_revision
            .checked_add(u64::from(self.peers.contains_key(&principal_id)))
            .ok_or(BuzzAudioGatewayError::RosterRevisionExhausted)?;
        let mut cleanup = BuzzAudioCleanupReport::default();
        if let Some(peer) = self.remove_peer(principal_id) {
            if self
                .bridge
                .detach_peer(self.identity, principal_id, peer.peer_index)
                .is_err()
            {
                cleanup.failed_peer_detaches.push(peer.peer_index);
            } else {
                cleanup.detached_peers = 1;
            }
            self.roster_revision = next_revision;
        }
        if matches!(huddle.lifecycle(), HuddleLifecycleState::Ended { .. }) {
            self.close_all_peers(&mut cleanup);
            self.close_generation(&mut cleanup);
        }
        Ok(BuzzAudioLeaveOutcome {
            canonical_outcome,
            roster_revision: self.roster_revision,
            cleanup,
        })
    }

    pub fn end(
        &mut self,
        huddle: &mut Huddle,
        actor_principal_id: PrincipalId,
        operation_id: OperationId,
        occurred_at_millis: u64,
    ) -> Result<BuzzAudioCleanupReport, BuzzAudioGatewayError> {
        self.validate_huddle(huddle)?;
        huddle.end(actor_principal_id, operation_id, occurred_at_millis)?;
        let mut cleanup = BuzzAudioCleanupReport::default();
        self.close_all_peers(&mut cleanup);
        self.close_generation(&mut cleanup);
        Ok(cleanup)
    }

    fn validate_huddle(&self, huddle: &Huddle) -> Result<(), BuzzAudioGatewayError> {
        if huddle.identity() != self.identity {
            return Err(BuzzAudioGatewayError::WrongHuddle);
        }
        Ok(())
    }

    fn validate_active_huddle(&self, huddle: &Huddle) -> Result<(), BuzzAudioGatewayError> {
        self.validate_huddle(huddle)?;
        if !matches!(huddle.lifecycle(), HuddleLifecycleState::Active) {
            return Err(BuzzAudioGatewayError::HuddleEnded);
        }
        Ok(())
    }

    fn allocate_peer_index(&self) -> Result<u8, BuzzAudioGatewayError> {
        (u8::MIN..=u8::MAX)
            .find(|index| !self.used_peer_indices.contains(index))
            .ok_or(BuzzAudioGatewayError::RoomFull)
    }

    fn remove_peer(&mut self, principal_id: PrincipalId) -> Option<BuzzAudioPeer> {
        let peer = self.peers.remove(&principal_id)?;
        self.used_peer_indices.remove(&peer.peer_index);
        Some(peer)
    }

    fn close_all_peers(&mut self, report: &mut BuzzAudioCleanupReport) {
        let peers = std::mem::take(&mut self.peers);
        self.used_peer_indices.clear();
        for (principal_id, peer) in peers {
            if self
                .bridge
                .detach_peer(self.identity, principal_id, peer.peer_index)
                .is_err()
            {
                report.failed_peer_detaches.push(peer.peer_index);
            } else {
                report.detached_peers += 1;
            }
        }
    }

    fn close_generation(&mut self, report: &mut BuzzAudioCleanupReport) {
        if self.generation_closed {
            return;
        }
        report.generation_close_attempted = true;
        report.generation_close_failed = self.bridge.close_generation(self.identity).is_err();
        self.generation_closed = !report.generation_close_failed;
    }
}

fn join_canonical_participant(
    huddle: &mut Huddle,
    principal_id: PrincipalId,
    role: HuddleParticipantRole,
    operation_id: OperationId,
    occurred_at_millis: u64,
) -> Result<HuddleCommandOutcome, HuddleError> {
    if principal_id == huddle.owner_principal_id() {
        if operation_id.as_uuid().is_nil() {
            return Err(HuddleError::InvalidOperation);
        }
        if occurred_at_millis == 0 {
            return Err(HuddleError::InvalidTimestamp);
        }
        let participant = huddle
            .participant(principal_id)
            .ok_or(HuddleError::ParticipantNotFound)?;
        if participant.presence() != HuddleParticipantPresence::Present {
            return Err(HuddleError::ParticipantNotPresent);
        }
        return if role == HuddleParticipantRole::Owner {
            Ok(HuddleCommandOutcome::Unchanged)
        } else {
            Err(HuddleError::ParticipantConflict)
        };
    }
    huddle.join(principal_id, role, operation_id, occurred_at_millis)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuzzAudioGatewayError {
    WrongHuddle,
    HuddleEnded,
    UnsupportedVersion,
    UpgradeRequired {
        pinned: BuzzAudioProtocolVersion,
        requested: BuzzAudioProtocolVersion,
    },
    CompatibilityCreationDisabled,
    InvalidCompatibilityWindow,
    CompatibilitySessionsActive,
    RoomFull,
    PeerNotAdmitted,
    InvalidFrame,
    NativeBridgeUnavailable,
    RosterRevisionExhausted,
    Canonical(HuddleError),
}

impl From<HuddleError> for BuzzAudioGatewayError {
    fn from(error: HuddleError) -> Self {
        Self::Canonical(error)
    }
}

impl fmt::Display for BuzzAudioGatewayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::WrongHuddle => "Buzz audio huddle scope does not match",
            Self::HuddleEnded => "Buzz audio huddle has ended",
            Self::UnsupportedVersion => "Buzz audio protocol version is unsupported",
            Self::UpgradeRequired { .. } => "Buzz audio room uses another protocol version",
            Self::CompatibilityCreationDisabled => {
                "Buzz audio compatibility no longer admits new sessions"
            }
            Self::InvalidCompatibilityWindow => "Buzz audio compatibility window is invalid",
            Self::CompatibilitySessionsActive => {
                "Buzz audio compatibility sessions must drain before retirement"
            }
            Self::RoomFull => "Buzz audio compatibility room is full",
            Self::PeerNotAdmitted => "Buzz audio peer is not admitted",
            Self::InvalidFrame => "Buzz audio frame is invalid",
            Self::NativeBridgeUnavailable => "Buzz audio native bridge is unavailable",
            Self::RosterRevisionExhausted => "Buzz audio roster revision is exhausted",
            Self::Canonical(_) => "Buzz audio canonical lifecycle command failed",
        })
    }
}

impl Error for BuzzAudioGatewayError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Canonical(error) => Some(error),
            _ => None,
        }
    }
}
