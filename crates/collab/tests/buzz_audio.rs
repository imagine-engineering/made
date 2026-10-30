use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
};

use collab::huddle::buzz_audio::{
    BUZZ_AUDIO_FLAG_DTX, BuzzAudioBridgeError, BuzzAudioCompatibilityAdapter,
    BuzzAudioCompatibilityPhase, BuzzAudioCompatibilityWindow, BuzzAudioFrame,
    BuzzAudioGatewayError, BuzzAudioNativeBridge, BuzzAudioProtocolVersion,
    MAX_BUZZ_AUDIO_FRAME_BYTES,
};
use collaboration_domain::{
    AggregateId, CommunityId, Huddle, HuddleCommandOutcome, HuddleGeneration, HuddleIdentity,
    HuddleLifecycleState, HuddleParticipantRole, OperationId, PrincipalId,
};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Observation {
    Attached {
        principal_id: PrincipalId,
        peer_index: u8,
        version: BuzzAudioProtocolVersion,
    },
    Frame {
        principal_id: PrincipalId,
        peer_index: u8,
        payload: Vec<u8>,
        header: Option<(u16, u32, i8, u8)>,
    },
    Detached {
        principal_id: PrincipalId,
        peer_index: u8,
    },
    Closed,
}

#[derive(Default)]
struct RecordingBridge {
    observations: Vec<Observation>,
    fail_attach: bool,
    fail_detach_indices: BTreeSet<u8>,
    fail_close: bool,
}

impl BuzzAudioNativeBridge for RecordingBridge {
    fn attach_peer(
        &mut self,
        _identity: HuddleIdentity,
        principal_id: PrincipalId,
        peer_index: u8,
        protocol_version: BuzzAudioProtocolVersion,
    ) -> Result<(), BuzzAudioBridgeError> {
        if self.fail_attach {
            return Err(BuzzAudioBridgeError);
        }
        self.observations.push(Observation::Attached {
            principal_id,
            peer_index,
            version: protocol_version,
        });
        Ok(())
    }

    fn forward_opus(
        &mut self,
        _identity: HuddleIdentity,
        principal_id: PrincipalId,
        frame: BuzzAudioFrame<'_>,
    ) -> Result<(), BuzzAudioBridgeError> {
        let header = frame.header().map(|header| {
            (
                header.sequence(),
                header.timestamp_48khz(),
                header.level_dbov(),
                header.flags(),
            )
        });
        self.observations.push(Observation::Frame {
            principal_id,
            peer_index: frame.peer_index(),
            payload: frame.opus_payload().to_vec(),
            header,
        });
        Ok(())
    }

    fn detach_peer(
        &mut self,
        _identity: HuddleIdentity,
        principal_id: PrincipalId,
        peer_index: u8,
    ) -> Result<(), BuzzAudioBridgeError> {
        if self.fail_detach_indices.contains(&peer_index) {
            return Err(BuzzAudioBridgeError);
        }
        self.observations.push(Observation::Detached {
            principal_id,
            peer_index,
        });
        Ok(())
    }

    fn close_generation(&mut self, _identity: HuddleIdentity) -> Result<(), BuzzAudioBridgeError> {
        if self.fail_close {
            return Err(BuzzAudioBridgeError);
        }
        self.observations.push(Observation::Closed);
        Ok(())
    }
}

struct BuzzLifecycleOracle {
    pinned_version: Option<BuzzAudioProtocolVersion>,
    peers: BTreeMap<PrincipalId, u8>,
    observations: Vec<Observation>,
}

impl BuzzLifecycleOracle {
    fn new() -> Self {
        Self {
            pinned_version: None,
            peers: BTreeMap::new(),
            observations: Vec::new(),
        }
    }

    fn join(&mut self, principal_id: PrincipalId, version: BuzzAudioProtocolVersion) {
        assert!(self.pinned_version.is_none_or(|pinned| pinned == version));
        if self.peers.contains_key(&principal_id) {
            return;
        }
        let peer_index = (u8::MIN..=u8::MAX)
            .find(|candidate| !self.peers.values().any(|index| index == candidate))
            .expect("oracle peer index");
        self.pinned_version.get_or_insert(version);
        self.peers.insert(principal_id, peer_index);
        self.observations.push(Observation::Attached {
            principal_id,
            peer_index,
            version,
        });
    }

    fn frame(&mut self, principal_id: PrincipalId, payload: &[u8]) {
        let peer_index = self.peers[&principal_id];
        self.observations.push(Observation::Frame {
            principal_id,
            peer_index,
            payload: payload.to_vec(),
            header: None,
        });
    }

    fn leave(&mut self, principal_id: PrincipalId) {
        let peer_index = self.peers.remove(&principal_id).expect("oracle peer");
        self.observations.push(Observation::Detached {
            principal_id,
            peer_index,
        });
    }

    fn end(&mut self) {
        let peers = std::mem::take(&mut self.peers);
        for (principal_id, peer_index) in peers {
            self.observations.push(Observation::Detached {
                principal_id,
                peer_index,
            });
        }
        self.observations.push(Observation::Closed);
    }
}

fn operation() -> OperationId {
    OperationId::new()
}

fn window(revision: u64, phase: BuzzAudioCompatibilityPhase) -> BuzzAudioCompatibilityWindow {
    BuzzAudioCompatibilityWindow::new(NonZeroU64::new(revision).expect("window revision"), phase)
}

fn huddle() -> (Huddle, PrincipalId) {
    let owner = PrincipalId::new();
    let identity = HuddleIdentity::new(
        CommunityId::new(),
        AggregateId::new(),
        AggregateId::new(),
        HuddleGeneration::new(1).expect("generation"),
    )
    .expect("identity");
    (
        Huddle::start(identity, owner, operation(), 1).expect("start huddle"),
        owner,
    )
}

#[test]
fn buzz_v1_and_canonical_gateway_emit_the_same_lifecycle_observations() {
    let (mut huddle, owner) = huddle();
    let speaker = PrincipalId::new();
    let mut oracle = BuzzLifecycleOracle::new();
    oracle.join(owner, BuzzAudioProtocolVersion::V1);
    oracle.join(speaker, BuzzAudioProtocolVersion::V1);
    oracle.frame(speaker, b"opus-v1");
    oracle.leave(speaker);
    oracle.end();

    let mut adapter = BuzzAudioCompatibilityAdapter::new(
        &huddle,
        window(1, BuzzAudioCompatibilityPhase::Migration),
        RecordingBridge::default(),
    )
    .expect("adapter");
    assert_eq!(
        adapter
            .join(
                &mut huddle,
                owner,
                HuddleParticipantRole::Owner,
                None,
                operation(),
                2,
            )
            .expect("owner join")
            .canonical_outcome(),
        HuddleCommandOutcome::Unchanged
    );
    assert_eq!(
        adapter
            .join(
                &mut huddle,
                speaker,
                HuddleParticipantRole::Speaker,
                Some(1),
                operation(),
                3,
            )
            .expect("speaker join")
            .canonical_outcome(),
        HuddleCommandOutcome::Applied
    );
    assert_eq!(
        adapter.forward_frame(&huddle, speaker, b"opus-v1"),
        Ok(None)
    );
    assert_eq!(
        adapter
            .leave(&mut huddle, speaker, operation(), 4)
            .expect("speaker leave")
            .canonical_outcome(),
        HuddleCommandOutcome::Applied
    );
    let cleanup = adapter
        .end(&mut huddle, owner, operation(), 5)
        .expect("end gateway");
    assert_eq!(cleanup.detached_peers(), 1);
    assert!(cleanup.generation_close_attempted());
    assert_eq!(adapter.bridge().observations, oracle.observations);
    assert!(matches!(
        huddle.lifecycle(),
        HuddleLifecycleState::Ended { .. }
    ));
}

#[test]
fn v2_frames_match_buzz_header_rules_and_invalid_frames_never_bridge() {
    let (mut huddle, owner) = huddle();
    let mut adapter = BuzzAudioCompatibilityAdapter::new(
        &huddle,
        window(1, BuzzAudioCompatibilityPhase::Migration),
        RecordingBridge::default(),
    )
    .expect("adapter");
    adapter
        .join(
            &mut huddle,
            owner,
            HuddleParticipantRole::Owner,
            Some(2),
            operation(),
            2,
        )
        .expect("v2 join");

    let mut frame = vec![
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x7f,
        BUZZ_AUDIO_FLAG_DTX,
    ];
    frame.extend_from_slice(b"opus-v2");
    let header = adapter
        .forward_frame(&huddle, owner, &frame)
        .expect("valid v2 frame")
        .expect("v2 header");
    assert_eq!(header.sequence(), 0x0102);
    assert_eq!(header.timestamp_48khz(), 0x0304_0506);
    assert_eq!(header.level_dbov(), -127);
    assert!(header.is_dtx());
    assert_eq!(
        adapter.bridge().observations.last(),
        Some(&Observation::Frame {
            principal_id: owner,
            peer_index: 0,
            payload: b"opus-v2".to_vec(),
            header: Some((0x0102, 0x0304_0506, -127, BUZZ_AUDIO_FLAG_DTX)),
        })
    );

    let observation_count = adapter.bridge().observations.len();
    assert_eq!(
        adapter.forward_frame(&huddle, owner, &[0; 8]),
        Err(BuzzAudioGatewayError::InvalidFrame)
    );
    assert_eq!(
        adapter.forward_frame(&huddle, owner, &vec![0; MAX_BUZZ_AUDIO_FRAME_BYTES + 1]),
        Err(BuzzAudioGatewayError::InvalidFrame)
    );
    assert_eq!(adapter.bridge().observations.len(), observation_count);
}

#[test]
fn default_and_pinned_versions_preserve_buzz_compatibility_errors() {
    let (mut huddle, owner) = huddle();
    let speaker = PrincipalId::new();
    let mut adapter = BuzzAudioCompatibilityAdapter::new(
        &huddle,
        window(1, BuzzAudioCompatibilityPhase::Migration),
        RecordingBridge::default(),
    )
    .expect("adapter");
    let owner_join = adapter
        .join(
            &mut huddle,
            owner,
            HuddleParticipantRole::Owner,
            None,
            operation(),
            2,
        )
        .expect("default v1 join");
    assert_eq!(
        owner_join.peer().protocol_version(),
        BuzzAudioProtocolVersion::V1
    );
    assert_eq!(
        adapter.join(
            &mut huddle,
            speaker,
            HuddleParticipantRole::Speaker,
            Some(2),
            operation(),
            3,
        ),
        Err(BuzzAudioGatewayError::UpgradeRequired {
            pinned: BuzzAudioProtocolVersion::V1,
            requested: BuzzAudioProtocolVersion::V2,
        })
    );
    assert!(huddle.participant(speaker).is_none());
    assert_eq!(
        adapter.join(
            &mut huddle,
            speaker,
            HuddleParticipantRole::Speaker,
            Some(0),
            operation(),
            3,
        ),
        Err(BuzzAudioGatewayError::UnsupportedVersion)
    );
    assert_eq!(
        adapter.join(
            &mut huddle,
            speaker,
            HuddleParticipantRole::Speaker,
            Some(3),
            operation(),
            3,
        ),
        Err(BuzzAudioGatewayError::UnsupportedVersion)
    );
}

#[test]
fn compatibility_window_drains_existing_sessions_before_retirement() {
    let (mut huddle, owner) = huddle();
    let speaker = PrincipalId::new();
    let mut adapter = BuzzAudioCompatibilityAdapter::new(
        &huddle,
        window(1, BuzzAudioCompatibilityPhase::Migration),
        RecordingBridge::default(),
    )
    .expect("adapter");
    adapter
        .join(
            &mut huddle,
            owner,
            HuddleParticipantRole::Owner,
            Some(1),
            operation(),
            2,
        )
        .expect("owner join");
    adapter
        .update_compatibility_window(window(2, BuzzAudioCompatibilityPhase::DrainOnly))
        .expect("start drain");

    let duplicate = adapter
        .join(
            &mut huddle,
            owner,
            HuddleParticipantRole::Owner,
            Some(1),
            operation(),
            3,
        )
        .expect("existing retry remains admitted");
    assert_eq!(
        duplicate.canonical_outcome(),
        HuddleCommandOutcome::Unchanged
    );
    assert_eq!(
        adapter.join(
            &mut huddle,
            speaker,
            HuddleParticipantRole::Speaker,
            Some(1),
            operation(),
            3,
        ),
        Err(BuzzAudioGatewayError::CompatibilityCreationDisabled)
    );
    assert_eq!(adapter.forward_frame(&huddle, owner, b"draining"), Ok(None));
    assert_eq!(
        adapter.update_compatibility_window(window(3, BuzzAudioCompatibilityPhase::Retired)),
        Err(BuzzAudioGatewayError::CompatibilitySessionsActive)
    );
    adapter
        .end(&mut huddle, owner, operation(), 4)
        .expect("end drained generation");
    adapter
        .update_compatibility_window(window(3, BuzzAudioCompatibilityPhase::Retired))
        .expect("retire after drain");
    assert_eq!(
        adapter.update_compatibility_window(window(4, BuzzAudioCompatibilityPhase::Migration)),
        Err(BuzzAudioGatewayError::InvalidCompatibilityWindow)
    );
}

#[test]
fn bridge_failures_retain_canonical_state_and_cleanup_remains_visible() {
    let (mut huddle, _owner) = huddle();
    let speaker = PrincipalId::new();
    let mut bridge = RecordingBridge::default();
    bridge.fail_attach = true;
    let mut adapter = BuzzAudioCompatibilityAdapter::new(
        &huddle,
        window(1, BuzzAudioCompatibilityPhase::Migration),
        bridge,
    )
    .expect("adapter");
    let join_operation = operation();
    assert_eq!(
        adapter.join(
            &mut huddle,
            speaker,
            HuddleParticipantRole::Speaker,
            Some(2),
            join_operation,
            2,
        ),
        Err(BuzzAudioGatewayError::NativeBridgeUnavailable)
    );
    assert!(huddle.participant(speaker).is_some());
    assert!(adapter.roster().peers().is_empty());

    adapter.bridge_mut().fail_attach = false;
    let retry = adapter
        .join(
            &mut huddle,
            speaker,
            HuddleParticipantRole::Speaker,
            Some(2),
            join_operation,
            2,
        )
        .expect("bridge retry");
    assert_eq!(retry.canonical_outcome(), HuddleCommandOutcome::Unchanged);
    adapter
        .bridge_mut()
        .fail_detach_indices
        .insert(retry.peer().peer_index());
    let leave = adapter
        .leave(&mut huddle, speaker, operation(), 3)
        .expect("canonical leave");
    assert_eq!(leave.canonical_outcome(), HuddleCommandOutcome::Applied);
    assert_eq!(leave.cleanup().detached_peers(), 0);
    assert_eq!(
        leave.cleanup().failed_peer_detaches(),
        &[retry.peer().peer_index()]
    );
    assert!(adapter.roster().peers().is_empty());
}
