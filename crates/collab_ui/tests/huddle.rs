#![cfg(feature = "multiplayer-tools")]

use std::{cell::RefCell, collections::VecDeque, num::NonZeroU64, rc::Rc};

use audio::{
    NativeHuddleCallbackScope, NativeHuddleParticipantIdentity, NativeHuddleRoomName,
    NativeHuddleTransportAdapter,
};
use collab::huddle::buzz_audio::{
    BuzzAudioBridgeError, BuzzAudioCompatibilityAdapter, BuzzAudioCompatibilityPhase,
    BuzzAudioCompatibilityWindow, BuzzAudioFrame, BuzzAudioNativeBridge, BuzzAudioProtocolVersion,
};
use collab_ui::{
    huddle::{
        HuddleParticipantDisplay, HuddleTranscriptDisplay, HuddleTranscriptDisplayState,
        HuddleWorkspaceAvailability, HuddleWorkspaceFailureReason, HuddleWorkspaceOutcome,
        HuddleWorkspaceRequest, HuddleWorkspaceSnapshot, HuddleWorkspaceView,
        NativeHuddleWorkspaceService,
    },
    huddle_controls::{
        HuddleAudioDeviceKind, HuddleAudioTransportError, HuddleControlOutcome, HuddleControlsView,
        NativeHuddleAudioControlTransport,
    },
};
use collaboration_domain::{
    AggregateId, CommunityId, Huddle, HuddleEvent, HuddleGeneration, HuddleIdentity,
    HuddleParticipantRole, HuddleTranscriptReference, HuddleTranscriptSegmentId, OperationId,
    PrincipalId,
};
use gpui::{AppContext as _, TestAppContext};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClientPath {
    Native,
    Buzz,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InteropObservation {
    TransportReady,
    DeviceFailureRetained,
    DeviceRetryApplied,
    TranscriptDisplayed,
    NetworkFailureRetained,
    NetworkRetryApplied,
    LeaveFailureRetained,
    LeaveRetryApplied,
    TransportClosed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InteropTrace {
    canonical_events: Vec<HuddleEvent>,
    observations: Vec<InteropObservation>,
}

#[derive(Default)]
struct RecordingBuzzBridge {
    attached: Vec<PrincipalId>,
    forwarded_frames: usize,
    detached: Vec<PrincipalId>,
    closed: bool,
}

impl BuzzAudioNativeBridge for RecordingBuzzBridge {
    fn attach_peer(
        &mut self,
        _identity: HuddleIdentity,
        principal_id: PrincipalId,
        _peer_index: u8,
        _protocol_version: BuzzAudioProtocolVersion,
    ) -> Result<(), BuzzAudioBridgeError> {
        self.attached.push(principal_id);
        Ok(())
    }

    fn forward_opus(
        &mut self,
        _identity: HuddleIdentity,
        _principal_id: PrincipalId,
        _frame: BuzzAudioFrame<'_>,
    ) -> Result<(), BuzzAudioBridgeError> {
        self.forwarded_frames += 1;
        Ok(())
    }

    fn detach_peer(
        &mut self,
        _identity: HuddleIdentity,
        principal_id: PrincipalId,
        _peer_index: u8,
    ) -> Result<(), BuzzAudioBridgeError> {
        self.detached.push(principal_id);
        Ok(())
    }

    fn close_generation(&mut self, _identity: HuddleIdentity) -> Result<(), BuzzAudioBridgeError> {
        self.closed = true;
        Ok(())
    }
}

#[derive(Default)]
struct AudioControlState {
    fail_next: Option<HuddleAudioTransportError>,
    microphone_updates: Vec<bool>,
}

struct RecordingAudioTransport(Rc<RefCell<AudioControlState>>);

impl NativeHuddleAudioControlTransport for RecordingAudioTransport {
    fn select_device(
        &mut self,
        _scope: &NativeHuddleCallbackScope,
        _kind: HuddleAudioDeviceKind,
        _device_id: Option<&collab_ui::huddle_controls::HuddleAudioDeviceId>,
    ) -> Result<(), HuddleAudioTransportError> {
        Ok(())
    }

    fn set_microphone_enabled(
        &mut self,
        _scope: &NativeHuddleCallbackScope,
        enabled: bool,
    ) -> Result<(), HuddleAudioTransportError> {
        let mut state = self.0.borrow_mut();
        if let Some(error) = state.fail_next.take() {
            return Err(error);
        }
        state.microphone_updates.push(enabled);
        Ok(())
    }

    fn set_playback_enabled(
        &mut self,
        _scope: &NativeHuddleCallbackScope,
        _enabled: bool,
    ) -> Result<(), HuddleAudioTransportError> {
        Ok(())
    }
}

struct QueueWorkspaceService(
    VecDeque<Result<HuddleWorkspaceSnapshot, HuddleWorkspaceFailureReason>>,
);

impl NativeHuddleWorkspaceService for QueueWorkspaceService {
    fn perform(
        &mut self,
        _request: HuddleWorkspaceRequest,
    ) -> Result<HuddleWorkspaceSnapshot, HuddleWorkspaceFailureReason> {
        self.0
            .pop_front()
            .unwrap_or(Err(HuddleWorkspaceFailureReason::ServiceUnavailable))
    }
}

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn operation(value: u128) -> OperationId {
    OperationId::from_uuid(Uuid::from_u128(value))
}

fn identity() -> HuddleIdentity {
    HuddleIdentity::new(
        CommunityId::from_uuid(Uuid::from_u128(1)),
        aggregate(2),
        aggregate(3),
        HuddleGeneration::new(4).expect("generation"),
    )
    .expect("identity")
}

fn participant_displays(owner: PrincipalId, speaker: PrincipalId) -> Vec<HuddleParticipantDisplay> {
    vec![
        HuddleParticipantDisplay::new(owner, "Owner").expect("owner display"),
        HuddleParticipantDisplay::new(speaker, "Speaker").expect("speaker display"),
    ]
}

fn transcript(speaker: PrincipalId) -> HuddleTranscriptDisplay {
    HuddleTranscriptDisplay::new(
        HuddleTranscriptSegmentId::new(aggregate(30)).expect("segment"),
        aggregate(31),
        speaker,
        HuddleTranscriptDisplayState::Final,
        Some("transport-neutral transcript".into()),
        300,
        340,
    )
    .expect("transcript display")
}

fn snapshot(
    huddle: Huddle,
    owner: PrincipalId,
    speaker: PrincipalId,
    network: HuddleWorkspaceAvailability,
) -> HuddleWorkspaceSnapshot {
    HuddleWorkspaceSnapshot::new(
        huddle,
        participant_displays(owner, speaker),
        vec![transcript(speaker)],
        network,
        HuddleWorkspaceAvailability::Ready,
        HuddleWorkspaceAvailability::Ready,
    )
    .expect("workspace snapshot")
}

fn scope(identity: HuddleIdentity) -> NativeHuddleCallbackScope {
    NativeHuddleCallbackScope::from_livekit(NativeHuddleRoomName::for_huddle(identity).as_str(), 1)
        .expect("native callback scope")
}

fn exercise_audio_controls(
    identity: HuddleIdentity,
    cx: &mut TestAppContext,
) -> gpui::Entity<HuddleControlsView> {
    let state = Rc::new(RefCell::new(AudioControlState {
        fail_next: Some(HuddleAudioTransportError::DeviceUnavailable),
        microphone_updates: Vec::new(),
    }));
    let controls = cx.new(|_| {
        HuddleControlsView::new(
            identity,
            scope(identity),
            Vec::new(),
            true,
            false,
            RecordingAudioTransport(state.clone()),
        )
        .expect("audio controls")
    });
    assert_eq!(
        controls.update(cx, HuddleControlsView::toggle_mute),
        HuddleControlOutcome::Failed
    );
    assert!(controls.read_with(cx, |controls, _| controls.is_muted()));
    assert_eq!(
        controls.update(cx, HuddleControlsView::retry_failed),
        HuddleControlOutcome::Applied
    );
    assert!(!controls.read_with(cx, |controls, _| controls.is_muted()));
    assert_eq!(state.borrow().microphone_updates, vec![true]);
    controls
}

fn audit_trace(trace: &InteropTrace) -> Result<(), &'static str> {
    let required = [
        InteropObservation::TransportReady,
        InteropObservation::DeviceFailureRetained,
        InteropObservation::DeviceRetryApplied,
        InteropObservation::TranscriptDisplayed,
        InteropObservation::NetworkFailureRetained,
        InteropObservation::NetworkRetryApplied,
        InteropObservation::LeaveFailureRetained,
        InteropObservation::LeaveRetryApplied,
        InteropObservation::TransportClosed,
    ];
    if trace.observations != required {
        return Err("incomplete or reordered interoperability observations");
    }
    if !matches!(
        trace.canonical_events.first(),
        Some(HuddleEvent::Started { .. })
    ) || !matches!(
        trace.canonical_events.get(1),
        Some(HuddleEvent::ParticipantJoined { .. })
    ) || !matches!(
        trace.canonical_events.get(2),
        Some(HuddleEvent::TranscriptLinked { .. })
    ) || !matches!(
        trace.canonical_events.get(3),
        Some(HuddleEvent::ParticipantLeft { .. })
    ) || !matches!(
        trace.canonical_events.get(4),
        Some(HuddleEvent::Ended { .. })
    ) || trace.canonical_events.len() != 5
    {
        return Err("canonical lifecycle trace diverged");
    }
    Ok(())
}

fn run_path(path: ClientPath, cx: &mut TestAppContext) -> InteropTrace {
    let identity = identity();
    let owner = principal(10);
    let speaker = principal(11);
    let mut huddle = Huddle::start(identity, owner, operation(20), 100).expect("start huddle");
    let mut native = None;
    let mut buzz = None;

    match path {
        ClientPath::Native => {
            huddle
                .join(speaker, HuddleParticipantRole::Speaker, operation(21), 200)
                .expect("native join");
            let mut adapter = NativeHuddleTransportAdapter::new(&huddle, speaker)
                .expect("native transport adapter");
            let request = adapter
                .begin_connect(&huddle)
                .expect("begin native connect");
            let remote = NativeHuddleParticipantIdentity::for_participant(identity, owner);
            let sync = adapter
                .finish_connect(&request.callback_scope(), &huddle, [remote])
                .expect("finish native connect");
            assert_eq!(sync.present(), &[owner]);
            native = Some(adapter);
        }
        ClientPath::Buzz => {
            let mut adapter = BuzzAudioCompatibilityAdapter::new(
                &huddle,
                BuzzAudioCompatibilityWindow::new(
                    NonZeroU64::new(1).expect("compatibility revision"),
                    BuzzAudioCompatibilityPhase::Migration,
                ),
                RecordingBuzzBridge::default(),
            )
            .expect("Buzz adapter");
            adapter
                .join(
                    &mut huddle,
                    owner,
                    HuddleParticipantRole::Owner,
                    Some(1),
                    operation(29),
                    150,
                )
                .expect("Buzz owner join");
            adapter
                .join(
                    &mut huddle,
                    speaker,
                    HuddleParticipantRole::Speaker,
                    Some(1),
                    operation(21),
                    200,
                )
                .expect("Buzz speaker join");
            adapter
                .forward_frame(&huddle, speaker, b"bounded-opus")
                .expect("Buzz audio frame");
            assert_eq!(adapter.bridge().forwarded_frames, 1);
            buzz = Some(adapter);
        }
    }

    let mut observations = vec![InteropObservation::TransportReady];
    let controls = exercise_audio_controls(identity, cx);
    observations.extend([
        InteropObservation::DeviceFailureRetained,
        InteropObservation::DeviceRetryApplied,
    ]);

    let reference = HuddleTranscriptReference::new(
        identity,
        HuddleTranscriptSegmentId::new(aggregate(30)).expect("segment"),
        aggregate(31),
        speaker,
        300,
        340,
    )
    .expect("transcript reference");
    huddle
        .link_transcript(reference, operation(22), 350)
        .expect("link transcript");
    let healthy = snapshot(
        huddle.clone(),
        owner,
        speaker,
        HuddleWorkspaceAvailability::Ready,
    );
    assert_eq!(
        healthy.transcripts()[0].text(),
        Some("transport-neutral transcript")
    );
    observations.push(InteropObservation::TranscriptDisplayed);

    let unavailable = snapshot(
        huddle.clone(),
        owner,
        speaker,
        HuddleWorkspaceAvailability::Failed(HuddleWorkspaceFailureReason::NetworkUnavailable),
    );
    let network_view = cx.new(|cx| {
        HuddleWorkspaceView::new(
            identity,
            speaker,
            Some(unavailable),
            Some(controls.clone()),
            QueueWorkspaceService(VecDeque::from([Ok(healthy.clone())])),
            cx,
        )
        .expect("network workspace")
    });
    let before_network_retry = network_view
        .read_with(cx, |view, _| view.snapshot().cloned())
        .expect("network snapshot");
    assert_eq!(
        network_view.update(cx, HuddleWorkspaceView::retry_network),
        HuddleWorkspaceOutcome::Applied
    );
    let after_network_retry = network_view
        .read_with(cx, |view, _| view.snapshot().cloned())
        .expect("recovered network snapshot");
    assert_eq!(before_network_retry.huddle(), after_network_retry.huddle());
    observations.extend([
        InteropObservation::NetworkFailureRetained,
        InteropObservation::NetworkRetryApplied,
    ]);

    let mut left = huddle.clone();
    match path {
        ClientPath::Native => {
            let adapter = native.as_mut().expect("native path adapter");
            left.leave(speaker, operation(23), 400)
                .expect("native leave");
            let cleanup = adapter.cancel();
            assert_eq!(cleanup.attempted(), 0);
        }
        ClientPath::Buzz => {
            buzz.as_mut()
                .expect("Buzz path adapter")
                .leave(&mut left, speaker, operation(23), 400)
                .expect("Buzz leave");
        }
    }
    let left_snapshot = snapshot(
        left.clone(),
        owner,
        speaker,
        HuddleWorkspaceAvailability::Ready,
    );
    let leave_view = cx.new(|cx| {
        HuddleWorkspaceView::new(
            identity,
            speaker,
            Some(healthy),
            Some(controls),
            QueueWorkspaceService(VecDeque::from([
                Err(HuddleWorkspaceFailureReason::NetworkUnavailable),
                Ok(left_snapshot),
            ])),
            cx,
        )
        .expect("leave workspace")
    });
    let before_failed_leave = leave_view
        .read_with(cx, |view, _| view.snapshot().cloned())
        .expect("pre-leave snapshot");
    assert_eq!(
        leave_view.update(cx, HuddleWorkspaceView::leave),
        HuddleWorkspaceOutcome::Failed
    );
    assert_eq!(
        leave_view.read_with(cx, |view, _| view.snapshot().cloned()),
        Some(before_failed_leave)
    );
    assert_eq!(
        leave_view.update(cx, HuddleWorkspaceView::retry_failed),
        HuddleWorkspaceOutcome::Applied
    );
    observations.extend([
        InteropObservation::LeaveFailureRetained,
        InteropObservation::LeaveRetryApplied,
    ]);

    match path {
        ClientPath::Native => {
            left.end(owner, operation(24), 500).expect("native end");
            native
                .as_mut()
                .expect("native path adapter")
                .end(&left)
                .expect("close native transport");
        }
        ClientPath::Buzz => {
            let adapter = buzz.as_mut().expect("Buzz path adapter");
            adapter
                .end(&mut left, owner, operation(24), 500)
                .expect("end Buzz transport");
            assert!(adapter.bridge().closed);
        }
    }
    observations.push(InteropObservation::TransportClosed);

    InteropTrace {
        canonical_events: left.fields().events.clone(),
        observations,
    }
}

#[gpui::test]
fn native_and_buzz_huddles_emit_equivalent_canonical_and_recovery_traces(cx: &mut TestAppContext) {
    let native = run_path(ClientPath::Native, cx);
    let buzz = run_path(ClientPath::Buzz, cx);
    audit_trace(&native).expect("native interoperability trace");
    audit_trace(&buzz).expect("Buzz interoperability trace");
    assert_eq!(native, buzz);
}

#[test]
fn interoperability_checker_rejects_missing_or_divergent_observations() {
    let events = Huddle::start(identity(), principal(10), operation(20), 100)
        .expect("huddle")
        .fields()
        .events
        .clone();
    let mut missing = InteropTrace {
        canonical_events: events,
        observations: vec![InteropObservation::TransportReady],
    };
    assert_eq!(
        audit_trace(&missing),
        Err("incomplete or reordered interoperability observations")
    );
    missing.observations = vec![
        InteropObservation::TransportReady,
        InteropObservation::DeviceFailureRetained,
        InteropObservation::DeviceRetryApplied,
        InteropObservation::TranscriptDisplayed,
        InteropObservation::NetworkFailureRetained,
        InteropObservation::NetworkRetryApplied,
        InteropObservation::LeaveFailureRetained,
        InteropObservation::LeaveRetryApplied,
        InteropObservation::TransportClosed,
    ];
    assert_eq!(
        audit_trace(&missing),
        Err("canonical lifecycle trace diverged")
    );
}
