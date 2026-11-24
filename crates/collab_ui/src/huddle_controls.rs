use std::{collections::BTreeSet, error::Error, fmt};

use audio::{NativeHuddleCallbackScope, NativeHuddleRoomName};
use collaboration_domain::HuddleIdentity;
use gpui::{Context, IntoElement, Render, Role, Window};
use ui::{Button, ButtonStyle, prelude::*};

const MAX_DEVICE_ID_BYTES: usize = 255;
const MAX_DEVICE_LABEL_BYTES: usize = 128;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HuddleAudioDeviceId(String);

impl HuddleAudioDeviceId {
    pub fn new(value: impl Into<String>) -> Result<Self, HuddleControlsError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_DEVICE_ID_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(HuddleControlsError::InvalidDevice);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HuddleAudioDeviceKind {
    Microphone,
    Speaker,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleAudioDevice {
    id: HuddleAudioDeviceId,
    label: String,
    kind: HuddleAudioDeviceKind,
}

impl HuddleAudioDevice {
    pub fn new(
        id: HuddleAudioDeviceId,
        label: impl Into<String>,
        kind: HuddleAudioDeviceKind,
    ) -> Result<Self, HuddleControlsError> {
        let label = label.into();
        let label = label.trim();
        if label.is_empty()
            || label.len() > MAX_DEVICE_LABEL_BYTES
            || label.chars().any(char::is_control)
        {
            return Err(HuddleControlsError::InvalidDevice);
        }
        Ok(Self {
            id,
            label: label.to_owned(),
            kind,
        })
    }

    pub const fn id(&self) -> &HuddleAudioDeviceId {
        &self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn kind(&self) -> HuddleAudioDeviceKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleAudioTransportError {
    PermissionDenied,
    DeviceUnavailable,
    ConnectionUnavailable,
    StaleAttempt,
}

pub trait NativeHuddleAudioControlTransport: 'static {
    fn select_device(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        kind: HuddleAudioDeviceKind,
        device_id: Option<&HuddleAudioDeviceId>,
    ) -> Result<(), HuddleAudioTransportError>;

    fn set_microphone_enabled(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        enabled: bool,
    ) -> Result<(), HuddleAudioTransportError>;

    fn set_playback_enabled(
        &mut self,
        scope: &NativeHuddleCallbackScope,
        enabled: bool,
    ) -> Result<(), HuddleAudioTransportError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleControlFunction {
    Microphone,
    Speaker,
    MicrophoneDevice,
    SpeakerDevice,
}

impl HuddleControlFunction {
    const fn label(self) -> &'static str {
        match self {
            Self::Microphone => "Microphone control failed",
            Self::Speaker => "Speaker control failed",
            Self::MicrophoneDevice => "Microphone selection failed",
            Self::SpeakerDevice => "Speaker selection failed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HuddleControlCommand {
    SetMuted(bool),
    SetSpeakerMuted(bool),
    SetPushToTalkEnabled(bool),
    SetPushToTalkHeld(bool),
    SelectDevice {
        kind: HuddleAudioDeviceKind,
        device_id: Option<HuddleAudioDeviceId>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleControlFailure {
    function: HuddleControlFunction,
    reason: HuddleAudioTransportError,
    command: HuddleControlCommand,
}

impl HuddleControlFailure {
    pub const fn function(&self) -> HuddleControlFunction {
        self.function
    }

    pub const fn reason(&self) -> HuddleAudioTransportError {
        self.reason
    }

    pub const fn retryable(&self) -> bool {
        true
    }

    pub const fn fallback_available(&self) -> bool {
        matches!(self.command, HuddleControlCommand::SelectDevice { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleControlOutcome {
    Applied,
    Unchanged,
    Failed,
}

pub struct HuddleControlsView {
    identity: HuddleIdentity,
    scope: NativeHuddleCallbackScope,
    transport: Box<dyn NativeHuddleAudioControlTransport>,
    devices: Vec<HuddleAudioDevice>,
    selected_microphone: Option<HuddleAudioDeviceId>,
    selected_speaker: Option<HuddleAudioDeviceId>,
    muted: bool,
    speaker_muted: bool,
    push_to_talk_enabled: bool,
    push_to_talk_held: bool,
    failure: Option<HuddleControlFailure>,
}

impl HuddleControlsView {
    pub fn new(
        identity: HuddleIdentity,
        scope: NativeHuddleCallbackScope,
        devices: Vec<HuddleAudioDevice>,
        initially_muted: bool,
        initially_speaker_muted: bool,
        transport: impl NativeHuddleAudioControlTransport,
    ) -> Result<Self, HuddleControlsError> {
        if scope.room_name() != &NativeHuddleRoomName::for_huddle(identity) {
            return Err(HuddleControlsError::WrongNativeRoom);
        }
        validate_devices(&devices)?;
        Ok(Self {
            identity,
            scope,
            transport: Box::new(transport),
            devices,
            selected_microphone: None,
            selected_speaker: None,
            muted: initially_muted,
            speaker_muted: initially_speaker_muted,
            push_to_talk_enabled: false,
            push_to_talk_held: false,
            failure: None,
        })
    }

    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn scope(&self) -> &NativeHuddleCallbackScope {
        &self.scope
    }

    pub fn devices(&self) -> &[HuddleAudioDevice] {
        &self.devices
    }

    pub const fn selected_microphone(&self) -> Option<&HuddleAudioDeviceId> {
        self.selected_microphone.as_ref()
    }

    pub const fn selected_speaker(&self) -> Option<&HuddleAudioDeviceId> {
        self.selected_speaker.as_ref()
    }

    pub const fn is_muted(&self) -> bool {
        self.muted
    }

    pub const fn is_speaker_muted(&self) -> bool {
        self.speaker_muted
    }

    pub const fn push_to_talk_enabled(&self) -> bool {
        self.push_to_talk_enabled
    }

    pub const fn push_to_talk_held(&self) -> bool {
        self.push_to_talk_held
    }

    pub const fn microphone_enabled(&self) -> bool {
        microphone_enabled(
            self.muted,
            self.push_to_talk_enabled,
            self.push_to_talk_held,
        )
    }

    pub const fn failure(&self) -> Option<&HuddleControlFailure> {
        self.failure.as_ref()
    }

    pub fn toggle_mute(&mut self, cx: &mut Context<Self>) -> HuddleControlOutcome {
        self.apply(HuddleControlCommand::SetMuted(!self.muted), cx)
    }

    pub fn toggle_speaker_mute(&mut self, cx: &mut Context<Self>) -> HuddleControlOutcome {
        self.apply(
            HuddleControlCommand::SetSpeakerMuted(!self.speaker_muted),
            cx,
        )
    }

    pub fn set_push_to_talk_enabled(
        &mut self,
        enabled: bool,
        cx: &mut Context<Self>,
    ) -> HuddleControlOutcome {
        self.apply(HuddleControlCommand::SetPushToTalkEnabled(enabled), cx)
    }

    pub fn set_push_to_talk_held(
        &mut self,
        held: bool,
        cx: &mut Context<Self>,
    ) -> HuddleControlOutcome {
        self.apply(HuddleControlCommand::SetPushToTalkHeld(held), cx)
    }

    pub fn select_device(
        &mut self,
        kind: HuddleAudioDeviceKind,
        device_id: Option<HuddleAudioDeviceId>,
        cx: &mut Context<Self>,
    ) -> HuddleControlOutcome {
        if device_id.as_ref().is_some_and(|device_id| {
            !self
                .devices
                .iter()
                .any(|device| device.kind == kind && &device.id == device_id)
        }) {
            return self.record_failure(
                control_function(kind),
                HuddleAudioTransportError::DeviceUnavailable,
                HuddleControlCommand::SelectDevice { kind, device_id },
                cx,
            );
        }
        self.apply(HuddleControlCommand::SelectDevice { kind, device_id }, cx)
    }

    pub fn replace_devices(
        &mut self,
        devices: Vec<HuddleAudioDevice>,
        cx: &mut Context<Self>,
    ) -> Result<Vec<HuddleControlOutcome>, HuddleControlsError> {
        validate_devices(&devices)?;
        self.devices = devices;
        let microphone_lost = self.selected_microphone.as_ref().is_some_and(|selected| {
            !self.devices.iter().any(|device| {
                device.kind == HuddleAudioDeviceKind::Microphone && &device.id == selected
            })
        });
        let speaker_lost = self.selected_speaker.as_ref().is_some_and(|selected| {
            !self.devices.iter().any(|device| {
                device.kind == HuddleAudioDeviceKind::Speaker && &device.id == selected
            })
        });
        let mut outcomes =
            Vec::with_capacity(usize::from(microphone_lost) + usize::from(speaker_lost));
        if microphone_lost {
            outcomes.push(self.apply(
                HuddleControlCommand::SelectDevice {
                    kind: HuddleAudioDeviceKind::Microphone,
                    device_id: None,
                },
                cx,
            ));
        }
        if speaker_lost {
            outcomes.push(self.apply(
                HuddleControlCommand::SelectDevice {
                    kind: HuddleAudioDeviceKind::Speaker,
                    device_id: None,
                },
                cx,
            ));
        }
        cx.notify();
        Ok(outcomes)
    }

    pub fn retry_failed(&mut self, cx: &mut Context<Self>) -> HuddleControlOutcome {
        let Some(command) = self.failure.as_ref().map(|failure| failure.command.clone()) else {
            return HuddleControlOutcome::Unchanged;
        };
        self.apply(command, cx)
    }

    pub fn use_default_device_fallback(&mut self, cx: &mut Context<Self>) -> HuddleControlOutcome {
        let Some(kind) = self
            .failure
            .as_ref()
            .and_then(|failure| match failure.command {
                HuddleControlCommand::SelectDevice { kind, .. } => Some(kind),
                _ => None,
            })
        else {
            return HuddleControlOutcome::Unchanged;
        };
        self.apply(
            HuddleControlCommand::SelectDevice {
                kind,
                device_id: None,
            },
            cx,
        )
    }

    fn apply(
        &mut self,
        command: HuddleControlCommand,
        cx: &mut Context<Self>,
    ) -> HuddleControlOutcome {
        let result = match &command {
            HuddleControlCommand::SetMuted(muted) => {
                if *muted == self.muted {
                    return HuddleControlOutcome::Unchanged;
                }
                let enabled =
                    microphone_enabled(*muted, self.push_to_talk_enabled, self.push_to_talk_held);
                let current_enabled = self.microphone_enabled();
                if enabled != current_enabled {
                    self.transport.set_microphone_enabled(&self.scope, enabled)
                } else {
                    Ok(())
                }
            }
            HuddleControlCommand::SetSpeakerMuted(muted) => {
                if *muted == self.speaker_muted {
                    return HuddleControlOutcome::Unchanged;
                }
                self.transport.set_playback_enabled(&self.scope, !*muted)
            }
            HuddleControlCommand::SetPushToTalkEnabled(enabled) => {
                if *enabled == self.push_to_talk_enabled {
                    return HuddleControlOutcome::Unchanged;
                }
                let microphone_enabled = microphone_enabled(self.muted, *enabled, false);
                if microphone_enabled != self.microphone_enabled() {
                    self.transport
                        .set_microphone_enabled(&self.scope, microphone_enabled)
                } else {
                    Ok(())
                }
            }
            HuddleControlCommand::SetPushToTalkHeld(held) => {
                if !self.push_to_talk_enabled || *held == self.push_to_talk_held {
                    return HuddleControlOutcome::Unchanged;
                }
                self.transport.set_microphone_enabled(&self.scope, *held)
            }
            HuddleControlCommand::SelectDevice { kind, device_id } => {
                let selected = match kind {
                    HuddleAudioDeviceKind::Microphone => &self.selected_microphone,
                    HuddleAudioDeviceKind::Speaker => &self.selected_speaker,
                };
                if selected == device_id {
                    return HuddleControlOutcome::Unchanged;
                }
                self.transport
                    .select_device(&self.scope, *kind, device_id.as_ref())
            }
        };

        if let Err(reason) = result {
            return self.record_failure(command.function(), reason, command, cx);
        }
        match command {
            HuddleControlCommand::SetMuted(muted) => self.muted = muted,
            HuddleControlCommand::SetSpeakerMuted(muted) => self.speaker_muted = muted,
            HuddleControlCommand::SetPushToTalkEnabled(enabled) => {
                self.push_to_talk_enabled = enabled;
                self.push_to_talk_held = false;
            }
            HuddleControlCommand::SetPushToTalkHeld(held) => self.push_to_talk_held = held,
            HuddleControlCommand::SelectDevice { kind, device_id } => match kind {
                HuddleAudioDeviceKind::Microphone => self.selected_microphone = device_id,
                HuddleAudioDeviceKind::Speaker => self.selected_speaker = device_id,
            },
        }
        self.failure = None;
        cx.notify();
        HuddleControlOutcome::Applied
    }

    fn record_failure(
        &mut self,
        function: HuddleControlFunction,
        reason: HuddleAudioTransportError,
        command: HuddleControlCommand,
        cx: &mut Context<Self>,
    ) -> HuddleControlOutcome {
        self.failure = Some(HuddleControlFailure {
            function,
            reason,
            command,
        });
        cx.notify();
        HuddleControlOutcome::Failed
    }

    fn selected_device_label(&self, kind: HuddleAudioDeviceKind) -> &str {
        let selected = match kind {
            HuddleAudioDeviceKind::Microphone => self.selected_microphone.as_ref(),
            HuddleAudioDeviceKind::Speaker => self.selected_speaker.as_ref(),
        };
        selected
            .and_then(|selected| {
                self.devices
                    .iter()
                    .find(|device| device.kind == kind && &device.id == selected)
            })
            .map_or("System default", HuddleAudioDevice::label)
    }
}

impl HuddleControlCommand {
    const fn function(&self) -> HuddleControlFunction {
        match self {
            Self::SetMuted(_) | Self::SetPushToTalkEnabled(_) | Self::SetPushToTalkHeld(_) => {
                HuddleControlFunction::Microphone
            }
            Self::SetSpeakerMuted(_) => HuddleControlFunction::Speaker,
            Self::SelectDevice { kind, .. } => control_function(*kind),
        }
    }
}

const fn control_function(kind: HuddleAudioDeviceKind) -> HuddleControlFunction {
    match kind {
        HuddleAudioDeviceKind::Microphone => HuddleControlFunction::MicrophoneDevice,
        HuddleAudioDeviceKind::Speaker => HuddleControlFunction::SpeakerDevice,
    }
}

const fn microphone_enabled(muted: bool, push_to_talk_enabled: bool, held: bool) -> bool {
    if push_to_talk_enabled { held } else { !muted }
}

fn validate_devices(devices: &[HuddleAudioDevice]) -> Result<(), HuddleControlsError> {
    let mut identities = BTreeSet::new();
    for device in devices {
        if !identities.insert((device.kind, device.id.clone())) {
            return Err(HuddleControlsError::DuplicateDevice);
        }
    }
    Ok(())
}

impl Render for HuddleControlsView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let microphone_label = if self.muted { "Unmute" } else { "Mute" };
        let speaker_label = if self.speaker_muted {
            "Enable speaker"
        } else {
            "Mute speaker"
        };
        let push_to_talk_label = if self.push_to_talk_enabled {
            "Disable push to talk"
        } else {
            "Enable push to talk"
        };
        v_flex()
            .id("huddle-audio-controls")
            .role(Role::Group)
            .aria_label("Huddle audio controls")
            .gap_2()
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new("huddle-microphone-toggle", microphone_label)
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.toggle_mute(cx);
                            })),
                    )
                    .child(
                        Button::new("huddle-speaker-toggle", speaker_label)
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.toggle_speaker_mute(cx);
                            })),
                    )
                    .child(
                        Button::new("huddle-push-to-talk-toggle", push_to_talk_label)
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.set_push_to_talk_enabled(!this.push_to_talk_enabled, cx);
                            })),
                    ),
            )
            .child(
                div()
                    .id("huddle-audio-devices")
                    .role(Role::Status)
                    .child(format!(
                        "Microphone: {}; Speaker: {}",
                        self.selected_device_label(HuddleAudioDeviceKind::Microphone),
                        self.selected_device_label(HuddleAudioDeviceKind::Speaker),
                    )),
            )
            .when_some(self.failure.clone(), |this, failure| {
                this.child(
                    v_flex()
                        .id("huddle-audio-failure")
                        .role(Role::Alert)
                        .aria_label(failure.function.label())
                        .gap_1()
                        .child(failure.function.label())
                        .child(
                            h_flex()
                                .gap_1()
                                .child(
                                    Button::new("huddle-audio-retry", "Retry")
                                        .style(ButtonStyle::Subtle)
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.retry_failed(cx);
                                        })),
                                )
                                .when(failure.fallback_available(), |this| {
                                    this.child(
                                        Button::new("huddle-audio-fallback", "Use system default")
                                            .style(ButtonStyle::Subtle)
                                            .on_click(cx.listener(|this, _, _window, cx| {
                                                this.use_default_device_fallback(cx);
                                            })),
                                    )
                                }),
                        ),
                )
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleControlsError {
    InvalidDevice,
    DuplicateDevice,
    WrongNativeRoom,
}

impl fmt::Display for HuddleControlsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDevice => "huddle audio device is invalid",
            Self::DuplicateDevice => "huddle audio device is duplicated",
            Self::WrongNativeRoom => "huddle audio controls use the wrong native room",
        })
    }
}

impl Error for HuddleControlsError {}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use audio::NativeHuddleCallbackScope;
    use collaboration_domain::{AggregateId, CommunityId, HuddleGeneration, HuddleIdentity};
    use gpui::TestAppContext;
    use uuid::Uuid;

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum Observation {
        Microphone(bool),
        Playback(bool),
        Device(HuddleAudioDeviceKind, Option<HuddleAudioDeviceId>),
    }

    #[derive(Default)]
    struct TransportState {
        observations: Vec<Observation>,
        fail_next: Option<HuddleAudioTransportError>,
    }

    struct RecordingTransport(Rc<RefCell<TransportState>>);

    impl RecordingTransport {
        fn observe(&self, observation: Observation) -> Result<(), HuddleAudioTransportError> {
            let mut state = self.0.borrow_mut();
            if let Some(error) = state.fail_next.take() {
                return Err(error);
            }
            state.observations.push(observation);
            Ok(())
        }
    }

    impl NativeHuddleAudioControlTransport for RecordingTransport {
        fn select_device(
            &mut self,
            _scope: &NativeHuddleCallbackScope,
            kind: HuddleAudioDeviceKind,
            device_id: Option<&HuddleAudioDeviceId>,
        ) -> Result<(), HuddleAudioTransportError> {
            self.observe(Observation::Device(kind, device_id.cloned()))
        }

        fn set_microphone_enabled(
            &mut self,
            _scope: &NativeHuddleCallbackScope,
            enabled: bool,
        ) -> Result<(), HuddleAudioTransportError> {
            self.observe(Observation::Microphone(enabled))
        }

        fn set_playback_enabled(
            &mut self,
            _scope: &NativeHuddleCallbackScope,
            enabled: bool,
        ) -> Result<(), HuddleAudioTransportError> {
            self.observe(Observation::Playback(enabled))
        }
    }

    fn identity() -> HuddleIdentity {
        HuddleIdentity::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            AggregateId::from_uuid(Uuid::from_u128(2)),
            AggregateId::from_uuid(Uuid::from_u128(3)),
            HuddleGeneration::new(4).expect("generation"),
        )
        .expect("identity")
    }

    fn scope(identity: HuddleIdentity) -> NativeHuddleCallbackScope {
        NativeHuddleCallbackScope::from_livekit(
            NativeHuddleRoomName::for_huddle(identity).as_str(),
            1,
        )
        .expect("callback scope")
    }

    fn device(value: &str, kind: HuddleAudioDeviceKind) -> HuddleAudioDevice {
        HuddleAudioDevice::new(
            HuddleAudioDeviceId::new(value).expect("device id"),
            format!("{value} device"),
            kind,
        )
        .expect("device")
    }

    fn view(
        cx: &mut TestAppContext,
        initially_muted: bool,
    ) -> (
        gpui::Entity<HuddleControlsView>,
        Rc<RefCell<TransportState>>,
    ) {
        let identity = identity();
        let state = Rc::new(RefCell::new(TransportState::default()));
        let transport = RecordingTransport(state.clone());
        let view = cx.new(|_| {
            HuddleControlsView::new(
                identity,
                scope(identity),
                vec![
                    device("built-in-mic", HuddleAudioDeviceKind::Microphone),
                    device("usb-mic", HuddleAudioDeviceKind::Microphone),
                    device("built-in-output", HuddleAudioDeviceKind::Speaker),
                    device("usb-output", HuddleAudioDeviceKind::Speaker),
                ],
                initially_muted,
                false,
                transport,
            )
            .expect("controls")
        });
        (view, state)
    }

    #[gpui::test]
    fn mute_speaker_and_push_to_talk_drive_confirmed_native_state(cx: &mut TestAppContext) {
        let (view, state) = view(cx, false);
        assert_eq!(
            view.update(cx, HuddleControlsView::toggle_mute),
            HuddleControlOutcome::Applied
        );
        assert!(view.read_with(cx, |view, _| view.is_muted()));
        assert_eq!(
            view.update(cx, HuddleControlsView::toggle_speaker_mute),
            HuddleControlOutcome::Applied
        );
        assert!(view.read_with(cx, |view, _| view.is_speaker_muted()));
        assert_eq!(
            view.update(cx, |view, cx| view.set_push_to_talk_enabled(true, cx)),
            HuddleControlOutcome::Applied
        );
        assert_eq!(
            view.update(cx, |view, cx| view.set_push_to_talk_held(true, cx)),
            HuddleControlOutcome::Applied
        );
        assert!(view.read_with(cx, |view, _| view.microphone_enabled()));
        assert_eq!(
            view.update(cx, |view, cx| view.set_push_to_talk_held(false, cx)),
            HuddleControlOutcome::Applied
        );
        assert!(!view.read_with(cx, |view, _| view.microphone_enabled()));
        assert_eq!(
            state.borrow().observations,
            vec![
                Observation::Microphone(false),
                Observation::Playback(false),
                Observation::Microphone(true),
                Observation::Microphone(false),
            ]
        );
    }

    #[gpui::test]
    fn device_switch_failure_retains_selection_and_safe_retry_commits(cx: &mut TestAppContext) {
        let (view, state) = view(cx, true);
        let built_in = HuddleAudioDeviceId::new("built-in-mic").expect("built-in id");
        let usb = HuddleAudioDeviceId::new("usb-mic").expect("usb id");
        assert_eq!(
            view.update(cx, |view, cx| {
                view.select_device(
                    HuddleAudioDeviceKind::Microphone,
                    Some(built_in.clone()),
                    cx,
                )
            }),
            HuddleControlOutcome::Applied
        );
        state.borrow_mut().fail_next = Some(HuddleAudioTransportError::DeviceUnavailable);
        assert_eq!(
            view.update(cx, |view, cx| {
                view.select_device(HuddleAudioDeviceKind::Microphone, Some(usb.clone()), cx)
            }),
            HuddleControlOutcome::Failed
        );
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_microphone().cloned()),
            Some(built_in)
        );
        assert_eq!(
            view.read_with(cx, |view, _| view
                .failure()
                .map(HuddleControlFailure::function)),
            Some(HuddleControlFunction::MicrophoneDevice)
        );
        assert_eq!(
            view.update(cx, HuddleControlsView::retry_failed),
            HuddleControlOutcome::Applied
        );
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_microphone().cloned()),
            Some(usb)
        );
    }

    #[gpui::test]
    fn device_loss_falls_back_without_ending_or_rebinding_the_huddle(cx: &mut TestAppContext) {
        let (view, state) = view(cx, true);
        let identity = view.read_with(cx, |view, _| view.identity());
        let usb = HuddleAudioDeviceId::new("usb-output").expect("usb output");
        assert_eq!(
            view.update(cx, |view, cx| {
                view.select_device(HuddleAudioDeviceKind::Speaker, Some(usb), cx)
            }),
            HuddleControlOutcome::Applied
        );
        let outcomes = view
            .update(cx, |view, cx| {
                view.replace_devices(
                    vec![
                        device("built-in-mic", HuddleAudioDeviceKind::Microphone),
                        device("built-in-output", HuddleAudioDeviceKind::Speaker),
                    ],
                    cx,
                )
            })
            .expect("replace devices");
        assert_eq!(outcomes, vec![HuddleControlOutcome::Applied]);
        assert_eq!(
            view.read_with(cx, |view, _| view.selected_speaker().cloned()),
            None
        );
        assert_eq!(view.read_with(cx, |view, _| view.identity()), identity);
        assert_eq!(
            state.borrow().observations.last(),
            Some(&Observation::Device(HuddleAudioDeviceKind::Speaker, None))
        );
    }

    #[gpui::test]
    fn permission_denial_preserves_mute_and_retry_restores_microphone(cx: &mut TestAppContext) {
        let (view, state) = view(cx, true);
        let identity = view.read_with(cx, |view, _| view.identity());
        state.borrow_mut().fail_next = Some(HuddleAudioTransportError::PermissionDenied);
        assert_eq!(
            view.update(cx, HuddleControlsView::toggle_mute),
            HuddleControlOutcome::Failed
        );
        assert!(view.read_with(cx, |view, _| view.is_muted()));
        assert_eq!(view.read_with(cx, |view, _| view.identity()), identity);
        assert_eq!(
            view.read_with(cx, |view, _| view
                .failure()
                .map(HuddleControlFailure::reason)),
            Some(HuddleAudioTransportError::PermissionDenied)
        );
        assert_eq!(
            view.update(cx, HuddleControlsView::retry_failed),
            HuddleControlOutcome::Applied
        );
        assert!(!view.read_with(cx, |view, _| view.is_muted()));
        assert_eq!(
            state.borrow().observations,
            vec![Observation::Microphone(true)]
        );
    }

    #[gpui::test]
    fn controls_reject_a_scope_for_another_native_room(cx: &mut TestAppContext) {
        let identity = identity();
        let other_identity = HuddleIdentity::new(
            identity.community_id(),
            identity.channel_id(),
            AggregateId::from_uuid(Uuid::from_u128(99)),
            identity.generation(),
        )
        .expect("other identity");
        let state = Rc::new(RefCell::new(TransportState::default()));
        assert_eq!(
            HuddleControlsView::new(
                identity,
                scope(other_identity),
                Vec::new(),
                true,
                false,
                RecordingTransport(state),
            )
            .err(),
            Some(HuddleControlsError::WrongNativeRoom)
        );
        cx.run_until_parked();
    }
}
