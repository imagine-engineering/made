use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use collaboration_domain::{
    AggregateId, Huddle, HuddleIdentity, HuddleLifecycleState, HuddleModerationState,
    HuddleParticipantPresence, HuddleParticipantRole, HuddleTranscriptSegmentId, PrincipalId,
};
use gpui::{App, Context, Entity, IntoElement, Render, Role, Window};
use ui::{Button, ButtonStyle, prelude::*};

use crate::huddle_controls::HuddleControlsView;

const MAX_HUDDLE_DISPLAY_LABEL_BYTES: usize = 128;
const MAX_HUDDLE_TRANSCRIPT_ROWS: usize = 512;
const MAX_HUDDLE_TRANSCRIPT_BYTES: usize = 16 * 1024;
const MAX_HUDDLE_TRANSCRIPT_CHARACTERS: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleParticipantDisplay {
    principal_id: PrincipalId,
    label: String,
}

impl HuddleParticipantDisplay {
    pub fn new(
        principal_id: PrincipalId,
        label: impl Into<String>,
    ) -> Result<Self, HuddleWorkspaceError> {
        let label = label.into();
        let label = label.trim();
        if principal_id.as_uuid().is_nil()
            || label.is_empty()
            || label.len() > MAX_HUDDLE_DISPLAY_LABEL_BYTES
            || label.chars().any(char::is_control)
        {
            return Err(HuddleWorkspaceError::InvalidParticipant);
        }
        Ok(Self {
            principal_id,
            label: label.to_string(),
        })
    }

    pub const fn principal_id(&self) -> PrincipalId {
        self.principal_id
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleParticipantRow {
    display: HuddleParticipantDisplay,
    role: HuddleParticipantRole,
    presence: HuddleParticipantPresence,
    moderation: HuddleModerationState,
}

impl HuddleParticipantRow {
    pub const fn display(&self) -> &HuddleParticipantDisplay {
        &self.display
    }

    pub const fn role(&self) -> HuddleParticipantRole {
        self.role
    }

    pub const fn presence(&self) -> HuddleParticipantPresence {
        self.presence
    }

    pub const fn moderation(&self) -> HuddleModerationState {
        self.moderation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleTranscriptDisplayState {
    Partial,
    Final,
    Redacted,
    Expired,
}

#[derive(Clone, Eq, PartialEq)]
pub struct HuddleTranscriptDisplay {
    segment_id: HuddleTranscriptSegmentId,
    message_id: AggregateId,
    participant_principal_id: PrincipalId,
    state: HuddleTranscriptDisplayState,
    text: Option<String>,
    started_at_millis: u64,
    ended_at_millis: u64,
}

impl HuddleTranscriptDisplay {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        segment_id: HuddleTranscriptSegmentId,
        message_id: AggregateId,
        participant_principal_id: PrincipalId,
        state: HuddleTranscriptDisplayState,
        text: Option<String>,
        started_at_millis: u64,
        ended_at_millis: u64,
    ) -> Result<Self, HuddleWorkspaceError> {
        if message_id.as_uuid().is_nil()
            || participant_principal_id.as_uuid().is_nil()
            || started_at_millis == 0
            || ended_at_millis <= started_at_millis
            || text.as_ref().is_some_and(|text| {
                text.trim().is_empty()
                    || text.len() > MAX_HUDDLE_TRANSCRIPT_BYTES
                    || text.chars().count() > MAX_HUDDLE_TRANSCRIPT_CHARACTERS
                    || text.chars().any(|character| character == '\0')
            })
            || matches!(
                state,
                HuddleTranscriptDisplayState::Partial | HuddleTranscriptDisplayState::Final
            ) != text.is_some()
        {
            return Err(HuddleWorkspaceError::InvalidTranscript);
        }
        Ok(Self {
            segment_id,
            message_id,
            participant_principal_id,
            state,
            text,
            started_at_millis,
            ended_at_millis,
        })
    }

    pub const fn segment_id(&self) -> HuddleTranscriptSegmentId {
        self.segment_id
    }

    pub const fn message_id(&self) -> AggregateId {
        self.message_id
    }

    pub const fn participant_principal_id(&self) -> PrincipalId {
        self.participant_principal_id
    }

    pub const fn state(&self) -> HuddleTranscriptDisplayState {
        self.state
    }

    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    pub const fn started_at_millis(&self) -> u64 {
        self.started_at_millis
    }

    pub const fn ended_at_millis(&self) -> u64 {
        self.ended_at_millis
    }
}

impl fmt::Debug for HuddleTranscriptDisplay {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HuddleTranscriptDisplay")
            .field("segment_id", &self.segment_id)
            .field("message_id", &self.message_id)
            .field("participant_principal_id", &self.participant_principal_id)
            .field("state", &self.state)
            .field("text_bytes", &self.text.as_ref().map(String::len))
            .field("started_at_millis", &self.started_at_millis)
            .field("ended_at_millis", &self.ended_at_millis)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleWorkspaceFailureReason {
    PermissionDenied,
    DeviceUnavailable,
    NetworkUnavailable,
    VoiceModelUnavailable,
    TranscriptUnavailable,
    StaleGeneration,
    ServiceUnavailable,
}

impl HuddleWorkspaceFailureReason {
    const fn label(self) -> &'static str {
        match self {
            Self::PermissionDenied => "Permission denied",
            Self::DeviceUnavailable => "Audio device unavailable",
            Self::NetworkUnavailable => "Huddle network unavailable",
            Self::VoiceModelUnavailable => "Voice model unavailable",
            Self::TranscriptUnavailable => "Transcript unavailable",
            Self::StaleGeneration => "Huddle state changed",
            Self::ServiceUnavailable => "Huddle service unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleWorkspaceAvailability {
    Ready,
    Recovering,
    Failed(HuddleWorkspaceFailureReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleWorkspaceSnapshot {
    huddle: Huddle,
    participants: Vec<HuddleParticipantRow>,
    transcripts: Vec<HuddleTranscriptDisplay>,
    network: HuddleWorkspaceAvailability,
    voice_model: HuddleWorkspaceAvailability,
    transcript: HuddleWorkspaceAvailability,
}

impl HuddleWorkspaceSnapshot {
    pub fn new(
        huddle: Huddle,
        participant_displays: Vec<HuddleParticipantDisplay>,
        mut transcripts: Vec<HuddleTranscriptDisplay>,
        network: HuddleWorkspaceAvailability,
        voice_model: HuddleWorkspaceAvailability,
        transcript: HuddleWorkspaceAvailability,
    ) -> Result<Self, HuddleWorkspaceError> {
        if transcripts.len() > MAX_HUDDLE_TRANSCRIPT_ROWS {
            return Err(HuddleWorkspaceError::InvalidTranscript);
        }
        let mut displays = BTreeMap::new();
        for display in participant_displays {
            if displays.insert(display.principal_id, display).is_some() {
                return Err(HuddleWorkspaceError::InvalidParticipant);
            }
        }
        let mut participants = Vec::new();
        for participant in huddle.participants() {
            let Some(display) = displays.remove(&participant.principal_id()) else {
                return Err(HuddleWorkspaceError::InvalidParticipant);
            };
            participants.push(HuddleParticipantRow {
                display,
                role: participant.role(),
                presence: participant.presence(),
                moderation: participant.moderation(),
            });
        }
        if !displays.is_empty() {
            return Err(HuddleWorkspaceError::InvalidParticipant);
        }
        participants.sort_by_key(|row| row.display.principal_id);

        transcripts.sort_by_key(|row| (row.started_at_millis, row.segment_id));
        let mut segment_ids = BTreeSet::new();
        for row in &transcripts {
            if !segment_ids.insert(row.segment_id)
                || huddle.participant(row.participant_principal_id).is_none()
                || (row.state != HuddleTranscriptDisplayState::Partial
                    && !huddle.transcript_references().any(|reference| {
                        reference.segment_id() == row.segment_id
                            && reference.message_id() == row.message_id
                            && reference.participant_principal_id() == row.participant_principal_id
                            && reference.started_at_millis() == row.started_at_millis
                            && reference.ended_at_millis() == row.ended_at_millis
                    }))
            {
                return Err(HuddleWorkspaceError::InvalidTranscript);
            }
        }
        Ok(Self {
            huddle,
            participants,
            transcripts,
            network,
            voice_model,
            transcript,
        })
    }

    pub const fn huddle(&self) -> &Huddle {
        &self.huddle
    }

    pub fn participants(&self) -> &[HuddleParticipantRow] {
        &self.participants
    }

    pub fn transcripts(&self) -> &[HuddleTranscriptDisplay] {
        &self.transcripts
    }

    pub const fn network(&self) -> HuddleWorkspaceAvailability {
        self.network
    }

    pub const fn voice_model(&self) -> HuddleWorkspaceAvailability {
        self.voice_model
    }

    pub const fn transcript(&self) -> HuddleWorkspaceAvailability {
        self.transcript
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleWorkspaceAction {
    Start,
    Join,
    Leave,
    End,
    RetryNetwork,
    RetryVoiceModel,
    RetryTranscript,
}

impl HuddleWorkspaceAction {
    const fn label(self) -> &'static str {
        match self {
            Self::Start => "Start huddle",
            Self::Join => "Join huddle",
            Self::Leave => "Leave huddle",
            Self::End => "End huddle",
            Self::RetryNetwork => "Retry network",
            Self::RetryVoiceModel => "Retry voice model",
            Self::RetryTranscript => "Retry transcript",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HuddleWorkspaceRequest {
    identity: HuddleIdentity,
    viewer_principal_id: PrincipalId,
    action: HuddleWorkspaceAction,
}

impl HuddleWorkspaceRequest {
    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn viewer_principal_id(&self) -> PrincipalId {
        self.viewer_principal_id
    }

    pub const fn action(&self) -> HuddleWorkspaceAction {
        self.action
    }
}

pub trait NativeHuddleWorkspaceService: 'static {
    fn perform(
        &mut self,
        request: HuddleWorkspaceRequest,
    ) -> Result<HuddleWorkspaceSnapshot, HuddleWorkspaceFailureReason>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HuddleWorkspaceFailure {
    action: HuddleWorkspaceAction,
    reason: HuddleWorkspaceFailureReason,
}

impl HuddleWorkspaceFailure {
    pub const fn action(self) -> HuddleWorkspaceAction {
        self.action
    }

    pub const fn reason(self) -> HuddleWorkspaceFailureReason {
        self.reason
    }

    pub const fn retryable(self) -> bool {
        !matches!(self.reason, HuddleWorkspaceFailureReason::PermissionDenied)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleWorkspaceOutcome {
    Applied,
    Unchanged,
    Failed,
}

pub struct HuddleWorkspaceView {
    identity: HuddleIdentity,
    viewer_principal_id: PrincipalId,
    snapshot: Option<HuddleWorkspaceSnapshot>,
    controls: Option<Entity<HuddleControlsView>>,
    service: Box<dyn NativeHuddleWorkspaceService>,
    failure: Option<HuddleWorkspaceFailure>,
}

impl HuddleWorkspaceView {
    pub fn new(
        identity: HuddleIdentity,
        viewer_principal_id: PrincipalId,
        snapshot: Option<HuddleWorkspaceSnapshot>,
        controls: Option<Entity<HuddleControlsView>>,
        service: impl NativeHuddleWorkspaceService,
        cx: &App,
    ) -> Result<Self, HuddleWorkspaceError> {
        if viewer_principal_id.as_uuid().is_nil()
            || snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.huddle.identity() != identity)
            || controls
                .as_ref()
                .is_some_and(|controls| controls.read(cx).identity() != identity)
        {
            return Err(HuddleWorkspaceError::WrongHuddle);
        }
        Ok(Self {
            identity,
            viewer_principal_id,
            snapshot,
            controls,
            service: Box::new(service),
            failure: None,
        })
    }

    pub const fn identity(&self) -> HuddleIdentity {
        self.identity
    }

    pub const fn viewer_principal_id(&self) -> PrincipalId {
        self.viewer_principal_id
    }

    pub const fn snapshot(&self) -> Option<&HuddleWorkspaceSnapshot> {
        self.snapshot.as_ref()
    }

    pub const fn controls(&self) -> Option<&Entity<HuddleControlsView>> {
        self.controls.as_ref()
    }

    pub const fn failure(&self) -> Option<HuddleWorkspaceFailure> {
        self.failure
    }

    pub fn start(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::Start, cx)
    }

    pub fn join(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::Join, cx)
    }

    pub fn leave(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::Leave, cx)
    }

    pub fn end(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::End, cx)
    }

    pub fn retry_network(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::RetryNetwork, cx)
    }

    pub fn retry_voice_model(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::RetryVoiceModel, cx)
    }

    pub fn retry_transcript(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        self.apply(HuddleWorkspaceAction::RetryTranscript, cx)
    }

    pub fn retry_failed(&mut self, cx: &mut Context<Self>) -> HuddleWorkspaceOutcome {
        let Some(failure) = self.failure else {
            return HuddleWorkspaceOutcome::Unchanged;
        };
        if !failure.retryable() {
            return HuddleWorkspaceOutcome::Unchanged;
        }
        self.apply(failure.action, cx)
    }

    fn apply(
        &mut self,
        action: HuddleWorkspaceAction,
        cx: &mut Context<Self>,
    ) -> HuddleWorkspaceOutcome {
        if !self.action_available(action) {
            return self.record_failure(action, HuddleWorkspaceFailureReason::StaleGeneration, cx);
        }
        let result = self.service.perform(HuddleWorkspaceRequest {
            identity: self.identity,
            viewer_principal_id: self.viewer_principal_id,
            action,
        });
        let snapshot = match result {
            Ok(snapshot) => snapshot,
            Err(reason) => return self.record_failure(action, reason, cx),
        };
        if !snapshot_confirms_action(&snapshot, self.identity, self.viewer_principal_id, action) {
            return self.record_failure(action, HuddleWorkspaceFailureReason::StaleGeneration, cx);
        }
        self.snapshot = Some(snapshot);
        self.failure = None;
        cx.notify();
        HuddleWorkspaceOutcome::Applied
    }

    fn action_available(&self, action: HuddleWorkspaceAction) -> bool {
        let participant = self
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.huddle.participant(self.viewer_principal_id));
        let active = self.snapshot.as_ref().is_some_and(|snapshot| {
            matches!(snapshot.huddle.lifecycle(), HuddleLifecycleState::Active)
        });
        match action {
            HuddleWorkspaceAction::Start => self.snapshot.is_none(),
            HuddleWorkspaceAction::Join => {
                active
                    && participant.is_none_or(|participant| {
                        participant.presence() == HuddleParticipantPresence::Left
                    })
            }
            HuddleWorkspaceAction::Leave => {
                active
                    && participant.is_some_and(|participant| {
                        participant.presence() == HuddleParticipantPresence::Present
                    })
            }
            HuddleWorkspaceAction::End => {
                active
                    && participant.is_some_and(|participant| {
                        participant.presence() == HuddleParticipantPresence::Present
                            && matches!(
                                participant.role(),
                                HuddleParticipantRole::Owner | HuddleParticipantRole::Moderator
                            )
                    })
            }
            HuddleWorkspaceAction::RetryNetwork => self.snapshot.as_ref().is_some_and(|snapshot| {
                matches!(snapshot.network, HuddleWorkspaceAvailability::Failed(_))
            }),
            HuddleWorkspaceAction::RetryVoiceModel => {
                self.snapshot.as_ref().is_some_and(|snapshot| {
                    matches!(snapshot.voice_model, HuddleWorkspaceAvailability::Failed(_))
                })
            }
            HuddleWorkspaceAction::RetryTranscript => {
                self.snapshot.as_ref().is_some_and(|snapshot| {
                    matches!(snapshot.transcript, HuddleWorkspaceAvailability::Failed(_))
                })
            }
        }
    }

    fn record_failure(
        &mut self,
        action: HuddleWorkspaceAction,
        reason: HuddleWorkspaceFailureReason,
        cx: &mut Context<Self>,
    ) -> HuddleWorkspaceOutcome {
        self.failure = Some(HuddleWorkspaceFailure { action, reason });
        cx.notify();
        HuddleWorkspaceOutcome::Failed
    }
}

impl Render for HuddleWorkspaceView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let snapshot = self.snapshot.clone();
        let participant = snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.huddle.participant(self.viewer_principal_id));
        let active = snapshot.as_ref().is_some_and(|snapshot| {
            matches!(snapshot.huddle.lifecycle(), HuddleLifecycleState::Active)
        });
        let viewer_present = participant.is_some_and(|participant| {
            participant.presence() == HuddleParticipantPresence::Present
        });
        let can_end = participant.is_some_and(|participant| {
            viewer_present
                && matches!(
                    participant.role(),
                    HuddleParticipantRole::Owner | HuddleParticipantRole::Moderator
                )
        });
        v_flex()
            .id("native-huddle-workspace")
            .role(Role::Region)
            .aria_label("Huddle workspace")
            .size_full()
            .gap_2()
            .p_3()
            .child(
                h_flex()
                    .id("native-huddle-header")
                    .justify_between()
                    .child(huddle_state_label(snapshot.as_ref()))
                    .child(
                        h_flex()
                            .gap_1()
                            .when(snapshot.is_none(), |this| {
                                this.child(
                                    Button::new("native-huddle-start", "Start")
                                        .style(ButtonStyle::Subtle)
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.start(cx);
                                        })),
                                )
                            })
                            .when(active && !viewer_present, |this| {
                                this.child(
                                    Button::new("native-huddle-join", "Join")
                                        .style(ButtonStyle::Subtle)
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.join(cx);
                                        })),
                                )
                            })
                            .when(active && viewer_present, |this| {
                                this.child(
                                    Button::new("native-huddle-leave", "Leave")
                                        .style(ButtonStyle::Subtle)
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.leave(cx);
                                        })),
                                )
                            })
                            .when(active && can_end, |this| {
                                this.child(
                                    Button::new("native-huddle-end", "End")
                                        .style(ButtonStyle::Subtle)
                                        .on_click(cx.listener(|this, _, _window, cx| {
                                            this.end(cx);
                                        })),
                                )
                            }),
                    ),
            )
            .when_some(snapshot, |this, snapshot| {
                this.child(render_participants(&snapshot))
                    .child(render_reactions(&snapshot))
                    .children(render_availability(&snapshot, cx))
                    .child(render_transcript(&snapshot))
            })
            .when_some(self.controls.clone(), |this, controls| this.child(controls))
            .when_some(self.failure, |this, failure| {
                this.child(
                    v_flex()
                        .id("native-huddle-failure")
                        .role(Role::Alert)
                        .aria_label(failure.reason.label())
                        .gap_1()
                        .child(failure.reason.label())
                        .when(failure.retryable(), |this| {
                            this.child(
                                Button::new("native-huddle-retry", "Retry")
                                    .style(ButtonStyle::Subtle)
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        this.retry_failed(cx);
                                    })),
                            )
                        }),
                )
            })
    }
}

fn snapshot_confirms_action(
    snapshot: &HuddleWorkspaceSnapshot,
    identity: HuddleIdentity,
    viewer_principal_id: PrincipalId,
    action: HuddleWorkspaceAction,
) -> bool {
    if snapshot.huddle.identity() != identity {
        return false;
    }
    let participant = snapshot.huddle.participant(viewer_principal_id);
    match action {
        HuddleWorkspaceAction::Start | HuddleWorkspaceAction::Join => {
            matches!(snapshot.huddle.lifecycle(), HuddleLifecycleState::Active)
                && participant.is_some_and(|participant| {
                    participant.presence() == HuddleParticipantPresence::Present
                })
        }
        HuddleWorkspaceAction::Leave => {
            !matches!(snapshot.huddle.lifecycle(), HuddleLifecycleState::Active)
                || participant.is_some_and(|participant| {
                    participant.presence() == HuddleParticipantPresence::Left
                })
        }
        HuddleWorkspaceAction::End => {
            matches!(
                snapshot.huddle.lifecycle(),
                HuddleLifecycleState::Ended { .. }
            )
        }
        HuddleWorkspaceAction::RetryNetwork => {
            snapshot.network == HuddleWorkspaceAvailability::Ready
        }
        HuddleWorkspaceAction::RetryVoiceModel => {
            snapshot.voice_model == HuddleWorkspaceAvailability::Ready
        }
        HuddleWorkspaceAction::RetryTranscript => {
            snapshot.transcript == HuddleWorkspaceAvailability::Ready
        }
    }
}

fn huddle_state_label(snapshot: Option<&HuddleWorkspaceSnapshot>) -> &'static str {
    match snapshot.map(|snapshot| snapshot.huddle.lifecycle()) {
        None => "Huddle ready",
        Some(HuddleLifecycleState::Active) => "Huddle active",
        Some(HuddleLifecycleState::Ended { .. }) => "Huddle ended",
    }
}

fn render_participants(snapshot: &HuddleWorkspaceSnapshot) -> impl IntoElement {
    v_flex()
        .id("native-huddle-participants")
        .role(Role::List)
        .aria_label("Huddle participants")
        .children(
            snapshot
                .participants
                .iter()
                .enumerate()
                .map(|(index, participant)| {
                    div()
                        .id(("native-huddle-participant-row", index))
                        .role(Role::ListItem)
                        .child(format!(
                            "{} · {} · {}",
                            participant.display.label,
                            role_label(participant.role),
                            presence_label(participant.presence),
                        ))
                }),
        )
}

fn render_reactions(snapshot: &HuddleWorkspaceSnapshot) -> impl IntoElement {
    let labels: BTreeMap<_, _> = snapshot
        .participants
        .iter()
        .map(|participant| {
            (
                participant.display.principal_id,
                participant.display.label.as_str(),
            )
        })
        .collect();
    v_flex()
        .id("native-huddle-reactions")
        .role(Role::List)
        .aria_label("Huddle reactions")
        .children(
            snapshot
                .huddle
                .reactions()
                .iter()
                .enumerate()
                .map(|(index, reaction)| {
                    div()
                        .id(("native-huddle-reaction-row", index))
                        .role(Role::ListItem)
                        .child(format!(
                            "{} reacted {}",
                            labels
                                .get(&reaction.participant_principal_id)
                                .copied()
                                .unwrap_or("Participant"),
                            reaction.value.as_str(),
                        ))
                }),
        )
}

fn render_availability(
    snapshot: &HuddleWorkspaceSnapshot,
    cx: &mut Context<HuddleWorkspaceView>,
) -> Vec<impl IntoElement> {
    [
        (
            "network",
            snapshot.network,
            HuddleWorkspaceAction::RetryNetwork,
        ),
        (
            "voice-model",
            snapshot.voice_model,
            HuddleWorkspaceAction::RetryVoiceModel,
        ),
        (
            "transcript",
            snapshot.transcript,
            HuddleWorkspaceAction::RetryTranscript,
        ),
    ]
    .into_iter()
    .filter_map(|(identifier, availability, action)| {
        let HuddleWorkspaceAvailability::Failed(reason) = availability else {
            return None;
        };
        Some(
            h_flex()
                .id(format!("native-huddle-{identifier}-failure"))
                .role(Role::Alert)
                .gap_1()
                .child(reason.label())
                .child(
                    Button::new(format!("native-huddle-{identifier}-retry"), action.label())
                        .style(ButtonStyle::Subtle)
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.apply(action, cx);
                        })),
                ),
        )
    })
    .collect()
}

fn render_transcript(snapshot: &HuddleWorkspaceSnapshot) -> impl IntoElement {
    let labels: BTreeMap<_, _> = snapshot
        .participants
        .iter()
        .map(|participant| {
            (
                participant.display.principal_id,
                participant.display.label.as_str(),
            )
        })
        .collect();
    v_flex()
        .id("native-huddle-transcript")
        .role(Role::Log)
        .aria_label("Huddle transcript")
        .children(snapshot.transcripts.iter().map(|row| {
            let text = match row.state {
                HuddleTranscriptDisplayState::Partial | HuddleTranscriptDisplayState::Final => {
                    row.text.as_deref().unwrap_or("Transcript unavailable")
                }
                HuddleTranscriptDisplayState::Redacted => "Transcript redacted",
                HuddleTranscriptDisplayState::Expired => "Transcript expired",
            };
            div().child(format!(
                "{}: {text}",
                labels
                    .get(&row.participant_principal_id)
                    .copied()
                    .unwrap_or("Participant"),
            ))
        }))
}

const fn role_label(role: HuddleParticipantRole) -> &'static str {
    match role {
        HuddleParticipantRole::Owner => "Owner",
        HuddleParticipantRole::Moderator => "Moderator",
        HuddleParticipantRole::Speaker => "Speaker",
        HuddleParticipantRole::Listener => "Listener",
    }
}

const fn presence_label(presence: HuddleParticipantPresence) -> &'static str {
    match presence {
        HuddleParticipantPresence::Present => "Present",
        HuddleParticipantPresence::Left => "Left",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HuddleWorkspaceError {
    WrongHuddle,
    InvalidParticipant,
    InvalidTranscript,
}

impl fmt::Display for HuddleWorkspaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::WrongHuddle => "huddle workspace scope does not match",
            Self::InvalidParticipant => "huddle workspace participant is invalid",
            Self::InvalidTranscript => "huddle workspace transcript is invalid",
        })
    }
}

impl Error for HuddleWorkspaceError {}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::VecDeque, rc::Rc};

    use audio::NativeHuddleCallbackScope;
    use collaboration_domain::{
        CommunityId, HuddleCommandOutcome, HuddleGeneration, HuddleTranscriptReference, OperationId,
    };
    use gpui::TestAppContext;

    use crate::huddle_controls::{
        HuddleAudioDeviceKind, HuddleAudioTransportError, HuddleControlOutcome,
        NativeHuddleAudioControlTransport,
    };

    use super::*;

    struct QueueService(
        Rc<RefCell<VecDeque<Result<HuddleWorkspaceSnapshot, HuddleWorkspaceFailureReason>>>>,
    );

    impl NativeHuddleWorkspaceService for QueueService {
        fn perform(
            &mut self,
            _request: HuddleWorkspaceRequest,
        ) -> Result<HuddleWorkspaceSnapshot, HuddleWorkspaceFailureReason> {
            self.0
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(HuddleWorkspaceFailureReason::ServiceUnavailable))
        }
    }

    struct AudioTransport {
        microphone_failure: Rc<RefCell<Option<HuddleAudioTransportError>>>,
    }

    impl NativeHuddleAudioControlTransport for AudioTransport {
        fn select_device(
            &mut self,
            _scope: &NativeHuddleCallbackScope,
            _kind: HuddleAudioDeviceKind,
            _device_id: Option<&crate::huddle_controls::HuddleAudioDeviceId>,
        ) -> Result<(), HuddleAudioTransportError> {
            Ok(())
        }

        fn set_microphone_enabled(
            &mut self,
            _scope: &NativeHuddleCallbackScope,
            _enabled: bool,
        ) -> Result<(), HuddleAudioTransportError> {
            self.microphone_failure
                .borrow_mut()
                .take()
                .map_or(Ok(()), Err)
        }

        fn set_playback_enabled(
            &mut self,
            _scope: &NativeHuddleCallbackScope,
            _enabled: bool,
        ) -> Result<(), HuddleAudioTransportError> {
            Ok(())
        }
    }

    struct Fixture {
        identity: HuddleIdentity,
        owner: PrincipalId,
        participant: PrincipalId,
        huddle: Huddle,
    }

    fn fixture() -> Fixture {
        let identity = HuddleIdentity::new(
            CommunityId::new(),
            AggregateId::new(),
            AggregateId::new(),
            HuddleGeneration::new(5).expect("generation"),
        )
        .expect("identity");
        let owner = PrincipalId::new();
        let participant = PrincipalId::new();
        let mut huddle =
            Huddle::start(identity, owner, OperationId::new(), 100).expect("start huddle");
        huddle
            .join(
                participant,
                HuddleParticipantRole::Speaker,
                OperationId::new(),
                110,
            )
            .expect("join participant");
        Fixture {
            identity,
            owner,
            participant,
            huddle,
        }
    }

    fn displays(fixture: &Fixture) -> Vec<HuddleParticipantDisplay> {
        vec![
            HuddleParticipantDisplay::new(fixture.owner, "Owner").expect("owner display"),
            HuddleParticipantDisplay::new(fixture.participant, "Speaker")
                .expect("participant display"),
        ]
    }

    fn snapshot(
        fixture: &Fixture,
        huddle: Huddle,
        network: HuddleWorkspaceAvailability,
        voice_model: HuddleWorkspaceAvailability,
        transcript: HuddleWorkspaceAvailability,
        transcripts: Vec<HuddleTranscriptDisplay>,
    ) -> HuddleWorkspaceSnapshot {
        HuddleWorkspaceSnapshot::new(
            huddle,
            displays(fixture),
            transcripts,
            network,
            voice_model,
            transcript,
        )
        .expect("snapshot")
    }

    fn healthy_snapshot(fixture: &Fixture, huddle: Huddle) -> HuddleWorkspaceSnapshot {
        snapshot(
            fixture,
            huddle,
            HuddleWorkspaceAvailability::Ready,
            HuddleWorkspaceAvailability::Ready,
            HuddleWorkspaceAvailability::Ready,
            Vec::new(),
        )
    }

    fn service(
        responses: Vec<Result<HuddleWorkspaceSnapshot, HuddleWorkspaceFailureReason>>,
    ) -> QueueService {
        QueueService(Rc::new(RefCell::new(responses.into())))
    }

    #[gpui::test]
    fn start_join_leave_and_end_commit_only_confirmed_canonical_snapshots(cx: &mut TestAppContext) {
        let fixture = fixture();
        let start_view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                None,
                None,
                service(vec![Ok(healthy_snapshot(&fixture, fixture.huddle.clone()))]),
                cx,
            )
            .expect("start view")
        });
        assert_eq!(
            start_view.update(cx, HuddleWorkspaceView::start),
            HuddleWorkspaceOutcome::Applied
        );

        let mut left_huddle = fixture.huddle.clone();
        left_huddle
            .leave(fixture.participant, OperationId::new(), 120)
            .expect("leave participant");
        let join_view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.participant,
                Some(healthy_snapshot(&fixture, left_huddle)),
                None,
                service(vec![Ok(healthy_snapshot(&fixture, fixture.huddle.clone()))]),
                cx,
            )
            .expect("join view")
        });
        assert_eq!(
            join_view.update(cx, HuddleWorkspaceView::join),
            HuddleWorkspaceOutcome::Applied
        );

        let mut owner_left = fixture.huddle.clone();
        owner_left
            .leave(fixture.owner, OperationId::new(), 130)
            .expect("owner leaves");
        let leave_view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                Some(healthy_snapshot(&fixture, fixture.huddle.clone())),
                None,
                service(vec![Ok(healthy_snapshot(&fixture, owner_left))]),
                cx,
            )
            .expect("leave view")
        });
        assert_eq!(
            leave_view.update(cx, HuddleWorkspaceView::leave),
            HuddleWorkspaceOutcome::Applied
        );

        let mut ended = fixture.huddle.clone();
        assert_eq!(
            ended.end(fixture.owner, OperationId::new(), 140),
            Ok(HuddleCommandOutcome::Applied)
        );
        let end_view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                Some(healthy_snapshot(&fixture, fixture.huddle.clone())),
                None,
                service(vec![Ok(healthy_snapshot(&fixture, ended))]),
                cx,
            )
            .expect("end view")
        });
        assert_eq!(
            end_view.update(cx, HuddleWorkspaceView::end),
            HuddleWorkspaceOutcome::Applied
        );
    }

    #[gpui::test]
    fn device_failure_stays_scoped_to_controls_and_retains_active_huddle(cx: &mut TestAppContext) {
        let fixture = fixture();
        let microphone_failure = Rc::new(RefCell::new(Some(
            HuddleAudioTransportError::PermissionDenied,
        )));
        let controls = cx.new(|_| {
            HuddleControlsView::new(
                fixture.identity,
                NativeHuddleCallbackScope::from_livekit(
                    audio::NativeHuddleRoomName::for_huddle(fixture.identity).as_str(),
                    1,
                )
                .expect("scope"),
                Vec::new(),
                false,
                false,
                AudioTransport { microphone_failure },
            )
            .expect("controls")
        });
        let view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                Some(healthy_snapshot(&fixture, fixture.huddle.clone())),
                Some(controls.clone()),
                service(Vec::new()),
                cx,
            )
            .expect("workspace")
        });

        assert_eq!(
            controls.update(cx, HuddleControlsView::toggle_mute),
            HuddleControlOutcome::Failed
        );
        assert!(controls.read_with(cx, |controls, _| controls.failure().is_some()));
        assert!(view.read_with(cx, |view, _| {
            matches!(
                view.snapshot()
                    .map(|snapshot| snapshot.huddle().lifecycle()),
                Some(HuddleLifecycleState::Active)
            )
        }));
    }

    #[gpui::test]
    fn voice_model_failure_is_visible_and_retry_recovers_without_rejoining(
        cx: &mut TestAppContext,
    ) {
        let fixture = fixture();
        let failed = snapshot(
            &fixture,
            fixture.huddle.clone(),
            HuddleWorkspaceAvailability::Ready,
            HuddleWorkspaceAvailability::Failed(
                HuddleWorkspaceFailureReason::VoiceModelUnavailable,
            ),
            HuddleWorkspaceAvailability::Ready,
            Vec::new(),
        );
        let recovered = healthy_snapshot(&fixture, fixture.huddle.clone());
        let view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                Some(failed),
                None,
                service(vec![Ok(recovered)]),
                cx,
            )
            .expect("workspace")
        });

        assert_eq!(
            view.update(cx, HuddleWorkspaceView::retry_voice_model),
            HuddleWorkspaceOutcome::Applied
        );
        assert_eq!(
            view.read_with(cx, |view, _| view
                .snapshot()
                .map(HuddleWorkspaceSnapshot::voice_model)),
            Some(HuddleWorkspaceAvailability::Ready)
        );
    }

    #[gpui::test]
    fn network_failure_retains_snapshot_and_exact_retry_can_confirm_leave(cx: &mut TestAppContext) {
        let fixture = fixture();
        let mut left = fixture.huddle.clone();
        left.leave(fixture.owner, OperationId::new(), 130)
            .expect("leave");
        let view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                Some(healthy_snapshot(&fixture, fixture.huddle.clone())),
                None,
                service(vec![
                    Err(HuddleWorkspaceFailureReason::NetworkUnavailable),
                    Ok(healthy_snapshot(&fixture, left)),
                ]),
                cx,
            )
            .expect("workspace")
        });

        assert_eq!(
            view.update(cx, HuddleWorkspaceView::leave),
            HuddleWorkspaceOutcome::Failed
        );
        assert!(view.read_with(cx, |view, _| {
            matches!(
                view.snapshot()
                    .map(|snapshot| snapshot.huddle().lifecycle()),
                Some(HuddleLifecycleState::Active)
            )
        }));
        assert_eq!(
            view.update(cx, HuddleWorkspaceView::retry_failed),
            HuddleWorkspaceOutcome::Applied
        );
    }

    #[gpui::test]
    fn transcript_display_preserves_partial_final_and_redacted_rows_without_debug_leak(
        cx: &mut TestAppContext,
    ) {
        let mut fixture = fixture();
        let partial_segment =
            HuddleTranscriptSegmentId::new(AggregateId::new()).expect("partial segment");
        let final_segment =
            HuddleTranscriptSegmentId::new(AggregateId::new()).expect("final segment");
        let redacted_segment =
            HuddleTranscriptSegmentId::new(AggregateId::new()).expect("redacted segment");
        let final_message = AggregateId::new();
        let redacted_message = AggregateId::new();
        for (segment, message, started, ended) in [
            (final_segment, final_message, 300, 400),
            (redacted_segment, redacted_message, 500, 600),
        ] {
            let reference = HuddleTranscriptReference::new(
                fixture.identity,
                segment,
                message,
                fixture.participant,
                started,
                ended,
            )
            .expect("reference");
            fixture
                .huddle
                .link_transcript(reference, OperationId::new(), ended + 10)
                .expect("link transcript");
        }
        let rows = vec![
            HuddleTranscriptDisplay::new(
                partial_segment,
                AggregateId::new(),
                fixture.participant,
                HuddleTranscriptDisplayState::Partial,
                Some("private partial".to_string()),
                150,
                200,
            )
            .expect("partial"),
            HuddleTranscriptDisplay::new(
                final_segment,
                final_message,
                fixture.participant,
                HuddleTranscriptDisplayState::Final,
                Some("private final".to_string()),
                300,
                400,
            )
            .expect("final"),
            HuddleTranscriptDisplay::new(
                redacted_segment,
                redacted_message,
                fixture.participant,
                HuddleTranscriptDisplayState::Redacted,
                None,
                500,
                600,
            )
            .expect("redacted"),
        ];
        let snapshot = snapshot(
            &fixture,
            fixture.huddle.clone(),
            HuddleWorkspaceAvailability::Ready,
            HuddleWorkspaceAvailability::Ready,
            HuddleWorkspaceAvailability::Ready,
            rows,
        );
        let view = cx.new(|cx| {
            HuddleWorkspaceView::new(
                fixture.identity,
                fixture.owner,
                Some(snapshot),
                None,
                service(Vec::new()),
                cx,
            )
            .expect("workspace")
        });

        view.read_with(cx, |view, _| {
            let transcripts = view.snapshot().expect("snapshot").transcripts();
            assert_eq!(transcripts.len(), 3);
            assert_eq!(
                transcripts[0].state(),
                HuddleTranscriptDisplayState::Partial
            );
            assert_eq!(transcripts[1].state(), HuddleTranscriptDisplayState::Final);
            assert_eq!(
                transcripts[2].state(),
                HuddleTranscriptDisplayState::Redacted
            );
            assert!(!format!("{transcripts:?}").contains("private final"));
        });
    }
}
