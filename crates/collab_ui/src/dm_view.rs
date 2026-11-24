use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use collaboration_domain::{
    AggregateId, CommunityId, DirectMessage, DmParticipantState, NostrEventId, PrincipalId,
};
use gpui::{AppContext as _, Context, Entity, IntoElement, Render, Role, SharedString, Window};
use ui::prelude::*;

use crate::message_timeline::{MessageTimeline, MessageTimelineError, MessageTimelinePage};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmParticipantPresentation {
    pub principal_id: PrincipalId,
    pub label: SharedString,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmParticipantRow {
    principal_id: PrincipalId,
    label: SharedString,
    state: DmParticipantState,
    current_viewer: bool,
}

impl DmParticipantRow {
    pub const fn principal_id(&self) -> PrincipalId {
        self.principal_id
    }

    pub fn label(&self) -> &SharedString {
        &self.label
    }

    pub const fn state(&self) -> DmParticipantState {
        self.state
    }

    pub const fn is_current_viewer(&self) -> bool {
        self.current_viewer
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmNavigationRow {
    dm_id: AggregateId,
    label: SharedString,
    active_participant_count: usize,
}

impl DmNavigationRow {
    pub const fn dm_id(&self) -> AggregateId {
        self.dm_id
    }

    pub fn label(&self) -> &SharedString {
        &self.label
    }

    pub const fn active_participant_count(&self) -> usize {
        self.active_participant_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmDecryptionFailureKind {
    MissingKey,
    MalformedEnvelope,
    UnsupportedVersion,
}

impl DmDecryptionFailureKind {
    pub const fn user_message(self) -> &'static str {
        match self {
            Self::MissingKey => "Unable to decrypt this message because its key is unavailable.",
            Self::MalformedEnvelope => "Unable to decrypt this malformed message.",
            Self::UnsupportedVersion => {
                "Unable to decrypt this message because its encryption version is unsupported."
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DmDecryptionFailure {
    pub event_id: NostrEventId,
    pub sender_id: PrincipalId,
    pub occurred_at: u64,
    pub kind: DmDecryptionFailureKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmReconnectOutcome {
    Available,
    ViewerRemoved,
}

#[derive(Debug)]
pub enum DmViewError {
    ViewerNotActive,
    NotAvailable,
    MismatchedConversation,
    InvalidParticipantPresentation,
    InvalidDecryptionFailure,
    Timeline(MessageTimelineError),
}

impl fmt::Display for DmViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ViewerNotActive => {
                formatter.write_str("the current viewer is not an active DM participant")
            }
            Self::NotAvailable => formatter.write_str("the DM is not available"),
            Self::MismatchedConversation => {
                formatter.write_str("the reconnected DM does not match the open conversation")
            }
            Self::InvalidParticipantPresentation => {
                formatter.write_str("the DM participant presentation is invalid")
            }
            Self::InvalidDecryptionFailure => {
                formatter.write_str("the DM decryption failure is invalid")
            }
            Self::Timeline(error) => error.fmt(formatter),
        }
    }
}

impl Error for DmViewError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Timeline(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MessageTimelineError> for DmViewError {
    fn from(error: MessageTimelineError) -> Self {
        Self::Timeline(error)
    }
}

#[derive(Clone)]
struct ActiveDmViewState {
    direct_message: DirectMessage,
    participants: Vec<DmParticipantRow>,
    navigation_label: SharedString,
    hidden: bool,
    decryption_failures: Vec<DmDecryptionFailure>,
}

enum DmViewAccessState {
    Active(ActiveDmViewState),
    ViewerRemoved,
}

pub struct DmView {
    community_id: CommunityId,
    dm_id: AggregateId,
    viewer_id: PrincipalId,
    access: DmViewAccessState,
    timeline: Option<Entity<MessageTimeline>>,
}

impl DmView {
    pub fn new(
        direct_message: DirectMessage,
        viewer_id: PrincipalId,
        participant_presentations: Vec<DmParticipantPresentation>,
        hidden: bool,
        cx: &mut Context<Self>,
    ) -> Result<Self, DmViewError> {
        if !direct_message.is_active_participant(viewer_id) {
            return Err(DmViewError::ViewerNotActive);
        }
        let community_id = direct_message.fields().community_id;
        let dm_id = direct_message.fields().dm_id;
        let active = active_state(
            direct_message,
            viewer_id,
            participant_presentations,
            hidden,
            Vec::new(),
        )?;
        Ok(Self {
            community_id,
            dm_id,
            viewer_id,
            access: DmViewAccessState::Active(active),
            timeline: Some(cx.new(MessageTimeline::new)),
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn dm_id(&self) -> AggregateId {
        self.dm_id
    }

    pub const fn viewer_id(&self) -> PrincipalId {
        self.viewer_id
    }

    pub fn is_available(&self) -> bool {
        matches!(&self.access, DmViewAccessState::Active(_))
    }

    pub fn is_hidden(&self) -> bool {
        match &self.access {
            DmViewAccessState::Active(active) => active.hidden,
            DmViewAccessState::ViewerRemoved => true,
        }
    }

    pub fn navigation_row(&self) -> Option<DmNavigationRow> {
        let DmViewAccessState::Active(active) = &self.access else {
            return None;
        };
        if active.hidden {
            return None;
        }
        Some(DmNavigationRow {
            dm_id: self.dm_id,
            label: active.navigation_label.clone(),
            active_participant_count: active
                .participants
                .iter()
                .filter(|participant| participant.state == DmParticipantState::Active)
                .count(),
        })
    }

    pub fn participant_rows(&self) -> Option<&[DmParticipantRow]> {
        match &self.access {
            DmViewAccessState::Active(active) => Some(&active.participants),
            DmViewAccessState::ViewerRemoved => None,
        }
    }

    pub fn decryption_failures(&self) -> Option<&[DmDecryptionFailure]> {
        match &self.access {
            DmViewAccessState::Active(active) => Some(&active.decryption_failures),
            DmViewAccessState::ViewerRemoved => None,
        }
    }

    pub fn message_timeline(&self) -> Option<Entity<MessageTimeline>> {
        self.timeline.clone()
    }

    pub fn set_hidden(&mut self, hidden: bool, cx: &mut Context<Self>) -> Result<(), DmViewError> {
        let DmViewAccessState::Active(active) = &mut self.access else {
            return Err(DmViewError::NotAvailable);
        };
        active.hidden = hidden;
        cx.notify();
        Ok(())
    }

    pub fn apply_history_page(
        &mut self,
        page: MessageTimelinePage,
        cx: &mut Context<Self>,
    ) -> Result<(), DmViewError> {
        let timeline = self
            .timeline
            .as_ref()
            .ok_or(DmViewError::NotAvailable)?
            .clone();
        timeline.update(cx, |timeline, cx| timeline.apply_history_page(page, cx))?;
        Ok(())
    }

    pub fn replace_decryption_failures(
        &mut self,
        failures: Vec<DmDecryptionFailure>,
        cx: &mut Context<Self>,
    ) -> Result<(), DmViewError> {
        let DmViewAccessState::Active(active) = &mut self.access else {
            return Err(DmViewError::NotAvailable);
        };
        let failures = validate_failures(&active.direct_message, failures)?;
        active.decryption_failures = failures;
        cx.notify();
        Ok(())
    }

    pub fn reconnect(
        &mut self,
        direct_message: DirectMessage,
        participant_presentations: Vec<DmParticipantPresentation>,
        hidden: bool,
        pages: Vec<MessageTimelinePage>,
        failures: Vec<DmDecryptionFailure>,
        cx: &mut Context<Self>,
    ) -> Result<DmReconnectOutcome, DmViewError> {
        if direct_message.fields().community_id != self.community_id
            || direct_message.fields().dm_id != self.dm_id
        {
            return Err(DmViewError::MismatchedConversation);
        }
        if !direct_message.is_active_participant(self.viewer_id) {
            self.access = DmViewAccessState::ViewerRemoved;
            self.timeline = None;
            cx.notify();
            return Ok(DmReconnectOutcome::ViewerRemoved);
        }

        let mut active = active_state(
            direct_message,
            self.viewer_id,
            participant_presentations,
            hidden,
            failures,
        )?;
        active.decryption_failures = validate_failures(
            &active.direct_message,
            std::mem::take(&mut active.decryption_failures),
        )?;
        let timeline = cx.new(MessageTimeline::new);
        for page in pages {
            timeline.update(cx, |timeline, cx| timeline.apply_history_page(page, cx))?;
        }

        self.access = DmViewAccessState::Active(active);
        self.timeline = Some(timeline);
        cx.notify();
        Ok(DmReconnectOutcome::Available)
    }
}

impl Render for DmView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let DmViewAccessState::Active(active) = &self.access else {
            return v_flex()
                .id("dm-view-unavailable")
                .size_full()
                .items_center()
                .justify_center()
                .role(Role::Alert)
                .aria_label("Direct message unavailable")
                .child(
                    Label::new("This direct message is no longer available.").color(Color::Muted),
                )
                .into_any_element();
        };
        let timeline = self.timeline.clone();
        let participants = active.participants.clone();
        let failures = active.decryption_failures.clone();
        let navigation_label = active.navigation_label.clone();

        v_flex()
            .id("dm-view")
            .size_full()
            .role(Role::Region)
            .aria_label(format!("Direct message with {navigation_label}"))
            .child(
                v_flex()
                    .flex_none()
                    .gap_1()
                    .p_3()
                    .border_b_1()
                    .child(Label::new(navigation_label).size(LabelSize::Large))
                    .child(h_flex().gap_2().children(participants.into_iter().map(
                        |participant| {
                            let state = participant_state_label(participant.state);
                            Label::new(format!("{} · {state}", participant.label))
                                .size(LabelSize::Small)
                                .color(if participant.state == DmParticipantState::Active {
                                    Color::Default
                                } else {
                                    Color::Muted
                                })
                        },
                    ))),
            )
            .children(failures.into_iter().map(|failure| {
                let message = failure.kind.user_message();
                h_flex()
                    .flex_none()
                    .gap_2()
                    .px_3()
                    .py_2()
                    .child(Icon::new(IconName::Warning).color(Color::Error))
                    .child(
                        Label::new(message)
                            .size(LabelSize::Small)
                            .color(Color::Error),
                    )
            }))
            .when_some(timeline, |this, timeline| {
                this.child(div().flex_1().overflow_hidden().child(timeline))
            })
            .into_any_element()
    }
}

fn active_state(
    direct_message: DirectMessage,
    viewer_id: PrincipalId,
    participant_presentations: Vec<DmParticipantPresentation>,
    hidden: bool,
    decryption_failures: Vec<DmDecryptionFailure>,
) -> Result<ActiveDmViewState, DmViewError> {
    let participant_states = &direct_message.fields().participant_states;
    let mut presentations = BTreeMap::new();
    for presentation in participant_presentations {
        if presentation.label.trim().is_empty()
            || presentations
                .insert(presentation.principal_id, presentation.label)
                .is_some()
        {
            return Err(DmViewError::InvalidParticipantPresentation);
        }
    }
    if presentations.len() != participant_states.len()
        || presentations
            .keys()
            .any(|principal_id| !participant_states.contains_key(principal_id))
    {
        return Err(DmViewError::InvalidParticipantPresentation);
    }

    let mut participants = participant_states
        .iter()
        .map(|(principal_id, state)| {
            let label = presentations
                .remove(principal_id)
                .ok_or(DmViewError::InvalidParticipantPresentation)?;
            Ok(DmParticipantRow {
                principal_id: *principal_id,
                label,
                state: *state,
                current_viewer: *principal_id == viewer_id,
            })
        })
        .collect::<Result<Vec<_>, DmViewError>>()?;
    participants.sort_by(|left, right| {
        left.label
            .cmp(&right.label)
            .then_with(|| left.principal_id.cmp(&right.principal_id))
    });
    let navigation_label = navigation_label(&participants, viewer_id);
    Ok(ActiveDmViewState {
        direct_message,
        participants,
        navigation_label,
        hidden,
        decryption_failures,
    })
}

fn navigation_label(participants: &[DmParticipantRow], viewer_id: PrincipalId) -> SharedString {
    let labels = participants
        .iter()
        .filter(|participant| {
            participant.principal_id != viewer_id && participant.state == DmParticipantState::Active
        })
        .map(|participant| participant.label.as_ref())
        .collect::<Vec<_>>();
    if labels.is_empty() {
        "Direct message".into()
    } else {
        labels.join(", ").into()
    }
}

fn validate_failures(
    direct_message: &DirectMessage,
    mut failures: Vec<DmDecryptionFailure>,
) -> Result<Vec<DmDecryptionFailure>, DmViewError> {
    failures.sort_by_key(|failure| (failure.occurred_at, failure.event_id));
    let participant_states = &direct_message.fields().participant_states;
    let mut event_ids = BTreeSet::new();
    for failure in &failures {
        if failure.event_id.as_bytes().iter().all(|byte| *byte == 0)
            || failure.sender_id.as_uuid().is_nil()
            || !participant_states.contains_key(&failure.sender_id)
            || !event_ids.insert(failure.event_id)
        {
            return Err(DmViewError::InvalidDecryptionFailure);
        }
    }
    Ok(failures)
}

const fn participant_state_label(state: DmParticipantState) -> &'static str {
    match state {
        DmParticipantState::Active => "Active",
        DmParticipantState::Left => "Left",
        DmParticipantState::Removed => "Removed",
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, TimeZone as _, Utc};
    use collaboration_domain::{
        AggregateVersion, AuthenticatedPrincipal, AuthorizationAction, AuthorizationRequest,
        AuthorizationResource, AuthorizationResourceKind, AuthorizationScope, ChannelMembership,
        CommunityMembership, DmCommandOutcome, DmOpenFields, MembershipRole, MembershipStatus,
        PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
    };
    use gpui::TestAppContext;
    use uuid::Uuid;

    use crate::message_timeline::{
        MessageTimelineAuthor, MessageTimelineAuthorKind, MessageTimelineContext,
        MessageTimelineEntry,
    };

    use super::*;

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn dm_id() -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(10))
    }

    fn viewer_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(20))
    }

    fn peer_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(21))
    }

    fn outsider_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(22))
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id(), "dm-view-test")
                    .expect("trusted route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("collaboration:dms:write").expect("scope")
    }

    fn principal(principal_id: PrincipalId, scope: &AuthorizationScope) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            principal_id,
            community_id(),
            ServiceAccountId::new(principal_id.as_uuid().as_u128() as u64),
            PrincipalScopes::new([scope.clone()]).expect("scopes"),
        )
    }

    fn community_membership(principal_id: PrincipalId) -> CommunityMembership {
        CommunityMembership {
            community_id: community_id(),
            principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        }
    }

    fn open_authorization<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(community_id().as_uuid()),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(community_membership(principal.principal_id())),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        }
    }

    fn channel_authorization<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action: AuthorizationAction::Write,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: dm_id(),
                owner_principal_id: None,
                channel_id: Some(dm_id()),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(community_membership(principal.principal_id())),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community_id(),
                channel_id: dm_id(),
                principal_id: principal.principal_id(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 100,
        }
    }

    fn direct_message() -> DirectMessage {
        let tenant = tenant();
        let scope = scope();
        let principal = principal(viewer_id(), &scope);
        DirectMessage::open(
            DmOpenFields {
                community_id: community_id(),
                dm_id: dm_id(),
                participants: vec![viewer_id(), peer_id()],
            },
            &open_authorization(&tenant, &principal, &scope),
        )
        .expect("direct message")
    }

    fn presentations() -> Vec<DmParticipantPresentation> {
        vec![
            DmParticipantPresentation {
                principal_id: viewer_id(),
                label: "You".into(),
            },
            DmParticipantPresentation {
                principal_id: peer_id(),
                label: "Avery".into(),
            },
        ]
    }

    fn timestamp(second: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 22, 12, 0, second)
            .single()
            .expect("timestamp")
    }

    fn message_entry(event_id: &str, second: u32) -> MessageTimelineEntry {
        MessageTimelineEntry {
            event_id: event_id.into(),
            operation_id: None,
            source_version: 1,
            author: MessageTimelineAuthor {
                kind: MessageTimelineAuthorKind::Human,
                id: peer_id().to_string(),
                label: "Avery".into(),
            },
            content: format!("message {event_id}"),
            reply_to: None,
            edited: false,
            deleted: false,
            reactions: Vec::new(),
            occurred_at: timestamp(second),
            projected_at: timestamp(second),
            context: MessageTimelineContext {
                community_id: Some(community_id().to_string()),
                project_id: None,
                thread_id: Some(dm_id().to_string()),
            },
        }
    }

    fn page(event_id: &str, second: u32) -> MessageTimelinePage {
        MessageTimelinePage {
            request_cursor: None,
            next_cursor: None,
            entries: vec![message_entry(event_id, second)],
        }
    }

    fn timeline_event_ids(view: &Entity<DmView>, cx: &TestAppContext) -> Vec<String> {
        view.read_with(cx, |view, cx| {
            let timeline = view.message_timeline().expect("authorized timeline");
            let projected = timeline.read(cx).timeline();
            projected
                .read(cx)
                .items()
                .iter()
                .map(|item| item.id.source_id().to_owned())
                .collect()
        })
    }

    fn failure(value: u8, sender_id: PrincipalId, occurred_at: u64) -> DmDecryptionFailure {
        DmDecryptionFailure {
            event_id: NostrEventId::from_bytes([value; 32]),
            sender_id,
            occurred_at,
            kind: DmDecryptionFailureKind::MissingKey,
        }
    }

    #[gpui::test]
    fn dm_view_opens_authorized_navigation_participants_and_native_timeline(
        cx: &mut TestAppContext,
    ) {
        let view = cx.new(|cx| {
            DmView::new(direct_message(), viewer_id(), presentations(), false, cx)
                .expect("authorized DM")
        });
        view.update(cx, |view, cx| {
            view.apply_history_page(page("event-1", 1), cx)
        })
        .expect("history");

        view.read_with(cx, |view, _| {
            let row = view.navigation_row().expect("visible navigation row");
            assert_eq!(row.dm_id(), dm_id());
            assert_eq!(row.label().as_ref(), "Avery");
            assert_eq!(row.active_participant_count(), 2);
            let participants = view.participant_rows().expect("participants");
            assert_eq!(participants.len(), 2);
            assert!(
                participants
                    .iter()
                    .all(|row| row.state() == DmParticipantState::Active)
            );
            assert_eq!(
                participants
                    .iter()
                    .filter(|row| row.is_current_viewer())
                    .count(),
                1
            );
        });
        assert_eq!(timeline_event_ids(&view, cx), ["event-1"]);
    }

    #[gpui::test]
    fn dm_view_hide_and_reopen_change_navigation_without_discarding_timeline(
        cx: &mut TestAppContext,
    ) {
        let view = cx.new(|cx| {
            DmView::new(direct_message(), viewer_id(), presentations(), false, cx)
                .expect("authorized DM")
        });
        view.update(cx, |view, cx| {
            view.apply_history_page(page("event-1", 1), cx)
        })
        .expect("history");
        let timeline_id = view.read_with(cx, |view, _| {
            view.message_timeline().expect("timeline").entity_id()
        });

        view.update(cx, |view, cx| view.set_hidden(true, cx))
            .expect("hide");
        view.read_with(cx, |view, _| {
            assert!(view.is_hidden());
            assert!(view.navigation_row().is_none());
            assert!(view.participant_rows().is_some());
            assert_eq!(
                view.message_timeline().expect("timeline").entity_id(),
                timeline_id
            );
        });
        assert_eq!(timeline_event_ids(&view, cx), ["event-1"]);

        view.update(cx, |view, cx| view.set_hidden(false, cx))
            .expect("reopen");
        assert!(view.read_with(cx, |view, _| view.navigation_row().is_some()));
    }

    #[gpui::test]
    fn dm_view_exposes_closed_decryption_failures_without_ciphertext(cx: &mut TestAppContext) {
        let view = cx.new(|cx| {
            DmView::new(direct_message(), viewer_id(), presentations(), false, cx)
                .expect("authorized DM")
        });
        view.update(cx, |view, cx| {
            view.replace_decryption_failures(
                vec![failure(2, peer_id(), 2), failure(1, peer_id(), 1)],
                cx,
            )
        })
        .expect("failures");
        let failures = view.read_with(cx, |view, _| {
            view.decryption_failures().expect("failures").to_vec()
        });
        assert_eq!(
            failures
                .iter()
                .map(|failure| failure.event_id)
                .collect::<Vec<_>>(),
            [
                NostrEventId::from_bytes([1; 32]),
                NostrEventId::from_bytes([2; 32])
            ]
        );
        assert!(
            failures[0]
                .kind
                .user_message()
                .starts_with("Unable to decrypt")
        );
        assert!(!format!("{failures:?}").contains("ciphertext-secret"));

        let previous = failures;
        let result = view.update(cx, |view, cx| {
            view.replace_decryption_failures(vec![failure(3, outsider_id(), 3)], cx)
        });
        assert!(matches!(result, Err(DmViewError::InvalidDecryptionFailure)));
        assert_eq!(
            view.read_with(cx, |view, _| {
                view.decryption_failures().expect("failures").to_vec()
            }),
            previous
        );
    }

    #[gpui::test]
    fn dm_view_removal_clears_navigation_participants_failures_and_timeline(
        cx: &mut TestAppContext,
    ) {
        let mut removed_dm = direct_message();
        let tenant = tenant();
        let scope = scope();
        let peer = principal(peer_id(), &scope);
        assert_eq!(
            removed_dm.remove_participant(
                AggregateVersion::FIRST,
                viewer_id(),
                &channel_authorization(&tenant, &peer, &scope),
            ),
            Ok(DmCommandOutcome::Applied)
        );

        let view = cx.new(|cx| {
            DmView::new(direct_message(), viewer_id(), presentations(), false, cx)
                .expect("authorized DM")
        });
        view.update(cx, |view, cx| {
            view.apply_history_page(page("event-1", 1), cx)?;
            view.replace_decryption_failures(vec![failure(1, peer_id(), 1)], cx)
        })
        .expect("private state");
        let outcome = view
            .update(cx, |view, cx| {
                view.reconnect(removed_dm, Vec::new(), false, Vec::new(), Vec::new(), cx)
            })
            .expect("removal");

        assert_eq!(outcome, DmReconnectOutcome::ViewerRemoved);
        view.read_with(cx, |view, _| {
            assert!(!view.is_available());
            assert!(view.navigation_row().is_none());
            assert!(view.participant_rows().is_none());
            assert!(view.decryption_failures().is_none());
            assert!(view.message_timeline().is_none());
        });
        assert!(matches!(
            view.update(cx, |view, cx| view.set_hidden(false, cx)),
            Err(DmViewError::NotAvailable)
        ));
    }

    #[gpui::test]
    fn dm_view_reconnect_atomically_rebuilds_authoritative_timeline(cx: &mut TestAppContext) {
        let view = cx.new(|cx| {
            DmView::new(direct_message(), viewer_id(), presentations(), true, cx)
                .expect("authorized DM")
        });
        view.update(cx, |view, cx| {
            view.apply_history_page(page("stale-event", 1), cx)
        })
        .expect("stale history");
        let stale_timeline_id = view.read_with(cx, |view, _| {
            view.message_timeline().expect("timeline").entity_id()
        });

        let outcome = view
            .update(cx, |view, cx| {
                view.reconnect(
                    direct_message(),
                    presentations(),
                    false,
                    vec![page("authoritative-event", 2)],
                    vec![failure(2, peer_id(), 2)],
                    cx,
                )
            })
            .expect("reconnect");
        assert_eq!(outcome, DmReconnectOutcome::Available);
        assert_eq!(timeline_event_ids(&view, cx), ["authoritative-event"]);
        let authoritative_timeline_id = view.read_with(cx, |view, _| {
            assert!(!view.is_hidden());
            assert!(view.navigation_row().is_some());
            view.message_timeline().expect("timeline").entity_id()
        });
        assert_ne!(authoritative_timeline_id, stale_timeline_id);

        let invalid_page = MessageTimelinePage {
            request_cursor: Some("unexpected".into()),
            next_cursor: None,
            entries: vec![message_entry("invalid-event", 3)],
        };
        let result = view.update(cx, |view, cx| {
            view.reconnect(
                direct_message(),
                presentations(),
                false,
                vec![invalid_page],
                Vec::new(),
                cx,
            )
        });
        assert!(matches!(result, Err(DmViewError::Timeline(_))));
        assert_eq!(timeline_event_ids(&view, cx), ["authoritative-event"]);
        assert_eq!(
            view.read_with(cx, |view, _| {
                view.message_timeline().expect("timeline").entity_id()
            }),
            authoritative_timeline_id
        );
    }
}
