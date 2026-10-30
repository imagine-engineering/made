use std::sync::Arc;

use gpui::{App, IntoElement, RenderOnce, Role, Window};
use ui::{Button, ButtonStyle, Color, Label, LabelSize, TintColor, Tooltip, prelude::*};

use crate::activity_projection::{
    ActivityActorKind, ActivityDetailHandle, ActivityItem, ActivityItemId, ActivityLifecycle,
    ActivityLink, ActivityObjectKind, ActivityOutcomeStatus, ActivitySemanticClass,
};

#[cfg(feature = "multiplayer-tools")]
use crate::activity_reconciliation::{ActivityProvenanceClass, ActivitySourceProvenance};

pub type ActivityCardToggleHandler = Arc<dyn Fn(&mut Window, &mut App)>;
pub type ActivityCardInterventionHandler =
    Arc<dyn Fn(ActivityCardIntervention, &mut Window, &mut App)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityCardKind {
    Message,
    Operation,
    Search,
    FileEdit,
    Command,
    Test,
    Thought,
    Plan,
    Permission,
    Error,
    Lifecycle,
    Raw,
    Suppressed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityCardTone {
    Neutral,
    Progress,
    Success,
    Attention,
    Error,
    Muted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityInterventionKind {
    ReviewPermission,
    InspectError,
    ResumeWaiting,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivityCardIntervention {
    pub kind: ActivityInterventionKind,
    pub label: String,
    pub link: ActivityLink,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivityCardSource {
    pub id: ActivityItemId,
    pub provenance: Option<&'static str>,
}

#[cfg(feature = "multiplayer-tools")]
impl From<ActivitySourceProvenance> for ActivityCardSource {
    fn from(source: ActivitySourceProvenance) -> Self {
        Self {
            id: source.source_id,
            provenance: Some(match source.class {
                ActivityProvenanceClass::Compatibility => "compatibility",
                ActivityProvenanceClass::Streaming => "streaming",
                ActivityProvenanceClass::Authoritative => "authoritative",
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivityCardPresentation {
    pub kind: ActivityCardKind,
    pub tone: ActivityCardTone,
    pub summary: String,
    pub kind_label: &'static str,
    pub lifecycle_label: &'static str,
    pub outcome: Option<String>,
    pub detail: Option<String>,
    pub raw_sources: Vec<String>,
    pub waiting_for_user: bool,
    pub interventions: Vec<ActivityCardIntervention>,
}

impl ActivityCardPresentation {
    pub fn new(item: &ActivityItem, sources: &[ActivityCardSource]) -> Self {
        let kind = activity_card_kind(item);
        let waiting_for_user = item.lifecycle == ActivityLifecycle::WaitingForUser;
        let raw_sources = if exposes_raw_source(item, kind) {
            sources
                .iter()
                .map(|source| match source.provenance {
                    Some(provenance) => format!(
                        "{provenance} {:?} {}",
                        source.id.source_kind(),
                        source.id.source_id()
                    ),
                    None => format!(
                        "projected {:?} {}",
                        source.id.source_kind(),
                        source.id.source_id()
                    ),
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            kind,
            tone: activity_card_tone(item, waiting_for_user),
            summary: activity_card_summary(item, kind),
            kind_label: activity_card_kind_label(kind),
            lifecycle_label: activity_lifecycle_label(item.lifecycle),
            outcome: item.outcome.summary.clone(),
            detail: item
                .details
                .as_ref()
                .map(activity_detail_summary)
                .or_else(|| {
                    exposes_raw_source(item, kind).then(|| {
                        format!(
                            "Unsupported {:?} event {}",
                            item.id.source_kind(),
                            item.id.source_id()
                        )
                    })
                }),
            raw_sources,
            waiting_for_user,
            interventions: activity_interventions(item, waiting_for_user),
        }
    }

    pub fn has_progressive_detail(&self) -> bool {
        self.detail.is_some() || !self.raw_sources.is_empty()
    }

    pub fn accessibility_label(&self) -> String {
        let mut label = format!(
            "{}. {}. {}",
            self.summary, self.kind_label, self.lifecycle_label
        );
        if let Some(outcome) = &self.outcome {
            label.push_str(". ");
            label.push_str(outcome);
        }
        if self.waiting_for_user {
            label.push_str(". Intervention required");
        }
        label
    }
}

#[derive(IntoElement)]
pub struct CollaborativeActivityCard {
    index: usize,
    item: ActivityItem,
    sources: Vec<ActivityCardSource>,
    expanded: bool,
    quoted_reply: Option<(String, String)>,
    on_toggle: Option<ActivityCardToggleHandler>,
    on_intervention: Option<ActivityCardInterventionHandler>,
}

impl CollaborativeActivityCard {
    pub fn new(index: usize, item: ActivityItem, expanded: bool) -> Self {
        let sources = vec![ActivityCardSource {
            id: item.id.clone(),
            provenance: None,
        }];
        Self {
            index,
            item,
            sources,
            expanded,
            quoted_reply: None,
            on_toggle: None,
            on_intervention: None,
        }
    }

    pub fn with_quoted_reply(mut self, quoted_reply: Option<(String, String)>) -> Self {
        self.quoted_reply = quoted_reply;
        self
    }

    pub fn with_sources(mut self, sources: impl IntoIterator<Item = ActivityCardSource>) -> Self {
        self.sources = sources.into_iter().collect();
        self
    }

    pub fn on_toggle(mut self, handler: ActivityCardToggleHandler) -> Self {
        self.on_toggle = Some(handler);
        self
    }

    pub fn on_intervention(mut self, handler: ActivityCardInterventionHandler) -> Self {
        self.on_intervention = Some(handler);
        self
    }
}

impl RenderOnce for CollaborativeActivityCard {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let presentation = ActivityCardPresentation::new(&self.item, &self.sources);
        let actor = self.item.actor.clone();
        let object_label = self.item.object.label.clone();
        let tone_color = tone_color(presentation.tone);
        let background = match presentation.tone {
            ActivityCardTone::Attention => cx.theme().status().warning_background,
            ActivityCardTone::Error => cx.theme().status().error_background,
            _ => cx.theme().colors().editor_background,
        };
        let detail = self.expanded.then(|| presentation.detail.clone()).flatten();
        let raw_sources = self
            .expanded
            .then(|| presentation.raw_sources.clone())
            .unwrap_or_default();
        let interventions = presentation.interventions.clone();
        let on_intervention = self.on_intervention.clone();
        let conversational = presentation.kind == ActivityCardKind::Message;
        let actor_fallback_icon = match actor.kind {
            ActivityActorKind::Agent => IconName::AcpRegistry,
            ActivityActorKind::Human => IconName::Person,
            ActivityActorKind::Service => IconName::Server,
            ActivityActorKind::System => IconName::Settings,
        };

        v_flex()
            .id(("collaborative-activity-card", self.index))
            .role(Role::ListItem)
            .aria_label(presentation.accessibility_label())
            .w_full()
            .px_3()
            .py_2()
            .gap_1()
            .bg(background)
            .when(conversational, |this| {
                this.child(
                    h_flex()
                        .w_full()
                        .items_start()
                        .gap_3()
                        .child(
                            div()
                                .mt_0p5()
                                .size_5()
                                .flex_none()
                                .rounded_full()
                                .border_1()
                                .border_color(cx.theme().colors().border)
                                .bg(cx.theme().colors().element_background)
                                .flex()
                                .items_center()
                                .justify_center()
                                .child(
                                    Icon::new(actor_fallback_icon)
                                        .size(IconSize::XSmall)
                                        .color(Color::Muted),
                                ),
                        )
                        .child(
                            v_flex()
                                .min_w_0()
                                .flex_1()
                                .gap_1()
                                .child(
                                    Label::new(actor.label.clone())
                                        .size(LabelSize::XSmall)
                                        .color(Color::Muted),
                                )
                                .when_some(self.quoted_reply.clone(), |this, (author, content)| {
                                    this.child(
                                        v_flex()
                                            .id(("collaborative-quoted-reply", self.index))
                                            .role(Role::Group)
                                            .aria_label(format!("Quoted reply to {author}"))
                                            .border_l_2()
                                            .border_color(cx.theme().colors().text_accent)
                                            .px(DynamicSpacing::Base08.rems(cx))
                                            .py(DynamicSpacing::Base04.rems(cx))
                                            .bg(cx.theme().colors().element_background)
                                            .child(
                                                Label::new(author)
                                                    .size(LabelSize::Small)
                                                    .color(Color::Muted),
                                            )
                                            .child(Label::new(content).size(LabelSize::Default)),
                                    )
                                })
                                .child(
                                    div()
                                        .font_ui(cx)
                                        .text_size(rems_from_px(ui::fractal::BODY_SIZE))
                                        .line_height(relative(ui::fractal::BODY_LINE_HEIGHT))
                                        .child(object_label.clone()),
                                ),
                        )
                        .when(presentation.has_progressive_detail(), |this| {
                            this.when_some(self.on_toggle.clone(), |this, on_toggle| {
                                this.child(
                                    IconButton::new(
                                        ("collaborative-activity-details", self.index),
                                        if self.expanded {
                                            IconName::ChevronDown
                                        } else {
                                            IconName::ChevronRight
                                        },
                                    )
                                    .icon_size(IconSize::XSmall)
                                    .tooltip(Tooltip::text(if self.expanded {
                                        "Hide details"
                                    } else {
                                        "Show details"
                                    }))
                                    .aria_expanded(self.expanded)
                                    .on_click(move |_, window, cx| on_toggle(window, cx)),
                                )
                            })
                        }),
                )
            })
            .when(!conversational, |this| {
                this.child(
                    h_flex()
                        .w_full()
                        .items_start()
                        .justify_between()
                        .gap_3()
                        .child(
                            v_flex()
                                .min_w_0()
                                .gap_0p5()
                                .child(
                                    Label::new(presentation.summary.clone()).size(LabelSize::Small),
                                )
                                .child(
                                    h_flex()
                                        .gap_1()
                                        .child(
                                            Label::new(presentation.kind_label)
                                                .size(LabelSize::XSmall)
                                                .color(tone_color),
                                        )
                                        .child(
                                            Label::new(format!(
                                                "· {}",
                                                presentation.lifecycle_label
                                            ))
                                            .size(LabelSize::XSmall)
                                            .color(Color::Muted),
                                        ),
                                ),
                        )
                        .when(presentation.has_progressive_detail(), |this| {
                            this.when_some(self.on_toggle.clone(), |this, on_toggle| {
                                this.child(
                                    IconButton::new(
                                        ("collaborative-activity-details", self.index),
                                        if self.expanded {
                                            IconName::ChevronDown
                                        } else {
                                            IconName::ChevronRight
                                        },
                                    )
                                    .icon_size(IconSize::XSmall)
                                    .tooltip(Tooltip::text(if self.expanded {
                                        "Hide details"
                                    } else {
                                        "Show details"
                                    }))
                                    .aria_expanded(self.expanded)
                                    .on_click(move |_, window, cx| on_toggle(window, cx)),
                                )
                            })
                        }),
                )
            })
            .when(presentation.waiting_for_user, |this| {
                this.child(
                    Label::new("Waiting for you — this work will not continue without input.")
                        .size(LabelSize::Small)
                        .color(Color::Warning),
                )
            })
            .when_some(presentation.outcome, |this, outcome| {
                this.child(
                    Label::new(outcome)
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
            })
            .when(!interventions.is_empty(), |this| {
                this.child(
                    h_flex()
                        .gap_1()
                        .children(interventions.into_iter().enumerate().map(
                            |(action_index, intervention)| {
                                let on_intervention = on_intervention.clone();
                                let label = intervention.label.clone();
                                Button::new(
                                    format!(
                                        "collaborative-activity-intervention-{}-{action_index}",
                                        self.index
                                    ),
                                    label,
                                )
                                .style(ButtonStyle::Tinted(match intervention.kind {
                                    ActivityInterventionKind::ReviewPermission
                                    | ActivityInterventionKind::ResumeWaiting => TintColor::Warning,
                                    ActivityInterventionKind::InspectError => TintColor::Error,
                                }))
                                .label_size(LabelSize::Small)
                                .when_some(
                                    on_intervention,
                                    |this, handler| {
                                        this.on_click(move |_, window, cx| {
                                            handler(intervention.clone(), window, cx)
                                        })
                                    },
                                )
                            },
                        )),
                )
            })
            .when_some(detail, |this, detail| {
                this.child(
                    v_flex()
                        .p_2()
                        .gap_1()
                        .rounded_sm()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .bg(cx.theme().colors().editor_background)
                        .child(
                            Label::new("Details")
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                        )
                        .child(Label::new(detail).size(LabelSize::Small)),
                )
            })
            .when(!raw_sources.is_empty(), |this| {
                this.child(
                    v_flex()
                        .pl_2()
                        .gap_0p5()
                        .border_l_2()
                        .border_color(cx.theme().colors().border)
                        .child(
                            Label::new("Raw protocol provenance")
                                .size(LabelSize::XSmall)
                                .color(Color::Muted),
                        )
                        .children(
                            raw_sources
                                .into_iter()
                                .map(|source| Label::new(source).size(LabelSize::XSmall)),
                        ),
                )
            })
    }
}

pub(crate) fn activity_card_summary(item: &ActivityItem, kind: ActivityCardKind) -> String {
    match kind {
        ActivityCardKind::Raw => format!(
            "{} reported an unsupported activity event",
            item.actor.label
        ),
        ActivityCardKind::Suppressed => "Activity hidden by policy".into(),
        _ => format!("{} {} {}", item.actor.label, item.verb, item.object.label),
    }
}

pub(crate) fn activity_detail_summary(detail: &ActivityDetailHandle) -> String {
    match detail {
        ActivityDetailHandle::AcpEntry {
            session_id,
            entry_id,
        } => format!("ACP session {session_id}, entry {entry_id}"),
        ActivityDetailHandle::NativeAction { action_id } => {
            format!("Native action {action_id}")
        }
        ActivityDetailHandle::ProtocolEvent { event_id } => {
            format!("Protocol event {event_id}")
        }
        ActivityDetailHandle::GitChange {
            repository_id,
            change_id,
        } => format!("Repository {repository_id}, change {change_id}"),
        ActivityDetailHandle::WorkflowRun { run_id, step_id } => step_id.as_ref().map_or_else(
            || format!("Workflow run {run_id}"),
            |step_id| format!("Workflow run {run_id}, step {step_id}"),
        ),
        ActivityDetailHandle::RawSource { item_id } => format!(
            "Raw {:?} event {}",
            item_id.source_kind(),
            item_id.source_id()
        ),
    }
}

pub(crate) fn activity_lifecycle_label(lifecycle: ActivityLifecycle) -> &'static str {
    match lifecycle {
        ActivityLifecycle::Pending => "Pending",
        ActivityLifecycle::Running => "Running",
        ActivityLifecycle::WaitingForUser => "Waiting for you",
        ActivityLifecycle::Idle => "Idle",
        ActivityLifecycle::Succeeded => "Completed",
        ActivityLifecycle::Failed => "Failed",
        ActivityLifecycle::Cancelled => "Cancelled",
        ActivityLifecycle::TimedOut => "Timed out",
        ActivityLifecycle::Disconnected => "Disconnected",
        ActivityLifecycle::Suppressed => "Suppressed",
    }
}

fn activity_card_kind(item: &ActivityItem) -> ActivityCardKind {
    match item.class {
        ActivitySemanticClass::Message => ActivityCardKind::Message,
        ActivitySemanticClass::PlatformOperation => ActivityCardKind::Operation,
        ActivitySemanticClass::FileEdit => ActivityCardKind::FileEdit,
        ActivitySemanticClass::ShellCommand
            if item.object.kind == ActivityObjectKind::TestSuite =>
        {
            ActivityCardKind::Test
        }
        ActivitySemanticClass::ShellCommand => ActivityCardKind::Command,
        ActivitySemanticClass::Lifecycle => ActivityCardKind::Lifecycle,
        ActivitySemanticClass::Thought => ActivityCardKind::Thought,
        ActivitySemanticClass::Plan => ActivityCardKind::Plan,
        ActivitySemanticClass::Permission => ActivityCardKind::Permission,
        ActivitySemanticClass::Error => ActivityCardKind::Error,
        ActivitySemanticClass::Generic
            if item.object.kind == ActivityObjectKind::Tool
                && item.verb.eq_ignore_ascii_case("searched") =>
        {
            ActivityCardKind::Search
        }
        ActivitySemanticClass::Generic
            if matches!(
                item.details,
                Some(
                    ActivityDetailHandle::AcpEntry { .. }
                        | ActivityDetailHandle::NativeAction { .. }
                )
            ) =>
        {
            ActivityCardKind::Operation
        }
        ActivitySemanticClass::Generic | ActivitySemanticClass::Raw => ActivityCardKind::Raw,
        ActivitySemanticClass::Suppressed => ActivityCardKind::Suppressed,
    }
}

fn activity_card_kind_label(kind: ActivityCardKind) -> &'static str {
    match kind {
        ActivityCardKind::Message => "Message",
        ActivityCardKind::Operation => "Operation",
        ActivityCardKind::Search => "Search",
        ActivityCardKind::FileEdit => "File edit",
        ActivityCardKind::Command => "Command",
        ActivityCardKind::Test => "Test",
        ActivityCardKind::Thought => "Thought summary",
        ActivityCardKind::Plan => "Plan",
        ActivityCardKind::Permission => "Permission",
        ActivityCardKind::Error => "Error",
        ActivityCardKind::Lifecycle => "Session",
        ActivityCardKind::Raw => "Raw event",
        ActivityCardKind::Suppressed => "Suppressed",
    }
}

fn activity_card_tone(item: &ActivityItem, waiting_for_user: bool) -> ActivityCardTone {
    if waiting_for_user || item.class == ActivitySemanticClass::Permission {
        return ActivityCardTone::Attention;
    }
    if item.class == ActivitySemanticClass::Error
        || matches!(
            item.outcome.status,
            ActivityOutcomeStatus::Failure | ActivityOutcomeStatus::TimedOut
        )
    {
        return ActivityCardTone::Error;
    }
    match item.outcome.status {
        ActivityOutcomeStatus::Success => ActivityCardTone::Success,
        ActivityOutcomeStatus::Pending => ActivityCardTone::Progress,
        ActivityOutcomeStatus::Cancelled
        | ActivityOutcomeStatus::NoChange
        | ActivityOutcomeStatus::Unknown => ActivityCardTone::Muted,
        ActivityOutcomeStatus::Failure | ActivityOutcomeStatus::TimedOut => ActivityCardTone::Error,
    }
}

fn exposes_raw_source(item: &ActivityItem, kind: ActivityCardKind) -> bool {
    kind == ActivityCardKind::Raw
        || matches!(
            item.details,
            Some(
                ActivityDetailHandle::ProtocolEvent { .. } | ActivityDetailHandle::RawSource { .. }
            )
        )
}

fn activity_interventions(
    item: &ActivityItem,
    waiting_for_user: bool,
) -> Vec<ActivityCardIntervention> {
    let kind = if item.class == ActivitySemanticClass::Permission {
        Some(ActivityInterventionKind::ReviewPermission)
    } else if item.class == ActivitySemanticClass::Error
        || matches!(
            item.outcome.status,
            ActivityOutcomeStatus::Failure | ActivityOutcomeStatus::TimedOut
        )
    {
        Some(ActivityInterventionKind::InspectError)
    } else if waiting_for_user {
        Some(ActivityInterventionKind::ResumeWaiting)
    } else {
        None
    };
    let Some(kind) = kind else {
        return Vec::new();
    };
    item.links
        .iter()
        .cloned()
        .map(|link| ActivityCardIntervention {
            kind,
            label: intervention_label(kind, &link).into(),
            link,
        })
        .collect()
}

fn intervention_label(kind: ActivityInterventionKind, link: &ActivityLink) -> &'static str {
    match (kind, link) {
        (ActivityInterventionKind::ReviewPermission, ActivityLink::Action { .. }) => {
            "Review request"
        }
        (ActivityInterventionKind::InspectError, ActivityLink::Action { .. }) => {
            "Open failed action"
        }
        (ActivityInterventionKind::ResumeWaiting, ActivityLink::Action { .. }) => {
            "Open waiting action"
        }
        (_, ActivityLink::GitChange { .. }) => "Open change",
        (_, ActivityLink::Entity { .. }) => "Open context",
    }
}

const fn tone_color(tone: ActivityCardTone) -> Color {
    match tone {
        ActivityCardTone::Neutral => Color::Default,
        ActivityCardTone::Progress => Color::Info,
        ActivityCardTone::Success => Color::Success,
        ActivityCardTone::Attention => Color::Warning,
        ActivityCardTone::Error => Color::Error,
        ActivityCardTone::Muted => Color::Muted,
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone as _, Utc};

    use super::*;
    use crate::activity_projection::{
        ActivityActor, ActivityActorKind, ActivityContext, ActivityObject, ActivityOutcome,
        ActivitySourceKind, ActivityVisibility,
    };

    fn item(
        source_id: &str,
        class: ActivitySemanticClass,
        object_kind: ActivityObjectKind,
        verb: &str,
    ) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 24, 12, 0, 0)
            .single()
            .expect("valid fixture timestamp");
        ActivityItem {
            id: ActivityItemId::new(ActivitySourceKind::Acp, source_id).expect("valid fixture ID"),
            source_version: 1,
            class,
            actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Builder".into(),
            },
            verb: verb.into(),
            object: ActivityObject {
                kind: object_kind,
                id: Some(format!("{source_id}-object")),
                label: source_id.into(),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: Some("In progress".into()),
            },
            lifecycle: ActivityLifecycle::Running,
            occurred_at: timestamp,
            projected_at: timestamp,
            context: ActivityContext::default(),
            visibility: ActivityVisibility::Private,
            details: Some(ActivityDetailHandle::AcpEntry {
                session_id: "session-1".into(),
                entry_id: source_id.into(),
            }),
            links: Vec::new(),
        }
    }

    fn presentation(item: &ActivityItem) -> ActivityCardPresentation {
        ActivityCardPresentation::new(
            item,
            &[ActivityCardSource {
                id: item.id.clone(),
                provenance: None,
            }],
        )
    }

    #[gpui::test]
    fn collaborative_activity_cards_cover_semantic_work_kinds() {
        for (class, object_kind, verb, expected) in [
            (
                ActivitySemanticClass::Thought,
                ActivityObjectKind::Plan,
                "considered",
                ActivityCardKind::Thought,
            ),
            (
                ActivitySemanticClass::Plan,
                ActivityObjectKind::Plan,
                "planned",
                ActivityCardKind::Plan,
            ),
            (
                ActivitySemanticClass::Generic,
                ActivityObjectKind::Tool,
                "searched",
                ActivityCardKind::Search,
            ),
            (
                ActivitySemanticClass::FileEdit,
                ActivityObjectKind::File,
                "edited",
                ActivityCardKind::FileEdit,
            ),
            (
                ActivitySemanticClass::ShellCommand,
                ActivityObjectKind::Command,
                "ran",
                ActivityCardKind::Command,
            ),
            (
                ActivitySemanticClass::ShellCommand,
                ActivityObjectKind::TestSuite,
                "tested",
                ActivityCardKind::Test,
            ),
        ] {
            let item = item("work", class, object_kind, verb);
            let presentation = presentation(&item);
            assert_eq!(presentation.kind, expected);
            assert!(presentation.has_progressive_detail());
            assert!(presentation.interventions.is_empty());
        }
    }

    #[gpui::test]
    fn collaborative_activity_cards_surface_permission_error_and_waiting_actions() {
        let mut permission = item(
            "permission",
            ActivitySemanticClass::Permission,
            ActivityObjectKind::Permission,
            "requested",
        );
        permission.lifecycle = ActivityLifecycle::WaitingForUser;
        permission.links = vec![ActivityLink::Action {
            action_id: "permission-action".into(),
        }];
        let permission = presentation(&permission);
        assert_eq!(permission.tone, ActivityCardTone::Attention);
        assert!(permission.waiting_for_user);
        assert_eq!(
            permission.interventions[0].kind,
            ActivityInterventionKind::ReviewPermission
        );
        assert!(
            permission
                .accessibility_label()
                .contains("Intervention required")
        );

        let mut error = item(
            "error",
            ActivitySemanticClass::Error,
            ActivityObjectKind::Command,
            "failed",
        );
        error.lifecycle = ActivityLifecycle::Failed;
        error.outcome.status = ActivityOutcomeStatus::Failure;
        error.links = vec![ActivityLink::Action {
            action_id: "failed-action".into(),
        }];
        let error = presentation(&error);
        assert_eq!(error.tone, ActivityCardTone::Error);
        assert_eq!(
            error.interventions[0].kind,
            ActivityInterventionKind::InspectError
        );

        let mut waiting = item(
            "waiting",
            ActivitySemanticClass::Lifecycle,
            ActivityObjectKind::Session,
            "paused",
        );
        waiting.lifecycle = ActivityLifecycle::WaitingForUser;
        waiting.links = vec![ActivityLink::Entity {
            entity_kind: "session".into(),
            entity_id: "session-1".into(),
        }];
        assert_eq!(
            presentation(&waiting).interventions[0].kind,
            ActivityInterventionKind::ResumeWaiting
        );
    }

    #[gpui::test]
    fn collaborative_activity_cards_disclose_raw_protocol_provenance_progressively() {
        let mut raw = item(
            "future-event",
            ActivitySemanticClass::Generic,
            ActivityObjectKind::Other,
            "reported",
        );
        raw.details = Some(ActivityDetailHandle::RawSource {
            item_id: raw.id.clone(),
        });
        let presentation = ActivityCardPresentation::new(
            &raw,
            &[
                ActivityCardSource {
                    id: raw.id.clone(),
                    provenance: Some("compatibility"),
                },
                ActivityCardSource {
                    id: ActivityItemId::new(ActivitySourceKind::System, "authority")
                        .expect("valid fixture ID"),
                    provenance: Some("authoritative"),
                },
            ],
        );
        assert_eq!(presentation.kind, ActivityCardKind::Raw);
        assert_eq!(presentation.raw_sources.len(), 2);
        assert!(presentation.has_progressive_detail());
        assert!(presentation.summary.contains("unsupported"));
    }
}
