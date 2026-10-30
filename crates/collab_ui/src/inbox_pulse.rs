use std::{collections::BTreeMap, error::Error, fmt};

use agent_ui::activity_projection::{
    ActivityActorKind, ActivityItem, ActivitySemanticClass, ActivitySourceKind,
};
use collaboration_domain::{InboxCategory, InboxItem, InboxProjection, InboxScope};
use gpui::{
    AnyElement, Context, IntoElement, ListAlignment, ListState, Render, Role, Window, list, px,
};
use ui::{Button, ButtonStyle, LabelSize, prelude::*};

const MAX_PULSE_ITEMS: usize = 100_000;
const MAX_PAGE_SIZE: usize = 200;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InboxPulseMode {
    Inbox,
    Pulse,
}

impl InboxPulseMode {
    fn label(self) -> &'static str {
        match self {
            Self::Inbox => "Inbox",
            Self::Pulse => "Pulse",
        }
    }

    fn id(self) -> u64 {
        match self {
            Self::Inbox => 0,
            Self::Pulse => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InboxFilter {
    All,
    Unread,
    Mentions,
    Replies,
    Reminders,
}

impl InboxFilter {
    const ALL: [Self; 5] = [
        Self::All,
        Self::Unread,
        Self::Mentions,
        Self::Replies,
        Self::Reminders,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Unread => "Unread",
            Self::Mentions => "Mentions",
            Self::Replies => "Replies",
            Self::Reminders => "Reminders",
        }
    }

    fn id(self) -> u64 {
        match self {
            Self::All => 0,
            Self::Unread => 1,
            Self::Mentions => 2,
            Self::Replies => 3,
            Self::Reminders => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PulseFilter {
    All,
    People,
    Agents,
    System,
}

impl PulseFilter {
    const ALL: [Self; 4] = [Self::All, Self::People, Self::Agents, Self::System];

    fn label(self) -> &'static str {
        match self {
            Self::All => "Everyone",
            Self::People => "People",
            Self::Agents => "Agents",
            Self::System => "System",
        }
    }

    fn id(self) -> u64 {
        match self {
            Self::All => 0,
            Self::People => 1,
            Self::Agents => 2,
            Self::System => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InboxPulseFreshness {
    Loading,
    Fresh { revision: u64 },
    Stale { revision: u64 },
    Retrying { revision: u64 },
}

impl InboxPulseFreshness {
    pub const fn revision(self) -> Option<u64> {
        match self {
            Self::Loading => None,
            Self::Fresh { revision } | Self::Stale { revision } | Self::Retrying { revision } => {
                Some(revision)
            }
        }
    }

    pub const fn is_stale(self) -> bool {
        matches!(self, Self::Stale { .. } | Self::Retrying { .. })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InboxPulseRow {
    Inbox(InboxItem),
    Pulse(ActivityItem),
}

pub struct InboxPulseView {
    scope: InboxScope,
    mode: InboxPulseMode,
    inbox_filter: InboxFilter,
    pulse_filter: PulseFilter,
    freshness: InboxPulseFreshness,
    page_size: usize,
    visible_limit: usize,
    has_more: bool,
    inbox_items: Vec<InboxItem>,
    pulse_items: Vec<ActivityItem>,
    visible_rows: Vec<InboxPulseRow>,
    list_state: ListState,
}

impl InboxPulseView {
    pub fn new(scope: InboxScope, requested_page_size: usize) -> Self {
        let page_size = requested_page_size.clamp(1, MAX_PAGE_SIZE);
        Self {
            scope,
            mode: InboxPulseMode::Inbox,
            inbox_filter: InboxFilter::All,
            pulse_filter: PulseFilter::All,
            freshness: InboxPulseFreshness::Loading,
            page_size,
            visible_limit: page_size,
            has_more: false,
            inbox_items: Vec::new(),
            pulse_items: Vec::new(),
            visible_rows: Vec::new(),
            list_state: ListState::new(0, ListAlignment::Top, px(1024.0)),
        }
    }

    pub fn apply_snapshot(
        &mut self,
        revision: u64,
        inbox: InboxProjection,
        pulse: Vec<ActivityItem>,
        cx: &mut Context<Self>,
    ) -> Result<(), InboxPulseError> {
        if revision == 0
            || self
                .freshness
                .revision()
                .is_some_and(|current| revision <= current)
        {
            return Err(InboxPulseError::StaleRevision);
        }
        if inbox.scope() != self.scope {
            return Err(InboxPulseError::ScopeMismatch);
        }
        let pulse = normalize_pulse(self.scope, pulse)?;
        self.inbox_items = inbox.items().to_vec();
        self.pulse_items = pulse;
        self.freshness = InboxPulseFreshness::Fresh { revision };
        self.reset_page();
        self.rebuild_rows();
        cx.notify();
        Ok(())
    }

    pub fn mark_stale(&mut self, cx: &mut Context<Self>) -> Result<(), InboxPulseError> {
        let Some(revision) = self.freshness.revision() else {
            return Err(InboxPulseError::SnapshotUnavailable);
        };
        self.freshness = InboxPulseFreshness::Stale { revision };
        cx.notify();
        Ok(())
    }

    pub fn mark_retrying(&mut self, cx: &mut Context<Self>) -> Result<(), InboxPulseError> {
        let Some(revision) = self.freshness.revision() else {
            return Err(InboxPulseError::SnapshotUnavailable);
        };
        self.freshness = InboxPulseFreshness::Retrying { revision };
        cx.notify();
        Ok(())
    }

    pub fn set_mode(&mut self, mode: InboxPulseMode, cx: &mut Context<Self>) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        self.reset_page();
        self.rebuild_rows();
        cx.notify();
    }

    pub fn set_inbox_filter(&mut self, filter: InboxFilter, cx: &mut Context<Self>) {
        if self.inbox_filter == filter {
            return;
        }
        self.inbox_filter = filter;
        self.reset_page();
        self.rebuild_rows();
        cx.notify();
    }

    pub fn set_pulse_filter(&mut self, filter: PulseFilter, cx: &mut Context<Self>) {
        if self.pulse_filter == filter {
            return;
        }
        self.pulse_filter = filter;
        self.reset_page();
        self.rebuild_rows();
        cx.notify();
    }

    pub fn load_more(&mut self, cx: &mut Context<Self>) -> bool {
        if !self.has_more {
            return false;
        }
        self.visible_limit = self.visible_limit.saturating_add(self.page_size);
        self.rebuild_rows();
        cx.notify();
        true
    }

    pub const fn scope(&self) -> InboxScope {
        self.scope
    }

    pub const fn mode(&self) -> InboxPulseMode {
        self.mode
    }

    pub const fn inbox_filter(&self) -> InboxFilter {
        self.inbox_filter
    }

    pub const fn pulse_filter(&self) -> PulseFilter {
        self.pulse_filter
    }

    pub const fn freshness(&self) -> InboxPulseFreshness {
        self.freshness
    }

    pub fn visible_rows(&self) -> &[InboxPulseRow] {
        &self.visible_rows
    }

    pub const fn has_more(&self) -> bool {
        self.has_more
    }

    pub fn list_state(&self) -> &ListState {
        &self.list_state
    }

    fn reset_page(&mut self) {
        self.visible_limit = self.page_size;
    }

    fn rebuild_rows(&mut self) {
        let mut rows = match self.mode {
            InboxPulseMode::Inbox => self
                .inbox_items
                .iter()
                .filter(|item| matches_inbox_filter(item, self.inbox_filter))
                .cloned()
                .map(InboxPulseRow::Inbox)
                .collect::<Vec<_>>(),
            InboxPulseMode::Pulse => self
                .pulse_items
                .iter()
                .filter(|item| matches_pulse_filter(item, self.pulse_filter))
                .cloned()
                .map(InboxPulseRow::Pulse)
                .collect::<Vec<_>>(),
        };
        self.has_more = rows.len() > self.visible_limit;
        rows.truncate(self.visible_limit);
        self.visible_rows = rows;
        self.list_state.reset(self.visible_rows.len());
    }

    fn render_row(&self, index: usize, _window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let Some(row) = self.visible_rows.get(index) else {
            return div().into_any_element();
        };
        let (summary, detail, accessibility_label) = match row {
            InboxPulseRow::Inbox(item) => {
                let categories = item
                    .categories()
                    .iter()
                    .map(|category| match category {
                        InboxCategory::Activity => "Activity",
                        InboxCategory::Mention => "Mention",
                        InboxCategory::Reply => "Reply",
                        InboxCategory::Reminder => "Reminder",
                    })
                    .collect::<Vec<_>>()
                    .join(" · ");
                let detail = format!(
                    "{} message{} · {} unread",
                    item.message_count(),
                    if item.message_count() == 1 { "" } else { "s" },
                    item.unread_message_count()
                );
                (
                    categories.clone(),
                    detail.clone(),
                    format!("Inbox item. {categories}. {detail}"),
                )
            }
            InboxPulseRow::Pulse(item) => {
                let summary = if item.class == ActivitySemanticClass::Suppressed {
                    "Activity hidden by policy".to_owned()
                } else {
                    format!("{} {} {}", item.actor.label, item.verb, item.object.label)
                };
                let detail = format!(
                    "{} · {:?}",
                    pulse_actor_label(item.actor.kind),
                    item.lifecycle
                );
                (
                    summary.clone(),
                    detail.clone(),
                    format!("Pulse item. {summary}. {detail}"),
                )
            }
        };
        v_flex()
            .id(("inbox-pulse-row", index))
            .role(Role::ListItem)
            .aria_label(accessibility_label)
            .w_full()
            .px_4()
            .py_2()
            .gap_0p5()
            .border_b_1()
            .border_color(cx.theme().colors().border_variant)
            .child(div().text_ui(cx).child(summary))
            .child(
                div()
                    .text_xs()
                    .text_color(cx.theme().colors().text_muted)
                    .child(detail),
            )
            .into_any_element()
    }
}

impl Render for InboxPulseView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        let mode_buttons = [InboxPulseMode::Inbox, InboxPulseMode::Pulse].map(|mode| {
            Button::new(("inbox-pulse-mode", mode.id()), mode.label())
                .style(if self.mode == mode {
                    ButtonStyle::Filled
                } else {
                    ButtonStyle::Subtle
                })
                .label_size(LabelSize::Small)
                .aria_role(Role::Tab)
                .aria_value(if self.mode == mode {
                    "selected"
                } else {
                    "not selected"
                })
                .on_click(cx.listener(move |this, _, _window, cx| this.set_mode(mode, cx)))
        });
        let filter_buttons = match self.mode {
            InboxPulseMode::Inbox => InboxFilter::ALL
                .into_iter()
                .map(|filter| {
                    Button::new(("inbox-filter", filter.id()), filter.label())
                        .style(if self.inbox_filter == filter {
                            ButtonStyle::Filled
                        } else {
                            ButtonStyle::Subtle
                        })
                        .label_size(LabelSize::Small)
                        .aria_value(if self.inbox_filter == filter {
                            "selected"
                        } else {
                            "not selected"
                        })
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.set_inbox_filter(filter, cx)
                        }))
                        .into_any_element()
                })
                .collect::<Vec<_>>(),
            InboxPulseMode::Pulse => PulseFilter::ALL
                .into_iter()
                .map(|filter| {
                    Button::new(("pulse-filter", filter.id()), filter.label())
                        .style(if self.pulse_filter == filter {
                            ButtonStyle::Filled
                        } else {
                            ButtonStyle::Subtle
                        })
                        .label_size(LabelSize::Small)
                        .aria_value(if self.pulse_filter == filter {
                            "selected"
                        } else {
                            "not selected"
                        })
                        .on_click(cx.listener(move |this, _, _window, cx| {
                            this.set_pulse_filter(filter, cx)
                        }))
                        .into_any_element()
                })
                .collect::<Vec<_>>(),
        };
        let content = if self.visible_rows.is_empty() {
            div()
                .id("inbox-pulse-empty")
                .role(Role::Status)
                .aria_label(match self.freshness {
                    InboxPulseFreshness::Loading => "Loading collaborative activity",
                    _ => "No collaborative activity matches this filter",
                })
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(cx.theme().colors().text_muted)
                .child(match self.freshness {
                    InboxPulseFreshness::Loading => "Loading…",
                    _ => "No matching activity",
                })
                .into_any_element()
        } else {
            list(self.list_state.clone(), move |index, window, cx| {
                view.update(cx, |this, cx| this.render_row(index, window, cx))
            })
            .size_full()
            .into_any_element()
        };

        v_flex()
            .id("inbox-pulse-view")
            .size_full()
            .child(
                h_flex()
                    .w_full()
                    .gap_1()
                    .p_2()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .children(mode_buttons),
            )
            .child(
                h_flex()
                    .w_full()
                    .gap_1()
                    .px_2()
                    .py_1()
                    .flex_wrap()
                    .children(filter_buttons),
            )
            .when(self.freshness.is_stale(), |this| {
                this.child(
                    div()
                        .id("inbox-pulse-stale")
                        .role(Role::Status)
                        .aria_label("Collaborative activity may be stale")
                        .w_full()
                        .px_3()
                        .py_1()
                        .bg(cx.theme().status().warning_background.opacity(0.2))
                        .text_xs()
                        .child(match self.freshness {
                            InboxPulseFreshness::Retrying { .. } => {
                                "Showing cached activity while reconnecting"
                            }
                            _ => "Showing cached activity",
                        }),
                )
            })
            .child(div().flex_1().min_h_0().child(content))
            .when(self.has_more, |this| {
                this.child(
                    div().w_full().p_2().child(
                        Button::new("inbox-pulse-load-more", "Load more")
                            .style(ButtonStyle::Subtle)
                            .label_size(LabelSize::Small)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.load_more(cx);
                            })),
                    ),
                )
            })
    }
}

fn normalize_pulse(
    scope: InboxScope,
    pulse: Vec<ActivityItem>,
) -> Result<Vec<ActivityItem>, InboxPulseError> {
    if pulse.len() > MAX_PULSE_ITEMS {
        return Err(InboxPulseError::TooManyPulseItems);
    }
    let community_id = scope.community_id().to_string();
    let mut normalized = BTreeMap::new();
    for item in pulse {
        if item.source_version == 0 || item.context.community_id.as_deref() != Some(&community_id) {
            return Err(InboxPulseError::InvalidPulseItem);
        }
        let key = (
            activity_source_order(item.id.source_kind()),
            item.id.source_id().to_owned(),
        );
        match normalized.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(item);
            }
            std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &item => {}
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(InboxPulseError::ConflictingPulseDuplicate);
            }
        }
    }
    let mut normalized = normalized.into_values().collect::<Vec<_>>();
    normalized.sort_by(|left, right| {
        right
            .occurred_at
            .cmp(&left.occurred_at)
            .then_with(|| {
                activity_source_order(left.id.source_kind())
                    .cmp(&activity_source_order(right.id.source_kind()))
            })
            .then_with(|| left.id.source_id().cmp(right.id.source_id()))
    });
    Ok(normalized)
}

fn matches_inbox_filter(item: &InboxItem, filter: InboxFilter) -> bool {
    match filter {
        InboxFilter::All => true,
        InboxFilter::Unread => item.unread_message_count() > 0,
        InboxFilter::Mentions => item.categories().contains(&InboxCategory::Mention),
        InboxFilter::Replies => item.categories().contains(&InboxCategory::Reply),
        InboxFilter::Reminders => item.categories().contains(&InboxCategory::Reminder),
    }
}

fn matches_pulse_filter(item: &ActivityItem, filter: PulseFilter) -> bool {
    match filter {
        PulseFilter::All => true,
        PulseFilter::People => item.actor.kind == ActivityActorKind::Human,
        PulseFilter::Agents => item.actor.kind == ActivityActorKind::Agent,
        PulseFilter::System => matches!(
            item.actor.kind,
            ActivityActorKind::System | ActivityActorKind::Service
        ),
    }
}

fn pulse_actor_label(kind: ActivityActorKind) -> &'static str {
    match kind {
        ActivityActorKind::Human => "Person",
        ActivityActorKind::Agent => "Agent",
        ActivityActorKind::System => "System",
        ActivityActorKind::Service => "Service",
    }
}

fn activity_source_order(source: ActivitySourceKind) -> u8 {
    match source {
        ActivitySourceKind::Acp => 0,
        ActivitySourceKind::NativeAction => 1,
        ActivitySourceKind::Nostr => 2,
        ActivitySourceKind::Git => 3,
        ActivitySourceKind::Workflow => 4,
        ActivitySourceKind::Ci => 5,
        ActivitySourceKind::Moderation => 6,
        ActivitySourceKind::System => 7,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InboxPulseError {
    ScopeMismatch,
    StaleRevision,
    SnapshotUnavailable,
    TooManyPulseItems,
    InvalidPulseItem,
    ConflictingPulseDuplicate,
}

impl fmt::Display for InboxPulseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScopeMismatch => formatter.write_str("inbox/pulse scope mismatch"),
            Self::StaleRevision => formatter.write_str("inbox/pulse revision did not advance"),
            Self::SnapshotUnavailable => formatter.write_str("inbox/pulse snapshot unavailable"),
            Self::TooManyPulseItems => formatter.write_str("too many pulse items"),
            Self::InvalidPulseItem => formatter.write_str("invalid pulse item"),
            Self::ConflictingPulseDuplicate => {
                formatter.write_str("conflicting pulse item duplicate")
            }
        }
    }
}

impl Error for InboxPulseError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use agent_ui::activity_projection::{
        ActivityActor, ActivityContext, ActivityItemId, ActivityLifecycle, ActivityObject,
        ActivityObjectKind, ActivityOutcome, ActivityOutcomeStatus, ActivityVisibility,
    };
    use chrono::{TimeZone as _, Utc};
    use collaboration_domain::{
        AggregateId, AggregateVersion, CommunityId, ManualUnreadRegister, Message, MessageAuthor,
        MessageContent, MessageRecordFields, MessageSource, NostrEventId, OwnerReadStateReplica,
        PrincipalId, ReadContextId, ReadState, ReadStateCompleteness, ReadStateScope,
    };
    use gpui::TestAppContext;
    use uuid::Uuid;

    use super::*;

    #[derive(Clone, Copy)]
    struct MessageSpec {
        sequence: u32,
        read_through: u32,
        mention: bool,
        reply: bool,
    }

    fn community_id(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn scope() -> InboxScope {
        InboxScope::new(community_id(1), principal_id(2))
    }

    fn projection(specs: &[MessageSpec]) -> InboxProjection {
        let scope = scope();
        let messages = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let source = MessageSource {
                    event_id: NostrEventId::from_bytes([index as u8 + 1; 32]),
                    event_created_at: u64::from(spec.sequence),
                };
                Message::from_record(MessageRecordFields {
                    community_id: scope.community_id(),
                    channel_id: aggregate_id(50),
                    message_id: aggregate_id(100 + index as u128),
                    author: MessageAuthor::principal(principal_id(3)),
                    content: MessageContent::new(format!("message {index}"))
                        .expect("bounded message"),
                    lifecycle_state: collaboration_domain::MessageLifecycleState::Active,
                    source,
                    current_source: source,
                    mutations: Vec::new(),
                    version: AggregateVersion::FIRST,
                })
                .expect("valid canonical message")
            })
            .collect::<Vec<_>>();
        let contexts = specs
            .iter()
            .enumerate()
            .map(|(index, _)| {
                ReadContextId::new(format!("conversation:{index}")).expect("valid context")
            })
            .collect::<Vec<_>>();
        let mentions = specs
            .iter()
            .map(|spec| {
                if spec.mention {
                    BTreeSet::from([scope.viewer_principal_id()])
                } else {
                    BTreeSet::new()
                }
            })
            .collect::<Vec<_>>();
        let read_scope = ReadStateScope::new(scope.community_id(), scope.viewer_principal_id());
        let replica = OwnerReadStateReplica::new(
            read_scope,
            scope.viewer_principal_id(),
            contexts
                .iter()
                .cloned()
                .zip(specs.iter().map(|spec| spec.read_through)),
            Vec::<(ReadContextId, ManualUnreadRegister)>::new(),
        )
        .expect("valid read replica");
        let read_state =
            ReadState::from_replicas(read_scope, ReadStateCompleteness::Complete, [replica])
                .expect("valid read state");
        let inputs =
            specs
                .iter()
                .enumerate()
                .map(|(index, spec)| collaboration_domain::InboxMessageInput {
                    message: &messages[index],
                    conversation_id: aggregate_id(200 + index as u128),
                    read_context: &contexts[index],
                    parent_read_context: None,
                    sequence: spec.sequence,
                    mentioned_principal_ids: &mentions[index],
                    reply_to_principal_id: spec.reply.then_some(scope.viewer_principal_id()),
                });
        InboxProjection::build(scope, inputs, &read_state, []).expect("valid inbox projection")
    }

    fn activity(id: &str, actor_kind: ActivityActorKind, second: u32) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 22, 12, 0, second)
            .single()
            .expect("valid timestamp");
        ActivityItem {
            id: ActivityItemId::new(ActivitySourceKind::Nostr, id).expect("valid activity id"),
            source_version: 1,
            class: ActivitySemanticClass::Message,
            actor: ActivityActor {
                kind: actor_kind,
                id: format!("actor-{id}"),
                label: format!("Actor {id}"),
            },
            verb: "posted".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Message,
                id: None,
                label: format!("activity {id}"),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Success,
                summary: None,
            },
            lifecycle: ActivityLifecycle::Succeeded,
            occurred_at: timestamp,
            projected_at: timestamp,
            context: ActivityContext {
                community_id: Some(scope().community_id().to_string()),
                ..ActivityContext::default()
            },
            visibility: ActivityVisibility::Community,
            details: None,
            links: Vec::new(),
        }
    }

    #[gpui::test]
    fn inbox_pulse_filters_unread_mentions_replies_and_actor_kinds(cx: &mut TestAppContext) {
        let view = cx.new(|_| InboxPulseView::new(scope(), 20));
        let inbox = projection(&[
            MessageSpec {
                sequence: 1,
                read_through: 0,
                mention: true,
                reply: false,
            },
            MessageSpec {
                sequence: 2,
                read_through: 2,
                mention: false,
                reply: true,
            },
            MessageSpec {
                sequence: 3,
                read_through: 0,
                mention: false,
                reply: false,
            },
        ]);
        view.update(cx, |view, cx| {
            view.apply_snapshot(
                1,
                inbox,
                vec![
                    activity("human", ActivityActorKind::Human, 1),
                    activity("agent", ActivityActorKind::Agent, 2),
                    activity("system", ActivityActorKind::System, 3),
                ],
                cx,
            )
        })
        .expect("snapshot should apply");

        view.update(cx, |view, cx| {
            view.set_inbox_filter(InboxFilter::Unread, cx)
        });
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 2);
        view.update(cx, |view, cx| {
            view.set_inbox_filter(InboxFilter::Mentions, cx)
        });
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 1);
        view.update(cx, |view, cx| {
            view.set_inbox_filter(InboxFilter::Replies, cx)
        });
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 1);

        view.update(cx, |view, cx| view.set_mode(InboxPulseMode::Pulse, cx));
        view.update(cx, |view, cx| {
            view.set_pulse_filter(PulseFilter::Agents, cx)
        });
        let actor_kind = cx.read(|cx| {
            let [InboxPulseRow::Pulse(item)] = view.read(cx).visible_rows() else {
                panic!("one agent pulse row expected");
            };
            item.actor.kind
        });
        assert_eq!(actor_kind, ActivityActorKind::Agent);
    }

    #[gpui::test]
    fn inbox_pulse_paginates_with_a_bounded_virtual_list(cx: &mut TestAppContext) {
        let view = cx.new(|_| InboxPulseView::new(scope(), 2));
        let inbox = projection(
            &[MessageSpec {
                sequence: 1,
                read_through: 0,
                mention: false,
                reply: false,
            }; 5],
        );
        view.update(cx, |view, cx| view.apply_snapshot(1, inbox, Vec::new(), cx))
            .expect("snapshot should apply");

        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 2);
        assert_eq!(cx.read(|cx| view.read(cx).list_state().item_count()), 2);
        assert!(cx.read(|cx| view.read(cx).has_more()));
        assert!(view.update(cx, InboxPulseView::load_more));
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 4);
        assert!(view.update(cx, InboxPulseView::load_more));
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 5);
        assert!(!cx.read(|cx| view.read(cx).has_more()));
        assert!(!view.update(cx, InboxPulseView::load_more));
    }

    #[gpui::test]
    fn inbox_pulse_preserves_empty_and_stale_projection_states(cx: &mut TestAppContext) {
        let view = cx.new(|_| InboxPulseView::new(scope(), 10));
        assert_eq!(
            cx.read(|cx| view.read(cx).freshness()),
            InboxPulseFreshness::Loading
        );
        view.update(cx, |view, cx| {
            view.apply_snapshot(1, projection(&[]), Vec::new(), cx)
        })
        .expect("empty snapshot should apply");
        assert!(cx.read(|cx| view.read(cx).visible_rows().is_empty()));
        assert_eq!(
            cx.read(|cx| view.read(cx).freshness()),
            InboxPulseFreshness::Fresh { revision: 1 }
        );

        let populated = projection(&[MessageSpec {
            sequence: 1,
            read_through: 0,
            mention: false,
            reply: false,
        }]);
        view.update(cx, |view, cx| {
            view.apply_snapshot(2, populated, Vec::new(), cx)
        })
        .expect("new snapshot should apply");
        view.update(cx, InboxPulseView::mark_stale)
            .expect("loaded snapshot can become stale");
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 1);
        assert_eq!(
            cx.read(|cx| view.read(cx).freshness()),
            InboxPulseFreshness::Stale { revision: 2 }
        );

        let stale_result = view.update(cx, |view, cx| {
            view.apply_snapshot(2, projection(&[]), Vec::new(), cx)
        });
        assert_eq!(stale_result, Err(InboxPulseError::StaleRevision));
        assert_eq!(cx.read(|cx| view.read(cx).visible_rows().len()), 1);
    }
}
