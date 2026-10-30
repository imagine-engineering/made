use std::{collections::HashSet, error::Error, fmt, sync::Arc};

use gpui::{
    AnyElement, App, Context, Entity, EntityId, EventEmitter, ListAlignment, ListState, Render,
    Role, Subscription, Window, list, px,
};
use ui::prelude::*;
use workspace::{
    Workspace,
    collaborative_accessibility::TIMELINE_LABEL,
    collaborative_timeline::{
        CollaborativeTimelineProvider, CollaborativeTimelineRegistration,
        CollaborativeTimelineRegistrationError,
    },
};

use crate::{
    AgentPanel,
    activity_projection::{
        ActivityItem, ActivityItemId, ActivityLink, ActivitySemanticClass, ActivitySourceKind,
    },
    activity_reducer::{ActivityReducer, ActivityReduction, ActivityReductionError},
    collaborative_activity_cards::{ActivityCardIntervention, CollaborativeActivityCard},
    conversation_view::ConversationView,
};

#[derive(Clone, Debug)]
pub enum CollaborativeTimelineEvent {
    InterventionRequested(ActivityCardIntervention),
}

pub struct CollaborativeTimeline {
    reducer: ActivityReducer,
    list_state: ListState,
    expanded_details: HashSet<ActivityItemId>,
}

impl CollaborativeTimeline {
    pub fn new() -> Self {
        let list_state = ListState::new(0, ListAlignment::Top, px(1024.0));
        list_state.set_follow_mode(gpui::FollowMode::Tail);
        Self {
            reducer: ActivityReducer::new(),
            list_state,
            expanded_details: HashSet::new(),
        }
    }

    pub fn items(&self) -> &[ActivityItem] {
        self.reducer.items()
    }

    pub fn list_state(&self) -> &ListState {
        &self.list_state
    }

    pub fn update_item(
        &mut self,
        item: ActivityItem,
        cx: &mut Context<Self>,
    ) -> Result<ActivityReduction, ActivityReductionError> {
        let reduction = self.apply_item(item)?;
        cx.notify();
        Ok(reduction)
    }

    pub fn is_detail_expanded(&self, id: &ActivityItemId) -> bool {
        self.expanded_details.contains(id)
    }

    fn apply_item(
        &mut self,
        item: ActivityItem,
    ) -> Result<ActivityReduction, ActivityReductionError> {
        let reduction = self.reducer.reduce(item)?;
        match reduction {
            ActivityReduction::Inserted { index } => self.list_state.splice(index..index, 1),
            ActivityReduction::Updated { index } => {
                self.list_state.remeasure_items(index..index + 1);
                if self
                    .items()
                    .get(index)
                    .is_some_and(|item| item.class == ActivitySemanticClass::Message)
                {
                    self.list_state.remeasure_items(0..self.items().len());
                }
            }
            ActivityReduction::Duplicate { .. } | ActivityReduction::IgnoredStale { .. } => {}
        }
        Ok(reduction)
    }

    fn toggle_details(&mut self, id: ActivityItemId, cx: &mut Context<Self>) {
        let Some(index) = self.set_detail_expanded(&id, !self.is_detail_expanded(&id)) else {
            return;
        };
        self.list_state.remeasure_items(index..index + 1);
        cx.notify();
    }

    fn set_detail_expanded(&mut self, id: &ActivityItemId, expanded: bool) -> Option<usize> {
        let index = self.items().iter().position(|item| &item.id == id)?;
        if expanded {
            self.expanded_details.insert(id.clone());
        } else {
            self.expanded_details.remove(id);
        }
        Some(index)
    }

    fn quoted_reply(&self, item: &ActivityItem) -> Option<(String, String)> {
        if item.class != ActivitySemanticClass::Message {
            return None;
        }
        let parent_id = item.links.iter().find_map(|link| match link {
            ActivityLink::Entity {
                entity_kind,
                entity_id,
            } if entity_kind == "message" && Some(entity_id) != item.object.id.as_ref() => {
                Some(entity_id)
            }
            _ => None,
        })?;
        let parent = self.items().iter().find(|parent| {
            parent.id.source_kind() == ActivitySourceKind::Nostr
                && parent.object.id.as_ref() == Some(parent_id)
                && parent.class == ActivitySemanticClass::Message
                && parent.context == item.context
                && parent.visibility == item.visibility
        });
        Some(match parent {
            Some(parent) => (parent.actor.label.clone(), parent.object.label.clone()),
            None => (
                "Original message".into(),
                "Not available in this conversation".into(),
            ),
        })
    }

    fn render_item(
        &self,
        index: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(item) = self.items().get(index) else {
            return div().into_any_element();
        };
        let item = item.clone();
        let item_id = item.id.clone();
        let is_expanded = self.is_detail_expanded(&item_id);
        let timeline = cx.entity();
        let toggle_timeline = timeline.clone();
        let quote = self.quoted_reply(&item);
        CollaborativeActivityCard::new(index, item, is_expanded)
            .with_quoted_reply(quote)
            .on_toggle(Arc::new(move |_window, cx| {
                toggle_timeline.update(cx, |this, cx| {
                    this.toggle_details(item_id.clone(), cx);
                });
            }))
            .on_intervention(Arc::new(move |intervention, _window, cx| {
                timeline.update(cx, |_this, cx| {
                    cx.emit(CollaborativeTimelineEvent::InterventionRequested(
                        intervention,
                    ));
                });
            }))
            .into_any_element()
    }
}

impl Default for CollaborativeTimeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Render for CollaborativeTimeline {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.items().is_empty() {
            return div()
                .id("collaborative-activity-empty")
                .role(Role::Status)
                .aria_label("Multiplayer Workspace activity timeline is empty")
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(cx.theme().colors().text_muted)
                .child("No activity yet")
                .into_any_element();
        }

        let view = cx.entity();
        div()
            .id("collaborative-activity-timeline")
            .role(Role::Document)
            .aria_label(TIMELINE_LABEL)
            .size_full()
            .child(
                list(self.list_state.clone(), move |index, window, cx| {
                    view.update(cx, |this, cx| this.render_item(index, window, cx))
                })
                .size_full(),
            )
            .into_any_element()
    }
}

impl EventEmitter<CollaborativeTimelineEvent> for CollaborativeTimeline {}

pub struct CollaborativeTimelineAdapter {
    conversation_view_id: EntityId,
    provider: CollaborativeTimelineProvider,
}

impl CollaborativeTimelineAdapter {
    pub fn from_agent_panel(
        agent_panel: &Entity<AgentPanel>,
        workspace: &Entity<Workspace>,
        cx: &mut App,
    ) -> Result<Self, CollaborativeTimelineAdapterError> {
        let conversation_view = agent_panel
            .read(cx)
            .active_conversation_view()
            .cloned()
            .ok_or(CollaborativeTimelineAdapterError::ConversationUnavailable)?;
        let project = conversation_view.read(cx).project().clone();
        if project.entity_id() != workspace.read(cx).project().entity_id() {
            return Err(CollaborativeTimelineAdapterError::ProjectMismatch);
        }
        let timeline = cx.new(|cx| CollaborativeAcpTimeline::new(conversation_view.clone(), cx));
        Ok(Self {
            conversation_view_id: conversation_view.entity_id(),
            provider: CollaborativeTimelineProvider::new(project, timeline.into()),
        })
    }

    pub fn conversation_view_id(&self) -> EntityId {
        self.conversation_view_id
    }

    pub fn register_in_workspace(
        self,
        workspace: &mut Workspace,
        cx: &mut gpui::Context<Workspace>,
    ) -> Result<CollaborativeTimelineRegistration, CollaborativeTimelineAdapterError> {
        workspace
            .register_collaborative_timeline_provider(self.provider, cx)
            .map_err(CollaborativeTimelineAdapterError::Registration)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeTimelineAdapterError {
    ConversationUnavailable,
    ProjectMismatch,
    Registration(CollaborativeTimelineRegistrationError),
}

impl fmt::Display for CollaborativeTimelineAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConversationUnavailable => {
                formatter.write_str("active agent conversation is unavailable")
            }
            Self::ProjectMismatch => {
                formatter.write_str("active agent thread belongs to a different project")
            }
            Self::Registration(error) => write!(formatter, "timeline registration failed: {error}"),
        }
    }
}

impl Error for CollaborativeTimelineAdapterError {}

struct CollaborativeAcpTimeline {
    conversation_view: Entity<ConversationView>,
    _conversation_view_observation: Subscription,
    observed_thread_view_id: Option<EntityId>,
    thread_view_observation: Option<Subscription>,
}

impl CollaborativeAcpTimeline {
    fn new(conversation_view: Entity<ConversationView>, cx: &mut Context<Self>) -> Self {
        let _conversation_view_observation = cx.observe(&conversation_view, |_, _, cx| cx.notify());
        Self {
            conversation_view,
            _conversation_view_observation,
            observed_thread_view_id: None,
            thread_view_observation: None,
        }
    }
}

impl Render for CollaborativeAcpTimeline {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let thread_view = self
            .conversation_view
            .read(cx)
            .collaborative_active_thread()
            .cloned();
        let thread_view_id = thread_view.as_ref().map(Entity::entity_id);
        if self.observed_thread_view_id != thread_view_id {
            self.thread_view_observation = thread_view
                .as_ref()
                .map(|thread_view| cx.observe(thread_view, |_, _, cx| cx.notify()));
            self.observed_thread_view_id = thread_view_id;
        }
        div()
            .id("collaborative-native-acp-timeline")
            .debug_selector(|| "COLLABORATIVE-NATIVE-ACP-TIMELINE".to_owned())
            .size_full()
            .min_w_0()
            .min_h_0()
            .overflow_hidden()
            .child(self.conversation_view.update(cx, |conversation, cx| {
                conversation.render_collaborative_timeline(window, cx)
            }))
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone as _, Utc};

    use super::*;
    use crate::{
        activity_projection::{
            ActivityActor, ActivityActorKind, ActivityContext, ActivityDetailHandle,
            ActivityLifecycle, ActivityObject, ActivityObjectKind, ActivityOutcome,
            ActivityOutcomeStatus, ActivitySemanticClass, ActivitySourceKind, ActivityVisibility,
        },
        collaborative_activity_cards::{ActivityCardPresentation, ActivityCardSource},
    };

    fn presentation(item: &ActivityItem) -> ActivityCardPresentation {
        ActivityCardPresentation::new(
            item,
            &[ActivityCardSource {
                id: item.id.clone(),
                provenance: None,
            }],
        )
    }

    fn item(source_id: &str, class: ActivitySemanticClass) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 14, 12, 0, 0)
            .single()
            .expect("test timestamp should be valid");
        ActivityItem {
            id: ActivityItemId::new(ActivitySourceKind::Acp, source_id)
                .expect("test source id should be valid"),
            source_version: 1,
            class,
            actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Builder".into(),
            },
            verb: "edited".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::File,
                id: Some("file-1".into()),
                label: "src/main.rs".into(),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: None,
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

    #[test]
    fn collaborative_reply_tracks_the_authoritative_parent_and_updates() {
        let mut timeline = CollaborativeTimeline::new();
        let mut parent = item("parent", ActivitySemanticClass::Message);
        parent.id =
            ActivityItemId::new(ActivitySourceKind::Nostr, "parent").expect("valid parent id");
        parent.object.id = Some("parent".into());
        parent.object.label = "Original claim".into();
        let mut reply = item("reply", ActivitySemanticClass::Message);
        reply.links.push(ActivityLink::Entity {
            entity_kind: "message".into(),
            entity_id: "parent".into(),
        });
        timeline.apply_item(parent.clone()).expect("parent inserts");
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Builder".into(), "Original claim".into()))
        );

        parent.source_version += 1;
        parent.object.label = "Corrected claim".into();
        timeline.apply_item(parent.clone()).expect("parent updates");
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Builder".into(), "Corrected claim".into()))
        );

        parent.source_version += 1;
        parent.object.label = "Deleted message".into();
        timeline.apply_item(parent).expect("deletion projects");
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Builder".into(), "Deleted message".into()))
        );
    }

    #[test]
    fn collaborative_reply_does_not_quote_other_visibility_or_source() {
        let mut timeline = CollaborativeTimeline::new();
        let mut parent = item("parent", ActivitySemanticClass::Message);
        parent.object.id = Some("parent".into());
        let mut reply = item("reply", ActivitySemanticClass::Message);
        reply.links.push(ActivityLink::Entity {
            entity_kind: "message".into(),
            entity_id: "parent".into(),
        });
        timeline.apply_item(parent).expect("ACP item inserts");
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some((
                "Original message".into(),
                "Not available in this conversation".into()
            ))
        );
        let mut channel_parent = item("channel-parent", ActivitySemanticClass::Message);
        channel_parent.id =
            ActivityItemId::new(ActivitySourceKind::Nostr, "channel-parent").expect("valid id");
        channel_parent.object.id = Some("parent".into());
        channel_parent.context.thread_id = Some("private-channel".into());
        timeline
            .apply_item(channel_parent.clone())
            .expect("scoped parent inserts");
        assert_eq!(
            timeline.quoted_reply(&reply).expect("reply placeholder").1,
            "Not available in this conversation"
        );
        channel_parent.source_version += 1;
        channel_parent.context = reply.context.clone();
        channel_parent.visibility = ActivityVisibility::Public;
        timeline
            .apply_item(channel_parent)
            .expect("visibility updates");
        assert_eq!(
            timeline.quoted_reply(&reply).expect("reply placeholder").1,
            "Not available in this conversation"
        );
        reply.object.id = Some("parent".into());
        assert_eq!(
            timeline.quoted_reply(&reply),
            None,
            "self link is not a reply"
        );
    }

    fn message_item(source_id: &str) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 21, 12, 0, 1)
            .single()
            .expect("test timestamp should be valid");
        ActivityItem {
            id: ActivityItemId::new(ActivitySourceKind::Nostr, source_id)
                .expect("test source id should be valid"),
            source_version: 1,
            class: ActivitySemanticClass::Message,
            actor: ActivityActor {
                kind: ActivityActorKind::Human,
                id: "author-1".into(),
                label: "Avery".into(),
            },
            verb: "said".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Message,
                id: Some(source_id.into()),
                label: format!("message {source_id}"),
            },
            outcome: ActivityOutcome {
                status: ActivityOutcomeStatus::Success,
                summary: None,
            },
            lifecycle: ActivityLifecycle::Succeeded,
            occurred_at: timestamp,
            projected_at: timestamp,
            context: ActivityContext {
                community_id: Some("community-1".into()),
                project_id: Some("project-1".into()),
                thread_id: Some("thread-1".into()),
                session_id: None,
            },
            visibility: ActivityVisibility::Community,
            details: None,
            links: Vec::new(),
        }
    }

    fn message_reply(
        item: &ActivityItem,
        parent_id: Option<&str>,
        include_self_link: bool,
    ) -> ActivityItem {
        let mut reply = item.clone();
        reply.verb = if parent_id.is_some() {
            "replied with".into()
        } else {
            "said".into()
        };
        reply.links = Vec::new();
        if include_self_link {
            reply.links.push(ActivityLink::Entity {
                entity_kind: "message".into(),
                entity_id: reply.object.id.clone().unwrap_or_default(),
            });
        }
        if let Some(parent_id) = parent_id {
            reply.links.push(ActivityLink::Entity {
                entity_kind: "message".into(),
                entity_id: parent_id.into(),
            });
        }
        reply
    }

    #[test]
    fn collaborative_reply_resolves_scoped_message_parent_and_tracks_lifecycle() {
        let mut timeline = CollaborativeTimeline::new();
        let parent = message_item("event-1");
        timeline.apply_item(parent).expect("parent inserts");
        let reply = message_reply(&message_item("event-2"), Some("event-1"), true);

        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Avery".into(), "message event-1".into()))
        );

        let mut edited = message_item("event-1");
        edited.source_version = 2;
        edited.object.label = "Corrected claim".into();
        timeline.apply_item(edited).expect("edit updates");
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Avery".into(), "Corrected claim".into()))
        );

        let mut deleted = message_item("event-1");
        deleted.source_version = 3;
        deleted.verb = "deleted".into();
        deleted.object.label = "Deleted message".into();
        timeline.apply_item(deleted).expect("deletion projects");
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Avery".into(), "Deleted message".into()))
        );
    }

    #[test]
    fn collaborative_reply_resolves_optimistic_message_parent_without_self_link() {
        let mut timeline = CollaborativeTimeline::new();
        timeline
            .apply_item(message_item("event-1"))
            .expect("parent inserts");
        let mut optimistic = message_item("event-2");
        optimistic.id = ActivityItemId::new(ActivitySourceKind::NativeAction, "operation-1")
            .expect("test operation id should be valid");
        optimistic.object.id = None;
        let reply = message_reply(&optimistic, Some("event-1"), false);
        assert_eq!(
            timeline.quoted_reply(&reply),
            Some(("Avery".into(), "message event-1".into()))
        );
    }

    #[test]
    fn collaborative_reply_falls_back_and_rejects_self_link_in_message_conversation() {
        let timeline = CollaborativeTimeline::new();
        let unloaded_reply = message_reply(&message_item("event-2"), Some("event-9"), true);
        assert_eq!(
            timeline.quoted_reply(&unloaded_reply),
            Some((
                "Original message".into(),
                "Not available in this conversation".into()
            ))
        );

        let self_reply = message_reply(&message_item("event-2"), Some("event-2"), true);
        assert_eq!(
            timeline.quoted_reply(&self_reply),
            None,
            "self link is the only message link and must not be a reply"
        );
    }

    #[test]
    fn collaborative_timeline_render_preserves_reducer_order() {
        let mut timeline = CollaborativeTimeline::new();
        timeline
            .apply_item(item("message-1", ActivitySemanticClass::Message))
            .expect("first item should insert");
        timeline
            .apply_item(item("edit-1", ActivitySemanticClass::FileEdit))
            .expect("second item should insert");

        assert_eq!(timeline.items()[0].id.source_id(), "message-1");
        assert_eq!(timeline.items()[1].id.source_id(), "edit-1");
    }

    #[test]
    fn collaborative_timeline_render_uses_virtualized_list_count() {
        let mut timeline = CollaborativeTimeline::new();
        for index in 0..1_000 {
            timeline
                .apply_item(item(
                    &format!("activity-{index}"),
                    ActivitySemanticClass::Message,
                ))
                .expect("unique activity should insert");
        }

        assert_eq!(timeline.items().len(), 1_000);
        assert_eq!(timeline.list_state().item_count(), 1_000);
    }

    #[test]
    fn collaborative_timeline_render_discloses_typed_details() {
        let item = item("edit-1", ActivitySemanticClass::FileEdit);
        let item_id = item.id.clone();
        let mut timeline = CollaborativeTimeline::new();
        timeline
            .apply_item(item.clone())
            .expect("detail-bearing item should insert");

        assert!(!timeline.is_detail_expanded(&item_id));
        assert_eq!(timeline.set_detail_expanded(&item_id, true), Some(0));
        assert!(timeline.is_detail_expanded(&item_id));
        assert_eq!(
            presentation(&item).detail.as_deref(),
            Some("ACP session session-1, entry edit-1")
        );
        assert_eq!(timeline.set_detail_expanded(&item_id, false), Some(0));
        assert!(!timeline.is_detail_expanded(&item_id));
    }

    #[test]
    fn collaborative_timeline_accessibility_labels_running_and_failed_activity() {
        let running = item("running-1", ActivitySemanticClass::ShellCommand);
        assert_eq!(
            presentation(&running).accessibility_label(),
            "Builder edited src/main.rs. Command. Running"
        );

        let mut failed = item("failed-1", ActivitySemanticClass::Error);
        failed.lifecycle = ActivityLifecycle::Failed;
        failed.outcome.summary = Some("Command exited with status 1".into());
        assert_eq!(
            presentation(&failed).accessibility_label(),
            "Builder edited src/main.rs. Error. Failed. Command exited with status 1"
        );
    }

    #[test]
    fn collaborative_timeline_render_labels_unknown_events_truthfully() {
        let mut unknown = item("future-1", ActivitySemanticClass::Generic);
        unknown.details = None;

        assert_eq!(
            presentation(&unknown).summary,
            "Builder reported an unsupported activity event"
        );
        assert_eq!(
            presentation(&unknown).detail.as_deref(),
            Some("Unsupported Acp event future-1")
        );
    }
}
