use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt,
};

use agent_ui::{
    activity_projection::{
        ActivityActor, ActivityActorKind, ActivityContext, ActivityDetailHandle, ActivityItem,
        ActivityItemId, ActivityLifecycle, ActivityLink, ActivityObject, ActivityObjectKind,
        ActivityOutcome, ActivityOutcomeStatus, ActivityProjectionContractError,
        ActivitySemanticClass, ActivitySourceKind, ActivityVisibility,
    },
    activity_reducer::ActivityReductionError,
    collaborative_timeline::CollaborativeTimeline,
};
use chrono::{DateTime, Utc};
use gpui::{AppContext as _, Context, Entity, IntoElement, Render, Window};
use ui::prelude::*;

use crate::message_reconciliation::{
    MessageDeliveryState, MessageReconciler, MessageReconciliationAction,
    MessageReconciliationError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageTimelineAuthorKind {
    Human,
    Agent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageTimelineAuthor {
    pub kind: MessageTimelineAuthorKind,
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageTimelineReaction {
    pub value: String,
    pub count: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MessageTimelineContext {
    pub community_id: Option<String>,
    pub project_id: Option<String>,
    pub thread_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageTimelineEntry {
    pub event_id: String,
    pub operation_id: Option<String>,
    pub source_version: u64,
    pub author: MessageTimelineAuthor,
    pub content: String,
    pub reply_to: Option<String>,
    pub edited: bool,
    pub deleted: bool,
    pub reactions: Vec<MessageTimelineReaction>,
    pub occurred_at: DateTime<Utc>,
    pub projected_at: DateTime<Utc>,
    pub context: MessageTimelineContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptimisticMessage {
    pub operation_id: String,
    pub author: MessageTimelineAuthor,
    pub content: String,
    pub reply_to: Option<String>,
    pub occurred_at: DateTime<Utc>,
    pub context: MessageTimelineContext,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageTimelinePage {
    pub request_cursor: Option<String>,
    pub next_cursor: Option<String>,
    /// Entries use the repository's newest-to-oldest page order.
    pub entries: Vec<MessageTimelineEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessageTimelineError {
    UnexpectedCursor {
        expected: Option<String>,
        received: Option<String>,
    },
    HistoryComplete,
    NonAdvancingCursor,
    DuplicateHistoryEvent(String),
    EmptyEventId,
    EmptyOperationId,
    EmptyAuthor,
    EmptyMessage,
    InvalidSourceVersion,
    InvalidReaction,
    ConflictingEventVersion {
        event_id: String,
        source_version: u64,
    },
    StaleEventVersion {
        event_id: String,
        current_version: u64,
        incoming_version: u64,
    },
    Reconciliation(MessageReconciliationError),
    Projection(ActivityProjectionContractError),
    Reduction(ActivityReductionError),
}

impl fmt::Display for MessageTimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedCursor { expected, received } => write!(
                formatter,
                "message page cursor mismatch: expected {expected:?}, received {received:?}"
            ),
            Self::HistoryComplete => formatter.write_str("message history is already complete"),
            Self::NonAdvancingCursor => {
                formatter.write_str("message page continuation must advance")
            }
            Self::DuplicateHistoryEvent(event_id) => {
                write!(formatter, "message history repeats event {event_id}")
            }
            Self::EmptyEventId => formatter.write_str("message event id must not be empty"),
            Self::EmptyOperationId => formatter.write_str("message operation id must not be empty"),
            Self::EmptyAuthor => formatter.write_str("message author must not be empty"),
            Self::EmptyMessage => formatter.write_str("message content must not be empty"),
            Self::InvalidSourceVersion => {
                formatter.write_str("message source version must be positive")
            }
            Self::InvalidReaction => {
                formatter.write_str("message reaction must have a value and positive count")
            }
            Self::ConflictingEventVersion {
                event_id,
                source_version,
            } => write!(
                formatter,
                "message event {event_id} has conflicting payloads at version {source_version}"
            ),
            Self::StaleEventVersion {
                event_id,
                current_version,
                incoming_version,
            } => write!(
                formatter,
                "message event {event_id} cannot regress from version {current_version} to {incoming_version}"
            ),
            Self::Reconciliation(error) => error.fmt(formatter),
            Self::Projection(error) => error.fmt(formatter),
            Self::Reduction(error) => error.fmt(formatter),
        }
    }
}

impl Error for MessageTimelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Reconciliation(error) => Some(error),
            Self::Projection(error) => Some(error),
            Self::Reduction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MessageReconciliationError> for MessageTimelineError {
    fn from(error: MessageReconciliationError) -> Self {
        Self::Reconciliation(error)
    }
}

impl From<ActivityProjectionContractError> for MessageTimelineError {
    fn from(error: ActivityProjectionContractError) -> Self {
        Self::Projection(error)
    }
}

impl From<ActivityReductionError> for MessageTimelineError {
    fn from(error: ActivityReductionError) -> Self {
        Self::Reduction(error)
    }
}

#[derive(Clone, Debug)]
struct OptimisticState {
    message: OptimisticMessage,
}

#[derive(Clone, Debug, Default)]
struct MessageTimelineState {
    pages: Vec<MessageTimelinePage>,
    live_entries: BTreeMap<String, MessageTimelineEntry>,
    optimistic: BTreeMap<String, OptimisticState>,
    suppressed_event_ids: HashSet<String>,
    reconciler: MessageReconciler<String, String, String>,
}

impl MessageTimelineState {
    fn apply_history_page(
        &mut self,
        mut page: MessageTimelinePage,
    ) -> Result<(), MessageTimelineError> {
        for entry in &mut page.entries {
            self.preserve_newer_entry(entry)?;
        }
        let expected_cursor = self.pages.last().and_then(|page| page.next_cursor.clone());
        if !self.pages.is_empty() && expected_cursor.is_none() {
            return Err(MessageTimelineError::HistoryComplete);
        }
        if page.request_cursor != expected_cursor {
            return Err(MessageTimelineError::UnexpectedCursor {
                expected: expected_cursor,
                received: page.request_cursor,
            });
        }
        if page.next_cursor.is_some() && page.next_cursor == page.request_cursor {
            return Err(MessageTimelineError::NonAdvancingCursor);
        }

        let mut history_event_ids = self
            .pages
            .iter()
            .flat_map(|page| page.entries.iter().map(|entry| entry.event_id.as_str()))
            .collect::<HashSet<_>>();
        for entry in &page.entries {
            validate_entry(entry)?;
            if !history_event_ids.insert(&entry.event_id) {
                return Err(MessageTimelineError::DuplicateHistoryEvent(
                    entry.event_id.clone(),
                ));
            }
        }

        for entry in &page.entries {
            self.reconcile_authoritative(entry, false)?;
            self.live_entries.remove(&entry.event_id);
        }
        self.pages.push(page);
        Ok(())
    }

    fn replace_history_head(
        &mut self,
        mut page: MessageTimelinePage,
    ) -> Result<(), MessageTimelineError> {
        if page.request_cursor.is_some() {
            return Err(MessageTimelineError::UnexpectedCursor {
                expected: None,
                received: page.request_cursor,
            });
        }
        for incoming in &mut page.entries {
            self.preserve_newer_entry(incoming)?;
        }
        self.pages.clear();
        self.apply_history_page(page)
    }

    fn upsert_live(&mut self, entry: MessageTimelineEntry) -> Result<(), MessageTimelineError> {
        validate_entry(&entry)?;
        self.validate_event_update(&entry)?;
        self.reconcile_authoritative(&entry, true)
    }

    fn preserve_newer_entry(
        &self,
        incoming: &mut MessageTimelineEntry,
    ) -> Result<(), MessageTimelineError> {
        validate_entry(incoming)?;
        let current = self.live_entries.get(&incoming.event_id).or_else(|| {
            self.pages
                .iter()
                .flat_map(|page| &page.entries)
                .find(|entry| entry.event_id == incoming.event_id)
        });
        let Some(current) = current else {
            return Ok(());
        };
        if let (Some(current_operation), Some(incoming_operation)) =
            (&current.operation_id, &incoming.operation_id)
            && current_operation != incoming_operation
        {
            return Err(MessageReconciliationError::EventOwnedByAnotherOperation.into());
        }
        let operation_id = incoming
            .operation_id
            .clone()
            .or_else(|| current.operation_id.clone());
        if incoming.source_version < current.source_version {
            *incoming = current.clone();
        }
        incoming.operation_id = operation_id;
        let mut comparable = incoming.clone();
        comparable.operation_id = current.operation_id.clone();
        self.validate_event_update(&comparable)
    }

    fn upsert_replayed(
        &mut self,
        mut entry: MessageTimelineEntry,
    ) -> Result<(), MessageTimelineError> {
        self.preserve_newer_entry(&mut entry)?;
        self.reconcile_authoritative(&entry, true)
    }

    fn begin_optimistic(&mut self, message: OptimisticMessage) -> Result<(), MessageTimelineError> {
        validate_optimistic(&message)?;
        self.reconciler.begin(message.operation_id.clone())?;
        self.optimistic
            .insert(message.operation_id.clone(), OptimisticState { message });
        Ok(())
    }

    fn retry_optimistic(&mut self, operation_id: &str) -> Result<(), MessageTimelineError> {
        self.reconciler.retry(&operation_id.to_owned())?;
        Ok(())
    }

    fn reject_optimistic(
        &mut self,
        operation_id: &str,
        reason: String,
    ) -> Result<(), MessageTimelineError> {
        let operation_id = operation_id.to_owned();
        self.reconciler.reject(&operation_id, reason)?;
        if !self.optimistic.contains_key(&operation_id) {
            return Err(MessageReconciliationError::UnknownOperation.into());
        }
        Ok(())
    }

    fn accept_optimistic(
        &mut self,
        operation_id: &str,
        mut entry: MessageTimelineEntry,
    ) -> Result<(), MessageTimelineError> {
        validate_entry(&entry)?;
        let operation_id = operation_id.to_owned();
        if entry
            .operation_id
            .as_ref()
            .is_some_and(|entry_operation_id| entry_operation_id != &operation_id)
        {
            return Err(MessageReconciliationError::EventOwnedByAnotherOperation.into());
        }
        entry.operation_id = Some(operation_id.clone());
        self.preserve_newer_entry(&mut entry)?;
        let action = self
            .reconciler
            .accept(&operation_id, entry.event_id.clone())?;
        self.apply_reconciliation_action(&action);
        self.suppressed_event_ids.remove(&entry.event_id);
        self.live_entries.insert(entry.event_id.clone(), entry);
        Ok(())
    }

    fn reconcile_authoritative(
        &mut self,
        entry: &MessageTimelineEntry,
        insert_live: bool,
    ) -> Result<(), MessageTimelineError> {
        let action = self
            .reconciler
            .observe_authoritative(entry.event_id.clone(), entry.operation_id.as_ref())?;
        self.apply_reconciliation_action(&action);
        self.suppressed_event_ids.remove(&entry.event_id);
        if insert_live {
            self.live_entries
                .insert(entry.event_id.clone(), entry.clone());
        }
        Ok(())
    }

    fn apply_reconciliation_action(
        &mut self,
        action: &MessageReconciliationAction<String, String>,
    ) {
        match action {
            MessageReconciliationAction::ReplaceOptimistic { operation_id, .. }
            | MessageReconciliationAction::SuppressDuplicateEcho { operation_id, .. } => {
                self.optimistic.remove(operation_id);
            }
            MessageReconciliationAction::ReplaceAuthoritative {
                operation_id,
                previous_event_id,
                ..
            } => {
                self.optimistic.remove(operation_id);
                self.live_entries.remove(previous_event_id);
                self.suppressed_event_ids.insert(previous_event_id.clone());
            }
            MessageReconciliationAction::InsertOptimistic { .. }
            | MessageReconciliationAction::RetryOptimistic { .. }
            | MessageReconciliationAction::MarkRejected { .. }
            | MessageReconciliationAction::InsertAuthoritative { .. }
            | MessageReconciliationAction::Unchanged { .. } => {}
        }
    }

    fn validate_event_update(
        &self,
        incoming: &MessageTimelineEntry,
    ) -> Result<(), MessageTimelineError> {
        let current = self.live_entries.get(&incoming.event_id).or_else(|| {
            self.pages
                .iter()
                .flat_map(|page| page.entries.iter())
                .find(|entry| entry.event_id == incoming.event_id)
        });
        let Some(current) = current else {
            return Ok(());
        };
        if incoming.source_version < current.source_version {
            return Err(MessageTimelineError::StaleEventVersion {
                event_id: incoming.event_id.clone(),
                current_version: current.source_version,
                incoming_version: incoming.source_version,
            });
        }
        if incoming.source_version == current.source_version && incoming != current {
            return Err(MessageTimelineError::ConflictingEventVersion {
                event_id: incoming.event_id.clone(),
                source_version: incoming.source_version,
            });
        }
        Ok(())
    }

    fn activity_items(&self) -> Result<Vec<ActivityItem>, MessageTimelineError> {
        let mut authoritative = BTreeMap::<String, MessageTimelineEntry>::new();
        for page in self.pages.iter().rev() {
            for entry in page.entries.iter().rev() {
                if !self.suppressed_event_ids.contains(&entry.event_id) {
                    authoritative.insert(entry.event_id.clone(), entry.clone());
                }
            }
        }
        for entry in self.live_entries.values() {
            if !self.suppressed_event_ids.contains(&entry.event_id) {
                authoritative.insert(entry.event_id.clone(), entry.clone());
            }
        }

        let mut authoritative = authoritative.into_values().collect::<Vec<_>>();
        authoritative.sort_by(|left, right| {
            left.occurred_at
                .cmp(&right.occurred_at)
                .then_with(|| left.event_id.cmp(&right.event_id))
        });
        let mut items = authoritative
            .iter()
            .map(project_authoritative)
            .collect::<Result<Vec<_>, _>>()?;

        let mut optimistic = self.optimistic.values().collect::<Vec<_>>();
        optimistic.sort_by(|left, right| {
            left.message
                .occurred_at
                .cmp(&right.message.occurred_at)
                .then_with(|| left.message.operation_id.cmp(&right.message.operation_id))
        });
        items.extend(
            optimistic
                .into_iter()
                .map(|state| project_optimistic(state, &self.reconciler))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(items)
    }
}

pub struct MessageTimeline {
    state: MessageTimelineState,
    timeline: Entity<CollaborativeTimeline>,
}

impl MessageTimeline {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            state: MessageTimelineState::default(),
            timeline: cx.new(|_| CollaborativeTimeline::new()),
        }
    }

    pub fn timeline(&self) -> Entity<CollaborativeTimeline> {
        self.timeline.clone()
    }

    pub fn apply_history_page(
        &mut self,
        page: MessageTimelinePage,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.apply_history_page(page))
    }

    pub fn replace_history_head(
        &mut self,
        page: MessageTimelinePage,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.replace_history_head(page))
    }

    pub fn upsert_live(
        &mut self,
        entry: MessageTimelineEntry,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.upsert_live(entry))
    }

    pub fn upsert_replayed(
        &mut self,
        entry: MessageTimelineEntry,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.upsert_replayed(entry))
    }

    pub fn begin_optimistic(
        &mut self,
        message: OptimisticMessage,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.begin_optimistic(message))
    }

    pub fn retry_optimistic(
        &mut self,
        operation_id: &str,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.retry_optimistic(operation_id))
    }

    pub fn reject_optimistic(
        &mut self,
        operation_id: &str,
        reason: impl Into<String>,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        let reason = reason.into();
        self.update_state(cx, |state| state.reject_optimistic(operation_id, reason))
    }

    pub fn accept_optimistic(
        &mut self,
        operation_id: &str,
        entry: MessageTimelineEntry,
        cx: &mut Context<Self>,
    ) -> Result<(), MessageTimelineError> {
        self.update_state(cx, |state| state.accept_optimistic(operation_id, entry))
    }

    fn update_state(
        &mut self,
        cx: &mut Context<Self>,
        update: impl FnOnce(&mut MessageTimelineState) -> Result<(), MessageTimelineError>,
    ) -> Result<(), MessageTimelineError> {
        let mut next_state = self.state.clone();
        update(&mut next_state)?;
        let items = next_state.activity_items()?;
        let next_timeline = cx.new(|_| CollaborativeTimeline::new());
        for item in items {
            next_timeline.update(cx, |timeline, cx| timeline.update_item(item, cx))?;
        }
        self.state = next_state;
        self.timeline = next_timeline;
        cx.notify();
        Ok(())
    }
}

impl Render for MessageTimeline {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.timeline.clone())
    }
}

fn validate_entry(entry: &MessageTimelineEntry) -> Result<(), MessageTimelineError> {
    if entry.event_id.trim().is_empty() {
        return Err(MessageTimelineError::EmptyEventId);
    }
    if entry
        .operation_id
        .as_ref()
        .is_some_and(|operation_id| operation_id.trim().is_empty())
    {
        return Err(MessageTimelineError::EmptyOperationId);
    }
    if entry.source_version == 0 {
        return Err(MessageTimelineError::InvalidSourceVersion);
    }
    validate_author(&entry.author)?;
    if !entry.deleted && entry.content.trim().is_empty() {
        return Err(MessageTimelineError::EmptyMessage);
    }
    if entry
        .reactions
        .iter()
        .any(|reaction| reaction.value.trim().is_empty() || reaction.count == 0)
    {
        return Err(MessageTimelineError::InvalidReaction);
    }
    Ok(())
}

fn validate_optimistic(message: &OptimisticMessage) -> Result<(), MessageTimelineError> {
    if message.operation_id.trim().is_empty() {
        return Err(MessageTimelineError::EmptyOperationId);
    }
    validate_author(&message.author)?;
    if message.content.trim().is_empty() {
        return Err(MessageTimelineError::EmptyMessage);
    }
    Ok(())
}

fn validate_author(author: &MessageTimelineAuthor) -> Result<(), MessageTimelineError> {
    if author.id.trim().is_empty() || author.label.trim().is_empty() {
        return Err(MessageTimelineError::EmptyAuthor);
    }
    Ok(())
}

fn project_authoritative(
    entry: &MessageTimelineEntry,
) -> Result<ActivityItem, MessageTimelineError> {
    let id = ActivityItemId::new(ActivitySourceKind::Nostr, entry.event_id.clone())?;
    let mut outcome_details = Vec::new();
    if entry.edited {
        outcome_details.push("Edited".into());
    }
    let mut reactions = entry.reactions.clone();
    reactions.sort_by(|left, right| left.value.cmp(&right.value));
    outcome_details.extend(
        reactions
            .into_iter()
            .map(|reaction| format!("{} {}", reaction.value, reaction.count)),
    );

    Ok(ActivityItem {
        id,
        source_version: entry.source_version,
        class: ActivitySemanticClass::Message,
        actor: project_actor(&entry.author),
        verb: if entry.deleted {
            "deleted".into()
        } else if entry.reply_to.is_some() {
            "replied with".into()
        } else if entry.edited {
            "edited".into()
        } else {
            "said".into()
        },
        object: ActivityObject {
            kind: ActivityObjectKind::Message,
            id: Some(entry.event_id.clone()),
            label: if entry.deleted {
                "Deleted message".into()
            } else {
                entry.content.clone()
            },
        },
        outcome: ActivityOutcome {
            status: ActivityOutcomeStatus::Success,
            summary: (!outcome_details.is_empty()).then(|| outcome_details.join(" · ")),
        },
        lifecycle: ActivityLifecycle::Succeeded,
        occurred_at: entry.occurred_at,
        projected_at: entry.projected_at,
        context: project_context(&entry.context),
        visibility: ActivityVisibility::Community,
        details: Some(ActivityDetailHandle::ProtocolEvent {
            event_id: entry.event_id.clone(),
        }),
        links: message_links(&entry.event_id, entry.reply_to.as_deref()),
    })
}

fn project_optimistic(
    state: &OptimisticState,
    reconciler: &MessageReconciler<String, String, String>,
) -> Result<ActivityItem, MessageTimelineError> {
    let operation_id = &state.message.operation_id;
    let id = ActivityItemId::new(
        ActivitySourceKind::NativeAction,
        format!("message-operation:{operation_id}"),
    )?;
    let delivery_state = reconciler
        .state(operation_id)
        .ok_or(MessageReconciliationError::UnknownOperation)?;
    let (source_version, lifecycle, outcome) = match delivery_state {
        MessageDeliveryState::Pending { attempt } => (
            u64::from(*attempt),
            ActivityLifecycle::Pending,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Pending,
                summary: (*attempt > 1).then(|| format!("Sending, attempt {attempt}")),
            },
        ),
        MessageDeliveryState::Rejected { reason } => (
            1,
            ActivityLifecycle::Failed,
            ActivityOutcome {
                status: ActivityOutcomeStatus::Failure,
                summary: Some(reason.clone()),
            },
        ),
        MessageDeliveryState::Accepted { .. } | MessageDeliveryState::Reconciled { .. } => {
            return Err(MessageReconciliationError::UnknownOperation.into());
        }
    };

    Ok(ActivityItem {
        id,
        source_version,
        class: ActivitySemanticClass::Message,
        actor: project_actor(&state.message.author),
        verb: if state.message.reply_to.is_some() {
            "is replying with".into()
        } else {
            "is sending".into()
        },
        object: ActivityObject {
            kind: ActivityObjectKind::Message,
            id: None,
            label: state.message.content.clone(),
        },
        outcome,
        lifecycle,
        occurred_at: state.message.occurred_at,
        projected_at: state.message.occurred_at,
        context: project_context(&state.message.context),
        visibility: ActivityVisibility::Community,
        details: Some(ActivityDetailHandle::NativeAction {
            action_id: operation_id.clone(),
        }),
        links: state
            .message
            .reply_to
            .as_deref()
            .map(|reply_to| ActivityLink::Entity {
                entity_kind: "message".into(),
                entity_id: reply_to.into(),
            })
            .into_iter()
            .collect(),
    })
}

fn project_actor(author: &MessageTimelineAuthor) -> ActivityActor {
    ActivityActor {
        kind: match author.kind {
            MessageTimelineAuthorKind::Human => ActivityActorKind::Human,
            MessageTimelineAuthorKind::Agent => ActivityActorKind::Agent,
        },
        id: author.id.clone(),
        label: author.label.clone(),
    }
}

fn project_context(context: &MessageTimelineContext) -> ActivityContext {
    ActivityContext {
        community_id: context.community_id.clone(),
        project_id: context.project_id.clone(),
        thread_id: context.thread_id.clone(),
        session_id: None,
    }
}

fn message_links(event_id: &str, reply_to: Option<&str>) -> Vec<ActivityLink> {
    let mut links = vec![ActivityLink::Entity {
        entity_kind: "message".into(),
        entity_id: event_id.into(),
    }];
    if let Some(reply_to) = reply_to {
        links.push(ActivityLink::Entity {
            entity_kind: "message".into(),
            entity_id: reply_to.into(),
        });
    }
    links
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone as _;
    use gpui::TestAppContext;

    use super::*;

    fn timestamp(second: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 8, 21, 12, 0, second)
            .single()
            .expect("test timestamp should be valid")
    }

    fn author(kind: MessageTimelineAuthorKind) -> MessageTimelineAuthor {
        MessageTimelineAuthor {
            kind,
            id: "author-1".into(),
            label: match kind {
                MessageTimelineAuthorKind::Human => "Avery",
                MessageTimelineAuthorKind::Agent => "Builder",
            }
            .into(),
        }
    }

    fn entry(event_id: &str, second: u32) -> MessageTimelineEntry {
        MessageTimelineEntry {
            event_id: event_id.into(),
            operation_id: None,
            source_version: 1,
            author: author(MessageTimelineAuthorKind::Human),
            content: format!("message {event_id}"),
            reply_to: None,
            edited: false,
            deleted: false,
            reactions: Vec::new(),
            occurred_at: timestamp(second),
            projected_at: timestamp(second),
            context: MessageTimelineContext {
                community_id: Some("community-1".into()),
                project_id: Some("project-1".into()),
                thread_id: Some("thread-1".into()),
            },
        }
    }

    #[test]
    fn replay_preserves_newer_content_and_create_ownership() {
        let mut state = MessageTimelineState::default();
        let mut created = entry("event-1", 1);
        created.operation_id = Some("create-1".into());
        state.upsert_live(created.clone()).expect("create");
        let mut edited = created.clone();
        edited.operation_id = None;
        edited.source_version = 3;
        edited.content = "edited body".into();
        edited.edited = true;
        state.upsert_replayed(edited.clone()).expect("edit receipt");
        state
            .upsert_replayed(created.clone())
            .expect("older replay");
        let current = state.live_entries.get("event-1").expect("one event");
        assert_eq!(current.content, "edited body");
        assert_eq!(current.source_version, 3);
        assert_eq!(current.operation_id.as_deref(), Some("create-1"));
        assert_eq!(state.activity_items().expect("items").len(), 1);
        assert!(matches!(
            state.upsert_live(created),
            Err(MessageTimelineError::StaleEventVersion { .. })
        ));
        let mut conflicting = edited.clone();
        conflicting.content = "different body at the same version".into();
        assert!(matches!(
            state.upsert_replayed(conflicting),
            Err(MessageTimelineError::ConflictingEventVersion { .. })
        ));
        edited.operation_id = Some("another-create".into());
        assert!(state.upsert_replayed(edited).is_err());
    }

    #[test]
    fn late_create_receipt_preserves_newer_head_and_resolves_optimistic_item() {
        let mut state = MessageTimelineState::default();
        state
            .begin_optimistic(OptimisticMessage {
                operation_id: "create-1".into(),
                author: author(MessageTimelineAuthorKind::Human),
                content: "original body".into(),
                reply_to: None,
                occurred_at: timestamp(1),
                context: MessageTimelineContext::default(),
            })
            .expect("optimistic create");
        let mut newest = entry("event-1", 1);
        newest.source_version = 5;
        newest.content = "edited before acknowledgement".into();
        newest.edited = true;
        state
            .apply_history_page(MessageTimelinePage {
                request_cursor: None,
                next_cursor: None,
                entries: vec![newest.clone()],
            })
            .expect("newer head");
        state
            .accept_optimistic("create-1", entry("event-1", 1))
            .expect("late create receipt");
        assert!(state.optimistic.is_empty());
        assert_eq!(state.activity_items().expect("items").len(), 1);
        let current = state.live_entries.get("event-1").expect("accepted message");
        assert_eq!(current.source_version, 5);
        assert_eq!(current.content, newest.content);
        assert_eq!(current.operation_id.as_deref(), Some("create-1"));
        state
            .replace_history_head(MessageTimelinePage {
                request_cursor: None,
                next_cursor: None,
                entries: vec![entry("event-1", 1)],
            })
            .expect("older reconnect snapshot");
        assert_eq!(state.pages[0].entries[0].source_version, 5);
        assert_eq!(
            state.pages[0].entries[0].operation_id.as_deref(),
            Some("create-1")
        );
    }

    #[gpui::test]
    fn reconnect_replaces_completed_history_without_losing_newer_live_updates(
        cx: &mut TestAppContext,
    ) {
        let timeline = cx.new(MessageTimeline::new);
        timeline.update(cx, |timeline, cx| {
            let head = MessageTimelinePage {
                request_cursor: None,
                next_cursor: None,
                entries: vec![entry("first", 1)],
            };
            timeline
                .apply_history_page(head.clone(), cx)
                .expect("initial head");
            let mut newer = entry("first", 1);
            newer.source_version = 2;
            newer.content = "edited while reconnecting".into();
            timeline
                .upsert_live(newer.clone(), cx)
                .expect("newer live update");
            timeline
                .replace_history_head(head, cx)
                .expect("reconnected head replaces complete history");
            assert_eq!(timeline.state.pages.len(), 1);
            assert_eq!(timeline.state.pages[0].entries[0], newer);
            let invalid = MessageTimelinePage {
                request_cursor: Some("stale".into()),
                next_cursor: None,
                entries: Vec::new(),
            };
            assert!(timeline.replace_history_head(invalid, cx).is_err());
            assert_eq!(timeline.state.pages[0].entries[0], newer);
        });
    }

    fn item_snapshots(
        timeline: &Entity<MessageTimeline>,
        cx: &TestAppContext,
    ) -> Vec<ActivityItem> {
        timeline.read_with(cx, |timeline, cx| {
            timeline.timeline.read(cx).items().to_vec()
        })
    }

    #[gpui::test]
    fn message_timeline_orders_pages_and_live_overlay(cx: &mut TestAppContext) {
        let timeline = cx.new(MessageTimeline::new);
        timeline
            .update(cx, |timeline, cx| {
                timeline.apply_history_page(
                    MessageTimelinePage {
                        request_cursor: None,
                        next_cursor: Some("older-1".into()),
                        entries: vec![entry("event-3", 3), entry("event-2", 2)],
                    },
                    cx,
                )?;
                timeline.apply_history_page(
                    MessageTimelinePage {
                        request_cursor: Some("older-1".into()),
                        next_cursor: None,
                        entries: vec![entry("event-1", 1)],
                    },
                    cx,
                )?;
                timeline.upsert_live(entry("event-4", 4), cx)
            })
            .expect("valid pages and live message should render");

        let items = item_snapshots(&timeline, cx);
        assert_eq!(
            items
                .iter()
                .map(|item| item.id.source_id())
                .collect::<Vec<_>>(),
            ["event-1", "event-2", "event-3", "event-4"]
        );
        let item_count = timeline.read_with(cx, |timeline, cx| {
            timeline.timeline.read(cx).list_state().item_count()
        });
        assert_eq!(item_count, 4);
    }

    #[gpui::test]
    fn message_timeline_projects_replies_edits_reactions_and_deletion(cx: &mut TestAppContext) {
        let timeline = cx.new(MessageTimeline::new);
        let mut reply = entry("event-2", 2);
        reply.author = author(MessageTimelineAuthorKind::Agent);
        reply.reply_to = Some("event-1".into());
        reply.edited = true;
        reply.reactions = vec![
            MessageTimelineReaction {
                value: "👍".into(),
                count: 2,
            },
            MessageTimelineReaction {
                value: "🚀".into(),
                count: 1,
            },
        ];
        timeline
            .update(cx, |timeline, cx| timeline.upsert_live(reply, cx))
            .expect("reply should render");

        let items = item_snapshots(&timeline, cx);
        assert_eq!(items[0].actor.kind, ActivityActorKind::Agent);
        assert_eq!(items[0].verb, "replied with");
        assert_eq!(
            items[0].outcome.summary.as_deref(),
            Some("Edited · 👍 2 · 🚀 1")
        );
        assert_eq!(items[0].links.len(), 2);

        let mut deleted = entry("event-2", 2);
        deleted.source_version = 2;
        deleted.deleted = true;
        deleted.content = "content that must not render".into();
        timeline
            .update(cx, |timeline, cx| timeline.upsert_live(deleted, cx))
            .expect("deletion should replace the message");
        let items = item_snapshots(&timeline, cx);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].object.label, "Deleted message");
        assert_eq!(items[0].verb, "deleted");
    }

    #[gpui::test]
    fn message_timeline_keeps_failed_optimistic_item_until_authoritative_reconcile(
        cx: &mut TestAppContext,
    ) {
        let timeline = cx.new(MessageTimeline::new);
        timeline
            .update(cx, |timeline, cx| {
                timeline.begin_optimistic(
                    OptimisticMessage {
                        operation_id: "operation-1".into(),
                        author: author(MessageTimelineAuthorKind::Human),
                        content: "hello".into(),
                        reply_to: None,
                        occurred_at: timestamp(1),
                        context: MessageTimelineContext::default(),
                    },
                    cx,
                )?;
                timeline.retry_optimistic("operation-1", cx)?;
                timeline.reject_optimistic("operation-1", "permission denied", cx)
            })
            .expect("rejected optimistic message should remain visible");

        let items = item_snapshots(&timeline, cx);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].lifecycle, ActivityLifecycle::Failed);
        assert_eq!(
            items[0].outcome.summary.as_deref(),
            Some("permission denied")
        );

        let mut accepted = entry("event-1", 1);
        accepted.operation_id = Some("operation-1".into());
        timeline
            .update(cx, |timeline, cx| timeline.upsert_live(accepted, cx))
            .expect("authority should replace the failed optimistic row");
        let items = item_snapshots(&timeline, cx);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id.source_id(), "event-1");
        assert_eq!(items[0].lifecycle, ActivityLifecycle::Succeeded);
    }
}
