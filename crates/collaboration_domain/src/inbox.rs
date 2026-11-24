use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::{
    AggregateId, CommunityId, Message, MessageLifecycleState, NostrEventId, PrincipalId,
    ReadContextId, ReadState, ReadStateCompleteness, ReadStateError, Reminder, ReminderError,
    ReminderId, ReminderLifecycle,
};

const MAX_INBOX_MESSAGES: usize = 100_000;
const MAX_INBOX_REMINDERS: usize = 10_000;
const MAX_MENTIONS_PER_MESSAGE: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InboxScope {
    community_id: CommunityId,
    viewer_principal_id: PrincipalId,
}

impl InboxScope {
    pub const fn new(community_id: CommunityId, viewer_principal_id: PrincipalId) -> Self {
        Self {
            community_id,
            viewer_principal_id,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn viewer_principal_id(self) -> PrincipalId {
        self.viewer_principal_id
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct InboxMessageInput<'a> {
    pub message: &'a Message,
    pub conversation_id: AggregateId,
    pub read_context: &'a ReadContextId,
    pub parent_read_context: Option<&'a ReadContextId>,
    pub sequence: u32,
    pub mentioned_principal_ids: &'a BTreeSet<PrincipalId>,
    pub reply_to_principal_id: Option<PrincipalId>,
}

impl fmt::Debug for InboxMessageInput<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InboxMessageInput")
            .field("conversation_id", &self.conversation_id)
            .field("sequence", &self.sequence)
            .field("mention_count", &self.mentioned_principal_ids.len())
            .field("reply", &self.reply_to_principal_id.is_some())
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InboxItemKey {
    Conversation(AggregateId),
    Reminder(ReminderId),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InboxCategory {
    Activity,
    Mention,
    Reply,
    Reminder,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InboxItem {
    key: InboxItemKey,
    representative_message_id: Option<AggregateId>,
    latest_activity_at_seconds: u64,
    sort_at_seconds: u64,
    message_count: u32,
    unread_message_count: u32,
    categories: BTreeSet<InboxCategory>,
    pending_reminder_ids: Vec<ReminderId>,
}

impl InboxItem {
    pub const fn key(&self) -> &InboxItemKey {
        &self.key
    }

    pub const fn representative_message_id(&self) -> Option<AggregateId> {
        self.representative_message_id
    }

    pub const fn latest_activity_at_seconds(&self) -> u64 {
        self.latest_activity_at_seconds
    }

    pub const fn sort_at_seconds(&self) -> u64 {
        self.sort_at_seconds
    }

    pub const fn message_count(&self) -> u32 {
        self.message_count
    }

    pub const fn unread_message_count(&self) -> u32 {
        self.unread_message_count
    }

    pub fn categories(&self) -> &BTreeSet<InboxCategory> {
        &self.categories
    }

    pub fn pending_reminder_ids(&self) -> &[ReminderId] {
        &self.pending_reminder_ids
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InboxProjection {
    scope: InboxScope,
    read_state_completeness: ReadStateCompleteness,
    items: Vec<InboxItem>,
}

impl InboxProjection {
    pub fn build<'message, 'reminder>(
        scope: InboxScope,
        messages: impl IntoIterator<Item = InboxMessageInput<'message>>,
        read_state: &ReadState,
        reminders: impl IntoIterator<Item = &'reminder Reminder>,
    ) -> Result<Self, InboxError> {
        if scope.community_id.as_uuid().is_nil() || scope.viewer_principal_id.as_uuid().is_nil() {
            return Err(InboxError::InvalidScope);
        }
        if read_state.scope().community_id() != scope.community_id
            || read_state.scope().owner_principal_id() != scope.viewer_principal_id
        {
            return Err(InboxError::ScopeMismatch);
        }

        let mut messages_by_id = BTreeMap::new();
        for (input_count, input) in messages.into_iter().enumerate() {
            if input_count >= MAX_INBOX_MESSAGES {
                return Err(InboxError::TooManyMessages);
            }
            let normalized = NormalizedMessage::from_input(scope, input)?;
            match messages_by_id.entry(normalized.message_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(normalized);
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() == &normalized => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(InboxError::ConflictingMessageDuplicate);
                }
            }
        }

        let mut conversations = BTreeMap::<AggregateId, Vec<NormalizedMessage<'message>>>::new();
        let mut source_conversations = BTreeMap::<NostrEventId, AggregateId>::new();
        for message in messages_by_id.into_values() {
            if message.lifecycle_state == MessageLifecycleState::Deleted {
                continue;
            }
            if source_conversations
                .insert(message.source_event_id, message.conversation_id)
                .is_some()
            {
                return Err(InboxError::ConflictingMessageDuplicate);
            }
            if message.author_principal_id != scope.viewer_principal_id {
                conversations
                    .entry(message.conversation_id)
                    .or_default()
                    .push(message);
            }
        }

        let mut items = BTreeMap::new();
        for (conversation_id, mut messages) in conversations {
            messages.sort_by_key(NormalizedMessage::order);
            let mut unread = Vec::new();
            let mut categories = BTreeSet::new();
            for message in &messages {
                categories.extend(message.categories.iter().copied());
                if read_state.is_unread(
                    scope.viewer_principal_id,
                    &message.read_context,
                    message.parent_read_context.as_ref(),
                    message.sequence,
                )? {
                    unread.push(message);
                }
            }
            let representative = unread
                .first()
                .copied()
                .or_else(|| messages.last())
                .ok_or(InboxError::InvalidMessageInput)?;
            let latest_activity_at_seconds = messages
                .iter()
                .map(|message| message.created_at_seconds)
                .max()
                .ok_or(InboxError::InvalidMessageInput)?;
            let message_count =
                u32::try_from(messages.len()).map_err(|_| InboxError::CountOverflow)?;
            let unread_message_count =
                u32::try_from(unread.len()).map_err(|_| InboxError::CountOverflow)?;
            items.insert(
                conversation_id,
                InboxItem {
                    key: InboxItemKey::Conversation(conversation_id),
                    representative_message_id: Some(representative.message_id),
                    latest_activity_at_seconds,
                    sort_at_seconds: latest_activity_at_seconds,
                    message_count,
                    unread_message_count,
                    categories,
                    pending_reminder_ids: Vec::new(),
                },
            );
        }

        let mut reminder_records = BTreeMap::new();
        let mut pending_reminders = BTreeMap::new();
        for (input_count, reminder) in reminders.into_iter().enumerate() {
            if input_count >= MAX_INBOX_REMINDERS {
                return Err(InboxError::TooManyReminders);
            }
            if reminder.scope().community_id() != scope.community_id
                || reminder.scope().owner_principal_id() != scope.viewer_principal_id
            {
                return Err(InboxError::ScopeMismatch);
            }
            let record = reminder.record(scope.viewer_principal_id)?;
            match reminder_records.entry(record.reminder_id.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(reminder);
                }
                std::collections::btree_map::Entry::Occupied(entry) if *entry.get() == reminder => {
                    continue;
                }
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(InboxError::ConflictingReminderDuplicate);
                }
            }
            let ReminderLifecycle::Pending { not_before_seconds } = record.lifecycle else {
                continue;
            };
            let pending = PendingReminder {
                reminder_id: record.reminder_id,
                target_event_id: record.content.target().and_then(|target| target.event_id()),
                not_before_seconds,
            };
            match pending_reminders.entry(pending.reminder_id.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(pending);
                }
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(InboxError::ConflictingReminderDuplicate);
                }
            }
        }

        let mut standalone_reminders = Vec::new();
        for pending in pending_reminders.into_values() {
            let conversation = pending
                .target_event_id
                .and_then(|event_id| source_conversations.get(&event_id))
                .and_then(|conversation_id| items.get_mut(conversation_id));
            if let Some(item) = conversation {
                item.categories.insert(InboxCategory::Reminder);
                item.sort_at_seconds = item.sort_at_seconds.max(pending.not_before_seconds);
                item.pending_reminder_ids.push(pending.reminder_id);
            } else {
                standalone_reminders.push(InboxItem {
                    key: InboxItemKey::Reminder(pending.reminder_id.clone()),
                    representative_message_id: None,
                    latest_activity_at_seconds: pending.not_before_seconds,
                    sort_at_seconds: pending.not_before_seconds,
                    message_count: 0,
                    unread_message_count: 0,
                    categories: BTreeSet::from([InboxCategory::Reminder]),
                    pending_reminder_ids: vec![pending.reminder_id],
                });
            }
        }

        let mut items = items.into_values().collect::<Vec<_>>();
        items.extend(standalone_reminders);
        for item in &mut items {
            item.pending_reminder_ids.sort();
        }
        items.sort_by(|left, right| {
            right
                .sort_at_seconds
                .cmp(&left.sort_at_seconds)
                .then_with(|| left.key.cmp(&right.key))
        });

        Ok(Self {
            scope,
            read_state_completeness: read_state.completeness(),
            items,
        })
    }

    pub const fn scope(&self) -> InboxScope {
        self.scope
    }

    pub const fn read_state_completeness(&self) -> ReadStateCompleteness {
        self.read_state_completeness
    }

    pub fn items(&self) -> &[InboxItem] {
        &self.items
    }
}

#[derive(Clone, Eq, PartialEq)]
struct NormalizedMessage<'a> {
    message: &'a Message,
    message_id: AggregateId,
    conversation_id: AggregateId,
    source_event_id: NostrEventId,
    author_principal_id: PrincipalId,
    created_at_seconds: u64,
    lifecycle_state: MessageLifecycleState,
    read_context: ReadContextId,
    parent_read_context: Option<ReadContextId>,
    sequence: u32,
    categories: BTreeSet<InboxCategory>,
}

impl<'a> NormalizedMessage<'a> {
    fn from_input(scope: InboxScope, input: InboxMessageInput<'a>) -> Result<Self, InboxError> {
        let fields = input.message.fields();
        if fields.community_id != scope.community_id {
            return Err(InboxError::ScopeMismatch);
        }
        if input.conversation_id.as_uuid().is_nil()
            || input.sequence == 0
            || input.mentioned_principal_ids.len() > MAX_MENTIONS_PER_MESSAGE
            || input
                .mentioned_principal_ids
                .iter()
                .any(|principal_id| principal_id.as_uuid().is_nil())
            || input
                .reply_to_principal_id
                .is_some_and(|principal_id| principal_id.as_uuid().is_nil())
        {
            return Err(InboxError::InvalidMessageInput);
        }
        let mut categories = BTreeSet::from([InboxCategory::Activity]);
        if input
            .mentioned_principal_ids
            .contains(&scope.viewer_principal_id)
        {
            categories.insert(InboxCategory::Mention);
        }
        if input.reply_to_principal_id == Some(scope.viewer_principal_id) {
            categories.insert(InboxCategory::Reply);
        }
        Ok(Self {
            message: input.message,
            message_id: fields.message_id,
            conversation_id: input.conversation_id,
            source_event_id: fields.source.event_id,
            author_principal_id: fields.author.principal_id(),
            created_at_seconds: fields.source.event_created_at,
            lifecycle_state: fields.lifecycle_state,
            read_context: input.read_context.clone(),
            parent_read_context: input.parent_read_context.cloned(),
            sequence: input.sequence,
            categories,
        })
    }

    fn order(&self) -> (u32, u64, NostrEventId, AggregateId) {
        (
            self.sequence,
            self.created_at_seconds,
            self.source_event_id,
            self.message_id,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingReminder {
    reminder_id: ReminderId,
    target_event_id: Option<NostrEventId>,
    not_before_seconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InboxError {
    InvalidScope,
    ScopeMismatch,
    InvalidMessageInput,
    TooManyMessages,
    TooManyReminders,
    ConflictingMessageDuplicate,
    ConflictingReminderDuplicate,
    CountOverflow,
    ReadState(ReadStateError),
    Reminder(ReminderError),
}

impl fmt::Display for InboxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScope => formatter.write_str("invalid inbox scope"),
            Self::ScopeMismatch => formatter.write_str("inbox scope mismatch"),
            Self::InvalidMessageInput => formatter.write_str("invalid inbox message input"),
            Self::TooManyMessages => formatter.write_str("too many inbox messages"),
            Self::TooManyReminders => formatter.write_str("too many inbox reminders"),
            Self::ConflictingMessageDuplicate => {
                formatter.write_str("conflicting inbox message duplicate")
            }
            Self::ConflictingReminderDuplicate => {
                formatter.write_str("conflicting inbox reminder duplicate")
            }
            Self::CountOverflow => formatter.write_str("inbox count overflow"),
            Self::ReadState(error) => write!(formatter, "inbox read state failed: {error}"),
            Self::Reminder(error) => write!(formatter, "inbox reminder failed: {error}"),
        }
    }
}

impl Error for InboxError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ReadState(error) => Some(error),
            Self::Reminder(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ReadStateError> for InboxError {
    fn from(value: ReadStateError) -> Self {
        Self::ReadState(value)
    }
}

impl From<ReminderError> for InboxError {
    fn from(value: ReminderError) -> Self {
        Self::Reminder(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AggregateVersion, ManualUnreadRegister, MessageContent, MessageMutation,
        MessageMutationKind, MessageRecordFields, MessageSource, OwnerReadStateReplica,
        ReadStateScope, ReminderContent, ReminderHead, ReminderScope, ReminderTarget,
    };
    use uuid::Uuid;

    fn community_id(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn event_id(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn message(
        community_id: CommunityId,
        message_id: AggregateId,
        author_principal_id: PrincipalId,
        event: NostrEventId,
        created_at_seconds: u64,
    ) -> Message {
        let source = MessageSource {
            event_id: event,
            event_created_at: created_at_seconds,
        };
        Message::from_record(MessageRecordFields {
            community_id,
            channel_id: aggregate_id(50),
            message_id,
            author: crate::MessageAuthor::principal(author_principal_id),
            content: MessageContent::new("canonical content").expect("bounded content"),
            lifecycle_state: MessageLifecycleState::Active,
            source,
            current_source: source,
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("valid canonical message")
    }

    fn deleted_message(
        community_id: CommunityId,
        message_id: AggregateId,
        author_principal_id: PrincipalId,
        event: NostrEventId,
        created_at_seconds: u64,
    ) -> Message {
        let source = MessageSource {
            event_id: event,
            event_created_at: created_at_seconds,
        };
        let delete_source = MessageSource {
            event_id: event_id(event.as_bytes()[0].saturating_add(100)),
            event_created_at: created_at_seconds + 1,
        };
        let second_version = AggregateVersion::FIRST.next().expect("second version");
        Message::from_record(MessageRecordFields {
            community_id,
            channel_id: aggregate_id(50),
            message_id,
            author: crate::MessageAuthor::principal(author_principal_id),
            content: MessageContent::new("deleted content").expect("bounded content"),
            lifecycle_state: MessageLifecycleState::Deleted,
            source,
            current_source: delete_source,
            mutations: vec![MessageMutation {
                source: delete_source,
                actor_principal_id: author_principal_id,
                kind: MessageMutationKind::Delete {
                    moderated: false,
                    metadata: None,
                },
                resulting_version: second_version,
            }],
            version: second_version,
        })
        .expect("valid deleted message")
    }

    fn read_state(
        scope: InboxScope,
        frontiers: impl IntoIterator<Item = (ReadContextId, u32)>,
    ) -> ReadState {
        let read_scope = ReadStateScope::new(scope.community_id, scope.viewer_principal_id);
        let replica = OwnerReadStateReplica::new(
            read_scope,
            scope.viewer_principal_id,
            frontiers,
            Vec::<(ReadContextId, ManualUnreadRegister)>::new(),
        )
        .expect("valid owner read state");
        ReadState::from_replicas(read_scope, ReadStateCompleteness::Complete, [replica])
            .expect("valid read state")
    }

    fn reminder(
        scope: InboxScope,
        reminder_id: &str,
        event: u8,
        target_event_id: Option<NostrEventId>,
        not_before_seconds: u64,
    ) -> Reminder {
        let target = target_event_id
            .map(|event_id| ReminderTarget::new(Some(event_id), None).expect("valid target"));
        Reminder::create(
            ReminderScope::new(scope.community_id, scope.viewer_principal_id),
            scope.viewer_principal_id,
            ReminderId::new(reminder_id).expect("valid reminder id"),
            ReminderHead::new(event_id(event), not_before_seconds.saturating_sub(1))
                .expect("valid reminder head"),
            ReminderContent::new(target, Some("private note".to_owned())),
            not_before_seconds,
        )
        .expect("valid reminder")
    }

    #[test]
    fn projection_derives_mentions_replies_and_oldest_unread_representative() {
        let scope = InboxScope::new(community_id(1), principal_id(2));
        let other = principal_id(3);
        let conversation_id = aggregate_id(10);
        let context = ReadContextId::new("thread:10").expect("valid context");
        let first = message(scope.community_id, aggregate_id(11), other, event_id(1), 10);
        let reply = message(scope.community_id, aggregate_id(12), other, event_id(2), 20);
        let mention = BTreeSet::from([scope.viewer_principal_id]);
        let no_mentions = BTreeSet::new();
        let state = read_state(scope, [(context.clone(), 1)]);

        let projection = InboxProjection::build(
            scope,
            [
                InboxMessageInput {
                    message: &first,
                    conversation_id,
                    read_context: &context,
                    parent_read_context: None,
                    sequence: 1,
                    mentioned_principal_ids: &no_mentions,
                    reply_to_principal_id: None,
                },
                InboxMessageInput {
                    message: &reply,
                    conversation_id,
                    read_context: &context,
                    parent_read_context: None,
                    sequence: 2,
                    mentioned_principal_ids: &mention,
                    reply_to_principal_id: Some(scope.viewer_principal_id),
                },
            ],
            &state,
            [],
        )
        .expect("valid projection");

        let [item] = projection.items() else {
            panic!("one conversation expected");
        };
        assert_eq!(item.key(), &InboxItemKey::Conversation(conversation_id));
        assert_eq!(item.representative_message_id(), Some(aggregate_id(12)));
        assert_eq!(item.unread_message_count(), 1);
        assert_eq!(item.message_count(), 2);
        assert_eq!(
            item.categories(),
            &BTreeSet::from([
                InboxCategory::Activity,
                InboxCategory::Mention,
                InboxCategory::Reply,
            ])
        );
    }

    #[test]
    fn pending_reminders_enrich_targets_and_sort_standalone_rows() {
        let scope = InboxScope::new(community_id(1), principal_id(2));
        let context = ReadContextId::new("channel:50").expect("valid context");
        let message = message(
            scope.community_id,
            aggregate_id(11),
            principal_id(3),
            event_id(1),
            10,
        );
        let state = read_state(scope, [(context.clone(), 10)]);
        let target = reminder(scope, "target", 20, Some(event_id(1)), 30);
        let standalone = reminder(scope, "standalone", 21, None, 40);
        let no_mentions = BTreeSet::new();

        let projection = InboxProjection::build(
            scope,
            [InboxMessageInput {
                message: &message,
                conversation_id: aggregate_id(10),
                read_context: &context,
                parent_read_context: None,
                sequence: 10,
                mentioned_principal_ids: &no_mentions,
                reply_to_principal_id: None,
            }],
            &state,
            [&target, &standalone],
        )
        .expect("valid projection");

        assert_eq!(projection.items().len(), 2);
        assert_eq!(
            projection.items()[0].key(),
            &InboxItemKey::Reminder(ReminderId::new("standalone").expect("valid id"))
        );
        let conversation = &projection.items()[1];
        assert!(conversation.categories().contains(&InboxCategory::Reminder));
        assert_eq!(conversation.sort_at_seconds(), 30);
        assert_eq!(
            conversation.pending_reminder_ids(),
            &[ReminderId::new("target").expect("valid id")]
        );
    }

    #[test]
    fn read_deleted_and_duplicate_records_remain_canonical() {
        let scope = InboxScope::new(community_id(1), principal_id(2));
        let context = ReadContextId::new("thread:10").expect("valid context");
        let active = message(
            scope.community_id,
            aggregate_id(11),
            principal_id(3),
            event_id(1),
            10,
        );
        let deleted = deleted_message(
            scope.community_id,
            aggregate_id(12),
            principal_id(3),
            event_id(2),
            20,
        );
        let state = read_state(scope, [(context.clone(), 1)]);
        let no_mentions = BTreeSet::new();
        let active_input = InboxMessageInput {
            message: &active,
            conversation_id: aggregate_id(10),
            read_context: &context,
            parent_read_context: None,
            sequence: 1,
            mentioned_principal_ids: &no_mentions,
            reply_to_principal_id: None,
        };
        let deleted_input = InboxMessageInput {
            message: &deleted,
            conversation_id: aggregate_id(10),
            read_context: &context,
            parent_read_context: None,
            sequence: 2,
            mentioned_principal_ids: &no_mentions,
            reply_to_principal_id: None,
        };

        let projection = InboxProjection::build(
            scope,
            [active_input.clone(), active_input, deleted_input],
            &state,
            [],
        )
        .expect("duplicates are idempotent");

        let [item] = projection.items() else {
            panic!("one conversation expected");
        };
        assert_eq!(item.message_count(), 1);
        assert_eq!(item.unread_message_count(), 0);
        assert_eq!(item.representative_message_id(), Some(aggregate_id(11)));
    }
}
