use std::{fmt, num::NonZeroU32};

use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, CustomEmoji, CustomEmojiPaletteEntry,
    CustomEmojiShortcode, DmLifecycleState, DmOpenFields, DmParticipantState, DmRecordFields,
    FeedbackCreateFields, FeedbackStatus, FeedbackStatusReason, FeedbackStatusSource,
    FeedbackStatusView, ForumPostCursor, ForumVoteDirection, ForumVoteSummary, ManualUnreadState,
    MessageLifecycleState, MessageRecordFields, NostrEventId, OperationId, PrincipalId,
    ReadContextId, ReadStateCompleteness, ReadStateScope, ReminderContent, ReminderDismissal,
    ReminderHead, ReminderId, ReminderLifecycle, ReminderRecordFields, ReminderScope,
};
use nostr_compat::{PublicKey, dm::Nip44Ciphertext};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

#[derive(Clone, Eq, PartialEq)]
pub enum SocialCliCommand {
    ListDms {
        community_id: CommunityId,
        owner_principal_id: PrincipalId,
        limit: NonZeroU32,
    },
    OpenDm {
        fields: DmOpenFields,
        operation_id: OperationId,
    },
    AddDmParticipant {
        community_id: CommunityId,
        dm_id: AggregateId,
        participant_id: PrincipalId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    LeaveDm {
        community_id: CommunityId,
        dm_id: AggregateId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
    SendEncryptedDm {
        community_id: CommunityId,
        dm_id: AggregateId,
        recipient: PublicKey,
        ciphertext: Nip44Ciphertext,
        operation_id: OperationId,
    },
    ReadStatus {
        scope: ReadStateScope,
        context: ReadContextId,
        parent: Option<ReadContextId>,
        latest_message_at: u32,
    },
    MarkRead {
        scope: ReadStateScope,
        context: ReadContextId,
        parent: Option<ReadContextId>,
        read_through: u32,
        operation_id: OperationId,
    },
    MarkUnread {
        scope: ReadStateScope,
        context: ReadContextId,
        parent: Option<ReadContextId>,
        operation_id: OperationId,
    },
    ListReminders {
        scope: ReminderScope,
    },
    CreateReminder {
        scope: ReminderScope,
        reminder_id: ReminderId,
        head: ReminderHead,
        content: ReminderContent,
        not_before_seconds: u64,
        operation_id: OperationId,
    },
    UpdateReminder {
        scope: ReminderScope,
        reminder_id: ReminderId,
        head: ReminderHead,
        content: ReminderContent,
        not_before_seconds: u64,
        operation_id: OperationId,
    },
    DismissReminder {
        scope: ReminderScope,
        reminder_id: ReminderId,
        head: ReminderHead,
        dismissal: ReminderDismissal,
        expiration_seconds: u64,
        operation_id: OperationId,
    },
    ListForumPosts {
        community_id: CommunityId,
        channel_id: AggregateId,
        cursor: Option<ForumPostCursor>,
        limit: NonZeroU32,
    },
    VoteForum {
        community_id: CommunityId,
        channel_id: AggregateId,
        target_event_id: NostrEventId,
        direction: ForumVoteDirection,
        operation_id: OperationId,
    },
    ListEmoji {
        community_id: CommunityId,
    },
    SetEmoji {
        community_id: CommunityId,
        emoji: CustomEmoji,
        operation_id: OperationId,
    },
    RemoveEmoji {
        community_id: CommunityId,
        shortcode: CustomEmojiShortcode,
        operation_id: OperationId,
    },
    SubmitFeedback {
        fields: FeedbackCreateFields,
        operation_id: OperationId,
    },
    GetFeedbackStatus {
        community_id: CommunityId,
        feedback_event_id: NostrEventId,
    },
    UpdateFeedbackStatus {
        community_id: CommunityId,
        feedback_event_id: NostrEventId,
        expected_version: AggregateVersion,
        status: FeedbackStatus,
        reason: FeedbackStatusReason,
        source: FeedbackStatusSource,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SocialCliVerb {
    ListDms,
    OpenDm,
    AddDmParticipant,
    LeaveDm,
    SendEncryptedDm,
    ReadStatus,
    MarkRead,
    MarkUnread,
    ListReminders,
    CreateReminder,
    UpdateReminder,
    DismissReminder,
    ListForumPosts,
    VoteForum,
    ListEmoji,
    SetEmoji,
    RemoveEmoji,
    SubmitFeedback,
    GetFeedbackStatus,
    UpdateFeedbackStatus,
}

impl SocialCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ListDms => "dm.list",
            Self::OpenDm => "dm.open",
            Self::AddDmParticipant => "dm.participant.add",
            Self::LeaveDm => "dm.leave",
            Self::SendEncryptedDm => "dm.send_encrypted",
            Self::ReadStatus => "read.status",
            Self::MarkRead => "read.mark",
            Self::MarkUnread => "unread.mark",
            Self::ListReminders => "reminder.list",
            Self::CreateReminder => "reminder.create",
            Self::UpdateReminder => "reminder.update",
            Self::DismissReminder => "reminder.dismiss",
            Self::ListForumPosts => "forum.posts",
            Self::VoteForum => "forum.vote",
            Self::ListEmoji => "emoji.list",
            Self::SetEmoji => "emoji.set",
            Self::RemoveEmoji => "emoji.remove",
            Self::SubmitFeedback => "feedback.submit",
            Self::GetFeedbackStatus => "feedback.status",
            Self::UpdateFeedbackStatus => "feedback.status.update",
        }
    }
}

impl SocialCliCommand {
    const fn verb(&self) -> SocialCliVerb {
        match self {
            Self::ListDms { .. } => SocialCliVerb::ListDms,
            Self::OpenDm { .. } => SocialCliVerb::OpenDm,
            Self::AddDmParticipant { .. } => SocialCliVerb::AddDmParticipant,
            Self::LeaveDm { .. } => SocialCliVerb::LeaveDm,
            Self::SendEncryptedDm { .. } => SocialCliVerb::SendEncryptedDm,
            Self::ReadStatus { .. } => SocialCliVerb::ReadStatus,
            Self::MarkRead { .. } => SocialCliVerb::MarkRead,
            Self::MarkUnread { .. } => SocialCliVerb::MarkUnread,
            Self::ListReminders { .. } => SocialCliVerb::ListReminders,
            Self::CreateReminder { .. } => SocialCliVerb::CreateReminder,
            Self::UpdateReminder { .. } => SocialCliVerb::UpdateReminder,
            Self::DismissReminder { .. } => SocialCliVerb::DismissReminder,
            Self::ListForumPosts { .. } => SocialCliVerb::ListForumPosts,
            Self::VoteForum { .. } => SocialCliVerb::VoteForum,
            Self::ListEmoji { .. } => SocialCliVerb::ListEmoji,
            Self::SetEmoji { .. } => SocialCliVerb::SetEmoji,
            Self::RemoveEmoji { .. } => SocialCliVerb::RemoveEmoji,
            Self::SubmitFeedback { .. } => SocialCliVerb::SubmitFeedback,
            Self::GetFeedbackStatus { .. } => SocialCliVerb::GetFeedbackStatus,
            Self::UpdateFeedbackStatus { .. } => SocialCliVerb::UpdateFeedbackStatus,
        }
    }
}

impl fmt::Debug for SocialCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SocialCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct SocialReadStatus {
    pub scope: ReadStateScope,
    pub context: ReadContextId,
    pub completeness: ReadStateCompleteness,
    pub effective_frontier: u32,
    pub manual_unread: ManualUnreadState,
    pub unread: bool,
}

impl fmt::Debug for SocialReadStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SocialReadStatus")
            .field("completeness", &self.completeness)
            .field("unread", &self.unread)
            .field("content", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ForumCliPost {
    pub message: MessageRecordFields,
    pub votes: ForumVoteSummary,
    pub reply_count: u64,
}

impl fmt::Debug for ForumCliPost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ForumCliPost")
            .field("message_id", &self.message.message_id)
            .field("votes", &self.votes)
            .field("reply_count", &self.reply_count)
            .field("content", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumCliPage {
    pub posts: Vec<ForumCliPost>,
    pub has_more: bool,
    pub next_cursor: Option<ForumPostCursor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncryptedDmReceipt {
    pub operation_id: OperationId,
    pub dm_id: AggregateId,
    pub recipient: PublicKey,
    pub event_id: NostrEventId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SocialResourceId {
    Dm(AggregateId),
    ReadState(PrincipalId),
    Reminder(ReminderId),
    Forum(NostrEventId),
    Emoji(CustomEmojiShortcode),
    Feedback(NostrEventId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SocialWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: SocialResourceId,
    pub version: Option<AggregateVersion>,
}

#[derive(Clone, Eq, PartialEq)]
pub enum SocialCliOutcome {
    Dms(Vec<DmRecordFields>),
    EncryptedDmSent(EncryptedDmReceipt),
    ReadStatus(SocialReadStatus),
    Reminders(Vec<ReminderRecordFields>),
    ForumPosts(ForumCliPage),
    EmojiPalette(Vec<CustomEmojiPaletteEntry>),
    FeedbackStatus(FeedbackStatusView),
    Applied(SocialWriteReceipt),
}

impl fmt::Debug for SocialCliOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let variant = match self {
            Self::Dms(_) => "Dms",
            Self::EncryptedDmSent(_) => "EncryptedDmSent",
            Self::ReadStatus(_) => "ReadStatus",
            Self::Reminders(_) => "Reminders",
            Self::ForumPosts(_) => "ForumPosts",
            Self::EmojiPalette(_) => "EmojiPalette",
            Self::FeedbackStatus(_) => "FeedbackStatus",
            Self::Applied(_) => "Applied",
        };
        formatter
            .debug_struct("SocialCliOutcome")
            .field("variant", &variant)
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SocialCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    AuthorizationDenied,
    PrivacyDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl SocialCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "social_cli_invalid_request",
            Self::NotFound => "social_cli_not_found",
            Self::Unavailable => "social_cli_unavailable",
            Self::AuthorizationDenied | Self::PrivacyDenied => "social_cli_private_resource_denied",
            Self::PartialFailure => "social_cli_completion_unknown",
            Self::Unexpected => "social_cli_unexpected_response",
            Self::Conflict => "social_cli_stale_version",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::AuthorizationDenied | Self::PrivacyDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait SocialCliExecutor {
    fn execute(&self, command: SocialCliCommand) -> Result<SocialCliOutcome, SocialCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SocialCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_social_command(
    executor: &impl SocialCliExecutor,
    command: SocialCliCommand,
) -> SocialCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => success_output(verb, outcome)
            .map(SocialCliExecution::success)
            .unwrap_or_else(|| error_output(verb, SocialCliError::Unexpected)),
        Err(error) => error_output(verb, error),
    }
}

impl SocialCliExecution {
    fn success(value: Value) -> Self {
        Self {
            stdout: format!("{value}\n"),
            stderr: String::new(),
            exit_code: 0,
        }
    }

    fn failure(value: Value, exit_code: i32) -> Self {
        Self {
            stdout: String::new(),
            stderr: format!("{value}\n"),
            exit_code,
        }
    }
}

fn success_output(verb: SocialCliVerb, outcome: SocialCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (SocialCliVerb::ListDms, SocialCliOutcome::Dms(dms)) => Some(json!({
            "command": verb.as_str(),
            "conversations": dms.iter().map(dm_output).collect::<Vec<_>>(),
            "ok": true,
        })),
        (SocialCliVerb::SendEncryptedDm, SocialCliOutcome::EncryptedDmSent(receipt)) => {
            Some(json!({
                "command": verb.as_str(),
                "dm_id": receipt.dm_id,
                "event_id": hex_bytes(receipt.event_id.as_bytes()),
                "ok": true,
                "operation_id": receipt.operation_id,
                "recipient": receipt.recipient.to_hex(),
            }))
        }
        (SocialCliVerb::ReadStatus, SocialCliOutcome::ReadStatus(status)) => {
            Some(read_status_output(verb, &status))
        }
        (SocialCliVerb::ListReminders, SocialCliOutcome::Reminders(reminders)) => Some(json!({
            "command": verb.as_str(),
            "ok": true,
            "reminders": reminders.iter().map(reminder_output).collect::<Vec<_>>(),
        })),
        (SocialCliVerb::ListForumPosts, SocialCliOutcome::ForumPosts(page)) => {
            Some(forum_page_output(verb, &page))
        }
        (SocialCliVerb::ListEmoji, SocialCliOutcome::EmojiPalette(entries)) => Some(json!({
            "command": verb.as_str(),
            "emoji": entries.iter().map(emoji_output).collect::<Vec<_>>(),
            "ok": true,
        })),
        (SocialCliVerb::GetFeedbackStatus, SocialCliOutcome::FeedbackStatus(status)) => {
            Some(feedback_status_output(verb, status))
        }
        (
            SocialCliVerb::OpenDm
            | SocialCliVerb::AddDmParticipant
            | SocialCliVerb::LeaveDm
            | SocialCliVerb::MarkRead
            | SocialCliVerb::MarkUnread
            | SocialCliVerb::CreateReminder
            | SocialCliVerb::UpdateReminder
            | SocialCliVerb::DismissReminder
            | SocialCliVerb::VoteForum
            | SocialCliVerb::SetEmoji
            | SocialCliVerb::RemoveEmoji
            | SocialCliVerb::SubmitFeedback
            | SocialCliVerb::UpdateFeedbackStatus,
            SocialCliOutcome::Applied(receipt),
        ) => Some(write_output(verb, receipt)),
        _ => None,
    }
}

fn error_output(verb: SocialCliVerb, error: SocialCliError) -> SocialCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    SocialCliExecution::failure(
        json!({
            "command": verb.as_str(),
            "error": contract.category,
            "error_code": diagnostic,
            "message": diagnostic,
            "ok": false,
            "retryable": contract.retryable,
        }),
        contract.exit_class as i32,
    )
}

fn dm_output(dm: &DmRecordFields) -> Value {
    json!({
        "community_id": dm.community_id,
        "dm_id": dm.dm_id,
        "lifecycle": match dm.lifecycle_state {
            DmLifecycleState::Open => "open",
            DmLifecycleState::Closed => "closed",
        },
        "participants": dm.participant_states.iter().map(|(principal_id, state)| json!({
            "principal_id": principal_id,
            "state": match state {
                DmParticipantState::Active => "active",
                DmParticipantState::Left => "left",
                DmParticipantState::Removed => "removed",
            },
        })).collect::<Vec<_>>(),
        "version": dm.version,
    })
}

fn read_status_output(verb: SocialCliVerb, status: &SocialReadStatus) -> Value {
    let manual_unread = match status.manual_unread {
        ManualUnreadState::Virgin => json!({ "state": "virgin" }),
        ManualUnreadState::Live(register) => json!({
            "baseline": register.baseline(),
            "clear": register.clear(),
            "set": register.set(),
            "state": "live",
        }),
        ManualUnreadState::Tombstone { counter } => {
            json!({ "counter": counter, "state": "tombstone" })
        }
    };
    json!({
        "command": verb.as_str(),
        "community_id": status.scope.community_id(),
        "completeness": match status.completeness {
            ReadStateCompleteness::Complete => "complete",
            ReadStateCompleteness::PotentiallyIncomplete => "potentially_incomplete",
        },
        "context": status.context.as_str(),
        "effective_frontier": status.effective_frontier,
        "manual_unread": manual_unread,
        "ok": true,
        "owner_principal_id": status.scope.owner_principal_id(),
        "unread": status.unread,
    })
}

fn reminder_output(reminder: &ReminderRecordFields) -> Value {
    let target = reminder.content.target().map(|target| {
        json!({
            "address": target.address(),
            "event_id": target.event_id().map(|event_id| hex_bytes(event_id.as_bytes())),
        })
    });
    json!({
        "community_id": reminder.scope.community_id(),
        "created_at_seconds": reminder.head.created_at_seconds(),
        "expiration_seconds": reminder.expiration_seconds,
        "lifecycle": match reminder.lifecycle {
            ReminderLifecycle::Pending { not_before_seconds } => json!({
                "not_before_seconds": not_before_seconds,
                "state": "pending",
            }),
            ReminderLifecycle::Done => json!({ "state": "done" }),
            ReminderLifecycle::Cancelled => json!({ "state": "cancelled" }),
        },
        "note": reminder.content.note(),
        "owner_principal_id": reminder.scope.owner_principal_id(),
        "reminder_id": reminder.reminder_id.as_str(),
        "target": target,
    })
}

fn forum_page_output(verb: SocialCliVerb, page: &ForumCliPage) -> Value {
    json!({
        "command": verb.as_str(),
        "has_more": page.has_more,
        "next_cursor": page.next_cursor.map(|cursor| json!({
            "created_at": cursor.created_at,
            "event_id": hex_bytes(cursor.event_id.as_bytes()),
        })),
        "ok": true,
        "posts": page.posts.iter().map(|post| json!({
            "author_principal_id": post.message.author.principal_id(),
            "channel_id": post.message.channel_id,
            "community_id": post.message.community_id,
            "content": (post.message.lifecycle_state != MessageLifecycleState::Deleted)
                .then(|| post.message.content.as_str()),
            "event_id": hex_bytes(post.message.current_source.event_id.as_bytes()),
            "message_id": post.message.message_id,
            "reply_count": post.reply_count,
            "version": post.message.version,
            "votes": vote_output(post.votes),
        })).collect::<Vec<_>>(),
    })
}

fn vote_output(votes: ForumVoteSummary) -> Value {
    json!({
        "downvotes": votes.downvotes,
        "score": votes.score,
        "upvotes": votes.upvotes,
        "viewer_vote": votes.viewer_vote.map(|direction| match direction {
            ForumVoteDirection::Up => "up",
            ForumVoteDirection::Down => "down",
        }),
    })
}

fn emoji_output(entry: &CustomEmojiPaletteEntry) -> Value {
    json!({
        "asset": entry.emoji.asset.as_str(),
        "owner_principal_id": entry.owner_principal_id,
        "shortcode": entry.emoji.shortcode.as_str(),
        "source_created_at": entry.source.event_created_at,
        "source_event_id": hex_bytes(entry.source.event_id.as_bytes()),
    })
}

fn feedback_status_output(verb: SocialCliVerb, status: FeedbackStatusView) -> Value {
    json!({
        "category": status.category.map(feedback_category),
        "command": verb.as_str(),
        "community_id": status.community_id,
        "feedback_event_id": hex_bytes(status.feedback_event_id.as_bytes()),
        "ok": true,
        "reason": status.reason.map(feedback_reason),
        "status": feedback_status(status.status),
        "updated_at": status.updated_at,
        "version": status.version,
    })
}

fn write_output(verb: SocialCliVerb, receipt: SocialWriteReceipt) -> Value {
    let (resource_kind, resource_id) = resource_output(receipt.resource_id);
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": resource_id,
        "resource_kind": resource_kind,
        "version": receipt.version,
    })
}

fn resource_output(resource: SocialResourceId) -> (&'static str, String) {
    match resource {
        SocialResourceId::Dm(dm_id) => ("dm", dm_id.to_string()),
        SocialResourceId::ReadState(principal_id) => ("read_state", principal_id.to_string()),
        SocialResourceId::Reminder(reminder_id) => ("reminder", reminder_id.as_str().to_owned()),
        SocialResourceId::Forum(event_id) => ("forum", hex_bytes(event_id.as_bytes())),
        SocialResourceId::Emoji(shortcode) => ("emoji", shortcode.as_str().to_owned()),
        SocialResourceId::Feedback(event_id) => ("feedback", hex_bytes(event_id.as_bytes())),
    }
}

const fn feedback_category(category: collaboration_domain::FeedbackCategory) -> &'static str {
    match category {
        collaboration_domain::FeedbackCategory::Bug => "bug",
        collaboration_domain::FeedbackCategory::Praise => "praise",
        collaboration_domain::FeedbackCategory::NeedsWork => "needs_work",
    }
}

const fn feedback_status(status: FeedbackStatus) -> &'static str {
    match status {
        FeedbackStatus::Submitted => "submitted",
        FeedbackStatus::Reviewing => "reviewing",
        FeedbackStatus::Resolved => "resolved",
        FeedbackStatus::Declined => "declined",
    }
}

const fn feedback_reason(reason: FeedbackStatusReason) -> &'static str {
    match reason {
        FeedbackStatusReason::Acknowledged => "acknowledged",
        FeedbackStatusReason::Addressed => "addressed",
        FeedbackStatusReason::Duplicate => "duplicate",
        FeedbackStatusReason::OutOfScope => "out_of_scope",
        FeedbackStatusReason::UnableToReproduce => "unable_to_reproduce",
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::{BTreeMap, BTreeSet},
    };

    use collaboration_domain::{
        FeedbackCategory, FeedbackStatusView, MessageAuthor, MessageContent, MessageSource,
        ReminderTarget,
    };
    use uuid::Uuid;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<SocialCliCommand>>,
        result: RefCell<Option<Result<SocialCliOutcome, SocialCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<SocialCliOutcome, SocialCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl SocialCliExecutor for TestExecutor {
        fn execute(&self, command: SocialCliCommand) -> Result<SocialCliOutcome, SocialCliError> {
            self.command.replace(Some(command));
            self.result
                .borrow_mut()
                .take()
                .expect("the test executor is called once")
        }
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(20))
    }

    fn event_id(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn source(value: u8) -> MessageSource {
        MessageSource {
            event_id: event_id(value),
            event_created_at: u64::from(value),
        }
    }

    fn write_receipt(resource_id: SocialResourceId) -> SocialWriteReceipt {
        SocialWriteReceipt {
            operation_id: operation_id(),
            resource_id,
            version: Some(AggregateVersion::FIRST),
        }
    }

    #[test]
    fn encrypted_dm_is_forwarded_but_never_logged_or_returned() {
        let wire = "AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let ciphertext = Nip44Ciphertext::parse(wire).expect("canonical NIP-44 ciphertext");
        let command = SocialCliCommand::SendEncryptedDm {
            community_id: community_id(),
            dm_id: aggregate_id(3),
            recipient: PublicKey::from_bytes([4; 32]),
            ciphertext,
            operation_id: operation_id(),
        };
        assert!(!format!("{command:?}").contains(wire));
        let executor =
            TestExecutor::returning(Ok(SocialCliOutcome::EncryptedDmSent(EncryptedDmReceipt {
                operation_id: operation_id(),
                dm_id: aggregate_id(3),
                recipient: PublicKey::from_bytes([4; 32]),
                event_id: event_id(5),
            })));
        let output = execute_social_command(&executor, command);
        assert_eq!(output.exit_code, 0);
        assert!(!output.stdout.contains(wire));
        assert_eq!(
            serde_json::from_str::<Value>(&output.stdout).expect("JSON")["recipient"],
            "04".repeat(32)
        );
        assert!(matches!(
            executor.command.borrow().as_ref(),
            Some(SocialCliCommand::SendEncryptedDm { ciphertext, .. })
                if ciphertext.wire_value() == wire
        ));

        let mut participants = BTreeMap::new();
        participants.insert(principal_id(2), DmParticipantState::Active);
        participants.insert(principal_id(3), DmParticipantState::Left);
        let dm = DmRecordFields {
            community_id: community_id(),
            dm_id: aggregate_id(3),
            initial_participants: BTreeSet::from([principal_id(2), principal_id(3)]),
            participant_states: participants,
            lifecycle_state: DmLifecycleState::Closed,
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        };
        let listed = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::Dms(vec![dm]))),
            SocialCliCommand::ListDms {
                community_id: community_id(),
                owner_principal_id: principal_id(2),
                limit: NonZeroU32::new(10).expect("limit"),
            },
        );
        assert!(listed.stdout.contains("closed"));
        assert!(listed.stdout.contains("left"));
    }

    #[test]
    fn read_and_unread_states_have_stable_owner_scoped_output() {
        let context = ReadContextId::new("channel:3").expect("context");
        let status = SocialReadStatus {
            scope: ReadStateScope::new(community_id(), principal_id(2)),
            context: context.clone(),
            completeness: ReadStateCompleteness::Complete,
            effective_frontier: 41,
            manual_unread: ManualUnreadState::Live(
                collaboration_domain::ManualUnreadRegister::new(2, 1, 41),
            ),
            unread: true,
        };
        assert!(!format!("{status:?}").contains("channel:3"));
        let output = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::ReadStatus(status))),
            SocialCliCommand::ReadStatus {
                scope: ReadStateScope::new(community_id(), principal_id(2)),
                context,
                parent: None,
                latest_message_at: 42,
            },
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(value["context"], "channel:3");
        assert_eq!(value["manual_unread"]["state"], "live");
        assert_eq!(value["unread"], true);

        let mismatch = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::Dms(Vec::new()))),
            SocialCliCommand::MarkUnread {
                scope: ReadStateScope::new(community_id(), principal_id(2)),
                context: ReadContextId::new("channel:3").expect("context"),
                parent: None,
                operation_id: operation_id(),
            },
        );
        assert_eq!(mismatch.exit_code, 4);
        assert!(mismatch.stdout.is_empty());
    }

    #[test]
    fn reminders_emit_owner_content_only_on_explicit_success() {
        let note = "private follow-up";
        let content = ReminderContent::new(
            Some(ReminderTarget::new(Some(event_id(7)), None).expect("target")),
            Some(note.into()),
        );
        let command = SocialCliCommand::CreateReminder {
            scope: ReminderScope::new(community_id(), principal_id(2)),
            reminder_id: ReminderId::new("reminder-1").expect("id"),
            head: ReminderHead::new(event_id(8), 100).expect("head"),
            content: content.clone(),
            not_before_seconds: 200,
            operation_id: operation_id(),
        };
        assert!(!format!("{command:?}").contains(note));
        let created = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::Applied(write_receipt(
                SocialResourceId::Reminder(ReminderId::new("reminder-1").expect("id")),
            )))),
            command,
        );
        assert!(!created.stdout.contains(note));

        let reminder = ReminderRecordFields {
            scope: ReminderScope::new(community_id(), principal_id(2)),
            reminder_id: ReminderId::new("reminder-1").expect("id"),
            head: ReminderHead::new(event_id(8), 100).expect("head"),
            content,
            lifecycle: ReminderLifecycle::Pending {
                not_before_seconds: 200,
            },
            expiration_seconds: None,
            handled: None,
        };
        let listed = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::Reminders(vec![reminder]))),
            SocialCliCommand::ListReminders {
                scope: ReminderScope::new(community_id(), principal_id(2)),
            },
        );
        assert!(listed.stdout.contains(note));
        assert!(listed.stdout.contains("pending"));
    }

    #[test]
    fn forum_and_emoji_outputs_preserve_canonical_projection() {
        let message = MessageRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(3),
            message_id: aggregate_id(4),
            author: MessageAuthor::principal(principal_id(2)),
            content: MessageContent::new("Forum question").expect("content"),
            lifecycle_state: MessageLifecycleState::Active,
            source: source(9),
            current_source: source(9),
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        };
        let page = ForumCliPage {
            posts: vec![ForumCliPost {
                message,
                votes: ForumVoteSummary {
                    upvotes: 3,
                    downvotes: 1,
                    score: 2,
                    viewer_vote: Some(ForumVoteDirection::Up),
                },
                reply_count: 2,
            }],
            has_more: false,
            next_cursor: None,
        };
        assert!(!format!("{page:?}").contains("Forum question"));
        let forum = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::ForumPosts(page))),
            SocialCliCommand::ListForumPosts {
                community_id: community_id(),
                channel_id: aggregate_id(3),
                cursor: None,
                limit: NonZeroU32::new(20).expect("limit"),
            },
        );
        assert!(forum.stdout.contains("Forum question"));
        assert!(forum.stdout.contains("viewer_vote"));

        let emoji = CustomEmoji::new("party", "https://example.com/party.png").expect("emoji");
        let emoji_output = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::EmojiPalette(vec![
                CustomEmojiPaletteEntry {
                    emoji,
                    owner_principal_id: principal_id(2),
                    source: source(10),
                },
            ]))),
            SocialCliCommand::ListEmoji {
                community_id: community_id(),
            },
        );
        assert!(emoji_output.stdout.contains("party"));
        assert!(
            emoji_output
                .stdout
                .contains("https://example.com/party.png")
        );
    }

    #[test]
    fn feedback_status_excludes_private_body_and_uses_canonical_values() {
        let status = FeedbackStatusView {
            community_id: community_id(),
            feedback_event_id: event_id(11),
            category: Some(FeedbackCategory::Bug),
            status: FeedbackStatus::Resolved,
            reason: Some(FeedbackStatusReason::Addressed),
            version: AggregateVersion::FIRST,
            updated_at: 100,
        };
        let output = execute_social_command(
            &TestExecutor::returning(Ok(SocialCliOutcome::FeedbackStatus(status))),
            SocialCliCommand::GetFeedbackStatus {
                community_id: community_id(),
                feedback_event_id: event_id(11),
            },
        );
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON");
        assert_eq!(value["category"], "bug");
        assert_eq!(value["status"], "resolved");
        assert_eq!(value["reason"], "addressed");
        assert!(value.get("body").is_none());
    }

    #[test]
    fn privacy_safe_error_and_complete_exit_matrix_are_stable() {
        let cases = [
            (SocialCliError::InvalidRequest, "user_error", 1, false),
            (SocialCliError::NotFound, "not_found", 1, false),
            (SocialCliError::Unavailable, "network_error", 2, true),
            (SocialCliError::PartialFailure, "delivery_unknown", 2, false),
            (SocialCliError::AuthorizationDenied, "auth_error", 3, false),
            (SocialCliError::PrivacyDenied, "auth_error", 3, false),
            (SocialCliError::Unexpected, "error", 4, false),
            (SocialCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_social_command(
                &TestExecutor::returning(Err(error)),
                SocialCliCommand::GetFeedbackStatus {
                    community_id: community_id(),
                    feedback_event_id: event_id(11),
                },
            );
            let value: Value = serde_json::from_str(&output.stderr).expect("error JSON");
            assert_eq!(value["error"], category);
            assert_eq!(value["retryable"], retryable);
            assert_eq!(output.exit_code, exit_code);
            assert!(!output.stderr.contains("private follow-up"));
        }
    }
}
