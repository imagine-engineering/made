use std::fmt;

use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, MarkerView, Message, MessageContent,
    MessageDeleteMetadata, MessageRecordFields, NostrEventId, OperationId, PrincipalId,
    ReactionGroup, ReactionValue, ScheduledMessage, ScheduledMessageRecordFields,
    ScheduledMessageState, ThreadCursor, ThreadPage, ThreadReference,
};
use serde_json::{Value, json};

use super::contracts::{ErrorClass, error_contract};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MessagePageCursor {
    pub created_at: u64,
    pub source_event_id: NostrEventId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessagePage {
    pub messages: Vec<MessageRecordFields>,
    pub has_more: bool,
    pub next_cursor: Option<MessagePageCursor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageSearchQuery {
    pub community_id: CommunityId,
    pub query: Option<String>,
    pub author_principal_id: Option<PrincipalId>,
    pub since: Option<u64>,
    pub limit: usize,
    pub cursor: Option<MessagePageCursor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReactionRequest {
    Add {
        value: ReactionValue,
    },
    Remove {
        value: ReactionValue,
        added_event_id: NostrEventId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarkerRequest {
    Pin,
    Unpin { pinned_event_id: NostrEventId },
    Bookmark,
    Unbookmark { bookmarked_event_id: NostrEventId },
}

#[derive(Clone, Eq, PartialEq)]
pub enum MessagesCliCommand {
    Get {
        community_id: CommunityId,
        message_id: AggregateId,
    },
    List {
        community_id: CommunityId,
        channel_id: AggregateId,
        limit: usize,
        cursor: Option<MessagePageCursor>,
    },
    Thread {
        community_id: CommunityId,
        channel_id: AggregateId,
        root_event_id: NostrEventId,
        limit: usize,
        depth_limit: Option<u16>,
        cursor: Option<ThreadCursor>,
    },
    Search(MessageSearchQuery),
    Send {
        community_id: CommunityId,
        channel_id: AggregateId,
        content: MessageContent,
        reply: ThreadReference,
        broadcast: bool,
        operation_id: OperationId,
    },
    Edit {
        community_id: CommunityId,
        channel_id: AggregateId,
        message_id: AggregateId,
        expected_version: AggregateVersion,
        content: MessageContent,
        operation_id: OperationId,
    },
    Delete {
        community_id: CommunityId,
        channel_id: AggregateId,
        message_id: AggregateId,
        expected_version: AggregateVersion,
        metadata: Option<MessageDeleteMetadata>,
        operation_id: OperationId,
    },
    Reactions {
        community_id: CommunityId,
        channel_id: AggregateId,
        message_id: AggregateId,
    },
    React {
        community_id: CommunityId,
        channel_id: AggregateId,
        message_id: AggregateId,
        expected_version: AggregateVersion,
        request: ReactionRequest,
        operation_id: OperationId,
    },
    Markers {
        community_id: CommunityId,
        channel_id: AggregateId,
        message_id: AggregateId,
    },
    SetMarker {
        community_id: CommunityId,
        channel_id: AggregateId,
        message_id: AggregateId,
        expected_version: AggregateVersion,
        request: MarkerRequest,
        operation_id: OperationId,
    },
    ListSchedules {
        community_id: CommunityId,
        channel_id: AggregateId,
    },
    Schedule {
        community_id: CommunityId,
        channel_id: AggregateId,
        content: MessageContent,
        scheduled_for_millis: u64,
        operation_id: OperationId,
    },
    UpdateSchedule {
        community_id: CommunityId,
        channel_id: AggregateId,
        schedule_id: AggregateId,
        expected_version: AggregateVersion,
        content: MessageContent,
        scheduled_for_millis: u64,
        operation_id: OperationId,
    },
    CancelSchedule {
        community_id: CommunityId,
        channel_id: AggregateId,
        schedule_id: AggregateId,
        expected_version: AggregateVersion,
        operation_id: OperationId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MessagesCliVerb {
    Get,
    List,
    Thread,
    Search,
    Send,
    Edit,
    Delete,
    Reactions,
    React,
    Markers,
    SetMarker,
    ListSchedules,
    Schedule,
    UpdateSchedule,
    CancelSchedule,
}

impl MessagesCliVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "message.get",
            Self::List => "message.list",
            Self::Thread => "message.thread",
            Self::Search => "message.search",
            Self::Send => "message.send",
            Self::Edit => "message.edit",
            Self::Delete => "message.delete",
            Self::Reactions => "reaction.list",
            Self::React => "reaction.set",
            Self::Markers => "marker.get",
            Self::SetMarker => "marker.set",
            Self::ListSchedules => "schedule.list",
            Self::Schedule => "schedule.create",
            Self::UpdateSchedule => "schedule.update",
            Self::CancelSchedule => "schedule.cancel",
        }
    }
}

impl MessagesCliCommand {
    const fn verb(&self) -> MessagesCliVerb {
        match self {
            Self::Get { .. } => MessagesCliVerb::Get,
            Self::List { .. } => MessagesCliVerb::List,
            Self::Thread { .. } => MessagesCliVerb::Thread,
            Self::Search(_) => MessagesCliVerb::Search,
            Self::Send { .. } => MessagesCliVerb::Send,
            Self::Edit { .. } => MessagesCliVerb::Edit,
            Self::Delete { .. } => MessagesCliVerb::Delete,
            Self::Reactions { .. } => MessagesCliVerb::Reactions,
            Self::React { .. } => MessagesCliVerb::React,
            Self::Markers { .. } => MessagesCliVerb::Markers,
            Self::SetMarker { .. } => MessagesCliVerb::SetMarker,
            Self::ListSchedules { .. } => MessagesCliVerb::ListSchedules,
            Self::Schedule { .. } => MessagesCliVerb::Schedule,
            Self::UpdateSchedule { .. } => MessagesCliVerb::UpdateSchedule,
            Self::CancelSchedule { .. } => MessagesCliVerb::CancelSchedule,
        }
    }
}

impl fmt::Debug for MessagesCliCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MessagesCliCommand")
            .field("verb", &self.verb().as_str())
            .field("payload", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MessageWriteReceipt {
    pub operation_id: OperationId,
    pub resource_id: AggregateId,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessagesCliOutcome {
    Message(MessageRecordFields),
    Page(MessagePage),
    Thread(ThreadPage),
    Reactions(Vec<ReactionGroup>),
    Markers(MarkerView),
    Schedules(Vec<ScheduledMessageRecordFields>),
    Applied(MessageWriteReceipt),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessagesCliError {
    InvalidRequest,
    NotFound,
    Unavailable,
    AuthorizationDenied,
    PartialFailure,
    Unexpected,
    Conflict,
}

impl MessagesCliError {
    const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::InvalidRequest => "messages_cli_invalid_request",
            Self::NotFound => "messages_cli_not_found",
            Self::Unavailable => "messages_cli_unavailable",
            Self::AuthorizationDenied => "messages_cli_authorization_denied",
            Self::PartialFailure => "messages_cli_completion_unknown",
            Self::Unexpected => "messages_cli_unexpected_response",
            Self::Conflict => "messages_cli_stale_version",
        }
    }

    const fn common_class(self) -> ErrorClass {
        match self {
            Self::InvalidRequest => ErrorClass::Usage,
            Self::NotFound => ErrorClass::NotFound,
            Self::Unavailable => ErrorClass::Network { retryable: true },
            Self::AuthorizationDenied => ErrorClass::Authorization,
            Self::PartialFailure => ErrorClass::DeliveryUnknown,
            Self::Unexpected => ErrorClass::Unexpected,
            Self::Conflict => ErrorClass::Conflict,
        }
    }
}

pub trait MessagesCliExecutor {
    fn execute(&self, command: MessagesCliCommand) -> Result<MessagesCliOutcome, MessagesCliError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessagesCliExecution {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub fn execute_messages_command(
    executor: &impl MessagesCliExecutor,
    command: MessagesCliCommand,
) -> MessagesCliExecution {
    let verb = command.verb();
    match executor.execute(command) {
        Ok(outcome) => match success_output(verb, outcome) {
            Some(output) => MessagesCliExecution {
                stdout: format!("{output}\n"),
                stderr: String::new(),
                exit_code: 0,
            },
            None => error_output(verb, MessagesCliError::Unexpected),
        },
        Err(error) => error_output(verb, error),
    }
}

fn error_output(verb: MessagesCliVerb, error: MessagesCliError) -> MessagesCliExecution {
    let contract = error_contract(error.common_class());
    let diagnostic = error.diagnostic_code();
    MessagesCliExecution {
        stdout: String::new(),
        stderr: format!(
            "{}\n",
            json!({
                "command": verb.as_str(),
                "error": contract.category,
                "error_code": diagnostic,
                "message": diagnostic,
                "ok": false,
                "retryable": contract.retryable,
            })
        ),
        exit_code: contract.exit_class as i32,
    }
}

fn success_output(verb: MessagesCliVerb, outcome: MessagesCliOutcome) -> Option<Value> {
    match (verb, outcome) {
        (MessagesCliVerb::Get, MessagesCliOutcome::Message(message)) => {
            message_output(&message, verb)
        }
        (MessagesCliVerb::List | MessagesCliVerb::Search, MessagesCliOutcome::Page(page)) => {
            page_output(verb, page)
        }
        (MessagesCliVerb::Thread, MessagesCliOutcome::Thread(page)) => {
            Some(thread_output(verb, page))
        }
        (MessagesCliVerb::Reactions, MessagesCliOutcome::Reactions(groups)) => {
            Some(reactions_output(verb, &groups))
        }
        (MessagesCliVerb::Markers, MessagesCliOutcome::Markers(markers)) => Some(json!({
            "bookmarked": markers.bookmarked,
            "command": verb.as_str(),
            "ok": true,
            "pinned": markers.pinned,
        })),
        (MessagesCliVerb::ListSchedules, MessagesCliOutcome::Schedules(schedules)) => {
            schedules_output(verb, schedules)
        }
        (
            MessagesCliVerb::Send
            | MessagesCliVerb::Edit
            | MessagesCliVerb::Delete
            | MessagesCliVerb::React
            | MessagesCliVerb::SetMarker
            | MessagesCliVerb::Schedule
            | MessagesCliVerb::UpdateSchedule
            | MessagesCliVerb::CancelSchedule,
            MessagesCliOutcome::Applied(receipt),
        ) => Some(write_output(verb, receipt)),
        _ => None,
    }
}

fn message_output(fields: &MessageRecordFields, verb: MessagesCliVerb) -> Option<Value> {
    let message = Message::from_record(fields.clone()).ok()?;
    Some(json!({
        "author_principal_id": fields.author.principal_id(),
        "channel_id": fields.channel_id,
        "command": verb.as_str(),
        "community_id": fields.community_id,
        "content": message.visible_content().map(MessageContent::as_str),
        "created_at": fields.source.event_created_at,
        "lifecycle": message_lifecycle(fields),
        "message_id": fields.message_id,
        "ok": true,
        "owner_principal_id": fields.author.owner_principal_id(),
        "source_event_id": hex_bytes(fields.source.event_id.as_bytes()),
        "version": fields.version,
    }))
}

fn message_lifecycle(fields: &MessageRecordFields) -> &'static str {
    use collaboration_domain::MessageLifecycleState;
    match fields.lifecycle_state {
        MessageLifecycleState::Active => "active",
        MessageLifecycleState::Edited => "edited",
        MessageLifecycleState::Deleted => "deleted",
    }
}

fn page_output(verb: MessagesCliVerb, page: MessagePage) -> Option<Value> {
    let messages = page
        .messages
        .iter()
        .map(|message| message_output(message, MessagesCliVerb::Get))
        .collect::<Option<Vec<_>>>()?;
    Some(json!({
        "command": verb.as_str(),
        "has_more": page.has_more,
        "messages": messages,
        "next_cursor": page.next_cursor.map(message_cursor_output),
        "ok": true,
    }))
}

fn message_cursor_output(cursor: MessagePageCursor) -> Value {
    json!({
        "created_at": cursor.created_at,
        "source_event_id": hex_bytes(cursor.source_event_id.as_bytes()),
    })
}

fn thread_output(verb: MessagesCliVerb, page: ThreadPage) -> Value {
    json!({
        "command": verb.as_str(),
        "has_more": page.has_more,
        "next_cursor": page.next_cursor.map(thread_cursor_output),
        "ok": true,
        "replies": page.replies.iter().map(|node| json!({
            "author_public_key": hex_bytes(node.event.author.as_bytes()),
            "channel_id": node.event.channel_id,
            "created_at": node.event.created_at,
            "depth": node.depth,
            "event_id": hex_bytes(node.event.event_id.as_bytes()),
            "parent_event_id": node.parent_event_id.map(|event_id| hex_bytes(event_id.as_bytes())),
            "root_event_id": hex_bytes(node.root_event_id.as_bytes()),
        })).collect::<Vec<_>>(),
    })
}

fn thread_cursor_output(cursor: ThreadCursor) -> Value {
    json!({
        "created_at": cursor.created_at,
        "event_id": hex_bytes(cursor.event_id.as_bytes()),
    })
}

fn reactions_output(verb: MessagesCliVerb, groups: &[ReactionGroup]) -> Value {
    json!({
        "command": verb.as_str(),
        "groups": groups.iter().map(|group| json!({
            "count": group.count(),
            "reactions": group.reactions.iter().map(|reaction| json!({
                "actor_principal_id": reaction.actor_principal_id,
                "event_id": hex_bytes(reaction.added_source.event_id.as_bytes()),
            })).collect::<Vec<_>>(),
            "value": group.value.as_str(),
        })).collect::<Vec<_>>(),
        "ok": true,
    })
}

fn schedules_output(
    verb: MessagesCliVerb,
    schedules: Vec<ScheduledMessageRecordFields>,
) -> Option<Value> {
    let schedules = schedules
        .into_iter()
        .map(|fields| {
            let schedule = ScheduledMessage::from_record(fields).ok()?;
            let fields = schedule.fields();
            Some(json!({
                "channel_id": fields.channel_id,
                "content": fields.content.as_str(),
                "schedule_id": fields.schedule_id,
                "scheduled_for_millis": fields.scheduled_for_millis,
                "state": schedule_state(fields.state),
                "version": fields.version,
            }))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(json!({
        "command": verb.as_str(),
        "ok": true,
        "schedules": schedules,
    }))
}

fn schedule_state(state: ScheduledMessageState) -> Value {
    match state {
        ScheduledMessageState::Pending => json!({ "kind": "pending" }),
        ScheduledMessageState::Claimed(claim) => json!({
            "attempt": claim.attempt,
            "claim_id": claim.claim_id,
            "kind": "claimed",
            "lease_expires_at_millis": claim.lease_expires_at_millis,
        }),
        ScheduledMessageState::Cancelled => json!({ "kind": "cancelled" }),
        ScheduledMessageState::Executed {
            claim_id,
            execution_attempt,
            published_message_id,
            published_event_id,
            executed_at_millis,
        } => json!({
            "claim_id": claim_id,
            "executed_at_millis": executed_at_millis,
            "execution_attempt": execution_attempt,
            "kind": "executed",
            "published_event_id": hex_bytes(published_event_id.as_bytes()),
            "published_message_id": published_message_id,
        }),
    }
}

fn write_output(verb: MessagesCliVerb, receipt: MessageWriteReceipt) -> Value {
    json!({
        "command": verb.as_str(),
        "ok": true,
        "operation_id": receipt.operation_id,
        "resource_id": receipt.resource_id,
        "version": receipt.version,
    })
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
    use std::cell::RefCell;

    use collaboration_domain::{
        ActiveReaction, MessageAuthor, MessageLifecycleState, MessageSource, ReactionGroup,
        ScheduledMessageRecordFields, ThreadEvent, ThreadNode,
    };
    use uuid::Uuid;

    use super::*;

    struct TestExecutor {
        command: RefCell<Option<MessagesCliCommand>>,
        result: RefCell<Option<Result<MessagesCliOutcome, MessagesCliError>>>,
    }

    impl TestExecutor {
        fn returning(result: Result<MessagesCliOutcome, MessagesCliError>) -> Self {
            Self {
                command: RefCell::new(None),
                result: RefCell::new(Some(result)),
            }
        }
    }

    impl MessagesCliExecutor for TestExecutor {
        fn execute(
            &self,
            command: MessagesCliCommand,
        ) -> Result<MessagesCliOutcome, MessagesCliError> {
            self.command.replace(Some(command));
            self.result
                .borrow_mut()
                .take()
                .expect("the test executor is called once")
        }
    }

    fn identifier(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn principal_id() -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(2))
    }

    fn operation_id() -> OperationId {
        OperationId::from_uuid(Uuid::from_u128(3))
    }

    fn event(byte: u8) -> NostrEventId {
        NostrEventId::from_bytes([byte; 32])
    }

    fn message(message_id: AggregateId, content: &str, created_at: u64) -> MessageRecordFields {
        let source = MessageSource {
            event_id: event(message_id.as_uuid().as_u128() as u8),
            event_created_at: created_at,
        };
        MessageRecordFields {
            community_id: community_id(),
            channel_id: identifier(4),
            message_id,
            author: MessageAuthor::principal(principal_id()),
            content: MessageContent::new(content).expect("valid content"),
            lifecycle_state: MessageLifecycleState::Active,
            source,
            current_source: source,
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        }
    }

    #[test]
    fn message_page_preserves_stable_cursor_and_content() {
        let next_cursor = MessagePageCursor {
            created_at: 20,
            source_event_id: event(6),
        };
        let output = execute_messages_command(
            &TestExecutor::returning(Ok(MessagesCliOutcome::Page(MessagePage {
                messages: vec![message(identifier(5), "hello", 10)],
                has_more: true,
                next_cursor: Some(next_cursor),
            }))),
            MessagesCliCommand::List {
                community_id: community_id(),
                channel_id: identifier(4),
                limit: 50,
                cursor: None,
            },
        );
        assert_eq!(output.exit_code, 0);
        let value: Value = serde_json::from_str(&output.stdout).expect("JSON page");
        assert_eq!(value["command"], "message.list");
        assert_eq!(value["messages"][0]["content"], "hello");
        assert_eq!(value["next_cursor"]["created_at"], 20);
        assert_eq!(value["next_cursor"]["source_event_id"], "06".repeat(32));
    }

    #[test]
    fn reply_edit_and_delete_forward_canonical_attribution() {
        let reply = MessagesCliCommand::Send {
            community_id: community_id(),
            channel_id: identifier(4),
            content: MessageContent::new("reply").expect("valid content"),
            reply: ThreadReference::Reply {
                parent_event_id: event(7),
                root_event_id: Some(event(8)),
            },
            broadcast: false,
            operation_id: operation_id(),
        };
        let executor =
            TestExecutor::returning(Ok(MessagesCliOutcome::Applied(MessageWriteReceipt {
                operation_id: operation_id(),
                resource_id: identifier(9),
                version: AggregateVersion::FIRST,
            })));
        let output = execute_messages_command(&executor, reply.clone());
        assert_eq!(executor.command.take(), Some(reply));
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("\"command\":\"message.send\""));

        let edit = MessagesCliCommand::Edit {
            community_id: community_id(),
            channel_id: identifier(4),
            message_id: identifier(9),
            expected_version: AggregateVersion::FIRST,
            content: MessageContent::new("updated").expect("valid content"),
            operation_id: operation_id(),
        };
        assert_eq!(edit.verb().as_str(), "message.edit");
        let delete = MessagesCliCommand::Delete {
            community_id: community_id(),
            channel_id: identifier(4),
            message_id: identifier(9),
            expected_version: AggregateVersion::FIRST,
            metadata: Some(
                MessageDeleteMetadata::new(None, Some("spam".into()), None)
                    .expect("valid deletion metadata"),
            ),
            operation_id: operation_id(),
        };
        assert_eq!(delete.verb().as_str(), "message.delete");
        assert!(!format!("{delete:?}").contains("spam"));
    }

    #[test]
    fn thread_reactions_and_markers_have_golden_read_shapes() {
        let thread = ThreadPage {
            replies: vec![ThreadNode {
                event: ThreadEvent {
                    event_id: event(10),
                    channel_id: identifier(4),
                    author: collaboration_domain::NostrPublicKey::from_bytes([11; 32]),
                    created_at: 12,
                    reference: ThreadReference::Reply {
                        parent_event_id: event(8),
                        root_event_id: Some(event(8)),
                    },
                    broadcast: false,
                    deleted: false,
                },
                parent_event_id: Some(event(8)),
                root_event_id: event(8),
                depth: 1,
            }],
            has_more: false,
            next_cursor: None,
        };
        let thread_output = execute_messages_command(
            &TestExecutor::returning(Ok(MessagesCliOutcome::Thread(thread))),
            MessagesCliCommand::Thread {
                community_id: community_id(),
                channel_id: identifier(4),
                root_event_id: event(8),
                limit: 20,
                depth_limit: Some(5),
                cursor: None,
            },
        );
        assert!(thread_output.stdout.contains("\"depth\":1"));

        let groups = vec![ReactionGroup {
            value: ReactionValue::new("👍").expect("valid reaction"),
            reactions: vec![ActiveReaction {
                actor_principal_id: principal_id(),
                added_source: MessageSource {
                    event_id: event(12),
                    event_created_at: 13,
                },
            }],
        }];
        let reactions = execute_messages_command(
            &TestExecutor::returning(Ok(MessagesCliOutcome::Reactions(groups))),
            MessagesCliCommand::Reactions {
                community_id: community_id(),
                channel_id: identifier(4),
                message_id: identifier(9),
            },
        );
        assert!(reactions.stdout.contains("\"count\":1"));
        assert!(reactions.stdout.contains("👍"));

        let markers = execute_messages_command(
            &TestExecutor::returning(Ok(MessagesCliOutcome::Markers(MarkerView {
                pinned: true,
                bookmarked: false,
            }))),
            MessagesCliCommand::Markers {
                community_id: community_id(),
                channel_id: identifier(4),
                message_id: identifier(9),
            },
        );
        assert!(markers.stdout.contains("\"pinned\":true"));
        assert!(markers.stdout.contains("\"bookmarked\":false"));
    }

    #[test]
    fn scheduled_messages_preserve_due_state_and_mutation_versions() {
        let schedule = ScheduledMessageRecordFields {
            community_id: community_id(),
            channel_id: identifier(4),
            schedule_id: identifier(13),
            author: MessageAuthor::principal(principal_id()),
            initial_content: MessageContent::new("later").expect("valid content"),
            initial_scheduled_for_millis: 1_000_000,
            content: MessageContent::new("later").expect("valid content"),
            scheduled_for_millis: 1_000_000,
            source: MessageSource {
                event_id: event(14),
                event_created_at: 1,
            },
            authored_mutations: Vec::new(),
            state: ScheduledMessageState::Pending,
            version: AggregateVersion::FIRST,
        };
        let output = execute_messages_command(
            &TestExecutor::returning(Ok(MessagesCliOutcome::Schedules(vec![schedule]))),
            MessagesCliCommand::ListSchedules {
                community_id: community_id(),
                channel_id: identifier(4),
            },
        );
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.contains("\"state\":{\"kind\":\"pending\"}"));
        assert!(output.stdout.contains("\"scheduled_for_millis\":1000000"));

        let cancel = MessagesCliCommand::CancelSchedule {
            community_id: community_id(),
            channel_id: identifier(4),
            schedule_id: identifier(13),
            expected_version: AggregateVersion::FIRST,
            operation_id: operation_id(),
        };
        assert_eq!(cancel.verb().as_str(), "schedule.cancel");
    }

    #[test]
    fn stable_errors_and_mismatched_outcomes_fail_closed() {
        let cases = [
            (MessagesCliError::InvalidRequest, "user_error", 1, false),
            (MessagesCliError::NotFound, "not_found", 1, false),
            (MessagesCliError::Unavailable, "network_error", 2, true),
            (
                MessagesCliError::PartialFailure,
                "delivery_unknown",
                2,
                false,
            ),
            (
                MessagesCliError::AuthorizationDenied,
                "auth_error",
                3,
                false,
            ),
            (MessagesCliError::Unexpected, "error", 4, false),
            (MessagesCliError::Conflict, "conflict", 5, false),
        ];
        for (error, category, exit_code, retryable) in cases {
            let output = execute_messages_command(
                &TestExecutor::returning(Err(error)),
                MessagesCliCommand::Get {
                    community_id: community_id(),
                    message_id: identifier(9),
                },
            );
            assert_eq!(output.exit_code, exit_code);
            assert!(output.stdout.is_empty());
            let envelope: Value =
                serde_json::from_str(&output.stderr).expect("JSON error envelope");
            assert_eq!(envelope["error"], category);
            assert_eq!(envelope["retryable"], retryable);
        }

        let mismatch = execute_messages_command(
            &TestExecutor::returning(Ok(MessagesCliOutcome::Markers(MarkerView {
                pinned: false,
                bookmarked: false,
            }))),
            MessagesCliCommand::Get {
                community_id: community_id(),
                message_id: identifier(9),
            },
        );
        assert_eq!(mismatch.exit_code, 4);
        assert!(mismatch.stderr.contains("messages_cli_unexpected_response"));
    }
}
