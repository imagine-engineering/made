use crate::ToolCallStatus;
use agent_client_protocol::schema::v1 as acp;
use chrono::{DateTime, SecondsFormat, Utc};
use nostr_compat::buzz_nips::agent_activity::{MAX_AGENT_PLAINTEXT_BYTES, ObserverTelemetry};
use serde_json::{Map, Value, json};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    error::Error,
    fmt,
};
use uuid::Uuid;

const OBSERVER_RETRY_CAPACITY: usize = 800;
const MAX_OBSERVER_IDENTIFIER_BYTES: usize = 4_096;
const MAX_ACP_METHOD_BYTES: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationObserverContext {
    pub channel_id: Uuid,
    pub session_id: acp::SessionId,
    pub agent_index: Option<u64>,
}

impl CollaborationObserverContext {
    pub fn new(
        channel_id: Uuid,
        session_id: acp::SessionId,
        agent_index: Option<u64>,
    ) -> Result<Self, CollaborationObserverError> {
        validate_identifier("session ID", session_id.0.as_ref())?;
        Ok(Self {
            channel_id,
            session_id,
            agent_index,
        })
    }
}

pub enum NativeCollaborationObserverEvent<'a> {
    TurnStarted,
    ProtocolRead(&'a Value),
    ProtocolWritten(&'a Value),
    ActionUpdated {
        action_id: &'a acp::ToolCallId,
        status: &'a ToolCallStatus,
    },
    SessionResolved(acp::StopReason),
}

pub enum CollaborationObserverPublish {
    Frame(ObserverTelemetry),
    Duplicate,
}

impl fmt::Debug for CollaborationObserverPublish {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Frame(frame) => formatter.debug_tuple("Frame").field(frame).finish(),
            Self::Duplicate => formatter.write_str("Duplicate"),
        }
    }
}

impl CollaborationObserverPublish {
    pub fn frame(&self) -> Option<&ObserverTelemetry> {
        match self {
            Self::Frame(frame) => Some(frame),
            Self::Duplicate => None,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum CollaborationObserverError {
    InvalidIdentifier(&'static str),
    InvalidProtocolFrame,
    InvalidActionTransition,
    TurnAlreadyResolved,
    SequenceExhausted,
    PayloadTooLarge,
}

impl fmt::Display for CollaborationObserverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier(field) => write!(formatter, "invalid {field}"),
            Self::InvalidProtocolFrame => formatter.write_str("invalid ACP protocol frame"),
            Self::InvalidActionTransition => {
                formatter.write_str("invalid ACP action status transition")
            }
            Self::TurnAlreadyResolved => formatter.write_str("ACP turn is already resolved"),
            Self::SequenceExhausted => formatter.write_str("observer sequence is exhausted"),
            Self::PayloadTooLarge => formatter.write_str("observer payload exceeds its limit"),
        }
    }
}

impl Error for CollaborationObserverError {}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActionState {
    Pending,
    WaitingForConfirmation,
    InProgress,
    Completed,
    Failed,
    Rejected,
    Cancelled,
}

impl ActionState {
    fn from_native(status: &ToolCallStatus) -> Self {
        match status {
            ToolCallStatus::Pending => Self::Pending,
            ToolCallStatus::WaitingForConfirmation { .. } => Self::WaitingForConfirmation,
            ToolCallStatus::InProgress => Self::InProgress,
            ToolCallStatus::Completed => Self::Completed,
            ToolCallStatus::Failed => Self::Failed,
            ToolCallStatus::Rejected => Self::Rejected,
            ToolCallStatus::Canceled => Self::Cancelled,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::WaitingForConfirmation => "waiting_for_confirmation",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
        }
    }

    fn rank(self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::WaitingForConfirmation => 1,
            Self::InProgress => 2,
            Self::Completed | Self::Failed | Self::Rejected | Self::Cancelled => 3,
        }
    }

    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Rejected | Self::Cancelled
        )
    }
}

pub struct CollaborationObserverAdapter {
    context: CollaborationObserverContext,
    next_sequence: u64,
    seen_source_ids: HashSet<String>,
    source_order: VecDeque<String>,
    resolved_turns: HashSet<String>,
    resolved_turn_order: VecDeque<String>,
    action_states: HashMap<(String, String), ActionState>,
    action_order: VecDeque<(String, String)>,
}

impl CollaborationObserverAdapter {
    pub fn new(context: CollaborationObserverContext) -> Self {
        Self {
            context,
            next_sequence: 1,
            seen_source_ids: HashSet::with_capacity(OBSERVER_RETRY_CAPACITY),
            source_order: VecDeque::with_capacity(OBSERVER_RETRY_CAPACITY),
            resolved_turns: HashSet::with_capacity(OBSERVER_RETRY_CAPACITY),
            resolved_turn_order: VecDeque::with_capacity(OBSERVER_RETRY_CAPACITY),
            action_states: HashMap::with_capacity(OBSERVER_RETRY_CAPACITY),
            action_order: VecDeque::with_capacity(OBSERVER_RETRY_CAPACITY),
        }
    }

    pub fn publish(
        &mut self,
        source_id: &str,
        turn_id: &str,
        timestamp: DateTime<Utc>,
        event: NativeCollaborationObserverEvent<'_>,
    ) -> Result<CollaborationObserverPublish, CollaborationObserverError> {
        validate_identifier("observer source ID", source_id)?;
        validate_identifier("turn ID", turn_id)?;
        if self.seen_source_ids.contains(source_id) {
            return Ok(CollaborationObserverPublish::Duplicate);
        }
        if self.resolved_turns.contains(turn_id) {
            return Err(CollaborationObserverError::TurnAlreadyResolved);
        }

        let resolves_turn = matches!(&event, NativeCollaborationObserverEvent::SessionResolved(_));
        let (kind, payload, action_state) = match event {
            NativeCollaborationObserverEvent::TurnStarted => {
                ("turn_started", json!({ "type": "turn_started" }), None)
            }
            NativeCollaborationObserverEvent::ProtocolRead(frame) => {
                ("acp_read", summarize_protocol_frame(frame)?, None)
            }
            NativeCollaborationObserverEvent::ProtocolWritten(frame) => {
                ("acp_write", summarize_protocol_frame(frame)?, None)
            }
            NativeCollaborationObserverEvent::ActionUpdated { action_id, status } => {
                validate_identifier("action ID", action_id.0.as_ref())?;
                let state = ActionState::from_native(status);
                let key = (turn_id.to_owned(), action_id.0.to_string());
                if let Some(previous) = self.action_states.get(&key).copied() {
                    if previous == state {
                        self.remember_source(source_id);
                        return Ok(CollaborationObserverPublish::Duplicate);
                    }
                    if previous.is_terminal() || state.rank() < previous.rank() {
                        return Err(CollaborationObserverError::InvalidActionTransition);
                    }
                }
                (
                    "acp_read",
                    json!({
                        "type": "tool_call_status",
                        "actionId": action_id.0.as_ref(),
                        "status": state.as_str(),
                    }),
                    Some((key, state)),
                )
            }
            NativeCollaborationObserverEvent::SessionResolved(stop_reason) => (
                "session_resolved",
                json!({
                    "type": "session_resolved",
                    "stopReason": stop_reason_name(stop_reason),
                }),
                None,
            ),
        };

        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(CollaborationObserverError::SequenceExhausted)?;
        let frame = ObserverTelemetry {
            seq: self.next_sequence,
            timestamp: timestamp.to_rfc3339_opts(SecondsFormat::Millis, true),
            kind: kind.to_owned(),
            agent_index: self.context.agent_index,
            channel_id: Some(self.context.channel_id.to_string()),
            session_id: Some(self.context.session_id.0.to_string()),
            turn_id: Some(turn_id.to_owned()),
            payload,
        };
        if serde_json::to_vec(&frame)
            .map_err(|_| CollaborationObserverError::PayloadTooLarge)?
            .len()
            > MAX_AGENT_PLAINTEXT_BYTES
        {
            return Err(CollaborationObserverError::PayloadTooLarge);
        }

        self.next_sequence = next_sequence;
        self.remember_source(source_id);
        if let Some((key, state)) = action_state {
            self.remember_action(key, state);
        }
        if resolves_turn {
            self.remember_resolved_turn(turn_id);
        }
        Ok(CollaborationObserverPublish::Frame(frame))
    }

    fn remember_source(&mut self, source_id: &str) {
        remember_bounded(
            &mut self.seen_source_ids,
            &mut self.source_order,
            source_id.to_owned(),
        );
    }

    fn remember_resolved_turn(&mut self, turn_id: &str) {
        remember_bounded(
            &mut self.resolved_turns,
            &mut self.resolved_turn_order,
            turn_id.to_owned(),
        );
    }

    fn remember_action(&mut self, key: (String, String), state: ActionState) {
        if !self.action_states.contains_key(&key) {
            if self.action_order.len() == OBSERVER_RETRY_CAPACITY
                && let Some(expired) = self.action_order.pop_front()
            {
                self.action_states.remove(&expired);
            }
            self.action_order.push_back(key.clone());
        }
        self.action_states.insert(key, state);
    }
}

fn summarize_protocol_frame(frame: &Value) -> Result<Value, CollaborationObserverError> {
    let object = frame
        .as_object()
        .ok_or(CollaborationObserverError::InvalidProtocolFrame)?;
    let mut summary = Map::new();
    summary.insert("type".to_owned(), Value::String("acp_frame".to_owned()));
    summary.insert("hasId".to_owned(), Value::Bool(object.contains_key("id")));
    summary.insert(
        "hasParams".to_owned(),
        Value::Bool(object.contains_key("params")),
    );
    summary.insert(
        "hasResult".to_owned(),
        Value::Bool(object.contains_key("result")),
    );
    summary.insert(
        "hasError".to_owned(),
        Value::Bool(object.contains_key("error")),
    );
    if let Some(method) = object.get("method").and_then(Value::as_str) {
        if method.is_empty()
            || method.len() > MAX_ACP_METHOD_BYTES
            || method.chars().any(char::is_control)
        {
            return Err(CollaborationObserverError::InvalidProtocolFrame);
        }
        summary.insert("method".to_owned(), Value::String(method.to_owned()));
    }
    Ok(Value::Object(summary))
}

fn stop_reason_name(reason: acp::StopReason) -> &'static str {
    match reason {
        acp::StopReason::EndTurn => "end_turn",
        acp::StopReason::MaxTokens => "max_tokens",
        acp::StopReason::MaxTurnRequests => "max_turn_requests",
        acp::StopReason::Refusal => "refusal",
        acp::StopReason::Cancelled => "cancelled",
        _ => "unknown",
    }
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), CollaborationObserverError> {
    if value.trim().is_empty()
        || value.len() > MAX_OBSERVER_IDENTIFIER_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(CollaborationObserverError::InvalidIdentifier(field));
    }
    Ok(())
}

fn remember_bounded(values: &mut HashSet<String>, order: &mut VecDeque<String>, value: String) {
    if values.insert(value.clone()) {
        if order.len() == OBSERVER_RETRY_CAPACITY
            && let Some(expired) = order.pop_front()
        {
            values.remove(&expired);
        }
        order.push_back(value);
    }
}
