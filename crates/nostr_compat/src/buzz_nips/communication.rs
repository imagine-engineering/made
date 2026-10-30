use crate::generated_kinds::{
    KIND_DM_VISIBILITY, KIND_EVENT_REMINDER, KIND_READ_STATE, KIND_THREAD_SUMMARY,
    KIND_WINDOW_BOUNDS,
};
use crate::{EventId, PublicKey, SignedEvent, TimestampPolicy, verify_signed_event};
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

const DEFAULT_WINDOW_LIMIT: u16 = 50;
const MAX_WINDOW_LIMIT: u16 = 200;
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_CONTEXTS: usize = 10_000;
const MAX_CONTEXT_BYTES: usize = 256;
const MAX_CLIENT_ID_CHARACTERS: usize = 64;
const MAX_NIP44_CIPHERTEXT_BYTES: usize = 87_472;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CommunicationCodecError {
    #[error("unsupported communication event kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid communication filter: {0}")]
    InvalidFilter(String),
    #[error("invalid communication envelope: {0}")]
    InvalidEnvelope(String),
    #[error("invalid communication payload: {0}")]
    InvalidPayload(String),
    #[error("manual-unread counter is exhausted")]
    CounterExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowCursor {
    pub created_at: u64,
    pub id: EventId,
}

impl WindowCursor {
    fn binding(self) -> String {
        format!("{}:{}", self.created_at, self.id)
    }

    fn from_value(value: &Value) -> Result<Self, CommunicationCodecError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid_payload("next_cursor must be an object"))?;
        let created_at = object
            .get("created_at")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid_payload("next_cursor.created_at must be an integer"))?;
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_payload("next_cursor.id must be a string"))?;
        Ok(Self {
            created_at,
            id: parse_event_id(id, "next_cursor.id")?,
        })
    }

    fn to_value(self) -> Value {
        json!({"created_at": self.created_at, "id": self.id.to_hex()})
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelWindowRequest {
    pub channel_id: String,
    pub kinds: Option<Vec<u16>>,
    pub limit: u16,
    pub cursor: Option<WindowCursor>,
    pub include_summaries: bool,
    pub include_aux: bool,
}

impl ChannelWindowRequest {
    pub fn parse_filter(value: &Value) -> Result<Option<Self>, CommunicationCodecError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid_filter("window filter must be an object"))?;
        if object.get("top_level") != Some(&Value::Bool(true)) {
            return Ok(None);
        }
        let channels = object
            .get("#h")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_filter("top_level requires exactly one #h channel"))?;
        if channels.len() != 1 {
            return Err(invalid_filter("top_level requires exactly one #h channel"));
        }
        let channel_id = channels[0]
            .as_str()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_filter("#h channel must be a non-empty string"))?
            .to_owned();
        let until = match object.get("until") {
            Some(value) => Some(
                value
                    .as_u64()
                    .ok_or_else(|| invalid_filter("until must be a non-negative integer"))?,
            ),
            None => None,
        };
        let before_id = match object.get("before_id") {
            Some(value) => Some(parse_event_id(
                value
                    .as_str()
                    .ok_or_else(|| invalid_filter("before_id must be a string"))?,
                "before_id",
            )?),
            None => None,
        };
        let cursor = match (until, before_id) {
            (None, None) => None,
            (Some(created_at), Some(id)) => Some(WindowCursor { created_at, id }),
            _ => {
                return Err(invalid_filter(
                    "until and before_id must both be present or both be absent",
                ));
            }
        };
        let limit = match object.get("limit") {
            None => DEFAULT_WINDOW_LIMIT,
            Some(value) => {
                let value = value
                    .as_u64()
                    .ok_or_else(|| invalid_filter("limit must be a positive integer"))?;
                if value == 0 {
                    return Err(invalid_filter("limit must be at least one"));
                }
                u16::try_from(value.min(u64::from(MAX_WINDOW_LIMIT)))
                    .map_err(|_| invalid_filter("limit is out of range"))?
            }
        };
        let kinds = match object.get("kinds") {
            None => None,
            Some(value) => {
                let values = value
                    .as_array()
                    .ok_or_else(|| invalid_filter("kinds must be an array"))?;
                let mut kinds = Vec::with_capacity(values.len());
                for value in values {
                    let kind = value
                        .as_u64()
                        .and_then(|value| u16::try_from(value).ok())
                        .ok_or_else(|| invalid_filter("kind must fit in u16"))?;
                    kinds.push(kind);
                }
                Some(kinds)
            }
        };
        Ok(Some(Self {
            channel_id,
            kinds,
            limit,
            cursor,
            include_summaries: object.get("include_summaries") == Some(&Value::Bool(true)),
            include_aux: object.get("include_aux") == Some(&Value::Bool(true)),
        }))
    }

    pub fn to_filter(&self) -> Value {
        let mut object = Map::new();
        object.insert("#h".into(), json!([self.channel_id]));
        object.insert("limit".into(), json!(self.limit));
        object.insert("top_level".into(), Value::Bool(true));
        if self.include_summaries {
            object.insert("include_summaries".into(), Value::Bool(true));
        }
        if self.include_aux {
            object.insert("include_aux".into(), Value::Bool(true));
        }
        if let Some(kinds) = &self.kinds {
            object.insert("kinds".into(), json!(kinds));
        }
        if let Some(cursor) = self.cursor {
            object.insert("until".into(), json!(cursor.created_at));
            object.insert("before_id".into(), json!(cursor.id.to_hex()));
        }
        Value::Object(object)
    }

    pub fn bounds_d_tag(&self) -> String {
        let suffix = self
            .cursor
            .map_or_else(|| "head".into(), WindowCursor::binding);
        format!("{}:{suffix}", self.channel_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadSummary {
    pub row_event_id: EventId,
    pub channel_id: String,
    pub reply_count: u64,
    pub descendant_count: u64,
    pub last_reply_at: Option<u64>,
    pub participants: Vec<PublicKey>,
}

impl ThreadSummary {
    pub fn parse_signed_event(
        event: &SignedEvent,
        relay: PublicKey,
    ) -> Result<Self, CommunicationCodecError> {
        verify_relay_event(event, relay, KIND_THREAD_SUMMARY)?;
        ensure_exact_tag_names(&event.event.tags, &["e", "d", "h"])?;
        let event_tag = parse_single_text_tag(&event.event.tags, "e", false)?;
        let d_tag = parse_single_text_tag(&event.event.tags, "d", false)?;
        if event_tag != d_tag {
            return Err(invalid_envelope("thread-summary e and d tags differ"));
        }
        let row_event_id = parse_event_id(&event_tag, "thread-summary event")?;
        let channel_id = parse_single_text_tag(&event.event.tags, "h", false)?;
        let content = parse_strict_json(event.event.content.as_bytes())?;
        let object = content
            .as_object()
            .ok_or_else(|| invalid_payload("thread-summary content must be an object"))?;
        let reply_count = required_u64(object, "reply_count")?;
        let descendant_count = required_u64(object, "descendant_count")?;
        let last_reply_at = match object.get("last_reply_at") {
            Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .ok_or_else(|| invalid_payload("last_reply_at must be integer or null"))?,
            ),
            None => return Err(invalid_payload("last_reply_at is required")),
        };
        let participant_values = object
            .get("participants")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_payload("participants must be an array"))?;
        if participant_values.len() > 10 {
            return Err(invalid_payload("participants exceeds ten entries"));
        }
        let mut participants = Vec::with_capacity(participant_values.len());
        let mut seen = BTreeSet::new();
        for value in participant_values {
            let participant = PublicKey::from_hex(
                value
                    .as_str()
                    .ok_or_else(|| invalid_payload("participant must be a string"))?,
            )
            .map_err(|error| invalid_payload(format!("invalid participant: {error}")))?;
            if !seen.insert(participant) {
                return Err(invalid_payload("duplicate thread participant"));
            }
            participants.push(participant);
        }
        Ok(Self {
            row_event_id,
            channel_id,
            reply_count,
            descendant_count,
            last_reply_at,
            participants,
        })
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        let event_id = self.row_event_id.to_hex();
        vec![
            vec!["e".into(), event_id.clone()],
            vec!["d".into(), event_id],
            vec!["h".into(), self.channel_id.clone()],
        ]
    }

    pub fn content_json(&self) -> String {
        json!({
            "reply_count": self.reply_count,
            "descendant_count": self.descendant_count,
            "last_reply_at": self.last_reply_at,
            "participants": self.participants.iter().map(|key| key.to_hex()).collect::<Vec<_>>(),
        })
        .to_string()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowBounds {
    pub channel_id: String,
    pub request_cursor: Option<WindowCursor>,
    pub has_more: bool,
    pub next_cursor: Option<WindowCursor>,
}

impl WindowBounds {
    pub fn parse_signed_event(
        event: &SignedEvent,
        relay: PublicKey,
        request: &ChannelWindowRequest,
    ) -> Result<Self, CommunicationCodecError> {
        verify_relay_event(event, relay, KIND_WINDOW_BOUNDS)?;
        ensure_exact_tag_names(&event.event.tags, &["d", "h"])?;
        let channel_id = parse_single_text_tag(&event.event.tags, "h", false)?;
        if channel_id != request.channel_id {
            return Err(invalid_envelope(
                "window-bounds channel does not match request",
            ));
        }
        let d_tag = parse_single_text_tag(&event.event.tags, "d", false)?;
        if d_tag != request.bounds_d_tag() {
            return Err(invalid_envelope("window-bounds request binding mismatch"));
        }
        let content = parse_strict_json(event.event.content.as_bytes())?;
        let object = content
            .as_object()
            .ok_or_else(|| invalid_payload("window-bounds content must be an object"))?;
        let has_more = object
            .get("has_more")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid_payload("has_more must be boolean"))?;
        let next_cursor = match object.get("next_cursor") {
            Some(Value::Null) => None,
            Some(value) => Some(WindowCursor::from_value(value)?),
            None => return Err(invalid_payload("next_cursor is required")),
        };
        if has_more != next_cursor.is_some() {
            return Err(invalid_payload(
                "has_more must be true exactly when next_cursor is present",
            ));
        }
        Ok(Self {
            channel_id,
            request_cursor: request.cursor,
            has_more,
            next_cursor,
        })
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        let suffix = self
            .request_cursor
            .map_or_else(|| "head".into(), WindowCursor::binding);
        vec![
            vec!["d".into(), format!("{}:{suffix}", self.channel_id)],
            vec!["h".into(), self.channel_id.clone()],
        ]
    }

    pub fn content_json(&self) -> String {
        json!({
            "has_more": self.has_more,
            "next_cursor": self.next_cursor.map(WindowCursor::to_value),
        })
        .to_string()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmVisibilitySnapshot {
    pub viewer: PublicKey,
    pub hidden_channels: BTreeSet<String>,
}

impl DmVisibilitySnapshot {
    pub fn parse_signed_event(
        event: &SignedEvent,
        relay: PublicKey,
        authenticated_reader: PublicKey,
    ) -> Result<Self, CommunicationCodecError> {
        verify_relay_event(event, relay, KIND_DM_VISIBILITY)?;
        if !event.event.content.is_empty() {
            return Err(invalid_envelope("DM visibility content must be empty"));
        }
        let viewer = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "d", false)?)
            .map_err(|error| invalid_envelope(format!("invalid DM visibility viewer: {error}")))?;
        let recipient = PublicKey::from_hex(&parse_single_text_tag(&event.event.tags, "p", false)?)
            .map_err(|error| {
                invalid_envelope(format!("invalid DM visibility recipient: {error}"))
            })?;
        if viewer != recipient || viewer != authenticated_reader {
            return Err(invalid_envelope(
                "DM visibility is readable only by its matching viewer",
            ));
        }
        let mut hidden_channels = BTreeSet::new();
        for tag in &event.event.tags {
            let name = tag.first().map(String::as_str).unwrap_or_default();
            if !matches!(name, "d" | "p" | "h") {
                return Err(invalid_envelope(format!(
                    "unexpected DM visibility tag {name:?}"
                )));
            }
            if name == "h" {
                if tag.len() != 2 || tag[1].is_empty() {
                    return Err(invalid_envelope(
                        "DM visibility h tag must have one non-empty value",
                    ));
                }
                hidden_channels.insert(tag[1].clone());
            }
        }
        Ok(Self {
            viewer,
            hidden_channels,
        })
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        let viewer = self.viewer.to_hex();
        let mut tags = vec![vec!["d".into(), viewer.clone()], vec!["p".into(), viewer]];
        tags.extend(
            self.hidden_channels
                .iter()
                .map(|channel| vec!["h".into(), channel.clone()]),
        );
        tags
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReminderStatus {
    Pending,
    Done,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReminderTarget {
    pub event_id: Option<EventId>,
    pub address: Option<String>,
    pub relays: Vec<String>,
    pub preview: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReminderBody {
    pub target: Option<ReminderTarget>,
    pub status: ReminderStatus,
    pub note: Option<String>,
}

impl ReminderBody {
    pub fn parse(plaintext: &[u8]) -> Result<Self, CommunicationCodecError> {
        let value = parse_strict_json(plaintext)?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_payload("reminder plaintext must be an object"))?;
        let status = match object.get("status").and_then(Value::as_str) {
            Some("pending") => ReminderStatus::Pending,
            Some("done") => ReminderStatus::Done,
            Some("cancelled") => ReminderStatus::Cancelled,
            Some(_) => return Err(invalid_payload("unknown reminder status")),
            None => return Err(invalid_payload("reminder status must be a string")),
        };
        let note = match object.get("note") {
            None => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => return Err(invalid_payload("reminder note must be a string")),
        };
        let target = match object.get("target") {
            None => None,
            Some(value) => Some(ReminderTarget::from_value(value)?),
        };
        if status == ReminderStatus::Pending
            && target.is_none()
            && note.as_ref().is_none_or(String::is_empty)
        {
            return Err(invalid_payload(
                "pending reminder requires a valid target or non-empty note",
            ));
        }
        Ok(Self {
            target,
            status,
            note,
        })
    }

    pub fn to_plaintext(&self) -> String {
        let status = match self.status {
            ReminderStatus::Pending => "pending",
            ReminderStatus::Done => "done",
            ReminderStatus::Cancelled => "cancelled",
        };
        let mut object = Map::new();
        object.insert("status".into(), Value::String(status.into()));
        if let Some(target) = &self.target {
            object.insert("target".into(), target.to_value());
        }
        if let Some(note) = &self.note {
            object.insert("note".into(), Value::String(note.clone()));
        }
        Value::Object(object).to_string()
    }
}

impl ReminderTarget {
    fn from_value(value: &Value) -> Result<Self, CommunicationCodecError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid_payload("reminder target must be an object"))?;
        let event_id = match object.get("id") {
            None => None,
            Some(value) => Some(parse_event_id(
                value
                    .as_str()
                    .ok_or_else(|| invalid_payload("target.id must be a string"))?,
                "target.id",
            )?),
        };
        let address = match object.get("a") {
            None => None,
            Some(value) => {
                let address = value
                    .as_str()
                    .ok_or_else(|| invalid_payload("target.a must be a string"))?;
                validate_nostr_address(address)?;
                Some(address.to_owned())
            }
        };
        let relays = match object.get("relays") {
            None => Vec::new(),
            Some(value) => {
                let values = value
                    .as_array()
                    .ok_or_else(|| invalid_payload("target.relays must be an array"))?;
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|relay| valid_websocket_url(relay))
                    .map(str::to_owned)
                    .collect()
            }
        };
        let preview = match object.get("preview") {
            None => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => return Err(invalid_payload("target.preview must be a string")),
        };
        if event_id.is_none() && address.is_none() {
            return Err(invalid_payload(
                "reminder target requires an event id or address",
            ));
        }
        Ok(Self {
            event_id,
            address,
            relays,
            preview,
        })
    }

    fn to_value(&self) -> Value {
        let mut object = Map::new();
        if let Some(event_id) = self.event_id {
            object.insert("id".into(), Value::String(event_id.to_hex()));
        }
        if let Some(address) = &self.address {
            object.insert("a".into(), Value::String(address.clone()));
        }
        if !self.relays.is_empty() {
            object.insert("relays".into(), json!(self.relays));
        }
        if let Some(preview) = &self.preview {
            object.insert("preview".into(), Value::String(preview.clone()));
        }
        Value::Object(object)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReminderEnvelope {
    pub author: PublicKey,
    pub coordinate: String,
    pub not_before: Option<u64>,
    pub expiration: Option<u64>,
    pub ciphertext: String,
}

impl ReminderEnvelope {
    pub fn parse_signed_event(
        event: &SignedEvent,
        authenticated_reader: PublicKey,
    ) -> Result<Self, CommunicationCodecError> {
        verify_user_event(event, KIND_EVENT_REMINDER)?;
        if event.event.public_key != authenticated_reader {
            return Err(invalid_envelope("reminder is readable only by its author"));
        }
        validate_nip44_ciphertext(&event.event.content)?;
        let coordinate = parse_single_text_tag(&event.event.tags, "d", false)?;
        let not_before =
            parse_optional_decimal_tag(&event.event.tags, "not_before", MAX_SAFE_INTEGER)?;
        let expiration = parse_optional_decimal_tag(&event.event.tags, "expiration", u64::MAX)?;
        if not_before
            .zip(expiration)
            .is_some_and(|(due, expiry)| expiry <= due)
        {
            return Err(invalid_envelope("expiration is not after not_before"));
        }
        Ok(Self {
            author: event.event.public_key,
            coordinate,
            not_before,
            expiration,
            ciphertext: event.event.content.clone(),
        })
    }

    pub fn validate_decrypted(
        &self,
        plaintext: &[u8],
    ) -> Result<ReminderBody, CommunicationCodecError> {
        let body = ReminderBody::parse(plaintext)?;
        match body.status {
            ReminderStatus::Pending if self.not_before.is_none() => Err(invalid_payload(
                "pending reminder requires exactly one not_before tag",
            )),
            ReminderStatus::Done | ReminderStatus::Cancelled if self.not_before.is_some() => {
                Err(invalid_payload("terminal reminder must omit not_before"))
            }
            _ => Ok(body),
        }
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        let mut tags = vec![
            vec!["d".into(), self.coordinate.clone()],
            vec!["alt".into(), "Encrypted reminder".into()],
        ];
        if let Some(not_before) = self.not_before {
            tags.push(vec!["not_before".into(), not_before.to_string()]);
        }
        if let Some(expiration) = self.expiration {
            tags.push(vec!["expiration".into(), expiration.to_string()]);
        }
        tags
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadStateEnvelope {
    pub author: PublicKey,
    pub slot_id: String,
    pub ciphertext: String,
}

impl ReadStateEnvelope {
    pub fn parse_signed_event(
        event: &SignedEvent,
        expected_author: PublicKey,
    ) -> Result<Self, CommunicationCodecError> {
        verify_user_event(event, KIND_READ_STATE)?;
        if event.event.public_key != expected_author {
            return Err(invalid_envelope("read state author mismatch"));
        }
        validate_nip44_ciphertext(&event.event.content)?;
        let coordinate = parse_single_text_tag(&event.event.tags, "d", false)?;
        let slot_id = coordinate
            .strip_prefix("read-state:")
            .filter(|value| valid_lower_hex(value, 32))
            .ok_or_else(|| invalid_envelope("invalid read-state coordinate"))?
            .to_owned();
        let read_state_markers = event
            .event
            .tags
            .iter()
            .filter(|tag| tag.first().map(String::as_str) == Some("t"))
            .filter(|tag| tag.get(1).map(String::as_str) == Some("read-state"))
            .count();
        if read_state_markers != 1 {
            return Err(invalid_envelope(
                "read state requires exactly one read-state t tag",
            ));
        }
        Ok(Self {
            author: expected_author,
            slot_id,
            ciphertext: event.event.content.clone(),
        })
    }

    pub fn coordinate(&self) -> String {
        format!("read-state:{}", self.slot_id)
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        vec![
            vec!["d".into(), self.coordinate()],
            vec!["t".into(), "read-state".into()],
        ]
    }

    pub fn validate_decrypted(
        &self,
        plaintext: &[u8],
        primary_slot_id: &str,
    ) -> Result<ReadStateVersion, CommunicationCodecError> {
        let version = ReadStateVersion::parse(plaintext)?;
        if let ReadStateVersion::V1(blob) = &version
            && !blob.overrides.is_empty()
            && self.slot_id != primary_slot_id
        {
            return Err(invalid_payload(
                "manual-unread overrides may only appear in the primary coordinate",
            ));
        }
        Ok(version)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReadStateVersion {
    V1(ReadStateBlob),
    Unsupported(u64),
}

impl ReadStateVersion {
    pub fn parse(plaintext: &[u8]) -> Result<Self, CommunicationCodecError> {
        let raw = RawReadState::parse(plaintext)?;
        if raw.version != 1 {
            return Ok(Self::Unsupported(raw.version));
        }
        Ok(Self::V1(ReadStateBlob::from_raw(raw)?))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OverrideRegister {
    pub set: u32,
    pub clear: u32,
    pub baseline: u32,
}

impl OverrideRegister {
    pub fn merge(self, other: Self) -> Self {
        Self {
            set: self.set.max(other.set),
            clear: self.clear.max(other.clear),
            baseline: self.baseline.max(other.baseline),
        }
    }

    pub fn is_active(self, effective_frontier: u32) -> bool {
        self.set > 0 && effective_frontier <= self.baseline && self.set > self.clear
    }

    pub fn mark_unread(&mut self, effective_frontier: u32) -> Result<(), CommunicationCodecError> {
        self.set = self
            .set
            .max(self.clear)
            .checked_add(1)
            .ok_or(CommunicationCodecError::CounterExhausted)?;
        self.baseline = effective_frontier;
        Ok(())
    }

    pub fn mark_read(&mut self, effective_frontier: u32) -> Result<(), CommunicationCodecError> {
        let Some(next) = self.set.max(self.clear).checked_add(1) else {
            if self.is_active(effective_frontier) {
                return Err(CommunicationCodecError::CounterExhausted);
            }
            return Ok(());
        };
        self.clear = next;
        Ok(())
    }

    fn canonical(self, effective_frontier: u32) -> CanonicalOverride {
        if self.is_active(effective_frontier) {
            CanonicalOverride::Live(self)
        } else if self.set > 0 || self.clear > 0 {
            CanonicalOverride::Tombstone(self.set.max(self.clear))
        } else {
            CanonicalOverride::Virgin
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CanonicalOverride {
    Live(OverrideRegister),
    Tombstone(u32),
    Virgin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadStateBlob {
    pub client_id: String,
    pub frontiers: BTreeMap<String, u32>,
    pub overrides: BTreeMap<String, OverrideRegister>,
}

impl ReadStateBlob {
    fn from_raw(raw: RawReadState) -> Result<Self, CommunicationCodecError> {
        let client_id = raw
            .client_id
            .ok_or_else(|| invalid_payload("client_id must be a string"))?;
        let character_count = client_id.chars().count();
        if !(1..=MAX_CLIENT_ID_CHARACTERS).contains(&character_count) {
            return Err(invalid_payload("client_id must contain 1-64 characters"));
        }
        let raw_contexts = raw
            .contexts
            .ok_or_else(|| invalid_payload("contexts must be an object"))?;
        let mut last_values = BTreeMap::new();
        for (key, value) in raw_contexts {
            last_values.insert(key, value);
        }
        let mut frontiers = BTreeMap::new();
        let mut override_parts: BTreeMap<String, RawOverrideParts> = BTreeMap::new();
        for (wire_key, value) in last_values {
            if wire_key.len() > MAX_CONTEXT_BYTES {
                continue;
            }
            let Some(value) = value.as_u64().and_then(|value| u32::try_from(value).ok()) else {
                if let Some((_, context)) = override_key(&wire_key) {
                    override_parts
                        .entry(context.to_owned())
                        .or_default()
                        .invalid = true;
                }
                continue;
            };
            if let Some((component, context)) = override_key(&wire_key) {
                let parts = override_parts.entry(context.to_owned()).or_default();
                match component {
                    OverrideComponent::Set => parts.set = Some(value),
                    OverrideComponent::Clear => parts.clear = Some(value),
                    OverrideComponent::Baseline => parts.baseline = Some(value),
                }
                continue;
            }
            if wire_key.starts_with("ov_") {
                continue;
            }
            let context = wire_key
                .strip_prefix("esc:")
                .unwrap_or(wire_key.as_str())
                .to_owned();
            frontiers.insert(context, value);
        }
        let mut overrides = BTreeMap::new();
        for (context, parts) in override_parts {
            if parts.invalid {
                continue;
            }
            let register = match (parts.set, parts.clear, parts.baseline) {
                (Some(set), Some(clear), Some(baseline)) => OverrideRegister {
                    set,
                    clear,
                    baseline,
                },
                (None, Some(clear), None) => OverrideRegister {
                    set: 0,
                    clear,
                    baseline: 0,
                },
                _ => continue,
            };
            overrides.insert(context, register);
        }
        Ok(Self {
            client_id,
            frontiers,
            overrides,
        })
    }

    pub fn merge(&mut self, other: &Self) {
        for (context, frontier) in &other.frontiers {
            self.frontiers
                .entry(context.clone())
                .and_modify(|current| *current = (*current).max(*frontier))
                .or_insert(*frontier);
        }
        for (context, register) in &other.overrides {
            self.overrides
                .entry(context.clone())
                .and_modify(|current| *current = current.merge(*register))
                .or_insert(*register);
        }
    }

    pub fn effective_frontier(&self, context: &str, parent: Option<&str>) -> u32 {
        let own = self.frontiers.get(context).copied().unwrap_or_default();
        parent.map_or(own, |parent| {
            own.max(self.frontiers.get(parent).copied().unwrap_or_default())
        })
    }

    pub fn to_plaintext(&self) -> Result<String, CommunicationCodecError> {
        let mut contexts = Map::new();
        for (context, frontier) in &self.frontiers {
            let wire_key = escape_frontier(context);
            contexts.insert(wire_key, json!(frontier));
        }
        for (context, register) in &self.overrides {
            let frontier = self.frontiers.get(context).copied().unwrap_or_default();
            match register.canonical(frontier) {
                CanonicalOverride::Live(register) => {
                    contexts.insert(format!("ov_s:{context}"), json!(register.set));
                    contexts.insert(format!("ov_c:{context}"), json!(register.clear));
                    contexts.insert(format!("ov_b:{context}"), json!(register.baseline));
                }
                CanonicalOverride::Tombstone(clear) => {
                    contexts.insert(format!("ov_c:{context}"), json!(clear));
                }
                CanonicalOverride::Virgin => {}
            }
        }
        if contexts.len() > MAX_CONTEXTS {
            return Err(invalid_payload("read state exceeds 10000 contexts"));
        }
        if contexts.keys().any(|key| key.len() > MAX_CONTEXT_BYTES) {
            return Err(invalid_payload("read-state context key exceeds 256 bytes"));
        }
        Ok(json!({
            "v": 1,
            "client_id": self.client_id,
            "contexts": contexts,
        })
        .to_string())
    }
}

#[derive(Clone, Debug)]
struct RawReadState {
    version: u64,
    client_id: Option<String>,
    contexts: Option<Vec<(String, Value)>>,
}

impl RawReadState {
    fn parse(bytes: &[u8]) -> Result<Self, CommunicationCodecError> {
        struct RawReadStateSeed;
        impl<'de> DeserializeSeed<'de> for RawReadStateSeed {
            type Value = RawReadState;

            fn deserialize<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                deserializer.deserialize_map(RawReadStateVisitor)
            }
        }

        struct RawReadStateVisitor;
        impl<'de> Visitor<'de> for RawReadStateVisitor {
            type Value = RawReadState;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a read-state object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut version = None;
                let mut client_id = None;
                let mut contexts = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "v" => {
                            if version.is_some() {
                                return Err(serde::de::Error::custom("duplicate key: v"));
                            }
                            version = Some(map.next_value::<Value>()?);
                        }
                        "client_id" => {
                            if client_id.is_some() {
                                return Err(serde::de::Error::custom("duplicate key: client_id"));
                            }
                            client_id = Some(map.next_value::<Value>()?);
                        }
                        "contexts" => {
                            if contexts.is_some() {
                                return Err(serde::de::Error::custom("duplicate key: contexts"));
                            }
                            contexts = Some(map.next_value_seed(ContextEntriesSeed)?);
                        }
                        _ => {
                            map.next_value::<Value>()?;
                        }
                    }
                }
                let version = version
                    .and_then(|value| value.as_u64())
                    .ok_or_else(|| serde::de::Error::custom("v must be an integer"))?;
                let client_id = client_id.and_then(|value| value.as_str().map(str::to_owned));
                Ok(RawReadState {
                    version,
                    client_id,
                    contexts,
                })
            }
        }

        struct ContextEntriesSeed;
        impl<'de> DeserializeSeed<'de> for ContextEntriesSeed {
            type Value = Vec<(String, Value)>;

            fn deserialize<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                deserializer.deserialize_map(ContextEntriesVisitor)
            }
        }

        struct ContextEntriesVisitor;
        impl<'de> Visitor<'de> for ContextEntriesVisitor {
            type Value = Vec<(String, Value)>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a read-state contexts object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut contexts = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    if contexts.len() == MAX_CONTEXTS {
                        return Err(serde::de::Error::custom(
                            "read state exceeds 10000 contexts",
                        ));
                    }
                    contexts.push((key, map.next_value::<Value>()?));
                }
                Ok(contexts)
            }
        }

        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let parsed = RawReadStateSeed
            .deserialize(&mut deserializer)
            .map_err(|error| invalid_payload(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| invalid_payload(error.to_string()))?;
        Ok(parsed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OverrideComponent {
    Set,
    Clear,
    Baseline,
}

#[derive(Default)]
struct RawOverrideParts {
    set: Option<u32>,
    clear: Option<u32>,
    baseline: Option<u32>,
    invalid: bool,
}

fn override_key(value: &str) -> Option<(OverrideComponent, &str)> {
    let (component, context) = if let Some(context) = value.strip_prefix("ov_s:") {
        (OverrideComponent::Set, context)
    } else if let Some(context) = value.strip_prefix("ov_c:") {
        (OverrideComponent::Clear, context)
    } else if let Some(context) = value.strip_prefix("ov_b:") {
        (OverrideComponent::Baseline, context)
    } else {
        return None;
    };
    (!context.is_empty()).then_some((component, context))
}

fn escape_frontier(context: &str) -> String {
    if context.starts_with("ov_") || context.starts_with("esc:") {
        format!("esc:{context}")
    } else {
        context.to_owned()
    }
}

fn verify_relay_event(
    event: &SignedEvent,
    relay: PublicKey,
    kind: u32,
) -> Result<(), CommunicationCodecError> {
    verify_user_event(event, kind)?;
    if event.event.public_key != relay {
        return Err(invalid_envelope("event is not signed by expected relay"));
    }
    Ok(())
}

fn verify_user_event(event: &SignedEvent, kind: u32) -> Result<(), CommunicationCodecError> {
    verify_signed_event(event, TimestampPolicy::Historical)
        .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
    if u32::from(event.event.kind) != kind {
        return Err(CommunicationCodecError::UnsupportedKind(event.event.kind));
    }
    Ok(())
}

fn ensure_exact_tag_names(
    tags: &[Vec<String>],
    expected: &[&str],
) -> Result<(), CommunicationCodecError> {
    if tags.len() != expected.len() {
        return Err(invalid_envelope("unexpected tag cardinality"));
    }
    for name in expected {
        parse_single_text_tag(tags, name, false)?;
    }
    if tags.iter().any(|tag| {
        tag.first()
            .is_none_or(|name| !expected.contains(&name.as_str()))
    }) {
        return Err(invalid_envelope("unexpected tag name"));
    }
    Ok(())
}

fn parse_single_text_tag(
    tags: &[Vec<String>],
    name: &str,
    allow_empty: bool,
) -> Result<String, CommunicationCodecError> {
    let mut matching = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some(name));
    let tag = matching
        .next()
        .ok_or_else(|| invalid_envelope(format!("missing {name} tag")))?;
    if matching.next().is_some() {
        return Err(invalid_envelope(format!("duplicate {name} tag")));
    }
    if tag.len() != 2 || (!allow_empty && tag[1].is_empty()) {
        return Err(invalid_envelope(format!("malformed {name} tag")));
    }
    Ok(tag[1].clone())
}

fn parse_optional_decimal_tag(
    tags: &[Vec<String>],
    name: &str,
    maximum: u64,
) -> Result<Option<u64>, CommunicationCodecError> {
    let mut matching = tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some(name));
    let Some(tag) = matching.next() else {
        return Ok(None);
    };
    if matching.next().is_some() || tag.len() != 2 {
        return Err(invalid_envelope(format!("malformed {name} tag")));
    }
    parse_canonical_decimal(&tag[1], maximum)
        .map(Some)
        .map_err(|reason| invalid_envelope(format!("malformed {name}: {reason}")))
}

fn parse_canonical_decimal(value: &str, maximum: u64) -> Result<u64, &'static str> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err("value is not a canonical decimal");
    }
    let value = value.parse::<u64>().map_err(|_| "value overflows u64")?;
    (value <= maximum)
        .then_some(value)
        .ok_or("value exceeds protocol maximum")
}

fn required_u64(object: &Map<String, Value>, key: &str) -> Result<u64, CommunicationCodecError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid_payload(format!("{key} must be an integer")))
}

fn parse_event_id(value: &str, field: &str) -> Result<EventId, CommunicationCodecError> {
    EventId::from_hex(value).map_err(|error| invalid_payload(format!("invalid {field}: {error}")))
}

fn validate_nostr_address(value: &str) -> Result<(), CommunicationCodecError> {
    let mut parts = value.splitn(3, ':');
    let kind = parts
        .next()
        .ok_or_else(|| invalid_payload("target.a is missing kind"))?;
    let public_key = parts
        .next()
        .ok_or_else(|| invalid_payload("target.a is missing public key"))?;
    let discriminator = parts
        .next()
        .ok_or_else(|| invalid_payload("target.a is missing discriminator"))?;
    parse_canonical_decimal(kind, u64::from(u16::MAX))
        .map_err(|reason| invalid_payload(format!("invalid target.a kind: {reason}")))?;
    PublicKey::from_hex(public_key)
        .map_err(|error| invalid_payload(format!("invalid target.a public key: {error}")))?;
    if discriminator.is_empty() {
        return Err(invalid_payload("target.a discriminator is empty"));
    }
    Ok(())
}

fn valid_websocket_url(value: &str) -> bool {
    let Some(authority_and_path) = value
        .strip_prefix("ws://")
        .or_else(|| value.strip_prefix("wss://"))
    else {
        return false;
    };
    let authority = authority_and_path
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    !authority.is_empty()
        && !authority.contains('@')
        && !authority.bytes().any(|byte| byte.is_ascii_whitespace())
}

fn validate_nip44_ciphertext(value: &str) -> Result<(), CommunicationCodecError> {
    if value.len() < 132
        || value.len() > MAX_NIP44_CIPHERTEXT_BYTES
        || !value.len().is_multiple_of(4)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
    {
        return Err(invalid_envelope("invalid NIP-44 ciphertext envelope"));
    }
    Ok(())
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, CommunicationCodecError> {
    struct StrictValue;
    impl<'de> DeserializeSeed<'de> for StrictValue {
        type Value = Value;

        fn deserialize<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for StrictValue {
        type Value = Value;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("JSON with unique object keys")
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
            Ok(Value::Bool(value))
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
            Ok(Value::Number(value.into()))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
            Ok(Value::Number(value.into()))
        }

        fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
            serde_json::Number::from_f64(value)
                .map(Value::Number)
                .ok_or_else(|| E::custom("non-finite number"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
            Ok(Value::String(value.to_owned()))
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
            Ok(Value::String(value))
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(Value::Null)
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(Value::Null)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_any(self)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while let Some(value) = sequence.next_element_seed(StrictValue)? {
                values.push(value);
            }
            Ok(Value::Array(values))
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut seen = HashSet::new();
            let mut values = Map::new();
            while let Some(key) = map.next_key::<String>()? {
                if !seen.insert(key.clone()) {
                    return Err(serde::de::Error::custom(format!("duplicate key: {key}")));
                }
                values.insert(key, map.next_value_seed(StrictValue)?);
            }
            Ok(Value::Object(values))
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue
        .deserialize(&mut deserializer)
        .map_err(|error| invalid_payload(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| invalid_payload(error.to_string()))?;
    Ok(value)
}

fn invalid_filter(reason: impl Into<String>) -> CommunicationCodecError {
    CommunicationCodecError::InvalidFilter(reason.into())
}

fn invalid_envelope(reason: impl Into<String>) -> CommunicationCodecError {
    CommunicationCodecError::InvalidEnvelope(reason.into())
}

fn invalid_payload(reason: impl Into<String>) -> CommunicationCodecError {
    CommunicationCodecError::InvalidPayload(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalEvent, EventSignature};
    use secp256k1::{Keypair, Message, SecretKey};

    const RELAY_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 1;
        secret
    };
    const USER_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 2;
        secret
    };
    const RELAY: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const USER: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";
    const EVENT_ID: &str = "7b4f3c2a1e9d8c7061524334aabbccddeeff00112233445566778899aabbccdd";

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("fixture public key")
    }

    fn sign(event: CanonicalEvent, secret: [u8; 32]) -> SignedEvent {
        let claimed_id = event.event_id().expect("event id");
        let secret = SecretKey::from_slice(&secret).expect("fixture secret");
        let keypair = Keypair::from_secret_key(&secp256k1::Secp256k1::new(), &secret);
        let signature = secp256k1::Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    fn ciphertext() -> String {
        "A".repeat(132)
    }

    #[test]
    fn channel_window_cursor_and_overlays_are_request_bound() {
        let cursor_id = EventId::from_hex(EVENT_ID).expect("cursor id");
        let request = ChannelWindowRequest::parse_filter(&json!({
            "#h": ["channel-a"],
            "limit": 500,
            "top_level": true,
            "include_summaries": true,
            "until": 1_751_499_000_u64,
            "before_id": EVENT_ID,
        }))
        .expect("window filter")
        .expect("window mode");
        assert_eq!(request.limit, MAX_WINDOW_LIMIT);
        assert_eq!(
            request.cursor,
            Some(WindowCursor {
                created_at: 1_751_499_000,
                id: cursor_id,
            })
        );
        assert_eq!(
            request.bounds_d_tag(),
            format!("channel-a:1751499000:{EVENT_ID}")
        );
        let next = WindowCursor {
            created_at: 1_751_498_000,
            id: EventId::from_hex(&"a".repeat(64)).expect("next id"),
        };
        let bounds = WindowBounds {
            channel_id: "channel-a".into(),
            request_cursor: request.cursor,
            has_more: true,
            next_cursor: Some(next),
        };
        let event = sign(
            CanonicalEvent::new(
                key(RELAY),
                1_751_500_000,
                KIND_WINDOW_BOUNDS as u16,
                bounds.to_tags(),
                bounds.content_json(),
            ),
            RELAY_SECRET,
        );
        assert_eq!(
            WindowBounds::parse_signed_event(&event, key(RELAY), &request).expect("window bounds"),
            bounds
        );
        assert!(
            ChannelWindowRequest::parse_filter(&json!({
                "#h": ["channel-a"],
                "top_level": true,
                "until": 1,
            }))
            .is_err()
        );
        let wrong_request = ChannelWindowRequest {
            channel_id: "channel-a".into(),
            kinds: None,
            limit: 50,
            cursor: None,
            include_summaries: false,
            include_aux: false,
        };
        assert!(WindowBounds::parse_signed_event(&event, key(RELAY), &wrong_request).is_err());
    }

    #[test]
    fn dm_visibility_is_relay_signed_owner_only_and_set_valued() {
        let snapshot = DmVisibilitySnapshot {
            viewer: key(USER),
            hidden_channels: BTreeSet::from(["dm-a".into(), "dm-b".into()]),
        };
        let event = sign(
            CanonicalEvent::new(
                key(RELAY),
                1_700_000_000,
                KIND_DM_VISIBILITY as u16,
                snapshot.to_tags(),
                String::new(),
            ),
            RELAY_SECRET,
        );
        assert_eq!(
            DmVisibilitySnapshot::parse_signed_event(&event, key(RELAY), key(USER))
                .expect("visibility"),
            snapshot
        );
        assert!(DmVisibilitySnapshot::parse_signed_event(&event, key(RELAY), key(RELAY)).is_err());
        let malformed = sign(
            CanonicalEvent::new(
                key(RELAY),
                1_700_000_001,
                KIND_DM_VISIBILITY as u16,
                vec![
                    vec!["d".into(), USER.into()],
                    vec!["p".into(), USER.into()],
                    vec!["h".into(), "dm-a".into()],
                    vec!["h".into(), "dm-b".into(), "unexpected".into()],
                ],
                String::new(),
            ),
            RELAY_SECRET,
        );
        assert!(
            DmVisibilitySnapshot::parse_signed_event(&malformed, key(RELAY), key(USER)).is_err()
        );
    }

    #[test]
    fn reminder_envelope_and_plaintext_enforce_schedule_and_privacy() {
        let envelope = ReminderEnvelope {
            author: key(USER),
            coordinate: "a3f8c2e1b4d79600e5d2f1a8c3b6094d".into(),
            not_before: Some(1_770_000_000),
            expiration: Some(1_780_000_000),
            ciphertext: ciphertext(),
        };
        let event = sign(
            CanonicalEvent::new(
                key(USER),
                1_769_990_000,
                KIND_EVENT_REMINDER as u16,
                envelope.to_tags(),
                envelope.ciphertext,
            ),
            USER_SECRET,
        );
        let parsed = ReminderEnvelope::parse_signed_event(&event, key(USER)).expect("reminder");
        let body = parsed
            .validate_decrypted(
                format!(
                    r#"{{"target":{{"id":"{EVENT_ID}","relays":["https://bad","wss://relay.example"]}},"status":"pending","note":"review"}}"#
                )
                .as_bytes(),
            )
            .expect("reminder body");
        assert_eq!(body.status, ReminderStatus::Pending);
        assert_eq!(body.target.expect("target").relays, ["wss://relay.example"]);
        assert!(ReminderEnvelope::parse_signed_event(&event, key(RELAY)).is_err());
        assert!(
            ReminderBody::parse(br#"{"status":"pending","status":"done","note":"x"}"#).is_err()
        );
        let invalid_expiration = sign(
            CanonicalEvent::new(
                key(USER),
                1_769_990_001,
                KIND_EVENT_REMINDER as u16,
                vec![
                    vec!["d".into(), envelope.coordinate],
                    vec!["not_before".into(), "1770000000".into()],
                    vec!["expiration".into(), "1770000000".into()],
                ],
                ciphertext(),
            ),
            USER_SECRET,
        );
        assert!(ReminderEnvelope::parse_signed_event(&invalid_expiration, key(USER)).is_err());
    }

    #[test]
    fn read_state_preserves_last_frontier_and_canonical_override_shapes() {
        let slot = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d";
        let envelope = ReadStateEnvelope {
            author: key(USER),
            slot_id: slot.into(),
            ciphertext: ciphertext(),
        };
        let event = sign(
            CanonicalEvent::new(
                key(USER),
                1_700_000_000,
                KIND_READ_STATE as u16,
                envelope.to_tags(),
                envelope.ciphertext,
            ),
            USER_SECRET,
        );
        let envelope =
            ReadStateEnvelope::parse_signed_event(&event, key(USER)).expect("read envelope");
        let plaintext = br#"{
            "v":1,
            "client_id":"desktop-v1",
            "contexts":{
                "general":2,
                "general":5,
                "esc:ov_s:opaque":7,
                "ov_s:general":2,
                "ov_c:general":1,
                "ov_b:general":5,
                "ov_s:partial":1,
                "partial":9,
                "bad":"not-an-integer"
            }
        }"#;
        let ReadStateVersion::V1(blob) = envelope
            .validate_decrypted(plaintext, slot)
            .expect("read-state blob")
        else {
            panic!("supported v1")
        };
        assert_eq!(blob.frontiers.get("general"), Some(&5));
        assert_eq!(blob.frontiers.get("ov_s:opaque"), Some(&7));
        assert_eq!(blob.frontiers.get("partial"), Some(&9));
        assert!(!blob.overrides.contains_key("partial"));
        assert!(blob.overrides["general"].is_active(5));
        let encoded = blob.to_plaintext().expect("canonical plaintext");
        assert!(encoded.contains(r#""esc:ov_s:opaque":7"#));
        assert!(encoded.contains(r#""ov_s:general":2"#));

        let additional = ReadStateEnvelope {
            slot_id: "f0e1d2c3b4a5968778695a4b3c2d1e0f".into(),
            ..envelope
        };
        assert!(additional.validate_decrypted(plaintext, slot).is_err());
    }

    #[test]
    fn read_state_merge_is_monotone_and_counters_never_wrap() {
        let mut left = ReadStateBlob {
            client_id: "left".into(),
            frontiers: BTreeMap::from([("general".into(), 10)]),
            overrides: BTreeMap::from([(
                "general".into(),
                OverrideRegister {
                    set: 2,
                    clear: 1,
                    baseline: 10,
                },
            )]),
        };
        let right = ReadStateBlob {
            client_id: "right".into(),
            frontiers: BTreeMap::from([("general".into(), 12), ("random".into(), 4)]),
            overrides: BTreeMap::from([(
                "general".into(),
                OverrideRegister {
                    set: 1,
                    clear: 3,
                    baseline: 11,
                },
            )]),
        };
        left.merge(&right);
        assert_eq!(left.frontiers["general"], 12);
        assert_eq!(
            left.overrides["general"],
            OverrideRegister {
                set: 2,
                clear: 3,
                baseline: 11,
            }
        );
        let encoded = left.to_plaintext().expect("tombstone encoding");
        assert!(encoded.contains(r#""ov_c:general":3"#));
        assert!(!encoded.contains("ov_s:general"));

        let mut exhausted = OverrideRegister {
            set: u32::MAX,
            clear: u32::MAX - 1,
            baseline: 20,
        };
        assert_eq!(
            exhausted.mark_unread(20),
            Err(CommunicationCodecError::CounterExhausted)
        );
        assert_eq!(
            exhausted.mark_read(20),
            Err(CommunicationCodecError::CounterExhausted)
        );
        assert_eq!(
            ReadStateVersion::parse(br#"{"v":2}"#).expect("forward-compatible version"),
            ReadStateVersion::Unsupported(2)
        );
    }
}
