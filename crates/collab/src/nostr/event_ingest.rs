use std::{error::Error, fmt, sync::Arc};

use collaboration_domain::{AuthenticatedPrincipalKind, OperationId};
use nostr_compat::{
    CanonicalEvent, EventCodecError, EventId, EventSignature, PublicKey, SignedEvent,
    TimestampPolicy, VerificationError, verify_signed_event,
};
use serde_json::Value;
use uuid::Uuid;

use crate::{
    collaboration_command::{
        DomainCommandDisposition, DomainCommandSink, DomainCommandSubmissionError,
    },
    nostr::{
        MAX_NOSTR_FRAME_BYTES,
        ingress::{
            CURRENT_NOSTR_INGRESS_VERSION, NostrIngressDeployment, NostrIngressError,
            NostrIngressRequest, VersionedNostrIngress,
        },
    },
    tenant_admission::AuthorizedRpcRequest,
};

const MAX_EVENT_TIMESTAMP_DRIFT_SECONDS: u64 = 15 * 60;
const KIND_GIFT_WRAP: u16 = 1_059;
const KIND_AUTH: u16 = 22_242;
const EVENT_OPERATION_NAMESPACE: Uuid = Uuid::from_u128(0x4805f355_9a23_5ae7_b303_78f0c37d302a);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrEventCommand {
    signed_event: SignedEvent,
    wire_event: Value,
}

impl NostrEventCommand {
    pub const fn signed_event(&self) -> &SignedEvent {
        &self.signed_event
    }

    pub const fn wire_event(&self) -> &Value {
        &self.wire_event
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrEventIngestStatus {
    Accepted,
    Duplicate,
    Invalid,
    Unauthorized,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NostrEventIngestOutcome {
    frame: String,
    status: NostrEventIngestStatus,
}

impl NostrEventIngestOutcome {
    fn accepted(event_id: EventId, status: NostrEventIngestStatus, message: &str) -> Self {
        Self {
            frame: ok_frame(event_id, true, message),
            status,
        }
    }

    fn rejected(event_id: EventId, status: NostrEventIngestStatus, message: &str) -> Self {
        Self {
            frame: ok_frame(event_id, false, message),
            status,
        }
    }

    pub fn frame(&self) -> &str {
        &self.frame
    }

    pub const fn status(&self) -> NostrEventIngestStatus {
        self.status
    }
}

pub struct NostrEventIngress<S> {
    ingress: VersionedNostrIngress<S>,
    deployment: NostrIngressDeployment,
}

impl<S> NostrEventIngress<S> {
    pub const fn new(command_sink: S, deployment: NostrIngressDeployment) -> Self {
        Self {
            ingress: VersionedNostrIngress::new(command_sink),
            deployment,
        }
    }

    pub async fn handle_frame(
        &self,
        admission: AuthorizedRpcRequest,
        raw: &str,
        now: u64,
    ) -> Result<NostrEventIngestOutcome, NostrEventFrameError>
    where
        S: DomainCommandSink<NostrEventCommand>,
    {
        let parsed = match parse_event_frame(raw) {
            Ok(parsed) => parsed,
            Err(error) => {
                return match error.event_id() {
                    Some(event_id) => Ok(NostrEventIngestOutcome::rejected(
                        event_id,
                        NostrEventIngestStatus::Invalid,
                        "invalid: malformed event",
                    )),
                    None => Err(error),
                };
            }
        };

        let NostrEventCommand {
            signed_event,
            wire_event,
        } = parsed.command;

        if signed_event.event.kind != KIND_GIFT_WRAP
            && !principal_matches_author(
                admission.principal().kind(),
                signed_event.event.public_key,
            )
        {
            return Ok(NostrEventIngestOutcome::rejected(
                parsed.event_id,
                NostrEventIngestStatus::Unauthorized,
                "invalid: event pubkey does not match authenticated identity",
            ));
        }

        if signed_event.event.kind == KIND_AUTH {
            return Ok(NostrEventIngestOutcome::rejected(
                parsed.event_id,
                NostrEventIngestStatus::Invalid,
                "invalid: AUTH events cannot be submitted via EVENT",
            ));
        }

        let signed_event = Arc::new(signed_event);
        let verification_event = Arc::clone(&signed_event);
        let verification = tokio::task::spawn_blocking(move || {
            verify_signed_event(
                &verification_event,
                TimestampPolicy::Bounded {
                    now,
                    max_past_seconds: MAX_EVENT_TIMESTAMP_DRIFT_SECONDS,
                    max_future_seconds: MAX_EVENT_TIMESTAMP_DRIFT_SECONDS,
                },
            )
        })
        .await;
        match verification {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let message = verification_message(&error);
                return Ok(NostrEventIngestOutcome::rejected(
                    parsed.event_id,
                    NostrEventIngestStatus::Invalid,
                    &message,
                ));
            }
            Err(_) => {
                return Ok(NostrEventIngestOutcome::rejected(
                    parsed.event_id,
                    NostrEventIngestStatus::Unavailable,
                    "error: internal server error",
                ));
            }
        }
        let signed_event = Arc::try_unwrap(signed_event).unwrap_or_else(|event| (*event).clone());
        let command = NostrEventCommand {
            signed_event,
            wire_event,
        };

        let operation_id =
            operation_id(admission.tenant().community_id().as_uuid(), parsed.event_id);
        let request = NostrIngressRequest::new(
            CURRENT_NOSTR_INGRESS_VERSION,
            CURRENT_NOSTR_INGRESS_VERSION,
            operation_id,
            None,
            None,
            self.deployment,
            command,
        );
        match self.ingress.submit(admission, request).await {
            Ok(receipt) => match receipt.disposition() {
                DomainCommandDisposition::Applied => Ok(NostrEventIngestOutcome::accepted(
                    parsed.event_id,
                    NostrEventIngestStatus::Accepted,
                    "",
                )),
                DomainCommandDisposition::Duplicate => Ok(NostrEventIngestOutcome::accepted(
                    parsed.event_id,
                    NostrEventIngestStatus::Duplicate,
                    "duplicate:",
                )),
            },
            Err(NostrIngressError::Command(DomainCommandSubmissionError::Rejected)) => {
                Ok(NostrEventIngestOutcome::rejected(
                    parsed.event_id,
                    NostrEventIngestStatus::Unauthorized,
                    "restricted: event rejected",
                ))
            }
            Err(
                NostrIngressError::Command(DomainCommandSubmissionError::Unavailable)
                | NostrIngressError::UnsupportedVersion { .. },
            ) => Ok(NostrEventIngestOutcome::rejected(
                parsed.event_id,
                NostrEventIngestStatus::Unavailable,
                "error: internal server error",
            )),
        }
    }
}

pub(crate) fn principal_matches_author(
    kind: &AuthenticatedPrincipalKind,
    author: PublicKey,
) -> bool {
    let expected = match kind {
        AuthenticatedPrincipalKind::NostrIdentity { public_key, .. } => *public_key.as_bytes(),
        AuthenticatedPrincipalKind::OwnerAttestedAgent {
            agent_public_key, ..
        } => *agent_public_key.as_bytes(),
        AuthenticatedPrincipalKind::SimAccount { .. }
        | AuthenticatedPrincipalKind::ScopedToken { .. }
        | AuthenticatedPrincipalKind::Service { .. } => return false,
    };
    expected == *author.as_bytes()
}

struct ParsedEventFrame {
    event_id: EventId,
    command: NostrEventCommand,
}

fn parse_event_frame(raw: &str) -> Result<ParsedEventFrame, NostrEventFrameError> {
    if raw.len() > MAX_NOSTR_FRAME_BYTES {
        return Err(NostrEventFrameError::new(
            NostrEventFrameErrorKind::FrameTooLarge,
            None,
        ));
    }
    let Value::Array(parts) = serde_json::from_str(raw)
        .map_err(|_| NostrEventFrameError::new(NostrEventFrameErrorKind::InvalidFrame, None))?
    else {
        return Err(NostrEventFrameError::new(
            NostrEventFrameErrorKind::InvalidFrame,
            None,
        ));
    };
    if parts.len() != 2 || parts.first().and_then(Value::as_str) != Some("EVENT") {
        return Err(NostrEventFrameError::new(
            NostrEventFrameErrorKind::InvalidFrame,
            None,
        ));
    }
    let wire_event = parts[1].clone();
    let event_id = wire_event
        .as_object()
        .and_then(|object| object.get("id"))
        .and_then(Value::as_str)
        .and_then(|value| EventId::from_hex(value).ok());
    parse_wire_signed_event(wire_event)
        .map(|(signed_event, wire_event)| {
            let event_id = signed_event.claimed_id;
            ParsedEventFrame {
                event_id,
                command: NostrEventCommand {
                    signed_event,
                    wire_event,
                },
            }
        })
        .map_err(|_| NostrEventFrameError::new(NostrEventFrameErrorKind::InvalidEvent, event_id))
}

pub(super) fn parse_wire_signed_event(
    wire_event: Value,
) -> Result<(SignedEvent, Value), EventCodecError> {
    let object = wire_event
        .as_object()
        .ok_or(EventCodecError::Serialization(
            "event must be an object".into(),
        ))?;
    let event_id = object
        .get("id")
        .and_then(Value::as_str)
        .ok_or(EventCodecError::Serialization("missing event id".into()))
        .and_then(EventId::from_hex)?;
    let public_key = object
        .get("pubkey")
        .and_then(Value::as_str)
        .ok_or(EventCodecError::Serialization("missing public key".into()))
        .and_then(PublicKey::from_hex)?;
    let created_at = object
        .get("created_at")
        .and_then(Value::as_u64)
        .ok_or(EventCodecError::Serialization("invalid created_at".into()))?;
    let kind = object
        .get("kind")
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(EventCodecError::Serialization("invalid kind".into()))?;
    let tags = serde_json::from_value(
        object
            .get("tags")
            .cloned()
            .ok_or(EventCodecError::Serialization("missing tags".into()))?,
    )
    .map_err(|error| EventCodecError::Serialization(error.to_string()))?;
    let content = object
        .get("content")
        .and_then(Value::as_str)
        .ok_or(EventCodecError::Serialization("invalid content".into()))?
        .to_owned();
    let signature = object
        .get("sig")
        .and_then(Value::as_str)
        .ok_or(EventCodecError::Serialization("missing signature".into()))
        .and_then(|value| {
            EventSignature::from_hex(value)
                .map_err(|error| EventCodecError::Serialization(error.to_string()))
        })?;
    let signed_event = SignedEvent {
        claimed_id: event_id,
        event: CanonicalEvent::new(public_key, created_at, kind, tags, content),
        signature,
    };
    let wire_event = serde_json::json!({
        "id": signed_event.claimed_id.to_hex(),
        "pubkey": signed_event.event.public_key.to_hex(),
        "created_at": signed_event.event.created_at,
        "kind": signed_event.event.kind,
        "tags": &signed_event.event.tags,
        "content": &signed_event.event.content,
        "sig": signed_event.signature.to_hex(),
    });
    Ok((signed_event, wire_event))
}

fn verification_message(error: &VerificationError) -> String {
    match error {
        VerificationError::TimestampOutsideWindow { .. } => {
            "invalid: event timestamp too far from server time".to_owned()
        }
        VerificationError::ContentTooLarge { actual, maximum } => {
            format!("invalid: content exceeds maximum size of {maximum} bytes (got {actual})")
        }
        VerificationError::CanonicalEventTooLarge { .. } => {
            "invalid: event exceeds maximum size".to_owned()
        }
        _ => format!("invalid: {error}"),
    }
}

pub(crate) fn operation_id(community_id: Uuid, event_id: EventId) -> OperationId {
    let mut source = [0_u8; 48];
    source[..16].copy_from_slice(community_id.as_bytes());
    source[16..].copy_from_slice(event_id.as_bytes());
    OperationId::from_uuid(Uuid::new_v5(&EVENT_OPERATION_NAMESPACE, &source))
}

fn ok_frame(event_id: EventId, accepted: bool, message: &str) -> String {
    serde_json::json!(["OK", event_id.to_hex(), accepted, message]).to_string()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NostrEventFrameErrorKind {
    FrameTooLarge,
    InvalidFrame,
    InvalidEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NostrEventFrameError {
    kind: NostrEventFrameErrorKind,
    event_id: Option<EventId>,
}

impl NostrEventFrameError {
    const fn new(kind: NostrEventFrameErrorKind, event_id: Option<EventId>) -> Self {
        Self { kind, event_id }
    }

    pub const fn kind(self) -> NostrEventFrameErrorKind {
        self.kind
    }

    pub const fn event_id(self) -> Option<EventId> {
        self.event_id
    }
}

impl fmt::Display for NostrEventFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.kind {
            NostrEventFrameErrorKind::FrameTooLarge => {
                "Nostr EVENT frame exceeds the configured limit"
            }
            NostrEventFrameErrorKind::InvalidFrame => "Nostr EVENT frame is invalid",
            NostrEventFrameErrorKind::InvalidEvent => "Nostr EVENT payload is invalid",
        })
    }
}

impl Error for NostrEventFrameError {}
