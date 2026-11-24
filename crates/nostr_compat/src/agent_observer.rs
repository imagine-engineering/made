use crate::{
    PublicKey, SignedEvent,
    buzz_nips::agent_activity::{
        AgentActivityCodecError, ObserverEnvelope, ObserverFrame, ObserverPayload,
        ObserverTelemetry,
    },
    dm::Nip44Ciphertext,
    filter::{EventFilter, FilterError, MAX_FILTERS_PER_REQUEST},
    generated_kinds::KIND_AGENT_OBSERVER_FRAME,
};
use uuid::Uuid;

pub const AGENT_OBSERVER_CONTROL_FRESHNESS_SECONDS: u64 = 300;
const MAX_OBSERVER_IDENTIFIER_BYTES: usize = 4_096;
const MAX_OBSERVER_KIND_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentObserverCipherVersion {
    Nip44V2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentObserverDirection {
    Telemetry,
    Control,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentObserverTelemetryKind {
    AcpRead,
    AcpWrite,
    TurnStarted,
    SessionResolved,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentObserverIngress {
    Telemetry {
        kind: AgentObserverTelemetryKind,
        channel_id: Option<Uuid>,
        frame: ObserverTelemetry,
    },
    CancelTurn {
        channel_id: Uuid,
    },
    Ignored,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentObserverFrame {
    envelope: ObserverEnvelope,
    ciphertext: Nip44Ciphertext,
    direction: AgentObserverDirection,
    owner: PublicKey,
}

impl AgentObserverFrame {
    pub fn parse_authorized(
        event: &SignedEvent,
        authenticated_recipient: PublicKey,
        now: u64,
        is_agent_owner: impl FnOnce(PublicKey, PublicKey) -> bool,
    ) -> Result<Self, AgentObserverCodecError> {
        let envelope = ObserverEnvelope::parse_signed_event(event)?;
        validate_envelope_tags(event)?;
        if let ObserverFrame::Unknown(frame) = &envelope.frame
            && (frame.len() > MAX_OBSERVER_KIND_BYTES || frame.chars().any(char::is_control))
        {
            return Err(invalid_envelope("observer frame value is invalid"));
        }
        validate_outer_identifier("observer channel scope", envelope.channel.as_deref())?;
        let (direction, owner) =
            if envelope.sender == envelope.agent && envelope.recipient != envelope.agent {
                (AgentObserverDirection::Telemetry, envelope.recipient)
            } else if envelope.recipient == envelope.agent && envelope.sender != envelope.agent {
                (AgentObserverDirection::Control, envelope.sender)
            } else {
                return Err(AgentObserverCodecError::InvalidDirection);
            };
        match (&envelope.frame, direction) {
            (ObserverFrame::Telemetry, AgentObserverDirection::Telemetry)
            | (ObserverFrame::Control, AgentObserverDirection::Control)
            | (ObserverFrame::Unknown(_), _) => {}
            _ => return Err(AgentObserverCodecError::InvalidDirection),
        }
        if envelope.recipient != authenticated_recipient {
            return Err(AgentObserverCodecError::WrongRecipient);
        }
        if !is_agent_owner(envelope.agent, owner) {
            return Err(AgentObserverCodecError::UnauthorizedOwner);
        }
        if direction == AgentObserverDirection::Control {
            let minimum = now.saturating_sub(AGENT_OBSERVER_CONTROL_FRESHNESS_SECONDS);
            let maximum = now.saturating_add(AGENT_OBSERVER_CONTROL_FRESHNESS_SECONDS);
            if !(minimum..=maximum).contains(&event.event.created_at) {
                return Err(AgentObserverCodecError::StaleControl {
                    created_at: event.event.created_at,
                    minimum,
                    maximum,
                });
            }
        }
        let ciphertext = Nip44Ciphertext::parse(event.event.content.clone())
            .map_err(|_| AgentObserverCodecError::InvalidCiphertext)?;
        Ok(Self {
            envelope,
            ciphertext,
            direction,
            owner,
        })
    }

    pub const fn sender(&self) -> PublicKey {
        self.envelope.sender
    }

    pub const fn recipient(&self) -> PublicKey {
        self.envelope.recipient
    }

    pub const fn agent(&self) -> PublicKey {
        self.envelope.agent
    }

    pub const fn owner(&self) -> PublicKey {
        self.owner
    }

    pub const fn direction(&self) -> AgentObserverDirection {
        self.direction
    }

    pub const fn cipher_version(&self) -> AgentObserverCipherVersion {
        AgentObserverCipherVersion::Nip44V2
    }

    pub fn ciphertext(&self) -> &Nip44Ciphertext {
        &self.ciphertext
    }

    pub fn channel_scope(&self) -> Option<&str> {
        self.envelope.channel.as_deref()
    }

    pub fn is_recognized(&self) -> bool {
        self.envelope.is_recognized()
    }

    pub fn parse_decrypted(
        &self,
        plaintext: &[u8],
    ) -> Result<AgentObserverIngress, AgentObserverCodecError> {
        match ObserverPayload::parse(&self.envelope.frame, plaintext)? {
            ObserverPayload::Ignored => Ok(AgentObserverIngress::Ignored),
            ObserverPayload::Control(control) => {
                let channel_id = control
                    .channel_id
                    .as_deref()
                    .ok_or_else(|| invalid_payload("cancel_turn is missing channelId"))?
                    .parse()
                    .map_err(|_| invalid_payload("cancel_turn channelId must be a UUID"))?;
                Ok(AgentObserverIngress::CancelTurn { channel_id })
            }
            ObserverPayload::Telemetry(frame) => {
                validate_identifier("sessionId", frame.session_id.as_deref())?;
                validate_identifier("turnId", frame.turn_id.as_deref())?;
                if frame.kind.is_empty()
                    || frame.kind.len() > MAX_OBSERVER_KIND_BYTES
                    || frame.kind.chars().any(char::is_control)
                {
                    return Err(invalid_payload("observer kind is invalid"));
                }
                let channel_id = frame
                    .channel_id
                    .as_deref()
                    .map(str::parse)
                    .transpose()
                    .map_err(|_| invalid_payload("telemetry channelId must be a UUID"))?;
                let kind = match frame.kind.as_str() {
                    "acp_read" => AgentObserverTelemetryKind::AcpRead,
                    "acp_write" => AgentObserverTelemetryKind::AcpWrite,
                    "turn_started" => AgentObserverTelemetryKind::TurnStarted,
                    "session_resolved" => AgentObserverTelemetryKind::SessionResolved,
                    _ => return Ok(AgentObserverIngress::Ignored),
                };
                Ok(AgentObserverIngress::Telemetry {
                    kind,
                    channel_id,
                    frame,
                })
            }
        }
    }
}

pub fn authorize_agent_observer_filters(
    filters: &[EventFilter],
    authenticated_recipient: PublicKey,
    now: u64,
) -> Result<(), AgentObserverCodecError> {
    if filters.is_empty() {
        return Err(AgentObserverCodecError::UnauthorizedFilter);
    }
    if filters.len() > MAX_FILTERS_PER_REQUEST {
        return Err(FilterError::TooManyFilters {
            actual: filters.len(),
            maximum: MAX_FILTERS_PER_REQUEST,
        }
        .into());
    }
    let expected_recipient = authenticated_recipient.to_hex();
    for filter in filters {
        filter.validate()?;
        let recipient_is_exact = filter
            .generic_tags
            .get(&'p')
            .is_some_and(|recipients| recipients.as_slice() == [expected_recipient.as_str()]);
        let generic_tags_are_safe = filter
            .generic_tags
            .keys()
            .all(|key| matches!(key, 'p' | 'h'));
        if filter.kinds.as_slice() != [KIND_AGENT_OBSERVER_FRAME as u16]
            || !filter.ids.is_empty()
            || filter.until.is_some()
            || filter.since.is_none_or(|since| since < now)
            || !recipient_is_exact
            || !generic_tags_are_safe
        {
            return Err(AgentObserverCodecError::UnauthorizedFilter);
        }
    }
    Ok(())
}

fn validate_identifier(
    field: &'static str,
    value: Option<&str>,
) -> Result<(), AgentObserverCodecError> {
    if value.is_some_and(|value| {
        value.trim().is_empty()
            || value.len() > MAX_OBSERVER_IDENTIFIER_BYTES
            || value.chars().any(char::is_control)
    }) {
        return Err(invalid_payload(format!("{field} is invalid")));
    }
    Ok(())
}

fn validate_envelope_tags(event: &SignedEvent) -> Result<(), AgentObserverCodecError> {
    if event.event.tags.iter().any(|tag| {
        tag.len() != 2
            || !matches!(
                tag.first().map(String::as_str),
                Some("p" | "agent" | "frame" | "h")
            )
    }) {
        return Err(invalid_envelope(
            "observer frame contains an unsupported or malformed tag",
        ));
    }
    Ok(())
}

fn validate_outer_identifier(
    field: &'static str,
    value: Option<&str>,
) -> Result<(), AgentObserverCodecError> {
    if value.is_some_and(|value| {
        value.trim().is_empty()
            || value.len() > MAX_OBSERVER_IDENTIFIER_BYTES
            || value.chars().any(char::is_control)
    }) {
        return Err(invalid_envelope(format!("{field} is invalid")));
    }
    Ok(())
}

fn invalid_envelope(reason: impl Into<String>) -> AgentObserverCodecError {
    AgentObserverCodecError::InvalidEnvelope(reason.into())
}

fn invalid_payload(reason: impl Into<String>) -> AgentObserverCodecError {
    AgentObserverCodecError::InvalidPayload(reason.into())
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AgentObserverCodecError {
    #[error(transparent)]
    Activity(#[from] AgentActivityCodecError),
    #[error("observer frame is not an agent-to-owner or owner-to-agent route")]
    InvalidDirection,
    #[error("observer frame is addressed to another recipient")]
    WrongRecipient,
    #[error("observer frame agent-owner relationship is not authorized")]
    UnauthorizedOwner,
    #[error("observer frame does not contain canonical NIP-44 v2 ciphertext")]
    InvalidCiphertext,
    #[error("invalid observer envelope: {0}")]
    InvalidEnvelope(String),
    #[error("observer control timestamp {created_at} is outside {minimum}..={maximum}")]
    StaleControl {
        created_at: u64,
        minimum: u64,
        maximum: u64,
    },
    #[error("invalid observer payload: {0}")]
    InvalidPayload(String),
    #[error("observer subscription is not recipient-only and ephemeral")]
    UnauthorizedFilter,
    #[error(transparent)]
    Filter(#[from] FilterError),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use secp256k1::{Keypair, Message, SecretKey};

    use crate::{CanonicalEvent, EventSignature};

    use super::*;

    const AGENT_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 1;
        secret
    };
    const OWNER_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 2;
        secret
    };
    const AGENT: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const OWNER: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";
    const OTHER: &str = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9";
    const NIP44_V2_CIPHERTEXT: &str = "AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABedgcxyfmpph68LBjCWZsTI5lb0Cbg8dIPVYVe/WVj/l4Yd8HGgzC8awyBi9bn9ClRdtd2IPsmont0jN/cajVSQhahTOwuNNwoJtZIg35aSsUzeCq4tQfd8E+fLoKomdPxjs=";
    const NOW: u64 = 1_700_000_000;

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

    fn observer_event(
        sender: PublicKey,
        recipient: PublicKey,
        frame: &str,
        created_at: u64,
        content: &str,
        secret: [u8; 32],
    ) -> SignedEvent {
        sign(
            CanonicalEvent::new(
                sender,
                created_at,
                KIND_AGENT_OBSERVER_FRAME as u16,
                vec![
                    vec!["p".into(), recipient.to_hex()],
                    vec!["agent".into(), AGENT.into()],
                    vec!["frame".into(), frame.into()],
                    vec!["h".into(), "project-room".into()],
                ],
                content.into(),
            ),
            secret,
        )
    }

    #[test]
    fn telemetry_and_control_golden_frames_parse_without_acp_execution() {
        let telemetry = observer_event(
            key(AGENT),
            key(OWNER),
            "telemetry",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        let telemetry =
            AgentObserverFrame::parse_authorized(&telemetry, key(OWNER), NOW, |agent, owner| {
                agent == key(AGENT) && owner == key(OWNER)
            })
            .expect("authorized telemetry");
        assert_eq!(
            telemetry.cipher_version(),
            AgentObserverCipherVersion::Nip44V2
        );
        assert_eq!(telemetry.direction(), AgentObserverDirection::Telemetry);
        assert_eq!(telemetry.channel_scope(), Some("project-room"));
        let channel_id = Uuid::from_u128(10);
        let payload = format!(
            "{{\"seq\":42,\"timestamp\":\"2026-04-29T12:00:41.500Z\",\"kind\":\"acp_write\",\"agentIndex\":0,\"channelId\":\"{channel_id}\",\"sessionId\":\"session-1\",\"turnId\":\"turn-1\",\"payload\":{{\"jsonrpc\":\"2.0\"}}}}"
        );
        let AgentObserverIngress::Telemetry {
            kind,
            channel_id: parsed_channel_id,
            frame,
        } = telemetry
            .parse_decrypted(payload.as_bytes())
            .expect("decrypted telemetry")
        else {
            panic!("expected telemetry")
        };
        assert_eq!(kind, AgentObserverTelemetryKind::AcpWrite);
        assert_eq!(parsed_channel_id, Some(channel_id));
        assert_eq!(frame.seq, 42);
        assert!(!format!("{frame:?}").contains("jsonrpc"));

        let control = observer_event(
            key(OWNER),
            key(AGENT),
            "control",
            NOW,
            NIP44_V2_CIPHERTEXT,
            OWNER_SECRET,
        );
        let control =
            AgentObserverFrame::parse_authorized(&control, key(AGENT), NOW, |agent, owner| {
                agent == key(AGENT) && owner == key(OWNER)
            })
            .expect("authorized control");
        assert_eq!(control.direction(), AgentObserverDirection::Control);
        assert_eq!(control.owner(), key(OWNER));
        assert_eq!(
            control
                .parse_decrypted(
                    format!("{{\"type\":\"cancel_turn\",\"channelId\":\"{channel_id}\"}}")
                        .as_bytes()
                )
                .expect("decrypted control"),
            AgentObserverIngress::CancelTurn { channel_id }
        );
    }

    #[test]
    fn observer_ingress_rejects_wrong_cipher_versions_and_malformed_frames() {
        let mut wrong_version = STANDARD
            .decode(NIP44_V2_CIPHERTEXT)
            .expect("fixture ciphertext");
        wrong_version[0] = 1;
        let wrong_version = STANDARD.encode(wrong_version);
        let event = observer_event(
            key(AGENT),
            key(OWNER),
            "telemetry",
            NOW,
            &wrong_version,
            AGENT_SECRET,
        );
        assert_eq!(
            AgentObserverFrame::parse_authorized(&event, key(OWNER), NOW, |_, _| true),
            Err(AgentObserverCodecError::InvalidCiphertext)
        );

        let mut duplicate_recipient = observer_event(
            key(AGENT),
            key(OWNER),
            "telemetry",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        duplicate_recipient
            .event
            .tags
            .push(vec!["p".into(), OWNER.into()]);
        duplicate_recipient = sign(duplicate_recipient.event, AGENT_SECRET);
        assert!(matches!(
            AgentObserverFrame::parse_authorized(&duplicate_recipient, key(OWNER), NOW, |_, _| {
                true
            }),
            Err(AgentObserverCodecError::Activity(_))
        ));

        let mut unsupported_tag = observer_event(
            key(AGENT),
            key(OWNER),
            "telemetry",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        unsupported_tag
            .event
            .tags
            .push(vec!["leak".into(), "metadata".into()]);
        unsupported_tag = sign(unsupported_tag.event, AGENT_SECRET);
        assert!(matches!(
            AgentObserverFrame::parse_authorized(&unsupported_tag, key(OWNER), NOW, |_, _| true),
            Err(AgentObserverCodecError::InvalidEnvelope(_))
        ));

        let self_control = observer_event(
            key(AGENT),
            key(AGENT),
            "control",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        assert_eq!(
            AgentObserverFrame::parse_authorized(&self_control, key(AGENT), NOW, |_, _| true),
            Err(AgentObserverCodecError::InvalidDirection)
        );
    }

    #[test]
    fn observer_privacy_rejects_other_readers_owners_and_historical_controls() {
        let telemetry = observer_event(
            key(AGENT),
            key(OWNER),
            "telemetry",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        assert_eq!(
            AgentObserverFrame::parse_authorized(&telemetry, key(OTHER), NOW, |_, _| true),
            Err(AgentObserverCodecError::WrongRecipient)
        );
        assert_eq!(
            AgentObserverFrame::parse_authorized(&telemetry, key(OWNER), NOW, |_, _| false),
            Err(AgentObserverCodecError::UnauthorizedOwner)
        );

        let stale_control = observer_event(
            key(OWNER),
            key(AGENT),
            "control",
            NOW - AGENT_OBSERVER_CONTROL_FRESHNESS_SECONDS - 1,
            NIP44_V2_CIPHERTEXT,
            OWNER_SECRET,
        );
        assert!(matches!(
            AgentObserverFrame::parse_authorized(&stale_control, key(AGENT), NOW, |_, _| true),
            Err(AgentObserverCodecError::StaleControl { .. })
        ));
    }

    #[test]
    fn observer_subscription_and_payload_gates_are_ephemeral_and_forward_compatible() {
        let mut filter = EventFilter {
            kinds: vec![KIND_AGENT_OBSERVER_FRAME as u16],
            since: Some(NOW),
            ..EventFilter::default()
        };
        filter.generic_tags.insert('p', vec![key(OWNER).to_hex()]);
        authorize_agent_observer_filters(&[filter.clone()], key(OWNER), NOW)
            .expect("recipient-only live filter");
        filter.since = Some(NOW - 1);
        assert_eq!(
            authorize_agent_observer_filters(&[filter], key(OWNER), NOW),
            Err(AgentObserverCodecError::UnauthorizedFilter)
        );
        let broad_filter = EventFilter {
            kinds: vec![KIND_AGENT_OBSERVER_FRAME as u16],
            since: Some(NOW),
            generic_tags: BTreeMap::new(),
            ..EventFilter::default()
        };
        assert_eq!(
            authorize_agent_observer_filters(&[broad_filter], key(OWNER), NOW),
            Err(AgentObserverCodecError::UnauthorizedFilter)
        );

        let unknown = observer_event(
            key(AGENT),
            key(OWNER),
            "future-frame",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        let unknown = AgentObserverFrame::parse_authorized(&unknown, key(OWNER), NOW, |_, _| true)
            .expect("forward-compatible envelope");
        assert!(!unknown.is_recognized());
        assert_eq!(
            unknown
                .parse_decrypted(b"not JSON because the frame is unknown")
                .expect("unknown frame is ignored"),
            AgentObserverIngress::Ignored
        );

        let telemetry = observer_event(
            key(AGENT),
            key(OWNER),
            "telemetry",
            NOW,
            NIP44_V2_CIPHERTEXT,
            AGENT_SECRET,
        );
        let telemetry =
            AgentObserverFrame::parse_authorized(&telemetry, key(OWNER), NOW, |_, _| true)
                .expect("telemetry");
        assert_eq!(
            telemetry
                .parse_decrypted(br#"{"seq":1,"timestamp":"2026-04-29T12:00:41Z","kind":"future_kind","agentIndex":null,"channelId":null,"sessionId":null,"turnId":null,"payload":{}}"#)
                .expect("unknown telemetry kind"),
            AgentObserverIngress::Ignored
        );
        assert!(
            telemetry
                .parse_decrypted(br#"{"seq":1,"timestamp":"invalid","kind":"acp_read","agentIndex":null,"channelId":null,"sessionId":null,"turnId":null,"payload":{}}"#)
                .is_err()
        );
    }
}
