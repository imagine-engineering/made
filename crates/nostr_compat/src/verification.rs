use crate::event::{CanonicalEvent, EventCodecError, EventId};
use secp256k1::{Message, Secp256k1, XOnlyPublicKey, schnorr::Signature};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

pub const MAX_EVENT_CONTENT_BYTES: usize = 256 * 1024;
pub const MAX_CANONICAL_EVENT_BYTES: usize = 512 * 1024;
const SIGNATURE_BYTES: usize = 64;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct EventSignature([u8; SIGNATURE_BYTES]);

impl EventSignature {
    pub fn from_hex(value: &str) -> Result<Self, VerificationError> {
        if value.len() != SIGNATURE_BYTES * 2
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(VerificationError::MalformedSignature);
        }
        let mut bytes = [0; SIGNATURE_BYTES];
        hex::decode_to_slice(value, &mut bytes)
            .map_err(|_| VerificationError::MalformedSignature)?;
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; SIGNATURE_BYTES] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

impl fmt::Debug for EventSignature {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("EventSignature")
            .field(&self.to_hex())
            .finish()
    }
}

impl Serialize for EventSignature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for EventSignature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_hex(&value).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedEvent {
    pub claimed_id: EventId,
    pub event: CanonicalEvent,
    pub signature: EventSignature,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestampPolicy {
    Historical,
    Bounded {
        now: u64,
        max_past_seconds: u64,
        max_future_seconds: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum VerificationError {
    #[error(transparent)]
    Codec(#[from] EventCodecError),
    #[error("event content is {actual} bytes, maximum is {maximum}")]
    ContentTooLarge { actual: usize, maximum: usize },
    #[error("canonical event is {actual} bytes, maximum is {maximum}")]
    CanonicalEventTooLarge { actual: usize, maximum: usize },
    #[error("event timestamp {created_at} is outside {minimum}..={maximum}")]
    TimestampOutsideWindow {
        created_at: u64,
        minimum: u64,
        maximum: u64,
    },
    #[error("invalid event id: computed {computed}, got {claimed}")]
    InvalidEventId { computed: EventId, claimed: EventId },
    #[error("invalid x-only secp256k1 public key")]
    InvalidPublicKey,
    #[error("signature must contain exactly 128 lowercase hexadecimal characters")]
    MalformedSignature,
    #[error("invalid schnorr signature")]
    InvalidSignature,
}

pub fn verify_signed_event(
    signed_event: &SignedEvent,
    timestamp_policy: TimestampPolicy,
) -> Result<(), VerificationError> {
    let content_bytes = signed_event.event.content.len();
    if content_bytes > MAX_EVENT_CONTENT_BYTES {
        return Err(VerificationError::ContentTooLarge {
            actual: content_bytes,
            maximum: MAX_EVENT_CONTENT_BYTES,
        });
    }

    let canonical_bytes = signed_event.event.canonical_bytes()?;
    if canonical_bytes.len() > MAX_CANONICAL_EVENT_BYTES {
        return Err(VerificationError::CanonicalEventTooLarge {
            actual: canonical_bytes.len(),
            maximum: MAX_CANONICAL_EVENT_BYTES,
        });
    }

    verify_timestamp(signed_event.event.created_at, timestamp_policy)?;

    let computed_id = signed_event.event.event_id()?;
    if computed_id != signed_event.claimed_id {
        return Err(VerificationError::InvalidEventId {
            computed: computed_id,
            claimed: signed_event.claimed_id,
        });
    }

    let public_key = XOnlyPublicKey::from_slice(signed_event.event.public_key.as_bytes())
        .map_err(|_| VerificationError::InvalidPublicKey)?;
    let signature = Signature::from_slice(signed_event.signature.as_bytes())
        .map_err(|_| VerificationError::MalformedSignature)?;
    let message = Message::from_digest(*signed_event.claimed_id.as_bytes());
    Secp256k1::verification_only()
        .verify_schnorr(&signature, &message, &public_key)
        .map_err(|_| VerificationError::InvalidSignature)
}

fn verify_timestamp(created_at: u64, policy: TimestampPolicy) -> Result<(), VerificationError> {
    let TimestampPolicy::Bounded {
        now,
        max_past_seconds,
        max_future_seconds,
    } = policy
    else {
        return Ok(());
    };
    let minimum = now.saturating_sub(max_past_seconds);
    let maximum = now.saturating_add(max_future_seconds);
    if !(minimum..=maximum).contains(&created_at) {
        return Err(VerificationError::TimestampOutsideWindow {
            created_at,
            minimum,
            maximum,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PublicKey;
    use serde_json::Value;

    const EVENTS: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
    ));

    fn event_fixture(name: &str) -> Value {
        let fixture: Value = serde_json::from_str(EVENTS).expect("valid frozen event corpus");
        fixture["events"][name].clone()
    }

    fn signed_fixture(name: &str) -> SignedEvent {
        let value = event_fixture(name);
        SignedEvent {
            claimed_id: EventId::from_hex(value["id"].as_str().expect("event id"))
                .expect("canonical event id"),
            event: CanonicalEvent::new(
                PublicKey::from_hex(value["pubkey"].as_str().expect("public key"))
                    .expect("canonical public key"),
                value["created_at"].as_u64().expect("created_at"),
                u16::try_from(value["kind"].as_u64().expect("kind")).expect("u16 kind"),
                serde_json::from_value(value["tags"].clone()).expect("string tags"),
                value["content"].as_str().expect("content").to_owned(),
            ),
            signature: EventSignature::from_hex(value["sig"].as_str().expect("signature"))
                .expect("canonical signature"),
        }
    }

    #[test]
    fn verification_accepts_frozen_valid_vectors() {
        for name in [
            "legacy_message",
            "v2_message",
            "profile_old",
            "profile_new",
            "profile_tie_a",
            "profile_tie_b",
            "author_only_push",
            "p_gated_visibility",
            "persona_private",
            "persona_shared",
        ] {
            verify_signed_event(&signed_fixture(name), TimestampPolicy::Historical)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
        }
    }

    #[test]
    fn verification_rejects_altered_and_invalid_crypto_vectors() {
        assert!(matches!(
            verify_signed_event(
                &signed_fixture("malformed_tampered_content"),
                TimestampPolicy::Historical
            ),
            Err(VerificationError::InvalidEventId { .. })
        ));
        assert_eq!(
            verify_signed_event(
                &signed_fixture("malformed_signature"),
                TimestampPolicy::Historical
            ),
            Err(VerificationError::InvalidSignature)
        );
        assert_eq!(
            verify_signed_event(
                &signed_fixture("malformed_pubkey"),
                TimestampPolicy::Historical
            ),
            Err(VerificationError::InvalidPublicKey)
        );
    }

    #[test]
    fn verification_applies_explicit_timestamp_policy() {
        let event = signed_fixture("legacy_message");
        assert!(
            verify_signed_event(
                &event,
                TimestampPolicy::Bounded {
                    now: event.event.created_at,
                    max_past_seconds: 900,
                    max_future_seconds: 900,
                }
            )
            .is_ok()
        );
        assert_eq!(
            verify_signed_event(
                &event,
                TimestampPolicy::Bounded {
                    now: event.event.created_at + 901,
                    max_past_seconds: 900,
                    max_future_seconds: 900,
                }
            ),
            Err(VerificationError::TimestampOutsideWindow {
                created_at: event.event.created_at,
                minimum: event.event.created_at + 1,
                maximum: event.event.created_at + 1801,
            })
        );
    }

    #[test]
    fn verification_rejects_oversized_content_before_crypto() {
        let mut event = signed_fixture("legacy_message");
        event.event.content = "x".repeat(MAX_EVENT_CONTENT_BYTES + 1);

        assert_eq!(
            verify_signed_event(&event, TimestampPolicy::Historical),
            Err(VerificationError::ContentTooLarge {
                actual: MAX_EVENT_CONTENT_BYTES + 1,
                maximum: MAX_EVENT_CONTENT_BYTES,
            })
        );
    }

    #[test]
    fn verification_rejects_malformed_signature_encoding() {
        assert_eq!(
            EventSignature::from_hex(&"AB".repeat(SIGNATURE_BYTES)),
            Err(VerificationError::MalformedSignature)
        );
        assert!(
            serde_json::from_str::<EventSignature>(&format!(
                "\"{}\"",
                "0".repeat(SIGNATURE_BYTES * 2 - 1)
            ))
            .is_err()
        );
    }
}
