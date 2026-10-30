use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt;

const EVENT_ID_BYTES: usize = 32;
const PUBLIC_KEY_BYTES: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EventCodecError {
    #[error("{field} must contain exactly {expected} lowercase hexadecimal characters")]
    InvalidHex {
        field: &'static str,
        expected: usize,
    },
    #[error("failed to serialize canonical event: {0}")]
    Serialization(String),
}

macro_rules! fixed_lower_hex {
    ($name:ident, $bytes:expr, $field:literal) => {
        #[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name([u8; $bytes]);

        impl $name {
            pub fn from_hex(value: &str) -> Result<Self, EventCodecError> {
                if value.len() != $bytes * 2
                    || !value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                {
                    return Err(EventCodecError::InvalidHex {
                        field: $field,
                        expected: $bytes * 2,
                    });
                }
                let mut bytes = [0; $bytes];
                hex::decode_to_slice(value, &mut bytes).map_err(|_| {
                    EventCodecError::InvalidHex {
                        field: $field,
                        expected: $bytes * 2,
                    }
                })?;
                Ok(Self(bytes))
            }

            pub const fn from_bytes(bytes: [u8; $bytes]) -> Self {
                Self(bytes)
            }

            pub const fn as_bytes(&self) -> &[u8; $bytes] {
                &self.0
            }

            pub fn to_hex(self) -> String {
                hex::encode(self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.to_hex())
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.to_hex())
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&self.to_hex())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::from_hex(&value).map_err(serde::de::Error::custom)
            }
        }
    };
}

fixed_lower_hex!(EventId, EVENT_ID_BYTES, "event id");
fixed_lower_hex!(PublicKey, PUBLIC_KEY_BYTES, "public key");

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CanonicalEvent {
    pub public_key: PublicKey,
    pub created_at: u64,
    pub kind: u16,
    pub tags: Vec<Vec<String>>,
    pub content: String,
}

impl CanonicalEvent {
    pub fn new(
        public_key: PublicKey,
        created_at: u64,
        kind: u16,
        tags: Vec<Vec<String>>,
        content: String,
    ) -> Self {
        Self {
            public_key,
            created_at,
            kind,
            tags,
            content,
        }
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, EventCodecError> {
        serde_json::to_vec(&(
            0_u8,
            self.public_key.to_hex(),
            self.created_at,
            self.kind,
            &self.tags,
            &self.content,
        ))
        .map_err(|error| EventCodecError::Serialization(error.to_string()))
    }

    pub fn event_id(&self) -> Result<EventId, EventCodecError> {
        let digest = Sha256::digest(self.canonical_bytes()?);
        let mut bytes = [0; EVENT_ID_BYTES];
        bytes.copy_from_slice(&digest);
        Ok(EventId::from_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const EVENTS: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
    ));

    fn fixture_event(value: &Value) -> Result<CanonicalEvent, EventCodecError> {
        let public_key = PublicKey::from_hex(
            value
                .get("pubkey")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        )?;
        let created_at = value.get("created_at").and_then(Value::as_u64).ok_or(
            EventCodecError::Serialization("invalid created_at fixture".into()),
        )?;
        let kind = value
            .get("kind")
            .and_then(Value::as_u64)
            .and_then(|kind| u16::try_from(kind).ok())
            .ok_or(EventCodecError::Serialization(
                "invalid kind fixture".into(),
            ))?;
        let tags = serde_json::from_value(value.get("tags").cloned().unwrap_or_default())
            .map_err(|error| EventCodecError::Serialization(error.to_string()))?;
        let content = value
            .get("content")
            .and_then(Value::as_str)
            .ok_or(EventCodecError::Serialization(
                "invalid content fixture".into(),
            ))?
            .to_owned();
        Ok(CanonicalEvent::new(
            public_key, created_at, kind, tags, content,
        ))
    }

    #[test]
    fn event_vectors_match_frozen_ids_and_bytes() {
        let fixture: Value = serde_json::from_str(EVENTS).expect("valid frozen event corpus");
        let events = fixture
            .get("events")
            .and_then(Value::as_object)
            .expect("fixture event map");
        let matching_ids = [
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
            "malformed_signature",
            "malformed_pubkey",
        ];

        for name in matching_ids {
            let value = &events[name];
            let event = fixture_event(value).expect("structurally valid event fixture");
            let expected = value["id"].as_str().expect("fixture event id");
            assert_eq!(
                event.event_id().expect("event id").to_hex(),
                expected,
                "{name}"
            );
        }

        let legacy = fixture_event(&events["legacy_message"]).expect("legacy event");
        assert_eq!(
            String::from_utf8(legacy.canonical_bytes().expect("canonical bytes"))
                .expect("canonical JSON is UTF-8"),
            "[0,\"79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798\",1786579200,9,[[\"h\",\"cafe0000-0000-0000-0000-000000000010\"]],\"legacy hello\"]"
        );
    }

    #[test]
    fn event_vectors_detect_tampered_content() {
        let fixture: Value = serde_json::from_str(EVENTS).expect("valid frozen event corpus");
        let tampered = &fixture["events"]["malformed_tampered_content"];
        let computed = fixture_event(tampered)
            .expect("structurally valid tampered fixture")
            .event_id()
            .expect("event id");

        assert_ne!(
            computed.to_hex(),
            tampered["id"].as_str().expect("fixture id")
        );
        assert_eq!(
            computed.to_hex(),
            "0c3a3f53d89b5dbb338fc7ecf0a6af734c3ec54b1ce28e3c4961a71342aeb904"
        );
    }

    #[test]
    fn event_vectors_preserve_unicode_and_json_escaping() {
        let event = CanonicalEvent::new(
            PublicKey::from_hex(&"01".repeat(32)).expect("public key"),
            7,
            1,
            vec![vec!["emoji".into(), "🦀".into()]],
            "line\n\"quoted\"".into(),
        );
        let bytes = event.canonical_bytes().expect("canonical bytes");
        let text = String::from_utf8(bytes).expect("canonical JSON is UTF-8");

        assert!(text.contains("🦀"));
        assert!(text.contains("line\\n\\\"quoted\\\""));
        assert!(!text.contains(": "));
    }

    #[test]
    fn event_vectors_reject_noncanonical_hex() {
        assert!(PublicKey::from_hex(&"AB".repeat(32)).is_err());
        assert!(EventId::from_hex("00").is_err());
        assert!(serde_json::from_str::<EventId>(&format!("\"{}\"", "GG".repeat(32))).is_err());
    }
}
