use crate::filter::{EventFilter, FilterError};
use crate::generated_kinds::KIND_GIFT_WRAP;
use crate::{CanonicalEvent, PublicKey, SignedEvent, TimestampPolicy, verify_signed_event};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use std::fmt;

const MIN_NIP44_CIPHERTEXT_BYTES: usize = 99;
const MAX_NIP44_CIPHERTEXT_CHARACTERS: usize = 87_472;
const NIP44_VERSION: u8 = 2;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum DirectMessageCodecError {
    #[error("invalid direct-message envelope: {0}")]
    InvalidEnvelope(&'static str),
    #[error("direct-message envelope is addressed to another recipient")]
    WrongRecipient,
    #[error("direct-message filter is not restricted to the authenticated recipient")]
    UnauthorizedFilter,
    #[error(transparent)]
    InvalidFilter(#[from] FilterError),
}

#[derive(Clone, Eq, PartialEq)]
pub struct Nip44Ciphertext(String);

impl Nip44Ciphertext {
    pub fn parse(value: impl Into<String>) -> Result<Self, DirectMessageCodecError> {
        let value = value.into();
        if value.len() < 132
            || value.len() > MAX_NIP44_CIPHERTEXT_CHARACTERS
            || !value.len().is_multiple_of(4)
        {
            return Err(invalid_envelope());
        }

        let decoded = STANDARD.decode(&value).map_err(|_| invalid_envelope())?;
        if decoded.len() < MIN_NIP44_CIPHERTEXT_BYTES
            || decoded.first().copied() != Some(NIP44_VERSION)
            || STANDARD.encode(&decoded) != value
        {
            return Err(invalid_envelope());
        }

        Ok(Self(value))
    }

    pub fn wire_value(&self) -> &str {
        &self.0
    }

    pub fn into_wire_value(self) -> String {
        self.0
    }
}

impl fmt::Debug for Nip44Ciphertext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Nip44Ciphertext")
            .field("encoded_bytes", &self.0.len())
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DirectMessageIndexing {
    Excluded,
}

#[derive(Clone, Eq, PartialEq)]
pub struct GiftWrapEnvelope {
    ephemeral_author: PublicKey,
    created_at: u64,
    recipient: PublicKey,
    ciphertext: Nip44Ciphertext,
}

impl GiftWrapEnvelope {
    pub fn new(
        ephemeral_author: PublicKey,
        created_at: u64,
        recipient: PublicKey,
        ciphertext: Nip44Ciphertext,
    ) -> Self {
        Self {
            ephemeral_author,
            created_at,
            recipient,
            ciphertext,
        }
    }

    pub fn parse_signed_event(
        event: &SignedEvent,
        authenticated_recipient: PublicKey,
    ) -> Result<Self, DirectMessageCodecError> {
        verify_signed_event(event, TimestampPolicy::Historical)
            .map_err(|_| DirectMessageCodecError::InvalidEnvelope("invalid signed gift wrap"))?;
        if u32::from(event.event.kind) != KIND_GIFT_WRAP {
            return Err(DirectMessageCodecError::InvalidEnvelope(
                "unsupported event kind",
            ));
        }

        let recipient = parse_recipient(&event.event.tags)?;
        if recipient != authenticated_recipient {
            return Err(DirectMessageCodecError::WrongRecipient);
        }
        let ciphertext = Nip44Ciphertext::parse(event.event.content.clone())?;

        Ok(Self::new(
            event.event.public_key,
            event.event.created_at,
            recipient,
            ciphertext,
        ))
    }

    pub fn ephemeral_author(&self) -> PublicKey {
        self.ephemeral_author
    }

    pub fn created_at(&self) -> u64 {
        self.created_at
    }

    pub fn recipient(&self) -> PublicKey {
        self.recipient
    }

    pub fn ciphertext(&self) -> &Nip44Ciphertext {
        &self.ciphertext
    }

    pub const fn indexing(&self) -> DirectMessageIndexing {
        DirectMessageIndexing::Excluded
    }

    pub fn is_visible_to(&self, reader: PublicKey) -> bool {
        self.recipient == reader
    }

    pub fn to_canonical_event(&self) -> CanonicalEvent {
        CanonicalEvent::new(
            self.ephemeral_author,
            self.created_at,
            KIND_GIFT_WRAP as u16,
            vec![vec!["p".into(), self.recipient.to_hex()]],
            self.ciphertext.wire_value().to_owned(),
        )
    }

    pub fn into_canonical_event(self) -> CanonicalEvent {
        CanonicalEvent::new(
            self.ephemeral_author,
            self.created_at,
            KIND_GIFT_WRAP as u16,
            vec![vec!["p".into(), self.recipient.to_hex()]],
            self.ciphertext.into_wire_value(),
        )
    }
}

impl fmt::Debug for GiftWrapEnvelope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GiftWrapEnvelope")
            .field("ephemeral_author", &self.ephemeral_author)
            .field("created_at", &self.created_at)
            .field("recipient", &self.recipient)
            .field("ciphertext", &"<redacted>")
            .finish()
    }
}

pub fn authorize_gift_wrap_filters(
    filters: &[EventFilter],
    authenticated_reader: PublicKey,
) -> Result<(), DirectMessageCodecError> {
    for filter in filters {
        filter.validate()?;
        if filter.kinds.is_empty() || filter.kinds.contains(&(KIND_GIFT_WRAP as u16)) {
            let expected_recipient = authenticated_reader.to_hex();
            let authorized = filter.generic_tags.get(&'p').is_some_and(|recipients| {
                recipients.len() == 1 && recipients[0] == expected_recipient
            });
            if !authorized {
                return Err(DirectMessageCodecError::UnauthorizedFilter);
            }
        }
    }
    Ok(())
}

fn parse_recipient(tags: &[Vec<String>]) -> Result<PublicKey, DirectMessageCodecError> {
    let [tag] = tags else {
        return Err(DirectMessageCodecError::InvalidEnvelope(
            "gift wrap must contain exactly one recipient tag",
        ));
    };
    let [name, value] = tag.as_slice() else {
        return Err(DirectMessageCodecError::InvalidEnvelope(
            "gift-wrap recipient tag must contain two fields",
        ));
    };
    if name != "p" {
        return Err(DirectMessageCodecError::InvalidEnvelope(
            "gift wrap must contain only a recipient tag",
        ));
    }
    PublicKey::from_hex(value).map_err(|_| {
        DirectMessageCodecError::InvalidEnvelope("gift-wrap recipient is not a public key")
    })
}

fn invalid_envelope() -> DirectMessageCodecError {
    DirectMessageCodecError::InvalidEnvelope("invalid NIP-44 v2 ciphertext")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventId, EventSignature};
    use secp256k1::{Keypair, Message, SecretKey};
    use std::collections::BTreeMap;

    const EPHEMERAL_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 1;
        secret
    };
    const EPHEMERAL_AUTHOR: &str =
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const RECIPIENT: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";
    const OTHER_RECIPIENT: &str =
        "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9";

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("fixture public key")
    }

    fn ciphertext_with_marker(marker: &[u8]) -> Nip44Ciphertext {
        let mut bytes = vec![0; MIN_NIP44_CIPHERTEXT_BYTES];
        bytes[0] = NIP44_VERSION;
        let end = 1 + marker.len();
        bytes[1..end].copy_from_slice(marker);
        Nip44Ciphertext::parse(STANDARD.encode(bytes)).expect("fixture ciphertext")
    }

    fn sign(event: CanonicalEvent) -> SignedEvent {
        let claimed_id = event.event_id().expect("event id");
        let secret = SecretKey::from_slice(&EPHEMERAL_SECRET).expect("fixture secret");
        let keypair = Keypair::from_secret_key(&secp256k1::Secp256k1::new(), &secret);
        let signature = secp256k1::Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    fn signed_wrap() -> SignedEvent {
        let envelope = GiftWrapEnvelope::new(
            key(EPHEMERAL_AUTHOR),
            1_756_000_000,
            key(RECIPIENT),
            ciphertext_with_marker(b"private message"),
        );
        sign(envelope.into_canonical_event())
    }

    #[test]
    fn gift_wrap_round_trips_the_opaque_wire_envelope() {
        let signed = signed_wrap();
        let parsed =
            GiftWrapEnvelope::parse_signed_event(&signed, key(RECIPIENT)).expect("valid gift wrap");

        assert_eq!(parsed.ephemeral_author(), key(EPHEMERAL_AUTHOR));
        assert_eq!(parsed.recipient(), key(RECIPIENT));
        assert_eq!(parsed.to_canonical_event(), signed.event);
        assert_eq!(parsed.indexing(), DirectMessageIndexing::Excluded);
    }

    #[test]
    fn gift_wrap_rejects_the_wrong_recipient_at_filter_and_result_gates() {
        let signed = signed_wrap();
        assert_eq!(
            GiftWrapEnvelope::parse_signed_event(&signed, key(OTHER_RECIPIENT)),
            Err(DirectMessageCodecError::WrongRecipient)
        );

        let parsed =
            GiftWrapEnvelope::parse_signed_event(&signed, key(RECIPIENT)).expect("valid gift wrap");
        assert!(!parsed.is_visible_to(key(OTHER_RECIPIENT)));

        let mut matching_filter = EventFilter {
            kinds: vec![KIND_GIFT_WRAP as u16],
            generic_tags: BTreeMap::from([('p', vec![RECIPIENT.into()])]),
            ..EventFilter::default()
        };
        assert!(authorize_gift_wrap_filters(&[matching_filter.clone()], key(RECIPIENT)).is_ok());
        matching_filter
            .generic_tags
            .insert('p', vec![OTHER_RECIPIENT.into()]);
        assert_eq!(
            authorize_gift_wrap_filters(&[matching_filter], key(RECIPIENT)),
            Err(DirectMessageCodecError::UnauthorizedFilter)
        );
        assert_eq!(
            authorize_gift_wrap_filters(&[EventFilter::default()], key(RECIPIENT)),
            Err(DirectMessageCodecError::UnauthorizedFilter)
        );
    }

    #[test]
    fn gift_wrap_rejects_malformed_outer_and_ciphertext_shapes() {
        let mut signed = signed_wrap();
        signed.event.tags.push(vec!["p".into(), RECIPIENT.into()]);
        signed = sign(signed.event);
        assert!(matches!(
            GiftWrapEnvelope::parse_signed_event(&signed, key(RECIPIENT)),
            Err(DirectMessageCodecError::InvalidEnvelope(_))
        ));

        let mut wrong_version = vec![0; MIN_NIP44_CIPHERTEXT_BYTES];
        wrong_version[0] = 1;
        assert!(Nip44Ciphertext::parse(STANDARD.encode(wrong_version)).is_err());
        assert!(Nip44Ciphertext::parse("not-base64").is_err());
    }

    #[test]
    fn direct_message_plaintext_and_ciphertext_are_redacted_from_debug_output() {
        let signed = signed_wrap();
        let parsed =
            GiftWrapEnvelope::parse_signed_event(&signed, key(RECIPIENT)).expect("valid gift wrap");
        let envelope_debug = format!("{parsed:?}");
        let ciphertext_debug = format!("{:?}", parsed.ciphertext());

        for output in [&envelope_debug, &ciphertext_debug] {
            assert!(!output.contains("private message"));
            assert!(!output.contains(parsed.ciphertext().wire_value()));
            assert!(output.contains("redacted"));
        }
    }

    #[test]
    fn invalid_signature_is_rejected_before_envelope_fields_are_used() {
        let mut signed = signed_wrap();
        signed.claimed_id = EventId::from_bytes([0; 32]);
        assert_eq!(
            GiftWrapEnvelope::parse_signed_event(&signed, key(RECIPIENT)),
            Err(DirectMessageCodecError::InvalidEnvelope(
                "invalid signed gift wrap"
            ))
        );
    }
}
