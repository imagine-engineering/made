use crate::generated_kinds::KIND_PUSH_LEASE;
use crate::{PublicKey, SignedEvent, TimestampPolicy, verify_signed_event};
use serde::de::{DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::fmt;

pub const MAX_SAFE_JSON_INTEGER: u64 = (1_u64 << 53) - 1;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PushLeaseCodecError {
    #[error("unsupported push-lease kind {0}")]
    UnsupportedKind(u16),
    #[error("invalid push-lease envelope: {0}")]
    InvalidEnvelope(String),
    #[error("invalid push-lease plaintext: {0}")]
    InvalidPlaintext(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushLeaseEnvelope {
    pub author: PublicKey,
    pub installation_id: String,
    pub expiration: u64,
    pub executor_key_id: String,
    pub ciphertext: String,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PriorityClass {
    Silent,
    Default,
    TimeSensitive,
    Urgent,
}

impl PriorityClass {
    fn parse(value: &str) -> Result<Self, PushLeaseCodecError> {
        match value {
            "silent" => Ok(Self::Silent),
            "default" => Ok(Self::Default),
            "time_sensitive" => Ok(Self::TimeSensitive),
            "urgent" => Ok(Self::Urgent),
            _ => Err(invalid_plaintext("unknown priority class")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Silent => "silent",
            Self::Default => "default",
            Self::TimeSensitive => "time_sensitive",
            Self::Urgent => "urgent",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeaseFilter {
    pub kinds: Vec<u16>,
    pub authors: Option<Vec<String>>,
    pub p_tags: Option<Vec<String>>,
    pub h_tags: Option<Vec<String>>,
    pub e_tags: Option<Vec<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeaseSubscription {
    pub filter: LeaseFilter,
    pub class: PriorityClass,
    pub ignore: Vec<LeaseFilter>,
    pub p_tags_max: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivePushLease {
    pub origin: String,
    pub app_profile: String,
    pub transport: String,
    pub endpoint: String,
    pub generation: u64,
    pub subscriptions: Vec<LeaseSubscription>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InactivePushLease {
    pub origin: String,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PushLeasePlaintext {
    Active(ActivePushLease),
    Inactive(InactivePushLease),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppProfile<'a> {
    pub id: &'a str,
    pub transport: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelGrammar {
    UuidV4Lowercase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushLeaseLimits {
    pub max_content_len: usize,
    pub max_plaintext_len: usize,
    pub max_lease_ttl: u64,
    pub allowed_skew: u64,
    pub max_subscriptions: usize,
    pub max_kinds: usize,
    pub max_authors: usize,
    pub max_h: usize,
    pub max_tag_values: usize,
    pub max_ignore: usize,
    pub max_endpoint_len: usize,
    pub max_string_len: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct PushLeasePolicy<'a> {
    pub expected_origin: &'a str,
    pub app_profiles: &'a [AppProfile<'a>],
    pub supported_classes: &'a [PriorityClass],
    pub push_kinds: &'a [u16],
    pub urgent_kinds: &'a [u16],
    pub channel_grammar: ChannelGrammar,
    pub limits: PushLeaseLimits,
}

impl PushLeaseEnvelope {
    pub fn parse_signed_event(
        event: &SignedEvent,
        authenticated_author: PublicKey,
        now: u64,
        limits: PushLeaseLimits,
    ) -> Result<Self, PushLeaseCodecError> {
        verify_signed_event(event, TimestampPolicy::Historical)
            .map_err(|error| invalid_envelope(format!("invalid signed event: {error}")))?;
        if u32::from(event.event.kind) != KIND_PUSH_LEASE {
            return Err(PushLeaseCodecError::UnsupportedKind(event.event.kind));
        }
        if event.event.public_key != authenticated_author {
            return Err(invalid_envelope(
                "event author does not match authenticated user",
            ));
        }
        if event.event.content.len() > limits.max_content_len {
            return Err(invalid_envelope("ciphertext exceeds descriptor limit"));
        }
        let mut d_tag = None;
        let mut expiration_tag = None;
        let mut executor_tag = None;
        let mut alt_seen = false;
        for tag in &event.event.tags {
            if tag.len() != 2 {
                return Err(invalid_envelope("public tag must have exactly one value"));
            }
            match tag[0].as_str() {
                "d" if d_tag.is_none() => d_tag = Some(tag[1].clone()),
                "expiration" if expiration_tag.is_none() => {
                    expiration_tag = Some(tag[1].clone());
                }
                "exec" if executor_tag.is_none() => executor_tag = Some(tag[1].clone()),
                "alt" if !alt_seen => alt_seen = true,
                "d" | "expiration" | "exec" | "alt" => {
                    return Err(invalid_envelope(format!("duplicate {} tag", tag[0])));
                }
                name => {
                    return Err(invalid_envelope(format!("unexpected public tag {name:?}")));
                }
            }
        }
        let installation_id = d_tag
            .filter(|value| !value.is_empty() && value.len() <= 64)
            .ok_or_else(|| invalid_envelope("d tag must contain 1-64 bytes"))?;
        let expiration = parse_canonical_decimal(
            expiration_tag
                .as_deref()
                .ok_or_else(|| invalid_envelope("missing expiration tag"))?,
            u64::MAX,
        )?;
        let lower_bound = now.saturating_sub(limits.allowed_skew);
        if expiration <= lower_bound {
            return Err(invalid_envelope("lease already expired"));
        }
        if expiration > now.saturating_add(limits.max_lease_ttl) {
            return Err(invalid_envelope("lease ttl too long"));
        }
        let executor_key_id = executor_tag
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_envelope("missing or empty exec tag"))?;
        Ok(Self {
            author: authenticated_author,
            installation_id,
            expiration,
            executor_key_id,
            ciphertext: event.event.content.clone(),
        })
    }

    pub fn to_tags(&self) -> Vec<Vec<String>> {
        vec![
            vec!["d".into(), self.installation_id.clone()],
            vec!["expiration".into(), self.expiration.to_string()],
            vec!["exec".into(), self.executor_key_id.clone()],
            vec!["alt".into(), "Push lease".into()],
        ]
    }

    pub fn validate_decrypted(
        &self,
        plaintext: &[u8],
        policy: &PushLeasePolicy<'_>,
    ) -> Result<PushLeasePlaintext, PushLeaseCodecError> {
        PushLeasePlaintext::parse(plaintext, self.author, policy)
    }
}

impl PushLeasePlaintext {
    pub fn parse(
        plaintext: &[u8],
        author: PublicKey,
        policy: &PushLeasePolicy<'_>,
    ) -> Result<Self, PushLeaseCodecError> {
        if plaintext.len() > policy.limits.max_plaintext_len {
            return Err(invalid_plaintext("plaintext exceeds descriptor limit"));
        }
        let value = parse_strict_json(plaintext)?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_plaintext("lease plaintext must be an object"))?;
        let active = object
            .get("active")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid_plaintext("active must be a boolean"))?;
        let required = if active {
            &[
                "v",
                "origin",
                "app_profile",
                "transport",
                "endpoint",
                "generation",
                "active",
                "subscriptions",
            ][..]
        } else {
            &["v", "origin", "generation", "active"][..]
        };
        if object.len() != required.len() || required.iter().any(|key| !object.contains_key(*key)) {
            return Err(invalid_plaintext(
                "lease fields do not match the active-state schema",
            ));
        }
        if object.get("v").and_then(Value::as_u64) != Some(1) {
            return Err(invalid_plaintext("v must be integer 1"));
        }
        let origin = required_string(object, "origin")?;
        check_string(origin, policy.limits.max_string_len, "origin")?;
        if origin != policy.expected_origin {
            return Err(invalid_plaintext("origin mismatch"));
        }
        let generation = object
            .get("generation")
            .and_then(Value::as_u64)
            .filter(|value| (1..=MAX_SAFE_JSON_INTEGER).contains(value))
            .ok_or_else(|| invalid_plaintext("generation must be a positive safe integer"))?;
        if !active {
            return Ok(Self::Inactive(InactivePushLease {
                origin: origin.into(),
                generation,
            }));
        }
        let app_profile = required_string(object, "app_profile")?;
        let transport = required_string(object, "transport")?;
        let endpoint = required_string(object, "endpoint")?;
        check_string(app_profile, policy.limits.max_string_len, "app_profile")?;
        check_string(transport, policy.limits.max_string_len, "transport")?;
        check_string(endpoint, policy.limits.max_endpoint_len, "endpoint")?;
        let advertised = policy
            .app_profiles
            .iter()
            .find(|profile| profile.id == app_profile)
            .ok_or_else(|| invalid_plaintext("app profile not supported"))?;
        if advertised.transport != transport {
            return Err(invalid_plaintext("transport mismatch"));
        }
        let subscription_values = object
            .get("subscriptions")
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty() && values.len() <= policy.limits.max_subscriptions)
            .ok_or_else(|| invalid_plaintext("subscription quota exceeded"))?;
        let subscriptions = subscription_values
            .iter()
            .map(|value| LeaseSubscription::parse(value, author, policy))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::Active(ActivePushLease {
            origin: origin.into(),
            app_profile: app_profile.into(),
            transport: transport.into(),
            endpoint: endpoint.into(),
            generation,
            subscriptions,
        }))
    }

    pub fn to_plaintext(&self) -> String {
        match self {
            Self::Inactive(lease) => json!({
                "v": 1,
                "origin": lease.origin,
                "generation": lease.generation,
                "active": false,
            })
            .to_string(),
            Self::Active(lease) => json!({
                "v": 1,
                "origin": lease.origin,
                "app_profile": lease.app_profile,
                "transport": lease.transport,
                "endpoint": lease.endpoint,
                "generation": lease.generation,
                "active": true,
                "subscriptions": lease.subscriptions.iter().map(LeaseSubscription::to_value).collect::<Vec<_>>(),
            })
            .to_string(),
        }
    }
}

impl LeaseSubscription {
    fn parse(
        value: &Value,
        author: PublicKey,
        policy: &PushLeasePolicy<'_>,
    ) -> Result<Self, PushLeaseCodecError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid_plaintext("subscription must be an object"))?;
        let allowed = ["filter", "class", "ignore", "suppress"];
        if object.keys().any(|key| !allowed.contains(&key.as_str()))
            || !object.contains_key("filter")
            || !object.contains_key("class")
        {
            return Err(invalid_plaintext("invalid subscription fields"));
        }
        let class = PriorityClass::parse(required_string(object, "class")?)?;
        if !policy.supported_classes.contains(&class) {
            return Err(invalid_plaintext("class not supported"));
        }
        let filter = LeaseFilter::parse(
            object
                .get("filter")
                .ok_or_else(|| invalid_plaintext("missing filter"))?,
            author,
            policy,
            true,
            Some(class),
        )?;
        let ignore = match object.get("ignore") {
            None => Vec::new(),
            Some(value) => {
                let values = value
                    .as_array()
                    .filter(|values| values.len() <= policy.limits.max_ignore)
                    .ok_or_else(|| invalid_plaintext("ignore quota exceeded"))?;
                values
                    .iter()
                    .map(|value| LeaseFilter::parse(value, author, policy, false, None))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        let p_tags_max = match object.get("suppress") {
            None => None,
            Some(value) => {
                let suppress = value
                    .as_object()
                    .filter(|object| object.len() == 1 && object.contains_key("p_tags_max"))
                    .ok_or_else(|| invalid_plaintext("invalid suppress fields"))?;
                Some(
                    suppress
                        .get("p_tags_max")
                        .and_then(Value::as_u64)
                        .filter(|value| *value > 0)
                        .ok_or_else(|| invalid_plaintext("p_tags_max must be positive"))?,
                )
            }
        };
        Ok(Self {
            filter,
            class,
            ignore,
            p_tags_max,
        })
    }

    fn to_value(&self) -> Value {
        let mut object = Map::new();
        object.insert("filter".into(), self.filter.to_value());
        object.insert("class".into(), Value::String(self.class.as_str().into()));
        if !self.ignore.is_empty() {
            object.insert(
                "ignore".into(),
                Value::Array(self.ignore.iter().map(LeaseFilter::to_value).collect()),
            );
        }
        if let Some(p_tags_max) = self.p_tags_max {
            object.insert("suppress".into(), json!({"p_tags_max": p_tags_max}));
        }
        Value::Object(object)
    }
}

impl LeaseFilter {
    fn parse(
        value: &Value,
        author: PublicKey,
        policy: &PushLeasePolicy<'_>,
        require_narrowing: bool,
        class: Option<PriorityClass>,
    ) -> Result<Self, PushLeaseCodecError> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid_plaintext("lease filter must be an object"))?;
        let allowed = ["kinds", "authors", "#p", "#h", "#e"];
        if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
            return Err(invalid_plaintext(format!(
                "filter member not permitted: {key}"
            )));
        }
        let kind_values = object
            .get("kinds")
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty() && values.len() <= policy.limits.max_kinds)
            .ok_or_else(|| invalid_plaintext("invalid kinds count"))?;
        let mut kinds = Vec::with_capacity(kind_values.len());
        for value in kind_values {
            let kind = value
                .as_u64()
                .and_then(|value| u16::try_from(value).ok())
                .filter(|value| policy.push_kinds.contains(value))
                .ok_or_else(|| invalid_plaintext("kind not push-eligible"))?;
            kinds.push(kind);
        }
        if class == Some(PriorityClass::Urgent)
            && kinds.iter().any(|kind| !policy.urgent_kinds.contains(kind))
        {
            return Err(invalid_plaintext("class not permitted for kind"));
        }
        let authors = parse_string_array(object, "authors", policy.limits.max_authors)?;
        let p_tags = parse_string_array(object, "#p", policy.limits.max_tag_values)?;
        let h_tags = parse_string_array(object, "#h", policy.limits.max_h)?;
        let e_tags = parse_string_array(object, "#e", policy.limits.max_tag_values)?;
        if require_narrowing && authors.is_none() && p_tags.is_none() && h_tags.is_none() {
            return Err(invalid_plaintext("lease filter not narrowed"));
        }
        for value in authors.iter().flatten() {
            check_exact_hex(value, "author")?;
        }
        let author_hex = author.to_hex();
        for value in p_tags.iter().flatten() {
            check_exact_hex(value, "p tag")?;
            if value != &author_hex {
                return Err(invalid_plaintext("p-tag must be self"));
            }
        }
        for value in h_tags.iter().flatten() {
            check_string(value, policy.limits.max_string_len, "h tag")?;
            match policy.channel_grammar {
                ChannelGrammar::UuidV4Lowercase => {
                    let parsed = uuid::Uuid::parse_str(value)
                        .map_err(|_| invalid_plaintext("invalid h tag"))?;
                    if parsed.get_version_num() != 4 || parsed.to_string() != *value {
                        return Err(invalid_plaintext("invalid h tag"));
                    }
                }
            }
        }
        for value in e_tags.iter().flatten() {
            check_exact_hex(value, "e tag")?;
        }
        Ok(Self {
            kinds,
            authors,
            p_tags,
            h_tags,
            e_tags,
        })
    }

    fn to_value(&self) -> Value {
        let mut object = Map::new();
        object.insert("kinds".into(), json!(self.kinds));
        if let Some(authors) = &self.authors {
            object.insert("authors".into(), json!(authors));
        }
        if let Some(p_tags) = &self.p_tags {
            object.insert("#p".into(), json!(p_tags));
        }
        if let Some(h_tags) = &self.h_tags {
            object.insert("#h".into(), json!(h_tags));
        }
        if let Some(e_tags) = &self.e_tags {
            object.insert("#e".into(), json!(e_tags));
        }
        Value::Object(object)
    }
}

fn parse_string_array(
    object: &Map<String, Value>,
    key: &str,
    maximum: usize,
) -> Result<Option<Vec<String>>, PushLeaseCodecError> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .filter(|values| !values.is_empty() && values.len() <= maximum)
        .ok_or_else(|| invalid_plaintext(format!("invalid {key} count")))?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid_plaintext(format!("{key} values must be strings")))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a str, PushLeaseCodecError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_plaintext(format!("{key} must be a string")))
}

fn parse_canonical_decimal(value: &str, maximum: u64) -> Result<u64, PushLeaseCodecError> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_digit())
        || (value.len() > 1 && value.starts_with('0'))
    {
        return Err(invalid_envelope("timestamp is not a canonical decimal"));
    }
    value
        .parse::<u64>()
        .ok()
        .filter(|value| *value <= maximum)
        .ok_or_else(|| invalid_envelope("timestamp is out of range"))
}

fn check_exact_hex(value: &str, label: &str) -> Result<(), PushLeaseCodecError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_plaintext(format!(
            "non-exact match value for {label}"
        )));
    }
    Ok(())
}

fn check_string(value: &str, maximum: usize, label: &str) -> Result<(), PushLeaseCodecError> {
    if value.is_empty() || value.len() > maximum {
        return Err(invalid_plaintext(format!("invalid {label} length")));
    }
    Ok(())
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, PushLeaseCodecError> {
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
        .map_err(|error| invalid_plaintext(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| invalid_plaintext(error.to_string()))?;
    Ok(value)
}

fn invalid_envelope(reason: impl Into<String>) -> PushLeaseCodecError {
    PushLeaseCodecError::InvalidEnvelope(reason.into())
}

fn invalid_plaintext(reason: impl Into<String>) -> PushLeaseCodecError {
    PushLeaseCodecError::InvalidPlaintext(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalEvent, EventSignature};
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    const AUTHOR_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 1;
        secret
    };
    const OTHER_SECRET: [u8; 32] = {
        let mut secret = [0; 32];
        secret[31] = 2;
        secret
    };
    const AUTHOR: &str = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    const OTHER: &str = "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5";
    const CHANNEL: &str = "3580ca9b-47b4-4af9-b22a-1068778f26c6";
    const PROFILES: &[AppProfile<'static>] = &[
        AppProfile {
            id: "buzz-ios-production",
            transport: "apns",
        },
        AppProfile {
            id: "buzz-ios-sandbox",
            transport: "apns",
        },
    ];
    const CLASSES: &[PriorityClass] = &[
        PriorityClass::Silent,
        PriorityClass::Default,
        PriorityClass::TimeSensitive,
        PriorityClass::Urgent,
    ];
    const PUSH_KINDS: &[u16] = &[7, 9, 1059, 40007, 46010];
    const URGENT_KINDS: &[u16] = &[46010];

    fn key(value: &str) -> PublicKey {
        PublicKey::from_hex(value).expect("fixture public key")
    }

    fn limits() -> PushLeaseLimits {
        PushLeaseLimits {
            max_content_len: 65_536,
            max_plaintext_len: 32_768,
            max_lease_ttl: 2_592_000,
            allowed_skew: 900,
            max_subscriptions: 16,
            max_kinds: 16,
            max_authors: 20,
            max_h: 50,
            max_tag_values: 20,
            max_ignore: 8,
            max_endpoint_len: 4_096,
            max_string_len: 512,
        }
    }

    fn policy() -> PushLeasePolicy<'static> {
        PushLeasePolicy {
            expected_origin: "wss://relay.example",
            app_profiles: PROFILES,
            supported_classes: CLASSES,
            push_kinds: PUSH_KINDS,
            urgent_kinds: URGENT_KINDS,
            channel_grammar: ChannelGrammar::UuidV4Lowercase,
            limits: limits(),
        }
    }

    fn sign(event: CanonicalEvent, secret: [u8; 32]) -> SignedEvent {
        let claimed_id = event.event_id().expect("event id");
        let secret = SecretKey::from_slice(&secret).expect("secret");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret);
        let signature = Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
        SignedEvent {
            claimed_id,
            event,
            signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
        }
    }

    fn envelope_event(tags: Vec<Vec<String>>) -> SignedEvent {
        sign(
            CanonicalEvent::new(
                key(AUTHOR),
                1_700_000_000,
                KIND_PUSH_LEASE as u16,
                tags,
                "opaque-ciphertext".into(),
            ),
            AUTHOR_SECRET,
        )
    }

    fn valid_tags() -> Vec<Vec<String>> {
        vec![
            vec!["d".into(), "installation-1".into()],
            vec!["expiration".into(), "1700100000".into()],
            vec!["exec".into(), "2026-06".into()],
            vec!["alt".into(), "Push lease".into()],
        ]
    }

    #[test]
    fn envelope_is_signed_author_scoped_expiring_and_public_tag_closed() {
        let event = envelope_event(valid_tags());
        let envelope =
            PushLeaseEnvelope::parse_signed_event(&event, key(AUTHOR), 1_700_000_000, limits())
                .expect("lease envelope");
        assert_eq!(envelope.installation_id, "installation-1");
        assert_eq!(envelope.executor_key_id, "2026-06");
        assert_eq!(envelope.to_tags(), valid_tags());
        assert!(
            PushLeaseEnvelope::parse_signed_event(&event, key(OTHER), 1_700_000_000, limits(),)
                .is_err()
        );
        let mut extra = valid_tags();
        extra.push(vec!["p".into(), AUTHOR.into()]);
        assert!(
            PushLeaseEnvelope::parse_signed_event(
                &envelope_event(extra),
                key(AUTHOR),
                1_700_000_000,
                limits(),
            )
            .is_err()
        );
        let expired = envelope_event(vec![
            vec!["d".into(), "installation-1".into()],
            vec!["expiration".into(), "1699999000".into()],
            vec!["exec".into(), "2026-06".into()],
        ]);
        assert!(
            PushLeaseEnvelope::parse_signed_event(&expired, key(AUTHOR), 1_700_000_000, limits(),)
                .is_err()
        );
    }

    #[test]
    fn active_lease_round_trips_narrow_filters_and_suppression() {
        let plaintext = format!(
            r##"{{"v":1,"origin":"wss://relay.example","app_profile":"buzz-ios-production","transport":"apns","endpoint":"opaque-grant","generation":3,"active":true,"subscriptions":[{{"filter":{{"kinds":[9],"#p":["{AUTHOR}"]}},"class":"time_sensitive"}},{{"filter":{{"kinds":[9],"#h":["{CHANNEL}"]}},"class":"default","ignore":[{{"kinds":[9],"authors":["{OTHER}"]}}],"suppress":{{"p_tags_max":20}}}}]}}"##
        );
        let parsed = PushLeasePlaintext::parse(plaintext.as_bytes(), key(AUTHOR), &policy())
            .expect("active lease");
        let PushLeasePlaintext::Active(active) = &parsed else {
            panic!("active lease")
        };
        assert_eq!(active.generation, 3);
        assert_eq!(active.subscriptions[0].class, PriorityClass::TimeSensitive);
        assert_eq!(active.subscriptions[1].p_tags_max, Some(20));
        assert_eq!(
            PushLeasePlaintext::parse(parsed.to_plaintext().as_bytes(), key(AUTHOR), &policy(),)
                .expect("round trip"),
            parsed
        );
    }

    #[test]
    fn inactive_lease_uses_only_the_minimal_tombstone_schema() {
        let body = br#"{"v":1,"origin":"wss://relay.example","generation":4,"active":false}"#;
        assert_eq!(
            PushLeasePlaintext::parse(body, key(AUTHOR), &policy()).expect("inactive"),
            PushLeasePlaintext::Inactive(InactivePushLease {
                origin: "wss://relay.example".into(),
                generation: 4,
            })
        );
        let nonminimal = br#"{"v":1,"origin":"wss://relay.example","generation":4,"active":false,"endpoint":"stale"}"#;
        assert!(PushLeasePlaintext::parse(nonminimal, key(AUTHOR), &policy()).is_err());
        let zero = br#"{"v":1,"origin":"wss://relay.example","generation":0,"active":false}"#;
        assert!(PushLeasePlaintext::parse(zero, key(AUTHOR), &policy()).is_err());
    }

    #[test]
    fn plaintext_rejects_duplicate_unknown_and_cross_user_filter_state() {
        let duplicate = br#"{"v":1,"origin":"wss://relay.example","generation":1,"generation":2,"active":false}"#;
        assert!(PushLeasePlaintext::parse(duplicate, key(AUTHOR), &policy()).is_err());
        let unknown = br#"{"v":1,"origin":"wss://relay.example","generation":1,"active":false,"future":true}"#;
        assert!(PushLeasePlaintext::parse(unknown, key(AUTHOR), &policy()).is_err());
        let cross_user = format!(
            r##"{{"v":1,"origin":"wss://relay.example","app_profile":"buzz-ios-production","transport":"apns","endpoint":"grant","generation":1,"active":true,"subscriptions":[{{"filter":{{"kinds":[9],"#p":["{OTHER}"]}},"class":"default"}}]}}"##
        );
        assert!(PushLeasePlaintext::parse(cross_user.as_bytes(), key(AUTHOR), &policy()).is_err());
    }

    #[test]
    fn filter_grammar_rejects_time_travel_bad_channels_and_ineligible_urgency() {
        let time_travel = format!(
            r##"{{"v":1,"origin":"wss://relay.example","app_profile":"buzz-ios-production","transport":"apns","endpoint":"grant","generation":1,"active":true,"subscriptions":[{{"filter":{{"kinds":[9],"#p":["{AUTHOR}"],"since":1}},"class":"default"}}]}}"##
        );
        assert!(PushLeasePlaintext::parse(time_travel.as_bytes(), key(AUTHOR), &policy()).is_err());
        let bad_channel = br##"{"v":1,"origin":"wss://relay.example","app_profile":"buzz-ios-production","transport":"apns","endpoint":"grant","generation":1,"active":true,"subscriptions":[{"filter":{"kinds":[9],"#h":["NOT-A-UUID"]},"class":"default"}]}"##;
        assert!(PushLeasePlaintext::parse(bad_channel, key(AUTHOR), &policy()).is_err());
        let urgent_message = format!(
            r##"{{"v":1,"origin":"wss://relay.example","app_profile":"buzz-ios-production","transport":"apns","endpoint":"grant","generation":1,"active":true,"subscriptions":[{{"filter":{{"kinds":[9],"#p":["{AUTHOR}"]}},"class":"urgent"}}]}}"##
        );
        assert!(
            PushLeasePlaintext::parse(urgent_message.as_bytes(), key(AUTHOR), &policy()).is_err()
        );

        let wrong_signer = sign(
            CanonicalEvent::new(
                key(OTHER),
                1_700_000_000,
                KIND_PUSH_LEASE as u16,
                valid_tags(),
                "opaque-ciphertext".into(),
            ),
            OTHER_SECRET,
        );
        assert!(
            PushLeaseEnvelope::parse_signed_event(
                &wrong_signer,
                key(AUTHOR),
                1_700_000_000,
                limits(),
            )
            .is_err()
        );
    }
}
