use crate::{CanonicalEvent, EventId, PublicKey};
use std::collections::BTreeMap;

pub const MAX_FILTERS_PER_REQUEST: usize = 10;
pub const MAX_FILTER_VALUES: usize = 1_024;
pub const MAX_GENERIC_TAGS: usize = 64;
pub const MAX_TAG_VALUE_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum FilterError {
    #[error("request contains {actual} filters, maximum is {maximum}")]
    TooManyFilters { actual: usize, maximum: usize },
    #[error("filter field {field} contains {actual} values, maximum is {maximum}")]
    TooManyValues {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("filter contains {actual} generic tags, maximum is {maximum}")]
    TooManyGenericTags { actual: usize, maximum: usize },
    #[error("{field} prefix must contain 1..=64 lowercase hexadecimal characters")]
    InvalidHexPrefix { field: &'static str },
    #[error("generic tag key must be one ASCII letter")]
    InvalidTagKey,
    #[error("generic tag value is {actual} bytes, maximum is {maximum}")]
    TagValueTooLong { actual: usize, maximum: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HexPrefix(String);

impl HexPrefix {
    pub fn new(field: &'static str, value: impl Into<String>) -> Result<Self, FilterError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(FilterError::InvalidHexPrefix { field });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EventFilter {
    pub ids: Vec<HexPrefix>,
    pub authors: Vec<PublicKey>,
    pub kinds: Vec<u16>,
    pub since: Option<u64>,
    pub until: Option<u64>,
    pub generic_tags: BTreeMap<char, Vec<String>>,
}

impl EventFilter {
    pub fn validate(&self) -> Result<(), FilterError> {
        validate_count("ids", self.ids.len())?;
        validate_count("authors", self.authors.len())?;
        validate_count("kinds", self.kinds.len())?;
        if self.generic_tags.len() > MAX_GENERIC_TAGS {
            return Err(FilterError::TooManyGenericTags {
                actual: self.generic_tags.len(),
                maximum: MAX_GENERIC_TAGS,
            });
        }
        for (key, values) in &self.generic_tags {
            if !key.is_ascii_alphabetic() {
                return Err(FilterError::InvalidTagKey);
            }
            validate_count("generic_tag", values.len())?;
            if let Some(value) = values
                .iter()
                .find(|value| value.len() > MAX_TAG_VALUE_BYTES)
            {
                return Err(FilterError::TagValueTooLong {
                    actual: value.len(),
                    maximum: MAX_TAG_VALUE_BYTES,
                });
            }
        }
        Ok(())
    }
}

fn validate_count(field: &'static str, actual: usize) -> Result<(), FilterError> {
    if actual > MAX_FILTER_VALUES {
        return Err(FilterError::TooManyValues {
            field,
            actual,
            maximum: MAX_FILTER_VALUES,
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct FilterEvent<'a> {
    pub id: EventId,
    pub event: &'a CanonicalEvent,
    pub stored_channel_id: Option<&'a str>,
}

pub fn filters_match(filters: &[EventFilter], event: FilterEvent<'_>) -> Result<bool, FilterError> {
    if filters.len() > MAX_FILTERS_PER_REQUEST {
        return Err(FilterError::TooManyFilters {
            actual: filters.len(),
            maximum: MAX_FILTERS_PER_REQUEST,
        });
    }
    for filter in filters {
        filter.validate()?;
    }
    Ok(filters.iter().any(|filter| filter_matches(filter, event)))
}

fn filter_matches(filter: &EventFilter, candidate: FilterEvent<'_>) -> bool {
    if !filter.ids.is_empty() {
        let event_id = candidate.id.to_hex();
        if !filter
            .ids
            .iter()
            .any(|prefix| event_id.starts_with(prefix.as_str()))
        {
            return false;
        }
    }
    if !filter.authors.is_empty() && !filter.authors.contains(&candidate.event.public_key) {
        return false;
    }
    if !filter.kinds.is_empty() && !filter.kinds.contains(&candidate.event.kind) {
        return false;
    }
    if filter
        .since
        .is_some_and(|since| candidate.event.created_at < since)
    {
        return false;
    }
    if filter
        .until
        .is_some_and(|until| candidate.event.created_at > until)
    {
        return false;
    }

    filter.generic_tags.iter().all(|(key, values)| {
        let mut matching_event_values = candidate.event.tags.iter().filter_map(|tag| {
            if tag.first().is_some_and(|name| name == &key.to_string()) {
                tag.get(1).map(String::as_str)
            } else {
                None
            }
        });
        if matching_event_values.any(|event_value| values.iter().any(|value| value == event_value))
        {
            return true;
        }
        *key == 'h'
            && !candidate
                .event
                .tags
                .iter()
                .any(|tag| tag.first().is_some_and(|name| name == "h"))
            && candidate
                .stored_channel_id
                .is_some_and(|channel_id| values.iter().any(|value| value == channel_id))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(tags: Vec<Vec<String>>) -> CanonicalEvent {
        CanonicalEvent::new(
            PublicKey::from_hex(&"01".repeat(32)).expect("public key"),
            100,
            9,
            tags,
            "hello".into(),
        )
    }

    #[test]
    fn filter_matches_and_or_prefix_time_and_tag_semantics() {
        let event = event(vec![vec!["p".into(), "alice".into()]]);
        let event_id = event.event_id().expect("event id");
        let mut matching = EventFilter {
            ids: vec![HexPrefix::new("ids", &event_id.to_hex()[..12]).expect("prefix")],
            authors: vec![event.public_key],
            kinds: vec![9],
            since: Some(100),
            until: Some(100),
            ..EventFilter::default()
        };
        matching.generic_tags.insert('p', vec!["alice".into()]);
        let miss = EventFilter {
            kinds: vec![1],
            ..EventFilter::default()
        };
        let candidate = FilterEvent {
            id: event_id,
            event: &event,
            stored_channel_id: None,
        };

        assert!(filters_match(&[miss, matching], candidate).expect("valid filters"));
        assert!(!filters_match(&[], candidate).expect("empty OR set"));
    }

    #[test]
    fn filter_h_fallback_never_overrides_explicit_tag() {
        let no_h = event(Vec::new());
        let explicit_h = event(vec![vec!["h".into(), "other".into()]]);
        let mut filter = EventFilter::default();
        filter.generic_tags.insert('h', vec!["channel-a".into()]);

        assert!(
            filters_match(
                std::slice::from_ref(&filter),
                FilterEvent {
                    id: no_h.event_id().expect("id"),
                    event: &no_h,
                    stored_channel_id: Some("channel-a"),
                }
            )
            .expect("valid filter")
        );
        assert!(
            !filters_match(
                &[filter],
                FilterEvent {
                    id: explicit_h.event_id().expect("id"),
                    event: &explicit_h,
                    stored_channel_id: Some("channel-a"),
                }
            )
            .expect("valid filter")
        );
    }

    #[test]
    fn filter_rejects_invalid_and_excessive_limits() {
        assert!(HexPrefix::new("ids", "").is_err());
        assert!(HexPrefix::new("ids", "AB").is_err());
        assert_eq!(
            filters_match(
                &vec![EventFilter::default(); MAX_FILTERS_PER_REQUEST + 1],
                FilterEvent {
                    id: EventId::from_bytes([0; 32]),
                    event: &event(Vec::new()),
                    stored_channel_id: None,
                }
            ),
            Err(FilterError::TooManyFilters {
                actual: MAX_FILTERS_PER_REQUEST + 1,
                maximum: MAX_FILTERS_PER_REQUEST,
            })
        );
        let oversized = EventFilter {
            kinds: vec![1; MAX_FILTER_VALUES + 1],
            ..EventFilter::default()
        };
        assert!(matches!(
            oversized.validate(),
            Err(FilterError::TooManyValues { field: "kinds", .. })
        ));
    }
}
