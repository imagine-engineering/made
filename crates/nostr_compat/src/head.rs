use crate::{CanonicalEvent, EventId, PublicKey};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistenceClass {
    Regular,
    Replaceable,
    Ephemeral,
    ParameterizedReplaceable,
}

pub const fn persistence_class(kind: u16) -> PersistenceClass {
    match kind {
        0 | 3 | 41 | 10_000..=19_999 => PersistenceClass::Replaceable,
        20_000..=29_999 => PersistenceClass::Ephemeral,
        30_000..=39_999 => PersistenceClass::ParameterizedReplaceable,
        _ => PersistenceClass::Regular,
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReplacementCoordinate {
    pub kind: u16,
    pub author: PublicKey,
    pub discriminator: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HeadError {
    #[error("event kind {0} is not replaceable")]
    NotReplaceable(u16),
    #[error("parameterized replaceable event must contain exactly one d tag")]
    InvalidDiscriminator,
    #[error("head candidates do not share one replacement coordinate")]
    MixedCoordinates,
}

pub fn replacement_coordinate(event: &CanonicalEvent) -> Result<ReplacementCoordinate, HeadError> {
    let discriminator = match persistence_class(event.kind) {
        PersistenceClass::Replaceable => None,
        PersistenceClass::ParameterizedReplaceable => {
            let mut values = event.tags.iter().filter_map(|tag| {
                if tag.first().is_some_and(|name| name == "d") {
                    tag.get(1).cloned()
                } else {
                    None
                }
            });
            let Some(value) = values.next() else {
                return Err(HeadError::InvalidDiscriminator);
            };
            if values.next().is_some() {
                return Err(HeadError::InvalidDiscriminator);
            }
            Some(value)
        }
        PersistenceClass::Regular | PersistenceClass::Ephemeral => {
            return Err(HeadError::NotReplaceable(event.kind));
        }
    };
    Ok(ReplacementCoordinate {
        kind: event.kind,
        author: event.public_key,
        discriminator,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct HeadCandidate<'a> {
    pub id: EventId,
    pub event: &'a CanonicalEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadOrder {
    pub created_at: u64,
    pub id: EventId,
}

impl HeadOrder {
    pub const fn from_candidate(candidate: HeadCandidate<'_>) -> Self {
        Self {
            created_at: candidate.event.created_at,
            id: candidate.id,
        }
    }

    pub fn accepts(self, candidate: HeadCandidate<'_>) -> bool {
        candidate.event.created_at > self.created_at
            || (candidate.event.created_at == self.created_at && candidate.id < self.id)
    }
}

pub fn select_head<'event>(
    candidates: &[HeadCandidate<'event>],
) -> Result<Option<HeadCandidate<'event>>, HeadError> {
    let Some(first) = candidates.first() else {
        return Ok(None);
    };
    let coordinate = replacement_coordinate(first.event)?;
    let mut head = *first;
    for candidate in &candidates[1..] {
        if replacement_coordinate(candidate.event)? != coordinate {
            return Err(HeadError::MixedCoordinates);
        }
        if is_newer(*candidate, head) {
            head = *candidate;
        }
    }
    Ok(Some(head))
}

fn is_newer(candidate: HeadCandidate<'_>, current: HeadCandidate<'_>) -> bool {
    HeadOrder::from_candidate(current).accepts(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::Value;

    const EVENTS: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
    ));

    fn profile(name: &str) -> (CanonicalEvent, EventId) {
        let fixture: Value = serde_json::from_str(EVENTS).expect("valid frozen corpus");
        let value = &fixture["events"][name];
        (
            CanonicalEvent::new(
                PublicKey::from_hex(value["pubkey"].as_str().expect("pubkey")).expect("public key"),
                value["created_at"].as_u64().expect("created_at"),
                u16::try_from(value["kind"].as_u64().expect("kind")).expect("kind"),
                serde_json::from_value(value["tags"].clone()).expect("tags"),
                value["content"].as_str().expect("content").to_owned(),
            ),
            EventId::from_hex(value["id"].as_str().expect("id")).expect("event id"),
        )
    }

    #[test]
    fn head_matches_frozen_timestamp_and_lowest_id_cases() {
        let (old, old_id) = profile("profile_old");
        let (new, new_id) = profile("profile_new");
        let latest = select_head(&[
            HeadCandidate {
                id: old_id,
                event: &old,
            },
            HeadCandidate {
                id: new_id,
                event: &new,
            },
        ])
        .expect("same coordinate")
        .expect("head");
        assert_eq!(latest.id, new_id);

        let (tie_a, tie_a_id) = profile("profile_tie_a");
        let (tie_b, tie_b_id) = profile("profile_tie_b");
        let tied = select_head(&[
            HeadCandidate {
                id: tie_b_id,
                event: &tie_b,
            },
            HeadCandidate {
                id: tie_a_id,
                event: &tie_a,
            },
        ])
        .expect("same coordinate")
        .expect("head");
        assert_eq!(tied.id, tie_a_id);
    }

    #[test]
    fn head_requires_exactly_one_parameterized_discriminator() {
        let public_key = PublicKey::from_hex(&"01".repeat(32)).expect("public key");
        for tags in [
            Vec::new(),
            vec![
                vec!["d".into(), "one".into()],
                vec!["d".into(), "two".into()],
            ],
        ] {
            let event = CanonicalEvent::new(public_key, 1, 30_000, tags, String::new());
            assert_eq!(
                replacement_coordinate(&event),
                Err(HeadError::InvalidDiscriminator)
            );
        }
    }

    #[test]
    fn head_delete_floor_prevents_stale_resurrection() {
        let (old, old_id) = profile("profile_old");
        let (deleted_head, deleted_head_id) = profile("profile_new");
        let floor = HeadOrder::from_candidate(HeadCandidate {
            id: deleted_head_id,
            event: &deleted_head,
        });
        let (tie_lower_id, tie_lower_id_value) = profile("profile_tie_a");
        let same_second_lower_id = CanonicalEvent::new(
            deleted_head.public_key,
            deleted_head.created_at,
            deleted_head.kind,
            deleted_head.tags.clone(),
            tie_lower_id.content,
        );
        let newer = CanonicalEvent::new(
            deleted_head.public_key,
            deleted_head.created_at + 1,
            deleted_head.kind,
            deleted_head.tags,
            "newer".into(),
        );

        assert!(!floor.accepts(HeadCandidate {
            id: old_id,
            event: &old,
        }));
        assert!(floor.accepts(HeadCandidate {
            id: tie_lower_id_value,
            event: &same_second_lower_id,
        }));
        assert!(floor.accepts(HeadCandidate {
            id: EventId::from_bytes([0xff; 32]),
            event: &newer,
        }));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn head_selection_is_permutation_invariant(
            first_timestamp in any::<u64>(),
            second_timestamp in any::<u64>(),
            first_id in any::<[u8; 32]>(),
            second_id in any::<[u8; 32]>(),
        ) {
            let public_key = PublicKey::from_bytes([1; 32]);
            let first = CanonicalEvent::new(public_key, first_timestamp, 0, Vec::new(), "first".into());
            let second = CanonicalEvent::new(public_key, second_timestamp, 0, Vec::new(), "second".into());
            let first_candidate = HeadCandidate { id: EventId::from_bytes(first_id), event: &first };
            let second_candidate = HeadCandidate { id: EventId::from_bytes(second_id), event: &second };
            let forward = select_head(&[first_candidate, second_candidate]).expect("same coordinate").expect("head");
            let reverse = select_head(&[second_candidate, first_candidate]).expect("same coordinate").expect("head");
            prop_assert_eq!(forward.id, reverse.id);
            let expected = if first_timestamp > second_timestamp
                || (first_timestamp == second_timestamp && first_id < second_id)
            {
                first_id
            } else {
                second_id
            };
            prop_assert_eq!(*forward.id.as_bytes(), expected);
        }
    }
}
