use serde::{Deserialize, Deserializer, Serialize, de};
use std::{fmt, num::NonZeroU64};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AggregateVersion(NonZeroU64);

impl AggregateVersion {
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub fn next(self) -> Option<Self> {
        self.get().checked_add(1).and_then(Self::new)
    }

    pub fn follows(self, previous: Self) -> bool {
        previous.next() == Some(self)
    }
}

impl fmt::Display for AggregateVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSystem {
    Zed,
    Buzz,
    Nostr,
    Acp,
    ExternalGit,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SourceRecordId(String);

impl SourceRecordId {
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        if value.is_empty() || value.len() > 1024 {
            return None;
        }
        Some(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SourceRecordId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| de::Error::custom("source record id must be 1..=1024 bytes"))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrityAlgorithm {
    Sha256,
    NostrEventId,
    GitObjectId,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct IntegrityReference {
    pub algorithm: IntegrityAlgorithm,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Provenance {
    pub source_system: SourceSystem,
    pub source_record_id: SourceRecordId,
    pub source_version: Option<String>,
    pub observed_at_millis: u64,
    pub integrity: Option<IntegrityReference>,
}

impl Provenance {
    pub fn new(
        source_system: SourceSystem,
        source_record_id: SourceRecordId,
        observed_at_millis: u64,
    ) -> Self {
        Self {
            source_system,
            source_record_id,
            source_version: None,
            observed_at_millis,
            integrity: None,
        }
    }

    pub fn with_source_version(mut self, source_version: impl Into<String>) -> Self {
        self.source_version = Some(source_version.into());
        self
    }

    pub fn with_integrity(mut self, integrity: IntegrityReference) -> Self {
        self.integrity = Some(integrity);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_versions_are_strictly_ordered() {
        let first = AggregateVersion::FIRST;
        let second = first.next().expect("first version has a successor");

        assert!(second > first);
        assert!(second.follows(first));
        assert!(!first.follows(second));
        assert_eq!(AggregateVersion::new(0), None);
        assert_eq!(
            AggregateVersion::new(u64::MAX).and_then(|version| version.next()),
            None
        );
    }

    #[test]
    fn provenance_round_trips_without_losing_source_fields() {
        let provenance = Provenance::new(
            SourceSystem::Buzz,
            SourceRecordId::new("event:abc").expect("valid source id"),
            1_700_000_000_000,
        )
        .with_source_version("42")
        .with_integrity(IntegrityReference {
            algorithm: IntegrityAlgorithm::NostrEventId,
            value: "ab".repeat(32),
        });

        let encoded = serde_json::to_string(&provenance).expect("serialize provenance");
        let decoded: Provenance = serde_json::from_str(&encoded).expect("deserialize provenance");

        assert_eq!(decoded, provenance);
    }

    #[test]
    fn provenance_source_record_id_is_bounded() {
        assert!(SourceRecordId::new("").is_none());
        assert!(SourceRecordId::new("x".repeat(1025)).is_none());
        assert!(serde_json::from_str::<SourceRecordId>("\"\"").is_err());
        assert!(
            serde_json::from_str::<SourceRecordId>(&format!("\"{}\"", "x".repeat(1025))).is_err()
        );
        assert_eq!(
            SourceRecordId::new("record-1").map(|id| id.as_str().to_owned()),
            Some("record-1".to_owned())
        );
    }
}
