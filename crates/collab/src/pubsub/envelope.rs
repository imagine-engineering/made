use std::collections::{HashSet, VecDeque};

use collaboration_domain::{CommunityId, Provenance, SourceSystem, TenantContext};
use serde::{Deserialize, Serialize};

pub const CURRENT_FANOUT_CONTRACT_VERSION: u16 = 1;
pub const MAX_FANOUT_ENVELOPE_BYTES: usize = 16 * 1024;
pub const MAX_FANOUT_TOPIC_BYTES: usize = 128;
pub const MAX_LOCAL_DEDUPLICATION_ENTRIES: usize = 65_536;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FanoutEnvelope {
    community_id: CommunityId,
    outbox_sequence: u64,
    topic: String,
    provenance: Provenance,
    payload_sha256: [u8; 32],
}

impl FanoutEnvelope {
    pub fn new(
        community_id: CommunityId,
        outbox_sequence: u64,
        topic: impl Into<String>,
        provenance: Provenance,
        payload_sha256: [u8; 32],
    ) -> Result<Self, FanoutEnvelopeError> {
        let topic = topic.into();
        if outbox_sequence == 0
            || outbox_sequence > i64::MAX as u64
            || !valid_topic(&topic)
            || !valid_provenance(&provenance)
        {
            return Err(FanoutEnvelopeError::InvalidEnvelope);
        }
        Ok(Self {
            community_id,
            outbox_sequence,
            topic,
            provenance,
            payload_sha256,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn outbox_sequence(&self) -> u64 {
        self.outbox_sequence
    }

    pub fn topic(&self) -> &str {
        &self.topic
    }

    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    pub const fn payload_sha256(&self) -> &[u8; 32] {
        &self.payload_sha256
    }

    pub fn encode(&self) -> Result<Vec<u8>, FanoutEnvelopeError> {
        let wire = FanoutEnvelopeWire {
            contract_version: CURRENT_FANOUT_CONTRACT_VERSION,
            community_id: self.community_id,
            outbox_sequence: self.outbox_sequence,
            topic: self.topic.clone(),
            provenance: self.provenance.clone(),
            payload_sha256: hex::encode(self.payload_sha256),
        };
        let encoded =
            serde_json::to_vec(&wire).map_err(|_| FanoutEnvelopeError::InvalidEnvelope)?;
        if encoded.len() > MAX_FANOUT_ENVELOPE_BYTES {
            return Err(FanoutEnvelopeError::EnvelopeTooLarge);
        }
        Ok(encoded)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FanoutEnvelopeError> {
        if bytes.len() > MAX_FANOUT_ENVELOPE_BYTES {
            return Err(FanoutEnvelopeError::EnvelopeTooLarge);
        }
        let wire: FanoutEnvelopeWire =
            serde_json::from_slice(bytes).map_err(|_| FanoutEnvelopeError::InvalidEnvelope)?;
        if wire.contract_version != CURRENT_FANOUT_CONTRACT_VERSION {
            return Err(FanoutEnvelopeError::UnsupportedVersion);
        }
        if wire.payload_sha256.len() != 64
            || wire
                .payload_sha256
                .bytes()
                .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
        {
            return Err(FanoutEnvelopeError::InvalidEnvelope);
        }
        let payload_sha256 = hex::decode(&wire.payload_sha256)
            .ok()
            .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
            .ok_or(FanoutEnvelopeError::InvalidEnvelope)?;
        Self::new(
            wire.community_id,
            wire.outbox_sequence,
            wire.topic,
            wire.provenance,
            payload_sha256,
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FanoutSourceKey {
    source_system: SourceSystem,
    source_record_id: String,
    source_version: String,
}

impl FanoutSourceKey {
    fn from_envelope(envelope: &FanoutEnvelope) -> Result<Self, FanoutEnvelopeError> {
        Ok(Self {
            source_system: envelope.provenance.source_system,
            source_record_id: envelope.provenance.source_record_id.as_str().to_owned(),
            source_version: envelope
                .provenance
                .source_version
                .clone()
                .ok_or(FanoutEnvelopeError::InvalidEnvelope)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FanoutAdmission {
    New,
    Duplicate,
}

pub struct LocalFanoutDeduplicator {
    community_id: CommunityId,
    capacity: usize,
    order: VecDeque<FanoutSourceKey>,
    entries: HashSet<FanoutSourceKey>,
}

impl LocalFanoutDeduplicator {
    pub fn new(community_id: CommunityId, capacity: usize) -> Result<Self, FanoutEnvelopeError> {
        if capacity == 0 || capacity > MAX_LOCAL_DEDUPLICATION_ENTRIES {
            return Err(FanoutEnvelopeError::InvalidDeduplicationCapacity);
        }
        Ok(Self {
            community_id,
            capacity,
            order: VecDeque::with_capacity(capacity),
            entries: HashSet::with_capacity(capacity),
        })
    }

    pub fn admit(
        &mut self,
        tenant: &TenantContext,
        envelope: &FanoutEnvelope,
    ) -> Result<FanoutAdmission, FanoutEnvelopeError> {
        if tenant.community_id() != self.community_id || envelope.community_id != self.community_id
        {
            return Err(FanoutEnvelopeError::TenantMismatch);
        }
        let key = FanoutSourceKey::from_envelope(envelope)?;
        if self.entries.contains(&key) {
            return Ok(FanoutAdmission::Duplicate);
        }
        if self.order.len() == self.capacity
            && let Some(expired) = self.order.pop_front()
        {
            self.entries.remove(&expired);
        }
        self.order.push_back(key.clone());
        self.entries.insert(key);
        Ok(FanoutAdmission::New)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum FanoutEnvelopeError {
    #[error("fan-out envelope is invalid")]
    InvalidEnvelope,
    #[error("fan-out envelope exceeds its transport bound")]
    EnvelopeTooLarge,
    #[error("fan-out envelope contract version is unsupported")]
    UnsupportedVersion,
    #[error("fan-out envelope crossed its tenant boundary")]
    TenantMismatch,
    #[error("fan-out deduplication capacity is invalid")]
    InvalidDeduplicationCapacity,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FanoutEnvelopeWire {
    contract_version: u16,
    community_id: CommunityId,
    outbox_sequence: u64,
    topic: String,
    provenance: Provenance,
    payload_sha256: String,
}

fn valid_topic(topic: &str) -> bool {
    !topic.is_empty()
        && topic.len() <= MAX_FANOUT_TOPIC_BYTES
        && topic.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b':' | b'_' | b'-')
        })
}

fn valid_provenance(provenance: &Provenance) -> bool {
    provenance
        .source_version
        .as_ref()
        .is_some_and(|version| !version.is_empty() && version.len() <= 1024)
        && provenance
            .integrity
            .as_ref()
            .is_none_or(|integrity| !integrity.value.is_empty() && integrity.value.len() <= 1024)
        && i64::try_from(provenance.observed_at_millis).is_ok()
}
