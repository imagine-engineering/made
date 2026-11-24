use collab::{
    pubsub::envelope::{
        FanoutAdmission, FanoutEnvelope, FanoutEnvelopeError, LocalFanoutDeduplicator,
        MAX_FANOUT_ENVELOPE_BYTES,
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    CommunityId, Provenance, SourceRecordId, SourceSystem, TenantContext, TrustedTenantRoute,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(value: u128) -> TenantContext {
    let community_id = community(value);
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "pubsub-envelope")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn envelope(community_id: CommunityId, sequence: u64, version: &str) -> FanoutEnvelope {
    FanoutEnvelope::new(
        community_id,
        sequence,
        "conversation.activity",
        Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new("conversation:42/activity:7").expect("source ID"),
            1_900_000_000_000,
        )
        .with_source_version(version),
        Sha256::digest(b"authoritative payload bytes").into(),
    )
    .expect("fan-out envelope")
}

#[test]
fn collaboration_pubsub_envelope_round_trips_only_an_authority_reference() {
    let envelope = envelope(community(1), 9, "17");
    let encoded = envelope.encode().expect("encode envelope");
    assert!(encoded.len() <= MAX_FANOUT_ENVELOPE_BYTES);
    let encoded_text = String::from_utf8(encoded.clone()).expect("JSON envelope");
    assert!(!encoded_text.contains("authoritative payload bytes"));
    assert!(!encoded_text.contains("content"));

    let decoded = FanoutEnvelope::decode(&encoded).expect("decode envelope");
    assert_eq!(decoded, envelope);
    assert_eq!(decoded.outbox_sequence(), 9);
    assert_eq!(decoded.topic(), "conversation.activity");
    assert_eq!(decoded.provenance().source_version.as_deref(), Some("17"));
}

#[test]
fn collaboration_pubsub_envelope_rejects_wrong_tenant_before_local_deduplication() {
    let local_tenant = tenant(1);
    let foreign = envelope(community(2), 1, "1");
    let mut deduplicator =
        LocalFanoutDeduplicator::new(local_tenant.community_id(), 16).expect("deduplicator");

    assert_eq!(
        deduplicator.admit(&local_tenant, &foreign),
        Err(FanoutEnvelopeError::TenantMismatch)
    );
    assert!(deduplicator.is_empty());
}

#[test]
fn collaboration_pubsub_envelope_deduplicates_source_versions_and_bounds_memory() {
    let tenant = tenant(1);
    let mut deduplicator =
        LocalFanoutDeduplicator::new(tenant.community_id(), 2).expect("deduplicator");
    let first = envelope(tenant.community_id(), 1, "1");
    let same_source_from_redis = FanoutEnvelope::new(
        tenant.community_id(),
        2,
        "conversation.activity",
        first.provenance().clone(),
        [3; 32],
    )
    .expect("same source envelope");

    assert_eq!(
        deduplicator.admit(&tenant, &first),
        Ok(FanoutAdmission::New)
    );
    assert_eq!(
        deduplicator.admit(&tenant, &same_source_from_redis),
        Ok(FanoutAdmission::Duplicate)
    );
    assert_eq!(
        deduplicator.admit(&tenant, &envelope(tenant.community_id(), 3, "2")),
        Ok(FanoutAdmission::New)
    );
    assert_eq!(deduplicator.len(), 2);
    assert_eq!(
        deduplicator.admit(&tenant, &envelope(tenant.community_id(), 4, "3")),
        Ok(FanoutAdmission::New)
    );
    assert_eq!(deduplicator.len(), 2);
    assert_eq!(
        deduplicator.admit(&tenant, &first),
        Ok(FanoutAdmission::New),
        "the bounded oldest source key was evicted"
    );
}

#[test]
fn collaboration_pubsub_envelope_rejects_unknown_fields_versions_and_unversioned_sources() {
    let encoded = envelope(community(1), 1, "1")
        .encode()
        .expect("encoded envelope");
    let mut value: Value = serde_json::from_slice(&encoded).expect("JSON");
    value["contract_version"] = Value::from(2);
    assert_eq!(
        FanoutEnvelope::decode(&serde_json::to_vec(&value).expect("JSON")),
        Err(FanoutEnvelopeError::UnsupportedVersion)
    );
    value["contract_version"] = Value::from(1);
    value["unexpected"] = Value::Bool(true);
    assert_eq!(
        FanoutEnvelope::decode(&serde_json::to_vec(&value).expect("JSON")),
        Err(FanoutEnvelopeError::InvalidEnvelope)
    );
    assert_eq!(
        FanoutEnvelope::new(
            community(1),
            1,
            "conversation.activity",
            Provenance::new(
                SourceSystem::Zed,
                SourceRecordId::new("conversation:42/activity:7").expect("source ID"),
                1_900_000_000_000,
            ),
            [0; 32],
        ),
        Err(FanoutEnvelopeError::InvalidEnvelope)
    );
    assert_eq!(
        FanoutEnvelope::decode(&vec![b'x'; MAX_FANOUT_ENVELOPE_BYTES + 1]),
        Err(FanoutEnvelopeError::EnvelopeTooLarge)
    );
}
