use collab::{
    migration::buzz::events::{BuzzEventImportError, BuzzEventImporter, BuzzEventSourceRecord},
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use nostr_compat::{CanonicalEvent, EventId, EventSignature, PublicKey, SignedEvent};
use sea_orm::{DatabaseBackend, MockDatabase};
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

const EVENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
));
const EVENT_MIGRATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000100_collaboration_events.up.sql"
));
const HEAD_MIGRATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000200_collaboration_event_heads.up.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "buzz-event-import")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn fixture_event(name: &str) -> SignedEvent {
    let fixtures: Value = serde_json::from_str(EVENTS).expect("valid frozen event corpus");
    let value = &fixtures["events"][name];
    SignedEvent {
        claimed_id: EventId::from_hex(value["id"].as_str().expect("event id"))
            .expect("valid event id"),
        event: CanonicalEvent::new(
            PublicKey::from_hex(value["pubkey"].as_str().expect("public key"))
                .expect("valid public key"),
            value["created_at"].as_u64().expect("created_at"),
            u16::try_from(value["kind"].as_u64().expect("kind")).expect("u16 kind"),
            serde_json::from_value(value["tags"].clone()).expect("tags"),
            value["content"].as_str().expect("content").to_owned(),
        ),
        signature: EventSignature::from_hex(value["sig"].as_str().expect("signature"))
            .expect("valid signature"),
    }
}

fn source_record(
    community_id: CommunityId,
    sequence: u64,
    name: &str,
    deleted: bool,
) -> BuzzEventSourceRecord {
    BuzzEventSourceRecord::new(
        community_id,
        sequence,
        fixture_event(name),
        1_900_000_000_000 + sequence,
        deleted.then_some(1_900_000_100_000 + sequence),
    )
    .expect("valid Buzz source record")
}

#[tokio::test]
async fn buzz_event_import_rejects_cross_tenant_batches_before_database_io() {
    let source_community = community(1);
    let importer =
        BuzzEventImporter::new(MockDatabase::new(DatabaseBackend::Postgres).into_connection())
            .expect("PostgreSQL importer");
    let result = importer
        .import_batch(
            &tenant(community(2)),
            &[source_record(source_community, 1, "legacy_message", false)],
        )
        .await;
    assert!(matches!(
        result,
        Err(BuzzEventImportError::TenantBoundaryViolation)
    ));
    assert!(importer.into_connection().into_transaction_log().is_empty());
}

#[tokio::test]
async fn buzz_event_import_preserves_signed_bytes_heads_and_interruption_idempotency() {
    let Some(database_url) = std::env::var("COLLAB_BUZZ_EVENT_IMPORT_TEST_DATABASE_URL").ok()
    else {
        eprintln!(
            "COLLAB_BUZZ_EVENT_IMPORT_TEST_DATABASE_URL is unset; live event import test skipped"
        );
        return;
    };
    let pool = PgPool::connect(&database_url)
        .await
        .expect("connect isolated PostgreSQL");
    sqlx::raw_sql(EVENT_MIGRATION)
        .execute(&pool)
        .await
        .expect("apply canonical event migration");
    sqlx::raw_sql(HEAD_MIGRATION)
        .execute(&pool)
        .await
        .expect("apply canonical head migration");

    let community_id = community(1);
    let tenant = tenant(community_id);
    let records = vec![
        source_record(community_id, 1, "legacy_message", false),
        source_record(community_id, 2, "profile_old", true),
        source_record(community_id, 3, "profile_new", true),
        source_record(community_id, 4, "profile_tie_b", false),
        source_record(community_id, 5, "profile_tie_a", true),
    ];
    let importer = BuzzEventImporter::new(
        sea_orm::Database::connect(&database_url)
            .await
            .expect("connect importer"),
    )
    .expect("Buzz importer");

    let interrupted = importer
        .import_batch(&tenant, &records[..3])
        .await
        .expect("first import batch");
    assert_eq!(interrupted.scanned, 3);
    assert_eq!(interrupted.inserted, 3);
    assert_eq!(interrupted.duplicates, 0);
    assert_eq!(interrupted.source_hash, interrupted.target_hash);

    let resumed_overlap = importer
        .import_batch(&tenant, &records[..3])
        .await
        .expect("replay interrupted batch");
    assert_eq!(resumed_overlap.inserted, 0);
    assert_eq!(resumed_overlap.duplicates, 3);
    assert_eq!(resumed_overlap.source_hash, interrupted.source_hash);
    assert_eq!(resumed_overlap.target_hash, interrupted.target_hash);

    let completed = importer
        .import_batch(&tenant, &records)
        .await
        .expect("resume with overlapping source window");
    assert_eq!(completed.scanned, 5);
    assert_eq!(completed.inserted, 2);
    assert_eq!(completed.duplicates, 3);
    assert_eq!(completed.addressable_coordinates, 1);
    assert_eq!(completed.final_source_sequence, 5);
    assert_eq!(completed.source_hash, completed.target_hash);

    let event_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM public.collaboration_events WHERE community_id = $1",
    )
    .bind(community_id.as_uuid())
    .fetch_one(&pool)
    .await
    .expect("count imported events");
    assert_eq!(event_count, 5);

    for record in &records {
        let stored: (Vec<u8>, Vec<u8>, Vec<u8>) = sqlx::query_as(
            "SELECT event_id, canonical_event_bytes, signature FROM public.collaboration_events WHERE community_id = $1 AND event_id = $2",
        )
        .bind(community_id.as_uuid())
        .bind(record.signed_event().claimed_id.as_bytes().as_slice())
        .fetch_one(&pool)
        .await
        .expect("read imported signed event");
        assert_eq!(stored.0, record.signed_event().claimed_id.as_bytes());
        assert_eq!(
            stored.1,
            record
                .signed_event()
                .event
                .canonical_bytes()
                .expect("canonical bytes")
        );
        assert_eq!(stored.2, record.signed_event().signature.as_bytes());
    }

    let heads: Vec<(Vec<u8>, Option<Vec<u8>>)> = sqlx::query_as(
        "SELECT head_event_id, live_event_id FROM public.collaboration_event_heads WHERE community_id = $1",
    )
    .bind(community_id.as_uuid())
    .fetch_all(&pool)
    .await
    .expect("read imported heads");
    assert_eq!(heads.len(), 1);
    assert_eq!(
        heads[0].0,
        records[4].signed_event().claimed_id.as_bytes(),
        "the lower event ID wins a same-second tie"
    );
    assert_eq!(
        heads[0].1, None,
        "a deleted winning Buzz head remains a tombstone watermark"
    );
}
