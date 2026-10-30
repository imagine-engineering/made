use std::collections::BTreeMap;

use collab::{
    db::collaboration::event_repository::{
        EventRepository, EventRepositoryError, EventRepositoryQuery, EventStoreOutcome,
        EventVerificationState, MAX_EVENT_QUERY_FILTERS, MAX_EVENT_QUERY_RESULTS,
        VerifiedEventRecord,
    },
    db::collaboration::persistence_policy::{EventPersistencePolicy, PrivacyAdmission},
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{CommunityId, TenantContext, TrustedTenantRoute};
use nostr_compat::{
    CanonicalEvent, EventId, EventSignature, PublicKey, SignedEvent, TimestampPolicy,
    filter::{EventFilter, HexPrefix},
    head::replacement_coordinate,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use serde_json::Value;
use uuid::Uuid;

const EVENTS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/protocol/events.json"
));
const HEAD_MIGRATION: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000200_collaboration_event_heads.up.sql"
));
const HEAD_ROLLBACK: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/migrations/20260820000200_collaboration_event_heads.down.sql"
));

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn tenant(value: u128) -> TenantContext {
    let community_id = community(value);
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "event-repository")
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
            .expect("valid signature encoding"),
    }
}

fn record(community_id: CommunityId, name: &str) -> VerifiedEventRecord {
    VerifiedEventRecord::new(
        community_id,
        fixture_event(name),
        EventVerificationState::Historical,
        1_900_000_000_000,
        TimestampPolicy::Historical,
    )
    .expect("verified fixture")
}

fn persistence_decision(
    record: &VerifiedEventRecord,
) -> collab::db::collaboration::persistence_policy::EventPersistenceDecision {
    EventPersistencePolicy::evaluate(
        record.signed_event().event.kind,
        PrivacyAdmission::community(),
    )
    .expect("fixture persistence decision")
}

fn event_row(community_id: CommunityId, event: &SignedEvent) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("community_id".into(), community_id.as_uuid().into()),
        (
            "event_id".into(),
            event.claimed_id.as_bytes().to_vec().into(),
        ),
        (
            "author_public_key".into(),
            event.event.public_key.as_bytes().to_vec().into(),
        ),
        (
            "event_created_at_text".into(),
            event.event.created_at.to_string().into(),
        ),
        ("kind".into(), i32::from(event.event.kind).into()),
        (
            "tags".into(),
            serde_json::to_value(&event.event.tags)
                .expect("tag JSON")
                .into(),
        ),
        ("content".into(), event.event.content.clone().into()),
        (
            "canonical_event_bytes".into(),
            event
                .event
                .canonical_bytes()
                .expect("canonical bytes")
                .into(),
        ),
        (
            "signature".into(),
            event.signature.as_bytes().to_vec().into(),
        ),
        (
            "signature_state".into(),
            "verified_historical".to_owned().into(),
        ),
        ("verified_at_millis".into(), 1_900_000_000_000_i64.into()),
    ])
}

fn repository(
    query_results: Vec<Vec<BTreeMap<String, SeaValue>>>,
    affected_rows: &[u64],
) -> EventRepository {
    let database =
        MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results(query_results)
            .append_exec_results(affected_rows.iter().copied().map(|rows_affected| {
                MockExecResult {
                    last_insert_id: 0,
                    rows_affected,
                }
            }))
            .into_connection();
    EventRepository::new(database).expect("Postgres event repository")
}

fn log(repository: EventRepository) -> String {
    format!("{:#?}", repository.into_connection().into_transaction_log())
}

#[tokio::test]
async fn event_repository_inserts_once_and_never_persists_ephemeral_events() {
    let tenant = tenant(1);
    let regular = record(tenant.community_id(), "legacy_message");
    let inserted_repository = repository(Vec::new(), &[1, 1]);
    assert_eq!(
        inserted_repository
            .store(&tenant, &regular, persistence_decision(&regular))
            .await
            .expect("insert regular event"),
        EventStoreOutcome::Inserted
    );
    let inserted_log = log(inserted_repository);
    assert!(inserted_log.contains("set_config('app.community_id'"));
    assert!(inserted_log.contains("INSERT INTO public.collaboration_events"));
    assert!(inserted_log.contains("ON CONFLICT"));

    let duplicate_repository = repository(Vec::new(), &[1, 0]);
    assert_eq!(
        duplicate_repository
            .store(&tenant, &regular, persistence_decision(&regular))
            .await
            .expect("duplicate regular event"),
        EventStoreOutcome::Duplicate
    );

    let ephemeral = SignedEvent {
        event: CanonicalEvent::new(
            regular.signed_event().event.public_key,
            regular.signed_event().event.created_at,
            20_001,
            Vec::new(),
            "typing".into(),
        ),
        ..regular.signed_event().clone()
    };
    let ephemeral = SignedEvent {
        claimed_id: ephemeral.event.event_id().expect("ephemeral id"),
        ..ephemeral
    };
    let ephemeral = VerifiedEventRecord::new(
        tenant.community_id(),
        ephemeral,
        EventVerificationState::Historical,
        1_900_000_000_000,
        TimestampPolicy::Historical,
    );
    assert!(
        ephemeral.is_err(),
        "changing a signed fixture must fail verification before persistence"
    );
    assert!(matches!(
        VerifiedEventRecord::new(
            tenant.community_id(),
            regular.signed_event().clone(),
            EventVerificationState::Live,
            1_900_000_000_000,
            TimestampPolicy::Historical,
        ),
        Err(EventRepositoryError::InvalidRecord)
    ));

    let signed_ephemeral = fixture_signed_event(20_001, "typing");
    let ephemeral = VerifiedEventRecord::new(
        tenant.community_id(),
        signed_ephemeral,
        EventVerificationState::Live,
        1_900_000_000_000,
        TimestampPolicy::Bounded {
            now: 1_900_000_000,
            max_past_seconds: 0,
            max_future_seconds: 0,
        },
    )
    .expect("verified ephemeral event");
    let ephemeral_repository = repository(Vec::new(), &[]);
    assert_eq!(
        ephemeral_repository
            .store(&tenant, &ephemeral, persistence_decision(&ephemeral))
            .await
            .expect("ephemeral classification"),
        EventStoreOutcome::EphemeralNotPersisted
    );
    assert!(
        ephemeral_repository
            .into_connection()
            .into_transaction_log()
            .is_empty()
    );
}

fn fixture_signed_event(kind: u16, content: &str) -> SignedEvent {
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey, XOnlyPublicKey};

    let secret = SecretKey::from_slice(&[8; 32]).expect("fixture secret");
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    let event = CanonicalEvent::new(
        PublicKey::from_bytes(public_key.serialize()),
        1_900_000_000,
        kind,
        Vec::new(),
        content.to_owned(),
    );
    let claimed_id = event.event_id().expect("event id");
    let signature =
        secp.sign_schnorr_no_aux_rand(&Message::from_digest(*claimed_id.as_bytes()), &keypair);
    SignedEvent {
        claimed_id,
        event,
        signature: EventSignature::from_hex(&signature.to_string()).expect("signature"),
    }
}

#[tokio::test]
async fn event_repository_queries_bounded_filters_and_exact_addressable_heads() {
    let tenant = tenant(1);
    let profile = fixture_event("profile_new");
    let mut generic_tags = BTreeMap::new();
    generic_tags.insert('d', vec!["profile".to_owned()]);
    let filter = EventFilter {
        ids: vec![HexPrefix::new("ids", &profile.claimed_id.to_hex()[..8]).expect("id prefix")],
        authors: vec![profile.event.public_key],
        kinds: vec![profile.event.kind],
        since: Some(profile.event.created_at.saturating_sub(1)),
        until: Some(profile.event.created_at.saturating_add(1)),
        generic_tags,
    };
    let query = EventRepositoryQuery::new(vec![filter.clone()], 50).expect("bounded query");
    let query_repository = repository(vec![vec![event_row(tenant.community_id(), &profile)]], &[1]);
    let events = query_repository
        .query(&tenant, &query)
        .await
        .expect("bounded event query");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].signed_event(), &profile);
    let query_log = log(query_repository);
    for expected in [
        "encode(e.event_id, 'hex') LIKE",
        "e.author_public_key IN",
        "e.kind IN",
        "e.event_created_at >= CAST",
        "e.event_created_at <= CAST",
        "e.tags @> CAST",
        "collaboration_event_heads",
        "ORDER BY e.event_created_at DESC, e.event_id ASC LIMIT",
    ] {
        assert!(
            query_log.contains(expected),
            "missing query clause {expected}"
        );
    }

    assert!(matches!(
        EventRepositoryQuery::new(vec![filter.clone(); MAX_EVENT_QUERY_FILTERS + 1], 1),
        Err(EventRepositoryError::InvalidQuery)
    ));
    assert!(matches!(
        EventRepositoryQuery::new(vec![filter.clone()], 0),
        Err(EventRepositoryError::InvalidQuery)
    ));
    assert!(matches!(
        EventRepositoryQuery::new(vec![filter], MAX_EVENT_QUERY_RESULTS + 1),
        Err(EventRepositoryError::InvalidQuery)
    ));

    let empty_tag_filter = EventFilter {
        generic_tags: BTreeMap::from([('p', Vec::new())]),
        ..EventFilter::default()
    };
    let empty_tag_repository = repository(vec![Vec::new()], &[1]);
    empty_tag_repository
        .query(
            &tenant,
            &EventRepositoryQuery::new(vec![empty_tag_filter], 1).expect("empty tag query"),
        )
        .await
        .expect("empty tag values match nothing");
    assert!(log(empty_tag_repository).contains("FALSE"));

    let coordinate = replacement_coordinate(&profile.event).expect("profile coordinate");
    let head_repository = repository(vec![vec![event_row(tenant.community_id(), &profile)]], &[1]);
    let head = head_repository
        .head(&tenant, &coordinate)
        .await
        .expect("head query")
        .expect("profile head");
    assert_eq!(head.signed_event(), &profile);
    let head_log = log(head_repository);
    assert!(head_log.contains("h.live_event_id"));
    assert!(head_log.contains("h.discriminator"));
}

#[tokio::test]
async fn event_repository_deletion_retains_the_head_floor_and_rejects_stale_resurrection() {
    let tenant = tenant(1);
    let current = fixture_event("profile_new");
    let delete_repository = repository(Vec::new(), &[1, 1, 1]);
    assert!(
        delete_repository
            .delete(&tenant, current.claimed_id)
            .await
            .expect("delete current head")
    );
    let delete_log = log(delete_repository);
    let clear_position = delete_log
        .find("SET live_event_id = NULL")
        .expect("head tombstone");
    let delete_position = delete_log
        .find("DELETE FROM public.collaboration_events")
        .expect("event delete");
    assert!(clear_position < delete_position);

    let old = record(tenant.community_id(), "profile_old");
    let watermark = BTreeMap::from([
        (
            "head_event_id".into(),
            current.claimed_id.as_bytes().to_vec().into(),
        ),
        ("live_event_id".into(), Option::<Vec<u8>>::None.into()),
    ]);
    let stale_repository = repository(vec![vec![watermark]], &[1, 0]);
    assert_eq!(
        stale_repository
            .store(&tenant, &old, persistence_decision(&old))
            .await
            .expect("stale replaceable event"),
        EventStoreOutcome::Stale
    );
    let stale_log = log(stale_repository);
    assert!(stale_log.contains("head_event_created_at"));
    assert!(!stale_log.contains("INSERT INTO public.collaboration_events"));
}

#[tokio::test]
async fn event_repository_fails_closed_on_cross_tenant_records() {
    let requested_tenant = tenant(1);
    let foreign_event = fixture_event("legacy_message");
    let repository = repository(vec![vec![event_row(community(2), &foreign_event)]], &[1]);

    let result = repository
        .exact(&requested_tenant, foreign_event.claimed_id)
        .await;
    assert!(matches!(
        result,
        Err(EventRepositoryError::TenantBoundaryViolation)
    ));
    let repository_log = log(repository);
    assert!(repository_log.contains("ROLLBACK"));
    assert!(repository_log.contains("set_config('app.community_id'"));
}

#[test]
fn event_repository_head_schema_is_tenant_fenced_and_preserves_tombstones() {
    for required in [
        "PRIMARY KEY (community_id, kind, author_public_key, discriminator)",
        "head_event_created_at BETWEEN 0 AND 18446744073709551615",
        "head_event_id bytea NOT NULL",
        "live_event_id bytea CHECK (live_event_id IS NULL",
        "collaboration_event_heads_live_event",
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "AS RESTRICTIVE",
        "current_setting('app.community_id', true)",
    ] {
        assert!(
            HEAD_MIGRATION.contains(required),
            "missing head-schema invariant {required}"
        );
    }
    assert_eq!(
        HEAD_ROLLBACK.trim(),
        "DROP TABLE public.collaboration_event_heads;"
    );
    assert!(!HEAD_ROLLBACK.contains("CASCADE"));
}
