use agent::{
    AgentMemoryRepository, AgentMemoryRepositoryError, AgentUsageCryptoError, AgentUsageRepository,
    AgentUsageRepositoryError, ManagedAgentSnapshotCompactionOutcome,
    ManagedAgentSnapshotDocuments, ManagedAgentSnapshotError, ManagedAgentSnapshotRepository,
    MemoryRetention, MemoryRotationOutcome, MemoryWriteOutcome, StoredEncryptedMemory,
    StoredTurnUsage, UsageQuery, UsageRetention, UsageRetentionOutcome, UsageWriteOutcome,
    decrypt_turn_usage_as_owner, encrypt_turn_usage_for_owner,
};
use agent_settings::{
    managed_agent::{
        EnvironmentReference, EnvironmentVariableName, ManagedAgentConfiguration,
        PrivateManagedAgentRecord, ProtectedCredentialReference, RuntimeId,
    },
    team::{NostrEventId, NostrPublicKey},
};
use nostr_compat::agent_memory::{
    AgentMemoryCodecError, AgentRelay, AgentRelayAccess, EngramRelayScope, decrypt_engram_as_owner,
    encrypt_engram_for_owner,
};
use nostr_compat::buzz_nips::agent_activity::{
    AgentTurnMetricPayload, EngramBody, PricingIdentity, StopReason, TokenCounts,
};
use nostr_compat::{EventId, PublicKey};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const DESKTOP_FIXTURES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/collaborative-workspace/fixtures/migrations/desktop-stores.json"
));
const AGENT_SECRET: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
];
const OWNER_SECRET: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
];
const ROTATED_OWNER_SECRET: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
];

fn owner() -> PublicKey {
    PublicKey::from_hex("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5")
        .expect("fixture owner public key must be valid")
}

fn rotated_owner() -> PublicKey {
    PublicKey::from_hex("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9")
        .expect("fixture rotated owner public key must be valid")
}

fn fixture(fixture_id: &str) -> Value {
    let document: Value =
        serde_json::from_str(DESKTOP_FIXTURES).expect("frozen desktop fixtures must be valid JSON");
    document
        .get("fixtures")
        .and_then(Value::as_array)
        .and_then(|fixtures| {
            fixtures.iter().find(|fixture| {
                fixture.get("fixture_id").and_then(Value::as_str) == Some(fixture_id)
            })
        })
        .cloned()
        .expect("named frozen desktop fixture must exist")
}

fn fixture_record(fixture_id: &str) -> Value {
    fixture(fixture_id)
        .get("records")
        .and_then(Value::as_array)
        .and_then(|records| records.first())
        .cloned()
        .expect("frozen desktop fixture must contain a record")
}

fn fixture_expected(fixture_id: &str, field: &str) -> Value {
    fixture(fixture_id)
        .get("expected")
        .and_then(|expected| expected.get(field))
        .cloned()
        .expect("frozen desktop fixture expectation must exist")
}

fn nostr_public_key(public_key: PublicKey) -> NostrPublicKey {
    NostrPublicKey::parse(public_key.to_hex()).expect("fixture settings public key must be valid")
}

fn runtime() -> PrivateManagedAgentRecord {
    let mut environment = BTreeMap::new();
    environment.insert(
        EnvironmentVariableName::parse("OPENAI_API_KEY")
            .expect("fixture environment name must be valid"),
        EnvironmentReference::ProtectedCredential(
            ProtectedCredentialReference::parse("credentials/openai/default")
                .expect("fixture credential reference must be valid"),
        ),
    );
    PrivateManagedAgentRecord::new(
        nostr_public_key(owner()),
        NostrPublicKey::parse("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")
            .expect("fixture agent public key must be valid"),
        NostrEventId::parse("33".repeat(32)).expect("fixture event ID must be valid"),
        ManagedAgentConfiguration::new(
            RuntimeId::parse("local-acp").expect("fixture runtime must be valid"),
            None,
            None,
            environment,
        )
        .expect("fixture configuration must be valid"),
    )
    .expect("fixture managed-agent runtime must be valid")
}

fn memory(owner: PublicKey, value: &str, event_byte: u8, created_at: u64) -> StoredEncryptedMemory {
    let encrypted = encrypt_engram_for_owner(
        &AGENT_SECRET,
        owner,
        &EngramBody::Memory {
            slug: "mem/conformance".to_owned(),
            value: Some(value.to_owned()),
        },
    )
    .expect("encrypt conformance memory");
    StoredEncryptedMemory::new(
        &encrypted,
        EventId::from_bytes([event_byte; 32]),
        created_at,
        MemoryRetention::new(1, None).expect("fixture memory retention must be valid"),
    )
    .expect("stored conformance memory must be valid")
}

fn token_counts(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
) -> TokenCounts {
    TokenCounts {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        total_tokens: Some(input_tokens + output_tokens),
        cost_usd: Some(0.01),
        cache_read_tokens,
        cache_write_tokens,
    }
}

fn usage_payload(is_current: bool) -> AgentTurnMetricPayload {
    AgentTurnMetricPayload {
        harness: if is_current { "zed" } else { "buzz" }.to_owned(),
        model: Some("gpt-5.2".to_owned()),
        channel_id: Some("private-channel".to_owned()),
        session_id: None,
        turn_id: Some(
            if is_current {
                "current-turn"
            } else {
                "legacy-turn"
            }
            .to_owned(),
        ),
        turn_seq: None,
        timestamp: if is_current {
            "2026-08-23T12:01:00Z"
        } else {
            "2026-08-23T12:00:00Z"
        }
        .to_owned(),
        turn: Some(if is_current {
            token_counts(20, 5, Some(7), Some(3))
        } else {
            token_counts(10, 2, None, None)
        }),
        cumulative: None,
        delta_reliable: true,
        stop_reason: Some(StopReason::EndTurn),
        pricing_identity: is_current.then(|| PricingIdentity {
            authority: "api.openai.com".to_owned(),
            model: "gpt-5.2".to_owned(),
            cache_class: Some("prompt".to_owned()),
        }),
    }
}

fn usage(
    payload: AgentTurnMetricPayload,
    created_at: u64,
) -> (agent::EncryptedTurnUsage, StoredTurnUsage) {
    let encrypted = encrypt_turn_usage_for_owner(&AGENT_SECRET, owner(), &payload)
        .expect("encrypt conformance usage");
    let event_id = encrypted
        .to_canonical_event(created_at)
        .event_id()
        .expect("derive conformance usage event ID");
    let stored = StoredTurnUsage::new(
        &encrypted,
        event_id,
        created_at,
        payload,
        UsageRetention::new(1, None).expect("fixture usage retention must be valid"),
    )
    .expect("stored conformance usage must be valid");
    (encrypted, stored)
}

#[gpui::test]
async fn agent_state_conformance_exports_legacy_and_current_usage_without_plaintext(
    cx: &mut gpui::TestAppContext,
) {
    cx.executor().allow_parking();
    assert_eq!(
        fixture("archive-schema-v0").get("version"),
        Some(&json!("schema-v0"))
    );
    assert_eq!(
        fixture_expected("archive-schema-v0", "raw_events_preserved"),
        json!(true)
    );
    assert_eq!(
        fixture_expected("archive-cache-read-v2", "legacy_values"),
        Value::Null
    );
    assert_eq!(
        fixture("archive-pricing-v3").get("version"),
        Some(&json!("schema-v3-pricing"))
    );

    let repository =
        AgentUsageRepository::open_test_database("agent_state_conformance_usage").await;
    let (legacy_encrypted, legacy) = usage(usage_payload(false), 100);
    let (current_encrypted, current) = usage(usage_payload(true), 200);
    for record in [&legacy, &current] {
        assert_eq!(
            repository
                .store(owner(), record)
                .await
                .expect("store conformance usage"),
            UsageWriteOutcome::Stored
        );
    }
    assert!(matches!(
        repository.store(rotated_owner(), &legacy).await,
        Err(AgentUsageRepositoryError::OwnerMismatch)
    ));
    assert!(matches!(
        decrypt_turn_usage_as_owner(&ROTATED_OWNER_SECRET, &legacy_encrypted),
        Err(AgentUsageCryptoError::WrongReader)
    ));
    assert_eq!(
        decrypt_turn_usage_as_owner(&OWNER_SECRET, &current_encrypted)
            .expect("owner decrypts current usage"),
        *current.payload()
    );

    let exports = repository
        .export_encrypted_for_owner(owner(), UsageQuery::default(), 201)
        .expect("export owner usage");
    assert_eq!(exports.len(), 2);
    assert_eq!(exports[0].event_id(), legacy.event_id());
    assert_eq!(exports[1].event_id(), current.event_id());
    assert_eq!(
        exports[0]
            .to_canonical_event()
            .event_id()
            .expect("derive legacy export event ID"),
        legacy.event_id()
    );
    assert_eq!(
        exports[1]
            .to_canonical_event()
            .event_id()
            .expect("derive current export event ID"),
        current.event_id()
    );
    assert!(
        repository
            .export_encrypted_for_owner(rotated_owner(), UsageQuery::default(), 201)
            .expect("foreign owner export is isolated")
            .is_empty()
    );
    let diagnostics = format!("{exports:?}");
    for private_value in [
        "legacy-turn",
        "current-turn",
        "private-channel",
        "api.openai.com",
    ] {
        assert!(!diagnostics.contains(private_value));
    }
    assert!(!diagnostics.contains(exports[0].ciphertext().wire_value()));
    assert!(!diagnostics.contains(exports[1].ciphertext().wire_value()));
    assert_eq!(
        legacy
            .payload()
            .turn
            .as_ref()
            .and_then(|turn| turn.cache_read_tokens),
        None
    );
    assert_eq!(legacy.payload().pricing_identity, None);
    assert_eq!(
        current
            .payload()
            .turn
            .as_ref()
            .and_then(|turn| turn.cache_write_tokens),
        Some(3)
    );
    assert_eq!(
        current
            .payload()
            .pricing_identity
            .as_ref()
            .map(|pricing| pricing.authority.as_str()),
        Some("api.openai.com")
    );
    assert_eq!(
        repository
            .expire(owner(), legacy.event_id(), 1, 250)
            .await
            .expect("expire legacy usage"),
        UsageRetentionOutcome::Applied
    );
    assert_eq!(
        repository
            .export_encrypted_for_owner(owner(), UsageQuery::default(), 250)
            .expect("export at retention boundary")
            .len(),
        1
    );
}

#[gpui::test]
async fn agent_state_conformance_rotates_owner_and_relay_without_cross_scope_reads(
    cx: &mut gpui::TestAppContext,
) {
    cx.executor().allow_parking();
    assert_eq!(
        fixture_expected("retention-global-v0", "source_preserved"),
        json!(true)
    );
    assert_eq!(
        fixture_expected("retention-scoped-v1", "cross_scope_reuse"),
        json!(false)
    );

    let repository =
        AgentMemoryRepository::open_test_database("agent_state_conformance_memory").await;
    let previous = memory(owner(), "owner-private-memory", 11, 100);
    let replacement = memory(rotated_owner(), "owner-private-memory", 12, 101);
    assert_eq!(
        repository
            .store(owner(), &previous)
            .await
            .expect("store previous memory"),
        MemoryWriteOutcome::Stored
    );
    assert!(matches!(
        repository.load_for_owner(rotated_owner(), previous.coordinate(), 101),
        Err(AgentMemoryRepositoryError::OwnerMismatch)
    ));
    assert_eq!(
        decrypt_engram_as_owner(
            &ROTATED_OWNER_SECRET,
            previous.coordinate(),
            previous.ciphertext(),
        ),
        Err(AgentMemoryCodecError::WrongReader)
    );

    let previous_scope = EngramRelayScope::resolve(
        &[
            AgentRelay::new("wss://stable.example", AgentRelayAccess::Write),
            AgentRelay::new("wss://departing.example", AgentRelayAccess::ReadWrite),
        ],
        &[],
    )
    .expect("resolve previous relay scope");
    let current_scope = EngramRelayScope::resolve(
        &[
            AgentRelay::new("wss://stable.example", AgentRelayAccess::ReadWrite),
            AgentRelay::new("wss://added.example", AgentRelayAccess::Write),
        ],
        &[],
    )
    .expect("resolve current relay scope");
    let rotation = previous_scope.rotate_to(&current_scope);
    assert!(rotation.requires_republication());
    assert_eq!(rotation.departing().len(), 1);
    assert_eq!(rotation.added().len(), 1);
    assert_eq!(rotation.publication_targets().len(), 3);

    assert_eq!(
        repository
            .rotate_owner(owner(), previous.coordinate(), 1, &replacement, 150)
            .await
            .expect("rotate memory owner"),
        MemoryRotationOutcome::Applied
    );
    assert!(
        repository
            .load_for_owner(owner(), previous.coordinate(), 150)
            .expect("old owner scope remains queryable")
            .is_none()
    );
    let loaded = repository
        .load_for_owner(rotated_owner(), replacement.coordinate(), 150)
        .expect("load rotated memory")
        .expect("rotated memory exists");
    assert_eq!(loaded, replacement);
    assert_eq!(
        decrypt_engram_as_owner(
            &ROTATED_OWNER_SECRET,
            loaded.coordinate(),
            loaded.ciphertext(),
        )
        .expect("rotated owner decrypts memory"),
        EngramBody::Memory {
            slug: "mem/conformance".to_owned(),
            value: Some("owner-private-memory".to_owned()),
        }
    );
    assert_eq!(
        decrypt_engram_as_owner(&OWNER_SECRET, loaded.coordinate(), loaded.ciphertext()),
        Err(AgentMemoryCodecError::WrongReader)
    );
    let diagnostics = format!("{loaded:?}");
    assert!(!diagnostics.contains("owner-private-memory"));
    assert!(!diagnostics.contains(loaded.ciphertext().wire_value()));
}

#[gpui::test]
async fn agent_state_conformance_compacts_snapshots_without_disclosing_private_state(
    cx: &mut gpui::TestAppContext,
) {
    cx.executor().allow_parking();
    assert_eq!(
        fixture_expected("personas-provider-v0", "legacy_file_preserved"),
        json!(true)
    );
    assert_eq!(
        fixture_expected("personas-runtime-v1", "idempotent"),
        json!(true)
    );
    assert_eq!(
        fixture_expected("teams-detached-v1", "source_preserved"),
        json!(true)
    );

    let repository =
        ManagedAgentSnapshotRepository::open_test_database("agent_state_conformance_snapshot")
            .await;
    let runtime = runtime();
    let legacy_persona = fixture_record("personas-provider-v0");
    let current_persona = fixture_record("personas-runtime-v1");
    let legacy_team = fixture_record("teams-directory-v0");
    let current_team = fixture_record("teams-detached-v1");
    let documents = [
        (legacy_persona.clone(), legacy_team),
        (current_persona.clone(), current_team.clone()),
        (
            json!({"revision": 3, "source": current_persona}),
            json!({"revision": 3, "source": current_team}),
        ),
        (
            json!({"revision": 4, "private_prompt": "Never disclose", "name": "reviewer"}),
            json!({"revision": 4, "members": ["reviewer"]}),
        ),
    ];
    let mut snapshots = Vec::new();
    for (index, (persona, team)) in documents.into_iter().enumerate() {
        snapshots.push(
            repository
                .create(
                    runtime.owner_public_key(),
                    &runtime,
                    ManagedAgentSnapshotDocuments::new(persona, Some(team))
                        .expect("fixture snapshot documents must be valid"),
                    u64::try_from(index + 1).expect("fixture index fits u64") * 100,
                )
                .await
                .expect("create conformance snapshot"),
        );
    }
    let foreign_owner = nostr_public_key(rotated_owner());
    assert!(
        repository
            .load(&foreign_owner, runtime.agent_public_key(), &snapshots[3])
            .expect("foreign owner query is isolated")
            .is_none()
    );
    assert!(matches!(
        repository.restore(&foreign_owner, &runtime, runtime.version(), &snapshots[3]),
        Err(ManagedAgentSnapshotError::OwnerMismatch)
    ));
    assert!(matches!(
        repository
            .compact(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &snapshots[3],
                2,
                500,
            )
            .await
            .expect("compact conformance snapshots"),
        ManagedAgentSnapshotCompactionOutcome::Compacted { removed: 2, .. }
    ));
    for removed in &snapshots[..2] {
        assert!(
            repository
                .load(
                    runtime.owner_public_key(),
                    runtime.agent_public_key(),
                    removed,
                )
                .expect("load compacted snapshot identity")
                .is_none()
        );
    }
    let latest = repository
        .load(
            runtime.owner_public_key(),
            runtime.agent_public_key(),
            &snapshots[3],
        )
        .expect("load latest snapshot")
        .expect("latest snapshot remains");
    assert_eq!(latest.persona()["revision"], 4);
    assert_eq!(latest.team().expect("latest team exists")["revision"], 4);
    assert_eq!(latest.runtime(), &runtime);
    let diagnostics = format!("{latest:?}");
    assert!(!diagnostics.contains("Never disclose"));
    assert!(!diagnostics.contains("credentials/openai/default"));
}
