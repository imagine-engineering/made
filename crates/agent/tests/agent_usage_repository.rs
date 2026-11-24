use agent::{
    AgentUsageCryptoError, AgentUsageRepository, AgentUsageRepositoryError,
    CLIENT_TELEMETRY_INTEGRATION, ClientTelemetryIntegration, StoredTurnUsage, UsageQuery,
    UsageRetention, UsageRetentionOutcome, UsageWriteOutcome, decrypt_turn_usage_as_agent,
    decrypt_turn_usage_as_owner, encrypt_turn_usage_for_owner,
};
use nostr_compat::PublicKey;
use nostr_compat::buzz_nips::agent_activity::{AgentTurnMetricPayload, StopReason, TokenCounts};

const AGENT_SECRET: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
];
const OWNER_SECRET: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
];
const OTHER_OWNER_SECRET: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
];

fn owner() -> PublicKey {
    PublicKey::from_hex("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5")
        .expect("fixture owner public key must be valid")
}

fn other_owner() -> PublicKey {
    PublicKey::from_hex("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9")
        .expect("fixture owner public key must be valid")
}

fn counts(input: Option<u64>, output: Option<u64>, cost: Option<f64>) -> TokenCounts {
    TokenCounts {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input.zip(output).map(|(input, output)| input + output),
        cost_usd: cost,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

fn metric(
    turn_id: &str,
    turn_sequence: Option<u64>,
    turn: Option<TokenCounts>,
    cumulative: Option<TokenCounts>,
    reliable: bool,
) -> AgentTurnMetricPayload {
    AgentTurnMetricPayload {
        harness: "zed".to_owned(),
        model: Some("claude-sonnet-4-5".to_owned()),
        channel_id: Some("private-channel".to_owned()),
        session_id: turn_sequence.map(|_| "session-1".to_owned()),
        turn_id: Some(turn_id.to_owned()),
        turn_seq: turn_sequence,
        timestamp: "2026-08-23T10:00:00Z".to_owned(),
        turn,
        cumulative,
        delta_reliable: reliable,
        stop_reason: Some(StopReason::EndTurn),
        pricing_identity: None,
    }
}

fn stored(
    payload: AgentTurnMetricPayload,
    created_at: u64,
    retention: UsageRetention,
) -> StoredTurnUsage {
    let encrypted = encrypt_turn_usage_for_owner(&AGENT_SECRET, owner(), &payload)
        .expect("encrypt fixture usage");
    let event_id = encrypted
        .to_canonical_event(created_at)
        .event_id()
        .expect("derive fixture event ID");
    StoredTurnUsage::new(&encrypted, event_id, created_at, payload, retention)
        .expect("create stored usage")
}

#[test]
fn usage_payload_is_nip44_encrypted_for_only_the_owner_and_agent() {
    let payload = metric(
        "turn-secret",
        None,
        Some(counts(Some(100), Some(20), Some(0.03))),
        None,
        true,
    );
    let encrypted =
        encrypt_turn_usage_for_owner(&AGENT_SECRET, owner(), &payload).expect("encrypt usage");
    assert_eq!(
        decrypt_turn_usage_as_owner(&OWNER_SECRET, &encrypted).expect("owner decrypts usage"),
        payload
    );
    assert_eq!(
        decrypt_turn_usage_as_agent(&AGENT_SECRET, &encrypted).expect("agent decrypts usage"),
        payload
    );
    assert!(matches!(
        decrypt_turn_usage_as_owner(&OTHER_OWNER_SECRET, &encrypted),
        Err(AgentUsageCryptoError::WrongReader)
    ));
    assert!(!encrypted.ciphertext().wire_value().contains("turn-secret"));
    let event = encrypted.to_canonical_event(1_777_111_200);
    assert_eq!(event.kind, 44_200);
    assert_eq!(event.public_key, encrypted.agent());
    assert_eq!(
        event.tags,
        vec![
            vec!["p".to_owned(), owner().to_hex()],
            vec!["agent".to_owned(), encrypted.agent().to_hex()],
        ]
    );
    let diagnostics = format!("{encrypted:?}");
    assert!(!diagnostics.contains("turn-secret"));
    assert!(!diagnostics.contains(encrypted.ciphertext().wire_value()));
}

#[gpui::test]
async fn aggregation_recomputes_cumulative_deltas_and_preserves_unknown_values() {
    let repository = AgentUsageRepository::open_test_database("agent_usage_aggregation").await;
    let records = [
        stored(
            metric(
                "turn-1",
                Some(1),
                Some(counts(Some(100), Some(20), Some(0.03))),
                Some(counts(Some(100), Some(20), Some(0.03))),
                true,
            ),
            10,
            UsageRetention::new(1, None).expect("retention"),
        ),
        stored(
            metric(
                "turn-2",
                Some(2),
                Some(counts(Some(999), Some(999), Some(9.99))),
                Some(counts(Some(160), Some(35), Some(0.05))),
                false,
            ),
            20,
            UsageRetention::new(1, None).expect("retention"),
        ),
        stored(
            metric(
                "turn-3",
                Some(3),
                Some(counts(None, Some(10), None)),
                Some(counts(Some(140), Some(45), None)),
                true,
            ),
            30,
            UsageRetention::new(1, None).expect("retention"),
        ),
    ];
    for record in &records {
        assert_eq!(
            repository
                .store(owner(), record)
                .await
                .expect("store usage"),
            UsageWriteOutcome::Stored
        );
    }

    let aggregate = repository
        .aggregate_for_owner(owner(), UsageQuery::default(), 31)
        .expect("aggregate usage");
    assert_eq!(aggregate.record_count, 3);
    assert_eq!(aggregate.accounted_turn_count, 3);
    assert_eq!(aggregate.input_tokens, Some(160));
    assert_eq!(aggregate.output_tokens, Some(45));
    assert_eq!(aggregate.total_tokens, Some(195));
    assert_eq!(aggregate.cost_usd, Some(0.05));
}

#[gpui::test]
async fn retention_and_export_are_owner_scoped_and_ciphertext_only(cx: &mut gpui::TestAppContext) {
    cx.executor().allow_parking();
    let database_directory = tempfile::tempdir().expect("create fixture database directory");
    let repository = AgentUsageRepository::open_test_file_database(database_directory.path()).await;
    let record = stored(
        metric(
            "private-turn",
            None,
            Some(counts(Some(7), Some(2), Some(0.01))),
            None,
            true,
        ),
        40,
        UsageRetention::new(1, None).expect("retention"),
    );
    assert_eq!(
        repository
            .store(owner(), &record)
            .await
            .expect("store usage"),
        UsageWriteOutcome::Stored
    );
    assert!(matches!(
        repository.store(other_owner(), &record).await,
        Err(AgentUsageRepositoryError::OwnerMismatch)
    ));
    assert_eq!(
        repository
            .store(owner(), &record)
            .await
            .expect("retry usage"),
        UsageWriteOutcome::AlreadyStored
    );
    drop(repository);

    let restarted = AgentUsageRepository::open_test_file_database(database_directory.path()).await;
    let exports = restarted
        .export_encrypted_for_owner(owner(), UsageQuery::default(), 41)
        .expect("export encrypted usage");
    assert_eq!(exports.len(), 1);
    assert_eq!(exports[0].event_id(), record.event_id());
    assert_eq!(exports[0].ciphertext(), record.ciphertext());
    assert_eq!(
        exports[0]
            .to_canonical_event()
            .event_id()
            .expect("derive exported event ID"),
        record.event_id()
    );
    let diagnostics = format!("{:?}", exports[0]);
    assert!(!diagnostics.contains("private-turn"));
    assert!(!diagnostics.contains(exports[0].ciphertext().wire_value()));

    assert_eq!(
        restarted
            .expire(owner(), record.event_id(), 1, 50)
            .await
            .expect("expire usage"),
        UsageRetentionOutcome::Applied
    );
    assert_eq!(
        restarted
            .expire(owner(), record.event_id(), 1, 60)
            .await
            .expect("stale expiry"),
        UsageRetentionOutcome::Stale
    );
    assert!(
        restarted
            .load_for_owner(owner(), record.event_id(), 50)
            .expect("load expired usage")
            .is_none()
    );
    assert_eq!(
        restarted
            .purge_expired(owner(), 50)
            .await
            .expect("purge expired usage"),
        1
    );
}

#[gpui::test]
async fn local_accounting_stays_available_without_client_telemetry() {
    assert_eq!(
        CLIENT_TELEMETRY_INTEGRATION,
        ClientTelemetryIntegration::DisabledByDesign
    );
    let repository = AgentUsageRepository::open_test_database("agent_usage_no_telemetry").await;
    let record = stored(
        metric(
            "local-only",
            None,
            Some(counts(Some(11), Some(4), None)),
            None,
            true,
        ),
        50,
        UsageRetention::new(1, None).expect("retention"),
    );
    repository
        .store(owner(), &record)
        .await
        .expect("store private local usage");
    assert_eq!(
        repository
            .aggregate_for_owner(owner(), UsageQuery::default(), 51)
            .expect("aggregate private local usage")
            .total_tokens,
        Some(15)
    );

    repository
        .corrupt_payload_for_test(owner(), record.event_id())
        .await
        .expect("corrupt payload fixture");
    assert!(matches!(
        repository.load_for_owner(owner(), record.event_id(), 51),
        Err(AgentUsageRepositoryError::CorruptPayload)
    ));
}
