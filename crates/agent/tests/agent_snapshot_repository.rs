use agent::{
    ManagedAgentSnapshotCompactionOutcome, ManagedAgentSnapshotDocuments,
    ManagedAgentSnapshotError, ManagedAgentSnapshotRepository,
};
use agent_settings::{
    managed_agent::{
        EnvironmentReference, EnvironmentVariableName, ManagedAgentConfiguration, ModelId,
        PrivateManagedAgentRecord, ProtectedCredentialReference, ProviderId, RuntimeId,
    },
    team::{NostrEventId, NostrPublicKey},
};
use serde_json::json;
use std::collections::BTreeMap;

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[usize::from(byte >> 4)] as char);
        result.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    result
}

fn public_key(byte: u8) -> NostrPublicKey {
    NostrPublicKey::parse(lower_hex(&[byte; 32])).expect("fixture public key must be valid")
}

fn event_id(byte: u8) -> NostrEventId {
    NostrEventId::parse(lower_hex(&[byte; 32])).expect("fixture event ID must be valid")
}

fn configuration(model: &str) -> ManagedAgentConfiguration {
    let mut environment = BTreeMap::new();
    environment.insert(
        EnvironmentVariableName::parse("ANTHROPIC_API_KEY")
            .expect("fixture environment name must be valid"),
        EnvironmentReference::ProtectedCredential(
            ProtectedCredentialReference::parse("credentials/anthropic/default")
                .expect("fixture credential reference must be valid"),
        ),
    );
    ManagedAgentConfiguration::new(
        RuntimeId::parse("claude-code").expect("fixture runtime must be valid"),
        Some(ProviderId::parse("anthropic").expect("fixture provider must be valid")),
        Some(ModelId::parse(model).expect("fixture model must be valid")),
        environment,
    )
    .expect("fixture configuration must be valid")
}

fn record() -> PrivateManagedAgentRecord {
    PrivateManagedAgentRecord::new(
        public_key(1),
        public_key(2),
        event_id(3),
        configuration("claude-opus-4-1"),
    )
    .expect("fixture managed agent must be valid")
}

fn documents(revision: u64) -> ManagedAgentSnapshotDocuments {
    ManagedAgentSnapshotDocuments::new(
        json!({
            "version": revision,
            "name": "reviewer",
            "prompt": format!("Review revision {revision}"),
        }),
        Some(json!({
            "name": "review-team",
            "members": ["reviewer"],
            "version": revision,
        })),
    )
    .expect("fixture documents must be valid")
}

#[gpui::test]
async fn snapshot_round_trip_preserves_persona_team_runtime_and_provenance(
    cx: &mut gpui::TestAppContext,
) {
    cx.executor().allow_parking();
    let database_directory = tempfile::tempdir().expect("create fixture database directory");
    let repository =
        ManagedAgentSnapshotRepository::open_test_file_database(database_directory.path()).await;
    let runtime = record();
    let first = repository
        .create(runtime.owner_public_key(), &runtime, documents(1), 10)
        .await
        .expect("create first snapshot");
    let second = repository
        .create(
            runtime.owner_public_key(),
            &runtime,
            ManagedAgentSnapshotDocuments::new(
                json!({"prompt": "Review revision 1", "name": "reviewer", "version": 1}),
                Some(json!({
                    "version": 1,
                    "members": ["reviewer"],
                    "name": "review-team",
                })),
            )
            .expect("reordered documents are valid"),
            10,
        )
        .await
        .expect("retry exact snapshot");
    assert_eq!(first, second);
    drop(repository);

    let restarted =
        ManagedAgentSnapshotRepository::open_test_file_database(database_directory.path()).await;
    let loaded = restarted
        .load(
            runtime.owner_public_key(),
            runtime.agent_public_key(),
            &first,
        )
        .expect("load snapshot after restart")
        .expect("snapshot exists");
    assert_eq!(loaded.persona()["prompt"], "Review revision 1");
    assert_eq!(
        loaded.team().expect("team snapshot")["members"][0],
        "reviewer"
    );
    assert_eq!(loaded.runtime(), &runtime);
    assert_eq!(loaded.source_version(), runtime.version());
    assert_eq!(loaded.created_at(), 10);
    assert!(loaded.predecessor_snapshot_id().is_none());
    let diagnostics = format!("{loaded:?}");
    assert!(!diagnostics.contains("Review revision 1"));
    assert!(!diagnostics.contains("credentials/anthropic/default"));
}

#[gpui::test]
async fn comparison_and_restore_are_exact_and_version_fenced() {
    let repository =
        ManagedAgentSnapshotRepository::open_test_database("managed_agent_snapshot_restore").await;
    let mut runtime = record();
    let first = repository
        .create(runtime.owner_public_key(), &runtime, documents(1), 10)
        .await
        .expect("create first snapshot");
    let expected = runtime.version().clone();
    runtime
        .replace(&expected, event_id(4), configuration("claude-sonnet-4-1"))
        .expect("advance fixture runtime");
    let second = repository
        .create(runtime.owner_public_key(), &runtime, documents(2), 20)
        .await
        .expect("create second snapshot");
    assert_eq!(
        repository
            .compare(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &first,
                &second,
            )
            .expect("compare snapshots"),
        agent::ManagedAgentSnapshotComparison {
            persona_changed: true,
            team_changed: true,
            runtime_changed: true,
        }
    );
    assert!(matches!(
        repository.restore(runtime.owner_public_key(), &runtime, &expected, &first),
        Err(ManagedAgentSnapshotError::StaleRestore)
    ));
    let restored = repository
        .restore(
            runtime.owner_public_key(),
            &runtime,
            runtime.version(),
            &first,
        )
        .expect("current version admits restore handoff");
    assert_eq!(restored.runtime().version(), &expected);
    assert_eq!(restored.persona()["version"], 1);
}

#[gpui::test]
async fn partial_corruption_fails_closed_and_prevents_compaction() {
    let repository =
        ManagedAgentSnapshotRepository::open_test_database("managed_agent_snapshot_corruption")
            .await;
    let runtime = record();
    let first = repository
        .create(runtime.owner_public_key(), &runtime, documents(1), 10)
        .await
        .expect("create first snapshot");
    let second = repository
        .create(runtime.owner_public_key(), &runtime, documents(2), 20)
        .await
        .expect("create second snapshot");
    repository
        .corrupt_persona_for_test(
            runtime.owner_public_key(),
            runtime.agent_public_key(),
            &first,
        )
        .await
        .expect("corrupt snapshot fixture");

    assert!(matches!(
        repository.load(
            runtime.owner_public_key(),
            runtime.agent_public_key(),
            &first
        ),
        Err(ManagedAgentSnapshotError::CorruptSnapshot)
    ));
    assert!(matches!(
        repository
            .compact(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &second,
                1,
                30,
            )
            .await,
        Err(ManagedAgentSnapshotError::CorruptSnapshot)
    ));
    assert!(
        repository
            .load(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &second
            )
            .expect("verified snapshot remains")
            .is_some()
    );
}

#[gpui::test]
async fn compaction_retains_verified_head_and_bounded_recent_history() {
    let repository =
        ManagedAgentSnapshotRepository::open_test_database("managed_agent_snapshot_compaction")
            .await;
    let runtime = record();
    let mut snapshots = Vec::new();
    for revision in 1..=4 {
        snapshots.push(
            repository
                .create(
                    runtime.owner_public_key(),
                    &runtime,
                    documents(revision),
                    revision * 10,
                )
                .await
                .expect("create compaction fixture snapshot"),
        );
    }
    assert!(matches!(
        repository
            .compact(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &snapshots[2],
                2,
                50,
            )
            .await
            .expect("stale compaction is an outcome"),
        ManagedAgentSnapshotCompactionOutcome::Stale
    ));
    assert!(matches!(
        repository
            .compact(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &snapshots[3],
                2,
                50,
            )
            .await
            .expect("compact verified history"),
        ManagedAgentSnapshotCompactionOutcome::Compacted { removed: 2, .. }
    ));
    for removed in &snapshots[..2] {
        assert!(
            repository
                .load(
                    runtime.owner_public_key(),
                    runtime.agent_public_key(),
                    removed
                )
                .expect("removed snapshot lookup is valid")
                .is_none()
        );
    }
    for retained in &snapshots[2..] {
        assert!(
            repository
                .load(
                    runtime.owner_public_key(),
                    runtime.agent_public_key(),
                    retained
                )
                .expect("retained snapshot lookup is valid")
                .is_some()
        );
    }
    assert_eq!(
        repository
            .compact(
                runtime.owner_public_key(),
                runtime.agent_public_key(),
                &snapshots[3],
                2,
                60,
            )
            .await
            .expect("bounded history is unchanged"),
        ManagedAgentSnapshotCompactionOutcome::Unchanged
    );
}
