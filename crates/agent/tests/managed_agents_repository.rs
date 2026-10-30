use agent::{
    ManagedAgentCasOutcome, ManagedAgentInsertOutcome, ManagedAgentRepository,
    ManagedAgentRepositoryError, ProjectionWriteOutcome,
};
use agent_settings::{
    managed_agent::{
        EnvironmentReference, EnvironmentVariableName, ManagedAgentConfiguration, ModelId,
        PrivateManagedAgentRecord, ProtectedCredentialReference, ProviderId, RuntimeId,
    },
    team::{NostrEventId as SettingsEventId, NostrPublicKey as SettingsPublicKey},
};
use collaboration_domain::{
    NostrPublicKey as DomainPublicKey, PublicAgentCatalogProjection, PublicPersonaProjection,
};
use std::collections::BTreeMap;

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn settings_public_key(byte: u8) -> SettingsPublicKey {
    SettingsPublicKey::parse(lower_hex(&[byte; 32])).expect("fixture public key must be valid")
}

fn settings_event_id(byte: u8) -> SettingsEventId {
    SettingsEventId::parse(lower_hex(&[byte; 32])).expect("fixture event ID must be valid")
}

fn configuration() -> ManagedAgentConfiguration {
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
        Some(ModelId::parse("claude-opus-4-1").expect("fixture model must be valid")),
        environment,
    )
    .expect("fixture configuration must be valid")
}

fn record() -> PrivateManagedAgentRecord {
    PrivateManagedAgentRecord::new(
        settings_public_key(1),
        settings_public_key(2),
        settings_event_id(3),
        configuration(),
    )
    .expect("fixture managed agent must be valid")
}

fn public_projection() -> PublicAgentCatalogProjection {
    PublicAgentCatalogProjection {
        owner_public_key: DomainPublicKey::from_bytes([1; 32]),
        personas: vec![PublicPersonaProjection {
            slug: "reviewer".to_string(),
            display_name: "Reviewer".to_string(),
            description: "Reviews changes".to_string(),
            system_prompt: Some("Review carefully".to_string()),
            avatar_url: None,
            runtime: Some("claude-code".to_string()),
            model: Some("claude-opus-4-1".to_string()),
            provider: Some("anthropic".to_string()),
        }],
        teams: Vec::new(),
    }
}

#[gpui::test]
async fn repository_compare_and_swap_is_exact_and_clears_projection() {
    let repository =
        ManagedAgentRepository::open_test_database("managed_agent_repository_cas").await;
    let mut record = record();
    assert_eq!(
        repository.insert(&record).await.expect("insert snapshot"),
        ManagedAgentInsertOutcome::Inserted
    );
    repository
        .rebuild_public_projection(&record, &public_projection(), 10)
        .await
        .expect("store projection");
    let expected = record.version().clone();
    record
        .replace(&expected, settings_event_id(4), configuration())
        .expect("advance fixture record");

    assert_eq!(
        repository
            .compare_and_swap(&expected, &record)
            .await
            .expect("apply current CAS"),
        ManagedAgentCasOutcome::Applied
    );
    assert!(
        repository
            .load_public_projection(record.owner_public_key(), record.agent_public_key())
            .expect("load invalidated projection")
            .is_none()
    );
    assert_eq!(
        repository
            .compare_and_swap(&expected, &record)
            .await
            .expect("stale CAS is an outcome"),
        ManagedAgentCasOutcome::Stale
    );
}

#[gpui::test]
async fn snapshot_survives_repository_restart(cx: &mut gpui::TestAppContext) {
    cx.executor().allow_parking();
    let database_directory = tempfile::tempdir().expect("create fixture database directory");
    let repository =
        ManagedAgentRepository::open_test_file_database(database_directory.path()).await;
    let record = record();
    repository.insert(&record).await.expect("insert snapshot");
    drop(repository);
    let restarted =
        ManagedAgentRepository::open_test_file_database(database_directory.path()).await;

    assert_eq!(
        restarted
            .load(record.owner_public_key(), record.agent_public_key())
            .expect("load restarted snapshot"),
        Some(record)
    );
}

#[gpui::test]
async fn missing_projection_rebuilds_from_current_snapshot() {
    let repository =
        ManagedAgentRepository::open_test_database("managed_agent_repository_projection_rebuild")
            .await;
    let record = record();
    repository.insert(&record).await.expect("insert snapshot");
    let projection = public_projection();
    assert_eq!(
        repository
            .rebuild_public_projection(&record, &projection, 10)
            .await
            .expect("store projection"),
        ProjectionWriteOutcome::Stored
    );
    repository
        .invalidate_public_projection(record.owner_public_key(), record.agent_public_key())
        .await
        .expect("remove derived projection");
    assert!(
        repository
            .load_public_projection(record.owner_public_key(), record.agent_public_key())
            .expect("missing projection is recoverable")
            .is_none()
    );

    repository
        .rebuild_public_projection(&record, &projection, 11)
        .await
        .expect("rebuild projection");
    let rebuilt = repository
        .load_public_projection(record.owner_public_key(), record.agent_public_key())
        .expect("load rebuilt projection")
        .expect("rebuilt projection exists");
    assert_eq!(rebuilt.source_version, *record.version());
    assert_eq!(rebuilt.projection_revision, 1);
    assert_eq!(rebuilt.projected_at, 11);
    assert_eq!(rebuilt.projection["personas"][0]["slug"], "reviewer");
}

#[gpui::test]
async fn corrupt_snapshot_fails_closed_without_deleting_source_row() {
    let repository =
        ManagedAgentRepository::open_test_database("managed_agent_repository_corrupt_snapshot")
            .await;
    let record = record();
    repository.insert(&record).await.expect("insert snapshot");
    repository
        .corrupt_snapshot_for_test(record.owner_public_key(), record.agent_public_key())
        .await
        .expect("corrupt fixture snapshot");

    assert!(matches!(
        repository.load(record.owner_public_key(), record.agent_public_key()),
        Err(ManagedAgentRepositoryError::CorruptSnapshot)
    ));
    assert_eq!(
        repository
            .insert(&record)
            .await
            .expect("source row remains"),
        ManagedAgentInsertOutcome::AlreadyExists
    );
}
