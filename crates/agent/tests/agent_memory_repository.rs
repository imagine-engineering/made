use agent::{
    AgentMemoryRepository, AgentMemoryRepositoryError, MemoryRetention, MemoryRetentionOutcome,
    MemoryRotationOutcome, MemoryWriteOutcome, StoredEncryptedMemory,
};
use nostr_compat::agent_memory::{decrypt_engram_as_owner, encrypt_engram_for_owner};
use nostr_compat::buzz_nips::agent_activity::EngramBody;
use nostr_compat::{EventId, PublicKey};

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

fn record(
    owner: PublicKey,
    owner_secret: &[u8; 32],
    slug: &str,
    value: &str,
    event_byte: u8,
    created_at: u64,
    retention: MemoryRetention,
) -> StoredEncryptedMemory {
    let encrypted = encrypt_engram_for_owner(
        &AGENT_SECRET,
        owner,
        &EngramBody::Memory {
            slug: slug.to_owned(),
            value: Some(value.to_owned()),
        },
    )
    .expect("encrypt fixture engram");
    let stored = StoredEncryptedMemory::new(
        &encrypted,
        EventId::from_bytes([event_byte; 32]),
        created_at,
        retention,
    )
    .expect("create stored fixture");
    assert_eq!(
        decrypt_engram_as_owner(owner_secret, stored.coordinate(), stored.ciphertext())
            .expect("fixture owner decrypt"),
        EngramBody::Memory {
            slug: slug.to_owned(),
            value: Some(value.to_owned()),
        }
    );
    stored
}

#[gpui::test]
async fn owner_reads_persisted_ciphertext_after_restart(cx: &mut gpui::TestAppContext) {
    cx.executor().allow_parking();
    let database_directory = tempfile::tempdir().expect("create fixture database directory");
    let repository =
        AgentMemoryRepository::open_test_file_database(database_directory.path()).await;
    let memory = record(
        owner(),
        &OWNER_SECRET,
        "mem/persisted",
        "owner-only plaintext",
        1,
        10,
        MemoryRetention::new(1, None).expect("fixture retention"),
    );
    assert_eq!(
        repository
            .store(owner(), &memory)
            .await
            .expect("store memory"),
        MemoryWriteOutcome::Stored
    );
    assert!(matches!(
        repository.load_for_owner(rotated_owner(), memory.coordinate(), 11),
        Err(AgentMemoryRepositoryError::OwnerMismatch)
    ));
    drop(repository);

    let restarted = AgentMemoryRepository::open_test_file_database(database_directory.path()).await;
    let loaded = restarted
        .load_for_owner(owner(), memory.coordinate(), 11)
        .expect("load memory")
        .expect("stored memory exists");
    assert_eq!(loaded, memory);
    assert_eq!(
        decrypt_engram_as_owner(&OWNER_SECRET, loaded.coordinate(), loaded.ciphertext())
            .expect("owner decrypts loaded ciphertext"),
        EngramBody::Memory {
            slug: "mem/persisted".to_owned(),
            value: Some("owner-only plaintext".to_owned()),
        }
    );
    assert!(!format!("{loaded:?}").contains("owner-only plaintext"));
    assert!(!format!("{loaded:?}").contains(loaded.ciphertext().wire_value()));
}

#[gpui::test]
async fn ciphertext_corruption_fails_closed_without_removing_evidence() {
    let repository = AgentMemoryRepository::open_test_database("agent_memory_integrity").await;
    let memory = record(
        owner(),
        &OWNER_SECRET,
        "mem/integrity",
        "integrity evidence",
        2,
        20,
        MemoryRetention::new(1, None).expect("fixture retention"),
    );
    repository
        .store(owner(), &memory)
        .await
        .expect("store memory");
    repository
        .corrupt_ciphertext_for_test(memory.coordinate())
        .await
        .expect("corrupt stored ciphertext");

    assert!(matches!(
        repository.load_for_owner(owner(), memory.coordinate(), 21),
        Err(AgentMemoryRepositoryError::CorruptCiphertext)
    ));
    assert_eq!(
        repository
            .store(owner(), &memory)
            .await
            .expect("retry source event"),
        MemoryWriteOutcome::AlreadyCurrent
    );
    assert!(matches!(
        repository.load_for_owner(owner(), memory.coordinate(), 21),
        Err(AgentMemoryRepositoryError::CorruptCiphertext)
    ));
}

#[gpui::test]
async fn retention_expiry_converges_reads_and_rejects_stale_updates() {
    let repository = AgentMemoryRepository::open_test_database("agent_memory_expiry").await;
    let memory = record(
        owner(),
        &OWNER_SECRET,
        "mem/expiry",
        "short lived",
        3,
        30,
        MemoryRetention::new(1, None).expect("fixture retention"),
    );
    repository
        .store(owner(), &memory)
        .await
        .expect("store memory");
    assert_eq!(
        repository
            .expire(owner(), memory.coordinate(), 1, 50)
            .await
            .expect("expire memory"),
        MemoryRetentionOutcome::Applied
    );
    assert_eq!(
        repository
            .expire(owner(), memory.coordinate(), 1, 60)
            .await
            .expect("stale expiry is an outcome"),
        MemoryRetentionOutcome::Stale
    );
    assert!(
        repository
            .load_for_owner(owner(), memory.coordinate(), 49)
            .expect("load before expiry")
            .is_some()
    );
    assert!(
        repository
            .load_for_owner(owner(), memory.coordinate(), 50)
            .expect("load at expiry")
            .is_none()
    );

    let later_head = record(
        owner(),
        &OWNER_SECRET,
        "mem/expiry",
        "later event",
        9,
        40,
        MemoryRetention::new(1, None).expect("fixture retention"),
    );
    assert_eq!(
        repository
            .store(owner(), &later_head)
            .await
            .expect("store later head"),
        MemoryWriteOutcome::Stored
    );
    assert!(
        repository
            .load_for_owner(owner(), later_head.coordinate(), 50)
            .expect("later event preserves expiry")
            .is_none()
    );
    assert_eq!(
        repository
            .expire(owner(), later_head.coordinate(), 1, 60)
            .await
            .expect("later event does not reset retention generation"),
        MemoryRetentionOutcome::Stale
    );
}

#[gpui::test]
async fn owner_rotation_atomically_expires_old_and_installs_reencrypted_memory() {
    let repository = AgentMemoryRepository::open_test_database("agent_memory_rotation").await;
    let previous = record(
        owner(),
        &OWNER_SECRET,
        "mem/rotation",
        "rotation payload",
        4,
        40,
        MemoryRetention::new(1, None).expect("fixture retention"),
    );
    let replacement = record(
        rotated_owner(),
        &ROTATED_OWNER_SECRET,
        "mem/rotation",
        "rotation payload",
        5,
        41,
        MemoryRetention::new(1, None).expect("fixture retention"),
    );
    repository
        .store(owner(), &previous)
        .await
        .expect("store previous memory");

    assert_eq!(
        repository
            .rotate_owner(owner(), previous.coordinate(), 1, &replacement, 100)
            .await
            .expect("rotate owner"),
        MemoryRotationOutcome::Applied
    );
    assert_eq!(
        repository
            .rotate_owner(owner(), previous.coordinate(), 1, &replacement, 100)
            .await
            .expect("retry owner rotation"),
        MemoryRotationOutcome::AlreadyApplied
    );
    assert!(
        repository
            .load_for_owner(owner(), previous.coordinate(), 100)
            .expect("old owner read")
            .is_none()
    );
    let loaded = repository
        .load_for_owner(rotated_owner(), replacement.coordinate(), 100)
        .expect("new owner read")
        .expect("replacement exists");
    assert_eq!(loaded, replacement);
    assert_eq!(
        decrypt_engram_as_owner(
            &ROTATED_OWNER_SECRET,
            loaded.coordinate(),
            loaded.ciphertext(),
        )
        .expect("new owner decrypts replacement"),
        EngramBody::Memory {
            slug: "mem/rotation".to_owned(),
            value: Some("rotation payload".to_owned()),
        }
    );
}
