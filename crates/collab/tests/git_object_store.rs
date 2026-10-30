use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_s3::primitives::ByteStream;
use collab::git::{
    object_store::{
        AwsS3GitObjectBackend, BackendObject, BackendWriteCondition, BackendWriteOutcome,
        EntityTag, GitContentDigest, GitObjectBackend, GitObjectBackendError, GitObjectId,
        GitObjectStore, GitObjectStoreError, GitObjectStoreLimits, GitRefManifest, GitRefName,
    },
    repository_registry::{
        ExternalProviderCoordinate, HostedAuthority, HostedRepository, HostedRepositoryLifecycle,
        RepositoryCoordinate,
    },
};
use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, NostrPublicKey, Provenance, SourceRecordId,
    SourceSystem,
};
use tokio::sync::Mutex;
use uuid::Uuid;

#[derive(Clone)]
struct StoredValue {
    bytes: Vec<u8>,
    entity_tag: EntityTag,
}

#[derive(Default)]
struct MemoryBackendState {
    objects: HashMap<String, StoredValue>,
    next_entity_tag: u64,
}

#[derive(Default)]
struct MemoryBackend {
    state: Mutex<MemoryBackendState>,
}

impl MemoryBackend {
    async fn corrupt_object(&self, digest: &GitContentDigest, bytes: Vec<u8>) {
        let mut state = self.state.lock().await;
        let suffix = format!("/objects/{}", digest.as_str());
        let value = state
            .objects
            .iter_mut()
            .find_map(|(key, value)| key.ends_with(&suffix).then_some(value))
            .expect("stored object");
        value.bytes = bytes;
    }

    async fn keys(&self) -> Vec<String> {
        self.state.lock().await.objects.keys().cloned().collect()
    }

    async fn replace_pointer_bytes(&self, bytes: Vec<u8>) {
        let mut state = self.state.lock().await;
        let value = state
            .objects
            .iter_mut()
            .find_map(|(key, value)| key.ends_with("/refs/pointer").then_some(value))
            .expect("stored pointer");
        value.bytes = bytes;
    }
}

#[async_trait]
impl GitObjectBackend for MemoryBackend {
    async fn get(
        &self,
        key: &str,
        max_bytes: u64,
    ) -> Result<Option<BackendObject>, GitObjectBackendError> {
        let state = self.state.lock().await;
        let Some(value) = state.objects.get(key) else {
            return Ok(None);
        };
        if u64::try_from(value.bytes.len()).map_or(true, |length| length > max_bytes) {
            return Err(GitObjectBackendError::ObjectTooLarge);
        }
        Ok(Some(BackendObject {
            bytes: value.bytes.clone(),
            entity_tag: value.entity_tag.clone(),
        }))
    }

    async fn put(
        &self,
        key: &str,
        bytes: Vec<u8>,
        _content_type: &'static str,
        condition: BackendWriteCondition,
    ) -> Result<BackendWriteOutcome, GitObjectBackendError> {
        let mut state = self.state.lock().await;
        let condition_matches = match condition {
            BackendWriteCondition::CreateOnly => !state.objects.contains_key(key),
            BackendWriteCondition::IfMatch(expected) => state
                .objects
                .get(key)
                .is_some_and(|value| value.entity_tag == expected),
        };
        if !condition_matches {
            return Ok(BackendWriteOutcome::PreconditionFailed);
        }
        state.next_entity_tag += 1;
        let entity_tag = EntityTag::parse(format!("etag-{}", state.next_entity_tag))?;
        state.objects.insert(
            key.to_owned(),
            StoredValue {
                bytes,
                entity_tag: entity_tag.clone(),
            },
        );
        Ok(BackendWriteOutcome::Stored(entity_tag))
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn repository_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn hosted_repository(
    community_id: CommunityId,
    repository_id: AggregateId,
    storage_handle_id: Uuid,
) -> HostedRepository {
    HostedRepository {
        community_id,
        repository_id,
        coordinate: RepositoryCoordinate::new(NostrPublicKey::from_bytes([7; 32]), "repository")
            .expect("coordinate"),
        authority: HostedAuthority::SimHostedNip34 { storage_handle_id },
        authority_version: AggregateVersion::FIRST,
        lifecycle: HostedRepositoryLifecycle::Active,
        provenance: Provenance::new(
            SourceSystem::Zed,
            SourceRecordId::new(format!("repository:{repository_id}")).expect("source record"),
            1_900_000_000_000,
        ),
        archived_at_millis: None,
        created_at_millis: 1_900_000_000_000,
        updated_at_millis: 1_900_000_000_000,
    }
}

fn store(
    backend: Arc<dyn GitObjectBackend>,
    community_value: u128,
    repository_value: u128,
    storage_value: u128,
) -> GitObjectStore {
    GitObjectStore::for_authorized_repository(
        backend,
        &hosted_repository(
            community(community_value),
            repository_id(repository_value),
            Uuid::from_u128(storage_value),
        ),
        GitObjectStoreLimits::default(),
    )
    .expect("object store")
}

fn scoped_object_key(
    community_id: CommunityId,
    repository_id: AggregateId,
    storage_handle_id: Uuid,
    digest: &GitContentDigest,
) -> String {
    format!(
        "collaboration-git/v1/communities/{community_id}/repositories/{repository_id}/storage/{storage_handle_id}/objects/{}",
        digest.as_str()
    )
}

fn branch_manifest(
    branch: &str,
    object_id: char,
    object: GitContentDigest,
    parent: Option<GitContentDigest>,
) -> GitRefManifest {
    let branch = GitRefName::parse(branch).expect("branch");
    GitRefManifest::new(
        Some(branch.clone()),
        BTreeMap::from([(
            branch,
            GitObjectId::parse(object_id.to_string().repeat(40)).expect("object id"),
        )]),
        BTreeSet::from([object]),
        parent,
    )
}

#[tokio::test]
async fn git_object_store_rejects_hash_mismatch_and_missing_objects() {
    let backend = Arc::new(MemoryBackend::default());
    let store = store(backend.clone(), 1, 2, 3);
    let digest = store
        .put_object(b"valid pack bytes".to_vec())
        .await
        .expect("put object");
    assert_eq!(
        store.get_object(&digest).await.expect("read object"),
        b"valid pack bytes"
    );
    backend
        .corrupt_object(&digest, b"corrupt pack bytes".to_vec())
        .await;
    assert!(matches!(
        store.get_object(&digest).await,
        Err(GitObjectStoreError::IntegrityMismatch)
    ));
    assert!(matches!(
        store.put_object(b"valid pack bytes".to_vec()).await,
        Err(GitObjectStoreError::IntegrityMismatch)
    ));

    let missing = GitContentDigest::parse("f".repeat(64)).expect("digest");
    assert!(matches!(
        store.get_object(&missing).await,
        Err(GitObjectStoreError::ObjectNotFound)
    ));
    assert!(matches!(
        store
            .compare_and_swap_refs(None, branch_manifest("refs/heads/main", 'a', missing, None),)
            .await,
        Err(GitObjectStoreError::ObjectNotFound)
    ));
    assert!(matches!(
        store.read_refs().await,
        Err(GitObjectStoreError::RefsNotFound)
    ));

    store
        .compare_and_swap_refs(None, GitRefManifest::empty(None))
        .await
        .expect("publish empty refs");
    let malformed_manifest = br#"{"version":1,"head":"refs/heads/main","refs":{"refs/heads/../../escape":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"objects":[],"parent":null}"#;
    let malformed_digest = GitContentDigest::for_bytes(malformed_manifest);
    let malformed_key = format!(
        "collaboration-git/v1/communities/{}/repositories/{}/storage/{}/manifests/{}",
        community(1),
        repository_id(2),
        Uuid::from_u128(3),
        malformed_digest.as_str()
    );
    assert!(matches!(
        backend
            .put(
                &malformed_key,
                malformed_manifest.to_vec(),
                "application/json",
                BackendWriteCondition::CreateOnly,
            )
            .await,
        Ok(BackendWriteOutcome::Stored(_))
    ));
    backend
        .replace_pointer_bytes(
            format!(
                "{{\"version\":1,\"manifest\":\"{}\"}}",
                malformed_digest.as_str()
            )
            .into_bytes(),
        )
        .await;
    assert!(matches!(
        store.read_refs().await,
        Err(GitObjectStoreError::InvalidManifest)
    ));
}

#[tokio::test]
async fn git_object_store_allows_exactly_one_concurrent_ref_update() {
    let backend = Arc::new(MemoryBackend::default());
    let store = Arc::new(store(backend, 10, 11, 12));
    let initial = store
        .compare_and_swap_refs(None, GitRefManifest::empty(None))
        .await
        .expect("publish empty repository");
    let object_a = store
        .put_object(b"pack a".to_vec())
        .await
        .expect("put object a");
    let object_b = store
        .put_object(b"pack b".to_vec())
        .await
        .expect("put object b");
    let manifest_a = branch_manifest(
        "refs/heads/main",
        'a',
        object_a,
        Some(initial.manifest_digest().clone()),
    );
    let manifest_b = branch_manifest(
        "refs/heads/main",
        'b',
        object_b,
        Some(initial.manifest_digest().clone()),
    );
    let (result_a, result_b) = tokio::join!(
        store.compare_and_swap_refs(Some(&initial), manifest_a),
        store.compare_and_swap_refs(Some(&initial), manifest_b),
    );
    assert_ne!(result_a.is_ok(), result_b.is_ok());
    let loser = if result_a.is_err() {
        result_a.as_ref().expect_err("result a loses")
    } else {
        result_b.as_ref().expect_err("result b loses")
    };
    assert!(matches!(loser, GitObjectStoreError::ConcurrentRefUpdate));
    let winner = result_a.ok().or_else(|| result_b.ok()).expect("one winner");
    let published = store.read_refs().await.expect("read winner");
    assert_eq!(published.manifest_digest(), winner.manifest_digest());
    assert_eq!(published.manifest(), winner.manifest());
}

#[tokio::test]
async fn git_object_store_fences_tenant_repository_and_storage_paths() {
    let backend = Arc::new(MemoryBackend::default());
    let store_a = store(backend.clone(), 20, 21, 22);
    let store_b = store(backend.clone(), 30, 31, 32);
    let digest = store_b
        .put_object(b"tenant b pack".to_vec())
        .await
        .expect("put tenant b object");
    assert!(matches!(
        store_a.get_object(&digest).await,
        Err(GitObjectStoreError::ObjectNotFound)
    ));
    let snapshot_a = store_a
        .compare_and_swap_refs(None, GitRefManifest::empty(None))
        .await
        .expect("publish tenant a refs");
    assert!(matches!(
        store_b
            .compare_and_swap_refs(
                Some(&snapshot_a),
                GitRefManifest::empty(Some(snapshot_a.manifest_digest().clone())),
            )
            .await,
        Err(GitObjectStoreError::ConcurrentRefUpdate)
    ));

    let keys = backend.keys().await;
    assert!(keys.iter().all(|key| !key.contains("..")));
    assert!(
        keys.iter()
            .any(|key| key.contains(&community(20).to_string()))
    );
    assert!(
        keys.iter()
            .any(|key| key.contains(&community(30).to_string()))
    );
    assert!(keys.iter().all(|key| {
        key.starts_with("collaboration-git/v1/communities/")
            && (key.contains(&repository_id(21).to_string())
                || key.contains(&repository_id(31).to_string()))
    }));
}

#[tokio::test]
async fn git_object_store_rejects_external_archived_and_oversized_inputs() {
    let backend = Arc::new(MemoryBackend::default());
    let mut external = hosted_repository(community(40), repository_id(41), Uuid::from_u128(42));
    external.authority = HostedAuthority::ExternalProvider(
        ExternalProviderCoordinate::new("github", "github.com", "owner", "repository")
            .expect("external coordinate"),
    );
    assert!(matches!(
        GitObjectStore::for_authorized_repository(
            backend.clone(),
            &external,
            GitObjectStoreLimits::default(),
        ),
        Err(GitObjectStoreError::UnsupportedAuthority)
    ));
    let mut archived = hosted_repository(community(40), repository_id(43), Uuid::from_u128(44));
    archived.lifecycle = HostedRepositoryLifecycle::Archived;
    archived.archived_at_millis = Some(1_900_000_000_001);
    assert!(matches!(
        GitObjectStore::for_authorized_repository(
            backend.clone(),
            &archived,
            GitObjectStoreLimits::default(),
        ),
        Err(GitObjectStoreError::RepositoryUnavailable)
    ));

    let limits = GitObjectStoreLimits {
        max_object_bytes: 4,
        ..GitObjectStoreLimits::default()
    };
    let store = GitObjectStore::for_authorized_repository(
        backend,
        &hosted_repository(community(40), repository_id(45), Uuid::from_u128(46)),
        limits,
    )
    .expect("bounded store");
    assert!(matches!(
        store.put_object(vec![0; 5]).await,
        Err(GitObjectStoreError::ObjectTooLarge)
    ));
    assert!(matches!(
        GitRefName::parse("refs/heads/../../tenant-b"),
        Err(GitObjectStoreError::InvalidRefName)
    ));
}

#[tokio::test]
async fn git_object_store_live_s3_verifies_integrity_and_conditional_ref_updates() {
    let Some(endpoint) = std::env::var("COLLAB_GIT_OBJECT_STORE_TEST_ENDPOINT").ok() else {
        eprintln!("COLLAB_GIT_OBJECT_STORE_TEST_ENDPOINT is unset; live S3 test skipped");
        return;
    };
    let access_key =
        std::env::var("COLLAB_GIT_OBJECT_STORE_TEST_ACCESS_KEY").expect("live S3 access key");
    let secret_key =
        std::env::var("COLLAB_GIT_OBJECT_STORE_TEST_SECRET_KEY").expect("live S3 secret key");
    let bucket = format!("collab-git-object-store-{}", Uuid::new_v4());
    let credentials = aws_sdk_s3::config::Credentials::new(
        access_key,
        secret_key,
        None,
        None,
        "git-object-store-test",
    );
    let shared_config = aws_config::defaults(BehaviorVersion::latest())
        .endpoint_url(endpoint)
        .region(Region::new("us-east-1"))
        .credentials_provider(credentials)
        .load()
        .await;
    let s3_config = aws_sdk_s3::config::Builder::from(&shared_config)
        .force_path_style(true)
        .build();
    let client = aws_sdk_s3::Client::from_conf(s3_config);
    client
        .create_bucket()
        .bucket(&bucket)
        .send()
        .await
        .expect("create isolated bucket");

    let backend =
        Arc::new(AwsS3GitObjectBackend::new(client.clone(), &bucket).expect("S3 object backend"));
    let community_id = community(100);
    let repository_id = repository_id(101);
    let storage_handle_id = Uuid::from_u128(102);
    let object_store = Arc::new(
        GitObjectStore::for_authorized_repository(
            backend.clone(),
            &hosted_repository(community_id, repository_id, storage_handle_id),
            GitObjectStoreLimits::default(),
        )
        .expect("live object store"),
    );
    let initial = object_store
        .compare_and_swap_refs(None, GitRefManifest::empty(None))
        .await
        .expect("publish initial manifest");
    let object_a = object_store
        .put_object(b"live pack a".to_vec())
        .await
        .expect("put live object a");
    let object_b = object_store
        .put_object(b"live pack b".to_vec())
        .await
        .expect("put live object b");
    assert_eq!(
        object_store
            .get_object(&object_a)
            .await
            .expect("read live object"),
        b"live pack a"
    );
    let update_a = branch_manifest(
        "refs/heads/main",
        'a',
        object_a.clone(),
        Some(initial.manifest_digest().clone()),
    );
    let update_b = branch_manifest(
        "refs/heads/main",
        'b',
        object_b,
        Some(initial.manifest_digest().clone()),
    );
    let (result_a, result_b) = tokio::join!(
        object_store.compare_and_swap_refs(Some(&initial), update_a),
        object_store.compare_and_swap_refs(Some(&initial), update_b),
    );
    assert_ne!(result_a.is_ok(), result_b.is_ok());
    assert!(
        [result_a.as_ref().err(), result_b.as_ref().err()]
            .into_iter()
            .flatten()
            .any(|error| matches!(error, GitObjectStoreError::ConcurrentRefUpdate))
    );

    let other_tenant = store(backend, 110, 111, 112);
    assert!(matches!(
        other_tenant.get_object(&object_a).await,
        Err(GitObjectStoreError::ObjectNotFound)
    ));
    client
        .put_object()
        .bucket(&bucket)
        .key(scoped_object_key(
            community_id,
            repository_id,
            storage_handle_id,
            &object_a,
        ))
        .body(ByteStream::from_static(b"corrupt live bytes"))
        .send()
        .await
        .expect("inject isolated corruption");
    assert!(matches!(
        object_store.get_object(&object_a).await,
        Err(GitObjectStoreError::IntegrityMismatch)
    ));

    let listed = client
        .list_objects_v2()
        .bucket(&bucket)
        .send()
        .await
        .expect("list isolated bucket");
    for object in listed.contents() {
        if let Some(key) = object.key() {
            client
                .delete_object()
                .bucket(&bucket)
                .key(key)
                .send()
                .await
                .expect("remove isolated object");
        }
    }
    client
        .delete_bucket()
        .bucket(&bucket)
        .send()
        .await
        .expect("remove isolated bucket");
}
