use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use axum::{
    Router,
    body::Bytes,
    extract::{Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use collab::{
    git::{
        object_store::{
            BackendObject, BackendWriteCondition, BackendWriteOutcome, EntityTag, GitObjectBackend,
            GitObjectBackendError, GitObjectStore, GitObjectStoreLimits, GitRefManifest,
            GitRefName,
        },
        repository_registry::{
            HostedAuthority, HostedRepository, HostedRepositoryLifecycle, RepositoryCoordinate,
        },
        smart_http_read::{
            GitReadAuthorizationError, GitReadAuthorizer, GitRequestEncoding,
            GitSmartHttpReadError, GitSmartHttpReadLimits, GitSmartHttpReadService,
            GitUploadPackRequest,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, NostrPublicKey,
    PrincipalId, PrincipalScopes, Provenance, ServiceAccountId, SourceRecordId, SourceSystem,
    TenantContext, TrustedTenantRoute,
};
use flate2::{Compression, write::GzEncoder};
use serde::Deserialize;
use tempfile::TempDir;
use tokio::{process::Command, sync::Mutex};
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
    reads: AtomicUsize,
}

impl MemoryBackend {
    fn reset_reads(&self) {
        self.reads.store(0, Ordering::SeqCst);
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl GitObjectBackend for MemoryBackend {
    async fn get(
        &self,
        key: &str,
        max_bytes: u64,
    ) -> Result<Option<BackendObject>, GitObjectBackendError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
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
        let permitted = match condition {
            BackendWriteCondition::CreateOnly => !state.objects.contains_key(key),
            BackendWriteCondition::IfMatch(expected) => state
                .objects
                .get(key)
                .is_some_and(|value| value.entity_tag == expected),
        };
        if !permitted {
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

struct StaticAuthorizer {
    repository: HostedRepository,
    allowed: AtomicBool,
}

impl StaticAuthorizer {
    fn new(repository: HostedRepository, allowed: bool) -> Self {
        Self {
            repository,
            allowed: AtomicBool::new(allowed),
        }
    }
}

#[async_trait]
impl GitReadAuthorizer for StaticAuthorizer {
    async fn authorize_read(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitReadAuthorizationError> {
        if !self.allowed.load(Ordering::SeqCst)
            || authorization.action != AuthorizationAction::Read
            || authorization.required_scope.as_str() != "git:read"
            || authorization.resource.kind != AuthorizationResourceKind::Repository
            || authorization.resource.resource_id != self.repository.repository_id
            || authorization.resource.community_id != self.repository.community_id
        {
            return Err(GitReadAuthorizationError::Denied);
        }
        Ok(self.repository.clone())
    }
}

struct TestAuthorizationContext {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    scope: AuthorizationScope,
    membership: CommunityMembership,
    repository_id: AggregateId,
}

impl TestAuthorizationContext {
    fn request(&self) -> AuthorizationRequest<'_> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.scope,
            action: AuthorizationAction::Read,
            resource: AuthorizationResource {
                community_id: self.tenant.community_id(),
                kind: AuthorizationResourceKind::Repository,
                resource_id: self.repository_id,
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 1_900_000_000_000,
        }
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn repository_id(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn principal_id(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
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

fn authorization_context(
    community_id: CommunityId,
    repository_id: AggregateId,
) -> TestAuthorizationContext {
    let tenant = bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "git-smart-http-test")
                .expect("tenant route"),
        ),
        &[],
    )
    .expect("tenant");
    let principal_id = principal_id(900);
    let scope = AuthorizationScope::new("git:read").expect("scope");
    let principal = AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(1),
        PrincipalScopes::new([scope.clone()]).expect("scopes"),
    );
    TestAuthorizationContext {
        tenant,
        principal,
        scope,
        membership: CommunityMembership {
            community_id,
            principal_id,
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        },
        repository_id,
    }
}

async fn run_git(directory: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(directory)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .await
        .expect("run git");
    assert!(
        output.status.success(),
        "git {arguments:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

async fn commit_source(source: &Path, contents: &str, message: &str) -> String {
    tokio::fs::write(source.join("hello.txt"), contents)
        .await
        .expect("write source");
    run_git(source, &["add", "hello.txt"]).await;
    run_git(source, &["commit", "--quiet", "-m", message]).await;
    run_git(source, &["repack", "-a", "-d", "--quiet"]).await;
    run_git(source, &["rev-parse", "HEAD"]).await
}

async fn source_pack_paths(source: &Path) -> Vec<PathBuf> {
    let mut entries = tokio::fs::read_dir(source.join(".git/objects/pack"))
        .await
        .expect("pack directory");
    let mut packs = Vec::new();
    while let Some(entry) = entries.next_entry().await.expect("pack entry") {
        if entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("pack")
        {
            packs.push(entry.path());
        }
    }
    packs.sort();
    packs
}

async fn publish_source(
    object_store: &GitObjectStore,
    source: &Path,
    head: &str,
) -> GitRefManifest {
    let mut objects = BTreeSet::new();
    for pack in source_pack_paths(source).await {
        let bytes = tokio::fs::read(pack).await.expect("read pack");
        objects.insert(object_store.put_object(bytes).await.expect("publish pack"));
    }
    let branch = GitRefName::parse("refs/heads/main").expect("branch");
    GitRefManifest::new(
        Some(branch.clone()),
        BTreeMap::from([(
            branch,
            collab::git::object_store::GitObjectId::parse(head).expect("head object"),
        )]),
        objects,
        None,
    )
}

fn service(
    backend: Arc<MemoryBackend>,
    repository: HostedRepository,
    scratch: &TempDir,
    limits: GitSmartHttpReadLimits,
    allowed: bool,
) -> Arc<GitSmartHttpReadService> {
    Arc::new(
        GitSmartHttpReadService::new(
            Arc::new(StaticAuthorizer::new(repository, allowed)),
            backend,
            scratch.path(),
            GitObjectStoreLimits::default(),
            limits,
        )
        .expect("smart HTTP service"),
    )
}

#[tokio::test]
async fn git_smart_http_read_denies_private_and_missing_before_subprocess_work() {
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let community_id = community(1);
    let repository_id = repository_id(2);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(3));
    let authorization = authorization_context(community_id, repository_id);
    let denied = service(
        backend.clone(),
        repository.clone(),
        &scratch,
        GitSmartHttpReadLimits::default(),
        false,
    );
    assert!(matches!(
        denied.advertise_refs(&authorization.request(), None).await,
        Err(GitSmartHttpReadError::RepositoryNotFound)
    ));
    assert!(matches!(
        denied
            .upload_pack(
                &authorization.request(),
                GitUploadPackRequest {
                    content_type: "application/x-git-upload-pack-request".to_owned(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: vec![0; 9 * 1024 * 1024],
                },
            )
            .await,
        Err(GitSmartHttpReadError::RepositoryNotFound)
    ));
    assert_eq!(backend.reads(), 0);

    let allowed = service(
        backend.clone(),
        repository,
        &scratch,
        GitSmartHttpReadLimits::default(),
        true,
    );
    assert!(matches!(
        allowed.advertise_refs(&authorization.request(), None).await,
        Err(GitSmartHttpReadError::RepositoryNotFound)
    ));
    assert_eq!(backend.reads(), 1);
}

#[tokio::test]
async fn git_smart_http_read_rejects_oversized_and_malformed_requests() {
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let community_id = community(10);
    let repository_id = repository_id(11);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(12));
    let authorization = authorization_context(community_id, repository_id);
    let limits = GitSmartHttpReadLimits {
        max_request_bytes: 4,
        max_decoded_request_bytes: 4,
        ..GitSmartHttpReadLimits::default()
    };
    let read_service = service(backend.clone(), repository, &scratch, limits, true);
    backend.reset_reads();
    assert!(matches!(
        read_service
            .upload_pack(
                &authorization.request(),
                GitUploadPackRequest {
                    content_type: "application/x-git-upload-pack-request".to_owned(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: vec![0; 5],
                },
            )
            .await,
        Err(GitSmartHttpReadError::PayloadTooLarge)
    ));
    assert_eq!(backend.reads(), 0);
    assert!(matches!(
        read_service
            .upload_pack(
                &authorization.request(),
                GitUploadPackRequest {
                    content_type: "text/plain".to_owned(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: Vec::new(),
                },
            )
            .await,
        Err(GitSmartHttpReadError::InvalidRequest)
    ));
    assert_eq!(backend.reads(), 0);

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(b"12345").expect("compress request");
    let compressed = encoder.finish().expect("finish compression");
    let gzip_limits = GitSmartHttpReadLimits {
        max_request_bytes: 1024,
        max_decoded_request_bytes: 4,
        ..GitSmartHttpReadLimits::default()
    };
    let gzip_service = service(
        backend.clone(),
        hosted_repository(community_id, repository_id, Uuid::from_u128(12)),
        &scratch,
        gzip_limits,
        true,
    );
    assert!(matches!(
        gzip_service
            .upload_pack(
                &authorization.request(),
                GitUploadPackRequest {
                    content_type: "application/x-git-upload-pack-request".to_owned(),
                    content_encoding: GitRequestEncoding::Gzip,
                    git_protocol: None,
                    body: compressed,
                },
            )
            .await,
        Err(GitSmartHttpReadError::PayloadTooLarge)
    ));
    assert_eq!(backend.reads(), 0);

    let object_store = GitObjectStore::for_authorized_repository(
        backend.clone(),
        &hosted_repository(community_id, repository_id, Uuid::from_u128(12)),
        GitObjectStoreLimits::default(),
    )
    .expect("object store");
    object_store
        .compare_and_swap_refs(None, GitRefManifest::empty(None))
        .await
        .expect("publish empty refs");
    let response_limits = GitSmartHttpReadLimits {
        max_advertisement_bytes: 1,
        ..GitSmartHttpReadLimits::default()
    };
    let response_service = service(
        backend,
        hosted_repository(community_id, repository_id, Uuid::from_u128(12)),
        &scratch,
        response_limits,
        true,
    );
    assert!(matches!(
        response_service
            .advertise_refs(&authorization.request(), None)
            .await,
        Err(GitSmartHttpReadError::PayloadTooLarge)
    ));
}

struct HttpTestState {
    service: Arc<GitSmartHttpReadService>,
    authorization: TestAuthorizationContext,
}

#[derive(Deserialize)]
struct InfoRefsQuery {
    service: String,
}

async fn info_refs_handler(
    State(state): State<Arc<HttpTestState>>,
    Query(query): Query<InfoRefsQuery>,
    headers: HeaderMap,
) -> Response {
    if query.service != "git-upload-pack" {
        return StatusCode::BAD_REQUEST.into_response();
    }
    let protocol = headers
        .get("git-protocol")
        .and_then(|value| value.to_str().ok());
    match state
        .service
        .advertise_refs(&state.authorization.request(), protocol)
        .await
    {
        Ok(response) => smart_http_response(response),
        Err(error) => error_response(error),
    }
}

async fn upload_pack_handler(
    State(state): State<Arc<HttpTestState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_encoding = match headers
        .get(header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
    {
        None | Some("identity") => GitRequestEncoding::Identity,
        Some("gzip") | Some("x-gzip") => GitRequestEncoding::Gzip,
        Some(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let request = GitUploadPackRequest {
        content_type: headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned(),
        content_encoding,
        git_protocol: headers
            .get("git-protocol")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        body: body.to_vec(),
    };
    match state
        .service
        .upload_pack(&state.authorization.request(), request)
        .await
    {
        Ok(response) => smart_http_response(response),
        Err(error) => error_response(error),
    }
}

fn smart_http_response(response: collab::git::smart_http_read::GitSmartHttpResponse) -> Response {
    let mut http_response = response.body.into_response();
    http_response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(response.content_type),
    );
    http_response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(response.cache_control),
    );
    http_response
}

fn error_response(error: GitSmartHttpReadError) -> Response {
    let status = match error {
        GitSmartHttpReadError::RepositoryNotFound => StatusCode::NOT_FOUND,
        GitSmartHttpReadError::InvalidRequest | GitSmartHttpReadError::InvalidConfiguration => {
            StatusCode::BAD_REQUEST
        }
        GitSmartHttpReadError::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        GitSmartHttpReadError::Busy => StatusCode::SERVICE_UNAVAILABLE,
        GitSmartHttpReadError::Timeout => StatusCode::GATEWAY_TIMEOUT,
        GitSmartHttpReadError::Unavailable => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, "repository unavailable").into_response()
}

#[tokio::test]
async fn git_smart_http_read_supports_real_clone_and_fetch() {
    let source = TempDir::new().expect("source");
    run_git(source.path(), &["init", "--quiet", "--initial-branch=main"]).await;
    run_git(source.path(), &["config", "user.email", "git@test.invalid"]).await;
    run_git(source.path(), &["config", "user.name", "Git Test"]).await;
    let first_head = commit_source(source.path(), "first\n", "first").await;

    let community_id = community(20);
    let repository_id = repository_id(21);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(22));
    let authorization = authorization_context(community_id, repository_id);
    let backend = Arc::new(MemoryBackend::default());
    let object_store = GitObjectStore::for_authorized_repository(
        backend.clone(),
        &repository,
        GitObjectStoreLimits::default(),
    )
    .expect("object store");
    let first_manifest = publish_source(&object_store, source.path(), &first_head).await;
    let first_snapshot = object_store
        .compare_and_swap_refs(None, first_manifest)
        .await
        .expect("publish first refs");
    let scratch = TempDir::new().expect("scratch");
    let service = service(
        backend,
        repository,
        &scratch,
        GitSmartHttpReadLimits::default(),
        true,
    );

    let state = Arc::new(HttpTestState {
        service,
        authorization,
    });
    let app = Router::new()
        .route("/repository.git/info/refs", get(info_refs_handler))
        .route("/repository.git/git-upload-pack", post(upload_pack_handler))
        .with_state(state);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind test server");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let address = listener.local_addr().expect("test server address");
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::Server::from_tcp(listener)
            .expect("test server")
            .serve(app.into_make_service())
            .with_graceful_shutdown(async {
                if shutdown_receiver.await.is_err() {
                    return;
                }
            })
            .await
            .expect("serve Git HTTP");
    });

    let clone_parent = TempDir::new().expect("clone parent");
    let clone_path = clone_parent.path().join("clone");
    run_git(
        clone_parent.path(),
        &[
            "clone",
            "--quiet",
            &format!("http://{address}/repository.git"),
            clone_path.to_str().expect("clone path"),
        ],
    )
    .await;
    assert_eq!(
        tokio::fs::read_to_string(clone_path.join("hello.txt"))
            .await
            .expect("cloned file"),
        "first\n"
    );

    let second_head = commit_source(source.path(), "second\n", "second").await;
    let mut second_manifest = publish_source(&object_store, source.path(), &second_head).await;
    second_manifest = GitRefManifest::new(
        second_manifest.head().cloned(),
        second_manifest.refs().clone(),
        second_manifest.objects().clone(),
        Some(first_snapshot.manifest_digest().clone()),
    );
    object_store
        .compare_and_swap_refs(Some(&first_snapshot), second_manifest)
        .await
        .expect("publish second refs");
    run_git(&clone_path, &["fetch", "--quiet", "origin"]).await;
    assert_eq!(
        run_git(&clone_path, &["rev-parse", "origin/main"]).await,
        second_head
    );

    shutdown_sender.send(()).expect("stop test server");
    server.await.expect("join test server");
}
