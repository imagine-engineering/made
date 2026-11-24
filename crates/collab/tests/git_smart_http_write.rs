use std::{
    collections::HashMap,
    path::Path,
    process::Output,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
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
            GitObjectBackendError, GitObjectStore, GitObjectStoreLimits,
        },
        repository_registry::{
            HostedAuthority, HostedRepository, HostedRepositoryLifecycle, RepositoryCoordinate,
        },
        smart_http_read::GitRequestEncoding,
        smart_http_write::{
            GitForcePushPolicy, GitPushReceipt, GitReceivePackRequest, GitSmartHttpWriteError,
            GitSmartHttpWriteLimits, GitSmartHttpWriteResponse, GitSmartHttpWriteService,
            GitWriteAuthorizationError, GitWriteAuthorizer,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, NostrPublicKey,
    OperationId, PrincipalId, PrincipalScopes, Provenance, ServiceAccountId, SourceRecordId,
    SourceSystem, TenantContext, TrustedTenantRoute,
};
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
    fail_next_pointer_compare_and_swap: AtomicBool,
}

impl MemoryBackend {
    fn reset_reads(&self) {
        self.reads.store(0, Ordering::SeqCst);
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }

    fn fail_next_pointer_compare_and_swap(&self) {
        self.fail_next_pointer_compare_and_swap
            .store(true, Ordering::SeqCst);
    }

    async fn remove_object(&self, digest: &str) {
        self.state
            .lock()
            .await
            .objects
            .retain(|key, _| !key.ends_with(&format!("/objects/{digest}")));
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
        if key.ends_with("/refs/pointer")
            && matches!(condition, BackendWriteCondition::IfMatch(_))
            && self
                .fail_next_pointer_compare_and_swap
                .swap(false, Ordering::SeqCst)
        {
            return Ok(BackendWriteOutcome::PreconditionFailed);
        }
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
    allowed: bool,
}

#[async_trait]
impl GitWriteAuthorizer for StaticAuthorizer {
    async fn authorize_write(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitWriteAuthorizationError> {
        if !self.allowed
            || authorization.action != AuthorizationAction::Write
            || authorization.required_scope.as_str() != "git:write"
            || authorization.resource.kind != AuthorizationResourceKind::Repository
            || authorization.resource.resource_id != self.repository.repository_id
            || authorization.resource.community_id != self.repository.community_id
        {
            return Err(GitWriteAuthorizationError::Denied);
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
            action: AuthorizationAction::Write,
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

fn operation_id(value: u128) -> OperationId {
    OperationId::from_uuid(Uuid::from_u128(value))
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
            TrustedTenantRoute::from_listener(community_id, "git-smart-http-write-test")
                .expect("tenant route"),
        ),
        &[],
    )
    .expect("tenant");
    let principal_id = principal_id(900);
    let scope = AuthorizationScope::new("git:write").expect("scope");
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

fn service(
    backend: Arc<MemoryBackend>,
    repository: HostedRepository,
    scratch: &TempDir,
    allowed: bool,
    force_policy: GitForcePushPolicy,
) -> Arc<GitSmartHttpWriteService> {
    Arc::new(
        GitSmartHttpWriteService::new(
            Arc::new(StaticAuthorizer {
                repository,
                allowed,
            }),
            backend,
            scratch.path(),
            GitObjectStoreLimits::default(),
            GitSmartHttpWriteLimits::default(),
            force_policy,
        )
        .expect("smart HTTP write service"),
    )
}

async fn run_git(directory: &Path, arguments: &[&str]) -> String {
    let output = run_git_output(directory, arguments).await;
    assert!(
        output.status.success(),
        "git {arguments:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

async fn run_git_output(directory: &Path, arguments: &[&str]) -> Output {
    Command::new("git")
        .args(arguments)
        .current_dir(directory)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .await
        .expect("run git")
}

async fn commit_source(source: &Path, contents: &str, message: &str) -> String {
    tokio::fs::write(source.join("hello.txt"), contents)
        .await
        .expect("write source");
    run_git(source, &["add", "hello.txt"]).await;
    run_git(source, &["commit", "--quiet", "-m", message]).await;
    run_git(source, &["rev-parse", "HEAD"]).await
}

struct HttpTestState {
    service: Arc<GitSmartHttpWriteService>,
    authorization: TestAuthorizationContext,
    operation_sequence: AtomicU64,
    receipts: Mutex<Vec<GitPushReceipt>>,
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
    if query.service != "git-receive-pack" {
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

async fn receive_pack_handler(
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
    let operation_sequence = state.operation_sequence.fetch_add(1, Ordering::SeqCst);
    let request = GitReceivePackRequest {
        operation_id: operation_id(u128::from(operation_sequence) + 1_000),
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
        .receive_pack(&state.authorization.request(), request)
        .await
    {
        Ok(response) => {
            if let Some(receipt) = response.receipt.clone() {
                state.receipts.lock().await.push(receipt);
            }
            smart_http_response(response)
        }
        Err(error) => error_response(error),
    }
}

fn smart_http_response(response: GitSmartHttpWriteResponse) -> Response {
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

fn error_response(error: GitSmartHttpWriteError) -> Response {
    let status = match error {
        GitSmartHttpWriteError::RepositoryNotFound => StatusCode::NOT_FOUND,
        GitSmartHttpWriteError::InvalidRequest | GitSmartHttpWriteError::InvalidConfiguration => {
            StatusCode::BAD_REQUEST
        }
        GitSmartHttpWriteError::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        GitSmartHttpWriteError::Busy => StatusCode::SERVICE_UNAVAILABLE,
        GitSmartHttpWriteError::Timeout => StatusCode::GATEWAY_TIMEOUT,
        GitSmartHttpWriteError::ConcurrentUpdate => StatusCode::CONFLICT,
        GitSmartHttpWriteError::MissingObject | GitSmartHttpWriteError::Unavailable => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    (status, "repository unavailable").into_response()
}

async fn start_server(
    service: Arc<GitSmartHttpWriteService>,
    authorization: TestAuthorizationContext,
) -> (String, Arc<HttpTestState>, tokio::sync::oneshot::Sender<()>) {
    let state = Arc::new(HttpTestState {
        service,
        authorization,
        operation_sequence: AtomicU64::new(0),
        receipts: Mutex::new(Vec::new()),
    });
    let app = Router::new()
        .route("/repository/info/refs", get(info_refs_handler))
        .route("/repository/git-receive-pack", post(receive_pack_handler))
        .with_state(state.clone());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind HTTP test server");
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");
    let address = listener.local_addr().expect("server address");
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        axum::Server::from_tcp(listener)
            .expect("test server")
            .serve(app.into_make_service())
            .with_graceful_shutdown(async {
                let _shutdown_result = shutdown_receiver.await;
            })
            .await
            .expect("serve Git HTTP test");
    });
    (
        format!("http://{address}/repository"),
        state,
        shutdown_sender,
    )
}

#[tokio::test]
async fn git_smart_http_write_denies_writer_before_request_or_storage_work() {
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let community_id = community(1);
    let repository_id = repository_id(2);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(3));
    let authorization = authorization_context(community_id, repository_id);
    let service = service(
        backend.clone(),
        repository,
        &scratch,
        false,
        GitForcePushPolicy::FastForwardOnly,
    );
    backend.reset_reads();
    assert!(matches!(
        service.advertise_refs(&authorization.request(), None).await,
        Err(GitSmartHttpWriteError::RepositoryNotFound)
    ));
    assert!(matches!(
        service
            .receive_pack(
                &authorization.request(),
                GitReceivePackRequest {
                    operation_id: operation_id(0),
                    content_type: "text/plain".to_owned(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: vec![0],
                },
            )
            .await,
        Err(GitSmartHttpWriteError::RepositoryNotFound)
    ));
    assert_eq!(backend.reads(), 0);
}

#[tokio::test]
async fn git_smart_http_write_supports_fast_forward_and_enforces_force_policy() {
    let source = TempDir::new().expect("source");
    run_git(source.path(), &["init", "--quiet", "--initial-branch=main"]).await;
    run_git(source.path(), &["config", "user.email", "git@test.invalid"]).await;
    run_git(source.path(), &["config", "user.name", "Git Test"]).await;
    let first_head = commit_source(source.path(), "first\n", "first").await;

    let community_id = community(20);
    let repository_id = repository_id(21);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(22));
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let write_service = service(
        backend.clone(),
        repository.clone(),
        &scratch,
        true,
        GitForcePushPolicy::FastForwardOnly,
    );
    let (remote_url, state, shutdown) = start_server(
        write_service,
        authorization_context(community_id, repository_id),
    )
    .await;
    run_git(source.path(), &["remote", "add", "origin", &remote_url]).await;
    run_git(source.path(), &["push", "--quiet", "origin", "main"]).await;
    let second_head = commit_source(source.path(), "second\n", "second").await;
    run_git(source.path(), &["push", "--quiet", "origin", "main"]).await;

    let object_store = GitObjectStore::for_authorized_repository(
        backend.clone(),
        &repository,
        GitObjectStoreLimits::default(),
    )
    .expect("object store");
    let snapshot = object_store.read_refs().await.expect("published refs");
    assert_eq!(
        snapshot
            .manifest()
            .refs()
            .get(&collab::git::object_store::GitRefName::parse("refs/heads/main").expect("ref"))
            .expect("main")
            .as_str(),
        second_head
    );

    run_git(source.path(), &["reset", "--hard", &first_head]).await;
    let divergent_head = commit_source(source.path(), "divergent\n", "divergent").await;
    let rejected = run_git_output(source.path(), &["push", "--force", "origin", "main"]).await;
    assert!(!rejected.status.success());
    assert_eq!(
        object_store
            .read_refs()
            .await
            .expect("refs after denied force")
            .manifest()
            .refs()
            .get(&collab::git::object_store::GitRefName::parse("refs/heads/main").expect("ref"))
            .expect("main")
            .as_str(),
        second_head
    );
    let receipts = state.receipts.lock().await;
    assert_eq!(receipts.len(), 3);
    assert!(receipts[0].applied && receipts[1].applied);
    assert!(!receipts[2].applied);
    assert!(
        receipts
            .iter()
            .all(|receipt| !receipt.operation_id.as_uuid().is_nil())
    );
    drop(receipts);
    shutdown.send(()).expect("stop fast-forward server");

    let force_service = service(
        backend.clone(),
        repository.clone(),
        &scratch,
        true,
        GitForcePushPolicy::AllowForce,
    );
    let (force_url, force_state, force_shutdown) = start_server(
        force_service,
        authorization_context(community_id, repository_id),
    )
    .await;
    run_git(source.path(), &["remote", "set-url", "origin", &force_url]).await;
    run_git(
        source.path(),
        &["push", "--quiet", "--force", "origin", "main"],
    )
    .await;
    assert_eq!(
        object_store
            .read_refs()
            .await
            .expect("refs after allowed force")
            .manifest()
            .refs()
            .get(&collab::git::object_store::GitRefName::parse("refs/heads/main").expect("ref"))
            .expect("main")
            .as_str(),
        divergent_head
    );
    assert!(force_state.receipts.lock().await[0].applied);
    force_shutdown.send(()).expect("stop force server");
}

#[tokio::test]
async fn git_smart_http_write_rejects_missing_parent_object() {
    let source = TempDir::new().expect("source");
    run_git(source.path(), &["init", "--quiet", "--initial-branch=main"]).await;
    run_git(source.path(), &["config", "user.email", "git@test.invalid"]).await;
    run_git(source.path(), &["config", "user.name", "Git Test"]).await;
    commit_source(source.path(), "first\n", "first").await;

    let community_id = community(30);
    let repository_id = repository_id(31);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(32));
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let write_service = service(
        backend.clone(),
        repository.clone(),
        &scratch,
        true,
        GitForcePushPolicy::FastForwardOnly,
    );
    let (remote_url, _state, shutdown) = start_server(
        write_service.clone(),
        authorization_context(community_id, repository_id),
    )
    .await;
    run_git(source.path(), &["remote", "add", "origin", &remote_url]).await;
    run_git(source.path(), &["push", "--quiet", "origin", "main"]).await;
    shutdown.send(()).expect("stop server");

    let object_store = GitObjectStore::for_authorized_repository(
        backend.clone(),
        &repository,
        GitObjectStoreLimits::default(),
    )
    .expect("object store");
    let snapshot = object_store.read_refs().await.expect("published refs");
    let digest = snapshot
        .manifest()
        .objects()
        .first()
        .expect("pack digest")
        .as_str()
        .to_owned();
    backend.remove_object(&digest).await;
    assert!(matches!(
        write_service
            .receive_pack(
                &authorization_context(community_id, repository_id).request(),
                GitReceivePackRequest {
                    operation_id: operation_id(40),
                    content_type: "application/x-git-receive-pack-request".to_owned(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: Vec::new(),
                },
            )
            .await,
        Err(GitSmartHttpWriteError::MissingObject)
    ));
}

#[tokio::test]
async fn git_smart_http_write_fails_closed_when_manifest_compare_and_swap_loses() {
    let source = TempDir::new().expect("source");
    run_git(source.path(), &["init", "--quiet", "--initial-branch=main"]).await;
    run_git(source.path(), &["config", "user.email", "git@test.invalid"]).await;
    run_git(source.path(), &["config", "user.name", "Git Test"]).await;
    let first_head = commit_source(source.path(), "first\n", "first").await;

    let community_id = community(50);
    let repository_id = repository_id(51);
    let repository = hosted_repository(community_id, repository_id, Uuid::from_u128(52));
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let write_service = service(
        backend.clone(),
        repository.clone(),
        &scratch,
        true,
        GitForcePushPolicy::FastForwardOnly,
    );
    let (remote_url, state, shutdown) = start_server(
        write_service,
        authorization_context(community_id, repository_id),
    )
    .await;
    run_git(source.path(), &["remote", "add", "origin", &remote_url]).await;
    run_git(source.path(), &["push", "--quiet", "origin", "main"]).await;
    commit_source(source.path(), "second\n", "second").await;
    backend.fail_next_pointer_compare_and_swap();
    let rejected = run_git_output(source.path(), &["push", "origin", "main"]).await;
    assert!(!rejected.status.success());

    let object_store = GitObjectStore::for_authorized_repository(
        backend,
        &repository,
        GitObjectStoreLimits::default(),
    )
    .expect("object store");
    assert_eq!(
        object_store
            .read_refs()
            .await
            .expect("winning refs")
            .manifest()
            .refs()
            .get(&collab::git::object_store::GitRefName::parse("refs/heads/main").expect("ref"))
            .expect("main")
            .as_str(),
        first_head
    );
    assert_eq!(state.receipts.lock().await.len(), 1);
    shutdown.send(()).expect("stop server");
}
