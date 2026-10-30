use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    process::Output,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use axum::{
    Router,
    body::Bytes,
    extract::{OriginalUri, Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    response::IntoResponse,
    routing::{any, get, post},
};
use base64::Engine as _;
use collab::{
    git::{
        object_store::{
            BackendObject, BackendWriteCondition, BackendWriteOutcome, EntityTag, GitObjectBackend,
            GitObjectBackendError, GitObjectStoreLimits,
        },
        repository_registry::{
            ExternalProviderCoordinate, HostedAuthority, HostedRepository,
            HostedRepositoryLifecycle, RepositoryCoordinate,
        },
        smart_http_read::{
            GitReadAuthorizationError, GitReadAuthorizer, GitRequestEncoding,
            GitSmartHttpReadError, GitSmartHttpReadLimits, GitSmartHttpReadService,
            GitSmartHttpResponse, GitUploadPackRequest,
        },
        smart_http_write::{
            GitForcePushPolicy, GitReceivePackRequest, GitSmartHttpWriteError,
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
use git_credential_nostr::{
    CredentialStore, CredentialStoreError, HelperConfig, HelperConfigError, StoredCredential,
};
use git_sign_nostr::{Clock, SignatureReadError, SignatureReader};
use nostr_compat::{
    PublicKey,
    buzz_nips::project_workflow::GitSignatureEnvelope,
    nip34_collaboration::{GitPatch, PatchPosition},
    nip34_repository::{GitObjectId, RepositoryCoordinate as Nip34Coordinate},
};
use secp256k1::{Keypair, Secp256k1, SecretKey};
use serde::Deserialize;
use tempfile::TempDir;
use tokio::{io::AsyncWriteExt as _, process::Command, sync::Mutex};
use uuid::Uuid;
use zeroize::Zeroizing;

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
    allow_read: AtomicBool,
    allow_write: AtomicBool,
}

#[async_trait]
impl GitReadAuthorizer for StaticAuthorizer {
    async fn authorize_read(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitReadAuthorizationError> {
        if !self.allow_read.load(Ordering::SeqCst)
            || authorization.action != AuthorizationAction::Read
            || authorization.required_scope.as_str() != "git:read"
            || authorization.resource.resource_id != self.repository.repository_id
        {
            return Err(GitReadAuthorizationError::Denied);
        }
        Ok(self.repository.clone())
    }
}

#[async_trait]
impl GitWriteAuthorizer for StaticAuthorizer {
    async fn authorize_write(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitWriteAuthorizationError> {
        if !self.allow_write.load(Ordering::SeqCst)
            || authorization.action != AuthorizationAction::Write
            || authorization.required_scope.as_str() != "git:write"
            || authorization.resource.resource_id != self.repository.repository_id
        {
            return Err(GitWriteAuthorizationError::Denied);
        }
        Ok(self.repository.clone())
    }
}

struct TestAuthorization {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    read_scope: AuthorizationScope,
    write_scope: AuthorizationScope,
    membership: CommunityMembership,
    repository_id: AggregateId,
}

impl TestAuthorization {
    fn read_request(&self) -> AuthorizationRequest<'_> {
        self.request(AuthorizationAction::Read, &self.read_scope)
    }

    fn write_request(&self) -> AuthorizationRequest<'_> {
        self.request(AuthorizationAction::Write, &self.write_scope)
    }

    fn request<'a>(
        &'a self,
        action: AuthorizationAction,
        scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: scope,
            action,
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

fn aggregate(value: u128) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(value))
}

fn repository(
    community_id: CommunityId,
    repository_id: AggregateId,
    authority: HostedAuthority,
) -> HostedRepository {
    HostedRepository {
        community_id,
        repository_id,
        coordinate: RepositoryCoordinate::new(NostrPublicKey::from_bytes([7; 32]), "repository")
            .expect("repository coordinate"),
        authority,
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

fn authorization(community_id: CommunityId, repository_id: AggregateId) -> TestAuthorization {
    let tenant = bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "git-conformance")
                .expect("tenant route"),
        ),
        &[],
    )
    .expect("tenant");
    let principal_id = PrincipalId::from_uuid(Uuid::from_u128(900));
    let read_scope = AuthorizationScope::new("git:read").expect("read scope");
    let write_scope = AuthorizationScope::new("git:write").expect("write scope");
    let principal = AuthenticatedPrincipal::zed_account(
        principal_id,
        community_id,
        ServiceAccountId::new(1),
        PrincipalScopes::new([read_scope.clone(), write_scope.clone()]).expect("principal scopes"),
    );
    TestAuthorization {
        tenant,
        principal,
        read_scope,
        write_scope,
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

struct ServerState {
    read_service: Arc<GitSmartHttpReadService>,
    write_service: Arc<GitSmartHttpWriteService>,
    authorization: TestAuthorization,
    legacy_root: std::path::PathBuf,
    operations: AtomicU64,
}

#[derive(Deserialize)]
struct InfoRefsQuery {
    service: String,
}

async fn new_info_refs(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<InfoRefsQuery>,
    headers: HeaderMap,
) -> axum::response::Response {
    let protocol = headers
        .get("git-protocol")
        .and_then(|value| value.to_str().ok());
    match query.service.as_str() {
        "git-upload-pack" => match state
            .read_service
            .advertise_refs(&state.authorization.read_request(), protocol)
            .await
        {
            Ok(response) => read_response(response),
            Err(error) => read_error(error),
        },
        "git-receive-pack" => match state
            .write_service
            .advertise_refs(&state.authorization.write_request(), protocol)
            .await
        {
            Ok(response) => write_response(response),
            Err(error) => write_error(error),
        },
        _ => StatusCode::BAD_REQUEST.into_response(),
    }
}

async fn new_upload_pack(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let request = GitUploadPackRequest {
        content_type: content_type(&headers),
        content_encoding: request_encoding(&headers),
        git_protocol: git_protocol(&headers),
        body: body.to_vec(),
    };
    match state
        .read_service
        .upload_pack(&state.authorization.read_request(), request)
        .await
    {
        Ok(response) => read_response(response),
        Err(error) => read_error(error),
    }
}

async fn new_receive_pack(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let sequence = state.operations.fetch_add(1, Ordering::SeqCst);
    let request = GitReceivePackRequest {
        operation_id: OperationId::from_uuid(Uuid::from_u128(u128::from(sequence) + 1)),
        content_type: content_type(&headers),
        content_encoding: request_encoding(&headers),
        git_protocol: git_protocol(&headers),
        body: body.to_vec(),
    };
    match state
        .write_service
        .receive_pack(&state.authorization.write_request(), request)
        .await
    {
        Ok(response) => write_response(response),
        Err(error) => write_error(error),
    }
}

fn content_type(headers: &HeaderMap) -> String {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

fn git_protocol(headers: &HeaderMap) -> Option<String> {
    headers
        .get("git-protocol")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn request_encoding(headers: &HeaderMap) -> GitRequestEncoding {
    match headers
        .get(header::CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
    {
        Some("gzip") | Some("x-gzip") => GitRequestEncoding::Gzip,
        _ => GitRequestEncoding::Identity,
    }
}

fn read_response(response: GitSmartHttpResponse) -> axum::response::Response {
    service_response(response.content_type, response.cache_control, response.body)
}

fn write_response(response: GitSmartHttpWriteResponse) -> axum::response::Response {
    service_response(response.content_type, response.cache_control, response.body)
}

fn service_response(
    content_type: &'static str,
    cache_control: &'static str,
    body: Vec<u8>,
) -> axum::response::Response {
    let mut response = body.into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(cache_control),
    );
    response
}

fn read_error(error: GitSmartHttpReadError) -> axum::response::Response {
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

fn write_error(error: GitSmartHttpWriteError) -> axum::response::Response {
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

async fn legacy_http_backend(
    State(state): State<Arc<ServerState>>,
    OriginalUri(uri): OriginalUri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let Some(path_info) = uri.path().strip_prefix("/legacy") else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut child = match Command::new("git")
        .arg("http-backend")
        .env("GIT_PROJECT_ROOT", &state.legacy_root)
        .env("GIT_HTTP_EXPORT_ALL", "1")
        .env("PATH_INFO", path_info)
        .env("REQUEST_METHOD", method.as_str())
        .env("QUERY_STRING", uri.query().unwrap_or_default())
        .env("CONTENT_TYPE", content_type(&headers))
        .env("CONTENT_LENGTH", body.len().to_string())
        .env(
            "HTTP_GIT_PROTOCOL",
            git_protocol(&headers).unwrap_or_default(),
        )
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let Some(mut stdin) = child.stdin.take() else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    if stdin.write_all(&body).await.is_err() {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    drop(stdin);
    let output = match child.wait_with_output().await {
        Ok(output) if output.status.success() => output.stdout,
        _ => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    cgi_response(&output).unwrap_or_else(|| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn cgi_response(output: &[u8]) -> Option<axum::response::Response> {
    let split = output
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4))
        .or_else(|| {
            output
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|index| (index, 2))
        })?;
    let headers = std::str::from_utf8(&output[..split.0]).ok()?;
    let mut status = StatusCode::OK;
    let mut response_headers = Vec::new();
    for line in headers.lines() {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("status") {
            status = value.trim().split(' ').next()?.parse().ok()?;
        } else {
            response_headers.push((
                HeaderName::from_bytes(name.trim().as_bytes()).ok()?,
                HeaderValue::from_str(value.trim()).ok()?,
            ));
        }
    }
    let mut response = (status, output[split.0 + split.1..].to_vec()).into_response();
    for (name, value) in response_headers {
        response.headers_mut().insert(name, value);
    }
    Some(response)
}

async fn start_server(
    legacy_root: &Path,
    repository: HostedRepository,
    backend: Arc<MemoryBackend>,
    scratch: &Path,
) -> (String, tokio::sync::oneshot::Sender<()>) {
    let authorizer = Arc::new(StaticAuthorizer {
        repository: repository.clone(),
        allow_read: AtomicBool::new(true),
        allow_write: AtomicBool::new(true),
    });
    let read_service = Arc::new(
        GitSmartHttpReadService::new(
            authorizer.clone(),
            backend.clone(),
            scratch,
            GitObjectStoreLimits::default(),
            GitSmartHttpReadLimits::default(),
        )
        .expect("read service"),
    );
    let write_service = Arc::new(
        GitSmartHttpWriteService::new(
            authorizer,
            backend,
            scratch,
            GitObjectStoreLimits::default(),
            GitSmartHttpWriteLimits::default(),
            GitForcePushPolicy::FastForwardOnly,
        )
        .expect("write service"),
    );
    let state = Arc::new(ServerState {
        read_service,
        write_service,
        authorization: authorization(repository.community_id, repository.repository_id),
        legacy_root: legacy_root.to_path_buf(),
        operations: AtomicU64::new(0),
    });
    let app = Router::new()
        .route("/new/repository.git/info/refs", get(new_info_refs))
        .route("/new/repository.git/git-upload-pack", post(new_upload_pack))
        .route(
            "/new/repository.git/git-receive-pack",
            post(new_receive_pack),
        )
        .route("/legacy/*path", any(legacy_http_backend))
        .with_state(state);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind server");
    listener.set_nonblocking(true).expect("nonblocking server");
    let address = listener.local_addr().expect("server address");
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        axum::Server::from_tcp(listener)
            .expect("HTTP server")
            .serve(app.into_make_service())
            .with_graceful_shutdown(async {
                if shutdown_receiver.await.is_err() {
                    return;
                }
            })
            .await
            .expect("serve Git conformance");
    });
    (format!("http://{address}"), shutdown_sender)
}

async fn run_git(directory: &Path, arguments: &[&str]) -> Output {
    Command::new("git")
        .args(arguments)
        .current_dir(directory)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .await
        .expect("run git")
}

async fn git_ok(directory: &Path, arguments: &[&str]) -> String {
    let output = run_git(directory, arguments).await;
    assert!(
        output.status.success(),
        "git {arguments:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

async fn commit(source: &Path, contents: &str, message: &str) -> String {
    tokio::fs::write(source.join("file.txt"), contents)
        .await
        .expect("write fixture");
    git_ok(source, &["add", "file.txt"]).await;
    git_ok(source, &["commit", "--quiet", "-m", message]).await;
    git_ok(source, &["rev-parse", "HEAD"]).await
}

#[tokio::test]
async fn git_conformance_legacy_and_consolidated_servers_clone_and_push() {
    let source = TempDir::new().expect("source");
    git_ok(source.path(), &["init", "--quiet", "--initial-branch=main"]).await;
    git_ok(source.path(), &["config", "user.email", "git@test.invalid"]).await;
    git_ok(source.path(), &["config", "user.name", "Git Conformance"]).await;
    let first_head = commit(source.path(), "first\n", "first").await;

    let legacy_root = TempDir::new().expect("legacy root");
    let legacy_repository = legacy_root.path().join("repository.git");
    git_ok(
        legacy_root.path(),
        &[
            "init",
            "--bare",
            "--quiet",
            legacy_repository.to_str().expect("legacy path"),
        ],
    )
    .await;
    git_ok(&legacy_repository, &["config", "http.receivepack", "true"]).await;

    let community_id = community(1);
    let repository_id = aggregate(2);
    let hosted = repository(
        community_id,
        repository_id,
        HostedAuthority::SimHostedNip34 {
            storage_handle_id: Uuid::from_u128(3),
        },
    );
    let backend = Arc::new(MemoryBackend::default());
    let scratch = TempDir::new().expect("scratch");
    let (server, shutdown) =
        start_server(legacy_root.path(), hosted, backend, scratch.path()).await;
    let legacy_url = format!("{server}/legacy/repository.git");
    let new_url = format!("{server}/new/repository.git");
    git_ok(source.path(), &["remote", "add", "legacy", &legacy_url]).await;
    git_ok(source.path(), &["remote", "add", "new", &new_url]).await;
    git_ok(source.path(), &["push", "--quiet", "legacy", "main"]).await;
    git_ok(source.path(), &["push", "--quiet", "new", "main"]).await;

    let second_head = commit(source.path(), "second\n", "second").await;
    git_ok(source.path(), &["push", "--quiet", "legacy", "main"]).await;
    git_ok(source.path(), &["push", "--quiet", "new", "main"]).await;
    assert_ne!(first_head, second_head);

    let clones = TempDir::new().expect("clones");
    let legacy_clone = clones.path().join("legacy");
    let new_clone = clones.path().join("new");
    git_ok(
        clones.path(),
        &[
            "clone",
            "--quiet",
            &legacy_url,
            legacy_clone.to_str().expect("legacy clone path"),
        ],
    )
    .await;
    git_ok(
        clones.path(),
        &[
            "clone",
            "--quiet",
            &new_url,
            new_clone.to_str().expect("new clone path"),
        ],
    )
    .await;
    assert_eq!(
        git_ok(&legacy_clone, &["rev-parse", "HEAD"]).await,
        second_head
    );
    assert_eq!(
        git_ok(&new_clone, &["rev-parse", "HEAD"]).await,
        second_head
    );
    assert_eq!(
        tokio::fs::read(legacy_clone.join("file.txt"))
            .await
            .expect("legacy clone contents"),
        tokio::fs::read(new_clone.join("file.txt"))
            .await
            .expect("new clone contents")
    );
    shutdown.send(()).expect("stop server");
}

#[derive(Default)]
struct MemoryConfig(BTreeMap<String, Vec<String>>);

impl HelperConfig for MemoryConfig {
    fn values(&self, key: &str) -> Result<Vec<String>, HelperConfigError> {
        Ok(self.0.get(key).cloned().unwrap_or_default())
    }
}

struct FixtureStore {
    username: String,
    secret: Vec<u8>,
    reads: AtomicUsize,
}

impl CredentialStore for FixtureStore {
    fn read(
        &self,
        _credential_identifier: &str,
    ) -> Result<Option<StoredCredential>, CredentialStoreError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(Some(StoredCredential::new(
            self.username.clone(),
            Zeroizing::new(self.secret.clone()),
        )))
    }
}

#[derive(Default)]
struct FixtureSignatures(StdMutex<BTreeMap<String, String>>);

impl FixtureSignatures {
    fn insert(&self, path: &str, signature: String) {
        self.0
            .lock()
            .expect("signature fixture lock")
            .insert(path.into(), signature);
    }
}

impl SignatureReader for FixtureSignatures {
    fn read(&self, path: &str) -> Result<String, SignatureReadError> {
        self.0
            .lock()
            .expect("signature fixture lock")
            .get(path)
            .cloned()
            .ok_or(SignatureReadError::Missing)
    }
}

struct FixedClock(u32);

impl Clock for FixedClock {
    fn unix_timestamp(&self) -> Result<u32, ()> {
        Ok(self.0)
    }
}

#[tokio::test]
async fn git_conformance_patch_authentication_and_signing_round_trip() {
    let secret = [3; 32];
    let secret_key = SecretKey::from_slice(&secret).expect("secret");
    let keypair = Keypair::from_secret_key(&Secp256k1::signing_only(), &secret_key);
    let public_key = keypair.x_only_public_key().0.to_string();
    let identifier = format!("zed-nostr://credential/v1/community/account/profile/{public_key}");
    let config = MemoryConfig(BTreeMap::from([
        ("nostr.allowedHost".into(), vec!["git.example.test".into()]),
        ("nostr.credentialIdentifier".into(), vec![identifier]),
        ("user.signingkey".into(), vec![public_key.clone()]),
    ]));
    let store = FixtureStore {
        username: public_key.clone(),
        secret: secret.to_vec(),
        reads: AtomicUsize::new(0),
    };
    let credential_request = b"capability[]=authtype\nprotocol=https\nhost=git.example.test\npath=git/owner/repository/info/refs\nwwwauth[]=Nostr method=\"GET\", realm=\"zed\"\n\n";
    let mut credential_stdout = Vec::new();
    assert_eq!(
        git_credential_nostr::run_with(
            Some("get"),
            credential_request,
            &config,
            &store,
            &mut credential_stdout,
            &mut Vec::new(),
        ),
        git_credential_nostr::EXIT_SUCCESS
    );
    let credential = std::str::from_utf8(&credential_stdout)
        .expect("credential output")
        .lines()
        .find_map(|line| line.strip_prefix("credential="))
        .expect("credential field");
    let event: serde_json::Value = serde_json::from_slice(
        &base64::engine::general_purpose::STANDARD
            .decode(credential)
            .expect("credential base64"),
    )
    .expect("credential event");
    assert_eq!(event["kind"], 27_235);
    assert_eq!(event["pubkey"], public_key);

    let author = PublicKey::from_hex(&public_key).expect("author");
    let patch = GitPatch {
        repository: Nip34Coordinate {
            owner: author,
            identifier: "repository".into(),
            relay_hint: Some("wss://relay.example.test".into()),
        },
        earliest_unique_commit: Some(
            GitObjectId::from_hex("11e84ad4a8ad6fb1129a1cc4a891c76b60fadd27")
                .expect("earliest commit"),
        ),
        recipients: vec![author],
        reply_to: None,
        position: PatchPosition::Root,
        commit: Some(
            GitObjectId::from_hex("22e84ad4a8ad6fb1129a1cc4a891c76b60fadd27").expect("commit"),
        ),
        parent_commit: Some(
            GitObjectId::from_hex("33e84ad4a8ad6fb1129a1cc4a891c76b60fadd27").expect("parent"),
        ),
        commit_pgp_signature: None,
        committer: None,
        content: "diff --git a/file.txt b/file.txt\n+conformance\n".into(),
        extra_tags: Vec::new(),
    };
    let patch_event = patch.to_event(author, 1_700_000_000).expect("patch event");
    assert_eq!(
        GitPatch::parse_event(&patch_event).expect("patch round trip"),
        patch
    );

    let payload = b"tree 0123456789abcdef\n\nconformance";
    let sign_arguments = vec!["--status-fd=2".into(), "-bsau".into(), public_key];
    let mut signature = Vec::new();
    let mut sign_status = Vec::new();
    assert_eq!(
        git_sign_nostr::run_with(
            &sign_arguments,
            payload,
            &config,
            &store,
            &FixtureSignatures::default(),
            &FixedClock(1_700_000_000),
            &mut signature,
            &mut sign_status,
            &mut Vec::new(),
        ),
        git_sign_nostr::EXIT_SUCCESS
    );
    let armored = String::from_utf8(signature).expect("signature armor");
    assert!(
        GitSignatureEnvelope::parse_armored(&armored)
            .expect("NIP-GS envelope")
            .verify(payload)
            .is_ok()
    );
    let signatures = FixtureSignatures::default();
    signatures.insert("signature.asc", armored);
    let verify_store = FixtureStore {
        username: String::new(),
        secret: Vec::new(),
        reads: AtomicUsize::new(0),
    };
    let verify_arguments = vec![
        "--status-fd=1".into(),
        "--verify".into(),
        "signature.asc".into(),
        "-".into(),
    ];
    let mut verify_status = Vec::new();
    assert_eq!(
        git_sign_nostr::run_with(
            &verify_arguments,
            payload,
            &config,
            &verify_store,
            &signatures,
            &FixedClock(0),
            &mut Vec::new(),
            &mut verify_status,
            &mut Vec::new(),
        ),
        git_sign_nostr::EXIT_SUCCESS
    );
    assert_eq!(verify_store.reads.load(Ordering::SeqCst), 0);
    assert!(
        std::str::from_utf8(&verify_status)
            .expect("verify status")
            .contains("[GNUPG:] TRUST_FULLY 0 shell")
    );
}

#[tokio::test]
async fn git_conformance_permissions_and_external_provider_fail_before_storage() {
    let community_id = community(20);
    let repository_id = aggregate(21);
    let scratch = TempDir::new().expect("scratch");
    let backend = Arc::new(MemoryBackend::default());
    let hosted = repository(
        community_id,
        repository_id,
        HostedAuthority::SimHostedNip34 {
            storage_handle_id: Uuid::from_u128(22),
        },
    );
    let denied = Arc::new(StaticAuthorizer {
        repository: hosted,
        allow_read: AtomicBool::new(false),
        allow_write: AtomicBool::new(false),
    });
    let read_service = GitSmartHttpReadService::new(
        denied.clone(),
        backend.clone(),
        scratch.path(),
        GitObjectStoreLimits::default(),
        GitSmartHttpReadLimits::default(),
    )
    .expect("read service");
    let write_service = GitSmartHttpWriteService::new(
        denied,
        backend.clone(),
        scratch.path(),
        GitObjectStoreLimits::default(),
        GitSmartHttpWriteLimits::default(),
        GitForcePushPolicy::FastForwardOnly,
    )
    .expect("write service");
    let access = authorization(community_id, repository_id);
    assert!(matches!(
        read_service
            .upload_pack(
                &access.read_request(),
                GitUploadPackRequest {
                    content_type: "text/plain".into(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: vec![0; 32],
                },
            )
            .await,
        Err(GitSmartHttpReadError::RepositoryNotFound)
    ));
    assert!(matches!(
        write_service
            .receive_pack(
                &access.write_request(),
                GitReceivePackRequest {
                    operation_id: OperationId::from_uuid(Uuid::nil()),
                    content_type: "text/plain".into(),
                    content_encoding: GitRequestEncoding::Identity,
                    git_protocol: None,
                    body: vec![0; 32],
                },
            )
            .await,
        Err(GitSmartHttpWriteError::RepositoryNotFound)
    ));
    assert_eq!(backend.reads.load(Ordering::SeqCst), 0);

    let external = repository(
        community_id,
        repository_id,
        HostedAuthority::ExternalProvider(
            ExternalProviderCoordinate::new(
                "github",
                "https://github.example.test",
                "owner",
                "repository",
            )
            .expect("external coordinate"),
        ),
    );
    let external_authorizer = Arc::new(StaticAuthorizer {
        repository: external,
        allow_read: AtomicBool::new(true),
        allow_write: AtomicBool::new(true),
    });
    let external_read = GitSmartHttpReadService::new(
        external_authorizer.clone(),
        backend.clone(),
        scratch.path(),
        GitObjectStoreLimits::default(),
        GitSmartHttpReadLimits::default(),
    )
    .expect("external read service");
    let external_write = GitSmartHttpWriteService::new(
        external_authorizer,
        backend.clone(),
        scratch.path(),
        GitObjectStoreLimits::default(),
        GitSmartHttpWriteLimits::default(),
        GitForcePushPolicy::FastForwardOnly,
    )
    .expect("external write service");
    assert!(matches!(
        external_read
            .advertise_refs(&access.read_request(), None)
            .await,
        Err(GitSmartHttpReadError::RepositoryNotFound)
    ));
    assert!(matches!(
        external_write
            .advertise_refs(&access.write_request(), None)
            .await,
        Err(GitSmartHttpWriteError::RepositoryNotFound)
    ));
    assert_eq!(backend.reads.load(Ordering::SeqCst), 0);
}
