use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use collaboration_domain::{AuthorizationRequest, OperationId};
use flate2::read::GzDecoder;
use tempfile::TempDir;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::{OwnedSemaphorePermit, Semaphore},
};

use super::{
    object_store::{
        GitContentDigest, GitObjectBackend, GitObjectStore, GitObjectStoreError,
        GitObjectStoreLimits, GitRefManifest, GitRefName, GitRefSnapshot,
    },
    repository_registry::{
        HostedRepository, HostedRepositoryRegistry, HostedRepositoryRegistryError,
        RepositoryPermission,
    },
    smart_http_read::GitRequestEncoding,
};

const ADVERTISEMENT_CONTENT_TYPE: &str = "application/x-git-receive-pack-advertisement";
const REQUEST_CONTENT_TYPE: &str = "application/x-git-receive-pack-request";
const RESULT_CONTENT_TYPE: &str = "application/x-git-receive-pack-result";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitForcePushPolicy {
    FastForwardOnly,
    AllowForce,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GitSmartHttpWriteLimits {
    pub max_request_bytes: u64,
    pub max_decoded_request_bytes: u64,
    pub max_advertisement_bytes: u64,
    pub max_response_bytes: u64,
    pub max_repository_bytes: u64,
    pub max_concurrent_processes: usize,
    pub process_timeout: Duration,
}

impl GitSmartHttpWriteLimits {
    pub fn validate(self) -> Result<Self, GitSmartHttpWriteError> {
        if self.max_request_bytes == 0
            || self.max_decoded_request_bytes == 0
            || self.max_advertisement_bytes == 0
            || self.max_response_bytes == 0
            || self.max_repository_bytes == 0
            || self.max_concurrent_processes == 0
            || self.process_timeout.is_zero()
            || usize::try_from(self.max_request_bytes).is_err()
            || usize::try_from(self.max_decoded_request_bytes).is_err()
            || usize::try_from(self.max_advertisement_bytes).is_err()
            || usize::try_from(self.max_response_bytes).is_err()
            || usize::try_from(self.max_repository_bytes).is_err()
        {
            return Err(GitSmartHttpWriteError::InvalidConfiguration);
        }
        Ok(self)
    }
}

impl Default for GitSmartHttpWriteLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 512 * 1024 * 1024,
            max_decoded_request_bytes: 512 * 1024 * 1024,
            max_advertisement_bytes: 4 * 1024 * 1024,
            max_response_bytes: 1024 * 1024,
            max_repository_bytes: 4 * 1024 * 1024 * 1024,
            max_concurrent_processes: 10,
            process_timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitReceivePackRequest {
    pub operation_id: OperationId,
    pub content_type: String,
    pub content_encoding: GitRequestEncoding,
    pub git_protocol: Option<String>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitPushReceipt {
    pub operation_id: OperationId,
    pub parent_manifest: Option<GitContentDigest>,
    pub published_manifest: Option<GitContentDigest>,
    pub applied: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitSmartHttpWriteResponse {
    pub content_type: &'static str,
    pub cache_control: &'static str,
    pub body: Vec<u8>,
    pub receipt: Option<GitPushReceipt>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GitWriteAuthorizationError {
    #[error("repository is unavailable")]
    Denied,
    #[error("repository authorization is unavailable")]
    Unavailable,
}

#[async_trait]
pub trait GitWriteAuthorizer: Send + Sync {
    async fn authorize_write(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitWriteAuthorizationError>;
}

#[async_trait]
impl GitWriteAuthorizer for HostedRepositoryRegistry {
    async fn authorize_write(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitWriteAuthorizationError> {
        self.authorize_access(authorization, RepositoryPermission::Write)
            .await
            .map_err(|error| match error {
                HostedRepositoryRegistryError::Unavailable(_) => {
                    GitWriteAuthorizationError::Unavailable
                }
                _ => GitWriteAuthorizationError::Denied,
            })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GitSmartHttpWriteError {
    #[error("Git smart-HTTP write configuration is invalid")]
    InvalidConfiguration,
    #[error("repository is unavailable")]
    RepositoryNotFound,
    #[error("Git smart-HTTP request is invalid")]
    InvalidRequest,
    #[error("Git smart-HTTP request, response or repository exceeds its configured limit")]
    PayloadTooLarge,
    #[error("Git smart-HTTP service is busy")]
    Busy,
    #[error("Git smart-HTTP operation timed out")]
    Timeout,
    #[error("Git push references an unavailable object")]
    MissingObject,
    #[error("Git repository changed concurrently")]
    ConcurrentUpdate,
    #[error("Git smart-HTTP backend is unavailable")]
    Unavailable,
}

pub struct GitSmartHttpWriteService {
    authorizer: Arc<dyn GitWriteAuthorizer>,
    backend: Arc<dyn GitObjectBackend>,
    scratch_directory: PathBuf,
    object_store_limits: GitObjectStoreLimits,
    limits: GitSmartHttpWriteLimits,
    force_policy: GitForcePushPolicy,
    process_permits: Arc<Semaphore>,
}

impl GitSmartHttpWriteService {
    pub fn new(
        authorizer: Arc<dyn GitWriteAuthorizer>,
        backend: Arc<dyn GitObjectBackend>,
        scratch_directory: impl Into<PathBuf>,
        object_store_limits: GitObjectStoreLimits,
        limits: GitSmartHttpWriteLimits,
        force_policy: GitForcePushPolicy,
    ) -> Result<Self, GitSmartHttpWriteError> {
        let scratch_directory = scratch_directory.into();
        let metadata = std::fs::metadata(&scratch_directory)
            .map_err(|_| GitSmartHttpWriteError::InvalidConfiguration)?;
        if !scratch_directory.is_absolute() || !metadata.is_dir() {
            return Err(GitSmartHttpWriteError::InvalidConfiguration);
        }
        let limits = limits.validate()?;
        let object_store_limits = object_store_limits
            .validate()
            .map_err(|_| GitSmartHttpWriteError::InvalidConfiguration)?;
        Ok(Self {
            authorizer,
            backend,
            scratch_directory,
            object_store_limits,
            limits,
            force_policy,
            process_permits: Arc::new(Semaphore::new(limits.max_concurrent_processes)),
        })
    }

    pub async fn advertise_refs(
        &self,
        authorization: &AuthorizationRequest<'_>,
        git_protocol: Option<&str>,
    ) -> Result<GitSmartHttpWriteResponse, GitSmartHttpWriteError> {
        let repository = self.authorize(authorization).await?;
        let git_protocol = validate_git_protocol(git_protocol)?;
        let permit = self.acquire_permit()?;
        let hydrated = self.hydrate(&repository).await?;
        let output = self
            .run_receive_pack(
                &hydrated,
                &[],
                true,
                git_protocol.as_deref(),
                self.limits.max_advertisement_bytes,
                permit,
            )
            .await?;
        if !output.status.success() {
            return Err(GitSmartHttpWriteError::Unavailable);
        }
        let body = prepend_service_line(
            "git-receive-pack",
            output.bytes,
            self.limits.max_advertisement_bytes,
        )?;
        Ok(GitSmartHttpWriteResponse {
            content_type: ADVERTISEMENT_CONTENT_TYPE,
            cache_control: "no-cache",
            body,
            receipt: None,
        })
    }

    pub async fn receive_pack(
        &self,
        authorization: &AuthorizationRequest<'_>,
        request: GitReceivePackRequest,
    ) -> Result<GitSmartHttpWriteResponse, GitSmartHttpWriteError> {
        let repository = self.authorize(authorization).await?;
        if request.operation_id.as_uuid().is_nil() || request.content_type != REQUEST_CONTENT_TYPE {
            return Err(GitSmartHttpWriteError::InvalidRequest);
        }
        let git_protocol = validate_git_protocol(request.git_protocol.as_deref())?;
        let operation_id = request.operation_id;
        let body = decode_request_body(request, self.limits)?;
        let permit = self.acquire_permit()?;
        let hydrated = self.hydrate(&repository).await?;
        let parent_manifest = hydrated
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.manifest_digest().clone());
        let refs_before = hydrated
            .snapshot
            .as_ref()
            .map_or_else(BTreeMap::new, |snapshot| snapshot.manifest().refs().clone());
        let output = self
            .run_receive_pack(
                &hydrated,
                &body,
                false,
                git_protocol.as_deref(),
                self.limits.max_response_bytes,
                permit,
            )
            .await?;
        let refs_after = snapshot_refs(
            hydrated.path(),
            self.object_store_limits.max_refs,
            self.limits.max_advertisement_bytes,
            self.limits.process_timeout,
        )
        .await?;
        if !output.status.success() || refs_after == refs_before {
            return Ok(GitSmartHttpWriteResponse {
                content_type: RESULT_CONTENT_TYPE,
                cache_control: "no-cache",
                body: output.bytes,
                receipt: Some(GitPushReceipt {
                    operation_id,
                    parent_manifest,
                    published_manifest: None,
                    applied: false,
                }),
            });
        }
        run_git_status(
            hydrated.path(),
            &["fsck", "--connectivity-only", "--no-dangling"],
            self.limits.process_timeout,
            GitSmartHttpWriteError::MissingObject,
        )
        .await?;
        let pack = capture_reachable_pack(
            hydrated.path(),
            &refs_after,
            self.object_store_limits
                .max_object_bytes
                .min(self.limits.max_repository_bytes),
            self.limits.process_timeout,
        )
        .await?;
        let mut objects = BTreeSet::new();
        if !refs_after.is_empty() {
            let pack = pack.ok_or(GitSmartHttpWriteError::MissingObject)?;
            objects.insert(
                hydrated
                    .object_store
                    .put_object(pack)
                    .await
                    .map_err(map_object_store_write_error)?,
            );
        }
        let head = choose_head(
            hydrated
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.manifest().head()),
            &refs_after,
        );
        let manifest = GitRefManifest::new(head, refs_after, objects, parent_manifest.clone());
        let published = hydrated
            .object_store
            .compare_and_swap_refs(hydrated.snapshot.as_ref(), manifest)
            .await
            .map_err(map_object_store_publish_error)?;
        Ok(GitSmartHttpWriteResponse {
            content_type: RESULT_CONTENT_TYPE,
            cache_control: "no-cache",
            body: output.bytes,
            receipt: Some(GitPushReceipt {
                operation_id,
                parent_manifest,
                published_manifest: Some(published.manifest_digest().clone()),
                applied: true,
            }),
        })
    }

    async fn authorize(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitSmartHttpWriteError> {
        self.authorizer
            .authorize_write(authorization)
            .await
            .map_err(|error| match error {
                GitWriteAuthorizationError::Denied => GitSmartHttpWriteError::RepositoryNotFound,
                GitWriteAuthorizationError::Unavailable => GitSmartHttpWriteError::Unavailable,
            })
    }

    fn acquire_permit(&self) -> Result<OwnedSemaphorePermit, GitSmartHttpWriteError> {
        self.process_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| GitSmartHttpWriteError::Busy)
    }

    async fn hydrate(
        &self,
        repository: &HostedRepository,
    ) -> Result<HydratedWriteRepository, GitSmartHttpWriteError> {
        let object_store = GitObjectStore::for_authorized_repository(
            self.backend.clone(),
            repository,
            self.object_store_limits,
        )
        .map_err(map_object_store_open_error)?;
        let snapshot = match object_store.read_refs().await {
            Ok(snapshot) => Some(snapshot),
            Err(GitObjectStoreError::RefsNotFound) => None,
            Err(error) => return Err(map_object_store_hydrate_error(error)),
        };
        let temporary_directory = TempDir::new_in(&self.scratch_directory)
            .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
        let repository_path = temporary_directory.path().to_path_buf();
        run_git_status(
            &repository_path,
            &["init", "--bare", "--quiet"],
            self.limits.process_timeout,
            GitSmartHttpWriteError::Unavailable,
        )
        .await?;
        if let Some(snapshot) = &snapshot {
            let pack_directory = repository_path.join("objects").join("pack");
            tokio::fs::create_dir_all(&pack_directory)
                .await
                .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
            let mut repository_bytes = 0u64;
            for digest in snapshot.manifest().objects() {
                let bytes = object_store
                    .get_object(digest)
                    .await
                    .map_err(map_object_store_hydrate_error)?;
                let object_bytes = u64::try_from(bytes.len())
                    .map_err(|_| GitSmartHttpWriteError::PayloadTooLarge)?;
                repository_bytes = repository_bytes
                    .checked_add(object_bytes)
                    .ok_or(GitSmartHttpWriteError::PayloadTooLarge)?;
                if repository_bytes > self.limits.max_repository_bytes {
                    return Err(GitSmartHttpWriteError::PayloadTooLarge);
                }
                let pack_path = pack_directory.join(format!("pack-{}.pack", digest.as_str()));
                tokio::fs::write(&pack_path, bytes)
                    .await
                    .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
                let pack_path = pack_path
                    .to_str()
                    .ok_or(GitSmartHttpWriteError::Unavailable)?;
                run_git_status(
                    &repository_path,
                    &["index-pack", pack_path],
                    self.limits.process_timeout,
                    GitSmartHttpWriteError::MissingObject,
                )
                .await?;
            }
            install_refs(&repository_path, snapshot.manifest()).await?;
            if !snapshot.manifest().refs().is_empty() {
                run_git_status(
                    &repository_path,
                    &["fsck", "--connectivity-only", "--no-dangling"],
                    self.limits.process_timeout,
                    GitSmartHttpWriteError::MissingObject,
                )
                .await?;
            }
        }
        Ok(HydratedWriteRepository {
            _temporary_directory: temporary_directory,
            path: repository_path,
            object_store,
            snapshot,
        })
    }

    async fn run_receive_pack(
        &self,
        repository: &HydratedWriteRepository,
        request_body: &[u8],
        advertise_refs: bool,
        git_protocol: Option<&str>,
        max_output_bytes: u64,
        _permit: OwnedSemaphorePermit,
    ) -> Result<ProcessOutput, GitSmartHttpWriteError> {
        let mut command = Command::new("git");
        command
            .arg("-c")
            .arg(match self.force_policy {
                GitForcePushPolicy::FastForwardOnly => "receive.denyNonFastForwards=true",
                GitForcePushPolicy::AllowForce => "receive.denyNonFastForwards=false",
            })
            .arg("-c")
            .arg("receive.denyDeleteCurrent=ignore")
            .arg("receive-pack")
            .arg("--stateless-rpc");
        if advertise_refs {
            command.arg("--advertise-refs");
        }
        command
            .arg(repository.path())
            .current_dir(repository.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        harden_git_environment(&mut command, repository.path(), git_protocol);
        run_child_with_input(
            command,
            request_body,
            max_output_bytes,
            self.limits.process_timeout,
        )
        .await
    }
}

struct HydratedWriteRepository {
    _temporary_directory: TempDir,
    path: PathBuf,
    object_store: GitObjectStore,
    snapshot: Option<GitRefSnapshot>,
}

impl HydratedWriteRepository {
    fn path(&self) -> &Path {
        &self.path
    }
}

struct ProcessOutput {
    bytes: Vec<u8>,
    status: ExitStatus,
}

struct BoundedOutput {
    bytes: Vec<u8>,
    exceeded_limit: bool,
}

async fn run_child_with_input(
    mut command: Command,
    input: &[u8],
    max_output_bytes: u64,
    timeout: Duration,
) -> Result<ProcessOutput, GitSmartHttpWriteError> {
    let mut child = command
        .spawn()
        .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or(GitSmartHttpWriteError::Unavailable)?;
    let input = input.to_vec();
    let stdin_task = tokio::spawn(async move {
        stdin.write_all(&input).await?;
        stdin.shutdown().await
    });
    let stdout = child
        .stdout
        .take()
        .ok_or(GitSmartHttpWriteError::Unavailable)?;
    let stdout_task = tokio::spawn(read_bounded_and_drain(stdout, max_output_bytes));
    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(_)) => {
            stdin_task.abort();
            stdout_task.abort();
            return Err(GitSmartHttpWriteError::Unavailable);
        }
        Err(_) => {
            stdin_task.abort();
            stdout_task.abort();
            child
                .kill()
                .await
                .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
            return Err(GitSmartHttpWriteError::Timeout);
        }
    };
    let stdin_result = stdin_task
        .await
        .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
    let output = stdout_task
        .await
        .map_err(|_| GitSmartHttpWriteError::Unavailable)??;
    if output.exceeded_limit {
        return Err(GitSmartHttpWriteError::PayloadTooLarge);
    }
    if stdin_result.is_err() && status.success() {
        return Err(GitSmartHttpWriteError::InvalidRequest);
    }
    Ok(ProcessOutput {
        bytes: output.bytes,
        status,
    })
}

async fn read_bounded_and_drain(
    mut reader: impl AsyncRead + Unpin,
    max_bytes: u64,
) -> Result<BoundedOutput, GitSmartHttpWriteError> {
    let capacity =
        usize::try_from(max_bytes).map_err(|_| GitSmartHttpWriteError::PayloadTooLarge)?;
    let mut bytes = Vec::with_capacity(capacity.min(64 * 1024));
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    let mut exceeded_limit = false;
    loop {
        let read = reader
            .read(&mut buffer)
            .await
            .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(read).map_err(|_| GitSmartHttpWriteError::PayloadTooLarge)?)
            .ok_or(GitSmartHttpWriteError::PayloadTooLarge)?;
        if total <= max_bytes {
            bytes.extend_from_slice(&buffer[..read]);
        } else {
            exceeded_limit = true;
        }
    }
    Ok(BoundedOutput {
        bytes,
        exceeded_limit,
    })
}

async fn snapshot_refs(
    repository_path: &Path,
    max_refs: usize,
    max_output_bytes: u64,
    timeout: Duration,
) -> Result<BTreeMap<GitRefName, super::object_store::GitObjectId>, GitSmartHttpWriteError> {
    let mut command = Command::new("git");
    command
        .args(["for-each-ref", "--format=%(refname)%00%(objectname)"])
        .current_dir(repository_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    harden_git_environment(&mut command, repository_path, None);
    let output = run_child_with_input(command, &[], max_output_bytes, timeout).await?;
    if !output.status.success() {
        return Err(GitSmartHttpWriteError::Unavailable);
    }
    let text =
        std::str::from_utf8(&output.bytes).map_err(|_| GitSmartHttpWriteError::Unavailable)?;
    let mut refs = BTreeMap::new();
    for line in text.lines() {
        let (name, object_id) = line
            .split_once('\0')
            .ok_or(GitSmartHttpWriteError::Unavailable)?;
        let name = GitRefName::parse(name).map_err(map_object_store_write_error)?;
        let object_id = super::object_store::GitObjectId::parse(object_id)
            .map_err(map_object_store_write_error)?;
        if refs.insert(name, object_id).is_some() || refs.len() > max_refs {
            return Err(GitSmartHttpWriteError::PayloadTooLarge);
        }
    }
    Ok(refs)
}

async fn capture_reachable_pack(
    repository_path: &Path,
    refs: &BTreeMap<GitRefName, super::object_store::GitObjectId>,
    max_pack_bytes: u64,
    timeout: Duration,
) -> Result<Option<Vec<u8>>, GitSmartHttpWriteError> {
    if refs.is_empty() {
        return Ok(None);
    }
    let mut input = Vec::new();
    for object_id in refs.values() {
        input.extend_from_slice(object_id.as_str().as_bytes());
        input.push(b'\n');
    }
    let mut command = Command::new("git");
    command
        .args(["pack-objects", "--stdout", "--revs"])
        .current_dir(repository_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    harden_git_environment(&mut command, repository_path, None);
    let output = run_child_with_input(command, &input, max_pack_bytes, timeout).await?;
    if !output.status.success() || !output.bytes.starts_with(b"PACK") {
        return Err(GitSmartHttpWriteError::MissingObject);
    }
    Ok(Some(output.bytes))
}

async fn install_refs(
    repository_path: &Path,
    manifest: &GitRefManifest,
) -> Result<(), GitSmartHttpWriteError> {
    for (ref_name, object_id) in manifest.refs() {
        let ref_path = repository_path.join(ref_name.as_str());
        let parent = ref_path
            .parent()
            .ok_or(GitSmartHttpWriteError::Unavailable)?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
        tokio::fs::write(ref_path, format!("{}\n", object_id.as_str()))
            .await
            .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
    }
    if let Some(head) = manifest.head() {
        tokio::fs::write(
            repository_path.join("HEAD"),
            format!("ref: {}\n", head.as_str()),
        )
        .await
        .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
    }
    Ok(())
}

async fn run_git_status(
    repository_path: &Path,
    arguments: &[&str],
    timeout: Duration,
    failure: GitSmartHttpWriteError,
) -> Result<(), GitSmartHttpWriteError> {
    let mut command = Command::new("git");
    command
        .args(arguments)
        .current_dir(repository_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    harden_git_environment(&mut command, repository_path, None);
    let status = tokio::time::timeout(timeout, command.status())
        .await
        .map_err(|_| GitSmartHttpWriteError::Timeout)?
        .map_err(|_| GitSmartHttpWriteError::Unavailable)?;
    if !status.success() {
        return Err(failure);
    }
    Ok(())
}

fn choose_head(
    parent_head: Option<&GitRefName>,
    refs: &BTreeMap<GitRefName, super::object_store::GitObjectId>,
) -> Option<GitRefName> {
    parent_head
        .filter(|head| refs.contains_key(*head))
        .cloned()
        .or_else(|| {
            ["refs/heads/main", "refs/heads/master"]
                .into_iter()
                .find_map(|name| {
                    refs.keys()
                        .find(|ref_name| ref_name.as_str() == name)
                        .cloned()
                })
        })
        .or_else(|| {
            refs.keys()
                .find(|ref_name| ref_name.as_str().starts_with("refs/heads/"))
                .cloned()
        })
        .or_else(|| refs.keys().next().cloned())
}

fn prepend_service_line(
    service: &str,
    output: Vec<u8>,
    max_bytes: u64,
) -> Result<Vec<u8>, GitSmartHttpWriteError> {
    let service_line = format!("# service={service}\n");
    let packet_length = service_line
        .len()
        .checked_add(4)
        .ok_or(GitSmartHttpWriteError::PayloadTooLarge)?;
    let prefix = format!("{packet_length:04x}");
    let total_length = prefix
        .len()
        .checked_add(service_line.len())
        .and_then(|length| length.checked_add(4))
        .and_then(|length| length.checked_add(output.len()))
        .ok_or(GitSmartHttpWriteError::PayloadTooLarge)?;
    if u64::try_from(total_length).map_or(true, |length| length > max_bytes) {
        return Err(GitSmartHttpWriteError::PayloadTooLarge);
    }
    let mut body = Vec::with_capacity(total_length);
    body.extend_from_slice(prefix.as_bytes());
    body.extend_from_slice(service_line.as_bytes());
    body.extend_from_slice(b"0000");
    body.extend_from_slice(&output);
    Ok(body)
}

fn harden_git_environment(command: &mut Command, home: &Path, git_protocol: Option<&str>) {
    command.env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        command.env("PATH", path);
    }
    command
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_HTTP_EXPORT_ALL", "1")
        .env("HOME", home);
    if let Some(git_protocol) = git_protocol {
        command.env("GIT_PROTOCOL", git_protocol);
    }
}

fn validate_git_protocol(value: Option<&str>) -> Result<Option<String>, GitSmartHttpWriteError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty()
        || value.len() > 256
        || !value
            .chars()
            .all(|character| character.is_ascii_graphic() && character != '\0')
    {
        return Err(GitSmartHttpWriteError::InvalidRequest);
    }
    Ok(Some(value.to_owned()))
}

fn decode_request_body(
    request: GitReceivePackRequest,
    limits: GitSmartHttpWriteLimits,
) -> Result<Vec<u8>, GitSmartHttpWriteError> {
    if u64::try_from(request.body.len()).map_or(true, |length| length > limits.max_request_bytes) {
        return Err(GitSmartHttpWriteError::PayloadTooLarge);
    }
    match request.content_encoding {
        GitRequestEncoding::Identity => {
            if u64::try_from(request.body.len())
                .map_or(true, |length| length > limits.max_decoded_request_bytes)
            {
                return Err(GitSmartHttpWriteError::PayloadTooLarge);
            }
            Ok(request.body)
        }
        GitRequestEncoding::Gzip => {
            let read_limit = limits
                .max_decoded_request_bytes
                .checked_add(1)
                .ok_or(GitSmartHttpWriteError::PayloadTooLarge)?;
            let mut decoded = Vec::new();
            GzDecoder::new(request.body.as_slice())
                .take(read_limit)
                .read_to_end(&mut decoded)
                .map_err(|_| GitSmartHttpWriteError::InvalidRequest)?;
            if u64::try_from(decoded.len())
                .map_or(true, |length| length > limits.max_decoded_request_bytes)
            {
                return Err(GitSmartHttpWriteError::PayloadTooLarge);
            }
            Ok(decoded)
        }
    }
}

fn map_object_store_open_error(error: GitObjectStoreError) -> GitSmartHttpWriteError {
    match error {
        GitObjectStoreError::UnsupportedAuthority | GitObjectStoreError::RepositoryUnavailable => {
            GitSmartHttpWriteError::RepositoryNotFound
        }
        GitObjectStoreError::ObjectTooLarge => GitSmartHttpWriteError::PayloadTooLarge,
        _ => GitSmartHttpWriteError::Unavailable,
    }
}

fn map_object_store_hydrate_error(error: GitObjectStoreError) -> GitSmartHttpWriteError {
    match error {
        GitObjectStoreError::ObjectTooLarge => GitSmartHttpWriteError::PayloadTooLarge,
        GitObjectStoreError::ObjectNotFound | GitObjectStoreError::IntegrityMismatch => {
            GitSmartHttpWriteError::MissingObject
        }
        _ => GitSmartHttpWriteError::Unavailable,
    }
}

fn map_object_store_write_error(error: GitObjectStoreError) -> GitSmartHttpWriteError {
    match error {
        GitObjectStoreError::ObjectTooLarge => GitSmartHttpWriteError::PayloadTooLarge,
        GitObjectStoreError::ObjectNotFound => GitSmartHttpWriteError::MissingObject,
        _ => GitSmartHttpWriteError::Unavailable,
    }
}

fn map_object_store_publish_error(error: GitObjectStoreError) -> GitSmartHttpWriteError {
    match error {
        GitObjectStoreError::ConcurrentRefUpdate => GitSmartHttpWriteError::ConcurrentUpdate,
        other => map_object_store_write_error(other),
    }
}
