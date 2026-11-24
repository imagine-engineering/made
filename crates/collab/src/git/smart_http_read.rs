use std::{
    io::Read,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use collaboration_domain::AuthorizationRequest;
use flate2::read::GzDecoder;
use tempfile::TempDir;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::{OwnedSemaphorePermit, Semaphore},
};

use super::{
    object_store::{
        GitObjectBackend, GitObjectStore, GitObjectStoreError, GitObjectStoreLimits, GitRefManifest,
    },
    repository_registry::{
        HostedRepository, HostedRepositoryRegistry, HostedRepositoryRegistryError,
        RepositoryPermission,
    },
};

const ADVERTISEMENT_CONTENT_TYPE: &str = "application/x-git-upload-pack-advertisement";
const REQUEST_CONTENT_TYPE: &str = "application/x-git-upload-pack-request";
const RESULT_CONTENT_TYPE: &str = "application/x-git-upload-pack-result";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GitSmartHttpReadLimits {
    pub max_request_bytes: u64,
    pub max_decoded_request_bytes: u64,
    pub max_advertisement_bytes: u64,
    pub max_response_bytes: u64,
    pub max_repository_bytes: u64,
    pub max_concurrent_processes: usize,
    pub process_timeout: Duration,
}

impl GitSmartHttpReadLimits {
    pub fn validate(self) -> Result<Self, GitSmartHttpReadError> {
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
            return Err(GitSmartHttpReadError::InvalidConfiguration);
        }
        Ok(self)
    }
}

impl Default for GitSmartHttpReadLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 8 * 1024 * 1024,
            max_decoded_request_bytes: 64 * 1024 * 1024,
            max_advertisement_bytes: 4 * 1024 * 1024,
            max_response_bytes: 512 * 1024 * 1024,
            max_repository_bytes: 4 * 1024 * 1024 * 1024,
            max_concurrent_processes: 20,
            process_timeout: Duration::from_secs(300),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GitRequestEncoding {
    Identity,
    Gzip,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitUploadPackRequest {
    pub content_type: String,
    pub content_encoding: GitRequestEncoding,
    pub git_protocol: Option<String>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitSmartHttpResponse {
    pub content_type: &'static str,
    pub cache_control: &'static str,
    pub body: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum GitReadAuthorizationError {
    #[error("repository is unavailable")]
    Denied,
    #[error("repository authorization is unavailable")]
    Unavailable,
}

#[async_trait]
pub trait GitReadAuthorizer: Send + Sync {
    async fn authorize_read(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitReadAuthorizationError>;
}

#[async_trait]
impl GitReadAuthorizer for HostedRepositoryRegistry {
    async fn authorize_read(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitReadAuthorizationError> {
        self.authorize_access(authorization, RepositoryPermission::Read)
            .await
            .map_err(|error| match error {
                HostedRepositoryRegistryError::Unavailable(_) => {
                    GitReadAuthorizationError::Unavailable
                }
                _ => GitReadAuthorizationError::Denied,
            })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GitSmartHttpReadError {
    #[error("Git smart-HTTP read configuration is invalid")]
    InvalidConfiguration,
    #[error("repository is unavailable")]
    RepositoryNotFound,
    #[error("Git smart-HTTP request is invalid")]
    InvalidRequest,
    #[error("Git smart-HTTP request or response exceeds its configured limit")]
    PayloadTooLarge,
    #[error("Git smart-HTTP service is busy")]
    Busy,
    #[error("Git smart-HTTP operation timed out")]
    Timeout,
    #[error("Git smart-HTTP backend is unavailable")]
    Unavailable,
}

pub struct GitSmartHttpReadService {
    authorizer: Arc<dyn GitReadAuthorizer>,
    backend: Arc<dyn GitObjectBackend>,
    scratch_directory: PathBuf,
    object_store_limits: GitObjectStoreLimits,
    limits: GitSmartHttpReadLimits,
    process_permits: Arc<Semaphore>,
}

impl GitSmartHttpReadService {
    pub fn new(
        authorizer: Arc<dyn GitReadAuthorizer>,
        backend: Arc<dyn GitObjectBackend>,
        scratch_directory: impl Into<PathBuf>,
        object_store_limits: GitObjectStoreLimits,
        limits: GitSmartHttpReadLimits,
    ) -> Result<Self, GitSmartHttpReadError> {
        let scratch_directory = scratch_directory.into();
        let metadata = std::fs::metadata(&scratch_directory)
            .map_err(|_| GitSmartHttpReadError::InvalidConfiguration)?;
        if !scratch_directory.is_absolute() || !metadata.is_dir() {
            return Err(GitSmartHttpReadError::InvalidConfiguration);
        }
        let limits = limits.validate()?;
        let object_store_limits = object_store_limits
            .validate()
            .map_err(|_| GitSmartHttpReadError::InvalidConfiguration)?;
        Ok(Self {
            authorizer,
            backend,
            scratch_directory,
            object_store_limits,
            limits,
            process_permits: Arc::new(Semaphore::new(limits.max_concurrent_processes)),
        })
    }

    pub async fn advertise_refs(
        &self,
        authorization: &AuthorizationRequest<'_>,
        git_protocol: Option<&str>,
    ) -> Result<GitSmartHttpResponse, GitSmartHttpReadError> {
        let repository = self.authorize(authorization).await?;
        let git_protocol = validate_git_protocol(git_protocol)?;
        let permit = self.acquire_permit()?;
        let hydrated = self.hydrate(&repository).await?;
        let output = self
            .run_upload_pack(
                &hydrated,
                &[],
                true,
                git_protocol.as_deref(),
                self.limits.max_advertisement_bytes,
                permit,
            )
            .await?;
        let service_line = b"# service=git-upload-pack\n";
        let packet_length = service_line
            .len()
            .checked_add(4)
            .ok_or(GitSmartHttpReadError::PayloadTooLarge)?;
        let prefix = format!("{packet_length:04x}");
        let total_length = prefix
            .len()
            .checked_add(service_line.len())
            .and_then(|length| length.checked_add(4))
            .and_then(|length| length.checked_add(output.len()))
            .ok_or(GitSmartHttpReadError::PayloadTooLarge)?;
        if u64::try_from(total_length)
            .map_or(true, |length| length > self.limits.max_advertisement_bytes)
        {
            return Err(GitSmartHttpReadError::PayloadTooLarge);
        }
        let mut body = Vec::with_capacity(total_length);
        body.extend_from_slice(prefix.as_bytes());
        body.extend_from_slice(service_line);
        body.extend_from_slice(b"0000");
        body.extend_from_slice(&output);
        Ok(GitSmartHttpResponse {
            content_type: ADVERTISEMENT_CONTENT_TYPE,
            cache_control: "no-cache",
            body,
        })
    }

    pub async fn upload_pack(
        &self,
        authorization: &AuthorizationRequest<'_>,
        request: GitUploadPackRequest,
    ) -> Result<GitSmartHttpResponse, GitSmartHttpReadError> {
        let repository = self.authorize(authorization).await?;
        if request.content_type != REQUEST_CONTENT_TYPE {
            return Err(GitSmartHttpReadError::InvalidRequest);
        }
        let git_protocol = validate_git_protocol(request.git_protocol.as_deref())?;
        let body = decode_request_body(request, self.limits)?;
        let permit = self.acquire_permit()?;
        let hydrated = self.hydrate(&repository).await?;
        let body = self
            .run_upload_pack(
                &hydrated,
                &body,
                false,
                git_protocol.as_deref(),
                self.limits.max_response_bytes,
                permit,
            )
            .await?;
        Ok(GitSmartHttpResponse {
            content_type: RESULT_CONTENT_TYPE,
            cache_control: "no-cache",
            body,
        })
    }

    async fn authorize(
        &self,
        authorization: &AuthorizationRequest<'_>,
    ) -> Result<HostedRepository, GitSmartHttpReadError> {
        self.authorizer
            .authorize_read(authorization)
            .await
            .map_err(|error| match error {
                GitReadAuthorizationError::Denied => GitSmartHttpReadError::RepositoryNotFound,
                GitReadAuthorizationError::Unavailable => GitSmartHttpReadError::Unavailable,
            })
    }

    fn acquire_permit(&self) -> Result<OwnedSemaphorePermit, GitSmartHttpReadError> {
        self.process_permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| GitSmartHttpReadError::Busy)
    }

    async fn hydrate(
        &self,
        repository: &HostedRepository,
    ) -> Result<HydratedRepository, GitSmartHttpReadError> {
        let object_store = GitObjectStore::for_authorized_repository(
            self.backend.clone(),
            repository,
            self.object_store_limits,
        )
        .map_err(map_object_store_open_error)?;
        let snapshot = object_store
            .read_refs()
            .await
            .map_err(|error| match error {
                GitObjectStoreError::RefsNotFound => GitSmartHttpReadError::RepositoryNotFound,
                other => map_object_store_hydrate_error(other),
            })?;
        let manifest = snapshot.manifest();
        let temporary_directory = TempDir::new_in(&self.scratch_directory)
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
        let repository_path = temporary_directory.path().to_path_buf();
        run_git(
            &repository_path,
            &["init", "--bare", "--quiet"],
            self.limits.process_timeout,
        )
        .await?;
        let pack_directory = repository_path.join("objects").join("pack");
        tokio::fs::create_dir_all(&pack_directory)
            .await
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
        let mut repository_bytes = 0u64;
        for digest in manifest.objects() {
            let bytes = object_store
                .get_object(digest)
                .await
                .map_err(map_object_store_hydrate_error)?;
            let object_bytes =
                u64::try_from(bytes.len()).map_err(|_| GitSmartHttpReadError::PayloadTooLarge)?;
            repository_bytes = repository_bytes
                .checked_add(object_bytes)
                .ok_or(GitSmartHttpReadError::PayloadTooLarge)?;
            if repository_bytes > self.limits.max_repository_bytes {
                return Err(GitSmartHttpReadError::PayloadTooLarge);
            }
            let pack_path = pack_directory.join(format!("pack-{}.pack", digest.as_str()));
            tokio::fs::write(&pack_path, bytes)
                .await
                .map_err(|_| GitSmartHttpReadError::Unavailable)?;
            let pack_path = pack_path
                .to_str()
                .ok_or(GitSmartHttpReadError::Unavailable)?;
            run_git(
                &repository_path,
                &["index-pack", pack_path],
                self.limits.process_timeout,
            )
            .await?;
        }
        install_refs(&repository_path, manifest).await?;
        if !manifest.refs().is_empty() {
            run_git(
                &repository_path,
                &["fsck", "--connectivity-only", "--no-dangling"],
                self.limits.process_timeout,
            )
            .await?;
        }
        Ok(HydratedRepository {
            _temporary_directory: temporary_directory,
            path: repository_path,
        })
    }

    async fn run_upload_pack(
        &self,
        repository: &HydratedRepository,
        request_body: &[u8],
        advertise_refs: bool,
        git_protocol: Option<&str>,
        max_output_bytes: u64,
        _permit: OwnedSemaphorePermit,
    ) -> Result<Vec<u8>, GitSmartHttpReadError> {
        let mut command = Command::new("git");
        command
            .arg("upload-pack")
            .arg("--stateless-rpc")
            .current_dir(repository.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if advertise_refs {
            command.arg("--advertise-refs");
        }
        command.arg(repository.path());
        harden_git_environment(&mut command, repository.path(), git_protocol);
        let mut child = command
            .spawn()
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or(GitSmartHttpReadError::Unavailable)?;
        let request_body = request_body.to_vec();
        let stdin_task = tokio::spawn(async move {
            stdin.write_all(&request_body).await?;
            stdin.shutdown().await
        });
        let stdout = child
            .stdout
            .take()
            .ok_or(GitSmartHttpReadError::Unavailable)?;
        let stdout_task = tokio::spawn(read_bounded_and_drain(stdout, max_output_bytes));
        let status = match tokio::time::timeout(self.limits.process_timeout, child.wait()).await {
            Ok(Ok(status)) => status,
            Ok(Err(_)) => {
                stdin_task.abort();
                stdout_task.abort();
                return Err(GitSmartHttpReadError::Unavailable);
            }
            Err(_) => {
                stdin_task.abort();
                stdout_task.abort();
                child
                    .kill()
                    .await
                    .map_err(|_| GitSmartHttpReadError::Unavailable)?;
                return Err(GitSmartHttpReadError::Timeout);
            }
        };
        let stdin_result = stdin_task
            .await
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
        let output = stdout_task
            .await
            .map_err(|_| GitSmartHttpReadError::Unavailable)??;
        if output.exceeded_limit {
            return Err(GitSmartHttpReadError::PayloadTooLarge);
        }
        if !status.success() || stdin_result.is_err() {
            return Err(if advertise_refs {
                GitSmartHttpReadError::Unavailable
            } else {
                GitSmartHttpReadError::InvalidRequest
            });
        }
        Ok(output.bytes)
    }
}

struct HydratedRepository {
    _temporary_directory: TempDir,
    path: PathBuf,
}

impl HydratedRepository {
    fn path(&self) -> &Path {
        &self.path
    }
}

struct BoundedOutput {
    bytes: Vec<u8>,
    exceeded_limit: bool,
}

async fn read_bounded_and_drain(
    mut reader: impl AsyncRead + Unpin,
    max_bytes: u64,
) -> Result<BoundedOutput, GitSmartHttpReadError> {
    let capacity =
        usize::try_from(max_bytes).map_err(|_| GitSmartHttpReadError::PayloadTooLarge)?;
    let mut bytes = Vec::with_capacity(capacity.min(64 * 1024));
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    let mut exceeded_limit = false;
    loop {
        let read = reader
            .read(&mut buffer)
            .await
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(u64::try_from(read).map_err(|_| GitSmartHttpReadError::PayloadTooLarge)?)
            .ok_or(GitSmartHttpReadError::PayloadTooLarge)?;
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

async fn install_refs(
    repository_path: &Path,
    manifest: &GitRefManifest,
) -> Result<(), GitSmartHttpReadError> {
    for (ref_name, object_id) in manifest.refs() {
        let ref_path = repository_path.join(ref_name.as_str());
        let parent = ref_path
            .parent()
            .ok_or(GitSmartHttpReadError::Unavailable)?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
        tokio::fs::write(ref_path, format!("{}\n", object_id.as_str()))
            .await
            .map_err(|_| GitSmartHttpReadError::Unavailable)?;
    }
    if let Some(head) = manifest.head() {
        tokio::fs::write(
            repository_path.join("HEAD"),
            format!("ref: {}\n", head.as_str()),
        )
        .await
        .map_err(|_| GitSmartHttpReadError::Unavailable)?;
    }
    Ok(())
}

async fn run_git(
    repository_path: &Path,
    arguments: &[&str],
    timeout: Duration,
) -> Result<(), GitSmartHttpReadError> {
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
        .map_err(|_| GitSmartHttpReadError::Timeout)?
        .map_err(|_| GitSmartHttpReadError::Unavailable)?;
    if !status.success() {
        return Err(GitSmartHttpReadError::Unavailable);
    }
    Ok(())
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

fn validate_git_protocol(value: Option<&str>) -> Result<Option<String>, GitSmartHttpReadError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_empty()
        || value.len() > 256
        || !value
            .chars()
            .all(|character| character.is_ascii_graphic() && character != '\0')
    {
        return Err(GitSmartHttpReadError::InvalidRequest);
    }
    Ok(Some(value.to_owned()))
}

fn decode_request_body(
    request: GitUploadPackRequest,
    limits: GitSmartHttpReadLimits,
) -> Result<Vec<u8>, GitSmartHttpReadError> {
    if u64::try_from(request.body.len()).map_or(true, |length| length > limits.max_request_bytes) {
        return Err(GitSmartHttpReadError::PayloadTooLarge);
    }
    match request.content_encoding {
        GitRequestEncoding::Identity => {
            if u64::try_from(request.body.len())
                .map_or(true, |length| length > limits.max_decoded_request_bytes)
            {
                return Err(GitSmartHttpReadError::PayloadTooLarge);
            }
            Ok(request.body)
        }
        GitRequestEncoding::Gzip => {
            let read_limit = limits
                .max_decoded_request_bytes
                .checked_add(1)
                .ok_or(GitSmartHttpReadError::PayloadTooLarge)?;
            let mut decoded = Vec::new();
            GzDecoder::new(request.body.as_slice())
                .take(read_limit)
                .read_to_end(&mut decoded)
                .map_err(|_| GitSmartHttpReadError::InvalidRequest)?;
            if u64::try_from(decoded.len())
                .map_or(true, |length| length > limits.max_decoded_request_bytes)
            {
                return Err(GitSmartHttpReadError::PayloadTooLarge);
            }
            Ok(decoded)
        }
    }
}

fn map_object_store_open_error(error: GitObjectStoreError) -> GitSmartHttpReadError {
    match error {
        GitObjectStoreError::UnsupportedAuthority | GitObjectStoreError::RepositoryUnavailable => {
            GitSmartHttpReadError::RepositoryNotFound
        }
        GitObjectStoreError::ObjectTooLarge => GitSmartHttpReadError::PayloadTooLarge,
        _ => GitSmartHttpReadError::Unavailable,
    }
}

fn map_object_store_hydrate_error(error: GitObjectStoreError) -> GitSmartHttpReadError {
    match error {
        GitObjectStoreError::ObjectTooLarge => GitSmartHttpReadError::PayloadTooLarge,
        _ => GitSmartHttpReadError::Unavailable,
    }
}
