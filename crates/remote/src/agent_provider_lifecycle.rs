use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

use crate::{
    agent_provider_discovery::{
        AgentProviderCandidate, AgentProviderDiscoveryError, AgentProviderDiscoveryReport,
        AgentProviderExecutableReference, AgentProviderTrust, validate_provider_id,
    },
    agent_provider_protocol::{
        AgentProviderCancellation, AgentProviderInfo, AgentProviderOperation,
        AgentProviderProtocolError, AgentProviderResponse, invoke_agent_provider_cancellable,
    },
};

const AGENT_IDENTITY_LENGTH: usize = 64;
const OPERATION_ID_LIMIT: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentProviderExecutionApproval {
    pub executable: AgentProviderExecutableReference,
    pub allow_untrusted: bool,
}

pub struct AgentProviderDeployInput {
    pub operation_id: String,
    pub work_directory: PathBuf,
    pub agent: Value,
    pub provider_config: Value,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentProviderDeploymentRecord {
    pub generation: u64,
    pub operation_id: String,
    pub agent_id: String,
    pub provider_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentProviderDeployDisposition {
    Deployed(AgentProviderDeploymentRecord),
    AlreadyDeployed(AgentProviderDeploymentRecord),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RemoteAgentLifecycleState {
    Ready,
    Deploying {
        generation: u64,
        operation_id: String,
    },
    DeploymentUncertain {
        generation: u64,
        operation_id: String,
    },
    Deployed(AgentProviderDeploymentRecord),
    TerminationRequested {
        deployment: AgentProviderDeploymentRecord,
        operation_id: String,
    },
    Terminated {
        deployment: AgentProviderDeploymentRecord,
        operation_id: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteAgentInspectionSource {
    CanonicalLifecycleOnly,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteAgentInspection {
    pub provider_id: String,
    pub agent_identity: String,
    pub state: RemoteAgentLifecycleState,
    pub source: RemoteAgentInspectionSource,
    pub presence_bound: bool,
    pub provider_inspection_supported: bool,
    pub provider_termination_supported: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteAgentShutdownRequest {
    pub operation_id: String,
    pub provider_id: String,
    pub agent_identity: String,
    pub agent_id: String,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RemoteAgentTerminationDisposition {
    NotDeployed,
    AlreadyRequested(RemoteAgentShutdownRequest),
    Requested(RemoteAgentShutdownRequest),
    AlreadyTerminated(RemoteAgentShutdownRequest),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteAgentShutdownError {
    diagnostic: String,
}

impl RemoteAgentShutdownError {
    pub fn new(diagnostic: impl Into<String>) -> Self {
        let diagnostic = diagnostic.into();
        Self {
            diagnostic: bounded_diagnostic(&diagnostic),
        }
    }
}

impl fmt::Display for RemoteAgentShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.diagnostic)
    }
}

impl std::error::Error for RemoteAgentShutdownError {}

#[async_trait]
pub trait RemoteAgentShutdown: Send + Sync {
    async fn request_shutdown(
        &self,
        request: &RemoteAgentShutdownRequest,
    ) -> Result<(), RemoteAgentShutdownError>;
}

#[async_trait]
pub trait AgentProviderDeploymentRunner: Send + Sync {
    async fn deploy(
        &self,
        candidate: &AgentProviderCandidate,
        input: &AgentProviderDeployInput,
        cancellation: &AgentProviderCancellation,
        background_executor: &gpui::BackgroundExecutor,
    ) -> Result<StagedAgentProviderDeployment, AgentProviderLifecycleError>;
}

pub struct FilesystemAgentProviderDeploymentRunner;

#[async_trait]
impl AgentProviderDeploymentRunner for FilesystemAgentProviderDeploymentRunner {
    async fn deploy(
        &self,
        candidate: &AgentProviderCandidate,
        input: &AgentProviderDeployInput,
        cancellation: &AgentProviderCancellation,
        background_executor: &gpui::BackgroundExecutor,
    ) -> Result<StagedAgentProviderDeployment, AgentProviderLifecycleError> {
        let candidate = candidate.clone();
        let staged = background_executor
            .spawn(async move { stage_provider(&candidate) })
            .await?;

        let info_request = json!({
            "op": "info",
            "request_id": input.operation_id,
        });
        let info = invoke_agent_provider_cancellable(
            &staged.candidate,
            &input.work_directory,
            AgentProviderOperation::Info,
            &info_request,
            cancellation,
            background_executor,
        )
        .await
        .map_err(|source| AgentProviderLifecycleError::ProviderOperation {
            operation: AgentProviderOperation::Info,
            source,
        })?;
        let AgentProviderResponse::Info(info) = info else {
            return Err(AgentProviderLifecycleError::UnexpectedResponse {
                operation: AgentProviderOperation::Info,
            });
        };

        let deploy_request = json!({
            "op": "deploy",
            "request_id": input.operation_id,
            "agent": input.agent,
            "provider_config": input.provider_config,
        });
        let deployment = invoke_agent_provider_cancellable(
            &staged.candidate,
            &input.work_directory,
            AgentProviderOperation::Deploy,
            &deploy_request,
            cancellation,
            background_executor,
        )
        .await
        .map_err(|source| AgentProviderLifecycleError::ProviderOperation {
            operation: AgentProviderOperation::Deploy,
            source,
        })?;
        let AgentProviderResponse::Deploy(deployment) = deployment else {
            return Err(AgentProviderLifecycleError::UnexpectedResponse {
                operation: AgentProviderOperation::Deploy,
            });
        };

        Ok(StagedAgentProviderDeployment {
            info,
            agent_id: deployment.agent_id,
            provider_sha256: staged.sha256.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StagedAgentProviderDeployment {
    pub info: AgentProviderInfo,
    pub agent_id: String,
    pub provider_sha256: String,
}

pub struct RemoteAgentProviderLifecycle<R = FilesystemAgentProviderDeploymentRunner> {
    provider_id: String,
    agent_identity: String,
    generation: u64,
    state: RemoteAgentLifecycleState,
    runner: R,
}

impl RemoteAgentProviderLifecycle<FilesystemAgentProviderDeploymentRunner> {
    pub fn new(
        provider_id: impl Into<String>,
        agent_identity: impl Into<String>,
    ) -> Result<Self, AgentProviderLifecycleError> {
        Self::with_runner(
            provider_id,
            agent_identity,
            FilesystemAgentProviderDeploymentRunner,
        )
    }
}

impl<R> RemoteAgentProviderLifecycle<R>
where
    R: AgentProviderDeploymentRunner,
{
    pub fn with_runner(
        provider_id: impl Into<String>,
        agent_identity: impl Into<String>,
        runner: R,
    ) -> Result<Self, AgentProviderLifecycleError> {
        let provider_id = provider_id.into();
        validate_provider_id(&provider_id)?;
        let agent_identity = agent_identity.into();
        if agent_identity.len() != AGENT_IDENTITY_LENGTH
            || !agent_identity
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AgentProviderLifecycleError::InvalidAgentIdentity);
        }
        Ok(Self {
            provider_id,
            agent_identity,
            generation: 0,
            state: RemoteAgentLifecycleState::Ready,
            runner,
        })
    }

    pub fn inspect(&self) -> RemoteAgentInspection {
        RemoteAgentInspection {
            provider_id: self.provider_id.clone(),
            agent_identity: self.agent_identity.clone(),
            state: self.state.clone(),
            source: RemoteAgentInspectionSource::CanonicalLifecycleOnly,
            presence_bound: false,
            provider_inspection_supported: false,
            provider_termination_supported: false,
        }
    }

    pub async fn deploy(
        &mut self,
        discovery: &AgentProviderDiscoveryReport,
        approval: &AgentProviderExecutionApproval,
        input: AgentProviderDeployInput,
        cancellation: &AgentProviderCancellation,
        background_executor: &gpui::BackgroundExecutor,
    ) -> Result<AgentProviderDeployDisposition, AgentProviderLifecycleError> {
        validate_operation_id(&input.operation_id)?;
        if approval.executable.provider_id != self.provider_id {
            return Err(AgentProviderLifecycleError::ProviderMismatch {
                expected: self.provider_id.clone(),
                actual: approval.executable.provider_id.clone(),
            });
        }
        let candidate = discovery.resolve_reference(&approval.executable)?;
        if candidate.trust == AgentProviderTrust::Untrusted && !approval.allow_untrusted {
            return Err(AgentProviderLifecycleError::UntrustedProviderNotApproved {
                provider_id: candidate.provider_id.clone(),
                path: candidate.canonical_path.clone(),
            });
        }
        match &self.state {
            RemoteAgentLifecycleState::Deployed(deployment) => {
                return Ok(AgentProviderDeployDisposition::AlreadyDeployed(
                    deployment.clone(),
                ));
            }
            RemoteAgentLifecycleState::Deploying { .. } => {
                return Err(AgentProviderLifecycleError::DeploymentInProgress);
            }
            RemoteAgentLifecycleState::TerminationRequested { .. } => {
                return Err(AgentProviderLifecycleError::TerminationPending);
            }
            RemoteAgentLifecycleState::Ready
            | RemoteAgentLifecycleState::DeploymentUncertain { .. }
            | RemoteAgentLifecycleState::Terminated { .. } => {}
        }

        let previous_state = self.state.clone();
        let generation = match &previous_state {
            RemoteAgentLifecycleState::DeploymentUncertain { generation, .. } => *generation,
            _ => self
                .generation
                .checked_add(1)
                .ok_or(AgentProviderLifecycleError::GenerationExhausted)?,
        };
        self.state = RemoteAgentLifecycleState::Deploying {
            generation,
            operation_id: input.operation_id.clone(),
        };
        let deployment = match self
            .runner
            .deploy(candidate, &input, cancellation, background_executor)
            .await
        {
            Ok(deployment) => deployment,
            Err(error) => {
                self.state = if error.is_uncertain_deploy_outcome() {
                    RemoteAgentLifecycleState::DeploymentUncertain {
                        generation,
                        operation_id: input.operation_id.clone(),
                    }
                } else {
                    previous_state
                };
                return Err(error);
            }
        };
        let deployment = AgentProviderDeploymentRecord {
            generation,
            operation_id: input.operation_id,
            agent_id: deployment.agent_id,
            provider_sha256: deployment.provider_sha256,
        };
        self.generation = generation;
        self.state = RemoteAgentLifecycleState::Deployed(deployment.clone());
        Ok(AgentProviderDeployDisposition::Deployed(deployment))
    }

    pub async fn terminate(
        &mut self,
        operation_id: impl Into<String>,
        shutdown: &dyn RemoteAgentShutdown,
    ) -> Result<RemoteAgentTerminationDisposition, AgentProviderLifecycleError> {
        let operation_id = operation_id.into();
        validate_operation_id(&operation_id)?;
        let deployment = match &self.state {
            RemoteAgentLifecycleState::Ready => {
                return Ok(RemoteAgentTerminationDisposition::NotDeployed);
            }
            RemoteAgentLifecycleState::Deploying { .. } => {
                return Err(AgentProviderLifecycleError::DeploymentInProgress);
            }
            RemoteAgentLifecycleState::DeploymentUncertain { .. } => {
                return Err(AgentProviderLifecycleError::DeploymentOutcomeUnknown);
            }
            RemoteAgentLifecycleState::Deployed(deployment) => deployment.clone(),
            RemoteAgentLifecycleState::TerminationRequested {
                deployment,
                operation_id,
            } => {
                return Ok(RemoteAgentTerminationDisposition::AlreadyRequested(
                    shutdown_request(
                        &self.provider_id,
                        &self.agent_identity,
                        deployment,
                        operation_id,
                    ),
                ));
            }
            RemoteAgentLifecycleState::Terminated {
                deployment,
                operation_id,
            } => {
                return Ok(RemoteAgentTerminationDisposition::AlreadyTerminated(
                    shutdown_request(
                        &self.provider_id,
                        &self.agent_identity,
                        deployment,
                        operation_id,
                    ),
                ));
            }
        };
        let request = shutdown_request(
            &self.provider_id,
            &self.agent_identity,
            &deployment,
            &operation_id,
        );
        shutdown.request_shutdown(&request).await?;
        self.state = RemoteAgentLifecycleState::TerminationRequested {
            deployment,
            operation_id,
        };
        Ok(RemoteAgentTerminationDisposition::Requested(request))
    }

    pub fn confirm_terminated(
        &mut self,
        agent_id: &str,
    ) -> Result<(), AgentProviderLifecycleError> {
        let RemoteAgentLifecycleState::TerminationRequested {
            deployment,
            operation_id,
        } = &self.state
        else {
            return Err(AgentProviderLifecycleError::TerminationNotRequested);
        };
        if deployment.agent_id != agent_id {
            return Err(AgentProviderLifecycleError::AgentIdMismatch);
        }
        self.state = RemoteAgentLifecycleState::Terminated {
            deployment: deployment.clone(),
            operation_id: operation_id.clone(),
        };
        Ok(())
    }
}

fn shutdown_request(
    provider_id: &str,
    agent_identity: &str,
    deployment: &AgentProviderDeploymentRecord,
    operation_id: &str,
) -> RemoteAgentShutdownRequest {
    RemoteAgentShutdownRequest {
        operation_id: operation_id.to_owned(),
        provider_id: provider_id.to_owned(),
        agent_identity: agent_identity.to_owned(),
        agent_id: deployment.agent_id.clone(),
        generation: deployment.generation,
    }
}

fn validate_operation_id(operation_id: &str) -> Result<(), AgentProviderLifecycleError> {
    if operation_id.is_empty() || operation_id.len() > OPERATION_ID_LIMIT {
        return Err(AgentProviderLifecycleError::InvalidOperationId);
    }
    if operation_id
        .bytes()
        .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
    {
        return Err(AgentProviderLifecycleError::InvalidOperationId);
    }
    Ok(())
}

struct StagedProvider {
    candidate: AgentProviderCandidate,
    sha256: String,
    _execution_guard: fs::File,
    _directory: tempfile::TempDir,
}

fn stage_provider(
    candidate: &AgentProviderCandidate,
) -> Result<StagedProvider, AgentProviderLifecycleError> {
    let directory = tempfile::Builder::new()
        .prefix("zed-agent-provider-")
        .tempdir()
        .map_err(|source| {
            staging_error(
                "create private directory",
                &candidate.canonical_path,
                source,
            )
        })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).map_err(
            |source| staging_error("protect private directory", directory.path(), source),
        )?;
    }

    let suffix = candidate
        .canonical_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| {
            ["exe", "bat", "cmd"]
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
        .map(|extension| format!(".{extension}"))
        .unwrap_or_default();
    let staged_path = directory.path().join(format!("provider{suffix}"));
    let mut source_file = fs::File::open(&candidate.canonical_path).map_err(|source| {
        staging_error(
            "open discovered executable",
            &candidate.canonical_path,
            source,
        )
    })?;
    let mut staged_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)
        .map_err(|source| staging_error("create staged executable", &staged_path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = io::Read::read(&mut source_file, &mut buffer).map_err(|source| {
            staging_error(
                "read discovered executable",
                &candidate.canonical_path,
                source,
            )
        })?;
        if count == 0 {
            break;
        }
        io::Write::write_all(&mut staged_file, &buffer[..count])
            .map_err(|source| staging_error("write staged executable", &staged_path, source))?;
        hasher.update(&buffer[..count]);
    }
    staged_file
        .sync_all()
        .map_err(|source| staging_error("sync staged executable", &staged_path, source))?;
    let mut permissions = staged_file
        .metadata()
        .map_err(|source| staging_error("inspect staged executable", &staged_path, source))?
        .permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        permissions.set_mode(0o500);
    }
    #[cfg(not(unix))]
    permissions.set_readonly(true);
    fs::set_permissions(&staged_path, permissions)
        .map_err(|source| staging_error("protect staged executable", &staged_path, source))?;
    drop(staged_file);

    #[cfg(windows)]
    let execution_guard = {
        use std::os::windows::fs::OpenOptionsExt as _;
        fs::OpenOptions::new()
            .read(true)
            .share_mode(windows::Win32::Storage::FileSystem::FILE_SHARE_READ.0)
            .open(&staged_path)
    };
    #[cfg(not(windows))]
    let execution_guard = fs::File::open(&staged_path);
    let execution_guard = execution_guard
        .map_err(|source| staging_error("lock staged executable", &staged_path, source))?;
    let sha256 = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(StagedProvider {
        candidate: AgentProviderCandidate {
            provider_id: candidate.provider_id.clone(),
            discovered_path: staged_path.clone(),
            canonical_path: staged_path,
            trust: candidate.trust,
        },
        sha256,
        _execution_guard: execution_guard,
        _directory: directory,
    })
}

fn staging_error(
    action: &'static str,
    path: &Path,
    source: io::Error,
) -> AgentProviderLifecycleError {
    AgentProviderLifecycleError::Staging {
        action,
        path: path.to_owned(),
        source,
    }
}

fn bounded_diagnostic(diagnostic: &str) -> String {
    const LIMIT: usize = 1024;
    if diagnostic.len() <= LIMIT {
        return diagnostic.to_owned();
    }
    let mut end = LIMIT;
    while !diagnostic.is_char_boundary(end) {
        end -= 1;
    }
    diagnostic[..end].to_owned()
}

#[derive(Debug)]
pub enum AgentProviderLifecycleError {
    Discovery(AgentProviderDiscoveryError),
    ProviderOperation {
        operation: AgentProviderOperation,
        source: AgentProviderProtocolError,
    },
    Shutdown(RemoteAgentShutdownError),
    InvalidAgentIdentity,
    InvalidOperationId,
    ProviderMismatch {
        expected: String,
        actual: String,
    },
    UntrustedProviderNotApproved {
        provider_id: String,
        path: PathBuf,
    },
    DeploymentInProgress,
    DeploymentOutcomeUnknown,
    TerminationPending,
    TerminationNotRequested,
    AgentIdMismatch,
    GenerationExhausted,
    Staging {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    UnexpectedResponse {
        operation: AgentProviderOperation,
    },
}

impl fmt::Display for AgentProviderLifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery(error) => error.fmt(formatter),
            Self::ProviderOperation { operation, source } => {
                write!(formatter, "provider {} failed: {source}", operation.name())
            }
            Self::Shutdown(error) => write!(formatter, "remote shutdown was not accepted: {error}"),
            Self::InvalidAgentIdentity => {
                formatter.write_str("agent identity must be 64 lowercase hexadecimal characters")
            }
            Self::InvalidOperationId => {
                formatter.write_str("operation ID must be 1-128 visible non-whitespace bytes")
            }
            Self::ProviderMismatch { expected, actual } => write!(
                formatter,
                "provider approval names {actual:?}; lifecycle requires {expected:?}"
            ),
            Self::UntrustedProviderNotApproved { provider_id, path } => write!(
                formatter,
                "untrusted provider {provider_id:?} at {} requires explicit approval",
                path.display()
            ),
            Self::DeploymentInProgress => {
                formatter.write_str("provider deployment is already in progress")
            }
            Self::DeploymentOutcomeUnknown => formatter.write_str(
                "provider deployment outcome is unknown; reconcile it before termination",
            ),
            Self::TerminationPending => {
                formatter.write_str("remote termination is awaiting canonical confirmation")
            }
            Self::TerminationNotRequested => {
                formatter.write_str("remote termination has not been requested")
            }
            Self::AgentIdMismatch => {
                formatter.write_str("termination confirmation names a different provider agent ID")
            }
            Self::GenerationExhausted => {
                formatter.write_str("remote deployment generation is exhausted")
            }
            Self::Staging {
                action,
                path,
                source,
            } => write!(
                formatter,
                "failed to {action} at {}: {source}",
                path.display()
            ),
            Self::UnexpectedResponse { operation } => write!(
                formatter,
                "provider returned the wrong response shape for {}",
                operation.name()
            ),
        }
    }
}

impl std::error::Error for AgentProviderLifecycleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Discovery(error) => Some(error),
            Self::ProviderOperation { source, .. } => Some(source),
            Self::Shutdown(error) => Some(error),
            Self::Staging { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<AgentProviderDiscoveryError> for AgentProviderLifecycleError {
    fn from(error: AgentProviderDiscoveryError) -> Self {
        Self::Discovery(error)
    }
}

impl AgentProviderLifecycleError {
    fn is_uncertain_deploy_outcome(&self) -> bool {
        matches!(
            self,
            Self::ProviderOperation {
                operation: AgentProviderOperation::Deploy,
                ..
            }
        )
    }
}

impl From<RemoteAgentShutdownError> for AgentProviderLifecycleError {
    fn from(error: RemoteAgentShutdownError) -> Self {
        Self::Shutdown(error)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt as _;

    use super::*;
    use crate::agent_provider_discovery::AgentProviderSearchDirectory;

    const AGENT_IDENTITY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[cfg(unix)]
    fn provider_script(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().expect("create provider fixture directory");
        let path = directory.path().join("buzz-backend-fixture");
        fs::write(&path, contents).expect("write provider fixture");
        let mut permissions = fs::metadata(&path)
            .expect("read provider fixture metadata")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).expect("mark provider fixture executable");
        (directory, path)
    }

    fn discovery(directory: &Path) -> AgentProviderDiscoveryReport {
        AgentProviderDiscoveryReport::discover([AgentProviderSearchDirectory::new(
            directory,
            AgentProviderTrust::Untrusted,
        )])
    }

    fn approval(report: &AgentProviderDiscoveryReport) -> AgentProviderExecutionApproval {
        AgentProviderExecutionApproval {
            executable: report
                .resolve("fixture")
                .expect("fixture provider should resolve")
                .executable_reference(),
            allow_untrusted: true,
        }
    }

    fn deploy_input(operation_id: &str, work_directory: &Path) -> AgentProviderDeployInput {
        AgentProviderDeployInput {
            operation_id: operation_id.to_owned(),
            work_directory: work_directory.to_owned(),
            agent: json!({"name": "fixture"}),
            provider_config: json!({}),
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn deploy_stages_once_negotiates_and_inspects_without_duplicate_instance(
        background_executor: gpui::BackgroundExecutor,
    ) {
        background_executor.allow_parking();
        let work_directory = tempfile::tempdir().expect("create work directory");
        let counter = work_directory.path().join("calls");
        let script = format!(
            "#!/bin/sh\nread request\nprintf x >> '{}'\ncase \"$request\" in\n  *'\"op\":\"info\"'*) printf '%s' '{{\"ok\":true,\"name\":\"fixture\",\"version\":\"1\",\"protocol_version\":1,\"description\":\"fixture\",\"config_schema\":{{}}}}' ;;\n  *) printf '%s' '{{\"ok\":true,\"agent_id\":\"remote-1\"}}' ;;\nesac\n",
            counter.display()
        );
        let (directory, provider_path) = provider_script(&script);
        let report = discovery(directory.path());
        let approval = approval(&report);
        let expected_digest = {
            let bytes = fs::read(provider_path).expect("read provider fixture");
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            hasher
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let mut lifecycle =
            RemoteAgentProviderLifecycle::new("fixture", AGENT_IDENTITY).expect("lifecycle");
        let cancellation = AgentProviderCancellation::default();
        let first = lifecycle
            .deploy(
                &report,
                &approval,
                deploy_input("deploy-1", work_directory.path()),
                &cancellation,
                &background_executor,
            )
            .await
            .expect("first deploy should succeed");
        let AgentProviderDeployDisposition::Deployed(deployment) = first else {
            panic!("first deployment should be new")
        };
        assert_eq!(deployment.agent_id, "remote-1");
        assert_eq!(deployment.provider_sha256, expected_digest);
        assert_eq!(deployment.generation, 1);
        let inspection = lifecycle.inspect();
        assert_eq!(
            inspection.state,
            RemoteAgentLifecycleState::Deployed(deployment.clone())
        );
        assert_eq!(
            inspection.source,
            RemoteAgentInspectionSource::CanonicalLifecycleOnly
        );
        assert!(!inspection.presence_bound);
        assert!(!inspection.provider_inspection_supported);
        assert!(!inspection.provider_termination_supported);

        let second = lifecycle
            .deploy(
                &report,
                &approval,
                deploy_input("deploy-2", work_directory.path()),
                &cancellation,
                &background_executor,
            )
            .await
            .expect("repeat deploy should converge");
        assert_eq!(
            second,
            AgentProviderDeployDisposition::AlreadyDeployed(deployment)
        );
        assert_eq!(fs::read(counter).expect("read invocation counter"), b"xx");
    }

    #[derive(Clone)]
    struct FailingRunner {
        error: fn() -> AgentProviderLifecycleError,
    }

    #[async_trait]
    impl AgentProviderDeploymentRunner for FailingRunner {
        async fn deploy(
            &self,
            _candidate: &AgentProviderCandidate,
            _input: &AgentProviderDeployInput,
            _cancellation: &AgentProviderCancellation,
            _background_executor: &gpui::BackgroundExecutor,
        ) -> Result<StagedAgentProviderDeployment, AgentProviderLifecycleError> {
            Err((self.error)())
        }
    }

    fn timeout_error() -> AgentProviderLifecycleError {
        AgentProviderLifecycleError::ProviderOperation {
            operation: AgentProviderOperation::Deploy,
            source: AgentProviderProtocolError::TimedOut {
                operation: AgentProviderOperation::Deploy,
                timeout: crate::agent_provider_protocol::AGENT_PROVIDER_DEPLOY_TIMEOUT,
            },
        }
    }

    #[gpui::test]
    async fn timeout_marks_deployment_outcome_uncertain(
        background_executor: gpui::BackgroundExecutor,
    ) {
        let directory = tempfile::tempdir().expect("create provider directory");
        let path = directory.path().join("buzz-backend-fixture");
        fs::write(&path, b"fixture").expect("write provider fixture");
        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(&path).expect("metadata").permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&path, permissions).expect("permissions");
        }
        let report = discovery(directory.path());
        let approval = approval(&report);
        let mut lifecycle = RemoteAgentProviderLifecycle::with_runner(
            "fixture",
            AGENT_IDENTITY,
            FailingRunner {
                error: timeout_error,
            },
        )
        .expect("lifecycle");
        let error = lifecycle
            .deploy(
                &report,
                &approval,
                deploy_input("deploy-timeout", directory.path()),
                &AgentProviderCancellation::default(),
                &background_executor,
            )
            .await
            .expect_err("timeout should fail");
        assert!(matches!(
            error,
            AgentProviderLifecycleError::ProviderOperation {
                operation: AgentProviderOperation::Deploy,
                source: AgentProviderProtocolError::TimedOut { .. }
            }
        ));
        assert!(matches!(
            lifecycle.inspect().state,
            RemoteAgentLifecycleState::DeploymentUncertain { generation: 1, .. }
        ));
    }

    #[derive(Default)]
    struct RecordingShutdown {
        requests: Mutex<Vec<RemoteAgentShutdownRequest>>,
    }

    #[async_trait]
    impl RemoteAgentShutdown for RecordingShutdown {
        async fn request_shutdown(
            &self,
            request: &RemoteAgentShutdownRequest,
        ) -> Result<(), RemoteAgentShutdownError> {
            self.requests
                .lock()
                .expect("shutdown request lock")
                .push(request.clone());
            Ok(())
        }
    }

    struct SuccessfulRunner;

    #[async_trait]
    impl AgentProviderDeploymentRunner for SuccessfulRunner {
        async fn deploy(
            &self,
            _candidate: &AgentProviderCandidate,
            _input: &AgentProviderDeployInput,
            _cancellation: &AgentProviderCancellation,
            _background_executor: &gpui::BackgroundExecutor,
        ) -> Result<StagedAgentProviderDeployment, AgentProviderLifecycleError> {
            Ok(StagedAgentProviderDeployment {
                info: AgentProviderInfo {
                    name: "fixture".to_owned(),
                    version: "1".to_owned(),
                    protocol_version: 1,
                    description: "fixture".to_owned(),
                    config_schema: json!({}),
                },
                agent_id: "remote-1".to_owned(),
                provider_sha256: "00".repeat(32),
            })
        }
    }

    #[gpui::test]
    async fn termination_requires_shutdown_acceptance_and_explicit_confirmation(
        background_executor: gpui::BackgroundExecutor,
    ) {
        let directory = tempfile::tempdir().expect("create provider directory");
        let path = directory.path().join("buzz-backend-fixture");
        fs::write(&path, b"fixture").expect("write provider fixture");
        #[cfg(unix)]
        {
            let mut permissions = fs::metadata(&path).expect("metadata").permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&path, permissions).expect("permissions");
        }
        let report = discovery(directory.path());
        let approval = approval(&report);
        let mut lifecycle =
            RemoteAgentProviderLifecycle::with_runner("fixture", AGENT_IDENTITY, SuccessfulRunner)
                .expect("lifecycle");
        lifecycle
            .deploy(
                &report,
                &approval,
                deploy_input("deploy-1", directory.path()),
                &AgentProviderCancellation::default(),
                &background_executor,
            )
            .await
            .expect("deploy");
        let shutdown = Arc::new(RecordingShutdown::default());
        let disposition = lifecycle
            .terminate("terminate-1", shutdown.as_ref())
            .await
            .expect("shutdown request");
        let RemoteAgentTerminationDisposition::Requested(request) = disposition else {
            panic!("termination should be requested")
        };
        assert_eq!(
            shutdown.requests.lock().expect("requests").as_slice(),
            std::slice::from_ref(&request)
        );
        assert!(matches!(
            lifecycle.inspect().state,
            RemoteAgentLifecycleState::TerminationRequested { .. }
        ));
        assert_eq!(
            lifecycle
                .terminate("terminate-2", shutdown.as_ref())
                .await
                .expect("repeat termination"),
            RemoteAgentTerminationDisposition::AlreadyRequested(request.clone())
        );
        assert_eq!(shutdown.requests.lock().expect("requests").len(), 1);
        lifecycle
            .confirm_terminated(&request.agent_id)
            .expect("confirm termination");
        assert!(matches!(
            lifecycle.inspect().state,
            RemoteAgentLifecycleState::Terminated { .. }
        ));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn cancellation_stops_in_flight_staged_deploy(
        background_executor: gpui::BackgroundExecutor,
    ) {
        background_executor.allow_parking();
        let work_directory = tempfile::tempdir().expect("create work directory");
        let marker = work_directory.path().join("deploy-started");
        let script = format!(
            "#!/bin/sh\nread request\ncase \"$request\" in\n  *'\"op\":\"info\"'*) printf '%s' '{{\"ok\":true,\"name\":\"fixture\",\"version\":\"1\",\"protocol_version\":1,\"description\":\"fixture\",\"config_schema\":{{}}}}' ;;\n  *) touch '{}'; while :; do :; done ;;\nesac\n",
            marker.display()
        );
        let (directory, _provider_path) = provider_script(&script);
        let report = discovery(directory.path());
        let approval = approval(&report);
        let cancellation = AgentProviderCancellation::default();
        let cancellation_thread = {
            let cancellation = cancellation.clone();
            let marker = marker.clone();
            std::thread::spawn(move || {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while !marker.exists() && std::time::Instant::now() < deadline {
                    std::thread::yield_now();
                }
                assert!(marker.exists(), "deploy fixture never started");
                cancellation.cancel();
            })
        };
        let mut lifecycle =
            RemoteAgentProviderLifecycle::new("fixture", AGENT_IDENTITY).expect("lifecycle");
        let error = lifecycle
            .deploy(
                &report,
                &approval,
                deploy_input("deploy-cancel", work_directory.path()),
                &cancellation,
                &background_executor,
            )
            .await
            .expect_err("cancelled deployment should fail");
        cancellation_thread.join().expect("cancellation thread");
        assert!(matches!(
            error,
            AgentProviderLifecycleError::ProviderOperation {
                operation: AgentProviderOperation::Deploy,
                source: AgentProviderProtocolError::Cancelled {
                    operation: AgentProviderOperation::Deploy
                }
            }
        ));
        assert!(matches!(
            lifecycle.inspect().state,
            RemoteAgentLifecycleState::DeploymentUncertain { generation: 1, .. }
        ));
    }
}
