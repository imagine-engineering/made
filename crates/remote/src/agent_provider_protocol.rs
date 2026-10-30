use std::{
    collections::BTreeSet,
    fmt, io,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use crate::agent_provider_discovery::{
    AgentProviderCandidate, validate_agent_provider_protocol_version,
};
use futures::{AsyncRead, AsyncReadExt as _, AsyncWriteExt as _};
use serde::Deserialize;
use serde_json::Value;

pub const AGENT_PROVIDER_STDOUT_LIMIT: usize = 1024 * 1024;
pub const AGENT_PROVIDER_STDERR_LIMIT: usize = 64 * 1024;
pub const AGENT_PROVIDER_REQUEST_LIMIT: usize = 1024 * 1024;
pub const AGENT_PROVIDER_INFO_TIMEOUT: Duration = Duration::from_secs(10);
pub const AGENT_PROVIDER_DEPLOY_TIMEOUT: Duration = Duration::from_secs(600);

const PROVIDER_ERROR_LIMIT: usize = 4096;
const PROVIDER_NAME_LIMIT: usize = 128;
const PROVIDER_VERSION_LIMIT: usize = 128;
const PROVIDER_DESCRIPTION_LIMIT: usize = 4096;
const PROVIDER_AGENT_ID_LIMIT: usize = 256;
const PIPE_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const REDACTION_MARKER: &str = "[REDACTED]";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentProviderOperation {
    Info,
    Deploy,
}

impl AgentProviderOperation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Deploy => "deploy",
        }
    }

    pub fn timeout(self) -> Duration {
        match self {
            Self::Info => AGENT_PROVIDER_INFO_TIMEOUT,
            Self::Deploy => AGENT_PROVIDER_DEPLOY_TIMEOUT,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentProviderInfo {
    pub name: String,
    pub version: String,
    pub protocol_version: u32,
    pub description: String,
    pub config_schema: Value,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentProviderDeployment {
    pub agent_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentProviderResponse {
    Info(AgentProviderInfo),
    Deploy(AgentProviderDeployment),
}

#[derive(Clone, Debug, Default)]
pub struct AgentProviderCancellation {
    cancelled: Arc<AtomicBool>,
}

impl AgentProviderCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
pub enum AgentProviderProtocolError {
    InvalidRequest {
        reason: &'static str,
    },
    RequestTooLarge {
        limit: usize,
    },
    Spawn {
        path: PathBuf,
        source: anyhow::Error,
    },
    Stdin {
        source: io::Error,
    },
    Wait {
        source: io::Error,
    },
    Cleanup {
        source: anyhow::Error,
    },
    Cancelled {
        operation: AgentProviderOperation,
    },
    TimedOut {
        operation: AgentProviderOperation,
        timeout: Duration,
    },
    StdoutTooLarge {
        limit: usize,
    },
    StderrTooLarge {
        limit: usize,
    },
    PipeRead {
        stream: &'static str,
        source: io::Error,
    },
    PipeClosed {
        stream: &'static str,
    },
    PipeDrainTimedOut,
    NonZeroExit {
        status: String,
        stderr: String,
    },
    MalformedResponse {
        operation: AgentProviderOperation,
        reason: String,
        stderr: String,
    },
    ProviderRejected {
        error: String,
    },
}

impl fmt::Display for AgentProviderProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { reason } => {
                write!(formatter, "invalid provider request: {reason}")
            }
            Self::RequestTooLarge { limit } => {
                write!(formatter, "provider request exceeds the {limit}-byte limit")
            }
            Self::Spawn { path, source } => {
                write!(
                    formatter,
                    "failed to start provider {}: {source}",
                    path.display()
                )
            }
            Self::Stdin { source } => {
                write!(formatter, "failed to write provider request: {source}")
            }
            Self::Wait { source } => write!(formatter, "failed to wait for provider: {source}"),
            Self::Cleanup { source } => write!(formatter, "failed to stop provider: {source}"),
            Self::Cancelled { operation } => {
                write!(formatter, "provider {} was cancelled", operation.name())
            }
            Self::TimedOut { operation, timeout } => write!(
                formatter,
                "provider {} timed out after {} seconds",
                operation.name(),
                timeout.as_secs_f64()
            ),
            Self::StdoutTooLarge { limit } => {
                write!(formatter, "provider stdout exceeds the {limit}-byte limit")
            }
            Self::StderrTooLarge { limit } => {
                write!(formatter, "provider stderr exceeds the {limit}-byte limit")
            }
            Self::PipeRead { stream, source } => {
                write!(formatter, "failed to read provider {stream}: {source}")
            }
            Self::PipeClosed { stream } => {
                write!(
                    formatter,
                    "provider {stream} reader closed without a result"
                )
            }
            Self::PipeDrainTimedOut => {
                formatter.write_str("provider output pipes remained open after process exit")
            }
            Self::NonZeroExit { status, stderr } if stderr.is_empty() => {
                write!(formatter, "provider failed ({status}, empty stderr)")
            }
            Self::NonZeroExit { status, stderr } => {
                write!(formatter, "provider failed ({status}). stderr: {stderr}")
            }
            Self::MalformedResponse {
                operation,
                reason,
                stderr,
            } if stderr.is_empty() => write!(
                formatter,
                "provider {} returned an invalid response: {reason}",
                operation.name()
            ),
            Self::MalformedResponse {
                operation,
                reason,
                stderr,
            } => write!(
                formatter,
                "provider {} returned an invalid response: {reason}. stderr: {stderr}",
                operation.name()
            ),
            Self::ProviderRejected { error } => {
                write!(formatter, "provider rejected request: {error}")
            }
        }
    }
}

impl std::error::Error for AgentProviderProtocolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Cleanup { source } => Some(source.as_ref()),
            Self::Stdin { source } | Self::Wait { source } | Self::PipeRead { source, .. } => {
                Some(source)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct InvocationLimits {
    stdout: usize,
    stderr: usize,
    timeout: Duration,
    pipe_drain_timeout: Duration,
}

impl InvocationLimits {
    fn for_operation(operation: AgentProviderOperation) -> Self {
        Self {
            stdout: AGENT_PROVIDER_STDOUT_LIMIT,
            stderr: AGENT_PROVIDER_STDERR_LIMIT,
            timeout: operation.timeout(),
            pipe_drain_timeout: PIPE_DRAIN_TIMEOUT,
        }
    }
}

pub async fn invoke_agent_provider(
    candidate: &AgentProviderCandidate,
    work_directory: &Path,
    operation: AgentProviderOperation,
    request: &Value,
    background_executor: &gpui::BackgroundExecutor,
) -> Result<AgentProviderResponse, AgentProviderProtocolError> {
    invoke_agent_provider_cancellable(
        candidate,
        work_directory,
        operation,
        request,
        &AgentProviderCancellation::default(),
        background_executor,
    )
    .await
}

pub async fn invoke_agent_provider_cancellable(
    candidate: &AgentProviderCandidate,
    work_directory: &Path,
    operation: AgentProviderOperation,
    request: &Value,
    cancellation: &AgentProviderCancellation,
    background_executor: &gpui::BackgroundExecutor,
) -> Result<AgentProviderResponse, AgentProviderProtocolError> {
    invoke_agent_provider_with_limits(
        candidate,
        work_directory,
        operation,
        request,
        InvocationLimits::for_operation(operation),
        cancellation,
        background_executor,
    )
    .await
}

async fn invoke_agent_provider_with_limits(
    candidate: &AgentProviderCandidate,
    work_directory: &Path,
    operation: AgentProviderOperation,
    request: &Value,
    limits: InvocationLimits,
    cancellation: &AgentProviderCancellation,
    background_executor: &gpui::BackgroundExecutor,
) -> Result<AgentProviderResponse, AgentProviderProtocolError> {
    if cancellation.is_cancelled() {
        return Err(AgentProviderProtocolError::Cancelled { operation });
    }
    validate_request(operation, request)?;
    let mut request_bytes =
        serde_json::to_vec(request).map_err(|_| AgentProviderProtocolError::InvalidRequest {
            reason: "request is not serializable JSON",
        })?;
    request_bytes.push(b'\n');
    if request_bytes.len() > AGENT_PROVIDER_REQUEST_LIMIT {
        return Err(AgentProviderProtocolError::RequestTooLarge {
            limit: AGENT_PROVIDER_REQUEST_LIMIT,
        });
    }
    let redactor = AgentProviderRedactor::from_request(request);

    let mut command = util::command::new_std_command(&candidate.canonical_path);
    command.current_dir(work_directory);
    let mut child =
        util::process::Child::spawn(command, Stdio::piped(), Stdio::piped(), Stdio::piped())
            .map_err(|source| AgentProviderProtocolError::Spawn {
                path: candidate.canonical_path.clone(),
                source,
            })?;

    let Some(mut stdin) = child.stdin.take() else {
        stop_provider(&mut child).await?;
        return Err(AgentProviderProtocolError::InvalidRequest {
            reason: "provider stdin was not captured",
        });
    };
    if let Err(source) = stdin.write_all(&request_bytes).await {
        stop_provider(&mut child).await?;
        return Err(AgentProviderProtocolError::Stdin { source });
    }
    if let Err(source) = stdin.close().await {
        stop_provider(&mut child).await?;
        return Err(AgentProviderProtocolError::Stdin { source });
    }
    drop(stdin);

    let Some(stdout) = child.stdout.take() else {
        stop_provider(&mut child).await?;
        return Err(AgentProviderProtocolError::PipeClosed { stream: "stdout" });
    };
    let Some(stderr) = child.stderr.take() else {
        stop_provider(&mut child).await?;
        return Err(AgentProviderProtocolError::PipeClosed { stream: "stderr" });
    };

    let (stdout_sender, stdout_receiver) = smol::channel::bounded(1);
    let stdout_task = background_executor.spawn(async move {
        let result = read_bounded(stdout, limits.stdout).await;
        if stdout_sender.send(result).await.is_err() {
            return;
        }
    });
    let (stderr_sender, stderr_receiver) = smol::channel::bounded(1);
    let stderr_task = background_executor.spawn(async move {
        let result = read_bounded(stderr, limits.stderr).await;
        if stderr_sender.send(result).await.is_err() {
            return;
        }
    });

    let deadline = background_executor.now() + limits.timeout;
    let mut pipe_deadline = None;
    let mut exit_status = None;
    let mut stdout_bytes = None;
    let mut stderr_bytes = None;

    loop {
        if let Err(error) =
            receive_pipe("stdout", &stdout_receiver, &mut stdout_bytes, limits.stdout)
        {
            stop_provider(&mut child).await?;
            return Err(error);
        }
        if let Err(error) =
            receive_pipe("stderr", &stderr_receiver, &mut stderr_bytes, limits.stderr)
        {
            stop_provider(&mut child).await?;
            return Err(error);
        }

        if exit_status.is_none() {
            exit_status = match child.try_status() {
                Ok(status) => status,
                Err(source) => {
                    child
                        .kill()
                        .map_err(|source| AgentProviderProtocolError::Cleanup { source })?;
                    return Err(AgentProviderProtocolError::Wait { source });
                }
            };
            if exit_status.is_some() {
                pipe_deadline = Some(background_executor.now() + limits.pipe_drain_timeout);
            }
        }

        if let (Some(exit_status), Some(stdout), Some(stderr)) =
            (exit_status, stdout_bytes.as_ref(), stderr_bytes.as_ref())
        {
            child
                .kill()
                .map_err(|source| AgentProviderProtocolError::Cleanup { source })?;
            drop(stdout_task);
            drop(stderr_task);
            return parse_agent_provider_response(
                candidate,
                operation,
                exit_status,
                stdout,
                stderr,
                &redactor,
            );
        }

        if cancellation.is_cancelled() {
            stop_provider(&mut child).await?;
            return Err(AgentProviderProtocolError::Cancelled { operation });
        }

        let now = background_executor.now();
        if now >= deadline {
            stop_provider(&mut child).await?;
            return Err(AgentProviderProtocolError::TimedOut {
                operation,
                timeout: limits.timeout,
            });
        }
        if pipe_deadline.is_some_and(|pipe_deadline| now >= pipe_deadline) {
            stop_provider(&mut child).await?;
            return Err(AgentProviderProtocolError::PipeDrainTimedOut);
        }
        background_executor.timer(PROCESS_POLL_INTERVAL).await;
    }
}

fn validate_request(
    operation: AgentProviderOperation,
    request: &Value,
) -> Result<(), AgentProviderProtocolError> {
    let Some(request) = request.as_object() else {
        return Err(AgentProviderProtocolError::InvalidRequest {
            reason: "request must be an object",
        });
    };
    if request.get("op").and_then(Value::as_str) != Some(operation.name()) {
        return Err(AgentProviderProtocolError::InvalidRequest {
            reason: "request operation does not match invocation operation",
        });
    }
    Ok(())
}

async fn stop_provider(child: &mut util::process::Child) -> Result<(), AgentProviderProtocolError> {
    let status = child
        .try_status()
        .map_err(anyhow::Error::from)
        .map_err(|source| AgentProviderProtocolError::Cleanup { source })?;
    child
        .kill()
        .map_err(|source| AgentProviderProtocolError::Cleanup { source })?;
    if status.is_none() {
        child
            .status()
            .await
            .map_err(anyhow::Error::from)
            .map_err(|source| AgentProviderProtocolError::Cleanup { source })?;
    }
    Ok(())
}

enum PipeFailure {
    TooLarge,
    Read(io::Error),
}

async fn read_bounded(
    mut reader: impl AsyncRead + Unpin,
    limit: usize,
) -> Result<Vec<u8>, PipeFailure> {
    let mut output = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0_u8; 8192];
    loop {
        let count = reader.read(&mut buffer).await.map_err(PipeFailure::Read)?;
        if count == 0 {
            return Ok(output);
        }
        let remaining = limit.saturating_sub(output.len());
        if count > remaining {
            return Err(PipeFailure::TooLarge);
        }
        output.extend_from_slice(&buffer[..count]);
    }
}

fn receive_pipe(
    stream: &'static str,
    receiver: &smol::channel::Receiver<Result<Vec<u8>, PipeFailure>>,
    output: &mut Option<Vec<u8>>,
    limit: usize,
) -> Result<(), AgentProviderProtocolError> {
    if output.is_some() {
        return Ok(());
    }
    match receiver.try_recv() {
        Ok(Ok(bytes)) => *output = Some(bytes),
        Ok(Err(PipeFailure::TooLarge)) if stream == "stdout" => {
            return Err(AgentProviderProtocolError::StdoutTooLarge { limit });
        }
        Ok(Err(PipeFailure::TooLarge)) => {
            return Err(AgentProviderProtocolError::StderrTooLarge { limit });
        }
        Ok(Err(PipeFailure::Read(source))) => {
            return Err(AgentProviderProtocolError::PipeRead { stream, source });
        }
        Err(smol::channel::TryRecvError::Closed) => {
            return Err(AgentProviderProtocolError::PipeClosed { stream });
        }
        Err(smol::channel::TryRecvError::Empty) => {}
    }
    Ok(())
}

fn parse_agent_provider_response(
    candidate: &AgentProviderCandidate,
    operation: AgentProviderOperation,
    exit_status: ExitStatus,
    stdout: &[u8],
    stderr: &[u8],
    redactor: &AgentProviderRedactor,
) -> Result<AgentProviderResponse, AgentProviderProtocolError> {
    let stderr = diagnostic_snippet(&redactor.redact(&String::from_utf8_lossy(stderr)));
    if !exit_status.success() {
        let status = exit_status
            .code()
            .map(|code| format!("exit code {code}"))
            .unwrap_or_else(|| "killed by signal".to_owned());
        return Err(AgentProviderProtocolError::NonZeroExit { status, stderr });
    }

    let mut value: Value = serde_json::from_slice(stdout).map_err(|error| {
        AgentProviderProtocolError::MalformedResponse {
            operation,
            reason: error.to_string(),
            stderr: stderr.clone(),
        }
    })?;
    redactor.redact_value(&mut value);
    let object =
        value
            .as_object()
            .ok_or_else(|| AgentProviderProtocolError::MalformedResponse {
                operation,
                reason: "response must be a JSON object".to_owned(),
                stderr: stderr.clone(),
            })?;
    match object.get("ok").and_then(Value::as_bool) {
        Some(false) => {
            let failure: FailureResponse =
                deserialize_response(value, operation, &stderr, redactor)?;
            validate_bounded_text(
                "error",
                &failure.error,
                PROVIDER_ERROR_LIMIT,
                operation,
                &stderr,
            )?;
            Err(AgentProviderProtocolError::ProviderRejected {
                error: failure.error,
            })
        }
        Some(true) => match operation {
            AgentProviderOperation::Info => {
                let response: InfoResponse =
                    deserialize_response(value, operation, &stderr, redactor)?;
                validate_bounded_text(
                    "name",
                    &response.name,
                    PROVIDER_NAME_LIMIT,
                    operation,
                    &stderr,
                )?;
                validate_bounded_text(
                    "version",
                    &response.version,
                    PROVIDER_VERSION_LIMIT,
                    operation,
                    &stderr,
                )?;
                validate_bounded_text(
                    "description",
                    &response.description,
                    PROVIDER_DESCRIPTION_LIMIT,
                    operation,
                    &stderr,
                )?;
                if !response.config_schema.is_object() {
                    return Err(AgentProviderProtocolError::MalformedResponse {
                        operation,
                        reason: "config_schema must be an object".to_owned(),
                        stderr,
                    });
                }
                validate_agent_provider_protocol_version(
                    candidate,
                    Some(response.protocol_version),
                )
                .map_err(|error| {
                    AgentProviderProtocolError::MalformedResponse {
                        operation,
                        reason: error.to_string(),
                        stderr: stderr.clone(),
                    }
                })?;
                Ok(AgentProviderResponse::Info(AgentProviderInfo {
                    name: response.name,
                    version: response.version,
                    protocol_version: response.protocol_version,
                    description: response.description,
                    config_schema: response.config_schema,
                }))
            }
            AgentProviderOperation::Deploy => {
                let response: DeployResponse =
                    deserialize_response(value, operation, &stderr, redactor)?;
                validate_bounded_text(
                    "agent_id",
                    &response.agent_id,
                    PROVIDER_AGENT_ID_LIMIT,
                    operation,
                    &stderr,
                )?;
                Ok(AgentProviderResponse::Deploy(AgentProviderDeployment {
                    agent_id: response.agent_id,
                }))
            }
        },
        _ => Err(AgentProviderProtocolError::MalformedResponse {
            operation,
            reason: "response must contain boolean ok".to_owned(),
            stderr,
        }),
    }
}

fn deserialize_response<T: for<'de> Deserialize<'de>>(
    value: Value,
    operation: AgentProviderOperation,
    stderr: &str,
    redactor: &AgentProviderRedactor,
) -> Result<T, AgentProviderProtocolError> {
    serde_json::from_value(value).map_err(|error| AgentProviderProtocolError::MalformedResponse {
        operation,
        reason: redactor.redact(&error.to_string()),
        stderr: stderr.to_owned(),
    })
}

fn validate_bounded_text(
    field: &'static str,
    value: &str,
    limit: usize,
    operation: AgentProviderOperation,
    stderr: &str,
) -> Result<(), AgentProviderProtocolError> {
    if value.is_empty() {
        return Err(AgentProviderProtocolError::MalformedResponse {
            operation,
            reason: format!("{field} must not be empty"),
            stderr: stderr.to_owned(),
        });
    }
    if value.len() > limit {
        return Err(AgentProviderProtocolError::MalformedResponse {
            operation,
            reason: format!("{field} exceeds the {limit}-byte limit"),
            stderr: stderr.to_owned(),
        });
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FailureResponse {
    #[serde(rename = "ok")]
    _ok: bool,
    error: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InfoResponse {
    #[serde(rename = "ok")]
    _ok: bool,
    name: String,
    version: String,
    protocol_version: u32,
    description: String,
    config_schema: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeployResponse {
    #[serde(rename = "ok")]
    _ok: bool,
    agent_id: String,
}

#[derive(Clone, Debug)]
struct AgentProviderRedactor {
    secrets: Vec<String>,
}

impl AgentProviderRedactor {
    fn from_request(request: &Value) -> Self {
        let mut secrets = BTreeSet::new();
        for pointer in ["/agent/private_key_nsec", "/agent/auth_tag"] {
            if let Some(value) = request.pointer(pointer).and_then(Value::as_str)
                && value.len() >= 4
            {
                secrets.insert(value.to_owned());
            }
        }
        for pointer in [
            "/agent/env_vars",
            "/agent/launch/env",
            "/agent/launch/policy_env",
        ] {
            let Some(values) = request.pointer(pointer).and_then(Value::as_object) else {
                continue;
            };
            for value in values.values().filter_map(Value::as_str) {
                if value.len() >= 4 {
                    secrets.insert(value.to_owned());
                }
            }
        }
        let mut secrets = secrets.into_iter().collect::<Vec<_>>();
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        Self { secrets }
    }

    fn redact(&self, input: &str) -> String {
        let mut output = input.to_owned();
        for secret in &self.secrets {
            output = output.replace(secret, REDACTION_MARKER);
        }
        for prefix in ["nsec1", "sprt_tok_"] {
            output = redact_prefixed_tokens(output, prefix);
        }
        output
    }

    fn redact_value(&self, value: &mut Value) {
        match value {
            Value::String(text) => *text = self.redact(text),
            Value::Array(values) => {
                for value in values {
                    self.redact_value(value);
                }
            }
            Value::Object(values) => {
                for value in values.values_mut() {
                    self.redact_value(value);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }
}

fn redact_prefixed_tokens(mut input: String, prefix: &str) -> String {
    let mut search_start = 0;
    while let Some(relative_start) = input[search_start..].find(prefix) {
        let start = search_start + relative_start;
        let end = input[start..]
            .char_indices()
            .find_map(|(offset, character)| {
                (!is_token_character(character)).then_some(start + offset)
            })
            .unwrap_or(input.len());
        input.replace_range(start..end, REDACTION_MARKER);
        search_start = start + REDACTION_MARKER.len();
    }
    input
}

fn is_token_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
}

fn diagnostic_snippet(input: &str) -> String {
    if input.len() <= PROVIDER_ERROR_LIMIT {
        return input.to_owned();
    }
    let mut end = PROVIDER_ERROR_LIMIT;
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    input[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[cfg(unix)]
    use std::os::unix::{fs::PermissionsExt as _, process::ExitStatusExt as _};
    #[cfg(windows)]
    use std::os::windows::process::ExitStatusExt as _;

    use serde_json::json;

    use super::*;
    use crate::agent_provider_discovery::AgentProviderTrust;

    fn candidate(path: &Path) -> AgentProviderCandidate {
        AgentProviderCandidate {
            provider_id: "fixture".to_owned(),
            discovered_path: path.to_owned(),
            canonical_path: path.canonicalize().expect("canonical fixture path"),
            trust: AgentProviderTrust::Untrusted,
        }
    }

    #[cfg(unix)]
    fn provider_script(contents: &str) -> (tempfile::TempDir, AgentProviderCandidate) {
        let directory = tempfile::tempdir().expect("create provider fixture directory");
        let path = directory.path().join("buzz-backend-fixture");
        fs::write(&path, contents).expect("write provider fixture");
        let mut permissions = fs::metadata(&path)
            .expect("read provider fixture metadata")
            .permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&path, permissions).expect("mark provider fixture executable");
        let candidate = candidate(&path);
        (directory, candidate)
    }

    fn success_status() -> ExitStatus {
        ExitStatus::from_raw(0)
    }

    #[test]
    fn strict_parser_accepts_info_and_rejects_malformed_corpus() {
        let directory = tempfile::tempdir().expect("create candidate fixture");
        let path = directory.path().join("provider");
        fs::write(&path, b"fixture").expect("write candidate fixture");
        let candidate = candidate(&path);
        let redactor = AgentProviderRedactor {
            secrets: Vec::new(),
        };
        let valid = br#"{"ok":true,"name":"kubernetes","version":"1.0.0","protocol_version":1,"description":"pods","config_schema":{}}"#;
        assert!(matches!(
            parse_agent_provider_response(
                &candidate,
                AgentProviderOperation::Info,
                success_status(),
                valid,
                b"",
                &redactor,
            ),
            Ok(AgentProviderResponse::Info(AgentProviderInfo {
                protocol_version: 1,
                ..
            }))
        ));

        let malformed = [
            b"".as_slice(),
            b"null",
            b"[]",
            br#"{}"#,
            br#"{"ok":"true"}"#,
            br#"{"ok":true}"#,
            br#"{"ok":true,"name":"x","version":"1","description":"x","config_schema":{}}"#,
            br#"{"ok":true,"name":"x","version":"1","protocol_version":2,"description":"x","config_schema":{}}"#,
            br#"{"ok":true,"name":"x","version":"1","protocol_version":1,"description":"x","config_schema":{},"extra":true}"#,
            br#"{"ok":true,"name":"x","version":"1","protocol_version":1,"description":"x","config_schema":{}} trailing"#,
        ];
        for bytes in malformed {
            assert!(matches!(
                parse_agent_provider_response(
                    &candidate,
                    AgentProviderOperation::Info,
                    success_status(),
                    bytes,
                    b"",
                    &redactor,
                ),
                Err(AgentProviderProtocolError::MalformedResponse { .. })
            ));
        }
    }

    #[test]
    fn provider_strings_and_diagnostics_are_redacted_before_return() {
        let directory = tempfile::tempdir().expect("create candidate fixture");
        let path = directory.path().join("provider");
        fs::write(&path, b"fixture").expect("write candidate fixture");
        let candidate = candidate(&path);
        let request = json!({
            "op": "deploy",
            "agent": {
                "private_key_nsec": "raw-identity-secret",
                "auth_tag": "authorization-secret",
                "env_vars": {"API_TOKEN": "long-env-secret"},
                "launch": {
                    "env": {"OTHER": "overlapping-secret-value"},
                    "policy_env": {"SHORT": "abc"}
                }
            }
        });
        let redactor = AgentProviderRedactor::from_request(&request);
        let stdout = br#"{"ok":false,"error":"raw-identity-secret authorization-secret long-env-secret nsec1private sprt_tok_hidden overlapping-secret-value"}"#;
        let error = parse_agent_provider_response(
            &candidate,
            AgentProviderOperation::Deploy,
            success_status(),
            stdout,
            b"stderr long-env-secret nsec1stderr",
            &redactor,
        )
        .expect_err("provider failure should remain a failure");
        let diagnostic = error.to_string();
        assert!(!diagnostic.contains("raw-identity-secret"));
        assert!(!diagnostic.contains("authorization-secret"));
        assert!(!diagnostic.contains("long-env-secret"));
        assert!(!diagnostic.contains("nsec1private"));
        assert!(!diagnostic.contains("sprt_tok_hidden"));
        assert!(!diagnostic.contains("overlapping-secret-value"));
        assert!(diagnostic.contains(REDACTION_MARKER));

        let error = parse_agent_provider_response(
            &candidate,
            AgentProviderOperation::Deploy,
            success_status(),
            br#"{"ok":true,"agent_id":"safe","long-env-secret":"nsec1unknown"}"#,
            b"",
            &redactor,
        )
        .expect_err("unknown secret-bearing field should fail");
        let diagnostic = error.to_string();
        assert!(!diagnostic.contains("long-env-secret"));
        assert!(!diagnostic.contains("nsec1unknown"));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn invocation_accepts_one_bounded_response(
        background_executor: gpui::BackgroundExecutor,
    ) {
        background_executor.allow_parking();
        let (directory, candidate) = provider_script(
            "#!/bin/sh\nread request\nprintf '%s' '{\"ok\":true,\"name\":\"fixture\",\"version\":\"1\",\"protocol_version\":1,\"description\":\"fixture provider\",\"config_schema\":{}}'\n",
        );
        let response = invoke_agent_provider_with_limits(
            &candidate,
            directory.path(),
            AgentProviderOperation::Info,
            &json!({"op": "info", "request_id": "fixture"}),
            InvocationLimits {
                timeout: Duration::from_secs(1),
                pipe_drain_timeout: Duration::from_millis(100),
                ..InvocationLimits::for_operation(AgentProviderOperation::Info)
            },
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect("bounded provider should succeed");
        assert!(matches!(response, AgentProviderResponse::Info(_)));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn invocation_rejects_oversized_output_and_nonzero_partial_json(
        background_executor: gpui::BackgroundExecutor,
    ) {
        background_executor.allow_parking();
        let (directory, candidate) = provider_script(
            "#!/bin/sh\nread request\nprintf '%s' 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'\n",
        );
        let error = invoke_agent_provider_with_limits(
            &candidate,
            directory.path(),
            AgentProviderOperation::Info,
            &json!({"op": "info"}),
            InvocationLimits {
                stdout: 32,
                stderr: 32,
                timeout: Duration::from_secs(1),
                pipe_drain_timeout: Duration::from_millis(100),
            },
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect_err("oversized stdout should fail");
        assert!(matches!(
            error,
            AgentProviderProtocolError::StdoutTooLarge { limit: 32 }
        ));

        let (directory, candidate) = provider_script(
            "#!/bin/sh\nread request\nprintf '%s' '{\"ok\":true,\"agent_id\":\"unsafe\"}'\nprintf '%s' 'nsec1must_not_escape' >&2\nexit 7\n",
        );
        let error = invoke_agent_provider_with_limits(
            &candidate,
            directory.path(),
            AgentProviderOperation::Deploy,
            &json!({"op": "deploy", "agent": {}}),
            InvocationLimits {
                timeout: Duration::from_secs(1),
                pipe_drain_timeout: Duration::from_millis(100),
                ..InvocationLimits::for_operation(AgentProviderOperation::Deploy)
            },
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect_err("nonzero exit should override valid partial JSON");
        assert!(matches!(
            error,
            AgentProviderProtocolError::NonZeroExit { .. }
        ));
        assert!(!error.to_string().contains("nsec1must_not_escape"));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn invocation_kills_a_hanging_provider_at_the_deadline(
        background_executor: gpui::BackgroundExecutor,
    ) {
        background_executor.allow_parking();
        let (directory, candidate) =
            provider_script("#!/bin/sh\nread request\nwhile :; do :; done\n");
        let started = std::time::Instant::now();
        let error = invoke_agent_provider_with_limits(
            &candidate,
            directory.path(),
            AgentProviderOperation::Info,
            &json!({"op": "info"}),
            InvocationLimits {
                stdout: 1024,
                stderr: 1024,
                timeout: Duration::from_millis(100),
                pipe_drain_timeout: Duration::from_millis(50),
            },
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect_err("hanging provider should time out");
        assert!(matches!(error, AgentProviderProtocolError::TimedOut { .. }));
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
