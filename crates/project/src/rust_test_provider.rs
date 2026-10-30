use feature_flags::{FeatureFlagAppExt as _, RustExactDoctestActionsExperimentalFeatureFlag};
#[cfg(any(test, feature = "test-support"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    ffi::OsString,
    path::{Path, PathBuf},
    process::Output,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail};
use cargo_metadata::{Message, TargetKind};
use collections::HashMap;
use futures::{
    FutureExt as _, StreamExt as _,
    channel::mpsc,
    future::{AbortHandle, Abortable, BoxFuture, Shared, WeakShared},
    io::{AsyncRead, AsyncReadExt as _},
};
use gpui::{
    App, AsyncApp, BackgroundExecutor, Context, Entity, EventEmitter, FutureExt as _, Subscription,
    Task, WeakEntity,
};
use language::Buffer;
use lsp::LanguageServerId;
use rpc::{AnyProtoClient, ErrorCodeExt as _, ErrorExt as _, TypedEnvelope, proto};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use task::{
    BuildTaskDefinition, DebugScenario, SaveStrategy, StructuredTaskHandle,
    StructuredTaskLifecycleEvent, StructuredTaskState, StructuredTerminalId, TaskId, TaskTemplate,
    VariableName,
};

use crate::{
    ProjectEnvironment, ProjectPath, WorktreeId,
    cargo_workspace::{
        CargoMessageDecoder, CargoPackageModel, CargoSnapshotCompleteness, CargoTargetKind,
        CargoWorkspaceErrorCategory, CargoWorkspaceModel, CargoWorkspaceSnapshot,
    },
    cargo_workspace_store::{CargoWorkspaceStore, prevent_passive_rust_downloads},
    lsp_store::{
        LanguageServerToQuery, LspStore, LspStoreEvent,
        lsp_ext_command::{GetLspRunnables, RustSemanticRunnable},
        rust_analyzer_ext::RUST_ANALYZER_NAME,
    },
    structured_execution::{
        DiscoveryGeneration, StructuredDebugCapability, StructuredExecutionStore,
        StructuredExecutionStoreEvent, StructuredNode, StructuredNodeId, StructuredNodeKind,
        StructuredProviderId, StructuredProviderSnapshot, StructuredProviderStatus,
        StructuredRunId,
    },
    task_store::PendingExperimentalFeatureOperation,
    trusted_worktrees::{PathTrust, TrustedWorktrees, TrustedWorktreesEvent},
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};

pub const RUST_TEST_PROTOCOL_VERSION: u32 = 1;
pub const MAX_RUST_TEST_CARGO_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_RUST_TEST_LISTING_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_RUST_TEST_LINES: usize = 50_000;
pub const MAX_RUST_TEST_HARNESSES: usize = 2_000;
pub const MAX_RUST_TEST_CASES: usize = 10_000;
pub const MAX_RUST_TEST_FIELD_BYTES: usize = 1_024;
pub const MAX_RUST_TEST_DIAGNOSTICS: usize = 64;
pub const RUST_TEST_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(20);
pub const RUST_TEST_PROVIDER_ID: &str = "rust-tests";
pub const RUST_TEST_DISCOVERY_PROTOCOL_VERSION: u32 = 2;
pub const MAX_RUST_TEST_REQUEST_WORKTREES: usize = 256;
pub const MAX_PENDING_RUST_TEST_CANCELLATIONS: usize = 256;
pub const MAX_RUST_TEST_RERUNS: usize = 4;
pub const MAX_RUST_TEST_ACTION_PLAN_BYTES: usize = 64 * 1024;
pub const MAX_RUST_TEST_ACTION_ARGUMENTS: usize = 256;
pub const MAX_PENDING_REMOTE_RUST_TEST_RUNS: usize = 256;
const MAX_RUST_TEST_SEMANTIC_BUFFERS: usize = 64;
const MAX_RUST_TEST_SEMANTIC_RUNNABLES: usize = 32;
const MAX_RUST_TEST_SEMANTIC_ARGUMENT_BYTES: usize = 4 * 1024;
const MAX_RUST_TEST_SEMANTIC_REFRESH_BUFFERS: usize = 8;
const REMOTE_RUST_TEST_SPAWN_FAILURE: &str =
    "Rust test task failed to start; details remain on the originating client";
#[cfg(any(test, feature = "test-support"))]
static LOCAL_RUST_TEST_DISCOVERY_CALLS: AtomicU64 = AtomicU64::new(0);

#[cfg(any(test, feature = "test-support"))]
pub fn local_rust_test_discovery_calls_for_test() -> u64 {
    LOCAL_RUST_TEST_DISCOVERY_CALLS.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustTestRemoteErrorKind {
    UnsupportedDiscoveryHost,
    UnsupportedActionHost,
    SupersededDiscovery,
}

#[derive(Debug)]
pub struct RustTestRemoteError {
    pub kind: RustTestRemoteErrorKind,
}

impl std::fmt::Display for RustTestRemoteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            RustTestRemoteErrorKind::UnsupportedDiscoveryHost => {
                formatter.write_str("The project host does not support Rust test discovery")
            }
            RustTestRemoteErrorKind::UnsupportedActionHost => {
                formatter.write_str("The project host does not support Rust test actions")
            }
            RustTestRemoteErrorKind::SupersededDiscovery => {
                formatter.write_str("stale Rust test discovery generation")
            }
        }
    }
}

impl std::error::Error for RustTestRemoteError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustTestAction {
    Run,
    Debug,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RustTestActionPlan {
    Task {
        run_id: StructuredRunId,
        discovery_generation: DiscoveryGeneration,
        scope_node_ids: Vec<StructuredNodeId>,
        worktree_id: WorktreeId,
        template: TaskTemplate,
    },
    Debug {
        worktree_id: WorktreeId,
        scenario: DebugScenario,
    },
}

impl RustTestActionPlan {
    pub fn required_experimental_features(&self) -> &[String] {
        match self {
            Self::Task { template, .. } => &template.required_experimental_features,
            Self::Debug { scenario, .. } => match &scenario.build {
                Some(BuildTaskDefinition::Template { task_template, .. }) => {
                    &task_template.required_experimental_features
                }
                _ => &[],
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum RustTestActionWirePlan {
    Task {
        run_id: String,
        discovery_generation: u64,
        scope_node_ids: Vec<String>,
        worktree_id: u64,
        template: TaskTemplate,
    },
    Debug {
        worktree_id: u64,
        build_template: TaskTemplate,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RustTestActionScope {
    Workspace,
    Package,
    Target,
    Group,
    Case,
}

#[derive(Clone, Debug)]
struct RustTestActionDescriptor {
    scope: RustTestActionScope,
    worktree_id: WorktreeId,
    manifest_path: ProjectPath,
    label: String,
    cargo_args: Vec<String>,
    executable_args: Vec<String>,
    target_kind: Option<RustTestTargetKind>,
    doctest_case_names: Option<Arc<[String]>>,
    doctest_ignored_run_supported: bool,
}

struct PendingRemoteRustAction {
    abort: AbortHandle,
    worktree_ids: HashSet<WorktreeId>,
}

#[derive(Clone, Debug)]
struct AuthorizedRemoteRun {
    discovery_generation: DiscoveryGeneration,
    scope_node_ids: Vec<StructuredNodeId>,
    worktree_id: WorktreeId,
    authorization_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RustTestTargetKind {
    Unit,
    Integration,
    Binary,
    Example,
    Benchmark,
    Doctest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RustTestCaseKind {
    Test,
    Benchmark,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RustTestListingMode {
    All,
    Ignored,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RustTestHarnessLocator {
    Executable(PathBuf),
    CargoDoctest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestHarness {
    pub id: String,
    pub package_id: String,
    pub target_name: String,
    pub target_kind: RustTestTargetKind,
    pub locator: RustTestHarnessLocator,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestRunnableSelector {
    pub cargo_args: Vec<String>,
    pub executable_args: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestCaseRecord {
    pub id: String,
    pub harness_id: String,
    pub name: String,
    pub kind: RustTestCaseKind,
    pub ignored: bool,
    pub source: Option<ProjectPath>,
    pub runnable: RustTestRunnableSelector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestProtocolDiagnostic {
    pub stage: RustTestProtocolStage,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustTestProtocolStage {
    CargoMessages,
    HarnessListing,
    Enrichment,
    Limits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustTestProtocolCapability {
    Supported,
    Partial,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestProtocolSnapshot {
    pub protocol_version: u32,
    pub toolchain: String,
    pub capability: RustTestProtocolCapability,
    pub harnesses: Vec<RustTestHarness>,
    pub cases: Vec<RustTestCaseRecord>,
    pub diagnostics: Vec<RustTestProtocolDiagnostic>,
    pub truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct RustTestListingCapture {
    pub package_id: String,
    pub target_name: String,
    pub target_kind: RustTestTargetKind,
    pub mode: RustTestListingMode,
    pub stdout: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestDiscoveryCapture {
    pub toolchain: String,
    pub cargo_messages: String,
    pub listings: Vec<RustTestListingCapture>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestSourceHint {
    pub package_id: String,
    pub target_name: String,
    pub target_kind: RustTestTargetKind,
    pub test_name: String,
    pub source: Option<ProjectPath>,
    pub runnable: Option<RustTestRunnableSelector>,
}

#[derive(Clone)]
struct RustSemanticBufferHints {
    buffer: WeakEntity<Buffer>,
    origin: ProjectPath,
    version: clock::Global,
    server_id: LanguageServerId,
    runnables: Vec<RustSemanticRunnable>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustTestDiscoveryRequest {
    pub workspace_root: ProjectPath,
    pub generation: u64,
    pub limits: RustTestProtocolLimits,
    pub manifest_path: Option<PathBuf>,
    pub working_directory: Option<PathBuf>,
    pub environment: Option<HashMap<String, String>>,
    pub doctest_targets: Vec<(String, String)>,
}

pub trait RustTestDiscoveryRunner: Send + Sync {
    fn discover(
        &self,
        request: RustTestDiscoveryRequest,
    ) -> BoxFuture<'static, Result<RustTestDiscoveryCapture>>;
}

struct ProcessRustTestDiscoveryRunner {
    executor: BackgroundExecutor,
}

#[derive(Clone, Copy, Debug)]
enum RustTestDiscoveryStage {
    Cargo,
    Harness,
    Doctest,
    Unknown,
}

impl RustTestDiscoveryStage {
    fn label(self) -> &'static str {
        match self {
            Self::Cargo => "Cargo test discovery",
            Self::Harness => "Rust test harness listing",
            Self::Doctest => "Rust doctest listing",
            Self::Unknown => "Rust test discovery",
        }
    }
}

#[derive(Clone, Debug)]
enum RustTestDiscoveryFailureKind {
    PassivePolicy,
    Start(std::io::ErrorKind),
    MissingPipe(&'static str),
    Read(&'static str, std::io::ErrorKind),
    Wait(std::io::ErrorKind),
    OutputLimit(&'static str),
    Timeout,
    InvalidUtf8,
    Exited(std::process::ExitStatus),
    Unknown,
}

#[derive(Clone, Debug)]
struct RustTestDiscoveryFailure {
    stage: RustTestDiscoveryStage,
    kind: RustTestDiscoveryFailureKind,
}

impl RustTestDiscoveryFailure {
    fn unknown() -> Self {
        Self {
            stage: RustTestDiscoveryStage::Unknown,
            kind: RustTestDiscoveryFailureKind::Unknown,
        }
    }

    fn from_error(error: &anyhow::Error) -> Self {
        error
            .downcast_ref::<Self>()
            .cloned()
            .unwrap_or_else(Self::unknown)
    }
}

impl std::fmt::Display for RustTestDiscoveryFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} ", self.stage.label())?;
        match &self.kind {
            RustTestDiscoveryFailureKind::PassivePolicy => formatter.write_str("cannot safely use the configured Rustup proxy without toolchain installation; update Rustup manually or use installed compiler binaries"),
            RustTestDiscoveryFailureKind::Start(kind) => write!(formatter, "could not start ({kind})"),
            RustTestDiscoveryFailureKind::MissingPipe(stream) => write!(formatter, "has no {stream} pipe"),
            RustTestDiscoveryFailureKind::Read(stream, kind) => write!(formatter, "could not read {stream} ({kind})"),
            RustTestDiscoveryFailureKind::Wait(kind) => write!(formatter, "could not wait for the command ({kind})"),
            RustTestDiscoveryFailureKind::OutputLimit(stream) => write!(formatter, "{stream} exceeded the output limit"),
            RustTestDiscoveryFailureKind::Timeout => formatter.write_str("timed out"),
            RustTestDiscoveryFailureKind::InvalidUtf8 => formatter.write_str("emitted non-UTF-8 output"),
            RustTestDiscoveryFailureKind::Exited(status) => write!(formatter, "exited with {status}"),
            RustTestDiscoveryFailureKind::Unknown => formatter.write_str("failed (unclassified runner error); refresh Tests or check the configured toolchain"),
        }
    }
}

impl std::error::Error for RustTestDiscoveryFailure {}

fn cargo_discovery_failure_message(category: CargoWorkspaceErrorCategory) -> &'static str {
    match category {
        CargoWorkspaceErrorCategory::Restricted => "Trust the project to discover Rust tests",
        CargoWorkspaceErrorCategory::CargoNotFound => {
            "Cargo metadata discovery could not find Cargo"
        }
        CargoWorkspaceErrorCategory::CargoFailed => "Cargo metadata command failed",
        CargoWorkspaceErrorCategory::InvalidMetadata => "Cargo metadata output was invalid",
        CargoWorkspaceErrorCategory::UnsupportedMetadata => "Cargo metadata format is unsupported",
        CargoWorkspaceErrorCategory::Disconnected => "Cargo metadata host disconnected",
        CargoWorkspaceErrorCategory::Cancelled => "Cargo metadata discovery was cancelled",
        CargoWorkspaceErrorCategory::Internal => "Cargo metadata discovery failed (internal error)",
    }
}

impl RustTestDiscoveryRunner for ProcessRustTestDiscoveryRunner {
    fn discover(
        &self,
        request: RustTestDiscoveryRequest,
    ) -> BoxFuture<'static, Result<RustTestDiscoveryCapture>> {
        #[cfg(any(test, feature = "test-support"))]
        LOCAL_RUST_TEST_DISCOVERY_CALLS.fetch_add(1, Ordering::Relaxed);
        let executor = self.executor.clone();
        Box::pin(async move {
            let manifest_path = request
                .manifest_path
                .context("Rust test discovery requires a host manifest path")?;
            let working_directory = request
                .working_directory
                .context("Rust test discovery requires a host working directory")?;
            let mut cargo = util::command::new_command("cargo");
            cargo.current_dir(&working_directory).kill_on_drop(true);
            let mut arguments = [
                "test",
                "--workspace",
                "--all-targets",
                "--no-run",
                "--offline",
                "--message-format=json",
                "--manifest-path",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>();
            arguments.push(manifest_path.clone().into_os_string());
            let output = bounded_command_output(
                cargo,
                arguments,
                &working_directory,
                request.environment.as_ref(),
                request.limits.timeout,
                request.limits.cargo_bytes,
                request.limits.cargo_bytes,
                RustTestDiscoveryStage::Cargo,
                &executor,
            )
            .await?;
            if !output.status.success() {
                return Err(RustTestDiscoveryFailure {
                    stage: RustTestDiscoveryStage::Cargo,
                    kind: RustTestDiscoveryFailureKind::Exited(output.status),
                }
                .into());
            }
            let cargo_messages =
                String::from_utf8(output.stdout).map_err(|_| RustTestDiscoveryFailure {
                    stage: RustTestDiscoveryStage::Cargo,
                    kind: RustTestDiscoveryFailureKind::InvalidUtf8,
                })?;
            let adapter = RustTestProtocolAdapter::new(request.limits);
            let harness_snapshot = adapter.adapt(
                RustTestDiscoveryCapture {
                    toolchain: String::new(),
                    cargo_messages: cargo_messages.clone(),
                    listings: Vec::new(),
                },
                &[],
            );
            let mut listings = Vec::new();
            for harness in harness_snapshot.harnesses {
                let RustTestHarnessLocator::Executable(executable) = harness.locator else {
                    continue;
                };
                for mode in [RustTestListingMode::All, RustTestListingMode::Ignored] {
                    let mut command = util::command::new_command(&executable);
                    let mut arguments = ["--list", "--format", "terse"]
                        .into_iter()
                        .map(OsString::from)
                        .collect::<Vec<_>>();
                    if mode == RustTestListingMode::Ignored {
                        arguments.push("--ignored".into());
                    }
                    command.current_dir(&working_directory).kill_on_drop(true);
                    let output = bounded_command_output(
                        command,
                        arguments,
                        &working_directory,
                        request.environment.as_ref(),
                        request.limits.timeout,
                        request.limits.listing_bytes,
                        request.limits.cargo_bytes,
                        RustTestDiscoveryStage::Harness,
                        &executor,
                    )
                    .await?;
                    if !output.status.success() {
                        return Err(RustTestDiscoveryFailure {
                            stage: RustTestDiscoveryStage::Harness,
                            kind: RustTestDiscoveryFailureKind::Exited(output.status),
                        }
                        .into());
                    }
                    listings.push(RustTestListingCapture {
                        package_id: harness.package_id.clone(),
                        target_name: harness.target_name.clone(),
                        target_kind: harness.target_kind,
                        mode,
                        stdout: String::from_utf8(output.stdout).map_err(|_| {
                            RustTestDiscoveryFailure {
                                stage: RustTestDiscoveryStage::Harness,
                                kind: RustTestDiscoveryFailureKind::InvalidUtf8,
                            }
                        })?,
                    });
                }
            }
            for (package_id, target_name) in request.doctest_targets {
                for mode in [RustTestListingMode::All, RustTestListingMode::Ignored] {
                    let mut command = util::command::new_command("cargo");
                    command.current_dir(&working_directory).kill_on_drop(true);
                    let mut arguments = ["test", "--offline", "--doc", "--package"]
                        .into_iter()
                        .map(OsString::from)
                        .collect::<Vec<_>>();
                    arguments.extend([
                        OsString::from(&package_id),
                        "--manifest-path".into(),
                        manifest_path.clone().into_os_string(),
                        "--".into(),
                        "--list".into(),
                        "--format".into(),
                        "terse".into(),
                    ]);
                    if mode == RustTestListingMode::Ignored {
                        arguments.push("--ignored".into());
                    }
                    let output = bounded_command_output(
                        command,
                        arguments,
                        &working_directory,
                        request.environment.as_ref(),
                        request.limits.timeout,
                        request.limits.listing_bytes,
                        request.limits.cargo_bytes,
                        RustTestDiscoveryStage::Doctest,
                        &executor,
                    )
                    .await?;
                    if !output.status.success() {
                        return Err(RustTestDiscoveryFailure {
                            stage: RustTestDiscoveryStage::Doctest,
                            kind: RustTestDiscoveryFailureKind::Exited(output.status),
                        }
                        .into());
                    }
                    listings.push(RustTestListingCapture {
                        package_id: package_id.clone(),
                        target_name: target_name.clone(),
                        target_kind: RustTestTargetKind::Doctest,
                        mode,
                        stdout: String::from_utf8(output.stdout).map_err(|_| {
                            RustTestDiscoveryFailure {
                                stage: RustTestDiscoveryStage::Doctest,
                                kind: RustTestDiscoveryFailureKind::InvalidUtf8,
                            }
                        })?,
                    });
                }
            }
            Ok(RustTestDiscoveryCapture {
                toolchain: "host".to_string(),
                cargo_messages,
                listings,
            })
        })
    }
}

async fn bounded_command_output(
    mut command: util::command::Command,
    arguments: Vec<OsString>,
    working_directory: &Path,
    environment: Option<&HashMap<String, String>>,
    timeout: Duration,
    stdout_limit: usize,
    stderr_limit: usize,
    stage: RustTestDiscoveryStage,
    executor: &BackgroundExecutor,
) -> Result<Output> {
    let output = futures::FutureExt::boxed(async move {
        if let Some(environment) = environment {
            command.envs(environment);
        }
        let options = prevent_passive_rust_downloads(
            &mut command,
            matches!(stage, RustTestDiscoveryStage::Doctest),
            working_directory,
            environment,
            executor,
        )
        .await
        .map_err(|_| RustTestDiscoveryFailure {
            stage,
            kind: RustTestDiscoveryFailureKind::PassivePolicy,
        })?;
        command.args(options).args(arguments);
        let mut child = command
            .stdin(util::command::Stdio::null())
            .stdout(util::command::Stdio::piped())
            .stderr(util::command::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| RustTestDiscoveryFailure {
                stage,
                kind: RustTestDiscoveryFailureKind::Start(error.kind()),
            })?;
        let stdout = child.stdout.take().ok_or(RustTestDiscoveryFailure {
            stage,
            kind: RustTestDiscoveryFailureKind::MissingPipe("stdout"),
        })?;
        let stderr = child.stderr.take().ok_or(RustTestDiscoveryFailure {
            stage,
            kind: RustTestDiscoveryFailureKind::MissingPipe("stderr"),
        })?;
        let (stdout, stderr) = futures::try_join!(
            bounded_discovery_stream(stdout, stdout_limit, "stdout", stage),
            bounded_discovery_stream(stderr, stderr_limit, "stderr", stage),
        )?;
        let status = child
            .status()
            .await
            .map_err(|error| RustTestDiscoveryFailure {
                stage,
                kind: RustTestDiscoveryFailureKind::Wait(error.kind()),
            })?;
        anyhow::Ok(Output {
            status,
            stdout,
            stderr,
        })
    });
    let timeout = futures::FutureExt::boxed(executor.timer(timeout));
    match futures::future::select(output, timeout).await {
        futures::future::Either::Left((output, _)) => output,
        futures::future::Either::Right(_) => Err(RustTestDiscoveryFailure {
            stage,
            kind: RustTestDiscoveryFailureKind::Timeout,
        }
        .into()),
    }
}

async fn bounded_discovery_stream(
    mut reader: impl AsyncRead + Unpin + Send,
    limit: usize,
    stream: &'static str,
    stage: RustTestDiscoveryStage,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let remaining = limit.saturating_sub(output.len());
        // Probe one byte beyond the budget without appending overflow to the captured output.
        let read_length = remaining.saturating_add(1).min(buffer.len());
        let bytes_read = reader
            .read(&mut buffer[..read_length])
            .await
            .map_err(|error| RustTestDiscoveryFailure {
                stage,
                kind: RustTestDiscoveryFailureKind::Read(stream, error.kind()),
            })?;
        if bytes_read == 0 {
            return Ok(output);
        }
        if bytes_read > remaining {
            return Err(RustTestDiscoveryFailure {
                stage,
                kind: RustTestDiscoveryFailureKind::OutputLimit(stream),
            }
            .into());
        }
        output.extend_from_slice(&buffer[..bytes_read]);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RustTestProtocolLimits {
    pub timeout: Duration,
    pub cargo_bytes: usize,
    pub listing_bytes: usize,
    pub lines: usize,
    pub harnesses: usize,
    pub cases: usize,
    pub field_bytes: usize,
    pub diagnostics: usize,
}

impl Default for RustTestProtocolLimits {
    fn default() -> Self {
        Self {
            timeout: RUST_TEST_DISCOVERY_TIMEOUT,
            cargo_bytes: MAX_RUST_TEST_CARGO_BYTES,
            listing_bytes: MAX_RUST_TEST_LISTING_BYTES,
            lines: MAX_RUST_TEST_LINES,
            harnesses: MAX_RUST_TEST_HARNESSES,
            cases: MAX_RUST_TEST_CASES,
            field_bytes: MAX_RUST_TEST_FIELD_BYTES,
            diagnostics: MAX_RUST_TEST_DIAGNOSTICS,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RustTestProtocolAdapter {
    limits: RustTestProtocolLimits,
}

impl RustTestProtocolAdapter {
    pub fn new(limits: RustTestProtocolLimits) -> Self {
        Self { limits }
    }

    pub fn adapt(
        &self,
        capture: RustTestDiscoveryCapture,
        source_hints: &[RustTestSourceHint],
    ) -> RustTestProtocolSnapshot {
        let mut diagnostics = Vec::new();
        let mut truncated = false;
        let mut harnesses =
            self.parse_cargo_messages(&capture.cargo_messages, &mut diagnostics, &mut truncated);
        self.add_doctest_harnesses(&capture.listings, &mut harnesses, &mut diagnostics);
        let mut cases = self.parse_listings(
            &capture.listings,
            &harnesses,
            source_hints,
            &mut diagnostics,
            &mut truncated,
        );
        harnesses.sort_by(|left, right| left.id.cmp(&right.id));
        cases.sort_by(|left, right| left.id.cmp(&right.id));
        if capture.toolchain.len() > self.limits.field_bytes {
            truncated = true;
            push_diagnostic(
                &mut diagnostics,
                self.limits,
                RustTestProtocolStage::Limits,
                "Rust toolchain descriptor exceeded the field limit".to_string(),
            );
        }
        let toolchain = bounded_field(&capture.toolchain, self.limits.field_bytes);
        let capability = if harnesses.is_empty() && cases.is_empty() {
            RustTestProtocolCapability::Unsupported
        } else if diagnostics.is_empty() && !truncated {
            RustTestProtocolCapability::Supported
        } else {
            RustTestProtocolCapability::Partial
        };
        RustTestProtocolSnapshot {
            protocol_version: RUST_TEST_PROTOCOL_VERSION,
            toolchain,
            capability,
            harnesses,
            cases,
            diagnostics,
            truncated,
        }
    }

    fn parse_cargo_messages(
        &self,
        input: &str,
        diagnostics: &mut Vec<RustTestProtocolDiagnostic>,
        truncated: &mut bool,
    ) -> Vec<RustTestHarness> {
        let decoder = CargoMessageDecoder::new(input, self.limits.cargo_bytes, self.limits.lines);
        if decoder.byte_limit_reached() {
            push_diagnostic(
                diagnostics,
                self.limits,
                RustTestProtocolStage::Limits,
                "Cargo JSON byte limit reached".to_string(),
            );
            *truncated = true;
        }
        let line_limit_reached = decoder.line_limit_reached();
        let mut harnesses = BTreeMap::new();
        for message in decoder {
            let message = match message {
                Ok(message) => message,
                Err(message) => {
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::CargoMessages,
                        message,
                    );
                    continue;
                }
            };
            match message {
                Message::CompilerArtifact(artifact) => {
                    if !artifact.profile.test {
                        continue;
                    }
                    let Some(executable) = artifact.executable else {
                        continue;
                    };
                    let Some(target_kind) = target_kind(&artifact.target.kind) else {
                        push_diagnostic(
                            diagnostics,
                            self.limits,
                            RustTestProtocolStage::CargoMessages,
                            format!(
                                "test artifact {} has an unsupported target kind",
                                artifact.target.name
                            ),
                        );
                        continue;
                    };
                    let package_id =
                        bounded_field(&artifact.package_id.repr, self.limits.field_bytes);
                    let target_name = bounded_field(&artifact.target.name, self.limits.field_bytes);
                    let id = harness_id(&package_id, &target_name, target_kind);
                    harnesses.insert(
                        id.clone(),
                        RustTestHarness {
                            id,
                            package_id,
                            target_name,
                            target_kind,
                            locator: RustTestHarnessLocator::Executable(
                                executable.into_std_path_buf(),
                            ),
                        },
                    );
                    if harnesses.len() >= self.limits.harnesses {
                        *truncated = true;
                        push_diagnostic(
                            diagnostics,
                            self.limits,
                            RustTestProtocolStage::Limits,
                            "Rust test harness limit reached".to_string(),
                        );
                        break;
                    }
                }
                Message::BuildFinished(result) if !result.success => push_diagnostic(
                    diagnostics,
                    self.limits,
                    RustTestProtocolStage::CargoMessages,
                    "Cargo reported an unsuccessful test-harness build".to_string(),
                ),
                Message::CompilerMessage(_)
                | Message::BuildScriptExecuted(_)
                | Message::BuildFinished(_) => {}
                _ => push_diagnostic(
                    diagnostics,
                    self.limits,
                    RustTestProtocolStage::CargoMessages,
                    "Cargo emitted a message unsupported by this protocol version".to_string(),
                ),
            }
        }
        if line_limit_reached {
            *truncated = true;
            push_diagnostic(
                diagnostics,
                self.limits,
                RustTestProtocolStage::Limits,
                "Cargo JSON line limit reached".to_string(),
            );
        }
        harnesses.into_values().collect()
    }

    fn add_doctest_harnesses(
        &self,
        listings: &[RustTestListingCapture],
        harnesses: &mut Vec<RustTestHarness>,
        diagnostics: &mut Vec<RustTestProtocolDiagnostic>,
    ) {
        let mut existing = harnesses
            .iter()
            .map(|harness| harness.id.clone())
            .collect::<BTreeSet<_>>();
        for listing in listings {
            if listing.target_kind != RustTestTargetKind::Doctest {
                continue;
            }
            let package_id = bounded_field(&listing.package_id, self.limits.field_bytes);
            let target_name = bounded_field(&listing.target_name, self.limits.field_bytes);
            let id = harness_id(&package_id, &target_name, RustTestTargetKind::Doctest);
            if existing.insert(id.clone()) {
                if harnesses.len() >= self.limits.harnesses {
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::Limits,
                        "Rust test harness limit reached before doctest discovery".to_string(),
                    );
                    return;
                }
                harnesses.push(RustTestHarness {
                    id,
                    package_id,
                    target_name,
                    target_kind: RustTestTargetKind::Doctest,
                    locator: RustTestHarnessLocator::CargoDoctest,
                });
            }
        }
    }

    fn parse_listings(
        &self,
        listings: &[RustTestListingCapture],
        harnesses: &[RustTestHarness],
        source_hints: &[RustTestSourceHint],
        diagnostics: &mut Vec<RustTestProtocolDiagnostic>,
        truncated: &mut bool,
    ) -> Vec<RustTestCaseRecord> {
        let harnesses = harnesses
            .iter()
            .map(|harness| {
                (
                    (
                        harness.package_id.as_str(),
                        harness.target_name.as_str(),
                        harness.target_kind,
                    ),
                    harness,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let hints = source_hints
            .iter()
            .map(|hint| {
                (
                    (
                        cargo_package_name(&hint.package_id),
                        hint.target_name.as_str(),
                        hint.target_kind,
                        hint.test_name.as_str(),
                    ),
                    hint,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let listing_modes = listings
            .iter()
            .map(|listing| {
                (
                    listing.package_id.as_str(),
                    listing.target_name.as_str(),
                    listing.target_kind,
                    listing.mode == RustTestListingMode::Ignored,
                )
            })
            .collect::<BTreeSet<_>>();
        for harness in harnesses
            .values()
            .filter(|harness| harness.target_kind == RustTestTargetKind::Doctest)
        {
            if [false, true].into_iter().any(|ignored| {
                !listing_modes.contains(&(
                    harness.package_id.as_str(),
                    harness.target_name.as_str(),
                    harness.target_kind,
                    ignored,
                ))
            }) {
                push_diagnostic(
                    diagnostics,
                    self.limits,
                    RustTestProtocolStage::HarnessListing,
                    "Doctest discovery requires both all and ignored listings".to_string(),
                );
            }
        }
        let mut cases = BTreeMap::new();
        let mut total_bytes = 0usize;
        let mut total_lines = 0usize;
        let mut doctest_listing_names = BTreeSet::new();
        for listing in listings {
            total_bytes = total_bytes.saturating_add(listing.stdout.len());
            if total_bytes > self.limits.listing_bytes {
                *truncated = true;
                push_diagnostic(
                    diagnostics,
                    self.limits,
                    RustTestProtocolStage::Limits,
                    "Rust test listing byte limit reached".to_string(),
                );
                break;
            }
            let Some(harness) = harnesses.get(&(
                listing.package_id.as_str(),
                listing.target_name.as_str(),
                listing.target_kind,
            )) else {
                push_diagnostic(
                    diagnostics,
                    self.limits,
                    RustTestProtocolStage::HarnessListing,
                    format!(
                        "listing for {}:{} has no structured Cargo harness",
                        listing.package_id, listing.target_name
                    ),
                );
                continue;
            };
            for (line_index, line) in listing.stdout.lines().enumerate() {
                total_lines = total_lines.saturating_add(1);
                if total_lines > self.limits.lines {
                    *truncated = true;
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::Limits,
                        "Rust test case or line limit reached".to_string(),
                    );
                    break;
                }
                let line = line.trim();
                if line.is_empty() || is_listing_summary(line) {
                    continue;
                }
                let Some((name, kind)) = parse_listing_line(line) else {
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::HarnessListing,
                        format!(
                            "{} listing line {} has an unknown record",
                            harness.target_name,
                            line_index + 1
                        ),
                    );
                    continue;
                };
                if name.len() > self.limits.field_bytes {
                    *truncated = true;
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::Limits,
                        format!(
                            "{} emitted a test name over the field limit",
                            harness.target_name
                        ),
                    );
                }
                let name = bounded_field(name, self.limits.field_bytes);
                if name.is_empty() {
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::HarnessListing,
                        format!("{} emitted an empty test name", harness.target_name),
                    );
                    continue;
                }
                if harness.target_kind == RustTestTargetKind::Doctest
                    && (kind != RustTestCaseKind::Test
                        || (listing.mode == RustTestListingMode::All
                            && !doctest_listing_names.insert((harness.id.clone(), name.clone()))))
                {
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::HarnessListing,
                        "Doctest listing contains ambiguous or unsupported case identities"
                            .to_string(),
                    );
                }
                let id = case_id(&harness.id, &name, kind);
                if cases.len() >= self.limits.cases && !cases.contains_key(&id) {
                    *truncated = true;
                    push_diagnostic(
                        diagnostics,
                        self.limits,
                        RustTestProtocolStage::Limits,
                        "Rust test case limit reached".to_string(),
                    );
                    break;
                }
                let hint = hints.get(&(
                    cargo_package_name(&harness.package_id),
                    harness.target_name.as_str(),
                    harness.target_kind,
                    name.as_str(),
                ));
                let source = hint.and_then(|hint| hint.source.clone());
                let runnable = hint
                    .and_then(|hint| hint.runnable.clone())
                    .unwrap_or_else(|| default_runnable(harness, &name));
                cases
                    .entry(id.clone())
                    .and_modify(|record: &mut RustTestCaseRecord| {
                        if listing.mode == RustTestListingMode::Ignored {
                            record.ignored = true;
                        }
                        if record.source.is_none() {
                            record.source = source.clone();
                        }
                    })
                    .or_insert(RustTestCaseRecord {
                        id,
                        harness_id: harness.id.clone(),
                        name,
                        kind,
                        ignored: listing.mode == RustTestListingMode::Ignored,
                        source,
                        runnable,
                    });
            }
            if *truncated {
                break;
            }
        }
        cases.into_values().collect()
    }
}

fn target_kind(kinds: &[TargetKind]) -> Option<RustTestTargetKind> {
    if kinds.contains(&TargetKind::Test) {
        Some(RustTestTargetKind::Integration)
    } else if kinds.contains(&TargetKind::Bench) {
        Some(RustTestTargetKind::Benchmark)
    } else if kinds.contains(&TargetKind::Example) {
        Some(RustTestTargetKind::Example)
    } else if kinds.contains(&TargetKind::Bin) {
        Some(RustTestTargetKind::Binary)
    } else if kinds.contains(&TargetKind::Lib) {
        Some(RustTestTargetKind::Unit)
    } else {
        None
    }
}

fn parse_listing_line(line: &str) -> Option<(&str, RustTestCaseKind)> {
    let (name, kind) = line.rsplit_once(": ")?;
    match kind {
        "test" => Some((name, RustTestCaseKind::Test)),
        "benchmark" => Some((name, RustTestCaseKind::Benchmark)),
        _ => None,
    }
}

fn is_listing_summary(line: &str) -> bool {
    if let Some(durations) = line.strip_prefix("all doctests ran in ") {
        return durations
            .split_once("; merged doctests compilation took ")
            .is_some_and(|(running, compilation)| {
                [running, compilation].into_iter().all(|duration| {
                    duration
                        .strip_suffix('s')
                        .and_then(|duration| duration.parse::<f64>().ok())
                        .is_some_and(|duration| duration.is_finite() && duration >= 0.0)
                })
            });
    }
    line.split_whitespace().all(|part| {
        part.parse::<usize>().is_ok()
            || matches!(
                part.trim_end_matches(','),
                "test" | "tests" | "benchmark" | "benchmarks"
            )
    })
}

fn default_runnable(harness: &RustTestHarness, test_name: &str) -> RustTestRunnableSelector {
    let mut cargo_args = vec![
        "test".to_string(),
        "--package".to_string(),
        harness.package_id.clone(),
    ];
    match harness.target_kind {
        RustTestTargetKind::Unit => cargo_args.push("--lib".to_string()),
        RustTestTargetKind::Integration => {
            cargo_args.extend(["--test".to_string(), harness.target_name.clone()]);
        }
        RustTestTargetKind::Binary => {
            cargo_args.extend(["--bin".to_string(), harness.target_name.clone()]);
        }
        RustTestTargetKind::Example => {
            cargo_args.extend(["--example".to_string(), harness.target_name.clone()]);
        }
        RustTestTargetKind::Benchmark => {
            cargo_args[0] = "bench".to_string();
            cargo_args.extend(["--bench".to_string(), harness.target_name.clone()]);
        }
        RustTestTargetKind::Doctest => cargo_args.push("--doc".to_string()),
    }
    RustTestRunnableSelector {
        cargo_args,
        executable_args: vec![test_name.to_string(), "--exact".to_string()],
    }
}

fn harness_id(package_id: &str, target_name: &str, kind: RustTestTargetKind) -> String {
    format!(
        "rust-harness:v1:{}:{}:{}:{}:{kind:?}",
        package_id.len(),
        package_id,
        target_name.len(),
        target_name
    )
}

fn case_id(harness_id: &str, name: &str, kind: RustTestCaseKind) -> String {
    format!(
        "rust-case:v1:{}:{}:{}:{}:{kind:?}",
        harness_id.len(),
        harness_id,
        name.len(),
        name
    )
}

fn bounded_field(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_string();
    }
    let mut end = limit;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_string()
}

fn push_diagnostic(
    diagnostics: &mut Vec<RustTestProtocolDiagnostic>,
    limits: RustTestProtocolLimits,
    stage: RustTestProtocolStage,
    message: String,
) {
    if diagnostics.len() >= limits.diagnostics {
        return;
    }
    diagnostics.push(RustTestProtocolDiagnostic {
        stage,
        message: bounded_field(&message, limits.field_bytes),
    });
}

#[derive(Clone, Debug)]
pub enum RustTestProviderStoreEvent {
    Invalidated,
    Changed(DiscoveryGeneration),
}

impl EventEmitter<RustTestProviderStoreEvent> for RustTestProviderStore {}

enum RustTestProviderStoreMode {
    Local {
        runner: Arc<dyn RustTestDiscoveryRunner>,
        cargo_store: Entity<CargoWorkspaceStore>,
        structured_store: Entity<StructuredExecutionStore>,
        environment: Entity<ProjectEnvironment>,
    },
    Remote {
        project_id: u64,
        client: AnyProtoClient,
        structured_store: Entity<StructuredExecutionStore>,
    },
}

type RustTestDiscoveryRefreshTask =
    Task<Result<Arc<StructuredProviderSnapshot>, Arc<anyhow::Error>>>;
type RemoteDiscoveryRetirement = Shared<Task<Result<(), Arc<anyhow::Error>>>>;

struct RemoteDiscoveryCancellation {
    retirement: Option<futures::channel::oneshot::Sender<bool>>,
    pending: bool,
}

impl RemoteDiscoveryCancellation {
    fn start(&mut self) {
        self.pending = true;
    }

    fn complete(&mut self) {
        self.pending = false;
    }
}

impl Drop for RemoteDiscoveryCancellation {
    fn drop(&mut self) {
        if let Some(retirement) = self.retirement.take()
            && retirement.send(self.pending).is_err()
        {
            log::debug!("Rust discovery retirement receiver was dropped");
        }
    }
}

pub struct RustTestProviderStore {
    mode: RustTestProviderStoreMode,
    worktree_store: Entity<WorktreeStore>,
    generation: u64,
    next_run_id: u64,
    source_hints: Vec<RustTestSourceHint>,
    semantic_hints: Vec<RustSemanticBufferHints>,
    lsp_store: Option<WeakEntity<LspStore>>,
    semantic_request_generation: Option<u64>,
    actions: HashMap<StructuredNodeId, RustTestActionDescriptor>,
    run_handles: HashMap<StructuredRunId, StructuredTaskHandle>,
    refresh_task: Task<()>,
    invalidation_task: Task<()>,
    active_remote_request_id: Option<u64>,
    active_remote_request_abort: Option<AbortHandle>,
    active_remote_request_retirement: Option<(u64, RemoteDiscoveryRetirement)>,
    disconnected: bool,
    connection_epoch: u64,
    unsupported_remote_discovery: Option<(u64, u64)>,
    active_remote_requests: HashMap<(proto::PeerId, u64), AbortHandle>,
    cancelled_remote_requests: HashSet<(proto::PeerId, u64)>,
    pending_discovery_refreshes:
        HashMap<Vec<WorktreeId>, (u64, WeakShared<RustTestDiscoveryRefreshTask>)>,
    #[cfg(any(test, feature = "test-support"))]
    discovery_reply_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    action_reply_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    authorized_remote_runs: HashMap<(proto::PeerId, StructuredRunId), AuthorizedRemoteRun>,
    pending_remote_actions: HashMap<(proto::PeerId, u64), PendingRemoteRustAction>,
    next_remote_action_id: u64,
    remote_lifecycle_tx: Option<mpsc::UnboundedSender<proto::UpdateRustTestRun>>,
    _remote_lifecycle_task: Task<()>,
    _subscriptions: Vec<Subscription>,
}

impl RustTestProviderStore {
    #[cfg(any(test, feature = "test-support"))]
    pub fn process_discovery_for_test(
        request: RustTestDiscoveryRequest,
        executor: BackgroundExecutor,
    ) -> BoxFuture<'static, Result<RustTestDiscoveryCapture>> {
        ProcessRustTestDiscoveryRunner { executor }.discover(request)
    }

    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_get_discovery);
        client.add_entity_request_handler(Self::handle_cancel_discovery);
        client.add_entity_request_handler(Self::handle_resolve_action);
        client.add_entity_request_handler(Self::handle_update_run);
    }

    pub fn local(
        worktree_store: Entity<WorktreeStore>,
        environment: Entity<ProjectEnvironment>,
        cargo_store: Entity<CargoWorkspaceStore>,
        structured_store: Entity<StructuredExecutionStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::local_with_runner(
            worktree_store,
            environment,
            cargo_store,
            structured_store,
            Arc::new(ProcessRustTestDiscoveryRunner {
                executor: cx.background_executor().clone(),
            }),
            cx,
        )
    }

    pub fn local_with_runner(
        worktree_store: Entity<WorktreeStore>,
        environment: Entity<ProjectEnvironment>,
        cargo_store: Entity<CargoWorkspaceStore>,
        structured_store: Entity<StructuredExecutionStore>,
        runner: Arc<dyn RustTestDiscoveryRunner>,
        cx: &mut Context<Self>,
    ) -> Self {
        let cargo_subscription = cx.subscribe(&cargo_store, |store, _, _, cx| {
            store.schedule_invalidation(cx)
        });
        let worktree_subscription = cx.subscribe(&worktree_store, |store, _, event, cx| {
            if rust_test_input_changed(event) {
                store.semantic_hints.clear();
                store.schedule_invalidation(cx);
            }
        });
        let mut subscriptions = vec![cargo_subscription, worktree_subscription];
        subscriptions.push(cx.observe_global::<feature_flags::FeatureFlagStore>({
            let mut enabled = cx.has_flag::<RustExactDoctestActionsExperimentalFeatureFlag>();
            move |store, cx| {
                let current = cx.has_flag::<RustExactDoctestActionsExperimentalFeatureFlag>();
                if current != enabled {
                    enabled = current;
                    if let Err(error) = store.renew_experimental_availability(cx) {
                        log::warn!(
                            "Failed to renew Rust test experimental availability: {error:#}"
                        );
                    }
                }
            }
        }));
        if let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(cx.subscribe(&trusted_worktrees, |store, _, event, cx| {
                let (worktree_store, paths) = match event {
                    TrustedWorktreesEvent::Trusted(worktree_store, paths)
                    | TrustedWorktreesEvent::Restricted(worktree_store, paths) => {
                        (worktree_store, paths)
                    }
                };
                if *worktree_store != store.worktree_store.downgrade() {
                    return;
                }
                if matches!(event, TrustedWorktreesEvent::Restricted(..)) {
                    store.restrict_remote_actions(paths);
                }
                store.semantic_hints.clear();
                store.schedule_invalidation(cx)
            }));
        }
        Self {
            mode: RustTestProviderStoreMode::Local {
                runner,
                cargo_store,
                structured_store,
                environment,
            },
            worktree_store,
            generation: 0,
            next_run_id: 0,
            source_hints: Vec::new(),
            semantic_hints: Vec::new(),
            lsp_store: None,
            semantic_request_generation: None,
            actions: HashMap::default(),
            run_handles: HashMap::default(),
            refresh_task: Task::ready(()),
            invalidation_task: Task::ready(()),
            active_remote_request_id: None,
            active_remote_request_abort: None,
            active_remote_request_retirement: None,
            disconnected: false,
            connection_epoch: 0,
            unsupported_remote_discovery: None,
            active_remote_requests: HashMap::default(),
            cancelled_remote_requests: HashSet::new(),
            pending_discovery_refreshes: HashMap::default(),
            #[cfg(any(test, feature = "test-support"))]
            discovery_reply_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            action_reply_gate: None,
            authorized_remote_runs: HashMap::default(),
            pending_remote_actions: HashMap::default(),
            next_remote_action_id: 0,
            remote_lifecycle_tx: None,
            _remote_lifecycle_task: Task::ready(()),
            _subscriptions: subscriptions,
        }
    }

    pub fn remote(
        project_id: u64,
        client: AnyProtoClient,
        worktree_store: Entity<WorktreeStore>,
        structured_store: Entity<StructuredExecutionStore>,
        cx: &mut Context<Self>,
    ) -> Self {
        let worktree_subscription = cx.subscribe(&worktree_store, |store, _, event, cx| {
            if rust_test_input_changed(event) {
                store.schedule_invalidation(cx);
            }
        });
        let mut subscriptions = vec![worktree_subscription];
        if let Some(trusted_worktrees) = TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(cx.subscribe(&trusted_worktrees, |store, _, event, _| {
                if let TrustedWorktreesEvent::Restricted(worktree_store, paths) = event
                    && *worktree_store == store.worktree_store.downgrade()
                {
                    store.restrict_remote_actions(paths);
                }
            }));
        }
        let (remote_lifecycle_tx, mut remote_lifecycle_rx) = mpsc::unbounded();
        let lifecycle_client = client.clone();
        let remote_lifecycle_task = cx.spawn(async move |_, _| {
            while let Some(request) = remote_lifecycle_rx.next().await {
                if let Err(error) = lifecycle_client.request(request).await {
                    log::warn!("Failed to publish remote Rust test lifecycle: {error:#}");
                }
            }
        });
        Self {
            mode: RustTestProviderStoreMode::Remote {
                project_id,
                client,
                structured_store,
            },
            worktree_store,
            generation: 0,
            next_run_id: 0,
            source_hints: Vec::new(),
            semantic_hints: Vec::new(),
            lsp_store: None,
            semantic_request_generation: None,
            actions: HashMap::default(),
            run_handles: HashMap::default(),
            refresh_task: Task::ready(()),
            invalidation_task: Task::ready(()),
            active_remote_request_id: None,
            active_remote_request_abort: None,
            active_remote_request_retirement: None,
            disconnected: false,
            connection_epoch: 0,
            unsupported_remote_discovery: None,
            active_remote_requests: HashMap::default(),
            cancelled_remote_requests: HashSet::new(),
            pending_discovery_refreshes: HashMap::default(),
            #[cfg(any(test, feature = "test-support"))]
            discovery_reply_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            action_reply_gate: None,
            authorized_remote_runs: HashMap::default(),
            pending_remote_actions: HashMap::default(),
            next_remote_action_id: 0,
            remote_lifecycle_tx: Some(remote_lifecycle_tx),
            _remote_lifecycle_task: remote_lifecycle_task,
            _subscriptions: subscriptions,
        }
    }

    pub fn provider_id() -> StructuredProviderId {
        StructuredProviderId(RUST_TEST_PROVIDER_ID.to_string())
    }

    pub fn generation(&self) -> DiscoveryGeneration {
        DiscoveryGeneration(self.generation)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn active_remote_discovery_request_count_for_test(&self) -> usize {
        self.active_remote_requests.len()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn discovery_idle_for_test(&self) -> bool {
        self.invalidation_task.is_ready()
            && self.refresh_task.is_ready()
            && self.active_remote_request_id.is_none()
            && self.active_remote_request_retirement.is_none()
            && self.active_remote_requests.is_empty()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_discovery_runner_for_test(
        &mut self,
        replacement: Arc<dyn RustTestDiscoveryRunner>,
    ) -> Result<()> {
        let RustTestProviderStoreMode::Local { runner, .. } = &mut self.mode else {
            anyhow::bail!("A remote Rust test provider has no local discovery runner");
        };
        *runner = replacement;
        Ok(())
    }

    pub fn unshared(&mut self) {
        self.authorized_remote_runs.clear();
        self.cancelled_remote_requests.clear();
        self.pending_discovery_refreshes.clear();
        for (_, pending) in self.pending_remote_actions.drain() {
            pending.abort.abort();
        }
        for (_, handle) in self.active_remote_requests.drain() {
            handle.abort();
        }
    }

    pub fn forget_peer(&mut self, peer_id: proto::PeerId) {
        self.authorized_remote_runs
            .retain(|(owner, _), _| *owner != peer_id);
        self.cancelled_remote_requests
            .retain(|(owner, _)| *owner != peer_id);
        self.pending_remote_actions.retain(|(owner, _), pending| {
            if *owner == peer_id {
                pending.abort.abort();
                false
            } else {
                true
            }
        });
        self.active_remote_requests.retain(|(owner, _), handle| {
            if *owner == peer_id {
                handle.abort();
                false
            } else {
                true
            }
        });
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_action_reply_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        self.action_reply_gate = Some((started_sender, release_receiver));
        (started_receiver, release_sender)
    }

    fn restrict_remote_actions(&mut self, paths: &collections::HashSet<PathTrust>) {
        self.authorized_remote_runs.retain(|_, authorization| {
            !paths.contains(&PathTrust::Worktree(authorization.worktree_id))
        });
        self.pending_remote_actions.retain(|_, pending| {
            if pending
                .worktree_ids
                .iter()
                .any(|id| paths.contains(&PathTrust::Worktree(*id)))
            {
                pending.abort.abort();
                false
            } else {
                true
            }
        });
    }

    fn remote_action_client(&self) -> Option<(u64, AnyProtoClient)> {
        match &self.mode {
            RustTestProviderStoreMode::Remote {
                project_id, client, ..
            } => Some((*project_id, client.clone())),
            RustTestProviderStoreMode::Local { .. } => None,
        }
    }

    async fn forward_remote_action<T: proto::RequestMessage>(
        this: &Entity<Self>,
        peer_id: proto::PeerId,
        allowed: &HashSet<WorktreeId>,
        client: AnyProtoClient,
        request: T,
        cx: &mut AsyncApp,
    ) -> Result<T::Response> {
        let (abort, registration) = AbortHandle::new_pair();
        let key = this.update(cx, |store, cx| {
            if store.disconnected {
                bail!("The Rust test action host is disconnected");
            }
            if store.pending_remote_actions.len() >= MAX_PENDING_REMOTE_RUST_TEST_RUNS {
                bail!("Too many pending remote Rust test actions");
            }
            for worktree_id in allowed {
                store.ensure_worktree_trusted(*worktree_id, cx)?;
            }
            store.next_remote_action_id = store
                .next_remote_action_id
                .checked_add(1)
                .context("Rust test action request IDs exhausted")?;
            let key = (peer_id, store.next_remote_action_id);
            store.pending_remote_actions.insert(
                key,
                PendingRemoteRustAction {
                    abort,
                    worktree_ids: allowed.clone(),
                },
            );
            Ok(key)
        })?;
        #[cfg(any(test, feature = "test-support"))]
        let reply_gate = this.update(cx, |store, _| store.action_reply_gate.take());
        let response = Abortable::new(
            async {
                let response = client.forward_request(peer_id, request).await?;
                #[cfg(any(test, feature = "test-support"))]
                if let Some((started, release)) = reply_gate {
                    started
                        .send(())
                        .map_err(|_| anyhow!("Rust action reply start receiver was dropped"))?;
                    release
                        .await
                        .context("Rust action reply release was dropped")?;
                }
                Ok(response)
            },
            registration,
        )
        .await;
        this.update(cx, |store, cx| {
            store
                .pending_remote_actions
                .remove(&key)
                .context("The remote Rust test action authorization expired")?;
            if store.disconnected {
                bail!("The Rust test action host is disconnected");
            }
            for worktree_id in allowed {
                store.ensure_worktree_trusted(*worktree_id, cx)?;
            }
            response.map_err(|_| anyhow!("The remote Rust test action was cancelled"))?
        })
    }

    fn authorize_remote_plan(
        &mut self,
        peer_id: proto::PeerId,
        authorization_generation: u64,
        plan: &RustTestActionPlan,
    ) -> Result<()> {
        if let RustTestActionPlan::Task {
            run_id,
            discovery_generation,
            scope_node_ids,
            worktree_id,
            ..
        } = plan
        {
            self.authorized_remote_runs
                .retain(|(owner, _), authorization| {
                    *owner != peer_id
                        || authorization.authorization_generation == authorization_generation
                });
            if self.authorized_remote_runs.len() >= MAX_PENDING_REMOTE_RUST_TEST_RUNS {
                bail!("Too many pending remote Rust test runs");
            }
            self.authorized_remote_runs.insert(
                (peer_id, run_id.clone()),
                AuthorizedRemoteRun {
                    discovery_generation: *discovery_generation,
                    scope_node_ids: scope_node_ids.clone(),
                    worktree_id: *worktree_id,
                    authorization_generation,
                },
            );
        }
        Ok(())
    }

    fn validate_remote_run(
        &mut self,
        peer_id: proto::PeerId,
        request: &proto::UpdateRustTestRun,
        allowed: &HashSet<WorktreeId>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let key = (peer_id, StructuredRunId(request.run_id.clone()));
        let authorization = self
            .authorized_remote_runs
            .get(&key)
            .context("Unknown or expired remote Rust test run")?;
        if authorization.authorization_generation != request.authorization_generation {
            self.authorized_remote_runs.remove(&key);
            bail!("Unknown or expired remote Rust test run authorization");
        }
        if authorization.discovery_generation.0 != request.discovery_generation
            || !authorization
                .scope_node_ids
                .iter()
                .map(|id| &id.0)
                .eq(request.scope_node_ids.iter())
        {
            bail!("Remote Rust test lifecycle identity mismatch");
        }
        if !allowed.contains(&authorization.worktree_id) {
            bail!("Remote Rust test lifecycle is outside the visible worktree scope");
        }
        if let Err(error) = self.ensure_worktree_trusted(authorization.worktree_id, cx) {
            self.authorized_remote_runs.remove(&key);
            return Err(error);
        }
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_discovery_reply_for_test(
        &mut self,
    ) -> (
        futures::channel::oneshot::Receiver<()>,
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    ) {
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
        self.discovery_reply_gate = Some((started_sender, release_receiver, completed_sender));
        (started_receiver, release_sender, completed_receiver)
    }

    fn renew_experimental_availability(&self, cx: &mut Context<Self>) -> Result<()> {
        let RustTestProviderStoreMode::Local {
            structured_store, ..
        } = &self.mode
        else {
            return Ok(());
        };
        structured_store.update(cx, |store, cx| {
            let Some(mut snapshot) = store.state().provider(&Self::provider_id()).cloned() else {
                return Ok(());
            };
            stamp_rust_test_experimental_availability(&mut snapshot, cx);
            let project_generation = store.state().project_generation();
            let outcome = store
                .state_mut()
                .apply_discovery(project_generation, snapshot, None)?;
            if outcome != crate::structured_execution::StructuredApplyOutcome::Duplicate {
                cx.emit(StructuredExecutionStoreEvent::Changed(Self::provider_id()));
            }
            Ok(())
        })
    }

    pub fn action_experimental_requirements(
        &self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        cx: &App,
    ) -> Result<Vec<String>> {
        match &self.mode {
            RustTestProviderStoreMode::Local { .. } => {
                anyhow::ensure!(
                    discovery_generation == self.generation(),
                    "The selected Rust test belongs to a stale discovery generation"
                );
                let descriptor = self
                    .actions
                    .get(node_id)
                    .context("The selected Rust test is not executable on this project host")?;
                Ok(
                    if descriptor.scope == RustTestActionScope::Case
                        && descriptor.target_kind == Some(RustTestTargetKind::Doctest)
                    {
                        vec!["rust-exact-doctest-actions-experimental".to_string()]
                    } else {
                        Vec::new()
                    },
                )
            }
            RustTestProviderStoreMode::Remote {
                structured_store, ..
            } => {
                if !self.disconnected
                    && self.unsupported_remote_discovery
                        == Some((self.generation, self.connection_epoch))
                {
                    return Err(anyhow!(RustTestRemoteError {
                        kind: RustTestRemoteErrorKind::UnsupportedActionHost,
                    }));
                }
                let snapshot = structured_store
                    .read(cx)
                    .state()
                    .provider(&Self::provider_id())
                    .context("Rust test discovery is unavailable. Refresh Tests")?;
                anyhow::ensure!(
                    snapshot.discovery_generation == discovery_generation,
                    "The selected Rust test belongs to a stale discovery generation"
                );
                let node = snapshot
                    .nodes
                    .iter()
                    .find(|node| &node.id == node_id)
                    .context("The selected Rust test is no longer available. Refresh Tests")?;
                anyhow::ensure!(
                    node.experimental_feature
                        .as_deref()
                        .is_none_or(|feature| feature == "rust-exact-doctest-actions-experimental"),
                    "The selected Rust test has unsupported experimental requirements"
                );
                anyhow::ensure!(
                    node.experimental_feature.is_none()
                        || node.experimental_feature_enabled == Some(true),
                    "Exact Rust Doctest actions require experimental opt-in on the project host"
                );
                Ok(node.experimental_feature.iter().cloned().collect())
            }
        }
    }

    pub fn plan_action(
        &mut self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        action: RustTestAction,
        cx: &mut Context<Self>,
    ) -> Task<Result<RustTestActionPlan>> {
        let pending_operation = match self
            .action_experimental_requirements(node_id, discovery_generation, cx)
            .and_then(|requirements| PendingExperimentalFeatureOperation::new(&requirements, cx))
        {
            Ok(operation) => operation,
            Err(error) => return Task::ready(Err(error)),
        };

        match &self.mode {
            RustTestProviderStoreMode::Local { .. } => Task::ready(
                self.plan_local_action(node_id, discovery_generation, action, cx)
                    .and_then(|plan| {
                        pending_operation
                            .check_requirements(plan.required_experimental_features(), cx)?;
                        Ok(plan)
                    }),
            ),
            RustTestProviderStoreMode::Remote {
                project_id, client, ..
            } => {
                let request = proto::ResolveRustTestAction {
                    project_id: *project_id,
                    protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                    discovery_generation: discovery_generation.0,
                    node_id: node_id.0.clone(),
                    action: action_to_proto(action),
                    worktree_ids: visible_worktree_ids(&self.worktree_store, cx),
                    authorization_generation: 0,
                };
                let client = client.clone();
                let selected_node = node_id.clone();
                cx.spawn(async move |this, cx| {
                    let response = client
                        .request(request)
                        .await
                        .map_err(map_remote_action_error)?;
                    this.read_with(cx, |store, cx| {
                        if let Some(descriptor) = store.actions.get(&selected_node) {
                            store.ensure_action_experimental_features(descriptor, cx)?;
                        }
                        let requirements = store.action_experimental_requirements(
                            &selected_node,
                            discovery_generation,
                            cx,
                        )?;
                        pending_operation.check_requirements(&requirements, cx)
                    })??;
                    let plan = decode_action_plan(response)?;
                    cx.update(|cx| {
                        pending_operation
                            .check_requirements(plan.required_experimental_features(), cx)
                    })?;
                    Ok(plan)
                })
            }
        }
    }

    fn plan_local_action(
        &mut self,
        node_id: &StructuredNodeId,
        discovery_generation: DiscoveryGeneration,
        action: RustTestAction,
        cx: &mut Context<Self>,
    ) -> Result<RustTestActionPlan> {
        if discovery_generation != self.generation() {
            bail!("The selected Rust test belongs to a stale discovery generation");
        }
        let descriptor = self
            .actions
            .get(node_id)
            .context("The selected Rust test is not executable on this project host")?
            .clone();
        self.ensure_action_experimental_features(&descriptor, cx)?;
        self.ensure_worktree_trusted(descriptor.worktree_id, cx)?;
        self.next_run_id = self.next_run_id.wrapping_add(1);
        plan_rust_test_action(
            &descriptor,
            node_id,
            discovery_generation,
            self.next_run_id,
            action,
        )
    }

    fn ensure_action_experimental_features(
        &self,
        descriptor: &RustTestActionDescriptor,
        cx: &App,
    ) -> Result<()> {
        anyhow::ensure!(
            descriptor.scope != RustTestActionScope::Case
                || descriptor.target_kind != Some(RustTestTargetKind::Doctest)
                || cx.has_flag::<RustExactDoctestActionsExperimentalFeatureFlag>(),
            "Exact Rust Doctest actions require local experimental opt-in"
        );
        Ok(())
    }

    fn ensure_worktree_trusted(
        &self,
        worktree_id: WorktreeId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if !self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .any(|worktree| worktree.read(cx).id() == worktree_id)
        {
            bail!("The selected Rust test worktree is no longer visible. Refresh Tests");
        }
        let trusted = TrustedWorktrees::try_get_global(cx)
            .map(|trusted| {
                trusted.update(cx, |trusted, cx| {
                    trusted.can_trust(&self.worktree_store, worktree_id, cx)
                })
            })
            .unwrap_or(true);
        if !trusted {
            bail!("The selected Rust test is restricted. Trust the worktree and refresh Tests");
        }
        Ok(())
    }

    pub fn register_run_handle(
        &mut self,
        run_id: StructuredRunId,
        discovery_generation: DiscoveryGeneration,
        scope_node_ids: Vec<StructuredNodeId>,
        handle: StructuredTaskHandle,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        match &self.mode {
            RustTestProviderStoreMode::Local {
                structured_store, ..
            } => {
                structured_store.update(cx, |store, cx| {
                    let project_generation = store.state().project_generation();
                    store.state_mut().begin_run(
                        project_generation,
                        &Self::provider_id(),
                        crate::structured_execution::StructuredRun::new(
                            run_id.clone(),
                            discovery_generation,
                            scope_node_ids.clone(),
                        ),
                    )?;
                    store.observe_task_handle(Self::provider_id(), run_id.clone(), &handle, cx)?;
                    cx.emit(StructuredExecutionStoreEvent::Changed(Self::provider_id()));
                    anyhow::Ok(())
                })?;
            }
            RustTestProviderStoreMode::Remote { project_id, .. } => {
                let lifecycle_tx = self
                    .remote_lifecycle_tx
                    .clone()
                    .context("The remote Rust test lifecycle channel is unavailable")?;
                let project_id = *project_id;
                let worktree_ids = visible_worktree_ids(&self.worktree_store, cx);
                let run_id = run_id.clone();
                handle.subscribe(cx, move |event, _| {
                    let request = lifecycle_to_proto(
                        project_id,
                        discovery_generation,
                        &run_id,
                        &scope_node_ids,
                        event,
                        worktree_ids.clone(),
                    );
                    if let Err(error) = lifecycle_tx.unbounded_send(request) {
                        log::warn!("Failed to queue remote Rust test lifecycle: {error}");
                    }
                });
            }
        }
        cancel_superseded_run_handles(&self.run_handles, &run_id, cx);
        if self.run_handles.len() >= MAX_RUST_TEST_RERUNS * 2 {
            self.run_handles
                .retain(|_, handle| !handle.state().is_terminal());
        }
        self.run_handles.insert(run_id, handle);
        Ok(())
    }

    pub fn has_active_runs(&self) -> bool {
        self.run_handles
            .values()
            .any(|handle| !handle.state().is_terminal())
    }

    pub fn cancel_run(&self, run_id: &StructuredRunId, cx: &mut App) -> Result<()> {
        let handle = self
            .run_handles
            .get(run_id)
            .context("The selected Rust test run is no longer active")?;
        if !handle.cancel(cx) {
            bail!("The selected Rust test run is already complete");
        }
        Ok(())
    }

    pub fn reveal_run_terminal(
        &self,
        run_id: &StructuredRunId,
        window: &mut gpui::Window,
        cx: &mut App,
    ) -> Result<()> {
        let handle = self
            .run_handles
            .get(run_id)
            .context("The selected Rust test run has no retained task terminal")?;
        if !handle.reveal_terminal(window, cx) {
            bail!("The selected Rust test run has no available task terminal");
        }
        Ok(())
    }

    pub fn set_source_hints(
        &mut self,
        source_hints: Vec<RustTestSourceHint>,
        cx: &mut Context<Self>,
    ) {
        self.source_hints = source_hints;
        self.schedule_invalidation(cx);
    }

    pub fn observe_lsp_store(&mut self, lsp_store: &Entity<LspStore>, cx: &mut Context<Self>) {
        self.lsp_store = Some(lsp_store.downgrade());
        self._subscriptions
            .push(cx.subscribe(lsp_store, |store, _, event, cx| match event {
                LspStoreEvent::RustRunnablesUpdated {
                    buffer,
                    server_id,
                    runnables,
                } => {
                    let version = buffer.read(cx).version();
                    let retained = runnables
                        .iter()
                        .filter(|runnable| semantic_runnable_identity(runnable).is_some())
                        .take(MAX_RUST_TEST_SEMANTIC_RUNNABLES)
                        .cloned()
                        .collect::<Vec<_>>();
                    if store.semantic_hints.iter().any(|entry| {
                        entry
                            .buffer
                            .upgrade()
                            .is_some_and(|stored| stored.entity_id() == buffer.entity_id())
                            && entry.server_id == *server_id
                            && entry.version == version
                            && entry.runnables == retained
                    }) {
                        return;
                    }
                    let previous = store.semantic_hints.len();
                    store.semantic_hints.retain(|entry| {
                        entry.buffer.upgrade().is_some_and(|stored| {
                            stored.entity_id() != buffer.entity_id()
                                || entry.server_id != *server_id
                        })
                    });
                    let origin = buffer.read(cx).file().map(|file| ProjectPath {
                        worktree_id: file.worktree_id(cx),
                        path: file.path().clone(),
                    });
                    let has_runnables = !retained.is_empty();
                    if let Some(origin) = origin.filter(|_| has_runnables) {
                        if store.semantic_hints.len() == MAX_RUST_TEST_SEMANTIC_BUFFERS {
                            store.semantic_hints.remove(0);
                        }
                        store.semantic_hints.push(RustSemanticBufferHints {
                            buffer: buffer.downgrade(),
                            origin,
                            version,
                            server_id: *server_id,
                            runnables: retained,
                        });
                    }
                    if (has_runnables || store.semantic_hints.len() != previous)
                        && store.semantic_request_generation != Some(store.generation)
                    {
                        store.schedule_invalidation(cx);
                    }
                }
                LspStoreEvent::LanguageServerRemoved(server_id) => {
                    let previous = store.semantic_hints.len();
                    store
                        .semantic_hints
                        .retain(|entry| entry.server_id != *server_id);
                    if store.semantic_hints.len() != previous {
                        store.schedule_invalidation(cx);
                    }
                }
                _ => {}
            }));
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn current_semantic_hint_count(&self, cx: &App) -> usize {
        self.semantic_hints
            .iter()
            .filter(|entry| {
                entry
                    .buffer
                    .upgrade()
                    .is_some_and(|buffer| buffer.read(cx).version() == entry.version)
            })
            .map(|entry| entry.runnables.len())
            .sum()
    }

    pub fn reconnected_to_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, RustTestProviderStoreMode::Remote { .. }) || !self.disconnected {
            return;
        }
        self.disconnected = false;
        cx.emit(RustTestProviderStoreEvent::Invalidated);
    }

    pub fn disconnected_from_host(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.mode, RustTestProviderStoreMode::Remote { .. }) {
            return;
        }
        if self.disconnected {
            return;
        }
        self.disconnected = true;
        self.unsupported_remote_discovery = None;
        self.connection_epoch = self.connection_epoch.wrapping_add(1);
        self.unshared();
        self.generation = self.generation.wrapping_add(1);
        self.active_remote_request_id = None;
        if let Some(abort) = self.active_remote_request_abort.take() {
            abort.abort();
        }
        self.actions.clear();
        cx.emit(RustTestProviderStoreEvent::Invalidated);
    }

    pub fn refresh(&mut self, cx: &mut Context<Self>) -> Task<Result<StructuredProviderSnapshot>> {
        self.refresh_scoped(None, cx)
    }

    fn refresh_scoped(
        &mut self,
        allowed_worktrees: Option<HashSet<WorktreeId>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<StructuredProviderSnapshot>> {
        if self.disconnected {
            return Task::ready(Err(anyhow!("The Rust test discovery host is disconnected")));
        }
        // This refresh incorporates queued input changes; a pending timer must not supersede it.
        self.unsupported_remote_discovery = None;
        self.invalidation_task = Task::ready(());
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        let connection_epoch = self.connection_epoch;
        self.semantic_request_generation = None;
        match &self.mode {
            RustTestProviderStoreMode::Remote {
                project_id,
                client,
                structured_store,
            } => {
                let project_id = *project_id;
                let client = client.clone();
                let structured_store = structured_store.clone();
                self.active_remote_request_id = Some(generation);
                let previous_retirement = self
                    .active_remote_request_retirement
                    .take()
                    .map(|(_, retirement)| retirement);
                let (retirement_sender, retirement_receiver) = futures::channel::oneshot::channel();
                let retirement = cx
                    .spawn({
                        let client = client.clone();
                        let previous_retirement = previous_retirement.clone();
                        async move |this, cx| {
                            let result = async {
                                let pending = retirement_receiver
                                    .await
                                    .context("Rust discovery retirement signal was lost")?;
                                // Unissued replacements still preserve their predecessor's retirement.
                                if let Some(retirement) = previous_retirement {
                                    retirement.await.map_err(|error| error.cloned())?;
                                }
                                if pending {
                                    client
                                        .request(proto::CancelRustTestDiscovery {
                                            project_id,
                                            request_id: generation,
                                        })
                                        .await
                                        .map_err(map_remote_discovery_error)?;
                                }
                                anyhow::Ok(())
                            }
                            .await;
                            if let Err(error) = this.update(cx, |store, _| {
                                if store
                                    .active_remote_request_retirement
                                    .as_ref()
                                    .is_some_and(|(request_id, _)| *request_id == generation)
                                {
                                    store.active_remote_request_retirement = None;
                                    if store.active_remote_request_id == Some(generation) {
                                        store.active_remote_request_id = None;
                                        store.active_remote_request_abort = None;
                                    }
                                }
                            }) {
                                log::debug!(
                                    "Rust discovery owner was released during retirement: {error:#}"
                                );
                            }
                            result.map_err(Arc::new)
                        }
                    })
                    .shared();
                // Retirement must finish even if a replacement waiting for its Ack is dropped.
                cx.spawn({
                    let retirement = retirement.clone();
                    async move |_, _| {
                        if let Err(error) = retirement.await {
                            log::warn!("Rust discovery cancellation could not retire its host request: {error:#}");
                        }
                    }
                })
                .detach();
                self.active_remote_request_retirement = Some((generation, retirement));
                let mut cancellation = RemoteDiscoveryCancellation {
                    retirement: Some(retirement_sender),
                    pending: false,
                };
                let worktree_ids = self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .map(|worktree| worktree.read(cx).id())
                    .filter(|worktree_id| {
                        allowed_worktrees
                            .as_ref()
                            .is_none_or(|allowed| allowed.contains(worktree_id))
                    })
                    .map(|worktree_id| worktree_id.to_proto())
                    .collect();
                let (abort_handle, abort_registration) = AbortHandle::new_pair();
                if let Some(previous) = self.active_remote_request_abort.replace(abort_handle) {
                    previous.abort();
                }
                cx.spawn(async move |this, cx| {
                    let operation = async {
                        if let Some(retirement) = previous_retirement {
                            retirement.await.map_err(|error| error.cloned())?;
                        }
                        cancellation.start();
                        let response = client
                            .request(proto::GetRustTestDiscovery {
                                project_id,
                                request_id: generation,
                                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                                worktree_ids,
                            })
                            .await;
                        cancellation.complete();
                        let response = response.map_err(map_remote_discovery_error)?;
                        if response.protocol_version != RUST_TEST_DISCOVERY_PROTOCOL_VERSION {
                            bail!("Rust test discovery protocol mismatch");
                        }
                        if response.request_id != generation {
                            bail!("Rust test discovery response identity mismatch");
                        }
                        if response.status == proto::StructuredProviderStatus::Mismatch as i32 {
                            bail!(
                                "{}",
                                response.diagnostic.as_deref().unwrap_or(
                                    "The project host does not support this Rust test protocol"
                                )
                            );
                        }
                        structured_store
                            .update(cx, |store, cx| {
                                store.refresh_provider(Self::provider_id(), cx)
                            })
                            .await
                    };
                    let result = Abortable::new(operation, abort_registration).await;
                    this.update(cx, |store, _| {
                        if store.active_remote_request_id == Some(generation) {
                            store.active_remote_request_id = None;
                            store.active_remote_request_abort = None;
                        }
                        if store.disconnected || store.connection_epoch != connection_epoch {
                            bail!("The Rust test discovery host disconnected during refresh");
                        }
                        if store.generation != generation {
                            return Err(superseded_discovery_error());
                        }
                        let result = result
                            .map_err(|_| anyhow!("Rust test discovery refresh was cancelled"))?;
                        store.unsupported_remote_discovery = match &result {
                            Err(error)
                                if error.downcast_ref::<RustTestRemoteError>().is_some_and(
                                    |error| {
                                        error.kind
                                            == RustTestRemoteErrorKind::UnsupportedDiscoveryHost
                                    },
                                ) =>
                            {
                                Some((generation, connection_epoch))
                            }
                            _ => None,
                        };
                        result
                    })?
                })
            }

            RustTestProviderStoreMode::Local {
                runner,
                cargo_store,
                structured_store,
                environment,
            } => {
                let runner = runner.clone();
                let cargo_refresh = cargo_store.update(cx, |store, cx| {
                    store.refresh_scoped(allowed_worktrees.clone(), cx)
                });
                let structured_store = structured_store.clone();
                let environment = environment.clone();
                let worktree_store = self.worktree_store.clone();
                let lsp_store = self.lsp_store.as_ref().and_then(WeakEntity::upgrade);
                let cached_semantic_hints = self.semantic_hints.clone();
                cx.spawn(async move |this, cx| {
                    let (abort_handle, abort_registration) = AbortHandle::new_pair();
                    let trust_subscription = cx.update(|cx| {
                        let worktree_store = worktree_store.downgrade();
                        TrustedWorktrees::try_get_global(cx).map(|trust| {
                            cx.subscribe(&trust, move |_, event, _| {
                                if matches!(event, TrustedWorktreesEvent::Restricted(store, _) if *store == worktree_store) {
                                    abort_handle.abort();
                                }
                            })
                        })
                    });
                    let collection = async {
                        let cargo_snapshot =
                            scoped_cargo_snapshot(cargo_refresh.await?, allowed_worktrees.as_ref());
                        if let Some(lsp_store) = &lsp_store {
                            let requests = cx.update(|cx| {
                                semantic_runnable_requests(
                                    lsp_store,
                                    &worktree_store,
                                    &cached_semantic_hints,
                                    allowed_worktrees.as_ref(),
                                    cx,
                                )
                            });
                            if !requests.is_empty() {
                                this.update(cx, |store, _| {
                                    store.semantic_request_generation = Some(generation);
                                })?;
                                let tasks = cx.update(|cx| {
                                    requests
                                        .into_iter()
                                        .map(|(buffer, server_id)| {
                                            let buffer_id = buffer.read(cx).remote_id();
                                            let requested_version = buffer.read(cx).version();
                                            lsp_store.update(cx, |store, cx| {
                                                store.request_lsp(
                                                    buffer,
                                                    LanguageServerToQuery::Other(server_id),
                                                    GetLspRunnables {
                                                        buffer_id,
                                                        position: None,
                                                        requested_version,
                                                    },
                                                    cx,
                                                )
                                            })
                                        })
                                        .collect::<Vec<_>>()
                                });
                                for task in tasks {
                                    match task
                                        .with_timeout(
                                            Duration::from_secs(10),
                                            &cx.background_executor(),
                                        )
                                        .await
                                    {
                                        Ok(Ok(_)) => {}
                                        Ok(Err(error)) => {
                                            log::debug!("Rust test runnable lookup failed: {error:#}");
                                        }
                                        Err(error) => {
                                            log::debug!("Rust test runnable lookup timed out: {error:?}");
                                        }
                                    }
                                }
                                this.update(cx, |store, _| {
                                    if store.semantic_request_generation == Some(generation) {
                                        store.semantic_request_generation = None;
                                    }
                                })?;
                            }
                        }
                        let prepared = cx.update(|cx| {
                            prepare_discovery_requests(
                                &cargo_snapshot,
                                &worktree_store,
                                &environment,
                                generation,
                                cx,
                            )
                        });
                        let mut captures = Vec::new();
                        let mut failures = Vec::new();
                        for (workspace, mut request, environment) in prepared {
                            request.environment = environment.await;
                            match runner.discover(request).await {
                                Ok(capture) => captures.push((workspace, capture)),
                                Err(error) => failures.push(RustTestDiscoveryFailure::from_error(&error)),
                            }
                        }
                        anyhow::Ok((cargo_snapshot, captures, failures))
                    };
                    let (cargo_snapshot, captures, failures) = Abortable::new(collection, abort_registration)
                        .await
                        .map_err(|_| {
                            proto::ErrorCode::Forbidden
                                .with_tag("reason", "rust_test_trust_revoked")
                                .message("Rust test discovery trust was revoked during collection".into())
                        })??;
                    drop(trust_subscription);
                    let (mut source_hints, semantic_hints) = this.update(cx, |store, _| {
                        (store.source_hints.clone(), store.semantic_hints.clone())
                    })?;
                    source_hints.extend(cx.update(|cx| {
                        project_semantic_hints(
                            &cargo_snapshot,
                            &semantic_hints,
                            &worktree_store,
                            cx,
                        )
                    }));
                    let projection = project_provider_projection(
                        &cargo_snapshot,
                        captures,
                        &source_hints,
                        failures,
                        DiscoveryGeneration(generation),
                    );
                    let mut snapshot = projection.snapshot;
                    let previous = structured_store.read_with(cx, |store, _| {
                        store.state().provider(&Self::provider_id()).cloned()
                    });
                    retain_stale_discovery(&mut snapshot, previous.as_ref());
                    cx.update(|cx| stamp_rust_test_experimental_availability(&mut snapshot, cx));
                    this.update(cx, |store, _| {
                        if store.generation != generation {
                            return Err(superseded_discovery_error());
                        }
                        Ok(())
                    })??;
                    structured_store.update(cx, |store, cx| {
                        let project_generation = store.state().project_generation();
                        store.state_mut().apply_discovery(
                            project_generation,
                            snapshot.clone(),
                            allowed_worktrees.as_ref(),
                        )?;
                        cx.emit(StructuredExecutionStoreEvent::Changed(Self::provider_id()));
                        anyhow::Ok(())
                    })?;
                    this.update(cx, |store, cx| {
                        store.actions = projection.actions;
                        store.authorized_remote_runs.retain(|_, authorization| {
                            authorization.discovery_generation == DiscoveryGeneration(generation)
                        });
                        cx.emit(RustTestProviderStoreEvent::Changed(DiscoveryGeneration(
                            generation,
                        )))
                    })?;
                    Ok(snapshot)
                })
            }
        }
    }

    fn schedule_invalidation(&mut self, cx: &mut Context<Self>) {
        cx.emit(RustTestProviderStoreEvent::Invalidated);
        self.invalidation_task = cx.spawn(async move |this, cx| {
            cx.background_executor()
                .timer(Duration::from_millis(150))
                .await;
            if let Err(error) = this.update(cx, |store, cx| {
                let refresh = store.refresh(cx);
                store.refresh_task = cx.spawn(async move |_, _| {
                    if let Err(error) = refresh.await {
                        log::warn!("Rust test discovery refresh failed: {error:#}");
                    }
                });
            }) {
                log::debug!("Rust test provider was dropped before refresh: {error:#}");
            }
        });
    }

    fn refresh_for_remote_request(
        &mut self,
        allowed_worktrees: HashSet<WorktreeId>,
        cx: &mut Context<Self>,
    ) -> Shared<RustTestDiscoveryRefreshTask> {
        let generation = self.generation;
        self.pending_discovery_refreshes
            .retain(|_, (captured, refresh)| {
                *captured == generation && refresh.upgrade().is_some()
            });
        let mut scope = allowed_worktrees.iter().copied().collect::<Vec<_>>();
        scope.sort_unstable();
        if let Some((_, refresh)) = self.pending_discovery_refreshes.get(&scope)
            && let Some(refresh) = refresh.upgrade()
        {
            return refresh;
        }
        // Concurrent peers share collection, but retain independent cancellation handles.
        let refresh = self.refresh_scoped(Some(allowed_worktrees), cx);
        let refresh = cx
            .spawn(async move |_, _| {
                refresh
                    .await
                    .map(Arc::new)
                    .map_err(tag_remote_rust_test_error)
                    .map_err(Arc::new)
            })
            .shared();
        if let Some(pending) = refresh.downgrade() {
            self.pending_discovery_refreshes
                .insert(scope, (self.generation, pending));
        }
        refresh
    }

    async fn handle_get_discovery(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetRustTestDiscovery>,
        mut cx: AsyncApp,
    ) -> Result<proto::RustTestDiscoveryResponse> {
        let request = envelope.payload;
        if request.protocol_version != RUST_TEST_DISCOVERY_PROTOCOL_VERSION {
            return Ok(proto::RustTestDiscoveryResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                request_id: request.request_id,
                discovery_generation: 0,
                status: proto::StructuredProviderStatus::Mismatch as i32,
                diagnostic: Some("Rust test discovery protocol mismatch".to_string()),
            });
        }
        if request.worktree_ids.len() > MAX_RUST_TEST_REQUEST_WORKTREES {
            bail!("Rust test discovery worktree scope exceeds the supported limit");
        }
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let key = (peer_id, request.request_id);
        let allowed = request
            .worktree_ids
            .into_iter()
            .map(WorktreeId::from_proto)
            .collect();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let proceed = this.update(&mut cx, |store, _| {
            if store.cancelled_remote_requests.remove(&key) {
                return Ok(false);
            }
            if store.active_remote_requests.contains_key(&key) {
                bail!("Duplicate pending Rust test discovery request");
            }
            if store.active_remote_requests.len() >= MAX_PENDING_RUST_TEST_CANCELLATIONS {
                bail!("Too many pending Rust test discovery requests");
            }
            store.active_remote_requests.insert(key, abort_handle);
            anyhow::Ok(true)
        })?;
        if !proceed {
            bail!("Rust test discovery request was cancelled");
        }
        let refresh = this.update(&mut cx, |store, cx| {
            store.refresh_for_remote_request(allowed, cx)
        });
        let result = Abortable::new(refresh, abort_registration).await;
        this.update(&mut cx, |store, _| {
            store.active_remote_requests.remove(&key);
            store.cancelled_remote_requests.remove(&key);
        });
        let snapshot = result
            .map_err(|_| anyhow!("Rust test discovery request was cancelled"))?
            .map_err(|error| error.cloned())?;
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release, completed)) =
            this.update(&mut cx, |store, _| store.discovery_reply_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow!("Rust discovery reply start receiver was dropped"))?;
            release
                .await
                .map_err(|_| anyhow!("Rust discovery reply release was dropped"))?;
            if completed.send(()).is_err() {
                log::debug!("Rust discovery reply completion receiver was dropped");
            }
        }
        Ok(proto::RustTestDiscoveryResponse {
            protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            request_id: request.request_id,
            discovery_generation: snapshot.discovery_generation.0,
            status: provider_status_to_proto(snapshot.status),
            diagnostic: snapshot.diagnostic.clone(),
        })
    }

    async fn handle_cancel_discovery(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CancelRustTestDiscovery>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        this.update(&mut cx, |store, _| {
            let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
            let key = (peer_id, envelope.payload.request_id);
            if let Some(handle) = store.active_remote_requests.remove(&key) {
                handle.abort();
                if store.active_remote_requests.is_empty() {
                    store.pending_discovery_refreshes.clear();
                }
            } else if store.cancelled_remote_requests.len() < MAX_PENDING_RUST_TEST_CANCELLATIONS {
                store.cancelled_remote_requests.insert(key);
            } else {
                log::warn!("Ignored excess pending Rust test cancellation");
            }
        });
        Ok(proto::Ack {})
    }

    async fn handle_resolve_action(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ResolveRustTestAction>,
        mut cx: AsyncApp,
    ) -> Result<proto::RustTestActionPlanResponse> {
        let mut request = envelope.payload;
        ensure_rust_test_protocol(request.protocol_version).map_err(tag_remote_rust_test_error)?;
        validate_identifier_field("Rust test node ID", &request.node_id)?;
        if request.project_id != proto::REMOTE_SERVER_PROJECT_ID
            && request.authorization_generation == 0
        {
            return Err(proto::ErrorCode::Internal
                .with_tag("reason", "rust_tools_unsupported")
                .message("The collaboration server does not support authorization-bound Rust test actions".to_string())
                .into());
        }
        if request.worktree_ids.len() > MAX_RUST_TEST_REQUEST_WORKTREES {
            bail!("Rust test action worktree scope exceeds the supported limit");
        }
        let allowed = request
            .worktree_ids
            .iter()
            .copied()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        let node_id = StructuredNodeId(request.node_id.clone());
        let action = action_from_proto(request.action)?;
        let sender_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        if let Some((project_id, client)) =
            this.read_with(&cx, |store, _| store.remote_action_client())
        {
            let authorization_generation = request.authorization_generation;
            let discovery_generation = request.discovery_generation;
            request.project_id = project_id;
            let response =
                Self::forward_remote_action(&this, sender_id, &allowed, client, request, &mut cx)
                    .await?;
            if response.authorization_generation != authorization_generation {
                return Err(proto::ErrorCode::Internal
                    .with_tag("reason", "rust_tools_unsupported")
                    .message(
                        "The upstream host does not support authorization-bound Rust test actions"
                            .to_string(),
                    )
                    .into());
            }
            let plan = decode_action_plan(response)?;
            this.update(&mut cx, |store, cx| {
                let worktree_id = match (&plan, action) {
                    (
                        RustTestActionPlan::Task {
                            discovery_generation: generation,
                            scope_node_ids,
                            worktree_id,
                            ..
                        },
                        RustTestAction::Run,
                    ) if generation.0 == discovery_generation
                        && scope_node_ids == std::slice::from_ref(&node_id) =>
                    {
                        *worktree_id
                    }
                    (RustTestActionPlan::Debug { worktree_id, .. }, RustTestAction::Debug) => {
                        *worktree_id
                    }
                    _ => bail!(
                        "The upstream Rust test action plan does not match the selected action"
                    ),
                };
                if !allowed.contains(&worktree_id) {
                    bail!("The upstream Rust test action is outside the visible worktree scope");
                }
                store.ensure_worktree_trusted(worktree_id, cx)?;
                store.authorize_remote_plan(sender_id, authorization_generation, &plan)?;
                Ok(proto::RustTestActionPlanResponse {
                    protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                    plan_json: encode_action_plan(&plan)?,
                    authorization_generation,
                })
            })
        } else {
            let plan_json = this.update(&mut cx, |store, cx| {
                if request.discovery_generation != store.generation {
                    bail!("The selected Rust test belongs to a stale discovery generation");
                }
                let descriptor = store
                    .actions
                    .get(&node_id)
                    .context("The selected Rust test is no longer executable")?;
                if !allowed.contains(&descriptor.worktree_id) {
                    bail!("The selected Rust test is outside the visible worktree scope");
                }
                let plan = store.plan_local_action(
                    &node_id,
                    DiscoveryGeneration(request.discovery_generation),
                    action,
                    cx,
                )?;
                store.authorize_remote_plan(sender_id, request.authorization_generation, &plan)?;
                encode_action_plan(&plan)
            })?;
            Ok(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json,
                authorization_generation: request.authorization_generation,
            })
        }
    }

    async fn handle_update_run(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::UpdateRustTestRun>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let request = envelope.payload;
        ensure_rust_test_protocol(request.protocol_version).map_err(tag_remote_rust_test_error)?;
        if request.worktree_ids.len() > MAX_RUST_TEST_REQUEST_WORKTREES {
            bail!("Rust test lifecycle worktree scope exceeds the supported limit");
        }
        let allowed = request
            .worktree_ids
            .iter()
            .copied()
            .map(WorktreeId::from_proto)
            .collect::<HashSet<_>>();
        let run_id = StructuredRunId(request.run_id.clone());
        let scope_node_ids = request
            .scope_node_ids
            .iter()
            .map(|node_id| StructuredNodeId(node_id.clone()))
            .collect::<Vec<_>>();
        let event = lifecycle_from_proto(&request)?;
        let peer_id = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let key = (peer_id, run_id.clone());
        this.update(&mut cx, |store, cx| {
            store.validate_remote_run(peer_id, &request, &allowed, cx)
        })?;
        if let Some((project_id, client)) =
            this.read_with(&cx, |store, _| store.remote_action_client())
        {
            let mut forwarded = request.clone();
            forwarded.project_id = project_id;
            forwarded.diagnostic = match &event.state {
                StructuredTaskState::SpawnError { message } => Some(message.clone()),
                _ => None,
            };
            let response =
                Self::forward_remote_action(&this, peer_id, &allowed, client, forwarded, &mut cx)
                    .await?;
            let structured_store = this.update(&mut cx, |store, cx| {
                store.validate_remote_run(peer_id, &request, &allowed, cx)?;
                match &store.mode {
                    RustTestProviderStoreMode::Remote {
                        structured_store, ..
                    }
                    | RustTestProviderStoreMode::Local {
                        structured_store, ..
                    } => anyhow::Ok(structured_store.clone()),
                }
            })?;
            structured_store
                .update(&mut cx, |store, cx| {
                    store.refresh_provider(Self::provider_id(), cx)
                })
                .await?;
            this.update(&mut cx, |store, cx| {
                store.validate_remote_run(peer_id, &request, &allowed, cx)?;
                if event.state.is_terminal() {
                    store.authorized_remote_runs.remove(&key);
                }
                anyhow::Ok(())
            })?;
            return Ok(response);
        }
        let structured_store = this.read_with(&cx, |store, _| match &store.mode {
            RustTestProviderStoreMode::Local {
                structured_store, ..
            }
            | RustTestProviderStoreMode::Remote {
                structured_store, ..
            } => structured_store.clone(),
        });
        structured_store.update(&mut cx, |store, cx| {
            let provider_id = Self::provider_id();
            let current_run = store
                .state()
                .provider(&provider_id)
                .and_then(|provider| provider.current_run.as_ref());
            if current_run.is_none_or(|run| run.id != run_id) {
                if !matches!(event.state, StructuredTaskState::Queued) {
                    bail!("Remote Rust test lifecycle did not begin with a queued state");
                }
                let project_generation = store.state().project_generation();
                store.state_mut().begin_run(
                    project_generation,
                    &provider_id,
                    crate::structured_execution::StructuredRun::new(
                        run_id.clone(),
                        DiscoveryGeneration(request.discovery_generation),
                        scope_node_ids.clone(),
                    ),
                )?;
            }
            store.apply_task_lifecycle(&provider_id, &run_id, &event, cx)?;
            anyhow::Ok(())
        })?;
        if event.state.is_terminal() {
            this.update(&mut cx, |store, _| {
                store.authorized_remote_runs.remove(&key);
            });
        }
        Ok(proto::Ack {})
    }
}

fn stamp_rust_test_experimental_availability(snapshot: &mut StructuredProviderSnapshot, cx: &App) {
    for node in &mut snapshot.nodes {
        node.experimental_feature_enabled = node
            .experimental_feature
            .as_ref()
            .map(|feature| cx.has_flag_by_name(feature));
    }
}

fn visible_worktree_ids(worktree_store: &Entity<WorktreeStore>, cx: &App) -> Vec<u64> {
    worktree_store
        .read(cx)
        .visible_worktrees(cx)
        .map(|worktree| worktree.read(cx).id().to_proto())
        .take(MAX_RUST_TEST_REQUEST_WORKTREES)
        .collect()
}

fn ensure_rust_test_protocol(protocol_version: u32) -> Result<()> {
    if protocol_version != RUST_TEST_DISCOVERY_PROTOCOL_VERSION {
        return Err(anyhow!(RustTestRemoteError {
            kind: RustTestRemoteErrorKind::UnsupportedActionHost
        })
        .context("Rust test action protocol mismatch"));
    }
    Ok(())
}

fn validate_identifier_field(name: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_RUST_TEST_FIELD_BYTES || value.contains('\0') {
        bail!("{name} is invalid or exceeds the supported limit");
    }
    Ok(())
}

fn action_to_proto(action: RustTestAction) -> i32 {
    (match action {
        RustTestAction::Run => proto::RustTestActionKind::Run,
        RustTestAction::Debug => proto::RustTestActionKind::Debug,
    }) as i32
}

fn action_from_proto(action: i32) -> Result<RustTestAction> {
    match proto::RustTestActionKind::from_i32(action) {
        Some(proto::RustTestActionKind::Run) => Ok(RustTestAction::Run),
        Some(proto::RustTestActionKind::Debug) => Ok(RustTestAction::Debug),
        _ => bail!("Unsupported Rust test action"),
    }
}

fn encode_action_plan(plan: &RustTestActionPlan) -> Result<String> {
    let wire_plan = match plan {
        RustTestActionPlan::Task {
            run_id,
            discovery_generation,
            scope_node_ids,
            worktree_id,
            template,
        } => RustTestActionWirePlan::Task {
            run_id: run_id.0.clone(),
            discovery_generation: discovery_generation.0,
            scope_node_ids: scope_node_ids
                .iter()
                .map(|node_id| node_id.0.clone())
                .collect(),
            worktree_id: worktree_id.to_proto(),
            template: template.clone(),
        },
        RustTestActionPlan::Debug {
            worktree_id,
            scenario,
        } => {
            let Some(BuildTaskDefinition::Template {
                task_template,
                locator_name,
            }) = &scenario.build
            else {
                bail!("Rust test debug plan has no Cargo build template");
            };
            if locator_name.as_deref() != Some("rust-cargo-locator") {
                bail!("Rust test debug plan has an unsupported locator");
            }
            RustTestActionWirePlan::Debug {
                worktree_id: worktree_id.to_proto(),
                build_template: task_template.clone(),
            }
        }
    };
    let serialized = serde_json::to_string(&wire_plan)?;
    if serialized.len() > MAX_RUST_TEST_ACTION_PLAN_BYTES {
        bail!("Rust test action plan exceeds the supported limit");
    }
    Ok(serialized)
}

fn decode_action_plan(response: proto::RustTestActionPlanResponse) -> Result<RustTestActionPlan> {
    ensure_rust_test_protocol(response.protocol_version)?;
    if response.plan_json.len() > MAX_RUST_TEST_ACTION_PLAN_BYTES {
        bail!("Remote Rust test action plan exceeds the supported limit");
    }
    let wire_plan = serde_json::from_str::<RustTestActionWirePlan>(&response.plan_json)
        .context("Remote Rust test action plan was malformed")?;
    match wire_plan {
        RustTestActionWirePlan::Task {
            run_id,
            discovery_generation,
            scope_node_ids,
            worktree_id,
            template,
        } => {
            validate_identifier_field("Rust test run ID", &run_id)?;
            if scope_node_ids.is_empty() || scope_node_ids.len() > MAX_RUST_TEST_RERUNS {
                bail!("Remote Rust test action scope exceeds the supported limit");
            }
            for node_id in &scope_node_ids {
                validate_identifier_field("Rust test node ID", node_id)?;
            }
            validate_action_template(&template)?;
            Ok(RustTestActionPlan::Task {
                run_id: StructuredRunId(run_id),
                discovery_generation: DiscoveryGeneration(discovery_generation),
                scope_node_ids: scope_node_ids.into_iter().map(StructuredNodeId).collect(),
                worktree_id: WorktreeId::from_proto(worktree_id),
                template,
            })
        }
        RustTestActionWirePlan::Debug {
            worktree_id,
            build_template,
        } => {
            validate_action_template(&build_template)?;
            Ok(RustTestActionPlan::Debug {
                worktree_id: WorktreeId::from_proto(worktree_id),
                scenario: rust_test_debug_scenario_from_build(build_template),
            })
        }
    }
}

fn validate_action_template(template: &TaskTemplate) -> Result<()> {
    task::validate_experimental_feature_requirements(&template.required_experimental_features)?;
    if template.experimental_policy_version != task::EXPERIMENTAL_TASK_POLICY_VERSION {
        return Err(anyhow!(RustTestRemoteError {
            kind: RustTestRemoteErrorKind::UnsupportedActionHost
        })
        .context("Rust test generated Task admission policy mismatch"));
    }
    if template.command != "cargo"
        || template.args.len() > MAX_RUST_TEST_ACTION_ARGUMENTS
        || !template.env.is_empty()
        || !template.hooks.is_empty()
    {
        bail!("Remote Rust test action plan contains unsupported task fields");
    }
    validate_identifier_field("Rust test task label", &template.label)?;
    let mut bytes = template.label.len() + template.command.len();
    for argument in &template.args {
        if argument.len() > MAX_RUST_TEST_FIELD_BYTES || argument.contains('\0') {
            bail!("Remote Rust test action argument exceeds the supported limit");
        }
        bytes = bytes.saturating_add(argument.len());
    }
    if let Some(cwd) = &template.cwd {
        if cwd.len() > MAX_RUST_TEST_FIELD_BYTES
            || cwd.contains('\0')
            || cwd.split('/').any(|component| component == "..")
        {
            bail!("Remote Rust test action working directory is unsafe");
        }
        bytes = bytes.saturating_add(cwd.len());
    }
    if bytes > MAX_RUST_TEST_ACTION_PLAN_BYTES {
        bail!("Remote Rust test action plan exceeds the supported limit");
    }
    if let Some(delimiter) = template.args.iter().position(|argument| argument == "--") {
        if template.args[..delimiter]
            .iter()
            .any(|argument| argument == "--doc")
            && template.args[delimiter + 1..].iter().any(|argument| {
                argument == "--exact"
                    || argument.is_empty()
                    || argument.chars().any(char::is_whitespace)
            })
        {
            bail!("Rust doctest action requires a whitespace-safe uniquely selecting filter");
        }
    }
    Ok(())
}

fn lifecycle_to_proto(
    project_id: u64,
    discovery_generation: DiscoveryGeneration,
    run_id: &StructuredRunId,
    scope_node_ids: &[StructuredNodeId],
    event: &StructuredTaskLifecycleEvent,
    worktree_ids: Vec<u64>,
) -> proto::UpdateRustTestRun {
    let (state, terminal_id, exit_code, success, termination_confirmed, diagnostic) =
        match &event.state {
            StructuredTaskState::Queued => (
                proto::RustTestTaskState::Queued,
                None,
                None,
                false,
                false,
                None,
            ),
            StructuredTaskState::Running { terminal_id } => (
                proto::RustTestTaskState::Running,
                terminal_id.map(|terminal_id| terminal_id.0),
                None,
                false,
                false,
                None,
            ),
            StructuredTaskState::Completed {
                terminal_id,
                exit_code,
                success,
            } => (
                proto::RustTestTaskState::Completed,
                terminal_id.map(|terminal_id| terminal_id.0),
                *exit_code,
                *success,
                false,
                None,
            ),
            StructuredTaskState::SpawnError { .. } => (
                proto::RustTestTaskState::SpawnError,
                None,
                None,
                false,
                false,
                Some(REMOTE_RUST_TEST_SPAWN_FAILURE.to_string()),
            ),
            StructuredTaskState::Cancelled {
                terminal_id,
                termination_confirmed,
            } => (
                proto::RustTestTaskState::Cancelled,
                terminal_id.map(|terminal_id| terminal_id.0),
                None,
                false,
                *termination_confirmed,
                None,
            ),
        };
    proto::UpdateRustTestRun {
        project_id,
        protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: discovery_generation.0,
        run_id: run_id.0.clone(),
        scope_node_ids: scope_node_ids
            .iter()
            .map(|node_id| node_id.0.clone())
            .collect(),
        task_id: event.task_id.0.clone(),
        state: state as i32,
        terminal_id,
        exit_code,
        success,
        termination_confirmed,
        diagnostic,
        worktree_ids,
        authorization_generation: 0,
    }
}

fn lifecycle_from_proto(
    request: &proto::UpdateRustTestRun,
) -> Result<StructuredTaskLifecycleEvent> {
    validate_identifier_field("Rust test run ID", &request.run_id)?;
    validate_identifier_field("Rust test task ID", &request.task_id)?;
    if request.scope_node_ids.is_empty() || request.scope_node_ids.len() > MAX_RUST_TEST_RERUNS {
        bail!("Rust test lifecycle scope exceeds the supported limit");
    }
    for node_id in &request.scope_node_ids {
        validate_identifier_field("Rust test node ID", node_id)?;
    }
    let terminal_id = request.terminal_id.map(StructuredTerminalId);
    let state = match proto::RustTestTaskState::from_i32(request.state) {
        Some(proto::RustTestTaskState::Queued) => StructuredTaskState::Queued,
        Some(proto::RustTestTaskState::Running) => StructuredTaskState::Running { terminal_id },
        Some(proto::RustTestTaskState::Completed) => StructuredTaskState::Completed {
            terminal_id,
            exit_code: request.exit_code,
            success: request.success,
        },
        Some(proto::RustTestTaskState::SpawnError) => StructuredTaskState::SpawnError {
            message: REMOTE_RUST_TEST_SPAWN_FAILURE.to_string(),
        },
        Some(proto::RustTestTaskState::Cancelled) => StructuredTaskState::Cancelled {
            terminal_id,
            termination_confirmed: request.termination_confirmed,
        },
        _ => bail!("Unsupported remote Rust test lifecycle state"),
    };
    Ok(StructuredTaskLifecycleEvent {
        task_id: TaskId(request.task_id.clone()),
        state,
    })
}

fn tag_remote_rust_test_error(error: anyhow::Error) -> anyhow::Error {
    if let Some(remote_error) = error.downcast_ref::<RustTestRemoteError>() {
        let reason = match remote_error.kind {
            RustTestRemoteErrorKind::SupersededDiscovery => "rust_test_discovery_superseded",
            RustTestRemoteErrorKind::UnsupportedDiscoveryHost
            | RustTestRemoteErrorKind::UnsupportedActionHost => "rust_tools_unsupported",
        };
        proto::ErrorCode::Internal
            .with_tag("reason", reason)
            .message(error.to_string())
            .into()
    } else {
        error
    }
}

fn map_remote_action_error(error: anyhow::Error) -> anyhow::Error {
    let unsupported = error
        .downcast_ref::<proto::RpcError>()
        .is_some_and(|error| {
            error.error_tag("reason") == Some("rust_tools_unsupported")
                || error.raw_message() == "Rust test action protocol mismatch"
                || error.raw_message().contains("was not handled")
                || error
                    .raw_message()
                    .starts_with("no handler registered for ")
        });
    if unsupported {
        anyhow!(RustTestRemoteError {
            kind: RustTestRemoteErrorKind::UnsupportedActionHost,
        })
    } else {
        anyhow!("The Rust test action host rejected the request: {error}")
    }
}

fn rust_test_debug_capability(descriptor: &RustTestActionDescriptor) -> StructuredDebugCapability {
    let reason = if descriptor.scope != RustTestActionScope::Case {
        Some("Debug is available for individual Rust test cases only")
    } else {
        match descriptor.target_kind {
            Some(RustTestTargetKind::Doctest) => {
                Some("Cargo DAP does not support debugging doctests")
            }
            Some(RustTestTargetKind::Benchmark) => {
                Some("Cargo DAP does not support debugging an individual benchmark")
            }
            None => Some("The selected Rust test has no debuggable harness"),
            Some(_) => None,
        }
    };
    match reason {
        Some(reason) => StructuredDebugCapability::Unsupported(reason.to_string()),
        None => StructuredDebugCapability::Supported,
    }
}

fn plan_rust_test_action(
    descriptor: &RustTestActionDescriptor,
    node_id: &StructuredNodeId,
    discovery_generation: DiscoveryGeneration,
    run_number: u64,
    action: RustTestAction,
) -> Result<RustTestActionPlan> {
    let mut template = rust_test_task_template(descriptor);
    if action == RustTestAction::Run && descriptor.target_kind == Some(RustTestTargetKind::Doctest)
    {
        if descriptor.scope == RustTestActionScope::Case {
            // Edition 2024's merged rustdoc runner replaces ignored bodies with successful stubs.
            if !descriptor.doctest_ignored_run_supported
                && descriptor
                    .executable_args
                    .iter()
                    .any(|argument| matches!(argument.as_str(), "--include-ignored" | "--ignored"))
            {
                bail!(
                    "Selected ignored doctest Run is unavailable for edition 2024 or an unknown edition: rustdoc may report success without executing its body"
                );
            }
            if descriptor.executable_args.iter().any(|argument| {
                argument != &descriptor.label
                    && !matches!(
                        argument.as_str(),
                        "--exact" | "--nocapture" | "--include-ignored" | "--ignored"
                    )
            }) {
                bail!("Selected doctest has unsupported harness arguments");
            }
            let selector = uniquely_selecting_doctest_token(descriptor)?;
            let mut arguments = descriptor.cargo_args.clone();
            arguments.extend(["--".to_string(), selector]);
            arguments.extend(
                descriptor
                    .executable_args
                    .iter()
                    .filter(|argument| {
                        matches!(
                            argument.as_str(),
                            "--nocapture" | "--include-ignored" | "--ignored"
                        )
                    })
                    .cloned(),
            );
            template.args = arguments;
        }
        validate_action_template(&template)?;
    }
    match action {
        RustTestAction::Run => Ok(RustTestActionPlan::Task {
            run_id: StructuredRunId(format!(
                "rust-test-run-{}-{run_number}",
                discovery_generation.0
            )),
            discovery_generation,
            scope_node_ids: vec![node_id.clone()],
            worktree_id: descriptor.worktree_id,
            template,
        }),
        RustTestAction::Debug => {
            if let StructuredDebugCapability::Unsupported(reason) =
                rust_test_debug_capability(descriptor)
            {
                bail!("{reason}");
            }
            Ok(RustTestActionPlan::Debug {
                worktree_id: descriptor.worktree_id,
                scenario: rust_test_debug_scenario(template)?,
            })
        }
    }
}

fn uniquely_selecting_doctest_token(descriptor: &RustTestActionDescriptor) -> Result<String> {
    let names = descriptor
        .doctest_case_names
        .as_ref()
        .context("Selected doctest requires complete supported same-harness discovery")?;
    if names.is_empty()
        || names.len() > MAX_RUST_TEST_CASES
        || names.iter().any(|name| {
            name.is_empty() || name.len() > MAX_RUST_TEST_FIELD_BYTES || name.contains('\0')
        })
        || !names.iter().any(|name| name == &descriptor.label)
    {
        bail!("Selected doctest discovery is invalid or exceeds the supported limit");
    }
    // Rustdoc splits test-args on whitespace, including a quoted full case name.
    let mut compared_bytes = 0usize;
    for token in descriptor.label.split_whitespace().rev().take(8) {
        if token.starts_with('-') {
            continue;
        }
        let mut selected_only = false;
        let mut ambiguous = false;
        for name in names.iter() {
            compared_bytes = compared_bytes.saturating_add(name.len());
            if compared_bytes > MAX_RUST_TEST_LISTING_BYTES * 4 {
                bail!("Selected doctest uniqueness check exceeds the supported limit");
            }
            if name.contains(token) {
                if selected_only || name != &descriptor.label {
                    ambiguous = true;
                    break;
                }
                selected_only = true;
            }
        }
        if selected_only && !ambiguous {
            return Ok(token.to_string());
        }
    }
    bail!(
        "Selected doctest has no supported uniquely selecting filter; run its harness explicitly instead"
    )
}

fn scoped_cargo_snapshot(
    mut snapshot: CargoWorkspaceSnapshot,
    allowed_worktrees: Option<&HashSet<WorktreeId>>,
) -> CargoWorkspaceSnapshot {
    let Some(allowed_worktrees) = allowed_worktrees else {
        return snapshot;
    };
    snapshot
        .workspaces
        .retain(|workspace| allowed_worktrees.contains(&workspace.key.root.worktree_id));
    snapshot
        .failures
        .retain(|failure| allowed_worktrees.contains(&failure.manifest_path.worktree_id));
    snapshot
}

fn retain_stale_discovery(
    snapshot: &mut StructuredProviderSnapshot,
    previous: Option<&StructuredProviderSnapshot>,
) {
    let Some(previous) = previous else {
        return;
    };
    if snapshot.status != StructuredProviderStatus::Error || previous.nodes.is_empty() {
        return;
    }
    snapshot.nodes = previous.nodes.clone();
    snapshot.status = StructuredProviderStatus::Stale;
    snapshot.partial = true;
    if snapshot.diagnostic.is_none() {
        snapshot.diagnostic = Some("Rust test discovery failed; showing stale results".to_string());
    }
}

type PreparedDiscoveryRequest = (
    CargoWorkspaceModel,
    RustTestDiscoveryRequest,
    Shared<Task<Option<HashMap<String, String>>>>,
);

fn prepare_discovery_requests(
    snapshot: &CargoWorkspaceSnapshot,
    worktree_store: &Entity<WorktreeStore>,
    environment: &Entity<ProjectEnvironment>,
    generation: u64,
    cx: &mut App,
) -> Vec<PreparedDiscoveryRequest> {
    snapshot
        .workspaces
        .iter()
        .filter_map(|workspace| {
            let manifest = workspace.root_manifest.as_ref().or_else(|| {
                workspace
                    .members
                    .first()
                    .map(|package| &package.manifest_path)
            })?;
            let manifest_path = worktree_store.read(cx).absolutize(manifest, cx)?;
            let working_directory = manifest_path.parent()?.to_path_buf();
            let worktree = worktree_store
                .read(cx)
                .worktree_for_id(manifest.worktree_id, cx)?;
            let environment = environment.update(cx, |environment, cx| {
                environment.worktree_environment(worktree, cx)
            });
            let doctest_targets = workspace
                .members
                .iter()
                .flat_map(|package| {
                    package.targets.iter().filter_map(|target| {
                        (target.kind == CargoTargetKind::Library)
                            .then(|| (package.name.clone(), target.name.clone()))
                    })
                })
                .collect();
            Some((
                workspace.clone(),
                RustTestDiscoveryRequest {
                    workspace_root: workspace.key.root.clone(),
                    generation,
                    limits: RustTestProtocolLimits::default(),
                    manifest_path: Some(manifest_path),
                    working_directory: Some(working_directory),
                    environment: None,
                    doctest_targets,
                },
                environment,
            ))
        })
        .collect()
}

struct RustTestProviderProjection {
    snapshot: StructuredProviderSnapshot,
    actions: HashMap<StructuredNodeId, RustTestActionDescriptor>,
}

fn semantic_runnable_requests(
    lsp_store: &Entity<LspStore>,
    worktree_store: &Entity<WorktreeStore>,
    cached_hints: &[RustSemanticBufferHints],
    allowed_worktrees: Option<&HashSet<WorktreeId>>,
    cx: &mut App,
) -> Vec<(Entity<Buffer>, LanguageServerId)> {
    let rust_servers = lsp_store
        .read(cx)
        .language_server_statuses()
        .filter_map(|(server_id, status)| (status.name == RUST_ANALYZER_NAME).then_some(server_id))
        .collect::<HashSet<_>>();
    if rust_servers.is_empty() {
        return Vec::new();
    }
    let worktrees = worktree_store
        .read(cx)
        .visible_worktrees(cx)
        .map(|worktree| (worktree.read(cx).id(), worktree))
        .collect::<BTreeMap<_, _>>();
    let buffer_store = lsp_store.read(cx).buffer_store();
    let buffers = buffer_store.read(cx).buffers().collect::<Vec<_>>();
    let mut requests = Vec::new();
    for buffer in buffers {
        let Some((origin, version)) = buffer.read_with(cx, |buffer, cx| {
            buffer.file().map(|file| {
                (
                    ProjectPath {
                        worktree_id: file.worktree_id(cx),
                        path: file.path().clone(),
                    },
                    buffer.version(),
                )
            })
        }) else {
            continue;
        };
        if origin.path.extension() != Some("rs")
            || allowed_worktrees.is_some_and(|ids| !ids.contains(&origin.worktree_id))
            || !worktrees.get(&origin.worktree_id).is_some_and(|worktree| {
                worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(origin.path.as_ref())
                    .is_some_and(|entry| !entry.is_private && !entry.is_external && !entry.is_fifo)
            })
        {
            continue;
        }
        let trusted = TrustedWorktrees::try_get_global(cx)
            .map(|trusted| {
                trusted.update(cx, |trusted, cx| {
                    trusted.can_trust(worktree_store, origin.worktree_id, cx)
                })
            })
            .unwrap_or(true);
        if !trusted {
            continue;
        }
        let server_ids = lsp_store.update(cx, |store, cx| {
            buffer.update(cx, |buffer, cx| {
                store.language_servers_for_local_buffer(buffer, cx)
            })
        });
        for server_id in server_ids {
            if !rust_servers.contains(&server_id)
                || cached_hints.iter().any(|entry| {
                    entry
                        .buffer
                        .upgrade()
                        .is_some_and(|stored| stored.entity_id() == buffer.entity_id())
                        && entry.server_id == server_id
                        && entry.version == version
                })
            {
                continue;
            }
            requests.push((buffer.clone(), server_id));
            if requests.len() == MAX_RUST_TEST_SEMANTIC_REFRESH_BUFFERS {
                return requests;
            }
        }
    }
    requests
}

fn project_semantic_hints(
    cargo_snapshot: &CargoWorkspaceSnapshot,
    semantic_hints: &[RustSemanticBufferHints],
    worktree_store: &Entity<WorktreeStore>,
    cx: &mut App,
) -> Vec<RustTestSourceHint> {
    let worktrees = worktree_store
        .read(cx)
        .visible_worktrees(cx)
        .map(|worktree| (worktree.read(cx).id(), worktree))
        .collect::<BTreeMap<_, _>>();
    let visible = |path: &ProjectPath, cx: &App| {
        worktrees.get(&path.worktree_id).is_some_and(|worktree| {
            worktree
                .read(cx)
                .snapshot()
                .entry_for_path(path.path.as_ref())
                .is_some_and(|entry| !entry.is_private && !entry.is_external && !entry.is_fifo)
        })
    };
    let mut hints = Vec::new();
    let mut identities = BTreeSet::new();
    for entry in semantic_hints {
        let trusted = TrustedWorktrees::try_get_global(cx)
            .map(|trusted| {
                trusted.update(cx, |trusted, cx| {
                    trusted.can_trust(worktree_store, entry.origin.worktree_id, cx)
                })
            })
            .unwrap_or(true);
        if !trusted {
            continue;
        }
        let Some(buffer) = entry.buffer.upgrade() else {
            continue;
        };
        if buffer.read(cx).version() != entry.version
            || !visible(&entry.origin, cx)
            || buffer.read(cx).file().is_none_or(|file| {
                file.worktree_id(cx) != entry.origin.worktree_id
                    || file.path().as_ref() != entry.origin.path.as_ref()
            })
        {
            continue;
        }
        for runnable in &entry.runnables {
            if hints.len() == MAX_RUST_TEST_CASES {
                return hints;
            }
            if runnable.source.worktree_id != entry.origin.worktree_id
                || runnable.source.path.extension() != Some("rs")
                || !visible(&runnable.source, cx)
            {
                continue;
            }
            let Some(hint) = match_semantic_runnable(cargo_snapshot, runnable) else {
                continue;
            };
            let identity = (
                hint.package_id.clone(),
                hint.target_name.clone(),
                hint.target_kind,
                hint.test_name.clone(),
            );
            if !identities.insert(identity) {
                continue;
            }
            hints.push(hint);
        }
    }
    hints
}

fn match_semantic_runnable(
    cargo_snapshot: &CargoWorkspaceSnapshot,
    runnable: &RustSemanticRunnable,
) -> Option<RustTestSourceHint> {
    let (package_name, target_kind, target_name, test_name) = semantic_runnable_identity(runnable)?;
    let matching_targets = cargo_snapshot
        .workspaces
        .iter()
        .flat_map(|workspace| &workspace.members)
        .filter(|package| {
            package.name == package_name
                && package.manifest_path.worktree_id == runnable.source.worktree_id
                && source_belongs_to_package(&runnable.source, &package.manifest_path)
        })
        .flat_map(|package| {
            package.targets.iter().filter_map(move |target| {
                ((cargo_target_kind(&target.kind) == Some(target_kind)
                    || (target_kind == RustTestTargetKind::Doctest
                        && target.kind == CargoTargetKind::Library))
                    && target_name.is_none_or(|name| name == target.name))
                .then_some((package, target))
            })
        })
        .collect::<Vec<_>>();
    let [(package, target)] = matching_targets.as_slice() else {
        return None;
    };
    Some(RustTestSourceHint {
        package_id: package.id.clone(),
        target_name: target.name.clone(),
        target_kind,
        test_name: test_name.to_string(),
        source: Some(runnable.source.clone()),
        runnable: Some(RustTestRunnableSelector {
            cargo_args: runnable.cargo_args.clone(),
            executable_args: runnable.executable_args.clone(),
        }),
    })
}

fn source_belongs_to_package(source: &ProjectPath, manifest: &ProjectPath) -> bool {
    if source.worktree_id != manifest.worktree_id {
        return false;
    }
    let manifest_path = manifest.path.as_unix_str();
    let package_directory = manifest_path
        .rsplit_once('/')
        .map_or("", |(directory, _)| directory);
    package_directory.is_empty()
        || source
            .path
            .as_unix_str()
            .strip_prefix(package_directory)
            .is_some_and(|remaining| remaining.starts_with('/'))
}

fn semantic_runnable_identity(
    runnable: &RustSemanticRunnable,
) -> Option<(&str, RustTestTargetKind, Option<&str>, &str)> {
    if runnable.overridden
        || runnable.cargo_args.first().map(String::as_str) != Some("test")
        || runnable.cargo_args.len() > MAX_RUST_TEST_ACTION_ARGUMENTS
        || runnable.executable_args.len() > MAX_RUST_TEST_ACTION_ARGUMENTS
        || runnable
            .cargo_args
            .iter()
            .chain(&runnable.executable_args)
            .any(|argument| argument.len() > MAX_RUST_TEST_FIELD_BYTES)
        || runnable
            .cargo_args
            .iter()
            .chain(&runnable.executable_args)
            .map(String::len)
            .sum::<usize>()
            > MAX_RUST_TEST_SEMANTIC_ARGUMENT_BYTES
    {
        return None;
    }
    let mut package_name = None;
    let mut target = None;
    let mut arguments = runnable.cargo_args.iter().skip(1).map(String::as_str);
    while let Some(argument) = arguments.next() {
        match argument {
            "--package" | "-p" if package_name.is_none() => package_name = Some(arguments.next()?),
            "--lib" if target.is_none() => target = Some((RustTestTargetKind::Unit, None)),
            "--bin" if target.is_none() => {
                target = Some((RustTestTargetKind::Binary, Some(arguments.next()?)))
            }
            "--test" if target.is_none() => {
                target = Some((RustTestTargetKind::Integration, Some(arguments.next()?)))
            }
            "--example" if target.is_none() => {
                target = Some((RustTestTargetKind::Example, Some(arguments.next()?)))
            }
            "--bench" if target.is_none() => {
                target = Some((RustTestTargetKind::Benchmark, Some(arguments.next()?)))
            }
            "--doc" if target.is_none() => target = Some((RustTestTargetKind::Doctest, None)),
            "--features" | "--target" | "--profile" => {
                arguments.next()?;
            }
            "--no-default-features"
            | "--all-features"
            | "--release"
            | "--locked"
            | "--offline"
            | "--quiet"
            | "--no-run" => {}
            _ => return None,
        }
    }
    let test_name = runnable.executable_args.first()?.as_str();
    if test_name.is_empty()
        || test_name.starts_with('-')
        || test_name.len() > MAX_RUST_TEST_FIELD_BYTES
        || !runnable
            .executable_args
            .iter()
            .any(|argument| argument == "--exact")
        || runnable.executable_args.iter().skip(1).any(|argument| {
            !matches!(
                argument.as_str(),
                "--exact" | "--ignored" | "--include-ignored" | "--nocapture"
            )
        })
    {
        return None;
    }
    let package_name =
        package_name.filter(|name| !name.is_empty() && name.len() <= MAX_RUST_TEST_FIELD_BYTES)?;
    let (target_kind, target_name) = target?;
    Some((package_name, target_kind, target_name, test_name))
}

fn project_provider_projection(
    cargo_snapshot: &CargoWorkspaceSnapshot,
    captures: Vec<(CargoWorkspaceModel, RustTestDiscoveryCapture)>,
    source_hints: &[RustTestSourceHint],
    failures: Vec<RustTestDiscoveryFailure>,
    generation: DiscoveryGeneration,
) -> RustTestProviderProjection {
    let provider_id = RustTestProviderStore::provider_id();
    let root_id = stable_node_id("provider", &[RUST_TEST_PROVIDER_ID]);
    let mut nodes = vec![StructuredNode {
        experimental_feature: None,
        experimental_feature_enabled: None,
        debug_capability: None,
        id: root_id.clone(),
        parent_id: None,
        label: "Rust Tests".to_string(),
        kind: StructuredNodeKind::Provider,
        path: None,
    }];
    let mut actions = HashMap::default();
    let captures = captures
        .into_iter()
        .map(|(workspace, capture)| {
            let scoped_hints = source_hints
                .iter()
                .filter_map(|hint| {
                    workspace
                        .members
                        .iter()
                        .find(|package| {
                            hint.package_id == package.id
                                || (cargo_package_name(&hint.package_id) == package.name
                                    && hint.source.as_ref().is_some_and(|source| {
                                        source_belongs_to_package(source, &package.manifest_path)
                                    }))
                        })
                        .map(|package| {
                            let mut hint = hint.clone();
                            hint.package_id = package.name.clone();
                            hint
                        })
                })
                .collect::<Vec<_>>();
            let protocol = RustTestProtocolAdapter::default().adapt(capture, &scoped_hints);
            (workspace.key, protocol)
        })
        .collect::<BTreeMap<_, _>>();
    let mut protocol_partial = false;
    let mut discovered_cases = 0usize;
    for workspace in &cargo_snapshot.workspaces {
        let workspace_id = stable_node_id(
            "workspace",
            &[
                &workspace.key.root.worktree_id.to_proto().to_string(),
                workspace.key.root.path.as_unix_str(),
            ],
        );
        nodes.push(StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: workspace_id.clone(),
            parent_id: Some(root_id.clone()),
            label: workspace.display_name.clone(),
            kind: StructuredNodeKind::Group,
            path: workspace.root_manifest.clone(),
        });
        if let Some(manifest_path) = workspace.root_manifest.clone().or_else(|| {
            workspace
                .members
                .first()
                .map(|package| package.manifest_path.clone())
        }) {
            actions.insert(
                workspace_id.clone(),
                RustTestActionDescriptor {
                    scope: RustTestActionScope::Workspace,
                    worktree_id: manifest_path.worktree_id,
                    manifest_path,
                    label: workspace.display_name.clone(),
                    cargo_args: vec!["test".to_string(), "--workspace".to_string()],
                    executable_args: Vec::new(),
                    target_kind: None,
                    doctest_case_names: None,
                    doctest_ignored_run_supported: false,
                },
            );
        }
        let protocol = captures.get(&workspace.key);
        if let Some(protocol) = protocol {
            protocol_partial |= protocol.capability != RustTestProtocolCapability::Supported;
        }
        for package in &workspace.members {
            project_package_nodes(
                package,
                &workspace_id,
                protocol,
                &mut nodes,
                &mut actions,
                &mut discovered_cases,
            );
        }
    }
    let restricted = cargo_snapshot.workspaces.is_empty()
        && cargo_snapshot
            .failures
            .iter()
            .any(|failure| failure.category == CargoWorkspaceErrorCategory::Restricted);
    let cargo_failure = cargo_snapshot
        .failures
        .iter()
        .find(|failure| failure.category != CargoWorkspaceErrorCategory::Restricted);
    let partial = cargo_snapshot.completeness == CargoSnapshotCompleteness::Partial
        || !cargo_snapshot.failures.is_empty()
        || protocol_partial
        || !failures.is_empty();
    let status = if restricted {
        StructuredProviderStatus::Restricted
    } else if cargo_snapshot.workspaces.is_empty() && cargo_failure.is_some() {
        StructuredProviderStatus::Error
    } else if cargo_snapshot.workspaces.is_empty() {
        StructuredProviderStatus::Empty
    } else if !failures.is_empty() && discovered_cases == 0 {
        StructuredProviderStatus::Error
    } else if partial {
        StructuredProviderStatus::Partial
    } else if discovered_cases == 0 {
        StructuredProviderStatus::Empty
    } else {
        StructuredProviderStatus::Current
    };
    let diagnostic = if restricted {
        Some("Trust the project to discover Rust tests".to_string())
    } else if !failures.is_empty() {
        Some(bounded_field(
            &failures
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
            MAX_RUST_TEST_FIELD_BYTES,
        ))
    } else if let Some(failure) = cargo_failure {
        Some(cargo_discovery_failure_message(failure.category).to_string())
    } else if protocol_partial {
        Some("Some Rust test records were unsupported or truncated".to_string())
    } else {
        None
    };
    for node in &mut nodes {
        node.debug_capability = actions.get(&node.id).map(rust_test_debug_capability);
        node.experimental_feature = actions
            .get(&node.id)
            .filter(|descriptor| {
                descriptor.scope == RustTestActionScope::Case
                    && descriptor.target_kind == Some(RustTestTargetKind::Doctest)
            })
            .map(|_| "rust-exact-doctest-actions-experimental".to_string());
    }
    let mut snapshot =
        StructuredProviderSnapshot::discovery(provider_id, generation, status, nodes);
    snapshot.partial = partial;
    snapshot.diagnostic = diagnostic;
    RustTestProviderProjection { snapshot, actions }
}

#[cfg(any(test, feature = "test-support"))]
pub fn project_provider_snapshot_for_test(
    cargo_snapshot: &CargoWorkspaceSnapshot,
    captures: Vec<(CargoWorkspaceModel, RustTestDiscoveryCapture)>,
    source_hints: &[RustTestSourceHint],
    failures: Vec<String>,
    generation: DiscoveryGeneration,
) -> StructuredProviderSnapshot {
    let failures = failures
        .into_iter()
        .map(|_| RustTestDiscoveryFailure::unknown())
        .collect();
    project_provider_projection(cargo_snapshot, captures, source_hints, failures, generation)
        .snapshot
}

fn project_package_nodes(
    package: &CargoPackageModel,
    workspace_id: &StructuredNodeId,
    protocol: Option<&RustTestProtocolSnapshot>,
    nodes: &mut Vec<StructuredNode>,
    actions: &mut HashMap<StructuredNodeId, RustTestActionDescriptor>,
    discovered_cases: &mut usize,
) {
    let mut doctest_case_names = BTreeMap::<String, Arc<[String]>>::new();
    if let Some(protocol) = protocol.filter(|protocol| {
        !protocol.truncated && protocol.capability == RustTestProtocolCapability::Supported
    }) {
        let mut names = BTreeMap::<String, Vec<String>>::new();
        let doctest_harnesses = protocol
            .harnesses
            .iter()
            .filter(|harness| {
                harness.target_kind == RustTestTargetKind::Doctest
                    && (harness.package_id == package.id
                        || cargo_package_name(&harness.package_id) == package.name)
            })
            .map(|harness| harness.id.as_str())
            .collect::<BTreeSet<_>>();
        for case in &protocol.cases {
            if doctest_harnesses.contains(case.harness_id.as_str()) {
                names
                    .entry(case.harness_id.clone())
                    .or_default()
                    .push(case.name.clone());
            }
        }
        doctest_case_names.extend(
            names
                .into_iter()
                .map(|(harness, names)| (harness, names.into())),
        );
    }
    let package_id = stable_node_id("package", &[&package.id]);
    nodes.push(StructuredNode {
        experimental_feature: None,
        experimental_feature_enabled: None,
        debug_capability: None,
        id: package_id.clone(),
        parent_id: Some(workspace_id.clone()),
        label: package.name.clone(),
        kind: StructuredNodeKind::Suite,
        path: Some(package.manifest_path.clone()),
    });
    actions.insert(
        package_id.clone(),
        RustTestActionDescriptor {
            scope: RustTestActionScope::Package,
            worktree_id: package.manifest_path.worktree_id,
            manifest_path: package.manifest_path.clone(),
            label: package.name.clone(),
            cargo_args: vec![
                "test".to_string(),
                "--package".to_string(),
                package.name.clone(),
            ],
            executable_args: Vec::new(),
            target_kind: None,
            doctest_case_names: None,
            doctest_ignored_run_supported: false,
        },
    );
    for target in &package.targets {
        let Some(target_kind) = cargo_target_kind(&target.kind) else {
            continue;
        };
        let target_id = stable_node_id(
            "target",
            &[&package.id, &target.name, &format!("{target_kind:?}")],
        );
        nodes.push(StructuredNode {
            experimental_feature: None,
            experimental_feature_enabled: None,
            debug_capability: None,
            id: target_id.clone(),
            parent_id: Some(package_id.clone()),
            label: target.name.clone(),
            kind: StructuredNodeKind::Group,
            path: target.source_path.clone(),
        });
        actions.insert(
            target_id.clone(),
            RustTestActionDescriptor {
                scope: RustTestActionScope::Target,
                worktree_id: package.manifest_path.worktree_id,
                manifest_path: package.manifest_path.clone(),
                label: target.name.clone(),
                cargo_args: target_cargo_args(&package.name, &target.name, target_kind),
                executable_args: Vec::new(),
                target_kind: Some(target_kind),
                doctest_case_names: None,
                doctest_ignored_run_supported: false,
            },
        );
        let Some(protocol) = protocol else {
            continue;
        };
        let harnesses = protocol
            .harnesses
            .iter()
            .filter(|harness| {
                (harness.package_id == package.id
                    || cargo_package_name(&harness.package_id) == package.name)
                    && harness.target_name == target.name
                    && (harness.target_kind == target_kind
                        || (target_kind == RustTestTargetKind::Unit
                            && harness.target_kind == RustTestTargetKind::Doctest))
            })
            .map(|harness| (&harness.id, harness.target_kind))
            .collect::<BTreeMap<_, _>>();
        let mut groups = BTreeMap::<String, StructuredNodeId>::new();
        for case in protocol
            .cases
            .iter()
            .filter(|case| harnesses.contains_key(&case.harness_id))
        {
            let mut parent_id = target_id.clone();
            let components = case.name.split("::").collect::<Vec<_>>();
            for depth in 0..components.len().saturating_sub(1) {
                let group_name = components[..=depth].join("::");
                let group_id = groups.entry(group_name.clone()).or_insert_with(|| {
                    let id = stable_node_id("group", &[&target_id.0, &group_name]);
                    nodes.push(StructuredNode {
                        experimental_feature: None,
                        experimental_feature_enabled: None,
                        debug_capability: None,
                        id: id.clone(),
                        parent_id: Some(parent_id.clone()),
                        label: components[depth].to_string(),
                        kind: StructuredNodeKind::Group,
                        path: None,
                    });
                    id
                });
                actions.entry(group_id.clone()).or_insert_with(|| {
                    let harness_kind = harnesses
                        .get(&case.harness_id)
                        .copied()
                        .unwrap_or(target_kind);
                    let mut cargo_args = case.runnable.cargo_args.clone();
                    normalize_package_selector(&mut cargo_args, &package.name);
                    RustTestActionDescriptor {
                        scope: RustTestActionScope::Group,
                        worktree_id: package.manifest_path.worktree_id,
                        manifest_path: package.manifest_path.clone(),
                        label: group_name.clone(),
                        cargo_args,
                        executable_args: vec![group_name.clone(), "--nocapture".to_string()],
                        target_kind: Some(harness_kind),
                        doctest_case_names: None,
                        doctest_ignored_run_supported: false,
                    }
                });
                parent_id = group_id.clone();
            }
            let case_id = stable_node_id("case", &[&target_id.0, &case.id]);
            nodes.push(StructuredNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                id: case_id.clone(),
                parent_id: Some(parent_id),
                label: components.last().copied().unwrap_or(&case.name).to_string(),
                kind: StructuredNodeKind::Case,
                path: case.source.clone(),
            });
            let mut cargo_args = case.runnable.cargo_args.clone();
            normalize_package_selector(&mut cargo_args, &package.name);
            let mut executable_args = case.runnable.executable_args.clone();
            if case.ignored
                && !executable_args
                    .iter()
                    .any(|argument| argument == "--include-ignored")
            {
                executable_args.push("--include-ignored".to_string());
            }
            if !executable_args
                .iter()
                .any(|argument| argument == "--nocapture")
            {
                executable_args.push("--nocapture".to_string());
            }
            actions.insert(
                case_id,
                RustTestActionDescriptor {
                    scope: RustTestActionScope::Case,
                    worktree_id: package.manifest_path.worktree_id,
                    manifest_path: package.manifest_path.clone(),
                    label: case.name.clone(),
                    cargo_args,
                    executable_args,
                    target_kind: harnesses.get(&case.harness_id).copied(),
                    doctest_case_names: doctest_case_names.get(&case.harness_id).cloned(),
                    doctest_ignored_run_supported: matches!(
                        target.edition.as_str(),
                        "2015" | "2018" | "2021"
                    ),
                },
            );
            *discovered_cases = discovered_cases.saturating_add(1);
        }
    }
}

fn target_cargo_args(
    package_name: &str,
    target_name: &str,
    target_kind: RustTestTargetKind,
) -> Vec<String> {
    let mut arguments = vec![
        if target_kind == RustTestTargetKind::Benchmark {
            "bench".to_string()
        } else {
            "test".to_string()
        },
        "--package".to_string(),
        package_name.to_string(),
    ];
    match target_kind {
        RustTestTargetKind::Unit => arguments.push("--lib".to_string()),
        RustTestTargetKind::Integration => {
            arguments.extend(["--test".to_string(), target_name.to_string()])
        }
        RustTestTargetKind::Binary => {
            arguments.extend(["--bin".to_string(), target_name.to_string()])
        }
        RustTestTargetKind::Example => {
            arguments.extend(["--example".to_string(), target_name.to_string()])
        }
        RustTestTargetKind::Benchmark => {
            arguments.extend(["--bench".to_string(), target_name.to_string()])
        }
        RustTestTargetKind::Doctest => arguments.push("--doc".to_string()),
    }
    arguments
}

fn rust_test_task_template(descriptor: &RustTestActionDescriptor) -> TaskTemplate {
    let mut arguments = descriptor.cargo_args.clone();
    if !descriptor.executable_args.is_empty() {
        arguments.push("--".to_string());
        arguments.extend(descriptor.executable_args.clone());
    }
    let worktree_root = VariableName::WorktreeRoot.template_value();
    let cwd = descriptor
        .manifest_path
        .path
        .parent()
        .filter(|parent| !parent.is_empty())
        .map(|parent| format!("{worktree_root}/{}", parent.as_unix_str()))
        .unwrap_or(worktree_root);
    TaskTemplate {
        experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
        required_experimental_features: if descriptor.scope == RustTestActionScope::Case
            && descriptor.target_kind == Some(RustTestTargetKind::Doctest)
        {
            vec!["rust-exact-doctest-actions-experimental".to_string()]
        } else {
            Vec::new()
        },
        label: format!("Rust test {}", descriptor.label),
        command: "cargo".to_string(),
        args: arguments,
        cwd: Some(cwd),
        use_new_terminal: true,
        allow_concurrent_runs: false,
        tags: vec!["rust-test".to_string(), "structured-test".to_string()],
        save: SaveStrategy::All,
        ..TaskTemplate::default()
    }
}

fn rust_test_debug_scenario(mut template: TaskTemplate) -> Result<DebugScenario> {
    let delimiter = template
        .args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(template.args.len());
    if !template.args[..delimiter]
        .iter()
        .any(|argument| argument == "--no-run")
    {
        template.args.insert(delimiter, "--no-run".to_string());
    }
    Ok(rust_test_debug_scenario_from_build(template))
}

fn rust_test_debug_scenario_from_build(template: TaskTemplate) -> DebugScenario {
    DebugScenario {
        adapter: "CodeLLDB".into(),
        label: format!("Debug {}", template.label).into(),
        build: Some(BuildTaskDefinition::Template {
            task_template: template,
            locator_name: Some("rust-cargo-locator".into()),
        }),
        config: serde_json::json!({ "sourceLanguages": ["rust"] }),
        tcp_connection: None,
    }
}

fn normalize_package_selector(arguments: &mut [String], package_name: &str) {
    let package_value = arguments
        .iter()
        .position(|argument| argument == "--package" || argument == "-p")
        .and_then(|index| index.checked_add(1));
    if let Some(package_value) = package_value.and_then(|index| arguments.get_mut(index)) {
        *package_value = package_name.to_string();
    }
}

fn cargo_target_kind(kind: &CargoTargetKind) -> Option<RustTestTargetKind> {
    match kind {
        CargoTargetKind::Library => Some(RustTestTargetKind::Unit),
        CargoTargetKind::Binary => Some(RustTestTargetKind::Binary),
        CargoTargetKind::Example => Some(RustTestTargetKind::Example),
        CargoTargetKind::Test => Some(RustTestTargetKind::Integration),
        CargoTargetKind::Bench => Some(RustTestTargetKind::Benchmark),
        CargoTargetKind::BuildScript | CargoTargetKind::Other(_) => None,
    }
}

fn cargo_package_name(package_id: &str) -> &str {
    package_id
        .rsplit_once('#')
        .map(|(_, package)| package)
        .unwrap_or(package_id)
        .split('@')
        .next()
        .unwrap_or(package_id)
}

fn stable_node_id(prefix: &str, components: &[&str]) -> StructuredNodeId {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for component in components {
        digest.update(component.len().to_le_bytes());
        digest.update(component.as_bytes());
    }
    StructuredNodeId(format!("rust-test-{prefix}-{:x}", digest.finalize()))
}

fn rust_test_input_changed(event: &WorktreeStoreEvent) -> bool {
    match event {
        WorktreeStoreEvent::WorktreeUpdatedEntries(_, entries) => {
            entries.iter().any(|(path, _, _)| {
                matches!(path.extension(), Some("rs" | "toml"))
                    || matches!(path.file_name(), Some("Cargo.lock" | "rust-toolchain"))
            })
        }
        WorktreeStoreEvent::WorktreeAdded(_)
        | WorktreeStoreEvent::WorktreeRemoved(_, _)
        | WorktreeStoreEvent::WorktreeReleased(_, _)
        | WorktreeStoreEvent::WorktreeOrderChanged
        | WorktreeStoreEvent::WorktreeDeletedEntry(_, _) => true,
        WorktreeStoreEvent::WorktreeUpdatedGitRepositories(_, _)
        | WorktreeStoreEvent::WorktreeUpdatedRootRepoCommonDir(_)
        | WorktreeStoreEvent::WorktreeUpdateSent(_) => false,
    }
}

fn cancel_superseded_run_handles(
    run_handles: &HashMap<StructuredRunId, StructuredTaskHandle>,
    new_run_id: &StructuredRunId,
    cx: &mut App,
) {
    for (run_id, handle) in run_handles {
        if run_id != new_run_id && !handle.state().is_terminal() {
            handle.cancel(cx);
        }
    }
}

fn provider_status_to_proto(status: StructuredProviderStatus) -> i32 {
    (match status {
        StructuredProviderStatus::Loading => proto::StructuredProviderStatus::Loading,
        StructuredProviderStatus::Current => proto::StructuredProviderStatus::Current,
        StructuredProviderStatus::Empty => proto::StructuredProviderStatus::Empty,
        StructuredProviderStatus::Partial => proto::StructuredProviderStatus::Partial,
        StructuredProviderStatus::Stale => proto::StructuredProviderStatus::Stale,
        StructuredProviderStatus::Error => proto::StructuredProviderStatus::Error,
        StructuredProviderStatus::Restricted => proto::StructuredProviderStatus::Restricted,
        StructuredProviderStatus::Disconnected => proto::StructuredProviderStatus::Disconnected,
        StructuredProviderStatus::Mismatch => proto::StructuredProviderStatus::Mismatch,
    }) as i32
}

fn superseded_discovery_error() -> anyhow::Error {
    anyhow!(RustTestRemoteError {
        kind: RustTestRemoteErrorKind::SupersededDiscovery,
    })
}

fn map_remote_discovery_error(error: anyhow::Error) -> anyhow::Error {
    if error
        .downcast_ref::<proto::RpcError>()
        .is_some_and(|error| error.error_tag("reason") == Some("rust_test_discovery_superseded"))
    {
        return superseded_discovery_error();
    }
    if error
        .downcast_ref::<proto::RpcError>()
        .is_some_and(|error| {
            error.error_code() == proto::ErrorCode::Forbidden
                && matches!(
                    error.error_tag("reason"),
                    Some("rust_test_trust_revoked" | "cargo_trust_revoked")
                )
        })
    {
        return anyhow!("Rust test discovery was restricted. Trust the worktree and refresh Tests");
    }
    let unsupported = error
        .downcast_ref::<proto::RpcError>()
        .is_some_and(|error| {
            error.error_tag("reason") == Some("rust_tools_unsupported")
                || error.raw_message().contains("was not handled")
                || error
                    .raw_message()
                    .starts_with("no handler registered for ")
        });
    if unsupported {
        anyhow!(RustTestRemoteError {
            kind: RustTestRemoteErrorKind::UnsupportedDiscoveryHost,
        })
    } else {
        error.context("Rust test discovery request failed")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        WorktreeId,
        cargo_workspace::{
            CargoCandidateFailure, CargoTargetModel, CargoWorkspaceConfiguration, CargoWorkspaceKey,
        },
    };
    use util::rel_path::RelPath;

    use super::*;

    struct FixtureRunner;

    impl RustTestDiscoveryRunner for FixtureRunner {
        fn discover(
            &self,
            _: RustTestDiscoveryRequest,
        ) -> BoxFuture<'static, Result<RustTestDiscoveryCapture>> {
            Box::pin(async { Ok(capture()) })
        }
    }

    const PACKAGE_ID: &str = "path+file:///fixture#rust-fixture@0.1.0";
    const CARGO_MESSAGES: &str =
        include_str!("../test_data/rust_test_provider/cargo_messages.jsonl");
    const LISTINGS: &str = include_str!("../test_data/rust_test_provider/listings.json");

    #[cfg(unix)]
    struct DiscoveryProcessFixture {
        directory: tempfile::TempDir,
        request: RustTestDiscoveryRequest,
    }

    #[cfg(unix)]
    impl DiscoveryProcessFixture {
        fn new(stage: &str, stream: &str) -> Self {
            use std::os::unix::fs::PermissionsExt;

            let directory = tempfile::tempdir().expect("discovery process fixture directory");
            let harness_path = directory.path().join("harness");
            let cargo_path = directory.path().join("cargo");
            let messages_path = directory.path().join("messages.jsonl");
            let pid_path = directory.path().join("child.pid");
            let mut artifact: serde_json::Value = serde_json::from_str(
                CARGO_MESSAGES
                    .lines()
                    .next()
                    .expect("Cargo artifact fixture"),
            )
            .expect("Cargo artifact JSON");
            artifact["executable"] = serde_json::json!(harness_path);
            std::fs::write(&messages_path, format!("{artifact}\n"))
                .expect("write Cargo artifact fixture");
            std::fs::write(directory.path().join("Cargo.toml"), "[workspace]\n")
                .expect("write process fixture manifest");
            let emit = match stream {
                "stdout" => {
                    "index=0; while [ \"$index\" -lt 1024 ]; do printf 'overflow'; index=$((index + 1)); done\n"
                }
                "stderr" => {
                    "index=0; while [ \"$index\" -lt 1024 ]; do printf 'overflow' >&2; index=$((index + 1)); done\n"
                }
                "idle" => "",
                "success" => {
                    "printf '%s\\n' '{\"reason\":\"build-finished\",\"success\":true}'\nprintf 'bounded diagnostic' >&2\nexit 0\n"
                }
                "failure" => "printf 'bounded diagnostic' >&2\nexit 7\n",
                _ => panic!("unknown discovery fixture stream"),
            };
            // Exec keeps the sleeping process under the runner's own PID, with no grandchildren.
            let body = format!(
                "printf '%s\\n' \"$$\" > \"$RUST_TEST_FIXTURE_PID.pending\"\n/bin/mv \"$RUST_TEST_FIXTURE_PID.pending\" \"$RUST_TEST_FIXTURE_PID\"\n{emit}exec /bin/sleep 30\n"
            );
            let cargo = match stage {
                "cargo" => body.clone(),
                "harness" => "exec /bin/cat \"$RUST_TEST_FIXTURE_MESSAGES\"\n".into(),
                "doctest" => format!(
                    "case \"$*\" in *--doc*) {body};; *) printf '%s\\n' '{{\"reason\":\"build-finished\",\"success\":true}}';; esac\n"
                ),
                _ => panic!("unknown discovery fixture stage"),
            };
            for (path, script) in [(&cargo_path, cargo), (&harness_path, body)] {
                std::fs::write(path, format!("#!/bin/sh\n{script}"))
                    .expect("write controlled discovery executable");
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                    .expect("make discovery fixture executable");
            }
            let request = RustTestDiscoveryRequest {
                workspace_root: project_path("Cargo.toml"),
                generation: 1,
                limits: RustTestProtocolLimits {
                    cargo_bytes: 4096,
                    listing_bytes: 128,
                    ..RustTestProtocolLimits::default()
                },
                manifest_path: Some(directory.path().join("Cargo.toml")),
                working_directory: Some(directory.path().to_path_buf()),
                environment: Some(HashMap::from_iter([
                    (
                        "PATH".into(),
                        directory.path().to_string_lossy().into_owned(),
                    ),
                    (
                        "RUST_TEST_FIXTURE_PID".into(),
                        pid_path.to_string_lossy().into_owned(),
                    ),
                    (
                        "RUST_TEST_FIXTURE_MESSAGES".into(),
                        messages_path.to_string_lossy().into_owned(),
                    ),
                ])),
                doctest_targets: if stage == "doctest" {
                    vec![("fixture".into(), "fixture".into())]
                } else {
                    Vec::new()
                },
            };
            Self { directory, request }
        }

        fn process_id(&self) -> Option<String> {
            match std::fs::read_to_string(self.directory.path().join("child.pid")) {
                Ok(contents) if contents.trim().is_empty() => None,
                Ok(contents) => {
                    let process_id = contents.trim().parse::<u32>().expect("fixture child PID");
                    Some(process_id.to_string())
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => panic!("could not read fixture child PID: {error}"),
            }
        }
    }

    #[cfg(unix)]
    fn poll_discovery_process<T>(
        cx: &gpui::TestAppContext,
        mut condition: impl FnMut() -> Option<T>,
    ) -> Option<T> {
        cx.executor().allow_parking();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(value) = condition() {
                return Some(value);
            }
            if std::time::Instant::now() >= deadline {
                return None;
            }
            // Real pipe notifications need CPU time outside the deterministic scheduler.
            let pause = if cx.executor().tick() {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(5)
            };
            std::thread::sleep(pause);
        }
    }

    #[cfg(unix)]
    fn require_discovery_process_exit(cx: &gpui::TestAppContext, process_id: &str) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let mut command = util::command::new_command("/bin/kill");
            command
                .args(["-0", process_id])
                .stdout(util::command::Stdio::null())
                .stderr(util::command::Stdio::null())
                .kill_on_drop(true);
            let mut probe = cx
                .background_executor
                .spawn(async move { command.status().await });
            let status = poll_discovery_process(cx, || (&mut probe).now_or_never())
                .expect("owned fixture process probe completes")
                .expect("inspect owned fixture process");
            if !status.success() {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "owned discovery child {process_id} was not terminated"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn rust_test_discovery_passive_rustup_guard_covers_each_process_stage(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;

        for stage in [
            RustTestDiscoveryStage::Cargo,
            RustTestDiscoveryStage::Harness,
            RustTestDiscoveryStage::Doctest,
        ] {
            let mut fixture = DiscoveryProcessFixture::new("harness", "success");
            let directory = fixture.directory.path();
            let manager = directory.join("rustup");
            let cargo = directory.join("cargo");
            let staged_cargo = directory.join("staged-cargo");
            let script = r#"#!/bin/sh
[ "$RUSTUP_AUTO_INSTALL" = 0 ] && [ "$CARGO_NET_OFFLINE" = true ] || exit 73
case "${0##*/}" in
    rustup)
        [ "$1" = --version ] || exit 74
        case "$RUSTUP_TOOLCHAIN" in rust-workspace-passive-version-probe-*) ;; *) exit 75;; esac
        printf 'rustup 1.27.1 (controlled fixture)\n'
        ;;
    *) printf launched > "$PASSIVE_LEGACY_COLLECTOR"; exit 76;;
esac
"#;
            std::fs::write(&manager, script).expect("write legacy manager fixture");
            std::fs::set_permissions(&manager, std::fs::Permissions::from_mode(0o700))
                .expect("executable legacy manager");
            let environment = fixture
                .request
                .environment
                .as_mut()
                .expect("fixture environment");
            environment.insert("RUSTUP_TOOLCHAIN".into(), "private-toolchain-value".into());
            environment.insert(
                "PASSIVE_LEGACY_COLLECTOR".into(),
                directory
                    .join("legacy-launched")
                    .to_string_lossy()
                    .into_owned(),
            );
            match stage {
                RustTestDiscoveryStage::Cargo => {
                    std::fs::copy(&manager, &cargo).expect("legacy initial Cargo proxy");
                }
                RustTestDiscoveryStage::Harness => {
                    std::fs::copy(&manager, directory.join("harness"))
                        .expect("legacy executable listing proxy");
                }
                RustTestDiscoveryStage::Doctest => {
                    fixture
                        .request
                        .doctest_targets
                        .push((PACKAGE_ID.into(), "rust_fixture".into()));
                    std::fs::copy(&manager, &staged_cargo).expect("staged legacy Cargo proxy");
                    environment.insert(
                        "PASSIVE_STAGED_CARGO".into(),
                        staged_cargo.to_string_lossy().into_owned(),
                    );
                    environment.insert(
                        "PASSIVE_CURRENT_CARGO".into(),
                        cargo.to_string_lossy().into_owned(),
                    );
                    let original = std::fs::read_to_string(&cargo).expect("initial Cargo fixture");
                    let body = original
                        .strip_prefix("#!/bin/sh\n")
                        .expect("fixture shebang");
                    // Replace the executable after preparation to require a fresh doctest check.
                    std::fs::write(&cargo, format!("#!/bin/sh\n/bin/mv \"$PASSIVE_STAGED_CARGO\" \"$PASSIVE_CURRENT_CARGO\"\n{body}")).expect("Cargo stage replacement fixture");
                }
                RustTestDiscoveryStage::Unknown => panic!("fixed discovery stage"),
            }
            let runner = ProcessRustTestDiscoveryRunner {
                executor: cx.background_executor.clone(),
            };
            let mut discovery = cx
                .background_executor
                .spawn(runner.discover(fixture.request.clone()));
            let error = poll_discovery_process(cx, || (&mut discovery).now_or_never())
                .expect("guarded stage completes")
                .expect_err("legacy proxy cannot execute a discovery stage");
            let failure = RustTestDiscoveryFailure::from_error(&error);
            assert!(
                matches!(failure.kind, RustTestDiscoveryFailureKind::PassivePolicy),
                "{error:#}"
            );
            assert_eq!(failure.stage.label(), stage.label());
            assert!(failure.to_string().contains("update Rustup manually"));
            assert!(!failure.to_string().contains("private-toolchain-value"));
            assert!(!directory.join("legacy-launched").exists());
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn rust_test_discovery_passive_processes_prevent_downloads(cx: &mut gpui::TestAppContext) {
        let mut fixture = DiscoveryProcessFixture::new("harness", "success");
        let check_environment = r#"[ "$RUSTUP_AUTO_INSTALL" = 0 ] &&
    [ "$CARGO_NET_OFFLINE" = true ] &&
    [ "$RUST_WORKSPACE_FIXTURE_VALUE" = retained ] || exit 73
"#;
        let cargo = r#"case " $* " in *" --offline "*) ;; *) exit 74;; esac
case " $* " in
    *" --doc "*)
        case " $* " in *" --ignored "*) ;; *) printf 'source.rs - example (line 1): test\n';; esac;;
    *) exec /bin/cat "$RUST_TEST_FIXTURE_MESSAGES";;
esac
"#;
        let harness = "printf 'unit_case: test\\n'\n";
        for (name, body) in [("cargo", cargo), ("harness", harness)] {
            std::fs::write(
                fixture.directory.path().join(name),
                format!("#!/bin/sh\n{check_environment}{body}"),
            )
            .expect("write passive discovery policy fixture");
        }
        fixture.request.doctest_targets = vec![("fixture".into(), "fixture".into())];
        let environment = fixture
            .request
            .environment
            .as_mut()
            .expect("controlled discovery environment");
        environment.extend([
            ("RUSTUP_AUTO_INSTALL".into(), "1".into()),
            ("CARGO_NET_OFFLINE".into(), "false".into()),
            ("RUST_WORKSPACE_FIXTURE_VALUE".into(), "retained".into()),
        ]);
        let runner = ProcessRustTestDiscoveryRunner {
            executor: cx.background_executor.clone(),
        };
        let mut discovery = cx
            .background_executor
            .spawn(runner.discover(fixture.request.clone()));
        let capture = poll_discovery_process(cx, || (&mut discovery).now_or_never())
            .expect("all passive command stages complete")
            .expect("project environment cannot enable passive downloads");
        assert_eq!(capture.listings.len(), 4);
        for kind in [RustTestTargetKind::Unit, RustTestTargetKind::Doctest] {
            for mode in [RustTestListingMode::All, RustTestListingMode::Ignored] {
                assert!(
                    capture
                        .listings
                        .iter()
                        .any(|listing| { listing.target_kind == kind && listing.mode == mode })
                );
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn rust_test_discovery_process_output_rejects_overflow_before_eof(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut failures = Vec::new();
        for stage in ["cargo", "harness", "doctest"] {
            for stream in ["stdout", "stderr"] {
                let fixture = DiscoveryProcessFixture::new(stage, stream);
                let runner = ProcessRustTestDiscoveryRunner {
                    executor: cx.background_executor.clone(),
                };
                let mut discovery = cx
                    .background_executor
                    .spawn(runner.discover(fixture.request.clone()));
                let result = poll_discovery_process(cx, || (&mut discovery).now_or_never());
                drop(discovery);
                let process_id = fixture.process_id().unwrap_or_else(|| {
                    panic!("actual emitting process started: stage={stage}, stream={stream}, result={result:?}")
                });
                require_discovery_process_exit(cx, &process_id);
                match result {
                    Some(Err(error)) => {
                        let message = format!("{error:#}");
                        if !message.contains(stream)
                            || !message.contains("limit")
                            || message.contains("timed out")
                        {
                            failures.push(format!("{stage} {stream}: {message}"));
                        }
                    }
                    Some(Ok(_)) => {
                        failures.push(format!("{stage} {stream}: overflow was accepted"))
                    }
                    None => {
                        failures.push(format!("{stage} {stream}: no overflow error before EOF"))
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[cfg(unix)]
    #[gpui::test]
    fn rust_test_discovery_process_output_preserves_status_and_bounded_diagnostics(
        cx: &mut gpui::TestAppContext,
    ) {
        for stream in ["success", "failure"] {
            let fixture = DiscoveryProcessFixture::new("cargo", stream);
            let runner = ProcessRustTestDiscoveryRunner {
                executor: cx.background_executor.clone(),
            };
            let mut discovery = cx
                .background_executor
                .spawn(runner.discover(fixture.request.clone()));
            let result = poll_discovery_process(cx, || (&mut discovery).now_or_never());
            drop(discovery);
            let result = result.expect("benign output reaches process EOF");
            match stream {
                "success" => {
                    let capture = result.expect("successful bounded discovery");
                    assert_eq!(
                        capture.cargo_messages,
                        "{\"reason\":\"build-finished\",\"success\":true}\n"
                    );
                    assert!(capture.listings.is_empty());
                }
                "failure" => {
                    let error = result.expect_err("failed child retains real exit status");
                    let message = error.to_string();
                    assert!(
                        message.contains('7')
                            && message.contains("Cargo test discovery")
                            && !message.contains("bounded diagnostic"),
                        "{message}"
                    );
                }
                _ => unreachable!("fixed fixture stream"),
            }
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    fn rust_test_discovery_process_output_timeout_and_cancellation_terminate_owned_child(
        cx: &mut gpui::TestAppContext,
    ) {
        for timeout in [false, true] {
            let fixture = DiscoveryProcessFixture::new("cargo", "idle");
            let runner = ProcessRustTestDiscoveryRunner {
                executor: cx.background_executor.clone(),
            };
            let mut discovery = cx
                .background_executor
                .spawn(runner.discover(fixture.request.clone()));
            let process_id = poll_discovery_process(cx, || fixture.process_id())
                .expect("owned process starts before cancellation or timeout");
            if timeout {
                // advance_clock drains runnable work before advancing, which can be kept
                // ready by the real pipe reactor while the child intentionally stays idle.
                cx.executor()
                    .dispatcher()
                    .as_test()
                    .expect("GPUI process test dispatcher")
                    .scheduler()
                    .clock()
                    .advance(RUST_TEST_DISCOVERY_TIMEOUT);
                let result = poll_discovery_process(cx, || (&mut discovery).now_or_never())
                    .expect("discovery timeout completes");
                let error = result.expect_err("timeout rejects discovery");
                assert!(error.to_string().contains("timed out"), "{error:#}");
            }
            drop(discovery);
            require_discovery_process_exit(cx, &process_id);
        }
    }

    #[gpui::test]
    async fn rust_test_new_run_cancels_superseded_terminal(cx: &mut gpui::TestAppContext) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let terminated = Arc::new(AtomicUsize::new(0));
        let prior_id = StructuredRunId("prior".into());
        let current_id = StructuredRunId("current".into());
        let prior = StructuredTaskHandle::new(TaskId("prior-task".into()));
        let current = StructuredTaskHandle::new(TaskId("current-task".into()));
        cx.update(|cx| {
            let prior_terminated = terminated.clone();
            assert!(prior.bind_terminal(
                StructuredTerminalId(1),
                move |_| {
                    prior_terminated.fetch_add(1, Ordering::SeqCst);
                    true
                },
                cx,
            ));
            let current_terminated = terminated.clone();
            assert!(current.bind_terminal(
                StructuredTerminalId(2),
                move |_| {
                    current_terminated.fetch_add(10, Ordering::SeqCst);
                    true
                },
                cx,
            ));
            let handles = HashMap::from_iter([
                (prior_id, prior.clone()),
                (current_id.clone(), current.clone()),
            ]);
            cancel_superseded_run_handles(&handles, &current_id, cx);
            assert!(matches!(
                prior.state(),
                StructuredTaskState::Cancelled { .. }
            ));
            assert!(matches!(
                current.state(),
                StructuredTaskState::Running { .. }
            ));
            assert_eq!(terminated.load(Ordering::SeqCst), 1);
            cancel_superseded_run_handles(&handles, &current_id, cx);
            assert_eq!(terminated.load(Ordering::SeqCst), 1);
        });
    }

    #[gpui::test]
    async fn rust_exact_doctest_opt_in_preserves_ordinary_scopes(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-exact-doctest-actions-experimental",
                false,
                cx,
            )
        });
        let filesystem = crate::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"Cargo.toml": "[workspace]"}),
            )
            .await;
        let project = crate::Project::test(filesystem, ["/workspace".as_ref()], cx).await;
        let store = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let mut descriptor = RustTestActionDescriptor {
            scope: RustTestActionScope::Case,
            worktree_id: WorktreeId::from_usize(1),
            manifest_path: project_path("Cargo.toml"),
            label: "src/lib.rs - exact case".into(),
            cargo_args: vec!["test".into(), "--doc".into()],
            executable_args: vec!["src/lib.rs - exact case".into()],
            target_kind: Some(RustTestTargetKind::Doctest),
            doctest_case_names: None,
            doctest_ignored_run_supported: false,
        };
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .ensure_action_experimental_features(&descriptor, cx))
                .is_err()
        );
        assert_eq!(
            rust_test_task_template(&descriptor).required_experimental_features,
            vec!["rust-exact-doctest-actions-experimental"]
        );
        for scope in [
            RustTestActionScope::Workspace,
            RustTestActionScope::Package,
            RustTestActionScope::Target,
            RustTestActionScope::Group,
        ] {
            descriptor.scope = scope;
            assert!(
                store
                    .read_with(cx, |store, cx| store
                        .ensure_action_experimental_features(&descriptor, cx))
                    .is_ok(),
                "ordinary whole-scope action remains available"
            );
            assert!(
                rust_test_task_template(&descriptor)
                    .required_experimental_features
                    .is_empty()
            );
        }
        descriptor.scope = RustTestActionScope::Case;
        descriptor.target_kind = Some(RustTestTargetKind::Unit);
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .ensure_action_experimental_features(&descriptor, cx))
                .is_ok()
        );
        descriptor.target_kind = Some(RustTestTargetKind::Doctest);
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-exact-doctest-actions-experimental",
                true,
                cx,
            )
        });
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .ensure_action_experimental_features(&descriptor, cx))
                .is_ok()
        );
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-exact-doctest-actions-experimental",
                false,
                cx,
            )
        });
        assert!(
            store
                .read_with(cx, |store, cx| store
                    .ensure_action_experimental_features(&descriptor, cx))
                .is_err()
        );
    }

    #[gpui::test]
    async fn rust_doctest_advertisement_uses_current_host_opt_in_without_gating_ordinary_nodes(
        cx: &mut gpui::TestAppContext,
    ) {
        struct ObservedRunner(Arc<AtomicU64>);
        impl RustTestDiscoveryRunner for ObservedRunner {
            fn discover(
                &self,
                _: RustTestDiscoveryRequest,
            ) -> BoxFuture<'static, Result<RustTestDiscoveryCapture>> {
                self.0.fetch_add(1, Ordering::SeqCst);
                async { Ok(capture()) }.boxed()
            }
        }

        let feature = "rust-exact-doctest-actions-experimental";
        cx.update(|cx| crate::task_store::set_experimental_feature_for_test(feature, false, cx));
        let project = crate::Project::test(crate::FakeFs::new(cx.executor()), [], cx).await;
        let (provider, structured) = project.read_with(cx, |project, _| {
            (
                project.rust_test_provider_store().clone(),
                project.structured_execution_store().clone(),
            )
        });
        let calls = Arc::new(AtomicU64::new(0));
        let exact = StructuredNodeId("exact".into());
        let generation = DiscoveryGeneration(7);
        provider.update(cx, |store, _| {
            let RustTestProviderStoreMode::Local { runner, .. } = &mut store.mode else {
                panic!("local provider fixture");
            };
            *runner = Arc::new(ObservedRunner(calls.clone()));
            store.generation = generation.0;
            store.next_run_id = 11;
            store.actions.insert(
                exact.clone(),
                RustTestActionDescriptor {
                    scope: RustTestActionScope::Case,
                    worktree_id: WorktreeId::from_usize(1),
                    manifest_path: project_path("Cargo.toml"),
                    label: "exact".into(),
                    cargo_args: vec!["test".into(), "--doc".into()],
                    executable_args: vec!["exact".into()],
                    target_kind: Some(RustTestTargetKind::Doctest),
                    doctest_case_names: None,
                    doctest_ignored_run_supported: false,
                },
            );
        });
        let snapshot = StructuredProviderSnapshot::discovery(
            RustTestProviderStore::provider_id(),
            generation,
            StructuredProviderStatus::Current,
            vec![
                StructuredNode {
                    experimental_feature: Some(feature.into()),
                    experimental_feature_enabled: Some(false),
                    debug_capability: None,
                    id: exact.clone(),
                    parent_id: None,
                    label: "exact".into(),
                    kind: StructuredNodeKind::Case,
                    path: None,
                },
                StructuredNode {
                    experimental_feature: None,
                    experimental_feature_enabled: None,
                    debug_capability: None,
                    id: StructuredNodeId("ordinary".into()),
                    parent_id: None,
                    label: "ordinary".into(),
                    kind: StructuredNodeKind::Case,
                    path: None,
                },
            ],
        );
        structured.update(cx, |store, _| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(project_generation, snapshot, None)
                .expect("host discovery DATA");
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &RustTestProviderStore::provider_id(),
                    crate::structured_execution::StructuredRun::new(
                        StructuredRunId("existing-run".into()),
                        generation,
                        vec![exact],
                    ),
                )
                .expect("existing run DATA");
        });
        let original = structured
            .read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&RustTestProviderStore::provider_id())
                    .cloned()
            })
            .expect("original host snapshot");
        let original_actions = provider.read_with(cx, |store, _| {
            store
                .actions
                .iter()
                .map(|(node, descriptor)| (node.clone(), format!("{descriptor:?}")))
                .collect::<BTreeMap<_, _>>()
        });
        for enabled in [true, false, true] {
            cx.update(|cx| {
                crate::task_store::set_experimental_feature_for_test(feature, enabled, cx)
            });
            cx.run_until_parked();
            cx.background_executor
                .timer(Duration::from_millis(200))
                .await;
            cx.run_until_parked();
            let actual = structured
                .read_with(cx, |store, _| {
                    store
                        .state()
                        .provider(&RustTestProviderStore::provider_id())
                        .cloned()
                })
                .expect("renewed host snapshot");
            assert_eq!(
                actual
                    .nodes
                    .first()
                    .expect("exact case")
                    .experimental_feature_enabled,
                Some(enabled)
            );
            assert_eq!(
                actual
                    .nodes
                    .get(1)
                    .expect("ordinary case")
                    .experimental_feature_enabled,
                None
            );
            let mut expected = original.clone();
            expected
                .nodes
                .first_mut()
                .expect("exact case")
                .experimental_feature_enabled = Some(enabled);
            assert_eq!(actual, expected, "only the availability hint may change");
            provider.read_with(cx, |store, _| {
                assert_eq!(store.generation(), generation);
                assert_eq!(store.next_run_id, 11);
                let actions = store
                    .actions
                    .iter()
                    .map(|(node, descriptor)| (node.clone(), format!("{descriptor:?}")))
                    .collect::<BTreeMap<_, _>>();
                assert_eq!(actions, original_actions);
                assert!(store.invalidation_task.is_ready());
                assert!(store.refresh_task.is_ready());
            });
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "flag changes must not run test discovery"
            );
        }
    }

    #[gpui::test]
    async fn rust_remote_action_plan_latches_doctest_revocation_and_checks_reply_requirements(
        cx: &mut gpui::TestAppContext,
    ) {
        use gpui::AppContext as _;

        struct HeldActionClient {
            replies: parking_lot::Mutex<
                std::collections::VecDeque<futures::channel::oneshot::Receiver<proto::Envelope>>,
            >,
            requests: AtomicU64,
            fallback: Arc<rpc::NoopProtoClient>,
        }

        impl rpc::ProtoClient for HeldActionClient {
            fn request(
                &self,
                envelope: proto::Envelope,
                _: &'static str,
            ) -> BoxFuture<'static, Result<proto::Envelope>> {
                let request =
                    <proto::ResolveRustTestAction as proto::EnvelopedMessage>::from_envelope(
                        envelope,
                    );
                let reply = self.replies.lock().pop_front();
                self.requests.fetch_add(1, Ordering::SeqCst);
                async move {
                    let request = request.context("expected typed Rust action request")?;
                    anyhow::ensure!(
                        request.protocol_version == RUST_TEST_DISCOVERY_PROTOCOL_VERSION
                    );
                    anyhow::ensure!(request.discovery_generation == 7);
                    reply
                        .context("unexpected action request")?
                        .await
                        .context("held action reply dropped")
                }
                .boxed()
            }
            fn send(&self, envelope: proto::Envelope, message_type: &'static str) -> Result<()> {
                self.fallback.send(envelope, message_type)
            }
            fn send_response(
                &self,
                envelope: proto::Envelope,
                message_type: &'static str,
            ) -> Result<()> {
                self.fallback.send_response(envelope, message_type)
            }
            fn message_handler_set(&self) -> &parking_lot::Mutex<rpc::ProtoMessageHandlerSet> {
                self.fallback.message_handler_set()
            }
            fn is_via_collab(&self) -> bool {
                false
            }
            fn has_wsl_interop(&self) -> bool {
                false
            }
        }

        let feature = "rust-exact-doctest-actions-experimental";
        cx.update(|cx| crate::task_store::set_experimental_feature_for_test(feature, true, cx));
        let filesystem = crate::FakeFs::new(cx.executor());
        filesystem
            .insert_tree(
                "/workspace",
                serde_json::json!({"Cargo.toml": "[workspace]"}),
            )
            .await;
        let project = crate::Project::test(filesystem, ["/workspace".as_ref()], cx).await;
        let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
        let structured_store =
            cx.update(|cx| cx.new(|_| StructuredExecutionStore::local(worktree_store.clone(), 1)));
        let generation = DiscoveryGeneration(7);
        let exact_node = StructuredNodeId("exact-case".into());
        let ordinary_node = StructuredNodeId("ordinary-case".into());
        let nodes = [
            (&exact_node, Some(feature.to_string())),
            (&ordinary_node, None),
        ]
        .into_iter()
        .map(|(id, experimental_feature)| StructuredNode {
            experimental_feature_enabled: experimental_feature.as_ref().map(|_| true),
            experimental_feature,
            debug_capability: None,
            id: id.clone(),
            parent_id: None,
            label: id.0.clone(),
            kind: StructuredNodeKind::Case,
            path: None,
        })
        .collect();
        structured_store.update(cx, |store, _| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(
                    project_generation,
                    StructuredProviderSnapshot::discovery(
                        RustTestProviderStore::provider_id(),
                        generation,
                        StructuredProviderStatus::Current,
                        nodes,
                    ),
                    None,
                )
                .expect("current remote discovery");
        });
        let mut replies = std::collections::VecDeque::new();
        let mut release = std::collections::VecDeque::new();
        for _ in 0..4 {
            let (sender, receiver) = futures::channel::oneshot::channel();
            release.push_back(sender);
            replies.push_back(receiver);
        }
        let client = Arc::new(HeldActionClient {
            replies: parking_lot::Mutex::new(replies),
            requests: AtomicU64::new(0),
            fallback: rpc::NoopProtoClient::new(),
        });
        let store = cx.update(|cx| {
            cx.new(|cx| {
                RustTestProviderStore::remote(
                    1,
                    AnyProtoClient::new(client.clone()),
                    worktree_store,
                    structured_store,
                    cx,
                )
            })
        });
        for (node, rejected_generation) in [
            (StructuredNodeId("unknown-case".into()), generation),
            (exact_node.clone(), DiscoveryGeneration(generation.0 - 1)),
        ] {
            let error = store
                .update(cx, |store, cx| {
                    store.plan_action(&node, rejected_generation, RustTestAction::Run, cx)
                })
                .await
                .expect_err("unobserved or stale action selection must not reach the host");
            assert!(
                error.downcast_ref::<RustTestRemoteError>().is_none(),
                "supported catalog selection errors must not claim unsupported host capability"
            );
            assert_eq!(client.requests.load(Ordering::SeqCst), 0);
        }
        for (index, node, returned_requirements, revoke, succeeds) in [
            (0, &exact_node, vec![feature.to_string()], true, false),
            (1, &exact_node, vec![feature.to_string()], false, true),
            (2, &ordinary_node, Vec::new(), true, true),
            (3, &exact_node, Vec::new(), false, false),
        ] {
            let pending = store.update(cx, |store, cx| {
                store.plan_action(node, generation, RustTestAction::Run, cx)
            });
            cx.run_until_parked();
            assert_eq!(client.requests.load(Ordering::SeqCst), index + 1);
            if revoke {
                cx.update(|cx| {
                    crate::task_store::set_experimental_feature_for_test(feature, false, cx)
                });
                cx.run_until_parked();
                cx.update(|cx| {
                    crate::task_store::set_experimental_feature_for_test(feature, true, cx)
                });
                cx.run_until_parked();
            }
            let plan = RustTestActionPlan::Task {
                run_id: StructuredRunId(format!("run-{index}")),
                discovery_generation: generation,
                scope_node_ids: vec![node.clone()],
                worktree_id: WorktreeId::from_proto(1),
                template: TaskTemplate {
                    experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
                    required_experimental_features: returned_requirements,
                    label: "current case".into(),
                    command: "cargo".into(),
                    args: vec!["test".into()],
                    ..Default::default()
                },
            };
            let response = proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json: encode_action_plan(&plan).expect("valid host reply DATA"),
                authorization_generation: 0,
            };
            release
                .pop_front()
                .expect("finite reply sender")
                .send(proto::EnvelopedMessage::into_envelope(
                    response, 0, None, None,
                ))
                .expect("pending action receiver");
            assert_eq!(
                pending.await.is_ok(),
                succeeds,
                "original operation must control the awaited reply"
            );
        }
        assert_eq!(client.requests.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn rust_test_action_protocol_errors_preserve_unsupported_kind_across_rpc() {
        use rpc::ErrorExt as _;

        ensure_rust_test_protocol(RUST_TEST_DISCOVERY_PROTOCOL_VERSION)
            .expect("current protocol remains supported");
        for protocol_version in [0, 1, u32::MAX] {
            let error = ensure_rust_test_protocol(protocol_version)
                .expect_err("incompatible protocol is refused locally");
            assert_eq!(
                error
                    .downcast_ref::<RustTestRemoteError>()
                    .expect("local typed unsupported host")
                    .kind,
                RustTestRemoteErrorKind::UnsupportedActionHost
            );
            let wire_error = tag_remote_rust_test_error(error).to_proto();
            assert_eq!(wire_error.code(), proto::ErrorCode::Internal);
            for request in ["ResolveRustTestAction", "UpdateRustTestRun"] {
                let remote_error = proto::RpcError::from_proto(&wire_error, request);
                assert_eq!(
                    remote_error.error_tag("reason"),
                    Some("rust_tools_unsupported")
                );
                let mapped_error = map_remote_action_error(remote_error);
                assert_eq!(
                    mapped_error
                        .downcast_ref::<RustTestRemoteError>()
                        .expect("RPC round trip preserves typed unsupported host")
                        .kind,
                    RustTestRemoteErrorKind::UnsupportedActionHost
                );
            }
        }
    }

    #[test]
    fn rust_test_action_protocol_requires_admitted_current_templates_and_preserves_harness_version()
    {
        assert_eq!(
            RUST_TEST_PROTOCOL_VERSION, 1,
            "machine harness protocol remains unchanged"
        );
        assert_eq!(RUST_TEST_DISCOVERY_PROTOCOL_VERSION, 2);
        let current = RustTestActionPlan::Task {
            run_id: StructuredRunId("current-run".into()),
            discovery_generation: DiscoveryGeneration(7),
            scope_node_ids: vec![StructuredNodeId("ordinary-case".into())],
            worktree_id: WorktreeId::from_proto(1),
            template: TaskTemplate {
                experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
                label: "ordinary case".into(),
                command: "cargo".into(),
                args: vec!["test".into()],
                ..Default::default()
            },
        };
        let encoded = encode_action_plan(&current).expect("current ordinary host plan");
        assert_eq!(
            decode_action_plan(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json: encoded.clone(),
                authorization_generation: 0
            })
            .expect("current ordinary action remains available"),
            current
        );
        let error = decode_action_plan(proto::RustTestActionPlanResponse {
            protocol_version: 1,
            plan_json: encoded.clone(),
            authorization_generation: 0,
        })
        .expect_err("old host protocol cannot assert exact-action admission");
        assert_eq!(
            error
                .downcast_ref::<RustTestRemoteError>()
                .expect("typed unsupported host")
                .kind,
            RustTestRemoteErrorKind::UnsupportedActionHost
        );
        let mut unadmitted: serde_json::Value = serde_json::from_str(&encoded).expect("wire plan");
        unadmitted["template"]["experimental_policy_version"] = serde_json::json!(0);
        assert!(
            decode_action_plan(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json: unadmitted.to_string(),
                authorization_generation: 0
            })
            .is_err(),
            "mixed-version host cannot omit generated policy"
        );
        unadmitted["template"]["experimental_policy_version"] =
            serde_json::json!(task::EXPERIMENTAL_TASK_POLICY_VERSION);
        unadmitted["template"]["required_experimental_features"] =
            serde_json::json!(["unknown-feature"]);
        assert!(
            decode_action_plan(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json: unadmitted.to_string(),
                authorization_generation: 0
            })
            .is_err(),
            "unknown metadata cannot grant permission"
        );
    }

    fn capture() -> RustTestDiscoveryCapture {
        RustTestDiscoveryCapture {
            toolchain: "1.95.0".to_string(),
            cargo_messages: CARGO_MESSAGES.to_string(),
            listings: serde_json::from_str(LISTINGS).expect("fixture listings should parse"),
        }
    }

    fn project_path(path: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(1),
            path: Arc::from(RelPath::from_unix_str(path).expect("fixture path should be relative")),
        }
    }

    fn project_path_in(worktree_id: usize, path: &str) -> ProjectPath {
        ProjectPath {
            worktree_id: WorktreeId::from_usize(worktree_id),
            path: Arc::from(RelPath::from_unix_str(path).expect("fixture path should be relative")),
        }
    }

    fn cargo_target(name: &str, kind: CargoTargetKind, source: &str) -> CargoTargetModel {
        CargoTargetModel {
            name: name.to_string(),
            kind,
            crate_types: Vec::new(),
            source_path: Some(project_path(source)),
            source_display_path: Some(source.to_string()),
            required_features: Vec::new(),
            edition: "2024".to_string(),
        }
    }

    fn cargo_workspace(worktree_id: usize, display_name: &str) -> CargoWorkspaceModel {
        let manifest_path = project_path_in(worktree_id, "Cargo.toml");
        CargoWorkspaceModel {
            key: CargoWorkspaceKey {
                root: manifest_path.clone(),
            },
            root_manifest: Some(manifest_path.clone()),
            display_name: display_name.to_string(),
            is_virtual: false,
            members: vec![CargoPackageModel {
                id: format!("{worktree_id}:Cargo.toml:rust-fixture@0.1.0"),
                name: "rust-fixture".to_string(),
                version: "0.1.0".to_string(),
                manifest_path,
                is_default_member: true,
                targets: vec![
                    cargo_target("rust_fixture", CargoTargetKind::Library, "src/lib.rs"),
                    cargo_target("cli", CargoTargetKind::Binary, "src/bin/cli.rs"),
                    cargo_target("api", CargoTargetKind::Test, "tests/api.rs"),
                    cargo_target("demo", CargoTargetKind::Example, "examples/demo.rs"),
                    cargo_target(
                        "throughput",
                        CargoTargetKind::Bench,
                        "benches/throughput.rs",
                    ),
                ],
                features: Vec::new(),
                dependencies: Vec::new(),
            }],
            dependency_graph: None,
            configuration: CargoWorkspaceConfiguration::unresolved(),
        }
    }

    fn cargo_snapshot(workspaces: Vec<CargoWorkspaceModel>) -> CargoWorkspaceSnapshot {
        CargoWorkspaceSnapshot {
            revision: 1,
            input_fingerprint: 1,
            workspaces,
            failures: Vec::new(),
            completeness: CargoSnapshotCompleteness::Complete,
        }
    }

    #[test]
    fn rust_test_semantic_runnable_requires_exact_structured_identity() {
        let runnable = RustSemanticRunnable {
            source: project_path("tests/api.rs"),
            overridden: false,
            cargo_args: vec![
                "test".into(),
                "--package".into(),
                "rust-fixture".into(),
                "--test".into(),
                "api".into(),
            ],
            executable_args: vec!["api::works".into(), "--exact".into()],
        };
        assert_eq!(
            semantic_runnable_identity(&runnable),
            Some((
                "rust-fixture",
                RustTestTargetKind::Integration,
                Some("api"),
                "api::works"
            ))
        );
        let snapshot = cargo_snapshot(vec![cargo_workspace(1, "fixture")]);
        let hint = match_semantic_runnable(&snapshot, &runnable).expect("exact Cargo target");
        assert_eq!(hint.package_id, "1:Cargo.toml:rust-fixture@0.1.0");
        assert_eq!(hint.target_name, "api");
        assert_eq!(hint.source, Some(project_path("tests/api.rs")));
        assert_eq!(
            hint.runnable.expect("typed selector").executable_args,
            ["api::works", "--exact"]
        );
        let duplicate = cargo_snapshot(vec![
            cargo_workspace(1, "first"),
            cargo_workspace(1, "second"),
        ]);
        assert!(match_semantic_runnable(&duplicate, &runnable).is_none());
        assert!(source_belongs_to_package(
            &project_path("tests/api.rs"),
            &project_path("Cargo.toml")
        ));
        assert!(!source_belongs_to_package(
            &project_path_in(2, "tests/api.rs"),
            &project_path("Cargo.toml")
        ));

        let mut missing_exact = runnable.clone();
        missing_exact.executable_args.pop();
        assert!(semantic_runnable_identity(&missing_exact).is_none());
        let mut overridden = runnable.clone();
        overridden.overridden = true;
        assert!(semantic_runnable_identity(&overridden).is_none());
        let mut ambiguous = runnable.clone();
        ambiguous.cargo_args.push("--all-targets".into());
        assert!(semantic_runnable_identity(&ambiguous).is_none());
        let mut extra_filter = runnable;
        extra_filter.executable_args.push("another_test".into());
        assert!(semantic_runnable_identity(&extra_filter).is_none());
    }

    #[test]
    fn rust_test_protocol_is_bounded_stable_and_partial_for_unknown_records() {
        let hint = RustTestSourceHint {
            package_id: PACKAGE_ID.to_string(),
            target_name: "rust_fixture".to_string(),
            target_kind: RustTestTargetKind::Unit,
            test_name: "unit::works".to_string(),
            source: Some(project_path("src/lib.rs")),
            runnable: Some(RustTestRunnableSelector {
                cargo_args: vec!["test".to_string(), "--lib".to_string()],
                executable_args: vec!["unit::works".to_string(), "--exact".to_string()],
            }),
        };
        let adapter = RustTestProtocolAdapter::default();
        let first = adapter.adapt(capture(), std::slice::from_ref(&hint));
        let second = adapter.adapt(capture(), &[hint]);
        assert_eq!(first, second);
        assert_eq!(first.capability, RustTestProtocolCapability::Supported);
        assert_eq!(first.harnesses.len(), 6);
        assert_eq!(first.cases.len(), 7);
        let unit = first
            .cases
            .iter()
            .find(|case| case.name == "unit::works")
            .expect("unit fixture should be present");
        assert_eq!(unit.source, Some(project_path("src/lib.rs")));
        assert_eq!(unit.runnable.cargo_args, ["test", "--lib"]);

        let mut unknown = capture();
        unknown
            .cargo_messages
            .push_str("{\"reason\":\"future-cargo-record\",\"secret\":\"not retained\"}\n");
        unknown.listings[0]
            .stdout
            .push_str("future listing record\n");
        let partial = adapter.adapt(unknown, &[]);
        assert_eq!(partial.capability, RustTestProtocolCapability::Partial);
        assert_eq!(partial.cases.len(), 7);
        assert_eq!(partial.diagnostics.len(), 2);
        assert!(
            partial
                .diagnostics
                .iter()
                .all(|diagnostic| !diagnostic.message.contains("not retained"))
        );

        let limited = RustTestProtocolAdapter::new(RustTestProtocolLimits {
            cases: 2,
            field_bytes: 64,
            diagnostics: 4,
            ..RustTestProtocolLimits::default()
        })
        .adapt(capture(), &[]);
        assert!(limited.truncated);
        assert_eq!(limited.cases.len(), 2);
        assert!(limited.diagnostics.len() <= 4);
        assert!(
            limited
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.message.len() <= 64)
        );

        let request = RustTestDiscoveryRequest {
            workspace_root: project_path(""),
            generation: 4,
            limits: RustTestProtocolLimits::default(),
            manifest_path: None,
            working_directory: None,
            environment: None,
            doctest_targets: Vec::new(),
        };
        let injected = smol::block_on(FixtureRunner.discover(request))
            .expect("injected fixture runner should complete without Cargo or network");
        assert_eq!(injected.toolchain, "1.95.0");
    }

    #[test]
    fn rust_workspace_adapts_ten_thousand_tests_without_machine_tools() {
        let listing = (0..MAX_RUST_TEST_CASES)
            .map(|index| format!("scale::case_{index:05}: test\n"))
            .collect::<String>();
        let mut large_capture = capture();
        large_capture.listings = vec![RustTestListingCapture {
            package_id: PACKAGE_ID.to_string(),
            target_name: "rust_fixture".to_string(),
            target_kind: RustTestTargetKind::Unit,
            mode: RustTestListingMode::All,
            stdout: listing,
        }];

        let first = RustTestProtocolAdapter::default().adapt(large_capture.clone(), &[]);
        let second = RustTestProtocolAdapter::default().adapt(large_capture, &[]);
        assert_eq!(first.cases.len(), MAX_RUST_TEST_CASES);
        assert!(!first.truncated);
        assert_eq!(
            first.cases.iter().map(|case| &case.id).collect::<Vec<_>>(),
            second.cases.iter().map(|case| &case.id).collect::<Vec<_>>()
        );
        assert!(first.cases.windows(2).all(|pair| pair[0].id < pair[1].id));
    }

    #[test]
    fn rust_test_provider_projects_stable_scoped_hierarchy_and_source_hints() {
        let workspace = cargo_workspace(1, "fixture");
        let workspace_snapshot = cargo_snapshot(vec![workspace.clone()]);
        let hint = RustTestSourceHint {
            package_id: PACKAGE_ID.to_string(),
            target_name: "rust_fixture".to_string(),
            target_kind: RustTestTargetKind::Unit,
            test_name: "unit::works".to_string(),
            source: Some(project_path("src/lib.rs")),
            runnable: None,
        };
        let first = project_provider_snapshot_for_test(
            &workspace_snapshot,
            vec![(workspace.clone(), capture())],
            std::slice::from_ref(&hint),
            Vec::new(),
            DiscoveryGeneration(1),
        );
        let second = project_provider_snapshot_for_test(
            &workspace_snapshot,
            vec![(workspace, capture())],
            &[hint],
            Vec::new(),
            DiscoveryGeneration(2),
        );
        assert_eq!(first.status, StructuredProviderStatus::Current);
        assert_eq!(
            first.nodes.iter().map(|node| &node.id).collect::<Vec<_>>(),
            second.nodes.iter().map(|node| &node.id).collect::<Vec<_>>()
        );
        let cases = first
            .nodes
            .iter()
            .filter(|node| node.kind == StructuredNodeKind::Case)
            .collect::<Vec<_>>();
        assert_eq!(cases.len(), 7);
        assert!(cases.iter().any(|node| {
            node.label == "works" && node.path == Some(project_path("src/lib.rs"))
        }));

        let scoped = scoped_cargo_snapshot(
            cargo_snapshot(vec![
                cargo_workspace(1, "visible"),
                cargo_workspace(2, "hidden"),
            ]),
            Some(&HashSet::from([WorktreeId::from_usize(1)])),
        );
        assert_eq!(scoped.workspaces.len(), 1);
        assert_eq!(scoped.workspaces[0].display_name, "visible");
    }

    #[test]
    fn rust_test_provider_does_not_share_source_hints_between_same_name_workspaces() {
        let first_workspace = cargo_workspace(1, "first");
        let second_workspace = cargo_workspace(2, "second");
        let snapshot = project_provider_snapshot_for_test(
            &cargo_snapshot(vec![first_workspace.clone(), second_workspace.clone()]),
            vec![(first_workspace, capture()), (second_workspace, capture())],
            &[RustTestSourceHint {
                package_id: "1:Cargo.toml:rust-fixture@0.1.0".to_string(),
                target_name: "rust_fixture".to_string(),
                target_kind: RustTestTargetKind::Unit,
                test_name: "unit::works".to_string(),
                source: Some(project_path_in(1, "src/lib.rs")),
                runnable: None,
            }],
            Vec::new(),
            DiscoveryGeneration(1),
        );
        let paths = snapshot
            .nodes
            .iter()
            .filter(|node| node.kind == StructuredNodeKind::Case && node.label == "works")
            .map(|node| node.path.clone())
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths, vec![Some(project_path_in(1, "src/lib.rs")), None]);
    }

    #[test]
    fn rust_test_lifecycle_privacy_excludes_spawn_details_from_remote_messages() {
        let sensitive = "fixture-only-task-environment /host-owner/private/task-shell";
        let local_event = StructuredTaskLifecycleEvent {
            task_id: TaskId("owned-task".into()),
            state: StructuredTaskState::SpawnError {
                message: sensitive.into(),
            },
        };
        let mut request = lifecycle_to_proto(
            42,
            DiscoveryGeneration(7),
            &StructuredRunId("owned-run".into()),
            &[StructuredNodeId("owned-case".into())],
            &local_event,
            vec![1],
        );
        let mut violations = Vec::new();
        if request
            .diagnostic
            .as_deref()
            .is_some_and(|text| text.contains(sensitive))
        {
            violations.push("outgoing lifecycle message retains sensitive spawn details");
        }
        assert_eq!(request.state, proto::RustTestTaskState::SpawnError as i32);
        assert_eq!(request.task_id, "owned-task");
        assert_eq!(request.run_id, "owned-run");
        assert!(
            request
                .diagnostic
                .as_ref()
                .is_some_and(|text| !text.is_empty())
        );
        request.diagnostic = Some(sensitive.into());
        let received = lifecycle_from_proto(&request).expect("supported old-client lifecycle");
        let StructuredTaskState::SpawnError { message } = received.state else {
            panic!("spawn failure must remain a terminal failure");
        };
        if message.contains(sensitive) {
            violations.push("incoming lifecycle retains arbitrary remote spawn details");
        }
        assert!(!message.is_empty());
        assert!(
            matches!(&local_event.state, StructuredTaskState::SpawnError { message } if message == sensitive)
        );
        for state in [
            StructuredTaskState::Queued,
            StructuredTaskState::Running { terminal_id: None },
            StructuredTaskState::Running {
                terminal_id: Some(StructuredTerminalId(17)),
            },
            StructuredTaskState::Completed {
                terminal_id: Some(StructuredTerminalId(17)),
                exit_code: Some(0),
                success: true,
            },
            StructuredTaskState::Completed {
                terminal_id: None,
                exit_code: Some(7),
                success: false,
            },
            StructuredTaskState::Cancelled {
                terminal_id: Some(StructuredTerminalId(17)),
                termination_confirmed: true,
            },
            StructuredTaskState::Cancelled {
                terminal_id: None,
                termination_confirmed: false,
            },
        ] {
            let event = StructuredTaskLifecycleEvent {
                task_id: local_event.task_id.clone(),
                state,
            };
            let mut request = lifecycle_to_proto(
                42,
                DiscoveryGeneration(7),
                &StructuredRunId("owned-run".into()),
                &[StructuredNodeId("owned-case".into())],
                &event,
                vec![1],
            );
            assert!(request.diagnostic.is_none());
            request.diagnostic = Some(sensitive.into());
            assert_eq!(
                lifecycle_from_proto(&request).expect("supported non-error lifecycle"),
                event
            );
        }
        assert!(
            violations.is_empty(),
            "lifecycle privacy violations: {violations:?}"
        );
    }

    #[test]
    fn rust_test_discovery_diagnostic_privacy_classifies_unknown_and_metadata_failures() {
        const SECRET: &str = "fixture-only-private-environment";
        const HOST_PATH: &str = "/host-owner/outside-worktree/private.rs";
        let raw_error = format!("unrecognized runner error: {SECRET} {HOST_PATH}");
        let unknown = project_provider_snapshot_for_test(
            &cargo_snapshot(vec![cargo_workspace(1, "fixture")]),
            Vec::new(),
            &[],
            vec![raw_error.clone()],
            DiscoveryGeneration(1),
        );
        assert_eq!(unknown.status, StructuredProviderStatus::Error);
        let diagnostic = unknown
            .diagnostic
            .as_deref()
            .expect("visible unknown failure category");
        assert!(diagnostic.contains("unclassified"), "{diagnostic}");
        assert!(!diagnostic.contains(SECRET) && !diagnostic.contains(HOST_PATH));
        for category in [
            CargoWorkspaceErrorCategory::CargoNotFound,
            CargoWorkspaceErrorCategory::CargoFailed,
            CargoWorkspaceErrorCategory::InvalidMetadata,
            CargoWorkspaceErrorCategory::UnsupportedMetadata,
            CargoWorkspaceErrorCategory::Disconnected,
            CargoWorkspaceErrorCategory::Cancelled,
            CargoWorkspaceErrorCategory::Internal,
        ] {
            let mut cargo = cargo_snapshot(Vec::new());
            cargo.failures.push(CargoCandidateFailure {
                manifest_path: project_path("Cargo.toml"),
                category,
                message: raw_error.clone(),
                has_stale_model: false,
            });
            let projected = project_provider_snapshot_for_test(
                &cargo,
                Vec::new(),
                &[],
                Vec::new(),
                DiscoveryGeneration(2),
            );
            assert_eq!(projected.status, StructuredProviderStatus::Error);
            let diagnostic = projected
                .diagnostic
                .as_deref()
                .expect("visible Cargo failure category");
            assert!(diagnostic.contains("Cargo metadata"), "{diagnostic}");
            assert!(!diagnostic.contains(SECRET) && !diagnostic.contains(HOST_PATH));
        }
    }

    #[test]
    fn rust_test_provider_isolates_partial_roots_and_retains_stale_results() {
        let good_workspace = cargo_workspace(1, "good");
        let failed_workspace = cargo_workspace(2, "failed");
        let mut cargo_snapshot = cargo_snapshot(vec![good_workspace.clone(), failed_workspace]);
        cargo_snapshot.completeness = CargoSnapshotCompleteness::Partial;
        cargo_snapshot.failures.push(CargoCandidateFailure {
            manifest_path: project_path_in(2, "Cargo.toml"),
            category: CargoWorkspaceErrorCategory::CargoFailed,
            message: "fixture failure".to_string(),
            has_stale_model: false,
        });
        let partial = project_provider_snapshot_for_test(
            &cargo_snapshot,
            vec![(good_workspace, capture())],
            &[],
            vec!["failed root".to_string()],
            DiscoveryGeneration(2),
        );
        assert_eq!(partial.status, StructuredProviderStatus::Partial);
        assert!(
            partial
                .nodes
                .iter()
                .any(|node| { node.kind == StructuredNodeKind::Case && node.label == "works" })
        );

        let previous = StructuredProviderSnapshot::discovery(
            RustTestProviderStore::provider_id(),
            DiscoveryGeneration(2),
            StructuredProviderStatus::Current,
            partial.nodes,
        );
        let mut failed = StructuredProviderSnapshot::discovery(
            RustTestProviderStore::provider_id(),
            DiscoveryGeneration(3),
            StructuredProviderStatus::Error,
            Vec::new(),
        );
        failed.diagnostic = Some("runner failed".to_string());
        retain_stale_discovery(&mut failed, Some(&previous));
        assert_eq!(failed.status, StructuredProviderStatus::Stale);
        assert_eq!(failed.nodes, previous.nodes);
        assert!(failed.partial);
        assert_eq!(failed.diagnostic.as_deref(), Some("runner failed"));
    }

    #[test]
    fn rust_test_debug_capability_matches_authoritative_planner() {
        let workspace = cargo_workspace(1, "fixture");
        let projection = project_provider_projection(
            &cargo_snapshot(vec![workspace.clone()]),
            vec![(workspace, capture())],
            &[],
            Vec::new(),
            DiscoveryGeneration(7),
        );
        let mut supported_cases = 0;
        let mut unsupported_cases = 0;
        for node in &projection.snapshot.nodes {
            let Some(descriptor) = projection.actions.get(&node.id) else {
                assert!(node.debug_capability.is_none());
                continue;
            };
            let capability = node
                .debug_capability
                .as_ref()
                .expect("authoritative capability");
            assert_eq!(capability, &rust_test_debug_capability(descriptor));
            let plan = plan_rust_test_action(
                descriptor,
                &node.id,
                DiscoveryGeneration(7),
                1,
                RustTestAction::Debug,
            );
            match capability {
                StructuredDebugCapability::Supported => {
                    assert_eq!(node.kind, StructuredNodeKind::Case);
                    assert!(matches!(
                        plan.expect("supported DAP plan"),
                        RustTestActionPlan::Debug { .. }
                    ));
                    supported_cases += 1;
                }
                StructuredDebugCapability::Unsupported(reason) => {
                    assert_eq!(plan.expect_err("unsupported Debug").to_string(), *reason);
                    if node.kind == StructuredNodeKind::Case {
                        assert!(matches!(
                            descriptor.target_kind,
                            Some(RustTestTargetKind::Doctest | RustTestTargetKind::Benchmark)
                        ));
                        unsupported_cases += 1;
                    }
                }
            }
        }
        assert_eq!((supported_cases, unsupported_cases), (5, 2));
        let mut descriptor = projection
            .actions
            .values()
            .find(|descriptor| descriptor.scope == RustTestActionScope::Case)
            .expect("case descriptor")
            .clone();
        descriptor.target_kind = None;
        assert_eq!(
            rust_test_debug_capability(&descriptor),
            StructuredDebugCapability::Unsupported(
                "The selected Rust test has no debuggable harness".into()
            )
        );
    }

    #[test]
    fn rust_test_actions_compile_exact_tasks_and_supported_debug_scenarios() {
        let workspace = cargo_workspace(1, "fixture");
        let projection = project_provider_projection(
            &cargo_snapshot(vec![workspace.clone()]),
            vec![(workspace, capture())],
            &[],
            Vec::new(),
            DiscoveryGeneration(7),
        );
        let case_id = projection
            .snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case && node.label == "works")
            .map(|node| node.id.clone())
            .expect("unit test case should be projected");
        let descriptor = projection
            .actions
            .get(&case_id)
            .expect("unit test case should have an action descriptor");
        let RustTestActionPlan::Task {
            run_id,
            template,
            scope_node_ids,
            ..
        } = plan_rust_test_action(
            descriptor,
            &case_id,
            DiscoveryGeneration(7),
            3,
            RustTestAction::Run,
        )
        .expect("unit test should compile to a task")
        else {
            panic!("run should produce a task plan")
        };
        assert_eq!(run_id.0, "rust-test-run-7-3");
        assert_eq!(scope_node_ids, std::slice::from_ref(&case_id));
        assert_eq!(template.command, "cargo");
        assert_eq!(
            template.args,
            [
                "test",
                "--package",
                "rust-fixture",
                "--lib",
                "--",
                "unit::works",
                "--exact",
                "--nocapture",
            ]
        );
        assert_eq!(template.save, SaveStrategy::All);
        assert!(template.use_new_terminal);
        assert!(!template.allow_concurrent_runs);

        let RustTestActionPlan::Debug { scenario, .. } = plan_rust_test_action(
            descriptor,
            &case_id,
            DiscoveryGeneration(7),
            4,
            RustTestAction::Debug,
        )
        .expect("unit test should compile to Cargo DAP") else {
            panic!("debug should produce a debug plan")
        };
        let BuildTaskDefinition::Template {
            task_template,
            locator_name,
        } = scenario
            .build
            .expect("debug scenario should build with Cargo")
        else {
            panic!("debug scenario should use a task template")
        };
        assert_eq!(locator_name.as_deref(), Some("rust-cargo-locator"));
        let delimiter = task_template
            .args
            .iter()
            .position(|argument| argument == "--")
            .expect("test args should use a harness delimiter");
        assert!(task_template.args[..delimiter].contains(&"--no-run".to_string()));

        let doctest_id = StructuredNodeId("doctest-case".to_string());
        let doctest = RustTestActionDescriptor {
            scope: RustTestActionScope::Case,
            worktree_id: WorktreeId::from_usize(1),
            manifest_path: project_path("Cargo.toml"),
            label: "src/lib.rs - example".to_string(),
            cargo_args: vec!["test".to_string(), "--doc".to_string()],
            executable_args: vec!["src/lib.rs - example".to_string()],
            target_kind: Some(RustTestTargetKind::Doctest),
            doctest_case_names: None,
            doctest_ignored_run_supported: false,
        };
        assert!(
            plan_rust_test_action(
                &doctest,
                &doctest_id,
                DiscoveryGeneration(7),
                5,
                RustTestAction::Debug,
            )
            .expect_err("doctest debug must be disabled")
            .to_string()
            .contains("doctests")
        );
    }

    #[test]
    fn rust_test_doctest_selection_preserves_one_case() {
        let workspace = cargo_workspace(1, "fixture");
        let mut captured_discovery = capture();
        let listing = captured_discovery
            .listings
            .iter_mut()
            .find(|listing| listing.target_kind == RustTestTargetKind::Doctest)
            .expect("doctest listing");
        listing.stdout = "src/lib.rs - add (line 3): test\nsrc/lib.rs - add (line 7): test\nall doctests ran in 0.71s; merged doctests compilation took 0.46s\n".into();
        let projection = project_provider_projection(
            &cargo_snapshot(vec![workspace.clone()]),
            vec![(workspace, captured_discovery)],
            &[],
            Vec::new(),
            DiscoveryGeneration(7),
        );
        let (node_id, descriptor) = projection
            .actions
            .iter()
            .find(|(_, descriptor)| descriptor.label == "src/lib.rs - add (line 3)")
            .expect("selected doctest");
        let RustTestActionPlan::Task { template, .. } = plan_rust_test_action(
            descriptor,
            node_id,
            DiscoveryGeneration(7),
            1,
            RustTestAction::Run,
        )
        .expect("singleton plan") else {
            panic!("Run must produce a Task");
        };
        assert_eq!(
            template.args,
            [
                "test",
                "--package",
                "rust-fixture",
                "--doc",
                "--",
                "3)",
                "--nocapture"
            ]
        );
        assert!(is_listing_summary(
            "all doctests ran in 0.71s; merged doctests compilation took 0.46s"
        ));
        for footer in [
            "all doctests ran in NaNs; merged doctests compilation took 0.46s",
            "all doctests ran in -1s; merged doctests compilation took 0.46s",
            "all doctests ran in 0.71s; merged doctests compilation took unknown",
            "unrecognized doctest output",
        ] {
            assert!(!is_listing_summary(footer), "{footer}");
        }
        let mut unavailable = descriptor.clone();
        unavailable.doctest_case_names = None;
        assert!(
            plan_rust_test_action(
                &unavailable,
                node_id,
                DiscoveryGeneration(7),
                1,
                RustTestAction::Run
            )
            .is_err()
        );
        unavailable.doctest_case_names =
            Some(vec![descriptor.label.clone(), descriptor.label.clone()].into());
        assert!(uniquely_selecting_doctest_token(&unavailable).is_err());
        unavailable.doctest_case_names = Some(
            vec![
                descriptor.label.clone(),
                format!("{} extra", descriptor.label),
            ]
            .into(),
        );
        assert!(uniquely_selecting_doctest_token(&unavailable).is_err());
        unavailable.doctest_case_names =
            Some(vec![descriptor.label.clone(), "malformed\0".into()].into());
        assert!(uniquely_selecting_doctest_token(&unavailable).is_err());
        let mut ignored = descriptor.clone();
        ignored.executable_args.push("--include-ignored".into());
        assert!(
            plan_rust_test_action(
                &ignored,
                node_id,
                DiscoveryGeneration(7),
                2,
                RustTestAction::Run
            )
            .expect_err("merged ignored doctest must not launch")
            .to_string()
            .contains("without executing")
        );
        ignored.doctest_ignored_run_supported = true;
        let RustTestActionPlan::Task {
            template: ignored_template,
            ..
        } = plan_rust_test_action(
            &ignored,
            node_id,
            DiscoveryGeneration(7),
            2,
            RustTestAction::Run,
        )
        .expect("ignored singleton")
        else {
            panic!("Task");
        };
        assert!(ignored_template.args.contains(&"--include-ignored".into()));
        assert!(!ignored_template.args.contains(&"--exact".into()));
        for stdout in [
            "src/lib.rs - add (line 3): test\nunrecognized doctest output\n".to_string(),
            format!("{}: test\n", "x".repeat(MAX_RUST_TEST_FIELD_BYTES + 1)),
            "src/lib.rs - add (line 3): test\nsrc/lib.rs - add (line 3): test\n".to_string(),
            "src/lib.rs - add (line 3): benchmark\n".to_string(),
        ] {
            let mut captured_discovery = capture();
            captured_discovery
                .listings
                .iter_mut()
                .find(|listing| listing.target_kind == RustTestTargetKind::Doctest)
                .expect("doctest listing")
                .stdout = stdout;
            let workspace = cargo_workspace(1, "fixture");
            let projection = project_provider_projection(
                &cargo_snapshot(vec![workspace.clone()]),
                vec![(workspace, captured_discovery)],
                &[],
                Vec::new(),
                DiscoveryGeneration(8),
            );
            let (node_id, descriptor) = projection
                .actions
                .iter()
                .find(|(_, descriptor)| {
                    descriptor.target_kind == Some(RustTestTargetKind::Doctest)
                        && descriptor.scope == RustTestActionScope::Case
                })
                .expect("partial doctest case remains visible");
            assert!(descriptor.doctest_case_names.is_none());
            assert!(
                plan_rust_test_action(
                    descriptor,
                    node_id,
                    DiscoveryGeneration(8),
                    1,
                    RustTestAction::Run
                )
                .is_err()
            );
        }
        let mut unsupported_arguments = descriptor.clone();
        unsupported_arguments
            .executable_args
            .push("--unsupported-fixture-option".into());
        assert!(
            plan_rust_test_action(
                &unsupported_arguments,
                node_id,
                DiscoveryGeneration(7),
                1,
                RustTestAction::Run
            )
            .is_err()
        );
        let mut missing_ignored = capture();
        missing_ignored.listings.retain(|listing| {
            !(listing.target_kind == RustTestTargetKind::Doctest
                && listing.mode == RustTestListingMode::Ignored)
        });
        let missing_snapshot = RustTestProtocolAdapter::default().adapt(missing_ignored, &[]);
        assert_eq!(
            missing_snapshot.capability,
            RustTestProtocolCapability::Partial
        );
        let mut at_limit = capture();
        at_limit
            .listings
            .retain(|listing| listing.target_kind == RustTestTargetKind::Doctest);
        let all_name = at_limit
            .listings
            .iter()
            .find(|listing| listing.mode == RustTestListingMode::All)
            .expect("all doctests")
            .stdout
            .clone();
        at_limit
            .listings
            .iter_mut()
            .find(|listing| listing.mode == RustTestListingMode::Ignored)
            .expect("ignored doctests")
            .stdout = all_name;
        let limited = RustTestProtocolAdapter::new(RustTestProtocolLimits {
            cases: 1,
            ..Default::default()
        })
        .adapt(at_limit, &[]);
        assert!(!limited.truncated);
        assert_eq!(limited.capability, RustTestProtocolCapability::Supported);
        assert_eq!(limited.cases.len(), 1);
        assert!(limited.cases.first().expect("single ignored case").ignored);
        let old_template = rust_test_task_template(descriptor);
        assert!(validate_action_template(&old_template).is_err());
        let mut broad_template = old_template;
        broad_template.args.retain(|argument| argument != "--exact");
        assert!(validate_action_template(&broad_template).is_err());
        let mut zero_template = template.clone();
        zero_template.args.push("--exact".into());
        assert!(validate_action_template(&zero_template).is_err());
        assert!(validate_action_template(&template).is_ok());
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_test_doctest_selection_preserves_one_case_in_cargo(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let result: Result<()> = async {
            let directory = tempfile::tempdir()?;
            let manifest_path = directory.path().join("Cargo.toml");
            let target_directory = directory.path().join("target");
            std::fs::create_dir(directory.path().join("src"))?;
            std::fs::write(directory.path().join("Cargo.lock"), r#"version = 4
    [[package]]
    name = "rust-fixture"
    version = "0.1.0"
    "#)?;
            std::fs::write(directory.path().join("src/lib.rs"), r#"/// Selected passing doctest.
    /// ```
    /// assert_eq!(rust_fixture::add(2, 3), 5);
    /// ```
    /// Unselected failing doctest.
    /// ```
    /// panic!("unselected fixture failure");
    /// ```
    /// Ignored failing doctest.
    /// ```ignore
    /// panic!("ignored fixture failure");
    /// ```
    pub fn add(left: u32, right: u32) -> u32 { left + right }
    "#)?;
            let mut rustdoc_version = None;
            for tool in ["cargo", "rustdoc"] {
                let mut command = std::process::Command::new(tool);
                command.arg("--version").current_dir(directory.path());
                let output = util::process::Child::spawn(
                    command,
                    std::process::Stdio::null(),
                    std::process::Stdio::piped(),
                    std::process::Stdio::piped(),
                )?.output().await?;
                assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
                let version = String::from_utf8(output.stdout)?;
                if tool == "rustdoc" {
                    rustdoc_version = Some(version.trim().to_string());
                }
                println!("{version}");
            }
            let rustdoc_version = rustdoc_version.context("actual rustdoc version")?;
            for edition in ["2024", "2021"] {
                std::fs::write(&manifest_path, format!(r#"[package]
    name = "rust-fixture"
    version = "0.1.0"
    edition = "{edition}"
    [workspace]
    "#))?;
                let mut captured_discovery = capture();
                captured_discovery.toolchain = rustdoc_version.clone();
                for mode in [RustTestListingMode::All, RustTestListingMode::Ignored] {
                    let mut command = std::process::Command::new("cargo");
                    command.args(["test", "--locked", "--offline", "--doc", "--", "--list"])
                        .current_dir(directory.path()).env("CARGO_TARGET_DIR", &target_directory);
                    if mode == RustTestListingMode::Ignored {
                        command.arg("--ignored");
                    }
                    let output = util::process::Child::spawn(
                        command,
                        std::process::Stdio::null(),
                        std::process::Stdio::piped(),
                        std::process::Stdio::piped(),
                    )?.output().await?;
                    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
                    captured_discovery.listings.iter_mut().find(|listing| {
                        listing.target_kind == RustTestTargetKind::Doctest && listing.mode == mode
                    }).context("actual doctest fixture listing")?.stdout = String::from_utf8(output.stdout)?;
                }
                let mut workspace = cargo_workspace(1, "fixture");
                for target in &mut workspace.members.iter_mut()
                    .find(|package| package.name == "rust-fixture")
                    .context("fixture package")?.targets {
                    target.edition = edition.to_string();
                }
                let projection = project_provider_projection(
                    &cargo_snapshot(vec![workspace.clone()]),
                    vec![(workspace, captured_discovery)],
                    &[],
                    Vec::new(),
                    DiscoveryGeneration(7),
                );
                for (line, success, ignored) in [(2, true, false), (6, false, false), (10, false, true)] {
                    let suffix = format!("(line {line})");
                    let (node_id, descriptor) = projection.actions.iter().find(|(_, descriptor)| {
                        descriptor.scope == RustTestActionScope::Case && descriptor.label.ends_with(&suffix)
                    }).context("actual discovered selected doctest")?;
                    for run in [1, 2] {
                        let plan = plan_rust_test_action(
                            descriptor,
                            node_id,
                            DiscoveryGeneration(7),
                            run,
                            RustTestAction::Run,
                        );
                        if ignored && edition == "2024" {
                            let error = plan.expect_err("merged ignored case must not launch");
                            assert!(error.to_string().contains("without executing"), "{error}");
                            println!("edition={edition}, selected={suffix}, run={run}, unavailable={error}");
                            continue;
                        }
                        let RustTestActionPlan::Task { template, .. } = plan? else {
                            bail!("expected Cargo Task");
                        };
                        assert_eq!(template.args.contains(&"--include-ignored".into()), ignored);
                        let mut command = std::process::Command::new(&template.command);
                        command.args(["--locked", "--offline"]).args(&template.args)
                            .current_dir(directory.path()).env("CARGO_TARGET_DIR", &target_directory);
                        let output = util::process::Child::spawn(
                            command,
                            std::process::Stdio::null(),
                            std::process::Stdio::piped(),
                            std::process::Stdio::piped(),
                        )?.output().await?;
                        let stdout = String::from_utf8(output.stdout)?;
                        assert_eq!(output.status.success(), success, "{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
                        assert!(stdout.contains("running 1 test"), "{stdout}");
                        assert!(stdout.contains("2 filtered out"), "{stdout}");
                        println!("edition={edition}, selected={suffix}, run={run}, command={:?}, success={success}\n{stdout}", template.args);
                    }
                }
            }
            Ok(())
        }.await;
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn rust_test_actions_bound_ignored_and_group_reruns() {
        let workspace = cargo_workspace(1, "fixture");
        let projection = project_provider_projection(
            &cargo_snapshot(vec![workspace.clone()]),
            vec![(workspace, capture())],
            &[],
            Vec::new(),
            DiscoveryGeneration(8),
        );
        let ignored = projection
            .snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case && node.label == "ignored")
            .expect("ignored case should be projected");
        let ignored_template = rust_test_task_template(
            projection
                .actions
                .get(&ignored.id)
                .expect("ignored case should remain executable"),
        );
        assert!(
            ignored_template
                .args
                .contains(&"--include-ignored".to_string())
        );

        let group = projection
            .snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Group
                    && node.label == "unit"
                    && projection.actions.contains_key(&node.id)
            })
            .expect("module group should be executable");
        let group_template = rust_test_task_template(
            projection
                .actions
                .get(&group.id)
                .expect("group should have an action descriptor"),
        );
        assert!(group_template.args.contains(&"unit".to_string()));
        assert!(!group_template.args.contains(&"--exact".to_string()));
        assert_eq!(MAX_RUST_TEST_RERUNS, 4);
    }

    #[test]
    fn rust_test_feature_remote_boundary_is_bounded_typed_and_restricted() {
        let workspace = cargo_workspace(1, "fixture");
        let projection = project_provider_projection(
            &cargo_snapshot(vec![workspace.clone()]),
            vec![(workspace, capture())],
            &[],
            Vec::new(),
            DiscoveryGeneration(12),
        );
        let case_id = projection
            .snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case && node.label == "works")
            .map(|node| node.id.clone())
            .expect("fixture case should be projected");
        let plan = plan_rust_test_action(
            projection
                .actions
                .get(&case_id)
                .expect("fixture case should be executable"),
            &case_id,
            DiscoveryGeneration(12),
            9,
            RustTestAction::Run,
        )
        .expect("fixture action should resolve");
        let plan_json = encode_action_plan(&plan).expect("action plan should encode");
        assert!(plan_json.len() <= MAX_RUST_TEST_ACTION_PLAN_BYTES);
        assert_eq!(
            decode_action_plan(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json,
                authorization_generation: 0,
            })
            .expect("action plan should decode"),
            plan
        );

        let RustTestActionPlan::Task {
            run_id,
            discovery_generation,
            scope_node_ids,
            mut template,
            ..
        } = plan
        else {
            panic!("run action should produce a task")
        };
        template
            .env
            .insert("TOKEN".to_string(), "secret".to_string());
        let unsafe_plan = RustTestActionWirePlan::Task {
            run_id: run_id.0.clone(),
            discovery_generation: discovery_generation.0,
            scope_node_ids: scope_node_ids
                .iter()
                .map(|node_id| node_id.0.clone())
                .collect(),
            worktree_id: 1,
            template,
        };
        let unsafe_plan = serde_json::to_string(&unsafe_plan).expect("fixture should serialize");
        assert!(
            decode_action_plan(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json: unsafe_plan,
                authorization_generation: 0,
            })
            .expect_err("remote task environments must be rejected")
            .to_string()
            .contains("unsupported task fields")
        );
        assert!(
            decode_action_plan(proto::RustTestActionPlanResponse {
                protocol_version: RUST_TEST_DISCOVERY_PROTOCOL_VERSION + 1,
                plan_json: "{}".to_string(),
                authorization_generation: 0,
            })
            .expect_err("feature mismatch must fail closed")
            .to_string()
            .contains("protocol mismatch")
        );

        let lifecycle = StructuredTaskLifecycleEvent {
            task_id: TaskId("task-remote".to_string()),
            state: StructuredTaskState::Completed {
                terminal_id: Some(StructuredTerminalId(17)),
                exit_code: Some(1),
                success: false,
            },
        };
        let lifecycle_proto = lifecycle_to_proto(
            42,
            discovery_generation,
            &run_id,
            &scope_node_ids,
            &lifecycle,
            vec![1],
        );
        assert_eq!(
            lifecycle_from_proto(&lifecycle_proto).expect("lifecycle should round-trip"),
            lifecycle
        );

        let mut restricted = cargo_snapshot(Vec::new());
        restricted.completeness = CargoSnapshotCompleteness::Partial;
        restricted.failures.push(CargoCandidateFailure {
            manifest_path: project_path("Cargo.toml"),
            category: CargoWorkspaceErrorCategory::Restricted,
            message: "sensitive host detail".to_string(),
            has_stale_model: false,
        });
        let restricted_projection = project_provider_projection(
            &restricted,
            Vec::new(),
            &[],
            Vec::new(),
            DiscoveryGeneration(13),
        );
        assert_eq!(
            restricted_projection.snapshot.status,
            StructuredProviderStatus::Restricted
        );
        assert!(restricted_projection.actions.is_empty());
        assert_eq!(
            restricted_projection.snapshot.diagnostic.as_deref(),
            Some("Trust the project to discover Rust tests")
        );
    }

    #[test]
    #[ignore = "stable-toolchain protocol gate; uses captured hermetic fixtures"]
    fn rust_test_protocol_fixture_matrix() {
        let snapshot = RustTestProtocolAdapter::default().adapt(capture(), &[]);
        assert_eq!(snapshot.toolchain, "1.95.0");
        assert_eq!(snapshot.capability, RustTestProtocolCapability::Supported);
        assert!(snapshot.diagnostics.is_empty());
        assert!(!snapshot.truncated);
        let target_kinds = snapshot
            .harnesses
            .iter()
            .map(|harness| harness.target_kind)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            target_kinds,
            BTreeSet::from([
                RustTestTargetKind::Unit,
                RustTestTargetKind::Integration,
                RustTestTargetKind::Binary,
                RustTestTargetKind::Example,
                RustTestTargetKind::Benchmark,
                RustTestTargetKind::Doctest,
            ])
        );
        assert!(snapshot.cases.iter().any(|case| case.ignored));
        assert!(snapshot.cases.iter().any(|case| {
            case.kind == RustTestCaseKind::Benchmark && case.harness_id.contains("Benchmark")
        }));
        assert!(snapshot.cases.iter().any(|case| {
            case.name.contains("src/lib.rs") && case.harness_id.contains("Doctest")
        }));
        assert!(snapshot.cases.iter().all(|case| {
            !case.runnable.cargo_args.is_empty()
                && case
                    .runnable
                    .executable_args
                    .ends_with(&["--exact".to_string()])
        }));
    }
}
