#[cfg(any(test, feature = "test-support"))]
use crate::transport::mock::ConnectGuard;
use crate::{
    SshConnectionOptions,
    protocol::MessageId,
    proxy::ProxyLaunchError,
    transport::{
        docker::{DockerConnectionOptions, DockerExecConnection},
        ssh::SshRemoteConnection,
        wsl::{WslConnectionOptions, WslRemoteConnection},
    },
};
use anyhow::{Context as _, Result, anyhow};
use askpass::EncryptedPassword;
use async_trait::async_trait;
use collections::HashMap;
use futures::{
    Future, FutureExt as _, StreamExt as _,
    channel::{
        mpsc::{self, Sender, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    future::{BoxFuture, Shared, WeakShared},
    select, select_biased,
    stream::BoxStream,
};
use gpui::{
    App, AppContext as _, AsyncApp, BackgroundExecutor, BorrowAppContext, Context, Entity,
    EventEmitter, FutureExt, Global, Task, TaskExt, WeakEntity,
};
use parking_lot::Mutex;

use release_channel::ReleaseChannel;
use rpc::{
    AnyProtoClient, ErrorExt, ProtoClient, ProtoMessageHandlerSet, RpcError,
    proto::{self, Envelope, EnvelopedMessage, PeerId, RequestMessage, build_typed_envelope},
};
use semver::Version;
use std::{
    collections::VecDeque,
    fmt,
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc, Weak,
        atomic::{AtomicU32, AtomicU64, Ordering::SeqCst},
    },
    time::{Duration, Instant},
};
use util::{
    ResultExt,
    paths::{PathStyle, RemotePathBuf},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RemoteOs {
    Linux,
    MacOs,
    Windows,
}

impl RemoteOs {
    pub fn as_str(&self) -> &'static str {
        match self {
            RemoteOs::Linux => "linux",
            RemoteOs::MacOs => "macos",
            RemoteOs::Windows => "windows",
        }
    }

    pub fn is_windows(&self) -> bool {
        matches!(self, RemoteOs::Windows)
    }

    /// A human-readable OS name for telemetry. Matches `client::telemetry::os_name`
    /// ignoring the compositor (as we run headless on remotes).
    pub fn display_name(&self) -> &'static str {
        match self {
            RemoteOs::Linux => "Linux",
            RemoteOs::MacOs => "macOS",
            RemoteOs::Windows => "Windows",
        }
    }
}

impl std::fmt::Display for RemoteOs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RemoteArch {
    X86_64,
    Aarch64,
}

impl RemoteArch {
    pub fn as_str(&self) -> &'static str {
        match self {
            RemoteArch::X86_64 => "x86_64",
            RemoteArch::Aarch64 => "aarch64",
        }
    }
}

impl std::fmt::Display for RemoteArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RemotePlatform {
    pub os: RemoteOs,
    pub arch: RemoteArch,
}

#[derive(Clone, Debug)]
pub struct CommandTemplate {
    pub program: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
}

/// Whether a command should be run with TTY allocation for interactive use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interactive {
    /// Allocate a pseudo-TTY for interactive terminal use.
    Yes,
    /// Do not allocate a TTY - for commands that communicate via piped stdio.
    No,
}

pub trait RemoteClientDelegate: Send + Sync {
    fn ask_password(
        &self,
        prompt: String,
        tx: oneshot::Sender<EncryptedPassword>,
        cx: &mut AsyncApp,
    );
    fn get_download_url(
        &self,
        platform: RemotePlatform,
        release_channel: ReleaseChannel,
        version: Option<Version>,
        cx: &mut AsyncApp,
    ) -> Task<Result<Option<String>>>;
    fn download_server_binary_locally(
        &self,
        platform: RemotePlatform,
        release_channel: ReleaseChannel,
        version: Option<Version>,
        cx: &mut AsyncApp,
    ) -> Task<Result<PathBuf>>;
    fn set_status(&self, status: Option<&str>, cx: &mut AsyncApp);
}

const MAX_MISSED_HEARTBEATS: usize = 5;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(5);
const INITIAL_CONNECTION_TIMEOUT: Duration =
    Duration::from_secs(if cfg!(debug_assertions) { 30 } else { 60 });

pub const MAX_RECONNECT_ATTEMPTS: usize = 3;

enum State {
    Connecting,
    Connected {
        remote_connection: Arc<dyn RemoteConnection>,
        delegate: Arc<dyn RemoteClientDelegate>,

        multiplex_task: Task<Result<()>>,
        heartbeat_task: Task<Result<()>>,
    },
    HeartbeatMissed {
        missed_heartbeats: usize,

        remote_connection: Arc<dyn RemoteConnection>,
        delegate: Arc<dyn RemoteClientDelegate>,

        multiplex_task: Task<Result<()>>,
        heartbeat_task: Task<Result<()>>,
    },
    Reconnecting,
    ReconnectFailed {
        remote_connection: Arc<dyn RemoteConnection>,
        delegate: Arc<dyn RemoteClientDelegate>,

        error: anyhow::Error,
        attempts: usize,
    },
    ReconnectExhausted,
    ServerNotRunning,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connecting => write!(f, "connecting"),
            Self::Connected { .. } => write!(f, "connected"),
            Self::Reconnecting => write!(f, "reconnecting"),
            Self::ReconnectFailed { .. } => write!(f, "reconnect failed"),
            Self::ReconnectExhausted => write!(f, "reconnect exhausted"),
            Self::HeartbeatMissed { .. } => write!(f, "heartbeat missed"),
            Self::ServerNotRunning { .. } => write!(f, "server not running"),
        }
    }
}

impl State {
    fn remote_connection(&self) -> Option<Arc<dyn RemoteConnection>> {
        match self {
            Self::Connected {
                remote_connection, ..
            } => Some(remote_connection.clone()),
            Self::HeartbeatMissed {
                remote_connection, ..
            } => Some(remote_connection.clone()),
            Self::ReconnectFailed {
                remote_connection, ..
            } => Some(remote_connection.clone()),
            _ => None,
        }
    }

    fn can_reconnect(&self) -> bool {
        match self {
            Self::Connected { .. }
            | Self::HeartbeatMissed { .. }
            | Self::ReconnectFailed { .. } => true,
            State::Connecting
            | State::Reconnecting
            | State::ReconnectExhausted
            | State::ServerNotRunning => false,
        }
    }

    fn is_reconnect_failed(&self) -> bool {
        matches!(self, Self::ReconnectFailed { .. })
    }

    fn is_reconnect_exhausted(&self) -> bool {
        matches!(self, Self::ReconnectExhausted { .. })
    }

    fn is_server_not_running(&self) -> bool {
        matches!(self, Self::ServerNotRunning)
    }

    fn is_reconnecting(&self) -> bool {
        matches!(self, Self::Reconnecting { .. })
    }

    fn heartbeat_recovered(self) -> Self {
        match self {
            Self::HeartbeatMissed {
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
                ..
            } => Self::Connected {
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
            },
            _ => self,
        }
    }

    fn heartbeat_missed(self) -> Self {
        match self {
            Self::Connected {
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
            } => Self::HeartbeatMissed {
                missed_heartbeats: 1,
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
            },
            Self::HeartbeatMissed {
                missed_heartbeats,
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
            } => Self::HeartbeatMissed {
                missed_heartbeats: missed_heartbeats + 1,
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
            },
            _ => self,
        }
    }
}

/// The state of the ssh connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    HeartbeatMissed,
    Reconnecting,
    Disconnected,
}

impl From<&State> for ConnectionState {
    fn from(value: &State) -> Self {
        match value {
            State::Connecting => Self::Connecting,
            State::Connected { .. } => Self::Connected,
            State::Reconnecting | State::ReconnectFailed { .. } => Self::Reconnecting,
            State::HeartbeatMissed { .. } => Self::HeartbeatMissed,
            State::ReconnectExhausted => Self::Disconnected,
            State::ServerNotRunning => Self::Disconnected,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProxyMode {
    Start,
    Reconnect,
    StartIfStopped(String),
}

impl ProxyMode {
    pub fn arguments(&self) -> Vec<String> {
        match self {
            Self::Start => Vec::new(),
            Self::Reconnect => vec!["--reconnect".into()],
            Self::StartIfStopped(instance) => vec!["--start-if-stopped".into(), instance.clone()],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoctorRemoteRestartOutcome {
    Ready,
    Incomplete,
    Cancelled,
}

pub struct DoctorRemoteRestartResult {
    pub outcome: DoctorRemoteRestartOutcome,
    pub replacement: Option<Entity<RemoteClient>>,
    pub shutdown_requested: bool,
    pub replacement_instance: Option<String>,
    pub detail: String,
}

impl DoctorRemoteRestartResult {
    fn incomplete(shutdown_requested: bool, detail: &str) -> Self {
        Self {
            outcome: DoctorRemoteRestartOutcome::Incomplete,
            replacement: None,
            shutdown_requested,
            replacement_instance: None,
            detail: detail.into(),
        }
    }
    fn cancelled(shutdown_requested: bool, detail: &str) -> Self {
        Self {
            outcome: DoctorRemoteRestartOutcome::Cancelled,
            ..Self::incomplete(shutdown_requested, detail)
        }
    }
}

fn validate_doctor_server_status(response: &proto::DoctorRemoteServerResponse) -> Result<()> {
    anyhow::ensure!(
        response.protocol_version == 1
            && response.process_id > 0
            && uuid::Uuid::parse_str(&response.instance).is_ok()
            && response.reason.len() <= 1024,
        "Unsupported or invalid remote restart capability"
    );
    Ok(())
}

struct DoctorStopObservationChild {
    child: util::process::Child,
    finished: bool,
}

impl Drop for DoctorStopObservationChild {
    fn drop(&mut self) {
        if !self.finished
            && let Err(error) = self.child.kill()
        {
            log::error!("Failed to terminate owned stop-observation helper: {error}");
        }
    }
}

async fn doctor_observe_stopped(
    connection: Arc<dyn RemoteConnection>,
    identifier: String,
    instance: String,
    process_id: u32,
    executor: BackgroundExecutor,
) -> Result<bool> {
    use futures::AsyncReadExt as _;
    uuid::Uuid::parse_str(&instance)?;
    anyhow::ensure!(
        !identifier.is_empty()
            && identifier.len() <= 64
            && identifier
                .bytes()
                .all(|character| character.is_ascii_alphanumeric()
                    || matches!(character, b'-' | b'_')),
        "Invalid server identifier"
    );
    let template = connection.build_command(
        Some(
            connection
                .doctor_server_binary()
                .context("Installed remote binary unavailable")?,
        ),
        &[
            "doctor-server-stopped".into(),
            "--identifier".into(),
            identifier,
            "--instance".into(),
            instance,
            "--process-id".into(),
            process_id.to_string(),
        ],
        &HashMap::default(),
        None,
        None,
        Interactive::No,
    )?;
    let mut command = std::process::Command::new(template.program);
    command.args(template.args).envs(template.env);
    let mut child = DoctorStopObservationChild {
        child: util::process::Child::spawn(
            command,
            std::process::Stdio::null(),
            std::process::Stdio::piped(),
            std::process::Stdio::null(),
        )?,
        finished: false,
    };
    let stdout = child
        .child
        .stdout
        .take()
        .context("Stop observation output unavailable")?;
    let operation = async {
        let mut output = Vec::new();
        stdout.take(33).read_to_end(&mut output).await?;
        anyhow::ensure!(output.len() <= 32, "Stop observation output exceeded bound");
        let status = child.child.status().await?;
        child.finished = true;
        anyhow::ensure!(
            status.success(),
            "Remote identity/stop check refused or failed"
        );
        match output.as_slice() {
            b"stopped\n" => Ok(true),
            b"running\n" => Ok(false),
            _ => Err(anyhow!("Unrecognized remote stop observation")),
        }
    };
    operation
        .with_timeout(Duration::from_secs(2), &executor)
        .await
        .map_err(|_| anyhow!("Stop observation timed out"))?
}

#[derive(Clone, Default)]
pub struct RemoteRequestActivity {
    handlers: Arc<std::sync::atomic::AtomicUsize>,
    sealed: Arc<std::sync::atomic::AtomicBool>,
}

impl RemoteRequestActivity {
    pub fn is_idle_for_control(&self) -> bool {
        !self.sealed.load(SeqCst) && self.handlers.load(SeqCst) == 1
    }

    pub fn seal_idle_control(&self) -> bool {
        if !self.is_idle_for_control() {
            return false;
        }
        self.sealed
            .compare_exchange(false, true, SeqCst, SeqCst)
            .is_ok()
    }
}

struct RemoteHandlerActivity(RemoteRequestActivity);
impl Drop for RemoteHandlerActivity {
    fn drop(&mut self) {
        self.0.handlers.fetch_sub(1, SeqCst);
    }
}

pub struct RemoteClient {
    client: Arc<ChannelClient>,
    unique_identifier: String,
    connection_options: RemoteConnectionOptions,
    path_style: PathStyle,
    platform: RemotePlatform,
    os_version: Option<String>,
    state: Option<State>,
}

#[derive(Debug)]
pub enum RemoteClientEvent {
    ConnectionStateChanged(ConnectionState),
    Disconnected { server_not_running: bool },
}

impl EventEmitter<RemoteClientEvent> for RemoteClient {}

/// Identifies the socket on the remote server so that reconnects
/// can re-join the same project.
pub enum ConnectionIdentifier {
    Setup(u64),
    Workspace(i64),
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl ConnectionIdentifier {
    pub fn setup() -> Self {
        Self::Setup(NEXT_ID.fetch_add(1, SeqCst))
    }

    // This string gets used in a socket name, and so must be relatively short.
    // The total length of:
    //   /home/{username}/.local/share/zed/server_state/{name}/stdout.sock
    // Must be less than about 100 characters
    //   https://unix.stackexchange.com/questions/367008/why-is-socket-path-length-limited-to-a-hundred-chars
    // So our strings should be at most 20 characters or so.
    fn to_string(&self, cx: &App) -> String {
        let identifier_prefix = match ReleaseChannel::global(cx) {
            ReleaseChannel::Stable => "".to_string(),
            release_channel => format!("{}-", release_channel.dev_name()),
        };
        match self {
            Self::Setup(setup_id) => format!("{identifier_prefix}setup-{setup_id}"),
            Self::Workspace(workspace_id) => {
                format!("{identifier_prefix}workspace-{workspace_id}",)
            }
        }
    }
}

pub async fn connect(
    connection_options: RemoteConnectionOptions,
    delegate: Arc<dyn RemoteClientDelegate>,
    cx: &mut AsyncApp,
) -> Result<Arc<dyn RemoteConnection>> {
    cx.update(|cx| {
        cx.update_default_global(|pool: &mut ConnectionPool, cx| {
            pool.connect(connection_options.clone(), delegate.clone(), cx)
        })
    })
    .await
    .map_err(|e| e.cloned())
}

/// Returns `true` if the global [`ConnectionPool`] already has a live
/// connection for the given options. Callers can use this to decide
/// whether to show interactive UI (e.g., a password modal) before
/// connecting.
pub fn has_active_connection(opts: &RemoteConnectionOptions, cx: &App) -> bool {
    cx.try_global::<ConnectionPool>().is_some_and(|pool| {
        matches!(
            pool.connections.get(opts),
            Some(ConnectionPoolEntry::Connected(remote))
                if remote.upgrade().is_some_and(|r| !r.has_been_killed())
        )
    })
}

impl RemoteClient {
    pub fn new(
        unique_identifier: ConnectionIdentifier,
        remote_connection: Arc<dyn RemoteConnection>,
        cancellation: oneshot::Receiver<()>,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut App,
    ) -> Task<Result<Option<Entity<Self>>>> {
        Self::new_with_mode(
            unique_identifier.to_string(cx),
            remote_connection,
            cancellation,
            delegate,
            ProxyMode::Start,
            cx,
        )
    }

    fn new_with_mode(
        unique_identifier: String,
        remote_connection: Arc<dyn RemoteConnection>,
        cancellation: oneshot::Receiver<()>,
        delegate: Arc<dyn RemoteClientDelegate>,
        mode: ProxyMode,
        cx: &mut App,
    ) -> Task<Result<Option<Entity<Self>>>> {
        cx.spawn(async move |cx| {
            let success = Box::pin(async move {
                let (outgoing_tx, outgoing_rx) = mpsc::unbounded::<Envelope>();
                let (incoming_tx, incoming_rx) = mpsc::unbounded::<Envelope>();
                let (connection_activity_tx, connection_activity_rx) = mpsc::channel::<()>(1);

                let client = cx.update(|cx| {
                    ChannelClient::new(
                        incoming_rx,
                        outgoing_tx,
                        cx,
                        "client",
                        remote_connection.has_wsl_interop(),
                    )
                });

                let path_style = remote_connection.path_style();
                let platform = remote_connection.remote_platform();
                let os_version = remote_connection.remote_os_version();
                let connection_options = remote_connection.connection_options();
                let connection_type = connection_options.connection_type();
                let this = cx.new(|_| Self {
                    client: client.clone(),
                    unique_identifier: unique_identifier.clone(),
                    connection_options,
                    path_style,
                    platform,
                    os_version: os_version.clone(),
                    state: Some(State::Connecting),
                });

                let io_task = remote_connection.start_proxy(
                    unique_identifier,
                    mode,
                    incoming_tx,
                    outgoing_rx,
                    connection_activity_tx,
                    delegate.clone(),
                    cx,
                );

                let ready = client
                    .wait_for_remote_started()
                    .with_timeout(INITIAL_CONNECTION_TIMEOUT, cx.background_executor())
                    .await;
                match ready {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        let mut error = "remote client exited before becoming ready".to_owned();
                        if let Some(status) = io_task.now_or_never() {
                            match status {
                                Ok(exit_code) => {
                                    error.push_str(&format!(", exit_code={exit_code:?}"))
                                }
                                Err(e) => error.push_str(&format!(", error={e:?}")),
                            }
                        }
                        let error = anyhow::anyhow!("{error}");
                        log::error!("failed to establish connection: {}", error);
                        return Err(error);
                    }
                    Err(_) => {
                        let mut error = String::new();
                        if let Some(status) = io_task.now_or_never() {
                            error.push_str("Client exited with ");
                            match status {
                                Ok(exit_code) => {
                                    error.push_str(&format!("exit_code {exit_code:?}"))
                                }
                                Err(e) => error.push_str(&format!("error {e:?}")),
                            }
                        } else {
                            error.push_str("client did not become ready within the timeout");
                        }
                        let error = anyhow::anyhow!("{error}");
                        log::error!("failed to establish connection: {error}");
                        return Err(error);
                    }
                }
                let multiplex_task = Self::monitor(this.downgrade(), io_task, cx);
                if let Err(error) = client.ping(HEARTBEAT_TIMEOUT).await {
                    log::error!("failed to establish connection: {}", error);
                    return Err(error);
                }

                let heartbeat_task = Self::heartbeat(this.downgrade(), connection_activity_rx, cx);

                this.update(cx, |this, _| {
                    this.state = Some(State::Connected {
                        remote_connection,
                        delegate,
                        multiplex_task,
                        heartbeat_task,
                    });
                });

                // Use the same `remote_*` property schema as the forwarded
                // remote events (see `client::telemetry::report_remote_event`)
                // so all remote-origin telemetry can be queried uniformly.
                telemetry::event!(
                    "Remote Connection Established",
                    remote = true,
                    remote_connection_type = connection_type,
                    remote_os_name = platform.os.display_name(),
                    remote_os_version = os_version,
                    remote_architecture = platform.arch.as_str(),
                );

                Ok(Some(this))
            });

            select! {
                _ = cancellation.fuse() => {
                    Ok(None)
                }
                result = success.fuse() =>  result
            }
        })
    }

    pub fn proto_client_from_channels(
        incoming_rx: mpsc::UnboundedReceiver<Envelope>,
        outgoing_tx: mpsc::UnboundedSender<Envelope>,
        cx: &App,
        name: &'static str,
        has_wsl_interop: bool,
    ) -> AnyProtoClient {
        ChannelClient::new(incoming_rx, outgoing_tx, cx, name, has_wsl_interop).into()
    }

    pub fn proto_client_from_channels_with_activity(
        incoming_rx: mpsc::UnboundedReceiver<Envelope>,
        outgoing_tx: mpsc::UnboundedSender<Envelope>,
        cx: &App,
        has_wsl_interop: bool,
    ) -> (AnyProtoClient, RemoteRequestActivity) {
        let client = ChannelClient::new(incoming_rx, outgoing_tx, cx, "server", has_wsl_interop);
        let activity = client.activity.clone();
        (client.into(), activity)
    }

    pub fn doctor_restart_status(
        &self,
        cx: &App,
    ) -> Task<Result<proto::DoctorRemoteServerResponse>> {
        if !matches!(self.state, Some(State::Connected { .. }))
            || self
                .remote_connection()
                .and_then(|connection| connection.doctor_server_binary())
                .is_none()
        {
            return Task::ready(Err(anyhow!(
                "This connected transport cannot establish Doctor restart ownership"
            )));
        }
        let request = self.client.request(proto::DoctorRemoteServer {
            expected_instance: String::new(),
            shutdown: false,
        });
        let executor = cx.background_executor().clone();
        let task_executor = executor.clone();
        executor.spawn(async move {
            let response = request
                .with_timeout(Duration::from_secs(5), &task_executor)
                .await
                .map_err(|_| anyhow!("Remote restart observation timed out"))??;
            validate_doctor_server_status(&response)?;
            Ok(response)
        })
    }

    pub fn doctor_restart(
        &mut self,
        expected: proto::DoctorRemoteServerResponse,
        cancelled: BoxFuture<'static, ()>,
        revalidate: impl FnMut(&App, bool) -> Result<()> + 'static,
        cx: &mut Context<Self>,
    ) -> Task<Result<DoctorRemoteRestartResult>> {
        let observation = self.doctor_restart_status(cx);
        self.doctor_restart_observed(expected, observation, cancelled, revalidate, cx)
    }

    fn doctor_restart_observed(
        &mut self,
        expected: proto::DoctorRemoteServerResponse,
        observation: Task<Result<proto::DoctorRemoteServerResponse>>,
        cancelled: BoxFuture<'static, ()>,
        mut revalidate: impl FnMut(&App, bool) -> Result<()> + 'static,
        cx: &mut Context<Self>,
    ) -> Task<Result<DoctorRemoteRestartResult>> {
        cx.spawn(async move |this, cx| {
            let cancelled = cancelled.shared();
            let observation = futures::future::select(observation.boxed(), cancelled.clone()).await;
            let observed = match observation {
                futures::future::Either::Left((response, _)) => response?,
                futures::future::Either::Right(_) => return Ok(DoctorRemoteRestartResult::cancelled(false, "Restart cancelled before shutdown")),
            };
            anyhow::ensure!(observed.instance == expected.instance && observed.process_id == expected.process_id && observed.exclusive && observed.idle, "Remote ownership/activity changed after review");
            if cancelled.clone().now_or_never().is_some() {
                return Ok(DoctorRemoteRestartResult::cancelled(false, "Restart cancelled before shutdown dispatch"));
            }
            // Project authority checks read this connection. Keep validation outside
            // its entity update, in the same foreground turn as shutdown dispatch.
            cx.update(|cx| revalidate(cx, false))?;
            let (request, connection, delegate, multiplex, identifier) = this.update(cx, |this, cx| {
                anyhow::ensure!(matches!(this.state, Some(State::Connected { .. })), "Remote connection changed before shutdown");
                let Some(State::Connected { remote_connection, delegate, multiplex_task, heartbeat_task }) = this.state.take() else {
                    anyhow::bail!("Remote connection changed before shutdown dispatch");
                };
                drop(heartbeat_task);
                // A lost acknowledgement must not cause automatic reconnect or replay against a server that may already be stopping.
                this.set_state(State::ServerNotRunning, cx);
                let request = this.client.request(proto::DoctorRemoteServer { expected_instance: expected.instance.clone(), shutdown: true });
                Ok::<_,anyhow::Error>((request, remote_connection, delegate, multiplex_task, this.unique_identifier.clone()))
            })??;
            let acknowledgement = request.with_timeout(Duration::from_secs(5), cx.background_executor());
            let response = match futures::future::select(acknowledgement.boxed_local(), cancelled.clone()).await {
                futures::future::Either::Left((response, _)) => response,
                futures::future::Either::Right(_) => return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled while waiting for shutdown acknowledgement; server state is uncertain; no replacement or replay")),
            };
            let response = match response {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => return Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Shutdown response unavailable or refused: {error}; server state is uncertain; no replacement or automatic reconnect started"))),
                Err(_) => return Ok(DoctorRemoteRestartResult::incomplete(true, "Shutdown acknowledgement timed out; server state is uncertain; no replacement or automatic reconnect started")),
            };
            if response.protocol_version != 1 || response.instance != expected.instance || !response.shutdown_accepted {
                return Ok(DoctorRemoteRestartResult::incomplete(true, "No accepted shutdown for the reviewed instance; server state is uncertain; no replacement or replay"));
            }
            let stopped_proxy = multiplex.with_timeout(Duration::from_secs(30), cx.background_executor());
            let stopped_proxy = match futures::future::select(stopped_proxy.boxed_local(), cancelled.clone()).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled while waiting for the old proxy; server stop is unobserved; no replacement or replay")),
            };
            match stopped_proxy {
                Ok(Ok(())) => {},
                Ok(Err(error)) => return Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Old proxy termination is uncertain: {error}; no replacement started"))),
                Err(_) => return Ok(DoctorRemoteRestartResult::incomplete(true, "Old proxy did not stop within 30 seconds; no replacement started")),
            }
            if cancelled.clone().now_or_never().is_some() {
                return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled after shutdown; reopen the stopped development project manually"));
            }
            let deadline = cx.background_executor().timer(Duration::from_secs(30));
            let stop_connection = connection.clone();
            let stop_identifier = identifier.clone();
            let stop_instance = expected.instance.clone();
            let stop_process_id = expected.process_id;
            let stop_executor = cx.background_executor().clone();
            let stop_observation = async move {
                loop {
                    if doctor_observe_stopped(stop_connection.clone(), stop_identifier.clone(), stop_instance.clone(), stop_process_id, stop_executor.clone()).await? { return Ok::<_,anyhow::Error>(()); }
                    stop_executor.timer(Duration::from_millis(100)).await;
                }
            };
            let stop_observation = futures::future::select(stop_observation.boxed_local(), deadline.boxed_local());
            let stop_observation = match futures::future::select(stop_observation.boxed_local(), cancelled.clone()).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled during stop observation; no replacement or replay")),
            };
            match stop_observation {
                futures::future::Either::Left((Ok(()), _)) => {},
                futures::future::Either::Left((Err(error), _)) => return Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Old server stop could not be established: {error}; no replacement started"))),
                futures::future::Either::Right(_) => return Ok(DoctorRemoteRestartResult::incomplete(true, "Old server stop observation timed out; no replacement started")),
            }
            if cancelled.clone().now_or_never().is_some() {
                return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled after observed stop; no replacement started"));
            }
            if let Err(error) = cx.update(|cx| revalidate(cx, true)) {
                return Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Restart authority changed after shutdown: {error}; no replacement started")));
            }
            if cancelled.clone().now_or_never().is_some() {
                return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled after authority recheck; no replacement started"));
            }
            let (cancel_sender, cancel_receiver) = oneshot::channel();
            let startup = cx.update(|cx| Self::new_with_mode(identifier, connection, cancel_receiver, delegate, ProxyMode::StartIfStopped(expected.instance.clone()), cx));
            let replacement = match futures::future::select(startup.boxed(), cancelled.clone()).await {
                futures::future::Either::Left((Ok(Some(replacement)), _)) => replacement,
                futures::future::Either::Left((Ok(None), _)) => return Ok(DoctorRemoteRestartResult::cancelled(true, "Replacement startup cancelled; an idle server may remain; no replay")),
                futures::future::Either::Left((Err(error), _)) => return Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Safe replacement startup failed: {error}; no replay"))),
                futures::future::Either::Right(_) => return Ok(DoctorRemoteRestartResult::cancelled(true, "Replacement startup cancelled; an idle server may remain; no replay")),
            };
            drop(cancel_sender);
            let ready = replacement.read_with(cx, |replacement, cx| replacement.doctor_restart_status(cx)).await;
            if cancelled.clone().now_or_never().is_some() {
                return Ok(DoctorRemoteRestartResult::cancelled(true, "Restart cancelled during readiness observation; an idle replacement may remain; no replay"));
            }
            if let Err(error) = cx.update(|cx| revalidate(cx, true)) {
                return Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Authority changed during replacement startup: {error}; an idle replacement may remain; original issue remains unverified")));
            }
            match ready {
                Ok(ready) if ready.instance != expected.instance && ready.exclusive => Ok(DoctorRemoteRestartResult { outcome: DoctorRemoteRestartOutcome::Ready, replacement: Some(replacement), shutdown_requested: true, replacement_instance: Some(ready.instance), detail: "Observed old server stop and a different ready exclusive server. Reopen through the existing Project/workspace owner and rerun the relevant check; the original issue is not verified fixed.".into() }),
                Ok(_) => Ok(DoctorRemoteRestartResult::incomplete(true, "Replacement identity/ownership was not established; original issue remains unverified")),
                Err(error) => Ok(DoctorRemoteRestartResult::incomplete(true, &format!("Replacement readiness could not be observed: {error}; original issue remains unverified"))),
            }
        })
    }

    pub fn proto_client_from_scoped_channels(
        incoming_rx: mpsc::UnboundedReceiver<(PeerId, Envelope)>,
        outgoing_tx: mpsc::UnboundedSender<Envelope>,
        cx: &App,
        name: &'static str,
        has_wsl_interop: bool,
    ) -> AnyProtoClient {
        ChannelClient::new_with_scopes(incoming_rx.boxed(), outgoing_tx, cx, name, has_wsl_interop)
            .into()
    }

    pub fn proto_client_from_scoped_channels_with_activity(
        incoming_rx: mpsc::UnboundedReceiver<(PeerId, Envelope)>,
        outgoing_tx: mpsc::UnboundedSender<Envelope>,
        cx: &App,
        has_wsl_interop: bool,
    ) -> (AnyProtoClient, RemoteRequestActivity) {
        let client = ChannelClient::new_with_scopes(
            incoming_rx.boxed(),
            outgoing_tx,
            cx,
            "server",
            has_wsl_interop,
        );
        let activity = client.activity.clone();
        (client.into(), activity)
    }

    pub fn shutdown_processes<T: RequestMessage>(
        &mut self,
        shutdown_request: Option<T>,
        executor: BackgroundExecutor,
    ) -> Option<impl Future<Output = ()> + use<T>> {
        let state = self.state.take()?;
        log::info!("shutting down remote processes");

        let State::Connected {
            multiplex_task,
            heartbeat_task,
            remote_connection,
            delegate,
        } = state
        else {
            return None;
        };

        let client = self.client.clone();

        Some(async move {
            if let Some(shutdown_request) = shutdown_request {
                client.send(shutdown_request).log_err();
                // We wait 50ms instead of waiting for a response, because
                // waiting for a response would require us to wait on the main thread
                // which we want to avoid in an `on_app_quit` callback.
                executor.timer(Duration::from_millis(50)).await;
            }

            // Drop `multiplex_task` because it owns our remote_connection_proxy_process, which is a
            // child of master_process.
            drop(multiplex_task);
            // Now drop the rest of state, which kills master process.
            drop(heartbeat_task);
            drop(remote_connection);
            drop(delegate);
        })
    }

    pub fn shutdown_recipe_test(&mut self, cx: &mut Context<Self>) -> Result<Task<()>> {
        anyhow::ensure!(
            matches!(&self.connection_options, RemoteConnectionOptions::Docker(options) if options.isolated_recipe_test.is_some()),
            "This connection is not an isolated recipe test"
        );
        let shutdown = self.shutdown_processes::<proto::ShutdownRemoteServer>(
            None,
            cx.background_executor().clone(),
        );
        Ok(cx.spawn(async move |_, _| {
            if let Some(shutdown) = shutdown {
                shutdown.await;
            }
        }))
    }

    fn reconnect(&mut self, cx: &mut Context<Self>) -> Result<()> {
        if matches!(&self.connection_options, RemoteConnectionOptions::Docker(options) if options.isolated_recipe_test.is_some())
        {
            self.set_state(State::ReconnectExhausted, cx);
            anyhow::bail!("Isolated recipe tests cannot reconnect or replay interrupted work");
        }
        let can_reconnect = self
            .state
            .as_ref()
            .map(|state| state.can_reconnect())
            .unwrap_or(false);
        if !can_reconnect {
            let state = if let Some(state) = self.state.as_ref() {
                state.to_string()
            } else {
                "no state set".to_string()
            };
            log::info!(
                "aborting reconnect, because not in state that allows reconnecting: {state}"
            );
            anyhow::bail!(
                "aborting reconnect, because not in state that allows reconnecting: {state}"
            );
        }

        let state = self.state.take().unwrap();
        let (attempts, remote_connection, delegate) = match state {
            State::Connected {
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
            }
            | State::HeartbeatMissed {
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task,
                ..
            } => {
                drop(multiplex_task);
                drop(heartbeat_task);
                (0, remote_connection, delegate)
            }
            State::ReconnectFailed {
                attempts,
                remote_connection,
                delegate,
                ..
            } => (attempts, remote_connection, delegate),
            State::Connecting
            | State::Reconnecting
            | State::ReconnectExhausted
            | State::ServerNotRunning => unreachable!(),
        };

        let attempts = attempts + 1;
        if attempts > MAX_RECONNECT_ATTEMPTS {
            log::error!(
                "Failed to reconnect to after {} attempts, giving up",
                MAX_RECONNECT_ATTEMPTS
            );
            self.set_state(State::ReconnectExhausted, cx);
            return Ok(());
        }

        self.set_state(State::Reconnecting, cx);

        log::info!(
            "Trying to reconnect to remote server... Attempt {}",
            attempts
        );

        let unique_identifier = self.unique_identifier.clone();
        let client = self.client.clone();
        let reconnect_task = cx.spawn(async move |this, cx| {
            macro_rules! failed {
                ($error:expr, $attempts:expr, $remote_connection:expr, $delegate:expr) => {
                    delegate.set_status(Some(&format!("{error:#}", error = $error)), cx);
                    return State::ReconnectFailed {
                        error: anyhow!($error),
                        attempts: $attempts,
                        remote_connection: $remote_connection,
                        delegate: $delegate,
                    };
                };
            }

            if let Err(error) = remote_connection
                .kill()
                .await
                .context("Failed to kill remote_connection process")
            {
                failed!(error, attempts, remote_connection, delegate);
            };

            let connection_options = remote_connection.connection_options();

            let (outgoing_tx, outgoing_rx) = mpsc::unbounded::<Envelope>();
            let (incoming_tx, incoming_rx) = mpsc::unbounded::<Envelope>();
            let (connection_activity_tx, connection_activity_rx) = mpsc::channel::<()>(1);

            let (remote_connection, io_task) = match async {
                let remote_connection = connect(connection_options, delegate.clone(), cx).await?;

                let io_task = remote_connection.start_proxy(
                    unique_identifier,
                    ProxyMode::Reconnect,
                    incoming_tx,
                    outgoing_rx,
                    connection_activity_tx,
                    delegate.clone(),
                    cx,
                );
                anyhow::Ok((remote_connection, io_task))
            }
            .await
            {
                Ok((remote_connection, io_task)) => (remote_connection, io_task),
                Err(error) => {
                    failed!(error, attempts, remote_connection, delegate);
                }
            };

            let multiplex_task = Self::monitor(this.clone(), io_task, cx);
            client.reconnect(incoming_rx, outgoing_tx, cx);

            if let Err(error) = client.resync(HEARTBEAT_TIMEOUT).await {
                failed!(error, attempts, remote_connection, delegate);
            };

            State::Connected {
                remote_connection,
                delegate,
                multiplex_task,
                heartbeat_task: Self::heartbeat(this.clone(), connection_activity_rx, cx),
            }
        });

        cx.spawn(async move |this, cx| {
            let new_state = reconnect_task.await;
            this.update(cx, |this, cx| {
                this.try_set_state(cx, |old_state| {
                    if old_state.is_reconnecting() {
                        match &new_state {
                            State::Connecting
                            | State::Reconnecting
                            | State::HeartbeatMissed { .. }
                            | State::ServerNotRunning => {}
                            State::Connected { .. } => {
                                log::info!("Successfully reconnected");
                            }
                            State::ReconnectFailed {
                                error, attempts, ..
                            } => {
                                log::error!(
                                    "Reconnect attempt {} failed: {:?}. Starting new attempt...",
                                    attempts,
                                    error
                                );
                            }
                            State::ReconnectExhausted => {
                                log::error!("Reconnect attempt failed and all attempts exhausted");
                            }
                        }
                        Some(new_state)
                    } else {
                        None
                    }
                });

                if this.state_is(State::is_reconnect_failed) {
                    this.reconnect(cx)
                } else if this.state_is(State::is_reconnect_exhausted) {
                    Ok(())
                } else {
                    log::debug!("State has transition from Reconnecting into new state while attempting reconnect.");
                    Ok(())
                }
            })
        })
        .detach_and_log_err(cx);

        Ok(())
    }

    fn heartbeat(
        this: WeakEntity<Self>,
        mut connection_activity_rx: mpsc::Receiver<()>,
        cx: &mut AsyncApp,
    ) -> Task<Result<()>> {
        let Ok(client) = this.read_with(cx, |this, _| this.client.clone()) else {
            return Task::ready(Err(anyhow!("remote_connectionRemoteClient lost")));
        };

        cx.spawn(async move |cx| {
            let mut missed_heartbeats = 0;

            let keepalive_timer = cx.background_executor().timer(HEARTBEAT_INTERVAL).fuse();
            futures::pin_mut!(keepalive_timer);

            loop {
                select_biased! {
                    result = connection_activity_rx.next().fuse() => {
                        if result.is_none() {
                            log::warn!("remote heartbeat: connection activity channel has been dropped. stopping.");
                            return Ok(());
                        }

                        if missed_heartbeats != 0 {
                            missed_heartbeats = 0;
                            let _ =this.update(cx, |this, cx| {
                                this.handle_heartbeat_result(missed_heartbeats, cx)
                            })?;
                        }
                    }
                    _ = keepalive_timer => {
                        log::debug!("Sending heartbeat to server...");

                        let result = select_biased! {
                            _ = connection_activity_rx.next().fuse() => {
                                Ok(())
                            }
                            ping_result = client.ping(HEARTBEAT_TIMEOUT).fuse() => {
                                ping_result
                            }
                        };

                        if result.is_err() {
                            missed_heartbeats += 1;
                            log::warn!(
                                "No heartbeat from server after {:?}. Missed heartbeat {} out of {}.",
                                HEARTBEAT_TIMEOUT,
                                missed_heartbeats,
                                MAX_MISSED_HEARTBEATS
                            );
                        } else if missed_heartbeats != 0 {
                            missed_heartbeats = 0;
                        } else {
                            continue;
                        }

                        let result = this.update(cx, |this, cx| {
                            this.handle_heartbeat_result(missed_heartbeats, cx)
                        })?;
                        if result.is_break() {
                            return Ok(());
                        }
                    }
                }

                keepalive_timer.set(cx.background_executor().timer(HEARTBEAT_INTERVAL).fuse());
            }
        })
    }

    fn handle_heartbeat_result(
        &mut self,
        missed_heartbeats: usize,
        cx: &mut Context<Self>,
    ) -> ControlFlow<()> {
        let state = self.state.take().unwrap();
        let next_state = if missed_heartbeats > 0 {
            state.heartbeat_missed()
        } else {
            state.heartbeat_recovered()
        };

        self.set_state(next_state, cx);

        if missed_heartbeats >= MAX_MISSED_HEARTBEATS {
            log::error!(
                "Missed last {} heartbeats. Reconnecting...",
                missed_heartbeats
            );

            self.reconnect(cx)
                .context("failed to start reconnect process after missing heartbeats")
                .log_err();
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    fn monitor(
        this: WeakEntity<Self>,
        io_task: Task<Result<i32>>,
        cx: &AsyncApp,
    ) -> Task<Result<()>> {
        cx.spawn(async move |cx| {
            let result = io_task.await;
            if this.read_with(cx, |this, _| {
                this.state_is(|state| matches!(state, State::ServerNotRunning))
            })? {
                return Ok(());
            }

            match result {
                Ok(exit_code) => {
                    if let Some(error) = ProxyLaunchError::from_exit_code(exit_code) {
                        match error {
                            ProxyLaunchError::ServerNotRunning => {
                                log::error!("failed to reconnect because server is not running");
                                this.update(cx, |this, cx| {
                                    this.set_state(State::ServerNotRunning, cx);
                                })?;
                            }
                        }
                    } else {
                        log::error!("proxy process terminated unexpectedly: {exit_code}");
                        this.update(cx, |this, cx| {
                            this.reconnect(cx).ok();
                        })?;
                    }
                }
                Err(error) => {
                    log::warn!(
                        "remote io task died with error: {:?}. reconnecting...",
                        error
                    );
                    this.update(cx, |this, cx| {
                        this.reconnect(cx).ok();
                    })?;
                }
            }

            Ok(())
        })
    }

    fn state_is(&self, check: impl FnOnce(&State) -> bool) -> bool {
        self.state.as_ref().is_some_and(check)
    }

    fn try_set_state(&mut self, cx: &mut Context<Self>, map: impl FnOnce(&State) -> Option<State>) {
        let new_state = self.state.as_ref().and_then(map);
        if let Some(new_state) = new_state {
            let previous = self.connection_state();
            self.state.replace(new_state);
            let current = self.connection_state();
            if previous != current {
                cx.emit(RemoteClientEvent::ConnectionStateChanged(current));
            }
            cx.notify();
        }
    }

    fn set_state(&mut self, state: State, cx: &mut Context<Self>) {
        log::info!("setting state to '{state}'");

        let previous = self.connection_state();
        let is_reconnect_exhausted = state.is_reconnect_exhausted();
        let is_server_not_running = state.is_server_not_running();
        self.state.replace(state);
        let current = self.connection_state();
        if previous != current {
            cx.emit(RemoteClientEvent::ConnectionStateChanged(current));
        }

        if is_reconnect_exhausted || is_server_not_running {
            cx.emit(RemoteClientEvent::Disconnected {
                server_not_running: is_server_not_running,
            });
        }
        cx.notify();
    }

    pub fn shell(&self) -> Option<String> {
        Some(self.remote_connection()?.shell())
    }

    pub fn default_system_shell(&self) -> Option<String> {
        Some(self.remote_connection()?.default_system_shell())
    }

    pub fn shares_network_interface(&self) -> bool {
        self.remote_connection()
            .map_or(false, |connection| connection.shares_network_interface())
    }

    pub fn has_wsl_interop(&self) -> bool {
        self.remote_connection()
            .map_or(false, |connection| connection.has_wsl_interop())
    }

    pub fn build_command(
        &self,
        program: Option<String>,
        args: &[String],
        env: &HashMap<String, String>,
        working_dir: Option<String>,
        port_forward: Option<(u16, String, u16)>,
        interactive: Interactive,
    ) -> Result<CommandTemplate> {
        let Some(connection) = self.remote_connection() else {
            return Err(anyhow!("no remote connection"));
        };
        connection.build_command(program, args, env, working_dir, port_forward, interactive)
    }

    pub fn build_forward_ports_command(
        &self,
        forwards: Vec<(u16, String, u16)>,
    ) -> Result<CommandTemplate> {
        let Some(connection) = self.remote_connection() else {
            return Err(anyhow!("no remote connection"));
        };
        connection.build_forward_ports_command(forwards)
    }

    pub fn upload_directory(
        &self,
        src_path: PathBuf,
        dest_path: RemotePathBuf,
        cx: &App,
    ) -> Task<Result<()>> {
        let Some(connection) = self.remote_connection() else {
            return Task::ready(Err(anyhow!("no remote connection")));
        };
        connection.upload_directory(src_path, dest_path, cx)
    }

    pub fn proto_client(&self) -> AnyProtoClient {
        self.client.clone().into()
    }

    pub fn connection_options(&self) -> RemoteConnectionOptions {
        self.connection_options.clone()
    }

    pub fn connection(&self) -> Option<Arc<dyn RemoteConnection>> {
        if let State::Connected {
            remote_connection, ..
        } = self.state.as_ref()?
        {
            Some(remote_connection.clone())
        } else {
            None
        }
    }

    pub fn connection_state(&self) -> ConnectionState {
        self.state
            .as_ref()
            .map(ConnectionState::from)
            .unwrap_or(ConnectionState::Disconnected)
    }

    pub fn is_disconnected(&self) -> bool {
        self.connection_state() == ConnectionState::Disconnected
    }

    pub fn path_style(&self) -> PathStyle {
        self.path_style
    }

    /// The platform (OS and architecture) of the remote host, detected during
    /// connection setup.
    pub fn remote_platform(&self) -> RemotePlatform {
        self.platform
    }

    /// The OS version of the remote host (e.g. `"ubuntu 24.04"`), detected
    /// during connection setup. `None` if it could not be determined.
    pub fn remote_os_version(&self) -> Option<String> {
        self.os_version.clone()
    }

    /// A stable identifier for the kind of remote connection (e.g. `"ssh"`,
    /// `"wsl"`, `"docker"`, `"podman"`).
    pub fn connection_type(&self) -> &'static str {
        self.connection_options.connection_type()
    }

    /// Forcibly disconnects from the remote server by killing the underlying connection.
    /// This will trigger the reconnection logic if reconnection attempts remain.
    /// Useful for testing reconnection behavior in real environments.
    pub fn force_disconnect(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let Some(connection) = self.remote_connection() else {
            return Task::ready(Err(anyhow!("no active remote connection to disconnect")));
        };

        log::info!("force_disconnect: killing remote connection");

        cx.spawn(async move |_, _| {
            connection.kill().await?;
            Ok(())
        })
    }

    /// Simulates a timeout by pausing heartbeat responses.
    /// This will cause heartbeat failures and eventually trigger reconnection
    /// after MAX_MISSED_HEARTBEATS are missed.
    /// Useful for testing timeout behavior in real environments.
    pub fn force_heartbeat_timeout(&mut self, attempts: usize, cx: &mut Context<Self>) {
        log::info!("force_heartbeat_timeout: triggering heartbeat failure state");

        if let Some(State::Connected {
            remote_connection,
            delegate,
            multiplex_task,
            heartbeat_task,
        }) = self.state.take()
        {
            self.set_state(
                if attempts == 0 {
                    State::HeartbeatMissed {
                        missed_heartbeats: MAX_MISSED_HEARTBEATS,
                        remote_connection,
                        delegate,
                        multiplex_task,
                        heartbeat_task,
                    }
                } else {
                    State::ReconnectFailed {
                        remote_connection,
                        delegate,
                        error: anyhow!("forced heartbeat timeout"),
                        attempts,
                    }
                },
                cx,
            );

            self.reconnect(cx)
                .context("failed to start reconnect after forced timeout")
                .log_err();
        } else {
            log::warn!("force_heartbeat_timeout: not in Connected state, ignoring");
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn force_server_not_running(&mut self, cx: &mut Context<Self>) {
        self.set_state(State::ServerNotRunning, cx);
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn simulate_disconnect(&self, client_cx: &mut App) -> Task<()> {
        let opts = self.connection_options();
        client_cx.spawn(async move |cx| {
            let connection = cx.update_global(|c: &mut ConnectionPool, _| {
                if let Some(ConnectionPoolEntry::Connected(c)) = c.connections.get(&opts) {
                    if let Some(connection) = c.upgrade() {
                        connection
                    } else {
                        panic!("connection was dropped")
                    }
                } else {
                    panic!("missing test connection")
                }
            });

            connection.simulate_disconnect(cx);
        })
    }

    /// Creates a mock connection pair for testing.
    ///
    /// This is the recommended way to create mock remote connections for tests.
    /// It returns the `MockConnectionOptions` (which can be passed to create a
    /// `HeadlessProject`), an `AnyProtoClient` for the server side and a
    /// `ConnectGuard` for the client side which blocks the connection from
    /// being established until dropped.
    ///
    /// # Example
    /// ```ignore
    /// let (opts, server_session, connect_guard) = RemoteClient::fake_server(cx, server_cx);
    /// // Set up HeadlessProject with server_session...
    /// drop(connect_guard);
    /// let client = RemoteClient::fake_client(opts, cx).await;
    /// ```
    #[cfg(any(test, feature = "test-support"))]
    pub fn fake_server(
        client_cx: &mut gpui::TestAppContext,
        server_cx: &mut gpui::TestAppContext,
    ) -> (RemoteConnectionOptions, AnyProtoClient, ConnectGuard) {
        use crate::transport::mock::MockConnection;
        let (opts, server_client, connect_guard) = MockConnection::new(client_cx, server_cx);
        (opts.into(), server_client, connect_guard)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn fake_server_with_activity(
        client_cx: &mut gpui::TestAppContext,
        server_cx: &mut gpui::TestAppContext,
    ) -> (
        RemoteConnectionOptions,
        AnyProtoClient,
        ConnectGuard,
        RemoteRequestActivity,
    ) {
        let (options, client, guard, activity) =
            crate::transport::mock::MockConnection::new_with_activity(client_cx, server_cx);
        (options.into(), client, guard, activity)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn fake_server_scoped(
        peer: PeerId,
        client_cx: &mut gpui::TestAppContext,
        server_cx: &mut gpui::TestAppContext,
    ) -> Result<(
        RemoteConnectionOptions,
        AnyProtoClient,
        ConnectGuard,
        RemoteRequestActivity,
    )> {
        let (options, client, guard, activity) =
            crate::transport::mock::MockConnection::new_scoped_with_activity(
                peer, client_cx, server_cx,
            )?;
        Ok((options.into(), client, guard, activity))
    }

    /// Registers a new mock server for existing connection options.
    ///
    /// Use this to simulate reconnection: after forcing a disconnect, register
    /// a new server so the next `connect()` call succeeds.
    #[cfg(any(test, feature = "test-support"))]
    pub fn fake_server_with_opts(
        opts: &RemoteConnectionOptions,
        client_cx: &mut gpui::TestAppContext,
        server_cx: &mut gpui::TestAppContext,
    ) -> (AnyProtoClient, ConnectGuard) {
        use crate::transport::mock::MockConnection;
        let mock_opts = match opts {
            RemoteConnectionOptions::Mock(mock_opts) => mock_opts.clone(),
            _ => panic!("fake_server_with_opts requires Mock connection options"),
        };
        MockConnection::new_with_opts(mock_opts, client_cx, server_cx)
    }

    /// Creates a `RemoteClient` connected to a mock server.
    ///
    /// Call `fake_server` first to get the connection options, set up the
    /// `HeadlessProject` with the server session, then call this method
    /// to create the client.
    #[cfg(any(test, feature = "test-support"))]
    pub async fn connect_mock(
        opts: RemoteConnectionOptions,
        client_cx: &mut gpui::TestAppContext,
    ) -> Entity<Self> {
        assert!(matches!(opts, RemoteConnectionOptions::Mock(..)));
        use crate::transport::mock::MockDelegate;
        let (_tx, rx) = oneshot::channel();
        let mut cx = client_cx.to_async();
        let connection = connect(opts, Arc::new(MockDelegate), &mut cx)
            .await
            .unwrap();
        client_cx
            .update(|cx| {
                Self::new(
                    ConnectionIdentifier::setup(),
                    connection,
                    rx,
                    Arc::new(MockDelegate),
                    cx,
                )
            })
            .await
            .unwrap()
            .unwrap()
    }

    pub fn remote_connection(&self) -> Option<Arc<dyn RemoteConnection>> {
        self.state
            .as_ref()
            .and_then(|state| state.remote_connection())
    }
}

enum ConnectionPoolEntry {
    Connecting(WeakShared<Task<Result<Arc<dyn RemoteConnection>, Arc<anyhow::Error>>>>),
    Connected(Weak<dyn RemoteConnection>),
}

#[derive(Default)]
struct ConnectionPool {
    connections: HashMap<RemoteConnectionOptions, ConnectionPoolEntry>,
}

impl Global for ConnectionPool {}

impl ConnectionPool {
    fn connect(
        &mut self,
        opts: RemoteConnectionOptions,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut App,
    ) -> Shared<Task<Result<Arc<dyn RemoteConnection>, Arc<anyhow::Error>>>> {
        let connection = self.connections.get(&opts);
        match connection {
            Some(ConnectionPoolEntry::Connecting(task)) => {
                if let Some(task) = task.upgrade() {
                    log::debug!("Connecting task is still alive");
                    cx.spawn(async move |cx| {
                        delegate.set_status(Some("Waiting for existing connection attempt"), cx)
                    })
                    .detach();
                    return task;
                }
                log::debug!("Connecting task is dead, removing it and restarting a connection");
                self.connections.remove(&opts);
            }
            Some(ConnectionPoolEntry::Connected(remote)) => {
                if let Some(remote) = remote.upgrade()
                    && !remote.has_been_killed()
                {
                    log::debug!("Connection is still alive");
                    return Task::ready(Ok(remote)).shared();
                }
                log::debug!("Connection is dead, removing it and restarting a connection");
                self.connections.remove(&opts);
            }
            None => {
                log::debug!("No existing connection found, starting a new one");
            }
        }

        let task = cx
            .spawn({
                let opts = opts.clone();
                let delegate = delegate.clone();
                async move |cx| {
                    let connection = match opts.clone() {
                        RemoteConnectionOptions::Ssh(opts) => {
                            SshRemoteConnection::new(opts, delegate, cx)
                                .await
                                .map(|connection| Arc::new(connection) as Arc<dyn RemoteConnection>)
                        }
                        RemoteConnectionOptions::Wsl(opts) => {
                            WslRemoteConnection::new(opts, delegate, cx)
                                .await
                                .map(|connection| Arc::new(connection) as Arc<dyn RemoteConnection>)
                        }
                        RemoteConnectionOptions::Docker(opts) => {
                            DockerExecConnection::new(opts, delegate, cx)
                                .await
                                .map(|connection| Arc::new(connection) as Arc<dyn RemoteConnection>)
                        }
                        #[cfg(any(test, feature = "test-support"))]
                        RemoteConnectionOptions::Mock(opts) => match cx.update(|cx| {
                            cx.default_global::<crate::transport::mock::MockConnectionRegistry>()
                                .take(&opts)
                        }) {
                            Some(connection) => Ok(connection.await as Arc<dyn RemoteConnection>),
                            None => Err(anyhow!(
                                "Mock connection not found. Call MockConnection::new() first."
                            )),
                        },
                    };

                    cx.update_global(|pool: &mut Self, _| {
                        debug_assert!(matches!(
                            pool.connections.get(&opts),
                            Some(ConnectionPoolEntry::Connecting(_))
                        ));
                        match connection {
                            Ok(connection) => {
                                pool.connections.insert(
                                    opts.clone(),
                                    ConnectionPoolEntry::Connected(Arc::downgrade(&connection)),
                                );
                                Ok(connection)
                            }
                            Err(error) => {
                                pool.connections.remove(&opts);
                                Err(Arc::new(error))
                            }
                        }
                    })
                }
            })
            .shared();
        if let Some(task) = task.downgrade() {
            self.connections
                .insert(opts.clone(), ConnectionPoolEntry::Connecting(task));
        }
        task
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RemoteConnectionOptions {
    Ssh(SshConnectionOptions),
    Wsl(WslConnectionOptions),
    Docker(DockerConnectionOptions),
    #[cfg(any(test, feature = "test-support"))]
    Mock(crate::transport::mock::MockConnectionOptions),
}

impl RemoteConnectionOptions {
    pub fn display_name(&self) -> String {
        match self {
            RemoteConnectionOptions::Ssh(opts) => opts
                .nickname
                .clone()
                .unwrap_or_else(|| opts.host.to_string()),
            RemoteConnectionOptions::Wsl(opts) => opts.distro_name.clone(),
            RemoteConnectionOptions::Docker(opts) => {
                if opts.use_podman {
                    format!("[podman] {}", opts.name)
                } else {
                    opts.name.clone()
                }
            }
            #[cfg(any(test, feature = "test-support"))]
            RemoteConnectionOptions::Mock(opts) => format!("mock-{}", opts.id),
        }
    }

    /// A stable identifier for the kind of remote connection, suitable for
    /// telemetry (e.g. `"ssh"`, `"wsl"`, `"docker"`, `"podman"`).
    pub fn connection_type(&self) -> &'static str {
        match self {
            RemoteConnectionOptions::Ssh(_) => "ssh",
            RemoteConnectionOptions::Wsl(_) => "wsl",
            RemoteConnectionOptions::Docker(opts) => {
                if opts.use_podman {
                    "podman"
                } else {
                    "docker"
                }
            }
            #[cfg(any(test, feature = "test-support"))]
            RemoteConnectionOptions::Mock(_) => "mock",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use rpc::{ErrorCodeExt, proto::ErrorCode};

    #[test]
    fn doctor_restart_request_activity_seals_only_idle_control() {
        let activity = RemoteRequestActivity::default();
        assert!(
            !activity.seal_idle_control(),
            "No control handler means no authority"
        );
        activity.handlers.store(2, SeqCst);
        assert!(
            !activity.seal_idle_control(),
            "An in-flight host request blocks shutdown"
        );
        activity.handlers.store(1, SeqCst);
        assert!(activity.is_idle_for_control());
        assert!(activity.seal_idle_control());
        assert!(
            !activity.seal_idle_control(),
            "Duplicate shutdown cannot be accepted"
        );
        assert!(
            !activity.is_idle_for_control(),
            "Sealed requests cannot be replayed"
        );
        drop(RemoteHandlerActivity(activity.clone()));
        assert_eq!(activity.handlers.load(SeqCst), 0);
    }

    #[test]
    fn doctor_restart_protocol_and_safe_mode_preserve_identity() {
        let mut status = proto::DoctorRemoteServerResponse {
            protocol_version: 1,
            instance: uuid::Uuid::new_v4().to_string(),
            process_id: 17,
            exclusive: true,
            idle: true,
            ..Default::default()
        };
        assert!(validate_doctor_server_status(&status).is_ok());
        assert_eq!(
            ProxyMode::StartIfStopped(status.instance.clone()).arguments(),
            vec!["--start-if-stopped".to_string(), status.instance.clone()]
        );
        status.protocol_version = 2;
        assert!(validate_doctor_server_status(&status).is_err());
        status.protocol_version = 1;
        status.process_id = 0;
        assert!(validate_doctor_server_status(&status).is_err());
        status.process_id = 17;
        status.instance = "../outside".into();
        assert!(validate_doctor_server_status(&status).is_err());
        status.instance = uuid::Uuid::new_v4().to_string();
        status.reason = "x".repeat(1025);
        assert!(validate_doctor_server_status(&status).is_err());
    }

    struct DoctorControlFixture {
        activity: RemoteRequestActivity,
        pending: Option<oneshot::Receiver<()>>,
        observed: usize,
    }

    impl DoctorControlFixture {
        async fn test_request(
            this: Entity<Self>,
            _: rpc::TypedEnvelope<proto::Test>,
            mut cx: AsyncApp,
        ) -> Result<proto::Test> {
            let pending = this
                .update(&mut cx, |this, _| this.pending.take())
                .context("Fixture request gate")?;
            pending.await.context("Fixture request cancelled")?;
            Ok(proto::Test { id: 1 })
        }

        async fn control(
            this: Entity<Self>,
            request: rpc::TypedEnvelope<proto::DoctorRemoteServer>,
            mut cx: AsyncApp,
        ) -> Result<proto::DoctorRemoteServerResponse> {
            Ok(this.update(&mut cx, |this, _| {
                this.observed += 1;
                let idle = this.activity.is_idle_for_control();
                let accepted = request.payload.shutdown && this.activity.seal_idle_control();
                proto::DoctorRemoteServerResponse {
                    protocol_version: 1,
                    instance: "00000000-0000-4000-8000-000000000001".into(),
                    process_id: 17,
                    exclusive: true,
                    idle,
                    shutdown_accepted: accepted,
                    ..Default::default()
                }
            }))
        }
    }

    #[gpui::test]
    async fn doctor_restart_channel_rejects_active_requests_and_seals_queued_requests(
        cx: &mut TestAppContext,
    ) {
        let (client_outgoing, server_incoming) = mpsc::unbounded();
        let (server_outgoing, client_incoming) = mpsc::unbounded();
        let (server, activity) = cx.update(|cx| {
            RemoteClient::proto_client_from_channels_with_activity(
                server_incoming,
                server_outgoing,
                cx,
                false,
            )
        });
        let client = cx.update(|cx| {
            RemoteClient::proto_client_from_channels(
                client_incoming,
                client_outgoing,
                cx,
                "test-client",
                false,
            )
        });
        let (release, pending) = oneshot::channel();
        let fixture = cx.new(|_| DoctorControlFixture {
            activity: activity.clone(),
            pending: Some(pending),
            observed: 0,
        });
        server.add_request_handler(fixture.downgrade(), DoctorControlFixture::test_request);
        server.add_request_handler(fixture.downgrade(), DoctorControlFixture::control);
        cx.run_until_parked();
        let pending = cx.executor().spawn(client.request(proto::Test { id: 1 }));
        cx.run_until_parked();
        assert_eq!(
            activity.handlers.load(SeqCst),
            1,
            "Existing request remains active"
        );
        let denied = client
            .request(proto::DoctorRemoteServer {
                expected_instance: String::new(),
                shutdown: true,
            })
            .await
            .expect("Observed busy control");
        assert!(!denied.idle && !denied.shutdown_accepted);
        release.send(()).expect("Release existing host request");
        pending.await.expect("Existing request completed");
        cx.run_until_parked();
        assert_eq!(activity.handlers.load(SeqCst), 0);
        let accepted = client
            .request(proto::DoctorRemoteServer {
                expected_instance: String::new(),
                shutdown: true,
            })
            .await
            .expect("Idle gate");
        assert!(accepted.idle && accepted.shutdown_accepted);
        let duplicate = cx
            .executor()
            .spawn(client.request(proto::DoctorRemoteServer {
                expected_instance: String::new(),
                shutdown: true,
            }));
        cx.run_until_parked();
        assert_eq!(
            fixture.read_with(cx, |fixture, _| fixture.observed),
            2,
            "Sealed channel never dispatches a duplicate or replayed operation"
        );
        drop(duplicate);
    }

    #[gpui::test]
    async fn doctor_restart_lost_acknowledgement_and_cancellation_never_reconnect(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| release_channel::init(Version::new(1, 16, 2), cx));
        for cancel in [false, true] {
            let (options, server, guard) = RemoteClient::fake_server(cx, server_cx);
            let owner = server_cx.new(|_| ());
            server
                .add_request_handler::<proto::Ping, _, _, _>(owner.downgrade(), |_, _, _| async {
                    Ok(proto::Ack {})
                });
            let (started, mut observed) = mpsc::unbounded::<()>();
            let (finish, pending) = oneshot::channel::<()>();
            let pending = Arc::new(Mutex::new(Some(pending)));
            let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            server.add_request_handler::<proto::DoctorRemoteServer, _, _, _>(owner.downgrade(), {
                let requests = requests.clone();
                move |_, envelope, _| {
                    assert!(envelope.payload.shutdown);
                    requests.fetch_add(1, SeqCst);
                    let started = started.clone();
                    let pending = pending
                        .lock()
                        .take()
                        .expect("Exactly one shutdown dispatch");
                    async move {
                        started.unbounded_send(())?;
                        pending.await?;
                        Ok(proto::DoctorRemoteServerResponse::default())
                    }
                }
            });
            drop(guard);
            let client = RemoteClient::connect_mock(options, cx).await;
            cx.run_until_parked();
            let status = proto::DoctorRemoteServerResponse {
                protocol_version: 1,
                instance: uuid::Uuid::new_v4().to_string(),
                process_id: 17,
                exclusive: true,
                idle: true,
                ..Default::default()
            };
            let (cancellation, cancelled) = oneshot::channel::<()>();
            let operation = client.update(cx, |client, cx| {
                client.doctor_restart_observed(
                    status.clone(),
                    Task::ready(Ok(status)),
                    async move {
                        match cancelled.await {
                            Ok(()) | Err(_) => {}
                        }
                    }
                    .boxed(),
                    |_, _| Ok(()),
                    cx,
                )
            });
            observed.next().await.expect("Actual shutdown dispatch");
            assert_eq!(
                client.read_with(cx, |client, _| client.connection_state()),
                ConnectionState::Disconnected
            );
            if cancel {
                cancellation.send(()).expect("Explicit cancellation");
            } else {
                cx.executor().advance_clock(Duration::from_secs(6));
            }
            let result = operation.await.expect("Bounded uncertain lifecycle");
            assert_eq!(
                result.outcome,
                if cancel {
                    DoctorRemoteRestartOutcome::Cancelled
                } else {
                    DoctorRemoteRestartOutcome::Incomplete
                }
            );
            assert!(result.shutdown_requested && result.replacement.is_none());
            drop(finish);
            cx.run_until_parked();
            assert_eq!(requests.load(SeqCst), 1);
            assert_eq!(
                client.read_with(cx, |client, _| client.connection_state()),
                ConnectionState::Disconnected
            );
        }
    }

    #[test]
    fn test_initial_connection_timeout_covers_remote_server_startup() {
        assert!(
            INITIAL_CONNECTION_TIMEOUT > Duration::from_secs(10),
            "the remote server may take ten seconds to publish its sockets"
        );
    }

    #[test]
    fn test_ssh_display_name_prefers_nickname() {
        let options = RemoteConnectionOptions::Ssh(SshConnectionOptions {
            host: "1.2.3.4".into(),
            nickname: Some("My Cool Project".to_string()),
            ..Default::default()
        });

        assert_eq!(options.display_name(), "My Cool Project");
    }

    #[test]
    fn test_ssh_display_name_falls_back_to_host() {
        let options = RemoteConnectionOptions::Ssh(SshConnectionOptions {
            host: "1.2.3.4".into(),
            ..Default::default()
        });

        assert_eq!(options.display_name(), "1.2.3.4");
    }

    #[test]
    fn test_connection_type() {
        assert_eq!(
            RemoteConnectionOptions::Ssh(SshConnectionOptions::default()).connection_type(),
            "ssh"
        );
        assert_eq!(
            RemoteConnectionOptions::Wsl(WslConnectionOptions {
                distro_name: "Ubuntu".to_string(),
                user: None,
            })
            .connection_type(),
            "wsl"
        );
        assert_eq!(
            RemoteConnectionOptions::Docker(DockerConnectionOptions {
                use_podman: false,
                ..Default::default()
            })
            .connection_type(),
            "docker"
        );
        assert_eq!(
            RemoteConnectionOptions::Docker(DockerConnectionOptions {
                use_podman: true,
                ..Default::default()
            })
            .connection_type(),
            "podman"
        );
    }

    #[gpui::test]
    async fn test_direct_client_reconnect_initializes_connection_pool(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use crate::transport::mock::{MockConnectionRegistry, MockDelegate};

        cx.update(|cx| release_channel::init(Version::new(1, 16, 2), cx));
        let (options, original_server, guard) = RemoteClient::fake_server(cx, server_cx);
        let server = server_cx.new(|_| ());
        original_server
            .add_request_handler::<proto::Ping, _, _, _>(server.downgrade(), |_, _, _| async {
                Ok(proto::Ack {})
            });
        drop(guard);
        let connection = cx
            .update(|cx| {
                let RemoteConnectionOptions::Mock(options) = &options else {
                    unreachable!("fake server must use mock options");
                };
                cx.global_mut::<MockConnectionRegistry>()
                    .take(options)
                    .expect("registered original connection")
            })
            .await;
        let (_cancel, cancellation) = oneshot::channel();
        let remote = cx
            .update(|cx| {
                RemoteClient::new(
                    ConnectionIdentifier::setup(),
                    connection,
                    cancellation,
                    Arc::new(MockDelegate),
                    cx,
                )
            })
            .await
            .expect("connect direct client")
            .expect("direct client was not cancelled");
        assert!(cx.read(|cx| cx.try_global::<ConnectionPool>().is_none()));

        let (replacement_server, guard) =
            RemoteClient::fake_server_with_opts(&options, cx, server_cx);
        replacement_server
            .add_request_handler::<proto::Ping, _, _, _>(server.downgrade(), |_, _, _| async {
                Ok(proto::Ack {})
            });
        drop(guard);
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(0, cx));
        cx.run_until_parked();
        assert_eq!(
            remote.read_with(cx, |remote, _| remote.connection_state()),
            ConnectionState::Connected
        );
        assert!(cx.read(|cx| has_active_connection(&options, cx)));
        let shutdown = remote.update(cx, |remote, cx| {
            remote.shutdown_processes::<proto::ShutdownRemoteServer>(
                None,
                cx.background_executor().clone(),
            )
        });
        if let Some(shutdown) = shutdown {
            shutdown.await;
        }
    }

    fn record_consumer_scopes(
        client: &AnyProtoClient,
        observed: &Entity<Vec<(PeerId, Option<PeerId>, u64)>>,
    ) {
        client.add_request_handler(
            observed.downgrade(),
            |observed, envelope: proto::TypedEnvelope<proto::SetJvmProjectConsumers>, mut cx| async move {
                let request = envelope.payload;
                observed.update(&mut cx, |observed, _| {
                    observed.push((envelope.sender_id, envelope.original_sender_id, request.generation));
                });
                Ok(proto::JvmProjectConsumersResponse {
                    protocol_version: request.protocol_version,
                    generation: request.generation,
                    accepted_roots: request.roots,
                })
            },
        );
    }

    #[gpui::test]
    async fn scoped_mock_connection_preserves_receiving_and_relay_identity(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| release_channel::init(Version::new(1, 16, 2), cx));
        for peer in [
            None,
            Some(PeerId {
                owner_id: 0,
                id: 17,
            }),
        ] {
            let (options, server, guard, activity) = if let Some(peer) = peer {
                RemoteClient::fake_server_scoped(peer, cx, server_cx).expect("scoped mock")
            } else {
                RemoteClient::fake_server_with_activity(cx, server_cx)
            };
            let observed = server_cx.new(|_| Vec::new());
            record_consumer_scopes(&server, &observed);
            server.add_request_handler::<proto::Ping, _, _, _>(
                observed.downgrade(),
                |_, _, _| async { Ok(proto::Ack {}) },
            );
            drop(guard);
            let remote = RemoteClient::connect_mock(options, cx).await;
            let client = remote.read_with(cx, |remote, _| remote.proto_client());
            let request = |generation| proto::SetJvmProjectConsumers {
                project_id: 7,
                protocol_version: 1,
                generation,
                roots: Vec::new(),
            };
            assert_eq!(
                client
                    .request(request(1))
                    .await
                    .expect("direct response")
                    .generation,
                1
            );
            let relay = PeerId {
                owner_id: 999,
                id: 999,
            };
            assert_eq!(
                client
                    .forward_request(relay, request(2))
                    .await
                    .expect("forwarded response")
                    .generation,
                2
            );
            server_cx.run_until_parked();
            assert_eq!(
                observed.read_with(server_cx, |observed, _| observed.clone()),
                vec![
                    (peer.unwrap_or_default(), None, 1),
                    (peer.unwrap_or_default(), Some(relay), 2)
                ]
            );
            assert_eq!(activity.handlers.load(SeqCst), 0);
            let shutdown = remote.update(cx, |remote, cx| {
                remote.shutdown_processes::<proto::ShutdownRemoteServer>(
                    None,
                    cx.background_executor().clone(),
                )
            });
            if let Some(shutdown) = shutdown {
                shutdown.await;
            }
        }
    }

    #[gpui::test]
    async fn scoped_mock_connection_retains_held_request_scope(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| release_channel::init(Version::new(1, 16, 2), cx));
        let old_peer = PeerId {
            owner_id: 0,
            id: 17,
        };
        let current_peer = PeerId {
            owner_id: 0,
            id: 18,
        };
        let (options, server, guard, old_activity) =
            RemoteClient::fake_server_scoped(old_peer, cx, server_cx).expect("old scoped mock");
        let observed = server_cx.new(|_| Vec::new());
        let (release, waiting) = oneshot::channel();
        let waiting = Arc::new(Mutex::new(Some(waiting)));
        server.add_request_handler(
            observed.downgrade(),
            move |observed,
                  envelope: proto::TypedEnvelope<proto::SetJvmProjectConsumers>,
                  mut cx| {
                let waiting = waiting.lock().take();
                async move {
                    waiting
                        .context("held scope gate")?
                        .await
                        .context("release held scope")?;
                    let request = envelope.payload;
                    observed.update(&mut cx, |observed, _| {
                        observed.push((
                            envelope.sender_id,
                            envelope.original_sender_id,
                            request.generation,
                        ))
                    });
                    Ok(proto::JvmProjectConsumersResponse {
                        protocol_version: 1,
                        generation: request.generation,
                        accepted_roots: request.roots,
                    })
                }
            },
        );
        server.add_request_handler::<proto::Ping, _, _, _>(observed.downgrade(), |_, _, _| async {
            Ok(proto::Ack {})
        });
        drop(guard);
        let old_remote = RemoteClient::connect_mock(options, cx).await;
        let old_client = old_remote.read_with(cx, |remote, _| remote.proto_client());
        let request = |generation| proto::SetJvmProjectConsumers {
            project_id: 7,
            protocol_version: 1,
            generation,
            roots: Vec::new(),
        };
        let held = old_client.request(request(1));
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert_eq!(old_activity.handlers.load(SeqCst), 1);
        assert!(observed.read_with(server_cx, |observed, _| observed.is_empty()));

        let (options, current_server, guard, _) =
            RemoteClient::fake_server_scoped(current_peer, cx, server_cx)
                .expect("current scoped mock");
        record_consumer_scopes(&current_server, &observed);
        current_server
            .add_request_handler::<proto::Ping, _, _, _>(observed.downgrade(), |_, _, _| async {
                Ok(proto::Ack {})
            });
        drop(guard);
        let current_remote = RemoteClient::connect_mock(options, cx).await;
        let current_client = current_remote.read_with(cx, |remote, _| remote.proto_client());
        current_client
            .request(request(2))
            .await
            .expect("current scope response");
        assert_eq!(
            observed.read_with(server_cx, |observed, _| observed.clone()),
            vec![(current_peer, None, 2)]
        );
        release.send(()).expect("release old scope");
        assert_eq!(held.await.expect("old response routing").generation, 1);
        assert_eq!(
            observed.read_with(server_cx, |observed, _| observed.clone()),
            vec![(current_peer, None, 2), (old_peer, None, 1)]
        );
        assert_eq!(old_activity.handlers.load(SeqCst), 0);
        for remote in [old_remote, current_remote] {
            let shutdown = remote.update(cx, |remote, cx| {
                remote.shutdown_processes::<proto::ShutdownRemoteServer>(
                    None,
                    cx.background_executor().clone(),
                )
            });
            if let Some(shutdown) = shutdown {
                shutdown.await;
            }
        }
    }

    #[gpui::test]
    async fn scoped_mock_connection_retains_doctor_activity_and_sealing(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| release_channel::init(Version::new(1, 16, 2), cx));
        let (options, server, guard, activity) = RemoteClient::fake_server_scoped(
            PeerId {
                owner_id: 0,
                id: 17,
            },
            cx,
            server_cx,
        )
        .expect("scoped mock");
        let (release, waiting) = oneshot::channel();
        let fixture = server_cx.new(|_| DoctorControlFixture {
            activity: activity.clone(),
            pending: Some(waiting),
            observed: 0,
        });
        server.add_request_handler::<proto::Ping, _, _, _>(fixture.downgrade(), |_, _, _| async {
            Ok(proto::Ack {})
        });
        server.add_request_handler(fixture.downgrade(), DoctorControlFixture::test_request);
        server.add_request_handler(fixture.downgrade(), DoctorControlFixture::control);
        let observed = server_cx.new(|_| Vec::new());
        record_consumer_scopes(&server, &observed);
        drop(guard);
        let remote = RemoteClient::connect_mock(options, cx).await;
        let client = remote.read_with(cx, |remote, _| remote.proto_client());
        let held = client.request(proto::Test { id: 1 });
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert_eq!(activity.handlers.load(SeqCst), 1);
        let control = proto::DoctorRemoteServer {
            expected_instance: String::new(),
            shutdown: true,
        };
        let busy = client.request(control.clone()).await.expect("busy control");
        assert!(!busy.idle && !busy.shutdown_accepted);
        release.send(()).expect("release regular request");
        held.await.expect("held response");
        let sealed = client.request(control).await.expect("idle sealing control");
        assert!(sealed.idle && sealed.shutdown_accepted);
        client
            .send(proto::SetJvmProjectConsumers {
                project_id: 7,
                protocol_version: 1,
                generation: 1,
                roots: Vec::new(),
            })
            .expect("send after seal");
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert!(observed.read_with(server_cx, |observed, _| observed.is_empty()));
        assert_eq!(
            fixture.read_with(server_cx, |fixture, _| fixture.observed),
            2
        );
        let shutdown = remote.update(cx, |remote, cx| {
            remote.shutdown_processes::<proto::ShutdownRemoteServer>(
                None,
                cx.background_executor().clone(),
            )
        });
        if let Some(shutdown) = shutdown {
            shutdown.await;
        }
    }

    #[gpui::test]
    async fn scoped_mock_connection_rejects_zero_before_registration(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use crate::transport::mock::MockConnectionRegistry;
        assert!(cx.read(|cx| cx.try_global::<MockConnectionRegistry>().is_none()));
        for owner_id in [0, 999] {
            let error =
                match RemoteClient::fake_server_scoped(PeerId { owner_id, id: 0 }, cx, server_cx) {
                    Ok(_) => panic!("zero receiving scope must refuse"),
                    Err(error) => error,
                };
            assert!(error.to_string().contains("nonzero receiving peer"));
        }
        assert!(cx.read(|cx| cx.try_global::<MockConnectionRegistry>().is_none()));
    }

    #[gpui::test]
    async fn test_channel_client_retains_queued_receiving_scope(cx: &mut TestAppContext) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<(PeerId, Envelope)>();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();
        let client = cx.update(|cx| {
            RemoteClient::proto_client_from_scoped_channels(
                incoming_rx,
                outgoing_tx,
                cx,
                "test-server",
                false,
            )
        });
        let observed = cx.new(|_| Vec::new());
        record_consumer_scopes(&client, &observed);
        let old_scope = PeerId {
            owner_id: 0,
            id: 17,
        };
        let current_scope = PeerId {
            owner_id: 0,
            id: 18,
        };
        let supplied_relay = PeerId {
            owner_id: 999,
            id: 999,
        };
        for (scope, generation, message_id, relay) in [
            (old_scope, 1, 41, Some(supplied_relay)),
            (current_scope, 1, 42, None),
        ] {
            incoming_tx
                .unbounded_send((
                    scope,
                    proto::SetJvmProjectConsumers {
                        project_id: 7,
                        protocol_version: 1,
                        generation,
                        roots: Vec::new(),
                    }
                    .into_envelope(message_id, None, relay),
                ))
                .expect("queue scoped request");
        }
        cx.run_until_parked();
        observed.read_with(cx, |observed, _| {
            assert_eq!(
                observed,
                &vec![
                    (old_scope, Some(supplied_relay), 1),
                    (current_scope, None, 1)
                ]
            );
        });
        let mut responses = Vec::new();
        while let Some(Some(envelope)) = outgoing_rx.next().now_or_never() {
            if let Some(proto::envelope::Payload::JvmProjectConsumersResponse(response)) =
                envelope.payload
            {
                responses.push((envelope.responding_to, response.generation));
            }
        }
        responses.sort();
        assert_eq!(responses, vec![(Some(41), 1), (Some(42), 1)]);
    }

    #[gpui::test]
    async fn test_channel_client_held_request_keeps_receiving_scope(cx: &mut TestAppContext) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<(PeerId, Envelope)>();
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded::<Envelope>();
        let client = cx.update(|cx| {
            RemoteClient::proto_client_from_scoped_channels(
                incoming_rx,
                outgoing_tx,
                cx,
                "test-server",
                false,
            )
        });
        let observed = cx.new(|_| Vec::<(PeerId, u64)>::new());
        let (release, waiting) = oneshot::channel::<()>();
        let waiting = Arc::new(Mutex::new(Some(waiting)));
        client.add_request_handler(
            observed.downgrade(),
            move |observed,
                  envelope: proto::TypedEnvelope<proto::SetJvmProjectConsumers>,
                  mut cx| {
                let waiting = if envelope.payload.generation == 1 {
                    waiting.lock().take()
                } else {
                    None
                };
                async move {
                    if let Some(waiting) = waiting {
                        waiting.await.context("scoped fixture gate closed")?;
                    }
                    let request = envelope.payload;
                    observed.update(&mut cx, |observed, _| {
                        observed.push((envelope.sender_id, request.generation));
                    });
                    Ok(proto::JvmProjectConsumersResponse {
                        protocol_version: request.protocol_version,
                        generation: request.generation,
                        accepted_roots: request.roots,
                    })
                }
            },
        );
        let old_scope = PeerId {
            owner_id: 0,
            id: 17,
        };
        let new_scope = PeerId {
            owner_id: 0,
            id: 18,
        };
        for (scope, generation) in [(old_scope, 1), (new_scope, 2)] {
            incoming_tx
                .unbounded_send((
                    scope,
                    proto::SetJvmProjectConsumers {
                        project_id: 7,
                        protocol_version: 1,
                        generation,
                        roots: Vec::new(),
                    }
                    .into_envelope(generation as u32, None, None),
                ))
                .expect("queue held scoped request");
            cx.run_until_parked();
        }
        observed.read_with(cx, |observed, _| {
            assert_eq!(observed, &vec![(new_scope, 2)])
        });
        release.send(()).expect("release original request");
        cx.run_until_parked();
        observed.read_with(cx, |observed, _| {
            assert_eq!(observed, &vec![(new_scope, 2), (old_scope, 1)])
        });
    }

    #[gpui::test]
    async fn test_scoped_activity_factory_preserves_receiving_scope(cx: &mut TestAppContext) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<(PeerId, Envelope)>();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();
        let (client, _activity) = cx.update(|cx| {
            RemoteClient::proto_client_from_scoped_channels_with_activity(
                incoming_rx,
                outgoing_tx,
                cx,
                false,
            )
        });
        let observed = cx.new(|_| Vec::new());
        record_consumer_scopes(&client, &observed);
        let old_scope = PeerId {
            owner_id: 0,
            id: 17,
        };
        let current_scope = PeerId {
            owner_id: 0,
            id: 18,
        };
        let supplied_relay = PeerId {
            owner_id: 999,
            id: 999,
        };
        for (scope, message_id, relay) in [
            (old_scope, 41, Some(supplied_relay)),
            (current_scope, 42, None),
        ] {
            incoming_tx
                .unbounded_send((
                    scope,
                    proto::SetJvmProjectConsumers {
                        project_id: 7,
                        protocol_version: 1,
                        generation: 1,
                        roots: Vec::new(),
                    }
                    .into_envelope(message_id, None, relay),
                ))
                .expect("Queue scoped request with returned activity");
        }
        cx.run_until_parked();
        observed.read_with(cx, |observed, _| {
            assert_eq!(
                observed,
                &vec![
                    (old_scope, Some(supplied_relay), 1),
                    (current_scope, None, 1)
                ]
            );
        });
        let mut responses = Vec::new();
        while let Some(Some(envelope)) = outgoing_rx.next().now_or_never() {
            if let Some(proto::envelope::Payload::JvmProjectConsumersResponse(response)) =
                envelope.payload
            {
                responses.push((envelope.responding_to, response.generation));
            }
        }
        responses.sort();
        assert_eq!(responses, vec![(Some(41), 1), (Some(42), 1)]);
    }

    #[gpui::test]
    async fn test_scoped_activity_factory_retains_doctor_sealing(cx: &mut TestAppContext) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<(PeerId, Envelope)>();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();
        let (server, activity) = cx.update(|cx| {
            RemoteClient::proto_client_from_scoped_channels_with_activity(
                incoming_rx,
                outgoing_tx,
                cx,
                false,
            )
        });
        let (release, pending) = oneshot::channel();
        let fixture = cx.new(|_| DoctorControlFixture {
            activity: activity.clone(),
            pending: Some(pending),
            observed: 0,
        });
        server.add_request_handler(fixture.downgrade(), DoctorControlFixture::test_request);
        server.add_request_handler(fixture.downgrade(), DoctorControlFixture::control);
        let scope = PeerId {
            owner_id: 0,
            id: 17,
        };
        incoming_tx
            .unbounded_send((scope, proto::Test { id: 1 }.into_envelope(41, None, None)))
            .expect("Hold existing scoped request");
        cx.run_until_parked();
        let control = proto::DoctorRemoteServer {
            expected_instance: String::new(),
            shutdown: true,
        };
        incoming_tx
            .unbounded_send((scope, control.clone().into_envelope(42, None, None)))
            .expect("Observe busy scoped control");
        cx.run_until_parked();
        release.send(()).expect("Complete existing scoped request");
        cx.run_until_parked();
        incoming_tx
            .unbounded_send((scope, control.clone().into_envelope(43, None, None)))
            .expect("Seal idle scoped control");
        cx.run_until_parked();
        incoming_tx
            .unbounded_send((scope, proto::Test { id: 2 }.into_envelope(44, None, None)))
            .expect("Queue request after scoped sealing");
        incoming_tx
            .unbounded_send((scope, control.into_envelope(45, None, None)))
            .expect("Queue duplicate scoped control");
        cx.run_until_parked();
        let mut responses = Vec::new();
        let mut controls = Vec::new();
        while let Some(Some(envelope)) = outgoing_rx.next().now_or_never() {
            if let Some(message_id) = envelope.responding_to {
                responses.push(message_id);
                if let Some(proto::envelope::Payload::DoctorRemoteServerResponse(response)) =
                    envelope.payload
                {
                    controls.push((message_id, response.idle, response.shutdown_accepted));
                }
            }
        }
        responses.sort();
        controls.sort();
        assert_eq!(responses, vec![41, 42, 43]);
        assert_eq!(controls, vec![(42, false, false), (43, true, true)]);
        fixture.read_with(cx, |fixture, _| assert_eq!(fixture.observed, 2));
        assert!(!activity.is_idle_for_control());
    }

    #[gpui::test]
    async fn test_channel_client_unscoped_factory_preserves_peer_zero(cx: &mut TestAppContext) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<Envelope>();
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded::<Envelope>();
        let client = cx.update(|cx| {
            RemoteClient::proto_client_from_channels(
                incoming_rx,
                outgoing_tx,
                cx,
                "test-server",
                false,
            )
        });
        let observed = cx.new(|_| Vec::new());
        record_consumer_scopes(&client, &observed);
        incoming_tx
            .unbounded_send(
                proto::SetJvmProjectConsumers {
                    project_id: 7,
                    protocol_version: 1,
                    generation: 1,
                    roots: Vec::new(),
                }
                .into_envelope(41, None, None),
            )
            .expect("queue unscoped request");
        cx.run_until_parked();
        observed.read_with(cx, |observed, _| {
            assert_eq!(observed, &vec![(PeerId { owner_id: 0, id: 0 }, None, 1)]);
        });
    }

    fn jvm_intent_for_channel_test(
        generation: u64,
        include_root: bool,
    ) -> proto::SetJvmProjectConsumers {
        proto::SetJvmProjectConsumers {
            project_id: 7,
            protocol_version: 1,
            generation,
            roots: if include_root {
                vec![proto::ProjectPath {
                    worktree_id: 17,
                    path: "private-request-canary".into(),
                }]
            } else {
                Vec::new()
            },
        }
    }

    #[gpui::test]
    fn test_channel_client_jvm_intent_cancellation_releases_bookkeeping(cx: &mut TestAppContext) {
        for include_root in [false, true] {
            for poll_request in [false, true] {
                let (_incoming_tx, incoming_rx) = mpsc::unbounded();
                let (outgoing_tx, _outgoing_rx) = mpsc::unbounded();
                let client = cx.update(|cx| {
                    ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-jvm", false)
                });
                let mut pending =
                    Box::pin(client.request(jvm_intent_for_channel_test(1, include_root)));
                if poll_request {
                    assert!(pending.as_mut().now_or_never().is_none());
                }
                assert_eq!(client.response_channels.lock().len(), 1);
                assert!(client.buffer.lock().is_empty());
                drop(pending);
                cx.run_until_parked();
                assert!(client.response_channels.lock().is_empty());
                assert!(client.buffer.lock().is_empty());
            }
        }
    }

    #[gpui::test]
    async fn test_channel_client_jvm_intent_does_not_replay_after_reconnect(
        cx: &mut TestAppContext,
    ) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded();
        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-jvm", false));
        let requests: AnyProtoClient = client.clone().into();
        cx.run_until_parked();
        while outgoing_rx.next().now_or_never().flatten().is_some() {}
        let grant = requests.request(jvm_intent_for_channel_test(1, true));
        let withdrawal = requests.request(jvm_intent_for_channel_test(2, false));
        let ordinary = client.request(proto::Test { id: 17 });
        let mut original = Vec::new();
        while let Some(Some(envelope)) = outgoing_rx.next().now_or_never() {
            original.push(envelope);
        }
        let old_intents = original
            .iter()
            .filter_map(|envelope| {
                matches!(
                    envelope.payload.as_ref(),
                    Some(proto::envelope::Payload::SetJvmProjectConsumers(_))
                )
                .then_some(envelope.id)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            old_intents.len(),
            2,
            "both grant and withdrawal were sent on the original connection"
        );
        let ordinary_id = original
            .iter()
            .find(|envelope| {
                matches!(
                    envelope.payload.as_ref(),
                    Some(proto::envelope::Payload::Test(_))
                )
            })
            .expect("original ordinary request")
            .id;
        assert_eq!(client.buffer.lock().len(), 1);
        assert_eq!(client.response_channels.lock().len(), 3);

        let (replacement_input, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, mut replacement_output) = mpsc::unbounded();
        client.reconnect(incoming_rx, outgoing_tx, &cx.to_async());
        let mut resync = Box::pin(client.resync(Duration::from_secs(5)));
        assert!(resync.as_mut().now_or_never().is_none());
        cx.run_until_parked();
        let mut startup = Vec::new();
        while let Some(Some(envelope)) = replacement_output.next().now_or_never() {
            startup.push(envelope);
        }
        let flush = startup
            .iter()
            .find(|envelope| {
                matches!(
                    envelope.payload.as_ref(),
                    Some(proto::envelope::Payload::FlushBufferedMessages(_))
                )
            })
            .expect("existing resync flush");
        replacement_input
            .unbounded_send(proto::Ack {}.into_envelope(100, Some(flush.id), None))
            .expect("flush reply");
        resync.await.expect("resync current transport");
        let mut replayed = Vec::new();
        while let Some(Some(envelope)) = replacement_output.next().now_or_never() {
            replayed.push(envelope);
        }
        assert!(
            replayed.iter().all(|envelope| !matches!(
                envelope.payload.as_ref(),
                Some(proto::envelope::Payload::SetJvmProjectConsumers(_))
            )),
            "startup authority cannot cross into the replacement connection by replay"
        );
        assert_eq!(
            replayed
                .iter()
                .filter(|envelope| matches!(
                    envelope.payload.as_ref(),
                    Some(proto::envelope::Payload::Test(_))
                ))
                .map(|envelope| envelope.id)
                .collect::<Vec<_>>(),
            [ordinary_id]
        );

        drop(grant);
        drop(withdrawal);
        assert_eq!(client.response_channels.lock().len(), 1);
        for message_id in old_intents {
            replacement_input
                .unbounded_send(
                    proto::JvmProjectConsumersResponse {
                        protocol_version: 1,
                        generation: 1,
                        accepted_roots: Vec::new(),
                    }
                    .into_envelope(101 + message_id, Some(message_id), None),
                )
                .expect("late old reply");
        }
        let fresh = requests.request(jvm_intent_for_channel_test(3, true));
        let fresh_message = replacement_output
            .next()
            .now_or_never()
            .flatten()
            .expect("fresh explicit intent");
        assert!(
            matches!(fresh_message.payload.as_ref(), Some(proto::envelope::Payload::SetJvmProjectConsumers(request)) if request.generation==3)
        );
        replacement_input
            .unbounded_send(
                proto::JvmProjectConsumersResponse {
                    protocol_version: 1,
                    generation: 3,
                    accepted_roots: jvm_intent_for_channel_test(3, true).roots,
                }
                .into_envelope(110, Some(fresh_message.id), None),
            )
            .expect("fresh reply");
        let response = fresh
            .await
            .expect("fresh intent routes through the existing response owner");
        assert_eq!(response.generation, 3);
        assert_eq!(
            response.accepted_roots,
            jvm_intent_for_channel_test(3, true).roots
        );
        let mut response = proto::Test { id: 19 }.into_envelope(111, Some(ordinary_id), None);
        response.ack_id = Some(ordinary_id);
        replacement_input
            .unbounded_send(response)
            .expect("ordinary reply");
        assert_eq!(
            ordinary
                .await
                .expect("ordinary Copper routing survives resync")
                .id,
            19
        );
        cx.run_until_parked();
        assert!(client.response_channels.lock().is_empty());
        assert!(client.buffer.lock().is_empty());
    }

    #[gpui::test]
    async fn test_channel_client_jvm_intent_closed_transport_is_actionable(
        cx: &mut TestAppContext,
    ) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, outgoing_rx) = mpsc::unbounded();
        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-jvm", false));
        cx.run_until_parked();
        drop(outgoing_rx);
        let requests: AnyProtoClient = client.clone().into();
        let error = requests
            .request(jvm_intent_for_channel_test(1, true))
            .await
            .expect_err("closed current transport");
        let message = format!("{error:#}");
        assert!(message.contains("reconnect") && message.contains("retry"));
        assert!(!message.contains("private-request-canary"));
        assert!(message.len() < 512);
        assert!(client.response_channels.lock().is_empty());
        assert!(client.buffer.lock().is_empty());
    }

    #[gpui::test]
    async fn test_channel_client_jvm_intent_one_way_send_is_connection_scoped(
        cx: &mut TestAppContext,
    ) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded();
        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-jvm", false));
        let messages: AnyProtoClient = client.clone().into();
        cx.run_until_parked();
        while outgoing_rx.next().now_or_never().flatten().is_some() {}
        for include_root in [true, false] {
            messages
                .send(jvm_intent_for_channel_test(1, include_root))
                .expect("send current intent without replay");
            let sent = outgoing_rx
                .next()
                .now_or_never()
                .flatten()
                .expect("current connection receives intent");
            assert!(matches!(
                sent.payload.as_ref(),
                Some(proto::envelope::Payload::SetJvmProjectConsumers(request))
                    if request.roots.is_empty() != include_root
            ));
            assert!(client.buffer.lock().is_empty());
        }
        drop(outgoing_rx);
        let error = messages
            .send(jvm_intent_for_channel_test(2, true))
            .expect_err("closed transport refuses startup intent");
        let message = format!("{error:#}");
        assert!(message.contains("reconnect") && message.contains("retry"));
        assert!(!message.contains("private-request-canary"));
        assert!(message.len() < 512);
        assert!(client.buffer.lock().is_empty());
        client
            .send(proto::Test { id: 19 })
            .expect("ordinary sends retain existing reconnect behavior");
        assert_eq!(client.buffer.lock().len(), 1);
        assert!(client.response_channels.lock().is_empty());
    }

    #[gpui::test]
    async fn test_channel_client_jvm_intent_does_not_retain_channel_owner(cx: &mut TestAppContext) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded();
        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-jvm", false));
        cx.run_until_parked();
        let owner = Arc::downgrade(&client);
        let responses = Arc::downgrade(&client.response_channels);
        let pending = client.request(jvm_intent_for_channel_test(1, true));
        assert_eq!(client.response_channels.lock().len(), 1);
        drop(client);
        cx.run_until_parked();
        assert!(owner.upgrade().is_none());
        assert!(
            responses.upgrade().is_none(),
            "pending intent cannot retain the response map"
        );
        let deadline = cx.background_executor.timer(Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(Box::pin(pending), Box::pin(deadline)).await,
            futures::future::Either::Left((Err(_), _))
        ));
    }

    fn project_restart_for_channel_test(all: bool) -> proto::RestartLanguageServers {
        proto::RestartLanguageServers {
            project_id: 7,
            all,
            buffer_ids: if all { Vec::new() } else { vec![17] },
            only_servers: vec![proto::LanguageServerSelector {
                selector: Some(proto::language_server_selector::Selector::Name(
                    "private-restart-canary".into(),
                )),
            }],
        }
    }

    #[gpui::test]
    async fn test_channel_client_project_restart_requires_current_connection(
        cx: &mut TestAppContext,
    ) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded();
        let client = cx.update(|cx| {
            ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-restart", false)
        });
        let requests: AnyProtoClient = client.clone().into();
        cx.run_until_parked();
        while outgoing_rx.next().now_or_never().flatten().is_some() {}
        let all = requests.request(project_restart_for_channel_test(true));
        let ordinary = requests.request(project_restart_for_channel_test(false));
        let mut original = Vec::new();
        while let Some(Some(envelope)) = outgoing_rx.next().now_or_never() {
            original.push(envelope);
        }
        let all_id = original.iter().find(|envelope| matches!(
            envelope.payload.as_ref(), Some(proto::envelope::Payload::RestartLanguageServers(request)) if request.all
        )).expect("current global restart").id;
        let ordinary_id = original.iter().find(|envelope| matches!(
            envelope.payload.as_ref(), Some(proto::envelope::Payload::RestartLanguageServers(request)) if !request.all && request.buffer_ids==[17]
        )).expect("ordinary Copper buffer restart").id;
        assert_eq!(client.buffer.lock().len(), 1);
        assert_eq!(client.response_channels.lock().len(), 2);
        let (replacement_input, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, mut replacement_output) = mpsc::unbounded();
        client.reconnect(incoming_rx, outgoing_tx, &cx.to_async());
        let mut resync = Box::pin(client.resync(Duration::from_secs(5)));
        assert!(resync.as_mut().now_or_never().is_none());
        cx.run_until_parked();
        let mut startup = Vec::new();
        while let Some(Some(envelope)) = replacement_output.next().now_or_never() {
            startup.push(envelope);
        }
        let flush = startup
            .iter()
            .find(|envelope| {
                matches!(
                    envelope.payload.as_ref(),
                    Some(proto::envelope::Payload::FlushBufferedMessages(_))
                )
            })
            .expect("existing resync flush");
        replacement_input
            .unbounded_send(proto::Ack {}.into_envelope(100, Some(flush.id), None))
            .expect("resync acknowledgment");
        resync.await.expect("resync current connection");
        let mut replayed = Vec::new();
        while let Some(Some(envelope)) = replacement_output.next().now_or_never() {
            replayed.push(envelope);
        }
        let restart_ids = replayed
            .iter()
            .filter_map(|envelope| {
                matches!(
                    envelope.payload.as_ref(),
                    Some(proto::envelope::Payload::RestartLanguageServers(_))
                )
                .then_some(envelope.id)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            restart_ids,
            [ordinary_id],
            "only the ordinary Copper restart is replayed"
        );
        drop(all);
        assert_eq!(client.response_channels.lock().len(), 1);
        replacement_input
            .unbounded_send(proto::Ack {}.into_envelope(101, Some(all_id), None))
            .expect("late obsolete global reply");
        let fresh = requests.request(project_restart_for_channel_test(true));
        let request = replacement_output
            .next()
            .now_or_never()
            .flatten()
            .expect("fresh explicit restart");
        assert!(
            matches!(request.payload.as_ref(), Some(proto::envelope::Payload::RestartLanguageServers(request)) if request.all)
        );
        replacement_input
            .unbounded_send(proto::Ack {}.into_envelope(102, Some(request.id), None))
            .expect("fresh acknowledgment");
        fresh.await.expect("fresh explicit restart response");
        let mut acknowledgment = proto::Ack {}.into_envelope(103, Some(ordinary_id), None);
        acknowledgment.ack_id = Some(ordinary_id);
        replacement_input
            .unbounded_send(acknowledgment)
            .expect("ordinary acknowledgment");
        ordinary
            .await
            .expect("ordinary Copper restart response still routes");
        cx.run_until_parked();
        assert!(client.response_channels.lock().is_empty());
        assert!(client.buffer.lock().is_empty());
    }

    #[gpui::test]
    async fn test_channel_client_project_restart_cancellation_send_and_failure(
        cx: &mut TestAppContext,
    ) {
        for poll_request in [false, true] {
            let (_incoming_tx, incoming_rx) = mpsc::unbounded();
            let (outgoing_tx, outgoing_rx) = mpsc::unbounded();
            let client = cx.update(|cx| {
                ChannelClient::new(incoming_rx, outgoing_tx, cx, "current-restart", false)
            });
            let requests: AnyProtoClient = client.clone().into();
            let mut pending = Box::pin(requests.request(project_restart_for_channel_test(true)));
            if poll_request {
                assert!(pending.as_mut().now_or_never().is_none());
            }
            assert_eq!(client.response_channels.lock().len(), 1);
            assert!(client.buffer.lock().is_empty());
            drop(pending);
            assert!(client.response_channels.lock().is_empty());
            requests
                .send(project_restart_for_channel_test(true))
                .expect("one-way current global restart");
            assert!(client.buffer.lock().is_empty());
            drop(outgoing_rx);
            let error = requests
                .request(project_restart_for_channel_test(true))
                .await
                .expect_err("closed current connection");
            let message = format!("{error:#}");
            assert!(
                message.contains("reconnect") && message.contains("retry") && message.len() < 512
            );
            assert!(!message.contains("private-restart-canary"));
            assert!(
                requests
                    .send(project_restart_for_channel_test(true))
                    .is_err()
            );
            assert!(client.response_channels.lock().is_empty());
            assert!(client.buffer.lock().is_empty());
            requests
                .send(project_restart_for_channel_test(false))
                .expect("ordinary Copper restart remains buffered during reconnect");
            assert_eq!(client.buffer.lock().len(), 1);
        }
    }

    #[gpui::test]
    fn test_channel_client_unpolled_stream_cancellation_releases_bookkeeping(
        cx: &mut TestAppContext,
    ) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded();
        let client = cx
            .update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "stream-cancel", false));
        let request = client
            .request_stream_dynamic(proto::Test { id: 17 }.into_envelope(0, None, None), "Test");
        assert_eq!(client.stream_response_channels.lock().len(), 1);
        assert_eq!(client.buffer.lock().len(), 1);
        drop(request);
        cx.run_until_parked();
        assert!(
            client.stream_response_channels.lock().is_empty(),
            "cancelling before the first poll must remove the registered stream"
        );
        assert_eq!(
            client.buffer.lock().len(),
            1,
            "local bookkeeping cleanup does not cancel ordinary remote execution or replay"
        );

        let stream = client
            .request_stream_dynamic(proto::Test { id: 18 }.into_envelope(0, None, None), "Test")
            .now_or_never()
            .expect("ordinary stream construction completes")
            .expect("ordinary stream remains supported");
        assert_eq!(client.stream_response_channels.lock().len(), 1);
        drop(stream);
        assert!(client.stream_response_channels.lock().is_empty());
    }

    #[gpui::test]
    fn test_channel_client_unpolled_stream_does_not_retain_response_owner(cx: &mut TestAppContext) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded();
        let (outgoing_tx, _outgoing_rx) = mpsc::unbounded();
        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "stream-owner", false));
        let response_owner = Arc::downgrade(&client.stream_response_channels);
        let channel_owner = Arc::downgrade(&client);
        let request = client
            .request_stream_dynamic(proto::Test { id: 17 }.into_envelope(0, None, None), "Test");
        assert_eq!(client.stream_response_channels.lock().len(), 1);
        drop(client);
        cx.run_until_parked();
        assert!(channel_owner.upgrade().is_none());
        assert!(
            response_owner.upgrade().is_none(),
            "an unpolled future cannot keep a released channel's response map alive"
        );
        drop(request);
    }

    #[gpui::test]
    async fn test_channel_client_request_stream_terminates_on_error(cx: &mut TestAppContext) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<Envelope>();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();

        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "test-client", false));

        // The client sends RemoteStarted on startup; drain the outgoing channel
        // so it doesn't block.
        let _drain_outgoing = cx
            .executor()
            .spawn(async move { while outgoing_rx.next().await.is_some() {} });

        let mut stream = client
            .request_stream_dynamic(proto::Test { id: 0 }.into_envelope(0, None, None), "Test")
            .await
            .unwrap();

        let request_id = 0;

        incoming_tx
            .unbounded_send(proto::Test { id: 1 }.into_envelope(100, Some(request_id), None))
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(
            proto::Test::from_envelope(first).unwrap(),
            proto::Test { id: 1 }
        );

        // Send an Error without a trailing EndStream. The Error alone should
        // terminate the stream.
        incoming_tx
            .unbounded_send(
                ErrorCode::Internal
                    .message("boom".to_string())
                    .to_proto()
                    .into_envelope(101, Some(request_id), None),
            )
            .unwrap();

        let second = stream.next().await.unwrap();
        let error = second.unwrap_err();
        assert!(
            format!("{error}").contains("boom"),
            "expected error to surface server message, got: {error}"
        );

        assert!(stream.next().await.is_none());
        assert_eq!(client.stream_response_channels.lock().len(), 0);
    }

    #[gpui::test]
    async fn test_channel_client_dropping_stream_request_before_response_cleans_up_channel(
        cx: &mut TestAppContext,
    ) {
        let (_incoming_tx, incoming_rx) = mpsc::unbounded::<Envelope>();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();

        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "test-client", false));

        let _drain_outgoing = cx
            .executor()
            .spawn(async move { while outgoing_rx.next().await.is_some() {} });

        let stream = client
            .request_stream_dynamic(proto::Test { id: 0 }.into_envelope(0, None, None), "Test")
            .await
            .unwrap();

        assert_eq!(client.stream_response_channels.lock().len(), 1);

        drop(stream);
        cx.run_until_parked();

        assert_eq!(
            client.stream_response_channels.lock().len(),
            0,
            "dropping a stream before any responses arrive should remove response channel bookkeeping"
        );
    }

    #[gpui::test]
    async fn test_channel_client_dropping_stream_request_before_completion(
        cx: &mut TestAppContext,
    ) {
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<Envelope>();
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();

        let client =
            cx.update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "test-client", false));

        let _drain_outgoing = cx
            .executor()
            .spawn(async move { while outgoing_rx.next().await.is_some() {} });

        let mut stream = client
            .request_stream_dynamic(proto::Test { id: 0 }.into_envelope(0, None, None), "Test")
            .await
            .unwrap();

        let request_id = 0;

        incoming_tx
            .unbounded_send(proto::Test { id: 1 }.into_envelope(100, Some(request_id), None))
            .unwrap();
        let _ = stream.next().await.unwrap().unwrap();

        assert_eq!(client.stream_response_channels.lock().len(), 1);

        drop(stream);

        // Inject an orphaned non-terminal response. The read loop should detect
        // that the consumer has been dropped and clean up its bookkeeping (no
        // EndStream sent here on purpose, otherwise the cleanup would happen
        // via the terminal-response path and mask the bug under test).
        incoming_tx
            .unbounded_send(proto::Test { id: 2 }.into_envelope(101, Some(request_id), None))
            .unwrap();

        cx.run_until_parked();

        assert_eq!(
            client.stream_response_channels.lock().len(),
            0,
            "stream channel should be removed once the consumer has dropped the stream"
        );
    }
}

impl From<SshConnectionOptions> for RemoteConnectionOptions {
    fn from(opts: SshConnectionOptions) -> Self {
        RemoteConnectionOptions::Ssh(opts)
    }
}

impl From<WslConnectionOptions> for RemoteConnectionOptions {
    fn from(opts: WslConnectionOptions) -> Self {
        RemoteConnectionOptions::Wsl(opts)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl From<crate::transport::mock::MockConnectionOptions> for RemoteConnectionOptions {
    fn from(opts: crate::transport::mock::MockConnectionOptions) -> Self {
        RemoteConnectionOptions::Mock(opts)
    }
}

#[cfg(target_os = "windows")]
/// Open a wsl path (\\wsl.localhost\<distro>\path)
#[derive(Debug, Clone, PartialEq, Eq, gpui::Action)]
#[action(namespace = workspace, no_json, no_register)]
pub struct OpenWslPath {
    pub distro: WslConnectionOptions,
    pub paths: Vec<PathBuf>,
}

#[async_trait(?Send)]
pub trait RemoteConnection: Send + Sync {
    fn start_proxy(
        &self,
        unique_identifier: String,
        mode: ProxyMode,
        incoming_tx: UnboundedSender<Envelope>,
        outgoing_rx: UnboundedReceiver<Envelope>,
        connection_activity_tx: Sender<()>,
        delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Task<Result<i32>>;
    fn doctor_server_binary(&self) -> Option<String> {
        None
    }

    fn upload_directory(
        &self,
        src_path: PathBuf,
        dest_path: RemotePathBuf,
        cx: &App,
    ) -> Task<Result<()>>;
    async fn kill(&self) -> Result<()>;
    fn has_been_killed(&self) -> bool;
    fn shares_network_interface(&self) -> bool {
        false
    }
    fn build_command(
        &self,
        program: Option<String>,
        args: &[String],
        env: &HashMap<String, String>,
        working_dir: Option<String>,
        port_forward: Option<(u16, String, u16)>,
        interactive: Interactive,
    ) -> Result<CommandTemplate>;
    fn build_forward_ports_command(
        &self,
        forwards: Vec<(u16, String, u16)>,
    ) -> Result<CommandTemplate>;
    fn connection_options(&self) -> RemoteConnectionOptions;
    fn path_style(&self) -> PathStyle;
    /// The remote platform (OS and architecture), detected during connection setup.
    fn remote_platform(&self) -> RemotePlatform;
    /// The remote host's OS version (e.g. `"ubuntu 24.04"` or `"15.6.1"`),
    /// detected during connection setup. `None` if it could not be determined.
    fn remote_os_version(&self) -> Option<String>;
    fn shell(&self) -> String;
    fn default_system_shell(&self) -> String;
    fn has_wsl_interop(&self) -> bool;

    #[cfg(any(test, feature = "test-support"))]
    fn simulate_disconnect(&self, _: &AsyncApp) {}
}

type ResponseChannels =
    Arc<Mutex<HashMap<MessageId, oneshot::Sender<(Envelope, oneshot::Sender<()>)>>>>;
type StreamResponseChannels =
    Arc<Mutex<HashMap<MessageId, UnboundedSender<(Result<Envelope>, oneshot::Sender<()>)>>>>;

struct Signal<T: 'static> {
    tx: Mutex<Option<oneshot::Sender<T>>>,
    rx: Shared<Task<Option<T>>>,
}

impl<T: Send + Clone + 'static> Signal<T> {
    pub fn new(cx: &App) -> Self {
        let (tx, rx) = oneshot::channel();

        let task = cx
            .background_executor()
            .spawn(async move { rx.await.ok() })
            .shared();

        Self {
            tx: Mutex::new(Some(tx)),
            rx: task,
        }
    }

    fn set(&self, value: T) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(value);
        }
    }

    fn wait(&self) -> Shared<Task<Option<T>>> {
        self.rx.clone()
    }
}

pub(crate) struct ChannelClient {
    next_message_id: AtomicU32,
    outgoing_tx: Mutex<mpsc::UnboundedSender<Envelope>>,
    buffer: Mutex<VecDeque<Envelope>>,
    response_channels: ResponseChannels,
    stream_response_channels: StreamResponseChannels,
    message_handlers: Mutex<ProtoMessageHandlerSet>,
    max_received: AtomicU32,
    activity: RemoteRequestActivity,
    name: &'static str,
    task: Mutex<Task<Result<()>>>,
    remote_started: Signal<()>,
    has_wsl_interop: bool,
    executor: BackgroundExecutor,
}

impl ChannelClient {
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn request_activity(&self) -> RemoteRequestActivity {
        self.activity.clone()
    }

    pub(crate) fn new(
        incoming_rx: mpsc::UnboundedReceiver<Envelope>,
        outgoing_tx: mpsc::UnboundedSender<Envelope>,
        cx: &App,
        name: &'static str,
        has_wsl_interop: bool,
    ) -> Arc<Self> {
        Self::new_with_scopes(
            incoming_rx
                .map(|envelope| (PeerId { owner_id: 0, id: 0 }, envelope))
                .boxed(),
            outgoing_tx,
            cx,
            name,
            has_wsl_interop,
        )
    }

    fn new_with_scopes(
        incoming_rx: BoxStream<'static, (PeerId, Envelope)>,
        outgoing_tx: mpsc::UnboundedSender<Envelope>,
        cx: &App,
        name: &'static str,
        has_wsl_interop: bool,
    ) -> Arc<Self> {
        Arc::new_cyclic(|this| Self {
            outgoing_tx: Mutex::new(outgoing_tx),
            next_message_id: AtomicU32::new(0),
            max_received: AtomicU32::new(0),
            activity: RemoteRequestActivity::default(),
            response_channels: ResponseChannels::default(),
            stream_response_channels: StreamResponseChannels::default(),
            message_handlers: Default::default(),
            buffer: Mutex::new(VecDeque::new()),
            name,
            executor: cx.background_executor().clone(),
            task: Mutex::new(Self::start_handling_messages(
                this.clone(),
                incoming_rx,
                &cx.to_async(),
            )),
            remote_started: Signal::new(cx),
            has_wsl_interop,
        })
    }

    fn wait_for_remote_started(&self) -> Shared<Task<Option<()>>> {
        self.remote_started.wait()
    }

    fn start_handling_messages(
        this: Weak<Self>,
        mut incoming_rx: BoxStream<'static, (PeerId, Envelope)>,
        cx: &AsyncApp,
    ) -> Task<Result<()>> {
        cx.spawn(async move |cx| {
            if let Some(this) = this.upgrade() {
                let envelope = proto::RemoteStarted {}.into_envelope(0, None, None);
                this.outgoing_tx.lock().unbounded_send(envelope).ok();
            };

            while let Some((peer_id, incoming)) = incoming_rx.next().await {
                let Some(this) = this.upgrade() else {
                    return anyhow::Ok(());
                };
                if let Some(ack_id) = incoming.ack_id {
                    let mut buffer = this.buffer.lock();
                    while buffer.front().is_some_and(|msg| msg.id <= ack_id) {
                        buffer.pop_front();
                    }
                }
                if incoming.responding_to.is_none() && this.activity.sealed.load(SeqCst) {
                    continue;
                }

                if let Some(proto::envelope::Payload::FlushBufferedMessages(_)) = &incoming.payload
                {
                    log::debug!(
                        "{}:remote message received. name:FlushBufferedMessages",
                        this.name
                    );
                    {
                        let buffer = this.buffer.lock();
                        for envelope in buffer.iter() {
                            this.outgoing_tx
                                .lock()
                                .unbounded_send(envelope.clone())
                                .ok();
                        }
                    }
                    let mut envelope = proto::Ack {}.into_envelope(0, Some(incoming.id), None);
                    envelope.id = this.next_message_id.fetch_add(1, SeqCst);
                    this.outgoing_tx.lock().unbounded_send(envelope).ok();
                    continue;
                }

                if let Some(proto::envelope::Payload::RemoteStarted(_)) = &incoming.payload {
                    this.remote_started.set(());
                    let mut envelope = proto::Ack {}.into_envelope(0, Some(incoming.id), None);
                    envelope.id = this.next_message_id.fetch_add(1, SeqCst);
                    this.outgoing_tx.lock().unbounded_send(envelope).ok();
                    continue;
                }

                this.max_received.store(incoming.id, SeqCst);

                if let Some(request_id) = incoming.responding_to {
                    let request_id = MessageId(request_id);
                    // An incoming response with no payload is malformed; drop
                    // it. The request future and any stream consumers will
                    // remain pending until either a real response arrives or
                    // the connection is torn down.
                    if incoming.payload.is_none() {
                        continue;
                    }
                    let sender = this.response_channels.lock().remove(&request_id);
                    if let Some(sender) = sender {
                        let (tx, rx) = oneshot::channel();
                        sender.send((incoming, tx)).ok();
                        rx.await.ok();
                    } else {
                        let terminal_stream_response = matches!(
                            &incoming.payload,
                            Some(proto::envelope::Payload::Error(_))
                                | Some(proto::envelope::Payload::EndStream(_))
                        );
                        let sender = if terminal_stream_response {
                            this.stream_response_channels.lock().remove(&request_id)
                        } else {
                            this.stream_response_channels
                                .lock()
                                .get(&request_id)
                                .cloned()
                        };
                        if let Some(sender) = sender {
                            let (tx, rx) = oneshot::channel();
                            if sender.unbounded_send((Ok(incoming), tx)).is_err() {
                                this.stream_response_channels.lock().remove(&request_id);
                                continue;
                            }
                            rx.await.ok();
                        }
                    }
                } else if let Some(envelope) =
                    build_typed_envelope(peer_id, Instant::now(), incoming)
                {
                    let type_name = envelope.payload_type_name();
                    let message_id = envelope.message_id();
                    if let Some(future) = ProtoMessageHandlerSet::handle_message(
                        &this.message_handlers,
                        envelope,
                        this.clone().into(),
                        cx.clone(),
                    ) {
                        this.activity.handlers.fetch_add(1, SeqCst);
                        let activity = RemoteHandlerActivity(this.activity.clone());
                        log::debug!("{}:remote message received. name:{type_name}", this.name);
                        cx.foreground_executor()
                            .spawn(async move {
                                let _activity = activity;
                                match future.await {
                                    Ok(_) => {
                                        log::debug!(
                                            "{}:remote message handled. name:{type_name}",
                                            this.name
                                        );
                                    }
                                    Err(error) => {
                                        log::error!(
                                            "{}:error handling message. type:{}, error:{:#}",
                                            this.name,
                                            type_name,
                                            format!("{error:#}").lines().fold(
                                                String::new(),
                                                |mut message, line| {
                                                    if !message.is_empty() {
                                                        message.push(' ');
                                                    }
                                                    message.push_str(line);
                                                    message
                                                }
                                            )
                                        );
                                    }
                                }
                            })
                            .detach()
                    } else {
                        log::error!("{}:unhandled remote message name:{type_name}", this.name);
                        if let Err(e) = AnyProtoClient::from(this.clone()).send_response(
                            message_id,
                            anyhow::anyhow!("no handler registered for {type_name}").to_proto(),
                        ) {
                            log::error!(
                                "{}:error sending error response for {type_name}:{e:#}",
                                this.name
                            );
                        }
                    }
                }
            }
            anyhow::Ok(())
        })
    }

    pub(crate) fn reconnect(
        self: &Arc<Self>,
        incoming_rx: UnboundedReceiver<Envelope>,
        outgoing_tx: UnboundedSender<Envelope>,
        cx: &AsyncApp,
    ) {
        self.replace_input(
            incoming_rx
                .map(|envelope| (PeerId { owner_id: 0, id: 0 }, envelope))
                .boxed(),
            outgoing_tx,
            cx,
        );
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn reconnect_with_peer(
        self: &Arc<Self>,
        incoming: UnboundedReceiver<Envelope>,
        outgoing: UnboundedSender<Envelope>,
        peer: PeerId,
        cx: &AsyncApp,
    ) {
        // A proxy's captured peer cannot change while its queued envelopes wait.
        self.replace_input(
            incoming.map(move |envelope| (peer, envelope)).boxed(),
            outgoing,
            cx,
        );
    }

    fn replace_input(
        self: &Arc<Self>,
        incoming: BoxStream<'static, (PeerId, Envelope)>,
        outgoing: UnboundedSender<Envelope>,
        cx: &AsyncApp,
    ) {
        *self.outgoing_tx.lock() = outgoing;
        *self.task.lock() = Self::start_handling_messages(Arc::downgrade(self), incoming, cx);
    }

    fn request<T: RequestMessage>(
        &self,
        payload: T,
    ) -> impl 'static + Future<Output = Result<T::Response>> + use<T> {
        self.request_internal(payload, true)
    }

    fn request_internal<T: RequestMessage>(
        &self,
        payload: T,
        use_buffer: bool,
    ) -> impl 'static + Future<Output = Result<T::Response>> + use<T> {
        log::debug!("remote request start. name:{}", T::NAME);
        let response =
            self.request_dynamic(payload.into_envelope(0, None, None), T::NAME, use_buffer);
        async move {
            let response = response.await?;
            log::debug!("remote request finish. name:{}", T::NAME);
            T::Response::from_envelope(response).context("received a response of the wrong type")
        }
    }

    async fn resync(&self, timeout: Duration) -> Result<()> {
        smol::future::or(
            async {
                self.request_internal(proto::FlushBufferedMessages {}, false)
                    .await?;

                for envelope in self.buffer.lock().iter() {
                    self.outgoing_tx
                        .lock()
                        .unbounded_send(envelope.clone())
                        .ok();
                }
                Ok(())
            },
            async {
                self.executor.timer(timeout).await;
                anyhow::bail!("Timed out resyncing remote client")
            },
        )
        .await
    }

    async fn ping(&self, timeout: Duration) -> Result<()> {
        smol::future::or(
            async {
                self.request(proto::Ping {}).await?;
                Ok(())
            },
            async {
                self.executor.timer(timeout).await;
                anyhow::bail!("Timed out pinging remote client")
            },
        )
        .await
    }

    fn send<T: EnvelopedMessage>(&self, payload: T) -> Result<()> {
        log::debug!("remote send name:{}", T::NAME);
        self.send_dynamic(payload.into_envelope(0, None, None))
    }

    fn request_dynamic(
        &self,
        mut envelope: proto::Envelope,
        type_name: &'static str,
        use_buffer: bool,
    ) -> impl 'static + Future<Output = Result<proto::Envelope>> + use<> {
        let use_buffer = use_buffer && Self::uses_reconnect_buffer(&envelope);
        envelope.id = self.next_message_id.fetch_add(1, SeqCst);
        let message_id = MessageId(envelope.id);
        let (tx, rx) = oneshot::channel();
        let mut response_channels_lock = self.response_channels.lock();
        response_channels_lock.insert(message_id, tx);
        drop(response_channels_lock);

        let cleanup = (!use_buffer).then(|| {
            // A pending request must not keep the channel's response map alive after owner loss.
            let response_channels = Arc::downgrade(&self.response_channels);
            util::defer(move || {
                if let Some(response_channels) = response_channels.upgrade() {
                    response_channels.lock().remove(&message_id);
                }
            })
        });
        let result = if use_buffer {
            self.send_buffered(envelope)
        } else {
            self.send_unbuffered(envelope)
        };
        async move {
            let _cleanup = cleanup;
            if let Err(error) = &result {
                log::error!("failed to send message: {error}");
                anyhow::bail!("failed to send message: {error}");
            }

            let response = rx.await.context("connection lost")?.0;
            if let Some(proto::envelope::Payload::Error(error)) = &response.payload {
                return Err(RpcError::from_proto(error, type_name));
            }
            Ok(response)
        }
    }

    fn request_stream_dynamic(
        &self,
        mut envelope: proto::Envelope,
        type_name: &'static str,
    ) -> impl 'static + Future<Output = Result<BoxStream<'static, Result<proto::Envelope>>>> {
        envelope.id = self.next_message_id.fetch_add(1, SeqCst);
        let message_id = MessageId(envelope.id);
        let (tx, rx) = mpsc::unbounded();
        self.stream_response_channels.lock().insert(message_id, tx);

        // Registration happens before the future is returned, so cleanup must also own cancellation
        // before its first poll, without keeping a released channel's response map alive.
        let cleanup_stream_response_channel = util::defer({
            let stream_response_channels = Arc::downgrade(&self.stream_response_channels);
            move || {
                if let Some(stream_response_channels) = stream_response_channels.upgrade() {
                    stream_response_channels.lock().remove(&message_id);
                }
            }
        });

        let result = self.send_buffered(envelope);
        async move {
            if let Err(error) = &result {
                log::error!("failed to send message: {error}");
                anyhow::bail!("failed to send message: {error}");
            }

            Ok(rx
                .filter_map(move |(response, _barrier)| {
                    // Keep the cleanup guard alive until the returned stream is dropped.
                    let _keep_cleanup_guard_alive = &cleanup_stream_response_channel;
                    futures::future::ready(match response {
                        Ok(response) => {
                            if let Some(proto::envelope::Payload::Error(error)) = &response.payload
                            {
                                Some(Err(RpcError::from_proto(error, type_name)))
                            } else if let Some(proto::envelope::Payload::EndStream(_)) =
                                &response.payload
                            {
                                None
                            } else {
                                Some(Ok(response))
                            }
                        }
                        Err(error) => Some(Err(error)),
                    })
                })
                .boxed())
        }
    }

    pub fn send_dynamic(&self, mut envelope: proto::Envelope) -> Result<()> {
        envelope.id = self.next_message_id.fetch_add(1, SeqCst);
        self.send_buffered(envelope)
    }

    fn uses_reconnect_buffer(envelope: &proto::Envelope) -> bool {
        // Replaying startup or global restart authority could override newer ownership or stop decisions.
        !matches!(
            envelope.payload.as_ref(),
            Some(proto::envelope::Payload::SetJvmProjectConsumers(_))
        ) && !matches!(
            envelope.payload.as_ref(),
            Some(proto::envelope::Payload::RestartLanguageServers(request)) if request.all
        )
    }

    fn send_buffered(&self, mut envelope: proto::Envelope) -> Result<()> {
        if !Self::uses_reconnect_buffer(&envelope) {
            return self.send_unbuffered(envelope);
        }
        envelope.ack_id = Some(self.max_received.load(SeqCst));
        self.buffer.lock().push_back(envelope.clone());
        // ignore errors on send (happen while we're reconnecting)
        // assume that the global "disconnected" overlay is sufficient.
        self.outgoing_tx.lock().unbounded_send(envelope).ok();
        Ok(())
    }

    fn send_unbuffered(&self, mut envelope: proto::Envelope) -> Result<()> {
        envelope.ack_id = Some(self.max_received.load(SeqCst));
        self.outgoing_tx
            .lock()
            .unbounded_send(envelope)
            .map_err(|_| {
                anyhow!("The current remote connection is unavailable; reconnect and retry")
            })
    }
}

impl ProtoClient for ChannelClient {
    fn request(
        &self,
        envelope: proto::Envelope,
        request_type: &'static str,
    ) -> BoxFuture<'static, Result<proto::Envelope>> {
        self.request_dynamic(envelope, request_type, true).boxed()
    }

    fn request_stream(
        &self,
        envelope: proto::Envelope,
        request_type: &'static str,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<proto::Envelope>>>> {
        self.request_stream_dynamic(envelope, request_type).boxed()
    }

    fn send(&self, envelope: proto::Envelope, _message_type: &'static str) -> Result<()> {
        self.send_dynamic(envelope)
    }

    fn send_response(&self, envelope: Envelope, _message_type: &'static str) -> anyhow::Result<()> {
        self.send_dynamic(envelope)
    }

    fn message_handler_set(&self) -> &Mutex<ProtoMessageHandlerSet> {
        &self.message_handlers
    }

    fn is_via_collab(&self) -> bool {
        false
    }

    fn has_wsl_interop(&self) -> bool {
        self.has_wsl_interop
    }
}
