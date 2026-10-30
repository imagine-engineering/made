//! Mock transport for testing remote connections.
//!
//! This module provides a mock implementation of the `RemoteConnection` trait
//! that allows testing remote editing functionality without actual SSH/WSL/Docker
//! connections.
//!
//! # Usage
//!
//! ```rust,ignore
//! use remote::{MockConnection, RemoteClient};
//!
//! #[gpui::test]
//! async fn test_remote_editing(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
//!     let (opts, server_session) = MockConnection::new(cx, server_cx);
//!
//!     // Create the headless project (server side)
//!     server_cx.update(HeadlessProject::init);
//!     let _headless = server_cx.new(|cx| {
//!         HeadlessProject::new(
//!             HeadlessAppState { session: server_session, /* ... */ },
//!             false,
//!             cx,
//!         )
//!     });
//!
//!     // Create the client using the helper
//!     let (client, server_client) = RemoteClient::new_mock(cx, server_cx).await;
//!     // ... test logic ...
//! }
//! ```

use crate::remote_client::{
    ChannelClient, CommandTemplate, Interactive, RemoteClientDelegate, RemoteConnection,
    RemoteConnectionOptions,
};
#[cfg(target_os = "macos")]
use anyhow::Context as _;
use anyhow::Result;
use async_trait::async_trait;
use collections::HashMap;
use futures::{
    FutureExt, SinkExt, StreamExt,
    channel::{
        mpsc::{self, Sender},
        oneshot,
    },
    select_biased,
};
use gpui::{App, AppContext as _, AsyncApp, Global, Task, TestAppContext};
use rpc::{
    AnyProtoClient,
    proto::{Envelope, PeerId},
};
#[cfg(target_os = "macos")]
use std::sync::atomic::AtomicBool;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
#[cfg(target_os = "macos")]
use util::command::{Stdio, new_command};
use util::paths::{PathStyle, RemotePathBuf};

/// Unique identifier for a mock connection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MockConnectionOptions {
    pub id: u64,
}

/// A mock implementation of `RemoteConnection` for testing.
pub struct MockRemoteConnection {
    options: MockConnectionOptions,
    server_channel: Arc<ChannelClient>,
    server_cx: SendableCx,
    receiving_peer: Option<PeerId>,
}

impl MockRemoteConnection {
    fn bind_server_input(
        &self,
        incoming: mpsc::UnboundedReceiver<Envelope>,
        outgoing: mpsc::UnboundedSender<Envelope>,
        cx: &AsyncApp,
    ) {
        if let Some(peer) = self.receiving_peer {
            self.server_channel
                .reconnect_with_peer(incoming, outgoing, peer, cx);
        } else {
            self.server_channel.reconnect(incoming, outgoing, cx);
        }
    }
}

/// Wrapper to pass `AsyncApp` across thread boundaries in tests.
///
/// # Safety
///
/// This is safe because in test mode, GPUI is always single-threaded and so
/// having access to one async app means being on the same main thread.
pub(crate) struct SendableCx(AsyncApp);

impl SendableCx {
    pub(crate) fn new(cx: &TestAppContext) -> Self {
        Self(cx.to_async())
    }

    pub(crate) fn get(&self, _: &AsyncApp) -> AsyncApp {
        self.0.clone()
    }
}

// SAFETY: In test mode, GPUI is always single-threaded, and SendableCx
// is only accessed from the main thread via the get() method which
// requires a valid AsyncApp reference.
unsafe impl Send for SendableCx {}
unsafe impl Sync for SendableCx {}

/// Global registry that holds pre-created mock connections.
///
/// When `ConnectionPool::connect` is called with `MockConnectionOptions`,
/// it retrieves the connection from this registry.
#[derive(Default)]
pub struct MockConnectionRegistry {
    pending: HashMap<u64, (oneshot::Receiver<()>, Arc<MockRemoteConnection>)>,
}

impl Global for MockConnectionRegistry {}

impl MockConnectionRegistry {
    /// Called by `ConnectionPool::connect` to retrieve a pre-registered mock connection.
    pub fn take(
        &mut self,
        opts: &MockConnectionOptions,
    ) -> Option<impl Future<Output = Arc<MockRemoteConnection>> + use<>> {
        let (guard, con) = self.pending.remove(&opts.id)?;
        Some(async move {
            _ = guard.await;
            con
        })
    }
}

/// Helper for creating mock connection pairs in tests.
pub struct MockConnection;

pub type ConnectGuard = oneshot::Sender<()>;

impl MockConnection {
    /// Creates a new mock connection pair for testing.
    ///
    /// This function:
    /// 1. Creates a unique `MockConnectionOptions` identifier
    /// 2. Sets up the server-side channel (returned as `AnyProtoClient`)
    /// 3. Creates a `MockRemoteConnection` and registers it in the global registry
    /// 4. The connection will be retrieved from the registry when `ConnectionPool::connect` is called
    ///
    /// Returns:
    /// - `MockConnectionOptions` to pass to `remote::connect()` or `RemoteClient` creation
    /// - `AnyProtoClient` to pass to `HeadlessProject::new()` as the session
    ///
    /// # Arguments
    /// - `client_cx`: The test context for the client side
    /// - `server_cx`: The test context for the server/headless side
    pub(crate) fn new(
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> (MockConnectionOptions, AnyProtoClient, ConnectGuard) {
        let (options, client, guard, _) = Self::new_with_activity(client_cx, server_cx);
        (options, client, guard)
    }

    pub(crate) fn new_with_activity(
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> (
        MockConnectionOptions,
        AnyProtoClient,
        ConnectGuard,
        crate::RemoteRequestActivity,
    ) {
        Self::new_with_scope_and_activity(None, client_cx, server_cx)
    }

    pub(crate) fn new_scoped_with_activity(
        peer: PeerId,
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> Result<(
        MockConnectionOptions,
        AnyProtoClient,
        ConnectGuard,
        crate::RemoteRequestActivity,
    )> {
        anyhow::ensure!(
            peer.id != 0,
            "Scoped mock connections require a nonzero receiving peer"
        );
        Ok(Self::new_with_scope_and_activity(
            Some(peer),
            client_cx,
            server_cx,
        ))
    }

    fn new_with_scope_and_activity(
        receiving_peer: Option<PeerId>,
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> (
        MockConnectionOptions,
        AnyProtoClient,
        ConnectGuard,
        crate::RemoteRequestActivity,
    ) {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let opts = MockConnectionOptions { id };
        let (server_client, connect_guard, activity) = Self::new_with_opts_and_scope_and_activity(
            opts.clone(),
            receiving_peer,
            client_cx,
            server_cx,
        );
        (opts, server_client, connect_guard, activity)
    }

    /// Creates a mock connection pair for existing `MockConnectionOptions`.
    ///
    /// This is useful when simulating reconnection: after a connection is torn
    /// down, register a new mock server under the same options so the next
    /// `ConnectionPool::connect` call finds it.
    pub(crate) fn new_with_opts(
        opts: MockConnectionOptions,
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> (AnyProtoClient, ConnectGuard) {
        let (client, guard, _) = Self::new_with_opts_and_activity(opts, client_cx, server_cx);
        (client, guard)
    }

    fn new_with_opts_and_activity(
        opts: MockConnectionOptions,
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> (AnyProtoClient, ConnectGuard, crate::RemoteRequestActivity) {
        Self::new_with_opts_and_scope_and_activity(opts, None, client_cx, server_cx)
    }

    fn new_with_opts_and_scope_and_activity(
        opts: MockConnectionOptions,
        receiving_peer: Option<PeerId>,
        client_cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) -> (AnyProtoClient, ConnectGuard, crate::RemoteRequestActivity) {
        let (outgoing_tx, _) = mpsc::unbounded::<Envelope>();
        let (_, incoming_rx) = mpsc::unbounded::<Envelope>();
        let server_client = server_cx
            .update(|cx| ChannelClient::new(incoming_rx, outgoing_tx, cx, "mock-server", false));

        let connection = Arc::new(MockRemoteConnection {
            options: opts.clone(),
            server_channel: server_client.clone(),
            server_cx: SendableCx::new(server_cx),
            receiving_peer,
        });

        let (tx, rx) = oneshot::channel();

        client_cx.update(|cx| {
            cx.default_global::<MockConnectionRegistry>()
                .pending
                .insert(opts.id, (rx, connection));
        });

        let activity = server_client.request_activity();
        (server_client.into(), tx, activity)
    }
}

#[async_trait(?Send)]
impl RemoteConnection for MockRemoteConnection {
    async fn kill(&self) -> Result<()> {
        Ok(())
    }

    fn has_been_killed(&self) -> bool {
        false
    }

    fn build_command(
        &self,
        program: Option<String>,
        args: &[String],
        env: &HashMap<String, String>,
        _working_dir: Option<String>,
        _port_forward: Option<(u16, String, u16)>,
        _interactive: Interactive,
    ) -> Result<CommandTemplate> {
        let shell_program = program.unwrap_or_else(|| "sh".to_string());
        let mut shell_args = Vec::new();
        shell_args.push(shell_program);
        shell_args.extend(args.iter().cloned());
        Ok(CommandTemplate {
            program: "mock".into(),
            args: shell_args,
            env: env.clone(),
        })
    }

    fn build_forward_ports_command(
        &self,
        forwards: Vec<(u16, String, u16)>,
    ) -> Result<CommandTemplate> {
        Ok(CommandTemplate {
            program: "mock".into(),
            args: std::iter::once("-N".to_owned())
                .chain(forwards.into_iter().map(|(local_port, host, remote_port)| {
                    format!("{local_port}:{host}:{remote_port}")
                }))
                .collect(),
            env: Default::default(),
        })
    }

    fn upload_directory(
        &self,
        _src_path: PathBuf,
        _dest_path: RemotePathBuf,
        _cx: &App,
    ) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn connection_options(&self) -> RemoteConnectionOptions {
        RemoteConnectionOptions::Mock(self.options.clone())
    }

    fn simulate_disconnect(&self, cx: &AsyncApp) {
        let (outgoing_tx, _) = mpsc::unbounded::<Envelope>();
        let (_, incoming_rx) = mpsc::unbounded::<Envelope>();
        self.bind_server_input(incoming_rx, outgoing_tx, &self.server_cx.get(cx));
    }

    fn start_proxy(
        &self,
        _unique_identifier: String,
        _mode: crate::ProxyMode,
        mut client_incoming_tx: mpsc::UnboundedSender<Envelope>,
        mut client_outgoing_rx: mpsc::UnboundedReceiver<Envelope>,
        mut connection_activity_tx: Sender<()>,
        _delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Task<Result<i32>> {
        let (mut server_incoming_tx, server_incoming_rx) = mpsc::unbounded::<Envelope>();
        let (server_outgoing_tx, mut server_outgoing_rx) = mpsc::unbounded::<Envelope>();

        self.bind_server_input(
            server_incoming_rx,
            server_outgoing_tx,
            &self.server_cx.get(cx),
        );

        cx.background_spawn(async move {
            loop {
                select_biased! {
                    server_to_client = server_outgoing_rx.next().fuse() => {
                        let Some(server_to_client) = server_to_client else {
                            return Ok(1)
                        };
                        connection_activity_tx.try_send(()).ok();
                        client_incoming_tx.send(server_to_client).await.ok();
                    }
                    client_to_server = client_outgoing_rx.next().fuse() => {
                        let Some(client_to_server) = client_to_server else {
                            return Ok(1)
                        };
                        server_incoming_tx.send(client_to_server).await.ok();
                    }
                }
            }
        })
    }

    fn path_style(&self) -> PathStyle {
        PathStyle::local()
    }

    fn remote_platform(&self) -> crate::RemotePlatform {
        crate::RemotePlatform {
            os: crate::RemoteOs::Linux,
            arch: crate::RemoteArch::X86_64,
        }
    }

    fn remote_os_version(&self) -> Option<String> {
        None
    }

    fn shell(&self) -> String {
        "sh".to_owned()
    }

    fn default_system_shell(&self) -> String {
        "sh".to_owned()
    }

    fn has_wsl_interop(&self) -> bool {
        false
    }
}

#[cfg(target_os = "macos")]
pub struct ProcessRemoteConnection {
    binary: PathBuf,
    server_state_dir: PathBuf,
    profile_page_gate_dir: Option<PathBuf>,
    isolated_server_namespace: bool,
    test_data_directory: Option<PathBuf>,
    options: MockConnectionOptions,
    identifier: parking_lot::Mutex<Option<String>>,
    killed: AtomicBool,
}

#[cfg(target_os = "macos")]
impl ProcessRemoteConnection {
    pub fn new(binary: PathBuf, server_state_dir: PathBuf) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            binary,
            server_state_dir,
            profile_page_gate_dir: None,
            isolated_server_namespace: false,
            test_data_directory: None,
            options: MockConnectionOptions {
                id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            },
            identifier: parking_lot::Mutex::new(None),
            killed: AtomicBool::new(false),
        }
    }

    pub fn with_test_data_directory(mut self, directory: PathBuf) -> Self {
        self.server_state_dir = directory.join("server_state");
        self.test_data_directory = Some(directory);
        self
    }

    pub fn with_profile_page_gate(mut self, directory: PathBuf) -> Self {
        self.profile_page_gate_dir = Some(directory);
        self
    }

    pub fn with_isolated_server_namespace(mut self) -> Self {
        self.isolated_server_namespace = true;
        self
    }
}

#[cfg(target_os = "macos")]
#[async_trait(?Send)]
impl RemoteConnection for ProcessRemoteConnection {
    fn doctor_server_binary(&self) -> Option<String> {
        Some(self.binary.display().to_string())
    }

    fn start_proxy(
        &self,
        unique_identifier: String,
        mode: crate::ProxyMode,
        incoming_tx: mpsc::UnboundedSender<Envelope>,
        outgoing_rx: mpsc::UnboundedReceiver<Envelope>,
        connection_activity_tx: Sender<()>,
        _delegate: Arc<dyn RemoteClientDelegate>,
        cx: &mut AsyncApp,
    ) -> Task<Result<i32>> {
        let unique_identifier = if self.isolated_server_namespace {
            let prepare_namespace = || -> Result<String> {
                // Test-support paths use a synthetic home, unlike the actual host executable.
                let native_state_dir = if let Some(directory) = &self.test_data_directory {
                    anyhow::ensure!(
                        directory.is_absolute()
                            && directory.is_dir()
                            && directory.canonicalize()? == *directory,
                        "process fixture data directory must be an existing canonical owned directory"
                    );
                    let state_directory = directory.join("server_state");
                    if !state_directory.exists() {
                        std::fs::create_dir(&state_directory)?;
                    }
                    state_directory
                } else {
                    PathBuf::from(
                        std::env::var_os("HOME")
                            .context("process fixture requires the native HOME")?,
                    )
                    .join("Library/Application Support")
                    .join(paths::APP_NAME_LOWERCASE)
                    .join("server_state")
                };
                anyhow::ensure!(
                    self.server_state_dir == native_state_dir,
                    "process fixture state directory must match the server's default: {:?}",
                    native_state_dir
                );
                let connection_id = u32::try_from(self.options.id)
                    .context("process fixture connection counter exhausted")?;
                // Separate GPUI seed processes can otherwise reuse the same setup counter.
                let identifier = format!("t{:x}-{:x}", std::process::id(), connection_id);
                if self.identifier.lock().is_none() {
                    std::fs::create_dir(self.server_state_dir.join(&identifier))
                        .context("reserve a fresh process fixture server namespace")?;
                }
                Ok(identifier)
            };
            match prepare_namespace() {
                Ok(identifier) => identifier,
                Err(error) => return Task::ready(Err(error)),
            }
        } else {
            unique_identifier
        };
        *self.identifier.lock() = Some(unique_identifier.clone());
        let mut command = new_command(&self.binary);
        if let Some(directory) = &self.test_data_directory {
            command.arg("--test-data-dir").arg(directory);
        }
        command
            .arg("proxy")
            .arg("--identifier")
            .arg(&unique_identifier)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if self.isolated_server_namespace {
            // Confine the server's relative binary-cache cleanup to this owned namespace.
            command.current_dir(self.server_state_dir.join(&unique_identifier));
        }
        if let Some(directory) = &self.profile_page_gate_dir {
            command.env("ZED_TEST_PROFILE_PAGE_GATE_DIR", directory);
        }
        command.args(mode.arguments());
        match command.spawn() {
            Ok(process) => super::handle_rpc_messages_over_child_process_stdio(
                process,
                incoming_tx,
                outgoing_rx,
                connection_activity_tx,
                cx,
            ),
            Err(error) => Task::ready(Err(error.into())),
        }
    }

    fn upload_directory(
        &self,
        _src_path: PathBuf,
        _dest_path: RemotePathBuf,
        _cx: &App,
    ) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!(
            "process test connection cannot upload a directory"
        )))
    }

    async fn kill(&self) -> Result<()> {
        self.killed.store(true, Ordering::SeqCst);
        let Some(identifier) = self.identifier.lock().take() else {
            return Ok(());
        };
        let server_dir = self.server_state_dir.join(&identifier);
        let pid_path = server_dir.join("server.pid");
        let process_id: i32 = std::fs::read_to_string(&pid_path)?.parse()?;
        if unsafe { libc::kill(process_id, libc::SIGTERM) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error.into());
            }
        }
        std::fs::remove_dir_all(&server_dir)?;
        Ok(())
    }

    fn has_been_killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst)
    }

    fn build_command(
        &self,
        program: Option<String>,
        args: &[String],
        env: &HashMap<String, String>,
        _working_dir: Option<String>,
        _port_forward: Option<(u16, String, u16)>,
        _interactive: Interactive,
    ) -> Result<CommandTemplate> {
        Ok(CommandTemplate {
            program: program.unwrap_or_else(|| "sh".to_string()),
            args: args.to_vec(),
            env: env.clone(),
        })
    }

    fn build_forward_ports_command(
        &self,
        _forwards: Vec<(u16, String, u16)>,
    ) -> Result<CommandTemplate> {
        Err(anyhow::anyhow!(
            "process test connection cannot forward ports"
        ))
    }

    fn connection_options(&self) -> RemoteConnectionOptions {
        RemoteConnectionOptions::Mock(self.options.clone())
    }

    fn path_style(&self) -> PathStyle {
        PathStyle::local()
    }

    fn remote_platform(&self) -> crate::RemotePlatform {
        crate::RemotePlatform {
            os: crate::RemoteOs::MacOs,
            arch: if cfg!(target_arch = "aarch64") {
                crate::RemoteArch::Aarch64
            } else {
                crate::RemoteArch::X86_64
            },
        }
    }

    fn remote_os_version(&self) -> Option<String> {
        None
    }

    fn shell(&self) -> String {
        "sh".to_string()
    }

    fn default_system_shell(&self) -> String {
        "sh".to_string()
    }

    fn has_wsl_interop(&self) -> bool {
        false
    }
}

/// Mock delegate for tests that don't need delegate functionality.
pub struct MockDelegate;

impl RemoteClientDelegate for MockDelegate {
    fn ask_password(
        &self,
        _prompt: String,
        _sender: futures::channel::oneshot::Sender<askpass::EncryptedPassword>,
        _cx: &mut AsyncApp,
    ) {
        unreachable!("MockDelegate::ask_password should not be called in tests")
    }

    fn download_server_binary_locally(
        &self,
        _platform: crate::RemotePlatform,
        _release_channel: release_channel::ReleaseChannel,
        _version: Option<semver::Version>,
        _cx: &mut AsyncApp,
    ) -> Task<Result<PathBuf>> {
        unreachable!("MockDelegate::download_server_binary_locally should not be called in tests")
    }

    fn get_download_url(
        &self,
        _platform: crate::RemotePlatform,
        _release_channel: release_channel::ReleaseChannel,
        _version: Option<semver::Version>,
        _cx: &mut AsyncApp,
    ) -> Task<Result<Option<String>>> {
        unreachable!("MockDelegate::get_download_url should not be called in tests")
    }

    fn set_status(&self, _status: Option<&str>, _cx: &mut AsyncApp) {}
}
