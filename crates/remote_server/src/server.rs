mod headless_project;

#[cfg(test)]
mod remote_editing_tests;

#[cfg(windows)]
pub mod windows;

pub use headless_project::{HeadlessAppState, HeadlessProject};

use anyhow::{Context as _, Result, anyhow};
use clap::Subcommand;
use client::ProxySettings;
use collections::HashMap;
use extension::ExtensionHostProxy;
use fs::{Fs, RealFs};
use futures::{
    AsyncRead, AsyncWrite, AsyncWriteExt, FutureExt, SinkExt,
    channel::{mpsc, oneshot},
    select, select_biased,
};
use git::GitHostingProviderRegistry;
use gpui::{App, AppContext as _, Context, Entity, TaskExt as _, UpdateGlobal as _};
use gpui_tokio::Tokio;
use http_client::{Url, read_proxy_from_env};
use language::LanguageRegistry;
use net::async_net::{UnixListener, UnixStream};
use node_runtime::{NodeBinaryOptions, NodeRuntime};
use paths::logs_dir;
use project::{project_settings::ProjectSettings, trusted_worktrees};
use proto::CrashReport;
use release_channel::{AppCommitSha, AppVersion, RELEASE_CHANNEL, ReleaseChannel};
use remote::{
    RemoteClient,
    json_log::LogRecord,
    protocol::{read_message, write_message},
    proxy::ProxyLaunchError,
};
use reqwest_client::ReqwestClient;
use rpc::proto::{self, Envelope, REMOTE_SERVER_PROJECT_ID};
use rpc::{AnyProtoClient, TypedEnvelope};
use settings::{Settings, SettingsStore, watch_config_file};
use smol::{
    Timer,
    channel::{Receiver, Sender},
    io::AsyncReadExt,
    stream::StreamExt as _,
};
use std::{
    ffi::OsStr,
    fs::File,
    io::Write,
    mem,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, LazyLock},
    time::Instant,
};
use thiserror::Error;
use util::{ResultExt, command::new_command};

#[derive(Subcommand)]
pub enum Commands {
    #[cfg(target_os = "macos")]
    #[command(hide = true)]
    ObserveTask {
        #[arg(long)]
        socket: PathBuf,
        #[arg(long, hide = true, conflicts_with_all = ["native_profile_capture", "capture_build_stdout"])]
        capture_stdout: bool,
        #[arg(long, hide = true, conflicts_with_all = ["capture_stdout", "capture_build_stdout"])]
        native_profile_capture: bool,
        #[arg(long, hide = true, conflicts_with_all = ["capture_stdout", "native_profile_capture"])]
        capture_build_stdout: bool,
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<std::ffi::OsString>,
    },
    Run {
        #[arg(long, hide = true)]
        isolated_recipe_test: bool,
        #[arg(long, hide = true)]
        exclusive_client: bool,
        #[arg(long)]
        log_file: PathBuf,
        #[arg(long)]
        pid_file: PathBuf,
        #[arg(long)]
        stdin_socket: PathBuf,
        #[arg(long)]
        stdout_socket: PathBuf,
        #[arg(long)]
        stderr_socket: PathBuf,
    },
    Proxy {
        #[arg(long, hide = true)]
        isolated_recipe_test: bool,
        #[arg(long)]
        reconnect: bool,
        #[arg(long, hide = true)]
        start_if_stopped: Option<String>,
        #[arg(long)]
        identifier: String,
    },
    #[command(hide = true)]
    DoctorServerStopped {
        #[arg(long)]
        identifier: String,
        #[arg(long)]
        instance: String,
        #[arg(long)]
        process_id: u32,
    },
    Version {
        #[arg(long, hide = true)]
        isolated_recipe_test: bool,
    },
}

pub fn run(command: Commands) -> anyhow::Result<()> {
    use anyhow::Context;
    use release_channel::{RELEASE_CHANNEL, ReleaseChannel};

    match command {
        #[cfg(target_os = "macos")]
        Commands::ObserveTask {
            socket,
            capture_stdout,
            native_profile_capture,
            capture_build_stdout,
            command,
        } => {
            anyhow::ensure!(
                usize::from(capture_stdout)
                    + usize::from(native_profile_capture)
                    + usize::from(capture_build_stdout)
                    <= 1,
                "Ambiguous Task helper mode"
            );
            let (program, arguments) = command.split_first().context("Missing Task command")?;
            let mut command = std::process::Command::new(program);
            command.args(arguments);
            if capture_stdout {
                let status =
                    util::macos_task_completion::register_and_capture_stdout(&socket, command)?;
                util::macos_task_completion::exit_with_observed_task_status(status);
            }
            if capture_build_stdout {
                let status = util::macos_task_completion::register_and_capture_build_stdout(
                    &socket, command,
                )?;
                util::macos_task_completion::exit_with_observed_task_status(status);
            }
            let error = if native_profile_capture {
                util::macos_task_completion::register_and_exec_native_profile_capture(
                    &socket,
                    &mut command,
                )
            } else {
                util::macos_task_completion::register_and_exec(&socket, &mut command)
            };
            Err(error.into())
        }
        Commands::Run {
            isolated_recipe_test,
            exclusive_client,
            log_file,
            pid_file,
            stdin_socket,
            stdout_socket,
            stderr_socket,
        } => execute_run_with_ownership(
            isolated_recipe_test,
            exclusive_client,
            log_file,
            pid_file,
            stdin_socket,
            stdout_socket,
            stderr_socket,
        ),
        Commands::Proxy {
            identifier,
            reconnect,
            isolated_recipe_test,
            start_if_stopped,
        } => execute_proxy_mode(
            identifier,
            reconnect,
            isolated_recipe_test,
            start_if_stopped,
        )
        .context("running proxy on the remote server"),
        Commands::DoctorServerStopped {
            identifier,
            instance,
            process_id,
        } => {
            let paths = ServerPaths::existing(&identifier)?;
            let _lease = acquire_proxy_lease(&paths, false)?;
            println!(
                "{}",
                if doctor_server_stopped(&paths, &instance, Some(process_id))? {
                    "stopped"
                } else {
                    "running"
                }
            );
            Ok(())
        }
        Commands::Version {
            isolated_recipe_test,
        } => {
            if isolated_recipe_test {
                anyhow::ensure!(
                    cfg!(target_os = "linux"),
                    "Isolated recipe runtimes require Linux"
                );
                println!("made-recipe-isolation-v1 {}", VERSION.as_str());
                return Ok(());
            }
            let release_channel = *RELEASE_CHANNEL;
            match release_channel {
                ReleaseChannel::Stable | ReleaseChannel::Preview => {
                    println!("{}", env!("ZED_PKG_VERSION"))
                }
                ReleaseChannel::Nightly | ReleaseChannel::Dev => {
                    let commit_sha =
                        option_env!("ZED_COMMIT_SHA").unwrap_or(release_channel.dev_name());
                    let build_id = option_env!("ZED_BUILD_ID");
                    if let Some(build_id) = build_id {
                        println!("{}+{}", build_id, commit_sha)
                    } else {
                        println!("{commit_sha}");
                    }
                }
            };
            Ok(())
        }
    }
}

pub static VERSION: LazyLock<String> = LazyLock::new(|| match *RELEASE_CHANNEL {
    ReleaseChannel::Stable | ReleaseChannel::Preview => env!("ZED_PKG_VERSION").to_owned(),
    ReleaseChannel::Nightly | ReleaseChannel::Dev => {
        let commit_sha = option_env!("ZED_COMMIT_SHA").unwrap_or("missing-zed-commit-sha");
        let build_identifier = option_env!("ZED_BUILD_ID");
        if let Some(build_id) = build_identifier {
            format!("{build_id}+{commit_sha}")
        } else {
            commit_sha.to_owned()
        }
    }
});

fn init_logging_proxy() {
    env_logger::builder()
        .format(|buf, record| {
            let mut log_record = LogRecord::new(record);
            log_record.message =
                std::borrow::Cow::Owned(format!("(remote proxy) {}", log_record.message));
            serde_json::to_writer(&mut *buf, &log_record)?;
            buf.write_all(b"\n")?;
            Ok(())
        })
        .init();
}

const REMOTE_SERVER_LOG_MAX_BYTES: u64 = 1024 * 1024;

struct RotatingLogFile {
    path: PathBuf,
    file: File,
    size_bytes: u64,
}

impl RotatingLogFile {
    fn open(path: &Path) -> Result<Self> {
        if std::fs::metadata(path)
            .map(|metadata| metadata.len() >= REMOTE_SERVER_LOG_MAX_BYTES)
            .unwrap_or(false)
        {
            rotate_log_file(path, &rotated_log_path(path))
                .context("failed to rotate existing remote server log")?;
        }

        let file = open_log_file(path).context("failed to open remote server log")?;
        let size_bytes = file
            .metadata()
            .context("failed to read remote server log metadata")?
            .len();

        Ok(Self {
            path: path.to_path_buf(),
            file,
            size_bytes,
        })
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        rotate_log_file(&self.path, &rotated_log_path(&self.path))?;
        self.file = open_log_file(&self.path)?;
        self.size_bytes = 0;
        Ok(())
    }
}

impl Write for RotatingLogFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.size_bytes.saturating_add(buf.len() as u64) > REMOTE_SERVER_LOG_MAX_BYTES {
            self.rotate()?;
        }

        self.file.write_all(buf)?;
        self.size_bytes += buf.len() as u64;

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

fn open_log_file(path: &Path) -> std::io::Result<File> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
}

fn rotated_log_path(path: &Path) -> PathBuf {
    path.with_extension("1.log")
}

fn rotate_log_file(path: &Path, rotated_path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(rotated_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    match std::fs::rename(path, rotated_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }

    Ok(())
}

fn init_logging_server(log_file_path: &Path) -> Result<Receiver<Vec<u8>>> {
    struct MultiWrite {
        file: RotatingLogFile,
        channel: Sender<Vec<u8>>,
        buffer: Vec<u8>,
    }

    impl Write for MultiWrite {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.file.write_all(buf)?;
            self.buffer.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.channel
                .send_blocking(self.buffer.clone())
                .map_err(std::io::Error::other)?;
            self.buffer.clear();
            self.file.flush()
        }
    }

    let log_file = RotatingLogFile::open(log_file_path)
        .context("Failed to open rotating remote server log file")?;

    let (tx, rx) = async_channel::unbounded();

    let target = Box::new(MultiWrite {
        file: log_file,
        channel: tx,
        buffer: Vec::new(),
    });

    let old_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let message = info.payload_as_str().unwrap_or("Box<Any>").to_owned();
        let location = info
            .location()
            .map_or_else(|| "<unknown>".to_owned(), |location| location.to_string());
        let current_thread = std::thread::current();
        let thread_name = current_thread.name().unwrap_or("<unnamed>");

        let msg = format!("thread '{thread_name}' panicked at {location}:\n{message}\n{backtrace}");
        // NOTE: This log never reaches the client, as the communication is handled on a main thread task
        // which will never run once we panic.
        log::error!("{msg}");
        old_hook(info);
    }));
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .target(env_logger::Target::Pipe(target))
        .format(|buf, record| {
            let mut log_record = LogRecord::new(record);
            log_record.message =
                std::borrow::Cow::Owned(format!("(remote server) {}", log_record.message));
            serde_json::to_writer(&mut *buf, &log_record)?;
            buf.write_all(b"\n")?;
            Ok(())
        })
        .init();

    Ok(rx)
}

/// Initializes the telemetry queue on the remote server, forwarding every
/// emitted event to the connected client over the proto channel.
///
/// The remote server cannot upload telemetry itself (it has no logged-in user,
/// no checksum seed, and no Telemetry instance), so without this its
/// `telemetry::event!` calls are silently dropped. The client attributes these
/// events to the remote host using the platform it already detected during
/// connection setup, so no OS metadata needs to be sent here.
fn init_telemetry_forwarding(session: AnyProtoClient, cx: &mut App) {
    let (tx, mut rx) = mpsc::unbounded::<telemetry::Event>();
    telemetry::init(tx);

    cx.background_spawn(async move {
        while let Some(event) = rx.next().await {
            let Some(event_json) = serde_json::to_string(&event).log_err() else {
                continue;
            };
            session
                .send(proto::TelemetryEvent {
                    project_id: REMOTE_SERVER_PROJECT_ID,
                    event_json,
                })
                .log_err();
        }
    })
    .detach();
}

fn handle_crash_files_requests(project: &Entity<HeadlessProject>, client: &AnyProtoClient) {
    client.add_request_handler(
        project.downgrade(),
        |_, _: TypedEnvelope<proto::GetCrashFiles>, _cx| async move {
            let mut legacy_panics = Vec::new();
            let mut crashes = Vec::new();
            let mut children = smol::fs::read_dir(paths::logs_dir()).await?;
            while let Some(child) = children.next().await {
                let child = child?;
                let child_path = child.path();

                let extension = child_path.extension();
                if extension == Some(OsStr::new("panic")) {
                    let filename = if let Some(filename) = child_path.file_name() {
                        filename.to_string_lossy()
                    } else {
                        continue;
                    };

                    if !filename.starts_with("zed") {
                        continue;
                    }

                    let file_contents = smol::fs::read_to_string(&child_path)
                        .await
                        .context("error reading panic file")?;

                    legacy_panics.push(file_contents);
                    smol::fs::remove_file(&child_path)
                        .await
                        .context("error removing panic")
                        .log_err();
                } else if extension == Some(OsStr::new("dmp")) {
                    let mut json_path = child_path.clone();
                    json_path.set_extension("json");
                    if let Ok(json_content) = smol::fs::read_to_string(&json_path).await {
                        crashes.push(CrashReport {
                            metadata: json_content,
                            minidump_contents: smol::fs::read(&child_path).await?,
                        });
                        smol::fs::remove_file(&child_path).await.log_err();
                        smol::fs::remove_file(&json_path).await.log_err();
                    } else {
                        log::error!("Couldn't find json metadata for crash: {child_path:?}");
                    }
                }
            }

            anyhow::Ok(proto::GetCrashFilesResponse { crashes })
        },
    );
}

struct ServerListeners {
    stdin: UnixListener,
    stdout: UnixListener,
    stderr: UnixListener,
}

impl ServerListeners {
    pub fn new(stdin_path: PathBuf, stdout_path: PathBuf, stderr_path: PathBuf) -> Result<Self> {
        Ok(Self {
            stdin: UnixListener::bind(stdin_path).context("failed to bind stdin socket")?,
            stdout: UnixListener::bind(stdout_path).context("failed to bind stdout socket")?,
            stderr: UnixListener::bind(stderr_path).context("failed to bind stderr socket")?,
        })
    }
}

#[derive(Default)]
struct NativeProjectTransport {
    last_connection_id: u32,
    connection_id: Option<proto::PeerId>,
}

impl NativeProjectTransport {
    fn accept(&mut self, cx: &mut Context<Self>) -> Result<proto::PeerId> {
        anyhow::ensure!(
            self.connection_id.is_none(),
            "Native project connection is already active"
        );
        let id = self
            .last_connection_id
            .checked_add(1)
            .context("Native project connection identity exhausted; restart the project host")?;
        let connection_id = proto::PeerId { owner_id: 0, id };
        self.last_connection_id = id;
        self.connection_id = Some(connection_id);
        cx.emit(true);
        Ok(connection_id)
    }

    fn disconnect(&mut self, cx: &mut Context<Self>) {
        self.connection_id = None;
        cx.emit(false);
    }
}

impl gpui::EventEmitter<bool> for NativeProjectTransport {}

fn start_server(
    listeners: ServerListeners,
    log_rx: Receiver<Vec<u8>>,
    cx: &mut App,
    is_wsl_interop: bool,
    project_transport: Entity<NativeProjectTransport>,
) -> (AnyProtoClient, remote::RemoteRequestActivity) {
    // This is the server idle timeout. If no connection comes in this timeout, the server will shut down.
    const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

    let (incoming_tx, incoming_rx) = mpsc::unbounded::<(proto::PeerId, Envelope)>();
    let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded::<Envelope>();
    let (app_quit_tx, mut app_quit_rx) = mpsc::unbounded::<()>();

    let quit_transport = project_transport.clone();
    cx.on_app_quit(move |cx| {
        quit_transport.update(cx, |transport, cx| transport.disconnect(cx));
        let mut app_quit_tx = app_quit_tx.clone();
        async move {
            log::info!("app quitting. sending signal to server main loop");
            app_quit_tx.send(()).await.ok();
        }
    })
    .detach();

    cx.spawn(async move |cx| {
        loop {
            let streams = futures::future::join3(
                listeners.stdin.accept(),
                listeners.stdout.accept(),
                listeners.stderr.accept(),
            );

            log::info!("accepting new connections");
            let result = select! {
                streams = streams.fuse() => {
                    let (Ok((stdin_stream, _)), Ok((stdout_stream, _)), Ok((stderr_stream, _))) = streams else {
                        log::error!("failed to accept new connections");
                        break;
                    };
                    log::info!("accepted new connections");
                    anyhow::Ok((stdin_stream, stdout_stream, stderr_stream))
                }
                _ = futures::FutureExt::fuse(cx.background_executor().timer(IDLE_TIMEOUT)) => {
                    log::warn!("timed out waiting for new connections after {:?}. exiting.", IDLE_TIMEOUT);
                    cx.update(|cx| {
                        // TODO: This is a hack, because in a headless project, shutdown isn't executed
                        // when calling quit, but it should be.
                        cx.shutdown();
                        cx.quit();
                    });
                    break;
                }
                _ = app_quit_rx.next().fuse() => {
                    log::info!("app quit requested");
                    break;
                }
            };

            let Ok((stdin_stream, stdout_stream, stderr_stream)) = result else {
                break;
            };

            let connection_id = project_transport.update(cx, |transport, cx| transport.accept(cx))?;
            let ended = handle_native_connection(
                connection_id,
                stdin_stream,
                stdout_stream,
                stderr_stream,
                &incoming_tx,
                &mut outgoing_rx,
                &mut app_quit_rx,
                &log_rx,
                cx,
            ).await;
            project_transport.update(cx, |transport, cx| transport.disconnect(cx));
            match ended? {
                NativeConnectionEnd::Quit => return anyhow::Ok(()),
                NativeConnectionEnd::Disconnected => {},
            }
        }
        anyhow::Ok(())
    })
    .detach_and_log_err(cx);

    RemoteClient::proto_client_from_scoped_channels_with_activity(
        incoming_rx,
        outgoing_tx,
        cx,
        is_wsl_interop,
    )
}

#[derive(Debug, Eq, PartialEq)]
enum NativeConnectionEnd {
    Disconnected,
    Quit,
}

async fn handle_native_connection(
    connection_id: proto::PeerId,
    mut stdin_stream: impl AsyncRead + Unpin + Send + 'static,
    mut stdout_stream: impl AsyncWrite + Unpin,
    mut stderr_stream: impl AsyncWrite + Unpin,
    incoming_tx: &mpsc::UnboundedSender<(proto::PeerId, Envelope)>,
    outgoing_rx: &mut mpsc::UnboundedReceiver<Envelope>,
    app_quit_rx: &mut mpsc::UnboundedReceiver<()>,
    log_rx: &Receiver<Vec<u8>>,
    cx: &mut gpui::AsyncApp,
) -> Result<NativeConnectionEnd> {
    let mut input_buffer = Vec::new();
    let mut output_buffer = Vec::new();

    let (mut stdin_msg_tx, mut stdin_msg_rx) = mpsc::unbounded::<(proto::PeerId, Envelope)>();
    let _input_reader = cx.background_spawn(async move {
        loop {
            match read_message(&mut stdin_stream, &mut input_buffer).await {
                Ok(msg) => {
                    if (stdin_msg_tx.send((connection_id, msg)).await).is_err() {
                        log::info!("stdin message channel closed, stopping stdin reader");
                        break;
                    }
                }
                Err(error) => {
                    log::warn!("stdin read failed: {error:?}");
                    break;
                }
            }
        }
    });

    loop {
        select_biased! {
            _ = app_quit_rx.next().fuse() => {
                return Ok(NativeConnectionEnd::Quit);
            }

            stdin_message = stdin_msg_rx.next().fuse() => {
                let Some(message) = stdin_message else {
                    log::warn!("error reading message on stdin, dropping connection.");
                    break;
                };
                if let Err(error) = incoming_tx.unbounded_send(message) {
                    log::error!("failed to send message to application: {error:?}. exiting.");
                    return Err(anyhow!(error));
                }
            }

            outgoing_message  = outgoing_rx.next().fuse() => {
                let Some(message) = outgoing_message else {
                    log::error!("stdout handler, no message");
                    break;
                };

                if let Err(error) =
                    write_message(&mut stdout_stream, &mut output_buffer, message).await
                {
                    log::error!("failed to write stdout message: {:?}", error);
                    break;
                }
                if let Err(error) = stdout_stream.flush().await {
                    log::error!("failed to flush stdout message: {:?}", error);
                    break;
                }
            }

            log_message = log_rx.recv().fuse() => {
                if let Ok(log_message) = log_message {
                    if let Err(error) = stderr_stream.write_all(&log_message).await {
                        log::error!("failed to write log message to stderr: {:?}", error);
                        break;
                    }
                    if let Err(error) = stderr_stream.flush().await {
                        log::error!("failed to flush stderr stream: {:?}", error);
                        break;
                    }
                }
            }
        }
    }
    Ok(NativeConnectionEnd::Disconnected)
}

fn init_paths() -> anyhow::Result<()> {
    for path in [
        paths::config_dir(),
        paths::extensions_dir(),
        paths::languages_dir(),
        paths::logs_dir(),
        paths::temp_dir(),
        paths::hang_traces_dir(),
        paths::remote_extensions_dir(),
        paths::remote_extensions_uploads_dir(),
    ]
    .iter()
    {
        std::fs::create_dir_all(path).with_context(|| format!("creating directory {path:?}"))?;
    }
    Ok(())
}

pub fn execute_run(
    isolated_recipe_test: bool,
    log_file: PathBuf,
    pid_file: PathBuf,
    stdin_socket: PathBuf,
    stdout_socket: PathBuf,
    stderr_socket: PathBuf,
) -> Result<()> {
    execute_run_with_ownership(
        isolated_recipe_test,
        false,
        log_file,
        pid_file,
        stdin_socket,
        stdout_socket,
        stderr_socket,
    )
}

fn execute_run_with_ownership(
    isolated_recipe_test: bool,
    exclusive_client: bool,
    log_file: PathBuf,
    pid_file: PathBuf,
    stdin_socket: PathBuf,
    stdout_socket: PathBuf,
    stderr_socket: PathBuf,
) -> Result<()> {
    anyhow::ensure!(
        !isolated_recipe_test || cfg!(target_os = "linux"),
        "Isolated recipe tests require a Linux container"
    );
    init_paths()?;

    let startup_time = Instant::now();
    let app = gpui_platform::headless();
    let pid = std::process::id();
    let id = pid.to_string();
    let should_install_crash_handler =
        !isolated_recipe_test && client::telemetry::should_install_crash_handler(*RELEASE_CHANNEL);

    let crash_handler = if should_install_crash_handler {
        Some(app.background_executor().spawn(crashes::init(
            crashes::InitCrashHandler {
                session_id: id,
                zed_version: VERSION.to_owned(),
                binary: "zed-remote-server".to_string(),
                release_channel: release_channel::RELEASE_CHANNEL_NAME.clone(),
                commit_sha: option_env!("ZED_COMMIT_SHA").unwrap_or("no_sha").to_owned(),
            },
            {
                let background_executor = app.background_executor();
                move |task| {
                    background_executor.spawn(task).detach();
                }
            },
            |pid| paths::temp_dir().join(format!("zed-remote-server-crash-handler-{pid}")),
            // we are running outside gpui
            #[allow(clippy::disallowed_methods)]
            |duration| FutureExt::map(Timer::after(duration), |_| ()),
        )))
    } else {
        crashes::force_backtrace();
        None
    };
    let log_rx = init_logging_server(&log_file)?;
    log::info!(
        "starting up with PID {}:\npid_file: {:?}, log_file: {:?}, stdin_socket: {:?}, stdout_socket: {:?}, stderr_socket: {:?}",
        pid,
        pid_file,
        log_file,
        stdin_socket,
        stdout_socket,
        stderr_socket
    );

    write_pid_file(&pid_file, pid)
        .with_context(|| format!("failed to write pid file: {pid_file:?}"))?;

    let instance = uuid::Uuid::new_v4().to_string();
    let instance_path = pid_file.with_file_name("server.identity");
    write_server_identity(&instance_path, &instance)?;
    let listeners = ServerListeners::new(stdin_socket, stdout_socket, stderr_socket)?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().div_ceil(2)))
        .stack_size(10 * 1024 * 1024)
        .thread_name(|ix| format!("RayonWorker{}", ix))
        .build_global()
        .unwrap();

    #[cfg(unix)]
    let shell_env_loaded_rx = if isolated_recipe_test {
        None
    } else {
        let (shell_env_loaded_tx, shell_env_loaded_rx) = oneshot::channel();
        app.background_executor()
            .spawn(async {
                util::load_login_shell_environment().await.log_err();
                shell_env_loaded_tx.send(()).ok();
            })
            .detach();
        Some(shell_env_loaded_rx)
    };
    #[cfg(windows)]
    let shell_env_loaded_rx: Option<oneshot::Receiver<()>> = None;

    let git_hosting_provider_registry = Arc::new(GitHostingProviderRegistry::new());
    let run = move |cx: &mut App| {
        if let Some(crash_handler) = crash_handler {
            cx.spawn(async move |_cx| {
                let _crash_handler = crash_handler.await;
                // cx.update(|cx| cx.set_global(CrashHandler(crash_handler)))
            })
            .detach();
        }
        settings::init(cx);
        let app_commit_sha = option_env!("ZED_COMMIT_SHA").map(|s| AppCommitSha::new(s.to_owned()));
        let app_version = AppVersion::load(
            env!("ZED_PKG_VERSION"),
            option_env!("ZED_BUILD_ID"),
            app_commit_sha,
        );
        release_channel::init(app_version, cx);
        gpui_tokio::init(cx);

        HeadlessProject::init(cx);

        let is_wsl_interop = if cfg!(target_os = "linux") {
            // See: https://learn.microsoft.com/en-us/windows/wsl/filesystems#disable-interoperability
            matches!(std::fs::read_to_string("/proc/sys/fs/binfmt_misc/WSLInterop").or_else(|_| std::fs::read_to_string("/proc/sys/fs/binfmt_misc/WSLInterop-late")), Ok(s) if s.contains("enabled"))
        } else {
            false
        };

        log::info!("gpui app started, initializing server");
        let project_transport = cx.new(|_| NativeProjectTransport::default());
        #[cfg(feature = "rust-tools")]
        let profile_import_transport = project_transport.clone();
        #[cfg(feature = "jvm-tools")]
        let jvm_project_transport = project_transport.clone();
        let (session, request_activity) =
            start_server(listeners, log_rx, cx, is_wsl_interop, project_transport);
        if !isolated_recipe_test {
            init_telemetry_forwarding(session.clone(), cx);
        }
        trusted_worktrees::init(HashMap::default(), cx);

        GitHostingProviderRegistry::set_global(git_hosting_provider_registry, cx);
        git_hosting_providers::init(cx);
        dap_adapters::init(cx);

        extension::init(cx);
        let extension_host_proxy = ExtensionHostProxy::global(cx);

        json_schema_store::init(cx);

        let project = cx.new(|cx| {
            let fs = Arc::new(RealFs::new(None, cx.background_executor().clone()));
            let (http_client, node_runtime): (Arc<dyn http_client::HttpClient>, NodeRuntime) =
                if isolated_recipe_test {
                    (
                        Arc::new(http_client::BlockedHttpClient),
                        NodeRuntime::unavailable(),
                    )
                } else {
                    let node_settings_rx = initialize_settings(session.clone(), fs.clone(), cx);

                    let proxy_url = read_proxy_settings(cx);

                    let http_client = {
                        let _guard = Tokio::handle(cx).enter();
                        Arc::new(
                            ReqwestClient::proxy_and_user_agent(
                                proxy_url,
                                &format!(
                                    "Zed-Server/{} ({}; {})",
                                    env!("CARGO_PKG_VERSION"),
                                    std::env::consts::OS,
                                    std::env::consts::ARCH
                                ),
                            )
                            .expect("Could not start HTTP client"),
                        )
                    };

                    let node_runtime = NodeRuntime::new(
                        http_client.clone(),
                        shell_env_loaded_rx,
                        node_settings_rx,
                    );

                    (http_client, node_runtime)
                };

            let mut languages = LanguageRegistry::new(cx.background_executor().clone());
            languages.set_language_server_download_dir(paths::languages_dir().clone());
            let languages = Arc::new(languages);

            let state = HeadlessAppState {
                session: session.clone(),
                fs,
                http_client,
                node_runtime,
                languages,
                extension_host_proxy,
                startup_time,
            };
            if isolated_recipe_test {
                HeadlessProject::isolated_recipe_test(state, cx)
            } else {
                HeadlessProject::new(state, true, cx)
            }
        });

        project.update(cx, |project, _| {
            project.configure_doctor_restart(
                instance,
                exclusive_client && !isolated_recipe_test,
                request_activity,
            )
        });

        #[cfg(feature = "rust-tools")]
        project.update(cx, |project, cx| {
            project.profile_results_store.update(cx, |store, cx| {
                store.observe_transport(&profile_import_transport, false, cx);
            });
        });
        #[cfg(feature = "jvm-tools")]
        if !isolated_recipe_test {
            project.update(cx, |project, cx| {
                project.observe_jvm_transport(&jvm_project_transport, cx);
            });
        }
        handle_crash_files_requests(&project, &session);

        cx.background_spawn(async move {
            cleanup_old_binaries_wsl();
            cleanup_old_binaries()
        })
        .detach();

        mem::forget(project);
    };
    // We do not reuse any of the state after unwinding, so we don't run risk of observing broken invariants.
    let app = std::panic::AssertUnwindSafe(app);
    let run = std::panic::AssertUnwindSafe(run);
    let res = std::panic::catch_unwind(move || { app }.0.run({ run }.0));
    if let Err(_) = res {
        log::error!("app panicked. quitting.");
        Err(anyhow::anyhow!("panicked"))
    } else {
        log::info!("gpui app is shut down. quitting.");
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ServerPathError {
    #[error("Invalid remote server identifier")]
    InvalidIdentifier,
    #[error("Failed to create server_dir `{path}`")]
    CreateServerDir {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },
    #[error("Failed to create logs_dir `{path}`")]
    CreateLogsDir {
        #[source]
        source: std::io::Error,
        path: PathBuf,
    },
}

#[derive(Clone, Debug)]
struct ServerPaths {
    log_file: PathBuf,
    pid_file: PathBuf,
    stdin_socket: PathBuf,
    stdout_socket: PathBuf,
    stderr_socket: PathBuf,
}

impl ServerPaths {
    fn new(identifier: &str) -> Result<Self, ServerPathError> {
        Self::paths(identifier, true)
    }

    fn existing(identifier: &str) -> Result<Self, ServerPathError> {
        Self::paths(identifier, false)
    }

    fn paths(identifier: &str, create: bool) -> Result<Self, ServerPathError> {
        if identifier.is_empty()
            || identifier.len() > 64
            || !identifier.bytes().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, b'-' | b'_')
            })
        {
            return Err(ServerPathError::InvalidIdentifier);
        }
        let server_dir = paths::remote_server_state_dir().join(identifier);
        if create {
            std::fs::create_dir_all(&server_dir).map_err(|source| {
                ServerPathError::CreateServerDir {
                    source,
                    path: server_dir.clone(),
                }
            })?;
        }
        let log_dir = logs_dir();
        if create {
            std::fs::create_dir_all(log_dir).map_err(|source| ServerPathError::CreateLogsDir {
                source,
                path: log_dir.clone(),
            })?;
        }

        let pid_file = server_dir.join("server.pid");
        let stdin_socket = server_dir.join("stdin.sock");
        let stdout_socket = server_dir.join("stdout.sock");
        let stderr_socket = server_dir.join("stderr.sock");
        let log_file = logs_dir().join(format!("server-{}.log", identifier));

        Ok(Self {
            pid_file,
            stdin_socket,
            stdout_socket,
            stderr_socket,
            log_file,
        })
    }
}

#[derive(Debug, Error)]
pub enum ExecuteProxyError {
    #[error("Remote server ownership/restart could not be established: {0:#}")]
    Ownership(#[source] anyhow::Error),
    #[error("Isolated recipe tests require a fresh Linux container session and cannot reconnect")]
    IsolatedRecipeTest,
    #[error("Failed to init server paths: {0:#}")]
    ServerPath(#[from] ServerPathError),

    #[error(transparent)]
    ServerNotRunning(#[from] ProxyLaunchError),

    #[error("Failed to check PidFile '{path}': {source:#}")]
    CheckPidFile {
        #[source]
        source: CheckPidError,
        path: PathBuf,
    },

    #[error("Failed to kill existing server with pid '{pid}'")]
    KillRunningServer { pid: u32 },

    #[error("failed to spawn server")]
    SpawnServer(#[source] SpawnServerError),

    #[error("stdin_task failed: {0:#}")]
    StdinTask(#[source] anyhow::Error),
    #[error("stdout_task failed: {0:#}")]
    StdoutTask(#[source] anyhow::Error),
    #[error("stderr_task failed: {0:#}")]
    StderrTask(#[source] anyhow::Error),
}

impl ExecuteProxyError {
    pub fn to_exit_code(&self) -> i32 {
        match self {
            ExecuteProxyError::ServerNotRunning(proxy_launch_error) => {
                proxy_launch_error.to_exit_code()
            }
            _ => 1,
        }
    }
}

#[cfg(test)]
pub(crate) fn execute_proxy(
    identifier: String,
    is_reconnecting: bool,
    isolated_recipe_test: bool,
) -> Result<(), ExecuteProxyError> {
    execute_proxy_mode(identifier, is_reconnecting, isolated_recipe_test, None)
}

fn execute_proxy_mode(
    identifier: String,
    is_reconnecting: bool,
    isolated_recipe_test: bool,
    start_if_stopped: Option<String>,
) -> Result<(), ExecuteProxyError> {
    if isolated_recipe_test && (is_reconnecting || !cfg!(target_os = "linux")) {
        return Err(ExecuteProxyError::IsolatedRecipeTest);
    }
    let server_paths = ServerPaths::new(&identifier)?;
    let _lease = acquire_proxy_lease(&server_paths, true).map_err(ExecuteProxyError::Ownership)?;
    if let Some(instance) = &start_if_stopped {
        if is_reconnecting
            || isolated_recipe_test
            || !doctor_server_stopped(&server_paths, instance, None)
                .map_err(ExecuteProxyError::Ownership)?
        {
            return Err(ExecuteProxyError::Ownership(anyhow!(
                "Reviewed server is still running or restart mode is unavailable"
            )));
        }
    }
    init_logging_proxy();

    let id = std::process::id().to_string();
    let should_install_crash_handler =
        !isolated_recipe_test && client::telemetry::should_install_crash_handler(*RELEASE_CHANNEL);

    if should_install_crash_handler {
        smol::spawn(crashes::init(
            crashes::InitCrashHandler {
                session_id: id,
                zed_version: VERSION.to_owned(),
                binary: "zed-remote-proxy".to_string(),
                release_channel: release_channel::RELEASE_CHANNEL_NAME.clone(),
                commit_sha: option_env!("ZED_COMMIT_SHA").unwrap_or("no_sha").to_owned(),
            },
            |task| {
                smol::spawn(task).detach();
            },
            |pid| paths::temp_dir().join(format!("zed-remote-server-proxy-crash-handler-{pid}")),
            // we are running outside gpui
            #[allow(clippy::disallowed_methods)]
            |duration| FutureExt::map(Timer::after(duration), |_| ()),
        ))
        .detach();
    };
    log::info!("starting proxy process. PID: {}", std::process::id());
    let server_pid = {
        let server_pid = check_pid_file(&server_paths.pid_file).map_err(|source| {
            ExecuteProxyError::CheckPidFile {
                source,
                path: server_paths.pid_file.clone(),
            }
        })?;
        if is_reconnecting {
            match server_pid {
                None => {
                    log::error!("attempted to reconnect, but no server running");
                    return Err(ExecuteProxyError::ServerNotRunning(
                        ProxyLaunchError::ServerNotRunning,
                    ));
                }
                Some(server_pid) => server_pid,
            }
        } else {
            if start_if_stopped.is_some() && server_pid.is_some() {
                return Err(ExecuteProxyError::Ownership(anyhow!(
                    "A remote server appeared after stop review; no process was killed"
                )));
            }
            if let Some(pid) = server_pid {
                log::info!(
                    "proxy found server already running with PID {}. Killing process and cleaning up files...",
                    pid
                );
                kill_running_server(pid, &server_paths)?;
            }
            gpui::block_on(spawn_server(&server_paths, isolated_recipe_test))
                .map_err(ExecuteProxyError::SpawnServer)?;
            std::fs::read_to_string(&server_paths.pid_file)
                .and_then(|contents| {
                    contents.parse::<u32>().map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid PID file contents",
                        )
                    })
                })
                .map_err(SpawnServerError::ProcessStatus)
                .map_err(ExecuteProxyError::SpawnServer)?
        }
    };

    let stdin_task = smol::spawn(async move {
        let stdin = smol::Unblock::new(std::io::stdin());
        let stream = UnixStream::connect(&server_paths.stdin_socket)
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to stdin socket {}",
                    server_paths.stdin_socket.display()
                )
            })?;
        handle_io(stdin, stream, "stdin").await
    });

    let stdout_task: smol::Task<Result<()>> = smol::spawn(async move {
        let stdout = smol::Unblock::new(std::io::stdout());
        let stream = UnixStream::connect(&server_paths.stdout_socket)
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to stdout socket {}",
                    server_paths.stdout_socket.display()
                )
            })?;
        handle_io(stream, stdout, "stdout").await
    });

    let stderr_task: smol::Task<Result<()>> = smol::spawn(async move {
        let mut stderr = smol::Unblock::new(std::io::stderr());
        let mut stream = UnixStream::connect(&server_paths.stderr_socket)
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to stderr socket {}",
                    server_paths.stderr_socket.display()
                )
            })?;
        let mut stderr_buffer = vec![0; 2048];
        loop {
            match stream
                .read(&mut stderr_buffer)
                .await
                .context("reading stderr")?
            {
                0 => {
                    let error =
                        std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "stderr closed");
                    Err(anyhow!(error))?;
                }
                n => {
                    stderr.write_all(&stderr_buffer[..n]).await?;
                    stderr.flush().await?;
                }
            }
        }
    });

    if let Err(forwarding_result) = gpui::block_on(async move {
        futures::select! {
            result = stdin_task.fuse() => result.map_err(ExecuteProxyError::StdinTask),
            result = stdout_task.fuse() => result.map_err(ExecuteProxyError::StdoutTask),
            result = stderr_task.fuse() => result.map_err(ExecuteProxyError::StderrTask),
        }
    }) {
        log::error!("encountered error while forwarding messages: {forwarding_result:#}",);
        if !matches!(gpui::block_on(check_server_running(server_pid)), Ok(true)) {
            log::error!("server exited unexpectedly");
            return Err(ExecuteProxyError::ServerNotRunning(
                ProxyLaunchError::ServerNotRunning,
            ));
        }
        return Err(forwarding_result);
    }

    Ok(())
}

fn kill_running_server(pid: u32, paths: &ServerPaths) -> Result<(), ExecuteProxyError> {
    log::info!("killing existing server with PID {}", pid);
    let system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_processes(sysinfo::ProcessRefreshKind::nothing()),
    );

    if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
        let killed = process.kill();
        if !killed {
            return Err(ExecuteProxyError::KillRunningServer { pid });
        }
    }

    for file in [
        &paths.pid_file,
        &paths.stdin_socket,
        &paths.stdout_socket,
        &paths.stderr_socket,
    ] {
        log::debug!("cleaning up file {:?} before starting new server", file);
        std::fs::remove_file(file).ok();
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum SpawnServerError {
    #[error("failed to remove stdin socket")]
    RemoveStdinSocket(#[source] std::io::Error),

    #[error("failed to remove stdout socket")]
    RemoveStdoutSocket(#[source] std::io::Error),

    #[error("failed to remove stderr socket")]
    RemoveStderrSocket(#[source] std::io::Error),

    #[error("failed to get current_exe")]
    CurrentExe(#[source] std::io::Error),

    #[error("failed to launch server process")]
    ProcessStatus(#[source] std::io::Error),

    #[error("failed to wait for server to be ready to accept connections")]
    Timeout,
}

async fn spawn_server(
    paths: &ServerPaths,
    isolated_recipe_test: bool,
) -> Result<(), SpawnServerError> {
    log::info!("spawning server process",);
    if paths.stdin_socket.exists() {
        std::fs::remove_file(&paths.stdin_socket).map_err(SpawnServerError::RemoveStdinSocket)?;
    }
    if paths.stdout_socket.exists() {
        std::fs::remove_file(&paths.stdout_socket).map_err(SpawnServerError::RemoveStdoutSocket)?;
    }
    if paths.stderr_socket.exists() {
        std::fs::remove_file(&paths.stderr_socket).map_err(SpawnServerError::RemoveStderrSocket)?;
    }

    let binary_name = std::env::current_exe().map_err(SpawnServerError::CurrentExe)?;

    #[cfg(windows)]
    {
        spawn_server_windows(&binary_name, paths)?;
    }

    #[cfg(not(windows))]
    {
        spawn_server_normal(&binary_name, paths, isolated_recipe_test)?;
    }

    let mut total_time_waited = std::time::Duration::from_secs(0);
    let wait_duration = std::time::Duration::from_millis(20);
    while !paths.stdout_socket.exists()
        || !paths.stdin_socket.exists()
        || !paths.stderr_socket.exists()
    {
        log::debug!("waiting for server to be ready to accept connections...");
        std::thread::sleep(wait_duration);
        total_time_waited += wait_duration;
        if total_time_waited > std::time::Duration::from_secs(10) {
            return Err(SpawnServerError::Timeout);
        }
    }

    log::info!(
        "server ready to accept connections. total time waited: {:?}",
        total_time_waited
    );

    Ok(())
}

#[cfg(windows)]
fn spawn_server_windows(binary_name: &Path, paths: &ServerPaths) -> Result<(), SpawnServerError> {
    let binary_path = binary_name.to_string_lossy().to_string();
    let parameters = format!(
        "run --exclusive-client --log-file \"{}\" --pid-file \"{}\" --stdin-socket \"{}\" --stdout-socket \"{}\" --stderr-socket \"{}\"",
        paths.log_file.to_string_lossy(),
        paths.pid_file.to_string_lossy(),
        paths.stdin_socket.to_string_lossy(),
        paths.stdout_socket.to_string_lossy(),
        paths.stderr_socket.to_string_lossy()
    );

    let directory = binary_name
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    crate::windows::shell_execute_from_explorer(&binary_path, &parameters, &directory)
        .map_err(|e| SpawnServerError::ProcessStatus(std::io::Error::other(e)))?;

    Ok(())
}

#[cfg(not(windows))]
fn spawn_server_normal(
    binary_name: &Path,
    paths: &ServerPaths,
    isolated_recipe_test: bool,
) -> Result<(), SpawnServerError> {
    let mut server_process = new_command(binary_name);
    #[cfg(feature = "test-support")]
    if let Some(directory) = paths::custom_data_dir() {
        server_process.arg("--test-data-dir").arg(directory);
    }
    server_process
        .stdin(util::command::Stdio::null())
        .stdout(util::command::Stdio::null())
        .stderr(util::command::Stdio::null())
        .arg("run")
        .arg("--exclusive-client")
        .arg("--log-file")
        .arg(&paths.log_file)
        .arg("--pid-file")
        .arg(&paths.pid_file)
        .arg("--stdin-socket")
        .arg(&paths.stdin_socket)
        .arg("--stdout-socket")
        .arg(&paths.stdout_socket)
        .arg("--stderr-socket")
        .arg(&paths.stderr_socket);

    if isolated_recipe_test {
        server_process.arg("--isolated-recipe-test");
    }

    server_process
        .spawn()
        .map_err(SpawnServerError::ProcessStatus)?;

    Ok(())
}

#[derive(Debug, Error)]
#[error("Failed to remove PID file for missing process (pid `{pid}`")]
pub struct CheckPidError {
    #[source]
    source: std::io::Error,
    pid: u32,
}
async fn check_server_running(pid: u32) -> std::io::Result<bool> {
    new_command("kill")
        .arg("-0")
        .arg(pid.to_string())
        .output()
        .await
        .map(|output| output.status.success())
}

fn check_pid_file(path: &Path) -> Result<Option<u32>, CheckPidError> {
    let Some(pid) = std::fs::read_to_string(path)
        .ok()
        .and_then(|contents| contents.parse::<u32>().ok())
    else {
        return Ok(None);
    };

    log::debug!("Checking if process with PID {} exists...", pid);

    let system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_processes(sysinfo::ProcessRefreshKind::nothing()),
    );

    if system.process(sysinfo::Pid::from_u32(pid)).is_some() {
        log::debug!(
            "Process with PID {} exists. NOT spawning new server, but attaching to existing one.",
            pid
        );
        Ok(Some(pid))
    } else {
        log::debug!("Found PID file, but process with that PID does not exist. Removing PID file.");
        std::fs::remove_file(path).map_err(|source| CheckPidError { source, pid })?;
        Ok(None)
    }
}

fn acquire_proxy_lease(paths: &ServerPaths, create: bool) -> Result<File> {
    let directory = paths
        .pid_file
        .parent()
        .context("Server directory unavailable")?;
    let metadata = std::fs::symlink_metadata(directory)?;
    anyhow::ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "Aliased server directory"
    );
    let lease_path = directory.join("proxy.lock");
    let mut options = std::fs::OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(create)
        .truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
        anyhow::ensure!(
            metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o022 == 0,
            "Server directory is not private to the current user"
        );
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    if let Ok(metadata) = std::fs::symlink_metadata(&lease_path) {
        anyhow::ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "Invalid proxy lease"
        );
    }
    let lease = options.open(&lease_path)?;
    let metadata = lease.metadata()?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() == 0,
        "Invalid proxy lease content"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o022 == 0,
            "Proxy lease is not private to the current user"
        );
    }
    lease
        .try_lock()
        .context("Another client already owns this remote server identifier")?;
    Ok(lease)
}

fn read_server_identity(path: &Path, maximum: u64) -> Result<String> {
    use std::io::Read as _;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    anyhow::ensure!(
        metadata.is_file() && metadata.len() <= maximum,
        "Invalid bounded server identity file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            metadata.uid() == unsafe { libc::geteuid() }
                && metadata.nlink() == 1
                && metadata.mode() & 0o022 == 0,
            "Aliased or unowned server identity file"
        );
    }
    let mut content = String::new();
    file.take(maximum + 1).read_to_string(&mut content)?;
    anyhow::ensure!(
        content.len() as u64 <= maximum,
        "Server identity changed while reading"
    );
    Ok(content)
}

fn write_server_identity(path: &Path, instance: &str) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    anyhow::ensure!(
        file.metadata()?.is_file(),
        "Invalid server identity destination"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = file.metadata()?;
        anyhow::ensure!(
            metadata.uid() == unsafe { libc::geteuid() }
                && metadata.nlink() == 1
                && metadata.mode() & 0o022 == 0,
            "Aliased or unowned server identity destination"
        );
    }
    file.set_len(0)?;
    file.write_all(instance.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

fn doctor_server_stopped(
    paths: &ServerPaths,
    expected_instance: &str,
    expected_pid: Option<u32>,
) -> Result<bool> {
    uuid::Uuid::parse_str(expected_instance).context("Invalid reviewed server instance")?;
    anyhow::ensure!(
        read_server_identity(&paths.pid_file.with_file_name("server.identity"), 64)?
            == expected_instance,
        "Remote server instance changed"
    );
    let pid = read_server_identity(&paths.pid_file, 16)?
        .parse::<u32>()
        .context("Invalid server PID")?;
    anyhow::ensure!(pid > 0, "Invalid server PID");
    anyhow::ensure!(
        expected_pid.is_none_or(|expected| expected == pid),
        "Reviewed server PID changed"
    );
    let system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_processes(sysinfo::ProcessRefreshKind::nothing()),
    );
    Ok(system.process(sysinfo::Pid::from_u32(pid)).is_none())
}

fn write_pid_file(path: &Path, pid: u32) -> Result<()> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    log::debug!("writing PID {} to file {:?}", pid, path);
    std::fs::write(path, pid.to_string()).context("Failed to write PID file")
}

async fn handle_io<R, W>(mut reader: R, mut writer: W, socket_name: &str) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    use remote::protocol::{read_message_raw, write_size_prefixed_buffer};

    let mut buffer = Vec::new();
    loop {
        read_message_raw(&mut reader, &mut buffer)
            .await
            .with_context(|| format!("failed to read message from {}", socket_name))?;
        write_size_prefixed_buffer(&mut writer, &mut buffer)
            .await
            .with_context(|| format!("failed to write message to {}", socket_name))?;
        writer.flush().await?;
        buffer.clear();
    }
}

fn initialize_settings(
    session: AnyProtoClient,
    fs: Arc<dyn Fs>,
    cx: &mut App,
) -> watch::Receiver<Option<NodeBinaryOptions>> {
    let (user_settings_file_rx, watcher_task) =
        watch_config_file(cx.background_executor(), fs, paths::settings_file().clone());

    handle_settings_file_changes(user_settings_file_rx, watcher_task, cx, {
        move |err, _cx| {
            if let Some(e) = err {
                log::info!("Server settings failed to change: {}", e);

                session
                    .send(proto::Toast {
                        project_id: REMOTE_SERVER_PROJECT_ID,
                        notification_id: "server-settings-failed".to_string(),
                        message: format!(
                            "Error in settings on remote host {:?}: {}",
                            paths::settings_file(),
                            e
                        ),
                    })
                    .log_err();
            } else {
                session
                    .send(proto::HideToast {
                        project_id: REMOTE_SERVER_PROJECT_ID,
                        notification_id: "server-settings-failed".to_string(),
                    })
                    .log_err();
            }
        }
    });

    let (mut tx, rx) = watch::channel(None);
    let mut node_settings = None;
    cx.observe_global::<SettingsStore>(move |cx| {
        let new_node_settings = &ProjectSettings::get_global(cx).node;
        if Some(new_node_settings) != node_settings.as_ref() {
            log::info!("Got new node settings: {new_node_settings:?}");
            let options = NodeBinaryOptions {
                allow_path_lookup: !new_node_settings.ignore_system_version,
                // TODO: Implement this setting
                allow_binary_download: true,
                use_paths: new_node_settings.path.as_ref().map(|node_path| {
                    let node_path = PathBuf::from(shellexpand::tilde(node_path).as_ref());
                    let npm_path = new_node_settings
                        .npm_path
                        .as_ref()
                        .map(|path| PathBuf::from(shellexpand::tilde(&path).as_ref()));
                    (
                        node_path.clone(),
                        npm_path.unwrap_or_else(|| {
                            let base_path = PathBuf::new();
                            node_path.parent().unwrap_or(&base_path).join("npm")
                        }),
                    )
                }),
            };
            node_settings = Some(new_node_settings.clone());
            tx.send(Some(options)).ok();
        }
    })
    .detach();

    rx
}

pub fn handle_settings_file_changes(
    mut server_settings_file: mpsc::UnboundedReceiver<String>,
    watcher_task: gpui::Task<()>,
    cx: &mut App,
    settings_changed: impl Fn(Option<anyhow::Error>, &mut App) + 'static,
) {
    let server_settings_content = cx
        .foreground_executor()
        .block_on(server_settings_file.next())
        .unwrap();
    SettingsStore::update_global(cx, |store, cx| {
        store
            .set_local_server_settings(&server_settings_content, cx)
            .log_err();
    });
    cx.spawn(async move |cx| {
        let _watcher_task = watcher_task;
        while let Some(server_settings_content) = server_settings_file.next().await {
            cx.update_global(|store: &mut SettingsStore, cx| {
                let result = store.set_local_server_settings(&server_settings_content, cx);
                if let Err(err) = &result {
                    log::error!("Failed to load server settings: {err}");
                }
                settings_changed(result.err(), cx);
                cx.refresh_windows();
            });
        }
    })
    .detach();
}

fn read_proxy_settings(cx: &mut Context<HeadlessProject>) -> Option<Url> {
    let proxy_str = ProxySettings::get_global(cx).proxy.to_owned();

    proxy_str
        .as_deref()
        .map(str::trim)
        .filter(|input| !input.is_empty())
        .and_then(|input| {
            input
                .parse::<Url>()
                .inspect_err(|e| log::error!("Error parsing proxy settings: {}", e))
                .ok()
        })
        .or_else(read_proxy_from_env)
}

fn cleanup_old_binaries() -> Result<()> {
    let server_dir = paths::remote_server_dir_relative();
    let release_channel = release_channel::RELEASE_CHANNEL.dev_name();
    let prefix = format!("zed-remote-server-{}-", release_channel);

    for entry in std::fs::read_dir(server_dir.as_std_path())? {
        let path = entry?.path();

        if let Some(file_name) = path.file_name()
            && let Some(version) = file_name.to_string_lossy().strip_prefix(&prefix)
            && !is_new_version(version)
            && !is_file_in_use(file_name)
        {
            log::info!("removing old remote server binary: {:?}", path);
            std::fs::remove_file(&path)?;
        }
    }

    Ok(())
}

// Remove this once 223 goes stable, we only have this to clean up old binaries on WSL
// we no longer download them into this folder, we use the same folder as other remote servers
fn cleanup_old_binaries_wsl() {
    let server_dir = paths::remote_wsl_server_dir_relative();
    if let Ok(()) = std::fs::remove_dir_all(server_dir.as_std_path()) {
        log::info!("removing old wsl remote server folder: {:?}", server_dir);
    }
}

fn is_new_version(version: &str) -> bool {
    semver::Version::from_str(version)
        .ok()
        .zip(semver::Version::from_str(env!("ZED_PKG_VERSION")).ok())
        .is_some_and(|(version, current_version)| version >= current_version)
}

fn is_file_in_use(file_name: &OsStr) -> bool {
    let info = sysinfo::System::new_with_specifics(sysinfo::RefreshKind::nothing().with_processes(
        sysinfo::ProcessRefreshKind::nothing().with_exe(sysinfo::UpdateKind::Always),
    ));

    for process in info.processes().values() {
        if process
            .exe()
            .is_some_and(|exe| exe.file_name().is_some_and(|name| name == file_name))
        {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doctor_restart_fixture_paths(root: &Path) -> ServerPaths {
        ServerPaths {
            log_file: root.join("server.log"),
            pid_file: root.join("server.pid"),
            stdin_socket: root.join("stdin.sock"),
            stdout_socket: root.join("stdout.sock"),
            stderr_socket: root.join("stderr.sock"),
        }
    }

    #[test]
    fn doctor_restart_proxy_lease_is_exclusive_and_preserves_other_files() {
        let directory = tempfile::tempdir().expect("Private server directory");
        let paths = doctor_restart_fixture_paths(directory.path());
        std::fs::write(directory.path().join("unrelated"), "preserve").expect("Sentinel");
        let lease = acquire_proxy_lease(&paths, true).expect("Own current identifier");
        assert!(
            acquire_proxy_lease(&paths, false).is_err(),
            "Second client must not connect or kill the current server"
        );
        drop(lease);
        assert!(acquire_proxy_lease(&paths, false).is_ok());
        assert_eq!(
            std::fs::read_to_string(directory.path().join("unrelated")).expect("Sentinel"),
            "preserve"
        );
        std::fs::write(directory.path().join("proxy.lock"), "foreign").expect("Unknown receipt");
        assert!(acquire_proxy_lease(&paths, true).is_err());
        assert_eq!(
            std::fs::read_to_string(directory.path().join("proxy.lock"))
                .expect("Preserved unknown content"),
            "foreign"
        );
    }

    #[test]
    fn doctor_restart_stop_requires_matching_instance_and_absent_process() {
        let directory = tempfile::tempdir().expect("Private server directory");
        let paths = doctor_restart_fixture_paths(directory.path());
        let instance = uuid::Uuid::new_v4().to_string();
        write_server_identity(&directory.path().join("server.identity"), &instance)
            .expect("Recorded instance");
        std::fs::write(&paths.pid_file, std::process::id().to_string()).expect("Current process");
        assert!(!doctor_server_stopped(&paths, &instance, None).expect("Alive observation"));
        assert!(
            doctor_server_stopped(
                &paths,
                &instance,
                Some(std::process::id().saturating_add(1))
            )
            .is_err()
        );
        std::fs::write(&paths.pid_file, u32::MAX.to_string()).expect("Absent PID");
        assert!(doctor_server_stopped(&paths, &instance, None).expect("Absent observation"));
        assert!(doctor_server_stopped(&paths, &uuid::Uuid::new_v4().to_string(), None).is_err());
        std::fs::write(&paths.pid_file, "unknown").expect("Invalid PID");
        assert!(doctor_server_stopped(&paths, &instance, None).is_err());
        std::fs::write(&paths.pid_file, "9".repeat(17)).expect("Oversized PID");
        assert!(doctor_server_stopped(&paths, &instance, None).is_err());
        assert!(doctor_server_stopped(&paths, "../outside", None).is_err());
        assert!(ServerPaths::existing("../outside").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn doctor_restart_aliases_cannot_redirect_control_files() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().expect("Private server directory");
        let paths = doctor_restart_fixture_paths(directory.path());
        let sentinel = directory.path().join("unrelated");
        std::fs::write(&sentinel, "preserve").expect("Sentinel");
        symlink(&sentinel, directory.path().join("proxy.lock")).expect("Aliased lease");
        assert!(acquire_proxy_lease(&paths, true).is_err());
        symlink(&sentinel, directory.path().join("server.identity")).expect("Aliased identity");
        assert!(
            write_server_identity(
                &directory.path().join("server.identity"),
                &uuid::Uuid::new_v4().to_string()
            )
            .is_err()
        );
        assert!(read_server_identity(&directory.path().join("server.identity"), 64).is_err());
        assert_eq!(
            std::fs::read_to_string(sentinel).expect("Sentinel"),
            "preserve"
        );
    }

    #[cfg(unix)]
    #[test]
    fn doctor_restart_hardlinks_preserve_unrelated_identity_content() {
        let directory = tempfile::tempdir().expect("Private server directory");
        let sentinel = directory.path().join("unrelated");
        let identity = directory.path().join("server.identity");
        std::fs::write(&sentinel, "preserve").expect("Sentinel");
        std::fs::hard_link(&sentinel, &identity).expect("Aliased identity");
        assert!(write_server_identity(&identity, &uuid::Uuid::new_v4().to_string()).is_err());
        assert!(read_server_identity(&identity, 64).is_err());
        assert_eq!(
            std::fs::read_to_string(&sentinel).expect("Preserved"),
            "preserve"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_task_completion_helper() {
        use clap::Parser as _;
        let Some(socket) = std::env::var_os("ZED_TEST_OBSERVER_SOCKET") else {
            return;
        };
        #[derive(clap::Parser)]
        struct Arguments {
            #[command(subcommand)]
            command: Commands,
        }
        let script = if std::env::var_os("ZED_TEST_OBSERVER_FAILURE").is_some() {
            "exit 7"
        } else {
            "test -z \"$ZED_PROFILE_TASK_TOKEN\" && test \"$1\" = 'argument with spaces'"
        };
        let native_capture = std::env::var_os("ZED_TEST_OBSERVER_NATIVE_CAPTURE").is_some();
        let checked_build = std::env::var_os("ZED_TEST_OBSERVER_CHECKED_BUILD").is_some();
        let mut input = vec![
            std::ffi::OsString::from("server"),
            "observe-task".into(),
            "--socket".into(),
            socket,
        ];
        if native_capture {
            input.push("--native-profile-capture".into());
        } else if checked_build {
            input.push("--capture-build-stdout".into());
        }
        let script = if native_capture {
            format!("{script} && test -z \"${{_NT_SYMBOL_PATH+x}}\"")
        } else if checked_build {
            format!("{script} && printf selected")
        } else {
            script.to_owned()
        };
        input.extend([
            "--".into(),
            "/bin/sh".into(),
            "-c".into(),
            script.into(),
            "fixture".into(),
            "argument with spaces".into(),
        ]);
        let arguments = Arguments::try_parse_from(input).expect("parse actual helper command");
        run(arguments.command).expect("exec replaces the helper on success");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_task_completion_command_executes_after_registration() {
        use util::macos_task_completion::{TASK_COMPLETION_TOKEN_ENV, TaskCompletionListener};
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let socket = listener.path();
            let mut child =
                smol::process::Command::new(std::env::current_exe().expect("test executable"))
                    .args([
                        "--exact",
                        "tests::native_task_completion_helper",
                        "--nocapture",
                    ])
                    .env("ZED_TEST_OBSERVER_SOCKET", &socket)
                    .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
                    .kill_on_drop(true)
                    .spawn()
                    .expect("spawn server command helper");
            let observed = listener.observe().await.expect("observe command");
            assert!(observed.success());
            assert_eq!(child.status().await.expect("wait"), observed);
            assert!(!socket.exists());
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_profile_capture_command_removes_inherited_symbol_setting() {
        use util::macos_task_completion::{TASK_COMPLETION_TOKEN_ENV, TaskCompletionListener};
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("bind");
            let socket = listener.path();
            let executable = std::env::current_exe().expect("test executable");
            let mut child = smol::process::Command::new(&executable)
                .args([
                    "--exact",
                    "tests::native_task_completion_helper",
                    "--nocapture",
                ])
                .env("ZED_TEST_OBSERVER_SOCKET", &socket)
                .env("ZED_TEST_OBSERVER_NATIVE_CAPTURE", "1")
                .env_remove("ZED_TEST_OBSERVER_FAILURE")
                .env("_NT_SYMBOL_PATH", "srv*http://127.0.0.1:9")
                .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
                .kill_on_drop(true)
                .spawn()
                .expect("actual server command helper");
            let observed = listener
                .observe_native_profile_with_helper_check(|actual| async move {
                    assert_eq!(actual, executable);
                    Ok(())
                })
                .await
                .expect("native helper registration");
            assert!(observed.success());
            assert_eq!(child.status().await.expect("actual exit"), observed);
            assert!(!socket.exists());
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_checked_build_stdout_command_records_actual_child() {
        use util::macos_task_completion::{
            TASK_COMPLETION_TOKEN_ENV, TaskBuildInvocation, TaskCompletionListener,
        };
        smol::block_on(async {
            let listener = TaskCompletionListener::bind([0x42; 32]).expect("checked listener");
            let executable = std::env::current_exe().expect("real helper image");
            let expected = TaskBuildInvocation::new(
                "/bin/sh".into(),
                vec!["-c".into(), "test -z \"$ZED_PROFILE_TASK_TOKEN\" && test \"$1\" = 'argument with spaces' && printf selected".into(), "fixture".into(), "argument with spaces".into()],
                std::env::current_dir().expect("actual cwd"),
                Default::default(),
            ).expect("exact expected invocation");
            let mut child = smol::process::Command::new(&executable)
                .args([
                    "--exact",
                    "tests::native_task_completion_helper",
                    "--nocapture",
                ])
                .env("ZED_TEST_OBSERVER_SOCKET", listener.path())
                .env("ZED_TEST_OBSERVER_CHECKED_BUILD", "1")
                .env_remove("ZED_TEST_OBSERVER_NATIVE_CAPTURE")
                .env_remove("ZED_TEST_OBSERVER_FAILURE")
                .env(TASK_COMPLETION_TOKEN_ENV, "42".repeat(32))
                .kill_on_drop(true)
                .spawn()
                .expect("actual production parser/dispatch");
            let report = listener
                .observe_checked_build_stdout_with_helper_check(
                    64,
                    expected.clone(),
                    |actual| async move {
                        assert_eq!(actual, executable);
                        Ok(())
                    },
                )
                .await
                .expect("actual checked child");
            let (status, bytes, actual) = report.into_parts_with_invocation();
            assert!(status.success());
            assert_eq!(child.status().await.expect("actual helper exit"), status);
            assert_eq!(bytes, b"selected");
            let actual = actual.expect("actual final child receipt");
            assert!(expected.matches_observed(&actual));
            assert!(!actual.environment().contains_key(TASK_COMPLETION_TOKEN_ENV));
        });
        use clap::Parser as _;
        #[derive(clap::Parser)]
        struct Arguments {
            #[command(subcommand)]
            command: Commands,
        }
        for other in ["--capture-stdout", "--native-profile-capture"] {
            assert!(
                Arguments::try_parse_from([
                    "server",
                    "observe-task",
                    "--socket",
                    "/tmp/private.sock",
                    "--capture-build-stdout",
                    other,
                    "--",
                    "/bin/sh"
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn recipe_scan_bootstrap_flags_use_the_existing_server_commands() {
        use clap::Parser as _;
        #[derive(clap::Parser)]
        struct Arguments {
            #[command(subcommand)]
            command: Commands,
        }
        let proxy = Arguments::try_parse_from([
            "server",
            "proxy",
            "--identifier",
            "recipe-test",
            "--isolated-recipe-test",
        ])
        .expect("isolated proxy arguments");
        assert!(matches!(
            proxy.command,
            Commands::Proxy {
                isolated_recipe_test: true,
                reconnect: false,
                ..
            }
        ));
        let server = Arguments::try_parse_from([
            "server",
            "run",
            "--isolated-recipe-test",
            "--log-file",
            "/tmp/log",
            "--pid-file",
            "/tmp/pid",
            "--stdin-socket",
            "/tmp/in",
            "--stdout-socket",
            "/tmp/out",
            "--stderr-socket",
            "/tmp/err",
        ])
        .expect("isolated server arguments");
        assert!(matches!(
            server.command,
            Commands::Run {
                isolated_recipe_test: true,
                ..
            }
        ));
        let ordinary = Arguments::try_parse_from(["server", "proxy", "--identifier", "ordinary"])
            .expect("ordinary proxy arguments");
        assert!(matches!(
            ordinary.command,
            Commands::Proxy {
                isolated_recipe_test: false,
                ..
            }
        ));
    }

    #[test]
    fn recipe_scan_reconnect_is_rejected_before_opening_server_paths() {
        assert!(matches!(
            execute_proxy("../must-not-open".into(), true, true),
            Err(ExecuteProxyError::IsolatedRecipeTest)
        ));
    }

    #[test]
    fn rotated_remote_log_path_uses_numbered_log_suffix() {
        assert_eq!(
            rotated_log_path(Path::new("server-workspace-12.log")),
            PathBuf::from("server-workspace-12.1.log")
        );
    }

    #[test]
    fn opening_remote_log_rotates_existing_oversized_log() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let log_path = temp_dir.path().join("server-workspace-12.log");
        let rotated_path = temp_dir.path().join("server-workspace-12.1.log");
        let existing_contents = vec![b'x'; REMOTE_SERVER_LOG_MAX_BYTES as usize];
        std::fs::write(&log_path, &existing_contents).expect("write oversized log");

        let _log_file = RotatingLogFile::open(&log_path).expect("open rotating log file");

        assert_eq!(
            std::fs::read(&rotated_path).expect("read rotated log"),
            existing_contents
        );
        assert_eq!(
            std::fs::metadata(&log_path)
                .expect("active log metadata")
                .len(),
            0
        );
    }

    #[test]
    fn writing_remote_log_rotates_before_exceeding_size_limit() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let log_path = temp_dir.path().join("server-workspace-12.log");
        let rotated_path = temp_dir.path().join("server-workspace-12.1.log");
        let existing_contents = vec![b'x'; REMOTE_SERVER_LOG_MAX_BYTES as usize - 1];
        let new_contents = b"yz";
        std::fs::write(&log_path, &existing_contents).expect("write existing log contents");
        let mut log_file = RotatingLogFile::open(&log_path).expect("open rotating log file");

        log_file
            .write_all(new_contents)
            .expect("write log contents");
        log_file.flush().expect("flush log file");

        assert_eq!(
            std::fs::read(&rotated_path).expect("read rotated log"),
            existing_contents
        );
        assert_eq!(
            std::fs::read(&log_path).expect("read active log"),
            new_contents
        );
    }
}
