use anyhow::{Context as _, Result, anyhow, bail};
#[cfg(any(test, feature = "test-support"))]
use async_pipe::{PipeReader, PipeWriter};
use dap_types::{
    ErrorResponse,
    messages::{Message, Response},
};
use futures::{AsyncRead, AsyncReadExt as _, AsyncWrite, FutureExt as _, channel::oneshot, select};
use gpui::{AppContext as _, AsyncApp, BackgroundExecutor, Task};
use parking_lot::Mutex;
use proto::ErrorExt;
use settings::Settings as _;
use smallvec::SmallVec;
use smol::{
    channel::{Receiver, Sender, unbounded},
    io::{AsyncBufReadExt as _, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
};
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use task::TcpArgumentsTemplate;
use util::{ConnectionResult, ResultExt, process::Child};

use crate::{
    adapters::{DebugAdapterBinary, TcpArguments},
    client::DapMessageHandler,
    debugger_settings::DebuggerSettings,
};

pub(crate) type IoMessage = str;
pub(crate) type Command = str;
pub type IoHandler = Box<dyn Send + FnMut(IoKind, Option<&Command>, &IoMessage)>;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum LogKind {
    Adapter,
    Rpc,
}

#[derive(Clone, Copy)]
pub enum IoKind {
    StdIn,
    StdOut,
    StdErr,
}

#[cfg(any(test, feature = "test-support"))]
pub enum RequestHandling<T> {
    Respond(T),
    Exit,
}

type LogHandlers = Arc<Mutex<SmallVec<[(LogKind, IoHandler); 2]>>>;

pub trait Transport: Send + Sync {
    fn has_adapter_logs(&self) -> bool;
    fn tcp_arguments(&self) -> Option<TcpArguments>;
    fn connect(
        &mut self,
    ) -> Task<
        Result<(
            Box<dyn AsyncWrite + Unpin + Send + 'static>,
            Box<dyn AsyncRead + Unpin + Send + 'static>,
        )>,
    >;
    fn kill(&mut self);
    #[cfg(any(test, feature = "test-support"))]
    fn as_fake(&self) -> &FakeTransport {
        unreachable!()
    }
}

#[cfg(any(test, feature = "test-support"))]
struct RealTransportForTest;

#[cfg(any(test, feature = "test-support"))]
impl gpui::Global for RealTransportForTest {}

#[cfg(any(test, feature = "test-support"))]
pub fn use_real_transport_for_test(cx: &mut gpui::App) {
    cx.set_global(RealTransportForTest);
}

async fn start(
    binary: &DebugAdapterBinary,
    log_handlers: LogHandlers,
    restricted: bool,
    cx: &mut AsyncApp,
) -> Result<Box<dyn Transport>> {
    #[cfg(any(test, feature = "test-support"))]
    if !cx.update(|cx| cx.has_global::<RealTransportForTest>()) {
        if let Some(connection) = binary.connection.clone() {
            return Ok(Box::new(FakeTransport::start_tcp(connection, cx).await?));
        } else {
            return Ok(Box::new(FakeTransport::start_stdio(cx).await?));
        }
    }

    if binary.connection.is_some() {
        Ok(Box::new(
            TcpTransport::start(binary, log_handlers, cx).await?,
        ))
    } else {
        Ok(Box::new(
            StdioTransport::start(binary, log_handlers, restricted, cx).await?,
        ))
    }
}

pub(crate) struct PendingRequests {
    inner: Option<HashMap<u64, oneshot::Sender<Result<Response>>>>,
}

impl PendingRequests {
    fn new() -> Self {
        Self {
            inner: Some(HashMap::default()),
        }
    }

    fn flush(&mut self, e: anyhow::Error) {
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        for (_, sender) in inner.drain() {
            sender.send(Err(e.cloned())).ok();
        }
    }

    pub(crate) fn insert(
        &mut self,
        sequence_id: u64,
        callback_tx: oneshot::Sender<Result<Response>>,
    ) -> anyhow::Result<()> {
        let Some(inner) = self.inner.as_mut() else {
            bail!("client is closed")
        };
        inner.insert(sequence_id, callback_tx);
        Ok(())
    }

    pub(crate) fn remove(
        &mut self,
        sequence_id: u64,
    ) -> anyhow::Result<Option<oneshot::Sender<Result<Response>>>> {
        let Some(inner) = self.inner.as_mut() else {
            bail!("client is closed");
        };
        Ok(inner.remove(&sequence_id))
    }

    fn disconnect(&mut self, error: anyhow::Error) {
        self.flush(error);
        self.inner = None;
    }

    pub(crate) fn shutdown(&mut self) {
        self.disconnect(anyhow!("transport shutdown"));
    }

    fn reconnect(&mut self) {
        self.inner = Some(HashMap::default());
    }
}

pub(crate) struct TransportDelegate {
    log_handlers: LogHandlers,
    pub(crate) pending_requests: Arc<Mutex<PendingRequests>>,
    pub(crate) transport: Mutex<Box<dyn Transport>>,
    pub(crate) server_tx: smol::lock::Mutex<Option<Sender<Message>>>,
    tasks: Mutex<Vec<Task<()>>>,
    input_limit: Option<usize>,
}

impl TransportDelegate {
    pub(crate) async fn start(
        binary: &DebugAdapterBinary,
        input_limit: Option<usize>,
        cx: &mut AsyncApp,
    ) -> Result<Self> {
        let log_handlers: LogHandlers = Default::default();
        let transport = start(binary, log_handlers.clone(), input_limit.is_some(), cx).await?;
        Ok(Self {
            transport: Mutex::new(transport),
            log_handlers,
            server_tx: Default::default(),
            pending_requests: Arc::new(Mutex::new(PendingRequests::new())),
            tasks: Default::default(),
            input_limit,
        })
    }

    pub async fn connect(
        &self,
        message_handler: DapMessageHandler,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        let (server_tx, client_rx) = unbounded::<Message>();
        self.tasks.lock().clear();
        self.pending_requests
            .lock()
            .disconnect(anyhow!("debugger transport reconnecting"));

        let log_dap_communications =
            cx.update(|cx| DebuggerSettings::get_global(cx).log_dap_communications);

        let connect = self.transport.lock().connect();
        let (input, output) = connect.await?;
        self.pending_requests.lock().reconnect();

        let log_handler = if log_dap_communications && self.input_limit.is_none() {
            Some(self.log_handlers.clone())
        } else {
            None
        };

        let pending_requests = self.pending_requests.clone();
        let output_log_handler = log_handler.clone();
        let input_limit = self.input_limit;
        {
            let mut tasks = self.tasks.lock();
            tasks.push(cx.background_spawn(async move {
                match Self::recv_from_server(
                    output,
                    message_handler,
                    pending_requests.clone(),
                    output_log_handler,
                    input_limit,
                )
                .await
                {
                    Ok(()) => {
                        pending_requests
                            .lock()
                            .disconnect(anyhow!("debugger shutdown unexpectedly"));
                    }
                    Err(e) => {
                        pending_requests.lock().disconnect(e);
                    }
                }
            }));

            tasks.push(cx.background_spawn(async move {
                match Self::send_to_server(input, client_rx, log_handler).await {
                    Ok(()) => {}
                    Err(e) => log::error!("Error handling debugger input: {e}"),
                }
            }));
        }

        *self.server_tx.lock().await = Some(server_tx.clone());

        Ok(())
    }

    pub(crate) fn tcp_arguments(&self) -> Option<TcpArguments> {
        self.transport.lock().tcp_arguments()
    }

    pub(crate) async fn send_message(&self, message: Message) -> Result<()> {
        if let Some(server_tx) = self.server_tx.lock().await.as_ref() {
            server_tx.send(message).await.context("sending message")
        } else {
            anyhow::bail!("Server tx already dropped")
        }
    }

    async fn handle_adapter_log(
        stdout: impl AsyncRead + Unpin + Send + 'static,
        iokind: IoKind,
        log_handlers: LogHandlers,
    ) {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();

        loop {
            line.clear();

            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    log::debug!("handle_adapter_log: {}", e);
                    break;
                }
            }

            // Clean up logs by trimming unnecessary whitespace/newlines before inserting into log.
            let line = line.trim();

            log::debug!("stderr: {line}");

            for (kind, handler) in log_handlers.lock().iter_mut() {
                if matches!(kind, LogKind::Adapter) {
                    handler(iokind, None, line);
                }
            }
        }
    }

    fn build_rpc_message(message: String) -> String {
        format!("Content-Length: {}\r\n\r\n{}", message.len(), message)
    }

    async fn send_to_server<Stdin>(
        mut server_stdin: Stdin,
        client_rx: Receiver<Message>,
        log_handlers: Option<LogHandlers>,
    ) -> Result<()>
    where
        Stdin: AsyncWrite + Unpin + Send + 'static,
    {
        let result = loop {
            match client_rx.recv().await {
                Ok(message) => {
                    let command = match &message {
                        Message::Request(request) => Some(request.command.as_str()),
                        Message::Response(response) => Some(response.command.as_str()),
                        _ => None,
                    };

                    let message = match serde_json::to_string(&message) {
                        Ok(message) => message,
                        Err(e) => break Err(e.into()),
                    };

                    if let Some(log_handlers) = log_handlers.as_ref() {
                        for (kind, log_handler) in log_handlers.lock().iter_mut() {
                            if matches!(kind, LogKind::Rpc) {
                                log_handler(IoKind::StdIn, command, &message);
                            }
                        }
                    }

                    if let Err(e) = server_stdin
                        .write_all(Self::build_rpc_message(message).as_bytes())
                        .await
                    {
                        break Err(e.into());
                    }

                    if let Err(e) = server_stdin.flush().await {
                        break Err(e.into());
                    }
                }
                Err(error) => break Err(error.into()),
            }
        };

        log::debug!("Handle adapter input dropped");

        result
    }

    async fn recv_from_server<Stdout>(
        server_stdout: Stdout,
        mut message_handler: DapMessageHandler,
        pending_requests: Arc<Mutex<PendingRequests>>,
        log_handlers: Option<LogHandlers>,
        mut remaining_bytes: Option<usize>,
    ) -> Result<()>
    where
        Stdout: AsyncRead + Unpin + Send + 'static,
    {
        let mut recv_buffer = String::new();
        let mut reader = BufReader::new(server_stdout);

        let result = loop {
            let result = Self::receive_server_message_with_limit(
                &mut reader,
                &mut recv_buffer,
                log_handlers.as_ref(),
                &mut remaining_bytes,
            )
            .await;
            match result {
                ConnectionResult::Timeout => anyhow::bail!("Timed out when connecting to debugger"),
                ConnectionResult::ConnectionReset => {
                    log::info!("Debugger closed the connection");
                    return Ok(());
                }
                ConnectionResult::Result(Ok(Message::Response(res))) => {
                    let tx = pending_requests.lock().remove(res.request_seq)?;
                    if let Some(tx) = tx {
                        let response = if remaining_bytes.is_some() && !res.success {
                            Err(anyhow!("Diagnostic debug adapter request failed"))
                        } else {
                            Self::process_response(res)
                        };
                        if let Err(error) = tx.send(response) {
                            if remaining_bytes.is_some() {
                                log::trace!("Diagnostic debugger response receiver was cancelled");
                            } else {
                                log::trace!("Did not send response `{:?}` for a cancelled", error);
                            }
                        }
                    } else {
                        message_handler(Message::Response(res))
                    }
                }
                ConnectionResult::Result(Ok(message)) => message_handler(message),
                ConnectionResult::Result(Err(e)) => break Err(e),
            }
        };

        log::debug!("Handle adapter output dropped");

        result
    }

    fn process_response(response: Response) -> Result<Response> {
        if response.success {
            Ok(response)
        } else {
            if let Some(error_message) = response
                .body
                .clone()
                .and_then(|body| serde_json::from_value::<ErrorResponse>(body).ok())
                .and_then(|response| response.error.map(|msg| msg.format))
                .or_else(|| response.message.clone())
            {
                anyhow::bail!(error_message);
            };

            anyhow::bail!(
                "Received error response from adapter. Response: {:?}",
                response
            );
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    async fn receive_server_message<Stdout>(
        reader: &mut BufReader<Stdout>,
        buffer: &mut String,
        log_handlers: Option<&LogHandlers>,
    ) -> ConnectionResult<Message>
    where
        Stdout: AsyncRead + Unpin + Send + 'static,
    {
        Self::receive_server_message_with_limit(reader, buffer, log_handlers, &mut None).await
    }

    async fn receive_server_message_with_limit<Stdout>(
        reader: &mut BufReader<Stdout>,
        buffer: &mut String,
        log_handlers: Option<&LogHandlers>,
        remaining_bytes: &mut Option<usize>,
    ) -> ConnectionResult<Message>
    where
        Stdout: AsyncRead + Unpin + Send + 'static,
    {
        let mut content_length = None;
        let mut header_bytes = 0usize;
        loop {
            buffer.clear();
            let line = if remaining_bytes.is_some() {
                Self::read_bounded_header_line(reader, buffer, &mut header_bytes).await
            } else {
                reader.read_line(buffer).await.map_err(anyhow::Error::from)
            };
            match line {
                Ok(0) => return ConnectionResult::ConnectionReset,
                Ok(_) => {}
                Err(error) => return ConnectionResult::Result(Err(error)),
            };

            if buffer == "\r\n" {
                break;
            }

            if let Some(("Content-Length", value)) = buffer.trim().split_once(": ") {
                match value.parse().context("invalid content length") {
                    Ok(length) => content_length = Some(length),
                    Err(e) => return ConnectionResult::Result(Err(e)),
                }
            }
        }

        let content_length = match content_length.context("missing content length") {
            Ok(length) => length,
            Err(e) => return ConnectionResult::Result(Err(e)),
        };

        if let Some(remaining) = remaining_bytes {
            let Some(cost) = header_bytes.checked_add(content_length) else {
                return ConnectionResult::Result(Err(anyhow!(
                    "Diagnostic debugger input size overflow"
                )));
            };
            let Some(available) = remaining.checked_sub(cost) else {
                return ConnectionResult::Result(Err(anyhow!(
                    "Diagnostic debugger input exceeds the approved aggregate limit"
                )));
            };
            *remaining = available;
        }
        let mut content = vec![0; content_length];
        if let Err(e) = reader
            .read_exact(&mut content)
            .await
            .with_context(|| "reading after a loop")
        {
            return ConnectionResult::Result(Err(e));
        }

        let message_str = match std::str::from_utf8(&content).context("invalid utf8 from server") {
            Ok(str) => str,
            Err(e) => return ConnectionResult::Result(Err(e)),
        };

        let message = serde_json::from_str::<Message>(message_str).map_err(|error| {
            if remaining_bytes.is_some() {
                anyhow!("Diagnostic debug adapter sent an invalid protocol message")
            } else {
                anyhow::Error::from(error).context("deserializing server message")
            }
        });

        if let Some(log_handlers) = log_handlers {
            let command = match &message {
                Ok(Message::Request(request)) => Some(request.command.as_str()),
                Ok(Message::Response(response)) => Some(response.command.as_str()),
                _ => None,
            };

            for (kind, log_handler) in log_handlers.lock().iter_mut() {
                if matches!(kind, LogKind::Rpc) {
                    log_handler(IoKind::StdOut, command, message_str);
                }
            }
        }

        ConnectionResult::Result(message)
    }

    async fn read_bounded_header_line<Reader: AsyncRead + Unpin>(
        reader: &mut Reader,
        buffer: &mut String,
        header_bytes: &mut usize,
    ) -> Result<usize> {
        let mut bytes = Vec::new();
        loop {
            let mut byte = [0u8];
            if reader.read(&mut byte).await? == 0 {
                return Ok(0);
            }
            *header_bytes += 1;
            anyhow::ensure!(
                *header_bytes <= 4096,
                "Diagnostic debugger header exceeds the size limit"
            );
            bytes.push(byte[0]);
            if byte[0] == b'\n' {
                break;
            }
        }
        *buffer = String::from_utf8(bytes)?;
        Ok(buffer.len())
    }

    pub fn has_adapter_logs(&self) -> bool {
        self.input_limit.is_none() && self.transport.lock().has_adapter_logs()
    }

    pub fn add_log_handler<F>(&self, f: F, kind: LogKind)
    where
        F: 'static + Send + FnMut(IoKind, Option<&Command>, &IoMessage),
    {
        if self.input_limit.is_some() {
            log::warn!(
                "Protocol and adapter logs are unavailable for a metadata-only diagnostic session"
            );
            return;
        }
        let mut log_handlers = self.log_handlers.lock();
        log_handlers.push((kind, Box::new(f)));
    }
}

fn terminate_adapter_process(mut process: Child, executor: &BackgroundExecutor) {
    process.kill().log_err();
    executor
        .spawn(async move {
            process
                .status()
                .await
                .context("Failed to reap debug adapter process")
                .log_err();
        })
        .detach();
}

pub struct TcpTransport {
    executor: BackgroundExecutor,
    pub port: u16,
    pub host: IpAddr,
    pub timeout: u64,
    process: Arc<Mutex<Option<Child>>>,
    _stderr_task: Option<Task<()>>,
    _stdout_task: Option<Task<()>>,
}

impl TcpTransport {
    /// Get an open port to use with the tcp client when not supplied by debug config
    pub async fn port(host: &TcpArgumentsTemplate) -> Result<u16> {
        if let Some(port) = host.port {
            Ok(port)
        } else {
            Self::unused_port(host.host()).await
        }
    }

    pub async fn unused_port(host: IpAddr) -> Result<u16> {
        Ok(TcpListener::bind(SocketAddr::new(host, 0))
            .await?
            .local_addr()?
            .port())
    }

    async fn start(
        binary: &DebugAdapterBinary,
        log_handlers: LogHandlers,
        cx: &mut AsyncApp,
    ) -> Result<Self> {
        let connection_args = binary
            .connection
            .as_ref()
            .context("No connection arguments provided")?;

        let host = connection_args.host;
        let port = connection_args.port;

        let mut process = None;
        let mut stdout_task = None;
        let mut stderr_task = None;

        if let Some(command) = &binary.command {
            let mut command = util::command::new_std_command(&command);

            if let Some(cwd) = &binary.cwd {
                command.current_dir(cwd);
            }

            command.args(&binary.arguments);
            command.envs(&binary.envs);

            let mut p = Child::spawn(command, Stdio::null(), Stdio::piped(), Stdio::piped())
                .with_context(|| "failed to start debug adapter.")?;

            stdout_task = p.stdout.take().map(|stdout| {
                cx.background_executor()
                    .spawn(TransportDelegate::handle_adapter_log(
                        stdout,
                        IoKind::StdOut,
                        log_handlers.clone(),
                    ))
            });
            stderr_task = p.stderr.take().map(|stderr| {
                cx.background_executor()
                    .spawn(TransportDelegate::handle_adapter_log(
                        stderr,
                        IoKind::StdErr,
                        log_handlers,
                    ))
            });
            process = Some(p);
        };

        let timeout = connection_args
            .timeout
            .unwrap_or_else(|| cx.update(|cx| DebuggerSettings::get_global(cx).timeout));

        log::info!(
            "Debug adapter has connected to TCP server {}:{}",
            host,
            port
        );

        let this = Self {
            executor: cx.background_executor().clone(),
            port,
            host,
            process: Arc::new(Mutex::new(process)),
            timeout,
            _stdout_task: stdout_task,
            _stderr_task: stderr_task,
        };

        Ok(this)
    }
}

impl Transport for TcpTransport {
    fn has_adapter_logs(&self) -> bool {
        true
    }

    fn kill(&mut self) {
        let process = self.process.lock().take();
        if let Some(process) = process {
            terminate_adapter_process(process, &self.executor);
        }
    }

    fn tcp_arguments(&self) -> Option<TcpArguments> {
        Some(TcpArguments {
            host: self.host,
            port: self.port,
            timeout: Some(self.timeout),
        })
    }

    fn connect(
        &mut self,
    ) -> Task<
        Result<(
            Box<dyn AsyncWrite + Unpin + Send + 'static>,
            Box<dyn AsyncRead + Unpin + Send + 'static>,
        )>,
    > {
        let executor = self.executor.clone();
        let timeout = self.timeout;
        let address = SocketAddr::new(self.host, self.port);
        let process = self.process.clone();
        let owns_process = process.lock().is_some();
        executor.clone().spawn(async move {
            select! {
                _ = executor.timer(Duration::from_millis(timeout)).fuse() => {
                    anyhow::bail!("Connection to TCP DAP timeout {address}");
                },
                result = executor.clone().spawn(async move {
                    loop {
                        if owns_process && process.lock().is_none() {
                            anyhow::bail!("Debug adapter process was stopped before debugger attached.");
                        }
                        match TcpStream::connect(address).await {
                            Ok(stream) => {
                                let (read, write) = stream.split();
                                return Ok((Box::new(write) as _, Box::new(read) as _))
                            },
                            Err(_) => {
                                let exited_process = {
                                    let mut process_slot = process.lock();
                                    if let Some(child) = process_slot.as_mut()
                                        && child.try_status().context("Failed to read debug adapter exit status")?.is_some()
                                    {
                                        process_slot.take()
                                    } else {
                                        None
                                    }
                                };
                                if let Some(child) = exited_process {
                                    let output = child.output().await?;
                                    let output = if output.stderr.is_empty() {
                                        String::from_utf8_lossy(&output.stdout).to_string()
                                    } else {
                                        String::from_utf8_lossy(&output.stderr).to_string()
                                    };
                                    anyhow::bail!("{output}\nerror: process exited before debugger attached.");
                                }

                                executor.timer(Duration::from_millis(100)).await;
                            }
                        }
                    }
                }).fuse() => result
            }
        })
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        self.kill();
    }
}

pub struct StdioTransport {
    executor: BackgroundExecutor,
    process: Mutex<Option<Child>>,
    _stderr_task: Option<Task<()>>,
}

impl StdioTransport {
    // #[allow(dead_code, reason = "This is used in non test builds of Zed")]
    async fn start(
        binary: &DebugAdapterBinary,
        log_handlers: LogHandlers,
        restricted: bool,
        cx: &mut AsyncApp,
    ) -> Result<Self> {
        let Some(binary_command) = &binary.command else {
            bail!(
                "When using the `stdio` transport, the path to a debug adapter binary must be set by Zed."
            );
        };
        let mut command = util::command::new_std_command(&binary_command);
        if restricted {
            command.env_clear();
        }

        if let Some(cwd) = &binary.cwd {
            command.current_dir(cwd);
        }

        command.args(&binary.arguments);
        command.envs(&binary.envs);

        let mut process = Child::spawn(
            command,
            Stdio::piped(),
            Stdio::piped(),
            if restricted {
                Stdio::null()
            } else {
                Stdio::piped()
            },
        )?;

        let _stderr_task = process.stderr.take().map(|stderr| {
            cx.background_spawn(TransportDelegate::handle_adapter_log(
                stderr,
                IoKind::StdErr,
                log_handlers,
            ))
        });

        let process = Mutex::new(Some(process));

        Ok(Self {
            executor: cx.background_executor().clone(),
            process,
            _stderr_task,
        })
    }
}

impl Transport for StdioTransport {
    fn has_adapter_logs(&self) -> bool {
        true
    }

    fn kill(&mut self) {
        let process = self.process.lock().take();
        if let Some(process) = process {
            terminate_adapter_process(process, &self.executor);
        }
    }

    fn connect(
        &mut self,
    ) -> Task<
        Result<(
            Box<dyn AsyncWrite + Unpin + Send + 'static>,
            Box<dyn AsyncRead + Unpin + Send + 'static>,
        )>,
    > {
        let result = util::maybe!({
            let mut process_slot = self.process.lock();
            let process = process_slot.as_mut().context("Cannot reconnect")?;
            Ok((
                Box::new(process.stdin.take().context("Cannot reconnect")?) as _,
                Box::new(process.stdout.take().context("Cannot reconnect")?) as _,
            ))
        });
        Task::ready(result)
    }

    fn tcp_arguments(&self) -> Option<TcpArguments> {
        None
    }
}

impl Drop for StdioTransport {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(any(test, feature = "test-support"))]
type RequestHandler = Box<dyn Send + FnMut(u64, serde_json::Value) -> RequestHandling<Response>>;

#[cfg(any(test, feature = "test-support"))]
type ResponseHandler = Box<dyn Send + Fn(Response)>;

#[cfg(any(test, feature = "test-support"))]
pub struct FakeTransport {
    // for sending fake response back from adapter side
    request_handlers: Arc<Mutex<HashMap<&'static str, RequestHandler>>>,
    // for reverse request responses
    response_handlers: Arc<Mutex<HashMap<&'static str, ResponseHandler>>>,
    message_handler: Option<Task<Result<()>>>,
    kind: FakeTransportKind,
}

#[cfg(any(test, feature = "test-support"))]
pub enum FakeTransportKind {
    Stdio {
        stdin_writer: Option<PipeWriter>,
        stdout_reader: Option<PipeReader>,
    },
    Tcp {
        connection: TcpArguments,
        executor: BackgroundExecutor,
    },
}

#[cfg(any(test, feature = "test-support"))]
impl FakeTransport {
    pub fn on_request<R: dap_types::requests::Request, F>(&self, mut handler: F)
    where
        F: 'static
            + Send
            + FnMut(u64, R::Arguments) -> RequestHandling<Result<R::Response, ErrorResponse>>,
    {
        self.request_handlers.lock().insert(
            R::COMMAND,
            Box::new(move |seq, args| {
                let result = handler(seq, serde_json::from_value(args).unwrap());
                let RequestHandling::Respond(response) = result else {
                    return RequestHandling::Exit;
                };
                let response = match response {
                    Ok(response) => Response {
                        seq: seq + 1,
                        request_seq: seq,
                        success: true,
                        command: R::COMMAND.into(),
                        body: Some(serde_json::to_value(response).unwrap()),
                        message: None,
                    },
                    Err(response) => Response {
                        seq: seq + 1,
                        request_seq: seq,
                        success: false,
                        command: R::COMMAND.into(),
                        body: Some(serde_json::to_value(response).unwrap()),
                        message: None,
                    },
                };
                RequestHandling::Respond(response)
            }),
        );
    }

    pub fn on_response<R: dap_types::requests::Request, F>(&self, handler: F)
    where
        F: 'static + Send + Fn(Response),
    {
        self.response_handlers
            .lock()
            .insert(R::COMMAND, Box::new(handler));
    }

    async fn start_tcp(connection: TcpArguments, cx: &mut AsyncApp) -> Result<Self> {
        Ok(Self {
            request_handlers: Arc::new(Mutex::new(HashMap::default())),
            response_handlers: Arc::new(Mutex::new(HashMap::default())),
            message_handler: None,
            kind: FakeTransportKind::Tcp {
                connection,
                executor: cx.background_executor().clone(),
            },
        })
    }

    async fn handle_messages(
        request_handlers: Arc<Mutex<HashMap<&'static str, RequestHandler>>>,
        response_handlers: Arc<Mutex<HashMap<&'static str, ResponseHandler>>>,
        stdin_reader: PipeReader,
        stdout_writer: PipeWriter,
    ) -> Result<()> {
        use dap_types::requests::{Request, RunInTerminal, StartDebugging};
        use serde_json::json;

        let mut reader = BufReader::new(stdin_reader);
        let stdout_writer = Arc::new(smol::lock::Mutex::new(stdout_writer));
        let mut buffer = String::new();

        loop {
            match TransportDelegate::receive_server_message(&mut reader, &mut buffer, None).await {
                ConnectionResult::Timeout => {
                    anyhow::bail!("Timed out when connecting to debugger");
                }
                ConnectionResult::ConnectionReset => {
                    log::info!("Debugger closed the connection");
                    break Ok(());
                }
                ConnectionResult::Result(Err(e)) => break Err(e),
                ConnectionResult::Result(Ok(message)) => {
                    match message {
                        Message::Request(request) => {
                            // redirect reverse requests to stdout writer/reader
                            if request.command == RunInTerminal::COMMAND
                                || request.command == StartDebugging::COMMAND
                            {
                                let message =
                                    serde_json::to_string(&Message::Request(request)).unwrap();

                                let mut writer = stdout_writer.lock().await;
                                writer
                                    .write_all(
                                        TransportDelegate::build_rpc_message(message).as_bytes(),
                                    )
                                    .await
                                    .unwrap();
                                writer.flush().await.unwrap();
                            } else {
                                let response = if let Some(handle) =
                                    request_handlers.lock().get_mut(request.command.as_str())
                                {
                                    handle(request.seq, request.arguments.unwrap_or(json!({})))
                                } else {
                                    panic!("No request handler for {}", request.command);
                                };
                                let response = match response {
                                    RequestHandling::Respond(response) => response,
                                    RequestHandling::Exit => {
                                        break Err(anyhow!("exit in response to request"));
                                    }
                                };
                                let success = response.success;
                                let message =
                                    serde_json::to_string(&Message::Response(response)).unwrap();

                                let mut writer = stdout_writer.lock().await;
                                writer
                                    .write_all(
                                        TransportDelegate::build_rpc_message(message).as_bytes(),
                                    )
                                    .await
                                    .unwrap();

                                if request.command == dap_types::requests::Initialize::COMMAND
                                    && success
                                {
                                    let message = serde_json::to_string(&Message::Event(Box::new(
                                        dap_types::messages::Events::Initialized(Some(
                                            Default::default(),
                                        )),
                                    )))
                                    .unwrap();
                                    writer
                                        .write_all(
                                            TransportDelegate::build_rpc_message(message)
                                                .as_bytes(),
                                        )
                                        .await
                                        .unwrap();
                                }

                                writer.flush().await.unwrap();
                            }
                        }
                        Message::Event(event) => {
                            let message = serde_json::to_string(&Message::Event(event)).unwrap();

                            let mut writer = stdout_writer.lock().await;
                            writer
                                .write_all(TransportDelegate::build_rpc_message(message).as_bytes())
                                .await
                                .unwrap();
                            writer.flush().await.unwrap();
                        }
                        Message::Response(response) => {
                            if let Some(handle) =
                                response_handlers.lock().get(response.command.as_str())
                            {
                                handle(response);
                            } else {
                                log::error!("No response handler for {}", response.command);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn start_stdio(cx: &mut AsyncApp) -> Result<Self> {
        let (stdin_writer, stdin_reader) = async_pipe::pipe();
        let (stdout_writer, stdout_reader) = async_pipe::pipe();
        let kind = FakeTransportKind::Stdio {
            stdin_writer: Some(stdin_writer),
            stdout_reader: Some(stdout_reader),
        };

        let mut this = Self {
            request_handlers: Arc::new(Mutex::new(HashMap::default())),
            response_handlers: Arc::new(Mutex::new(HashMap::default())),
            message_handler: None,
            kind,
        };

        let request_handlers = this.request_handlers.clone();
        let response_handlers = this.response_handlers.clone();

        this.message_handler = Some(cx.background_spawn(Self::handle_messages(
            request_handlers,
            response_handlers,
            stdin_reader,
            stdout_writer,
        )));

        Ok(this)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Transport for FakeTransport {
    fn tcp_arguments(&self) -> Option<TcpArguments> {
        match &self.kind {
            FakeTransportKind::Stdio { .. } => None,
            FakeTransportKind::Tcp { connection, .. } => Some(connection.clone()),
        }
    }

    fn connect(
        &mut self,
    ) -> Task<
        Result<(
            Box<dyn AsyncWrite + Unpin + Send + 'static>,
            Box<dyn AsyncRead + Unpin + Send + 'static>,
        )>,
    > {
        let result = match &mut self.kind {
            FakeTransportKind::Stdio {
                stdin_writer,
                stdout_reader,
            } => util::maybe!({
                Ok((
                    Box::new(stdin_writer.take().context("Cannot reconnect")?) as _,
                    Box::new(stdout_reader.take().context("Cannot reconnect")?) as _,
                ))
            }),
            FakeTransportKind::Tcp { executor, .. } => {
                let (stdin_writer, stdin_reader) = async_pipe::pipe();
                let (stdout_writer, stdout_reader) = async_pipe::pipe();

                let request_handlers = self.request_handlers.clone();
                let response_handlers = self.response_handlers.clone();

                self.message_handler = Some(executor.spawn(Self::handle_messages(
                    request_handlers,
                    response_handlers,
                    stdin_reader,
                    stdout_writer,
                )));

                Ok((Box::new(stdin_writer) as _, Box::new(stdout_reader) as _))
            }
        };
        Task::ready(result)
    }

    fn has_adapter_logs(&self) -> bool {
        false
    }

    fn kill(&mut self) {
        self.message_handler.take();
    }

    #[cfg(any(test, feature = "test-support"))]
    fn as_fake(&self) -> &FakeTransport {
        self
    }
}

#[cfg(all(test, unix))]
mod process_cleanup_tests {
    use super::*;
    use gpui::TestAppContext;

    fn adapter_process() -> Child {
        let mut command = util::command::new_std_command("/bin/sleep");
        command.arg("60");
        Child::spawn(command, Stdio::null(), Stdio::null(), Stdio::null())
            .expect("spawn adapter process")
    }

    fn tcp_transport(cx: &TestAppContext) -> (TcpTransport, u32) {
        let process = adapter_process();
        let process_id = process.id();
        (
            TcpTransport {
                executor: cx.background_executor.clone(),
                host: "127.0.0.1".parse().expect("loopback address"),
                port: 0,
                timeout: 100,
                process: Arc::new(Mutex::new(Some(process))),
                _stderr_task: None,
                _stdout_task: None,
            },
            process_id,
        )
    }

    fn stdio_transport(cx: &TestAppContext) -> (StdioTransport, u32) {
        let process = adapter_process();
        let process_id = process.id();
        (
            StdioTransport {
                executor: cx.background_executor.clone(),
                process: Mutex::new(Some(process)),
                _stderr_task: None,
            },
            process_id,
        )
    }

    async fn assert_adapter_reaped(process_id: u32) {
        // Kernel process retirement needs a real deadline; advancing the GPUI
        // test clock cannot give the external child or reaper CPU time.
        let reaped = smol::unblock(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                let result = unsafe { libc::kill(process_id as i32, 0) };
                if result == -1
                    && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
                {
                    return true;
                }
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        })
        .await;
        assert!(reaped, "adapter PID {process_id} remains live or defunct");
    }

    #[gpui::test]
    async fn tcp_adapter_kill_reaps_while_transport_is_retained(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let (mut transport, process_id) = tcp_transport(cx);
        transport.kill();
        transport.kill();
        assert_adapter_reaped(process_id).await;
        drop(transport);
    }

    #[gpui::test]
    async fn tcp_adapter_drop_reaps_process(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let (transport, process_id) = tcp_transport(cx);
        drop(transport);
        assert_adapter_reaped(process_id).await;
    }

    #[gpui::test]
    async fn tcp_adapter_kill_retires_pending_connection(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let (mut transport, process_id) = tcp_transport(cx);
        let connection = transport.connect();
        transport.kill();
        let error = connection.await.err().expect("stopped adapter connection");
        assert!(
            error
                .to_string()
                .contains("was stopped before debugger attached")
        );
        assert_adapter_reaped(process_id).await;
    }

    #[gpui::test]
    async fn stdio_adapter_kill_reaps_while_transport_is_retained(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let (mut transport, process_id) = stdio_transport(cx);
        transport.kill();
        transport.kill();
        assert_adapter_reaped(process_id).await;
        assert!(transport.connect().await.is_err());
        drop(transport);
    }

    #[gpui::test]
    async fn stdio_adapter_drop_reaps_process(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let (transport, process_id) = stdio_transport(cx);
        drop(transport);
        assert_adapter_reaped(process_id).await;
    }
}

#[cfg(test)]
mod diagnostic_debug_tests {
    use super::*;

    #[gpui::test]
    async fn diagnostic_debug_transport_bounds_headers_payloads_and_aggregate() {
        let message = r#"{"seq":1,"type":"event","event":"exited","body":{"exitCode":7}}"#;
        let frame = TransportDelegate::build_rpc_message(message.into());
        let mut reader = BufReader::new(futures::io::Cursor::new(
            format!("{frame}{frame}").into_bytes(),
        ));
        let mut buffer = String::new();
        let mut remaining = Some(frame.len());
        let first = TransportDelegate::receive_server_message_with_limit(
            &mut reader,
            &mut buffer,
            None,
            &mut remaining,
        )
        .await;
        assert!(matches!(
            first,
            ConnectionResult::Result(Ok(Message::Event(_)))
        ));
        assert_eq!(remaining, Some(0));
        let second = TransportDelegate::receive_server_message_with_limit(
            &mut reader,
            &mut buffer,
            None,
            &mut remaining,
        )
        .await;
        assert!(matches!(second, ConnectionResult::Result(Err(_))));
        for invalid in [
            format!("Content-Length: {}\r\n\r\n", usize::MAX),
            "Content-Length: 1000000\r\n\r\n".to_owned(),
            format!("{}: 1\r\n\r\n", "X".repeat(4096)),
            "Content-Length: -1\r\n\r\n".into(),
            "Content-Length: invalid\r\n\r\n".into(),
        ] {
            let mut reader = BufReader::new(futures::io::Cursor::new(invalid.into_bytes()));
            let mut remaining = Some(128);
            assert!(matches!(
                TransportDelegate::receive_server_message_with_limit(
                    &mut reader,
                    &mut buffer,
                    None,
                    &mut remaining
                )
                .await,
                ConnectionResult::Result(Err(_))
            ));
        }
        let mut reader = BufReader::new(futures::io::Cursor::new(frame.into_bytes()));
        assert!(matches!(
            TransportDelegate::receive_server_message(&mut reader, &mut buffer, None).await,
            ConnectionResult::Result(Ok(Message::Event(_)))
        ));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn diagnostic_debug_transport_uses_only_explicit_environment(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let binary = DebugAdapterBinary {
            command: Some("/usr/bin/env".into()),
            arguments: Vec::new(),
            envs: collections::HashMap::from_iter([("DOCTOR_REVIEWED".into(), "selected".into())]),
            cwd: None,
            connection: None,
            request_args: dap_types::StartDebuggingRequestArguments {
                request: dap_types::StartDebuggingRequestArgumentsRequest::Launch,
                configuration: serde_json::json!({}),
            },
        };
        let mut transport = StdioTransport::start(
            &binary,
            Arc::new(Mutex::new(SmallVec::new())),
            true,
            &mut cx.to_async(),
        )
        .await
        .expect("Explicit environment test process");
        let (_, mut output) = transport.connect().await.expect("Process streams");
        let mut bytes = Vec::new();
        output
            .read_to_end(&mut bytes)
            .await
            .expect("Process output");
        assert_eq!(bytes, b"DOCTOR_REVIEWED=selected\n");
    }

    #[gpui::test]
    async fn diagnostic_debug_transport_protocol_errors_exclude_runtime_values() {
        let invalid =
            r#"{"seq":"private-runtime-value","type":"request","command":"pause","arguments":{}}"#;
        let frame = TransportDelegate::build_rpc_message(invalid.into());
        let mut reader = BufReader::new(futures::io::Cursor::new(frame.into_bytes()));
        let mut buffer = String::new();
        let result = TransportDelegate::receive_server_message_with_limit(
            &mut reader,
            &mut buffer,
            None,
            &mut Some(4096),
        )
        .await;
        let ConnectionResult::Result(Err(error)) = result else {
            panic!("Invalid protocol envelope must fail");
        };
        assert!(!error.to_string().contains("private-runtime-value"));
        assert!(error.to_string().contains("invalid protocol message"));
    }

    #[gpui::test]
    async fn diagnostic_debug_transport_refuses_invalid_limit_and_tcp(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut binary = DebugAdapterBinary {
            command: Some("not-executed".into()),
            arguments: vec![],
            envs: Default::default(),
            cwd: None,
            connection: None,
            request_args: dap_types::StartDebuggingRequestArguments {
                request: dap_types::StartDebuggingRequestArgumentsRequest::Launch,
                configuration: serde_json::json!({}),
            },
        };
        for limit in [0, 10 * 1024 * 1024 + 1] {
            assert!(
                crate::client::DebugAdapterClient::start_with_input_limit(
                    crate::client::SessionId(1),
                    binary.clone(),
                    Box::new(|_| {}),
                    Some(limit),
                    &mut cx.to_async()
                )
                .await
                .is_err()
            );
        }
        binary.connection = Some(TcpArguments {
            host: "127.0.0.1".parse().expect("loopback"),
            port: 1,
            timeout: None,
        });
        assert!(
            crate::client::DebugAdapterClient::start_with_input_limit(
                crate::client::SessionId(1),
                binary,
                Box::new(|_| {}),
                Some(1024),
                &mut cx.to_async()
            )
            .await
            .is_err()
        );
    }
}
