use crate::http;
use crate::{
    ClientId, FragmentKind, HostRequestContext, HttpBody, HttpRequest, HttpResponse, InputFragment,
    NativeApiHost, NativeHttpServices, OutboundWireKind, ReconnectProjection,
};
use async_channel::{Receiver, Sender, TrySendError};
use async_tungstenite::tungstenite::{
    Error as WebSocketError, Message, accept_with_config, protocol::WebSocketConfig,
};
use bytes::Bytes;
use sim_types::{CancellationToken, HttpMethod};
use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    str,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

const SOCKET_TIMEOUT: Duration = Duration::from_secs(2);
const LISTENER_POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_REQUEST_TARGET_BYTES: usize = 16 * 1024;
const TRANSPORT_DIAGNOSTIC_CAPACITY: usize = 256;

#[derive(Clone)]
pub struct NativeTlsAcceptor {
    certificate_identity: String,
    config: Arc<rustls::ServerConfig>,
}

impl NativeTlsAcceptor {
    pub fn new(
        certificate_identity: impl Into<String>,
        config: Arc<rustls::ServerConfig>,
    ) -> Result<Self, NativeTransportError> {
        let certificate_identity = certificate_identity.into();
        if certificate_identity.trim().is_empty() {
            return Err(NativeTransportError::InvalidConfiguration(
                "TLS certificate identity cannot be empty".into(),
            ));
        }
        Ok(Self {
            certificate_identity,
            config,
        })
    }
}

#[derive(Clone)]
pub struct NativeApiServerConfig {
    pub bind_address: SocketAddr,
    pub tls: Option<NativeTlsAcceptor>,
    pub maximum_connections: usize,
    pub reconnect_projection: Arc<
        dyn Fn(&str, &ClientId) -> Result<ReconnectProjection, crate::NativeApiHostError>
            + Send
            + Sync
            + 'static,
    >,
}

impl NativeApiServerConfig {
    pub fn new(bind_address: SocketAddr) -> Self {
        Self {
            bind_address,
            tls: None,
            maximum_connections: 128,
            reconnect_projection: Arc::new(|_, _| Ok(ReconnectProjection::default())),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeTransportDiagnostic {
    pub peer_address: Option<SocketAddr>,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum NativeTransportError {
    #[error("invalid native API transport configuration: {0}")]
    InvalidConfiguration(String),
    #[error("native API transport I/O failed: {0}")]
    Io(String),
    #[error("native API TLS failed: {0}")]
    Tls(String),
    #[error("native API HTTP protocol failed: {0}")]
    Http(String),
    #[error("native API WebSocket protocol failed: {0}")]
    WebSocket(String),
    #[error("native API transport thread panicked")]
    ThreadPanicked,
}

impl NativeTransportError {
    fn io(error: io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

trait TransportStream: Read + Write {}
impl<T: Read + Write> TransportStream for T {}

struct ConnectionPermit {
    active_connections: Arc<AtomicUsize>,
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.active_connections.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct NativeApiServer<S>
where
    S: NativeHttpServices,
{
    local_address: SocketAddr,
    shutdown: CancellationToken,
    listener_thread: Option<JoinHandle<()>>,
    connection_threads: Arc<Mutex<Vec<JoinHandle<()>>>>,
    diagnostics: Receiver<NativeTransportDiagnostic>,
    dropped_diagnostics: Arc<AtomicUsize>,
    host: Arc<NativeApiHost<S>>,
}

impl<S> NativeApiServer<S>
where
    S: NativeHttpServices,
{
    pub fn start(
        host: Arc<NativeApiHost<S>>,
        config: NativeApiServerConfig,
    ) -> Result<Self, NativeTransportError> {
        validate_server_config(&host, &config)?;
        let listener = TcpListener::bind(config.bind_address).map_err(NativeTransportError::io)?;
        listener
            .set_nonblocking(true)
            .map_err(NativeTransportError::io)?;
        let local_address = listener.local_addr().map_err(NativeTransportError::io)?;
        let shutdown = CancellationToken::default();
        let active_connections = Arc::new(AtomicUsize::new(0));
        let connection_threads = Arc::new(Mutex::new(Vec::new()));
        let (diagnostic_sender, diagnostics) =
            async_channel::bounded(TRANSPORT_DIAGNOSTIC_CAPACITY);
        let dropped_diagnostics = Arc::new(AtomicUsize::new(0));
        let next_client_id = Arc::new(AtomicU64::new(1));
        let listener_thread = thread::Builder::new()
            .name("sim-native-api-listener".into())
            .spawn({
                let host = host.clone();
                let shutdown = shutdown.clone();
                let connection_threads = connection_threads.clone();
                let dropped_diagnostics = dropped_diagnostics.clone();
                move || {
                    run_listener(
                        listener,
                        host,
                        config,
                        shutdown,
                        active_connections,
                        connection_threads,
                        diagnostic_sender,
                        dropped_diagnostics,
                        next_client_id,
                    );
                }
            })
            .map_err(NativeTransportError::io)?;
        Ok(Self {
            local_address,
            shutdown,
            listener_thread: Some(listener_thread),
            connection_threads,
            diagnostics,
            dropped_diagnostics,
            host,
        })
    }

    pub fn local_address(&self) -> SocketAddr {
        self.local_address
    }

    pub fn take_diagnostics(&self) -> Vec<NativeTransportDiagnostic> {
        let mut diagnostics = Vec::new();
        while let Ok(diagnostic) = self.diagnostics.try_recv() {
            diagnostics.push(diagnostic);
        }
        diagnostics
    }

    pub fn dropped_diagnostic_count(&self) -> usize {
        self.dropped_diagnostics.load(Ordering::Acquire)
    }

    pub fn shutdown(mut self) -> Result<(), NativeTransportError> {
        self.host
            .shutdown("native API transport stopped")
            .map_err(|error| NativeTransportError::Io(error.to_string()))?;
        self.shutdown.cancel();
        if let Some(listener_thread) = self.listener_thread.take() {
            listener_thread
                .join()
                .map_err(|_| NativeTransportError::ThreadPanicked)?;
        }
        let connection_threads = {
            let mut threads = self.connection_threads.lock().map_err(|_| {
                NativeTransportError::Io("connection thread state is unavailable".into())
            })?;
            std::mem::take(&mut *threads)
        };
        for connection_thread in connection_threads {
            connection_thread
                .join()
                .map_err(|_| NativeTransportError::ThreadPanicked)?;
        }
        Ok(())
    }
}

impl<S> Drop for NativeApiServer<S>
where
    S: NativeHttpServices,
{
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

fn validate_server_config<S>(
    host: &NativeApiHost<S>,
    config: &NativeApiServerConfig,
) -> Result<(), NativeTransportError>
where
    S: NativeHttpServices,
{
    if config.maximum_connections == 0 {
        return Err(NativeTransportError::InvalidConfiguration(
            "native API maximum connection count must be non-zero".into(),
        ));
    }
    if config.bind_address.ip() != host.security_config().bind_address {
        return Err(NativeTransportError::InvalidConfiguration(
            "transport bind address must equal the security policy bind address".into(),
        ));
    }
    match (&host.security_config().tls, &config.tls) {
        (crate::security::TlsPolicy::Disabled, None) => Ok(()),
        (crate::security::TlsPolicy::Disabled, Some(_)) => {
            Err(NativeTransportError::InvalidConfiguration(
                "TLS transport was supplied while the security policy disables TLS".into(),
            ))
        }
        (
            crate::security::TlsPolicy::Required {
                certificate_identity,
            },
            Some(tls),
        ) if certificate_identity == &tls.certificate_identity => Ok(()),
        (crate::security::TlsPolicy::Required { .. }, Some(_)) => {
            Err(NativeTransportError::InvalidConfiguration(
                "TLS certificate identity does not match the security policy".into(),
            ))
        }
        (crate::security::TlsPolicy::Required { .. }, None) => {
            Err(NativeTransportError::InvalidConfiguration(
                "the security policy requires an actual TLS acceptor".into(),
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_listener<S>(
    listener: TcpListener,
    host: Arc<NativeApiHost<S>>,
    config: NativeApiServerConfig,
    shutdown: CancellationToken,
    active_connections: Arc<AtomicUsize>,
    connection_threads: Arc<Mutex<Vec<JoinHandle<()>>>>,
    diagnostic_sender: Sender<NativeTransportDiagnostic>,
    dropped_diagnostics: Arc<AtomicUsize>,
    next_client_id: Arc<AtomicU64>,
) where
    S: NativeHttpServices,
{
    while !shutdown.is_cancelled() {
        reap_connection_threads(
            &connection_threads,
            &diagnostic_sender,
            &dropped_diagnostics,
        );
        match listener.accept() {
            Ok((stream, peer_address)) => {
                let maximum_connections = config.maximum_connections;
                let previous = active_connections.fetch_add(1, Ordering::AcqRel);
                if previous >= maximum_connections {
                    active_connections.fetch_sub(1, Ordering::AcqRel);
                    send_diagnostic(
                        &diagnostic_sender,
                        &dropped_diagnostics,
                        Some(peer_address),
                        "connection rejected because the native API concurrency limit was reached",
                    );
                    continue;
                }
                let permit = ConnectionPermit {
                    active_connections: active_connections.clone(),
                };
                let thread = thread::Builder::new()
                    .name("sim-native-api-connection".into())
                    .spawn({
                        let host = host.clone();
                        let shutdown = shutdown.clone();
                        let tls = config.tls.clone();
                        let projection = config.reconnect_projection.clone();
                        let diagnostic_sender = diagnostic_sender.clone();
                        let dropped_diagnostics = dropped_diagnostics.clone();
                        let next_client_id = next_client_id.clone();
                        move || {
                            let _permit = permit;
                            if let Err(error) = handle_connection(
                                stream,
                                peer_address,
                                host,
                                tls,
                                projection,
                                shutdown,
                                next_client_id,
                            ) {
                                send_diagnostic(
                                    &diagnostic_sender,
                                    &dropped_diagnostics,
                                    Some(peer_address),
                                    &error.to_string(),
                                );
                            }
                        }
                    });
                match thread {
                    Ok(thread) => match connection_threads.lock() {
                        Ok(mut threads) => threads.push(thread),
                        Err(_) => send_diagnostic(
                            &diagnostic_sender,
                            &dropped_diagnostics,
                            Some(peer_address),
                            "connection thread could not be tracked",
                        ),
                    },
                    Err(error) => send_diagnostic(
                        &diagnostic_sender,
                        &dropped_diagnostics,
                        Some(peer_address),
                        &format!("failed to start connection thread: {error}"),
                    ),
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(LISTENER_POLL_INTERVAL);
            }
            Err(error) => {
                send_diagnostic(
                    &diagnostic_sender,
                    &dropped_diagnostics,
                    None,
                    &format!("listener accept failed: {error}"),
                );
                thread::sleep(LISTENER_POLL_INTERVAL);
            }
        }
    }
    reap_connection_threads(
        &connection_threads,
        &diagnostic_sender,
        &dropped_diagnostics,
    );
}

fn reap_connection_threads(
    connection_threads: &Mutex<Vec<JoinHandle<()>>>,
    diagnostic_sender: &Sender<NativeTransportDiagnostic>,
    dropped_diagnostics: &AtomicUsize,
) {
    let Ok(mut threads) = connection_threads.lock() else {
        send_diagnostic(
            diagnostic_sender,
            dropped_diagnostics,
            None,
            "connection thread state is unavailable",
        );
        return;
    };
    let mut active = Vec::with_capacity(threads.len());
    for thread in threads.drain(..) {
        if thread.is_finished() {
            if thread.join().is_err() {
                send_diagnostic(
                    diagnostic_sender,
                    dropped_diagnostics,
                    None,
                    "connection thread panicked",
                );
            }
        } else {
            active.push(thread);
        }
    }
    *threads = active;
}

fn send_diagnostic(
    sender: &Sender<NativeTransportDiagnostic>,
    dropped_diagnostics: &AtomicUsize,
    peer_address: Option<SocketAddr>,
    message: &str,
) {
    let diagnostic = NativeTransportDiagnostic {
        peer_address,
        message: message.to_owned(),
    };
    match sender.try_send(diagnostic) {
        Ok(()) => {}
        Err(TrySendError::Full(_) | TrySendError::Closed(_)) => {
            dropped_diagnostics.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn handle_connection<S>(
    stream: TcpStream,
    peer_address: SocketAddr,
    host: Arc<NativeApiHost<S>>,
    tls: Option<NativeTlsAcceptor>,
    reconnect_projection: Arc<
        dyn Fn(&str, &ClientId) -> Result<ReconnectProjection, crate::NativeApiHostError>
            + Send
            + Sync
            + 'static,
    >,
    shutdown: CancellationToken,
    next_client_id: Arc<AtomicU64>,
) -> Result<(), NativeTransportError>
where
    S: NativeHttpServices,
{
    stream
        .set_nonblocking(false)
        .map_err(NativeTransportError::io)?;
    stream
        .set_read_timeout(Some(SOCKET_TIMEOUT))
        .map_err(NativeTransportError::io)?;
    stream
        .set_write_timeout(Some(SOCKET_TIMEOUT))
        .map_err(NativeTransportError::io)?;
    let transport_tls = tls.is_some();
    let mut stream: Box<dyn TransportStream + Send> = match tls {
        Some(tls) => {
            let connection = rustls::ServerConnection::new(tls.config)
                .map_err(|error| NativeTransportError::Tls(error.to_string()))?;
            Box::new(rustls::StreamOwned::new(connection, stream))
        }
        None => Box::new(stream),
    };
    let maximum_header_bytes = host.security_config().limits.maximum_header_bytes;
    let request_bytes = match read_request_head(&mut *stream, maximum_header_bytes) {
        Ok(request_bytes) => request_bytes,
        Err(error) => return write_protocol_error(&mut *stream, error),
    };
    let parsed = match parse_request(
        &request_bytes,
        host.security_config().limits.maximum_header_count,
    ) {
        Ok(parsed) => parsed,
        Err(error) => return write_protocol_error(&mut *stream, error),
    };
    let (path, query) = match parse_target(&parsed.target) {
        Ok(target) => target,
        Err(error) => return write_protocol_error(&mut *stream, error),
    };
    let context = match transport_context(peer_address.ip(), &parsed.headers, transport_tls) {
        Ok(context) => context,
        Err(error) => return write_protocol_error(&mut *stream, error),
    };
    if is_websocket_upgrade(&parsed.headers) && path == "/ws" {
        let client_id = match query.get("clientId").and_then(|values| values.first()) {
            Some(client_id) => ClientId::new(client_id.clone()),
            None => {
                let sequence = next_client_id.fetch_add(1, Ordering::AcqRel);
                ClientId::new(format!("native-{sequence}"))
            }
        }
        .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?;
        let client_id = match host.connect_websocket_projected(
            client_id,
            parsed.headers,
            context,
            |principal, client_id| reconnect_projection(principal, client_id),
        ) {
            Ok(client_id) => client_id,
            Err(error) => {
                let response = error.into_http_response();
                write_http_response(&mut *stream, response)?;
                return Ok(());
            }
        };
        let prefixed = PrefixedStream::new(request_bytes, stream);
        let websocket_limit = host.websocket_limits().max_message_bytes;
        let websocket_config = WebSocketConfig::default()
            .read_buffer_size(websocket_limit.min(128 * 1024))
            .max_message_size(Some(websocket_limit))
            .max_frame_size(Some(websocket_limit));
        let websocket = match accept_with_config(prefixed, Some(websocket_config)) {
            Ok(websocket) => websocket,
            Err(error) => {
                host.disconnect_websocket(&client_id)
                    .map_err(|disconnect_error| {
                        NativeTransportError::WebSocket(disconnect_error.to_string())
                    })?;
                return Err(NativeTransportError::WebSocket(error.to_string()));
            }
        };
        return run_websocket(websocket, host, client_id, shutdown);
    }
    let maximum_body_bytes = host.security_config().limits.maximum_body_bytes;
    let body = match read_request_body(&mut *stream, &parsed, maximum_body_bytes) {
        Ok(body) => body,
        Err(error) => return write_protocol_error(&mut *stream, error),
    };
    if parsed.method.eq_ignore_ascii_case("OPTIONS") {
        let response = host.handle_preflight(&path, &parsed.headers, context);
        return write_http_response(&mut *stream, response);
    }
    let Some(method) = parse_method(&parsed.method) else {
        return write_http_response(
            &mut *stream,
            HttpResponse::json(
                405,
                serde_json::json!({"error":{"code":"method_not_allowed","message":"HTTP method is not supported"}}),
            ),
        );
    };
    let suppress_body = matches!(method, HttpMethod::Head);
    let response = host.handle_http(
        HttpRequest {
            method,
            path,
            query,
            headers: parsed.headers,
            body: Bytes::from(body),
        },
        context,
    );
    write_http_response_with_body_control(&mut *stream, response, suppress_body)
}

fn write_protocol_error(
    stream: &mut dyn TransportStream,
    error: NativeTransportError,
) -> Result<(), NativeTransportError> {
    let NativeTransportError::Http(message) = error else {
        return Err(error);
    };
    let oversized =
        message.contains("exceed") || message.contains("too many") || message.contains("too large");
    let status = if oversized { 413 } else { 400 };
    let code = if oversized {
        "request_too_large"
    } else {
        "malformed_http_request"
    };
    write_http_response(
        stream,
        HttpResponse::json(
            status,
            serde_json::json!({"error":{"code":code,"message":message}}),
        ),
    )
}

fn read_request_head(
    stream: &mut dyn TransportStream,
    maximum_header_bytes: usize,
) -> Result<Vec<u8>, NativeTransportError> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        if find_bytes(&bytes, b"\r\n\r\n").is_some() {
            return Ok(bytes);
        }
        if bytes.len() >= maximum_header_bytes {
            return Err(NativeTransportError::Http(
                "request headers exceed the configured limit".into(),
            ));
        }
        let remaining = maximum_header_bytes.saturating_sub(bytes.len());
        let read_capacity = remaining.min(chunk.len());
        let read = stream
            .read(&mut chunk[..read_capacity])
            .map_err(NativeTransportError::io)?;
        if read == 0 {
            return Err(NativeTransportError::Http(
                "connection closed before the HTTP headers completed".into(),
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
}

#[derive(Debug)]
struct ParsedRequest {
    method: String,
    target: String,
    headers: BTreeMap<String, Vec<String>>,
    buffered_body: Vec<u8>,
}

fn parse_request(
    bytes: &[u8],
    maximum_header_count: usize,
) -> Result<ParsedRequest, NativeTransportError> {
    let mut header_storage = [httparse::EMPTY_HEADER; 256];
    let mut request = httparse::Request::new(&mut header_storage);
    let body_offset = match request
        .parse(bytes)
        .map_err(|error| NativeTransportError::Http(error.to_string()))?
    {
        httparse::Status::Complete(offset) => offset,
        httparse::Status::Partial => {
            return Err(NativeTransportError::Http(
                "HTTP request headers are incomplete".into(),
            ));
        }
    };
    if request.headers.len() > maximum_header_count {
        return Err(NativeTransportError::Http(
            "request has too many headers".into(),
        ));
    }
    let method = request
        .method
        .ok_or_else(|| NativeTransportError::Http("HTTP method is missing".into()))?
        .to_owned();
    let target = request
        .path
        .ok_or_else(|| NativeTransportError::Http("HTTP request target is missing".into()))?
        .to_owned();
    if target.len() > MAX_REQUEST_TARGET_BYTES {
        return Err(NativeTransportError::Http(
            "HTTP request target is too large".into(),
        ));
    }
    let mut headers = BTreeMap::<String, Vec<String>>::new();
    for header in request.headers {
        let value = str::from_utf8(header.value)
            .map_err(|_| NativeTransportError::Http("HTTP header is not UTF-8".into()))?;
        headers
            .entry(header.name.to_ascii_lowercase())
            .or_default()
            .push(value.to_owned());
    }
    const SINGLETON_HEADERS: [&str; 17] = [
        "host",
        "connection",
        "upgrade",
        "content-length",
        "transfer-encoding",
        "authorization",
        "origin",
        "idempotency-key",
        "x-operation-id",
        "x-forwarded-for",
        "x-zed-plugin-profile",
        "x-zed-plugin-id",
        "x-zed-plugin-digest",
        "x-zed-plugin-capabilities",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-protocol",
    ];
    for name in SINGLETON_HEADERS {
        if headers.get(name).is_some_and(|values| values.len() != 1) {
            return Err(NativeTransportError::Http(format!(
                "HTTP header {name} must occur exactly once when present"
            )));
        }
    }
    if headers.contains_key("content-length") && headers.contains_key("transfer-encoding") {
        return Err(NativeTransportError::Http(
            "Content-Length and Transfer-Encoding cannot be combined".into(),
        ));
    }
    if headers
        .get("content-length")
        .and_then(|values| values.first())
        .is_some_and(|value| value.contains(','))
    {
        return Err(NativeTransportError::Http(
            "Content-Length cannot contain a list".into(),
        ));
    }
    Ok(ParsedRequest {
        method,
        target,
        headers,
        buffered_body: bytes.get(body_offset..).unwrap_or_default().to_vec(),
    })
}

fn read_request_body(
    stream: &mut dyn TransportStream,
    request: &ParsedRequest,
    maximum_body_bytes: usize,
) -> Result<Vec<u8>, NativeTransportError> {
    let transfer_encoding = map_header(&request.headers, "transfer-encoding");
    if transfer_encoding.is_some_and(|value| !value.eq_ignore_ascii_case("chunked")) {
        return Err(NativeTransportError::Http(
            "unsupported HTTP transfer encoding".into(),
        ));
    }
    if transfer_encoding.is_some() {
        return read_chunked_body(stream, request.buffered_body.clone(), maximum_body_bytes);
    }
    let content_length = map_header(&request.headers, "content-length")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| NativeTransportError::Http("invalid Content-Length".into()))
        })
        .transpose()?
        .unwrap_or(0);
    if content_length > maximum_body_bytes {
        return Err(NativeTransportError::Http(
            "request body exceeds the configured limit".into(),
        ));
    }
    if request.buffered_body.len() > content_length {
        return Err(NativeTransportError::Http(
            "request includes bytes beyond Content-Length".into(),
        ));
    }
    let mut body = request.buffered_body.clone();
    let mut chunk = [0_u8; 8192];
    while body.len() < content_length {
        let remaining = content_length - body.len();
        let read_capacity = remaining.min(chunk.len());
        let read = stream
            .read(&mut chunk[..read_capacity])
            .map_err(NativeTransportError::io)?;
        if read == 0 {
            return Err(NativeTransportError::Http(
                "connection closed before the request body completed".into(),
            ));
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Ok(body)
}

fn read_chunked_body(
    stream: &mut dyn TransportStream,
    mut buffered: Vec<u8>,
    maximum_body_bytes: usize,
) -> Result<Vec<u8>, NativeTransportError> {
    let mut body = Vec::new();
    let mut cursor = 0;
    loop {
        let line_end = loop {
            if let Some(relative) = find_bytes(&buffered[cursor..], b"\r\n") {
                break cursor + relative;
            }
            read_more(
                stream,
                &mut buffered,
                maximum_body_bytes.saturating_add(64 * 1024),
            )?;
        };
        let size_text = str::from_utf8(&buffered[cursor..line_end])
            .map_err(|_| NativeTransportError::Http("chunk size is not UTF-8".into()))?;
        let size_text = size_text.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|_| NativeTransportError::Http("invalid chunk size".into()))?;
        cursor = line_end + 2;
        if size == 0 {
            while buffered.len() < cursor + 2 {
                read_more(
                    stream,
                    &mut buffered,
                    maximum_body_bytes.saturating_add(64 * 1024),
                )?;
            }
            if buffered.get(cursor..cursor + 2) != Some(b"\r\n") {
                return Err(NativeTransportError::Http(
                    "chunked trailers are not supported".into(),
                ));
            }
            return Ok(body);
        }
        if body.len().saturating_add(size) > maximum_body_bytes {
            return Err(NativeTransportError::Http(
                "chunked request body exceeds the configured limit".into(),
            ));
        }
        let chunk_end = cursor
            .checked_add(size)
            .and_then(|end| end.checked_add(2))
            .ok_or_else(|| NativeTransportError::Http("chunk length overflow".into()))?;
        while buffered.len() < chunk_end {
            read_more(
                stream,
                &mut buffered,
                maximum_body_bytes.saturating_add(64 * 1024),
            )?;
        }
        body.extend_from_slice(&buffered[cursor..cursor + size]);
        if buffered.get(cursor + size..chunk_end) != Some(b"\r\n") {
            return Err(NativeTransportError::Http(
                "chunk data is missing its terminator".into(),
            ));
        }
        cursor = chunk_end;
        if cursor > 64 * 1024 {
            buffered.drain(..cursor);
            cursor = 0;
        }
    }
}

fn read_more(
    stream: &mut dyn TransportStream,
    bytes: &mut Vec<u8>,
    maximum_bytes: usize,
) -> Result<(), NativeTransportError> {
    if bytes.len() >= maximum_bytes {
        return Err(NativeTransportError::Http(
            "chunked request framing exceeds its bound".into(),
        ));
    }
    let mut chunk = [0_u8; 8192];
    let remaining = maximum_bytes - bytes.len();
    let read_capacity = remaining.min(chunk.len());
    let read = stream
        .read(&mut chunk[..read_capacity])
        .map_err(NativeTransportError::io)?;
    if read == 0 {
        return Err(NativeTransportError::Http(
            "connection closed during a chunked request".into(),
        ));
    }
    bytes.extend_from_slice(&chunk[..read]);
    Ok(())
}

fn parse_target(
    target: &str,
) -> Result<(String, BTreeMap<String, Vec<String>>), NativeTransportError> {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    if !path.starts_with('/') {
        return Err(NativeTransportError::Http(
            "HTTP request target must be origin-form".into(),
        ));
    }
    http::decode_uri_component(path, false)
        .map_err(|()| NativeTransportError::Http("invalid encoded request path".into()))?;
    let mut parameters = BTreeMap::<String, Vec<String>>::new();
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        parameters
            .entry(
                http::decode_uri_component(name, true).map_err(|()| {
                    NativeTransportError::Http("invalid encoded query name".into())
                })?,
            )
            .or_default()
            .push(
                http::decode_uri_component(value, true).map_err(|()| {
                    NativeTransportError::Http("invalid encoded query value".into())
                })?,
            );
    }
    Ok((path.to_owned(), parameters))
}

fn transport_context(
    peer_address: IpAddr,
    headers: &BTreeMap<String, Vec<String>>,
    transport_tls: bool,
) -> Result<HostRequestContext, NativeTransportError> {
    let forwarded_for = map_header(headers, "x-forwarded-for")
        .map(|value| value.split(',').next().unwrap_or_default().trim())
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<IpAddr>()
                .map_err(|_| NativeTransportError::Http("invalid X-Forwarded-For".into()))
        })
        .transpose()?;
    let now_epoch_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| NativeTransportError::Io(error.to_string()))?
        .as_secs();
    Ok(HostRequestContext {
        peer_address,
        forwarded_for,
        transport_tls,
        now_epoch_seconds,
    })
}

fn is_websocket_upgrade(headers: &BTreeMap<String, Vec<String>>) -> bool {
    map_header(headers, "upgrade").is_some_and(|value| value.eq_ignore_ascii_case("websocket"))
        && map_header(headers, "connection").is_some_and(|value| {
            value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("upgrade"))
        })
}

fn run_websocket<S>(
    mut websocket: async_tungstenite::tungstenite::WebSocket<PrefixedStream>,
    host: Arc<NativeApiHost<S>>,
    client_id: ClientId,
    shutdown: CancellationToken,
) -> Result<(), NativeTransportError>
where
    S: NativeHttpServices,
{
    let mut close_sent = false;
    let result = (|| {
        loop {
            for message in host
                .drain_websocket(&client_id)
                .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?
            {
                let message = match message.wire_kind {
                    OutboundWireKind::Text => Message::Text(
                        String::from_utf8(message.payload)
                            .map_err(|_| {
                                NativeTransportError::WebSocket(
                                    "native text frame is not UTF-8".into(),
                                )
                            })?
                            .into(),
                    ),
                    OutboundWireKind::Binary => Message::Binary(message.payload.into()),
                    OutboundWireKind::Close => {
                        close_sent = true;
                        Message::Close(None)
                    }
                };
                websocket
                    .send(message)
                    .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?;
            }
            if shutdown.is_cancelled() && !close_sent {
                return Ok(());
            }
            match websocket.read() {
                Ok(Message::Text(_)) if close_sent => {}
                Ok(Message::Text(text)) => {
                    host.process_websocket_fragment(
                        &client_id,
                        InputFragment {
                            kind: FragmentKind::Text,
                            bytes: text.as_bytes().to_vec(),
                            final_fragment: true,
                        },
                    )
                    .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?;
                }
                Ok(Message::Binary(_)) if close_sent => {}
                Ok(Message::Binary(bytes)) => {
                    host.process_websocket_fragment(
                        &client_id,
                        InputFragment {
                            kind: FragmentKind::Binary,
                            bytes: bytes.to_vec(),
                            final_fragment: true,
                        },
                    )
                    .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?;
                }
                Ok(Message::Ping(_)) if close_sent => {}
                Ok(Message::Ping(bytes)) => websocket
                    .send(Message::Pong(bytes))
                    .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?,
                Ok(Message::Pong(_)) => {}
                Ok(Message::Close(_)) => return Ok(()),
                Ok(Message::Frame(_)) => {
                    return Err(NativeTransportError::WebSocket(
                        "unexpected raw WebSocket frame".into(),
                    ));
                }
                Err(WebSocketError::Io(error))
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    if close_sent {
                        return Ok(());
                    }
                }
                Err(WebSocketError::Io(error))
                    if close_sent
                        && matches!(
                            error.kind(),
                            io::ErrorKind::ConnectionAborted
                                | io::ErrorKind::ConnectionReset
                                | io::ErrorKind::UnexpectedEof
                        ) =>
                {
                    return Ok(());
                }
                Err(WebSocketError::ConnectionClosed | WebSocketError::AlreadyClosed) => {
                    return Ok(());
                }
                Err(error) => {
                    return Err(NativeTransportError::WebSocket(error.to_string()));
                }
            }
        }
    })();
    host.disconnect_websocket(&client_id)
        .map_err(|error| NativeTransportError::WebSocket(error.to_string()))?;
    result
}

fn write_http_response(
    stream: &mut dyn TransportStream,
    response: HttpResponse,
) -> Result<(), NativeTransportError> {
    write_http_response_with_body_control(stream, response, false)
}

fn write_http_response_with_body_control(
    stream: &mut dyn TransportStream,
    response: HttpResponse,
    suppress_body: bool,
) -> Result<(), NativeTransportError> {
    let mut headers = response.headers;
    headers.insert("content-type".into(), response.content_type);
    headers.insert("connection".into(), "close".into());
    let reason = reason_phrase(response.status);
    match response.body {
        HttpBody::Empty => {
            headers.insert("content-length".into(), "0".into());
            write_response_head(stream, response.status, reason, &headers)?;
        }
        HttpBody::Bytes(bytes) => {
            headers.insert("content-length".into(), bytes.len().to_string());
            write_response_head(stream, response.status, reason, &headers)?;
            if !suppress_body {
                stream.write_all(&bytes).map_err(NativeTransportError::io)?;
            }
        }
        HttpBody::Json(value) => {
            let body = serde_json::to_vec(&value)
                .map_err(|error| NativeTransportError::Http(error.to_string()))?;
            headers.insert("content-length".into(), body.len().to_string());
            write_response_head(stream, response.status, reason, &headers)?;
            if !suppress_body {
                stream.write_all(&body).map_err(NativeTransportError::io)?;
            }
        }
        HttpBody::Stream(body) => {
            headers.insert("transfer-encoding".into(), "chunked".into());
            write_response_head(stream, response.status, reason, &headers)?;
            if !suppress_body {
                while let Some(chunk) = smol::block_on(body.next()) {
                    let chunk =
                        chunk.map_err(|error| NativeTransportError::Http(error.to_string()))?;
                    write!(stream, "{:X}\r\n", chunk.len()).map_err(NativeTransportError::io)?;
                    stream.write_all(&chunk).map_err(NativeTransportError::io)?;
                    stream
                        .write_all(b"\r\n")
                        .map_err(NativeTransportError::io)?;
                }
                stream
                    .write_all(b"0\r\n\r\n")
                    .map_err(NativeTransportError::io)?;
            }
        }
    }
    stream.flush().map_err(NativeTransportError::io)
}

fn write_response_head(
    stream: &mut dyn TransportStream,
    status: u16,
    reason: &str,
    headers: &BTreeMap<String, String>,
) -> Result<(), NativeTransportError> {
    write!(stream, "HTTP/1.1 {status} {reason}\r\n").map_err(NativeTransportError::io)?;
    for (name, value) in headers {
        if name.contains(['\r', '\n', ':']) || value.contains(['\r', '\n']) {
            return Err(NativeTransportError::Http(
                "response contains an unsafe header".into(),
            ));
        }
        write!(stream, "{name}: {value}\r\n").map_err(NativeTransportError::io)?;
    }
    stream.write_all(b"\r\n").map_err(NativeTransportError::io)
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        206 => "Partial Content",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Content Too Large",
        416 => "Range Not Satisfiable",
        426 => "Upgrade Required",
        428 => "Precondition Required",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Response",
    }
}

fn parse_method(method: &str) -> Option<HttpMethod> {
    match method {
        "GET" => Some(HttpMethod::Get),
        "POST" => Some(HttpMethod::Post),
        "PUT" => Some(HttpMethod::Put),
        "PATCH" => Some(HttpMethod::Patch),
        "DELETE" => Some(HttpMethod::Delete),
        "HEAD" => Some(HttpMethod::Head),
        _ => None,
    }
}

fn map_header<'a>(headers: &'a BTreeMap<String, Vec<String>>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.first())
        .map(String::as_str)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

struct PrefixedStream {
    prefix: io::Cursor<Vec<u8>>,
    stream: Box<dyn TransportStream + Send>,
}

impl PrefixedStream {
    fn new(prefix: Vec<u8>, stream: Box<dyn TransportStream + Send>) -> Self {
        Self {
            prefix: io::Cursor::new(prefix),
            stream,
        }
    }
}

impl Read for PrefixedStream {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.prefix.read(buffer)?;
        if read == 0 {
            self.stream.read(buffer)
        } else {
            Ok(read)
        }
    }
}

impl Write for PrefixedStream {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.stream.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream.flush()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        HttpCapabilities, HttpLimits, NativeServiceRequest, NativeServiceResponse, WebSocketLimits,
        security::{
            ApiSecurityConfig, ApiSecurityError, IdempotencySnapshot, IdempotencySnapshotStore,
            TlsPolicy,
        },
    };

    const TEST_TLS_CERTIFICATE_DER_HEX: &str = concat!(
        "3082017330820118a0030201020209009239e51ef94926b2300a06082a8648ce3d04030230143112301006035504030c",
        "096c6f63616c686f73743020170d3236303731343037343231325a180f32313236303632303037343231325a30143112",
        "301006035504030c096c6f63616c686f73743059301306072a8648ce3d020106082a8648ce3d03010703420004e8a53a",
        "e45a5102a16513b8eca4846b0d7fb839433b9f63455b4f8e3c87a65d917c2cb89b05efe4670c574cc84c8e49501f41",
        "5535c796f2208b7be37c98a1d276a351304f301a0603551d110413301182096c6f63616c686f737487047f000001300c",
        "0603551d130101ff04023000300e0603551d0f0101ff04040302078030130603551d25040c300a06082b060105050703",
        "01300a06082a8648ce3d0403020349003046022100d28feed17c2977f293e9710d60f7d2e8f7030fcfbad64a6b811fe5",
        "9bad22841e02210096c34263e1724068eb1c57d894beef8beda8e95b68400ca908cee911ef72f06c",
    );
    const TEST_TLS_PRIVATE_KEY_DER_HEX: &str = concat!(
        "308187020100301306072a8648ce3d020106082a8648ce3d030107046d306b02010104203fad6b2b66129a76f0de09af",
        "738ada166ebe481204d8fc4c0d189d5dd23a7c8aa14403420004e8a53ae45a5102a16513b8eca4846b0d7fb839433b9f",
        "63455b4f8e3c87a65d917c2cb89b05efe4670c574cc84c8e49501f415535c796f2208b7be37c98a1d276",
    );

    #[derive(Default)]
    struct MemorySnapshotStore {
        snapshot: Mutex<Option<IdempotencySnapshot>>,
    }

    impl IdempotencySnapshotStore for MemorySnapshotStore {
        fn load(&self) -> Result<Option<IdempotencySnapshot>, ApiSecurityError> {
            self.snapshot
                .lock()
                .map(|snapshot| snapshot.clone())
                .map_err(|_| ApiSecurityError::Persistence("test snapshot lock failed".into()))
        }

        fn save(&self, snapshot: &IdempotencySnapshot) -> Result<(), ApiSecurityError> {
            self.snapshot
                .lock()
                .map(|mut stored| *stored = Some(snapshot.clone()))
                .map_err(|_| ApiSecurityError::Persistence("test snapshot lock failed".into()))
        }
    }

    struct ProbeServices;

    impl NativeHttpServices for ProbeServices {
        fn dispatch(
            &self,
            request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, crate::NativeServiceError> {
            Ok(NativeServiceResponse::json(
                200,
                serde_json::json!({
                    "feature_id": request.route.canonical_feature_id,
                    "native": true,
                }),
            ))
        }
    }

    fn test_host(
        security: ApiSecurityConfig,
    ) -> Result<Arc<NativeApiHost<ProbeServices>>, crate::NativeApiHostError> {
        NativeApiHost::new(
            "profile-a",
            Arc::new(ProbeServices),
            HttpLimits::default(),
            HttpCapabilities::default(),
            WebSocketLimits::default(),
            security,
            Arc::new(
                sim_runtime::PermissionPolicy::native_runtime_services("profile-a")
                    .map_err(|error| crate::NativeApiHostError::Runtime(error.to_string()))?,
            ),
            Arc::new(MemorySnapshotStore::default()),
        )
        .map(Arc::new)
    }

    fn decode_hex_fixture(value: &str) -> Result<Vec<u8>, NativeTransportError> {
        if !value.len().is_multiple_of(2) {
            return Err(NativeTransportError::InvalidConfiguration(
                "test TLS fixture has an odd number of hexadecimal digits".into(),
            ));
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| {
                let digits = str::from_utf8(digits).map_err(|error| {
                    NativeTransportError::InvalidConfiguration(error.to_string())
                })?;
                u8::from_str_radix(digits, 16)
                    .map_err(|error| NativeTransportError::InvalidConfiguration(error.to_string()))
            })
            .collect()
    }

    fn test_tls_configs()
    -> Result<(Arc<rustls::ServerConfig>, Arc<rustls::ClientConfig>), Box<dyn std::error::Error>>
    {
        let certificate = rustls::pki_types::CertificateDer::from(decode_hex_fixture(
            TEST_TLS_CERTIFICATE_DER_HEX,
        )?);
        let private_key =
            rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
                decode_hex_fixture(TEST_TLS_PRIVATE_KEY_DER_HEX)?,
            ));
        let server = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        .with_single_cert(vec![certificate.clone()], private_key)?;
        let mut roots = rustls::RootCertStore::empty();
        roots.add(certificate)?;
        let client = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()?
        .with_root_certificates(roots)
        .with_no_client_auth();
        Ok((Arc::new(server), Arc::new(client)))
    }

    fn read_complete_http_response(
        stream: &mut impl Read,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut response = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            let read = match stream.read(&mut chunk) {
                Ok(read) => read,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset
                    ) && complete_http_response_length(&response)?.is_some() =>
                {
                    break;
                }
                Err(error) => return Err(error.into()),
            };
            if read == 0 {
                break;
            }
            response.extend_from_slice(&chunk[..read]);
            if complete_http_response_length(&response)?.is_some() {
                break;
            }
        }
        if complete_http_response_length(&response)?.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "HTTPS response closed before its declared body completed",
            )
            .into());
        }
        Ok(response)
    }

    fn complete_http_response_length(
        response: &[u8],
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let Some(header_index) = find_bytes(response, b"\r\n\r\n") else {
            return Ok(None);
        };
        let header_end = header_index + 4;
        let headers = str::from_utf8(&response[..header_index])?;
        let content_length = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        });
        Ok(content_length
            .filter(|length| response.len() >= header_end + length)
            .map(|length| header_end + length))
    }

    fn connect_test_tls(
        address: SocketAddr,
        client: Arc<rustls::ClientConfig>,
    ) -> Result<rustls::StreamOwned<rustls::ClientConnection, TcpStream>, Box<dyn std::error::Error>>
    {
        let tcp = TcpStream::connect(address)?;
        tcp.set_read_timeout(Some(Duration::from_secs(2)))?;
        tcp.set_write_timeout(Some(Duration::from_secs(2)))?;
        let server_name = rustls::pki_types::ServerName::try_from("localhost")?;
        let connection = rustls::ClientConnection::new(client, server_name)?;
        Ok(rustls::StreamOwned::new(connection, tcp))
    }

    #[test]
    fn parses_percent_encoded_targets_and_duplicate_query_values() {
        let (path, query) =
            parse_target("/view/a%20b?x=1&x=2&name=a+b").expect("target should parse");
        assert_eq!(path, "/view/a%20b");
        assert_eq!(query.get("x"), Some(&vec!["1".into(), "2".into()]));
        assert_eq!(query.get("name"), Some(&vec!["a b".into()]));
        let (literal_plus, _) = parse_target("/view/a+b").expect("literal plus path should parse");
        assert_eq!(literal_plus, "/view/a+b");

        let (encoded_separator, encoded_query) =
            parse_target("/view/a%2Fb%3Fc?name=%252e").expect("encoded target should parse");
        assert_eq!(encoded_separator, "/view/a%2Fb%3Fc");
        assert_eq!(encoded_query.get("name"), Some(&vec!["%2e".into()]));
        assert!(parse_target("/view/%ff").is_err());
        assert!(parse_target("/view/%").is_err());
    }

    #[test]
    fn parses_bounded_chunked_request_body() {
        let mut stream = io::Cursor::new(b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n".to_vec());
        let body =
            read_chunked_body(&mut stream, Vec::new(), 9).expect("chunked body should parse");
        assert_eq!(body, b"Wikipedia");
    }

    #[test]
    pub(crate) fn rejects_ambiguous_http_framing_and_security_headers() {
        let duplicate_authorization = parse_request(
            b"GET /system_stats HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer a\r\nAuthorization: Bearer b\r\n\r\n",
            16,
        )
        .expect_err("duplicate authorization must fail");
        assert!(
            duplicate_authorization
                .to_string()
                .contains("authorization")
        );
        let ambiguous_framing = parse_request(
            b"POST /prompt HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n",
            16,
        )
        .expect_err("ambiguous framing must fail");
        assert!(ambiguous_framing.to_string().contains("cannot be combined"));
        for header in ["Connection", "Upgrade"] {
            let request = format!(
                "GET /ws HTTP/1.1\r\nHost: localhost\r\n{header}: websocket\r\n{header}: keep-alive\r\n\r\n"
            );
            let error = parse_request(request.as_bytes(), 16)
                .expect_err("duplicate WebSocket handshake header must fail");
            assert!(
                error
                    .to_string()
                    .to_ascii_lowercase()
                    .contains(&header.to_ascii_lowercase())
            );
        }
    }

    pub(crate) fn serves_http_and_websocket_over_real_loopback_sockets()
    -> Result<(), Box<dyn std::error::Error>> {
        let host = test_host(ApiSecurityConfig::loopback())?;
        let server = NativeApiServer::start(
            host,
            NativeApiServerConfig::new(SocketAddr::from(([127, 0, 0, 1], 0))),
        )?;

        let mut http = TcpStream::connect(server.local_address())?;
        http.set_read_timeout(Some(Duration::from_secs(10)))?;
        http.write_all(
            b"GET /system_stats HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )?;
        let mut response = Vec::new();
        http.read_to_end(&mut response)?;
        let response = String::from_utf8(response)?;
        assert!(
            response.starts_with("HTTP/1.1 200 OK\r\n"),
            "unexpected native HTTP response: {response:?}"
        );
        assert!(response.contains("\"native\":true"));

        let mut head = TcpStream::connect(server.local_address())?;
        head.set_read_timeout(Some(Duration::from_secs(10)))?;
        head.write_all(
            b"HEAD /api/assets/metadata/test HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )?;
        let mut head_response = Vec::new();
        head.read_to_end(&mut head_response)?;
        let header_end = find_bytes(&head_response, b"\r\n\r\n")
            .ok_or("HEAD response did not contain a complete header block")?;
        assert_eq!(head_response.len(), header_end + 4);

        let mut malformed = TcpStream::connect(server.local_address())?;
        malformed.set_read_timeout(Some(Duration::from_secs(10)))?;
        malformed
            .write_all(b"GET /invalid% HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
        let mut malformed_response = Vec::new();
        malformed.read_to_end(&mut malformed_response)?;
        assert!(malformed_response.starts_with(b"HTTP/1.1 400 Bad Request\r\n"));

        let websocket_stream = TcpStream::connect(server.local_address())?;
        websocket_stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        websocket_stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        let request = format!("ws://{}/ws?clientId=transport-test", server.local_address());
        let (mut websocket, upgrade) =
            async_tungstenite::tungstenite::client(request, websocket_stream)?;
        assert_eq!(upgrade.status().as_u16(), 101);
        let status = websocket.read()?;
        assert!(matches!(status, Message::Text(_)));
        let shutdown = thread::spawn(move || server.shutdown());
        websocket.send(Message::Ping(Vec::new().into()))?;
        let mut received_close = false;
        for _ in 0..3 {
            match websocket.read()? {
                Message::Close(_) => {
                    received_close = true;
                    break;
                }
                Message::Pong(_) => {}
                message => return Err(format!("unexpected shutdown frame {message:?}").into()),
            }
        }
        assert!(received_close);
        shutdown
            .join()
            .map_err(|_| NativeTransportError::ThreadPanicked)??;
        Ok(())
    }

    #[test]
    fn val_cancel_001_transport_shutdown_closes_http_and_websocket_work()
    -> Result<(), Box<dyn std::error::Error>> {
        serves_http_and_websocket_over_real_loopback_sockets()
    }

    #[test]
    pub(crate) fn refuses_to_start_when_tls_policy_has_no_real_acceptor()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut security = ApiSecurityConfig::loopback();
        security.tls = TlsPolicy::Required {
            certificate_identity: "test-certificate".into(),
        };
        let host = test_host(security)?;
        let result = NativeApiServer::start(
            host,
            NativeApiServerConfig::new(SocketAddr::from(([127, 0, 0, 1], 0))),
        );
        assert!(matches!(
            result,
            Err(NativeTransportError::InvalidConfiguration(_))
        ));
        Ok(())
    }

    #[test]
    pub(crate) fn serves_https_with_a_trusted_rustls_handshake_and_rejects_duplicate_headers()
    -> Result<(), Box<dyn std::error::Error>> {
        let (server_tls, client_tls) = test_tls_configs()?;
        let mut security = ApiSecurityConfig::loopback();
        security.tls = TlsPolicy::Required {
            certificate_identity: "localhost".into(),
        };
        let host = test_host(security)?;
        let mut config = NativeApiServerConfig::new(SocketAddr::from(([127, 0, 0, 1], 0)));
        config.tls = Some(NativeTlsAcceptor::new("localhost", server_tls)?);
        let server = NativeApiServer::start(host, config)?;
        let local_address = server.local_address();

        let request_result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let mut https =
                connect_test_tls(local_address, client_tls.clone()).map_err(|error| {
                    io::Error::other(format!("valid HTTPS connection failed: {error}"))
                })?;
            https
                .write_all(
                    b"GET /system_stats HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                )
                .map_err(|error| io::Error::other(format!("valid HTTPS write failed: {error}")))?;
            let response = read_complete_http_response(&mut https).map_err(|error| {
                io::Error::other(format!("valid HTTPS response failed: {error}"))
            })?;
            let response = str::from_utf8(&response)?;
            assert!(
                response.starts_with("HTTP/1.1 200 OK\r\n"),
                "unexpected native HTTPS response: {response:?}"
            );
            assert!(response.contains("\"native\":true"));

            let mut duplicate =
                connect_test_tls(local_address, client_tls.clone()).map_err(|error| {
                    io::Error::other(format!("duplicate-header HTTPS connection failed: {error}"))
                })?;
            duplicate.write_all(
                b"GET /system_stats HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer first\r\nAuthorization: Bearer second\r\nConnection: close\r\n\r\n",
            )
            .map_err(|error| {
                io::Error::other(format!("duplicate-header HTTPS write failed: {error}"))
            })?;
            let duplicate_response =
                read_complete_http_response(&mut duplicate).map_err(|error| {
                    io::Error::other(format!("duplicate-header HTTPS response failed: {error}"))
                })?;
            assert!(duplicate_response.starts_with(b"HTTP/1.1 400 Bad Request\r\n"));

            let mut ambiguous = connect_test_tls(local_address, client_tls).map_err(|error| {
                io::Error::other(format!(
                    "ambiguous-framing HTTPS connection failed: {error}"
                ))
            })?;
            ambiguous.write_all(
                b"POST /prompt HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .map_err(|error| {
                io::Error::other(format!("ambiguous-framing HTTPS write failed: {error}"))
            })?;
            let ambiguous_response =
                read_complete_http_response(&mut ambiguous).map_err(|error| {
                    io::Error::other(format!("ambiguous-framing HTTPS response failed: {error}"))
                })?;
            assert!(ambiguous_response.starts_with(b"HTTP/1.1 400 Bad Request\r\n"));
            Ok(())
        })();
        let diagnostics = server.take_diagnostics();
        let shutdown_result = server.shutdown();
        if let Err(error) = request_result {
            return Err(io::Error::other(format!(
                "{error}; native transport diagnostics: {diagnostics:?}"
            ))
            .into());
        }
        shutdown_result?;
        Ok(())
    }
}
