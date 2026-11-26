use std::str;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering::SeqCst},
};

use anyhow::{Context as _, Result, ensure};
use collections::HashMap;
use futures::{
    AsyncBufReadExt, AsyncRead, AsyncReadExt as _, SinkExt as _,
    channel::mpsc::{Receiver, Sender, channel},
    io::BufReader,
};
use gpui::{BackgroundExecutor, Task};
use log::warn;
use parking_lot::Mutex;

use crate::{
    AnyResponse, CONTENT_LEN_HEADER, IoHandler, IoKind, NotificationOrRequest, RequestId,
    ResponseHandler,
};

const HEADER_DELIMITER: &[u8; 4] = b"\r\n\r\n";

/// Bounds the number of incoming LSP messages buffered between the background
/// reader and the foreground dispatcher. When the queue is full, the reader
/// stops reading the server's stdout, letting the OS pipe apply backpressure
/// to the server instead of buffering messages in memory without limit while
/// the foreground thread is unresponsive.
pub(crate) const INCOMING_MESSAGE_QUEUE_CAPACITY: usize = 128;

/// Handler for stdout of language server.
pub struct LspStdoutHandler {
    pub(super) loop_handle: Task<Result<()>>,
    pub(super) incoming_messages: Receiver<NotificationOrRequest>,
}

async fn read_headers<Stdout>(reader: &mut BufReader<Stdout>, buffer: &mut Vec<u8>) -> Result<bool>
where
    Stdout: AsyncRead + Unpin + Send + 'static,
{
    while !buffer.ends_with(HEADER_DELIMITER) {
        if reader.read_until(b'\n', buffer).await? == 0 {
            ensure!(buffer.is_empty(), "truncated LSP message header");
            return Ok(false);
        }
    }
    Ok(true)
}

impl LspStdoutHandler {
    pub fn new<Input>(
        stdout: Input,
        response_handlers: Arc<Mutex<Option<HashMap<RequestId, ResponseHandler>>>>,
        io_handlers: Arc<Mutex<HashMap<i32, IoHandler>>>,
        shutdown_acknowledged: Arc<AtomicBool>,
        cx: BackgroundExecutor,
    ) -> Self
    where
        Input: AsyncRead + Unpin + Send + 'static,
    {
        let (tx, notifications_channel) = channel(INCOMING_MESSAGE_QUEUE_CAPACITY);
        let loop_handle = cx.spawn(Self::handler(
            stdout,
            tx,
            response_handlers,
            io_handlers,
            shutdown_acknowledged,
        ));
        Self {
            loop_handle,
            incoming_messages: notifications_channel,
        }
    }

    async fn handler<Input>(
        stdout: Input,
        mut notifications_sender: Sender<NotificationOrRequest>,
        response_handlers: Arc<Mutex<Option<HashMap<RequestId, ResponseHandler>>>>,
        io_handlers: Arc<Mutex<HashMap<i32, IoHandler>>>,
        shutdown_acknowledged: Arc<AtomicBool>,
    ) -> anyhow::Result<()>
    where
        Input: AsyncRead + Unpin + Send + 'static,
    {
        let mut stdout = BufReader::new(stdout);

        let mut buffer = Vec::new();

        loop {
            buffer.clear();

            if !read_headers(&mut stdout, &mut buffer).await? {
                ensure!(
                    shutdown_acknowledged.load(SeqCst),
                    "language server stdout closed unexpectedly"
                );
                return Ok(());
            }

            let headers = std::str::from_utf8(&buffer)?;

            let message_len = headers
                .split('\n')
                .find(|line| line.starts_with(CONTENT_LEN_HEADER))
                .and_then(|line| line.strip_prefix(CONTENT_LEN_HEADER))
                .with_context(|| format!("invalid LSP message header {headers:?}"))?
                .trim_end()
                .parse()?;

            buffer.resize(message_len, 0);
            stdout
                .read_exact(&mut buffer)
                .await
                .context("cannot read LSP message body")?;

            if let Ok(message) = str::from_utf8(&buffer) {
                log::trace!("incoming message: {message}");
                for handler in io_handlers.lock().values_mut() {
                    handler(IoKind::StdOut, message);
                }
            }

            if let Ok(msg) = serde_json::from_slice::<NotificationOrRequest>(&buffer) {
                notifications_sender.send(msg).await?;
            } else if let Ok(AnyResponse {
                id, error, result, ..
            }) = serde_json::from_slice(&buffer)
            {
                let handler = {
                    response_handlers
                        .lock()
                        .as_mut()
                        .and_then(|handlers| handlers.remove(&id))
                };
                if let Some(handler) = handler {
                    if let Some(error) = error {
                        handler(Err(error)).await;
                    } else if let Some(result) = result {
                        handler(Ok(result.get().into())).await;
                    } else {
                        handler(Ok("null".into())).await;
                    }
                }
            } else {
                warn!(
                    "failed to deserialize LSP message:\n{}",
                    std::str::from_utf8(&buffer)?
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{AsyncWriteExt as _, StreamExt as _};
    use gpui::TestAppContext;

    #[gpui::test]
    async fn test_backpressure_when_messages_are_not_consumed(cx: &mut TestAppContext) {
        let total_messages = INCOMING_MESSAGE_QUEUE_CAPACITY * 4;
        let (mut writer, reader) = async_pipe::pipe();
        let mut handler = LspStdoutHandler::new(
            reader,
            Arc::new(Mutex::new(Some(HashMap::default()))),
            Arc::new(Mutex::new(HashMap::default())),
            Arc::new(AtomicBool::new(false)),
            cx.background_executor.clone(),
        );

        cx.background_executor
            .spawn(async move {
                let payload = r#"{"jsonrpc":"2.0","method":"test/notification","params":{}}"#;
                let message = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);
                for _ in 0..total_messages {
                    writer.write_all(message.as_bytes()).await.unwrap();
                }
            })
            .detach();

        cx.run_until_parked();
        let mut received = 0;
        while handler.incoming_messages.try_recv().is_ok() {
            received += 1;
        }
        assert!(
            received < total_messages,
            "the reader buffered all {total_messages} messages while the consumer was wedged"
        );
        assert!(
            received <= INCOMING_MESSAGE_QUEUE_CAPACITY + 2,
            "expected at most {} buffered messages, got {received}",
            INCOMING_MESSAGE_QUEUE_CAPACITY + 2
        );

        while received < total_messages {
            assert!(
                handler.incoming_messages.next().await.is_some(),
                "the message stream ended after {received} of {total_messages} messages"
            );
            received += 1;
        }
    }

    #[gpui::test]
    async fn test_header_read_distinguishes_frame_boundary_eof() {
        let mut reader = BufReader::new(b"" as &[u8]);
        let mut buffer = Vec::new();
        assert!(
            !read_headers(&mut reader, &mut buffer)
                .await
                .expect("read frame boundary")
        );

        let mut reader = BufReader::new(b"Content-Length: 12\r\n" as &[u8]);
        assert!(read_headers(&mut reader, &mut buffer).await.is_err());
    }

    #[gpui::test]
    async fn test_stdout_eof_preserves_shutdown_and_protocol_errors(cx: &mut TestAppContext) {
        let payload = r#"{"jsonrpc":"2.0","method":"test/notification"}"#;
        let frame = format!("Content-Length: {}\r\n\r\n{payload}", payload.len());
        for (bytes, shutdown_acknowledged, error) in [
            (frame.as_bytes(), true, None),
            (
                frame.as_bytes(),
                false,
                Some("language server stdout closed unexpectedly"),
            ),
            (
                b"Content-Length: 12\r\n",
                true,
                Some("truncated LSP message header"),
            ),
            (
                b"Content-Length: 12\r\n\r\n{}",
                true,
                Some("cannot read LSP message body"),
            ),
        ] {
            let mut handler = LspStdoutHandler::new(
                futures::io::Cursor::new(bytes.to_vec()),
                Arc::new(Mutex::new(Some(HashMap::default()))),
                Arc::new(Mutex::new(HashMap::default())),
                Arc::new(AtomicBool::new(shutdown_acknowledged)),
                cx.background_executor.clone(),
            );
            if bytes == frame.as_bytes() {
                let message = handler.incoming_messages.next().await.unwrap();
                assert_eq!(message.method, "test/notification");
            }
            assert!(handler.incoming_messages.next().await.is_none());
            match error {
                Some(expected) => {
                    assert_eq!(handler.loop_handle.await.unwrap_err().to_string(), expected)
                }
                None => handler.loop_handle.await.unwrap(),
            }
        }
    }

    #[gpui::test]
    async fn test_read_headers() {
        let mut buf = Vec::new();
        let mut reader = BufReader::new(b"Content-Length: 123\r\n\r\n" as &[u8]);
        read_headers(&mut reader, &mut buf).await.unwrap();
        assert_eq!(buf, b"Content-Length: 123\r\n\r\n");

        let mut buf = Vec::new();
        let mut reader = BufReader::new(b"Content-Type: application/vscode-jsonrpc\r\nContent-Length: 1235\r\n\r\n{\"somecontent\":123}" as &[u8]);
        read_headers(&mut reader, &mut buf).await.unwrap();
        assert_eq!(
            buf,
            b"Content-Type: application/vscode-jsonrpc\r\nContent-Length: 1235\r\n\r\n"
        );

        let mut buf = Vec::new();
        let mut reader = BufReader::new(b"Content-Length: 1235\r\nContent-Type: application/vscode-jsonrpc\r\n\r\n{\"somecontent\":true}" as &[u8]);
        read_headers(&mut reader, &mut buf).await.unwrap();
        assert_eq!(
            buf,
            b"Content-Length: 1235\r\nContent-Type: application/vscode-jsonrpc\r\n\r\n"
        );
    }
}
