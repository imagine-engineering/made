/// todo(windows)
/// The tests in this file assume that server_cx is running on Windows too.
/// We neead to find a way to test Windows-Non-Windows interactions.
use crate::headless_project::HeadlessProject;
use agent::{
    AgentTool, NativeAgent, NativeAgentConnection, ReadFileTool, ReadFileToolInput, SkillTool,
    SkillToolInput, SkillToolOutput, Templates, ThreadStore, ToolCallEventStream, ToolInput,
    skill_body_resolver_for_project, skills_resolver_for_project,
};
use client::{Client, UserStore};
use clock::FakeSystemClock;
use collections::{HashMap, HashSet};
use language_model::{LanguageModelRegistry, LanguageModelToolResultContent};
use languages::rust_lang;

use editor::{
    Editor, SelectionEffects,
    actions::{ConfirmCodeAction, ToggleCodeActions},
    code_context_menus::CodeContextMenu,
};
use extension::ExtensionHostProxy;
use fs::{FakeFs, Fs};
use git::{
    Oid,
    repository::{CommitData, GitCommitTemplate, RepoPath, Worktree as GitWorktree},
};
use gpui::{
    AppContext as _, Entity, ImageSource, IntoElement as _, SharedString, TestAppContext,
    UpdateGlobal, VisualContext, img, px, size,
};
use http_client::{BlockedHttpClient, FakeHttpClient};
use language::{
    Buffer, FakeLspAdapter, LanguageConfig, LanguageMatcher, LanguageRegistry, LineEnding, Point,
    language_settings::{AllLanguageSettings, ConfiguredLanguageServer, LanguageSettings},
};
use lsp::{
    CompletionContext, CompletionResponse, CompletionTriggerKind, DEFAULT_LSP_REQUEST_TIMEOUT,
    LanguageServerName,
};
use node_runtime::NodeRuntime;
use project::{
    ProgressToken, Project, ProjectPath,
    agent_server_store::AgentServerCommand,
    image_store,
    search::{SearchQuery, SearchResult},
};
use remote::RemoteClient;
use rpc::proto;
use serde_json::json;
use settings::{Settings, SettingsLocation, SettingsStore, initial_server_settings_content};
use smol::stream::StreamExt;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use unindent::Unindent as _;
use util::{path, path_list::PathList, paths::PathMatcher, rel_path::rel_path};

mod native_input_lifetime_tests {
    use super::*;
    use crate::{NativeConnectionEnd, NativeProjectTransport, handle_native_connection};
    use futures::{Stream, channel::mpsc};
    use proto::EnvelopedMessage as _;
    use std::{
        io::{Cursor, Read as _},
        pin::Pin,
        sync::atomic::{AtomicBool, Ordering},
        task::{Context, Poll},
    };

    struct HeldInput {
        chunks: mpsc::UnboundedReceiver<Vec<u8>>,
        current: Cursor<Vec<u8>>,
        entered: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    }

    impl futures::AsyncRead for HeldInput {
        fn poll_read(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buffer: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            if buffer.is_empty() {
                return Poll::Ready(Ok(0));
            }
            loop {
                if self.current.position() < self.current.get_ref().len() as u64 {
                    return Poll::Ready(self.current.read(buffer));
                }
                match Pin::new(&mut self.chunks).poll_next(context) {
                    Poll::Ready(Some(bytes)) => self.current = Cursor::new(bytes),
                    Poll::Ready(None) => return Poll::Ready(Ok(0)),
                    Poll::Pending => {
                        self.entered.store(true, Ordering::SeqCst);
                        return Poll::Pending;
                    }
                }
            }
        }
    }

    impl Drop for HeldInput {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    struct BrokenOutput;

    impl futures::AsyncWrite for BrokenOutput {
        fn poll_write(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            _buffer: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Poll::Ready(Err(std::io::ErrorKind::BrokenPipe.into()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    enum Termination {
        BrokenOutput,
        BrokenStderr,
        ClosedIncoming,
        MalformedInput,
        Quit,
    }

    struct NativeConnectionFixture {
        input_sender: Option<mpsc::UnboundedSender<Vec<u8>>>,
        incoming_receiver: Option<mpsc::UnboundedReceiver<(proto::PeerId, proto::Envelope)>>,
        outgoing_sender: mpsc::UnboundedSender<proto::Envelope>,
        quit_sender: mpsc::UnboundedSender<()>,
        log_sender: smol::channel::Sender<Vec<u8>>,
        task: Option<gpui::Task<anyhow::Result<NativeConnectionEnd>>>,
        entered: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    }

    impl NativeConnectionFixture {
        fn new(cx: &mut TestAppContext) -> Self {
            Self::with_scope(proto::PeerId { owner_id: 0, id: 1 }, cx)
        }

        fn with_scope(connection_id: proto::PeerId, cx: &mut TestAppContext) -> Self {
            let (input_sender, chunks) = mpsc::unbounded();
            let entered = Arc::new(AtomicBool::new(false));
            let dropped = Arc::new(AtomicBool::new(false));
            let reader = HeldInput {
                chunks,
                current: Cursor::new(Vec::new()),
                entered: entered.clone(),
                dropped: dropped.clone(),
            };
            let (incoming_sender, incoming_receiver) = mpsc::unbounded();
            let (outgoing_sender, mut outgoing_receiver) = mpsc::unbounded();
            let (quit_sender, mut quit_receiver) = mpsc::unbounded();
            let (log_sender, log_receiver) = smol::channel::unbounded();
            let task = cx.update(|cx| {
                cx.spawn(async move |cx| {
                    handle_native_connection(
                        connection_id,
                        reader,
                        BrokenOutput,
                        BrokenOutput,
                        &incoming_sender,
                        &mut outgoing_receiver,
                        &mut quit_receiver,
                        &log_receiver,
                        cx,
                    )
                    .await
                })
            });
            Self {
                input_sender: Some(input_sender),
                incoming_receiver: Some(incoming_receiver),
                outgoing_sender,
                quit_sender,
                log_sender,
                task: Some(task),
                entered,
                dropped,
            }
        }

        async fn wait(&mut self) -> anyhow::Result<NativeConnectionEnd> {
            self.task.take().expect("Connection task").await
        }

        fn cleanup(&mut self, cx: &mut TestAppContext) {
            drop(self.input_sender.take());
            cx.run_until_parked();
            assert!(
                self.dropped.load(Ordering::SeqCst),
                "Fixture cleanup releases input"
            );
        }
    }

    async fn assert_reader_released(cx: &mut TestAppContext, termination: Termination) {
        let mut fixture = NativeConnectionFixture::new(cx);
        cx.run_until_parked();
        assert!(
            fixture.entered.load(Ordering::SeqCst),
            "Native read is held"
        );
        assert!(!fixture.dropped.load(Ordering::SeqCst));
        match termination {
            Termination::BrokenOutput => fixture
                .outgoing_sender
                .unbounded_send(proto::Ping {}.into_envelope(1, None, None))
                .expect("Trigger output failure"),
            Termination::BrokenStderr => fixture
                .log_sender
                .send(vec![1])
                .await
                .expect("Trigger stderr failure"),
            Termination::ClosedIncoming => {
                drop(fixture.incoming_receiver.take());
                let mut stream = futures::io::Cursor::new(Vec::new());
                remote::protocol::write_message(
                    &mut stream,
                    &mut Vec::new(),
                    proto::Ping {}.into_envelope(1, None, None),
                )
                .await
                .expect("Encode production frame");
                fixture
                    .input_sender
                    .as_ref()
                    .expect("Input sender")
                    .unbounded_send(stream.into_inner())
                    .expect("Deliver production frame");
            }
            Termination::MalformedInput => fixture
                .input_sender
                .as_ref()
                .expect("Input sender")
                .unbounded_send(vec![1, 0, 0, 0, 0xff])
                .expect("Deliver invalid one-byte protobuf frame"),
            Termination::Quit => fixture
                .quit_sender
                .unbounded_send(())
                .expect("Quit connection"),
        }
        cx.run_until_parked();
        let result = fixture.wait().await;
        match termination {
            Termination::BrokenOutput | Termination::BrokenStderr | Termination::MalformedInput => {
                assert_eq!(
                    result.expect("Connection closes"),
                    NativeConnectionEnd::Disconnected
                );
            }
            Termination::ClosedIncoming => assert!(result.is_err()),
            Termination::Quit => {
                assert_eq!(result.expect("Quit observed"), NativeConnectionEnd::Quit)
            }
        }
        cx.run_until_parked();
        let released_before_cleanup = fixture.dropped.load(Ordering::SeqCst);
        fixture.cleanup(cx);
        assert!(
            released_before_cleanup,
            "Terminated native connection retained its held input reader"
        );
    }

    #[gpui::test]
    async fn native_connection_retains_receiving_scope_in_queued_frames(cx: &mut TestAppContext) {
        let transport = cx.new(|_| NativeProjectTransport::default());
        let first_scope = transport
            .update(cx, |transport, cx| transport.accept(cx))
            .expect("First scope");
        let mut first = NativeConnectionFixture::with_scope(first_scope, cx);
        let relay = proto::PeerId {
            owner_id: 9,
            id: 99,
        };
        let mut stream = futures::io::Cursor::new(Vec::new());
        remote::protocol::write_message(
            &mut stream,
            &mut Vec::new(),
            proto::Ping {}.into_envelope(17, None, Some(relay)),
        )
        .await
        .expect("Encode frame with untrusted relay");
        first
            .input_sender
            .as_ref()
            .expect("First input")
            .unbounded_send(stream.into_inner())
            .expect("Queue old input");
        cx.run_until_parked();
        transport.update(cx, |transport, cx| transport.disconnect(cx));
        let second_scope = transport
            .update(cx, |transport, cx| transport.accept(cx))
            .expect("Second scope");
        assert_ne!(first_scope, second_scope);
        assert_ne!(first_scope.id, 0);
        assert_ne!(second_scope.id, 0);
        let mut second = NativeConnectionFixture::with_scope(second_scope, cx);
        let mut stream = futures::io::Cursor::new(Vec::new());
        remote::protocol::write_message(
            &mut stream,
            &mut Vec::new(),
            proto::Ping {}.into_envelope(18, None, Some(first_scope)),
        )
        .await
        .expect("Encode second frame");
        second
            .input_sender
            .as_ref()
            .expect("Second input")
            .unbounded_send(stream.into_inner())
            .expect("Queue current input");
        // Delay consuming the application queue until the next connection owns the transport.
        let (scope, envelope) = first
            .incoming_receiver
            .as_mut()
            .expect("First queue")
            .next()
            .await
            .expect("Queued frame");
        assert_eq!(scope, first_scope);
        assert_eq!(envelope.original_sender_id, Some(relay));
        assert_eq!(envelope.id, 17);
        assert_eq!(
            transport.read_with(cx, |transport, _| transport.connection_id),
            Some(second_scope)
        );
        let (scope, envelope) = second
            .incoming_receiver
            .as_mut()
            .expect("Second queue")
            .next()
            .await
            .expect("Current frame");
        assert_eq!(scope, second_scope);
        assert_eq!(envelope.original_sender_id, Some(first_scope));
        assert_eq!(envelope.id, 18);
        assert!(
            transport
                .update(cx, |transport, cx| transport.accept(cx))
                .is_err()
        );
        assert_eq!(
            transport.read_with(cx, |transport, _| transport.connection_id),
            Some(second_scope)
        );
        first.quit_sender.unbounded_send(()).expect("Quit first");
        second.quit_sender.unbounded_send(()).expect("Quit second");
        assert_eq!(
            first.wait().await.expect("First ends"),
            NativeConnectionEnd::Quit
        );
        assert_eq!(
            second.wait().await.expect("Second ends"),
            NativeConnectionEnd::Quit
        );
        first.cleanup(cx);
        second.cleanup(cx);
    }

    #[gpui::test]
    fn native_connection_identity_overflow_preserves_transport(cx: &mut TestAppContext) {
        let transport = cx.new(|_| NativeProjectTransport {
            last_connection_id: u32::MAX - 1,
            connection_id: None,
        });
        let events = cx.new(|_| Vec::new());
        let subscription = cx.update(|cx| {
            cx.subscribe(&transport, {
                let events = events.clone();
                move |_, connected, cx| events.update(cx, |events, _| events.push(*connected))
            })
        });
        let last = transport
            .update(cx, |transport, cx| transport.accept(cx))
            .expect("Last valid identity");
        assert_eq!(last.id, u32::MAX);
        transport.update(cx, |transport, cx| transport.disconnect(cx));
        assert!(
            transport
                .update(cx, |transport, cx| transport.accept(cx))
                .is_err()
        );
        assert_eq!(
            transport.read_with(cx, |transport, _| transport.connection_id),
            None
        );
        assert_eq!(
            transport.read_with(cx, |transport, _| transport.last_connection_id),
            u32::MAX
        );
        assert_eq!(
            events.read_with(cx, |events, _| events.clone()),
            vec![true, false]
        );
        drop(subscription);
    }

    #[gpui::test]
    async fn native_connection_output_failure_releases_input(cx: &mut TestAppContext) {
        assert_reader_released(cx, Termination::BrokenOutput).await;
    }

    #[gpui::test]
    async fn native_connection_stderr_failure_releases_input(cx: &mut TestAppContext) {
        assert_reader_released(cx, Termination::BrokenStderr).await;
    }

    #[gpui::test]
    async fn native_connection_forward_failure_releases_input(cx: &mut TestAppContext) {
        assert_reader_released(cx, Termination::ClosedIncoming).await;
    }

    #[gpui::test]
    async fn native_connection_malformed_input_closes_connection(cx: &mut TestAppContext) {
        assert_reader_released(cx, Termination::MalformedInput).await;
    }

    #[gpui::test]
    async fn native_connection_quit_releases_input(cx: &mut TestAppContext) {
        assert_reader_released(cx, Termination::Quit).await;
    }

    #[gpui::test]
    async fn native_connection_termination_preserves_other_input(cx: &mut TestAppContext) {
        let mut first = NativeConnectionFixture::new(cx);
        let mut second = NativeConnectionFixture::new(cx);
        cx.run_until_parked();
        assert!(first.entered.load(Ordering::SeqCst) && second.entered.load(Ordering::SeqCst));
        first
            .quit_sender
            .unbounded_send(())
            .expect("Quit first connection");
        cx.run_until_parked();
        assert_eq!(
            first.wait().await.expect("First quit"),
            NativeConnectionEnd::Quit
        );
        cx.run_until_parked();
        let first_released_before_cleanup = first.dropped.load(Ordering::SeqCst);
        let second_still_held = !second.dropped.load(Ordering::SeqCst);
        first.cleanup(cx);
        second
            .quit_sender
            .unbounded_send(())
            .expect("Quit second connection");
        cx.run_until_parked();
        assert_eq!(
            second.wait().await.expect("Second quit"),
            NativeConnectionEnd::Quit
        );
        second.cleanup(cx);
        assert!(
            second_still_held,
            "Terminating first connection stopped another input reader"
        );
        assert!(
            first_released_before_cleanup,
            "First connection retained its held input reader"
        );
    }
}

#[gpui::test]
async fn scoped_native_buffer_routes_preserve_lsp_and_close_ownership(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/scoped-buffer"),
        json!({
            "ordinary.txt": "ordinary",
            "bounded.txt": "bounded",
            "search.txt": "scoped-buffer-regression-needle",
        }),
    )
    .await;
    for context in [&mut *cx, &mut *server_cx] {
        context.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
    }
    let peer = proto::PeerId {
        owner_id: 0,
        id: 17,
    };
    let (options, session, guard, _) =
        RemoteClient::fake_server_scoped(peer, cx, server_cx).expect("scoped native connection");
    server_cx.update(HeadlessProject::init);
    let headless = server_cx.new(|cx| {
        HeadlessProject::new(
            crate::HeadlessAppState {
                session,
                fs: fs.clone(),
                http_client: Arc::new(BlockedHttpClient),
                node_runtime: NodeRuntime::unavailable(),
                languages: Arc::new(LanguageRegistry::new(cx.background_executor().clone())),
                extension_host_proxy: Arc::new(ExtensionHostProxy::new()),
                startup_time: std::time::Instant::now(),
            },
            false,
            cx,
        )
    });
    drop(guard);
    let remote = RemoteClient::connect_mock(options, cx).await;
    let client = remote.read_with(cx, |remote, _| remote.proto_client());
    let project = build_project(remote, cx);
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/scoped-buffer"), true, cx)
        })
        .await
        .expect("scoped buffer worktree");
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
    let buffer_store = headless.read_with(server_cx, |headless, _| headless.buffer_store.clone());
    let closed_buffers = server_cx.new(|_| Vec::new());
    let _close_subscription = closed_buffers.update(server_cx, |_, cx| {
        cx.subscribe(&buffer_store, |closed, _, event, _| {
            if let project::buffer_store::BufferStoreEvent::SharedBufferClosed(peer, buffer) = event
            {
                closed.push((*peer, *buffer));
            }
        })
    });
    cx.run_until_parked();
    assert!(!buffer_store.read_with(server_cx, |store, _| store.has_shared_buffers()));
    for route in 0..5 {
        let buffer = match route {
            0 => project
                .update(cx, |project, cx| {
                    project.open_buffer((worktree_id, rel_path("ordinary.txt")), cx)
                })
                .await
                .expect("ordinary scoped buffer"),
            1 => project
                .update(cx, |project, cx| project.create_buffer(None, true, cx))
                .await
                .expect("new scoped buffer"),
            2 => project
                .update(cx, |project, cx| {
                    project.open_buffer_with_limit(
                        ProjectPath {
                            worktree_id,
                            path: rel_path("bounded.txt").into(),
                        },
                        1024,
                        cx,
                    )
                })
                .await
                .expect("bounded scoped buffer"),
            3 => project
                .update(cx, |project, cx| project.open_server_settings(cx))
                .await
                .expect("scoped server settings buffer"),
            _ => {
                let mut buffers = do_search_and_assert(
                    &project,
                    "scoped-buffer-regression-needle",
                    Default::default(),
                    false,
                    &[path!("scoped-buffer/search.txt")],
                    cx.clone(),
                )
                .await;
                assert_eq!(buffers.len(), 1);
                buffers.pop().expect("one scoped search buffer")
            }
        };
        let expected_text = match route {
            0 => "ordinary".to_string(),
            1 => String::new(),
            2 => "bounded".to_string(),
            3 => initial_server_settings_content().replace("\r\n", "\n"),
            _ => "scoped-buffer-regression-needle".to_string(),
        };
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            expected_text
        );
        let buffer_id = buffer.read_with(cx, |buffer, _| buffer.remote_id());
        assert!(buffer_store.read_with(server_cx, |store, _| store.has_shared_buffers()));
        client
            .request(proto::RegisterBufferWithLanguageServers {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                buffer_id: buffer_id.to_proto(),
                only_servers: Vec::new(),
            })
            .await
            .expect("actual scoped LSP handle registration Ack");
        cx.update(|_| drop(buffer));
        cx.run_until_parked();
        closed_buffers.read_with(server_cx, |closed, _| {
            assert_eq!(closed.len(), route + 1);
            assert_eq!(closed.last(), Some(&(peer, buffer_id)));
        });
        assert!(!buffer_store.read_with(server_cx, |store, _| store.has_shared_buffers()));
    }
}

#[gpui::test]
async fn test_basic_remote_editing(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
            "project2": {
                "README.md": "# project 2",
            },
        }),
    )
    .await;
    fs.set_index_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", "fn one() -> usize { 0 }".into())],
    );

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    // The client sees the worktree's contents.
    cx.executor().run_until_parked();
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
    worktree.update(cx, |worktree, _cx| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            vec![
                rel_path("README.md"),
                rel_path("src"),
                rel_path("src/lib.rs"),
            ]
        );
    });

    // The user opens a buffer in the remote worktree. The buffer's
    // contents are loaded from the remote filesystem.
    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    let diff = project
        .update(cx, |project, cx| {
            project.open_unstaged_diff(buffer.clone(), cx)
        })
        .await
        .unwrap();

    diff.update(cx, |diff, cx| {
        assert_eq!(
            diff.base_text_string(cx).unwrap(),
            "fn one() -> usize { 0 }"
        );
    });

    buffer.update(cx, |buffer, cx| {
        assert_eq!(buffer.text(), "fn one() -> usize { 1 }");
        let ix = buffer.text().find('1').unwrap();
        buffer.edit([(ix..ix + 1, "100")], None, cx);
    });

    // The user saves the buffer. The new contents are written to the
    // remote filesystem.
    project
        .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
        .await
        .unwrap();
    assert_eq!(
        fs.load("/code/project1/src/lib.rs".as_ref()).await.unwrap(),
        "fn one() -> usize { 100 }"
    );

    // A new file is created in the remote filesystem. The user
    // sees the new file.
    fs.save(
        path!("/code/project1/src/main.rs").as_ref(),
        &"fn main() {}".into(),
        Default::default(),
    )
    .await
    .unwrap();
    cx.executor().run_until_parked();
    worktree.update(cx, |worktree, _cx| {
        assert_eq!(
            worktree.paths().collect::<Vec<_>>(),
            vec![
                rel_path("README.md"),
                rel_path("src"),
                rel_path("src/lib.rs"),
                rel_path("src/main.rs"),
            ]
        );
    });

    // A file that is currently open in a buffer is renamed.
    fs.rename(
        path!("/code/project1/src/lib.rs").as_ref(),
        path!("/code/project1/src/lib2.rs").as_ref(),
        Default::default(),
    )
    .await
    .unwrap();
    cx.executor().run_until_parked();
    buffer.update(cx, |buffer, _| {
        assert_eq!(&**buffer.file().unwrap().path(), rel_path("src/lib2.rs"));
    });

    fs.set_index_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib2.rs", "fn one() -> usize { 100 }".into())],
    );
    cx.executor().run_until_parked();
    diff.update(cx, |diff, cx| {
        assert_eq!(
            diff.base_text_string(cx).unwrap(),
            "fn one() -> usize { 100 }"
        );
    });
}

#[cfg(feature = "jvm-tools")]
mod jvm_tools_runtime_tests {
    use super::*;
    use dap::{
        DapRegistry,
        adapters::{DebugTaskDefinition, FakeAdapter},
        client::SessionId,
    };
    use futures::FutureExt as _;
    use gpui::BorrowAppContext as _;
    use language::{Language, LanguageQueries};
    use project::structured_execution::{
        StructuredNodeKind, StructuredProviderId, StructuredProviderStatus, StructuredRunPhase,
    };
    use std::borrow::Cow;
    use task::{StructuredTaskHandle, TaskId};
    use workspace::Panel as _;

    #[cfg(unix)]
    #[gpui::test]
    #[ignore = "requires GROOVY_JDTLS_LAUNCHER and GROOVY_LSP_JAVA_HOME with cached Maven dependencies"]
    async fn connected_jdtls_process_obeys_host_trust(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        connected_jvm_process_obeys_host_trust("jdtls", cx, server_cx).await;
    }

    #[cfg(unix)]
    #[gpui::test]
    #[ignore = "requires GROOVY_KOTLIN_LSP and GROOVY_LSP_JAVA_HOME with cached Maven dependencies"]
    async fn connected_kotlin_process_obeys_host_trust(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        connected_jvm_process_obeys_host_trust("kotlin-lsp", cx, server_cx).await;
    }

    #[cfg(unix)]
    async fn connected_jvm_process_obeys_host_trust(
        name: &'static str,
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use language::{Language, LanguageConfig, LanguageMatcher};
        use project::lsp_store::{LanguageServerState, LspStoreEvent};
        use project::trusted_worktrees::{self, DbTrustedPaths};
        use std::time::{Duration, Instant};

        cx.executor().allow_parking();
        server_cx.executor().allow_parking();
        cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
        server_cx.update(|cx| trusted_worktrees::init(DbTrustedPaths::default(), cx));
        let state = tempfile::tempdir().expect("isolated JVM state");
        let root = state.path().join("project");
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../project/test_data/groovy_jvm_maven");
        for relative in [
            "pom.xml",
            "library/pom.xml",
            "application/pom.xml",
            "library/src/main/kotlin/evidence/Greeting.kt",
            "application/src/main/java/evidence/Main.java",
        ] {
            let destination = root.join(relative);
            std::fs::create_dir_all(destination.parent().expect("fixture parent"))
                .expect("create fixture parent");
            std::fs::copy(fixture.join(relative), destination).expect("copy Maven source fixture");
        }
        let root = root.canonicalize().expect("canonical fixture root");
        std::fs::create_dir_all(root.join(".mvn")).expect("Maven configuration directory");
        std::fs::write(root.join(".mvn/maven.config"), "-o\n").expect("cached Maven only");
        let java_home = std::env::var("GROOVY_LSP_JAVA_HOME").expect("explicit server JDK");
        assert!(Path::new(&java_home).join("bin/java").is_file());
        let (language_name, suffix, source, launcher_variable) = if name == "jdtls" {
            (
                "Java",
                "java",
                "application/src/main/java/evidence/Main.java",
                "GROOVY_JDTLS_LAUNCHER",
            )
        } else {
            (
                "Kotlin",
                "kt",
                "library/src/main/kotlin/evidence/Greeting.kt",
                "GROOVY_KOTLIN_LSP",
            )
        };
        let launcher =
            std::env::var(launcher_variable).expect("explicit pinned JVM server launcher");
        assert!(Path::new(&launcher).is_absolute() && Path::new(&launcher).is_file());
        if name == "jdtls" {
            assert!(
                Path::new(&launcher)
                    .parent()
                    .expect("JDTLS launcher directory")
                    .join("jdtls.py")
                    .is_file(),
                "JDTLS candidate is incomplete: missing launcher companion"
            );
        }
        let trace = state.path().join("launcher-pids");
        let mut arguments = vec![
            "-c".to_string(),
            "printf '%s\\n' \"$$\" >> \"$1\"; shift; exec \"$@\"".to_string(),
            "groovy-lsp-trace".to_string(),
            trace.to_string_lossy().into_owned(),
            launcher,
        ];
        if name == "jdtls" {
            arguments.extend([
                format!("--java-executable={java_home}/bin/java"),
                "-data".to_string(),
                state
                    .path()
                    .join("jdtls-data")
                    .to_string_lossy()
                    .into_owned(),
            ]);
        } else {
            arguments.push("--stdio".to_string());
        }
        let tool_options = ["config", "system", "plugins", "log"]
            .into_iter()
            .map(|kind| format!("-Didea.{kind}.path={}", state.path().join(kind).display()))
            .collect::<Vec<_>>()
            .join(" ");
        let settings = json!({"lsp": {name: {"binary": {
            "path": "/bin/sh", "arguments": arguments,
            "env": {"JAVA_HOME": java_home, "JAVA_TOOL_OPTIONS": tool_options}
        }}}});
        std::fs::create_dir_all(root.join(".zed")).expect("settings directory");
        std::fs::write(root.join(".zed/settings.json"), settings.to_string())
            .expect("explicit binary override");
        let fs = Arc::new(fs::RealFs::new(None, server_cx.executor()));
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        for context in [&mut *cx, &mut *server_cx] {
            context.update(|cx| {
                cx.update_global::<SettingsStore, _>(|store, cx| {
                    store
                        .set_user_settings(&settings.to_string(), cx)
                        .expect("explicit override before host admission");
                });
            });
        }
        let (languages, worktree_store, lsp_store) =
            headless.read_with(server_cx, |headless, _| {
                (
                    headless.languages.clone(),
                    headless.worktree_store.clone(),
                    headless.lsp_store.clone(),
                )
            });
        server_cx.update(|cx| {
            trusted_worktrees::track_worktree_trust(
                worktree_store.clone(),
                None,
                Some((
                    headless.read(cx).session.clone(),
                    client::ProjectId(proto::REMOTE_SERVER_PROJECT_ID),
                )),
                None,
                cx,
            );
        });
        let client_languages = project.read_with(cx, |project, _| project.languages().clone());
        for registry in [&languages, &client_languages] {
            registry.add(Arc::new(Language::new(
                LanguageConfig {
                    name: language_name.into(),
                    matcher: LanguageMatcher {
                        path_suffixes: vec![suffix.to_string()],
                        ..Default::default()
                    }
                    .into(),
                    ..Default::default()
                },
                None,
            )));
            registry.register_fake_lsp_adapter(
                language_name,
                FakeLspAdapter {
                    name,
                    ..FakeLspAdapter::default()
                },
            );
        }
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(&root, true, cx)
            })
            .await
            .expect("connected host worktree");
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let client_store = project.read_with(cx, |project, _| project.lsp_store());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("connected host")
                .read(cx)
                .proto_client()
        });
        let (_buffer, _handle) = project
            .update(cx, |project, cx| {
                project.open_buffer_with_lsp((worktree_id, rel_path(source)), cx)
            })
            .await
            .expect("open host JVM source");
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert!(!trace.exists(), "host launcher ran before trust");
        assert!(
            lsp_store.read_with(server_cx, |store, _| store.as_local().is_some_and(
                |local| local
                    .language_servers
                    .values()
                    .any(|state| matches!(state, LanguageServerState::Starting { .. }))
            )),
            "host provider is not waiting at the shared trust gate"
        );
        assert!(
            client_store.read_with(cx, |store, _| store.as_remote().is_some()
                && store.as_local().is_none())
        );
        assert!(lsp_store.read_with(server_cx, |store, _| {
            store.language_server_statuses().next().is_none()
        }));
        let mut previous_id = None;
        for generation in 1..=2 {
            let (started, response) = futures::channel::oneshot::channel();
            let mut started = Some(started);
            let subscription = lsp_store.update(server_cx, |_, cx| {
                cx.subscribe(&lsp_store, move |_, _, event, _| {
                    if let LspStoreEvent::LanguageServerAdded(id, _, _) = event
                        && let Some(started) = started.take()
                    {
                        started.send(*id).expect("server admission observer");
                    }
                })
            });
            client
                .request(proto::TrustWorktrees {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    trusted_paths: vec![proto::PathTrust {
                        content: Some(proto::path_trust::Content::WorktreeId(
                            worktree_id.to_proto(),
                        )),
                    }],
                })
                .await
                .expect("trust through existing host RPC");
            let initialization = response.fuse();
            let timeout = cx
                .background_executor
                .timer(Duration::from_secs(180))
                .fuse();
            futures::pin_mut!(initialization, timeout);
            let id = futures::select_biased! {
                response = initialization => response.expect("real provider initialization"),
                () = timeout => panic!("{name} did not initialize within the bounded fixture timeout"),
            };
            drop(subscription);
            assert_ne!(Some(id), previous_id);
            let server = lsp_store.read_with(server_cx, |store, _| {
                store
                    .language_server_for_id(id)
                    .expect("running real provider")
            });
            let pid = server.process_id().expect("real host process ID");
            let trace_text = std::fs::read_to_string(&trace).expect("launcher trace");
            assert_eq!(trace_text.lines().count(), generation);
            assert_eq!(
                trace_text
                    .lines()
                    .last()
                    .expect("current launcher PID")
                    .parse::<u32>()
                    .expect("numeric PID"),
                pid
            );
            assert!(
                util::command::new_command("kill")
                    .args(["-0", &pid.to_string()])
                    .status()
                    .await
                    .expect("probe admitted process")
                    .success()
            );
            println!(
                "{name}: generation={generation}, server={id}, pid={pid}, version={:?}, admitted after trust",
                server.version()
            );
            cx.run_until_parked();
            server_cx.run_until_parked();
            assert!(
                client_store.read_with(cx, |store, _| {
                    store
                        .language_server_statuses()
                        .any(|(server_id, _)| server_id == id)
                        && store.language_server_for_id(id).is_none()
                }),
                "client must expose only the host server status"
            );
            client
                .request(proto::RestrictWorktrees {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_ids: vec![worktree_id.to_proto()],
                })
                .await
                .expect("restrict through existing host RPC");
            let deadline = Instant::now() + Duration::from_secs(30);
            while server.process_id().is_some() && Instant::now() < deadline {
                cx.run_until_parked();
                server_cx.run_until_parked();
                cx.background_executor
                    .timer(Duration::from_millis(100))
                    .await;
            }
            assert!(
                server.process_id().is_none(),
                "restricted provider process did not shut down"
            );
            assert!(
                !util::command::new_command("kill")
                    .args(["-0", &pid.to_string()])
                    .stderr(util::command::Stdio::null())
                    .status()
                    .await
                    .expect("probe stopped process")
                    .success(),
                "restricted provider PID remains alive"
            );
            assert!(lsp_store.read_with(server_cx, |store, _| {
                store.language_server_for_id(id).is_none()
            }));
            assert_eq!(
                std::fs::read_to_string(&trace)
                    .expect("retained trace")
                    .lines()
                    .count(),
                generation
            );
            println!("{name}: generation={generation}, pid={pid}, stopped after restriction");
            cx.run_until_parked();
            server_cx.run_until_parked();
            assert!(
                client_store.read_with(cx, |store, _| {
                    store
                        .language_server_statuses()
                        .all(|(server_id, _)| server_id != id)
                        && store.as_local().is_none()
                }),
                "restricted host generation must disappear from the client"
            );
            previous_id = Some(id);
        }
    }

    #[gpui::test]
    async fn remote_tcp_only_debug_adapter_keeps_port_forward_open(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/remote-debug-forward"),
            json!({"Main.java": "class Main {}"}),
        )
        .await;
        let (project, _) = init_test(&fs, cx, server_cx).await;
        server_cx.update(|cx| DapRegistry::global(cx).add_adapter(Arc::new(FakeAdapter::new())));
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/remote-debug-forward"), true, cx)
            })
            .await
            .expect("remote debug worktree");

        let definition = DebugTaskDefinition {
            label: "TCP-only adapter".into(),
            adapter: FakeAdapter::ADAPTER_NAME.into(),
            config: json!({"request": "launch", "tcp_only": true}),
            tcp_connection: Some(task::TcpArgumentsTemplate {
                host: Some(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
                port: Some(4343),
                timeout: None,
            }),
        };
        let (console, _) = futures::channel::mpsc::unbounded();
        let binary = project
            .read_with(cx, |project, _| project.dap_store())
            .update(cx, |store, cx| {
                store.get_debug_adapter_binary(
                    definition.clone(),
                    SessionId(1),
                    &worktree,
                    console,
                    cx,
                )
            })
            .await
            .expect("remote TCP adapter binary");
        assert_eq!(binary.command.as_deref(), Some("mock"));
        assert_eq!(binary.arguments.first().map(String::as_str), Some("-N"));
        assert!(
            binary
                .arguments
                .last()
                .is_some_and(|forward| forward.ends_with(":127.0.0.1:4343"))
        );
        assert!(binary.connection.is_some());

        let (console, _) = futures::channel::mpsc::unbounded();
        let command_binary = project
            .read_with(cx, |project, _| project.dap_store())
            .update(cx, |store, cx| {
                store.get_debug_adapter_binary(
                    DebugTaskDefinition {
                        config: json!({"request": "launch"}),
                        ..definition
                    },
                    SessionId(2),
                    &worktree,
                    console,
                    cx,
                )
            })
            .await
            .expect("remote command adapter binary");
        assert_eq!(command_binary.command.as_deref(), Some("mock"));
        assert_eq!(command_binary.arguments, ["command"]);
    }

    #[gpui::test]
    async fn remote_java_adapter_resolution_requires_host_trust(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/remote-java-debug"),
            json!({"Main.java": "class Main { public static void main(String[] args) {} }"}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/remote-java-debug"), true, cx)
            })
            .await
            .expect("remote Java worktree");
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let request = || proto::GetDebugAdapterBinary {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            session_id: 1,
            worktree_id: worktree_id.to_proto(),
            definition: Some(proto::DebugTaskDefinition {
                adapter: "Java".into(),
                label: "Java main".into(),
                config: json!({"request": "launch", "mainClass": "Main"}).to_string(),
                tcp_connection: None,
            }),
        };
        let denied = client
            .request(request())
            .await
            .expect_err("untrusted host must reject Java adapter resolution");
        assert!(denied.to_string().contains("Trust the project"));
        let mut copper_request = request();
        copper_request
            .definition
            .as_mut()
            .expect("debug definition")
            .adapter = "CodeLLDB".into();
        let copper_result = client
            .request(copper_request)
            .await
            .expect_err("fake host has no installed Copper adapter");
        assert!(!copper_result.to_string().contains("Trust the project"));

        server_cx.update(|cx| {
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        let no_adapter = client
            .request(request())
            .await
            .expect_err("fake host has no installed Java adapter");
        assert!(!no_adapter.to_string().contains("Trust the project"));
    }

    #[gpui::test]
    async fn remote_jvm_consumer_intent_refuses_isolated_authority(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use futures::channel::mpsc;
        cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
        server_cx.update(|cx| {
            release_channel::init(semver::Version::new(0, 0, 0), cx);
            HeadlessProject::init(cx);
            project::trusted_worktrees::init(HashMap::default(), cx);
        });
        let (host_output, client_input) = mpsc::unbounded();
        let (client_output, host_input) = mpsc::unbounded();
        let session = server_cx.update(|cx| {
            RemoteClient::proto_client_from_channels(
                host_input,
                host_output,
                cx,
                "isolated-jvm-host",
                false,
            )
        });
        let client = cx.update(|cx| {
            RemoteClient::proto_client_from_channels(
                client_input,
                client_output,
                cx,
                "isolated-jvm-client",
                false,
            )
        });
        let fs = FakeFs::new(server_cx.executor());
        let languages = Arc::new(LanguageRegistry::new(server_cx.executor()));
        let headless = server_cx.new(|cx| {
            HeadlessProject::isolated_recipe_test(
                crate::HeadlessAppState {
                    session,
                    fs,
                    http_client: Arc::new(BlockedHttpClient),
                    node_runtime: NodeRuntime::unavailable(),
                    languages: languages.clone(),
                    extension_host_proxy: Arc::new(ExtensionHostProxy::new()),
                    startup_time: std::time::Instant::now(),
                },
                cx,
            )
        });
        let lsp_store = headless.read_with(server_cx, |headless, _| headless.lsp_store.clone());
        let denied = lsp_store
            .update(server_cx, |store, cx| {
                store.set_host_jvm_session(Some(proto::PeerId { owner_id: 0, id: 1 }), cx)
            })
            .expect_err("isolated profile cannot acquire a native JVM session");
        assert!(
            format!("{denied:#}").contains("isolated execution profile"),
            "isolated refusal: {denied:#?}"
        );
        // Match RemoteClient startup before using its low-level channel request seam.
        server_cx.run_until_parked();
        cx.run_until_parked();
        let denied = client
            .request(proto::SetJvmProjectConsumers {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: 1,
                generation: 1,
                roots: vec![],
            })
            .await
            .expect_err("isolated RPC cannot acquire or withdraw JVM authority");
        assert!(
            format!("{denied:#}").contains("isolated execution profile"),
            "isolated refusal: {denied:#?}"
        );
        server_cx.run_until_parked();
        lsp_store.read_with(server_cx, |store, _| {
            assert!(store.language_server_statuses().next().is_none());
        });
        assert!(languages.available_language_for_name("Java").is_none());
        assert!(languages.available_language_for_name("Kotlin").is_none());
    }

    #[gpui::test]
    async fn remote_jvm_consumer_intent_preserves_independent_host_owners(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use futures::channel::mpsc;
        use proto::Message as _;
        init_logger();

        cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
        server_cx.update(|cx| {
            release_channel::init(semver::Version::new(0, 0, 0), cx);
            HeadlessProject::init(cx);
            project::trusted_worktrees::init(HashMap::default(), cx);
        });
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm-consumer"),
            json!({
                "pom.xml":"<project/>", "Main.java":"class Main {}",
                ".env":{"pom.xml":"private"}
            }),
        )
        .await;
        let transport = server_cx.new(|_| crate::NativeProjectTransport::default());
        let peer = transport
            .update(server_cx, |transport, cx| transport.accept(cx))
            .expect("accepted receiving scope");
        let receiving_peer = Arc::new(std::sync::Mutex::new(peer));
        let relayed_restart = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (host_input, host_receiver) = mpsc::unbounded();
        let (host_output, client_receiver) = mpsc::unbounded();
        let (client_output, mut client_sender) = mpsc::unbounded();
        let session = server_cx.update(|cx| {
            RemoteClient::proto_client_from_scoped_channels(
                host_receiver,
                host_output,
                cx,
                "jvm-host",
                false,
            )
        });
        let client = cx.update(|cx| {
            RemoteClient::proto_client_from_channels(
                client_receiver,
                client_output,
                cx,
                "jvm-client",
                false,
            )
        });
        let _relay = server_cx.background_spawn({
            let receiving_peer = receiving_peer.clone();
            let relayed_restart = relayed_restart.clone();
            async move {
                while let Some(envelope) = client_sender.next().await {
                    let peer = *receiving_peer.lock().expect("receiving scope");
                    let encoded = envelope.encode_to_vec();
                    let mut envelope = proto::Envelope::decode(encoded.as_slice())?;
                    if relayed_restart.load(std::sync::atomic::Ordering::SeqCst) {
                        envelope.original_sender_id = Some(proto::PeerId { owner_id: 9, id: 1 });
                    }
                    host_input
                        .unbounded_send((peer, envelope))
                        .map_err(|_| anyhow::anyhow!("host fixture input closed"))?;
                }
                anyhow::Ok(())
            }
        });
        let languages = Arc::new(LanguageRegistry::new(server_cx.executor()));
        let (mut java_servers, mut kotlin_servers) = server_cx.update(|_| {
            for (name, suffix) in [("Java", "java"), ("Kotlin", "kt")] {
                languages.register_test_language(LanguageConfig {
                    name: name.into(),
                    matcher: LanguageMatcher {
                        path_suffixes: vec![suffix.into()],
                        ..Default::default()
                    }
                    .into(),
                    ..Default::default()
                });
            }
            (
                languages.register_fake_lsp(
                    "Java",
                    FakeLspAdapter {
                        name: "jdtls",
                        ..Default::default()
                    },
                ),
                languages.register_fake_lsp(
                    "Kotlin",
                    FakeLspAdapter {
                        name: "kotlin-lsp",
                        ..Default::default()
                    },
                ),
            )
        });
        let headless = server_cx.new(|cx| {
            HeadlessProject::new(
                crate::HeadlessAppState {
                    session,
                    fs: fs.clone(),
                    http_client: Arc::new(BlockedHttpClient),
                    node_runtime: NodeRuntime::unavailable(),
                    languages: languages.clone(),
                    extension_host_proxy: Arc::new(ExtensionHostProxy::new()),
                    startup_time: std::time::Instant::now(),
                },
                false,
                cx,
            )
        });
        headless.update(server_cx, |headless, cx| {
            headless.observe_jvm_transport(&transport, cx)
        });
        let worktree_id = project::WorktreeId::from_proto(17);
        let next_entry_id =
            headless.read_with(server_cx, |headless, _| headless.next_entry_id.clone());
        let mut asynchronous = server_cx.to_async();
        let host_worktree = worktree::Worktree::local(
            Arc::from(std::path::Path::new(path!("/jvm-consumer"))),
            true,
            fs.clone(),
            next_entry_id,
            true,
            worktree_id,
            &mut asynchronous,
        )
        .await
        .expect("host-owned project folder");
        headless.update(server_cx, |headless, cx| {
            headless
                .worktree_store
                .update(cx, |store, cx| store.add(&host_worktree, cx));
        });
        server_cx.run_until_parked();
        let root = ProjectPath::root_path(worktree_id);
        let request = |generation| proto::SetJvmProjectConsumers {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: 1,
            generation,
            roots: vec![root.to_proto()],
        };
        let denied = client
            .request(request(1))
            .await
            .expect_err("trust precedes startup");
        assert!(
            denied
                .to_string()
                .contains("Trust the current host project")
        );
        assert!(java_servers.next().now_or_never().is_none());
        assert!(kotlin_servers.next().now_or_never().is_none());
        let lsp_store = headless.read_with(server_cx, |headless, _| headless.lsp_store.clone());
        server_cx.update(|cx| {
            let worktrees = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktrees,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                )
            });
        });
        for invalid in [
            proto::SetJvmProjectConsumers {
                protocol_version: 0,
                ..request(1)
            },
            request(0),
            proto::SetJvmProjectConsumers {
                roots: vec![proto::ProjectPath {
                    worktree_id: u64::MAX,
                    path: String::new(),
                }],
                ..request(1)
            },
            proto::SetJvmProjectConsumers {
                project_id: u64::MAX,
                ..request(1)
            },
            proto::SetJvmProjectConsumers {
                roots: vec![root.to_proto(); 2],
                ..request(1)
            },
            proto::SetJvmProjectConsumers {
                roots: vec![root.to_proto(); 129],
                ..request(1)
            },
            proto::SetJvmProjectConsumers {
                roots: vec![proto::ProjectPath {
                    worktree_id: worktree_id.to_proto(),
                    path: ".env".into(),
                }],
                ..request(1)
            },
            proto::SetJvmProjectConsumers {
                roots: vec![proto::ProjectPath {
                    worktree_id: worktree_id.to_proto(),
                    path: "../foreign".into(),
                }],
                ..request(1)
            },
            proto::SetJvmProjectConsumers {
                roots: vec![proto::ProjectPath {
                    worktree_id: worktree_id.to_proto(),
                    path: "x".repeat(4097),
                }],
                ..request(1)
            },
        ] {
            let error = client
                .request(invalid)
                .await
                .expect_err("invalid scope is atomic")
                .to_string();
            assert!(
                !error.contains("/jvm-consumer")
                    && !error.contains(".env")
                    && !error.contains("foreign")
            );
        }
        assert!(java_servers.next().now_or_never().is_none());
        assert!(kotlin_servers.next().now_or_never().is_none());
        let java = languages
            .language_for_name("Java")
            .await
            .expect("Java language");
        lsp_store
            .update(server_cx, |store, cx| {
                store.register_project_language(root.clone(), java, cx)
            })
            .expect("independent explicit Java owner");
        server_cx.run_until_parked();
        let mut explicit_java = java_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("explicit Java server");
        let admitted = client
            .request(request(1))
            .await
            .expect("current scoped JVM intent");
        assert_eq!(admitted.protocol_version, 1);
        assert_eq!(admitted.generation, 1);
        assert_eq!(admitted.accepted_roots, [root.to_proto()]);
        server_cx.run_until_parked();
        assert!(
            java_servers.next().now_or_never().is_none(),
            "lease shares explicit server"
        );
        let mut leased_kotlin = kotlin_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("bufferless Kotlin server");
        assert!(
            client
                .request(request(1))
                .await
                .expect_err("old generation")
                .to_string()
                .contains("stale")
        );
        client
            .request(request(2))
            .await
            .expect("repeat current scope");
        server_cx.run_until_parked();
        lsp_store.read_with(server_cx, |store, _| {
            assert!(
                store
                    .language_server_for_id(leased_kotlin.server.server_id())
                    .is_some(),
                "repeated current scope must share its existing Kotlin server"
            );
        });
        assert!(
            kotlin_servers.next().now_or_never().is_none(),
            "no duplicate server on repeated intent"
        );
        let restart = || proto::RestartLanguageServers {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            all: true,
            buffer_ids: vec![],
            only_servers: vec![],
        };
        assert!(headless.read_with(server_cx, |headless, cx| {
            headless.buffer_store.read(cx).buffers().next().is_none()
        }));
        *receiving_peer.lock().expect("old receiving scope") = proto::PeerId {
            owner_id: 0,
            id: peer.id + 1,
        };
        let denied = client
            .request(restart())
            .await
            .expect_err("old scope cannot restart leased providers");
        assert!(
            denied
                .to_string()
                .contains("current scoped host connection")
        );
        *receiving_peer
            .lock()
            .expect("restore current receiving scope") = peer;
        relayed_restart.store(true, std::sync::atomic::Ordering::SeqCst);
        let denied = client
            .request(restart())
            .await
            .expect_err("relayed global restart is refused");
        assert!(
            denied
                .to_string()
                .contains("direct current host connection")
        );
        relayed_restart.store(false, std::sync::atomic::Ordering::SeqCst);
        lsp_store.read_with(server_cx, |store, _| {
            assert!(
                store
                    .language_server_for_id(explicit_java.server.server_id())
                    .is_some()
            );
            assert!(
                store
                    .language_server_for_id(leased_kotlin.server.server_id())
                    .is_some()
            );
        });
        let old_java_id = explicit_java.server.server_id();
        let old_kotlin_id = leased_kotlin.server.server_id();
        client
            .request(restart())
            .await
            .expect("current bufferless host restart");
        server_cx.run_until_parked();
        explicit_java = java_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("restarted explicit/leased Java owner");
        leased_kotlin = kotlin_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("restarted leased Kotlin owner");
        assert_ne!(old_java_id, explicit_java.server.server_id());
        assert_ne!(old_kotlin_id, leased_kotlin.server.server_id());
        assert!(java_servers.next().now_or_never().is_none());
        assert!(kotlin_servers.next().now_or_never().is_none());
        client
            .request(proto::SetJvmProjectConsumers {
                roots: vec![],
                ..request(3)
            })
            .await
            .expect("withdraw leased ownership");
        server_cx.run_until_parked();
        lsp_store.read_with(server_cx, |store, _| {
            assert!(
                store
                    .language_server_for_id(explicit_java.server.server_id())
                    .is_some()
            );
            assert!(
                store
                    .language_server_for_id(leased_kotlin.server.server_id())
                    .is_none()
            );
        });
        lsp_store
            .update(server_cx, |store, cx| {
                store.unregister_project_language(&root, &"Java".into(), cx)
            })
            .expect("retire explicit owner");
        server_cx.run_until_parked();
        lsp_store
            .update(server_cx, |store, cx| {
                store.set_default_jvm_project_languages_enabled(true, cx)
            })
            .expect("independent local default Java owner");
        server_cx.run_until_parked();
        let default_java = java_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("default Java server");
        client
            .request(request(4))
            .await
            .expect("lease with local default");
        server_cx.run_until_parked();
        let leased_kotlin = kotlin_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("fresh leased Kotlin");
        assert!(java_servers.next().now_or_never().is_none());
        transport.update(server_cx, |transport, cx| transport.disconnect(cx));
        server_cx.run_until_parked();
        lsp_store.read_with(server_cx, |store, _| {
            assert!(
                store
                    .language_server_for_id(default_java.server.server_id())
                    .is_some()
            );
            assert!(
                store
                    .language_server_for_id(leased_kotlin.server.server_id())
                    .is_none()
            );
        });
        assert!(
            client
                .request(request(4))
                .await
                .expect_err("old receiving scope after disconnect")
                .to_string()
                .contains("current scoped host connection")
        );
        let replacement = transport
            .update(server_cx, |transport, cx| transport.accept(cx))
            .expect("new receiving scope");
        *receiving_peer.lock().expect("receiving scope") = replacement;
        client
            .request(request(1))
            .await
            .expect("new session requires its own intent");
        server_cx.run_until_parked();
        let recovered = kotlin_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("new-session Kotlin owner");
        assert_ne!(
            recovered.server.server_id(),
            leased_kotlin.server.server_id()
        );
        client
            .request(proto::SetJvmProjectConsumers {
                roots: vec![],
                ..request(2)
            })
            .await
            .expect("new-session withdrawal");
        lsp_store
            .update(server_cx, |store, cx| {
                store.set_default_jvm_project_languages_enabled(false, cx)
            })
            .expect("restore local defaults");
        server_cx.run_until_parked();
        let mut generation = 3;
        for scenario in ["supersede", "trust", "input", "settings", "session"] {
            let (started, started_receiver) = futures::channel::oneshot::channel();
            let (release, release_receiver) = futures::channel::oneshot::channel();
            lsp_store.update(server_cx, |store, _| {
                store.gate_next_host_jvm_language_loaded_for_test(started, release_receiver);
            });
            let pending = client.request(request(generation));
            futures::pin_mut!(pending);
            let pending = match futures::future::select(pending, started_receiver).await {
                futures::future::Either::Right((Ok(observed), pending)) => {
                    assert_eq!(observed, generation);
                    pending
                }
                _ => panic!("host request did not reach the real language-loaded checkpoint"),
            };
            assert!(java_servers.next().now_or_never().is_none());
            assert!(kotlin_servers.next().now_or_never().is_none());
            let mut fresh_generation = generation + 1;
            match scenario {
                "supersede" => {
                    client
                        .request(request(fresh_generation))
                        .await
                        .expect("newer host intent");
                }
                "trust" => {
                    server_cx.update(|cx| {
                        let worktrees = headless.read(cx).worktree_store.clone();
                        let trusted =
                            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                                .expect("host trust");
                        trusted.update(cx, |trusted, cx| {
                            trusted.restrict(
                                worktrees.downgrade(),
                                [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                                    .into_iter()
                                    .collect(),
                                cx,
                            )
                        });
                    });
                    server_cx.run_until_parked();
                    server_cx.update(|cx| {
                        let worktrees = headless.read(cx).worktree_store.clone();
                        let trusted =
                            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                                .expect("host trust");
                        trusted.update(cx, |trusted, cx| {
                            trusted.trust(
                                &worktrees,
                                [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                                    .into_iter()
                                    .collect(),
                                cx,
                            )
                        });
                    });
                }
                "input" => {
                    fs.remove_file(
                        std::path::Path::new(path!("/jvm-consumer/pom.xml")),
                        fs::RemoveOptions::default(),
                    )
                    .await
                    .expect("remove disposable build input");
                    server_cx.run_until_parked();
                    fs.insert_file(
                        std::path::Path::new(path!("/jvm-consumer/pom.xml")),
                        b"<project/>".to_vec(),
                    )
                    .await;
                }
                "settings" => {
                    fs.insert_tree(path!("/jvm-consumer"), json!({".zed":{"settings.json":
                        json!({"languages":{"Kotlin":{"enable_language_server":false}}}).to_string()}})).await;
                    server_cx.run_until_parked();
                    fs.remove_file(
                        std::path::Path::new(path!("/jvm-consumer/.zed/settings.json")),
                        fs::RemoveOptions::default(),
                    )
                    .await
                    .expect("restore disposable settings");
                }
                "session" => {
                    transport.update(server_cx, |transport, cx| transport.disconnect(cx));
                    server_cx.run_until_parked();
                    let peer = transport
                        .update(server_cx, |transport, cx| transport.accept(cx))
                        .expect("fresh scope after held setup");
                    *receiving_peer.lock().expect("receiving scope") = peer;
                    fresh_generation = 1;
                }
                _ => unreachable!("fixed scenario list"),
            }
            server_cx.run_until_parked();
            if release.send(()).is_err() {
                log::debug!("Obsolete host JVM setup was cancelled before gate release");
            }
            let error = pending
                .await
                .expect_err("restored conditions cannot rejuvenate old setup")
                .to_string();
            assert!(!error.contains("/jvm-consumer") && !error.contains("settings.json"));
            if scenario != "supersede" {
                assert!(
                    java_servers.next().now_or_never().is_none(),
                    "no stale Java start: {scenario}"
                );
                assert!(
                    kotlin_servers.next().now_or_never().is_none(),
                    "no stale Kotlin start: {scenario}"
                );
                client
                    .request(request(fresh_generation))
                    .await
                    .expect("fresh current authority recovers");
            }
            server_cx.run_until_parked();
            let current_java = java_servers
                .next()
                .now_or_never()
                .flatten()
                .expect("fresh Java owner");
            let current_kotlin = kotlin_servers
                .next()
                .now_or_never()
                .flatten()
                .expect("fresh Kotlin owner");
            client
                .request(proto::SetJvmProjectConsumers {
                    roots: vec![],
                    ..request(fresh_generation + 1)
                })
                .await
                .expect("retire only current owned intent");
            server_cx.run_until_parked();
            lsp_store.read_with(server_cx, |store, _| {
                assert!(
                    store
                        .language_server_for_id(current_java.server.server_id())
                        .is_none()
                );
                assert!(
                    store
                        .language_server_for_id(current_kotlin.server.server_id())
                        .is_none()
                );
            });
            generation = fresh_generation + 2;
        }
        let peer = *receiving_peer.lock().expect("receiving scope");
        let relayed = client
            .forward_request(peer, request(generation))
            .await
            .expect_err("relay metadata cannot grant default startup")
            .to_string();
        assert!(relayed.contains("relayed clients"));
        assert!(java_servers.next().now_or_never().is_none());
        assert!(kotlin_servers.next().now_or_never().is_none());
        let mut tree = json!({});
        let mut selected_build_files = Vec::new();
        let mut bounded_roots = Vec::new();
        for index in 0..128 {
            let path = format!("module-{index:03}");
            tree[&path] = json!({
                "settings.gradle.kts":format!("rootProject.name = {path:?}"),
                "build.gradle.kts":"plugins { java }",
            });
            selected_build_files.push(format!("{path}/settings.gradle.kts"));
            bounded_roots.push(proto::ProjectPath {
                worktree_id: worktree_id.to_proto(),
                path,
            });
        }
        tree[".zed"] = json!({"settings.json":json!({"jvm_builds":{"maven":{"selected_poms":[]},"gradle":{"selected_build_files":selected_build_files}}}).to_string()});
        fs.insert_tree(path!("/jvm-consumer"), tree).await;
        server_cx.run_until_parked();
        server_cx
            .executor()
            .advance_clock(std::time::Duration::from_millis(350));
        server_cx.run_until_parked();
        let admitted = client
            .request(proto::SetJvmProjectConsumers {
                roots: bounded_roots.clone(),
                ..request(generation)
            })
            .await
            .expect("exact128 distinct eligible host roots");
        assert_eq!(admitted.accepted_roots, bounded_roots);
        server_cx.run_until_parked();
        let bounded_java = java_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("shared128-root Java server");
        let bounded_kotlin = kotlin_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("shared128-root Kotlin server");
        assert!(java_servers.next().now_or_never().is_none());
        assert!(kotlin_servers.next().now_or_never().is_none());
        let mut oversized_roots = bounded_roots;
        oversized_roots.push(proto::ProjectPath {
            worktree_id: worktree_id.to_proto(),
            path: "outside-limit".into(),
        });
        for roots in [
            oversized_roots,
            vec![proto::ProjectPath {
                worktree_id: worktree_id.to_proto(),
                path: "x".repeat(1_048_577),
            }],
        ] {
            let error = client
                .request(proto::SetJvmProjectConsumers {
                    roots,
                    ..request(generation + 1)
                })
                .await
                .expect_err("scope and byte budgets refuse before mutation")
                .to_string();
            assert!(error.contains("scope limit"));
            lsp_store.read_with(server_cx, |store, _| {
                assert!(
                    store
                        .language_server_for_id(bounded_java.server.server_id())
                        .is_some()
                );
                assert!(
                    store
                        .language_server_for_id(bounded_kotlin.server.server_id())
                        .is_some()
                );
            });
        }
        client
            .request(proto::SetJvmProjectConsumers {
                roots: vec![],
                ..request(generation + 1)
            })
            .await
            .expect("withdraw bounded scope after refusals");
        server_cx.run_until_parked();
        lsp_store.read_with(server_cx, |store, _| {
            assert!(
                store
                    .language_server_for_id(bounded_java.server.server_id())
                    .is_none()
            );
            assert!(
                store
                    .language_server_for_id(bounded_kotlin.server.server_id())
                    .is_none()
            );
        });
        fs.remove_file(
            std::path::Path::new(path!("/jvm-consumer/.zed/settings.json")),
            fs::RemoveOptions::default(),
        )
        .await
        .expect("restore quota fixture settings");
        server_cx.run_until_parked();

        let selection = json!({"jvm_builds":{
            "maven":{"selected_poms":["pom.xml"]},
            "gradle":{"selected_build_files":[]}
        }});
        fs.insert_tree(
            path!("/jvm-consumer"),
            json!({".zed":{"settings.json":selection.to_string()}}),
        )
        .await;
        server_cx.run_until_parked();
        transport.update(server_cx, |transport, cx| transport.disconnect(cx));
        server_cx.run_until_parked();
        let peer = transport
            .update(server_cx, |transport, cx| transport.accept(cx))
            .expect("fresh client-owned receiving scope");
        *receiving_peer.lock().expect("receiving scope") = peer;
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            cx.update_global::<SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings(&selection.to_string(), cx)
                    .expect("matching client selection intent");
            });
            project::trusted_worktrees::init(HashMap::default(), cx);
        });
        let (metadata, entries) = host_worktree.read_with(server_cx, |worktree, _| {
            (
                worktree.metadata_proto(),
                worktree
                    .snapshot()
                    .entries(true, 0)
                    .map(proto::Entry::from)
                    .collect::<Vec<_>>(),
            )
        });
        let remote_worktree = cx.update(|cx| {
            worktree::Worktree::remote(
                proto::REMOTE_SERVER_PROJECT_ID,
                clock::ReplicaId::LOCAL,
                metadata.clone(),
                client.clone(),
                util::paths::PathStyle::Unix,
                cx,
            )
        });
        let remote_worktrees = cx.new(|_| {
            project::worktree_store::WorktreeStore::remote(
                true,
                client.clone(),
                proto::REMOTE_SERVER_PROJECT_ID,
                util::paths::PathStyle::Unix,
                Default::default(),
            )
        });
        remote_worktrees.update(cx, |worktrees, cx| worktrees.add(&remote_worktree, cx));
        remote_worktree.update(cx, |worktree, _| {
            worktree
                .as_remote()
                .expect("remote snapshot")
                .update_from_remote(proto::UpdateWorktree {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: metadata.id,
                    abs_path: metadata.abs_path,
                    root_name: metadata.root_name,
                    scan_id: 1,
                    is_last_update: true,
                    updated_entries: entries,
                    ..Default::default()
                })
        });
        cx.run_until_parked();
        let client_languages = Arc::new(LanguageRegistry::new(cx.executor()));
        let (mut forbidden_java, mut forbidden_kotlin) = cx.update(|_| {
            for (name, suffix) in [("Java", "java"), ("Kotlin", "kt")] {
                client_languages.register_test_language(LanguageConfig {
                    name: name.into(),
                    matcher: LanguageMatcher {
                        path_suffixes: vec![suffix.into()],
                        ..Default::default()
                    }
                    .into(),
                    ..Default::default()
                });
            }
            (
                client_languages.register_fake_lsp(
                    "Java",
                    FakeLspAdapter {
                        name: "jdtls",
                        ..Default::default()
                    },
                ),
                client_languages.register_fake_lsp(
                    "Kotlin",
                    FakeLspAdapter {
                        name: "kotlin-lsp",
                        ..Default::default()
                    },
                ),
            )
        });
        let client_store = cx.update(|cx| {
            project::lsp_store::LspStore::remote_for_test(
                remote_worktrees.clone(),
                client_languages,
                client.clone(),
                proto::REMOTE_SERVER_PROJECT_ID,
                cx,
            )
        });
        cx.update(|cx| {
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client host-trust observation");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &remote_worktrees,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                )
            });
        });
        client_store
            .update(cx, |store, cx| {
                store.set_default_jvm_project_languages_enabled(true, cx)
            })
            .expect("current typed client intent");
        cx.condition(&client_store, |store, _| {
            store.remote_jvm_consumer_roots() == std::slice::from_ref(&root)
        })
        .await;
        let first_java = java_servers.next().await.expect("client-owned host Java");
        let first_kotlin = kotlin_servers
            .next()
            .await
            .expect("client-owned host Kotlin");
        assert!(headless.read_with(server_cx, |headless, cx| {
            headless.buffer_store.read(cx).buffers().next().is_none()
        }));
        client_store.update(cx, |store, cx| store.restart_all_language_servers(cx));
        assert!(
            client_store.read_with(cx, |store, _| store.remote_jvm_consumer_roots().is_empty())
        );
        cx.condition(&client_store, |store, _| {
            store.remote_jvm_consumer_roots() == std::slice::from_ref(&root)
        })
        .await;
        let restarted_java = java_servers
            .next()
            .await
            .expect("acknowledged host Java replacement");
        let restarted_kotlin = kotlin_servers
            .next()
            .await
            .expect("acknowledged host Kotlin replacement");
        assert_ne!(
            first_java.server.server_id(),
            restarted_java.server.server_id()
        );
        assert_ne!(
            first_kotlin.server.server_id(),
            restarted_kotlin.server.server_id()
        );
        server_cx.run_until_parked();
        assert!(java_servers.next().now_or_never().is_none());
        assert!(kotlin_servers.next().now_or_never().is_none());
        assert!(forbidden_java.next().now_or_never().is_none());
        assert!(forbidden_kotlin.next().now_or_never().is_none());
        let retired = client_store.downgrade();
        drop(client_store);
        cx.update(|_| {});
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert!(retired.upgrade().is_none());
        lsp_store.read_with(server_cx, |store, _| {
            assert!(
                store
                    .language_server_for_id(restarted_java.server.server_id())
                    .is_none()
            );
            assert!(
                store
                    .language_server_for_id(restarted_kotlin.server.server_id())
                    .is_none()
            );
        });
        fs.remove_file(
            std::path::Path::new(path!("/jvm-consumer/.zed/settings.json")),
            fs::RemoveOptions::default(),
        )
        .await
        .expect("restore client restart fixture settings");
        server_cx.run_until_parked();
    }

    struct GroovyRemoteWorkspaceFixture {
        project: Entity<Project>,
        headless: Entity<HeadlessProject>,
        _transport: Entity<crate::NativeProjectTransport>,
        root: ProjectPath,
        maven: bool,
        java_servers: futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
        kotlin_servers: futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
        forbidden_java: futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
        forbidden_kotlin: futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
    }

    impl GroovyRemoteWorkspaceFixture {
        async fn new(
            maven: bool,
            scoped: bool,
            cx: &mut TestAppContext,
            server_cx: &mut TestAppContext,
        ) -> Self {
            cx.update(|cx| {
                let settings = SettingsStore::test(cx);
                cx.set_global(settings);
                release_channel::init(semver::Version::new(0, 0, 0), cx);
                project::trusted_worktrees::init(HashMap::default(), cx);
                theme_settings::init(theme::LoadThemes::JustBase, cx);
                editor::init(cx);
                jvm_ui::init(cx);
            });
            server_cx.update(|cx| {
                release_channel::init(semver::Version::new(0, 0, 0), cx);
                HeadlessProject::init(cx);
                project::trusted_worktrees::init(HashMap::default(), cx);
            });
            let root_path = if maven {
                path!("/groovy-remote-maven")
            } else {
                path!("/groovy-remote-gradle")
            };
            let root_uri = lsp::Uri::from_file_path(root_path)
                .expect("host fixture file URI")
                .to_string();
            let root_uri = format!("{}/", root_uri.trim_end_matches('/'));
            let module_uri = format!("{root_uri}application/");
            let source_path = Path::new(root_path)
                .join("application")
                .join("src")
                .join("main")
                .join("kotlin")
                .to_string_lossy()
                .into_owned();
            let fs = FakeFs::new(server_cx.executor());
            let mut tree =
                json!({"application":{"src":{"main":{"kotlin":{"Main.kt":"fun main() {}"}}}}});
            if maven {
                tree["pom.xml"] = "<project/>".into();
                tree["application"]["pom.xml"] = "<project/>".into();
            } else {
                tree["settings.gradle.kts"] = "include(\":application\")".into();
                tree["application"]["build.gradle.kts"] = "plugins { kotlin(\"jvm\") }".into();
            }
            fs.insert_tree(root_path, tree).await;
            let transport = server_cx.new(|_| crate::NativeProjectTransport::default());
            let peer = transport
                .update(server_cx, |transport, cx| transport.accept(cx))
                .expect("current native scope");
            let (options, session, guard, _) = if scoped {
                RemoteClient::fake_server_scoped(peer, cx, server_cx)
                    .expect("scoped mounted connection")
            } else {
                RemoteClient::fake_server_with_activity(cx, server_cx)
            };
            let languages = Arc::new(LanguageRegistry::new(server_cx.executor()));
            for (name, suffix) in [("Java", "java"), ("Kotlin", "kt")] {
                languages.register_test_language(LanguageConfig {
                    name: name.into(),
                    matcher: LanguageMatcher {
                        path_suffixes: vec![suffix.into()],
                        ..Default::default()
                    }
                    .into(),
                    ..Default::default()
                });
            }
            let java_servers = languages.register_fake_lsp("Java", FakeLspAdapter {
                name:"jdtls",
                capabilities:lsp::ServerCapabilities {execute_command_provider:Some(lsp::ExecuteCommandOptions {
                    commands:vec!["java.project.getAll".into(),"java.project.getSettings".into(),"groovy.maven.importedProjects".into(),"groovy.gradle.importedProjects".into()], ..Default::default()
                }), ..Default::default()},
                initializer:Some(Box::new(move |server| {
                    let root_uri = root_uri.clone(); let module_uri = module_uri.clone(); let source_path = source_path.clone();
                    server.set_request_handler::<lsp::request::ExecuteCommand, _, _>(move |params, _| {
                        let root_uri = root_uri.clone(); let module_uri = module_uri.clone(); let source_path = source_path.clone();
                        async move { match params.command.as_str() {
                            "java.project.getAll" => Ok(Some(json!([module_uri]))),
                            "java.project.getSettings" => Ok(Some(json!({"org.eclipse.jdt.ls.core.sourcePaths":[source_path],"org.eclipse.jdt.ls.core.classpathEntries":[]}))),
                            "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                let argument = params.arguments.first().expect("provider root");
                                assert_eq!(argument["rootUri"], root_uri); assert_eq!(argument["cachedOnly"], true);
                                let selected = (params.command == "groovy.maven.importedProjects") == maven;
                                Ok(Some(json!({"version":1,"rootUri":root_uri,"projects":if selected {vec![json!({"projectUri":module_uri,"javaNature":true})]} else {Vec::new()},"partial":false,"fullSynchronizationKnown":false})))
                            }
                            _ => anyhow::bail!("Unexpected mounted fixture provider command"),
                        }}
                    });
                })), ..Default::default()
            });
            let kotlin_servers = languages.register_fake_lsp(
                "Kotlin",
                FakeLspAdapter {
                    name: "kotlin-lsp",
                    ..Default::default()
                },
            );
            let headless = server_cx.new(|cx| {
                HeadlessProject::new(
                    crate::HeadlessAppState {
                        session,
                        fs: fs.clone(),
                        http_client: Arc::new(BlockedHttpClient),
                        node_runtime: NodeRuntime::unavailable(),
                        languages,
                        extension_host_proxy: Arc::new(ExtensionHostProxy::new()),
                        startup_time: std::time::Instant::now(),
                    },
                    false,
                    cx,
                )
            });
            headless.update(server_cx, |headless, cx| {
                headless.observe_jvm_transport(&transport, cx)
            });
            drop(guard);
            let remote = RemoteClient::connect_mock(options, cx).await;
            let project = build_project_with_worktree_trust(remote, true, cx);
            let (forbidden_java, forbidden_kotlin) = project.read_with(cx, |project, _| {
                let languages = project.languages();
                for (name, suffix) in [("Java", "java"), ("Kotlin", "kt")] {
                    languages.register_test_language(LanguageConfig {
                        name: name.into(),
                        matcher: LanguageMatcher {
                            path_suffixes: vec![suffix.into()],
                            ..Default::default()
                        }
                        .into(),
                        ..Default::default()
                    });
                }
                (
                    languages.register_fake_lsp(
                        "Java",
                        FakeLspAdapter {
                            name: "jdtls",
                            ..Default::default()
                        },
                    ),
                    languages.register_fake_lsp(
                        "Kotlin",
                        FakeLspAdapter {
                            name: "kotlin-lsp",
                            ..Default::default()
                        },
                    ),
                )
            });
            let (worktree, _) = project
                .update(cx, |project, cx| {
                    project.find_or_create_worktree(root_path, true, cx)
                })
                .await
                .expect("host project folder");
            let root = ProjectPath::root_path(worktree.read_with(cx, |worktree, _| worktree.id()));
            Self {
                project,
                headless,
                _transport: transport,
                root,
                maven,
                java_servers,
                kotlin_servers,
                forbidden_java,
                forbidden_kotlin,
            }
        }

        fn mount(&self, cx: &mut TestAppContext) -> Entity<workspace::MultiWorkspace> {
            cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(self.project.clone(), window, cx)
            })
            .0
        }

        fn mount_again(
            &self,
            first: &Entity<workspace::MultiWorkspace>,
            cx: &mut TestAppContext,
        ) -> Entity<workspace::MultiWorkspace> {
            let workspace = first.read_with(cx, |container, _| container.workspace().clone());
            let app_state = workspace.read_with(cx, |workspace, _| workspace.app_state().clone());
            cx.add_window_view(|window, cx| {
                let workspace = cx.new(|cx| {
                    workspace::Workspace::new(None, self.project.clone(), app_state, window, cx)
                });
                workspace::MultiWorkspace::new(workspace, window, cx)
            })
            .0
        }

        fn trust(&self, cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
            server_cx.update(|cx| {
                let worktrees = self.headless.read(cx).worktree_store.clone();
                let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                    .expect("host trust owner");
                trusted.update(cx, |trusted, cx| {
                    trusted.trust(
                        &worktrees,
                        [project::trusted_worktrees::PathTrust::Worktree(
                            self.root.worktree_id,
                        )]
                        .into_iter()
                        .collect(),
                        cx,
                    )
                });
            });
            cx.update(|cx| {
                let worktrees = self.project.read(cx).worktree_store();
                let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                    .expect("client trust observation");
                trusted.update(cx, |trusted, cx| {
                    trusted.trust(
                        &worktrees,
                        [project::trusted_worktrees::PathTrust::Worktree(
                            self.root.worktree_id,
                        )]
                        .into_iter()
                        .collect(),
                        cx,
                    )
                });
            });
        }

        fn assert_client_has_no_provider(&mut self, cx: &mut TestAppContext) {
            assert!(self.forbidden_java.next().now_or_never().is_none());
            assert!(self.forbidden_kotlin.next().now_or_never().is_none());
            self.project.read_with(cx, |project, cx| {
                assert!(project.buffer_store().read(cx).buffers().next().is_none())
            });
        }

        async fn verify(&mut self, cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
            let first = self.mount(cx);
            cx.run_until_parked();
            server_cx.run_until_parked();
            assert!(self.java_servers.next().now_or_never().is_none());
            assert!(self.kotlin_servers.next().now_or_never().is_none());
            self.trust(cx, server_cx);
            let client_store = self.project.read_with(cx, |project, _| project.lsp_store());
            if product_flavor::ID != "jvm" {
                cx.run_until_parked();
                server_cx.run_until_parked();
                cx.run_until_parked();
                assert!(
                    client_store
                        .read_with(cx, |store, _| store.remote_jvm_consumer_roots().is_empty())
                );
                assert!(self.java_servers.next().now_or_never().is_none());
                assert!(self.kotlin_servers.next().now_or_never().is_none());
                self.assert_client_has_no_provider(cx);
                return;
            }
            cx.condition(&client_store, |store, _| {
                store.remote_jvm_consumer_roots() == std::slice::from_ref(&self.root)
            })
            .await;
            let java = self
                .java_servers
                .next()
                .await
                .expect("default remote Java provider");
            let kotlin = self
                .kotlin_servers
                .next()
                .await
                .expect("default remote Kotlin provider");
            let host_store = self
                .headless
                .read_with(server_cx, |headless, _| headless.lsp_store.clone());
            server_cx
                .condition(&host_store, |store, _| {
                    store
                        .jdtls_project_snapshot(self.root.worktree_id)
                        .is_some_and(|(_, current)| current)
                })
                .await;
            host_store.read_with(server_cx, |store, _| {
                let (snapshot, current) = store
                    .jdtls_project_snapshot(self.root.worktree_id)
                    .expect("bufferless host metadata");
                assert!(current);
                assert_eq!(snapshot.projects.len(), 1);
                let module = snapshot.projects.first().expect("imported module");
                assert_eq!(
                    module.imported_build_kind,
                    if self.maven {
                        project::jvm_project::JdtlsImportedBuildKind::Maven
                    } else {
                        project::jvm_project::JdtlsImportedBuildKind::Gradle
                    }
                );
                assert_eq!(module.module_root.path.as_unix_str(), "application");
                assert_eq!(module.source_roots.len(), 1);
            });
            let workspace = first.read_with(cx, |container, _| container.workspace().clone());
            assert!(workspace.read_with(cx, |workspace, cx| {
                workspace.panel::<jvm_ui::DependenciesPanel>(cx).is_none()
            }));
            assert!(self.headless.read_with(server_cx, |headless, cx| {
                headless.buffer_store.read(cx).buffers().next().is_none()
            }));
            let _second = self.mount_again(&first, cx);
            cx.condition(&client_store, |store, _| {
                store.remote_jvm_consumer_roots() == std::slice::from_ref(&self.root)
            })
            .await;
            server_cx.run_until_parked();
            cx.run_until_parked();
            assert!(self.java_servers.next().now_or_never().is_none());
            assert!(self.kotlin_servers.next().now_or_never().is_none());
            host_store.read_with(server_cx, |store, _| {
                assert!(
                    store
                        .language_server_for_id(java.server.server_id())
                        .is_some()
                );
                assert!(
                    store
                        .language_server_for_id(kotlin.server.server_id())
                        .is_some()
                );
            });
            client_store.update(cx, |store, cx| store.stop_all_language_servers(cx));
            cx.run_until_parked();
            server_cx.run_until_parked();
            cx.run_until_parked();
            assert!(
                client_store.read_with(cx, |store, _| store.remote_jvm_consumer_roots().is_empty())
            );
            host_store.read_with(server_cx, |store, _| {
                assert!(
                    store
                        .language_server_for_id(java.server.server_id())
                        .is_none()
                );
                assert!(
                    store
                        .language_server_for_id(kotlin.server.server_id())
                        .is_none()
                );
            });
            client_store.update(cx, |store, cx| store.restart_all_language_servers(cx));
            cx.condition(&client_store, |store, _| {
                store.remote_jvm_consumer_roots() == std::slice::from_ref(&self.root)
            })
            .await;
            let recovered_java = self
                .java_servers
                .next()
                .await
                .expect("explicit host Java recovery");
            let recovered_kotlin = self
                .kotlin_servers
                .next()
                .await
                .expect("explicit host Kotlin recovery");
            assert_ne!(java.server.server_id(), recovered_java.server.server_id());
            assert_ne!(
                kotlin.server.server_id(),
                recovered_kotlin.server.server_id()
            );
            self.assert_client_has_no_provider(cx);
        }
    }

    #[gpui::test]
    async fn compiled_remote_maven_workspace_uses_catalog_default(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        GroovyRemoteWorkspaceFixture::new(true, true, cx, server_cx)
            .await
            .verify(cx, server_cx)
            .await;
    }

    #[gpui::test]
    async fn compiled_remote_gradle_workspace_uses_catalog_default(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        GroovyRemoteWorkspaceFixture::new(false, true, cx, server_cx)
            .await
            .verify(cx, server_cx)
            .await;
    }

    #[gpui::test]
    async fn compiled_remote_workspace_refuses_unsupported_host_scope(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let mut fixture = GroovyRemoteWorkspaceFixture::new(true, false, cx, server_cx).await;
        let store = fixture
            .project
            .read_with(cx, |project, _| project.lsp_store());
        let notices = cx.new(|_| Vec::<String>::new());
        let _subscription = cx.update(|cx| {
            cx.subscribe(&store, {
                let notices = notices.clone();
                move |_, event, cx| {
                    if let project::lsp_store::LspStoreEvent::Notification(message) = event {
                        notices.update(cx, |notices, cx| {
                            notices.push(message.clone());
                            cx.notify();
                        });
                    }
                }
            })
        });
        fixture.trust(cx, server_cx);
        let _workspace = fixture.mount(cx);
        if product_flavor::ID == "jvm" {
            cx.condition(&notices, |notices, _| !notices.is_empty())
                .await;
            notices.read_with(cx, |notices, _| {
                assert!(notices.iter().all(|notice| notice.len() < 512
                    && notice.contains("host")
                    && !notice.contains("groovy-remote")))
            });
        } else {
            cx.run_until_parked();
            server_cx.run_until_parked();
            cx.run_until_parked();
            assert!(notices.read_with(cx, |notices, _| notices.is_empty()));
        }
        assert!(store.read_with(cx, |store, _| store.remote_jvm_consumer_roots().is_empty()));
        assert!(fixture.java_servers.next().now_or_never().is_none());
        assert!(fixture.kotlin_servers.next().now_or_never().is_none());
        fixture.assert_client_has_no_provider(cx);
    }

    #[gpui::test]
    async fn remote_jdtls_snapshot_is_reduced_on_the_host(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let project_vm = tempfile::tempdir().expect("project VM fixture");
        std::fs::write(
            project_vm.path().join("release"),
            "JAVA_VERSION=\"17.0.18\"\n",
        )
        .expect("project VM release metadata");
        let project_vm_location = project_vm.path().to_string_lossy().into_owned();
        let launch_jdk = tempfile::tempdir().expect("host launch JDK fixture");
        std::fs::write(
            launch_jdk.path().join("release"),
            "JAVA_VERSION=\"25.0.1\"\n",
        )
        .expect("host launch JDK release metadata");
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/remote-jdtls"),
            json!({
                "build.gradle.kts": "plugins { java }",
                "pom.xml": "<project/>",
                ".zed": { "settings.json": json!({
                    "lsp": { "jdtls": { "binary": {
                        "path": launch_jdk.path().join("java-lsp-proxy"),
                        "arguments": [
                            launch_jdk.path(),
                            launch_jdk.path().join("bin/java"),
                            "-Declipse.application=org.eclipse.jdt.ls.core.id1"
                        ]
                    } } }
                }).to_string() },
                "application": {
                    "pom.xml": "<project/>",
                    ".env": "private",
                    "Main.java": "class Main {}",
                    "generated": { "Generated.java": "class Generated {}" }
                },
                "resource": { "pom.xml": "<project/>" },
                "library": { "pom.xml": "<project/>", "build.gradle.kts": "plugins { java }",
                    ".env": "private", "generated": {"Generated.java": "class Generated {}"} }
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (store, languages) = project.read_with(cx, |project, _| {
            (project.lsp_store(), project.languages().clone())
        });
        let loaded_before = languages.to_vec().len();
        let notices = cx.new(|_| Vec::<String>::new());
        let _setup_subscription = cx.update(|cx| {
            cx.subscribe(&store, {
                let notices = notices.clone();
                move |_, event, cx| {
                    if let project::lsp_store::LspStoreEvent::Notification(message) = event {
                        notices.update(cx, |notices, _| notices.push(message.clone()));
                    }
                }
            })
        });
        store
            .update(cx, |store, cx| {
                store.set_default_jvm_project_languages_enabled(true, cx)
            })
            .expect("client retains intent without granting host authority");
        cx.run_until_parked();
        server_cx.run_until_parked();
        cx.run_until_parked();
        assert!(store.read_with(cx, |store, _| store.remote_jvm_consumer_roots().is_empty()));
        assert!(headless.read_with(server_cx, |host, cx| {
            host.lsp_store.read(cx).language_server_statuses.is_empty()
        }));
        notices.read_with(cx, |notices, _| {
            assert!(
                notices
                    .iter()
                    .any(|message| message.contains("host") && message.contains("retry")),
                "unscoped host refusal is actionable"
            );
        });
        assert_eq!(
            languages.to_vec().len(),
            loaded_before,
            "retaining intent does not load a client language"
        );
        store
            .update(cx, |store, cx| {
                store.set_default_jvm_project_languages_enabled(false, cx)
            })
            .expect("disable isolated startup intent before the metadata journey");
        cx.run_until_parked();
        server_cx.run_until_parked();
        cx.run_until_parked();
        project.update(cx, |project, _| {
            project.languages().register_test_language(LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            });
            project.languages().register_fake_lsp_adapter(
                "Java",
                FakeLspAdapter {
                    name: "jdtls",
                    ..Default::default()
                },
            );
        });
        server_cx.update(|cx| {
            let languages = &headless.read(cx).languages;
            languages.register_test_language(LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            });
            languages.register_fake_lsp_adapter(
                "Java",
                FakeLspAdapter {
                    name: "jdtls",
                    ..Default::default()
                },
            );
        });
        let query_gate = Arc::new(std::sync::Mutex::new(
            None::<(
                futures::channel::oneshot::Sender<()>,
                futures::channel::oneshot::Receiver<()>,
            )>,
        ));
        let graph_scope = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let project_queries = Arc::new(AtomicUsize::new(0));
        let configuring_queries = Arc::new(AtomicUsize::new(0));
        let mut fake_servers = server_cx.update(|cx| {
            headless.read(cx).languages.register_fake_lsp_server(
                LanguageServerName::new_static("jdtls"),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![
                            "java.project.getAll".to_string(),
                            "java.project.getSettings".to_string(),
                            "groovy.gradle.projectDetails".to_string(),
                            "groovy.gradle.dependencyGraph".to_string(),
                            "groovy.maven.dependencyGraph".to_string(),
                            "groovy.maven.projectDetails".to_string(),
                            "groovy.maven.goalDetails".to_string(),
                            "groovy.maven.importedProjects".to_string(),
                            "groovy.gradle.importedProjects".to_string(),
                            "groovy.maven.importReactor".to_string(),
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                Some(Box::new({
                    let graph_scope = graph_scope.clone();
                    let query_gate = query_gate.clone();
                    let project_queries = project_queries.clone();
                    let configuring_queries = configuring_queries.clone();
                    let project_vm_location = project_vm_location.clone();
                    move |fake| {
                        fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                            let graph_scope = graph_scope.clone();
                    let query_gate = query_gate.clone();
                            let project_queries = project_queries.clone();
                            let configuring_queries = configuring_queries.clone();
                            let project_vm_location = project_vm_location.clone();
                            move |params, _| {
                                let graph_scope = graph_scope.clone();
                    let query_gate = query_gate.clone();
                                let project_queries = project_queries.clone();
                                let configuring_queries = configuring_queries.clone();
                                let project_vm_location = project_vm_location.clone();
                                async move {
                                    if params.command == "java.project.getAll" {
                                        project_queries.fetch_add(1, Ordering::SeqCst);
                                        let gate =
                                            query_gate.lock().expect("query gate lock").take();
                                        if let Some((started, release)) = gate {
                                            started.send(()).map_err(|_| {
                                                anyhow::anyhow!("query start receiver closed")
                                            })?;
                                            release.await?;
                                        }
                                    }
                                    match params.command.as_str() {
                                        "groovy.maven.importReactor" => {
                                            assert_eq!(params.arguments, [json!({"rootUri":"file:///remote-jdtls/","pomUri":"file:///remote-jdtls/application/pom.xml","excludedProjectUris":[],"offline":true})]);
                                            Ok(Some(json!({"version":1,"rootUri":"file:///remote-jdtls/","reactorPomUri":"file:///remote-jdtls/application/pom.xml","importedPomUris":["file:///remote-jdtls/application/pom.xml"],"excludedProjectUris":[],"selectedImportPhaseComplete":true,"fullSynchronizationKnown":false})))
                                        }
                                        "java.project.getAll" => {
                                            if params.arguments.is_empty() {
                                                Ok(Some(json!(["file:///remote-jdtls/application", "file:///remote-jdtls/library"])))
                                            } else {
                                                assert_eq!(params.arguments, [json!(r#"{"includeNonJava":true}"#)]);
                                                let resources = if graph_scope.load(Ordering::SeqCst) {
                                                    json!(["file:///remote-jdtls/application", "file:///remote-jdtls/library", "file:///remote-jdtls/resource"])
                                                } else {
                                                    json!(["file:///remote-jdtls/application", "file:///remote-jdtls/library", "file:///remote-jdtls/resource", "file:///remote-jdtls/library/.env"])
                                                };
                                                Ok(Some(resources))
                                            }
                                        },
                                        "groovy.maven.importedProjects" | "groovy.gradle.importedProjects" => {
                                            assert_eq!(params.arguments, [json!({"rootUri":"file:///remote-jdtls/","cachedOnly":true})]);
                                            let projects = if params.command == "groovy.maven.importedProjects" {
                                                if graph_scope.load(Ordering::SeqCst) {
                                                    json!([{"projectUri":"file:///remote-jdtls/application/","javaNature":true}, {"projectUri":"file:///remote-jdtls/resource/","javaNature":false}])
                                                } else {
                                                    json!([{"projectUri":"file:///remote-jdtls/application/","javaNature":true},
                                                        {"projectUri":"file:///remote-jdtls/resource/","javaNature":false},
                                                        {"projectUri":"file:///remote-jdtls/library/.env/","javaNature":false}])
                                                }
                                            } else { json!([{"projectUri":"file:///remote-jdtls/library/","javaNature":true}]) };
                                            Ok(Some(json!({"version":1,"rootUri":"file:///remote-jdtls/","projects":projects,"partial":false,"fullSynchronizationKnown":false})))
                                        }
                                        "java.project.getSettings" => {
                                            let module = params
                                                .arguments
                                                .first()
                                                .and_then(|argument| argument.as_str())
                                                .expect("JDTLS module URI");
                                            assert!(!module.ends_with("/resource") && !module.ends_with("/.env"), "non-Java imported resources must not query Java semantics");
                                            if module.ends_with("/application") {
                                                Ok(Some(json!({
                                                    "org.eclipse.jdt.ls.core.sourcePaths": [
                                                        "/remote-jdtls/application/generated",
                                                        "/private/host/source"
                                                    ],
                                                    "org.eclipse.jdt.ls.core.classpathEntries": [
                                                        {"kind": 2, "path": "/library"},
                                                        {
                                                            "kind": 1,
                                                            "path": "/private/host/cache.jar",
                                                            "attributes": {
                                                                "maven.groupId": "org.example",
                                                                "maven.artifactId": "cache",
                                                                "maven.version": "1.2.3",
                                                                "maven.scope": "runtime"
                                                            }
                                                        },
                                                        {
                                                            "kind": 1,
                                                            "path": "/private/host/gradle.jar",
                                                            "attributes": {
                                                                "gradle_used_by_scope": "main,test"
                                                            }
                                                        }
                                                    ],
                                                    "org.eclipse.jdt.core.compiler.source": "17",
                                                    "org.eclipse.jdt.core.compiler.codegen.targetPlatform": "17",
                                                    "org.eclipse.jdt.ls.core.vm.location": project_vm_location
                                                })))
                                            } else {
                                                Ok(Some(json!({
                                                    "org.eclipse.jdt.ls.core.sourcePaths": [],
                                                    "org.eclipse.jdt.ls.core.classpathEntries": []
                                                })))
                                            }
                                        }
                                        "groovy.gradle.projectDetails" => {
                                            configuring_queries.fetch_add(1, Ordering::SeqCst);
                                            let configuration_declarations = json!([
                                                {"name":"implementation","description":null,"resolvable":false,"consumable":false,"extendsFrom":[]},
                                                {"name":"runtimeClasspath","description":"Runtime declaration","resolvable":true,"consumable":false,"extendsFrom":["implementation"]}
                                            ]);
                                            assert_eq!(params.arguments[0]["rootUri"], "file:///remote-jdtls/");
                                            assert_eq!(params.arguments[0]["offline"], true);
                                            Ok(Some(json!({"version":2,"rootUri":"file:///remote-jdtls/","sourceSetsKnown":false,
                                                "projects":[{"buildRootUri":"file:///remote-jdtls/","moduleUri":"file:///remote-jdtls/library/",
                                                    "projectPath":":library","name":"library","tasks":[
                                                        {"path":":library:test","name":"test","group":"verification",
                                                            "description":"Run selected tests","optionsKnown":false}],
                                                    "detailsJson":json!({"javaSourceSetsKnown":true,"configurationsKnown":true,
                                                        "dependencyGraphKnown":false,"configurations":configuration_declarations,"sourceSets":[
                                                        {"name":"main","generatednessKnown":false,
                                                            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":"file:///remote-jdtls/library/generated","roots":[
                                                            "file:///remote-jdtls/library/generated",
                                                            "file:///remote-jdtls/library/.env/missing",
                                                            "file:///remote-jdtls/library/not-yet-generated"]},
                                                        {"name":"noCompilerOutput","generatednessKnown":false,"roots":[],
                                                            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":null},
                                                        {"name":"privateCompilerOutput","generatednessKnown":false,"roots":[],
                                                            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":"file:///remote-jdtls/library/.env/missing"},
                                                        {"name":"missingCompilerOutput","generatednessKnown":false,"roots":[],
                                                            "annotationProcessorOutputKnown":true,"annotationProcessorOutputUri":"file:///remote-jdtls/library/not-yet-generated"}],
                                                        "tasks":[{"path":":library:test","optionsComplete":true,"options":[
                                                            {"name":"tests","description":"Selected test filter","declaredType":"java.util.List<java.lang.String>"},
                                                            {"name":"mode","description":"Declared enum members","declaredType":"fixture.Mode","enumConstantsKnown":true,"enumConstants":["ALPHA","ZULU"]},
                                                            {"name":"emptyMode","description":"Known-empty enum","declaredType":"fixture.Empty","enumConstantsKnown":true,"enumConstants":[]}]}]}).to_string()}]})))
                                        }
                                        "groovy.maven.dependencyGraph" => {
                                            assert_eq!(params.arguments, [json!({"rootUri":"file:///remote-jdtls/","pomUri":"file:///remote-jdtls/application/pom.xml","offline":true,"cachedOnly":true})]);
                                            let component = json!(["evidence", "application", "jar", "", "1.0"]).to_string();
                                            let node = format!("node:{}", "0".repeat(64));
                                            Ok(Some(json!({"version":2,"rootUri":"file:///remote-jdtls/","pomUri":"file:///remote-jdtls/application/pom.xml",
                                                "occurrencesKnown":true,"rootNode":node,"rootComponent":component,
                                                "nodes":[{"id":node,"component":component}],
                                                "components":[{"id":component,"groupId":"evidence","artifactId":"application","extension":"jar","classifier":"","version":"1.0"}],
                                                "edges":[],"unavailableDescriptors":[],"collectionComplete":true,"structuralCoverageKnown":true,"problem":null,
                                                "originsKnown":false,"artifactsKnown":false,"offline":true,"cachedOnly":true,"cycleCount":0,"cycles":[],"cycleOccurrencesKnown":false})))
                                        }
                                        "groovy.gradle.dependencyGraph" => {
                                            assert_eq!(params.arguments, [json!({"rootUri":"file:///remote-jdtls/","moduleUri":"file:///remote-jdtls/library/","configuration":"runtimeClasspath","offline":true})]);
                                            let component = json!(["project", ":library"]).to_string();
                                            let resolution = json!({"configuration":"runtimeClasspath","rootComponent":component,
                                                "structuralCoverageKnown":true,"resolutionComplete":true,"originsKnown":false,"artifactsKnown":false,
                                                "components":[{"id":component,"kind":"project","buildTreePath":":library","projectPath":":library",
                                                    "conflictResolution":false,"forced":false,"selectedByRule":false,"compositeSubstitution":false,"constrained":false,"selectionCauses":["ROOT"]}],"edges":[]});
                                            Ok(Some(json!({"version":1,"rootUri":"file:///remote-jdtls/","projects":[{
                                                "buildRootUri":"file:///remote-jdtls/","moduleUri":"file:///remote-jdtls/library/","projectPath":":library","name":"library","graphJson":resolution.to_string()}]})))
                                        }
                                        "groovy.maven.goalDetails" => {
                                            let parameter_declarations = json!([{"name":"skipTests","declaredType":"boolean","required":false,"editable":true}]);
                                            assert_eq!(params.arguments[0]["rootUri"], "file:///remote-jdtls/");
                                            assert_eq!(params.arguments[0]["cachedOnly"], true);
                                            Ok(Some(json!({"version":3,"rootUri":"file:///remote-jdtls/","cachedOnly":true,
                                                "reactorOrderKnown":true,"reactorPomUris":["file:///remote-jdtls/application/pom.xml"],
                                                "unavailablePomUris":[],"lifecycles":[{"id":"default","phases":["test","verify"]}],
                                                "projects":[{"pomUri":"file:///remote-jdtls/application/pom.xml","groupId":"evidence",
                                                    "artifactId":"application","version":"1.0","packaging":"jar",
                                                    "moduleUris":["file:///remote-jdtls/library/","file:///remote-jdtls/application/.env/missing"],
                                                    "profiles":{"parity":true},"inheritedProfileDefinitionsKnown":true,"profileDefinitions":[
                                                        {"id":"parity","sourceGroupId":"evidence","sourceArtifactId":"application",
                                                            "sourceVersion":"1.0","sourcePomUri":"file:///remote-jdtls/application/pom.xml",
                                                            "inherited":false,"activeAtSource":true},
                                                        {"id":"hidden","sourceGroupId":"evidence","sourceArtifactId":"parent","sourceVersion":"1.0",
                                                            "sourcePomUri":"file:///remote-jdtls/application/.env/missing.xml","inherited":true,"activeAtSource":false}],
                                                    "plugins":[{"groupId":"org.apache.maven.plugins","artifactId":"maven-surefire-plugin",
                                                        "version":"3.5.2","managedOnly":false,"goalsKnown":true,"goalPrefix":"surefire",
                                                        "goals":[{"name":"test","phase":"test","description":"Run tests","parametersKnown":true,"parameters":parameter_declarations}]}]}]})))
                                        }
                                        _ => anyhow::bail!("unexpected JDTLS command"),
                                    }
                                }
                            }
                        });
                    }
                })),
            )
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/remote-jdtls"), true, cx)
            })
            .await
            .expect("remote Java worktree");
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        server_cx.update(|cx| {
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            jvm_ui::init(cx);
        });
        let (auto_refresh_container, auto_refresh_cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let auto_refresh_workspace = auto_refresh_container
            .read_with(auto_refresh_cx, |container, _| {
                container.workspace().clone()
            });
        let async_cx = auto_refresh_cx.update(|window, cx| window.to_async(cx));
        let auto_refresh_panel =
            jvm_ui::DependenciesPanel::load(auto_refresh_workspace.downgrade(), async_cx)
                .await
                .expect("early remote Dependencies panel");
        auto_refresh_panel.update_in(auto_refresh_cx, |panel, window, cx| {
            panel.set_active(true, window, cx);
        });
        assert!(auto_refresh_panel.read_with(auto_refresh_cx, |panel, _| {
            panel.snapshot_for_test().1.is_empty()
        }));
        let (_buffer, _handle) = project
            .update(auto_refresh_cx, |project, cx| {
                project.open_buffer_with_lsp((worktree_id, rel_path("application/Main.java")), cx)
            })
            .await
            .expect("open remote Java source");
        auto_refresh_cx.run_until_parked();
        server_cx.run_until_parked();
        let original_server = fake_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("host JDTLS provider");
        auto_refresh_cx.run_until_parked();
        server_cx.run_until_parked();

        let client = project.read_with(auto_refresh_cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            0,
            "host startup and early framed reads must not configure Gradle"
        );
        let host_lsp_store =
            headless.read_with(server_cx, |headless, _| headless.lsp_store.clone());
        host_lsp_store
            .update(server_cx, |store, cx| {
                store.import_local_maven_reactor(
                    worktree_id,
                    project::ProjectPath {
                        worktree_id,
                        path: util::rel_path::RelPath::from_unix_str("application/pom.xml")
                            .expect("fixture POM")
                            .into(),
                    },
                    Vec::new(),
                    cx,
                )
            })
            .expect("host-only selected Maven phase")
            .await
            .expect("phase receipt");
        server_cx
            .executor()
            .advance_clock(std::time::Duration::from_millis(300));
        server_cx.run_until_parked();
        host_lsp_store
            .update(server_cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect("explicit host-only Gradle metadata synchronization");
        server_cx
            .executor()
            .advance_clock(std::time::Duration::from_millis(300));
        server_cx.run_until_parked();
        assert_eq!(configuring_queries.load(Ordering::SeqCst), 1);
        let host_generation = host_lsp_store.read_with(server_cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .expect("explicit host snapshot")
                .0
                .generation
        });
        let response = client
            .request(proto::GetJdtlsProjectSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_id: worktree_id.to_proto(),
                protocol_version: 1,
                request_id: 7,
            })
            .await
            .expect("host-reduced JDTLS snapshot");
        assert_eq!(response.request_id, 7);
        assert_eq!(response.protocol_version, 1);
        assert!(
            response
                .projects
                .iter()
                .all(|project| project.gradle.is_none() && project.maven.is_none())
        );
        let extended = client
            .request(proto::GetJdtlsProjectSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_id: worktree_id.to_proto(),
                protocol_version: project::jvm_project::JDTLS_SNAPSHOT_PROTOCOL_VERSION,
                request_id: 71,
            })
            .await
            .expect("host build facts over framed RPC");
        assert_eq!(
            extended.protocol_version,
            project::jvm_project::JDTLS_SNAPSHOT_PROTOCOL_VERSION
        );
        assert_eq!(extended.request_id, 71);
        let selection = extended
            .maven_selection
            .as_ref()
            .expect("host phase facts over framed RPC");
        assert_eq!(selection.imported_poms, ["application/pom.xml"]);
        assert!(selection.ignored_poms.is_empty());
        let legacy_selection = client
            .request(proto::GetJdtlsProjectSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_id: worktree_id.to_proto(),
                protocol_version: 6,
                request_id: 72,
            })
            .await
            .expect("older framed peer");
        assert!(legacy_selection.maven_selection.is_none());
        assert!(legacy_selection.partial);
        assert!(legacy_selection.projects.iter().all(|project| {
            project.gradle.as_ref().is_none_or(|gradle| {
                gradle.source_sets.iter().all(|source| {
                    source.annotation_processor_output_known.is_none()
                        && source.annotation_processor_output.is_none()
                })
            })
        }));

        let imported_resource = extended
            .projects
            .iter()
            .find(|entry| entry.module_root == "resource")
            .expect("framed resource identity");
        assert_eq!(
            imported_resource.membership,
            proto::JdtlsProjectMembership::ImportedNonJava as i32
        );
        assert_eq!(
            imported_resource.imported_build_kind,
            proto::JdtlsImportedBuildKind::Maven as i32
        );
        assert_eq!(
            extended
                .projects
                .iter()
                .find(|entry| entry.module_root == "application")
                .expect("Maven imported Java")
                .imported_build_kind,
            proto::JdtlsImportedBuildKind::Maven as i32
        );
        assert_eq!(
            extended
                .projects
                .iter()
                .find(|entry| entry.module_root == "library")
                .expect("Gradle imported Java")
                .imported_build_kind,
            proto::JdtlsImportedBuildKind::Gradle as i32
        );
        for version in 1..=3 {
            let legacy = client
                .request(proto::GetJdtlsProjectSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: version,
                    request_id: 80 + u64::from(version),
                })
                .await
                .expect("legacy framed snapshot projection");
            assert!(
                legacy.projects.iter().all(|entry| entry.imported_build_kind
                    == proto::JdtlsImportedBuildKind::Unknown as i32)
            );
        }
        assert!(imported_resource.source_roots.is_empty() && !imported_resource.project_vm_known);
        assert!(imported_resource.gradle.is_none() && imported_resource.maven.is_none());
        assert!(
            response
                .projects
                .iter()
                .all(|entry| entry.membership == proto::JdtlsProjectMembership::Unknown as i32)
        );

        assert!(extended.partial);
        assert!(!format!("{extended:?}").contains("/remote-jdtls"));
        assert!(!format!("{extended:?}").contains(".env"));
        assert!(!format!("{extended:?}").contains("not-yet-generated"));
        let module = extended
            .projects
            .iter()
            .find(|project| project.module_root == "application")
            .expect("extended module");
        let gradle = extended
            .projects
            .iter()
            .find(|project| project.module_root == "library")
            .and_then(|project| project.gradle.as_ref())
            .expect("host Gradle facts");
        assert_eq!(gradle.project_path, ":library");
        assert_eq!(gradle.tasks[0].options[0].name, "tests");
        assert!(gradle.tasks[0].options_known);
        assert_eq!(gradle.source_sets[0].roots.len(), 1);
        assert_eq!(gradle.source_sets.len(), 4);
        assert_eq!(
            gradle.source_sets[0].annotation_processor_output_known,
            Some(true)
        );
        let output = gradle.source_sets[0]
            .annotation_processor_output
            .as_ref()
            .expect("Visible compiler output");
        assert_eq!(output.path, "library/generated");
        assert!(output.navigable);
        assert_eq!(
            gradle.source_sets[1].annotation_processor_output_known,
            Some(true)
        );
        assert!(gradle.source_sets[1].annotation_processor_output.is_none());
        for source in &gradle.source_sets[2..] {
            assert_eq!(source.annotation_processor_output_known, Some(false));
            assert!(source.annotation_processor_output.is_none());
        }
        assert!(gradle.configurations_known);
        assert_eq!(gradle.configurations[1].extends_from, ["implementation"]);
        assert!(gradle.configurations[1].resolvable);
        let maven = module.maven.as_ref().expect("host Maven facts");
        assert_eq!(maven.reactor_position, Some(0));
        assert_eq!(maven.modules, ["library"]);
        assert!(!maven.profile_definitions_known);
        assert_eq!(maven.profile_definitions.len(), 1);
        assert_eq!(maven.plugins[0].goals[0].name, "test");
        assert!(maven.plugins[0].goals[0].parameters_known);
        assert_eq!(maven.plugins[0].goals[0].parameters[0].name, "skipTests");

        assert!(response.partial);
        assert_eq!(response.projects.len(), 3);
        assert!(!format!("{response:?}").contains("/private/host"));
        assert!(!format!("{response:?}").contains(&project_vm_location));
        assert!(
            !format!("{response:?}").contains(&launch_jdk.path().to_string_lossy().to_string())
        );
        let application = response
            .projects
            .iter()
            .find(|project| project.module_root == "application")
            .expect("application module");
        assert!(application.project_vm_known);
        assert_eq!(application.project_vm_java_major, 17);
        assert_eq!(application.java_target_level, 17);
        assert_eq!(application.server_launch_java_major, 25);
        assert_eq!(application.project_references, ["library"]);
        assert_eq!(application.library_entries, 2);
        assert_eq!(application.library_file_names, ["gradle.jar"]);
        assert_eq!(
            application.library_source_sets[0].source_sets,
            ["main", "test"]
        );
        assert_eq!(application.library_artifacts.len(), 1);
        assert_eq!(
            application.library_artifacts[0].coordinate,
            "org.example:cache:1.2.3"
        );
        assert_eq!(application.library_artifacts[0].scope, "runtime");
        assert_eq!(application.source_roots.len(), 1);
        assert!(
            client
                .request(proto::GetJdtlsProjectSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: u32::MAX,
                    request_id: 8,
                })
                .await
                .is_err()
        );
        let lsp_store = project.read_with(auto_refresh_cx, |project, _| project.lsp_store());
        let denied = lsp_store
            .update(auto_refresh_cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect_err("client RPC must never grant configuring execution");
        assert!(
            denied
                .to_string()
                .contains("unavailable for remote projects")
        );
        lsp_store.update(auto_refresh_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx)
        });
        server_cx.run_until_parked();
        auto_refresh_cx.run_until_parked();
        let (snapshot, current) = lsp_store
            .read_with(auto_refresh_cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .map(|(snapshot, current)| (snapshot.clone(), current))
            })
            .expect("client JDTLS snapshot");
        assert!(current);
        let client_selection = snapshot
            .metadata
            .as_ref()
            .expect("phase facts admitted on the client generation");
        assert_eq!(
            client_selection
                .imported_poms
                .first()
                .expect("imported identity")
                .path
                .as_unix_str(),
            "application/pom.xml"
        );
        assert!(client_selection.ignored_poms.is_empty());
        for revoke_authority in [false, true] {
            let (started, started_receiver) = futures::channel::oneshot::channel();
            let (release, release_receiver) = futures::channel::oneshot::channel();
            lsp_store.update(auto_refresh_cx, |store, cx| {
                store.gate_next_remote_jdtls_reply_for_test(started, release_receiver);
                store.refresh_jdtls_project(worktree_id, cx);
            });
            let deadline = auto_refresh_cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            let held = match futures::future::select(started_receiver, deadline).await {
                futures::future::Either::Left((Ok(response), _)) => {
                    response.expect("successful framed host reply")
                }
                _ => panic!("framed Maven phase reply did not reach client admission gate"),
            };
            assert_eq!(
                held.maven_selection
                    .expect("held real Maven facts")
                    .imported_poms,
                ["application/pom.xml"]
            );
            let held_generation = held.request_id;
            if revoke_authority {
                let trust = auto_refresh_cx.update(|_, cx| {
                    cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>()
                });
                release.send(()).expect("release after authority loss");
                auto_refresh_cx.run_until_parked();
                server_cx.run_until_parked();
                assert!(lsp_store.read_with(auto_refresh_cx, |store, _| {
                    store.jdtls_project_snapshot(worktree_id).is_none()
                }));
                auto_refresh_cx.update(|_, cx| cx.set_global(trust));
                lsp_store.update(auto_refresh_cx, |store, cx| {
                    store.refresh_jdtls_project(worktree_id, cx)
                });
            } else {
                lsp_store.update(auto_refresh_cx, |store, cx| {
                    store.refresh_jdtls_project(worktree_id, cx)
                });
                if release.send(()).is_err() {
                    log::debug!("Superseded JDTLS admission was cancelled before release");
                }
            }
            server_cx.run_until_parked();
            auto_refresh_cx.run_until_parked();
            lsp_store.read_with(auto_refresh_cx, |store, _| {
                let (current, fresh) = store
                    .jdtls_project_snapshot(worktree_id)
                    .expect("fresh admission after supersession or restored authority");
                assert!(fresh && current.generation > held_generation);
                assert!(current.metadata.as_ref().is_some_and(|facts| {
                    facts
                        .imported_poms
                        .first()
                        .is_some_and(|pom| pom.path.as_unix_str() == "application/pom.xml")
                        && facts.ignored_poms.is_empty()
                }));
            });
        }
        let client_gradle = snapshot
            .projects
            .iter()
            .find(|project| project.module_root.path.as_unix_str() == "library")
            .and_then(|project| project.gradle.as_ref())
            .expect("client Gradle facts");
        let options = &client_gradle
            .tasks
            .first()
            .expect("client Gradle task")
            .options;
        assert_eq!(
            options
                .iter()
                .find(|option| option.name == "mode")
                .expect("client enum declaration")
                .enum_constants
                .as_deref(),
            Some(["ALPHA".to_string(), "ZULU".to_string()].as_slice())
        );
        assert_eq!(
            options
                .iter()
                .find(|option| option.name == "emptyMode")
                .expect("client known-empty declaration")
                .enum_constants
                .as_deref(),
            Some([].as_slice())
        );

        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            1,
            "legacy, extended and panel reads reuse host metadata without configuring"
        );
        assert_eq!(
            host_lsp_store.read_with(server_cx, |store, _| store
                .jdtls_project_snapshot(worktree_id)
                .expect("current host snapshot")
                .0
                .generation),
            host_generation,
            "reads must not mint a new host metadata generation"
        );

        assert_eq!(snapshot.projects.len(), 3);
        let build_module = snapshot
            .projects
            .iter()
            .find(|project| project.module_root.path.as_unix_str() == "application")
            .expect("client build facts admitted to existing snapshot");
        let gradle_module = snapshot
            .projects
            .iter()
            .find(|project| project.module_root.path.as_unix_str() == "library")
            .expect("client Gradle module");
        assert_eq!(
            gradle_module
                .gradle
                .as_ref()
                .expect("client Gradle facts")
                .tasks[0]
                .path,
            ":library:test"
        );
        assert_eq!(
            build_module
                .maven
                .as_ref()
                .expect("client Maven facts")
                .pom
                .worktree_id,
            worktree_id
        );
        assert_eq!(
            build_module
                .maven
                .as_ref()
                .expect("client Maven facts")
                .plugins[0]
                .goals[0]
                .name,
            "test"
        );

        assert_eq!(
            snapshot
                .projects
                .iter()
                .find(|project| project.module_root.path.as_unix_str() == "application")
                .expect("client host launch Java")
                .server_launch_java_major,
            Some(25)
        );
        assert_eq!(
            snapshot
                .projects
                .iter()
                .find(|project| project.module_root.path.as_unix_str() == "application")
                .expect("client application target level")
                .java_target_level,
            Some(17)
        );
        let (auto_status, auto_rows, _) =
            auto_refresh_panel.read_with(auto_refresh_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            auto_status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(_)
        ));
        assert_eq!(auto_rows.len(), 3);
        let queries_before_unrelated_update = project_queries.load(Ordering::SeqCst);
        lsp_store.update(auto_refresh_cx, |_, cx| cx.notify());
        auto_refresh_cx.run_until_parked();
        server_cx.run_until_parked();
        assert_eq!(
            project_queries.load(Ordering::SeqCst),
            queries_before_unrelated_update,
            "an unchanged language server must not restart the provider query"
        );
        assert_eq!(
            snapshot
                .projects
                .iter()
                .find(|project| project.module_root.path.as_unix_str() == "application")
                .expect("client application module")
                .java_source_level,
            Some(17)
        );
        assert_eq!(
            snapshot
                .projects
                .iter()
                .find(|project| project.module_root.path.as_unix_str() == "application")
                .expect("client application module")
                .project_vm_java_major,
            Some(17)
        );
        assert_eq!(
            snapshot
                .projects
                .iter()
                .find(|project| project.module_root.path.as_unix_str() == "application")
                .expect("client application module")
                .library_files[0]
                .gradle_source_sets,
            ["main", "test"]
        );
        assert_eq!(
            snapshot
                .projects
                .iter()
                .find(|project| project.module_root.path.as_unix_str() == "application")
                .expect("client application module")
                .library_artifacts
                .len(),
            1
        );
        let container = auto_refresh_container;
        let panel_cx = auto_refresh_cx;
        let panel = auto_refresh_panel;
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(_)
        ));
        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter()
                .any(|row| row.accessibility_label.contains("Java source level 17"))
        );
        assert!(rows.iter().any(|row| {
            row.accessibility_label
                .contains("project VM Java 17 reported by JDTLS")
        }));
        assert!(rows.iter().any(|row| {
            row.accessibility_label
                .contains("Java target level 17 reported by JDTLS")
        }));
        assert!(
            rows.iter()
                .all(|row| row.accessibility_label.contains("JDTLS module"))
        );
        assert_eq!(navigation.len(), 4);
        assert!(
            navigation
                .values()
                .any(|path| path.path.as_unix_str() == "library/generated"),
            "Current host compiler-output directory reuses shared project navigation"
        );
        assert!(
            navigation
                .values()
                .any(|path| path.path.as_unix_str() == "application/pom.xml")
        );
        graph_scope.store(true, Ordering::SeqCst);
        host_lsp_store.update(server_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx)
        });
        server_cx
            .executor()
            .advance_clock(std::time::Duration::from_millis(300));
        server_cx.run_until_parked();
        let (maven_module, gradle_module) = host_lsp_store.read_with(server_cx, |store, _| {
            let (snapshot, current) = store
                .jdtls_project_snapshot(worktree_id)
                .expect("fresh host ownership");
            assert!(current);
            (
                snapshot
                    .projects
                    .iter()
                    .find(|project| project.module_root.path.as_unix_str() == "application")
                    .expect("Maven module")
                    .maven
                    .as_ref()
                    .expect("Maven metadata")
                    .pom
                    .clone(),
                snapshot
                    .projects
                    .iter()
                    .find(|project| project.module_root.path.as_unix_str() == "library")
                    .expect("Gradle module")
                    .module_root
                    .clone(),
            )
        });
        host_lsp_store
            .update(server_cx, |store, cx| {
                store.query_local_maven_dependency_graph(worktree_id, maven_module, cx)
            })
            .expect("host selected Maven graph")
            .await
            .expect("Maven graph admitted");
        server_cx.run_until_parked();
        host_lsp_store
            .update(server_cx, |store, cx| {
                store.query_local_gradle_dependency_graph(
                    worktree_id,
                    gradle_module,
                    "runtimeClasspath".into(),
                    cx,
                )
            })
            .expect("host selected Gradle graph")
            .await
            .expect("Gradle graph admitted");
        server_cx.run_until_parked();
        for version in [4, 5, project::jvm_project::JDTLS_SNAPSHOT_PROTOCOL_VERSION] {
            let response = client
                .request(proto::GetJdtlsProjectSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: version,
                    request_id: 90 + u64::from(version),
                })
                .await
                .expect("selected graphs over framed snapshot");
            let maven = response
                .projects
                .iter()
                .find(|project| project.module_root == "application")
                .expect("Maven wire owner")
                .maven
                .as_ref()
                .expect("Maven metadata");
            let gradle = response
                .projects
                .iter()
                .find(|project| project.module_root == "library")
                .expect("Gradle wire owner")
                .gradle
                .as_ref()
                .expect("Gradle metadata");
            assert_eq!(maven.dependency_graph_json.is_some(), version >= 5);
            assert_eq!(gradle.dependency_graph_json.is_some(), version >= 5);
            let options = &gradle.tasks.first().expect("host task").options;
            let mode = options
                .iter()
                .find(|option| option.name == "mode")
                .expect("host enum");
            let empty_mode = options
                .iter()
                .find(|option| option.name == "emptyMode")
                .expect("host empty enum");
            if version >= 6 {
                assert_eq!(mode.enum_constants_known, Some(true));
                assert_eq!(mode.enum_constants, ["ALPHA", "ZULU"]);
                assert_eq!(empty_mode.enum_constants_known, Some(true));
                assert!(empty_mode.enum_constants.is_empty());
            } else {
                assert!(mode.enum_constants_known.is_none() && mode.enum_constants.is_empty());
                assert!(
                    empty_mode.enum_constants_known.is_none()
                        && empty_mode.enum_constants.is_empty()
                );
            }
            for payload in [
                maven.dependency_graph_json.as_ref(),
                gradle.dependency_graph_json.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                assert!(
                    !payload.contains("file:")
                        && !payload.contains("/remote-jdtls")
                        && !payload.contains("/private/host")
                );
            }
        }
        lsp_store.update(panel_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx)
        });
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(panel_cx, |store, _| {
            let (snapshot, current) = store
                .jdtls_project_snapshot(worktree_id)
                .expect("remote graph admission");
            current
                && snapshot.projects.iter().any(|project| {
                    project
                        .maven
                        .as_ref()
                        .is_some_and(|maven| maven.dependency_graph.is_some())
                })
                && snapshot.projects.iter().any(|project| {
                    project
                        .gradle
                        .as_ref()
                        .is_some_and(|gradle| gradle.dependency_graph.is_some())
                })
        }));
        fs.insert_file(
            std::path::Path::new("/remote-jdtls/build.gradle.kts"),
            b"plugins { java }; // external edit".to_vec(),
        )
        .await;
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(host_lsp_store.read_with(server_cx, |store, _| {
            !store
                .jdtls_project_snapshot(worktree_id)
                .expect("stale host facts")
                .1
        }));
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            1,
            "external edits invalidate without configuring"
        );
        let mut client_cx = panel_cx.cx.clone();
        let (generation_started, generation_started_receiver) =
            futures::channel::oneshot::channel();
        let (generation_release, generation_release_receiver) =
            futures::channel::oneshot::channel();
        *query_gate.lock().expect("generation query gate") =
            Some((generation_started, generation_release_receiver));
        let held_response = client.request(proto::GetJdtlsProjectSnapshot {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            worktree_id: worktree_id.to_proto(),
            protocol_version: project::jvm_project::JDTLS_SNAPSHOT_PROTOCOL_VERSION,
            request_id: 72,
        });
        futures::pin_mut!(held_response);
        let held_response =
            match futures::future::select(held_response, generation_started_receiver).await {
                futures::future::Either::Right((Ok(()), response)) => response,
                _ => panic!("uncached read must reach the held nonexecuting query"),
            };
        fs.insert_file(
            std::path::Path::new("/remote-jdtls/build.gradle.kts"),
            b"plugins { java }; // second external edit".to_vec(),
        )
        .await;
        server_cx.run_until_parked();
        generation_release
            .send(())
            .expect("release generation-held query");
        let rejected = held_response
            .await
            .expect_err("late host input generation must reject");
        assert!(
            rejected
                .to_string()
                .contains("inputs changed during refresh")
        );
        assert_eq!(
            configuring_queries.load(Ordering::SeqCst),
            1,
            "late read must not configure"
        );

        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        *query_gate.lock().expect("query gate lock") = Some((started, release_receiver));
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx)
        });
        let deadline = client_cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        client_cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.restrict(
                    worktree_store.downgrade(),
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        client_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
        }));
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Restricted(_)
        ));
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx)
        });
        server_cx.run_until_parked();
        client_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
        }));
        server_cx.update(|cx| {
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.restrict(
                    worktree_store.downgrade(),
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        release.send(()).expect("release held JDTLS query");
        server_cx.run_until_parked();
        client_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
        }));
        assert!(
            client
                .request(proto::GetJdtlsProjectSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: 1,
                    request_id: 9,
                })
                .await
                .is_err()
        );
        let host_trust = server_cx
            .update(|cx| cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>());
        assert!(
            client
                .request(proto::GetJdtlsProjectSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: 1,
                    request_id: 10,
                })
                .await
                .is_err()
        );
        server_cx.update(|cx| {
            cx.set_global(host_trust);
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("restored host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        client_cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        server_cx.run_until_parked();
        client_cx.run_until_parked();
        let restored_server = fake_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("fresh host JDTLS provider after renewed trust");
        assert_ne!(
            restored_server.server.server_id(),
            original_server.server.server_id()
        );
        assert!(
            host_lsp_store.read_with(server_cx, |store, _| {
                store
                    .jdtls_project_snapshot(worktree_id)
                    .is_none_or(|(snapshot, _)| {
                        snapshot
                            .projects
                            .iter()
                            .all(|project| project.gradle.is_none())
                    })
            }),
            "renewed trust/server must not recover old configuring metadata"
        );
        assert_eq!(configuring_queries.load(Ordering::SeqCst), 1);
        host_lsp_store
            .update(server_cx, |store, cx| {
                store.synchronize_local_gradle_project(Some(worktree_id), cx)
            })
            .expect("explicit synchronization after trust/server recovery");
        server_cx
            .executor()
            .advance_clock(std::time::Duration::from_millis(300));
        server_cx.run_until_parked();
        assert_eq!(configuring_queries.load(Ordering::SeqCst), 2);
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx)
        });
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(_)
        ));
        assert_eq!(rows.len(), 3);
        assert_eq!(navigation.len(), 4);
        let previous_generation = lsp_store.read_with(&client_cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .expect("current remote JDTLS generation")
                .0
                .generation
        });
        let remote = project.read_with(&client_cx, |project, _| {
            project.remote_client().expect("remote project client")
        });
        remote.update(&mut client_cx, |remote, cx| {
            remote.force_heartbeat_timeout(0, cx)
        });
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
        }));
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Disconnected(_)
        ));
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert_eq!(
            remote.read_with(&client_cx, |remote, _| remote.connection_state()),
            remote::ConnectionState::Connected
        );
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(_)
        ));
        assert_eq!(rows.len(), 3);
        assert_eq!(navigation.len(), 4);
        let restored_generation = lsp_store.read_with(&client_cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .expect("new remote JDTLS generation")
                .0
                .generation
        });
        assert!(restored_generation > previous_generation);
        remote.update(&mut client_cx, |remote, cx| {
            remote.force_server_not_running(cx)
        });
        panel_cx.run_until_parked();
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Disconnected(_)
        ));
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        panel_cx.deactivate_window();
        container.update_in(panel_cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn remote_jvm_workspace_providers_are_host_reduced_and_permission_checked(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/remote-kotlin"),
            json!({
                "application": {
                    "build.gradle.kts": "plugins { kotlin(\"jvm\") }",
                    "Main.java": "class Main {}",
                    "src": { "main": { "kotlin": { "Main.kt": "class Main" } } }
                }
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        project.update(cx, |project, _| {
            project.languages().register_test_language(LanguageConfig {
                name: "Kotlin".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["kt".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            });
            project.languages().register_fake_lsp_adapter(
                "Kotlin",
                FakeLspAdapter {
                    name: "kotlin-lsp",
                    ..Default::default()
                },
            );
            project.languages().register_test_language(LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            });
            project.languages().register_fake_lsp_adapter(
                "Java",
                FakeLspAdapter {
                    name: "jdtls",
                    ..Default::default()
                },
            );
        });
        server_cx.update(|cx| {
            let languages = &headless.read(cx).languages;
            languages.register_test_language(LanguageConfig {
                name: "Kotlin".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["kt".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            });
            languages.register_fake_lsp_adapter(
                "Kotlin",
                FakeLspAdapter {
                    name: "kotlin-lsp",
                    ..Default::default()
                },
            );
            languages.register_test_language(LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            });
            languages.register_fake_lsp_adapter(
                "Java",
                FakeLspAdapter {
                    name: "jdtls",
                    ..Default::default()
                },
            );
        });
        let query_gate = Arc::new(std::sync::Mutex::new(
            None::<(
                futures::channel::oneshot::Sender<()>,
                futures::channel::oneshot::Receiver<()>,
            )>,
        ));
        let mut fake_servers = server_cx.update(|cx| {
            headless.read(cx).languages.register_fake_lsp_server(
                LanguageServerName::new_static("kotlin-lsp"),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec!["exportWorkspace".to_string()],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                Some(Box::new({
                    let query_gate = query_gate.clone();
                    move |fake| {
                    fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>({
                        let query_gate = query_gate.clone();
                        move |params, _| {
                            let query_gate = query_gate.clone();
                            async move {
                            anyhow::ensure!(params.command == "exportWorkspace");
                            let gate = query_gate.lock().expect("query gate lock").take();
                            if let Some((started, release)) = gate {
                                started.send(()).map_err(|_| anyhow::anyhow!("query start receiver closed"))?;
                                release.await?;
                            }
                            let directory = params
                                .arguments
                                .first()
                                .and_then(serde_json::Value::as_str)
                                .ok_or_else(|| anyhow::anyhow!("missing export directory"))?;
                            std::fs::write(
                                std::path::Path::new(directory).join("workspace.json"),
                                serde_json::to_vec(&json!({
                                    "externalSystem": "GRADLE",
                                    "libraries": [
                                        {"name": "Gradle: org.example:api:1.2.3"},
                                        {"name": "/private/host/cache.jar"}
                                    ],
                                    "modules": [{
                                        "name": "application.main",
                                        "externalProjectPath": "/remote-kotlin/application",
                                        "contentRoots": [{"sourceRoots": [
                                            {"path": "/remote-kotlin/application/src/main/kotlin", "generated": true},
                                            {"path": "/private/host/source"}
                                        ]}],
                                        "dependencies": [
                                            {"type": "library", "name": "Gradle: org.example:api:1.2.3", "scope": "compile"},
                                            {"type": "library", "name": "/private/host/cache.jar", "scope": "test"}
                                        ]
                                    }]
                                }))?,
                            )?;
                            Ok(None)
                            }
                        }
                    });
                    }
                })),
            )
        });
        let mut java_servers = server_cx.update(|cx| {
            headless.read(cx).languages.register_fake_lsp_server(
                LanguageServerName::new_static("jdtls"),
                lsp::ServerCapabilities {
                    execute_command_provider: Some(lsp::ExecuteCommandOptions {
                        commands: vec![
                            "java.project.getAll".to_string(),
                            "java.project.getSettings".to_string(),
                        ],
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                Some(Box::new(|fake| {
                    fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>(
                        |params, _| async move {
                            match params.command.as_str() {
                                "java.project.getAll" => {
                                    Ok(Some(json!(["file:///remote-kotlin/application"])))
                                }
                                "java.project.getSettings" => Ok(Some(json!({
                                    "org.eclipse.jdt.ls.core.sourcePaths": [],
                                    "org.eclipse.jdt.ls.core.classpathEntries": [{
                                        "kind": 1,
                                        "path": "/private/host/junit-jupiter.jar",
                                        "attributes": {"gradle_used_by_scope": "main,test"}
                                    }]
                                }))),
                                _ => anyhow::bail!("unexpected JDTLS command"),
                            }
                        },
                    );
                })),
            )
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/remote-kotlin"), true, cx)
            })
            .await
            .expect("remote Kotlin worktree");
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        server_cx.update(|cx| {
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        let kotlin_buffer = project
            .update(cx, |project, cx| {
                project.open_buffer_with_lsp(
                    (worktree_id, rel_path("application/src/main/kotlin/Main.kt")),
                    cx,
                )
            })
            .await
            .expect("open remote Kotlin source");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let _kotlin_server = fake_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("host Kotlin provider");
        cx.run_until_parked();
        server_cx.run_until_parked();

        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let response = client
            .request(proto::GetKotlinWorkspaceSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_id: worktree_id.to_proto(),
                protocol_version: 1,
                request_id: 7,
            })
            .await
            .expect("host-reduced Kotlin snapshot");
        assert_eq!(response.request_id, 7);
        assert!(response.partial);
        assert_eq!(response.projects.len(), 1);
        assert!(!format!("{response:?}").contains("/private/host"));
        assert_eq!(response.projects[0].module_root, "application");
        assert_eq!(response.projects[0].build_system, "GRADLE");
        assert_eq!(response.projects[0].source_roots.len(), 1);
        assert!(response.projects[0].source_roots[0].generated);
        assert_eq!(response.projects[0].dependencies.len(), 1);
        assert_eq!(
            response.projects[0].dependencies[0].name,
            "org.example:api:1.2.3"
        );
        assert!(
            client
                .request(proto::GetKotlinWorkspaceSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: u32::MAX,
                    request_id: 8,
                })
                .await
                .is_err()
        );
        let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
        lsp_store.update(cx, |store, cx| {
            store.refresh_kotlin_workspace(worktree_id, cx)
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        assert!(lsp_store.read_with(cx, |store, _| {
            store
                .kotlin_workspace_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            jvm_ui::init(cx);
        });
        let (container, panel_cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(panel_cx, |container, _| container.workspace().clone());
        let async_cx = panel_cx.update(|window, cx| window.to_async(cx));
        let panel = jvm_ui::DependenciesPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("remote Groovy Dependencies panel");
        panel.update_in(panel_cx, |panel, window, cx| {
            panel.set_active(true, window, cx);
        });
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Partial(_)
        ));
        assert_eq!(rows.len(), 1);
        assert!(rows[0].accessibility_label.contains("Kotlin LSP module"));
        assert!(
            navigation
                .values()
                .any(|path| path.path.as_unix_str() == "application/build.gradle.kts")
        );
        let mut client_cx = panel_cx.cx.clone();
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        *query_gate.lock().expect("query gate lock") = Some((started, release_receiver));
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_kotlin_workspace(worktree_id, cx)
        });
        let deadline = client_cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        client_cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.restrict(
                    worktree_store.downgrade(),
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        client_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.kotlin_workspace_snapshot(worktree_id).is_none()
        }));
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Restricted(_)
        ));
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        release.send(()).expect("release held Kotlin export");
        server_cx.run_until_parked();
        client_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.kotlin_workspace_snapshot(worktree_id).is_none()
        }));
        server_cx.update(|cx| {
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.restrict(
                    worktree_store.downgrade(),
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        assert!(
            client
                .request(proto::GetKotlinWorkspaceSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: 1,
                    request_id: 9,
                })
                .await
                .is_err()
        );
        server_cx.update(|cx| {
            let worktree_store = headless.read(cx).worktree_store.clone();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("host trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        client_cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_kotlin_workspace(worktree_id, cx)
        });
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store
                .kotlin_workspace_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        let java_buffer = project
            .update(&mut client_cx, |project, cx| {
                project.open_buffer_with_lsp((worktree_id, rel_path("application/Main.java")), cx)
            })
            .await
            .expect("open remote Java source");
        client_cx.run_until_parked();
        server_cx.run_until_parked();
        let _java_server = java_servers
            .next()
            .now_or_never()
            .flatten()
            .expect("host JDTLS provider");
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_jdtls_project(worktree_id, cx);
        });
        server_cx.run_until_parked();
        client_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store
                .jdtls_project_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_kotlin_workspace(worktree_id, cx);
        });
        server_cx.run_until_parked();
        client_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store
                .kotlin_workspace_snapshot(worktree_id)
                .is_some_and(|(_, current)| current)
        }));
        let (_, rows, navigation) = panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "JDTLS: application");
        assert_eq!(rows[1].label, "Kotlin LSP: application.main");
        assert_eq!(navigation.len(), 2);

        let (replacement_query_started, replacement_query_started_receiver) =
            futures::channel::oneshot::channel();
        let (release_replacement_query, replacement_query_release_receiver) =
            futures::channel::oneshot::channel();
        *query_gate.lock().expect("query gate lock") = Some((
            replacement_query_started,
            replacement_query_release_receiver,
        ));
        lsp_store.update(&mut client_cx, |store, cx| {
            store.refresh_kotlin_workspace(worktree_id, cx)
        });
        server_cx.run_until_parked();
        let deadline = client_cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(replacement_query_started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        drop(java_buffer);
        drop(kotlin_buffer);
        project.update(&mut client_cx, |project, cx| {
            project.remove_worktree(worktree_id, cx)
        });
        client_cx.run_until_parked();
        server_cx.run_until_parked();
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
                && store.kotlin_workspace_snapshot(worktree_id).is_none()
        }));
        let (_, rows, navigation) = panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        drop(worktree);
        client_cx.run_until_parked();
        server_cx.run_until_parked();
        let (replacement, _) = project
            .update(&mut client_cx, |project, cx| {
                project.find_or_create_worktree(path!("/remote-kotlin"), true, cx)
            })
            .await
            .expect("same-path JVM worktree replacement");
        let replacement_id = replacement.read_with(&client_cx, |worktree, _| worktree.id());
        assert_ne!(replacement_id, worktree_id);
        release_replacement_query
            .send(())
            .expect("release pre-replacement Kotlin export");
        client_cx.run_until_parked();
        server_cx.run_until_parked();
        let host_worktree_ids = headless.read_with(server_cx, |headless, cx| {
            headless
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).id())
                .collect::<Vec<_>>()
        });
        assert_eq!(host_worktree_ids, vec![replacement_id]);
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.jdtls_project_snapshot(worktree_id).is_none()
                && store.kotlin_workspace_snapshot(worktree_id).is_none()
        }));
        let (_, rows, navigation) = panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        assert!(
            client
                .request(proto::GetJdtlsProjectSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: 1,
                    request_id: 11,
                })
                .await
                .is_err()
        );
        assert!(
            client
                .request(proto::GetKotlinWorkspaceSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    worktree_id: worktree_id.to_proto(),
                    protocol_version: 1,
                    request_id: 12,
                })
                .await
                .is_err()
        );

        let remote = project.read_with(&client_cx, |project, _| {
            project.remote_client().expect("remote project client")
        });
        remote.update(&mut client_cx, |remote, cx| {
            remote.force_server_not_running(cx)
        });
        panel_cx.run_until_parked();
        assert!(lsp_store.read_with(&client_cx, |store, _| {
            store.kotlin_workspace_snapshot(worktree_id).is_none()
        }));
        let (status, rows, navigation) =
            panel.read_with(panel_cx, |panel, _| panel.snapshot_for_test());
        assert!(matches!(
            status,
            language_tools::language_tool_tree::LanguageToolTreeStatus::Disconnected(_)
        ));
        assert!(rows.is_empty());
        assert!(navigation.is_empty());
        panel_cx.deactivate_window();
        container.update_in(panel_cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn remote_java_discovery_uses_the_host_and_rejects_mismatched_requests(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm"),
            json!({
                "application": {
                    "pom.xml": "<project/>",
                    "src": { "test": { "java": { "evidence": {
                        "GreetingTest.java": "package evidence; class GreetingTest {}"
                    } } } },
                    "target": { "surefire-reports": {
                        "TEST-evidence.GreetingTest.xml": include_str!("../../project/test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml")
                    } }
                }
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let mismatch = client
            .request(proto::GetJvmTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: u32::MAX,
                worktree_ids: vec![worktree_id],
                request_id: 1,
            })
            .await
            .expect("typed JVM mismatch response");
        assert_eq!(
            mismatch.status,
            proto::StructuredProviderStatus::Mismatch as i32
        );
        let scoped = client
            .request(proto::GetJvmTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: Vec::new(),
                request_id: 2,
            })
            .await
            .expect("scoped JVM discovery");
        assert_ne!(
            scoped.status,
            proto::StructuredProviderStatus::Mismatch as i32
        );
        let host_snapshot = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&project::structured_execution::StructuredProviderId(
                    project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.to_string(),
                ))
                .cloned()
        });
        assert!(
            host_snapshot
                .expect("host discovery")
                .nodes
                .iter()
                .all(|node| node.kind != StructuredNodeKind::Case)
        );

        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        assert!(project.read_with(cx, |project, cx| {
            project.worktree_store().read(cx).fs().is_none()
        }));
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("client receives host Java cases");
        assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
        let case = snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .expect("host Java case");
        let action_error = provider.read_with(cx, |provider, cx| {
            provider
                .plan_selected_case(&case.id, snapshot.discovery_generation, cx)
                .expect_err("remote action remains unavailable")
        });
        assert!(action_error.to_string().contains("host action protocol"));
    }

    #[gpui::test]
    async fn remote_java_results_hide_a_module_after_host_privacy_revocation(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm"),
            json!({
                "application": {
                    "pom.xml": "<project/>",
                    "src": { "test": { "java": { "evidence": {
                        "GreetingTest.java": "package evidence; class GreetingTest {}"
                    } } } },
                    "target": { "surefire-reports": {
                        "TEST-evidence.GreetingTest.xml": include_str!("../../project/test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml")
                    }, "generated-sources": {} }
                }
            }),
        )
        .await;
        fs.insert_tree(
            path!("/outside"),
            json!({ "External.java": "class External {}" }),
        )
        .await;
        fs.insert_symlink(
            path!("/jvm/application/target/generated-sources/External.java"),
            PathBuf::from(path!("/outside/External.java")),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("host JVM worktree");
            assert!(
                host_worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path(
                        "application/target/generated-sources/External.java"
                    ))
                    .expect("outside-root source entry")
                    .is_external
            );
        });
        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        let discovered = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("host Java discovery");
        let case = discovered
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .expect("host Java case");
        let run_id = project::structured_execution::StructuredRunId("private-java-run".into());
        headless.update(server_cx, |headless, cx| {
            headless.structured_execution_store.update(cx, |store, _| {
                use project::structured_execution::{
                    StructuredExecutionEvent, StructuredNodeState, StructuredRun,
                    StructuredRunPhase,
                };

                let generation = store.state().project_generation();
                let provider_id = discovered.provider_id.clone();
                store
                    .state_mut()
                    .begin_run(
                        generation,
                        &provider_id,
                        StructuredRun::new(
                            run_id.clone(),
                            discovered.discovery_generation,
                            vec![case.id.clone()],
                        ),
                    )
                    .expect("host Java run begins");
                store
                    .state_mut()
                    .apply_event(
                        generation,
                        &provider_id,
                        &run_id,
                        StructuredExecutionEvent {
                            sequence: 0,
                            node_id: case.id.clone(),
                            state: StructuredNodeState::Passed,
                            duration_millis: None,
                            message: Some("private Java result".into()),
                            location: case.path.clone(),
                        },
                        None,
                    )
                    .expect("host Java result applies");
                store
                    .state_mut()
                    .apply_event(
                        generation,
                        &provider_id,
                        &run_id,
                        StructuredExecutionEvent {
                            sequence: 1,
                            node_id: case.id.clone(),
                            state: StructuredNodeState::Failed,
                            duration_millis: None,
                            message: Some("outside-root Java result".into()),
                            location: Some(ProjectPath {
                                worktree_id,
                                path: rel_path(
                                    "application/target/generated-sources/External.java",
                                )
                                .into(),
                            }),
                        },
                        None,
                    )
                    .expect("outside-root Java result applies");
                store
                    .state_mut()
                    .set_run_phase(
                        generation,
                        &provider_id,
                        &run_id,
                        StructuredRunPhase::Completed,
                        None,
                    )
                    .expect("host Java run completes");
            })
        });

        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let request = || proto::GetStructuredExecutionSnapshot {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            provider_id: project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.to_string(),
            discovery_generation: 0,
            page_start: 0,
            page_size: 100,
            worktree_ids: vec![worktree_id.to_proto()],
        };
        let before = client
            .request(request())
            .await
            .expect("visible host Java result");
        assert!(
            before
                .nodes
                .iter()
                .any(|node| node.label == "messageIsAvailable")
        );
        assert_eq!(
            before
                .last_complete_run
                .as_ref()
                .and_then(|run| run.summary.as_ref())
                .map(|summary| summary.passed),
            Some(1)
        );
        assert_eq!(
            before
                .last_complete_run
                .as_ref()
                .and_then(|run| run.summary.as_ref())
                .map(|summary| summary.failed),
            Some(0)
        );
        let visible_events = client
            .request(proto::GetStructuredExecutionEvents {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.to_string(),
                run_id: run_id.0.clone(),
                start_sequence: 0,
                chunk_size: 100,
                worktree_ids: vec![worktree_id.to_proto()],
            })
            .await
            .expect("visible Java events");
        assert_eq!(visible_events.events.len(), 1);

        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/pom.xml".into()]));
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("host JVM worktree");
            assert!(
                host_worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("application/pom.xml"))
                    .expect("host JVM manifest")
                    .is_private
            );
        });
        let after = client
            .request(request())
            .await
            .expect("privacy-filtered host Java result");
        assert!(
            after
                .nodes
                .iter()
                .all(|node| node.label != "messageIsAvailable")
        );
        assert!(after.partial);
        assert_eq!(after.status, proto::StructuredProviderStatus::Stale as i32);
        assert!(after.current_run.is_none());
        assert!(after.last_complete_run.is_none());
        let event_error = client
            .request(proto::GetStructuredExecutionEvents {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.to_string(),
                run_id: run_id.0,
                start_sequence: 0,
                chunk_size: 100,
                worktree_ids: vec![worktree_id.to_proto()],
            })
            .await
            .expect_err("a peer cannot fetch events from a revoked Java run");
        assert!(event_error.to_string().contains("unknown structured run"));
    }

    #[gpui::test]
    async fn remote_java_discovery_rejects_a_reply_from_before_reconnect(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(path!("/jvm"), json!({ "pom.xml": "<project/>" }))
            .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        let first = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("first Java discovery");
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless.jvm_test_provider_store.update(cx, |provider, _| {
                provider.gate_next_discovery_reply_for_test()
            })
        });
        let pending = provider.update(cx, |provider, cx| provider.refresh(cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(1, cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(pending, deadline).await,
            futures::future::Either::Left((Err(_), _))
        ));
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            futures::pin_mut!(recovery);
            assert!(matches!(
                futures::future::select(recovery, deadline).await,
                futures::future::Either::Left(_)
            ));
        }
        release.send(()).expect("held Java discovery reply");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(completed, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        structured.read_with(cx, |store, _| {
            let snapshot = store
                .state()
                .provider(&project::structured_execution::StructuredProviderId(
                    project::jvm_test_provider::JVM_JUNIT_PROVIDER_ID.to_string(),
                ))
                .expect("cached Java discovery");
            assert_eq!(snapshot.discovery_generation, first.discovery_generation);
            assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
        });
        let refreshed = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("fresh Java discovery after reconnect");
        assert!(refreshed.discovery_generation > first.discovery_generation);
    }

    #[gpui::test]
    async fn remote_java_discovery_rejects_a_reply_after_same_path_replacement(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm"),
            json!({
                "pom.xml": "<project/>",
                "mvnw": "#!/bin/sh\n",
                "src": { "test": { "java": {
                    "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
                } } },
                "target": { "surefire-reports": {
                    "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
                } }
            }),
        )
        .await;
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        headless.update(server_cx, |host, cx| {
            project::trusted_worktrees::track_worktree_trust(
                host.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let java = Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        )
        .with_queries(LanguageQueries {
            runnables: Some(Cow::Borrowed(
                "(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag \"java-test-method\"))",
            )),
            ..Default::default()
        })
        .expect("fixture Java runnable query");
        headless.read_with(server_cx, |headless, _| {
            headless.languages.add(Arc::new(java));
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let original_id = worktree.read_with(cx, |worktree, _| worktree.id());
        headless.update(server_cx, |host, cx| {
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("replacement fixture host trust");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &host.worktree_store,
                    [project::trusted_worktrees::PathTrust::AbsPath(
                        path!("/jvm").into(),
                    )]
                    .into_iter()
                    .collect(),
                    cx,
                );
            });
        });
        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        let first = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("first Java discovery");
        let original_case = first
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .expect("case from the original worktree")
            .id
            .clone();
        assert!(
            first
                .nodes
                .iter()
                .any(|node| node.kind == StructuredNodeKind::Group
                    && node
                        .path
                        .as_ref()
                        .is_some_and(|path| path.worktree_id == original_id)),
            "original discovery must belong to the original worktree"
        );
        let original_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&original_case, first.discovery_generation, None, cx)
            })
            .await
            .expect("source-backed case can be authorized before replacement");
        assert_eq!(original_plan.worktree_id, original_id);
        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless.jvm_test_provider_store.update(cx, |provider, _| {
                provider.gate_next_discovery_reply_for_test()
            })
        });
        let pending = provider.update(cx, |provider, cx| provider.refresh(cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));

        project.update(cx, |project, cx| project.remove_worktree(original_id, cx));
        drop(worktree);
        cx.run_until_parked();
        server_cx.run_until_parked();
        let (replacement, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("same-path Java worktree replacement");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let replacement_id = replacement.read_with(cx, |worktree, _| worktree.id());
        assert_ne!(replacement_id, original_id);
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(pending, deadline).await,
            futures::future::Either::Left((Err(_), _))
        ));
        release.send(()).expect("held old Java discovery reply");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(completed, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        assert!(
            provider
                .update(cx, |provider, cx| {
                    provider.plan_remote_action(
                        &original_case,
                        first.discovery_generation,
                        None,
                        cx,
                    )
                })
                .await
                .is_err(),
            "a delayed reply must not authorize the replaced Java case"
        );
        let refreshed = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("fresh Java discovery after same-path replacement");
        assert!(refreshed.discovery_generation > first.discovery_generation);
        assert!(refreshed.nodes.iter().all(|node| {
            node.id != original_case
                && node
                    .path
                    .as_ref()
                    .is_none_or(|path| path.worktree_id == replacement_id)
        }));
        assert!(refreshed.nodes.iter().any(|node| {
            node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
        }));
        assert!(
            refreshed
                .nodes
                .iter()
                .any(|node| node.kind == StructuredNodeKind::Group
                    && node
                        .path
                        .as_ref()
                        .is_some_and(|path| path.worktree_id == replacement_id)),
            "replacement discovery must belong to the new worktree"
        );
        let replacement_case = refreshed
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .expect("case from the replacement worktree");
        let replacement_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(
                    &replacement_case.id,
                    refreshed.discovery_generation,
                    None,
                    cx,
                )
            })
            .await
            .expect("replacement source-backed case can be authorized");
        assert_eq!(replacement_plan.worktree_id, replacement_id);
    }

    #[cfg(feature = "doctor-jvm-test")]
    #[gpui::test]
    async fn doctor_jvm_reproduction_native_inspection_requires_approval_and_exact_execution_review(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm"),
            json!({
                "pom.xml": "<project/>",
                "mvnw": "#!/bin/sh\n",
                "src": { "test": { "java": {
                    "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
                } } },
                "target": { "surefire-reports": {
                    "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
                } }
            }),
        )
        .await;
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let java = Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        )
        .with_queries(LanguageQueries {
            runnables: Some(Cow::Borrowed(
                "(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag \"java-test-method\"))",
            )),
            ..Default::default()
        })
        .expect("fixture Java runnable query");
        headless.read_with(server_cx, |headless, _| {
            headless.languages.add(Arc::new(java));
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("host Java discovery");
        let _initial_case = snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .unwrap_or_else(|| panic!("source-backed host case: {snapshot:#?}"));
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("Explicit host trust");
        let snapshot = provider
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("Current selected host discovery");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("Selected case");
        use diagnostics::doctor::{
            self, DiagnosticText, DoctorHistoryStore, DoctorProduct, DoctorRepairBinding,
            DoctorRepairPhase, DoctorReport, DoctorScope,
        };
        let directory = ProjectPath {
            worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
            path: util::rel_path::RelPath::empty_arc(),
        };
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store
                    .set_user_settings("{}", cx)
                    .expect("Known configuration revision");
            });
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                project.read(cx).worktree_store(),
                None,
                None,
                None,
                cx,
            );
        });
        let trust = cx.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("Existing client trust owner")
        });
        let store = project.read_with(cx, |owner, _| owner.worktree_store());
        trust.update(cx, |trust, cx| {
            trust.trust(
                &store,
                HashSet::from_iter([project::trusted_worktrees::PathTrust::Worktree(
                    directory.worktree_id,
                )]),
                cx,
            )
        });
        let database = db::AppDatabase::test_new();
        let history = cx.update(|cx| {
            cx.set_global(database);
            DoctorHistoryStore::for_app(cx).expect("Existing application history owner")
        });
        let scope = DoctorScope {
            product: DiagnosticText::new("Groovy"),
            project: DiagnosticText::new("selected fixture"),
            worktree: DiagnosticText::new("selected root"),
            host: DiagnosticText::new("owned remote development host"),
        };
        let create_review = |cx: &gpui::App| {
            doctor::prepare_remote_test_inspection(
                &project,
                directory.clone(),
                case.id.clone(),
                snapshot.discovery_generation,
                1,
                10,
                cx,
            )
            .expect("Native reviewed metadata proposal")
        };
        let review = cx.update(|cx| create_review(cx));
        assert_eq!(review.phase(), DoctorRepairPhase::Proposed);
        let mut report = DoctorReport::new(scope.clone(), 1, 10);
        report.push(review.before().clone()).expect("Scoped report");
        let (cancel, cancelled) = async_channel::bounded(1);
        assert!(
            cx.update(|cx| doctor::inspect_remote_test(
                review,
                project.clone(),
                report,
                DoctorProduct::Groovy,
                history.clone(),
                10,
                cancelled,
                cx
            ))
            .await
            .is_err(),
            "Unapproved proposal must not inspect host reports"
        );
        assert!(
            history
                .records(DoctorProduct::Groovy, 10)
                .await
                .expect("No unauthorized receipt")
                .is_empty()
        );
        let binding = cx
            .update(|cx| {
                DoctorRepairBinding::capture(
                    &project,
                    directory.clone(),
                    1,
                    cx.global::<SettingsStore>()
                        .observed_revision()
                        .expect("Revision"),
                    cx,
                )
            })
            .expect("Selected inspection authority");
        let mut cancelled_review = cx.update(|cx| create_review(cx));
        cancelled_review
            .approve(&binding)
            .expect("Explicit metadata consent");
        let mut cancelled_report = DoctorReport::new(scope.clone(), 1, 10);
        cancelled_report
            .push(cancelled_review.before().clone())
            .expect("Selected report");
        let host_provider =
            headless.read_with(server_cx, |owner, _| owner.jvm_test_provider_store.clone());
        let (captured, release) = host_provider.update(server_cx, |owner, _| {
            owner.gate_next_action_reply_for_test()
        });
        let (cancel_inspection, cancelled) = async_channel::bounded(1);
        let pending = cx.update(|cx| {
            doctor::inspect_remote_test(
                cancelled_review,
                project.clone(),
                cancelled_report,
                DoctorProduct::Groovy,
                history.clone(),
                10,
                cancelled,
                cx,
            )
        });
        captured
            .await
            .expect("Durable receipt precedes approved host collection");
        let mut duplicate = cx.update(|cx| create_review(cx));
        duplicate
            .approve(&binding)
            .expect("Independently reviewed duplicate");
        let mut duplicate_report = DoctorReport::new(scope.clone(), 1, 10);
        duplicate_report
            .push(duplicate.before().clone())
            .expect("Selected report");
        let (_duplicate_cancel, cancelled) = async_channel::bounded(1);
        assert!(
            cx.update(|cx| doctor::inspect_remote_test(
                duplicate,
                project.clone(),
                duplicate_report,
                DoctorProduct::Groovy,
                history.clone(),
                10,
                cancelled,
                cx
            ))
            .await
            .is_err(),
            "Duplicate active proposal cannot read another baseline or create another receipt"
        );
        cancel_inspection.close();
        let (cancelled_review, inspected) = pending.await.expect("Observed cancellation");
        assert_eq!(cancelled_review.phase(), DoctorRepairPhase::Cancelled);
        assert!(inspected.is_none());
        cx.run_until_parked();
        server_cx.run_until_parked();
        release.send(()).expect("Release abandoned host reply");
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert_eq!(
            history
                .records(DoctorProduct::Groovy, 10)
                .await
                .expect("One cancelled receipt")
                .len(),
            1
        );
        for scenario in ["drop", "revoke"] {
            let mut review = cx.update(|cx| create_review(cx));
            let binding = cx
                .update(|cx| {
                    DoctorRepairBinding::capture(
                        &project,
                        directory.clone(),
                        1,
                        cx.global::<SettingsStore>()
                            .observed_revision()
                            .expect("Revision"),
                        cx,
                    )
                })
                .expect("Current selected authority");
            review
                .approve(&binding)
                .expect("Explicit selected inspection consent");
            let mut scoped_report = DoctorReport::new(scope.clone(), 1, 10);
            scoped_report
                .push(review.before().clone())
                .expect("Selected report");
            let (captured, release) = host_provider.update(server_cx, |owner, _| {
                owner.gate_next_action_reply_for_test()
            });
            let (_cancel, cancelled) = async_channel::bounded(1);
            let mut pending = Some(cx.update(|cx| {
                doctor::inspect_remote_test(
                    review,
                    project.clone(),
                    scoped_report,
                    DoctorProduct::Groovy,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            }));
            captured.await.expect("Bounded host collection pending");
            if scenario == "drop" {
                drop(pending.take());
            } else {
                cx.update(|cx| {
                    cx.update_global::<SettingsStore, _>(|store, cx| {
                        store
                            .set_user_settings("{\"vim_mode\":true}", cx)
                            .expect("Inputs revoked");
                    })
                });
            }
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(100));
            cx.run_until_parked();
            server_cx.run_until_parked();
            if let Some(pending) = pending {
                let (review, plan) = pending.await.expect("Observed stale inspection");
                assert_eq!(review.phase(), DoctorRepairPhase::Incomplete);
                assert!(plan.is_none());
            }
            release.send(()).expect("Release obsolete host reply");
            cx.run_until_parked();
            server_cx.run_until_parked();
        }
        let mut review = cx.update(|cx| create_review(cx));
        let binding = cx
            .update(|cx| {
                DoctorRepairBinding::capture(
                    &project,
                    directory.clone(),
                    1,
                    cx.global::<SettingsStore>()
                        .observed_revision()
                        .expect("Revision"),
                    cx,
                )
            })
            .expect("Selected authority");
        review
            .approve(&binding)
            .expect("Explicit inspection approval");
        let mut report = DoctorReport::new(scope, 1, 10);
        report.push(review.before().clone()).expect("Scoped report");
        let (_cancel, cancelled) = async_channel::bounded(1);
        let (review, inspected) = cx
            .update(|cx| {
                doctor::inspect_remote_test(
                    review,
                    project.clone(),
                    report,
                    DoctorProduct::Groovy,
                    history.clone(),
                    10,
                    cancelled,
                    cx,
                )
            })
            .await
            .expect("Bounded host observation");
        assert_eq!(
            review.phase(),
            DoctorRepairPhase::Incomplete,
            "Metadata inspection never verifies a repair"
        );
        let selected = doctor::DoctorInvestigationSelection {
            scope: doctor::DoctorInvestigationScope::Application,
            problem: doctor::DoctorInvestigationProblem::Test,
            symptom: DiagnosticText::new("Selected failing test"),
            validation_target: DiagnosticText::new("Selected Java case"),
            source_revision: DiagnosticText::new("Saved fixture source"),
            cache_conditions: DiagnosticText::new("Existing development cache"),
            workload: DiagnosticText::new("One reviewed functional test"),
            configuration: DiagnosticText::new("Selected Maven wrapper"),
            input_identity: doctor::DoctorInvestigationInputIdentity::from_fields(&[
                "fixture Java source",
            ])
            .expect("Opaque identity"),
        };
        let execution = cx
            .update(|cx| {
                doctor::prepare_inspected_remote_test_reproduction(
                    inspected.expect("Acknowledged host plan"),
                    &project,
                    selected,
                    doctor::DoctorReproductionEffects::Development {
                        file_writes: "Target test outputs and development cache writes".into(),
                        subprocesses: "Maven wrapper and selected Java test".into(),
                    },
                    60,
                    10,
                    cx,
                )
            })
            .expect("Separate exact execution proposal");
        assert_eq!(
            execution.phase(),
            DoctorRepairPhase::Proposed,
            "Inspection approval never authorizes execution"
        );
        assert!(
            execution
                .action()
                .exact_preview()
                .iter()
                .any(|line| line.contains("./mvnw"))
        );
        let records = history
            .records(DoctorProduct::Groovy, 10)
            .await
            .expect("Retained selected inspection");
        assert_eq!(records.len(), 4);
        assert!(!format!("{records:?}").contains("<testsuite>"));
        drop(execution);
        cancel.close();
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.read_with(server_cx, |owner, cx| {
            assert!(
                owner
                    .structured_execution_store
                    .read(cx)
                    .state()
                    .provider(&StructuredProviderId("jvm-junit".into()))
                    .and_then(|provider| provider.current_run.as_ref())
                    .is_none(),
                "Cancelling execution review starts no Task"
            )
        });
    }

    #[gpui::test]
    async fn doctor_jvm_reproduction_acknowledges_limit_and_preserves_active_work(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm"),
            json!({
                "pom.xml": "<project/>",
                "mvnw": "#!/bin/sh\n",
                "src": { "test": { "java": {
                    "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
                } } },
                "target": { "surefire-reports": {
                    "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
                } }
            }),
        )
        .await;
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let java = Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        )
        .with_queries(LanguageQueries {
            runnables: Some(Cow::Borrowed(
                "(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag \"java-test-method\"))",
            )),
            ..Default::default()
        })
        .expect("fixture Java runnable query");
        headless.read_with(server_cx, |headless, _| {
            headless.languages.add(Arc::new(java));
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("host Java discovery");
        let _initial_case = snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .unwrap_or_else(|| panic!("source-backed host case: {snapshot:#?}"));
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("Explicit host trust");
        let snapshot = provider
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("Current selected host discovery");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("Selected case");
        let request = proto::ResolveJvmTestAction {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::jvm_test_provider::JVM_TEST_BOUNDED_ACTION_PROTOCOL_VERSION,
            discovery_generation: snapshot.discovery_generation.0,
            node_id: case.id.0.clone(),
            failed_run_id: None,
            worktree_ids: vec![worktree_id],
            report_maximum_bytes: Some(256),
            requested_run_id: Some(format!("jvm-bounded-{:032x}", 1_u128)),
        };
        for limit in [
            0,
            project::jvm_test_provider::MAX_JUNIT_DISCOVERY_BYTES as u64 + 1,
        ] {
            assert!(
                client
                    .request(proto::ResolveJvmTestAction {
                        report_maximum_bytes: Some(limit),
                        ..request.clone()
                    })
                    .await
                    .is_err()
            );
        }
        let host_provider =
            headless.read_with(server_cx, |owner, _| owner.jvm_test_provider_store.clone());
        let (captured, release) = host_provider.update(server_cx, |owner, _| {
            owner.gate_next_action_reply_for_test()
        });
        let pending = provider.update(cx, |owner, cx| {
            owner.plan_remote_action_with_limit(
                &case.id,
                snapshot.discovery_generation,
                None,
                Some(256),
                cx,
            )
        });
        captured
            .await
            .expect("Host captured approved bounded metadata");
        drop(pending);
        cx.run_until_parked();
        server_cx.run_until_parked();
        release
            .send(())
            .expect("Release cancelled in-flight inspection");
        cx.run_until_parked();
        server_cx.run_until_parked();
        for _ in 0..33 {
            let abandoned = provider
                .update(cx, |owner, cx| {
                    owner.plan_remote_action_with_limit(
                        &case.id,
                        snapshot.discovery_generation,
                        None,
                        Some(256),
                        cx,
                    )
                })
                .await
                .expect("Released reservations do not exhaust owner capacity");
            drop(abandoned);
            cx.run_until_parked();
            server_cx.run_until_parked();
        }
        let reserved = client
            .request(request.clone())
            .await
            .expect("Bounded host plan");
        assert_eq!(reserved.report_maximum_bytes, Some(256));
        headless.read_with(server_cx, |owner, cx| {
            assert!(
                owner
                    .structured_execution_store
                    .read(cx)
                    .state()
                    .provider(&StructuredProviderId("jvm-junit".into()))
                    .and_then(|provider| provider.current_run.as_ref())
                    .is_none(),
                "Planning never executes or begins a run"
            )
        });
        let plan = provider
            .update(cx, |owner, cx| {
                owner.plan_remote_action_with_limit(
                    &case.id,
                    snapshot.discovery_generation,
                    None,
                    Some(256),
                    cx,
                )
            })
            .await
            .expect("Acknowledged client plan");
        assert_eq!(plan.report_maximum_bytes(), Some(256));
        let handle = StructuredTaskHandle::new(TaskId("doctor-owned-remote-test".into()));
        provider
            .update(cx, |owner, cx| {
                owner.register_remote_run_handle(plan.clone(), handle.clone(), cx)
            })
            .expect("Existing owner lifecycle");
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert!(
            provider
                .update(cx, |owner, cx| owner.plan_remote_action_with_limit(
                    &case.id,
                    snapshot.discovery_generation,
                    None,
                    Some(256),
                    cx
                ))
                .await
                .is_err(),
            "Client active work cannot be replaced"
        );
        assert!(
            client.request(request.clone()).await.is_err(),
            "Host active work cannot be replaced"
        );
        let queued = proto::UpdateJvmTestRun {
            abandon_unlaunched: false,
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: reserved.discovery_generation,
            run_id: reserved.run_id,
            node_id: reserved.node_id,
            task_id: "unlaunched-doctor-plan".into(),
            state: proto::JvmTestTaskState::Queued as i32,
            terminal_id: None,
            exit_code: None,
            success: false,
            termination_confirmed: false,
            diagnostic: None,
            worktree_ids: vec![worktree_id],
        };
        assert!(
            client.request(queued.clone()).await.is_err(),
            "An earlier idle grant cannot supersede newly active work"
        );
        client
            .request(proto::UpdateJvmTestRun {
                state: proto::JvmTestTaskState::Cancelled as i32,
                termination_confirmed: true,
                abandon_unlaunched: true,
                ..queued.clone()
            })
            .await
            .expect("Unlaunched plan cancellation preserves the active run");
        assert!(
            client.request(queued).await.is_err(),
            "Cancelled grant never replays"
        );
        assert!(
            !handle.state().is_terminal(),
            "Unrelated active handle was not cancelled"
        );
        fs.insert_file(path!("/jvm/target/surefire-reports/TEST-GreetingTest.xml"), format!("<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"{}\"/></testcase></testsuite>", "x".repeat(1024)).into_bytes()).await;
        cx.update(|cx| {
            assert!(handle.mark_running(None, cx));
            assert!(handle.mark_completed(std::process::ExitStatus::default(), cx));
        });
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.read_with(server_cx, |owner, cx| {
            let run = owner
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|provider| provider.current_run.as_ref())
                .expect("Selected host outcome");
            assert_eq!(run.id, plan.run_id);
            assert_eq!(run.phase, StructuredRunPhase::Completed);
            assert_eq!(run.summary.passed, 0);
            assert_eq!(run.summary.failed, 0);
            assert!(
                run.diagnostic
                    .as_deref()
                    .is_some_and(|detail| detail.contains("without a fresh readable JUnit result"))
            );
        });
        assert!(
            client.request(request).await.is_err(),
            "Oversized next baseline also blocks planning"
        );
    }

    #[gpui::test]
    async fn remote_java_action_requires_a_fresh_host_report(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/jvm"),
            json!({
                "pom.xml": "<project/>",
                "mvnw": "#!/bin/sh\n",
                "src": { "test": { "java": {
                    "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
                } } },
                "target": { "surefire-reports": {
                    "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
                } }
            }),
        )
        .await;
        server_cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let java = Language::new(
            LanguageConfig {
                name: "Java".into(),
                matcher: LanguageMatcher {
                    path_suffixes: vec!["java".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        )
        .with_queries(LanguageQueries {
            runnables: Some(Cow::Borrowed(
                "(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag \"java-test-method\"))",
            )),
            ..Default::default()
        })
        .expect("fixture Java runnable query");
        headless.read_with(server_cx, |headless, _| {
            headless.languages.add(Arc::new(java));
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("remote JVM worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let provider = project.read_with(cx, |project, _| {
            project
                .jvm_test_provider_store()
                .expect("remote JVM provider")
                .clone()
        });
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("host Java discovery");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .unwrap_or_else(|| panic!("source-backed host case: {snapshot:#?}"));
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let action_request = proto::ResolveJvmTestAction {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: snapshot.discovery_generation.0,
            node_id: case.id.0.clone(),
            failed_run_id: None,
            report_maximum_bytes: None,
            requested_run_id: None,
            worktree_ids: vec![worktree_id],
        };
        let restricted_error = client
            .request(action_request.clone())
            .await
            .expect_err("untrusted host must reject a direct Java action request");
        assert!(
            restricted_error
                .to_string()
                .contains("Trust the project host")
        );
        assert!(!restricted_error.to_string().contains("/jvm"));
        let trust_request = proto::TrustWorktrees {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            trusted_paths: vec![proto::PathTrust {
                content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
            }],
        };
        client
            .request(trust_request.clone())
            .await
            .expect("trust host Java worktree");
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("fresh Java discovery after host trust");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("trusted source-backed Java case");
        let action_request = proto::ResolveJvmTestAction {
            discovery_generation: snapshot.discovery_generation.0,
            node_id: case.id.0.clone(),
            ..action_request
        };
        let trust_loss_grant = client
            .request(action_request.clone())
            .await
            .expect("host grants a Java action before trust state disappears");
        let host_trust = server_cx
            .update(|cx| cx.remove_global::<project::trusted_worktrees::TrustedWorktrees>());
        let missing_trust_error = client
            .request(action_request.clone())
            .await
            .expect_err("missing host trust must deny a new Java action grant");
        assert!(
            missing_trust_error
                .to_string()
                .contains("trust is unavailable")
        );
        assert!(!missing_trust_error.to_string().contains("/jvm"));
        let trust_loss_lifecycle = proto::UpdateJvmTestRun {
            abandon_unlaunched: false,
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: trust_loss_grant.discovery_generation,
            run_id: trust_loss_grant.run_id,
            node_id: trust_loss_grant.node_id,
            task_id: "trust-loss-java-task".into(),
            state: proto::JvmTestTaskState::Queued as i32,
            terminal_id: None,
            exit_code: None,
            success: false,
            termination_confirmed: false,
            diagnostic: None,
            worktree_ids: vec![worktree_id],
        };
        let missing_trust_lifecycle_error = client
            .request(trust_loss_lifecycle.clone())
            .await
            .expect_err("missing host trust must expire an issued Java action grant");
        assert!(
            missing_trust_lifecycle_error
                .to_string()
                .contains("trust is unavailable")
        );
        headless.read_with(server_cx, |host, cx| {
            assert!(
                host.structured_execution_store
                    .read(cx)
                    .state()
                    .provider(&StructuredProviderId("jvm-junit".into()))
                    .expect("host Java provider remains visible")
                    .current_run
                    .is_none(),
                "missing trust must not create a queued structured run"
            );
        });
        server_cx.update(|cx| cx.set_global(host_trust));
        let expired_trust_loss_grant = client
            .request(trust_loss_lifecycle)
            .await
            .expect_err("restoring host trust must not restore an expired Java action grant");
        assert!(
            expired_trust_loss_grant
                .to_string()
                .contains("expired remote Java test run")
        );
        let host_provider =
            headless.read_with(server_cx, |host, _| host.jvm_test_provider_store.clone());
        let (capture_finished, release_action) = host_provider.update(server_cx, |provider, _| {
            provider.gate_next_action_reply_for_test()
        });
        let held_action = cx.background_spawn({
            let client = client.clone();
            let action_request = action_request.clone();
            async move { client.request(action_request).await }
        });
        capture_finished
            .await
            .expect("host captured the previous report");
        client
            .request(proto::RestrictWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("restrict the host while its Java action reply is held");
        release_action
            .send(())
            .expect("release the held host action");
        let held_error = held_action
            .await
            .expect_err("restriction after report capture must prevent a run grant");
        assert!(held_error.to_string().contains("stale discovery"));
        client
            .request(trust_request.clone())
            .await
            .expect("restore trust after the held Java action is rejected");
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("fresh Java discovery after the held action");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("current Java case after the held action");
        let action_request = proto::ResolveJvmTestAction {
            discovery_generation: snapshot.discovery_generation.0,
            node_id: case.id.0.clone(),
            ..action_request
        };
        let plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("host-authorized Java test plan");
        assert_eq!(plan.template.command, "./mvnw");
        assert!(
            client
                .request(proto::ResolveJvmTestAction {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version: u32::MAX,
                    discovery_generation: snapshot.discovery_generation.0,
                    node_id: case.id.0.clone(),
                    failed_run_id: None,
                    report_maximum_bytes: None,
                    requested_run_id: None,
                    worktree_ids: vec![worktree_id],
                })
                .await
                .is_err()
        );
        client
            .request(proto::RestrictWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("restrict host after Java discovery");
        let revoked_error = client
            .request(action_request)
            .await
            .expect_err("live host trust must override cached Java discovery");
        assert!(revoked_error.to_string().contains("stale discovery"));
        let expired_grant = client
            .request(proto::UpdateJvmTestRun {
                abandon_unlaunched: false,
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
                discovery_generation: plan.discovery_generation.0,
                run_id: plan.run_id.0.clone(),
                node_id: plan.node_id.0.clone(),
                task_id: "restricted-java-task".into(),
                state: proto::JvmTestTaskState::Queued as i32,
                terminal_id: None,
                exit_code: None,
                success: false,
                termination_confirmed: false,
                diagnostic: None,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect_err("restriction must expire an already-issued Java run grant");
        assert!(
            expired_grant
                .to_string()
                .contains("expired remote Java test run")
        );
        client
            .request(trust_request)
            .await
            .expect("restore host Java trust");
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("refresh Java discovery after restoring host trust");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("restored source-backed Java case");
        let plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("new Java grant after restoring trust and refreshing");
        let handle = StructuredTaskHandle::new(TaskId("remote-java-fixture".into()));
        provider
            .update(cx, |provider, cx| {
                provider.register_remote_run_handle(plan.clone(), handle.clone(), cx)
            })
            .expect("registered remote Java Task handle");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let host_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        assert_eq!(
            host_run.expect("host queued run").phase,
            StructuredRunPhase::Queued
        );
        let client_run = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        assert_eq!(
            client_run.expect("client receives the queued run").phase,
            StructuredRunPhase::Queued
        );
        assert!(
            provider
                .update(cx, |provider, cx| provider.refresh(cx))
                .await
                .is_err(),
            "refresh must not cancel a selected remote Java Task"
        );
        assert!(!handle.state().is_terminal());
        cx.update(|cx| {
            assert!(handle.mark_spawn_error(
                "PRIVATE_SPAWN_DETAIL /outside/private/runtime token=secret",
                cx
            ));
        });
        cx.run_until_parked();
        server_cx.run_until_parked();
        let host_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        let host_run = host_run.expect("host terminal run");
        assert_eq!(host_run.phase, StructuredRunPhase::SpawnError);
        assert_eq!(
            host_run.diagnostic.as_deref(),
            Some(
                "The Java test Task failed to start; inspect the Task terminal and project-host logs"
            )
        );
        assert!(!format!("{host_run:?}").contains("PRIVATE_SPAWN_DETAIL"));
        assert!(
            matches!(handle.state(), task::StructuredTaskState::SpawnError { message } if message.contains("PRIVATE_SPAWN_DETAIL"))
        );
        cx.run_until_parked();
        let client_run = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        assert_eq!(
            client_run
                .expect("client receives the host terminal result")
                .phase,
            StructuredRunPhase::SpawnError
        );

        let stale_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("second host-authorized Java test plan");
        let lifecycle =
            |run_id: &str, task_id: &str, state: proto::JvmTestTaskState| proto::UpdateJvmTestRun {
                abandon_unlaunched: false,
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
                discovery_generation: snapshot.discovery_generation.0,
                run_id: run_id.to_string(),
                node_id: case.id.0.clone(),
                task_id: task_id.to_string(),
                state: state as i32,
                terminal_id: None,
                exit_code: Some(0),
                success: true,
                termination_confirmed: false,
                diagnostic: None,
                worktree_ids: vec![worktree_id],
            };
        assert!(
            client
                .request(lifecycle(
                    &stale_plan.run_id.0,
                    "stale-report-task",
                    proto::JvmTestTaskState::Completed,
                ))
                .await
                .is_err(),
            "a terminal event must not create a run before its queued event"
        );
        client
            .request(lifecycle(
                &stale_plan.run_id.0,
                "stale-report-task",
                proto::JvmTestTaskState::Queued,
            ))
            .await
            .expect("host accepts queued Java run");
        assert!(
            client
                .request(lifecycle(
                    &stale_plan.run_id.0,
                    "different-task",
                    proto::JvmTestTaskState::Completed,
                ))
                .await
                .is_err(),
            "a terminal event cannot switch the authorized Task identity"
        );
        client
            .request(lifecycle(
                &stale_plan.run_id.0,
                "stale-report-task",
                proto::JvmTestTaskState::Completed,
            ))
            .await
            .expect("host accepts terminal Task event");
        server_cx.run_until_parked();
        let stale_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        let stale_run = stale_run.expect("completed Java run without a fresh report");
        assert_eq!(stale_run.phase, StructuredRunPhase::Completed);
        assert_eq!(stale_run.summary.passed, 0);
        assert!(
            stale_run
                .diagnostic
                .as_deref()
                .is_some_and(|diagnostic| diagnostic.contains("fresh readable JUnit result"))
        );

        let cancellation_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("host authorizes a cancellable Java test");
        client
            .request(lifecycle(
                &cancellation_plan.run_id.0,
                "cancel-task",
                proto::JvmTestTaskState::Queued,
            ))
            .await
            .expect("host accepts queued cancellation run");
        let cancelled = lifecycle(
            &cancellation_plan.run_id.0,
            "cancel-task",
            proto::JvmTestTaskState::Cancelled,
        );
        client
            .request(cancelled.clone())
            .await
            .expect("host accepts cancellation request");
        server_cx.run_until_parked();
        let unconfirmed = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        assert_eq!(
            unconfirmed
                .expect("cancelled run before host exit")
                .diagnostic
                .as_deref(),
            Some("Cancellation requested; host termination was not confirmed")
        );
        let mut confirmed = cancelled;
        confirmed.termination_confirmed = true;
        client
            .request(confirmed.clone())
            .await
            .expect("host accepts matching termination confirmation");
        server_cx.run_until_parked();
        let host_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        let host_run = host_run.expect("confirmed cancellation remains visible");
        assert_eq!(host_run.phase, StructuredRunPhase::Cancelled);
        assert!(host_run.diagnostic.is_none());
        assert!(
            client.request(confirmed).await.is_err(),
            "host must expire the cancellation authorization after confirmation"
        );

        let fresh_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("third host-authorized Java test plan");
        client
            .request(lifecycle(
                &fresh_plan.run_id.0,
                "fresh-report-task",
                proto::JvmTestTaskState::Queued,
            ))
            .await
            .expect("host accepts fresh queued Java run");
        fs.insert_file(
            path!("/jvm/target/surefire-reports/TEST-GreetingTest.xml"),
            b"<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"expected greeting\"/></testcase></testsuite>".to_vec(),
        )
        .await;
        client
            .request(lifecycle(
                &fresh_plan.run_id.0,
                "fresh-report-task",
                proto::JvmTestTaskState::Completed,
            ))
            .await
            .expect("host accepts fresh terminal Task event");
        server_cx.run_until_parked();
        let fresh_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        let fresh_run = fresh_run.expect("completed Java run with a fresh report");
        assert_eq!(fresh_run.phase, StructuredRunPhase::Completed);
        assert_eq!(fresh_run.summary.failed, 1);

        let legacy_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("host authorizes a legacy-peer Java test plan");
        let mut legacy_event = lifecycle(
            &legacy_plan.run_id.0,
            "legacy-spawn-error-task",
            proto::JvmTestTaskState::Queued,
        );
        legacy_event.exit_code = None;
        legacy_event.success = false;
        client
            .request(legacy_event.clone())
            .await
            .expect("host accepts the legacy-peer queued identity");
        legacy_event.state = proto::JvmTestTaskState::SpawnError as i32;
        legacy_event.diagnostic =
            Some("PRIVATE_SPAWN_DETAIL /outside/private/runtime token=secret".into());
        client
            .request(legacy_event)
            .await
            .expect("host accepts and reduces the legacy-peer spawn failure");
        server_cx.run_until_parked();
        let legacy_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        let legacy_run = legacy_run.expect("host keeps the reduced legacy-peer failure");
        assert_eq!(legacy_run.phase, StructuredRunPhase::SpawnError);
        assert_eq!(legacy_run.summary.passed, 0);
        assert_eq!(legacy_run.summary.failed, 1);
        assert_eq!(
            legacy_run.diagnostic.as_deref(),
            Some(
                "The Java test Task failed to start; inspect the Task terminal and project-host logs"
            )
        );
        assert!(!format!("{legacy_run:?}").contains("PRIVATE_SPAWN_DETAIL"));

        let immediate_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("host authorizes an immediate-failure Java test plan");
        let immediate_handle = StructuredTaskHandle::new(TaskId("immediate-spawn-error".into()));
        cx.update(|cx| {
            assert!(immediate_handle.mark_spawn_error("wrapper could not start", cx));
        });
        provider
            .update(cx, |provider, cx| {
                provider.register_remote_run_handle(immediate_plan, immediate_handle, cx)
            })
            .expect("terminal handle gets a queued host identity first");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let immediate_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        assert_eq!(
            immediate_run.expect("immediate terminal host run").phase,
            StructuredRunPhase::SpawnError
        );

        let unstarted_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
            })
            .await
            .expect("host authorizes an action that cannot be prepared");
        provider
            .read_with(cx, |provider, cx| {
                provider.fail_remote_action_plan(&unstarted_plan, cx)
            })
            .expect("unstarted action reports a bounded Task failure");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let unstarted_run = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .cloned()
        });
        assert_eq!(
            unstarted_run.expect("unstarted terminal host run").phase,
            StructuredRunPhase::SpawnError
        );

        let source_path = project.read_with(cx, |project, cx| {
            project
                .find_project_path(path!("/jvm/src/test/java/GreetingTest.java"), cx)
                .expect("visible remote Java test source")
        });
        let source_buffer = project
            .update(cx, |project, cx| project.open_buffer(source_path, cx))
            .await
            .expect("remote Java test buffer");
        source_buffer.update(cx, |buffer, cx| {
            buffer.edit([(0..0, "// edited\n")], None, cx);
        });
        cx.run_until_parked();
        server_cx.run_until_parked();
        let edited = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("remote discovery with unsaved Java source");
        let host_diagnostic = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&StructuredProviderId("jvm-junit".into()))
                .and_then(|snapshot| snapshot.diagnostic.clone())
        });
        assert!(host_diagnostic.as_deref().is_some_and(|diagnostic| {
            diagnostic.contains("display-only") && !diagnostic.contains("/jvm")
        }));
        assert_eq!(
            edited.diagnostic.as_deref(),
            Some("Structured provider details remain on the project host.")
        );
        let edited_case = edited
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("unsaved Java case remains visible");
        let edited_error = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&edited_case.id, edited.discovery_generation, None, cx)
            })
            .await
            .expect_err("unsaved source must not receive a remote action plan");
        let visible_error = edited_error.to_string();
        assert!(visible_error.contains("Save its source and refresh Tests"));
        assert!(visible_error.contains("host worktree is trusted"));
        assert!(visible_error.contains("reconnect to sync the installed Java extension"));
        assert!(!visible_error.contains("/jvm"));
        project
            .update(cx, |project, cx| project.save_buffer(source_buffer, cx))
            .await
            .expect("save remote Java test source");
        let saved = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("remote discovery after saving Java source");
        let saved_case = saved
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("saved Java case remains visible");
        provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(&saved_case.id, saved.discovery_generation, None, cx)
            })
            .await
            .expect("host reauthorizes saved Java test");

        let original_id = worktree.read_with(cx, |worktree, _| worktree.id());
        project.update(cx, |project, cx| project.remove_worktree(original_id, cx));
        drop(worktree);
        cx.run_until_parked();
        server_cx.run_until_parked();
        let (replacement, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/jvm"), true, cx)
            })
            .await
            .expect("same-path Java worktree replacement");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let replacement_id = replacement.read_with(cx, |worktree, _| worktree.id());
        assert_ne!(replacement_id, original_id);
        let host_worktree_ids = headless.read_with(server_cx, |headless, cx| {
            headless
                .worktree_store
                .read(cx)
                .visible_worktrees(cx)
                .map(|worktree| worktree.read(cx).id())
                .collect::<Vec<_>>()
        });
        assert_eq!(host_worktree_ids, vec![replacement_id]);
        assert!(
            provider
                .update(cx, |provider, cx| {
                    provider.plan_remote_action(&case.id, snapshot.discovery_generation, None, cx)
                })
                .await
                .is_err(),
            "a selected case from the replaced worktree must not run"
        );
        let refreshed = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("fresh Java discovery after worktree replacement");
        assert!(
            refreshed.nodes.iter().all(|node| node.id != case.id
                && node
                    .path
                    .as_ref()
                    .is_none_or(|path| path.worktree_id == replacement_id)),
            "the refreshed provider must not expose a case or path from the old worktree"
        );
        let replacement_case = refreshed
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case && node.label == "messageIsAvailable"
            })
            .expect("case on the replacement worktree");
        assert_ne!(replacement_case.id, case.id);
        let replacement_error = client
            .request(proto::ResolveJvmTestAction {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: project::jvm_test_provider::JVM_TEST_DISCOVERY_PROTOCOL_VERSION,
                discovery_generation: refreshed.discovery_generation.0,
                node_id: replacement_case.id.0.clone(),
                failed_run_id: None,
                report_maximum_bytes: None,
                requested_run_id: None,
                worktree_ids: vec![replacement_id.to_proto()],
            })
            .await
            .expect_err("same-path replacement must not inherit worktree trust");
        assert!(
            replacement_error
                .to_string()
                .contains("Trust the project host")
        );
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(
                        replacement_id.to_proto(),
                    )),
                }],
            })
            .await
            .expect("explicitly trust the replacement Java worktree");
        let refreshed = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("refresh the newly trusted replacement Java worktree");
        let replacement_case = refreshed
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case)
            .expect("trusted replacement Java case");
        let replacement_plan = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(
                    &replacement_case.id,
                    refreshed.discovery_generation,
                    None,
                    cx,
                )
            })
            .await
            .expect("replacement worktree action is current");
        let running_handle = StructuredTaskHandle::new(TaskId("disconnect-task".into()));
        provider
            .update(cx, |provider, cx| {
                provider.register_remote_run_handle(
                    replacement_plan.clone(),
                    running_handle.clone(),
                    cx,
                )
            })
            .expect("selected remote Java run before disconnect");
        cx.update(|cx| assert!(running_handle.mark_running(None, cx)));
        cx.run_until_parked();
        server_cx.run_until_parked();
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(1, cx));
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            futures::pin_mut!(recovery);
            assert!(matches!(
                futures::future::select(recovery, deadline).await,
                futures::future::Either::Left(_)
            ));
        }
        assert!(matches!(
            running_handle.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));
        cx.update(|cx| {
            assert!(!running_handle.mark_spawn_error("late disconnected result", cx));
        });
        let stale_error = provider
            .update(cx, |provider, cx| {
                provider.plan_remote_action(
                    &replacement_case.id,
                    refreshed.discovery_generation,
                    None,
                    cx,
                )
            })
            .await
            .expect_err("the pre-disconnect selection must not authorize another run");
        assert!(stale_error.to_string().contains("refresh Tests"));
        let recovered = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("remote Java discovery after selected-run disconnect");
        assert!(recovered.discovery_generation > refreshed.discovery_generation);
        assert!(
            recovered
                .current_run
                .as_ref()
                .is_none_or(|run| run.id != replacement_plan.run_id),
            "the abandoned run must not remain actionable after reconnect"
        );
    }
}

#[gpui::test]
async fn test_remote_telemetry_event_forwarding(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    // This mirrors `init_test`, but retains the server-side session so the test
    // can drive a forwarded telemetry event over the proto channel (as
    // `init_telemetry_forwarding` does on a real remote server).
    let server_fs = FakeFs::new(server_cx.executor());
    server_fs
        .insert_tree(
            path!("/code"),
            json!({ "project1": { "README.md": "# project 1" } }),
        )
        .await;

    cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
    server_cx.update(|cx| release_channel::init(semver::Version::new(0, 0, 0), cx));
    init_logger();

    let (opts, server_session, _) = RemoteClient::fake_server(cx, server_cx);
    let http_client = Arc::new(BlockedHttpClient);
    let node_runtime = NodeRuntime::unavailable();
    let languages = Arc::new(LanguageRegistry::new(cx.executor()));
    let proxy = Arc::new(ExtensionHostProxy::new());
    server_cx.update(HeadlessProject::init);
    let headless = server_cx.new(|cx| {
        HeadlessProject::new(
            crate::HeadlessAppState {
                session: server_session.clone(),
                fs: server_fs.clone(),
                http_client,
                node_runtime,
                languages,
                extension_host_proxy: proxy,
                startup_time: std::time::Instant::now(),
            },
            false,
            cx,
        )
    });

    let ssh = RemoteClient::connect_mock(opts, cx).await;
    let project = build_project(ssh, cx);
    project
        .update(cx, {
            let headless = headless.clone();
            |_, cx| cx.on_release(|_, _| drop(headless))
        })
        .detach();

    // The remote server forwards a bare `FlexibleEvent` as JSON; mirror that
    // here by sending the proto message the forwarding task would send.
    let event_json = json!({
        "event_type": "fs_watcher_poll",
        "event_properties": { "path": "/code/project1" },
    })
    .to_string();
    server_session
        .send(proto::TelemetryEvent {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            event_json,
        })
        .unwrap();
    cx.executor().run_until_parked();

    let events = project.read_with(cx, |project, _| {
        project.client().telemetry().queued_events()
    });
    assert_eq!(
        events.len(),
        1,
        "the forwarded event should be reported once"
    );
    let event = &events[0];
    assert_eq!(event.event_type, "fs_watcher_poll");
    // The event's original properties survive the round-trip.
    assert_eq!(
        event.event_properties.get("path"),
        Some(&serde_json::Value::String("/code/project1".to_string()))
    );
    // The client stamps the remote host metadata it learned at connection time.
    // The mock connection reports a Linux/x86_64 host over a "mock" connection.
    assert_eq!(
        event.event_properties.get("remote"),
        Some(&serde_json::Value::Bool(true))
    );
    assert_eq!(
        event.event_properties.get("remote_connection_type"),
        Some(&serde_json::Value::String("mock".to_string()))
    );
    assert_eq!(
        event.event_properties.get("remote_os_name"),
        Some(&serde_json::Value::String("Linux".to_string()))
    );
    assert_eq!(
        event.event_properties.get("remote_architecture"),
        Some(&serde_json::Value::String("x86_64".to_string()))
    );
}

async fn do_search_and_assert(
    project: &Entity<Project>,
    query: &str,
    files_to_include: PathMatcher,
    match_full_paths: bool,
    expected_paths: &[&str],
    mut cx: TestAppContext,
) -> Vec<Entity<Buffer>> {
    let query = query.to_string();
    let receiver = project.update(&mut cx, |project, cx| {
        project.search(
            SearchQuery::text(
                query,
                false,
                true,
                false,
                files_to_include,
                Default::default(),
                match_full_paths,
                None,
            )
            .unwrap(),
            cx,
        )
    });

    let mut buffers = Vec::new();
    for expected_path in expected_paths {
        let buffer = loop {
            let response = receiver.rx.recv().await.unwrap();
            match response {
                SearchResult::Buffer { buffer, .. } => break buffer,
                SearchResult::LimitReached => panic!("incorrect result"),
                SearchResult::WaitingForScan | SearchResult::Searching => continue,
            }
        };
        buffer.update(&mut cx, |buffer, cx| {
            assert_eq!(
                buffer.file().unwrap().full_path(cx).to_string_lossy(),
                *expected_path
            )
        });
        buffers.push(buffer);
    }

    assert!(receiver.rx.recv().await.is_err());
    buffers
}

struct RemoteImageTestView {
    source: ImageSource,
}

impl gpui::Render for RemoteImageTestView {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        _cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        img(self.source.clone())
    }
}

#[gpui::test]
async fn test_remote_project_image_source(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project": {
                "docs": {
                    "image.ppm": "P3\n1 1\n255\n255 0 0\n"
                }
            }
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project"), true, cx)
        })
        .await
        .expect("remote worktree should open");
    let worktree_id = worktree.read_with(cx, |worktree, _cx| worktree.id());
    let source = image_store::project_image_source(
        project.downgrade(),
        ProjectPath {
            worktree_id,
            path: rel_path("docs/image.ppm").into(),
        },
    );
    let ImageSource::Custom(load_image) = source else {
        panic!("expected a project-backed image source");
    };
    let loaded_bytes = Arc::new(std::sync::Mutex::new(None));
    let observed_source = ImageSource::from({
        let loaded_bytes = loaded_bytes.clone();
        move |window: &mut gpui::Window, cx: &mut gpui::App| {
            let result = load_image(window, cx);
            if let Some(Ok(image)) = &result {
                *loaded_bytes.lock().expect("loaded image mutex poisoned") =
                    image.as_bytes(0).map(ToOwned::to_owned);
            }
            result
        }
    });

    let (view, cx) = cx.add_window_view(|_window, _cx| RemoteImageTestView {
        source: observed_source,
    });
    cx.draw(Default::default(), size(px(10.), px(10.)), {
        let view = view.clone();
        move |_, _| view.into_any_element()
    });
    cx.run_until_parked();
    cx.draw(Default::default(), size(px(10.), px(10.)), move |_, _| {
        view.into_any_element()
    });

    assert_eq!(
        loaded_bytes
            .lock()
            .expect("loaded image mutex poisoned")
            .as_deref(),
        Some([0, 0, 255, 255].as_slice())
    );
}

#[gpui::test]
async fn test_remote_project_search(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    let buffers = do_search_and_assert(
        &project,
        "project",
        Default::default(),
        false,
        &[path!("project1/README.md")],
        cx.clone(),
    )
    .await;
    let buffer = buffers.into_iter().next().unwrap();

    // test that the headless server is tracking which buffers we have open correctly.
    cx.run_until_parked();
    headless.update(server_cx, |headless, cx| {
        assert!(headless.buffer_store.read(cx).has_shared_buffers())
    });
    do_search_and_assert(
        &project,
        "project",
        Default::default(),
        false,
        &[path!("project1/README.md")],
        cx.clone(),
    )
    .await;
    server_cx.run_until_parked();
    cx.update(|_| {
        drop(buffer);
    });
    cx.run_until_parked();
    server_cx.run_until_parked();
    headless.update(server_cx, |headless, cx| {
        assert!(!headless.buffer_store.read(cx).has_shared_buffers())
    });

    do_search_and_assert(
        &project,
        "project",
        Default::default(),
        false,
        &[path!("project1/README.md")],
        cx.clone(),
    )
    .await;
}

#[gpui::test]
async fn test_remote_project_search_single_cpu(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    // Simulate a single-CPU environment (e.g. a devcontainer with 1 visible CPU).
    // This causes the worker pool in project search to spawn num_cpus - 1 = 0 workers,
    // which silently drops all search channels and produces zero results.
    server_cx.executor().set_num_cpus(1);

    let (project, _) = init_test(&fs, cx, server_cx).await;

    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    do_search_and_assert(
        &project,
        "project",
        Default::default(),
        false,
        &[path!("project1/README.md")],
        cx.clone(),
    )
    .await;
}

#[gpui::test]
async fn test_remote_project_search_inclusion(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                "README.md": "# project 1",
            },
            "project2": {
                "README.md": "# project 2",
            },
        }),
    )
    .await;

    let (project, _) = init_test(&fs, cx, server_cx).await;

    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project2"), true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    // Case 1: Test search with path matcher limiting to only one worktree
    let path_matcher = PathMatcher::new(
        &["project1/*.md".to_owned()],
        util::paths::PathStyle::local(),
    )
    .unwrap();
    do_search_and_assert(
        &project,
        "project",
        path_matcher,
        true, // should be true in case of multiple worktrees
        &[path!("project1/README.md")],
        cx.clone(),
    )
    .await;

    // Case 2: Test search without path matcher, matching both worktrees
    do_search_and_assert(
        &project,
        "project",
        Default::default(),
        true, // should be true in case of multiple worktrees
        &[path!("project1/README.md"), path!("project2/README.md")],
        cx.clone(),
    )
    .await;
}

#[gpui::test]
async fn test_remote_settings(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/code",
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    cx.update_global(|settings_store: &mut SettingsStore, cx| {
        settings_store.set_user_settings(
            r#"{"languages":{"Rust":{"language_servers":["from-local-settings"]}}}"#,
            cx,
        )
    })
    .unwrap();

    cx.run_until_parked();

    server_cx.read(|cx| {
        assert_eq!(
            AllLanguageSettings::get_global(cx)
                .language(None, Some(&"Rust".into()), cx)
                .language_servers,
            [ConfiguredLanguageServer::new("from-local-settings")],
            "User language settings should be synchronized with the server settings"
        )
    });

    server_cx
        .update_global(|settings_store: &mut SettingsStore, cx| {
            settings_store.set_server_settings(
                r#"{"languages":{"Rust":{"language_servers":["from-server-settings"]}}}"#,
                cx,
            )
        })
        .unwrap();

    cx.run_until_parked();

    server_cx.read(|cx| {
        assert_eq!(
            AllLanguageSettings::get_global(cx)
                .language(None, Some(&"Rust".into()), cx)
                .language_servers,
            [ConfiguredLanguageServer::new("from-server-settings")],
            "Server language settings should take precedence over the user settings"
        )
    });

    fs.insert_tree(
        "/code/project1/.zed",
        json!({
            "settings.json": r#"
                  {
                    "languages": {"Rust":{"language_servers":["override-rust-analyzer"]}},
                    "lsp": {
                      "override-rust-analyzer": {
                        "binary": {
                          "path": "~/.cargo/bin/rust-analyzer"
                        }
                      }
                    }
                  }"#
        }),
    )
    .await;

    let worktree_id = project
        .update(cx, |project, cx| {
            project.languages().add(rust_lang());
            project.find_or_create_worktree("/code/project1", true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());

    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    server_cx.read(|cx| {
        let worktree_id = headless
            .read(cx)
            .worktree_store
            .read(cx)
            .worktrees()
            .next()
            .unwrap()
            .read(cx)
            .id();
        assert_eq!(
            AllLanguageSettings::get(
                Some(SettingsLocation {
                    worktree_id,
                    path: rel_path("src/lib.rs")
                }),
                cx
            )
            .language(None, Some(&"Rust".into()), cx)
            .language_servers,
            [ConfiguredLanguageServer::new("override-rust-analyzer")]
        )
    });

    cx.read(|cx| {
        assert_eq!(
            LanguageSettings::for_buffer(buffer.read(cx), cx).language_servers,
            [ConfiguredLanguageServer::new("override-rust-analyzer")]
        )
    });
}

#[gpui::test]
async fn test_remote_lsp(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    fs.insert_tree(
        path!("/code/project1/.zed"),
        json!({
            "settings.json": r#"
          {
            "languages": {"Rust":{"language_servers":["rust-analyzer", "fake-analyzer"]}},
            "lsp": {
              "rust-analyzer": {
                "binary": {
                  "path": "~/.cargo/bin/rust-analyzer"
                }
              },
              "fake-analyzer": {
               "binary": {
                "path": "~/.cargo/bin/rust-analyzer"
               }
              }
            }
          }"#
        }),
    )
    .await;

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                capabilities: lsp::ServerCapabilities {
                    completion_provider: Some(lsp::CompletionOptions::default()),
                    rename_provider: Some(lsp::OneOf::Left(true)),
                    ..lsp::ServerCapabilities::default()
                },
                ..FakeLspAdapter::default()
            },
        );
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "fake-analyzer",
                capabilities: lsp::ServerCapabilities {
                    completion_provider: Some(lsp::CompletionOptions::default()),
                    rename_provider: Some(lsp::OneOf::Left(true)),
                    ..lsp::ServerCapabilities::default()
                },
                ..FakeLspAdapter::default()
            },
        )
    });

    let mut fake_lsp = server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            lsp::ServerCapabilities {
                completion_provider: Some(lsp::CompletionOptions::default()),
                rename_provider: Some(lsp::OneOf::Left(true)),
                ..lsp::ServerCapabilities::default()
            },
            None,
        )
    });

    let mut fake_second_lsp = server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "fake-analyzer",
                capabilities: lsp::ServerCapabilities {
                    completion_provider: Some(lsp::CompletionOptions::default()),
                    rename_provider: Some(lsp::OneOf::Left(true)),
                    ..lsp::ServerCapabilities::default()
                },
                ..FakeLspAdapter::default()
            },
        );
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("fake-analyzer".into()),
            lsp::ServerCapabilities {
                completion_provider: Some(lsp::CompletionOptions::default()),
                rename_provider: Some(lsp::OneOf::Left(true)),
                ..lsp::ServerCapabilities::default()
            },
            None,
        )
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.languages().add(rust_lang());
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());

    // Wait for the settings to synchronize
    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let fake_lsp = fake_lsp.next().await.unwrap();
    let fake_second_lsp = fake_second_lsp.next().await.unwrap();

    cx.read(|cx| {
        assert_eq!(
            LanguageSettings::for_buffer(buffer.read(cx), cx).language_servers,
            [
                ConfiguredLanguageServer::new("rust-analyzer"),
                ConfiguredLanguageServer::new("fake-analyzer"),
            ]
        )
    });

    let buffer_id = cx.read(|cx| {
        let buffer = buffer.read(cx);
        assert_eq!(buffer.language().unwrap().name(), "Rust");
        buffer.remote_id()
    });

    server_cx.read(|cx| {
        let buffer = headless
            .read(cx)
            .buffer_store
            .read(cx)
            .get(buffer_id)
            .unwrap();

        assert_eq!(buffer.read(cx).language().unwrap().name(), "Rust");
    });

    server_cx.read(|cx| {
        let lsp_store = headless.read(cx).lsp_store.read(cx);
        assert_eq!(lsp_store.as_local().unwrap().language_servers.len(), 2);
    });

    fake_lsp.set_request_handler::<lsp::request::Completion, _, _>(|_, _| async move {
        Ok(Some(CompletionResponse::Array(vec![lsp::CompletionItem {
            label: "boop".to_string(),
            ..Default::default()
        }])))
    });

    fake_second_lsp.set_request_handler::<lsp::request::Completion, _, _>(|_, _| async move {
        Ok(Some(CompletionResponse::Array(vec![lsp::CompletionItem {
            label: "beep".to_string(),
            ..Default::default()
        }])))
    });

    let result = project
        .update(cx, |project, cx| {
            project.completions(
                &buffer,
                0,
                CompletionContext {
                    trigger_kind: CompletionTriggerKind::INVOKED,
                    trigger_character: None,
                },
                cx,
            )
        })
        .await
        .unwrap();

    assert_eq!(
        result
            .into_iter()
            .flat_map(|response| response.completions)
            .map(|c| c.label.text)
            .collect::<Vec<_>>(),
        vec!["boop".to_string(), "beep".to_string()]
    );

    fake_lsp.set_request_handler::<lsp::request::Rename, _, _>(|_, _| async move {
        Ok(Some(lsp::WorkspaceEdit {
            changes: Some(
                [(
                    lsp::Uri::from_file_path(path!("/code/project1/src/lib.rs")).unwrap(),
                    vec![lsp::TextEdit::new(
                        lsp::Range::new(lsp::Position::new(0, 3), lsp::Position::new(0, 6)),
                        "two".to_string(),
                    )],
                )]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        }))
    });

    project
        .update(cx, |project, cx| {
            project.perform_rename(buffer.clone(), 3, "two".to_string(), cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();
    buffer.update(cx, |buffer, _| {
        assert_eq!(buffer.text(), "fn two() -> usize { 1 }")
    })
}

#[gpui::test]
async fn test_remote_code_action_resolve(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
        editor::init(cx);
    });

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    let capabilities = lsp::ServerCapabilities {
        code_action_provider: Some(lsp::CodeActionProviderCapability::Options(
            lsp::CodeActionOptions {
                resolve_provider: Some(true),
                ..lsp::CodeActionOptions::default()
            },
        )),
        ..lsp::ServerCapabilities::default()
    };

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                capabilities: capabilities.clone(),
                ..FakeLspAdapter::default()
            },
        );
    });

    server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            capabilities,
            Some(Box::new(|fake_lsp| {
                fake_lsp.set_request_handler::<lsp::request::CodeActionRequest, _, _>(
                    |_, _| async move {
                        Ok(Some(vec![lsp::CodeActionOrCommand::CodeAction(
                            lsp::CodeAction {
                                title: "Use two".to_string(),
                                data: Some(serde_json::json!({ "id": "action" })),
                                ..lsp::CodeAction::default()
                            },
                        )]))
                    },
                );
                fake_lsp.set_request_handler::<lsp::request::CodeActionResolveRequest, _, _>(
                    |mut action, _| async move {
                        action.edit = Some(lsp::WorkspaceEdit {
                            changes: Some(
                                [(
                                    lsp::Uri::from_file_path(path!("/code/project1/src/lib.rs"))
                                        .unwrap(),
                                    vec![lsp::TextEdit::new(
                                        lsp::Range::new(
                                            lsp::Position::new(0, 3),
                                            lsp::Position::new(0, 6),
                                        ),
                                        "two".to_string(),
                                    )],
                                )]
                                .into_iter()
                                .collect(),
                            ),
                            ..lsp::WorkspaceEdit::default()
                        });
                        Ok(action)
                    },
                );
            })),
        );
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.languages().add(rust_lang());
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());
    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let cx = cx.add_empty_window();
    let workspace = cx.new_window_entity(|window, cx| {
        workspace::Workspace::test_new(project.clone(), window, cx)
    });
    let editor = cx.new_window_entity(|window, cx| {
        Editor::for_buffer(buffer.clone(), Some(project.clone()), window, cx)
    });
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_item_to_active_pane(Box::new(editor.clone()), None, true, window, cx);
    });
    cx.run_until_parked();

    editor.update_in(cx, |editor, window, cx| {
        editor.change_selections(SelectionEffects::default(), window, cx, |s| {
            s.select_ranges([Point::new(0, 0)..Point::new(0, 0)]);
        });
    });
    cx.executor()
        .advance_clock(editor::CODE_ACTIONS_DEBOUNCE_TIMEOUT * 2);
    cx.run_until_parked();

    editor.update_in(cx, |editor, window, cx| {
        editor.toggle_code_actions(
            &ToggleCodeActions {
                deployed_from: None,
                quick_launch: false,
            },
            window,
            cx,
        );
    });
    cx.run_until_parked();

    editor.update(cx, |editor, _| assert!(editor.context_menu_visible()));

    let confirm_action = editor
        .update_in(cx, |editor, window, cx| {
            Editor::confirm_code_action(editor, &ConfirmCodeAction { item_ix: Some(0) }, window, cx)
        })
        .unwrap();
    confirm_action.await.unwrap();

    buffer.read_with(cx, |buffer, _| {
        assert_eq!(buffer.text(), "fn two() -> usize { 1 }");
    });
}

#[gpui::test]
async fn test_remote_code_lens_fetch_after_lsp_starts(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    let capabilities = lsp::ServerCapabilities {
        code_lens_provider: Some(lsp::CodeLensOptions {
            resolve_provider: None,
        }),
        ..lsp::ServerCapabilities::default()
    };

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                capabilities: capabilities.clone(),
                ..FakeLspAdapter::default()
            },
        );
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.languages().add(rust_lang());
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());
    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    // Prime the code lens cache while no LSP server exists on the host.
    // This simulates the race where the editor fetches code lenses during
    // initial paint before the language server has started.
    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let initial_actions = lsp_store
        .update(cx, |lsp_store, cx| lsp_store.code_lens_actions(&buffer, cx))
        .await
        .unwrap();
    assert_eq!(
        initial_actions.map(|a| a.len()),
        Some(0),
        "Before any LSP starts, code lenses should be empty"
    );

    // Now register the LSP on the host. This triggers server startup and
    // buffer registration, propagating capabilities to the client.
    server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            capabilities,
            Some(Box::new(|fake_lsp| {
                fake_lsp.set_request_handler::<lsp::request::CodeLensRequest, _, _>(
                    |_, _| async move {
                        Ok(Some(vec![lsp::CodeLens {
                            range: lsp::Range::new(
                                lsp::Position::new(0, 0),
                                lsp::Position::new(0, 9),
                            ),
                            command: Some(lsp::Command {
                                title: "1 reference".to_string(),
                                command: "lens_cmd".to_string(),
                                arguments: None,
                            }),
                            data: None,
                        }]))
                    },
                );
            })),
        );
    });

    // Trigger re-evaluation of language servers for the already-open buffer.
    server_cx.update_entity(&headless, |headless, cx| {
        headless.lsp_store.update(cx, |lsp_store, cx| {
            lsp_store.restart_all_language_servers(cx);
        });
    });
    cx.run_until_parked();

    // A subsequent fetch must bypass the stale (empty) cache now that a
    // new server is available.
    let actions = lsp_store
        .update(cx, |lsp_store, cx| lsp_store.code_lens_actions(&buffer, cx))
        .await
        .unwrap();
    let actions = actions.expect("Should have code lens actions after LSP starts");
    assert_eq!(
        actions.len(),
        1,
        "Should have fetched one code lens from the newly started LSP"
    );
    assert_eq!(
        actions.values().next().unwrap().lsp_action.title(),
        "1 reference",
    );
}

#[gpui::test]
async fn test_remote_code_lens_refetch_after_refresh(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    let capabilities = lsp::ServerCapabilities {
        code_lens_provider: Some(lsp::CodeLensOptions {
            resolve_provider: None,
        }),
        ..lsp::ServerCapabilities::default()
    };

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: Arc::new(LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..LanguageMatcher::default()
            }),
            ..LanguageConfig::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                capabilities: capabilities.clone(),
                ..FakeLspAdapter::default()
            },
        );
    });

    let lens_generation = Arc::new(AtomicUsize::new(0));
    let mut fake_lsp_servers = server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            capabilities,
            Some(Box::new({
                let lens_generation = lens_generation.clone();
                move |fake_lsp| {
                    let lens_generation = lens_generation.clone();
                    fake_lsp.set_request_handler::<lsp::request::CodeLensRequest, _, _>(
                        move |_, _| {
                            let generation = lens_generation.load(Ordering::Acquire);
                            async move {
                                Ok(Some(vec![lsp::CodeLens {
                                    range: lsp::Range::new(
                                        lsp::Position::new(0, 0),
                                        lsp::Position::new(0, 9),
                                    ),
                                    command: Some(lsp::Command {
                                        title: format!("{generation} references"),
                                        command: "lens_cmd".to_string(),
                                        arguments: None,
                                    }),
                                    data: None,
                                }]))
                            }
                        },
                    );
                }
            })),
        )
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.languages().add(rust_lang());
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());
    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    let fake_lsp = fake_lsp_servers.next().await.unwrap();
    cx.run_until_parked();

    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    let actions = lsp_store
        .update(cx, |lsp_store, cx| lsp_store.code_lens_actions(&buffer, cx))
        .await
        .unwrap()
        .expect("Should have code lens actions after the initial fetch");
    assert_eq!(
        actions
            .values()
            .map(|action| action.lsp_action.title().to_owned())
            .collect::<Vec<_>>(),
        vec!["0 references".to_string()],
        "Expected the initial code lens data on the client",
    );

    lens_generation.store(1, Ordering::Release);
    fake_lsp
        .request::<lsp::request::CodeLensRefresh>((), DEFAULT_LSP_REQUEST_TIMEOUT)
        .await
        .into_response()
        .unwrap();
    cx.run_until_parked();

    let actions = lsp_store
        .update(cx, |lsp_store, cx| lsp_store.code_lens_actions(&buffer, cx))
        .await
        .unwrap()
        .expect("Should have code lens actions after the refresh");
    assert_eq!(
        actions
            .values()
            .map(|action| action.lsp_action.title().to_owned())
            .collect::<Vec<_>>(),
        vec!["1 references".to_string()],
        "Expected the client to re-fetch code lens data after the server-initiated refresh, without any buffer edits",
    );
}

#[gpui::test]
async fn test_remote_lsp_data_cache_converges_with_multiple_servers(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
            "project2": {
                ".git": {},
                "src": {
                    "lib.rs": "fn two() -> usize { 2 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    let capabilities = lsp::ServerCapabilities {
        code_lens_provider: Some(lsp::CodeLensOptions {
            resolve_provider: None,
        }),
        ..lsp::ServerCapabilities::default()
    };

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: Arc::new(LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..LanguageMatcher::default()
            }),
            ..LanguageConfig::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                capabilities: capabilities.clone(),
                ..FakeLspAdapter::default()
            },
        );
    });

    let queried_uris = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut fake_lsp_servers = server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            capabilities,
            Some(Box::new({
                let queried_uris = queried_uris.clone();
                move |fake_lsp| {
                    let queried_uris = queried_uris.clone();
                    fake_lsp.set_request_handler::<lsp::request::CodeLensRequest, _, _>(
                        move |params, _| {
                            queried_uris.lock().unwrap().push(params.text_document.uri);
                            async move {
                                Ok(Some(vec![lsp::CodeLens {
                                    range: lsp::Range::new(
                                        lsp::Position::new(0, 0),
                                        lsp::Position::new(0, 9),
                                    ),
                                    command: Some(lsp::Command {
                                        title: "the lens".to_string(),
                                        command: "lens_cmd".to_string(),
                                        arguments: None,
                                    }),
                                    data: None,
                                }]))
                            }
                        },
                    );
                }
            })),
        )
    });

    cx.run_until_parked();

    project.update(cx, |project, cx| {
        project.languages().add(rust_lang());
        cx.notify();
    });
    let mut buffers = Vec::new();
    let mut fake_lsps = Vec::new();
    for project_root in [path!("/code/project1"), path!("/code/project2")] {
        let worktree_id = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(project_root, true, cx)
            })
            .await
            .unwrap()
            .0
            .read_with(cx, |worktree, _| worktree.id());
        cx.run_until_parked();
        buffers.push(
            project
                .update(cx, |project, cx| {
                    project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
                })
                .await
                .unwrap(),
        );
        fake_lsps.push(fake_lsp_servers.next().await.unwrap());
        cx.run_until_parked();
    }
    let (buffer_1, _handle_1) = &buffers[0];
    let queried_lib_1 = vec![lsp::Uri::from_file_path(path!("/code/project1/src/lib.rs")).unwrap()];

    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    lsp_store
        .update(cx, |lsp_store, cx| {
            lsp_store.code_lens_actions(buffer_1, cx)
        })
        .await
        .unwrap();
    assert_eq!(
        *queried_uris.lock().unwrap(),
        queried_lib_1,
        "Expected the initial fetch to query the buffer's own language server only",
    );

    lsp_store
        .update(cx, |lsp_store, cx| {
            lsp_store.code_lens_actions(buffer_1, cx)
        })
        .await
        .unwrap();
    assert_eq!(
        *queried_uris.lock().unwrap(),
        queried_lib_1,
        "Expected a repeated fetch to be served from the cache, despite the other worktree's server \
        never answering for this buffer",
    );

    let (buffer_1, _) = &buffers[0];
    buffer_1.update(cx, |buffer, cx| buffer.set_language(None, cx));
    cx.run_until_parked();
    let actions = lsp_store
        .update(cx, |lsp_store, cx| {
            lsp_store.code_lens_actions(buffer_1, cx)
        })
        .await
        .unwrap()
        .expect("Should have code lens actions cached");
    assert_eq!(
        actions
            .values()
            .map(|action| action.lsp_action.title().to_owned())
            .collect::<Vec<_>>(),
        vec!["the lens".to_string()],
        "Expected the cached data to survive an empty relevant server set (buffer without a language)",
    );
    assert_eq!(
        *queried_uris.lock().unwrap(),
        queried_lib_1,
        "Expected no new queries for the buffer without a language",
    );
}

#[gpui::test]
async fn test_remote_per_server_refresh_queries_only_the_refreshed_server(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.fake": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    let capabilities = lsp::ServerCapabilities {
        code_lens_provider: Some(lsp::CodeLensOptions {
            resolve_provider: None,
        }),
        ..lsp::ServerCapabilities::default()
    };

    // Both the client and the headless registries need the language and the adapters:
    // the client filters relevant servers by adapter names, while the headless one
    // detects the buffer's language and starts the servers.
    let fake_language_config = || LanguageConfig {
        name: "Fake".into(),
        matcher: Arc::new(LanguageMatcher {
            path_suffixes: vec!["fake".into()],
            ..LanguageMatcher::default()
        }),
        ..LanguageConfig::default()
    };
    let fake_adapter =
        |server_name: &'static str, capabilities: &lsp::ServerCapabilities| FakeLspAdapter {
            name: server_name,
            capabilities: capabilities.clone(),
            ..FakeLspAdapter::default()
        };
    cx.update_entity(&project, |project, _| {
        project
            .languages()
            .register_test_language(fake_language_config());
        for server_name in ["server-a", "server-b"] {
            project
                .languages()
                .register_fake_lsp_adapter("Fake", fake_adapter(server_name, &capabilities));
        }
    });

    let request_counter = |fake_lsp: &mut lsp::FakeLanguageServer, counter: &Arc<AtomicUsize>| {
        let counter = counter.clone();
        fake_lsp.set_request_handler::<lsp::request::CodeLensRequest, _, _>(move |_, _| {
            counter.fetch_add(1, Ordering::Release);
            async move { Ok(Some(Vec::new())) }
        });
    };
    let lens_requests_a = Arc::new(AtomicUsize::new(0));
    let lens_requests_b = Arc::new(AtomicUsize::new(0));
    let (mut fake_lsp_servers_a, mut fake_lsp_servers_b) = server_cx.update(|cx| {
        let languages = &headless.read(cx).languages;
        languages.register_test_language(fake_language_config());
        for server_name in ["server-a", "server-b"] {
            languages.register_fake_lsp_adapter("Fake", fake_adapter(server_name, &capabilities));
        }
        let fake_lsp_servers_a = languages.register_fake_lsp_server(
            LanguageServerName("server-a".into()),
            capabilities.clone(),
            Some(Box::new({
                let lens_requests_a = lens_requests_a.clone();
                move |fake_lsp| request_counter(fake_lsp, &lens_requests_a)
            })),
        );
        let fake_lsp_servers_b = languages.register_fake_lsp_server(
            LanguageServerName("server-b".into()),
            capabilities.clone(),
            Some(Box::new({
                let lens_requests_b = lens_requests_b.clone();
                move |fake_lsp| request_counter(fake_lsp, &lens_requests_b)
            })),
        );
        (fake_lsp_servers_a, fake_lsp_servers_b)
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());
    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.fake")), cx)
        })
        .await
        .unwrap();
    let fake_lsp_a = fake_lsp_servers_a.next().await.unwrap();
    let _fake_lsp_b = fake_lsp_servers_b.next().await.unwrap();
    cx.run_until_parked();

    let lsp_store = project.read_with(cx, |project, _| project.lsp_store());
    lsp_store
        .update(cx, |lsp_store, cx| lsp_store.code_lens_actions(&buffer, cx))
        .await
        .unwrap();
    assert_eq!(
        (
            lens_requests_a.load(Ordering::Acquire),
            lens_requests_b.load(Ordering::Acquire),
        ),
        (1, 1),
        "Expected the initial fetch to query both servers",
    );

    fake_lsp_a
        .request::<lsp::request::CodeLensRefresh>((), DEFAULT_LSP_REQUEST_TIMEOUT)
        .await
        .into_response()
        .unwrap();
    cx.run_until_parked();

    lsp_store
        .update(cx, |lsp_store, cx| lsp_store.code_lens_actions(&buffer, cx))
        .await
        .unwrap();
    assert_eq!(
        (
            lens_requests_a.load(Ordering::Acquire),
            lens_requests_b.load(Ordering::Acquire),
        ),
        (2, 1),
        "Expected the refetch after a per-server refresh to query the refreshed server only",
    );
}

#[gpui::test]
async fn test_remote_code_lens_resolve(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
        editor::init(cx);
        SettingsStore::update_global(cx, |store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.editor.code_lens = Some(settings::CodeLens::Menu);
            });
        });
    });

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    let capabilities = lsp::ServerCapabilities {
        code_lens_provider: Some(lsp::CodeLensOptions {
            resolve_provider: Some(true),
        }),
        ..lsp::ServerCapabilities::default()
    };

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                capabilities: capabilities.clone(),
                ..FakeLspAdapter::default()
            },
        );
    });

    server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            capabilities,
            Some(Box::new(|fake_lsp| {
                fake_lsp.set_request_handler::<lsp::request::CodeLensRequest, _, _>(
                    |_, _| async move {
                        Ok(Some(vec![lsp::CodeLens {
                            range: lsp::Range::new(
                                lsp::Position::new(0, 0),
                                lsp::Position::new(0, 9),
                            ),
                            command: None,
                            data: Some(serde_json::json!({ "id": "lens" })),
                        }]))
                    },
                );
                fake_lsp.set_request_handler::<lsp::request::CodeLensResolve, _, _>(
                    |lens, _| async move {
                        Ok(lsp::CodeLens {
                            command: Some(lsp::Command {
                                title: "1 reference".to_string(),
                                command: "noop".to_string(),
                                arguments: None,
                            }),
                            ..lens
                        })
                    },
                );
            })),
        );
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.languages().add(rust_lang());
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());
    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let cx = cx.add_empty_window();
    let workspace = cx.new_window_entity(|window, cx| {
        workspace::Workspace::test_new(project.clone(), window, cx)
    });
    let editor = cx.new_window_entity(|window, cx| {
        Editor::for_buffer(buffer.clone(), Some(project.clone()), window, cx)
    });
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_item_to_active_pane(Box::new(editor.clone()), None, true, window, cx);
    });

    editor.update_in(cx, |editor, window, cx| {
        editor.change_selections(SelectionEffects::default(), window, cx, |s| {
            s.select_ranges([Point::new(0, 0)..Point::new(0, 0)]);
        });
    });
    cx.executor()
        .advance_clock(editor::CODE_ACTIONS_DEBOUNCE_TIMEOUT * 2);
    cx.run_until_parked();

    editor.update_in(cx, |editor, window, cx| {
        editor.toggle_code_actions(
            &ToggleCodeActions {
                deployed_from: None,
                quick_launch: false,
            },
            window,
            cx,
        );
    });
    cx.run_until_parked();

    editor.update(cx, |editor, _| {
        assert!(editor.context_menu_visible());
        let menu = editor.context_menu().borrow();
        let actions_menu = match menu.as_ref() {
            Some(CodeContextMenu::CodeActions(m)) => m,
            _ => panic!("Expected code actions menu to be visible"),
        };
        let item = actions_menu
            .actions
            .get(0)
            .expect("Expected at least one item in menu");
        assert_eq!(item.label(), "1 reference");
    });
}

#[gpui::test]
async fn test_remote_cancel_language_server_work(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    fs.insert_tree(
        path!("/code/project1/.zed"),
        json!({
            "settings.json": r#"
          {
            "languages": {"Rust":{"language_servers":["rust-analyzer"]}},
            "lsp": {
              "rust-analyzer": {
                "binary": {
                  "path": "~/.cargo/bin/rust-analyzer"
                }
              }
            }
          }"#
        }),
    )
    .await;

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                ..Default::default()
            },
        )
    });

    let mut fake_lsp = server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            Default::default(),
            None,
        )
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());

    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    let mut fake_lsp = fake_lsp.next().await.unwrap();

    // Cancelling all language server work for a given buffer
    {
        // Two operations, one cancellable and one not.
        fake_lsp
            .start_progress_with(
                "another-token",
                lsp::WorkDoneProgressBegin {
                    cancellable: Some(false),
                    ..Default::default()
                },
                DEFAULT_LSP_REQUEST_TIMEOUT,
            )
            .await;

        let progress_token = "the-progress-token";
        fake_lsp
            .start_progress_with(
                progress_token,
                lsp::WorkDoneProgressBegin {
                    cancellable: Some(true),
                    ..Default::default()
                },
                DEFAULT_LSP_REQUEST_TIMEOUT,
            )
            .await;

        cx.executor().run_until_parked();

        project.update(cx, |project, cx| {
            project.cancel_language_server_work_for_buffers([buffer.clone()], cx)
        });

        cx.executor().run_until_parked();

        // Verify the cancellation was received on the server side
        let cancel_notification = fake_lsp
            .receive_notification::<lsp::notification::WorkDoneProgressCancel>()
            .await;
        assert_eq!(
            cancel_notification.token,
            lsp::NumberOrString::String(progress_token.into())
        );
    }

    // Cancelling work by server_id and token
    {
        let server_id = fake_lsp.server.server_id();
        let progress_token = "the-progress-token";

        fake_lsp
            .start_progress_with(
                progress_token,
                lsp::WorkDoneProgressBegin {
                    cancellable: Some(true),
                    ..Default::default()
                },
                DEFAULT_LSP_REQUEST_TIMEOUT,
            )
            .await;

        cx.executor().run_until_parked();

        project.update(cx, |project, cx| {
            project.cancel_language_server_work(
                server_id,
                Some(ProgressToken::String(SharedString::from(progress_token))),
                cx,
            )
        });

        cx.executor().run_until_parked();

        // Verify the cancellation was received on the server side
        let cancel_notification = fake_lsp
            .receive_notification::<lsp::notification::WorkDoneProgressCancel>()
            .await;
        assert_eq!(
            cancel_notification.token,
            lsp::NumberOrString::String(progress_token.to_owned())
        );
    }
}

#[gpui::test]
async fn test_remote_reload(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    let worktree_id = cx.update(|cx| worktree.read(cx).id());

    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    fs.save(
        &PathBuf::from(path!("/code/project1/src/lib.rs")),
        &("bangles".to_string().into()),
        LineEnding::Unix,
    )
    .await
    .unwrap();

    cx.run_until_parked();

    buffer.update(cx, |buffer, cx| {
        assert_eq!(buffer.text(), "bangles");
        buffer.edit([(0..0, "a")], None, cx);
    });

    fs.save(
        &PathBuf::from(path!("/code/project1/src/lib.rs")),
        &("bloop".to_string().into()),
        LineEnding::Unix,
    )
    .await
    .unwrap();

    cx.run_until_parked();
    cx.update(|cx| {
        assert!(buffer.read(cx).has_conflict());
    });

    project
        .update(cx, |project, cx| {
            project.reload_buffers([buffer.clone()].into_iter().collect(), false, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    cx.update(|cx| {
        assert!(!buffer.read(cx).has_conflict());
    });
}

#[gpui::test]
async fn test_remote_resolve_path_in_buffer(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    // Even though we are not testing anything from project1, it is necessary to test if project2 is picking up correct worktree
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
            "project2": {
                ".git": {},
                "README.md": "# project 2",
                "src": {
                    "lib.rs": "fn two() -> usize { 2 }"
                }
            }
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;

    let _ = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    let (worktree2, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project2"), true, cx)
        })
        .await
        .unwrap();

    let worktree2_id = cx.update(|cx| worktree2.read(cx).id());

    cx.run_until_parked();

    let buffer2 = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree2_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    let path = project
        .update(cx, |project, cx| {
            project.resolve_path_in_buffer(path!("/code/project2/README.md"), &buffer2, cx)
        })
        .await
        .unwrap();
    assert!(path.is_file());
    assert_eq!(path.abs_path().unwrap(), path!("/code/project2/README.md"));

    let path = project
        .update(cx, |project, cx| {
            project.resolve_path_in_buffer("../README.md", &buffer2, cx)
        })
        .await
        .unwrap();
    assert!(path.is_file());
    assert_eq!(
        path.project_path().unwrap().clone(),
        (worktree2_id, rel_path("README.md")).into()
    );

    let path = project
        .update(cx, |project, cx| {
            project.resolve_path_in_buffer("../src", &buffer2, cx)
        })
        .await
        .unwrap();
    assert_eq!(
        path.project_path().unwrap().clone(),
        (worktree2_id, rel_path("src")).into()
    );
    assert!(path.is_dir());
}

#[gpui::test]
async fn test_remote_resolve_abs_path(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;

    let path = project
        .update(cx, |project, cx| {
            project.resolve_abs_path(path!("/code/project1/README.md"), cx)
        })
        .await
        .unwrap();

    assert!(path.is_file());
    assert_eq!(path.abs_path().unwrap(), path!("/code/project1/README.md"));

    let path = project
        .update(cx, |project, cx| {
            project.resolve_abs_path(path!("/code/project1/src"), cx)
        })
        .await
        .unwrap();

    assert!(path.is_dir());
    assert_eq!(path.abs_path().unwrap(), path!("/code/project1/src"));

    let path = project
        .update(cx, |project, cx| {
            project.resolve_abs_path(path!("/code/project1/DOESNOTEXIST"), cx)
        })
        .await;
    assert!(path.is_none());
}

#[gpui::test(iterations = 10)]
async fn test_canceling_buffer_opening(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/code",
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/project1", true, cx)
        })
        .await
        .unwrap();
    let worktree_id = worktree.read_with(cx, |tree, _| tree.id());

    // Open a buffer on the client but cancel after a random amount of time.
    let buffer = project.update(cx, |p, cx| {
        p.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
    });
    cx.executor().simulate_random_delay().await;
    drop(buffer);

    // Try opening the same buffer again as the client, and ensure we can
    // still do it despite the cancellation above.
    let buffer = project
        .update(cx, |p, cx| {
            p.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    buffer.read_with(cx, |buf, _| {
        assert_eq!(buf.text(), "fn one() -> usize { 1 }")
    });
}

#[gpui::test]
async fn test_adding_then_removing_then_adding_worktrees(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
            "project2": {
                "README.md": "# project 2",
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (_worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    let (worktree_2, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project2"), true, cx)
        })
        .await
        .unwrap();
    let worktree_id_2 = worktree_2.read_with(cx, |tree, _| tree.id());

    project.update(cx, |project, cx| project.remove_worktree(worktree_id_2, cx));

    let (worktree_2, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project2"), true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();
    worktree_2.update(cx, |worktree, _cx| {
        assert!(worktree.is_visible());
        let entries = worktree.entries(true, 0).collect::<Vec<_>>();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].path.as_unix_str(), "README.md")
    })
}

#[gpui::test]
async fn test_open_server_settings(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let buffer = project.update(cx, |project, cx| project.open_server_settings(cx));
    cx.executor().run_until_parked();

    let buffer = buffer.await.unwrap();

    cx.update(|cx| {
        assert_eq!(
            buffer.read(cx).text(),
            initial_server_settings_content()
                .to_string()
                .replace("\r\n", "\n")
        )
    })
}

#[gpui::test(iterations = 20)]
async fn test_reconnect(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;

    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    buffer.update(cx, |buffer, cx| {
        assert_eq!(buffer.text(), "fn one() -> usize { 1 }");
        let ix = buffer.text().find('1').unwrap();
        buffer.edit([(ix..ix + 1, "100")], None, cx);
    });

    let client = cx.read(|cx| project.read(cx).remote_client().unwrap());
    client
        .update(cx, |client, cx| client.simulate_disconnect(cx))
        .detach();

    project
        .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
        .await
        .unwrap();

    assert_eq!(
        fs.load(path!("/code/project1/src/lib.rs").as_ref())
            .await
            .unwrap(),
        "fn one() -> usize { 100 }"
    );
}

#[gpui::test]
async fn test_remote_root_rename(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/code",
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
            },
        }),
    )
    .await;

    let (project, _) = init_test(&fs, cx, server_cx).await;

    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/project1", true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    fs.rename(
        &PathBuf::from("/code/project1"),
        &PathBuf::from("/code/project2"),
        Default::default(),
    )
    .await
    .unwrap();

    cx.run_until_parked();
    worktree.update(cx, |worktree, _| {
        assert_eq!(worktree.root_name(), "project2")
    })
}

#[gpui::test]
async fn test_remote_rename_entry(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/code",
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
            },
        }),
    )
    .await;

    let (project, _) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/project1", true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    let entry = project
        .update(cx, |project, cx| {
            let worktree = worktree.read(cx);
            let entry = worktree.entry_for_path(rel_path("README.md")).unwrap();
            project.rename_entry(entry.id, (worktree.id(), rel_path("README.rst")).into(), cx)
        })
        .await
        .unwrap()
        .into_included()
        .unwrap();

    cx.run_until_parked();

    worktree.update(cx, |worktree, _| {
        assert_eq!(
            worktree.entry_for_path(rel_path("README.rst")).unwrap().id,
            entry.id
        )
    });
}

#[gpui::test]
async fn test_copy_file_into_remote_project(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let remote_fs = FakeFs::new(server_cx.executor());
    remote_fs
        .insert_tree(
            path!("/code"),
            json!({
                "project1": {
                    ".git": {},
                    "README.md": "# project 1",
                    "src": {
                        "main.rs": ""
                    }
                },
            }),
        )
        .await;

    let (project, _) = init_test(&remote_fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    let local_fs = project
        .read_with(cx, |project, _| project.fs().clone())
        .as_fake();
    local_fs
        .insert_tree(
            path!("/local-code"),
            json!({
                "dir1": {
                    "file1": "file 1 content",
                    "dir2": {
                        "file2": "file 2 content",
                        "dir3": {
                            "file3": ""
                        },
                        "dir4": {}
                    },
                    "dir5": {}
                },
                "file4": "file 4 content"
            }),
        )
        .await;

    worktree
        .update(cx, |worktree, cx| {
            worktree.copy_external_entries(
                rel_path("src").into(),
                vec![
                    Path::new(path!("/local-code/dir1/file1")).into(),
                    Path::new(path!("/local-code/dir1/dir2")).into(),
                ],
                local_fs.clone(),
                cx,
            )
        })
        .await
        .unwrap();

    assert_eq!(
        remote_fs.paths(true),
        vec![
            PathBuf::from(path!("/")),
            PathBuf::from(path!("/code")),
            PathBuf::from(path!("/code/project1")),
            PathBuf::from(path!("/code/project1/.git")),
            PathBuf::from(path!("/code/project1/README.md")),
            PathBuf::from(path!("/code/project1/src")),
            PathBuf::from(path!("/code/project1/src/dir2")),
            PathBuf::from(path!("/code/project1/src/file1")),
            PathBuf::from(path!("/code/project1/src/main.rs")),
            PathBuf::from(path!("/code/project1/src/dir2/dir3")),
            PathBuf::from(path!("/code/project1/src/dir2/dir4")),
            PathBuf::from(path!("/code/project1/src/dir2/file2")),
            PathBuf::from(path!("/code/project1/src/dir2/dir3/file3")),
        ]
    );
    assert_eq!(
        remote_fs
            .load(path!("/code/project1/src/file1").as_ref())
            .await
            .unwrap(),
        "file 1 content"
    );
    assert_eq!(
        remote_fs
            .load(path!("/code/project1/src/dir2/file2").as_ref())
            .await
            .unwrap(),
        "file 2 content"
    );
    assert_eq!(
        remote_fs
            .load(path!("/code/project1/src/dir2/dir3/file3").as_ref())
            .await
            .unwrap(),
        ""
    );
}

#[gpui::test]
async fn test_remote_root_repo_common_dir(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/code",
        json!({
            "main_repo": {
                ".git": {},
                "file.txt": "content",
            },
            "no_git": {
                "file.txt": "content",
            },
        }),
    )
    .await;

    // Create a linked worktree that points back to main_repo's .git.
    fs.add_linked_worktree_for_repo(
        Path::new("/code/main_repo/.git"),
        false,
        GitWorktree {
            path: PathBuf::from("/code/linked_worktree"),
            ref_name: Some("refs/heads/feature-branch".into()),
            sha: "abc123".into(),
            is_main: false,
            is_bare: false,
        },
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;

    // Main repo: root_repo_common_dir should be the .git directory itself.
    let (worktree_main, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/main_repo", true, cx)
        })
        .await
        .unwrap();
    cx.executor().run_until_parked();

    let (common_dir, is_linked_worktree) = worktree_main.read_with(cx, |worktree, _| {
        (
            worktree.snapshot().root_repo_common_dir().cloned(),
            worktree.snapshot().root_repo_is_linked_worktree(),
        )
    });
    assert_eq!(
        common_dir.as_deref(),
        Some(Path::new("/code/main_repo/.git")),
    );
    assert!(!is_linked_worktree);

    // Linked worktree: root_repo_common_dir should point to the main repo's .git.
    let (worktree_linked, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/linked_worktree", true, cx)
        })
        .await
        .unwrap();
    cx.executor().run_until_parked();

    let (common_dir, is_linked_worktree) = worktree_linked.read_with(cx, |worktree, _| {
        (
            worktree.snapshot().root_repo_common_dir().cloned(),
            worktree.snapshot().root_repo_is_linked_worktree(),
        )
    });
    assert_eq!(
        common_dir.as_deref(),
        Some(Path::new("/code/main_repo/.git")),
    );
    assert!(is_linked_worktree);

    let main_worktree_paths = project.read_with(cx, |project, cx| {
        project
            .worktree_paths(cx)
            .main_worktree_path_list()
            .ordered_paths()
            .map(|path| path.to_path_buf())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        main_worktree_paths,
        vec![
            PathBuf::from("/code/main_repo"),
            PathBuf::from("/code/main_repo"),
        ]
    );

    // No git repo: root_repo_common_dir should be None.
    let (worktree_no_git, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/no_git", true, cx)
        })
        .await
        .unwrap();
    cx.executor().run_until_parked();

    let (common_dir, is_linked_worktree) = worktree_no_git.read_with(cx, |worktree, _| {
        (
            worktree.snapshot().root_repo_common_dir().cloned(),
            worktree.snapshot().root_repo_is_linked_worktree(),
        )
    });
    assert_eq!(common_dir, None);
    assert!(!is_linked_worktree);
}

#[gpui::test]
async fn test_remote_search_commits_streams_proto_chunks(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    const COMMIT_COUNT: usize = 900;
    const RESPONSE_MAX_SIZE: usize = 100;

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "file.txt": "content",
            },
        }),
    )
    .await;

    let commit_data = (0..COMMIT_COUNT)
        .map(|index| {
            let sha = Oid::from_str(&format!("{:040x}", index + 1)).unwrap();
            (
                CommitData {
                    sha,
                    parents: Default::default(),
                    author_name: SharedString::from("Author"),
                    author_email: SharedString::from("author@example.com"),
                    commit_timestamp: index as i64,
                    subject: SharedString::from(format!("Subject {index}")),
                    message: SharedString::from(format!("needle commit {index}")),
                },
                false,
            )
        })
        .collect::<Vec<_>>();
    let expected_shas = commit_data
        .iter()
        .map(|(commit_data, _)| commit_data.sha.to_string())
        .collect::<HashSet<_>>();
    fs.set_commit_data(Path::new(path!("/code/project1/.git")), commit_data);

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .expect("should open remote worktree");
    server_cx.run_until_parked();
    cx.run_until_parked();
    project
        .update(cx, |project, cx| project.git_scans_complete(cx))
        .await;

    let (remote_client, repository_id) = project.read_with(cx, |project, cx| {
        let repository = project
            .active_repository(cx)
            .expect("remote project should have an active repository");
        let repository_id = repository.read(cx).snapshot().id;
        let remote_client = project
            .remote_client()
            .expect("project should have a remote client");
        (remote_client, repository_id)
    });
    let proto_client = remote_client.read_with(cx, |remote_client, _| remote_client.proto_client());
    let mut stream = proto_client
        .request_stream(proto::SearchCommits {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            repository_id: repository_id.to_proto(),
            log_source: Some(proto::GitLogSource {
                source: Some(proto::git_log_source::Source::All(
                    proto::GitLogSourceAll {},
                )),
            }),
            query: "needle".to_string(),
            case_sensitive: true,
        })
        .await
        .expect("search commits stream should start");

    let mut chunks = Vec::new();
    while let Some(response) = futures::StreamExt::next(&mut stream).await {
        chunks.push(response.expect("search commits chunk should succeed").shas);
    }

    assert!(
        chunks.len() > 1,
        "expected search results to stream in multiple chunks"
    );
    for chunk in chunks.iter().take(chunks.len() - 1) {
        assert!(
            chunk.len() <= RESPONSE_MAX_SIZE,
            "non-final chunks should meet the target byte size"
        );
    }

    let actual_shas = chunks.into_iter().flatten().collect::<HashSet<_>>();
    assert_eq!(actual_shas, expected_shas);
}

#[gpui::test]
async fn test_remote_archive_git_operations_are_supported(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/project",
        json!({
            ".git": {},
            "file.txt": "content",
        }),
    )
    .await;
    fs.set_branch_name(Path::new("/project/.git"), Some("main"));
    fs.set_head_for_repo(
        Path::new("/project/.git"),
        &[("file.txt", "content".into())],
        "head-sha",
    );

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(Path::new("/project"), true, cx)
        })
        .await
        .expect("should open remote worktree");
    cx.run_until_parked();

    let repository = project.read_with(cx, |project, cx| {
        project
            .active_repository(cx)
            .expect("remote project should have an active repository")
    });

    let head_sha = cx
        .update(|cx| repository.update(cx, |repository, _| repository.head_sha()))
        .await
        .expect("head_sha request should complete")
        .expect("head_sha should succeed")
        .expect("HEAD should exist");

    cx.run_until_parked();

    cx.update(|cx| {
        repository.update(cx, |repository, _| {
            repository.update_ref("refs/zed-tests/archive-checkpoint".to_string(), head_sha)
        })
    })
    .await
    .expect("update_ref request should complete")
    .expect("update_ref should succeed for remote repository");

    cx.run_until_parked();

    cx.update(|cx| {
        repository.update(cx, |repository, _| {
            repository.delete_ref("refs/zed-tests/archive-checkpoint".to_string())
        })
    })
    .await
    .expect("delete_ref request should complete")
    .expect("delete_ref should succeed for remote repository");

    cx.run_until_parked();

    cx.update(|cx| repository.update(cx, |repository, _| repository.repair_worktrees()))
        .await
        .expect("repair_worktrees request should complete")
        .expect("repair_worktrees should succeed for remote repository");

    cx.run_until_parked();

    let (staged_commit_sha, unstaged_commit_sha) = cx
        .update(|cx| repository.update(cx, |repository, _| repository.create_archive_checkpoint()))
        .await
        .expect("create_archive_checkpoint request should complete")
        .expect("create_archive_checkpoint should succeed for remote repository");

    cx.run_until_parked();

    cx.update(|cx| {
        repository.update(cx, |repository, _| {
            repository.restore_archive_checkpoint(staged_commit_sha, unstaged_commit_sha)
        })
    })
    .await
    .expect("restore_archive_checkpoint request should complete")
    .expect("restore_archive_checkpoint should succeed for remote repository");
}

#[gpui::test]
async fn test_add_path_to_gitignore_in_remote_repository(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/project",
        json!({
            ".git": {},
            ".gitignore": "existing\n",
            "logs": {
                "app.log": ""
            },
            "tmp.txt": "",
        }),
    )
    .await;
    fs.set_branch_name(Path::new("/project/.git"), Some("main"));
    fs.set_head_for_repo(
        Path::new("/project/.git"),
        &[(".gitignore", "existing\n".into())],
        "head-sha",
    );

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(Path::new("/project"), true, cx)
        })
        .await
        .expect("should open remote worktree");
    cx.run_until_parked();

    let repository = project.read_with(cx, |project, cx| {
        project
            .active_repository(cx)
            .expect("remote project should have an active repository")
    });

    for (path, is_dir) in [("tmp.txt", false), ("logs", true), ("tmp.txt", false)] {
        let repo_path = RepoPath::new(path).expect("path should be a valid repo path");
        cx.update(|cx| {
            repository.update(cx, |repository, _| {
                repository.add_path_to_gitignore(&repo_path, is_dir)
            })
        })
        .await
        .expect("add to .gitignore request should complete")
        .expect("add to .gitignore should succeed for remote repository");
    }

    server_cx.run_until_parked();
    cx.run_until_parked();

    assert_eq!(
        fs.load(Path::new("/project/.gitignore"))
            .await
            .expect(".gitignore should be readable"),
        "existing\ntmp.txt\nlogs/\n"
    );
}

#[gpui::test]
async fn test_add_path_to_git_info_exclude_in_remote_repository(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/project",
        json!({
            ".git": {},
            "logs": {
                "app.log": ""
            },
            "tmp.txt": "",
        }),
    )
    .await;
    fs.set_branch_name(Path::new("/project/.git"), Some("main"));
    fs.set_head_for_repo(Path::new("/project/.git"), &[], "head-sha");

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(Path::new("/project"), true, cx)
        })
        .await
        .expect("should open remote worktree");
    cx.run_until_parked();

    let repository = project.read_with(cx, |project, cx| {
        project
            .active_repository(cx)
            .expect("remote project should have an active repository")
    });

    for (path, is_dir) in [("tmp.txt", false), ("logs", true), ("tmp.txt", false)] {
        let repo_path = RepoPath::new(path).expect("path should be a valid repo path");
        cx.update(|cx| {
            repository.update(cx, |repository, _| {
                repository.add_path_to_git_info_exclude(&repo_path, is_dir)
            })
        })
        .await
        .expect("add to info/exclude request should complete")
        .expect("add to info/exclude should succeed for remote repository");
    }

    server_cx.run_until_parked();
    cx.run_until_parked();

    assert_eq!(
        fs.load(Path::new("/project/.git/info/exclude"))
            .await
            .expect("info/exclude should be readable"),
        "tmp.txt\nlogs/\n"
    );
}

#[gpui::test]
async fn test_remote_git_diffs(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let text_2 = "
        fn one() -> usize {
            1
        }
    "
    .unindent();
    let text_1 = "
        fn one() -> usize {
            0
        }
    "
    .unindent();

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/code",
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": text_2
                },
                "README.md": "# project 1",
            },
        }),
    )
    .await;
    fs.set_index_for_repo(
        Path::new("/code/project1/.git"),
        &[("src/lib.rs", text_1.clone())],
    );
    fs.set_head_for_repo(
        Path::new("/code/project1/.git"),
        &[("src/lib.rs", text_1.clone())],
        "deadbeef",
    );

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/code/project1", true, cx)
        })
        .await
        .unwrap();
    let worktree_id = cx.update(|cx| worktree.read(cx).id());
    cx.executor().run_until_parked();

    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    let diff = project
        .update(cx, |project, cx| {
            project.open_uncommitted_diff(buffer.clone(), cx)
        })
        .await
        .unwrap();

    diff.read_with(cx, |diff, cx| {
        assert_eq!(diff.base_text_string(cx).unwrap(), text_1);
        assert_eq!(
            diff.secondary_diff()
                .unwrap()
                .read(cx)
                .base_text_string(cx)
                .unwrap(),
            text_1
        );
    });

    // stage the current buffer's contents
    fs.set_index_for_repo(
        Path::new("/code/project1/.git"),
        &[("src/lib.rs", text_2.clone())],
    );

    cx.executor().run_until_parked();
    diff.read_with(cx, |diff, cx| {
        assert_eq!(diff.base_text_string(cx).unwrap(), text_1);
        assert_eq!(
            diff.secondary_diff()
                .unwrap()
                .read(cx)
                .base_text_string(cx)
                .unwrap(),
            text_2
        );
    });

    // commit the current buffer's contents
    fs.set_head_for_repo(
        Path::new("/code/project1/.git"),
        &[("src/lib.rs", text_2.clone())],
        "deadbeef",
    );

    cx.executor().run_until_parked();
    diff.read_with(cx, |diff, cx| {
        assert_eq!(diff.base_text_string(cx).unwrap(), text_2);
        assert_eq!(
            diff.secondary_diff()
                .unwrap()
                .read(cx)
                .base_text_string(cx)
                .unwrap(),
            text_2
        );
    });
}

#[gpui::test]
async fn test_remote_git_diffs_when_recv_update_repository_delay(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
        editor::init(cx);
    });

    use editor::Editor;
    use gpui::VisualContext;
    let text_2 = "
        fn one() -> usize {
            1
        }
    "
    .unindent();
    let text_1 = "
        fn one() -> usize {
            0
        }
    "
    .unindent();

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                "src": {
                    "lib.rs": text_2
                },
                "README.md": "# project 1",
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();
    let worktree_id = cx.update(|cx| worktree.read(cx).id());
    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();
    let buffer_id = cx.update(|cx| buffer.read(cx).remote_id());

    let cx = cx.add_empty_window();
    let editor = cx.new_window_entity(|window, cx| {
        Editor::for_buffer(buffer, Some(project.clone()), window, cx)
    });

    // Remote server will send proto::UpdateRepository after the instance of Editor create.
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
            },
        }),
    )
    .await;

    fs.set_index_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", text_1.clone())],
    );
    fs.set_head_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", text_1.clone())],
        "sha",
    );

    cx.executor().run_until_parked();
    let diff = editor
        .read_with(cx, |editor, cx| {
            editor
                .buffer()
                .read_with(cx, |buffer, _| buffer.diff_for(buffer_id))
        })
        .unwrap();

    diff.read_with(cx, |diff, cx| {
        assert_eq!(diff.base_text_string(cx).unwrap(), text_1);
        assert_eq!(
            diff.secondary_diff()
                .unwrap()
                .read(cx)
                .base_text_string(cx)
                .unwrap(),
            text_1
        );
    });

    // stage the current buffer's contents
    fs.set_index_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", text_2.clone())],
    );

    cx.executor().run_until_parked();
    diff.read_with(cx, |diff, cx| {
        assert_eq!(diff.base_text_string(cx).unwrap(), text_1);
        assert_eq!(
            diff.secondary_diff()
                .unwrap()
                .read(cx)
                .base_text_string(cx)
                .unwrap(),
            text_2
        );
    });

    // commit the current buffer's contents
    fs.set_head_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", text_2.clone())],
        "sha",
    );

    cx.executor().run_until_parked();
    diff.read_with(cx, |diff, cx| {
        assert_eq!(diff.base_text_string(cx).unwrap(), text_2);
        assert_eq!(
            diff.secondary_diff()
                .unwrap()
                .read(cx)
                .base_text_string(cx)
                .unwrap(),
            text_2
        );
    });
}

#[gpui::test]
async fn test_remote_git_branches(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
            },
        }),
    )
    .await;

    let (project, headless_project) = init_test(&fs, cx, server_cx).await;
    let branches = ["main", "dev", "feature-1"];
    let branches_set = branches
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    fs.insert_branches(Path::new(path!("/code/project1/.git")), &branches);

    let (_worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();
    // Give the worktree a bit of time to index the file system
    cx.run_until_parked();

    let repository = project.update(cx, |project, cx| project.active_repository(cx).unwrap());

    let remote_branches = repository
        .update(cx, |repository, _| repository.branches())
        .await
        .unwrap()
        .unwrap()
        .branches;

    let new_branch = branches[2];

    let remote_branches = remote_branches
        .into_iter()
        .map(|branch| branch.name().to_string())
        .collect::<HashSet<_>>();

    assert_eq!(&remote_branches, &branches_set);

    cx.update(|cx| {
        repository.update(cx, |repository, _cx| {
            repository.change_branch(new_branch.to_string())
        })
    })
    .await
    .unwrap()
    .unwrap();

    cx.run_until_parked();

    let server_branch = server_cx.update(|cx| {
        headless_project.update(cx, |headless_project, cx| {
            headless_project.git_store.update(cx, |git_store, cx| {
                git_store
                    .repositories()
                    .values()
                    .next()
                    .unwrap()
                    .read(cx)
                    .branch
                    .as_ref()
                    .unwrap()
                    .clone()
            })
        })
    });

    assert_eq!(server_branch.name(), branches[2]);

    // Also try creating a new branch
    cx.update(|cx| {
        repository.update(cx, |repo, _cx| {
            repo.create_branch("totally-new-branch".to_string(), None)
        })
    })
    .await
    .unwrap()
    .unwrap();

    cx.update(|cx| {
        repository.update(cx, |repo, _cx| {
            repo.change_branch("totally-new-branch".to_string())
        })
    })
    .await
    .unwrap()
    .unwrap();

    cx.run_until_parked();

    let server_branch = server_cx.update(|cx| {
        headless_project.update(cx, |headless_project, cx| {
            headless_project.git_store.update(cx, |git_store, cx| {
                git_store
                    .repositories()
                    .values()
                    .next()
                    .unwrap()
                    .read(cx)
                    .branch
                    .as_ref()
                    .unwrap()
                    .clone()
            })
        })
    });

    assert_eq!(server_branch.name(), "totally-new-branch");

    let default_branch = cx
        .update(|cx| repository.update(cx, |repository, _cx| repository.default_branch(false)))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(default_branch.as_deref(), Some("main"));

    let default_branch_with_remote = cx
        .update(|cx| repository.update(cx, |repository, _cx| repository.default_branch(true)))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(default_branch_with_remote.as_deref(), Some("origin/main"));
}

#[gpui::test]
async fn test_remote_git_checkpoints(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "file.txt": "original content",
            },
        }),
    )
    .await;

    let (project, _headless) = init_test(&fs, cx, server_cx).await;

    let (_worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let repository = project.update(cx, |project, cx| project.active_repository(cx).unwrap());

    // 1. Create a checkpoint of the original state
    let checkpoint_1 = repository
        .update(cx, |repository, _| repository.checkpoint())
        .await
        .unwrap()
        .unwrap();

    // 2. Modify a file on the server-side fs
    fs.write(
        Path::new(path!("/code/project1/file.txt")),
        b"modified content",
    )
    .await
    .unwrap();

    // 3. Create a second checkpoint with the modified state
    let checkpoint_2 = repository
        .update(cx, |repository, _| repository.checkpoint())
        .await
        .unwrap()
        .unwrap();

    // 4. compare_checkpoints: same checkpoint with itself => equal
    let equal = repository
        .update(cx, |repository, _| {
            repository.compare_checkpoints(checkpoint_1.clone(), checkpoint_1.clone())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(equal, "a checkpoint compared with itself should be equal");

    // 5. compare_checkpoints: different states => not equal
    let equal = repository
        .update(cx, |repository, _| {
            repository.compare_checkpoints(checkpoint_1.clone(), checkpoint_2.clone())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(
        !equal,
        "checkpoints of different states should not be equal"
    );

    // 6. diff_checkpoints: same checkpoint => empty diff
    let diff = repository
        .update(cx, |repository, _| {
            repository.diff_checkpoints(checkpoint_1.clone(), checkpoint_1.clone())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(
        diff.is_empty(),
        "diff of identical checkpoints should be empty"
    );

    // 7. diff_checkpoints: different checkpoints => non-empty diff mentioning the changed file
    let diff = repository
        .update(cx, |repository, _| {
            repository.diff_checkpoints(checkpoint_1.clone(), checkpoint_2.clone())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(
        !diff.is_empty(),
        "diff of different checkpoints should be non-empty"
    );
    assert!(
        diff.contains("file.txt"),
        "diff should mention the changed file"
    );
    assert!(
        diff.contains("original content"),
        "diff should contain removed content"
    );
    assert!(
        diff.contains("modified content"),
        "diff should contain added content"
    );

    // 8. restore_checkpoint: restore to original state
    repository
        .update(cx, |repository, _| {
            repository.restore_checkpoint(checkpoint_1.clone())
        })
        .await
        .unwrap()
        .unwrap();
    cx.run_until_parked();

    // 9. Create a checkpoint after restore
    let checkpoint_3 = repository
        .update(cx, |repository, _| repository.checkpoint())
        .await
        .unwrap()
        .unwrap();

    // 10. compare_checkpoints: restored state matches original
    let equal = repository
        .update(cx, |repository, _| {
            repository.compare_checkpoints(checkpoint_1.clone(), checkpoint_3.clone())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(equal, "restored state should match original checkpoint");

    // 11. diff_checkpoints: restored state vs original => empty diff
    let diff = repository
        .update(cx, |repository, _| {
            repository.diff_checkpoints(checkpoint_1.clone(), checkpoint_3.clone())
        })
        .await
        .unwrap()
        .unwrap();
    assert!(diff.is_empty(), "diff after restore should be empty");
}

#[gpui::test]
async fn test_remote_agent_fs_tool_calls(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/project"),
        json!({
            "a.txt": "A",
            "b.txt": "B",
        }),
    )
    .await;

    let (project, _headless_project) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/project"), true, cx)
        })
        .await
        .unwrap();

    let action_log = cx.new(|_| action_log::ActionLog::new(project.clone()));

    let input = ReadFileToolInput {
        path: "project/b.txt".into(),
        start_line: None,
        end_line: None,
    };
    let read_tool = Arc::new(ReadFileTool::new(project, action_log, true));
    let (event_stream, _) = ToolCallEventStream::test();

    let exists_result = cx.update(|cx| {
        read_tool
            .clone()
            .run(ToolInput::resolved(input), event_stream.clone(), cx)
    });
    let output = exists_result.await.unwrap();
    assert_eq!(
        output,
        LanguageModelToolResultContent::Text("     1\tB".into())
    );

    let input = ReadFileToolInput {
        path: "project/c.txt".into(),
        start_line: None,
        end_line: None,
    };
    let does_not_exist_result =
        cx.update(|cx| read_tool.run(ToolInput::resolved(input), event_stream, cx));
    does_not_exist_result.await.unwrap_err();
}

#[gpui::test]
async fn test_adding_remote_skill(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    use acp_thread::AgentConnection as _;

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/project"),
        json!({
            ".agents": {
                "skills": {
                    "test-skill": {
                        "SKILL.md": "---\nname: test-skill\ndescription: test description\n---\ntest body"
                    }
                }
            }
        }),
    )
    .await;

    let (project, _headless_project) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        LanguageModelRegistry::test(cx);
    });
    let (_worktree, _rel_path) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/project"), true, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();
    let thread_store = cx.new(|cx| ThreadStore::new(cx));
    let agent = cx.update(|cx| NativeAgent::new(thread_store, Templates::new(), fs.clone(), cx));
    let connection = Rc::new(NativeAgentConnection(agent.clone()));
    let _acp_thread = cx
        .update(|cx| {
            connection.clone().new_session(
                project.clone(),
                PathList::new(&[Path::new("/project")]),
                cx,
            )
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let skill_tool = Arc::new(SkillTool::with_body_resolver(
        skills_resolver_for_project(agent.downgrade(), project.entity_id()),
        skill_body_resolver_for_project(project.clone(), fs.clone()),
    ));
    let (event_stream, mut event_stream_rx) = ToolCallEventStream::test();

    let input = SkillToolInput {
        name: "test-skill".into(),
    };
    let task = cx.update(|cx| {
        skill_tool
            .clone()
            .run(ToolInput::resolved(input), event_stream.clone(), cx)
    });

    // The project-local skill is not a built-in, so the tool requests
    // authorization. Approve it so the tool can proceed.
    let authorization = event_stream_rx.expect_authorization().await;
    authorization
        .response
        .send(acp_thread::SelectedPermissionOutcome::new(
            agent_client_protocol::schema::v1::PermissionOptionId::new("allow"),
            agent_client_protocol::schema::v1::PermissionOptionKind::AllowOnce,
        ))
        .unwrap();

    let output = task.await.unwrap();
    cx.run_until_parked();
    let expected = format!(
        concat!(
            "<skill_content name=\"test-skill\">\n",
            "<source>project-local</source>\n",
            "<worktree>project</worktree>\n",
            "<directory>{}</directory>\n",
            "Relative paths in this skill resolve against <directory>.\n",
            "\n",
            "test body\n",
            "</skill_content>\n",
        ),
        path!("/project/.agents/skills/test-skill"),
    );
    assert_eq!(output, SkillToolOutput::Found { rendered: expected });

    fs.create_dir(Path::new(path!("/project/.agents/skills/test-2")))
        .await
        .unwrap();
    fs.insert_file(
        path!("/project/.agents/skills/test-2/SKILL.md"),
        "---\nname: test-2\ndescription: test description\n---\ntest body"
            .as_bytes()
            .into(),
    )
    .await;

    cx.run_until_parked();
    cx.update(|cx| connection.refresh_skills_for_project(project, cx));
    cx.run_until_parked();

    let input2 = SkillToolInput {
        name: "test-2".into(),
    };
    let task = cx.update(|cx| {
        skill_tool
            .clone()
            .run(ToolInput::resolved(input2), event_stream.clone(), cx)
    });

    let authorization = event_stream_rx.expect_authorization().await;
    authorization
        .response
        .send(acp_thread::SelectedPermissionOutcome::new(
            agent_client_protocol::schema::v1::PermissionOptionId::new("allow"),
            agent_client_protocol::schema::v1::PermissionOptionKind::AllowOnce,
        ))
        .unwrap();

    let output = task.await.unwrap();
    let expected2 = format!(
        concat!(
            "<skill_content name=\"test-2\">\n",
            "<source>project-local</source>\n",
            "<worktree>project</worktree>\n",
            "<directory>{}</directory>\n",
            "Relative paths in this skill resolve against <directory>.\n",
            "\n",
            "test body\n",
            "</skill_content>\n",
        ),
        path!("/project/.agents/skills/test-2"),
    );
    assert_eq!(
        output,
        SkillToolOutput::Found {
            rendered: expected2
        }
    );
}

#[gpui::test]
async fn test_remote_external_agent_server(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(path!("/project"), json!({})).await;

    let (project, _headless_project) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/project"), true, cx)
        })
        .await
        .unwrap();
    let names = project.update(cx, |project, cx| {
        project
            .agent_server_store()
            .read(cx)
            .external_agents()
            .map(|name| name.to_string())
            .collect::<Vec<_>>()
    });
    pretty_assertions::assert_eq!(names, Vec::<String>::new());
    server_cx.update_global::<SettingsStore, _>(|settings_store, cx| {
        settings_store
            .set_server_settings(
                &json!({
                    "agent_servers": {
                        "foo": {
                            "type": "custom",
                            "command": "foo-cli",
                            "args": ["--flag"],
                            "env": {
                                "VAR": "val"
                            }
                        }
                    }
                })
                .to_string(),
                cx,
            )
            .unwrap();
    });
    server_cx.run_until_parked();
    cx.run_until_parked();
    let names = project.update(cx, |project, cx| {
        project
            .agent_server_store()
            .read(cx)
            .external_agents()
            .map(|name| name.to_string())
            .collect::<Vec<_>>()
    });
    pretty_assertions::assert_eq!(names, ["foo"]);
    let command = project
        .update(cx, |project, cx| {
            project.agent_server_store().update(cx, |store, cx| {
                store
                    .get_external_agent(&"foo".into())
                    .unwrap()
                    .get_command(
                        vec![],
                        HashMap::from_iter([("OTHER_VAR".into(), "other-val".into())]),
                        &mut cx.to_async(),
                    )
            })
        })
        .await
        .unwrap();
    assert_eq!(
        command,
        AgentServerCommand {
            path: "foo-cli".into(),
            args: vec!["--flag".into()],
            env: Some(HashMap::from_iter([
                ("NO_BROWSER".into(), "1".into()),
                ("VAR".into(), "val".into()),
                ("OTHER_VAR".into(), "other-val".into())
            ]))
        }
    );
}

#[gpui::test]
async fn test_remote_apply_code_action_skips_unadvertised_command(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "README.md": "# project 1",
                "src": {
                    "lib.rs": "fn one() -> usize { 1 }"
                }
            },
        }),
    )
    .await;

    let (project, headless) = init_test(&fs, cx, server_cx).await;

    fs.insert_tree(
        path!("/code/project1/.zed"),
        json!({
            "settings.json": r#"
          {
            "languages": {"Rust":{"language_servers":["rust-analyzer"]}},
            "lsp": {
              "rust-analyzer": {
                "binary": {
                  "path": "~/.cargo/bin/rust-analyzer"
                }
              }
            }
          }"#
        }),
    )
    .await;

    cx.update_entity(&project, |project, _| {
        project.languages().register_test_language(LanguageConfig {
            name: "Rust".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["rs".into()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        });
        project.languages().register_fake_lsp_adapter(
            "Rust",
            FakeLspAdapter {
                name: "rust-analyzer",
                ..Default::default()
            },
        )
    });

    // Register the fake LSP with an empty execute_command_provider and a handler that panics
    // if it is ever reached: commands not advertised by the server must be rejected by
    // `apply_code_action` before dispatching to the language server.
    let mut fake_lsp = server_cx.update(|cx| {
        headless.read(cx).languages.register_fake_lsp_server(
            LanguageServerName("rust-analyzer".into()),
            lsp::ServerCapabilities {
                execute_command_provider: Some(lsp::ExecuteCommandOptions {
                    commands: Vec::new(),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Some(Box::new(|fake| {
                fake.set_request_handler::<lsp::request::ExecuteCommand, _, _>(
                    |params, _| async move {
                        panic!(
                            "Unadvertised command {} must not reach the language server",
                            params.command
                        );
                    },
                );
            })),
        )
    });

    cx.run_until_parked();

    let worktree_id = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/code/project1"), true, cx)
        })
        .await
        .unwrap()
        .0
        .read_with(cx, |worktree, _| worktree.id());

    cx.run_until_parked();

    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    cx.run_until_parked();

    let _fake_lsp = fake_lsp.next().await.unwrap();

    let server_id = server_cx.read(|cx| {
        *headless
            .read(cx)
            .lsp_store
            .read(cx)
            .as_local()
            .unwrap()
            .language_servers
            .keys()
            .next()
            .unwrap()
    });
    let buffer_id = cx.read(|cx| buffer.read(cx).remote_id());

    let action = project::CodeAction {
        server_id,
        range: language::Anchor::min_min_range_for_buffer(buffer_id),
        lsp_action: project::LspAction::Command(lsp::Command {
            title: "\u{25b6}\u{fe0e} Run Tests".into(),
            command: "rust-analyzer.runSingle".into(),
            arguments: Some(vec![json!({"label": "test-mod tests"})]),
        }),
        resolved: true,
    };

    let transaction = project
        .update(cx, |project, cx| {
            project.apply_code_action(buffer.clone(), action, true, cx)
        })
        .await
        .expect("Unadvertised command must not be forwarded to executeCommand");
    assert_eq!(transaction.0.len(), 0);
}

#[gpui::test]
async fn test_remote_restore_unstaged_hunk_clears_diff(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    cx.update(|cx| {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        release_channel::init(semver::Version::new(0, 0, 0), cx);
        editor::init(cx);
    });

    use editor::Editor;
    use gpui::VisualContext;

    let base_text = "
        fn one() -> usize {
            1
        }
    "
    .unindent();
    let modified_text = "
        fn one() -> usize {
            100
        }
    "
    .unindent();

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/code"),
        json!({
            "project1": {
                ".git": {},
                "src": {
                    "lib.rs": modified_text
                },
            },
        }),
    )
    .await;
    fs.set_index_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", base_text.clone())],
    );
    fs.set_head_for_repo(
        Path::new(path!("/code/project1/.git")),
        &[("src/lib.rs", base_text.clone())],
        "deadbeef",
    );

    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let worktree_id = {
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/code/project1"), true, cx)
            })
            .await
            .unwrap();
        cx.update(|cx| worktree.read(cx).id())
    };
    cx.executor().run_until_parked();

    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer((worktree_id, rel_path("src/lib.rs")), cx)
        })
        .await
        .unwrap();

    let cx = cx.add_empty_window();
    let editor = cx.new_window_entity(|window, cx| {
        Editor::for_buffer(buffer, Some(project.clone()), window, cx)
    });
    cx.executor().run_until_parked();

    editor.update_in(cx, |editor, window, cx| {
        let snapshot = editor.snapshot(window, cx);
        let hunks: Vec<_> = editor
            .diff_hunks_in_ranges(
                &[editor::Anchor::Min..editor::Anchor::Max],
                &snapshot.buffer_snapshot(),
            )
            .collect();
        assert!(!hunks.is_empty(), "should have diff hunks before restore");
    });

    cx.update_window_entity(&editor, |editor, window, cx| {
        editor.select_all(&editor::actions::SelectAll, window, cx);
        editor.git_restore(&git::Restore, window, cx);
    });
    cx.executor().run_until_parked();

    editor.update_in(cx, |editor, _window, cx| {
        let snapshot = editor.buffer().read(cx).snapshot(cx);
        assert_eq!(
            snapshot.text(),
            base_text,
            "buffer text should match base after restoring all hunks"
        );

        let hunks: Vec<_> = editor
            .diff_hunks_in_ranges(&[editor::Anchor::Min..editor::Anchor::Max], &snapshot)
            .collect();
        assert!(hunks.is_empty(), "should have no diff hunks after restore");
    });
}

#[gpui::test]
async fn test_remote_load_commit_template(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());

    fs.insert_tree(
        path!("/root"),
        json!({
            "project": {
                ".git": {},
                "README.md": "# Project's README"
            }
        }),
    )
    .await;

    fs.with_git_state(Path::new(path!("/root/project/.git")), false, |state| {
        state.commit_template = Some(GitCommitTemplate {
            template: "chore: commit template".to_string(),
        })
    })
    .expect("Should successfully update repository state");

    let (project, _) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/root/project"), true, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let repository = project.update(cx, |project, cx| project.active_repository(cx).unwrap());
    let commit_template = repository
        .update(cx, |repository, _| repository.load_commit_template_text())
        .await
        .unwrap()
        .unwrap()
        .expect("Loading commit template in remote should work");

    assert_eq!(commit_template.template, "chore: commit template");
}

#[gpui::test]
async fn test_remote_delete_project_entry_with_trash(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());

    fs.insert_tree(
        path!("/root"),
        json!({
            "zed": {
                "file_a.txt": "File A"
            }
        }),
    )
    .await;

    let (project, _headless_project) = init_test(&fs, cx, server_cx).await;
    let (worktree, _path) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/root/zed"), true, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let entry_id = worktree.update(cx, |worktree, _cx| {
        worktree.entry_for_path(rel_path("file_a.txt")).unwrap().id
    });

    let remote_client = project.read_with(cx, |project, _cx| {
        project
            .remote_client()
            .expect("project should have a remote client")
    });

    let proto_client =
        remote_client.read_with(cx, |remote_client, _cx| remote_client.proto_client());

    proto_client
        .request(proto::DeleteProjectEntry {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            entry_id: entry_id.to_proto(),
            #[allow(deprecated)]
            use_trash: true,
        })
        .await
        .unwrap();

    assert_eq!(
        fs.trashed_paths(),
        vec![PathBuf::from(path!("/root/zed/file_a.txt"))]
    );
}

#[gpui::test]
async fn test_remote_trash_restore(cx: &mut TestAppContext, server_cx: &mut TestAppContext) {
    let fs = FakeFs::new(server_cx.executor());

    fs.insert_tree(
        path!("/root"),
        json!({
            "zed": {
                "file_a.txt": "File A"
            }
        }),
    )
    .await;

    let (project, _headless_project) = init_test(&fs, cx, server_cx).await;
    let (worktree, _path) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/root/zed"), true, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    let worktree_id = worktree.update(cx, |worktree, _cx| worktree.id());
    let entry_id = worktree.update(cx, |worktree, _cx| {
        worktree.entry_for_path(rel_path("file_a.txt")).unwrap().id
    });

    let trash_id = project
        .update(cx, |project, cx| project.trash_entry(entry_id, cx))
        .unwrap()
        .await
        .unwrap();
    cx.run_until_parked();

    worktree.update(cx, |worktree, _cx| {
        assert!(worktree.entry_for_path(rel_path("file_a.txt")).is_none());
    });

    project
        .update(cx, |project, cx| {
            project.restore_entry(worktree_id, trash_id, cx)
        })
        .await
        .unwrap();
    cx.run_until_parked();

    worktree.update(cx, |worktree, _cx| {
        assert!(worktree.entry_for_path(rel_path("file_a.txt")).is_some());
    });
}

#[cfg(feature = "rust-tools")]
mod rust_tools_runtime_tests {
    use super::*;
    use project::{
        cargo_workspace_store::{CargoMetadataRequest, CargoMetadataRunner, CargoWorkspaceStore},
        rust_test_provider::{
            RustTestDiscoveryCapture, RustTestDiscoveryRequest, RustTestDiscoveryRunner,
            RustTestListingCapture, RustTestListingMode, RustTestProviderStore, RustTestTargetKind,
        },
    };

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_rejects_structured_fetch_after_disconnect(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use project::structured_execution::{
            DiscoveryGeneration, StructuredProviderId, StructuredProviderSnapshot,
            StructuredProviderStatus,
        };
        let fs = FakeFs::new(server_cx.executor());
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let provider = StructuredProviderId("disconnect-fixture".into());
        headless.update(server_cx, |headless, cx| {
            headless.structured_execution_store.update(cx, |store, _| {
                let generation = store.state().project_generation();
                store
                    .state_mut()
                    .apply_discovery(
                        generation,
                        StructuredProviderSnapshot::discovery(
                            provider.clone(),
                            DiscoveryGeneration(1),
                            StructuredProviderStatus::Empty,
                            Vec::new(),
                        ),
                        None,
                    )
                    .expect("host provider fixture");
            });
        });
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store
            .update(cx, |store, cx| store.refresh_provider(provider.clone(), cx))
            .await
            .expect("initial structured snapshot");
        headless.update(server_cx, |headless, cx| {
            headless.structured_execution_store.update(cx, |store, _| {
                let generation = store.state().project_generation();
                store
                    .state_mut()
                    .apply_discovery(
                        generation,
                        StructuredProviderSnapshot::discovery(
                            provider.clone(),
                            DiscoveryGeneration(2),
                            StructuredProviderStatus::Empty,
                            Vec::new(),
                        ),
                        None,
                    )
                    .expect("new host generation");
            });
        });
        let request = store.update(cx, |store, cx| store.refresh_provider(provider.clone(), cx));
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_server_not_running(cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(1));
        match futures::future::select(request, deadline).await {
            futures::future::Either::Left((result, _)) => {
                let error = result.expect_err("disconnected structured fetch must fail");
                assert!(error.to_string().contains("disconnected"), "{error:#}");
            }
            futures::future::Either::Right(_) => {
                panic!("disconnected structured fetch did not terminate")
            }
        }
        store.read_with(cx, |store, _| {
            let snapshot = store
                .state()
                .provider(&provider)
                .expect("cached provider retained");
            assert_eq!(
                snapshot.discovery_generation,
                DiscoveryGeneration(1),
                "a pending fetch must not replace cached state after disconnect"
            );
            assert_eq!(snapshot.status, StructuredProviderStatus::Disconnected);
        });
        let error = store
            .update(cx, |store, cx| store.refresh_provider(provider, cx))
            .await
            .expect_err("disconnected store must reject new fetches");
        assert!(error.to_string().contains("disconnected"), "{error:#}");
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_structured_reconnect_invalidates_pending(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use project::structured_execution::{
            DiscoveryGeneration, StructuredProviderId, StructuredProviderSnapshot,
            StructuredProviderStatus,
        };

        let fs = FakeFs::new(server_cx.executor());
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let provider = StructuredProviderId("reconnect-fixture".into());
        let publish_host_generation = |generation, server_cx: &mut TestAppContext| {
            headless.update(server_cx, |headless, cx| {
                headless.structured_execution_store.update(cx, |store, _| {
                    let project_generation = store.state().project_generation();
                    store
                        .state_mut()
                        .apply_discovery(
                            project_generation,
                            StructuredProviderSnapshot::discovery(
                                provider.clone(),
                                DiscoveryGeneration(generation),
                                StructuredProviderStatus::Empty,
                                Vec::new(),
                            ),
                            None,
                        )
                        .expect("host provider fixture");
                });
            });
        };
        publish_host_generation(1, server_cx);
        let store = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        store
            .update(cx, |store, cx| store.refresh_provider(provider.clone(), cx))
            .await
            .expect("initial structured snapshot");
        publish_host_generation(2, server_cx);

        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .update(cx, |store, _| store.gate_next_snapshot_for_test())
        });

        let pending = store.update(cx, |store, cx| store.refresh_provider(provider.clone(), cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "registered headless structured handler must receive the delayed request"
        );
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(1, cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => {
                result.expect_err("connection interruption must invalidate pending results");
            }
            futures::future::Either::Right(_) => {
                panic!("structured request survived connection interruption")
            }
        }
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            futures::pin_mut!(recovery);
            assert!(
                matches!(
                    futures::future::select(recovery, deadline).await,
                    futures::future::Either::Left(_)
                ),
                "mock connection must recover"
            );
        }
        store.read_with(cx, |store, _| {
            let snapshot = store
                .state()
                .provider(&provider)
                .expect("cached provider retained");
            assert_eq!(snapshot.discovery_generation, DiscoveryGeneration(1));
            assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
        });

        release
            .send(())
            .expect("delayed headless handler must remain active");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(completed, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "registered headless handler must finish its late snapshot"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
        store.read_with(cx, |store, _| {
            let snapshot = store
                .state()
                .provider(&provider)
                .expect("cached provider retained after late host response");
            assert_eq!(snapshot.discovery_generation, DiscoveryGeneration(1));
            assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
        });

        publish_host_generation(3, server_cx);
        let snapshot = store
            .update(cx, |store, cx| store.refresh_provider(provider.clone(), cx))
            .await
            .expect("fresh results succeed after reconnect");
        assert_eq!(snapshot.discovery_generation, DiscoveryGeneration(3));
        assert_eq!(snapshot.status, StructuredProviderStatus::Empty);
    }

    #[gpui::test(iterations = 10)]
    async fn rust_tools_headless_runtime_interpreted_coverage_hides_private_source(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        let coverage_artifact = serde_json::json!({
            "type": "llvm.coverage.json.export",
            "version": "2.0.1",
            "cargo_llvm_cov": {"version": "0.9.0", "manifest_path": "/coverage-artifact/Cargo.toml"},
            "data": [{"files": [
                {"filename": "/coverage-artifact/source.rs", "segments": [[1, 1, 1, true, true, false]], "summary": {"lines": {"count": 1, "covered": 1}}},
                {"filename": "/coverage-artifact/.env", "segments": [[1, 1, 1, true, true, false]], "summary": {"lines": {"count": 1, "covered": 1}}}
            ]}]
        });
        fs.insert_tree(
            path!("/coverage-artifact"),
            serde_json::json!({
                ".gitignore": ".env\n",
                "source.rs": "fn public() {}\n",
                ".env": "SECRET=private\n",
                "coverage.json": coverage_artifact.to_string(),
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });
        server_cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            )
        });

        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/coverage-artifact"), true, cx)
            })
            .await
            .expect("remote coverage artifact worktree");
        cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(project::WorktreeId::from_proto(worktree_id), cx)
                .expect("host coverage artifact worktree");
            assert!(
                host_worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path(".env"))
                    .is_some_and(|entry| entry.is_private),
                "the host fixture must mark the hidden source private"
            );
        });
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let response = client
            .request(proto::InterpretRustCoverageArtifact {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                artifact_path: Some(proto::ProjectPath {
                    worktree_id,
                    path: "coverage.json".into(),
                }),
                max_bytes: project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64,
            })
            .await
            .expect("registered host coverage interpretation");
        assert_eq!(response.files.len(), 1);
        assert_eq!(
            response.files[0].path,
            Some(proto::ProjectPath {
                worktree_id,
                path: "source.rs".into(),
            })
        );
        assert!(response.truncated);
        assert!(!format!("{response:?}").contains(".env"));
        let (started, release) = headless.update(server_cx, |headless, cx| {
            headless
                .rust_coverage_provider_store
                .update(cx, |store, _| store.gate_next_host_preparation_for_test())
        });
        let request = proto::InterpretRustCoverageArtifact {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            artifact_path: Some(proto::ProjectPath {
                worktree_id,
                path: "coverage.json".into(),
            }),
            max_bytes: project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64,
        };
        let pending = client.request(request.clone());
        let pending = cx.background_spawn(pending);
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "actual host interpretation completed preparation before being held"
        );
        for enabled in [false, true] {
            server_cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-coverage-experimental",
                    enabled,
                    cx,
                );
            });
            server_cx.run_until_parked();
        }
        assert!(project.read_with(cx, |_, cx| {
            use feature_flags::FeatureFlagAppExt as _;
            cx.has_flag::<feature_flags::RustCoverageExperimentalFeatureFlag>()
        }));
        release
            .send(())
            .expect("release revoked host interpretation");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => {
                assert!(
                    result.is_err(),
                    "host Off then On cannot rejuvenate held interpretation"
                );
            }
            futures::future::Either::Right(_) => {
                panic!("revoked host interpretation did not retire")
            }
        }
        let fresh = client
            .request(request)
            .await
            .expect("fresh host interpretation after local opt-in recovery");
        assert_eq!(fresh.files.len(), 1);
        assert!(fresh.truncated);
        assert!(!format!("{fresh:?}").contains(".env"));
        fs.pause_events();
        fs.insert_file(
            path!("/coverage-artifact/fresh-coverage.json"),
            coverage_artifact.to_string().into_bytes(),
        )
        .await;
        headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(project::WorktreeId::from_proto(worktree_id), cx)
                .expect("host coverage artifact worktree");
            assert!(
                host_worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("fresh-coverage.json"))
                    .is_none(),
                "the fresh artifact must require the explicit host refresh"
            );
        });
        let fresh_response = client
            .request(proto::InterpretRustCoverageArtifact {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                artifact_path: Some(proto::ProjectPath {
                    worktree_id,
                    path: "fresh-coverage.json".into(),
                }),
                max_bytes: project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64,
            })
            .await
            .expect("host refreshes a newly written coverage artifact before interpretation");
        assert_eq!(fresh_response.files.len(), 1);
        assert!(fresh_response.truncated);
        assert!(!format!("{fresh_response:?}").contains(".env"));
        fs.insert_file(
            path!("/coverage-artifact/provider-coverage.json"),
            coverage_artifact.to_string().into_bytes(),
        )
        .await;
        assert!(
            worktree.read_with(cx, |worktree, _| {
                worktree
                    .snapshot()
                    .entry_for_path(rel_path("provider-coverage.json"))
                    .is_none()
            }),
            "the client must not rely on a scanned artifact entry"
        );
        let path = project.read_with(cx, |project, cx| {
            project.find_project_path(path!("/coverage-artifact/provider-coverage.json"), cx)
        });
        let path = path.expect("visible worktree maps a newly created absolute report path");
        let provider = project.read_with(cx, |project, _| {
            project.rust_coverage_provider_store().clone()
        });
        let snapshot = provider
            .update(cx, |provider, cx| {
                provider.interpret_artifact(
                    path,
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .expect("production remote Project client receives refreshed host report");
        assert_eq!(snapshot.files.len(), 1);
        assert!(snapshot.truncated);
        assert!(!format!("{snapshot:?}").contains(".env"));
        {
            use cargo_ui::CargoActionDispatcher as _;
            use std::{cell::RefCell, process::ExitStatus, rc::Rc};

            cx.update(|cx| {
                theme_settings::init(theme::LoadThemes::JustBase, cx);
                editor::init(cx);
                language_tools::source_coverage_summary::init(cx);
                cargo_ui::init(cx);
            });

            struct CoverageTerminal {
                fs: Arc<FakeFs>,
                report: String,
                calls: Rc<RefCell<Vec<task::SpawnInTerminal>>>,
            }

            impl workspace::TerminalProvider for CoverageTerminal {
                fn spawn(
                    &self,
                    task: task::SpawnInTerminal,
                    _: &mut gpui::Window,
                    cx: &mut gpui::App,
                ) -> gpui::Task<Option<anyhow::Result<ExitStatus>>> {
                    self.calls.borrow_mut().push(task);
                    let fs = self.fs.clone();
                    let report = self.report.clone();
                    cx.background_spawn(async move {
                        fs.insert_tree(
                            path!("/coverage-artifact/target"),
                            serde_json::json!({"zed-coverage.json": report}),
                        )
                        .await;
                        Some(Ok(ExitStatus::default()))
                    })
                }
            }

            let (container, cx) = cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(project.clone(), window, cx)
            });
            let workspace = container.read_with(cx, |container, _| container.workspace().clone());
            let calls = Rc::new(RefCell::new(Vec::new()));
            workspace.update(cx, |workspace, _| {
                workspace.set_terminal_provider(CoverageTerminal {
                    fs: fs.clone(),
                    report: coverage_artifact.to_string(),
                    calls: calls.clone(),
                });
            });
            workspace.update_in(cx, |workspace, window, cx| {
                let mut dispatcher =
                    cargo_ui::WorkspaceCargoActionDispatcher::new(workspace, window, cx);
                dispatcher.schedule_task(
                    project::TaskSourceKind::Language {
                        name: "Cargo".into(),
                    },
                    task::TaskTemplate {
                        experimental_policy_version: task::EXPERIMENTAL_TASK_POLICY_VERSION,
                        label: "Run with Coverage".into(),
                        command: "cargo".into(),
                        args: vec![
                            "llvm-cov".into(),
                            "--json".into(),
                            "--output-path".into(),
                            "target/zed-coverage.json".into(),
                            "run".into(),
                        ],
                        cwd: Some("/coverage-artifact".into()),
                        save: task::SaveStrategy::None,
                        artifact: Some(task::TaskArtifact {
                            path: "target/zed-coverage.json".into(),
                            kind: task::TaskArtifactKind::Data,
                            max_bytes: 1024 * 1024,
                        }),
                        ..task::TaskTemplate::default()
                    },
                    task::TaskContext::default(),
                    false,
                );
            });
            cx.run_until_parked();
            assert_eq!(calls.borrow().len(), 1);
            let coverage_store =
                project.read_with(cx, |project, _| project.source_coverage_store().clone());
            coverage_store.read_with(cx, |store, _| {
                let selected = store
                    .selected_snapshot()
                    .expect("remote Task published coverage");
                assert_eq!(selected.files.len(), 1, "{selected:?}");
                assert!(selected.truncated);
                assert!(!format!("{selected:?}").contains(".env"));
            });
            cx.deactivate_window();
            container.update_in(cx, |_, window, _| window.remove_window());
        }
        fs.unpause_events_and_flush();
    }

    #[gpui::test(iterations = 10)]
    async fn rust_tools_headless_runtime_coverage_reconnect_invalidates_pending(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use gpui::BorrowAppContext;
        use project::source_coverage::{
            SourceCoverageFile, SourceCoveragePoint, SourceCoverageProviderId, SourceCoverageRange,
            SourceCoverageSnapshot, SourceCoverageStatus,
        };

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/coverage-fixture"),
            serde_json::json!({"main.rs": "fn main() {}"}),
        )
        .await;
        fs.insert_tree(
            path!("/coverage-hidden"),
            serde_json::json!({"hidden.rs": "fn hidden() {}"}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            language_tools::source_coverage_summary::init(cx);
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/coverage-fixture"), true, cx)
            })
            .await
            .expect("remote coverage worktree");
        cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let (hidden_worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/coverage-hidden"), false, cx)
            })
            .await
            .expect("remote invisible coverage worktree");
        let hidden_worktree_id = hidden_worktree.read_with(cx, |worktree, _| worktree.id());
        headless.read_with(server_cx, |headless, cx| {
            let store = headless.worktree_store.read(cx);
            assert!(store.worktree_for_id(hidden_worktree_id, cx).is_some());
            assert!(
                !store
                    .visible_worktrees(cx)
                    .any(|worktree| worktree.read(cx).id() == hidden_worktree_id)
            );
        });
        let buffer = project
            .update(cx, |project, cx| {
                project.open_buffer((worktree_id, rel_path("main.rs")), cx)
            })
            .await
            .expect("remote covered source");
        let cx = cx.add_empty_window();
        let workspace = cx.new_window_entity(|window, cx| {
            workspace::Workspace::test_new(project.clone(), window, cx)
        });
        let editor = cx.new_window_entity(|window, cx| {
            Editor::for_buffer(buffer, Some(project.clone()), window, cx)
        });
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_item_to_active_pane(Box::new(editor.clone()), None, true, window, cx);
        });
        let provider = SourceCoverageProviderId("remote-fixture".into());
        let publish_host_generation = |generation, server_cx: &mut TestAppContext| {
            headless.update(server_cx, |headless, cx| {
                headless.source_coverage_store.update(cx, |store, cx| {
                    store
                        .publish(
                            SourceCoverageSnapshot {
                                project_generation: 1,
                                provider_id: provider.clone(),
                                generation,
                                status: SourceCoverageStatus::Current,
                                files: [
                                    (worktree_id, "main.rs"),
                                    (hidden_worktree_id, "hidden.rs"),
                                ]
                                .into_iter()
                                .map(|(worktree_id, source)| SourceCoverageFile {
                                    path: ProjectPath {
                                        worktree_id,
                                        path: rel_path(source).into(),
                                    },
                                    ranges: vec![SourceCoverageRange {
                                        start: SourceCoveragePoint { line: 0, column: 0 },
                                        end: SourceCoveragePoint { line: 0, column: 2 },
                                        hit_count: generation,
                                    }],
                                    covered_lines: 1,
                                    uncovered_lines: 0,
                                    truncated: false,
                                })
                                .collect(),
                                truncated: false,
                                diagnostic: None,
                            },
                            cx,
                        )
                        .expect("host coverage generation");
                });
            });
        };
        publish_host_generation(1, server_cx);
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        let first = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("initial host coverage snapshot");
        assert_eq!(first.generation, 1);
        assert_eq!(first.files.len(), 1);
        assert!(
            first.truncated,
            "host must report that it filtered an invisible source"
        );
        assert_eq!(first.files[0].path.worktree_id, worktree_id);
        headless.read_with(server_cx, |headless, cx| {
            let hidden = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(hidden_worktree_id, cx)
                .expect("invisible worktree remains on the host during the response");
            assert!(
                hidden
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("hidden.rs"))
                    .is_some()
            );
        });
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0),
            "remote host coverage must reach the production Editor gutter"
        );
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("production remote coverage toolbar");
        assert!(label.contains("current"), "{label}");
        assert_eq!(navigation_count, 1);
        publish_host_generation(2, server_cx);

        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .source_coverage_store
                .update(cx, |store, _| store.gate_next_snapshot_for_test())
        });
        let pending = store.update(cx, |store, cx| store.snapshot(provider.clone(), cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "registered headless coverage handler must receive the delayed request"
        );
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(1, cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => {
                result.expect_err("connection interruption must invalidate pending coverage");
            }
            futures::future::Either::Right(_) => {
                panic!("source coverage request survived connection interruption")
            }
        }
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            futures::pin_mut!(recovery);
            assert!(
                matches!(
                    futures::future::select(recovery, deadline).await,
                    futures::future::Either::Left(_)
                ),
                "mock coverage connection must recover"
            );
        }
        store.read_with(cx, |store, _| {
            let cached = store.selected_snapshot().expect("retained coverage");
            assert_eq!(cached.generation, 1);
            assert_eq!(cached.status, SourceCoverageStatus::Stale);
        });
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0),
            "reconnected Editor must not retain stale coverage"
        );
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("retained remote coverage toolbar");
        assert!(label.contains("stale"), "{label}");
        assert_eq!(navigation_count, 0);

        release
            .send(())
            .expect("delayed headless coverage handler must remain active");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(completed, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "registered headless coverage handler must finish its late snapshot"
        );
        store.read_with(cx, |store, _| {
            let cached = store.selected_snapshot().expect("retained coverage");
            assert_eq!(cached.generation, 1);
            assert_eq!(cached.status, SourceCoverageStatus::Stale);
        });

        publish_host_generation(3, server_cx);
        let fresh = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("fresh host coverage after reconnect");
        assert_eq!(fresh.generation, 3);
        assert_eq!(fresh.status, SourceCoverageStatus::Current);
        assert_eq!(fresh.files[0].ranges[0].hit_count, 3);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0),
            "fresh remote host report must restore the Editor gutter"
        );
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("refreshed remote coverage toolbar");
        assert!(label.contains("current"), "{label}");
        assert_eq!(navigation_count, 1);

        let alternate = SourceCoverageProviderId("remote-alternate".into());
        let mut alternate_file = fresh.files[0].clone();
        alternate_file.ranges[0].hit_count = 0;
        alternate_file.covered_lines = 0;
        alternate_file.uncovered_lines = 1;
        headless.update(server_cx, |headless, cx| {
            headless.source_coverage_store.update(cx, |store, cx| {
                store
                    .publish(
                        SourceCoverageSnapshot {
                            project_generation: 1,
                            provider_id: alternate.clone(),
                            generation: 1,
                            status: SourceCoverageStatus::Current,
                            files: vec![alternate_file],
                            truncated: false,
                            diagnostic: None,
                        },
                        cx,
                    )
                    .expect("alternate host coverage provider");
            });
        });
        let alternate_snapshot = store
            .update(cx, |store, cx| store.snapshot(alternate.clone(), cx))
            .await
            .expect("alternate remote host coverage");
        assert_eq!(alternate_snapshot.provider_id, alternate);
        store
            .update(cx, |store, cx| store.select_provider(alternate.clone(), cx))
            .expect("select alternate remote coverage provider");
        cx.run_until_parked();
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("alternate remote coverage toolbar");
        assert!(
            label.contains("remote-alternate") && label.contains("current"),
            "{label}"
        );
        assert_eq!(navigation_count, 1);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 1)
        );
        store
            .update(cx, |store, cx| store.select_provider(provider.clone(), cx))
            .expect("restore original remote coverage provider");
        cx.run_until_parked();
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("restored remote coverage toolbar");
        assert!(
            label.contains("remote-fixture") && label.contains("current"),
            "{label}"
        );
        assert_eq!(navigation_count, 1);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0)
        );

        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/main.rs".into()]));
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("host coverage worktree");
            assert!(
                host_worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("main.rs"))
                    .expect("host coverage source entry")
                    .is_private
            );
        });
        let private_snapshot = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("privacy-filtered host coverage");
        assert_eq!(private_snapshot.status, SourceCoverageStatus::Stale);
        assert!(private_snapshot.files.is_empty());
        assert!(private_snapshot.truncated);
        cx.run_until_parked();
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("privacy-filtered remote coverage toolbar");
        assert!(
            label.contains("stale") && label.contains("truncated"),
            "{label}"
        );
        assert_eq!(navigation_count, 0);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None;
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        publish_host_generation(4, server_cx);
        let restored = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("restored host coverage source");
        assert_eq!(restored.generation, 4);
        assert_eq!(restored.files.len(), 1);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0)
        );

        publish_host_generation(5, server_cx);
        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .source_coverage_store
                .update(cx, |store, _| store.gate_next_snapshot_for_test())
        });
        let pending = store.update(cx, |store, cx| store.snapshot(provider.clone(), cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/main.rs".into()]));
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        headless.read_with(server_cx, |headless, cx| {
            let worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("host source worktree");
            assert!(
                worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("main.rs"))
                    .expect("host source entry")
                    .is_private
            );
        });
        release.send(()).expect("resume coverage response");
        completed.await.expect("host completed coverage response");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        let revoked = match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => {
                result.expect("coverage response after privacy change")
            }
            futures::future::Either::Right(_) => {
                panic!("coverage response did not finish after privacy change")
            }
        };
        assert!(revoked.files.is_empty());
        assert!(revoked.truncated);
        cx.run_until_parked();
        let (_, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("privacy-revoked remote toolbar");
        assert_eq!(navigation_count, 0);
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None;
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        publish_host_generation(6, server_cx);
        let fresh = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("fresh host coverage after privacy recovery");
        assert_eq!(fresh.files.len(), 1);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0)
        );

        publish_host_generation(7, server_cx);
        let (assembled, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .source_coverage_store
                .update(cx, |store, _| store.gate_next_assembled_response_for_test())
        });
        let pending = store.update(cx, |store, cx| store.snapshot(provider.clone(), cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/main.rs".into()]));
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        release
            .send(())
            .expect("release previously assembled report");
        completed.await.expect("host released assembled report");
        let late = pending.await.expect("late assembled coverage response");
        assert!(late.files.is_empty(), "late report exposed a private file");
        assert!(late.truncated);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        let (_, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("toolbar after delayed privacy revocation");
        assert_eq!(navigation_count, 0);
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None;
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        publish_host_generation(8, server_cx);
        let restored = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("fresh host report after late privacy recovery");
        assert_eq!(restored.files.len(), 1);

        cx.cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
        });
        server_cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-coverage-experimental",
                true,
                cx,
            );
        });
        cx.run_until_parked();
        server_cx.run_until_parked();
        let rust_provider = SourceCoverageProviderId("cargo-llvm-cov".into());
        let mut rust_snapshot = headless.read_with(server_cx, |headless, cx| {
            headless
                .source_coverage_store
                .read(cx)
                .cached_snapshots()
                .find(|snapshot| snapshot.provider_id == provider)
                .expect("actual generic host source snapshot")
                .clone()
        });
        rust_snapshot.provider_id = rust_provider.clone();
        let publish_rust_generation = |generation, server_cx: &mut TestAppContext| {
            let mut snapshot = rust_snapshot.clone();
            snapshot.generation = generation;
            headless.update(server_cx, |headless, cx| {
                headless.source_coverage_store.update(cx, |store, cx| {
                    store
                        .publish(snapshot, cx)
                        .expect("Rust host snapshot with explicit local opt-in")
                });
            });
        };
        publish_rust_generation(1, server_cx);
        let original_rust = store
            .update(cx, |store, cx| store.snapshot(rust_provider.clone(), cx))
            .await
            .expect("initial actual Rust host snapshot");
        assert_eq!(original_rust.generation, 1);
        assert_eq!(original_rust.status, SourceCoverageStatus::Current);
        assert_eq!(original_rust.files.len(), 1);
        publish_rust_generation(2, server_cx);
        let (assembled, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .source_coverage_store
                .update(cx, |store, _| store.gate_next_assembled_response_for_test())
        });
        let pending = store.update(cx, |store, cx| store.snapshot(rust_provider.clone(), cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        for enabled in [false, true] {
            server_cx.update(|cx| {
                project::task_store::set_experimental_feature_for_test(
                    "rust-coverage-experimental",
                    enabled,
                    cx,
                );
            });
            server_cx.run_until_parked();
            if !enabled {
                let generic = store
                    .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
                    .await
                    .expect("ordinary coverage RPC remains available while Rust opt-in is Off");
                assert_eq!(generic.provider_id, provider);
                assert_eq!(generic.generation, 8);
                assert_eq!(generic.status, SourceCoverageStatus::Current);
                assert_eq!(generic.files.len(), 1);
            }
        }
        assert!(project.read_with(cx, |_, cx| {
            use feature_flags::FeatureFlagAppExt as _;
            cx.has_flag::<feature_flags::RustCoverageExperimentalFeatureFlag>()
        }));
        release
            .send(())
            .expect("release host-revoked assembled report");
        completed
            .await
            .expect("host reached final assembled-response check");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => {
                assert!(
                    result.is_err(),
                    "host Off then On cannot return old Current coverage"
                );
            }
            futures::future::Either::Right(_) => {
                panic!("host-revoked assembled response did not retire")
            }
        }
        assert_eq!(
            store.read_with(cx, |store, _| store
                .cached_snapshots()
                .find(|snapshot| snapshot.provider_id == rust_provider)
                .expect("previous client result retained")
                .generation),
            1
        );
        publish_rust_generation(3, server_cx);
        let fresh = store
            .update(cx, |store, cx| store.snapshot(rust_provider.clone(), cx))
            .await
            .expect("fresh host coverage after local opt-in recovery");
        assert_eq!(fresh.generation, 3);
        assert_eq!(fresh.status, SourceCoverageStatus::Current);
        assert_eq!(fresh.files.len(), 1);
        publish_host_generation(9, server_cx);
        let (assembled, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .source_coverage_store
                .update(cx, |store, _| store.gate_next_assembled_response_for_test())
        });
        let pending = store.update(cx, |store, cx| store.snapshot(provider.clone(), cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx.cx.update(|cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.restrict(
                    worktree_store.downgrade(),
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        release
            .send(())
            .expect("release assembled report after client restriction");
        completed.await.expect("host released unrestricted report");
        let restricted = pending.await.expect("client-filtered coverage response");
        assert_eq!(restricted.status, SourceCoverageStatus::Restricted);
        assert!(restricted.files.is_empty());
        assert!(restricted.truncated);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0)
        );
        let (_, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("toolbar after client trust revocation");
        assert_eq!(navigation_count, 0);
        cx.cx.update(|cx| {
            let worktree_store = project.read(cx).worktree_store();
            let trusted = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust store");
            trusted.update(cx, |trusted, cx| {
                trusted.trust(
                    &worktree_store,
                    [project::trusted_worktrees::PathTrust::Worktree(worktree_id)]
                        .into_iter()
                        .collect(),
                    cx,
                );
            });
        });
        publish_host_generation(10, server_cx);
        let restored = store
            .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
            .await
            .expect("fresh host report after client trust recovery");
        assert_eq!(restored.files.len(), 1);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 0)
        );

        for (status, expected_label, expected_links, expected_gutters) in [
            (SourceCoverageStatus::Partial, "partial", 1, (1, 0)),
            (SourceCoverageStatus::Loading, "loading", 0, (0, 0)),
            (SourceCoverageStatus::Empty, "empty", 0, (0, 0)),
            (SourceCoverageStatus::Stale, "stale", 0, (0, 0)),
            (SourceCoverageStatus::Error, "error", 0, (0, 0)),
            (SourceCoverageStatus::Restricted, "restricted", 0, (0, 0)),
            (
                SourceCoverageStatus::Disconnected,
                "disconnected",
                0,
                (0, 0),
            ),
            (SourceCoverageStatus::Mismatch, "host mismatch", 0, (0, 0)),
            (SourceCoverageStatus::Cancelled, "cancelled", 0, (0, 0)),
        ] {
            headless.update(server_cx, |headless, cx| {
                headless.source_coverage_store.update(cx, |store, cx| {
                    store
                        .mark_provider_status(provider.clone(), status, None, cx)
                        .expect("host coverage status transition");
                });
            });
            let updated = store
                .update(cx, |store, cx| store.snapshot(provider.clone(), cx))
                .await
                .expect("remote coverage status transition");
            assert_eq!(updated.status, status);
            cx.run_until_parked();
            let (label, navigation_count) = cx
                .cx
                .update(|cx| {
                    language_tools::source_coverage_summary::toolbar_snapshot_for_test(
                        &workspace, cx,
                    )
                })
                .expect("coverage toolbar after host status transition");
            assert!(label.contains(expected_label), "{label}");
            assert_eq!(navigation_count, expected_links);
            assert_eq!(
                editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
                expected_gutters
            );
        }

        project.update(cx, |project, cx| project.close(cx));
        store.read_with(cx, |store, _| {
            let cached = store
                .selected_snapshot()
                .expect("retained coverage on close");
            assert_eq!(cached.generation, 10);
            assert_eq!(cached.status, SourceCoverageStatus::Disconnected);
        });
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0),
            "closed Project must clear the remote Editor gutter"
        );
        let (label, navigation_count) = cx
            .cx
            .update(|cx| {
                language_tools::source_coverage_summary::toolbar_snapshot_for_test(&workspace, cx)
            })
            .expect("closed remote coverage toolbar");
        assert!(label.contains("disconnected"), "{label}");
        assert_eq!(navigation_count, 0);
        let error = store
            .update(cx, |store, cx| store.snapshot(provider, cx))
            .await
            .expect_err("closed project cannot request coverage");
        assert!(error.to_string().contains("disconnected"), "{error:#}");
    }

    struct MetadataRunner(Arc<std::sync::Mutex<Vec<CargoMetadataRequest>>>);

    #[async_trait::async_trait]
    impl CargoMetadataRunner for MetadataRunner {
        async fn run(&self, request: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
            self.0.lock().expect("runner request lock").push(request);
            Ok(
                include_bytes!("../../project/test_data/cargo_workspace/standalone-v1.json")
                    .to_vec(),
            )
        }
    }

    struct DiscoveryRunner(Arc<std::sync::Mutex<Vec<RustTestDiscoveryRequest>>>);

    #[cfg(unix)]
    struct ProcessDiagnosticRunner {
        executor: gpui::BackgroundExecutor,
        directory: PathBuf,
        environment: HashMap<String, String>,
    }

    #[cfg(unix)]
    impl RustTestDiscoveryRunner for ProcessDiagnosticRunner {
        fn discover(
            &self,
            mut request: RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
            request.environment = Some(self.environment.clone());
            request.working_directory = Some(self.directory.clone());
            request.manifest_path = Some(self.directory.join("Cargo.toml"));
            RustTestProviderStore::process_discovery_for_test(request, self.executor.clone())
        }
    }

    #[cfg(unix)]
    struct OpaqueDiagnosticRunner(String);

    #[cfg(unix)]
    impl RustTestDiscoveryRunner for OpaqueDiagnosticRunner {
        fn discover(
            &self,
            _: RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
            let message = self.0.clone();
            Box::pin(async move { anyhow::bail!(message) })
        }
    }

    #[cfg(unix)]
    fn await_diagnostic_process<F: std::future::Future>(
        executors: &[gpui::BackgroundExecutor],
        future: F,
    ) -> F::Output {
        futures::pin_mut!(future);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let waker = futures::task::noop_waker();
        loop {
            // Real child I/O must complete without advancing the discovery's GPUI timeout.
            for executor in executors {
                while executor.tick() {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "diagnostic fixture deadline"
                    );
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "diagnostic fixture deadline"
            );
            if let std::task::Poll::Ready(result) = future
                .as_mut()
                .poll(&mut std::task::Context::from_waker(&waker))
            {
                return result;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[gpui::test]
    async fn rust_tools_headless_runtime_preserves_admitted_worktree_scope(
        cx: &mut TestAppContext,
        bridge_cx: &mut TestAppContext,
        owner_cx: &mut TestAppContext,
    ) {
        use futures::FutureExt as _;
        use rpc::{AnyProtoClient, ProtoClient, ProtoMessageHandlerSet};

        #[derive(Default, Debug)]
        struct ScopeObservations {
            cargo_scopes: Vec<Vec<u64>>,
            discovery_scopes: Vec<Vec<u64>>,
            metadata_directories: Vec<PathBuf>,
            compiler_directories: Vec<PathBuf>,
        }

        struct ScopedMetadataRunner(Arc<std::sync::Mutex<ScopeObservations>>);

        #[async_trait::async_trait]
        impl CargoMetadataRunner for ScopedMetadataRunner {
            async fn run(&self, request: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
                let directory = request.working_directory;
                anyhow::ensure!(
                    directory.as_path() == std::path::Path::new(path!("/standalone"))
                        || directory.as_path() == std::path::Path::new(path!("/excluded")),
                    "unexpected scope fixture metadata directory: {directory:?}"
                );
                self.0
                    .lock()
                    .expect("metadata scopes")
                    .metadata_directories
                    .push(directory.clone());
                let metadata =
                    include_str!("../../project/test_data/cargo_workspace/standalone-v1.json");
                Ok(
                    if directory.as_path() == std::path::Path::new(path!("/excluded")) {
                        metadata.replace("standalone", "excluded").into_bytes()
                    } else {
                        metadata.as_bytes().to_vec()
                    },
                )
            }
        }

        struct ScopedCompilerProbe(Arc<std::sync::Mutex<ScopeObservations>>);

        #[async_trait::async_trait]
        impl project::cargo_workspace_store::CargoConfigurationProbe for ScopedCompilerProbe {
            async fn run(
                &self,
                request: project::cargo_workspace_store::CargoConfigurationProbeRequest,
            ) -> anyhow::Result<Vec<u8>> {
                self.0
                    .lock()
                    .expect("compiler scopes")
                    .compiler_directories
                    .push(request.working_directory);
                Ok(b"rustc 1.95.0\nbinary: rustc\ncommit-hash: fixture-only\ncommit-date: unknown\nhost: aarch64-apple-darwin\nrelease: 1.95.0\nLLVM version: 21.1.0\n".to_vec())
            }
        }

        struct ScopedDiscoveryRunner(Arc<std::sync::Mutex<Vec<RustTestDiscoveryRequest>>>);

        impl RustTestDiscoveryRunner for ScopedDiscoveryRunner {
            fn discover(
                &self,
                request: RustTestDiscoveryRequest,
            ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>>
            {
                let directory = request
                    .working_directory
                    .clone()
                    .expect("scope fixture working directory");
                let capture = DiscoveryRunner(self.0.clone()).discover(request);
                async move {
                    let mut capture = capture.await?;
                    if directory.as_path() == std::path::Path::new(path!("/excluded")) {
                        capture.cargo_messages =
                            capture.cargo_messages.replace("standalone", "excluded");
                        for listing in &mut capture.listings {
                            listing.package_id =
                                listing.package_id.replace("standalone", "excluded");
                            listing.target_name = "excluded".to_string();
                        }
                    } else {
                        anyhow::ensure!(
                            directory.as_path() == std::path::Path::new(path!("/standalone")),
                            "unexpected discovery fixture root"
                        );
                    }
                    Ok(capture)
                }
                .boxed()
            }
        }

        struct ObservedScopeClient {
            client: AnyProtoClient,
            observations: Arc<std::sync::Mutex<ScopeObservations>>,
            handlers: parking_lot::Mutex<ProtoMessageHandlerSet>,
        }

        impl ProtoClient for ObservedScopeClient {
            fn request(
                &self,
                envelope: proto::Envelope,
                request_type: &'static str,
            ) -> futures::future::BoxFuture<'static, anyhow::Result<proto::Envelope>> {
                let client = self.client.clone();
                let original_sender = envelope.original_sender_id;
                let observations = self.observations.clone();
                macro_rules! forward {
                    ($message:ty) => {{
                        async move {
                            let request =
                                <$message as proto::EnvelopedMessage>::from_envelope(envelope)
                                    .expect("scope observer typed request");
                            let response = match original_sender {
                                Some(peer) => client.forward_request(peer, request).await?,
                                None => client.request(request).await?,
                            };
                            Ok(proto::EnvelopedMessage::into_envelope(
                                response, 0, None, None,
                            ))
                        }
                        .boxed()
                    }};
                    ($message:ty, $field:ident) => {{
                        async move {
                            let request =
                                <$message as proto::EnvelopedMessage>::from_envelope(envelope)
                                    .expect("scope observer typed request");
                            observations
                                .lock()
                                .expect("upstream scopes")
                                .$field
                                .push(request.worktree_ids.clone());
                            let response = match original_sender {
                                Some(peer) => client.forward_request(peer, request).await?,
                                None => client.request(request).await?,
                            };
                            Ok(proto::EnvelopedMessage::into_envelope(
                                response, 0, None, None,
                            ))
                        }
                        .boxed()
                    }};
                }
                match envelope.payload.as_ref() {
                    Some(proto::envelope::Payload::GetCargoWorkspace(_)) => {
                        forward!(proto::GetCargoWorkspace, cargo_scopes)
                    }
                    Some(proto::envelope::Payload::GetRustTestDiscovery(_)) => {
                        forward!(proto::GetRustTestDiscovery, discovery_scopes)
                    }
                    Some(proto::envelope::Payload::CancelCargoWorkspace(_)) => {
                        forward!(proto::CancelCargoWorkspace)
                    }
                    Some(proto::envelope::Payload::CancelRustTestDiscovery(_)) => {
                        forward!(proto::CancelRustTestDiscovery)
                    }
                    Some(proto::envelope::Payload::GetStructuredExecutionSnapshot(_)) => {
                        forward!(proto::GetStructuredExecutionSnapshot)
                    }
                    Some(proto::envelope::Payload::GetStructuredExecutionEvents(_)) => {
                        forward!(proto::GetStructuredExecutionEvents)
                    }
                    _ => async move {
                        anyhow::bail!("unexpected scope observer request: {request_type}")
                    }
                    .boxed(),
                }
            }

            fn send(&self, _: proto::Envelope, message_type: &'static str) -> anyhow::Result<()> {
                anyhow::bail!("unexpected scope observer send: {message_type}")
            }
            fn send_response(
                &self,
                _: proto::Envelope,
                message_type: &'static str,
            ) -> anyhow::Result<()> {
                anyhow::bail!("unexpected scope observer response: {message_type}")
            }
            fn message_handler_set(&self) -> &parking_lot::Mutex<ProtoMessageHandlerSet> {
                &self.handlers
            }
            fn is_via_collab(&self) -> bool {
                false
            }
            fn has_wsl_interop(&self) -> bool {
                false
            }
        }

        let observations = Arc::new(std::sync::Mutex::new(ScopeObservations::default()));
        let discovery_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let bridge_fs = FakeFs::new(bridge_cx.executor());
        let (project, bridge) = init_test(&bridge_fs, cx, bridge_cx).await;
        let owner_fs = FakeFs::new(owner_cx.executor());
        for name in ["standalone", "excluded"] {
            owner_fs.insert_tree(
                &PathBuf::from(path!("/")).join(name),
                json!({
                    "Cargo.toml": format!("[package]\nname = \"{name}\"\nversion = \"1.0.0\"\n"),
                    "src": {"main.rs": "fn main() {}"},
                }),
            ).await;
        }
        let (upstream_project, owner) = init_test(&owner_fs, bridge_cx, owner_cx).await;
        owner.update(owner_cx, |owner, cx| {
            owner.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runners(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    Arc::new(ScopedMetadataRunner(observations.clone())),
                    Arc::new(ScopedCompilerProbe(observations.clone())),
                );
                store.shared(proto::REMOTE_SERVER_PROJECT_ID, owner.session.clone(), cx);
            });
            owner.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    owner.cargo_workspace_store.clone(),
                    owner.structured_execution_store.clone(),
                    Arc::new(ScopedDiscoveryRunner(discovery_requests.clone())),
                    cx,
                );
            });
        });
        let mut worktree_ids = Vec::new();
        for root in [path!("/standalone"), path!("/excluded")] {
            let (worktree, _) = upstream_project
                .update(bridge_cx, |project, cx| {
                    project.find_or_create_worktree(root, true, cx)
                })
                .await
                .expect("scope fixture worktree");
            let worktree_id = worktree.read_with(bridge_cx, |worktree, _| worktree.id());
            let scan = owner.read_with(owner_cx, |owner, cx| {
                owner
                    .worktree_store
                    .read(cx)
                    .worktree_for_id(worktree_id, cx)
                    .expect("owner scoped worktree")
                    .read(cx)
                    .as_local()
                    .expect("owner scan")
                    .scan_complete()
            });
            scan.await;
            worktree_ids.push(worktree_id);
        }
        let admitted = *worktree_ids.first().expect("admitted fixture worktree");
        let excluded = *worktree_ids.get(1).expect("excluded fixture worktree");
        let (upstream, worktrees) = upstream_project.read_with(bridge_cx, |project, cx| {
            (
                project
                    .remote_client()
                    .expect("scope upstream client")
                    .read(cx)
                    .proto_client(),
                project.worktree_store(),
            )
        });
        let observed = AnyProtoClient::new(Arc::new(ObservedScopeClient {
            client: upstream,
            observations: observations.clone(),
            handlers: Default::default(),
        }));
        bridge.update(bridge_cx, |bridge, cx| {
            bridge.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    worktrees.clone(),
                    observed.clone(),
                    cx,
                );
            });
            bridge.structured_execution_store.update(cx, |store, _| {
                *store = project::structured_execution::StructuredExecutionStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    observed.clone(),
                    worktrees.clone(),
                );
            });
            bridge.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    observed.clone(),
                    worktrees.clone(),
                    bridge.structured_execution_store.clone(),
                    cx,
                );
            });
        });
        let control = bridge
            .update(bridge_cx, |bridge, cx| {
                bridge
                    .rust_test_provider_store
                    .update(cx, |store, cx| store.refresh(cx))
            })
            .await
            .expect("unscoped two-workspace control");
        assert_eq!(
            control.status,
            project::structured_execution::StructuredProviderStatus::Current
        );
        for id in [admitted, excluded] {
            assert!(
                discovery_requests
                    .lock()
                    .expect("control discovery")
                    .iter()
                    .any(|request| request.workspace_root.worktree_id == id)
            );
        }
        cx.run_until_parked();
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
        *observations.lock().expect("reset scopes") = ScopeObservations::default();
        discovery_requests.lock().expect("reset discovery").clear();
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("scope outer client")
                .read(cx)
                .proto_client()
        });
        let cargo = client
            .request(proto::GetCargoWorkspace {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 810,
                worktree_ids: vec![admitted.to_proto()],
            })
            .await
            .expect("admitted Cargo request");
        let cargo_scopes = observations
            .lock()
            .expect("cargo scopes")
            .cargo_scopes
            .clone();
        let cargo_directories = observations
            .lock()
            .expect("cargo collection")
            .metadata_directories
            .clone();
        *observations.lock().expect("reset Rust scopes") = ScopeObservations::default();
        let discovery = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 811,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![admitted.to_proto()],
            })
            .await
            .expect("admitted Rust request");
        let page = client
            .request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: discovery.discovery_generation,
                page_start: 0,
                page_size: 128,
                worktree_ids: vec![admitted.to_proto()],
            })
            .await
            .expect("admitted structured export");
        {
            let observed = observations.lock().expect("admitted observations");
            let requests = discovery_requests.lock().expect("admitted discovery");
            assert!(
                cargo_scopes == vec![vec![admitted.to_proto()]]
                    && cargo_directories == vec![PathBuf::from(path!("/standalone"))]
                    && observed.discovery_scopes == vec![vec![admitted.to_proto()]]
                    && observed.metadata_directories == vec![PathBuf::from(path!("/standalone"))]
                    && observed.compiler_directories == vec![PathBuf::from(path!("/standalone"))]
                    && requests.len() == 1
                    && requests
                        .iter()
                        .all(|request| request.workspace_root.worktree_id == admitted),
                "scoped request widened collection: Cargo scopes={cargo_scopes:?}, Cargo directories={cargo_directories:?}, Rust={observed:?}, discovery={requests:?}"
            );
        }
        assert_eq!(cargo.workspaces.len(), 1);
        assert_eq!(
            discovery.status,
            proto::StructuredProviderStatus::Current as i32
        );
        assert!(page.nodes.iter().any(|node| node.label == "host_case"));
        assert!(
            page.nodes
                .iter()
                .filter_map(|node| node.path.as_ref())
                .all(|path| path.worktree_id == admitted.to_proto())
        );
        *observations.lock().expect("reset empty scopes") = ScopeObservations::default();
        discovery_requests
            .lock()
            .expect("reset empty discovery")
            .clear();
        client
            .request(proto::GetCargoWorkspace {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 812,
                worktree_ids: Vec::new(),
            })
            .await
            .expect("empty Cargo admission");
        let discovery = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 813,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: Vec::new(),
            })
            .await
            .expect("empty Rust admission");
        let page = client
            .request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: discovery.discovery_generation,
                page_start: 0,
                page_size: 128,
                worktree_ids: Vec::new(),
            })
            .await
            .expect("empty scoped export");
        let observed = observations.lock().expect("empty observations");
        assert_eq!(observed.cargo_scopes, vec![Vec::<u64>::new()]);
        assert_eq!(observed.discovery_scopes, vec![Vec::<u64>::new()]);
        assert!(observed.metadata_directories.is_empty());
        assert!(observed.compiler_directories.is_empty());
        assert!(
            discovery_requests
                .lock()
                .expect("empty discovery")
                .is_empty()
        );
        assert!(page.nodes.iter().all(|node| node.path.is_none()));
    }

    #[gpui::test]
    async fn rust_tools_headless_runtime_final_waiter_retires_two_hop_discovery(
        cx: &mut TestAppContext,
        bridge_cx: &mut TestAppContext,
        owner_cx: &mut TestAppContext,
    ) {
        use futures::FutureExt as _;
        use rpc::{AnyProtoClient, ProtoClient, ProtoMessageHandlerSet};

        #[derive(Default)]
        struct RetirementObservations {
            discovery_ids: Vec<u64>,
            cancellation_ids: Vec<u64>,
            cancellation_gate: Option<(
                futures::channel::oneshot::Sender<()>,
                futures::channel::oneshot::Receiver<()>,
            )>,
        }

        struct ObservedDiscovery {
            client: AnyProtoClient,
            observations: Arc<std::sync::Mutex<RetirementObservations>>,
            handlers: parking_lot::Mutex<ProtoMessageHandlerSet>,
        }

        impl ProtoClient for ObservedDiscovery {
            fn request(
                &self,
                envelope: proto::Envelope,
                request_type: &'static str,
            ) -> futures::future::BoxFuture<'static, anyhow::Result<proto::Envelope>> {
                let client = self.client.clone();
                let original_sender = envelope.original_sender_id;
                let observations = self.observations.clone();
                macro_rules! forward {
                    ($message:ty) => {{
                        async move {
                            let request =
                                <$message as proto::EnvelopedMessage>::from_envelope(envelope)
                                    .expect("observed typed discovery request");
                            let response = match original_sender {
                                Some(peer) => client.forward_request(peer, request).await?,
                                None => client.request(request).await?,
                            };
                            Ok(proto::EnvelopedMessage::into_envelope(
                                response, 0, None, None,
                            ))
                        }
                        .boxed()
                    }};
                }
                match envelope.payload.as_ref() {
                    Some(proto::envelope::Payload::GetRustTestDiscovery(_)) => async move {
                        let request = <proto::GetRustTestDiscovery as proto::EnvelopedMessage>::from_envelope(envelope)
                            .expect("observed Get discovery");
                        observations.lock().expect("discovery observations").discovery_ids.push(request.request_id);
                        let response = match original_sender {
                            Some(peer) => client.forward_request(peer, request).await?,
                            None => client.request(request).await?,
                        };
                        Ok(proto::EnvelopedMessage::into_envelope(response, 0, None, None))
                    }.boxed(),
                    Some(proto::envelope::Payload::CancelRustTestDiscovery(_)) => async move {
                        let request = <proto::CancelRustTestDiscovery as proto::EnvelopedMessage>::from_envelope(envelope)
                            .expect("observed Cancel discovery");
                        let gate = {
                            let mut observations = observations.lock().expect("cancellation observations");
                            observations.cancellation_ids.push(request.request_id);
                            observations.cancellation_gate.take()
                        };
                        if let Some((started, release)) = gate {
                            started.send(()).expect("cancel gate starts");
                            release.await.expect("cancel gate releases");
                        }
                        let response = match original_sender {
                            Some(peer) => client.forward_request(peer, request).await?,
                            None => client.request(request).await?,
                        };
                        Ok(proto::EnvelopedMessage::into_envelope(response, 0, None, None))
                    }.boxed(),
                    Some(proto::envelope::Payload::GetStructuredExecutionSnapshot(_)) => forward!(proto::GetStructuredExecutionSnapshot),
                    Some(proto::envelope::Payload::GetStructuredExecutionEvents(_)) => forward!(proto::GetStructuredExecutionEvents),
                    _ => async move { anyhow::bail!("unexpected discovery observer request: {request_type}") }.boxed(),
                }
            }

            fn send(&self, _: proto::Envelope, message_type: &'static str) -> anyhow::Result<()> {
                anyhow::bail!("unexpected discovery observer send: {message_type}")
            }
            fn send_response(
                &self,
                _: proto::Envelope,
                message_type: &'static str,
            ) -> anyhow::Result<()> {
                anyhow::bail!("unexpected discovery observer response: {message_type}")
            }
            fn message_handler_set(&self) -> &parking_lot::Mutex<ProtoMessageHandlerSet> {
                &self.handlers
            }
            fn is_via_collab(&self) -> bool {
                false
            }
            fn has_wsl_interop(&self) -> bool {
                false
            }
        }

        struct FixtureCompilerProbe;

        #[async_trait::async_trait]
        impl project::cargo_workspace_store::CargoConfigurationProbe for FixtureCompilerProbe {
            async fn run(
                &self,
                request: project::cargo_workspace_store::CargoConfigurationProbeRequest,
            ) -> anyhow::Result<Vec<u8>> {
                assert_eq!(
                    request.working_directory,
                    PathBuf::from(path!("/standalone"))
                );
                Ok(b"rustc 1.95.0\nbinary: rustc\ncommit-hash: fixture-only\ncommit-date: unknown\nhost: aarch64-apple-darwin\nrelease: 1.95.0\nLLVM version: 21.1.0\n".to_vec())
            }
        }

        let bridge_fs = FakeFs::new(bridge_cx.executor());
        let (project, bridge) = init_test(&bridge_fs, cx, bridge_cx).await;
        let owner_fs = FakeFs::new(owner_cx.executor());
        owner_fs
            .insert_tree(
                path!("/standalone"),
                json!({
                    "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n",
                    "src": {"main.rs": "fn main() {}"},
                }),
            )
            .await;
        let (upstream_project, owner) = init_test(&owner_fs, bridge_cx, owner_cx).await;
        owner.update(owner_cx, |owner, cx| {
            owner.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runners(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    Arc::new(FixtureCompilerProbe),
                );
                store.shared(proto::REMOTE_SERVER_PROJECT_ID, owner.session.clone(), cx);
            });
            owner.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    owner.cargo_workspace_store.clone(),
                    owner.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
            });
        });
        let (worktree, _) = upstream_project
            .update(bridge_cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("two-hop upstream worktree");
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
        let worktree_id = worktree.read_with(bridge_cx, |worktree, _| worktree.id());
        let scan = owner.read_with(owner_cx, |owner, cx| {
            owner
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("owner worktree")
                .read(cx)
                .as_local()
                .expect("owner scan")
                .scan_complete()
        });
        scan.await;
        owner
            .update(owner_cx, |owner, cx| {
                owner
                    .cargo_workspace_store
                    .update(cx, |store, cx| store.refresh(cx))
            })
            .await
            .expect("settled owner metadata");
        let (upstream, worktrees) = upstream_project.read_with(bridge_cx, |project, cx| {
            (
                project
                    .remote_client()
                    .expect("upstream client")
                    .read(cx)
                    .proto_client(),
                project.worktree_store(),
            )
        });
        let observations = Arc::new(std::sync::Mutex::new(RetirementObservations::default()));
        let observed_upstream = AnyProtoClient::new(Arc::new(ObservedDiscovery {
            client: upstream.clone(),
            observations: observations.clone(),
            handlers: Default::default(),
        }));
        bridge.update(bridge_cx, |bridge, cx| {
            bridge.structured_execution_store.update(cx, |store, _| {
                *store = project::structured_execution::StructuredExecutionStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    observed_upstream.clone(),
                    worktrees.clone(),
                );
            });
            bridge.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    observed_upstream.clone(),
                    worktrees.clone(),
                    bridge.structured_execution_store.clone(),
                    cx,
                );
            });
        });
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("outer client")
                .read(cx)
                .proto_client()
        });
        let warmed = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 1,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id.to_proto()],
            })
            .await
            .expect("warm registered two-hop discovery");
        assert_eq!(
            warmed.status,
            proto::StructuredProviderStatus::Current as i32
        );
        cx.run_until_parked();
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        owner.update(owner_cx, |owner, cx| {
            owner.rust_test_provider_store.update(cx, |store, _| {
                store
                    .set_discovery_runner_for_test(Arc::new(GatedDiscoveryRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }))
                    .expect("gate the upstream owner runner");
            });
        });
        let first_peer = proto::PeerId { owner_id: 9, id: 1 };
        let last_peer = proto::PeerId { owner_id: 9, id: 2 };
        let request = proto::GetRustTestDiscovery {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 701,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        };
        let first_discovery = client.request(super::peer_requests::FromPeer {
            peer: first_peer,
            payload: request.clone(),
        });
        let cancellation = async {
            started_receiver.await.expect("upstream runner starts");
            let last_discovery = cx.background_executor.spawn({
                let client = client.clone();
                async move {
                    client
                        .request(super::peer_requests::FromPeer {
                            peer: last_peer,
                            payload: request,
                        })
                        .await
                }
            });
            for attempt in 0..100 {
                cx.run_until_parked();
                bridge_cx.run_until_parked();
                owner_cx.run_until_parked();
                if bridge.read_with(bridge_cx, |bridge, cx| {
                    bridge
                        .rust_test_provider_store
                        .read(cx)
                        .active_remote_discovery_request_count_for_test()
                }) == 2
                {
                    break;
                }
                assert!(attempt < 99, "both outer waiters must be admitted");
                cx.background_executor
                    .timer(std::time::Duration::from_millis(1))
                    .await;
            }
            assert_eq!(
                owner.read_with(owner_cx, |owner, cx| {
                    owner
                        .rust_test_provider_store
                        .read(cx)
                        .active_remote_discovery_request_count_for_test()
                }),
                1,
                "same-scope waiters share one upstream owner collection"
            );
            client
                .request(super::peer_requests::FromPeer {
                    peer: first_peer,
                    payload: proto::CancelRustTestDiscovery {
                        project_id: proto::REMOTE_SERVER_PROJECT_ID,
                        request_id: 701,
                    },
                })
                .await
                .expect("first waiter cancellation");
            cx.run_until_parked();
            bridge_cx.run_until_parked();
            owner_cx.run_until_parked();
            assert!(
                !release.is_canceled(),
                "first cancellation preserves the remaining waiter"
            );
            client
                .request(super::peer_requests::FromPeer {
                    peer: last_peer,
                    payload: proto::CancelRustTestDiscovery {
                        project_id: proto::REMOTE_SERVER_PROJECT_ID,
                        request_id: 701,
                    },
                })
                .await
                .expect("final waiter cancellation");
            let error = last_discovery
                .await
                .expect_err("final outer waiter is cancelled");
            assert!(error.to_string().contains("cancelled"), "{error:#}");
        };
        let (result, ()) = futures::join!(first_discovery, cancellation);
        let error = result.expect_err("first outer waiter is cancelled");
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        cx.run_until_parked();
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
        for _ in 0..100 {
            cx.run_until_parked();
            bridge_cx.run_until_parked();
            owner_cx.run_until_parked();
            if release.is_canceled() {
                break;
            }
            cx.background_executor
                .timer(std::time::Duration::from_millis(1))
                .await;
        }
        assert!(
            release.is_canceled(),
            "final waiter must drop the upstream owner's discovery runner within the bound"
        );
        assert_eq!(
            owner.read_with(owner_cx, |owner, cx| {
                owner
                    .rust_test_provider_store
                    .read(cx)
                    .active_remote_discovery_request_count_for_test()
            }),
            0,
            "upstream request identities must retire"
        );
        assert!(
            bridge.read_with(bridge_cx, |bridge, cx| {
                bridge
                    .rust_test_provider_store
                    .read(cx)
                    .discovery_idle_for_test()
            }),
            "the bridge must retire its pending remote identity"
        );

        let provider = bridge.read_with(bridge_cx, |bridge, _| {
            bridge.rust_test_provider_store.clone()
        });
        {
            let mut observed = observations.lock().expect("retired first collection");
            assert_eq!(
                observed.cancellation_ids.len(),
                1,
                "only the pending Get is cancelled"
            );
            observed.discovery_ids.clear();
            observed.cancellation_ids.clear();
        }
        let (owner_started, owner_started_receiver) = futures::channel::oneshot::channel();
        let (owner_release, owner_release_receiver) = futures::channel::oneshot::channel();
        owner.update(owner_cx, |owner, cx| {
            owner.rust_test_provider_store.update(cx, |store, _| {
                store
                    .set_discovery_runner_for_test(Arc::new(GatedDiscoveryRunner {
                        started: std::sync::Mutex::new(Some(owner_started)),
                        release: std::sync::Mutex::new(Some(owner_release_receiver)),
                    }))
                    .expect("second pending collection");
            });
        });
        let (cancel_started, cancel_started_receiver) = futures::channel::oneshot::channel();
        let (cancel_release, cancel_release_receiver) = futures::channel::oneshot::channel();
        observations
            .lock()
            .expect("arm cancellation gate")
            .cancellation_gate = Some((cancel_started, cancel_release_receiver));
        let initial = provider.update(bridge_cx, |store, cx| store.refresh(cx));
        let replacements = async {
            owner_started_receiver
                .await
                .expect("second owner runner starts");
            let unissued = provider.update(bridge_cx, |store, cx| store.refresh(cx));
            cancel_started_receiver
                .await
                .expect("predecessor cancellation reaches gate");
            assert!(
                !owner_release.is_canceled(),
                "the upstream cancel has not been delivered yet"
            );
            drop(unissued);
            let recovery = provider.update(bridge_cx, |store, cx| store.refresh(cx));
            for _ in 0..10 {
                cx.run_until_parked();
                bridge_cx.run_until_parked();
                owner_cx.run_until_parked();
                cx.background_executor
                    .timer(std::time::Duration::from_millis(1))
                    .await;
            }
            {
                let observed = observations.lock().expect("held retirement ordering");
                assert_eq!(
                    observed.discovery_ids.len(),
                    1,
                    "replacement Get must wait for predecessor Ack"
                );
                assert_eq!(
                    observed.cancellation_ids, observed.discovery_ids,
                    "unissued replacement IDs must not acquire cancellation tombstones"
                );
            }
            owner.update(owner_cx, |owner, cx| {
                owner.rust_test_provider_store.update(cx, |store, _| {
                    store
                        .set_discovery_runner_for_test(Arc::new(DiscoveryRunner(Arc::new(
                            std::sync::Mutex::new(Vec::new()),
                        ))))
                        .expect("restore owner discovery runner");
                });
            });
            cancel_release
                .send(())
                .expect("release actual upstream cancellation");
            let snapshot = recovery
                .await
                .expect("fresh replacement succeeds after retirement");
            assert_eq!(
                snapshot.status,
                project::structured_execution::StructuredProviderStatus::Current
            );
        };
        let (result, ()) = futures::join!(initial, replacements);
        let superseded = result.expect_err("initial refresh is superseded");
        assert_eq!(
            superseded
                .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
                .map(|error| error.kind),
            Some(project::rust_test_provider::RustTestRemoteErrorKind::SupersededDiscovery),
            "{superseded:#}"
        );
        for _ in 0..100 {
            cx.run_until_parked();
            bridge_cx.run_until_parked();
            owner_cx.run_until_parked();
            if provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()) {
                break;
            }
            cx.background_executor
                .timer(std::time::Duration::from_millis(1))
                .await;
        }
        assert!(
            owner_release.is_canceled(),
            "predecessor runner is retired after its gate releases"
        );
        assert!(provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()));
        {
            let observed = observations.lock().expect("fresh replacement observation");
            assert_eq!(
                observed.discovery_ids.len(),
                2,
                "only pending original and final replacement were issued"
            );
            assert_eq!(observed.cancellation_ids.len(), 1);
            assert_eq!(
                observed.cancellation_ids.first(),
                observed.discovery_ids.first()
            );
        }

        {
            let mut observed = observations.lock().expect("reset unpolled observation");
            observed.discovery_ids.clear();
            observed.cancellation_ids.clear();
        }
        let unpolled = provider.update(bridge_cx, |store, cx| store.refresh(cx));
        drop(unpolled);
        for _ in 0..100 {
            cx.run_until_parked();
            bridge_cx.run_until_parked();
            owner_cx.run_until_parked();
            if provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()) {
                break;
            }
            cx.background_executor
                .timer(std::time::Duration::from_millis(1))
                .await;
        }
        assert!(provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()));
        {
            let observed = observations.lock().expect("unpolled refresh observation");
            assert!(
                observed.discovery_ids.is_empty(),
                "unpolled refresh never issues Get"
            );
            assert!(
                observed.cancellation_ids.is_empty(),
                "unpolled refresh never issues Cancel"
            );
        }

        let next_request_id =
            provider.read_with(bridge_cx, |store, _| store.generation().0.wrapping_add(1));
        upstream
            .request(proto::CancelRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: next_request_id,
            })
            .await
            .expect("fixture arranges an actual cancelled Get response");
        let error = provider
            .update(bridge_cx, |store, cx| store.refresh(cx))
            .await
            .expect_err("host returns the arranged cancelled response");
        assert!(
            error.chain().any(|cause| {
                cause
                    .to_string()
                    .contains("Rust test discovery request was cancelled")
            }),
            "{error:#}"
        );
        for _ in 0..100 {
            cx.run_until_parked();
            bridge_cx.run_until_parked();
            owner_cx.run_until_parked();
            if provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()) {
                break;
            }
            cx.background_executor
                .timer(std::time::Duration::from_millis(1))
                .await;
        }
        assert!(provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()));
        {
            let observed = observations
                .lock()
                .expect("completed failed Get observation");
            assert_eq!(observed.discovery_ids, [next_request_id]);
            assert!(
                observed.cancellation_ids.is_empty(),
                "a completed failed Get must not add a new cancellation tombstone"
            );
        }

        {
            let mut observed = observations.lock().expect("reset snapshot observation");
            observed.discovery_ids.clear();
            observed.cancellation_ids.clear();
        }
        let (snapshot_started, snapshot_release, snapshot_completed) =
            owner.update(owner_cx, |owner, cx| {
                owner
                    .structured_execution_store
                    .update(cx, |store, _| store.gate_next_snapshot_for_test())
            });
        let snapshot_refresh = provider.update(bridge_cx, |store, cx| store.refresh(cx));
        snapshot_started
            .await
            .expect("Get finishes before structured snapshot gate");
        drop(snapshot_refresh);
        for _ in 0..100 {
            cx.run_until_parked();
            bridge_cx.run_until_parked();
            owner_cx.run_until_parked();
            if provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()) {
                break;
            }
            cx.background_executor
                .timer(std::time::Duration::from_millis(1))
                .await;
        }
        assert!(provider.read_with(bridge_cx, |store, _| store.discovery_idle_for_test()));
        {
            let observed = observations
                .lock()
                .expect("completed Get snapshot boundary");
            assert_eq!(observed.discovery_ids.len(), 1);
            assert!(
                observed.cancellation_ids.is_empty(),
                "dropping the structured snapshot does not cancel the completed Get"
            );
        }
        snapshot_release
            .send(())
            .expect("release the held structured snapshot");
        snapshot_completed
            .await
            .expect("held snapshot handler retires");
        cx.run_until_parked();
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
    }

    #[gpui::test]
    async fn rust_tools_headless_action_reply_latches_client_doctest_revocation(
        cx: &mut TestAppContext,
        bridge_cx: &mut TestAppContext,
        owner_cx: &mut TestAppContext,
    ) {
        use project::rust_test_provider::{RustTestAction, RustTestActionPlan};
        use project::structured_execution::StructuredNodeKind;

        struct DoctestMetadataRunner;
        #[async_trait::async_trait]
        impl CargoMetadataRunner for DoctestMetadataRunner {
            async fn run(&self, _: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
                let mut metadata: serde_json::Value = serde_json::from_slice(include_bytes!(
                    "../../project/test_data/cargo_workspace/standalone-v1.json"
                ))?;
                metadata["packages"][0]["targets"]
                    .as_array_mut()
                    .expect("fixture targets")
                    .push(json!({"kind":["lib"],"crate_types":["lib"],"name":"standalone", "src_path":"/standalone/src/lib.rs", "edition":"2021", "doc":true,"doctest":true,"test":true}));
                Ok(serde_json::to_vec(&metadata)?)
            }
        }
        struct DoctestDiscoveryRunner;
        impl RustTestDiscoveryRunner for DoctestDiscoveryRunner {
            fn discover(
                &self,
                request: RustTestDiscoveryRequest,
            ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>>
            {
                let ordinary =
                    DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new()))).discover(request);
                Box::pin(async move {
                    let mut capture = ordinary.await?;
                    capture.listings.push(RustTestListingCapture {
                        package_id: "path+file:///standalone#standalone@1.0.0".into(),
                        target_name: "standalone".into(),
                        target_kind: RustTestTargetKind::Doctest,
                        mode: RustTestListingMode::All,
                        stdout: "src/lib.rs - example (line 3): test\n1 test, 0 benchmarks\n"
                            .into(),
                    });
                    capture.listings.push(RustTestListingCapture {
                        package_id: "path+file:///standalone#standalone@1.0.0".into(),
                        target_name: "standalone".into(),
                        target_kind: RustTestTargetKind::Doctest,
                        mode: RustTestListingMode::Ignored,
                        stdout: "0 tests, 0 benchmarks\n".into(),
                    });
                    let protocol = project::rust_test_provider::RustTestProtocolAdapter::default()
                        .adapt(capture.clone(), &[]);
                    anyhow::ensure!(
                        protocol.capability
                            == project::rust_test_provider::RustTestProtocolCapability::Supported,
                        "fixture requires complete supported same-harness discovery"
                    );
                    Ok(capture)
                })
            }
        }

        let bridge_fs = FakeFs::new(bridge_cx.executor());
        bridge_fs.insert_tree(path!("/standalone"), json!({
            "Cargo.toml":"[package]\nname = \"standalone\"\nversion = \"1.0.0\"\nedition = \"2021\"\n",
            "src":{"main.rs":"fn main() {}", "lib.rs":"pub fn example() {}"}
        })).await;
        let (project, bridge) = init_test(&bridge_fs, cx, bridge_cx).await;
        let owner_fs = FakeFs::new(owner_cx.executor());
        owner_fs
            .insert_tree(
                path!("/standalone"),
                json!({
                    "Cargo.toml":"[package]\nname = \"standalone\"\nversion = \"1.0.0\"\nedition = \"2021\"\n",
                    "src":{"main.rs":"fn main() {}", "lib.rs":"/// Example\n/// ```\n/// assert!(true);\n/// ```\npub fn example() {}"}
                }),
            )
            .await;
        let (upstream_project, owner) = init_test(&owner_fs, bridge_cx, owner_cx).await;
        let feature = "rust-exact-doctest-actions-experimental";
        for context in [&mut *cx, &mut *bridge_cx, &mut *owner_cx] {
            context.update(|cx| {
                project::task_store::set_experimental_feature_for_test(feature, true, cx)
            });
        }
        owner.update(owner_cx, |owner, cx| {
            owner.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    Arc::new(DoctestMetadataRunner),
                );
                store.shared(proto::REMOTE_SERVER_PROJECT_ID, owner.session.clone(), cx);
            });
            owner.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    owner.cargo_workspace_store.clone(),
                    owner.structured_execution_store.clone(),
                    Arc::new(DoctestDiscoveryRunner),
                    cx,
                );
            });
        });
        let (worktree, _) = upstream_project
            .update(bridge_cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("upstream worktree");
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
        let worktree_id = worktree.read_with(bridge_cx, |worktree, _| worktree.id());
        owner
            .read_with(owner_cx, |owner, cx| {
                owner
                    .worktree_store
                    .read(cx)
                    .worktree_for_id(worktree_id, cx)
                    .expect("owner worktree")
                    .read(cx)
                    .as_local()
                    .expect("local host worktree")
                    .scan_complete()
            })
            .await;
        owner
            .update(owner_cx, |owner, cx| {
                owner
                    .cargo_workspace_store
                    .update(cx, |store, cx| store.refresh(cx))
            })
            .await
            .expect("host metadata");
        let (upstream, worktrees) = upstream_project.read_with(bridge_cx, |project, cx| {
            (
                project
                    .remote_client()
                    .expect("upstream client")
                    .read(cx)
                    .proto_client(),
                project.worktree_store(),
            )
        });
        let bridge_provider = bridge.update(bridge_cx, |bridge, cx| {
            bridge.structured_execution_store.update(cx, |store, _| {
                *store = project::structured_execution::StructuredExecutionStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    upstream.clone(),
                    worktrees.clone(),
                );
            });
            bridge.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    upstream,
                    worktrees,
                    bridge.structured_execution_store.clone(),
                    cx,
                );
            });
            bridge.rust_test_provider_store.clone()
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("client worktree");
        assert_eq!(
            worktree.read_with(cx, |worktree, _| worktree.id()),
            worktree_id
        );
        let provider =
            project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let snapshot = provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .expect("registered two-hop host discovery");
        let exact = snapshot
            .nodes
            .iter()
            .find(|node| {
                node.kind == StructuredNodeKind::Case
                    && node.experimental_feature.as_deref() == Some(feature)
            })
            .expect("host-discovered doctest");
        assert_eq!(exact.experimental_feature_enabled, Some(true));
        let ordinary = snapshot
            .nodes
            .iter()
            .find(|node| node.kind == StructuredNodeKind::Case && node.label == "host_case")
            .expect("ordinary host case");
        assert_eq!(ordinary.experimental_feature, None);
        for (node, revoke, expected_requirements) in [
            (&exact.id, true, Some(vec![feature.to_string()])),
            (&exact.id, false, Some(vec![feature.to_string()])),
            (&ordinary.id, true, Some(Vec::new())),
        ] {
            let (assembled, release) = bridge_provider.update(bridge_cx, |provider, _| {
                provider.gate_next_action_reply_for_test()
            });
            let pending = provider.update(cx, |provider, cx| {
                provider.plan_action(node, snapshot.discovery_generation, RustTestAction::Run, cx)
            });
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            assert!(matches!(
                futures::future::select(assembled, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ));
            if revoke {
                for enabled in [false, true] {
                    cx.update(|cx| {
                        project::task_store::set_experimental_feature_for_test(feature, enabled, cx)
                    });
                    cx.run_until_parked();
                    if !enabled {
                        let ordinary_plan = provider.update(cx, |provider, cx| {
                            provider.plan_action(&ordinary.id, snapshot.discovery_generation,
                                RustTestAction::Run, cx)
                        }).await.expect("ordinary actual host action remains admitted while client Exact is Off");
                        assert!(ordinary_plan.required_experimental_features().is_empty());
                    }
                }
            }
            bridge_cx.read(|cx| {
                use feature_flags::FeatureFlagAppExt as _;
                assert!(
                    cx.has_flag_by_name(feature),
                    "forwarding host opt-in remains On"
                );
            });
            owner_cx.read(|cx| {
                use feature_flags::FeatureFlagAppExt as _;
                assert!(
                    cx.has_flag_by_name(feature),
                    "only the client opt-in changed"
                );
            });
            release
                .send(())
                .expect("release actual forwarded host reply");
            let result = pending.await;
            if revoke && node == &exact.id {
                let error =
                    result.expect_err("old admitted host reply must not survive client Off→On");
                assert!(error.to_string().contains("revoked"));
            } else {
                let plan = result.expect("fresh Exact and ordinary host plans remain available");
                assert_eq!(
                    Some(plan.required_experimental_features().to_vec()),
                    expected_requirements
                );
                let RustTestActionPlan::Task {
                    scope_node_ids,
                    discovery_generation,
                    worktree_id: planned_worktree,
                    ..
                } = plan
                else {
                    panic!("Run must return an admitted Task plan");
                };
                assert_eq!(scope_node_ids, vec![node.clone()]);
                assert_eq!(discovery_generation, snapshot.discovery_generation);
                assert_eq!(planned_worktree, worktree_id);
            }
        }
    }

    #[gpui::test]
    async fn rust_tools_headless_runtime_lifecycle_forwarding_privacy(
        cx: &mut TestAppContext,
        bridge_cx: &mut TestAppContext,
        owner_cx: &mut TestAppContext,
    ) {
        use futures::FutureExt as _;
        use rpc::{AnyProtoClient, ProtoClient, ProtoMessageHandlerSet};

        struct ObservedUpstream {
            client: AnyProtoClient,
            requests: Arc<std::sync::Mutex<Vec<(Option<proto::PeerId>, proto::UpdateRustTestRun)>>>,
            handlers: parking_lot::Mutex<ProtoMessageHandlerSet>,
        }

        impl ProtoClient for ObservedUpstream {
            fn request(
                &self,
                envelope: proto::Envelope,
                request_type: &'static str,
            ) -> futures::future::BoxFuture<'static, anyhow::Result<proto::Envelope>> {
                let client = self.client.clone();
                let original_sender = envelope.original_sender_id;
                let observed = self.requests.clone();
                macro_rules! forward {
                    ($message:ty) => {{
                        async move {
                            let request =
                                <$message as proto::EnvelopedMessage>::from_envelope(envelope)
                                    .expect("observed typed upstream request");
                            let response = match original_sender {
                                Some(peer) => client.forward_request(peer, request).await?,
                                None => client.request(request).await?,
                            };
                            Ok(proto::EnvelopedMessage::into_envelope(
                                response, 0, None, None,
                            ))
                        }
                        .boxed()
                    }};
                }
                match envelope.payload.as_ref() {
                    Some(proto::envelope::Payload::UpdateRustTestRun(_)) => async move {
                        let request =
                            <proto::UpdateRustTestRun as proto::EnvelopedMessage>::from_envelope(
                                envelope,
                            )
                            .expect("observed typed upstream lifecycle");
                        observed
                            .lock()
                            .expect("upstream lifecycle observation")
                            .push((original_sender, request.clone()));
                        let response = match original_sender {
                            Some(peer) => client.forward_request(peer, request).await?,
                            None => client.request(request).await?,
                        };
                        Ok(proto::EnvelopedMessage::into_envelope(
                            response, 0, None, None,
                        ))
                    }
                    .boxed(),
                    Some(proto::envelope::Payload::ResolveRustTestAction(_)) => {
                        forward!(proto::ResolveRustTestAction)
                    }
                    Some(proto::envelope::Payload::GetRustTestDiscovery(_)) => {
                        forward!(proto::GetRustTestDiscovery)
                    }
                    Some(proto::envelope::Payload::GetStructuredExecutionSnapshot(_)) => {
                        forward!(proto::GetStructuredExecutionSnapshot)
                    }
                    Some(proto::envelope::Payload::GetStructuredExecutionEvents(_)) => {
                        forward!(proto::GetStructuredExecutionEvents)
                    }
                    Some(proto::envelope::Payload::CancelRustTestDiscovery(_)) => {
                        forward!(proto::CancelRustTestDiscovery)
                    }
                    _ => async move {
                        anyhow::bail!("unexpected lifecycle observer request: {request_type}")
                    }
                    .boxed(),
                }
            }

            fn send(&self, _: proto::Envelope, message_type: &'static str) -> anyhow::Result<()> {
                anyhow::bail!("unexpected lifecycle observer send: {message_type}")
            }

            fn send_response(
                &self,
                _: proto::Envelope,
                message_type: &'static str,
            ) -> anyhow::Result<()> {
                anyhow::bail!("unexpected lifecycle observer response: {message_type}")
            }

            fn message_handler_set(&self) -> &parking_lot::Mutex<ProtoMessageHandlerSet> {
                &self.handlers
            }

            fn is_via_collab(&self) -> bool {
                false
            }
            fn has_wsl_interop(&self) -> bool {
                false
            }
        }

        const SECRET: &str = "fixture-only-forwarded-private-environment";
        const PRIVATE_PATH: &str = "/host-owner/private-outside-worktree/task-shell";
        let sensitive = format!("could not start task: {SECRET} {PRIVATE_PATH}");
        let bridge_fs = FakeFs::new(bridge_cx.executor());
        let (project, bridge) = init_test(&bridge_fs, cx, bridge_cx).await;
        let owner_fs = FakeFs::new(owner_cx.executor());
        owner_fs
            .insert_tree(
                path!("/standalone"),
                json!({
                    "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n",
                    "src": {"main.rs": "fn main() {}"},
                }),
            )
            .await;
        let (upstream_project, owner) = init_test(&owner_fs, bridge_cx, owner_cx).await;
        owner.update(owner_cx, |owner, cx| {
            owner.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(proto::REMOTE_SERVER_PROJECT_ID, owner.session.clone(), cx);
            });
            owner.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    owner.worktree_store.clone(),
                    owner.environment.clone(),
                    owner.cargo_workspace_store.clone(),
                    owner.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
            });
        });
        let (worktree, _) = upstream_project
            .update(bridge_cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("two-hop upstream worktree");
        bridge_cx.run_until_parked();
        owner_cx.run_until_parked();
        let worktree_id = worktree.read_with(bridge_cx, |worktree, _| worktree.id());
        let scan = owner.read_with(owner_cx, |owner, cx| {
            owner
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("owner worktree")
                .read(cx)
                .as_local()
                .expect("owner scan")
                .scan_complete()
        });
        scan.await;
        owner
            .update(owner_cx, |owner, cx| {
                owner
                    .cargo_workspace_store
                    .update(cx, |store, cx| store.refresh(cx))
            })
            .await
            .expect("settled owner metadata");
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (upstream, worktrees) = upstream_project.read_with(bridge_cx, |project, cx| {
            (
                project
                    .remote_client()
                    .expect("upstream client")
                    .read(cx)
                    .proto_client(),
                project.worktree_store(),
            )
        });
        let upstream = AnyProtoClient::new(Arc::new(ObservedUpstream {
            client: upstream,
            requests: observed.clone(),
            handlers: Default::default(),
        }));
        bridge.update(bridge_cx, |bridge, cx| {
            bridge.structured_execution_store.update(cx, |store, _| {
                *store = project::structured_execution::StructuredExecutionStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    upstream.clone(),
                    worktrees.clone(),
                );
            });
            bridge.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::remote(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    upstream.clone(),
                    worktrees.clone(),
                    bridge.structured_execution_store.clone(),
                    cx,
                );
            });
        });
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("outer client")
                .read(cx)
                .proto_client()
        });
        let discovered = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 1,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id.to_proto()],
            })
            .await
            .expect("registered two-hop discovery");
        let page = client
            .request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: discovered.discovery_generation,
                page_start: 0,
                page_size: 128,
                worktree_ids: vec![worktree_id.to_proto()],
            })
            .await
            .expect("bridge projected owner nodes");
        let node = page
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("two-hop executable case");
        let peer = proto::PeerId {
            owner_id: 17,
            id: 1,
        };
        let other_peer = proto::PeerId {
            owner_id: 17,
            id: 2,
        };
        let plan = client
            .request(super::peer_requests::FromPeer {
                peer,
                payload: proto::ResolveRustTestAction {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version:
                        project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                    discovery_generation: discovered.discovery_generation,
                    node_id: node.node_id.clone(),
                    action: proto::RustTestActionKind::Run as i32,
                    worktree_ids: vec![worktree_id.to_proto()],
                    authorization_generation: 0,
                },
            })
            .await
            .expect("original-peer two-hop plan");
        #[derive(serde::Deserialize)]
        struct RunIdentity {
            run_id: String,
            scope_node_ids: Vec<String>,
        }
        let identity: RunIdentity =
            serde_json::from_str(&plan.plan_json).expect("authorized plan identity");
        let queued = proto::UpdateRustTestRun {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: discovered.discovery_generation,
            run_id: identity.run_id.clone(),
            scope_node_ids: identity.scope_node_ids.clone(),
            task_id: "two-hop-privacy-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            diagnostic: Some(sensitive.clone()),
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        };
        client
            .request(super::peer_requests::FromPeer {
                peer: other_peer,
                payload: queued.clone(),
            })
            .await
            .expect_err("foreign peer cannot consume original grant");
        assert!(observed.lock().expect("observations").is_empty());
        client
            .request(super::peer_requests::FromPeer {
                peer,
                payload: queued.clone(),
            })
            .await
            .expect("owner queued lifecycle crosses both hops");
        client
            .request(super::peer_requests::FromPeer {
                peer,
                payload: proto::UpdateRustTestRun {
                    state: proto::RustTestTaskState::SpawnError as i32,
                    ..queued.clone()
                },
            })
            .await
            .expect("owner spawn failure crosses both hops");
        client
            .request(super::peer_requests::FromPeer {
                peer,
                payload: queued,
            })
            .await
            .expect_err("terminal lifecycle retires both run grants");
        let requests = observed
            .lock()
            .expect("observed forwarded lifecycle")
            .clone();
        assert_eq!(requests.len(), 2);
        assert!(
            requests
                .iter()
                .all(|(sender, request)| *sender == Some(peer)
                    && request.run_id == identity.run_id
                    && request.scope_node_ids == identity.scope_node_ids
                    && request.worktree_ids == vec![worktree_id.to_proto()]
                    && request.discovery_generation == discovered.discovery_generation)
        );
        let mut violations = Vec::new();
        for (_, request) in &requests {
            for value in [SECRET, PRIVATE_PATH] {
                if request
                    .diagnostic
                    .as_deref()
                    .is_some_and(|message| message.contains(value))
                {
                    violations
                        .push("upstream envelope retains controlled sensitive lifecycle data");
                }
            }
        }
        if requests
            .first()
            .is_some_and(|(_, request)| request.diagnostic.is_some())
        {
            violations.push("non-error lifecycle forwards an unused diagnostic");
        }
        let terminal = owner.read_with(owner_cx, |owner, cx| {
            owner
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .and_then(|provider| {
                    provider
                        .completed_runs
                        .iter()
                        .find(|run| run.id.0 == identity.run_id)
                        .cloned()
                })
                .expect("owner terminal result")
        });
        assert_eq!(terminal.id.0, identity.run_id);
        assert_eq!(
            terminal.phase,
            project::structured_execution::StructuredRunPhase::SpawnError
        );
        assert!(
            terminal
                .diagnostic
                .as_ref()
                .is_some_and(|message| !message.is_empty())
        );
        let owner_details = format!("{terminal:?}");
        for value in [SECRET, PRIVATE_PATH] {
            if owner_details.contains(value) {
                violations.push("owner run retains controlled forwarded spawn details");
            }
        }
        assert!(
            violations.is_empty(),
            "actual two-hop lifecycle privacy violations: {violations:?}"
        );
    }

    #[gpui::test]
    async fn rust_tools_headless_runtime_lifecycle_diagnostic_privacy(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use project::{
            rust_test_provider::{RustTestAction, RustTestActionPlan},
            structured_execution::StructuredRunPhase,
        };
        use task::{StructuredTaskHandle, StructuredTaskState, TaskId};

        const SECRET: &str = "fixture-only-task-private-environment";
        const PRIVATE_PATH: &str = "/host-owner/private-outside-worktree/task-shell";
        let sensitive = format!("could not start task: {SECRET} {PRIVATE_PATH}");
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n",
                "src": {"main.rs": "fn main() {}"},
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
            });
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote lifecycle worktree");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let scan = headless.read_with(server_cx, |headless, cx| {
            let worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("host lifecycle worktree");
            worktree
                .read(cx)
                .as_local()
                .expect("host lifecycle scan")
                .scan_complete()
        });
        scan.await;
        headless
            .update(server_cx, |headless, cx| {
                headless
                    .cargo_workspace_store
                    .update(cx, |store, cx| store.refresh(cx))
            })
            .await
            .expect("settled host lifecycle metadata");
        let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let snapshot = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("authoritative lifecycle discovery");
        let node_id = snapshot
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("authorized lifecycle case")
            .id
            .clone();
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote lifecycle client")
                .read(cx)
                .proto_client()
        });
        let mut violations = Vec::new();
        for (index, route) in ["subscribed Task handle", "older client diagnostic"]
            .into_iter()
            .enumerate()
        {
            let RustTestActionPlan::Task {
                run_id,
                discovery_generation,
                scope_node_ids,
                ..
            } = tests
                .update(cx, |store, cx| {
                    store.plan_action(
                        &node_id,
                        snapshot.discovery_generation,
                        RustTestAction::Run,
                        cx,
                    )
                })
                .await
                .expect("authorized lifecycle Task plan")
            else {
                panic!("Run must produce a Task plan");
            };
            if index == 0 {
                let handle = StructuredTaskHandle::new(TaskId("privacy-owned-task".into()));
                tests
                    .update(cx, |store, cx| {
                        store.register_run_handle(
                            run_id.clone(),
                            discovery_generation,
                            scope_node_ids.clone(),
                            handle.clone(),
                            cx,
                        )
                    })
                    .expect("production remote lifecycle subscription");
                cx.update(|cx| assert!(handle.mark_spawn_error(sensitive.clone(), cx)));
                assert_eq!(
                    handle.state(),
                    StructuredTaskState::SpawnError {
                        message: sensitive.clone()
                    }
                );
            } else {
                let queued = proto::UpdateRustTestRun {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version:
                        project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                    discovery_generation: discovery_generation.0,
                    run_id: run_id.0.clone(),
                    scope_node_ids: scope_node_ids.iter().map(|node| node.0.clone()).collect(),
                    task_id: "privacy-legacy-task".into(),
                    state: proto::RustTestTaskState::Queued as i32,
                    worktree_ids: vec![worktree_id.to_proto()],
                    ..Default::default()
                };
                client
                    .request(queued.clone())
                    .await
                    .expect("older client queued lifecycle");
                client
                    .request(proto::UpdateRustTestRun {
                        state: proto::RustTestTaskState::SpawnError as i32,
                        diagnostic: Some(sensitive.clone()),
                        ..queued
                    })
                    .await
                    .expect("older client spawn failure lifecycle");
            }
            let mut completed = None;
            for _ in 0..100 {
                cx.run_until_parked();
                server_cx.run_until_parked();
                completed = headless.read_with(server_cx, |headless, cx| {
                    let provider = headless
                        .structured_execution_store
                        .read(cx)
                        .state()
                        .provider(&RustTestProviderStore::provider_id())?;
                    provider
                        .completed_runs
                        .iter()
                        .chain(provider.last_complete_run.iter())
                        .find(|run| run.id == run_id)
                        .cloned()
                });
                if completed.is_some() {
                    break;
                }
                cx.background_executor
                    .timer(std::time::Duration::from_millis(1))
                    .await;
            }
            let completed =
                completed.expect("authorized spawn failure reaches host terminal state");
            assert_eq!(completed.phase, StructuredRunPhase::SpawnError);
            assert_eq!(completed.scope_node_ids, scope_node_ids);
            assert!(
                completed
                    .diagnostic
                    .as_ref()
                    .is_some_and(|message| !message.is_empty())
            );
            let host_details = format!("{completed:?}");
            for value in [SECRET, PRIVATE_PATH] {
                if host_details.contains(value) {
                    violations.push(format!("{route} exported controlled sensitive task data"));
                }
            }
            let page = client
                .request(proto::GetStructuredExecutionSnapshot {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    provider_id: RustTestProviderStore::provider_id().0,
                    discovery_generation: snapshot.discovery_generation.0,
                    page_start: 0,
                    page_size: 128,
                    worktree_ids: vec![worktree_id.to_proto()],
                })
                .await
                .expect("independently redacted shared lifecycle snapshot");
            let shared = format!("{page:?}");
            assert!(!shared.contains(SECRET) && !shared.contains(PRIVATE_PATH));
        }
        assert!(
            violations.is_empty(),
            "actual lifecycle privacy violations: {violations:?}"
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn rust_tools_headless_runtime_discovery_diagnostic_privacy(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt;
        const SECRET: &str = "fixture-only-rust-discovery-private-environment";
        const PRIVATE_PATH: &str = "/host-owner/private-outside-worktree/secret.rs";
        let executors = [cx.executor(), server_cx.executor()];
        for executor in &executors {
            executor.allow_parking();
        }
        let directory = tempfile::tempdir().expect("controlled diagnostic process directory");
        let cargo_path = directory.path().join("cargo");
        let harness_path = directory.path().join("harness");
        std::fs::write(&cargo_path, concat!(
            "#!/bin/sh\n",
            "printf '%s\\n' \"$$\" > \"$RUST_DIAGNOSTIC_PID\"\n",
            "if [ -f \"$RUST_DIAGNOSTIC_RECOVER\" ]; then exec /bin/cat \"$RUST_DIAGNOSTIC_MESSAGES\"; fi\n",
            "printf '%s %s\\n' \"$RUST_DIAGNOSTIC_SECRET\" \"$RUST_DIAGNOSTIC_PRIVATE_PATH\" >&2\n",
            "exit 7\n",
        )).expect("write failing real Cargo fixture");
        std::fs::write(&harness_path, concat!(
            "#!/bin/sh\n",
            "case \"$*\" in *--ignored*) printf '0 tests, 0 benchmarks\\n';; *) printf 'host_case: test\\n1 test, 0 benchmarks\\n';; esac\n",
        )).expect("write recovery harness");
        for path in [&cargo_path, &harness_path] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .expect("make controlled fixture executable");
        }
        let capture = DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))
            .discover(RustTestDiscoveryRequest {
                workspace_root: project::ProjectPath {
                    worktree_id: project::WorktreeId::from_usize(1),
                    path: rel_path("Cargo.toml").into(),
                },
                generation: 1,
                limits: Default::default(),
                manifest_path: None,
                working_directory: None,
                environment: None,
                doctest_targets: Vec::new(),
            })
            .await
            .expect("existing registered host capture");
        let mut message: serde_json::Value =
            serde_json::from_str(&capture.cargo_messages).expect("existing Cargo artifact record");
        message
            .as_object_mut()
            .expect("controlled artifact object")
            .insert("executable".into(), json!(harness_path));
        std::fs::write(directory.path().join("messages.jsonl"), message.to_string())
            .expect("write recovery Cargo artifact");
        let environment = HashMap::from_iter([
            (
                "PATH".into(),
                directory.path().to_string_lossy().into_owned(),
            ),
            ("RUST_DIAGNOSTIC_SECRET".into(), SECRET.into()),
            ("RUST_DIAGNOSTIC_PRIVATE_PATH".into(), PRIVATE_PATH.into()),
            (
                "RUST_DIAGNOSTIC_PID".into(),
                directory
                    .path()
                    .join("child.pid")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "RUST_DIAGNOSTIC_RECOVER".into(),
                directory
                    .path()
                    .join("recover")
                    .to_string_lossy()
                    .into_owned(),
            ),
            (
                "RUST_DIAGNOSTIC_MESSAGES".into(),
                directory
                    .path()
                    .join("messages.jsonl")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ]);
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n",
                "src": {"main.rs": "fn main() {}"},
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
            });
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("registered remote discovery fixture");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let fixture_worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let scan = headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(fixture_worktree_id, cx)
                .expect("host fixture worktree");
            host_worktree
                .read(cx)
                .as_local()
                .expect("local host scan")
                .scan_complete()
        });
        scan.await;
        let metadata = headless
            .update(server_cx, |headless, cx| {
                headless
                    .cargo_workspace_store
                    .update(cx, |store, cx| store.refresh(cx))
            })
            .await
            .expect("settled fixture Cargo metadata");
        assert_eq!(metadata.workspaces.len(), 1, "{metadata:?}");
        headless.update(server_cx, |headless, cx| {
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(ProcessDiagnosticRunner {
                        executor: cx.background_executor().clone(),
                        directory: directory.path().to_path_buf(),
                        environment,
                    }),
                    cx,
                );
            });
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let request = proto::GetRustTestDiscovery {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 1,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id],
        };
        let expected_failure_status = headless.read_with(server_cx, |headless, cx| {
            if headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .is_some_and(|previous| !previous.nodes.is_empty())
            {
                proto::StructuredProviderStatus::Stale
            } else {
                proto::StructuredProviderStatus::Error
            }
        });
        let response = await_diagnostic_process(&executors, client.request(request.clone()))
            .expect("registered real-process failure response");
        assert_eq!(
            response.status, expected_failure_status as i32,
            "{response:?}"
        );
        let diagnostic = response
            .diagnostic
            .as_deref()
            .expect("visible failure category");
        assert!(
            diagnostic.contains("Cargo") && diagnostic.contains('7'),
            "{diagnostic}"
        );
        let mut violations = Vec::new();
        let host = headless.read_with(server_cx, |headless, cx| {
            headless
                .structured_execution_store
                .read(cx)
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .cloned()
                .expect("host diagnostic projection")
        });
        let page = await_diagnostic_process(
            &executors,
            client.request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: response.discovery_generation,
                page_start: 0,
                page_size: 128,
                worktree_ids: vec![worktree_id],
            }),
        )
        .expect("existing independently redacted shared snapshot route");
        for (route, diagnostic) in [
            ("discovery RPC", response.diagnostic.as_deref()),
            ("host provider projection", host.diagnostic.as_deref()),
            ("structured RPC", page.diagnostic.as_deref()),
        ] {
            for sensitive in [SECRET, PRIVATE_PATH] {
                if diagnostic.is_some_and(|text| text.contains(sensitive)) {
                    violations.push(format!("{route} retains controlled sensitive process data"));
                }
            }
        }
        let pid = std::fs::read_to_string(directory.path().join("child.pid"))
            .expect("real Cargo process wrote PID");
        let process_id = pid
            .trim()
            .parse::<u32>()
            .expect("real Cargo process PID")
            .to_string();
        let mut probe = util::command::new_command("/bin/kill");
        probe
            .args(["-0", &process_id])
            .stdout(util::command::Stdio::null())
            .stderr(util::command::Stdio::null())
            .kill_on_drop(true);
        assert!(
            !await_diagnostic_process(&executors, probe.status())
                .expect("owned PID probe")
                .success()
        );
        std::fs::write(directory.path().join("recover"), "").expect("enable real-process recovery");
        let recovered = await_diagnostic_process(
            &executors,
            client.request(proto::GetRustTestDiscovery {
                request_id: 2,
                ..request.clone()
            }),
        )
        .expect("fresh production discovery after bounded failure");
        let expected_recovery_status = if metadata.completeness
            == project::cargo_workspace::CargoSnapshotCompleteness::Partial
        {
            proto::StructuredProviderStatus::Partial
        } else {
            proto::StructuredProviderStatus::Current
        };
        assert_eq!(
            recovered.status, expected_recovery_status as i32,
            "{recovered:?}; initial metadata: {metadata:?}"
        );
        assert!(recovered.diagnostic.is_none(), "{recovered:?}");
        let recovered_page = await_diagnostic_process(
            &executors,
            client.request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: recovered.discovery_generation,
                page_start: 0,
                page_size: 128,
                worktree_ids: vec![worktree_id],
            }),
        )
        .expect("fresh shared recovery snapshot");
        assert!(
            recovered_page
                .nodes
                .iter()
                .any(|node| node.label == "host_case"),
            "{recovered_page:?}"
        );
        headless.update(server_cx, |headless, cx| {
            headless.rust_test_provider_store.update(cx, |store, _| {
                store
                    .set_discovery_runner_for_test(Arc::new(OpaqueDiagnosticRunner(format!(
                        "unclassified runner error {SECRET} {PRIVATE_PATH}"
                    ))))
                    .expect("install opaque runner failure");
            });
        });
        let opaque = await_diagnostic_process(
            &executors,
            client.request(proto::GetRustTestDiscovery {
                request_id: 3,
                ..request
            }),
        )
        .expect("registered opaque failure response");
        assert_eq!(
            opaque.status,
            proto::StructuredProviderStatus::Stale as i32,
            "{opaque:?}"
        );
        let diagnostic = opaque
            .diagnostic
            .as_deref()
            .expect("visible opaque failure category");
        assert!(diagnostic.contains("unclassified"), "{diagnostic}");
        let opaque_page = await_diagnostic_process(
            &executors,
            client.request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: opaque.discovery_generation,
                page_start: 0,
                page_size: 128,
                worktree_ids: vec![worktree_id],
            }),
        )
        .expect("shared stale rows after opaque runner failure");
        assert_eq!(
            opaque_page.status,
            proto::StructuredProviderStatus::Stale as i32
        );
        assert!(
            opaque_page.partial
                && opaque_page
                    .nodes
                    .iter()
                    .any(|node| node.label == "host_case")
        );
        assert!(
            !opaque_page
                .diagnostic
                .as_deref()
                .is_some_and(|text| text.contains(SECRET) || text.contains(PRIVATE_PATH))
        );
        for sensitive in [SECRET, PRIVATE_PATH] {
            if diagnostic.contains(sensitive) {
                violations
                    .push("opaque discovery RPC retains arbitrary sensitive error text".into());
            }
        }
        assert!(
            violations.is_empty(),
            "actual diagnostic privacy violations: {violations:?}"
        );
    }

    struct GatedMetadataRunner {
        started: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        release: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }

    #[async_trait::async_trait]
    impl CargoMetadataRunner for GatedMetadataRunner {
        async fn run(&self, _: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
            let release = self
                .release
                .lock()
                .expect("metadata gate lock")
                .take()
                .expect("one metadata invocation");
            self.started
                .lock()
                .expect("metadata start lock")
                .take()
                .expect("one start notification")
                .send(())
                .expect("metadata start receiver");
            release.await?;
            Ok(
                include_bytes!("../../project/test_data/cargo_workspace/standalone-v1.json")
                    .to_vec(),
            )
        }
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_revokes_inflight_metadata(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote metadata worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting metadata");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(GatedMetadataRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
        });
        let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let request = cargo.update(cx, |store, cx| store.refresh(cx));
        started_receiver
            .await
            .expect("metadata runner is in flight");
        client
            .request(proto::RestrictWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("revoke trust during metadata execution");
        let error = request
            .await
            .expect_err("revoked metadata must not reach the client");
        assert_eq!(
            error
                .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
                .expect("typed remote restriction")
                .kind,
            project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::Restricted
        );
        assert!(
            error
                .to_string()
                .contains("Trust the worktree and refresh Cargo"),
            "{error:#}"
        );
        assert!(
            release.send(()).is_err(),
            "revocation drops the active runner"
        );
    }

    struct GatedDiscoveryRunner {
        started: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        release: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }

    impl RustTestDiscoveryRunner for GatedDiscoveryRunner {
        fn discover(
            &self,
            request: RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
            let release = self
                .release
                .lock()
                .expect("discovery gate lock")
                .take()
                .expect("one discovery invocation");
            self.started
                .lock()
                .expect("discovery start lock")
                .take()
                .expect("one start notification")
                .send(())
                .expect("discovery start receiver");
            Box::pin(async move {
                release.await?;
                DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))
                    .discover(request)
                    .await
            })
        }
    }

    struct LateDiscoveryRunner(GatedDiscoveryRunner);

    impl RustTestDiscoveryRunner for LateDiscoveryRunner {
        fn discover(
            &self,
            request: RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
            let release = self.0.release.lock().expect("discovery gate lock").take();
            if release.is_some() {
                self.0
                    .started
                    .lock()
                    .expect("discovery start lock")
                    .take()
                    .expect("first discovery invocation")
                    .send(())
                    .expect("discovery start receiver");
            }
            Box::pin(async move {
                if let Some(release) = release {
                    release.await?;
                }
                DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))
                    .discover(request)
                    .await
            })
        }
    }

    struct NthGatedDiscoveryRunner {
        requests: Arc<std::sync::Mutex<Vec<RustTestDiscoveryRequest>>>,
        calls: AtomicUsize,
        gated_index: usize,
        started: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        release: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }

    impl RustTestDiscoveryRunner for NthGatedDiscoveryRunner {
        fn discover(
            &self,
            request: RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
            let release = if self.calls.fetch_add(1, Ordering::SeqCst) == self.gated_index {
                self.started
                    .lock()
                    .expect("discovery start lock")
                    .take()
                    .expect("one gated discovery")
                    .send(())
                    .expect("gated discovery start receiver");
                self.release.lock().expect("discovery release lock").take()
            } else {
                None
            };
            let requests = self.requests.clone();
            Box::pin(async move {
                if let Some(release) = release {
                    release.await?;
                }
                DiscoveryRunner(requests).discover(request).await
            })
        }
    }

    struct LateMetadataRunner {
        started: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        release: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }

    #[async_trait::async_trait]
    impl CargoMetadataRunner for LateMetadataRunner {
        async fn run(&self, _: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
            let release = self.release.lock().expect("metadata gate lock").take();
            if let Some(release) = release {
                self.started
                    .lock()
                    .expect("metadata start lock")
                    .take()
                    .expect("first metadata invocation")
                    .send(())
                    .expect("metadata start receiver");
                release.await?;
            }
            Ok(
                include_bytes!("../../project/test_data/cargo_workspace/standalone-v1.json")
                    .to_vec(),
            )
        }
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_metadata_reconnect_invalidates_pending(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote metadata worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting metadata");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(LateMetadataRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
        });
        let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let request = cargo.update(cx, |store, cx| store.refresh(cx));
        started_receiver.await.expect("metadata runner is active");
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote
            .update(cx, |remote, cx| remote.simulate_disconnect(cx))
            .await;
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        match futures::future::select(request, deadline).await {
            futures::future::Either::Left((result, _)) => {
                result.expect_err("connection interruption must invalidate pending metadata");
            }
            futures::future::Either::Right(_) => {
                panic!("metadata request survived connection interruption")
            }
        }
        if release.send(()).is_err() {
            log::debug!("Interrupted metadata runner was dropped");
        }
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            futures::pin_mut!(recovery);
            assert!(
                matches!(
                    futures::future::select(recovery, deadline).await,
                    futures::future::Either::Left(_)
                ),
                "mock connection must recover"
            );
        }
        let snapshot = cargo
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("new metadata request succeeds after reconnect");
        assert_eq!(snapshot.workspaces.len(), 1);
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_rejects_metadata_after_disconnect(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote metadata worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting metadata");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(LateMetadataRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
        });
        let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let request = cargo.update(cx, |store, cx| store.refresh(cx));
        started_receiver.await.expect("metadata runner is active");
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_server_not_running(cx));
        cx.run_until_parked();
        server_cx.run_until_parked();
        if release.send(()).is_err() {
            log::debug!("Metadata runner was dropped before releasing its late reply");
        }
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(1));
        match futures::future::select(request, deadline).await {
            futures::future::Either::Left((result, _)) => {
                let error =
                    result.expect_err("a disconnected client must reject the late metadata reply");
                assert_eq!(
                    error
                        .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
                        .expect("typed disconnected result")
                        .kind,
                    project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::Disconnected,
                );
            }
            futures::future::Either::Right(_) => {
                panic!("disconnected metadata request did not terminate within one second");
            }
        }
        let error = cargo
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect_err("a terminally disconnected store must reject another refresh");
        assert_eq!(
            error
                .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
                .expect("typed disconnected refresh result")
                .kind,
            project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::Disconnected,
        );
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_cancels_only_own_inflight_metadata(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote metadata worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting metadata");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(GatedMetadataRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
        });
        let owner = proto::PeerId { owner_id: 9, id: 1 };
        let other = proto::PeerId { owner_id: 9, id: 2 };
        let cancellation = proto::CancelCargoWorkspace {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 501,
        };
        let metadata = client.request(super::peer_requests::FromPeer {
            peer: owner,
            payload: proto::GetCargoWorkspace {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 501,
                worktree_ids: vec![worktree_id],
            },
        });
        let cancel = async {
            started_receiver.await.expect("owner metadata is in flight");
            client
                .request(super::peer_requests::FromPeer {
                    peer: other,
                    payload: cancellation.clone(),
                })
                .await
                .expect("foreign cancellation acknowledged without affecting owner");
            cx.run_until_parked();
            server_cx.run_until_parked();
            assert!(
                !release.is_canceled(),
                "another peer must not drop the owner's runner"
            );
            client
                .request(super::peer_requests::FromPeer {
                    peer: owner,
                    payload: cancellation,
                })
                .await
                .expect("owner cancellation acknowledged");
        };
        let (result, ()) = futures::join!(metadata, cancel);
        let error = result.expect_err("owner cancellation must reject active metadata");
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert!(
            release.send(()).is_err(),
            "owner cancellation drops the active runner"
        );
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_rejects_discovery_after_disconnect(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        verify_discovery_connection_invalidation(cx, server_cx, false).await;
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_discovery_reconnect_invalidates_pending(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        verify_discovery_connection_invalidation(cx, server_cx, true).await;
    }

    #[gpui::test(iterations = 10)]
    async fn rust_tools_headless_runtime_discovery_rejects_completed_reply_after_reconnect(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use project::structured_execution::StructuredProviderStatus;

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote discovery worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before discovery");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
            });
        });
        let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let first = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("first host discovery");
        assert!(first.nodes.iter().any(|node| node.label == "host_case"));
        let first_generation = first.discovery_generation;
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .rust_test_provider_store
                .update(cx, |store, _| store.gate_next_discovery_reply_for_test())
        });
        let pending = tests.update(cx, |store, cx| store.refresh(cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "registered host Rust discovery handler must complete before the reply gate"
        );
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(1, cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        match futures::future::select(pending, deadline).await {
            futures::future::Either::Left((result, _)) => {
                result.expect_err("connection interruption must reject the pending discovery");
            }
            futures::future::Either::Right(_) => {
                panic!("discovery survived connection interruption")
            }
        }
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            futures::pin_mut!(recovery);
            assert!(
                matches!(
                    futures::future::select(recovery, deadline).await,
                    futures::future::Either::Left(_)
                ),
                "mock connection must recover"
            );
        }
        structured.read_with(cx, |store, _| {
            let snapshot = store
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .expect("cached Rust discovery retained");
            assert_eq!(snapshot.discovery_generation, first_generation);
            assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
        });
        release
            .send(())
            .expect("delayed registered Rust discovery reply remains active");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(completed, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "registered host must finish its old discovery reply"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
        structured.read_with(cx, |store, _| {
            let snapshot = store
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .expect("cached Rust discovery retained after old reply");
            assert_eq!(snapshot.discovery_generation, first_generation);
            assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
        });
        let refreshed = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("fresh discovery after reconnect");
        assert!(refreshed.discovery_generation > first_generation);
        assert!(refreshed.nodes.iter().any(|node| node.label == "host_case"));
    }

    async fn verify_discovery_connection_invalidation(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
        reconnect: bool,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote discovery worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting discovery");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(LateDiscoveryRunner(GatedDiscoveryRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    })),
                    cx,
                );
            });
        });
        let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let request = tests.update(cx, |store, cx| store.refresh(cx));
        started_receiver.await.expect("discovery runner is active");
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        if reconnect {
            remote
                .update(cx, |remote, cx| remote.simulate_disconnect(cx))
                .await;
        } else {
            remote.update(cx, |remote, cx| remote.force_server_not_running(cx));
        }
        cx.run_until_parked();
        server_cx.run_until_parked();
        if release.send(()).is_err() {
            log::debug!("Discovery runner dropped before late reply release");
        }
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(1));
        match futures::future::select(request, deadline).await {
            futures::future::Either::Left((result, _)) => {
                let error = result.expect_err("disconnected discovery must reject a late reply");
                assert!(error.to_string().contains("disconnected"), "{error:#}");
            }
            futures::future::Either::Right(_) => {
                panic!("disconnected discovery request did not terminate")
            }
        }
        if reconnect {
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            {
                let recovery = async {
                    while remote.read_with(cx, |remote, _| remote.connection_state())
                        != remote::ConnectionState::Connected
                    {
                        cx.background_executor
                            .timer(std::time::Duration::from_millis(10))
                            .await;
                    }
                };
                futures::pin_mut!(recovery);
                assert!(
                    matches!(
                        futures::future::select(recovery, deadline).await,
                        futures::future::Either::Left(_)
                    ),
                    "mock discovery connection must recover"
                );
            }
            let snapshot = tests
                .update(cx, |store, cx| store.refresh(cx))
                .await
                .expect("new discovery succeeds after reconnect");
            assert!(snapshot.nodes.iter().any(|node| node.label == "host_case"));
        } else {
            let error = tests
                .update(cx, |store, cx| store.refresh(cx))
                .await
                .expect_err("terminally disconnected discovery must reject a new refresh");
            assert!(error.to_string().contains("disconnected"), "{error:#}");
        }
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_cancels_only_own_inflight_discovery(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote discovery worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting discovery");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(GatedDiscoveryRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                    cx,
                );
            });
        });
        let owner = proto::PeerId { owner_id: 9, id: 1 };
        let other = proto::PeerId { owner_id: 9, id: 2 };
        let cancellation = proto::CancelRustTestDiscovery {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 501,
        };
        let discovery = client.request(super::peer_requests::FromPeer {
            peer: owner,
            payload: proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 501,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id],
            },
        });
        let cancel = async {
            started_receiver
                .await
                .expect("owner discovery is in flight");
            client
                .request(super::peer_requests::FromPeer {
                    peer: other,
                    payload: cancellation.clone(),
                })
                .await
                .expect("foreign cancellation acknowledged without affecting owner");
            cx.run_until_parked();
            server_cx.run_until_parked();
            assert!(
                !release.is_canceled(),
                "another peer must not drop the owner's runner"
            );
            client
                .request(super::peer_requests::FromPeer {
                    peer: owner,
                    payload: cancellation,
                })
                .await
                .expect("owner cancellation acknowledged");
        };
        let (result, ()) = futures::join!(discovery, cancel);
        let error = result.expect_err("owner cancellation must reject active discovery");
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        cx.run_until_parked();
        server_cx.run_until_parked();
        assert!(
            release.send(()).is_err(),
            "owner cancellation drops the active runner"
        );
    }

    #[gpui::test(iterations = 20)]
    async fn rust_tools_headless_runtime_revokes_inflight_discovery(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({"Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n", "src": {"main.rs": "fn main() {}"}}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote discovery worktree");
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("trust before starting discovery");
        cx.run_until_parked();
        server_cx.run_until_parked();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(GatedDiscoveryRunner {
                        started: std::sync::Mutex::new(Some(started)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                    cx,
                );
            });
        });
        let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let request = tests.update(cx, |store, cx| store.refresh(cx));
        started_receiver
            .await
            .expect("discovery runner is in flight");
        client
            .request(proto::RestrictWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("revoke trust during discovery execution");
        let error = request
            .await
            .expect_err("revoked discovery must not reach the client");
        assert!(
            error
                .to_string()
                .contains("Trust the worktree and refresh Tests"),
            "{error:#}"
        );
        assert!(
            release.send(()).is_err(),
            "revocation drops the active runner"
        );
        headless.update(server_cx, |headless, cx| {
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
            });
        });
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("restore trust after cancellation");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let snapshot = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("discovery recovers after trust is restored");
        assert!(snapshot.nodes.iter().any(|node| node.label == "host_case"));
    }

    impl RustTestDiscoveryRunner for DiscoveryRunner {
        fn discover(
            &self,
            request: RustTestDiscoveryRequest,
        ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
            self.0.lock().expect("runner request lock").push(request);
            Box::pin(async {
                let capture = RustTestDiscoveryCapture {
                    toolchain: "fixture-only".to_string(),
                    cargo_messages: json!({
                        "reason": "compiler-artifact",
                        "package_id": "path+file:///standalone#standalone@1.0.0",
                        "manifest_path": "/standalone/Cargo.toml",
                        "target": {"name": "standalone", "required-features": [], "kind": ["bin"], "crate_types": ["bin"], "src_path": "/standalone/src/main.rs", "edition": "2021", "test": true, "doctest": false, "doc": true},
                        "profile": {"opt_level": "0", "debuginfo": 0, "debug_assertions": true, "overflow_checks": true, "test": true},
                        "features": [], "filenames": ["/standalone/target/debug/deps/standalone"],
                        "executable": "/standalone/target/debug/deps/standalone", "fresh": false
                    }).to_string(),
                    listings: vec![RustTestListingCapture {
                        package_id: "path+file:///standalone#standalone@1.0.0".to_string(),
                        target_name: "standalone".to_string(),
                        target_kind: RustTestTargetKind::Binary,
                        mode: RustTestListingMode::All,
                        stdout: "host_case: test\n1 test, 0 benchmarks\n".to_string(),
                    }],
                };
                let adapted = project::rust_test_provider::RustTestProtocolAdapter::default()
                    .adapt(capture.clone(), &[]);
                anyhow::ensure!(
                    adapted.cases.len() == 1,
                    "fixture adaptation failed: {adapted:?}"
                );
                Ok(capture)
            })
        }
    }

    #[gpui::test(iterations = 10)]
    async fn rust_tools_headless_runtime_cold_tests_panel_discovers_on_open(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\nedition = \"2021\"\n",
                "src": {"main.rs": "fn main() {}"}
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let metadata_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let discovery_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (active_started_sender, active_started) = futures::channel::oneshot::channel();
        let (active_release, active_release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(metadata_requests.clone())),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(NthGatedDiscoveryRunner {
                        requests: discovery_requests.clone(),
                        calls: AtomicUsize::new(0),
                        gated_index: 3,
                        started: std::sync::Mutex::new(Some(active_started_sender)),
                        release: std::sync::Mutex::new(Some(active_release_receiver)),
                    }),
                    cx,
                );
            });
        });
        project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("the remote worktree should open");
        cx.run_until_parked();
        let provider_id = RustTestProviderStore::provider_id();
        let provider_before = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&provider_id)
                .cloned()
        });
        assert!(
            provider_before.is_none(),
            "remote panel starts without a provider"
        );
        assert!(metadata_requests.lock().expect("runner lock").is_empty());
        assert!(discovery_requests.lock().expect("runner lock").is_empty());

        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            tasks_ui::init(cx);
        });
        let (workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::Workspace::test_new(project.clone(), window, cx)
        });
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = tasks_ui::TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load cold remote Tests panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel, window, cx);
            workspace.toggle_panel_focus::<tasks_ui::TestsPanel>(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(metadata_requests.lock().expect("runner lock").len(), 1);
        assert_eq!(discovery_requests.lock().expect("runner lock").len(), 1);
        let provider_after = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&provider_id)
                .cloned()
        });
        assert!(
            provider_after.as_ref().is_some_and(|provider| provider
                .nodes
                .iter()
                .any(|node| node.label == "host_case")),
            "cold panel must publish the host's authoritative case: {provider_after:?}"
        );
        let initial_generation = provider_after
            .expect("current host provider")
            .discovery_generation;
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        structured.update(cx, |_, cx| {
            cx.emit(
                project::structured_execution::StructuredExecutionStoreEvent::Changed(
                    provider_id.clone(),
                ),
            );
        });
        cx.run_until_parked();
        assert_eq!(metadata_requests.lock().expect("runner lock").len(), 1);
        assert_eq!(discovery_requests.lock().expect("runner lock").len(), 1);
        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .rust_test_provider_store
                .update(cx, |store, _| store.gate_next_discovery_reply_for_test())
        });
        cx.dispatch_action(language_tools::language_tool_tree::Refresh);
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "Tests-panel Refresh must reach the registered host before reply gating"
        );
        let remote = project.read_with(cx, |project, _| {
            project.remote_client().expect("remote client")
        });
        remote.update(cx, |remote, cx| remote.force_heartbeat_timeout(1, cx));
        {
            let recovery = async {
                while remote.read_with(cx, |remote, _| remote.connection_state())
                    != remote::ConnectionState::Connected
                {
                    cx.background_executor
                        .timer(std::time::Duration::from_millis(10))
                        .await;
                }
            };
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5));
            futures::pin_mut!(recovery);
            assert!(
                matches!(
                    futures::future::select(recovery, deadline).await,
                    futures::future::Either::Left(_)
                ),
                "remote client must reconnect"
            );
        }
        release
            .send(())
            .expect("the old registered-host reply remains active");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(completed, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "the old registered-host reply must complete"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
        let stale_provider = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&provider_id)
                .cloned()
        });
        let stale_provider = stale_provider.expect("cached provider after reconnect");
        assert_eq!(stale_provider.discovery_generation, initial_generation);
        assert_eq!(
            stale_provider.status,
            project::structured_execution::StructuredProviderStatus::Stale
        );
        cx.dispatch_action(language_tools::language_tool_tree::Refresh);
        cx.run_until_parked();
        let refreshed_provider = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&provider_id)
                .cloned()
        });
        let refreshed_provider = refreshed_provider.expect("new host provider");
        assert!(refreshed_provider.discovery_generation > initial_generation);
        assert!(
            matches!(
                refreshed_provider.status,
                project::structured_execution::StructuredProviderStatus::Current
                    | project::structured_execution::StructuredProviderStatus::Partial
            ) && refreshed_provider
                .nodes
                .iter()
                .any(|node| node.label == "host_case"),
            "fresh host discovery must replace the stale provider: {refreshed_provider:?}"
        );
        let recovered_generation = refreshed_provider.discovery_generation;
        let completed_discoveries = discovery_requests.lock().expect("runner lock").len();
        cx.dispatch_action(language_tools::language_tool_tree::Refresh);
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(active_started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "the panel must start the gated host runner"
        );
        assert_eq!(
            discovery_requests.lock().expect("runner lock").len(),
            completed_discoveries,
            "the gated runner has not produced a test snapshot"
        );
        cx.dispatch_action(language_tools::language_tool_tree::Refresh);
        cx.run_until_parked();
        assert_eq!(
            discovery_requests.lock().expect("runner lock").len(),
            completed_discoveries + 1,
            "the replacement panel refresh must run on the host"
        );
        assert!(
            active_release.send(()).is_err(),
            "superseding the active panel refresh must cancel its host runner"
        );
        let replacement_provider = project.read_with(cx, |project, cx| {
            project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&provider_id)
                .cloned()
        });
        let replacement_provider = replacement_provider.expect("replacement host provider");
        assert!(replacement_provider.discovery_generation > recovered_generation);
        assert!(
            replacement_provider
                .nodes
                .iter()
                .any(|node| node.label == "host_case"),
            "replacement host discovery must retain the authoritative case: {replacement_provider:?}"
        );
        cx.deactivate_window();
        workspace.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test(iterations = 10)]
    async fn rust_tools_headless_runtime_rerun_rejects_late_host_plan(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use project::structured_execution::{
            StructuredExecutionEvent, StructuredExecutionStoreEvent, StructuredNodeState,
            StructuredRun, StructuredRunId, StructuredRunPhase,
        };

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\nedition = \"2021\"\n",
                "src": {"main.rs": "#[test]\nfn host_case() {}\n"}
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let discovery_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (started_sender, started_receiver) = futures::channel::oneshot::channel();
        let (release_sender, release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(NthGatedDiscoveryRunner {
                        requests: discovery_requests.clone(),
                        calls: AtomicUsize::new(0),
                        gated_index: 1,
                        started: std::sync::Mutex::new(Some(started_sender)),
                        release: std::sync::Mutex::new(Some(release_receiver)),
                    }),
                    cx,
                );
            });
        });
        project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote Rust worktree");
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            tasks_ui::init(cx);
        });
        let (workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::Workspace::test_new(project.clone(), window, cx)
        });
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = tasks_ui::TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("remote Tests panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel, window, cx);
            workspace.toggle_panel_focus::<tasks_ui::TestsPanel>(window, cx);
        });
        cx.run_until_parked();
        assert_eq!(discovery_requests.lock().expect("runner lock").len(), 1);

        let provider_id = RustTestProviderStore::provider_id();
        let structured = project.read_with(cx, |project, _| {
            project.structured_execution_store().clone()
        });
        let snapshot = structured
            .read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
            .expect("client Rust provider");
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("host case")
            .id
            .clone();
        let failed_run_id = StructuredRunId("remote-rerun-failed".to_string());
        structured.update(cx, |store, cx| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &provider_id,
                    StructuredRun::new(
                        failed_run_id.clone(),
                        snapshot.discovery_generation,
                        vec![case.clone()],
                    ),
                )
                .expect("client failed run");
            store
                .state_mut()
                .apply_event(
                    project_generation,
                    &provider_id,
                    &failed_run_id,
                    StructuredExecutionEvent {
                        sequence: 0,
                        node_id: case.clone(),
                        state: StructuredNodeState::Failed,
                        duration_millis: None,
                        message: Some("fixture failure".to_string()),
                        location: None,
                    },
                    None,
                )
                .expect("client failed case");
            store
                .state_mut()
                .set_run_phase(
                    project_generation,
                    &provider_id,
                    &failed_run_id,
                    StructuredRunPhase::Completed,
                    None,
                )
                .expect("client failed run complete");
            cx.emit(StructuredExecutionStoreEvent::Changed(provider_id.clone()));
        });
        cx.run_until_parked();
        let client_provider = structured
            .read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
            .expect("failed client projection");
        assert_eq!(
            tasks_ui::project_test_explorer(
                &[client_provider],
                &tasks_ui::TestExplorerFilter::default()
            )
            .failed()
            .len(),
            1
        );
        let initial_notifications =
            workspace.read_with(cx, |workspace, _| workspace.notification_ids().len());

        let host_refresh = headless.update(server_cx, |headless, cx| {
            headless
                .rust_test_provider_store
                .update(cx, |store, cx| store.refresh(cx))
        });
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        assert_eq!(
            structured.read_with(cx, |store, _| {
                store
                    .state()
                    .provider(&provider_id)
                    .expect("client provider during held host refresh")
                    .discovery_generation
            }),
            snapshot.discovery_generation,
            "the client must retain its failed projection while the host advances"
        );
        cx.dispatch_action(tasks_ui::RerunFailedTests);
        for _ in 0..100 {
            cx.run_until_parked();
            if workspace.read_with(cx, |workspace, _| workspace.notification_ids().len())
                > initial_notifications
            {
                break;
            }
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        assert!(
            workspace.read_with(cx, |workspace, _| workspace.notification_ids().len())
                > initial_notifications,
            "rejected host plan must show the skipped-count notification"
        );
        let after = structured
            .read_with(cx, |store, _| store.state().provider(&provider_id).cloned())
            .expect("client provider after rejected plan");
        assert_eq!(
            after.current_run.as_ref().map(|run| &run.id),
            Some(&failed_run_id),
            "a rejected host plan must not schedule a new client Task"
        );
        assert!(
            headless
                .read_with(server_cx, |headless, cx| {
                    headless
                        .structured_execution_store
                        .read(cx)
                        .state()
                        .provider(&provider_id)
                        .and_then(|provider| provider.current_run.clone())
                })
                .is_none(),
            "a rejected host plan must not start a host run"
        );
        release_sender.send(()).expect("release held host refresh");
        host_refresh.await.expect("host discovery after rejection");
        cx.deactivate_window();
        workspace.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn rust_tools_headless_runtime_projects_host_lsp_source_location(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use gpui::BorrowAppContext;
        use project::lsp_store::lsp_ext_command::{
            CargoRunnableArgs, Runnable, RunnableArgs, Runnables,
        };

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\nedition = \"2021\"\n",
                "src": {"main.rs": "#[test]\nfn host_case() {}\n"}
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(Arc::new(std::sync::Mutex::new(Vec::new())))),
                    cx,
                );
                store.observe_lsp_store(&headless.lsp_store, cx);
            });
        });
        project.update(cx, |project, _| {
            project.languages().add(rust_lang());
            project.languages().register_fake_lsp_adapter(
                "Rust",
                FakeLspAdapter {
                    name: "rust-analyzer",
                    ..FakeLspAdapter::default()
                },
            );
        });
        let mut fake_servers = server_cx.update(|cx| {
            headless.read(cx).languages.register_fake_lsp_server(
                LanguageServerName("rust-analyzer".into()),
                lsp::ServerCapabilities::default(),
                None,
            )
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("remote Rust worktree");
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
        let (_buffer, _handle) = project
            .update(cx, |project, cx| {
                project.open_buffer_with_lsp((worktree_id, rel_path("src/main.rs")), cx)
            })
            .await
            .expect("remote Rust buffer");
        let fake_server = fake_servers.next().await.expect("host Rust Analyzer");
        let request_count = Arc::new(AtomicUsize::new(0));
        fake_server.set_request_handler::<Runnables, _, _>({
            let request_count = request_count.clone();
            move |_, _| {
                request_count.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok(vec![Runnable {
                        label: "test host_case".into(),
                        location: Some(lsp::LocationLink {
                            origin_selection_range: None,
                            target_uri: lsp::Uri::from_file_path(path!("/standalone/src/main.rs"))
                                .expect("host source URI"),
                            target_range: lsp::Range::new(
                                lsp::Position::new(1, 0),
                                lsp::Position::new(1, 17),
                            ),
                            target_selection_range: lsp::Range::new(
                                lsp::Position::new(1, 0),
                                lsp::Position::new(1, 17),
                            ),
                        }),
                        args: RunnableArgs::Cargo(CargoRunnableArgs {
                            environment: HashMap::default(),
                            cwd: path!("/standalone").into(),
                            override_cargo: None,
                            workspace_root: None,
                            cargo_args: vec![
                                "test".into(),
                                "--package".into(),
                                "standalone".into(),
                                "--bin".into(),
                                "standalone".into(),
                            ],
                            executable_args: vec!["host_case".into(), "--exact".into()],
                        }),
                    }])
                }
            }
        });
        cx.run_until_parked();
        server_cx.run_until_parked();
        let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let snapshot = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("remote Rust discovery with host source");
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            headless.read_with(server_cx, |headless, cx| {
                headless
                    .rust_test_provider_store
                    .read(cx)
                    .current_semantic_hint_count(cx)
            }),
            1,
            "the headless host must retain the source hint"
        );
        assert_eq!(
            tests.read_with(cx, |store, cx| store.current_semantic_hint_count(cx)),
            0,
            "the remote client must not own a local semantic hint"
        );
        let case = snapshot
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("host test case");
        assert_eq!(
            case.path.as_ref().map(|path| path.path.as_unix_str()),
            Some("src/main.rs")
        );

        let (started, release, completed) = headless.update(server_cx, |headless, cx| {
            headless
                .rust_test_provider_store
                .update(cx, |store, _| store.gate_next_discovery_reply_for_test())
        });
        let pending = tests.update(cx, |store, cx| store.refresh(cx));
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        assert!(
            matches!(
                futures::future::select(started, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "host must assemble the source-bearing reply before revocation"
        );
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/main.rs".into()]));
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        headless.read_with(server_cx, |headless, cx| {
            let host_worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(worktree_id, cx)
                .expect("host Rust worktree");
            assert!(
                host_worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("src/main.rs"))
                    .expect("host Rust source entry")
                    .is_private
            );
        });
        release.send(()).expect("release the pre-revocation reply");
        completed.await.expect("host released the discovery reply");
        let delayed_snapshot = pending
            .await
            .expect("client rechecks source visibility after the delayed host reply");
        assert!(
            delayed_snapshot
                .nodes
                .iter()
                .filter(|node| node.label == "host_case")
                .all(|node| node.path.is_none()),
            "the completed old host response must not reintroduce a private source: {delayed_snapshot:?}"
        );
        let private_snapshot = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("remote Rust discovery after source became private");
        let private_case = private_snapshot
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("host test case remains discoverable");
        assert!(
            private_case.path.is_none(),
            "private host source must not reach the remote Tests panel: {private_case:?}"
        );
    }

    #[gpui::test]
    async fn rust_tools_headless_runtime_dispatches_to_authoritative_runners(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            path!("/standalone"),
            json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\nedition = \"2021\"\n",
                "src": {"main.rs": "fn main() {}"}
            }),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        let metadata_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let discovery_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        headless.update(server_cx, |headless, cx| {
            headless.cargo_workspace_store.update(cx, |store, cx| {
                *store = CargoWorkspaceStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    Arc::new(MetadataRunner(metadata_requests.clone())),
                );
                store.shared(
                    proto::REMOTE_SERVER_PROJECT_ID,
                    headless.session.clone(),
                    cx,
                );
            });
            headless.rust_test_provider_store.update(cx, |store, cx| {
                *store = RustTestProviderStore::local_with_runner(
                    headless.worktree_store.clone(),
                    headless.environment.clone(),
                    headless.cargo_workspace_store.clone(),
                    headless.structured_execution_store.clone(),
                    Arc::new(DiscoveryRunner(discovery_requests.clone())),
                    cx,
                );
            });
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/standalone"), true, cx)
            })
            .await
            .expect("the remote worktree should open");
        cx.run_until_parked();
        let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
        let snapshot = cargo
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("the registered Cargo handler should reply over RPC");
        assert_eq!(snapshot.workspaces.len(), 1, "{snapshot:?}");
        assert_eq!(
            metadata_requests.lock().expect("runner request lock").len(),
            1
        );
        assert_eq!(
            metadata_requests
                .lock()
                .expect("runner request lock")
                .first()
                .expect("host metadata request")
                .working_directory,
            PathBuf::from(path!("/standalone"))
        );
        let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
        let snapshot = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("discovery and structured result handlers should reply over RPC");
        assert_eq!(snapshot.provider_id, RustTestProviderStore::provider_id());
        assert!(
            snapshot.nodes.iter().any(|node| node.kind
                == project::structured_execution::StructuredNodeKind::Case
                && node.label == "host_case"
                && node.debug_capability
                    == Some(project::structured_execution::StructuredDebugCapability::Supported)),
            "{snapshot:?}"
        );
        assert_eq!(
            metadata_requests.lock().expect("runner request lock").len(),
            2
        );
        assert_eq!(
            discovery_requests
                .lock()
                .expect("runner request lock")
                .len(),
            1
        );
        assert_eq!(
            discovery_requests
                .lock()
                .expect("runner request lock")
                .first()
                .expect("host discovery request")
                .working_directory
                .as_deref(),
            Some(Path::new(path!("/standalone")))
        );
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote project client")
                .read(cx)
                .proto_client()
        });
        let mismatch = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 100,
                protocol_version: u32::MAX,
                worktree_ids: Vec::new(),
            })
            .await
            .expect("mismatch should have a typed response");
        assert_eq!(
            mismatch.status,
            proto::StructuredProviderStatus::Mismatch as i32
        );
        assert_eq!(mismatch.request_id, 100);
        assert_eq!(
            metadata_requests.lock().expect("runner request lock").len(),
            2
        );
        assert_eq!(
            discovery_requests
                .lock()
                .expect("runner request lock")
                .len(),
            1
        );
        client
            .request(proto::CancelRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 101,
            })
            .await
            .expect("the cancellation handler should acknowledge the request");
        let cancelled = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 101,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: Vec::new(),
            })
            .await
            .expect_err("a pre-cancelled request must not succeed");
        assert!(cancelled.to_string().contains("cancelled"), "{cancelled:#}");

        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        headless.update(server_cx, |headless, cx| {
            project::trusted_worktrees::init(HashMap::default(), cx);
            project::trusted_worktrees::track_worktree_trust(
                headless.worktree_store.clone(),
                None,
                None,
                None,
                cx,
            );
        });
        client
            .request(proto::RestrictWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("registered restriction handler");
        let restricted = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 102,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("restricted discovery response");
        assert_eq!(
            restricted.status,
            proto::StructuredProviderStatus::Restricted as i32
        );
        assert_eq!(discovery_requests.lock().expect("runner lock").len(), 1);
        assert_eq!(metadata_requests.lock().expect("runner lock").len(), 2);
        client
            .request(proto::TrustWorktrees {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                trusted_paths: vec![proto::PathTrust {
                    content: Some(proto::path_trust::Content::WorktreeId(worktree_id)),
                }],
            })
            .await
            .expect("registered trust handler");
        let trusted = client
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 103,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("trusted discovery response");
        assert_ne!(
            trusted.status,
            proto::StructuredProviderStatus::Restricted as i32
        );
        assert_eq!(discovery_requests.lock().expect("runner lock").len(), 2);

        let cancelling_peer = proto::PeerId { owner_id: 7, id: 1 };
        let other_peer = proto::PeerId { owner_id: 7, id: 2 };
        let metadata_count = metadata_requests.lock().expect("runner lock").len();
        client
            .request(super::peer_requests::FromPeer {
                peer: cancelling_peer,
                payload: proto::CancelCargoWorkspace {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    request_id: 301,
                },
            })
            .await
            .expect("peer-scoped Cargo cancellation");
        let request = proto::GetCargoWorkspace {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 301,
            worktree_ids: vec![worktree_id],
        };
        let other_result = client
            .request(super::peer_requests::FromPeer {
                peer: other_peer,
                payload: request.clone(),
            })
            .await
            .expect("another peer's Cargo request must not be cancelled");
        assert_eq!(other_result.workspaces.len(), 1);
        assert!(
            client
                .request(super::peer_requests::FromPeer {
                    peer: cancelling_peer,
                    payload: request,
                })
                .await
                .expect_err("the original peer's cancellation remains effective")
                .to_string()
                .contains("cancelled")
        );
        assert_eq!(
            metadata_requests.lock().expect("runner lock").len(),
            metadata_count + 1
        );

        client
            .request(super::peer_requests::FromPeer {
                peer: cancelling_peer,
                payload: proto::CancelRustTestDiscovery {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    request_id: 401,
                },
            })
            .await
            .expect("peer-scoped discovery cancellation");
        let request = proto::GetRustTestDiscovery {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 401,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id],
        };
        let other_result = client
            .request(super::peer_requests::FromPeer {
                peer: other_peer,
                payload: request.clone(),
            })
            .await
            .expect("another peer's discovery must not be cancelled");
        assert_ne!(
            other_result.status,
            proto::StructuredProviderStatus::Restricted as i32
        );
        assert!(
            client
                .request(super::peer_requests::FromPeer {
                    peer: cancelling_peer,
                    payload: request,
                })
                .await
                .expect_err("the original peer's discovery cancellation remains effective")
                .to_string()
                .contains("cancelled")
        );
        assert_eq!(discovery_requests.lock().expect("runner lock").len(), 3);

        let discovered = client
            .request(proto::GetStructuredExecutionSnapshot {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                provider_id: RustTestProviderStore::provider_id().0,
                discovery_generation: other_result.discovery_generation,
                page_start: 0,
                page_size: 100,
                worktree_ids: vec![worktree_id],
            })
            .await
            .expect("latest authoritative discovery nodes");
        let node = discovered
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("discovered executable case");
        let plan = client
            .request(super::peer_requests::FromPeer {
                peer: other_peer,
                payload: proto::ResolveRustTestAction {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version:
                        project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                    discovery_generation: other_result.discovery_generation,
                    node_id: node.node_id.clone(),
                    action: proto::RustTestActionKind::Run as i32,
                    worktree_ids: vec![worktree_id],
                    authorization_generation: 0,
                },
            })
            .await
            .expect("authorize the selected peer's Task plan");
        #[derive(serde::Deserialize)]
        struct RunIdentity {
            run_id: String,
            scope_node_ids: Vec<String>,
        }
        let identity: RunIdentity =
            serde_json::from_str(&plan.plan_json).expect("Task plan identity");
        let queued = proto::UpdateRustTestRun {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: other_result.discovery_generation,
            run_id: identity.run_id,
            scope_node_ids: identity.scope_node_ids,
            task_id: "peer-owned-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id],
            ..Default::default()
        };
        assert!(
            client
                .request(super::peer_requests::FromPeer {
                    peer: cancelling_peer,
                    payload: queued.clone(),
                })
                .await
                .expect_err("a different original peer cannot update the Task lifecycle")
                .to_string()
                .contains("Unknown or expired")
        );
        client
            .request(super::peer_requests::FromPeer {
                peer: other_peer,
                payload: queued,
            })
            .await
            .expect("the authorized original peer can queue its Task");

        let metadata_before = metadata_requests.lock().expect("runner lock").len();
        let discovery_before = discovery_requests.lock().expect("runner lock").len();
        let refreshed = tests
            .update(cx, |store, cx| store.refresh(cx))
            .await
            .expect("a repeated remote Tests refresh reaches the host");
        assert!(refreshed.discovery_generation > snapshot.discovery_generation);
        assert!(
            refreshed.nodes.iter().any(|node| node.label == "host_case"),
            "repeated discovery preserves the authoritative case"
        );
        assert_eq!(
            metadata_requests.lock().expect("runner lock").len(),
            metadata_before + 1
        );
        assert_eq!(
            discovery_requests.lock().expect("runner lock").len(),
            discovery_before + 1
        );

        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            tasks_ui::init(cx);
        });
        let (workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::Workspace::test_new(project.clone(), window, cx)
        });
        let async_cx = cx.update(|window, cx| window.to_async(cx));
        let panel = tasks_ui::TestsPanel::load(workspace.downgrade(), async_cx)
            .await
            .expect("load remote Tests panel");
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.add_panel(panel.clone(), window, cx);
            workspace.toggle_panel_focus::<tasks_ui::TestsPanel>(window, cx);
        });
        cx.run_until_parked();
        let metadata_before_panel = metadata_requests.lock().expect("runner lock").len();
        let discovery_before_panel = discovery_requests.lock().expect("runner lock").len();
        cx.dispatch_action(language_tools::language_tool_tree::Refresh);
        cx.run_until_parked();
        assert_eq!(
            metadata_requests.lock().expect("runner lock").len(),
            metadata_before_panel + 1,
            "Tests-panel Refresh must run Cargo metadata on the host"
        );
        assert_eq!(
            discovery_requests.lock().expect("runner lock").len(),
            discovery_before_panel + 1,
            "Tests-panel Refresh must run Rust discovery on the host"
        );
        cx.deactivate_window();
        workspace.update_in(cx, |_, window, _| window.remove_window());
    }
}

#[cfg(not(feature = "rust-tools"))]
#[gpui::test]
async fn rust_tools_headless_runtime_disabled_has_no_handlers(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/fixture",
        json!({"profile.json": "{}", "profile.syms.json": "{}"}),
    )
    .await;
    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/fixture", true, cx)
        })
        .await
        .expect("remote fixture");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let paths = project.read_with(cx, |project, cx| {
        [
            project
                .find_project_path("/fixture/profile.json", cx)
                .expect("profile path"),
            project
                .find_project_path("/fixture/profile.syms.json", cx)
                .expect("symbols path"),
        ]
    });
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let native = project
        .read_with(cx, |project, cx| {
            project.import_remote_profile(
                paths,
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("native request task")
        .await
        .expect_err("unsupported native host");
    assert_eq!(
        native.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );
    let client = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    let cargo = client
        .request(proto::GetCargoWorkspace {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 1,
            worktree_ids: Vec::new(),
        })
        .await
        .expect_err("disabled Cargo handler");
    let discovery = client
        .request(proto::GetRustTestDiscovery {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            request_id: 2,
            protocol_version: 1,
            worktree_ids: Vec::new(),
        })
        .await
        .expect_err("disabled discovery handler");
    let results = client
        .request(proto::GetStructuredExecutionSnapshot {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            provider_id: "rust-tests".to_string(),
            discovery_generation: 1,
            page_start: 0,
            page_size: 1,
            worktree_ids: Vec::new(),
        })
        .await
        .expect_err("disabled structured-result handler");
    let profile_import = client
        .request(proto::ImportProfileResults {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 3,
            protocol_version: 1,
            profile_path: Some(proto::ProjectPath {
                worktree_id: 1,
                path: "profile.json".into(),
            }),
            symbols_path: Some(proto::ProjectPath {
                worktree_id: 1,
                path: "profile.syms.json".into(),
            }),
            consent_to_transfer: true,
            task_run_id: None,
            artifact_format: "samply-0.13.1".into(),
        })
        .await
        .expect_err("disabled native import handler");
    let profile_page = client
        .request(proto::GetProfileResultPage {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 3,
            generation: 1,
            protocol_version: 1,
            page_index: 0,
        })
        .await
        .expect_err("disabled native page handler");
    let profile_cancel = client
        .request(proto::CancelProfileImport {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 3,
            generation: 1,
        })
        .await
        .expect_err("disabled native cancellation handler");
    for error in [
        cargo,
        discovery,
        results,
        profile_import,
        profile_page,
        profile_cancel,
    ] {
        assert!(
            error.to_string().contains("no handler registered"),
            "{error:#}"
        );
    }
    let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
    let error = cargo
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect_err("unsupported Cargo host");
    assert_eq!(
        error
            .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
            .expect("typed capability error")
            .kind,
        project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::UnsupportedHost
    );
    let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
    let error = tests
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect_err("unsupported discovery host");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed Rust discovery capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedDiscoveryHost
    );
    assert!(
        error
            .to_string()
            .contains("does not support Rust test discovery"),
        "{error:#}"
    );
    let error = tests
        .update(cx, |store, cx| {
            store.plan_action(
                &project::structured_execution::StructuredNodeId("disabled-host-case".into()),
                project::structured_execution::DiscoveryGeneration(1),
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect_err("unsupported Rust action host");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed Rust action capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedActionHost
    );
    let structured = project.read_with(cx, |project, _| {
        project.structured_execution_store().clone()
    });
    let error = structured
        .update(cx, |store, cx| {
            store.refresh_provider(
                project::structured_execution::StructuredProviderId("disabled-host".into()),
                cx,
            )
        })
        .await
        .expect_err("unsupported structured-results host");
    assert!(
        error
            .downcast_ref::<project::structured_execution::StructuredExecutionUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );
}

#[cfg(all(feature = "rust-tools-process-test", target_os = "macos"))]
#[gpui::test]
async fn rust_tools_headless_runtime_separate_disabled_host_typed_client(cx: &mut TestAppContext) {
    use remote::{ConnectionIdentifier, MockDelegate, ProcessRemoteConnection, RemoteConnection};

    let binary = std::env::var_os("RUST_TOOLS_DISABLED_HOST_BINARY")
        .map(PathBuf::from)
        .expect(
            "set RUST_TOOLS_DISABLED_HOST_BINARY to a separately built no-Rust-tools remote_server",
        );
    let server_state_dir = std::env::var_os("RUST_TOOLS_DISABLED_HOST_STATE_DIR")
        .map(PathBuf::from)
        .expect("set RUST_TOOLS_DISABLED_HOST_STATE_DIR to that binary's server_state directory");
    assert!(binary.is_file(), "missing disabled host binary: {binary:?}");
    assert!(
        server_state_dir.is_dir(),
        "missing disabled host state directory: {server_state_dir:?}"
    );
    let fixture = tempfile::tempdir_in("/private/tmp").expect("create host fixture");
    std::fs::create_dir(fixture.path().join("src")).expect("create source directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        "[package]\nname = \"disabled_host_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo manifest");
    std::fs::write(fixture.path().join("src/lib.rs"), "pub fn fixture() {}\n")
        .expect("write source");
    std::fs::write(fixture.path().join("profile.json"), "{}").expect("write profile artifact");
    std::fs::write(fixture.path().join("profile.syms.json"), "{}").expect("write symbol artifact");
    std::fs::write(fixture.path().join("coverage.json"), "{}").expect("write coverage artifact");

    cx.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
    init_logger();
    cx.executor().allow_parking();
    let connection = Arc::new(
        ProcessRemoteConnection::new(binary, server_state_dir).with_isolated_server_namespace(),
    );
    let (_cancel_tx, cancel_rx) = futures::channel::oneshot::channel();
    let client = cx
        .update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::Workspace(-i64::from(std::process::id()) - 1),
                connection.clone(),
                cancel_rx,
                Arc::new(MockDelegate),
                cx,
            )
        })
        .await
        .expect("connect to separate disabled host")
        .expect("remote client");
    let project = build_project(client.clone(), cx);
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        );
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(fixture.path(), true, cx)
        })
        .await
        .expect("remote host fixture worktree");
    let initial_scan = worktree.update(cx, |worktree, _| worktree.wait_for_snapshot(1));
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    match futures::future::select(Box::pin(initial_scan), Box::pin(deadline)).await {
        futures::future::Either::Left((result, _)) => {
            result.expect("client receives the disabled host's initial fixture scan");
        }
        futures::future::Either::Right(_) => panic!("disabled host fixture scan timed out"),
    }
    cx.run_until_parked();
    assert_eq!(
        project.read_with(cx, |project, cx| project.visible_worktrees(cx).count()),
        1,
        "disabled host fixture must have one visible completed worktree"
    );
    for path in ["Cargo.toml", "src/lib.rs"] {
        assert!(worktree.read_with(cx, |worktree, _| {
            worktree.snapshot().entry_for_path(rel_path(path)).is_some()
        }));
    }

    let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
    let error = cargo
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect_err("disabled host cannot serve Cargo workspace");
    assert_eq!(
        error
            .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
            .expect("typed Cargo capability error")
            .kind,
        project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::UnsupportedHost
    );

    let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
    let error = tests
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect_err("disabled host cannot serve Rust discovery");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed discovery capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedDiscoveryHost
    );
    let error = tests
        .update(cx, |store, cx| {
            store.plan_action(
                &project::structured_execution::StructuredNodeId("disabled-host-case".into()),
                project::structured_execution::DiscoveryGeneration(1),
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect_err("disabled host cannot resolve a Rust test action");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed action capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedActionHost
    );

    let structured = project.read_with(cx, |project, _| {
        project.structured_execution_store().clone()
    });
    let error = structured
        .update(cx, |store, cx| {
            store.refresh_provider(
                project::structured_execution::StructuredProviderId("rust-tests".into()),
                cx,
            )
        })
        .await
        .expect_err("disabled host cannot serve structured results");
    assert!(
        error
            .downcast_ref::<project::structured_execution::StructuredExecutionUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );

    let coverage_path = project.read_with(cx, |project, cx| {
        project
            .find_project_path(fixture.path().join("coverage.json"), cx)
            .expect("host coverage path")
    });
    let coverage = project.read_with(cx, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let error = coverage
        .update(cx, |store, cx| {
            store.interpret_artifact(
                coverage_path,
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect_err("disabled host cannot interpret Rust coverage");
    assert!(
        error
            .downcast_ref::<project::source_coverage::SourceCoverageUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );

    let paths = project.read_with(cx, |project, cx| {
        [
            project
                .find_project_path(fixture.path().join("profile.json"), cx)
                .expect("host profile path"),
            project
                .find_project_path(fixture.path().join("profile.syms.json"), cx)
                .expect("host symbols path"),
        ]
    });
    let result = project
        .read_with(cx, |project, cx| {
            project.prepare_remote_profile_task("disabled-host-profile".into(), paths, cx)
        })
        .expect("profile request task")
        .await;
    let error = match result {
        Ok(_) => panic!("disabled host has no native profile Task helper"),
        Err(error) => error,
    };
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );

    let shutdown = client.update(cx, |client, cx| {
        client.shutdown_processes::<proto::ShutdownRemoteServer>(
            None,
            cx.background_executor().clone(),
        )
    });
    if let Some(shutdown) = shutdown {
        shutdown.await;
    }
    connection
        .kill()
        .await
        .expect("stop separate disabled host");
}

#[cfg(all(feature = "rust-tools-process-test", target_os = "macos"))]
#[gpui::test]
async fn rust_tools_headless_runtime_separate_enabled_host_profile_and_coverage(
    cx: &mut TestAppContext,
) {
    use remote::{ConnectionIdentifier, MockDelegate, ProcessRemoteConnection, RemoteConnection};
    use std::os::unix::fs::PermissionsExt as _;
    use std::sync::atomic::AtomicBool;

    let binary = std::env::var_os("RUST_TOOLS_ENABLED_HOST_BINARY")
        .map(PathBuf::from)
        .expect(
            "set RUST_TOOLS_ENABLED_HOST_BINARY to a separately built Rust-tools remote_server",
        );
    let server_state_dir = std::env::var_os("RUST_TOOLS_ENABLED_HOST_STATE_DIR")
        .map(PathBuf::from)
        .expect("set RUST_TOOLS_ENABLED_HOST_STATE_DIR to that binary's server_state directory");
    assert!(binary.is_file(), "missing enabled host binary: {binary:?}");
    assert!(
        server_state_dir.is_dir(),
        "missing enabled host state directory: {server_state_dir:?}"
    );
    let fixture = tempfile::tempdir_in("/private/tmp").expect("create host fixture");
    std::fs::create_dir(fixture.path().join("src")).expect("create Cargo source directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        "[package]\nname = \"enabled_host_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo manifest");
    std::fs::write(
        fixture.path().join("src/lib.rs"),
        "pub fn fixture() {}\n#[cfg(test)] mod tests { #[test] fn unit_passes() {} }\n",
    )
    .expect("write Cargo source");
    std::fs::write(fixture.path().join("workload.rs"), "fn main() {}\n")
        .expect("write mapped source");
    std::fs::write(
        fixture.path().join("profile.json"),
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        ),
    )
    .expect("write captured profile");
    let symbols = include_str!(
        "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"
    )
    .replace("/fixture/workload.rs", &fixture.path().join("workload.rs").to_string_lossy());
    std::fs::write(fixture.path().join("profile.syms.json"), symbols)
        .expect("write captured symbols");
    let coverage = json!({
        "type": "llvm.coverage.json.export",
        "version": "2.0.1",
        "cargo_llvm_cov": {"version": "0.9.0", "manifest_path": fixture.path().join("Cargo.toml")},
        "data": [{"files": [{
            "filename": fixture.path().join("workload.rs"),
            "segments": [[1, 1, 1, true, true, false]],
            "summary": {"lines": {"count": 1, "covered": 1}}
        }]}]
    });
    std::fs::write(fixture.path().join("coverage.json"), coverage.to_string())
        .expect("write coverage report");
    let capture_script = fixture.path().join("emit-profile.sh");
    std::fs::write(
        &capture_script,
        "#!/bin/sh\nset -e\ncp \"$1\" \"$2\"\ncp \"$3\" \"$4\"\n",
    )
    .expect("write explicit profile Task script");
    std::fs::set_permissions(&capture_script, std::fs::Permissions::from_mode(0o755))
        .expect("make profile Task script executable");

    cx.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
    cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
    init_logger();
    cx.executor().allow_parking();
    let local_cargo_calls_before =
        project::cargo_workspace_store::local_cargo_process_calls_for_test();
    let local_discovery_calls_before =
        project::rust_test_provider::local_rust_test_discovery_calls_for_test();
    let local_coverage_interpret_calls_before =
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test();
    let local_profile_loads_before = project::profile_results::samply_profile_load_calls_for_test();
    let host_data_directory = fixture.path().join("host-data");
    std::fs::create_dir(&host_data_directory).expect("create owned host data directory");
    std::fs::create_dir(host_data_directory.join("config"))
        .expect("create owned host configuration");
    std::fs::write(host_data_directory.join("config/settings.json"), r#"{"feature_flags":{"rust-coverage-experimental":"on","rust-native-profiling-experimental":"on"}}"#).expect("write explicit local host experimental opt-in");
    let connection = Arc::new(
        ProcessRemoteConnection::new(binary, server_state_dir)
            .with_test_data_directory(host_data_directory)
            .with_isolated_server_namespace(),
    );
    let (_cancel_tx, cancel_rx) = futures::channel::oneshot::channel();
    let client = cx
        .update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::Workspace(-i64::from(std::process::id())),
                connection.clone(),
                cancel_rx,
                Arc::new(MockDelegate),
                cx,
            )
        })
        .await
        .expect("connect to separate enabled host")
        .expect("remote client");
    let project = build_project_with_worktree_trust(client.clone(), true, cx);
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        );
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        );
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(fixture.path(), true, cx)
        })
        .await
        .expect("remote host fixture worktree");
    let initial_scan = worktree.update(cx, |worktree, _| worktree.wait_for_snapshot(1));
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    match futures::future::select(Box::pin(initial_scan), Box::pin(deadline)).await {
        futures::future::Either::Left((result, _)) => {
            result.expect("client receives the host's initial fixture scan");
        }
        futures::future::Either::Right(_) => panic!("host fixture scan timed out"),
    }
    cx.run_until_parked();

    let visible_worktrees =
        project.read_with(cx, |project, cx| project.visible_worktrees(cx).count());
    assert_eq!(
        visible_worktrees, 1,
        "fixture worktree must be visible to the client"
    );
    assert!(worktree.read_with(cx, |worktree, _| {
        worktree
            .snapshot()
            .entry_for_path(rel_path("Cargo.toml"))
            .is_some()
    }));
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    assert_eq!(
        cx.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust service")
                .read(cx)
                .observed_worktree_trust(&worktree_store, worktree_id, cx)
        }),
        Some(false),
        "registered remote fixture starts restricted before explicit approval"
    );
    cx.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("client trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(worktree_id))
                        .collect(),
                    cx,
                );
            });
    });
    assert_eq!(
        cx.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust service")
                .read(cx)
                .observed_worktree_trust(&worktree_store, worktree_id, cx)
        }),
        Some(true),
        "the explicit local approval trusts the registered remote fixture"
    );
    let transport = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    transport
        .request(proto::TrustWorktrees {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            trusted_paths: vec![proto::PathTrust {
                content: Some(proto::path_trust::Content::WorktreeId(
                    worktree_id.to_proto(),
                )),
            }],
        })
        .await
        .expect("host accepts explicit worktree trust");

    let structured = project.read_with(cx, |project, _| {
        project.structured_execution_store().clone()
    });
    let rust_provider = project::rust_test_provider::RustTestProviderStore::provider_id();
    let discovery_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let snapshot = structured.read_with(cx, |store, _| {
            store.state().provider(&rust_provider).cloned()
        });
        if snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.status == project::structured_execution::StructuredProviderStatus::Current
                && snapshot
                    .nodes
                    .iter()
                    .any(|node| node.label == "unit_passes")
        }) {
            break;
        }
        assert!(
            std::time::Instant::now() < discovery_deadline,
            "separate host never published Rust test discovery: {snapshot:?}"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(100))
            .await;
        cx.run_until_parked();
    }

    let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
    let metadata = cargo
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("enabled host runs Cargo metadata");
    assert_eq!(metadata.workspaces.len(), 1, "{metadata:?}");

    let coverage_path = project.read_with(cx, |project, cx| {
        project
            .find_project_path(fixture.path().join("coverage.json"), cx)
            .expect("host coverage path")
    });
    let coverage_store = project.read_with(cx, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let report = coverage_store
        .update(cx, |store, cx| {
            store.interpret_artifact(
                coverage_path,
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("enabled host interprets coverage report");
    assert_eq!(report.files.len(), 1);
    assert_eq!(report.files[0].path.path.as_unix_str(), "workload.rs");

    let paths = project.read_with(cx, |project, cx| {
        ["profile.json", "profile.syms.json"].map(|name| {
            project
                .find_project_path(fixture.path().join(name), cx)
                .expect("host profile path")
        })
    });
    let denied = project.read_with(cx, |project, cx| {
        project.import_remote_profile(paths.clone(), false, Arc::new(AtomicBool::new(false)), cx)
    });
    assert!(
        denied.is_err(),
        "profile transfer requires explicit consent"
    );
    let model = project
        .read_with(cx, |project, cx| {
            project.import_remote_profile(paths, true, Arc::new(AtomicBool::new(false)), cx)
        })
        .expect("consented profile request")
        .await
        .expect("enabled host returns normalized profile");
    assert_eq!(model.total_samples, 30);
    assert!(model.frames.iter().any(|frame| {
        frame.label == "workload::leaf"
            && frame
                .source
                .as_ref()
                .is_some_and(|(path, _)| path.path.as_unix_str() == "workload.rs")
    }));

    let task_paths = project.read_with(cx, |project, cx| {
        ["profile.json", "profile.syms.json"].map(|name| {
            project
                .find_project_path(fixture.path().join(name), cx)
                .expect("host Task artifact path")
        })
    });
    let binding = project
        .read_with(cx, |project, cx| {
            project.prepare_remote_profile_task(
                "enabled-process-task".into(),
                task_paths.clone(),
                cx,
            )
        })
        .expect("profile Task preparation request")
        .await
        .expect("enabled process host prepares observed Task");
    let mut observed_task = smol::process::Command::new(&binding.helper_path)
        .args([
            "observe-task",
            "--socket",
            &binding.socket_path,
            "--",
            "/usr/bin/true",
        ])
        .env(
            util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
            binding.token,
        )
        .kill_on_drop(true)
        .spawn()
        .expect("launch actual enabled-host observer helper");
    project
        .read_with(cx, |project, cx| {
            project.get_remote_profile_task("enabled-process-task".into(), cx)
        })
        .expect("observed Task completion request")
        .await
        .expect("separate host observes child exit");
    assert!(
        observed_task
            .status()
            .await
            .expect("observer helper exit")
            .success()
    );
    let task_model = project
        .read_with(cx, |project, cx| {
            project.import_remote_task_profile(
                task_paths,
                "enabled-process-task".into(),
                Arc::new(AtomicBool::new(false)),
                cx,
            )
        })
        .expect("Task-bound profile import request")
        .await
        .expect("separate host verifies sealed Task artifacts");
    assert_eq!(task_model.total_samples, 30);
    assert!(task_model.frames.iter().any(|frame| {
        frame.label == "workload::leaf"
            && frame
                .source
                .as_ref()
                .is_some_and(|(path, _)| path.path.as_unix_str() == "workload.rs")
    }));
    project
        .read_with(cx, |project, cx| {
            project.cancel_remote_profile_task("enabled-process-task".into(), cx)
        })
        .expect("release observed Task request")
        .await
        .expect("release observed Task binding");

    cx.update(|cx| {
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
        language_tools::init(cx);
        terminal_view::init(cx);
    });
    let (workspace, cx) = cx
        .add_window_view(|window, cx| workspace::Workspace::test_new(project.clone(), window, cx));
    let panel = terminal_view::terminal_panel::TerminalPanel::load(
        workspace.downgrade(),
        cx.update(|window, cx| window.to_async(cx)),
    )
    .await
    .expect("production terminal provider with checked Task preparation");
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.add_panel(panel, window, cx);
    });
    let ready = Rc::new(std::cell::RefCell::new(Vec::new()));
    let _subscription = cx.update(|_, cx| {
        cx.subscribe(&workspace, {
            let ready = ready.clone();
            move |_, event, _| {
                if let workspace::Event::TaskArtifactGroupReady(group) = event {
                    ready.borrow_mut().push(group.clone());
                }
            }
        })
    });
    let task_group = task::TaskArtifactGroup {
        format: "samply-0.13.1".into(),
        files: std::collections::BTreeMap::from_iter([
            (
                "profile".into(),
                task::TaskArtifact {
                    path: "task-profile.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 32 * 1024 * 1024,
                },
            ),
            (
                "symbols".into(),
                task::TaskArtifact {
                    path: "task-profile.syms.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 16 * 1024 * 1024,
                },
            ),
        ]),
    };
    let resolved = task::TaskTemplate {
        label: "Explicit process profile Task".into(),
        command: capture_script.to_string_lossy().into_owned(),
        args: [
            "profile.json",
            "task-profile.json",
            "profile.syms.json",
            "task-profile.syms.json",
        ]
        .map(|path| fixture.path().join(path).to_string_lossy().into_owned())
        .into(),
        cwd: Some(fixture.path().to_string_lossy().into_owned()),
        save: task::SaveStrategy::None,
        artifact_group: Some(task_group),
        ..Default::default()
    }
    .resolve_task("process-profile-task", &task::TaskContext::default())
    .expect("explicit process profile Task");
    let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            resolved,
            true,
            move |result, _| {
                completed_sender
                    .send(result)
                    .expect("Task completion receiver")
            },
            window,
            cx,
        );
    });
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(30));
    let approved_completion = async {
        while !cx.has_pending_prompt() {
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        let (message, detail) = cx.pending_prompt().expect("actual capture privacy review");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert_eq!(
            detail.lines().take(2).collect::<Vec<_>>(),
            [
                format!(
                    "Profile: {}",
                    fixture.path().join("task-profile.json").display()
                ),
                format!(
                    "Symbols: {}",
                    fixture.path().join("task-profile.syms.json").display()
                ),
            ]
        );
        for requirement in [
            "function names, source and library paths, and workload details",
            "Processing stays local to the authoritative host",
            "not uploaded or sent as telemetry",
            "explicitly consented remote import",
            "raw files remain on that host",
            "Delete them through the project file controls",
            "session-only result",
            "does not delete these files",
            "only to this capture",
        ] {
            assert!(detail.contains(requirement), "{requirement}");
        }
        assert!(
            ready.borrow().is_empty(),
            "approval cannot be replaced by an offer"
        );
        assert!(
            !fixture.path().join("task-profile.json").exists()
                && !fixture.path().join("task-profile.syms.json").exists(),
            "the actual capture Task cannot write either output before approval"
        );
        cx.simulate_prompt_answer("Capture");
        completed_receiver.await
    };
    let completion = match futures::future::select(Box::pin(approved_completion), deadline).await {
        futures::future::Either::Left((result, _)) => result.expect("Task completion"),
        futures::future::Either::Right(_) => panic!("process profile Task timed out"),
    };
    assert_eq!(completion, workspace::tasks::ScheduledTaskResult::Success);
    cx.run_until_parked();
    assert_eq!(ready.borrow().len(), 1, "one host-verified profile offer");
    let offer = ready.borrow()[0].clone();
    assert!(offer.host_verified());
    assert!(cx.read(|cx| offer.is_current(&project, cx)));
    let notification_id = workspace::notifications::NotificationId::unique::<
        workspace::tasks::CompletedTaskArtifactGroup,
    >();
    let notification = workspace
        .read_with(cx, |workspace, _| {
            workspace.message_notification_for_test(&notification_id)
        })
        .expect("production Task notification");
    assert!(notification.update_in(cx, |notification, window, cx| {
        notification.click_primary_for_test(window, cx)
    }));
    let view = workspace
        .read_with(cx, |workspace, cx| {
            workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
        })
        .expect("Task notification opens native viewer");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while view.read_with(cx, |view, _| view.sample_count_for_test()) != Some(30) {
        assert!(
            std::time::Instant::now() < deadline,
            "production Task viewer did not display the host profile"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
        cx.run_until_parked();
    }
    assert_eq!(
        project::cargo_workspace_store::local_cargo_process_calls_for_test(),
        local_cargo_calls_before,
        "the client must not run local Cargo or rustc for the enabled host"
    );
    assert_eq!(
        project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
        local_discovery_calls_before,
        "the client must not run local Rust test discovery for the enabled host"
    );
    assert_eq!(
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
        local_coverage_interpret_calls_before,
        "the client must not interpret host coverage locally"
    );
    assert_eq!(
        project::profile_results::samply_profile_load_calls_for_test(),
        local_profile_loads_before,
        "the client must not parse the enabled host profile locally"
    );
    cx.deactivate_window();
    workspace.update_in(cx, |_, window, _| window.remove_window());

    let shutdown = client.update(cx, |client, cx| {
        client.shutdown_processes::<proto::ShutdownRemoteServer>(
            None,
            cx.background_executor().clone(),
        )
    });
    if let Some(shutdown) = shutdown {
        shutdown.await;
    }
    connection.kill().await.expect("stop separate enabled host");
}

#[cfg(all(feature = "rust-tools-process-test", target_os = "macos"))]
async fn rust_tools_enabled_host_local_opt_in_fixture(
    host_settings: Option<&str>,
    revoke: bool,
    cx: &mut TestAppContext,
) {
    use remote::{ConnectionIdentifier, MockDelegate, ProcessRemoteConnection, RemoteConnection};
    use std::sync::atomic::AtomicBool;

    let binary = std::env::var_os("RUST_TOOLS_ENABLED_HOST_BINARY")
        .map(PathBuf::from)
        .expect(
            "set RUST_TOOLS_ENABLED_HOST_BINARY to a separately built Rust-tools remote_server",
        );
    let server_state_dir = std::env::var_os("RUST_TOOLS_ENABLED_HOST_STATE_DIR")
        .map(PathBuf::from)
        .expect("set RUST_TOOLS_ENABLED_HOST_STATE_DIR to that binary's server_state directory");
    assert!(binary.is_file(), "missing enabled host binary: {binary:?}");
    assert!(
        server_state_dir.is_dir(),
        "missing enabled host state directory: {server_state_dir:?}"
    );
    let fixture = tempfile::tempdir_in("/private/tmp").expect("create host fixture");
    std::fs::create_dir(fixture.path().join("src")).expect("create Cargo source directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        "[package]\nname = \"enabled_host_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo manifest");
    std::fs::write(
        fixture.path().join("src/lib.rs"),
        "pub fn fixture() {}\n#[cfg(test)] mod tests { #[test] fn unit_passes() {} }\n",
    )
    .expect("write Cargo source");
    std::fs::write(fixture.path().join("workload.rs"), "fn main() {}\n")
        .expect("write mapped source");
    std::fs::write(
        fixture.path().join("profile.json"),
        profile_transport_tests::paged_capture(),
    )
    .expect("write captured multi-page profile");
    let symbols = include_str!(
        "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"
    )
    .replace("/fixture/workload.rs", &fixture.path().join("workload.rs").to_string_lossy());
    std::fs::write(fixture.path().join("profile.syms.json"), symbols)
        .expect("write captured symbols");
    let coverage = json!({
        "type": "llvm.coverage.json.export",
        "version": "2.0.1",
        "cargo_llvm_cov": {"version": "0.9.0", "manifest_path": fixture.path().join("Cargo.toml")},
        "data": [{"files": [{
            "filename": fixture.path().join("workload.rs"),
            "segments": [[1, 1, 1, true, true, false]],
            "summary": {"lines": {"count": 1, "covered": 1}}
        }]}]
    });
    std::fs::write(fixture.path().join("coverage.json"), coverage.to_string())
        .expect("write coverage report");
    cx.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
    cx.update(|cx| project::trusted_worktrees::init(HashMap::default(), cx));
    init_logger();
    cx.executor().allow_parking();
    let local_cargo_calls_before =
        project::cargo_workspace_store::local_cargo_process_calls_for_test();
    let local_discovery_calls_before =
        project::rust_test_provider::local_rust_test_discovery_calls_for_test();
    let local_coverage_interpret_calls_before =
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test();
    let local_profile_loads_before = project::profile_results::samply_profile_load_calls_for_test();
    let host_data_directory = fixture.path().join("host-data");
    std::fs::create_dir(&host_data_directory).expect("create owned host data directory");
    std::fs::create_dir(host_data_directory.join("config"))
        .expect("create owned host configuration");
    let settings_path = host_data_directory.join("config/settings.json");
    if let Some(content) = host_settings {
        std::fs::write(&settings_path, content).expect("write controlled host-local settings");
    } else {
        assert!(
            !settings_path.exists(),
            "host starts without any local opt-in file"
        );
    }
    let gate_directory = fixture.path().join("page-gate");
    std::fs::create_dir(&gate_directory).expect("create owned page gate");
    let connection = Arc::new(
        ProcessRemoteConnection::new(binary, server_state_dir)
            .with_test_data_directory(host_data_directory)
            .with_profile_page_gate(gate_directory.clone())
            .with_isolated_server_namespace(),
    );
    let (_cancel_tx, cancel_rx) = futures::channel::oneshot::channel();
    let client = cx
        .update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::Workspace(-i64::from(std::process::id())),
                connection.clone(),
                cancel_rx,
                Arc::new(MockDelegate),
                cx,
            )
        })
        .await
        .expect("connect to separate enabled host")
        .expect("remote client");
    let project = build_project_with_worktree_trust(client.clone(), true, cx);
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        );
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        );
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(fixture.path(), true, cx)
        })
        .await
        .expect("remote host fixture worktree");
    let initial_scan = worktree.update(cx, |worktree, _| worktree.wait_for_snapshot(1));
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    match futures::future::select(Box::pin(initial_scan), Box::pin(deadline)).await {
        futures::future::Either::Left((result, _)) => {
            result.expect("client receives the host's initial fixture scan");
        }
        futures::future::Either::Right(_) => panic!("host fixture scan timed out"),
    }
    cx.run_until_parked();

    let visible_worktrees =
        project.read_with(cx, |project, cx| project.visible_worktrees(cx).count());
    assert_eq!(
        visible_worktrees, 1,
        "fixture worktree must be visible to the client"
    );
    assert!(worktree.read_with(cx, |worktree, _| {
        worktree
            .snapshot()
            .entry_for_path(rel_path("Cargo.toml"))
            .is_some()
    }));
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    assert_eq!(
        cx.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust service")
                .read(cx)
                .observed_worktree_trust(&worktree_store, worktree_id, cx)
        }),
        Some(false),
        "registered remote fixture starts restricted before explicit approval"
    );
    cx.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("client trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(worktree_id))
                        .collect(),
                    cx,
                );
            });
    });
    assert_eq!(
        cx.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("client trust service")
                .read(cx)
                .observed_worktree_trust(&worktree_store, worktree_id, cx)
        }),
        Some(true),
        "the explicit local approval trusts the registered remote fixture"
    );
    let transport = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    transport
        .request(proto::TrustWorktrees {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            trusted_paths: vec![proto::PathTrust {
                content: Some(proto::path_trust::Content::WorktreeId(
                    worktree_id.to_proto(),
                )),
            }],
        })
        .await
        .expect("host accepts explicit worktree trust");

    let structured = project.read_with(cx, |project, _| {
        project.structured_execution_store().clone()
    });
    let rust_provider = project::rust_test_provider::RustTestProviderStore::provider_id();
    let discovery_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let snapshot = structured.read_with(cx, |store, _| {
            store.state().provider(&rust_provider).cloned()
        });
        if snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.status == project::structured_execution::StructuredProviderStatus::Current
                && snapshot
                    .nodes
                    .iter()
                    .any(|node| node.label == "unit_passes")
        }) {
            break;
        }
        assert!(
            std::time::Instant::now() < discovery_deadline,
            "separate host never published Rust test discovery: {snapshot:?}"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(100))
            .await;
        cx.run_until_parked();
    }

    let cargo = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
    let metadata = cargo
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("enabled host runs Cargo metadata");
    assert_eq!(metadata.workspaces.len(), 1, "{metadata:?}");

    let tests = project.read_with(cx, |project, _| project.rust_test_provider_store().clone());
    let snapshot = tests
        .update(cx, |store, cx| store.refresh(cx))
        .await
        .expect("ordinary discovery remains available without experimental opt-in");
    let node = snapshot
        .nodes
        .iter()
        .find(|node| {
            node.label == "unit_passes"
                && node.kind == project::structured_execution::StructuredNodeKind::Case
        })
        .expect("actual host-discovered ordinary test");
    let plan = tests
        .update(cx, |store, cx| {
            store.plan_action(
                &node.id,
                snapshot.discovery_generation,
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect("ordinary host test action remains available");
    let project::rust_test_provider::RustTestActionPlan::Task { template, .. } = plan else {
        panic!("ordinary Run must resolve a Task");
    };
    assert!(template.required_experimental_features.is_empty());
    assert!(template.args.iter().any(|argument| argument == "--exact"));
    assert!(
        template
            .args
            .iter()
            .any(|argument| argument.ends_with("unit_passes"))
    );

    let coverage_path = project.read_with(cx, |project, cx| {
        project
            .find_project_path(fixture.path().join("coverage.json"), cx)
            .expect("host coverage path")
    });
    let coverage_store = project.read_with(cx, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let paths = project.read_with(cx, |project, cx| {
        ["profile.json", "profile.syms.json"].map(|name| {
            project
                .find_project_path(fixture.path().join(name), cx)
                .expect("host profile path")
        })
    });
    if !revoke {
        let error = coverage_store
            .update(cx, |store, cx| {
                store.interpret_artifact(
                    coverage_path,
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .err()
            .expect("enabled host without local opt-in denies experimental coverage");
        assert!(
            error
                .downcast_ref::<project::source_coverage::SourceCoverageUnsupportedHost>()
                .is_some(),
            "{error:#}"
        );
        let result = project
            .read_with(cx, |project, cx| {
                project.prepare_remote_profile_task("host-opt-out-task".into(), paths.clone(), cx)
            })
            .expect("client opt-in admits the RPC")
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("client opt-in must not confer host native Task permission"),
        };
        assert_eq!(
            error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Unsupported)
        );
        let result = project
            .read_with(cx, |project, cx| {
                project.import_remote_profile(paths, true, Arc::new(AtomicBool::new(false)), cx)
            })
            .expect("client opt-in admits native import RPC")
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("client opt-in must not confer host native import permission"),
        };
        assert_eq!(
            error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Unsupported)
        );
    } else {
        use project::profile_results_wire::PROFILE_PROTOCOL_VERSION;
        let report = coverage_store
            .update(cx, |store, cx| {
                store.interpret_artifact(
                    coverage_path.clone(),
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .expect("actual host file opt-in enables coverage before revocation");
        assert_eq!(report.files.len(), 1);
        let run_id = "host-revocation-task";
        let binding = project
            .read_with(cx, |project, cx| {
                project.prepare_remote_profile_task(run_id.into(), paths.clone(), cx)
            })
            .expect("native preparation RPC")
            .await
            .expect("actual host file opt-in enables native Task preparation");
        let mut child = smol::process::Command::new(&binding.helper_path)
            .args([
                "observe-task",
                "--socket",
                &binding.socket_path,
                "--",
                "/usr/bin/true",
            ])
            .env(
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                binding.token,
            )
            .kill_on_drop(true)
            .spawn()
            .expect("launch genuine admitted host helper");
        project
            .read_with(cx, |project, cx| {
                project.get_remote_profile_task(run_id.into(), cx)
            })
            .expect("actual Task completion poll")
            .await
            .expect("host observes genuine kernel completion");
        assert!(
            child
                .status()
                .await
                .expect("actual helper status")
                .success()
        );
        let import_request = proto::ImportProfileResults {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 910,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            profile_path: Some(paths[0].to_proto()),
            symbols_path: Some(paths[1].to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: Some(run_id.into()),
        };
        let first = transport
            .request(import_request.clone())
            .await
            .expect("genuine Task-bound multi-page import");
        assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!first.complete);
        std::fs::write(gate_directory.join("armed"), b"owned")
            .expect("arm existing host page gate");
        let pending = transport.request(proto::GetProfileResultPage {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: first.import_id,
            generation: first.generation,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        });
        let revoke_held_page = async {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !gate_directory.join("started").exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "actual host page did not enter the existing filesystem gate"
                );
                cx.background_executor
                    .timer(std::time::Duration::from_millis(20))
                    .await;
            }
            std::fs::write(&settings_path,
                r#"{"feature_flags":{"rust-coverage-experimental":"off","rust-native-profiling-experimental":"off"}}"#)
                .expect("revoke only owned host filesystem opt-ins");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                let current = transport
                    .request(proto::GetProfileTaskBinding {
                        project_id: proto::REMOTE_SERVER_PROJECT_ID,
                        protocol_version: PROFILE_PROTOCOL_VERSION,
                        task_run_id: run_id.into(),
                    })
                    .await
                    .expect("actual original Task poll after file change");
                if current.status == proto::ProfileResultStatus::Stale as i32 {
                    assert!(current.complete);
                    break;
                }
                assert_eq!(current.status, proto::ProfileResultStatus::Ready as i32);
                assert!(
                    std::time::Instant::now() < deadline,
                    "host filesystem watcher did not revoke the original Task"
                );
                cx.background_executor
                    .timer(std::time::Duration::from_millis(20))
                    .await;
            }
            let refused = transport
                .request(proto::PrepareProfileTaskBinding {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version: PROFILE_PROTOCOL_VERSION,
                    task_run_id: "host-revoked-new-task".into(),
                    profile_path: Some(paths[0].to_proto()),
                    symbols_path: Some(paths[1].to_proto()),
                    artifact_format: "samply-0.13.1".into(),
                })
                .await
                .expect("fresh native request after authoritative revocation");
            assert_eq!(
                refused.status,
                proto::ProfileResultStatus::Unsupported as i32
            );
            assert!(
                refused.helper_path.is_none()
                    && refused.socket_path.is_none()
                    && refused.token.is_none()
            );
            std::fs::write(gate_directory.join("release"), b"owned")
                .expect("release held page after actual revocation");
        };
        let (page, ()) = futures::join!(pending, revoke_held_page);
        let page = page.expect("held host page response after filesystem revocation");
        assert_eq!(page.status, proto::ProfileResultStatus::Unsupported as i32);
        assert!(page.frames.is_empty() && page.nodes.is_empty() && page.summary.is_none());
        let error = coverage_store
            .update(cx, |store, cx| {
                store.interpret_artifact(
                    coverage_path,
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .err()
            .expect("revoked host rejects coverage despite unchanged client opt-in");
        assert!(
            error
                .downcast_ref::<project::source_coverage::SourceCoverageUnsupportedHost>()
                .is_some(),
            "{error:#}"
        );
        std::fs::write(
            &settings_path,
            host_settings.expect("initial explicit host opt-in"),
        )
        .expect("restore owned host-local opt-in");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let fresh_binding = loop {
            match project
                .read_with(cx, |project, cx| {
                    project.prepare_remote_profile_task(
                        "host-restored-task".into(),
                        paths.clone(),
                        cx,
                    )
                })
                .expect("unchanged opted-in client")
                .await
            {
                Ok(binding) => break binding,
                Err(error) => assert_eq!(
                    error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                    Some(&project::profile_results_store::RemoteProfileError::Unsupported)
                ),
            }
            assert!(
                std::time::Instant::now() < deadline,
                "restored host-local opt-in never admitted a fresh Task"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        };
        let stale = transport
            .request(proto::GetProfileResultPage {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: first.import_id,
                generation: first.generation,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index: 1,
            })
            .await
            .expect("original retained page after restored host opt-in");
        assert_eq!(stale.status, proto::ProfileResultStatus::Stale as i32);
        assert!(stale.frames.is_empty() && stale.nodes.is_empty() && stale.summary.is_none());
        let mut child = smol::process::Command::new(&fresh_binding.helper_path)
            .args([
                "observe-task",
                "--socket",
                &fresh_binding.socket_path,
                "--",
                "/usr/bin/true",
            ])
            .env(
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                fresh_binding.token,
            )
            .kill_on_drop(true)
            .spawn()
            .expect("launch fresh genuine host helper");
        project
            .read_with(cx, |project, cx| {
                project.get_remote_profile_task("host-restored-task".into(), cx)
            })
            .expect("fresh Task poll")
            .await
            .expect("fresh genuine Task completion");
        assert!(
            child
                .status()
                .await
                .expect("fresh actual helper status")
                .success()
        );
        let mut fresh_import = import_request;
        fresh_import.import_id += 1;
        fresh_import.task_run_id = Some("host-restored-task".into());
        let fresh_page = transport
            .request(fresh_import)
            .await
            .expect("fresh Task-bound import after restoration");
        assert_eq!(fresh_page.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!fresh_page.complete);
        transport
            .request(proto::CancelProfileImport {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: fresh_page.import_id,
                generation: fresh_page.generation,
            })
            .await
            .expect("release fresh retained import");
        project
            .read_with(cx, |project, cx| {
                project.cancel_remote_profile_task("host-restored-task".into(), cx)
            })
            .expect("fresh Task cleanup")
            .await
            .expect("retire fresh host binding");
    }
    assert_eq!(
        project::cargo_workspace_store::local_cargo_process_calls_for_test(),
        local_cargo_calls_before,
        "ordinary Cargo metadata and action planning must run on the authoritative host"
    );
    assert_eq!(
        project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
        local_discovery_calls_before,
        "ordinary discovery must run on the authoritative host"
    );
    assert_eq!(
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
        local_coverage_interpret_calls_before,
        "host flag controls must not fall back to local coverage interpretation"
    );
    assert_eq!(
        project::profile_results::samply_profile_load_calls_for_test(),
        local_profile_loads_before,
        "host flag controls must not fall back to local profile parsing"
    );
    let shutdown = client.update(cx, |client, cx| {
        client.shutdown_processes::<proto::ShutdownRemoteServer>(
            None,
            cx.background_executor().clone(),
        )
    });
    if let Some(shutdown) = shutdown {
        shutdown.await;
    }
    connection
        .kill()
        .await
        .expect("stop owned separate enabled host");
}

#[cfg(all(feature = "rust-tools-process-test", target_os = "macos"))]
#[gpui::test]
async fn rust_tools_headless_runtime_enabled_host_local_opt_out_preserves_ordinary_actions(
    cx: &mut TestAppContext,
) {
    use futures::FutureExt as _;
    for settings in [
        None,
        Some(
            r#"{"feature_flags":{"rust-coverage-experimental":"off","rust-native-profiling-experimental":"off"}}"#,
        ),
    ] {
        rust_tools_enabled_host_local_opt_in_fixture(settings, false, cx)
            .boxed_local()
            .await;
    }
}

#[cfg(all(feature = "rust-tools-process-test", target_os = "macos"))]
#[gpui::test]
async fn rust_tools_headless_runtime_host_file_revocation_retires_held_native_page_and_recovers(
    cx: &mut TestAppContext,
) {
    use futures::FutureExt as _;
    rust_tools_enabled_host_local_opt_in_fixture(Some(r#"{"feature_flags":{"rust-coverage-experimental":"on","rust-native-profiling-experimental":"on"}}"#), true, cx).boxed_local().await;
}

#[cfg(not(feature = "rust-tools"))]
#[gpui::test]
async fn rust_tools_headless_runtime_disabled_profile_task_keeps_original_command(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    struct OriginalProfileTerminal(Arc<AtomicUsize>);

    impl workspace::TerminalProvider for OriginalProfileTerminal {
        fn spawn_prepared(
            &self,
            launch: task::SpawnInTerminal,
            _: Option<task::StructuredTaskHandle>,
            prepare: workspace::TaskLaunchPreparation,
            window: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            let terminal = Self(self.0.clone());
            window.to_async(cx).spawn(async move |cx| {
                let original = launch.clone();
                let prepared = match prepare(launch, cx).await {
                    Ok(prepared) => prepared,
                    Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                    Err(error) => return Some(Err(error)),
                };
                if let Err(error) = prepared.validate_task_identity(&original) {
                    return Some(Err(error));
                }
                assert_eq!(prepared.launch.command, original.command);
                assert_eq!(prepared.launch.args, original.args);
                assert!(
                    prepared.launch.env == original.env,
                    "Task environment changed"
                );
                assert_eq!(prepared.launch.cwd, original.cwd);
                assert_eq!(prepared.launch.command_label, original.command_label);
                let spawned =
                    match cx.update(|window, cx| terminal.spawn(prepared.launch, window, cx)) {
                        Ok(spawned) => spawned,
                        Err(error) => return Some(Err(error)),
                    };
                match prepared.completion {
                    Some(complete) => complete(spawned, cx).await,
                    None => spawned.await,
                }
            })
        }

        fn spawn(
            &self,
            launch: task::SpawnInTerminal,
            _: &mut gpui::Window,
            _: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            assert_eq!(launch.command.as_deref(), Some("fixture-profiler"));
            assert_eq!(launch.args, ["--sample"]);
            assert!(launch.command_label.contains("fixture-profiler"));
            assert!(!launch.env.contains_key("ZED_PROFILE_TASK_TOKEN"));
            self.0.fetch_add(1, Ordering::Relaxed);
            gpui::Task::ready(Some(Ok(std::process::ExitStatus::default())))
        }
    }

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/fixture",
        json!({"profile.json": "{}", "profile.syms.json": "{}"}),
    )
    .await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        );
    });
    cx.update(|cx| {
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
    });
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/fixture", true, cx)
        })
        .await
        .expect("remote fixture worktree");
    cx.run_until_parked();
    let (workspace, cx) = cx
        .add_window_view(|window, cx| workspace::Workspace::test_new(project.clone(), window, cx));
    let offers = Rc::new(std::cell::Cell::new(0));
    let _subscription = cx.update(|_, cx| {
        cx.subscribe(&workspace, {
            let offers = offers.clone();
            move |_, event, _| {
                if matches!(event, workspace::Event::TaskArtifactGroupReady(_)) {
                    offers.set(offers.get() + 1);
                }
            }
        })
    });
    let artifact_group = task::TaskArtifactGroup {
        format: "samply-0.13.1".into(),
        files: std::collections::BTreeMap::from_iter([
            (
                "profile".into(),
                task::TaskArtifact {
                    path: "profile.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 32 * 1024 * 1024,
                },
            ),
            (
                "symbols".into(),
                task::TaskArtifact {
                    path: "profile.syms.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 16 * 1024 * 1024,
                },
            ),
        ]),
    };
    let task = task::TaskTemplate {
        label: "Profile on older host".into(),
        command: "fixture-profiler".into(),
        args: vec!["--sample".into()],
        cwd: Some("/fixture".into()),
        save: task::SaveStrategy::None,
        artifact_group: Some(artifact_group),
        ..Default::default()
    }
    .resolve_task("older-host-profile", &task::TaskContext::default())
    .expect("explicit profile Task");
    let spawned = Arc::new(AtomicUsize::new(0));
    let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(OriginalProfileTerminal(spawned.clone()));
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            task,
            true,
            move |result, _| {
                completed_sender
                    .send(result)
                    .expect("Task completion receiver")
            },
            window,
            cx,
        );
    });
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(30));
    let approved_completion = async {
        while !cx.has_pending_prompt() {
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        let (message, detail) = cx.pending_prompt().expect("actual capture privacy review");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert_eq!(
            detail.lines().take(2).collect::<Vec<_>>(),
            [
                "Profile: /fixture/profile.json",
                "Symbols: /fixture/profile.syms.json"
            ]
        );
        for requirement in [
            "function names, source and library paths, and workload details",
            "Processing stays local to the authoritative host",
            "not uploaded or sent as telemetry",
            "explicitly consented remote import",
            "raw files remain on that host",
            "Delete them through the project file controls",
            "session-only result",
            "does not delete these files",
            "only to this capture",
        ] {
            assert!(detail.contains(requirement), "{requirement}");
        }
        assert_eq!(
            spawned.load(Ordering::Relaxed),
            0,
            "no Task before approval"
        );
        assert_eq!(offers.get(), 0, "no native offer before approval");
        cx.simulate_prompt_answer("Capture");
        completed_receiver.await
    };
    let completion = match futures::future::select(Box::pin(approved_completion), deadline).await {
        futures::future::Either::Left((result, _)) => result.expect("Task completion"),
        futures::future::Either::Right(_) => panic!("Task completion timed out"),
    };
    assert_eq!(completion, workspace::tasks::ScheduledTaskResult::Success);
    cx.run_until_parked();
    assert_eq!(spawned.load(Ordering::Relaxed), 1);
    assert_eq!(offers.get(), 0, "older host cannot verify a native offer");
    assert!(workspace.read_with(cx, |workspace, _| {
        workspace
            .notification_ids()
            .contains(&workspace::notifications::NotificationId::unique::<
                workspace::tasks::CompletedTaskArtifactGroup,
            >())
    }));
    drop(_subscription);
    cx.deactivate_window();
    workspace.update_in(cx, |_, window, _| window.remove_window());
    let weak_workspace = workspace.downgrade();
    drop(workspace);
    let weak_project = project.downgrade();
    drop(project);
    cx.cx.update(|_| {});
    cx.run_until_parked();
    cx.cx.update(|_| {});
    assert!(weak_workspace.upgrade().is_none(), "workspace released");
    assert!(weak_project.upgrade().is_none(), "client project released");
    drop(headless);
    server_cx.run_until_parked();
}

#[cfg(not(feature = "rust-tools"))]
#[gpui::test]
async fn rust_tools_headless_runtime_disabled_profile_task_rejects_other_host_errors(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    for (status, mismatched_version) in [
        (Some(proto::ProfileResultStatus::Restricted), false),
        (Some(proto::ProfileResultStatus::Cancelled), false),
        (Some(proto::ProfileResultStatus::Stale), false),
        (Some(proto::ProfileResultStatus::Mismatch), false),
        (Some(proto::ProfileResultStatus::Error), false),
        (Some(proto::ProfileResultStatus::Unsupported), true),
        (None, false),
    ] {
        struct OriginalProfileTerminal(Arc<AtomicUsize>);

        impl workspace::TerminalProvider for OriginalProfileTerminal {
            fn spawn_prepared(
                &self,
                launch: task::SpawnInTerminal,
                _: Option<task::StructuredTaskHandle>,
                prepare: workspace::TaskLaunchPreparation,
                window: &mut gpui::Window,
                cx: &mut gpui::App,
            ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
                let terminal = Self(self.0.clone());
                window.to_async(cx).spawn(async move |cx| {
                    let original = launch.clone();
                    let prepared = match prepare(launch, cx).await {
                        Ok(prepared) => prepared,
                        Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    if let Err(error) = prepared.validate_task_identity(&original) {
                        return Some(Err(error));
                    }
                    assert_eq!(prepared.launch.command, original.command);
                    assert_eq!(prepared.launch.args, original.args);
                    assert!(
                        prepared.launch.env == original.env,
                        "Task environment changed"
                    );
                    assert_eq!(prepared.launch.cwd, original.cwd);
                    assert_eq!(prepared.launch.command_label, original.command_label);
                    let spawned =
                        match cx.update(|window, cx| terminal.spawn(prepared.launch, window, cx)) {
                            Ok(spawned) => spawned,
                            Err(error) => return Some(Err(error)),
                        };
                    match prepared.completion {
                        Some(complete) => complete(spawned, cx).await,
                        None => spawned.await,
                    }
                })
            }

            fn spawn(
                &self,
                launch: task::SpawnInTerminal,
                _: &mut gpui::Window,
                _: &mut gpui::App,
            ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
                assert_eq!(launch.command.as_deref(), Some("fixture-profiler"));
                assert_eq!(launch.args, ["--sample"]);
                assert!(launch.command_label.contains("fixture-profiler"));
                assert!(!launch.env.contains_key("ZED_PROFILE_TASK_TOKEN"));
                self.0.fetch_add(1, Ordering::Relaxed);
                gpui::Task::ready(Some(Ok(std::process::ExitStatus::default())))
            }
        }

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree(
            "/fixture",
            json!({"profile.json": "{}", "profile.syms.json": "{}"}),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            );
        });
        let preparation_requests = Arc::new(AtomicUsize::new(0));
        server_cx.update(|cx| {
            let preparation_requests = preparation_requests.clone();
            headless.read(cx).session.add_request_handler(
                headless.downgrade(),
                move |_, request: rpc::TypedEnvelope<proto::PrepareProfileTaskBinding>, _| {
                    let preparation_requests = preparation_requests.clone();
                    async move {
                        preparation_requests.fetch_add(1, Ordering::Relaxed);
                        assert_eq!(
                            request.payload.protocol_version,
                            project::profile_results_wire::PROFILE_PROTOCOL_VERSION
                        );
                        assert!(project::profile_results_wire::valid_profile_task_run_id(
                            &request.payload.task_run_id
                        ));
                        let Some(status) = status else {
                            anyhow::bail!("Controlled host preparation failure");
                        };
                        Ok(proto::ProfileTaskBindingResponse {
                            protocol_version:
                                project::profile_results_wire::PROFILE_PROTOCOL_VERSION
                                    + u32::from(mismatched_version),
                            task_run_id: request.payload.task_run_id,
                            status: status as i32,
                            complete: false,
                            helper_path: None,
                            socket_path: None,
                            token: None,
                        })
                    }
                },
            );
        });
        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });
        project
            .update(cx, |project, cx| {
                project.find_or_create_worktree("/fixture", true, cx)
            })
            .await
            .expect("remote fixture worktree");
        cx.run_until_parked();
        let (workspace, cx) = cx.add_window_view(|window, cx| {
            workspace::Workspace::test_new(project.clone(), window, cx)
        });
        let offers = Rc::new(std::cell::Cell::new(0));
        let _subscription = cx.update(|_, cx| {
            cx.subscribe(&workspace, {
                let offers = offers.clone();
                move |_, event, _| {
                    if matches!(event, workspace::Event::TaskArtifactGroupReady(_)) {
                        offers.set(offers.get() + 1);
                    }
                }
            })
        });
        let artifact_group = task::TaskArtifactGroup {
            format: "samply-0.13.1".into(),
            files: std::collections::BTreeMap::from_iter([
                (
                    "profile".into(),
                    task::TaskArtifact {
                        path: "profile.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 32 * 1024 * 1024,
                    },
                ),
                (
                    "symbols".into(),
                    task::TaskArtifact {
                        path: "profile.syms.json".into(),
                        kind: task::TaskArtifactKind::Data,
                        max_bytes: 16 * 1024 * 1024,
                    },
                ),
            ]),
        };
        let task = task::TaskTemplate {
            label: "Profile on older host".into(),
            command: "fixture-profiler".into(),
            args: vec!["--sample".into()],
            cwd: Some("/fixture".into()),
            save: task::SaveStrategy::None,
            artifact_group: Some(artifact_group),
            ..Default::default()
        }
        .resolve_task("older-host-profile", &task::TaskContext::default())
        .expect("explicit profile Task");
        let spawned = Arc::new(AtomicUsize::new(0));
        let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
        workspace.update_in(cx, |workspace, window, cx| {
            workspace.set_terminal_provider(OriginalProfileTerminal(spawned.clone()));
            workspace.schedule_resolved_task_with_completion(
                project::TaskSourceKind::UserInput,
                task,
                true,
                move |result, _| {
                    completed_sender
                        .send(result)
                        .expect("Task completion receiver")
                },
                window,
                cx,
            );
        });
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(30));
        let approved_completion = async {
            while !cx.has_pending_prompt() {
                cx.background_executor
                    .timer(std::time::Duration::from_millis(10))
                    .await;
            }
            let (message, detail) = cx.pending_prompt().expect("actual capture privacy review");
            assert_eq!(message, "Capture a native profile on the project host?");
            assert_eq!(
                detail.lines().take(2).collect::<Vec<_>>(),
                [
                    "Profile: /fixture/profile.json",
                    "Symbols: /fixture/profile.syms.json"
                ]
            );
            for requirement in [
                "function names, source and library paths, and workload details",
                "Processing stays local to the authoritative host",
                "not uploaded or sent as telemetry",
                "explicitly consented remote import",
                "raw files remain on that host",
                "Delete them through the project file controls",
                "session-only result",
                "does not delete these files",
                "only to this capture",
            ] {
                assert!(detail.contains(requirement), "{requirement}");
            }
            assert_eq!(
                spawned.load(Ordering::Relaxed),
                0,
                "no Task before approval"
            );
            assert_eq!(offers.get(), 0, "no native offer before approval");
            cx.simulate_prompt_answer("Capture");
            completed_receiver.await
        };
        let completion =
            match futures::future::select(Box::pin(approved_completion), deadline).await {
                futures::future::Either::Left((result, _)) => result.expect("Task completion"),
                futures::future::Either::Right(_) => panic!("Task completion timed out"),
            };
        assert_eq!(
            completion,
            workspace::tasks::ScheduledTaskResult::SpawnFailed
        );
        assert_eq!(
            preparation_requests.load(Ordering::Relaxed),
            1,
            "each controlled host response must reach the actual preparation RPC"
        );
        cx.run_until_parked();
        assert_eq!(
            spawned.load(Ordering::Relaxed),
            0,
            "only typed Unsupported may preserve an authored launch"
        );
        assert_eq!(offers.get(), 0, "older host cannot verify a native offer");
        assert!(!workspace.read_with(cx, |workspace, _| {
            workspace.notification_ids().contains(
                &workspace::notifications::NotificationId::unique::<
                    workspace::tasks::CompletedTaskArtifactGroup,
                >(),
            )
        }));
        drop(_subscription);
        cx.deactivate_window();
        workspace.update_in(cx, |_, window, _| window.remove_window());
        let weak_workspace = workspace.downgrade();
        drop(workspace);
        let weak_project = project.downgrade();
        drop(project);
        cx.cx.update(|_| {});
        cx.run_until_parked();
        cx.cx.update(|_| {});
        assert!(weak_workspace.upgrade().is_none(), "workspace released");
        assert!(weak_project.upgrade().is_none(), "client project released");
        drop(headless);
        server_cx.run_until_parked();
    }
}

#[gpui::test]
async fn profile_artifact_lifecycle_remote_fresh_ignored_output(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    struct HostProducer(Arc<FakeFs>);
    impl workspace::TerminalProvider for HostProducer {
        fn spawn(
            &self,
            _: task::SpawnInTerminal,
            _: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            let fs = self.0.clone();
            cx.background_spawn(async move {
                fs.insert_tree(
                    path!("/profile-fixture/target"),
                    json!({"profile.svg": "<svg>host output</svg>"}),
                )
                .await;
                Some(Ok(std::process::ExitStatus::default()))
            })
        }
    }

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/profile-fixture"),
        json!({".git": {}, ".gitignore": "target\n", "target": {}, "source.rs": "fn main() {}"}),
    )
    .await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/profile-fixture"), true, cx)
        })
        .await
        .expect("remote fixture worktree");
    cx.run_until_parked();
    let artifact_path = ProjectPath {
        worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
        path: rel_path("target/profile.svg").into(),
    };
    assert!(project.read_with(cx, |project, cx| {
        project.entry_for_path(&artifact_path, cx).is_none()
    }));
    let client_fs = project.read_with(cx, |project, _| project.fs().clone());
    let (workspace, cx) = cx
        .add_window_view(|window, cx| workspace::Workspace::test_new(project.clone(), window, cx));
    let task = task::TaskTemplate {
        label: "Host profiling output".into(),
        command: "explicit-host-profiler".into(),
        cwd: Some(path!("/profile-fixture").into()),
        save: task::SaveStrategy::None,
        artifact: Some(task::TaskArtifact {
            path: "target/profile.svg".into(),
            kind: task::TaskArtifactKind::Svg,
            max_bytes: 1024,
        }),
        ..Default::default()
    }
    .resolve_task("host-profile-fixture", &task::TaskContext::default())
    .expect("explicit artifact declaration");
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(HostProducer(fs.clone()));
        workspace.schedule_resolved_task(
            project::TaskSourceKind::UserInput,
            task,
            true,
            window,
            cx,
        );
    });
    cx.condition(&workspace, |workspace, cx| {
        workspace.item_of_type::<Editor>(cx).is_some()
    })
    .await;
    assert!(
        client_fs
            .metadata(Path::new(path!("/profile-fixture/target/profile.svg")))
            .await
            .expect("client filesystem check")
            .is_none()
    );
    let weak_editor = workspace.read_with(cx, |workspace, cx| {
        let editor = workspace
            .item_of_type::<Editor>(cx)
            .expect("fresh host artifact opened");
        assert_eq!(
            editor.read(cx).buffer().read(cx).snapshot(cx).text(),
            "<svg>host output</svg>"
        );
        editor.downgrade()
    });
    cx.deactivate_window();
    workspace.update_in(cx, |_, window, _| window.remove_window());
    let weak_workspace = workspace.downgrade();
    drop(workspace);
    let weak_project = project.downgrade();
    drop(project);
    // Entity destruction cancels editor tasks; drain their releases before
    // tearing down the server context retained by the client project.
    cx.cx.update(|_| {});
    cx.run_until_parked();
    cx.cx.update(|_| {});
    assert!(weak_workspace.upgrade().is_none(), "workspace released");
    assert!(weak_editor.upgrade().is_none(), "editor released");
    assert!(weak_project.upgrade().is_none(), "client project released");
    drop(headless);
    server_cx.run_until_parked();
}

#[gpui::test]
async fn profile_artifact_lifecycle_remote_late_success_after_disconnect(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    profile_artifact_late_completion_after_interruption(
        ProfileTaskInterruption::Disconnect,
        cx,
        server_cx,
    )
    .await;
}

#[gpui::test]
async fn profile_artifact_lifecycle_remote_late_success_after_reconnect(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    profile_artifact_late_completion_after_interruption(
        ProfileTaskInterruption::Reconnect,
        cx,
        server_cx,
    )
    .await;
}

#[gpui::test]
async fn profile_artifact_lifecycle_remote_late_success_after_cancellation(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    profile_artifact_late_completion_after_interruption(
        ProfileTaskInterruption::Cancel,
        cx,
        server_cx,
    )
    .await;
}

#[derive(Clone, Copy)]
enum ProfileTaskInterruption {
    Disconnect,
    Reconnect,
    Cancel,
    ReplaceWorktree,
}

#[gpui::test]
async fn profile_artifact_lifecycle_remote_late_success_after_worktree_replacement(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    profile_artifact_late_completion_after_interruption(
        ProfileTaskInterruption::ReplaceWorktree,
        cx,
        server_cx,
    )
    .await;
}

async fn profile_artifact_late_completion_after_interruption(
    interruption: ProfileTaskInterruption,
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    struct DelayedProducer {
        fs: Arc<FakeFs>,
        started: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        completion: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
        produced: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
    }
    impl workspace::TerminalProvider for DelayedProducer {
        fn spawn(
            &self,
            _: task::SpawnInTerminal,
            _: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            self.started
                .lock()
                .expect("Task start lock")
                .take()
                .expect("one scheduled Task")
                .send(())
                .expect("Task start receiver");
            let completion = self
                .completion
                .lock()
                .expect("completion gate lock")
                .take()
                .expect("completion gate");
            let produced = self
                .produced
                .lock()
                .expect("produced lock")
                .take()
                .expect("one output signal");
            let fs = self.fs.clone();
            cx.background_spawn(async move {
                completion.await.expect("release late Task completion");
                fs.insert_tree(
                    path!("/late-profile/target"),
                    json!({"late.svg": "<svg>late host output</svg>"}),
                )
                .await;
                produced.send(()).expect("output receiver");
                Some(Ok(std::process::ExitStatus::default()))
            })
        }
    }
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/late-profile"),
        json!({".gitignore": "target\n", "target": {}}),
    )
    .await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
    });
    let (original_worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/late-profile"), true, cx)
        })
        .await
        .expect("remote worktree");
    let remote = project.read_with(cx, |project, _| {
        project.remote_client().expect("remote client")
    });
    let (workspace, cx) = cx
        .add_window_view(|window, cx| workspace::Workspace::test_new(project.clone(), window, cx));
    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, completion) = futures::channel::oneshot::channel();
    let (finished, result) = futures::channel::oneshot::channel();
    let (produced, output) = futures::channel::oneshot::channel();
    let task = task::TaskTemplate {
        label: "Delayed host profile".into(),
        command: "explicit-host-profiler".into(),
        cwd: Some(path!("/late-profile").into()),
        save: task::SaveStrategy::None,
        artifact: Some(task::TaskArtifact {
            path: "target/late.svg".into(),
            kind: task::TaskArtifactKind::Svg,
            max_bytes: 1024,
        }),
        ..Default::default()
    }
    .resolve_task("late-host-profile", &task::TaskContext::default())
    .expect("declared Task artifact");
    let handle = workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(DelayedProducer {
            fs: fs.clone(),
            started: std::sync::Mutex::new(Some(started)),
            completion: std::sync::Mutex::new(Some(completion)),
            produced: std::sync::Mutex::new(Some(produced)),
        });
        if matches!(interruption, ProfileTaskInterruption::Cancel) {
            Some(workspace.schedule_resolved_task_with_structured_handle(
                project::TaskSourceKind::UserInput,
                task,
                true,
                window,
                cx,
            ))
        } else {
            workspace.schedule_resolved_task_with_completion(
                project::TaskSourceKind::UserInput,
                task,
                true,
                move |result, _| finished.send(result).expect("Task result receiver"),
                window,
                cx,
            );
            None
        }
    });
    started_receiver
        .await
        .expect("Task reached the completion gate");
    if let Some(handle) = &handle {
        cx.update(|_, cx| assert!(handle.cancel(cx), "running Task accepts cancellation"));
    } else if matches!(interruption, ProfileTaskInterruption::ReplaceWorktree) {
        let original_id = original_worktree.read_with(cx, |worktree, _| worktree.id());
        project.update(cx, |project, cx| project.remove_worktree(original_id, cx));
        cx.run_until_parked();
        server_cx.run_until_parked();
        let (replacement, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree(path!("/late-profile"), true, cx)
            })
            .await
            .expect("same-path replacement worktree");
        assert_ne!(
            replacement.read_with(cx, |worktree, _| worktree.id()),
            original_id
        );
    } else if matches!(interruption, ProfileTaskInterruption::Reconnect) {
        remote
            .update(cx, |remote, cx| remote.simulate_disconnect(cx))
            .await;
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5));
        let recovery = async {
            while remote.read_with(cx, |remote, _| remote.connection_state())
                != remote::ConnectionState::Connected
            {
                cx.background_executor
                    .timer(std::time::Duration::from_millis(10))
                    .await;
            }
        };
        futures::pin_mut!(recovery);
        assert!(
            matches!(
                futures::future::select(recovery, deadline).await,
                futures::future::Either::Left(_)
            ),
            "mock connection must recover before late completion"
        );
    } else {
        remote.update(cx, |remote, cx| remote.force_server_not_running(cx));
        cx.run_until_parked();
        assert!(project.read_with(cx, |project, cx| project.is_disconnected(cx)));
    }
    release.send(()).expect("release the successful host Task");
    output.await.expect("late host artifact produced");
    if let Some(handle) = &handle {
        assert!(matches!(
            handle.state(),
            task::StructuredTaskState::Cancelled { .. }
        ));
    } else {
        assert_eq!(
            result.await.expect("late Task completion callback"),
            workspace::tasks::ScheduledTaskResult::Success
        );
    }
    cx.run_until_parked();
    assert_eq!(
        fs.load(Path::new(path!("/late-profile/target/late.svg")))
            .await
            .expect("host produced the artifact"),
        "<svg>late host output</svg>"
    );
    assert!(workspace.read_with(cx, |workspace, cx| {
        workspace.item_of_type::<Editor>(cx).is_none()
    }));
    cx.deactivate_window();
    workspace.update_in(cx, |_, window, _| window.remove_window());
    drop(workspace);
    drop(project);
    drop(remote);
    cx.cx.update(|_| {});
    cx.run_until_parked();
    cx.cx.update(|_| {});
    drop(headless);
    server_cx.run_until_parked();
}

#[gpui::test]
async fn profile_artifact_lifecycle_remote_rejects_private_oversized_and_history(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/bounded-remote"),
        json!({
            "profile.svg": "ok", "allowed.svg": "ok", ".env": "private", "large.svg": "oversized"
        }),
    )
    .await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/bounded-remote"), true, cx)
        })
        .await
        .expect("remote bounded worktree");
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
    for (path, limit) in [(".env", 1024), ("large.svg", 2), ("missing.svg", 1024)] {
        let path = ProjectPath {
            worktree_id,
            path: rel_path(path).into(),
        };
        assert!(
            project
                .update(cx, |project, cx| {
                    project.open_buffer_with_limit(path.clone(), limit, cx)
                })
                .await
                .is_err()
        );
        assert!(project.read_with(cx, |project, cx| {
            project.buffer_store().read(cx).get_by_path(&path).is_none()
        }));
    }
    let allowed = project
        .update(cx, |project, cx| {
            project.open_buffer_with_limit(
                ProjectPath {
                    worktree_id,
                    path: rel_path("allowed.svg").into(),
                },
                2,
                cx,
            )
        })
        .await
        .expect("bounded host file");
    assert_eq!(allowed.read_with(cx, |buffer, _| buffer.text()), "ok");
    let path = ProjectPath {
        worktree_id,
        path: rel_path("profile.svg").into(),
    };
    let store = headless.read_with(server_cx, |headless, _| headless.buffer_store.clone());
    let buffer = store
        .update(server_cx, |store, cx| store.open_buffer(path.clone(), cx))
        .await
        .expect("host buffer");
    buffer.update(server_cx, |buffer, cx| {
        buffer.edit([(0..0, "historical content")], None, cx);
        buffer.edit([(0.."historical content".len(), "")], None, cx);
    });
    store
        .update(server_cx, |store, cx| store.save_buffer(buffer.clone(), cx))
        .await
        .expect("save small current content");
    assert_eq!(buffer.read_with(server_cx, |buffer, _| buffer.text()), "ok");
    let error = project
        .update(cx, |project, cx| {
            project.open_buffer_with_limit(path.clone(), 2, cx)
        })
        .await
        .expect_err("retained history exceeds artifact bound");
    assert!(format!("{error:#}").contains("history"), "{error:#}");
    assert!(project.read_with(cx, |project, cx| {
        project.buffer_store().read(cx).get_by_path(&path).is_none()
    }));
}

#[gpui::test(iterations = 20)]
async fn profile_artifact_lifecycle_remote_cancel_and_retry(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/cancelled-profile"),
        json!({"profile.svg": "<svg>bounded retry</svg>"}),
    )
    .await;
    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/cancelled-profile"), true, cx)
        })
        .await
        .expect("remote cancellation worktree");
    let path = ProjectPath {
        worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
        path: rel_path("profile.svg").into(),
    };
    let opening = project.update(cx, |project, cx| {
        project.open_buffer_with_limit(path.clone(), 1024, cx)
    });
    cx.executor().simulate_random_delay().await;
    drop(opening);
    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer_with_limit(path.clone(), 1024, cx)
        })
        .await
        .expect("retry after cancelled bounded open");
    assert_eq!(
        buffer.read_with(cx, |buffer, _| buffer.text()),
        "<svg>bounded retry</svg>"
    );
    drop(buffer);
    cx.update(|_| {});
    cx.run_until_parked();
    let reopened = project
        .update(cx, |project, cx| {
            project.open_buffer_with_limit(path, 1024, cx)
        })
        .await
        .expect("reopen after releasing the admitted buffer");
    assert_eq!(
        reopened.read_with(cx, |buffer, _| buffer.text()),
        "<svg>bounded retry</svg>"
    );
}

#[gpui::test(iterations = 20)]
async fn profile_artifact_lifecycle_remote_reconnect_bounded_open(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        path!("/reconnected-profile"),
        json!({"profile.svg": "<svg>reconnected host</svg>"}),
    )
    .await;
    let (project, _headless) = init_test(&fs, cx, server_cx).await;
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/reconnected-profile"), true, cx)
        })
        .await
        .expect("remote reconnect worktree");
    let path = ProjectPath {
        worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
        path: rel_path("profile.svg").into(),
    };
    let remote = project.read_with(cx, |project, _| {
        project.remote_client().expect("remote client")
    });
    remote
        .update(cx, |remote, cx| remote.simulate_disconnect(cx))
        .await;
    let buffer = project
        .update(cx, |project, cx| {
            project.open_buffer_with_limit(path.clone(), 1024, cx)
        })
        .await
        .expect("bounded opening resumes after mock channel reconnect");
    assert_eq!(
        buffer.read_with(cx, |buffer, _| buffer.text()),
        "<svg>reconnected host</svg>"
    );
    assert!(
        project
            .update(cx, |project, cx| {
                project.open_buffer_with_limit(path, 2, cx)
            })
            .await
            .is_err(),
        "reconnection must preserve the bound even for an already open buffer"
    );
}

pub async fn init_test<F: Fs + 'static>(
    server_fs: &Arc<F>,
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) -> (Entity<Project>, Entity<HeadlessProject>) {
    let (project, headless, _) = init_test_with_restart_activity(server_fs, cx, server_cx).await;
    (project, headless)
}

async fn init_test_with_restart_activity<F: Fs + 'static>(
    server_fs: &Arc<F>,
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) -> (
    Entity<Project>,
    Entity<HeadlessProject>,
    remote::RemoteRequestActivity,
) {
    let server_fs = server_fs.clone();
    cx.update(|cx| {
        release_channel::init(semver::Version::new(0, 0, 0), cx);
    });
    server_cx.update(|cx| {
        release_channel::init(semver::Version::new(0, 0, 0), cx);
    });
    init_logger();

    let (opts, ssh_server_client, _, activity) =
        RemoteClient::fake_server_with_activity(cx, server_cx);
    let http_client = Arc::new(BlockedHttpClient);
    let node_runtime = NodeRuntime::unavailable();
    let languages = Arc::new(LanguageRegistry::new(cx.executor()));
    let proxy = Arc::new(ExtensionHostProxy::new());
    server_cx.update(HeadlessProject::init);
    let headless = server_cx.new(|cx| {
        HeadlessProject::new(
            crate::HeadlessAppState {
                session: ssh_server_client,
                fs: server_fs.clone(),
                http_client,
                node_runtime,
                languages,
                extension_host_proxy: proxy,
                startup_time: std::time::Instant::now(),
            },
            false,
            cx,
        )
    });

    let ssh = RemoteClient::connect_mock(opts, cx).await;
    let project = build_project(ssh, cx);
    project
        .update(cx, {
            let headless = headless.clone();
            |_, cx| cx.on_release(|_, _| drop(headless))
        })
        .detach();
    (project, headless, activity)
}

fn init_logger() {
    zlog::init_test();
}

fn build_project(ssh: Entity<RemoteClient>, cx: &mut TestAppContext) -> Entity<Project> {
    build_project_with_worktree_trust(ssh, false, cx)
}

fn build_project_with_worktree_trust(
    ssh: Entity<RemoteClient>,
    init_worktree_trust: bool,
    cx: &mut TestAppContext,
) -> Entity<Project> {
    cx.update(|cx| {
        if !cx.has_global::<SettingsStore>() {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        }
    });

    let client = cx.update(|cx| {
        Client::new(
            Arc::new(FakeSystemClock::new()),
            FakeHttpClient::with_404_response(),
            cx,
        )
    });

    let node = NodeRuntime::unavailable();
    let user_store = cx.new(|cx| UserStore::new(client.clone(), cx));
    let languages = Arc::new(LanguageRegistry::test(cx.executor()));
    let fs = FakeFs::new(cx.executor());

    cx.update(|cx| {
        Project::init(&client, cx);
    });

    cx.update(|cx| {
        Project::remote(
            ssh,
            client,
            node,
            user_store,
            languages,
            fs,
            init_worktree_trust,
            cx,
        )
    })
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
#[gpui::test]
async fn profile_task_binding_publishes_host_verified_workspace_artifacts(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    struct ObservedProfileTerminal {
        fail: bool,
        expected_command: &'static str,
        expected_arguments: &'static [&'static str],
    }

    impl workspace::TerminalProvider for ObservedProfileTerminal {
        fn spawn_prepared(
            &self,
            launch: task::SpawnInTerminal,
            _: Option<task::StructuredTaskHandle>,
            prepare: workspace::TaskLaunchPreparation,
            window: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            let terminal = Self {
                fail: self.fail,
                expected_command: self.expected_command,
                expected_arguments: self.expected_arguments,
            };
            window.to_async(cx).spawn(async move |cx| {
                let original = launch.clone();
                let prepared = match prepare(launch, cx).await {
                    Ok(prepared) => prepared,
                    Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                    Err(error) => return Some(Err(error)),
                };
                if let Err(error) = prepared.validate_task_identity(&original) {
                    return Some(Err(error));
                }
                assert!(
                    prepared.native_terminal.is_none(),
                    "authored remote Task must use the original observed host binding"
                );
                let spawned =
                    match cx.update(|window, cx| terminal.spawn(prepared.launch, window, cx)) {
                        Ok(spawned) => spawned,
                        Err(error) => return Some(Err(error)),
                    };
                match prepared.completion {
                    Some(complete) => complete(spawned, cx).await,
                    None => spawned.await,
                }
            })
        }

        fn spawn(
            &self,
            launch: task::SpawnInTerminal,
            _: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            use util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV;

            let executable = std::env::current_exe().expect("test host executable");
            assert_eq!(launch.command.as_deref(), executable.to_str());
            assert_eq!(
                launch.args.first().map(String::as_str),
                Some("observe-task")
            );
            assert_eq!(launch.args.get(1).map(String::as_str), Some("--socket"));
            assert_eq!(launch.args.get(3).map(String::as_str), Some("--"));
            assert_eq!(
                launch.args.get(4).map(String::as_str),
                Some(self.expected_command)
            );
            assert_eq!(
                launch.args.get(5..).unwrap_or_default(),
                self.expected_arguments
            );
            assert!(launch.preserve_command_label);
            assert!(launch.command_label.contains("fixture-profiler"));
            let socket = launch.args.get(2).cloned().expect("host observer socket");
            let token = launch
                .env
                .get(TASK_COMPLETION_TOKEN_ENV)
                .cloned()
                .expect("host observer token");
            assert!(!launch.command_label.contains(&token));
            let mut child = smol::process::Command::new(executable);
            child
                .args([
                    "--exact",
                    "tests::native_task_completion_helper",
                    "--nocapture",
                ])
                .env("ZED_TEST_OBSERVER_SOCKET", socket)
                .env(TASK_COMPLETION_TOKEN_ENV, token)
                .kill_on_drop(true);
            if self.fail {
                child.env("ZED_TEST_OBSERVER_FAILURE", "1");
            }
            cx.background_spawn(
                async move { Some(child.status().await.map_err(anyhow::Error::from)) },
            )
        }
    }

    struct DelayedObservedProfileTerminal {
        observed: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
        release: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
    }

    impl workspace::TerminalProvider for DelayedObservedProfileTerminal {
        fn spawn_prepared(
            &self,
            launch: task::SpawnInTerminal,
            _: Option<task::StructuredTaskHandle>,
            prepare: workspace::TaskLaunchPreparation,
            window: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            let terminal = Self {
                observed: std::sync::Mutex::new(
                    self.observed.lock().expect("observer signal lock").take(),
                ),
                release: std::sync::Mutex::new(
                    self.release.lock().expect("observer release lock").take(),
                ),
            };
            window.to_async(cx).spawn(async move |cx| {
                let original = launch.clone();
                let prepared = match prepare(launch, cx).await {
                    Ok(prepared) => prepared,
                    Err(error) if error.is::<workspace::TaskLaunchCancelled>() => return None,
                    Err(error) => return Some(Err(error)),
                };
                if let Err(error) = prepared.validate_task_identity(&original) {
                    return Some(Err(error));
                }
                assert!(
                    prepared.native_terminal.is_none(),
                    "authored remote Task must use the original observed host binding"
                );
                let spawned =
                    match cx.update(|window, cx| terminal.spawn(prepared.launch, window, cx)) {
                        Ok(spawned) => spawned,
                        Err(error) => return Some(Err(error)),
                    };
                match prepared.completion {
                    Some(complete) => complete(spawned, cx).await,
                    None => spawned.await,
                }
            })
        }

        fn spawn(
            &self,
            launch: task::SpawnInTerminal,
            _: &mut gpui::Window,
            cx: &mut gpui::App,
        ) -> gpui::Task<Option<anyhow::Result<std::process::ExitStatus>>> {
            use util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV;

            let executable = std::env::current_exe().expect("test host executable");
            assert_eq!(launch.command.as_deref(), executable.to_str());
            assert_eq!(
                launch.args.first().map(String::as_str),
                Some("observe-task")
            );
            assert_eq!(
                launch.args.get(4).map(String::as_str),
                Some("fixture-profiler")
            );
            let socket = launch.args.get(2).cloned().expect("host observer socket");
            let token = launch
                .env
                .get(TASK_COMPLETION_TOKEN_ENV)
                .cloned()
                .expect("host observer token");
            let observed = self
                .observed
                .lock()
                .expect("observer signal lock")
                .take()
                .expect("one observer signal");
            let release = self
                .release
                .lock()
                .expect("observer release lock")
                .take()
                .expect("one observer release");
            let mut child = smol::process::Command::new(executable);
            child
                .args([
                    "--exact",
                    "tests::native_task_completion_helper",
                    "--nocapture",
                ])
                .env("ZED_TEST_OBSERVER_SOCKET", socket)
                .env(TASK_COMPLETION_TOKEN_ENV, token)
                .kill_on_drop(true);
            cx.background_spawn(async move {
                let status = child.status().await.map_err(anyhow::Error::from);
                observed.send(()).expect("observer completion receiver");
                release.await.expect("release observed Task completion");
                Some(status)
            })
        }
    }

    cx.executor().allow_parking();
    server_cx.executor().allow_parking();
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree("/fixture", json!({"workload.rs": "fn main() {}"}))
        .await;
    fs.insert_file(
        "/fixture/profile.json",
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        )
        .to_vec(),
    )
    .await;
    fs.insert_file(
        "/fixture/profile.syms.json",
        include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json").to_vec(),
    ).await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    server_cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });

    cx.update(|cx| {
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
        language_tools::init(cx);
    });
    project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/fixture", true, cx)
        })
        .await
        .expect("remote profile worktree");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let (workspace, cx) = cx
        .add_window_view(|window, cx| workspace::Workspace::test_new(project.clone(), window, cx));
    let ready = Rc::new(std::cell::RefCell::new(Vec::new()));
    let _subscription = cx.update(|_, cx| {
        cx.subscribe(&workspace, {
            let ready = ready.clone();
            move |_, event, _| {
                if let workspace::Event::TaskArtifactGroupReady(group) = event {
                    ready.borrow_mut().push(group.clone());
                }
            }
        })
    });
    let group = task::TaskArtifactGroup {
        format: "samply-0.13.1".into(),
        files: std::collections::BTreeMap::from_iter([
            (
                "profile".into(),
                task::TaskArtifact {
                    path: "profile.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 32 * 1024 * 1024,
                },
            ),
            (
                "symbols".into(),
                task::TaskArtifact {
                    path: "profile.syms.json".into(),
                    kind: task::TaskArtifactKind::Data,
                    max_bytes: 16 * 1024 * 1024,
                },
            ),
        ]),
    };
    let task = task::TaskTemplate {
        label: "Observed native profile".into(),
        command: "fixture-profiler".into(),
        args: vec!["--sample".into(), "10".into()],
        cwd: Some("/fixture".into()),
        save: task::SaveStrategy::None,
        artifact_group: Some(group.clone()),
        ..Default::default()
    }
    .resolve_task("observed-native-profile", &task::TaskContext::default())
    .expect("explicit native profile Task");
    let (completed_sender, completed_receiver) = futures::channel::oneshot::channel();
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(ObservedProfileTerminal {
            fail: false,
            expected_command: "fixture-profiler",
            expected_arguments: &["--sample", "10"],
        });
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            task,
            true,
            move |result, _| {
                completed_sender
                    .send(result)
                    .expect("Task completion receiver")
            },
            window,
            cx,
        );
    });
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(30));
    let approved_completion = async {
        while !cx.has_pending_prompt() {
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        let (message, detail) = cx.pending_prompt().expect("actual host capture review");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert_eq!(
            detail.lines().take(2).collect::<Vec<_>>(),
            [
                "Profile: /fixture/profile.json",
                "Symbols: /fixture/profile.syms.json"
            ]
        );
        assert!(detail.contains("raw files remain on that host"));
        assert_eq!(
            ready.borrow().len(),
            0,
            "held consent cannot publish a new host Task offer"
        );
        cx.simulate_prompt_answer("Capture");
        completed_receiver.await
    };
    let completion = match futures::future::select(Box::pin(approved_completion), deadline).await {
        futures::future::Either::Left((result, _)) => result.expect("Task completion"),
        futures::future::Either::Right(_) => panic!("Task completion timed out"),
    };
    assert_eq!(completion, workspace::tasks::ScheduledTaskResult::Success);
    cx.run_until_parked();
    assert_eq!(ready.borrow().len(), 1, "host-verified Task offer");
    let offer = ready.borrow()[0].clone();
    assert!(offer.host_verified());
    assert!(cx.read(|cx| offer.is_current(&project, cx)));
    assert_eq!(offer.group().format, "samply-0.13.1");
    assert_eq!(
        offer
            .path("profile")
            .expect("profile path")
            .path
            .as_unix_str(),
        "profile.json"
    );
    assert_eq!(
        offer
            .path("symbols")
            .expect("symbol path")
            .path
            .as_unix_str(),
        "profile.syms.json"
    );
    let imported = project
        .read_with(cx, |project, cx| {
            project.import_remote_task_profile(
                [
                    offer.path("profile").expect("profile path").clone(),
                    offer.path("symbols").expect("symbol path").clone(),
                ],
                offer.run_id().to_string(),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("verified Task import request")
        .await
        .expect("normalized Task profile transfer");
    assert_eq!(imported.total_samples, 30);
    let notification_id = workspace::notifications::NotificationId::unique::<
        workspace::tasks::CompletedTaskArtifactGroup,
    >();
    let notification = workspace
        .read_with(cx, |workspace, _| {
            workspace.message_notification_for_test(&notification_id)
        })
        .expect("host-verified profile import notification");
    assert!(notification.update_in(cx, |notification, window, cx| {
        notification.click_primary_for_test(window, cx)
    }));
    let view = workspace
        .read_with(cx, |workspace, cx| {
            workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
        })
        .expect("Task notification opens the native viewer");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while view.read_with(cx, |view, _| view.sample_count_for_test()) != Some(30) {
        assert!(
            std::time::Instant::now() < deadline,
            "Task notification did not display the host-verified profile"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
    }
    let client_fs = project.read_with(cx, |project, _| project.fs().clone());
    assert!(
        client_fs
            .metadata(Path::new("/fixture/profile.json"))
            .await
            .expect("client filesystem check")
            .is_none()
    );
    let failed_task = task::TaskTemplate {
        label: "Failed native profile".into(),
        command: "fixture-profiler".into(),
        args: vec!["--sample".into(), "10".into()],
        cwd: Some("/fixture".into()),
        save: task::SaveStrategy::None,
        artifact_group: Some(group.clone()),
        ..Default::default()
    }
    .resolve_task("failed-native-profile", &task::TaskContext::default())
    .expect("explicit failing profile Task");
    let (failed_sender, failed_receiver) = futures::channel::oneshot::channel();
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(ObservedProfileTerminal {
            fail: true,
            expected_command: "fixture-profiler",
            expected_arguments: &["--sample", "10"],
        });
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            failed_task,
            true,
            move |result, _| failed_sender.send(result).expect("failed Task receiver"),
            window,
            cx,
        );
    });
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(30));
    let approved_failure = async {
        while !cx.has_pending_prompt() {
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        let (message, detail) = cx.pending_prompt().expect("actual host capture review");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert_eq!(
            detail.lines().take(2).collect::<Vec<_>>(),
            [
                "Profile: /fixture/profile.json",
                "Symbols: /fixture/profile.syms.json"
            ]
        );
        assert!(detail.contains("raw files remain on that host"));
        assert_eq!(
            ready.borrow().len(),
            1,
            "held consent cannot publish a new host Task offer"
        );
        cx.simulate_prompt_answer("Capture");
        failed_receiver.await
    };
    let failure = match futures::future::select(Box::pin(approved_failure), deadline).await {
        futures::future::Either::Left((result, _)) => result.expect("failed Task completion"),
        futures::future::Either::Right(_) => panic!("failed Task completion timed out"),
    };
    assert_eq!(failure, workspace::tasks::ScheduledTaskResult::Failure);
    cx.run_until_parked();
    assert_eq!(
        ready.borrow().len(),
        1,
        "failed Task must not publish another offer"
    );
    assert!(cx.read(|cx| offer.is_current(&project, cx)));
    assert_eq!(
        view.read_with(cx, |view, _| view.sample_count_for_test()),
        Some(30)
    );
    fs.insert_file("/fixture/profile.json", b"{}".to_vec())
        .await;
    server_cx.run_until_parked();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while view
        .read_with(cx, |view, _| view.sample_count_for_test())
        .is_some()
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the native viewer retained a replaced Task artifact"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
    }
    assert!(!cx.read(|cx| offer.is_current(&project, cx)));
    assert!(view.read_with(cx, |view, _| {
        view.notice_for_test().contains("Profile cleared")
    }));
    fs.insert_file(
        "/fixture/profile.json",
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        )
        .to_vec(),
    )
    .await;
    server_cx.run_until_parked();
    let recovered_task = task::TaskTemplate {
        label: "Recovered native profile".into(),
        command: "/usr/bin/fixture-profiler".into(),
        args: vec!["--sample".into(), "20".into()],
        cwd: Some("/fixture".into()),
        save: task::SaveStrategy::None,
        artifact_group: Some(group.clone()),
        ..Default::default()
    }
    .resolve_task("recovered-native-profile", &task::TaskContext::default())
    .expect("explicit replacement profile Task");
    let (recovered_sender, recovered_receiver) = futures::channel::oneshot::channel();
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(ObservedProfileTerminal {
            fail: false,
            expected_command: "/usr/bin/fixture-profiler",
            expected_arguments: &["--sample", "20"],
        });
        workspace.schedule_resolved_task_with_completion(
            project::TaskSourceKind::UserInput,
            recovered_task,
            true,
            move |result, _| {
                recovered_sender
                    .send(result)
                    .expect("recovered Task receiver")
            },
            window,
            cx,
        );
    });
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(30));
    let approved_recovery = async {
        while !cx.has_pending_prompt() {
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        let (message, detail) = cx.pending_prompt().expect("actual host capture review");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert_eq!(
            detail.lines().take(2).collect::<Vec<_>>(),
            [
                "Profile: /fixture/profile.json",
                "Symbols: /fixture/profile.syms.json"
            ]
        );
        assert!(detail.contains("raw files remain on that host"));
        assert_eq!(
            ready.borrow().len(),
            1,
            "held consent cannot publish a new host Task offer"
        );
        cx.simulate_prompt_answer("Capture");
        recovered_receiver.await
    };
    let recovery = match futures::future::select(Box::pin(approved_recovery), deadline).await {
        futures::future::Either::Left((result, _)) => result.expect("recovered Task completion"),
        futures::future::Either::Right(_) => panic!("recovered Task completion timed out"),
    };
    assert_eq!(recovery, workspace::tasks::ScheduledTaskResult::Success);
    cx.run_until_parked();
    assert_eq!(
        ready.borrow().len(),
        2,
        "a fresh Task publishes one new offer"
    );
    let recovered_offer = ready.borrow()[1].clone();
    assert!(recovered_offer.host_verified());
    assert!(cx.read(|cx| recovered_offer.is_current(&project, cx)));
    assert!(!cx.read(|cx| offer.is_current(&project, cx)));
    let recovered_notification = workspace
        .read_with(cx, |workspace, _| {
            workspace.message_notification_for_test(&notification_id)
        })
        .expect("replacement profile import notification");
    assert!(
        recovered_notification.update_in(cx, |notification, window, cx| {
            notification.click_primary_for_test(window, cx)
        })
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while view.read_with(cx, |view, _| view.sample_count_for_test()) != Some(30) {
        assert!(
            std::time::Instant::now() < deadline,
            "the replacement Task did not reload the native viewer"
        );
        cx.background_executor
            .timer(std::time::Duration::from_millis(20))
            .await;
    }
    let cancelled_task = task::TaskTemplate {
        label: "Cancelled native profile".into(),
        command: "fixture-profiler".into(),
        cwd: Some("/fixture".into()),
        save: task::SaveStrategy::None,
        artifact_group: Some(group),
        ..Default::default()
    }
    .resolve_task("cancelled-native-profile", &task::TaskContext::default())
    .expect("explicit cancellable profile Task");
    let (observed_sender, observed_receiver) = futures::channel::oneshot::channel();
    let (release_sender, release_receiver) = futures::channel::oneshot::channel();
    let handle = workspace.update_in(cx, |workspace, window, cx| {
        workspace.set_terminal_provider(DelayedObservedProfileTerminal {
            observed: std::sync::Mutex::new(Some(observed_sender)),
            release: std::sync::Mutex::new(Some(release_receiver)),
        });
        workspace.schedule_resolved_task_with_structured_handle(
            project::TaskSourceKind::UserInput,
            cancelled_task,
            true,
            window,
            cx,
        )
    });
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(30));
    let approved_observation = async {
        while !cx.has_pending_prompt() {
            cx.background_executor
                .timer(std::time::Duration::from_millis(10))
                .await;
        }
        let (message, detail) = cx.pending_prompt().expect("actual host capture review");
        assert_eq!(message, "Capture a native profile on the project host?");
        assert_eq!(
            detail.lines().take(2).collect::<Vec<_>>(),
            [
                "Profile: /fixture/profile.json",
                "Symbols: /fixture/profile.syms.json"
            ]
        );
        assert!(detail.contains("raw files remain on that host"));
        assert_eq!(
            ready.borrow().len(),
            2,
            "held consent cannot publish a new host Task offer"
        );
        cx.simulate_prompt_answer("Capture");
        observed_receiver.await
    };
    match futures::future::select(Box::pin(approved_observation), deadline).await {
        futures::future::Either::Left((result, _)) => {
            result.expect("host observed Task completion")
        }
        futures::future::Either::Right(_) => panic!("host observer timed out"),
    }
    cx.update(|_, cx| {
        assert!(
            handle.cancel(cx),
            "running profile Task accepts cancellation"
        )
    });
    release_sender.send(()).expect("release late Task result");
    cx.run_until_parked();
    assert_eq!(
        ready.borrow().len(),
        2,
        "cancelled Task emits no native offer"
    );
    assert_eq!(
        view.read_with(cx, |view, _| view.sample_count_for_test()),
        Some(30)
    );
    assert!(cx.read(|cx| recovered_offer.is_current(&project, cx)));
    let remote = project.read_with(cx, |project, _| {
        project.remote_client().expect("remote client")
    });
    remote
        .update(cx, |remote, cx| remote.simulate_disconnect(cx))
        .await;
    cx.run_until_parked();
    assert_eq!(
        view.read_with(cx, |view, _| view.sample_count_for_test()),
        None
    );
    assert!(!cx.read(|cx| recovered_offer.is_current(&project, cx)));
    drop(recovered_notification);
    drop(recovered_offer);
    drop(view);
    drop(notification);
    drop(offer);
    ready.borrow_mut().clear();
    drop(_subscription);
    cx.deactivate_window();
    workspace.update_in(cx, |_, window, _| window.remove_window());
    let weak_workspace = workspace.downgrade();
    drop(workspace);
    let weak_project = project.downgrade();
    drop(project);
    cx.cx.update(|_| {});
    cx.run_until_parked();
    cx.cx.update(|_| {});
    assert!(weak_workspace.upgrade().is_none(), "workspace released");
    assert!(weak_project.upgrade().is_none(), "client project released");
    drop(headless);
    server_cx.run_until_parked();
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
#[gpui::test]
async fn profile_task_binding_requires_host_completion_and_sealed_artifacts(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    use self::peer_requests::FromPeer;
    use project::profile_results_wire::PROFILE_PROTOCOL_VERSION;

    fn set_host_opt_in(enabled: bool, cx: &mut gpui::App) {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            enabled,
            cx,
        );
        // Forwarded client user settings must not replace the host's own opt-in.
        SettingsStore::update_global(cx, |settings, cx| {
            settings
                .set_local_server_settings(
                    &json!({
                        "feature_flags": {
                            "rust-native-profiling-experimental": if enabled { "on" } else { "off" }
                        }
                    })
                    .to_string(),
                    cx,
                )
                .expect("independent host-local profiling settings");
        });
    }

    cx.executor().allow_parking();
    server_cx.executor().allow_parking();
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree("/fixture", json!({"workload.rs": "fn main() {}"}))
        .await;
    fs.insert_file(
        "/fixture/profile.json",
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        )
        .to_vec(),
    )
    .await;
    fs.insert_file(
        "/fixture/profile.syms.json",
        include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json").to_vec(),
    ).await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    server_cx.update(|cx| {
        set_host_opt_in(true, cx);
    });

    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/fixture", true, cx)
        })
        .await
        .expect("remote fixture");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
    let client = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    let owner = proto::PeerId {
        owner_id: 17,
        id: 1,
    };
    let other = proto::PeerId {
        owner_id: 17,
        id: 2,
    };
    let profile_path = proto::ProjectPath {
        worktree_id,
        path: "profile.json".into(),
    };
    let symbols_path = proto::ProjectPath {
        worktree_id,
        path: "profile.syms.json".into(),
    };
    let import = proto::ImportProfileResults {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        import_id: 31,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        profile_path: Some(profile_path.clone()),
        symbols_path: Some(symbols_path.clone()),
        consent_to_transfer: true,
        task_run_id: Some("observed-run".into()),
        artifact_format: "samply-0.13.1".into(),
    };
    let unproven = client
        .request(FromPeer {
            peer: owner,
            payload: import.clone(),
        })
        .await
        .expect("unproven import response");
    assert_eq!(unproven.status, proto::ProfileResultStatus::Stale as i32);
    let prepare = proto::PrepareProfileTaskBinding {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        task_run_id: "observed-run".into(),
        profile_path: Some(profile_path),
        symbols_path: Some(symbols_path),
        artifact_format: "samply-0.13.1".into(),
    };
    let prepared = client
        .request(FromPeer {
            peer: owner,
            payload: prepare.clone(),
        })
        .await
        .expect("prepare host record");
    assert_eq!(prepared.status, proto::ProfileResultStatus::Ready as i32);
    assert!(!prepared.complete);
    assert_eq!(prepared.token.as_deref().map(str::len), Some(64));
    let replay = client
        .request(FromPeer {
            peer: owner,
            payload: prepare.clone(),
        })
        .await
        .expect("replay response");
    assert_eq!(replay.status, proto::ProfileResultStatus::Error as i32);
    assert!(replay.token.is_none());
    let foreign = client
        .request(FromPeer {
            peer: other,
            payload: proto::GetProfileTaskBinding {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                task_run_id: "observed-run".into(),
            },
        })
        .await
        .expect("foreign record response");
    assert_eq!(foreign.status, proto::ProfileResultStatus::Stale as i32);
    let mut child = smol::process::Command::new(std::env::current_exe().expect("test executable"))
        .args([
            "--exact",
            "tests::native_task_completion_helper",
            "--nocapture",
        ])
        .env(
            "ZED_TEST_OBSERVER_SOCKET",
            prepared.socket_path.expect("socket"),
        )
        .env(
            util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
            prepared.token.expect("token"),
        )
        .kill_on_drop(true)
        .spawn()
        .expect("Task helper process");
    let sealed = client
        .request(FromPeer {
            peer: owner,
            payload: proto::GetProfileTaskBinding {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                task_run_id: "observed-run".into(),
            },
        })
        .await
        .expect("sealed record response");
    assert!(child.status().await.expect("Task completion").success());
    assert_eq!(sealed.status, proto::ProfileResultStatus::Ready as i32);
    assert!(sealed.complete);
    assert!(sealed.token.is_none() && sealed.socket_path.is_none());
    let page = client
        .request(FromPeer {
            peer: owner,
            payload: import.clone(),
        })
        .await
        .expect("bound import response");
    assert_eq!(page.status, proto::ProfileResultStatus::Ready as i32);
    let foreign_import = client
        .request(FromPeer {
            peer: other,
            payload: import.clone(),
        })
        .await
        .expect("foreign import response");
    assert_eq!(
        foreign_import.status,
        proto::ProfileResultStatus::Stale as i32
    );

    let mut failing_prepare = prepare.clone();
    failing_prepare.task_run_id = "failed-run".into();
    let failing = client
        .request(FromPeer {
            peer: owner,
            payload: failing_prepare,
        })
        .await
        .expect("failed run preparation");
    assert_eq!(failing.status, proto::ProfileResultStatus::Ready as i32);
    let mut failed_child =
        smol::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "tests::native_task_completion_helper",
                "--nocapture",
            ])
            .env(
                "ZED_TEST_OBSERVER_SOCKET",
                failing.socket_path.expect("socket"),
            )
            .env(
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                failing.token.expect("token"),
            )
            .env("ZED_TEST_OBSERVER_FAILURE", "1")
            .kill_on_drop(true)
            .spawn()
            .expect("failing Task helper process");
    let failed = client
        .request(FromPeer {
            peer: owner,
            payload: proto::GetProfileTaskBinding {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                task_run_id: "failed-run".into(),
            },
        })
        .await
        .expect("failed Task result");
    assert_eq!(
        failed_child
            .status()
            .await
            .expect("failing Task completion")
            .code(),
        Some(7)
    );
    assert_eq!(failed.status, proto::ProfileResultStatus::Error as i32);
    assert!(failed.complete);

    let mut reserved = Vec::new();
    for index in 3..=4 {
        let mut request = prepare.clone();
        request.task_run_id = format!("reserved-{index}");
        let binding = client
            .request(FromPeer {
                peer: owner,
                payload: request,
            })
            .await
            .expect("bounded preparation");
        assert_eq!(binding.status, proto::ProfileResultStatus::Ready as i32);
        assert!(binding.socket_path.is_some());
        reserved.push(format!("reserved-{index}"));
    }
    let mut overflow_request = prepare.clone();
    overflow_request.task_run_id = "overflow".into();
    let overflow = client
        .request(FromPeer {
            peer: owner,
            payload: overflow_request.clone(),
        })
        .await
        .expect("quota response");
    assert_eq!(overflow.status, proto::ProfileResultStatus::Error as i32);
    assert!(overflow.token.is_none());
    client
        .request(FromPeer {
            peer: owner,
            payload: proto::CancelProfileTaskBinding {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                task_run_id: reserved[0].clone(),
            },
        })
        .await
        .expect("release reserved slot");
    let released = client
        .request(FromPeer {
            peer: owner,
            payload: proto::GetProfileTaskBinding {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                task_run_id: reserved[0].clone(),
            },
        })
        .await
        .expect("released slot status");
    assert_eq!(released.status, proto::ProfileResultStatus::Stale as i32);
    let recovered = client
        .request(FromPeer {
            peer: owner,
            payload: overflow_request,
        })
        .await
        .expect("quota recovery");
    assert_eq!(recovered.status, proto::ProfileResultStatus::Ready as i32);
    assert!(recovered.socket_path.is_some());

    fs.pause_events();
    let mut changed_profile = include_bytes!(
        "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
    )
    .to_vec();
    changed_profile.push(b' ');
    fs.write("/fixture/profile.json".as_ref(), &changed_profile)
        .await
        .expect("in-place profile rewrite");
    let changed = client
        .request(FromPeer {
            peer: owner,
            payload: proto::ImportProfileResults {
                import_id: 32,
                ..import
            },
        })
        .await
        .expect("changed artifact response");
    assert_eq!(changed.status, proto::ProfileResultStatus::Error as i32);
    fs.unpause_events_and_flush();
    for run_id in ["observed-run", "failed-run", "reserved-4", "overflow"] {
        client
            .request(FromPeer {
                peer: owner,
                payload: proto::CancelProfileTaskBinding {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    task_run_id: run_id.into(),
                },
            })
            .await
            .expect("cancel binding");
    }
    for run_id in ["reserved-4", "overflow"] {
        let removed = client
            .request(FromPeer {
                peer: owner,
                payload: proto::GetProfileTaskBinding {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version: PROFILE_PROTOCOL_VERSION,
                    task_run_id: run_id.into(),
                },
            })
            .await
            .expect("cancelled binding status");
        assert_eq!(removed.status, proto::ProfileResultStatus::Stale as i32);
    }

    let paths = ["profile.json", "profile.syms.json"].map(|path| {
        project::ProjectPath::from_proto(proto::ProjectPath {
            worktree_id,
            path: path.into(),
        })
        .expect("visible profile path")
    });
    let binding = project
        .read_with(cx, |project, cx| {
            project.prepare_remote_profile_task("typed-client-run".into(), paths.clone(), cx)
        })
        .expect("typed client preparation")
        .await
        .expect("host binding");
    let mut typed_child =
        smol::process::Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "tests::native_task_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_OBSERVER_SOCKET", binding.socket_path)
            .env(
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                binding.token,
            )
            .kill_on_drop(true)
            .spawn()
            .expect("typed Task helper");
    project
        .read_with(cx, |project, cx| {
            project.get_remote_profile_task("typed-client-run".into(), cx)
        })
        .expect("typed completion request")
        .await
        .expect("host-observed Task completion");
    assert!(
        typed_child
            .status()
            .await
            .expect("typed helper exit")
            .success()
    );
    let opt_in_paths = paths.clone();
    let result = project
        .read_with(cx, |project, cx| {
            project.import_remote_task_profile(
                paths,
                "typed-client-run".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("typed import request")
        .await
        .expect("verified typed import");
    assert_eq!(result.total_samples, 30);
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            false,
            cx,
        )
    });
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        for error in [
            project
                .get_remote_profile_task("typed-client-run".into(), cx)
                .err(),
            project
                .import_remote_task_profile(
                    opt_in_paths.clone(),
                    "typed-client-run".into(),
                    Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    cx,
                )
                .err(),
            project
                .import_remote_profile(
                    opt_in_paths.clone(),
                    true,
                    Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    cx,
                )
                .err(),
        ] {
            assert_eq!(
                error
                    .expect("disabled public native route")
                    .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                Some(&project::profile_results_store::RemoteProfileError::Unsupported),
            );
        }
    });
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    for route in 0..3 {
        use futures::FutureExt as _;
        let cancellation = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let held = project.read_with(cx, |project, cx| match route {
            0 => project
                .get_remote_profile_task("typed-client-run".into(), cx)
                .expect("enabled polling admission")
                .boxed_local(),
            1 => project
                .import_remote_profile(opt_in_paths.clone(), true, cancellation.clone(), cx)
                .expect("enabled manual admission")
                .map(|result| result.map(|_| ()))
                .boxed_local(),
            _ => project
                .import_remote_task_profile(
                    opt_in_paths.clone(),
                    "typed-client-run".into(),
                    cancellation.clone(),
                    cx,
                )
                .expect("enabled task-bound admission")
                .map(|result| result.map(|_| ()))
                .boxed_local(),
        });
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                false,
                cx,
            )
        });
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(10));
        let result = match futures::future::select(held, deadline).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right(_) => {
                panic!("unpolled native admission revocation exceeded ten seconds")
            }
        };
        assert_eq!(
            result
                .err()
                .expect("Off then On before first poll revokes original admission")
                .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Unsupported),
        );
        if route != 0 {
            assert!(cancellation.load(std::sync::atomic::Ordering::Relaxed));
        }
    }
    for task_bound in [false, true] {
        let (started_sender, started) = futures::channel::oneshot::channel();
        let (release, released) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, _| {
                store.gate_next_import_for_test(started_sender, released);
            });
        });
        let cancellation = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let importing = project
            .read_with(cx, |project, cx| {
                if task_bound {
                    project.import_remote_task_profile(
                        opt_in_paths.clone(),
                        "typed-client-run".into(),
                        cancellation.clone(),
                        cx,
                    )
                } else {
                    project.import_remote_profile(
                        opt_in_paths.clone(),
                        true,
                        cancellation.clone(),
                        cx,
                    )
                }
            })
            .expect("enabled public native route reaches original host parser");
        let (result, ()) = {
            let deadline = cx
                .background_executor
                .timer(std::time::Duration::from_secs(10));
            let revoke = async {
                started
                    .await
                    .expect("original host parser is actually held");

                cx.update(|cx| {
                    project::task_store::set_experimental_feature_for_test(
                        "rust-native-profiling-experimental",
                        false,
                        cx,
                    )
                });
                cx.run_until_parked();
                assert!(cancellation.load(std::sync::atomic::Ordering::Relaxed));

                cx.update(|cx| {
                    project::task_store::set_experimental_feature_for_test(
                        "rust-native-profiling-experimental",
                        true,
                        cx,
                    )
                });
                drop(release);
            };
            let completion = async { futures::join!(importing, revoke) };
            futures::pin_mut!(completion);
            match futures::future::select(completion, deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => {
                    panic!(
                        "held native result revocation exceeded original ten-second fixture bound"
                    )
                }
            }
        };
        assert_eq!(
            result
                .err()
                .expect("Off then On cannot restore original import")
                .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Unsupported),
        );
        server_cx.run_until_parked();
        let recovered = project
            .read_with(cx, |project, cx| {
                if task_bound {
                    project.import_remote_task_profile(
                        opt_in_paths.clone(),
                        "typed-client-run".into(),
                        Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        cx,
                    )
                } else {
                    project.import_remote_profile(
                        opt_in_paths.clone(),
                        true,
                        Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        cx,
                    )
                }
            })
            .expect("fresh public request after local re-opt-in");
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(10));
        let result = match futures::future::select(recovered, deadline).await {
            futures::future::Either::Left((result, _)) => {
                result.expect("fresh original host profile import")
            }
            futures::future::Either::Right(_) => {
                panic!("fresh native result recovery exceeded ten seconds")
            }
        };
        assert_eq!(result.total_samples, 30);
    }
    let (started_sender, started) = futures::channel::oneshot::channel();
    let (release, released) = futures::channel::oneshot::channel();
    headless.update(server_cx, |headless, cx| {
        headless.profile_results_store.update(cx, |store, _| {
            store.gate_next_import_for_test(started_sender, released);
        });
    });
    let importing = project
        .read_with(cx, |project, cx| {
            project.import_remote_profile(
                opt_in_paths.clone(),
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("public manual import reaches the genuine host owner");
    let (result, ()) = {
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(10));
        let revoke = async {
            started.await.expect("manual native host parser is held");
            server_cx.update(|cx| {
                set_host_opt_in(false, cx);
            });
            server_cx.run_until_parked();
            let denied = project
                .read_with(cx, |project, cx| {
                    project.import_remote_profile(
                        opt_in_paths.clone(),
                        true,
                        Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        cx,
                    )
                })
                .expect("enabled client reaches disabled native host")
                .await;
            assert_eq!(
                denied
                    .err()
                    .expect("native host rejects manual import while Off")
                    .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                Some(&project::profile_results_store::RemoteProfileError::Unsupported),
            );
            server_cx.update(|cx| {
                set_host_opt_in(true, cx);
            });
            drop(release);
        };
        let completion = async { futures::join!(importing, revoke) };
        futures::pin_mut!(completion);
        match futures::future::select(completion, deadline).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right(_) => {
                panic!("manual native host revocation exceeded ten seconds")
            }
        }
    };
    assert_eq!(
        result
            .err()
            .expect("host Off then On cannot restore original manual import")
            .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Cancelled),
    );
    let recovery = project
        .read_with(cx, |project, cx| {
            project.import_remote_profile(
                opt_in_paths,
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("fresh manual import after original host re-opt-in");
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    let result = match futures::future::select(recovery, deadline).await {
        futures::future::Either::Left((result, _)) => result.expect("fresh manual host import"),
        futures::future::Either::Right(_) => panic!("manual host recovery exceeded ten seconds"),
    };
    assert_eq!(result.total_samples, 30);
    project
        .read_with(cx, |project, cx| {
            project.cancel_remote_profile_task("typed-client-run".into(), cx)
        })
        .expect("typed cancellation request")
        .await
        .expect("typed cancellation acknowledgement");
    assert!(
        project
            .read_with(cx, |project, cx| {
                project.get_remote_profile_task("typed-client-run".into(), cx)
            })
            .expect("typed stale request")
            .await
            .is_err(),
        "cancelled binding must not remain importable"
    );
    let preparation_paths = ["profile.json", "profile.syms.json"].map(|path| {
        project::ProjectPath::from_proto(proto::ProjectPath {
            worktree_id,
            path: path.into(),
        })
        .expect("unchanged visible preparation path")
    });
    let unpolled = project
        .read_with(cx, |project, cx| {
            project.prepare_remote_profile_task(
                "unpolled-preparation".into(),
                preparation_paths.clone(),
                cx,
            )
        })
        .expect("original preparation API admission");
    for enabled in [false, true] {
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                enabled,
                cx,
            )
        });
    }
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    let result = match futures::future::select(unpolled, deadline).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => panic!("unpolled preparation revocation timed out"),
    };
    assert_eq!(
        result
            .err()
            .expect("unpolled Off then On cannot issue an original binding")
            .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported),
    );

    let (started_sender, mut started) = futures::channel::oneshot::channel();
    let (release, released) = futures::channel::oneshot::channel();
    headless.update(server_cx, |headless, cx| {
        headless.profile_results_store.update(cx, |store, _| {
            store.gate_next_task_preparation_response_for_test(started_sender, released);
        });
    });
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            false,
            cx,
        )
    });
    let disabled = project.read_with(cx, |project, cx| {
        project
            .prepare_remote_profile_task(
                "synchronous-off-preparation".into(),
                preparation_paths.clone(),
                cx,
            )
            .err()
            .expect("disabled preparation refuses synchronously")
    });
    assert_eq!(
        disabled.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported),
    );
    cx.run_until_parked();
    server_cx.run_until_parked();
    assert_eq!(
        started
            .try_recv()
            .expect("host response gate remains owned"),
        None,
        "synchronous Off must not send a preparation request"
    );
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let enabled = project
        .read_with(cx, |project, cx| {
            project.prepare_remote_profile_task(
                "enabled-preparation-control".into(),
                preparation_paths.clone(),
                cx,
            )
        })
        .expect("fresh preparation admission");
    let completion = async {
        started
            .await
            .expect("fresh request creates a genuine host row and socket");
        release
            .send(())
            .expect("release genuine host preparation response");
        enabled
            .await
            .expect("fresh opted-in preparation returns the actual binding")
    };
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    let accepted = match futures::future::select(Box::pin(completion), deadline).await {
        futures::future::Either::Left((binding, _)) => binding,
        futures::future::Either::Right(_) => panic!("fresh preparation control timed out"),
    };
    assert!(!accepted.helper_path.is_empty() && !accepted.socket_path.is_empty());
    let completion = async {
        let duplicate = project
            .read_with(cx, |project, cx| {
                project.prepare_remote_profile_task(
                    "enabled-preparation-control".into(),
                    preparation_paths.clone(),
                    cx,
                )
            })
            .expect("duplicate API request")
            .await;
        assert_eq!(
            duplicate
                .err()
                .expect("host refuses duplicate issuance")
                .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Invalid),
        );
        let mut accepted_child = smol::process::Command::new(
            std::env::current_exe().expect("actual Task helper executable"),
        )
        .args([
            "--exact",
            "tests::native_task_completion_helper",
            "--nocapture",
        ])
        .env("ZED_TEST_OBSERVER_SOCKET", accepted.socket_path)
        .env(
            util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
            accepted.token,
        )
        .kill_on_drop(true)
        .spawn()
        .expect("launch previously returned genuine binding after duplicate refusal");
        project
            .read_with(cx, |project, cx| {
                project.get_remote_profile_task("enabled-preparation-control".into(), cx)
            })
            .expect("original accepted binding completion")
            .await
            .expect("duplicate refusal must not cancel an existing returned binding");
        assert!(
            accepted_child
                .status()
                .await
                .expect("actual accepted child status")
                .success()
        );
        project
            .read_with(cx, |project, cx| {
                project.cancel_remote_profile_task("enabled-preparation-control".into(), cx)
            })
            .expect("returned binding remains caller-owned")
            .await
            .expect("explicit original-run cancellation");
    };
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    match futures::future::select(Box::pin(completion), deadline).await {
        futures::future::Either::Left(((), _)) => {}
        futures::future::Either::Right(_) => panic!("accepted binding completion timed out"),
    }

    for (run_id, drop_future, acceptance_gate) in [
        ("revoked-held-preparation", false, false),
        ("dropped-held-preparation", true, false),
        ("buffered-revoked-preparation", false, true),
    ] {
        let (started_sender, started) = futures::channel::oneshot::channel();
        let (release, released) = futures::channel::oneshot::channel();
        if acceptance_gate {
            project
                .read_with(cx, |project, _| {
                    project
                        .profile_results_store_for_test()
                        .expect("client profile owner")
                        .clone()
                })
                .update(cx, |store, _| {
                    store.gate_next_task_preparation_acceptance_for_test(started_sender, released);
                });
        } else {
            headless.update(server_cx, |headless, cx| {
                headless.profile_results_store.update(cx, |store, _| {
                    store.gate_next_task_preparation_response_for_test(started_sender, released);
                });
            });
        }
        let pending = project
            .read_with(cx, |project, cx| {
                project.prepare_remote_profile_task(run_id.into(), preparation_paths.clone(), cx)
            })
            .expect("genuine original host preparation");
        let completion = async {
            started
                .await
                .expect("actual issued host row is held before client acceptance");
            assert!(
                server_cx.read(|cx| {
                    use feature_flags::FeatureFlagAppExt as _;
                    cx.has_flag::<feature_flags::RustNativeProfilingExperimentalFeatureFlag>()
                }),
                "host remains opted in; only client lifecycle is revoked"
            );
            if drop_future {
                drop(pending);
                release
                    .send(())
                    .expect("drain response after real preparation future destruction");
            } else {
                for enabled in [false, true] {
                    cx.update(|cx| {
                        project::task_store::set_experimental_feature_for_test(
                            "rust-native-profiling-experimental",
                            enabled,
                            cx,
                        )
                    });
                }
                release
                    .send(())
                    .expect("drain original response after client Off then On");
                assert_eq!(
                    pending
                        .await
                        .err()
                        .expect("original revoked preparation refuses")
                        .downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                    Some(&project::profile_results_store::RemoteProfileError::Unsupported),
                );
            }
            let retired = client
                .request(proto::GetProfileTaskBinding {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    protocol_version: PROFILE_PROTOCOL_VERSION,
                    task_run_id: run_id.into(),
                })
                .await
                .expect("actual original host-row retirement reply");
            assert_eq!(
                retired.status,
                proto::ProfileResultStatus::Stale as i32,
                "only the genuinely issued original binding must retire"
            );
            assert!(
                retired.helper_path.is_none()
                    && retired.socket_path.is_none()
                    && retired.token.is_none()
            );
            let fresh_run = format!("fresh-{run_id}");
            let fresh = project
                .read_with(cx, |project, cx| {
                    project.prepare_remote_profile_task(
                        fresh_run.clone(),
                        preparation_paths.clone(),
                        cx,
                    )
                })
                .expect("new opted-in preparation admission")
                .await
                .expect("fresh genuine row recovers after original cleanup");
            assert!(!fresh.socket_path.is_empty());
            project
                .read_with(cx, |project, cx| {
                    project.cancel_remote_profile_task(fresh_run, cx)
                })
                .expect("explicit fresh-row cleanup")
                .await
                .expect("fresh cleanup acknowledgement");
        };
        let deadline = cx
            .background_executor
            .timer(std::time::Duration::from_secs(10));
        match futures::future::select(Box::pin(completion), deadline).await {
            futures::future::Either::Left(((), _)) => {}
            futures::future::Either::Right(_) => {
                panic!("held preparation retirement/recovery timed out")
            }
        }
    }
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
#[gpui::test]
async fn profile_results_runtime_rejects_real_fifo_replacement(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    use project::profile_results_wire::PROFILE_PROTOCOL_VERSION;
    use std::os::unix::fs::FileTypeExt;

    cx.executor().allow_parking();
    server_cx.executor().allow_parking();
    let directory = tempfile::TempDir::new().expect("profile fixture directory");
    let profile = directory.path().join("profile.json");
    let symbols = directory.path().join("profile.syms.json");
    std::fs::write(
        &profile,
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        ),
    )
    .expect("captured profile");
    std::fs::write(
        &symbols,
        include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"),
    )
    .expect("captured symbols");
    let fs = Arc::new(fs::RealFs::new(None, server_cx.background_executor.clone()));
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    server_cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(directory.path(), true, cx)
        })
        .await
        .expect("remote real-fs profile worktree");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
    let client = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    let request = proto::ImportProfileResults {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        import_id: 1,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        profile_path: Some(proto::ProjectPath {
            worktree_id,
            path: "profile.json".into(),
        }),
        symbols_path: Some(proto::ProjectPath {
            worktree_id,
            path: "profile.syms.json".into(),
        }),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    };
    assert_eq!(
        client
            .request(request.clone())
            .await
            .expect("regular host profile response")
            .status,
        proto::ProfileResultStatus::Ready as i32
    );

    let (started_sender, started) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    headless.update(server_cx, |headless, cx| {
        headless.profile_results_store.update(cx, |store, _| {
            store.gate_next_import_for_test(started_sender, release_receiver);
        });
    });
    let mut replacement = request;
    replacement.import_id = 2;
    let importing = client.request(replacement);
    let replacing = async {
        started.await.expect("host parser admitted regular input");
        std::fs::remove_file(&profile).expect("remove admitted profile");
        assert!(
            smol::process::Command::new("/usr/bin/mkfifo")
                .arg(&profile)
                .status()
                .await
                .expect("create FIFO replacement")
                .success()
        );
        assert!(
            std::fs::metadata(&profile)
                .expect("FIFO replacement metadata")
                .file_type()
                .is_fifo()
        );
        release.send(()).expect("host parser remains held");
    };
    let completion = async { futures::join!(importing, replacing) };
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    futures::pin_mut!(completion);
    let (response, ()) = match futures::future::select(completion, deadline).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => {
            panic!("host FIFO replacement did not complete within ten seconds")
        }
    };
    let response = response.expect("host FIFO replacement response");
    assert!(
        response.status == proto::ProfileResultStatus::Error as i32
            || response.status == proto::ProfileResultStatus::Cancelled as i32,
        "the loader or filesystem watcher must reject the replaced input: {}",
        response.status
    );
    assert!(response.frames.is_empty() && response.nodes.is_empty());
    assert_eq!(
        client
            .request(proto::GetProfileResultPage {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: 2,
                generation: response.generation,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index: 0,
            })
            .await
            .expect("replaced FIFO page request")
            .status,
        proto::ProfileResultStatus::Stale as i32
    );
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
#[gpui::test]
async fn profile_results_runtime_rejects_real_parent_symlink_replacement(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    use project::profile_results_wire::PROFILE_PROTOCOL_VERSION;
    use std::os::unix::fs::symlink;

    cx.executor().allow_parking();
    server_cx.executor().allow_parking();
    let directory = tempfile::TempDir::new().expect("profile fixture directory");
    let artifacts = directory.path().join("artifacts");
    std::fs::create_dir(&artifacts).expect("artifact directory");
    std::fs::write(
        artifacts.join("profile.json"),
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        ),
    )
    .expect("captured profile");
    std::fs::write(
        artifacts.join("profile.syms.json"),
        include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"),
    )
    .expect("captured symbols");
    let fs = Arc::new(fs::RealFs::new(None, server_cx.background_executor.clone()));
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    server_cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(directory.path(), true, cx)
        })
        .await
        .expect("remote real-fs profile worktree");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
    let client = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    let request = proto::ImportProfileResults {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        import_id: 1,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        profile_path: Some(proto::ProjectPath {
            worktree_id,
            path: "artifacts/profile.json".into(),
        }),
        symbols_path: Some(proto::ProjectPath {
            worktree_id,
            path: "artifacts/profile.syms.json".into(),
        }),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    };
    assert_eq!(
        client
            .request(request.clone())
            .await
            .expect("regular host profile response")
            .status,
        proto::ProfileResultStatus::Ready as i32
    );

    let (started_sender, started) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    headless.update(server_cx, |headless, cx| {
        headless.profile_results_store.update(cx, |store, _| {
            store.gate_next_import_for_test(started_sender, release_receiver);
        });
    });
    let mut replacement = request;
    replacement.import_id = 2;
    let importing = client.request(replacement);
    let replacing = async {
        started.await.expect("host parser admitted regular inputs");
        let moved = directory.path().join("moved-artifacts");
        std::fs::rename(&artifacts, &moved).expect("move admitted artifact directory");
        symlink(&moved, &artifacts).expect("replace artifact parent with symlink");
        assert!(
            std::fs::symlink_metadata(&artifacts)
                .expect("replacement parent metadata")
                .file_type()
                .is_symlink()
        );
        release.send(()).expect("host parser remains held");
    };
    let completion = async { futures::join!(importing, replacing) };
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    futures::pin_mut!(completion);
    let (response, ()) = match futures::future::select(completion, deadline).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => {
            panic!("host parent replacement did not complete within ten seconds")
        }
    };
    let response = response.expect("host parent replacement response");
    assert!(
        response.status == proto::ProfileResultStatus::Error as i32
            || response.status == proto::ProfileResultStatus::Cancelled as i32,
        "the loader or filesystem watcher must reject the replaced parent: {}",
        response.status
    );
    assert!(response.frames.is_empty() && response.nodes.is_empty());
    assert_eq!(
        client
            .request(proto::GetProfileResultPage {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: 2,
                generation: response.generation,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index: 0,
            })
            .await
            .expect("replaced parent page request")
            .status,
        proto::ProfileResultStatus::Stale as i32
    );
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
#[gpui::test]
async fn profile_results_runtime_rejects_real_root_replacement_with_original_files(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    use project::profile_results_wire::PROFILE_PROTOCOL_VERSION;
    use std::os::unix::fs::MetadataExt;

    cx.executor().allow_parking();
    server_cx.executor().allow_parking();
    let directory = tempfile::TempDir::new().expect("profile fixture directory");
    let root = directory.path().join("workspace");
    let artifacts = root.join("artifacts");
    std::fs::create_dir_all(&artifacts).expect("artifact directory");
    std::fs::write(
        artifacts.join("profile.json"),
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        ),
    )
    .expect("captured profile");
    std::fs::write(
        artifacts.join("profile.syms.json"),
        include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"),
    )
    .expect("captured symbols");
    let original_root_inode = std::fs::metadata(&root)
        .expect("original root metadata")
        .ino();
    let fs = Arc::new(fs::RealFs::new(None, server_cx.background_executor.clone()));
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    server_cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(&root, true, cx)
        })
        .await
        .expect("remote real-fs profile worktree");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
    let client = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    let request = proto::ImportProfileResults {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        import_id: 1,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        profile_path: Some(proto::ProjectPath {
            worktree_id,
            path: "artifacts/profile.json".into(),
        }),
        symbols_path: Some(proto::ProjectPath {
            worktree_id,
            path: "artifacts/profile.syms.json".into(),
        }),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    };
    assert_eq!(
        client
            .request(request.clone())
            .await
            .expect("regular host profile response")
            .status,
        proto::ProfileResultStatus::Ready as i32
    );

    let (started_sender, started) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    headless.update(server_cx, |headless, cx| {
        headless.profile_results_store.update(cx, |store, _| {
            store.gate_next_import_for_test(started_sender, release_receiver);
        });
    });
    let mut replacement = request;
    replacement.import_id = 2;
    let importing = client.request(replacement);
    let replacing = async {
        started.await.expect("host parser admitted original root");
        let moved_root = directory.path().join("moved-workspace");
        std::fs::rename(&root, &moved_root).expect("move admitted worktree root");
        std::fs::create_dir_all(&artifacts).expect("replace worktree root");
        for name in ["profile.json", "profile.syms.json"] {
            std::fs::hard_link(
                moved_root.join("artifacts").join(name),
                artifacts.join(name),
            )
            .expect("retain admitted artifact file identity");
        }
        assert_ne!(
            std::fs::metadata(&root)
                .expect("replacement root metadata")
                .ino(),
            original_root_inode
        );
        release.send(()).expect("host parser remains held");
    };
    let completion = async { futures::join!(importing, replacing) };
    let deadline = cx
        .background_executor
        .timer(std::time::Duration::from_secs(10));
    futures::pin_mut!(completion);
    let (response, ()) = match futures::future::select(completion, deadline).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => {
            panic!("host root replacement did not complete within ten seconds")
        }
    };
    let response = response.expect("host root replacement response");
    assert!(
        response.status == proto::ProfileResultStatus::Error as i32
            || response.status == proto::ProfileResultStatus::Cancelled as i32,
        "the loader or filesystem watcher must reject the replaced root: {}",
        response.status
    );
    assert!(response.frames.is_empty() && response.nodes.is_empty());
    assert_eq!(
        client
            .request(proto::GetProfileResultPage {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: 2,
                generation: response.generation,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index: 0,
            })
            .await
            .expect("replaced root page request")
            .status,
        proto::ProfileResultStatus::Stale as i32
    );
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
#[gpui::test]
async fn profile_results_runtime_imports_on_host_and_checks_admission(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    use project::profile_results_wire::{PROFILE_PROTOCOL_VERSION, ProfilePageAssembler};
    use std::sync::atomic::AtomicBool;

    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(
        "/fixture",
        json!({"workload.rs": "fn main() {}", "invalid.json": "invalid"}),
    )
    .await;
    fs.insert_file(
        "/fixture/profile.json",
        include_bytes!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        )
        .to_vec(),
    )
    .await;
    fs.insert_file("/fixture/profile.syms.json", include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json").to_vec()).await;
    fs.insert_symlink(
        "/fixture/profile-link.json",
        PathBuf::from("/fixture/profile.json"),
    )
    .await;
    let (project, headless) = init_test(&fs, cx, server_cx).await;
    cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    server_cx.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree("/fixture", true, cx)
        })
        .await
        .expect("remote fixture");
    cx.run_until_parked();
    server_cx.run_until_parked();
    let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
    headless.read_with(server_cx, |headless, cx| {
        let worktree = headless
            .worktree_store
            .read(cx)
            .worktree_for_id(project::WorktreeId::from_proto(worktree_id), cx)
            .expect("host profile worktree");
        let snapshot = worktree.read(cx).snapshot();
        let linked = snapshot
            .entry_for_path(rel_path("profile-link.json"))
            .expect("scanned internal profile symlink");
        assert!(linked.is_file() && !linked.is_external);
        assert!(linked.canonical_path.is_some());
    });
    let client = project.read_with(cx, |project, cx| {
        project
            .remote_client()
            .expect("remote client")
            .read(cx)
            .proto_client()
    });
    let request = proto::ImportProfileResults {
        project_id: proto::REMOTE_SERVER_PROJECT_ID,
        import_id: 1,
        protocol_version: PROFILE_PROTOCOL_VERSION,
        profile_path: Some(proto::ProjectPath {
            worktree_id,
            path: "profile.json".into(),
        }),
        symbols_path: Some(proto::ProjectPath {
            worktree_id,
            path: "profile.syms.json".into(),
        }),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    };
    let page = client
        .request(request.clone())
        .await
        .expect("registered import handler");
    assert_eq!(page.status, proto::ProfileResultStatus::Ready as i32);
    assert!(page.complete);
    let generation = page.generation;
    let mut assembler = ProfilePageAssembler::new(1, generation).expect("import identity");
    let cancel = AtomicBool::new(false);
    assembler
        .push(page, &cancel, |path| {
            path.worktree_id.to_proto() == worktree_id
        })
        .expect("bounded normalized page");
    let result = assembler
        .finish(&cancel, |_| true)
        .expect("complete host model");
    assert_eq!(result.total_samples, 30);
    assert!(result.frames.iter().any(|frame| {
        frame.label == "workload::leaf"
            && frame
                .source
                .as_ref()
                .is_some_and(|(path, _)| path.path.as_unix_str() == "workload.rs")
    }));
    let stale = client
        .request(proto::GetProfileResultPage {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 1,
            generation,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            page_index: 0,
        })
        .await
        .expect("registered page handler");
    assert_eq!(stale.status, proto::ProfileResultStatus::Stale as i32);

    let mut denied = request.clone();
    denied.consent_to_transfer = false;
    assert_eq!(
        client
            .request(denied)
            .await
            .expect("consent response")
            .status,
        proto::ProfileResultStatus::Restricted as i32
    );
    let mut mismatch = request.clone();
    mismatch.protocol_version += 1;
    assert_eq!(
        client
            .request(mismatch)
            .await
            .expect("version response")
            .status,
        proto::ProfileResultStatus::Mismatch as i32
    );
    let mut missing = request.clone();
    missing.profile_path.as_mut().expect("fixture input").path = "missing.json".into();
    assert_eq!(
        client
            .request(missing)
            .await
            .expect("visibility response")
            .status,
        proto::ProfileResultStatus::Restricted as i32
    );
    let mut malformed = request.clone();
    malformed.profile_path.as_mut().expect("fixture input").path = "invalid.json".into();
    assert_eq!(
        client
            .request(malformed)
            .await
            .expect("parse response")
            .status,
        proto::ProfileResultStatus::Error as i32
    );
    let mut linked = request.clone();
    linked.profile_path.as_mut().expect("fixture input").path = "profile-link.json".into();
    let linked_result = client
        .request(linked)
        .await
        .expect("linked profile response");
    assert_eq!(
        linked_result.status,
        proto::ProfileResultStatus::Error as i32
    );
    assert!(linked_result.frames.is_empty() && linked_result.nodes.is_empty());
    client
        .request(proto::CancelProfileImport {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 2,
            generation: 0,
        })
        .await
        .expect("registered cancel handler");
    let mut cancelled = request.clone();
    cancelled.import_id = 2;
    assert_eq!(
        client
            .request(cancelled)
            .await
            .expect("pre-cancel response")
            .status,
        proto::ProfileResultStatus::Cancelled as i32
    );

    let (started_sender, started) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    headless.update(server_cx, |headless, cx| {
        headless.profile_results_store.update(cx, |store, _| {
            store.gate_next_import_for_test(started_sender, release_receiver);
        });
    });
    let mut in_flight = request.clone();
    in_flight.import_id = 3;
    let importing = client.request(in_flight.clone());
    let cancelling = async {
        started.await.expect("host parsing started");
        client
            .request(proto::CancelProfileImport {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: 3,
                generation: 0,
            })
            .await
            .expect("cancel active host import");
        release.send(()).expect("host parser is still held");
    };
    let (cancelled_page, ()) = futures::join!(importing, cancelling);
    let cancelled_page = cancelled_page.expect("active cancellation response");
    assert_eq!(
        cancelled_page.status,
        proto::ProfileResultStatus::Cancelled as i32,
        "cancelling an active host parse must not return normalized data"
    );
    assert_eq!(
        client
            .request(proto::GetProfileResultPage {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: 3,
                generation: cancelled_page.generation,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index: 0,
            })
            .await
            .expect("cancelled import cannot retain pages")
            .status,
        proto::ProfileResultStatus::Stale as i32
    );
    assert_eq!(
        client
            .request(in_flight)
            .await
            .expect("fresh import after active cancellation")
            .status,
        proto::ProfileResultStatus::Ready as i32,
        "active cancellation must release host capacity"
    );

    headless.update(server_cx, |headless, cx| {
        project::trusted_worktrees::init(HashMap::default(), cx);
        project::trusted_worktrees::track_worktree_trust(
            headless.worktree_store.clone(),
            None,
            None,
            None,
            cx,
        );
    });
    client
        .request(proto::RestrictWorktrees {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            worktree_ids: vec![worktree_id],
        })
        .await
        .expect("restriction handler");
    assert_eq!(
        client
            .request(request)
            .await
            .expect("restricted host response")
            .status,
        proto::ProfileResultStatus::Restricted as i32
    );
}

#[cfg(feature = "rust-tools")]
mod peer_requests {
    #[derive(Clone, Debug, serde::Serialize)]
    pub(super) struct FromPeer<T> {
        pub(super) peer: proto::PeerId,
        pub(super) payload: T,
    }

    impl<T: proto::RequestMessage> proto::EnvelopedMessage for FromPeer<T> {
        const NAME: &'static str = T::NAME;
        const PRIORITY: proto::MessagePriority = T::PRIORITY;

        fn into_envelope(
            self,
            id: u32,
            responding_to: Option<u32>,
            _: Option<proto::PeerId>,
        ) -> proto::Envelope {
            self.payload
                .into_envelope(id, responding_to, Some(self.peer))
        }

        fn from_envelope(envelope: proto::Envelope) -> Option<Self> {
            let peer = envelope.original_sender_id?;
            T::from_envelope(envelope).map(|payload| Self { peer, payload })
        }
    }

    impl<T: proto::RequestMessage> proto::RequestMessage for FromPeer<T> {
        type Response = T::Response;
    }
}

#[cfg(all(feature = "rust-tools", target_os = "macos"))]
mod profile_transport_tests {
    use super::{peer_requests::FromPeer, *};
    use gpui::BorrowAppContext as _;
    use project::profile_results_wire::PROFILE_PROTOCOL_VERSION;

    #[gpui::test]
    async fn profile_results_runtime_relay_identity_and_transport_invalidation(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree("/fixture", json!({
            "profile.json": include_str!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"),
            "profile.syms.json": include_str!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json"),
        })).await;
        fs.insert_file("/fixture/profile-paged.json", paged_capture())
            .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        server_cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree("/fixture", true, cx)
            })
            .await
            .expect("remote fixture");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let transport = server_cx.new(|_| crate::NativeProjectTransport::default());
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, cx| {
                store.observe_transport(&transport, false, cx)
            });
        });
        let request = proto::ImportProfileResults {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 4,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            profile_path: Some(proto::ProjectPath {
                worktree_id,
                path: "profile.json".into(),
            }),
            symbols_path: Some(proto::ProjectPath {
                worktree_id,
                path: "profile.syms.json".into(),
            }),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        };
        assert_eq!(
            client
                .request(request.clone())
                .await
                .expect("disconnected response")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );
        transport.update(server_cx, |_, cx| cx.emit(true));
        let first = proto::PeerId { owner_id: 7, id: 1 };
        let second = proto::PeerId { owner_id: 7, id: 2 };
        client
            .request(FromPeer {
                peer: first,
                payload: proto::CancelProfileImport {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    import_id: 4,
                    generation: 0,
                },
            })
            .await
            .expect("first peer cancellation");
        let second_page = client
            .request(FromPeer {
                peer: second,
                payload: request.clone(),
            })
            .await
            .expect("second peer response");
        assert_eq!(second_page.status, proto::ProfileResultStatus::Ready as i32);
        assert_eq!(
            client
                .request(FromPeer {
                    peer: first,
                    payload: request.clone()
                })
                .await
                .expect("first peer response")
                .status,
            proto::ProfileResultStatus::Cancelled as i32
        );
        client
            .request(FromPeer {
                peer: first,
                payload: proto::CancelProfileImport {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    import_id: 8,
                    generation: 0,
                },
            })
            .await
            .expect("pending cancellation before disconnect");
        transport.update(server_cx, |_, cx| cx.emit(false));
        assert_eq!(
            client
                .request(request.clone())
                .await
                .expect("disconnected response")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );
        transport.update(server_cx, |_, cx| cx.emit(true));
        let mut fresh = request.clone();
        fresh.import_id = 8;
        let fresh = client
            .request(FromPeer {
                peer: first,
                payload: fresh,
            })
            .await
            .expect("reconnected response");
        assert_eq!(fresh.status, proto::ProfileResultStatus::Ready as i32);
        assert!(fresh.generation > second_page.generation);

        let (started_sender, started) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, _| {
                store.gate_next_import_for_test(started_sender, release_receiver);
            });
        });
        let mut in_flight = request;
        in_flight.import_id = 9;
        let importing = client.request(FromPeer {
            peer: first,
            payload: in_flight.clone(),
        });
        let invalidating = async {
            started
                .await
                .expect("host parser is held before reading inputs");
            transport.update(server_cx, |_, cx| cx.emit(false));
            transport.update(server_cx, |_, cx| cx.emit(true));
            release.send(()).expect("host parser is still held");
        };
        let (invalidated_page, ()) = futures::join!(importing, invalidating);
        let invalidated_page = invalidated_page.expect("invalidated import response");
        assert_eq!(
            invalidated_page.status,
            proto::ProfileResultStatus::Cancelled as i32,
            "transport loss must discard a parse started in the old session"
        );
        assert!(invalidated_page.frames.is_empty() && invalidated_page.nodes.is_empty());
        assert_eq!(
            client
                .request(FromPeer {
                    peer: first,
                    payload: proto::GetProfileResultPage {
                        project_id: proto::REMOTE_SERVER_PROJECT_ID,
                        import_id: 9,
                        generation: invalidated_page.generation,
                        protocol_version: PROFILE_PROTOCOL_VERSION,
                        page_index: 0,
                    },
                })
                .await
                .expect("discarded stream page result")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );
        in_flight.import_id = 10;
        assert_eq!(
            client
                .request(FromPeer {
                    peer: first,
                    payload: in_flight.clone(),
                })
                .await
                .expect("fresh import after transport recovery")
                .status,
            proto::ProfileResultStatus::Ready as i32
        );

        let (started_sender, started) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, _| {
                store.gate_next_import_for_test(started_sender, release_receiver);
            });
        });
        in_flight.import_id = 11;
        let importing = client.request(FromPeer {
            peer: first,
            payload: in_flight.clone(),
        });
        let forgetting = async {
            started
                .await
                .expect("host parser is held before peer disposal");
            headless.update(server_cx, |headless, cx| {
                headless.profile_results_store.update(cx, |store, _| {
                    store.forget_peer(first);
                });
            });
            release.send(()).expect("host parser is still held");
        };
        let (forgotten_page, ()) = futures::join!(importing, forgetting);
        let forgotten_page = forgotten_page.expect("forgotten peer import response");
        assert_eq!(
            forgotten_page.status,
            proto::ProfileResultStatus::Cancelled as i32,
            "peer disposal must discard an active import"
        );
        assert!(forgotten_page.frames.is_empty() && forgotten_page.nodes.is_empty());
        assert_eq!(
            client
                .request(FromPeer {
                    peer: first,
                    payload: proto::GetProfileResultPage {
                        project_id: proto::REMOTE_SERVER_PROJECT_ID,
                        import_id: 11,
                        generation: forgotten_page.generation,
                        protocol_version: PROFILE_PROTOCOL_VERSION,
                        page_index: 0,
                    },
                })
                .await
                .expect("forgotten peer page result")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );
        in_flight.import_id = 12;
        assert_eq!(
            client
                .request(FromPeer {
                    peer: first,
                    payload: in_flight.clone(),
                })
                .await
                .expect("fresh import after peer disposal")
                .status,
            proto::ProfileResultStatus::Ready as i32
        );

        let mut paged = in_flight;
        paged.import_id = 13;
        paged.profile_path = Some(proto::ProjectPath {
            worktree_id,
            path: "profile-paged.json".into(),
        });
        let first_page = client
            .request(FromPeer {
                peer: first,
                payload: paged.clone(),
            })
            .await
            .expect("multi-page import before transport loss");
        assert_eq!(first_page.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!first_page.complete);
        let (started_sender, started) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, _| {
                assert_eq!(store.active_import_count_for_test(), 1);
                store.gate_next_page_for_test(started_sender, release_receiver);
            });
        });
        let pending_page = client.request(FromPeer {
            peer: first,
            payload: proto::GetProfileResultPage {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                import_id: 13,
                generation: first_page.generation,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index: 1,
            },
        });
        let losing_transport = async {
            started.await.expect("host admitted the pending page");
            transport.update(server_cx, |_, cx| cx.emit(false));
            headless.read_with(server_cx, |headless, cx| {
                assert_eq!(
                    headless
                        .profile_results_store
                        .read_with(cx, |store, _| store.active_import_count_for_test()),
                    0,
                    "transport loss releases the in-flight page's retained profile"
                );
            });
            transport.update(server_cx, |_, cx| cx.emit(true));
            release.send(()).expect("old page handler remains held");
        };
        let (old_page, ()) = futures::join!(pending_page, losing_transport);
        let old_page = old_page.expect("invalidated page response");
        assert_eq!(old_page.status, proto::ProfileResultStatus::Stale as i32);
        assert!(old_page.frames.is_empty() && old_page.nodes.is_empty());
        paged.import_id = 14;
        assert_eq!(
            client
                .request(FromPeer {
                    peer: first,
                    payload: paged,
                })
                .await
                .expect("fresh paged import after transport recovery")
                .status,
            proto::ProfileResultStatus::Ready as i32
        );
    }
    pub(super) fn paged_capture() -> Vec<u8> {
        let mut profile: serde_json::Value = serde_json::from_str(include_str!(
            "../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json"
        ))
        .expect("captured JSON");
        let thread = &mut profile["threads"][0];
        let old_frames = thread["frameTable"]["length"]
            .as_u64()
            .expect("frame count") as usize;
        let old_functions = thread["funcTable"]["length"]
            .as_u64()
            .expect("function count") as usize;
        let old_stacks = thread["stackTable"]["length"]
            .as_u64()
            .expect("stack count") as usize;
        let old_samples = thread["samples"]["length"].as_u64().expect("sample count") as usize;
        let old_strings = thread["stringArray"].as_array().expect("strings").len();
        const EXTRA: usize = 3_000;
        for name in ["frameTable", "funcTable", "stackTable", "samples"] {
            let table = thread[name].as_object_mut().expect("table");
            let length = table["length"].as_u64().expect("table length") as usize;
            for column in table
                .values_mut()
                .filter_map(serde_json::Value::as_array_mut)
            {
                let fill = column.last().cloned().expect("nonempty captured column");
                column.resize(length + EXTRA, fill);
            }
            table.insert("length".into(), json!(length + EXTRA));
        }
        for index in 0..EXTRA {
            thread["stringArray"]
                .as_array_mut()
                .expect("strings")
                .push(json!(format!("synthetic_{index:04}_{}", "x".repeat(400))));
            thread["funcTable"]["name"][old_functions + index] = json!(old_strings + index);
            thread["funcTable"]["resource"][old_functions + index] = json!(-1);
            thread["frameTable"]["func"][old_frames + index] = json!(old_functions + index);
            thread["frameTable"]["address"][old_frames + index] = json!(0);
            thread["stackTable"]["frame"][old_stacks + index] = json!(old_frames + index);
            thread["stackTable"]["prefix"][old_stacks + index] = serde_json::Value::Null;
            thread["samples"]["stack"][old_samples + index] = json!(old_stacks + index);
            thread["samples"]["weight"][old_samples + index] = json!(1);
        }
        serde_json::to_vec(&profile).expect("synthetic scale input")
    }

    #[gpui::test]
    async fn profile_results_runtime_client_assembles_host_pages(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use gpui::{BorrowAppContext, Focusable, InputEvent as _};
        use open_path_prompt::{OpenPathDelegate, OpenPathPrompt};
        use picker::{Picker, PickerDelegate};
        use project::profile_results_store::RemoteProfileError;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree("/fixture", json!({"workload.rs": "fn main() {}"}))
            .await;
        fs.insert_file("/fixture/profile.json", paged_capture())
            .await;
        let mut replacement_profile: serde_json::Value =
            serde_json::from_slice(&paged_capture()).expect("replacement capture JSON");
        *replacement_profile["threads"][0]["samples"]["weight"]
            .as_array_mut()
            .expect("replacement sample weights")
            .last_mut()
            .expect("last replacement sample") = json!(2);
        fs.insert_file(
            "/fixture/profile-replacement.json",
            serde_json::to_vec(&replacement_profile).expect("replacement capture bytes"),
        )
        .await;
        fs.insert_file("/fixture/profile.syms.json", include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json").to_vec()).await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        server_cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        project
            .update(cx, |project, cx| {
                project.find_or_create_worktree("/fixture", true, cx)
            })
            .await
            .expect("remote fixture");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let paths = project.read_with(cx, |project, cx| {
            [
                project
                    .find_project_path("/fixture/profile.json", cx)
                    .expect("profile path"),
                project
                    .find_project_path("/fixture/profile.syms.json", cx)
                    .expect("symbols path"),
            ]
        });
        let client_fs = project.read_with(cx, |project, _| project.fs().clone());
        assert!(!client_fs.is_file(Path::new("/fixture/profile.json")).await);
        let denied = project
            .read_with(cx, |project, cx| {
                project.import_remote_profile(
                    paths.clone(),
                    false,
                    Arc::new(AtomicBool::new(false)),
                    cx,
                )
            })
            .expect_err("explicit transfer consent required");
        assert_eq!(
            denied.downcast_ref::<RemoteProfileError>(),
            Some(&RemoteProfileError::Restricted)
        );
        let cancelled = project
            .read_with(cx, |project, cx| {
                project.import_remote_profile(
                    paths.clone(),
                    true,
                    Arc::new(AtomicBool::new(true)),
                    cx,
                )
            })
            .expect("cancelled import task")
            .await
            .expect_err("pre-cancelled client");
        assert_eq!(
            cancelled.downcast_ref::<RemoteProfileError>(),
            Some(&RemoteProfileError::Cancelled)
        );
        for _ in 0..2 {
            let result = project
                .read_with(cx, |project, cx| {
                    project.import_remote_profile(
                        paths.clone(),
                        true,
                        Arc::new(AtomicBool::new(false)),
                        cx,
                    )
                })
                .expect("explicit host import")
                .await
                .expect("assembled normalized host pages");
            assert_eq!(result.total_samples, 3030);
            assert!(result.frames.iter().any(|frame| frame.source.is_some()));
        }
        assert!(!client_fs.is_file(Path::new("/fixture/profile.json")).await);

        cx.update(|cx| {
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            language_tools::init(cx);
        });
        let (container, cx) = cx.add_window_view(|window, cx| {
            workspace::MultiWorkspace::test_new(project.clone(), window, cx)
        });
        let workspace = container.read_with(cx, |container, _| container.workspace().clone());
        let selections = Arc::new(AtomicUsize::new(0));
        workspace.update(cx, |workspace, _| {
            let selections = selections.clone();
            workspace.set_prompt_for_open_path(Box::new(move |_, lister, _, cx| {
                assert!(!lister.is_local(cx), "remote project path picker");
                selections.fetch_add(1, Ordering::Relaxed);
                let (sender, receiver) = futures::channel::oneshot::channel();
                sender
                    .send(Some(vec![
                        "/fixture/profile.json".into(),
                        "/fixture/profile.syms.json".into(),
                    ]))
                    .expect("profile path prompt receiver");
                receiver
            }));
        });
        cx.run_until_parked();
        workspace.update_in(cx, |workspace, window, cx| {
            window.focus(&workspace.focus_handle(cx), cx);
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        cx.dispatch_action(language_tools::profile_view::OpenNativeProfile);
        cx.run_until_parked();
        let view = workspace
            .read_with(cx, |workspace, cx| {
                workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
            })
            .expect("registered action opens native profile view in remote Workspace");
        let focus = view.read_with(cx, |view, cx| view.focus_handle(cx));
        let dispatch = |window: &mut gpui::Window, cx: &mut gpui::App, key: &str| {
            window.refresh();
            window.draw(cx).clear(cx);
            let keystroke = gpui::Keystroke::parse(key).expect("test keystroke");
            window.dispatch_event(
                gpui::KeyDownEvent {
                    keystroke: keystroke.clone(),
                    is_held: false,
                    prefer_character_input: false,
                }
                .to_platform_input(),
                cx,
            );
            window.dispatch_event(gpui::KeyUpEvent { keystroke }.to_platform_input(), cx);
        };
        for (selection, answer) in [(1, "Cancel"), (2, "Import Results")] {
            cx.update(|window, cx| {
                window.focus(&focus, cx);
                dispatch(window, cx, "tab");
                dispatch(window, cx, "enter");
            });
            cx.run_until_parked();
            assert_eq!(
                selections.load(Ordering::Relaxed),
                selection,
                "Import Profile requests project paths"
            );
            let (message, details) = cx.pending_prompt().expect("remote transfer consent");
            assert!(message.contains("Import normalized profile results"));
            assert!(details.contains("/fixture/profile.json"));
            assert!(details.contains("/fixture/profile.syms.json"));
            assert!(details.contains("Raw files stay on the host"));
            assert!(details.contains("No profile contents are uploaded to telemetry"));
            cx.simulate_prompt_answer(answer);
            cx.run_until_parked();
            if answer == "Cancel" {
                view.read_with(cx, |view, _| {
                    assert_eq!(view.sample_count_for_test(), None);
                    assert!(view.notice_for_test().starts_with("Import cancelled"));
                });
            }
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !view.read_with(cx, |view, _| {
            view.sample_count_for_test() == Some(3030)
                && view.notice_for_test().starts_with("3030 samples;")
        }) {
            assert!(
                std::time::Instant::now() < deadline,
                "remote viewer did not display the assembled host profile"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        assert!(!client_fs.is_file(Path::new("/fixture/profile.json")).await);
        let (started_sender, started) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, _| {
                store.gate_next_import_for_test(started_sender, release_receiver);
            });
        });
        cx.update(|window, cx| {
            window.focus(&focus, cx);
            dispatch(window, cx, "tab");
            dispatch(window, cx, "enter");
        });
        cx.run_until_parked();
        assert_eq!(selections.load(Ordering::Relaxed), 3);
        assert!(cx.has_pending_prompt(), "replacement transfer consent");
        cx.simulate_prompt_answer("Import Results");
        cx.run_until_parked();
        started
            .await
            .expect("host parsing starts before cancellation");
        cx.update(|window, cx| {
            window.focus(&focus, cx);
            for _ in 0..3 {
                dispatch(window, cx, "tab");
            }
            dispatch(window, cx, "enter");
        });
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.sample_count_for_test(), Some(3030));
            assert!(view.notice_for_test().starts_with("Import cancelled"));
        });
        if release.send(()).is_err() {
            view.read_with(cx, |view, _| {
                assert!(view.notice_for_test().starts_with("Import cancelled"));
            });
        }
        server_cx.run_until_parked();
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.sample_count_for_test(), Some(3030));
            assert!(view.notice_for_test().starts_with("Import cancelled"));
        });
        workspace.update_in(cx, |workspace, window, cx| {
            OpenPathPrompt::register(workspace, Some(window), cx);
        });
        cx.update(|window, cx| {
            window.focus(&focus, cx);
            dispatch(window, cx, "tab");
            dispatch(window, cx, "enter");
        });
        cx.run_until_parked();
        for path in ["/fixture/profile.json", "/fixture/profile.syms.json"] {
            let picker = workspace
                .read_with(cx, |workspace, cx| {
                    workspace.active_modal::<Picker<OpenPathDelegate>>(cx)
                })
                .expect("remote project path picker opens");
            picker.update_in(cx, |picker, window, cx| {
                picker.set_query(path, window, cx);
            });
            cx.run_until_parked();
            picker.update_in(cx, |picker, window, cx| {
                assert!(picker.delegate.match_count() > 0, "profile path suggestion");
                picker.delegate.confirm(false, window, cx);
            });
            cx.run_until_parked();
        }
        let (message, details) = cx
            .pending_prompt()
            .expect("project picker transfer consent");
        assert!(message.contains("Import normalized profile results"));
        assert!(details.contains("/fixture/profile.json"));
        assert!(details.contains("/fixture/profile.syms.json"));
        cx.simulate_prompt_answer("Import Results");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !view.read_with(cx, |view, _| {
            view.sample_count_for_test() == Some(3030)
                && view.notice_for_test().starts_with("3030 samples;")
        }) {
            assert!(
                std::time::Instant::now() < deadline,
                "project picker import did not display host profile"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        assert!(!client_fs.is_file(Path::new("/fixture/profile.json")).await);
        let (page_started_sender, page_started) = futures::channel::oneshot::channel();
        let (page_release, page_release_receiver) = futures::channel::oneshot::channel();
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, _| {
                assert_eq!(store.active_import_count_for_test(), 0);
                store.gate_next_page_for_test(page_started_sender, page_release_receiver);
            });
        });
        view.update(cx, |view, cx| {
            view.import_paths_with_consent_for_test(
                vec![
                    "/fixture/profile-replacement.json".into(),
                    "/fixture/profile.syms.json".into(),
                ],
                true,
                cx,
            )
            .expect("consented replacement starts a multi-page host import");
        });
        page_started
            .await
            .expect("first normalized host page precedes follow-up transfer");
        headless.read_with(server_cx, |headless, cx| {
            assert_eq!(
                headless
                    .profile_results_store
                    .read_with(cx, |store, _| store.active_import_count_for_test()),
                1,
                "the host retains the admitted multi-page import"
            );
        });
        cx.update(|window, cx| {
            window.focus(&focus, cx);
            for _ in 0..3 {
                dispatch(window, cx, "tab");
            }
            dispatch(window, cx, "enter");
        });
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.sample_count_for_test(), Some(3030));
            assert!(view.notice_for_test().starts_with("Import cancelled"));
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while headless.read_with(server_cx, |headless, cx| {
            headless
                .profile_results_store
                .read_with(cx, |store, _| store.active_import_count_for_test())
                != 0
        }) {
            assert!(
                std::time::Instant::now() < deadline,
                "viewer cancellation did not release the host's retained transfer"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        if page_release.send(()).is_err() {
            view.read_with(cx, |view, _| {
                assert!(view.notice_for_test().starts_with("Import cancelled"));
            });
        }
        server_cx.run_until_parked();
        cx.run_until_parked();
        view.read_with(cx, |view, _| {
            assert_eq!(view.sample_count_for_test(), Some(3030));
            assert!(view.notice_for_test().starts_with("Import cancelled"));
        });
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/workload.rs".into()]));
                });
            });
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        let source_path = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/fixture/workload.rs", cx)
                .expect("remote source path")
        });
        let host_private = headless.read_with(server_cx, |headless, cx| {
            headless
                .worktree_store
                .read(cx)
                .worktree_for_id(source_path.worktree_id, cx)
                .expect("host source worktree")
                .read(cx)
                .snapshot()
                .entry_for_path(util::rel_path::rel_path("workload.rs"))
                .expect("host source entry")
                .is_private
        });
        assert!(
            host_private,
            "the authoritative host must revoke its source"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view
            .read_with(cx, |view, _| view.sample_count_for_test())
            .is_some()
        {
            assert!(
                std::time::Instant::now() < deadline,
                "remote viewer retained a profile after host source privacy revocation"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        view.read_with(cx, |view, _| {
            assert!(view.notice_for_test().contains("no longer permitted"));
        });
        cx.deactivate_window();
        container.update_in(cx, |_, window, _| window.remove_window());
    }

    #[gpui::test]
    async fn profile_results_runtime_retained_pages_enforce_peer_and_permission_boundaries(
        cx: &mut TestAppContext,
        server_cx: &mut TestAppContext,
    ) {
        use project::profile_results_wire::{MAX_PROFILE_PAGE_BYTES, ProfilePageAssembler};
        use proto::Message as _;
        use std::sync::atomic::AtomicBool;

        let fs = FakeFs::new(server_cx.executor());
        fs.insert_tree("/fixture", json!({"workload.rs": "fn main() {}"}))
            .await;
        fs.insert_file("/fixture/profile.json", paged_capture())
            .await;
        fs.insert_file("/fixture/profile.syms.json", include_bytes!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.syms.json").to_vec()).await;
        fs.insert_tree("/outside", json!({
            "profile.json": include_str!("../../../.agents/specs/rust-workspace/rust-profiling/fixtures/macos-samply-0.13.1.json")
        })).await;
        fs.insert_symlink(
            "/fixture/external.json",
            PathBuf::from("/outside/profile.json"),
        )
        .await;
        let (project, headless) = init_test(&fs, cx, server_cx).await;
        cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        server_cx.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let (worktree, _) = project
            .update(cx, |project, cx| {
                project.find_or_create_worktree("/fixture", true, cx)
            })
            .await
            .expect("remote fixture");
        cx.run_until_parked();
        server_cx.run_until_parked();
        let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id().to_proto());
        let client = project.read_with(cx, |project, cx| {
            project
                .remote_client()
                .expect("remote client")
                .read(cx)
                .proto_client()
        });
        let transport = server_cx.new(|_| crate::NativeProjectTransport::default());
        headless.update(server_cx, |headless, cx| {
            headless.profile_results_store.update(cx, |store, cx| {
                store.observe_transport(&transport, true, cx)
            })
        });
        let owner = proto::PeerId { owner_id: 9, id: 1 };
        let other = proto::PeerId { owner_id: 9, id: 2 };
        let request = proto::ImportProfileResults {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: 1,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            profile_path: Some(proto::ProjectPath {
                worktree_id,
                path: "profile.json".into(),
            }),
            symbols_path: Some(proto::ProjectPath {
                worktree_id,
                path: "profile.syms.json".into(),
            }),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        };
        let page_request = |import_id, generation, page_index| proto::GetProfileResultPage {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id,
            generation,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            page_index,
        };
        let cancel_request = |import_id, generation| proto::CancelProfileImport {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id,
            generation,
        };
        headless.read_with(server_cx, |headless, cx| {
            let worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(project::WorktreeId::from_proto(worktree_id), cx)
                .expect("host worktree");
            assert!(
                worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(rel_path("external.json"))
                    .expect("external symlink entry")
                    .is_external
            );
        });
        let mut external = request.clone();
        external.import_id = 90;
        external.profile_path.as_mut().expect("profile input").path = "external.json".into();
        assert_eq!(
            client
                .request(FromPeer {
                    peer: owner,
                    payload: external
                })
                .await
                .expect("external input response")
                .status,
            proto::ProfileResultStatus::Restricted as i32
        );
        let first = client
            .request(FromPeer {
                peer: owner,
                payload: request.clone(),
            })
            .await
            .expect("first page");
        assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!first.complete);
        assert!(first.encoded_len() <= MAX_PROFILE_PAGE_BYTES);
        let generation = first.generation;
        let denied = client
            .request(FromPeer {
                peer: other,
                payload: page_request(1, generation, 1),
            })
            .await
            .expect("other peer page");
        assert_eq!(denied.status, proto::ProfileResultStatus::Stale as i32);
        assert!(denied.frames.is_empty() && denied.nodes.is_empty() && denied.summary.is_none());
        client
            .request(FromPeer {
                peer: other,
                payload: cancel_request(1, generation),
            })
            .await
            .expect("other peer cancellation");
        let duplicate = client
            .request(FromPeer {
                peer: owner,
                payload: page_request(1, generation, 0),
            })
            .await
            .expect("duplicate page");
        assert_eq!(duplicate.status, proto::ProfileResultStatus::Stale as i32);
        let mut assembler = ProfilePageAssembler::new(1, generation).expect("stream identity");
        let cancel = AtomicBool::new(false);
        assembler
            .push(first, &cancel, |_| true)
            .expect("first normalized page");
        let mut next_page = 1;
        loop {
            assert!(next_page < 16, "bounded fixture page count");
            let page = client
                .request(FromPeer {
                    peer: owner,
                    payload: page_request(1, generation, next_page),
                })
                .await
                .expect("owner page");
            assert_eq!(page.status, proto::ProfileResultStatus::Ready as i32);
            assert!(page.encoded_len() <= MAX_PROFILE_PAGE_BYTES);
            let complete = page.complete;
            assembler
                .push(page, &cancel, |_| true)
                .expect("ordered normalized page");
            next_page += 1;
            if complete {
                break;
            }
        }
        assert!(next_page >= 3);
        let result = assembler
            .finish(&cancel, |_| true)
            .expect("complete scale model");
        assert_eq!(result.total_samples, 3_030);
        assert!(result.frames.iter().any(|frame| frame.source.is_some()));
        let stale = client
            .request(FromPeer {
                peer: owner,
                payload: page_request(1, generation, next_page),
            })
            .await
            .expect("completed stream");
        assert_eq!(stale.status, proto::ProfileResultStatus::Stale as i32);

        let mut second = request.clone();
        second.import_id = 2;
        let replaced = client
            .request(FromPeer {
                peer: owner,
                payload: second.clone(),
            })
            .await
            .expect("replaceable stream");
        assert!(!replaced.complete);
        let first = client
            .request(FromPeer {
                peer: owner,
                payload: second,
            })
            .await
            .expect("cancellable stream");
        assert!(!first.complete);
        assert!(first.generation > replaced.generation);
        assert_eq!(
            client
                .request(FromPeer {
                    peer: owner,
                    payload: page_request(2, replaced.generation, 1)
                })
                .await
                .expect("replaced stream")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );
        client
            .request(FromPeer {
                peer: owner,
                payload: cancel_request(2, replaced.generation),
            })
            .await
            .expect("replaced-generation cancellation");
        client
            .request(FromPeer {
                peer: owner,
                payload: cancel_request(2, first.generation + 1),
            })
            .await
            .expect("stale cancellation");
        let page = client
            .request(FromPeer {
                peer: owner,
                payload: page_request(2, first.generation, 1),
            })
            .await
            .expect("stream after stale cancellation");
        assert_eq!(page.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!page.complete);
        client
            .request(FromPeer {
                peer: owner,
                payload: cancel_request(2, first.generation),
            })
            .await
            .expect("owner cancellation");
        assert_eq!(
            client
                .request(FromPeer {
                    peer: owner,
                    payload: page_request(2, first.generation, 2)
                })
                .await
                .expect("cancelled stream")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );

        let mut third = request.clone();
        third.import_id = 3;
        let first = client
            .request(FromPeer {
                peer: owner,
                payload: third,
            })
            .await
            .expect("privacy stream");
        assert!(!first.complete);
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/workload.rs".into()]))
                });
            })
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        headless.read_with(server_cx, |headless, cx| {
            let worktree = headless
                .worktree_store
                .read(cx)
                .worktree_for_id(project::WorktreeId::from_proto(worktree_id), cx)
                .expect("host worktree");
            assert!(
                worktree
                    .read(cx)
                    .snapshot()
                    .entry_for_path(util::rel_path::rel_path("workload.rs"))
                    .expect("source entry")
                    .is_private
            );
        });
        let denied = client
            .request(FromPeer {
                peer: owner,
                payload: page_request(3, first.generation, 1),
            })
            .await
            .expect("revoked source stream");
        assert!(
            denied.status == proto::ProfileResultStatus::Restricted as i32
                || denied.status == proto::ProfileResultStatus::Stale as i32
        );
        assert!(denied.frames.is_empty() && denied.nodes.is_empty() && denied.summary.is_none());
        server_cx.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files = None
                });
            })
        });
        server_cx.run_until_parked();
        cx.run_until_parked();
        let mut fourth = request.clone();
        fourth.import_id = 4;
        let first = client
            .request(FromPeer {
                peer: owner,
                payload: fourth,
            })
            .await
            .expect("restored source stream");
        assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!first.complete);
        transport.update(server_cx, |_, cx| cx.emit(false));
        transport.update(server_cx, |_, cx| cx.emit(true));
        assert_eq!(
            client
                .request(FromPeer {
                    peer: owner,
                    payload: page_request(4, first.generation, 1)
                })
                .await
                .expect("previous connection stream")
                .status,
            proto::ProfileResultStatus::Stale as i32
        );

        let mut retained = Vec::new();
        for import_id in 10..14 {
            let mut import = request.clone();
            import.import_id = import_id;
            let page = client
                .request(FromPeer {
                    peer: owner,
                    payload: import,
                })
                .await
                .expect("quota slot");
            assert_eq!(page.status, proto::ProfileResultStatus::Ready as i32);
            assert!(!page.complete);
            retained.push((import_id, page.generation));
        }
        let mut overflow = request;
        overflow.import_id = 14;
        assert_eq!(
            client
                .request(FromPeer {
                    peer: owner,
                    payload: overflow.clone()
                })
                .await
                .expect("quota response")
                .status,
            proto::ProfileResultStatus::Error as i32
        );
        let (import_id, generation) = retained[0];
        client
            .request(FromPeer {
                peer: owner,
                payload: cancel_request(import_id, generation),
            })
            .await
            .expect("release quota slot");
        assert_eq!(
            client
                .request(FromPeer {
                    peer: owner,
                    payload: overflow
                })
                .await
                .expect("reused quota slot")
                .status,
            proto::ProfileResultStatus::Ready as i32
        );
        transport.update(server_cx, |_, cx| cx.emit(false));
    }
}

#[cfg(all(feature = "doctor-process-test", target_os = "macos"))]
struct DoctorNativeServerFixture {
    directory: tempfile::TempDir,
    wrapper: PathBuf,
}

#[cfg(all(feature = "doctor-process-test", target_os = "macos"))]
impl DoctorNativeServerFixture {
    fn new() -> Self {
        Self::with_stop_delay(false)
    }

    fn with_stop_delay(delay: bool) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let binary = std::env::var_os("MADE_DOCTOR_SERVER_BINARY")
            .map(PathBuf::from)
            .expect(
                "Set MADE_DOCTOR_SERVER_BINARY to the current test-support remote_server binary",
            );
        assert!(binary.is_file());
        let directory = tempfile::tempdir_in("/private/tmp").expect("Private native host state");
        std::fs::write(directory.path().join("unrelated"), "preserve").expect("Sentinel");
        let wrapper = directory.path().join("server-wrapper");
        let quote = |value: &Path| format!("'{}'", value.to_string_lossy().replace('\'', "'\\''"));
        let delay_command = if delay {
            format!(
                "if [ \"$1\" = doctor-server-stopped ]; then printf '%s' \"$$\" > {}; exec /bin/sleep 60; fi\n",
                quote(&directory.path().join("stop-observation.pid"))
            )
        } else {
            String::new()
        };
        std::fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\n{delay_command}exec {} --test-data-dir {} \"$@\"\n",
                quote(&binary),
                quote(directory.path())
            ),
        )
        .expect("Exact native test launcher");
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))
            .expect("Private launcher");
        Self { directory, wrapper }
    }

    async fn connect(&self, cx: &mut TestAppContext) -> Entity<RemoteClient> {
        use remote::{ConnectionIdentifier, MockDelegate, ProcessRemoteConnection};
        cx.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
        init_logger();
        cx.executor().allow_parking();
        let connection = Arc::new(ProcessRemoteConnection::new(
            self.wrapper.clone(),
            self.directory.path().join("server_state"),
        ));
        let (_cancel, receiver) = futures::channel::oneshot::channel();
        cx.update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::setup(),
                connection,
                receiver,
                Arc::new(MockDelegate),
                cx,
            )
        })
        .await
        .expect("Current native server starts")
        .expect("Native client")
    }

    async fn observe_exit(process_id: u32) {
        smol::unblock(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                if unsafe { libc::kill(process_id as i32, 0) } != 0 {
                    assert_eq!(
                        std::io::Error::last_os_error().raw_os_error(),
                        Some(libc::ESRCH)
                    );
                    return;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "Native server did not stop; no force-kill fallback"
                );
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
        .await;
    }

    async fn shutdown(client: &Entity<RemoteClient>, cx: &mut TestAppContext) {
        let status = client
            .read_with(cx, |client, cx| client.doctor_restart_status(cx))
            .await
            .expect("Current native instance");
        let (cancel, cancelled) = futures::channel::oneshot::channel();
        let result = client
            .update(cx, |client, cx| {
                use futures::FutureExt as _;
                client.doctor_restart(
                    status.clone(),
                    async move {
                        match cancelled.await {
                            Ok(()) | Err(_) => {}
                        }
                    }
                    .boxed(),
                    move |_, after_shutdown| {
                        if after_shutdown {
                            anyhow::bail!(
                                "Fixture leaves its private host stopped after graceful shutdown"
                            );
                        }
                        Ok(())
                    },
                    cx,
                )
            })
            .await
            .expect("Graceful owner cleanup");
        drop(cancel);
        assert!(result.shutdown_requested && result.replacement.is_none());
        Self::observe_exit(status.process_id).await;
    }
}

#[cfg(all(feature = "doctor-process-test", target_os = "macos"))]
#[gpui::test(iterations = 1)]
async fn doctor_restart_real_process_preserves_pre_cancel_and_observes_replacement(
    cx: &mut TestAppContext,
) {
    use futures::FutureExt as _;
    let fixture = DoctorNativeServerFixture::new();
    let client = fixture.connect(cx).await;
    let original = client
        .read_with(cx, |client, cx| client.doctor_restart_status(cx))
        .await
        .expect("Native owned instance");
    assert!(original.exclusive && original.idle);
    let cancelled = client
        .update(cx, |client, cx| {
            client.doctor_restart(
                original.clone(),
                futures::future::ready(()).boxed(),
                |_, _| Ok(()),
                cx,
            )
        })
        .await
        .expect("Pre-dispatch cancellation");
    assert!(!cancelled.shutdown_requested && cancelled.replacement.is_none());
    let mut stale = original.clone();
    stale.instance = uuid::Uuid::new_v4().to_string();
    assert!(
        client
            .update(cx, |client, cx| client.doctor_restart(
                stale,
                futures::future::pending().boxed(),
                |_, _| Ok(()),
                cx
            ))
            .await
            .is_err()
    );
    let unchanged = client
        .read_with(cx, |client, cx| client.doctor_restart_status(cx))
        .await
        .expect("Still original server");
    assert_eq!(unchanged.instance, original.instance);
    let reached = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let result = client
        .update(cx, |client, cx| {
            let reached = reached.clone();
            client.doctor_restart(
                original.clone(),
                futures::future::pending().boxed(),
                move |_, after_shutdown| {
                    reached.lock().push(after_shutdown);
                    Ok(())
                },
                cx,
            )
        })
        .await
        .expect("Current owner restart");
    assert!(result.shutdown_requested);
    assert_eq!(*reached.lock(), [false, true, true]);
    let replacement = result.replacement.expect("Observed ready replacement");
    assert_ne!(result.replacement_instance, Some(original.instance));
    DoctorNativeServerFixture::observe_exit(original.process_id).await;
    assert_eq!(
        std::fs::read_to_string(fixture.directory.path().join("unrelated")).expect("Sentinel"),
        "preserve"
    );
    assert!(result.detail.contains("not verified fixed"));
    DoctorNativeServerFixture::shutdown(&replacement, cx).await;
}

#[cfg(all(feature = "doctor-process-test", target_os = "macos"))]
#[gpui::test(iterations = 1)]
async fn doctor_restart_real_process_revocation_after_shutdown_never_starts_replacement(
    cx: &mut TestAppContext,
) {
    use futures::FutureExt as _;
    let fixture = DoctorNativeServerFixture::new();
    let client = fixture.connect(cx).await;
    let original = client
        .read_with(cx, |client, cx| client.doctor_restart_status(cx))
        .await
        .expect("Native owned instance");
    let result = client
        .update(cx, |client, cx| {
            client.doctor_restart(
                original.clone(),
                futures::future::pending().boxed(),
                |_, after_shutdown| {
                    anyhow::ensure!(!after_shutdown, "Trust revoked during shutdown");
                    Ok(())
                },
                cx,
            )
        })
        .await
        .expect("Reached partial restart");
    assert!(result.shutdown_requested && result.replacement.is_none());
    assert!(result.detail.contains("authority changed"));
    DoctorNativeServerFixture::observe_exit(original.process_id).await;
    assert_eq!(
        std::fs::read_to_string(fixture.directory.path().join("unrelated")).expect("Sentinel"),
        "preserve"
    );
}

#[gpui::test]
async fn doctor_remote_restart_headless_rejects_dirty_shared_and_stale_control(
    cx: &mut TestAppContext,
    server_cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(server_cx.executor());
    fs.insert_tree(path!("/doctor-host"), json!({ "app.rs": "fn main() {}" }))
        .await;
    let (project, headless, activity) = init_test_with_restart_activity(&fs, cx, server_cx).await;
    let instance = uuid::Uuid::new_v4().to_string();
    headless.update(server_cx, |headless, _| {
        headless.configure_doctor_restart(instance.clone(), true, activity.clone());
    });
    cx.run_until_parked();
    let (worktree, _) = project
        .update(cx, |project, cx| {
            project.find_or_create_worktree(path!("/doctor-host"), true, cx)
        })
        .await
        .expect("Remote root");
    let path = ProjectPath {
        worktree_id: worktree.read_with(cx, |worktree, _| worktree.id()),
        path: rel_path("app.rs").into(),
    };
    let buffer = project
        .update(cx, |project, cx| project.open_buffer(path, cx))
        .await
        .expect("Remote buffer");
    cx.run_until_parked();
    let client = project
        .read_with(cx, |project, _| project.remote_client())
        .expect("Current transport");
    let protocol = client.read_with(cx, |client, _| client.proto_client());
    let status = protocol
        .request(proto::DoctorRemoteServer {
            expected_instance: instance.clone(),
            shutdown: false,
        })
        .await
        .expect("Typed current owner observation");
    assert!(status.exclusive && status.idle && !status.shutdown_accepted);
    assert!(
        protocol
            .request(proto::DoctorRemoteServer {
                expected_instance: uuid::Uuid::new_v4().to_string(),
                shutdown: true
            })
            .await
            .is_err()
    );
    headless.update(server_cx, |headless, _| {
        headless.configure_doctor_restart(instance.clone(), false, activity.clone());
    });
    let shared = protocol
        .request(proto::DoctorRemoteServer {
            expected_instance: instance.clone(),
            shutdown: false,
        })
        .await
        .expect("Shared status");
    assert!(!shared.exclusive);
    assert!(
        protocol
            .request(proto::DoctorRemoteServer {
                expected_instance: instance.clone(),
                shutdown: true
            })
            .await
            .is_err()
    );
    headless.update(server_cx, |headless, _| {
        headless.configure_doctor_restart(instance.clone(), true, activity);
    });
    buffer.update(cx, |buffer, cx| {
        buffer.edit([(0..0, "// changed\n")], None, cx)
    });
    cx.run_until_parked();
    let busy = protocol
        .request(proto::DoctorRemoteServer {
            expected_instance: instance.clone(),
            shutdown: false,
        })
        .await
        .expect("Dirty status");
    assert!(!busy.idle && busy.reason.contains("buffers"));
    assert!(
        protocol
            .request(proto::DoctorRemoteServer {
                expected_instance: instance,
                shutdown: true
            })
            .await
            .is_err()
    );
    assert!(
        project
            .read_with(cx, |project, cx| project
                .doctor_remote_restart_idle(false, cx))
            .is_err()
    );
    assert!(buffer.read_with(cx, |buffer, _| buffer.is_dirty()));
    assert!(
        fs.load(Path::new(path!("/doctor-host/app.rs")))
            .await
            .expect("Preserved original")
            .starts_with("fn main()")
    );
}

#[cfg(all(feature = "doctor-process-test", target_os = "macos"))]
#[gpui::test(iterations = 1)]
async fn doctor_restart_real_process_stop_timeout_cleans_only_owned_helper(
    cx: &mut TestAppContext,
) {
    use futures::FutureExt as _;
    let fixture = DoctorNativeServerFixture::with_stop_delay(true);
    let client = fixture.connect(cx).await;
    let original = client
        .read_with(cx, |client, cx| client.doctor_restart_status(cx))
        .await
        .expect("Owned native instance");
    let operation = client.update(cx, |client, cx| {
        client.doctor_restart(
            original.clone(),
            futures::future::pending().boxed(),
            |_, _| Ok(()),
            cx,
        )
    });
    let marker = fixture.directory.path().join("stop-observation.pid");
    let helper_pid = smol::unblock(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if let Ok(pid) = std::fs::read_to_string(&marker)
                && let Ok(pid) = pid.parse::<u32>()
            {
                return pid;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Actual bounded helper must start"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    })
    .await;
    assert_eq!(
        client.read_with(cx, |client, _| client.connection_state()),
        remote::ConnectionState::Disconnected
    );
    cx.executor()
        .dispatcher()
        .as_test()
        .expect("Test dispatcher")
        .scheduler()
        .clock()
        .advance(std::time::Duration::from_secs(3));
    let result = operation.await.expect("Bounded stop observation");
    assert!(result.shutdown_requested && result.replacement.is_none());
    assert!(result.detail.contains("timed out"));
    DoctorNativeServerFixture::observe_exit(original.process_id).await;
    DoctorNativeServerFixture::observe_exit(helper_pid).await;
    assert_eq!(
        std::fs::read_to_string(fixture.directory.path().join("unrelated")).expect("Sentinel"),
        "preserve"
    );
}

#[cfg(all(feature = "doctor-process-test", target_os = "macos"))]
#[gpui::test(iterations = 1)]
async fn doctor_restart_real_process_cancel_after_stop_never_starts_replacement(
    cx: &mut TestAppContext,
) {
    use futures::FutureExt as _;
    let fixture = DoctorNativeServerFixture::new();
    let client = fixture.connect(cx).await;
    let original = client
        .read_with(cx, |client, cx| client.doctor_restart_status(cx))
        .await
        .expect("Owned native instance");
    let (cancel, cancelled) = async_channel::bounded::<()>(1);
    let result = client
        .update(cx, |client, cx| {
            client.doctor_restart(
                original.clone(),
                async move {
                    match cancelled.recv().await {
                        Ok(()) | Err(_) => {}
                    }
                }
                .boxed(),
                move |_, after_shutdown| {
                    if after_shutdown {
                        cancel.close();
                    }
                    Ok(())
                },
                cx,
            )
        })
        .await
        .expect("Reached cancellation");
    assert!(result.shutdown_requested && result.replacement.is_none());
    assert!(result.detail.contains("cancelled after authority"));
    assert_eq!(
        client.read_with(cx, |client, _| client.connection_state()),
        remote::ConnectionState::Disconnected
    );
    DoctorNativeServerFixture::observe_exit(original.process_id).await;
}
