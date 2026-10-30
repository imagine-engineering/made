use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use fs::FakeFs;
use futures::StreamExt as _;
use gpui::TestAppContext;
use language::{
    FakeLspAdapter, Language, LanguageConfig, LanguageMatcher, rust_lang, tree_sitter_typescript,
};
use lsp::{CallHierarchyServerCapability, LanguageServerId, Uri};
use project::Project;
use serde_json::json;
use util::path;

use super::init_test;

struct CancellationGuard(Arc<AtomicBool>);

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn typescript_lang() -> Arc<Language> {
    Arc::new(Language::new(
        LanguageConfig {
            name: "TypeScript".into(),
            matcher: (LanguageMatcher {
                path_suffixes: vec!["ts".to_string()],
                ..Default::default()
            })
            .into(),
            ..Default::default()
        },
        Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
    ))
}

fn item(path: &Path, name: &str, row: u32, data: serde_json::Value) -> lsp::CallHierarchyItem {
    lsp::CallHierarchyItem {
        name: name.to_string(),
        kind: lsp::SymbolKind::FUNCTION,
        tags: None,
        detail: Some(format!("fn {name}()")),
        uri: Uri::from_file_path(path).expect("fixture path should convert to a URI"),
        range: lsp::Range::new(
            lsp::Position::new(row, 0),
            lsp::Position::new(row, name.len() as u32 + 5),
        ),
        selection_range: lsp::Range::new(
            lsp::Position::new(row, 3),
            lsp::Position::new(row, name.len() as u32 + 3),
        ),
        data: Some(data),
    }
}

#[gpui::test]
async fn call_hierarchy_lsp_routes_prepare_incoming_and_outgoing_with_bounds(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        path!("/dir"),
        json!({
            "root.rs": "fn root() {}\n",
            "caller.rs": "fn caller() { root(); }\n",
            "callee.rs": "fn callee() {}\n",
        }),
    )
    .await;
    let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());
    let mut fake_servers = language_registry.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: lsp::ServerCapabilities {
                call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
                ..lsp::ServerCapabilities::default()
            },
            ..FakeLspAdapter::default()
        },
    );
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/dir/root.rs"), cx)
        })
        .await
        .expect("fixture buffer should open");
    let fake_server = fake_servers
        .next()
        .await
        .expect("fake language server should start");
    cx.executor().run_until_parked();

    fake_server.set_request_handler::<lsp::request::CallHierarchyPrepare, _, _>(
        |params, _| async move {
            assert_eq!(
                params
                    .text_document_position_params
                    .text_document
                    .uri
                    .to_file_path(),
                Ok(path!("/dir/root.rs").into())
            );
            assert_eq!(
                params.text_document_position_params.position,
                lsp::Position::new(0, 4)
            );
            let valid = item(
                Path::new(path!("/dir/root.rs")),
                "root",
                0,
                json!({"token": 1}),
            );
            let mut malformed = valid.clone();
            malformed.name = "malformed".to_string();
            malformed.selection_range =
                lsp::Range::new(lsp::Position::new(5, 0), lsp::Position::new(5, 1));
            Ok(Some(vec![valid, malformed]))
        },
    );
    let prepared = project
        .update(cx, |project, cx| {
            project.prepare_call_hierarchy(&buffer, 4, cx)
        })
        .await
        .expect("prepare request should succeed")
        .expect("capable server should return a supported result");
    assert_eq!(prepared.items.len(), 1);
    assert_eq!(prepared.malformed_count, 1);
    assert!(!prepared.truncated);
    assert_eq!(prepared.items[0].server_id, LanguageServerId(0));

    fake_server.set_request_handler::<lsp::request::CallHierarchyIncomingCalls, _, _>(
        |params, _| async move {
            assert_eq!(params.item.name, "root");
            assert_eq!(params.item.data, Some(json!({"token": 1})));
            Ok(Some(vec![lsp::CallHierarchyIncomingCall {
                from: item(
                    Path::new(path!("/dir/caller.rs")),
                    "caller",
                    0,
                    json!({"token": 2}),
                ),
                from_ranges: vec![lsp::Range::new(
                    lsp::Position::new(0, 14),
                    lsp::Position::new(0, 18),
                )],
            }]))
        },
    );
    let incoming = project
        .update(cx, |project, cx| {
            project.incoming_calls(prepared.items[0].clone(), cx)
        })
        .await
        .expect("incoming request should succeed");
    assert_eq!(incoming.calls.len(), 1);
    assert_eq!(incoming.calls[0].item.name, "caller");
    assert_eq!(incoming.calls[0].from_ranges.len(), 1);

    fake_server.set_request_handler::<lsp::request::CallHierarchyOutgoingCalls, _, _>(
        |params, _| async move {
            assert_eq!(params.item.name, "root");
            Ok(Some(vec![lsp::CallHierarchyOutgoingCall {
                to: item(
                    Path::new(path!("/dir/callee.rs")),
                    "callee",
                    0,
                    json!({"token": 3}),
                ),
                from_ranges: vec![lsp::Range::new(
                    lsp::Position::new(0, 3),
                    lsp::Position::new(0, 7),
                )],
            }]))
        },
    );
    let outgoing = project
        .update(cx, |project, cx| {
            project.outgoing_calls(prepared.items[0].clone(), cx)
        })
        .await
        .expect("outgoing request should succeed");
    assert_eq!(outgoing.calls.len(), 1);
    assert_eq!(outgoing.calls[0].item.name, "callee");
}

#[gpui::test]
async fn call_hierarchy_lsp_distinguishes_unsupported_servers(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/dir"), json!({ "root.rs": "fn root() {}\n" }))
        .await;
    let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());
    let mut fake_servers = language_registry.register_fake_lsp("Rust", FakeLspAdapter::default());
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/dir/root.rs"), cx)
        })
        .await
        .expect("fixture buffer should open");
    fake_servers
        .next()
        .await
        .expect("fake language server should start");
    cx.executor().run_until_parked();

    let prepared = project
        .update(cx, |project, cx| {
            project.prepare_call_hierarchy(&buffer, 4, cx)
        })
        .await
        .expect("unsupported request should not fail");
    assert!(prepared.is_none());
}

#[gpui::test]
async fn call_hierarchy_lsp_is_language_agnostic_and_bounds_large_responses(
    cx: &mut TestAppContext,
) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/dir"), json!({ "root.ts": "function root() {}\n" }))
        .await;
    let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(typescript_lang());
    let mut fake_servers = language_registry.register_fake_lsp(
        "TypeScript",
        FakeLspAdapter {
            capabilities: lsp::ServerCapabilities {
                call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
                ..lsp::ServerCapabilities::default()
            },
            ..FakeLspAdapter::default()
        },
    );
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/dir/root.ts"), cx)
        })
        .await
        .expect("TypeScript fixture buffer should open");
    let fake_server = fake_servers
        .next()
        .await
        .expect("fake TypeScript server should start");
    cx.executor().run_until_parked();

    fake_server.set_request_handler::<lsp::request::CallHierarchyPrepare, _, _>(
        |_, _| async move {
            Ok(Some(
                (0..257)
                    .map(|index| {
                        item(
                            Path::new(path!("/dir/root.ts")),
                            &format!("root_{index}"),
                            0,
                            json!({"index": index}),
                        )
                    })
                    .collect(),
            ))
        },
    );
    let prepared = project
        .update(cx, |project, cx| {
            project.prepare_call_hierarchy(&buffer, 9, cx)
        })
        .await
        .expect("TypeScript prepare request should succeed")
        .expect("TypeScript server advertises call hierarchy");
    assert_eq!(prepared.items.len(), 256);
    assert!(prepared.truncated);
    assert_eq!(prepared.items[255].name.as_ref(), "root_255");
}

#[gpui::test]
async fn call_hierarchy_lsp_cancels_dropped_requests_with_gpui_timers(cx: &mut TestAppContext) {
    init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(path!("/dir"), json!({ "root.rs": "fn root() {}\n" }))
        .await;
    let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
    let language_registry = project.read_with(cx, |project, _| project.languages().clone());
    language_registry.add(rust_lang());
    let mut fake_servers = language_registry.register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            capabilities: lsp::ServerCapabilities {
                call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
                ..lsp::ServerCapabilities::default()
            },
            ..FakeLspAdapter::default()
        },
    );
    let (buffer, _handle) = project
        .update(cx, |project, cx| {
            project.open_local_buffer_with_lsp(path!("/dir/root.rs"), cx)
        })
        .await
        .expect("fixture buffer should open");
    let fake_server = fake_servers
        .next()
        .await
        .expect("fake language server should start");
    cx.executor().run_until_parked();

    let request_started = Arc::new(AtomicBool::new(false));
    let request_dropped = Arc::new(AtomicBool::new(false));
    let executor = cx.executor();
    fake_server.set_request_handler::<lsp::request::CallHierarchyPrepare, _, _>({
        let request_started = request_started.clone();
        let request_dropped = request_dropped.clone();
        move |_, _| {
            request_started.store(true, Ordering::SeqCst);
            let guard = CancellationGuard(request_dropped.clone());
            let executor = executor.clone();
            async move {
                executor.timer(Duration::from_secs(30)).await;
                drop(guard);
                Ok(None)
            }
        }
    });
    let request = project.update(cx, |project, cx| {
        project.prepare_call_hierarchy(&buffer, 4, cx)
    });
    cx.executor().run_until_parked();
    assert!(request_started.load(Ordering::SeqCst));
    drop(request);
    cx.executor().run_until_parked();
    assert!(request_dropped.load(Ordering::SeqCst));
}
