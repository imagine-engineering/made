use crate::TestServer;
use call::ActiveCall;
use collab::db::ChannelId;
use editor::Editor;
#[cfg(feature = "rust-tools-test")]
use futures::StreamExt;
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
use gpui::Focusable;
use gpui::{BackgroundExecutor, TestAppContext};
#[cfg(feature = "rust-tools-test")]
use language::{FakeLspAdapter, rust_lang};
#[cfg(feature = "rust-tools-test")]
use project::cargo_workspace_store::{
    CargoMetadataRequest, CargoMetadataRunner, CargoWorkspaceStore,
};
#[cfg(feature = "rust-tools-test")]
use project::lsp_store::lsp_ext_command::{CargoRunnableArgs, Runnable, RunnableArgs, Runnables};
#[cfg(feature = "rust-tools-test")]
use project::rust_test_provider::{
    RustTestDiscoveryCapture, RustTestDiscoveryRequest, RustTestDiscoveryRunner,
    RustTestListingCapture, RustTestListingMode, RustTestProviderStore, RustTestTargetKind,
};
#[cfg(feature = "rust-tools-test")]
use project::source_coverage::{
    SourceCoverageFile, SourceCoveragePoint, SourceCoverageProviderId, SourceCoverageRange,
    SourceCoverageSnapshot, SourceCoverageStatus,
};
#[cfg(feature = "rust-tools-test")]
use project::structured_execution::{
    DiscoveryGeneration, StructuredExecutionEvent, StructuredNode, StructuredNodeId,
    StructuredNodeKind, StructuredNodeState, StructuredProviderId, StructuredProviderSnapshot,
    StructuredProviderStatus, StructuredRun, StructuredRunId,
};
use rpc::ProtoClient as _;
use rpc::proto;
#[cfg(feature = "rust-tools-test")]
use std::sync::{Arc, Mutex};
use util::rel_path::rel_path;

#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn wait_for_scanned_relay_artifacts(
    project: &gpui::Entity<project::Project>,
    root: &std::path::Path,
    artifacts: &[&str],
    executor: &BackgroundExecutor,
    cx: &TestAppContext,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    for artifact in artifacts {
        while !project.read_with(cx, |project, cx| {
            let Some(path) = project.find_project_path(root.join(artifact), cx) else {
                return false;
            };
            project
                .visible_worktrees(cx)
                .find(|worktree| worktree.read(cx).id() == path.worktree_id)
                .is_some_and(|worktree| {
                    worktree
                        .read(cx)
                        .snapshot()
                        .entry_for_path(path.path.as_ref())
                        .is_some_and(|entry| {
                            entry.is_file()
                                && !entry.is_private
                                && !entry.is_fifo
                                && !entry.is_external
                        })
                })
        }) {
            if std::time::Instant::now() >= deadline {
                project.read_with(cx, |project, cx| {
                    let worktrees = project
                        .worktrees(cx)
                        .map(|worktree| {
                            worktree.read_with(cx, |worktree, _| {
                                (
                                    worktree.id(),
                                    worktree.is_visible(),
                                    worktree.scan_id(),
                                    worktree.completed_scan_id(),
                                    worktree.snapshot().entry_for_path(rel_path(artifact)).map(
                                        |entry| {
                                            (
                                                entry.is_file(),
                                                entry.is_private,
                                                entry.is_fifo,
                                                entry.is_external,
                                            )
                                        },
                                    ),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "relay scan timeout: collab={}; disconnected={}; artifact={artifact}; worktrees={worktrees:?}",
                        project.is_via_collab(),
                        project.is_disconnected(cx)
                    );
                });
            }
            assert!(
                std::time::Instant::now() < deadline,
                "remote project did not scan visible regular artifact {artifact}"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
    }
}

#[gpui::test]
#[cfg(not(feature = "rust-tools-test"))]
async fn rust_tools_relay_feature_disabled_host_has_no_rust_handlers(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    use std::any::TypeId;

    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let channel_id = server
        .make_public_channel("rust-tools-disabled-host", &client_a, cx_a)
        .await;
    client_a
        .fs()
        .insert_tree(
            "/relay",
            serde_json::json!({
                "Cargo.toml": "[package]\nname = \"relay\"\nversion = \"0.1.0\"\n",
                "profile.json": "{}",
                "profile.syms.json": "{}",
            }),
        )
        .await;
    let project_a = client_a
        .build_local_project_with_trust("/relay", cx_a)
        .await
        .0;
    let active_call_a = cx_a.read(ActiveCall::global);
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .expect("host joins channel");
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("host shares project");
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .expect("member joins channel");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("promote member");
    executor.run_until_parked();

    let (workspace_b, cx_b) = client_b.active_workspace(cx_b);
    let project_b = workspace_b.read_with(cx_b, |workspace, _| workspace.project().clone());
    assert_eq!(
        project_b.read_with(cx_b, |project, _| project.remote_id()),
        Some(project_id)
    );

    {
        let host_handlers = client_a.client().message_handler_set();
        let host_handlers = host_handlers.lock();
        for (request_type, request_name) in [
            (
                TypeId::of::<proto::GetCargoWorkspace>(),
                "GetCargoWorkspace",
            ),
            (
                TypeId::of::<proto::CancelCargoWorkspace>(),
                "CancelCargoWorkspace",
            ),
            (
                TypeId::of::<proto::GetRustTestDiscovery>(),
                "GetRustTestDiscovery",
            ),
            (
                TypeId::of::<proto::CancelRustTestDiscovery>(),
                "CancelRustTestDiscovery",
            ),
            (
                TypeId::of::<proto::ResolveRustTestAction>(),
                "ResolveRustTestAction",
            ),
            (
                TypeId::of::<proto::UpdateRustTestRun>(),
                "UpdateRustTestRun",
            ),
            (
                TypeId::of::<proto::GetStructuredExecutionSnapshot>(),
                "GetStructuredExecutionSnapshot",
            ),
            (
                TypeId::of::<proto::GetStructuredExecutionEvents>(),
                "GetStructuredExecutionEvents",
            ),
            (
                TypeId::of::<proto::InterpretRustCoverageArtifact>(),
                "InterpretRustCoverageArtifact",
            ),
        ] {
            assert!(
                !host_handlers.message_handlers.contains_key(&request_type),
                "the feature-disabled host must not register {request_name}"
            );
        }
        for request_type in [
            TypeId::of::<proto::PrepareProfileTaskBinding>(),
            TypeId::of::<proto::GetProfileTaskBinding>(),
            TypeId::of::<proto::CancelProfileTaskBinding>(),
            TypeId::of::<proto::ImportProfileResults>(),
            TypeId::of::<proto::GetProfileResultPage>(),
            TypeId::of::<proto::CancelProfileImport>(),
        ] {
            assert!(
                host_handlers.message_handlers.contains_key(&request_type),
                "generic profile protocol remains available without the native Task helper"
            );
        }
        assert!(
            host_handlers
                .message_handlers
                .contains_key(&TypeId::of::<proto::GetSourceCoverage>()),
            "generic coverage remains available without Rust tools"
        );
    }

    let profile_paths = project_b.read_with(cx_b, |project, cx| {
        [
            project
                .find_project_path("/relay/profile.json", cx)
                .expect("scanned profile artifact"),
            project
                .find_project_path("/relay/profile.syms.json", cx)
                .expect("scanned symbols artifact"),
        ]
    });
    cx_b.update(|_, cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let profile_result = project_b
        .read_with(cx_b, |project, cx| {
            project.prepare_remote_profile_task("disabled-profile".into(), profile_paths, cx)
        })
        .expect("profile Task request reaches the host")
        .await;
    let Err(profile_error) = profile_result else {
        panic!("feature-disabled host has no native Task helper");
    };
    assert_eq!(
        profile_error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );

    let cargo = client_b
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 1,
            worktree_ids: Vec::new(),
        })
        .await
        .expect_err("disabled Cargo host");
    let rust_tests = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 2,
            protocol_version: 1,
            worktree_ids: Vec::new(),
        })
        .await
        .expect_err("disabled Rust test host");
    let structured = client_b
        .client()
        .request(proto::GetStructuredExecutionSnapshot {
            project_id,
            provider_id: "rust-tests".into(),
            discovery_generation: 1,
            page_start: 0,
            page_size: 1,
            worktree_ids: Vec::new(),
        })
        .await
        .expect_err("disabled structured-result host");
    let coverage = client_b
        .client()
        .request(proto::InterpretRustCoverageArtifact {
            project_id,
            artifact_path: Some(proto::ProjectPath {
                worktree_id: 1,
                path: "coverage.json".into(),
            }),
            max_bytes: 1,
        })
        .await
        .expect_err("disabled Rust coverage host");
    for (error, request_name) in [
        (cargo, "GetCargoWorkspace"),
        (rust_tests, "GetRustTestDiscovery"),
        (structured, "GetStructuredExecutionSnapshot"),
        (coverage, "InterpretRustCoverageArtifact"),
    ] {
        let rpc_error = error
            .downcast_ref::<proto::RpcError>()
            .expect("the collaboration relay must return a structured RPC error");
        assert_eq!(
            rpc_error.raw_message(),
            format!("message {request_name} was not handled")
        );
    }
}

#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
#[test]
fn native_profile_relay_completion_helper() {
    let Some(socket) = std::env::var_os("ZED_TEST_OBSERVER_SOCKET") else {
        return;
    };
    let mut command = std::process::Command::new("/bin/sh");
    command.arg("-c").arg("test -z \"$ZED_PROFILE_TASK_TOKEN\"");
    panic!(
        "{}",
        util::macos_task_completion::register_and_exec(std::path::Path::new(&socket), &mut command)
    );
}

#[cfg(feature = "rust-tools-test")]
struct RelayGatedMetadataRunner {
    started: Mutex<Option<futures::channel::oneshot::Sender<()>>>,
    release: Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
}

#[cfg(feature = "rust-tools-test")]
#[async_trait::async_trait]
impl CargoMetadataRunner for RelayGatedMetadataRunner {
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
            include_str!("../../../project/test_data/cargo_workspace/standalone-v1.json")
                .replace("/standalone", "/relay")
                .into_bytes(),
        )
    }
}

#[cfg(feature = "rust-tools-test")]
struct RelayFixtureMetadataRunner;

#[cfg(feature = "rust-tools-test")]
#[async_trait::async_trait]
impl CargoMetadataRunner for RelayFixtureMetadataRunner {
    async fn run(&self, _: CargoMetadataRequest) -> anyhow::Result<Vec<u8>> {
        Ok(
            include_str!("../../../project/test_data/cargo_workspace/standalone-v1.json")
                .replace("/standalone", "/relay")
                .into_bytes(),
        )
    }
}

#[cfg(feature = "rust-tools-test")]
struct RelayGatedDiscoveryRunner {
    started: Mutex<Option<futures::channel::oneshot::Sender<()>>>,
    release: Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
}

#[cfg(feature = "rust-tools-test")]
struct RelayCancellationDiscoveryRunner {
    started: Mutex<Option<futures::channel::oneshot::Sender<()>>>,
    releases: Arc<Mutex<Vec<(u64, futures::channel::oneshot::Sender<()>)>>>,
}

#[cfg(feature = "rust-tools-test")]
fn relay_discovery_capture() -> RustTestDiscoveryCapture {
    RustTestDiscoveryCapture {
        toolchain: "fixture-only".into(),
        cargo_messages: serde_json::json!({
            "reason": "compiler-artifact",
            "package_id": "path+file:///relay#standalone@1.0.0",
            "manifest_path": "/relay/Cargo.toml",
            "target": {
                "name": "standalone",
                "required-features": [],
                "kind": ["bin"],
                "crate_types": ["bin"],
                "src_path": "/relay/src/main.rs",
                "edition": "2021",
                "test": true,
                "doctest": false,
                "doc": true
            },
            "profile": {
                "opt_level": "0",
                "debuginfo": 0,
                "debug_assertions": true,
                "overflow_checks": true,
                "test": true
            },
            "features": [],
            "filenames": ["/relay/target/debug/deps/standalone"],
            "executable": "/relay/target/debug/deps/standalone",
            "fresh": false
        })
        .to_string(),
        listings: vec![RustTestListingCapture {
            package_id: "path+file:///relay#standalone@1.0.0".into(),
            target_name: "standalone".into(),
            target_kind: RustTestTargetKind::Binary,
            mode: RustTestListingMode::All,
            stdout: "host_case: test\n1 test, 0 benchmarks\n".into(),
        }],
    }
}

#[cfg(feature = "rust-tools-test")]
struct RelayFixtureDiscoveryRunner;

#[cfg(feature = "rust-tools-test")]
impl RustTestDiscoveryRunner for RelayFixtureDiscoveryRunner {
    fn discover(
        &self,
        _: RustTestDiscoveryRequest,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
        Box::pin(async { Ok(relay_discovery_capture()) })
    }
}

#[cfg(feature = "rust-tools-test")]
impl RustTestDiscoveryRunner for RelayGatedDiscoveryRunner {
    fn discover(
        &self,
        _: RustTestDiscoveryRequest,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
        let Some(release) = self.release.lock().expect("discovery gate lock").take() else {
            return Box::pin(async { Ok(relay_discovery_capture()) });
        };
        self.started
            .lock()
            .expect("discovery start lock")
            .take()
            .expect("one discovery start notification")
            .send(())
            .expect("discovery start receiver");
        Box::pin(async move {
            release.await?;
            Ok(relay_discovery_capture())
        })
    }
}

#[cfg(feature = "rust-tools-test")]
impl RustTestDiscoveryRunner for RelayCancellationDiscoveryRunner {
    fn discover(
        &self,
        request: RustTestDiscoveryRequest,
    ) -> futures::future::BoxFuture<'static, anyhow::Result<RustTestDiscoveryCapture>> {
        let (release, released) = futures::channel::oneshot::channel();
        self.releases
            .lock()
            .expect("discovery release lock")
            .push((request.generation, release));
        if let Some(started) = self.started.lock().expect("discovery start lock").take() {
            started.send(()).expect("discovery start receiver");
        }
        Box::pin(async move {
            released.await?;
            Ok(relay_discovery_capture())
        })
    }
}

#[cfg(feature = "rust-tools-test")]
fn relay_paged_profile() -> Vec<u8> {
    let mut profile: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"
    ))
    .expect("captured profile fixture");
    let thread = &mut profile["threads"][0];
    let old_frames = thread["frameTable"]["length"].as_u64().expect("frames") as usize;
    let old_functions = thread["funcTable"]["length"].as_u64().expect("functions") as usize;
    let old_stacks = thread["stackTable"]["length"].as_u64().expect("stacks") as usize;
    let old_samples = thread["samples"]["length"].as_u64().expect("samples") as usize;
    let old_strings = thread["stringArray"].as_array().expect("strings").len();
    const EXTRA: usize = 3_000;
    for name in ["frameTable", "funcTable", "stackTable", "samples"] {
        let table = thread[name].as_object_mut().expect("table");
        let length = table["length"].as_u64().expect("table length") as usize;
        for column in table
            .values_mut()
            .filter_map(serde_json::Value::as_array_mut)
        {
            let fill = column.last().cloned().expect("nonempty column");
            column.resize(length + EXTRA, fill);
        }
        table.insert("length".into(), serde_json::json!(length + EXTRA));
    }
    for index in 0..EXTRA {
        thread["stringArray"]
            .as_array_mut()
            .expect("strings")
            .push(serde_json::json!(format!(
                "synthetic_{index:04}_{}",
                "x".repeat(400)
            )));
        thread["funcTable"]["name"][old_functions + index] = serde_json::json!(old_strings + index);
        thread["funcTable"]["resource"][old_functions + index] = serde_json::json!(-1);
        thread["frameTable"]["func"][old_frames + index] = serde_json::json!(old_functions + index);
        thread["frameTable"]["address"][old_frames + index] = serde_json::json!(0);
        thread["stackTable"]["frame"][old_stacks + index] = serde_json::json!(old_frames + index);
        thread["stackTable"]["prefix"][old_stacks + index] = serde_json::Value::Null;
        thread["samples"]["stack"][old_samples + index] = serde_json::json!(old_stacks + index);
        thread["samples"]["weight"][old_samples + index] = serde_json::json!(1);
    }
    serde_json::to_vec(&profile).expect("bounded paged profile fixture")
}

#[cfg(any(feature = "rust-tools-test", feature = "structured-execution-test"))]
#[gpui::test]
async fn jvm_results_relay_filters_guest_module_revocation(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    use gpui::BorrowAppContext as _;
    use project::structured_execution::{
        DiscoveryGeneration, StructuredExecutionEvent, StructuredNode, StructuredNodeId,
        StructuredNodeKind, StructuredNodeState, StructuredProviderId, StructuredProviderSnapshot,
        StructuredProviderStatus, StructuredRun, StructuredRunId, StructuredRunPhase,
    };

    let mut server = TestServer::start(executor.clone()).await;
    let host = server.create_client(cx_a, "jvm_host").await;
    let guest = server.create_client(cx_b, "jvm_guest").await;
    let channel_id = server.make_public_channel("jvm-results", &host, cx_a).await;
    host.fs()
        .insert_tree(
            "/jvm-relay",
            serde_json::json!({ "module": {
                "pom.xml": "<project/>",
                "src": { "test": { "java": { "GreetingTest.java": "class GreetingTest {}" } } },
                "target": { "generated-sources": {} }
            } }),
        )
        .await;
    host.fs()
        .insert_tree(
            "/outside",
            serde_json::json!({ "External.java": "class External {}" }),
        )
        .await;
    host.fs()
        .insert_symlink(
            "/jvm-relay/module/target/generated-sources/External.java",
            std::path::PathBuf::from("/outside/External.java"),
        )
        .await;
    let project = host
        .build_local_project_with_trust("/jvm-relay", cx_a)
        .await
        .0;
    let call = cx_a.read(ActiveCall::global);
    call.update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .expect("host joins the result fixture channel");
    let project_id = call
        .update(cx_a, |call, cx| call.share_project(project.clone(), cx))
        .await
        .expect("host shares the JVM result fixture");
    cx_b.update(|cx| workspace::join_channel(channel_id, guest.app_state.clone(), None, None, cx))
        .await
        .expect("guest joins the shared project");
    executor.run_until_parked();
    let guest_project = cx_b.read(ActiveCall::global).read_with(cx_b, |call, _| {
        call.location()
            .expect("guest project location")
            .upgrade()
            .expect("guest project")
    });
    assert!(guest_project.read_with(cx_b, |project, cx| project.is_read_only(cx)));
    let (manifest, source, generated, store) = project.read_with(cx_a, |project, cx| {
        let generated = project
            .find_project_path(
                "/jvm-relay/module/target/generated-sources/External.java",
                cx,
            )
            .expect("scanned outside-root generated source");
        assert!(
            project
                .entry_for_path(&generated, cx)
                .is_some_and(|entry| entry.is_external)
        );
        (
            project
                .find_project_path("/jvm-relay/module/pom.xml", cx)
                .expect("module manifest"),
            project
                .find_project_path("/jvm-relay/module/src/test/java/GreetingTest.java", cx)
                .expect("test source"),
            generated,
            project.structured_execution_store().clone(),
        )
    });
    let provider_id = StructuredProviderId("jvm-junit".into());
    let run_id = StructuredRunId("guest-jvm-run".into());
    let case_id = StructuredNodeId("jvm-case".into());
    store.update(cx_a, |store, _| {
        let generation = store.state().project_generation();
        store
            .state_mut()
            .apply_discovery(
                generation,
                StructuredProviderSnapshot::discovery(
                    provider_id.clone(),
                    DiscoveryGeneration(1),
                    StructuredProviderStatus::Current,
                    vec![
                        StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: StructuredNodeId("jvm-root".into()),
                            parent_id: None,
                            label: "Java tests".into(),
                            kind: StructuredNodeKind::Provider,
                            path: None,
                        },
                        StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: StructuredNodeId("jvm-module".into()),
                            parent_id: Some(StructuredNodeId("jvm-root".into())),
                            label: "SECRET module".into(),
                            kind: StructuredNodeKind::Group,
                            path: Some(manifest.clone()),
                        },
                        StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: case_id.clone(),
                            parent_id: Some(StructuredNodeId("jvm-module".into())),
                            label: "messageIsAvailable".into(),
                            kind: StructuredNodeKind::Case,
                            path: Some(source.clone()),
                        },
                    ],
                ),
                None,
            )
            .expect("host publishes JVM-shaped discovery");
        store
            .state_mut()
            .begin_run(
                generation,
                &provider_id,
                StructuredRun::new(
                    run_id.clone(),
                    DiscoveryGeneration(1),
                    vec![case_id.clone()],
                ),
            )
            .expect("host begins the result fixture run");
        for (sequence, location) in [(0, source.clone()), (1, generated.clone())] {
            store
                .state_mut()
                .apply_event(
                    generation,
                    &provider_id,
                    &run_id,
                    StructuredExecutionEvent {
                        sequence,
                        node_id: case_id.clone(),
                        state: StructuredNodeState::Passed,
                        duration_millis: None,
                        message: Some("HOST_SECRET result details".into()),
                        location: Some(location),
                    },
                    None,
                )
                .expect("host publishes the result fixture event");
        }
        store
            .state_mut()
            .set_run_phase(
                generation,
                &provider_id,
                &run_id,
                StructuredRunPhase::Completed,
                None,
            )
            .expect("host completes the result fixture run");
    });
    let snapshot_request = || proto::GetStructuredExecutionSnapshot {
        project_id,
        provider_id: provider_id.0.clone(),
        discovery_generation: 1,
        page_start: 0,
        page_size: 10,
        worktree_ids: vec![manifest.worktree_id.to_proto()],
    };
    let events_request = || proto::GetStructuredExecutionEvents {
        project_id,
        provider_id: provider_id.0.clone(),
        run_id: run_id.0.clone(),
        start_sequence: 0,
        chunk_size: 10,
        worktree_ids: vec![manifest.worktree_id.to_proto()],
    };
    let visible = guest
        .client()
        .request(snapshot_request())
        .await
        .expect("guest reads visible module");
    assert_eq!(visible.nodes.len(), 3);
    assert!(visible.current_run.is_some());
    assert!(visible.last_complete_run.is_some());
    let events = guest
        .client()
        .request(events_request())
        .await
        .expect("guest reads confined events");
    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].location.as_ref(), Some(&source.to_proto()));
    assert!(!format!("{events:?}").contains("HOST_SECRET"));
    assert!(!format!("{events:?}").contains("External.java"));

    let (assembled, release, completed) = store.update(cx_a, |store, _| {
        store.gate_next_assembled_response_for_test()
    });
    let pending = guest.client().request(snapshot_request());
    let revoke_manifest = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_a.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/pom.xml".into()]));
                });
            });
        });
        executor.run_until_parked();
        project.read_with(cx_a, |project, cx| {
            assert!(
                project
                    .entry_for_path(&manifest, cx)
                    .is_some_and(|entry| entry.is_private)
            );
            assert!(
                project
                    .entry_for_path(&source, cx)
                    .is_some_and(|entry| !entry.is_private)
            );
        });
        release
            .send(())
            .expect("release assembled guest snapshot after revocation");
        completed
            .await
            .expect("host rechecked the assembled guest response");
    };
    let (revoked, ()) = futures::join!(pending, revoke_manifest);
    let revoked = revoked.expect("guest receives a currently filtered response");
    assert_eq!(revoked.nodes.len(), 1);
    assert!(revoked.partial);
    assert_eq!(
        revoked.status,
        proto::StructuredProviderStatus::Partial as i32
    );
    assert!(revoked.current_run.is_none());
    assert!(revoked.last_complete_run.is_none());
    assert!(!format!("{revoked:?}").contains("SECRET"));
    assert!(!format!("{revoked:?}").contains("messageIsAvailable"));
    assert!(guest.client().request(events_request()).await.is_err());
    store.read_with(cx_a, |store, _| {
        let host_snapshot = store
            .state()
            .provider(&provider_id)
            .expect("authoritative host result");
        assert_eq!(host_snapshot.nodes.len(), 3);
        assert_eq!(
            host_snapshot
                .current_run
                .as_ref()
                .expect("host run retained")
                .events
                .len(),
            2
        );
    });
    cx_a.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files = None
            });
        });
    });
    executor.run_until_parked();
    let restored = guest
        .client()
        .request(snapshot_request())
        .await
        .expect("fresh guest read after permission restoration");
    assert_eq!(restored.nodes.len(), 3);
    assert!(restored.current_run.is_some());
    cx_b.read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .expect("guest leaves the shared project");
    executor.run_until_parked();
    assert!(guest.client().request(snapshot_request()).await.is_err());
}

#[gpui::test]
async fn test_channel_guests(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let channel_id = server
        .make_public_channel("the-channel", &client_a, cx_a)
        .await;

    // Client A shares a project in the channel
    let project_a = client_a.build_test_project(cx_a).await;
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .unwrap();
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    cx_a.executor().run_until_parked();

    // Client B joins channel A as a guest
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .unwrap();

    // b should be following a in the shared project.
    // B is a guest,
    executor.run_until_parked();

    let active_call_b = cx_b.read(ActiveCall::global);
    let project_b =
        active_call_b.read_with(cx_b, |call, _| call.location().unwrap().upgrade().unwrap());
    let room_b = active_call_b.update(cx_b, |call, _| call.room().unwrap().clone());

    assert_eq!(
        project_b.read_with(cx_b, |project, _| project.remote_id()),
        Some(project_id),
    );
    assert!(project_b.read_with(cx_b, |project, cx| project.is_read_only(cx)));
    assert!(
        project_b
            .update(cx_b, |project, cx| {
                let worktree_id = project.worktrees(cx).next().unwrap().read(cx).id();
                project.create_entry((worktree_id, rel_path("b.txt")), false, cx)
            })
            .await
            .is_err()
    );
    assert!(room_b.read_with(cx_b, |room, _| room.is_muted()));
}

#[gpui::test]
#[cfg(feature = "rust-tools-test")]
async fn rust_tools_relay_enforces_guest_capability(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    use client::ErrorExt as _;
    use gpui::BorrowAppContext;
    use project::profile_results_wire::{MAX_PROFILE_PAGE_BYTES, ProfilePageAssembler};
    use proto::Message as _;
    use std::sync::atomic::AtomicBool;

    #[cfg(target_os = "macos")]
    {
        executor.allow_parking();
        cx_a.executor().allow_parking();
        cx_b.executor().allow_parking();
        cx_c.executor().allow_parking();
    }

    let mut server = TestServer::start(executor.clone()).await;
    #[cfg(target_os = "macos")]
    cx_b.update(language_tools::profile_view::init);
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    cx_a.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        );
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        )
    });
    cx_b.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    cx_c.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let channel_id = server
        .make_public_channel("rust-tools-relay", &client_a, cx_a)
        .await;
    client_a
        .fs()
        .insert_tree(
            "/relay",
            serde_json::json!({
                ".gitignore": ".env\n",
                ".env": "SECRET=private\n",
                "source.rs": "fn public() {}\n",
            }),
        )
        .await;
    let coverage_artifact = serde_json::json!({
        "type": "llvm.coverage.json.export",
        "version": "2.0.1",
        "cargo_llvm_cov": {"version": "0.9.0", "manifest_path": "/relay/Cargo.toml"},
        "data": [{"files": [
            {"filename": "/relay/source.rs", "segments": [[1, 1, 1, true, true, false]], "summary": {"lines": {"count": 1, "covered": 1}}},
            {"filename": "/relay/.env", "segments": [[1, 1, 1, true, true, false]], "summary": {"lines": {"count": 1, "covered": 1}}}
        ]}]
    });
    client_a
        .fs()
        .insert_file(
            "/relay/coverage.json",
            coverage_artifact.to_string().into_bytes(),
        )
        .await;
    let project_a = client_a
        .build_local_project_with_trust("/relay", cx_a)
        .await
        .0;
    let active_call_a = cx_a.read(ActiveCall::global);
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .expect("host joins channel");
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("host shares project");
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .expect("guest joins channel");
    cx_c.update(|cx| {
        workspace::join_channel(channel_id, client_c.app_state.clone(), None, None, cx)
    })
    .await
    .expect("second guest joins channel");
    executor.run_until_parked();

    let (path, private_path, coverage_artifact_path, store) =
        project_a.read_with(cx_a, |project, cx| {
            (
                project
                    .find_project_path("/relay/source.rs", cx)
                    .expect("visible Rust source"),
                project
                    .find_project_path("/relay/.env", cx)
                    .expect("private source"),
                project
                    .find_project_path("/relay/coverage.json", cx)
                    .expect("visible coverage artifact"),
                project.source_coverage_store().clone(),
            )
        });
    project_a.read_with(cx_a, |project, cx| {
        assert!(
            project
                .entry_for_path(&private_path, cx)
                .is_some_and(|entry| entry.is_private),
            "the fixture must contain a scanned private entry"
        );
    });
    store
        .update(cx_a, |store, cx| {
            store.publish(
                SourceCoverageSnapshot {
                    project_generation: 1,
                    provider_id: SourceCoverageProviderId("rust-tools-relay".into()),
                    generation: 1,
                    status: SourceCoverageStatus::Current,
                    truncated: false,
                    diagnostic: None,
                    files: [path.clone(), private_path.clone()]
                        .into_iter()
                        .map(|path| SourceCoverageFile {
                            path,
                            ranges: vec![SourceCoverageRange {
                                start: SourceCoveragePoint { line: 0, column: 0 },
                                end: SourceCoveragePoint { line: 0, column: 1 },
                                hit_count: 1,
                            }],
                            covered_lines: 1,
                            uncovered_lines: 0,
                            truncated: false,
                        })
                        .collect(),
                },
                cx,
            )
        })
        .expect("publish host coverage");

    let result = client_b
        .client()
        .request(proto::GetSourceCoverage {
            project_id,
            provider_id: "rust-tools-relay".into(),
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("guest can read visible coverage through relay");
    assert_eq!(result.files.len(), 1);
    assert!(result.truncated, "a private source was withheld");
    assert_eq!(result.files[0].path.as_ref(), Some(&path.to_proto()));
    assert_eq!(result.files[0].ranges[0].hit_count, 1);

    let (assembled, release, completed) = store.update(cx_a, |store, _| {
        store.gate_next_assembled_response_for_test()
    });
    let guest_client = Arc::clone(client_b.client());
    let delayed_path = path.clone();
    let delayed_request = executor.spawn(async move {
        guest_client
            .request(proto::GetSourceCoverage {
                project_id,
                provider_id: "rust-tools-relay".into(),
                worktree_ids: vec![delayed_path.worktree_id.to_proto()],
            })
            .await
    });
    let deadline = executor.timer(std::time::Duration::from_secs(5));
    assert!(matches!(
        futures::future::select(assembled, deadline).await,
        futures::future::Either::Left((Ok(()), _))
    ));
    cx_a.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files =
                    Some(settings::ExtendingVec(vec!["**/source.rs".into()]));
            });
        });
    });
    executor.run_until_parked();
    project_a.read_with(cx_a, |project, cx| {
        assert!(
            project
                .entry_for_path(&path, cx)
                .is_some_and(|entry| entry.is_private),
            "the published source must become private on the host"
        );
    });
    release
        .send(())
        .expect("release assembled relay coverage response");
    completed.await.expect("host released relay response");
    let delayed = delayed_request
        .await
        .expect("guest receives delayed coverage reply");
    assert!(delayed.files.is_empty());
    assert!(delayed.truncated);
    let revoked = client_b
        .client()
        .request(proto::GetSourceCoverage {
            project_id,
            provider_id: "rust-tools-relay".into(),
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("guest receives filtered coverage after host revocation");
    assert!(revoked.files.is_empty());
    assert!(revoked.truncated);
    cx_a.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files = None;
            });
        });
    });
    executor.run_until_parked();
    project_a.read_with(cx_a, |project, cx| {
        assert!(
            project
                .entry_for_path(&path, cx)
                .is_some_and(|entry| !entry.is_private),
            "the published source must become visible again"
        );
    });
    let restored = client_b
        .client()
        .request(proto::GetSourceCoverage {
            project_id,
            provider_id: "rust-tools-relay".into(),
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("guest receives restored visible coverage");
    assert_eq!(restored.files.len(), 1);
    assert_eq!(restored.files[0].path.as_ref(), Some(&path.to_proto()));

    let structured_provider = StructuredProviderId("rust-tools-relay-read-only".into());
    project_a
        .read_with(cx_a, |project, _| {
            project.structured_execution_store().clone()
        })
        .update(cx_a, |store, _| {
            let project_generation = store.state().project_generation();
            let mut snapshot = StructuredProviderSnapshot::discovery(
                structured_provider.clone(),
                DiscoveryGeneration(1),
                StructuredProviderStatus::Current,
                vec![
                    StructuredNode {
                        experimental_feature: None,
                        experimental_feature_enabled: None,
                        debug_capability: None,
                        id: StructuredNodeId("root".into()),
                        parent_id: None,
                        label: "Rust tests".into(),
                        kind: StructuredNodeKind::Provider,
                        path: None,
                    },
                    StructuredNode {
                        experimental_feature: None,
                        experimental_feature_enabled: None,
                        debug_capability: None,
                        id: StructuredNodeId("public".into()),
                        parent_id: Some(StructuredNodeId("root".into())),
                        label: "public case".into(),
                        kind: StructuredNodeKind::Case,
                        path: Some(path.clone()),
                    },
                    StructuredNode {
                        experimental_feature: None,
                        experimental_feature_enabled: None,
                        debug_capability: None,
                        id: StructuredNodeId("private".into()),
                        parent_id: Some(StructuredNodeId("root".into())),
                        label: "SECRET private case".into(),
                        kind: StructuredNodeKind::Case,
                        path: Some(private_path.clone()),
                    },
                ],
            );
            snapshot.diagnostic = Some("/relay/.env TOKEN=secret".into());
            store
                .state_mut()
                .apply_discovery(project_generation, snapshot, None)
                .expect("publish host structured snapshot");
            let run_id = StructuredRunId("relay-run".into());
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &structured_provider,
                    StructuredRun::new(
                        run_id.clone(),
                        DiscoveryGeneration(1),
                        vec![
                            StructuredNodeId("public".into()),
                            StructuredNodeId("private".into()),
                        ],
                    ),
                )
                .expect("start host structured run");
            for (sequence, node_id, location) in [
                (0, "public", path.clone()),
                (1, "private", private_path.clone()),
                (2, "public", private_path.clone()),
            ] {
                store
                    .state_mut()
                    .apply_event(
                        project_generation,
                        &structured_provider,
                        &run_id,
                        StructuredExecutionEvent {
                            sequence,
                            node_id: StructuredNodeId(node_id.into()),
                            state: StructuredNodeState::Passed,
                            duration_millis: None,
                            message: Some("TOKEN=secret".into()),
                            location: Some(location),
                        },
                        None,
                    )
                    .expect("publish host structured event");
            }
        });
    let structured = client_b
        .client()
        .request(proto::GetStructuredExecutionSnapshot {
            project_id,
            provider_id: structured_provider.0.clone(),
            discovery_generation: 1,
            page_start: 0,
            page_size: 10,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("guest reads host structured snapshot through relay");
    assert_eq!(structured.provider_id, structured_provider.0);
    assert_eq!(structured.discovery_generation, 1);
    assert_eq!(
        structured.status,
        proto::StructuredProviderStatus::Partial as i32
    );
    assert_eq!(structured.nodes.len(), 2);
    assert!(structured.partial);
    assert!(!format!("{structured:?}").contains("SECRET"));
    assert!(!format!("{structured:?}").contains("TOKEN=secret"));
    assert_eq!(
        structured
            .current_run
            .as_ref()
            .and_then(|run| run.summary.as_ref())
            .map(|summary| summary.total),
        Some(1)
    );
    let events = client_b
        .client()
        .request(proto::GetStructuredExecutionEvents {
            project_id,
            provider_id: structured_provider.0.clone(),
            run_id: "relay-run".into(),
            start_sequence: 0,
            chunk_size: 10,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("guest reads filtered host structured events");
    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].node_id, "public");
    assert_eq!(events.events[0].sequence, 0);
    assert!(!format!("{events:?}").contains("TOKEN=secret"));

    let denied_coverage_artifact = client_b
        .client()
        .request(proto::InterpretRustCoverageArtifact {
            project_id,
            artifact_path: Some(path.to_proto()),
            max_bytes: 0,
        })
        .await;
    assert!(
        denied_coverage_artifact.is_err(),
        "a read-only guest cannot interpret a host coverage artifact"
    );
    let denied_profile_page = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 1,
            generation: 1,
            protocol_version: u32::MAX,
            page_index: 0,
        })
        .await;
    assert!(
        denied_profile_page.is_err(),
        "a read-only guest cannot request a host profile page"
    );
    let denied_profile_task = client_b
        .client()
        .request(proto::PrepareProfileTaskBinding {
            project_id,
            protocol_version: 1,
            task_run_id: "guest-profile-task".into(),
            profile_path: Some(path.to_proto()),
            symbols_path: Some(private_path.to_proto()),
            artifact_format: "samply-0.13.1".into(),
        })
        .await;
    assert!(
        denied_profile_task.is_err(),
        "a read-only guest cannot request a host profile task token"
    );
    let denied_profile_task_status = client_b
        .client()
        .request(proto::GetProfileTaskBinding {
            project_id,
            protocol_version: 1,
            task_run_id: "guest-profile-task".into(),
        })
        .await;
    assert!(
        denied_profile_task_status.is_err(),
        "a read-only guest cannot read a host profile task binding"
    );
    let denied_profile_task_cancel = client_b
        .client()
        .request(proto::CancelProfileTaskBinding {
            project_id,
            task_run_id: "guest-profile-task".into(),
        })
        .await;
    assert!(
        denied_profile_task_cancel.is_err(),
        "a read-only guest cannot cancel a host profile task binding"
    );

    let denied_cargo = client_b
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 21,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await;
    assert!(
        denied_cargo.is_err(),
        "a read-only guest cannot start Cargo metadata collection"
    );
    let denied_discovery_cancel = client_b
        .client()
        .request(proto::CancelRustTestDiscovery {
            project_id,
            request_id: 22,
        })
        .await;
    assert!(
        denied_discovery_cancel.is_err(),
        "a read-only guest cannot cancel Rust test discovery"
    );
    let denied_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: 1,
            node_id: "guest-rust-case".into(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![path.worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await;
    assert!(
        denied_action.is_err(),
        "a read-only guest cannot resolve a Rust test action"
    );
    let denied_run = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: 1,
            run_id: "guest-rust-run".into(),
            task_id: "guest-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![path.worktree_id.to_proto()],
            ..Default::default()
        })
        .await;
    assert!(
        denied_run.is_err(),
        "a read-only guest cannot update a Rust test run"
    );
    let denied_import = client_b
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 23,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(path.to_proto()),
            symbols_path: Some(private_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await;
    assert!(
        denied_import.is_err(),
        "a read-only guest cannot import a native profile"
    );
    let denied_import_cancel = client_b
        .client()
        .request(proto::CancelProfileImport {
            project_id,
            import_id: 23,
            generation: 1,
        })
        .await;
    assert!(
        denied_import_cancel.is_err(),
        "a read-only guest cannot cancel a native profile import"
    );

    let denied = client_b
        .client()
        .request(proto::CancelCargoWorkspace {
            project_id,
            request_id: 1,
        })
        .await;
    assert!(denied.is_err(), "a read-only guest cannot cancel a runner");
    let denied_discovery = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 3,
            protocol_version: u32::MAX,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await;
    assert!(
        denied_discovery.is_err(),
        "a read-only guest cannot start Rust test discovery"
    );
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("guest identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("promote collaborator");
    executor.run_until_parked();
    cx_b.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        )
    });
    cx_a.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            false,
            cx,
        )
    });
    executor.run_until_parked();
    let disabled_interpretation = client_b
        .client()
        .request(proto::InterpretRustCoverageArtifact {
            project_id,
            artifact_path: Some(coverage_artifact_path.to_proto()),
            max_bytes: project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64,
        })
        .await
        .expect_err("client opt-in cannot enable coverage on the authoritative host");
    assert_eq!(
        disabled_interpretation
            .downcast_ref::<proto::RpcError>()
            .and_then(|error| error.error_tag("reason")),
        Some("rust_tools_unsupported")
    );
    let ordinary_coverage = client_b
        .client()
        .request(proto::GetSourceCoverage {
            project_id,
            provider_id: "rust-tools-relay".into(),
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("Rust opt-out preserves ordinary host coverage");
    assert_eq!(ordinary_coverage.files.len(), 1);
    assert_eq!(
        ordinary_coverage.files[0].path.as_ref(),
        Some(&path.to_proto())
    );
    assert!(ordinary_coverage.truncated);
    cx_a.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        )
    });
    executor.run_until_parked();
    let interpreted_coverage = client_b
        .client()
        .request(proto::InterpretRustCoverageArtifact {
            project_id,
            artifact_path: Some(coverage_artifact_path.to_proto()),
            max_bytes: project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES as u64,
        })
        .await
        .expect("member interpretation reaches the authoritative host");
    assert_eq!(interpreted_coverage.files.len(), 1);
    assert_eq!(
        interpreted_coverage.files[0].path.as_ref(),
        Some(&path.to_proto())
    );
    assert!(
        interpreted_coverage.truncated,
        "private source was withheld"
    );
    let discovery_mismatch = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 4,
            protocol_version: u32::MAX,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("member discovery reaches the authoritative host");
    assert_eq!(discovery_mismatch.request_id, 4);
    assert_eq!(
        discovery_mismatch.status,
        proto::StructuredProviderStatus::Mismatch as i32
    );
    let profile_mismatch = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 2,
            generation: 1,
            protocol_version: u32::MAX,
            page_index: 0,
        })
        .await
        .expect("member profile request reaches the authoritative host");
    assert_eq!(
        profile_mismatch.status,
        proto::ProfileResultStatus::Mismatch as i32
    );
    let profile_task_mismatch = client_b
        .client()
        .request(proto::GetProfileTaskBinding {
            project_id,
            protocol_version: u32::MAX,
            task_run_id: "member-profile-task".into(),
        })
        .await
        .expect("member profile task request reaches the authoritative host");
    assert_eq!(
        profile_task_mismatch.status,
        proto::ProfileResultStatus::Mismatch as i32
    );
    assert!(profile_task_mismatch.token.is_none());
    client_b
        .client()
        .request(proto::CancelCargoWorkspace {
            project_id,
            request_id: 2,
        })
        .await
        .expect("collaborator cancellation reaches the host");

    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_c.user_id().expect("second guest identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("promote second collaborator");
    executor.run_until_parked();
    client_b
        .client()
        .request(proto::CancelRustTestDiscovery {
            project_id,
            request_id: 909,
        })
        .await
        .expect("first member Rust discovery cancellation is recorded");
    let other_discovery = client_c
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 909,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("another member's same-ID Rust discovery is not cancelled");
    assert_eq!(other_discovery.request_id, 909);
    let cancelled_discovery = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 909,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await;
    assert!(
        cancelled_discovery
            .expect_err("only the owner sees its Rust discovery cancellation")
            .to_string()
            .contains("cancelled")
    );
    client_b
        .client()
        .request(proto::CancelCargoWorkspace {
            project_id,
            request_id: 910,
        })
        .await
        .expect("first collaborator cancellation is recorded");
    let other_request = client_c
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 910,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect("another collaborator is not cancelled by the first peer");
    assert_eq!(other_request.request_id, 910);
    let cancelled = client_b
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 910,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await;
    assert!(cancelled.is_err(), "only the owner sees its cancellation");

    client_a
        .fs()
        .insert_file(
            "/relay/Cargo.toml",
            "[package]\nname = \"relay\"\nversion = \"0.1.0\"\n".into(),
        )
        .await;
    executor.run_until_parked();
    project_a.read_with(cx_a, |project, cx| {
        assert!(
            project.find_project_path("/relay/Cargo.toml", cx).is_some(),
            "metadata fixture must be scanned before the request"
        );
    });
    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    let (store, worktree_store, environment) = project_a.read_with(cx_a, |project, _| {
        (
            project.cargo_workspace_store().clone(),
            project.worktree_store(),
            project.environment().clone(),
        )
    });
    store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store,
            environment,
            Arc::new(RelayGatedMetadataRunner {
                started: Mutex::new(Some(started)),
                release: Mutex::new(Some(release_receiver)),
            }),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    let metadata = client_c.client().request(proto::GetCargoWorkspace {
        project_id,
        request_id: 911,
        worktree_ids: vec![path.worktree_id.to_proto()],
    });
    let cancel = async {
        started_receiver.await.expect("metadata runner started");
        client_b
            .client()
            .request(proto::CancelCargoWorkspace {
                project_id,
                request_id: 911,
            })
            .await
            .expect("other member cancellation acknowledged");
        executor.run_until_parked();
        assert!(
            !release.is_canceled(),
            "another member must not drop the active host runner"
        );
        client_c
            .client()
            .request(proto::CancelCargoWorkspace {
                project_id,
                request_id: 911,
            })
            .await
            .expect("owner cancellation acknowledged");
    };
    let (result, ()) = futures::join!(metadata, cancel);
    let error = result.expect_err("owner cancellation rejects active metadata");
    assert!(error.to_string().contains("cancelled"), "{error:#}");
    executor.run_until_parked();
    assert!(release.is_canceled(), "owner cancellation drops the runner");

    let (discovery_started, discovery_started_receiver) = futures::channel::oneshot::channel();
    let discovery_releases = Arc::new(Mutex::new(Vec::new()));
    let (cargo_store, rust_store, worktree_store, environment, structured_store) = project_a
        .read_with(cx_a, |project, _| {
            (
                project.cargo_workspace_store().clone(),
                project.rust_test_provider_store().clone(),
                project.worktree_store(),
                project.environment().clone(),
                project.structured_execution_store().clone(),
            )
        });
    cargo_store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store.clone(),
            environment.clone(),
            Arc::new(RelayFixtureMetadataRunner),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    rust_store.update(cx_a, |store, cx| {
        *store = RustTestProviderStore::local_with_runner(
            worktree_store,
            environment,
            cargo_store,
            structured_store,
            Arc::new(RelayCancellationDiscoveryRunner {
                started: Mutex::new(Some(discovery_started)),
                releases: discovery_releases.clone(),
            }),
            cx,
        );
    });
    let discovery = client_c.client().request(proto::GetRustTestDiscovery {
        project_id,
        request_id: 912,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        worktree_ids: vec![path.worktree_id.to_proto()],
    });
    let cancel_discovery = async {
        discovery_started_receiver
            .await
            .expect("discovery runner started");
        client_b
            .client()
            .request(proto::CancelRustTestDiscovery {
                project_id,
                request_id: 912,
            })
            .await
            .expect("other member discovery cancellation acknowledged");
        executor.run_until_parked();
        {
            let releases = discovery_releases.lock().expect("discovery release lock");
            let (_, latest) = releases
                .iter()
                .max_by_key(|(generation, _)| *generation)
                .expect("active discovery generation");
            assert!(
                !latest.is_canceled(),
                "another member must not drop the active discovery runner"
            );
        }
        let concurrent_discovery = executor.spawn({
            let client = client_b.client().clone();
            let worktree_id = path.worktree_id.to_proto();
            async move {
                client
                    .request(proto::GetRustTestDiscovery {
                        project_id,
                        request_id: 913,
                        protocol_version:
                            project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                        worktree_ids: vec![worktree_id],
                    })
                    .await
            }
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while rust_store.read_with(cx_a, |store, _| {
            store.active_remote_discovery_request_count_for_test()
        }) != 2
        {
            assert!(
                std::time::Instant::now() < deadline,
                "both members' discovery requests must be admitted"
            );
            executor.timer(std::time::Duration::from_millis(10)).await;
        }
        {
            let releases = discovery_releases.lock().expect("discovery release lock");
            assert_eq!(releases.len(), 1, "same-scope peers share one collection");
            assert!(
                !releases.first().expect("shared collection").1.is_canceled(),
                "the second peer must not supersede the first collection"
            );
        }
        client_c
            .client()
            .request(proto::CancelRustTestDiscovery {
                project_id,
                request_id: 912,
            })
            .await
            .expect("owner discovery cancellation acknowledged");
        executor.run_until_parked();
        assert!(
            !discovery_releases
                .lock()
                .expect("discovery release lock")
                .first()
                .expect("remaining member's collection")
                .1
                .is_canceled(),
            "one owner's cancellation must preserve the other owner's collection"
        );
        client_b
            .client()
            .request(proto::CancelRustTestDiscovery {
                project_id,
                request_id: 913,
            })
            .await
            .expect("last owner discovery cancellation acknowledged");
        let error = concurrent_discovery
            .await
            .expect_err("last owner receives its discovery cancellation");
        assert!(error.to_string().contains("cancelled"), "{error:#}");
    };
    let (result, ()) = futures::join!(discovery, cancel_discovery);
    let error = result.expect_err("owner cancellation rejects active discovery");
    assert!(error.to_string().contains("cancelled"), "{error:#}");
    executor.run_until_parked();
    {
        let releases = discovery_releases.lock().expect("discovery release lock");
        let (_, latest) = releases
            .iter()
            .max_by_key(|(generation, _)| *generation)
            .expect("cancelled discovery generation");
        assert!(
            latest.is_canceled(),
            "owner cancellation drops the discovery runner"
        );
    }

    client_a
        .fs()
        .insert_file("/relay/profile.json", relay_paged_profile())
        .await;
    client_a
        .fs()
        .insert_file("/relay/workload.rs", "fn leaf() {}\n".into())
        .await;
    client_a
        .fs()
        .insert_file(
            "/relay/profile.syms.json",
            include_str!("../../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json")
                .replace("/fixture/", "/relay/")
                .into_bytes(),
        )
        .await;
    executor.run_until_parked();
    let (profile_path, symbols_path) = project_a.read_with(cx_a, |project, cx| {
        (
            project
                .find_project_path("/relay/profile.json", cx)
                .expect("scanned profile artifact"),
            project
                .find_project_path("/relay/profile.syms.json", cx)
                .expect("scanned symbols artifact"),
        )
    });
    let unavailable_task_helper = client_b
        .client()
        .request(proto::PrepareProfileTaskBinding {
            project_id,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            task_run_id: "member-profile-task".into(),
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            artifact_format: "samply-0.13.1".into(),
        })
        .await
        .expect("member profile Task preparation reaches the authoritative host");
    assert_eq!(
        unavailable_task_helper.status,
        proto::ProfileResultStatus::Unsupported as i32
    );
    assert!(unavailable_task_helper.token.is_none());
    client_b
        .client()
        .request(proto::CancelProfileTaskBinding {
            project_id,
            task_run_id: "member-profile-task".into(),
        })
        .await
        .expect("member profile Task cancellation reaches the authoritative host");
    #[cfg(target_os = "macos")]
    {
        let project_b = {
            let (workspace_b, window_cx_b) = client_b.active_workspace(cx_b);
            workspace_b.read_with(window_cx_b, |workspace, _| workspace.project().clone())
        };
        assert_eq!(
            project_b.read_with(cx_b, |project, _| project.remote_id()),
            Some(project_id)
        );
        let profile_store = project_a.read_with(cx_a, |project, _| {
            project
                .profile_results_store_for_test()
                .cloned()
                .expect("host profile results store")
        });
        profile_store.update(cx_a, |store, _| {
            store.set_task_helper(std::env::current_exe().expect("test helper path"));
        });
        assert!(project_b.read_with(cx_b, |project, _| {
            project.profile_results_store_for_test().is_some()
        }));
        let import_owner = client_b.peer_id().expect("joined native importer");
        let imports_before = profile_store.read_with(cx_a, |store, _| {
            store.active_import_count_for_peer_for_test(import_owner)
        });
        cx_b.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                false,
                cx,
            )
        });
        executor.run_until_parked();
        let disabled = project_b
            .read_with(cx_b, |project, cx| {
                project.import_remote_profile(
                    [profile_path.clone(), symbols_path.clone()],
                    true,
                    Arc::new(AtomicBool::new(false)),
                    cx,
                )
            })
            .err()
            .expect("local native opt-out refuses before an import Task is issued");
        assert_eq!(
            disabled.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Unsupported)
        );
        assert_eq!(
            profile_store.read_with(cx_a, |store, _| {
                store.active_import_count_for_peer_for_test(import_owner)
            }),
            imports_before
        );
        cx_b.update(|cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        executor.run_until_parked();
        for held_on_host in [false, true] {
            let mut release = None;
            let mut started = None;
            if held_on_host {
                let (start_sender, start_receiver) = futures::channel::oneshot::channel();
                let (release_sender, release_receiver) = futures::channel::oneshot::channel();
                profile_store.update(cx_a, |store, _| {
                    store.gate_next_import_for_test(start_sender, release_receiver);
                });
                started = Some(start_receiver);
                release = Some(release_sender);
            }
            let cancellation = Arc::new(AtomicBool::new(false));
            let pending = project_b
                .read_with(cx_b, |project, cx| {
                    project.import_remote_profile(
                        [profile_path.clone(), symbols_path.clone()],
                        true,
                        cancellation.clone(),
                        cx,
                    )
                })
                .expect("joined member creates a native import with local opt-in");
            if let Some(started) = started {
                let deadline = executor.timer(std::time::Duration::from_secs(5));
                assert!(matches!(
                    futures::future::select(started, deadline).await,
                    futures::future::Either::Left((Ok(()), _))
                ));
                assert_eq!(
                    profile_store.read_with(cx_a, |store, _| {
                        store.active_import_count_for_peer_for_test(import_owner)
                    }),
                    imports_before + 1,
                    "the held public import has an actual authoritative host row"
                );
            }
            for enabled in [false, true] {
                cx_b.update(|cx| {
                    project::task_store::set_experimental_feature_for_test(
                        "rust-native-profiling-experimental",
                        enabled,
                        cx,
                    )
                });
                executor.run_until_parked();
            }
            let deadline = executor.timer(std::time::Duration::from_secs(10));
            let result = match futures::future::select(pending, deadline).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => {
                    panic!("native opt-out did not retire the original joined import")
                }
            };
            let error = result.expect_err("re-enabling native cannot revive the original import");
            assert_eq!(
                error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                Some(&project::profile_results_store::RemoteProfileError::Unsupported)
            );
            assert!(cancellation.load(std::sync::atomic::Ordering::Relaxed));
            if let Some(release) = release {
                match release.send(()) {
                    Ok(()) => {}
                    Err(()) => {
                        assert!(cancellation.load(std::sync::atomic::Ordering::Relaxed));
                    }
                }
            }
            let deadline = executor.timer(std::time::Duration::from_secs(10));
            let retirement = async {
                loop {
                    executor.run_until_parked();
                    if profile_store.read_with(cx_a, |store, _| {
                        store.active_import_count_for_peer_for_test(import_owner)
                    }) == imports_before
                    {
                        break;
                    }
                    executor.timer(std::time::Duration::from_millis(20)).await;
                }
            };
            assert!(matches!(
                futures::future::select(Box::pin(retirement), deadline).await,
                futures::future::Either::Left(((), _))
            ));
        }
        let prepare = || proto::PrepareProfileTaskBinding {
            project_id,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            task_run_id: "member-verified-profile-task".into(),
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            artifact_format: "samply-0.13.1".into(),
        };
        let prepared = client_b
            .client()
            .request(prepare())
            .await
            .expect("member prepares native Task on collaboration host");
        assert_eq!(prepared.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!prepared.complete);
        assert_eq!(prepared.token.as_deref().map(str::len), Some(64));
        let host_reshared = Arc::new(Mutex::new(false));
        let _reshared_subscription = project_a.update(cx_a, |_, cx| {
            let host_reshared = host_reshared.clone();
            cx.subscribe(&project_a, move |_, _, event, _| {
                if matches!(event, project::Event::Reshared) {
                    *host_reshared
                        .lock()
                        .expect("profile host reshare observation") = true;
                }
            })
        });
        for reconnect_host in [true, false] {
            let original_host_peer = client_a.peer_id().expect("profile host peer");
            let original_member_peer = client_b.peer_id().expect("profile owner peer");
            let original_other_peer = client_c.peer_id().expect("other profile member peer");
            assert_eq!(
                profile_store.read_with(cx_a, |store, _| {
                    store.task_binding_count_for_peer_for_test(original_member_peer)
                }),
                1
            );
            let previous_peer = if reconnect_host {
                *host_reshared
                    .lock()
                    .expect("profile host reshare observation") = false;
                original_host_peer
            } else {
                original_member_peer
            };
            server.disconnect_client(previous_peer);
            let step = std::time::Duration::from_millis(100);
            let mut remaining = rpc::RECEIVE_TIMEOUT;
            loop {
                executor.run_until_parked();
                let ready = if reconnect_host {
                    client_a.peer_id().is_some_and(|peer| peer != previous_peer)
                        && *host_reshared
                            .lock()
                            .expect("profile host reshare observation")
                } else {
                    client_b.peer_id().is_some_and(|peer| {
                        peer != previous_peer
                            && project_a.read_with(cx_a, |project, _| {
                                project.collaborators().contains_key(&peer)
                            })
                    })
                };
                if ready {
                    break;
                }
                assert!(remaining >= step, "profile binding route did not reconnect");
                remaining = remaining.saturating_sub(step);
                executor.advance_clock(step);
            }
            assert_eq!(client_c.peer_id(), Some(original_other_peer));
            if reconnect_host {
                assert_eq!(client_b.peer_id(), Some(original_member_peer));
            } else {
                assert_eq!(client_a.peer_id(), Some(original_host_peer));
            }
            assert_eq!(
                project_a.read_with(cx_a, |project, _| project.remote_id()),
                Some(project_id)
            );
            assert_eq!(
                profile_store.read_with(cx_a, |store, _| {
                    store.task_binding_count_for_peer_for_test(original_member_peer)
                }),
                0,
                "reconnect must release the old owner's native binding"
            );
            let obsolete = client_b
                .client()
                .request(proto::GetProfileTaskBinding {
                    project_id,
                    protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                    task_run_id: "member-verified-profile-task".into(),
                })
                .await
                .expect("reconnected member receives typed obsolete binding result");
            eprintln!(
                "native profile binding reconnect host={reconnect_host}: status={}",
                obsolete.status
            );
            assert_eq!(obsolete.status, proto::ProfileResultStatus::Stale as i32);
            assert!(obsolete.token.is_none());
            let recovered = client_b
                .client()
                .request(prepare())
                .await
                .expect("owner reserves a fresh native binding after reconnect");
            assert_eq!(recovered.status, proto::ProfileResultStatus::Ready as i32);
            assert!(!recovered.complete);
            assert_eq!(recovered.token.as_deref().map(str::len), Some(64));
        }
        project_a.read_with(cx_a, |project, _| {
            for (peer_id, collaborator) in project.collaborators() {
                assert_eq!(
                    *peer_id, collaborator.peer_id,
                    "reconnected collaborator identity"
                );
            }
        });
        project_b.read_with(cx_b, |project, _| {
            for (peer_id, collaborator) in project.collaborators() {
                assert_eq!(
                    *peer_id, collaborator.peer_id,
                    "reconnected guest collaborator identity"
                );
            }
        });
        for malformed_snapshot in [true, false] {
            let original_member_peer = client_b.peer_id().expect("snapshot native owner");
            let original_host_peer = client_a.peer_id().expect("snapshot native host");
            let original_other_peer = client_c.peer_id().expect("other snapshot native member");
            let collaborators = project_a.read_with(cx_a, |project, _| {
                project
                    .collaborators()
                    .values()
                    .map(|collaborator| proto::Collaborator {
                        peer_id: Some(collaborator.peer_id),
                        replica_id: u32::from(collaborator.replica_id.as_u16()),
                        user_id: collaborator.user_id,
                        is_host: collaborator.is_host,
                        committer_name: collaborator.committer_name.clone(),
                        committer_email: collaborator.committer_email.clone(),
                    })
                    .collect::<Vec<_>>()
            });
            assert!(
                collaborators
                    .iter()
                    .any(|collaborator| collaborator.peer_id == Some(original_member_peer))
            );
            let import_id = if malformed_snapshot { 12_001 } else { 12_002 };
            let retained = client_b
                .client()
                .request(proto::ImportProfileResults {
                    project_id,
                    import_id,
                    protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                    profile_path: Some(profile_path.to_proto()),
                    symbols_path: Some(symbols_path.to_proto()),
                    consent_to_transfer: true,
                    artifact_format: "samply-0.13.1".into(),
                    task_run_id: None,
                })
                .await
                .expect("retained native page before snapshot replacement");
            assert_eq!(retained.status, proto::ProfileResultStatus::Ready as i32);
            assert!(!retained.complete);
            let cancellation = profile_store
                .read_with(cx_a, |store, _| {
                    assert_eq!(
                        store.task_binding_count_for_peer_for_test(original_member_peer),
                        1
                    );
                    assert!(store.active_import_count_for_peer_for_test(original_member_peer) >= 1);
                    store.import_cancellation_for_test(original_member_peer, import_id)
                })
                .expect("retained snapshot import cancellation");
            assert!(!cancellation.load(std::sync::atomic::Ordering::Relaxed));
            let omitted = Arc::new(Mutex::new(false));
            let _omitted_subscription = project_a.update(cx_a, |_, cx| {
                let omitted = omitted.clone();
                cx.subscribe(&project_a, move |_, _, event, _| {
                    if matches!(event, project::Event::CollaboratorLeft(peer) if *peer == original_member_peer) {
                        *omitted.lock().expect("snapshot omission observation") = true;
                    }
                })
            });
            let replacement = if malformed_snapshot {
                vec![proto::Collaborator::default()]
            } else {
                collaborators
                    .iter()
                    .filter(|collaborator| collaborator.peer_id != Some(original_member_peer))
                    .cloned()
                    .collect()
            };
            let result = project_a.update(cx_a, |project, cx| {
                project.reshared(
                    proto::ResharedProject {
                        id: project_id,
                        collaborators: replacement,
                    },
                    cx,
                )
            });
            if malformed_snapshot {
                let error = result.expect_err("invalid native collaborator snapshot rejected");
                assert!(error.to_string().contains("invalid peer id"), "{error:#}");
                project_a.read_with(cx_a, |project, _| {
                    assert!(project.collaborators().contains_key(&original_member_peer));
                });
            } else {
                result.expect("valid native snapshot omits former collaborator");
                executor.run_until_parked();
                assert!(*omitted.lock().expect("snapshot omission observation"));
                project_a.read_with(cx_a, |project, _| {
                    assert!(!project.collaborators().contains_key(&original_member_peer));
                    assert!(project.collaborators().contains_key(&original_other_peer));
                });
            }
            assert!(cancellation.load(std::sync::atomic::Ordering::Relaxed));
            profile_store.read_with(cx_a, |store, _| {
                assert_eq!(
                    store.task_binding_count_for_peer_for_test(original_member_peer),
                    0
                );
                assert_eq!(
                    store.active_import_count_for_peer_for_test(original_member_peer),
                    0
                );
            });
            let old_binding = client_b
                .client()
                .request(proto::GetProfileTaskBinding {
                    project_id,
                    protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                    task_run_id: "member-verified-profile-task".into(),
                })
                .await
                .expect("old snapshot native binding returns typed denial");
            assert_eq!(old_binding.status, proto::ProfileResultStatus::Stale as i32);
            assert!(old_binding.token.is_none());
            let old_page = client_b
                .client()
                .request(proto::GetProfileResultPage {
                    project_id,
                    import_id,
                    generation: retained.generation,
                    protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                    page_index: 1,
                })
                .await
                .expect("old snapshot native page returns typed denial");
            assert_eq!(old_page.status, proto::ProfileResultStatus::Stale as i32);
            assert!(
                old_page.frames.is_empty()
                    && old_page.nodes.is_empty()
                    && old_page.summary.is_none()
            );
            eprintln!(
                "native snapshot malformed={malformed_snapshot}: bindings=0 imports=0 binding={} page={}",
                old_binding.status, old_page.status
            );
            project_a
                .update(cx_a, |project, cx| {
                    project.reshared(
                        proto::ResharedProject {
                            id: project_id,
                            collaborators,
                        },
                        cx,
                    )
                })
                .expect("restore authoritative fixture collaborators");
            assert_eq!(client_a.peer_id(), Some(original_host_peer));
            assert_eq!(client_b.peer_id(), Some(original_member_peer));
            assert_eq!(client_c.peer_id(), Some(original_other_peer));
            let recovered = client_b
                .client()
                .request(prepare())
                .await
                .expect("fresh native binding recovers after snapshot replacement");
            assert_eq!(recovered.status, proto::ProfileResultStatus::Ready as i32);
            assert!(!recovered.complete);
            assert_eq!(recovered.token.as_deref().map(str::len), Some(64));
            let recovered_model = project_b
                .read_with(cx_b, |project, cx| {
                    project.import_remote_profile(
                        [profile_path.clone(), symbols_path.clone()],
                        true,
                        Arc::new(AtomicBool::new(false)),
                        cx,
                    )
                })
                .expect("fresh native import after restored snapshot")
                .await
                .expect("restored snapshot admits fresh consented model");
            assert_eq!(recovered_model.total_samples, 3_030);
        }
        client_c
            .client()
            .request(proto::CancelProfileTaskBinding {
                project_id,
                task_run_id: "member-verified-profile-task".into(),
            })
            .await
            .expect("another member cannot cancel the owner's binding");
        let replay = client_b
            .client()
            .request(prepare())
            .await
            .expect("original binding remains reserved after foreign cancellation");
        assert_eq!(replay.status, proto::ProfileResultStatus::Error as i32);
        assert!(replay.token.is_none());
        client_b
            .client()
            .request(proto::CancelProfileTaskBinding {
                project_id,
                task_run_id: "member-verified-profile-task".into(),
            })
            .await
            .expect("owner cancels native Task binding");
        let after_cancel = client_b
            .client()
            .request(proto::GetProfileTaskBinding {
                project_id,
                protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                task_run_id: "member-verified-profile-task".into(),
            })
            .await
            .expect("cancelled binding has typed host result");
        assert_eq!(
            after_cancel.status,
            proto::ProfileResultStatus::Stale as i32
        );
        assert!(after_cancel.token.is_none());

        let run_id = "member-completed-profile-task";
        let binding = project_b
            .read_with(cx_b, |project, cx| {
                project.prepare_remote_profile_task(
                    run_id.into(),
                    [profile_path.clone(), symbols_path.clone()],
                    cx,
                )
            })
            .expect("production Project client prepares the host Task")
            .await
            .expect("member receives an authoritative Task binding");
        assert_eq!(binding.token.len(), 64);
        let mut helper = smol::process::Command::new(&binding.helper_path)
            .args([
                "--exact",
                "channel_guest_tests::native_profile_relay_completion_helper",
                "--nocapture",
            ])
            .env("ZED_TEST_OBSERVER_SOCKET", binding.socket_path)
            .env(
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                binding.token,
            )
            .kill_on_drop(true)
            .spawn()
            .expect("spawn observed Task helper");
        let completion = project_b
            .read_with(cx_b, |project, cx| {
                project.get_remote_profile_task(run_id.into(), cx)
            })
            .expect("production Project client polls host completion");
        let deadline = executor.timer(std::time::Duration::from_secs(15));
        match futures::future::select(Box::pin(completion), Box::pin(deadline)).await {
            futures::future::Either::Left((result, _)) => {
                result.expect("owner receives host-observed Task completion")
            }
            futures::future::Either::Right(_) => panic!("collaboration Task completion timed out"),
        };
        let helper_exit = helper.status();
        let helper_deadline = executor.timer(std::time::Duration::from_secs(15));
        let helper_status =
            match futures::future::select(Box::pin(helper_exit), Box::pin(helper_deadline)).await {
                futures::future::Either::Left((result, _)) => result.expect("Task helper exit"),
                futures::future::Either::Right(_) => panic!("collaboration Task helper timed out"),
            };
        assert!(helper_status.success());
        let complete = client_b
            .client()
            .request(proto::GetProfileTaskBinding {
                project_id,
                protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                task_run_id: run_id.into(),
            })
            .await
            .expect("completed Task has a typed host status");
        assert_eq!(complete.status, proto::ProfileResultStatus::Ready as i32);
        assert!(complete.complete);
        assert!(complete.token.is_none() && complete.socket_path.is_none());
        let request = |import_id| proto::ImportProfileResults {
            project_id,
            import_id,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: Some(run_id.into()),
        };
        let foreign = client_c
            .client()
            .request(request(921))
            .await
            .expect("other member sees typed Task-bound denial");
        assert_eq!(foreign.status, proto::ProfileResultStatus::Stale as i32);
        assert!(foreign.frames.is_empty() && foreign.nodes.is_empty());
        let first = client_b
            .client()
            .request(request(922))
            .await
            .expect("owner receives Task-bound normalized profile page");
        assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
        assert!(!first.frames.is_empty());
        let model = project_b
            .read_with(cx_b, |project, cx| {
                project.import_remote_task_profile(
                    [profile_path.clone(), symbols_path.clone()],
                    run_id.into(),
                    Arc::new(AtomicBool::new(false)),
                    cx,
                )
            })
            .expect("collaboration Project accepts the host Task-bound import")
            .await
            .expect("production Project client assembles the Task-bound profile");
        assert_eq!(model.total_samples, 3_030);
        {
            let (workspace_b, window_cx_b) = client_b.active_workspace(cx_b);
            workspace_b.update_in(window_cx_b, |workspace, window, cx| {
                window.focus(&workspace.focus_handle(cx), cx);
            });
            window_cx_b.update(|window, cx| window.draw(cx).clear(cx));
            window_cx_b.dispatch_action(language_tools::profile_view::OpenNativeProfile);
            window_cx_b.run_until_parked();
            let view = workspace_b
                .read_with(window_cx_b, |workspace, cx| {
                    workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
                })
                .expect("profile action opens the collaboration viewer");
            view.update(window_cx_b, |view, cx| {
                view.import_paths_with_consent_for_test(
                    vec![
                        "/relay/profile.json".into(),
                        "/relay/profile.syms.json".into(),
                    ],
                    true,
                    cx,
                )
                .expect("explicit guest consent starts the host profile import");
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while view.read_with(window_cx_b, |view, _| view.sample_count_for_test()) != Some(3_030)
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "collaboration viewer did not display the host-normalized profile"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
        }
        let view = {
            let (workspace_b, window_cx_b) = client_b.active_workspace(cx_b);
            workspace_b
                .read_with(window_cx_b, |workspace, cx| {
                    workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
                })
                .expect("loaded collaboration profile viewer")
        };
        let member_rejoined = Arc::new(Mutex::new(false));
        let _rejoined_subscription = project_b.update(cx_b, |_, cx| {
            let member_rejoined = member_rejoined.clone();
            cx.subscribe(&project_b, move |_, _, event, _| {
                if matches!(event, project::Event::Rejoined) {
                    *member_rejoined
                        .lock()
                        .expect("profile member rejoin observation") = true;
                }
            })
        });
        let session_metadata =
            project_a.read_with(cx_a, |project, cx| project.worktree_metadata_protos(cx));
        client_a
            .client()
            .request(proto::UpdateProject {
                project_id,
                worktrees: session_metadata.clone(),
                upstream_session_generation: Some(0),
            })
            .await
            .expect("host establishes a known upstream generation");
        executor.run_until_parked();
        assert_eq!(
            view.read_with(cx_b, |view, _| view.sample_count_for_test()),
            None
        );
        view.update(cx_b, |view, cx| {
            view.import_paths_with_consent_for_test(
                vec![
                    "/relay/profile.json".into(),
                    "/relay/profile.syms.json".into(),
                ],
                true,
                cx,
            )
            .expect("fresh consented view for known generation");
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view.read_with(cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
            assert!(
                std::time::Instant::now() < deadline,
                "known-generation profile did not load"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
        for reconnect_host in [true, false] {
            assert_eq!(
                view.read_with(cx_b, |view, _| view.sample_count_for_test()),
                Some(3_030)
            );
            let original_host_peer = client_a.peer_id().expect("loaded profile host peer");
            let original_member_peer = client_b.peer_id().expect("loaded profile member peer");
            let original_other_peer = client_c.peer_id().expect("other loaded profile member");
            let previous_peer = if reconnect_host {
                *host_reshared
                    .lock()
                    .expect("loaded profile reshare observation") = false;
                original_host_peer
            } else {
                *member_rejoined
                    .lock()
                    .expect("profile member rejoin observation") = false;
                original_member_peer
            };
            server.disconnect_client(previous_peer);
            let step = std::time::Duration::from_millis(100);
            let mut remaining = rpc::RECEIVE_TIMEOUT;
            loop {
                executor.run_until_parked();
                let ready = if reconnect_host {
                    client_a.peer_id().is_some_and(|peer| {
                        peer != previous_peer
                            && *host_reshared
                                .lock()
                                .expect("loaded profile reshare observation")
                            && project_b.read_with(cx_b, |project, _| {
                                project
                                    .collaborators()
                                    .get(&peer)
                                    .is_some_and(|peer| peer.is_host)
                            })
                    })
                } else {
                    client_b.peer_id().is_some_and(|peer| {
                        peer != previous_peer
                            && *member_rejoined
                                .lock()
                                .expect("profile member rejoin observation")
                            && project_a.read_with(cx_a, |project, _| {
                                project.collaborators().contains_key(&peer)
                            })
                    })
                };
                if ready {
                    break;
                }
                assert!(
                    remaining >= step,
                    "loaded profile session did not reconnect"
                );
                remaining = remaining.saturating_sub(step);
                executor.advance_clock(step);
            }
            assert_eq!(client_c.peer_id(), Some(original_other_peer));
            if reconnect_host {
                assert_eq!(client_b.peer_id(), Some(original_member_peer));
            } else {
                assert_eq!(client_a.peer_id(), Some(original_host_peer));
            }
            view.read_with(cx_b, |view, _| {
                eprintln!("loaded native profile reconnect host={reconnect_host}: samples={:?}; notice={}", view.sample_count_for_test(), view.notice_for_test());
                assert_eq!(view.sample_count_for_test(), None, "reconnected session must clear the old native model");
                assert!(view.notice_for_test().contains("session"));
            });
            view.update(cx_b, |view, cx| {
                view.import_paths_with_consent_for_test(
                    vec![
                        "/relay/profile.json".into(),
                        "/relay/profile.syms.json".into(),
                    ],
                    true,
                    cx,
                )
                .expect("fresh consented profile import after reconnect");
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while view.read_with(cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "fresh profile did not recover after reconnect"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
            client_a
                .client()
                .request(proto::UpdateProject {
                    project_id,
                    worktrees: session_metadata.clone(),
                    upstream_session_generation: Some(0),
                })
                .await
                .expect("host repeats the numeric generation after reconnect");
            executor.run_until_parked();
            let expected_samples = if reconnect_host { None } else { Some(3_030) };
            assert_eq!(
                view.read_with(cx_b, |view, _| view.sample_count_for_test()),
                expected_samples,
                "a new host identity resets generation tracking; member-only rejoin preserves it"
            );
            eprintln!(
                "native numeric generation after reconnect host={reconnect_host}: samples={expected_samples:?}"
            );
            if reconnect_host {
                view.update(cx_b, |view, cx| {
                    view.import_paths_with_consent_for_test(
                        vec![
                            "/relay/profile.json".into(),
                            "/relay/profile.syms.json".into(),
                        ],
                        true,
                        cx,
                    )
                    .expect("fresh consent after new host generation");
                });
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while view.read_with(cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "new host generation profile did not recover"
                    );
                    executor.timer(std::time::Duration::from_millis(20)).await;
                }
            }
        }
        let original_member_peer = client_b
            .peer_id()
            .expect("loaded profile client before offline interval");
        let original_host_peer = client_a
            .peer_id()
            .expect("host before member offline interval");
        let original_other_peer = client_c
            .peer_id()
            .expect("other member before offline interval");
        assert_eq!(
            view.read_with(cx_b, |view, _| view.sample_count_for_test()),
            Some(3_030)
        );
        *member_rejoined
            .lock()
            .expect("offline member rejoin observation") = false;
        server.forbid_connections();
        server.disconnect_client(original_member_peer);
        let step = std::time::Duration::from_millis(100);
        let mut remaining = rpc::RECEIVE_TIMEOUT;
        loop {
            executor.run_until_parked();
            if !client_b.client().status().borrow().is_connected() {
                break;
            }
            assert!(
                remaining >= step,
                "profile member did not enter offline interval"
            );
            remaining = remaining.saturating_sub(step);
            executor.advance_clock(step);
            smol::future::yield_now().await;
        }
        assert!(
            !*member_rejoined
                .lock()
                .expect("offline member rejoin observation")
        );
        view.read_with(cx_b, |view, _| {
            eprintln!(
                "native profile offline interval: samples={:?}; notice={}",
                view.sample_count_for_test(),
                view.notice_for_test()
            );
            assert_eq!(
                view.sample_count_for_test(),
                None,
                "offline collaboration must clear the native model before rejoin"
            );
            assert!(view.notice_for_test().contains("session"));
        });
        server.allow_connections();
        let mut remaining = rpc::RECEIVE_TIMEOUT;
        loop {
            executor.run_until_parked();
            if client_b.peer_id().is_some_and(|peer| {
                peer != original_member_peer
                    && *member_rejoined
                        .lock()
                        .expect("offline member rejoin observation")
                    && project_a.read_with(cx_a, |project, _| {
                        project.collaborators().contains_key(&peer)
                    })
            }) {
                break;
            }
            assert!(remaining >= step, "offline profile member did not rejoin");
            remaining = remaining.saturating_sub(step);
            executor.advance_clock(step);
            smol::future::yield_now().await;
        }
        assert_eq!(client_a.peer_id(), Some(original_host_peer));
        assert_eq!(client_c.peer_id(), Some(original_other_peer));
        view.update(cx_b, |view, cx| {
            view.import_paths_with_consent_for_test(
                vec![
                    "/relay/profile.json".into(),
                    "/relay/profile.syms.json".into(),
                ],
                true,
                cx,
            )
            .expect("fresh consented import after offline interval");
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view.read_with(cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
            assert!(
                std::time::Instant::now() < deadline,
                "profile did not recover after offline interval"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
        for (pending_stage, reconnect_host) in [
            ("import", false),
            ("primary picker", false),
            ("companion picker", false),
            ("consent", false),
            ("primary picker", true),
            ("companion picker", true),
            ("consent", true),
        ] {
            use gpui::InputEvent as _;
            let original_member_peer = client_b.peer_id().expect("pending profile member");
            let original_host_peer = client_a.peer_id().expect("pending profile host");
            let original_other_peer = client_c.peer_id().expect("other pending profile member");
            assert_eq!(
                view.read_with(cx_b, |view, _| view.sample_count_for_test()),
                Some(3_030)
            );
            let mut import_release = None;
            let mut active_cancellation = None;
            let pending_selections = Arc::new(Mutex::new(Vec::new()));
            let selection_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            if pending_stage == "import" {
                let (started, started_receiver) = futures::channel::oneshot::channel();
                let (release, release_receiver) = futures::channel::oneshot::channel();
                profile_store.update(cx_a, |store, _| {
                    store.gate_next_import_for_test(started, release_receiver);
                });
                import_release = Some(release);
                active_cancellation = view.update(cx_b, |view, cx| {
                    view.import_paths_with_consent_for_test(
                        vec![
                            "/relay/profile.json".into(),
                            "/relay/profile.syms.json".into(),
                        ],
                        true,
                        cx,
                    )
                    .expect("replacement profile import before offline interval");
                    view.active_import_cancellation_for_test()
                });
                let deadline = executor.timer(std::time::Duration::from_secs(5));
                assert!(matches!(
                    futures::future::select(started_receiver, deadline).await,
                    futures::future::Either::Left((Ok(()), _))
                ));
                assert!(
                    !active_cancellation
                        .as_ref()
                        .expect("active viewer import")
                        .load(std::sync::atomic::Ordering::Relaxed)
                );
            } else {
                let (workspace, window_cx) = client_b.active_workspace(cx_b);
                workspace.update(window_cx, |workspace, _| {
                    let pending_selections = pending_selections.clone();
                    let selection_count = selection_count.clone();
                    workspace.set_prompt_for_open_path(Box::new(move |_, lister, _, cx| {
                        assert!(
                            !lister.is_local(cx),
                            "collaboration uses the project picker"
                        );
                        let selection =
                            selection_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let (sender, receiver) = futures::channel::oneshot::channel();
                        if pending_stage == "companion picker" && selection == 0 {
                            sender
                                .send(Some(vec!["/relay/profile.json".into()]))
                                .expect("first artifact selection");
                        } else if pending_stage == "consent" {
                            sender
                                .send(Some(vec![
                                    "/relay/profile.json".into(),
                                    "/relay/profile.syms.json".into(),
                                ]))
                                .expect("artifacts awaiting explicit consent");
                        } else {
                            pending_selections
                                .lock()
                                .expect("held picker selections")
                                .push(sender);
                        }
                        receiver
                    }));
                });
                workspace.update_in(window_cx, |workspace, window, cx| {
                    assert!(workspace.activate_item(&view, true, true, window, cx));
                    window.focus(&view.read(cx).focus_handle(cx), cx);
                });
                window_cx.update(|window, cx| {
                    for key in ["tab", "enter"] {
                        window.refresh();
                        window.draw(cx).clear(cx);
                        let keystroke = gpui::Keystroke::parse(key).expect("profile import key");
                        window.dispatch_event(
                            gpui::KeyDownEvent {
                                keystroke: keystroke.clone(),
                                is_held: false,
                                prefer_character_input: false,
                            }
                            .to_platform_input(),
                            cx,
                        );
                        window
                            .dispatch_event(gpui::KeyUpEvent { keystroke }.to_platform_input(), cx);
                    }
                });
                window_cx.run_until_parked();
                assert_eq!(
                    selection_count.load(std::sync::atomic::Ordering::Relaxed),
                    if pending_stage == "companion picker" {
                        2
                    } else {
                        1
                    }
                );
                assert!(view.read_with(window_cx, |view, _| view.is_selecting_for_test()));
                assert_eq!(window_cx.has_pending_prompt(), pending_stage == "consent");
            }
            *member_rejoined
                .lock()
                .expect("pending offline rejoin observation") = false;
            if reconnect_host {
                *host_reshared
                    .lock()
                    .expect("pending host reshare observation") = false;
                server.disconnect_client(original_host_peer);
            } else {
                server.forbid_connections();
                server.disconnect_client(original_member_peer);
            }
            let step = std::time::Duration::from_millis(100);
            let mut remaining = rpc::RECEIVE_TIMEOUT;
            loop {
                executor.run_until_parked();
                let disposed = if reconnect_host {
                    client_a.peer_id().is_some_and(|peer| {
                        peer != original_host_peer
                            && *host_reshared
                                .lock()
                                .expect("pending host reshare observation")
                            && project_b.read_with(cx_b, |project, _| {
                                project.host().is_some_and(|host| host.peer_id == peer)
                            })
                    })
                } else {
                    !client_b.client().status().borrow().is_connected()
                };
                if disposed {
                    break;
                }
                assert!(
                    remaining >= step,
                    "pending profile session did not reach its disposal boundary"
                );
                remaining = remaining.saturating_sub(step);
                executor.advance_clock(step);
                smol::future::yield_now().await;
            }
            assert!(
                !*member_rejoined
                    .lock()
                    .expect("pending offline rejoin observation")
            );
            view.read_with(cx_b, |view, _| {
                assert_eq!(
                    view.sample_count_for_test(),
                    None,
                    "{pending_stage} offline model"
                );
                assert!(!view.is_selecting_for_test());
                assert!(view.active_import_cancellation_for_test().is_none());
                assert!(view.notice_for_test().contains("session"));
            });
            if let Some(cancellation) = active_cancellation {
                assert!(cancellation.load(std::sync::atomic::Ordering::Relaxed));
            }
            if let Some(release) = import_release {
                match release.send(()) {
                    Ok(()) => eprintln!("released offline viewer import gate"),
                    Err(()) => eprintln!("offline viewer import gate already dropped"),
                }
            } else {
                let (_, window_cx) = client_b.active_workspace(cx_b);
                if pending_stage == "consent" {
                    window_cx.simulate_prompt_answer("Import Results");
                } else {
                    let selection = pending_selections
                        .lock()
                        .expect("late picker selection")
                        .pop()
                        .expect("held project picker callback");
                    let paths = if pending_stage == "companion picker" {
                        vec!["/relay/profile.syms.json".into()]
                    } else {
                        vec![
                            "/relay/profile.json".into(),
                            "/relay/profile.syms.json".into(),
                        ]
                    };
                    selection
                        .send(Some(paths))
                        .expect("late picker callback receiver");
                }
                window_cx.run_until_parked();
                assert!(
                    !window_cx.has_pending_prompt(),
                    "late selection cannot request new consent"
                );
            }
            executor.run_until_parked();
            view.read_with(cx_b, |view, _| {
                eprintln!(
                    "pending native profile stage={pending_stage} host_replacement={reconnect_host}: samples={:?}; notice={}",
                    view.sample_count_for_test(),
                    view.notice_for_test()
                );
                assert_eq!(
                    view.sample_count_for_test(),
                    None,
                    "late {pending_stage} cannot restore model"
                );
                assert!(!view.is_selecting_for_test());
                assert!(view.active_import_cancellation_for_test().is_none());
                assert!(view.notice_for_test().contains("session"));
            });
            server.allow_connections();
            let mut remaining = rpc::RECEIVE_TIMEOUT;
            loop {
                executor.run_until_parked();
                let recovered = if reconnect_host {
                    client_a.peer_id().is_some_and(|peer| {
                        peer != original_host_peer
                            && *host_reshared
                                .lock()
                                .expect("pending host reshare observation")
                            && project_b.read_with(cx_b, |project, _| {
                                project.host().is_some_and(|host| host.peer_id == peer)
                            })
                    })
                } else {
                    client_b.peer_id().is_some_and(|peer| {
                        peer != original_member_peer
                            && *member_rejoined
                                .lock()
                                .expect("pending offline rejoin observation")
                            && project_a.read_with(cx_a, |project, _| {
                                project.collaborators().contains_key(&peer)
                            })
                    })
                };
                if recovered {
                    break;
                }
                assert!(remaining >= step, "pending profile member did not rejoin");
                remaining = remaining.saturating_sub(step);
                executor.advance_clock(step);
                smol::future::yield_now().await;
            }
            if reconnect_host {
                assert_eq!(client_b.peer_id(), Some(original_member_peer));
            } else {
                assert_eq!(client_a.peer_id(), Some(original_host_peer));
            }
            assert_eq!(client_c.peer_id(), Some(original_other_peer));
            view.update(cx_b, |view, cx| {
                view.import_paths_with_consent_for_test(
                    vec![
                        "/relay/profile.json".into(),
                        "/relay/profile.syms.json".into(),
                    ],
                    true,
                    cx,
                )
                .expect("fresh consented import after pending offline cancellation");
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while view.read_with(cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "profile did not recover after {pending_stage} loss"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
        }
        let offline_host_peer = client_a.peer_id().expect("pending view host");
        let stable_member_peer = client_b.peer_id().expect("pending view member");
        let stable_other_peer = client_c.peer_id().expect("other pending view member");
        assert_eq!(
            view.read_with(cx_b, |view, _| view.sample_count_for_test()),
            Some(3_030)
        );
        let (host_import_started, host_import_started_receiver) =
            futures::channel::oneshot::channel();
        let (host_import_release, host_import_release_receiver) =
            futures::channel::oneshot::channel();
        profile_store.update(cx_a, |store, _| {
            store.gate_next_import_for_test(host_import_started, host_import_release_receiver);
        });
        let host_loss_cancellation = view.update(cx_b, |view, cx| {
            view.import_paths_with_consent_for_test(
                vec![
                    "/relay/profile.json".into(),
                    "/relay/profile.syms.json".into(),
                ],
                true,
                cx,
            )
            .expect("replacement view import before host loss");
            view.active_import_cancellation_for_test()
                .expect("pending host import cancellation")
        });
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(host_import_started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        assert!(!host_loss_cancellation.load(std::sync::atomic::Ordering::Relaxed));
        *host_reshared
            .lock()
            .expect("offline host reshare observation") = false;
        server.forbid_connections();
        server.disconnect_client(offline_host_peer);
        let step = std::time::Duration::from_millis(100);
        let mut remaining = rpc::RECEIVE_TIMEOUT;
        loop {
            executor.run_until_parked();
            if !client_a.client().status().borrow().is_connected()
                && host_loss_cancellation.load(std::sync::atomic::Ordering::Relaxed)
            {
                break;
            }
            assert!(
                remaining >= step,
                "host loss did not reach the pending viewer request"
            );
            remaining = remaining.saturating_sub(step);
            executor.advance_clock(step);
            smol::future::yield_now().await;
        }
        executor.timer(std::time::Duration::from_millis(20)).await;
        executor.run_until_parked();
        assert!(
            !*host_reshared
                .lock()
                .expect("offline host reshare observation")
        );
        assert_eq!(client_b.peer_id(), Some(stable_member_peer));
        assert_eq!(client_c.peer_id(), Some(stable_other_peer));
        assert!(client_b.client().status().borrow().is_connected());
        view.read_with(cx_b, |view, _| {
            eprintln!(
                "native pending host offline: samples={:?}; importing={}; notice={}",
                view.sample_count_for_test(),
                view.active_import_cancellation_for_test().is_some(),
                view.notice_for_test()
            );
            assert_eq!(
                view.sample_count_for_test(),
                None,
                "known profile host loss must clear the previous model before reshare"
            );
            assert!(
                view.active_import_cancellation_for_test().is_none(),
                "failed host import must become idle"
            );
            assert!(!view.is_selecting_for_test());
            assert!(view.notice_for_test().contains("session"));
        });
        match host_import_release.send(()) {
            Ok(()) => eprintln!("released offline host import gate"),
            Err(()) => eprintln!("offline host import gate already dropped"),
        }
        executor.run_until_parked();
        assert_eq!(
            view.read_with(cx_b, |view, _| view.sample_count_for_test()),
            None
        );
        server.allow_connections();
        let mut remaining = rpc::RECEIVE_TIMEOUT;
        loop {
            executor.run_until_parked();
            if client_a.peer_id().is_some_and(|peer| {
                peer != offline_host_peer
                    && *host_reshared
                        .lock()
                        .expect("offline host reshare observation")
                    && project_b.read_with(cx_b, |project, _| {
                        project.host().is_some_and(|host| host.peer_id == peer)
                    })
            }) {
                break;
            }
            assert!(
                remaining >= step,
                "profile host did not rejoin after pending view disposal"
            );
            remaining = remaining.saturating_sub(step);
            executor.advance_clock(step);
            smol::future::yield_now().await;
        }
        assert_eq!(client_b.peer_id(), Some(stable_member_peer));
        assert_eq!(client_c.peer_id(), Some(stable_other_peer));
        view.update(cx_b, |view, cx| {
            view.import_paths_with_consent_for_test(
                vec![
                    "/relay/profile.json".into(),
                    "/relay/profile.syms.json".into(),
                ],
                true,
                cx,
            )
            .expect("fresh consented viewer import after host loss");
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while view.read_with(cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
            assert!(
                std::time::Instant::now() < deadline,
                "viewer did not recover after host loss"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
        client_b
            .client()
            .request(proto::CancelProfileImport {
                project_id,
                import_id: 922,
                generation: first.generation,
            })
            .await
            .expect("owner drops retained Task-bound import");
        project_b
            .read_with(cx_b, |project, cx| {
                project.cancel_remote_profile_task(run_id.into(), cx)
            })
            .expect("production Project client cancels the completed binding")
            .await
            .expect("owner releases completed Task binding");
        let removed = client_b
            .client()
            .request(proto::GetProfileTaskBinding {
                project_id,
                protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                task_run_id: run_id.into(),
            })
            .await
            .expect("cancelled completed binding has a typed host result");
        assert_eq!(removed.status, proto::ProfileResultStatus::Stale as i32);
        for (page_request, reconnect_host, import_id) in [
            (false, true, 10_001),
            (false, false, 10_002),
            (true, true, 10_003),
            (true, false, 10_004),
        ] {
            let request = || proto::ImportProfileResults {
                project_id,
                import_id,
                protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                profile_path: Some(profile_path.to_proto()),
                symbols_path: Some(symbols_path.to_proto()),
                consent_to_transfer: true,
                artifact_format: "samply-0.13.1".into(),
                task_run_id: None,
            };
            let generation = if page_request {
                let first = client_b
                    .client()
                    .request(request())
                    .await
                    .expect("prepare retained profile page");
                assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
                assert!(!first.complete);
                first.generation
            } else {
                0
            };
            let (started, started_receiver) = futures::channel::oneshot::channel();
            let (release, release_receiver) = futures::channel::oneshot::channel();
            profile_store.update(cx_a, |store, _| {
                if page_request {
                    store.gate_next_page_for_test(started, release_receiver);
                } else {
                    store.gate_next_import_for_test(started, release_receiver);
                }
            });
            let pending = async {
                if page_request {
                    client_b
                        .client()
                        .request(proto::GetProfileResultPage {
                            project_id,
                            import_id,
                            generation,
                            protocol_version:
                                project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                            page_index: 1,
                        })
                        .await
                } else {
                    client_b.client().request(request()).await
                }
            };
            let reconnect = async {
                let deadline = executor.timer(std::time::Duration::from_secs(5));
                assert!(matches!(
                    futures::future::select(started_receiver, deadline).await,
                    futures::future::Either::Left((Ok(()), _))
                ));
                let original_host_peer = client_a.peer_id().expect("admitted profile host peer");
                let original_member_peer =
                    client_b.peer_id().expect("admitted profile member peer");
                let original_other_peer =
                    client_c.peer_id().expect("other admitted profile member");
                let cancellation = profile_store
                    .read_with(cx_a, |store, _| {
                        store.import_cancellation_for_test(original_member_peer, import_id)
                    })
                    .expect("host retained admitted import cancellation");
                assert!(!cancellation.load(std::sync::atomic::Ordering::Relaxed));
                let previous_peer = if reconnect_host {
                    *host_reshared
                        .lock()
                        .expect("admitted profile reshare observation") = false;
                    original_host_peer
                } else {
                    *member_rejoined
                        .lock()
                        .expect("admitted profile rejoin observation") = false;
                    original_member_peer
                };
                server.disconnect_client(previous_peer);
                let step = std::time::Duration::from_millis(100);
                let mut remaining = rpc::RECEIVE_TIMEOUT;
                loop {
                    executor.run_until_parked();
                    let ready = if reconnect_host {
                        client_a.peer_id().is_some_and(|peer| {
                            peer != previous_peer
                                && *host_reshared
                                    .lock()
                                    .expect("admitted profile reshare observation")
                                && project_b.read_with(cx_b, |project, _| {
                                    project
                                        .collaborators()
                                        .get(&peer)
                                        .is_some_and(|collaborator| collaborator.is_host)
                                })
                        })
                    } else {
                        client_b.peer_id().is_some_and(|peer| {
                            peer != previous_peer
                                && *member_rejoined
                                    .lock()
                                    .expect("admitted profile rejoin observation")
                                && project_a.read_with(cx_a, |project, _| {
                                    project.collaborators().contains_key(&peer)
                                })
                        })
                    };
                    if ready {
                        break;
                    }
                    assert!(remaining >= step, "admitted native route did not reconnect");
                    remaining = remaining.saturating_sub(step);
                    executor.advance_clock(step);
                    // Let the sibling request release its response barrier before further clock steps.
                    smol::future::yield_now().await;
                }
                assert_eq!(client_c.peer_id(), Some(original_other_peer));
                if reconnect_host {
                    assert_eq!(client_b.peer_id(), Some(original_member_peer));
                } else {
                    assert_eq!(client_a.peer_id(), Some(original_host_peer));
                }
                assert!(
                    cancellation.load(std::sync::atomic::Ordering::Relaxed),
                    "admitted native processing must receive cancellation"
                );
                assert_eq!(
                    profile_store.read_with(cx_a, |store, _| store
                        .active_import_count_for_peer_for_test(original_member_peer)),
                    0
                );
                match release.send(()) {
                    Ok(()) => eprintln!(
                        "released admitted native gate page={page_request} host={reconnect_host}"
                    ),
                    Err(()) => eprintln!(
                        "admitted native gate already dropped page={page_request} host={reconnect_host}"
                    ),
                }
            };
            let (response, ()) = futures::join!(pending, reconnect);
            let error = response.expect_err("obsolete native RPC cannot return profile payload");
            eprintln!(
                "admitted native reconnect page={page_request} host={reconnect_host}: {error:#}"
            );
            let model = project_b
                .read_with(cx_b, |project, cx| {
                    project.import_remote_profile(
                        [profile_path.clone(), symbols_path.clone()],
                        true,
                        Arc::new(AtomicBool::new(false)),
                        cx,
                    )
                })
                .expect("fresh consented native request after reconnect")
                .await
                .expect("native client recovers after admitted RPC loss");
            assert_eq!(model.total_samples, 3_030);
        }
        cx_a.run_until_parked();
        executor.forbid_parking();
        cx_a.executor().forbid_parking();
        cx_b.executor().forbid_parking();
        cx_c.executor().forbid_parking();
    }
    let first = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 913,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("owner profile import reaches host");
    assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
    assert!(
        !first.complete,
        "the fixture must retain more than one page"
    );
    let page = |page_index| proto::GetProfileResultPage {
        project_id,
        import_id: 913,
        generation: first.generation,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        page_index,
    };
    let foreign = client_b
        .client()
        .request(page(1))
        .await
        .expect("other member receives a typed denial");
    assert_eq!(foreign.status, proto::ProfileResultStatus::Stale as i32);
    assert!(foreign.frames.is_empty() && foreign.nodes.is_empty() && foreign.summary.is_none());
    client_b
        .client()
        .request(proto::CancelProfileImport {
            project_id,
            import_id: 913,
            generation: first.generation,
        })
        .await
        .expect("other member cancellation is acknowledged");
    let owner = client_c
        .client()
        .request(page(1))
        .await
        .expect("owner retains the next page");
    assert_eq!(owner.status, proto::ProfileResultStatus::Ready as i32);
    assert!(!owner.complete, "the fixture must retain a later page");
    client_c
        .client()
        .request(proto::CancelProfileImport {
            project_id,
            import_id: 913,
            generation: first.generation,
        })
        .await
        .expect("owner cancellation is acknowledged");
    let cancelled = client_c
        .client()
        .request(page(2))
        .await
        .expect("owner sees a typed cancelled stream");
    assert_eq!(cancelled.status, proto::ProfileResultStatus::Stale as i32);

    let first = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 917,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("owner starts complete profile transfer through relay");
    let generation = first.generation;
    let mut assembler = ProfilePageAssembler::new(917, generation).expect("stream identity");
    let cancel = AtomicBool::new(false);
    assert!(first.encoded_len() <= MAX_PROFILE_PAGE_BYTES);
    assembler
        .push(first, &cancel, |_| true)
        .expect("first bounded relay page");
    let mut next_page = 1;
    loop {
        assert!(next_page < 16, "bounded fixture page count");
        let page = client_c
            .client()
            .request(proto::GetProfileResultPage {
                project_id,
                import_id: 917,
                generation,
                protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
                page_index: next_page,
            })
            .await
            .expect("owner receives retained relay page");
        assert!(page.encoded_len() <= MAX_PROFILE_PAGE_BYTES);
        let complete = page.complete;
        assembler
            .push(page, &cancel, |_| true)
            .expect("ordered bounded relay page");
        next_page += 1;
        if complete {
            break;
        }
    }
    assert!(next_page >= 3);
    let model = assembler
        .finish(&cancel, |_| true)
        .expect("complete normalized relay model");
    assert_eq!(model.total_samples, 3_030);
    assert!(model.frames.iter().any(|frame| frame.source.is_some()));

    let retained = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 919,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("member starts a retained native profile transfer");
    assert_eq!(retained.status, proto::ProfileResultStatus::Ready as i32);
    assert!(!retained.complete, "profile must have a retained next page");
    cx_c.read(ActiveCall::global)
        .update(cx_c, |call, cx| call.hang_up(cx))
        .await
        .expect("profile owner leaves channel");
    executor.run_until_parked();
    cx_c.update(|cx| {
        workspace::join_channel(channel_id, client_c.app_state.clone(), None, None, cx)
    })
    .await
    .expect("profile owner rejoins channel");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_c.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore profile owner edit capability");
    executor.run_until_parked();
    let old_page = client_c
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 919,
            generation: retained.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("old stream returns a typed status after rejoin");
    assert_eq!(
        old_page.status,
        proto::ProfileResultStatus::Stale as i32,
        "leaving must revoke the previous member's retained profile pages"
    );
    assert!(old_page.frames.is_empty() && old_page.nodes.is_empty());
    let fresh = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 920,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("new profile transfer after rejoin");
    assert_eq!(fresh.status, proto::ProfileResultStatus::Ready as i32);

    let profile_store = project_a.read_with(cx_a, |project, _| {
        project
            .profile_results_store_for_test()
            .expect("local host profile store")
            .clone()
    });
    let (import_started, import_started_receiver) = futures::channel::oneshot::channel();
    let (release_import, release_import_receiver) = futures::channel::oneshot::channel();
    profile_store.update(cx_a, |store, _| {
        store.gate_next_import_for_test(import_started, release_import_receiver)
    });
    let pending_import = client_c.client().request(proto::ImportProfileResults {
        project_id,
        import_id: 921,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        profile_path: Some(profile_path.to_proto()),
        symbols_path: Some(symbols_path.to_proto()),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    });
    let leave_and_rejoin = async {
        import_started_receiver
            .await
            .expect("authenticated host import entered parsing");
        cx_c.read(ActiveCall::global)
            .update(cx_c, |call, cx| call.hang_up(cx))
            .await
            .expect("profile importer leaves during parsing");
        executor.run_until_parked();
        cx_c.update(|cx| {
            workspace::join_channel(channel_id, client_c.app_state.clone(), None, None, cx)
        })
        .await
        .expect("profile importer rejoins during parsing");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_c.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore profile importer edit capability");
        executor.run_until_parked();
        release_import
            .send(())
            .expect("release host import after membership changed");
    };
    let (old_import, ()) = futures::join!(pending_import, leave_and_rejoin);
    let error = old_import.expect_err("prior member session cannot receive parsed profile");
    assert!(
        error
            .to_string()
            .contains("project request authorization changed while awaiting host response"),
        "{error:#}"
    );
    let new_import = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 922,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("fresh member session can import a profile");
    assert_eq!(new_import.status, proto::ProfileResultStatus::Ready as i32);

    let (import_started, import_started_receiver) = futures::channel::oneshot::channel();
    let (release_import, release_import_receiver) = futures::channel::oneshot::channel();
    profile_store.update(cx_a, |store, _| {
        store.gate_next_import_for_test(import_started, release_import_receiver)
    });
    let pending_import = client_c.client().request(proto::ImportProfileResults {
        project_id,
        import_id: 923,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        profile_path: Some(profile_path.to_proto()),
        symbols_path: Some(symbols_path.to_proto()),
        consent_to_transfer: true,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    });
    let demote = async {
        import_started_receiver
            .await
            .expect("host profile import entered parsing before role revocation");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_c.user_id().expect("member identity"),
                        proto::ChannelRole::Guest,
                        cx,
                    )
                })
            })
            .await
            .expect("revoke profile importer edit capability while parsing");
        executor.run_until_parked();
        release_import
            .send(())
            .expect("host profile import remains pending after role revocation");
    };
    let (response, ()) = futures::join!(pending_import, demote);
    let error = response.expect_err("demoted member cannot receive a pending profile import");
    assert!(
        error
            .to_string()
            .contains("not authorized to edit projects"),
        "{error:#}"
    );
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_c.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore profile importer role after stale reply");
    executor.run_until_parked();
    let restored_import = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 924,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("restored member starts a fresh profile import");
    assert_eq!(
        restored_import.status,
        proto::ProfileResultStatus::Ready as i32
    );

    assert!(!restored_import.complete);
    let unrelated_import = client_b
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 925,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("another member retains an independent native import");
    assert_eq!(
        unrelated_import.status,
        proto::ProfileResultStatus::Ready as i32
    );
    assert!(!unrelated_import.complete);
    let owner_peer = client_c.peer_id().expect("native page owner");
    let owner_cancellation = profile_store
        .read_with(cx_a, |store, _| {
            store.import_cancellation_for_test(owner_peer, 924)
        })
        .expect("retained native page cancellation");
    let (page_started, page_started_receiver) = futures::channel::oneshot::channel();
    let (page_release, page_release_receiver) = futures::channel::oneshot::channel();
    profile_store.update(cx_a, |store, _| {
        store.gate_next_page_for_test(page_started, page_release_receiver);
    });
    let pending_page = client_c.client().request(proto::GetProfileResultPage {
        project_id,
        import_id: 924,
        generation: restored_import.generation,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        page_index: 1,
    });
    let restore_role = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(page_started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        for role in [proto::ChannelRole::Guest, proto::ChannelRole::Member] {
            active_call_a
                .update(cx_a, |call, cx| {
                    call.room().expect("host room").update(cx, |room, cx| {
                        room.set_participant_role(
                            client_c.user_id().expect("native page owner identity"),
                            role,
                            cx,
                        )
                    })
                })
                .await
                .expect("cycle native page owner role while its response is pending");
            executor.run_until_parked();
        }
        assert_eq!(client_c.peer_id(), Some(owner_peer));
        page_release
            .send(())
            .expect("release role-restored native page");
    };
    let (response, ()) = futures::join!(pending_page, restore_role);
    let error = response.expect_err("role restoration cannot revive an admitted native page");
    assert!(
        error
            .to_string()
            .contains("authorization changed while awaiting host response"),
        "{error:#}"
    );
    executor.run_until_parked();
    assert!(owner_cancellation.load(std::sync::atomic::Ordering::Relaxed));
    let obsolete_page = client_c
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 924,
            generation: restored_import.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("cancelled role-restored native page has a typed status");
    assert_eq!(
        obsolete_page.status,
        proto::ProfileResultStatus::Stale as i32
    );
    assert!(obsolete_page.frames.is_empty() && obsolete_page.nodes.is_empty());
    let other_page = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 925,
            generation: unrelated_import.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("role-restored cancellation preserves another member's native import");
    assert_eq!(other_page.status, proto::ProfileResultStatus::Ready as i32);
    let recovered_import = client_c
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id: 926,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("fresh native import recovers after role restoration");
    assert_eq!(
        recovered_import.status,
        proto::ProfileResultStatus::Ready as i32
    );

    let structured_request = || proto::GetStructuredExecutionSnapshot {
        project_id,
        provider_id: structured_provider.0.clone(),
        discovery_generation: 1,
        page_start: 0,
        page_size: 10,
        worktree_ids: vec![path.worktree_id.to_proto()],
    };
    let baseline = client_b
        .client()
        .request(structured_request())
        .await
        .expect("member reads visible structured source before revocation");
    assert!(
        baseline
            .nodes
            .iter()
            .any(|node| node.label == "public case" && node.path.is_some())
    );
    let structured_store = project_a.read_with(cx_a, |project, _| {
        project.structured_execution_store().clone()
    });
    let (assembled, release, completed) = structured_store.update(cx_a, |store, _| {
        store.gate_next_assembled_response_for_test()
    });
    let pending = client_b.client().request(structured_request());
    let revoke_source = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_a.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/source.rs".into()]));
                });
            });
        });
        executor.run_until_parked();
        project_a.read_with(cx_a, |project, cx| {
            assert!(
                project
                    .entry_for_path(&path, cx)
                    .is_some_and(|entry| entry.is_private),
                "the source must be private before releasing the assembled reply"
            );
        });
        release
            .send(())
            .expect("assembled structured response remains pending through revocation");
        completed
            .await
            .expect("host rechecked the assembled structured response");
    };
    let (response, ()) = futures::join!(pending, revoke_source);
    let delayed = response.expect("member receives a revalidated structured page");
    assert!(
        delayed.nodes.iter().all(|node| node.label != "public case"),
        "an assembled response must not reveal a newly private source"
    );
    assert!(delayed.partial);
    let current = client_b
        .client()
        .request(structured_request())
        .await
        .expect("member receives the current filtered structured page");
    assert!(current.nodes.iter().all(|node| node.label != "public case"));
    // Artifact refresh may start passive discovery after the cancellation assertions.
    discovery_releases
        .lock()
        .expect("discovery release lock")
        .clear();
    executor.run_until_parked();

    cx_c.read(ActiveCall::global)
        .update(cx_c, |call, cx| call.hang_up(cx))
        .await
        .expect("member leaves before unauthorized route check");
    executor.run_until_parked();
    let denied_read = client_c
        .client()
        .request(proto::GetStructuredExecutionSnapshot {
            project_id,
            provider_id: structured_provider.0.clone(),
            discovery_generation: 1,
            page_start: 0,
            page_size: 10,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect_err("a signed-in nonmember cannot read host structured results");
    assert!(
        denied_read
            .to_string()
            .contains("not authorized to read projects")
    );
    let denied_mutation = client_c
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 923,
            worktree_ids: vec![path.worktree_id.to_proto()],
        })
        .await
        .expect_err("a signed-in nonmember cannot run host Cargo metadata");
    assert!(
        denied_mutation
            .to_string()
            .contains("not authorized to edit projects")
    );

    macro_rules! assert_nonmember_denied {
        ($message:ident, $permission:literal) => {{
            let error = client_c
                .client()
                .request(proto::$message {
                    project_id,
                    ..Default::default()
                })
                .await
                .expect_err(concat!(
                    "nonmember route must reject ",
                    stringify!($message)
                ));
            assert!(
                error.to_string().contains($permission),
                "{}: {error:#}",
                stringify!($message)
            );
        }};
    }

    assert_nonmember_denied!(
        GetStructuredExecutionEvents,
        "not authorized to read projects"
    );
    assert_nonmember_denied!(GetSourceCoverage, "not authorized to read projects");
    assert_nonmember_denied!(CancelCargoWorkspace, "not authorized to edit projects");
    assert_nonmember_denied!(GetRustTestDiscovery, "not authorized to edit projects");
    assert_nonmember_denied!(CancelRustTestDiscovery, "not authorized to edit projects");
    assert_nonmember_denied!(ResolveRustTestAction, "not authorized to edit projects");
    assert_nonmember_denied!(UpdateRustTestRun, "not authorized to edit projects");
    assert_nonmember_denied!(
        InterpretRustCoverageArtifact,
        "not authorized to edit projects"
    );
    assert_nonmember_denied!(PrepareProfileTaskBinding, "not authorized to edit projects");
    assert_nonmember_denied!(GetProfileTaskBinding, "not authorized to edit projects");
    assert_nonmember_denied!(CancelProfileTaskBinding, "not authorized to edit projects");
    assert_nonmember_denied!(ImportProfileResults, "not authorized to edit projects");
    assert_nonmember_denied!(GetProfileResultPage, "not authorized to edit projects");
    assert_nonmember_denied!(CancelProfileImport, "not authorized to edit projects");
}

#[gpui::test]
#[cfg(feature = "rust-tools-test")]
async fn rust_tools_relay_reports_missing_host_capabilities(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    use std::any::TypeId;

    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    cx_a.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    cx_b.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        );
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        )
    });
    let channel_id = server
        .make_public_channel("rust-tools-missing-host", &client_a, cx_a)
        .await;
    client_a
        .fs()
        .insert_tree(
            "/relay",
            serde_json::json!({
                "Cargo.toml": "[package]\nname = \"relay\"\nversion = \"0.1.0\"\n",
                "profile.json": "{}",
                "profile.syms.json": "{}",
                "src": {"lib.rs": "pub fn example() {}\n"},
            }),
        )
        .await;
    client_a
        .fs()
        .insert_file("/relay/profile.json", relay_paged_profile())
        .await;
    client_a
        .fs()
        .insert_file(
            "/relay/profile.syms.json",
            include_str!("../../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json")
                .replace("/fixture/", "/relay/")
                .into_bytes(),
        )
        .await;
    let project_a = client_a
        .build_local_project_with_trust("/relay", cx_a)
        .await
        .0;
    let active_call_a = cx_a.read(ActiveCall::global);
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .expect("host joins channel");
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("host shares project");
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .expect("member joins channel");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("promote member");
    executor.run_until_parked();

    let (workspace_b, cx_b) = client_b.active_workspace(cx_b);
    let project_b = workspace_b.read_with(cx_b, |workspace, _| workspace.project().clone());
    assert_eq!(
        project_b.read_with(cx_b, |project, _| project.remote_id()),
        Some(project_id)
    );

    let event_provider = StructuredProviderId("missing-events".into());
    let event_path = project_a.read_with(cx_a, |project, cx| {
        project
            .find_project_path("/relay/src/lib.rs", cx)
            .expect("scanned event source")
    });
    project_a
        .read_with(cx_a, |project, _| {
            project.structured_execution_store().clone()
        })
        .update(cx_a, |store, _| {
            let project_generation = store.state().project_generation();
            store
                .state_mut()
                .apply_discovery(
                    project_generation,
                    StructuredProviderSnapshot::discovery(
                        event_provider.clone(),
                        DiscoveryGeneration(1),
                        StructuredProviderStatus::Current,
                        vec![StructuredNode {
                            experimental_feature: None,
                            experimental_feature_enabled: None,
                            debug_capability: None,
                            id: StructuredNodeId("case".into()),
                            parent_id: None,
                            label: "host case".into(),
                            kind: StructuredNodeKind::Case,
                            path: Some(event_path.clone()),
                        }],
                    ),
                    None,
                )
                .expect("publish host structured snapshot");
            let run_id = StructuredRunId("missing-events-run".into());
            store
                .state_mut()
                .begin_run(
                    project_generation,
                    &event_provider,
                    StructuredRun::new(
                        run_id.clone(),
                        DiscoveryGeneration(1),
                        vec![StructuredNodeId("case".into())],
                    ),
                )
                .expect("publish host structured run");
            store
                .state_mut()
                .apply_event(
                    project_generation,
                    &event_provider,
                    &run_id,
                    StructuredExecutionEvent {
                        sequence: 0,
                        node_id: StructuredNodeId("case".into()),
                        state: StructuredNodeState::Passed,
                        duration_millis: None,
                        message: None,
                        location: Some(event_path),
                    },
                    None,
                )
                .expect("publish host structured event");
        });
    {
        let host_handlers = client_a.client().message_handler_set();
        assert!(
            host_handlers
                .lock()
                .message_handlers
                .remove(&TypeId::of::<proto::GetStructuredExecutionEvents>())
                .is_some(),
            "the enabled host must register structured events before removal"
        );
    }
    let structured = project_b.read_with(cx_b, |project, _| {
        project.structured_execution_store().clone()
    });
    let error = structured
        .update(cx_b, |store, cx| {
            store.refresh_provider(event_provider.clone(), cx)
        })
        .await
        .expect_err("missing host structured event handler");
    assert!(
        error
            .downcast_ref::<project::structured_execution::StructuredExecutionUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );
    assert!(
        structured.read_with(cx_b, |store, _| store
            .state()
            .provider(&event_provider)
            .is_none()),
        "a missing event handler must not publish a partial host snapshot"
    );

    let page_profile_paths = project_b.read_with(cx_b, |project, cx| {
        [
            project
                .find_project_path("/relay/profile.json", cx)
                .expect("scanned profile artifact"),
            project
                .find_project_path("/relay/profile.syms.json", cx)
                .expect("scanned symbols artifact"),
        ]
    });
    {
        let host_handlers = client_a.client().message_handler_set();
        assert!(
            host_handlers
                .lock()
                .message_handlers
                .remove(&TypeId::of::<proto::GetProfileResultPage>())
                .is_some(),
            "the enabled host must register profile paging before removal"
        );
    }
    let error = project_b
        .read_with(cx_b, |project, cx| {
            project.import_remote_profile(
                page_profile_paths,
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("native import reaches the collaboration host")
        .await
        .expect_err("missing host profile page handler");
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );

    {
        let host_handlers = client_a.client().message_handler_set();
        let mut host_handlers = host_handlers.lock();
        for message_type in [
            TypeId::of::<proto::GetCargoWorkspace>(),
            TypeId::of::<proto::GetRustTestDiscovery>(),
            TypeId::of::<proto::ResolveRustTestAction>(),
            TypeId::of::<proto::GetStructuredExecutionSnapshot>(),
            TypeId::of::<proto::GetSourceCoverage>(),
            TypeId::of::<proto::InterpretRustCoverageArtifact>(),
            TypeId::of::<proto::PrepareProfileTaskBinding>(),
            TypeId::of::<proto::GetProfileTaskBinding>(),
            TypeId::of::<proto::CancelProfileTaskBinding>(),
            TypeId::of::<proto::ImportProfileResults>(),
        ] {
            assert!(
                host_handlers
                    .message_handlers
                    .remove(&message_type)
                    .is_some(),
                "the enabled host must register the handler before it is removed"
            );
        }
    }

    let cargo = project_b.read_with(cx_b, |project, _| project.cargo_workspace_store().clone());
    let error = cargo
        .update(cx_b, |store, cx| store.refresh(cx))
        .await
        .expect_err("missing host Cargo handler");
    assert_eq!(
        error
            .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
            .expect("typed Cargo capability error")
            .kind,
        project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::UnsupportedHost
    );

    let tests = project_b.read_with(cx_b, |project, _| {
        project.rust_test_provider_store().clone()
    });
    let error = tests
        .update(cx_b, |store, cx| store.refresh(cx))
        .await
        .expect_err("missing host Rust discovery handler");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed Rust discovery capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedDiscoveryHost
    );
    let error = tests
        .update(cx_b, |store, cx| {
            store.plan_action(
                &project::structured_execution::StructuredNodeId("missing-host-case".into()),
                project::structured_execution::DiscoveryGeneration(1),
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect_err("missing host Rust action handler");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed Rust action capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedActionHost
    );
    rpc::AnyProtoClient::from(client_a.client().clone()).add_entity_request_handler(
        |_store: gpui::Entity<RustTestProviderStore>,
         _request: rpc::TypedEnvelope<proto::ResolveRustTestAction>,
         _cx: gpui::AsyncApp| async {
            Ok(proto::RustTestActionPlanResponse {
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                plan_json: "{}".into(),
                authorization_generation: 0,
            })
        },
    );
    let error = tests
        .update(cx_b, |store, cx| {
            store.plan_action(
                &StructuredNodeId("legacy-host-case".into()),
                DiscoveryGeneration(1),
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect_err("legacy host cannot issue an unbound action through the current relay");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed legacy Rust action capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedActionHost
    );

    let structured = project_b.read_with(cx_b, |project, _| {
        project.structured_execution_store().clone()
    });
    let error = structured
        .update(cx_b, |store, cx| {
            store.refresh_provider(
                project::structured_execution::StructuredProviderId("missing-host".into()),
                cx,
            )
        })
        .await
        .expect_err("missing host structured-results handler");
    assert!(
        error
            .downcast_ref::<project::structured_execution::StructuredExecutionUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );

    let profile_paths = project_b.read_with(cx_b, |project, cx| {
        [
            project
                .find_project_path("/relay/profile.json", cx)
                .expect("scanned profile artifact"),
            project
                .find_project_path("/relay/profile.syms.json", cx)
                .expect("scanned symbols artifact"),
        ]
    });
    let rust_coverage = project_b.read_with(cx_b, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let error = rust_coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(profile_paths[0].clone(), 1, cx)
        })
        .await
        .expect_err("missing host Rust coverage artifact handler");
    assert!(
        error
            .downcast_ref::<project::source_coverage::SourceCoverageUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );
    let error = project_b
        .read_with(cx_b, |project, cx| {
            project.prepare_remote_profile_task(
                "missing-host-profile-task".into(),
                profile_paths.clone(),
                cx,
            )
        })
        .expect("profile Task preparation reaches the collaboration host")
        .await
        .err()
        .expect("missing host profile Task preparation handler");
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );
    let error = project_b
        .read_with(cx_b, |project, cx| {
            project.get_remote_profile_task("missing-host-profile-task".into(), cx)
        })
        .expect("profile Task status reaches the collaboration host")
        .await
        .expect_err("missing host profile Task status handler");
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );
    let error = project_b
        .read_with(cx_b, |project, cx| {
            project.cancel_remote_profile_task("missing-host-profile-task".into(), cx)
        })
        .expect("profile Task cancellation reaches the collaboration host")
        .await
        .expect_err("missing host profile Task cancellation handler");
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );
    let error = project_b
        .read_with(cx_b, |project, cx| {
            project.import_remote_profile(
                profile_paths,
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("native import reaches the collaboration host")
        .await
        .expect_err("missing host native profile handler");
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );

    let coverage = project_b.read_with(cx_b, |project, _| project.source_coverage_store().clone());
    let error = coverage
        .update(cx_b, |store, cx| {
            store.snapshot(SourceCoverageProviderId("missing-host".into()), cx)
        })
        .await
        .expect_err("missing host coverage handler");
    assert!(
        error
            .downcast_ref::<project::source_coverage::SourceCoverageUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_enabled_host_imports_consented_profile(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::NativeProfile,
    )
    .await;
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_enabled_host_coverage_reply_filters(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::CoverageVisibility,
    )
    .await;
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_legacy_host_rejects_unbound_actions(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::LegacyAuthorization,
    )
    .await;
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_enabled_host_rejects_actions_after_project_leave(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::ActionOwnerLeave,
    )
    .await;
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_enabled_host_retires_actual_cargo_discovery(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::DiscoveryCancellation,
    )
    .await;
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_enabled_host_keeps_native_task_host_owned(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::NativeTaskOwnership,
    )
    .await;
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_enabled_host_coverage_reply_rejects_replaced_sessions(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    run_separate_enabled_host_tools(
        executor,
        cx_a,
        cx_b,
        cx_c,
        SeparateEnabledHostScenario::CoverageSession,
    )
    .await;
}

#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
#[derive(PartialEq)]
enum SeparateEnabledHostScenario {
    NativeProfile,
    CoverageVisibility,
    CoverageSession,
    LegacyAuthorization,
    ActionOwnerLeave,
    DiscoveryCancellation,
    NativeTaskOwnership,
}

#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn run_separate_enabled_host_tools(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
    scenario: SeparateEnabledHostScenario,
) {
    use remote::{
        ConnectionIdentifier, MockDelegate, ProcessRemoteConnection, RemoteClient, RemoteConnection,
    };
    use std::path::PathBuf;

    let needs_isolated_build_output = matches!(
        scenario,
        SeparateEnabledHostScenario::ActionOwnerLeave
            | SeparateEnabledHostScenario::DiscoveryCancellation
            | SeparateEnabledHostScenario::NativeTaskOwnership
            | SeparateEnabledHostScenario::CoverageSession
    );
    let discovery_cancellation_gate =
        matches!(scenario, SeparateEnabledHostScenario::DiscoveryCancellation).then(|| {
            tempfile::tempdir_in("/private/tmp").expect("create owned discovery compiler gate")
        });
    let coverage_visibility_only =
        matches!(scenario, SeparateEnabledHostScenario::CoverageVisibility);
    let binary_environment = if matches!(scenario, SeparateEnabledHostScenario::LegacyAuthorization)
    {
        "RUST_TOOLS_LEGACY_ENABLED_HOST_BINARY"
    } else {
        "RUST_TOOLS_ENABLED_HOST_BINARY"
    };
    let binary = std::env::var_os(binary_environment)
        .map(PathBuf::from)
        .expect("set the matching separate enabled or legacy host binary");
    let server_state_dir = std::env::var_os("RUST_TOOLS_ENABLED_HOST_STATE_DIR")
        .map(PathBuf::from)
        .expect("set RUST_TOOLS_ENABLED_HOST_STATE_DIR for that host");
    assert!(binary.is_file(), "missing enabled host binary: {binary:?}");
    assert!(
        server_state_dir.is_dir(),
        "missing enabled host state: {server_state_dir:?}"
    );
    let fixture = tempfile::tempdir_in("/private/tmp").expect("create remote profile fixture");
    std::fs::write(fixture.path().join("profile.json"), relay_paged_profile())
        .expect("write paged profile");
    std::fs::write(
        fixture.path().join("profile.syms.json"),
        include_str!("../../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json")
            .replace("/fixture/", &format!("{}/", fixture.path().display())),
    )
    .expect("write profile symbols");
    std::fs::write(fixture.path().join("workload.rs"), "fn leaf() {}\n")
        .expect("write mapped source");
    std::fs::create_dir(fixture.path().join("src")).expect("create Cargo source directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        "[package]\nname = \"collaboration_host_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo manifest");
    std::fs::write(
        fixture.path().join("src/lib.rs"),
        "#[cfg(test)] mod tests { #[test] fn unit_passes() {} }\n",
    )
    .expect("write Cargo test source");
    let ownership_build_output = needs_isolated_build_output
        .then(|| tempfile::tempdir_in("/private/tmp").expect("create isolated owner build output"));
    let compiler_wrapper = discovery_cancellation_gate.as_ref().map(|gate| {
        use std::os::unix::fs::PermissionsExt;
        let wrapper = gate.path().join("compiler-gate.py");
        std::fs::write(&wrapper, r#"#!/usr/bin/python3
import json,os,pathlib,sys,time

gate=pathlib.Path(__file__).parent
arguments=sys.argv[1:]
compiler=arguments[0] if arguments else None
if compiler is None:
 raise SystemExit('missing Cargo-provided compiler argument')
compile_test=(
 '--crate-name' in arguments
 and arguments[arguments.index('--crate-name')+1:arguments.index('--crate-name')+2]==['collaboration_host_fixture']
 and '--test' in arguments
 and not any(argument.startswith('--print') for argument in arguments)
)
if compile_test and (gate/'armed').exists():
 parent=os.getppid()
 phase=(gate/'armed').read_text().strip()
 identity={'phase':phase,'cargo_pid':parent,'wrapper_pid':os.getpid(),'compiler':compiler,'arguments':arguments}
 # An atomic rename keeps the test from parsing a partially written marker.
 temporary=gate/('started-'+phase+'-'+str(os.getpid())+'.tmp')
 temporary.write_text(json.dumps(identity))
 temporary.replace(gate/('started-'+phase+'-'+str(os.getpid())+'.json'))
 deadline=time.monotonic()+30
 while not (gate/'release').exists():
  if os.getppid()!=parent:
   temporary=gate/('parent-exited-'+str(os.getpid())+'.tmp')
   temporary.write_text(str(parent))
   temporary.replace(gate/('parent-exited-'+str(os.getpid())))
   raise SystemExit(86)
  if time.monotonic()>=deadline:
   raise SystemExit('owned fixture compiler gate timed out')
  time.sleep(0.005)
os.execvp(compiler,arguments)
"#).expect("write owned actual-Cargo compiler gate");
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o755))
            .expect("enable only the owned fixture wrapper");
        wrapper
    });
    // Cargo-generated worktree inputs must not confound the scenario under test.
    std::fs::write(
        fixture.path().join("Cargo.lock"),
        "# This file is automatically @generated by Cargo.\n# It is not intended for manual editing.\nversion = 4\n\n[[package]]\nname = \"collaboration_host_fixture\"\nversion = \"0.1.0\"\n",
    )
    .expect("write stable remote fixture lockfile");
    if let Some(output) = &ownership_build_output {
        std::fs::create_dir(fixture.path().join(".cargo"))
            .expect("create owner fixture Cargo configuration");
        let target_path = serde_json::to_string(output.path()).expect("encode test output path");
        let wrapper_configuration = compiler_wrapper
            .as_ref()
            .map(|wrapper| {
                format!(
                    "rustc-wrapper = {}\n",
                    serde_json::to_string(wrapper).expect("encode owned wrapper path")
                )
            })
            .unwrap_or_default();
        std::fs::write(
            fixture.path().join(".cargo/config.toml"),
            format!("[build]\ntarget-dir = {target_path}\n{wrapper_configuration}"),
        )
        .expect("configure Cargo output outside the shared worktree");
    }
    let coverage = serde_json::json!({
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
        .expect("write coverage artifact");
    let host_page_gate_dir = std::env::var_os("RUST_TOOLS_PROFILE_HOST_PAGE_GATE")
        .map(|_| fixture.path().join("profile-page-gate"));
    if let Some(directory) = &host_page_gate_dir {
        std::fs::create_dir(directory).expect("create host page gate directory");
    }

    executor.allow_parking();
    cx_a.executor().allow_parking();
    cx_b.executor().allow_parking();
    cx_c.executor().allow_parking();
    cx_a.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
    cx_a.update(|cx| project::trusted_worktrees::init(Default::default(), cx));
    cx_b.update(|cx| project::trusted_worktrees::init(Default::default(), cx));
    cx_c.update(|cx| project::trusted_worktrees::init(Default::default(), cx));
    cx_b.update(language_tools::profile_view::init);
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "profile_host_member").await;
    let client_b = server.create_client(cx_b, "profile_importer").await;
    let client_c = server.create_client(cx_c, "other_profile_member").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b), (&client_c, cx_c)])
        .await;

    let local_cargo_calls_before =
        project::cargo_workspace_store::local_cargo_process_calls_for_test();
    let local_discovery_calls_before =
        project::rust_test_provider::local_rust_test_discovery_calls_for_test();
    let local_coverage_interpret_calls_before =
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test();
    let connection =
        ProcessRemoteConnection::new(binary, server_state_dir).with_isolated_server_namespace();
    let connection = if scenario != SeparateEnabledHostScenario::LegacyAuthorization {
        let host_data_directory = fixture.path().join("host-data");
        std::fs::create_dir(&host_data_directory).expect("create owned host data directory");
        std::fs::create_dir(host_data_directory.join("config"))
            .expect("create owned host configuration");
        std::fs::write(host_data_directory.join("config/settings.json"), r#"{"feature_flags":{"rust-coverage-experimental":"on","rust-native-profiling-experimental":"on"}}"#).expect("write explicit local host experimental opt-in");
        for context in [&mut *cx_a, &mut *cx_b, &mut *cx_c] {
            context.update(|cx| {
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
        }
        connection.with_test_data_directory(host_data_directory)
    } else {
        connection
    };
    let connection = if let Some(directory) = &host_page_gate_dir {
        connection.with_profile_page_gate(directory.clone())
    } else {
        connection
    };
    let connection = Arc::new(connection);
    let (_cancel_tx, cancel_rx) = futures::channel::oneshot::channel();
    let remote_client = cx_a
        .update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::setup(),
                connection.clone(),
                cancel_rx,
                Arc::new(MockDelegate),
                cx,
            )
        })
        .await
        .expect("connect to separate enabled host")
        .expect("remote client");
    let (project_a, worktree_id_a) = client_a
        .build_ssh_project(fixture.path(), remote_client.clone(), true, cx_a)
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share enabled remote host project");
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    let other_project = client_c.join_remote_project(project_id, cx_c).await;
    let other_project = needs_isolated_build_output.then_some(other_project);
    executor.run_until_parked();
    assert!(
        project_b.read_with(cx_b, |project, cx| {
            !project.is_local()
                && project.remote_id() == Some(project_id)
                && project
                    .profile_results_store_for_test()
                    .is_some_and(|store| {
                        store
                            .read(cx)
                            .is_import_only_remote_proxy_for_test(project_id)
                    })
        }),
        "collaboration client must own only an import proxy for the joined project, with no local profile helper, Task rows, or authoritative imports"
    );

    for (project, context) in [(&project_a, &*cx_a), (&project_b, &*cx_b)] {
        wait_for_scanned_relay_artifacts(
            project,
            fixture.path(),
            &[
                "Cargo.toml",
                "src/lib.rs",
                "coverage.json",
                "profile.json",
                "profile.syms.json",
                "workload.rs",
            ],
            &executor,
            context,
        )
        .await;
    }

    if coverage_visibility_only {
        for (project, context) in [(&project_a, &*cx_a), (&project_b, &*cx_b)] {
            wait_for_scanned_relay_artifacts(
                project,
                fixture.path(),
                &["Cargo.lock"],
                &executor,
                context,
            )
            .await;
        }
    }

    if needs_isolated_build_output {
        wait_for_scanned_relay_artifacts(
            other_project.as_ref().expect("retained unrelated Project"),
            fixture.path(),
            &[
                "Cargo.toml",
                "Cargo.lock",
                "src/lib.rs",
                ".cargo/config.toml",
            ],
            &executor,
            cx_c,
        )
        .await;
    }

    if matches!(
        scenario,
        SeparateEnabledHostScenario::ActionOwnerLeave
            | SeparateEnabledHostScenario::DiscoveryCancellation
            | SeparateEnabledHostScenario::NativeTaskOwnership
            | SeparateEnabledHostScenario::CoverageSession
    ) {
        for (project, context) in [(&project_a, &*cx_a), (&project_b, &*cx_b)] {
            wait_for_scanned_relay_artifacts(
                project,
                fixture.path(),
                &["Cargo.lock", ".cargo/config.toml"],
                &executor,
                context,
            )
            .await;
        }
    }

    let worktree_store_a = project_a.read_with(cx_a, |project, _| project.worktree_store());
    let worktree_store_b = project_b.read_with(cx_b, |project, _| project.worktree_store());
    cx_a.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("host member trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store_a,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                        worktree_id_a,
                    ))
                    .collect(),
                    cx,
                );
            });
    });
    let worktree_id_b = project_b.read_with(cx_b, |project, cx| {
        project
            .worktrees(cx)
            .next()
            .expect("shared fixture worktree")
            .read(cx)
            .id()
    });
    cx_b.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("importer trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store_b,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                        worktree_id_b,
                    ))
                    .collect(),
                    cx,
                );
            });
    });
    executor.run_until_parked();

    let cargo = project_b.read_with(cx_b, |project, _| project.cargo_workspace_store().clone());
    let metadata = cargo
        .update(cx_b, |store, cx| store.refresh(cx))
        .await
        .expect("collaborator receives enabled host Cargo metadata");
    assert_eq!(metadata.workspaces.len(), 1);
    if matches!(scenario, SeparateEnabledHostScenario::LegacyAuthorization) {
        use client::ErrorExt as _;

        let structured = project_b.read_with(cx_b, |project, _| {
            project.structured_execution_store().clone()
        });
        let rust_tests = project_b.read_with(cx_b, |project, _| {
            project.rust_test_provider_store().clone()
        });
        let rust_provider = project::rust_test_provider::RustTestProviderStore::provider_id();
        let upstream = remote_client.read_with(cx_a, |remote, _| remote.proto_client());
        // The sealed 1e591e host predates discovery v2; its raw catalog must never enter the current store.
        let legacy_protocol = 1;
        let catalog = async {
            let discovery = upstream
                .request(proto::GetRustTestDiscovery {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    request_id: 42,
                    protocol_version: legacy_protocol,
                    worktree_ids: vec![worktree_id_a.to_proto()],
                })
                .await
                .expect("actual legacy v1 discovery reaches the sealed host");
            assert_eq!(discovery.protocol_version, legacy_protocol);
            assert_eq!(discovery.request_id, 42);
            assert_eq!(
                discovery.status,
                proto::StructuredProviderStatus::Current as i32
            );
            assert_ne!(discovery.discovery_generation, 0);
            let mut page_start = 0;
            let mut node_count = 0;
            let mut case_id = None;
            loop {
                let page = upstream
                    .request(proto::GetStructuredExecutionSnapshot {
                        project_id: proto::REMOTE_SERVER_PROJECT_ID,
                        provider_id: rust_provider.0.clone(),
                        discovery_generation: discovery.discovery_generation,
                        page_start,
                        page_size: project::structured_execution::MAX_STRUCTURED_PAGE_SIZE as u32,
                        worktree_ids: vec![worktree_id_a.to_proto()],
                    })
                    .await
                    .expect("read bounded genuine legacy catalog directly");
                assert_eq!(
                    page.protocol_version,
                    project::structured_execution::STRUCTURED_EXECUTION_PROTOCOL_VERSION
                );
                assert_eq!(page.provider_id, rust_provider.0);
                assert_eq!(page.discovery_generation, discovery.discovery_generation);
                assert_eq!(page.status, proto::StructuredProviderStatus::Current as i32);
                assert!(
                    page.nodes.len() <= project::structured_execution::MAX_STRUCTURED_PAGE_SIZE
                );
                node_count += page.nodes.len();
                assert!(node_count <= project::structured_execution::MAX_STRUCTURED_NODES);
                for node in &page.nodes {
                    if node.label == "unit_passes"
                        && node.kind == proto::StructuredExecutionNodeKind::Case as i32
                    {
                        assert!(case_id.is_none(), "legacy fixture case must be unique");
                        case_id = Some(StructuredNodeId(node.node_id.clone()));
                    }
                }
                if page.next_page_start == 0 {
                    break;
                }
                assert!(
                    page.next_page_start > page_start,
                    "legacy catalog paging must advance"
                );
                assert!(
                    page.next_page_start as usize
                        <= project::structured_execution::MAX_STRUCTURED_NODES
                );
                page_start = page.next_page_start;
            }
            (
                DiscoveryGeneration(discovery.discovery_generation),
                case_id.expect("actual legacy unit case"),
            )
        };
        let (discovery_generation, case_id) = match futures::future::select(
            Box::pin(catalog),
            Box::pin(executor.timer(std::time::Duration::from_secs(30))),
        )
        .await
        {
            futures::future::Either::Left((catalog, _)) => catalog,
            futures::future::Either::Right(_) => {
                panic!("legacy discovery did not complete within the original discovery deadline")
            }
        };
        let mismatch = upstream
            .request(proto::GetRustTestDiscovery {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                request_id: 43,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id_a.to_proto()],
            })
            .await
            .expect("current discovery protocol receives an explicit legacy mismatch");
        assert_eq!(mismatch.protocol_version, legacy_protocol);
        assert_eq!(mismatch.request_id, 43);
        assert_eq!(
            mismatch.status,
            proto::StructuredProviderStatus::Mismatch as i32
        );
        assert_eq!(mismatch.discovery_generation, 0);
        let probe = upstream
            .request(proto::ResolveRustTestAction {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                protocol_version: legacy_protocol,
                discovery_generation: discovery_generation.0,
                node_id: case_id.0.clone(),
                action: proto::RustTestActionKind::Run as i32,
                worktree_ids: vec![worktree_id_a.to_proto()],
                authorization_generation: 42,
            })
            .await
            .expect("legacy host resolves a direct Run without echoing authorization");
        assert_eq!(probe.protocol_version, legacy_protocol);
        assert_eq!(probe.authorization_generation, 0);
        assert!(
            probe.plan_json.len() <= project::rust_test_provider::MAX_RUST_TEST_ACTION_PLAN_BYTES
        );
        let legacy_plan: serde_json::Value = serde_json::from_str(&probe.plan_json)
            .expect("legacy host returns an actual structured Task plan");
        assert_eq!(
            legacy_plan.get("kind").and_then(serde_json::Value::as_str),
            Some("task")
        );
        let legacy_run_id = legacy_plan
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            .expect("legacy Run identity");
        for action in [
            project::rust_test_provider::RustTestAction::Run,
            project::rust_test_provider::RustTestAction::Debug,
        ] {
            let error = rust_tests
                .update(cx_b, |store, cx| {
                    store.plan_action(&case_id, discovery_generation, action, cx)
                })
                .await
                .expect_err("collaboration cannot use an unbound legacy action plan");
            assert!(
                error
                    .to_string()
                    .contains("Rust test discovery is unavailable. Refresh Tests"),
                "an incompatible catalog cannot authorize the public action entry point: {error:#}"
            );
            let error = client_b
                .client()
                .request(proto::ResolveRustTestAction {
                    project_id,
                    protocol_version: legacy_protocol,
                    discovery_generation: discovery_generation.0,
                    node_id: case_id.0.clone(),
                    action: match action {
                        project::rust_test_provider::RustTestAction::Run => {
                            proto::RustTestActionKind::Run
                        }
                        project::rust_test_provider::RustTestAction::Debug => {
                            proto::RustTestActionKind::Debug
                        }
                    } as i32,
                    worktree_ids: vec![worktree_id_b.to_proto()],
                    authorization_generation: 0,
                })
                .await
                .expect_err("the current relay cannot admit an incompatible unbound legacy action");
            assert_eq!(
                error
                    .downcast_ref::<proto::RpcError>()
                    .and_then(|error| error.error_tag("reason")),
                Some("rust_tools_unsupported"),
                "{error:#}"
            );
        }
        let lifecycle_request = proto::UpdateRustTestRun {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: discovery_generation.0,
            run_id: legacy_run_id.into(),
            scope_node_ids: vec![case_id.0.clone()],
            task_id: "legacy-authorization-probe".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id_b.to_proto()],
            ..Default::default()
        };
        let error = client_b
            .client()
            .request(lifecycle_request.clone())
            .await
            .expect_err("a declined legacy action cannot authorize member lifecycle events");
        assert!(
            error
                .to_string()
                .contains("Unknown or expired remote Rust test run"),
            "{error:#}"
        );
        let error = client_b
            .client()
            .request(proto::UpdateRustTestRun {
                protocol_version: legacy_protocol,
                ..lifecycle_request
            })
            .await
            .expect_err("an incompatible protocol cannot publish member lifecycle events");
        assert_eq!(
            error
                .downcast_ref::<proto::RpcError>()
                .and_then(|error| error.error_tag("reason")),
            Some("rust_tools_unsupported"),
            "{error:#}"
        );
        structured.read_with(cx_b, |store, _| {
            if let Some(snapshot) = store.state().provider(&rust_provider) {
                assert_ne!(snapshot.status, StructuredProviderStatus::Current);
                assert!(snapshot.current_run.is_none());
                assert!(snapshot.last_complete_run.is_none());
            }
        });
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            local_cargo_calls_before
        );
        assert_eq!(
            project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
            local_discovery_calls_before
        );
        assert_eq!(
            project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
            local_coverage_interpret_calls_before
        );
        return;
    }

    if matches!(
        scenario,
        SeparateEnabledHostScenario::ActionOwnerLeave
            | SeparateEnabledHostScenario::DiscoveryCancellation
            | SeparateEnabledHostScenario::NativeTaskOwnership
            | SeparateEnabledHostScenario::CoverageSession
    ) {
        // Initial scan and trust-triggered discovery must retire before the controlled collection.
        let initial_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            let idle = project_a.read_with(cx_a, |project, cx| {
                project
                    .rust_test_provider_store()
                    .read(cx)
                    .discovery_idle_for_test()
            }) && project_b.read_with(cx_b, |project, cx| {
                project
                    .rust_test_provider_store()
                    .read(cx)
                    .discovery_idle_for_test()
            }) && other_project
                .as_ref()
                .expect("retained unrelated Project")
                .read_with(cx_c, |project, cx| {
                    project
                        .rust_test_provider_store()
                        .read(cx)
                        .discovery_idle_for_test()
                });
            if idle {
                break;
            }
            assert!(
                std::time::Instant::now() < initial_deadline,
                "initial discovery must retire before its explicit baseline"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
    }
    if needs_isolated_build_output {
        // Initial input delivery may supersede this refresh; departure needs the latest result.
        let discovery = project_b
            .read_with(cx_b, |project, _| {
                project.rust_test_provider_store().clone()
            })
            .update(cx_b, |store, cx| store.refresh(cx))
            .await;
        if matches!(
            scenario,
            SeparateEnabledHostScenario::ActionOwnerLeave
                | SeparateEnabledHostScenario::DiscoveryCancellation
                | SeparateEnabledHostScenario::NativeTaskOwnership
                | SeparateEnabledHostScenario::CoverageSession
        ) {
            let discovery = discovery.expect("settled actual-Cargo baseline must recover once");
            assert_eq!(discovery.status, StructuredProviderStatus::Current);
        } else {
            match discovery {
                Ok(discovery) => assert_eq!(
                    discovery.status,
                    project::structured_execution::StructuredProviderStatus::Current
                ),
                Err(error) => assert_eq!(
                    error
                        .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
                        .map(|error| error.kind),
                    Some(project::rust_test_provider::RustTestRemoteErrorKind::SupersededDiscovery),
                    "initial discovery must remain connected: {error:#}"
                ),
            }
        }
    }

    let structured = project_b.read_with(cx_b, |project, _| {
        project.structured_execution_store().clone()
    });
    let rust_provider = project::rust_test_provider::RustTestProviderStore::provider_id();
    let discovery_deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let (discovery_generation, case_id) = loop {
        if needs_isolated_build_output {
            let clients_idle = project_a.read_with(cx_a, |project, cx| {
                project
                    .rust_test_provider_store()
                    .read(cx)
                    .discovery_idle_for_test()
            }) && project_b.read_with(cx_b, |project, cx| {
                project
                    .rust_test_provider_store()
                    .read(cx)
                    .discovery_idle_for_test()
            }) && other_project
                .as_ref()
                .expect("retained unrelated Project")
                .read_with(cx_c, |project, cx| {
                    project
                        .rust_test_provider_store()
                        .read(cx)
                        .discovery_idle_for_test()
                });
            if !clients_idle {
                assert!(
                    std::time::Instant::now() < discovery_deadline,
                    "initial client discovery did not settle"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
                continue;
            }
        }
        let snapshot = structured.read_with(cx_b, |store, _| {
            store.state().provider(&rust_provider).cloned()
        });
        if let Some(snapshot) = snapshot.as_ref().filter(|snapshot| {
            snapshot.status == project::structured_execution::StructuredProviderStatus::Current
        }) && let Some(case) = snapshot
            .nodes
            .iter()
            .find(|node| node.label == "unit_passes")
        {
            break (snapshot.discovery_generation, case.id.clone());
        }
        assert!(
            std::time::Instant::now() < discovery_deadline,
            "collaborator did not receive enabled host Rust discovery: {snapshot:?}"
        );
        executor.timer(std::time::Duration::from_millis(100)).await;
        executor.run_until_parked();
    };

    if matches!(scenario, SeparateEnabledHostScenario::CoverageSession) {
        let coverage_path = project_b.read_with(cx_b, |project, cx| {
            project
                .find_project_path(fixture.path().join("coverage.json"), cx)
                .expect("session fixture coverage artifact")
        });
        let coverage = project_b.read_with(cx_b, |project, _| {
            project.rust_coverage_provider_store().clone()
        });
        let source_coverage =
            project_b.read_with(cx_b, |project, _| project.source_coverage_store().clone());
        let baseline = coverage
            .update(cx_b, |store, cx| {
                store.interpret_artifact(
                    coverage_path.clone(),
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .expect("initial two-hop coverage reaches the enabled host");
        assert_eq!(baseline.files.len(), 1);
        assert_eq!(
            baseline.status,
            project::source_coverage::SourceCoverageStatus::Current
        );
        let source_path = baseline
            .files
            .first()
            .expect("covered source path")
            .path
            .clone();
        let buffer = project_b
            .update(cx_b, |project, cx| project.open_buffer(source_path, cx))
            .await
            .expect("open actual two-hop covered source buffer");
        cx_b.update(language_tools::source_coverage_summary::init);
        let (workspace_b, editor) = {
            use gpui::VisualContext;
            let (workspace, window_cx) = client_b.build_workspace(&project_b, cx_b);
            let editor = window_cx.new_window_entity(|window, cx| {
                Editor::for_buffer(buffer, Some(project_b.clone()), window, cx)
            });
            workspace.update_in(window_cx, |workspace, window, cx| {
                workspace.add_item_to_active_pane(Box::new(editor.clone()), None, true, window, cx);
            });
            (workspace, editor)
        };
        let assert_coverage_ui =
            |status: &str, gutters, navigation_count, cx: &mut TestAppContext| {
                let (_, window_cx) = client_b.active_workspace(cx);
                window_cx.run_until_parked();
                window_cx.update(|window, cx| window.draw(cx).clear(cx));
                assert_eq!(
                    editor.read_with(window_cx, |editor, _| editor
                        .source_coverage_gutter_counts()),
                    gutters
                );
                let (label, count) = workspace_b
                    .read_with(window_cx, |_, cx| {
                        language_tools::source_coverage_summary::toolbar_snapshot_for_test(
                            &workspace_b,
                            cx,
                        )
                    })
                    .expect("production collaboration coverage toolbar");
                assert!(label.contains(status), "{label}");
                assert_eq!(count, navigation_count);
            };
        assert_coverage_ui("current", (1, 0), 1, cx_b);
        let host_reshared = Arc::new(Mutex::new(false));
        let _reshared_subscription = project_a.update(cx_a, |_, cx| {
            let host_reshared = host_reshared.clone();
            cx.subscribe(&project_a, move |_, _, event, _| {
                if matches!(event, project::Event::Reshared) {
                    *host_reshared
                        .lock()
                        .expect("coverage host reshare observation") = true;
                }
            })
        });
        for reconnect_host in [false, true] {
            let (started, release) =
                coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
            let pending = coverage.update(cx_b, |store, cx| {
                store.interpret_artifact(
                    coverage_path.clone(),
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            });
            let replace_session = async {
                assert!(
                    matches!(
                        futures::future::select(
                            started,
                            executor.timer(std::time::Duration::from_secs(10))
                        )
                        .await,
                        futures::future::Either::Left((Ok(()), _))
                    ),
                    "client must hold the fully completed host interpretation before publication"
                );
                let host_peer = client_a.peer_id().expect("coverage host peer");
                let member_peer = client_b.peer_id().expect("coverage member peer");
                let other_peer = client_c.peer_id().expect("unrelated coverage peer");
                let previous_peer = if reconnect_host {
                    *host_reshared.lock().expect("reset coverage host reshare") = false;
                    host_peer
                } else {
                    member_peer
                };
                server.disconnect_client(previous_peer);
                let step = std::time::Duration::from_millis(100);
                let mut remaining = rpc::RECEIVE_TIMEOUT;
                loop {
                    executor.run_until_parked();
                    let ready = if reconnect_host {
                        client_a.peer_id().is_some_and(|peer| peer != previous_peer)
                            && *host_reshared.lock().expect("observe coverage host reshare")
                    } else {
                        client_b.peer_id().is_some_and(|peer| {
                            peer != previous_peer
                                && project_a.read_with(cx_a, |project, _| {
                                    project.collaborators().contains_key(&peer)
                                })
                        })
                    };
                    if ready {
                        break;
                    }
                    assert!(
                        remaining >= step,
                        "coverage collaboration session did not reconnect"
                    );
                    remaining = remaining.saturating_sub(step);
                    executor.advance_clock(step);
                }
                assert_eq!(client_c.peer_id(), Some(other_peer));
                if reconnect_host {
                    assert_eq!(client_b.peer_id(), Some(member_peer));
                } else {
                    assert_eq!(client_a.peer_id(), Some(host_peer));
                }
                source_coverage.read_with(cx_b, |store, _| {
                    let cached = store
                        .selected_snapshot()
                        .expect("coverage cache survives peer replacement");
                    assert_eq!(
                        cached.status,
                        project::source_coverage::SourceCoverageStatus::Stale
                    );
                    assert_eq!(cached.files, baseline.files);
                });
                assert_coverage_ui("stale", (0, 0), 0, cx_b);
                release
                    .send(())
                    .expect("release completed coverage after actual peer replacement");
            };
            let (obsolete, ()) = futures::join!(pending, replace_session);
            let error = obsolete
                .expect_err("new peer cannot revive a completed old-session interpretation");
            assert!(
                error.to_string().contains("expired project session"),
                "{error:#}"
            );
            source_coverage.read_with(cx_b, |store, _| {
                let cached = store
                    .selected_snapshot()
                    .expect("stale coverage after late reply");
                assert_eq!(
                    cached.status,
                    project::source_coverage::SourceCoverageStatus::Stale
                );
                assert_eq!(cached.files, baseline.files);
            });
            assert_coverage_ui("stale", (0, 0), 0, cx_b);
            let fresh = coverage
                .update(cx_b, |store, cx| {
                    store.interpret_artifact(
                        coverage_path.clone(),
                        project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                        cx,
                    )
                })
                .await
                .expect("fresh host interpretation recovers under the replacement session");
            assert_eq!(
                fresh.status,
                project::source_coverage::SourceCoverageStatus::Current
            );
            assert_eq!(fresh.files, baseline.files);
            assert_coverage_ui("current", (1, 0), 1, cx_b);
        }
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            local_cargo_calls_before
        );
        assert_eq!(
            project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
            local_discovery_calls_before
        );
        assert_eq!(
            project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
            local_coverage_interpret_calls_before
        );
        return;
    }

    if matches!(scenario, SeparateEnabledHostScenario::NativeTaskOwnership) {
        use std::sync::atomic::AtomicBool;

        let paths_a = project_a.read_with(cx_a, |project, cx| {
            ["profile.json", "profile.syms.json"].map(|name| {
                project
                    .find_project_path(fixture.path().join(name), cx)
                    .expect("direct host native Task artifact")
            })
        });
        let paths_b = project_b.read_with(cx_b, |project, cx| {
            ["profile.json", "profile.syms.json"].map(|name| {
                project
                    .find_project_path(fixture.path().join(name), cx)
                    .expect("collaboration native Task artifact")
            })
        });
        let run_id = "direct-host-owned-native-task";
        let binding = project_a
            .read_with(cx_a, |project, cx| {
                project.prepare_remote_profile_task(run_id.into(), paths_a.clone(), cx)
            })
            .expect("direct host native Task preparation request")
            .await
            .expect("direct enabled host owns the native Task binding");
        let direct_client = project_a.read_with(cx_a, |project, cx| {
            project
                .remote_client()
                .expect("direct enabled host connection")
                .read(cx)
                .proto_client()
        });
        let prepare_direct = || proto::PrepareProfileTaskBinding {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            task_run_id: run_id.into(),
            profile_path: Some(paths_a[0].to_proto()),
            symbols_path: Some(paths_a[1].to_proto()),
            artifact_format: "samply-0.13.1".into(),
        };
        let host_reshared = Arc::new(Mutex::new(false));
        let _reshared_subscription = project_a.update(cx_a, |_, cx| {
            let host_reshared = host_reshared.clone();
            cx.subscribe(&project_a, move |_, _, event, _| {
                if matches!(event, project::Event::Reshared) {
                    *host_reshared
                        .lock()
                        .expect("direct host reshare observation") = true;
                }
            })
        });
        for reconnect_host in [false, true] {
            let unsupported = project_b
                .read_with(cx_b, |project, cx| {
                    project.prepare_remote_profile_task(run_id.into(), paths_b.clone(), cx)
                })
                .expect("member preparation reaches remote-backed collaboration host")
                .await
                .err()
                .expect("guest Task restriction must remain explicit");
            assert_eq!(
                unsupported.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                Some(&project::profile_results_store::RemoteProfileError::Unsupported)
            );
            let unsupported = project_b
                .read_with(cx_b, |project, cx| {
                    project.get_remote_profile_task(run_id.into(), cx)
                })
                .expect("member status request reaches remote-backed host")
                .await
                .expect_err("member cannot retrieve a direct host-owned binding");
            assert_eq!(
                unsupported.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                Some(&project::profile_results_store::RemoteProfileError::Unsupported)
            );
            for client in [&client_b, &client_c] {
                client
                    .client()
                    .request(proto::CancelProfileTaskBinding {
                        project_id,
                        task_run_id: run_id.into(),
                    })
                    .await
                    .expect("foreign cancellation does not mutate the direct host owner");
            }
            let retained = direct_client
                .request(prepare_direct())
                .await
                .expect("direct native Task reservation remains observable");
            assert_eq!(retained.status, proto::ProfileResultStatus::Error as i32);
            assert!(retained.token.is_none());
            let host_peer = client_a.peer_id().expect("current host member peer");
            let member_peer = client_b.peer_id().expect("current member peer");
            let other_peer = client_c.peer_id().expect("current unrelated member peer");
            let previous_peer = if reconnect_host {
                *host_reshared
                    .lock()
                    .expect("reset host reshare observation") = false;
                host_peer
            } else {
                member_peer
            };
            server.disconnect_client(previous_peer);
            let step = std::time::Duration::from_millis(100);
            let mut remaining = rpc::RECEIVE_TIMEOUT;
            loop {
                executor.run_until_parked();
                let ready = if reconnect_host {
                    client_a.peer_id().is_some_and(|peer| peer != previous_peer)
                        && *host_reshared.lock().expect("observe host reshare")
                } else {
                    client_b.peer_id().is_some_and(|peer| {
                        peer != previous_peer
                            && project_a.read_with(cx_a, |project, _| {
                                project.collaborators().contains_key(&peer)
                            })
                    })
                };
                if ready {
                    break;
                }
                assert!(
                    remaining >= step,
                    "native Task member route did not reconnect"
                );
                remaining = remaining.saturating_sub(step);
                executor.advance_clock(step);
            }
            assert_eq!(client_c.peer_id(), Some(other_peer));
            if reconnect_host {
                assert_eq!(client_b.peer_id(), Some(member_peer));
            } else {
                assert_eq!(client_a.peer_id(), Some(host_peer));
            }
            let retained = direct_client
                .request(prepare_direct())
                .await
                .expect("independent direct owner survives collaboration peer replacement");
            assert_eq!(retained.status, proto::ProfileResultStatus::Error as i32);
            assert!(retained.token.is_none());
            let unsupported = project_b
                .read_with(cx_b, |project, cx| {
                    project.prepare_remote_profile_task(run_id.into(), paths_b.clone(), cx)
                })
                .expect("fresh member preparation after peer replacement")
                .await
                .err()
                .expect("reconnect does not grant guest Task capability");
            assert_eq!(
                unsupported.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
                Some(&project::profile_results_store::RemoteProfileError::Unsupported)
            );
        }
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
            .expect("launch actual direct host-owned observer helper");
        project_a
            .read_with(cx_a, |project, cx| {
                project.get_remote_profile_task(run_id.into(), cx)
            })
            .expect("direct host Task completion request")
            .await
            .expect("actual observer seals the direct owner's artifacts");
        assert!(
            observed_task
                .status()
                .await
                .expect("actual observer exit")
                .success()
        );
        let local_profile_loads_before =
            project::profile_results::samply_profile_load_calls_for_test();
        let model = project_a
            .read_with(cx_a, |project, cx| {
                project.import_remote_task_profile(
                    paths_a,
                    run_id.into(),
                    Arc::new(AtomicBool::new(false)),
                    cx,
                )
            })
            .expect("direct owner's sealed Task profile request")
            .await
            .expect("direct host Task profile imports after collaboration reconnects");
        assert_eq!(model.total_samples, 3_030);
        assert!(model.frames.iter().any(|frame| {
            frame.label == "workload::leaf"
                && frame
                    .source
                    .as_ref()
                    .is_some_and(|(path, _)| path.path.as_unix_str() == "workload.rs")
        }));
        project_a
            .read_with(cx_a, |project, cx| {
                project.cancel_remote_profile_task(run_id.into(), cx)
            })
            .expect("direct host owner releases native Task binding")
            .await
            .expect("owner cancellation completes");
        let stale = project_a
            .read_with(cx_a, |project, cx| {
                project.get_remote_profile_task(run_id.into(), cx)
            })
            .expect("released direct Task status request")
            .await
            .expect_err("owner cancellation retires the sealed Task binding");
        assert_eq!(
            stale.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
            Some(&project::profile_results_store::RemoteProfileError::Stale)
        );
        let consented = project_b
            .read_with(cx_b, |project, cx| {
                project.import_remote_profile(paths_b, true, Arc::new(AtomicBool::new(false)), cx)
            })
            .expect("fresh explicit guest import after collaboration reconnects")
            .await
            .expect("guest retains approved consented import/view capability");
        assert_eq!(consented.total_samples, 3_030);
        assert_eq!(
            project::profile_results::samply_profile_load_calls_for_test(),
            local_profile_loads_before,
            "native profiles are parsed only by the separate authoritative host"
        );
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            local_cargo_calls_before
        );
        assert_eq!(
            project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
            local_discovery_calls_before
        );
        assert_eq!(
            project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
            local_coverage_interpret_calls_before
        );
        assert_eq!(
            other_project
                .as_ref()
                .expect("retained unrelated Project")
                .read_with(cx_c, |project, _| project.remote_id()),
            Some(project_id)
        );
        return;
    }

    let rust_tests = project_b.read_with(cx_b, |project, _| {
        project.rust_test_provider_store().clone()
    });
    if matches!(scenario, SeparateEnabledHostScenario::DiscoveryCancellation) {
        #[derive(Debug, serde::Deserialize)]
        struct CargoIdentity {
            phase: String,
            cargo_pid: u32,
            wrapper_pid: u32,
            compiler: String,
            arguments: Vec<String>,
        }

        fn identities(directory: &std::path::Path, phase: &str) -> Vec<CargoIdentity> {
            std::fs::read_dir(directory)
                .expect("read owned compiler gate directory")
                .map(|entry| entry.expect("owned gate entry").path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with(&format!("started-{phase}-"))
                                && name.ends_with(".json")
                        })
                })
                .map(|path| {
                    serde_json::from_slice(
                        &std::fs::read(path).expect("read complete atomic identity"),
                    )
                    .expect("decode owned compiler identity")
                })
                .collect()
        }

        async fn assert_cargo_live(identity: &CargoIdentity) {
            let process = smol::process::Command::new("/bin/ps")
                .args(["-p", &identity.cargo_pid.to_string(), "-o", "comm="])
                .output()
                .await
                .expect("inspect only recorded owned Cargo PID");
            assert!(
                process.status.success(),
                "owned Cargo is absent: {identity:?}"
            );
            let executable = String::from_utf8(process.stdout).expect("owned command name");
            assert_eq!(
                std::path::Path::new(executable.trim())
                    .file_name()
                    .and_then(|name| name.to_str()),
                Some("cargo"),
                "recorded parent must be real Cargo, not the gate wrapper: {identity:?}"
            );
            assert_eq!(identity.arguments.first(), Some(&identity.compiler));
            assert!(
                identity
                    .arguments
                    .iter()
                    .any(|argument| argument == "--test")
            );
        }

        async fn assert_pid_absent(pid: u32) {
            let process = smol::process::Command::new("/bin/ps")
                .args(["-p", &pid.to_string(), "-o", "pid="])
                .output()
                .await
                .expect("check recorded owned process exit");
            assert_eq!(
                process.status.code(),
                Some(1),
                "owned process {pid} is still present"
            );
            assert!(process.stdout.iter().all(u8::is_ascii_whitespace));
            assert!(process.stderr.is_empty());
        }

        async fn wait_for_owned_exit(
            identity: &CargoIdentity,
            gate: &std::path::Path,
            executor: &BackgroundExecutor,
        ) {
            let started = std::time::Instant::now();
            let marker = gate.join(format!("parent-exited-{}", identity.wrapper_pid));
            while !marker.exists() {
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(5),
                    "final cancellation did not retire actual Cargo: {identity:?}"
                );
                executor.timer(std::time::Duration::from_millis(5)).await;
            }
            assert_eq!(
                std::fs::read_to_string(marker).expect("parent exit marker"),
                identity.cargo_pid.to_string()
            );
            loop {
                let process = smol::process::Command::new("/bin/ps")
                    .args(["-p", &identity.wrapper_pid.to_string(), "-o", "pid="])
                    .output()
                    .await
                    .expect("inspect owned fixture wrapper exit");
                if process.status.code() == Some(1) {
                    assert!(process.stdout.iter().all(u8::is_ascii_whitespace));
                    assert!(process.stderr.is_empty());
                    break;
                }
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(5),
                    "owned fixture wrapper did not finish after parent loss: {identity:?}"
                );
                executor.timer(std::time::Duration::from_millis(5)).await;
            }
            assert_pid_absent(identity.cargo_pid).await;
            assert_pid_absent(identity.wrapper_pid).await;
            eprintln!(
                "authenticated Cargo retirement phase={} cargo_pid={} wrapper_pid={} elapsed_ms={}",
                identity.phase,
                identity.cargo_pid,
                identity.wrapper_pid,
                started.elapsed().as_millis()
            );
        }

        let gate = discovery_cancellation_gate
            .as_ref()
            .expect("owned discovery gate");
        let build_output = ownership_build_output
            .as_ref()
            .expect("isolated Cargo target");
        let bridge = project_a.read_with(cx_a, |project, _| {
            project.rust_test_provider_store().clone()
        });
        let request = |request_id| proto::GetRustTestDiscovery {
            project_id,
            request_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id_a.to_proto()],
        };

        std::fs::remove_dir_all(build_output.path()).expect("retire only idle fixture target");
        std::fs::create_dir(build_output.path()).expect("recreate isolated fixture target");
        std::fs::write(gate.path().join("armed"), "coalesced")
            .expect("arm actual Cargo compiler gate");
        let first = executor.spawn({
            let client = client_b.client().clone();
            let request = request(920);
            async move { client.request(request).await }
        });
        let admission_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let first_identity = loop {
            let mut observed = identities(gate.path(), "coalesced");
            if !observed.is_empty() {
                assert_eq!(
                    observed.len(),
                    1,
                    "one actual Cargo collection before coalescing"
                );
                break observed.pop().expect("first actual Cargo identity");
            }
            assert!(
                std::time::Instant::now() < admission_deadline,
                "actual Cargo did not enter the compiler gate"
            );
            executor.timer(std::time::Duration::from_millis(5)).await;
        };
        assert_cargo_live(&first_identity).await;
        let second = executor.spawn({
            let client = client_c.client().clone();
            let request = request(921);
            async move { client.request(request).await }
        });
        while bridge.read_with(cx_a, |store, _| {
            store.active_remote_discovery_request_count_for_test()
        }) != 2
        {
            assert!(
                std::time::Instant::now() < admission_deadline,
                "both authenticated waiters must be admitted"
            );
            executor.timer(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(
            identities(gate.path(), "coalesced").len(),
            1,
            "same-scope peers share actual Cargo"
        );
        client_c
            .client()
            .request(proto::CancelRustTestDiscovery {
                project_id,
                request_id: 920,
            })
            .await
            .expect("foreign same-ID cancellation is acknowledged without consuming another peer");
        assert_eq!(
            bridge.read_with(cx_a, |store, _| store
                .active_remote_discovery_request_count_for_test()),
            2
        );
        assert_cargo_live(&first_identity).await;
        client_b
            .client()
            .request(proto::CancelRustTestDiscovery {
                project_id,
                request_id: 920,
            })
            .await
            .expect("first owner cancels its own admitted request");
        let error = first.await.expect_err("first owner sees cancellation");
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        assert_eq!(
            bridge.read_with(cx_a, |store, _| store
                .active_remote_discovery_request_count_for_test()),
            1
        );
        assert_cargo_live(&first_identity).await;
        assert!(
            !gate
                .path()
                .join(format!("parent-exited-{}", first_identity.wrapper_pid))
                .exists()
        );
        client_c
            .client()
            .request(proto::CancelRustTestDiscovery {
                project_id,
                request_id: 921,
            })
            .await
            .expect("final owner cancels its own admitted request");
        let error = second.await.expect_err("final owner sees cancellation");
        assert!(error.to_string().contains("cancelled"), "{error:#}");
        wait_for_owned_exit(&first_identity, gate.path(), &executor).await;
        let retirement_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !bridge.read_with(cx_a, |store, _| store.discovery_idle_for_test()) {
            assert!(
                std::time::Instant::now() < retirement_deadline,
                "bridge must retire the last upstream request"
            );
            executor.timer(std::time::Duration::from_millis(5)).await;
        }
        assert_eq!(
            bridge.read_with(cx_a, |store, _| store
                .active_remote_discovery_request_count_for_test()),
            0
        );
        std::fs::remove_file(gate.path().join("armed")).expect("disarm fixture compiler gate");
        let recovered = rust_tests
            .update(cx_b, |store, cx| store.refresh(cx))
            .await
            .expect("fresh actual Cargo recovery");
        assert_eq!(recovered.status, StructuredProviderStatus::Current);
        assert!(
            recovered
                .nodes
                .iter()
                .any(|node| node.label == "unit_passes")
        );

        let recovery_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !project_a.read_with(cx_a, |project, cx| {
            project
                .rust_test_provider_store()
                .read(cx)
                .discovery_idle_for_test()
        }) || !project_b.read_with(cx_b, |project, cx| {
            project
                .rust_test_provider_store()
                .read(cx)
                .discovery_idle_for_test()
        }) || !other_project
            .as_ref()
            .expect("retained unrelated Project")
            .read_with(cx_c, |project, cx| {
                project
                    .rust_test_provider_store()
                    .read(cx)
                    .discovery_idle_for_test()
            })
        {
            assert!(
                std::time::Instant::now() < recovery_deadline,
                "recovery must settle before clearing its isolated target"
            );
            executor.timer(std::time::Duration::from_millis(5)).await;
        }
        std::fs::remove_dir_all(build_output.path())
            .expect("retire completed recovery fixture target");
        std::fs::create_dir(build_output.path()).expect("recreate replacement fixture target");
        std::fs::write(gate.path().join("armed"), "replacement")
            .expect("arm rapid replacement gate");
        let superseded = rust_tests.update(cx_b, |store, cx| store.refresh(cx));
        let replacement_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let superseded_identity = loop {
            let mut observed = identities(gate.path(), "replacement");
            if !observed.is_empty() {
                assert_eq!(observed.len(), 1);
                break observed.pop().expect("superseded actual Cargo identity");
            }
            assert!(
                std::time::Instant::now() < replacement_deadline,
                "replacement baseline Cargo did not start"
            );
            executor.timer(std::time::Duration::from_millis(5)).await;
        };
        assert_cargo_live(&superseded_identity).await;
        let replacement = rust_tests.update(cx_b, |store, cx| store.refresh(cx));
        let error = superseded
            .await
            .expect_err("rapid refresh supersedes owned pending discovery");
        assert_eq!(
            error
                .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
                .map(|error| error.kind),
            Some(project::rust_test_provider::RustTestRemoteErrorKind::SupersededDiscovery),
            "{error:#}"
        );
        wait_for_owned_exit(&superseded_identity, gate.path(), &executor).await;
        let replacement_identity = loop {
            let mut observed = identities(gate.path(), "replacement");
            observed.retain(|identity| identity.wrapper_pid != superseded_identity.wrapper_pid);
            if !observed.is_empty() {
                assert_eq!(
                    observed.len(),
                    1,
                    "only one new actual Cargo after retirement"
                );
                break observed.pop().expect("fresh replacement Cargo identity");
            }
            assert!(
                std::time::Instant::now() < replacement_deadline,
                "fresh actual Cargo replacement did not start"
            );
            executor.timer(std::time::Duration::from_millis(5)).await;
        };
        assert_ne!(
            replacement_identity.cargo_pid,
            superseded_identity.cargo_pid
        );
        assert_cargo_live(&replacement_identity).await;
        std::fs::write(gate.path().join("release"), "release")
            .expect("release only owned fixture compiler");
        let fresh = replacement
            .await
            .expect("current actual Cargo replacement recovers");
        assert_eq!(fresh.status, StructuredProviderStatus::Current);
        assert!(fresh.nodes.iter().any(|node| node.label == "unit_passes"));
        assert_pid_absent(replacement_identity.cargo_pid).await;
        assert_pid_absent(replacement_identity.wrapper_pid).await;
        eprintln!(
            "authenticated Cargo replacement completed cargo_pid={} wrapper_pid={}",
            replacement_identity.cargo_pid, replacement_identity.wrapper_pid
        );
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            local_cargo_calls_before
        );
        assert_eq!(
            project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
            local_discovery_calls_before
        );
        assert_eq!(
            project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
            local_coverage_interpret_calls_before
        );
        assert_eq!(
            other_project
                .as_ref()
                .expect("retained unrelated member Project")
                .read_with(cx_c, |project, _| project.remote_id()),
            Some(project_id)
        );
        return;
    }
    let run_plan = rust_tests
        .update(cx_b, |store, cx| {
            store.plan_action(
                &case_id,
                discovery_generation,
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect("collaborator resolves Run through the enabled authoritative host");
    let project::rust_test_provider::RustTestActionPlan::Task {
        discovery_generation: planned_generation,
        scope_node_ids,
        worktree_id,
        template,
        run_id,
    } = run_plan
    else {
        panic!("Run must resolve an ordinary Cargo Task plan");
    };
    assert_eq!(planned_generation, discovery_generation);
    assert_eq!(scope_node_ids, std::slice::from_ref(&case_id));
    assert_eq!(worktree_id, worktree_id_b);
    assert_eq!(template.command, "cargo");
    assert_eq!(
        template.args,
        [
            "test",
            "--package",
            "collaboration_host_fixture",
            "--lib",
            "--",
            "tests::unit_passes",
            "--exact",
            "--nocapture",
        ]
    );
    let debug_plan = rust_tests
        .update(cx_b, |store, cx| {
            store.plan_action(
                &case_id,
                discovery_generation,
                project::rust_test_provider::RustTestAction::Debug,
                cx,
            )
        })
        .await
        .expect("collaborator resolves Debug through the enabled authoritative host");
    let project::rust_test_provider::RustTestActionPlan::Debug {
        worktree_id,
        scenario: debug_scenario,
    } = debug_plan
    else {
        panic!("Debug must resolve a Cargo DAP scenario");
    };
    assert_eq!(worktree_id, worktree_id_b);
    assert_eq!(debug_scenario.adapter.as_ref(), "CodeLLDB");
    let Some(task::BuildTaskDefinition::Template {
        task_template,
        locator_name,
    }) = debug_scenario.build
    else {
        panic!("Debug must use the Cargo build template");
    };
    assert_eq!(locator_name.as_deref(), Some("rust-cargo-locator"));
    let mut expected_build_arguments = template.args;
    expected_build_arguments.insert(4, "--no-run".to_string());
    assert_eq!(task_template.command, "cargo");
    assert_eq!(task_template.args, expected_build_arguments);

    let resolve_run = || proto::ResolveRustTestAction {
        project_id,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: discovery_generation.0,
        node_id: case_id.0.clone(),
        action: proto::RustTestActionKind::Run as i32,
        worktree_ids: vec![worktree_id_b.to_proto()],
        authorization_generation: 0,
    };
    let member_authorization = client_b
        .client()
        .request(resolve_run())
        .await
        .expect("member receives authorization-bound upstream plan")
        .authorization_generation;
    assert!(member_authorization > 0);
    let lifecycle = |state: proto::RustTestTaskState| proto::UpdateRustTestRun {
        project_id,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: discovery_generation.0,
        run_id: run_id.0.clone(),
        scope_node_ids: vec![case_id.0.clone()],
        task_id: "two-hop-planning-lifecycle".into(),
        state: state as i32,
        exit_code: (state == proto::RustTestTaskState::Completed).then_some(0),
        success: state == proto::RustTestTaskState::Completed,
        worktree_ids: vec![worktree_id_b.to_proto()],
        authorization_generation: member_authorization,
        ..Default::default()
    };
    client_c
        .client()
        .request(lifecycle(proto::RustTestTaskState::Queued))
        .await
        .expect_err("another member cannot consume the upstream owner's Run grant");
    for role in [proto::ChannelRole::Guest, proto::ChannelRole::Member] {
        active_call_a
            .update(cx_a, |call, cx| {
                call.room()
                    .expect("shared host room")
                    .update(cx, |room, cx| {
                        room.set_participant_role(
                            client_c.user_id().expect("other member identity"),
                            role,
                            cx,
                        )
                    })
            })
            .await
            .expect("change only the other member's authorization generation");
        executor.run_until_parked();
    }
    let other_plan = client_c
        .client()
        .request(resolve_run())
        .await
        .expect("restored other member resolves its own upstream Run");
    assert!(other_plan.authorization_generation > member_authorization);
    for state in [
        proto::RustTestTaskState::Queued,
        proto::RustTestTaskState::Completed,
    ] {
        client_b.client().request(lifecycle(state)).await.expect(
            "another member's changed generation preserves the original upstream Run grant",
        );
    }
    client_b
        .client()
        .request(lifecycle(proto::RustTestTaskState::Queued))
        .await
        .expect_err("completed upstream Run grant cannot be reused");
    let recorded = client_b
        .client()
        .request(proto::GetStructuredExecutionSnapshot {
            project_id,
            provider_id: rust_provider.0.clone(),
            discovery_generation: discovery_generation.0,
            page_start: 0,
            page_size: 10,
            worktree_ids: vec![worktree_id_b.to_proto()],
        })
        .await
        .expect("read authoritative synthetic lifecycle result through both hops");
    assert_eq!(
        recorded
            .last_complete_run
            .as_ref()
            .map(|run| run.run_id.as_str()),
        Some(run_id.0.as_str())
    );

    let bridge_rust_tests = project_a.read_with(cx_a, |project, _| {
        project.rust_test_provider_store().clone()
    });
    // Trust restoration below schedules discovery; isolate departure from that invalidation.
    if matches!(scenario, SeparateEnabledHostScenario::ActionOwnerLeave) {
        let other_project = other_project.expect("retain the initial unrelated Project");
        let upstream = remote_client.read_with(cx_a, |remote, _| remote.proto_client());
        let discovery_snapshot = || proto::GetStructuredExecutionSnapshot {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            provider_id: rust_provider.0.clone(),
            discovery_generation: discovery_generation.0,
            page_start: 0,
            page_size: 10,
            worktree_ids: vec![worktree_id_a.to_proto()],
        };
        let original_member_peer = client_b.peer_id().expect("action owner peer");
        let original_host_peer = client_a.peer_id().expect("bridge peer");
        let original_other_peer = client_c.peer_id().expect("unrelated member peer");
        let mut member_project = Some(project_b);
        for action in [
            proto::RustTestActionKind::Run,
            proto::RustTestActionKind::Debug,
        ] {
            let before_leave = upstream
                .request(discovery_snapshot())
                .await
                .expect("read current authoritative discovery before departure");
            assert_eq!(before_leave.discovery_generation, discovery_generation.0);
            assert_eq!(
                before_leave.status,
                proto::StructuredProviderStatus::Current as i32
            );
            let departed_plan = client_b
                .client()
                .request(resolve_run())
                .await
                .expect("owner holds a fresh Run grant before departure");
            let departed_plan_json: serde_json::Value =
                serde_json::from_str(&departed_plan.plan_json).expect("departing member Task plan");
            let departed_run_id = departed_plan_json
                .get("run_id")
                .and_then(serde_json::Value::as_str)
                .expect("departing member Run ID")
                .to_string();
            let protected_plan = client_c
                .client()
                .request(resolve_run())
                .await
                .expect("other member holds a fresh authorization-bound Run grant");
            let protected_plan_json: serde_json::Value =
                serde_json::from_str(&protected_plan.plan_json).expect("other member Task plan");
            let protected_run_id = protected_plan_json
                .get("run_id")
                .and_then(serde_json::Value::as_str)
                .expect("other member Run ID")
                .to_string();
            let weak_member_project = member_project
                .as_ref()
                .expect("joined action owner Project")
                .downgrade();
            let (reply_started, reply_release) =
                bridge_rust_tests.update(cx_a, |store, _| store.gate_next_action_reply_for_test());
            let pending_action = client_b.client().request(proto::ResolveRustTestAction {
                action: action as i32,
                ..resolve_run()
            });
            let leave_before_publication = async {
                assert!(matches!(
                    futures::future::select(
                        reply_started,
                        executor.timer(std::time::Duration::from_secs(10))
                    )
                    .await,
                    futures::future::Either::Left((Ok(()), _))
                ));
                drop(
                    member_project
                        .take()
                        .expect("release actual member Project"),
                );
                // Entity release callbacks run when the owning App finishes an update.
                cx_b.update(|_| {});
                executor.run_until_parked();
                assert!(weak_member_project.upgrade().is_none());
                let error = rust_tests
                    .update(cx_b, |store, cx| store.refresh(cx))
                    .await
                    .expect_err(
                        "a retained provider cannot discover after its Project is released",
                    );
                assert!(
                    error.to_string().contains("host is disconnected"),
                    "{error:#}"
                );
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while project_a.read_with(cx_a, |project, _| {
                    project.collaborators().contains_key(&original_member_peer)
                }) {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "owning Project release did not deliver LeaveProject"
                    );
                    executor.timer(std::time::Duration::from_millis(20)).await;
                }
                executor.run_until_parked();
                assert!(
                    reply_release.send(()).is_err(),
                    "LeaveProject must dispose the held upstream action reply"
                );
                assert_eq!(client_b.peer_id(), Some(original_member_peer));
                assert_eq!(client_a.peer_id(), Some(original_host_peer));
                assert_eq!(client_c.peer_id(), Some(original_other_peer));
                assert!(project_a.read_with(cx_a, |project, _| {
                    project.collaborators().contains_key(&original_other_peer)
                }));
            };
            let (obsolete_action, ()) = futures::join!(pending_action, leave_before_publication);
            obsolete_action.expect_err("the departed Project cannot receive its old action plan");
            let after_leave = upstream
                .request(discovery_snapshot())
                .await
                .expect("read authoritative discovery after owner departure");
            assert_eq!(
                after_leave.discovery_generation,
                before_leave.discovery_generation
            );
            assert_eq!(
                after_leave.status,
                proto::StructuredProviderStatus::Current as i32
            );
            // Room membership still permits rejoining; departure must retire this Project's grant.
            client_b
                .client()
                .request(proto::UpdateRustTestRun {
                    run_id: departed_run_id,
                    task_id: format!("departed-owner-{}", action as i32),
                    authorization_generation: departed_plan.authorization_generation,
                    ..lifecycle(proto::RustTestTaskState::Queued)
                })
                .await
                .expect_err("the departed Project cannot reuse its old Run grant");
            for state in [
                proto::RustTestTaskState::Queued,
                proto::RustTestTaskState::Completed,
            ] {
                client_c
                    .client()
                    .request(proto::UpdateRustTestRun {
                        run_id: protected_run_id.clone(),
                        task_id: format!("other-member-owner-leave-{}", action as i32),
                        authorization_generation: protected_plan.authorization_generation,
                        ..lifecycle(state)
                    })
                    .await
                    .expect(
                        "unrelated member retains its existing Run grant after owner departure",
                    );
            }
            member_project = Some(client_b.join_remote_project(project_id, cx_b).await);
            client_b
                .client()
                .request(proto::ResolveRustTestAction {
                    action: action as i32,
                    ..resolve_run()
                })
                .await
                .expect("a fresh Project join restores current Run or Debug planning");
            assert_eq!(client_b.peer_id(), Some(original_member_peer));
            assert_eq!(client_a.peer_id(), Some(original_host_peer));
            assert_eq!(client_c.peer_id(), Some(original_other_peer));
        }
        assert_eq!(
            other_project.read_with(cx_c, |project, _| project.remote_id()),
            Some(project_id)
        );
        assert_eq!(
            project::cargo_workspace_store::local_cargo_process_calls_for_test(),
            local_cargo_calls_before
        );
        assert_eq!(
            project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
            local_discovery_calls_before
        );
        assert_eq!(
            project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
            local_coverage_interpret_calls_before
        );
        return;
    }

    let (action_started, action_release) =
        bridge_rust_tests.update(cx_a, |store, _| store.gate_next_action_reply_for_test());
    let pending_action = client_b.client().request(resolve_run());
    let restrict_and_restore = async {
        assert!(matches!(
            futures::future::select(
                action_started,
                executor.timer(std::time::Duration::from_secs(5))
            )
            .await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_a.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("bridge trust service")
                .update(cx, |trust, cx| {
                    trust.restrict(
                        worktree_store_a.downgrade(),
                        std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                            worktree_id_a,
                        ))
                        .collect(),
                        cx,
                    );
                    trust.trust(
                        &worktree_store_a,
                        std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                            worktree_id_a,
                        ))
                        .collect(),
                        cx,
                    );
                });
        });
        executor.run_until_parked();
        assert!(
            action_release.send(()).is_err(),
            "restriction must dispose the held upstream action reply even after re-trust"
        );
    };
    let (obsolete_action, ()) = futures::join!(pending_action, restrict_and_restore);
    let error = obsolete_action.expect_err("re-trust cannot revive the held upstream Run plan");
    assert!(
        error
            .to_string()
            .contains("remote Rust test action authorization expired"),
        "{error:#}"
    );
    client_b
        .client()
        .request(resolve_run())
        .await
        .expect("fresh upstream Run planning recovers after re-trust");
    let collab_client: rpc::AnyProtoClient = client_b.client().clone().into();
    let error = collab_client
        .forward_request(
            client_c.peer_id().expect("other member peer"),
            resolve_run(),
        )
        .await
        .expect_err("a collaboration client cannot forge a forwarded peer identity");
    assert!(
        error.to_string().contains("Cannot forward a peer identity"),
        "{error:#}"
    );

    let coverage_path = project_b.read_with(cx_b, |project, cx| {
        project
            .find_project_path(fixture.path().join("coverage.json"), cx)
            .expect("scanned coverage artifact")
    });
    let coverage = project_b.read_with(cx_b, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let source_coverage =
        project_b.read_with(cx_b, |project, _| project.source_coverage_store().clone());
    if coverage_visibility_only {
        source_coverage.read_with(cx_b, |store, _| {
            assert!(store.selected_snapshot().is_none());
        });
        let (first_coverage_started, first_coverage_release) =
            coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
        let pending_first_coverage = coverage.update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        let revoke_restore_first = async {
            assert!(matches!(
                futures::future::select(
                    first_coverage_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ));
            cx_b.update(|cx| {
                project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                    .expect("coverage client trust service")
                    .update(cx, |trust, cx| {
                        let paths = std::iter::once(
                            project::trusted_worktrees::PathTrust::Worktree(worktree_id_b),
                        )
                        .collect();
                        trust.restrict(worktree_store_b.downgrade(), paths, cx);
                        trust.trust(
                            &worktree_store_b,
                            std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                                worktree_id_b,
                            ))
                            .collect(),
                            cx,
                        );
                    });
            });
            executor.run_until_parked();
            first_coverage_release
                .send(())
                .expect("release the first completed report after client re-trust");
        };
        let (obsolete_first, ()) = futures::join!(pending_first_coverage, revoke_restore_first);
        obsolete_first
            .expect_err("client re-trust cannot revive the completed pre-revocation first report");
        source_coverage.read_with(cx_b, |store, _| {
            assert!(store.selected_snapshot().is_none());
        });
    }
    if coverage_visibility_only {
        let source_path = project_b.read_with(cx_b, |project, cx| {
            project
                .find_project_path(fixture.path().join("workload.rs"), cx)
                .expect("first-report source path")
        });
        let original_version = project_b.read_with(cx_b, |project, cx| {
            let entry = project
                .entry_for_path(&source_path, cx)
                .expect("initial source entry");
            (entry.id, entry.inode, entry.mtime)
        });
        source_coverage.read_with(cx_b, |store, _| {
            assert!(store.selected_snapshot().is_none())
        });
        let (source_started, source_release) =
            coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
        let pending_source = coverage.update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        let replace_before_publication = async {
            assert!(matches!(
                futures::future::select(
                    source_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ));
            let log_source_version = |phase: &str| {
                use std::os::unix::fs::MetadataExt as _;
                let observe_project =
                    |project: &gpui::Entity<project::Project>, context: &TestAppContext| {
                        project.read_with(context, |project, cx| {
                            let entry = project
                                .find_project_path(fixture.path().join("workload.rs"), cx)
                                .and_then(|path| project.entry_for_path(&path, cx))
                                .map(|entry| (entry.id, entry.inode, entry.mtime, entry.size));
                            let scan = project.worktrees(cx).next().map(|tree| {
                                let tree = tree.read(cx);
                                (
                                    tree.id(),
                                    tree.abs_path(),
                                    tree.scan_id(),
                                    tree.completed_scan_id(),
                                )
                            });
                            (entry, scan, project.is_disconnected(cx))
                        })
                    };
                let disk = std::fs::metadata(fixture.path().join("workload.rs"))
                    .map(|metadata| {
                        (
                            metadata.ino(),
                            metadata.mtime(),
                            metadata.mtime_nsec(),
                            metadata.len(),
                        )
                    })
                    .map_err(|error| error.kind());
                let canonical_root =
                    std::fs::canonicalize(fixture.path()).map_err(|error| error.kind());
                let canonical_source = std::fs::canonicalize(fixture.path().join("workload.rs"))
                    .map_err(|error| error.kind());
                eprintln!(
                    "coverage-source-watch phase={phase} root={:?} canonical_root={canonical_root:?} canonical_source={canonical_source:?} original={original_version:?} disk={disk:?} bridge={:?} client={:?}",
                    fixture.path(),
                    observe_project(&project_a, &*cx_a),
                    observe_project(&project_b, &*cx_b),
                );
            };
            log_source_version("before-disk-replacement");
            let original_source = fixture.path().join("workload.rs");
            let backup_source = fixture.path().join("coverage-original-source");
            std::fs::rename(&original_source, &backup_source)
                .expect("preserve original fixture source");
            std::fs::write(
                &original_source,
                "fn replaced_source() { unreachable!(); }\n",
            )
            .expect("replace temporary host source content");
            log_source_version("after-disk-replacement");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while project_b.read_with(cx_b, |project, cx| {
                project
                    .entry_for_path(&source_path, cx)
                    .is_none_or(|entry| (entry.id, entry.inode, entry.mtime) == original_version)
            }) {
                if std::time::Instant::now() >= deadline {
                    log_source_version("replacement-deadline");
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "replacement source version did not reach client"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
            log_source_version("replacement-observed");
            source_release
                .send(())
                .expect("release first reply after source replacement");
        };
        let (obsolete_source, ()) = futures::join!(pending_source, replace_before_publication);
        obsolete_source
            .expect_err("the completed first report cannot publish for replaced source content");
        source_coverage.read_with(cx_b, |store, _| {
            assert!(store.selected_snapshot().is_none())
        });
        let bridge_source_coverage =
            project_a.read_with(cx_a, |project, _| project.source_coverage_store().clone());
        let refreshed_bridge = bridge_source_coverage
            .update(cx_a, |store, cx| {
                store.snapshot(
                    SourceCoverageProviderId(
                        project::rust_coverage_provider::RUST_COVERAGE_PROVIDER_ID.into(),
                    ),
                    cx,
                )
            })
            .await
            .expect("fresh generic snapshot after source replacement");
        assert_eq!(
            refreshed_bridge.status,
            project::source_coverage::SourceCoverageStatus::Stale,
            "a fresh generic query cannot revive the authoritative pre-edit report: {refreshed_bridge:?}"
        );
        let refreshed_member = source_coverage
            .update(cx_b, |store, cx| {
                store.snapshot(refreshed_bridge.provider_id.clone(), cx)
            })
            .await
            .expect("collaboration snapshot after source replacement");
        assert_eq!(
            refreshed_member.status,
            project::source_coverage::SourceCoverageStatus::Stale
        );
        let original_source = fixture.path().join("workload.rs");
        std::fs::remove_file(&original_source).expect("remove temporary replacement source");
        std::fs::rename(
            fixture.path().join("coverage-original-source"),
            &original_source,
        )
        .expect("restore original fixture source");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while project_b.read_with(cx_b, |project, cx| {
            project
                .entry_for_path(&source_path, cx)
                .is_none_or(|entry| entry.mtime != original_version.2)
        }) {
            assert!(
                std::time::Instant::now() < deadline,
                "restored source version did not reach client"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
    }
    let report = coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("collaborator receives enabled host coverage");
    assert_eq!(report.files.len(), 1);
    assert_eq!(report.files[0].path.path.as_unix_str(), "workload.rs");
    let member_rejoined = Arc::new(Mutex::new(false));
    let _coverage_rejoin_subscription = cx_b.update(|cx| {
        let member_rejoined = member_rejoined.clone();
        cx.subscribe(&project_b, move |_, event, _| {
            if matches!(event, project::Event::Rejoined) {
                *member_rejoined
                    .lock()
                    .expect("coverage member rejoin observation") = true;
            }
        })
    });
    let original_member_peer = client_b
        .peer_id()
        .expect("coverage member before reconnect");
    let original_host_peer = client_a
        .peer_id()
        .expect("coverage bridge before member reconnect");
    let original_other_peer = client_c
        .peer_id()
        .expect("other member before coverage reconnect");
    let (coverage_started, release_coverage) =
        coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
    let pending_coverage = coverage.update(cx_b, |store, cx| {
        store.interpret_artifact(
            coverage_path.clone(),
            project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
            cx,
        )
    });
    let reconnect_and_release = async {
        assert!(
            matches!(
                futures::future::select(
                    coverage_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "client did not hold the completed authoritative coverage reply"
        );
        server.disconnect_client(original_member_peer);
        let step = std::time::Duration::from_millis(100);
        let mut remaining = rpc::RECEIVE_TIMEOUT;
        loop {
            executor.run_until_parked();
            if client_b.peer_id().is_some_and(|peer| {
                peer != original_member_peer
                    && *member_rejoined
                        .lock()
                        .expect("coverage member rejoin observation")
                    && project_a.read_with(cx_a, |project, _| {
                        project.collaborators().contains_key(&peer)
                    })
            }) {
                break;
            }
            assert!(
                remaining >= step,
                "coverage member did not rejoin the same shared Project"
            );
            remaining = remaining.saturating_sub(step);
            executor.advance_clock(step);
            smol::future::yield_now().await;
        }
        assert_eq!(client_a.peer_id(), Some(original_host_peer));
        assert_eq!(client_c.peer_id(), Some(original_other_peer));
        release_coverage
            .send(())
            .expect("release obsolete coverage reply after actual member rejoin");
    };
    let (obsolete, ()) = futures::join!(pending_coverage, reconnect_and_release);
    obsolete.expect_err(
        "coverage reply from the departed member session must not publish after rejoin",
    );
    source_coverage.read_with(cx_b, |store, _| {
        let snapshot = store
            .selected_snapshot()
            .expect("retained coverage provider after rejoin");
        assert_eq!(
            snapshot.status,
            project::source_coverage::SourceCoverageStatus::Stale
        );
    });
    let recovered = coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("fresh authorized coverage interpretation recovers after member rejoin");
    assert_eq!(recovered.files.len(), 1);
    assert_eq!(
        recovered.status,
        project::source_coverage::SourceCoverageStatus::Current
    );
    let offline_member_peer = client_b.peer_id().expect("coverage member before offline");
    *member_rejoined
        .lock()
        .expect("coverage offline rejoin observation") = false;
    let (offline_coverage_started, offline_coverage_release) =
        coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
    let pending_offline_coverage = coverage.update(cx_b, |store, cx| {
        store.interpret_artifact(
            coverage_path.clone(),
            project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
            cx,
        )
    });
    let disconnect_and_release = async {
        assert!(
            matches!(
                futures::future::select(
                    offline_coverage_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "client did not hold the completed coverage reply before offline"
        );
        server.forbid_connections();
        server.disconnect_client(offline_member_peer);
        let step = std::time::Duration::from_millis(100);
        let mut remaining = rpc::RECEIVE_TIMEOUT;
        loop {
            executor.run_until_parked();
            if !client_b.client().status().borrow().is_connected() {
                break;
            }
            assert!(remaining >= step, "coverage member did not become offline");
            remaining = remaining.saturating_sub(step);
            executor.advance_clock(step);
            smol::future::yield_now().await;
        }
        assert!(
            !*member_rejoined
                .lock()
                .expect("coverage offline rejoin observation")
        );
        assert_eq!(client_a.peer_id(), Some(original_host_peer));
        assert_eq!(client_c.peer_id(), Some(original_other_peer));
        offline_coverage_release
            .send(())
            .expect("release completed coverage while the requester remains offline");
    };
    let (obsolete_offline, ()) = futures::join!(pending_offline_coverage, disconnect_and_release);
    obsolete_offline.expect_err(
        "completed coverage must not publish during the offline interval before rejoin",
    );
    source_coverage.read_with(cx_b, |store, _| {
        assert_eq!(
            store
                .selected_snapshot()
                .expect("retained offline coverage")
                .status,
            project::source_coverage::SourceCoverageStatus::Disconnected
        );
    });
    server.allow_connections();
    let step = std::time::Duration::from_millis(100);
    let mut remaining = rpc::RECEIVE_TIMEOUT;
    loop {
        executor.run_until_parked();
        if client_b.peer_id().is_some_and(|peer| {
            peer != offline_member_peer
                && *member_rejoined
                    .lock()
                    .expect("coverage offline recovery rejoin observation")
                && project_a.read_with(cx_a, |project, _| {
                    project.collaborators().contains_key(&peer)
                })
        }) {
            break;
        }
        assert!(remaining >= step, "offline coverage member did not rejoin");
        remaining = remaining.saturating_sub(step);
        executor.advance_clock(step);
        smol::future::yield_now().await;
    }
    source_coverage.read_with(cx_b, |store, _| {
        assert_eq!(
            store
                .selected_snapshot()
                .expect("coverage after offline recovery")
                .status,
            project::source_coverage::SourceCoverageStatus::Stale
        );
    });
    let offline_recovered = coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("fresh authorized coverage recovers after the offline interval");
    assert_eq!(offline_recovered.files.len(), 1);
    assert_eq!(
        offline_recovered.status,
        project::source_coverage::SourceCoverageStatus::Current
    );
    let (role_coverage_started, role_coverage_release) =
        coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
    let pending_role_coverage = coverage.update(cx_b, |store, cx| {
        store.interpret_artifact(
            coverage_path.clone(),
            project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
            cx,
        )
    });
    let revoke_restore_and_release = async {
        assert!(
            matches!(
                futures::future::select(
                    role_coverage_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "client did not hold the completed coverage reply before role loss"
        );
        for role in [proto::ChannelRole::Guest, proto::ChannelRole::Member] {
            active_call_a
                .update(cx_a, |call, cx| {
                    call.room()
                        .expect("coverage permission room")
                        .update(cx, |room, cx| {
                            room.set_participant_role(
                                client_b.user_id().expect("coverage member identity"),
                                role,
                                cx,
                            )
                        })
                })
                .await
                .expect("change coverage requester's permission");
            executor.run_until_parked();
            assert_eq!(
                project_b.read_with(cx_b, |project, cx| project.is_read_only(cx)),
                role == proto::ChannelRole::Guest
            );
        }
        role_coverage_release
            .send(())
            .expect("release obsolete coverage reply after role restoration");
    };
    let (obsolete_role, ()) = futures::join!(pending_role_coverage, revoke_restore_and_release);
    obsolete_role.expect_err(
        "role restoration cannot authorize the already completed pre-revocation coverage reply",
    );
    source_coverage.read_with(cx_b, |store, _| {
        assert_eq!(
            store
                .selected_snapshot()
                .expect("coverage provider after role restoration")
                .status,
            project::source_coverage::SourceCoverageStatus::Stale
        )
    });
    let role_recovered = coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("fresh authorized coverage after role restoration");
    assert_eq!(role_recovered.files.len(), 1);
    assert_eq!(
        role_recovered.status,
        project::source_coverage::SourceCoverageStatus::Current
    );
    let (trust_coverage_started, trust_coverage_release) =
        coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
    let pending_trust_coverage = coverage.update(cx_b, |store, cx| {
        store.interpret_artifact(
            coverage_path.clone(),
            project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
            cx,
        )
    });
    let restrict_before_publication = async {
        assert!(
            matches!(
                futures::future::select(
                    trust_coverage_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ),
            "client did not hold the completed coverage reply before trust loss"
        );
        cx_b.update(|cx| {
            project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                .expect("coverage client trust service")
                .update(cx, |trust, cx| {
                    trust.restrict(
                        worktree_store_b.downgrade(),
                        std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                            worktree_id_b,
                        ))
                        .collect(),
                        cx,
                    );
                });
        });
        executor.run_until_parked();
        cx_b.update(|cx| {
            assert!(
                project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &worktree_store_b,
                    cx,
                )
            );
        });
        trust_coverage_release
            .send(())
            .expect("release completed coverage while the client worktree is restricted");
    };
    let (restricted_coverage, ()) =
        futures::join!(pending_trust_coverage, restrict_before_publication);
    restricted_coverage.expect_err(
        "client trust loss expires the completed interpretation instead of publishing its facts",
    );
    source_coverage.read_with(cx_b, |store, _| {
        let snapshot = store
            .selected_snapshot()
            .expect("cached restricted coverage");
        assert!(snapshot.files.is_empty());
        assert_eq!(
            snapshot.status,
            project::source_coverage::SourceCoverageStatus::Restricted
        );
    });
    cx_b.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("coverage client trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store_b,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                        worktree_id_b,
                    ))
                    .collect(),
                    cx,
                );
            });
    });
    executor.run_until_parked();
    let trust_recovered = coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("fresh authorized coverage recovers after client re-trust");
    assert_eq!(trust_recovered.files.len(), 1);
    assert_eq!(
        trust_recovered.status,
        project::source_coverage::SourceCoverageStatus::Current
    );
    if coverage_visibility_only {
        let (cached_started, cached_release) =
            coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
        let pending_cached = coverage.update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        let restrict_restore_cached = async {
            assert!(matches!(
                futures::future::select(
                    cached_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ));
            cx_b.update(|cx| {
                project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
                    .expect("cached coverage client trust service")
                    .update(cx, |trust, cx| {
                        trust.restrict(
                            worktree_store_b.downgrade(),
                            std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                                worktree_id_b,
                            ))
                            .collect(),
                            cx,
                        );
                        trust.trust(
                            &worktree_store_b,
                            std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                                worktree_id_b,
                            ))
                            .collect(),
                            cx,
                        );
                    });
            });
            executor.run_until_parked();
            cached_release
                .send(())
                .expect("release cached trust ABA reply");
        };
        let (obsolete_cached, ()) = futures::join!(pending_cached, restrict_restore_cached);
        obsolete_cached.expect_err("re-trust cannot revive the cached pre-revocation report");
        source_coverage.read_with(cx_b, |store, _| {
            let snapshot = store.selected_snapshot().expect("cached trust transition");
            assert!(snapshot.files.is_empty());
            assert_eq!(
                snapshot.status,
                project::source_coverage::SourceCoverageStatus::Stale
            );
        });
        let recovered_cached = coverage
            .update(cx_b, |store, cx| {
                store.interpret_artifact(
                    coverage_path.clone(),
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .expect("fresh coverage after cached trust restoration");
        assert_eq!(recovered_cached.files.len(), 1);
        assert_eq!(
            recovered_cached.status,
            project::source_coverage::SourceCoverageStatus::Current
        );
        let withdrawn_source = fixture.path().join("workload.rs");
        let source_contents = std::fs::read(&withdrawn_source).expect("preserve fixture source");
        let source_path_b = trust_recovered
            .files
            .first()
            .expect("recovered coverage fixture source")
            .path
            .clone();
        let (visibility_coverage_started, visibility_coverage_release) =
            coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
        let pending_visibility_coverage = coverage.update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        let withdraw_before_publication = async {
            assert!(
                matches!(
                    futures::future::select(
                        visibility_coverage_started,
                        executor.timer(std::time::Duration::from_secs(10))
                    )
                    .await,
                    futures::future::Either::Left((Ok(()), _))
                ),
                "client did not hold the completed coverage reply before source removal"
            );
            std::fs::remove_file(&withdrawn_source).expect("remove temporary host fixture source");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while project_b.read_with(cx_b, |project, cx| {
                project.entry_for_path(&source_path_b, cx).is_some()
            }) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "authoritative source removal did not reach the collaboration client"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
            visibility_coverage_release
                .send(())
                .expect("release completed coverage after actual source removal");
        };
        let (withdrawn_coverage, ()) =
            futures::join!(pending_visibility_coverage, withdraw_before_publication);
        withdrawn_coverage.expect_err("source withdrawal expires the held interpretation");
        source_coverage.read_with(cx_b, |store, _| {
            let snapshot = store
                .selected_snapshot()
                .expect("cached withdrawn-source coverage");
            assert_eq!(
                snapshot.status,
                project::source_coverage::SourceCoverageStatus::Stale
            );
            assert!(snapshot.files.is_empty());
            assert!(snapshot.truncated);
        });
        std::fs::write(&withdrawn_source, source_contents).expect("restore temporary host source");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while project_b.read_with(cx_b, |project, cx| {
            project.entry_for_path(&source_path_b, cx).is_none()
        }) {
            assert!(
                std::time::Instant::now() < deadline,
                "restored authoritative source did not reach the collaboration client"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
        let visibility_recovered = coverage
            .update(cx_b, |store, cx| {
                store.interpret_artifact(
                    coverage_path.clone(),
                    project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                    cx,
                )
            })
            .await
            .expect("fresh authorized coverage recovers after host source restoration");
        assert_eq!(visibility_recovered.files.len(), 1);
        assert_eq!(
            visibility_recovered.status,
            project::source_coverage::SourceCoverageStatus::Current
        );
    }

    assert_eq!(
        project::cargo_workspace_store::local_cargo_process_calls_for_test(),
        local_cargo_calls_before,
        "two-hop Cargo must not run locally"
    );
    assert_eq!(
        project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
        local_discovery_calls_before,
        "two-hop discovery must not run locally"
    );
    assert_eq!(
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
        local_coverage_interpret_calls_before,
        "two-hop coverage must not be interpreted locally"
    );

    if coverage_visibility_only {
        let weak_project = project_b.downgrade();
        let departing_peer = client_b.peer_id().expect("coverage project owner peer");
        let (close_started, close_release) =
            coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
        let pending_close = coverage.update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        });
        let close_project_before_publication = async {
            assert!(matches!(
                futures::future::select(
                    close_started,
                    executor.timer(std::time::Duration::from_secs(10))
                )
                .await,
                futures::future::Either::Left((Ok(()), _))
            ));
            drop(project_b);
            executor.run_until_parked();
            assert!(
                weak_project.upgrade().is_none(),
                "release the actual client Project"
            );
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while project_a.read_with(cx_a, |project, _| {
                project.collaborators().contains_key(&departing_peer)
            }) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "Project release did not leave the shared project"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
            assert_eq!(client_b.peer_id(), Some(departing_peer));
            assert_eq!(client_a.peer_id(), Some(original_host_peer));
            assert_eq!(client_c.peer_id(), Some(original_other_peer));
            source_coverage.read_with(cx_b, |store, _| {
                assert_eq!(
                    store
                        .selected_snapshot()
                        .expect("coverage before late release")
                        .status,
                    project::source_coverage::SourceCoverageStatus::Disconnected,
                );
            });
            close_release
                .send(())
                .expect("release the completed reply after Project teardown");
        };
        let (obsolete_close, ()) = futures::join!(pending_close, close_project_before_publication);
        obsolete_close
            .expect_err("a completed interpretation cannot publish after Project teardown");
        source_coverage.read_with(cx_b, |store, _| {
            assert_eq!(
                store
                    .selected_snapshot()
                    .expect("closed coverage cache")
                    .status,
                project::source_coverage::SourceCoverageStatus::Disconnected
            );
        });
        return;
    }

    let profile_paths = project_b.read_with(cx_b, |project, cx| {
        [
            project
                .find_project_path(fixture.path().join("profile.json"), cx)
                .expect("scanned profile artifact"),
            project
                .find_project_path(fixture.path().join("profile.syms.json"), cx)
                .expect("scanned profile symbols"),
        ]
    });
    let local_profile_loads_before = project::profile_results::samply_profile_load_calls_for_test();
    let import_id = 945;
    let first = client_b
        .client()
        .request(proto::ImportProfileResults {
            project_id,
            import_id,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_paths[0].to_proto()),
            symbols_path: Some(profile_paths[1].to_proto()),
            consent_to_transfer: true,
            artifact_format: "samply-0.13.1".into(),
            task_run_id: None,
        })
        .await
        .expect("consented import reaches separate enabled host");
    assert_eq!(first.status, proto::ProfileResultStatus::Ready as i32);
    assert!(!first.complete, "large profile retains another page");
    assert_eq!(first.import_id, import_id);
    let page_request = |page_index| proto::GetProfileResultPage {
        project_id,
        import_id,
        generation: first.generation,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        page_index,
    };
    let foreign = client_c
        .client()
        .request(page_request(1))
        .await
        .expect("foreign member receives typed denial");
    assert_eq!(foreign.status, proto::ProfileResultStatus::Stale as i32);
    assert!(foreign.frames.is_empty() && foreign.nodes.is_empty());
    client_c
        .client()
        .request(proto::CancelProfileImport {
            project_id,
            import_id,
            generation: first.generation,
        })
        .await
        .expect("foreign cancellation is acknowledged");
    let owner = client_b
        .client()
        .request(page_request(1))
        .await
        .expect("original importer retains the next page");
    assert_eq!(owner.status, proto::ProfileResultStatus::Ready as i32);
    client_b
        .client()
        .request(proto::CancelProfileImport {
            project_id,
            import_id,
            generation: first.generation,
        })
        .await
        .expect("owner cancels retained page stream");
    let cancelled = client_b
        .client()
        .request(page_request(2))
        .await
        .expect("cancelled page has typed status");
    assert_eq!(cancelled.status, proto::ProfileResultStatus::Stale as i32);

    let model = project_b
        .read_with(cx_b, |project, cx| {
            project.import_remote_profile(
                profile_paths.clone(),
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("production Project starts consented two-hop import")
        .await
        .expect("production Project assembles the enabled host profile");
    assert_eq!(model.total_samples, 3_030);

    let (workspace_b, window_cx_b) = client_b.build_workspace(&project_b, cx_b);
    workspace_b.update_in(window_cx_b, |workspace, window, cx| {
        window.focus(&workspace.focus_handle(cx), cx);
    });
    window_cx_b.update(|window, cx| window.draw(cx).clear(cx));
    window_cx_b.dispatch_action(language_tools::profile_view::OpenNativeProfile);
    window_cx_b.run_until_parked();
    let view = workspace_b
        .read_with(window_cx_b, |workspace, cx| {
            workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
        })
        .expect("profile action opens collaboration viewer");
    view.update(window_cx_b, |view, cx| {
        view.import_paths_with_consent_for_test(
            vec![
                fixture.path().join("profile.json"),
                fixture.path().join("profile.syms.json"),
            ],
            true,
            cx,
        )
        .expect("explicit collaborator consent starts native import");
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while view.read_with(window_cx_b, |view, _| view.sample_count_for_test()) != Some(3_030) {
        assert!(
            std::time::Instant::now() < deadline,
            "two-hop collaboration viewer did not display the normalized profile"
        );
        executor.timer(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(
        project::profile_results::samply_profile_load_calls_for_test(),
        local_profile_loads_before,
        "the collaborator and owner bridge must not parse the host profile locally"
    );

    let import = |import_id, consent_to_transfer| proto::ImportProfileResults {
        project_id,
        import_id,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        profile_path: Some(profile_paths[0].to_proto()),
        symbols_path: Some(profile_paths[1].to_proto()),
        consent_to_transfer,
        artifact_format: "samply-0.13.1".into(),
        task_run_id: None,
    };
    let denied = client_b
        .client()
        .request(import(946, false))
        .await
        .expect("missing consent receives typed denial");
    assert_eq!(denied.status, proto::ProfileResultStatus::Restricted as i32);
    assert!(denied.frames.is_empty() && denied.nodes.is_empty());

    let pending = client_b
        .client()
        .request(import(947, true))
        .await
        .expect("start a stream before source trust is revoked");
    assert_eq!(pending.status, proto::ProfileResultStatus::Ready as i32);
    assert!(!pending.complete);
    cx_a.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("host member trust service")
            .update(cx, |trust, cx| {
                trust.restrict(
                    worktree_store_a.downgrade(),
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                        worktree_id_a,
                    ))
                    .collect(),
                    cx,
                );
            });
    });
    executor.run_until_parked();
    let revoked = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 947,
            generation: pending.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("revoked stream receives typed status");
    assert_eq!(revoked.status, proto::ProfileResultStatus::Stale as i32);
    assert!(revoked.frames.is_empty() && revoked.nodes.is_empty());

    cx_a.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("host member trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &worktree_store_a,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                        worktree_id_a,
                    ))
                    .collect(),
                    cx,
                );
            });
    });
    executor.run_until_parked();
    let profile_store_a = project_a.read_with(cx_a, |project, _| {
        project
            .profile_results_store_for_test()
            .expect("owner bridge store")
            .clone()
    });
    let before_leave = client_b
        .client()
        .request(import(948, true))
        .await
        .expect("start a retained profile stream before collaborator leaves");
    assert_eq!(
        before_leave.status,
        proto::ProfileResultStatus::Ready as i32
    );
    assert!(!before_leave.complete);
    assert_eq!(
        profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()),
        1
    );
    cx_b.read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .expect("profile importer leaves the room");
    executor.run_until_parked();
    assert_eq!(
        profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()),
        0,
        "collaborator departure releases the retained upstream profile"
    );
    let member_id = client_b.current_user_id(cx_b).to_proto();
    active_call_a
        .update(cx_a, |call, cx| call.invite(member_id, None, cx))
        .await
        .expect("host reinvites the profile importer");
    cx_b.executor().run_until_parked();
    cx_b.read(ActiveCall::global)
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .expect("profile importer rejoins the room");
    let rejoined_project_b = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();
    assert_eq!(
        rejoined_project_b.read_with(cx_b, |project, _| project.remote_id()),
        Some(project_id)
    );
    let old_page = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 948,
            generation: before_leave.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("rejoined collaborator receives typed old-stream status");
    assert_eq!(old_page.status, proto::ProfileResultStatus::Stale as i32);
    assert!(old_page.frames.is_empty() && old_page.nodes.is_empty());

    let before_in_flight_leave = client_b
        .client()
        .request(import(949, true))
        .await
        .expect("start a new retained profile stream before an in-flight page");
    assert_eq!(
        before_in_flight_leave.status,
        proto::ProfileResultStatus::Ready as i32
    );
    assert!(!before_in_flight_leave.complete);
    assert_eq!(
        profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()),
        1
    );
    let importer_peer = client_b
        .client()
        .peer_id()
        .expect("connected profile importer");
    let (host_import_id, host_generation) = profile_store_a.read_with(cx_a, |store, _| {
        store
            .proxied_import_identity_for_test(importer_peer, 949)
            .expect("the pending collaboration import is retained upstream")
    });
    assert_eq!(host_generation, before_in_flight_leave.generation);
    let (page_started, page_started_receiver) = futures::channel::oneshot::channel();
    let (release_page, release_page_receiver) = futures::channel::oneshot::channel();
    profile_store_a.update(cx_a, |store, _| {
        store.gate_next_page_for_test(page_started, release_page_receiver);
    });
    let in_flight_page = client_b.client().request(proto::GetProfileResultPage {
        project_id,
        import_id: 949,
        generation: before_in_flight_leave.generation,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        page_index: 1,
    });
    let leave_and_release = async {
        page_started_receiver
            .await
            .expect("the collaboration bridge admitted the pending page");
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("profile importer leaves while a page is pending");
        executor.run_until_parked();
        release_page
            .send(())
            .expect("the old page handler is still waiting after departure");
    };
    let (old_in_flight_page, ()) = futures::join!(in_flight_page, leave_and_release);
    match old_in_flight_page {
        Ok(page) => {
            assert_eq!(page.status, proto::ProfileResultStatus::Stale as i32);
            assert!(page.frames.is_empty() && page.nodes.is_empty());
        }
        Err(error) => assert!(
            error.to_string().contains("authorization changed")
                || error
                    .to_string()
                    .contains("not authorized to edit projects"),
            "{error:#}"
        ),
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()) != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "authorization rejection did not cancel the old retained import"
        );
        executor.timer(std::time::Duration::from_millis(20)).await;
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !profile_store_a.read_with(cx_a, |store, _| {
        store.host_cancellation_acknowledged_for_test(host_import_id, host_generation)
    }) {
        assert!(
            std::time::Instant::now() < deadline,
            "authoritative host did not acknowledge cancellation of the retained import"
        );
        executor.timer(std::time::Duration::from_millis(20)).await;
    }
    let host_client =
        remote_client.read_with(cx_a, |remote_client, _| remote_client.proto_client());
    let host_old_page = host_client
        .request(proto::GetProfileResultPage {
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
            import_id: host_import_id,
            generation: host_generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("authoritative host answers the cancelled import identity");
    assert_eq!(
        host_old_page.status,
        proto::ProfileResultStatus::Stale as i32
    );
    assert!(host_old_page.frames.is_empty() && host_old_page.nodes.is_empty());
    active_call_a
        .update(cx_a, |call, cx| call.invite(member_id, None, cx))
        .await
        .expect("host reinvites the pending-page importer");
    cx_b.executor().run_until_parked();
    cx_b.read(ActiveCall::global)
        .update(cx_b, |call, cx| call.accept_incoming(cx))
        .await
        .expect("pending-page importer rejoins the room");
    let current_project_b = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();
    let old_page = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 949,
            generation: before_in_flight_leave.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("rejoined collaborator receives typed old in-flight stream status");
    assert_eq!(old_page.status, proto::ProfileResultStatus::Stale as i32);
    assert!(old_page.frames.is_empty() && old_page.nodes.is_empty());

    wait_for_scanned_relay_artifacts(
        &current_project_b,
        fixture.path(),
        &["profile.json", "profile.syms.json", "workload.rs"],
        &executor,
        cx_b,
    )
    .await;
    let current_worktree_store =
        current_project_b.read_with(cx_b, |project, _| project.worktree_store());
    let current_worktree_id = current_project_b.read_with(cx_b, |project, cx| {
        project
            .worktrees(cx)
            .next()
            .expect("current shared profile worktree")
            .read(cx)
            .id()
    });
    cx_b.update(|cx| {
        project::trusted_worktrees::TrustedWorktrees::try_get_global(cx)
            .expect("current importer trust service")
            .update(cx, |trust, cx| {
                trust.trust(
                    &current_worktree_store,
                    std::iter::once(project::trusted_worktrees::PathTrust::Worktree(
                        current_worktree_id,
                    ))
                    .collect(),
                    cx,
                );
            });
    });
    let current_metadata =
        worktree_store_a.read_with(cx_a, |store, cx| store.worktree_metadata_protos(cx));
    client_a
        .client()
        .request(proto::UpdateProject {
            project_id,
            worktrees: current_metadata.clone(),
            upstream_session_generation: Some(0),
        })
        .await
        .expect("host publishes initial upstream session metadata");
    executor.run_until_parked();
    let (current_workspace, current_window_cx) = client_b.build_workspace(&current_project_b, cx_b);
    current_workspace.update_in(current_window_cx, |workspace, window, cx| {
        window.focus(&workspace.focus_handle(cx), cx);
    });
    current_window_cx.update(|window, cx| window.draw(cx).clear(cx));
    current_window_cx.dispatch_action(language_tools::profile_view::OpenNativeProfile);
    let current_view = current_workspace
        .read_with(current_window_cx, |workspace, cx| {
            workspace.item_of_type::<language_tools::profile_view::ProfileView>(cx)
        })
        .expect("current rejoined profile view");
    current_view.update(current_window_cx, |view, cx| {
        view.import_paths_with_consent_for_test(
            vec![
                fixture.path().join("profile.json"),
                fixture.path().join("profile.syms.json"),
            ],
            true,
            cx,
        )
        .expect("load current guest view before authoritative process loss");
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while current_view.read_with(current_window_cx, |view, _| view.sample_count_for_test())
        != Some(3_030)
    {
        assert!(
            std::time::Instant::now() < deadline,
            "current guest view did not load before host process loss"
        );
        executor.timer(std::time::Duration::from_millis(20)).await;
    }
    let stable_member_peer = client_b.peer_id().expect("connected current importer");
    let stable_other_peer = client_c.peer_id().expect("connected other member");
    for generation in [Some(0), None] {
        client_a
            .client()
            .request(proto::UpdateProject {
                project_id,
                worktrees: current_metadata.clone(),
                upstream_session_generation: generation,
            })
            .await
            .expect("host repeats current or legacy metadata");
        executor.run_until_parked();
        assert_eq!(
            current_view.read_with(cx_b, |view, _| view.sample_count_for_test()),
            Some(3_030)
        );
    }
    for client in [&client_b, &client_c] {
        let denied = client
            .client()
            .request(proto::UpdateProject {
                project_id,
                worktrees: Vec::new(),
                upstream_session_generation: Some(u64::MAX),
            })
            .await
            .expect_err("a collaborator cannot forge host session metadata");
        assert!(denied.to_string().contains("no such project"), "{denied:#}");
    }
    executor.run_until_parked();
    assert_eq!(
        current_view.read_with(cx_b, |view, _| view.sample_count_for_test()),
        Some(3_030)
    );

    let current_coverage_path = current_project_b.read_with(cx_b, |project, cx| {
        project
            .find_project_path(fixture.path().join("coverage.json"), cx)
            .expect("current coverage artifact")
    });
    let current_coverage = current_project_b.read_with(cx_b, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let current_source_coverage =
        current_project_b.read_with(cx_b, |project, _| project.source_coverage_store().clone());
    let before_host_loss = current_coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                current_coverage_path.clone(),
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect("current coverage before authoritative host loss");
    assert_eq!(before_host_loss.files.len(), 1);
    assert_eq!(
        before_host_loss.status,
        project::source_coverage::SourceCoverageStatus::Current
    );
    let (upstream_coverage_started, upstream_coverage_release) =
        current_coverage.update(cx_b, |store, _| store.gate_next_remote_reply_for_test());
    let pending_upstream_coverage = current_coverage.update(cx_b, |store, cx| {
        store.interpret_artifact(
            current_coverage_path,
            project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
            cx,
        )
    });
    assert!(
        matches!(
            futures::future::select(
                upstream_coverage_started,
                executor.timer(std::time::Duration::from_secs(10))
            )
            .await,
            futures::future::Either::Left((Ok(()), _))
        ),
        "client did not hold the completed coverage reply before authoritative host loss"
    );

    let before_disconnect = client_b
        .client()
        .request(import(950, true))
        .await
        .expect("start a new retained profile stream after the in-flight session change");
    assert_eq!(
        before_disconnect.status,
        proto::ProfileResultStatus::Ready as i32
    );
    assert!(!before_disconnect.complete);
    assert_eq!(
        profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()),
        1
    );
    let (page_started_receiver, release_page) = if let Some(directory) = &host_page_gate_dir {
        std::fs::write(directory.join("armed"), b"").expect("arm authoritative host page gate");
        (None, None)
    } else {
        let (page_started, page_started_receiver) = futures::channel::oneshot::channel();
        let (release_page, release_page_receiver) = futures::channel::oneshot::channel();
        profile_store_a.update(cx_a, |store, _| {
            store.gate_next_page_for_test(page_started, release_page_receiver);
        });
        (Some(page_started_receiver), Some(release_page))
    };
    let page_during_host_loss = client_b.client().request(proto::GetProfileResultPage {
        project_id,
        import_id: 950,
        generation: before_disconnect.generation,
        protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
        page_index: 1,
    });
    let (view_import_started, view_import_started_receiver) = futures::channel::oneshot::channel();
    let (view_import_release, view_import_release_receiver) = futures::channel::oneshot::channel();
    profile_store_a.update(cx_a, |store, _| {
        store.gate_next_import_for_test(view_import_started, view_import_release_receiver);
    });
    let pending_view_cancellation = current_view.update(cx_b, |view, cx| {
        view.import_paths_with_consent_for_test(
            vec![
                fixture.path().join("profile.json"),
                fixture.path().join("profile.syms.json"),
            ],
            true,
            cx,
        )
        .expect("replacement viewer import before upstream process loss");
        view.active_import_cancellation_for_test()
            .expect("pending replacement import cancellation")
    });
    let deadline = executor.timer(std::time::Duration::from_secs(5));
    assert!(matches!(
        futures::future::select(view_import_started_receiver, deadline).await,
        futures::future::Either::Left((Ok(()), _))
    ));
    assert!(!pending_view_cancellation.load(std::sync::atomic::Ordering::Relaxed));
    let stop_host_and_release = async {
        if let Some(page_started_receiver) = page_started_receiver {
            page_started_receiver
                .await
                .expect("collaboration bridge admitted the pending page");
        } else if let Some(directory) = &host_page_gate_dir {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !directory.join("started").exists() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "authoritative host did not admit the pending page"
                );
                executor.timer(std::time::Duration::from_millis(20)).await;
            }
        }
        connection.kill().await.expect("stop separate enabled host");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()) != 0
        {
            assert!(
                std::time::Instant::now() < deadline,
                "upstream transport loss did not release the retained collaboration import"
            );
            executor.timer(std::time::Duration::from_millis(20)).await;
        }
        if let Some(release_page) = release_page {
            release_page
                .send(())
                .expect("the old page handler is still waiting after host loss");
        }
    };
    let (old_page, ()) = futures::join!(page_during_host_loss, stop_host_and_release);
    let old_page = old_page.expect("old page receives typed status after host loss");
    assert_eq!(old_page.status, proto::ProfileResultStatus::Stale as i32);
    assert!(old_page.frames.is_empty() && old_page.nodes.is_empty());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while profile_store_a.read_with(cx_a, |store, _| store.active_import_count_for_test()) != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "upstream transport loss did not release the retained collaboration import"
        );
        executor.timer(std::time::Duration::from_millis(20)).await;
    }
    let disconnected = client_b
        .client()
        .request(proto::GetProfileResultPage {
            project_id,
            import_id: 950,
            generation: before_disconnect.generation,
            protocol_version: project::profile_results_wire::PROFILE_PROTOCOL_VERSION,
            page_index: 1,
        })
        .await
        .expect("stale collaboration page receives typed status after transport loss");
    assert_eq!(
        disconnected.status,
        proto::ProfileResultStatus::Stale as i32
    );
    assert!(disconnected.frames.is_empty() && disconnected.nodes.is_empty());
    executor.run_until_parked();
    assert!(pending_view_cancellation.load(std::sync::atomic::Ordering::Relaxed));
    current_view.read_with(cx_b, |view, _| {
        eprintln!(
            "pending upstream view before release: samples={:?}; notice={}",
            view.sample_count_for_test(),
            view.notice_for_test()
        );
        assert_eq!(view.sample_count_for_test(), None);
        assert!(view.active_import_cancellation_for_test().is_none());
        assert!(!view.is_selecting_for_test());
        assert!(view.notice_for_test().contains("session"));
    });
    current_source_coverage.read_with(cx_b, |store, _| {
        assert_eq!(store.selected_snapshot().expect("retained provider after upstream loss").status,
            project::source_coverage::SourceCoverageStatus::Stale,"actual upstream session loss must stale the cached guest coverage before late reply release");
    });
    upstream_coverage_release
        .send(())
        .expect("release obsolete coverage after actual authoritative process loss");
    let error = pending_upstream_coverage
        .await
        .expect_err("late pre-loss coverage cannot restore Current after upstream process loss");
    assert!(
        error.to_string().contains("expired project session"),
        "{error:#}"
    );
    current_source_coverage.read_with(cx_b, |store, _| {
        assert_eq!(
            store
                .selected_snapshot()
                .expect("provider remains stale after obsolete upstream reply")
                .status,
            project::source_coverage::SourceCoverageStatus::Stale
        )
    });
    match view_import_release.send(()) {
        Ok(()) => eprintln!("released obsolete upstream viewer import gate"),
        Err(()) => eprintln!("obsolete upstream viewer import gate already dropped"),
    }
    executor.run_until_parked();
    assert_eq!(client_b.peer_id(), Some(stable_member_peer));
    assert_eq!(client_c.peer_id(), Some(stable_other_peer));
    current_view.read_with(cx_b, |view, _| {
        eprintln!(
            "native authoritative process loss: samples={:?}; notice={}",
            view.sample_count_for_test(),
            view.notice_for_test()
        );
        assert_eq!(
            view.sample_count_for_test(),
            None,
            "authoritative process loss must clear the loaded guest model without a new import"
        );
        assert!(view.active_import_cancellation_for_test().is_none());
        assert!(!view.is_selecting_for_test());
        assert!(view.notice_for_test().contains("session"));
    });
    cx_c.read(ActiveCall::global)
        .update(cx_c, |call, cx| call.hang_up(cx))
        .await
        .expect("other collaborator leaves before test server teardown");
    cx_b.read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .expect("profile importer leaves before test server teardown");
    active_call_a
        .update(cx_a, |call, cx| call.hang_up(cx))
        .await
        .expect("host member leaves before test server teardown");
    executor.run_until_parked();
    executor.forbid_parking();
    cx_a.executor().forbid_parking();
    cx_b.executor().forbid_parking();
    cx_c.executor().forbid_parking();
}

#[gpui::test]
#[cfg(all(feature = "rust-tools-test", target_os = "macos"))]
async fn rust_tools_relay_separate_disabled_host_returns_typed_unsupported(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
) {
    use remote::{
        ConnectionIdentifier, MockDelegate, ProcessRemoteConnection, RemoteClient, RemoteConnection,
    };
    use std::path::PathBuf;

    let binary = std::env::var_os("RUST_TOOLS_DISABLED_HOST_BINARY")
        .map(PathBuf::from)
        .expect("set RUST_TOOLS_DISABLED_HOST_BINARY to a separately built disabled host");
    let server_state_dir = std::env::var_os("RUST_TOOLS_DISABLED_HOST_STATE_DIR")
        .map(PathBuf::from)
        .expect("set RUST_TOOLS_DISABLED_HOST_STATE_DIR for that host");
    assert!(binary.is_file(), "missing disabled host binary: {binary:?}");
    assert!(
        server_state_dir.is_dir(),
        "missing disabled host state: {server_state_dir:?}"
    );
    let fixture = tempfile::tempdir_in("/private/tmp").expect("create remote fixture");
    std::fs::create_dir(fixture.path().join("src")).expect("create remote source directory");
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        "[package]\nname = \"disabled_collaboration_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write remote Cargo manifest");
    std::fs::write(fixture.path().join("src/lib.rs"), "pub fn fixture() {}\n")
        .expect("write remote source");
    std::fs::write(fixture.path().join("coverage.json"), "{}").expect("write coverage artifact");
    std::fs::write(
        fixture.path().join("profile.json"),
        include_str!(
            "../../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"
        ),
    )
    .expect("write profile artifact");
    std::fs::write(
        fixture.path().join("profile.syms.json"),
        include_str!("../../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json")
            .replace("/fixture/", &format!("{}/", fixture.path().display())),
    )
    .expect("write profile symbols");

    cx_a.update(|cx| release_channel::init(semver::Version::new(1, 16, 2), cx));
    cx_a.executor().allow_parking();
    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "host_member").await;
    let client_b = server.create_client(cx_b, "collaborator").await;
    server
        .create_room(&mut [(&client_a, cx_a), (&client_b, cx_b)])
        .await;

    let connection = Arc::new(
        ProcessRemoteConnection::new(binary, server_state_dir).with_isolated_server_namespace(),
    );
    let (_cancel_tx, cancel_rx) = futures::channel::oneshot::channel();
    let remote_client = cx_a
        .update(|cx| {
            RemoteClient::new(
                ConnectionIdentifier::setup(),
                connection.clone(),
                cancel_rx,
                Arc::new(MockDelegate),
                cx,
            )
        })
        .await
        .expect("connect to separate disabled host")
        .expect("remote client");
    let (project_a, _) = client_a
        .build_ssh_project(fixture.path(), remote_client.clone(), true, cx_a)
        .await;
    let active_call_a = cx_a.read(ActiveCall::global);
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("share disabled remote host project");
    let project_b = client_b.join_remote_project(project_id, cx_b).await;
    executor.run_until_parked();

    for (project, context) in [(&project_a, &*cx_a), (&project_b, &*cx_b)] {
        wait_for_scanned_relay_artifacts(
            project,
            fixture.path(),
            &["coverage.json", "profile.json", "profile.syms.json"],
            &executor,
            context,
        )
        .await;
    }

    let local_cargo_calls_before =
        project::cargo_workspace_store::local_cargo_process_calls_for_test();
    let local_discovery_calls_before =
        project::rust_test_provider::local_rust_test_discovery_calls_for_test();
    let local_coverage_interpret_calls_before =
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test();

    let cargo = project_b.read_with(cx_b, |project, _| project.cargo_workspace_store().clone());
    let error = cargo
        .update(cx_b, |store, cx| store.refresh(cx))
        .await
        .expect_err("disabled host cannot serve Cargo metadata through relay");
    assert_eq!(
        error
            .downcast_ref::<project::cargo_workspace_store::CargoWorkspaceRemoteError>()
            .expect("typed Cargo capability error")
            .kind,
        project::cargo_workspace_store::CargoWorkspaceRemoteErrorKind::UnsupportedHost
    );

    let tests = project_b.read_with(cx_b, |project, _| {
        project.rust_test_provider_store().clone()
    });
    let error = tests
        .update(cx_b, |store, cx| store.refresh(cx))
        .await
        .expect_err("disabled host cannot serve Rust discovery through relay");
    assert_eq!(
        error
            .downcast_ref::<project::rust_test_provider::RustTestRemoteError>()
            .expect("typed Rust discovery capability error")
            .kind,
        project::rust_test_provider::RustTestRemoteErrorKind::UnsupportedDiscoveryHost
    );

    let structured = project_b.read_with(cx_b, |project, _| {
        project.structured_execution_store().clone()
    });
    let error = structured
        .update(cx_b, |store, cx| {
            store.refresh_provider(
                project::structured_execution::StructuredProviderId("rust-tests".into()),
                cx,
            )
        })
        .await
        .expect_err("disabled host cannot serve structured results through relay");
    assert!(
        error
            .downcast_ref::<project::structured_execution::StructuredExecutionUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );

    let coverage_path = project_b.read_with(cx_b, |project, cx| {
        project
            .find_project_path(fixture.path().join("coverage.json"), cx)
            .expect("scanned coverage artifact")
    });
    cx_b.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-coverage-experimental",
            true,
            cx,
        )
    });
    let coverage = project_b.read_with(cx_b, |project, _| {
        project.rust_coverage_provider_store().clone()
    });
    let error = coverage
        .update(cx_b, |store, cx| {
            store.interpret_artifact(
                coverage_path,
                project::rust_coverage_provider::MAX_RUST_COVERAGE_ARTIFACT_BYTES,
                cx,
            )
        })
        .await
        .expect_err("disabled host cannot interpret Rust coverage through relay");
    assert!(
        error
            .downcast_ref::<project::source_coverage::SourceCoverageUnsupportedHost>()
            .is_some(),
        "{error:#}"
    );

    let profile_paths = project_b.read_with(cx_b, |project, cx| {
        [
            project
                .find_project_path(fixture.path().join("profile.json"), cx)
                .expect("scanned profile artifact"),
            project
                .find_project_path(fixture.path().join("profile.syms.json"), cx)
                .expect("scanned profile symbols"),
        ]
    });
    let local_profile_loads_before = project::profile_results::samply_profile_load_calls_for_test();
    cx_b.update(|cx| {
        project::task_store::set_experimental_feature_for_test(
            "rust-native-profiling-experimental",
            true,
            cx,
        )
    });
    let error = project_b
        .read_with(cx_b, |project, cx| {
            project.import_remote_profile(
                profile_paths,
                true,
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cx,
            )
        })
        .expect("consented import reaches shared remote-backed project")
        .await
        .expect_err("separate feature-disabled host has no native profile handler");
    assert_eq!(
        error.downcast_ref::<project::profile_results_store::RemoteProfileError>(),
        Some(&project::profile_results_store::RemoteProfileError::Unsupported)
    );
    assert_eq!(
        project::profile_results::samply_profile_load_calls_for_test(),
        local_profile_loads_before,
        "a disabled host must not cause local profile parsing"
    );
    assert_eq!(
        project::cargo_workspace_store::local_cargo_process_calls_for_test(),
        local_cargo_calls_before,
        "a disabled host must not cause local Cargo or rustc execution"
    );
    assert_eq!(
        project::rust_test_provider::local_rust_test_discovery_calls_for_test(),
        local_discovery_calls_before,
        "a disabled host must not cause local Rust test discovery"
    );
    assert_eq!(
        project::rust_coverage_provider::local_rust_coverage_interpret_calls_for_test(),
        local_coverage_interpret_calls_before,
        "a disabled host must not cause local coverage interpretation"
    );

    let shutdown = remote_client.update(cx_a, |client, cx| {
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

#[gpui::test]
#[cfg(feature = "rust-tools-test")]
async fn rust_tools_relay_rejects_reply_after_member_leaves(
    executor: BackgroundExecutor,
    cx_a: &mut TestAppContext,
    cx_b: &mut TestAppContext,
    cx_c: &mut TestAppContext,
) {
    use gpui::BorrowAppContext;

    let mut server = TestServer::start(executor.clone()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let client_c = server.create_client(cx_c, "user_c").await;
    client_a.language_registry().add(rust_lang());
    let mut fake_servers = client_a.language_registry().register_fake_lsp(
        "Rust",
        FakeLspAdapter {
            name: "rust-analyzer",
            ..FakeLspAdapter::default()
        },
    );
    let channel_id = server
        .make_public_channel("rust-tools-relay-leave", &client_a, cx_a)
        .await;
    client_a
        .fs()
        .insert_tree(
            "/relay",
            serde_json::json!({
                "Cargo.toml": "[package]\nname = \"standalone\"\nversion = \"1.0.0\"\n",
                "src": {"main.rs": "#[test]\nfn host_case() {}\n"},
            }),
        )
        .await;
    let project_a = client_a
        .build_local_project_with_trust("/relay", cx_a)
        .await
        .0;
    let active_call_a = cx_a.read(ActiveCall::global);
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .expect("host joins channel");
    let project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("host shares project");
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .expect("member joins channel");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("promote member");
    executor.run_until_parked();

    let (store, worktree_store, environment, worktree_id) =
        project_a.read_with(cx_a, |project, cx| {
            (
                project.cargo_workspace_store().clone(),
                project.worktree_store(),
                project.environment().clone(),
                project
                    .find_project_path("/relay/Cargo.toml", cx)
                    .expect("scanned manifest")
                    .worktree_id,
            )
        });
    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store,
            environment,
            Arc::new(RelayGatedMetadataRunner {
                started: Mutex::new(Some(started)),
                release: Mutex::new(Some(release_receiver)),
            }),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });

    let request = client_b.client().request(proto::GetCargoWorkspace {
        project_id,
        request_id: 914,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let leave = async {
        started_receiver
            .await
            .expect("host metadata runner started");
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("member leaves channel");
        executor.run_until_parked();
        release
            .send(())
            .expect("host runner remains pending after member leaves");
    };
    let (response, ()) = futures::join!(request, leave);
    let error = response.expect_err("the old request cannot receive a reply after leaving");
    assert!(
        error
            .to_string()
            .contains("not authorized to edit projects"),
        "{error:#}"
    );

    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .expect("member rejoins channel");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore member role");
    executor.run_until_parked();
    let (worktree_store, environment) = project_a.read_with(cx_a, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store,
            environment,
            Arc::new(RelayFixtureMetadataRunner),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    let fresh = client_b
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 915,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("a new request works after rejoining");
    assert_eq!(fresh.request_id, 915);

    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    let (worktree_store, environment) = project_a.read_with(cx_a, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store,
            environment,
            Arc::new(RelayGatedMetadataRunner {
                started: Mutex::new(Some(started)),
                release: Mutex::new(Some(release_receiver)),
            }),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    let request = client_b.client().request(proto::GetCargoWorkspace {
        project_id,
        request_id: 919,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let demote = async {
        started_receiver
            .await
            .expect("host metadata runner started before role revocation");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Guest,
                        cx,
                    )
                })
            })
            .await
            .expect("revoke member edit capability while request is pending");
        executor.run_until_parked();
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore Member before releasing admitted Cargo metadata");
        executor.run_until_parked();
        release
            .send(())
            .expect("host runner remains pending after role revocation");
    };
    let (response, ()) = futures::join!(request, demote);
    let error = response.expect_err("restoring Member cannot revive admitted Cargo metadata");
    assert!(
        error
            .to_string()
            .contains("authorization changed while awaiting host response"),
        "{error:#}"
    );
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore member edit capability after stale reply");
    executor.run_until_parked();
    let (worktree_store, environment) = project_a.read_with(cx_a, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store,
            environment,
            Arc::new(RelayFixtureMetadataRunner),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    let fresh = client_b
        .client()
        .request(proto::GetCargoWorkspace {
            project_id,
            request_id: 920,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("new mutating request succeeds after role restoration");
    assert_eq!(fresh.request_id, 920);

    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    let (worktree_store, environment) = project_a.read_with(cx_a, |project, _| {
        (project.worktree_store(), project.environment().clone())
    });
    store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store,
            environment,
            Arc::new(RelayGatedMetadataRunner {
                started: Mutex::new(Some(started)),
                release: Mutex::new(Some(release_receiver)),
            }),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    let request = client_b.client().request(proto::GetCargoWorkspace {
        project_id,
        request_id: 916,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let rejoin = async {
        started_receiver.await.expect("second runner started");
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("member leaves again");
        executor.run_until_parked();
        cx_b.update(|cx| {
            workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
        })
        .await
        .expect("member rejoins before the old response");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role again");
        executor.run_until_parked();
        release
            .send(())
            .expect("host runner remains pending through member rejoin");
    };
    let (response, ()) = futures::join!(request, rejoin);
    let error = response.expect_err("rejoining must not revive the old request");
    assert!(
        error
            .to_string()
            .contains("project request authorization changed"),
        "{error:#}"
    );

    let (discovery_started, discovery_started_receiver) = futures::channel::oneshot::channel();
    let (discovery_release, discovery_release_receiver) = futures::channel::oneshot::channel();
    let (_buffer, _handle) = project_a
        .update(cx_a, |project, cx| {
            project.open_buffer_with_lsp((worktree_id, rel_path("src/main.rs")), cx)
        })
        .await
        .expect("open relay Rust source with LSP");
    let fake_server = fake_servers.next().await.expect("host Rust Analyzer");
    fake_server.set_request_handler::<Runnables, _, _>(|_, _| async {
        Ok(vec![Runnable {
            label: "test host_case".into(),
            location: Some(lsp::LocationLink {
                origin_selection_range: None,
                target_uri: lsp::Uri::from_file_path("/relay/src/main.rs")
                    .expect("relay source URI"),
                target_range: lsp::Range::new(lsp::Position::new(1, 0), lsp::Position::new(1, 17)),
                target_selection_range: lsp::Range::new(
                    lsp::Position::new(1, 0),
                    lsp::Position::new(1, 17),
                ),
            }),
            args: RunnableArgs::Cargo(CargoRunnableArgs {
                environment: Default::default(),
                cwd: "/relay".into(),
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
    });
    executor.run_until_parked();
    cx_a.update(|cx| {
        project::trusted_worktrees::init(Default::default(), cx);
        let worktree_store = project_a.read(cx).worktree_store();
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
    let (cargo_store, rust_store, worktree_store, environment, structured_store, lsp_store) =
        project_a.read_with(cx_a, |project, _| {
            (
                project.cargo_workspace_store().clone(),
                project.rust_test_provider_store().clone(),
                project.worktree_store(),
                project.environment().clone(),
                project.structured_execution_store().clone(),
                project.lsp_store(),
            )
        });
    cargo_store.update(cx_a, |store, cx| {
        *store = CargoWorkspaceStore::local_with_runner(
            worktree_store.clone(),
            environment.clone(),
            Arc::new(RelayFixtureMetadataRunner),
        );
        store.shared(project_id, client_a.client().clone().into(), cx);
    });
    rust_store.update(cx_a, |store, cx| {
        *store = RustTestProviderStore::local_with_runner(
            worktree_store.clone(),
            environment.clone(),
            cargo_store.clone(),
            structured_store.clone(),
            Arc::new(RelayGatedDiscoveryRunner {
                started: Mutex::new(Some(discovery_started)),
                release: Mutex::new(Some(discovery_release_receiver)),
            }),
            cx,
        );
        store.observe_lsp_store(&lsp_store, cx);
    });
    let request = client_b.client().request(proto::GetRustTestDiscovery {
        project_id,
        request_id: 917,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let rejoin = async {
        discovery_started_receiver
            .await
            .expect("host Rust discovery started");
        assert_eq!(
            rust_store.read_with(cx_a, |store, cx| store.current_semantic_hint_count(cx)),
            1,
            "the in-flight host discovery has a real source hint"
        );
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("member leaves during Rust discovery");
        executor.run_until_parked();
        cx_b.update(|cx| {
            workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
        })
        .await
        .expect("member rejoins before Rust discovery completes");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role after Rust discovery interruption");
        executor.run_until_parked();
        assert!(
            discovery_release.send(()).is_err(),
            "departed member's active discovery runner must be dropped before rejoin"
        );
    };
    let (response, ()) = futures::join!(request, rejoin);
    let error = response.expect_err("rejoining must not revive an old Rust discovery reply");
    assert!(
        error
            .to_string()
            .contains("Rust test discovery request was cancelled"),
        "{error:#}"
    );

    rust_store.update(cx_a, |store, cx| {
        *store = RustTestProviderStore::local_with_runner(
            worktree_store,
            environment,
            cargo_store,
            structured_store,
            Arc::new(RelayFixtureDiscoveryRunner),
            cx,
        );
        store.observe_lsp_store(&lsp_store, cx);
    });
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 918,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("fresh Rust discovery works after rejoin");
    assert_eq!(fresh.request_id, 918);
    assert!(fresh.discovery_generation > 0);
    assert!(
        fresh.status == proto::StructuredProviderStatus::Current as i32
            || fresh.status == proto::StructuredProviderStatus::Partial as i32,
        "new member discovery must succeed: {:?}",
        fresh.diagnostic
    );
    let source_page = client_b
        .client()
        .request(proto::GetStructuredExecutionSnapshot {
            project_id,
            provider_id: RustTestProviderStore::provider_id().0.to_string(),
            discovery_generation: fresh.discovery_generation,
            page_start: 0,
            page_size: 100,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("fresh member reads source-bearing Rust discovery");
    let source_case = source_page
        .nodes
        .iter()
        .find(|node| node.label == "host_case")
        .expect("fresh member sees host case");
    assert_eq!(
        source_case.path.as_ref().map(|path| path.path.as_str()),
        Some("src/main.rs")
    );
    let case_id = project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let snapshot = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .expect("host Rust discovery snapshot");
        snapshot
            .nodes
            .iter()
            .find(|node| node.label == "host_case")
            .expect("host case")
            .id
            .0
            .clone()
    });
    let (discovery_started, discovery_started_receiver) = futures::channel::oneshot::channel();
    let (discovery_release, discovery_release_receiver) = futures::channel::oneshot::channel();
    let (cargo_store, worktree_store, environment, structured_store, lsp_store) = project_a
        .read_with(cx_a, |project, _| {
            (
                project.cargo_workspace_store().clone(),
                project.worktree_store(),
                project.environment().clone(),
                project.structured_execution_store().clone(),
                project.lsp_store(),
            )
        });
    rust_store.update(cx_a, |store, cx| {
        *store = RustTestProviderStore::local_with_runner(
            worktree_store,
            environment,
            cargo_store,
            structured_store,
            Arc::new(RelayGatedDiscoveryRunner {
                started: Mutex::new(Some(discovery_started)),
                release: Mutex::new(Some(discovery_release_receiver)),
            }),
            cx,
        );
        store.observe_lsp_store(&lsp_store, cx);
    });
    let request = client_b.client().request(proto::GetRustTestDiscovery {
        project_id,
        request_id: 921,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let demote = async {
        discovery_started_receiver
            .await
            .expect("host Rust discovery started before role revocation");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Guest,
                        cx,
                    )
                })
            })
            .await
            .expect("revoke member edit capability during Rust discovery");
        executor.run_until_parked();
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role before the admitted discovery reply");
        executor.run_until_parked();
        discovery_release
            .send(())
            .expect("host Rust discovery remains pending after role revocation");
    };
    let (response, ()) = futures::join!(request, demote);
    let error =
        response.expect_err("restoring Member cannot revive an admitted old discovery reply");
    assert!(
        error
            .to_string()
            .contains("authorization changed while awaiting host response"),
        "{error:#}"
    );
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Guest,
                    cx,
                )
            })
        })
        .await
        .expect("restrict restored member before checking action denial");
    executor.run_until_parked();
    let denied_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await;
    assert!(
        denied_action.is_err(),
        "a read-only guest cannot resolve an otherwise valid host Rust test action"
    );
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore member role before host action");
    executor.run_until_parked();
    let expired_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member resolves a host action before leaving");
    let expired_plan: serde_json::Value =
        serde_json::from_str(&expired_action.plan_json).expect("old host action plan");
    cx_b.read(ActiveCall::global)
        .update(cx_b, |call, cx| call.hang_up(cx))
        .await
        .expect("action owner leaves before queueing its Task");
    executor.run_until_parked();
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .expect("action owner rejoins the shared project");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore member role after the action lease expired");
    executor.run_until_parked();
    let expired_update = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: expired_plan["run_id"]
                .as_str()
                .expect("expired run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "expired-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect_err("rejoin must not restore the departed member's action authorization");
    assert!(
        expired_update.to_string().contains("Unknown or expired"),
        "{expired_update:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let snapshot = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .expect("host Rust provider after action expiry");
        assert!(
            snapshot.current_run.is_none(),
            "expired action must create no host run"
        );
    });
    let action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("rejoined member resolves an authoritative host action");
    let plan: serde_json::Value =
        serde_json::from_str(&action.plan_json).expect("bounded host action plan");
    assert_eq!(plan["kind"], "task");
    assert_eq!(plan["discovery_generation"], fresh.discovery_generation);
    assert_eq!(plan["scope_node_ids"], serde_json::json!([case_id]));
    assert!(
        plan["run_id"]
            .as_str()
            .is_some_and(|run_id| !run_id.is_empty())
    );
    cx_c.update(|cx| {
        workspace::join_channel(channel_id, client_c.app_state.clone(), None, None, cx)
    })
    .await
    .expect("second member joins channel");
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_c.user_id().expect("second member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("promote second member");
    executor.run_until_parked();
    let queued = proto::UpdateRustTestRun {
        project_id,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: fresh.discovery_generation,
        run_id: plan["run_id"].as_str().expect("run ID").into(),
        scope_node_ids: plan["scope_node_ids"]
            .as_array()
            .expect("case scope")
            .iter()
            .map(|node_id| node_id.as_str().expect("scope ID").to_string())
            .collect(),
        task_id: "channel-owned-rust-task".into(),
        state: proto::RustTestTaskState::Queued as i32,
        worktree_ids: vec![worktree_id.to_proto()],
        ..Default::default()
    };
    let denied_lifecycle = client_c
        .client()
        .request(queued.clone())
        .await
        .expect_err("a second member cannot update the first member's Rust Task");
    assert!(
        denied_lifecycle.to_string().contains("Unknown or expired"),
        "{denied_lifecycle:#}"
    );
    client_b
        .client()
        .request(queued)
        .await
        .expect("the owner can queue the host Rust Task");
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let snapshot = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .expect("host Rust provider");
        assert_eq!(
            snapshot.current_run.as_ref().map(|run| run.id.0.as_str()),
            plan["run_id"].as_str()
        );
    });

    let structured_store = project_a.read_with(cx_a, |project, _| {
        project.structured_execution_store().clone()
    });
    let revoke_and_restore_member = async |cx_a: &mut TestAppContext| {
        for role in [proto::ChannelRole::Guest, proto::ChannelRole::Member] {
            active_call_a
                .update(cx_a, |call, cx| {
                    call.room().expect("host room").update(cx, |room, cx| {
                        room.set_participant_role(
                            client_b.user_id().expect("member identity"),
                            role,
                            cx,
                        )
                    })
                })
                .await
                .expect("cycle member authorization while the host read reply is admitted");
            executor.run_until_parked();
        }
    };
    let event_request = || proto::GetStructuredExecutionEvents {
        project_id,
        provider_id: RustTestProviderStore::provider_id().0,
        run_id: plan["run_id"].as_str().expect("run ID").into(),
        start_sequence: 0,
        chunk_size: 100,
        worktree_ids: vec![worktree_id.to_proto()],
    };
    let baseline_events = client_b
        .client()
        .request(event_request())
        .await
        .expect("member reads host structured run events");
    assert!(
        !baseline_events.events.is_empty(),
        "the relay must expose an actual host run event before session change"
    );
    let (events_started, events_release, events_completed) =
        structured_store.update(cx_a, |store, _| store.gate_next_events_for_test());
    let pending_events = client_b.client().request(event_request());
    let leave_rejoin = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(events_started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("structured events member leaves before host reply");
        executor.run_until_parked();
        cx_b.update(|cx| {
            workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
        })
        .await
        .expect("structured events member rejoins before old reply");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role before fresh structured events");
        executor.run_until_parked();
        events_release
            .send(())
            .expect("host events response remains pending through rejoin");
        events_completed
            .await
            .expect("host released old events reply");
    };
    let (response, ()) = futures::join!(pending_events, leave_rejoin);
    let error = response.expect_err("rejoining must not revive old structured events");
    assert!(
        error
            .to_string()
            .contains("project request authorization changed"),
        "{error:#}"
    );
    let fresh_events = client_b
        .client()
        .request(event_request())
        .await
        .expect("fresh member reads host structured run events");
    assert_eq!(fresh_events.events.len(), baseline_events.events.len());
    let (started, release, completed) =
        structured_store.update(cx_a, |store, _| store.gate_next_events_for_test());
    let pending = client_b.client().request(event_request());
    let role_change = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        revoke_and_restore_member(cx_a).await;
        release.send(()).expect("release role-restored host events");
        completed
            .await
            .expect("host completed role-restored events");
    };
    let (response, ()) = futures::join!(pending, role_change);
    let error = response.expect_err("role restoration cannot revive admitted structured events");
    assert!(
        error
            .to_string()
            .contains("authorization changed while awaiting host response"),
        "{error:#}"
    );
    let fresh_events = client_b
        .client()
        .request(event_request())
        .await
        .expect("fresh structured events recover after role restoration");
    assert_eq!(fresh_events.events.len(), baseline_events.events.len());

    let (coverage_path, coverage_store) = project_a.read_with(cx_a, |project, cx| {
        (
            project
                .find_project_path("/relay/src/main.rs", cx)
                .expect("visible coverage source"),
            project.source_coverage_store().clone(),
        )
    });
    coverage_store
        .update(cx_a, |store, cx| {
            store.publish(
                SourceCoverageSnapshot {
                    project_generation: 1,
                    provider_id: SourceCoverageProviderId("rust-tools-relay-epoch".into()),
                    generation: 1,
                    status: SourceCoverageStatus::Current,
                    truncated: false,
                    diagnostic: None,
                    files: vec![SourceCoverageFile {
                        path: coverage_path.clone(),
                        ranges: vec![SourceCoverageRange {
                            start: SourceCoveragePoint { line: 0, column: 0 },
                            end: SourceCoveragePoint { line: 0, column: 1 },
                            hit_count: 1,
                        }],
                        covered_lines: 1,
                        uncovered_lines: 0,
                        truncated: false,
                    }],
                },
                cx,
            )
        })
        .expect("publish host coverage before relay epoch change");
    let coverage_request = || proto::GetSourceCoverage {
        project_id,
        provider_id: "rust-tools-relay-epoch".into(),
        worktree_ids: vec![worktree_id.to_proto()],
    };
    let baseline = client_b
        .client()
        .request(coverage_request())
        .await
        .expect("member receives authoritative host coverage");
    assert_eq!(baseline.files.len(), 1);
    assert_eq!(
        baseline.files[0].path.as_ref(),
        Some(&coverage_path.to_proto())
    );

    let (assembled, release, completed) = coverage_store.update(cx_a, |store, _| {
        store.gate_next_assembled_response_for_test()
    });
    let request = client_b.client().request(coverage_request());
    let leave_rejoin = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("coverage member leaves before the host reply");
        executor.run_until_parked();
        cx_b.update(|cx| {
            workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
        })
        .await
        .expect("coverage member rejoins before the old reply");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role for fresh coverage");
        executor.run_until_parked();
        release
            .send(())
            .expect("host coverage response remains pending through rejoin");
        completed.await.expect("host released coverage response");
    };
    let (response, ()) = futures::join!(request, leave_rejoin);
    let error = response.expect_err("rejoining must not revive old coverage");
    assert!(
        error
            .to_string()
            .contains("project request authorization changed"),
        "{error:#}"
    );
    let fresh = client_b
        .client()
        .request(coverage_request())
        .await
        .expect("a new coverage request works after rejoining");
    assert_eq!(fresh.files.len(), 1);
    assert_eq!(
        fresh.files[0].path.as_ref(),
        Some(&coverage_path.to_proto())
    );

    let (started, release, completed) = coverage_store.update(cx_a, |store, _| {
        store.gate_next_assembled_response_for_test()
    });
    let pending = client_b.client().request(coverage_request());
    let role_change = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        revoke_and_restore_member(cx_a).await;
        release
            .send(())
            .expect("release role-restored source coverage");
        completed
            .await
            .expect("host completed role-restored coverage");
    };
    let (response, ()) = futures::join!(pending, role_change);
    let error = response.expect_err("role restoration cannot revive admitted source coverage");
    assert!(
        error
            .to_string()
            .contains("authorization changed while awaiting host response"),
        "{error:#}"
    );
    let fresh = client_b
        .client()
        .request(coverage_request())
        .await
        .expect("fresh source coverage recovers after role restoration");
    assert_eq!(fresh.files.len(), 1);
    assert_eq!(
        fresh.files[0].path.as_ref(),
        Some(&coverage_path.to_proto())
    );

    let structured_store = project_a.read_with(cx_a, |project, _| {
        project.structured_execution_store().clone()
    });
    let snapshot_request = || proto::GetStructuredExecutionSnapshot {
        project_id,
        provider_id: RustTestProviderStore::provider_id().0,
        discovery_generation: source_page.discovery_generation,
        page_start: 0,
        page_size: 100,
        worktree_ids: vec![worktree_id.to_proto()],
    };
    let (stale_snapshot_started, stale_snapshot_release, stale_snapshot_completed) =
        structured_store.update(cx_a, |store, _| store.gate_next_snapshot_for_test());
    let pending = client_b.client().request(snapshot_request());
    let leave_rejoin = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(stale_snapshot_started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("structured snapshot member leaves before the host reply");
        executor.run_until_parked();
        cx_b.update(|cx| {
            workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
        })
        .await
        .expect("structured snapshot member rejoins before the old reply");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role for fresh structured snapshot");
        executor.run_until_parked();
        stale_snapshot_release
            .send(())
            .expect("host structured snapshot remains pending through rejoin");
        stale_snapshot_completed
            .await
            .expect("host released old structured snapshot");
    };
    let (response, ()) = futures::join!(pending, leave_rejoin);
    let error = response.expect_err("rejoining must not revive old structured snapshot");
    assert!(
        error
            .to_string()
            .contains("project request authorization changed"),
        "{error:#}"
    );
    let fresh_snapshot = client_b
        .client()
        .request(snapshot_request())
        .await
        .expect("a new structured snapshot works after rejoining");
    assert!(
        fresh_snapshot
            .nodes
            .iter()
            .any(|node| node.label == "host_case")
    );

    let (started, release, completed) =
        structured_store.update(cx_a, |store, _| store.gate_next_snapshot_for_test());
    let pending = client_b.client().request(snapshot_request());
    let role_change = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        revoke_and_restore_member(cx_a).await;
        release
            .send(())
            .expect("release role-restored structured snapshot");
        completed
            .await
            .expect("host completed role-restored snapshot");
    };
    let (response, ()) = futures::join!(pending, role_change);
    let error = response.expect_err("role restoration cannot revive admitted structured snapshot");
    assert!(
        error
            .to_string()
            .contains("authorization changed while awaiting host response"),
        "{error:#}"
    );
    let fresh_snapshot = client_b
        .client()
        .request(snapshot_request())
        .await
        .expect("fresh structured snapshot recovers after role restoration");
    assert!(
        fresh_snapshot
            .nodes
            .iter()
            .any(|node| node.label == "host_case")
    );

    let (snapshot_started, snapshot_release, snapshot_completed) =
        structured_store.update(cx_a, |store, _| store.gate_next_snapshot_for_test());
    let pending = client_b.client().request(snapshot_request());
    let revoke_source = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(snapshot_started, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_a.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/main.rs".into()]));
                });
            });
        });
        executor.run_until_parked();
        project_a.read_with(cx_a, |project, cx| {
            assert!(
                project
                    .entry_for_path(&coverage_path, cx)
                    .is_some_and(|entry| entry.is_private),
                "host source must be private before releasing the snapshot"
            );
        });
        snapshot_release
            .send(())
            .expect("host structured snapshot remains pending after privacy revocation");
        snapshot_completed
            .await
            .expect("host completed the privacy-filtered snapshot");
    };
    let (response, ()) = futures::join!(pending, revoke_source);
    let delayed = response.expect("authorized member receives the filtered snapshot");
    assert!(
        delayed.nodes.iter().all(|node| node.label != "host_case"),
        "an in-flight snapshot must not reveal a now-private Rust source"
    );
    assert!(delayed.partial, "private source removal must be explicit");
    let current = client_b
        .client()
        .request(snapshot_request())
        .await
        .expect("fresh member can read the filtered provider");
    assert!(current.nodes.iter().all(|node| node.label != "host_case"));

    cx_a.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files = None;
            });
        });
    });
    executor.run_until_parked();
    project_a.read_with(cx_a, |project, cx| {
        assert!(
            project
                .entry_for_path(&coverage_path, cx)
                .is_some_and(|entry| !entry.is_private),
            "source must be visible before assembling the old response"
        );
    });
    let visible = client_b
        .client()
        .request(snapshot_request())
        .await
        .expect("member sees the restored source before the epoch change");
    assert!(
        visible
            .nodes
            .iter()
            .any(|node| node.label == "host_case" && node.path.is_some())
    );

    let (assembled, release, completed) = structured_store.update(cx_a, |store, _| {
        store.gate_next_assembled_response_for_test()
    });
    let pending = client_b.client().request(snapshot_request());
    let revoke_and_rejoin = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(assembled, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        cx_a.update(|cx| {
            cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.worktree.private_files =
                        Some(settings::ExtendingVec(vec!["**/main.rs".into()]));
                });
            });
        });
        executor.run_until_parked();
        project_a.read_with(cx_a, |project, cx| {
            assert!(
                project
                    .entry_for_path(&coverage_path, cx)
                    .is_some_and(|entry| entry.is_private),
                "source must be private before releasing the old response"
            );
        });
        cx_b.read(ActiveCall::global)
            .update(cx_b, |call, cx| call.hang_up(cx))
            .await
            .expect("structured member leaves before the host reply");
        executor.run_until_parked();
        cx_b.update(|cx| {
            workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
        })
        .await
        .expect("structured member rejoins before the old reply");
        active_call_a
            .update(cx_a, |call, cx| {
                call.room().expect("host room").update(cx, |room, cx| {
                    room.set_participant_role(
                        client_b.user_id().expect("member identity"),
                        proto::ChannelRole::Member,
                        cx,
                    )
                })
            })
            .await
            .expect("restore member role for a fresh structured request");
        executor.run_until_parked();
        release
            .send(())
            .expect("old structured response remains held through rejoin");
        completed.await.expect("host completed the old response");
    };
    let (response, ()) = futures::join!(pending, revoke_and_rejoin);
    let error = response.expect_err("rejoining must not revive the old structured reply");
    assert!(
        error
            .to_string()
            .contains("project request authorization changed"),
        "{error:#}"
    );
    let fresh = client_b
        .client()
        .request(snapshot_request())
        .await
        .expect("rejoined member receives a current filtered structured page");
    assert!(fresh.nodes.iter().all(|node| node.label != "host_case"));
    assert!(fresh.partial);

    cx_a.update(|cx| {
        project::trusted_worktrees::init(Default::default(), cx);
        let worktree_store = project_a.read(cx).worktree_store();
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
    let leased_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member obtains a run grant before worktree restriction");
    let leased_plan: serde_json::Value =
        serde_json::from_str(&leased_action.plan_json).expect("pre-restriction run plan");
    let generation = rust_store.read_with(cx_a, |store, _| store.generation());
    rust_store
        .update(cx_a, |store, cx| {
            store.plan_action(
                &StructuredNodeId(case_id.clone()),
                generation,
                project::rust_test_provider::RustTestAction::Run,
                cx,
            )
        })
        .await
        .expect("trusted current host case remains runnable before revocation");
    let restricted_plans = cx_a.update(|cx| {
        let worktree_store = project_a.read(cx).worktree_store();
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
        rust_store.update(cx, |store, cx| {
            assert_eq!(store.generation(), generation);
            [
                project::rust_test_provider::RustTestAction::Run,
                project::rust_test_provider::RustTestAction::Debug,
            ]
            .map(|action| {
                store.plan_action(&StructuredNodeId(case_id.clone()), generation, action, cx)
            })
        })
    });
    for plan in restricted_plans {
        let error = plan
            .await
            .expect_err("revocation must reject cached actions before the debounced refresh");
        assert!(
            error.to_string().contains("Trust the worktree"),
            "restricted action must provide trust guidance: {error:#}"
        );
    }
    cx_a.update(|cx| {
        let worktree_store = project_a.read(cx).worktree_store();
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
    executor.run_until_parked();
    assert_eq!(
        rust_store.read_with(cx_a, |store, _| store.generation()),
        generation,
        "revocation is checked before the discovery debounce publishes"
    );
    let previous_run = project_a.read_with(cx_a, |project, cx| {
        project
            .structured_execution_store()
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .and_then(|provider| provider.current_run.as_ref().map(|run| run.id.clone()))
    });
    let expired_update = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: leased_plan["run_id"]
                .as_str()
                .expect("leased run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "retrusted-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect_err("restoring trust must not revive a run grant issued before restriction");
    assert!(
        expired_update.to_string().contains("Unknown or expired"),
        "{expired_update:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let current_run = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .and_then(|provider| provider.current_run.as_ref().map(|run| run.id.clone()));
        assert_eq!(
            current_run, previous_run,
            "expired grant must not replace the host run"
        );
    });
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id,
            request_id: 925,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("current discovery succeeds after restoring trust");
    let action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("a fresh action succeeds after trust restoration");
    let plan: serde_json::Value =
        serde_json::from_str(&action.plan_json).expect("fresh trusted plan");
    client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: plan["run_id"].as_str().expect("fresh run ID").into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "fresh-retrusted-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect("fresh owner can queue after refreshed trusted discovery");
    let pending_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member obtains a pending run grant before host unshare");
    let pending_plan: serde_json::Value =
        serde_json::from_str(&pending_action.plan_json).expect("pre-unshare plan");
    active_call_a
        .update(cx_a, |call, cx| call.unshare_project(project_a.clone(), cx))
        .expect("host unshares the authoritative project");
    executor.run_until_parked();
    let reshared_project_id = active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .expect("host shares a new project session");
    assert_ne!(reshared_project_id, project_id);
    let _rejoined_project = client_b
        .join_remote_project(reshared_project_id, cx_b)
        .await;
    let expired_update = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: pending_plan["run_id"]
                .as_str()
                .expect("pending run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "reshared-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect_err("host reshare must not restore a run grant from the former session");
    assert!(
        expired_update.to_string().contains("Unknown or expired"),
        "{expired_update:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let run = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .and_then(|provider| provider.current_run.as_ref());
        assert_eq!(
            run.map(|run| run.id.0.as_str()),
            plan["run_id"].as_str(),
            "expired former-session grant must leave the existing host run intact"
        );
    });
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 926,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("new host sharing session publishes fresh discovery");
    let action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member resolves a current action after host reshare");
    let plan: serde_json::Value = serde_json::from_str(&action.plan_json).expect("reshared plan");
    client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: plan["run_id"].as_str().expect("reshared run ID").into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "fresh-reshared-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect("a fresh owner grant queues successfully after host reshare");
    let pending_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member resolves pending action before role revocation");
    let pending_plan: serde_json::Value =
        serde_json::from_str(&pending_action.plan_json).expect("pre-revocation action plan");
    let queued = proto::UpdateRustTestRun {
        project_id: reshared_project_id,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: fresh.discovery_generation,
        run_id: pending_plan["run_id"]
            .as_str()
            .expect("pending run ID")
            .into(),
        scope_node_ids: vec![case_id.clone()],
        task_id: "role-restored-channel-rust-task".into(),
        authorization_generation: pending_action.authorization_generation,
        state: proto::RustTestTaskState::Queued as i32,
        worktree_ids: vec![worktree_id.to_proto()],
        ..Default::default()
    };
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Guest,
                    cx,
                )
            })
        })
        .await
        .expect("revoke member role after resolving pending action");
    executor.run_until_parked();
    let error = client_b
        .client()
        .request(queued.clone())
        .await
        .expect_err("guest cannot queue the pending action");
    assert!(
        error
            .to_string()
            .contains("not authorized to edit projects"),
        "{error:#}"
    );
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().expect("host room").update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().expect("member identity"),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .expect("restore member role without restoring old action grant");
    executor.run_until_parked();
    let error = client_b
        .client()
        .request(queued)
        .await
        .expect_err("role restoration must not revive a pre-revocation action grant");
    assert!(
        error.to_string().contains("Unknown or expired"),
        "{error:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let current_run = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .and_then(|provider| provider.current_run.as_ref());
        assert_eq!(
            current_run.map(|run| run.id.0.as_str()),
            plan["run_id"].as_str(),
            "expired role grant must leave the previous host run intact"
        );
    });
    executor.advance_clock(std::time::Duration::from_secs(1));
    executor.run_until_parked();
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 930,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("settle pending input invalidations before reconnect isolation");
    let restored_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: pending_action.authorization_generation,
        })
        .await
        .expect("restored member resolves a fresh action");
    assert!(
        restored_action.authorization_generation > pending_action.authorization_generation,
        "relay must replace client-supplied authorization with the live generation"
    );
    let restored_plan: serde_json::Value =
        serde_json::from_str(&restored_action.plan_json).expect("fresh restored action plan");
    client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: restored_plan["run_id"]
                .as_str()
                .expect("fresh restored run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "fresh-role-restored-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: pending_action.authorization_generation,
            ..Default::default()
        })
        .await
        .expect("a fresh grant queues after role restoration");
    let pending_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member resolves pending grant before host reconnect");
    let pending_plan: serde_json::Value =
        serde_json::from_str(&pending_action.plan_json).expect("pre-reconnect action plan");
    let host_reshared = Arc::new(Mutex::new(false));
    let _reshared_subscription = project_a.update(cx_a, |_, cx| {
        let host_reshared = host_reshared.clone();
        cx.subscribe(&project_a, move |_, _, event, _| {
            if matches!(event, project::Event::Reshared) {
                *host_reshared.lock().expect("host reshare observation") = true;
            }
        })
    });
    let previous_host_peer = client_a
        .peer_id()
        .expect("host connection before reconnect");
    let original_member_peer = client_b.peer_id().expect("original member connection");
    server.disconnect_client(previous_host_peer);
    executor.advance_clock(rpc::RECEIVE_TIMEOUT);
    executor.run_until_parked();
    assert_ne!(client_a.peer_id(), Some(previous_host_peer));
    assert_eq!(client_b.peer_id(), Some(original_member_peer));
    assert!(
        *host_reshared.lock().expect("host reshare observation"),
        "the actual room reconnect must apply Project::reshared"
    );
    assert_eq!(
        project_a.read_with(cx_a, |project, _| project.remote_id()),
        Some(reshared_project_id),
        "short host reconnect preserves the shared project ID"
    );
    assert_eq!(
        rust_store.read_with(cx_a, |store, _| store.generation()),
        DiscoveryGeneration(fresh.discovery_generation),
        "host connection expiry must not depend on a new discovery generation"
    );
    let error = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: pending_plan["run_id"]
                .as_str()
                .expect("pre-reconnect run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "host-reconnected-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: pending_action.authorization_generation,
            ..Default::default()
        })
        .await
        .expect_err("host reconnect must expire the former connection's pending grant");
    assert!(
        error.to_string().contains("Unknown or expired"),
        "{error:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let run = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .and_then(|provider| provider.current_run.as_ref());
        assert_eq!(
            run.map(|run| run.id.0.as_str()),
            restored_plan["run_id"].as_str(),
            "an expired reconnect grant must preserve the existing local Task run"
        );
    });
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 928,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("discovery recovers after the authoritative host reconnects");
    let action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("fresh action recovers after host reconnect");
    let plan: serde_json::Value =
        serde_json::from_str(&action.plan_json).expect("reconnected plan");
    client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: plan["run_id"].as_str().expect("reconnected run ID").into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "fresh-host-reconnected-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect("fresh grant queues after host reconnect");
    let pending_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("member resolves pending grant before member reconnect");
    let pending_plan: serde_json::Value =
        serde_json::from_str(&pending_action.plan_json).expect("pre-member-reconnect action plan");
    let previous_member_peer = client_b.peer_id().expect("member before reconnect");
    server.disconnect_client(previous_member_peer);
    executor.advance_clock(rpc::RECEIVE_TIMEOUT);
    executor.run_until_parked();
    let current_member_peer = client_b.peer_id().expect("member after reconnect");
    assert_ne!(current_member_peer, previous_member_peer);
    project_a.read_with(cx_a, |project, _| {
        assert!(!project.collaborators().contains_key(&previous_member_peer));
        assert!(project.collaborators().contains_key(&current_member_peer));
    });
    let error = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: pending_plan["run_id"]
                .as_str()
                .expect("pre-reconnect run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "member-reconnected-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: pending_action.authorization_generation,
            ..Default::default()
        })
        .await
        .expect_err("a replacement peer must not inherit the former connection's pending grant");
    assert!(
        error.to_string().contains("Unknown or expired"),
        "{error:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let store = project.structured_execution_store();
        let run = store
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .and_then(|provider| provider.current_run.as_ref());
        assert_eq!(
            run.map(|run| run.id.0.as_str()),
            plan["run_id"].as_str(),
            "an expired reconnect grant must preserve the existing local Task run"
        );
    });
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 929,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("discovery recovers after the original member connection is replaced");
    let action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            node_id: case_id.clone(),
            action: proto::RustTestActionKind::Run as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: 0,
        })
        .await
        .expect("fresh action recovers after member reconnect");
    let plan: serde_json::Value =
        serde_json::from_str(&action.plan_json).expect("reconnected plan");
    client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: plan["run_id"].as_str().expect("reconnected run ID").into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "fresh-member-reconnected-channel-rust-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            ..Default::default()
        })
        .await
        .expect("fresh grant queues after member reconnect");
    executor.advance_clock(std::time::Duration::from_secs(1));
    executor.run_until_parked();
    for reconnect_host in [true, false] {
        let (started, started_receiver) = futures::channel::oneshot::channel();
        let (release, release_receiver) = futures::channel::oneshot::channel();
        rust_store.update(cx_a, |store, _| {
            store
                .set_discovery_runner_for_test(Arc::new(RelayGatedDiscoveryRunner {
                    started: Mutex::new(Some(started)),
                    release: Mutex::new(Some(release_receiver)),
                }))
                .expect("replace host discovery runner without resetting its generation");
        });
        let request_id = if reconnect_host { 930 } else { 932 };
        let pending = client_b.client().request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        });
        let reconnect = async {
            let deadline = executor.timer(std::time::Duration::from_secs(5));
            assert!(matches!(
                futures::future::select(started_receiver, deadline).await,
                futures::future::Either::Left((Ok(()), _))
            ));
            let admitted_generation = rust_store.read_with(cx_a, |store, _| store.generation());
            let original_host_peer = client_a.peer_id().expect("admitted host peer");
            let original_member_peer = client_b.peer_id().expect("admitted member peer");
            let previous_peer = if reconnect_host {
                *host_reshared.lock().expect("host reshare observation") = false;
                client_a.peer_id().expect("active host connection")
            } else {
                client_b.peer_id().expect("active member connection")
            };
            assert!(
                !release.is_canceled(),
                "the admitted runner must still be waiting"
            );
            server.disconnect_client(previous_peer);
            let step = std::time::Duration::from_millis(100);
            let mut remaining = rpc::RECEIVE_TIMEOUT;
            loop {
                executor.run_until_parked();
                let ready = if reconnect_host {
                    client_a.peer_id().is_some_and(|peer| peer != previous_peer)
                        && *host_reshared.lock().expect("host reshare observation")
                } else {
                    client_b.peer_id().is_some_and(|peer| {
                        peer != previous_peer
                            && project_a.read_with(cx_a, |project, _| {
                                project.collaborators().contains_key(&peer)
                            })
                    })
                };
                if ready {
                    break;
                }
                assert!(remaining >= step, "active request route did not reconnect");
                remaining = remaining.saturating_sub(step);
                // A single receive-timeout jump can expire the uninvolved peer too.
                executor.advance_clock(step);
            }
            if reconnect_host {
                assert_ne!(client_a.peer_id(), Some(previous_peer));
                assert_eq!(client_b.peer_id(), Some(original_member_peer));
                assert!(*host_reshared.lock().expect("host reshare observation"));
            } else {
                let new_peer = client_b.peer_id().expect("reconnected active member");
                assert_ne!(new_peer, previous_peer);
                assert_eq!(client_a.peer_id(), Some(original_host_peer));
                project_a.read_with(cx_a, |project, _| {
                    assert!(!project.collaborators().contains_key(&previous_peer));
                    assert!(project.collaborators().contains_key(&new_peer));
                });
            }
            assert_eq!(
                project_a.read_with(cx_a, |project, _| project.remote_id()),
                Some(reshared_project_id)
            );
            assert_eq!(
                rust_store.read_with(cx_a, |store, _| store.generation()),
                admitted_generation,
                "connection expiry must dispose discovery without input-generation invalidation"
            );
            assert!(
                release.send(()).is_err(),
                "reconnect must drop admitted discovery; host={reconnect_host}"
            );
        };
        let (response, ()) = futures::join!(pending, reconnect);
        let error = response.expect_err("the disconnected route cannot return admitted discovery");
        eprintln!("admitted discovery reconnect host={reconnect_host}: {error:#}");
        let fresh = client_b
            .client()
            .request(proto::GetRustTestDiscovery {
                project_id: reshared_project_id,
                request_id: request_id + 1,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                worktree_ids: vec![worktree_id.to_proto()],
            })
            .await
            .expect("fresh discovery recovers after active reconnect");
        assert!(
            fresh.status == proto::StructuredProviderStatus::Current as i32
                || fresh.status == proto::StructuredProviderStatus::Partial as i32,
            "fresh reconnect discovery must be actionable: {fresh:?}"
        );
        let action = client_b
            .client()
            .request(proto::ResolveRustTestAction {
                project_id: reshared_project_id,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                discovery_generation: fresh.discovery_generation,
                node_id: case_id.clone(),
                action: proto::RustTestActionKind::Run as i32,
                worktree_ids: vec![worktree_id.to_proto()],
                authorization_generation: 0,
            })
            .await
            .expect("fresh action recovers after active reconnect");
        let plan: serde_json::Value =
            serde_json::from_str(&action.plan_json).expect("active reconnect action plan");
        assert!(plan["run_id"].as_str().is_some());
    }
    let fresh = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 940,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("discover before malformed reshare");
    let resolve = || proto::ResolveRustTestAction {
        project_id: reshared_project_id,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        discovery_generation: fresh.discovery_generation,
        node_id: case_id.clone(),
        action: proto::RustTestActionKind::Run as i32,
        worktree_ids: vec![worktree_id.to_proto()],
        authorization_generation: 0,
    };
    let retained = client_b
        .client()
        .request(resolve())
        .await
        .expect("resolve retained run");
    let retained_plan: serde_json::Value =
        serde_json::from_str(&retained.plan_json).expect("retained action plan");
    let retained_run_id = retained_plan["run_id"]
        .as_str()
        .expect("retained run ID")
        .to_string();
    client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: retained_run_id.clone(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "malformed-reshare-retained-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: retained.authorization_generation,
            ..Default::default()
        })
        .await
        .expect("record retained run before malformed reshare");
    let action = client_b
        .client()
        .request(resolve())
        .await
        .expect("resolve pending malformed-session grant");
    let pending_plan: serde_json::Value =
        serde_json::from_str(&action.plan_json).expect("pending malformed-session plan");
    let malformed = || proto::ResharedProject {
        id: reshared_project_id,
        collaborators: vec![proto::Collaborator::default()],
    };
    let admitted_generation = rust_store.read_with(cx_a, |store, _| store.generation());
    let error = project_a
        .update(cx_a, |project, cx| project.reshared(malformed(), cx))
        .expect_err("reject collaborator snapshot without peer identity");
    eprintln!("malformed reshare snapshot: {error:#}");
    assert_eq!(
        rust_store.read_with(cx_a, |store, _| store.generation()),
        admitted_generation
    );
    let error = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: fresh.discovery_generation,
            run_id: pending_plan["run_id"]
                .as_str()
                .expect("pending run ID")
                .into(),
            scope_node_ids: vec![case_id.clone()],
            task_id: "malformed-reshare-expired-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: action.authorization_generation,
            ..Default::default()
        })
        .await
        .expect_err("a rejected reshare must still expire the former connection grant");
    assert!(
        error.to_string().contains("Unknown or expired"),
        "{error:#}"
    );
    project_a.read_with(cx_a, |project, cx| {
        let provider = project
            .structured_execution_store()
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .expect("retained Rust provider");
        assert_eq!(
            provider.current_run.as_ref().map(|run| run.id.0.as_str()),
            Some(retained_run_id.as_str())
        );
    });
    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    rust_store.update(cx_a, |store, _| {
        store
            .set_discovery_runner_for_test(Arc::new(RelayGatedDiscoveryRunner {
                started: Mutex::new(Some(started)),
                release: Mutex::new(Some(release_receiver)),
            }))
            .expect("gate admitted malformed-session discovery");
    });
    let pending = client_b.client().request(proto::GetRustTestDiscovery {
        project_id: reshared_project_id,
        request_id: 942,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let reject_snapshot = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        assert!(!release.is_canceled());
        let admitted_generation = rust_store.read_with(cx_a, |store, _| store.generation());
        project_a
            .update(cx_a, |project, cx| project.reshared(malformed(), cx))
            .expect_err("reject malformed snapshot during admitted discovery");
        executor.run_until_parked();
        assert_eq!(
            rust_store.read_with(cx_a, |store, _| store.generation()),
            admitted_generation
        );
        assert!(
            release.send(()).is_err(),
            "malformed reshare must drop admitted discovery"
        );
    };
    let (response, ()) = futures::join!(pending, reject_snapshot);
    let error = response.expect_err("rejected reshare cannot return admitted discovery");
    assert!(
        error
            .to_string()
            .contains("Rust test discovery request was cancelled"),
        "{error:#}"
    );
    let recovered = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 943,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("discovery recovers after malformed snapshot rejection");
    assert!(
        recovered.status == proto::StructuredProviderStatus::Current as i32
            || recovered.status == proto::StructuredProviderStatus::Partial as i32
    );
    client_b
        .client()
        .request(proto::ResolveRustTestAction {
            discovery_generation: recovered.discovery_generation,
            ..resolve()
        })
        .await
        .expect("fresh action recovers after malformed snapshot rejection");
    let _snapshot_other_project = client_c
        .join_remote_project(reshared_project_id, cx_c)
        .await;
    executor.run_until_parked();
    let snapshot_member_peer = client_b.peer_id().expect("omitted snapshot member");
    let snapshot_host_peer = client_a.peer_id().expect("omitted snapshot host");
    let snapshot_other_peer = client_c.peer_id().expect("other snapshot member");
    let collaborators = project_a.read_with(cx_a, |project, _| {
        project
            .collaborators()
            .iter()
            .map(|(peer_id, collaborator)| {
                assert_eq!(*peer_id, collaborator.peer_id);
                proto::Collaborator {
                    peer_id: Some(collaborator.peer_id),
                    replica_id: u32::from(collaborator.replica_id.as_u16()),
                    user_id: collaborator.user_id,
                    is_host: collaborator.is_host,
                    committer_name: collaborator.committer_name.clone(),
                    committer_email: collaborator.committer_email.clone(),
                }
            })
            .collect::<Vec<_>>()
    });
    assert!(
        collaborators
            .iter()
            .any(|collaborator| collaborator.peer_id == Some(snapshot_member_peer))
    );
    let retained_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            discovery_generation: recovered.discovery_generation,
            ..resolve()
        })
        .await
        .expect("reserve recorded result before owner omission");
    let retained_plan: serde_json::Value =
        serde_json::from_str(&retained_action.plan_json).expect("retained omitted-snapshot plan");
    let snapshot_retained_run_id = retained_plan["run_id"]
        .as_str()
        .expect("retained omitted-snapshot run ID")
        .to_string();
    for state in [
        proto::RustTestTaskState::Queued,
        proto::RustTestTaskState::Completed,
    ] {
        client_b
            .client()
            .request(proto::UpdateRustTestRun {
                project_id: reshared_project_id,
                protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
                discovery_generation: recovered.discovery_generation,
                run_id: snapshot_retained_run_id.clone(),
                scope_node_ids: vec![case_id.clone()],
                task_id: "omitted-snapshot-retained-task".into(),
                state: state as i32,
                terminal_id: Some(944),
                exit_code: (state == proto::RustTestTaskState::Completed).then_some(0),
                success: state == proto::RustTestTaskState::Completed,
                worktree_ids: vec![worktree_id.to_proto()],
                authorization_generation: retained_action.authorization_generation,
                ..Default::default()
            })
            .await
            .expect("record completed Rust lifecycle result before omission");
    }
    let preserved_completed_run = project_a.read_with(cx_a, |project, cx| {
        let provider = project
            .structured_execution_store()
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .expect("recorded snapshot provider");
        let run = provider
            .last_complete_run
            .clone()
            .expect("completed snapshot result");
        assert_eq!(run.id.0, snapshot_retained_run_id);
        run
    });
    let pending_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            discovery_generation: recovered.discovery_generation,
            ..resolve()
        })
        .await
        .expect("reserve run grant before valid owner omission");
    let pending_plan: serde_json::Value =
        serde_json::from_str(&pending_action.plan_json).expect("omitted owner run plan");
    let pending_run_id = pending_plan["run_id"]
        .as_str()
        .expect("omitted owner pending run ID")
        .to_string();
    let omitted = Arc::new(Mutex::new(false));
    let _omitted_subscription = project_a.update(cx_a, |_, cx| {
        let omitted = omitted.clone();
        cx.subscribe(&project_a, move |_, _, event, _| {
            if matches!(event, project::Event::CollaboratorLeft(peer) if *peer == snapshot_member_peer) {
                *omitted.lock().expect("Rust snapshot omission observation") = true;
            }
        })
    });
    let (omitted_started, omitted_started_receiver) = futures::channel::oneshot::channel();
    let (omitted_release, omitted_release_receiver) = futures::channel::oneshot::channel();
    rust_store.update(cx_a, |store, _| {
        store
            .set_discovery_runner_for_test(Arc::new(RelayGatedDiscoveryRunner {
                started: Mutex::new(Some(omitted_started)),
                release: Mutex::new(Some(omitted_release_receiver)),
            }))
            .expect("gate admitted discovery before valid owner omission");
    });
    let pending_discovery = client_b.client().request(proto::GetRustTestDiscovery {
        project_id: reshared_project_id,
        request_id: 944,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let replace_snapshot = async {
        let deadline = executor.timer(std::time::Duration::from_secs(5));
        assert!(matches!(
            futures::future::select(omitted_started_receiver, deadline).await,
            futures::future::Either::Left((Ok(()), _))
        ));
        assert!(!omitted_release.is_canceled());
        let generation = rust_store.read_with(cx_a, |store, _| store.generation());
        project_a.read_with(cx_a, |project, cx| {
            let provider = project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .expect("pre-omission result provider");
            assert_eq!(
                provider.last_complete_run.as_ref(),
                Some(&preserved_completed_run)
            );
            assert!(provider.completed_runs.contains(&preserved_completed_run));
        });
        let replacement = collaborators
            .iter()
            .filter(|collaborator| collaborator.peer_id != Some(snapshot_member_peer))
            .cloned()
            .collect();
        project_a
            .update(cx_a, |project, cx| {
                project.reshared(
                    proto::ResharedProject {
                        id: reshared_project_id,
                        collaborators: replacement,
                    },
                    cx,
                )
            })
            .expect("valid Rust snapshot omits the requesting peer");
        executor.run_until_parked();
        assert!(*omitted.lock().expect("Rust snapshot omission observation"));
        assert_eq!(
            rust_store.read_with(cx_a, |store, _| store.generation()),
            generation,
            "snapshot expiry must not rely on an input generation change"
        );
        project_a.read_with(cx_a, |project, cx| {
            assert!(!project.collaborators().contains_key(&snapshot_member_peer));
            assert!(project.collaborators().contains_key(&snapshot_other_peer));
            let provider = project
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&RustTestProviderStore::provider_id())
                .expect("retained local Rust provider");
            assert_eq!(
                provider.last_complete_run.as_ref(),
                Some(&preserved_completed_run)
            );
            assert!(provider.completed_runs.contains(&preserved_completed_run));
        });
        assert!(
            omitted_release.send(()).is_err(),
            "owner omission must drop the admitted Rust runner"
        );
    };
    let (response, ()) = futures::join!(pending_discovery, replace_snapshot);
    let error = response.expect_err("omitted peer cannot receive an admitted discovery result");
    assert!(
        error
            .to_string()
            .contains("Rust test discovery request was cancelled"),
        "{error:#}"
    );
    let grant_error = client_b
        .client()
        .request(proto::UpdateRustTestRun {
            project_id: reshared_project_id,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            discovery_generation: recovered.discovery_generation,
            run_id: pending_run_id,
            scope_node_ids: vec![case_id.clone()],
            task_id: "omitted-snapshot-expired-task".into(),
            state: proto::RustTestTaskState::Queued as i32,
            worktree_ids: vec![worktree_id.to_proto()],
            authorization_generation: pending_action.authorization_generation,
            ..Default::default()
        })
        .await
        .expect_err("valid owner omission expires the old run grant");
    assert!(
        grant_error.to_string().contains("Unknown or expired"),
        "{grant_error:#}"
    );
    eprintln!("Rust omitted snapshot: discovery={error:#}; grant={grant_error:#}");
    project_a
        .update(cx_a, |project, cx| {
            project.reshared(
                proto::ResharedProject {
                    id: reshared_project_id,
                    collaborators,
                },
                cx,
            )
        })
        .expect("restore complete Rust collaborator snapshot");
    assert_eq!(client_a.peer_id(), Some(snapshot_host_peer));
    assert_eq!(client_b.peer_id(), Some(snapshot_member_peer));
    assert_eq!(client_c.peer_id(), Some(snapshot_other_peer));
    let recovered_snapshot = client_b
        .client()
        .request(proto::GetRustTestDiscovery {
            project_id: reshared_project_id,
            request_id: 945,
            protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
            worktree_ids: vec![worktree_id.to_proto()],
        })
        .await
        .expect("fresh discovery after restored Rust snapshot");
    assert!(
        recovered_snapshot.status == proto::StructuredProviderStatus::Current as i32
            || recovered_snapshot.status == proto::StructuredProviderStatus::Partial as i32
    );
    let recovered_action = client_b
        .client()
        .request(proto::ResolveRustTestAction {
            discovery_generation: recovered_snapshot.discovery_generation,
            ..resolve()
        })
        .await
        .expect("fresh action after restored Rust snapshot");
    let recovered_plan: serde_json::Value =
        serde_json::from_str(&recovered_action.plan_json).expect("recovered snapshot action plan");
    assert!(recovered_plan["run_id"].as_str().is_some());
    project_a.read_with(cx_a, |project, cx| {
        let provider = project
            .structured_execution_store()
            .read(cx)
            .state()
            .provider(&RustTestProviderStore::provider_id())
            .expect("recovered retained result provider");
        assert_eq!(
            provider.last_complete_run.as_ref(),
            Some(&preserved_completed_run)
        );
        assert!(provider.completed_runs.contains(&preserved_completed_run));
    });
    let (started, started_receiver) = futures::channel::oneshot::channel();
    let (release, release_receiver) = futures::channel::oneshot::channel();
    let (worktree_store, environment, cargo_store, structured_store, lsp_store) = project_a
        .read_with(cx_a, |project, _| {
            (
                project.worktree_store(),
                project.environment().clone(),
                project.cargo_workspace_store().clone(),
                project.structured_execution_store().clone(),
                project.lsp_store(),
            )
        });
    rust_store.update(cx_a, |store, cx| {
        *store = RustTestProviderStore::local_with_runner(
            worktree_store,
            environment,
            cargo_store,
            structured_store,
            Arc::new(RelayGatedDiscoveryRunner {
                started: Mutex::new(Some(started)),
                release: Mutex::new(Some(release_receiver)),
            }),
            cx,
        );
        store.observe_lsp_store(&lsp_store, cx);
    });
    let pending = client_b.client().request(proto::GetRustTestDiscovery {
        project_id: reshared_project_id,
        request_id: 927,
        protocol_version: project::rust_test_provider::RUST_TEST_DISCOVERY_PROTOCOL_VERSION,
        worktree_ids: vec![worktree_id.to_proto()],
    });
    let unshare = async {
        started_receiver
            .await
            .expect("host discovery runner started before unshare");
        active_call_a
            .update(cx_a, |call, cx| call.unshare_project(project_a.clone(), cx))
            .expect("host unshares while member discovery is active");
        executor.run_until_parked();
        assert!(
            release.send(()).is_err(),
            "unsharing must drop the active member discovery runner"
        );
    };
    let (response, ()) = futures::join!(pending, unshare);
    let error = response.expect_err("old sharing session cannot return an active discovery reply");
    assert!(
        error
            .to_string()
            .contains("Rust test discovery request was cancelled"),
        "{error:#}"
    );
}

#[gpui::test]
async fn test_channel_guest_promotion(cx_a: &mut TestAppContext, cx_b: &mut TestAppContext) {
    let mut server = TestServer::start(cx_a.executor()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let active_call_a = cx_a.read(ActiveCall::global);

    let channel_id = server
        .make_public_channel("the-channel", &client_a, cx_a)
        .await;

    let project_a = client_a.build_test_project(cx_a).await;
    cx_a.update(|cx| {
        workspace::join_channel(channel_id, client_a.app_state.clone(), None, None, cx)
    })
    .await
    .unwrap();

    // Client A shares a project in the channel
    active_call_a
        .update(cx_a, |call, cx| call.share_project(project_a.clone(), cx))
        .await
        .unwrap();
    cx_a.run_until_parked();

    // Client B joins channel A as a guest
    cx_b.update(|cx| {
        workspace::join_channel(channel_id, client_b.app_state.clone(), None, None, cx)
    })
    .await
    .unwrap();
    cx_a.run_until_parked();

    // client B opens 1.txt as a guest
    let (workspace_b, cx_b) = client_b.active_workspace(cx_b);
    let room_b = cx_b
        .read(ActiveCall::global)
        .update(cx_b, |call, _| call.room().unwrap().clone());
    cx_b.simulate_keystrokes("cmd-p");
    cx_a.run_until_parked();
    cx_b.simulate_keystrokes("1 enter");

    let (project_b, editor_b) = workspace_b.update(cx_b, |workspace, cx| {
        (
            workspace.project().clone(),
            workspace.active_item_as::<Editor>(cx).unwrap(),
        )
    });
    assert!(project_b.read_with(cx_b, |project, cx| project.is_read_only(cx)));
    assert!(editor_b.update(cx_b, |e, cx| e.read_only(cx)));
    cx_b.update(|_window, cx_b| {
        assert!(room_b.read_with(cx_b, |room, _| !room.can_use_microphone()));
    });
    assert!(
        room_b
            .update(cx_b, |room, cx| room.share_microphone(cx))
            .await
            .is_err()
    );

    // B is promoted
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().unwrap().update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().unwrap(),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .unwrap();
    cx_a.run_until_parked();

    // project and buffers are now editable
    assert!(project_b.read_with(cx_b, |project, cx| !project.is_read_only(cx)));
    assert!(editor_b.update(cx_b, |editor, cx| !editor.read_only(cx)));

    // B sees themselves as muted, and can unmute.
    cx_b.update(|_window, cx_b| {
        assert!(room_b.read_with(cx_b, |room, _| room.can_use_microphone()));
    });
    room_b.read_with(cx_b, |room, _| assert!(room.is_muted()));
    room_b.update(cx_b, |room, cx| room.toggle_mute(cx));
    cx_a.run_until_parked();
    room_b.read_with(cx_b, |room, _| assert!(!room.is_muted()));

    // B is demoted
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().unwrap().update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().unwrap(),
                    proto::ChannelRole::Guest,
                    cx,
                )
            })
        })
        .await
        .unwrap();
    cx_a.run_until_parked();

    // project and buffers are no longer editable
    assert!(project_b.read_with(cx_b, |project, cx| project.is_read_only(cx)));
    assert!(editor_b.update(cx_b, |editor, cx| editor.read_only(cx)));
    assert!(
        room_b
            .update(cx_b, |room, cx| room.share_microphone(cx))
            .await
            .is_err()
    );
}

#[gpui::test]
async fn test_channel_requires_zed_cla(cx_a: &mut TestAppContext, cx_b: &mut TestAppContext) {
    let mut server = TestServer::start(cx_a.executor()).await;
    let client_a = server.create_client(cx_a, "user_a").await;
    let client_b = server.create_client(cx_b, "user_b").await;
    let active_call_a = cx_a.read(ActiveCall::global);
    let active_call_b = cx_b.read(ActiveCall::global);

    // Create a parent channel that requires the Zed CLA
    let parent_channel_id = server
        .make_channel("the-channel", None, (&client_a, cx_a), &mut [])
        .await;
    server
        .app_state
        .db
        .set_channel_requires_zed_cla(ChannelId::from_proto(parent_channel_id.0), true)
        .await
        .unwrap();

    // Create a public channel that is a child of the parent channel.
    let channel_id = client_a
        .channel_store()
        .update(cx_a, |store, cx| {
            store.create_channel("the-sub-channel", Some(parent_channel_id), cx)
        })
        .await
        .unwrap();
    client_a
        .channel_store()
        .update(cx_a, |store, cx| {
            store.set_channel_visibility(parent_channel_id, proto::ChannelVisibility::Public, cx)
        })
        .await
        .unwrap();
    client_a
        .channel_store()
        .update(cx_a, |store, cx| {
            store.set_channel_visibility(channel_id, proto::ChannelVisibility::Public, cx)
        })
        .await
        .unwrap();

    // Users A and B join the channel. B is a guest.
    active_call_a
        .update(cx_a, |call, cx| call.join_channel(channel_id, cx))
        .await
        .unwrap();
    active_call_b
        .update(cx_b, |call, cx| call.join_channel(channel_id, cx))
        .await
        .unwrap();
    cx_a.run_until_parked();
    let room_b = cx_b
        .read(ActiveCall::global)
        .update(cx_b, |call, _| call.room().unwrap().clone());
    cx_b.update(|cx_b| {
        assert!(room_b.read_with(cx_b, |room, _| !room.can_use_microphone()));
    });

    // A tries to grant write access to B, but cannot because B has not
    // yet signed the zed CLA.
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().unwrap().update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().unwrap(),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .unwrap_err();
    cx_a.run_until_parked();
    assert!(room_b.read_with(cx_b, |room, _| !room.can_share_projects()));
    cx_b.update(|cx_b| {
        assert!(room_b.read_with(cx_b, |room, _| !room.can_use_microphone()));
    });

    // A tries to grant write access to B, but cannot because B has not
    // yet signed the zed CLA.
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().unwrap().update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().unwrap(),
                    proto::ChannelRole::Talker,
                    cx,
                )
            })
        })
        .await
        .unwrap();
    cx_a.run_until_parked();
    assert!(room_b.read_with(cx_b, |room, _| !room.can_share_projects()));
    cx_b.update(|cx_b| {
        assert!(room_b.read_with(cx_b, |room, _| room.can_use_microphone()));
    });

    // User B signs the zed CLA.
    let user_b = server
        .app_state
        .user_service
        .get_user_by_github_login("user_b")
        .await
        .unwrap()
        .expect("user_b not found");
    server
        .app_state
        .db
        .add_contributor(user_b.id)
        .await
        .unwrap();

    // A can now grant write access to B.
    active_call_a
        .update(cx_a, |call, cx| {
            call.room().unwrap().update(cx, |room, cx| {
                room.set_participant_role(
                    client_b.user_id().unwrap(),
                    proto::ChannelRole::Member,
                    cx,
                )
            })
        })
        .await
        .unwrap();
    cx_a.run_until_parked();
    assert!(room_b.read_with(cx_b, |room, _| room.can_share_projects()));
    cx_b.update(|cx_b| {
        assert!(room_b.read_with(cx_b, |room, _| room.can_use_microphone()));
    });
}
