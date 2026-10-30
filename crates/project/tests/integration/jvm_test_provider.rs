use std::{borrow::Cow, collections::HashSet, path::Path, sync::Arc};

#[cfg(windows)]
use fs::RealFs;
use fs::{FakeFs, Fs as _};
use gpui::{Entity, TestAppContext};
use language::{
    Language, LanguageConfig, LanguageMatcher, LanguageQueries, Runnable, RunnableRange,
    tree_sitter_typescript,
};
use path::PathStyle;
use project::{
    PathChange, Project, ProjectEntryId, ProjectPath, WorktreeId,
    jvm_test_provider::{
        JVM_JUNIT_PROVIDER_ID, JvmTestBuildTool, JvmTestDiscovery, JvmTestSelector,
        JvmVerifiedCase, MAX_JUNIT_SOURCE_BYTES, collect_junit_reports, fresh_junit_case,
        matching_junit_case, parse_junit_report, selected_junit_task, verify_junit_runnable,
    },
    structured_execution::{
        DiscoveryGeneration, StructuredExecutionState, StructuredNodeKind, StructuredNodeState,
        StructuredProviderId, StructuredProviderSnapshot, StructuredProviderStatus,
        StructuredRunPhase,
    },
    worktree_store::WorktreeStoreEvent,
};
use serde_json::json;
use task::{RunnableTag, StructuredTaskHandle, TaskId};
use util::rel_path::RelPath;

fn scope_path(worktree_id: usize, path: &str) -> ProjectPath {
    ProjectPath {
        worktree_id: WorktreeId::from_usize(worktree_id),
        path: Arc::from(RelPath::from_unix_str(path).expect("valid fixture scope")),
    }
}

#[cfg(unix)]
fn successful_exit_status() -> std::process::ExitStatus {
    use std::os::unix::process::ExitStatusExt as _;
    std::process::ExitStatus::from_raw(0)
}

#[cfg(windows)]
fn successful_exit_status() -> std::process::ExitStatus {
    use std::os::windows::process::ExitStatusExt as _;
    std::process::ExitStatus::from_raw(0)
}

#[gpui::test]
async fn selected_junit_result_requires_a_fresh_host_report(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    let report_path = Path::new("/fixture/target/surefire-reports/TEST-GreetingTest.xml");
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "pom.xml": "<project/>",
            "target": { "surefire-reports": {
                "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            } }
        }),
    )
    .await;
    let selector = JvmTestSelector {
        scope_path: scope_path(1, "pom.xml"),
        module_path: ".".to_string(),
        class_name: "GreetingTest".to_string(),
        method_name: "messageIsAvailable".to_string(),
    };
    let before = collect_junit_reports(
        fs.as_ref(),
        Path::new("/fixture"),
        selector.scope_path.clone(),
        ".",
        JvmTestBuildTool::Maven,
    )
    .await
    .expect("initial JUnit report");
    let (_, previous_version) = matching_junit_case(&before, &selector).expect("selected case");
    assert!(fresh_junit_case(&before, &selector, previous_version).is_err());
    fs.insert_file(
        report_path,
        b"<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"expected greeting\"/></testcase></testsuite>".to_vec(),
    )
    .await;
    let after = collect_junit_reports(
        fs.as_ref(),
        Path::new("/fixture"),
        selector.scope_path.clone(),
        ".",
        JvmTestBuildTool::Maven,
    )
    .await
    .expect("new JUnit report");
    let case = fresh_junit_case(&after, &selector, previous_version).expect("fresh selected case");
    assert_eq!(case.state, StructuredNodeState::Failed);
    assert_eq!(case.message.as_deref(), Some("expected greeting"));
}

fn java_test_runnable() -> RunnableRange {
    let buffer_id = text::BufferId::new(1).expect("fixture buffer ID");
    let mut tags = smallvec::SmallVec::new();
    tags.push(RunnableTag("java-test-method".into()));
    RunnableRange {
        buffer_id,
        run_range: 0..1,
        full_range: 0..1,
        runnable: Runnable {
            tags,
            language: Arc::new(Language::new(
                LanguageConfig {
                    name: "Java".into(),
                    ..LanguageConfig::default()
                },
                None,
            )),
            buffer: buffer_id,
        },
        extra_captures: [
            ("annotation_name".to_string(), "Test".to_string()),
            ("java_package_name".to_string(), "evidence".to_string()),
            ("java_class_name".to_string(), "GreetingTest".to_string()),
            (
                "java_method_name".to_string(),
                "messageIsAvailable".to_string(),
            ),
        ]
        .into_iter()
        .collect(),
    }
}

fn register_java_test_fixture_language(project: &Entity<Project>, cx: &mut TestAppContext) {
    let languages = project.read_with(cx, |project, _| project.languages().clone());
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
    .expect("fixture runnable query");
    languages.add(Arc::new(java));
}

#[gpui::test]
async fn local_project_publishes_only_visible_junit_sources_as_stale(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "application": {
                "pom.xml": "<project/>",
                "src": { "test": { "java": { "evidence": { "GreetingTest.java": "package evidence; class GreetingTest {}" } } } },
                "target": { "surefire-reports": {
                    "TEST-evidence.GreetingTest.xml": include_str!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml"),
                    "TEST-evidence.HiddenTest.xml": "<testsuite><testcase classname=\"evidence.HiddenTest\" name=\"privateMethod\"/></testsuite>"
                } }
            }
        }),
    )
    .await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    let (provider, structured) = project.read_with(cx, |project, _| {
        (
            project
                .jvm_test_provider_store()
                .expect("local JVM provider")
                .clone(),
            project.structured_execution_store().clone(),
        )
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("report refresh");
    assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
    assert!(
        snapshot.diagnostic.as_deref().is_some_and(
            |text| text.contains("only tests run in this session have verified results")
        )
    );
    assert!(
        snapshot
            .diagnostic
            .as_deref()
            .is_some_and(|text| text.contains("display-only") && text.contains("@Test"))
    );
    assert!(
        snapshot
            .nodes
            .iter()
            .any(|node| node.label == "messageIsAvailable")
    );
    assert!(
        !snapshot
            .nodes
            .iter()
            .any(|node| node.label == "privateMethod")
    );
    let published = structured
        .read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId("jvm-junit".to_string()))
                .cloned()
        })
        .expect("published Java provider");
    assert_eq!(published.nodes, snapshot.nodes);
    assert_eq!(published.status, StructuredProviderStatus::Stale);
    assert!(published.current_run.is_none());
}

#[gpui::test]
async fn selected_java_test_rejects_an_ignored_build_wrapper(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            ".gitignore": "mvnw\n",
            "pom.xml": "<project/>",
            "mvnw": "#!/bin/sh\n",
            "src": { "test": { "java": {
                "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
            } } },
            "target": { "surefire-reports": {
                "TEST-GreetingTest.xml": "<testsuite name=\"GreetingTest\"><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            } }
        }),
    )
    .await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    register_java_test_fixture_language(&project, cx);
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local JVM provider")
            .clone()
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("source-backed discovery");
    let case = snapshot
        .nodes
        .iter()
        .find(|node| node.label == "messageIsAvailable")
        .expect("JUnit case");
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_some()));
    let error = provider
        .read_with(cx, |provider, cx| {
            provider.plan_selected_case(&case.id, snapshot.discovery_generation, cx)
        })
        .expect_err("ignored wrapper must not run a selected test");
    assert!(error.to_string().contains("wrapper is unavailable"));
}

#[gpui::test]
async fn first_discovery_waits_for_a_lazy_java_language(cx: &mut TestAppContext) {
    java_discovery_with_a_lazy_grammar(cx, false).await;
}

#[gpui::test]
async fn failed_lazy_java_grammar_keeps_discovery_display_only(cx: &mut TestAppContext) {
    java_discovery_with_a_lazy_grammar(cx, true).await;
}

async fn java_discovery_with_a_lazy_grammar(cx: &mut TestAppContext, missing_grammar: bool) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "pom.xml": "<project/>",
            "mvnw": "#!/bin/sh\n",
            "src": { "test": { "java": {
                "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
            } } },
            "target": { "surefire-reports": {
                "TEST-GreetingTest.xml": "<testsuite name=\"GreetingTest\"><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            } }
        }),
    )
    .await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    let languages = project.read_with(cx, |project, _| project.languages().clone());
    let config = LanguageConfig {
        name: "Java".into(),
        grammar: Some(
            if missing_grammar {
                "missing-jvm-fixture-java"
            } else {
                "jvm-fixture-java"
            }
            .into(),
        ),
        matcher: LanguageMatcher {
            path_suffixes: vec!["java".into()],
            ..Default::default()
        }
        .into(),
        ..Default::default()
    };
    languages.register_native_grammars([(
        "jvm-fixture-java",
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
    )]);
    let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    languages.register_language(
        config.name.clone(),
        config.grammar.clone(),
        config.matcher.clone(),
        false,
        None,
        Arc::new({
            let loads = loads.clone();
            move || {
                loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(language::LoadedLanguage {
                    config: config.clone(),
                    queries: LanguageQueries {
                        runnables: Some(Cow::Borrowed(
                            "(program (class_declaration name: (type_identifier) @java_class_name body: (class_body (decorator (identifier) @annotation_name) (method_definition name: (property_identifier) @run @java_method_name))) @_ (#set! tag \"java-test-method\"))",
                        )),
                        ..Default::default()
                    },
                    context_provider: None,
                    toolchain_provider: None,
                    manifest_name: None,
                })
            }
        }),
    );
    assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 0);
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("JVM provider")
            .clone()
    });
    let refresh = provider.update(cx, |provider, cx| provider.refresh(cx));
    if missing_grammar {
        cx.run_until_parked();
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(5));
    }
    let snapshot = refresh.await.expect("first discovery");
    let case = snapshot
        .nodes
        .iter()
        .find(|node| node.label == "messageIsAvailable")
        .expect("reported Java case");
    assert_eq!(
        provider.read_with(cx, |provider, _| provider.selector(&case.id).is_some()),
        !missing_grammar
    );
    if missing_grammar {
        assert!(snapshot.partial);
        assert!(snapshot.diagnostic.as_deref().is_some_and(|diagnostic| {
            diagnostic.contains("Java syntax loading") && diagnostic.contains("refresh Tests")
        }));
        assert!(snapshot.current_run.is_none());
    }
    assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[gpui::test]
async fn local_provider_verifies_runnable_captures_before_offering_a_selector(
    cx: &mut TestAppContext,
) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "pom.xml": "<project/>",
            "mvnw": "#!/bin/sh\n",
            "src": { "test": { "java": {
                "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
            } } },
            "target": { "surefire-reports": {
                "TEST-GreetingTest.xml": "<testsuite name=\"GreetingTest\"><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            } }
        }),
    ).await;
    let project = Project::test(fs.clone(), [Path::new("/fixture")], cx).await;
    register_java_test_fixture_language(&project, cx);
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local JVM provider")
            .clone()
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("source-backed discovery");
    let case = snapshot
        .nodes
        .iter()
        .find(|node| node.label == "messageIsAvailable")
        .expect("JUnit case");
    let selector = provider.read_with(cx, |provider, _| provider.selector(&case.id).cloned());
    assert_eq!(
        selector.expect("verified selector").method_name,
        "messageIsAvailable"
    );
    assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
    let plan = provider
        .read_with(cx, |provider, cx| {
            provider.plan_selected_case(&case.id, snapshot.discovery_generation, cx)
        })
        .expect("verified Java method Task plan");
    assert_eq!(plan.node_id, case.id);
    assert_eq!(plan.template.command, "./mvnw");
    assert_eq!(
        plan.template.args,
        ["-Dtest=GreetingTest#messageIsAvailable", "test"]
    );
    let captured_version = provider
        .update(cx, |provider, cx| {
            provider.capture_report_version(&plan, cx)
        })
        .await
        .expect("pre-run host report version");
    let repeated_version = provider
        .update(cx, |provider, cx| {
            provider.capture_report_version(&plan, cx)
        })
        .await
        .expect("same pre-run host report version");
    assert_eq!(captured_version, repeated_version);
    assert!(
        provider
            .read_with(cx, |provider, cx| {
                provider.plan_selected_case(
                    &case.id,
                    DiscoveryGeneration(snapshot.discovery_generation.0 + 1),
                    cx,
                )
            })
            .is_err()
    );
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let report_path = Arc::from(
        RelPath::from_unix_str("target/surefire-reports/TEST-GreetingTest.xml")
            .expect("report path"),
    );
    worktree_store.update(cx, |_, cx| {
        cx.emit(WorktreeStoreEvent::WorktreeUpdatedEntries(
            plan.worktree_id,
            Arc::from([(report_path, ProjectEntryId::MIN, PathChange::Updated)]),
        ));
    });
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_some()));
    let handle = StructuredTaskHandle::new(TaskId("java-selected-fixture".to_string()));
    let run_id = provider
        .update(cx, |provider, cx| {
            provider.register_run_handle(plan.clone(), captured_version.clone(), handle.clone(), cx)
        })
        .expect("register selected Java Task");
    cx.update(|cx| {
        assert!(handle.mark_running(None, cx));
    });
    cx.run_until_parked();
    let structured = project.read_with(cx, |project, _| {
        project.structured_execution_store().clone()
    });
    assert_eq!(
        structured.read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|snapshot| snapshot.current_run.as_ref())
                .map(|run| run.phase)
        }),
        Some(StructuredRunPhase::Running)
    );
    assert!(
        provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .is_err()
    );
    assert!(!handle.state().is_terminal());
    fs.insert_file(
        Path::new("/fixture/target/surefire-reports/TEST-GreetingTest.xml"),
        b"<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"expected greeting\"/></testcase></testsuite>".to_vec(),
    )
    .await;
    cx.update(|cx| {
        assert!(handle.mark_completed(successful_exit_status(), cx));
    });
    cx.run_until_parked();
    let completed = structured
        .read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|snapshot| snapshot.last_complete_run.clone())
        })
        .expect("completed selected Java run");
    assert_eq!(completed.id, run_id);
    assert_eq!(completed.phase, StructuredRunPhase::Completed);
    assert_eq!(completed.summary.failed, 1);
    assert_eq!(completed.events[0].state, StructuredNodeState::Failed);
    assert_eq!(
        completed.events[0].message.as_deref(),
        Some("expected greeting")
    );
    assert_eq!(
        provider
            .read_with(cx, |provider, cx| {
                provider.plan_failed_case(&case.id, snapshot.discovery_generation, &run_id, cx)
            })
            .expect("current failed case can be rerun"),
        plan
    );
    let current_version = provider
        .update(cx, |provider, cx| {
            provider.capture_report_version(&plan, cx)
        })
        .await
        .expect("second run report baseline");
    let cancelled_handle = StructuredTaskHandle::new(TaskId("java-cancelled-fixture".to_string()));
    let cancelled_run = provider
        .update(cx, |provider, cx| {
            provider.register_run_handle(
                plan.clone(),
                current_version,
                cancelled_handle.clone(),
                cx,
            )
        })
        .expect("register cancellable Java Task");
    provider
        .update(cx, |provider, cx| provider.cancel_run(&cancelled_run, cx))
        .expect("cancel selected Java Task");
    cx.run_until_parked();
    let cancelled = structured
        .read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|snapshot| snapshot.current_run.clone())
        })
        .expect("cancelled Java run");
    assert_eq!(cancelled.id, cancelled_run);
    assert_eq!(cancelled.phase, StructuredRunPhase::Cancelled);
    assert_eq!(cancelled.summary.cancelled, 1);
    assert!(
        provider
            .read_with(cx, |provider, cx| {
                provider.plan_failed_case(&case.id, snapshot.discovery_generation, &run_id, cx)
            })
            .is_err(),
        "an older failed run cannot be rerun after a newer run starts"
    );
    cx.update(|cx| {
        assert!(!cancelled_handle.mark_completed(successful_exit_status(), cx));
    });
    let unchanged_version = provider
        .update(cx, |provider, cx| {
            provider.capture_report_version(&plan, cx)
        })
        .await
        .expect("unchanged report baseline");
    let unverified_handle = StructuredTaskHandle::new(TaskId("java-no-report-fixture".to_string()));
    let unverified_run = provider
        .update(cx, |provider, cx| {
            provider.register_run_handle(
                plan.clone(),
                unchanged_version,
                unverified_handle.clone(),
                cx,
            )
        })
        .expect("register unverified Java Task");
    cx.update(|cx| {
        assert!(unverified_handle.mark_running(None, cx));
        assert!(unverified_handle.mark_completed(successful_exit_status(), cx));
    });
    cx.run_until_parked();
    let unverified = structured
        .read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|snapshot| snapshot.current_run.clone())
        })
        .expect("unverified Java run");
    assert_eq!(unverified.id, unverified_run);
    assert_eq!(unverified.phase, StructuredRunPhase::Completed);
    assert_eq!(unverified.summary.passed, 0);
    assert_eq!(unverified.summary.failed, 0);
    assert!(
        unverified
            .diagnostic
            .as_deref()
            .is_some_and(|diagnostic| diagnostic.contains("without a fresh readable JUnit result"))
    );
    let stale_version = provider
        .update(cx, |provider, cx| {
            provider.capture_report_version(&plan, cx)
        })
        .await
        .expect("stale-run report baseline");
    let stale_handle = StructuredTaskHandle::new(TaskId("java-stale-fixture".to_string()));
    let stale_run = provider
        .update(cx, |provider, cx| {
            provider.register_run_handle(plan.clone(), stale_version, stale_handle.clone(), cx)
        })
        .expect("register soon-stale Java Task");
    cx.update(|cx| {
        assert!(stale_handle.mark_running(None, cx));
    });
    cx.run_until_parked();
    let source = project.read_with(cx, |project, cx| {
        let source = project
            .find_project_path(Path::new("/fixture/src/test/java/GreetingTest.java"), cx)
            .expect("visible source path");
        source
    });
    let buffer = project
        .update(cx, |project, cx| project.open_buffer(source, cx))
        .await
        .expect("source buffer");
    buffer.update(cx, |buffer, cx| {
        buffer.edit([(0..0, "// edited\n")], None, cx);
    });
    cx.run_until_parked();
    assert!(stale_handle.state().is_terminal());
    let invalidated = structured
        .read_with(cx, |store, _| {
            store
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.to_string()))
                .and_then(|snapshot| snapshot.current_run.clone())
        })
        .expect("invalidated Java run");
    assert_eq!(invalidated.id, stale_run);
    assert_eq!(invalidated.phase, StructuredRunPhase::Cancelled);
    cx.update(|cx| {
        assert!(!stale_handle.mark_completed(successful_exit_status(), cx));
    });
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_none()));
    assert!(
        provider
            .update(cx, |provider, cx| provider
                .capture_report_version(&plan, cx))
            .await
            .is_err()
    );
    assert!(
        provider
            .read_with(cx, |provider, cx| {
                provider.plan_selected_case(&case.id, snapshot.discovery_generation, cx)
            })
            .is_err()
    );
    let edited_snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("source refresh");
    assert_eq!(edited_snapshot.status, StructuredProviderStatus::Stale);
    assert!(edited_snapshot.partial);
    assert!(
        edited_snapshot
            .diagnostic
            .as_deref()
            .is_some_and(|text| text.contains("save edited files"))
    );
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_none()));
    project
        .update(cx, |project, cx| project.save_buffer(buffer, cx))
        .await
        .expect("save edited Java test source");
    let saved_snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("refresh saved Java test source");
    let saved_case = saved_snapshot
        .nodes
        .iter()
        .find(|node| node.label == "messageIsAvailable")
        .expect("saved Java method");
    assert!(provider.read_with(cx, |provider, _| {
        provider.selector(&saved_case.id).is_some()
    }));
    provider
        .read_with(cx, |provider, cx| {
            provider.plan_selected_case(&saved_case.id, saved_snapshot.discovery_generation, cx)
        })
        .expect("saved Java method is authorized");
    project.update(cx, |project, cx| project.close(cx));
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_none()));
    assert!(
        provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .is_err()
    );
}

#[gpui::test]
async fn restricting_a_java_worktree_cancels_its_retained_run(cx: &mut TestAppContext) {
    use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
    use task::{StructuredTaskHandle, StructuredTaskState, TaskId};

    super::init_test(cx);
    cx.update(|cx| project::trusted_worktrees::init(Default::default(), cx));
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "pom.xml": "<project/>",
            "mvnw": "#!/bin/sh\n",
            "src": { "test": { "java": {
                "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }"
            } } },
            "target": { "surefire-reports": {
                "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            } },
        }),
    ).await;
    fs.insert_tree(Path::new("/other"), json!({})).await;
    let project = Project::test_with_worktree_trust(fs.clone(), [Path::new("/fixture")], cx).await;
    register_java_test_fixture_language(&project, cx);
    let worktree_store = project.read_with(cx, |project, _| project.worktree_store());
    let worktree_id = worktree_store.read_with(cx, |store, cx| {
        store
            .visible_worktrees(cx)
            .next()
            .expect("fixture worktree")
            .read(cx)
            .id()
    });
    let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("shared trust service"));
    trust.update(cx, |trust, cx| {
        trust.trust(
            &worktree_store,
            [PathTrust::Worktree(worktree_id)].into_iter().collect(),
            cx,
        );
    });
    cx.run_until_parked();
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("Java provider")
            .clone()
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("Java discovery");
    let case = snapshot
        .nodes
        .iter()
        .find(|node| node.kind == StructuredNodeKind::Case)
        .expect("Java case");
    let plan = provider
        .read_with(cx, |provider, cx| {
            provider.plan_selected_case(&case.id, snapshot.discovery_generation, cx)
        })
        .expect("current Java plan");
    let report_version = provider
        .update(cx, |provider, cx| {
            provider.capture_report_version(&plan, cx)
        })
        .await
        .expect("prior report version");
    let handle = StructuredTaskHandle::new(TaskId("restricted-java-run".into()));
    let run_id = provider
        .update(cx, |provider, cx| {
            provider.register_run_handle(plan, report_version, handle.clone(), cx)
        })
        .expect("retained Java run");
    cx.update(|cx| assert!(handle.mark_running(None, cx)));
    cx.run_until_parked();

    let other_project = Project::test_with_worktree_trust(fs, [Path::new("/other")], cx).await;
    let other_store = other_project.read_with(cx, |project, _| project.worktree_store());
    let other_id = other_store.read_with(cx, |store, cx| {
        store
            .visible_worktrees(cx)
            .next()
            .expect("other worktree")
            .read(cx)
            .id()
    });
    trust.update(cx, |trust, cx| {
        trust.restrict(
            other_store.downgrade(),
            [PathTrust::Worktree(other_id)].into_iter().collect(),
            cx,
        );
    });
    cx.run_until_parked();
    assert!(matches!(
        handle.state(),
        StructuredTaskState::Running { .. }
    ));
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_some()));

    trust.update(cx, |trust, cx| {
        trust.restrict(
            worktree_store.downgrade(),
            [PathTrust::Worktree(worktree_id)].into_iter().collect(),
            cx,
        );
    });
    cx.run_until_parked();
    assert!(matches!(
        handle.state(),
        StructuredTaskState::Cancelled { .. }
    ));
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_none()));
    assert!(
        provider
            .read_with(cx, |provider, cx| {
                provider.plan_selected_case(&case.id, snapshot.discovery_generation, cx)
            })
            .is_err()
    );
    let run = project.read_with(cx, |project, cx| {
        project
            .structured_execution_store()
            .read(cx)
            .state()
            .provider(&StructuredProviderId("jvm-junit".into()))
            .and_then(|provider| provider.current_run.clone())
            .expect("cancelled Java result")
    });
    assert_eq!(run.id, run_id);
    assert_eq!(run.phase, StructuredRunPhase::Cancelled);
    cx.update(|cx| assert!(!handle.mark_completed(successful_exit_status(), cx)));
}

#[gpui::test]
async fn oversized_java_source_keeps_cached_case_display_only(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "pom.xml": "<project/>",
            "src": { "test": { "java": { "GreetingTest.java": "A".repeat(MAX_JUNIT_SOURCE_BYTES + 1) } } },
            "target": { "surefire-reports": {
                "TEST-GreetingTest.xml": "<testsuite name=\"GreetingTest\"><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>"
            } }
        }),
    ).await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local JVM provider")
            .clone()
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("bounded discovery");
    assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
    assert!(snapshot.partial);
    assert!(
        snapshot
            .diagnostic
            .as_deref()
            .is_some_and(|text| text.contains("reduce their size"))
    );
    let case = snapshot
        .nodes
        .iter()
        .find(|node| node.label == "messageIsAvailable")
        .expect("cached case");
    assert!(provider.read_with(cx, |provider, _| provider.selector(&case.id).is_none()));
}

#[gpui::test]
async fn partial_report_failure_does_not_make_cached_cases_current(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "application": {
                "pom.xml": "<project/>",
                "src": { "test": { "java": { "evidence": { "GreetingTest.java": "class GreetingTest {}" } } } },
                "target": { "surefire-reports": {
                    "TEST-evidence.GreetingTest.xml": include_str!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml")
                } }
            },
            "broken": {
                "pom.xml": "<project/>",
                "target": { "surefire-reports": { "TEST-broken.xml": "<testsuite><testcase" } }
            }
        }),
    )
    .await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local JVM provider")
            .clone()
    });
    let snapshot = provider
        .update(cx, |provider, cx| provider.refresh(cx))
        .await
        .expect("partial report refresh");
    assert_eq!(snapshot.status, StructuredProviderStatus::Stale);
    assert!(snapshot.partial);
    assert!(
        snapshot
            .nodes
            .iter()
            .any(|node| node.label == "messageIsAvailable")
    );
    let diagnostic = snapshot.diagnostic.expect("stale and failure explanation");
    assert!(diagnostic.contains("only tests run in this session have verified results"));
    assert!(diagnostic.contains("Could not read reports"));
}

#[gpui::test]
async fn closing_local_project_invalidates_its_jvm_provider(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(Path::new("/fixture"), json!({})).await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local JVM provider")
            .clone()
    });
    project.update(cx, |project, cx| project.close(cx));
    drop(project);
    assert!(
        provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .is_err()
    );
}

#[gpui::test]
async fn releasing_local_project_closes_a_retained_jvm_provider(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(Path::new("/fixture"), json!({})).await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    let provider = project.read_with(cx, |project, _| {
        project
            .jvm_test_provider_store()
            .expect("local JVM provider")
            .clone()
    });
    drop(project);
    cx.executor().run_until_parked();
    assert!(
        provider
            .update(cx, |provider, cx| provider.refresh(cx))
            .await
            .is_err()
    );
}

#[cfg(feature = "rust-tests")]
#[gpui::test]
async fn local_jvm_provider_coexists_with_rust_provider(cx: &mut TestAppContext) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(Path::new("/fixture"), json!({})).await;
    let project = Project::test(fs, [Path::new("/fixture")], cx).await;
    project.read_with(cx, |project, _| {
        assert!(project.jvm_test_provider_store().is_some());
        assert_ne!(
            project.rust_test_provider_store().entity_id(),
            project
                .jvm_test_provider_store()
                .expect("JVM provider")
                .entity_id()
        );
    });
}

#[test]
fn java_runnable_verifies_only_the_matching_junit_method_and_source() {
    let report = parse_junit_report(
        include_bytes!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml"),
        scope_path(1, "application/pom.xml"),
        "application",
    )
    .expect("Maven fixture report");
    let case = &report.cases[0];
    let source = scope_path(1, "application/src/test/java/evidence/GreetingTest.java");
    let runnable = java_test_runnable();
    let verified = verify_junit_runnable(&report, case, &source, &runnable)
        .expect("exact Java runnable verifies the report case");
    assert_eq!(verified.method_name, "messageIsAvailable");
    assert_eq!(verified.class_name, "evidence.GreetingTest");

    let mut wrong_method = java_test_runnable();
    wrong_method.extra_captures.insert(
        "java_method_name".to_string(),
        "differentMethod".to_string(),
    );
    assert!(verify_junit_runnable(&report, case, &source, &wrong_method).is_none());
    let mut wrong_annotation = java_test_runnable();
    wrong_annotation.extra_captures.insert(
        "annotation_name".to_string(),
        "ParameterizedTest".to_string(),
    );
    assert!(verify_junit_runnable(&report, case, &source, &wrong_annotation).is_none());
    let other_source = scope_path(1, "other/src/test/java/evidence/GreetingTest.java");
    assert!(verify_junit_runnable(&report, case, &other_source, &runnable).is_none());
    let other_worktree = scope_path(2, "application/src/test/java/evidence/GreetingTest.java");
    assert!(verify_junit_runnable(&report, case, &other_worktree, &runnable).is_none());
    let mut custom_label = case.clone();
    custom_label.display_name = "A friendly test name".to_string();
    assert!(verify_junit_runnable(&report, &custom_label, &source, &runnable).is_none());

    let gradle_report = parse_junit_report(
        include_bytes!("../../test_data/groovy_jvm_gradle/reports/TEST-evidence.GreetingTest.xml"),
        scope_path(1, "application/build.gradle.kts"),
        "application",
    )
    .expect("Gradle fixture report");
    assert!(
        verify_junit_runnable(&gradle_report, &gradle_report.cases[0], &source, &runnable)
            .is_some(),
        "Gradle's method() display name maps to the same source runnable"
    );
}

#[gpui::test]
async fn collects_checked_in_maven_and_gradle_reports_from_the_host_worktree(
    cx: &mut TestAppContext,
) {
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "application": {
                "pom.xml": "<project/>",
                "target": {
                    "surefire-reports": {
                        "TEST-evidence.GreetingTest.xml": include_str!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml"),
                        "application.log": "not a JUnit report"
                    }
                }
            },
            "gradle-app": {
                "build.gradle.kts": "plugins {}",
                "build": {
                    "test-results": {
                        "test": {
                            "TEST-evidence.GreetingTest.xml": include_str!("../../test_data/groovy_jvm_gradle/reports/TEST-evidence.GreetingTest.xml")
                        }
                    }
                }
            }
        }),
    )
    .await;
    let reports = collect_junit_reports(
        fs.as_ref(),
        Path::new("/fixture"),
        scope_path(1, "application/pom.xml"),
        "application",
        JvmTestBuildTool::Maven,
    )
    .await
    .expect("bounded Maven report collection");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].cases.len(), 1);
    assert_eq!(reports[0].cases[0].class_name, "evidence.GreetingTest");
    assert_eq!(reports[0].cases[0].state, StructuredNodeState::Passed);

    let gradle_reports = collect_junit_reports(
        fs.as_ref(),
        Path::new("/fixture"),
        scope_path(1, "gradle-app/build.gradle.kts"),
        "gradle-app",
        JvmTestBuildTool::Gradle,
    )
    .await
    .expect("bounded Gradle report collection");
    assert_eq!(gradle_reports.len(), 1);
    assert_eq!(gradle_reports[0].cases.len(), 1);
    assert_eq!(
        gradle_reports[0].cases[0].state,
        StructuredNodeState::Passed
    );
}

#[gpui::test]
async fn rejects_junit_report_symlink_outside_the_host_worktree(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "application": {
                "pom.xml": "<project/>",
                "target": { "surefire-reports": {} }
            }
        }),
    )
    .await;
    fs.insert_tree(
        Path::new("/outside"),
        json!({ "report.xml": "<testsuite/>" }),
    )
    .await;
    fs.create_symlink(
        Path::new("/fixture/application/target/surefire-reports/TEST-external.xml"),
        Path::new("/outside/report.xml").to_path_buf(),
    )
    .await
    .expect("create report symlink");
    let error = collect_junit_reports(
        fs.as_ref(),
        Path::new("/fixture"),
        scope_path(1, "application/pom.xml"),
        "application",
        JvmTestBuildTool::Maven,
    )
    .await
    .expect_err("external report must not be read");
    assert!(error.to_string().contains("not a regular file"));
}

#[cfg(windows)]
#[gpui::test]
async fn collects_windows_maven_report_and_rejects_parent_junction(cx: &mut TestAppContext) {
    // RealFs metadata waits for OS file I/O outside GPUI's test scheduler.
    cx.executor().allow_parking();

    let temporary = tempfile::tempdir().expect("temporary worktree");
    let root = temporary.path().join("root");
    let application = root.join("application");
    let target = application.join("target");
    let report_directory = target.join("surefire-reports");
    std::fs::create_dir_all(&report_directory).expect("create Maven report directory");
    std::fs::write(application.join("pom.xml"), "<project/>").expect("write Maven manifest");
    std::fs::write(
        report_directory.join("TEST-evidence.GreetingTest.xml"),
        include_str!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml"),
    )
    .expect("write Maven report");
    let fs = RealFs::new(None, cx.executor());
    let scope = scope_path(1, "application/pom.xml");
    let reports = collect_junit_reports(
        &fs,
        &root,
        scope.clone(),
        "application",
        JvmTestBuildTool::Maven,
    )
    .await
    .expect("collect Windows Maven report");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].cases.len(), 1);
    assert_eq!(reports[0].cases[0].state, StructuredNodeState::Passed);

    std::fs::remove_dir_all(&target).expect("remove Maven target");
    let outside = temporary.path().join("outside");
    let outside_reports = outside.join("surefire-reports");
    std::fs::create_dir_all(&outside_reports).expect("create outside report directory");
    std::fs::write(
        outside_reports.join("TEST-evidence.GreetingTest.xml"),
        include_str!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml"),
    )
    .expect("write outside report");
    let output = std::process::Command::new("cmd")
        .arg("/C")
        .arg("mklink")
        .arg("/J")
        .arg(&target)
        .arg(&outside)
        .output()
        .expect("create Maven target junction");
    assert!(
        output.status.success(),
        "Maven target junction creation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        collect_junit_reports(&fs, &root, scope, "application", JvmTestBuildTool::Maven,)
            .await
            .is_err(),
        "a parent junction must not expose an outside JUnit report"
    );
    std::fs::remove_dir(&target).expect("remove Maven target junction");
}

#[test]
fn checked_in_gradle_report_keeps_the_verified_junit_method_runnable() {
    let report = parse_junit_report(
        include_bytes!("../../test_data/groovy_jvm_gradle/reports/TEST-evidence.GreetingTest.xml"),
        scope_path(1, "application/build.gradle.kts"),
        "application",
    )
    .expect("Gradle fixture report");
    let verified_cases = HashSet::from([JvmVerifiedCase {
        scope_path: report.scope_path.clone(),
        module_path: "application".to_string(),
        class_name: "evidence.GreetingTest".to_string(),
        display_name: "messageIsAvailable()".to_string(),
        method_name: "messageIsAvailable".to_string(),
    }]);
    let discovery =
        JvmTestDiscovery::from_reports(&[report], DiscoveryGeneration(1), &verified_cases)
            .expect("verified Gradle report");
    assert_eq!(discovery.selectors.len(), 1);
    assert_eq!(discovery.results.len(), 1);
    assert_eq!(discovery.results[0].1, StructuredNodeState::Passed);
    let task = selected_junit_task(
        discovery.selectors.values().next().expect("selected case"),
        JvmTestBuildTool::Gradle,
        PathStyle::Unix,
    )
    .expect("Gradle fixture selector");
    assert_eq!(task.command, "./gradlew");
    assert!(task.show_summary);
    assert_eq!(
        task.args,
        [
            ":application:test",
            "--tests",
            "evidence.GreetingTest.messageIsAvailable",
            "--rerun-tasks",
        ]
    );
}

#[gpui::test]
async fn matches_only_an_unambiguous_verified_gradle_method_report(cx: &mut TestAppContext) {
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(
        Path::new("/fixture"),
        json!({
            "application": {
                "build.gradle.kts": "plugins {}",
                "build": { "test-results": { "test": {
                    "TEST-evidence.GreetingTest.xml": include_str!("../../test_data/groovy_jvm_gradle/reports/TEST-evidence.GreetingTest.xml")
                } } }
            }
        }),
    )
    .await;
    let selector = JvmTestSelector {
        scope_path: scope_path(1, "application/build.gradle.kts"),
        module_path: "application".to_string(),
        class_name: "evidence.GreetingTest".to_string(),
        method_name: "messageIsAvailable".to_string(),
    };
    let reports = collect_junit_reports(
        fs.as_ref(),
        Path::new("/fixture"),
        selector.scope_path.clone(),
        "application",
        JvmTestBuildTool::Gradle,
    )
    .await
    .expect("checked-in Gradle report");
    let (case, _) = matching_junit_case(&reports, &selector).expect("exact Gradle method()");
    assert_eq!(case.display_name, "messageIsAvailable()");

    let mut parameterized = reports.clone();
    parameterized[0].cases[0].display_name = "messageIsAvailable(parameters)".to_string();
    assert!(matching_junit_case(&parameterized, &selector).is_err());

    let mut ambiguous = reports.clone();
    let mut duplicate = ambiguous[0].cases[0].clone();
    duplicate.display_name = "messageIsAvailable".to_string();
    ambiguous[0].cases.push(duplicate);
    assert!(matching_junit_case(&ambiguous, &selector).is_err());
}

#[test]
fn checked_in_maven_report_keeps_the_verified_junit_method_runnable() {
    let report = parse_junit_report(
        include_bytes!("../../test_data/groovy_jvm_maven/reports/TEST-evidence.GreetingTest.xml"),
        scope_path(1, "application/pom.xml"),
        "application",
    )
    .expect("Maven fixture report");
    let verified_cases = HashSet::from([JvmVerifiedCase {
        scope_path: report.scope_path.clone(),
        module_path: "application".to_string(),
        class_name: "evidence.GreetingTest".to_string(),
        display_name: "messageIsAvailable".to_string(),
        method_name: "messageIsAvailable".to_string(),
    }]);
    let discovery =
        JvmTestDiscovery::from_reports(&[report], DiscoveryGeneration(1), &verified_cases)
            .expect("verified Maven report");
    assert_eq!(discovery.selectors.len(), 1);
    assert_eq!(discovery.results.len(), 1);
    assert_eq!(discovery.results[0].1, StructuredNodeState::Passed);
    let task = selected_junit_task(
        discovery.selectors.values().next().expect("selected case"),
        JvmTestBuildTool::Maven,
        PathStyle::Unix,
    )
    .expect("Maven fixture selector");
    assert_eq!(task.command, "./mvnw");
    assert_eq!(
        task.args,
        [
            "-pl",
            "application",
            "-am",
            "-Dsurefire.failIfNoSpecifiedTests=false",
            "-Dtest=evidence.GreetingTest#messageIsAvailable",
            "test",
        ]
    );
}

#[test]
fn verified_java_selection_resolves_to_one_wrapper_task_without_a_shell_script() {
    let report = parse_junit_report(
        br#"<testsuite name="org.example.SampleTest"><testcase classname="org.example.SampleTest" name="works"/></testsuite>"#,
        scope_path(1, "app/pom.xml"),
        "app",
    )
    .expect("JUnit report");
    let verified_cases = HashSet::from([JvmVerifiedCase {
        scope_path: report.scope_path.clone(),
        module_path: report.module_path.clone(),
        class_name: "org.example.SampleTest".to_string(),
        display_name: "works".to_string(),
        method_name: "works".to_string(),
    }]);
    let discovery =
        JvmTestDiscovery::from_reports(&[report], DiscoveryGeneration(1), &verified_cases)
            .expect("verified report selection");
    let maven = discovery
        .selectors
        .values()
        .next()
        .expect("one verified selector")
        .clone();
    let maven_task = selected_junit_task(&maven, JvmTestBuildTool::Maven, PathStyle::Unix)
        .expect("Maven module selector");
    assert_eq!(maven_task.command, "./mvnw");
    assert_eq!(
        maven_task.args,
        [
            "-pl",
            "app",
            "-am",
            "-Dsurefire.failIfNoSpecifiedTests=false",
            "-Dtest=org.example.SampleTest#works",
            "test",
        ]
    );
    assert_eq!(maven_task.cwd.as_deref(), Some("$ZED_WORKTREE_ROOT"));
    assert!(maven_task.show_summary);
    let windows_maven = selected_junit_task(&maven, JvmTestBuildTool::Maven, PathStyle::Windows)
        .expect("Windows Maven module selector");
    assert_eq!(windows_maven.command, ".\\mvnw.cmd");

    let gradle = JvmTestSelector {
        scope_path: scope_path(1, "service/api/build.gradle.kts"),
        module_path: "service/api".to_string(),
        ..maven.clone()
    };
    let gradle_task = selected_junit_task(&gradle, JvmTestBuildTool::Gradle, PathStyle::Unix)
        .expect("Gradle module selector");
    assert_eq!(gradle_task.command, "./gradlew");
    assert_eq!(
        gradle_task.args,
        [
            ":service:api:test",
            "--tests",
            "org.example.SampleTest.works",
            "--rerun-tasks",
        ]
    );
    let windows_gradle = selected_junit_task(&gradle, JvmTestBuildTool::Gradle, PathStyle::Windows)
        .expect("Windows Gradle module selector");
    assert_eq!(windows_gradle.command, ".\\gradlew.bat");

    for invalid in [
        JvmTestSelector {
            method_name: "works; echo unexpected".to_string(),
            ..maven.clone()
        },
        JvmTestSelector {
            module_path: "../app".to_string(),
            ..maven.clone()
        },
        JvmTestSelector {
            scope_path: scope_path(1, "other/pom.xml"),
            ..maven
        },
    ] {
        assert!(selected_junit_task(&invalid, JvmTestBuildTool::Maven, PathStyle::Unix).is_err());
    }
}

#[test]
fn junit_reports_keep_module_case_identity_and_withhold_ambiguous_selectors() {
    let maven = parse_junit_report(
        br#"<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="org.example.SampleTest">
  <testcase classname="org.example.SampleTest" name="works"/>
  <testcase classname="org.example.SampleTest" name="fails"><failure message="expected 1 &amp; got 2"/></testcase>
  <testcase classname="org.example.SampleTest" name="duplicate"/>
  <testcase classname="org.example.SampleTest" name="duplicate"/>
  <testcase classname="org.example.SampleTest" name="addition(int)[1]"/>
  <testcase classname="org.example.SampleTest" name="prettyName"/>
</testsuite>"#,
        scope_path(1, "maven/app/pom.xml"),
        "maven/app",
    )
    .expect("Maven Surefire report");
    let gradle = parse_junit_report(
        br#"<testsuite name="org.example.SampleTest">
  <testcase classname="org.example.SampleTest" name="works"/>
  <testcase classname="org.example.SampleTest" name="skips"><skipped/></testcase>
</testsuite>"#,
        scope_path(1, "gradle/lib/build.gradle.kts"),
        "gradle/lib",
    )
    .expect("Gradle JUnit report");

    let verified_cases = [
        ("maven/app", "works"),
        ("maven/app", "fails"),
        ("gradle/lib", "works"),
        ("gradle/lib", "skips"),
    ]
    .into_iter()
    .map(|(module_path, method_name)| JvmVerifiedCase {
        scope_path: scope_path(
            1,
            if module_path == "maven/app" {
                "maven/app/pom.xml"
            } else {
                "gradle/lib/build.gradle.kts"
            },
        ),
        module_path: module_path.to_string(),
        class_name: "org.example.SampleTest".to_string(),
        display_name: method_name.to_string(),
        method_name: method_name.to_string(),
    })
    .chain([
        JvmVerifiedCase {
            scope_path: scope_path(1, "maven/app/pom.xml"),
            module_path: "maven/app".to_string(),
            class_name: "org.example.SampleTest".to_string(),
            display_name: "duplicate".to_string(),
            method_name: "duplicate".to_string(),
        },
        JvmVerifiedCase {
            scope_path: scope_path(1, "maven/app/pom.xml"),
            module_path: "maven/app".to_string(),
            class_name: "org.example.SampleTest".to_string(),
            display_name: "prettyName".to_string(),
            method_name: "works".to_string(),
        },
    ])
    .collect();
    let discovery = JvmTestDiscovery::from_reports(
        &[maven.clone(), gradle.clone()],
        DiscoveryGeneration(1),
        &verified_cases,
    )
    .expect("bounded discovery");
    assert_eq!(discovery.snapshot.provider_id.0, "jvm-junit");
    assert_eq!(discovery.snapshot.status, StructuredProviderStatus::Current);
    assert_eq!(
        discovery
            .snapshot
            .nodes
            .iter()
            .filter(|node| node.kind == StructuredNodeKind::Case)
            .count(),
        8
    );
    assert_eq!(discovery.selectors.len(), 4);
    assert!(discovery.selectors.values().any(|selector| {
        selector.module_path == "maven/app"
            && selector.class_name == "org.example.SampleTest"
            && selector.method_name == "works"
    }));
    assert!(!discovery.selectors.values().any(|selector| {
        matches!(
            selector.method_name.as_str(),
            "duplicate" | "addition" | "prettyName"
        )
    }));
    assert!(discovery.results.iter().any(|(_, state, message)| {
        *state == StructuredNodeState::Failed && message.as_deref() == Some("expected 1 & got 2")
    }));
    assert!(
        discovery
            .results
            .iter()
            .any(|(_, state, _)| *state == StructuredNodeState::Skipped)
    );

    let reordered =
        JvmTestDiscovery::from_reports(&[gradle, maven], DiscoveryGeneration(1), &verified_cases)
            .expect("report order does not change runnable identity");
    assert_eq!(
        discovery
            .selectors
            .keys()
            .collect::<std::collections::HashSet<_>>(),
        reordered.selectors.keys().collect()
    );

    let mut hidden_state = StructuredExecutionState::new(9);
    let hidden = hidden_state
        .apply_discovery(9, discovery.snapshot.clone(), Some(&HashSet::new()))
        .expect("permission filtered discovery");
    assert_eq!(
        hidden,
        project::structured_execution::StructuredApplyOutcome::Truncated
    );
    assert_eq!(
        hidden_state
            .providers()
            .next()
            .expect("provider")
            .nodes
            .len(),
        1
    );

    let mut state = StructuredExecutionState::new(9);
    state
        .apply_discovery(
            9,
            discovery.snapshot,
            Some(&HashSet::from([WorktreeId::from_usize(1)])),
        )
        .expect("JVM provider shares the structured store");
    state
        .apply_discovery(
            9,
            StructuredProviderSnapshot::discovery(
                StructuredProviderId("rust-tests".into()),
                DiscoveryGeneration(1),
                StructuredProviderStatus::Empty,
                Vec::new(),
            ),
            None,
        )
        .expect("Rust provider remains independent");
    assert_eq!(state.providers().count(), 2);
    assert!(state.apply_discovery(8, reordered.snapshot, None,).is_err());
}

#[test]
fn junit_report_rejects_unbounded_or_unsafe_input() {
    let scope = scope_path(1, "app/pom.xml");
    assert!(parse_junit_report(b"<!DOCTYPE testsuite><testsuite/>", scope.clone(), "app").is_err());
    assert!(parse_junit_report(b"<testsuite><testcase", scope.clone(), "app").is_err());
    assert!(parse_junit_report(&vec![b'x'; 4 * 1024 * 1024 + 1], scope.clone(), "app").is_err());
    assert!(
        parse_junit_report(
            b"<testsuite><testcase name=\"missing-class\"/></testsuite>",
            scope,
            "app"
        )
        .is_err()
    );
}

#[gpui::test]
async fn doctor_jvm_reproduction_owner_limits_apply_to_baseline_and_fresh_result(
    cx: &mut TestAppContext,
) {
    super::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    fs.insert_tree(Path::new("/fixture"), json!({
        "pom.xml": "<project/>", "mvnw": "#!/bin/sh\n",
        "src": { "test": { "java": { "GreetingTest.java": "class GreetingTest { @Test messageIsAvailable() {} }" } } },
        "target": { "surefire-reports": { "TEST-GreetingTest.xml": "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"/></testsuite>" } }
    })).await;
    let project = Project::test(fs.clone(), [Path::new("/fixture")], cx).await;
    register_java_test_fixture_language(&project, cx);
    let provider = project
        .read_with(cx, |owner, _| owner.jvm_test_provider_store().cloned())
        .expect("Native JVM owner");
    let snapshot = provider
        .update(cx, |owner, cx| owner.refresh(cx))
        .await
        .expect("Source-backed discovery");
    let node = snapshot
        .nodes
        .iter()
        .find(|node| node.label == "messageIsAvailable")
        .expect("Selected case");
    let plan = provider
        .read_with(cx, |owner, cx| {
            owner.plan_selected_case(&node.id, snapshot.discovery_generation, cx)
        })
        .expect("Exact selected owner plan");
    for maximum in [
        0,
        1,
        project::jvm_test_provider::MAX_JUNIT_DISCOVERY_BYTES + 1,
    ] {
        assert!(
            provider
                .update(cx, |owner, cx| owner
                    .capture_report_version_with_limit(&plan, maximum, cx))
                .await
                .is_err()
        );
    }
    let version = provider
        .update(cx, |owner, cx| {
            owner.capture_report_version_with_limit(&plan, 256, cx)
        })
        .await
        .expect("Bounded baseline");
    let handle = StructuredTaskHandle::new(TaskId("doctor-bounded-java-evidence".into()));
    assert!(
        provider
            .update(cx, |owner, cx| owner.register_run_handle_with_limit(
                plan.clone(),
                version.clone(),
                handle.clone(),
                0,
                cx
            ))
            .is_err()
    );
    assert!(!provider.read_with(cx, |owner, _| owner.has_active_runs()));
    let run = provider
        .update(cx, |owner, cx| {
            owner.register_run_handle_with_limit(plan, version, handle.clone(), 256, cx)
        })
        .expect("Existing owner run");
    assert!(provider.read_with(cx, |owner, _| owner.has_active_runs()));
    let oversized = format!(
        "<testsuite><testcase classname=\"GreetingTest\" name=\"messageIsAvailable\"><failure message=\"{}\"/></testcase></testsuite>",
        "x".repeat(1024)
    );
    fs.insert_file(
        Path::new("/fixture/target/surefire-reports/TEST-GreetingTest.xml"),
        oversized.into_bytes(),
    )
    .await;
    cx.update(|cx| {
        assert!(handle.mark_running(None, cx));
        assert!(handle.mark_completed(successful_exit_status(), cx));
    });
    cx.run_until_parked();
    let result = project
        .read_with(cx, |owner, cx| {
            owner
                .structured_execution_store()
                .read(cx)
                .state()
                .provider(&StructuredProviderId(JVM_JUNIT_PROVIDER_ID.into()))
                .and_then(|provider| provider.current_run.clone())
        })
        .expect("Observed owner result");
    assert_eq!(result.id, run);
    assert_eq!(result.summary.passed, 0);
    assert_eq!(result.summary.failed, 0);
    assert!(
        result
            .diagnostic
            .as_deref()
            .is_some_and(|detail| detail.contains("without a fresh readable JUnit result"))
    );
    assert!(!provider.read_with(cx, |owner, _| owner.has_active_runs()));
}
